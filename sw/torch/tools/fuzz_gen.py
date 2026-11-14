"""Random small-model generator of the full-mode fuzzer (fuzz_full.py).

gen(seed) -> (source, meta): Python source that defines class M(nn.Module) (parameters drawn from a
generator seeded by the seed) and def inputs(T, seed=0) returning the forward arguments
(x, pos, mb, ma, sc): x [1, T, n0] float32, pos [1, T] int64, mb [1, 1, T, T] bool tril, ma the additive
0 / -inf form of mb, sc a Python float. The model is a chain of 3 to 11 steps drawn from the forms that
full mode lowers and the near misses it must refuse: linear layers (nn.Linear, F.linear, x @ W.T, addmm,
bias forms), residuals, in-place ops, RMS norm variants (eps as a buffer or a forward float, F.rms_norm,
per head), activations and GLU, scalars (Python, 0-dim buffers), rows and broadcasts, views and
reshapes, RoPE and near misses, SDPA with GQA and masks, CPU-only ops, dtype islands, a block called
twice, buffer mutation and Python float forward arguments.

gen_elem(seed) is the same generator without a matmul (a linear layer becomes a tanh, a slice or a
tile): full mode then has no 8-bit weights or activations, so it must match eager to float32 accuracy.
"""
from __future__ import annotations

import math
import random

HID = [64, 128, 192, 256]


class Val:
    def __init__(self, name, kind, n, H=None, D=None, mut=True):
        self.name, self.kind, self.n, self.H, self.D, self.mut = name, kind, n, H, D, mut

    def shape_key(self):
        return (self.kind, self.n, self.H, self.D)


class G:
    def __init__(self, seed):
        self.r = random.Random(seed)
        self.seed = seed
        self.init: list[str] = []
        self.methods: list[str] = []
        self.fwd: list[str] = []
        self.vals: list[Val] = []
        self.k = 0
        self.tags: set = set()
        self.forced: list[str] = []             # values that also leave the model (drawn by the features below)
        self.weights: list[tuple] = []          # (expr, N, K)
        self.n0 = self.r.choice(HID)
        self.vals.append(Val("x", "tm", self.n0, mut=False))
        self.ind = "        "

    # ------------------------------------------------------------------ helpers
    def new(self, p="v"):
        self.k += 1
        return f"{p}{self.k}"

    def emit(self, line):
        self.fwd.append(self.ind + line)

    def tag(self, t):
        self.tags.add(t)

    def side(self, name):
        """A random stream of its own for a feature that was added to the generator later: drawing from it leaves the
        main stream, and so every model that does not draw the feature, as it was."""
        return random.Random(f"{self.seed}:{name}:{self.k}")

    def param(self, shape, scale=None, buffer=False, init=None, prefix="p"):
        name = self.new(prefix)
        if init is None:
            sc = scale if scale is not None else 1.0 / math.sqrt(shape[-1])
            init = f"torch.randn({list(shape)}, generator=g) * {sc!r}"
        if buffer:
            self.init.append(f"self.register_buffer('{name}', {init})")
        else:
            self.init.append(f"self.{name} = nn.Parameter({init})")
        return f"self.{name}"

    def pick(self, pred=lambda v: True, recent=0.7):
        c = [v for v in self.vals if pred(v)]
        if not c:
            return None
        if self.r.random() < recent:
            return c[-1] if self.r.random() < 0.6 else self.r.choice(c[-3:])
        return self.r.choice(c)

    def add(self, name, kind, n, H=None, D=None, mut=True):
        v = Val(name, kind, n, H, D, mut)
        self.vals.append(v)
        return v

    def tm(self, recent=0.7):
        return self.pick(lambda v: v.kind == "tm", recent)

    # ------------------------------------------------------------------ linear
    def weight(self, N, K):
        """An [N, K] weight expression: new, or shared with an earlier layer."""
        same = [w for w in self.weights if w[1] == N and w[2] == K]
        if same and self.r.random() < 0.2:
            self.tag("shared_weight")
            return self.r.choice(same)[0]
        w = self.param([N, K])
        self.weights.append((w, N, K))
        return w

    def linear_expr(self, v, N=None, allow_bias=True, exact=False):
        """Code of a linear layer applied to value v (tm or flat): (expr, N)."""
        K = v.n
        N = N or self.r.choice(HID)
        r = self.r.random()
        x = v.name
        bias = allow_bias and self.r.random() < 0.35
        if r < 0.3:
            name = self.new("l")
            self.init.append(f"self.{name} = nn.Linear({K}, {N}, bias={bias})")
            self.init.append(f"with torch.no_grad(): self.{name}.weight.copy_(torch.randn({N}, {K}, generator=g) * {1 / math.sqrt(K)!r})")
            if bias:
                self.init.append(f"with torch.no_grad(): self.{name}.bias.copy_(torch.randn({N}, generator=g) * 0.1)")
            self.weights.append((f"self.{name}.weight", N, K))
            self.tag("nn.Linear")
            return f"self.{name}({x})", N
        w = self.weight(N, K)
        b = self.param([N], 0.1) if bias else None
        if r < 0.5:
            self.tag("F.linear")
            return (f"F.linear({x}, {w}, {b})" if b else f"F.linear({x}, {w})"), N
        if r < 0.75:
            form = self.r.choice(["{x} @ {w}.T", "{x} @ {w}.t()", "torch.matmul({x}, {w}.transpose(0, 1))",
                                  "{x} @ {w}.mT", "{x} @ {w}.permute(1, 0)", "torch.matmul({x}, {w}.transpose(-1, -2))"])
            e = form.format(x=x, w=w)
            self.tag("x@W.T")
            if b:
                e = self.r.choice([f"({e}) + {b}", f"{b} + ({e})"])
            return e, N
        if r < 0.85:
            self.tag("addmm")
            bb = b or self.param([N], 0.1)
            kw = ""
            if self.r.random() < 0.15:
                kw = self.r.choice([", beta=0.5", ", alpha=2.0", ", beta=1.0, alpha=1.0"])
                self.tag("addmm_alpha_beta")
            x2 = x if v.kind == "flat" else f"{x}.reshape(-1, {K})"
            e = f"torch.addmm({bb}, {x2}, {w}.t(){kw})"
            if v.kind == "tm":
                e = f"({e}).view(1, T, {N})"
            return e, N
        # near misses / other weight uses
        r2 = self.r.random()
        if exact and 0.35 <= r2 < 0.6:
            r2 = 0.1
        if r2 < 0.35:
            # a weight [K, N] used without a transpose
            wsq = self.param([K, N], 1.0 / math.sqrt(K))
            self.tag("x@W_noT")
            return f"{x} @ {wsq}", N
        if r2 < 0.6:
            # an existing [K2=?, K] ... weight used not transposed: needs [K, N2]
            cands = [wt for wt in self.weights if wt[1] == K]
            if cands:
                wexp, N2, K2 = self.r.choice(cands)
                self.tag("tied_noT")
                return f"{x} @ {wexp}", K2
            wsq = self.param([K, N])
            return f"{x} @ {wsq}", N
        if r2 < 0.8:
            self.tag("W@x.T")
            w2 = self.weight(N, K)
            if v.kind == "flat":
                return f"({w2} @ {x}.t()).t()", N
            return f"({w2} @ {x}.transpose(-1, -2)).transpose(-1, -2)", N
        # bias computed on the CPU
        self.tag("bias_expr")
        w2 = self.weight(N, K)
        b2 = self.param([N], 0.1)
        return f"F.linear({x}, {w2}, {b2} * 2.0)", N

    def op_linear(self):
        v = self.pick(lambda v: v.kind in ("tm", "flat") and v.n % 32 == 0)
        if v is None:
            return
        e, N = self.linear_expr(v)
        y = self.new()
        self.emit(f"{y} = {e}")
        self.add(y, v.kind, N)

    # ------------------------------------------------------------------ residual / binary
    def two_same(self):
        a = self.pick(lambda v: v.kind in ("tm", "flat", "tm4", "hm"))
        if a is None:
            return None, None
        same = [v for v in self.vals if v.shape_key() == a.shape_key()]
        if self.r.random() < 0.15:
            return a, a
        others = [v for v in same if v is not a]
        if not others:
            # make a second one through a linear (a residual that depends on the product)
            if a.kind in ("tm", "flat"):
                e, N = self.linear_expr(a, N=a.n, exact=True)
                y = self.new()
                self.emit(f"{y} = {e}")
                b = self.add(y, a.kind, N)
                self.tag("residual_of_product")
                return a, b
            return a, a
        return a, self.r.choice(others)

    def op_add(self):
        a, b = self.two_same()
        if a is None:
            return
        y = self.new()
        r = self.r.random()
        if r < 0.45:
            self.emit(f"{y} = {a.name} + {b.name}")
            self.tag("add")
        elif r < 0.6:
            self.emit(f"{y} = {b.name} + {a.name}")
            self.tag("add")
        elif r < 0.68:
            self.emit(f"{y} = torch.add({a.name}, {b.name}, alpha={self.r.choice([0.5, 2.0, -1.0])})")
            self.tag("add_alpha")
        elif r < 0.78:
            self.emit(f"{y} = {a.name} - {b.name}")
            self.tag("sub")
        elif r < 0.9 and a.mut:
            self.emit(f"{y} = {a.name}.add_({b.name})")
            self.tag("add_")
        else:
            self.emit(f"{y} = {a.name} + {a.name}")
            self.tag("add_twice")
        self.add(y, a.kind, a.n, a.H, a.D)

    def op_mul(self):
        a, b = self.two_same()
        if a is None:
            return
        y = self.new()
        r = self.r.random()
        if r < 0.5:
            self.emit(f"{y} = {a.name} * {b.name}")
            self.tag("mul")
        elif r < 0.65 and a.mut:
            self.emit(f"{y} = {a.name}.mul_({b.name})")
            self.tag("mul_")
        elif r < 0.8:
            self.emit(f"{y} = {a.name} / (1.0 + {b.name} * {b.name})")
            self.tag("div_tensor")
        else:
            self.emit(f"{y} = torch.mul({b.name}, {a.name})")
            self.tag("mul")
        self.add(y, a.kind, a.n, a.H, a.D)

    # ------------------------------------------------------------------ scalars and rows
    def op_scalar(self):
        a = self.pick(lambda v: v.kind in ("tm", "flat", "tm4", "hm"))
        if a is None:
            return
        x = a.name
        c = self.r.choice([0.5, 0.3, 2.0, 3.0, 0.125, -1.5, 1e-3, 7.0])
        ci = self.r.choice([2, 3, -1])
        r = self.r.random()
        forms = [
            (f"{x} * {c!r}", "mul_scalar"), (f"{c!r} * {x}", "mul_scalar_left"), (f"{x} / {c!r}", "div_scalar"),
            (f"{x} + {c!r}", "add_scalar"), (f"{c!r} + {x}", "add_scalar_left"), (f"{x} - {c!r}", "sub_scalar"),
            (f"{c!r} - {x}", "rsub_scalar"), (f"{x} * {ci}", "mul_int"), (f"{x} / {ci}", "div_int"),
            (f"-{x}", "neg"), (f"torch.add({x}, {c!r}, alpha=2.0)", "add_scalar_alpha"),
            (f"torch.sub({x}, {c!r}, alpha=0.5)", "sub_scalar_alpha"), (f"{x}.mul({c!r})", "mul_scalar"),
            (f"{x} * torch.tensor({c!r})", "lift_fresh"),
        ]
        if r < 0.55:
            e, t = self.r.choice(forms)
            if self.r.random() < 0.3:
                e, t = self.r.choice([(f"{x} * sc", "mul_arg"), (f"sc * {x}", "mul_arg_left"), (f"{x} / sc", "div_arg"),
                                      (f"{x} + sc", "add_arg"), (f"{x} - sc", "sub_arg"), (f"sc - {x}", "rsub_arg"),
                                      (f"{x} * (sc * 2.0)", "mul_arg_expr")])
        else:
            # a 0-dim buffer (or parameter), possibly through a dtype conversion
            dt = self.r.choice(["", "", "", "dtype=torch.float64", "dtype=torch.bfloat16", "dtype=torch.float16"])
            val = self.r.choice([0.5, 0.3, 1.7, 0.1, -2.0, 1e-3, 3.3])
            init = f"torch.tensor({val!r}{', ' + dt if dt else ''})"
            s = self.param([], buffer=self.r.random() < 0.8, init=init)
            conv = self.r.choice(["", "", "", ".float()", ".half().float()", ".bfloat16().float()", ".double()",
                                  ".to(torch.bfloat16)", ".half()"])
            if dt and "float16" in dt or "bfloat16" in dt:
                pass
            se = s + conv
            sf = [(f"{x} * {se}", "mul_0dim"), (f"{se} * {x}", "mul_0dim_left"), (f"{x} / {se}", "div_0dim"),
                  (f"{x} + {se}", "add_0dim"), (f"{se} + {x}", "add_0dim_left"), (f"{x} - {se}", "sub_0dim"),
                  (f"{se} - {x}", "rsub_0dim"), (f"{se} / (1.0 + {x} * {x})", "rdiv_0dim")]
            e, t = self.r.choice(sf)
            if t == "div_0dim" and abs(val) < 0.01:
                # dividing by 1e-3 scales the signal by 1000: the attention scores saturate the softmax, and the 8-bit
                # quantization of the device flips the maximum (a numeric difference, not a defect of the lowering)
                self.init[-1] = self.init[-1].replace(f"torch.tensor({val!r}", "torch.tensor(1.7")
            if conv:
                t += "_conv"
            if dt:
                t += "_" + dt.split(".")[-1]
        y = self.new()
        self.emit(f"{y} = {e}")
        self.tag(t)
        self.add(y, a.kind, a.n, a.H, a.D)

    def op_row(self):
        a = self.pick(lambda v: v.kind in ("tm", "flat", "tm4"))
        if a is None:
            return
        x = a.name
        n = a.n
        r = self.r.random()
        y = self.new()
        if a.kind == "tm4" and r < 0.5:
            # a [D] row broadcast over heads: must not be taken as a row of n
            p = self.param([a.D], 0.5)
            e = self.r.choice([f"{x} * {p}", f"{x} + {p}"])
            self.tag("row_over_heads")
            self.emit(f"{y} = {e}")
            self.add(y, a.kind, a.n, a.H, a.D)
            return
        if a.kind == "tm4":
            p = self.param([a.H, a.D], 0.5)
            self.emit(f"{y} = {x} * {p}")
            self.tag("row_hd")
            self.add(y, a.kind, a.n, a.H, a.D)
            return
        shape = self.r.choice([[n], [n], [1, n], [1, 1, n] if a.kind == "tm" else [1, n], [1]])
        p = self.param(shape, 0.5, buffer=self.r.random() < 0.3)
        op = self.r.choice(["+", "*", "*", "-", "/"])
        pv = p
        if self.r.random() < 0.2:
            pv = self.r.choice([f"{p}.view({list(shape) if len(shape) > 1 else [1, n] if shape == [n] else [1]})",
                                f"{p}.float()", f"{p}.clone()", f"{p}[None]" if a.kind == "tm" and len(shape) < 3 else p])
        if op == "/":
            e = f"{x} / (1.0 + {pv} * {pv})"
        elif self.r.random() < 0.5:
            e = f"{x} {op} {pv}"
        else:
            e = f"{pv} {op} {x}"
        self.emit(f"{y} = {e}")
        self.tag(f"row{op}{'1' if shape == [1] else ''}")
        self.add(y, a.kind, a.n, a.H, a.D)

    def op_bcast(self):
        """Broadcasts between values that must not be taken as rows."""
        a = self.pick(lambda v: v.kind == "tm")
        if a is None:
            return
        x = a.name
        y = self.new()
        e = self.r.choice([f"{x} * {x}.mean(-1, keepdim=True)", f"{x} * {x}[:, -1:, :]", f"{x} + {x}.sum(1, keepdim=True)",
                           f"{x} * {x}[:, :1, :]", f"{x} - {x}.amax(-1, keepdim=True)",
                           f"{x} * torch.sigmoid({x}[..., :1])"])
        self.emit(f"{y} = {e}")
        self.tag("bcast")
        self.add(y, "tm", a.n)

    # ------------------------------------------------------------------ norms
    def op_rms(self):
        a = self.pick(lambda v: v.kind in ("tm", "tm4", "flat"))
        if a is None:
            return
        x = a.name
        n = a.n if a.kind != "tm4" else a.D
        y = self.new()
        r = self.r.random
        eps_v = self.r.choice([1e-6, 1e-5, 1e-6, 0.1, 1.0])
        if r() < 0.25:
            eps = self.param([], buffer=True, init=f"torch.tensor({eps_v!r})")
            self.tag("eps_buffer")
        else:
            eps = repr(eps_v)
        sq = self.r.choice([f"{x}.pow(2)", f"({x} * {x})", f"({x} ** 2)", f"torch.square({x})"])
        dim = "-1"
        keep = True
        if r() < 0.1:
            dim = "1" if a.kind != "flat" else "0"
            self.tag("rms_other_dim")
        if r() < 0.1:
            keep = False
            self.tag("rms_nokeep")
        mean = f"{sq}.mean({dim}, keepdim=True)" if keep else f"{sq}.mean({dim})[..., None]" if dim == "-1" else f"{sq}.mean({dim}, keepdim=True)"
        if r() < 0.15:
            core = f"{x} / torch.sqrt({mean} + {eps})"
            self.tag("rms_div_sqrt")
        elif r() < 0.15:
            core = f"{x} * (1.0 / torch.sqrt({mean} + {eps}))"
            self.tag("rms_recip_sqrt")
        elif r() < 0.08:
            core = f"{x} * torch.rsqrt(torch.add({mean}, {eps}, alpha=2.0))"
            self.tag("rms_eps_alpha")
        else:
            core = f"{x} * torch.rsqrt({mean} + {eps})"
        wr = r()
        if wr < 0.25:
            e = core
            self.tag("rms_noweight")
        elif wr < 0.55:
            w = self.param([n], init=f"1.0 + 0.1 * torch.randn({n}, generator=g)")
            e = self.r.choice([f"{w} * ({core})", f"({core}) * {w}"])
            self.tag("rms_w")
        elif wr < 0.75:
            w = self.param([n], init=f"0.1 * torch.randn({n}, generator=g)")
            e = self.r.choice([f"(1.0 + {w}) * ({core})", f"({core}) * (1 + {w})", f"({core}) * ({w} + 1.0)"])
            self.tag("rms_1pw")
        elif wr < 0.85:
            w = self.param([n], init=f"1.0 + 0.1 * torch.randn({n}, generator=g)")
            e = f"F.rms_norm({x}, ({n},), {w}, {eps_v!r})"
            self.tag("F.rms_norm")
        else:
            w = self.param([n], init=f"1.0 + 0.1 * torch.randn({n}, generator=g)")
            e = f"{w} * ({core}).float()"
            self.tag("rms_hf_float")
        if a.kind == "tm4":
            self.tag("rms_per_head")
        self.emit(f"{y} = {e}")
        self.add(y, a.kind, a.n, a.H, a.D)

    def op_rms_headsplit(self):
        """A norm over a head-split view of a tm value, merged back (Qwen3 q_norm)."""
        a = self.pick(lambda v: v.kind == "tm" and v.n >= 64)
        if a is None:
            return
        D = self.r.choice([d for d in (32, 64, 128) if a.n % d == 0])
        H = a.n // D
        w = self.param([D], init=f"1.0 + 0.1 * torch.randn({D}, generator=g)")
        t = self.new()
        y = self.new()
        self.emit(f"{t} = {a.name}.view(1, T, {H}, {D})")
        self.emit(f"{t} = {w} * ({t} * torch.rsqrt({t}.pow(2).mean(-1, keepdim=True) + 1e-6))")
        self.emit(f"{y} = {t}.reshape(1, T, {a.n})")
        self.tag("rms_headsplit")
        self.add(y, "tm", a.n)

    # ------------------------------------------------------------------ activations
    def op_act(self):
        a = self.pick(lambda v: v.kind in ("tm", "flat", "tm4", "hm"))
        if a is None:
            return
        x = a.name
        f = self.r.choice(["F.silu({x})", "torch.sigmoid({x})", "torch.tanh({x})", "torch.exp(0.25 * {x})",
                           "torch.exp({x} * 0.1)", "F.gelu({x}, approximate='tanh')", "F.gelu({x})", "torch.relu({x})",
                           "{x}.sigmoid()", "{x}.tanh()", "torch.exp({x}).clamp(max=50.0)", "{x} * torch.sigmoid({x})"])
        y = self.new()
        self.emit(f"{y} = " + f.format(x=x))
        self.tag("act:" + f.split("(")[0].replace("{x}.", "."))
        self.add(y, a.kind, a.n, a.H, a.D)

    def op_glu(self):
        a = self.pick(lambda v: v.kind in ("tm", "flat"))
        if a is None:
            return
        r = self.r.random()
        if r < 0.6:
            # an MLP: gate and up products of one input, down
            Fh = self.r.choice(HID + [512])
            eg, _ = self.linear_expr(a, Fh, allow_bias=False, exact=True)
            eu, _ = self.linear_expr(a, Fh, allow_bias=False, exact=True)
            g, u = self.new(), self.new()
            self.emit(f"{g} = {eg}")
            self.emit(f"{u} = {eu}")
            gv = self.add(g, a.kind, Fh)
            uv = self.add(u, a.kind, Fh)
        else:
            gv, uv = self.two_same()
            if gv is None or gv.kind not in ("tm", "flat"):
                return
            if gv is uv:
                self.tag("glu_same")
        act = self.r.choice(["F.silu", "F.silu", "F.gelu_t", "F.gelu", "torch.sigmoid"])
        aexpr = (f"F.gelu({gv.name}, approximate='tanh')" if act == "F.gelu_t" else f"{act}({gv.name})")
        y = self.new()
        if self.r.random() < 0.15:
            t = self.new()
            self.emit(f"{t} = {aexpr}")
            self.emit(f"{y} = {t} * {uv.name}")
            self.add(t, gv.kind, gv.n)
            self.tag("glu_act_two_users")
        else:
            self.emit(f"{y} = " + self.r.choice([f"{aexpr} * {uv.name}", f"{uv.name} * {aexpr}"]))
        self.tag("glu:" + act)
        self.add(y, gv.kind, gv.n)
        if r < 0.6 and self.r.random() < 0.8 and gv.n % 32 == 0:
            e, N = self.linear_expr(self.vals[-1], self.r.choice(HID))
            z = self.new()
            self.emit(f"{z} = {e}")
            self.add(z, gv.kind, N)

    # ------------------------------------------------------------------ views
    def op_view(self):
        a = self.pick(lambda v: True)
        x = a.name
        y = self.new()
        r = self.r.random()
        if a.kind == "tm":
            Ds = [d for d in (32, 64, 128) if a.n % d == 0]
            D = self.r.choice(Ds)
            H = a.n // D
            c = self.r.choice(["tm4", "hm", "flat", "id", "scramble", "hm_noT", "permute_hm", "tm_T", "slice"])
            if c == "tm4":
                self.emit(f"{y} = " + self.r.choice([f"{x}.view(1, T, {H}, {D})", f"{x}.reshape(1, T, {H}, {D})",
                                                      f"{x}.unflatten(-1, ({H}, {D}))"]))
                self.add(y, "tm4", a.n, H, D, a.mut)
            elif c == "hm":
                self.emit(f"{y} = {x}.view(1, T, {H}, {D}).transpose(1, 2)")
                self.add(y, "hm", a.n, H, D, a.mut)
            elif c == "permute_hm":
                self.emit(f"{y} = {x}.view(1, T, {H}, {D}).permute(0, 2, 1, 3)")
                self.add(y, "hm", a.n, H, D, a.mut)
                self.tag("view_permute")
            elif c == "flat":
                self.emit(f"{y} = " + self.r.choice([f"{x}.view(T, {a.n})", f"{x}.squeeze(0)", f"{x}.reshape(-1, {a.n})",
                                                      f"{x}[0]", f"{x}.flatten(0, 1)"]))
                self.add(y, "flat", a.n, mut=a.mut)
            elif c == "id":
                self.emit(f"{y} = " + self.r.choice([f"{x}.contiguous()", f"{x}.clone()", f"{x}[:, :, :]", f"{x}.view(1, T, {a.n})",
                                                      f"{x}.unsqueeze(0).squeeze(0)", f"{x}.detach()", f"{x}.to(torch.float32)",
                                                      f"{x}[:, :T]", f"{x}.transpose(1, 2).transpose(1, 2)"]))
                self.add(y, "tm", a.n, mut=a.mut)
            elif c == "scramble":
                # a reshape that does not keep the memory order: [1, H, T, D] read as [1, T, n]
                self.emit(f"{y} = {x}.view(1, T, {H}, {D}).transpose(1, 2).reshape(1, T, {a.n})")
                self.add(y, "tm", a.n, mut=False)
                self.tag("view_scramble")
            elif c == "hm_noT":
                # token-major memory read as head-major without a transpose
                self.emit(f"{y} = {x}.view(1, {H}, T, {D})")
                self.add(y, "hm", a.n, H, D, a.mut)
                self.tag("view_hm_noT")
            elif c == "tm_T":
                self.emit(f"{y} = {x}.transpose(1, 2).contiguous().transpose(1, 2)")
                self.add(y, "tm", a.n, mut=False)
                self.tag("view_tT")
            else:
                h = a.n // 2
                if h % 32 == 0:
                    e = self.r.choice([f"{x}[..., :{h}]", f"{x}[..., {h}:]", f"{x}.chunk(2, -1)[{self.r.randint(0, 1)}]",
                                       f"{x}.split({h}, -1)[1]", f"{x}.narrow(-1, 0, {h})"])
                    self.emit(f"{y} = {e}")
                    self.add(y, "tm", h, mut=False)
                    self.tag("view_slice")
                else:
                    self.emit(f"{y} = {x}[..., :32]")
                    self.add(y, "tm", 32, mut=False)
                    self.tag("view_slice")
        elif a.kind == "tm4":
            c = self.r.choice(["tm", "hm", "flat"])
            if c == "tm":
                self.emit(f"{y} = " + self.r.choice([f"{x}.reshape(1, T, {a.n})", f"{x}.flatten(2)", f"{x}.view(1, T, -1)"]))
                self.add(y, "tm", a.n, mut=a.mut)
            elif c == "hm":
                self.emit(f"{y} = " + self.r.choice([f"{x}.transpose(1, 2)", f"{x}.transpose(2, 1)", f"{x}.permute(0, 2, 1, 3)",
                                                      f"{x}.transpose(-3, -2)"]))
                self.add(y, "hm", a.n, a.H, a.D, a.mut)
            else:
                self.emit(f"{y} = {x}.reshape(T, {a.n})")
                self.add(y, "flat", a.n, mut=a.mut)
        elif a.kind == "hm":
            c = self.r.choice(["tm", "tm", "tm4", "contig", "scramble"])
            if c == "tm":
                self.emit(f"{y} = " + self.r.choice([f"{x}.transpose(1, 2).reshape(1, T, {a.n})",
                                                      f"{x}.transpose(1, 2).contiguous().view(1, T, {a.n})",
                                                      f"{x}.permute(0, 2, 1, 3).reshape(1, T, {a.n})",
                                                      f"{x}.transpose(1, 2).flatten(2)", f"{x}.transpose(1, 2).flatten(-2)"]))
                self.add(y, "tm", a.n, mut=False)
            elif c == "tm4":
                self.emit(f"{y} = {x}.transpose(1, 2)")
                self.add(y, "tm4", a.n, a.H, a.D, a.mut)
            elif c == "contig":
                self.emit(f"{y} = " + self.r.choice([f"{x}.contiguous()", f"{x}.clone()"]))
                self.add(y, "hm", a.n, a.H, a.D)
            else:
                self.emit(f"{y} = {x}.reshape(1, T, {a.n})")
                self.add(y, "tm", a.n, mut=False)
                self.tag("view_scramble_hm")
        else:  # flat
            c = self.r.choice(["tm", "tm", "t"])
            if c == "tm":
                self.emit(f"{y} = " + self.r.choice([f"{x}.unsqueeze(0)", f"{x}.view(1, T, {a.n})", f"{x}[None]"]))
                self.add(y, "tm", a.n, mut=a.mut)
            else:
                self.emit(f"{y} = {x}.t().contiguous().t()")
                self.add(y, "flat", a.n, mut=False)
                self.tag("view_tT")

    def op_cat(self):
        a = self.pick(lambda v: v.kind == "tm")
        b = self.pick(lambda v: v.kind == "tm", recent=0.3)
        if a is None or b is None or a.n + b.n > 512:
            return
        y = self.new()
        self.emit(f"{y} = torch.cat([{a.name}, {b.name}], -1)")
        self.add(y, "tm", a.n + b.n, mut=False)
        self.tag("cat")

    # ------------------------------------------------------------------ rope
    def rope_code(self, x, H, D, variant=None):
        """Lines that rotate the head-major value x [1, H, T, D]; returns the result name."""
        variant = variant or self.r.choice(["ok"] * 6 + ["pos_off", "sin_pos", "sin_inv", "partial", "cached", "factor",
                                                         "factor_cos", "swapped", "order", "sub", "offset_int",
                                                         "factor_buf", "inv_param", "chunk"])
        self.tag("rope:" + variant)
        inv = self.param([D // 2], buffer=True, init=f"1.0 / (10000 ** (torch.arange(0, {D}, 2).float() / {D}))")
        y = self.new("r")
        e = lambda s: self.emit(s)  # noqa: E731
        if variant == "inv_param":
            inv = self.param([D // 2], init=f"1.0 / (10000 ** (torch.arange(0, {D}, 2).float() / {D}))")
        p = "pos"
        if variant == "offset_int":
            p = "(pos + 7)"
        e(f"{y}_f = {p}[:, :, None].float() * {inv}")
        if variant == "pos_off":
            e(f"{y}_f = ({p}[:, :, None].float() + 3.0) * {inv}")
        e(f"{y}_emb = torch.cat([{y}_f, {y}_f], -1)")
        e(f"{y}_cos = {y}_emb.cos()")
        e(f"{y}_sin = {y}_emb.sin()")
        if variant == "sin_pos":
            e(f"{y}_sin = torch.cat([({p} + 1)[:, :, None].float() * {inv}] * 2, -1).sin()")
        if variant == "sin_inv":
            inv2 = self.param([D // 2], buffer=True, init=f"1.0 / (5000 ** (torch.arange(0, {D}, 2).float() / {D}))")
            e(f"{y}_sin = torch.cat([{p}[:, :, None].float() * {inv2}] * 2, -1).sin()")
        if variant == "factor":
            e(f"{y}_cos = {y}_cos * 1.25")
            e(f"{y}_sin = {y}_sin * 1.25")
        if variant == "factor_cos":
            e(f"{y}_cos = {y}_cos * 1.25")
        if variant == "factor_buf":
            s = self.param([], buffer=True, init="torch.tensor(0.75)")
            e(f"{y}_cos = {y}_cos * {s}")
            e(f"{y}_sin = {y}_sin * {s}")
        if variant == "cached":
            cc = self.param([64, D], buffer=True, init=f"torch.cat([torch.arange(64)[:, None].float() * (1.0 / (10000 ** (torch.arange(0, {D}, 2).float() / {D})))] * 2, -1).cos()")
            ss = self.param([64, D], buffer=True, init=f"torch.cat([torch.arange(64)[:, None].float() * (1.0 / (10000 ** (torch.arange(0, {D}, 2).float() / {D})))] * 2, -1).sin()")
            e(f"{y}_cos = {cc}[pos.clamp(max=63)]")
            e(f"{y}_sin = {ss}[pos.clamp(max=63)]")
        e(f"{y}_cos = {y}_cos.unsqueeze(1)")
        e(f"{y}_sin = {y}_sin.unsqueeze(1)")
        h = D // 2
        if variant == "partial":
            rd = D // 2
            e(f"{y}_xr, {y}_xp = {x}[..., :{rd}], {x}[..., {rd}:]")
            e(f"{y}_c2, {y}_s2 = {y}_cos[..., :{rd // 2}].repeat(1, 1, 1, 2), {y}_sin[..., :{rd // 2}].repeat(1, 1, 1, 2)")
            e(f"{y}_rot = torch.cat([-{y}_xr[..., {rd // 2}:], {y}_xr[..., :{rd // 2}]], -1)")
            e(f"{y} = torch.cat([{y}_xr * {y}_c2 + {y}_rot * {y}_s2, {y}_xp], -1)")
            return y
        if variant == "chunk":
            e(f"{y}_x1, {y}_x2 = {x}.chunk(2, -1)")
        else:
            e(f"{y}_x1 = {x}[..., :{h}]")
            e(f"{y}_x2 = {x}[..., {h}:]")
        if variant == "swapped":
            e(f"{y}_rot = torch.cat([{y}_x2, -{y}_x1], -1)")
        else:
            e(f"{y}_rot = torch.cat([-{y}_x2, {y}_x1], -1)")
        if variant == "order":
            e(f"{y} = {y}_sin * {y}_rot + {y}_cos * {x}")
        elif variant == "sub":
            e(f"{y} = {x} * {y}_cos - {y}_rot * {y}_sin")
        else:
            e(f"{y} = {x} * {y}_cos + {y}_rot * {y}_sin")
        return y

    def op_rope(self):
        a = self.pick(lambda v: v.kind == "hm" and v.D % 2 == 0)
        if a is None:
            a0 = self.pick(lambda v: v.kind == "tm" and v.n % 64 == 0)
            if a0 is None:
                return
            D = self.r.choice([d for d in (64, 128) if a0.n % d == 0])
            H = a0.n // D
            t = self.new()
            self.emit(f"{t} = {a0.name}.view(1, T, {H}, {D}).transpose(1, 2)")
            a = self.add(t, "hm", a0.n, H, D)
        y = self.rope_code(a.name, a.H, a.D)
        self.add(y, "hm", a.n, a.H, a.D)

    # ------------------------------------------------------------------ attention
    def op_attn(self):
        r = self.r.random
        src = self.pick(lambda v: v.kind in ("tm",) and v.n % 32 == 0)
        if src is None:
            return
        D = self.r.choice([64, 64, 64, 128, 32])
        H = self.r.choice([1, 2, 4])
        Hkv = self.r.choice([h for h in (1, 2, 4) if H % h == 0])
        g = H // Hkv
        q, k, v = self.new("q"), self.new("k"), self.new("vv")
        eq, _ = self.linear_expr(src, H * D, exact=True)
        ksrc = src
        if r() < 0.15:
            o = self.pick(lambda v: v.kind == "tm" and v.n % 32 == 0, recent=0.2)
            if o is not None:
                ksrc = o
                self.tag("attn_kv_other_source")
        ek, _ = self.linear_expr(ksrc, Hkv * D, exact=True)
        vsrc = ksrc
        if r() < 0.15:
            o = self.pick(lambda v: v.kind == "tm" and v.n % 32 == 0, recent=0.2)
            if o is not None:
                vsrc = o
                self.tag("attn_v_other_source")
        ev, _ = self.linear_expr(vsrc, Hkv * D, exact=True)
        self.emit(f"{q} = ({eq}).view(1, T, {H}, {D}).transpose(1, 2)")
        self.emit(f"{k} = ({ek}).view(1, T, {Hkv}, {D}).transpose(1, 2)")
        self.emit(f"{v} = ({ev}).view(1, T, {Hkv}, {D}).transpose(1, 2)")
        if r() < 0.3:
            nq = self.rope_code(q, H, D, "ok" if r() < 0.8 else None)
            nk = self.rope_code(k, Hkv, D, "ok")
            q, k = nq, nk
        if r() < 0.1:
            # q, k normed per head (Qwen3)
            for nm in (q, k):
                w = self.param([D], init=f"1.0 + 0.1 * torch.randn({D}, generator=g)")
                self.emit(f"{nm} = {w} * ({nm} * torch.rsqrt({nm}.pow(2).mean(-1, keepdim=True) + 1e-6))")
            self.tag("attn_qk_norm")
        gqa_kw = ""
        if g > 1:
            form = self.r.choice(["repeat_interleave", "expand", "expand_clone", "tile", "repeat", "enable_gqa"])
            self.tag("gqa:" + form)
            for nm in (k, v):
                if form == "repeat_interleave":
                    self.emit(f"{nm} = {nm}.repeat_interleave({g}, dim=1)")
                elif form == "expand":
                    self.emit(f"{nm} = {nm}[:, :, None, :, :].expand(1, {Hkv}, {g}, T, {D}).reshape(1, {H}, T, {D})")
                elif form == "expand_clone":
                    self.emit(f"{nm} = {nm}.unsqueeze(2).expand(1, {Hkv}, {g}, T, {D}).contiguous().view(1, {H}, T, {D})")
                elif form == "tile":
                    self.emit(f"{nm} = {nm}.unsqueeze(1).expand(1, {g}, {Hkv}, T, {D}).reshape(1, {H}, T, {D})")
                elif form == "repeat":
                    self.emit(f"{nm} = {nm}.repeat(1, {g}, 1, 1)")
            if form == "enable_gqa":
                gqa_kw = ", enable_gqa=True"
        mk = self.r.choice(["causal"] * 4 + ["bool_in", "bool_tril", "bool_keep_upper", "add_in", "add_finfo", "add_1e9",
                                             "add_1e4", "add_alibi", "where_keep", "where_masked", "none", "bool_window",
                                             "causal_and_mask", "add_2d", "bool_heads", "bool_pad", "add_pad_inf", "scale_arg"])
        self.tag("mask:" + mk)
        kw = ""
        if mk == "causal":
            kw = ", is_causal=True"
        elif mk == "bool_in":
            kw = ", attn_mask=mb"
        elif mk == "bool_tril":
            kw = ", attn_mask=torch.ones(T, T, dtype=torch.bool).tril()"
        elif mk == "bool_keep_upper":
            kw = ", attn_mask=torch.ones(T, T, dtype=torch.bool).triu()"
        elif mk == "add_in":
            kw = ", attn_mask=ma"
        elif mk == "add_finfo":
            kw = ", attn_mask=torch.zeros(1, 1, T, T).masked_fill(~mb, torch.finfo(torch.float32).min)"
        elif mk == "add_1e9":
            kw = ", attn_mask=torch.zeros(T, T).masked_fill(~mb[0, 0], -1e9)"
        elif mk == "add_1e4":
            kw = ", attn_mask=torch.zeros(T, T).masked_fill(~mb[0, 0], -1e4)"
        elif mk == "add_alibi":
            kw = ", attn_mask=ma - 0.1 * (torch.arange(T)[:, None] - torch.arange(T)[None, :]).abs().float()"
        elif mk == "where_keep":
            kw = ", attn_mask=torch.where(mb, 0.0, float('-inf'))"
        elif mk == "where_masked":
            kw = ", attn_mask=torch.where(~mb, float('-inf'), 0.0)"
        elif mk == "bool_window":
            kw = ", attn_mask=mb & (torch.arange(T)[:, None] - torch.arange(T)[None, :] < 3)"
        elif mk == "causal_and_mask":
            kw = ", attn_mask=None, is_causal=True"
        elif mk == "add_2d":
            kw = ", attn_mask=ma[0, 0]"
        elif mk == "bool_pad":
            kw = ", attn_mask=mb & (torch.arange(T)[None, :] >= 2)"
        elif mk == "add_pad_inf":
            kw = ", attn_mask=ma.masked_fill(torch.arange(T)[None, :] < 2, float('-inf'))"
        elif mk == "scale_arg":
            kw = ", is_causal=True, scale=sc * 0.125"
        elif mk == "bool_heads":
            kw = f", attn_mask=mb.expand(1, {H}, T, T)"
        if r() < 0.25 and "scale=" not in kw:
            sc = self.r.choice([0.1, 0.05, 1.0, 0.125])
            kw += f", scale={sc!r}"
            self.tag("attn_scale")
        a = self.new("a")
        self.emit(f"{a} = F.scaled_dot_product_attention({q}, {k}, {v}{kw}{gqa_kw})")
        o = self.new()
        merge = self.r.choice(["t_reshape", "t_reshape", "t_contig_view", "permute", "flatten", "noT", "keep"])
        self.tag("merge:" + merge)
        n = H * D
        if merge == "t_reshape":
            self.emit(f"{o} = {a}.transpose(1, 2).reshape(1, T, {n})")
        elif merge == "t_contig_view":
            self.emit(f"{o} = {a}.transpose(1, 2).contiguous().view(1, T, {n})")
        elif merge == "permute":
            self.emit(f"{o} = {a}.permute(0, 2, 1, 3).reshape(1, T, {n})")
        elif merge == "flatten":
            self.emit(f"{o} = {a}.transpose(1, 2).flatten(2)")
        elif merge == "noT":
            self.emit(f"{o} = {a}.reshape(1, T, {n})")
        else:
            self.add(a, "hm", n, H, D)
            return
        self.add(o, "tm", n)
        if r() < 0.7:
            e, N = self.linear_expr(self.vals[-1], self.r.choice(HID))
            z = self.new()
            self.emit(f"{z} = {e}")
            self.add(z, "tm", N)

    def op_self_attn(self):
        """sdpa of pool values directly (aliasing operands)."""
        a = self.pick(lambda v: v.kind == "hm" and v.D in (64, 128))
        if a is None:
            return
        b = self.pick(lambda v: v.kind == "hm" and v.D == a.D and v.H == a.H, recent=0.3)
        o = self.new("a")
        self.emit(f"{o} = F.scaled_dot_product_attention({a.name}, {b.name}, {a.name}, is_causal=True)")
        self.tag("attn_pool")
        self.add(o, "hm", a.n, a.H, a.D)

    # ------------------------------------------------------------------ CPU ops, dtypes
    def op_cpu(self):
        a = self.pick(lambda v: v.kind in ("tm", "flat"))
        if a is None:
            return
        x = a.name
        y = self.new()
        f = self.r.choice([f"F.layer_norm({x}, ({a.n},))", f"torch.softmax({x}, -1)", f"torch.cumsum({x}, -2) * 0.2",
                           f"{x}.flip(-1)", f"torch.roll({x}, 1, -1)", f"{x}.abs()", f"torch.clamp({x}, -1.0, 1.0)",
                           f"torch.erf({x})", f"{x} ** 3 * 0.1"])
        self.emit(f"{y} = {f}")
        self.tag("cpu:" + f.split("(")[0])
        self.add(y, a.kind, a.n)

    def op_dtype(self):
        a = self.pick(lambda v: v.kind in ("tm", "flat", "hm", "tm4"))
        if a is None:
            return
        x = a.name
        y = self.new()
        f = self.r.choice([f"{x}.to(torch.bfloat16).float()", f"({x}.half() * 2).float()", f"{x}.double().tanh().float()",
                           f"((({x} * 8).round().long()).float() / 8)", f"{x}.to(torch.bfloat16).to(torch.float32) * 1.0",
                           f"({x}.bfloat16() + 1).float() - 1"])
        self.emit(f"{y} = {f}")
        self.tag("dtype_island")
        self.add(y, a.kind, a.n, a.H, a.D, mut=True)

    def op_block_twice(self):
        """A method (norm, MLP, residual) called twice: same module twice."""
        a = self.pick(lambda v: v.kind == "tm")
        if a is None:
            return
        n = a.n
        m = self.new("blk")
        w = self.param([n], init=f"1.0 + 0.1 * torch.randn({n}, generator=g)")
        w1 = self.param([2 * n, n])
        w2 = self.param([n, 2 * n])
        body = [f"    def {m}(self, h):",
                f"        t = {w} * (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6))",
                f"        t = F.silu(F.linear(t, {w1}))",
                f"        return h + F.linear(t, {w2})"]
        self.methods.append("\n".join(body))
        y = self.new()
        self.emit(f"{y} = self.{m}(self.{m}({a.name}))")
        self.tag("block_twice")
        self.add(y, "tm", n)

    def op_bufmut(self):
        """A buffer read as a row and mutated in forward (before or after the read)."""
        a = self.pick(lambda v: v.kind == "tm")
        if a is None:
            return
        b = self.param([a.n], buffer=True, init=f"1.0 + 0.1 * torch.randn({a.n}, generator=g)")
        y = self.new()
        if self.r.random() < 0.5:
            self.emit(f"{y} = {a.name} * {b}")
            self.emit(f"{b}.mul_(0.75).add_(0.1)")
            self.tag("bufmut_after")
        else:
            self.emit(f"{b}.mul_(0.75).add_(0.1)")
            self.emit(f"{y} = {a.name} + {b}")
            self.tag("bufmut_before")
        self.add(y, "tm", a.n)
        self.bufmut_live(a)

    def bufmut_live(self, a):
        """A buffer, or the input x, that is read by ops of a region, changed in place after the last read, and
        returned: the outputs depend on the old value (through the ops that read it, which nothing else reads) and on the
        new one (through the returned copy), so a region call that is placed after the change reads the wrong value."""
        rb = self.side("bufmut_live")
        if rb.random() >= 0.3:
            return
        src, mutate = (a, None) if rb.random() < 0.5 else (self.vals[0], "x")
        b = self.param([src.n], buffer=True, init=f"1.0 + 0.1 * torch.randn({src.n}, generator=g)")
        y, z = self.new(), self.new()
        self.emit(f"{y} = {src.name} * {b}")
        self.emit(f"{z} = F.silu({y}) * 0.5 + {y}")
        if mutate is None:
            self.emit(f"{b}.mul_(0.75).add_(0.1)")
            self.forced += [z, f"{b}.clone()"]
        else:
            self.emit("x.mul_(0.5).add_(0.25)")
            self.forced += [z, "x.clone()"]
        self.tag("bufmut_live")

    def op_rms_arg(self):
        a = self.pick(lambda v: v.kind == "tm")
        if a is None:
            return
        y = self.new()
        self.emit(f"{y} = {a.name} * torch.rsqrt({a.name}.pow(2).mean(-1, keepdim=True) + sc * 1e-5)")
        self.tag("rms_eps_arg")
        self.add(y, "tm", a.n)

    def op_inplace_chain(self):
        a = self.pick(lambda v: v.kind == "tm" and v.mut)
        if a is None:
            return
        y = self.new()
        self.emit(f"{y} = {a.name}.clone()")
        self.emit(f"{y}.mul_({self.r.choice([0.5, 2.0])}).add_({a.name})")
        if self.r.random() < 0.5:
            self.emit(f"{y} = F.silu({y}, inplace=False)")
        self.tag("inplace_chain")
        self.add(y, "tm", a.n)

    # ------------------------------------------------------------------ the model
    OPS = [("op_linear", 18), ("op_add", 9), ("op_mul", 4), ("op_scalar", 8), ("op_row", 6), ("op_bcast", 2),
           ("op_rms", 9), ("op_rms_headsplit", 2), ("op_act", 7), ("op_glu", 6), ("op_view", 10), ("op_cat", 1),
           ("op_rope", 3), ("op_attn", 7), ("op_self_attn", 1), ("op_cpu", 3), ("op_dtype", 2),
           ("op_block_twice", 1), ("op_inplace_chain", 2), ("op_bufmut", 2), ("op_rms_arg", 1)]

    def build(self):
        nsteps = self.r.randint(3, 11)
        names, wts = zip(*self.OPS)
        # start with a product most of the time, so that something can be lowered
        if self.r.random() < 0.8:
            self.op_linear()
        for _ in range(nsteps):
            op = self.r.choices(names, wts)[0]
            getattr(self, op)()
        # end with a product or not
        last = self.vals[-1]
        if last.kind in ("tm", "flat") and self.r.random() < 0.6:
            e, N = self.linear_expr(last, self.r.choice(HID))
            z = self.new()
            self.emit(f"{z} = {e}")
            self.add(z, last.kind, N)
        outs = [self.vals[-1].name]
        if self.r.random() < 0.4 and len(self.vals) > 2:
            o = self.r.choice(self.vals[1:-1])
            if o.name not in outs:
                outs.append(o.name)
                self.tag("extra_output")
        if self.r.random() < 0.15 and len(self.vals) > 3:
            o = self.r.choice(self.vals[1:-1])
            # an output through a view chain
            if o.kind == "hm":
                e = f"{o.name}.transpose(1, 2).reshape(1, T, {o.n}).sum(-1)"
            elif o.kind == "tm":
                e = f"{o.name}.view(1, T, 2, {o.n // 2})"
            else:
                e = f"{o.name}.reshape(-1)"
            outs.append(e)
            self.tag("view_output")
        # a value and a view or a copy of it leave together (its own random stream: the other models stay as they were)
        rr = random.Random(self.seed * 31 + 5)
        if rr.random() < 0.15 and len(self.vals) > 3:
            o = rr.choice(self.vals[1:-1])
            form = rr.choice(["view", "clone", "detach", "view_of_clone"])
            if form == "view" or form == "view_of_clone":
                base = o.name if form == "view" else f"{o.name}.clone()"
                if o.kind == "hm":
                    e = f"{base}.transpose(1, 2)"
                elif o.kind == "tm":
                    e = f"{base}.view(1, T, 2, {o.n // 2})"
                else:
                    e = f"{base}.reshape(-1)"
            else:
                e = f"{o.name}.{form}()"
            if o.name not in outs:
                outs.append(o.name)
            outs.append(e)
            self.tag("share_output")
        self.attn_block()
        self.hm_copy_block()
        outs += [f for f in self.forced if f not in outs]
        return outs

    def attn_block(self):
        """A grouped-query attention of its own, on the input x, that leaves the model (on a stream of its
        own): the expanded keys or values are also returned or read by a CPU op (tag gqa:exp_out: the lowering
        bypasses the expansion, which something else reads), or the mask is an additive padding mask with a fully
        masked row of finite values (tags mask:add_pad_finfo, mask:add_pad_1e9: the device refuses such a call and
        the CPU runs it, with a warning)."""
        rs = self.side("attn_block")
        if rs.random() >= 0.12:
            return
        src = self.vals[0]                # the input x: its projections are lowered whatever the model does
        kind = rs.choice(["exp_out", "exp_out", "pad_finfo", "pad_1e9"])
        D, Hkv, g = rs.choice([64, 128]), rs.choice([1, 2]), rs.choice([2, 4])
        H, n = Hkv * g, Hkv * g * D
        form = rs.choice(["repeat_interleave", "expand", "expand_clone"])
        q, k, v, a, o, z = (self.new(p) for p in ("q", "k", "vv", "a", "o", "z"))
        self.emit(f"{q} = F.linear({src.name}, {self.param([n, src.n])}).view(1, T, {H}, {D}).transpose(1, 2)")
        for nm in (k, v):
            self.emit(f"{nm} = F.linear({src.name}, {self.param([Hkv * D, src.n])}).view(1, T, {Hkv}, {D}).transpose(1, 2)")
        for nm in (k, v):
            if form == "repeat_interleave":
                self.emit(f"{nm} = {nm}.repeat_interleave({g}, dim=1)")
            elif form == "expand":
                self.emit(f"{nm} = {nm}[:, :, None, :, :].expand(1, {Hkv}, {g}, T, {D}).reshape(1, {H}, T, {D})")
            else:
                self.emit(f"{nm} = {nm}.unsqueeze(2).expand(1, {Hkv}, {g}, T, {D}).contiguous().view(1, {H}, T, {D})")
        if kind == "pad_finfo":
            kw = ", attn_mask=torch.zeros(1, 1, T, T).masked_fill(~mb | (torch.arange(T)[None, :] < 2), torch.finfo(torch.float32).min)"
        elif kind == "pad_1e9":
            kw = ", attn_mask=torch.zeros(T, T).masked_fill(~mb[0, 0] | (torch.arange(T)[None, :] < 2), -1e9)"
        else:
            kw = ", is_causal=True"
        self.emit(f"{a} = F.scaled_dot_product_attention({q}, {k}, {v}{kw})")
        self.emit(f"{o} = {a}.transpose(1, 2).reshape(1, T, {n})")
        self.emit(f"{z} = F.linear({o}, {self.param([128, n])})")
        self.forced.append(z)
        if kind == "exp_out":
            for nm in rs.choice([(k,), (v,), (k, v)]):
                if rs.random() < 0.5:
                    self.forced.append(nm)
                else:
                    self.emit(f"{nm}_c = torch.softmax({nm}, -1).sum(-1)")
                    self.forced.append(f"{nm}_c")
            self.tag("gqa:exp_out")
        else:
            self.tag("mask:add_" + kind)

    def hm_copy_block(self):
        """A head-major projection of the input x that is made contiguous in the head-major order and returned
        (tag out:hm_copy): its strides are not those of the rows of the device value, so the output is a copy
        made by the rewrite. A later .view() of the output fails on another layout (on a stream of its own)."""
        rs = self.side("hm_copy")
        if rs.random() >= 0.06:
            return
        src = self.vals[0]
        D, H = rs.choice([32, 64]), rs.choice([2, 4])
        n = H * D
        hm, u, z = (self.new(p) for p in ("hm", "u", "z"))
        self.emit(f"{hm} = F.linear({src.name}, {self.param([n, src.n])}).view(1, T, {H}, {D}).transpose(1, 2).contiguous()")
        self.emit(f"{u} = {rs.choice(['torch.tanh', 'torch.sigmoid', 'F.silu'])}({hm})")
        self.emit(f"{z} = F.linear({u}.transpose(1, 2).reshape(1, T, {n}), {self.param([128, n])})")
        self.forced += [z, u]
        self.tag("out:hm_copy")

    def source(self):
        outs = self.build()
        lines = ["class M(nn.Module):", "    def __init__(self):", "        super().__init__()",
                 f"        g = torch.Generator().manual_seed({self.seed})"]
        lines += ["        " + l for l in self.init]
        lines += self.methods
        lines += ["    def forward(self, x, pos, mb, ma, sc):", "        T = x.shape[1]"]
        lines += self.fwd
        lines.append(f"        return ({', '.join(outs)},)")
        lines += ["", f"N0 = {self.n0}", f"POS_OFF = {self.r.choice([0, 0, 3, 100])}",
                  f"X_SCALE = {self.r.choice([1.0, 1.0, 0.5, 2.0])}"]
        return "\n".join(lines) + "\n"


HEADER = """import torch
import torch.nn as nn
import torch.nn.functional as F
"""

INPUTS = """
def inputs(T, seed=0):
    gg = torch.Generator().manual_seed(1000 + seed * 31 + T)
    x = torch.randn(1, T, N0, generator=gg) * X_SCALE
    pos = (torch.arange(T) + POS_OFF)[None]
    mb = torch.ones(1, 1, T, T, dtype=torch.bool).tril()
    ma = torch.zeros(1, 1, T, T).masked_fill(~mb, float("-inf"))
    return x, pos, mb, ma, [0.5, 0.75, 1.25, 0.6][seed % 4]
"""


def gen(seed):
    g = G(seed)
    src = HEADER + g.source() + INPUTS
    return src, {"tags": sorted(g.tags), "n0": g.n0}


class ElemGen(G):
    """The generator without matmuls (see the module docstring)."""

    OPS = [("op_add", 8), ("op_mul", 5), ("op_scalar", 10), ("op_row", 8), ("op_bcast", 2), ("op_rms", 10),
           ("op_rms_headsplit", 2), ("op_act", 8), ("op_glu", 4), ("op_view", 10), ("op_rope", 5), ("op_cpu", 2),
           ("op_dtype", 2), ("op_inplace_chain", 2), ("op_rms_arg", 2)]

    def op_linear(self):
        return None

    def attn_block(self):
        return None

    def hm_copy_block(self):
        return None

    def linear_expr(self, v, N=None, allow_bias=True, exact=False):
        if N is None or N == v.n:
            return f"(torch.tanh({v.name}) * 0.9)", v.n
        if N < v.n:
            return f"({v.name}[..., :{N}] * 1.0)", N
        k = -(-N // v.n)
        return f"torch.cat([{v.name}] * {k}, -1)[..., :{N}]", N

    def op_glu(self):
        gv, uv = self.two_same()
        if gv is None or gv.kind not in ("tm", "flat"):
            return
        act = self.r.choice(["F.silu({g})", "F.gelu({g}, approximate='tanh')"])
        y = self.new()
        self.emit(f"{y} = {act.format(g=gv.name)} * {uv.name}")
        self.tag("glu")
        self.add(y, gv.kind, gv.n)


def gen_elem(seed):
    g = ElemGen(seed)
    src = HEADER + g.source() + INPUTS
    return src, {"tags": sorted(g.tags), "n0": g.n0}


if __name__ == "__main__":
    import sys

    kind = gen_elem if "--elem" in sys.argv else gen
    src, meta = kind(int([a for a in sys.argv[1:] if not a.startswith("--")][0]))
    print(src)
    print(meta)
