"""Whole-graph lowering (sw/torch/tesseract_torch/lower.py): device free.

Real aten graphs are captured the way the backend does (Dynamo, then AOTAutograd with a capturing
compiler). The tests check which patterns match, which near misses do not, the regions that form, the
program of a region against the schema of DESIGN.md §3, and the flattening recipes of the inputs and
outputs. No device, no extension: lower.py is loaded by path.

    PYTHONPATH=sw/torch build/tools/torchvenv/bin/python -m pytest -q sw/torch/tests/test_lower.py
"""
from __future__ import annotations

import importlib.util
import math
import struct
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from functorch.compile import make_boxed_func
from torch._dynamo.backends.common import aot_autograd

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
_spec = importlib.util.spec_from_file_location("tsr_lower_under_test", _HERE.parent / "tesseract_torch" / "lower.py")
L = importlib.util.module_from_spec(_spec)
sys.modules["tsr_lower_under_test"] = L
_spec.loader.exec_module(L)

aten = torch.ops.aten
HF_QWEN = _ROOT / "models" / "hf" / "Qwen2.5-0.5B-Instruct"


# ------------------------------------------------------------------------------------------
# capture: the aten graph and the weights as the backend sees them
# ------------------------------------------------------------------------------------------
def _dynamo_weights(gm, example_inputs) -> dict:
    """As backend._dynamo_weights: Dynamo placeholder index -> name for parameters and buffers."""
    out = {}
    for i, n in enumerate(n for n in gm.graph.nodes if n.op == "placeholder"):
        ex = example_inputs[i] if i < len(example_inputs) else None
        ga = n.meta.get("grapharg")
        src = ""
        if ga is not None and getattr(ga, "source", None) is not None:
            try:
                src = ga.source.name()
            except Exception:  # noqa: BLE001
                src = ""
        if isinstance(ex, torch.nn.Parameter) or "._parameters[" in src or "._buffers[" in src:
            out[i] = src or n.name
    return out


def capture(model, calls, dynamic=None):
    """Compile ``model`` with a capturing AOTAutograd backend, call it with every (args, kwargs) of
    ``calls`` and return (aten graph of the last compile, {weight node: name})."""
    from torch._functorch._aot_autograd import descriptors as D

    res = {}

    def backend(gm, example_inputs):
        dyn = _dynamo_weights(gm, example_inputs)

        def comp(aot_gm, aot_inputs):
            w = {}
            for n in aot_gm.graph.nodes:
                if n.op == "placeholder":
                    d = n.meta.get("desc")
                    if isinstance(d, (D.ParamAOTInput, D.BufferAOTInput)):
                        w[n] = str(d.target)
                    elif isinstance(d, D.PlainAOTInput) and d.idx in dyn:
                        w[n] = dyn[d.idx]
                elif n.op == "get_attr":
                    w[n] = str(n.target)
            res["gm"], res["w"] = aot_gm, w
            return make_boxed_func(aot_gm.forward)

        return aot_autograd(fw_compiler=comp, inference_compiler=comp, keep_inference_input_mutations=True)(gm, example_inputs)

    torch._dynamo.reset()
    cm = torch.compile(model, backend=backend, dynamic=dynamic)
    with torch.no_grad():
        for call in calls:
            if len(call) == 2 and isinstance(call[1], dict):
                args, kwargs = call
            elif len(call) == 1 and isinstance(call[0], tuple):
                args, kwargs = call[0], {}
            else:
                args, kwargs = call, {}
            cm(*args, **kwargs)
    torch._dynamo.reset()
    return res["gm"], res["w"]


def lower(model, calls, dynamic=None, probe=None, opts=None):
    gm, w = capture(model, calls, dynamic)
    regions, why = L.form_regions(gm, w, probe, opts)
    return gm, w, regions, why


def ops_of(region):
    return [o["op"] for o in region.ops]


def one_region(model, calls, **kw):
    gm, w, regions, why = lower(model, calls, **kw)
    assert len(regions) == 1, [r.describe() for r in regions]
    return gm, w, regions[0], why


def node_of(gm, name):
    return next(n for n in gm.graph.nodes if n.name == name)


def by_target(gm, target):
    return [n for n in gm.graph.nodes if n.op == "call_function" and n.target is target]


# ------------------------------------------------------------------------------------------
# the schema of DESIGN.md §3
# ------------------------------------------------------------------------------------------
OP_KEYS = {
    "mul_mat": {"w", "x", "dst", "addend", "tag"},
    "add": {"a", "b", "dst", "tag"},
    "mul": {"a", "b", "dst", "tag"},
    "scale": {"x", "dst", "s", "b", "tag"},
    "rms_norm": {"x", "dst", "weight", "eps", "tag"},
    "glu": {"gate", "up", "dst", "variant", "tag"},
    "unary": {"x", "dst", "func", "tag"},
    "rope": {"x", "dst", "n_heads", "head_dim", "n_dims", "table", "pos", "factor", "tag"},
    "attn": {"q", "k", "v", "dst", "n_head", "n_head_kv", "d", "scale", "mask", "tag"},
    "cpy": {"x", "dst", "tag"},
}
VALUE_KEYS = {"in": {"kind", "n"}, "tmp": {"kind", "n", "out"}, "const": {"kind", "n", "rows", "source"},
              "pos": {"kind"}, "mask": {"kind"}}


def _is_f32(x) -> bool:
    return isinstance(x, float) and struct.unpack("<f", struct.pack("<f", x))[0] == x


def check_program(prog: dict, region=None):
    """Raise AssertionError unless ``prog`` is a well formed program of DESIGN.md §3."""
    assert set(prog) == {"values", "ops", "kv_size"}, set(prog)
    assert isinstance(prog["kv_size"], int) and prog["kv_size"] > 0
    vals, ops = prog["values"], prog["ops"]
    for i, v in enumerate(vals):
        assert v["kind"] in VALUE_KEYS, v
        assert set(v) == VALUE_KEYS[v["kind"]], v
        if "n" in v:
            assert isinstance(v["n"], int) and v["n"] >= 1
        if v["kind"] == "tmp":
            assert isinstance(v["out"], bool)
        if v["kind"] == "const":
            assert v["rows"] == 1 and isinstance(v["source"], int)
            if region is not None:
                assert 0 <= v["source"] < len(region.consts)
                assert region.consts[v["source"]].n == v["n"]
    written, ready = set(), {i for i, v in enumerate(vals) if v["kind"] != "tmp"}

    def val(i, *kinds):
        assert isinstance(i, int) and not isinstance(i, bool) and 0 <= i < len(vals), i
        assert vals[i]["kind"] in kinds, (i, vals[i], kinds)
        return vals[i]

    def src(i):
        v = val(i, "in", "tmp", "const")
        assert i in ready, f"value {i} read before it is written"
        return v

    for o in ops:
        assert o["op"] in OP_KEYS, o["op"]
        assert set(o) == OP_KEYS[o["op"]] | {"op"}, (o["op"], set(o) ^ (OP_KEYS[o["op"]] | {"op"}))
        assert isinstance(o["tag"], int)
        d = val(o["dst"], "tmp")
        assert o["dst"] not in written, "dst written twice"
        k = o["op"]
        if k == "mul_mat":
            x = src(o["x"])
            assert isinstance(o["w"], int) and o["w"] >= 0
            if region is not None:
                w = region.weights[o["w"]]
                assert x["n"] == w.K and d["n"] == w.N
            if o["addend"] is not None:
                a = src(o["addend"])
                assert a["n"] == d["n"]
                if a["kind"] == "const":
                    assert a["rows"] == 1
            assert o["x"] != o["dst"] and o["addend"] != o["dst"]
        elif k in ("add", "mul"):
            a, b = src(o["a"]), src(o["b"])
            assert a["n"] == b["n"] == d["n"]
            assert a["kind"] != "const"
            assert o["dst"] not in (o["a"], o["b"])
        elif k == "scale":
            x = src(o["x"])
            assert x["n"] == d["n"] and _is_f32(o["s"]) and _is_f32(o["b"])
        elif k == "rms_norm":
            x = src(o["x"])
            assert x["n"] == d["n"] and _is_f32(o["eps"])
            if o["weight"] is not None:
                w = src(o["weight"])
                assert w["kind"] == "const" and w["n"] == x["n"]
        elif k == "glu":
            g, u = src(o["gate"]), src(o["up"])
            assert g["n"] == u["n"] == d["n"] and o["variant"] in (0, 2)
        elif k == "unary":
            assert src(o["x"])["n"] == d["n"] and o["func"] in (0, 2, 3, 4, 5)
        elif k == "rope":
            x = src(o["x"])
            assert x["n"] == d["n"] == o["n_heads"] * o["head_dim"]
            assert o["n_dims"] == o["head_dim"] and _is_f32(o["factor"])
            t = val(o["table"], "const")
            assert t["n"] * 2 == o["n_dims"]
            val(o["pos"], "pos")
        elif k == "attn":
            q, kk, v = src(o["q"]), src(o["k"]), src(o["v"])
            assert q["n"] == d["n"] == o["n_head"] * o["d"]
            assert kk["n"] == v["n"] == o["n_head_kv"] * o["d"]
            assert o["n_head"] % o["n_head_kv"] == 0 and _is_f32(o["scale"])
            if o["mask"] is not None:
                val(o["mask"], "mask")
        written.add(o["dst"])
        ready.add(o["dst"])
    # every value is used or an output
    assert any(v["kind"] == "tmp" and v["out"] for v in vals) or not ops
    for i, v in enumerate(vals):
        if v["kind"] == "tmp":
            assert i in written


def check_region(r, gm=None):
    prog = r.to_program([1.0] * len(r.scalars)) if r.scalars else r.to_program()
    check_program(prog, r)
    # inputs are the values of kind in / pos / mask, in value order
    kinds = [(i, v["kind"]) for i, v in enumerate(prog["values"]) if v["kind"] in ("in", "pos", "mask")]
    assert [(inp.value, inp.kind) for inp in r.inputs] == kinds
    # outputs are exactly the tmp values flagged out
    flagged = {i for i, v in enumerate(prog["values"]) if v["kind"] == "tmp" and v["out"]}
    assert {o.value for o in r.outputs} == flagged
    for o in r.outputs:
        for d in o.recipe["dims"]:
            if isinstance(d, dict):
                assert r.inputs[d["input"]].kind == "in"
    return prog


# ------------------------------------------------------------------------------------------
# small modules
# ------------------------------------------------------------------------------------------
D64 = 64


class Lin(nn.Module):
    """x -> mm -> body -> mm, so that a pattern in the middle has device neighbours (a region needs two ops)."""

    def __init__(self, body, d=D64):
        super().__init__()
        self.a = nn.Linear(d, d, bias=False)
        self.b = nn.Linear(d, d, bias=False)
        self.body = body

    def forward(self, x):
        return self.b(self.body(self.a(x)))


def rms_ok(x, w, eps=1e-6):
    return w * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))


def rms_noweight(x, w, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def rms_gemma(x, w, eps=1e-6):
    return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)) * (1.0 + w)


def rms_wrong_order(x, w, eps=1e-6):                 # eps added after rsqrt
    return w * (x * (torch.rsqrt(x.pow(2).mean(-1, keepdim=True)) + eps))


def rms_wrong_pow(x, w, eps=1e-6):                    # x ** 3
    return w * (x * torch.rsqrt(x.pow(3).mean(-1, keepdim=True) + eps))


def rms_wrong_dim(x, w, eps=1e-6):                    # mean over the token axis
    return w * (x * torch.rsqrt(x.pow(2).mean(1, keepdim=True) + eps))


class Norm(nn.Module):
    def __init__(self, fn, d=D64):
        super().__init__()
        self.fn = fn
        self.weight = nn.Parameter(torch.randn(d))

    def forward(self, x):
        return self.fn(x, self.weight)


X5 = (torch.randn(1, 5, D64),)


def test_module_imports_without_the_extension():
    import subprocess

    code = ("import importlib.util, sys; "
            f"s = importlib.util.spec_from_file_location('lower', {str(_HERE.parent / 'tesseract_torch' / 'lower.py')!r}); "
            "m = importlib.util.module_from_spec(s); sys.modules['lower'] = m; s.loader.exec_module(m); "
            "bad = [k for k in sys.modules if k.startswith('tesseract_torch') or '_tesseract_rt' in k]; "
            "assert not bad, bad; print('ok')")
    r = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True,
                       env={"PATH": "", "TSR_TORCH_EXT_DIR": "/nonexistent"})
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr


# ------------------------------------------------------------------------------------------
# 1. layout rules (§5.2)
# ------------------------------------------------------------------------------------------
class Views(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(D64, D64)

    def forward(self, x):
        h = self.lin(x)                                   # [1, T, 64]
        a = h.view(1, 5, 2, 32).transpose(1, 2)           # head-major [1, 2, T, 32]
        b = a.transpose(1, 2).reshape(1, 5, 64)           # back to rows
        c = h.permute(0, 2, 1)                            # not a view of the rows
        d = h.double()                                    # not float32
        e = h.unsqueeze(0).squeeze(0).clone()
        return a, b, c, d, e


@pytest.fixture(scope="module")
def views_graph():
    m = Views().eval()
    gm, w = capture(m, [((torch.randn(1, 5, D64),),)])
    return gm, w, L.LowerCtx(gm, w, None, L.LowerOpts())


def test_view_rules(views_graph):
    gm, w, ctx = views_graph
    addmm = by_target(gm, aten.addmm.default)[0]
    outs = [n for n in gm.graph.nodes if n.op == "output"][0].args[0]
    a, b, c, d, e = outs
    va, _ = ctx.view_of(a)
    assert (va.form, va.base_hm, va.n, va.heads, va.hd) == ("hm", False, 64, 2, 32) and va.base is addmm
    vb, _ = ctx.view_of(b)
    assert (vb.form, vb.n, vb.heads) == ("tm", 64, None) and vb.base is addmm
    # a permuted tensor is its own base: it is not the rows of the matmul result
    vc, _ = ctx.view_of(c)
    assert vc.base is c and vc.n == 5
    # float64 has no device value, with the reason
    vd, why = ctx.view_of(d)
    assert vd is None and "float64" in why
    ve, _ = ctx.view_of(e)
    assert ve.base is addmm and ve.form == "tm" and ve.n == 64
    # the chain lists the view nodes down to the base
    assert addmm not in va.chain and va.chain[0] is a


def test_view_hm_base_and_symbolic():
    """A tensor that is its own base can be read head-major when it is [1, H, T, D]."""
    gm = torch.fx.symbolic_trace(nn.Identity())
    g = torch.fx.Graph()
    p = g.placeholder("x")
    p.meta["val"] = torch.zeros(1, 4, 6, 32)
    g.output(p)
    gm = torch.fx.GraphModule(nn.Module(), g)
    ctx = L.LowerCtx(gm, {}, None, L.LowerOpts())
    v, _ = ctx.view_of(p, "hm")
    assert (v.form, v.base_hm, v.n, v.heads, v.hd) == ("hm", True, 128, 4, 32)
    v, _ = ctx.view_of(p)
    assert v.form == "tm" and v.n == 32            # read as rows of 32
    v, _ = ctx.view_of(p, "tm4")                    # [1, T, H, D] read as T = 4, H = 6, D = 32
    assert (v.form, v.n, v.heads, v.hd) == ("tm", 192, 6, 32)


# ------------------------------------------------------------------------------------------
# 2. patterns, positive and negative
# ------------------------------------------------------------------------------------------
def region_ops(model, x=None, **kw):
    gm, w, regions, why = lower(model.eval(), [((x if x is not None else X5[0],),)], **kw)
    return gm, w, regions, why


def test_rms_norm_forms():
    for fn, weight, transform in ((rms_ok, True, "identity"), (rms_noweight, False, None), (rms_gemma, True, "one_plus")):
        gm, w, regions, why = region_ops(Lin(Norm(fn)))
        assert len(regions) == 1
        r = regions[0]
        assert ops_of(r) == ["mul_mat", "rms_norm", "mul_mat"], fn.__name__
        op = r.ops[1]
        assert op["eps"] == L.f32(1e-6)
        assert (op["weight"] is not None) == weight
        if weight:
            assert r.consts[0].transform == transform and r.consts[0].n == D64
        # the six aten ops of the norm are replaced
        names = {n.target for n in r.nodes}
        for t in (aten.pow.Tensor_Scalar, aten.mean.dim, aten.rsqrt.default):
            assert t in names
        check_region(r)


@pytest.mark.parametrize("fn", [rms_wrong_order, rms_wrong_pow, rms_wrong_dim])
def test_rms_norm_near_misses(fn):
    gm, w, regions, why = region_ops(Lin(Norm(fn)))
    for r in regions:
        assert "rms_norm" not in ops_of(r)
    rsqrt = by_target(gm, aten.rsqrt.default)[0]
    assert rsqrt in why                                   # the norm's ops stay on the CPU, with a reason
    assert all(rsqrt not in set(r.nodes) for r in regions)


class Gated(nn.Module):
    def __init__(self, act, share=False, d=D64, ff=96):
        super().__init__()
        self.g = nn.Linear(d, ff, bias=False)
        self.u = nn.Linear(d, ff, bias=False)
        self.dn = nn.Linear(ff, d, bias=False)
        self.act, self.share = act, share

    def forward(self, x):
        a = self.act(self.g(x))
        y = self.dn(a * self.u(x))
        return (y, a) if self.share else y


def test_glu_variants():
    gm, w, regions, why = region_ops(Gated(F.silu))
    assert ops_of(regions[0]) == ["mul_mat", "mul_mat", "glu", "mul_mat"] and regions[0].ops[2]["variant"] == 0
    gm, w, regions, why = region_ops(Gated(lambda t: F.gelu(t, approximate="tanh")))
    assert ops_of(regions[0]) == ["mul_mat", "mul_mat", "glu", "mul_mat"] and regions[0].ops[2]["variant"] == 2
    check_region(regions[0])


def test_gelu_erf_is_not_lowered():
    gm, w, regions, why = region_ops(Gated(F.gelu))
    gelu = by_target(gm, aten.gelu.default)[0]
    assert "erf" in why[gelu]
    for r in regions:
        assert "glu" not in ops_of(r) and "unary" not in ops_of(r)
    # the product of the erf GELU and the up projection is a plain mul of two device values
    assert any("mul" in ops_of(r) for r in regions)


def test_silu_with_two_users_is_not_a_glu():
    gm, w, regions, why = region_ops(Gated(F.silu, share=True))
    ops = ops_of(regions[0])
    assert "glu" not in ops and "unary" in ops and "mul" in ops
    assert regions[0].ops[ops.index("unary")]["func"] == 2
    check_region(regions[0])


class Elementwise(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.a = nn.Linear(D64, D64, bias=False)
        self.b = nn.Linear(D64, D64, bias=False)
        self.row = nn.Parameter(torch.randn(D64))
        self.f = f

    def forward(self, x):
        return self.b(self.f(self, self.a(x)))


@pytest.mark.parametrize("name,f,op,check", [
    ("tanh", lambda s, t: torch.tanh(t), "unary", lambda o: o["func"] == 0),
    ("sigmoid", lambda s, t: torch.sigmoid(t), "unary", lambda o: o["func"] == 5),
    ("exp", lambda s, t: torch.exp(t), "unary", lambda o: o["func"] == 3),
    ("gelu_tanh", lambda s, t: F.gelu(t, approximate="tanh"), "unary", lambda o: o["func"] == 4),
    ("silu", lambda s, t: F.silu(t), "unary", lambda o: o["func"] == 2),
    ("add_row", lambda s, t: t + s.row, "add", lambda o: True),
    ("mul_row", lambda s, t: s.row * t, "mul", lambda o: True),
    ("mul_scalar", lambda s, t: t * 0.3, "scale", lambda o: o["s"] == L.f32(0.3) and o["b"] == 0.0),
    ("div_scalar", lambda s, t: t / 4.0, "scale", lambda o: o["s"] == 0.25),
    ("add_scalar", lambda s, t: t + 0.5, "scale", lambda o: o["s"] == 1.0 and o["b"] == 0.5),
    ("sub_scalar", lambda s, t: t - 0.5, "scale", lambda o: o["s"] == 1.0 and o["b"] == -0.5),
    ("add_dev", lambda s, t: t + F.silu(t), "add", lambda o: True),
])
def test_elementwise(name, f, op, check):
    gm, w, regions, why = region_ops(Elementwise(f))
    assert len(regions) == 1, name
    ops = ops_of(regions[0])
    assert op in ops, (name, ops)
    o = regions[0].ops[ops.index(op)]
    assert check(o), (name, o)
    check_region(regions[0])


def test_elementwise_near_misses():
    # a broadcast over rows (a [T, 1] column) has no kind
    gm, w, regions, why = region_ops(Elementwise(lambda s, t: t * t.mean(-1, keepdim=True)))
    for r in regions:
        assert "mul" not in ops_of(r)
    # a 0-dim value computed on the CPU from a parameter is not a scalar of the program
    gm, w, regions, why = region_ops(Elementwise(lambda s, t: t * s.row.sum()))
    for r in regions:
        assert "scale" not in ops_of(r)
    mul = [n for n in why if n.target is aten.mul.Tensor]
    assert mul


# -- mul_mat forms and the add fold
class Mats(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.a = nn.Linear(D64, D64, bias=True)
        self.b = nn.Linear(D64, D64, bias=False)
        self.w = nn.Parameter(torch.randn(D64, D64))
        self.bb = nn.Parameter(torch.randn(D64))

    def forward(self, x):
        h = self.a(x)
        m = self.mode
        if m == "fold":
            return self.b(h) + x
        if m == "fold_swapped":
            return x + self.b(h)
        if m == "no_fold_bias":                 # a matmul with a bias already has its addend
            return F.linear(h, self.w, self.bb) + x
        if m == "no_fold_two_users":
            y = self.b(h)
            return y + x, y.sum()
        if m == "no_fold_dependent":            # the other addend is computed from the matmul result
            y = self.b(h)
            return y + torch.tanh(y)
        if m == "matmul_w":
            return torch.matmul(h, self.w.t()) + x
        if m == "weight_left":
            return (self.w @ h[0].t()).t()
        if m == "act_x_act":
            return torch.matmul(h, h.transpose(1, 2))


@pytest.mark.parametrize("mode", ["fold", "fold_swapped"])
def test_mul_mat_add_fold(mode):
    gm, w, regions, why = region_ops(Mats(mode))
    r = regions[0] if regions else None
    assert r is not None
    # a(x) with bias, then b(h) with the residual as addend: no separate add
    assert ops_of(r) == ["mul_mat", "mul_mat"]
    assert r.ops[0]["addend"] is not None and r.ops[1]["addend"] is not None
    assert r.values[r.ops[0]["addend"]]["kind"] == "const"             # the bias
    assert r.values[r.ops[1]["addend"]]["kind"] == "in"                # the residual
    adds = by_target(gm, aten.add.Tensor)
    assert all(n in set(r.nodes) for n in adds)
    check_region(r)


@pytest.mark.parametrize("mode", ["no_fold_bias", "no_fold_two_users", "no_fold_dependent"])
def test_add_fold_near_misses(mode):
    gm, w, regions, why = region_ops(Mats(mode))
    ops = [o for r in regions for o in ops_of(r)]
    if mode == "no_fold_bias":
        r = regions[0]
        assert ops == ["mul_mat", "mul_mat", "add"]
        assert r.ops[1]["addend"] is not None                      # the bias; the residual is an add op
    else:
        assert "add" in ops
    for r in regions:
        check_region(r)
        for o in r.ops:
            if o["op"] == "mul_mat" and o["addend"] is not None and r.values[o["addend"]]["kind"] == "tmp":
                # a folded addend must not come from the matmul's own result
                assert o["addend"] != o["dst"]


def test_weight_forms():
    gm, w, regions, why = region_ops(Mats("matmul_w"))
    assert ops_of(regions[0]) == ["mul_mat", "mul_mat"]
    # a weight on the left of the product, and a product of two activations, stay on the CPU
    for mode in ("weight_left", "act_x_act"):
        gm, w, regions, why = region_ops(Mats(mode))
        mm = [n for n in why if n.target in (aten.mm.default, aten.bmm.default)]
        assert mm, mode
        assert all("weight" in why[n] or "matmul of two" in why[n] for n in mm), {n.name: why[n] for n in mm}


def test_mul_mat_linear_and_f32_only():
    class M(nn.Module):
        def __init__(self, dt):
            super().__init__()
            self.a = nn.Linear(D64, D64, bias=False).to(dt)
            self.b = nn.Linear(D64, D64, bias=False).to(dt)

        def forward(self, x):
            return self.b(torch.tanh(self.a(x)))

    gm, w, regions, why = region_ops(M(torch.float32))
    assert ops_of(regions[0]) == ["mul_mat", "unary", "mul_mat"]
    # D3: a bfloat16 graph forms no region
    gm, w, regions, why = region_ops(M(torch.bfloat16), x=torch.randn(1, 5, D64).bfloat16())
    assert regions == []
    assert any("float32" in v or "dtype" in v for v in why.values())


def test_k_not_multiple_of_32_stays_cpu():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(48, 48, bias=False)
            self.b = nn.Linear(48, 64, bias=False)

        def forward(self, x):
            return self.b(torch.tanh(self.a(x)))

    gm, w, regions, why = region_ops(M(), x=torch.randn(1, 5, 48))
    assert all("mul_mat" not in ops_of(r) for r in regions)
    assert any("multiple of 32" in v for v in why.values())


# -- RoPE
class RopeMod(nn.Module):
    def __init__(self, variant="ok", d=128, heads=2, dim=64, factor=1.0):
        super().__init__()
        self.q = nn.Linear(d, heads * dim, bias=False)
        self.h, self.dim, self.variant, self.factor = heads, dim, variant, factor
        self.register_buffer("inv_freq", 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim)), persistent=False)

    def forward(self, x, pos):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.h, self.dim).transpose(1, 2)
        freqs = pos[:, :, None].float() * self.inv_freq
        emb = torch.cat([freqs, freqs], -1)
        cos = (emb.cos() * self.factor).unsqueeze(1)
        sin = (emb.sin() * self.factor).unsqueeze(1)
        x1, x2 = q[..., : self.dim // 2], q[..., self.dim // 2:]
        if self.variant == "ok":
            rot = torch.cat([-x2, x1], -1)
        elif self.variant == "swapped":              # cat([x2, -x1])
            rot = torch.cat([x2, -x1], -1)
        elif self.variant == "swapped2":             # cat([-x1, x2])
            rot = torch.cat([-x1, x2], -1)
        elif self.variant == "no_neg":
            rot = torch.cat([x2, x1], -1)
        return q * cos + rot * sin


def rope_inputs(T=5, d=128):
    return (torch.randn(1, T, d), torch.arange(T)[None])


def test_rope_positive():
    gm, w, regions, why = lower(RopeMod().eval(), [(rope_inputs(),)])
    r = regions[0]
    assert ops_of(r) == ["mul_mat", "rope"]
    o = r.ops[1]
    assert (o["n_heads"], o["head_dim"], o["n_dims"], o["factor"]) == (2, 64, 64, 1.0)
    assert r.values[o["table"]] == {"kind": "const", "n": 32, "rows": 1, "source": 0}
    assert r.consts[0].node.name.startswith("arg") and "inv_freq" in w[r.consts[0].node]
    # positions come in as an input with its own recipe
    pos = [i for i in r.inputs if i.kind == "pos"][0]
    assert pos.recipe == {"reshape": [-1]} and pos.node.meta["val"].dtype == torch.int64
    # the head-major result is returned: reshape to [1, T, H, D], then transpose(1, 2)
    out = r.outputs[0]
    assert out.recipe == {"dims": [1, 5, 2, 64], "transpose12": True, "contiguous": False}
    # cos / sin and their chain are read, not consumed
    assert any(n.target is aten.cos.default for n in r.bypassed)
    check_region(r)


def test_rope_factor():
    gm, w, regions, why = lower(RopeMod(factor=1.25).eval(), [(rope_inputs(),)])
    assert regions[0].ops[1]["factor"] == 1.25


@pytest.mark.parametrize("variant", ["swapped", "swapped2", "no_neg"])
def test_rope_near_misses(variant):
    gm, w, regions, why = lower(RopeMod(variant).eval(), [(rope_inputs(),)])
    assert all("rope" not in ops_of(r) for r in regions)
    root = [n for n in gm.graph.nodes if n.target is aten.add.Tensor][-1]
    assert "rotate_half" in why[root] or root in why


def test_rope_positions_must_be_one_per_row():
    """One position broadcast over the tokens is not a rotary embedding of T rows."""
    gm, w, regions, why = lower(RopeMod().eval(), [((torch.randn(1, 5, 128), torch.tensor([[3]])),)])
    assert all("rope" not in ops_of(r) for r in regions)
    assert any("positions" in v for v in why.values())


def test_rope_dynamic_sequence():
    gm, w, regions, why = lower(RopeMod().eval(), [(rope_inputs(5),), (rope_inputs(8),)])
    r = regions[0]
    assert ops_of(r) == ["mul_mat", "rope"]
    dims = r.outputs[0].recipe["dims"]
    assert dims[0] == 1 and dims[2:] == [2, 64] and dims[1] == {"input": 0, "dim": 1}
    check_region(r)


# -- attention
class AttnMod(nn.Module):
    def __init__(self, heads=4, kv=2, d=64, mask="causal", dropout=0.0, batch=1, past=0, scale=None, gqa="expand"):
        super().__init__()
        self.H, self.Hkv, self.D, self.mask, self.dropout, self.B, self.gqa, self.scale = heads, kv, d, mask, dropout, batch, gqa, scale
        hid = heads * d
        self.q = nn.Linear(hid, heads * d, bias=False)
        self.k = nn.Linear(hid, kv * d, bias=False)
        self.v = nn.Linear(hid, kv * d, bias=False)
        self.o = nn.Linear(heads * d, hid, bias=False)
        self.past = past
        if past:
            self.register_buffer("pk", torch.randn(1, kv, past, d))
            self.register_buffer("pv", torch.randn(1, kv, past, d))

    def forward(self, x, mask=None):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.H, self.D).transpose(1, 2)
        k = self.k(x).view(B, T, self.Hkv, self.D).transpose(1, 2)
        v = self.v(x).view(B, T, self.Hkv, self.D).transpose(1, 2)
        if self.past:
            k = torch.cat([self.pk, k], 2)
            v = torch.cat([self.pv, v], 2)
        g = self.H // self.Hkv
        if g > 1:
            if self.gqa == "expand":
                k = k.unsqueeze(2).expand(B, self.Hkv, g, k.shape[2], self.D).reshape(B, self.H, k.shape[2], self.D)
                v = v.unsqueeze(2).expand(B, self.Hkv, g, v.shape[2], self.D).reshape(B, self.H, v.shape[2], self.D)
            else:
                k = k.repeat_interleave(g, dim=1)
                v = v.repeat_interleave(g, dim=1)
        kw = dict(dropout_p=self.dropout, scale=self.scale)
        if self.mask == "causal":
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True, **kw)
        elif self.mask == "bool":
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, **kw)
        elif self.mask == "where":
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=torch.where(mask, 0.0, float("-inf")), **kw)
        elif self.mask == "additive":
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, **kw)
        else:
            a = F.scaled_dot_product_attention(q, k, v, **kw)
        return self.o(a.transpose(1, 2).reshape(B, T, self.H * self.D))


def attn_inputs(mask, T=6, batch=1, hid=256):
    x = torch.randn(batch, T, hid)
    if mask in ("bool", "where"):
        return (x, torch.ones(1, 1, T, T, dtype=torch.bool).tril())
    if mask == "additive":
        return (x, torch.zeros(1, 1, T, T).masked_fill(~torch.ones(T, T, dtype=torch.bool).tril(), float("-inf")))
    return (x,)


@pytest.mark.parametrize("mask,form", [("causal", None), ("bool", "bool"), ("where", "bool"), ("additive", "additive")])
def test_attn_mask_forms(mask, form):
    gm, w, regions, why = lower(AttnMod(mask=mask).eval(), [(attn_inputs(mask),)])
    assert len(regions) == 1
    r = regions[0]
    assert ops_of(r) == ["mul_mat", "mul_mat", "mul_mat", "attn", "mul_mat"]
    a = r.ops[3]
    assert (a["n_head"], a["n_head_kv"], a["d"]) == (4, 2, 64) and a["scale"] == 0.125
    masks = [i for i in r.inputs if i.kind == "mask"]
    if form is None:
        assert a["mask"] is None and not masks
    else:
        assert len(masks) == 1 and masks[0].recipe == {"form": form, "reshape": "TxT"}
        assert r.values[a["mask"]] == {"kind": "mask"}
    check_region(r)


def test_attn_repeat_interleave_and_mha_and_scale():
    gm, w, regions, why = lower(AttnMod(gqa="repeat", scale=0.2).eval(), [(attn_inputs("causal"),)])
    a = [o for o in regions[0].ops if o["op"] == "attn"]
    assert a and (a[0]["n_head_kv"], a[0]["scale"]) == (2, L.f32(0.2))
    gm, w, regions, why = lower(AttnMod(heads=4, kv=4).eval(), [(attn_inputs("causal"),)])
    a = [o for o in regions[0].ops if o["op"] == "attn"]
    assert a and a[0]["n_head_kv"] == 4
    check_region(regions[0])


def attn_op_present(regions):
    return any("attn" in ops_of(r) for r in regions)


def test_attn_near_misses():
    # batch 2
    gm, w, regions, why = lower(AttnMod().eval(), [(attn_inputs("causal", batch=2),)])
    assert not attn_op_present(regions)
    sd = [n for n in gm.graph.nodes if n.op == "call_function" and "scaled_dot" in str(n.target)][0]
    assert "batch" in why[sd]
    # dropout
    gm, w, regions, why = lower(AttnMod(dropout=0.1).eval(), [(attn_inputs("causal"),)])
    assert not attn_op_present(regions)
    # AOTAutograd decomposes attention with dropout: no fused attention node is left to lower
    assert not [n for n in gm.graph.nodes if n.op == "call_function" and "scaled_dot" in str(n.target)]
    assert any("softmax" in v or "bmm" in v or "matmul of two" in v for v in why.values())
    # k and v are not fresh rows: a cache is concatenated in front of them
    gm, w, regions, why = lower(AttnMod(past=3).eval(), [(attn_inputs("causal"),)])
    assert not attn_op_present(regions)
    sd = [n for n in gm.graph.nodes if n.op == "call_function" and "scaled_dot" in str(n.target)][0]
    assert "fresh" in why[sd] or "not fresh" in why[sd]
    # not causal and no mask
    gm, w, regions, why = lower(AttnMod(mask="none").eval(), [(attn_inputs("none"),)])
    assert not attn_op_present(regions)


def test_cat_with_empty_is_the_identity():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(D64, D64, bias=False)
            self.b = nn.Linear(D64, D64, bias=False)
            self.register_buffer("empty", torch.zeros(0), persistent=False)

        def forward(self, x):
            y = torch.tanh(self.a(x))
            c = torch.cat([self.empty, y], dim=-2)            # the empty cache in front: y
            return self.b(torch.tanh(c)), c

    gm, w, regions, why = lower(M().eval(), [(X5,)])
    r = regions[0]
    assert ops_of(r).count("unary") == 2 and ops_of(r).count("mul_mat") == 2
    # the concatenation is an output (a view of the first tanh) and the empty buffer is read only
    cat = by_target(gm, aten.cat.default)[0]
    assert cat in set(r.nodes) and any(o.node is cat for o in r.outputs)
    check_region(r)


# ------------------------------------------------------------------------------------------
# 3. regions: D2, convexity, the probe, symbolic shapes
# ------------------------------------------------------------------------------------------
def test_single_ops_form_no_region():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(D64, D64, bias=False)

        def forward(self, x):
            return self.a(x)

    gm, w, regions, why = region_ops(M())
    assert regions == []
    mm = by_target(gm, aten.mm.default)[0]
    assert "at least two" in why[mm]


class Convex(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.g = nn.Linear(D64, D64, bias=False)
        self.u = nn.Linear(D64, D64, bias=False)
        self.m1 = nn.Linear(D64, D64, bias=False)
        self.m2 = nn.Linear(D64, D64, bias=False)

    def forward(self, x):
        if self.kind == "split_chain":               # mm, silu | softmax | mm, silu
            a = F.silu(self.m1(x))
            c = torch.softmax(a, -1)
            return F.silu(self.m2(c))
        # diamond: g feeds an add directly and through a CPU softmax
        g = F.silu(self.g(x)) * self.u(x)
        c = torch.softmax(g, -1)
        return g + c


def test_cpu_op_in_a_chain_splits_the_region():
    gm, w, regions, why = region_ops(Convex("split_chain"))
    assert [ops_of(r) for r in regions] == [["mul_mat", "unary"], ["mul_mat", "unary"]]
    soft = by_target(gm, aten._softmax.default)[0]
    assert "softmax" in why[soft]
    for r in regions:
        check_region(r)


def test_region_is_convex():
    """g feeds g + softmax(g): the add cannot join g's region (its other operand comes back from the CPU)."""
    gm, w, regions, why = region_ops(Convex("diamond"))
    assert len(regions) == 1 and ops_of(regions[0]) == ["mul_mat", "mul_mat", "glu"]
    add = by_target(gm, aten.add.Tensor)[0]
    assert add not in set(regions[0].nodes)
    assert "at least two" in why[add]
    # g is an output of the region (it is used by the add and by the softmax)
    assert regions[0].outputs and regions[0].outputs[0].recipe["dims"] == [1, 5, 64]
    check_region(regions[0])


def test_probe_asks_per_candidate_and_refusals_are_cpu_nodes():
    calls = []

    def probe(prog):
        calls.append(prog)
        assert len(prog["ops"]) == 1 and "weights" in prog
        return None

    model = Lin(Norm(rms_ok))
    gm, w, regions, why = lower(model.eval(), [(X5,)], probe=probe)
    assert len(regions) == 1
    assert {c["ops"][0]["op"] for c in calls} == {"mul_mat", "rms_norm"}
    assert any(c["ops"][0]["op"] == "mul_mat" and c["weights"] == [{"N": 64, "K": 64, "kind": "dense", "fmt": None}] for c in calls)
    # the same op twice is asked once
    assert sum(c["ops"][0]["op"] == "mul_mat" for c in calls) == 1

    def refuse_rms(prog):
        return "rms_norm: row length 64 is not supported here" if prog["ops"][0]["op"] == "rms_norm" else None

    gm, w, regions, why = lower(model.eval(), [(X5,)], probe=refuse_rms)
    assert regions == []                                   # the two matmuls are no longer connected
    rsqrt_root = [n for n in gm.graph.nodes if n.target is aten.mul.Tensor][-1]
    assert why[rsqrt_root].startswith("device: rms_norm")
    # a probe that refuses the folded matmul with its addend: the matmul and the add are separate ops
    gm, w, regions, why = lower(Mats("fold").eval(), [(X5,)],
                                probe=lambda p: "no addend" if p["ops"][0]["op"] == "mul_mat" and p["ops"][0]["addend"] is not None and p["values"][p["ops"][0]["addend"]]["kind"] == "in" else None)
    assert "add" in [o for r in regions for o in ops_of(r)]


def test_symbolic_row_length_is_cpu():
    class M(nn.Module):
        def forward(self, x):
            return torch.tanh(x) * 2.0 + x

    x = torch.randn(1, 5, 64)
    torch._dynamo.reset()
    torch._dynamo.mark_dynamic(x, 2)
    gm, w = capture(M(), [((x,),)])
    regions, why = L.form_regions(gm, w)
    assert regions == []
    tanh = by_target(gm, aten.tanh.default)[0]
    assert "symbolic" in why[tanh] and "integer" in why[tanh]


# ------------------------------------------------------------------------------------------
# 4. a Qwen-style decoder
# ------------------------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(d) * 0.1 + 1.0)
        self.eps = eps

    def forward(self, x):
        return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps))


def rotate_half(x):
    return torch.cat([-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]], dim=-1)


class QLayer(nn.Module):
    def __init__(self, hid=256, heads=4, kv=2, ff=512):
        super().__init__()
        self.H, self.Hkv, self.D = heads, kv, hid // heads
        self.ln1, self.ln2 = RMSNorm(hid), RMSNorm(hid)
        self.q = nn.Linear(hid, hid)
        self.k = nn.Linear(hid, kv * self.D)
        self.v = nn.Linear(hid, kv * self.D)
        self.o = nn.Linear(hid, hid, bias=False)
        self.gate = nn.Linear(hid, ff, bias=False)
        self.up = nn.Linear(hid, ff, bias=False)
        self.down = nn.Linear(ff, hid, bias=False)

    def forward(self, x, cos, sin, mask):
        B, T, _ = x.shape
        h = self.ln1(x)
        q = self.q(h).view(B, T, -1, self.D).transpose(1, 2)
        k = self.k(h).view(B, T, -1, self.D).transpose(1, 2)
        v = self.v(h).view(B, T, -1, self.D).transpose(1, 2)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        g = self.H // self.Hkv
        k = k[:, :, None].expand(B, self.Hkv, g, T, self.D).reshape(B, self.H, T, self.D)
        v = v[:, :, None].expand(B, self.Hkv, g, T, self.D).reshape(B, self.H, T, self.D)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=self.D ** -0.5)
        x = x + self.o(a.transpose(1, 2).reshape(B, T, -1))
        h = self.ln2(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))


class QwenLike(nn.Module):
    def __init__(self, vocab=256, hid=256, layers=2, mask="where", eps=1e-6):
        super().__init__()
        self.embed = nn.Embedding(vocab, hid)
        self.layers = nn.ModuleList(QLayer(hid) for _ in range(layers))
        self.norm = RMSNorm(hid, eps)
        self.mask = mask
        D = hid // 4
        self.register_buffer("inv_freq", 1.0 / (1e6 ** (torch.arange(0, D, 2).float() / D)), persistent=False)

    def forward(self, ids, use_cache=False):
        T = ids.shape[1]
        x = self.embed(ids)
        pos = torch.arange(T)[None, :]
        freqs = pos[:, :, None].float() * self.inv_freq
        emb = torch.cat([freqs, freqs], -1)
        cos, sin = emb.cos()[:, None], emb.sin()[:, None]
        allowed = torch.ones(T, T, dtype=torch.bool).tril()[None, None]
        mask = torch.where(allowed, 0.0, float("-inf")) if self.mask == "where" else allowed
        for layer in self.layers:
            x = layer(x, cos, sin, mask)
        return F.linear(self.norm(x), self.embed.weight)            # tied head


LAYER_OPS = ["rms_norm", "mul_mat", "mul_mat", "mul_mat", "rope", "rope", "attn", "mul_mat",
             "rms_norm", "mul_mat", "mul_mat", "glu", "mul_mat"]
QWEN_OPS = LAYER_OPS * 2 + ["rms_norm", "mul_mat"]


def ids(T):
    return (torch.randint(0, 256, (1, T)),)


def check_qwen_region(gm, w, r, n_layers=2):
    assert ops_of(r) == (LAYER_OPS * n_layers + ["rms_norm", "mul_mat"])
    prog = check_region(r)
    assert len(r.weights) == 7 * n_layers + 1                      # the tied head shares the embedding
    # the head's weight is the embedding parameter, seen through t()
    assert w[r.weights[-1].node].endswith("embed.weight") or "embed" in w[r.weights[-1].node]
    # inputs: the embedding output, the positions, the mask
    assert [i.kind for i in r.inputs] == ["in", "pos", "mask"]
    assert r.inputs[0].node.target is aten.embedding.default
    assert r.inputs[2].recipe == {"form": "bool", "reshape": "TxT"}
    # one output: the logits, rebuilt with the sequence length taken from the first input
    assert len(r.outputs) == 1
    assert r.outputs[0].recipe["dims"][2] == 256 and r.outputs[0].recipe["transpose12"] is False
    # the embedding, the mask arithmetic and the cos / sin table are CPU nodes
    inside = set(r.nodes)
    for target in (aten.embedding.default, aten.cos.default, aten.sin.default, aten.tril.default, aten.ones.default):
        nodes = by_target(gm, target)
        assert nodes and all(n not in inside for n in nodes), target
    # projections with a bias have the bias as a const addend; the residual is the addend of o and down
    mm = [o for o in r.ops if o["op"] == "mul_mat"]
    bias_mm = [o for o in mm if o["addend"] is not None and r.values[o["addend"]]["kind"] == "const"]
    res_mm = [o for o in mm if o["addend"] is not None and r.values[o["addend"]]["kind"] in ("in", "tmp")]
    assert len(bias_mm) == 3 * n_layers and len(res_mm) == 2 * n_layers
    attn = [o for o in r.ops if o["op"] == "attn"]
    assert all((a["n_head"], a["n_head_kv"], a["d"], a["mask"] is not None) == (4, 2, 64, True) for a in attn)
    assert all(a["scale"] == 0.125 for a in attn)
    ropes = [o for o in r.ops if o["op"] == "rope"]
    assert [o["n_heads"] for o in ropes] == [4, 2] * n_layers and all(o["table"] == ropes[0]["table"] for o in ropes)
    return prog


def test_qwen_like_static():
    m = QwenLike().eval()
    gm, w, r, why = one_region(m, [(ids(6),)])
    check_qwen_region(gm, w, r)
    assert r.scalars == []                                  # eps and the factor are literals here
    assert r.ops[0]["eps"] == L.f32(1e-6)


@pytest.mark.parametrize("dynamic,calls", [(None, [6, 9]), (True, [6])])
def test_qwen_like_symbolic_sequence(dynamic, calls):
    m = QwenLike().eval()
    gm, w, r, why = one_region(m, [(ids(t),) for t in calls], dynamic=dynamic)
    check_qwen_region(gm, w, r)
    assert str(r.rows).startswith("s")                      # T is a SymInt
    if r.scalars:                                           # Dynamo lifted eps to a 0-dim input
        prog = r.to_program([2.5e-7] * len(r.scalars))
        eps = {o["eps"] for o in prog["ops"] if o["op"] == "rms_norm"}
        assert eps == {L.f32(2.5e-7)}
        with pytest.raises(ValueError):
            r.to_program()


def test_qwen_like_bool_mask():
    m = QwenLike(mask="bool").eval()
    gm, w, r, why = one_region(m, [(ids(6),)])
    assert ops_of(r) == QWEN_OPS


# ------------------------------------------------------------------------------------------
# 5. the Block of test_backend.py: LayerNorm stays on the CPU
# ------------------------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, d=128, heads=4, ff=352):
        super().__init__()
        self.h = heads
        self.ln1 = nn.LayerNorm(d)
        self.q, self.k, self.v = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.o = nn.Linear(d, d, bias=False)
        self.ln2 = nn.LayerNorm(d)
        self.gate = nn.Linear(d, ff, bias=False)
        self.up = nn.Linear(d, ff, bias=False)
        self.down = nn.Linear(ff, d, bias=False)

    def forward(self, x):
        B, S, D = x.shape
        h = self.ln1(x)
        q, k, v = (m(h).view(B, S, self.h, D // self.h).transpose(1, 2) for m in (self.q, self.k, self.v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.o(a.transpose(1, 2).reshape(B, S, D))
        return x + self.down(F.silu(self.gate(self.ln2(x))) * self.up(self.ln2(x)))


def test_block_regions():
    gm, w, regions, why = lower(Block().eval(), [((torch.randn(2, 12, 128),),)])
    assert [ops_of(r) for r in regions] == [["mul_mat", "mul_mat", "glu", "mul_mat"]]
    r = regions[0]
    # gate, up, glu, down with the residual: the two LayerNorm outputs and the residual are the inputs
    assert [i.node.name for i in r.inputs] == ["getitem_5", "getitem_8", "add"]
    assert r.values[r.ops[3]["addend"]]["kind"] == "in"
    assert r.outputs[0].recipe == {"dims": [2, 12, 128], "transpose12": False, "contiguous": False}
    # LayerNorm, and attention with batch 2, stay on the CPU, with their reasons
    ln = by_target(gm, aten.native_layer_norm.default)
    assert len(ln) == 3 and all("layer norm" in why[n] for n in ln)
    sd = [n for n in gm.graph.nodes if n.op == "call_function" and "scaled_dot" in str(n.target)][0]
    assert "batch" in why[sd]
    # q, k, v, o: lone matmuls (the o projection folded with its residual) stay device_mm nodes
    for n in by_target(gm, aten.addmm.default) + by_target(gm, aten.mm.default)[:1]:
        assert "at least two" in why[n]
    # what touches the region on the CPU is listed with a reason
    assert {n.name for n in r.touching} >= {"getitem_5", "getitem_8", "add"}
    check_region(r)


def test_block_symbolic_batch_and_sequence():
    x = torch.randn(2, 12, 128)
    gm, w, regions, why = lower(Block().eval(), [((x,),), ((torch.randn(2, 20, 128),),)])
    assert [ops_of(r) for r in regions] == [["mul_mat", "mul_mat", "glu", "mul_mat"]]
    dims = regions[0].outputs[0].recipe["dims"]
    assert dims[0] == 2 and dims[2] == 128 and isinstance(dims[1], dict)
    check_region(regions[0])


# ------------------------------------------------------------------------------------------
# 6. Qwen2.5-0.5B from models/hf
# ------------------------------------------------------------------------------------------
hf = pytest.mark.skipif(not (HF_QWEN / "config.json").exists(), reason="models/hf/Qwen2.5-0.5B-Instruct is not present")


def _load_hf():
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(str(HF_QWEN), dtype=torch.float32, use_safetensors=True,
                                                trust_remote_code=False).eval()


def _hf_capture(use_cache, lengths):
    m = _load_hf()
    calls = [((), {"input_ids": torch.tensor([[9707, 11, 1246, 525, 498, 30, 5, 6, 7][:t]]), "use_cache": use_cache})
             for t in lengths]
    return capture(m, calls)


@pytest.fixture(scope="module")
def hf_graph():
    gm, w = _hf_capture(False, [6, 9])           # the second length makes Dynamo compile a dynamic graph
    return gm, w, *L.form_regions(gm, w)


@pytest.fixture(scope="module")
def hf_graph_cache():
    gm, w = _hf_capture(True, [6])
    return gm, w, *L.form_regions(gm, w)


def _no_cpu_inside(gm, r):
    idx = {n: i for i, n in enumerate(gm.graph.nodes)}
    lo, hi = min(idx[o] for o in r.op_nodes), max(idx[o] for o in r.op_nodes)
    inside, bypass = set(r.nodes), set(r.bypassed)
    extra = [n for n in gm.graph.nodes if lo < idx[n] < hi and n.op == "call_function"
             and n not in inside and n not in bypass]
    return extra


@hf
def test_hf_qwen_one_region(hf_graph):
    gm, w, regions, why = hf_graph
    assert len(regions) == 1
    r = regions[0]
    c = r.op_counts()
    assert c["attn"] == 24 and c["mul_mat"] == 169 and c["rope"] == 48 and c["glu"] == 24 and c["rms_norm"] == 49
    assert len(r.ops) == 24 * 13 + 2
    # nothing between the first norm and the logits runs on the CPU
    assert _no_cpu_inside(gm, r) == []
    assert [i.kind for i in r.inputs] == ["in", "pos", "mask"]
    assert len(r.outputs) == 1 and r.outputs[0].node.meta["val"].shape[-1] == 151936
    # the embedding, the mask arithmetic and the cos / sin chain stay on the CPU
    assert why[by_target(gm, aten.embedding.default)[0]]
    assert len(r.weights) == 169 and len(r.consts) == 72 + 49 + 1
    prog = r.to_program([1e-6] * len(r.scalars)) if r.scalars else r.to_program()
    check_program(prog, r)


@hf
def test_hf_qwen_with_cache_outputs(hf_graph_cache):
    gm, w, regions, why = hf_graph_cache
    assert len(regions) == 1
    r = regions[0]
    assert r.op_counts()["attn"] == 24 and r.op_counts()["mul_mat"] == 169
    # the logits and, per layer, the k and v that the cache returns (head-major views of the k rows)
    assert len(r.outputs) == 49
    hm = [o for o in r.outputs if o.recipe["transpose12"]]
    assert len(hm) == 48 and all(o.recipe["dims"][2:] == [2, 64] for o in hm)
    assert _no_cpu_inside(gm, r) == []
    check_region(r)
