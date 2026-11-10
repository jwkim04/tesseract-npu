"""torch.compile(backend="tesseract"): partition, numerics against eager, fallbacks, weight cache,
program cache, T tiling, training refusal, dynamic shapes, debug output."""
from __future__ import annotations

import copy
import gc
import math
import warnings

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import spec_model as sm
import tesseract_torch as tt

NMSE_MAX = 5e-4


def _nmse(y, ref) -> float:
    return sm.nmse(y.detach().double().numpy(), ref.detach().double().numpy())


def _same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    iv = {torch.float32: torch.int32, torch.float16: torch.int16, torch.bfloat16: torch.int16, torch.float64: torch.int64}
    return torch.equal(a.contiguous().view(iv[a.dtype]), b.contiguous().view(iv[b.dtype]))


def eager_ref():
    """Eager execution with every supported nn.Linear on the reference model (tt.reference_linears)."""
    return tt.reference_linears()


class MLP(nn.Module):
    def __init__(self, d_in=256, d_h=512, d_out=64):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_h)
        self.fc2 = nn.Linear(d_h, 128)
        self.norm = nn.LayerNorm(128)
        self.fc3 = nn.Linear(128, d_out, bias=False)

    def forward(self, x):
        return self.fc3(self.norm(F.gelu(self.fc2(F.gelu(self.fc1(x))))))


class Block(nn.Module):
    """Pre-norm decoder block: causal multi-head attention (q / k / v / o projections, SDPA) and a
    SwiGLU MLP, all nn.Linear."""

    def __init__(self, d=128, heads=4, ff=352):
        super().__init__()
        self.h = heads
        self.ln1 = nn.LayerNorm(d)
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
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


def _model(cls, dtype=torch.float32, seed=0, **kw):
    torch.manual_seed(seed)
    return cls(**kw).eval().to(dtype)


def _x(shape, dtype=torch.float32, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=g).to(dtype)


# ------------------------------------------------------------------------------------------
def test_backend_registered():
    assert "tesseract" in torch._dynamo.list_backends(exclude_tags=())
    m = _model(MLP)
    c = tt.compile(m)
    with torch.no_grad():
        y = c(_x((2, 256)))
    assert y.shape == (2, 64)
    assert tt.counters()["device_calls"] == 3


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_mlp_compiled_vs_eager(dtype):
    m = _model(MLP, dtype)
    x = _x((3, 7, 256), dtype)
    c = torch.compile(m, backend="tesseract")
    with torch.no_grad():
        y = c(x)
        cnt = tt.counters()
        y_eager = m(x)
        y_exact = copy.deepcopy(m).double()(x.double())
        with eager_ref():
            y_ref = m(x)
    assert y.dtype == dtype and y.shape == (3, 7, 64)
    assert (cnt["nodes_device"], cnt["nodes_cpu"], cnt["device_calls"], cnt["device_tokens"]) == (3, 0, 3, 63)
    assert cnt["uploads"] == 3 and cnt["rt_linears"] == 3
    assert _same_bits(y, y_ref), "compiled != eager with reference linears"
    e = _nmse(y, y_exact)
    assert e < NMSE_MAX, e
    if dtype == torch.float32:
        assert _nmse(y, y_eager) < NMSE_MAX


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_transformer_block_compiled_vs_eager(dtype):
    m = _model(Block, dtype)
    x = _x((2, 12, 128), dtype)
    c = tt.compile(m, verify=True)
    try:
        with torch.no_grad():
            y = c(x)
            cnt = tt.counters()
            y_exact = copy.deepcopy(m).double()(x.double())
            with eager_ref():
                y_ref = m(x)
    finally:
        tt.set_verify(False)
    assert (cnt["nodes_device"], cnt["device_calls"], cnt["verified_calls"]) == (7, 7, 7)
    assert cnt["nodes_cpu"] == 0   # attention (q k^T, p v) stays inside the fused SDPA op on the CPU
    assert cnt["uploads"] == 7
    e = _nmse(y, y_exact)
    assert e < NMSE_MAX, e
    assert _same_bits(y, y_ref), "compiled != eager with reference linears"


def test_fallback_unsupported_shapes_and_dtypes():
    """K % 32 != 0 and float64 weights stay on the CPU (results identical to eager); eligible
    neighbors still go to the device."""

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.odd = nn.Linear(48, 64)          # K = 48
            self.ok = nn.Linear(64, 32)
            self.d64 = nn.Linear(32, 32).double()

        def forward(self, x):
            h = self.ok(F.relu(self.odd(x)))
            return h, self.d64(h.double())

    torch.manual_seed(0)
    m = M().eval()
    x = _x((5, 48))
    c = torch.compile(m, backend="tesseract")
    with torch.no_grad():
        h, z = c(x)
        with eager_ref():
            h_ref, z_ref = m(x)
    cnt = tt.counters()
    assert (cnt["nodes_device"], cnt["nodes_cpu"], cnt["device_calls"]) == (1, 2, 1)
    assert _same_bits(h, h_ref) and _same_bits(z, z_ref)
    reasons = [d.reason for d in _decisions(m, x)]
    assert any("K % 32 != 0 (K = 48)" in r for r in reasons)
    assert any("dtype float64" in r for r in reasons)


def _decisions(model, *args):
    """Partition decisions of a fresh compile (offload off: report only)."""
    torch._dynamo.reset()
    out = []
    orig = tt.backend.partition

    def spy(aot_gm, weights, opts, gid):
        ds = orig(aot_gm, weights, opts, gid)
        out.extend(ds)
        return ds

    tt.backend.partition = spy
    try:
        with torch.no_grad():
            tt.compile(model, offload=False)(*args)
    finally:
        tt.backend.partition = orig
    return out


def test_all_cpu_model_unchanged():
    """A model without supported weight matmuls produces exactly eager's results."""
    m = nn.Sequential(nn.Linear(33, 17), nn.Tanh(), nn.Linear(17, 5)).eval()
    x = _x((4, 33))
    with torch.no_grad():
        y = torch.compile(m, backend="tesseract")(x)
        assert torch.equal(y, m(x))
    assert tt.counters()["device_calls"] == 0 and tt.counters()["nodes_cpu"] == 2


def test_weight_inplace_update_reuploads():
    m = _model(MLP)
    x = _x((4, 256))
    c = torch.compile(m, backend="tesseract")
    with torch.no_grad():
        y0 = c(x)
        m.fc1.weight.mul_(0.5)                     # in place: bumps _version
        y1 = c(x)
        with eager_ref():
            y1_ref = m(x)
        assert not torch.equal(y0, y1) and _same_bits(y1, y1_ref)
        cnt = tt.counters()
        assert (cnt["uploads"], cnt["reuploads"], cnt["cached_weights"]) == (4, 1, 3)
        # load_state_dict copies in place: every weight is uploaded again on the next call
        sd = {k: v * 1.25 for k, v in _model(MLP, seed=5).state_dict().items()}
        m.load_state_dict(sd)
        y2 = c(x)
        with eager_ref():
            y2_ref = m(x)
        assert _same_bits(y2, y2_ref)
        cnt = tt.counters()
        assert (cnt["uploads"], cnt["reuploads"]) == (7, 4)
        y3 = c(x)
        assert _same_bits(y3, y2) and tt.counters()["uploads"] == 7


def test_repeated_calls_hit_program_cache():
    m = _model(MLP)
    x = _x((1, 256))
    c = torch.compile(m, backend="tesseract")
    with torch.no_grad():
        y0 = c(x)
        first = tt.counters()
        tt.reset_counters()
        ys = [c(x) for _ in range(5)]
    cnt = tt.counters()
    assert (first["rt_prog_misses"], first["uploads"]) == (3, 3)
    assert (cnt["rt_prog_hits"], cnt["rt_prog_misses"], cnt["rt_execs"]) == (15, 0, 15)
    assert (cnt["uploads"], cnt["cache_hits"], cnt["rt_h2d_weight_bytes"]) == (0, 15, 0)
    # host traffic per call: x in, y out for each of the 3 matmuls (N % 16 == 0: y read directly)
    per_call_in = 4 * (256 + 512 + 128)
    per_call_out = 4 * (512 + 128 + 64)
    assert cnt["rt_h2d_bytes"] == 5 * per_call_in and cnt["rt_d2h_bytes"] == 5 * per_call_out
    assert all(_same_bits(y, y0) for y in ys)


def test_tokens_beyond_t_max():
    """B * S > T_MAX tokens: one device program per matmul call, T tiles inside it."""
    tmax = tt.device().config["t_max"]
    m = _model(MLP)
    c = torch.compile(m, backend="tesseract", dynamic=False)
    for S in (3 * tmax + 5, 150):
        x = _x((2, S, 256), seed=S)
        torch._dynamo.reset()
        tt.reset_counters()
        with torch.no_grad():
            y = c(x)
            with eager_ref():
                y_ref = m(x)
        cnt = tt.counters()
        assert cnt["rt_execs"] == 3 and cnt["device_tokens"] == 3 * 2 * S
        assert _same_bits(y, y_ref)


def test_training_refused_runs_eager():
    m = _model(MLP)
    x = _x((4, 256))
    c = torch.compile(m, backend="tesseract")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        y = c(x)                                    # grad mode on, parameters require grad
    assert any("needs autograd" in str(i.message) for i in w)
    assert y.requires_grad
    y_eager = m(x)
    assert torch.equal(y, y_eager)
    y.sum().backward()                               # eager graph: autograd works
    assert m.fc1.weight.grad is not None
    cnt = tt.counters()
    assert (cnt["graphs_eager"], cnt["device_calls"]) == (1, 0)
    # strict: refuse instead
    torch._dynamo.reset()
    with pytest.raises(Exception, match="needs autograd"):
        tt.compile(m, strict=True)(x)
    # the same model under no_grad goes to the device (requires_grad parameters are fine)
    torch._dynamo.reset()
    with torch.no_grad():
        y2 = tt.compile(m)(x)
    assert tt.counters()["device_calls"] == 3 and not y2.requires_grad


def test_inference_mode():
    m = _model(Block)
    x = _x((1, 5, 128))
    with torch.inference_mode():
        y = tt.compile(m)(x)
        with eager_ref():
            y_ref = m(x)
    assert _same_bits(y, y_ref) and tt.counters()["device_calls"] == 7


def test_weight_operand_forms():
    """Weights on either side and in other layouts: W @ x, x @ W (Conv1D-style [K, N]), matmul with a
    transposed weight, 1D input, mv, addmm with alpha / beta, a buffer, a sliced weight."""

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.w_left = nn.Parameter(torch.randn(48, 64))     # [N, K]: W @ x
            self.w_kn = nn.Parameter(torch.randn(64, 80))       # [K, N]: x @ W
            self.w_nk = nn.Parameter(torch.randn(96, 64))       # [N, K]: matmul(x, W.t())
            self.b = nn.Parameter(torch.randn(96))
            self.register_buffer("w_buf", torch.randn(40, 64))
            self.w_big = nn.Parameter(torch.randn(64, 128))     # sliced: w_big[:, 32:96] is [64, 64]

        def forward(self, x, v):
            r1 = self.w_left @ x.t()                             # [48, T]
            r2 = x @ self.w_kn                                   # [T, 80]
            r3 = torch.matmul(x, self.w_nk.t())                  # [T, 96]
            r4 = F.linear(v, self.w_nk, self.b)                  # 1D input -> [96]
            r5 = torch.mv(self.w_left, v)                        # [48]
            r6 = torch.addmm(self.b, x, self.w_nk.t(), beta=0.5, alpha=2.0)
            r7 = x @ self.w_buf.t()
            r8 = x @ self.w_big[:, 32:96].t()
            return r1, r2, r3, r4, r5, r6, r7, r8

    torch.manual_seed(3)
    m = M().eval()
    x = _x((6, 64))
    v = _x((64,), seed=2)
    with torch.no_grad():
        outs = torch.compile(m, backend="tesseract")(x, v)
        L = tt.linear_ref
        exp = (
            L(x, m.w_left).t().contiguous(),
            L(x, m.w_kn.t().contiguous()),
            L(x, m.w_nk),
            L(v, m.w_nk, m.b),
            L(v, m.w_left),
            torch.add(L(x, m.w_nk) * 2.0, m.b, alpha=0.5),
            L(x, m.w_buf),
            L(x, m.w_big[:, 32:96].contiguous()),
        )
    cnt = tt.counters()
    assert cnt["nodes_device"] == 8 and cnt["nodes_cpu"] == 0, tt.format_counters()
    for i, (a, b) in enumerate(zip(outs, exp)):
        assert _same_bits(a, b), f"output {i}"


def test_addmm_beta_zero_ignores_input():
    w = nn.Parameter(torch.randn(32, 64), requires_grad=False)
    bias = torch.full((32,), float("nan"))

    def f(x):
        return torch.addmm(bias, x, w.t(), beta=0.0)

    x = _x((3, 64))
    with torch.no_grad():
        y = torch.compile(f, backend="tesseract")(x)
    assert torch.isfinite(y).all()
    assert _same_bits(y, tt.linear_ref(x, w))


def test_dynamic_sequence_lengths_upload_once():
    m = _model(Block)
    c = torch.compile(m, backend="tesseract")
    with torch.no_grad():
        for S in (5, 7, 9, 16, 3):
            x = _x((1, S, 128), seed=S)
            y = c(x)
            with eager_ref():
                assert _same_bits(y, m(x)), S
    cnt = tt.counters()
    assert cnt["uploads"] == 7 and cnt["reuploads"] == 0
    assert cnt["device_tokens"] == 7 * (5 + 7 + 9 + 16 + 3)


def test_tied_and_shared_weights_upload_once():
    class Tied(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(64, 64, bias=False)
            self.b = nn.Linear(64, 64, bias=False)
            self.b.weight = self.a.weight                         # tied

        def forward(self, x):
            return self.b(torch.tanh(self.a(x))) + self.a(x)

    m = _model(Tied)
    x = _x((2, 64))
    with torch.no_grad():
        y = torch.compile(m, backend="tesseract")(x)
        with eager_ref():
            y_ref = m(x)
    cnt = tt.counters()
    assert cnt["uploads"] == 1 and cnt["device_calls"] == cnt["nodes_device"] == 3
    assert _same_bits(y, y_ref)


def test_graph_breaks_share_weights():
    class GB(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(64, 64)

        def forward(self, x):
            h = self.fc(x)
            torch._dynamo.graph_break()
            return self.fc(h)

    m = _model(GB)
    x = _x((3, 64))
    with torch.no_grad():
        y = torch.compile(m, backend="tesseract")(x)
        with eager_ref():
            y_ref = m(x)
    cnt = tt.counters()
    assert cnt["graphs"] == 2 and cnt["uploads"] == 1 and cnt["device_calls"] == 2
    assert _same_bits(y, y_ref)


def test_dead_weights_are_released():
    m = _model(MLP)
    with torch.no_grad():
        torch.compile(m, backend="tesseract")(_x((1, 256)))
    assert len(tt.weight_cache()) == 3
    n_before = tt.device().memory()["n_weights"]
    del m
    torch._dynamo.reset()
    gc.collect()
    assert tt.weight_cache().sweep() == 3
    assert tt.device().memory()["n_weights"] == n_before - 3
    assert tt.counters()["weights_freed"] == 3


def test_debug_output(capfd):
    tt.set_debug(1)
    try:
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(64, 40)
                self.odd = nn.Linear(40, 8)

            def forward(self, x):
                return self.odd(self.fc(x))

        with torch.no_grad():
            tt.compile(_model(M))(_x((2, 64)))
    finally:
        tt.set_debug(0)
    err = capfd.readouterr().err
    assert "matmul(s) on Tesseract, 0 on the CPU" not in err
    assert "1 matmul(s) on Tesseract, 1 on the CPU" in err
    assert "tesseract  addmm" in err and "fc.weight  [N=40 K=64]" in err
    assert "cpu        addmm_1" in err and "odd.weight: K % 32 != 0 (K = 40)" in err
    assert "upload fc.weight [40 x 64] float32 -> q8_0" in err


def test_model_conversion_bf16_large_dims():
    """A wider bf16 model (Qwen2.5-0.5B-like MLP widths): device vs eager in float64."""

    class Wide(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = nn.Linear(896, 4864, bias=False)
            self.up = nn.Linear(896, 4864, bias=False)
            self.down = nn.Linear(4864, 896, bias=False)

        def forward(self, x):
            return self.down(F.silu(self.gate(x)) * self.up(x))

    m = _model(Wide, torch.bfloat16)
    x = _x((1, 4, 896), torch.bfloat16)
    with torch.no_grad():
        y = tt.compile(m)(x)
        with eager_ref():
            y_ref = m(x)
        y64 = copy.deepcopy(m).double()(x.double())
    assert _same_bits(y, y_ref)
    e = _nmse(y, y64)
    assert e < NMSE_MAX, e
    assert math.isclose(tt.counters()["upload_bytes"], 3 * 896 * 4864 * 2)


def test_partition_of_aten_linear_and_matmul_nodes():
    """aten.linear / aten.matmul / aten.mm nodes (as in graphs that were not decomposed) are lowered too."""
    import torch.fx as fx

    g = fx.Graph()
    W = g.placeholder("w")
    b = g.placeholder("b")
    x3 = g.placeholder("x3")
    v = g.placeholder("v")
    lin = g.call_function(torch.ops.aten.linear.default, (x3, W, b))
    mat = g.call_function(torch.ops.aten.matmul.default, (x3, g.call_function(torch.ops.aten.t.default, (W,))))
    mvv = g.call_function(torch.ops.aten.matmul.default, (W, v))
    g.output((lin, mat, mvv))
    gm = fx.GraphModule(torch.nn.Module(), g)
    Wt, bt = torch.randn(48, 96), torch.randn(48)
    xt, vt = _x((2, 5, 96)), _x((96,), seed=3)
    vals = {"w": Wt, "b": bt, "x3": xt, "v": vt}
    for n in gm.graph.nodes:                      # metadata as AOT would record it
        if n.op == "placeholder":
            n.meta["val"] = vals[n.name]
    from torch.fx.passes.shape_prop import ShapeProp
    ShapeProp(gm).propagate(Wt, bt, xt, vt)
    for n in gm.graph.nodes:
        if n.op == "call_function":
            n.meta["val"] = torch.empty(n.meta["tensor_meta"].shape, dtype=n.meta["tensor_meta"].dtype)
    weights = {W: "W"}
    ds = tt.backend.partition(gm, weights, tt.Options(), gid=999)
    assert [d.device for d in ds] == [True, True, True], [d.reason for d in ds]
    with torch.no_grad():
        r_lin, r_mat, r_mv = gm(Wt, bt, xt, vt)
    assert _same_bits(r_lin, tt.linear_ref(xt, Wt, bt))
    assert _same_bits(r_mat, tt.linear_ref(xt, Wt))
    assert _same_bits(r_mv, tt.linear_ref(vt, Wt))


def test_without_inlined_nn_modules():
    """inline_inbuilt_nn_modules=False: parameters are graph attributes, lifted by AOT."""
    m = _model(MLP)
    x = _x((3, 256))
    with torch._dynamo.config.patch(inline_inbuilt_nn_modules=False), torch.no_grad():
        y = torch.compile(m, backend="tesseract")(x)
    with torch.no_grad(), eager_ref():
        y_ref = m(x)
    assert tt.counters()["nodes_device"] == 3
    assert _same_bits(y, y_ref)


def test_close_and_reopen_device():
    m = _model(MLP)
    x = _x((2, 256))
    c = torch.compile(m, backend="tesseract")
    with torch.no_grad():
        y0 = c(x)
        tt.close_device()
        assert not tt._C.is_open() and len(tt.weight_cache()) == 0
        y1 = c(x)                                   # reopens the device, uploads again
    assert _same_bits(y0, y1)
    assert tt.counters()["uploads"] == 6 and tt._C.is_open()   # 3 before the close, 3 after


def test_weight_that_cannot_be_placed_runs_on_cpu(monkeypatch):
    """If a weight cannot be uploaded (e.g. WEIGHT heap full), its node runs on the CPU with a warning."""
    m = _model(MLP)
    x = _x((2, 256))
    orig = tt.weight_cache().acquire

    def acquire(w, name=""):
        if name == "fc2.weight":
            raise tt.TesseractError("tsr_rt: WEIGHT heap full (test)")
        return orig(w, name)

    monkeypatch.setattr(tt.weight_cache(), "acquire", acquire)
    with warnings.catch_warnings(record=True) as w, torch.no_grad():
        warnings.simplefilter("always")
        c = torch.compile(m, backend="tesseract")
        y = c(x)
        y2 = c(x)

        class Fc2OnCpu:
            def __enter__(self):
                self.orig = nn.Linear.forward
                ref = tt.reference_linears()
                ref.__enter__()
                self.ref = ref
                patched = nn.Linear.forward

                def fwd(mod, inp):
                    return self.orig(mod, inp) if mod is m.fc2 else patched(mod, inp)

                nn.Linear.forward = fwd

            def __exit__(self, *a):
                nn.Linear.forward = self.orig
                self.ref.__exit__(*a)

        with Fc2OnCpu():
            y_ref = m(x)
    assert any("fc2.weight" in str(i.message) and "stays on the CPU" in str(i.message) for i in w)
    cnt = tt.counters()
    assert cnt["cpu_fallback_calls"] == 2 and cnt["device_calls"] == 4
    assert _same_bits(y, y_ref) and _same_bits(y2, y_ref)


def test_weight_rewritten_every_call_goes_back_to_cpu():
    """A weight changed in place before every call is uploaded each time; after 4 such calls in a row
    its node runs on the CPU (warning), with the correct result either way."""
    m = _model(MLP)
    x = _x((2, 256))
    c = torch.compile(m, backend="tesseract")
    with warnings.catch_warnings(record=True) as w, torch.no_grad():
        warnings.simplefilter("always")
        for i in range(6):
            m.fc2.weight.mul_(1.01)
            y = c(x)
            if i < 3:                                  # fc2 still on the device
                with eager_ref():
                    assert _same_bits(y, m(x))
            assert _nmse(y, m(x)) < NMSE_MAX
    assert any("fc2.weight" in str(i.message) and "consecutive calls" in str(i.message) for i in w)
    cnt = tt.counters()
    assert cnt["cpu_fallback_calls"] == 2            # calls 5 and 6
    assert cnt["uploads"] == 3 + 3                   # fc1, fc2, fc3 + fc2 on calls 2..4

def test_local_scale_region_and_shared_result(dev):
    model = torch.nn.Linear(128, 69, bias=False).eval()
    x = _x((5, 128), seed=54)
    with torch.no_grad(), eager_ref():
        ref = model(x)
    tt.reset_counters()
    with torch.no_grad():
        compiled = tt.compile(lambda t: model(t) * 2.0)
        assert _same_bits(compiled(x), ref * 2.0)
        if dev.config["local_scale"]:
            assert tt.counters()["local_scale_calls"] == 1
        else:
            assert tt.counters()["local_scale_calls"] == 0

        def escapes(t):
            y = model(t)
            return y, y * 2.0

        before = tt.counters()["local_scale_calls"]
        y, y2 = tt.compile(escapes)(x)
        assert _same_bits(y, ref) and _same_bits(y2, ref * 2.0)
        assert tt.counters()["local_scale_calls"] == before

@pytest.mark.parametrize("dtype,bias,scale", [(torch.bfloat16, False, 2.0), (torch.float32, True, 2.0),
                                             (torch.float32, False, 0.5)])
def test_local_scale_region_preserves_cpu_contract(dtype, bias, scale):
    model = torch.nn.Linear(128, 69, bias=bias, dtype=dtype).eval()
    x = _x((5, 128), dtype=dtype, seed=66)
    with torch.no_grad(), eager_ref():
        ref = model(x) * scale
    tt.reset_counters()
    with torch.no_grad():
        y = tt.compile(lambda t: model(t) * scale)(x)
    assert _same_bits(y, ref) and y.dtype == dtype
    assert tt.counters()["device_calls"] == 1 and tt.counters()["local_scale_calls"] == 0
