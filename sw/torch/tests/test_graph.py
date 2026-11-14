"""Full mode of the backend: regions of ops, one launch per region (sw/torch/tesseract_torch/lower.py,
backend.partition_regions, runtime.device_region).

Every test runs on the simulator (TSR_DEVICE=sim) with TSR_TORCH_VERIFY on: each region call is also
executed by the reference executor on the device's memory and compared bit for bit. Against eager
float32 the results are statistical (the device quantizes activations to 8 bits and has its own
norms, activations, RoPE and attention): NMSE bounds.
"""
from __future__ import annotations

import threading
import warnings

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import spec_model as sm
import tesseract_torch as tt
from tesseract_torch import backend as B
from tesseract_torch import lower as LW
from tesseract_torch import runtime as RT

import test_lower as TL
from test_lower import AttnMod, Block, Elementwise, Gated, Lin, Mats, Norm, QwenLike, RopeMod, ids

D64 = TL.D64
NMSE_FREE = 5e-4      # a region without attention: 8-bit weights and activations
NMSE_ATTN = 5e-3      # with attention (8-bit K, V, Q) and a stack of layers


_FIRST = [0]


@pytest.fixture(autouse=True)
def _verify_on():
    old = tt.verify_enabled()
    tt.set_verify(True)
    _FIRST[0] = len(tt.regions())
    yield
    tt.set_verify(old)
    tt.set_debug(0)


def regs():
    """The regions formed by the current test."""
    return tt.regions()[_FIRST[0]:]


def nmse(y, ref) -> float:
    y, ref = y.detach().double(), ref.detach().double()
    return float(((y - ref) ** 2).sum() / (ref ** 2).sum())


def full(model, **kw):
    return tt.compile(model, mode="full", **kw)


def region_ops(i=-1) -> list[str]:
    return [o["op"] for o in regs()[i].region.ops]


def run(model, *args, **kw):
    """(compiled, y, eager y) for one call of `model` in full mode."""
    c = full(model, **kw)
    with torch.no_grad():
        y = c(*args)
        ref = model(*args)
    return c, y, ref


def cnt():
    return tt.counters()


# ------------------------------------------------------------------------------------------
# every pattern alone (a region needs two device ops, so the pattern sits between two matmuls)
# ------------------------------------------------------------------------------------------
X64 = torch.randn(1, 5, D64, generator=torch.Generator().manual_seed(3))

ELEMENTWISE = [
    ("tanh", lambda s, t: torch.tanh(t), "unary"),
    ("sigmoid", lambda s, t: torch.sigmoid(t), "unary"),
    ("exp", lambda s, t: torch.exp(0.3 * t), "unary"),
    ("gelu_tanh", lambda s, t: F.gelu(t, approximate="tanh"), "unary"),
    ("silu", lambda s, t: F.silu(t), "unary"),
    ("add_row", lambda s, t: t + s.row, "add"),
    ("mul_row", lambda s, t: s.row * t, "mul"),
    ("mul_scalar", lambda s, t: t * 0.3, "scale"),
    ("div_scalar", lambda s, t: t / 4.0, "scale"),
    ("add_scalar", lambda s, t: t + 0.5, "scale"),
    ("add_dev", lambda s, t: t + F.silu(t), "add"),
]


@pytest.mark.parametrize("name,f,op", ELEMENTWISE, ids=[e[0] for e in ELEMENTWISE])
def test_elementwise_alone(name, f, op):
    torch.manual_seed(1)
    c, y, ref = run(Elementwise(f).eval(), X64)
    assert op in region_ops() and cnt()["regions"] == 1
    assert (cnt()["region_calls"], cnt()["rt_graph_launches"], cnt()["device_calls"]) == (1, 1, 0)
    assert nmse(y, ref) < NMSE_FREE, nmse(y, ref)


@pytest.mark.parametrize("fn,transform", [(TL.rms_ok, "identity"), (TL.rms_noweight, None), (TL.rms_gemma, "one_plus")],
                         ids=["weight", "no_weight", "gemma"])
def test_rms_norm_alone(fn, transform):
    torch.manual_seed(2)
    c, y, ref = run(Lin(Norm(fn)).eval(), X64)
    assert region_ops() == ["mul_mat", "rms_norm", "mul_mat"]
    if transform:
        assert regs()[-1].region.consts[0].transform == transform
    assert (cnt()["regions"], cnt()["region_calls"], cnt()["rt_graph_launches"]) == (1, 1, 1)
    assert nmse(y, ref) < NMSE_FREE, nmse(y, ref)


@pytest.mark.parametrize("act,variant", [(F.silu, 0), (lambda t: F.gelu(t, approximate="tanh"), 2)], ids=["swiglu", "geglu"])
def test_glu_alone(act, variant):
    torch.manual_seed(3)
    c, y, ref = run(Gated(act).eval(), X64)
    g = [o for o in regs()[-1].region.ops if o["op"] == "glu"]
    assert len(g) == 1 and g[0]["variant"] == variant
    assert (cnt()["region_calls"], cnt()["rt_graph_launches"]) == (1, 1)
    assert nmse(y, ref) < NMSE_FREE, nmse(y, ref)


@pytest.mark.parametrize("mode", ["fold", "fold_swapped"])
def test_mul_mat_with_bias_and_residual_alone(mode):
    torch.manual_seed(4)
    c, y, ref = run(Mats(mode).eval(), X64)
    mm = [o for o in regs()[-1].region.ops if o["op"] == "mul_mat"]
    assert len(mm) == 2 and all(o["addend"] is not None for o in mm)      # the bias and the residual are addends
    assert "add" not in region_ops()
    assert nmse(y, ref) < NMSE_FREE, nmse(y, ref)


def test_rope_alone():
    torch.manual_seed(5)
    x, pos = TL.rope_inputs(5)
    pos = torch.arange(5)[None] + 3
    c, y, ref = run(RopeMod().eval(), x, pos)
    assert region_ops() == ["mul_mat", "rope"]
    assert y.shape == ref.shape == (1, 2, 5, 64)
    assert (cnt()["region_calls"], cnt()["rt_graph_launches"]) == (1, 1)
    assert nmse(y, ref) < NMSE_FREE, nmse(y, ref)


def test_rope_factor_alone():
    torch.manual_seed(5)
    x, pos = TL.rope_inputs(5)
    c, y, ref = run(RopeMod(factor=1.25).eval(), x, pos)
    assert regs()[-1].region.ops[1]["factor"] == 1.25
    assert nmse(y, ref) < NMSE_FREE, nmse(y, ref)


@pytest.mark.parametrize("mask", ["causal", "bool", "where", "additive"])
def test_attention_alone(mask):
    torch.manual_seed(6)
    args = TL.attn_inputs(mask, T=6)
    c, y, ref = run(AttnMod(mask=mask).eval(), *args)
    assert region_ops() == ["mul_mat", "mul_mat", "mul_mat", "attn", "mul_mat"]
    assert (cnt()["regions"], cnt()["region_calls"], cnt()["rt_graph_launches"], cnt()["device_calls"]) == (1, 1, 1, 0)
    assert nmse(y, ref) < NMSE_ATTN, nmse(y, ref)


def test_attention_repeat_interleave_and_scale():
    torch.manual_seed(6)
    c, y, ref = run(AttnMod(gqa="repeat", scale=0.2).eval(), *TL.attn_inputs("causal", T=7))
    assert "attn" in region_ops()
    assert nmse(y, ref) < NMSE_ATTN, nmse(y, ref)


@pytest.mark.parametrize("heads,kv,gqa", [(4, 1, "expand"), (4, 1, "repeat"), (2, 1, "expand"), (1, 1, "expand")])
def test_one_kv_head_and_one_head_attention_take_their_projections_into_the_region(heads, kv, gqa):
    """[1, T, 1, D] is one head: the projections of K and V (one KV head) and of Q, K and V (one head) are
    in the attention region, not lone matmuls."""
    torch.manual_seed(6)
    x = torch.randn(1, 7, heads * 64)
    c, y, ref = run(AttnMod(heads=heads, kv=kv, gqa=gqa).eval(), x)
    assert region_ops() == ["mul_mat", "mul_mat", "mul_mat", "attn", "mul_mat"]
    assert nmse(y, ref) < NMSE_ATTN, nmse(y, ref)


def test_attention_is_causal_not_a_full_mask():
    """The mask is in effect: a model without one gives a different result than the causal one."""
    torch.manual_seed(6)
    m = AttnMod(mask="causal").eval()
    x = TL.attn_inputs("causal", T=6)[0]
    c = full(m)
    with torch.no_grad():
        y = c(x)
        ref_causal = m(x)
        m.mask = "none"
        ref_full = m(x)
    assert nmse(y, ref_causal) < NMSE_ATTN < nmse(y, ref_full)


def _blocks_model(fmt, K, N):
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = tt.QuantLinear(torch.from_numpy(sm_blocks(fmt, N, K, 1)), K, fmt)
            self.b = tt.QuantLinear(torch.from_numpy(sm_blocks(fmt, N, N, 2)), N, fmt)

        def forward(self, x):
            return self.b(F.silu(self.a(x)))

    return M().eval()


def sm_blocks(fmt, N, K, seed):
    import test_quant
    return test_quant._blocks(fmt, N, K, seed)


@pytest.mark.parametrize("fmt,K,N", [("q8_0", 64, 64), ("q4_k", 256, 256)])
def test_quantized_blocks_weights_in_a_region(fmt, K, N):
    m = _blocks_model(fmt, K, N)
    x = torch.randn(1, 3, K) * 0.5
    c = full(m)
    with torch.no_grad():
        y = c(x)
    assert region_ops() == ["mul_mat", "unary", "mul_mat"]
    assert [w.kind for w in regs()[-1].region.weights] == ["blocks", "blocks"]
    assert (cnt()["region_calls"], cnt()["rt_graph_launches"], cnt()["device_calls"]) == (1, 1, 0)
    with torch.no_grad():
        ref = m(x)
    assert nmse(y, ref) < 2e-3, nmse(y, ref)


# ------------------------------------------------------------------------------------------
# the decoder: one region, one launch
# ------------------------------------------------------------------------------------------
def test_decoder_is_one_region_and_one_launch():
    torch.manual_seed(7)
    m = QwenLike().eval()
    x = ids(6)
    c = full(m)
    with torch.no_grad():
        y = c(*x)
        ref = m(*x)
    c1 = cnt()
    assert (c1["regions"], c1["region_calls"], c1["region_tokens"], c1["region_cpu_fallback_calls"]) == (1, 1, 6, 0)
    assert (c1["rt_graphs"], c1["rt_graph_runs"], c1["rt_graph_launches"]) == (1, 1, 1)
    assert (c1["verified_regions"], c1["device_calls"], c1["nodes_device"]) == (1, 0, 0)
    assert c1["uploads"] == 15 and c1["consts_uploaded"] == 12 and c1["rt_consts"] == 12
    assert c1["rt_graph_plan_misses"] == 1
    assert regs()[0].region.op_counts() == {"rms_norm": 5, "mul_mat": 15, "rope": 4, "attn": 2, "glu": 2}
    # the records of the one list: every kind of the two layers and the head, no more
    rg = regs()[0]
    info = tt.device().graph_info(rg.graph, 6)
    by = info["by_kind"]
    assert (info["lists"], by["RMS_NORM"], by["ROPE"], by["SET_ROWS_KV"], by["FLASH_ATTN"], by["GLU"]) == (1, 5, 4, 4, 2, 2)
    assert by.get("MUL_MAT", 0) + by.get("MUL_MAT_BEGIN", 0) == 15
    assert c1["rt_graph_records"] == info["records"]
    assert nmse(y, ref) < NMSE_ATTN, nmse(y, ref)
    assert (y.argmax(-1) == ref.argmax(-1)).float().mean() >= 0.5
    # a second call with the same T: no upload, a plan hit, one more launch
    with torch.no_grad():
        y2 = c(*x)
    c2 = cnt()
    assert torch.equal(y2.view(torch.int32), y.view(torch.int32))
    assert (c2["region_calls"], c2["rt_graph_launches"], c2["uploads"], c2["consts_uploaded"], c2["rt_graphs"]) == (2, 2, 15, 12, 1)
    assert c2["rt_graph_plan_hits"] > c1["rt_graph_plan_hits"] and c2["rt_graph_plan_misses"] == 1
    assert c2["rt_graph_records"] == 2 * info["records"]


def test_the_cpu_nodes_in_front_of_the_region_are_listed():
    torch.manual_seed(7)
    m = QwenLike().eval()
    with torch.no_grad():
        full(m)(*ids(6))
    r = regs()[0].region
    assert r.inputs[0].node.target is torch.ops.aten.embedding.default      # the embedding stays on the CPU
    assert {i.kind for i in r.inputs} == {"in", "pos", "mask"}


@pytest.mark.parametrize("mask", ["where", "bool"])
def test_decoder_mask_forms(mask):
    torch.manual_seed(7)
    m = QwenLike(mask=mask).eval()
    x = ids(5)
    c, y, ref = run(m, *x)
    assert cnt()["region_calls"] == 1 and cnt()["rt_graph_launches"] == 1
    assert nmse(y, ref) < NMSE_ATTN


def test_decoder_with_eps_and_a_changed_value():
    """Dynamo lifts a changed Python float to a graph input: the region reads it at every call."""
    torch.manual_seed(7)
    m = QwenLike(eps=1e-5).eval()
    c = full(m)
    x = ids(6)
    with torch.no_grad():
        y = c(*x)
        m.norm.eps = 3e-2            # a larger eps: the head's norm changes visibly
        y2 = c(*x)
        ref2 = m(*x)
        m.norm.eps = 4.0             # now a graph input of the same compiled graph: its value is read at every call
        y3 = c(*x)
        ref3 = m(*x)
    assert nmse(y2, ref2) < NMSE_ATTN and nmse(y3, ref3) < NMSE_ATTN
    assert nmse(y, y2) > 1e-6 and nmse(y2, y3) > 1e-6
    assert cnt()["regions"] == 2 and cnt()["region_cpu_fallback_calls"] == 0
    assert cnt()["rt_graphs"] == 3                    # the first graph, the second, and the second again for the new eps


# ------------------------------------------------------------------------------------------
# the Block of test_backend.py: LayerNorm on the CPU between regions
# ------------------------------------------------------------------------------------------
def test_block_with_layernorm_between_the_regions():
    """batch 2: the attention stays on the CPU, the MLP is one region; q, k, v, o are device_mm nodes."""
    torch.manual_seed(8)
    m = Block().eval()
    x = torch.randn(2, 12, 128)
    c, y, ref = run(m, x)
    k = cnt()
    assert region_ops() == ["mul_mat", "mul_mat", "glu", "mul_mat"]
    assert (k["regions"], k["region_calls"], k["rt_graph_launches"]) == (1, 1, 1)
    assert (k["nodes_device"], k["device_calls"]) == (4, 4)            # q, k, v, o
    assert nmse(y, ref) < NMSE_ATTN, nmse(y, ref)


class Block256(Block):
    def __init__(self):
        super().__init__(d=256, heads=4, ff=352)


def test_block_with_batch_one_lowers_attention_and_has_two_regions():
    torch.manual_seed(8)
    m = Block256().eval()
    x = torch.randn(1, 12, 256)
    c, y, ref = run(m, x)
    k = cnt()
    assert [[o["op"] for o in r.region.ops] for r in regs()] == [
        ["mul_mat", "mul_mat", "mul_mat", "attn", "mul_mat"], ["mul_mat", "mul_mat", "glu", "mul_mat"]]
    assert (k["regions"], k["region_calls"], k["rt_graph_launches"], k["device_calls"]) == (2, 2, 2, 0)
    assert k["verified_regions"] == 2
    assert nmse(y, ref) < NMSE_ATTN, nmse(y, ref)
    with torch.no_grad():
        c(x)
    assert (cnt()["region_calls"], cnt()["rt_graph_launches"]) == (4, 4)


# ------------------------------------------------------------------------------------------
# shapes: static, then dynamic after a second sequence length; dynamic=True
# ------------------------------------------------------------------------------------------
def test_changing_sequence_length_recompiles_with_symbolic_shapes():
    torch.manual_seed(9)
    m = QwenLike().eval()
    c = full(m)
    with torch.no_grad():
        for T in (6, 9, 6, 13, 2):
            x = ids(T)
            y = c(*x)
            assert y.shape == (1, T, 256)
            assert nmse(y, m(*x)) < NMSE_ATTN, T
    k = cnt()
    assert k["regions"] == 2                                  # the static graph, then the dynamic one
    assert (k["region_calls"], k["rt_graph_launches"], k["region_tokens"]) == (5, 5, 6 + 9 + 6 + 13 + 2)
    assert (k["uploads"], k["consts_uploaded"]) == (15, 12)   # the second graph shares every weight and constant


def test_dynamic_true_from_the_first_compile():
    torch.manual_seed(9)
    m = QwenLike().eval()
    c = full(m, dynamic=True)
    with torch.no_grad():
        for T in (7, 4, 11):
            x = ids(T)
            assert nmse(c(*x), m(*x)) < NMSE_ATTN, T
    k = cnt()
    assert (k["regions"], k["region_calls"], k["rt_graph_launches"], k["region_cpu_fallback_calls"]) == (1, 3, 3, 0)


def test_input_mutations_survive_the_rewrite():
    """AOT keeps a buffer update as a copy_ at the end of the graph; the region does not touch it."""
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = Lin(Norm(TL.rms_ok))
            self.register_buffer("steps", torch.zeros(1))

        def forward(self, x):
            self.steps.add_(1)
            return self.body(x) * self.steps

    torch.manual_seed(10)
    m = M().eval()
    ref_m = M().eval()
    ref_m.load_state_dict(m.state_dict())
    c = full(m)
    with torch.no_grad():
        for _ in range(3):
            y = c(X64)
            ref = ref_m(X64)
    assert float(m.steps) == 3.0
    assert cnt()["region_calls"] == 3
    assert nmse(y, ref) < NMSE_FREE


# ------------------------------------------------------------------------------------------
# weights and constants changed in place
# ------------------------------------------------------------------------------------------
def test_a_weight_changed_in_place_is_uploaded_again():
    torch.manual_seed(11)
    m = QwenLike().eval()
    c = full(m)
    x = ids(6)
    with torch.no_grad():
        y0 = c(*x)
        old = regs()[0].graph
        m.layers[0].gate.weight.mul_(1.7)
        y1 = c(*x)
        ref1 = m(*x)
    with pytest.raises(ValueError):
        tt.device().graph_shape(old)                   # the graph of the old handle is freed
    k = cnt()
    assert k["reuploads"] == 1 and k["uploads"] == 16 and k["verified_regions"] == 2
    assert nmse(y1, ref1) < NMSE_ATTN
    assert nmse(y0, y1) > 1e-4                         # the result moved with the weight
    assert k["rt_graphs"] == 2 and k["rt_consts"] == 12   # a new graph for the new handle; the old one is freed


def test_a_norm_weight_and_a_bias_changed_in_place_are_uploaded_again():
    torch.manual_seed(11)
    m = QwenLike().eval()
    c = full(m)
    x = ids(6)
    with torch.no_grad():
        y0 = c(*x)
        m.norm.weight.mul_(0.4)
        m.layers[1].q.bias.add_(0.5)
        y1 = c(*x)
        ref1 = m(*x)
    k = cnt()
    assert k["const_reuploads"] == 2 and k["consts_uploaded"] == 14
    assert nmse(y1, ref1) < NMSE_ATTN
    assert nmse(y0, y1) > 1e-4
    with torch.no_grad():
        c(*x)
    assert cnt()["consts_uploaded"] == 14              # unchanged: no upload on the third call


def test_release_weights_frees_graphs_and_constants_and_they_come_back():
    torch.manual_seed(11)
    m = QwenLike().eval()
    c = full(m)
    x = ids(6)
    with torch.no_grad():
        y0 = c(*x)
        free_before = tt.device().memory()
        tt.release_weights()
        assert tt.counters()["cached_consts"] == 0 and tt.counters()["cached_weights"] == 0
        assert regs()[0].graph is None
        y1 = c(*x)
    assert torch.equal(y0.view(torch.int32), y1.view(torch.int32))
    assert free_before is not None and cnt()["rt_graphs"] == 2


def test_an_input_computed_after_the_first_user_of_an_output():
    """The call goes before the first user of an output; a region input computed later moves up."""
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(D64, D64, bias=False)
            self.b = nn.Linear(D64, D64, bias=False)

        def forward(self, x):
            a = self.a(x)
            t = torch.cumsum(a, 1)           # CPU, uses the region's first value
            c = torch.cumsum(x, 1)           # CPU, computed after t, the addend of the second matmul
            return t, self.b(a) + c

    torch.manual_seed(18)
    m = M().eval()
    c, (t, y), (rt, ry) = run(m, X64)
    assert region_ops() == ["mul_mat", "mul_mat"] and cnt()["region_calls"] == 1
    assert nmse(t, rt) < NMSE_FREE and nmse(y, ry) < NMSE_FREE


def test_the_rewrite_leaves_a_clean_graph():
    """What a region replaced is gone (the norms, the cos / sin tables, the mask arithmetic, the matmuls);
    the graph lints and has one device_region call."""
    torch.manual_seed(19)
    m = QwenLike().eval()
    gm, w = TL.capture(m, [(ids(6),)])
    rd = B.partition_regions(gm, w, tt.Options(), 0)
    assert len(rd) == 1
    gm.graph.lint()
    aten = torch.ops.aten
    targets = [n.target for n in gm.graph.nodes if n.op == "call_function"]
    assert targets.count(B.device_region) == 1
    gone = {aten.rsqrt.default, aten.pow.Tensor_Scalar, aten.mean.dim, aten.cos.default, aten.sin.default, aten.where.self,
            aten.where.ScalarSelf, aten.mm.default, aten.addmm.default, aten.silu.default, aten.t.default,
            aten._scaled_dot_product_flash_attention_for_cpu.default}
    assert not gone & set(targets), gone & set(targets)
    assert rd[0].node.op == "call_function" and rd[0].rt.fallback is not None


# ------------------------------------------------------------------------------------------
# D8: only placement falls back to the CPU
# ------------------------------------------------------------------------------------------
def test_t_above_kv_size_runs_the_region_on_the_cpu():
    torch.manual_seed(12)
    m = QwenLike().eval()
    c = full(m, kv_size=8)
    with torch.no_grad(), warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        x6 = ids(6)
        y6 = c(*x6)
        long_x = ids(12)
        y12 = c(*long_x)
        y12b = c(*long_x)
        y6b = c(*x6)
    k = cnt()
    assert (k["region_calls"], k["region_cpu_fallback_calls"], k["rt_graph_launches"]) == (2, 2, 2)
    runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning) and "does not fit" in str(x.message)]
    assert len(runtime_warnings) == 1, [str(x.message) for x in w]                # one per region
    assert sum(r.cpu_calls for r in regs()) == 2
    assert nmse(y12, m(*long_x)) < 1e-9 and torch.equal(y12, y12b)                # the aten subgraph itself, f32
    with torch.no_grad():
        assert nmse(y6, m(*x6)) < NMSE_ATTN and torch.equal(y6, y6b)             # T = 6 fits: back on the device


def test_the_cpu_fallback_of_a_head_major_output(monkeypatch):
    """The saved subgraph returns the original shape; the region hands back [T, n] rows and the graph
    rebuilds the shape: the head-major result of a RoPE (transpose) must come out right."""
    torch.manual_seed(12)
    m = RopeMod().eval()
    x, pos = TL.rope_inputs(5)

    def no_fit(*a):
        raise RT.TesseractError("T = 5 does not fit the test")

    monkeypatch.setattr(RT, "_device_run", no_fit)
    c = full(m)
    with torch.no_grad(), pytest.warns(RuntimeWarning, match="does not fit"):
        y = c(x, pos)
        ref = m(x, pos)
    assert y.shape == ref.shape == (1, 2, 5, 64) and nmse(y, ref) < 1e-12
    assert cnt()["region_cpu_fallback_calls"] == 1 and cnt()["region_calls"] == 0


def test_the_cpu_fallback_computes_what_eager_computes():
    torch.manual_seed(12)
    m = QwenLike().eval()
    c = full(m, kv_size=4)
    x = ids(9)
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y = c(*x)
        ref = m(*x)
    assert nmse(y, ref) < 1e-10
    assert cnt()["region_calls"] == 0 and cnt()["region_cpu_fallback_calls"] == 1


def test_a_full_weight_heap_runs_the_region_on_the_cpu(monkeypatch):
    torch.manual_seed(12)
    m = QwenLike().eval()
    c = full(m)

    def heap_full(w, name=""):
        raise RT.TesseractError("tsr_rt: WEIGHT heap full: a q8_0 weight does not fit")

    monkeypatch.setattr(tt.weight_cache(), "acquire", heap_full)
    x = ids(6)
    with torch.no_grad(), pytest.warns(RuntimeWarning, match="heap full"):
        y = c(*x)
    assert nmse(y, m(*x)) < 1e-10 and cnt()["region_cpu_fallback_calls"] == 1


def test_other_errors_are_raised(monkeypatch):
    torch.manual_seed(12)
    m = QwenLike().eval()
    c = full(m)

    def broken(w, name=""):
        raise RT.TesseractError("tsr_rt: the transport failed")

    monkeypatch.setattr(tt.weight_cache(), "acquire", broken)
    with torch.no_grad(), pytest.raises(tt.TesseractError, match="transport failed"):
        c(*ids(6))
    assert cnt()["region_cpu_fallback_calls"] == 0


# ------------------------------------------------------------------------------------------
# modes
# ------------------------------------------------------------------------------------------
def test_matmul_mode_is_todays_behaviour_bit_for_bit():
    torch.manual_seed(13)
    m = Block().eval()
    x = torch.randn(2, 12, 128)
    c = tt.compile(m, mode="matmul")
    with torch.no_grad():
        y = c(x)
        with tt.reference_linears():
            ref = m(x)
    assert torch.equal(y.view(torch.int32), ref.view(torch.int32))
    k = cnt()
    assert (k["regions"], k["region_calls"], k["rt_graph_runs"], k["consts_uploaded"]) == (0, 0, 0, 0)
    assert (k["nodes_device"], k["device_calls"]) == (7, 7)
    assert not regs()


def test_matmul_mode_leaves_the_graph_without_region_nodes():
    m = QwenLike().eval()
    c = tt.compile(m, mode="matmul")
    with torch.no_grad():
        c(*ids(6))
    assert cnt()["regions"] == 0 and cnt()["device_calls"] == 15


def test_mode_through_options_and_the_environment(monkeypatch):
    m = QwenLike().eval()
    x = ids(6)
    with torch.no_grad():
        torch.compile(m, backend="tesseract", options={"mode": "matmul"})(*x)
    assert cnt()["regions"] == 0 and cnt()["device_calls"] == 15
    torch._dynamo.reset()
    tt.reset_counters()
    monkeypatch.setenv("TSR_TORCH_MODE", "matmul")
    with torch.no_grad():
        torch.compile(m, backend="tesseract")(*x)
        assert cnt()["regions"] == 0
        torch._dynamo.reset()
        torch.compile(m, backend="tesseract", options={"mode": "full"})(*x)      # the option beats the variable
    assert cnt()["regions"] == 1
    torch._dynamo.reset()
    tt.reset_counters()
    monkeypatch.delenv("TSR_TORCH_MODE")
    with torch.no_grad():
        torch.compile(m, backend="tesseract")(*x)                                # the default is full
    assert cnt()["regions"] == 1 and cnt()["region_calls"] == 1


def test_unknown_mode_is_an_error(monkeypatch):
    with pytest.raises(ValueError, match="unknown mode"):
        tt.compile(QwenLike().eval(), mode="fast")
    with pytest.raises(ValueError, match="unknown mode"):
        tt.Options.from_dict({"mode": "fast"})
    monkeypatch.setenv("TSR_TORCH_MODE", "fast")
    with pytest.raises(ValueError, match="unknown mode"):
        B.resolve_mode(None)


def test_offload_false_forms_no_region():
    m = QwenLike().eval()
    with torch.no_grad():
        full(m, offload=False)(*ids(6))
    assert cnt()["regions"] == 0 and cnt()["device_calls"] == 0


def test_bf16_gets_matmul_nodes_only():
    torch.manual_seed(14)
    m = Block().eval().to(torch.bfloat16)
    x = torch.randn(1, 12, 128).to(torch.bfloat16)
    c = full(m)
    with torch.no_grad():
        y = c(x)
        ref = m(x)
    k = cnt()
    assert (k["regions"], k["region_calls"]) == (0, 0) and k["device_calls"] == 7 and y.dtype == torch.bfloat16
    assert nmse(y.float(), ref.float()) < 2e-2


# ------------------------------------------------------------------------------------------
# negative controls: a wrong program must not pass
# ------------------------------------------------------------------------------------------
def _program_edit(monkeypatch, edit):
    orig = LW.Region.to_program

    def to_program(self, scalar_values=None):
        p = orig(self, scalar_values)
        edit(p)
        return p

    monkeypatch.setattr(LW.Region, "to_program", to_program)


def test_a_wrong_eps_fails_the_nmse_bound(monkeypatch):
    torch.manual_seed(15)
    m = QwenLike().eval()
    x = ids(6)
    _, ok, ref = run(m, *x)
    assert nmse(ok, ref) < NMSE_ATTN
    torch._dynamo.reset()
    tt.release_weights()

    def bad_eps(p):
        for o in p["ops"]:
            if o["op"] == "rms_norm":
                o["eps"] = LW.f32(5.0)
    _program_edit(monkeypatch, bad_eps)
    c = full(m)
    with torch.no_grad():
        y = c(*x)                                 # device == reference executor (both run the wrong program) ...
    assert cnt()["verified_regions"] >= 1
    assert nmse(y, ref) > 10 * NMSE_ATTN          # ... but it is not the model


def test_q_and_k_swapped_fail_the_nmse_bound(monkeypatch):
    torch.manual_seed(15)
    m = AttnMod().eval()
    args = TL.attn_inputs("causal", T=6)
    with torch.no_grad():
        ref = m(*args)

    def swap(p):
        for o in p["ops"]:
            if o["op"] == "attn":
                # q has n_head * d columns and k n_head_kv * d: swap the roles through two same-size values
                o["q"], o["k"] = o["k"], o["q"]
    # q [T, 256] and k [T, 128] differ in size: the device refuses the program (verify of the program, not a bit check)
    _program_edit(monkeypatch, swap)
    c = full(m)
    with torch.no_grad(), pytest.raises((tt.TesseractError, ValueError)):
        c(*args)


def test_q_and_k_swapped_in_a_model_with_equal_heads(monkeypatch):
    torch.manual_seed(15)
    m = AttnMod(heads=4, kv=4).eval()
    args = TL.attn_inputs("causal", T=6)
    with torch.no_grad():
        ref = m(*args)

    def swap(p):
        for o in p["ops"]:
            if o["op"] == "attn":
                o["q"], o["k"] = o["k"], o["q"]
    _program_edit(monkeypatch, swap)
    c = full(m)
    with torch.no_grad():
        y = c(*args)
    assert cnt()["verified_regions"] == 1
    assert nmse(y, ref) > 5 * NMSE_ATTN


def test_a_device_result_that_differs_from_the_reference_raises(monkeypatch):
    """The verify step compares every output bit for bit and names the region, the output and the element."""
    torch.manual_seed(15)
    m = QwenLike().eval()
    c = full(m)
    real = RT.device()

    class Flip:
        def __getattr__(self, name):
            return getattr(real, name)

        def graph_run(self, h, inputs):
            outs = real.graph_run(h, inputs)
            v = outs[0].view(torch.int32)
            v[2, 7] ^= 1                           # one bit of one element
            return outs

    monkeypatch.setattr(RT, "device", lambda: Flip())
    with torch.no_grad(), pytest.raises(tt.TesseractError, match=r"region \d+\.0 .* output 0 .*first y\[2, 7\]"):
        c(*ids(6))


# ------------------------------------------------------------------------------------------
# threads
# ------------------------------------------------------------------------------------------
def test_a_graph_called_from_two_threads():
    torch.manual_seed(16)
    m = QwenLike().eval()
    c = full(m)
    xs = [ids(6) for _ in range(4)]
    with torch.no_grad():
        want = [c(*x) for x in xs]                 # compiles and makes the graph
        got: list = [None] * 4
        errs: list = []

        def work(i):
            try:
                with torch.no_grad():
                    for _ in range(2):
                        got[i] = c(*xs[i])
            except Exception as e:  # noqa: BLE001
                errs.append(e)

        ts = [threading.Thread(target=work, args=(i,)) for i in range(4)]
        [t.start() for t in ts]
        [t.join() for t in ts]
    assert not errs, errs
    for g, w in zip(got, want):
        assert torch.equal(g.view(torch.int32), w.view(torch.int32))
    k = cnt()
    assert (k["region_calls"], k["rt_graph_launches"], k["rt_graphs"]) == (4 + 8, 4 + 8, 1)


# ------------------------------------------------------------------------------------------
# debug report and counters
# ------------------------------------------------------------------------------------------
def test_debug_report_names_regions_and_the_cpu_nodes_beside_them(capfd):
    torch.manual_seed(17)
    m = QwenLike().eval()
    c = full(m, debug=1)
    with torch.no_grad():
        c(*ids(6))
    err = capfd.readouterr().err
    assert "1 region(s)" in err and "region " in err and "rms_norm x5" in err
    assert "cpu beside the region: embedding" in err
    assert "first plan T=6" in err and "1 list(s)" in err and "FLASH_ATTN x2" in err


def test_counters_keys():
    k = tt.counters()
    for key in ("regions", "region_calls", "region_tokens", "region_cpu_fallback_calls", "consts_uploaded",
                "verified_regions", "rt_graphs", "rt_graph_runs", "rt_graph_launches", "rt_graph_records", "rt_consts"):
        assert key in k, key


# ------------------------------------------------------------------------------------------
# review fixes: a region is a correct lowering or a refusal, never a silent wrong result
# ------------------------------------------------------------------------------------------
class Body(nn.Module):
    """a -> body(self, a(x)) -> b: the body sits between two matmuls."""

    def __init__(self, body, d=64, bias=False):
        super().__init__()
        self.a = nn.Linear(d, d, bias=bias)
        self.b = nn.Linear(d, d, bias=bias)
        self.w = nn.Parameter(torch.randn(d) * 3)
        self.register_buffer("s0", torch.tensor(2.5))
        self.body = body

    def forward(self, x):
        return self.b(self.body(self, self.a(x)))


def _rms(x, e=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + e)


def _check_body(body, bound=NMSE_FREE, x=None, bias=False):
    torch.manual_seed(0)
    m = Body(body, bias=bias).eval()
    x = torch.randn(1, 5, 64) if x is None else x
    c, y, ref = run(m, x)
    assert nmse(y, ref) < bound
    return c


@pytest.mark.parametrize("conv", ["long", "bool", "int", "bf16", "half"])
@pytest.mark.parametrize("use", ["mul", "add", "rms"])
def test_a_narrowed_and_widened_constant_is_not_uploaded_unrounded(conv, use):
    def row(s):
        w = s.w
        return {"long": w.long(), "bool": w.bool(), "int": w.int(),
                "bf16": w.to(torch.bfloat16), "half": w.half()}[conv].float()

    body = {"mul": lambda s, x: torch.tanh(x) * row(s),
            "add": lambda s, x: torch.tanh(x) + row(s),
            "rms": lambda s, x: row(s) * _rms(x)}[use]
    _check_body(body)


def test_a_narrowed_bias_is_not_uploaded_unrounded():
    torch.manual_seed(0)
    m = Body(lambda s, x: torch.tanh(x)).eval()
    m.a.bias = nn.Parameter(torch.randn(64) * 30)
    m.forward = lambda x: m.b(torch.tanh(F.linear(x, m.a.weight, m.a.bias.long().float())))
    c, y, ref = run(m, torch.randn(1, 5, 64))
    assert nmse(y, ref) < NMSE_FREE


def test_rms_eps_add_with_alpha_is_not_taken_as_eps():
    _check_body(lambda s, x: s.w * (x * torch.rsqrt(torch.add(x.pow(2).mean(-1, keepdim=True), 1e-6, alpha=1e4))),
                bound=NMSE_FREE)


@pytest.mark.parametrize("form", ["reshape", "contiguous_view", "flatten"])
def test_a_head_major_value_through_a_copy_before_a_cpu_view(form):
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(64, 64)
            self.b = nn.Linear(64, 64)

        def forward(self, x):
            t = torch.tanh(self.a(x)).view(1, 5, 2, 32).transpose(1, 2) * 2.0
            hm = {"reshape": t.reshape(1, 2, 160).sum(1),
                  "contiguous_view": t.contiguous().view(1, 5, 64).cumsum(1),
                  "flatten": t.flatten(2).sum(1)}[form]
            return self.b(torch.tanh(self.a(x))), hm

    torch.manual_seed(0)
    m = M().eval()
    x = torch.randn(1, 5, 64)
    c = full(m)
    with torch.no_grad():
        y, ref = c(x), m(x)
    for a, b in zip(y, ref):
        assert a.shape == b.shape and nmse(a, b) < NMSE_FREE


def test_head_major_output_cloned_and_viewed_on_the_cpu():
    """the review's case: q = rope(q) is a region output, and q.reshape(B, H*T, D) copies it (clone + view)."""
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(64, 64)
            self.b = nn.Linear(64, 64)

        def forward(self, x):
            h = self.a(x)
            q = (torch.tanh(h).view(1, 5, 2, 32).transpose(1, 2)) * 2.0
            return self.b(q.transpose(1, 2).reshape(1, 5, 64)), q.reshape(1, 10, 32).sum(1)

    torch.manual_seed(0)
    m = M().eval()
    x = torch.randn(1, 5, 64)
    c = full(m)
    with torch.no_grad():
        y, ref = c(x), m(x)
    for a, b in zip(y, ref):
        assert nmse(a, b) < NMSE_FREE


@pytest.mark.parametrize("cpu", ["layer_norm", "flip", "roll", "erf_gelu", "exp"])
def test_a_cpu_node_between_two_reads_of_a_view_keeps_the_regions_convex(cpu):
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc, self.fc2, self.fc3 = nn.Linear(64, 64), nn.Linear(64, 64), nn.Linear(64, 64)
            self.ln = nn.LayerNorm(64)

        def forward(self, x):
            h = self.fc(x)
            o = {"layer_norm": self.ln(h), "flip": h.flip(-1), "roll": torch.roll(h, 1, 1),
                 "erf_gelu": F.gelu(h), "exp": torch.exp(-h)}[cpu]
            y = h * o if cpu in ("erf_gelu", "exp") else h + o
            return self.fc3(torch.tanh(self.fc2(y)))

    torch.manual_seed(0)
    m = M().eval()
    x = torch.randn(1, 5, 64)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        c = tt.compile(m, mode="full", strict=True)
        with torch.no_grad():
            y, ref = c(x), m(x)
    assert nmse(y, ref) < NMSE_FREE
    assert cnt()["region_calls"] >= 1
    assert all(r.calls > 0 for r in regs()) if hasattr(regs()[0], "calls") else True


def test_regions_of_a_dead_model_are_released():
    import gc

    def once(T):
        torch.manual_seed(0)
        m = AttnMod().eval()
        c = full(m)
        with torch.no_grad():
            c(*TL.attn_inputs("causal", T=T))
        assert regs()[-1].graph is not None

    for T in (6, 9, 1):
        once(T)
    torch._dynamo.reset()
    gc.collect()
    assert len(regs()) >= 3 and all(r.graph is None for r in regs())


class _Attn(nn.Module):
    def __init__(self, gqa="interleave", mask="where_inv"):
        super().__init__()
        self.gqa, self.mask = gqa, mask
        self.q, self.k, self.v = nn.Linear(256, 256, bias=False), nn.Linear(256, 128, bias=False), nn.Linear(256, 128, bias=False)
        self.o = nn.Linear(256, 256, bias=False)

    def forward(self, x, m):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, 4, 64).transpose(1, 2)
        k = self.k(x).view(B, T, 2, 64).transpose(1, 2)
        v = self.v(x).view(B, T, 2, 64).transpose(1, 2)
        if self.gqa == "interleave":
            k, v = k.repeat_interleave(2, 1), v.repeat_interleave(2, 1)
        else:
            k = k.unsqueeze(1).expand(B, 2, 2, T, 64).reshape(B, 4, T, 64)
            v = v.unsqueeze(1).expand(B, 2, 2, T, 64).reshape(B, 4, T, 64)
        am = torch.where(m, float("-inf"), 0.0)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=am)
        return self.o(a.transpose(1, 2).reshape(B, T, 256))


@pytest.mark.parametrize("gqa", ["interleave", "tile"])
def test_where_with_the_true_entries_masked_and_tiled_gqa(gqa):
    torch.manual_seed(0)
    m = _Attn(gqa).eval()
    x = torch.randn(1, 6, 256)
    masked = torch.ones(1, 1, 6, 6, dtype=torch.bool).triu(1)
    c = full(m)
    with torch.no_grad():
        y, ref = c(x, masked), m(x, masked)
    assert nmse(y, ref) < NMSE_ATTN


@pytest.mark.parametrize("op", ["rsub", "div"])
def test_a_zero_dim_buffer_as_the_left_operand(op):
    body = {"rsub": lambda s, x: s.s0 - torch.tanh(x),
            "div": lambda s, x: s.s0 / (torch.tanh(x).abs() + 1.0)}[op]
    _check_body(body)


def _unary(n, tag):
    return {"values": [{"kind": "in", "n": n}, {"kind": "tmp", "n": n, "out": True}],
            "ops": [{"op": "unary", "x": 0, "func": 0, "dst": 1, "tag": tag}], "kv_size": 4096}


def test_graph_unsupported_malformed_programs_are_value_errors():
    d = tt.device()
    p = {"values": [{"kind": "const", "n": 64, "rows": 1, "handle": 987654}, {"kind": "in", "n": 64},
                    {"kind": "tmp", "n": 64, "out": True}],
         "ops": [{"op": "add", "a": 1, "b": 0, "dst": 2, "tag": 3}], "kv_size": 4096}
    with pytest.raises(ValueError, match="unknown const handle"):
        d.graph_unsupported(p)
    with pytest.raises(ValueError, match="tag"):
        d.graph_unsupported(_unary(64, 2 ** 31))
    with pytest.raises(ValueError, match="tag"):
        d.graph_unsupported(_unary(64, 2 ** 70))


def test_graph_unsupported_answers_are_per_thread():
    d = tt.device()
    # a program the device refuses (a row of 2^23 values): the reason names the tag
    want = {t: d.graph_unsupported(_unary(1 << 23, t)) for t in (11, 22)}
    assert all(w and f"tag {t}" in w for t, w in want.items())
    bad = []

    def worker(t):
        for _ in range(300):
            if d.graph_unsupported(_unary(1 << 23, t)) != want[t]:
                bad.append(t)

    ths = [threading.Thread(target=worker, args=(t,)) for t in want]
    [t.start() for t in ths]
    [t.join() for t in ths]
    assert not bad


def test_a_kv_size_whose_planes_cannot_be_addressed_is_not_run_on_the_device():
    torch.manual_seed(0)
    m = AttnMod().eval()
    x = TL.attn_inputs("causal", T=6)
    c = tt.compile(m, mode="full", kv_size=1 << 26)
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, ref = c(*x), m(*x)
    assert nmse(y, ref) < NMSE_ATTN


# ------------------------------------------------------------------------------------------
# the planner against a placement-free oracle: a program run as ONE graph equals the same ops run one by one
# as single-op graphs (graph_run against graph_run_ref shares the planner's blobs and cannot see a placement bug)
# ------------------------------------------------------------------------------------------
def _layers_program(dev, hid, nh, nkv, ff, layers, kv_size, mask, parts):
    D = 64
    ws, cs_ = [], []

    def up(n, k, **kw):
        ws.append(dev.upload(torch.randn(n, k) * 0.05, **kw))
        return ws[-1]

    def cst(t):
        cs_.append(dev.const_upload(t.contiguous()))
        return cs_[-1]

    tab = torch.tensor([1000000.0 ** (-2.0 * i / D) for i in range(D // 2)], dtype=torch.float32)
    vals, ops = [], []

    def val(**kw):
        vals.append(kw)
        return len(vals) - 1

    def op(**kw):
        kw["tag"] = len(ops)
        ops.append(kw)

    cur = x0 = val(kind="in", n=hid)
    pos = val(kind="pos")
    mk = val(kind="mask") if mask != "causal" else None
    for l in range(layers):
        kw = dict(max_part_rows=64, max_part_k=64) if parts and l == 0 else {}
        w = dict(q=up(nh * D, hid, **kw), k=up(nkv * D, hid), v=up(nkv * D, hid),
                 o=up(hid, nh * D, **({"max_part_k": 64} if parts else {})),
                 g=up(ff, hid), u=up(ff, hid, **({"max_part_rows": 128} if parts else {})), d=up(hid, ff))
        cs = {k: val(kind="const", n=n, handle=cst(t)) for k, n, t in (
            ("an", hid, 1 + 0.2 * torch.randn(hid)), ("fn", hid, 1 + 0.2 * torch.randn(hid)),
            ("bq", nh * D, 0.3 * torch.randn(nh * D)), ("bk", nkv * D, 0.3 * torch.randn(nkv * D)),
            ("bv", nkv * D, 0.3 * torch.randn(nkv * D)), ("tab", D // 2, tab))}
        t = lambda n, **kw: val(kind="tmp", n=n, **kw)                       # noqa: E731
        xn, q, k, v, qr, kr, att, hh, hn = (t(hid), t(nh * D), t(nkv * D), t(nkv * D), t(nh * D), t(nkv * D),
                                            t(nh * D), t(hid), t(hid))
        g, u, gl, s = t(ff), t(ff), t(ff), t(hid)
        y = t(hid, out=(l == layers - 1 or l == 0))
        op(op="rms_norm", x=cur, dst=xn, weight=cs["an"], eps=1e-6)
        op(op="mul_mat", w=w["q"], x=xn, dst=q, addend=cs["bq"])
        op(op="mul_mat", w=w["k"], x=xn, dst=k, addend=cs["bk"])
        op(op="mul_mat", w=w["v"], x=xn, dst=v, addend=cs["bv"])
        op(op="rope", x=q, dst=qr, n_heads=nh, head_dim=D, n_dims=D, table=cs["tab"], pos=pos, factor=1.0)
        op(op="rope", x=k, dst=kr, n_heads=nkv, head_dim=D, n_dims=D, table=cs["tab"], pos=pos, factor=1.0)
        a = dict(op="attn", q=qr, k=kr, v=v, dst=att, n_head=nh, n_head_kv=nkv, d=D, scale=0.125)
        if mk is not None:
            a["mask"] = mk
        op(**a)
        op(op="mul_mat", w=w["o"], x=att, dst=hh, addend=cur)
        op(op="rms_norm", x=hh, dst=hn, weight=cs["fn"], eps=1e-6)
        op(op="mul_mat", w=w["g"], x=hn, dst=g)
        op(op="mul_mat", w=w["u"], x=hn, dst=u)
        op(op="glu", gate=g, up=u, dst=gl, variant=0)
        op(op="mul_mat", w=w["d"], x=gl, dst=s, addend=hh)
        op(op="scale", x=s, dst=y, s=0.5, b=0.0)
        cur = y
    return {"values": vals, "ops": ops, "kv_size": kv_size}, x0, pos, mk, ws, cs_


def _single_op_graph(prog, o):
    vals, remap, args = [], {}, []

    def use(vid):
        if vid is None:
            return None
        if vid not in remap:
            src = prog["values"][vid]
            if src["kind"] == "const":
                vals.append(dict(src))
            elif src["kind"] in ("pos", "mask"):
                vals.append({"kind": src["kind"]})
                args.append(vid)
            else:
                vals.append({"kind": "in", "n": src["n"]})
                args.append(vid)
            remap[vid] = len(vals) - 1
        return remap[vid]

    oo = dict(o)
    for key in ("x", "a", "b", "addend", "weight", "gate", "up", "table", "pos", "q", "k", "v", "mask"):
        if key in oo and not (key == "b" and oo["op"] == "scale"):
            oo[key] = use(oo[key])
    vals.append({"kind": "tmp", "n": prog["values"][o["dst"]]["n"], "out": True})
    oo["dst"] = len(vals) - 1
    return {"values": vals, "ops": [oo], "kv_size": prog["kv_size"]}, args


@pytest.mark.parametrize("mask,parts,Ts,hid", [("causal", False, (5, 17), 128), ("bool", False, (9,), 128),
                                                ("causal", True, (6, 70), 128), ("causal", False, (7,), 256),
                                                ("causal", False, (7,), 64)])
def test_a_program_as_one_graph_equals_its_ops_run_one_by_one(mask, parts, Ts, hid):
    torch.manual_seed(1)
    dev = tt.device()
    nh, nkv, ff = 4, 2, 256
    prog, x0, pos, mk, ws, cs_ = _layers_program(dev, hid, nh, nkv, ff, 2, max(Ts), mask, parts)
    assert dev.graph_unsupported(prog) is None
    h = dev.graph_create(prog)
    singles = [_single_op_graph(prog, o) for o in prog["ops"]]
    sh = [dev.graph_create(s[0]) for s in singles]
    for T in Ts:
        xin, pv = torch.randn(T, hid), torch.arange(T, dtype=torch.int64) + 3
        ins = [xin, pv]
        env = {x0: xin, pos: pv}
        if mk is not None:
            env[mk] = torch.ones(T, T, dtype=torch.bool).tril()
            ins.append(env[mk])
        outs = dev.graph_run(h, ins)
        for (sp, args), hs, o in zip(singles, sh, prog["ops"]):
            env[o["dst"]] = dev.graph_run(hs, [env[a] for a in args])[0]
        outv = [i for i, v in enumerate(prog["values"]) if v.get("out")]
        assert len(outv) == len(outs) == 2
        for j, v in enumerate(outv):
            assert torch.equal(outs[j].view(torch.int32), env[v].view(torch.int32)), f"T={T} output {j}"
    for x in [h] + sh:
        dev.graph_free(x)
    for x in ws:
        dev.free(x)
    for x in cs_:
        dev.const_free(x)


# ------------------------------------------------------------------------------------------
# a program the device refuses as a whole, and large negative additive mask values
# ------------------------------------------------------------------------------------------
def _bits(y):
    return y.contiguous().view(torch.int32)


def test_a_program_refused_at_graph_create_runs_on_the_cpu(monkeypatch):
    torch.manual_seed(12)
    m = QwenLike().eval()
    c = full(m)
    real = RT.device()
    calls = []

    class Refuse:
        def __getattr__(self, name):
            return getattr(real, name)

        def graph_create(self, prog):
            calls.append(1)
            raise RT.TesseractError("tsr_rt: graph: op #3 (tag 7, mul_mat): the weight rows are not a multiple of 16")

    monkeypatch.setattr(RT, "device", lambda: Refuse())
    x = ids(6)
    with torch.no_grad(), warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        y = c(*x)
        y2 = c(*x)
        ref = m(*x)
    rw = [e for e in w if issubclass(e.category, RuntimeWarning) and "multiple of 16" in str(e.message)]
    assert len(rw) == 1, [str(e.message) for e in w]
    assert torch.equal(_bits(y), _bits(ref)) and torch.equal(_bits(y2), _bits(ref))
    k = cnt()
    assert (k["region_cpu_fallback_calls"], k["region_calls"], k["rt_graph_launches"]) == (2, 0, 0)
    assert sum(r.cpu_calls for r in regs()) == 2
    assert len(calls) == 1                         # the second call with the same key does not ask again


def test_a_malformed_program_is_still_raised(monkeypatch):
    torch.manual_seed(12)
    m = QwenLike().eval()
    c = full(m)
    real = RT.device()

    class Malformed:
        def __getattr__(self, name):
            return getattr(real, name)

        def graph_create(self, prog):
            raise ValueError("tsr_rt: graph: value 4: bad")

    monkeypatch.setattr(RT, "device", lambda: Malformed())
    with torch.no_grad(), pytest.raises(ValueError, match="value 4"):
        c(*ids(6))
    assert cnt()["region_cpu_fallback_calls"] == 0


class MaskedQwen(QwenLike):
    """QwenLike with the additive mask value `neg` in place of -inf."""

    def __init__(self, neg, **kw):
        super().__init__(**kw)
        self.neg = neg

    def forward(self, ids, use_cache=False):
        T = ids.shape[1]
        x = self.embed(ids)
        pos = torch.arange(T)[None, :]
        freqs = pos[:, :, None].float() * self.inv_freq
        emb = torch.cat([freqs, freqs], -1)
        cos, sin = emb.cos()[:, None], emb.sin()[:, None]
        allowed = torch.ones(T, T, dtype=torch.bool).tril()[None, None]
        mask = torch.where(allowed, 0.0, self.neg)
        for layer in self.layers:
            x = layer(x, cos, sin, mask)
        return F.linear(self.norm(x), self.embed.weight)


def _masked_run(neg, x):
    torch.manual_seed(21)
    m = MaskedQwen(neg).eval()
    c = full(m)
    with torch.no_grad(), warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        y = c(*x)
    return m, y, [e for e in w if issubclass(e.category, RuntimeWarning)]


@pytest.mark.parametrize("neg", [float(torch.finfo(torch.float32).min), -1e9], ids=["finfo_min", "minus_1e9"])
def test_large_negative_additive_mask_is_masked(neg):
    x = ids(6)
    _, y_inf, w0 = _masked_run(float("-inf"), x)
    assert not w0 and cnt()["region_cpu_fallback_calls"] == 0
    m, y, w = _masked_run(neg, x)
    assert not w, [str(e.message) for e in w]
    assert cnt()["region_cpu_fallback_calls"] == 0 and cnt()["rt_graph_launches"] == 2
    assert torch.equal(_bits(y), _bits(y_inf))
    with torch.no_grad():
        assert nmse(y, m(*x)) < NMSE_ATTN


def test_another_additive_mask_value_runs_on_the_cpu():
    x = ids(6)
    m, y, w = _masked_run(-5.0, x)
    assert any("the CPU" in str(e.message) for e in w), [str(e.message) for e in w]
    assert cnt()["region_cpu_fallback_calls"] == 1 and cnt()["rt_graph_launches"] == 0
    with torch.no_grad():
        assert nmse(y, m(*x)) < 1e-10


# ------------------------------------------------------------------------------------------
# in-place updates after a region, masks with no live key, scalars, outputs, two regions in a graph
# ------------------------------------------------------------------------------------------
def _no_fit(monkeypatch):
    def no_fit(*a):
        raise RT.TesseractError("the test forces a placement failure: it does not fit")

    monkeypatch.setattr(RT, "_device_run", no_fit)


def _path(monkeypatch, path):
    """The context manager of a run on the device or, for path == "cpu", with a forced placement failure."""
    if path == "cpu":
        _no_fit(monkeypatch)
        return pytest.warns(RuntimeWarning, match="does not fit")
    return warnings_as_errors()


def warnings_as_errors():
    return _AsErrors()


class _AsErrors:
    def __enter__(self):
        self.cm = warnings.catch_warnings()
        self.cm.__enter__()
        warnings.simplefilter("error")

    def __exit__(self, *a):
        return self.cm.__exit__(*a)


def _calls(path):
    r = regs()[-1]
    return r.cpu_calls if path == "cpu" else r.calls


class _StateAfterRead(nn.Module):
    """A region reads the buffer `r` (a constant row), then forward changes it in place."""

    def __init__(self):
        super().__init__()
        self.a = nn.Linear(64, 64, bias=False)
        self.b = nn.Linear(64, 64, bias=False)
        self.register_buffer("r", torch.ones(64))

    def forward(self, x):
        h = self.b(torch.tanh(self.a(x)) * self.r)
        self.r.mul_(0.5)
        return h


class _InputAfterRead(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(64, 64, bias=False)
        self.b = nn.Linear(64, 64, bias=False)

    def forward(self, x):
        h = self.b(torch.tanh(self.a(x)))
        x.mul_(2.0)
        return h


class _ScalarAfterRead(nn.Module):
    """The scale of a region is a 0-dim buffer that forward updates after the read."""

    def __init__(self):
        super().__init__()
        self.a = nn.Linear(64, 64, bias=False)
        self.b = nn.Linear(64, 64, bias=False)
        self.register_buffer("s", torch.tensor(0.5))

    def forward(self, x):
        h = self.b(self.a(x) * self.s)
        self.s.add_(1.0)
        return h


@pytest.mark.parametrize("path", ["device", "cpu"])
@pytest.mark.parametrize("cls", [_StateAfterRead, _InputAfterRead, _ScalarAfterRead])
def test_a_region_runs_before_the_forward_changes_what_it_reads(monkeypatch, cls, path):
    """AOT keeps the in-place update of an input or a buffer as a copy_ at the end of the graph; the call
    (and its CPU fallback) goes before it, so it reads the values eager reads."""
    torch.manual_seed(5)
    m = cls().eval()
    ref_m = cls().eval()
    ref_m.load_state_dict(m.state_dict())
    c = full(m)
    with _path(monkeypatch, path), torch.no_grad():
        for _ in range(2):
            x = torch.randn(1, 8, 64)
            xr = x.clone()
            y, ref = c(x), ref_m(xr)
            assert nmse(y, ref) < (1e-9 if path == "cpu" else NMSE_FREE)
            assert torch.equal(x, xr)
    for (k, a), (_, b) in zip(m.state_dict().items(), ref_m.state_dict().items()):
        assert torch.equal(a, b), k
    assert len(regs()) == 1 and _calls(path) == 2


class _Pad(nn.Module):
    """One attention layer with an additive [T, T] mask that is an argument."""

    def __init__(self):
        super().__init__()
        self.q, self.k, self.v, self.o = (nn.Linear(128, 128, bias=False) for _ in range(4))

    def forward(self, x, m):
        T = x.shape[1]

        def sp(t):
            return t.view(1, T, 2, 64).transpose(1, 2)

        a = F.scaled_dot_product_attention(sp(self.q(x)), sp(self.k(x)), sp(self.v(x)), attn_mask=m)
        return self.o(a.transpose(1, 2).reshape(1, T, 128))


def _pad_mask(T, fill, pad=2, kind="additive"):
    keep = torch.ones(T, T, dtype=torch.bool).tril() & (torch.arange(T)[None, :] >= pad)
    if kind == "bool":
        return keep[None, None]
    return torch.zeros(1, 1, T, T).masked_fill(~keep, fill)


def _pad_run(mask):
    torch.manual_seed(6)
    m = _Pad().eval()
    x = torch.randn(1, 8, 128)
    c = full(m)
    with torch.no_grad(), warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        y = c(x, mask)
    return y, m(x, mask), [e for e in w if issubclass(e.category, RuntimeWarning)]


@pytest.mark.parametrize("fill", [float(torch.finfo(torch.float32).min), -1e9], ids=["finfo_min", "minus_1e9"])
def test_a_mask_row_with_every_key_masked_by_a_finite_value_runs_on_the_cpu(fill):
    """Eager gives a uniform softmax over such a row (a left-padded query); the device would not."""
    y, ref, w = _pad_run(_pad_mask(8, fill))
    assert len(w) == 1 and "every key masked" in str(w[0].message), [str(e.message) for e in w]
    assert regs()[-1].cpu_calls == 1 and cnt()["rt_graph_launches"] == 0
    assert nmse(y, ref) < 1e-10


@pytest.mark.parametrize("kind", ["minus_inf", "bool"])
def test_a_mask_row_with_every_key_masked_by_minus_inf_or_a_bool_stays_on_the_device(kind):
    mask = _pad_mask(8, float("-inf"), kind="bool" if kind == "bool" else "additive")
    y, ref, w = _pad_run(mask)
    assert not w, [str(e.message) for e in w]
    assert regs()[-1].cpu_calls == 0 and regs()[-1].calls == 1
    assert nmse(y[:, 2:], ref[:, 2:]) < NMSE_ATTN


def test_a_finite_mask_with_live_keys_in_every_row_stays_on_the_device():
    y, ref, w = _pad_run(_pad_mask(8, -1e9, pad=0))
    assert not w and regs()[-1].calls == 1
    assert nmse(y, ref) < NMSE_ATTN


_NARROW = {
    "float8": lambda s: s.to(torch.float8_e4m3fn).float(),
    "half": lambda s: s.half().float(),
    "bf16": lambda s: s.bfloat16().float(),
}


class _ScalarUse(nn.Module):
    def __init__(self, use, cast, val):
        super().__init__()
        self.a = nn.Linear(64, 64, bias=False)
        self.b = nn.Linear(64, 64, bias=False)
        self.register_buffer("s", torch.tensor(val))
        self.use, self.cast = use, cast

    def forward(self, x):
        s = self.cast(self.s)
        h = self.a(x)
        if self.use == "scale":
            h = h * s
        elif self.use == "eps":
            h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + s)
        return self.b(torch.tanh(h))


@pytest.mark.parametrize("cast", ["float8", "half", "bf16"])
@pytest.mark.parametrize("use,val", [("scale", 0.3), ("eps", 0.011)])
def test_a_scalar_through_a_narrowing_cast_is_not_used_unrounded(use, val, cast):
    torch.manual_seed(7)
    m = _ScalarUse(use, _NARROW[cast], val).eval()
    x = torch.randn(1, 8, 64)
    c = full(m)
    with torch.no_grad():
        y, ref = c(x), m(x)
    # the rounded value (float8: 0.3 -> 0.3125) is far from the unrounded one; the device bound is NMSE_FREE
    assert nmse(y, ref) < NMSE_FREE
    for r in regs():
        assert not r.region.scalars


def test_a_scalar_through_a_float64_cast_is_still_a_runtime_scalar():
    torch.manual_seed(7)
    m = _ScalarUse("scale", lambda s: s.double().float(), 0.3).eval()
    c = full(m)
    with torch.no_grad():
        y, ref = c(torch.randn(1, 8, 64)), None
    assert len(regs()) == 1 and len(regs()[0].region.scalars) == 1


class _Views(nn.Module):
    """A region value and views of it leave the region."""

    def __init__(self, form):
        super().__init__()
        self.a = nn.Linear(64, 64, bias=False)
        self.b = nn.Linear(64, 64, bias=False)
        self.form = form

    def forward(self, x):
        T = x.shape[1]
        h = torch.tanh(self.a(x))
        y = self.b(h)
        if self.form == "one_view":
            return y, h, h.view(1, T, 2, 32)
        if self.form == "two_views":
            return y, h.view(1, T, 2, 32), h.view(1, T, 4, 16), h
        return y, h.view(1, T, 2, 32).transpose(1, 2), h.view(1, T, 2, 32)    # head-major view of the value


@pytest.mark.parametrize("path", ["device", "cpu"])
@pytest.mark.parametrize("form", ["one_view", "two_views", "head_major"])
def test_a_value_and_its_views_can_all_leave_a_region(monkeypatch, form, path):
    torch.manual_seed(8)
    m = _Views(form).eval()
    x = torch.randn(1, 8, 64)
    c = full(m)
    with _path(monkeypatch, path), torch.no_grad():
        ys, refs = c(x), m(x)
    assert len(ys) == len(refs)
    for a, b in zip(ys, refs):
        assert a.shape == b.shape and a.stride() == b.stride() and nmse(a, b) < (1e-9 if path == "cpu" else NMSE_FREE)
    assert len(regs()) == 1 and _calls(path) == 1


class _RopeOut(nn.Module):
    """q is contiguous head-major in eager; the region (matmul, rope) rebuilds it as a transpose."""

    def __init__(self, contiguous):
        super().__init__()
        self.w = nn.Linear(128, 128, bias=False)
        self.register_buffer("inv", 1.0 / (10000 ** (torch.arange(0, 64, 2).float() / 64)))
        self.contiguous = contiguous

    def forward(self, x, pos):
        T = x.shape[1]
        q = self.w(x).view(1, T, 2, 64).transpose(1, 2)
        if self.contiguous:
            q = q.contiguous()
        f = pos[:, :, None].float() * self.inv
        emb = torch.cat([f, f], -1)
        cos, sin = emb.cos().unsqueeze(1), emb.sin().unsqueeze(1)
        r = q * cos + torch.cat([-q[..., 32:], q[..., :32]], -1) * sin
        return r.reshape(1, 2 * T, 64).sum(-1), r


class _Permuted(nn.Module):
    """Elementwise ops on permuted views: eager keeps the permuted layout of the operands."""

    def __init__(self, perm):
        super().__init__()
        self.perm = perm
        self.a = nn.Linear(64, 64, bias=False)

    def forward(self, x, y):
        T = x.shape[1]
        p = lambda t: t.view(1, T, 2, 32).permute(*self.perm)   # noqa: E731
        z = (p(self.a(x)) + p(y)) * 0.5
        return z, z.transpose(1, 2).reshape(-1)


STRIDE_CASES = [
    ("rope_contiguous", lambda: _RopeOut(True), lambda T: (torch.randn(1, T, 128), torch.arange(T)[None])),
    ("rope_head_major_view", lambda: _RopeOut(False), lambda T: (torch.randn(1, T, 128), torch.arange(T)[None])),
    ("token_major", lambda: _Permuted((0, 1, 2, 3)), lambda T: (torch.randn(1, T, 64), torch.randn(1, T, 64))),
    ("permuted_1_2", lambda: _Permuted((0, 2, 1, 3)), lambda T: (torch.randn(1, T, 64), torch.randn(1, T, 64))),
    ("permuted_other", lambda: _Permuted((0, 3, 1, 2)), lambda T: (torch.randn(1, T, 64), torch.randn(1, T, 64))),
]


@pytest.mark.parametrize("T", [8, 5])
@pytest.mark.parametrize("name,make,args", STRIDE_CASES, ids=[c[0] for c in STRIDE_CASES])
def test_every_region_output_has_the_strides_eager_gives_it(name, make, args, T):
    torch.manual_seed(9)
    m = make().eval()
    a = args(T)
    c = full(m)
    with torch.no_grad():
        ys, refs = c(*a), m(*a)
        ys2 = c(*args(T))     # a second call, the cached graph
    for y, ref in zip(ys, refs):
        assert y.stride() == ref.stride(), (name, y.stride(), ref.stride())
        assert nmse(y, ref) < NMSE_FREE
    assert [y.stride() for y in ys2] == [y.stride() for y in refs]


def test_the_strides_of_the_outputs_hold_with_a_symbolic_sequence_length():
    torch.manual_seed(9)
    m = _RopeOut(True).eval()
    c = full(m, dynamic=True)
    with torch.no_grad():
        for T in (8, 6):
            a = (torch.randn(1, T, 128), torch.arange(T)[None])
            ys, refs = c(*a), m(*a)
            for y, ref in zip(ys, refs):
                assert y.stride() == ref.stride() and nmse(y, ref) < NMSE_FREE


class _TwoRegions(nn.Module):
    """Two regions in one graph: a CPU op between them reads the first, and the second reads it too."""

    def __init__(self, form):
        super().__init__()
        self.a, self.b, self.c = (nn.Linear(64, 64, bias=False) for _ in range(3))
        self.form = form

    def forward(self, x):
        if self.form == "view_base":      # region 1 reads the matmul of region 0 through a view
            h = torch.tanh(self.a(x))
            u = self.b(h)
            e = torch.erf(h)
            return self.c(torch.sigmoid(e) * u)
        h = self.a(x)                      # an output of region 0 used only by region 1
        t = torch.tanh(h)
        e = torch.erf(h)
        return self.c(t * torch.sigmoid(e))


@pytest.mark.parametrize("path", ["device", "cpu"])
@pytest.mark.parametrize("form", ["view_base", "output_only_in_region_1"])
def test_two_regions_in_one_graph_keep_full_mode(monkeypatch, form, path):
    torch.manual_seed(10)
    m = _TwoRegions("view_base" if form == "view_base" else "out").eval()
    x = torch.randn(1, 8, 64)
    c = full(m, strict=True)
    with _path(monkeypatch, path), torch.no_grad():
        y, ref = c(x), m(x)
    assert len(regs()) == 2
    assert nmse(y, ref) < NMSE_FREE      # a matmul beside the regions still runs on the device


def test_two_graphs_that_fail_to_form_regions_each_warn(monkeypatch):
    def boom(*a, **k):
        raise KeyError("test")

    monkeypatch.setattr(B, "rewrite_region", boom)
    torch.manual_seed(10)
    for i in range(2):
        m = _TwoRegions("out" if i else "view_base").eval()
        torch._dynamo.reset()
        with warnings.catch_warnings(record=True) as w, torch.no_grad():
            warnings.simplefilter("always")
            full(m)(torch.randn(1, 8, 64))
        assert sum("forming regions failed" in str(e.message) for e in w) == 1, i


class _Shares(nn.Module):
    """A region value and copies or views of it leave the region."""

    def __init__(self, form):
        super().__init__()
        self.a = nn.Linear(64, 64, bias=False)
        self.b = nn.Linear(64, 64, bias=False)
        self.form = form

    def forward(self, x):
        T = x.shape[1]
        h = torch.tanh(self.a(x))
        y = self.b(h)
        f = self.form
        if f == "value_view":
            return y, h, h.view(1, T, 2, 32)
        if f == "view_view":
            return y, h.view(1, T, 2, 32), h.view(1, T, 4, 16)
        if f == "value_head_major":
            return y, h, h.view(1, T, 2, 32).transpose(1, 2)
        if f == "value_clone":
            return y, h, h.clone()
        if f == "value_detach":
            return y, h, h.detach()
        return y, h, h.clone().view(1, T, 2, 32)            # value_clone_view


SHARE_FORMS = ["value_view", "view_view", "value_head_major", "value_clone", "value_detach", "value_clone_view"]


def _sharing(outs) -> list:
    """For each output, the index of the first output that has its storage."""
    ptrs = [t.untyped_storage().data_ptr() for t in outs]
    return [ptrs.index(p) for p in ptrs]


def _write_shows(outs) -> list:
    """Add one to the second output in place; for every output, whether its values changed."""
    before = [t.detach().clone() for t in outs]
    outs[1].add_(1.0)
    return [bool((a != b).any()) for a, b in zip(outs, before)]


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("path", ["device", "cpu"])
@pytest.mark.parametrize("form", SHARE_FORMS)
def test_outputs_of_a_region_share_storage_as_they_do_in_eager(monkeypatch, form, path, dynamic):
    torch.manual_seed(12)
    m = _Shares(form).eval()
    x = torch.randn(1, 8, 64)
    with torch.no_grad():
        want = list(m(x))
        want_share, want_shows = _sharing(want), _write_shows(want)
        got = {}
        for mode in ("matmul", "full"):
            torch._dynamo.reset()
            c = tt.compile(m, mode=mode, dynamic=dynamic)
            if mode == "full":
                with _path(monkeypatch, path):
                    got[mode] = list(c(x))
            else:
                got[mode] = list(c(x))
    assert len(regs()) == 1
    ref = m(x)
    for mode, outs in got.items():
        for a, b in zip(outs, ref):
            assert a.shape == b.shape and a.stride() == b.stride()
            assert nmse(a, b) < (1e-9 if mode == "full" and path == "cpu" else NMSE_FREE)
        assert _sharing(outs) == want_share, mode
        assert _write_shows(outs) == want_shows, mode


class _RopeShare(_RopeOut):
    """r is contiguous in eager (the region rebuilds a transpose) and a detached alias of it leaves as well."""

    def forward(self, x, pos):
        T = x.shape[1]
        q = self.w(x).view(1, T, 2, 64).transpose(1, 2).contiguous()
        f = pos[:, :, None].float() * self.inv
        emb = torch.cat([f, f], -1)
        cos, sin = emb.cos().unsqueeze(1), emb.sin().unsqueeze(1)
        r = q * cos + torch.cat([-q[..., 32:], q[..., :32]], -1) * sin
        return r, r.detach()


@pytest.mark.parametrize("T", [8, 5])
def test_a_shared_group_with_a_member_that_needs_the_contiguous_copy_keeps_eagers_sharing(T):
    """The rebuilt head-major value is not r's contiguous layout, so r is rebuilt through a copy and could not
    share storage with its view: that op stays on the CPU, and the returned tensors share as in eager."""
    torch.manual_seed(13)
    m = _RopeShare(True).eval()
    x, pos = torch.randn(1, T, 128), torch.arange(T)[None]
    with torch.no_grad():
        want = list(m(x, pos))
        want_share, want_shows = _sharing(want), _write_shows(want)
        got = list(full(m)(x, pos))
    assert want_share == [0, 0]
    assert _sharing(got) == want_share and _write_shows(got) == want_shows


def test_the_storage_root_walks_views_and_stops_at_copies():
    import torch.fx as fx
    A = torch.ops.aten
    g = fx.Graph()
    x = g.placeholder("x")
    u = g.call_function(A._unsafe_view.default, (x, [4, 4]))
    v = g.call_function(A.view.default, (u, [16]))
    t = g.call_function(A.detach.default, (v,))
    c = g.call_function(A.clone.default, (t,))
    cv = g.call_function(A.view.default, (c, [4, 4]))
    k = g.call_function(A.contiguous.default, (u,))
    s = g.call_function(A.add.Tensor, (u, 1.0))
    assert [LW.storage_root(n) for n in (u, v, t)] == [x, x, x]
    assert LW.storage_root(c) is c and LW.storage_root(cv) is c and LW.storage_root(k) is k
    assert LW.storage_root(s) is s
    assert LW.storage_root(t, stop=u) is u
    assert B._alias_root(v) is x


class _Gqa(nn.Module):
    """Grouped-query attention whose expanded K is also returned or read by a CPU op."""

    def __init__(self, expand, use, H=4, Hk=2, D=64):
        super().__init__()
        self.expand, self.use, self.H, self.Hk, self.D = expand, use, H, Hk, D
        self.q = nn.Linear(H * D, H * D, bias=False)
        self.k = nn.Linear(H * D, Hk * D, bias=False)
        self.v = nn.Linear(H * D, Hk * D, bias=False)
        self.o = nn.Linear(H * D, H * D, bias=False)

    def rep(self, t):
        B, Hk, T, D = t.shape
        if self.expand == "repeat_interleave":
            return t.repeat_interleave(self.H // Hk, 1)
        return t[:, :, None].expand(B, Hk, self.H // Hk, T, D).reshape(B, self.H, T, D)

    def forward(self, x):
        T = x.shape[1]
        q = self.q(x).view(1, T, self.H, self.D).transpose(1, 2)
        k = self.k(x).view(1, T, self.Hk, self.D).transpose(1, 2)
        v = self.v(x).view(1, T, self.Hk, self.D).transpose(1, 2)
        kr, vr = self.rep(k), self.rep(v)
        a = F.scaled_dot_product_attention(q, kr, vr, is_causal=True)
        y = self.o(a.transpose(1, 2).reshape(1, T, self.H * self.D))
        if self.use == "out":
            return y, kr
        if self.use == "cpu":
            return y + kr.sum((1, 3))[..., None] * 0.01
        return y


@pytest.mark.parametrize("path", ["device", "cpu"])
@pytest.mark.parametrize("use", ["none", "out", "cpu"])
@pytest.mark.parametrize("expand", ["expand_reshape", "repeat_interleave"])
def test_a_gqa_expansion_that_is_also_read_outside_the_region_keeps_full_mode(monkeypatch, expand, use, path):
    torch.manual_seed(14)
    m = _Gqa(expand, use).eval()
    c = full(m, strict=True)
    with _path(monkeypatch, path), torch.no_grad():
        for _ in range(2):
            x = torch.randn(1, 8, 256)
            ys, refs = c(x), m(x)
            ys, refs = (ys if isinstance(ys, tuple) else (ys,)), (refs if isinstance(refs, tuple) else (refs,))
            for a, b in zip(ys, refs):
                assert a.shape == b.shape and a.stride() == b.stride()
                assert nmse(a, b) < (1e-9 if path == "cpu" else NMSE_ATTN)
    # one region with the four matmuls and the attention: the expansion is not an op of it
    assert len(regs()) == 1 and regs()[0].region.op_counts() == {"mul_mat": 4, "attn": 1}
    assert _calls(path) == 2


# ------------------------------------------------------------------------------------------
# the option attention
# ------------------------------------------------------------------------------------------
def _attn_regions():
    return [r for r in regs() if r.region.op_counts().get("attn")]


def test_attention_cpu_keeps_attention_out_of_the_regions():
    torch.manual_seed(7)
    m = QwenLike().eval()
    x = ids(6)
    with torch.no_grad():
        ref = m(*x)
        y_cpu = full(m, attention="cpu", strict=True)(*x)
        n_cpu = len(regs())
        ops_cpu = sum((r.region.op_counts().get("attn", 0) for r in regs()))
        torch._dynamo.reset()
        y_mm = tt.compile(m, mode="matmul")(*x)
        torch._dynamo.reset()
        y_dev = full(m, attention="device")(*x)
        y_def = full(m)(*x)
    assert ops_cpu == 0 and n_cpu >= 3                       # the regions end at each of the two attentions
    assert nmse(y_cpu, ref) < NMSE_FREE and nmse(y_cpu, ref) < 2 * nmse(y_mm, ref) + 1e-6
    # "device" and the default lower attention; the rest of the regions are the same
    assert sum(r.region.op_counts().get("attn", 0) for r in regs()[n_cpu:]) == 4
    assert torch.equal(y_dev, y_def)
    assert nmse(y_cpu, ref) < nmse(y_dev, ref)               # float32 attention is closer to eager


def test_attention_cpu_says_why_in_the_notes():
    torch.manual_seed(7)
    m = QwenLike().eval()
    with torch.no_grad():
        full(m, attention="cpu", debug=1)(*ids(6))
    r = regs()[0].region
    reasons = [v for v in r.touching.values() if v]
    assert any("attention='cpu'" in v for v in reasons), reasons


def test_attention_through_the_environment(monkeypatch):
    m = QwenLike().eval()
    x = ids(6)
    monkeypatch.setenv("TSR_TORCH_ATTENTION", "cpu")
    with torch.no_grad():
        torch.compile(m, backend="tesseract", options={"mode": "full"})(*x)
        assert not _attn_regions()
        torch._dynamo.reset()
        torch.compile(m, backend="tesseract", options={"mode": "full", "attention": "device"})(*x)    # the option beats the variable
    assert len(_attn_regions()) == 1
    torch._dynamo.reset()
    monkeypatch.delenv("TSR_TORCH_ATTENTION")
    with torch.no_grad():
        torch.compile(m, backend="tesseract", options={"mode": "full"})(*x)                                      # the default is device
    assert len(_attn_regions()) == 2


def test_unknown_attention_is_an_error(monkeypatch):
    with pytest.raises(ValueError, match=r"unknown attention 'gpu' \(device, cpu\)"):
        tt.compile(QwenLike().eval(), attention="gpu")
    with pytest.raises(ValueError, match="unknown attention"):
        tt.Options.from_dict({"attention": "gpu"})
    monkeypatch.setenv("TSR_TORCH_ATTENTION", "gpu")
    with pytest.raises(ValueError, match="unknown attention"):
        B.resolve_attention(None)
    with pytest.raises(ValueError, match="attention"):
        tt.Options.from_dict({"nope": 1})


def test_matmul_mode_ignores_the_attention_option():
    m = QwenLike().eval()
    with torch.no_grad():
        tt.compile(m, mode="matmul", attention="cpu")(*ids(6))
    assert cnt()["regions"] == 0 and cnt()["device_calls"] == 15


def test_an_unknown_attention_in_the_environment_is_an_error_in_full_mode_only(monkeypatch):
    """The resolved attention (the option, else TSR_TORCH_ATTENTION) is checked in full mode, before the regions are
    formed; matmul mode has no attention option and ignores the variable."""
    monkeypatch.setenv("TSR_TORCH_ATTENTION", "bogus")
    m = QwenLike().eval()
    x = ids(6)
    with torch.no_grad():
        with pytest.raises(Exception, match="unknown attention 'bogus'"):
            tt.compile(m, mode="full")(*x)
        assert cnt()["regions"] == 0
        torch._dynamo.reset()
        y = tt.compile(m, mode="matmul")(*x)           # no error: matmul mode does not read the variable
        assert cnt()["regions"] == 0 and cnt()["device_calls"] == 15
        assert nmse(y, m(*x)) < NMSE_ATTN
        torch._dynamo.reset()
        tt.compile(m, mode="full", attention="cpu")(*x)    # an explicit option wins over the variable
    assert len(regs()) >= 1


class ReaderBeforeMaker(nn.Module):
    """Two attentions. The second reads the head-major value v2, whose projection (with a bias) sits after the
    projections of q2 and k2 in the graph. The per-head norm on the CPU reads the same view of the attention
    output that the projection of v2 reads."""

    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(1)

        def mk(*s):
            return nn.Parameter(torch.randn(*s, generator=g) * 0.1)

        self.wq, self.wk, self.wv, self.w48, self.wo = mk(128, 64), mk(128, 64), mk(128, 64), mk(256, 128), mk(64, 256)
        self.n1, self.n2, self.n3 = nn.Parameter(torch.ones(128)), nn.Parameter(torch.ones(128)), nn.Parameter(torch.ones(32))
        self.b = nn.Parameter(torch.randn(256, generator=g) * 0.1)

    @staticmethod
    def rms(t, w):
        return w * (t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + 1e-6))

    def forward(self, h):
        T = h.shape[1]
        q = self.rms((h @ self.wq.t()).view(1, T, 1, 128).transpose(1, 2), self.n1)
        k = self.rms((h @ self.wk.t()).view(1, T, 1, 128).transpose(1, 2), self.n2)
        v = (h @ self.wv.t()).view(1, T, 1, 128).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        u = a.transpose(1, 2).contiguous().view(1, T, 128)
        un = self.rms(u.view(1, T, 4, 32), self.n3).reshape(1, T, 128)
        q2 = torch.matmul(un, self.w48.t()).view(1, T, 4, 64).transpose(1, 2)
        k2 = torch.matmul(un, self.w48.t()).view(1, T, 4, 64).transpose(1, 2)
        v2 = ((u @ self.w48.t()) + self.b).view(1, T, 4, 64).transpose(1, 2)
        a2 = F.scaled_dot_product_attention(q2, k2, v2, is_causal=True)
        return a2.transpose(1, 2).flatten(2) @ self.wo.t()


def test_a_view_that_a_cpu_op_reads_does_not_split_a_region():
    """Calls with T = 13, 13, 14 (the third is the recompile with a symbolic T). The view of the attention
    output that v2's projection reads is read by a per-head norm on the CPU too; it carries the output of the
    CPU attention, not a value of the region, so it does not make the regions read each other. The second
    attention forms one region with its projections on every call; strict mode with warnings as errors."""
    torch.manual_seed(0)
    m = ReaderBeforeMaker().eval()
    inputs = [torch.randn(1, T, 64, generator=torch.Generator().manual_seed(T)) for T in (13, 13, 14)]
    with torch.no_grad():
        refs = [m(x) for x in inputs]
        mm = tt.compile(m, mode="matmul")
        bound = max(20 * max(nmse(mm(x), r) for x, r in zip(inputs, refs)), NMSE_ATTN)
        torch._dynamo.reset()
        c = full(m, strict=True)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            for x, ref in zip(inputs, refs):
                first = len(regs())
                y = c(x)
                new = regs()[first:]
                assert nmse(y, ref) < bound
                if new:
                    assert len(new) == 1, [r.region.op_counts() for r in new]
                    assert new[0].region.op_counts().get("attn") == 1 and new[0].region.op_counts().get("mul_mat", 0) >= 3


class SharedView(nn.Module):
    """One region: the projection of ``a`` (with a bias) and the chain b1 -> tanh -> mul that reads a view of ``a``;
    the first op of the chain is earlier in the graph than the projection of ``a``. Both read one view of the flipped input."""

    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(1)
        self.wa, self.wb, self.wo = (nn.Parameter(torch.randn(64, 64, generator=g) * 0.125) for _ in range(3))
        self.ba = nn.Parameter(torch.randn(64, generator=g) * 0.1)

    def forward(self, x):
        T = x.shape[1]
        uv = torch.flip(x, [-1]).view(1, T, 4, 16)
        b2 = torch.tanh(uv.abs().reshape(1, T, 64) @ self.wb.t())
        a = uv.reshape(1, T, 64) @ self.wa.t() + self.ba
        y = (b2.view(1, T, 4, 16) * a.view(1, T, 4, 16)).reshape(1, T, 64)
        return y @ self.wo.t()


def test_a_view_of_a_cpu_value_that_two_chains_read_stays_one_region():
    torch.manual_seed(0)
    m = SharedView().eval()
    with torch.no_grad():
        c = full(m, strict=True)
        for T in (8, 9):
            x = torch.randn(1, T, 64)
            first = len(regs())
            y = c(x)
            assert nmse(y, m(x)) < NMSE_FREE
            new = regs()[first:]
            assert len(new) == 1 and new[0].region.op_counts().get("mul_mat") == 3, [r.region.op_counts() for r in new]


class ReadsThroughCpu(nn.Module):
    """The region {b1, tanh, mul, ...} has the earliest op and reads, through a CPU flip, the value of the region
    {a, bias, sigmoid} that comes later in the graph."""

    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(7)
        self.wa, self.wb, self.wo = (nn.Parameter(torch.randn(64, 64, generator=g) * 0.125) for _ in range(3))
        self.ba = nn.Parameter(torch.randn(64, generator=g) * 0.1)

    def forward(self, x):
        u = torch.flip(x, [-1])
        b2 = torch.tanh(u.abs() @ self.wb.t())
        a = torch.sigmoid(u @ self.wa.t() + self.ba)
        return (b2 * torch.flip(a, [-1])) @ self.wo.t()


def test_a_region_that_reads_another_through_a_cpu_op_comes_after_it():
    """The rewrite replaces the regions in order; a region whose value another reads, also through nodes on the CPU,
    comes first."""
    torch.manual_seed(0)
    m = ReadsThroughCpu().eval()
    with torch.no_grad():
        c = full(m, strict=True)
        for T in (8, 9):
            x = torch.randn(1, T, 64)
            first = len(regs())
            y = c(x)
            assert nmse(y, m(x)) < NMSE_FREE
            new = regs()[first:]
            if T == 8:
                assert [("add" in r.region.op_counts(), "mul" in r.region.op_counts()) for r in new] == [(True, False), (False, True)], \
                    [r.region.op_counts() for r in new]


class ReadEachOther(nn.Module):
    """Two matmul chains that read each other through CPU ops: A = {mm, tanh, add} reads flip(B), B = {mm, sigmoid, mul}
    reads roll(A). The two cannot be two regions."""

    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(5)
        self.w1, self.w2 = (nn.Parameter(torch.randn(64, 64, generator=g) * 0.125) for _ in range(2))

    def forward(self, x):
        a1 = torch.tanh(x @ self.w1.t())
        b2 = torch.sigmoid(x @ self.w2.t())
        b3 = torch.roll(a1, 1, -1) * b2
        a3 = a1 + torch.flip(b2, [-1])
        return a3, b3


def test_regions_that_read_each_other_through_cpu_ops_are_cut_apart():
    """The graph keeps full mode: the cand that reads through the CPU moves out of its region, strict mode with warnings
    as errors forms regions on every call, and the values are right."""
    torch.manual_seed(0)
    m = ReadEachOther().eval()
    with torch.no_grad():
        c = full(m, strict=True)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            for T in (8, 8, 9):
                x = torch.randn(1, T, 64)
                y = c(x)
                ref = m(x)
                assert all(nmse(a, b) < NMSE_FREE for a, b in zip(y, ref))
        assert len(regs()) >= 2 and all(sum(r.region.op_counts().values()) >= 2 for r in regs())
