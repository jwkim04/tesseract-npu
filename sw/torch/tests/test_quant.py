"""QuantLinear: weights that arrive as ggml blocks (Q8_0, Q4_K) run on the device as they are.

The blocks are random but valid (sw/tools/numcheck.py's generators), so the expected y is the numpy
spec model of the format (B1b), not a second quantization. Eager and compiled calls must give the
spec model's bits; the device holds exactly the blocks that were given."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

import spec_model as sm
import tesseract_torch as tt

nc = sm.nc
REPO = Path(__file__).resolve().parents[3]


def _bits(t: torch.Tensor) -> np.ndarray:
    return t.contiguous().view(torch.int32).numpy()


def _blocks(fmt: str, N: int, K: int, seed: int) -> np.ndarray:
    """uint8 [N, row_bytes] random valid ggml blocks."""
    rng = np.random.default_rng(seed)
    if fmt == "q8_0":
        nb = K // 32
        b = nc.make_q8_0_blocks(nc.rand_fp16(rng, (N, nb), emin=8, emax=14), rng.integers(-128, 128, size=(N, nb, 32)))
    else:
        ns = K // 256
        b = nc.make_q4_K_blocks(nc.rand_fp16(rng, (N, ns), emin=8, emax=14), nc.rand_fp16(rng, (N, ns), emin=8, emax=14),
                                rng.integers(0, 64, size=(N, ns, 8)), rng.integers(0, 64, size=(N, ns, 8)),
                                rng.integers(0, 16, size=(N, ns, 256)))
    return np.ascontiguousarray(b).reshape(N, -1)


def _spec(fmt: str, b: np.ndarray, x: np.ndarray, N: int, K: int) -> np.ndarray:
    """y [T, N] float32 of the spec model."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    if fmt == "q8_0":
        return sm.spec_y(b, x, N, K)
    return nc.spec_model_y(nc.FMT_Q4_K, nc.parse_q4_K(b, N, K), nc.np_qact(nc.QA_A256, x), x.shape[0])


def _x(shape, seed, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    return ((torch.rand(shape, generator=g) * 2 - 1) * 2.0).to(dtype)


SHAPES = {"q8_0": [(40, 512), (130, 1024), (3, 64)], "q4_k": [(40, 512), (130, 1024), (7, 256)]}
CASES = [(f, N, K) for f, v in SHAPES.items() for N, K in v]


@pytest.mark.parametrize("fmt,N,K", CASES)
@pytest.mark.parametrize("T", [1, 5, 9])
def test_eager_and_compiled_equal_the_spec_model(dev, fmt, N, K, T):
    b = _blocks(fmt, N, K, seed=N * 31 + K + T)
    layer = QuantLayer(b, K, fmt)
    x = _x((T, K), seed=T + K)
    want = _spec(fmt, b, x.numpy(), N, K)
    with torch.no_grad():
        y_eager = layer(x)
        compiled = tt.compile(layer)
        y_c = compiled(x)
        y_c2 = compiled(x)                       # the second call hits the weight cache and the op-list cache
    assert y_eager.dtype == torch.float32 and tuple(y_eager.shape) == (T, N)
    assert np.array_equal(_bits(y_eager), want.view(np.int32))
    assert np.array_equal(_bits(y_c), want.view(np.int32)) and np.array_equal(_bits(y_c2), want.view(np.int32))
    c = tt.counters()
    assert c["uploads"] == 1 and c["cache_hits"] == 2 and c["device_calls"] == 3
    h = tt.weight_cache().acquire_blocks(layer.blocks, fmt, K)[1]
    assert dev.info(h)["fmt"] == fmt
    assert np.array_equal(dev.read_blocks(h).numpy(), b)           # stored as given


def QuantLayer(b, K, fmt, bias=None):
    return tt.QuantLinear(torch.from_numpy(b), K, fmt, bias)


@pytest.mark.parametrize("fmt", ["q8_0", "q4_k"])
def test_bias_dtype_and_batch_dims(dev, fmt):
    N, K = 48, 512
    b = _blocks(fmt, N, K, seed=7)
    bias = _x((N,), seed=8)
    layer = QuantLayer(b, K, fmt, bias)
    x = _x((2, 3, K), seed=9, dtype=torch.bfloat16)
    want = _spec(fmt, b, x.float().reshape(-1, K).numpy(), N, K)
    want = torch.from_numpy(want) + bias
    with torch.no_grad():
        y = tt.compile(layer)(x)
    assert y.dtype == torch.bfloat16 and tuple(y.shape) == (2, 3, N)
    assert torch.equal(y.float().reshape(-1, N), want.to(torch.bfloat16).float())


def test_mixed_model_float_and_blocks_and_verify(dev):
    """A float nn.Linear (partitioned to the device as always) next to a Q4_K layer; every call checked
    against the reference model."""
    K, H = 512, 256
    b = _blocks("q4_k", H, K, seed=11)
    m = nn.Sequential(QuantLayer(b, K, "q4_k"), nn.ReLU(), nn.Linear(H, 64, bias=False)).eval()
    x = _x((4, K), seed=12)
    with torch.no_grad(), tt.reference_linears():
        ref = m(x)                                                    # eager: Q4_K layer and linear_ref float layer
    tt.reset_counters()
    tt.set_verify(True)
    try:
        with torch.no_grad():
            y = tt.compile(m)(x)
    finally:
        tt.set_verify(False)
    assert np.array_equal(_bits(y), _bits(ref))
    c = tt.counters()
    assert c["nodes_device"] == 1 and c["device_calls"] == 2 and c["verified_calls"] == 2 and c["cpu_fallback_calls"] == 0


def test_in_place_change_is_uploaded_again(dev):
    N, K = 32, 256
    b = _blocks("q4_k", N, K, seed=21)
    layer = QuantLayer(b, K, "q4_k")
    x = _x((2, K), seed=22)
    with torch.no_grad():
        y1 = layer(x)
        layer.blocks.copy_(torch.from_numpy(_blocks("q4_k", N, K, seed=23)))
        y2 = layer(x)
    assert np.array_equal(_bits(y2), _spec("q4_k", layer.blocks.numpy(), x.numpy(), N, K).view(np.int32))
    assert not np.array_equal(_bits(y1), _bits(y2))
    assert tt.counters()["uploads"] == 2 and tt.counters()["reuploads"] == 1


def test_errors():
    with pytest.raises(ValueError, match="not a multiple"):
        tt.QuantLinear(torch.zeros(4, 144, dtype=torch.uint8), 288, "q4_k")
    with pytest.raises(ValueError, match="uint8"):
        tt.QuantLinear(torch.zeros(4, 34, dtype=torch.int8), 32, "q8_0")
    with pytest.raises(ValueError, match=r"uint8 \[N, 144\]"):
        tt.QuantLinear(torch.zeros(4, 100, dtype=torch.uint8), 256, "q4_k")
    with pytest.raises(ValueError, match="unknown format"):
        tt.QuantLinear(torch.zeros(4, 34, dtype=torch.uint8), 32, "q5_0")
    layer = tt.QuantLinear(torch.zeros(4, 34, dtype=torch.uint8), 32, "q8_0")
    with pytest.raises(ValueError, match="features"):
        tt.linear_blocks(torch.zeros(1, 64), layer.blocks, "q8_0", 32)


@pytest.mark.parametrize("fmt", ["q8_0", "q4_k"])
def test_from_gguf_tensor(dev, tmp_path, fmt):
    """The blocks of a GGUF tensor (written and read by gguf-py) are what the device holds."""
    sys.path.insert(0, str(REPO / "third_party" / "llama.cpp" / "gguf-py"))
    gguf = pytest.importorskip("gguf")
    N, K = 24, 512
    b = _blocks(fmt, N, K, seed=31)
    qt = gguf.GGMLQuantizationType.Q8_0 if fmt == "q8_0" else gguf.GGMLQuantizationType.Q4_K
    path = tmp_path / "w.gguf"
    wr = gguf.GGUFWriter(str(path), "llama")
    wr.add_tensor("blk.0.ffn_up.weight", b, raw_dtype=qt)
    wr.add_tensor("blk.0.attn_norm.weight", np.ones(K, np.float32))
    wr.write_header_to_file()
    wr.write_kv_data_to_file()
    wr.write_tensors_to_file()
    wr.close()
    tensors = {t.name: t for t in gguf.GGUFReader(str(path)).tensors}
    layer = tt.QuantLinear.from_gguf(tensors["blk.0.ffn_up.weight"])
    assert (layer.in_features, layer.out_features, layer.fmt) == (K, N, fmt)
    x = _x((3, K), seed=32)
    with torch.no_grad():
        y = layer(x)
    assert np.array_equal(_bits(y), _spec(fmt, b, x.numpy(), N, K).view(np.int32))
    with pytest.raises(ValueError, match="not Q8_0 / Q4_K"):
        tt.QuantLinear.from_gguf(tensors["blk.0.attn_norm.weight"])
