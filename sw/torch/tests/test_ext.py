"""The C++ extension _tesseract_rt (tsr_rt bindings): device vs reference vs an independent model, bit
for bit; the Q8_0 conversion vs an independent quantize_row_q8_0_ref; strides, K parts, T tiling,
host traffic, errors."""
from __future__ import annotations

import numpy as np
import pytest
import torch

import spec_model as sm
import tesseract_torch as tt

_C = tt._C


def _bits(t: torch.Tensor) -> np.ndarray:
    return t.contiguous().view(torch.int32).numpy()


def _same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and np.array_equal(_bits(a), _bits(b))


def _rand(shape, seed, dtype=torch.float32, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return ((torch.rand(shape, generator=g) * 2 - 1) * scale).to(dtype)


# (T, N, K): N not a multiple of 16 or of NUM_CH, N < NUM_CH, partial chunks (K % 1024 != 0),
# Qwen2.5-0.5B shapes (896, 4864), T > T_MAX
SHAPES = [
    (1, 1, 32), (1, 3, 64), (2, 16, 256), (3, 64, 512), (5, 100, 96), (17, 256, 1024),
    (1, 896, 896), (4, 128, 4864), (2, 4864, 896), (40, 48, 2048), (9, 1000, 1056),
]


@pytest.mark.parametrize("T,N,K", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_bit_exact_vs_reference_and_spec_model(dev, T, N, K, dtype):
    seed = T * 1000003 + N * 101 + K
    w = _rand((N, K), seed, dtype)
    x = _rand((T, K), seed + 1, torch.float32, scale=3.0)
    h = dev.upload(w)
    try:
        info = dev.info(h)
        assert (info["N"], info["K"], info["fmt"]) == (N, K, "q8_0")
        # stored blocks == quantize_row_q8_0_ref of the (exactly widened) weight, byte for byte
        blocks = dev.read_blocks(h).numpy()
        ref_blocks = sm.q8_0_ref_blocks(w.float().numpy())
        assert np.array_equal(blocks, ref_blocks)
        y = dev.linear(h, x)
        y_ref = dev.linear_ref(h, x)
        assert y.dtype == torch.float32 and tuple(y.shape) == (T, N)
        assert _same_bits(y, y_ref), "device != libtsr_ref"
        y_np = sm.spec_y(blocks, x.numpy(), N, K)
        assert np.array_equal(_bits(y), y_np.view(np.int32)), "device != numpy spec model"
    finally:
        dev.free(h)


def test_q8_0_conversion_rounding_boundaries(dev):
    """Directed rows: x * id exactly on .5 (roundf rounds away from zero: {127, 2.5, -0.5, 0.5} ->
    {127, 3, -1, 1}, where the x86 SIMD quantizer gives {127, 2, 0, 0}), zero blocks, d below the fp16
    normal range (stored as an fp16 subnormal or 0), d an f32 subnormal, f16 / bf16 widening.
    Not covered: blocks with amax < ~3.7e-37, where quantize_row_q8_0_ref's id = 1/d overflows to inf
    and ggml casts inf to int8 (undefined in C)."""
    K = 128
    rows = np.zeros((4, K), np.float32)
    rows[0, :6] = [127, 2.5, -0.5, 0.5, 1.5, -126.5]
    rows[1, 32:64] = np.linspace(-1e-6, 1e-6, 32)          # d = amax/127 below the fp16 normal range
    rows[1, 64:96] = np.linspace(-3e-3, 2e-3, 32)          # d an fp16 subnormal
    rows[2, 64:96] = np.linspace(-1e-36, 1e-36, 32)        # d = 7.9e-39: an f32 subnormal, id finite
    rows[3] = np.arange(K, dtype=np.float32) - 63.5
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        w = torch.from_numpy(rows).to(dtype)
        h = dev.upload(w)
        try:
            blocks = dev.read_blocks(h).numpy()
            assert np.array_equal(blocks, sm.q8_0_ref_blocks(w.float().numpy())), dtype
            if dtype == torch.float32:
                q = blocks[0, 2:8].view(np.int8)
                assert q.tolist() == [127, 3, -1, 1, 2, -127]
            x = _rand((2, K), 3)
            assert _same_bits(dev.linear(h, x), dev.linear_ref(h, x))
        finally:
            dev.free(h)


def test_large_head_rows_subset(dev):
    """N = 151936 (Qwen2.5 output head), K = 896: device == reference on all rows, == spec model on
    a sample of rows."""
    N, K, T = 151936, 896, 3
    w = _rand((N, K), 7, torch.bfloat16)
    x = _rand((T, K), 8)
    h = dev.upload(w)
    try:
        y = dev.linear(h, x)
        assert _same_bits(y, dev.linear_ref(h, x))
        rows = np.sort(np.random.default_rng(1).choice(N, 2048, replace=False))
        blocks = dev.read_blocks(h).numpy()[rows]
        assert np.array_equal(blocks, sm.q8_0_ref_blocks(w.float().numpy()[rows]))
        y_np = sm.spec_y(blocks, x.numpy(), len(rows), K)
        assert np.array_equal(_bits(y)[:, rows], y_np.view(np.int32))
    finally:
        dev.free(h)


def test_k_parts_beyond_k_max(dev):
    """K > K_MAX: the runtime splits K and accumulates with MVM.ACC; device == reference."""
    kmax = dev.config["k_max"]
    K = kmax + 256
    N, T = 24, 3
    w = _rand((N, K), 11, torch.float16)
    x = _rand((T, K), 12)
    h = dev.upload(w)
    try:
        info = dev.info(h)
        assert info["k_parts"] == 2
        y = dev.linear(h, x)
        assert _same_bits(y, dev.linear_ref(h, x))
        assert np.array_equal(dev.read_blocks(h).numpy(), sm.q8_0_ref_blocks(w.float().numpy()))
        # each K part on its own is the spec model; the parts are added once in f32 (spec ACC)
        kp = info["k_part"]
        blocks = dev.read_blocks(h).numpy()
        bpp = kp // 32 * 34
        y0 = sm.spec_y(blocks[:, :bpp], x.numpy()[:, :kp], N, kp)
        y1 = sm.spec_y(blocks[:, bpp:], x.numpy()[:, kp:], N, K - kp)
        assert np.array_equal(_bits(y), sm.nc.fadd(y0, y1).view(np.int32))
    finally:
        dev.free(h)


def test_strided_and_misaligned_operands(dev):
    """Weight rows with a stride (a column slice), a weight at a 2-byte offset, strided / transposed x,
    an output view: all equal the packed computation bit for bit."""
    N, K, T = 80, 160, 6
    big = _rand((N, K + 48), 21, torch.float16)
    w = big[:, 16:16 + K]                     # row stride K + 48, offset 16 elements
    assert w.stride() == (K + 48, 1)
    h1 = dev.upload(w)
    h2 = dev.upload(w.contiguous())
    flat = _rand((N * K + 1,), 22, torch.float16)
    wm = flat[1:].view(N, K)                  # storage offset 1 element = 2 bytes
    h3 = dev.upload(wm)
    h4 = dev.upload(wm.contiguous())
    try:
        assert np.array_equal(dev.read_blocks(h1).numpy(), dev.read_blocks(h2).numpy())
        assert np.array_equal(dev.read_blocks(h3).numpy(), dev.read_blocks(h4).numpy())
        xb = _rand((T, K + 32), 23)
        xs = xb[:, 32:]                       # row stride K + 32
        xt = _rand((K, T), 24).t()            # transposed (column-major) x
        y_packed = dev.linear(h2, xs.contiguous())
        assert _same_bits(dev.linear(h1, xs), y_packed)
        assert _same_bits(dev.linear(h1, xt), dev.linear(h2, xt.contiguous()))
        out_big = torch.full((T, N + 16), 7.0)
        out = out_big[:, 8:8 + N]
        r = dev.linear(h1, xs, out=out)
        assert r.data_ptr() == out.data_ptr() and _same_bits(out.contiguous(), y_packed)
        assert torch.all(out_big[:, :8] == 7.0) and torch.all(out_big[:, 8 + N:] == 7.0)
    finally:
        for h in (h1, h2, h3, h4):
            dev.free(h)


def test_t_tiling_and_host_traffic(dev):
    """T > T_MAX tokens run as T tiles inside one program; host traffic is exactly x in and y out."""
    tmax = dev.config["t_max"]
    N, K = 64, 512
    w = _rand((N, K), 31)
    h = dev.upload(w)
    try:
        for T in sorted({1, tmax, tmax + 1, 3 * tmax + 5, 300}):
            x = _rand((T, K), 32 + T)
            dev.reset_counters()
            y = dev.linear(h, x)
            c = dev.counters()
            assert c["execs"] == 1 and c["linears"] == 1 and c["tokens"] == T
            # first call of this T: x plus the new program (32-byte instructions) go to the device
            assert (c["prog_misses"], c["h2d_weight_bytes"]) == (1, 0)
            prog = c["h2d_bytes"] - T * K * 4
            assert prog > 0 and prog % 32 == 0
            assert c["d2h_bytes"] == T * N * 4
            assert _same_bits(y, dev.linear_ref(h, x))
            # a repeated call is a program-cache hit and moves only x and y
            dev.reset_counters()
            y2 = dev.linear(h, x)
            c = dev.counters()
            assert (c["prog_hits"], c["prog_misses"], c["execs"]) == (1, 0, 1)
            assert (c["h2d_bytes"], c["h2d_weight_bytes"], c["d2h_bytes"]) == (T * K * 4, 0, T * N * 4)
            assert _same_bits(y, y2)
    finally:
        dev.free(h)


def test_t_max8_device(monkeypatch):
    """A device of its own with T_MAX = 8 (sim only): T tiles of 8 tokens, bit-exact vs reference."""
    if (tt.device().transport != "sim"):
        pytest.skip("needs the sim transport")
    monkeypatch.setenv("TSR_SIM_TMAX", "8")
    d8 = _C.Device("sim")
    try:
        assert d8.config["t_max"] == 8
        N, K = 200, 1024
        w = _rand((N, K), 41, torch.bfloat16)
        h = d8.upload(w)
        for T in (1, 7, 8, 9, 37, 129):
            x = _rand((T, K), 42 + T)
            d8.reset_counters()
            y = d8.linear(h, x)
            assert d8.counters()["execs"] == 1
            assert _same_bits(y, d8.linear_ref(h, x))
            y_np = sm.spec_y(d8.read_blocks(h).numpy(), x.numpy(), N, K)
            assert np.array_equal(_bits(y), y_np.view(np.int32))
    finally:
        d8.close()
    assert d8.closed


def test_native_blocks(dev):
    """upload_blocks: ggml Q8_0 / Q4_K blocks are stored as they are; device == reference == spec model."""
    rng = np.random.default_rng(5)
    N, K, T = 40, 512, 4
    d16 = sm.nc.rand_fp16(rng, (N, K // 32), emin=8, emax=14)
    qs = rng.integers(-128, 128, size=(N, K // 32, 32))
    b8 = sm.nc.make_q8_0_blocks(d16, qs)
    h8 = dev.upload_blocks("q8_0", torch.from_numpy(b8), N, K)
    ns = K // 256
    b4 = sm.nc.make_q4_K_blocks(sm.nc.rand_fp16(rng, (N, ns), emin=8, emax=14), sm.nc.rand_fp16(rng, (N, ns), emin=8, emax=14),
                                rng.integers(0, 64, size=(N, ns, 8)), rng.integers(0, 64, size=(N, ns, 8)),
                                rng.integers(0, 16, size=(N, ns, 256)))
    h4 = dev.upload_blocks("q4_k", torch.from_numpy(b4), N, K)
    x = _rand((T, K), 51, scale=2.0)
    try:
        assert dev.info(h8)["fmt"] == "q8_0" and dev.info(h4)["fmt"] == "q4_k"
        assert np.array_equal(dev.read_blocks(h8).numpy().reshape(-1), b8)
        assert np.array_equal(dev.read_blocks(h4).numpy().reshape(-1), b4)
        y8, y4 = dev.linear(h8, x), dev.linear(h4, x)
        assert _same_bits(y8, dev.linear_ref(h8, x)) and _same_bits(y4, dev.linear_ref(h4, x))
        assert np.array_equal(_bits(y8), sm.spec_y(b8.reshape(N, -1), x.numpy(), N, K).view(np.int32))
        wp = sm.nc.parse_q4_K(b4.reshape(N, -1), N, K)
        y4_np = sm.nc.spec_model_y(sm.nc.FMT_Q4_K, wp, sm.nc.np_qact(sm.nc.QA_A256, x.numpy()), T)
        assert np.array_equal(_bits(y4), y4_np.view(np.int32))
    finally:
        dev.free(h8)
        dev.free(h4)


def test_errors(dev):
    with pytest.raises(ValueError, match="multiple of 32"):
        dev.upload(torch.zeros(4, 33))
    with pytest.raises(ValueError, match="not supported"):
        dev.upload(torch.zeros(4, 32, dtype=torch.float64))
    with pytest.raises(ValueError, match="2D"):
        dev.upload(torch.zeros(32))
    assert dev.unsupported_reason(torch.float16, 8, 48) is not None
    assert dev.unsupported_reason("bf16", 8, 64) is None
    h = dev.upload(torch.ones(8, 64))
    with pytest.raises(ValueError, match="columns"):
        dev.linear(h, torch.ones(2, 32))
    with pytest.raises(ValueError, match="float32"):
        dev.linear(h, torch.ones(2, 64, dtype=torch.float16))
    assert tuple(dev.linear(h, torch.ones(0, 64)).shape) == (0, 8)
    dev.free(h)
    with pytest.raises(ValueError, match="unknown weight handle"):
        dev.free(h)
    with pytest.raises(ValueError, match="unknown weight handle"):
        dev.linear(h, torch.ones(1, 64))
    assert isinstance(_C.TesseractError("x"), RuntimeError)
