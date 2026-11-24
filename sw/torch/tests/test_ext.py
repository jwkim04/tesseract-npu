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


# ---------------------------------------------------------------------------------------------
# KV objects and cached attention (the graph API with kv_slots)
# ---------------------------------------------------------------------------------------------
# Measured on the simulator (the shapes and seeds below): the relative NMSE of the cached steps against the stateless run of the same
# rows is 0 for a causal mask and at most 3.2e-15 for a given mask; a run at a wrong pos0 is about 2e-2. The gate is 1e-10.
NMSE_CACHED = 1e-10


def _nmse(y: torch.Tensor, ref: torch.Tensor) -> float:
    y, ref = y.double(), ref.double()
    den = float((ref ** 2).sum())
    return float(((y - ref) ** 2).sum()) / den if den > 0 else float(((y - ref) ** 2).sum())


def _att_prog(nh, nkv, d, cells=None, mask=False, kv_size=0):
    """One attn op: in q, k, v [, mask] -> out. cells: a cached attention over a KV slot of that many cells."""
    values = [{"kind": "in", "n": nh * d}, {"kind": "in", "n": nkv * d}, {"kind": "in", "n": nkv * d}]
    if mask:
        values.append({"kind": "mask", "cols": cells} if cells else {"kind": "mask"})
    values.append({"kind": "tmp", "n": nh * d, "out": True})
    op = {"op": "attn", "q": 0, "k": 1, "v": 2, "dst": len(values) - 1, "n_head": nh, "n_head_kv": nkv, "d": d,
          "scale": 1.0 / float(d) ** 0.5}
    if mask:
        op["mask"] = 3
    prog = {"values": values, "ops": [op], "kv_size": kv_size}
    if cells:
        op["kv"] = 0
        prog["kv_slots"] = [cells]
    return prog


def _cached_mask(T, pos0, cells, seed, holes=0.0):
    g = torch.Generator().manual_seed(seed)
    m = torch.full((T, cells), float("-inf"))
    for t in range(T):
        m[t, : pos0 + t + 1] = 0.0
        if holes:
            hole = torch.rand(pos0 + t, generator=g) < holes
            m[t, : pos0 + t][hole] = float("-inf")
    return m


def _tmax_device(monkeypatch, tmax):
    if tt.device().transport != "sim":
        pytest.skip("needs the sim transport")
    if tmax:
        monkeypatch.setenv("TSR_SIM_TMAX", str(tmax))
    return _C.Device("sim")


def test_kv_object_store_read(dev):
    """kv_create / kv_store / kv_read / kv_info / kv_free: the stored cells are the q8_0 blocks of an independent quantizer."""
    E, cells = 128, 100
    free0 = dev.memory()["weight_free"]
    dev.reset_counters()
    h = dev.kv_create(E, cells)
    try:
        info = dev.kv_info(h)
        assert (info["E"], info["cells"]) == (E, cells) and info["bytes"] > 0
        assert info["k_off"] % 4096 == 0 and info["v_off"] % 4096 == 0 and info["k_off"] != info["v_off"]
        assert dev.memory()["weight_free"] <= free0 - info["bytes"]
        k0, v0 = dev.kv_read(h, 0, cells)
        assert k0.dtype == torch.uint8 and tuple(k0.shape) == (cells, E // 32 * 34) and tuple(v0.shape) == tuple(k0.shape)
        assert not k0.any() and not v0.any(), "a new object is zero-filled"
        n, c0 = 40, 13
        base_k, base_v = _rand((n, E + 7), 3, scale=2.0), _rand((n, E + 7), 4, scale=2.0)
        k, v = base_k[:, :E], base_v[:, :E]                      # a row stride above E
        assert k.stride(0) == E + 7
        dev.kv_store(h, c0, k, v)
        kb, vb = dev.kv_read(h, c0, n)
        assert np.array_equal(kb.numpy(), sm.q8_0_ref_blocks(k.contiguous().numpy()))
        assert np.array_equal(vb.numpy(), sm.q8_0_ref_blocks(v.contiguous().numpy()))
        ka, va = dev.kv_read(h, 0, cells)
        outside = np.ones(cells, bool)
        outside[c0:c0 + n] = False
        assert not ka.numpy()[outside].any() and not va.numpy()[outside].any(), "the other cells stay zero"
        # one row, overwriting; an empty store is a no-op
        k1, v1 = _rand((1, E), 5, scale=9.0), _rand((1, E), 6, scale=9.0)
        dev.kv_store(h, c0 + 3, k1, v1)
        dev.kv_store(h, 0, torch.zeros(0, E), torch.zeros(0, E))
        kb2, vb2 = dev.kv_read(h, c0 + 3, 1)
        assert np.array_equal(kb2.numpy(), sm.q8_0_ref_blocks(k1.numpy())) and np.array_equal(vb2.numpy(), sm.q8_0_ref_blocks(v1.numpy()))
        kb3, _ = dev.kv_read(h, c0, n)
        assert np.array_equal(kb3.numpy()[:3], kb.numpy()[:3]) and np.array_equal(kb3.numpy()[4:], kb.numpy()[4:])
        c = dev.counters()
        assert (c["kv_creates"], c["kv_stores"], c["kv_store_rows"]) == (1, 2, n + 1)
        # errors
        with pytest.raises(ValueError, match="beyond"):
            dev.kv_store(h, cells - 1, k[:2], v[:2])
        with pytest.raises(ValueError, match="beyond"):
            dev.kv_read(h, cells - 1, 2)
        with pytest.raises(ValueError, match="same shape"):
            dev.kv_store(h, 0, k[:3], v[:2])
        with pytest.raises(ValueError, match="float32"):
            dev.kv_store(h, 0, k[:2].half(), v[:2].half())
        with pytest.raises(ValueError, match="KV object E"):
            dev.kv_store(h, 0, torch.zeros(2, 64), torch.zeros(2, 64))
        with pytest.raises(ValueError, match="multiple of 64"):
            dev.kv_create(100, 16)
        with pytest.raises(ValueError, match="cells"):
            dev.kv_create(64, 0)
        with pytest.raises(_C.TesseractError, match="does not fit"):
            dev.kv_create(511 * 64, 16 * 16 * 65535)
    finally:
        dev.kv_free(h)
    assert dev.memory()["weight_free"] == free0
    for f in (lambda: dev.kv_free(h), lambda: dev.kv_info(h), lambda: dev.kv_read(h, 0, 1), lambda: dev.kv_store(h, 0, k, v)):
        with pytest.raises(ValueError, match="unknown KV handle"):
            f()


def test_kv_arguments_are_checked(dev):
    """A handle that is negative or above 64 bits, a bool where an int is meant, and cells beyond the planes are ValueErrors."""
    E, cells = 64, 32
    h = dev.kv_create(E, cells)
    hg = dev.graph_create(_att_prog(2, 1, 64, cells))
    try:
        a = [_rand((1, 128), 41), _rand((1, E), 42), _rand((1, E), 43)]
        for bad in (-1, 2 ** 64, 2 ** 70, -(2 ** 70)):
            for run in (dev.graph_run, dev.graph_run_ref):
                with pytest.raises(ValueError, match="kv handle"):
                    run(hg, a, kv=[bad], pos0=0)
            for f in (lambda: dev.kv_free(bad), lambda: dev.kv_info(bad), lambda: dev.kv_read(bad, 0, 1),
                      lambda: dev.kv_store(bad, 0, torch.zeros(1, E), torch.zeros(1, E))):
                with pytest.raises(ValueError, match="handle"):
                    f()
        with pytest.raises(ValueError, match="bool"):
            dev.graph_run(hg, a, kv=[h], pos0=True)
        with pytest.raises(ValueError, match="bool"):
            dev.graph_run_ref(hg, a, kv=[h], pos0=True)
        with pytest.raises(ValueError, match="bool"):
            dev.kv_create(True, cells)
        with pytest.raises(ValueError, match="bool"):
            dev.kv_create(E, True)
        with pytest.raises(ValueError, match="bool"):
            dev.kv_read(h, True, 1)
        with pytest.raises(ValueError, match="bool"):
            dev.kv_read(h, 0, True)
        with pytest.raises(ValueError, match="bool"):
            dev.kv_store(h, True, torch.zeros(1, E), torch.zeros(1, E))
        for f in (lambda: dev.kv_free(True), lambda: dev.kv_info(True), lambda: dev.kv_read(True, 0, 1)):
            with pytest.raises(ValueError, match="bool"):
                f()
        with pytest.raises(ValueError, match="int"):
            dev.kv_read(h, 0.5, 1)
        # a run with a numpy int for pos0 and a numpy handle behaves like the Python ints
        assert dev.kv_read(h, np.int64(0), np.int64(1))[0].shape == (1, E // 32 * 34)
        # no rows: cell0 is still checked against the cells
        assert dev.kv_read(h, cells, 0)[0].shape == (0, E // 32 * 34)
        assert dev.kv_read(h, 0, 0)[0].shape == (0, E // 32 * 34)
        with pytest.raises(ValueError, match="beyond"):
            dev.kv_read(h, cells + 1, 0)
        with pytest.raises(ValueError, match="beyond"):
            dev.kv_store(h, cells + 1, torch.zeros(0, E), torch.zeros(0, E))
        dev.kv_store(h, cells, torch.zeros(0, E), torch.zeros(0, E))
    finally:
        dev.graph_free(hg)
        dev.kv_free(h)


def test_graph_info_of_a_graph_that_runs_in_chunks(dev):
    """graph_info(h, T) is the plan the first chunk of a run of T rows starts with; pos0 plays no part."""
    nh, nkv, d, cells = 2, 1, 64, 800
    E = nkv * d
    h = dev.graph_create(_att_prog(nh, nkv, d, cells))
    kvh = dev.kv_create(E, cells)
    try:
        R = 700
        dev.reset_counters()
        dev.graph_run(h, [_rand((R, nh * d), 25), _rand((R, E), 26), _rand((R, E), 27)], kv=[kvh], pos0=0)
        misses = dev.counters()["graph_plan_misses"]
        info = dev.graph_info(h, R)
        assert info["T"] == 680 and info["T"] < R
        assert {k: v for k, v in info.items()} == dev.graph_info(h, 680)
        assert dev.counters()["graph_plan_misses"] == misses, "no second plan for what the run planned"
        for key in ("T", "records", "lists", "by_kind", "io_bytes"):
            assert key in info
        # a run at the last cell leaves graph_info of two rows answerable
        dev.graph_run(h, [_rand((1, nh * d), 28), _rand((1, E), 29), _rand((1, E), 30)], kv=[kvh], pos0=cells - 1)
        assert dev.graph_info(h, 2)["T"] == 2
        with pytest.raises(_C.TesseractError, match="does not fit"):
            dev.graph_info(h, cells + 1)
    finally:
        dev.graph_free(h)
        dev.kv_free(kvh)


@pytest.mark.parametrize("given", [False, True])
@pytest.mark.parametrize("request_tmax", [0, 4, 8])
def test_cached_attention_program(monkeypatch, given, request_tmax):
    """A cached attention through the bindings: device == reference, the cells are the quantizer's blocks, the steps are the
    stateless run of the same rows (tiny NMSE), a wrong pos0 is not."""
    dev = tt.device() if request_tmax == 0 else _tmax_device(monkeypatch, request_tmax)
    nh, nkv, d, cells = 4, 2, 64, 64
    E = nkv * d
    steps = [5, 1, 1, 3, 1]
    R = sum(steps)
    h = h2 = kvh = None
    try:
        h = dev.graph_create(_att_prog(nh, nkv, d, cells, mask=given))
        sh = dev.graph_shape(h)
        assert sh["kv_slots"] == [cells] and sh["kv_bytes"] == 0
        assert [a["kind"] for a in sh["args"]] == ["in", "in", "in"] + (["mask"] if given else [])
        if given:
            assert sh["args"][3]["n"] == cells
        kvh = dev.kv_create(E, cells)
        q, k, v = _rand((R, nh * d), 11, scale=2.0), _rand((R, E), 12, scale=2.0), _rand((R, E), 13, scale=2.0)
        M = _cached_mask(R, 0, cells, 14, holes=0.2 if given else 0.0)
        pos0, outs = 0, []
        for T in steps:
            sl = slice(pos0, pos0 + T)
            args = [q[sl], k[sl], v[sl]] + ([M[sl].reshape(1, 1, T, cells)] if given else [])
            y = dev.graph_run(h, args, kv=[kvh], pos0=pos0)[0]
            yr = dev.graph_run_ref(h, args, kv=[kvh], pos0=pos0)[0]
            assert _same_bits(y, yr), f"step pos0 {pos0}: device != reference"
            outs.append(y)
            pos0 += T
        got = torch.cat(outs)
        kb, vb = dev.kv_read(kvh, 0, R)
        assert np.array_equal(kb.numpy(), sm.q8_0_ref_blocks(k.numpy())) and np.array_equal(vb.numpy(), sm.q8_0_ref_blocks(v.numpy()))
        # the stateless run of the same rows (own planes, [R, R] mask)
        h2 = dev.graph_create(_att_prog(nh, nkv, d, None, mask=given, kv_size=R))
        args2 = [q, k, v] + ([M[:, :R].reshape(1, 1, R, R).contiguous()] if given else [])
        ref = dev.graph_run(h2, args2)[0]
        assert _nmse(got, ref) < NMSE_CACHED, _nmse(got, ref)
        print(f"cached steps vs stateless run (given mask {given}, T_MAX {request_tmax}): NMSE {_nmse(got, ref):.3g}")
        if not given:
            # a causal run at another pos0 (rows 0 .. 6 of the cache are earlier cells for it) is another result: the control of the above
            wrong = dev.graph_run(h, [q[:3], k[:3], v[:3]], kv=[kvh], pos0=7)[0]
            right = dev.graph_run(h, [q[:3], k[:3], v[:3]], kv=[kvh], pos0=0)[0]
            assert _nmse(wrong, right) > 1e-4
    finally:
        for g in (h, h2):
            if g is not None:
                dev.graph_free(g)
        if kvh is not None:
            dev.kv_free(kvh)
        if request_tmax:
            dev.close()


def test_cached_attention_refusals_and_program_errors(dev):
    nh, nkv, d, cells = 2, 1, 64, 32
    E = nkv * d
    h = dev.graph_create(_att_prog(nh, nkv, d, cells, mask=True))
    kvh = dev.kv_create(E, cells)
    try:
        q, k, v = _rand((2, nh * d), 21), _rand((2, E), 22), _rand((2, E), 23)
        ok = _cached_mask(2, 10, cells, 24)
        dev.graph_run(h, [q, k, v, ok], kv=[kvh], pos0=10)
        with pytest.raises(_C.TesseractError, match="does not fit"):
            dev.graph_run(h, [q, k, v, ok], kv=[kvh], pos0=31)                  # pos0 + T above the cells
        bad = ok.clone()
        bad[1, 20] = 0.0                                                        # a cell at or above pos0 + T
        with pytest.raises(_C.TesseractError, match=r"does not fit.*mask row 1 shows cell 20|mask row 1 shows cell 20.*does not fit"):
            dev.graph_run(h, [q, k, v, bad], kv=[kvh], pos0=10)
        with pytest.raises(ValueError, match=r"\[T, 32\]"):
            dev.graph_run(h, [q, k, v, torch.zeros(2, 2)], kv=[kvh], pos0=10)   # the [T, T] form of an own-planes mask
        with pytest.raises(ValueError, match="KV handles given"):
            dev.graph_run(h, [q, k, v, ok], pos0=10)
        with pytest.raises(ValueError, match="KV handles given"):
            dev.graph_run(h, [q, k, v, ok], kv=[kvh, kvh], pos0=10)
        with pytest.raises(ValueError, match="unknown KV handle"):
            dev.graph_run(h, [q, k, v, ok], kv=[kvh + 1000], pos0=10)
        with pytest.raises(ValueError, match="list of KV handles"):
            dev.graph_run(h, [q, k, v, ok], kv=kvh, pos0=10)
        other = dev.kv_create(E, cells + 16)
        try:
            with pytest.raises(ValueError, match="KV slot 0 needs E"):
                dev.graph_run(h, [q, k, v, ok], kv=[other], pos0=10)
        finally:
            dev.kv_free(other)
        # the bool form of a cached mask is taken
        yb = dev.graph_run(h, [q, k, v, (ok == 0).reshape(1, 1, 2, cells)], kv=[kvh], pos0=10)[0]
        yf = dev.graph_run(h, [q, k, v, ok], kv=[kvh], pos0=10)[0]
        assert _same_bits(yb, yf)
        # T above graph_max_attn_tokens runs in chunks for a graph whose attn ops all have a slot
        c0 = dev.counters()
        big = dev.kv_create(E, 800)
        try:
            hb = dev.graph_create(_att_prog(nh, nkv, d, 800))
            try:
                R = 700
                ys = dev.graph_run(hb, [_rand((R, nh * d), 25), _rand((R, E), 26), _rand((R, E), 27)], kv=[big], pos0=0)[0]
                assert ys.shape == (R, nh * d) and torch.isfinite(ys).all()
                c1 = dev.counters()
                assert c1["graph_runs"] - c0["graph_runs"] == 1 and c1["graph_launches"] - c0["graph_launches"] >= 2
                with pytest.raises(_C.TesseractError, match="does not fit"):
                    dev.graph_run(hb, [_rand((R, nh * d), 25), _rand((R, E), 26), _rand((R, E), 27)], kv=[big], pos0=101)
            finally:
                dev.graph_free(hb)
        finally:
            dev.kv_free(big)
        # graph_info keeps working for a graph with slots
        assert dev.graph_info(h, 2)["lists"] >= 1
    finally:
        dev.graph_free(h)
        dev.kv_free(kvh)
    # a graph without slots takes no handles
    hn = dev.graph_create(_att_prog(nh, nkv, d, None, kv_size=8))
    try:
        a = [_rand((2, nh * d), 28), _rand((2, E), 29), _rand((2, E), 30)]
        assert dev.graph_shape(hn)["kv_slots"] == []
        dev.graph_run(hn, a)
        dev.graph_run(hn, a, kv=None, pos0=5)
        kvh = dev.kv_create(E, 16)
        try:
            with pytest.raises(ValueError, match="no KV slots"):
                dev.graph_run(hn, a, kv=[kvh])
        finally:
            dev.kv_free(kvh)
    finally:
        dev.graph_free(hn)
    # the program dict
    def bad(edit, match):
        p = _att_prog(nh, nkv, d, cells, mask=True)
        edit(p)
        with pytest.raises(ValueError, match=match):
            dev.graph_unsupported(p)
    bad(lambda p: p.update(kv_slots=cells), "'kv_slots' must be a list")
    bad(lambda p: p.update(kv_slots=[True]), "must be an int")
    bad(lambda p: p.update(kv_slots=[0]), "out of range")
    bad(lambda p: p["ops"][0].update(kv="0"), "must be an int")
    bad(lambda p: p["values"][3].update(cols="x"), "must be an int")
    bad(lambda p: p["values"][3].update(extra=1), "unknown|extra")
    with pytest.raises(ValueError, match="out of range"):
        p = _att_prog(nh, nkv, d, cells, mask=True)
        p["ops"][0]["kv"] = 1
        dev.graph_create(p)
    with pytest.raises(ValueError, match="cols = 7"):
        p = _att_prog(nh, nkv, d, cells, mask=True)
        p["values"][3]["cols"] = 7
        dev.graph_create(p)
