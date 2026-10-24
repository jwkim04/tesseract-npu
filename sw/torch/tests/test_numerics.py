"""NMSE of the device against torch's own float32 matmul on random data (test-backend-ops' metric and
limit, 5e-4), and the known single-output exception (docs/spec.md 검증 전략, "연산 단위 NMSE의 예외").

Run with ``-s`` to see the measured distribution."""
from __future__ import annotations

import zlib

import numpy as np
import pytest
import torch

import spec_model as sm
import tesseract_torch as tt

NMSE_MAX = 5e-4
# (T, N, K): weight-matmul shapes like test-backend-ops' (m = N >= 16, n = T, k = K) and Qwen2.5-0.5B's.
# Shapes with very few outputs (N * T < 16) are not in the table: their NMSE is the mean of a few squared
# relative errors and behaves like the single-output case below.
SHAPES = [
    (1, 16, 256), (4, 16, 256), (1, 64, 4096), (8, 128, 1024), (16, 896, 896), (1, 4864, 896),
    (3, 896, 4864), (32, 256, 2048), (7, 100, 96), (64, 64, 64), (1, 16, 2048), (2, 32, 32),
]
DTYPES = [torch.float32, torch.float16, torch.bfloat16]
SEEDS = range(5)


def _uniform(shape, g, dtype=torch.float32):
    return (torch.rand(shape, generator=g, dtype=torch.float64) * 2 - 1).to(dtype)


def _nmse(y: torch.Tensor, ref: torch.Tensor) -> float:
    return sm.nmse(y.double().numpy(), ref.double().numpy())


def test_nmse_distribution(dev, capsys):
    rows = []
    allv = []
    for (T, N, K) in SHAPES:
        for dtype in DTYPES:
            vals = []
            for s in SEEDS:
                g = torch.Generator().manual_seed(zlib.crc32(repr((T, N, K, str(dtype), s)).encode()))
                w = _uniform((N, K), g, dtype)
                x = _uniform((T, K), g)
                h = dev.upload(w)
                try:
                    y = dev.linear(h, x)
                    assert torch.equal(y, dev.linear_ref(h, x))
                finally:
                    dev.free(h)
                ref = x @ w.float().t()                     # torch float32 (CPU BLAS)
                vals.append(_nmse(y, ref))
            rows.append(((T, N, K), str(dtype).replace("torch.", ""), min(vals), float(np.median(vals)), max(vals)))
            allv += vals
    with capsys.disabled():
        print(f"\nNMSE vs torch float32 x @ W.T ({len(SEEDS)} seeds per row, uniform [-1, 1]):")
        print(f"  {'T':>3} {'N':>5} {'K':>5}  {'dtype':<9} {'min':>9} {'median':>9} {'max':>9}")
        for (T, N, K), dt, lo, med, hi in rows:
            print(f"  {T:>3} {N:>5} {K:>5}  {dt:<9} {lo:9.2e} {med:9.2e} {hi:9.2e}")
        q = np.quantile(allv, [0, 0.5, 0.9, 0.99, 1])
        print(f"  all {len(allv)} cases: min {q[0]:.2e}  median {q[1]:.2e}  p90 {q[2]:.2e}  p99 {q[3]:.2e}  "
              f"max {q[4]:.2e}  (limit {NMSE_MAX:.0e}, max = {q[4] / NMSE_MAX:.1%} of it)")
    for (T, N, K), dt, lo, med, hi in rows:
        assert hi < NMSE_MAX, f"T={T} N={N} K={K} {dt}: NMSE {hi:.3e}"


def test_single_output_exception(dev, capsys):
    """N = T = 1 (test-backend-ops m=1, n=1, k=2048, f32 x f32): one output, so the NMSE is one squared
    relative error and exceeds 5e-4 whenever |y| is small. Here the device must be bit-exact with the
    reference model on every trial, and its failures must be exactly those of the float64 product
    of the same 8-bit operands (the quantization, not the device arithmetic, causes them)."""
    K, trials = 2048, 1000
    fails_dev, fails_exact = set(), set()
    vals = []
    for i in range(trials):
        g = torch.Generator().manual_seed(1_000_000 + i)
        w = _uniform((1, K), g)
        x = _uniform((1, K), g)
        h = dev.upload(w)
        try:
            y = dev.linear(h, x)
            assert torch.equal(y, dev.linear_ref(h, x)), f"trial {i}: device != reference"
            blocks = dev.read_blocks(h).numpy()
        finally:
            dev.free(h)
        ref = x @ w.t()
        e = _nmse(y, ref)
        vals.append(e)
        if e >= NMSE_MAX:
            fails_dev.add(i)
        if sm.nmse(sm.exact_y(blocks, x.numpy(), 1, K), ref.double().numpy()) >= NMSE_MAX:
            fails_exact.add(i)
    with capsys.disabled():
        q = np.quantile(vals, [0.5, 0.9, 1])
        print(f"\nsingle output (N = T = 1, K = {K}, f32), {trials} seeded trials: NMSE >= {NMSE_MAX:.0e} in "
              f"{len(fails_dev)} ({len(fails_dev) / trials:.1%}); exact 8-bit product: {len(fails_exact)}; "
              f"same trials: {fails_dev == fails_exact}; NMSE median {q[0]:.2e} p90 {q[1]:.2e} max {q[2]:.2e}")
    assert fails_dev == fails_exact
    assert 0.05 < len(fails_dev) / trials < 0.30   # docs/llama-pin.md: 15.4% of 10000 trials


@pytest.mark.parametrize("dtype", DTYPES)
def test_linear_api_matches_reference(dtype):
    """tesseract_torch.linear (cache, flattening, bias, dtype) == linear_ref bit for bit."""
    g = torch.Generator().manual_seed(9)
    w = _uniform((96, 256), g, dtype)
    b = _uniform((96,), g, dtype)
    x = _uniform((2, 3, 256), g, dtype)
    with torch.no_grad():
        y = tt.linear(x, w, b)
        y_ref = tt.linear_ref(x, w, b)
    assert y.dtype == dtype and tuple(y.shape) == (2, 3, 96)
    assert torch.equal(y.view(torch.int16 if dtype != torch.float32 else torch.int32),
                       y_ref.view(torch.int16 if dtype != torch.float32 else torch.int32))
    exact = torch.nn.functional.linear(x.double(), w.double(), b.double())
    assert _nmse(y.float(), exact.float()) < NMSE_MAX
