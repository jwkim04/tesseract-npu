"""Independent numpy models for the tests (no libtsr_ref, no ggml code):

- ``q8_0_ref_blocks``: ggml's ``quantize_row_q8_0_ref`` written from its definition (amax / 127,
  id = 1 / d, q = roundf(x * id) half away from zero, d stored as fp16 RNE) -> ggml block_q8_0 bytes.
- ``spec_y``: the spec's arithmetic for a Q8_0 weight (docs/spec.md 수치 규칙: QACT A32, block terms,
  16 partial sums + fixed tree, DAZ / FTZ) from sw/tools/numcheck.py's numpy model (B1b), which was
  written from the spec text only.
- ``exact_y``: the float64 value of the quantized product (dequantized weight blocks x dequantized
  QA), i.e. the same 8-bit operands without any fp32 rounding.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "sw" / "tools"))
import numcheck as nc  # noqa: E402

F32 = np.float32


def q8_0_ref_blocks(w: np.ndarray) -> np.ndarray:
    """w [N, K] float32 (K % 32 == 0) -> uint8 [N, K / 32 * 34] (ggml block_q8_0: fp16 d, int8 qs[32])."""
    w = np.ascontiguousarray(w, dtype=F32)
    N, K = w.shape
    nb = K // 32
    xb = w.reshape(N, nb, 32)
    amax = np.abs(xb).max(axis=2)                                  # f32, exact
    with np.errstate(all="ignore"):
        d = (amax / F32(127)).astype(F32)                           # one f32 division (RNE)
        idv = np.where(d != 0, F32(1) / d, F32(0)).astype(F32)
        x0 = (xb * idv[..., None]).astype(F32)                      # f32 product
    x0d = x0.astype(np.float64)                                     # exact
    q = np.copysign(np.floor(np.abs(x0d) + 0.5), x0d)               # roundf: half away from zero
    q = q.astype(np.int8)
    out = np.empty((N, nb, 34), np.uint8)
    out[:, :, 0:2] = d.astype(np.float16).view(np.uint8).reshape(N, nb, 2)
    out[:, :, 2:34] = q.view(np.uint8)
    return out.reshape(N, nb * 34)


def spec_y(blocks: np.ndarray, x: np.ndarray, N: int, K: int) -> np.ndarray:
    """y [T, N] float32 of the spec model for Q8_0 blocks [N, K/32*34] and x [T, K] float32."""
    x = np.ascontiguousarray(x, dtype=F32)
    T = x.shape[0]
    wp = nc.parse_q8_0(np.ascontiguousarray(blocks), N, K)
    qa = nc.np_qact(nc.QA_A32, x)
    return nc.spec_model_y(nc.FMT_Q8_0, wp, qa, T)


def exact_y(blocks: np.ndarray, x: np.ndarray, N: int, K: int) -> np.ndarray:
    """float64 [T, N]: sum over blocks of d_w * D_x * (q_w . q_x), exactly (no fp32 rounding)."""
    x = np.ascontiguousarray(x, dtype=F32)
    dw, qw = nc.parse_q8_0(np.ascontiguousarray(blocks), N, K)       # dw [N, nb] f64 (exact), qw [N, nb, 32]
    qx, Dx, _ = nc.np_qact(nc.QA_A32, x)                              # qx [T, K], Dx [T, nb] f32
    nb = K // 32
    isum = np.einsum("nbj,tbj->tnb", qw.astype(np.float64), qx.reshape(-1, nb, 32).astype(np.float64))
    return np.einsum("tnb,nb,tb->tn", isum, dw, Dx.astype(np.float64))


def nmse(y: np.ndarray, ref: np.ndarray) -> float:
    """test-backend-ops' NMSE: sum (y - ref)^2 / sum ref^2 (float64)."""
    y = np.asarray(y, np.float64)
    ref = np.asarray(ref, np.float64)
    den = float((ref * ref).sum())
    return float(((y - ref) ** 2).sum()) / den if den > 0 else float(((y - ref) ** 2).sum())
