#!/usr/bin/env python3
"""numcheck.py - numeric verification of the Tesseract backend on one prompt (docs/numcheck.md).

Subcommands
  compare    --cpu DIR --floor DIR --dev DIR [-m MODEL] -o OUT   -> OUT/report.json, OUT/report.md
             (A) drift of every dumped node, dev vs cpu against the 1-ulp floor, logits KL / top-1
             (B) every weight MUL_MAT of the dev run:
                 B1  y_dev == libtsr_ref model (bit for bit, every element)
                 B1b y_dev == independent numpy model of the spec text (spec_model_y; bit for bit)
                 B2  |y_dev - y_q64| <= proven fp32 forward-error bound (y_q64 = fp64 value of the same
                     quantized operands; activation quantized by the independent numpy QACT model)
                 B3  ||y - W x|| / ||W x|| with W = ggml to_float of the original GGUF weight, dev vs cpu,
                     tokens holding > 50 % of ||W x||^2 listed and left out of the ratio
             exit 0 PASS, 1 FAIL, 3 INCOMPLETE (a selected weight MUL_MAT not checked), 2 input error / crash
             --src1-from-producer: weight MUL_MATs without a src1 dump take x from the dumped producer node
                 (attn_norm-N, kqv_out-N or with FA the FLASH_ATTN_EXT node, ffn_norm-N, ffn_swiglu-N, result_norm) after the mapping is verified
                 bit for bit on every src1 dump
  boundtest  >= 1e6 random / adversarial dot products: the B2 bound vs libtsr_ref (zero violations),
             the numpy spec model vs libtsr_ref (bit-exact; sequential-sum mutant must differ),
             plus the numpy QACT model and fp32 primitives vs libtsr_ref (bit-exact)
  selftest   boundtest + a synthetic run set (tiny GGUF + fake dumps) through compare, with injected
             faults that must be caught

Dump format (written by tsr-numcheck, docs/numcheck.md): <run>/meta.json {"format": "tsr-numcheck",
"version": 2, "run", "model", "n_tokens", "n_vocab", "logits_file", "info", "nodes": [{"i", "name", "op", "ne",
"file", "backend", "weight", "weight_type", "src1_file", "src1_ne"}]}, little-endian f32 files. Version 2
(P3.3) records the settings in "info" (SW-22: "phase" m6|m7, "flash_attn", "type_k", "type_v",
"offload_kqv", "attn_rot_disable"); compare requires the same settings in every run and reports them.
Version 1 dumps (FA-off settings: FA off, f16 KV, no rotation setting) are still read.

Native code: build/numcheck/sw/sim/libtsr_ref.so and build/numcheck/sw/tools/libnumcheck_helper.so
(cmake --build build/numcheck --target numcheck_helper tsr_ref_shared -j4).
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse  # noqa: E402
import ctypes  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DEFAULT_BUILD = REPO / "build" / "numcheck"
GGUF_PY = REPO / "third_party" / "llama.cpp" / "gguf-py"

F32 = np.float32
F64 = np.float64
U = 2.0 ** -24          # binary32 unit roundoff (round to nearest even)
U64 = 2.0 ** -53        # binary64 unit roundoff
LAM = 2.0 ** -126       # one flushed (FTZ) result is off by less than 2^-126
FMT_Q8_0, FMT_Q4_K = 0, 1
QA_A32, QA_A256 = 0, 1
FMT_NAME = {FMT_Q8_0: "q8_0", FMT_Q4_K: "q4_K"}
T_F32, T_F16, T_Q8_0, T_Q4_K, T_Q6_K = 0, 1, 8, 12, 14
BOUND_VERSION = 3       # bump when the boundtest changes (2: B2 bound formula; 3: + numpy spec model, B1b)
NOT_SAMPLED = "no src1 dump in the dev run (not in the dumper's --mm-layers)"


def die(msg: str) -> None:
    print(f"numcheck.py: error: {msg}", file=sys.stderr)
    sys.exit(2)


def log(msg: str) -> None:
    print(msg, flush=True)


def gamma(n: float, u: float = U) -> float:
    return n * u / (1.0 - n * u)


# =============================================================================================
# native libraries (libtsr_ref.so + libnumcheck_helper.so)
# =============================================================================================
class Native:
    def __init__(self, build: Path):
        ref_path = build / "sw" / "sim" / "libtsr_ref.so"
        nc_path = build / "sw" / "tools" / "libnumcheck_helper.so"
        for p in (ref_path, nc_path):
            if not p.exists():
                die(f"{p} missing: cmake --build {build} --target numcheck_helper tsr_ref_shared -j4")
        c = ctypes
        P = c.c_void_p
        self.ref = c.CDLL(str(ref_path))
        self.nc = c.CDLL(str(nc_path))
        nc, ref = self.nc, self.ref

        def sig(f, res, *args):
            f.restype = res
            f.argtypes = list(args)

        sig(nc.nc_abi_version, c.c_int)
        sig(nc.nc_type_count, c.c_int)
        sig(nc.nc_type_name, c.c_char_p, c.c_int)
        sig(nc.nc_blck_size, c.c_int64, c.c_int)
        sig(nc.nc_type_size, c.c_int64, c.c_int)
        sig(nc.nc_row_size, c.c_int64, c.c_int, c.c_int64)
        sig(nc.nc_has_to_float, c.c_int, c.c_int)
        sig(nc.nc_to_float, c.c_int, c.c_int, P, P, c.c_int64)
        sig(nc.nc_quantize_q8_0_ref, None, P, P, c.c_int64)
        sig(nc.nc_dev_fmt, c.c_int, c.c_int)
        sig(nc.nc_to_dev_blocks, c.c_int, c.c_int, P, c.c_int64, c.c_int64, P)
        sig(nc.nc_qa_lines, c.c_uint32, c.c_uint32, c.c_uint32)
        sig(nc.nc_qact, c.c_int, c.c_uint32, P, c.c_uint32, c.c_uint32, P)
        sig(nc.nc_model_mm, c.c_int, c.c_uint32, P, c.c_uint32, c.c_uint32, P, c.c_uint32, P, c.c_int)
        sig(nc.nc_model_mm_qa, c.c_int, c.c_uint32, P, c.c_uint32, c.c_uint32, P, c.c_uint32, P, c.c_int)
        if nc.nc_abi_version() != 1:
            die("libnumcheck_helper ABI mismatch (rebuild numcheck_helper)")
        for fn in ("tsr_fadd", "tsr_fsub", "tsr_fmul", "tsr_fdiv"):
            sig(getattr(ref, fn), c.c_float, c.c_float, c.c_float)
        sig(ref.tsr_frecip, c.c_float, c.c_float)
        sig(ref.tsr_f32_to_fp16, c.c_uint16, c.c_float)
        sig(ref.tsr_fp16_to_f32, c.c_float, c.c_uint16)
        sig(ref.tsr_bits_f32, c.c_float, c.c_uint32)
        sig(ref.tsr_f32_bits, c.c_uint32, c.c_float)
        sig(ref.tsr_i32_to_f32, c.c_float, c.c_int32)
        self.type_names = {i: nc.nc_type_name(i).decode() for i in range(nc.nc_type_count())}
        self.type_ids = {v: k for k, v in self.type_names.items()}

    @staticmethod
    def _p(a: np.ndarray):
        assert a.flags["C_CONTIGUOUS"]
        return a.ctypes.data_as(ctypes.c_void_p)

    def type_name(self, t: int) -> str:
        return self.type_names.get(t, f"type{t}")

    def row_size(self, t: int, k: int) -> int:
        return int(self.nc.nc_row_size(t, k))

    def dev_fmt(self, t: int) -> int:
        return int(self.nc.nc_dev_fmt(t))

    def to_float(self, t: int, raw: np.ndarray, n: int) -> np.ndarray:
        raw = np.ascontiguousarray(raw)
        out = np.empty(n, dtype=F32)
        r = self.nc.nc_to_float(t, self._p(raw), self._p(out), n)
        if r != 0:
            raise RuntimeError(f"to_float({self.type_name(t)}) failed: {r}")
        return out

    def quantize_q8_0(self, x: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(x, dtype=F32)
        out = np.empty(x.size // 32 * 34, dtype=np.uint8)
        self.nc.nc_quantize_q8_0_ref(self._p(x), self._p(out), x.size)
        return out

    def dev_blocks(self, t: int, raw: np.ndarray, nrows: int, K: int) -> tuple[int, np.ndarray]:
        raw = np.ascontiguousarray(raw)
        fmt = self.dev_fmt(t)
        bb = 34 * (K // 32) if fmt == FMT_Q8_0 else 144 * (K // 256)
        out = np.empty(nrows * bb, dtype=np.uint8)
        r = self.nc.nc_to_dev_blocks(t, self._p(raw), nrows, K, self._p(out))
        if r != fmt:
            raise RuntimeError(f"to_dev_blocks({self.type_name(t)}, K={K}) failed: {r}")
        return fmt, out

    def qa_bytes(self, kind: int, K: int) -> int:
        return int(self.nc.nc_qa_lines(kind, K)) * 64

    def qact(self, kind: int, x: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(x, dtype=F32)
        T, K = x.shape
        out = np.empty((T, self.qa_bytes(kind, K)), dtype=np.uint8)
        if self.nc.nc_qact(kind, self._p(x), K, T, self._p(out)) != 0:
            raise RuntimeError("nc_qact failed")
        return out

    def model_mm(self, fmt: int, blocks: np.ndarray, N: int, K: int, x: np.ndarray, threads: int) -> np.ndarray:
        x = np.ascontiguousarray(x, dtype=F32)
        T = x.shape[0]
        y = np.empty((T, N), dtype=F32)
        r = self.nc.nc_model_mm(fmt, self._p(np.ascontiguousarray(blocks)), N, K, self._p(x), T, self._p(y), threads)
        if r != 0:
            raise RuntimeError(f"nc_model_mm failed: {r}")
        return y

    def model_mm_qa(self, fmt: int, blocks: np.ndarray, N: int, K: int, qa: np.ndarray, threads: int) -> np.ndarray:
        qa = np.ascontiguousarray(qa, dtype=np.uint8)
        T = qa.shape[0]
        y = np.empty((T, N), dtype=F32)
        r = self.nc.nc_model_mm_qa(fmt, self._p(np.ascontiguousarray(blocks)), N, K, self._p(qa), T, self._p(y), threads)
        if r != 0:
            raise RuntimeError(f"nc_model_mm_qa failed: {r}")
        return y

    # scalar reference primitives (tests only)
    def f32(self, bits: int) -> float:
        return self.ref.tsr_bits_f32(bits)

    def bits(self, v: float) -> int:
        return int(self.ref.tsr_f32_bits(v))


# =============================================================================================
# independent numpy model of the spec's f32 rules and of QACT (spec quantization rules, common fp
# rules, implementation clarifications v0.2.1). Native IEEE binary32 numpy ops (RNE, gradual
# underflow) + explicit DAZ on inputs, FTZ and canonical NaN on results. Integer conversion via
# exact float64.
# =============================================================================================
_SIGN = np.uint32(0x80000000)
_EXP = np.uint32(0x7F800000)
_MANT = np.uint32(0x007FFFFF)
_CNAN = np.uint32(0x7FC00000)


def f32bits(x) -> np.ndarray:
    return np.asarray(x, dtype=F32).view(np.uint32)


def daz(x) -> np.ndarray:
    u = f32bits(x)
    return np.where((u & _EXP) == 0, u & _SIGN, u).astype(np.uint32).view(F32)


def fin(r) -> np.ndarray:
    u = f32bits(r)
    e = u & _EXP
    out = np.where(e == 0, u & _SIGN, u)
    out = np.where((e == _EXP) & ((u & _MANT) != 0), _CNAN, out)
    return out.astype(np.uint32).view(F32)


def fmul(a, b) -> np.ndarray:
    with np.errstate(all="ignore"):
        return fin(daz(a) * daz(b))


def fdiv(a, b) -> np.ndarray:
    with np.errstate(all="ignore"):
        return fin(daz(a) / daz(b))


def fadd(a, b) -> np.ndarray:
    with np.errstate(all="ignore"):
        return fin(daz(a) + daz(b))


def fsub(a, b) -> np.ndarray:
    with np.errstate(all="ignore"):
        return fin(daz(a) - daz(b))


def frecip(a) -> np.ndarray:
    with np.errstate(all="ignore"):
        return fin(F32(1.0) / daz(a))


def round_sat8(v, half_even: bool) -> np.ndarray:
    """f32 -> int8 as QACT: NaN -> 0, round (half even | half away from zero), saturate [-128, 127]."""
    w = np.asarray(v, dtype=F32).astype(F64)          # exact
    w = np.where(np.isnan(w), 0.0, w)
    with np.errstate(invalid="ignore"):
        r = np.rint(w) if half_even else np.copysign(np.floor(np.abs(w) + 0.5), w)
    return np.clip(r, -128.0, 127.0).astype(np.int8)


def fp16_round(d: np.ndarray) -> np.ndarray:
    """f32 -> fp16 (RNE) -> f32, numpy's own conversion."""
    with np.errstate(all="ignore"):
        return np.asarray(d, dtype=F32).astype(np.float16).astype(F32)


def np_qact(kind: int, x: np.ndarray):
    """QACT of T token vectors x[T][K] -> (q[T][K] int8, D[T][nb] f32, bsums[T][K/16] int16 | None)."""
    x = np.asarray(x, dtype=F32)
    T, K = x.shape
    bs = 32 if kind == QA_A32 else 256
    nb = K // bs
    xb = daz(x).reshape(T, nb, bs)
    with np.errstate(all="ignore"):
        mag = np.abs(xb)
    isn = np.isnan(mag)
    if kind == QA_A32:
        amax = np.where(isn, F32(0), mag).max(axis=2)
        d = fdiv(amax, F32(127))
        nz = (f32bits(d) & np.uint32(0x7FFFFFFF)) != 0
        idv = np.where(nz, frecip(d), F32(0)).astype(F32)
        q = round_sat8(fmul(xb, idv[..., None]), half_even=False)
        return q.reshape(T, K), fp16_round(d), None
    magc = np.where(isn, F32(-1), mag)
    idx = np.argmax(magc, axis=2)                       # first largest |x|
    amax = np.take_along_axis(magc, idx[..., None], 2)[..., 0]
    mx = np.take_along_axis(xb, idx[..., None], 2)[..., 0]
    nz = amax > 0
    iscale = fdiv(np.full_like(mx, F32(-127)), mx)
    q = round_sat8(fmul(iscale[..., None], xb), half_even=True)
    q = np.where(nz[..., None], q, np.int8(0)).astype(np.int8)
    d = np.where(nz, frecip(iscale), F32(0)).astype(F32)
    bsum = q.reshape(T, K // 16, 16).astype(np.int32).sum(-1).astype(np.int16)
    return q.reshape(T, K), d, bsum


def qa_layout(kind: int, K: int):
    bs = 32 if kind == QA_A32 else 256
    qs_l = -(-K // 64)
    d_l = -(-(4 * (K // bs)) // 64)
    bs_l = -(-(2 * (K // 16)) // 64) if kind == QA_A256 else 0
    return qs_l * 64, (qs_l + d_l) * 64, (qs_l + d_l + bs_l) * 64


def qa_image(kind: int, K: int, q, d, bsum) -> np.ndarray:
    T = q.shape[0]
    d_off, bs_off, size = qa_layout(kind, K)
    img = np.zeros((T, size), dtype=np.uint8)
    img[:, :K] = q.view(np.uint8)
    img[:, d_off:d_off + 4 * d.shape[1]] = np.ascontiguousarray(d, dtype="<f4").view(np.uint8).reshape(T, -1)
    if kind == QA_A256:
        img[:, bs_off:bs_off + 2 * bsum.shape[1]] = np.ascontiguousarray(bsum, dtype="<i2").view(np.uint8).reshape(T, -1)
    return img


def parse_qa_image(kind: int, K: int, img: np.ndarray):
    T = img.shape[0]
    bs = 32 if kind == QA_A32 else 256
    d_off, bs_off, _ = qa_layout(kind, K)
    q = np.ascontiguousarray(img[:, :K]).view(np.int8)
    d = np.ascontiguousarray(img[:, d_off:d_off + 4 * (K // bs)]).view("<f4").astype(F32).reshape(T, -1)
    bsum = None
    if kind == QA_A256:
        bsum = np.ascontiguousarray(img[:, bs_off:bs_off + 2 * (K // 16)]).view("<i2").astype(np.int16).reshape(T, -1)
    return q, d, bsum


# =============================================================================================
# weight blocks (ggml layout of the device format) -> ints and scales
# =============================================================================================
def _fp16_field(a: np.ndarray, off: int) -> np.ndarray:
    return np.ascontiguousarray(a[..., off:off + 2]).view("<f2")[..., 0].astype(F64)   # exact


def parse_q8_0(blocks: np.ndarray, N: int, K: int):
    a = blocks.reshape(N, K // 32, 34)
    return _fp16_field(a, 0), np.ascontiguousarray(a[:, :, 2:34]).view(np.int8)


def unpack_scales_k4(s: np.ndarray):
    """get_scale_min_k4 for j = 0..7 on scales[..., 12] -> (sc[..., 8], m[..., 8])."""
    s = s.astype(np.int32)
    sc = np.empty(s.shape[:-1] + (8,), np.int32)
    m = np.empty_like(sc)
    sc[..., 0:4] = s[..., 0:4] & 63
    m[..., 0:4] = s[..., 4:8] & 63
    sc[..., 4:8] = (s[..., 8:12] & 0xF) | ((s[..., 0:4] >> 6) << 4)
    m[..., 4:8] = (s[..., 8:12] >> 4) | ((s[..., 4:8] >> 6) << 4)
    return sc, m


def pack_scales_k4(sc: np.ndarray, m: np.ndarray) -> np.ndarray:
    sc = sc.astype(np.int32)
    m = m.astype(np.int32)
    s = np.empty(sc.shape[:-1] + (12,), np.int32)
    s[..., 0:4] = sc[..., 0:4] | ((sc[..., 4:8] >> 4) << 6)
    s[..., 4:8] = m[..., 0:4] | ((m[..., 4:8] >> 4) << 6)
    s[..., 8:12] = (sc[..., 4:8] & 0xF) | ((m[..., 4:8] & 0xF) << 4)
    return s.astype(np.uint8)


def parse_q4_K(blocks: np.ndarray, N: int, K: int):
    ns = K // 256
    a = blocks.reshape(N, ns, 144)
    d = _fp16_field(a, 0)
    dmin = _fp16_field(a, 2)
    sc, m = unpack_scales_k4(a[:, :, 4:16])
    qs = a[:, :, 16:144]
    q4 = np.empty((N, ns, 256), np.uint8)
    for g in range(4):
        q4[:, :, 64 * g:64 * g + 32] = qs[:, :, 32 * g:32 * g + 32] & 0xF
        q4[:, :, 64 * g + 32:64 * g + 64] = qs[:, :, 32 * g:32 * g + 32] >> 4
    return d, dmin, sc, m, q4


def parse_weights(fmt: int, blocks: np.ndarray, N: int, K: int):
    return parse_q8_0(blocks, N, K) if fmt == FMT_Q8_0 else parse_q4_K(blocks, N, K)


# =============================================================================================
# B1b: independent numpy model of the spec's fp32 arithmetic (numeric rules: block terms, row
# accumulation, common fp rules).
# No libtsr_ref code: numpy QACT (np_qact), numpy weight parsing, numpy binary32 ops with DAZ / FTZ.
# =============================================================================================
def _pos0(a: np.ndarray) -> np.ndarray:
    return a + 0.0            # integer-valued: -0 -> +0 (int -> f32 of 0 is +0)


def spec_terms(fmt: int, wp, qa, T: int) -> np.ndarray:
    """Block terms t[i][N][T] in f32, each multiply / subtract rounded separately (spec block terms)."""
    qx, Dx, bsum = qa
    D = np.asarray(Dx, dtype=F32).T[:, None, :]                                         # (nt, 1, T)
    if fmt == FMT_Q8_0:
        dw, qw = wp
        N, nt = dw.shape
        W = np.ascontiguousarray(qw.transpose(1, 0, 2), dtype=F32)                    # (nt, N, 32)
        X = np.ascontiguousarray(qx.reshape(T, nt, 32).transpose(1, 2, 0), dtype=F32)  # (nt, 32, T)
        isum = _pos0(np.matmul(W, X))                       # exact: |partial sums| <= 2^19
        return fmul(fmul(dw.T.astype(F32)[:, :, None], D), isum)
    d, dmin, sc, m, q4 = wp
    N, nt = d.shape
    W = np.ascontiguousarray(q4.reshape(N, nt * 8, 32).transpose(1, 0, 2), dtype=F32)
    X = np.ascontiguousarray(qx.reshape(T, nt * 8, 32).transpose(1, 2, 0), dtype=F32)
    S = np.matmul(W, X).astype(F64).reshape(nt, 8, N, T)   # exact: |S_j| <= 61440
    isum = _pos0((S * sc.transpose(1, 2, 0).astype(F64)[:, :, :, None]).sum(axis=1))  # exact ints (f64)
    bp = bsum.astype(np.int64).reshape(T, nt, 8, 2).sum(-1).astype(F64)
    msum = _pos0((m.transpose(1, 2, 0).astype(F64)[:, :, :, None] * bp.transpose(1, 2, 0)[:, :, None, :]).sum(axis=1))
    t1 = fmul(fmul(d.T.astype(F32)[:, :, None], D), isum.astype(F32))       # f32(isum): RNE, |isum| may be > 2^24
    t2 = fmul(fmul(dmin.T.astype(F32)[:, :, None], D), msum.astype(F32))    # exact: |msum| < 2^24
    return fsub(t1, t2)


def spec_model_y(fmt: int, wp, qa, T: int, order: str = "tree") -> np.ndarray:
    """y[T][N] in f32 from the spec text: 16 partials p_(i mod 16) from +0.0, term i added in order,
    then the complete binary tree ((p0+p1)+(p2+p3)) + ... (left pairs, 4 levels).
    order="sequential" (one running sum) is the negative-control mutant used by the self-tests."""
    t = spec_terms(fmt, wp, qa, T)
    nt = t.shape[0]
    if order == "sequential":
        y = np.zeros(t.shape[1:], F32)
        for i in range(nt):
            y = fadd(y, t[i])
        return y.T
    G = -(-nt // 16)
    if G * 16 != nt:          # pad with -0.0, the exact identity of RNE addition (p + -0 == p, also for p = ±0)
        t = np.concatenate([t, np.full((G * 16 - nt,) + t.shape[1:], -0.0, F32)])
    t = t.reshape((G, 16) + t.shape[1:])
    p = np.zeros((16,) + t.shape[2:], F32)
    for g in range(G):
        p = fadd(p, t[g])
    while p.shape[0] > 1:
        p = fadd(p[0::2], p[1::2])
    return p[0].T


# =============================================================================================
# B2: fp64 value of the quantized dot product and the proven fp32 forward-error bound
# (derivation: docs/numcheck.md "B2 bound")
# =============================================================================================
def b2_eval(fmt: int, wp, qa, T: int):
    """Returns y64[N][T], bound[N][T], in_domain[N][T] (bool), S0[N][T] = sum of |term parts|,
    uf[N][T] (bool: some term may flush to zero, i.e. the FTZ slack is active)."""
    qx, Dx, bsum = qa
    Dx64 = daz(Dx).astype(F64)                               # (T, nterms); DAZ of an f32 input
    if fmt == FMT_Q8_0:
        dw, qw = wp
        N, nt = dw.shape
        W = np.ascontiguousarray(qw.transpose(1, 0, 2), dtype=F32)                    # (nt, N, 32)
        X = np.ascontiguousarray(qx.reshape(T, nt, 32).transpose(1, 2, 0), dtype=F32)  # (nt, 32, T)
        isum = np.matmul(W, X).astype(F64)                   # exact: |partial sums| < 2^24
        del W, X
        a = dw.T[:, :, None] * Dx64.T[:, None, :]            # exact (11 x 24 bits)
        Ti = a * isum
        M = np.abs(Ti)
        # fl(d*Dx) may flush (or round up to 2^-126); fl(a*f) never does: a is 0 or normal, f an integer
        c1 = np.abs(a) < 2.0 ** -125
        eps = LAM * (1.0 + U) * (c1 * np.abs(isum))
        gc = gamma(2)
        amax_a = np.abs(a).max(axis=0)
        uf = (c1 & (a != 0) & (isum != 0)).any(axis=0)
        del a, isum, c1
    else:
        d, dmin, sc, m, q4 = wp
        N, nt = d.shape
        nsub = nt * 8
        W = np.ascontiguousarray(q4.reshape(N, nsub, 32).transpose(1, 0, 2), dtype=F32)    # (nsub, N, 32)
        X = np.ascontiguousarray(qx.reshape(T, nsub, 32).transpose(1, 2, 0), dtype=F32)   # (nsub, 32, T)
        S = np.matmul(W, X).astype(F64).reshape(nt, 8, N, T)  # exact: |S_j| <= 61440
        del W, X
        isum = (S * sc.transpose(1, 2, 0).astype(F64)[:, :, :, None]).sum(axis=1)     # (nt, N, T), exact ints
        del S
        bp = bsum.astype(np.int64).reshape(T, nt, 8, 2).sum(-1).astype(F64)            # (T, nt, 8) pair bsums
        msum = (m.transpose(1, 2, 0).astype(F64)[:, :, :, None]
                * bp.transpose(1, 2, 0)[:, :, None, :]).sum(axis=1)                     # (nt, N, T), exact ints
        a1 = d.T[:, :, None] * Dx64.T[:, None, :]
        a2 = dmin.T[:, :, None] * Dx64.T[:, None, :]
        T1 = a1 * isum
        T2 = a2 * msum
        Ti = T1 - T2
        aT1, aT2 = np.abs(T1), np.abs(T2)
        M = aT1 + aT2
        gc = gamma(4)
        ai, am = np.abs(isum), np.abs(msum)
        c1 = np.abs(a1) < 2.0 ** -125                        # fl(d*Dx) may flush
        c1m = np.abs(a2) < 2.0 ** -125                       # fl(dmin*Dx) may flush
        # fl(t1 - t2) may flush only if |t1 - t2| < 2^-126; |t1 - t2 - T| <= gc*M + 2*LAM*(|isum| + |msum|)
        c3 = (M > 0) & (np.abs(Ti) < 2.0 ** -125 + gc * M + 2.0 * LAM * (ai + am))
        eps = LAM * (c1 * ai * (1.0 + U) ** 3 + c1m * am * (1.0 + U) ** 2 + c3)
        amax_a = np.maximum(np.abs(a1).max(axis=0), np.abs(a2).max(axis=0))
        uf = ((c1 & (a1 != 0) & (isum != 0)) | (c1m & (a2 != 0) & (msum != 0))).any(axis=0)
        del a1, a2, T1, T2, aT1, aT2, isum, msum, ai, am, c1, c1m, c3
    with np.errstate(all="ignore"):
        y64 = Ti.sum(axis=0)
        S0 = M.sum(axis=0)
        E = eps.sum(axis=0)
        h = -(-nt // 16) + 3                                 # roundings on the longest path: partial + tree
        gh = gamma(h)
        bound = (gc * S0 + E + gh * (S0 * (1.0 + gc) + E) + (nt + 15) * LAM * (1.0 + gh)
                 + gamma(nt + 4, U64) * S0 * 1.01) * (1.0 + 2.0 ** -40)
        dom = np.isfinite(y64) & np.isfinite(S0) & (S0 < 2.0 ** 126) & (amax_a < 2.0 ** 126)
    return y64, bound, dom, S0, uf


# =============================================================================================
# GGUF access (gguf-py for metadata, pread for row chunks; multi-file splits)
# =============================================================================================
class GGUFModel:
    def __init__(self, path: str):
        if str(GGUF_PY) not in sys.path:
            sys.path.insert(0, str(GGUF_PY))
        from gguf import GGUFReader  # noqa: WPS433

        p = Path(path)
        m = re.match(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", p.name)
        shards = [p]
        if m:
            n = int(m.group(3))
            shards = [p.with_name(f"{m.group(1)}-{i:05d}-of-{n:05d}.gguf") for i in range(1, n + 1)]
        self.path = str(p)
        self.shards = [str(s) for s in shards]
        self.tensors: dict[str, tuple[str, int, int, list[int], int]] = {}
        for s in shards:
            if not s.exists():
                die(f"GGUF shard {s} missing")
            r = GGUFReader(str(s))
            for t in r.tensors:
                self.tensors[t.name] = (str(s), int(t.data_offset), int(t.tensor_type),
                                        [int(v) for v in t.shape], int(t.n_bytes))
            del r
        self._fds: dict[str, int] = {}

    def info(self, name: str):
        return self.tensors.get(name)

    def read_rows(self, nat: Native, name: str, r0: int, r1: int) -> np.ndarray:
        path, off, typ, ne, _ = self.tensors[name]
        rs = nat.row_size(typ, ne[0])
        fd = self._fds.get(path)
        if fd is None:
            fd = self._fds[path] = os.open(path, os.O_RDONLY)
        n = (r1 - r0) * rs
        buf = os.pread(fd, n, off + r0 * rs)
        if len(buf) != n:
            die(f"short read of {name} rows {r0}..{r1}")
        return np.frombuffer(buf, dtype=np.uint8)

    def close(self):
        for fd in self._fds.values():
            os.close(fd)
        self._fds = {}


# =============================================================================================
# run directories
# =============================================================================================
class Run:
    def __init__(self, d: str, label: str):
        self.dir = Path(d)
        mp = self.dir / "meta.json"
        if not mp.exists():
            die(f"{mp} missing")
        self.meta = json.loads(mp.read_text())
        if self.meta.get("format") != "tsr-numcheck" or self.meta.get("version") not in (1, 2):
            die(f"{mp}: not a tsr-numcheck version 1 / 2 dump")
        self.label = label
        self.nodes = self.meta["nodes"]

    def _load(self, rel: str, ne) -> np.ndarray:
        n = int(np.prod(ne))
        a = np.fromfile(self.dir / rel, dtype="<f4")
        if a.size != n:
            die(f"{self.dir / rel}: {a.size} floats, expected {n} (ne {ne})")
        return a.astype(F32, copy=False).reshape(-1, int(ne[0]))

    def load(self, node) -> np.ndarray:
        return self._load(node["file"], node["ne"])

    def load_src1(self, node):
        if not node.get("src1_file"):
            return None
        return self._load(node["src1_file"], node["src1_ne"])

    def logits(self) -> np.ndarray:
        T, V = int(self.meta["n_tokens"]), int(self.meta["n_vocab"])
        a = np.fromfile(self.dir / self.meta.get("logits_file", "logits.f32"), dtype="<f4")
        if a.size != T * V:
            die(f"{self.dir}: logits has {a.size} floats, expected {T}x{V}")
        return a.reshape(T, V)


def node_key(n):
    return (n["name"], n["op"], tuple(int(v) for v in n["ne"]))


def align_nodes(ref: list, other: list):
    """Pairs (ref_node, other_node | None) by graph index, falling back to (name, op, ne, occurrence)."""
    if len(ref) == len(other) and all(node_key(a) == node_key(b) for a, b in zip(ref, other)):
        return [(a, b) for a, b in zip(ref, other)], 0
    occ: dict = {}
    table = {}
    for n in other:
        k = node_key(n)
        c = occ.get(k, 0)
        occ[k] = c + 1
        table[(k, c)] = n
    occ = {}
    out = []
    miss = 0
    for n in ref:
        k = node_key(n)
        c = occ.get(k, 0)
        occ[k] = c + 1
        b = table.get((k, c))
        miss += b is None
        out.append((n, b))
    return out, miss


# =============================================================================================
# statistics
# =============================================================================================
def cmp_stats(ref: np.ndarray, x: np.ndarray) -> dict:
    r = ref.astype(F64).ravel()
    v = x.astype(F64).ravel()
    fr, fv = np.isfinite(r), np.isfinite(v)
    both = fr & fv
    nonfinite_mismatch = int(np.count_nonzero(fr != fv) + np.count_nonzero(~fr & ~fv & (f32bits(ref).ravel() != f32bits(x).ravel())))
    r, v = r[both], v[both]
    d = v - r
    se = float(np.dot(d, d))
    sr = float(np.dot(r, r))
    sv = float(np.dot(v, v))
    nmse = se / sr if sr > 0 else (0.0 if se == 0 else math.inf)
    cos = float(np.dot(r, v) / math.sqrt(sr * sv)) if sr > 0 and sv > 0 else (1.0 if se == 0 else 0.0)
    return {
        "nmse": nmse,
        "maxabs": float(np.abs(d).max()) if d.size else 0.0,
        "cos": cos,
        "rms_ref": math.sqrt(sr / max(1, r.size)),
        "identical": bool(np.array_equal(f32bits(ref), f32bits(x))),
        "nonfinite_mismatch": nonfinite_mismatch,
    }


def ratio(a: float, b: float):
    if b > 0:
        return a / b
    return 1.0 if a == 0 else math.inf


def log_softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(F64)
    mx = np.max(x, axis=-1, keepdims=True)
    z = x - mx
    return z - np.log(np.sum(np.exp(z), axis=-1, keepdims=True))


def kl_rows(lp: np.ndarray, lq: np.ndarray) -> np.ndarray:
    return np.sum(np.exp(lp) * (lp - lq), axis=-1)


def ulp_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    def ordered(x):
        i = f32bits(x).astype(np.int64)
        return np.where(i & 0x80000000, 0x80000000 - i, i)  # monotone integer image
    return np.abs(ordered(a) - ordered(b))


def parse_layers(spec):
    """--mm-layers as the dumper reads it: all (default), or a comma list of N, A-B and output / out / head
    (weights outside blk.N). Raises ValueError on anything else ("none" too: B would check nothing)."""
    if not spec or spec == "all":
        return None
    if spec == "none":
        raise ValueError("'none' leaves nothing for B to check")
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if part in ("output", "out", "head"):
            out.add("output")
        elif re.fullmatch(r"\d+-\d+", part):
            a, b = (int(v) for v in part.split("-"))
            if b < a:
                raise ValueError(f"empty range {part!r}")
            out.update(range(a, b + 1))
        elif re.fullmatch(r"\d+", part):
            out.add(int(part))
        else:
            raise ValueError(f"bad item {part!r} (expected N, A-B, output / out / head, or all)")
    if not out:
        raise ValueError("empty list")
    return out


def weight_layer(name: str):
    m = re.match(r"^blk\.(\d+)\.", name)
    if m:
        return int(m.group(1))
    if name.startswith("output") or name.startswith("token_embd"):
        return "output"
    return name


def pct(a: np.ndarray, q: float) -> float:
    return float(np.percentile(a, q)) if a.size else 0.0


# =============================================================================================
# (B) per weight MUL_MAT
# =============================================================================================
# --src1-from-producer: a weight MUL_MAT outside the dumper's --mm-layers has no src1 dump, but its
# src1 is the output of a named node that is dumped (llama.cpp graph names). The scheduler's input
# copy on the device is a byte copy of that output; the mapping is verified bit for bit on every
# weight MUL_MAT that does have a src1 dump before it is used (producer_validation), and B1 itself
# would flag a wrong x.
PRODUCERS = {  # weight role -> producer node names (first match wins), {l} = layer
    "attn_q": ("attn_norm-{l}",), "attn_k": ("attn_norm-{l}",), "attn_v": ("attn_norm-{l}",),
    "attn_qkv": ("attn_norm-{l}",), "attn_output": ("kqv_out-{l}",),
    "ffn_gate": ("ffn_norm-{l}",), "ffn_up": ("ffn_norm-{l}",),
    "ffn_down": ("ffn_swiglu-{l}", "ffn_geglu-{l}", "ffn_gelu-{l}", "ffn_silu-{l}", "ffn_relu-{l}"),
}
OUTPUT_PRODUCERS = {"output.weight": ("result_norm",), "token_embd.weight": ("result_norm",)}
# With flash attention (the baseline settings, SW-22) attn_output's src1 is the FLASH_ATTN_EXT output
# [D, n_head, T] seen through a reshape view (kqv_out-N, not dumped): the same bytes as [K, T].
FA_ROLES = ("attn_output",)


def producer_names(wname: str):
    m = re.match(r"^blk\.(\d+)\.([A-Za-z0-9_]+)\.weight$", wname)
    if m:
        return tuple(p.format(l=m.group(1)) for p in PRODUCERS.get(m.group(2), ()))
    return OUTPUT_PRODUCERS.get(wname, ())


def find_producer(run, nd: dict, K: int):
    """Latest node before nd named like nd's src1 producer with ne [K, T, 1, 1]; for attn_output with flash
    attention, the latest FLASH_ATTN_EXT node before nd with ne [D, H, T, 1], D x H = K; or None."""
    idx = getattr(run, "_by_name", None)
    if idx is None:
        idx = run._by_name = {}
        for n in run.nodes:
            idx.setdefault(n["name"], []).append(n)
    T = int(nd["ne"][1])
    for pn in producer_names(nd["weight"]):
        c = [n for n in idx.get(pn, []) if n["i"] < nd["i"] and [int(v) for v in n["ne"]] == [K, T, 1, 1]]
        if c:
            return c[-1]
    m = re.match(r"^blk\.\d+\.([A-Za-z0-9_]+)\.weight$", nd["weight"] or "")
    if m and m.group(1) in FA_ROLES:
        c = [n for n in run.nodes if n["i"] < nd["i"] and n["op"] == "FLASH_ATTN_EXT"]
        if c:
            ne = [int(v) for v in c[-1]["ne"]]
            if ne[0] * ne[1] == K and ne[2] == T and ne[3] == 1:
                return c[-1]
    return None


def load_producer(run, p: dict, K: int, T: int) -> np.ndarray:
    """A producer's output as [T][K] (a FLASH_ATTN_EXT output [D, H, T] is contiguous: D x H = K per token)."""
    return run.load(p).reshape(T, K)


def producer_validation(runs: dict, model: GGUFModel) -> dict:
    """Producer output == src1 dump, bit for bit, for every weight MUL_MAT with a src1 dump (dev and cpu)."""
    out = {"checked": 0, "identical": 0, "differ": [], "no_producer": []}
    for label in ("dev", "cpu"):
        run = runs[label]
        for nd in run.nodes:
            if not nd.get("weight") or not nd.get("src1_file") or model.info(nd["weight"]) is None:
                continue
            K = model.info(nd["weight"])[3][0]
            p = find_producer(run, nd, K)
            if p is None:
                out["no_producer"].append(f"{label}:{nd['weight']}")
                continue
            out["checked"] += 1
            if np.array_equal(f32bits(load_producer(run, p, K, int(nd["ne"][1]))), f32bits(run.load_src1(nd))):
                out["identical"] += 1
            else:
                out["differ"].append(f"{label}:{nd['weight']} vs node {p['i']} {p['name']}")
    out["ok"] = out["checked"] > 0 and not out["differ"] and not out["no_producer"]
    return out


def load_x(run, nd: dict, K: int, glob: dict, label: str):
    """src1 of a weight MUL_MAT: its dump, or (--src1-from-producer, validated) the producer's output."""
    x = run.load_src1(nd)
    if x is not None:
        return x, "src1 dump"
    if not glob.get("use_producer"):
        return None, None
    p = find_producer(run, nd, K)
    if p is None:
        return None, None
    return load_producer(run, p, K, int(nd["ne"][1])), f"node {p['i']} {p['name']}"


def check_weight_mm(nat: Native, model: GGUFModel, nd: dict, nc: dict | None, runs: dict, args, glob: dict) -> dict:
    wname = nd["weight"]
    res = {"i": nd["i"], "name": nd["name"], "weight": wname, "backend": nd.get("backend"),
           "layer": weight_layer(wname)}
    info = model.info(wname)
    if info is None:
        res["skip"] = "weight not in GGUF"
        return res
    _, _, typ, ne, _ = info
    K, N = ne[0], ne[1]
    tname = nat.type_name(typ)
    fmt = nat.dev_fmt(typ)
    res.update({"type": tname, "dev_fmt": FMT_NAME[fmt], "native": typ in (T_Q8_0, T_Q4_K), "N": N, "K": K})
    if nd.get("weight_type") and nd["weight_type"] != tname:
        res["type_note"] = f"meta says {nd['weight_type']}"
    x_dev, res["x_source"] = load_x(runs["dev"], nd, K, glob, "dev")
    if x_dev is None:
        res["skip"] = NOT_SAMPLED
        return res
    y_dev = runs["dev"].load(nd)
    if x_dev.shape[1] != K or y_dev.shape[1] != N or y_dev.shape[0] != x_dev.shape[0]:
        res["skip"] = f"shape mismatch: src1 {list(x_dev.shape)} dst {list(y_dev.shape)} weight K={K} N={N}"
        return res
    T = x_dev.shape[0]
    res["T"] = T
    x_cpu = y_cpu = None
    if nc is not None and nc.get("weight") == wname:
        x_cpu, _ = load_x(runs["cpu"], nc, K, glob, "cpu")
        if x_cpu is not None:
            y_cpu = runs["cpu"].load(nc)
    on_dev = "tesseract" in str(nd.get("backend", "")).lower()
    res["on_device"] = on_dev
    kind = QA_A32 if fmt == FMT_Q8_0 else QA_A256

    # QACT: independent numpy model, cross-checked bit for bit against libtsr_ref on this x
    q, D, bsum = np_qact(kind, x_dev)
    img_tsr = nat.qact(kind, x_dev)
    qact_bad = int(np.count_nonzero(np.any(qa_image(kind, K, q, D, bsum) != img_tsr, axis=1)))
    res["qact_model_mismatch_tokens"] = qact_bad
    del img_tsr

    rows = max(1, min(N, args.chunk_elems // max(1, (K // 32) * T), (1 << 24) // K))
    b1_bad = 0
    b1_first = []
    b1_maxulp = 0
    b1b_bad = 0
    b1b_first = []
    b2_viol = 0
    b2_first = []
    b2_ood = 0
    b2_uf = 0
    ratios = []
    e2d, r2d, e2c, r2c = (np.zeros(T, F64) for _ in range(4))   # B3 sums per token
    xd64 = x_dev.astype(F64)
    xc64 = x_cpu.astype(F64) if x_cpu is not None else None
    for r0 in range(0, N, rows):
        r1 = min(N, r0 + rows)
        nr = r1 - r0
        raw = model.read_rows(nat, wname, r0, r1)
        _, blocks = nat.dev_blocks(typ, raw, nr, K)
        yd = y_dev[:, r0:r1]
        # B1
        if on_dev:
            ym = nat.model_mm(fmt, blocks, nr, K, x_dev, args.threads)
            bad = f32bits(ym) != f32bits(yd)
            nbad = int(np.count_nonzero(bad))
            if nbad:
                b1_bad += nbad
                ul = ulp_dist(ym[bad], yd[bad])
                b1_maxulp = max(b1_maxulp, int(ul.max()))
                for t, n in zip(*np.nonzero(bad)):
                    if len(b1_first) >= 8:
                        break
                    b1_first.append({"row": int(r0 + n), "token": int(t), "dev": float(yd[t, n]),
                                     "dev_bits": f"0x{int(f32bits(yd[t, n])):08x}", "model": float(ym[t, n]),
                                     "model_bits": f"0x{int(f32bits(ym[t, n])):08x}",
                                     "ulp": int(ulp_dist(ym[t:t + 1, n], yd[t:t + 1, n])[0])})
        # B2 (device arithmetic only: a MUL_MAT the CPU computed follows other rules)
        ydT = yd.T.astype(F64)
        if on_dev:
            wp = parse_weights(fmt, blocks, nr, K)
            # B1b: the independent numpy spec model (no libtsr_ref) must give the same bits
            ys = spec_model_y(fmt, wp, (q, D, bsum), T)
            bad = f32bits(ys) != f32bits(yd)
            nbad = int(np.count_nonzero(bad))
            if nbad:
                b1b_bad += nbad
                for t, n in zip(*np.nonzero(bad)):
                    if len(b1b_first) >= 8:
                        break
                    b1b_first.append({"row": int(r0 + n), "token": int(t), "dev": float(yd[t, n]),
                                      "dev_bits": f"0x{int(f32bits(yd[t, n])):08x}", "model": float(ys[t, n]),
                                      "model_bits": f"0x{int(f32bits(ys[t, n])):08x}",
                                      "ulp": int(ulp_dist(ys[t:t + 1, n], yd[t:t + 1, n])[0])})
            del ys, bad
            y64, bnd, dom, _, uf = b2_eval(fmt, wp, (q, D, bsum), T)
            b2_uf += int(np.count_nonzero(uf & dom))
            with np.errstate(all="ignore"):
                diff = np.abs(ydT - y64)
            viol = dom & ~(diff <= bnd)
            b2_ood += int(np.count_nonzero(~dom))
            nv = int(np.count_nonzero(viol))
            if nv:
                b2_viol += nv
                for n, t in zip(*np.nonzero(viol)):
                    if len(b2_first) >= 8:
                        break
                    b2_first.append({"row": int(r0 + n), "token": int(t), "dev": float(ydT[n, t]),
                                     "y_q64": float(y64[n, t]), "abs_err": float(diff[n, t]), "bound": float(bnd[n, t])})
            with np.errstate(all="ignore"):
                rr = np.where(bnd > 0, diff / bnd, np.where(diff > 0, np.inf, 0.0))
            ratios.append(rr[dom].astype(F32))
            del wp, y64, bnd, dom, diff, viol, rr
        # B3
        Wf = nat.to_float(typ, raw, nr * K).reshape(nr, K).astype(F64)
        ye = Wf @ xd64.T                                     # (nr, T)
        e2d += np.sum((ydT - ye) ** 2, axis=0)
        r2d += np.sum(ye ** 2, axis=0)
        if xc64 is not None:
            yc = Wf @ xc64.T
            e2c += np.sum((y_cpu[:, r0:r1].T.astype(F64) - yc) ** 2, axis=0)
            r2c += np.sum(yc ** 2, axis=0)
        del Wf, ye, ydT
    rat = np.concatenate(ratios) if ratios else np.zeros(0, F32)
    glob["b2_ratios"].append(rat)
    res["b1"] = {"checked": bool(on_dev), "elements": int(N * T) if on_dev else 0, "mismatches": b1_bad,
                 "max_ulp": b1_maxulp, "first": b1_first}
    res["b1b"] = {"checked": bool(on_dev), "elements": int(N * T) if on_dev else 0, "mismatches": b1b_bad,
                  "first": b1b_first}
    res["b2"] = {"elements": int(rat.size), "violations": b2_viol, "out_of_domain": b2_ood, "ftz_slack_active": b2_uf,
                 "ratio_median": float(np.median(rat)) if rat.size else 0.0,
                 "ratio_p99": pct(rat, 99), "ratio_max": float(rat.max()) if rat.size else 0.0, "first": b2_first}
    res["b3"] = b3_summary(e2d, r2d, e2c if xc64 is not None else None, r2c if xc64 is not None else None)
    return res


DOMINANT_SHARE = 0.5     # B3: a token holding more than this share of a tensor's ||W x||^2 is reported apart


def _rel(e2: np.ndarray, r2: np.ndarray, keep=None) -> float:
    if keep is not None:
        e2, r2 = e2[keep], r2[keep]
    se, sr = float(np.sum(e2)), float(np.sum(r2))
    return math.sqrt(se / sr) if sr > 0 else 0.0


def b3_summary(e2d: np.ndarray, r2d: np.ndarray, e2c, r2c) -> dict:
    """B3 of one tensor from per-token sums e2 = ||y_t - W x_t||^2, r2 = ||W x_t||^2.
    Pooled over all tokens, a single token with a massive activation decides the value (32B token 0:
    > 99 % of ||W x||^2 in some ffn_down). Tokens with a share > DOMINANT_SHARE of ||W x||^2 (dev or cpu)
    are therefore listed and left out of the robust ratio ("ratio"); "ratio_all_tokens" keeps them."""
    T = int(e2d.size)
    share = r2d / r2d.sum() if r2d.sum() > 0 else np.zeros(T)
    if e2c is not None and r2c.sum() > 0:
        share = np.maximum(share, r2c / r2c.sum())
    dom = share > DOMINANT_SHARE if T > 1 else np.zeros(T, bool)
    keep = ~dom
    out = {"rel_err_dev_all_tokens": _rel(e2d, r2d), "rel_err_dev": _rel(e2d, r2d, keep),
           "dominant_tokens": [{"token": int(t), "share": float(share[t])} for t in np.nonzero(dom)[0]],
           "rel_err_cpu_all_tokens": None, "rel_err_cpu": None, "ratio_all_tokens": None, "ratio": None,
           "ratio_token_median": None,
           "e2_dev": float(e2d.sum()), "r2_dev": float(r2d.sum()), "e2_dev_kept": float(e2d[keep].sum()),
           "r2_dev_kept": float(r2d[keep].sum()), "e2_cpu": 0.0, "r2_cpu": 0.0, "e2_cpu_kept": 0.0, "r2_cpu_kept": 0.0}
    with np.errstate(all="ignore"):
        out["per_token"] = {"rel_err_dev": np.sqrt(e2d / r2d).tolist(), "share": share.tolist()}
    if e2c is None:
        return out
    out.update({"rel_err_cpu_all_tokens": _rel(e2c, r2c), "rel_err_cpu": _rel(e2c, r2c, keep),
                "e2_cpu": float(e2c.sum()), "r2_cpu": float(r2c.sum()),
                "e2_cpu_kept": float(e2c[keep].sum()), "r2_cpu_kept": float(r2c[keep].sum())})
    out["ratio_all_tokens"] = ratio(out["rel_err_dev_all_tokens"], out["rel_err_cpu_all_tokens"])
    out["ratio"] = ratio(out["rel_err_dev"], out["rel_err_cpu"])
    with np.errstate(all="ignore"):
        rc_t = np.sqrt(e2c / r2c)
        rt = np.sqrt(e2d / r2d) / rc_t
    out["per_token"]["rel_err_cpu"] = rc_t.tolist()
    ok = np.isfinite(rt)
    out["ratio_token_median"] = float(np.median(rt[ok])) if np.any(ok) else None
    return out


# =============================================================================================
# boundtest: B2 bound vs libtsr_ref, QACT model and fp32 primitives vs libtsr_ref
# =============================================================================================
def rand_fp16(rng, shape, emin=1, emax=30, p_sub=0.0, signed=False) -> np.ndarray:
    """Random finite fp16 bit patterns with exponent field in [emin, emax] (p_sub: subnormal share)."""
    e = rng.integers(emin, emax + 1, size=shape)
    if p_sub > 0:
        e = np.where(rng.random(shape) < p_sub, 0, e)
    m = rng.integers(0, 1024, size=shape)
    b = (e << 10) | m
    if signed:
        b |= (rng.random(shape) < 0.5).astype(np.int64) << 15
    return b.astype(np.uint16)


def make_q8_0_blocks(d16: np.ndarray, qs: np.ndarray) -> np.ndarray:
    N, nb = d16.shape
    a = np.empty((N, nb, 34), np.uint8)
    a[:, :, 0:2] = d16.astype("<u2").view(np.uint8).reshape(N, nb, 2)
    a[:, :, 2:] = qs.astype(np.int8).view(np.uint8)
    return a.reshape(-1)


def make_q4_K_blocks(d16, dmin16, sc, m, q4) -> np.ndarray:
    N, ns = d16.shape
    a = np.empty((N, ns, 144), np.uint8)
    a[:, :, 0:2] = d16.astype("<u2").view(np.uint8).reshape(N, ns, 2)
    a[:, :, 2:4] = dmin16.astype("<u2").view(np.uint8).reshape(N, ns, 2)
    a[:, :, 4:16] = pack_scales_k4(sc, m)
    q4 = q4.astype(np.uint8).reshape(N, ns, 256)
    for g in range(4):
        a[:, :, 16 + 32 * g:16 + 32 * g + 32] = q4[:, :, 64 * g:64 * g + 32] | (q4[:, :, 64 * g + 32:64 * g + 64] << 4)
    return a.reshape(-1)


def rand_x(rng, T, K, lo, hi, heavy=False) -> np.ndarray:
    blk = 32
    scale = 10.0 ** rng.uniform(lo, hi, size=(T, K // blk, 1))
    base = rng.standard_t(2, size=(T, K // blk, blk)) if heavy else rng.standard_normal((T, K // blk, blk))
    with np.errstate(all="ignore"):
        return (base * scale).reshape(T, K).astype(F32)


def gen_case(nat: Native, rng, case: str):
    """One batch: (fmt, K, blocks, N, qa_images | None, x | None)."""
    if case.startswith("q8"):
        fmt = FMT_Q8_0
        nb = int(rng.choice([1, 2, 3, 7, 16, 17, 28, 31, 32, 33, 64, 100, 152, 160]) if rng.random() < 0.85
                 else rng.integers(161, 1025))
        K = 32 * nb
        N = int(max(16, min(512, 300000 // K)))
        T = 8
        if case == "q8_quant":                      # realistic: quantize_row_q8_0_ref of random weights
            w = rand_x(rng, N, K, -4, 1, heavy=rng.random() < 0.5)
            blocks = nat.quantize_q8_0(w)
            x = rand_x(rng, T, K, -3, 2, heavy=rng.random() < 0.5)
            return fmt, K, blocks, N, None, x
        if case == "q8_raw":                        # raw ints incl. -128, fp16 d incl. subnormal and 65504
            d16 = rand_fp16(rng, (N, nb), 1, 30, p_sub=0.1, signed=True)
            if rng.random() < 0.3:
                d16[:, ::3] = 0x7BFF                # max finite fp16
            qs = rng.integers(-128, 128, size=(N, nb, 32))
            x = rand_x(rng, T, K, -7, 5.5, heavy=rng.random() < 0.5)      # amax > 127 * 65504: D = Inf
            return fmt, K, make_q8_0_blocks(d16, qs), N, None, x
        if case == "q8_extreme":                    # |isum| at its maximum, all x at amax
            d16 = rand_fp16(rng, (N, nb), 10, 30)
            qs = np.where(rng.random((N, nb, 32)) < 0.5, -128, 127)
            x = (np.where(rng.random((T, K)) < 0.5, -1.0, 1.0) * 10.0 ** rng.uniform(-3, 6, size=(T, 1))).astype(F32)
            return fmt, K, make_q8_0_blocks(d16, qs), N, None, x
        if case == "q8_cancel":                     # block b + h = -(block b): exact sum 0 (+ tiny extra)
            h = max(1, nb // 2)
            nb2 = 2 * h + (nb % 2)
            K = 32 * nb2
            N = int(max(16, min(512, 300000 // K)))
            d16 = rand_fp16(rng, (N, nb2), 5, 25)
            qs = rng.integers(-127, 128, size=(N, nb2, 32))
            d16[:, h:2 * h] = d16[:, :h]
            qs[:, h:2 * h] = -qs[:, :h]
            x = rand_x(rng, T, K, -3, 3)
            x.reshape(T, nb2, 32)[:, h:2 * h] = x.reshape(T, nb2, 32)[:, :h]
            if nb2 > 2 * h:
                x.reshape(T, nb2, 32)[:, -1] *= F32(1e-6)
            return fmt, K, make_q8_0_blocks(d16, qs), N, None, x
        if case == "q8_directqa":                   # arbitrary QA images (q in [-128,127], any f32 D)
            d16 = rand_fp16(rng, (N, nb), 0, 30, p_sub=0.05, signed=True)
            qs = rng.integers(-128, 128, size=(N, nb, 32))
            q = rng.integers(-128, 128, size=(T, K)).astype(np.int8)
            e = rng.uniform(-140, 60, size=(T, nb))
            D = (np.where(rng.random((T, nb)) < 0.5, -1, 1) * 2.0 ** e * rng.uniform(1, 2, size=(T, nb))).astype(F32)
            return fmt, K, make_q8_0_blocks(d16, qs), N, qa_image(QA_A32, K, q, D, None), None
        raise ValueError(case)
    fmt = FMT_Q4_K
    ns = int(rng.choice([1, 2, 3, 4, 5, 8, 16, 17, 20, 32]) if rng.random() < 0.9 else rng.integers(33, 109))
    K = 256 * ns
    N = int(max(16, min(512, 300000 // K)))
    T = 8
    sc = rng.integers(0, 64, size=(N, ns, 8))
    m = rng.integers(0, 64, size=(N, ns, 8))
    q4 = rng.integers(0, 16, size=(N, ns, 256))
    d16 = rand_fp16(rng, (N, ns), 5, 22)
    dm16 = rand_fp16(rng, (N, ns), 5, 22)
    if case == "q4_random":
        x = rand_x(rng, T, K, -3, 2, heavy=rng.random() < 0.5)
        if rng.random() < 0.3:
            d16 = rand_fp16(rng, (N, ns), 0, 30, p_sub=0.1, signed=True)
            dm16 = rand_fp16(rng, (N, ns), 0, 30, p_sub=0.1, signed=True)
        return fmt, K, make_q4_K_blocks(d16, dm16, sc, m, q4), N, None, x
    if case == "q4_cancel":                         # x second half = -(first half): Dx flips sign, exact 0
        h = max(1, ns // 2)
        ns2 = 2 * h
        K = 256 * ns2
        N = int(max(16, min(512, 300000 // K)))
        sc = rng.integers(0, 64, size=(N, ns2, 8))
        m = rng.integers(0, 64, size=(N, ns2, 8))
        q4 = rng.integers(0, 16, size=(N, ns2, 256))
        d16 = rand_fp16(rng, (N, ns2), 5, 22)
        dm16 = rand_fp16(rng, (N, ns2), 5, 22)
        for arr in (sc, m, q4, d16, dm16):
            arr[:, h:] = arr[:, :h]
        x = rand_x(rng, T, K, -3, 3)
        x.reshape(T, ns2, 256)[:, h:] = -x.reshape(T, ns2, 256)[:, :h]
        return fmt, K, make_q4_K_blocks(d16, dm16, sc, m, q4), N, None, x
    if case == "q4_termcancel":                     # d*sc*q ~ dmin*m: T1 ~ T2 inside each term
        m = rng.integers(1, 64, size=(N, ns, 8))
        sc = m.copy()
        q4 = np.ones((N, ns, 256), np.int64)
        if rng.random() < 0.5:
            q4 = rng.integers(1, 3, size=(N, ns, 256))
        dv = np.float16(2.0) ** rng.integers(-12, 2, size=(N, ns)).astype(np.float16)
        d16 = dv.view(np.uint16)
        dm16 = (dv.astype(F64) * (1 + rng.integers(-3, 4, size=(N, ns)) * 2.0 ** -10)).astype(np.float16).view(np.uint16)
        x = rand_x(rng, T, K, -3, 3)
        return fmt, K, make_q4_K_blocks(d16, dm16, sc, m, q4), N, None, x
    if case == "q4_bigisum":                        # |isum| > 2^24: the int -> f32 conversion rounds
        sc = np.full((N, ns, 8), 63)
        q4 = np.where(rng.random((N, ns, 256)) < 0.97, 15, rng.integers(0, 16, size=(N, ns, 256)))
        x = (10.0 ** rng.uniform(-2, 2, size=(T, 1)) * rng.uniform(0.996, 1.0, size=(T, K))).astype(F32)
        x *= np.where(rng.random((T, 1)) < 0.5, -1, 1).astype(F32)
        return fmt, K, make_q4_K_blocks(d16, dm16, sc, m, q4), N, None, x
    if case == "q4_tiny":                           # Dx near 2^-126: FTZ in d*Dx, flushed d, iscale = Inf
        d16 = rand_fp16(rng, (N, ns), 0, 15, p_sub=0.2)
        dm16 = rand_fp16(rng, (N, ns), 0, 15, p_sub=0.2)
        e = rng.uniform(-41, -31, size=(T, K // 256, 1))
        x = (rng.standard_normal((T, K // 256, 256)) * 10.0 ** e).astype(F32).reshape(T, K)
        return fmt, K, make_q4_K_blocks(d16, dm16, sc, m, q4), N, None, x
    if case == "q4_huge":
        d16 = rand_fp16(rng, (N, ns), 15, 30)
        dm16 = rand_fp16(rng, (N, ns), 15, 30)
        x = rand_x(rng, T, K, 18, 24)
        return fmt, K, make_q4_K_blocks(d16, dm16, sc, m, q4), N, None, x
    if case == "q4_directqa":
        d16 = rand_fp16(rng, (N, ns), 0, 30, p_sub=0.05, signed=True)
        dm16 = rand_fp16(rng, (N, ns), 0, 30, p_sub=0.05, signed=True)
        q = rng.integers(-128, 128, size=(T, K)).astype(np.int8)
        bsum = q.reshape(T, K // 16, 16).astype(np.int32).sum(-1).astype(np.int16)
        e = rng.uniform(-140, 40, size=(T, ns))
        D = (np.where(rng.random((T, ns)) < 0.5, -1, 1) * 2.0 ** e * rng.uniform(1, 2, size=(T, ns))).astype(F32)
        return fmt, K, make_q4_K_blocks(d16, dm16, sc, m, q4), N, qa_image(QA_A256, K, q, D, bsum), None
    raise ValueError(case)


BOUND_CASES = [  # (case, share of the rows)
    ("q8_quant", 0.20), ("q8_raw", 0.10), ("q8_extreme", 0.04), ("q8_cancel", 0.09), ("q8_directqa", 0.08),
    ("q4_random", 0.16), ("q4_cancel", 0.07), ("q4_termcancel", 0.07), ("q4_bigisum", 0.04),
    ("q4_tiny", 0.07), ("q4_huge", 0.03), ("q4_directqa", 0.05),
]


def bound_test(nat: Native, rows: int, seed: int, threads: int, quiet=False) -> dict:
    rng = np.random.default_rng(seed)
    t0 = time.time()
    cases = {}
    all_rat = []
    all_rat_n = []
    for case, share in BOUND_CASES:
        want = int(math.ceil(rows * share))
        st = {"rows": 0, "in_domain": 0, "violations": 0, "qact_mismatch": 0, "exact_zero_rows": 0,
              "nonzero_err_rows": 0, "batches": 0, "max_abs_err": 0.0, "ftz_rows": 0, "first": [],
              "spec_model_mismatch": 0, "spec_first": [], "multi_term_rows": 0, "seq_mutant_differs": 0}
        rats = []
        rats_n = []
        while st["in_domain"] < want:
            fmt, K, blocks, N, qa, x = gen_case(nat, rng, case)
            kind = QA_A32 if fmt == FMT_Q8_0 else QA_A256
            if qa is None:
                q, D, bsum = np_qact(kind, x)
                img = nat.qact(kind, x)
                st["qact_mismatch"] += int(np.count_nonzero(np.any(qa_image(kind, K, q, D, bsum) != img, axis=1)))
                y = nat.model_mm(fmt, blocks, N, K, x, threads)
            else:
                q, D, bsum = parse_qa_image(kind, K, qa)
                y = nat.model_mm_qa(fmt, blocks, N, K, qa, threads)
            T = y.shape[0]
            wp = parse_weights(fmt, blocks, N, K)
            # B1b model: bit-exact with libtsr_ref on every row (also out of the bound's domain); the
            # sequential-sum mutant must differ wherever order matters (negative control)
            ys = spec_model_y(fmt, wp, (q, D, bsum), T)
            sbad = f32bits(ys) != f32bits(y)
            st["spec_model_mismatch"] += int(np.count_nonzero(sbad))
            for t, n in zip(*np.nonzero(sbad)):
                if len(st["spec_first"]) >= 5:
                    break
                st["spec_first"].append({"K": K, "row": int(n), "token": int(t), "y_ref": f"0x{int(f32bits(y[t, n])):08x}",
                                         "spec_model": f"0x{int(f32bits(ys[t, n])):08x}"})
            if K // (32 if fmt == FMT_Q8_0 else 256) > 1:
                yq = spec_model_y(fmt, wp, (q, D, bsum), T, order="sequential")
                st["multi_term_rows"] += int(N * T)
                st["seq_mutant_differs"] += int(np.count_nonzero(f32bits(yq) != f32bits(y)))
            y64, bnd, dom, S0, uf = b2_eval(fmt, wp, (q, D, bsum), T)
            yT = y.T.astype(F64)
            with np.errstate(all="ignore"):
                diff = np.abs(yT - y64)
                rat = np.where(bnd > 0, diff / bnd, np.where(diff > 0, np.inf, 0.0))
            viol = dom & ~(diff <= bnd)
            st["rows"] += int(N * T)
            st["in_domain"] += int(np.count_nonzero(dom))
            st["batches"] += 1
            st["exact_zero_rows"] += int(np.count_nonzero(dom & (y64 == 0)))
            st["nonzero_err_rows"] += int(np.count_nonzero(dom & (diff > 0)))
            if np.any(dom):
                st["max_abs_err"] = max(st["max_abs_err"], float(diff[dom].max()))
            nv = int(np.count_nonzero(viol))
            if nv:
                st["violations"] += nv
                for n, t in zip(*np.nonzero(viol)):
                    if len(st["first"]) >= 5:
                        break
                    st["first"].append({"K": K, "row": int(n), "token": int(t), "y_ref": float(yT[n, t]),
                                        "y_q64": float(y64[n, t]), "abs_err": float(diff[n, t]), "bound": float(bnd[n, t])})
            rats.append(rat[dom].astype(F32))
            st["ftz_rows"] += int(np.count_nonzero(dom & uf))
            rats_n.append(rat[dom & ~uf].astype(F32))
            if st["batches"] >= 5000:
                st["note"] = "stopped after 5000 batches"
                break
        r = np.concatenate(rats)
        rn = np.concatenate(rats_n)
        all_rat.append(r)
        all_rat_n.append(rn)
        st.update({"ratio_median": float(np.median(r)), "ratio_p99": pct(r, 99), "ratio_p999": pct(r, 99.9),
                   "ratio_max": float(r.max()), "no_ftz_rows": int(rn.size),
                   "no_ftz_ratio_max": float(rn.max()) if rn.size else None})
        cases[case] = st
        if not quiet:
            log(f"  boundtest {case:14s} rows {st['rows']:8d} in-domain {st['in_domain']:8d} viol {st['violations']} "
                f"qact-mismatch {st['qact_mismatch']} |err|/bound median {st['ratio_median']:.2e} max {st['ratio_max']:.3f} "
                f"(no-FTZ rows {st['no_ftz_rows']}: max {st['no_ftz_ratio_max'] or 0:.3f}); spec model mismatch "
                f"{st['spec_model_mismatch']}, sequential mutant differs {st['seq_mutant_differs']}/{st['multi_term_rows']}")
    r = np.concatenate(all_rat)
    rn = np.concatenate(all_rat_n)
    tot = {k: sum(c[k] for c in cases.values()) for k in ("rows", "in_domain", "violations", "qact_mismatch", "ftz_rows",
                                                          "spec_model_mismatch", "multi_term_rows", "seq_mutant_differs")}
    out = {"bound_version": BOUND_VERSION, "seed": seed, "requested_rows": rows, **tot,
           "ratio_median": float(np.median(r)), "ratio_p99": pct(r, 99), "ratio_max": float(r.max()),
           "no_ftz_ratio_median": float(np.median(rn)), "no_ftz_ratio_p99": pct(rn, 99), "no_ftz_ratio_max": float(rn.max()),
           "seconds": round(time.time() - t0, 1), "cases": cases}
    # the mutant must be caught often enough to show that bit-exactness sees the accumulation order
    out["seq_mutant_detected"] = bool(tot["seq_mutant_differs"] >= 0.05 * max(1, tot["multi_term_rows"]))
    out["passed"] = bool(tot["violations"] == 0 and tot["qact_mismatch"] == 0 and tot["in_domain"] >= rows
                         and tot["spec_model_mismatch"] == 0 and out["seq_mutant_detected"])
    return out


def qact_vectors(rng, kind: int):
    """Directed and random activation sets for the QACT model test: list of (label, x[T][K])."""
    bs = 32 if kind == QA_A32 else 256
    sets = []
    sub = np.array([1, 2, 0x7FFFFF, 0x400000], np.uint32).view(F32)          # subnormals
    # spec example: x = {127, 2.5, -0.5, 0, ...}
    x = np.zeros((1, bs), F32)
    x[0, :3] = [127, 2.5, -0.5]
    sets.append(("spec_example", x))
    # rounding boundaries: max 127 -> scale 1, others k + 0.5 / -(k + 0.5)
    ks = np.arange(-127, 127) + 0.5
    n = -(-len(ks) // (bs - 1))
    x = np.zeros((n, bs), F32)
    x[:, 0] = 127
    flat = np.zeros(n * (bs - 1), F32)
    flat[:len(ks)] = ks
    x[:, 1:] = flat.reshape(n, bs - 1)
    sets.append(("half_boundaries", x))
    sets.append(("half_boundaries_neg", -x))
    # ties of the largest |x| (first one wins for A256), opposite signs
    x = rng.standard_normal((64, bs)).astype(F32)
    i, j = 3, 17
    x[:, i] = 5
    x[:32, j] = -5
    x[32:, j] = 5
    sets.append(("max_ties", x))
    x = -x
    sets.append(("max_ties_neg", x))
    # zero / all-subnormal / NaN / Inf blocks, subnormals mixed with normals
    x = np.zeros((8, bs), F32)
    x[1] = np.resize(sub, bs)
    x[2] = np.resize(-sub, bs)
    x[3] = np.nan
    x[4, ::2] = np.nan
    x[4, 1] = 3.0
    x[5, 7] = np.inf
    x[6, 9] = -np.inf
    x[6, 10] = 1.0
    x[7] = np.resize(np.concatenate([sub, [1e-3, -2e-3]]).astype(F32), bs)
    x[7, 0] = -0.0
    sets.append(("special_blocks", x))
    # amax ranges: tiny (d flushes, iscale overflows), huge (fp16 D -> Inf), fp16 subnormal D
    for lo, hi, lab in ((-45, -30, "tiny"), (-9, -4, "fp16_subnormal_D"), (4, 7, "near_fp16_max"),
                        (30, 38.5, "huge"), (-3, 3, "normal")):
        n = 400
        e = rng.uniform(lo, hi, size=(n, 1))
        with np.errstate(over="ignore"):
            x = (rng.standard_normal((n, bs)) * 10.0 ** e).astype(F32)
        sets.append((lab, x))
    # heavy tails with many ties after rounding, integer-valued vectors, random bit patterns
    sets.append(("student_t", (rng.standard_t(1.5, size=(400, bs)) * 3).astype(F32)))
    sets.append(("integers", rng.integers(-200, 200, size=(400, bs)).astype(F32)))
    bits = rng.integers(0, 2 ** 32, size=(400, bs), dtype=np.uint64).astype(np.uint32)
    sets.append(("random_bits", bits.view(F32)))
    # multi-block vectors (K = 4 blocks) with per-block scale jumps
    x = rand_x(rng, 400, 4 * bs, -6, 6, heavy=True)
    sets.append(("multi_block", x))
    return sets


def qact_test(nat: Native, seed: int, n_random: int) -> dict:
    rng = np.random.default_rng(seed)
    out = {"kinds": {}}
    total_bad = 0
    total_vec = 0
    for kind, kname in ((QA_A32, "A32"), (QA_A256, "A256")):
        res = {}
        for lab, x in qact_vectors(rng, kind):
            K = x.shape[1]
            q, D, bsum = np_qact(kind, x)
            bad = int(np.count_nonzero(np.any(qa_image(kind, K, q, D, bsum) != nat.qact(kind, x), axis=1)))
            res[lab] = {"vectors": int(x.shape[0]), "mismatch": bad}
            total_bad += bad
            total_vec += x.shape[0]
        # random bulk: K random multiple of the block, mixed distributions
        bs = 32 if kind == QA_A32 else 256
        done = bad = 0
        while done < n_random:
            K = bs * int(rng.integers(1, 9))
            T = 256
            x = rand_x(rng, T, K, -8, 8, heavy=rng.random() < 0.5)
            if rng.random() < 0.2:
                m = rng.random(x.shape) < 0.05
                x[m] = np.array([1, 0x80000001], np.uint32).view(F32)[rng.integers(0, 2, size=int(m.sum()))]
            q, D, bsum = np_qact(kind, x)
            bad += int(np.count_nonzero(np.any(qa_image(kind, K, q, D, bsum) != nat.qact(kind, x), axis=1)))
            done += T
        res["random_bulk"] = {"vectors": done, "mismatch": bad}
        total_bad += bad
        total_vec += done
        out["kinds"][kname] = res
    out["fp_primitives"] = fp_primitive_test(nat, rng)
    out["vectors"] = total_vec
    out["mismatch"] = total_bad
    out["passed"] = bool(total_bad == 0 and out["fp_primitives"]["mismatch"] == 0)
    return out


def fp_primitive_test(nat: Native, rng) -> dict:
    """numpy fmul/fdiv/frecip/fadd and fp16 rounding vs tsr_* on directed + random operands."""
    directed = [(0x3F7FFFFF, 0x00800000), (0x3F1FA0BA, 0x00CD4708), (0x3F7FFFFE, 0x00800000), (0x00FFFFFF, 0x3F000000),
                (0x7F7FFFFF, 0x40000000), (0x00000001, 0x3F800000), (0x80000001, 0x3F800000), (0x7FC00001, 0x3F800000),
                (0x7F800000, 0x00000000), (0xFF800000, 0x7F800000), (0x00800000, 0x00800000), (0x80000000, 0x00000000)]
    a = rng.integers(0, 2 ** 32, size=4000, dtype=np.uint64).astype(np.uint32)
    b = rng.integers(0, 2 ** 32, size=4000, dtype=np.uint64).astype(np.uint32)
    # bias toward the FTZ boundary: exponents near -126 products
    e1 = rng.integers(100, 140, size=2000)
    e2 = 253 - e1 + rng.integers(-3, 3, size=2000)
    a2 = ((e1 << 23) | rng.integers(0, 1 << 23, size=2000)).astype(np.uint32)
    b2 = ((np.clip(e2, 1, 254) << 23) | rng.integers(0, 1 << 23, size=2000)).astype(np.uint32)
    A = np.concatenate([np.array([p[0] for p in directed], np.uint32), a, a2])
    B = np.concatenate([np.array([p[1] for p in directed], np.uint32), b, b2])
    Af, Bf = A.view(F32), B.view(F32)
    bad = 0
    n = 0
    ops = (("fmul", fmul, nat.ref.tsr_fmul), ("fdiv", fdiv, nat.ref.tsr_fdiv), ("fadd", fadd, nat.ref.tsr_fadd))
    for _, f_np, f_ref in ops:
        r_np = f32bits(f_np(Af, Bf))
        for k in range(A.size):
            r = nat.bits(f_ref(nat.f32(int(A[k])), nat.f32(int(B[k]))))
            bad += int(r != int(r_np[k]))
            n += 1
    rr = f32bits(frecip(Af))
    for k in range(A.size):
        bad += int(nat.bits(nat.ref.tsr_frecip(nat.f32(int(A[k])))) != int(rr[k]))
        n += 1
    # f32 -> fp16 (RNE, overflow, fp16 subnormals): finite non-subnormal inputs (QACT's d is never NaN)
    e = rng.integers(90, 160, size=20000)
    c = ((e << 23) | rng.integers(0, 1 << 23, size=20000)).astype(np.uint32)
    c[:4] = [0x477FEFFF, 0x477FF000, 0x33000000, 0x33000001]      # 65519.99, 65520 (-> inf), 2^-25 ties
    with np.errstate(over="ignore"):
        h_np = np.asarray(c.view(F32)).astype(np.float16).view(np.uint16)
    for k in range(c.size):
        bad += int(int(nat.ref.tsr_f32_to_fp16(nat.f32(int(c[k])))) != int(h_np[k]))
        n += 1
    return {"operations": n, "mismatch": bad}


# =============================================================================================
# compare
# =============================================================================================
def selfcheck(nat: Native, args) -> dict:
    if args.boundtest_report:
        rep = json.loads(Path(args.boundtest_report).read_text())
        ok = (rep.get("bound", {}).get("bound_version") == BOUND_VERSION and rep.get("passed") is True
              and rep["bound"]["in_domain"] >= 1_000_000 and rep["bound"].get("spec_model_mismatch") == 0)
        if not ok:
            log(f"note: {args.boundtest_report} is not a passed boundtest of version {BOUND_VERSION} (rerun boundtest)")
        rep["reused_from"] = str(args.boundtest_report)
        rep["passed"] = bool(ok)
        return rep
    log(f"selfcheck: QACT model + fp primitives vs libtsr_ref, B2 bound on {args.boundtest_rows} rows")
    qa = qact_test(nat, args.seed, args.qact_random)
    log(f"  qact: {qa['vectors']} vectors, {qa['mismatch']} mismatches; fp primitives "
        f"{qa['fp_primitives']['operations']} ops, {qa['fp_primitives']['mismatch']} mismatches")
    bt = bound_test(nat, args.boundtest_rows, args.seed, args.threads)
    return {"qact": qa, "bound": bt, "passed": bool(qa["passed"] and bt["passed"])}


def nmse_rows(ref: np.ndarray, x: np.ndarray, keep) -> float:
    r = ref[keep].astype(F64)
    d = x[keep].astype(F64) - r
    sr = float(np.sum(r * r))
    return float(np.sum(d * d)) / sr if sr > 0 else math.inf


def drift_section(runs: dict, args, floor_still=()) -> dict:
    """floor_still: token positions where the floor logits equal cpu (the 1-ulp nudge vanished there);
    layer outputs also get the NMSE ratio over the other positions ("moved")."""
    cpu, dev, flo = runs["cpu"], runs["dev"], runs["floor"]
    pd, miss_d = align_nodes(cpu.nodes, dev.nodes)
    pf, miss_f = align_nodes(cpu.nodes, flo.nodes)
    fmap = {id(a): b for a, b in pf}
    rows = []
    first_div = None
    lead_identical = 0
    t0 = time.time()
    for a, b in pd:
        f = fmap.get(id(a))
        if b is None or f is None:
            continue
        rc = cpu.load(a)
        sd = cmp_stats(rc, dev.load(b))
        sf = cmp_stats(rc, flo.load(f))
        row = {"i": a["i"], "name": a["name"], "op": a["op"], "ne": a["ne"], "backend_dev": b.get("backend"),
               "weight": b.get("weight"),
               "nmse_dev": sd["nmse"], "nmse_floor": sf["nmse"], "ratio": ratio(sd["nmse"], sf["nmse"]),
               "maxabs_dev": sd["maxabs"], "maxabs_floor": sf["maxabs"], "cos_dev": sd["cos"], "cos_floor": sf["cos"],
               "rms": sd["rms_ref"], "identical_dev": sd["identical"], "identical_floor": sf["identical"],
               "nonfinite_mismatch_dev": sd["nonfinite_mismatch"]}
        rows.append(row)
        if first_div is None:
            if sd["identical"]:
                lead_identical += 1
            else:
                first_div = {"i": a["i"], "name": a["name"], "op": a["op"], "backend_dev": b.get("backend")}
    log(f"drift: {len(rows)} nodes compared in {time.time() - t0:.1f} s")
    layers = []
    for r in rows:
        m = re.match(r"^l_out-(\d+)$", r["name"])
        if m:
            layers.append({"layer": int(m.group(1)), **{k: r[k] for k in ("i", "nmse_dev", "nmse_floor", "ratio",
                                                                           "cos_dev", "cos_floor", "maxabs_dev", "rms")}})
    finals = [r for r in rows if r["name"] in ("result_norm", "result_output")]
    T = int(cpu.meta.get("n_tokens", 0))
    still = sorted(set(int(p) for p in floor_still))
    for r in layers + finals:
        r["ratio_moved"] = None
        if still and len(still) < T:
            nd = next(n for n in cpu.nodes if n["i"] == r["i"])
            if int(nd["ne"][1]) != T:
                continue
            keep = np.ones(T, bool)
            keep[still] = False
            rc = cpu.load(nd)
            r["nmse_dev_moved"] = nmse_rows(rc, dev.load(next(n for n in dev.nodes if n["i"] == r["i"])), keep)
            r["nmse_floor_moved"] = nmse_rows(rc, flo.load(next(n for n in flo.nodes if n["i"] == r["i"])), keep)
            r["ratio_moved"] = ratio(r["nmse_dev_moved"], r["nmse_floor_moved"])
    nz = [r for r in rows if r["rms"] > 0 and not r["identical_dev"]]
    worst_nmse = sorted(nz, key=lambda r: -r["nmse_dev"])[:args.top]
    worst_ratio = sorted([r for r in nz if math.isfinite(r["ratio"])], key=lambda r: -r["ratio"])[:args.top]
    ratios = np.array([r["ratio"] for r in nz if math.isfinite(r["ratio"]) and r["nmse_floor"] > 0])
    first_dev = next((r for r in rows if "tesseract" in str(r["backend_dev"]).lower()), None)
    return {"nodes_compared": len(rows), "unmatched_dev": miss_d, "unmatched_floor": miss_f,
            "leading_identical_dev": lead_identical, "first_divergent_dev": first_div,
            "first_device_node": None if first_dev is None else {k: first_dev[k] for k in ("i", "name", "op")},
            "node_ratio_median": float(np.median(ratios)) if ratios.size else None,
            "node_ratio_max": float(ratios.max()) if ratios.size else None,
            "layers": sorted(layers, key=lambda r: r["layer"]), "final": finals, "floor_still_positions": still,
            "worst_nmse": worst_nmse, "worst_ratio": worst_ratio,
            "nonfinite_mismatch_nodes": sum(1 for r in rows if r["nonfinite_mismatch_dev"]), "nodes": rows}


def logits_section(runs: dict, extra_dev=None) -> dict:
    lc = runs["cpu"].logits()
    ld = runs["dev"].logits()
    lf = runs["floor"].logits()
    T = lc.shape[0]
    pc, pd, pf = log_softmax(lc), log_softmax(ld), log_softmax(lf)
    kd = kl_rows(pc, pd)
    kf = kl_rows(pc, pf)
    am = [np.argmax(v, axis=1) for v in (lc, ld, lf)]
    out = {"n_tokens": int(T), "n_vocab": int(lc.shape[1]),
           "kl_dev_mean": float(kd.mean()), "kl_dev_max": float(kd.max()), "kl_floor_mean": float(kf.mean()),
           "kl_floor_max": float(kf.max()), "kl_ratio": ratio(float(kd.mean()), float(kf.mean())),
           "top1_dev": float(np.mean(am[0] == am[1])), "top1_floor": float(np.mean(am[0] == am[2])),
           "floor_identical_positions": [int(p) for p in range(T) if np.array_equal(f32bits(lc[p]), f32bits(lf[p]))],
           "kl_dev_mean_floor_moved": float(kd[kf > 0].mean()) if np.any(kf > 0) else None,
           "kl_floor_mean_floor_moved": float(kf[kf > 0].mean()) if np.any(kf > 0) else None,
           "per_position": [{"pos": int(p), "kl_dev": float(kd[p]), "kl_floor": float(kf[p]), "argmax_cpu": int(am[0][p]),
                             "argmax_dev": int(am[1][p]), "argmax_floor": int(am[2][p])} for p in range(T)]}
    out["observe_check"] = None
    if extra_dev is not None:                  # a Run, validated by check_run_set
        lo = extra_dev.logits()
        out["observe_check"] = {"dir": str(extra_dev.dir), "bit_identical": bool(np.array_equal(f32bits(lo), f32bits(ld))),
                                "differing": int(np.count_nonzero(f32bits(lo) != f32bits(ld)))}
    return out


def mm_section(nat: Native, model: GGUFModel, runs: dict, args) -> dict:
    layers = args.layer_set if hasattr(args, "layer_set") else parse_layers(args.mm_layers)
    pairs, _ = align_nodes(runs["dev"].nodes, runs["cpu"].nodes)
    out = []
    glob = {"b2_ratios": []}
    pv = None
    if getattr(args, "src1_from_producer", False):
        pv = producer_validation(runs, model)
        glob["use_producer"] = pv["ok"]
        log(f"src1 from producer nodes: mapping checked on {pv['checked']} src1 dumps, {pv['identical']} bit-identical, "
            f"{len(pv['differ'])} differ, {len(pv['no_producer'])} without producer -> "
            f"{'used' if pv['ok'] else 'NOT used'} for weight MUL_MATs without a src1 dump")
    t0 = time.time()
    for nd, nc in pairs:
        if not nd.get("weight"):
            continue
        if layers is not None and weight_layer(nd["weight"]) not in layers:
            continue
        if nd["op"] != "MUL_MAT":        # the dumper also tags MUL_MAT_ID (MoE experts): not modelled here
            out.append({"i": nd["i"], "name": nd["name"], "weight": nd["weight"], "backend": nd.get("backend"),
                        "layer": weight_layer(nd["weight"]), "skip": f"op {nd['op']} is not checked (B covers MUL_MAT)"})
            continue
        t1 = time.time()
        r = check_weight_mm(nat, model, nd, nc, runs, args, glob)
        r["seconds"] = round(time.time() - t1, 2)
        out.append(r)
        if not args.quiet and r.get("skip") != NOT_SAMPLED:
            if "skip" in r:
                log(f"  mm {r['name']:<22s} {r['weight']:<28s} skip: {r['skip']}")
            else:
                b3 = r["b3"]
                rt = "-" if b3["ratio"] is None else f"{b3['ratio']:.3f}"
                if b3["dominant_tokens"] and b3["ratio_all_tokens"] is not None:
                    rt += f" (all tokens {b3['ratio_all_tokens']:.3f}, token {b3['dominant_tokens'][0]['token']} dominant)"
                xs = "" if r.get("x_source") == "src1 dump" else " (x: producer)"
                log(f"  mm {r['weight']:<28s} {r['type']:5s} N={r['N']:<6d} K={r['K']:<6d} T={r['T']:<3d} "
                    f"B1 {r['b1']['mismatches']}/{r['b1']['elements']} B1b {r['b1b']['mismatches']} "
                    f"B2 viol {r['b2']['violations']} (max {r['b2']['ratio_max']:.3f}) B3 dev {b3['rel_err_dev']:.3e} "
                    f"ratio {rt}  {r['seconds']} s{xs}")
    rat = np.concatenate(glob["b2_ratios"]) if glob["b2_ratios"] else np.zeros(0, F32)
    done = [r for r in out if "skip" not in r]
    types = {}
    for r in done:
        t = types.setdefault(r["type"], {"type": r["type"], "dev_fmt": r["dev_fmt"], "native": r["native"], "tensors": 0,
                                         "b1_elements": 0, "b1_mismatches": 0, "b1b_mismatches": 0, "b2_violations": 0,
                                         "b2_ratio_max": 0.0, "tensors_with_dominant_token": 0,
                                         "s": {k: 0.0 for k in ("e2_dev", "r2_dev", "e2_cpu", "r2_cpu", "e2_dev_kept",
                                                                "r2_dev_kept", "e2_cpu_kept", "r2_cpu_kept")},
                                         "ratios": [], "ratios_all": []})
        t["tensors"] += 1
        t["b1_elements"] += r["b1"]["elements"]
        t["b1_mismatches"] += r["b1"]["mismatches"]
        t["b1b_mismatches"] += r["b1b"]["mismatches"]
        t["b2_violations"] += r["b2"]["violations"]
        t["b2_ratio_max"] = max(t["b2_ratio_max"], r["b2"]["ratio_max"])
        b3 = r["b3"]
        for k in t["s"]:
            t["s"][k] += b3[k]
        t["tensors_with_dominant_token"] += bool(b3["dominant_tokens"])
        if b3["ratio"] is not None:
            t["ratios"].append(b3["ratio"])
            t["ratios_all"].append(b3["ratio_all_tokens"])
    for t in types.values():
        s = t.pop("s")

        def rel(e, r):
            return math.sqrt(s[e] / s[r]) if s[r] > 0 else None
        rd, rc = rel("e2_dev_kept", "r2_dev_kept"), rel("e2_cpu_kept", "r2_cpu_kept")
        rda, rca = rel("e2_dev", "r2_dev"), rel("e2_cpu", "r2_cpu")
        rs, ra = np.array(t.pop("ratios")), np.array(t.pop("ratios_all"))
        t.update({"rel_err_dev_pooled": rd, "rel_err_cpu_pooled": rc, "ratio_pooled": ratio(rd, rc) if rc else None,
                  "rel_err_dev_pooled_all_tokens": rda, "rel_err_cpu_pooled_all_tokens": rca,
                  "ratio_pooled_all_tokens": ratio(rda, rca) if rca else None,
                  "ratio_median": float(np.median(rs)) if rs.size else None,
                  "ratio_min": float(rs.min()) if rs.size else None, "ratio_max": float(rs.max()) if rs.size else None,
                  "ratio_median_all_tokens": float(np.median(ra)) if ra.size else None,
                  "ratio_min_all_tokens": float(ra.min()) if ra.size else None,
                  "ratio_max_all_tokens": float(ra.max()) if ra.size else None})
    n_ns = sum(1 for r in out if r.get("skip") == NOT_SAMPLED)
    log(f"mm: {len(done)} weight MUL_MATs checked, {n_ns} not sampled, {len(out) - len(done) - n_ns} skipped, "
        f"{time.time() - t0:.1f} s")
    return {"layers": args.mm_layers or "all", "tensors": out, "types": types, "producer_validation": pv,
            "x_from_producer": sum(1 for r in done if r.get("x_source") not in (None, "src1 dump")),
            "checked": len(done), "not_sampled": n_ns, "skipped": len(out) - len(done) - n_ns,
            "b2_ftz_slack_active": sum(r["b2"]["ftz_slack_active"] for r in done),
            "b1_elements": sum(r["b1"]["elements"] for r in done),
            "b1_mismatches": sum(r["b1"]["mismatches"] for r in done),
            "b1_on_device": sum(1 for r in done if r["b1"]["checked"]),
            "b1b_mismatches": sum(r["b1b"]["mismatches"] for r in done),
            "weight_mm_selected": len(out),
            "coverage_gaps": [f"{r['weight']} (node {r['i']}): " + (r["skip"] if "skip" in r else f"ran on {r['backend']}, not on the device")
                              for r in out if (r.get("skip") not in (None, NOT_SAMPLED)) or ("skip" not in r and not r["on_device"])],
            "b2_elements": int(rat.size), "b2_violations": sum(r["b2"]["violations"] for r in done),
            "b2_out_of_domain": sum(r["b2"]["out_of_domain"] for r in done),
            "b2_ratio_median": float(np.median(rat)) if rat.size else None,
            "b2_ratio_p99": pct(rat, 99), "b2_ratio_max": float(rat.max()) if rat.size else None,
            "qact_model_mismatch_tokens": sum(r["qact_model_mismatch_tokens"] for r in done)}


EXIT = {"PASS": 0, "FAIL": 1, "INCOMPLETE": 3}     # 2: usage / input error or crash (die(), exceptions)


def verdict(rep: dict) -> tuple[str, list, list, list]:
    """PASS / FAIL / INCOMPLETE (no failure, but a selected weight MUL_MAT could not be checked), fails, gaps, warnings."""
    mm, sc = rep["mm"], rep["selfcheck"]
    fails, warns = [], []
    gaps = list(mm["coverage_gaps"])
    if not sc.get("passed"):
        fails.append("bound/QACT/spec-model self-test did not pass")
    if mm["b1_on_device"] == 0:
        fails.append("no weight MUL_MAT ran on the device (B1 checked nothing)")
    if mm["b1_mismatches"]:
        fails.append(f"B1: {mm['b1_mismatches']} elements differ from the libtsr_ref model")
    if mm["b1b_mismatches"]:
        fails.append(f"B1b: {mm['b1b_mismatches']} elements differ from the independent numpy spec model")
    if mm["b2_violations"]:
        fails.append(f"B2: {mm['b2_violations']} elements outside the proven fp32 bound")
    if mm["qact_model_mismatch_tokens"]:
        fails.append(f"numpy QACT model != libtsr_ref QACT on {mm['qact_model_mismatch_tokens']} real token vectors")
    if mm["b2_out_of_domain"]:
        warns.append(f"B2: {mm['b2_out_of_domain']} elements outside the bound's domain (non-finite / overflow), not verified")
    pv = mm.get("producer_validation")
    if pv is not None and not pv["ok"]:
        warns.append(f"--src1-from-producer not used: producer outputs vs src1 dumps: {pv['checked']} checked, "
                     f"{len(pv['differ'])} differ, {len(pv['no_producer'])} without producer")
    if mm["not_sampled"]:
        warns.append(f"{mm['not_sampled']} weight MUL_MATs have no src1 dump (not sampled by the dumper), B not checked there")
    for r in mm["tensors"]:
        if "skip" in r:
            continue
        rt = r["b3"]["ratio"]
        if r["native"] and rt is not None and rt > 1.5:
            dn = r["b3"]["dominant_tokens"]
            warns.append(f"native {r['type']} {r['weight']}: rel_err_dev/rel_err_cpu = {rt:.3f} > 1.5"
                         + (f" (without dominant token(s) {[d['token'] for d in dn]})" if dn else ""))
    dr = rep["drift"]
    if dr["unmatched_dev"] or dr["unmatched_floor"]:
        warns.append(f"graph nodes not matched: dev {dr['unmatched_dev']}, floor {dr['unmatched_floor']}")
    fd, fdev = dr["first_divergent_dev"], dr["first_device_node"]
    if fd is not None and fdev is not None and fd["i"] < fdev["i"]:
        warns.append(f"dev diverges from cpu at node {fd['i']} {fd['name']} ({fd['op']}) before the first device node {fdev['i']}")
    if dr["nonfinite_mismatch_nodes"]:
        warns.append(f"{dr['nonfinite_mismatch_nodes']} nodes with non-finite values differing between dev and cpu")
    oc = rep["logits"].get("observe_check")
    if oc is not None and not oc["bit_identical"]:
        fails.append(f"dev logits change when nodes are observed ({oc['differing']} values)")
    if oc is None:
        warns.append("no --dev-noobs: observation invariance not checked; B1 / B1b cover the observed (node-by-node) "
                     "execution only")
    return ("FAIL" if fails else "INCOMPLETE" if gaps else "PASS"), fails, gaps, warns


def e(x, p=3):
    if x is None:
        return "-"
    if isinstance(x, float) and (math.isinf(x) or math.isnan(x)):
        return str(x)
    return f"{x:.{p}e}"


def f(x, p=3):
    return "-" if x is None else (str(x) if isinstance(x, float) and not math.isfinite(x) else f"{x:.{p}f}")


def write_md(rep: dict, path: Path) -> None:
    L = []
    w = L.append
    w(f"# numcheck report: {rep['verdict']}\n")
    w(f"model `{rep['model']}`, prompt {rep['logits']['n_tokens']} tokens, runs cpu `{rep['runs']['cpu']}`, "
      f"floor `{rep['runs']['floor']}`, dev `{rep['runs']['dev']}`; {rep['seconds']} s\n")
    st = rep.get("settings") or {}
    if st:
        w(f"settings ({st.get('phase')}, every run): flash attention {st.get('flash_attn')}, KV {st.get('type_k')}/"
          f"{st.get('type_v')} on the {'device' if st.get('offload_kqv') else 'host'}, KV rotation "
          f"{'off (LLAMA_ATTN_ROT_DISABLE=1)' if st.get('attn_rot_disable') else 'not disabled'}; device "
          f"`{rep.get('device')}`\n")
    vs = rep.get("versions")
    if vs:
        kinds = " ".join(f"{k} r{r}" for k, r in (vs.get("kinds") or {}).items())
        w(f"versions (dev run): Tesseract Core {vs.get('core')}, Tesseract Basis {vs.get('basis')} (ABI {vs.get('abi')}, "
          f"image {vs.get('image')}, kind table {vs.get('kinds_sig')}: {kinds})\n")
    else:
        w("versions (dev run): not recorded\n")
    w("PASS iff B1 (libtsr_ref model) and B1b (independent numpy spec model) have zero mismatches, B2 zero bound "
      "violations, the bound/QACT/spec-model self-test passed and every selected weight MUL_MAT was checked on the "
      "device (else INCOMPLETE). B3 and drift are reported, not judged (WARN for native tensors with "
      "rel_err_dev/rel_err_cpu > 1.5, dominant tokens left out).\n")
    for x in rep["fails"]:
        w(f"- **FAIL** {x}")
    for x in rep.get("coverage_gaps", []):
        w(f"- **not checked** {x}")
    for x in rep["warnings"]:
        w(f"- WARN {x}")
    if rep["fails"] or rep["warnings"] or rep.get("coverage_gaps"):
        w("")
    sc = rep["selfcheck"]
    w("## Self-test (model-free)\n")
    if "bound" in sc:
        b = sc["bound"]
        w(f"- B2 bound vs libtsr_ref: {b['rows']} dot products, {b['in_domain']} in domain, **{b['violations']} violations**; "
          f"|err|/bound median {e(b['ratio_median'], 2)}, p99 {e(b['ratio_p99'], 2)}, max {f(b['ratio_max'])}; "
          f"rows where no term can flush ({b['in_domain'] - b.get('ftz_rows', 0)}): median {e(b.get('no_ftz_ratio_median'), 2)}, "
          f"max {f(b.get('no_ftz_ratio_max'))} ({b['seconds']} s)")
        w(f"- numpy spec model (B1b) vs libtsr_ref on the same {b['rows']} dot products: **{b.get('spec_model_mismatch')} "
          f"mismatches**; sequential-sum mutant: differs in {b.get('seq_mutant_differs')} of {b.get('multi_term_rows')} "
          f"rows with more than one term")
    if "qact" in sc:
        q = sc["qact"]
        w(f"- numpy QACT model vs libtsr_ref QACT: {q['vectors']} vectors (A32 + A256, directed + random), "
          f"**{q['mismatch']} mismatches**; fp32 primitives {q['fp_primitives']['operations']} ops, "
          f"{q['fp_primitives']['mismatch']} mismatches")
    if "reused_from" in sc:
        w(f"- reused from `{sc['reused_from']}`")
    w(f"- passed: {sc['passed']}\n")
    if "bound" in sc:
        w("| case | rows | in domain | violations | FTZ-slack rows | median | p99.9 | max | max (no FTZ) |")
        w("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for k, c in sc["bound"]["cases"].items():
            w(f"| {k} | {c['rows']} | {c['in_domain']} | {c['violations']} | {c.get('ftz_rows', '-')} | "
              f"{e(c['ratio_median'], 2)} | {e(c['ratio_p999'], 2)} | {f(c['ratio_max'])} | {f(c.get('no_ftz_ratio_max'))} |")
        w("")
    lg = rep["logits"]
    w("## Logits\n")
    w("| | mean KL(cpu‖x) | max KL | top-1 vs cpu |")
    w("| --- | ---: | ---: | ---: |")
    w(f"| dev | {e(lg['kl_dev_mean'])} | {e(lg['kl_dev_max'])} | {lg['top1_dev']:.3f} |")
    w(f"| floor (1 ulp) | {e(lg['kl_floor_mean'])} | {e(lg['kl_floor_max'])} | {lg['top1_floor']:.3f} |")
    w(f"\nKL ratio dev/floor = **{f(lg['kl_ratio'])}**\n")
    fz = lg["floor_identical_positions"]
    if fz:
        # no position moved at all (tiny models at small T): the means are None, there is no ratio
        kdm, kfm = lg["kl_dev_mean_floor_moved"], lg["kl_floor_mean_floor_moved"]
        w(f"- floor logits bit-identical to cpu at {len(fz)} positions {fz}: the residual stream there still carries "
          f"the 1-ulp nudge, but every int8 activation quantization of the CPU (each MUL_MAT, the output head "
          f"included) maps it to the same ints and scales. Means over the other positions: KL dev "
          f"{e(kdm)}, floor {e(kfm)}, ratio {f(ratio(kdm, kfm) if kdm is not None and kfm is not None else None)}")
    if lg.get("observe_check") is not None:
        w(f"- dev logits with vs without observation (`{lg['observe_check']['dir']}`) bit-identical: "
          f"{lg['observe_check']['bit_identical']}")
    else:
        w("- no `--dev-noobs`: dev logits without observation not compared (B1 / B1b cover node-by-node execution)")
    w("")
    mm = rep["mm"]
    w("## Weight MUL_MATs (B)\n")
    w(f"layers: {mm['layers']}; {mm['checked']} checked, {mm['not_sampled']} not sampled (no src1 dump), "
      f"{mm['skipped']} skipped. "
      f"B1: {mm['b1_mismatches']} mismatches in {mm['b1_elements']} elements ({mm['b1_on_device']} tensors on the device). "
      f"B1b (numpy spec model): {mm['b1b_mismatches']} mismatches. "
      f"B2: {mm['b2_violations']} violations in {mm['b2_elements']} elements, |err|/bound median {e(mm['b2_ratio_median'], 2)}, "
      f"p99 {e(mm['b2_ratio_p99'], 2)}, max {f(mm['b2_ratio_max'])}; out of domain {mm['b2_out_of_domain']}, "
      f"FTZ slack active in {mm['b2_ftz_slack_active']} elements. "
      f"numpy QACT vs libtsr_ref on real x: {mm['qact_model_mismatch_tokens']} differing tokens.\n")
    pv = mm.get("producer_validation")
    if pv is not None:
        w(f"x of weight MUL_MATs without a src1 dump taken from the producer node's output (`--src1-from-producer`): "
          f"{'used' if pv['ok'] else 'NOT used'} for {mm.get('x_from_producer', 0)} tensors; mapping verified on "
          f"{pv['checked']} src1 dumps (dev + cpu), {pv['identical']} bit-identical, {len(pv['differ'])} differ, "
          f"{len(pv['no_producer'])} without producer.\n")
    w(f"B3 = ‖y−Wx‖ / ‖Wx‖. A token holding more than {DOMINANT_SHARE:.0%} of a tensor's ‖Wx‖² (a massive activation) "
      f"would decide the pooled value alone; such tokens are listed and left out of the B3 ratios below (the "
      f"\"all tokens\" columns keep them). Per type: pooled = sqrt(Σ‖y−Wx‖² / Σ‖Wx‖²) over the type's tensors; "
      f"median / min / max of the per-tensor ratios.\n")
    w("| type | dev fmt | tensors | B1 mism. | B1b mism. | B2 viol. | B2 max ratio | rel_err dev | rel_err cpu | dev/cpu pooled | "
      "median | min | max | tensors with dominant token | pooled, all tokens | median, all tokens | max, all tokens |")
    w("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for t in mm["types"].values():
        w(f"| {t['type']} | {t['dev_fmt']}{'' if t['native'] else ' (converted)'} | {t['tensors']} | {t['b1_mismatches']} | "
          f"{t['b1b_mismatches']} | {t['b2_violations']} | {f(t['b2_ratio_max'])} | {e(t['rel_err_dev_pooled'])} | "
          f"{e(t['rel_err_cpu_pooled'])} | {f(t['ratio_pooled'])} | {f(t['ratio_median'])} | {f(t['ratio_min'])} | "
          f"{f(t['ratio_max'])} | {t['tensors_with_dominant_token']} | {f(t['ratio_pooled_all_tokens'])} | "
          f"{f(t['ratio_median_all_tokens'])} | {f(t['ratio_max_all_tokens'])} |")
    w("\nPer tensor:\n")
    w("| node | weight | type | N | K | T | x | B1 mism./elems | B1b mism. | B2 viol. | B2 median | B2 max | rel_err dev | "
      "rel_err cpu | dev/cpu | dominant token (share) | dev/cpu all tokens |")
    w("| ---: | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |")
    for r in mm["tensors"]:
        if r.get("skip") == NOT_SAMPLED:
            continue
        if "skip" in r:
            w(f"| {r['i']} | {r['weight']} | {r.get('type', '-')} | | | | | not checked: {r['skip']} | | | | | | | | | |")
            continue
        b1 = f"{r['b1']['mismatches']}/{r['b1']['elements']}" if r["b1"]["checked"] else "not on device"
        b1b = r["b1b"]["mismatches"] if r["b1b"]["checked"] else "-"
        xs = "src1" if r.get("x_source") == "src1 dump" else "producer"
        b3 = r["b3"]
        dn = ", ".join(f"{d['token']} ({d['share']:.3f})" for d in b3["dominant_tokens"]) or "-"
        w(f"| {r['i']} | {r['weight']} | {r['type']} | {r['N']} | {r['K']} | {r['T']} | {xs} | {b1} | {b1b} | "
          f"{r['b2']['violations']} | {e(r['b2']['ratio_median'], 2)} | {f(r['b2']['ratio_max'])} | {e(b3['rel_err_dev'])} | "
          f"{e(b3['rel_err_cpu'])} | {f(b3['ratio'])} | {dn} | {f(b3['ratio_all_tokens'])} |")
    bad = [r for r in mm["tensors"] if "skip" not in r and (r["b1"]["first"] or r["b1b"]["first"] or r["b2"]["first"])]
    if bad:
        w("\nFirst mismatches / violations:\n")
        for r in bad:
            for x in r["b1"]["first"]:
                w(f"- B1 {r['weight']} row {x['row']} token {x['token']}: dev {x['dev']!r} ({x['dev_bits']}) vs model "
                  f"{x['model']!r} ({x['model_bits']}), {x['ulp']} ulp")
            for x in r["b1b"]["first"]:
                w(f"- B1b {r['weight']} row {x['row']} token {x['token']}: dev {x['dev']!r} ({x['dev_bits']}) vs numpy "
                  f"spec model {x['model']!r} ({x['model_bits']}), {x['ulp']} ulp")
            for x in r["b2"]["first"]:
                w(f"- B2 {r['weight']} row {x['row']} token {x['token']}: dev {x['dev']!r}, y_q64 {x['y_q64']!r}, "
                  f"|err| {x['abs_err']:.3e} > bound {x['bound']:.3e}")
    dr = rep["drift"]
    w("\n## Drift (A)\n")
    w(f"{dr['nodes_compared']} nodes; dev bit-identical to cpu for the first {dr['leading_identical_dev']} nodes "
      f"(first divergent: {dr['first_divergent_dev']}, first device node: {dr['first_device_node']}). "
      f"Node NMSE ratio dev/floor: median {f(dr['node_ratio_median'])}, max {f(dr['node_ratio_max'])}.\n")
    st = dr.get("floor_still_positions") or []
    w("Layer outputs" + (f" (\"ratio, floor moved\": NMSE ratio over the positions other than {st}, where the floor "
                         f"logits are bit-identical to cpu)" if st else "") + ":\n")
    w("| layer | NMSE dev | NMSE floor | ratio | ratio, floor moved | cos dev | cos floor | max|Δ| dev | rms |")
    w("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in dr["layers"] + dr["final"]:
        w(f"| {r.get('layer', r.get('name'))} | {e(r['nmse_dev'])} | {e(r['nmse_floor'])} | {f(r['ratio'])} | "
          f"{f(r.get('ratio_moved'))} | {r['cos_dev']:.8f} | {r['cos_floor']:.8f} | {e(r['maxabs_dev'])} | {e(r['rms'])} |")
    for title, key in (("Worst nodes by NMSE dev", "worst_nmse"), ("Worst nodes by NMSE ratio dev/floor", "worst_ratio")):
        w(f"\n{title}:\n")
        w("| node | name | op | backend (dev) | NMSE dev | NMSE floor | ratio | cos dev |")
        w("| ---: | --- | --- | --- | ---: | ---: | ---: | ---: |")
        for r in dr[key]:
            w(f"| {r['i']} | {r['name']} | {r['op']} | {r['backend_dev']} | {e(r['nmse_dev'])} | {e(r['nmse_floor'])} | "
              f"{f(r['ratio'])} | {r['cos_dev']:.8f} |")
    w("\nPer position (logits):\n")
    w("| pos | KL dev | KL floor | argmax cpu | dev | floor |")
    w("| ---: | ---: | ---: | ---: | ---: | ---: |")
    for p in lg["per_position"]:
        w(f"| {p['pos']} | {e(p['kl_dev'])} | {e(p['kl_floor'])} | {p['argmax_cpu']} | {p['argmax_dev']} | {p['argmax_floor']} |")
    path.write_text("\n".join(L) + "\n")


def json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))


def sanitize(o):
    if isinstance(o, float) and not math.isfinite(o):
        return str(o)
    if isinstance(o, dict):
        return {k: sanitize(v) for k, v in o.items()}
    if isinstance(o, list):
        return [sanitize(v) for v in o]
    return o


MODEL_KEYS = ("model_desc", "n_params", "n_layer", "n_embd")     # meta.json "info" fields that identify the model
# meta.json "info" fields of the settings (SW-22); a version 1 dump has the FA-off values
SETTING_KEYS = ("phase", "flash_attn", "type_k", "type_v", "offload_kqv", "attn_rot_disable")
V1_SETTINGS = {"phase": "v1", "flash_attn": "off", "type_k": "f16", "type_v": "f16", "offload_kqv": False, "attn_rot_disable": False}


def run_settings(meta: dict) -> dict:
    """The settings a run was made with (version 1: the FA-off settings)."""
    if meta.get("version", 1) < 2:
        return dict(V1_SETTINGS)
    info = meta.get("info") or {}
    return {k: info.get(k) for k in SETTING_KEYS}


def check_run_set(runs: dict, noobs) -> list:
    """Dies unless cpu, floor, dev (and dev-noobs) come from one model and one prompt; returns notes."""
    notes = []
    ref = runs["dev"].meta
    others = dict(runs)
    if noobs is not None:
        others["dev-noobs"] = noobs
    for k, r in others.items():
        m = r.meta
        if os.path.realpath(str(m.get("model"))) != os.path.realpath(str(ref.get("model"))):
            die(f"{k} run is from model {m.get('model')!r}, dev from {ref.get('model')!r}: the runs must come from one model")
        for f in MODEL_KEYS:
            a, b = (m.get("info") or {}).get(f), (ref.get("info") or {}).get(f)
            if a is not None and b is not None and a != b:
                die(f"{k} run: info.{f} = {a!r}, dev has {b!r} (different model files under one path?)")
        for f in ("n_tokens", "n_vocab", "tokens"):
            if m.get(f) != ref.get(f):
                die(f"{k} run: {f} differs from the dev run (the runs must use the same prompt and model)")
        if run_settings(m) != run_settings(ref):    # one comparison, one attention / KV configuration (SW-22)
            die(f"{k} run: settings {run_settings(m)} differ from the dev run's {run_settings(ref)}")
    if noobs is not None:
        m = noobs.meta
        if os.path.realpath(noobs.dir) == os.path.realpath(runs["dev"].dir):
            die("--dev-noobs is the observed dev run itself; pass the run made with --no-observe")
        info = m.get("info") or {}
        if m.get("run") != "dev" or m.get("input") != "embd" or info.get("observe") is not False:
            die(f"--dev-noobs {noobs.dir}: run {m.get('run')!r}, input {m.get('input')!r}, info.observe "
                f"{info.get('observe')!r}; expected a dev run with input embd made with --no-observe")
        if m.get("device") != ref.get("device"):
            die(f"--dev-noobs device {m.get('device')!r} != dev device {ref.get('device')!r}")
    return notes


def check_model_file(args, runs: dict, model: GGUFModel, nat: Native) -> list:
    """The GGUF the comparer reads must be the dev run's model: same path (or, if that file is gone, the same
    file name) and, for every weight MUL_MAT node, the same weight type and shape (parameter count: note)."""
    notes = []
    dm = runs["dev"].meta
    if args.model and os.path.realpath(args.model) != os.path.realpath(str(dm.get("model"))):
        if os.path.exists(str(dm.get("model"))):
            die(f"-m {args.model} is not the dev run's model {dm.get('model')} (omit -m to use the run's model)")
        if Path(args.model).name != Path(str(dm.get("model"))).name:
            die(f"-m {args.model}: the dev run's model {dm.get('model')} no longer exists and the file names differ")
        notes.append(f"dev run's model {dm.get('model')} not found; using -m {args.model} (same file name)")
    n_par = (dm.get("info") or {}).get("n_params")
    tot = sum(int(np.prod(v[3])) for v in model.tensors.values())
    if n_par is not None and int(n_par) != tot:     # a note only: llama.cpp may count a tied output head twice
        notes.append(f"{model.path}: {tot} elements in GGUF tensors, the dev run's llama_model_n_params was {n_par}")
    bad = []
    for label, r in runs.items():
        for nd in r.nodes:
            if not nd.get("weight"):
                continue
            info = model.info(nd["weight"])
            if info is None:
                continue                       # reported per tensor ("weight not in GGUF")
            typ_name = nat.type_name(info[2])
            K, N = info[3][0], info[3][1]
            if (nd.get("weight_type") not in (None, typ_name) or int(nd["ne"][0]) != N
                    or (nd.get("src1_ne") and int(nd["src1_ne"][0]) != K)):
                bad.append(f"{label}:{nd['weight']} ({nd.get('weight_type')}, ne {nd['ne']}) vs GGUF {typ_name} [{K}, {N}]")
    if bad:
        die(f"{len(bad)} weight MUL_MATs do not match {model.path}: " + "; ".join(bad[:4]))
    return notes


def clear_report(out: Path) -> None:
    for name in ("report.json", "report.md", "drift_nodes.json"):
        try:
            (out / name).unlink()
        except FileNotFoundError:
            pass


def write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def cmd_compare(args) -> int:
    t0 = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    clear_report(out)                           # an error or crash must not leave an older report behind
    try:                                        # validated before any work (the dumper's syntax)
        args.layer_set = parse_layers(args.mm_layers)
    except ValueError as ex:
        die(f"--mm-layers {args.mm_layers!r}: {ex}")
    nat = Native(Path(args.build))
    runs = {"cpu": Run(args.cpu, "cpu"), "floor": Run(args.floor, "floor"), "dev": Run(args.dev, "dev")}
    noobs = Run(args.dev_noobs, "dev-noobs") if args.dev_noobs else None
    run_warns = []
    want_input = {"cpu": "embd", "floor": "embd+1ulp", "dev": "embd"}
    for k, r in runs.items():
        if r.meta.get("run") not in (None, k):
            run_warns.append(f"{r.dir}/meta.json says run={r.meta.get('run')!r}, used as {k}")
        if r.meta.get("input") not in (None, want_input[k]):
            run_warns.append(f"{k} run has input {r.meta.get('input')!r}, expected {want_input[k]!r} (graphs may differ)")
    if runs["dev"].meta.get("device") in ("none",) or runs["cpu"].meta.get("device") not in (None, "none"):
        run_warns.append(f"devices: cpu {runs['cpu'].meta.get('device')!r}, dev {runs['dev'].meta.get('device')!r}")
    run_warns += check_run_set(runs, noobs)
    model_path = args.model or runs["dev"].meta.get("model")
    if not model_path:
        die("no model: pass -m")
    model = GGUFModel(model_path)
    run_warns += check_model_file(args, runs, model, nat)
    for x in run_warns:
        log(f"note: {x}")
    toks = runs["dev"].meta.get("tokens")
    rep = {"format": "tsr-numcheck-report", "version": 2, "model": model_path,
           "runs": {k: str(r.dir) for k, r in runs.items()}, "tokens": toks,
           "settings": run_settings(runs["dev"].meta), "device": runs["dev"].meta.get("device"),
           # SW §6.9: the Tesseract Core capability, Tesseract Basis version and kind revisions of the
           # dev run (tsr-numcheck's meta.json; None for runs written before the field)
           "versions": runs["dev"].meta.get("versions")}
    rep["selfcheck"] = selfcheck(nat, args)
    rep["logits"] = logits_section(runs, noobs)
    lg = rep["logits"]
    log(f"logits: KL dev {lg['kl_dev_mean']:.3e} floor {lg['kl_floor_mean']:.3e} ratio {lg['kl_ratio']:.3f}; "
        f"top-1 dev {lg['top1_dev']:.3f} floor {lg['top1_floor']:.3f}")
    rep["drift"] = drift_section(runs, args, lg["floor_identical_positions"])
    rep["mm"] = mm_section(nat, model, runs, args)
    model.close()
    rep["verdict"], rep["fails"], rep["coverage_gaps"], rep["warnings"] = verdict(rep)
    rep["warnings"] = run_warns + rep["warnings"]
    rep["seconds"] = round(time.time() - t0, 1)
    full_nodes = rep["drift"].pop("nodes")
    write_atomic(out / "drift_nodes.json", json.dumps(sanitize(full_nodes), default=json_default) + "\n")
    md = out / "report.md.tmp"
    write_md(rep, md)
    os.replace(md, out / "report.md")
    write_atomic(out / "report.json", json.dumps(sanitize(rep), indent=1, default=json_default) + "\n")
    log(f"verdict: {rep['verdict']}")
    for x in rep["fails"]:
        log(f"  FAIL {x}")
    for x in rep["coverage_gaps"][:20]:
        log(f"  NOT CHECKED {x}")
    for x in rep["warnings"][:20]:
        log(f"  WARN {x}")
    log(f"report: {out / 'report.md'}")
    return EXIT[rep["verdict"]]


def cmd_boundtest(args) -> int:
    nat = Native(Path(args.build))
    qa = qact_test(nat, args.seed, args.qact_random)
    log(f"qact: {qa['vectors']} vectors, {qa['mismatch']} mismatches; fp primitives {qa['fp_primitives']['operations']} ops, "
        f"{qa['fp_primitives']['mismatch']} mismatches")
    bt = bound_test(nat, args.rows, args.seed, args.threads)
    rep = {"qact": qa, "bound": bt, "passed": bool(qa["passed"] and bt["passed"])}
    log(f"bound: {bt['rows']} rows, {bt['in_domain']} in domain, {bt['violations']} violations, |err|/bound median "
        f"{bt['ratio_median']:.3e} p99 {bt['ratio_p99']:.3e} max {bt['ratio_max']:.4f}; rows without FTZ slack "
        f"{bt['in_domain'] - bt['ftz_rows']}: median {bt['no_ftz_ratio_median']:.3e} max {bt['no_ftz_ratio_max']:.4f} "
        f"({bt['seconds']} s)")
    log(f"spec model (B1b): {bt['spec_model_mismatch']} mismatches vs libtsr_ref on {bt['rows']} rows; sequential-sum "
        f"mutant differs in {bt['seq_mutant_differs']} of {bt['multi_term_rows']} multi-term rows "
        f"({100.0 * bt['seq_mutant_differs'] / max(1, bt['multi_term_rows']):.1f} %)")
    log(f"boundtest: {'PASS' if rep['passed'] else 'FAIL'}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(sanitize(rep), indent=1, default=json_default) + "\n")
    return 0 if rep["passed"] else 1


# =============================================================================================
# selftest: synthetic GGUF + fake run dirs
# =============================================================================================
def write_synth_gguf(nat: Native, path: Path, rng) -> dict:
    if str(GGUF_PY) not in sys.path:
        sys.path.insert(0, str(GGUF_PY))
    from gguf import GGMLQuantizationType, GGUFWriter  # noqa: WPS433

    specs = {  # name: (ggml type, K, N)
        "blk.0.attn_q.weight": (T_Q8_0, 256, 96),
        "blk.0.ffn_up.weight": (T_Q4_K, 512, 80),
        "blk.0.ffn_down.weight": (T_Q6_K, 256, 64),
        "blk.1.attn_k.weight": (T_F16, 128, 48),
        "blk.1.attn_v.weight": (T_F32, 96, 32),
        "output.weight": (T_Q8_0, 128, 200),
    }
    w = GGUFWriter(str(path), arch="numcheck-synth")
    for name, (typ, K, N) in specs.items():
        if typ == T_Q8_0:
            raw = nat.quantize_q8_0(rand_x(rng, N, K, -2, 0))
        elif typ == T_Q4_K:
            raw = make_q4_K_blocks(rand_fp16(rng, (N, K // 256), 8, 12), rand_fp16(rng, (N, K // 256), 7, 11),
                                   rng.integers(0, 64, size=(N, K // 256, 8)), rng.integers(0, 64, size=(N, K // 256, 8)),
                                   rng.integers(0, 16, size=(N, K // 256, 256)))
        elif typ == T_Q6_K:      # 210 B / 256: ql[128] qh[64] scales[16] (int8) d (fp16)
            nb = K // 256
            a = rng.integers(0, 256, size=(N, nb, 210)).astype(np.uint8)
            a[:, :, 192:208] = rng.integers(-40, 40, size=(N, nb, 16)).astype(np.int8).view(np.uint8)
            a[:, :, 208:210] = rand_fp16(rng, (N, nb), 7, 10).astype("<u2").view(np.uint8).reshape(N, nb, 2)
            raw = a.reshape(-1)
        elif typ == T_F16:
            raw = rand_x(rng, N, K, -2, 0).astype(np.float16).view(np.uint8).reshape(-1)
        else:
            raw = rand_x(rng, N, K, -2, 0).view(np.uint8).reshape(-1)
        rs = nat.row_size(typ, K)
        qt = GGMLQuantizationType(typ)
        if typ in (T_F16, T_F32):
            arr = raw.view(np.float16 if typ == T_F16 else F32).reshape(N, K)
            w.add_tensor(name, arr)
        else:
            w.add_tensor(name, raw.reshape(N, rs), raw_dtype=qt)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return specs


# The versions block a dev run's meta.json carries (tsr-numcheck, SW §6.9), for the synthetic runs.
SYNTH_VERSIONS = {"core": "1.0", "caps": 105, "basis": "0.1.0", "abi": 1, "image": "0123456789abcdef", "kinds_sig": "0x512e",
                  "kinds": {"NOP": 1, "JOIN": 1, "MUL_MAT": 1}}


def write_synth_runs(nat: Native, root: Path, model: Path, specs: dict, rng, T=6, V=256) -> dict:
    """cpu / floor / dev run dirs with a fake graph: norm -> weight MUL_MATs -> l_out nodes -> result_output."""
    rng_x = np.random.default_rng(11)
    nodes = []
    gm = GGUFModel(str(model))

    def add(name, op, data: dict, weight=None, wtype=None, src1: dict | None = None):
        nodes.append({"name": name, "op": op, "data": data, "weight": weight, "wtype": wtype, "src1": src1})

    x0 = rand_x(rng_x, T, 256, -1, 0)
    xs = {"cpu": x0, "floor": x0 + np.float32(1e-7) * x0, "dev": x0}
    add("attn_norm-0", "RMS_NORM", dict(xs))
    for name, (typ, K, N) in specs.items():
        info = gm.info(name)
        raw = gm.read_rows(nat, name, 0, N)
        Wf = nat.to_float(typ, raw, N * K).reshape(N, K).astype(F64)
        fmt, blocks = nat.dev_blocks(typ, raw, N, K)
        xc = rand_x(rng_x, T, K, -1, 0)
        xin = {"cpu": xc, "floor": (xc * np.float32(1 + 2e-7)).astype(F32), "dev": (xc * np.float32(1 + 1e-6)).astype(F32)}
        y = {}
        for k in ("cpu", "floor"):          # fake CPU: exact product + a little noise
            ye = Wf @ xin[k].astype(F64).T
            y[k] = (ye.T * (1 + 3e-4 * rng_x.standard_normal(ye.T.shape))).astype(F32)
        y["dev"] = nat.model_mm(fmt, blocks, N, K, xin["dev"], 2)
        add(producer_names(name)[0], "MUL", dict(xin))          # the node whose output is src1 (llama.cpp names)
        add(name.replace(".weight", "") + "-mm", "MUL_MAT", y, weight=name, wtype=nat.type_name(typ), src1=xin)
        assert info is not None
    gm.close()
    for il in range(2):
        base = rand_x(rng_x, T, 64, 0, 1)
        add(f"l_out-{il}", "ADD", {"cpu": base, "floor": base * np.float32(1 + 1e-6), "dev": base * np.float32(1 + 2e-6)})
    lg = rand_x(rng_x, T, V, 0, 1)
    add("result_output", "MUL_MAT", {"cpu": lg, "floor": lg + np.float32(1e-3), "dev": lg + np.float32(2e-3)})
    dirs = {}
    for run in ("cpu", "floor", "dev"):
        d = root / run
        (d / "t").mkdir(parents=True, exist_ok=True)
        meta_nodes = []
        for i, n in enumerate(nodes):
            arr = np.ascontiguousarray(n["data"][run], dtype="<f4")
            fn = f"t/{i:06d}.f32"
            arr.tofile(d / fn)
            ent = {"i": i, "name": n["name"], "op": n["op"], "ne": [int(arr.shape[1]), int(arr.shape[0]), 1, 1], "file": fn,
                   "backend": "Tesseract0" if (run == "dev" and n["weight"]) else "CPU",
                   "weight": n["weight"], "weight_type": n["wtype"], "src1_file": None, "src1_ne": None}
            if n["src1"] is not None:
                s = np.ascontiguousarray(n["src1"][run], dtype="<f4")
                s.tofile(d / f"t/{i:06d}.src1.f32")
                ent["src1_file"] = f"t/{i:06d}.src1.f32"
                ent["src1_ne"] = [int(s.shape[1]), int(s.shape[0]), 1, 1]
            meta_nodes.append(ent)
        np.ascontiguousarray(nodes[-1]["data"][run], dtype="<f4").tofile(d / "logits.f32")
        meta = {"format": "tsr-numcheck", "version": 2, "model": str(model), "run": run,
                "device": "sim" if run == "dev" else "none", "input": "embd+1ulp" if run == "floor" else "embd",
                "seed": 1, "n_tokens": T, "tokens": list(range(T)), "n_vocab": V, "logits_file": "logits.f32",
                "versions": SYNTH_VERSIONS if run == "dev" else None,
                "info": {"phase": "m6", "flash_attn": "on", "type_k": "q8_0", "type_v": "q8_0", "offload_kqv": False,
                         "attn_rot_disable": True},
                "nodes": meta_nodes}
        (d / "meta.json").write_text(json.dumps(meta, indent=1))
        dirs[run] = d
    return dirs


def cmd_selftest(args) -> int:
    nat = Native(Path(args.build))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ok = True
    t0 = time.time()
    # 1. QACT model + bound (full size: >= 1e6 in-domain rows)
    bt_path = out / "boundtest.json"
    rc = cmd_boundtest(argparse.Namespace(build=args.build, seed=args.seed, qact_random=args.qact_random, rows=args.rows,
                                          threads=args.threads, out=str(bt_path)))
    ok &= rc == 0
    # 2. synthetic compare must PASS
    rng = np.random.default_rng(5)
    model = out / "synth.gguf"
    specs = write_synth_gguf(nat, model, rng)
    dirs = write_synth_runs(nat, out / "runs", model, specs, rng)
    base = dict(build=args.build, cpu=str(dirs["cpu"]), floor=str(dirs["floor"]), dev=str(dirs["dev"]), model=str(model),
                mm_layers=None, threads=args.threads, chunk_elems=1 << 12, boundtest_report=str(bt_path),
                boundtest_rows=0, qact_random=0, seed=args.seed, top=5, quiet=True, dev_noobs=None)
    rc = cmd_compare(argparse.Namespace(**base, out=str(out / "report-clean")))
    rep = json.loads((out / "report-clean" / "report.json").read_text())
    types = rep["mm"]["types"]
    bt = json.loads(bt_path.read_text())["bound"]
    checks = [("boundtest: >= 1e6 rows in domain, 0 violations", bt["in_domain"] >= 1_000_000 and bt["violations"] == 0),
              ("boundtest is sensitive: half the bound would be violated (also without FTZ slack)",
               bt["ratio_max"] > 0.5 and bt["no_ftz_ratio_max"] > 0.5),
              ("clean synthetic compare PASS", rc == 0 and rep["verdict"] == "PASS"),
              ("6 weight MUL_MATs checked", rep["mm"]["checked"] == 6),
              ("converted types present (q6_K, f16, f32)", all(t in types and not types[t]["native"] for t in ("q6_K", "f16", "f32"))),
              ("B1 elements = sum N*T", rep["mm"]["b1_elements"] == sum(N for (_, _, N) in specs.values()) * 6),
              ("settings reported (m6, FA on, q8_0 / q8_0 KV on the host, rotation off)",
               rep.get("settings") == {"phase": "m6", "flash_attn": "on", "type_k": "q8_0", "type_v": "q8_0",
                                       "offload_kqv": False, "attn_rot_disable": True}),
              ("the dev run's versions reported (SW §6.9)", rep.get("versions") == SYNTH_VERSIONS and
               "Tesseract Basis 0.1.0 (ABI 1" in (out / "report-clean" / "report.md").read_text())]
    # 2b. runs with different settings (here: the floor made with an f16 V cache) must be refused
    fm = dirs["floor"] / "meta.json"
    fmeta = json.loads(fm.read_text())
    fm_orig = fm.read_text()
    fmeta["info"]["type_v"] = "f16"
    fm.write_text(json.dumps(fmeta))
    try:
        cmd_compare(argparse.Namespace(**base, out=str(out / "report-settings")))
        refused = False
    except SystemExit as ex:
        refused = ex.code == 2
    fm.write_text(fm_orig)
    checks.append(("runs with different settings refused", refused))
    # 2c. version 1 dumps (FA-off settings) are still read: as a set of three they compare
    for d in dirs.values():
        mp = d / "meta.json"
        m1 = json.loads(mp.read_text())
        m1["version"] = 1
        m1.pop("info", None)
        (d / "meta.v2.json").write_text(mp.read_text())
        mp.write_text(json.dumps(m1))
    rc1 = cmd_compare(argparse.Namespace(**base, out=str(out / "report-v1")))
    r1v = json.loads((out / "report-v1" / "report.json").read_text())
    for d in dirs.values():
        (d / "meta.v2.json").replace(d / "meta.json")
    checks.append(("version 1 dumps read with the FA-off settings", rc1 == 0 and r1v.get("settings", {}).get("phase") == "v1"))
    # 3. fault injection: 1-ulp change in one dev output of the Q4_K tensor -> B1 must report exactly that element
    meta = json.loads((dirs["dev"] / "meta.json").read_text())
    node = next(n for n in meta["nodes"] if n["weight"] == "blk.0.ffn_up.weight")
    fpath = dirs["dev"] / node["file"]
    orig = np.fromfile(fpath, dtype="<f4")
    bad = orig.copy()
    N = node["ne"][0]
    t_bad, n_bad = 3, 17
    bad.view(np.uint32)[t_bad * N + n_bad] += 1
    bad.tofile(fpath)
    rc1 = cmd_compare(argparse.Namespace(**base, out=str(out / "report-1ulp")))
    r1 = json.loads((out / "report-1ulp" / "report.json").read_text())
    t1 = next(t for t in r1["mm"]["tensors"] if t["weight"] == "blk.0.ffn_up.weight")
    checks.append(("1-ulp fault: FAIL, exactly 1 B1 mismatch at the right place",
                   rc1 == 1 and r1["mm"]["b1_mismatches"] == 1 and t1["b1"]["first"][0]["row"] == n_bad
                   and t1["b1"]["first"][0]["token"] == t_bad and t1["b1"]["first"][0]["ulp"] == 1))
    # 4. a gross fault (x 1.01) must also break the B2 bound
    bad = orig.copy()
    bad[t_bad * N + n_bad] *= np.float32(1.01)
    bad.tofile(fpath)
    cmd_compare(argparse.Namespace(**base, out=str(out / "report-gross")))
    r2 = json.loads((out / "report-gross" / "report.json").read_text())
    checks.append(("1% fault: B1 and B2 both flag it", r2["mm"]["b1_mismatches"] == 1 and r2["mm"]["b2_violations"] == 1))
    orig.tofile(fpath)
    # 5. --src1-from-producer: drop the src1 dumps of three weights in cpu and dev; x must come from the producer
    #    nodes (mapping verified on the other three) and give the clean run's B results
    drop = {"blk.0.ffn_up.weight", "blk.1.attn_k.weight", "output.weight"}
    pdirs = {}
    for run in ("cpu", "dev"):
        d = out / "runs-dropped" / run
        d.mkdir(parents=True, exist_ok=True)
        for f in ("t", "logits.f32"):
            if not (d / f).exists():
                (d / f).symlink_to((dirs[run] / f).resolve())
        m = json.loads((dirs[run] / "meta.json").read_text())
        for n in m["nodes"]:
            if n["weight"] in drop:
                n["src1_file"] = n["src1_ne"] = None
        (d / "meta.json").write_text(json.dumps(m, indent=1))
        pdirs[run] = d
    pbase = dict(base, cpu=str(pdirs["cpu"]), dev=str(pdirs["dev"]))
    cmd_compare(argparse.Namespace(**pbase, out=str(out / "report-dropped")))
    rn = json.loads((out / "report-dropped" / "report.json").read_text())["mm"]
    cmd_compare(argparse.Namespace(**pbase, src1_from_producer=True, out=str(out / "report-producer")))
    rp_full = json.loads((out / "report-producer" / "report.json").read_text())
    rp = rp_full["mm"]
    same = all(a.get("b3") == b.get("b3") and a.get("b1") == b.get("b1")
               for a, b in zip(rep["mm"]["tensors"], rp["tensors"]))
    checks.append(("dropped src1 without --src1-from-producer: 3 weight MUL_MATs not sampled", rn["not_sampled"] == 3))
    checks.append(("--src1-from-producer: mapping verified (3 x 2 dumps), 3 tensors from producers, PASS, B1/B3 as the clean run",
                   rp["producer_validation"]["ok"] and rp["producer_validation"]["checked"] == 6 and rp["x_from_producer"] == 3
                   and rp["checked"] == 6 and rp_full["verdict"] == "PASS" and same))
    # 6. a producer that differs from its src1 dump disables the fallback (WARN, tensors stay unsampled)
    meta_d = json.loads((pdirs["dev"] / "meta.json").read_text())
    pnode = next(n for n in meta_d["nodes"] if n["name"] == "ffn_swiglu-0")
    pfile = dirs["dev"] / pnode["file"]
    porig = np.fromfile(pfile, dtype="<f4")
    pbad = porig.copy()
    pbad.view(np.uint32)[0] ^= 1
    pbad.tofile(pfile)
    cmd_compare(argparse.Namespace(**pbase, src1_from_producer=True, out=str(out / "report-producer-bad")))
    pb = json.loads((out / "report-producer-bad" / "report.json").read_text())
    porig.tofile(pfile)
    checks.append(("producer != src1 dump: fallback not used, 3 not sampled, WARN",
                   not pb["mm"]["producer_validation"]["ok"] and pb["mm"]["not_sampled"] == 3
                   and any("src1-from-producer not used" in x for x in pb["warnings"])))

    def run_compare(ns: dict, name: str):
        """(exit code, report.json or None); die() inside compare is exit code 2."""
        o = out / name
        try:
            rc = cmd_compare(argparse.Namespace(**ns, out=str(o)))
        except SystemExit as ex:
            rc = ex.code
        p = o / "report.json"
        return rc, (json.loads(p.read_text()) if p.exists() else None)

    def variant(run: str, name: str, edit) -> str:
        """Copy of a synthetic run dir with an edited meta.json (t/ and logits shared by symlink)."""
        d = out / "variants" / name / run
        d.mkdir(parents=True, exist_ok=True)
        for x in ("t", "logits.f32"):
            if not (d / x).exists():
                (d / x).symlink_to((dirs[run] / x).resolve())
        m = json.loads((dirs[run] / "meta.json").read_text())
        edit(m)
        (d / "meta.json").write_text(json.dumps(m, indent=1))
        return str(d)

    # 7. sequential-sum mutant as the device output of the q8_0 tensor (K = 256, 8 terms): B1 and the numpy
    #    spec model (B1b) catch it, the B2 bound cannot (it does not see the accumulation order)
    wn = "blk.0.attn_q.weight"
    gm = GGUFModel(str(model))
    _, _, wt, wne, _ = gm.info(wn)
    Kq, Nq = wne
    _, qblocks = nat.dev_blocks(wt, gm.read_rows(nat, wn, 0, Nq), Nq, Kq)
    gm.close()
    rdev = Run(str(dirs["dev"]), "dev")
    nq = next(n for n in rdev.nodes if n["weight"] == wn)
    xq = rdev.load_src1(nq)
    ymut = spec_model_y(FMT_Q8_0, parse_q8_0(qblocks, Nq, Kq), np_qact(QA_A32, xq), xq.shape[0], order="sequential")
    yfile = dirs["dev"] / nq["file"]
    yorig = np.fromfile(yfile, dtype="<f4")
    n_diff = int(np.count_nonzero(np.ascontiguousarray(ymut, dtype="<f4").view(np.uint32).ravel() != yorig.view(np.uint32)))
    np.ascontiguousarray(ymut, dtype="<f4").tofile(yfile)
    rc7, r7 = run_compare(base, "report-seqmutant")
    yorig.tofile(yfile)
    t7 = next(t for t in r7["mm"]["tensors"] if t["weight"] == wn)
    checks.append((f"sequential-sum mutant ({n_diff} of {yorig.size} outputs differ): FAIL by B1 and B1b, B2 0 violations",
                   n_diff > 0 and rc7 == 1 and t7["b1"]["mismatches"] == n_diff and t7["b1b"]["mismatches"] == n_diff
                   and t7["b2"]["violations"] == 0))

    # 8. B3 with a dominant token: token 0 holds ~all of ||W x||^2 with ratio 2, the others 0.5
    T8 = 6
    e2c8 = np.full(T8, 1e-4)
    r2c8 = np.ones(T8)
    r2c8[0] = 1e6
    e2c8[0] = 1e-4 * 1e6
    e2d8 = e2c8 * 0.25
    e2d8[0] = e2c8[0] * 4.0
    b8 = b3_summary(e2d8, r2c8.copy(), e2c8, r2c8)
    checks.append(("B3: a dominant token is left out of the ratio (all tokens 2.0, robust 0.5, per-token median 0.5)",
                   [d["token"] for d in b8["dominant_tokens"]] == [0] and abs(b8["ratio_all_tokens"] - 2.0) < 1e-3
                   and abs(b8["ratio"] - 0.5) < 1e-9 and abs(b8["ratio_token_median"] - 0.5) < 1e-9))

    # 9. coverage: weight MUL_MATs that ran on the CPU in the dev run -> INCOMPLETE (exit 3), not PASS
    def to_cpu(m):
        k = 0
        for n in m["nodes"]:
            if n["weight"] and k < 2:
                n["backend"] = "CPU"
                k += 1
    rc9, r9 = run_compare(dict(base, dev=variant("dev", "offdev", to_cpu)), "report-offdev")
    checks.append(("2 weight MUL_MATs off the device: INCOMPLETE (exit 3), 2 coverage gaps",
                   rc9 == 3 and r9["verdict"] == "INCOMPLETE" and len(r9["coverage_gaps"]) == 2))

    # 10. run sets from different models, -m that is not the dev run's model: refused (exit 2) before any work
    def other_model(m):
        m["model"] = str(out / "elsewhere" / "other.gguf")
    rc10, _ = run_compare(dict(base, cpu=variant("cpu", "othermodel", other_model)), "report-mixed")
    mcopy = out / "synth-copy.gguf"
    mcopy.write_bytes(model.read_bytes())
    rc10b, _ = run_compare(dict(base, model=str(mcopy)), "report-wrongm")
    checks.append(("cpu run from another model, -m not the dev run's model: both refused (exit 2)", rc10 == 2 and rc10b == 2))

    # 11. --dev-noobs: the observed dev dir itself is refused; a proper --no-observe run is compared
    def noobs(m):
        m["info"] = dict(m.get("info") or {}, observe=False)
        m["nodes"] = []
    rc11a, _ = run_compare(dict(base, dev_noobs=str(dirs["dev"])), "report-noobs-self")
    rc11b, r11b = run_compare(dict(base, dev_noobs=variant("dev", "noobs", noobs)), "report-noobs")
    checks.append(("--dev-noobs = dev refused (exit 2); a --no-observe dev run: PASS, logits bit-identical, no WARN",
                   rc11a == 2 and rc11b == 0 and r11b["logits"]["observe_check"]["bit_identical"]
                   and not any("dev-noobs" in x for x in r11b["warnings"])))

    # 12. --mm-layers: the dumper's synonyms work; a bad list exits 2 and leaves no stale report; a crash
    #     exits 2 (not the FAIL code 1) and leaves no report either
    rc12, r12 = run_compare(dict(base, mm_layers="out"), "report-layers")
    rc12b, r12b = run_compare(dict(base, mm_layers="bogus"), "report-layers")
    def lost_file(m):
        m["nodes"][0]["file"] = "t/missing.f32"
    crash = out / "report-crash"
    crash.mkdir(exist_ok=True)
    (crash / "report.md").write_text("# numcheck report: PASS\n")
    rc12c = main(["--build", str(args.build), "compare", "--cpu", base["cpu"], "--floor", variant("floor", "crash", lost_file),
                  "--dev", base["dev"], "-m", base["model"], "--boundtest-report", base["boundtest_report"], "--quiet",
                  "-o", str(crash)])
    checks.append(("--mm-layers out = output head only (PASS); 'bogus' exits 2 without a report; a crash exits 2 "
                   "without a report", rc12 == 0 and r12["mm"]["checked"] == 1 and rc12b == 2 and r12b is None
                   and rc12c == 2 and not (crash / "report.md").exists()))
    # 13. the boundtest's negative control: the sequential mutant differs from libtsr_ref on many rows
    checks.append(("boundtest: numpy spec model bit-exact on every row, sequential mutant detected",
                   bt["spec_model_mismatch"] == 0 and json.loads(bt_path.read_text())["bound"]["seq_mutant_detected"]))
    for name, good in checks:
        log(f"selftest: {'ok  ' if good else 'FAIL'} {name}")
        ok &= bool(good)
    log(f"selftest: {'PASS' if ok else 'FAIL'} ({time.time() - t0:.0f} s)")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", default=str(DEFAULT_BUILD), help="build dir with libtsr_ref.so / libnumcheck_helper.so")
    sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("compare", help="compare cpu / floor / dev dumps")
    c.add_argument("--cpu", required=True)
    c.add_argument("--floor", required=True)
    c.add_argument("--dev", required=True)
    c.add_argument("--dev-noobs", default=None, help="dev run dir made with --no-observe: logits must be bit-identical")
    c.add_argument("-m", "--model", default=None, help="GGUF (first shard); default: dev meta.json model")
    c.add_argument("-o", "--out", required=True)
    c.add_argument("--mm-layers", default=None, help="B checks only for these layers: comma list of N, A-B, output (out, head), "
                                                   "or all (default)")
    c.add_argument("--src1-from-producer", action="store_true",
                   help="weight MUL_MATs without a src1 dump: use the producer node's output (attn_norm-N, kqv_out-N, "
                        "ffn_norm-N, ffn_swiglu-N, result_norm) as x, after verifying the mapping on every src1 dump")
    c.add_argument("--threads", type=int, default=4)
    c.add_argument("--chunk-elems", type=int, default=1 << 21, help="rows per chunk ~ this / (K/32 * T)")
    c.add_argument("--boundtest-report", default=None, help="reuse a passed boundtest JSON instead of running it")
    c.add_argument("--boundtest-rows", type=int, default=1_000_000)
    c.add_argument("--qact-random", type=int, default=20000)
    c.add_argument("--seed", type=int, default=1)
    c.add_argument("--top", type=int, default=15)
    c.add_argument("--quiet", action="store_true")
    b = sp.add_parser("boundtest", help="B2 bound + QACT model vs libtsr_ref")
    b.add_argument("--rows", type=int, default=1_000_000)
    b.add_argument("--seed", type=int, default=1)
    b.add_argument("--qact-random", type=int, default=20000)
    b.add_argument("--threads", type=int, default=4)
    b.add_argument("-o", "--out", default=None, help="write the JSON result here")
    s = sp.add_parser("selftest", help="boundtest + synthetic compare with injected faults")
    s.add_argument("-o", "--out", required=True)
    s.add_argument("--rows", type=int, default=1_000_000)
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--qact-random", type=int, default=20000)
    s.add_argument("--threads", type=int, default=4)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "compare":
            return cmd_compare(a)
        if a.cmd == "boundtest":
            return cmd_boundtest(a)
        return cmd_selftest(a)
    except Exception:                            # exit 2, never the FAIL code 1 (compare left no report behind)
        import traceback
        traceback.print_exc()
        print("numcheck.py: error: unexpected exception (exit 2)", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
