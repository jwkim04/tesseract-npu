#!/usr/bin/env python3
"""Independent numpy model of the VU ops on an SPM image (SW-23, SW §10.1, VU §10 item 1).

Written from the text of docs/design/02-vector-unit.md alone: §4.1-§4.5 (fp rules, the rows x m
operand model with line strides, stride 0, the tail rule, the n = 0 rule, c from an s register or
an SPM row vector, aliasing), §5.2-§5.8 (what each field means), §6.2 (RED: 128 partials, the
complete tree, the z() skip rule, the post-ops), §6.7 (ROPE: both angle forms, the rotation, the
raw tail copy) and §6.8 (total order, MAXT / MINT, TOPK). The element math of §6.3-§6.6 (EXPK and
the composites, SIN / COS) and the f32 primitives come from ref_math_np.py, the independent
numpy model of tsr_ref_math.c (it shares no code with the C either).

This model works on a whole SPM image (a flat uint32 array; line X is words 16X .. 16X + 15) and
takes the encodings' fields: line addresses, line strides, s-register values. It deliberately
does not follow the structure of sw/sim/tsr_ref_vu.c (pointer-based, element loops): operands
are gathered as index arrays, evaluated as whole numpy arrays and scattered back.

Usage:
  ref_vu_np.py --compare <libtsr_ref.so> [--seed N] [--cases N]
      Generates random and directed cases for VOP (every function, cmode, broadcast and stride
      shapes, in place), RED (every kind and post-op, TO_S, m = 8k + r rows with -0 subtrees, the
      skip model), ROPE (NORM / NEOX, recurrence with FF / FS and the A-table form, tails, in
      place), TOPK and SOP; runs each on the C reference through ctypes and on this model, and
      compares the whole SPM image (and s results) bit for bit. Exit 0 iff everything matches.
  ref_vu_np.py --self-test
      The model's own directed checks (written from the document's examples and rules), so a
      model regression is not hidden by a matching C regression.
"""
import argparse
import ctypes
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ref_math_np as RM  # noqa: E402  (the §6.3-§6.6 element model)

U32 = np.uint32
CANON = 0x7FC00000
NINF = 0xFF800000
PINF = 0x7F800000
SIGNB = 0x80000000

# Codes of the encodings (VU §5.2-§5.8).
VOP_NAMES = ["ADD", "SUB", "MUL", "MAX", "MULC", "ADDC", "SILU", "SWIGLU", "EXP", "COPY", "MIN",
             "SUBC", "MULCB", "MULCADD", "ADDMULC", "EXPSUB", "EXPSUBC", "SIGMOID", "GELU",
             "GEGLU", "TANH", "RECIP", "SETC", "GELU16", "GEGLU16"]
CMODE_NONE, CMODE_SREG, CMODE_VEC = 0, 1, 2
RED_SUM, RED_SUMSQ, RED_MAX = 0, 1, 2
POST_NONE, POST_RCP, POST_RMS = 0, 1, 2
ROPE_NORM, ROPE_NEOX = 0, 1
SOP_NAMES = ["MOVI", "ADD", "MUL", "ADDI", "MULI", "SQRT", "RECIP", "LDS", "STS", "SUB", "DIV",
             "RSQRT", "MAX", "MIN", "EXP"]


def u32(x):
    return np.asarray(x, dtype=U32)


def L(n):
    """L(m) = ceil(m/16) lines (VU §4.2)."""
    return (int(n) + 15) // 16


def E(n, s, ln):
    """Extent in lines of n rows of ln lines at stride s (VU §5.9)."""
    return 0 if n == 0 or ln == 0 else (n - 1) * s + ln


# ---------------------------------------------------------------------------
# Total order, MAXT, MINT (VU §6.8)
# ---------------------------------------------------------------------------
def order_key(x):
    """key = sign ? ~bits : bits | 2^31 of DAZ(x), as int64; NaN (canonical after DAZ) -> 2^32,
    above every value."""
    xd = RM.daz(x).astype(np.int64)
    neg = (xd & SIGNB) != 0
    k = np.where(neg, (~xd) & 0xFFFFFFFF, xd | SIGNB)
    return np.where(RM.isnan(x), np.int64(1 << 32), k)


def maxt(a, b):
    a, b = RM.daz(a), RM.daz(b)
    y = np.where(order_key(a) >= order_key(b), a, b)
    return np.where(RM.isnan(a) | RM.isnan(b), U32(CANON), y).astype(U32)


def mint(a, b):
    a, b = RM.daz(a), RM.daz(b)
    na, nb = RM.isnan(a), RM.isnan(b)
    y = np.where(order_key(a) <= order_key(b), a, b)     # NaN keys are the largest: NaN loses
    return np.where(na & nb, U32(CANON), y).astype(U32)


# ---------------------------------------------------------------------------
# VOP element functions (VU §5.2 table); each takes uint32 arrays a, b, c
# ---------------------------------------------------------------------------
def _expsub(a, x):
    # EXPSUB(a, x) = (DAZ(a) = -inf) ? +0 : EXPK(FS(a, x), P1)   (VU §6.5, VU-21)
    return np.where(RM.daz(a) == NINF, U32(0), RM.expk(RM.FS(a, x), "p1")).astype(U32)


VOP_FUNCS = {
    0x00: ("ab", lambda a, b, c: RM.FA(a, b)),
    0x01: ("ab", lambda a, b, c: RM.FS(a, b)),
    0x02: ("ab", lambda a, b, c: RM.FM(a, b)),
    0x03: ("ab", lambda a, b, c: maxt(a, b)),
    0x04: ("ac", lambda a, b, c: RM.FM(a, c)),
    0x05: ("ac", lambda a, b, c: RM.FA(a, c)),
    0x06: ("a", lambda a, b, c: RM.FM(a, RM.RC(RM.FA(RM.ONE, RM.expk(a, "n1"))))),   # SILU = FM(x, SIG(x))
    0x07: ("ab", lambda a, b, c: RM.FM(RM.silu(a), b)),
    0x08: ("a", lambda a, b, c: RM.expk(a, "p1")),
    0x09: ("a", lambda a, b, c: u32(a).copy()),                                       # raw
    0x0A: ("ab", lambda a, b, c: mint(a, b)),
    0x0B: ("ac", lambda a, b, c: RM.FS(a, c)),
    0x0C: ("abc", lambda a, b, c: RM.FM(RM.FM(a, c), b)),
    0x0D: ("abc", lambda a, b, c: RM.FA(RM.FM(a, c), b)),
    0x0E: ("abc", lambda a, b, c: RM.FM(RM.FA(a, b), c)),
    0x0F: ("ab", lambda a, b, c: _expsub(a, b)),
    0x10: ("ac", lambda a, b, c: _expsub(a, c)),
    0x11: ("a", lambda a, b, c: RM.RC(RM.FA(RM.ONE, RM.expk(a, "n1")))),             # SIG
    0x12: ("a", lambda a, b, c: RM.gelu_tanh(a)),
    0x13: ("ab", lambda a, b, c: RM.FM(RM.gelu_tanh(a), b)),
    0x14: ("a", lambda a, b, c: RM.tanh(a)),
    0x15: ("a", lambda a, b, c: RM.RC(a)),
    0x16: ("c", lambda a, b, c: u32(c).copy()),                                       # raw
    0x17: ("a", lambda a, b, c: RM.gelu16(a)),
    0x18: ("ab", lambda a, b, c: RM.geglu16(a, b)),
}


def vop_elem(func, a, b, c):
    with np.errstate(all="ignore"):
        return u32(VOP_FUNCS[func][1](u32(a), u32(b), u32(c)))


# ---------------------------------------------------------------------------
# Operand model (VU §4.2): word indices of rows x m elements
# ---------------------------------------------------------------------------
def row_index(X, s, rows, m):
    """Word index of element j of row r: 16·(X + r·s) + j, as a (rows, m) array."""
    r = np.arange(rows, dtype=np.int64)[:, None]
    j = np.arange(m, dtype=np.int64)[None, :]
    return 16 * (X + r * s) + j


def write_rows(spm, X, s, rows, n, vals):
    """Write rows of n words at line X, stride s, and zero words [n, 16·L(n)) of each row."""
    full = np.zeros((rows, 16 * L(n)), dtype=U32)
    full[:, :n] = vals
    spm[row_index(X, s, rows, 16 * L(n))] = full


def vop(spm, func, cmode, d, a, b, c, m, rows, d_stride, a_stride, b_stride, si_val=0):
    """VOP (VU §5.2) on the SPM image spm (modified in place). si_val is s[si] for cmode 1."""
    if rows == 0 or m == 0:                       # n = 0 rule: nothing read or written
        return
    reads = VOP_FUNCS[func][0]
    A = spm[row_index(a, a_stride, rows, m)] if "a" in reads else np.zeros((rows, m), U32)
    B = spm[row_index(b, b_stride, rows, m)] if "b" in reads else np.zeros((rows, m), U32)
    if cmode == CMODE_SREG:
        C = np.full((rows, m), si_val, dtype=U32)
    elif cmode == CMODE_VEC:
        C = np.repeat(spm[16 * c + np.arange(rows)][:, None], m, axis=1)
    else:
        C = np.zeros((rows, m), U32)
    Y = vop_elem(func, A, B, C)                   # all sources are read before any write
    write_rows(spm, d, d_stride, rows, m, Y)


# ---------------------------------------------------------------------------
# RED (VU §6.2)
# ---------------------------------------------------------------------------
def red_partials(kind, X):
    """(rows, 128) partials: p_(i mod 128) <- FA(p_(i mod 128), x_i), SUMSQ with FM(x_i, x_i)."""
    rows, m = X.shape
    with np.errstate(all="ignore"):
        V = RM.FM(X, X) if kind == RED_SUMSQ else X
        p = np.zeros((rows, 128), dtype=U32)
        for k0 in range(0, m, 128):
            n = min(128, m - k0)
            p[:, :n] = RM.FA(p[:, :n], V[:, k0:k0 + n])   # only the partials this block updates
    return p


def tree_full(p):
    """Complete binary tree over the 128 partials, adjacent pairs left to right, 7 levels."""
    v = u32(p)
    with np.errstate(all="ignore"):
        while v.shape[-1] > 1:
            v = RM.FA(v[..., 0::2], v[..., 1::2])
    return v[..., 0]


def z(q):
    return np.where(u32(q) == SIGNB, U32(0), u32(q))


def tree_skip(p, present):
    """The §6.2 skip rule: subtrees without a present partial are skipped; a pairing with one
    present side passes z() of it; an all-skipped tree is +0. present: bool (..., 128)."""
    v, pr = u32(p), np.asarray(present, bool)
    with np.errstate(all="ignore"):
        while v.shape[-1] > 1:
            l, r = v[..., 0::2], v[..., 1::2]
            pl, prr = pr[..., 0::2], pr[..., 1::2]
            v = np.where(pl & prr, RM.FA(l, r), np.where(pl, z(l), z(r)))
            pr = pl | prr
    return np.where(pr[..., 0], v[..., 0], U32(0)).astype(U32)


def red_post(post, y, sk, se):
    with np.errstate(all="ignore"):
        if post == POST_RCP:
            return RM.RC(y)
        if post == POST_RMS:
            return RM.RC(RM.SQ(RM.FA(RM.FM(y, U32(sk)), U32(se))))
    return u32(y)


def red_rows(kind, post, X, sk, se):
    """Row results of a (rows, m) array after the post-op."""
    rows, m = X.shape
    if kind == RED_MAX:
        if m == 0:
            y = np.full(rows, NINF, dtype=U32)
        else:
            Xd = RM.daz(X)
            idx = np.argmax(order_key(Xd), axis=1)
            y = Xd[np.arange(rows), idx]
            y = np.where(RM.isnan(X).any(axis=1), U32(CANON), y).astype(U32)
    else:
        y = tree_full(red_partials(kind, X)) if m else np.zeros(rows, dtype=U32)
    return red_post(post, y, sk, se)


def red(spm, kind, post, to_s, d, a, m, rows, a_stride, sk=0, se=0):
    """RED (VU §5.3). TO_S (rows = 1): returns the value for s[sd] (also for m = 0) and writes no
    SPM. Else writes the per-row vector at line d (rows words, tail rule) unless m or rows is 0."""
    if to_s:
        X = spm[row_index(a, a_stride, 1, m)] if m else np.zeros((1, 0), U32)
        return int(red_rows(kind, post, X, sk, se)[0])
    if rows == 0 or m == 0:
        return None
    y = red_rows(kind, post, spm[row_index(a, a_stride, rows, m)], sk, se)
    write_rows(spm, d, 0, 1, rows, y[None, :])
    return None


# ---------------------------------------------------------------------------
# ROPE (VU §5.4, §6.7)
# ---------------------------------------------------------------------------
def i32_to_f32(p):
    """tsr_i32_to_f32: RNE (int32 -> float64 is exact, then one rounding to binary32)."""
    return np.asarray(p, dtype=np.int64).astype(np.float64).astype(np.float32).view(U32)


def rope_angles(pos, n_dims, ff, fs, atab, ts, fscale, table):
    """phi[t, i], i < n_dims/2, for int32 positions pos[t] (VU §6.7). table: uint32 words."""
    half = n_dims // 2
    Pf = i32_to_f32(pos)
    with np.errstate(all="ignore"):
        if atab:
            return RM.FM(Pf[:, None], u32(table[:half])[None, :])
        phi = np.empty((len(Pf), half), dtype=U32)
        th = Pf
        for i in range(half):                    # theta_(i+1) = FM(theta_i, TS)
            phi[:, i] = th
            th = RM.FM(th, U32(ts))
        if ff:
            phi = RM.FD(phi, u32(table[:half])[None, :])
        if fs:
            phi = RM.FM(U32(fscale), phi)
    return u32(phi)


def rope(spm, mode, ff, fs, atab, Mv, ts, fscale, d, src, pos, table, head_dim, n_dims,
         n_heads, T, src_stride, d_stride):
    """ROPE (VU §5.4) on the SPM image. Mv, ts, fscale: the values of s[sm], s[sts], s[sfs]."""
    if T == 0 or n_heads == 0:
        return
    H = n_heads * head_dim // 16
    ss = src_stride or H
    ds = d_stride or H
    half = n_dims // 2
    P = spm[16 * pos + np.arange(T)].view(np.int32)
    tab = spm[16 * table + np.arange(half)] if (ff or atab) else None
    phi = rope_angles(P, n_dims, ff, fs, atab, ts, fscale, tab)
    with np.errstate(all="ignore"):
        S, C = RM.sincos(phi)
        c = RM.FM(U32(Mv), C)[:, None, :]          # c_i = FM(M, COS(phi_i)), (T, 1, half)
        s = RM.FM(U32(Mv), S)[:, None, :]
    t = np.arange(T, dtype=np.int64)[:, None, None]
    h = np.arange(n_heads, dtype=np.int64)[None, :, None]
    j = np.arange(head_dim, dtype=np.int64)[None, None, :]
    X = spm[16 * (src + t * ss) + h * head_dim + j]          # (T, n_heads, head_dim)
    Y = X.copy()                                               # [n_dims, head_dim): raw copy
    if mode == ROPE_NEOX:
        i0 = np.arange(half)
        i1 = i0 + half
    else:
        i0 = np.arange(0, n_dims, 2)
        i1 = i0 + 1
    x0, x1 = X[..., i0], X[..., i1]
    with np.errstate(all="ignore"):
        Y[..., i0] = RM.FS(RM.FM(x0, c), RM.FM(x1, s))
        Y[..., i1] = RM.FA(RM.FM(x0, s), RM.FM(x1, c))
    spm[16 * (d + t * ds) + h * head_dim + j] = Y


# ---------------------------------------------------------------------------
# TOPK (VU §6.8)
# ---------------------------------------------------------------------------
def topk(spm, k, d, a, m, rows, d_stride, a_stride):
    if rows == 0 or m == 0:
        return
    ds = d_stride or (k + 7) // 8
    out = np.empty((rows, 2 * k), dtype=U32)
    for r in range(rows):
        x = spm[16 * (a + r * a_stride) + np.arange(m)]
        xh = np.where(RM.isnan(x), U32(CANON), RM.daz(x)).astype(U32)
        order = np.lexsort((np.arange(m), -order_key(xh)))   # value descending, then j ascending
        sel = order[:k]
        pairs = np.empty((k, 2), dtype=U32)
        pairs[:, 0] = 0xFFFFFFFF
        pairs[:, 1] = NINF
        pairs[:len(sel), 0] = sel
        pairs[:len(sel), 1] = xh[sel]
        out[r] = pairs.reshape(-1)
    write_rows(spm, d, ds, rows, 2 * k, out)


# ---------------------------------------------------------------------------
# SOP (VU §5.8), vectorised over operand arrays
# ---------------------------------------------------------------------------
def sop(sub, sa, sb, imm, mem):
    sa, sb, imm, mem = u32(sa), u32(sb), u32(imm), u32(mem)
    with np.errstate(all="ignore"):
        f = {
            0: lambda: imm.copy(),
            1: lambda: RM.FA(sa, sb),
            2: lambda: RM.FM(sa, sb),
            3: lambda: RM.FA(sa, imm),
            4: lambda: RM.FM(sa, imm),
            5: lambda: RM.SQ(sa),
            6: lambda: RM.RC(sa),
            7: lambda: mem.copy(),
            8: lambda: sa.copy(),
            9: lambda: RM.FS(sa, sb),
            10: lambda: RM.FD(sa, sb),
            11: lambda: RM.RC(RM.SQ(sa)),
            12: lambda: maxt(sa, sb),
            13: lambda: mint(sa, sb),
            14: lambda: RM.expk(sa, "p1"),
        }[sub]
        return u32(f())


# ---------------------------------------------------------------------------
# Value generation
# ---------------------------------------------------------------------------
SPECIALS = np.array([
    0x00000000, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000, 0xFFC00000, 0x7F800001,
    0xFFFFFFFF, 0x7FBFFFFF, 0x00000001, 0x807FFFFF, 0x00400000, 0x80400000, 0x00800000,
    0x80800000, 0x80C00000, 0x00C00000, 0x7F7FFFFF, 0xFF7FFFFF, 0x41200000, 0xC1200000,
    0x411FFFFF, 0xC11FFFFF, 0x41200001, 0xC1200001, 0x3F800000, 0xBF800000, 0x42B17218,
    0xC2B17218, 0x42CE0000, 0xC2CE0000, 0x3C800000, 0x3C7FFFFF, 0xBC800000, 0x21800000,
    0x4E800000, 0x387FC000, 0x33800000, 0x47800000, 0xC7800000, 0x477FE000,
], dtype=U32)


def gen_values(rng, n, special_frac=0.15, rand_frac=0.05):
    scale = rng.choice([1.0, 4.0, 30.0, 1e-3])
    v = (rng.standard_normal(n) * scale).astype(np.float32).view(U32).copy()
    pick = rng.random(n)
    sp = pick < special_frac
    v[sp] = rng.choice(SPECIALS, size=int(sp.sum()))
    rb = (pick >= special_frac) & (pick < special_frac + rand_frac)
    v[rb] = rng.integers(0, 1 << 32, size=int(rb.sum()), dtype=np.uint64).astype(U32)
    return v


def gen_finite(rng, n, scale=1.0):
    return (rng.standard_normal(n) * scale).astype(np.float32).view(U32).copy()


# ---------------------------------------------------------------------------
# The C reference through ctypes
# ---------------------------------------------------------------------------
class RopeCfg(ctypes.Structure):
    _fields_ = [(nm, ctypes.c_uint32) for nm in
                ("mode", "ff", "fs", "atab", "m", "ts", "freq_scale", "head_dim", "n_dims",
                 "n_heads", "T")]


class CRef:
    def __init__(self, path):
        lib = ctypes.CDLL(path)
        u, p = ctypes.c_uint32, ctypes.c_void_p
        lib.tsr_ref_vop.argtypes = [u, u, u, u, p, u, p, u, p, u, p]
        lib.tsr_ref_vop.restype = None
        lib.tsr_ref_red.argtypes = [u, u, u, u, p, p, u, u, u]
        lib.tsr_ref_red.restype = None
        lib.tsr_ref_red_row.argtypes = [u, u, p, u, u, u]
        lib.tsr_ref_red_row.restype = u
        lib.tsr_ref_red_tree_skip.argtypes = [p, u, u]
        lib.tsr_ref_red_tree_skip.restype = u
        lib.tsr_ref_red_tree128.argtypes = [p]
        lib.tsr_ref_red_tree128.restype = u
        lib.tsr_ref_rope.argtypes = [ctypes.POINTER(RopeCfg), p, u, p, u, p, p]
        lib.tsr_ref_rope.restype = None
        lib.tsr_ref_rope_angles.argtypes = [ctypes.POINTER(RopeCfg), ctypes.c_int32, p, p]
        lib.tsr_ref_rope_angles.restype = None
        lib.tsr_ref_topk.argtypes = [u, u, u, p, u, p, u]
        lib.tsr_ref_topk.restype = None
        lib.tsr_ref_sop.argtypes = [u, u, u, u, u]
        lib.tsr_ref_sop.restype = u
        lib.tsr_ref_vop_elem.argtypes = [u, u, u, u]
        lib.tsr_ref_vop_elem.restype = u
        self.lib = lib

    @staticmethod
    def line(spm, X):
        return spm.ctypes.data + 64 * int(X)


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
class Cmp:
    def __init__(self):
        self.fail = 0
        self.checks = 0

    def same(self, label, got, want):
        self.checks += 1
        got, want = u32(got), u32(want)
        if got.shape != want.shape or not np.array_equal(got, want):
            self.fail += 1
            if self.fail <= 30:
                bad = np.flatnonzero(got.reshape(-1) != want.reshape(-1))
                i = int(bad[0]) if len(bad) else -1
                g = int(got.reshape(-1)[i]) if i >= 0 else -1
                w = int(want.reshape(-1)[i]) if i >= 0 else -1
                print(f"FAIL {label}: {len(bad)} words differ; first word {i}: C 0x{g:08x}, "
                      f"model 0x{w:08x}", file=sys.stderr)


class Placer:
    """Non-overlapping line extents in an SPM image of n lines, with random gaps."""

    def __init__(self, rng, n_lines):
        self.rng, self.n, self.next = rng, n_lines, 1

    def take(self, lines):
        start = self.next + int(self.rng.integers(0, 3))
        self.next = start + max(int(lines), 1)
        if self.next > self.n:
            raise RuntimeError("SPM image too small")
        return start


def fresh_spm(rng, n_lines):
    return gen_values(rng, 16 * n_lines)


def cmp_vop(c, ck, rng, n_cases):
    lib = c.lib
    for case in range(n_cases):
        func = case % 25
        reads = VOP_FUNCS[func][0]
        rows = int(rng.choice([0, 1, 1, 2, 3, 5, 17]))
        m = int(rng.choice([0, 1, 5, 15, 16, 17, 31, 32, 33, 64, 100, 257]))
        Lm = L(m)
        d_stride = Lm + int(rng.integers(0, 3)) if rows > 1 else int(rng.integers(0, 5))
        a_stride = int(rng.choice([0, Lm, Lm + 1, Lm + 2])) if "a" in reads else 0
        b_stride = int(rng.choice([0, Lm, Lm + 3])) if "b" in reads else 0
        cmode = CMODE_NONE
        if "c" in reads:
            cmode = int(rng.choice([CMODE_SREG, CMODE_VEC]))
        spm = fresh_spm(rng, 2048)
        pl = Placer(rng, 2048)
        d = pl.take(E(rows, d_stride, Lm))
        inplace = "a" in reads and rows > 0 and rng.random() < 0.25
        a = (d if inplace else pl.take(E(rows, a_stride, Lm))) if "a" in reads else 0
        if inplace:
            a_stride = d_stride
        b = pl.take(E(rows, b_stride, Lm)) if "b" in reads else 0
        if "b" in reads and "a" in reads and not inplace and rng.random() < 0.2:
            b, b_stride = a, a_stride                       # sources may overlap each other
        elif "b" in reads and not inplace and rows > 0 and rng.random() < 0.15:
            b, b_stride = d, d_stride                       # b in place (VU §4.5)
            inplace = "b"
        cl = pl.take(L(rows)) if cmode == CMODE_VEC else 0
        si = int(rng.choice(SPECIALS)) if rng.random() < 0.3 else int(gen_values(rng, 1)[0])
        want = spm.copy()
        vop(want, func, cmode, d, a, b, cl, m, rows, d_stride, a_stride, b_stride, si)
        got = spm.copy()
        sreg = np.array([si], dtype=U32)
        cptr = (sreg.ctypes.data if cmode == CMODE_SREG else
                CRef.line(got, cl) if cmode == CMODE_VEC else None)
        lib.tsr_ref_vop(func, cmode, rows, m, CRef.line(got, d), d_stride,
                        CRef.line(got, a) if "a" in reads else None, a_stride,
                        CRef.line(got, b) if "b" in reads else None, b_stride, cptr)
        ck.same(f"vop {VOP_NAMES[func]} cmode {cmode} rows {rows} m {m} strides "
                f"{d_stride}/{a_stride}/{b_stride} inplace {inplace}", got, want)
    # Element functions on the special values crossed with each other (a x b, c = b).
    A = np.repeat(SPECIALS, len(SPECIALS))
    B = np.tile(SPECIALS, len(SPECIALS))
    for func in range(25):
        want = vop_elem(func, A, B, B)
        got = np.array([lib.tsr_ref_vop_elem(func, int(x), int(y), int(y)) for x, y in zip(A, B)],
                       dtype=U32)
        ck.same(f"vop_elem {VOP_NAMES[func]} specials", got, want)


def neg_zero_row(rng, m):
    """A row whose level-1 pairs (0x80C00000, 0x00800000) sum to -0 (VU §6.2), placed at random
    even partial indices, the rest +0 or small values that keep other subtrees at -0 / +0."""
    x = np.zeros(m, dtype=U32)
    for i in range(0, min(m, 128) - 1, 2):
        if rng.random() < 0.6:
            x[i], x[i + 1] = 0x80C00000, 0x00800000
    if m > 128 and rng.random() < 0.5:
        x[128:] = 0
    return x


def cmp_red(c, ck, rng, n_cases):
    lib = c.lib
    sk_vals = [0x3B000000, 0x3D800000, 0x3F800000, 0x358637BD, 0x00000000, 0x7F800000]
    se_vals = [0x358637BD, 0x3727C5AC, 0x00000000, 0x80000000]
    for case in range(n_cases):
        kind = case % 3
        post = POST_NONE if kind == RED_MAX else int(rng.integers(0, 3))
        rows = int(rng.choice([0, 1, 2, 3, 16, 17, 33]))
        k8 = int(rng.integers(0, 17))
        m = int(rng.choice([0, 1, 2, 7, 8 * k8 + int(rng.integers(0, 8)), 128, 129, 255, 256,
                            300, 1000]))
        a_stride = int(rng.choice([0, L(m), L(m) + 1]))
        sk, se = int(rng.choice(sk_vals)), int(rng.choice(se_vals))
        spm = fresh_spm(rng, 4096)
        pl = Placer(rng, 4096)
        d = pl.take(L(rows))
        a = pl.take(E(rows, a_stride, L(m)))
        style = case % 5
        for r in range(rows if a_stride else min(rows, 1)):
            base = 16 * (a + r * a_stride)
            if style == 0:
                spm[base:base + m] = neg_zero_row(rng, m)
            elif style == 1:
                spm[base:base + m] = gen_finite(rng, m, float(rng.choice([1.0, 1e-19, 1e3])))
            elif style == 2:
                # -0 and negative subnormals only: every partial is FA(+0, x) = +0, never a raw -0
                spm[base:base + m] = rng.choice(u32([0x80000000, 0x80000001, 0x807FFFFF]), size=m)
        want = spm.copy()
        red(want, kind, post, False, d, a, m, rows, a_stride, sk, se)
        got = spm.copy()
        lib.tsr_ref_red(kind, post, rows, m, CRef.line(got, d), CRef.line(got, a), a_stride, sk, se)
        ck.same(f"red kind {kind} post {post} rows {rows} m {m} a_stride {a_stride} style {style}",
                got, want)
        # TO_S: rows = 1 (also m = 0)
        s_want = red(spm.copy(), kind, post, True, 0, a, m, 1, a_stride, sk, se)
        s_got = lib.tsr_ref_red_row(kind, post, CRef.line(spm, a), m, sk, se)
        ck.same(f"red TO_S kind {kind} post {post} m {m}", [s_got], [s_want])
    # The skip model (VU §6.2): for m < 128 the skip trees equal the complete tree, in the model
    # and in the C, on rows built to produce -0 subtrees at m = 8k + r.
    for case in range(400):
        m = int(rng.integers(0, 129)) if case % 2 else 8 * int(rng.integers(0, 16)) + int(rng.integers(0, 8))
        x = neg_zero_row(rng, m) if case % 3 else gen_values(rng, m)
        if case % 5 == 0 and m:
            x[:] = 0x80000000                      # partials FA(+0, -0) = +0
        p = red_partials(RED_SUM, x[None, :])[0] if m else np.zeros(128, U32)
        full = int(tree_full(p))
        for g in (1, 8):
            npres = min(128, g * ((min(m, 128) + g - 1) // g))
            present = np.arange(128) < npres
            ck.same(f"model skip tree g{g} m {m}", [int(tree_skip(p, present))], [full])
            pc = np.ascontiguousarray(p)
            ck.same(f"C skip tree g{g} m {m}",
                    [lib.tsr_ref_red_tree_skip(pc.ctypes.data, min(m, 128), g)], [full])
        ck.same(f"C tree128 m {m}", [lib.tsr_ref_red_tree128(np.ascontiguousarray(p).ctypes.data)],
                [full])


def f32b(x):
    return int(np.array([x], dtype=np.float32).view(U32)[0])


def cmp_rope(c, ck, rng, n_cases):
    lib = c.lib
    shapes = [(16, 16, ROPE_NORM), (32, 16, ROPE_NORM), (64, 32, ROPE_NEOX), (64, 64, ROPE_NEOX),
              (128, 128, ROPE_NEOX), (128, 64, ROPE_NORM), (256, 256, ROPE_NEOX),
              (512, 512, ROPE_NEOX), (48, 32, ROPE_NEOX), (96, 48, ROPE_NORM)]
    for case in range(n_cases):
        head_dim, n_dims, mode = shapes[case % len(shapes)]
        form = case % 4                      # 0 plain, 1 FF, 2 FS, 3 ATAB
        ff, fs, atab = int(form == 1 or (form == 2 and rng.random() < 0.5)), int(form == 2), int(form == 3)
        n_heads = int(rng.choice([0, 1, 2, 3])) if case % 7 else 1
        T = int(rng.choice([0, 1, 2, 4]))
        H = n_heads * head_dim // 16
        src_stride = int(rng.choice([0, H, H + 1])) if T > 1 else int(rng.integers(0, 3))
        d_stride = int(rng.choice([0, H, H + 2])) if T > 1 else int(rng.integers(0, 3))
        base = float(rng.choice([1e4, 1e6, 5e5, 10.0]))
        ts = f32b(np.float32(base) ** np.float32(-2.0 / n_dims))
        Mv = f32b(rng.choice([1.0, 0.75, 1.3]))
        fscale = f32b(rng.choice([1.0, 0.25, 0.5]))
        spm = fresh_spm(rng, 8192)
        pl = Placer(rng, 8192)
        dl = pl.take(E(T if n_heads else 0, d_stride or H, H))
        inplace = rng.random() < 0.25 and T > 0 and n_heads > 0
        if inplace:
            src_stride = d_stride
            sl = dl
        else:
            sl = pl.take(E(T if n_heads else 0, src_stride or H, H))
        posl = pl.take(L(T))
        tabl = pl.take(L(n_dims // 2))
        pos = rng.integers(0, 1 << 17, size=T)
        if case % 5 == 0:
            pos = rng.integers(-(1 << 31), 1 << 31, size=T)
        spm[16 * posl + np.arange(T)] = pos.astype(np.int64).astype(np.int32).view(U32)
        if ff:
            tab = np.where(rng.random(n_dims // 2) < 0.6, np.float32(1.0), np.float32(1e30))
            if rng.random() < 0.3:
                tab = rng.uniform(0.5, 40.0, n_dims // 2).astype(np.float32)
            spm[16 * tabl + np.arange(n_dims // 2)] = tab.astype(np.float32).view(U32)
        elif atab:
            A = (base ** (-2.0 * np.arange(n_dims // 2) / n_dims) * rng.uniform(0.3, 1.0)).astype(np.float32)
            spm[16 * tabl + np.arange(n_dims // 2)] = A.view(U32)
        if case % 3 == 0:                         # specials in the source rows
            pass
        else:
            nt = E(T if n_heads else 0, src_stride or H, H)
            spm[16 * sl:16 * (sl + nt)] = gen_finite(rng, 16 * nt, float(rng.choice([1.0, 100.0])))
        want = spm.copy()
        rope(want, mode, ff, fs, atab, Mv, ts, fscale, dl, sl, posl, tabl, head_dim, n_dims,
             n_heads, T, src_stride, d_stride)
        got = spm.copy()
        cfg = RopeCfg(mode, ff, fs, atab, Mv, ts, fscale, head_dim, n_dims, n_heads, T)
        lib.tsr_ref_rope(ctypes.byref(cfg), CRef.line(got, dl), d_stride, CRef.line(got, sl),
                         src_stride, CRef.line(got, posl),
                         CRef.line(got, tabl) if (ff or atab) else None)
        ck.same(f"rope mode {mode} hd {head_dim} nd {n_dims} ff {ff} fs {fs} atab {atab} "
                f"heads {n_heads} T {T} strides {src_stride}/{d_stride} inplace {inplace}", got, want)


def cmp_topk(c, ck, rng, n_cases):
    lib = c.lib
    for case in range(n_cases):
        k = 1 + case % 16
        rows = int(rng.choice([0, 1, 2, 5]))
        m = int(rng.choice([0, 1, 3, k - 1 if k > 1 else 1, k, k + 1, 40, 300]))
        kl = (k + 7) // 8
        d_stride = int(rng.choice([0, kl, kl + 1])) if rows > 1 else int(rng.integers(0, 3))
        a_stride = int(rng.choice([0, L(m), L(m) + 1]))
        spm = fresh_spm(rng, 2048)
        pl = Placer(rng, 2048)
        d = pl.take(E(rows, d_stride or kl, kl))
        a = pl.take(E(rows, a_stride, L(m)))
        if case % 3 == 0:                          # many ties: a few distinct values
            vals = gen_values(rng, 4)
            n_a = E(rows, a_stride, L(m))
            spm[16 * a:16 * (a + n_a)] = rng.choice(vals, size=16 * n_a)
        want = spm.copy()
        topk(want, k, d, a, m, rows, d_stride, a_stride)
        got = spm.copy()
        lib.tsr_ref_topk(k, rows, m, CRef.line(got, d), d_stride, CRef.line(got, a), a_stride)
        ck.same(f"topk k {k} rows {rows} m {m} strides {d_stride}/{a_stride}", got, want)


def cmp_sop(c, ck, rng, n):
    lib = c.lib
    for sub in range(15):
        sa, sb, imm, mem = (gen_values(rng, n, 0.3, 0.1) for _ in range(4))
        want = sop(sub, sa, sb, imm, mem)
        got = np.array([lib.tsr_ref_sop(sub, int(a), int(b), int(i), int(mm))
                        for a, b, i, mm in zip(sa, sb, imm, mem)], dtype=U32)
        ck.same(f"sop {SOP_NAMES[sub]}", got, want)


def compare(lib_path, seed, cases):
    RM.check_environment()
    c = CRef(lib_path)
    ck = Cmp()
    rng = np.random.default_rng(seed)
    cmp_vop(c, ck, rng, cases)
    cmp_red(c, ck, rng, cases)
    cmp_rope(c, ck, rng, max(cases // 2, 40))
    cmp_topk(c, ck, rng, cases)
    cmp_sop(c, ck, rng, 2000)
    if ck.fail:
        print(f"ref_vu_np: {ck.fail} of {ck.checks} comparisons FAILED (seed {seed})", file=sys.stderr)
        return 1
    print(f"ref_vu_np: all {ck.checks} comparisons bit-exact (C tsr_ref_vu vs numpy model, seed {seed})")
    return 0


# ---------------------------------------------------------------------------
# Self-test: directed facts stated by the document
# ---------------------------------------------------------------------------
def self_test():
    RM.check_environment()
    ck = Cmp()
    # §6.8 order: -inf < -1 < -0 < +0 < 1 < +inf < NaN; MAX propagates NaN, MIN drops it.
    seq = u32([0xFF800000, 0xBF800000, 0x80000000, 0x00000000, 0x3F800000, 0x7F800000])
    ck.same("order key monotonic", np.diff(order_key(seq)) > 0, np.ones(5, bool))
    ck.same("maxt(+0,-0)", maxt(0, SIGNB), 0)
    ck.same("mint(+0,-0)", mint(0, SIGNB), SIGNB)
    ck.same("maxt(NaN,1)", maxt(0xFFC00001, 0x3F800000), CANON)
    ck.same("mint(NaN,1)", mint(0xFFC00001, 0x3F800000), 0x3F800000)
    ck.same("mint(NaN,NaN)", mint(0x7F800001, 0xFFFFFFFF), CANON)
    ck.same("mint(sub,NaN) DAZ", mint(0x80000001, 0x7FC00000), SIGNB)
    ck.same("maxt(sub,-1) DAZ", maxt(0x00000005, 0xBF800000), 0)
    # §6.2: FA(0x80C00000, 0x00800000) = -0, FA(-0, +0) = +0; the complete tree of that pair is +0
    ck.same("FA -0 subtree", RM.FA(0x80C00000, 0x00800000), SIGNB)
    p = np.zeros(128, U32)
    p[0], p[1] = 0x80C00000, 0x00800000
    ck.same("tree of -0 pair", tree_full(p), 0)
    ck.same("skip without z() would be -0", RM.FA(0x80C00000, 0x00800000), SIGNB)
    ck.same("skip with z()", tree_skip(p, np.arange(128) < 2), 0)
    # partials start as FA(+0, x): a row of 256 -0 / negative subnormals sums to +0 (a raw load
    # of the first element would give -0 at every level)
    negz = u32([[0x80000000, 0x80000001] * 128])
    ck.same("sum of -0 row", red_rows(RED_SUM, POST_NONE, negz, 0, 0), [0])
    # empty rows: SUM +0, MAX -inf; RCP of an empty SUM is +inf
    ck.same("empty sum", red_rows(RED_SUM, POST_NONE, np.zeros((1, 0), U32), 0, 0), [0])
    ck.same("empty max", red_rows(RED_MAX, POST_NONE, np.zeros((1, 0), U32), 0, 0), [NINF])
    ck.same("empty sum rcp", red_rows(RED_SUM, POST_RCP, np.zeros((1, 0), U32), 0, 0), [PINF])
    # SUM of 1..256 (exact in f32): 32896; SUMSQ of 1..4: 30; MAX with NaN -> NaN
    x = np.arange(1, 257, dtype=np.float32).view(U32)[None, :]
    ck.same("sum 1..256", red_rows(RED_SUM, POST_NONE, x, 0, 0), [f32b(32896.0)])
    ck.same("sumsq 1..4", red_rows(RED_SUMSQ, POST_NONE, x[:, :4], 0, 0), [f32b(30.0)])
    ck.same("max nan", red_rows(RED_MAX, POST_NONE, u32([[0x3F800000, 0xFF800001]]), 0, 0), [CANON])
    # RMS = RC(SQ(FA(FM(y, sk), se))): y = 4, sk = 1/4, se = 0 -> 1
    ck.same("rms", red_post(POST_RMS, f32b(4.0), f32b(0.25), 0), f32b(1.0))
    # VOP rows: stride 0 broadcast, tail zeroing, SETC raw NaN payload, COPY raw subnormal
    spm = np.full(16 * 16, 0xDEADBEEF, dtype=U32)
    spm[16:16 + 5] = np.arange(1, 6, dtype=np.float32).view(U32)
    spm[32:32 + 5] = np.float32(10).view(U32)
    vop(spm, 0x00, 0, 4, 1, 2, 0, 5, 2, 2, 0, 0)          # d line 4, rows 2, stride 2; b broadcast
    ck.same("add broadcast", spm[64:69].view(np.float32), np.arange(11, 16, dtype=np.float32))
    ck.same("add row 1", spm[96:101].view(np.float32), np.arange(11, 16, dtype=np.float32))
    ck.same("tail zero", spm[69:80], np.zeros(11, U32))
    ck.same("between rows untouched", spm[80:96], np.full(16, 0xDEADBEEF, U32))
    vop(spm, 0x16, CMODE_SREG, 10, 0, 0, 0, 3, 1, 0, 0, 0, 0x7FA00001)
    ck.same("setc raw", spm[160:163], np.full(3, 0x7FA00001, U32))
    spm[176:178] = [0x00000001, 0x807FFFFF]
    vop(spm, 0x09, 0, 12, 11, 0, 0, 2, 1, 0, 0, 0)
    ck.same("copy raw", spm[192:194], [0x00000001, 0x807FFFFF])
    # EXPSUB(-inf, x) = +0 also for x = -inf and NaN (VU-21)
    ck.same("expsub -inf", vop_elem(0x0F, u32([NINF, NINF, NINF]), u32([NINF, CANON, 0]), 0), [0, 0, 0])
    # ROPE: position 0 gives phi = 0, cos 1, sin 0: the rotation is the identity (M = 1)
    spm = gen_finite(np.random.default_rng(1), 16 * 64)
    spm[16 * 40] = 0
    want = spm[16 * 8:16 * 12].copy()
    rope(spm, ROPE_NEOX, 0, 0, 0, f32b(1.0), f32b(0.5), 0, 20, 8, 40, 0, 64, 64, 1, 1, 0, 0)
    ck.same("rope pos 0 identity", spm[16 * 20:16 * 24], want)
    # recurrence: theta_1 = FM(Pf, TS)
    ph = rope_angles(np.array([3]), 4, 0, 0, 0, f32b(0.5), 0, None)
    ck.same("rope angles", ph[0], [f32b(3.0), f32b(1.5)])
    # TOPK: ties by index, NaN first, padding
    spm = np.zeros(64, U32)
    spm[16:20] = [0x3F800000, 0x7FC00001, 0x3F800000, 0x80000000]
    topk(spm, 6, 2, 1, 4, 1, 0, 0)
    ck.same("topk", spm[32:44], [1, CANON, 0, 0x3F800000, 2, 0x3F800000, 3, SIGNB,
                                 0xFFFFFFFF, NINF, 0xFFFFFFFF, NINF])
    ck.same("topk tail", spm[44:48], [0, 0, 0, 0])
    # SOP RSQRT = RC(SQ(x)) with two roundings
    ck.same("sop rsqrt 4", sop(11, f32b(4.0), 0, 0, 0), f32b(0.5))
    if ck.fail:
        print(f"ref_vu_np --self-test: {ck.fail} of {ck.checks} FAILED", file=sys.stderr)
        return 1
    print(f"ref_vu_np --self-test: all {ck.checks} checks passed")
    return 0


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--compare", metavar="LIBTSR_REF_SO")
    ap.add_argument("--seed", type=int, default=20261030)
    ap.add_argument("--cases", type=int, default=600)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    rc = 0
    if args.self_test:
        rc |= self_test()
    if args.compare:
        rc |= compare(args.compare, args.seed, args.cases)
    if not args.self_test and not args.compare:
        ap.print_help()
        return 2
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
