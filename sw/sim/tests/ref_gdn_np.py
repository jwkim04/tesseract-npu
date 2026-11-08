#!/usr/bin/env python3
"""Independent numpy model of the fused GDN state update (VU opcode 0x39, VU §7.12, VU-28 … VU-33).

Written from the text of docs/design/02-vector-unit.md §7.12.3 alone: the head tables (a block
frame with 24-bit entries, round half up), the row frame (26-bit window, truncation), the exact
integer dot product, the conversion of a fixed-point sum to f32 (RNE on the binary32 grid with
gradual underflow, then the FTZ rule), the scalar and the element formulas, and the poison rule
for Inf and NaN. The f32 primitives come from ref_math_np.py (the independent numpy model of the
VU primitives). Nothing is shared with sw/sim/tsr_ref_gdn.c; the conversion here is derived on
the binary32 grid (round the integer to the grid spacing of the result's binade, then decide
normal / flush), where the C works from the bit length.

Usage:
  ref_gdn_np.py --compare <libtsr_ref.so> [--seed N] [--cases N]
      Random and directed heads (m = 16 .. 256, rows, exponent spreads, zeros, subnormals, Inf /
      NaN, alpha / beta edge values) through tsr_ref_gdn on the C reference (ctypes) and through
      this model; S' and o compared bit for bit. Also tsr_ref_gdn_fx2f on random magnitudes.
  ref_gdn_np.py --self-test
      The model's own directed checks (the examples of the document).
"""
import argparse
import ctypes
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ref_math_np as RM  # noqa: E402

U32 = np.uint32
CANON = 0x7FC00000
PINF = 0x7F800000
SIGNB = 0x80000000


def u32(x):
    return np.asarray(x, dtype=U32)


def f2u(x):
    return int(np.float32(x).view(U32))


# ---------------------------------------------------------------------------
# Fixed point to f32
# ---------------------------------------------------------------------------
def fx2f(neg, mag, x):
    """f32 bits of (-1)^neg * mag * 2^x (Python integers): round on the binary32 grid (RNE; the grid
    spacing of the result's binade, 2^-149 below 2^-126), overflow to Inf, a subnormal result is
    flushed to a signed zero, mag = 0 gives +0."""
    if mag == 0:
        return 0
    sign = SIGNB if neg else 0
    e = mag.bit_length() - 1 + x                   # exponent of the leading bit
    ulp = max(e, -126) - 23                        # exponent of the grid spacing
    sh = ulp - x                                   # mag / 2^sh is the value in grid units
    if sh <= 0:
        q = mag << (-sh)
    else:
        q = mag >> sh
        rem = mag - (q << sh)
        half = 1 << (sh - 1)
        if rem > half or (rem == half and (q & 1)):
            q += 1
    if e < -126:
        # subnormal grid: q < 2^23 is subnormal (flush), q == 2^23 is 2^-126
        return sign | (1 << 23) if q >= (1 << 23) else sign
    if q == (1 << 24):
        q = 1 << 23
        e += 1
    if e + 127 >= 255:
        return sign | PINF
    return sign | ((e + 127) << 23) | (q & 0x7FFFFF)


# ---------------------------------------------------------------------------
# Tables and dot products
# ---------------------------------------------------------------------------
def parts(x):
    x = u32(x)
    E = ((x >> 23) & 0xFF).astype(np.int64)
    M = ((x & 0x7FFFFF) | 0x800000).astype(np.int64)
    M = np.where(E == 0, 0, M)
    S = ((x >> 31) & 1).astype(np.int64)
    return S, E, M


def table(x):
    """(entries int64 (m,), Emax, bad)"""
    S, E, M = parts(x)
    bad = bool(np.any(E == 255))
    fin = (E != 0) & (E != 255)
    emax = int(E[fin].max()) if fin.any() else 0
    sh = np.where(fin, emax - E, 0)
    shc = np.clip(sh, 1, 62)
    ent = np.where(sh == 0, M, np.where(sh >= 25, 0, (M + (np.int64(1) << (shc - 1))) >> shc))
    ent = np.where(fin, ent, 0)
    ent = np.where(S == 1, -ent, ent)
    return ent, emax, bad


def rowdot(rows, tab):
    """rows (R, m) uint32 against a table; returns uint32 (R,) f32 bits."""
    ent, et, tbad = tab
    S, E, M = parts(rows)
    fin = (E != 0) & (E != 255)
    rbad = np.any(E == 255, axis=1)
    es = np.where(fin, E, 0).max(axis=1)
    sh = np.where(fin, es[:, None] - E, 0)
    A = np.where(sh >= 26, 0, (M << 2) >> np.clip(sh, 0, 62))
    A = np.where(fin, A, 0)
    A = np.where(S == 1, -A, A)
    acc = (A * ent[None, :]).sum(axis=1)
    out = np.zeros(rows.shape[0], dtype=U32)
    for i in range(rows.shape[0]):
        if tbad or rbad[i]:
            out[i] = CANON
        else:
            a = int(acc[i])
            out[i] = fx2f(a < 0, abs(a), int(es[i]) + et - 302)
    return out


def head(m, alpha, beta, k, q, v, S):
    """S (R, m) uint32. Returns (S', o) as uint32 arrays."""
    k, q, v, S = u32(k), u32(q), u32(v), u32(S)
    al, be = U32(alpha), U32(beta)
    with np.errstate(all="ignore"):
        ab = RM.FM(al, be)
        kp = RM.FM(ab, k)
        bv = RM.FM(be, v)
        tk = table(kp)
        tq = table(q)
        kappa = rowdot(k[None, :], tq)[0]
        u = rowdot(S, tk)
        w = rowdot(S, tq)
        e = RM.FS(bv, u)
        Sn = RM.FA(RM.FM(al, S), RM.FM(e[:, None], k[None, :]))
        o = RM.FA(RM.FM(al, w), RM.FM(e, U32(kappa)))
    return u32(Sn), u32(o)


# ---------------------------------------------------------------------------
# Self test (the document's examples)
# ---------------------------------------------------------------------------
def self_test():
    ok = True

    def chk(name, got, want):
        nonlocal ok
        if got != want:
            ok = False
            print("SELF-TEST FAIL %s: got %08x want %08x" % (name, got, want))

    one = 0x3F800000
    chk("one", fx2f(0, 1, 0), one)
    chk("zero", fx2f(1, 0, 3), 0)
    chk("tie to even", fx2f(0, (1 << 24) + 1, 0), f2u(16777216.0))
    chk("up", fx2f(0, (1 << 24) + 3, 0), f2u(16777220.0))
    chk("carry", fx2f(0, (1 << 25) - 1, 0), f2u(33554432.0))
    chk("overflow", fx2f(0, 1, 128), PINF)
    chk("max finite", fx2f(0, 0xFFFFFF, 104), 0x7F7FFFFF)
    chk("tie at 2^128 - 2^103", fx2f(0, 0x1FFFFFF, 103), PINF)
    chk("min normal", fx2f(0, 1, -126), 0x00800000)
    chk("flush", fx2f(0, 1, -127), 0)
    chk("flush edge", fx2f(0, 0xFFFFFF, -150), 0x00800000)
    chk("below edge", fx2f(0, 0xFFFFFE, -150), 0)
    chk("flush neg", fx2f(1, 1, -140), SIGNB)
    # integer dot: [1, 2, 3, 4, ...] . [-1, 1, -1, 1, ...] over 16 elements = 8
    row = u32([f2u(float(i + 1)) for i in range(16)])[None, :]
    tab = table(u32([f2u(1.0 if i % 2 else -1.0) for i in range(16)]))
    chk("integer dot", int(rowdot(row, tab)[0]), f2u(8.0))
    # table frame: 1.0 with 0.5 and 2^-24
    t = table(u32([one, 0x3F000000, f2u(2.0 ** -24)] + [0] * 13))
    chk("frame exp", t[1], 127)
    chk("frame entries", tuple(int(x) for x in t[0][:3]), (0x800000, 0x400000, 1))
    # poison
    row[0, 3] = PINF
    chk("poison", int(rowdot(row, tab)[0]), CANON)
    print("ref_gdn_np self-test:", "ok" if ok else "FAILED")
    return ok


# ---------------------------------------------------------------------------
# Comparison with the C reference
# ---------------------------------------------------------------------------
def load_lib(path):
    lib = ctypes.CDLL(path)
    pu = np.ctypeslib.ndpointer(dtype=np.uint32, flags="C_CONTIGUOUS")
    lib.tsr_ref_gdn.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
                                pu, pu, pu, pu, pu, pu]
    lib.tsr_ref_gdn.restype = None
    lib.tsr_ref_gdn_fx2f.argtypes = [ctypes.c_uint32, ctypes.c_uint64, ctypes.c_int32]
    lib.tsr_ref_gdn_fx2f.restype = ctypes.c_uint32
    return lib


def c_head(lib, m, alpha, beta, k, q, v, S):
    rows = S.shape[0]
    out = np.zeros_like(S)
    o = np.zeros(16 * ((rows + 15) // 16), dtype=U32)
    lib.tsr_ref_gdn(m, rows, alpha, beta, np.ascontiguousarray(k, U32), np.ascontiguousarray(q, U32),
                    np.ascontiguousarray(v, U32), np.ascontiguousarray(S, U32), out, o)
    return out, o


def rand_vec(rng, n, kind):
    """Random f32 bit patterns of several classes."""
    if kind == "gauss":
        x = rng.standard_normal(n).astype(np.float32)
    elif kind == "spread":      # exponents over a wide range
        x = (rng.standard_normal(n) * np.exp2(rng.integers(-60, 60, n))).astype(np.float32)
    elif kind == "narrow":      # nearly equal magnitudes
        x = (rng.choice([-1, 1], n) * (1.0 + rng.random(n) * 0.01)).astype(np.float32)
    elif kind == "sparse":
        x = (rng.standard_normal(n) * (rng.random(n) < 0.1)).astype(np.float32)
    elif kind == "tiny":        # around the underflow region
        x = (rng.standard_normal(n) * np.exp2(rng.integers(-130, -60, n))).astype(np.float32)
    elif kind == "huge":
        x = (rng.standard_normal(n) * np.exp2(rng.integers(60, 120, n))).astype(np.float32)
    elif kind == "bits":        # arbitrary bit patterns (subnormals, Inf, NaN included)
        return rng.integers(0, 1 << 32, n, dtype=np.uint64).astype(U32)
    elif kind == "ints":
        x = rng.integers(-20, 21, n).astype(np.float32)
    else:
        raise ValueError(kind)
    return x.view(U32).copy()


def compare(lib, seed, cases):
    rng = np.random.default_rng(seed)
    kinds = ["gauss", "spread", "narrow", "sparse", "tiny", "huge", "ints"]
    bad = 0
    # the conversion on random magnitudes
    for it in range(20000):
        ln = int(rng.integers(1, 60))
        mag = int(rng.integers(0, 1 << min(ln, 62))) | (1 << (ln - 1))
        if it % 3 == 0:
            mag = (mag & ~0xFFFF) | int(rng.choice([0, 0x8000, 0x8001, 0x7FFF]))
        if it % 7 == 0 and ln > 24:
            mag = (mag >> (ln - 24)) << (ln - 24)
        x = int(rng.choice([-1, 1]) * rng.integers(0, 300) if it % 2 else rng.integers(-400, 250))
        if it % 5 == 0:
            x = -126 - ln + int(rng.integers(-3, 6))
        if it % 11 == 0:
            x = 128 - ln + int(rng.integers(-3, 4))
        neg = int(rng.integers(0, 2))
        g, w = lib.tsr_ref_gdn_fx2f(neg, mag, x), fx2f(neg, mag, x)
        if g != w:
            bad += 1
            print("fx2f mismatch neg %d mag %x x %d: C %08x model %08x" % (neg, mag, x, g, w))
            if bad > 20:
                return bad
    # directed conversions: the rounding ties, the FTZ edge, the overflow edge, signs
    for neg in (0, 1):
        for mag, x in [(1, 0), (1, -126), (1, -127), (0xFFFFFF, -150), (0xFFFFFE, -150), (0xFFFFFD, -150),
                       (0x1FFFFFF, -151), (0x1FFFFFE, -151), (0x1000001, -150), (0x1000003, -150),
                       (0xFFFFFF, 104), (0x1FFFFFF, 103), (0x1FFFFFE, 103), (0x1000000, 104),
                       ((1 << 24) + 1, 0), ((1 << 24) + 3, 0), ((1 << 25) - 1, 0), (1, 128), (1, 127),
                       (0x7FFFFF, -149), (0x800000, -149), (0x7FFFFFFFFFFFFF, -200), (3, -150), (1, -150)]:
            g, w = lib.tsr_ref_gdn_fx2f(neg, mag, x), fx2f(neg, mag, x)
            if g != w:
                bad += 1
                print("fx2f directed mismatch neg %d mag %x x %d: C %08x model %08x" % (neg, mag, x, g, w))
    for case in range(cases):
        m = 16 * int(rng.integers(1, 17))
        rows = int(rng.integers(1, 41))
        ks, qs, ss, vs = (kinds[int(rng.integers(len(kinds)))] for _ in range(4))
        if case % 10 == 9:
            ks = qs = ss = vs = "bits"            # specials everywhere
        k, q = rand_vec(rng, m, ks), rand_vec(rng, m, qs)
        v = rand_vec(rng, rows, vs)
        S = rand_vec(rng, rows * m, ss).reshape(rows, m)
        if case % 4 == 1:
            S[int(rng.integers(rows)), :] = 0       # a zero row
        if case % 6 == 2:
            k[:] = 0
        alpha = f2u(float(np.exp(-rng.random() * 3)))
        beta = f2u(float(1.0 / (1.0 + np.exp(-(rng.random() * 8 - 4)))))
        if case % 13 == 0:
            alpha = int(rng.choice([0, SIGNB, PINF, CANON, 0x00400000, f2u(1.0), f2u(-0.5)]))
        if case % 17 == 0:
            beta = int(rng.choice([0, SIGNB, PINF, CANON, f2u(1.0), f2u(2.0)]))
        cs, co = c_head(lib, m, alpha, beta, k, q, v, S)
        ms, mo = head(m, alpha, beta, k, q, v, S)
        co = co[:rows]
        if not (np.array_equal(cs, ms) and np.array_equal(co, mo)):
            bad += 1
            ds = np.argwhere(cs != ms)
            do = np.argwhere(co != mo)
            print("case %d (m %d rows %d, %s/%s/%s/%s): S' mismatches %d, o mismatches %d" %
                  (case, m, rows, ks, qs, ss, vs, len(ds), len(do)))
            for i, j in ds[:3]:
                print("   S'[%d][%d] C %08x model %08x" % (i, j, cs[i, j], ms[i, j]))
            for (i,) in do[:3]:
                print("   o[%d] C %08x model %08x" % (i, co[i], mo[i]))
            if bad > 10:
                break
    print("ref_gdn_np compare: %d cases, %d mismatching" % (cases, bad))
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", metavar="LIB")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--cases", type=int, default=600)
    a = ap.parse_args()
    rc = 0
    if a.self_test and not self_test():
        rc = 1
    if a.compare:
        if compare(load_lib(a.compare), a.seed, a.cases):
            rc = 1
    if not a.self_test and not a.compare:
        ap.print_help()
        rc = 2
    return rc


if __name__ == "__main__":
    sys.exit(main())
