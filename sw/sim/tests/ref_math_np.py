#!/usr/bin/env python3
"""Independent numpy bit model of the VU's approximate and composite math (SW-23).

Written from the text of docs/design/02-vector-unit.md §6 alone (with the fp rules it cites: VU §4.1,
docs/spec.md "Common fp rules" and "Implementation clarifications"). It deliberately shares no code with
sw/sim/tsr_ref_math.c: the ref_math_* ctests compare the two bit for bit, so a coding error in
either one shows up as a mismatch.

Usage:
  ref_math_np.py <func> <start_hex> <count> <stride> [<b_hex>]
      Evaluates <func> on the f32 bit patterns start, start + stride, ... (count values, uint32
      wraparound) and writes the uint32 result bit patterns little-endian raw to stdout.
      <b_hex> is the fixed second operand of a binary function (for expk: the κ index).
      count and stride are decimal or 0x-prefixed hex.
  ref_math_np.py --self-test     special values, constants, properties, accuracy vs double
  ref_math_np.py --list          the function names

Emulation rules:
  - Values are uint32 bit-pattern arrays. Every f32 operation is one of the §6.1 primitives FA,
    FS, FM, FD, SQ, RC: DAZ on the inputs by bit test, one numpy float32 operation (IEEE binary32,
    round-to-nearest-even, gradual underflow, no FMA), then FTZ (subnormal -> signed zero) and
    canonical NaN by bit test. The FTZ boundary is the spec's: a result flushes iff the exact
    value is below 2^-126 - 2^-150 in magnitude, which is what flushing the gradual-underflow
    result gives.
  - Everything else is integer arithmetic on numpy uint64 / int64, with the 88- and 125-bit
    products of SIN/COS done in two 64-bit words.
  - Large counts are processed in chunks of 2^16 values (fastest here, and ~50 MB peak).
"""
import sys

import numpy as np

U64 = np.uint64
I64 = np.int64

SIGN = 0x80000000
EXPM = 0x7F800000
MANT = 0x007FFFFF
ABS = 0x7FFFFFFF
CANON_NAN = 0x7FC00000
PINF = 0x7F800000

# Appendix A constants (bit patterns).
ONE, HALF, THREE = 0x3F800000, 0x3F000000, 0x40400000
THIRD = 0x3EAAAAAB
KG = 0x41B2E92F
P1F, P2F, P3F = 0xB17217F8, 0x3D7F7BFF, 0x0E35846C
C64 = 0xA2F9836E4E44152A
PIO2 = 0x6487ED5110B4611A
SN1, SN2, SN3 = 0xBE2AAAA3, 0x3C08839E, 0xB94CA1F9
CS1, CS2, CS3 = 0x3D2AAAA5, 0xBAB6061A, 0x37CCF5CE

# κ -> (sK, LK). The index order (P1, N1, N2, NG) is the order of the §6.3 table; it is the
# meaning of b for the binary CLI name "expk".
KAPPA = {
    "p1": (0, 0x5C551D94AE),
    "n1": (1, 0x5C551D94AE),
    "n2": (1, 0xB8AA3B295C),
    "ng": (1, 0x6969F3DD6),
}
KAPPA_BY_INDEX = ("p1", "n1", "n2", "ng")

# T64[j] = round(2^(j/64) * 2^30), Appendix A.
T64 = np.array([
    0x40000000, 0x40B268FA, 0x4166C34C, 0x421D1462, 0x42D561B4, 0x438FB0CB, 0x444C0740, 0x450A6ABB,
    0x45CAE0F2, 0x468D6FAE, 0x47521CC6, 0x4818EE22, 0x48E1E9BA, 0x49AD1598, 0x4A7A77D4, 0x4B4A169C,
    0x4C1BF829, 0x4CF022CA, 0x4DC69CDD, 0x4E9F6CD4, 0x4F7A9930, 0x50582888, 0x51382182, 0x521A8AD7,
    0x52FF6B55, 0x53E6C9DA, 0x54D0AD5A, 0x55BD1CDB, 0x56AC1F75, 0x579DBC57, 0x5891FAC1, 0x5988E209,
    0x5A82799A, 0x5B7EC8F2, 0x5C7DD7A4, 0x5D7FAD59, 0x5E8451D0, 0x5F8BCCDB, 0x60962665, 0x61A3666D,
    0x62B39509, 0x63C6BA64, 0x64DCDEC3, 0x65F60A7F, 0x6712460B, 0x683199ED, 0x69540EC9, 0x6A79AD56,
    0x6BA27E65, 0x6CCE8AE1, 0x6DFDDBCC, 0x6F307A41, 0x70666F76, 0x719FC4B9, 0x72DC8374, 0x741CB528,
    0x75606374, 0x76A7980F, 0x77F25CCE, 0x7940BB9E, 0x7A92BE8B, 0x7BE86FBA, 0x7D41D96E, 0x7E9F0606,
], dtype=np.uint64)

M32 = U64(0xFFFFFFFF)
MASK62 = U64((1 << 62) - 1)
CHUNK = 1 << 16


# ---------------------------------------------------------------------------
# f32 primitives (VU §6.1): DAZ inputs, one native binary32 RNE op, FTZ + canonical NaN.
# ---------------------------------------------------------------------------
def u32(x):
    return np.asarray(x, dtype=np.uint32)


def daz(u):
    u = u32(u)
    return np.where((u & EXPM) == 0, u & SIGN, u)


def isnan(u):
    u = u32(u)
    return ((u & EXPM) == EXPM) & ((u & MANT) != 0)


def _fin(r):
    u = u32(r)
    e = u & EXPM
    u = np.where((e == EXPM) & ((u & MANT) != 0), np.uint32(CANON_NAN), u)
    return np.where(e == 0, u & SIGN, u)


def _f(u):
    return daz(u).view(np.float32)


def FA(a, b):
    return _fin((_f(a) + _f(b)).view(np.uint32))


def FS(a, b):
    return _fin((_f(a) - _f(b)).view(np.uint32))


def FM(a, b):
    return _fin((_f(a) * _f(b)).view(np.uint32))


def FD(a, b):
    return _fin((_f(a) / _f(b)).view(np.uint32))


def SQ(a):
    return _fin(np.sqrt(_f(a)).view(np.uint32))


def RC(a):
    return _fin((np.float32(1.0) / _f(a)).view(np.uint32))


def neg(u):
    return u32(u) ^ np.uint32(SIGN)


# ---------------------------------------------------------------------------
# Wide integer helpers (uint64 arrays; shifts by >= 64 are made explicit)
# ---------------------------------------------------------------------------
def _sh(s):
    """Clip an int64 shift array into [0, 63] as uint64 (callers mask the out-of-range lanes)."""
    return np.clip(s, 0, 63).astype(np.uint64)


def _shr64(x, s):
    """x >> s for uint64 x and int64 s >= 0; a shift by >= 64 gives 0 (VU §6.1)."""
    return np.where(s >= 64, U64(0), x >> _sh(s))


def _shr128_lo(hi, lo, s):
    """Low 64 bits of ((hi:lo) >> s), s an int64 array in [0, 127]."""
    small = (lo >> _sh(s)) | np.where((s > 0) & (s < 64), hi << _sh(64 - s), U64(0))
    big = hi >> _sh(s - 64)
    return np.where(s < 64, small, big)


def _low_nonzero128(hi, lo, s):
    """((hi:lo) mod 2^s) != 0, s an int64 array in [0, 127]."""
    mlo = np.where(s >= 64, U64(0xFFFFFFFFFFFFFFFF), (U64(1) << _sh(s)) - U64(1))
    mhi = np.where(s > 64, (U64(1) << _sh(s - 64)) - U64(1), U64(0))
    return ((lo & mlo) | (hi & mhi)) != 0


def _mul64(a, b):
    """Exact 64 x 64 -> 128-bit product of uint64 arrays, as (hi, lo)."""
    a0, a1 = a & M32, a >> U64(32)
    b0, b1 = b & M32, b >> U64(32)
    p00, p01, p10, p11 = a0 * b0, a0 * b1, a1 * b0, a1 * b1
    mid = (p00 >> U64(32)) + (p01 & M32) + (p10 & M32)
    lo = (p00 & M32) | (mid << U64(32))
    hi = p11 + (p01 >> U64(32)) + (p10 >> U64(32)) + (mid >> U64(32))
    return hi, lo


def _bitlen64(x):
    """Bit length of each uint64 (0 for 0), as int64."""
    x = np.array(x, dtype=np.uint64)
    n = np.zeros(x.shape, dtype=np.int64)
    for s in (32, 16, 8, 4, 2, 1):
        m = x >= (U64(1) << U64(s))
        n += np.where(m, s, 0)
        x = np.where(m, x >> U64(s), x)
    return n + (x != 0)


# ---------------------------------------------------------------------------
# EXPK(v, κ), VU §6.3
# ---------------------------------------------------------------------------
def expk(u, kappa):
    sK, LK = KAPPA[kappa]
    v = daz(u)                                                     # 1.
    nan = isnan(v)
    sgn = ((v >> 31) ^ sK) != 0                                    # 2.
    inf = (v & ABS) == PINF                                        # 3.
    zero = (v & ABS) == 0
    Ev = ((v >> 23) & 0xFF).astype(I64)                            # 4.
    Mv = ((v & MANT) | 0x800000).astype(U64)
    sat = Ev > 156                                                 # 5.
    P = Mv * U64(LK)                                               # 6. (< 2^64)
    T = _shr64(P, 156 - Ev)
    sat |= T >= U64(1 << 40)
    T = np.where(sat, U64(0), T)
    Ti = T.astype(I64)
    t = np.where(sgn, -Ti, Ti)                                     # 7.
    n = t >> 32                                                    # arithmetic (floor)
    F = (t & 0xFFFFFFFF).astype(U64)
    j = (F >> U64(26)).astype(np.intp)                             # 8.
    G = F & U64((1 << 26) - 1)
    X3 = (G * U64(P3F)) >> U64(32)                                 # 9.
    S2 = U64(P2F) + X3
    X2 = (G * S2) >> U64(32)
    S1 = U64(P1F) + X2
    PF = (G * S1) >> U64(32)
    Tj = T64[j]                                                    # 10.
    M = Tj + ((Tj * PF) >> U64(32))
    big = M >= U64(1 << 31)                                        # 11.
    drop = np.where(big, U64(8), U64(7))
    n = n + big
    sig = M >> drop                                                # 12.
    rem = M & ((U64(1) << drop) - U64(1))
    half = U64(1) << (drop - U64(1))
    up = (rem > half) | ((rem == half) & ((sig & U64(1)) == U64(1)))
    sig = sig + up
    carry = sig == U64(1 << 24)
    sig = np.where(carry, U64(1 << 23), sig)
    n = n + carry
    be = n + 127                                                   # 13.
    y = (be << 23) | (sig.astype(I64) - (1 << 23))
    y = np.where(be >= 255, PINF, np.where(be <= 0, 0, y))
    ovf = np.where(sgn, 0, PINF)
    y = np.where(sat, ovf, y)
    y = np.where(zero, ONE, y)
    y = np.where(inf, ovf, y)
    y = np.where(nan, CANON_NAN, y)
    return y.astype(np.uint32)


# ---------------------------------------------------------------------------
# Composite functions, VU §6.5
# ---------------------------------------------------------------------------
def exp(u):
    return expk(u, "p1")


def sigmoid(u):
    return RC(FA(ONE, expk(u, "n1")))


def silu(u):
    return FM(u, sigmoid(u))


def gelu_tanh(u):
    x = u32(u)
    w = FM(x, x)
    w = FA(w, KG)
    v = FM(w, x)
    return FM(x, RC(FA(ONE, expk(v, "ng"))))


def tanh(u):
    x = u32(u)
    ax = daz(x) & np.uint32(ABS)
    small = ax < 0x3C800000
    q = FM(ax, ax)
    w = FS(THREE, q)
    ts = FM(FM(w, ax), THIRD)
    e = expk(ax, "n2")
    tl = FM(FS(ONE, e), RC(FA(ONE, e)))
    t = np.where(small, ts, tl)
    y = np.where((x >> 31) != 0, neg(t), t)
    return np.where(isnan(x), np.uint32(CANON_NAN), y).astype(np.uint32)


def f32_to_f16_rne(u):
    """F16: DAZ, then RNE to binary16 with gradual underflow; overflow gives ±inf. NaN gives
    sign | 0x7E00 (docs/spec.md "NaN in fp16 conversion"). Returns the 16-bit pattern in a uint32."""
    x = daz(u)
    sign = (x >> 16) & 0x8000
    e = ((x >> 23) & 0xFF).astype(I64)
    m = ((x & MANT) | 0x800000).astype(U64)
    sh = np.minimum(np.maximum(13, 126 - e), 40).astype(U64)      # 13: normal, more: subnormal
    r = m >> sh
    rem = m & ((U64(1) << sh) - U64(1))
    half = U64(1) << (sh - U64(1))
    r = r + ((rem > half) | ((rem == half) & ((r & U64(1)) == U64(1))))
    h = (np.maximum(e - 113, 0).astype(U64) << U64(10)) + r     # a carry walks into the exponent
    h = np.minimum(h, U64(0x7C00))
    h = np.where(e == 0, U64(0), h)
    h = np.where(e == 255, np.where((x & MANT) != 0, U64(0x7E00), U64(0x7C00)), h)
    return (sign | h.astype(np.uint32)).astype(np.uint32)


def f16_to_f32(h):
    """F32: exact widening of the binary16 pattern in the low 16 bits (subnormals included; NaN
    keeps sign and payload and gets the quiet bit, docs/spec.md "NaN in fp16 conversion")."""
    h = (u32(h) & 0xFFFF).astype(I64)
    sign = (h & 0x8000) << 16
    e = (h >> 10) & 0x1F
    f = h & 0x3FF
    normal = sign | ((e + 112) << 23) | (f << 13)
    p = _bitlen64(f.astype(U64)) - 1                              # msb of a subnormal fraction
    sub = sign | ((p + 103) << 23) | ((f << np.clip(23 - p, 0, 31)) & MANT)
    special = sign | PINF | (f << 13) | np.where(f != 0, 0x400000, 0)
    y = np.where(e == 31, special, np.where(e == 0, np.where(f == 0, sign, sub), normal))
    return y.astype(np.uint32)


def _H(u):
    return f16_to_f32(f32_to_f16_rne(u))


def gelu16(u):
    x = daz(u)
    xf = x.view(np.float32)
    y = _H(gelu_tanh(_H(x)))
    y = np.where(xf >= np.float32(10.0), x, y)
    y = np.where(xf <= np.float32(-10.0), np.uint32(0), y)
    return np.where(isnan(x), np.uint32(CANON_NAN), y).astype(np.uint32)


def geglu16(a, b):
    af = daz(a).view(np.float32)
    return np.where(af <= np.float32(-10.0), np.uint32(0), FM(gelu16(a), b)).astype(np.uint32)


def expsub(a, x):
    return np.where(daz(a) == 0xFF800000, np.uint32(0), expk(FS(a, x), "p1")).astype(np.uint32)


def swiglu(a, b):
    return FM(silu(a), b)


def geglu(a, b):
    return FM(gelu_tanh(a), b)


# ---------------------------------------------------------------------------
# SIN and COS, VU §6.6
# ---------------------------------------------------------------------------
def _f32_from_q124(Fs):
    """The f32 nearest-even to (Fs × PIO2)·2^-124, Fs a signed int64 array in [-2^61, 2^61)."""
    negf = Fs < 0
    a = np.abs(Fs).astype(U64)
    rh, rl = _mul64(a, U64(PIO2))
    p = np.where(rh != 0, 64 + _bitlen64(rh), _bitlen64(rl)) - 1  # msb index of |R|
    sa = p - 24                                                    # leaves 25 bits: sig + round
    top = _shr128_lo(rh, rl, np.clip(sa, 0, 127))
    sig = top >> U64(1)
    rbit = (top & U64(1)) != 0
    sticky = _low_nonzero128(rh, rl, np.clip(sa, 0, 127))
    sig = sig + (rbit & (sticky | ((sig & U64(1)) != 0)))
    carry = sig == U64(1 << 24)
    sig = np.where(carry, U64(1 << 23), sig)
    be = p + 3 + carry                                             # 2^(p-124) -> biased p + 3
    y = (negf.astype(I64) << 31) | (be << 23) | (sig.astype(I64) - (1 << 23))
    return np.where(a == 0, 0, y).astype(np.uint32)


def sincos_reduce(th):
    """Steps 3-5 on DAZ'd θ bits: (q = k mod 4 as uint64, r as f32 bits). Meaningful for lanes
    with 115 <= E <= 254; the other lanes return garbage that sincos() masks."""
    E = ((th >> 23) & 0xFF).astype(I64)
    M = ((th & MANT) | 0x800000).astype(U64)

    # 3. P = M × C64 (88 bits) as (phi:plo); θ·2/π ≈ P·2^-sh.
    pa = M * U64(C64 & 0xFFFFFFFF)
    pb = M * U64(C64 >> 32)
    plo = pa + (pb << U64(32))
    phi = (pb >> U64(32)) + (plo < pa).astype(U64)
    sh = 214 - E
    k = np.where(sh >= 0,
                 _shr128_lo(phi, plo, np.clip(sh, 0, 127)),
                 plo << _sh(-sh)) & U64(3)                         # only k mod 4 is used
    F = np.where(sh >= 62, _shr128_lo(phi, plo, np.clip(sh - 62, 0, 127)) & MASK62,
                 np.where(sh >= 0, (plo << _sh(62 - sh)) & MASK62, U64(0)))
    # 4.
    wrap = F >= U64(1 << 61)
    q = (k + wrap.astype(U64)) & U64(3)
    Fs = F.astype(I64) - np.where(wrap, I64(1 << 62), I64(0))
    # 5.
    return q, _f32_from_q124(Fs)


def sincos(u):
    th = daz(u)
    s = (th >> 31) != 0
    E = (th >> 23) & 0xFF
    q, r = sincos_reduce(th)

    # 6. polynomials
    z = FM(r, r)
    t = FM(SN3, z)
    t = FA(t, SN2)
    t = FM(t, z)
    t = FA(t, SN1)
    t = FM(t, z)
    t = FM(t, r)
    sn = FA(t, r)
    t = FM(CS3, z)
    t = FA(t, CS2)
    t = FM(t, z)
    t = FA(t, CS1)
    t = FM(t, z)
    t = FM(t, z)
    h = FM(HALF, z)
    v = FS(t, h)
    cs = FA(v, ONE)

    # 7.-8. quadrant and sign
    S = np.select([q == 0, q == 1, q == 2], [sn, cs, neg(sn)], neg(cs))
    C = np.select([q == 0, q == 1, q == 2], [cs, neg(sn), neg(cs)], sn)
    S = np.where(s, neg(S), S)

    # specials: 1.-2. (E < 115 includes E = 0) and NaN / ±inf
    tiny = E < 115
    S = np.where(tiny, th, S)
    C = np.where(tiny, np.uint32(ONE), C)
    bad = E == 255
    S = np.where(bad, np.uint32(CANON_NAN), S)
    C = np.where(bad, np.uint32(CANON_NAN), C)
    return S.astype(np.uint32), C.astype(np.uint32)


def sin(u):
    return sincos(u)[0]


def cos(u):
    return sincos(u)[1]


def _sincos_interleaved(u):
    s, c = sincos(u)
    return np.stack([s, c], axis=-1).reshape(-1)


# ---------------------------------------------------------------------------
# Function table and CLI
# ---------------------------------------------------------------------------
UNARY = {
    "exp": exp,
    "expk_p1": lambda u: expk(u, "p1"),
    "expk_n1": lambda u: expk(u, "n1"),
    "expk_n2": lambda u: expk(u, "n2"),
    "expk_ng": lambda u: expk(u, "ng"),
    "sigmoid": sigmoid,
    "silu": silu,
    "gelu_tanh": gelu_tanh,
    "gelu16": gelu16,
    "tanh": tanh,
    "sin": sin,
    "cos": cos,
    "sincos": _sincos_interleaved,                # two words per input: sin, cos
    "f32_to_f16_rne": f32_to_f16_rne,             # 16-bit result, zero-extended
    "f16_to_f32": f16_to_f32,                     # input: the low 16 bits
    "fsqrt": SQ,
    "frecip": RC,
}
BINARY = {
    "expk": lambda u, b: expk(u, KAPPA_BY_INDEX[b]),   # b = κ index (P1, N1, N2, NG)
    "expsub": expsub,
    "geglu16": geglu16,
    "geglu": geglu,
    "swiglu": swiglu,
    "fadd": FA,
    "fsub": FS,
    "fmul": FM,
    "fdiv": FD,
}
ALIASES = {"sig": "sigmoid", "gelu": "gelu_tanh"}


def canon_name(name):
    n = name.lower()
    for pre in ("tsr_m_", "tsr_"):
        if n.startswith(pre):
            n = n[len(pre):]
            break
    return ALIASES.get(n, n)


def evaluate(name, u, b=None):
    """Bits of <name> on the uint32 array u (b: the fixed second operand of a binary function)."""
    n = canon_name(name)
    u = u32(u)
    with np.errstate(all="ignore"):
        if n in UNARY:
            if b is not None:
                raise ValueError(f"{name} is unary; no b operand")
            return u32(UNARY[n](u))
        if n in BINARY:
            if b is None:
                raise ValueError(f"{name} needs a b operand")
            if n == "expk":
                if not 0 <= b < len(KAPPA_BY_INDEX):
                    raise ValueError("expk: b is the κ index 0..3 (P1, N1, N2, NG)")
                return u32(BINARY[n](u, b))
            return u32(BINARY[n](u, np.uint32(b)))
    raise ValueError(f"unknown function {name!r}")


def inputs(start, i0, n, stride):
    idx = np.arange(i0, i0 + n, dtype=np.uint64)
    return ((U64(start) + idx * U64(stride)) & M32).astype(np.uint32)


def check_environment():
    """numpy float32 arithmetic must run with gradual underflow (no FTZ/DAZ in MXCSR)."""
    with np.errstate(all="ignore"):
        x = np.array([0x00800000], dtype=np.uint32).view(np.float32)
        y = (x * np.float32(0.5)).view(np.uint32)
    if int(y[0]) != 0x00400000:
        sys.exit("ref_math_np: float32 arithmetic flushes subnormals in this process; "
                 "the model needs the default IEEE environment")


def run(func, start, count, stride, b):
    out = sys.stdout.buffer
    for i0 in range(0, count, CHUNK):
        n = min(CHUNK, count - i0)
        y = evaluate(func, inputs(start, i0, n, stride), b)
        out.write(y.astype("<u4").tobytes())
    out.flush()


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
class _Checker:
    def __init__(self):
        self.fails = 0
        self.count = 0

    def eq(self, label, got, want):
        self.count += 1
        ok = int(got) == int(want)
        if not ok:
            self.fails += 1
        print(f"  {'ok  ' if ok else 'FAIL'} {label:<44} got 0x{int(got):08X} want 0x{int(want):08X}")

    def true(self, label, cond, detail=""):
        self.count += 1
        if not cond:
            self.fails += 1
        print(f"  {'ok  ' if cond else 'FAIL'} {label}{(' ' + detail) if detail else ''}")


def _bits_of(x):
    return int(np.array([x], dtype=np.float32).view(np.uint32)[0])


def _ev1(name, x, b=None):
    return int(evaluate(name, np.array([x], dtype=np.uint32), b)[0])


def _ulp_err(y_bits, ref):
    """|y - ref| in f32 ulps of ref (ref: float64 array, normal-range f32 magnitudes)."""
    y = y_bits.view(np.float32).astype(np.float64)
    _, e = np.frexp(ref)
    ulp = np.ldexp(1.0, np.maximum(e - 24, -149))
    return np.abs(y - ref) / ulp


def _py_expk(u, kappa):
    """Scalar transcription of §6.3 with Python ints (cross-checks the vectorised integer code)."""
    sK, LK = KAPPA[kappa]
    if (u & EXPM) == 0:
        u &= SIGN
    if (u & EXPM) == EXPM and (u & MANT):
        return CANON_NAN
    sgn = (u >> 31) ^ sK
    sat = 0 if sgn else PINF
    if (u & ABS) == PINF:
        return sat
    if (u & ABS) == 0:
        return ONE
    Ev, Mv = (u >> 23) & 0xFF, (1 << 23) | (u & MANT)
    if Ev > 156:
        return sat
    T = (Mv * LK) >> (156 - Ev)
    if T >= 1 << 40:
        return sat
    t = -T if sgn else T
    n, F = t >> 32, t % (1 << 32)
    j, G = F >> 26, F % (1 << 26)
    S2 = P2F + ((G * P3F) >> 32)
    S1 = P1F + ((G * S2) >> 32)
    PF = (G * S1) >> 32
    M = int(T64[j]) + ((int(T64[j]) * PF) >> 32)
    drop = 8 if M >= 1 << 31 else 7
    n += drop - 7
    sig, rem, half = M >> drop, M % (1 << drop), 1 << (drop - 1)
    if rem > half or (rem == half and sig & 1):
        sig += 1
    if sig == 1 << 24:
        sig, n = 1 << 23, n + 1
    be = n + 127
    return PINF if be >= 255 else 0 if be <= 0 else (be << 23) | (sig - (1 << 23))


def _py_reduce(u):
    """Scalar transcription of §6.6 steps 3-5 with Python ints: (q, r bits), 115 <= E <= 254."""
    E, M = (u >> 23) & 0xFF, (1 << 23) | (u & MANT)
    P, sh = M * C64, 214 - E
    if sh >= 62:
        k, F = P >> sh, (P >> (sh - 62)) % (1 << 62)
    elif sh >= 0:
        k, F = P >> sh, (P << (62 - sh)) % (1 << 62)
    else:
        k, F = P << -sh, 0
    if F >= 1 << 61:
        k, F = k + 1, F - (1 << 62)
    R = F * PIO2
    if R == 0:
        return k % 4, 0
    a = abs(R)
    p = a.bit_length() - 1
    sa = p - 23
    sig, rem, half = a >> sa, a % (1 << sa), 1 << (sa - 1)
    if rem > half or (rem == half and sig & 1):
        sig += 1
    if sig == 1 << 24:
        sig, p = 1 << 23, p + 1
    return k % 4, (int(R < 0) << 31) | ((p + 3) << 23) | (sig - (1 << 23))


def _self_test_scalar(ck, rng):
    u = rng.integers(0, 1 << 32, 20000, dtype=np.uint64).astype(np.uint32)
    edge = np.array([0x3F800000, 0x43200000, 0x431C0000, 0x4E000000, 0x4E7FFFFF, 0xCE000001,
                     0x00800000, 0x80800000, 0x7F7FFFFF, 0xFF7FFFFF, 0x3F317218, 0xC2AEAC50,
                     0x42B17218, 0x42B17217, 0xC2CE0000, 0xC2FFFFFF, 0x43000000],
                    dtype=np.uint32)
    u = np.concatenate([u, edge])
    for kap in KAPPA_BY_INDEX:
        got = evaluate(f"expk_{kap}", u)
        bad = [int(x) for x, g in zip(u, got) if _py_expk(int(x), kap) != int(g)]
        ck.true(f"expk_{kap}: vectorised = scalar Python-int transcription ({len(u)} inputs)",
                not bad, f"first mismatch 0x{bad[0]:08X}" if bad else "")
    # every exponent the reduction handles (115..254), random mantissas, both signs
    E = np.repeat(np.arange(115, 255, dtype=np.uint32), 150)
    m = rng.integers(0, 1 << 23, E.size, dtype=np.uint64).astype(np.uint32)
    m[::150] = 0
    m[1::150] = MANT
    s = rng.integers(0, 2, E.size, dtype=np.uint64).astype(np.uint32) << 31
    th = s | (E << 23) | m
    with np.errstate(all="ignore"):
        q, r = sincos_reduce(th)
    bad = [int(x) for x, qq, rr in zip(th, q, r) if _py_reduce(int(x)) != (int(qq), int(rr))]
    ck.true(f"SIN/COS reduction: vectorised = scalar Python-int transcription ({th.size} inputs)",
            not bad, f"first mismatch 0x{bad[0]:08X}" if bad else "")


def _self_test_constants(ck):
    try:
        import mpmath as mp
    except ImportError:
        print("  skip constants (no mpmath)")
        return
    mp.mp.prec = 300
    log2e = 1 / mp.log(2)
    c2g = 2 * mp.sqrt(2 / mp.pi) * mp.mpf("0.044715")
    ck.eq("LK(P1) = round(log2e 2^38)", KAPPA["p1"][1], int(mp.nint(log2e * 2 ** 38)))
    ck.eq("LK(N2) = 2 LK(P1)", KAPPA["n2"][1], 2 * KAPPA["p1"][1])
    ck.eq("LK(NG) = round(C2G log2e 2^38)", KAPPA["ng"][1], int(mp.nint(c2g * log2e * 2 ** 38)))
    ln2 = mp.log(2)
    ck.eq("P1F = round(ln2 2^32)", P1F, int(mp.nint(ln2 * 2 ** 32)))
    ck.eq("P2F = round(ln2^2/2 2^32)", P2F, int(mp.nint(ln2 ** 2 / 2 * 2 ** 32)))
    ck.eq("P3F = round(ln2^3/6 2^32)", P3F, int(mp.nint(ln2 ** 3 / 6 * 2 ** 32)))
    t64_ok = all(int(T64[j]) == int(mp.nint(mp.mpf(2) ** (mp.mpf(j) / 64) * 2 ** 30))
                 for j in range(64))
    ck.true("T64[j] = round(2^(j/64) 2^30), j = 0..63", t64_ok)
    ck.true("C64 = round(2/pi 2^64)", C64 == int(mp.nint(2 / mp.pi * mp.mpf(2) ** 64)))
    ck.true("PIO2 = round(pi/2 2^62)", PIO2 == int(mp.nint(mp.pi / 2 * mp.mpf(2) ** 62)))

    def f32_of(x):  # f32 nearest to a positive normal mpf
        e = int(mp.floor(mp.log(x, 2)))
        m = int(mp.nint(x * mp.mpf(2) ** (23 - e)))
        if m == 1 << 24:
            m, e = 1 << 23, e + 1
        return ((e + 127) << 23) | (m - (1 << 23))

    ck.eq("KG = f32(1/0.044715)", KG, f32_of(1 / mp.mpf("0.044715")))
    ck.eq("THIRD = f32(1/3)", THIRD, f32_of(mp.mpf(1) / 3))


def _self_test_specials(ck):
    NINF, PZ, NZ, NAN, SNAN, NNAN = 0xFF800000, 0, SIGN, CANON_NAN, 0x7F800001, 0xFFC12345
    DEN, NDEN = 0x00000001, 0x807FFFFF
    # spec fp boundary vectors (§21) and the §6.2 FTZ examples
    ck.eq("FM(0x3F7FFFFF, 0x00800000)", _ev1("fmul", 0x3F7FFFFF, 0x00800000), 0x00800000)
    ck.eq("FM(0x3F1FA0BA, 0x00CD4708)", _ev1("fmul", 0x3F1FA0BA, 0x00CD4708), 0x00800000)
    ck.eq("FD(0x00FFFFFF, 2)", _ev1("fdiv", 0x00FFFFFF, 0x40000000), 0x00800000)
    ck.eq("FM(0x3F7FFFFE, 0x00800000)", _ev1("fmul", 0x3F7FFFFE, 0x00800000), 0)
    ck.eq("FA(0x80C00000, 0x00800000) = -0", _ev1("fadd", 0x80C00000, 0x00800000), NZ)
    ck.eq("FA(-0, +0) = +0", _ev1("fadd", NZ, PZ), PZ)
    ck.eq("FA(NaN payload, 1) canonical", _ev1("fadd", NNAN, ONE), NAN)
    ck.eq("RC(denormal) = +inf", _ev1("frecip", DEN), PINF)
    ck.eq("SQ(-1) = NaN", _ev1("fsqrt", 0xBF800000), NAN)
    # EXPK
    for kap in KAPPA_BY_INDEX:
        sK = KAPPA[kap][0]
        ck.eq(f"expk_{kap}(+0) = 1", _ev1(f"expk_{kap}", PZ), ONE)
        ck.eq(f"expk_{kap}(-0) = 1", _ev1(f"expk_{kap}", NZ), ONE)
        ck.eq(f"expk_{kap}(denormal) = 1", _ev1(f"expk_{kap}", NDEN), ONE)
        ck.eq(f"expk_{kap}(+inf)", _ev1(f"expk_{kap}", PINF), 0 if sK else PINF)
        ck.eq(f"expk_{kap}(-inf)", _ev1(f"expk_{kap}", NINF), PINF if sK else 0)
        ck.eq(f"expk_{kap}(sNaN) canonical", _ev1(f"expk_{kap}", SNAN), NAN)
    ck.eq("exp(1) = f32(e)", _ev1("exp", ONE), 0x402DF854)
    ck.eq("exp(-1) = f32(1/e)", _ev1("exp", 0xBF800000), 0x3EBC5AB2)
    ck.eq("exp(ln2 rounded) ~ 2", _ev1("exp", 0x3F317218), 0x40000000)
    ck.eq("exp(88.72) = +inf", _ev1("exp", _bits_of(88.73)), PINF)
    ck.eq("exp(-103) = +0 (no subnormal)", _ev1("exp", _bits_of(-103.0)), 0)
    ck.eq("expk(x, b=1) = expk_n1(x)", _ev1("expk", ONE, 1), _ev1("expk_n1", ONE))
    # composites: the §6.5 specials table
    table = [("exp", 0, PINF), ("sigmoid", 0, ONE), ("silu", NAN, PINF), ("gelu_tanh", NAN, PINF),
             ("gelu16", 0, PINF), ("tanh", 0xBF800000, ONE)]
    for fn, at_ninf, at_pinf in table:
        ck.eq(f"{fn}(-inf)", _ev1(fn, NINF), at_ninf)
        ck.eq(f"{fn}(+inf)", _ev1(fn, PINF), at_pinf)
        ck.eq(f"{fn}(NaN) canonical", _ev1(fn, NNAN), NAN)
    ck.eq("sigmoid(0) = 0.5", _ev1("sigmoid", PZ), HALF)
    ck.eq("silu(+0) = +0", _ev1("silu", PZ), PZ)
    ck.eq("silu(-0) = -0", _ev1("silu", NZ), NZ)
    ck.eq("gelu_tanh(0) = 0", _ev1("gelu_tanh", PZ), PZ)
    ck.eq("gelu_tanh(-1e20) = -0", _ev1("gelu_tanh", _bits_of(-1e20)), NZ)
    ck.eq("tanh(+0) = +0", _ev1("tanh", PZ), PZ)
    ck.eq("tanh(-0) = -0", _ev1("tanh", NZ), NZ)
    ck.eq("tanh(-denormal) = -0", _ev1("tanh", NDEN), NZ)
    ck.eq("tanh(20) = 1", _ev1("tanh", _bits_of(20.0)), ONE)
    ck.eq("gelu16(0) = 0", _ev1("gelu16", PZ), PZ)
    ck.eq("gelu16(10) = 10", _ev1("gelu16", _bits_of(10.0)), _bits_of(10.0))
    ck.eq("gelu16(12.3) = 12.3", _ev1("gelu16", _bits_of(12.3)), _bits_of(12.3))
    ck.eq("gelu16(-10) = +0", _ev1("gelu16", _bits_of(-10.0)), 0)
    ck.eq("gelu16(1) = F16(0.8412)", _ev1("gelu16", ONE), _bits_of(np.float16(0.8411920)))
    ck.eq("geglu16(-10, 3) = +0", _ev1("geglu16", _bits_of(-10.0), THREE), 0)
    ck.eq("geglu16(NaN, 3) = NaN", _ev1("geglu16", NNAN, THREE), NAN)
    ck.eq("expsub(-inf, -inf) = +0", _ev1("expsub", NINF, NINF), 0)
    ck.eq("expsub(1, 1) = 1", _ev1("expsub", ONE, ONE), ONE)
    ck.eq("expsub(+inf, +inf) = NaN", _ev1("expsub", PINF, PINF), NAN)
    # fp16 helpers
    ck.eq("F16(65520) = inf", _ev1("f32_to_f16_rne", _bits_of(65520.0)), 0x7C00)
    ck.eq("F16(65519) = 65504", _ev1("f32_to_f16_rne", _bits_of(65519.0)), 0x7BFF)
    ck.eq("F16(2^-24) = 0x0001", _ev1("f32_to_f16_rne", 0x33800000), 0x0001)
    ck.eq("F16(2^-25) = 0 (tie to even)", _ev1("f32_to_f16_rne", 0x33000000), 0x0000)
    ck.eq("F16(-1.5 2^-25) = 0x8001", _ev1("f32_to_f16_rne", 0xB3400000), 0x8001)
    ck.eq("F16(NaN) = sign | 0x7E00", _ev1("f32_to_f16_rne", NNAN), 0xFE00)
    ck.eq("F32(0x0001) = 2^-24", _ev1("f16_to_f32", 0x0001), 0x33800000)
    ck.eq("F32(0x83FF)", _ev1("f16_to_f32", 0x83FF), _bits_of(-1023 * 2.0 ** -24))
    ck.eq("F32(0x7C01) quiet NaN", _ev1("f16_to_f32", 0x7C01), 0x7FC02000)
    # SIN / COS
    ck.eq("sin(+0) = +0", _ev1("sin", PZ), PZ)
    ck.eq("sin(-0) = -0", _ev1("sin", NZ), NZ)
    ck.eq("cos(+0) = 1", _ev1("cos", PZ), ONE)
    ck.eq("sin(-denormal) = -0", _ev1("sin", NDEN), NZ)
    ck.eq("sin(2^-13) = 2^-13", _ev1("sin", 0x39000000), 0x39000000)
    ck.eq("cos(2^-13) = 1", _ev1("cos", 0x39000000), ONE)
    for x in (NAN, NNAN, PINF, NINF):
        ck.eq(f"sin(0x{x:08X}) = NaN", _ev1("sin", x), NAN)
        ck.eq(f"cos(0x{x:08X}) = NaN", _ev1("cos", x), NAN)
    si = evaluate("sincos", np.array([ONE], dtype=np.uint32))
    ck.true("sincos interleaves (sin, cos)",
            int(si[0]) == _ev1("sin", ONE) and int(si[1]) == _ev1("cos", ONE))
    ck.eq("sin(pi/2 f32) = 1", _ev1("sin", 0x3FC90FDB), ONE)
    ck.eq("cos(pi f32) = -1", _ev1("cos", 0x40490FDB), 0xBF800000)


def _self_test_properties(ck, rng):
    # EXPK(v, N2) = EXPK(2v, N1) whenever 2v is exact (normal v with E <= 253)
    v = rng.integers(0, 1 << 32, 1 << 20, dtype=np.uint64).astype(np.uint32)
    v = v[((v >> 23) & 0xFF) <= 253]
    v = v[((v >> 23) & 0xFF) != 0]
    with np.errstate(all="ignore"):
        two_v = (v.view(np.float32) * np.float32(2)).view(np.uint32)
    ck.true("EXPK(v, N2) = EXPK(2v, N1) on 1M random v",
            bool(np.array_equal(evaluate("expk_n2", v), evaluate("expk_n1", two_v))))
    # never -0 or subnormal; monotonic sweep
    allv = inputs(0, 0, 1 << 22, 1024)
    for kap in KAPPA_BY_INDEX:
        y = evaluate(f"expk_{kap}", allv)
        ok = not np.any(y == SIGN) and not np.any(((y & EXPM) == 0) & ((y & MANT) != 0))
        ck.true(f"expk_{kap} never -0 or subnormal (2^22 strided inputs)", ok)
    xs = np.unique(rng.uniform(-104, 89, 1 << 20).astype(np.float32))
    y = evaluate("exp", xs.view(np.uint32)).view(np.float32)
    ck.true("EXP monotonic on 1M sorted samples", bool(np.all(y[1:] >= y[:-1])))
    xs = np.unique(rng.uniform(-1240, 1240, 1 << 20).astype(np.float32))
    y = evaluate("expk_ng", xs.view(np.uint32)).view(np.float32)
    ck.true("EXPK(NG) monotonic non-increasing on 1M sorted samples", bool(np.all(y[1:] <= y[:-1])))
    s, c = sincos(allv)
    sf, cf = s.view(np.float32), c.view(np.float32)
    fin = ~(np.isnan(sf) | np.isnan(cf))
    ck.true("SIN/COS within [-1, 1] (2^22 strided inputs)",
            bool(np.all(np.abs(sf[fin]) <= 1) and np.all(np.abs(cf[fin]) <= 1)))
    # primitives against float64-then-round (correctly rounded for + - * / sqrt)
    a = rng.integers(0, 1 << 32, 1 << 20, dtype=np.uint64).astype(np.uint32)
    b = rng.integers(0, 1 << 32, 1 << 20, dtype=np.uint64).astype(np.uint32)
    with np.errstate(all="ignore"):
        ad, bd = daz(a).view(np.float32).astype(np.float64), daz(b).view(np.float32).astype(np.float64)
        for name, op in (("fadd", np.add), ("fsub", np.subtract), ("fmul", np.multiply),
                         ("fdiv", np.divide)):
            want = _fin(op(ad, bd).astype(np.float32).view(np.uint32))
            ck.true(f"{name} = f64 op rounded once (1M random pairs)",
                    bool(np.array_equal(BINARY[name](a, b), want)))
        want = _fin(np.sqrt(ad).astype(np.float32).view(np.uint32))
        ck.true("fsqrt = f64 sqrt rounded once (1M random)", bool(np.array_equal(SQ(a), want)))


def _self_test_accuracy(ck, rng):
    """Max error against double on samples; gate = 2 × the §6.9 / §6.3 / §6.6 *model* figure."""
    N = 1 << 20

    def report(label, err, bound, unit="ulp"):
        m = float(np.max(err))
        ck.true(f"{label}: max {m:.4g} {unit} (model {bound:g}, gate {2 * bound:g})", m <= 2 * bound)

    def samp(lo, hi):
        return rng.uniform(lo, hi, N).astype(np.float32)

    with np.errstate(all="ignore"):
        for kap, K, lo, hi in (("p1", 1.0, -87.0, 88.0), ("n1", -1.0, -88.0, 87.0),
                               ("n2", -2.0, -44.0, 43.5),
                               ("ng", -2 * np.sqrt(2 / np.pi) * 0.044715, -1219.0, 1219.0)):
            x = samp(lo, hi)
            y = evaluate(f"expk_{kap}", x.view(np.uint32))
            report(f"EXPK {kap.upper()} on [{lo:g}, {hi:g}]", _ulp_err(y, np.exp(K * x.astype(np.float64))),
                   0.526)
        x = (rng.uniform(-1, 1, N) * 2.0 ** rng.integers(-40, -10, N)).astype(np.float32)
        y = evaluate("exp", x.view(np.uint32))
        report("EXPK P1 on tiny |v|", _ulp_err(y, np.exp(x.astype(np.float64))), 0.526)
        x = samp(-80, 80)
        xd = x.astype(np.float64)
        sig = 1 / (1 + np.exp(-xd))
        report("SIG on [-80, 80]", _ulp_err(evaluate("sigmoid", x.view(np.uint32)), sig), 2.0)
        report("SILU on [-80, 80]", _ulp_err(evaluate("silu", x.view(np.uint32)), xd * sig), 2.3)
        x = samp(-3, 8)
        xd = x.astype(np.float64)
        g = 0.5 * xd * (1 + np.tanh(np.sqrt(2 / np.pi) * (xd + 0.044715 * xd ** 3)))
        e = _ulp_err(evaluate("gelu_tanh", x.view(np.uint32)), g)
        report("GELU on [-3, 8]", e[np.abs(g) >= 2.0 ** -126], 11)
        x = samp(-8, 8)
        xd = x.astype(np.float64)
        g = 0.5 * xd * (1 + np.tanh(np.sqrt(2 / np.pi) * (xd + 0.044715 * xd ** 3)))
        y = evaluate("gelu_tanh", x.view(np.uint32)).view(np.float32).astype(np.float64)
        report("GELU on [-8, 8], absolute", np.abs(y - g), 5.2e-7, "abs")
        x = np.concatenate([samp(-10, 10), (rng.uniform(-1, 1, N) * 2.0 ** -6).astype(np.float32)])
        xd = x.astype(np.float64)
        y = evaluate("tanh", x.view(np.uint32))
        e = _ulp_err(y, np.tanh(xd))
        report("TANH on [-10, 10] and |x| < 2^-6", e[np.abs(xd) >= 2.0 ** -100], 9.8)
        report("TANH absolute", np.abs(y.view(np.float32).astype(np.float64) - np.tanh(xd)), 1.2e-7,
               "abs")
        x = np.concatenate([samp(-4, 4), samp(0, 2.0 ** 18)])
        xd = x.astype(np.float64)
        s, c = sincos(x.view(np.uint32))
        for name, y, ref in (("SIN", s, np.sin(xd)), ("COS", c, np.cos(xd))):
            yd = y.view(np.float32).astype(np.float64)
            report(f"{name} absolute on [-4, 4] and [0, 2^18]", np.abs(yd - ref), 7.5e-8, "abs")
            big = np.abs(ref) > 1e-3
            report(f"{name} where |y| > 1e-3", _ulp_err(y[big], ref[big]), 1.47)
        # beyond 2^18 the reduction error grows as θ·2^-65 quadrants (C64's rounding); this
        # exercises the sh < 62 path of step 3 up to θ = 2^40
        x = (2.0 ** rng.uniform(18, 40, N)).astype(np.float32)
        xd = x.astype(np.float64)
        s, c = sincos(x.view(np.uint32))
        for name, y, ref in (("SIN", s, np.sin(xd)), ("COS", c, np.cos(xd))):
            err = np.abs(y.view(np.float32).astype(np.float64) - ref) - xd * 2.0 ** -64
            report(f"{name} absolute on [2^18, 2^40] (minus θ 2^-64)", err, 7.5e-8, "abs")
        # GELU16 on every fp16 input against F16(double GELU(x)): < 1 %, each <= 1 fp16 ulp. (The
        # CPU table differs from GELU16 through its own f32 arithmetic, libm tanhf and the
        # cancellation for x << 0, by up to 2 fp16 ulp: VU §6.5, ref_math_gelu16.)
        h = np.arange(1 << 16, dtype=np.uint32)
        h = h[((h >> 10) & 0x1F) != 0x1F]
        x = f16_to_f32(h)
        xd = x.view(np.float32).astype(np.float64)
        g = 0.5 * xd * (1 + np.tanh(np.sqrt(2 / np.pi) * (xd + 0.044715 * xd ** 3)))
        g = np.where(xd <= -10, 0.0, np.where(xd >= 10, xd, g))
        want = f32_to_f16_rne(g.astype(np.float32).view(np.uint32))
        got = f32_to_f16_rne(gelu16(x))
        diff = got != want
        far = np.abs(got.astype(np.int64) - want.astype(np.int64))[diff]
        far = far[(got[diff] ^ want[diff]) & 0x8000 == 0]
        rate = float(np.mean(diff))
        ck.true(f"GELU16 vs F16(double GELU) over {h.size} fp16 inputs: {np.count_nonzero(diff)} "
                f"differ ({100 * rate:.3f} %), max {int(far.max()) if far.size else 0} fp16 ulp",
                rate < 0.01 and (far.size == 0 or int(far.max()) <= 1))


def self_test():
    ck = _Checker()
    rng = np.random.default_rng(20261030)
    print("constants (Appendix A against mpmath):")
    _self_test_constants(ck)
    print("special values:")
    _self_test_specials(ck)
    print("integer paths against scalar Python-int transcriptions:")
    _self_test_scalar(ck, rng)
    print("properties:")
    _self_test_properties(ck, rng)
    print("accuracy against double (samples):")
    _self_test_accuracy(ck, rng)
    print(f"ref_math_np self-test: {ck.count - ck.fails}/{ck.count} passed")
    return 1 if ck.fails else 0


def main(argv):
    check_environment()
    if len(argv) == 2 and argv[1] == "--self-test":
        return self_test()
    if len(argv) == 2 and argv[1] == "--list":
        print("unary: " + " ".join(UNARY))
        print("binary: " + " ".join(BINARY) + "   (expk: b = κ index 0 P1, 1 N1, 2 N2, 3 NG)")
        return 0
    if len(argv) not in (5, 6):
        sys.stderr.write(__doc__)
        return 2
    func = argv[1]
    start = int(argv[2], 16) & 0xFFFFFFFF
    count = int(argv[3], 0)
    stride = int(argv[4], 0) & 0xFFFFFFFF
    b = int(argv[5], 16) if len(argv) == 6 else None
    if count < 0:
        sys.exit("ref_math_np: count must be >= 0")
    try:
        evaluate(func, np.zeros(1, dtype=np.uint32), b)     # validate the name and operands
    except ValueError as e:
        sys.exit(f"ref_math_np: {e}")
    try:
        run(func, start, count, stride, b)
    except BrokenPipeError:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
