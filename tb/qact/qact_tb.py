"""Shared pieces of the tsr_qact testbenches (tb/qact/test_qact.py).

  - activation vectors: per-block generators for every special case of spec 구현 명확화 (zeros,
    subnormals, NaN, Inf, rounding ties, first-largest ties placed across lines and merge-tree
    branches, tiny max -> iscale overflow / saturation, d FTZ, fp16(d) overflow / subnormal,
    divider FTZ / overflow boundaries, raw random bit patterns, ...) and random mixes of them;
  - SpmModel: a behavioural SPM client port (tsr_spm contract, docs/rtl-m2.md §5): read data
    exactly SPM_RD_LAT cycles after the accepting edge without backpressure, random (or bursty)
    rd_req_ready / wr_ready, garbage on rd_rsp_data in cycles without a response, R2 checks of
    the unit's read and write requests, a log of every read / write / done;
  - QactDriver: the dispatch protocol (cmd_valid held until cmd_ready, garbage cmd afterwards),
    and the per-command checks:
      * every destination image line written exactly once, nothing else written, full byte
        enables, contents == tsr_ref.qact() of each token (libtsr_ref.so, bit-exact);
      * every source line read exactly once, in order, nothing else read;
      * done is a single-cycle pulse in the cycle right after the edge that accepted the last
        write; cmd_ready is low from the accept to done and high otherwise; no SPM traffic
        between commands.
Timing discipline (tb/common): drive right after a rising edge, sample at the falling edge.
"""
from __future__ import annotations

import collections
import random
import struct

import numpy as np
from cocotb.triggers import FallingEdge, RisingEdge

from common import isa
from common.tsr_ref import QA_A32, QA_A256, ref

LINE_BYTES = 64
LINE_MASK = (1 << 512) - 1
BE_ALL = (1 << 64) - 1

F32_INF = 0x7F800000
F32_NINF = 0xFF800000
F32_MAX = 0x7F7FFFFF
F32_MIN_NORM = 0x00800000


# =============================================================================================
# f32 helpers
# =============================================================================================
def f32_of(vals) -> list:
    """Python floats -> f32 bit patterns (RNE; overflow -> +-Inf)."""
    a = np.asarray(vals, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        b = a.astype(np.float32)
    return [int(v) for v in b.view(np.uint32)]


def f32_one(v: float) -> int:
    return f32_of([v])[0]


def bits_val(u: int) -> float:
    return struct.unpack("<f", struct.pack("<I", u & 0xFFFFFFFF))[0]


def neighbours(u: int, r: random.Random, span: int = 4) -> int:
    """A random f32 pattern within +-span ulps of u (same sign, finite magnitude)."""
    s = u & 0x80000000
    m = (u & 0x7FFFFFFF) + r.randint(-span, span)
    m = max(1, min(m, F32_MAX))
    return s | m


def block_size(kind: int) -> int:
    return 256 if kind == QA_A256 else 32


# =============================================================================================
# Block generators: each returns `bs` f32 bit patterns
# =============================================================================================
def _gauss(r, bs, e_lo=-20, e_hi=20):
    sc = 2.0 ** r.randint(e_lo, e_hi)
    return f32_of([r.gauss(0.0, 1.0) * sc for _ in range(bs)])


def blk_normal(r, bs, kind):
    return _gauss(r, bs)


def blk_wide(r, bs, kind):
    """Elements with independent random exponents over most of the normal range."""
    return f32_of([(-1.0) ** r.getrandbits(1) * r.random() * 2.0 ** r.randint(-120, 120)
                   for _ in range(bs)])


def blk_bits(r, bs, kind):
    return [r.getrandbits(32) for _ in range(bs)]


def blk_zero(r, bs, kind):
    return [r.choice((0, 0x80000000)) for _ in range(bs)]


def blk_subnormal(r, bs, kind):
    """Zeros and subnormals only: amax = 0 after DAZ (A256: d = +0, q = 0, BS = 0)."""
    return [(r.getrandbits(1) << 31) | (r.randint(1, 0x7FFFFF) if r.random() < 0.7 else 0)
            for _ in range(bs)]


def _nan(r):
    return (r.getrandbits(1) << 31) | 0x7F800000 | r.randint(1, 0x7FFFFF)


def blk_nan(r, bs, kind):
    """Only NaNs (quiet / signalling, both signs, random payloads), some zeros / subnormals."""
    out = []
    for _ in range(bs):
        p = r.random()
        out.append(_nan(r) if p < 0.8 else (r.getrandbits(1) << 31) | r.randint(0, 0x7FFFFF))
    return out


def blk_nan_mixed(r, bs, kind):
    x = _gauss(r, bs)
    for _ in range(r.randint(1, max(1, bs // 4))):
        x[r.randrange(bs)] = _nan(r)
    if r.random() < 0.3:                 # a NaN with a larger "magnitude" than any number
        x[r.randrange(bs)] = 0x7FFFFFFF
    return x


def blk_inf(r, bs, kind):
    x = _gauss(r, bs)
    for _ in range(r.randint(1, 3)):
        x[r.randrange(bs)] = r.choice((F32_INF, F32_NINF))
    if r.random() < 0.3:
        x[r.randrange(bs)] = _nan(r)
    return x


def blk_ties(r, bs, kind):
    """Exact .5 products: A32 amax = 127 -> d = id = 1; A256 max = -127 -> iscale = 1 (or +127 ->
    iscale = -1). q = round(x) with the kind's tie rule."""
    vals = [float(r.randint(-120, 119)) + 0.5 for _ in range(bs)]
    if r.random() < 0.3:
        vals = [float(r.randint(-127, 127)) + r.choice((0.5, 0.25, 0.75, 0.0)) for _ in range(bs)]
        vals = [max(-126.5, min(126.5, v)) for v in vals]
    p = r.randrange(bs)
    vals[p] = r.choice((127.0, -127.0))
    return f32_of(vals)


def blk_maxtie(r, bs, kind):
    """Several elements share the largest magnitude with mixed signs; the FIRST one decides the
    sign of max (A256). Positions are spread over lines and merge-tree branches."""
    x = f32_of([r.uniform(-1.0, 1.0) for _ in range(bs)])
    m = f32_one(r.choice((4.0, 1.5, 3.0e5, 7.0e-30)))
    special = [(0, bs - 1), (15, 16), (bs // 2 - 1, bs // 2), (7, 8), (1, 0)]
    if r.random() < 0.5:
        pos = list(r.choice(special))
    else:
        pos = r.sample(range(bs), r.randint(2, 5))
    for p in pos:
        x[p] = m | (r.getrandbits(1) << 31)
    return x


def blk_onehot(r, bs, kind):
    x = [0] * bs
    x[r.randrange(bs)] = f32_one(r.choice((1.0, -1.0, 3.5, -1e-3, 6e4))) if r.random() < 0.7 \
        else r.getrandbits(32) & 0xBFFFFFFF
    return x


def blk_small_ints(r, bs, kind):
    return f32_of([float(r.randint(-127, 127)) for _ in range(bs)])


def blk_tiny(r, bs, kind):
    """Tiny normals (2^-126 .. 2^-110) with zeros and subnormals: A32 d = amax/127 subnormal ->
    FTZ -> d = 0, q = 0; A256 iscale = -127/max may overflow to -+Inf -> saturated q."""
    out = []
    for _ in range(bs):
        p = r.random()
        if p < 0.4:
            out.append((r.getrandbits(1) << 31) | (r.randint(1, 16) << 23) | r.getrandbits(23))
        elif p < 0.7:
            out.append((r.getrandbits(1) << 31) | r.randint(0, 0x7FFFFF))
        else:
            out.append(r.choice((0, 0x80000000)))
    return out


def blk_huge(r, bs, kind):
    """Exponents 200..254 incl. +-FLT_MAX (A32: fp16(d) overflows to Inf)."""
    x = [(r.getrandbits(1) << 31) | (r.randint(200, 254) << 23) | r.getrandbits(23) for _ in range(bs)]
    if r.random() < 0.5:
        x[r.randrange(bs)] = F32_MAX | (r.getrandbits(1) << 31)
    return x


def _boundary_amax(r, kind) -> int:
    """|max| values around the thresholds where the divides / fp16 conversion change behaviour."""
    if kind == QA_A32:
        targets = [
            127.0 * 2.0 ** -126,            # d = amax / 127 at the FTZ boundary
            127.0 * (2.0 ** -126 - 2.0 ** -150),
            127.0 * 65504.0, 127.0 * 65520.0,   # fp16(d) max finite / overflow
            127.0 * 2.0 ** -14, 127.0 * 2.0 ** -24, 127.0 * 2.0 ** -25,  # fp16 normal / subnormal / zero
            127.0 * 2.0 ** 126, 127.0 * 2.0 ** 127,  # id = 1/d at the FTZ boundary
            127.0, 1.0, 3.0e38,
        ]
    else:
        targets = [
            127.0 / 3.4028234663852886e38,  # -127/max overflows below this
            127.0 * 2.0 ** -128, 127.0 * 2.0 ** -126,
            127.0 * 2.0 ** 126, 127.0 * 2.0 ** 127,  # iscale = -127/max at the FTZ boundary
            2.0 ** -126, 2.0 ** 127, 127.0, 3.0e38,
        ]
    t = r.choice(targets)
    return neighbours(f32_one(t), r, 6)


def blk_boundary(r, bs, kind):
    m = _boundary_amax(r, kind)
    mv = bits_val(m)
    x = f32_of([r.uniform(-1.0, 1.0) * mv for _ in range(bs)])
    x[r.randrange(bs)] = m | (r.getrandbits(1) << 31)
    if r.random() < 0.3:      # elements whose products sit just below / above .5 steps
        for _ in range(4):
            x[r.randrange(bs)] = neighbours(f32_one(mv * (r.randint(1, 253) + 0.5) / 254.0), r, 2)
    return x


def blk_daz(r, bs, kind):
    """spec 구현 명확화 DAZ vectors: a small normal max next to the largest subnormals (products
    of ~0.99 without DAZ), as in sw/sim/tests/test_ref_ggml.c."""
    x = [0] * bs
    x[r.randrange(bs)] = 0x04000000 | (0x80000000 if (kind == QA_A256 and r.random() < 0.7) else 0)
    for _ in range(r.randint(1, 4)):
        x[r.randrange(bs)] = r.choice((0x007FFFFF, 0x807FFFFF, 0x00400000, 0x80300000))
    return x


GENERATORS = {
    "normal": (blk_normal, 30), "wide": (blk_wide, 8), "bits": (blk_bits, 8), "zero": (blk_zero, 3),
    "subnormal": (blk_subnormal, 3), "nan": (blk_nan, 3), "nan_mixed": (blk_nan_mixed, 4),
    "inf": (blk_inf, 4), "ties": (blk_ties, 6), "maxtie": (blk_maxtie, 6), "onehot": (blk_onehot, 3),
    "small_ints": (blk_small_ints, 4), "tiny": (blk_tiny, 4), "huge": (blk_huge, 3),
    "boundary": (blk_boundary, 8), "daz": (blk_daz, 3),
}


def gen_block(r: random.Random, kind: int, name: str | None = None) -> list:
    bs = block_size(kind)
    if name is None:
        names = list(GENERATORS)
        name = r.choices(names, weights=[GENERATORS[n][1] for n in names])[0]
    return GENERATORS[name][0](r, bs, kind)


def gen_vector(r: random.Random, kind: int, K: int, mode: str = "mix") -> list:
    """K f32 patterns, block by block. mode: 'mix' (weighted generators), 'normal' (plain
    gaussian rows), or a generator name for every block."""
    bs = block_size(kind)
    out = []
    for _ in range(K // bs):
        if mode == "mix":
            out += gen_block(r, kind)
        elif mode == "normal":
            out += _gauss(r, bs, -4, 4)
        else:
            out += gen_block(r, kind, mode)
    return out


# Directed vectors (fixed), mirroring sw/sim/tests/test_ref_ggml.c "outside the ggml domain".
def directed_vectors() -> list:
    """[(name, kind, K, [x...])] with K = one block (or two for the cross-block cases)."""
    v = []
    u = f32_one
    # A32 all-subnormal -> D = 0, q = 0.
    v.append(("a32_all_subnormal", QA_A32, 32, [0x00400000 + j for j in range(32)]))
    # A32 2^-119 next to +-largest subnormal: q = 127 0 0.
    x = [0] * 32
    x[0], x[1], x[2] = 0x04000000, 0x007FFFFF, 0x807FFFFF
    v.append(("a32_daz", QA_A32, 32, x))
    # A256 -2^-119 next to subnormals: q = -127 0 0, BS = -127 0.
    x = [0] * 256
    x[0], x[1], x[17] = u(-2.0 ** -119), 0x007FFFFF, 0x807FFFFF
    v.append(("a256_daz", QA_A256, 256, x))
    # A256 zeros + subnormals: d = +0, q = 0, BS = 0.
    x = [0] * 256
    x[5], x[9] = 0x00400000, 0x80300000
    v.append(("a256_subnormal_only", QA_A256, 256, x))
    # A32 tiny amax -> d FTZ.
    x = [0] * 32
    x[3], x[4] = u(2.0 ** -122), u(-2.0 ** -123)
    v.append(("a32_tiny_amax", QA_A32, 32, x))
    # A32 NaN ignored; Inf amax -> D = Inf, q = 0.
    x = f32_of([float(j - 16) for j in range(32)])
    x[5] = 0x7FC00000
    v.append(("a32_nan_ignored", QA_A32, 32, list(x)))
    x[5] = F32_INF
    v.append(("a32_inf_amax", QA_A32, 32, list(x)))
    # A256 all NaN; Inf max -> d = -Inf, q = 0; tiny max -> iscale = -Inf, saturated q, d = -0.
    v.append(("a256_all_nan", QA_A256, 256, [0xFFC00000] * 256))
    x = f32_of([1.0] * 256)
    x[7] = F32_INF
    v.append(("a256_inf_max", QA_A256, 256, x))
    x = [0] * 256
    x[0], x[1] = u(2.0 ** -125), u(-2.0 ** -126)
    v.append(("a256_tiny_max", QA_A256, 256, x))
    # First-largest ties (A256), both orders.
    x = f32_of([1.0] * 256)
    x[10], x[20] = u(-4.0), u(4.0)
    v.append(("a256_first_max_neg", QA_A256, 256, list(x)))
    x[10], x[20] = u(4.0), u(-4.0)
    v.append(("a256_first_max_pos", QA_A256, 256, list(x)))
    # Rounding ties: A32 x0 = 127 (d = 1), A256 x0 = -127 (iscale = 1); x = m + 0.5.
    for kind, x0 in ((QA_A32, 127.0), (QA_A256, -127.0)):
        n = block_size(kind)
        vals = [x0] + [float(j % 200 - 100) + 0.5 for j in range(1, n)]
        v.append(("ties_%d" % kind, kind, n, f32_of(vals)))
    # All zero rows of both kinds (two blocks: the zero state must not leak into the next block).
    v.append(("a32_zero_then_normal", QA_A32, 64, [0] * 32 + f32_of([float(j) for j in range(32)])))
    v.append(("a256_zero_then_normal", QA_A256, 512, [0x80000000] * 256 + f32_of([j / 7.0 for j in range(256)])))
    # max equal magnitude at the two ends of the block, and across the two lines of an A32 block.
    x = [0] * 256
    x[0], x[255] = u(2.5), u(-2.5)
    v.append(("a256_tie_ends", QA_A256, 256, x))
    x = [0] * 32
    x[15], x[16] = u(-9.0), u(9.0)
    v.append(("a32_tie_lines", QA_A32, 32, x))
    # +-FLT_MAX (A32: d = FLT_MAX / 127 -> fp16 Inf).
    x = f32_of([1.0] * 32)
    x[31] = F32_MAX | 0x80000000
    v.append(("a32_flt_max", QA_A32, 32, x))
    # Saturation to -128 / 127 with q of every sign in one BS group.
    x = [0] * 256
    x[0], x[1], x[2], x[3] = u(2.0 ** -125), u(-2.0 ** -125), u(2.0 ** -126), u(-3.0 * 2.0 ** -126)
    v.append(("a256_sat_bs", QA_A256, 256, x))
    return v


# =============================================================================================
# Reference image / expected lines
# =============================================================================================
def ref_image(kind: int, x: list, K: int) -> bytes:
    return ref().qact(kind, x, K)


def describe_mismatch(kind: int, K: int, x: list, got: bytes, want: bytes) -> str:
    """First differing element of a QA image, with region and block context."""
    R = ref()
    qs_n = R.qa_qs_lines(kind, K) * 64
    d_n = R.qa_d_lines(kind, K) * 64
    bs = block_size(kind)
    for i in range(len(want)):
        if got[i] != want[i]:
            break
    else:
        return "no mismatch"
    if i < qs_n:
        if i >= K:
            return "QS tail byte %d: got %02x want 00" % (i, got[i])
        b = i // bs
        blk = x[b * bs:(b + 1) * bs]
        return ("QS q[%d] (block %d): got %d want %d; x = %08x; block d = %08x; block head %s" % (
            i, b, struct.unpack("b", got[i:i + 1])[0], struct.unpack("b", want[i:i + 1])[0], x[i],
            struct.unpack_from("<I", want, qs_n + 4 * b)[0], " ".join("%08x" % v for v in blk[:8])))
    if i < qs_n + d_n:
        j = (i - qs_n) // 4
        o = qs_n + 4 * j
        if j >= K // bs:
            return "D tail word %d: got %08x want 0" % (j, struct.unpack_from("<I", got, o)[0])
        blk = x[j * bs:(j + 1) * bs]
        return "D[%d]: got %08x want %08x; block %s" % (
            j, struct.unpack_from("<I", got, o)[0], struct.unpack_from("<I", want, o)[0],
            " ".join("%08x" % v for v in blk))
    j = (i - qs_n - d_n) // 2
    o = qs_n + d_n + 2 * j
    return "BS[%d]: got %d want %d" % (j, struct.unpack_from("<h", got, o)[0], struct.unpack_from("<h", want, o)[0])


def lines_of(xs: list) -> list:
    """f32 patterns (multiple of 16) -> list of 512-bit line ints."""
    b = struct.pack("<%dI" % len(xs), *[v & 0xFFFFFFFF for v in xs])
    return [int.from_bytes(b[64 * i:64 * i + 64], "little") for i in range(len(xs) // 16)]


# =============================================================================================
# Functional coverage (classified from the inputs and the reference image)
# =============================================================================================
def _daz_mag(u: int) -> int:
    """|DAZ(x)| bits, or -1 for NaN (excluded from the amax search)."""
    m = u & 0x7FFFFFFF
    if m > F32_INF:
        return -1
    return 0 if (m >> 23) == 0 else m


class Coverage:
    """Counts blocks per special case, so a run can show which cases it actually exercised."""

    def __init__(self):
        self.c = collections.Counter()

    def add(self, kind: int, K: int, x: list, img: bytes):
        R = ref()
        bs = block_size(kind)
        qs_n = R.qa_qs_lines(kind, K) * 64
        a = np.asarray(x, dtype=np.uint32)
        for b in range(K // bs):
            xb = [int(v) for v in a[b * bs:(b + 1) * bs]]
            mags = [_daz_mag(v) for v in xb]
            amax = max(mags)
            d = struct.unpack_from("<I", img, qs_n + 4 * b)[0]
            q = struct.unpack_from("<%db" % bs, img, b * bs)
            k = "a32" if kind == QA_A32 else "a256"
            self.c[k + "_blocks"] += 1
            if any(m < 0 for m in mags):
                self.c[k + "_nan_in_block"] += 1
            if any(m == 0 and (v & 0x7FFFFFFF) != 0 for m, v in zip(mags, xb)):
                self.c[k + "_subnormal_in_block"] += 1
            if amax <= 0:
                self.c[k + "_amax_zero"] += 1
                continue
            if amax == F32_INF:
                self.c[k + "_amax_inf"] += 1
            if (d & 0x7FFFFFFF) == 0:
                self.c[k + "_d_zero_nonzero_amax"] += 1      # A32 FTZ of d / A256 iscale overflow
            if (d & 0x7FFFFFFF) == F32_INF:
                self.c[k + "_d_inf"] += 1
            if -128 in q:
                self.c[k + "_q_m128"] += 1
            if 127 in q or -127 in q:
                self.c[k + "_q_pm127"] += 1
            ties = [i for i, m in enumerate(mags) if m == amax]
            if len(ties) > 1 and len({xb[i] >> 31 for i in ties}) > 1:
                self.c[k + "_max_tie_mixed_sign"] += 1
            if kind == QA_A32:
                if amax != F32_INF and (d & 0x7FFFFFFF) == F32_INF:
                    self.c["a32_fp16_overflow"] += 1
                if 0 < (d & 0x7FFFFFFF) < 0x38800000:        # D below 2^-14: fp16 subnormal d
                    self.c["a32_fp16_subnormal"] += 1
                dd = R.fdiv(amax, 0x42FE0000)
                scale = 0 if (dd & 0x7FFFFFFF) == 0 else R.frecip(dd)
            else:
                mx = xb[ties[0]] if (xb[ties[0]] >> 23) & 0xFF else xb[ties[0]] & 0x80000000
                if mx >> 31:
                    self.c["a256_neg_max"] += 1
                scale = R.fdiv(0xC2FE0000, mx)
            with np.errstate(all="ignore"):
                prod = np.asarray(xb, dtype=np.uint32).view(np.float32) * \
                    np.asarray([scale], dtype=np.uint32).view(np.float32)[0]
                fr = np.abs(prod) - np.floor(np.abs(prod))
            if np.any(fr == 0.5):
                self.c[k + "_round_tie"] += 1

    def report(self) -> str:
        return ", ".join("%s %d" % kv for kv in sorted(self.c.items()))

    def require(self, keys):
        miss = [k for k in keys if self.c[k] == 0]
        assert not miss, "coverage holes: %s (have: %s)" % (miss, self.report())


COVER_KEYS = ["a32_nan_in_block", "a32_subnormal_in_block", "a32_amax_zero", "a32_amax_inf",
              "a32_d_zero_nonzero_amax", "a32_d_inf", "a32_q_pm127", "a32_max_tie_mixed_sign",
              "a32_fp16_overflow", "a32_fp16_subnormal", "a32_round_tie",
              "a256_nan_in_block", "a256_subnormal_in_block", "a256_amax_zero", "a256_amax_inf",
              "a256_d_zero_nonzero_amax", "a256_d_inf", "a256_q_m128", "a256_q_pm127",
              "a256_max_tie_mixed_sign", "a256_neg_max", "a256_round_tie"]


# =============================================================================================
# Ready policies
# =============================================================================================
class Ready:
    """Per-cycle ready decisions: 'always', 'random' (probability p), 'bursty' (stall runs)."""

    def __init__(self, rnd: random.Random, mode: str = "always", p: float = 0.7,
                 stall=(1, 12), p_stall: float = 0.04):
        self.rnd, self.mode, self.p, self.stall, self.p_stall = rnd, mode, p, stall, p_stall
        self.left = 0

    def __call__(self) -> int:
        if self.mode == "always":
            return 1
        if self.mode == "random":
            return int(self.rnd.random() < self.p)
        if self.left > 0:
            self.left -= 1
            return 0
        if self.rnd.random() < self.p_stall:
            self.left = self.rnd.randint(*self.stall)
            return 0
        return 1


# =============================================================================================
# Behavioural SPM client port
# =============================================================================================
class SpmModel:
    def __init__(self, dut, *, rd_lat: int, rnd: random.Random, lines: int):
        self.dut, self.rd_lat, self.rnd, self.lines = dut, rd_lat, rnd, lines
        self.mem = {}                 # line -> 512-bit int
        self.cycle = 0
        self.reads = []               # (cycle of the accepting edge, line)
        self.writes = []              # (cycle, line)
        self.done_cycles = []         # cycles in which done was high (falling-edge samples)
        self.ready_low_while_busy = 0
        self.cmd_ready = []           # (cycle, cmd_ready)
        self.rd_ready = Ready(rnd)
        self.wr_ready = Ready(rnd)
        self.garbage = True
        self.errors = []
        self.rsp = collections.deque()

    def err(self, msg):
        self.errors.append("cycle %d: %s" % (self.cycle, msg))
        raise AssertionError(self.errors[-1])

    async def run(self):
        dut = self.dut
        dut.spm_rd_req_ready.value = 0
        dut.spm_wr_ready.value = 0
        dut.spm_rd_rsp_valid.value = 0
        dut.spm_rd_rsp_data.value = 0
        rd_rdy = wr_rdy = 0
        held_rd = held_wr = None
        while True:
            await FallingEdge(dut.clk)
            if not int(dut.rst_n.value):
                # Reset (synchronous): the unit drops its requests, tsr_spm its response pipeline.
                held_rd = held_wr = None
                self.rsp.clear()
            rv = int(dut.spm_rd_req_valid.value)
            wv = int(dut.spm_wr_valid.value)
            rl = int(dut.spm_rd_req_line.value) if rv else None
            wl = wd = wbe = None
            if wv:
                wl = int(dut.spm_wr_line.value)
                wd = int(dut.spm_wr_data.value)
                wbe = int(dut.spm_wr_be.value)
            if held_rd is not None and (not rv or rl != held_rd):
                self.err("R2: read request dropped or changed before acceptance (%s -> %s)" % (held_rd, rl if rv else None))
            if held_wr is not None and (not wv or (wl, wd, wbe) != held_wr):
                self.err("R2: write request dropped or changed before acceptance (line %d)" % held_wr[0])
            if rv and rl >= self.lines:
                self.err("read of line %d >= SPM lines %d" % (rl, self.lines))
            if wv and wl >= self.lines:
                self.err("write of line %d >= SPM lines %d" % (wl, self.lines))
            rd_x = bool(rv and rd_rdy)
            wr_x = bool(wv and wr_rdy)
            held_rd = rl if (rv and not rd_rdy) else None
            held_wr = (wl, wd, wbe) if (wv and not wr_rdy) else None
            if int(self.dut.done.value):
                self.done_cycles.append(self.cycle)
            self.cmd_ready.append((self.cycle, int(self.dut.cmd_ready.value)))
            await RisingEdge(dut.clk)
            self.cycle += 1
            if rd_x:
                if rl not in self.mem:
                    self.err("read of uninitialised SPM line %d" % rl)
                self.rsp.append((self.cycle + self.rd_lat - 1, self.mem[rl]))
                self.reads.append((self.cycle, rl))
            if wr_x:
                if wbe != BE_ALL:
                    self.err("write of line %d with byte enables %x" % (wl, wbe))
                self.mem[wl] = wd
                self.writes.append((self.cycle, wl))
            if self.rsp and self.rsp[0][0] == self.cycle:
                _, d = self.rsp.popleft()
                dut.spm_rd_rsp_valid.value = 1
                dut.spm_rd_rsp_data.value = d
            else:
                if self.rsp and self.rsp[0][0] < self.cycle:
                    self.err("internal: missed a response slot")
                dut.spm_rd_rsp_valid.value = 0
                if self.garbage and self.rnd.random() < 0.3:
                    dut.spm_rd_rsp_data.value = self.rnd.getrandbits(512)
            rd_rdy = self.rd_ready()
            wr_rdy = self.wr_ready()
            dut.spm_rd_req_ready.value = rd_rdy
            dut.spm_wr_ready.value = wr_rdy


# =============================================================================================
# Dispatch driver + checks
# =============================================================================================
class QactDriver:
    def __init__(self, dut, spm: SpmModel, rnd: random.Random, cfg: dict):
        self.dut, self.spm, self.rnd, self.cfg = dut, spm, rnd, cfg
        self.n_cmds = 0
        self.stats = []               # (kind, K, T, cycles accept->done)
        self.cov = Coverage()
        self.collect = True
        dut.cmd_valid.value = 0
        dut.cmd.value = 0

    def place(self, kind: int, K: int, T: int, gap: int = 0, at_end: str | None = None,
              avoid=()):
        """Random (src, dst) placement in the SPM: src and dst do not overlap each other nor any
        (lo, hi) line range in `avoid` (hi exclusive). Returns (src, dst, stride)."""
        R = ref()
        lines = self.cfg["LINES"]
        n = K // 16
        stride = n + gap
        qa = R.qa_lines(kind, K)
        src_sz = (T - 1) * stride + n
        dst_sz = T * qa
        if src_sz + dst_sz > lines:
            raise ValueError("does not fit")

        def free(lo, hi):
            return all(hi <= a or b <= lo for a, b in avoid)

        for _ in range(2000):
            src = lines - src_sz if at_end == "src" else self.rnd.randrange(0, lines - src_sz + 1)
            if at_end == "dst":
                dst = lines - dst_sz
            elif at_end == "dst0":
                dst = 0
            else:
                dst = self.rnd.randrange(0, lines - dst_sz + 1)
            if (dst + dst_sz <= src or src + src_sz <= dst) and free(src, src + src_sz) and \
                    free(dst, dst + dst_sz):
                return src, dst, stride
        raise ValueError("no placement")

    @staticmethod
    def regions(ctx) -> list:
        return [(ctx["src"], ctx["src"] + (ctx["T"] - 1) * ctx["stride"] + ctx["K"] // 16),
                (ctx["dst"], ctx["dst"] + ctx["T"] * ctx["qa"])]

    async def start(self, kind: int, K: int, T: int, xs: list, src: int, dst: int, stride: int,
                    stride_field: int | None = None) -> dict:
        """Load the source, garbage the destination, offer the command right after a rising edge
        and hold it until accepted. Returns the command context (t_acc = the cycle that starts at
        the accepting edge)."""
        dut, spm, R = self.dut, self.spm, ref()
        qa = R.qa_lines(kind, K)
        src_lines = []
        for t in range(T):
            for i, ln in enumerate(lines_of(xs[t])):
                spm.mem[src + t * stride + i] = ln
                src_lines.append(src + t * stride + i)
        for ln in range(dst, dst + T * qa):
            spm.mem[ln] = self.rnd.getrandbits(512)   # every byte of the image must be written
        sf = stride if stride_field is None else stride_field
        cmd = isa.QACT_CMD_T.pack(kind=kind, dst_line=dst, src_line=src, k=K, t=T,
                                  src_stride=sf & 0xFFFFFFFF, qa_lines=qa)
        await RisingEdge(dut.clk)
        dut.cmd.value = cmd
        dut.cmd_valid.value = 1
        while True:
            await FallingEdge(dut.clk)
            if int(dut.cmd_ready.value):
                break
        t_acc = spm.cycle + 1         # spm.cycle only changes right after rising edges
        await RisingEdge(dut.clk)
        dut.cmd_valid.value = 0
        dut.cmd.value = self.rnd.getrandbits(isa.QACT_CMD_T.width)
        self.n_cmds += 1
        return dict(kind=kind, K=K, T=T, xs=xs, src=src, dst=dst, stride=stride, qa=qa,
                    src_lines=src_lines, t_acc=t_acc, n=self.n_cmds)

    async def finish(self, ctx: dict, timeout: int | None = None, check_timing: bool = True) -> int:
        """Wait for the command's done and check everything it did. Returns accept->done cycles."""
        dut, spm = self.dut, self.spm
        kind, K, T, xs, src, dst, stride, qa = (ctx[k] for k in ("kind", "K", "T", "xs", "src", "dst",
                                                                  "stride", "qa"))
        t_acc = ctx["t_acc"]
        n = K // 16
        limit = timeout or (300 + T * (n * 40 + 64 * qa) * 4)
        tag = "QACT #%d kind %d K %d T %d src %d dst %d stride %d" % (ctx["n"], kind, K, T, src, dst, stride)

        def dones():
            return [c for c in spm.done_cycles if c >= t_acc]

        while not dones():
            await FallingEdge(dut.clk)
            if spm.cycle - t_acc > limit:
                raise AssertionError("%s: no done within %d cycles" % (tag, limit))
        t_done = dones()[0]
        await FallingEdge(dut.clk)    # one more sample: done must be a single-cycle pulse
        assert [c for c in dones() if c <= t_done + 1] == [t_done], \
            "%s: done not a single pulse: %s" % (tag, dones()[:4])
        self.stats.append((kind, K, T, t_done - t_acc))
        reads = [ln for c, ln in spm.reads if t_acc <= c <= t_done]
        writes = [(c, ln) for c, ln in spm.writes if t_acc <= c <= t_done]
        assert writes, "%s: no writes" % tag
        if check_timing:
            # done is high in the cycle that starts at the edge accepting the last write.
            assert t_done == writes[-1][0], "%s: done in cycle %d, last write accepted at %d" % (
                tag, t_done, writes[-1][0])
        busy = [c for c, v in spm.cmd_ready if t_acc <= c < t_done and v]
        assert not busy, "%s: cmd_ready high while busy (cycles %s)" % (tag, busy[:5])
        # Reads: every source line once, in order, nothing else.
        assert reads == ctx["src_lines"], "%s: read sequence differs (first reads %s, expected %s)" % (
            tag, reads[:8], ctx["src_lines"][:8])
        # Writes: every destination line exactly once, nothing else.
        dst_lines = list(range(dst, dst + T * qa))
        wl = [ln for _, ln in writes]
        assert sorted(wl) == dst_lines, "%s: written lines differ: extra %s missing %s dup %s" % (
            tag, sorted(set(wl) - set(dst_lines))[:8], sorted(set(dst_lines) - set(wl))[:8],
            sorted(x for x, c in collections.Counter(wl).items() if c > 1)[:8])
        for t in range(T):
            want = ref_image(kind, xs[t], K)
            got = b"".join(spm.mem[dst + t * qa + i].to_bytes(64, "little") for i in range(qa))
            if got != want:
                raise AssertionError("%s token %d: %s" % (tag, t, describe_mismatch(kind, K, xs[t], got, want)))
            if self.collect:
                self.cov.add(kind, K, xs[t], want)
        for t in range(T):
            for i, ln in enumerate(lines_of(xs[t])):
                assert spm.mem[src + t * stride + i] == ln, "%s: source line modified" % tag
        return t_done - t_acc

    async def qact(self, kind: int, K: int, T: int, xs: list, src: int, dst: int, stride: int,
                   stride_field: int | None = None, **kw) -> int:
        ctx = await self.start(kind, K, T, xs, src, dst, stride, stride_field)
        return await self.finish(ctx, **kw)

    async def idle_check(self, cycles: int):
        """No SPM traffic and cmd_ready high while no command is offered."""
        spm = self.spm
        r0, w0 = len(spm.reads), len(spm.writes)
        c0 = spm.cycle
        for _ in range(cycles):
            await FallingEdge(self.dut.clk)
            assert int(self.dut.spm_rd_req_valid.value) == 0 and int(self.dut.spm_wr_valid.value) == 0, \
                "SPM request while idle"
            assert int(self.dut.cmd_ready.value) == 1, "cmd_ready low while idle"
            assert int(self.dut.done.value) == 0, "done while idle"
        assert len(spm.reads) == r0 and len(spm.writes) == w0 and spm.cycle >= c0
