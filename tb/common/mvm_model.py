"""Python golden model of the MVM unit's blocks (docs/rtl-mvm.md), built on libtsr_ref.

Every value here is either produced by libtsr_ref (sw/sim/tsr_ref.c, the bit-exact definition) or
derived from the device layout and cross-checked against it (Cluster.check() recomputes every
term from the tagged beat stream and compares it with tsr_term_q8_0 / tsr_term_q4_K, and every
row result with tsr_dot_row), so the model is itself verified before an RTL block is compared
with it.

Front end (§3.3)         front_beats(fmt, K, rows) -> the exact tsr_mvm_beat_t stream (dicts);
                         pack_beat(d) -> the struct as an int (isa.MVM_BEAT_T)
Back end (§7.1)          beat_math(fmt, beat, qa, K): integer intermediates (group sums, isum part,
                         msum, Dx) of one beat for one token; term_from_values(...)
Cluster                  Cluster(fmt, K, rows, qas): beats, terms[r][t], partials[r][t],
                         y[r][t]; actbuf images; check()
Activation buffer (§5)   actbuf_image(kind, K, qa) per token; bc_lines(kind, K, qas) (§3.1)
Gather (§6.3)            gather_jobs(N, num_ch, T, dst, stride); apply_mvm(spm, ...) = the
                         tsr_sim.c ex_mvm write loop; mvm_reference(...) = whole MVM semantics
Data                     rand_blocks / rand_row / rand_qa (realistic and raw, special values),
                         layout_channels (symmetric allocation), ftz_window_dx (FTZ boundary)

All f32 values are 32-bit patterns (int). Weights / activations are bytes (repacked rows exactly
as in DRAM, QA images exactly as in the SPM).
"""
from __future__ import annotations

import random
import struct
from dataclasses import dataclass, field
from fractions import Fraction

from . import isa
from .tsr_ref import ref

FMT_Q8_0, FMT_Q4_K = 0, 1
QA_A32, QA_A256 = 0, 1
M32 = 0xFFFFFFFF
F32_POS_ZERO = 0


# =============================================================================================
# Layout helpers (pure Python, cross-checked with libtsr_ref in tb/common/test_common.py)
# =============================================================================================
def block_size(fmt: int) -> int:
    return 256 if fmt == FMT_Q4_K else 32


def chunk_blocks(fmt: int) -> int:
    return 4 if fmt == FMT_Q4_K else 32


def quant_beats(fmt: int, nblk: int) -> int:
    """Quant beats of a chunk holding nblk blocks: ceil(nblk * QB / 64)."""
    return 2 * nblk if fmt == FMT_Q4_K else (nblk + 1) // 2


def chunk_beats(fmt: int, nblk: int) -> int:
    return 1 + quant_beats(fmt, nblk)


def row_chunks(fmt: int, K: int) -> list:
    """Blocks per chunk along a row: full chunks of S blocks, then the partial remainder."""
    nb, S = K // block_size(fmt), chunk_blocks(fmt)
    return [S] * (nb // S) + ([nb % S] if nb % S else [])


def row_beats(fmt: int, K: int) -> int:
    return sum(chunk_beats(fmt, n) for n in row_chunks(fmt, K))


def qa_kind(fmt: int) -> int:
    return QA_A256 if fmt == FMT_Q4_K else QA_A32


def rows_c(N: int, num_ch: int, c: int) -> int:
    """Rows of channel c: max(0, min((c+1)R, N) - cR), R = ceil(N / num_ch)."""
    R = -(-N // num_ch)
    return max(0, min(R, N - c * R))


def s8(v: int) -> int:
    return v - 256 if v >= 128 else v


def s16(v: int) -> int:
    return v - 65536 if v >= 32768 else v


def f16le(b: bytes, off: int) -> int:
    return b[off] | (b[off + 1] << 8)


def get_scale_min_k4(j: int, q: bytes):
    """ggml get_scale_min_k4 (tsr_ref.c 359-367) on the 12 scale bytes."""
    if j < 4:
        return q[j] & 63, q[j + 4] & 63
    return (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4)


# =============================================================================================
# Front end: tagged beat stream (docs/rtl-mvm.md §3.3)
# =============================================================================================
def front_beats(fmt: int, K: int, rows) -> list:
    """The tsr_mvm_beat_t stream of a cluster whose rows are `rows` (repacked row images, at
    least ROW_BEATS * 64 bytes each, any content). One dict per QUANT beat, fields as in the
    struct (sc, m, w as lists; w signed) plus the extra keys row / chunk / beat (beat index
    inside the row image, META beats included) that are not part of the struct."""
    rb = row_beats(fmt, K)
    out = []
    S = chunk_blocks(fmt)
    for r, row in enumerate(rows):
        if len(row) < rb * 64:
            raise ValueError("front_beats: row %d has %d bytes, need %d" % (r, len(row), rb * 64))
        beat0 = 0
        for c, nblk in enumerate(row_chunks(fmt, K)):
            meta = row[64 * beat0:64 * beat0 + 64]
            for j in range(quant_beats(fmt, nblk)):
                bi = beat0 + 1 + j
                q = row[64 * bi:64 * bi + 64]
                d = dict(last=0, row_last=0, row=r, chunk=c, beat=bi)
                if fmt == FMT_Q8_0:
                    two = int(2 * j + 1 < nblk)
                    d.update(blk=S * c + 2 * j, two=two, half=0,
                             d_a=ref().fp16_to_f32(f16le(meta, 4 * j)),
                             d_b=ref().fp16_to_f32(f16le(meta, 4 * j + 2)) if two else F32_POS_ZERO,
                             sc=[0] * 4, m=[0] * 8,
                             w=[s8(q[L]) for L in range(32)] + [s8(q[L]) if two else 0 for L in range(32, 64)]
                             + [0] * 64)
                else:
                    half = j % 2
                    m16 = meta[16 * (j // 2):16 * (j // 2) + 16]
                    scm = [get_scale_min_k4(k, m16[4:16]) for k in range(8)]
                    w = []
                    for g in range(4):
                        for ln in range(32):
                            byte = q[32 * (g // 2) + ln]
                            w.append(byte & 0xF if g % 2 == 0 else byte >> 4)
                    d.update(blk=S * c + j // 2, two=0, half=half,
                             d_a=ref().fp16_to_f32(f16le(m16, 0)), d_b=ref().fp16_to_f32(f16le(m16, 2)),
                             sc=[scm[4 * half + g][0] for g in range(4)], m=[scm[k][1] for k in range(8)],
                             w=w)
                out.append(d)
            beat0 += chunk_beats(fmt, nblk)
        if out:
            out[-1]["row_last"] = 1
    if out:
        out[-1]["last"] = 1
    return out


BEAT_KEYS = tuple(n for n, _ in isa.MVM_BEAT_T.fields)


def pack_beat(d: dict) -> int:
    """front_beats() dict -> tsr_mvm_beat_t as an int."""
    v = {k: d[k] for k in BEAT_KEYS if k not in ("sc", "m", "w")}
    v["sc"] = isa.pack_array(d["sc"], 6)
    v["m"] = isa.pack_array(d["m"], 6)
    v["w"] = isa.pack_array(d["w"], 8)
    return isa.MVM_BEAT_T.pack(**v)


def unpack_beat(x: int) -> dict:
    d = isa.MVM_BEAT_T.unpack(x)
    d["sc"] = isa.unpack_array(d["sc"], 4, 6)
    d["m"] = isa.unpack_array(d["m"], 8, 6)
    d["w"] = isa.unpack_array(d["w"], 128, 8, signed=True)
    return d


def beat_fields(d: dict) -> dict:
    """Only the struct fields of a front_beats() dict (for comparisons with an unpacked beat)."""
    return {k: d[k] for k in BEAT_KEYS}


# =============================================================================================
# QA image views
# =============================================================================================
@dataclass
class QaView:
    kind: int
    K: int
    img: bytes
    qs: list = field(default_factory=list)    # int8 x K
    d: list = field(default_factory=list)     # f32 bits per block
    bs: list = field(default_factory=list)    # int16 x K/16 (A256)

    @classmethod
    def of(cls, kind: int, K: int, img: bytes) -> "QaView":
        v = ref().qa_split(kind, K, bytes(img))
        return cls(kind, K, bytes(img), v["qs"], v["d"], v["bs"])


def qa_qs_lines(K: int) -> int:
    return (K + 63) // 64


def qa_d_lines(kind: int, K: int) -> int:
    return isa.qa_d_lines(kind, K)


# =============================================================================================
# Back end: per-beat arithmetic (docs/rtl-mvm.md §7.1)
# =============================================================================================
def beat_math(fmt: int, beat: dict, qa: QaView) -> dict:
    """Integer intermediates of one tagged beat for one token: S[g] (group sums of the lanes in
    use), Dx (f32 bits of the D element(s)), and for Q4_K part = sum_g sc[g]*S[g] and msum."""
    b = beat["blk"]
    w = beat["w"]
    if fmt == FMT_Q8_0:
        ng = 2 if beat["two"] else 1
        S = [sum(w[32 * g + L] * qa.qs[32 * (b + g) + L] for L in range(32)) for g in range(ng)]
        return dict(S=S, Dx=[qa.d[b + g] for g in range(ng)])
    s, h = b, beat["half"]
    base = 256 * s + 128 * h
    S = [sum(w[32 * g + L] * qa.qs[base + 32 * g + L] for L in range(32)) for g in range(4)]
    part = sum(beat["sc"][g] * S[g] for g in range(4))
    msum = sum(beat["m"][k] * (qa.bs[16 * s + 2 * k] + qa.bs[16 * s + 2 * k + 1]) for k in range(8))
    return dict(S=S, Dx=[qa.d[s]], part=part, msum=msum)


def i32(v: int) -> int:
    """Python int -> int32 bit pattern (asserts the value fits; integer sums are exact)."""
    assert -(1 << 31) <= v < (1 << 31), v
    return v & M32


def term_q8_0_values(d: int, dx: int, isum: int) -> int:
    """fmul(fmul(d, Dx), i32_to_f32(isum)) (tsr_ref.c term_q8_0_ptr)."""
    r = ref()
    return r.fmul(r.fmul(d, dx), r.i32_to_f32(i32(isum)))


def term_q4_K_values(d: int, dmin: int, dx: int, isum: int, msum: int) -> int:
    """fsub(fmul(fmul(d, Dx), f32(isum)), fmul(fmul(dmin, Dx), f32(msum))) (term_q4_K_ptr)."""
    r = ref()
    t1 = r.fmul(r.fmul(d, dx), r.i32_to_f32(i32(isum)))
    t2 = r.fmul(r.fmul(dmin, dx), r.i32_to_f32(i32(msum)))
    return r.fsub(t1, t2)


@dataclass
class Term:
    i: int          # term index inside the row (Q8_0 block, Q4_K superblock)
    partial: int    # i % 16
    lane: int       # accumulate lane = partial % 2 (docs/rtl-mvm.md §7.3)
    bits: int       # f32 term
    isum: int
    msum: int = 0   # Q4_K only


def terms_from_beats(fmt: int, beats: list, qa: QaView) -> list:
    """Per row: the list of Term computed from the tagged beat stream alone (front_beats) with
    libtsr_ref fp ops. Cluster.check() compares these with tsr_term_*, which validates
    front_beats against the reference's own row walk."""
    rows, cur, part0 = [], [], 0
    for bt in beats:
        mth = beat_math(fmt, bt, qa)
        if fmt == FMT_Q8_0:
            for g, (S, dx) in enumerate(zip(mth["S"], mth["Dx"])):
                i = bt["blk"] + g
                d = bt["d_a"] if g == 0 else bt["d_b"]
                cur.append(Term(i, i % 16, i % 2, term_q8_0_values(d, dx, S), S))
        else:
            if bt["half"] == 0:
                part0 = mth["part"]
            else:
                s, isum = bt["blk"], part0 + mth["part"]
                cur.append(Term(s, s % 16, s % 2, term_q4_K_values(bt["d_a"], bt["d_b"], mth["Dx"][0],
                                                                    isum, mth["msum"]), isum, mth["msum"]))
        if bt["row_last"]:
            rows.append(cur)
            cur = []
    return rows


def accumulate(terms: list) -> list:
    """16 partials from +0.0, term i -> p[i % 16] = fadd(p, t) in term order (tsr_dot_row)."""
    r = ref()
    p = [F32_POS_ZERO] * 16
    for t in terms:
        p[t.partial] = r.fadd(p[t.partial], t.bits)
    return p


def tree16(p: list) -> int:
    """tsr_tree16 order, computed with fadd here (the self-test compares with ref().tree16)."""
    r = ref()
    lv = list(p)
    while len(lv) > 1:
        lv = [r.fadd(lv[2 * k], lv[2 * k + 1]) for k in range(len(lv) // 2)]
    return lv[0]


# =============================================================================================
# Cluster model
# =============================================================================================
class Cluster:
    """Golden model of one cluster for one MVM: rows (repacked row images of this channel, row
    order), qas (QA images of the T tokens, SPM byte images). Attributes:
      beats          front_beats(fmt, K, rows)
      terms[r][t]    list of Term (from the reference: tsr_term_q8_0 / tsr_term_q4_K bits)
      partials[r][t] 16 partials after the row (f32 bits)
      y[r][t]        row result (tsr_dot_row)
    check() cross-validates beats / terms / partials / y against libtsr_ref (raises on mismatch)."""

    def __init__(self, fmt: int, K: int, rows, qas, *, verify: bool = True):
        self.fmt, self.K = fmt, K
        self.kind = qa_kind(fmt)
        self.rows = [bytes(r) for r in rows]
        self.qas = [bytes(q) for q in qas]
        self.T = len(self.qas)
        self.views = [QaView.of(self.kind, K, q) for q in self.qas]
        self.beats = front_beats(fmt, K, self.rows)
        r = ref()
        nb = K // block_size(fmt)
        self.terms, self.partials, self.y = [], [], []
        for row in self.rows:
            tr, pr, yr = [], [], []
            for qa in self.qas:
                ts = [Term(i, i % 16, i % 2, r.term(fmt, row, K, i, qa), 0) for i in range(nb)]
                p = accumulate(ts)
                tr.append(ts)
                pr.append(p)
                yr.append(r.dot_row(fmt, row, K, qa))
            self.terms.append(tr)
            self.partials.append(pr)
            self.y.append(yr)
        if verify:
            self.check()

    def check(self):
        r = ref()
        for t, view in enumerate(self.views):
            rows = terms_from_beats(self.fmt, self.beats, view)
            assert len(rows) == len(self.rows), "beat stream has %d rows, expected %d" % (len(rows), len(self.rows))
            for ri, ts in enumerate(rows):
                want = self.terms[ri][t]
                assert [x.i for x in ts] == [x.i for x in want], "row %d: term order" % ri
                for a, b in zip(ts, want):
                    assert a.bits == b.bits, "row %d token %d term %d: beats give 0x%08x, tsr_term 0x%08x" % (
                        ri, t, a.i, a.bits, b.bits)
                # keep the integer intermediates found through the beat stream
                for a, b in zip(ts, want):
                    b.isum, b.msum = a.isum, a.msum
                p = self.partials[ri][t]
                assert r.tree16(p) == self.y[ri][t] == tree16(p), "row %d token %d: tree / dot_row" % (ri, t)

    def results(self) -> list:
        """[(row, [y_t for t < T])] in row order (what the back end + wrapper deliver)."""
        return [(ri, list(self.y[ri])) for ri in range(len(self.rows))]

    def actbuf(self, t_max: int | None = None) -> list:
        """Per token t < max(T, t_max): actbuf_image() (tokens >= T: None)."""
        n = max(self.T, t_max or 0)
        return [actbuf_image(self.kind, self.K, self.qas[t]) if t < self.T else None for t in range(n)]


# =============================================================================================
# Activation buffer and broadcast (docs/rtl-mvm.md §3.1, §5)
# =============================================================================================
def actbuf_image(kind: int, K: int, qa: bytes) -> dict:
    """What tsr_mvm_actbuf holds for one token after the broadcast:
    qs[w] = 1024-bit int {line 2w+1, line 2w} (a missing odd line reads as None in qs_hi),
    dbs[a] = 512-bit int of QA line ceil(K/64) + a."""
    ql = qa_qs_lines(K)
    lines = [int.from_bytes(qa[64 * i:64 * i + 64], "little") for i in range(len(qa) // 64)]
    qs = []
    for w in range((ql + 1) // 2):
        lo = lines[2 * w]
        hi = lines[2 * w + 1] if 2 * w + 1 < ql else 0
        qs.append(lo | (hi << 512))
    dbs = lines[ql:]
    return dict(qs=qs, qs_lines=ql, dbs=dbs, odd_tail=(ql % 2 == 1))


def bc_lines(kind: int, K: int, qas) -> list:
    """The broadcast items (tsr_mvm_bcline_t field dicts) of an MVM with these T QA images, in
    SPM order: token 0 lines 0..QA_LINES-1, token 1, ..."""
    ql = qa_qs_lines(K)
    n = isa.qa_lines(kind, K)
    out = []
    for t, qa in enumerate(qas):
        assert len(qa) == 64 * n, "bc_lines: QA image of token %d has %d bytes" % (t, len(qa))
        for ln in range(n):
            dbs = int(ln >= ql)
            out.append(dict(last=0, tok=t, dbs=dbs, addr=ln - ql if dbs else ln,
                            data=int.from_bytes(qa[64 * ln:64 * ln + 64], "little")))
    if out:
        out[-1]["last"] = 1
    return out


def ccmd(fmt: int, K: int, N: int, num_ch: int, w_off: int) -> dict:
    """tsr_mvm_ccmd_t fields of an MVM (docs/rtl-mvm.md §3.1)."""
    return dict(fmt=fmt, k=K, n=N, rows_per_ch=-(-N // num_ch), row_beats=row_beats(fmt, K),
                w_off=w_off, d_lines=qa_d_lines(qa_kind(fmt), K))


def fcmd(fmt: int, K: int, N: int, num_ch: int, c: int, w_off: int) -> dict | None:
    """tsr_mvm_fcmd_t of cluster c (None when the channel has no rows; docs/rtl-mvm.md §3.2)."""
    rc = rows_c(N, num_ch, c)
    if rc == 0:
        return None
    rb = row_beats(fmt, K)
    return dict(fmt=fmt, k=K, row_beats=rb, w_off=w_off, rows=rc, beats=rc * rb)


# =============================================================================================
# Gather (docs/rtl-mvm.md §6.3) and the whole MVM (tsr_sim.c ex_mvm)
# =============================================================================================
@dataclass
class Job:
    ch: int
    tok: int
    line: int       # SPM line
    lo: int         # first lane of this cluster's rows in the line
    hi: int         # last lane
    tail: bool      # the line holds row N-1: lanes > hi are written as 0
    rows: tuple     # (first row, last row) of the group

    def be(self) -> int:
        """spm_wr_be of the job (64 bits)."""
        lanes = set(range(self.lo, self.hi + 1)) | (set(range(self.hi + 1, 16)) if self.tail else set())
        v = 0
        for ln in lanes:
            v |= 0xF << (4 * ln)
        return v


def gather_jobs(N: int, num_ch: int, T: int, dst: int, stride: int) -> list:
    """Every partial-line write of the gather, per cluster in row order, tokens inner."""
    jobs = []
    if N == 0:
        return jobs
    R = -(-N // num_ch)
    for c in range(num_ch):
        n0, rc = c * R, rows_c(N, num_ch, c)
        n = n0
        while n < n0 + rc:
            ln = n // 16
            last = min(n0 + rc - 1, 16 * ln + 15)
            for tk in range(T):
                jobs.append(Job(c, tk, dst + tk * stride + ln, n % 16, last % 16, last == N - 1, (n, last)))
            n = last + 1
    return jobs


def apply_mvm(spm: bytearray, dst: int, stride: int, N: int, T: int, acc: bool, ys) -> None:
    """The tsr_sim.c ex_mvm write loop (788-803): ys[t][n] f32 bits; stride already resolved
    (0 -> y_lines). Token by token; ACC = fadd(old, y); tail bytes of each token's last line 0."""
    r = ref()
    yl = (N + 15) // 16
    for t in range(T):
        base = 64 * (dst + t * stride)
        for n in range(N):
            v = ys[t][n]
            if acc:
                v = r.fadd(int.from_bytes(spm[base + 4 * n:base + 4 * n + 4], "little"), v)
            spm[base + 4 * n:base + 4 * n + 4] = v.to_bytes(4, "little")
        spm[base + 4 * N:base + 64 * yl] = bytes(64 * yl - 4 * N)


def mvm_rows(fmt: int, K: int, N: int, num_ch: int, w_off: int, read_beats) -> list:
    """The N repacked rows of an MVM from DRAM: read_beats(ch, beat, n) -> bytes."""
    rb = row_beats(fmt, K)
    R = -(-N // num_ch) if N else 0
    out = []
    for n in range(N):
        c = n // R
        out.append(bytes(read_beats(c, w_off + (n - c * R) * rb, rb)))
    return out


def mvm_reference(spm: bytearray, read_beats, *, fmt: int, acc: bool, dst: int, x: int, w_off: int,
                  N: int, K: int, T: int, dst_stride: int, num_ch: int) -> list:
    """Whole MVM like ex_mvm (operands legal): computes every y first, then writes. Returns
    ys[t][n]. dst_stride: the w6 value (0 -> ceil(N/16))."""
    kind = qa_kind(fmt)
    ql = isa.qa_lines(kind, K)
    rows = mvm_rows(fmt, K, N, num_ch, w_off, read_beats)
    qas = [bytes(spm[64 * (x + t * ql):64 * (x + (t + 1) * ql)]) for t in range(T)]
    r = ref()
    ys = [[r.dot_row(fmt, rows[n], K, qas[t]) for n in range(N)] for t in range(T)]
    apply_mvm(spm, dst, dst_stride or (N + 15) // 16, N, T, acc, ys)
    return ys


def layout_channels(fmt: int, K: int, rows, num_ch: int) -> list:
    """Per channel: the bytes to place at W_OFF (rows c*R .. of channel c, ROW_BEATS apart;
    spec Symmetric allocation). Channels without rows get b''."""
    N, rb = len(rows), row_beats(fmt, K)
    out = []
    for c in range(num_ch):
        R = -(-N // num_ch) if N else 0
        n0, rc = c * R, rows_c(N, num_ch, c) if N else 0
        out.append(b"".join(bytes(rows[n]).ljust(64 * rb, b"\0")[:64 * rb] for n in range(n0, n0 + rc)))
    return out


# =============================================================================================
# Random data
# =============================================================================================
_F16_SPECIAL = (0x0000, 0x8000, 0x0001, 0x83FF, 0x0400, 0x7BFF, 0xFBFF, 0x7C00, 0xFC00, 0x7E00, 0x7C01, 0xFE01)
_F32_SPECIAL = (0x00000000, 0x80000000, 0x00000001, 0x807FFFFF, 0x00800000, 0x80800000, 0x7F7FFFFF,
                0xFF7FFFFF, 0x7F800000, 0xFF800000, 0x7FC00000, 0xFFC00001, 0x7F800001, 0x3F800000,
                0x00FFFFFF, 0x3F7FFFFF)


def rand_f16(rnd: random.Random, special: float = 0.0) -> int:
    """fp16 bits: mostly a realistic weight scale (positive, 2^-14 .. 2^-2), sometimes any value."""
    if rnd.random() < special:
        return rnd.choice(_F16_SPECIAL) if rnd.random() < 0.6 else rnd.getrandbits(16)
    e = rnd.randint(1, 13)                               # biased exponent: 2^-14 .. 2^-2
    return (e << 10) | rnd.getrandbits(10)


def rand_f32(rnd: random.Random, special: float = 0.0, lo: int = -20, hi: int = 4) -> int:
    if rnd.random() < special:
        return rnd.choice(_F32_SPECIAL) if rnd.random() < 0.6 else rnd.getrandbits(32)
    return (rnd.getrandbits(1) << 31) | ((127 + rnd.randint(lo, hi)) << 23) | rnd.getrandbits(23)


def rand_blocks(fmt: int, K: int, rnd: random.Random, special: float = 0.0) -> bytes:
    """ggml blocks of one row: Q8_0 {d fp16, qs[32] int8 (full range)}, Q4_K {d, dmin, scales[12],
    qs[128]} with random bytes."""
    out = bytearray()
    for _ in range(K // block_size(fmt)):
        if fmt == FMT_Q8_0:
            out += struct.pack("<H", rand_f16(rnd, special)) + rnd.randbytes(32)
        else:
            out += struct.pack("<HH", rand_f16(rnd, special), rand_f16(rnd, special)) + rnd.randbytes(12 + 128)
    return bytes(out)


def rand_row(fmt: int, K: int, rnd: random.Random, special: float = 0.0, raw: bool = False) -> bytes:
    """One repacked row image (ROW_BEATS * 64 bytes). raw=True: every byte random (also the meta
    entries / quant tails of a partial chunk, which the hardware must ignore)."""
    if raw:
        return rnd.randbytes(64 * row_beats(fmt, K))
    return ref().repack_row(fmt, rand_blocks(fmt, K, rnd, special), K)


def rand_qa(kind: int, K: int, rnd: random.Random, mode: str = "qact", special: float = 0.0) -> bytes:
    """One QA image (QA_LINES * 64 bytes).
    mode 'qact': QACT of random f32 activations (spec-realistic; `special` adds NaN / Inf /
                 subnormal / zero elements and all-zero blocks);
    mode 'raw' : arbitrary bytes (legal MVM input): QS any int8 (incl. -128), D any f32 (special
                 values with probability `special`), BS any int16 (extremes included)."""
    n = isa.qa_lines(kind, K)
    if mode == "qact":
        x = []
        zero_blk = rnd.random() < special
        for i in range(K):
            if zero_blk and i < (256 if kind else 32):
                x.append(0)
            elif rnd.random() < special * 0.05:
                x.append(rnd.choice(_F32_SPECIAL))
            else:
                x.append(rand_f32(rnd, 0.0, -6, 3))
        return ref().qact(kind, x, K)
    img = bytearray(rnd.randbytes(64 * n))
    ql, nd = qa_qs_lines(K), K // (256 if kind else 32)
    for b in range(nd):
        img[64 * ql + 4 * b:64 * ql + 4 * b + 4] = rand_f32(rnd, special, -12, 2).to_bytes(4, "little")
    if kind == QA_A256 and rnd.random() < 0.3:
        bs0 = 64 * (ql + qa_d_lines(kind, K))
        for i in range(K // 16):
            if rnd.random() < 0.5:
                img[bs0 + 2 * i:bs0 + 2 * i + 2] = rnd.choice((0x8000, 0x7FFF)).to_bytes(2, "little")
    return bytes(img)


def ftz_window_dx(d: int, rnd: random.Random | None = None, tries: int = 64) -> int | None:
    """An f32 Dx (bits) such that the exact product d * Dx lies in the FTZ boundary window
    [2^-126 - 2^-150, 2^-126) (spec Implementation clarifications, "FTZ decision point": fmul gives +-2^-126 there), or None.
    d: f32 bits of a normal, positive or negative value."""
    def val(u):
        e, m = (u >> 23) & 0xFF, u & 0x7FFFFF
        assert 0 < e < 255
        v = Fraction((1 << 23) | m, 1 << 23) * (Fraction(2) ** (e - 127))
        return -v if u >> 31 else v
    try:
        dv = abs(val(d))
    except AssertionError:
        return None
    lo, hi = Fraction(2) ** -126 - Fraction(2) ** -150, Fraction(2) ** -126
    target = (lo + hi) / 2 / dv
    if target <= Fraction(2) ** -126 or target >= Fraction(2) ** 128:
        return None
    e = target.numerator.bit_length() - target.denominator.bit_length()
    while Fraction(2) ** e > target:
        e -= 1
    while Fraction(2) ** (e + 1) <= target:
        e += 1
    m = int((target / Fraction(2) ** e - 1) * (1 << 23))
    base = ((e + 127) << 23) | m
    for k in range(tries):
        u = base + (k // 2 if k % 2 == 0 else -(k // 2 + 1))
        if lo <= dv * abs(val(u)) < hi:
            sign = (rnd.getrandbits(1) << 31) if rnd else 0
            return u | sign
    return None
