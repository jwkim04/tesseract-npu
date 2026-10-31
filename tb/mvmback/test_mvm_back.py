"""tsr_mvm_back / tsr_mvm_actbuf / tsr_mac_lane tests (docs/rtl-m3.md §3.3-§3.5, §5, §7,
§10.2 "back").

DUT: tb/mvmback/hdl/tsr_tb_mvm_back.sv = tsr_mvm_actbuf written through its broadcast-side port
(tsr_mvm_bcline_t items exactly as mvm_model.bc_lines produces them) + tsr_mvm_back reading it.
Golden model: tb/common/mvm_model.py (front_beats = the tagged beat stream of §3.3, bc_lines,
actbuf_image) and libtsr_ref tsr_dot_row for every row result (the bit-exact definition; the
model's Cluster cross-check of terms / partials / tree is used for diagnostics on a mismatch).

Checks on every MVM:
  - every row result res_y[t] (t < T) == tsr_dot_row(fmt, row, K, QA_t), in row order, exactly
    one result per row and nothing more;
  - R2 on the result stream (res_valid held, res_y stable while not taken);
  - the back end's activation reads (observed at the actbuf ports): one QS read and one DBS read
    exactly one cycle after each accepted beat, at the docs/rtl-m3.md §3.4 addresses (QS word b/4
    or 2s + half, D line blk >> 4, Q4_K half 1: BS line d_lines + s/2);
  - idle = 0 from the cycle after the first accepted beat until the last result is taken, 1 after;
  - simulation assertions in the RTL (tsr_mvm_back / tsr_mvm_actbuf / fp units) fail the test.

cocotb tests (each configuration runs all of them):
  back_actbuf      broadcast of T_MAX raw QA images (A32 and A256, K = K_MAX), then testbench reads
                   of every QS word / DBS line through the buffer's read ports: contents ==
                   mvm_model.actbuf_image for every token, data exactly ACT_RD_LAT cycles after the
                   request (back-to-back and random reads), write -> read distance 2 cycles
  back_directed    K over block multiples incl. partial chunks (Q8_0 odd / even remainders, Q4_K
                   rem 1..3), K = block size, K_MAX; rows 1..4; T = 1 .. T_MAX; data modes
                   cycled (realistic QACT images and repacked rows, raw bytes, special values)
  back_special     constructed extremes: Q8_0 isum +-2^19 (w = -128 with QS -128 / 127), Q4_K
                   isum > 2^24 (nibbles 15, scales / mins 63, random large QS), msum at the 26-bit
                   extremes (BS -32768 / 32767, m = 63), every special fp16 d / dmin (subnormal,
                   zero, -0, Inf, NaN, max), special Dx (NaN, Inf, subnormal, +-0, max), d * Dx in
                   the FTZ window [2^-126 - 2^-150, 2^-126), tiny terms of mixed sign whose partial
                   and tree sums are subnormal (FTZ to signed zero), raw -128 everywhere
  back_random      random fmt / K / rows / T / data modes, random beat_valid gaps and res_ready
                   backpressure, back-to-back MVMs
  back_throughput  beats offered every cycle, res_ready = 1: no beat_ready drop for rows of
                   ROW_BEATS >= RB_MIN = 9 (Q8_0 K 480 / 512 / 1024 / 1056, Q4_K K 1024 / 2304 /
                   K_MAX; only those that fit K_MAX and have >= RB_MIN beats, logged when a format
                   has none) at T = 1 and T = T_MAX; ROW_BEATS 8 (Q8_0 K 448) with the front end's
                   META cycles: no drop; without them: exactly one bubble per row (the measurement
                   discriminates); short rows: cycles per row; latency row_last -> result
  back_backpressure res_ready = 0 for a long phase: exactly RES_DEPTH rows are taken, then nothing;
                   Q4_K superblocks s >= 16 and Q8_0 rows with beat gaps up to 2 ADD_LAT + 4 cycles;
                   Q8_0 odd block counts with NaN / Inf in the unused D element of the D line's tail
  back_reset       synchronous reset in the middle of an MVM (beats, partial sums, tree and result
                   FIFO occupied, also with res_ready = 0), reset values, then checked MVMs
  back_comb_probe  no combinational path from any input to any output (clock stopped, inputs wiggled)
Configurations: t1 (U200: K_MAX 32768, T_MAX 1, default latencies), t8 (T_MAX 8, defaults),
lat1 (every latency 1), lat8 (every latency 8), mix (K_MAX 65280, T_MAX 5, ACT 3 / MAC 1 / MUL 8 /
ADD 7: the d*Dx path is longer than the integer path, so the f32 sums are the delayed ones), k256
(K_MAX 256 = the docs/rtl-m3.md §1 edge value, T_MAX 8, ACT 1 / MAC 8 / MUL 1 / ADD 8: no row
reaches RB_MIN beats), k768 (K_MAX 768, T_MAX 2, ACT 4 / MAC 2 / MUL 2 / ADD 3: Q8_0 rows reach
RB_MIN, Q4_K rows do not).

Separate pytest items:
  test_mvm_back_mac_lane[...]  tsr_mac_lane alone (N 32 / 5 / 1, T 1..8, LAT 1..6 and 9): random and
                               extreme operands (-128), random ce stalls and in_valid; out_valid / y
                               against a LAT-deep model that advances only with ce (mac_lane); and
                               the register placement measured on the layer signals g_l[i].d / .q
                               (mac_lane_structure, docs/rtl-m3.md §9.2 item 4: no input register, a
                               product register for LAT >= 2, balanced adder segments; N 32 LAT 3 =
                               products | 2 adder levels | 3 adder levels)
  test_mvm_back_i2f_sweep      tsr_mvm_i2f (the 3-stage int -> f32 split of §8.2) exhaustively: all
                               2^26 inputs vs libtsr_ref tsr_i32_to_f32 (Verilator C++ harness)
Environment: TSR_BACK_TESTS=a,b (subset of the cocotb tests), TSR_BACK_SCALE (random test size
factor, default 1.0), TSR_MVMBACK_RTL (alternative rtl/filelist.f, used by the mutation campaign).
"""
from __future__ import annotations

import json
import os
import random
import struct
import subprocess
from collections import deque

import cocotb
import pytest
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, FallingEdge, RisingEdge, Timer

from common import isa
from common import mvm_model as mm
from common.runner import REPO, build_root, rtl_args, rtl_sources, run
from common.tsr_ref import ref

TB_SV = REPO / "tb" / "mvmback" / "hdl" / "tsr_tb_mvm_back.sv"
I2F_SV = REPO / "tb" / "mvmback" / "hdl" / "tsr_tb_mvm_i2f.sv"
I2F_CPP = REPO / "tb" / "mvmback" / "cpp" / "i2f_sweep.cpp"

CFGS = {
    "t1":   dict(K_MAX=32768, T_MAX=1, ACT_RD_LAT=2, MAC_LAT=3, MUL_LAT=4, ADD_LAT=5),
    "t8":   dict(K_MAX=4096, T_MAX=8, ACT_RD_LAT=2, MAC_LAT=3, MUL_LAT=4, ADD_LAT=5),
    "lat1": dict(K_MAX=2048, T_MAX=2, ACT_RD_LAT=1, MAC_LAT=1, MUL_LAT=1, ADD_LAT=1),
    "lat8": dict(K_MAX=8192, T_MAX=3, ACT_RD_LAT=8, MAC_LAT=8, MUL_LAT=8, ADD_LAT=8),
    "mix":  dict(K_MAX=65280, T_MAX=5, ACT_RD_LAT=3, MAC_LAT=1, MUL_LAT=8, ADD_LAT=7),
    # small K_MAX (docs/rtl-m3.md §1 edge value 256): no row reaches RB_MIN beats at K_MAX 256;
    # at K_MAX 768 the Q8_0 rows do (13 beats), the Q4_K ones do not (7 beats)
    "k256": dict(K_MAX=256, T_MAX=8, ACT_RD_LAT=1, MAC_LAT=8, MUL_LAT=1, ADD_LAT=8),
    "k768": dict(K_MAX=768, T_MAX=2, ACT_RD_LAT=4, MAC_LAT=2, MUL_LAT=2, ADD_LAT=3),
}

TESTS = ["back_actbuf", "back_directed", "back_special", "back_random", "back_throughput", "back_backpressure",
         "back_reset", "back_comb_probe"]

MAC_CFGS = {
    "n32_t1_l1": dict(N=32, T=1, LAT=1),
    "n32_t1_l2": dict(N=32, T=1, LAT=2),
    "n32_t3_l3": dict(N=32, T=3, LAT=3),
    "n32_t1_l3": dict(N=32, T=1, LAT=3),
    "n32_t2_l4": dict(N=32, T=2, LAT=4),
    "n32_t1_l5": dict(N=32, T=1, LAT=5),
    "n32_t2_l6": dict(N=32, T=2, LAT=6),
    "n32_t8_l9": dict(N=32, T=8, LAT=9),
    "n5_t2_l4":  dict(N=5, T=2, LAT=4),
    "n1_t1_l2":  dict(N=1, T=1, LAT=2),
}

# tsr_mac_lane register placement (docs/rtl-m3.md §9.2 item 4), N = 32: logic layer i (0 = the
# products, 1..5 = adder levels) -> the number of register stages between the operands and that
# layer's inputs. Default MAC_LAT 3: products | 2 adder levels | 3 adder levels.
MAC_STAGES_N32 = {1: [0, 0, 0, 0, 0, 0], 2: [0, 1, 1, 1, 1, 1], 3: [0, 1, 1, 2, 2, 2],
                  4: [0, 1, 2, 2, 3, 3], 5: [0, 1, 2, 3, 4, 4], 6: [0, 1, 2, 3, 4, 5]}

RB_MIN = 9          # achieved bound (tsr_mvm_back header): rows of >= 8 quant beats never stall

INPUTS = ["wr_valid", "wr_line", "cfg_fmt", "cfg_d_lines", "beat_valid", "beat", "res_ready", "tb_rd",
          "tb_qs_re", "tb_qs_addr", "tb_dbs_re", "tb_dbs_addr"]
OUTPUTS = ["beat_ready", "res_valid", "res_y", "idle", "qs_data", "dbs_data", "be_qs_re", "be_qs_addr",
           "be_dbs_re", "be_dbs_addr"]

F16_SPECIAL = (0x0000, 0x8000, 0x0001, 0x8001, 0x03FF, 0x83FF, 0x0400, 0x7BFF, 0xFBFF, 0x7C00, 0xFC00,
               0x7E00, 0x7C01, 0xFE01, 0x7FFF, 0x3C00)
F32_SPECIAL = (0x00000000, 0x80000000, 0x00000001, 0x807FFFFF, 0x00800000, 0x80800000, 0x7F7FFFFF,
               0xFF7FFFFF, 0x7F800000, 0xFF800000, 0x7FC00000, 0xFFC00001, 0x7F800001, 0x3F800000)


def sources():
    fl = os.environ.get("TSR_MVMBACK_RTL")
    return rtl_sources(fl) if fl else None


# =============================================================================================
# pytest side
# =============================================================================================
@pytest.mark.parametrize("cfg", list(CFGS))
def test_mvm_back(cfg):
    p = CFGS[cfg]
    sel = os.environ.get("TSR_BACK_TESTS")
    tests = [t for t in TESTS if not sel or t in sel.split(",")]
    run("tsr_tb_mvm_back", __name__, parameters=p, extra_sources=[TB_SV], sources=sources(), testcase=tests,
        extra_env={"TSR_BACK_CFG": json.dumps(dict(p, name=cfg))})


@pytest.mark.parametrize("cfg", list(MAC_CFGS))
def test_mvm_back_mac_lane(cfg):
    p = MAC_CFGS[cfg]
    run("tsr_mac_lane", __name__, parameters=p, sources=sources(), testcase=["mac_lane", "mac_lane_structure"],
        extra_env={"TSR_MAC_CFG": json.dumps(p)})


def test_mvm_back_i2f_sweep():
    """tsr_mvm_i2f (IW 26) for every 26-bit input against libtsr_ref tsr_i32_to_f32."""
    lib = ref().path
    from pathlib import Path
    lib = Path(lib).resolve()
    vl = os.environ.get("VERILATOR", "verilator")
    import hashlib
    srcs = sources() or rtl_sources()
    key = "i2f_sweep_" + hashlib.sha1(repr([str(s) for s in srcs]).encode()).hexdigest()[:10]
    bdir = build_root() / key
    bdir.mkdir(parents=True, exist_ok=True)
    jobs = os.environ.get("TSR_BUILD_JOBS", "4")
    cmd = [vl, "--cc", "--exe", "--build", "-O3", "-j", jobs, "--top-module", "tsr_tb_mvm_i2f",
           "-CFLAGS", "-O2 -I%s" % (REPO / "sw" / "sim" / "include"),
           "-LDFLAGS", "-L%s -ltsr_ref -Wl,-rpath,%s" % (lib.parent, lib.parent),
           "--Mdir", str(bdir), "-o", "i2f_sweep"]
    cmd += rtl_args() + [str(s) for s in srcs] + [str(I2F_SV), str(I2F_CPP)]   # rtl_args: rtl/ctrl's (W4-A)
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode == 0, "i2f_sweep build failed:\n%s\n%s" % (p.stdout[-4000:], p.stderr[-4000:])
    exe = bdir / "i2f_sweep"
    # four quarters of [-2^25, 2^25) in parallel
    q = 1 << 24
    jobs_ = [(-(1 << 25) + i * q, -(1 << 25) + (i + 1) * q) for i in range(4)]
    procs = [subprocess.Popen([str(exe), str(a), str(b)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True) for a, b in jobs_]
    outs = [(pr.communicate()[0], pr.returncode) for pr in procs]
    for o, rc in outs:
        print(o.strip())
    assert all(rc == 0 for _, rc in outs), "i2f sweep mismatches:\n%s" % "\n".join(o for o, _ in outs)
    assert sum(int(o.strip().splitlines()[-1].split(":")[1].split()[0]) for o, _ in outs) == 1 << 26


# =============================================================================================
# Helpers
# =============================================================================================
def cfg_from_env() -> dict:
    return json.loads(os.environ["TSR_BACK_CFG"])


def scale() -> float:
    return float(os.environ.get("TSR_BACK_SCALE", "1.0"))


def s8b(v: int) -> int:
    return v & 0xFF


def f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


class Mvm:
    """One MVM of one cluster: the tagged beats, the broadcast lines and the expected row results."""

    def __init__(self, fmt: int, K: int, rows, qas, tag: str = ""):
        self.fmt, self.K, self.tag = fmt, K, tag
        self.rows = [bytes(r) for r in rows]
        self.qas = [bytes(q) for q in qas]
        self.T = len(self.qas)
        self.kind = mm.qa_kind(fmt)
        self.d_lines = isa.qa_d_lines(self.kind, K)
        self.rb = mm.row_beats(fmt, K)
        self.beats = mm.front_beats(fmt, K, self.rows)
        self.packed = [mm.pack_beat(d) for d in self.beats]
        cb = 17 if fmt == mm.FMT_Q8_0 else 9
        self.chunk0 = [d["beat"] % cb == 1 for d in self.beats]      # first quant beat of a chunk
        self.addr = []                                               # (QS word, DBS line) per beat
        for d in self.beats:
            b = d["blk"]
            if fmt == mm.FMT_Q8_0:
                self.addr.append((b >> 2, b >> 4))
            else:
                self.addr.append((2 * b + d["half"], self.d_lines + b // 2 if d["half"] else b >> 4))
        r = ref()
        self.y = [[r.dot_row(fmt, row, K, qa) for qa in self.qas] for row in self.rows]
        self.lines = [isa.MVM_BCLINE_T.pack(**ln) for ln in mm.bc_lines(self.kind, K, self.qas)]

    def diag(self, ri: int, t: int) -> str:
        """Model intermediates of row ri, token t (for a mismatch message)."""
        r = ref()
        nb = self.K // mm.block_size(self.fmt)
        terms = [r.term(self.fmt, self.rows[ri], self.K, i, self.qas[t]) for i in range(nb)]
        p = [0] * 16
        for i, tb in enumerate(terms):
            p[i % 16] = r.fadd(p[i % 16], tb)
        return "terms[:8] %s partials %s" % (["%08x" % x for x in terms[:8]], ["%08x" % x for x in p])

    def __repr__(self):
        return "Mvm(%s fmt %d K %d rows %d T %d RB %d)" % (self.tag, self.fmt, self.K, len(self.rows), self.T,
                                                         self.rb)


# ---------------------------------------------------------------------------------------------
# Data construction
# ---------------------------------------------------------------------------------------------
def k_choices(fmt: int, k_max: int):
    bs = mm.block_size(fmt)
    if fmt == mm.FMT_Q8_0:
        # incl. Qwen2.5-0.5B shapes K 896 (hidden, 15 beats) and 4864 (ffn_down, 85 beats)
        ks = {32, 64, 96, 416, 448, 480, 512, 896, 992, 1024, 1056, 1120, 2048 + 96, 3072 - 32, 4864,
              k_max - 32, k_max}
    else:
        ks = {256, 512, 768, 1024, 1280, 1536, 1792, 2304, 4096 + 768, 4864, k_max - 256, k_max}
    return sorted(k for k in ks if 0 < k <= k_max and k % bs == 0)


def ggml_q8_0(blocks) -> bytes:
    """blocks: [(d fp16 bits, [32 int8])] -> ggml block_q8_0 bytes."""
    out = bytearray()
    for d, qs in blocks:
        out += struct.pack("<H", d) + bytes(s8b(q) for q in qs)
    return bytes(out)


def ggml_q4_K(blocks) -> bytes:
    """blocks: [(d, dmin, scales[12] bytes, qs[128] bytes)] -> ggml block_q4_K bytes."""
    out = bytearray()
    for d, dmin, sc, qs in blocks:
        out += struct.pack("<HH", d, dmin) + bytes(sc) + bytes(qs)
    return bytes(out)


def repack(fmt: int, K: int, ggml: bytes) -> bytes:
    return ref().repack_row(fmt, ggml, K)


def make_qa(kind: int, K: int, qs, dx, bs=None, rnd: random.Random | None = None) -> bytes:
    """QA image (tsr_ref.c layout): qs = K int8, dx = f32 bits per block, bs = int16 per 16 (A256).
    Tail bytes of each region are random if rnd is given (any content is legal), else 0."""
    ql = mm.qa_qs_lines(K)
    dl = isa.qa_d_lines(kind, K)
    n = isa.qa_lines(kind, K)
    img = bytearray(rnd.randbytes(64 * n) if rnd else bytes(64 * n))
    img[0:K] = bytes(s8b(q) for q in qs)
    for b, v in enumerate(dx):
        img[64 * ql + 4 * b:64 * ql + 4 * b + 4] = (v & 0xFFFFFFFF).to_bytes(4, "little")
    if kind == mm.QA_A256:
        o = 64 * (ql + dl)
        for i, v in enumerate(bs):
            img[o + 2 * i:o + 2 * i + 2] = (v & 0xFFFF).to_bytes(2, "little")
    return bytes(img)


def gen_rows(fmt: int, K: int, n: int, rnd: random.Random, mode: str = "mix") -> list:
    rows = []
    for _ in range(n):
        m = mode if mode != "mix" else rnd.choice(("raw", "pack", "pack_special", "raw"))
        if m == "raw":
            rows.append(mm.rand_row(fmt, K, rnd, raw=True))
        elif m == "pack":
            rows.append(mm.rand_row(fmt, K, rnd))
        else:
            rows.append(mm.rand_row(fmt, K, rnd, special=0.2))
    return rows


def gen_qas(fmt: int, K: int, T: int, rnd: random.Random, mode: str = "mix") -> list:
    kind = mm.qa_kind(fmt)
    out = []
    for _ in range(T):
        m = mode if mode != "mix" else rnd.choice(("qact", "qact_special", "raw", "raw_special"))
        if m == "qact":
            out.append(mm.rand_qa(kind, K, rnd, "qact"))
        elif m == "qact_special":
            out.append(mm.rand_qa(kind, K, rnd, "qact", special=0.3))
        elif m == "raw":
            out.append(mm.rand_qa(kind, K, rnd, "raw"))
        else:
            out.append(mm.rand_qa(kind, K, rnd, "raw", special=0.3))
    return out


def tiny_f32(rnd: random.Random) -> int:
    """A normal f32 close to 2^-126 (exponent 1..3), random sign."""
    return (rnd.getrandbits(1) << 31) | (rnd.randint(1, 3) << 23) | rnd.getrandbits(23)


def special_cases(rnd: random.Random, k_max: int, t_max: int) -> list:
    """Constructed extreme MVMs (see back_special in the module docstring)."""
    r = ref()
    cases = []
    T = t_max
    # ---------------- Q8_0 ----------------
    K = min(k_max, 1056)                  # one full chunk + 1 block (odd partial chunk)
    nb = K // 32
    one = 0x3C00
    # isum extremes: w = -128 with QS -128 (+2^19) or 127 (-520192); d = 1, Dx = 1
    rows = [repack(0, K, ggml_q8_0([(one, [-128] * 32)] * nb))]
    qas = [make_qa(0, K, [(-128 if t % 2 == 0 else 127)] * K, [0x3F800000] * nb) for t in range(T)]
    cases.append(Mvm(0, K, rows, qas, "q8 isum extremes"))
    # every special fp16 d, special Dx, raw -128 sprinkled
    blocks = []
    for b in range(nb):
        blocks.append((F16_SPECIAL[b % len(F16_SPECIAL)], [rnd.choice((-128, 127, rnd.randrange(-128, 128)))
                                                          for _ in range(32)]))
    rows = [repack(0, K, ggml_q8_0(blocks)), repack(0, K, ggml_q8_0(blocks[::-1]))]
    qas = []
    for t in range(T):
        dx = [rnd.choice(F32_SPECIAL) if rnd.random() < 0.5 else mm.rand_f32(rnd) for _ in range(nb)]
        qas.append(make_qa(0, K, [rnd.choice((-128, 127, rnd.randrange(-128, 128))) for _ in range(K)], dx,
                           rnd=rnd))
    cases.append(Mvm(0, K, rows, qas, "q8 special d / Dx"))
    # d * Dx in the FTZ window (d < 1 so that Dx is normal; both rows share each block's d),
    # isum = +-1..3 -> terms +-2^-126 * isum; and tiny mixed-sign terms whose partial and tree sums
    # are subnormal (FTZ to a signed zero)
    for variant in ("window", "tiny"):
        K2 = min(k_max, 2048)
        nb2 = K2 // 32
        if variant == "window":
            ds = [(rnd.randint(1, 14) << 10) | rnd.getrandbits(10) for _ in range(nb2)]     # (2^-14, 1)
        else:
            ds = [rnd.choice((0x3C00, 0x3E00, 0x3800, (rnd.randint(10, 20) << 10) | rnd.getrandbits(10)))
                  for _ in range(nb2)]
        rows, qas = [], []
        for _ in range(2):
            blocks = []
            for b in range(nb2):
                w = [0] * 32
                w[rnd.randrange(32)] = rnd.choice((1, -1, 2, -3))
                blocks.append((ds[b], w))
            rows.append(repack(0, K2, ggml_q8_0(blocks)))
        nwin = 0
        for t in range(T):
            qs = [rnd.choice((1, -1)) for _ in range(K2)]
            dx = []
            for b in range(nb2):
                dv = mm.ftz_window_dx(r.fp16_to_f32(ds[b]), rnd) if variant == "window" else None
                nwin += dv is not None
                dx.append(dv if dv is not None else tiny_f32(rnd))
            qas.append(make_qa(0, K2, qs, dx))
        if variant == "window":
            assert nwin >= T * nb2 // 2, "too few FTZ-window Dx found (%d)" % nwin
        cases.append(Mvm(0, K2, rows, qas, "q8 ftz %s" % variant))
    # ---------------- Q4_K ----------------
    K = min(k_max, 1280)                   # one full chunk + 1 superblock
    nb = K // 256
    ff12 = bytes([0xFF] * 12)
    # isum > 2^24: nibbles 15, sc = m = 63, QS large; msum extremes: BS -32768 / 32767
    blocks = [(0x3C00, 0x3C00, ff12, bytes([0xFF] * 128)) for _ in range(nb)]
    rows = [repack(1, K, ggml_q4_K(blocks))]
    blocks2 = [(rnd.choice(F16_SPECIAL), rnd.choice(F16_SPECIAL), ff12, bytes([0xFF] * 128)) for _ in range(nb)]
    rows.append(repack(1, K, ggml_q4_K(blocks2)))
    qas = []
    for t in range(T):
        sel = t % 4
        if sel == 0:
            qs = [rnd.randint(100, 127) for _ in range(K)]
            bs = [-32768] * (K // 16)
        elif sel == 1:
            qs = [rnd.randint(-128, -100) for _ in range(K)]
            bs = [32767] * (K // 16)
        elif sel == 2:
            qs = [127] * K
            bs = [rnd.choice((-32768, 32767)) for _ in range(K // 16)]
        else:
            qs = [-128] * K
            bs = [rnd.randrange(-32768, 32768) for _ in range(K // 16)]
        dx = [0x3F800000 if rnd.random() < 0.5 else mm.rand_f32(rnd) for _ in range(nb)]
        qas.append(make_qa(1, K, qs, dx, bs, rnd=rnd))
    cases.append(Mvm(1, K, rows, qas, "q4k isum > 2^24 / msum 26 bits"))
    # random scale bytes (all 6-bit scale / min patterns), special Dx, raw content
    blocks = [(rnd.choice(F16_SPECIAL + (0x3C00,) * 4), rnd.choice(F16_SPECIAL + (0x3C00,) * 4), rnd.randbytes(12),
               rnd.randbytes(128)) for _ in range(nb)]
    rows = [repack(1, K, ggml_q4_K(blocks))]
    qas = []
    for t in range(T):
        dx = [rnd.choice(F32_SPECIAL) for _ in range(nb)]
        qas.append(make_qa(1, K, [rnd.randrange(-128, 128) for _ in range(K)], dx,
                           [rnd.randrange(-32768, 32768) for _ in range(K // 16)], rnd=rnd))
    cases.append(Mvm(1, K, rows, qas, "q4k special d / dmin / Dx"))
    # tiny mixed-sign terms (FTZ partial / tree sums): sc_0 = 1, m_0 = 1, single weight, msum +-1
    K2 = min(k_max, 4096)
    nb2 = K2 // 256
    rows = []
    for _ in range(2):
        blocks = []
        for s in range(nb2):
            sc = bytes([1, 0, 0, 0, rnd.choice((0, 1)), 0, 0, 0, 0, 0, 0, 0])   # sc_0 = 1, m_0 = 0 / 1
            qs = bytearray(128)
            qs[rnd.randrange(32)] = rnd.choice((1, 2, 3))                       # sub-block 0 (low nibble)
            blocks.append((rnd.choice((0x3C00, 0x3800)), rnd.choice((0x3C00, 0x3E00, 0x0000)), sc, bytes(qs)))
        rows.append(repack(1, K2, ggml_q4_K(blocks)))
    qas = []
    for t in range(T):
        qs = [rnd.choice((1, -1)) for _ in range(K2)]
        dx = [tiny_f32(rnd) for _ in range(nb2)]
        bs = [rnd.choice((1, -1, 0)) for _ in range(K2 // 16)]
        qas.append(make_qa(1, K2, qs, dx, bs))
    cases.append(Mvm(1, K2, rows, qas, "q4k ftz tiny"))
    return cases


# =============================================================================================
# Bench
# =============================================================================================
class Bench:
    def __init__(self, dut, cfg: dict, rnd: random.Random):
        self.dut, self.cfg, self.rnd = dut, cfg, rnd
        self.tmax, self.kmax = cfg["T_MAX"], cfg["K_MAX"]
        self.rd_lat = cfg["ACT_RD_LAT"]
        self.clock = None
        self.log = dut._log

    async def start(self, cycles: int = 5):
        dut = self.dut
        dut.rst_n.value = 0
        dut.clk.value = 0
        for n in INPUTS:
            getattr(dut, n).value = 0
        await Timer(1, "ns")
        self.clock = Clock(dut.clk, 10, unit="ns")
        self.clock.start(start_high=True)
        await ClockCycles(dut.clk, cycles)
        await RisingEdge(dut.clk)
        dut.rst_n.value = 1

    async def broadcast(self, m: Mvm, gap: float = 0.0):
        """Drive the MVM's broadcast lines into the activation buffer (we are after a rising edge)."""
        dut, rnd = self.dut, self.rnd
        for ln in m.lines:
            while gap and rnd.random() < gap:
                dut.wr_valid.value = 0
                await RisingEdge(dut.clk)
            dut.wr_valid.value = 1
            dut.wr_line.value = ln
            await RisingEdge(dut.clk)
        dut.wr_valid.value = 0

    async def run_mvm(self, m: Mvm, p_valid: float = 1.0, p_ready: float = 1.0, meta_gaps: bool = False,
                      no_stall: bool = False, abort_after: int | None = None, bc_gap: float = 0.0,
                      gap_max: int = 0, ready_hold: int = 0, hold_probe=None) -> dict:
        """Broadcast, then stream m's beats and collect / check its results. Returns statistics.
        abort_after: stop after that many cycles of the beat phase (for reset tests; no final checks).
        gap_max: 0..gap_max idle cycles (uniform) before each beat; ready_hold: res_ready = 0 for the
        first ready_hold cycles; hold_probe(stats) is called once at the end of the hold."""
        dut, rnd = self.dut, self.rnd
        await self.broadcast(m, bc_gap)
        dut.cfg_fmt.value = m.fmt
        dut.cfg_d_lines.value = m.d_lines
        nb, nrows = len(m.packed), len(m.y)
        bi, ri, valid, cyc = 0, 0, 0, 0
        stalls, drops = 0, 0
        first_acc, last_acc = None, None
        rl_cyc, res_cyc = [], []
        exp_rd = deque()
        acc_beats, pop_beats = 0, 0                 # idle == (every accepted beat's row was taken)
        row_q = [sum(1 for d in m.beats if d["row"] == r) for r in range(nrows)]
        held = None
        gap_for = -1
        gap_left = rnd.randint(0, gap_max) if gap_max else 0
        rl_acc = 0
        timeout = 40 * nb + 400 * nrows + 2000 + ready_hold + nb * gap_max
        while ri < nrows:
            # ---- drive (right after a rising edge) ----
            if not valid and bi < nb:
                want = rnd.random() < p_valid
                if meta_gaps and m.chunk0[bi] and gap_for != bi:
                    want, gap_for = False, bi
                if gap_left:
                    want, gap_left = False, gap_left - 1
                if want:
                    valid = 1
                    dut.beat.value = m.packed[bi]
            dut.beat_valid.value = valid
            rr = 1 if rnd.random() < p_ready else 0
            if cyc < ready_hold:
                rr = 0
            if hold_probe is not None and cyc == ready_hold:
                hold_probe(dict(row_lasts=rl_acc, beats=bi, rows_out=ri))
            dut.res_ready.value = rr
            await FallingEdge(dut.clk)
            cyc += 1
            br = int(dut.beat_ready.value)
            rv = int(dut.res_valid.value)
            # activation reads: exactly one cycle after each accepted beat, at the §3.4 addresses
            qre, dre = int(dut.be_qs_re.value), int(dut.be_dbs_re.value)
            if exp_rd and exp_rd[0][0] == cyc:
                _, qa, da, idx = exp_rd.popleft()
                assert qre and dre, "%s: no activation read in the cycle after beat %d" % (m, idx)
                gq, gd = int(dut.be_qs_addr.value), int(dut.be_dbs_addr.value)
                assert (gq, gd) == (qa, da), "%s beat %d: read addresses QS %d DBS %d, expected %d %d" % (
                    m, idx, gq, gd, qa, da)
            else:
                assert not qre and not dre, "%s: activation read without an accepted beat (cycle %d)" % (m, cyc)
            assert int(dut.idle.value) == int(acc_beats == pop_beats), \
                "%s: idle %d with %d beats accepted, %d beats of taken rows (cycle %d)" % (
                    m, int(dut.idle.value), acc_beats, pop_beats, cyc)
            if bi < nb and not br:
                drops += 1
                if no_stall:
                    raise AssertionError("%s: beat_ready dropped at beat %d (cycle %d) with rows of %d beats" % (
                        m, bi, cyc, m.rb))
            if valid and not br:
                stalls += 1
            if valid and br:
                exp_rd.append((cyc + 1,) + m.addr[bi] + (bi,))
                if first_acc is None:
                    first_acc = cyc
                last_acc = cyc
                if m.beats[bi]["row_last"]:
                    rl_cyc.append(cyc)
                    rl_acc += 1
                bi += 1
                acc_beats += 1
                valid = 0
                if gap_max:
                    gap_left = rnd.randint(0, gap_max)
            if rv:
                y = int(dut.res_y.value)
                if held is not None:
                    assert y == held, "%s: res_y changed while res_valid and not taken (R2)" % m
                if rr:
                    for t in range(m.T):
                        got = (y >> (32 * t)) & 0xFFFFFFFF
                        exp = m.y[ri][t]
                        if got != exp:
                            raise AssertionError("%s row %d token %d: got 0x%08x expected 0x%08x (%s)" % (
                                m, ri, t, got, exp, m.diag(ri, t)))
                    res_cyc.append(cyc)
                    pop_beats += row_q[ri]
                    ri += 1
                    held = None
                else:
                    held = y
            else:
                assert held is None, "%s: res_valid dropped before the result was taken (R2)" % m
            if abort_after is not None and cyc >= abort_after:
                return dict(aborted=True, beats=bi, rows=ri)
            assert cyc < timeout, "%s: timeout (%d of %d beats, %d of %d results)" % (m, bi, nb, ri, nrows)
            await RisingEdge(dut.clk)
        # ---- after the last result: nothing more, idle ----
        dut.beat_valid.value = 0
        dut.res_ready.value = 1
        for k in range(4):
            await FallingEdge(dut.clk)
            assert not int(dut.res_valid.value), "%s: extra result after the last row" % m
            if k >= 1:
                assert int(dut.idle.value) == 1, "%s: not idle after the last result" % m
            await RisingEdge(dut.clk)
        assert bi == nb, "%s: %d of %d beats taken" % (m, bi, nb)
        lat = [res_cyc[i] - rl_cyc[i] for i in range(min(len(rl_cyc), len(res_cyc)))]
        return dict(cycles=cyc, stalls=stalls, drops=drops, first_acc=first_acc, last_acc=last_acc,
                    beat_span=(last_acc - first_acc + 1) if first_acc is not None else 0, lat=lat,
                    rl_cyc=rl_cyc, res_cyc=res_cyc)


async def setup(dut, salt: int):
    cfg = cfg_from_env()
    seed = int(os.environ.get("TSR_SEED", "1"), 0)
    rnd = random.Random(seed * 7919 + salt)
    b = Bench(dut, cfg, rnd)
    await b.start()
    dut._log.info("config %s seed %d", cfg, seed)
    return cfg, rnd, b


# =============================================================================================
# cocotb tests
# =============================================================================================
@cocotb.test()
async def back_actbuf(dut):
    """Activation buffer contents and read timing through its read ports (testbench-driven)."""
    cfg, rnd, b = await setup(dut, 1)
    tmax, kmax, lat = cfg["T_MAX"], cfg["K_MAX"], cfg["ACT_RD_LAT"]
    qs_words = (kmax + 127) // 128
    checked = 0
    for fmt in (0, 1):
        kind = mm.qa_kind(fmt)
        for K in sorted({kmax, kmax - mm.block_size(fmt), mm.block_size(fmt) * 3} - {0}):
            if K <= 0 or K > kmax:
                continue
            qas = [mm.rand_qa(kind, K, rnd, "raw", special=0.3) for _ in range(tmax)]
            lines = [isa.MVM_BCLINE_T.pack(**ln) for ln in mm.bc_lines(kind, K, qas)]
            dut.tb_rd.value = 1
            for ln in lines:
                dut.wr_valid.value = 1
                dut.wr_line.value = ln
                await RisingEdge(dut.clk)
            dut.wr_valid.value = 0
            dut.tb_qs_re.value = 0
            dut.tb_dbs_re.value = 0
            await RisingEdge(dut.clk)        # write -> read distance >= 2 (one write input register)
            imgs = [mm.actbuf_image(kind, K, q) for q in qas]
            ql = imgs[0]["qs_lines"]
            nw, nd = (ql + 1) // 2, len(imgs[0]["dbs"])
            # read plan: sequential back-to-back, then random addresses with gaps
            plan = [(w % nw, w % nd) for w in range(max(nw, nd))]
            plan += [(rnd.randrange(nw), rnd.randrange(nd)) if rnd.random() < 0.7 else None for _ in range(200)]
            pend = deque()
            cyc = 0
            for item in plan + [None] * (lat + 2):
                if item is None:
                    dut.tb_qs_re.value = 0
                    dut.tb_dbs_re.value = 0
                else:
                    dut.tb_qs_re.value = 1
                    dut.tb_dbs_re.value = 1
                    dut.tb_qs_addr.value = item[0]
                    dut.tb_dbs_addr.value = item[1]
                    pend.append((cyc + lat, item))
                await FallingEdge(dut.clk)
                if pend and pend[0][0] == cyc:
                    _, (w, a) = pend.popleft()
                    qd, dd = int(dut.qs_data.value), int(dut.dbs_data.value)
                    for t in range(tmax):
                        got_q = (qd >> (1024 * t)) & ((1 << 1024) - 1)
                        exp_q = imgs[t]["qs"][w]
                        mask = (1 << 1024) - 1 if 2 * w + 1 < ql else (1 << 512) - 1
                        assert got_q & mask == exp_q & mask, "fmt %d K %d token %d QS word %d differs" % (fmt, K, t, w)
                        got_d = (dd >> (512 * t)) & ((1 << 512) - 1)
                        assert got_d == imgs[t]["dbs"][a], "fmt %d K %d token %d DBS line %d differs" % (fmt, K, t, a)
                        checked += 1
                await RisingEdge(dut.clk)
                cyc += 1
            assert not pend
            assert nw <= qs_words
    # write -> read distance: a line written in cycle c is read back by a read in cycle c + 2
    dut.tb_rd.value = 1
    for rep in range(20):
        t, w = rnd.randrange(tmax), rnd.randrange(qs_words)
        data = rnd.getrandbits(512)
        dut.wr_valid.value = 1
        dut.wr_line.value = isa.MVM_BCLINE_T.pack(last=0, tok=t, dbs=0, addr=2 * w, data=data)
        await RisingEdge(dut.clk)
        dut.wr_valid.value = 0
        dut.tb_qs_re.value = 0
        await RisingEdge(dut.clk)
        dut.tb_qs_re.value = 1
        dut.tb_qs_addr.value = w
        await RisingEdge(dut.clk)
        dut.tb_qs_re.value = 0
        for _ in range(lat - 1):
            await RisingEdge(dut.clk)
        await FallingEdge(dut.clk)
        got = (int(dut.qs_data.value) >> (1024 * t)) & ((1 << 512) - 1)
        assert got == data, "write -> read (2 cycles) returned stale data"
        await RisingEdge(dut.clk)
    dut.tb_rd.value = 0
    dut._log.info("actbuf: %d token-reads compared with actbuf_image, read latency %d, write->read 2 cycles",
                  checked, lat)


@cocotb.test()
async def back_directed(dut):
    """K sweep over block multiples / partial chunks, both formats, T 1..T_MAX, data modes cycled."""
    cfg, rnd, b = await setup(dut, 2)
    tmax, kmax = cfg["T_MAX"], cfg["K_MAX"]
    modes = [("pack", "qact"), ("raw", "raw"), ("pack_special", "qact_special"), ("raw", "raw_special"),
             ("mix", "mix")]
    n = 0
    for fmt in (0, 1):
        for i, K in enumerate(k_choices(fmt, kmax)):
            rm, qm = modes[(i + fmt) % len(modes)]
            nrows = 1 + (i % 4)
            if K > 8192:
                nrows = 1 + (i % 2)
            T = 1 + (i % tmax)
            m = Mvm(fmt, K, gen_rows(fmt, K, nrows, rnd, rm), gen_qas(fmt, K, T, rnd, qm), "dir %s/%s" % (rm, qm))
            st = await b.run_mvm(m, p_valid=0.85, p_ready=0.8, bc_gap=0.1)
            n += 1
            dut._log.info("%s: %d cycles, %d stalls", m, st["cycles"], st["stalls"])
    dut._log.info("directed: %d MVMs checked", n)


@cocotb.test()
async def back_special(dut):
    """Constructed numeric extremes (module docstring)."""
    cfg, rnd, b = await setup(dut, 3)
    for m in special_cases(rnd, cfg["K_MAX"], cfg["T_MAX"]):
        st = await b.run_mvm(m, p_valid=0.9, p_ready=0.7)
        dut._log.info("%s: ok (%d cycles)", m, st["cycles"])


@cocotb.test()
async def back_random(dut):
    """Random MVMs with random gaps / backpressure, back to back."""
    cfg, rnd, b = await setup(dut, 4)
    tmax, kmax = cfg["T_MAX"], cfg["K_MAX"]
    nmvm = max(4, int(40 * scale()))
    tot_rows = 0
    for i in range(nmvm):
        fmt = rnd.randrange(2)
        bs = mm.block_size(fmt)
        cap = kmax if rnd.random() < 0.1 else min(kmax, 6144)
        K = bs * rnd.randint(1, cap // bs)
        rb = mm.row_beats(fmt, K)
        nrows = rnd.randint(1, max(1, min(14, 2400 // rb)))
        T = rnd.randint(1, tmax)
        p_valid = rnd.choice((1.0, 1.0, 0.9, 0.6, 0.3))
        p_ready = rnd.choice((1.0, 1.0, 0.8, 0.5, 0.1))
        m = Mvm(fmt, K, gen_rows(fmt, K, nrows, rnd), gen_qas(fmt, K, T, rnd), "rand%d" % i)
        st = await b.run_mvm(m, p_valid=p_valid, p_ready=p_ready, bc_gap=rnd.choice((0.0, 0.3)))
        tot_rows += nrows
        if rnd.random() < 0.3:
            await ClockCycles(dut.clk, rnd.randint(1, 30))
    dut._log.info("random: %d MVMs, %d rows checked", nmvm, tot_rows)


def schedule(cfg: dict) -> dict:
    """The tsr_mvm_back schedule (its header): accept -> accumulate issue, row_last -> result."""
    m0 = 1 + cfg["ACT_RD_LAT"]
    g = m0 + cfg["MAC_LAT"]
    fi, fd = g + 6, m0 + cfg["MUL_LAT"]
    p = max(fi, fd) + cfg["MUL_LAT"]
    a4 = p + cfg["ADD_LAT"]
    hold = a4 + 5 * cfg["ADD_LAT"] + 10
    return dict(A8=p, A4=a4, lat8=hold - cfg["ADD_LAT"], lat4=hold, depth=(hold + 7) // 8 + 1)


@cocotb.test()
async def back_throughput(dut):
    """RB_MIN: no beat_ready drop with beats offered every cycle and res_ready = 1."""
    cfg, rnd, b = await setup(dut, 5)
    tmax, kmax = cfg["T_MAX"], cfg["K_MAX"]
    sch = schedule(cfg)
    dut._log.info("schedule %s", sch)
    lines = []
    # the RB_MIN rows that fit K_MAX (K_MAX < 1024: no Q4_K row has RB_MIN beats, K_MAX 256: no row)
    long_cases = [(f, K) for f, K in ((0, 480), (0, 512), (0, 1024), (0, 1056), (1, 1024), (1, 2304), (0, kmax),
                                      (1, kmax)) if K <= kmax and mm.row_beats(f, K) >= RB_MIN]
    for fmt in (0, 1):
        if not any(f == fmt for f, _ in long_cases):
            lines.append("%s: no row of >= RB_MIN = %d beats fits K_MAX %d (longest: %d beats): RB_MIN not "
                         "measured" % (("Q8_0", "Q4_K")[fmt], RB_MIN, kmax, mm.row_beats(fmt, kmax)))
    for T in sorted({1, tmax}):
        for fmt, K in long_cases:
            rb = mm.row_beats(fmt, K)
            nrows = max(4, min(40, 3000 // rb))
            m = Mvm(fmt, K, gen_rows(fmt, K, nrows, rnd, "raw"), gen_qas(fmt, K, T, rnd, "raw"), "tp")
            st = await b.run_mvm(m, p_valid=1.0, p_ready=1.0, no_stall=True)
            nq = len(m.packed)
            assert st["beat_span"] == nq, "%s: %d cycles for %d beats" % (m, st["beat_span"], nq)
            want = sch["lat4"] if fmt else sch["lat8"]
            assert set(st["lat"]) == {want}, "%s: row_last -> result latency %s, expected %d" % (
                m, sorted(set(st["lat"])), want)
            lines.append("%s T %d: %d quant beats in %d cycles, 0 drops, latency row_last->result %s" % (
                m, T, nq, st["beat_span"], sorted(set(st["lat"]))))
        # ROW_BEATS 8 (Q8_0 K 448 / 416): with the front end's META cycle no drop; without it one
        # bubble per row (the measurement discriminates)
        for K in (448, 416):
            if K > kmax:
                continue
            m = Mvm(0, K, gen_rows(0, K, 30, rnd, "raw"), gen_qas(0, K, T, rnd, "raw"), "rb8-meta")
            st = await b.run_mvm(m, p_valid=1.0, p_ready=1.0, meta_gaps=True)
            assert st["stalls"] == 0, "%s with META cycles: %d stalled beats" % (m, st["stalls"])
            assert st["beat_span"] == 30 * 8 - 1, "%s with META cycles: %d cycles" % (m, st["beat_span"])
            lines.append("%s T %d with META cycles: 0 stalled beats, %d cycles for 30 rows (8 per row); "
                         "beat_ready low in %d META cycles (no beat offered)" % (m, T, st["beat_span"], st["drops"]))
            m = Mvm(0, K, gen_rows(0, K, 30, rnd, "raw"), gen_qas(0, K, T, rnd, "raw"), "rb8-nometa")
            st = await b.run_mvm(m, p_valid=1.0, p_ready=1.0)
            assert st["stalls"] == 29, "%s back to back: expected 29 one-cycle stalls, got %d" % (m, st["stalls"])
            lines.append("%s T %d back to back: %d stalls (one per row start), %d cycles" % (
                m, T, st["stalls"], st["beat_span"]))
        # short rows: one row per 8 cycles
        for fmt, K in ((0, 32), (0, 64), (0, 96), (0, 224), (1, 256), (1, 512), (1, 768)):
            if K > kmax:
                continue
            nrows = 24
            m = Mvm(fmt, K, gen_rows(fmt, K, nrows, rnd, "raw"), gen_qas(fmt, K, T, rnd, "raw"), "short")
            st = await b.run_mvm(m, p_valid=1.0, p_ready=1.0)
            rate = (st["rl_cyc"][-1] - st["rl_cyc"][0]) / (nrows - 1)
            assert rate == 8.0, "%s: %.2f cycles per row, expected 8" % (m, rate)
            lines.append("%s T %d: %.2f cycles per row (row_last spacing)" % (m, T, rate))
    for ln in lines:
        dut._log.info("throughput: %s", ln)


@cocotb.test()
async def back_backpressure(dut):
    """Result-space reservation and long gaps:
    (a) res_ready = 0 for a long phase: exactly RES_DEPTH rows are taken (their row_last beats),
        then beat_ready stays 0; after the release every result is exact;
    (b) Q4_K rows with superblocks s >= 16 (K > 4096) and beat gaps of up to 2 ADD_LAT + 4 cycles,
        so the two halves of a superblock (and Q8_0 terms of one partial) are further apart than
        the adder latency (holds of part0 / Dx, partial read-after-write over idle cycles);
    (c) Q8_0 odd block counts whose unused D element (block nb, in the D line's tail) is NaN / Inf:
        the absent block b+1 must not contribute."""
    cfg, rnd, b = await setup(dut, 8)
    tmax, kmax = cfg["T_MAX"], cfg["K_MAX"]
    sch = schedule(cfg)
    depth = sch["depth"]
    # (a)
    for fmt, K in ((0, 64), (1, 256), (0, min(kmax, 1024)), (1, min(kmax, 1024))):
        nrows = depth + 6
        m = Mvm(fmt, K, gen_rows(fmt, K, nrows, rnd), gen_qas(fmt, K, tmax, rnd), "bp-hold")
        hold = nrows * (mm.row_beats(fmt, K) + 8) + sch["lat4"] + 200
        seen = {}
        st = await b.run_mvm(m, p_valid=1.0, p_ready=1.0, ready_hold=hold, hold_probe=seen.update)
        assert seen["row_lasts"] == depth, "%s: %d rows taken while res_ready = 0, RES_DEPTH %d" % (
            m, seen["row_lasts"], depth)
        assert seen["rows_out"] == 0
        dut._log.info("%s: %d rows reserved while res_ready = 0 (RES_DEPTH %d), %d stalled cycles", m,
                      seen["row_lasts"], depth, st["stalls"])
    # (b)
    gmax = 2 * cfg["ADD_LAT"] + 4
    for fmt, K in ((1, min(kmax, 8192)), (1, min(kmax, 4352)), (0, min(kmax, 2080))):
        m = Mvm(fmt, K, gen_rows(fmt, K, 3, rnd), gen_qas(fmt, K, tmax, rnd), "gaps")
        await b.run_mvm(m, p_valid=1.0, p_ready=0.7, gap_max=gmax)
        dut._log.info("%s: beat gaps 0..%d cycles: ok", m, gmax)
    # (c)
    for K in (96, 1056, 2080, 32 * 15):
        if K > kmax:
            continue
        nb = K // 32
        assert nb % 2 == 1
        qas = []
        for t in range(tmax):
            q = bytearray(mm.rand_qa(mm.QA_A32, K, rnd, "raw"))
            o = 64 * mm.qa_qs_lines(K) + 4 * nb
            q[o:o + 4] = rnd.choice((0x7FC00000, 0x7F800000, 0xFF800000, 0x7F800001)).to_bytes(4, "little")
            qas.append(bytes(q))
        m = Mvm(0, K, gen_rows(0, K, 3, rnd), qas, "odd-nb NaN tail")
        await b.run_mvm(m, p_valid=0.8, p_ready=0.8)
    dut._log.info("backpressure / gaps / unused D tail: ok")


@cocotb.test()
async def back_reset(dut):
    """Synchronous reset in the middle of MVMs, then fully checked MVMs."""
    cfg, rnd, b = await setup(dut, 6)
    tmax, kmax = cfg["T_MAX"], cfg["K_MAX"]
    for rep in range(4):
        fmt = rep % 2
        K = min(kmax, (1024, 2304)[fmt] if rep < 2 else (480, 1024)[fmt])
        rb = mm.row_beats(fmt, K)
        nrows = max(3, min(16, 1600 // rb))
        m = Mvm(fmt, K, gen_rows(fmt, K, nrows, rnd), gen_qas(fmt, K, tmax, rnd), "pre-reset")
        p_ready = 0.0 if rep == 3 else 0.5
        await b.run_mvm(m, p_valid=0.9, p_ready=p_ready, abort_after=rnd.randint(rb, nrows * rb))
        await RisingEdge(dut.clk)
        dut.rst_n.value = 0
        dut.beat_valid.value = 0
        dut.res_ready.value = 0
        await ClockCycles(dut.clk, 3)
        await FallingEdge(dut.clk)
        assert int(dut.beat_ready.value) == 0, "beat_ready during reset"
        assert int(dut.res_valid.value) == 0, "res_valid during reset"
        assert int(dut.idle.value) == 1, "idle during reset"
        assert int(dut.be_qs_re.value) == 0 and int(dut.be_dbs_re.value) == 0, "reads during reset"
        await RisingEdge(dut.clk)
        dut.rst_n.value = 1
        for i in range(3):
            fmt2 = rnd.randrange(2)
            K2 = min(kmax, rnd.choice((mm.block_size(fmt2), 1024 + mm.block_size(fmt2), 2048)))
            m2 = Mvm(fmt2, K2, gen_rows(fmt2, K2, rnd.randint(1, 5), rnd),
                     gen_qas(fmt2, K2, rnd.randint(1, tmax), rnd), "post-reset")
            await b.run_mvm(m2, p_valid=0.8, p_ready=0.7)
    dut._log.info("reset: 4 mid-MVM resets (one with a full result FIFO), then checked MVMs")


@cocotb.test()
async def back_comb_probe(dut):
    """No combinational path from any input to any output (R5)."""
    cfg, rnd, b = await setup(dut, 7)
    K = min(cfg["K_MAX"], 2304)
    m = Mvm(1, K, gen_rows(1, K, 6, rnd), gen_qas(1, K, cfg["T_MAX"], rnd), "probe")
    task = cocotb.start_soon(b.run_mvm(m, p_valid=0.7, p_ready=0.5))
    pokes = 0
    bad = set()
    for _ in range(4):
        await ClockCycles(dut.clk, rnd.randint(5, 25))
        await FallingEdge(dut.clk)
        b.clock.stop()
        await Timer(1, "ns")
        outs = [(n, getattr(dut, n)) for n in OUTPUTS]
        snap = {n: s.value for n, s in outs}
        for name in INPUTS:
            sig = getattr(dut, name)
            saved = sig.value
            for _ in range(2):
                sig.value = rnd.getrandbits(len(sig))
                pokes += 1
                await Timer(1, "ns")
                for n, s in outs:
                    if s.value != snap[n]:
                        bad.add((name, n))
            sig.value = saved
            await Timer(1, "ns")
        for n, s in outs:
            assert s.value == snap[n], "output %s did not return to its value after the probe" % n
        b.clock.start(start_high=True)
    assert not bad, "combinational input->output paths: %s" % sorted(bad)
    await task
    dut._log.info("comb probe: %d input pokes, no input->output path; the MVM still completed exactly", pokes)


# =============================================================================================
# tsr_mac_lane alone
# =============================================================================================
@cocotb.test()
async def mac_lane(dut):
    """Random / extreme operands, random ce and in_valid; out_valid and y against a LAT-deep model."""
    p = json.loads(os.environ["TSR_MAC_CFG"])
    N, T, LAT = p["N"], p["T"], p["LAT"]
    seed = int(os.environ.get("TSR_SEED", "1"), 0)
    rnd = random.Random(seed * 31 + N * 7 + T * 3 + LAT)
    dut.rst_n.value = 0
    dut.ce.value = 0
    dut.in_valid.value = 0
    dut.w.value = 0
    dut.x.value = 0
    clk = Clock(dut.clk, 10, unit="ns")
    clk.start(start_high=True)
    await ClockCycles(dut.clk, 3)
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    pipe = [(0, None)] * LAT          # model: [oldest .. newest] (valid, sums)
    checked = 0

    def operand(kind):
        if kind == 0:
            return rnd.randrange(-128, 128)
        if kind == 1:
            return rnd.choice((-128, 127, -127, 0, 1, -1))
        return -128

    for cyc in range(4000):
        kind = rnd.choice((0, 0, 1, 2)) if cyc % 500 else 2
        w = [operand(kind) for _ in range(N)]
        x = [[operand(kind) for _ in range(N)] for _ in range(T)]
        ce = 1 if rnd.random() < (0.7 if cyc % 1000 < 500 else 1.0) else 0
        iv = 1 if rnd.random() < 0.8 else 0
        dut.w.value = isa.pack_array(w, 8)
        dut.x.value = sum(isa.pack_array(x[t], 8) << (8 * N * t) for t in range(T))
        dut.ce.value = ce
        dut.in_valid.value = iv
        await FallingEdge(dut.clk)
        # outputs now reflect the pipeline after the previous edges
        ov = int(dut.out_valid.value)
        mv, msum = pipe[0]
        assert ov == mv, "cycle %d: out_valid %d, model %d" % (cyc, ov, mv)
        if mv:
            yv = int(dut.y.value)
            sw = 16 + max(0, (N - 1).bit_length())
            for t in range(T):
                got = (yv >> (sw * t)) & ((1 << sw) - 1)
                if got >> (sw - 1):
                    got -= 1 << sw
                assert got == msum[t], "cycle %d token %d: y %d, expected %d" % (cyc, t, got, msum[t])
                checked += 1
        await RisingEdge(dut.clk)
        if ce:
            sums = [sum(w[i] * x[t][i] for i in range(N)) for t in range(T)]
            pipe = pipe[1:] + [(iv, sums)]
    assert checked > 1000
    dut._log.info("mac_lane N %d T %d LAT %d: %d sums checked with random ce stalls", N, T, LAT, checked)


def mac_layers(w, x, N: int, T: int) -> list:
    """Node values of the generic tsr_mac_lane's logic layers: [layer][token][node]; layer 0 = the
    products (N padded to a power of two with zero products), layer i = adder level i."""
    lv = (N - 1).bit_length() if N > 1 else 0
    layers = [[[w[k] * x[t][k] if k < N else 0 for k in range(1 << lv)] for t in range(T)]]
    for _ in range(lv):
        prev = layers[-1]
        layers.append([[prev[t][2 * k] + prev[t][2 * k + 1] for k in range(len(prev[t]) // 2)] for t in range(T)])
    return layers


def mac_pack(vals, wi: int) -> int:
    """[token][node] -> the packed logic [T-1:0][NI-1:0][WI-1:0] value."""
    v, m = 0, (1 << wi) - 1
    for t, row in enumerate(vals):
        for k, e in enumerate(row):
            v |= (e & m) << (wi * (t * len(row) + k))
    return v


@cocotb.test()
async def mac_lane_structure(dut):
    """Register placement of the generic tsr_mac_lane (docs/rtl-m3.md §9.2 item 4: product register,
    about two adder levels per stage), measured on the layer signals g_l[i].d / g_l[i].q: operands A
    are held until the pipeline is full, then operands B; the number of clock edges after which a
    layer shows B's values is its register depth (every layer must show exactly A's or B's values).
    Checks: no register in front of the products (callers drive w / x from flops); for LAT >= 2 a
    product register; the adder levels spread over min(LAT - 1, LV) register segments as evenly as
    possible (no segment deeper than ceil(LV / segments)); y after exactly LAT edges; N = 32: the
    exact placement MAC_STAGES_N32 (default MAC_LAT 3: products | 2 levels | 3 levels)."""
    p = json.loads(os.environ["TSR_MAC_CFG"])
    N, T, LAT = p["N"], p["T"], p["LAT"]
    lv = (N - 1).bit_length() if N > 1 else 0
    sw = 16 + lv
    seed = int(os.environ.get("TSR_SEED", "1"), 0)
    rnd = random.Random(seed * 17 + N * 5 + T * 3 + LAT)
    dut.rst_n.value = 0
    dut.ce.value = 1
    dut.in_valid.value = 1
    clk = Clock(dut.clk, 10, unit="ns")
    clk.start(start_high=True)

    def operands():
        while True:
            w = [rnd.randrange(-128, 128) for _ in range(N)]
            x = [[rnd.randrange(-128, 128) for _ in range(N)] for _ in range(T)]
            ls = mac_layers(w, x, N, T)
            if all(any(e != 0 for row in layer for e in row) for layer in ls):
                return w, x, ls

    def drive(w, x):
        dut.w.value = isa.pack_array(w, 8)
        dut.x.value = sum(isa.pack_array(x[t], 8) << (8 * N * t) for t in range(T))

    wa, xa, la = operands()
    while True:
        wb, xb, lb = operands()
        if all(mac_pack(la[i], 16 + i) != mac_pack(lb[i], 16 + i) for i in range(lv + 1)):
            break
    drive(wa, xa)
    await ClockCycles(dut.clk, 3)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, LAT + lv + 4)
    await FallingEdge(dut.clk)
    drive(wb, xb)
    await Timer(1, "ns")
    sd, sq, sy = [None] * (lv + 1), [None] * (lv + 1), None
    ya, yb = mac_pack([[la[lv][t][0]] for t in range(T)], sw), mac_pack([[lb[lv][t][0]] for t in range(T)], sw)
    for e in range(LAT + 2):          # e = clock edges since B was applied
        for i in range(lv + 1):
            for kind, stage in (("d", sd), ("q", sq)):
                v = int(getattr(dut.g_l[i], kind).value)
                a, b = mac_pack(la[i], 16 + i), mac_pack(lb[i], 16 + i)
                assert v in (a, b), "g_l[%d].%s after %d edges: neither A's nor B's values" % (i, kind, e)
                if v == b and stage[i] is None:
                    stage[i] = e
                assert stage[i] is None or v == b, "g_l[%d].%s went back to A's values" % (i, kind)
        y = int(dut.y.value)
        assert y in (ya, yb), "y after %d edges: neither A's nor B's sum" % e
        if y == yb and sy is None:
            sy = e
        await RisingEdge(dut.clk)
        await FallingEdge(dut.clk)
    assert None not in sd and None not in sq and sy is not None, "a layer never showed B's values"
    desc = " | ".join("L%d:%d" % (i, sd[i]) for i in range(lv + 1))
    dut._log.info("mac_lane N %d LAT %d: register stages in front of each layer: %s; y after %d", N, LAT, desc, sy)
    assert sy == LAT, "y after %d edges, LAT %d" % (sy, LAT)
    assert sd[0] == 0, "a register in front of the products (stage %d)" % sd[0]
    for i in range(1, lv + 1):
        assert sd[i] == sq[i - 1]
    if lv >= 1:
        assert sq[0] == (1 if LAT >= 2 else 0), "product register: g_l[0].q at stage %d, LAT %d" % (sq[0], LAT)
        segs = {}
        for i in range(1, lv + 1):
            segs[sd[i]] = segs.get(sd[i], 0) + 1
        nseg = max(1, min(LAT - 1, lv))
        assert len(segs) == nseg, "adder levels in %d register segments, expected %d (%s)" % (len(segs), nseg, desc)
        assert max(segs.values()) == -(-lv // nseg), "unbalanced adder segments: %s" % desc
    if N == 32 and LAT in MAC_STAGES_N32:
        assert sd == MAC_STAGES_N32[LAT], "N 32 LAT %d: stages %s, expected %s" % (LAT, sd, MAC_STAGES_N32[LAT])
