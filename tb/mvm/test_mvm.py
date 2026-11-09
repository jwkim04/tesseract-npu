"""tsr_mvm tests: the whole MVM unit (docs/rtl-mvm.md §3.0-§3.2, §3.5, §6; §10.2 "top").

DUT: tb/mvm/hdl/tsr_tb_mvm.sv = tsr_mvm (NUM_CH clusters: front end, activation buffer, back end;
broadcast pipes, result pipes, gather) on SPM client 2 of a real tsr_spm and on the WS source of
one real tsr_mem_arb per channel, whose memory ports are served by common.mem_model.MemModel
(random latency / backpressure / stalls, protocol assertions, per-source beat counts).

Back end / activation buffer: the rtl/core modules are implemented in parallel by the back agent,
so the default build replaces them with the test doubles of tb/mvm/hdl/dbl (same module names and
ports): the activation-buffer double is a plain array (write / read checks), the back-end double
reads the activations through its ports and computes every row with the fp operations of
libtsr_ref through DPI-C (tsr_mvm_dbl.cpp), with random beat_ready stalls, random result latency
and a configurable result capacity (knobs written per MVM). test_mvm_real runs the same cocotb
tests with the real rtl/core modules (pytest skips it while they are still skeletons).

Oracle: common.mvm_model.mvm_reference (libtsr_ref dot rows + the tsr_sim.c ex_mvm write loop) on
a model of the SPM holding the same x images and old y. mvm_reference is tied to the
simulator's op-level API by tb/common test_mvm_model_vs_simulator. GOLDEN_DIGEST is pinned from the
simulator. Checks on every MVM:
  - the y lines (all tokens, incl. tails and token gaps) == the oracle's SPM, bit-exact
    (mvm_golden: through the digest of the DUT's y bytes against the pinned value);
  - SPM client 2 writes: every byte of the T y ranges written exactly once (byte enables), no
    other byte; reads: the broadcast reads x_line .. x_line + T*QA_LINES - 1 first and in order,
    then (ACC only) reads of y lines only; N = 0: no SPM access at all;
  - weight traffic (MemModel.src_beats): channel c reads exactly rows_c * ROW_BEATS WS beats
    (0 for rows_c = 0 and N = 0), nothing from other sources;
  - done: one registered pulse per command, never in the accepting cycle, and at done no SPM
    request / write pending, no WS request pending or beat in flight, no memory read outstanding
    (MemModel idle); cmd_ready only while idle;
  - perf_mvm_busy == 1 exactly from the cycle after the accepting edge through the done cycle;
  - the broadcast as seen by the first and the last cluster == mvm_model.ccmd / bc_lines (every
    field of every item, in order, the command first, never together with a line), and every
    cluster's front command == mvm_model.fcmd (none for rows_c = 0);
  - simulation assertions in the RTL and in the doubles fail the test.

cocotb tests (each configuration runs all of them):
  mvm_directed    N = 0 (also with ACC and T = T_MAX), N = 1, N < NUM_CH (rows_c = 0 clusters),
                  N = 5 on >= 4 channels (last channel empty), R % 16 != 0 (lines shared by two or
                  more channels), N % 16 != 0 (tails), N = 16 k, ACC with special old values (NaN
                  payloads, subnormals, signed zeros, Inf - Inf, overflow), T = T_MAX with
                  dst_stride gaps, T = 1 with a huge (unused) dst_stride, K = block size / partial
                  chunks / K_MAX, x and y at the SPM ends, W_OFF ending at the DRAM limit
  mvm_random      random MVMs back to back: fmt, K, N, T, ACC, strides, placements, raw / QACT
                  activations, special weights; random memory regimes and back-end knobs
  mvm_golden      a fixed MVM list (T = 1, K <= 2048, 64 KiB of SPM): the digest of the y bytes
                  read back from the DUT must equal the pinned GOLDEN_DIGEST (= the oracle's)
                  in every configuration (BC_PIPE / RES_PIPE / SPM_RD_LAT / NUM_CH / T_MAX)
  mvm_perf        ideal memory, ready back end, asserted: steady-state cycles per weight beat
                  (1 per cycle per cluster where the FIFO sizing rule holds; real back end: rows
                  >= RB_MIN), fixed accept -> done overhead (§9.1 budget; tight with the double),
                  N = 0 / N = 1 latency, gather line writes per cycle on 2-beat rows (double)
  mvm_b2b         MVM sequences issued back to back (accepted the cycle after the previous done),
                  incl. ACC MVMs onto the previous MVM's y; the whole sequence vs the oracle
  mvm_overlap     undefined overlaps (dst inside x, overlapping token ranges): must finish
                  normally, quiescent, writes only inside the y ranges (values only logged)
  mvm_reset       synchronous reset in the middle of MVMs, then fully checked MVMs
  mvm_bc_flush    one-cycle reset while a command / line occupies each broadcast stage; every
                  production cluster port is checked against independent old per-cluster pipes
  mvm_cycles      cycle exactness (CC-25: "cycle-identical, so every replay stays exact"): a fixed MVM list
                  under a fixed memory model (the channels' random generators re-seeded, so $TSR_SEED does not
                  matter), in the fast and the random regime; the digest of every MVM's {accept -> done,
                  busy, first / last write relative to the accept, SPM writes} must equal the pinned
                  CYCLE_DIGEST of the configuration (a timing change of the weight path, however small, fails
                  here; re-pin with the change understood). The same digests hold with HAS_ATT = 1 (the
                  attention modes add no cycle to the weight path): test_mvm_real_att
Configurations (CFGS): NUM_CH 1 / 3 / 4 / 16, T_MAX 1 / 2 / 3 / 8, BC_PIPE and RES_PIPE 0 / 1 / 3,
SPM_RD_LAT 1 / 2 / 3 / 4, ADD_LAT 1 / 3 / 5 / 8, WS FIFO 64..512, RD_OUTSTANDING 2..16.
Environment: TSR_MVM_TESTS=a,b (subset), TSR_MVM_SCALE (random test size factor, default 1.0),
TSR_MVM_BUILD_ARGS (extra Verilator arguments, e.g. -DSYNTHESIS to drop the RTL's simulation
assertions and show that the testbench checks alone catch a fault), TSR_MVM_VERBOSE=1.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import struct
import tempfile
from pathlib import Path

import cocotb
import pytest
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, FallingEdge, RisingEdge, Timer

from common import isa
from common import mvm_model as mm
from common.mem_model import MemModel
from common.runner import REPO, build_root, rtl_sources, run
from common.tbutil import seed as tb_seed

HDL = REPO / "tb" / "mvm" / "hdl"
DBL = HDL / "dbl"
TB_SV = HDL / "tsr_tb_mvm.sv"
REF_A = REPO / "build" / "dev" / "sw" / "sim" / "libtsr_ref.a"

CFGS = {
    # U200-like: 4 channels, T_MAX 1, K_MAX 32768, default latencies and pipes (smaller SPM)
    "c4":    dict(NUM_CH=4, T_MAX=1, K_MAX=32768, SPM_BYTES=262144, SPM_RD_LAT=2, ADDR_BEAT_W=28,
                  WS_FIFO_BEATS=512, BC_PIPE=1, RES_PIPE=1, ACT_RD_LAT=2, ADD_LAT=5, WS_RD_LAT=2,
                  RD_OUTSTANDING=16),
    "c3t3":  dict(NUM_CH=3, T_MAX=3, K_MAX=4096, SPM_BYTES=262144, SPM_RD_LAT=4, ADDR_BEAT_W=22,
                  WS_FIFO_BEATS=128, BC_PIPE=3, RES_PIPE=3, ACT_RD_LAT=3, ADD_LAT=8, WS_RD_LAT=4,
                  RD_OUTSTANDING=4),
    "c1t8":  dict(NUM_CH=1, T_MAX=8, K_MAX=2048, SPM_BYTES=65536, SPM_RD_LAT=1, ADDR_BEAT_W=22,
                  WS_FIFO_BEATS=64, BC_PIPE=0, RES_PIPE=0, ACT_RD_LAT=1, ADD_LAT=1, WS_RD_LAT=1,
                  RD_OUTSTANDING=2),
    "c16t2": dict(NUM_CH=16, T_MAX=2, K_MAX=8192, SPM_BYTES=262144, SPM_RD_LAT=3, ADDR_BEAT_W=24,
                  WS_FIFO_BEATS=64, BC_PIPE=1, RES_PIPE=0, ACT_RD_LAT=2, ADD_LAT=3, WS_RD_LAT=2,
                  RD_OUTSTANDING=16),
    "c4t8":  dict(NUM_CH=4, T_MAX=8, K_MAX=4096, SPM_BYTES=262144, SPM_RD_LAT=2, ADDR_BEAT_W=26,
                  WS_FIFO_BEATS=256, BC_PIPE=0, RES_PIPE=3, ACT_RD_LAT=2, ADD_LAT=5, WS_RD_LAT=3,
                  RD_OUTSTANDING=8),
}
REAL_CFGS = tuple(CFGS)          # the same tests with the real tsr_mvm_back / tsr_mvm_actbuf
ATT_CFGS = ("c4", "c3t3")        # ... and with HAS_ATT = 1 (non-power-of-two NUM_CH and T_MAX 3 among them)

TESTS = ["mvm_directed", "mvm_random", "mvm_golden", "mvm_b2b", "mvm_overlap", "mvm_perf", "mvm_reset", "mvm_cycles", "mvm_bc_flush"]
ATT_TESTS = ["mvm_directed", "mvm_b2b", "mvm_perf", "mvm_cycles"]   # HAS_ATT = 1: the weight path unchanged

W_LO = 1 << 21             # lowest weight beat (every channel, same W_OFF; was above the simulator's staging area)


# =============================================================================================
# pytest side
# =============================================================================================
def _ref_archive() -> Path:
    """A private, content-addressed copy of libtsr_ref.a for the DPI shim (a rebuild of build/dev
    while a test links cannot change it)."""
    if not REF_A.exists():
        pytest.skip("%s missing (cmake --build build/dev --target tsr_ref)" % REF_A)
    data = REF_A.read_bytes()
    d = build_root() / "tsr_ref_snap" / hashlib.sha256(data).hexdigest()[:16]
    dst = d / "libtsr_ref.a"
    if not dst.exists():
        d.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=d)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, dst)
    return dst


def build_spec(real: bool):
    """(sources, build_args): the rtl filelist with the back end / activation buffer replaced by
    the doubles (real=False), plus the testbench toplevel."""
    srcs = []
    for s in rtl_sources():
        if not real and s.name in ("tsr_mvm_back.sv", "tsr_mvm_actbuf.sv"):
            s = DBL / s.name
        srcs.append(s)
    srcs += [HDL / "tsr_mvm_bc_observer.sv", TB_SV]
    if real:
        return srcs, []
    return srcs, [str(DBL / "tsr_mvm_dbl.cpp"), str(_ref_archive()),
                  "-CFLAGS", "-I%s" % (REPO / "sw" / "sim" / "include")]


def is_skeleton(path: Path) -> bool:
    return "SKELETON" in path.read_text()


def _run(cfg: str, real: bool, att: bool = False, *, overrides=None, only=None, suffix=""):
    p = dict(CFGS[cfg], **(overrides or {}))
    sel = os.environ.get("TSR_MVM_TESTS")
    tests = [t for t in (only or (ATT_TESTS if att else TESTS)) if not sel or t in sel.split(",")]
    if not real:
        tests = [t for t in tests if t != "mvm_cycles"]   # the double back end is not the weight path's timing
    srcs, bargs = build_spec(real)
    extra = os.environ.get("TSR_MVM_BUILD_ARGS", "").split()   # e.g. -DSYNTHESIS (no RTL assertions)
    run("tsr_tb_mvm", __name__, parameters=dict(p, HAS_ATT=1) if att else p, sources=srcs,
        build_args=bargs + extra, testcase=tests,
        name="tsr_tb_mvm_%s%s%s%s%s" % ("real_" if real else "", cfg, "_att" if att else "", "_x" if extra else "", suffix),
        extra_env={"TSR_MVM_CFG": json.dumps(dict(p, name=cfg, real=real, att=att))})


@pytest.mark.parametrize("cfg", list(CFGS))
def test_mvm(cfg):
    _run(cfg, real=False)


@pytest.mark.parametrize("cfg", REAL_CFGS)
def test_mvm_real(cfg):
    core = REPO / "rtl" / "core"
    if is_skeleton(core / "tsr_mvm_back.sv") or is_skeleton(core / "tsr_mvm_actbuf.sv"):
        pytest.skip("rtl/core tsr_mvm_back / tsr_mvm_actbuf are still skeletons")
    _run(cfg, real=True)


@pytest.mark.parametrize("cfg", ATT_CFGS)
def test_mvm_real_att(cfg):
    """The same weight-path checks and the same cycle digests with the attention modes built in."""
    _run(cfg, real=True, att=True)


def test_mvm_bc_reset_cycles():
    """Adding local reset copies keeps the existing c3t3 cycle digest, including BC_PIPE 3."""
    _run("c3t3", real=True, att=True, overrides=dict(RST_LOCAL=2),
         only=["mvm_cycles", "mvm_bc_flush"], suffix="_rst2")


def test_mvm_bc_u200():
    """Four channels with BC_PIPE 2, RST_LOCAL 2 and HAS_ATT; compare all cluster ports each cycle."""
    _run("c4", real=True, att=True, overrides=dict(BC_PIPE=2, RST_LOCAL=2),
         only=["mvm_directed", "mvm_bc_flush", "mvm_perf"], suffix="_bc2_rst2")


def test_mvm_u200_t4():
    """Four-cluster T4 arithmetic and overlapping commands against the C oracle."""
    _run("c4", real=True, att=True,
         overrides=dict(T_MAX=4, SPM_RD_LAT=4, BC_PIPE=2, RES_PIPE=2, MAC_LAT=9, RST_LOCAL=2),
         only=["mvm_directed", "mvm_b2b"], suffix="_u200_t4")


# =============================================================================================
# Configuration helpers
# =============================================================================================
def cfg_from_env() -> dict:
    return json.loads(os.environ["TSR_MVM_CFG"])


def scale() -> float:
    return float(os.environ.get("TSR_MVM_SCALE", "1.0"))


MEM_REGIMES = {
    "rand": dict(rd_lat=(1, 40), p_rd_req_ready=0.7, p_rd_rsp_valid=0.8, p_stall=0.01, stall_len=(1, 30)),
    "fast": dict(rd_lat=(0, 1), p_rd_req_ready=1.0, p_rd_rsp_valid=1.0, p_stall=0.0, stall_len=(1, 1)),
    "slow": dict(rd_lat=(20, 160), p_rd_req_ready=0.3, p_rd_rsp_valid=0.5, p_stall=0.02, stall_len=(5, 60)),
    "burst": dict(rd_lat=(1, 8), p_rd_req_ready=0.9, p_rd_rsp_valid=0.95, p_stall=0.03, stall_len=(10, 80)),
}


# =============================================================================================
# MVM description and data
# =============================================================================================
def y_lines(N: int) -> int:
    return (N + 15) // 16


def resolved_stride(m: dict) -> int:
    return m["stride"] or y_lines(m["N"])


def mvm_cmd(m: dict, num_ch: int) -> int:
    N, K, fmt = m["N"], m["K"], m["fmt"]
    return isa.MVM_CMD_T.pack(
        fmt=fmt, acc=int(m["acc"]), dst_line=m["dst"], x_line=m["x"], w_off=m["w_off"], n=N, k=K,
        t=m["T"], dst_stride=resolved_stride(m) if N else 0, y_lines=y_lines(N),
        rows_per_ch=-(-N // num_ch), row_beats=mm.row_beats(fmt, K),
        qa_lines=isa.qa_lines(mm.qa_kind(fmt), K))


def rand_y_old(rnd: random.Random, nbytes: int, special: float) -> bytes:
    out = bytearray()
    for _ in range(nbytes // 4):
        out += struct.pack("<I", mm.rand_f32(rnd, special, -30, 30))
    return bytes(out)


ACC_SPECIAL_OLD = (0x7FC00000, 0xFFC00001, 0x7F800001, 0x7FBFFFFF, 0x7F800000, 0xFF800000,
                   0x00000000, 0x80000000, 0x00000001, 0x807FFFFF, 0x00800000, 0x80800000,
                   0x7F7FFFFF, 0xFF7FFFFF, 0x3F800000, 0xBF800000)


def make_mvm(rnd: random.Random, p: dict, *, fmt=None, K=None, N=None, T=None, acc=None,
             stride=None, x=None, dst=None, w_off=None, rows_mode="mix", qa_mode=None,
             y_special=0.1, kmax_eff=None) -> dict:
    """A legal MVM for configuration p with its data (rows: repacked row images; qas: QA images;
    y_old: the dst span's previous bytes)."""
    nch, L = p["NUM_CH"], p["SPM_BYTES"] // 64
    limit = 1 << p["ADDR_BEAT_W"]
    fmt = rnd.randrange(2) if fmt is None else fmt
    bs = mm.block_size(fmt)
    kmax = min(p["K_MAX"], kmax_eff or p["K_MAX"])
    if K is None:
        K = rnd.choice([bs, 2 * bs, 3 * bs, 4 * bs, 5 * bs, 1024, 1024 + bs, rnd.randrange(1, 9) * bs,
                        rnd.randrange(1, kmax // bs + 1) * bs])
        K = min(K, kmax - kmax % bs)
    T = rnd.randint(1, p["T_MAX"]) if T is None else T
    acc = bool(rnd.getrandbits(1)) if acc is None else acc
    if N is None:
        N = rnd.choice([1, 2, 3, nch - 1 or 1, nch, nch + 1, 5, 15, 16, 17, 31, 33, 16 * nch + 3,
                        17 * nch, rnd.randint(1, 120)])
    yl = y_lines(N)
    if stride is None:
        if N == 0:
            stride = 0
        elif T > 1:
            stride = rnd.choice([0, yl + rnd.randint(0, 3), yl + rnd.randint(0, 40)])
        else:
            stride = rnd.choice([0, 0, yl, rnd.getrandbits(32)])
    ds = stride or yl
    kind = mm.qa_kind(fmt)
    ql = isa.qa_lines(kind, K)
    dspan = (T - 1) * ds + yl if N else 0
    xspan = T * ql
    free = L - xspan - dspan
    assert free >= 0, "MVM does not fit the SPM"
    # non-overlapping placement: x before or after the y span, random gaps, sometimes flush with
    # the SPM start / end
    g0 = rnd.choice([0, rnd.randint(0, free)])
    g1 = rnd.choice([0, rnd.randint(0, free - g0), free - g0])
    if rnd.random() < 0.5:
        xa, da = g0, g0 + xspan + g1
    else:
        da, xa = g0, g0 + dspan + g1
    x = xa if x is None else x
    dst = da if dst is None else dst
    rb = mm.row_beats(fmt, K)
    R = -(-N // nch) if N else 0
    if w_off is None:
        span = R * rb
        lo = W_LO
        w_off = rnd.choice([lo, lo + rnd.randrange(64), limit - span, rnd.randrange(lo, limit - span + 1)])
    rows = []
    for _ in range(N):
        mode = rows_mode if rows_mode != "mix" else rnd.choice(("raw", "pack", "pack", "special"))
        if mode == "raw":
            rows.append(mm.rand_row(fmt, K, rnd, raw=True))
        elif mode == "special":
            rows.append(mm.rand_row(fmt, K, rnd, special=0.3))
        else:
            rows.append(mm.rand_row(fmt, K, rnd))
    qm = qa_mode or rnd.choice(("qact", "qact", "raw"))
    qas = [mm.rand_qa(kind, K, rnd, qm, special=0.05) for _ in range(T)]
    y_old = rand_y_old(rnd, 64 * dspan, y_special) if acc else rnd.randbytes(64 * dspan)
    return dict(fmt=fmt, K=K, N=N, T=T, acc=acc, stride=stride, x=x, dst=dst, w_off=w_off,
                rows=rows, qas=qas, y_old=y_old)


def describe(m: dict) -> str:
    return "fmt %d K %d N %d T %d acc %d stride %d x %d dst %d w_off 0x%x" % (
        m["fmt"], m["K"], m["N"], m["T"], m["acc"], m["stride"], m["x"], m["dst"], m["w_off"])


# =============================================================================================
# Oracle
# =============================================================================================
class Oracle:
    """The same MVMs on a model of the SPM: mvm_model.mvm_reference (whole-MVM semantics of
    tsr_sim.c ex_mvm on libtsr_ref; every y computed before the first write) with the weights of
    each MVM as laid out in DRAM. The SPM model persists across calls, like the DUT's SPM."""

    def __init__(self, p: dict):
        self.p = p
        self.spm = bytearray(p["SPM_BYTES"])

    def _mvm(self, m: dict):
        nch = self.p["NUM_CH"]
        N, K, fmt = m["N"], m["K"], m["fmt"]
        if not N:
            return                                   # N = 0: no traffic, nothing written
        rows = dict(enumerate(mm.layout_channels(fmt, K, m["rows"], nch)))
        rd = lambda c, b, n: rows[c][64 * (b - m["w_off"]):64 * (b - m["w_off"] + n)]  # noqa: E731
        mm.mvm_reference(self.spm, rd, fmt=fmt, acc=m["acc"], dst=m["dst"], x=m["x"], w_off=m["w_off"],
                         N=N, K=K, T=m["T"], dst_stride=m["stride"], num_ch=nch)

    def _put(self, line: int, data: bytes):
        self.spm[64 * line:64 * line + len(data)] = data

    def run(self, m: dict) -> bytes:
        """One MVM after the x images and the old y are written: the y span afterwards."""
        self._put(m["x"], b"".join(m["qas"]))
        if m["y_old"]:
            self._put(m["dst"], m["y_old"])
        self._mvm(m)
        span = len(m["y_old"])
        return bytes(self.spm[64 * m["dst"]:64 * m["dst"] + span])

    def run_seq(self, ms: list, preload: list) -> bytes:
        """Several MVMs back to back after the preload regions [(spm_line, bytes)] are written (in
        order, later ones win). Returns the whole final SPM."""
        for line, data in preload:
            self._put(line, data)
        for m in ms:
            self._mvm(m)
        return bytes(self.spm)


# =============================================================================================
# Testbench
# =============================================================================================
class Monitor:
    """Samples the DUT at every falling edge: command / done / perf_mvm_busy protocol, SPM client
    2 traffic, WS requests; collects per-MVM records."""

    def __init__(self, dut, p: dict):
        self.dut, self.p = dut, p
        self.nch = p["NUM_CH"]
        self.cyc = 0
        self.hold = False          # the unit holds an instruction (cycle after accept .. done)
        self.pend = False
        self.errors = []
        self.clear()
        self.done_count = 0
        self.accept_cycle = None
        self.done_cycle = None
        self.writes_per_cycle = []

    def clear(self):
        self.wr = []               # (cycle, line, be)
        self.rd = []               # (cycle, line)
        self.ws_req = [0] * self.nch
        self.busy_cycles = 0
        self.first_wr = self.last_wr = None

    def err(self, msg):
        self.errors.append("cycle %d: %s" % (self.cyc, msg))
        raise AssertionError(self.errors[-1])

    async def run(self):
        d = self.dut
        while True:
            await FallingEdge(d.clk)
            self.cyc += 1
            if int(d.o_mvm_rst_n.value) == 0:
                self.hold = self.pend = False
                continue
            acc = int(d.cmd_valid.value) & int(d.cmd_ready.value)
            done = int(d.done.value)
            busy = int(d.perf_mvm_busy.value)
            if self.pend:
                self.hold, self.pend = True, False
            if busy != int(self.hold):
                self.err("perf_mvm_busy = %d, expected %d" % (busy, int(self.hold)))
            if busy:
                self.busy_cycles += 1
            if int(d.cmd_ready.value) and self.hold:
                self.err("cmd_ready while an instruction is held")
            wv = int(d.o_wr_valid.value) & int(d.o_wr_ready.value)
            rv = int(d.o_rd_req_valid.value) & int(d.o_rd_req_ready.value)
            wsv = int(d.o_ws_req_valid.value)
            wsr = int(d.o_ws_req_ready.value)
            if (wv or rv or wsv or int(d.o_rd_req_valid.value) or int(d.o_wr_valid.value)) and not self.hold:
                self.err("SPM / WS traffic while no instruction is held (wr %d rd %d ws %x)" % (wv, rv, wsv))
            if wv:
                self.wr.append((self.cyc, int(d.o_wr_line.value), int(d.o_wr_be.value)))
                if self.first_wr is None:
                    self.first_wr = self.cyc
                self.last_wr = self.cyc
            if rv:
                self.rd.append((self.cyc, int(d.o_rd_req_line.value)))
            if wsv & wsr:
                for c in range(self.nch):
                    if (wsv & wsr) >> c & 1:
                        self.ws_req[c] += 1
            if acc:
                if self.hold:
                    self.err("command accepted while one is held")
                if done:
                    self.err("done in the accepting cycle")
                self.pend = True
                self.accept_cycle = self.cyc
            if done:
                if not self.hold:
                    self.err("done without a held instruction")
                self.done_count += 1
                self.done_cycle = self.cyc
                # quiescence at done (docs/rtl-mvm.md §3.0, §6.4)
                if int(d.o_wr_valid.value) or int(d.o_rd_req_valid.value):
                    self.err("done with an SPM request pending")
                if wsv or int(d.o_ws_rsp_valid.value):
                    self.err("done with a WS request / beat pending (req %x rsp %x)" % (wsv, int(d.o_ws_rsp_valid.value)))
                if self.mem is not None and not self.mem.idle():
                    self.err("done with memory reads outstanding")
                self.hold = False

    mem = None


class BcMon:
    """The broadcast as the clusters see it (docs/rtl-mvm.md §3.1) and every cluster's front command
    (§3.2), observed inside the DUT: watched clusters (first and last) record every bc_cmd /
    bc_line item; every cluster records its tsr_mvm_fcmd_t handshakes."""

    def __init__(self, dut, nch: int):
        self.dut = dut
        self.cl = [dut.u_mvm.g_cl[c].u_cl for c in range(nch)]
        self.watch = list(range(nch))
        self.clear()

    def clear(self):
        self.items = {c: [] for c in self.watch}
        self.fcmds = {c: [] for c in range(len(self.cl))}

    async def run(self):
        d = self.dut
        while True:
            await FallingEdge(d.clk)
            if int(d.o_mvm_rst_n.value) == 0:
                continue
            for c in self.watch:
                cl = self.cl[c]
                cv, lv = int(cl.bc_cmd_valid.value), int(cl.bc_line_valid.value)
                if cv and lv:
                    raise AssertionError("cluster %d: bc_cmd and bc_line in one cycle" % c)
                if cv:
                    self.items[c].append(("cmd", int(cl.bc_cmd.value)))
                if lv:
                    self.items[c].append(("line", int(cl.bc_line.value)))
            for c, cl in enumerate(self.cl):
                if int(cl.fc_v_q.value) and int(cl.f_cmd_ready.value):
                    self.fcmds[c].append(int(cl.fc.value))

    def check(self, m: dict, nch: int):
        N, K, fmt = m["N"], m["K"], m["fmt"]
        if N == 0:
            want = []
        else:
            cc = mm.ccmd(fmt, K, N, nch, m["w_off"])
            want = [("cmd", isa.MVM_CCMD_T.pack(**cc))]
            want += [("line", isa.MVM_BCLINE_T.pack(**b)) for b in mm.bc_lines(mm.qa_kind(fmt), K, m["qas"])]
        for c in self.watch:
            got = self.items[c]
            if got != want:
                i = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))
                g = got[i] if i < len(got) else None
                w = want[i] if i < len(want) else None
                def fields(x):
                    if x is None:
                        return None
                    st = isa.MVM_CCMD_T if x[0] == "cmd" else isa.MVM_BCLINE_T
                    f = st.unpack(x[1])
                    if "data" in f:
                        f["data"] = "%x..." % (f["data"] & 0xFFFFFFFF)
                    return (x[0], f)
                raise AssertionError("cluster %d broadcast item %d of %d/%d: got %s, want %s (%s)" % (
                    c, i, len(got), len(want), fields(g), fields(w), describe(m)))
        for c in range(nch):
            fc = mm.fcmd(fmt, K, N, nch, c, m["w_off"]) if N else None
            want_f = [isa.MVM_FCMD_T.pack(**fc)] if fc else []
            if self.fcmds[c] != want_f:
                raise AssertionError("cluster %d front commands %s, want %s (%s)" % (
                    c, [isa.MVM_FCMD_T.unpack(x) for x in self.fcmds[c]], fc, describe(m)))


class Tb:
    def __init__(self, dut, p: dict):
        self.dut, self.p = dut, p
        self.nch = p["NUM_CH"]
        self.L = p["SPM_BYTES"] // 64
        self.limit = 1 << p["ADDR_BEAT_W"]
        self.mem = None
        self.mon = Monitor(dut, p)
        self.oracle = Oracle(p)
        self.real = bool(p.get("real"))
        self.stats = []

    async def start(self):
        d = self.dut
        d.rst_n.value = 0
        d.cmd_valid.value = 0
        d.cmd.value = 0
        d.h_rd_req_valid.value = 0
        d.h_rd_req_line.value = 0
        d.h_wr_valid.value = 0
        d.h_wr_line.value = 0
        d.h_wr_data.value = 0
        d.h_wr_be.value = 0
        await Timer(1, "ns")
        self.clock = Clock(d.clk, 10, unit="ns")
        self.clock.start(start_high=True)
        self.mem = MemModel(d, num_ch=self.nch, addr_w=self.p["ADDR_BEAT_W"], rst_n=d.o_mvm_rst_n,
                            rd_outstanding=self.p["RD_OUTSTANDING"], limit_beats=self.limit,
                            seed=tb_seed(), wr_order="any")
        self.mem.start()
        self.mon.mem = self.mem
        cocotb.start_soon(self.mon.run())
        self.bc = BcMon(d, self.nch)
        cocotb.start_soon(self.bc.run())
        await ClockCycles(d.clk, 5)
        await RisingEdge(d.clk)
        d.rst_n.value = 1
        await ClockCycles(d.clk, self.p.get("RST_LOCAL", 0) + 2)

    async def reset(self, cycles: int = 4, *, start_now: bool = False):
        d = self.dut
        if not start_now:
            await FallingEdge(d.clk)
        d.rst_n.value = 0
        d.cmd_valid.value = 0
        d.h_rd_req_valid.value = 0
        d.h_wr_valid.value = 0
        await ClockCycles(d.clk, cycles)
        await FallingEdge(d.clk)
        d.rst_n.value = 1
        await ClockCycles(d.clk, self.p.get("RST_LOCAL", 0) + 2)

    def regime(self, name: str):
        for k, v in MEM_REGIMES[name].items():
            setattr(self.mem, k, v)

    def knobs(self, rnd: random.Random | None, stall=None, lat=None, cap=None):
        """Back-end double knobs per cluster (no-op with the real back end)."""
        if self.real:
            return
        for c in range(self.nch):
            bk = self.dut.u_mvm.g_cl[c].u_cl.u_back
            st = stall if stall is not None else rnd.choice((0, 0, 0, 10, 40, 80))
            la = lat if lat is not None else rnd.choice((0, 0, 2, 8, 60))
            ca = cap if cap is not None else rnd.choice((1, 2, 3, 4, 16))
            bk.knob_stall.value = st
            bk.knob_lat.value = la
            bk.knob_cap.value = ca

    # ---- host access to the SPM (client 0) ----
    async def spm_write(self, line: int, data: bytes):
        d = self.dut
        n = len(data) // 64
        for i in range(n):
            d.h_wr_valid.value = 1
            d.h_wr_line.value = line + i
            d.h_wr_data.value = int.from_bytes(data[64 * i:64 * i + 64], "little")
            d.h_wr_be.value = (1 << 64) - 1
            while True:
                await FallingEdge(d.clk)
                if int(d.h_wr_ready.value):
                    break
            await RisingEdge(d.clk)
        d.h_wr_valid.value = 0

    async def spm_read(self, line: int, n: int) -> bytes:
        d = self.dut
        out = []
        sent = 0
        while len(out) < n:
            if sent < n:
                d.h_rd_req_valid.value = 1
                d.h_rd_req_line.value = line + sent
            else:
                d.h_rd_req_valid.value = 0
            await FallingEdge(d.clk)
            if int(d.h_rd_rsp_valid.value):
                out.append(int(d.h_rd_rsp_data.value).to_bytes(64, "little"))
            fire = sent < n and int(d.h_rd_req_ready.value)
            await RisingEdge(d.clk)
            if fire:
                sent += 1
        d.h_rd_req_valid.value = 0
        return b"".join(out)

    # ---- one MVM ----
    async def mvm(self, m: dict, *, check: bool = True, compare: bool = True,
                  timeout: int | None = None) -> dict:
        """Run one MVM. check: oracle, traffic / broadcast checks and y readback (st["y"] = the
        DUT's y span, st["want"] = the oracle's); compare: also raise when they differ
        (mvm_golden judges the y bytes by its digest instead)."""
        d, nch = self.dut, self.nch
        N, K, fmt, T = m["N"], m["K"], m["fmt"], m["T"]
        rb = mm.row_beats(fmt, K)
        per_ch = mm.layout_channels(fmt, K, m["rows"], nch) if N else [b""] * nch
        for c, data in enumerate(per_ch):
            if data:
                self.mem.load(c, 64 * m["w_off"], data)
        xs = b"".join(m["qas"])
        await self.spm_write(m["x"], xs)
        if m["y_old"]:
            await self.spm_write(m["dst"], m["y_old"])
        want = self.oracle.run(m) if check else None
        self.mon.clear()
        self.bc.clear()
        self.mem.clear_tracking()
        await RisingEdge(d.clk)
        d.cmd.value = mvm_cmd(m, nch)
        d.cmd_valid.value = 1
        while True:
            await FallingEdge(d.clk)
            if int(d.cmd_ready.value):
                break
        await RisingEdge(d.clk)
        d.cmd_valid.value = 0
        n0 = self.mon.done_count
        R = -(-N // nch) if N else 0
        max_beats = max([mm.rows_c(N, nch, c) * rb for c in range(nch)] + [0]) if N else 0
        if timeout is None:
            timeout = 4000 + 40 * max_beats + 40 * T * isa.qa_lines(mm.qa_kind(fmt), K) + 200 * T * (y_lines(N) + nch)
        cyc0 = self.mon.cyc
        while self.mon.done_count == n0:
            await FallingEdge(d.clk)
            if self.mon.cyc - cyc0 > timeout:
                raise AssertionError("MVM did not finish within %d cycles (%s)" % (timeout, describe(m)))
        cycles = self.mon.done_cycle - self.mon.accept_cycle
        st = dict(cycles=cycles, busy=self.mon.busy_cycles, max_beats=max_beats,
                  first_wr=self.mon.first_wr, last_wr=self.mon.last_wr, n_wr=len(self.mon.wr),
                  accept=self.mon.accept_cycle, done=self.mon.done_cycle)
        await RisingEdge(d.clk)
        if check:
            self.check_traffic(m, R, rb)
            self.bc.check(m, nch)
            got = await self.spm_read(m["dst"], len(m["y_old"]) // 64) if m["y_old"] else b""
            st["y"], st["want"] = got, want
            if compare and got != want:
                raise AssertionError("y mismatch (%s): %s" % (describe(m), self.diff(m, got, want)))
        self.stats.append((m, st))
        if os.environ.get("TSR_MVM_VERBOSE"):
            self.dut._log.info("MVM %s: %s", describe(m), st)
        return st

    async def issue(self, m: dict):
        """Present the command (from the next rising edge) and return after its accepting edge."""
        d = self.dut
        await RisingEdge(d.clk)
        d.cmd.value = mvm_cmd(m, self.nch)
        d.cmd_valid.value = 1
        while True:
            await FallingEdge(d.clk)
            if int(d.cmd_ready.value):
                break
        await RisingEdge(d.clk)
        d.cmd_valid.value = 0

    async def wait_done(self, n0: int, timeout: int = 200000):
        cyc0 = self.mon.cyc
        while self.mon.done_count == n0:
            await FallingEdge(self.dut.clk)
            if self.mon.cyc - cyc0 > timeout:
                raise AssertionError("no done within %d cycles" % timeout)

    def load_weights(self, m: dict):
        if m["N"]:
            for c, data in enumerate(mm.layout_channels(m["fmt"], m["K"], m["rows"], self.nch)):
                if data:
                    self.mem.load(c, 64 * m["w_off"], data)

    def check_traffic(self, m: dict, R: int, rb: int):
        N, T, nch = m["N"], m["T"], self.nch
        ql = isa.qa_lines(mm.qa_kind(m["fmt"]), m["K"])
        assert self.mon.busy_cycles == self.mon.done_cycle - self.mon.accept_cycle, \
            "perf_mvm_busy counted %d cycles, accept->done %d" % (
                self.mon.busy_cycles, self.mon.done_cycle - self.mon.accept_cycle)
        src = self.mem.src_beats
        for c in range(nch):
            want = mm.rows_c(N, nch, c) * rb if N else 0
            assert src[c][isa.SRC_WS] == want, "ch%d: %d WS beats, expected rows_c * ROW_BEATS = %d (%s)" % (
                c, src[c][isa.SRC_WS], want, describe(m))
            assert src[c][isa.SRC_DMA] == 0 and src[c][isa.SRC_AUX] == 0, src[c]
            if want == 0:
                assert self.mon.ws_req[c] == 0, "ch%d without rows requested weights" % c
        if N == 0:
            assert not self.mon.wr and not self.mon.rd, "N = 0 accessed the SPM (%d writes, %d reads)" % (
                len(self.mon.wr), len(self.mon.rd))
            return
        # reads: the broadcast in order, then ACC reads of y lines only
        rd = [ln for _, ln in self.mon.rd]
        bc = list(range(m["x"], m["x"] + T * ql))
        assert rd[:len(bc)] == bc, "broadcast reads %s... expected %d.. (%s)" % (rd[:8], m["x"], describe(m))
        ds, yl = resolved_stride(m), y_lines(N)
        ylines = {m["dst"] + t * ds + i for t in range(T) for i in range(yl)}
        rest = rd[len(bc):]
        if not m["acc"]:
            assert not rest, "non-ACC MVM read %d extra lines" % len(rest)
        else:
            bad = [ln for ln in rest if ln not in ylines]
            assert not bad, "ACC read lines outside y: %s" % bad[:8]
            assert set(rest) == ylines, "ACC did not read every y line"
        # writes: every y byte exactly once, nothing else
        cnt = {}
        for _, ln, be in self.mon.wr:
            for b in range(64):
                if be >> b & 1:
                    k = 64 * ln + b
                    cnt[k] = cnt.get(k, 0) + 1
        ybytes = {64 * ln + b for ln in ylines for b in range(64)}
        extra = sorted(set(cnt) - ybytes)
        missing = sorted(ybytes - set(cnt))
        twice = sorted(k for k, v in cnt.items() if v != 1)
        assert not extra and not missing and not twice, \
            "SPM write bytes: %d outside y (first line %s), %d y bytes missing (first line %s), %d written twice (%s)" % (
                len(extra), extra[0] // 64 if extra else None, len(missing), missing[0] // 64 if missing else None,
                len(twice), describe(m))
        # the number of partial-line writes is that of the gather_jobs model
        jobs = mm.gather_jobs(N, nch, T, m["dst"], ds)
        if ds >= yl or T == 1:
            assert len(self.mon.wr) == len(jobs), "%d SPM writes, gather_jobs has %d" % (len(self.mon.wr), len(jobs))

    def diff(self, m: dict, got: bytes, want: bytes) -> str:
        ds, N = resolved_stride(m), m["N"]
        out = []
        for i in range(0, min(len(got), len(want)), 4):
            if got[i:i + 4] != want[i:i + 4]:
                line = m["dst"] + i // 64
                t, off = divmod(i // 64, ds)
                n = off * 16 + (i % 64) // 4
                out.append("line %d (token %d, row %d%s): got 0x%08x want 0x%08x" % (
                    line, t, n, " tail" if n >= N else "", int.from_bytes(got[i:i + 4], "little"),
                    int.from_bytes(want[i:i + 4], "little")))
                if len(out) >= 6:
                    break
        return "; ".join(out) + " (%d bytes differ)" % sum(1 for a, b in zip(got, want) if a != b)


async def new_tb(dut) -> Tb:
    p = cfg_from_env()
    tb = Tb(dut, p)
    await tb.start()
    return tb


# =============================================================================================
# cocotb tests
# =============================================================================================
@cocotb.test()
async def mvm_directed(dut):
    tb = await new_tb(dut)
    p, nch, TM, L = tb.p, tb.nch, tb.p["T_MAX"], tb.L
    rnd = random.Random(tb_seed() * 7 + 1)
    tb.regime("rand")
    tb.knobs(None, stall=0, lat=0, cap=4)
    kmax = p["K_MAX"]
    cases = []
    # N = 0 (with ACC, T = T_MAX): nothing read, nothing written
    cases.append(dict(N=0, fmt=0, K=32, T=TM, acc=True))
    cases.append(dict(N=0, fmt=1, K=256, T=1, acc=False))
    # one row; rows_c = 0 clusters (N < NUM_CH); N = 5 with >= 4 channels: last channel empty
    cases.append(dict(N=1, fmt=0, K=32, T=1, acc=False))
    cases.append(dict(N=1, fmt=1, K=256, T=TM, acc=True))
    if nch > 1:
        cases.append(dict(N=nch - 1, fmt=0, K=64, T=TM, acc=False))
    if nch >= 4:
        cases.append(dict(N=5, fmt=1, K=512, T=1, acc=True))
    # R % 16 != 0: lines shared by two or more channels; N % 16 != 0 tails; N = 16 k
    cases.append(dict(N=17 * nch + 2, fmt=0, K=96, T=min(2, TM), acc=False))
    cases.append(dict(N=17 * nch + 2, fmt=0, K=96, T=min(2, TM), acc=True))
    cases.append(dict(N=3 * nch + 1, fmt=1, K=256, T=TM, acc=True))
    cases.append(dict(N=16 * nch, fmt=0, K=1024, T=1, acc=False))
    cases.append(dict(N=32, fmt=1, K=768, T=TM, acc=False))
    cases.append(dict(N=15, fmt=0, K=1056, T=TM, acc=True))
    # T = T_MAX with dst_stride gaps; T = 1 with a huge (unused) dst_stride
    cases.append(dict(N=37, fmt=0, K=64, T=TM, acc=True, stride=y_lines(37) + 5 if TM > 1 else 0))
    cases.append(dict(N=20, fmt=1, K=256, T=TM, acc=False, stride=y_lines(20) + 1 if TM > 1 else 0))
    cases.append(dict(N=9, fmt=0, K=32, T=1, acc=True, stride=0xFFFFFFFF))
    # K edges: partial chunks, K_MAX (few rows)
    cases.append(dict(N=nch + 2, fmt=0, K=min(kmax, 1024 + 32 * 31), T=1, acc=False))
    cases.append(dict(N=nch, fmt=1, K=min(kmax, 1024 + 256 * 3), T=1, acc=True))
    cases.append(dict(N=2, fmt=0, K=kmax, T=1, acc=False))
    cases.append(dict(N=2, fmt=1, K=kmax, T=1, acc=True))
    for c in cases:
        m = make_mvm(rnd, p, **c)
        await tb.mvm(m)
    # placements at the SPM ends, W_OFF ending at the DRAM limit
    for fmt in (0, 1):
        K = 256 * (fmt + 1)
        N = 2 * nch + 3
        T = TM
        yl, ql = y_lines(N), isa.qa_lines(mm.qa_kind(fmt), K)
        R = -(-N // nch)
        m = make_mvm(rnd, p, fmt=fmt, K=K, N=N, T=T, acc=bool(fmt), stride=0, x=0,
                     dst=L - ((T - 1) * yl + yl), w_off=tb.limit - R * mm.row_beats(fmt, K))
        await tb.mvm(m)
        m = make_mvm(rnd, p, fmt=fmt, K=K, N=N, T=T, acc=not fmt, stride=0, dst=0, x=L - T * ql)
        await tb.mvm(m)
    # ACC with special old values in every lane (NaN payloads -> canonical, DAZ of subnormal old
    # values, signed zeros, Inf - Inf, overflow) and special y (raw QA, special weights)
    for it in range(4):
        N = rnd.choice([16, 31, 2 * nch + 1, 40])
        T = rnd.randint(1, TM)
        m = make_mvm(rnd, p, fmt=it % 2, K=256 if it % 2 else 64, N=N, T=T, acc=True, rows_mode="special",
                     qa_mode="raw")
        y = bytearray(m["y_old"])
        for i in range(0, len(y), 4):
            y[i:i + 4] = rnd.choice(ACC_SPECIAL_OLD).to_bytes(4, "little")
        m["y_old"] = bytes(y)
        await tb.mvm(m)
    dut._log.info("mvm_directed: %d MVMs checked", len(tb.stats))


@cocotb.test()
async def mvm_random(dut):
    tb = await new_tb(dut)
    p = tb.p
    rnd = random.Random(tb_seed() * 7 + 2)
    n = max(4, int(60 * scale()))
    kmax_eff = 4096
    for i in range(n):
        tb.regime(rnd.choice(("rand", "rand", "fast", "slow", "burst")))
        tb.knobs(rnd)
        m = make_mvm(rnd, p, kmax_eff=kmax_eff, y_special=rnd.choice((0.0, 0.1, 0.5)))
        # keep the weight stream moderate
        if m["N"] * mm.row_beats(m["fmt"], m["K"]) > 6000 * p["NUM_CH"]:
            m = make_mvm(rnd, p, kmax_eff=1024)
        await tb.mvm(m)
    dut._log.info("mvm_random: %d MVMs checked", n)


def golden_list(rnd: random.Random) -> list:
    """Fixed MVM shapes (T = 1, K <= 2048, fits a 64 KiB SPM), valid in every configuration."""
    out = []
    for i in range(12):
        fmt = i % 2
        K = rnd.choice([32, 96, 1056, 2048] if fmt == 0 else [256, 768, 1280, 2048])
        N = rnd.choice([1, 2, 3, 5, 16, 17, 33, 47, 70])
        out.append(dict(fmt=fmt, K=K, N=N, T=1, acc=bool(i % 3 == 0), stride=0))
    return out


@cocotb.test()
async def mvm_golden(dut):
    """A fixed MVM list with fixed data in every configuration. The y bytes the DUT wrote (read
    back from its SPM) are hashed and the DUT digest must equal the pinned GOLDEN_DIGEST, so the
    results are identical across BC_PIPE / RES_PIPE / SPM_RD_LAT / NUM_CH / T_MAX by this check
    alone (the per-MVM compare with the oracle is off here; every other check of Tb.mvm runs).
    The oracle's digest of the same list is checked separately: if it moves, the data
    generator or the oracle changed, not the RTL."""
    tb = await new_tb(dut)
    p = dict(tb.p)
    rnd = random.Random(1234)                 # fixed: same data in every configuration
    tb.regime("rand")
    tb.knobs(random.Random(tb_seed()))
    h_dut, h_sim = hashlib.sha256(), hashlib.sha256()
    recs = []
    for c in golden_list(rnd):
        q = dict(p, SPM_BYTES=65536, NUM_CH=1, ADDR_BEAT_W=22)   # data / placement independent of the config
        m = make_mvm(rnd, q, **c)
        m["w_off"] = W_LO + (m["w_off"] - W_LO) % 4096
        st = await tb.mvm(m, compare=False)
        h_dut.update(st["y"])
        h_sim.update(st["want"])
        recs.append((m, st["y"], st["want"]))
    dut_d, sim_d = h_dut.hexdigest()[:16], h_sim.hexdigest()[:16]
    dut._log.info("mvm_golden: DUT digest %s, oracle digest %s, pinned %s", dut_d, sim_d,
                  GOLDEN_DIGEST or "-")
    assert not GOLDEN_DIGEST or sim_d == GOLDEN_DIGEST, (
        "oracle digest %s != pinned %s: golden_list / make_mvm data or the oracle changed (not an "
        "RTL result); re-pin GOLDEN_DIGEST once the change is understood" % (sim_d, GOLDEN_DIGEST))
    ref = GOLDEN_DIGEST or sim_d
    if dut_d != ref:
        bad = [i for i, (_, g, w) in enumerate(recs) if g != w]
        i = bad[0]
        m, g, w = recs[i]
        raise AssertionError("DUT digest %s != golden %s: %d of %d MVMs differ, first #%d (%s): %s" % (
            dut_d, ref, len(bad), len(recs), i, describe(m), tb.diff(m, g, w)))


GOLDEN_DIGEST = "5e28386055f9e494"   # y bytes of golden_list (pinned from the ISA simulator = the
                                     # mvm_reference oracle = every DUT configuration)


def cycles_list(p: dict) -> list:
    """Fixed MVM shapes of mvm_cycles, legal in every configuration: both formats, N = 0 / 1 / around
    NUM_CH / 16 NUM_CH, T = 1 and T_MAX, ACC, long rows."""
    nch, TM = p["NUM_CH"], p["T_MAX"]
    return [
        dict(fmt=0, K=32, N=1, T=1, acc=False),
        dict(fmt=1, K=256, N=nch + 1, T=TM, acc=True),
        dict(fmt=0, K=1056, N=17 * nch + 2, T=min(2, TM), acc=False),
        dict(fmt=1, K=768, N=70, T=TM, acc=True),
        dict(fmt=0, K=2048, N=2 * nch, T=1, acc=False),
        dict(fmt=1, K=1280, N=16 * nch, T=min(2, TM), acc=False),
        dict(fmt=0, K=32, N=0, T=TM, acc=False),
        dict(fmt=0, K=96, N=3 * nch + 1, T=TM, acc=True),
    ]


@cocotb.test()
async def mvm_cycles(dut):
    """Cycle exactness of the weight path: per MVM {accept -> done, busy, first / last SPM write relative
    to the accept, number of SPM writes} under a fixed memory model, hashed and compared with the pinned
    CYCLE_DIGEST of the configuration. The memory model's channel generators are re-seeded and the
    back-end double's knobs fixed, so the cycles do not depend on $TSR_SEED. Two phases: the fast regime
    (ideal memory, no back-pressure) and the random one (variable latency, back-pressure and stalls)."""
    tb = await new_tb(dut)
    p = tb.p
    rec = []
    for phase, knobs in (("fast", dict(stall=0, lat=0, cap=16)), ("rand", dict(stall=10, lat=2, cap=4))):
        rnd = random.Random(0xC7C1E5 + len(rec))
        tb.regime(phase)
        tb.knobs(None, **knobs)
        for c, ch in enumerate(tb.mem.ch):
            ch.rng = random.Random(0x5EED00 + 97 * c + (phase == "rand"))
        for c in cycles_list(p):
            m = make_mvm(rnd, p, **c)
            st = await tb.mvm(m)
            a = st["accept"]
            fw = -1 if st["first_wr"] is None else st["first_wr"] - a
            lw = -1 if st["last_wr"] is None else st["last_wr"] - a
            rec.append((phase, st["cycles"], st["busy"], fw, lw, st["n_wr"]))
    key = p["name"]
    dig = hashlib.sha256(repr(rec).encode()).hexdigest()[:16]
    dut._log.info("mvm_cycles %s: digest %s (%d MVMs), pinned %s", key, dig, len(rec), CYCLE_DIGEST.get(key, "-"))
    for i, r in enumerate(rec):
        dut._log.info("mvm_cycles %s #%d: %s", key, i, r)
    assert key in CYCLE_DIGEST, "no pinned cycle digest for %s (measured %s)" % (key, dig)
    assert dig == CYCLE_DIGEST[key], (
        "cycle digest %s != pinned %s for %s: the weight path's timing changed (accept -> done, busy, write "
        "span); the per-MVM tuples are in the log above. Re-pin CYCLE_DIGEST only once the change is "
        "understood" % (dig, CYCLE_DIGEST[key], key))


# Pinned with the real back end (the back-end double's cycles depend on its own random state), equal on the
# tree at W4-A (HEAD 4149614), on the W4-B tree, for $TSR_SEED 1 and 7, and with HAS_ATT = 1.
CYCLE_DIGEST = {
    "c4":    "6a215afa9d4a2fff",
    "c3t3":  "3df890a39be82a6a",
    "c1t8":  "816d9a4a60303627",
    "c16t2": "de23fbf976a873df",
    "c4t8":  "129b3691537eca8a",
}


# ---- mvm_perf bounds (docs/rtl-mvm.md M3 scope, §4 FIFO sizing, §7.5 RB_MIN, §9.1 overhead) ----
RB_MIN = 9          # §7.5: the back end never drops beat_ready on rows of >= RB_MIN (<= 9) beats
FAST_RT = 12        # weight round trip budget with the "fast" memory (cycles; measured ~6.5), §4
CPB_EPS = 0.01      # slack of a measured cycles-per-beat (the fast regime's 0/1-cycle latency)
DBL_FIXED = 25      # fixed-overhead budget with the back-end double (+ pipes / latencies, below)
GATHER_MIN = 0.85   # double, 2-beat rows: SPM line writes per cycle >= GATHER_MIN x row-rate bound


def perf_cpb_max(p: dict, rb: int):
    """Bound of the steady-state cycles per weight beat at T = 1 with the "fast" memory and a ready
    back end, or None where the contract allows any rate (real back end, rows shorter than RB_MIN,
    §7.5). §4: one beat per cycle while round trip + 64 <= WS_FIFO_BEATS, else about (round trip +
    64) / WS_FIFO_BEATS (64-beat reservations; the 64-entry FIFO configurations)."""
    if p.get("real") and rb < RB_MIN:
        return None
    return max(1.0, (FAST_RT + 64) / p["WS_FIFO_BEATS"]) + CPB_EPS


def perf_fixed_max(p: dict) -> int:
    """Bound of the fixed part of accept -> done: the cycles beyond the weight stream (beats x
    cycles per beat) and the T x QA_LINES broadcast reads."""
    if p.get("real"):
        # §9.1: setup ~20 + back-end pipeline ~30 + tree <= 4 ADD_LAT + 8 + gather ~20, plus the
        # pipes and the SPM read latency. The real back end's own latency is the back agent's, so
        # this is the contract budget (measured 33-99 cycles in CFGS, bound 84-124).
        return 78 + 4 * p["ADD_LAT"] + p["BC_PIPE"] + p["RES_PIPE"] + 2 * p["SPM_RD_LAT"]
    # back-end double (a row's result 2 + ACT_RD_LAT cycles after its last beat): the top's own
    # latency (accept, cluster setup, pipes, gather, done; the weight round trip and WS_RD_LAT
    # hide behind the broadcast) plus the double's. Measured 15-18.5 + BC_PIPE + RES_PIPE +
    # SPM_RD_LAT + ACT_RD_LAT cycles over 12 seeds in the configurations whose FIFO sustains one
    # beat per cycle (margin 6.5-7.5 cycles: mutants adding 12 or 16 cycles before done / before
    # act_ready fail there, +8 mostly passes); with a 64-entry FIFO the rate is ~1.10 and
    # jitters, which makes the fixed-part estimate noisier (+8).
    fifo_bound = p["WS_FIFO_BEATS"] < FAST_RT + 64
    # W4-A spm: + 2 for the even / odd SPM (CC §10.2): the broadcast's second QA line and the
    # gather's second y line are on the other bank, whose read / write grant another client
    # (the host port) may own: one grant hand-over cycle per bank port instead of one per port.
    return DBL_FIXED + p["BC_PIPE"] + p["RES_PIPE"] + p["SPM_RD_LAT"] + p["ACT_RD_LAT"] + 8 * fifo_bound + 2


@cocotb.test()
async def mvm_perf(dut):
    """Throughput and latency at T = 1 with the "fast" memory and a ready back end (double: no
    stalls, no result delay, capacity 16). Asserted (bounds above):
      - steady-state cycles per weight beat, from two MVMs of one shape (r and 3r rows per
        channel): d(cycles) / d(beats per channel) <= perf_cpb_max, i.e. one weight beat per
        cycle per cluster where the FIFO sizing rule holds (rows >= RB_MIN for the real back end);
      - fixed overhead (cycles - beats x measured rate - T x QA_LINES) <= perf_fixed_max, and
        every MVM's accept -> done <= beats x cpb_max + T x QA_LINES + fixed_max (also the
        single-point long-K shape and the N = 1 latency);
      - N = 0: done at most 2 cycles after the accept (the design: 1);
      - gather (double only; with the real back end 2-beat rows are back-end bound, §7.5): SPM
        line writes per cycle >= GATHER_MIN x the row-rate bound min(1, NUM_CH x T / 32).
    Measurements are logged; TSR_MVM_PERF_OUT appends them to a file."""
    tb = await new_tb(dut)
    p, nch = tb.p, tb.nch
    rnd = random.Random(tb_seed() * 7 + 3)
    tb.regime("fast")
    tb.knobs(None, stall=0, lat=0, cap=16)
    fixed_max = perf_fixed_max(p)
    cpb_long = perf_cpb_max(p, RB_MIN)
    lines, fails = [], []

    def ql_of(fmt, K):
        return isa.qa_lines(mm.qa_kind(fmt), K)

    def bound(fmt, K, beats, cyc, what, cpb_max, extra=0):
        lim = beats * cpb_max + ql_of(fmt, K) + fixed_max + extra
        if cyc > lim:
            fails.append("%s: %d cycles accept->done > bound %.0f (%d beats x %.3f + %d QA lines + %d + %d)"
                         % (what, cyc, lim, beats, cpb_max, ql_of(fmt, K), fixed_max, extra))
        return lim

    async def one(fmt, K, N):
        m = make_mvm(rnd, p, fmt=fmt, K=K, N=N, T=1, acc=False, stride=0, w_off=W_LO)
        return await tb.mvm(m)

    kl = min(p["K_MAX"], 4096)
    for fmt, K, r in ((0, kl, 16), (1, kl, 16), (0, 896, 64), (1, 768, 64)):
        rb = mm.row_beats(fmt, K)
        s1 = await one(fmt, K, r * nch)
        s2 = await one(fmt, K, 3 * r * nch)
        b1, b2 = s1["max_beats"], s2["max_beats"]
        cpb = (s2["cycles"] - s1["cycles"]) / (b2 - b1)
        fixed = s1["cycles"] - b1 * cpb - ql_of(fmt, K)
        cmax = perf_cpb_max(p, rb)
        what = "fmt %d K %5d (%2d-beat rows)" % (fmt, K, rb)
        lines.append("%s: rows/ch %d / %d: %d / %d cycles for %d / %d beats/ch -> %.4f cycles/beat "
                     "(bound %s), fixed %.1f (bound %d), QA lines %d" % (
                         what, r, 3 * r, s1["cycles"], s2["cycles"], b1, b2, cpb,
                         "%.3f" % cmax if cmax else "none: rows < RB_MIN", fixed, fixed_max, ql_of(fmt, K)))
        if cmax is None:
            continue
        if cpb > cmax:
            fails.append("%s: %.4f cycles per weight beat > %.3f" % (what, cpb, cmax))
        if fixed > fixed_max:
            fails.append("%s: fixed overhead %.1f cycles > %d" % (what, fixed, fixed_max))
        bound(fmt, K, b1, s1["cycles"], what + " r", cmax)
        bound(fmt, K, b2, s2["cycles"], what + " 3r", cmax)
    if p["K_MAX"] >= 27648:
        st = await one(1, 27648, 8 * nch)
        lines.append("fmt 1 K 27648 rows/ch 8: %d cycles for %d beats/ch, QA lines %d" % (
            st["cycles"], st["max_beats"], ql_of(1, 27648)))
        bound(1, 27648, st["max_beats"], st["cycles"], "fmt 1 K 27648", cpb_long)
    # N = 0 and a 1-row MVM: latency
    st = await tb.mvm(make_mvm(rnd, p, fmt=0, K=32, N=0, T=1, acc=False))
    lines.append("N = 0: done %d cycle(s) after the accept" % st["cycles"])
    if st["cycles"] > 2:
        fails.append("N = 0: done %d cycles after the accept (> 2)" % st["cycles"])
    # one row: a 2-line broadcast does not hide the weight round trip (+ FAST_RT)
    st = await one(0, 32, 1)
    lim = bound(0, 32, 2, st["cycles"], "N = 1, K = 32", cpb_long, FAST_RT)
    lines.append("N = 1, K = 32: %d cycles accept->done (bound %.0f)" % (st["cycles"], lim))
    # gather throughput: 2-beat rows (Q8_0 K = 32); with the double the clusters produce
    # NUM_CH * T / 32 lines per cycle, more than the SPM write port takes when that is >= 1
    for acc in (False, True):
        T = p["T_MAX"]
        N = min(64 * nch * 4, 4096)
        m = make_mvm(rnd, p, fmt=0, K=32, N=N, T=T, acc=acc, stride=0)
        st = await tb.mvm(m)
        span = st["last_wr"] - st["first_wr"] + 1
        rate, rbnd = st["n_wr"] / span, min(1.0, nch * T / 32)
        lines.append("gather %s: K 32 N %d T %d: %d line writes in %d cycles = %.3f writes/cycle "
                     "(row rate bound %.3f%s)" % ("ACC" if acc else "plain", N, T, st["n_wr"], span,
                                                  rate, rbnd, "" if tb.real else ", min %.3f" % (GATHER_MIN * rbnd)))
        if not tb.real and rate < GATHER_MIN * rbnd:
            fails.append("gather %s: %.3f writes/cycle < %.3f x %.3f" % (
                "ACC" if acc else "plain", rate, GATHER_MIN, rbnd))
    for ln in lines:
        dut._log.info("mvm_perf: %s", ln)
    out = Path(os.environ.get("TSR_MVM_PERF_OUT", "")) if os.environ.get("TSR_MVM_PERF_OUT") else None
    if out:
        with open(out, "a") as f:
            f.write("[%s%s]\n" % ("real_" if tb.real else "", p["name"]) + "\n".join(lines) +
                    "".join("\nFAIL " + x for x in fails) + "\n")
    assert not fails, "mvm_perf: " + "; ".join(fails)


def _carve(rnd: random.Random, L: int, sizes: list) -> list:
    """Disjoint SPM regions of the given sizes (lines), random order and gaps; None if no fit."""
    total = sum(sizes)
    if total > L:
        return None
    gaps = [rnd.randint(0, (L - total) // (len(sizes) + 1)) for _ in sizes]
    order = list(range(len(sizes)))
    rnd.shuffle(order)
    out, pos = [None] * len(sizes), 0
    for i in order:
        pos += gaps[i]
        out[i] = pos
        pos += sizes[i]
    return out


@cocotb.test()
async def mvm_b2b(dut):
    """MVMs issued back to back (the next command presented while the previous one runs, so it is
    accepted in the first cycle cmd_ready allows), including ACC MVMs onto the previous MVM's y
    (DONE => visible: the ACC reads must see the new y). The whole sequence vs the oracle."""
    tb = await new_tb(dut)
    p, nch, L = tb.p, tb.nch, tb.L
    rnd = random.Random(tb_seed() * 7 + 5)
    for rep in range(3):
        tb.regime(rnd.choice(("rand", "fast", "burst")))
        tb.knobs(rnd)
        ms = []
        while len(ms) < 6:
            if ms and rnd.random() < 0.4:
                prev = ms[-1]           # ACC onto the previous y, same shape, new weights / x
                m = make_mvm(rnd, p, fmt=rnd.randrange(2), N=prev["N"], T=prev["T"], acc=True,
                             stride=prev["stride"], kmax_eff=1024)
                m["chain"] = True
            else:
                m = make_mvm(rnd, p, kmax_eff=1024, N=rnd.choice([1, 5, 17, 3 * nch + 2, 40]))
                m["chain"] = False
            ms.append(m)
        # disjoint regions: x per MVM, y per chain head (chained ACC MVMs share it)
        sizes, owners = [], []
        for i, m in enumerate(ms):
            sizes.append(m["T"] * isa.qa_lines(mm.qa_kind(m["fmt"]), m["K"]))
            owners.append(("x", i))
            if not m["chain"] and m["N"]:
                sizes.append(len(m["y_old"]) // 64)
                owners.append(("y", i))
        pos = _carve(rnd, L, sizes)
        if pos is None:
            continue
        head = None
        for (kind, i), at in zip(owners, pos):
            if kind == "x":
                ms[i]["x"] = at
            else:
                ms[i]["dst"] = at
        for m in ms:
            if not m["chain"]:
                head = m
            else:
                m["dst"] = head["dst"]
        # weights: disjoint windows
        wo = W_LO
        for m in ms:
            m["w_off"] = wo
            wo += (-(-m["N"] // nch) if m["N"] else 0) * mm.row_beats(m["fmt"], m["K"]) + rnd.randint(0, 70)
            tb.load_weights(m)
        preload = [(m["x"], b"".join(m["qas"])) for m in ms]
        preload += [(m["dst"], m["y_old"]) for m in ms if not m["chain"] and m["y_old"]]
        for line, data in preload:
            await tb.spm_write(line, data)
        want = tb.oracle.run_seq(ms, preload)
        n0 = tb.mon.done_count
        for i, m in enumerate(ms):
            await tb.issue(m)          # waits for cmd_ready: accepted the cycle after the previous done
            if i:
                assert tb.mon.accept_cycle == tb.mon.done_cycle + 1, \
                    "command %d accepted %d cycles after the previous done" % (i, tb.mon.accept_cycle - tb.mon.done_cycle)
            await tb.wait_done(n0 + i)
        assert tb.mon.done_count == n0 + len(ms)
        await RisingEdge(dut.clk)
        for m in ms:
            if m["y_old"]:
                n = len(m["y_old"]) // 64
                got = await tb.spm_read(m["dst"], n)
                exp = want[64 * m["dst"]:64 * (m["dst"] + n)]
                assert got == exp, "b2b: y mismatch for %s: %s" % (describe(m), tb.diff(m, got, exp))
    dut._log.info("mvm_b2b: ok")


@cocotb.test()
async def mvm_overlap(dut):
    """Undefined operand overlaps (spec Implementation clarifications, "Overlapping operands"): dst inside x, token ranges
    overlapping (T > 1, dst_stride < y_lines). The unit must finish normally (done, quiescent) and
    write only inside the union of the token ranges; values are logged, not required (our order
    reads all of x before the first write, so dst-vs-x overlaps still match the oracle)."""
    tb = await new_tb(dut)
    p, nch, L, TM = tb.p, tb.nch, tb.L, tb.p["T_MAX"]
    rnd = random.Random(tb_seed() * 7 + 6)
    same = diff = 0
    for it in range(8):
        tb.regime(rnd.choice(("rand", "fast")))
        tb.knobs(rnd)
        fmt = rnd.randrange(2)
        K = 256 * rnd.randint(1, 2)
        N = rnd.choice([17, 33, 3 * nch + 1, 40])
        yl = y_lines(N)
        if it % 2 == 0 or TM == 1:
            T = rnd.randint(1, TM)
            stride = 0
            ql = isa.qa_lines(mm.qa_kind(fmt), K)
            x = rnd.randrange(0, L - T * ql - yl)
            dst = x + rnd.randrange(0, T * ql)        # y starts inside x
            dst = min(dst, L - ((T - 1) * yl + yl))
        else:
            T = rnd.randint(2, TM)
            stride = rnd.randint(1, yl - 1) if yl > 1 else 1
            x = 0
            ql = isa.qa_lines(mm.qa_kind(fmt), K)
            dst = T * ql + rnd.randint(0, 5)
        m = make_mvm(rnd, p, fmt=fmt, K=K, N=N, T=T, acc=bool(rnd.getrandbits(1)), stride=stride, x=x, dst=dst)
        tb.load_weights(m)
        await tb.spm_write(m["x"], b"".join(m["qas"]))
        span = (T - 1) * resolved_stride(m) + yl
        old = m["y_old"][:64 * span].ljust(64 * span, b"\0")
        await tb.spm_write(m["dst"], old)
        # the oracle sees the same SPM: x first, then the y span on top (overlapping lines)
        before = await tb.spm_read(min(m["x"], m["dst"]), max(m["x"] + T * ql, m["dst"] + span) - min(m["x"], m["dst"]))
        base = min(m["x"], m["dst"])
        tb.mon.clear()
        tb.mem.clear_tracking()
        n0 = tb.mon.done_count
        await tb.issue(m)
        await tb.wait_done(n0)
        ds = resolved_stride(m)
        allowed = {m["dst"] + t * ds + i for t in range(T) for i in range(yl)}
        bad = [ln for _, ln, _ in tb.mon.wr if ln not in allowed]
        assert not bad, "overlap MVM wrote outside the y ranges: %s" % bad[:4]
        src = tb.mem.src_beats
        for c in range(nch):
            assert src[c][isa.SRC_WS] == mm.rows_c(N, nch, c) * mm.row_beats(fmt, K)
        await RisingEdge(dut.clk)
        got = await tb.spm_read(m["dst"], span)
        want = tb.oracle.run_seq([m], [(base, before)])[64 * m["dst"]:64 * (m["dst"] + span)]
        if got == want:
            same += 1
        else:
            diff += 1
    dut._log.info("mvm_overlap: 8 overlapping MVMs finished normally (%d equal to the oracle, %d differ: undefined)",
                  same, diff)


@cocotb.test()
async def mvm_reset(dut):
    tb = await new_tb(dut)
    p = tb.p
    rnd = random.Random(tb_seed() * 7 + 4)
    for i in range(3):
        tb.regime(rnd.choice(("rand", "fast")))
        tb.knobs(rnd)
        m = make_mvm(rnd, p, kmax_eff=1024, N=rnd.choice([20, 40, 3 * p["NUM_CH"] + 1]))
        # start it, then reset after a random number of cycles
        for c, data in enumerate(mm.layout_channels(m["fmt"], m["K"], m["rows"], tb.nch)):
            if data:
                tb.mem.load(c, 64 * m["w_off"], data)
        await tb.spm_write(m["x"], b"".join(m["qas"]))
        d = dut
        await RisingEdge(d.clk)
        d.cmd.value = mvm_cmd(m, tb.nch)
        d.cmd_valid.value = 1
        while True:
            await FallingEdge(d.clk)
            if int(d.cmd_ready.value):
                break
        await RisingEdge(d.clk)
        d.cmd_valid.value = 0
        await ClockCycles(d.clk, rnd.randint(1, 300))
        await tb.reset()
        assert int(d.cmd_ready.value) == 1 and int(d.perf_mvm_busy.value) == 0 and int(d.done.value) == 0
        # fully checked MVMs after the reset
        for _ in range(2):
            await tb.mvm(make_mvm(rnd, p, kmax_eff=1024))
    dut._log.info("mvm_reset: ok")


@cocotb.test()
async def mvm_bc_flush(dut):
    """Exercise each occupied stage across a one-cycle reset, then run complete oracle checks.

    The HDL observer checks the actual production source and every cluster every cycle, including
    bubbles and reset flushes. No DUT signal is forced and no expected digest is re-pinned.
    """
    tb = await new_tb(dut)
    p = tb.p
    rnd = random.Random(tb_seed() * 11 + 0xBC2)
    tb.regime("fast")
    stages = max(1, p["BC_PIPE"])
    local_reset = p.get("RST_LOCAL", 0)
    for kind in (2, 1):                    # source valid mapping: MVM command, broadcast line
        for offset in range(stages):
            m = make_mvm(rnd, p, fmt=0, K=min(2048, p["K_MAX"]), N=17 * tb.nch, T=1, acc=False)
            tb.load_weights(m)
            await tb.spm_write(m["x"], b"".join(m["qas"]))
            await tb.issue(m)
            # A command's su_q or a line's bc_rsp becomes bc_v_q on the next edge,
            # then enters reference stage 0 on the edge after that. A two-stage
            # local reset must therefore start at that earlier producer, while
            # reset 0/1 can start from the actual broadcast source. Observe these
            # events at falling edges; do not infer them from issue()'s return.
            for _ in range(1000):
                await FallingEdge(dut.clk)
                if local_reset == 2:
                    producer = dut.u_mvm.su_q if kind == 2 else dut.u_mvm.bc_rsp
                    present = int(producer.value)
                else:
                    present = int(dut.o_bc_source_valid.value) & kind
                if present:
                    break
            else:
                raise AssertionError("broadcast producer valid %d not observed" % kind)
            delay = offset + max(0, 1 - local_reset)
            if delay:
                await ClockCycles(dut.clk, delay)
                await FallingEdge(dut.clk)
            pulse = cocotb.start_soon(tb.reset(cycles=1, start_now=True))
            # Inspect the first local-low half-cycle, before its next edge clears
            # the histories. This proves each individual injection hits the
            # requested stage, even if an earlier test already set coverage bits.
            await Timer(1, "ps")
            for _ in range(local_reset + 1):
                if not int(dut.o_mvm_rst_n.value):
                    break
                await FallingEdge(dut.clk)
                await Timer(1, "ps")
            else:
                raise AssertionError("local broadcast reset did not arrive")
            if p["BC_PIPE"]:
                for c in range(tb.nch):
                    hist = dut.u_bc_observer.g_reference[c].g_history.valid_history
                    assert ((int(hist.value) >> (3 * offset)) & kind), (
                        "cluster %d: local reset missed kind %d at stage %d (history %#x)" %
                        (c, kind, offset, int(hist.value)))
                dut._log.info("BC flush phase: raw/local reset %d cycles, depth %d, kind %d, stage %d occupied",
                              local_reset, p["BC_PIPE"], kind, offset)
            await pulse
            await FallingEdge(dut.clk)
            assert int(dut.o_bc_cluster_valid.value) == 0, "broadcast valid survived reset"
            assert int(dut.cmd_ready.value) == 1 and int(dut.perf_mvm_busy.value) == 0
            await tb.mvm(make_mvm(rnd, p, fmt=offset % 2, K=512, N=tb.nch + 1, T=p["T_MAX"], acc=True))
    for i in range(4):
        tb.regime("fast" if i % 2 == 0 else "rand")
        await tb.mvm(make_mvm(rnd, p, K=min(2048, p["K_MAX"]), N=17 * tb.nch, T=p["T_MAX"]))
    if p["BC_PIPE"]:
        discarded = int(dut.o_bc_reset_discarded.value)
        for c in range(tb.nch):
            assert ((discarded >> (3 * c)) & 3) == 3, "cluster %d: command/line reset flush not exercised" % c
        dut._log.info("mvm_bc_flush: command/line resets in %d stages, all %d cluster ports checked", stages, tb.nch)
    else:
        dut._log.info("mvm_bc_flush: BC_PIPE0 has no buffered item to discard; reset/recovery and %d wire ports checked",
                      tb.nch)
