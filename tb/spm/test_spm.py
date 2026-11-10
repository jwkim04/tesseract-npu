"""tsr_spm testbench: the even / odd 2R2W SPM (docs/design/04-control-core.md "CC" §10, CC-12;
docs/rtl-core.md §5), its single-port adapter tsr_spm_cport and the RISC-V bridge tsr_spm_bridge.

Reference: the ISA simulator's SPM (spm_sim.SimSpm: the SPM bytes of a tsr_sim instance through
libtsr_sim.so, and the simulator's RISC-V SPM-window semantics through its bus model, CC §4.1 /
§10.5). The SPM is undefined at start: bytes the bench never wrote are not compared.

Bench 1, tsr_spm alone (toplevel tsr_spm, every (client, bank) port driven from Python), checked
every cycle against an exact model of the contract in rtl/core/tsr_spm.sv:
  - arbitration per bank and port: ready == the registered round-robin grant
    (gnt <= |req ? first requester after the owner, wrapping (none: the lowest) : gnt; reset: none);
  - read latency: rd_rsp_valid[c][b] is exactly the client of the read accepted on bank b RD_LAT
    cycles earlier, 0 otherwise;
  - data per bank, byte-exact against the simulator's SPM; reads accepted at an edge see every write
    accepted at earlier edges and not the one accepted at the same edge (read-first);
  - forward progress: a waiting request is accepted within NCL - 1 cycles once its port has an owner
    (CC §10.2 / §10.3 (iii)), and a port with a waiting requester is never idle two cycles in a row.
  Requests not accepted are held unchanged (rule R2), as the RTL clients must.
Bench 2, tb/spm/hdl/tsr_tb_spm.sv: all six clients of CC §10.2 (0-2 single port through
tsr_spm_cport, 3 the bridge through tsr_spm_cport, 4-5 dual-bank on the bank ports). A scoreboard
on tsr_spm's bank ports (the same exact model) plus checks at every client: the cport's steering
(bank = line[0], ready = the steered bank's grant) and response select, every client's response
stream, the bridge's loads and stores against the simulator's RISC-V window (tsr_bus SBA path on the
same SPM bytes: byte / half / word, line fills, accesses beyond SPM_BYTES), saturation by every
client in random and adversarial patterns with forward progress asserted (the no-deadlock argument
of CC §10.3), and the orderings the memory model needs from the RTL (a bridge store is answered
only after tsr_spm accepted it, so a later read by any client returns it; a write accepted before a
bridge load is seen by it; write -> read order per line).

pytest functions:
  test_spm[cfg]            bench 1 in one configuration
  test_spm_sys[cfg]        bench 2 in one configuration
  test_spm_lint_configs    verilator --lint-only -Wall of tsr_spm / tsr_spm_cport / tsr_spm_bridge /
                           tsr_sram_1r1w in edge configurations (simulation and synthesis views)
  test_spm_assertions[...] the simulation-only contract checks of tsr_spm fire, with the expected
                           message, in the cycle a client presents a line >= SPM lines, a line on
                           the wrong bank, or changes a request before it is accepted;
                           spm_neg_prereset: they stay quiet before the first reset (garbage on every
                           input) and are armed by it
  test_spm_time0_seeds     X-randomized start-up over many seeds must not trip the checks
cocotb tests (spm_*, sys_*) are selected by the runner. tb/tech/test_tech.py reuses spm_reset_state,
spm_directed, spm_sweep, spm_random and spm_reset_midflight on the xilinx SRAM wrapper.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
import xml.etree.ElementTree as ET
from collections import deque

import cocotb
import pytest
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge, Timer
from cocotb.utils import get_sim_time

from common import isa
from common.runner import REPO, build_root, rtl_sources, run
from common.tbutil import CLK_PERIOD_NS
from common.tbutil import rng as tb_rng
from common.tbutil import start_clock_and_reset

from spm_sim import SimSpm, mut_sources

LINE_BYTES = 64
LINE_BITS = 512
LINE_MASK = (1 << LINE_BITS) - 1
ALL_BE = (1 << LINE_BYTES) - 1
CL_BRIDGE = 3                     # TSR_SPM_CL_BRIDGE
CL_VU = 4                         # TSR_SPM_CL_VU (first dual-bank client of the wrapper)

# tsr_spm defaults (tsr_core instantiates it with SPM_BYTES, NCL = TSR_SPM_NCL and
# RD_LAT = SPM_RD_LAT = tsr_pkg TSR_DEF_SPM_RD_LAT = 2; BANK_LINES keeps the module default).
DEFAULTS = dict(SPM_BYTES=isa.U200["spm_bytes"], NCL=isa.SPM_NCL, RD_LAT=2, BANK_LINES=16384)

SMALL_TESTS = ["spm_reset_state", "spm_directed", "spm_sweep", "spm_random", "spm_rr", "spm_reset_midflight"]
FULL_TESTS = ["spm_reset_state", "spm_directed", "spm_sweep", "spm_random", "spm_rr"]
# name -> (parameter overrides, cocotb tests, random cycles). Banks: E = ceil(lines / 2) even
# lines, O = floor(lines / 2) odd lines, each in depth banks of BANK_LINES.
CONFIGS = {
    # 1024 lines: 2 banks x 2 depth banks of 256, default latency (IN_REG + RAM 1)
    "small": (dict(SPM_BYTES=64 * 1024, BANK_LINES=256), SMALL_TESTS, 20000),
    # 1000 lines: 500 + 500 = 256 + 244 each; RD_LAT 4 = IN_REG + RAM 2 + OUT_REG
    "odd_lat4": (dict(SPM_BYTES=64 * 1000, BANK_LINES=256, RD_LAT=4), SMALL_TESTS, 12000),
    # 1025 lines: E 513 = 512 + 1 (two depth banks), O 512 (one); RD_LAT 1 (no input register), 2 clients
    "lat1_2cl": (dict(SPM_BYTES=64 * 1025, BANK_LINES=512, RD_LAT=1, NCL=2), SMALL_TESTS, 12000),
    # 64 + 64 lines, one depth bank each, RD_LAT 3 (IN_REG + RAM 2), one client
    "onebank_lat3_1cl": (dict(SPM_BYTES=64 * 128, RD_LAT=3, NCL=1), SMALL_TESTS, 8000),
    # 10 lines: 5 + 5 in depth banks of 2 (3 each, not a power of two), RD_LAT 5 = IN + RAM 3 + OUT
    "banks5_lat5": (dict(SPM_BYTES=64 * 10, BANK_LINES=2, RD_LAT=5), SMALL_TESTS, 8000),
    # 3 lines: E 2 lines (one depth bank of 2), O 1 line
    "three_lines": (dict(SPM_BYTES=64 * 3, BANK_LINES=2), SMALL_TESTS, 4000),
    # a single line (bank O holds none)
    "one_line": (dict(SPM_BYTES=64), SMALL_TESTS, 4000),
    # full size exactly as tsr_core instantiates it (4 MiB, 2 x 2 depth banks of 16 K lines, RD_LAT 2)
    "full": (dict(), FULL_TESTS, 6000),
    # full size with the latency of the U200 bitstreams
    "full_lat4": (dict(RD_LAT=4), ["spm_reset_state", "spm_directed", "spm_random", "spm_rr"], 4000),
}

SYS_TESTS = ["sys_saturate", "sys_bridge", "sys_order", "sys_stream", "sys_reset"]
SYS_CONFIGS = {
    "small": (dict(SPM_BYTES=64 * 1024, BANK_LINES=256, RD_LAT=2), 6000),
    # 1000 lines (the bridge's word address reaches lines 1000..1023: access faults)
    "odd_lat4": (dict(SPM_BYTES=64 * 1000, BANK_LINES=256, RD_LAT=4), 5000),
    "lat1": (dict(SPM_BYTES=8192, BANK_LINES=32, RD_LAT=1), 5000),
    # U200 size and latency
    "u200": (dict(SPM_BYTES=4 * 1024 * 1024, BANK_LINES=16384, RD_LAT=4), 3000),
}


def spm_sources(extra=()):
    """Only the files tsr_spm needs (from rtl/filelist.f, in its order)."""
    names = ("tsr_tables_pkg.sv", "tsr_pkg.sv", "tsr_sram_1r1w.sv", "tsr_spm.sv") + tuple(extra)
    srcs = [s for s in rtl_sources() if s.name in names]
    assert sorted(s.name for s in srcs) == sorted(names), srcs
    return mut_sources(srcs)


def full_config(over: dict) -> dict:
    cfg = dict(DEFAULTS)
    cfg.update(over)
    lines = cfg["SPM_BYTES"] // LINE_BYTES
    bank_lines = cfg["BANK_LINES"]
    cfg["LINES"] = lines
    cfg["AW"] = max(1, (lines - 1).bit_length())
    cfg["LINES_B"] = [(lines + 1) // 2, lines // 2]
    cfg["NDB"] = [-(-n // bank_lines) for n in cfg["LINES_B"]]
    cfg["NBANK"] = sum(cfg["NDB"])          # RAM groups g_bank[]
    return cfg


# =============================================================================================
# pytest side
# =============================================================================================
@pytest.mark.parametrize("cfg", list(CONFIGS))
def test_spm(cfg):
    over, tests, cycles = CONFIGS[cfg]
    full = full_config(over)
    run("tsr_spm", __name__, parameters=over, sources=spm_sources(), name="spm_" + cfg,
        testcase=tests,
        extra_env={"TSR_SPM_CFG": json.dumps(full), "TSR_SPM_CYCLES": str(cycles)})


TB_SV = REPO / "tb" / "spm" / "hdl" / "tsr_tb_spm.sv"


@pytest.mark.parametrize("cfg", list(SYS_CONFIGS))
def test_spm_sys(cfg):
    over, cycles = SYS_CONFIGS[cfg]
    full = full_config(dict(over, NCL=isa.SPM_NCL))
    run("tsr_tb_spm", __name__, parameters=over,
        sources=spm_sources(("tsr_spm_cport.sv", "tsr_spm_bridge.sv")), extra_sources=[TB_SV],
        name="spmsys_" + cfg, testcase=SYS_TESTS,
        extra_env={"TSR_SPM_CFG": json.dumps(full), "TSR_SPM_CYCLES": str(cycles)})


LINT_CONFIGS = [
    ("tsr_spm", {}),
    ("tsr_spm", {"SPM_BYTES": 64 * 1024, "BANK_LINES": 256}),
    ("tsr_spm", {"SPM_BYTES": 64 * 1000, "BANK_LINES": 256, "RD_LAT": 4}),
    ("tsr_spm", {"SPM_BYTES": 64 * 1025, "BANK_LINES": 512, "RD_LAT": 1, "NCL": 2}),
    ("tsr_spm", {"SPM_BYTES": 64 * 128, "RD_LAT": 3, "NCL": 1}),
    ("tsr_spm", {"SPM_BYTES": 64 * 10, "BANK_LINES": 2, "RD_LAT": 5}),
    ("tsr_spm", {"SPM_BYTES": 64}),
    ("tsr_spm", {"SPM_BYTES": 64, "RD_LAT": 1, "NCL": 1}),
    ("tsr_spm", {"SPM_BYTES": 64 * 3, "BANK_LINES": 2, "RD_LAT": 1}),
    ("tsr_spm", {"SPM_BYTES": 64 * 2, "BANK_LINES": 2, "RD_LAT": 4}),
    ("tsr_spm", {"RD_LAT": 4}),
    ("tsr_spm", {"RD_LAT": 7, "BANK_LINES": 65536}),
    ("tsr_spm_cport", {}),
    ("tsr_spm_cport", {"SPM_BYTES": 64}),
    ("tsr_spm_bridge", {}),
    ("tsr_spm_bridge", {"SPM_BYTES": 64}),
    ("tsr_spm_bridge", {"SPM_BYTES": 64 * 1000}),
    ("tsr_sram_1r1w", {}),
    ("tsr_sram_1r1w", {"DEPTH": 100, "RD_LAT": 3}),
    ("tsr_sram_1r1w", {"DEPTH": 1, "WIDTH": 72, "RD_LAT": 2}),
    ("tsr_sram_1r1w", {"DEPTH": 4096, "WIDTH": 8}),
]


def test_spm_lint_configs():
    """Edge configurations rtl/lint.sh does not cover (it lints module defaults and tsr_core)."""
    verilator = os.environ.get("VERILATOR", "verilator")
    srcs = [str(s) for s in spm_sources(("tsr_spm_cport.sv", "tsr_spm_bridge.sv"))]
    bad = []
    for top, params in LINT_CONFIGS:
        for synth in (False, True):
            cmd = [verilator, "--lint-only", "-Wall", "--top-module", top]
            cmd += ["-DSYNTHESIS"] if synth else []
            cmd += ["-G%s=%s" % kv for kv in params.items()] + srcs
            p = subprocess.run(cmd, capture_output=True, text=True)
            if p.returncode != 0:
                bad.append("%s %s%s:\n%s" % (top, params, " SYNTHESIS" if synth else "",
                                             p.stdout + p.stderr))
    assert not bad, "\n".join(bad)


# Negative tests: 1000 lines (line 1000 is representable). case -> (the one assertion message
# expected, X randomization on). spm_neg_prereset runs without X randomization so that no
# time-0 clock edge with a random rst_n can arm the checks before the test drives rst_n high.
NEG_CFG = dict(SPM_BYTES=64 * 1000, BANK_LINES=256)
NEG_CASES = {
    "spm_neg_line_range": ("client 0 bank 0 read line 1000 >= 1000 SPM lines", True),
    "spm_neg_bank": ("client 1 write line 6 on bank 1 (line[0] must be the bank)", True),
    "spm_neg_rd_hold": ("client 0 bank 1 dropped or changed a read request before it was accepted (R2)", True),
    "spm_neg_wr_hold": ("client 0 bank 1 dropped or changed a write request before it was accepted (R2)", True),
    "spm_neg_prereset": ("client 1 bank 1 write line 1001 >= 1000 SPM lines", False),
}
NEG_MARK = "TSR_NEG violation driven at"


@pytest.mark.parametrize("case", list(NEG_CASES))
def test_spm_assertions(case, capfd):
    """The simulation-only checks in tsr_spm stop the simulation on client contract violations
    (the cocotb test itself would pass): the run must fail, through the intended assertion, at the
    first clock edge that samples the violation (the cocotb test logs when it drives it)."""
    expect, x_random = NEG_CASES[case]
    full = full_config(NEG_CFG)
    t0 = time.time()
    with pytest.raises((SystemExit, RuntimeError)):
        run("tsr_spm", __name__, parameters=NEG_CFG, sources=spm_sources(),
            name="spm_neg" if x_random else "spm_negx0", x_random=x_random, testcase=case,
            extra_env={"TSR_SPM_CFG": json.dumps(full)})
    cap = capfd.readouterr()
    log = cap.out + cap.err
    tail = "\n".join(log.splitlines()[-40:])
    # exactly one assertion fired, and it is the intended one
    errs = [ln for ln in log.splitlines() if "%Error" in ln and "Assertion failed" in ln]
    assert len(errs) == 1 and expect in errs[0], "expected one '%s' assertion, got %s\n%s" % (
        expect, errs, tail)
    # the simulator stopped (the RTL assertion's $stop): cocotb records that as SimFailure
    xmls = [p for p in build_root().glob("spm_neg*/*%s*.result.xml" % case)
            if p.stat().st_mtime >= t0 - 1]
    assert xmls, "no fresh cocotb results file for %s" % case
    tc = [t for t in ET.parse(max(xmls, key=lambda p: p.stat().st_mtime)).iter("testcase")
          if t.get("name") == case]
    assert tc, "testcase %s missing from the results" % case
    fail = tc[0].find("failure")
    assert fail is not None and fail.get("error_type") == "SimFailure", ET.tostring(tc[0])
    # ... at the edge right after the violation was driven (which is after the reset release)
    m = re.search(re.escape(NEG_MARK) + r" ([0-9.]+) ns", log)
    assert m, "the cocotb test did not reach the violation\n" + tail
    t_v = float(m.group(1))
    t_stop = float(tc[0].get("sim_time_ns"))
    assert t_v < t_stop <= t_v + CLK_PERIOD_NS, "violation driven at %.1f ns, stop at %.1f ns" % (
        t_v, t_stop)


# Seeds for the start-up regression: with the checks armed from time 0, about
# one seed in ten stopped at 0 ns (seeds 5, 11, 19, 40 among them).
TIME0_SEEDS = list(range(1, 33)) + [40]


def test_spm_time0_seeds():
    """X-randomized start-up must not trip the checks: Verilator's first evaluation may see a
    rising clock edge with random rst_n (high), random inputs and random check state before any
    testbench drives rst_n. One build (config small), spm_reset_state on many seeds."""
    over = CONFIGS["small"][0]
    full = full_config(over)
    bad = []
    for seed in TIME0_SEEDS:
        try:
            run("tsr_spm", __name__, parameters=over, sources=spm_sources(), name="spm_small",
                testcase="spm_reset_state", seed=seed, extra_env={"TSR_SPM_CFG": json.dumps(full)})
        except (SystemExit, RuntimeError):
            bad.append(seed)
    assert not bad, "start-up failures with seeds %s" % bad


# =============================================================================================
# Shared model pieces
# =============================================================================================
def rr_next(req: int, gnt: int) -> int:
    """tsr_spm's round robin: the lowest requester above the owner, else the lowest requester."""
    above = req & ~(gnt | (gnt - 1)) if gnt else 0
    pick = above or req
    return pick & -pick


def bits(v: int):
    i = 0
    while v:
        if v & 1:
            yield i
        v >>= 1
        i += 1


def fld(v: int, i: int, w: int) -> int:
    return (v >> (i * w)) & ((1 << w) - 1)


def first_diff(got: int, want: int, known: int):
    diff = (got ^ want) & known
    if not diff:
        return None
    b = next(j for j in range(LINE_BYTES) if (diff >> (8 * j)) & 0xFF)
    n = sum(1 for j in range(LINE_BYTES) if (diff >> (8 * j)) & 0xFF)
    return "%d byte(s) differ, first byte %d got %#04x expected %#04x" % (
        n, b, (got >> (8 * b)) & 0xFF, (want >> (8 * b)) & 0xFF)


class PortModel:
    """Exact model of one bank port (read or write) of tsr_spm: the registered round-robin grant,
    the wait counters of the requesters and the no-idle-twice check."""

    def __init__(self, ncl: int, name: str):
        self.ncl, self.name = ncl, name
        self.gnt = 0
        self.wait = [0] * ncl
        self.max_wait = 0
        self.idle_waiting = 0      # consecutive cycles with a requester and no transfer
        self.busy = 0              # cycles with a transfer
        self.reqd = 0              # cycles with a requester

    def cycle(self, req: int, cyc: int):
        """req: requesting clients this cycle. Returns the client that transfers (or None) and
        advances the grant."""
        take = req & self.gnt
        if req:
            self.reqd += 1
        if take:
            self.busy += 1
            self.idle_waiting = 0
        elif req:
            self.idle_waiting += 1
            assert self.idle_waiting <= 1, "cycle %d: %s idle %d cycles with requesters %#x" % (
                cyc, self.name, self.idle_waiting, req)
        for c in range(self.ncl):
            if (req >> c) & 1 and not (take >> c) & 1:
                if self.gnt:
                    self.wait[c] += 1
                    self.max_wait = max(self.max_wait, self.wait[c])
                    assert self.wait[c] <= self.ncl - 1, (
                        "cycle %d: %s client %d waited %d cycles (round-robin bound NCL - 1 = %d)" % (
                            cyc, self.name, c, self.wait[c], self.ncl - 1))
            else:
                self.wait[c] = 0
        if req:
            self.gnt = rr_next(req, self.gnt)
        return (take.bit_length() - 1) if take else None


def bench_cfg():
    return json.loads(os.environ["TSR_SPM_CFG"])


def bench_cycles(default: int) -> int:
    return int(os.environ.get("TSR_SPM_CYCLES", str(default)))


def lines_of_bank(cfg, b):
    return cfg["LINES_B"][b]


def special_lines(cfg):
    """First / last line of each bank, each depth-bank boundary +-1 (bank-local), the middle."""
    lines, bl = cfg["LINES"], cfg["BANK_LINES"]
    s = {0, lines - 1, lines // 2, 1, lines - 2}
    for b in (0, 1):
        for d in range(1, cfg["NDB"][b]):
            for da in (-1, 0, 1):
                s.add(2 * (d * bl + da) + b)
    return sorted(x for x in s if 0 <= x < lines)


def pick_be(rnd):
    r = rnd.random()
    if r < 0.6:
        return ALL_BE
    if r < 0.85:
        return rnd.getrandbits(LINE_BYTES)
    if r < 0.92:
        return 1 << rnd.randrange(LINE_BYTES)
    return 0 if r < 0.95 else ALL_BE ^ (1 << rnd.randrange(LINE_BYTES))


# =============================================================================================
# Bench 1: tsr_spm alone
# =============================================================================================
class SpmBench:
    """Drives tsr_spm's (client, bank) ports and checks every cycle against the contract model.

    Usage: set self.rd[c][b] = line / self.wr[c][b] = (line, data, be) for idle ports (line[0] = b),
    then `taken_rd, taken_wr = await self.step()` (lists of (c, b)). Requests not accepted stay in
    place (R2). step() is entered right after a rising edge and returns right after the next one."""

    def __init__(self, dut, cfg: dict):
        self.dut = dut
        self.cfg = cfg
        self.ncl = cfg["NCL"]
        self.lines = cfg["LINES"]
        self.aw = cfg["AW"]
        self.rd_lat = cfg["RD_LAT"]
        self.rd = [[None, None] for _ in range(self.ncl)]
        self.wr = [[None, None] for _ in range(self.ncl)]
        self.mem = SimSpm(cfg["SPM_BYTES"])
        self.cyc = 0
        self.reset_model()
        self.n_rd = self.n_wr = self.n_rsp = self.n_rsp_bytes_checked = 0
        self.collisions = 0          # same-edge read + write of one line
        self.raw_next = 0            # read accepted the edge after a write of that line
        self.contention = 0          # cycles with >= 2 clients requesting one bank port
        self.dual = 0                # cycles with one client transferring on both banks
        self._last = None
        self._last_wr_lines = set()

    def reset_model(self):
        self.mem.known = {}          # contents undefined (start, and after a hardware reset)
        self.pend = [{}, {}]         # bank -> cycle -> (client, line, expected data, mask)
        self.prd = [PortModel(self.ncl, "bank %d read port" % b) for b in (0, 1)]
        self.pwr = [PortModel(self.ncl, "bank %d write port" % b) for b in (0, 1)]

    def max_wait(self):
        return max(p.max_wait for p in self.prd + self.pwr)

    # -- signals -------------------------------------------------------------------------------
    def drive(self):
        rv = rl = wv = wl = wd = wb = 0
        for c in range(self.ncl):
            for b in (0, 1):
                i = 2 * c + b
                if self.rd[c][b] is not None:
                    rv |= 1 << i
                    rl |= self.rd[c][b] << (i * self.aw)
                if self.wr[c][b] is not None:
                    line, data, be = self.wr[c][b]
                    wv |= 1 << i
                    wl |= line << (i * self.aw)
                    wd |= data << (i * LINE_BITS)
                    wb |= be << (i * LINE_BYTES)
        cur = (rv, rl, wv, wl, wd, wb)
        if cur != self._last:
            d = self.dut
            d.rd_req_valid.value = rv
            d.rd_req_line.value = rl
            d.wr_valid.value = wv
            d.wr_line.value = wl
            d.wr_data.value = wd
            d.wr_be.value = wb
            self._last = cur
        return rv, wv

    def grants(self, ports):
        return sum(((p.gnt >> c) & 1) << (2 * c + b) for b, p in enumerate(ports) for c in range(self.ncl))

    async def step(self):
        d = self.dut
        rv, wv = self.drive()
        await FallingEdge(d.clk)
        # ---- outputs of this cycle ----
        rdy_r = int(d.rd_req_ready.value)
        rdy_w = int(d.wr_ready.value)
        assert rdy_r == self.grants(self.prd), "cycle %d: rd_req_ready %#x, expected grants %#x" % (
            self.cyc, rdy_r, self.grants(self.prd))
        assert rdy_w == self.grants(self.pwr), "cycle %d: wr_ready %#x, expected grants %#x" % (
            self.cyc, rdy_w, self.grants(self.pwr))
        rsp_v = int(d.rd_rsp_valid.value)
        exp_v = 0
        exp = [self.pend[b].pop(self.cyc, None) for b in (0, 1)]
        for b, e in enumerate(exp):
            if e is not None:
                exp_v |= 1 << (2 * e[0] + b)
        assert rsp_v == exp_v, "cycle %d: rd_rsp_valid %#x, expected %#x" % (self.cyc, rsp_v, exp_v)
        if exp_v:
            got_all = int(d.rd_rsp_data.value)
            for b, e in enumerate(exp):
                if e is None:
                    continue
                c, line, data, known = e
                bad = first_diff(fld(got_all, b, LINE_BITS), data, known)
                if bad:
                    raise AssertionError("cycle %d: client %d bank %d line %d: %s" % (self.cyc, c, b, line, bad))
                self.n_rsp += 1
                self.n_rsp_bytes_checked += bin(known).count("1") // 8
        # ---- transfers at the coming edge: reads see the state before this edge's writes ----
        taken_rd, taken_wr = [], []
        for b in (0, 1):
            req_r = sum(((rv >> (2 * c + b)) & 1) << c for c in range(self.ncl))
            req_w = sum(((wv >> (2 * c + b)) & 1) << c for c in range(self.ncl))
            if (req_r & (req_r - 1)) or (req_w & (req_w - 1)):
                self.contention += 1
            cr = self.prd[b].cycle(req_r, self.cyc)
            cw = self.pwr[b].cycle(req_w, self.cyc)
            assert (cr is not None) == any((rdy_r & rv) >> (2 * c + b) & 1 for c in range(self.ncl))
            if cr is not None:
                taken_rd.append((cr, b))
            if cw is not None:
                taken_wr.append((cw, b))
        rd_lines = set()
        for c, b in taken_rd:
            line = self.rd[c][b]
            # an illegal line (negative tests) is left to tsr_spm's check, which stops the run
            data, known = self.mem.read_line(line) if line < self.lines else (0, 0)
            due = self.cyc + self.rd_lat
            assert due not in self.pend[b]
            self.pend[b][due] = (c, line, data, known)
            rd_lines.add(line)
            if line in self._last_wr_lines:
                self.raw_next += 1
            self.n_rd += 1
        wr_lines = set()
        for c, b in taken_wr:
            line, data, be = self.wr[c][b]
            if line in rd_lines:
                self.collisions += 1
            if line < self.lines:
                self.mem.write_line(line, data, be)
            wr_lines.add(line)
            self.n_wr += 1
        self._last_wr_lines = wr_lines
        if any(x[0] == y[0] and x[1] != y[1] for x in taken_rd + taken_wr for y in taken_rd + taken_wr):
            self.dual += 1
        for c, b in taken_rd:
            self.rd[c][b] = None
        for c, b in taken_wr:
            self.wr[c][b] = None
        await RisingEdge(d.clk)
        self.cyc += 1
        return taken_rd, taken_wr

    def idle(self):
        return all(x is None for r in self.rd + self.wr for x in r)

    async def drain(self):
        """Stop issuing, wait until every response has been checked (plus a few idle cycles)."""
        n = 0
        while not self.idle():
            await self.step()
            n += 1
            assert n < 1000, "requests not accepted"
        for _ in range(self.rd_lat + 3):
            await self.step()
        assert not self.pend[0] and not self.pend[1], "responses missing: %s" % self.pend


def init_inputs(dut):
    dut.rd_req_valid.value = 0
    dut.rd_req_line.value = 0
    dut.wr_valid.value = 0
    dut.wr_line.value = 0
    dut.wr_data.value = 0
    dut.wr_be.value = 0


async def setup(dut):
    init_inputs(dut)
    await start_clock_and_reset(dut)
    return SpmBench(dut, bench_cfg())


def bank_line(cfg, rnd, b, pool=None):
    """A random line of bank b (from pool if given), None if the bank holds no line."""
    if pool:
        cand = [x for x in pool if x % 2 == b]
        if cand:
            return rnd.choice(cand)
    n = lines_of_bank(cfg, b)
    return 2 * rnd.randrange(n) + b if n else None


# =============================================================================================
# cocotb tests, bench 1
# =============================================================================================
@cocotb.test()
async def spm_reset_state(dut):
    """During and right after reset: no grant (ready 0), no response; idle stays quiet."""
    init_inputs(dut)
    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, unit="ns").start())
    dut.rst_n.value = 0
    for _ in range(4):
        await RisingEdge(dut.clk)
    for _ in range(3):
        await FallingEdge(dut.clk)
        assert int(dut.rd_req_ready.value) == 0 and int(dut.wr_ready.value) == 0
        assert int(dut.rd_rsp_valid.value) == 0
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    sb = SpmBench(dut, bench_cfg())
    for _ in range(20):
        await sb.step()     # checks ready == 0 (no grant yet) and rd_rsp_valid == 0


@cocotb.test()
async def spm_directed(dut):
    """Exact-cycle scenarios per bank: first-request bubble, write -> read next cycle (new data),
    same-cycle read + write of one line (old data), byte-enable merging, latency of an isolated
    read, one client on both banks in one cycle (two reads and two writes), client hand-over,
    round-robin order of simultaneous requesters, a single-port stream alternating banks at one
    line per cycle (both grants kept), special lines."""
    sb = await setup(dut)
    cfg = sb.cfg
    rnd = tb_rng(11)
    ncl = sb.ncl
    a, z = 0, ncl - 1                       # two clients (the same one if NCL = 1)
    two = sb.lines >= 2
    LE = (sb.lines // 2) & ~1               # an even line, and an odd one
    LO = LE + 1 if LE + 1 < sb.lines else 1
    banks = [(0, LE)] + ([(1, LO)] if two else [])

    for b, L in banks:
        # first request on this bank waits one cycle for the grant (no owner yet), then streams
        sb.wr[a][b] = (L, rnd.getrandbits(LINE_BITS), ALL_BE)
        assert (await sb.step())[1] == [], "write accepted without a grant"
        assert (await sb.step())[1] == [(a, b)]
        sb.rd[z][b] = L
        assert (await sb.step())[0] == []
        assert (await sb.step())[0] == [(z, b)]
        # write L, then read L in the next cycle: new data (checked by the model)
        sb.wr[a][b] = (L, rnd.getrandbits(LINE_BITS), ALL_BE)
        assert (await sb.step())[1] == [(a, b)]
        sb.rd[z][b] = L
        assert (await sb.step())[0] == [(z, b)]
        # same-cycle read and write of L: the read returns the old data
        c0 = sb.collisions
        sb.rd[z][b] = L
        sb.wr[a][b] = (L, rnd.getrandbits(LINE_BITS), ALL_BE)
        t_rd, t_wr = await sb.step()
        assert t_rd == [(z, b)] and t_wr == [(a, b)] and sb.collisions == c0 + 1
        # byte-enable merges: partial writes back to back, then read
        for be in (0x5555_5555_5555_5555, 0x0000_FFFF_0000_0001, 0):
            sb.wr[a][b] = (L, rnd.getrandbits(LINE_BITS), be)
            assert (await sb.step())[1] == [(a, b)]
        sb.rd[z][b] = L
        assert (await sb.step())[0] == [(z, b)]
        await sb.drain()
        # isolated read: response exactly RD_LAT cycles later, nothing else (checked every cycle)
        sb.rd[z][b] = L
        await sb.step()
        for _ in range(sb.rd_lat + 2):
            await sb.step()

    if two:
        # one client on both banks in one cycle: warm up its four grants, then 2 reads + 2 writes
        for b, L in banks:
            sb.rd[a][b] = L
            sb.wr[a][b] = (L, rnd.getrandbits(LINE_BITS), ALL_BE)
        await sb.drain()
        d0 = sb.dual
        for k in range(8):
            for b, L in banks:
                sb.rd[a][b] = L
                sb.wr[a][b] = (L, rnd.getrandbits(LINE_BITS), rnd.getrandbits(LINE_BYTES))
            t_rd, t_wr = await sb.step()
            assert sorted(t_rd) == [(a, 0), (a, 1)] and sorted(t_wr) == [(a, 0), (a, 1)], (k, t_rd, t_wr)
        assert sb.dual >= d0 + 8
        await sb.drain()

    if ncl > 1:
        # hand-over between clients on bank 0: one-cycle grant bubble
        sb.wr[z][0] = (LE, rnd.getrandbits(LINE_BITS), ALL_BE)
        assert (await sb.step())[1] == []                    # the grant is a's
        assert (await sb.step())[1] == [(z, 0)]
        # round-robin order: every client requests at once; the owner (z) goes last
        for c in range(ncl):
            sb.wr[c][0] = (LE, rnd.getrandbits(LINE_BITS), ALL_BE)
        order = []
        for _ in range(ncl):
            order += [c for c, _ in (await sb.step())[1]]
        want = [z] + list(range(ncl - 1))                  # the owner z first, then 0, 1, ..
        assert order == want, "round-robin order %s, expected %s" % (order, want)
        await sb.drain()

    # single-port stream over consecutive lines (alternating banks), write and read one line per
    # cycle each after the warm-up of both grants (client a writes, client z reads, as a cport does)
    seq = [x for x in range(min(sb.lines, 64))] + special_lines(cfg)
    for c, port in ((a, "wr"), (z, "rd")):
        for b, L in banks:
            if port == "wr":
                sb.wr[c][b] = (L, rnd.getrandbits(LINE_BITS), ALL_BE)
            else:
                sb.rd[c][b] = L
    await sb.drain()
    for i, line in enumerate(seq):
        sb.wr[a][line % 2] = (line, rnd.getrandbits(LINE_BITS), ALL_BE)
        if i >= 1:
            sb.rd[z][seq[i - 1] % 2] = seq[i - 1]      # the line written in the previous cycle
        t_rd, t_wr = await sb.step()
        assert len(t_wr) == 1, "write stream stalled at %d (line %d)" % (i, line)
        if i >= 1:
            assert len(t_rd) == 1, "read stream stalled at %d" % i
    await sb.drain()
    assert sb.raw_next >= len(seq) - 2
    dut._log.info("directed: %d reads, %d writes, %d responses checked, %d collisions, %d dual-bank cycles",
                  sb.n_rd, sb.n_wr, sb.n_rsp, sb.collisions, sb.dual)


@cocotb.test()
async def spm_sweep(dut):
    """Every line written (unique random contents, full lines) and read back, two lines per cycle
    (bank E and bank O streams of one client in the same cycles); then a second pass with random
    byte enables by other clients, descending order, read back. Catches aliasing between lines,
    banks and depth banks."""
    sb = await setup(dut)
    rnd = tb_rng(12)
    ncl = sb.ncl

    async def stream(kind, client, items):
        """Offer items on one client, each on its line's bank, as fast as accepted (both banks in
        parallel). Returns cycles until the last offer was accepted."""
        q = [deque(x for x in items if (x if kind == "rd" else x[0]) % 2 == b) for b in (0, 1)]
        cycles = 0
        while q[0] or q[1] or not sb.idle():
            for b in (0, 1):
                slot = sb.rd if kind == "rd" else sb.wr
                if slot[client][b] is None and q[b]:
                    slot[client][b] = q[b].popleft()
            await sb.step()
            cycles += 1
        await sb.drain()
        return cycles

    def wr_items(order, be_fn):
        return [(line, (rnd.getrandbits(LINE_BITS - 32) << 32) | line, be_fn()) for line in order]

    order = list(range(sb.lines))
    nE = sb.cfg["LINES_B"][0]
    c_w, c_r = ncl - 1, 0
    cyc = await stream("wr", c_w, wr_items(order, lambda: ALL_BE))
    assert cyc <= nE + 2, "write sweep not two lines per cycle: %d cycles for %d lines" % (cyc, sb.lines)
    rnd.shuffle(order)
    cyc = await stream("rd", c_r, order)
    assert cyc <= nE + 2 + sb.lines // 8, "read sweep too slow: %d cycles for %d lines" % (cyc, sb.lines)
    order.sort(reverse=True)
    c_w, c_r = min(1, ncl - 1), ncl - 1
    await stream("wr", c_w, wr_items(order, lambda: rnd.getrandbits(LINE_BYTES)))
    await stream("rd", c_r, order)
    assert sb.n_rsp == 2 * sb.lines
    dut._log.info("sweep: %d lines x 2 passes, %d bytes compared", sb.lines, sb.n_rsp_bytes_checked)


async def random_phase(sb, rnd, cycles, p_rd, p_wr, hot):
    """Random traffic on every (client, bank) port: idle ports issue with probability p_rd[c] /
    p_wr[c] per cycle; lines from a hot set (collisions, write->read), special lines or uniform."""
    special = special_lines(sb.cfg)
    for _ in range(cycles):
        for c in range(sb.ncl):
            for b in (0, 1):
                r = rnd.random()
                pool = hot if r < 0.3 else (special if r < 0.45 else None)
                if sb.rd[c][b] is None and rnd.random() < p_rd[c]:
                    sb.rd[c][b] = bank_line(sb.cfg, rnd, b, pool)
                if sb.wr[c][b] is None and rnd.random() < p_wr[c]:
                    line = bank_line(sb.cfg, rnd, b, pool)
                    if line is not None:
                        sb.wr[c][b] = (line, rnd.getrandbits(LINE_BITS), pick_be(rnd))
        await sb.step()


@cocotb.test()
async def spm_random(dut):
    """Random reads / writes from every client on both banks (contention, stalls, back-to-back,
    collisions, partial byte enables, two lines per client and cycle) over the whole address
    range, checked every cycle."""
    sb = await setup(dut)
    rnd = tb_rng(13)
    total = bench_cycles(10000)
    ncl = sb.ncl
    hot = sorted({rnd.randrange(sb.lines) for _ in range(4)} | {0, sb.lines - 1, min(1, sb.lines - 1)})
    # preload a part of the SPM so that early reads compare data too
    for line in special_lines(sb.cfg) + hot:
        sb.wr[0][line % 2] = (line, rnd.getrandbits(LINE_BITS), ALL_BE)
        while sb.wr[0][line % 2] is not None:
            await sb.step()
    phases = [
        # (fraction, p_rd per client, p_wr per client)
        (0.25, [0.5] * ncl, [0.5] * ncl),                      # mixed, contention
        (0.25, [1.0] + [0.0] * (ncl - 1), [1.0] + [0.0] * (ncl - 1)),   # one client on both banks
        (0.20, [0.15] * ncl, [0.15] * ncl),                    # sparse
        (0.15, [1.0] * ncl, [1.0] * ncl),                      # everybody saturating
        (0.15, [rnd.random() for _ in range(ncl)], [rnd.random() for _ in range(ncl)]),
    ]
    for frac, p_rd, p_wr in phases:
        await random_phase(sb, rnd, int(total * frac), p_rd, p_wr, hot)
    await sb.drain()
    dut._log.info("random: %d cycles, %d reads, %d writes, %d responses (%d bytes) checked, "
                  "%d same-edge collisions, %d write->next-cycle reads, %d contention cycles, "
                  "%d dual-bank cycles, max wait %d", sb.cyc, sb.n_rd, sb.n_wr, sb.n_rsp,
                  sb.n_rsp_bytes_checked, sb.collisions, sb.raw_next, sb.contention, sb.dual, sb.max_wait())
    assert sb.n_rsp == sb.n_rd and sb.n_rd > total // 4 and sb.n_wr > total // 4
    if sb.lines > 1:
        assert sb.raw_next > 0 and sb.dual > 0
    assert sb.collisions > 0
    if ncl > 1:
        assert sb.contention > 0 and sb.max_wait() == ncl - 1


@cocotb.test()
async def spm_rr(dut):
    """Round robin (CC §10.2): every client requesting one bank port every cycle (reads and writes,
    each bank alone, then both) gets an equal share (within one transfer), every wait is <= NCL - 1
    and the bound is reached; the port never idles while a request waits (except the one-cycle grant
    move); the owner keeps the grant while the port idles (a stream that pauses resumes without a
    bubble); a rarely requesting client next to greedy ones is granted within NCL - 1 cycles."""
    sb = await setup(dut)
    rnd = tb_rng(16)
    ncl = sb.ncl
    banks = (0, 1) if sb.lines >= 2 else (0,)
    for sel in [(b,) for b in banks] + ([banks] if len(banks) == 2 else []):
        n_rd = [0] * ncl
        n_wr = [0] * ncl
        for cyc in range(30 * ncl + 4):
            for c in range(ncl):
                for b in sel:
                    if sb.rd[c][b] is None:
                        sb.rd[c][b] = bank_line(sb.cfg, rnd, b)
                    if sb.wr[c][b] is None:
                        sb.wr[c][b] = (bank_line(sb.cfg, rnd, b), rnd.getrandbits(LINE_BITS), pick_be(rnd))
            t_rd, t_wr = await sb.step()
            if cyc >= 2:                           # after the first grant of every port
                for c, _ in t_rd:
                    n_rd[c] += 1
                for c, _ in t_wr:
                    n_wr[c] += 1
        await sb.drain()
        for n in (n_rd, n_wr):
            per = [x / len(sel) for x in n]
            assert max(per) - min(per) <= 1, "unequal round-robin shares on banks %s: %s" % (sel, n)
        if ncl > 1:
            assert sb.max_wait() == ncl - 1
    # the owner keeps the grant over idle cycles: a pausing stream resumes without a bubble
    for b in banks:
        L = bank_line(sb.cfg, rnd, b)
        sb.rd[0][b] = L
        await sb.drain()
        for gap in (0, 1, 5):
            for _ in range(gap):
                await sb.step()
            sb.rd[0][b] = L
            assert (await sb.step())[0] == [(0, b)], "owner lost the grant over %d idle cycles" % gap
    # greedy clients 0 .. NCL-2 re-raise every cycle; the last client asks every 7 cycles
    if ncl > 1:
        late = ncl - 1
        accepted = 0
        for cyc in range(40 * ncl):
            for c in range(ncl - 1):
                for b in banks:
                    if sb.wr[c][b] is None:
                        sb.wr[c][b] = (bank_line(sb.cfg, rnd, b), rnd.getrandbits(LINE_BITS), ALL_BE)
            if cyc % 7 == 0 and sb.wr[late][0] is None:
                sb.wr[late][0] = (bank_line(sb.cfg, rnd, 0), rnd.getrandbits(LINE_BITS), ALL_BE)
            t_rd, t_wr = await sb.step()
            accepted += sum(1 for c, _ in t_wr if c == late)
        await sb.drain()
        assert accepted >= (40 * ncl) // 7 - 1, "late client accepted %d times" % accepted
    util = [p.busy / max(1, p.reqd) for p in sb.prd + sb.pwr if p.reqd]
    dut._log.info("rr: max wait %d (bound %d), port utilization while requested %s", sb.max_wait(), ncl - 1,
                  ["%.3f" % u for u in util])


@cocotb.test()
async def spm_reset_midflight(dut):
    """Reset while reads are in flight on both banks: no response comes out, grants are dropped,
    and the SPM works normally afterward."""
    sb = await setup(dut)
    rnd = tb_rng(14)
    hot = [0, sb.lines - 1]
    await random_phase(sb, rnd, 200, [0.7] * sb.ncl, [0.7] * sb.ncl, hot)
    # keep the read ports busy (3 reads accepted back to back), then reset in the middle of the
    # response pipeline; requests still waiting are withdrawn together with the reset
    accepted = 0
    while accepted < 3:
        for b in ((0, 1) if sb.lines >= 2 else (0,)):
            if sb.rd[0][b] is None:
                sb.rd[0][b] = bank_line(sb.cfg, rnd, b)
        t_rd, _ = await sb.step()
        accepted = accepted + 1 if (0, 0) in t_rd else 0
    sb.rd = [[None, None] for _ in range(sb.ncl)]
    sb.wr = [[None, None] for _ in range(sb.ncl)]
    sb.drive()
    dut.rst_n.value = 0
    # Synchronous reset: it takes effect at the next edge; responses due after it are dropped
    # (with RD_LAT 1 the last response is due in this very cycle, so none is in flight).
    if sb.rd_lat >= 2:
        assert any(due > sb.cyc for p in sb.pend for due in p), "no read in flight across the reset edge"
    await RisingEdge(dut.clk)
    for _ in range(sb.rd_lat + 3):
        await FallingEdge(dut.clk)
        assert int(dut.rd_rsp_valid.value) == 0, "response after reset"
        assert int(dut.rd_req_ready.value) == 0 and int(dut.wr_ready.value) == 0
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    sb.reset_model()            # contents are undefined after a hardware reset (not checked)
    sb.cyc += 10_000            # forget the old cycle numbering
    await random_phase(sb, rnd, 1500, [0.5] * sb.ncl, [0.5] * sb.ncl, hot)
    await sb.drain()
    assert sb.n_rsp > 0


# ---- negative tests (the simulation must stop on the tsr_spm assertion) -------------------------
def mark_violation(dut):
    """Logged right where a negative test drives its violation (test_spm_assertions parses it)."""
    dut._log.info("%s %.1f ns", NEG_MARK, get_sim_time("ns"))


@cocotb.test()
async def spm_neg_line_range(dut):
    sb = await setup(dut)
    sb.rd[0][0] = sb.lines       # one past the end (1000: even)
    mark_violation(dut)
    for _ in range(4):
        await sb.step()
        sb.rd[0][0] = None


@cocotb.test()
async def spm_neg_bank(dut):
    sb = await setup(dut)
    sb.wr[1][1] = (6, 1, ALL_BE)   # an even line on bank O
    mark_violation(dut)
    for _ in range(4):
        await sb.step()
        sb.wr[1][1] = None


@cocotb.test()
async def spm_neg_rd_hold(dut):
    sb = await setup(dut)
    sb.rd[0][1] = 5
    await sb.step()              # no grant yet: not accepted
    sb.rd[0][1] = 7              # R2 violation: payload changed while waiting
    mark_violation(dut)
    for _ in range(4):
        await sb.step()


@cocotb.test()
async def spm_neg_wr_hold(dut):
    sb = await setup(dut)
    sb.wr[0][1] = (5, 1, ALL_BE)
    await sb.step()              # not accepted
    sb.wr[0][1] = None           # R2 violation: valid dropped while waiting
    mark_violation(dut)
    for _ in range(4):
        await sb.step()


@cocotb.test()
async def spm_neg_prereset(dut):
    """Clock running with rst_n high and no reset yet (power-up, reset not asserted): every port
    gets contract-violating garbage (lines >= SPM lines, wrong banks, payloads changing and valids
    dropping while not accepted). The checks must stay quiet. Then a reset, checked legal traffic,
    and a write to line 1001 on client 1 bank 1, which must stop the simulation (checks armed)."""
    cfg = bench_cfg()
    ncl, aw, lines = cfg["NCL"], cfg["AW"], cfg["LINES"]
    assert ncl >= 2 and lines < (1 << aw), "needs 2 clients and a representable line >= LINES"
    rnd = tb_rng(15)
    init_inputs(dut)
    dut.rst_n.value = 1
    await Timer(1, unit="ns")
    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, unit="ns").start(start_high=False))
    np_ = 2 * ncl

    def pack(vals, w):
        return sum(v << (i * w) for i, v in enumerate(vals))

    for cyc in range(48):
        if cyc == 0:                 # every port: line 2^AW - 1 >= LINES
            rv = wv = (1 << np_) - 1
            rl = wl = [(1 << aw) - 1] * np_
        else:                        # random: drops, changes, out-of-range lines, wrong banks
            rv = rnd.getrandbits(np_)
            wv = rnd.getrandbits(np_)
            rl = [rnd.randrange(1 << aw) for _ in range(np_)]
            wl = [rnd.randrange(lines, 1 << aw) if rnd.random() < 0.3 else rnd.randrange(1 << aw)
                  for _ in range(np_)]
        dut.rd_req_valid.value = rv
        dut.rd_req_line.value = pack(rl, aw)
        dut.wr_valid.value = wv
        dut.wr_line.value = pack(wl, aw)
        dut.wr_data.value = rnd.getrandbits(np_ * LINE_BITS)
        dut.wr_be.value = rnd.getrandbits(np_ * LINE_BYTES)
        await RisingEdge(dut.clk)
    # reset (synchronous), inputs idle
    init_inputs(dut)
    dut.rst_n.value = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    sb = SpmBench(dut, cfg)
    await random_phase(sb, rnd, 400, [0.5] * ncl, [0.5] * ncl, [0, lines - 1])
    await sb.drain()
    assert sb.n_rsp > 0
    sb.wr[1][1] = (lines + 1, 0, ALL_BE)
    mark_violation(dut)
    for _ in range(4):
        await sb.step()


# =============================================================================================
# Bench 2: all six clients (tb/spm/hdl/tsr_tb_spm.sv)
# =============================================================================================
NP = 3                            # single-port clients 0..2 on the p_* ports


class SysBench:
    """Scoreboard of tsr_tb_spm: drives the single-port clients (p_*), the bridge (br_*) and the
    dual-bank clients (v_*), observes tsr_spm's bank ports (u_spm) and checks, every cycle:
    the exact port model (grants, waits, no idle-twice), each cport's steering and response,
    every client's responses against the simulator's SPM, and the bridge's accesses against the
    simulator's RISC-V window."""

    def __init__(self, dut, cfg: dict, rnd):
        self.dut, self.cfg, self.rnd = dut, cfg, rnd
        self.ncl = cfg["NCL"]
        self.lines, self.aw, self.rd_lat = cfg["LINES"], cfg["AW"], cfg["RD_LAT"]
        self.mem = SimSpm(cfg["SPM_BYTES"])
        self.cyc = 0
        self.p_rd = [None] * NP                      # line
        self.p_wr = [None] * NP                      # (line, data, be)
        self.v_rd = [[None, None] for _ in range(2)]
        self.v_wr = [[None, None] for _ in range(2)]
        self.br_q = deque()                          # pending RISC-V accesses
        self.br = None                               # the current one (dict)
        self.br_rsp_ready_p = 0.7
        self.br_done = []                            # finished accesses (for directed tests)
        self.reset_model()
        self.stats = dict(p_rsp=0, v_rsp=0, br_ld=0, br_st=0, br_err=0, br_lat_max=0, bytes=0)
        self._last = None

    def reset_model(self):
        self.mem.known = {}
        self.prd = [PortModel(self.ncl, "bank %d read port" % b) for b in (0, 1)]
        self.pwr = [PortModel(self.ncl, "bank %d write port" % b) for b in (0, 1)]
        self.p_exp = [dict() for _ in range(NP)]    # client -> due -> (line, data, known)
        self.v_exp = [[dict(), dict()] for _ in range(2)]
        self.br_exp = {}                             # due -> bridge read response (data, known)

    def max_wait(self):
        return max(p.max_wait for p in self.prd + self.pwr)

    # -- bridge accesses -----------------------------------------------------------------------
    def br_access(self, kind, off, size=4, val=0):
        """Queue one RISC-V access: kind 'ld' / 'st', SPM byte offset, size 1 / 2 / 4."""
        assert off % size == 0
        self.br_q.append(dict(kind=kind, off=off, size=size, val=val & ((1 << (8 * size)) - 1)))

    def br_rand(self, pool=None):
        r = self.rnd
        size = r.choice([1, 2, 4, 4])
        if pool:
            line = r.choice(pool)
        else:
            line = r.randrange(self.lines)
        off = 64 * line + r.randrange(0, 64, size)
        if r.random() < 0.02 and self.lines < (1 << self.aw):
            off = 64 * r.randrange(self.lines, 1 << self.aw) + r.randrange(0, 64, size)   # faults
        self.br_access("st" if r.random() < 0.5 else "ld", off, size, r.getrandbits(32))

    # -- signals -------------------------------------------------------------------------------
    def drive(self):
        d = self.dut
        prv = prl = pwv = pwl = pwd = pwb = 0
        for c in range(NP):
            if self.p_rd[c] is not None:
                prv |= 1 << c
                prl |= self.p_rd[c] << (c * self.aw)
            if self.p_wr[c] is not None:
                line, data, be = self.p_wr[c]
                pwv |= 1 << c
                pwl |= line << (c * self.aw)
                pwd |= data << (c * LINE_BITS)
                pwb |= be << (c * LINE_BYTES)
        vrv = vrl = vwv = vwl = vwd = vwb = 0
        for v in range(2):
            for b in (0, 1):
                i = 2 * v + b
                if self.v_rd[v][b] is not None:
                    vrv |= 1 << i
                    vrl |= self.v_rd[v][b] << (i * self.aw)
                if self.v_wr[v][b] is not None:
                    line, data, be = self.v_wr[v][b]
                    vwv |= 1 << i
                    vwl |= line << (i * self.aw)
                    vwd |= data << (i * LINE_BITS)
                    vwb |= be << (i * LINE_BYTES)
        if self.br is None and self.br_q:
            self.br = dict(self.br_q.popleft(), state="req", t_req=self.cyc)
        br = self.br
        if br is not None and br["state"] == "req":
            o = br["off"] & 3
            brv, brw, bra = 1, int(br["kind"] == "st"), br["off"] >> 2
            brb = ((1 << br["size"]) - 1) << o
            brd = (br["val"] << (8 * o)) & 0xFFFFFFFF
        else:
            brv, brw, bra, brb, brd = 0, 0, 0, 0, 0
        brr = int(self.rnd.random() < self.br_rsp_ready_p)
        cur = (prv, prl, pwv, pwl, pwd, pwb, vrv, vrl, vwv, vwl, vwd, vwb, brv, brw, bra, brb, brd)
        if cur != self._last:
            d.p_rd_req_valid.value = prv
            d.p_rd_req_line.value = prl
            d.p_wr_valid.value = pwv
            d.p_wr_line.value = pwl
            d.p_wr_data.value = pwd
            d.p_wr_be.value = pwb
            d.v_rd_req_valid.value = vrv
            d.v_rd_req_line.value = vrl
            d.v_wr_valid.value = vwv
            d.v_wr_line.value = vwl
            d.v_wr_data.value = vwd
            d.v_wr_be.value = vwb
            d.br_req_valid.value = brv
            d.br_req_we.value = brw
            d.br_req_waddr.value = bra
            d.br_req_be.value = brb
            d.br_req_wdata.value = brd
            self._last = cur
        d.br_rsp_ready.value = brr
        return brr

    def idle(self):
        return (all(x is None for x in self.p_rd + self.p_wr)
                and all(x is None for r in self.v_rd + self.v_wr for x in r)
                and self.br is None and not self.br_q)

    async def step(self):
        d, u = self.dut, self.dut.u_spm
        brr = self.drive()
        await FallingEdge(d.clk)
        cyc = self.cyc
        # ---- tsr_spm bank ports ----
        rv, rr, rl = int(u.rd_req_valid.value), int(u.rd_req_ready.value), int(u.rd_req_line.value)
        wv, wr = int(u.wr_valid.value), int(u.wr_ready.value)
        wl, wd, wb = int(u.wr_line.value), int(u.wr_data.value), int(u.wr_be.value)
        gr = sum(((p.gnt >> c) & 1) << (2 * c + b) for b, p in enumerate(self.prd) for c in range(self.ncl))
        gw = sum(((p.gnt >> c) & 1) << (2 * c + b) for b, p in enumerate(self.pwr) for c in range(self.ncl))
        assert rr == gr and wr == gw, "cycle %d: ready rd %#x wr %#x, expected grants %#x %#x" % (cyc, rr, wr, gr, gw)
        # ---- cport steering (clients 0..2 from p_*, client 3 from the bridge's single port) ----
        p_rdy_r, p_rdy_w = int(d.p_rd_req_ready.value), int(d.p_wr_ready.value)
        sp = [(self.p_rd[c], self.p_wr[c], (p_rdy_r >> c) & 1, (p_rdy_w >> c) & 1) for c in range(NP)]
        b_rv, b_wv = int(d.u_br.spm_rd_req_valid.value), int(d.u_br.spm_wr_valid.value)
        sp.append((int(d.u_br.spm_rd_req_line.value) if b_rv else None,
                   (int(d.u_br.spm_wr_line.value), int(d.u_br.spm_wr_data.value), int(d.u_br.spm_wr_be.value))
                   if b_wv else None,
                   int(d.u_br.spm_rd_req_ready.value), int(d.u_br.spm_wr_ready.value)))
        for c, (rq, wq, rdy_r, rdy_w) in enumerate(sp):
            for b in (0, 1):
                i = 2 * c + b
                want_r = rq is not None and rq % 2 == b
                assert (rv >> i) & 1 == want_r, "cycle %d: client %d bank %d read valid %d (request %s)" % (
                    cyc, c, b, (rv >> i) & 1, rq)
                if want_r:
                    assert fld(rl, i, self.aw) == rq, "cycle %d: client %d read line %d -> %d" % (
                        cyc, c, rq, fld(rl, i, self.aw))
                    assert rdy_r == (rr >> i) & 1, "cycle %d: client %d read ready != bank %d grant" % (cyc, c, b)
                want_w = wq is not None and wq[0] % 2 == b
                assert (wv >> i) & 1 == want_w, "cycle %d: client %d bank %d write valid" % (cyc, c, b)
                if want_w:
                    assert (fld(wl, i, self.aw), fld(wd, i, LINE_BITS), fld(wb, i, LINE_BYTES)) == wq, \
                        "cycle %d: client %d write payload changed by the cport" % (cyc, c)
                    assert rdy_w == (wr >> i) & 1, "cycle %d: client %d write ready != bank %d grant" % (cyc, c, b)
        # ---- dual-bank clients pass straight ----
        assert int(d.v_rd_req_ready.value) == fld(rr, 2, 4) and int(d.v_wr_ready.value) == fld(wr, 2, 4)
        # ---- responses ----
        self._check_responses(cyc, int(d.p_rd_rsp_valid.value), int(d.v_rd_rsp_valid.value))
        # ---- bridge ----
        self._bridge_sample(cyc, int(d.br_req_ready.value), int(d.br_rsp_valid.value), brr)
        # ---- transfers at the coming edge (reads first: they see the state before the writes) ----
        reads, writes = [], []
        for b in (0, 1):
            req_r = sum(((rv >> (2 * c + b)) & 1) << c for c in range(self.ncl))
            req_w = sum(((wv >> (2 * c + b)) & 1) << c for c in range(self.ncl))
            cr = self.prd[b].cycle(req_r, cyc)
            cw = self.pwr[b].cycle(req_w, cyc)
            if cr is not None:
                reads.append((cr, b, fld(rl, 2 * cr + b, self.aw)))
            if cw is not None:
                i = 2 * cw + b
                writes.append((cw, b, fld(wl, i, self.aw), fld(wd, i, LINE_BITS), fld(wb, i, LINE_BYTES)))
        due = cyc + self.rd_lat
        for c, b, line in reads:
            data, known = self.mem.read_line(line)
            if c < NP:
                assert self.p_rd[c] == line
                assert due not in self.p_exp[c]
                self.p_exp[c][due] = (line, data, known)
                self.p_rd[c] = None
            elif c == CL_BRIDGE:
                br = self.br
                assert br is not None and br["state"] == "busy" and br["kind"] == "ld", "unexpected bridge read"
                assert line == br["off"] // 64
                br["exp_line"] = (data, known)
                br["exp_val"] = self.mem.rv_load(br["off"], br["size"])     # the simulator's load
                br["state"] = "spm"
                br["t_spm"] = cyc
                self.br_exp[due] = True
            else:
                v = c - CL_VU
                assert self.v_rd[v][b] == line
                self.v_exp[v][b][due] = (line, data, known)
                self.v_rd[v][b] = None
        for c, b, line, data, be in writes:
            if c == CL_BRIDGE:
                br = self.br
                assert br is not None and br["state"] == "busy" and br["kind"] == "st", "unexpected bridge write"
                o = br["off"] % 64
                want_be = ((1 << br["size"]) - 1) << o
                assert line == br["off"] // 64 and be == want_be, "bridge write line %d be %#x, want %d %#x" % (
                    line, be, br["off"] // 64, want_be)
                got = (data >> (8 * o)) & ((1 << (8 * br["size"])) - 1)
                assert got == br["val"], "bridge write data %#x, want %#x" % (got, br["val"])
                assert self.mem.rv_store(br["off"], br["size"], br["val"])      # the simulator's store
                br["state"] = "spm"
                br["t_spm"] = cyc
            else:
                self.mem.write_line(line, data, be)
                if c < NP:
                    assert self.p_wr[c] == (line, data, be)
                    self.p_wr[c] = None
                else:
                    v = c - CL_VU
                    assert self.v_wr[v][b] == (line, data, be)
                    self.v_wr[v][b] = None
        await RisingEdge(d.clk)
        self.cyc += 1

    def _check_responses(self, cyc, p_v, v_v):
        d = self.dut
        p_data = v_data = None
        for c in range(NP):
            e = self.p_exp[c].pop(cyc, None)
            assert ((p_v >> c) & 1) == (e is not None), "cycle %d: client %d rd_rsp_valid %d, expected %s" % (
                cyc, c, (p_v >> c) & 1, e is not None)
            if e is not None:
                p_data = int(d.p_rd_rsp_data.value) if p_data is None else p_data
                bad = first_diff(fld(p_data, c, LINE_BITS), e[1], e[2])
                assert not bad, "cycle %d: client %d line %d: %s" % (cyc, c, e[0], bad)
                self.stats["p_rsp"] += 1
                self.stats["bytes"] += bin(e[2]).count("1") // 8
        for v in range(2):
            for b in (0, 1):
                e = self.v_exp[v][b].pop(cyc, None)
                i = 2 * v + b
                assert ((v_v >> i) & 1) == (e is not None), "cycle %d: client %d bank %d rd_rsp_valid" % (
                    cyc, CL_VU + v, b)
                if e is not None:
                    v_data = int(d.v_rd_rsp_data.value) if v_data is None else v_data
                    bad = first_diff(fld(v_data, b, LINE_BITS), e[1], e[2])
                    assert not bad, "cycle %d: client %d bank %d line %d: %s" % (cyc, CL_VU + v, b, e[0], bad)
                    self.stats["v_rsp"] += 1
                    self.stats["bytes"] += bin(e[2]).count("1") // 8
        if self.br_exp.pop(cyc, None):
            assert int(d.u_br.spm_rd_rsp_valid.value) == 1, "cycle %d: bridge read response missing" % cyc
            assert self.br is not None and self.br["state"] == "spm"
            self.br["state"] = "rdata"

    def _bridge_sample(self, cyc, req_ready, rsp_valid, rsp_ready):
        d, br = self.dut, self.br
        if br is None:
            assert not rsp_valid, "cycle %d: bridge response without an access" % cyc
            return
        if br["state"] == "req":
            assert not rsp_valid
            if req_ready:                    # accepted at the coming edge
                br["state"] = "busy"
                br["t_acc"] = cyc
                bad = br["off"] >= self.cfg["SPM_BYTES"]
                br["err"] = bad
                if bad:                      # the simulator faults the access too
                    assert (self.mem.rv_load(br["off"], br["size"]) if br["kind"] == "ld"
                            else not self.mem.rv_store(br["off"], br["size"], br["val"])) in (None, True)
                    br["state"] = "noacc"
            return
        if not rsp_valid:
            assert "rsp" not in br, "cycle %d: bridge rsp_valid dropped before the handshake" % cyc
            return
        # rsp_valid: only once the SPM has done the access (a store: accepted at an earlier edge)
        err = int(d.br_rsp_err.value)
        if br["state"] == "noacc":
            assert err == 1, "cycle %d: access beyond SPM_BYTES answered without rsp_err" % cyc
        else:
            assert br["state"] in ("spm", "rdata") and err == 0, "cycle %d: bridge answered in state %s err %d" % (
                cyc, br["state"], err)
            assert br["t_spm"] < cyc
            if br["kind"] == "ld":
                assert br["state"] == "rdata", "cycle %d: bridge load answered before the SPM data" % cyc
                line = int(d.br_rsp_line.value)
                data, known = br["exp_line"]
                bad = first_diff(line, data, known)
                assert not bad, "cycle %d: bridge load line (off %#x): %s" % (cyc, br["off"], bad)
                o, n = br["off"] % 64, br["size"]
                kmask = (known >> (8 * o)) & ((1 << (8 * n)) - 1)
                got = (line >> (8 * o)) & ((1 << (8 * n)) - 1)
                assert (got ^ br["exp_val"]) & kmask == 0, "cycle %d: bridge load off %#x size %d: %#x, simulator %#x" % (
                    cyc, br["off"], n, got, br["exp_val"])
                self.stats["bytes"] += bin(known).count("1") // 8
        snap = (err, int(d.br_rsp_line.value) if br["kind"] == "ld" else 0)
        if "rsp" in br:
            assert br["rsp"] == snap, "cycle %d: bridge response changed while waiting for rsp_ready" % cyc
        br["rsp"] = snap
        lat = cyc - br["t_acc"]
        self.stats["br_lat_max"] = max(self.stats["br_lat_max"], lat)
        assert lat <= self.ncl - 1 + self.rd_lat + 4 or br.get("held"), \
            "cycle %d: bridge access took %d cycles (bound NCL - 1 + RD_LAT + 4)" % (cyc, lat)
        if rsp_ready:                        # handshake at the coming edge
            self.stats["br_err" if br["state"] == "noacc" else "br_" + br["kind"]] += 1
            self.br_done.append(dict(br, t_rsp=cyc))
            self.br = None
        else:
            br["held"] = True

    async def drain(self, limit=4000):
        n = 0
        while not self.idle():
            await self.step()
            n += 1
            assert n < limit, "requests not completed (bridge %s)" % self.br
        for _ in range(self.rd_lat + 3):
            await self.step()
        assert not any(self.p_exp) and not any(x for r in self.v_exp for x in r) and not self.br_exp


def sys_init_inputs(dut):
    for n in ("p_rd_req_valid", "p_rd_req_line", "p_wr_valid", "p_wr_line", "p_wr_data", "p_wr_be",
              "v_rd_req_valid", "v_rd_req_line", "v_wr_valid", "v_wr_line", "v_wr_data", "v_wr_be",
              "br_req_valid", "br_req_we", "br_req_waddr", "br_req_be", "br_req_wdata", "br_rsp_ready"):
        getattr(dut, n).value = 0


async def sys_setup(dut, seed):
    sys_init_inputs(dut)
    await start_clock_and_reset(dut)
    return SysBench(dut, bench_cfg(), tb_rng(seed))


class Traffic:
    """Request generator of one saturation pattern for every client of SysBench."""

    def __init__(self, sb, pattern, p=1.0, hot=None):
        self.sb, self.pattern, self.p = sb, pattern, p
        self.hot = hot or [0, 1]
        self.par = [0] * NP                  # 'alternate': next parity per single-port client

    def line(self, b=None, c=None):
        sb, r = self.sb, self.sb.rnd
        if self.pattern == "one_bank":
            b = 0
        elif self.pattern == "alternate" and c is not None:
            b = self.par[c]
            self.par[c] ^= 1
        if b is None:
            b = r.randrange(2) if sb.lines >= 2 else 0
        pool = self.hot if self.pattern == "hot" else None
        return bank_line(sb.cfg, r, b, pool)

    def fill(self):
        sb, r = self.sb, self.sb.rnd
        for c in range(NP):
            if sb.p_rd[c] is None and r.random() < self.p:
                sb.p_rd[c] = self.line(c=c)
            if sb.p_wr[c] is None and r.random() < self.p:
                sb.p_wr[c] = (self.line(c=c), r.getrandbits(LINE_BITS), pick_be(r))
        for v in range(2):
            for b in ((0,) if self.pattern == "one_bank" or sb.lines < 2 else (0, 1)):
                if sb.v_rd[v][b] is None and r.random() < self.p:
                    sb.v_rd[v][b] = self.line(b)
                if sb.v_wr[v][b] is None and r.random() < self.p:
                    sb.v_wr[v][b] = (self.line(b), r.getrandbits(LINE_BITS), pick_be(r))
        if not sb.br_q and sb.br is None and r.random() < max(self.p, 0.3):
            sb.br_rand(self.hot if self.pattern == "hot" else None)


# =============================================================================================
# cocotb tests, bench 2
# =============================================================================================
@cocotb.test()
async def sys_saturate(dut):
    """Every client saturating (single-port clients one read and one write request per cycle, the
    dual-bank clients four, the bridge back to back) in random and adversarial patterns: all on one
    bank, a two-line hot spot (racing writes ordered by the arbiter, read-first collisions), every
    single-port client alternating banks in lockstep, sparse, and random mixes. Forward progress:
    every wait <= NCL - 1 cycles, no bank port idle twice with a waiting request, every bridge
    access within NCL - 1 + RD_LAT + 4 cycles, each port busy every cycle it is requested
    (saturation) except the grant moves; data exact against the simulator's SPM."""
    sb = await sys_setup(dut, 21)
    total = bench_cycles(6000)
    hot = sorted({0, 1, sb.lines - 1, sb.rnd.randrange(sb.lines)})
    # preload: every line of the hot set and some others are known, so reads compare
    for line in hot + special_lines(sb.cfg):
        sb.p_wr[0] = (line, sb.rnd.getrandbits(LINE_BITS), ALL_BE)
        while sb.p_wr[0] is not None:
            await sb.step()
    phases = [("random", 1.0), ("one_bank", 1.0), ("hot", 1.0), ("alternate", 1.0), ("random", 0.2),
              ("hot", 0.6), ("random", 0.7)]
    util = {}
    for pattern, p in phases:
        tr = Traffic(sb, pattern, p, hot)
        b0 = [(x.busy, x.reqd) for x in sb.prd + sb.pwr]
        for _ in range(total // len(phases)):
            tr.fill()
            await sb.step()
        util[(pattern, p)] = ["%.3f" % ((x.busy - b) / max(1, x.reqd - q)) for x, (b, q) in zip(sb.prd + sb.pwr, b0)]
        await sb.drain()
    dut._log.info("saturate: %d cycles, stats %s, max wait %d (bound %d), port utilization while requested %s",
                  sb.cyc, sb.stats, sb.max_wait(), sb.ncl - 1, util)
    assert sb.max_wait() == sb.ncl - 1, "the round-robin bound was never reached (no full contention)"
    assert sb.stats["br_ld"] > 20 and sb.stats["br_st"] > 20 and sb.stats["p_rsp"] > total // 4
    for (pattern, p), u in util.items():
        if p == 1.0 and pattern != "one_bank":
            assert all(float(x) >= 0.97 for x in u), "saturated ports not busy: %s %s" % (pattern, u)


@cocotb.test()
async def sys_bridge(dut):
    """The bridge against the simulator's RISC-V window: byte / half / word stores and loads at every
    offset class of a line, line fills, the first / last line, accesses beyond SPM_BYTES (rsp_err,
    the simulator faults too, no SPM access), random rsp_ready delays (response held stable), then
    random accesses interleaved with traffic of every other client on the same lines."""
    sb = await sys_setup(dut, 22)
    r = sb.rnd
    L = [0, 1, sb.lines // 2, sb.lines - 1]
    for line in L:                           # make the lines known (word stores over the whole line)
        for w in range(16):
            sb.br_access("st", 64 * line + 4 * w, 4, r.getrandbits(32))
    for line in L:
        for size in (1, 2, 4):
            for o in range(0, 64, size):
                if r.random() < 0.35 or o in (0, 64 - size):
                    sb.br_access("st", 64 * line + o, size, r.getrandbits(32))
                    sb.br_access("ld", 64 * line + (o if r.random() < 0.5 else r.randrange(0, 64, size)), size)
    if sb.lines < (1 << sb.aw):              # word addresses beyond SPM_BYTES
        for off in (64 * sb.lines, 64 * sb.lines + 60, 64 * ((1 << sb.aw) - 1) + 3):
            sb.br_access("ld", off - off % 4, 4)
            sb.br_access("st", off, 1, 0x5A)
    await sb.drain(limit=40000)
    n0 = len(sb.br_done)
    tr = Traffic(sb, "hot", 0.5, L)
    for _ in range(bench_cycles(5000) // 2):
        tr.fill()
        if not sb.br_q and sb.br is None:
            sb.br_rand(L)
        await sb.step()
    await sb.drain()
    dut._log.info("bridge: %d accesses (%d with other traffic), stats %s", len(sb.br_done),
                  len(sb.br_done) - n0, sb.stats)
    assert sb.stats["br_ld"] > 100 and sb.stats["br_st"] > 100
    if sb.lines < (1 << sb.aw):
        assert sb.stats["br_err"] >= 6


@cocotb.test()
async def sys_order(dut):
    """Orderings the memory model needs (CC §7.1, §10.5; memory model rule 2): a bridge store is
    answered only after tsr_spm accepted it, so a read by any client issued after the answer
    returns it; a unit write accepted before a bridge load is seen by the load; a bridge line fill
    after word stores returns them all; a dual-bank client writing one bank while the bridge reads
    the other, and the reverse."""
    sb = await sys_setup(dut, 23)
    r = sb.rnd
    sb.br_rsp_ready_p = 1.0
    for k in range(40):
        line = r.randrange(sb.lines)
        # bridge store, then a read of the line by client k % 3 in the cycle after the answer
        sb.br_access("st", 64 * line + 4 * r.randrange(16), 4, r.getrandbits(32))
        while sb.br is not None or sb.br_q:
            await sb.step()
        assert sb.br_done[-1]["t_spm"] < sb.br_done[-1]["t_rsp"]
        sb.p_rd[k % NP] = line
        await sb.drain()
        # unit write, then a bridge load issued in the cycle after the write was accepted
        sb.p_wr[k % NP] = (line, r.getrandbits(LINE_BITS), r.getrandbits(LINE_BYTES) | 1)
        while sb.p_wr[k % NP] is not None:
            await sb.step()
        sb.br_access("ld", 64 * line + 4 * r.randrange(16), 4)
        await sb.drain()
        # word stores then a line fill
        for w in r.sample(range(16), 3):
            sb.br_access("st", 64 * line + 4 * w, 4, r.getrandbits(32))
        sb.br_access("ld", 64 * line, 4)
        await sb.drain()
        if sb.lines >= 2:
            # dual-bank client 4 writes the other bank's line while the bridge reads this one
            other = line ^ 1 if (line ^ 1) < sb.lines else line - 1
            sb.v_wr[0][other % 2] = (other, r.getrandbits(LINE_BITS), ALL_BE)
            sb.br_access("ld", 64 * line + 4 * r.randrange(16), 4)
            sb.v_rd[1][other % 2] = other
            await sb.drain()
    dut._log.info("order: %d bridge accesses, stats %s", len(sb.br_done), sb.stats)


@cocotb.test()
async def sys_stream(dut):
    """cport streaming: one single-port client streaming consecutive lines alone (banks alternate)
    keeps both grants: one read and one write per cycle; two single-port clients on opposite
    parities stream at one line per cycle each (both banks busy every cycle)."""
    sb = await sys_setup(dut, 24)
    r = sb.rnd
    n = min(sb.lines, 256)

    async def run_streams(streams):
        """streams: list of (kind, client, deque of lines). Returns cycles."""
        cyc = 0
        while any(q for _, _, q in streams) or not sb.idle():
            for kind, c, q in streams:
                if q:
                    if kind == "rd" and sb.p_rd[c] is None:
                        sb.p_rd[c] = q.popleft()
                    elif kind == "wr" and sb.p_wr[c] is None:
                        line = q.popleft()
                        sb.p_wr[c] = (line, r.getrandbits(LINE_BITS), ALL_BE)
            await sb.step()
            cyc += 1
        await sb.drain()
        return cyc

    # warm-up: client 0 owns every grant
    await run_streams([("wr", 0, deque([0, 1])), ("rd", 0, deque([0, 1]))])
    # reads trail the writes by two lines (each read sees the line written two cycles before)
    b0 = sb.stats["bytes"]
    cyc = await run_streams([("wr", 0, deque(range(n))), ("rd", 0, deque([0, 1] + list(range(n - 2))))])
    assert cyc <= n + 2, "single-port stream: %d cycles for %d lines" % (cyc, n)
    assert sb.stats["bytes"] - b0 == 64 * n, "stream reads did not see the stream's writes"
    if sb.lines >= 4:
        ev, od = deque(range(0, n, 2)), deque(range(1, n, 2))
        await run_streams([("wr", 1, deque([0])), ("wr", 2, deque([1]))])   # grants: E-wr 1, O-wr 2
        cyc = await run_streams([("wr", 1, ev), ("wr", 2, od)])
        assert cyc <= n // 2 + 2, "two clients on opposite banks: %d cycles for %d lines" % (cyc, n)
    dut._log.info("stream: %d lines, stats %s", n, sb.stats)


@cocotb.test()
async def sys_reset(dut):
    """Reset with traffic of every client in flight (a bridge access outstanding): no response
    after the reset, the bridge is idle again (req_ready, no rsp_valid), and every client works."""
    sb = await sys_setup(dut, 25)
    tr = Traffic(sb, "random", 0.8)
    for k in range(3):
        for _ in range(300):
            tr.fill()
            await sb.step()
        while sb.br is None or sb.br["state"] == "req":
            tr.fill()
            await sb.step()
        dut.rst_n.value = 0
        sys_init_inputs(dut)
        for _ in range(sb.rd_lat + 3):
            await RisingEdge(dut.clk)
            await FallingEdge(dut.clk)
            assert int(dut.p_rd_rsp_valid.value) == 0 and int(dut.v_rd_rsp_valid.value) == 0
            assert int(dut.br_rsp_valid.value) == 0 and int(dut.br_req_ready.value) == 1
        await RisingEdge(dut.clk)
        dut.rst_n.value = 1
        await RisingEdge(dut.clk)
        cyc = sb.cyc
        sb.__init__(dut, sb.cfg, sb.rnd)
        sb.cyc = cyc + 10_000
        tr = Traffic(sb, "random", 0.8)
    for _ in range(500):
        tr.fill()
        await sb.step()
    await sb.drain()
    assert sb.stats["p_rsp"] > 0 and sb.stats["br_ld"] + sb.stats["br_st"] > 0
