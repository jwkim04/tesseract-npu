"""tsr_spm testbench (docs/rtl-m2.md §5, block (d)).

The bench drives every client port of tsr_spm from Python and checks, every cycle, against an
exact model of the contract in rtl/core/tsr_spm.sv:
  - arbitration: rd_req_ready / wr_ready equal the registered fixed-priority grant
    (gnt <= |valid ? lowest set bit of valid : gnt; reset: no grant);
  - read latency: rd_rsp_valid is exactly the one-hot client of the read accepted RD_LAT cycles
    earlier, and 0 in every other cycle;
  - data: byte-exact against a line model with byte enables; bytes never written are unknown
    (the SPM is undefined at start) and not compared;
  - order: reads accepted at an edge see every write accepted at earlier edges and not the
    write accepted at the same edge (read-first).
Requests not accepted are held unchanged (rule R2), as the RTL clients must.

pytest functions:
  test_spm[cfg]            build tsr_spm in one configuration and run the cocotb tests on it
  test_spm_lint_configs    verilator --lint-only -Wall of tsr_spm / tsr_sram_1r1w in edge
                           configurations (simulation and synthesis views)
  test_spm_assertions[...] the simulation-only contract checks of tsr_spm fire, with the expected
                           message, in the cycle a client presents a line >= SPM lines or changes a
                           request before it is accepted; spm_neg_prereset: they stay quiet before
                           the first reset (garbage on every input) and are armed by it
  test_spm_time0_seeds     X-randomised start-up over many seeds (Verilator's time-0 clock edge
                           with random rst_n / inputs / check state) must not trip the checks
cocotb tests (spm_*) are selected by the runner.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
import xml.etree.ElementTree as ET

import cocotb
import pytest
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge, Timer
from cocotb.utils import get_sim_time

from common import isa
from common.runner import build_root, rtl_sources, run
from common.tbutil import CLK_PERIOD_NS
from common.tbutil import rng as tb_rng
from common.tbutil import start_clock_and_reset

LINE_BYTES = 64
LINE_BITS = 512
LINE_MASK = (1 << LINE_BITS) - 1

# tsr_spm defaults (tsr_core instantiates it with SPM_BYTES, NCL = TSR_SPM_NCL and
# RD_LAT = SPM_RD_LAT = tsr_pkg TSR_DEF_SPM_RD_LAT = 2; BANK_LINES keeps the module default).
DEFAULTS = dict(SPM_BYTES=isa.U200_V1["spm_bytes"], NCL=isa.SPM_NCL, RD_LAT=2, BANK_LINES=16384)

# name -> (parameter overrides, cocotb tests, random cycles)
SMALL_TESTS = ["spm_reset_state", "spm_directed", "spm_sweep", "spm_random", "spm_reset_midflight"]
FULL_TESTS = ["spm_reset_state", "spm_directed", "spm_sweep", "spm_random"]
CONFIGS = {
    # 1024 lines, 4 banks of 256, default latency (IN_REG + RAM 1)
    "small": (dict(SPM_BYTES=64 * 1024, BANK_LINES=256), SMALL_TESTS, 20000),
    # 1000 lines: 3 banks of 256 + one of 232; RD_LAT 4 = IN_REG + RAM 2 + OUT_REG
    "odd_lat4": (dict(SPM_BYTES=64 * 1000, BANK_LINES=256, RD_LAT=4), SMALL_TESTS, 12000),
    # 1025 lines: bank of 1024 + bank of 1 line; RD_LAT 1 (no input register), 2 clients
    "lat1_2cl": (dict(SPM_BYTES=64 * 1025, BANK_LINES=1024, RD_LAT=1, NCL=2), SMALL_TESTS, 12000),
    # one bank of 128 lines, RD_LAT 3 (IN_REG + RAM 2), one client
    "onebank_lat3_1cl": (dict(SPM_BYTES=64 * 128, RD_LAT=3, NCL=1), SMALL_TESTS, 8000),
    # 10 lines in 5 banks of 2 (bank count not a power of two), RD_LAT 5 = IN + RAM 3 + OUT
    "banks5_lat5": (dict(SPM_BYTES=64 * 10, BANK_LINES=2, RD_LAT=5), SMALL_TESTS, 8000),
    # a single line
    "one_line": (dict(SPM_BYTES=64), SMALL_TESTS, 4000),
    # full size exactly as tsr_core instantiates it (4 MiB, 4 banks of 16 K lines, RD_LAT 2)
    "full": (dict(), FULL_TESTS, 6000),
    # full size with the latency expected for U200 timing (M4)
    "full_lat4": (dict(RD_LAT=4), ["spm_reset_state", "spm_directed", "spm_random"], 4000),
}


def spm_sources():
    """Only the files tsr_spm needs (from rtl/filelist.f, in its order)."""
    names = ("tsr_pkg.sv", "tsr_sram_1r1w.sv", "tsr_spm.sv")
    srcs = [s for s in rtl_sources() if s.name in names]
    assert [s.name for s in srcs] == list(names), srcs
    return srcs


def full_config(over: dict) -> dict:
    cfg = dict(DEFAULTS)
    cfg.update(over)
    lines = cfg["SPM_BYTES"] // LINE_BYTES
    bank_lines = cfg["BANK_LINES"]
    cfg["LINES"] = lines
    cfg["AW"] = max(1, (lines - 1).bit_length())
    cfg["NBANK"] = -(-lines // bank_lines)
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


LINT_CONFIGS = [
    ("tsr_spm", {}),
    ("tsr_spm", {"SPM_BYTES": 64 * 1024, "BANK_LINES": 256}),
    ("tsr_spm", {"SPM_BYTES": 64 * 1000, "BANK_LINES": 256, "RD_LAT": 4}),
    ("tsr_spm", {"SPM_BYTES": 64 * 1025, "BANK_LINES": 1024, "RD_LAT": 1, "NCL": 2}),
    ("tsr_spm", {"SPM_BYTES": 64 * 128, "RD_LAT": 3, "NCL": 1}),
    ("tsr_spm", {"SPM_BYTES": 64 * 10, "BANK_LINES": 2, "RD_LAT": 5}),
    ("tsr_spm", {"SPM_BYTES": 64}),
    ("tsr_spm", {"SPM_BYTES": 64, "RD_LAT": 1, "NCL": 1}),
    ("tsr_spm", {"SPM_BYTES": 64 * 3, "BANK_LINES": 2, "RD_LAT": 1}),
    ("tsr_spm", {"RD_LAT": 4}),
    ("tsr_spm", {"RD_LAT": 7, "BANK_LINES": 65536}),
    ("tsr_sram_1r1w", {}),
    ("tsr_sram_1r1w", {"DEPTH": 100, "RD_LAT": 3}),
    ("tsr_sram_1r1w", {"DEPTH": 1, "WIDTH": 72, "RD_LAT": 2}),
    ("tsr_sram_1r1w", {"DEPTH": 4096, "WIDTH": 8}),
]


def test_spm_lint_configs():
    """Edge configurations rtl/lint.sh does not cover (it lints module defaults and tsr_core)."""
    verilator = os.environ.get("VERILATOR", "verilator")
    srcs = [str(s) for s in spm_sources()]
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
# expected, X randomisation on). spm_neg_prereset runs without X randomisation so that no
# time-0 clock edge with a random rst_n can arm the checks before the test drives rst_n high.
NEG_CFG = dict(SPM_BYTES=64 * 1000, BANK_LINES=256)
NEG_CASES = {
    "spm_neg_line_range": ("client 0 read line 1000 >= 1000 SPM lines", True),
    "spm_neg_rd_hold": ("client 0 dropped or changed a read request before it was accepted (R2)", True),
    "spm_neg_wr_hold": ("client 0 dropped or changed a write request before it was accepted (R2)", True),
    "spm_neg_prereset": ("client 1 write line 1001 >= 1000 SPM lines", False),
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


# Seeds for the start-up regression: with the checks armed from time 0 (before the fix), about
# one seed in ten stopped at 0 ns (the reviewer's 5, 11, 19, 40 among them).
TIME0_SEEDS = list(range(1, 33)) + [40]


def test_spm_time0_seeds():
    """X-randomised start-up must not trip the checks: Verilator's first evaluation may see a
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
# Bench
# =============================================================================================
def be_mask(be: int) -> int:
    """64-bit byte enable -> 512-bit bit mask."""
    m = 0
    b = 0
    while be:
        if be & 1:
            m |= 0xFF << (8 * b)
        be >>= 1
        b += 1
    return m


class SpmBench:
    """Drives tsr_spm's client ports and checks every cycle against the contract model.

    Usage: set self.rd[c] = line / self.wr[c] = (line, data, be) for idle clients, then
    `taken_rd, taken_wr = await self.step()`. Requests not accepted stay in place (R2). step()
    is entered right after a rising edge and returns right after the next one."""

    def __init__(self, dut, cfg: dict):
        self.dut = dut
        self.ncl = cfg["NCL"]
        self.lines = cfg["LINES"]
        self.aw = cfg["AW"]
        self.rd_lat = cfg["RD_LAT"]
        self.bank_lines = cfg["BANK_LINES"]
        self.nbank = cfg["NBANK"]
        self.rd = [None] * self.ncl
        self.wr = [None] * self.ncl
        self.cyc = 0
        self.reset_model()
        self.n_rd = self.n_wr = self.n_rsp = self.n_rsp_bytes_checked = 0
        self.collisions = 0          # same-edge read+write of one line
        self.raw_next = 0            # read accepted the edge after a write of that line
        self.contention = 0          # cycles with >= 2 clients requesting one port
        self._last = None
        self._last_wr_lines = set()

    def reset_model(self):
        self.mem = {}                # line -> (data, known-bit mask)
        self.pend = {}               # cycle -> (client, line, expected data, mask)
        self.rd_gnt = 0
        self.wr_gnt = 0

    # -- model ---------------------------------------------------------------------------------
    def model_read(self, line):
        return self.mem.get(line, (0, 0))

    def model_write(self, line, data, be):
        m = be_mask(be)
        old, known = self.mem.get(line, (0, 0))
        self.mem[line] = ((old & ~m) | (data & m), known | m)

    # -- signals -------------------------------------------------------------------------------
    def drive(self):
        rv = rl = wv = wl = wd = wb = 0
        for c in range(self.ncl):
            if self.rd[c] is not None:
                rv |= 1 << c
                rl |= self.rd[c] << (c * self.aw)
            if self.wr[c] is not None:
                line, data, be = self.wr[c]
                wv |= 1 << c
                wl |= line << (c * self.aw)
                wd |= data << (c * LINE_BITS)
                wb |= be << (c * LINE_BYTES)
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

    async def step(self):
        d = self.dut
        rv, wv = self.drive()
        await FallingEdge(d.clk)
        # ---- outputs of this cycle ----
        rdy_r = int(d.rd_req_ready.value)
        rdy_w = int(d.wr_ready.value)
        assert rdy_r == self.rd_gnt, "cycle %d: rd_req_ready %#x, expected grant %#x" % (
            self.cyc, rdy_r, self.rd_gnt)
        assert rdy_w == self.wr_gnt, "cycle %d: wr_ready %#x, expected grant %#x" % (
            self.cyc, rdy_w, self.wr_gnt)
        rsp_v = int(d.rd_rsp_valid.value)
        exp = self.pend.pop(self.cyc, None)
        if exp is None:
            assert rsp_v == 0, "cycle %d: spurious rd_rsp_valid %#x" % (self.cyc, rsp_v)
        else:
            c, line, data, known = exp
            assert rsp_v == 1 << c, "cycle %d: rd_rsp_valid %#x, expected client %d (line %d)" % (
                self.cyc, rsp_v, c, line)
            got = int(d.rd_rsp_data.value)
            diff = (got ^ data) & known
            if diff:
                bad = [b for b in range(LINE_BYTES) if (diff >> (8 * b)) & 0xFF]
                raise AssertionError(
                    "cycle %d: client %d line %d: %d byte(s) differ, first byte %d got %#04x "
                    "expected %#04x" % (self.cyc, c, line, len(bad), bad[0],
                                        (got >> (8 * bad[0])) & 0xFF,
                                        (data >> (8 * bad[0])) & 0xFF))
            self.n_rsp += 1
            self.n_rsp_bytes_checked += bin(known).count("1") // 8
        # ---- transfers at the coming edge: reads see the state before this edge's writes ----
        take_r = rv & rdy_r
        take_w = wv & rdy_w
        assert take_r & (take_r - 1) == 0 and take_w & (take_w - 1) == 0
        if (rv & (rv - 1)) or (wv & (wv - 1)):
            self.contention += 1
        taken_rd = [c for c in range(self.ncl) if (take_r >> c) & 1]
        taken_wr = [c for c in range(self.ncl) if (take_w >> c) & 1]
        rd_lines = set()
        for c in taken_rd:
            line = self.rd[c]
            data, known = self.model_read(line)
            due = self.cyc + self.rd_lat
            assert due not in self.pend
            self.pend[due] = (c, line, data, known)
            rd_lines.add(line)
            if line in self._last_wr_lines:
                self.raw_next += 1
            self.n_rd += 1
        wr_lines = set()
        for c in taken_wr:
            line, data, be = self.wr[c]
            if line in rd_lines:
                self.collisions += 1
            self.model_write(line, data, be)
            wr_lines.add(line)
            self.n_wr += 1
        self._last_wr_lines = wr_lines
        # ---- grant model ----
        if rv:
            self.rd_gnt = rv & -rv
        if wv:
            self.wr_gnt = wv & -wv
        for c in taken_rd:
            self.rd[c] = None
        for c in taken_wr:
            self.wr[c] = None
        await RisingEdge(d.clk)
        self.cyc += 1
        return taken_rd, taken_wr

    async def drain(self):
        """Stop issuing, wait until every response has been checked (plus a few idle cycles)."""
        n = 0
        while any(r is not None for r in self.rd) or any(w is not None for w in self.wr):
            await self.step()
            n += 1
            assert n < 1000, "requests not accepted"
        for _ in range(self.rd_lat + 3):
            await self.step()
        assert not self.pend, "responses missing: %s" % sorted(self.pend)

    def idle(self):
        return all(r is None for r in self.rd) and all(w is None for w in self.wr)

    def special_lines(self):
        """First/last line, each bank boundary +-1, middle."""
        s = {0, self.lines - 1, self.lines // 2}
        for b in range(1, self.nbank):
            for dl in (-1, 0, 1):
                s.add(b * self.bank_lines + dl)
        return sorted(x for x in s if 0 <= x < self.lines)


def bench_cfg():
    return json.loads(os.environ["TSR_SPM_CFG"])


def bench_cycles(default: int) -> int:
    return int(os.environ.get("TSR_SPM_CYCLES", str(default)))


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


ALL_BE = (1 << LINE_BYTES) - 1


# =============================================================================================
# cocotb tests
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
    """Exact-cycle scenarios: first-request bubble, write -> read next cycle (new data),
    same-cycle read + write of one line (old data), byte-enable merging, latency of an isolated
    read, client hand-over, back-to-back streams at one transfer per cycle, special lines."""
    sb = await setup(dut)
    rnd = tb_rng(11)
    ncl = sb.ncl
    a, b = 0, ncl - 1                       # two clients (the same one if NCL = 1)
    lines = sb.special_lines()
    L = lines[len(lines) // 2]

    # first request after reset waits one cycle for the grant, then streams
    sb.wr[a] = (L, rnd.getrandbits(LINE_BITS), ALL_BE)
    t_rd, t_wr = await sb.step()
    assert t_wr == [], "write accepted without a grant"
    t_rd, t_wr = await sb.step()
    assert t_wr == [a]
    # read port: warm up the grant of client b
    sb.rd[b] = L
    assert (await sb.step())[0] == []
    assert (await sb.step())[0] == [b]
    # write L, then read L in the next cycle: new data (checked by the model)
    d1 = rnd.getrandbits(LINE_BITS)
    sb.wr[a] = (L, d1, ALL_BE)
    assert (await sb.step())[1] == [a]
    sb.rd[b] = L
    assert (await sb.step())[0] == [b]
    # same-cycle read and write of L: the read returns the old data (d1)
    sb.rd[b] = L
    sb.wr[a] = (L, rnd.getrandbits(LINE_BITS), ALL_BE)
    t_rd, t_wr = await sb.step()
    assert t_rd == [b] and t_wr == [a]
    assert sb.collisions >= 1
    # byte-enable merges: two partial writes back to back, then read
    sb.wr[a] = (L, rnd.getrandbits(LINE_BITS), 0x5555_5555_5555_5555)
    assert (await sb.step())[1] == [a]
    sb.wr[a] = (L, rnd.getrandbits(LINE_BITS), 0x0000_FFFF_0000_0001)
    assert (await sb.step())[1] == [a]
    sb.wr[a] = (L, rnd.getrandbits(LINE_BITS), 0)            # no byte written
    assert (await sb.step())[1] == [a]
    sb.rd[b] = L
    assert (await sb.step())[0] == [b]
    await sb.drain()
    # isolated read: response exactly RD_LAT cycles later, nothing else (step() checks every cycle)
    sb.rd[b] = L
    await sb.step()
    for _ in range(sb.rd_lat + 2):
        await sb.step()
    # hand-over between clients: a's write right after b's write (one-cycle grant bubble)
    if ncl > 1:
        sb.wr[b] = (lines[0], rnd.getrandbits(LINE_BITS), ALL_BE)
        assert (await sb.step())[1] == []                    # grant is still a's
        assert (await sb.step())[1] == [b]
        sb.wr[a] = (lines[-1], rnd.getrandbits(LINE_BITS), ALL_BE)
        assert (await sb.step())[1] == []
        assert (await sb.step())[1] == [a]
        # both request at once: the lower index wins, the other waits (R2 hold)
        sb.wr[a] = (lines[0], rnd.getrandbits(LINE_BITS), ALL_BE)
        sb.wr[b] = (lines[-1], rnd.getrandbits(LINE_BITS), ALL_BE)
        assert (await sb.step())[1] == [a]
        assert (await sb.step())[1] == []                    # grant moves to b next cycle
        assert (await sb.step())[1] == [b]
    # back-to-back streams over the special lines: one write and one read per cycle
    # (warm up a's write grant and b's read grant first)
    sb.wr[a] = (lines[0], rnd.getrandbits(LINE_BITS), ALL_BE)
    sb.rd[b] = lines[0]
    await sb.drain()
    seq = lines * 3
    for i, line in enumerate(seq):
        sb.wr[a] = (line, rnd.getrandbits(LINE_BITS), ALL_BE)
        if i >= 1:
            sb.rd[b] = seq[i - 1]        # read the line written in the previous cycle
        t_rd, t_wr = await sb.step()
        assert t_wr == [a], "write stream stalled at %d" % i
        if i >= 2:
            assert t_rd == [b], "read stream stalled at %d" % i
    await sb.drain()
    assert sb.raw_next >= len(seq) - 2
    dut._log.info("directed: %d reads, %d writes, %d responses checked, %d collisions",
                  sb.n_rd, sb.n_wr, sb.n_rsp, sb.collisions)


@cocotb.test()
async def spm_sweep(dut):
    """Every line written (unique random contents, full lines) and read back, back to back at
    one transfer per cycle; then a second pass with random byte enables and a read-back. Catches
    aliasing between lines and banks. Clients rotate between passes."""
    sb = await setup(dut)
    rnd = tb_rng(12)
    ncl = sb.ncl
    order = list(range(sb.lines))

    async def stream(kind, client, items):
        """Offer items on one client as fast as accepted; returns cycles until the last offer."""
        it = iter(items)
        nxt = next(it, None)
        cycles = 0
        while nxt is not None:
            if kind == "wr" and sb.wr[client] is None:
                sb.wr[client] = nxt
                nxt = next(it, None)
            elif kind == "rd" and sb.rd[client] is None:
                sb.rd[client] = nxt
                nxt = next(it, None)
            await sb.step()
            cycles += 1
        await sb.drain()
        return cycles

    def wr_items(be_fn):
        for line in order:
            yield (line, (rnd.getrandbits(LINE_BITS - 32) << 32) | line, be_fn())

    c_w = ncl - 1
    c_r = 0
    cyc = await stream("wr", c_w, wr_items(lambda: ALL_BE))
    assert cyc <= sb.lines + 1, "write sweep not back to back: %d cycles for %d lines" % (cyc, sb.lines)
    rnd.shuffle(order)
    cyc = await stream("rd", c_r, order)
    assert cyc <= sb.lines + 1, "read sweep not back to back"
    # second pass, other clients, random byte enables, descending order
    order.sort(reverse=True)
    c_w = min(1, ncl - 1)
    c_r = ncl - 1
    await stream("wr", c_w, wr_items(lambda: rnd.getrandbits(LINE_BYTES)))
    await stream("rd", c_r, order)
    assert sb.n_rsp == 2 * sb.lines
    dut._log.info("sweep: %d lines x 2 passes, %d bytes compared", sb.lines, sb.n_rsp_bytes_checked)


async def random_phase(sb, rnd, cycles, p_rd, p_wr, hot):
    """Random traffic on every client: idle clients issue with probability p_rd / p_wr per
    cycle; lines from a hot set (collisions, write->read), special lines or uniform."""
    special = sb.special_lines()

    def pick_line():
        r = rnd.random()
        if r < 0.3:
            return rnd.choice(hot)
        if r < 0.45:
            return rnd.choice(special)
        return rnd.randrange(sb.lines)

    def pick_be():
        r = rnd.random()
        if r < 0.6:
            return ALL_BE
        if r < 0.85:
            return rnd.getrandbits(LINE_BYTES)
        if r < 0.92:
            return 1 << rnd.randrange(LINE_BYTES)
        return 0 if r < 0.95 else ALL_BE ^ (1 << rnd.randrange(LINE_BYTES))

    for _ in range(cycles):
        for c in range(sb.ncl):
            if sb.rd[c] is None and rnd.random() < p_rd[c]:
                sb.rd[c] = pick_line()
            if sb.wr[c] is None and rnd.random() < p_wr[c]:
                sb.wr[c] = (pick_line(), rnd.getrandbits(LINE_BITS), pick_be())
        await sb.step()


@cocotb.test()
async def spm_random(dut):
    """Random reads/writes from all clients (contention, stalls, back-to-back, collisions,
    partial byte enables) over the whole address range, checked every cycle."""
    sb = await setup(dut)
    rnd = tb_rng(13)
    total = bench_cycles(10000)
    ncl = sb.ncl
    hot = sorted({rnd.randrange(sb.lines) for _ in range(4)} | {0, sb.lines - 1})
    # preload a part of the SPM so that early reads compare data too
    for line in sb.special_lines() + hot:
        sb.wr[0] = (line, rnd.getrandbits(LINE_BITS), ALL_BE)
        while sb.wr[0] is not None:
            await sb.step()
    phases = [
        # (fraction, p_rd per client, p_wr per client)
        (0.25, [0.5] * ncl, [0.5] * ncl),                      # mixed, contention
        (0.25, [1.0] + [0.0] * (ncl - 1), [1.0] + [0.0] * (ncl - 1)),   # one client streaming
        (0.20, [0.15] * ncl, [0.15] * ncl),                    # sparse
        (0.15, [1.0] * ncl, [1.0] * ncl),                      # everybody saturating
        (0.15, [rnd.random() for _ in range(ncl)], [rnd.random() for _ in range(ncl)]),
    ]
    for frac, p_rd, p_wr in phases:
        await random_phase(sb, rnd, int(total * frac), p_rd, p_wr, hot)
    await sb.drain()
    dut._log.info("random: %d cycles, %d reads, %d writes, %d responses (%d bytes) checked, "
                  "%d same-edge collisions, %d write->next-cycle reads, %d contention cycles",
                  sb.cyc, sb.n_rd, sb.n_wr, sb.n_rsp, sb.n_rsp_bytes_checked, sb.collisions,
                  sb.raw_next, sb.contention)
    assert sb.n_rsp == sb.n_rd and sb.n_rd > total // 4 and sb.n_wr > total // 4
    if sb.lines > 1:
        assert sb.raw_next > 0
    assert sb.collisions > 0
    if ncl > 1:
        assert sb.contention > 0


@cocotb.test()
async def spm_reset_midflight(dut):
    """Reset while reads are in flight: no response comes out, grants are dropped, and the SPM
    works normally afterwards."""
    sb = await setup(dut)
    rnd = tb_rng(14)
    hot = [0, sb.lines - 1]
    await random_phase(sb, rnd, 200, [0.7] * sb.ncl, [0.7] * sb.ncl, hot)
    # keep the read port busy (3 reads accepted back to back), then reset in the middle of the
    # response pipeline; requests still waiting are withdrawn together with the reset
    accepted = 0
    while accepted < 3:
        if sb.rd[0] is None:
            sb.rd[0] = rnd.randrange(sb.lines)
        t_rd, _ = await sb.step()
        accepted = accepted + 1 if t_rd == [0] else 0
    sb.rd = [None] * sb.ncl
    sb.wr = [None] * sb.ncl
    sb.drive()
    dut.rst_n.value = 0
    # Synchronous reset: it takes effect at the next edge; responses due after it are dropped
    # (with RD_LAT 1 the last response is due in this very cycle, so none is in flight).
    if sb.rd_lat >= 2:
        assert any(due > sb.cyc for due in sb.pend), "no read in flight across the reset edge"
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
    sb.rd[0] = sb.lines          # one past the end
    mark_violation(dut)
    for _ in range(4):
        await sb.step()
        sb.rd[0] = None


@cocotb.test()
async def spm_neg_rd_hold(dut):
    sb = await setup(dut)
    sb.rd[0] = 5
    await sb.step()              # no grant yet: not accepted
    sb.rd[0] = 6                 # R2 violation: payload changed while waiting
    mark_violation(dut)
    for _ in range(4):
        await sb.step()


@cocotb.test()
async def spm_neg_wr_hold(dut):
    sb = await setup(dut)
    sb.wr[0] = (5, 1, ALL_BE)
    await sb.step()              # not accepted
    sb.wr[0] = None              # R2 violation: valid dropped while waiting
    mark_violation(dut)
    for _ in range(4):
        await sb.step()


@cocotb.test()
async def spm_neg_prereset(dut):
    """Clock running with rst_n high and no reset yet (power-up, reset not asserted): every client
    drives contract-violating garbage (lines >= SPM lines, payloads changing and valids dropping
    while not accepted). The checks must stay quiet. Then a reset, checked legal traffic, and a
    write to line 1001 on client 1, which must stop the simulation (checks armed by the reset)."""
    cfg = bench_cfg()
    ncl, aw, lines = cfg["NCL"], cfg["AW"], cfg["LINES"]
    assert ncl >= 2 and lines < (1 << aw), "needs 2 clients and a representable line >= LINES"
    rnd = tb_rng(15)
    init_inputs(dut)
    dut.rst_n.value = 1
    await Timer(1, unit="ns")
    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, unit="ns").start(start_high=False))

    def pack(vals, w):
        return sum(v << (i * w) for i, v in enumerate(vals))

    for cyc in range(48):
        if cyc == 0:                 # every client, both ports: line 2^AW - 1 >= LINES
            rv = wv = (1 << ncl) - 1
            rl = wl = [(1 << aw) - 1] * ncl
        else:                        # random: drops, changes, out-of-range lines
            rv = rnd.getrandbits(ncl)
            wv = rnd.getrandbits(ncl)
            rl = [rnd.randrange(1 << aw) for _ in range(ncl)]
            wl = [rnd.randrange(lines, 1 << aw) if rnd.random() < 0.3 else rnd.randrange(1 << aw)
                  for _ in range(ncl)]
        dut.rd_req_valid.value = rv
        dut.rd_req_line.value = pack(rl, aw)
        dut.wr_valid.value = wv
        dut.wr_line.value = pack(wl, aw)
        dut.wr_data.value = rnd.getrandbits(ncl * LINE_BITS)
        dut.wr_be.value = rnd.getrandbits(ncl * LINE_BYTES)
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
    sb.wr[1] = (lines + 1, 0, ALL_BE)
    mark_violation(dut)
    for _ in range(4):
        await sb.step()
