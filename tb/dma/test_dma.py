"""tsr_dma tests (docs/rtl-core.md §3.2, §4).

pytest functions:
  test_dma_sys[cfg]   cocotb on tb/dma/hdl/tsr_tb_dma.sv (tsr_dma + one tsr_mem_arb per channel,
                      Python SPM model, MemModel on the memory ports, background AUX/WS readers):
                        sys_directed     4 KiB edges, 1..max lines (whole SPM), DRAM limit, all
                                         channels, LD/ST round trips
                        sys_random       random LD/ST (sizes, offsets, channels, back-to-back),
                                         background traffic, random SPM/memory backpressure,
                                         periodic combinational-path probes
                        sys_outstanding  long memory latency: RD_OUTSTANDING reached and never
                                         exceeded (shared with the background sources)
                        sys_throughput   memory/SPM always ready: ~1 beat per cycle
                        sys_wr_modes     ST under the memory's req_first / data_first orders (M5)
                      configurations: u200 (U200: 4 ch, 4 MiB SPM, ADDR_BEAT_W 28, RD_OUT 16),
                      ch3aw36 (3 ch, 256-line SPM, ADDR_BEAT_W 36: DRAM operands crossing 2^32,
                      RD_OUT 3, SPM_RD_LAT 1), tiny (1 ch, 3-line SPM, ADDR_BEAT_W 5: 32-beat
                      channel, RD_OUT 1, SPM_RD_LAT 4), aw33 (2 ch, 128-line SPM, ADDR_BEAT_W 33:
                      2^32 crossings with a small SPM, RD_OUT 64, SPM_RD_LAT 8: 16-entry queue)
  test_dma_unit[q]    cocotb on tsr_dma alone (MemModel on its source ports, SPM model with
                      variable in-order latency), beat queue LDQ_BEATS 512 (default) and 64:
                      dma_unit random ops + combinational-path probes; dma_reset synchronous reset
                      in the middle of LD/ST, reset state of every output, then fully checked
                      ops; dma_reserve LDs into an SPM whose write port stalls for long periods
                      (fast memory): the reservation reaches the queue limit, rd_rsp_ready stays 1
  test_dma_rtl_spm[q] the wrapper with the real even / odd tsr_spm (RTL_SPM = 1; the DMA and the
                      contending clients through tsr_spm_cport), random initial state, beat queue
                      512 and 64 (the minimum: the saturated SPM fills it):
                        sys_rtl_spm          LD/ST round trips (checked through DRAM)
                        sys_rtl_spm_contend  the same with the other SPM clients writing in bursts
                                             (SpmContender: clients 1..3 single port, client 4 on
                                             both banks) and reading (SpmReader), forcing
                                             round-robin waits of the DMA's SPM writes during LDs
                        sys_rtl_spm_saturate the deadlock case of CC §15.1: LD / ST and the weight
                                             streams on every channel while every other SPM client
                                             saturates both write ports and both read ports: every
                                             op completes, and each channel's response path is
                                             never stalled (m_rd_rsp_ready stays 1: the MVM weight
                                             beats never wait behind a DMA beat, CC §10.4)

LD burst reservation (CC §10.4, CC-13; ReserveMonitor in every test): the DMA's rd_rsp_ready is 1
in every cycle, and the LD beats requested and not yet written to the SPM never exceed the beat
queue (LDQ_BEATS) plus the 3 places of its read-out (RAM read stages / skid, output register). Simulator cross-check (spm_sim.SimOps, every test):
each op also runs through tsr_sim_op on a simulator with the same DRAM contents (and, for an ST,
the same SPM source lines); the result code is 0 and the simulator's SPM lines (LD) / DRAM beats
(ST) equal the RTL's.

Every op is checked for: data (SPM lines / DRAM beats), the exact burst list (D1 split, read id
{DMA, 0}, only channel ch), the set of DRAM beats written (== the ST range), the SPM lines read /
written (exactly the range, in order, each once), `done` only after the last SPM write (LD) /
the last write response (ST) with the memory side quiescent for the DMA, and the dispatch
protocol (DmaDriver).
"""
from __future__ import annotations

import json
import random

import cocotb
import pytest
from cocotb.triggers import ClockCycles, FallingEdge, RisingEdge

from common import isa
from common.mem_model import MemModel
from common.runner import REPO, rtl_sources, run
from common.tbutil import seed as tb_seed

import dma_tb
from dma_tb import (NX, BurstMonitor, DmaDriver, PerfMonitor, ReadSources, ReserveMonitor, SpmContender,
                    SpmModel, SpmReader, clock_reset, comb_probe, split)

import sys
sys.path.insert(0, str(REPO / "tb" / "spm"))
from spm_sim import SimOps, mut_sources  # noqa: E402 (tb/spm: the simulator's SPM / op-level API)

TB_SV = REPO / "tb" / "dma" / "hdl" / "tsr_tb_dma.sv"
MIB = 1024 * 1024

CFGS = {
    "u200":    dict(NUM_CH=4, ADDR_BEAT_W=28, SPM_BYTES=4 * MIB, SPM_RD_LAT=2, RD_OUTSTANDING=16),
    "ch3aw36": dict(NUM_CH=3, ADDR_BEAT_W=36, SPM_BYTES=16384, SPM_RD_LAT=1, RD_OUTSTANDING=3),
    "tiny":    dict(NUM_CH=1, ADDR_BEAT_W=5, SPM_BYTES=192, SPM_RD_LAT=4, RD_OUTSTANDING=1),
    "aw33":    dict(NUM_CH=2, ADDR_BEAT_W=33, SPM_BYTES=8192, SPM_RD_LAT=8, RD_OUTSTANDING=64),
}
UNIT_CFGS = {
    "q512": dict(NUM_CH=2, ADDR_BEAT_W=20, SPM_BYTES=65536, SPM_RD_LAT=3),
    "q64": dict(NUM_CH=2, ADDR_BEAT_W=20, SPM_BYTES=65536, SPM_RD_LAT=3, LDQ_BEATS=64),
}
RTL_SPM_CFGS = {
    "q512": dict(NUM_CH=2, ADDR_BEAT_W=20, SPM_BYTES=16384, SPM_RD_LAT=2, RD_OUTSTANDING=8, RTL_SPM=1),
    # the minimum queue: one burst in flight per 64 beats, so the saturated SPM fills it
    "q64": dict(NUM_CH=2, ADDR_BEAT_W=20, SPM_BYTES=16384, SPM_RD_LAT=4, RD_OUTSTANDING=8, RTL_SPM=1,
                LDQ_BEATS=64),
}

SYS_TESTS = ["sys_directed", "sys_random", "sys_outstanding", "sys_throughput", "sys_wr_modes"]

SYS_INPUTS = ["rst_n", "cmd_valid", "cmd", "spm_rd_req_ready", "spm_rd_rsp_valid", "spm_rd_rsp_data",
              "spm_wr_ready", "f_rd_req_valid", "f_rd_req_addr", "f_rd_req_len", "f_rd_req_id",
              "f_rd_rsp_ready", "w_rd_req_valid", "w_rd_req_addr", "w_rd_req_len", "w_rd_req_id",
              "w_rd_rsp_ready", "m_rd_req_ready", "m_rd_rsp_valid", "m_rd_rsp_data", "m_rd_rsp_id",
              "m_rd_rsp_last", "m_wr_req_ready", "m_wr_dat_ready", "m_wr_rsp_valid", "m_wr_rsp_id"]
SYS_OUTPUTS = ["cmd_ready", "done", "spm_rd_req_valid", "spm_rd_req_line", "spm_wr_valid", "spm_wr_line",
               "spm_wr_data", "spm_wr_be", "f_rd_req_ready", "f_rd_rsp_valid", "f_rd_rsp_data",
               "f_rd_rsp_last", "f_rd_rsp_id", "w_rd_req_ready", "w_rd_rsp_valid", "w_rd_rsp_data",
               "w_rd_rsp_last", "w_rd_rsp_id", "m_rd_req_valid", "m_rd_req_addr", "m_rd_req_len",
               "m_rd_req_id", "m_rd_rsp_ready", "m_wr_req_valid", "m_wr_req_addr", "m_wr_req_len",
               "m_wr_req_id", "m_wr_dat_valid", "m_wr_dat_data", "m_wr_dat_last", "m_wr_rsp_ready",
               "perf_rd_beat", "perf_rd_wait"]
X_INPUTS = ["x_wr_valid", "x_wr_line", "x_wr_data", "y_rd_req_valid", "y_rd_req_line"]   # RTL_SPM = 1
# The contending lanes 0..2 are tsr_spm_cport client sides: their ready depends on their own request
# line by design (CC §10.2, the adapter's one LUT inside the client), so x_wr_ready / y_rd_req_ready
# are not probed; every other output is probed against the lanes' inputs.
X_OUTPUTS = []
DMA_INPUTS = ["rst_n", "cmd_valid", "cmd", "rd_req_ready", "rd_rsp_valid", "rd_rsp_data", "rd_rsp_last",
              "rd_rsp_id", "wr_req_ready", "wr_dat_ready", "wr_rsp_valid", "wr_rsp_id", "spm_rd_req_ready",
              "spm_rd_rsp_valid", "spm_rd_rsp_data", "spm_wr_ready"]
DMA_OUTPUTS = ["cmd_ready", "done", "rd_req_valid", "rd_req_addr", "rd_req_len", "rd_req_id", "rd_rsp_ready",
               "wr_req_valid", "wr_req_addr", "wr_req_len", "wr_req_id", "wr_dat_valid", "wr_dat_data",
               "wr_dat_last", "wr_rsp_ready", "spm_rd_req_valid", "spm_rd_req_line", "spm_wr_valid",
               "spm_wr_line", "spm_wr_data", "spm_wr_be"]


# =============================================================================================
# pytest side
# =============================================================================================
@pytest.mark.parametrize("cfg", list(CFGS))
def test_dma_sys(cfg):
    p = CFGS[cfg]
    run("tsr_tb_dma", __name__, parameters=p, sources=mut_sources(rtl_sources()), extra_sources=[TB_SV],
        testcase=SYS_TESTS,
        extra_env={"TSR_DMA_CFG": json.dumps(dict(p, name=cfg))})


@pytest.mark.parametrize("q", list(UNIT_CFGS))
def test_dma_unit(q):
    p = UNIT_CFGS[q]
    run("tsr_dma", __name__, parameters=p, sources=mut_sources(rtl_sources()),
        testcase=["dma_unit", "dma_reset", "dma_reserve"],
        name="tsr_dma_" + q, extra_env={"TSR_DMA_CFG": json.dumps(dict(p, name="unit_" + q))})


@pytest.mark.parametrize("q", list(RTL_SPM_CFGS))
def test_dma_rtl_spm(q):
    # Random initial state like every other build (tsr_spm arms its checks at the first reset).
    p = RTL_SPM_CFGS[q]
    run("tsr_tb_dma", __name__, parameters=p, sources=mut_sources(rtl_sources()), extra_sources=[TB_SV],
        name="tsr_tb_dma_rtl_" + q,
        testcase=["sys_rtl_spm", "sys_rtl_spm_contend", "sys_rtl_spm_saturate"],
        extra_env={"TSR_DMA_CFG": json.dumps(dict(p, name="rtl_spm_" + q))})


# =============================================================================================
# Environment
# =============================================================================================
class Env:
    """DMA + memory model + SPM model (+ background readers for the wrapper)."""

    def __init__(self, dut, cfg, rnd, *, system: bool = True, rtl_spm: bool = False, bg: bool = True,
                 mem_kw=None, spm_kw=None, contend: bool = False):
        self.dut, self.cfg, self.rnd, self.system = dut, cfg, rnd, system
        self.n = cfg["NUM_CH"]
        self.aw = cfg["ADDR_BEAT_W"]
        self.limit = 1 << self.aw
        self.lines = cfg["SPM_BYTES"] // 64
        self.rdo = cfg.get("RD_OUTSTANDING", 1 << 30)
        # DRAM layout per channel: [0, bg_hi) read-only background region (never written),
        # [bg_hi, limit) LD/ST region. Random ops stay in a window of `win` beats unless they
        # target the end of the address space.
        self.bg_hi = min(4096, self.limit // 2)
        self.win = min(self.limit, 1 << 17)
        mem_kw = dict(mem_kw or {})
        if system:
            self.mem = MemModel(dut, num_ch=self.n, prefix="m_", addr_w=self.aw, rd_outstanding=self.rdo,
                                limit_beats=self.limit, seed=rnd.getrandbits(31), **mem_kw)
            self.dma_id = isa.SRC_DMA << isa.SRC_ID_W
        else:
            self.mem = MemModel(dut, num_ch=self.n, prefix="", addr_w=self.aw, id_w=isa.SRC_ID_W,
                                rd_outstanding=1 << 30, limit_beats=self.limit, seed=rnd.getrandbits(31),
                                **mem_kw)
            self.dma_id = 0
        # the simulator executing the same ops (same DRAM contents; ST sources synced)
        self.sim = SimOps(num_ch=self.n, spm_bytes=cfg["SPM_BYTES"], ch_bytes=64 << self.aw,
                          addr_beat_w=self.aw, rd_outstanding=min(self.rdo, 1 << 16))
        self.sim_ops = 0
        for c in range(self.n):
            init = rnd.getrandbits(512 * self.bg_hi).to_bytes(64 * self.bg_hi, "little")
            self.mem.load(c, 0, init)
            self.sim.dram_write(c, 0, init)
        self.touched = [set(range(self.bg_hi)) for _ in range(self.n)]
        self.spm = None if rtl_spm else SpmModel(dut, lines=self.lines, rd_lat=cfg["SPM_RD_LAT"], rnd=rnd,
                                                 **(spm_kw or {}))
        self.spm_shadow = {}            # line -> content written by an LD of this test
        # real SPM: second SPM writer (only with contend) and the LD rd_rsp_ready check
        self.contend = contend
        self.xw = SpmContender(dut, lines=self.lines, rnd=rnd, content=self.spm_shadow.get) if rtl_spm else None
        self.yr = SpmReader(dut, lines=self.lines, rnd=rnd) if rtl_spm else None
        self.qd = cfg.get("LDQ_BEATS", 512)
        self.ldmon = ReserveMonitor(dut, dut.u_dma if system else dut, self.n, self.qd, contend=rtl_spm)
        self.bm = BurstMonitor(dut, prefix="m_" if system else "", num_ch=self.n, aw=self.aw,
                               id_w=isa.MEM_ID_W if system else isa.SRC_ID_W)
        self.drv = DmaDriver(dut, rnd)
        self.perf = PerfMonitor(dut, self.n) if system else None
        self.bgf = self.bgw = None
        if system:
            mb = self.mem.beat
            self.bgf = ReadSources(dut, prefix="f_", lanes=1, aw=self.aw, rnd=rnd, expect=lambda k, b: mb(0, b),
                                   region=(0, self.bg_hi), shared_rsp=True, name="aux")
            self.bgw = ReadSources(dut, prefix="w_", lanes=self.n, aw=self.aw, rnd=rnd,
                                   expect=lambda k, b: mb(k, b), region=(0, self.bg_hi), shared_rsp=False,
                                   name="ws")
            self.bgf.auto = self.bgw.auto = bg
        self.ops = 0
        self.beats = 0
        self.cross32 = 0

    def start(self):
        self.mem.start()
        if self.spm is not None:
            self.spm.start()
        if self.xw is not None:
            self.xw.start()
            self.yr.start()
        if self.bgf is not None:
            self.bgf.start()
            self.bgw.start()
        return self

    def stop(self):
        for x in (self.mem, self.spm, self.bgf, self.bgw, self.bm, self.drv, self.perf, self.xw, self.yr, self.ldmon):
            if x is not None:
                x.stop()
        self.sim.close()

    async def quiesce(self, cycles: int = 10):
        if self.bgf is not None:
            await self.bgf.drain()
            await self.bgw.drain()
        await ClockCycles(self.dut.clk, cycles)
        self.mem.check(idle=True)

    # ------------------------------------------------------------------------------------------
    def legal(self, is_st, ch, spm, dram, n) -> bool:
        return (n >= 1 and 0 <= ch < self.n and spm >= 0 and dram >= 0
                and spm + n <= self.lines and dram + n <= self.limit
                and dram < (1 << 32) and (not is_st or dram >= self.bg_hi))

    def _fill(self, ch, dram, n):
        """Give never-touched beats of an LD source random content (host write while idle)."""
        t = self.touched[ch]
        for b in range(dram, dram + n):
            if b not in t:
                data = self.rnd.getrandbits(512).to_bytes(64, "little")
                self.mem.load(ch, 64 * b, data)
                self.sim.dram_write(ch, 64 * b, data)
                t.add(b)

    async def op(self, is_st, ch, spm, dram, n, gap=0):
        assert self.legal(is_st, ch, spm, dram, n), (is_st, ch, spm, dram, n)
        mem, c = self.mem, self.mem.ch[ch]
        if is_st:
            if self.spm is not None:
                exp = [self.spm.line(spm + i) for i in range(n)]
            else:
                exp = [self.spm_shadow[spm + i] for i in range(n)]
        else:
            self._fill(ch, dram, n)
            exp = [mem.beat(ch, dram + i) for i in range(n)]
        if self.spm is not None:
            self.spm.clear_log()
        mem.clear_tracking()
        log0 = len(self.bm.log)
        dma_id = self.dma_id

        def at_done():
            if is_st:
                assert not c.wreqs and not c.wdata and not c.wrsp and not c.wrsp_valid, \
                    "ST done while write traffic is still in flight on ch%d" % ch
            else:
                assert not any(b.id == dma_id for b in c.rd_bursts), "LD done with DMA reads in flight"
                if self.spm is not None:
                    assert len(self.spm.writes) == n, "LD done after %d of %d SPM writes" % (
                        len(self.spm.writes), n)

        if self.xw is not None:
            # the contender never writes the lines an LD is writing (it rewrites current contents)
            await self.xw.window(range(0) if is_st else range(spm, spm + n), enabled=self.contend)
            self.yr.enabled = self.contend
        self.ldmon.ld_active = not is_st
        await self.drv.op(is_st, ch, spm, dram, n, gap=gap, on_done=at_done)
        self.ldmon.ld_active = False
        # the simulator: the same op on the same DRAM (and, for an ST, the same SPM source lines)
        if is_st:
            self.sim.spm_write(64 * spm, b"".join(x.to_bytes(64, "little") for x in exp))
        enc = isa.enc_st if is_st else isa.enc_ld
        err = self.sim.op(enc(ch, spm, dram, n))
        assert err == 0, "simulator refused %s ch%d spm %d dram 0x%x n %d: error %d" % (
            "ST" if is_st else "LD", ch, spm, dram, n, err)
        self.sim_ops += 1
        what = "%s ch%d spm %d dram 0x%x n %d" % ("ST" if is_st else "LD", ch, spm, dram, n)
        # data (the RTL against the expectation and against the simulator)
        if is_st:
            sim_d = self.sim.dram_read(ch, 64 * dram, 64 * n)
            for i in range(n):
                got = mem.beat(ch, dram + i)
                assert got == exp[i], "%s: DRAM beat 0x%x differs from SPM line %d" % (what, dram + i, spm + i)
                assert got == int.from_bytes(sim_d[64 * i:64 * i + 64], "little"), \
                    "%s: DRAM beat 0x%x differs from the simulator's ST" % (what, dram + i)
            self.touched[ch].update(range(dram, dram + n))
        else:
            sim_s = self.sim.spm_read(64 * spm, 64 * n)
            for i in range(n):
                assert int.from_bytes(sim_s[64 * i:64 * i + 64], "little") == exp[i], \
                    "%s: SPM line %d differs from the simulator's LD" % (what, spm + i)
                self.spm_shadow[spm + i] = exp[i]
                if self.spm is not None:
                    assert self.spm.mem.get(spm + i) == exp[i], "%s: SPM line %d differs from DRAM beat 0x%x" % (
                        what, spm + i, dram + i)
        # SPM accesses: exactly the range, in order, each line once
        if self.spm is not None:
            rng = list(range(spm, spm + n))
            assert self.spm.reads == (rng if is_st else []), "%s: SPM reads %s" % (what, self.spm.reads[:8])
            assert self.spm.writes == ([] if is_st else rng), "%s: SPM writes %s" % (what, self.spm.writes[:8])
        # DRAM bursts of the DMA: the D1 split, read id {DMA, 0}, only channel ch
        mine = [e for e in self.bm.log[log0:] if e[5] == dma_id or (e[1] == "wr")]
        kind = "wr" if is_st else "rd"
        assert all(e[1] == kind and e[2] == ch and e[5] == dma_id for e in mine), "%s: unexpected bursts %s" % (
            what, [e for e in mine if not (e[1] == kind and e[2] == ch and e[5] == dma_id)][:4])
        got = [(e[3], e[4]) for e in mine]
        assert got == split(dram, n), "%s: bursts %s, expected %s" % (what, got[:6], split(dram, n)[:6])
        # DRAM beats written == the ST range (nothing else, no other channel)
        for k in range(self.n):
            want = set(range(dram, dram + n)) if (is_st and k == ch) else set()
            assert mem.written[k] == want, "%s: ch%d written set differs (%d beats vs %d)" % (
                what, k, len(mem.written[k]), len(want))
        self.ops += 1
        self.beats += n
        if dram < (1 << 32) < dram + n:
            self.cross32 += 1

    # ------------------------------------------------------------------------------------------
    def pick_n(self, nmax):
        c = [1, 2, 3, 63, 64, 65, 127, 128, 129, self.rnd.randint(1, 256), self.rnd.randint(1, nmax), nmax]
        return self.rnd.choice([x for x in c if 1 <= x <= nmax])

    def pick_dram(self, is_st, n):
        lo = self.bg_hi if is_st else 0
        dmax = min(self.limit - n, (1 << 32) - 1)
        r = self.rnd.random()
        if r < 0.12:
            return dmax                                   # ends exactly at the limit (or 2^32 - 1)
        if r < 0.2 and self.limit > (1 << 32) and n > 1:
            return (1 << 32) - self.rnd.randint(1, n - 1)  # crosses 2^32 (64-bit address arithmetic)
        hi = max(lo, min(dmax, lo + self.win - n))
        base = self.rnd.randint(lo, hi)
        off = self.rnd.choice([0, 1, 2, 61, 62, 63, self.rnd.randrange(64)])
        return min(max(base - base % 64 + off, lo), dmax)

    def rand_op(self, nmax_cap):
        while True:
            is_st = int(self.rnd.random() < 0.5)
            ch = self.rnd.randrange(self.n)
            nmax = min(self.lines, nmax_cap, self.limit - (self.bg_hi if is_st else 0))
            n = self.pick_n(nmax)
            spm = self.rnd.choice([0, self.lines - n, self.rnd.randint(0, self.lines - n)])
            dram = self.pick_dram(is_st, n)
            if self.legal(is_st, ch, spm, dram, n):
                return is_st, ch, spm, dram, n


def _cfg_rnd(salt):
    cfg = dma_tb.cfg_from_env()
    return cfg, random.Random(tb_seed() * 7919 + salt)


PROBES = {"probes": 0, "pokes": 0}


async def _prober(dut, clock, rnd, inputs, outputs, every=(40, 400)):
    """Combinational-path probe at random times (mid-transfer states included)."""
    while True:
        await ClockCycles(dut.clk, rnd.randint(*every))
        PROBES["pokes"] += await comb_probe(dut, clock, inputs, outputs, rnd)
        PROBES["probes"] += 1


# =============================================================================================
# cocotb tests (wrapper)
# =============================================================================================
@cocotb.test()
async def sys_directed(dut):
    cfg, rnd = _cfg_rnd(1)
    await clock_reset(dut)
    env = Env(dut, cfg, rnd, bg=False).start()
    L, lim, b0 = env.lines, env.limit, env.bg_hi
    for ch in range(env.n):
        a = b0 + 64 * 3 if lim >= b0 + 64 * 8 else b0
        ops = [
            (0, ch, 0, a + 63, 1),            # last beat of a block
            (0, ch, 1, a + 63, 2),            # 1 + 1 across a 4 KiB boundary
            (0, ch, 3, a + 64, 64),           # one full aligned burst
            (0, ch, 4, a + 65, 64),           # 63 + 1
            (0, ch, 0, a + 60, 130),          # 4 + 64 + 62
            (0, ch, 0, a, 63),
            (0, ch, L - 5, a + 1, 5),         # ends at the last SPM line
            (0, ch, 0, lim - 3, 3),           # ends at the DRAM limit
            (0, ch, 0, lim - 70, 70),         # ... across a block
            (1, ch, 0, a + 1024 + 63, 1),
            (1, ch, 1, a + 1024 + 63, 2),
            (1, ch, 3, a + 1024 + 128, 64),
            (1, ch, 4, a + 1024 + 129, 64),
            (1, ch, 0, a + 1024 + 60, 130),
            (1, ch, L - 5, a + 1024 + 7, 5),
            (1, ch, 0, lim - 5, 5),
            (1, ch, 0, lim - 70, 70),
            (0, ch, L - 70, lim - 70, 70),    # read back what the ST wrote
        ]
        for op in ops:
            if env.legal(*op):
                await env.op(*op)
        # tiny configurations: every legal (spm, dram, n) combination near the edges
        if lim <= 64:
            for is_st in (0, 1):
                for n in range(1, min(L, 3) + 1):
                    for dram in (b0, lim - n, lim - n - 1):
                        for spm in (0, L - n):
                            if env.legal(is_st, ch, spm, dram, n):
                                await env.op(is_st, ch, spm, dram, n)
    # DRAM operands crossing 2^32 (ADDR_BEAT_W > 32: 32-bit start, 64-bit range check). Sized from
    # the SPM (legal for any SPM of >= 2 lines): ST 3 + 64 + 3 beats (fewer with a small SPM),
    # LD of 1 + 1, LD of the ST's beats into other lines, ST from 2^32 - 1.
    if env.limit > (1 << 32) and L >= 2:
        na = min(70, L)
        da = (1 << 32) - min(3, na - 1)
        xops = [(1, 0, 0, da, na), (0, env.n - 1, 0, (1 << 32) - 1, 2),
                (0, 0, min(100, L - na), da, na), (1, 0, 0, (1 << 32) - 1, min(L, 200))]
        c0 = env.cross32
        for op in xops:
            await env.op(*op)
        assert env.cross32 - c0 == len(xops), "only %d of %d ops crossed 2^32" % (env.cross32 - c0, len(xops))
    # the whole SPM in one LD and one ST (1..max lines)
    ch = env.n - 1
    dram = min(env.limit - L, (1 << 32) - 1 - L)
    if env.legal(0, ch, 0, dram, L) and env.legal(1, 0, 0, max(b0, env.limit - L), L):
        await env.op(0, ch, 0, dram, L)
        await env.op(1, 0, 0, max(b0, env.limit - L), L)
    await env.quiesce()
    dut._log.info("sys_directed: %d ops, %d lines, %d DMA cycles, %d ops crossing 2^32", env.ops, env.beats,
                  env.drv.cycles, env.cross32)
    env.stop()


@cocotb.test()
async def sys_random(dut):
    cfg, rnd = _cfg_rnd(2)
    clock = await clock_reset(dut)
    env = Env(dut, cfg, rnd, bg=True, spm_kw=dict(p_rd_ready=0.8, p_wr_ready=0.8)).start()
    probe = cocotb.start_soon(_prober(dut, clock, rnd, SYS_INPUTS, SYS_OUTPUTS))
    nops = {"u200": 70, "ch3aw36": 120, "tiny": 120}.get(cfg["name"], 80)
    for k in range(nops):
        op = env.rand_op(nmax_cap=2048)
        gap = rnd.choice([0, 0, 0, 1, 2, rnd.randint(0, 20)])
        if k % 10 == 9:                          # toggle background traffic now and then
            env.bgf.auto = env.bgw.auto = not env.bgw.auto
        await env.op(*op, gap=gap)
    probe.cancel()
    await env.quiesce()
    await ClockCycles(dut.clk, 3)
    st = env.mem.stats()
    for c in range(env.n):
        assert env.perf.beats[c] == st[c]["rd_beats"], "ch%d: perf_rd_beat %d != memory read beats %d" % (
            c, env.perf.beats[c], st[c]["rd_beats"])
        assert st[c]["max_outstanding"] <= env.rdo
    bg_beats = env.bgf.beats[0] + sum(env.bgw.beats)
    dut._log.info("sys_random: %d ops (%d crossing 2^32), %d lines, %d DMA cycles, %d background beats, %s, "
                  "mem stats %s", env.ops, env.cross32, env.beats, env.drv.cycles, bg_beats, PROBES, st)
    assert PROBES["probes"] > 0
    env.stop()


@cocotb.test()
async def sys_outstanding(dut):
    """Long latency: the DMA alone fills RD_OUTSTANDING; then DMA + background sources share it."""
    cfg, rnd = _cfg_rnd(3)
    await clock_reset(dut)
    env = Env(dut, cfg, rnd, bg=False,
              mem_kw=dict(rd_lat=(150, 200), p_rd_req_ready=1.0, p_rd_rsp_valid=1.0, p_stall=0.0),
              spm_kw=dict(p_rd_ready=1.0, p_wr_ready=1.0, p_stall=0.0)).start()
    for ch in range(env.n):
        n = min(env.lines, 64 * (env.rdo + 3), env.limit - env.bg_hi)
        dram = env.bg_hi
        await env.op(0, ch, 0, dram, n)
        nb = len(split(dram, n))
        mo = env.mem.stats()[ch]["max_outstanding"]
        # LD burst reservation (CC §10.4): the DMA alone keeps at most LDQ_BEATS / 64 bursts of
        # 64 beats in flight (dram is 64-aligned), below RD_OUTSTANDING in the u200 configuration
        want = min(env.rdo, nb, max(1, env.qd // 64))
        assert mo == want, "ch%d: max outstanding %d, expected %d (%d bursts, queue %d)" % (
            ch, mo, want, nb, env.qd)
    w0 = list(env.perf.wait)
    assert all(w > 0 for w in w0), "perf_rd_wait never counted: %s" % w0
    # shared budget: background readers on every channel while the DMA streams
    env.bgf.auto = env.bgw.auto = True
    env.bgf.p_req = env.bgw.p_req = 1.0
    n2 = min(env.lines, 256, env.limit - env.bg_hi)
    for _ in range(2 * env.n):
        ch = rnd.randrange(env.n)
        await env.op(0, ch, 0, env.bg_hi + rnd.randint(0, min(63, env.limit - env.bg_hi - n2)), n2)
    await env.quiesce()
    await ClockCycles(dut.clk, 3)
    st = env.mem.stats()
    for c in range(env.n):
        assert st[c]["max_outstanding"] == env.rdo
        assert env.perf.beats[c] == st[c]["rd_beats"]
    # idle port: no wait cycles counted
    w1 = list(env.perf.wait)
    await ClockCycles(dut.clk, 50)
    assert env.perf.wait == w1, "perf_rd_wait counted while the port is idle"
    dut._log.info("sys_outstanding: stats %s, wait cycles %s", st, env.perf.wait)
    env.stop()


@cocotb.test()
async def sys_throughput(dut):
    """Memory and SPM always ready, minimal latency: LD/ST stream ~1 beat per cycle."""
    cfg, rnd = _cfg_rnd(4)
    await clock_reset(dut)
    env = Env(dut, cfg, rnd, bg=False,
              mem_kw=dict(rd_lat=(1, 1), wr_lat=(1, 1), p_rd_req_ready=1.0, p_rd_rsp_valid=1.0,
                          p_wr_req_ready=1.0, p_wr_dat_ready=1.0, p_wr_rsp_valid=1.0, p_stall=0.0,
                          wr_order="any"),
              spm_kw=dict(p_rd_ready=1.0, p_wr_ready=1.0, p_stall=0.0)).start()
    n = min(env.lines, 1024, env.limit - env.bg_hi)
    res = {}
    for is_st in (0, 1):
        c0 = env.drv.cycles
        await env.op(is_st, 0, 0, env.bg_hi, n)
        res[is_st] = env.drv.cycles - c0
    dut._log.info("sys_throughput: %d lines: LD %d cycles, ST %d cycles", n, res[0], res[1])
    if n >= 256:
        for is_st in (0, 1):
            assert res[is_st] <= n + 40, "%s of %d lines took %d cycles" % ("ST" if is_st else "LD", n, res[is_st])
    await env.quiesce()
    env.stop()


@cocotb.test()
async def sys_wr_modes(dut):
    """ST under each memory write-ordering mode (rule M5: request and data never wait for each
    other's handshake), each followed by an LD read-back of the same beats."""
    cfg, rnd = _cfg_rnd(5)
    await clock_reset(dut)
    for mode in ("req_first", "data_first", "any"):
        env = Env(dut, cfg, rnd, bg=True, mem_kw=dict(wr_order=mode)).start()
        done = 0
        while done < 12:
            _, ch, spm, dram, n = env.rand_op(nmax_cap=300)
            if not env.legal(1, ch, spm, dram, n):
                continue
            await env.op(1, ch, spm, dram, n)
            await env.op(0, ch, rnd.randint(0, env.lines - n), dram, n)
            done += 1
        await env.quiesce()
        dut._log.info("sys_wr_modes %s: %d ops, stats %s", mode, env.ops, env.mem.stats())
        env.stop()
        await ClockCycles(dut.clk, 5)


@cocotb.test()
async def sys_rtl_spm(dut):
    """Real tsr_spm: LD the whole SPM, then random ST/LD round trips checked through DRAM."""
    cfg, rnd = _cfg_rnd(6)
    dut.x_wr_valid.value = 0
    clock = await clock_reset(dut)
    env = Env(dut, cfg, rnd, rtl_spm=True, bg=True).start()
    probe = cocotb.start_soon(_prober(dut, clock, rnd, SYS_INPUTS + X_INPUTS, SYS_OUTPUTS + X_OUTPUTS))
    await env.op(0, 0, 0, env.bg_hi + 5, env.lines)
    for _ in range(60):
        await env.op(*env.rand_op(nmax_cap=env.lines), gap=rnd.choice([0, 0, 3]))
    probe.cancel()
    await env.quiesce()
    dut._log.info("sys_rtl_spm: %d ops, %d lines, %d DMA cycles, %d LD cycles with rd_rsp_ready checked, %s",
                  env.ops, env.beats, env.drv.cycles, env.ldmon.cycles, PROBES)
    assert env.ldmon.cycles > 0
    env.stop()


@cocotb.test()
async def sys_rtl_spm_contend(dut):
    """Real tsr_spm with the other SPM clients writing in bursts (clients 1..3 through their cports,
    client 4 on both banks) and reading at random during the ops: the DMA's SPM writes wait for
    their round-robin turn. The LD burst reservation keeps rd_rsp_ready at 1 whatever the SPM does
    (ReserveMonitor). Data checked through DRAM and against the simulator as above."""
    cfg, rnd = _cfg_rnd(9)
    dut.x_wr_valid.value = 0
    clock = await clock_reset(dut)
    env = Env(dut, cfg, rnd, rtl_spm=True, bg=True, contend=True).start()
    probe = cocotb.start_soon(_prober(dut, clock, rnd, SYS_INPUTS + X_INPUTS, SYS_OUTPUTS + X_OUTPUTS))
    await env.op(0, 0, 0, env.bg_hi + 5, env.lines)          # every line known: the contender can start
    for k in range(80):
        if k == 40:                                            # second half: memory always ready
            env.mem.p_rd_rsp_valid = 1.0
        await env.op(*env.rand_op(nmax_cap=max(1, env.lines // 2)), gap=rnd.choice([0, 0, 3]))
    probe.cancel()
    await env.xw.window(range(0), enabled=False)
    await env.quiesce()
    m = env.ldmon
    dut._log.info("sys_rtl_spm_contend: %d ops, %d lines, %d LD cycles checked, %d DMA SPM write stalls, "
                  "%d contender writes during LDs (%d in total), beat queue max %d", env.ops, env.beats, m.cycles,
                  m.dma_wr_stalls, m.x_writes, env.xw.fires, m.q_max)
    assert m.x_writes > 0 and m.dma_wr_stalls > 0, "no SPM write-port contention during LDs"
    env.stop()


class HolMonitor:
    """Head-of-line check of CC §10.4 on the memory side: with every read source of a channel ready
    (the DMA by its reservation, the WS and AUX background readers forced ready), the arbiter's
    response path never stalls, i.e. m_rd_rsp_ready stays 1 in every cycle after the reset. Counts
    the response beats delivered while an LD of the channel was running."""

    def __init__(self, dut, n):
        self.dut, self.full = dut, (1 << n) - 1
        self.on = False
        self.ld_ch = None
        self.beats_during_ld = 0
        self.cycles = 0
        self._task = cocotb.start_soon(self._run())

    def stop(self):
        self._task.cancel()

    async def _run(self):
        d = self.dut
        while True:
            await FallingEdge(d.clk)
            if not self.on or int(d.rst_n.value) == 0:
                continue
            self.cycles += 1
            r = int(d.m_rd_rsp_ready.value)
            assert r == self.full, "memory response path stalled (m_rd_rsp_ready = 0x%x): a weight beat " \
                "would wait behind a DMA beat" % r
            if self.ld_ch is not None and (int(d.m_rd_rsp_valid.value) >> self.ld_ch) & 1:
                self.beats_during_ld += 1


@cocotb.test()
async def sys_rtl_spm_saturate(dut):
    """The deadlock case of CC §15.1: LD and ST on every channel, the weight-stream (and AUX)
    readers on the same channels always requesting and always ready, while every other SPM client
    saturates both bank write ports and both bank read ports (SpmContender / SpmReader with every
    lane valid in every cycle; their writes rewrite current contents). Every op completes within
    the driver's bound; the DMA's rd_rsp_ready stays 1 and its claim never exceeds the queue
    (ReserveMonitor); every channel's memory response path never stalls (HolMonitor: the weight
    beats never wait behind a DMA beat); the DMA's SPM writes do wait (round robin under load) and
    tsr_spm's own round-robin bound holds (its simulation assertion). Data against DRAM and the
    simulator."""
    cfg, rnd = _cfg_rnd(10)
    dut.x_wr_valid.value = 0
    dut.y_rd_req_valid.value = 0
    await clock_reset(dut)
    # fast memory (short latency, no stalls): it outruns the saturated SPM
    env = Env(dut, cfg, rnd, rtl_spm=True, bg=True, contend=True,
              mem_kw=dict(rd_lat=(4, 16), p_rd_req_ready=1.0, p_rd_rsp_valid=1.0, p_stall=0.0)).start()
    await env.op(0, 0, 0, env.bg_hi + 5, env.lines)          # every line known
    env.xw.saturate = env.yr.saturate = True
    # weight / AUX streams always ready and sparse enough that the DMA gets most of each channel:
    # memory then delivers faster than the saturated SPM takes the DMA's writes, so its beat queue
    # backs up (with the old flow control rd_rsp_ready would drop and stall the weight beats)
    env.bgf.p_rsp_ready = env.bgw.p_rsp_ready = 1.0
    env.bgf.p_req = env.bgw.p_req = 0.01
    hol = HolMonitor(dut, env.n)
    await ClockCycles(dut.clk, 40)                            # background responses settle
    hol.on = True
    m = env.ldmon
    w0, x0 = m.dma_wr_stalls, m.x_writes
    for k in range(40):
        is_st, ch, spm, dram, n = env.rand_op(nmax_cap=max(1, env.lines // 2))
        hol.ld_ch = None if is_st else ch
        await env.op(is_st, ch, spm, dram, n, gap=rnd.choice([0, 0, 2]))
    hol.ld_ch = None
    hol.on = False
    await env.xw.window(range(0), enabled=False)
    env.yr.enabled = False
    await env.quiesce()
    dut._log.info("sys_rtl_spm_saturate: %d ops, %d lines, %d DMA cycles, DMA SPM write stalls %d, "
                  "contending writes during LDs %d (lanes %s), contending reads %d, response beats "
                  "during LDs %d, %d cycles head-of-line checked, claim max %d, queue max %d, %d sim ops",
                  env.ops, env.beats, env.drv.cycles, m.dma_wr_stalls - w0, m.x_writes - x0,
                  env.xw.lane_fires, env.yr.fires, hol.beats_during_ld, hol.cycles, m.claim_max, m.q_max,
                  env.sim_ops)
    assert m.dma_wr_stalls - w0 > 0 and m.x_writes - x0 > 0 and env.yr.fires > 0
    assert all(f > 0 for f in env.xw.lane_fires), "a contending lane never wrote: %s" % env.xw.lane_fires
    assert hol.beats_during_ld > 0 and hol.cycles > 0
    if env.qd <= 128:
        # the queue is smaller than the LDs: it backs up (a 64-beat burst is admitted only into a
        # queue with room for all of it, so the peak is the queue minus what drains meanwhile)
        assert m.q_max >= env.qd // 2, "beat queue never backed up (max %d of %d)" % (m.q_max, env.qd)
    hol.stop()
    env.stop()


# =============================================================================================
# cocotb test (tsr_dma alone)
# =============================================================================================
@cocotb.test()
async def dma_unit(dut):
    cfg, rnd = _cfg_rnd(7)
    clock = await clock_reset(dut)
    env = Env(dut, cfg, rnd, system=False, spm_kw=dict(var_lat=3, p_rd_ready=0.7, p_wr_ready=0.7)).start()
    probe = cocotb.start_soon(_prober(dut, clock, rnd, DMA_INPUTS, DMA_OUTPUTS, every=(20, 200)))
    for _ in range(80):
        await env.op(*env.rand_op(nmax_cap=700), gap=rnd.choice([0, 0, 1, 5]))
    probe.cancel()
    await env.quiesce()
    dut._log.info("dma_unit: %d ops, %d lines, %d DMA cycles, %s", env.ops, env.beats, env.drv.cycles, PROBES)
    assert PROBES["probes"] > 0
    env.stop()


@cocotb.test()
async def dma_reserve(dut):
    """LD burst reservation at the unit (CC §10.4): fast memory (always ready, short latency) and an
    SPM write port that stalls for long periods: the DMA keeps requesting bursts until the queue is
    claimed, then waits; the claim (beats requested, not yet written) reaches the queue size minus
    one burst and never exceeds it plus the read-out; rd_rsp_ready stays 1 (ReserveMonitor);
    every LD's data checked."""
    cfg, rnd = _cfg_rnd(11)
    await clock_reset(dut)
    env = Env(dut, cfg, rnd, system=False,
              mem_kw=dict(rd_lat=(2, 6), p_rd_req_ready=1.0, p_rd_rsp_valid=1.0, p_stall=0.0),
              spm_kw=dict(p_rd_ready=1.0, p_wr_ready=0.4, p_stall=0.04, stall_len=(60, 400))).start()
    qd = env.qd
    for k in range(10):
        n = min(env.lines, rnd.randint(qd + 64, qd + 300), env.limit - 64)
        dram = rnd.randrange(0, env.limit - n)
        await env.op(0, rnd.randrange(env.n), rnd.randint(0, env.lines - n), dram, n)
    await env.quiesce()
    m = env.ldmon
    dut._log.info("dma_reserve: %d LDs, %d lines, claim max %d, queue max %d (queue %d), %d cycles with "
                  "rd_rsp_ready checked", env.ops, env.beats, m.claim_max, m.q_max, qd, m.ready_cycles)
    assert m.claim_max >= qd - 63, "the reservation never reached the queue limit (%d of %d)" % (m.claim_max, qd)
    assert m.q_max >= qd - 64, "the beat queue never filled (%d of %d)" % (m.q_max, qd)
    env.stop()


@cocotb.test()
async def dma_reset(dut):
    """Synchronous reset in the middle of an LD/ST: every output returns to its reset state
    (cmd_ready 1, no valid, rd_rsp_ready 1), then later commands run normally."""
    cfg, rnd = _cfg_rnd(8)
    await clock_reset(dut)
    env = Env(dut, cfg, rnd, system=False, spm_kw=dict(p_rd_ready=0.7, p_wr_ready=0.7)).start()
    for k in range(8):
        is_st, ch, spm, dram, n = env.rand_op(nmax_cap=700)
        task = cocotb.start_soon(env.drv.op(is_st, ch, spm, dram, n))
        await ClockCycles(dut.clk, rnd.randint(1, max(2, 2 * n)))
        task.cancel()
        dut.rst_n.value = 0
        env.drv.reset()
        await ClockCycles(dut.clk, rnd.randint(1, 4))
        await FallingEdge(dut.clk)
        exp = dict(cmd_ready=1, done=0, rd_req_valid=0, wr_req_valid=0, wr_dat_valid=0, spm_rd_req_valid=0,
                   spm_wr_valid=0, rd_rsp_ready=(1 << env.n) - 1, wr_rsp_ready=(1 << env.n) - 1)
        for name, v in exp.items():
            assert int(getattr(dut, name).value) == v, "in reset: %s = %d" % (name, int(getattr(dut, name).value))
        await RisingEdge(dut.clk)
        dut.rst_n.value = 1
        for _ in range(4):
            await env.op(*env.rand_op(nmax_cap=300), gap=rnd.choice([0, 2]))
    await env.quiesce()
    dut._log.info("dma_reset: %d checked ops after 8 mid-op resets", env.ops)
    env.stop()
