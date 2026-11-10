"""tsr_mem_arb tests (docs/rtl-core.md §4): the arbiter alone, MemModel on its m_* port.

pytest function test_mem_arb[cfg] runs the cocotb tests
  arb_random  three read sources (random bursts, 4 local ids each, random rsp_ready) and the
              write source (random bursts, request/data offered independently) against a
              random-latency/backpressure memory: data, per-source per-id order, `last`, local id
              routing, write responses, written memory == shadow; RD_OUTSTANDING reached and never
              exceeded (MemModel strict); perf_rd_beat == memory read beats; memory-side ids
              {source, local}; combinational-path probes at random times
  arb_rr      all sources backlogged: grants strictly rotate 0,1,2,... (and 0,2,0,2 with source 1
              idle); a source holding rsp_ready low (local reason) delays only the shared
              response channel, nothing is lost
  arb_reset   synchronous reset in the middle of read/write traffic: outputs in reset state, the
              outstanding budget is whole again afterward (RD_OUTSTANDING reached, never exceeded)
configurations: rdo4 (RD_OUTSTANDING 4, ADDR_BEAT_W 20), u200 (16, 28), rdo1 (1, 13)
"""
from __future__ import annotations

import json
import random

import cocotb
import pytest
from cocotb.triggers import ClockCycles, FallingEdge, RisingEdge

from common import isa
from common.mem_model import MemModel
from common.runner import run
from common.tbutil import seed as tb_seed

import dma_tb
from dma_tb import BurstMonitor, PerfMonitor, ReadSources, WriteSource, clock_reset, comb_probe

CFGS = {
    "rdo4": dict(ADDR_BEAT_W=20, RD_OUTSTANDING=4),
    "u200": dict(ADDR_BEAT_W=28, RD_OUTSTANDING=16),
    "rdo1": dict(ADDR_BEAT_W=13, RD_OUTSTANDING=1),
}

ARB_INPUTS = ["rst_n", "s_rd_req_valid", "s_rd_req_addr", "s_rd_req_len", "s_rd_req_id", "s_rd_rsp_ready",
              "s_wr_req_valid", "s_wr_req_addr", "s_wr_req_len", "s_wr_req_id", "s_wr_dat_valid",
              "s_wr_dat_data", "s_wr_dat_mask", "s_wr_dat_last", "s_wr_rsp_ready", "m_rd_req_ready", "m_rd_rsp_valid",
              "m_rd_rsp_data", "m_rd_rsp_id", "m_rd_rsp_last", "m_wr_req_ready", "m_wr_dat_ready",
              "m_wr_rsp_valid", "m_wr_rsp_id"]
ARB_OUTPUTS = ["s_rd_req_ready", "s_rd_rsp_valid", "s_rd_rsp_data", "s_rd_rsp_last", "s_rd_rsp_id",
               "s_wr_req_ready", "s_wr_dat_ready", "s_wr_rsp_valid", "s_wr_rsp_id", "m_rd_req_valid",
               "m_rd_req_addr", "m_rd_req_len", "m_rd_req_id", "m_rd_rsp_ready", "m_wr_req_valid",
               "m_wr_req_addr", "m_wr_req_len", "m_wr_req_id", "m_wr_dat_valid", "m_wr_dat_data",
               "m_wr_dat_mask", "m_wr_dat_last", "m_wr_rsp_ready", "perf_rd_beat", "perf_rd_wait"]

NSRC = 3                       # tsr_pkg TSR_NSRC: AUX, DMA, WS
RD_REGION = (0, 2048)          # read-only, preloaded
WR_REGION = (4096, 6144)       # written by the write source only


@pytest.mark.parametrize("cfg", list(CFGS))
def test_mem_arb(cfg):
    p = CFGS[cfg]
    run("tsr_mem_arb", __name__, parameters=p, testcase=["arb_random", "arb_rr", "arb_reset"],
        extra_env={"TSR_DMA_CFG": json.dumps(dict(p, name=cfg))})


def _setup(dut, rnd, **mem_kw):
    cfg = dma_tb.cfg_from_env()
    aw, rdo = cfg["ADDR_BEAT_W"], cfg["RD_OUTSTANDING"]
    mem = MemModel(dut, num_ch=1, prefix="m_", addr_w=aw, rd_outstanding=rdo, limit_beats=1 << aw,
                   seed=rnd.getrandbits(31), **mem_kw)
    n = RD_REGION[1] - RD_REGION[0]
    mem.load(0, 64 * RD_REGION[0], rnd.getrandbits(512 * n).to_bytes(64 * n, "little"))
    src = ReadSources(dut, prefix="s_", lanes=NSRC, aw=aw, rnd=rnd,
                      expect=lambda k, b: mem.beat(0, b), region=RD_REGION, shared_rsp=True, name="arb")
    return cfg, aw, rdo, mem, src


async def _prober(dut, clock, rnd, count):
    while True:
        await ClockCycles(dut.clk, rnd.randint(30, 300))
        await comb_probe(dut, clock, ARB_INPUTS, ARB_OUTPUTS, rnd)
        count[0] += 1


@cocotb.test()
async def arb_random(dut):
    rnd = random.Random(tb_seed() * 7919 + 11)
    clock = await clock_reset(dut)
    cfg, aw, rdo, mem, src = _setup(dut, rnd, rd_lat=(1, 60), wr_lat=(1, 40))
    wr = WriteSource(dut, prefix="s_", masked=True, aw=aw, rnd=rnd, region=WR_REGION)
    bm = BurstMonitor(dut, prefix="m_", num_ch=1, aw=aw, id_w=isa.MEM_ID_W)
    perf = PerfMonitor(dut, 1)
    mem.start()
    src.start()
    wr.start()
    probes = [0]
    probe = cocotb.start_soon(_prober(dut, clock, rnd, probes))
    src.auto = True
    wr.plan_random(120)
    for _ in range(12000):
        await RisingEdge(dut.clk)
    # phase 2: skewed request rates (a lone requester, one heavy and two light requesters)
    for pl in ({0: 0.9, 1: 0.0, 2: 0.0}, {0: 0.0, 1: 0.0, 2: 0.9}, {0: 0.2, 1: 1.0, 2: 0.2}):
        src.p_req_lane = pl
        await ClockCycles(dut.clk, 1500)
    src.p_req_lane = {}
    await src.drain()
    await wr.drain()
    probe.cancel()
    await ClockCycles(dut.clk, 5)
    mem.check(idle=True)
    st = mem.stats()[0]
    assert st["max_outstanding"] == rdo, "max outstanding %d, RD_OUTSTANDING %d" % (st["max_outstanding"], rdo)
    assert perf.beats[0] == st["rd_beats"], "perf_rd_beat %d != read beats %d" % (perf.beats[0], st["rd_beats"])
    assert perf.wait[0] > 0
    assert sum(src.beats) == st["rd_beats"]
    assert all(g > 0 for g in src.grants), src.grants
    # memory-side ids: {source, local}; writes always {DMA, local}
    for _, kind, _, a, ln, mid in bm.log:
        if kind == "rd":
            assert (mid >> isa.SRC_ID_W) < NSRC, "read request with source id %d" % (mid >> 2)
        else:
            assert (mid >> isa.SRC_ID_W) == isa.SRC_DMA, "write request id 0x%x" % mid
    assert len([e for e in bm.log if e[1] == "rd"]) == sum(src.grants)
    # writes landed exactly
    assert mem.written[0] == set(wr.shadow), "written beat set differs from the write plan"
    for b, v in wr.shadow.items():
        assert mem.beat(0, b) == v, "beat 0x%x: write data mismatch" % b
    idle_wait = perf.wait[0]
    await ClockCycles(dut.clk, 30)
    assert perf.wait[0] == idle_wait, "perf_rd_wait counted while idle"
    dut._log.info("arb_random: grants %s, beats %s, %d write bursts, %d probes, mem %s",
                  src.grants, src.beats, wr.bursts, probes[0], st)


@cocotb.test()
async def arb_rr(dut):
    rnd = random.Random(tb_seed() * 7919 + 12)
    await clock_reset(dut)
    cfg, aw, rdo, mem, src = _setup(dut, rnd, rd_lat=(1, 3), p_rd_req_ready=1.0, p_rd_rsp_valid=1.0,
                                    p_stall=0.0)
    src.p_rsp_ready = 1.0
    bm = BurstMonitor(dut, prefix="m_", num_ch=1, aw=aw, id_w=isa.MEM_ID_W)
    mem.start()
    src.start()

    async def phase(active, count):
        log0 = len(bm.log)
        for k in active:
            for _ in range(count):
                a, ln, i = src.rand_req()
                src.plan[k].append((a, min(ln, 4), i))
        await src.drain()
        seq = [e[5] >> isa.SRC_ID_W for e in bm.log[log0:] if e[1] == "rd"]
        assert len(seq) == count * len(active)
        return seq

    # all three backlogged: strict rotation (the first grant after reset goes to source 0)
    seq = await phase([0, 1, 2], 40)
    assert seq == [0, 1, 2] * 40, "grant order %s" % seq[:24]
    # source 1 idle: 0 and 2 alternate
    seq = await phase([0, 2], 30)
    assert all(seq[j] != seq[j + 1] for j in range(len(seq) - 1)), "grant order %s" % seq[:24]
    # a lone source gets every grant back to back
    seq = await phase([1], 20)
    assert seq == [1] * 20
    # head-of-line: source 2 drops rsp_ready for 300 cycles (a local reason, rule M6) while all
    # three sources read: the shared response channel waits, nothing is lost or reordered.
    for k in range(3):
        for _ in range(10):
            src.plan[k].append(src.rand_req())
    src.p_rsp_lane = {2: 0.0}
    await ClockCycles(dut.clk, 300)
    assert src.n_out[2] > 0, "source 2 had nothing outstanding during its stall"
    src.p_rsp_lane = {}
    await src.drain()
    await ClockCycles(dut.clk, 5)
    mem.check(idle=True)
    dut._log.info("arb_rr: ok, grants %s", src.grants)


@cocotb.test()
async def arb_reset(dut):
    rnd = random.Random(tb_seed() * 7919 + 13)
    await clock_reset(dut)
    cfg, aw, rdo, mem, src = _setup(dut, rnd, rd_lat=(20, 80), wr_lat=(1, 40))
    wr = WriteSource(dut, prefix="s_", aw=aw, rnd=rnd, region=WR_REGION)
    mem.start()
    src.start()
    wr.start()
    for k in range(4):
        src.auto = True
        src.p_req = 0.8
        wr.plan_random(20)
        await ClockCycles(dut.clk, rnd.randint(50, 400))
        dut.rst_n.value = 0                       # mid-traffic: reads and writes in flight
        await ClockCycles(dut.clk, 2)
        await FallingEdge(dut.clk)
        for name, v in dict(m_rd_req_valid=0, m_wr_req_valid=0, m_wr_dat_valid=0, s_rd_rsp_valid=0,
                            s_wr_rsp_valid=0, perf_rd_beat=0, perf_rd_wait=0, s_rd_req_ready=7,
                            s_wr_req_ready=1, s_wr_dat_ready=1, m_rd_rsp_ready=1, m_wr_rsp_ready=1).items():
            assert int(getattr(dut, name).value) == v, "in reset: %s = %d" % (name, int(getattr(dut, name).value))
        await RisingEdge(dut.clk)
        dut.rst_n.value = 1
        src.auto = False
        await ClockCycles(dut.clk, 2)
        mem.clear_tracking()
        # after reset: full budget available again, every read checked
        st0 = dict(mem.stats()[0])
        mem.ch[0].max_outstanding = 0
        src.p_req = 1.0
        src.auto = True
        await ClockCycles(dut.clk, 300)
        await src.drain()
        assert mem.stats()[0]["max_outstanding"] == rdo, "budget after reset %d != %d" % (
            mem.stats()[0]["max_outstanding"], rdo)
        del st0
    wr.plan_random(30)
    await wr.drain()
    await ClockCycles(dut.clk, 5)
    mem.check(idle=True)
    for b, v in wr.shadow.items():
        assert mem.beat(0, b) == v
    dut._log.info("arb_reset: ok, grants %s", src.grants)
