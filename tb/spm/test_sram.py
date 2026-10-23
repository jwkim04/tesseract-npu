"""tsr_sram_1r1w (rtl/tech/generic) testbench: the contract every target variant must meet.

Checked every cycle against a byte-enable model: rdata = mem[raddr] exactly RD_LAT cycles after
a cycle with re = 1 (fully pipelined), read-first on a same-cycle same-address write, a read in
any later cycle sees the write, byte-granular write enables. Bytes never written are unknown
(no array reset) and not compared; cycles without a read result are not compared.

pytest: test_sram[cfg] builds one configuration and runs the cocotb tests sram_*.
"""
from __future__ import annotations

import json
import os

import cocotb
import pytest
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge

from common.runner import rtl_sources, run
from common.tbutil import CLK_PERIOD_NS
from common.tbutil import rng as tb_rng

CONFIGS = {
    "d100_w512_lat1": dict(DEPTH=100, WIDTH=512, RD_LAT=1),
    "d1024_w512_lat3": dict(DEPTH=1024, WIDTH=512, RD_LAT=3),
    "d1_w72_lat2": dict(DEPTH=1, WIDTH=72, RD_LAT=2),
    "d37_w8_lat4": dict(DEPTH=37, WIDTH=8, RD_LAT=4),
}


def sram_sources():
    srcs = [s for s in rtl_sources() if s.name == "tsr_sram_1r1w.sv"]
    assert len(srcs) == 1
    return srcs


@pytest.mark.parametrize("cfg", list(CONFIGS))
def test_sram(cfg):
    params = CONFIGS[cfg]
    run("tsr_sram_1r1w", __name__, parameters=params, sources=sram_sources(), name="sram_" + cfg,
        testcase=["sram_sweep", "sram_random"], extra_env={"TSR_SRAM_CFG": json.dumps(params)})


class SramBench:
    def __init__(self, dut, cfg):
        self.dut = dut
        self.depth = cfg["DEPTH"]
        self.width = cfg["WIDTH"]
        self.bw = self.width // 8
        self.lat = cfg["RD_LAT"]
        self.mem = {}       # addr -> (data, known mask)
        self.pend = {}      # cycle -> (addr, data, known)
        self.cyc = 0
        self.n_chk = 0
        self.collisions = 0

    def mask(self, be):
        m = 0
        for b in range(self.bw):
            if (be >> b) & 1:
                m |= 0xFF << (8 * b)
        return m

    async def step(self, we=0, waddr=0, wdata=0, wbe=0, re=0, raddr=0):
        d = self.dut
        d.we.value = we
        d.waddr.value = waddr
        d.wdata.value = wdata
        d.wbe.value = wbe
        d.re.value = re
        d.raddr.value = raddr
        await FallingEdge(d.clk)
        exp = self.pend.pop(self.cyc, None)
        if exp is not None:
            addr, data, known = exp
            got = int(d.rdata.value)
            assert (got ^ data) & known == 0, "cycle %d addr %d: got %#x expected %#x (mask %#x)" % (
                self.cyc, addr, got, data, known)
            self.n_chk += 1
        if re:
            self.pend[self.cyc + self.lat] = (raddr,) + self.mem.get(raddr, (0, 0))
        if we:
            if re and raddr == waddr:
                self.collisions += 1
            m = self.mask(wbe)
            old, known = self.mem.get(waddr, (0, 0))
            self.mem[waddr] = ((old & ~m) | (wdata & m), known | m)
        await RisingEdge(d.clk)
        self.cyc += 1

    async def drain(self):
        for _ in range(self.lat + 2):
            await self.step()
        assert not self.pend


async def setup(dut):
    cfg = json.loads(os.environ["TSR_SRAM_CFG"])
    dut.we.value = 0
    dut.re.value = 0
    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, unit="ns").start())
    await RisingEdge(dut.clk)
    return SramBench(dut, cfg)


@cocotb.test()
async def sram_sweep(dut):
    """Every address written with full byte enables, then read back back to back."""
    sb = await setup(dut)
    rnd = tb_rng(21)
    full = (1 << sb.bw) - 1
    for a in range(sb.depth):
        await sb.step(we=1, waddr=a, wdata=rnd.getrandbits(sb.width), wbe=full)
    for a in reversed(range(sb.depth)):
        await sb.step(re=1, raddr=a)
    await sb.drain()
    assert sb.n_chk == sb.depth


@cocotb.test()
async def sram_random(dut):
    """Random reads/writes every cycle, partial byte enables, hot addresses for collisions and
    write -> next-cycle read."""
    sb = await setup(dut)
    rnd = tb_rng(22)
    full = (1 << sb.bw) - 1
    for a in range(sb.depth):
        await sb.step(we=1, waddr=a, wdata=rnd.getrandbits(sb.width), wbe=full)
    hot = [0, sb.depth - 1, sb.depth // 2]
    prev_w = 0
    for _ in range(6000):
        def addr():
            return rnd.choice(hot) if rnd.random() < 0.4 else rnd.randrange(sb.depth)
        we = int(rnd.random() < 0.6)
        wa = addr()
        r = rnd.random()
        wbe = full if r < 0.5 else (rnd.getrandbits(sb.bw) if r < 0.9 else 0)
        re = int(rnd.random() < 0.7)
        ra = prev_w if rnd.random() < 0.2 else addr()
        await sb.step(we=we, waddr=wa, wdata=rnd.getrandbits(sb.width), wbe=wbe, re=re, raddr=ra)
        prev_w = wa
    await sb.drain()
    dut._log.info("sram random: %d reads checked, %d same-cycle collisions", sb.n_chk, sb.collisions)
    assert sb.collisions > 0 and sb.n_chk > 3000
