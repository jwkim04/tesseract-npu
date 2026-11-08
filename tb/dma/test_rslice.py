"""tb/dma test_rslice — the pointer and occupancy copies of tsr_rslice FORM 1 (RPC) and of tsr_mvm_pair_rx are
cycle-identical to the single flops: the same ready / valid in every cycle and the same data while valid; FORM 0 and
FORM 1 move the same items in the same cycles. Random valid and ready patterns, W = 577 and an odd width."""
from __future__ import annotations

import random
from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge

from common.runner import run

HERE = Path(__file__).resolve().parent


@cocotb.test()
async def rslice_copies(dut):
    rng = random.Random(7)
    W = int(dut.W.value)
    cocotb.start_soon(Clock(dut.clk, 4, unit="ns").start())
    dut.rst_n.value = 0
    dut.s_valid.value = 0
    dut.f_valid.value = 0
    dut.m_ready.value = 0
    dut.s_data.value = 0
    dut.f_data.value = 0
    for _ in range(5):
        await FallingEdge(dut.clk)
    dut.rst_n.value = 1
    items = [[] for _ in range(4)]
    pitems = [[], []]
    n_cmp = 0
    for cyc in range(6000):
        pv = rng.choice([0.2, 0.6, 0.95])
        if cyc % 500 == 0:
            ps = rng.choice([0.2, 0.6, 0.95])
            pr = rng.choice([0.2, 0.6, 0.95])
        dut.s_valid.value = int(rng.random() < ps)
        dut.s_data.value = rng.getrandbits(W)
        dut.f_valid.value = int(rng.random() < ps)
        dut.f_data.value = rng.getrandbits((W + 1) // 2)
        dut.m_ready.value = int(rng.random() < pr)
        await FallingEdge(dut.clk)
        sr = int(dut.s_ready.value)
        mv = int(dut.m_valid.value)
        fr = int(dut.f_ready.value)
        pvd = int(dut.p_valid.value)
        # FORM 1 with one flop, with 8 copies and with 3 copies: identical every cycle
        assert (sr >> 1) & 1 == (sr >> 2) & 1 == (sr >> 3) & 1, "s_ready differs at %d: %x" % (cyc, sr)
        assert (mv >> 1) & 1 == (mv >> 2) & 1 == (mv >> 3) & 1, "m_valid differs at %d: %x" % (cyc, mv)
        # FORM 0: same ready / valid too (the cycle-identical contract of the header)
        assert sr & 1 == (sr >> 1) & 1 and mv & 1 == (mv >> 1) & 1, "FORM 0 differs at %d: s_ready %x m_valid %x" % (cyc, sr, mv)
        assert fr & 1 == (fr >> 1) & 1 and pvd & 1 == (pvd >> 1) & 1, "pair_rx copies differ at %d" % cyc
        if mv & 1:
            d = int(dut.m_data.value)
            ds = [(d >> (W * k)) & ((1 << W) - 1) for k in range(4)]
            assert ds[0] == ds[1] == ds[2] == ds[3], "data differs at %d" % cyc
            n_cmp += 1
        if pvd & 1:
            d = int(dut.p_data.value)
            assert (d & ((1 << W) - 1)) == ((d >> W) & ((1 << W) - 1)), "pair_rx data differs at %d" % cyc
            n_cmp += 1
    assert n_cmp > 500


def test_rslice_copies():
    for w in (577, 130):
        run("tsr_tb_rslice", __name__, parameters={"W": w}, extra_sources=[HERE / "hdl" / "tsr_tb_rslice.sv"],
            name="rslice_w%d" % w, testcase=["rslice_copies"])
