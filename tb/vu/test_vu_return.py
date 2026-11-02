"""Cycle-exact miter for sharing the front/back return without serializing it."""
from __future__ import annotations

import random
from pathlib import Path

import cocotb
import pytest
from cocotb.triggers import Timer

from common.runner import REPO, run

HERE = Path(__file__).resolve().parent


@cocotb.test()
async def return_timing(dut):
    rng = random.Random(0xFB512)
    dut.clk.value = 0
    dut.rst_n.value = 0
    dut.exit_valid.value = 0
    dut.exit_back.value = 0
    dut.raw_front.value = 0
    dut.raw_back.value = 0
    checked = 0

    async def cycle(reset=False, back=None):
        nonlocal checked
        dut.clk.value = 0
        dut.rst_n.value = not reset
        dut.exit_valid.value = back is not None
        dut.exit_back.value = bool(back)
        dut.raw_front.value = rng.getrandbits(512)
        dut.raw_back.value = rng.getrandbits(256)  # BACK8's other words are constant0
        await Timer(5, unit="ns")
        dut.clk.value = 1
        await Timer(5, unit="ns")
        if int(dut.valid.value):
            assert not int(dut.collision.value), "test scheduled conflicting exits"
            assert int(dut.actual.value) == int(dut.expected.value), "return changed value/cycle"
            checked += 1

    for _ in range(8):
        await cycle(reset=True)
    # Absolute scheduling at the injector's descriptor port; no implementation
    # counter is copied. Reserve consecutive exits for 1/2/4 sub-slots, then
    # admit the next slot at the earliest cycle that leaves its exit after them.
    now, last_exit = 0, -1
    for is_back, count in [(True, 2), (False, 1), (False, 7), (True, 4),
                           (False, 2), (True, 2), (False, 8)] * 4:
        latency = 78 if is_back else 42
        while now + latency <= last_exit:
            await cycle()
            now += 1
        for _ in range(count):
            await cycle(back=is_back)
            last_exit = now + latency
            now += 1
    for _ in range(100):
        await cycle()
    # Leave old back descriptors in flight, reset for one cycle, then send
    # front results before that old back latency would have drained.
    for _ in range(6):
        await cycle(back=True)
    # Age those back tokens so they would collide with the new front exits
    # if the local selector delay were not cleared by reset.
    for _ in range(30):
        await cycle()
    await cycle(reset=True)
    for _ in range(10):
        await cycle(back=False)
    for _ in range(100):
        await cycle()
    assert checked >= 100


@pytest.mark.parametrize("pipe", [0, 2, 3])
def test_vu_return_timing(pipe):
    run("tsr_vu_return_tb", __name__, name="vu_return_%d" % pipe,
        parameters={"PIPE": pipe}, sources=[REPO / "rtl/core/tsr_vu_pkg.sv",
            REPO / "rtl/core/tsr_vu_lane_share.sv", HERE / "hdl/tsr_vu_return_tb.sv"])
