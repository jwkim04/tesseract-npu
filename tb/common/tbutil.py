"""Small cocotb helpers shared by the Tesseract testbenches."""
from __future__ import annotations

import os
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, FallingEdge, RisingEdge

CLK_PERIOD_NS = 10   # simulation only; the U200 target is 300 MHz


def seed() -> int:
    """Test seed: $TSR_SEED (set by tb/common/runner.py from pytest --tsr-seed), else 1."""
    return int(os.environ.get("TSR_SEED", "1"), 0)


def rng(salt: int = 0) -> random.Random:
    return random.Random(seed() * 1000003 + salt)


async def start_clock_and_reset(dut, clk=None, rst_n=None, cycles: int = 5):
    """Start a clock on dut.clk and hold dut.rst_n low for `cycles` cycles (synchronous reset).
    Returns after the edge at which rst_n is released (DUT inputs may be driven right away)."""
    clk = clk if clk is not None else dut.clk
    rst_n = rst_n if rst_n is not None else dut.rst_n
    cocotb.start_soon(Clock(clk, CLK_PERIOD_NS, unit="ns").start())
    rst_n.value = 0
    await ClockCycles(clk, cycles)
    await RisingEdge(clk)
    rst_n.value = 1


async def wait_level(clk, sig, value: int = 1, timeout_cycles: int = 100000) -> int:
    """Wait (sampling at falling edges) until sig == value; returns the number of cycles waited."""
    for n in range(timeout_cycles):
        await FallingEdge(clk)
        if int(sig.value) == value:
            return n
    raise TimeoutError("%s did not become %d within %d cycles" % (sig._name, value, timeout_cycles))
