"""The compact row-max packet and local score address mapping on actual VU lanes."""

from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge

from common.runner import run


ONE = 0x3F800000
NEG_INF = 0xFF800000
LINE_MASK = (1 << 512) - 1
LINE_ONE = sum(ONE << (32 * j) for j in range(16))


def command(*, start=0, slot=1, first=0, groups=0, max_valid=0, row=0, value=0):
    return (start << 53 | slot << 51 | first << 49 | groups << 41 |
            max_valid << 40 | row << 32 | value)


@cocotb.test()
async def owner_groups(dut):
    dut.rst_n.value = 0
    dut.cmd.value = 0
    dut.load_valid.value = 0
    cocotb.start_soon(Clock(dut.clk, 4, unit="ns").start())
    for _ in range(6):
        await FallingEdge(dut.clk)
    dut.rst_n.value = 1

    for first in range(4):
        for groups in (1, 5, 8, 13):
            expected = {}
            rows = 3
            stride = (groups + 3) // 4
            for row in range(rows):
                # Normalization of a lane equal to the row max is exactly one.
                maximum = ONE if row & 1 else 0
                for rel in range(groups):
                    channel = (first + rel) % 4
                    if channel & 1:
                        continue
                    owner = channel // 2
                    address = row * stride + rel // 4
                    masked = (row + rel) & 1
                    dut.load_valid.value = 1
                    dut.load_owner.value = owner
                    dut.load_address.value = address
                    dut.load_data.value = sum((NEG_INF if masked else maximum) << (32 * w) for w in range(16))
                    expected[owner, address] = 0 if masked else LINE_ONE
                    await FallingEdge(dut.clk)
                dut.load_valid.value = 0
                dut.cmd.value = command(max_valid=1, row=row, value=maximum)
                await FallingEdge(dut.clk)
                dut.cmd.value = 0
            dut.cmd.value = command(start=1, slot=2, first=first, groups=groups, row=rows, value=ONE)
            await FallingEdge(dut.clk)
            dut.cmd.value = 0
            for _ in range(500):
                if dut.done.value:
                    assert not expected
                    break
                write = int(dut.wr.value)
                for owner in range(2):
                    if write >> owner & 1:
                        addr = int(dut.wr_address.value) >> (13 * owner) & 0x1FFF
                        line = int(dut.wr_data.value) >> (512 * owner) & LINE_MASK
                        assert line == expected.pop((owner, addr))
                await FallingEdge(dut.clk)
            else:
                raise AssertionError("ASMAX did not retire")
            await FallingEdge(dut.clk)


def test_local_asmax():
    sv = Path(__file__).resolve().parent / "hdl" / "tsr_tb_local_asmax.sv"
    run("tsr_tb_local_asmax", __name__, extra_sources=[sv], testcase=["owner_groups"])
