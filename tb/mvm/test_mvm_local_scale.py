"""Ordered local SCALE output with masked words, queue stalls and reset."""

import random
import struct
from collections import deque
from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge, Timer

from common.runner import run


def bits(x):
    return int.from_bytes(struct.pack("<f", x), "little")


# tsr_m_tanh(tsr_fmul(x, -0.25)) * 2. The reference consumes FP32 x.
SOFTCAP = (
    (-4.0, 0x3FC2F7D5), (-2.0, 0x3F6C9A9F), (-0.4, 0x3E4C1EB9),
    (-0.0, 0x00000000), (0.0, 0x80000000), (0.4, 0xBE4C1EB9),
    (2.0, 0xBF6C9A9F), (4.0, 0xBFC2F7D5), (100.0, 0xC0000000), (-100.0, 0x40000000),
)


def packet(meta, words):
    return meta << 512 | sum(word << (32 * i) for i, word in enumerate(words))


@cocotb.test()
async def ordered_scale(dut):
    dut.s_valid.value = 0
    dut.m_ready.value = 0
    dut.scale.value = bits(-0.25)
    dut.softcap.value = 0
    dut.post_scale.value = bits(1)
    dut.rst_n.value = 0
    cocotb.start_soon(Clock(dut.clk, 4, unit="ns").start())
    rng = random.Random(4)

    async def workload(count, reset_cycle=None, softcap=False):
        for _ in range(5):
            await FallingEdge(dut.clk)
        dut.rst_n.value = 1
        dut.softcap.value = softcap
        dut.post_scale.value = bits(2)
        expected = deque()
        offered = None
        sent = 0
        stalls = 0
        for cycle in range(8000):
            await FallingEdge(dut.clk)
            if offered is None and sent < count:
                own = rng.getrandbits(16)
                if softcap:
                    words, values = zip(*(rng.choice(SOFTCAP) for _ in range(16)))
                else:
                    words = [rng.randrange(1, 200) for _ in range(16)]
                    values = [bits(-0.25 * word) for word in words]
                meta = own << 16 | sent << 32 | rng.getrandbits(16)
                offered = (packet(meta, list(map(bits, words))),
                           packet(meta, [v if own >> j & 1 else 0
                                         for j, v in enumerate(values)]))
            dut.s_valid.value = int(offered is not None)
            dut.s_data.value = offered[0] if offered else 0
            ready = reset_cycle is None and cycle > 200 and rng.randrange(4) != 0
            dut.m_ready.value = ready
            await Timer(1, unit="ns")
            if offered is not None and dut.s_ready.value:
                expected.append(offered[1])
                sent += 1
                offered = None
            elif offered is not None:
                stalls += 1
            if dut.m_valid.value and ready:
                assert int(dut.m_data.value) == expected.popleft()
            await RisingEdge(dut.clk)
            if cycle == reset_cycle:
                assert len(expected) == 16 and stalls
                dut.rst_n.value = 0
                dut.s_valid.value = 0
                dut.m_ready.value = 0
                return
            if sent == count and not expected:
                assert stalls
                return
        raise AssertionError("scaled output queue did not drain")

    await workload(32, reset_cycle=100)
    await workload(32, reset_cycle=180, softcap=True)
    await workload(100)
    await workload(50, softcap=True)
    await workload(40)


def test_local_scale():
    sv = Path(__file__).resolve().parent / "hdl" / "tsr_tb_mvm_local_scale.sv"
    run("tsr_tb_mvm_local_scale", __name__, extra_sources=[sv], testcase=["ordered_scale"])
