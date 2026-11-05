"""Cycle/order capture for the bank-local first reduction level."""
import json
import os
import struct
from pathlib import Path

import cocotb
from cocotb.triggers import FallingEdge

from common.runner import run
import test_vu_unit as U


@cocotb.test()
async def local_reduce_order(dut):
    bench = U.Bench(dut, 0xB4A1, noise=True)
    await bench.start()
    trace = []
    active = True
    origin = bench.rtl.cycles
    vu = dut.u_vu
    address_width = (bench.L - 1).bit_length()

    async def capture():
        while active:
            await FallingEdge(dut.clk)
            cycle = bench.rtl.cycles - origin
            valid = int(vu.wr_valid.value) & int(vu.wr_ready.value)
            for bank in range(2):
                if valid & (1 << bank):
                    trace.append(dict(
                        cycle=cycle, kind="SPM", bank=bank,
                        line=(int(vu.wr_line.value) >> (bank * address_width)) & ((1 << address_width)-1),
                        data=hex((int(vu.wr_data.value) >> (512 * bank)) & ((1 << 512)-1)),
                        enables=hex((int(vu.wr_be.value) >> (64 * bank)) & ((1 << 64)-1)),
                    ))
            if int(vu.u_red.s_we.value):
                trace.append(dict(cycle=cycle, kind="SREG", index=int(vu.u_red.s_wi.value),
                                  data=hex(int(vu.u_red.s_wd.value))))

    observer = cocotb.start_soon(capture())
    source = 100
    length = 16 * 65 + 7
    lines = U.V.L(length)
    values = [U.f2u((-1.0 if (i // 16 + i % 8) % 2 else 1.0)
                     * 2.0 ** (24 if i % 16 < 8 else -10))
              for i in range(lines * 16 * 3)]
    await bench.prewrite([
        (source + line, struct.pack("<16I", *values[16*line:16*line+16]))
        for line in range(lines * 3)
    ])
    # Empty/MAX/SUM mode changes immediately around multiple groups and two
    # logical histories. Independent front/back exits can overlap their drains.
    await bench.batch([
        U.V.enc_red(0, 0, 0, 0, to_s=1, sd=1),
        U.V.enc_red(2, 0, 0, 0, to_s=1, sd=2),
        U.V.enc_red(0, 1000, source, length, rows=3),
        U.V.enc_vop(0x06, 1500, 300, 0, 0, 513),
        U.V.enc_red(2, 1002, source, 129, rows=2),
        U.V.enc_vop(0x00, 1600, 300, 400, 0, 129),
        U.V.enc_red(1, 0, source, length, to_s=1, sd=10),
    ], "empty/multiple groups/ordinary exits")
    for tail in (1, 7, 8, 9, 15, 17, 129):
        await bench.batch([
            U.V.enc_red(0, 0, source, tail, to_s=1, sd=3),
            U.V.enc_red(2, 0, source, tail, to_s=1, sd=4),
        ], "cross-context tail %d" % tail)
    for _ in range(4):
        operation = U.gen_amask(bench)
        await bench.prewrite(U.mask_prewrite(bench, operation))
        await bench.batch([operation], "AMASK local pairs")
    # Reset after eight groups have issued: the local staging/FP/return
    # pipelines contain real work, not only a queued injector row token.
    await bench.rtl.push(U.V.enc_red(0, 0, source, length, to_s=1, sd=7))
    for _ in range(20000):
        await FallingEdge(dut.clk)
        if int(vu.u_red.dq_v.value) and int(vu.u_red.q_q.value) >= 8:
            break
    else:
        raise AssertionError("no active reduction drain before reset")
    await bench.rtl.reset(noise=True, seed=19)
    bench.rtl.done = bench.rtl.pushed = 0
    for _ in range(160):
        await FallingEdge(dut.clk)
        assert not int(dut.probe_write_pending.value)
    bench.o.close()
    bench.o = U.Oracle(bench.spm_bytes, U.cfg()["num_ch"], caps=0x21 | 0x8 | 0x40)
    await bench.prewrite([(i, U.rnd_line(bench.r)) for i in range(bench.L)])
    await bench.batch([
        U.V.enc_red(0, 1700, 100, 16 * 17 + 9, rows=2),
        U.V.enc_vop(0x09, 1800, 200, 0, 0, 129),
        U.V.enc_red(2, 0, 100, 129, to_s=1, sd=8),
    ], "reset/ordinary exit/new row")
    active = False
    observer.cancel()
    record = dict(config=U.cfg(), operations=bench.cycle_records,
                  ordered_writes=trace, final_sreg=bench.rtl.sreg())
    Path(os.environ["TSR_LOCAL_REDUCE_RECORD"]).write_text(json.dumps(record, indent=2) + "\n")
    bench.finish("local_reduce_order")


def test_local_reduce_order(tmp_path):
    pipe = int(os.environ.get("TSR_LOCAL_REDUCE_PIPE", "2"))
    record = os.environ.get("TSR_LOCAL_REDUCE_RECORD", str(tmp_path / "ordered_results.json"))
    config = dict(spm_bytes=262144, num_ch=4, n=1.0, topk=1, dqrow=1,
                  front_lanes=8, exp_lanes=8, back_lanes=4, lane_pipe=pipe)
    run("tsr_vu_tb", __name__, sources=U.vu_sources(), name="vu_local_reduce_order",
        parameters=dict(SPM_BYTES=262144, SPM_RD_LAT=4, NUM_CH=4,
                        FRONT_LANES=8, EXP_LANES=8, BACK_LANES=4, LANE_PIPE=pipe,
                        HAS_TOPK=1, HAS_DQROW=1),
        extra_env={"TSR_VU_UNIT_CFG": json.dumps(config), "TSR_LOCAL_REDUCE_RECORD": record})
