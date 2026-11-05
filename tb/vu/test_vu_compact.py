"""C-oracle and transfer capture for the eight-lane VU coefficient transport."""

import json
import os
import struct
from pathlib import Path

import cocotb
from cocotb.triggers import FallingEdge
import pytest

from common.runner import run
import test_vu_unit as U


def rope_ops():
    # The first command rotates only half of each head. Four tokens force both
    # angle halves to be released and reused before the NEOX command starts.
    return [
        U.V.enc_rope(0, 1000, 100, 900, 128, 64, 2, T=4, sm=1, sts=2),
        U.V.enc_rope(1, 1100, 200, 900, 128, 128, 2, T=4, sm=1, sts=2),
    ]


@cocotb.test()
async def compact_coefficients(dut):
    bench = U.Bench(dut, 0xCAFE, noise=True)
    await bench.start()
    vu = dut.u_vu
    writes, releases, producer_releases = [], [], []
    origin = bench.rtl.cycles
    address_bits = (bench.L - 1).bit_length()
    active = True

    async def capture():
        while active:
            await FallingEdge(dut.clk)
            if not int(dut.rst_n.value):
                continue
            cycle = bench.rtl.cycles - origin
            valid = int(vu.wr_valid.value) & int(vu.wr_ready.value)
            for bank in range(2):
                if valid & (1 << bank):
                    writes.append(dict(
                        cycle=cycle, kind="SPM", bank=bank,
                        line=(int(vu.wr_line.value) >> (bank * address_bits)) & ((1 << address_bits) - 1),
                        data=hex((int(vu.wr_data.value) >> (bank * 512)) & ((1 << 512) - 1)),
                        enables=hex((int(vu.wr_be.value) >> (bank * 64)) & ((1 << 64) - 1)),
                    ))
            if int(vu.u_red.s_we.value):
                writes.append(dict(cycle=cycle, kind="SREG", index=int(vu.u_red.s_wi.value),
                                   data=hex(int(vu.u_red.s_wd.value))))
            if int(vu.ang_rel.value):
                releases.append(dict(cycle=cycle, half=int(vu.ang_rel_h.value)))
            if int(vu.u_ang.ang_rel.value):
                producer_releases.append(dict(cycle=cycle, half=int(vu.u_ang.ang_rel_h.value)))

    observer = cocotb.start_soon(capture())

    # init=0 exercises scalar word selection and elementwise coefficients in the
    # same AMERGE. Seventeen and forty rows cover both contexts and partial lines.
    for rows, init in [(17, 0), (40, 1), (40, 0)]:
        pre, op = U.gen_amerge(bench, R=rows, dcode=0, init=init)
        await bench.prewrite(pre)
        await bench.batch([op], f"AMERGE R={rows} init={init}")

    positions = struct.pack("<16I", 1, 17, 73, 111, *([0] * 12))
    await bench.prewrite([(900, positions)])
    await bench.batch([bench.movi(1, U.f2u(0.75)), bench.movi(2, U.f2u(0.9))], "ROPE setup")
    await bench.batch(rope_ops(), "NORM partial head and NEOX with angle half reuse")
    assert [entry["half"] for entry in releases] == [0, 1, 0, 1] * 2, releases

    for fmt in (0, 1):
        pre, dqrow = U.gen_dqrow(bench, fmt=fmt, K=256, rows=2)
        await bench.prewrite(pre)
        await bench.batch([
            U.V.enc_red(0, 0, 500, 16 * 65 + 7, to_s=1, sd=8),
            dqrow,
            U.V.enc_vop(0x06, 3500, 600, 0, 0, 129),
            U.V.enc_vop(0x00, 3600, 600, 700, 0, 129),
            U.V.enc_red(2, 0, 500, 17, to_s=1, sd=9),
        ], f"DQROW fmt={fmt}, reductions, front and back exits")

    # Interrupt a live ROPE command after its first half becomes consumable.
    await bench.rtl.push(rope_ops()[1])
    for _ in range(20000):
        await FallingEdge(dut.clk)
        if int(vu.ang_ready.value):
            break
    else:
        raise AssertionError("no populated angle half before reset")
    await bench.rtl.reset(noise=True, seed=19)
    bench.rtl.done = bench.rtl.pushed = 0
    for _ in range(160):
        await FallingEdge(dut.clk)
        assert not int(dut.probe_write_pending.value)
        assert not int(vu.ang_ready.value)
    bench.o.close()
    bench.o = U.Oracle(bench.spm_bytes, U.cfg()["num_ch"], caps=0x21 | 0x8 | 0x40)
    await bench.prewrite([(i, U.rnd_line(bench.r)) for i in range(bench.L)])
    await bench.prewrite([(900, positions)])
    await bench.batch([bench.movi(1, U.f2u(1.0)), bench.movi(2, U.f2u(0.8))], "ROPE setup after reset")
    await bench.batch(rope_ops(), "both ROPE modes after reset")

    active = False
    observer.cancel()
    assert bench.rtl.wait_cycles > 0
    record = dict(config=U.cfg(), batches=bench.cycle_records, ordered_writes=writes,
                  angle_releases=releases, producer_angle_releases=producer_releases,
                  final_sreg=bench.rtl.sreg(),
                  spm_wait_cycles=bench.rtl.wait_cycles)
    bench.finish("compact_coefficients")
    Path(os.environ["TSR_VU_COMPACT_RECORD"]).write_text(json.dumps(record, indent=2) + "\n")


@pytest.mark.parametrize("pipe", [0, 2, 3])
def test_compact_coefficients(pipe, tmp_path):
    config = dict(spm_bytes=262144, num_ch=4, n=1.0, topk=1, dqrow=1,
                  front_lanes=8, exp_lanes=8, back_lanes=4, lane_pipe=pipe)
    sources = U.vu_sources()
    baseline = os.environ.get("TSR_VU_COMPACT_SOURCE_ROOT")
    if baseline:
        sources = [Path(baseline) / source.relative_to(U.REPO) for source in sources]
        assert all(source.is_file() for source in sources)
    record_dir = Path(os.environ.get("TSR_VU_COMPACT_RECORD_DIR", tmp_path))
    record_dir.mkdir(parents=True, exist_ok=True)
    run("tsr_vu_tb", __name__, sources=sources, name="vu_compact_coefficients",
        parameters=dict(SPM_BYTES=262144, SPM_RD_LAT=4, NUM_CH=4,
                        FRONT_LANES=8, EXP_LANES=8, BACK_LANES=4, LANE_PIPE=pipe,
                        HAS_TOPK=1, HAS_DQROW=1),
        extra_env={"TSR_VU_UNIT_CFG": json.dumps(config),
                   "TSR_VU_COMPACT_RECORD": str((record_dir / f"pipe{pipe}.json").resolve())})
