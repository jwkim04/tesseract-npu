"""Integrated LD/KVST/ST channel handover and matched-cycle selector regression."""
import json
import os
import sys
from pathlib import Path

import cocotb
from cocotb.triggers import RisingEdge

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import test_at_kvst as kvst
from common import isa
from common.runner import rtl_sources, run


@cocotb.test()
async def response_handover(dut):
    cfg = kvst.cfg_env()
    bench = kvst.Bench(dut, cfg, 27)
    for name in ("chk_req_valid", "chk_req_insn", "cmd_valid", "cmd", "cmd_xc", "xcmd"):
        getattr(dut, name).value = 0
    bench.port.start()
    bench.mem.start()
    await kvst.B.clock_reset(dut)
    cocotb.start_soon(bench.perf_mon())

    cycles = []
    original_dma = bench.dma

    async def measured_dma(**kwargs):
        elapsed = await original_dma(**kwargs)
        if kwargs.get("xc"):
            kind = "KVST"
        else:
            command = isa.DMA_CMD_T.unpack(kwargs.get("cmd", 0))
            kind = "ST" if command["is_st"] else "LD"
        cycles.append(dict(kind=kind, cycles=elapsed))
        return elapsed

    bench.dma = measured_dma

    async def ldst(is_store, channel):
        n_lines = 73
        line = 100
        beat = 127 + channel * 128
        data = bench.r.randbytes(n_lines * 64)
        if is_store:
            bench.spm[line * 64:(line + n_lines) * 64] = data
            bench.sim.spm_write(line * 64, data)
        else:
            bench.mem.load(channel, beat * 64, data)
            bench.sim.dram_write(channel, beat * 64, data)
        instruction = [
            (isa.OP_ST if is_store else isa.OP_LD) | (channel << 8),
            line, beat, n_lines, 0, 0, 0, 0,
        ]
        assert bench.sim.op(instruction)[0] == 0
        command = isa.DMA_CMD_T.pack(
            is_st=is_store, ch=channel, spm_line=line, dram_beat=beat, n_lines=n_lines,
        )
        await bench.dma(cmd=command)
        for _ in range(4):
            await RisingEdge(dut.clk)
        bench.mem.check(idle=True)
        if is_store:
            assert bench.mem.read_bytes(channel, beat * 64, n_lines * 64) == bench.sim.dram_read(
                channel, beat * 64, n_lines * 64,
            )
        else:
            assert bytes(bench.spm[line * 64:(line + n_lines) * 64]) == bench.sim.spm_read(
                line * 64, n_lines * 64,
            )

    for order in ("req_first", "data_first", "any"):
        bench.mem.wr_order = order
        for channel in range(bench.C):
            await ldst(False, channel)
            # K and V cross group/channel boundaries with partial metadata RMW.
            # A full sixteen-row group also exercises the no-RMW path. The
            # shared Bench checks every DRAM/SPM byte against the C oracle.
            for plane in (0, 1):
                await bench.kvst(
                    plane, 3, 2, 15 + 16 * channel, 19,
                    src=512, stride=5, B0=4096,
                )
                await bench.kvst(
                    plane, 2, 2, 16 * channel, 16,
                    src=1024, stride=0, B0=8192,
                )
            await ldst(True, (channel + 1) % bench.C)

    bench.port.stop()
    bench.mem.stop()
    bench.sim.close()
    Path(os.environ["TSR_SHARE_CYCLES"]).write_text(json.dumps(cycles, indent=2) + "\n")


def test_response_handover(tmp_path):
    cfg = dict(
        num_ch=4, t_max=8, k_max=32768, spm_bytes=4 << 20, ch_bytes=16 << 30,
        spm_rd_lat=4, addr_beat_w=28, rd_out=16, n_ops=0,
    )
    sources = rtl_sources() + [HERE / "hdl/tsr_tb_at_dma.sv"]
    baseline = os.environ.get("TSR_DMA_BASELINE")
    if baseline:
        sources = [
            Path(baseline) / source.name
            if source.name in ("tsr_dma.sv", "tsr_dma_kvst.sv") else source
            for source in sources
        ]
    params = dict(
        NUM_CH=4, T_MAX=8, K_MAX=32768, SPM_BYTES=4 << 20,
        CH_BYTES="64'd17179869184", SPM_RD_LAT=4, ADDR_BEAT_W=28, RD_OUTSTANDING=16,
    )
    cycle_path = os.environ.get("TSR_SHARE_CYCLES", str(tmp_path / "cycles.json"))
    run(
        "tsr_tb_at_dma", __name__, parameters=params, sources=sources,
        name="dma_response_share_" + ("baseline" if baseline else "candidate"),
        extra_env={"TSR_KVST_CFG": json.dumps(cfg), "TSR_SHARE_CYCLES": cycle_path},
    )
