"""Replay a simulator-captured graph through the real core with K4 locality."""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import cocotb
import pytest
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, Timer

from common import tables as TT
from common.mem_model import MemModel
from common.runner import run
from test_core import C, Host


@cocotb.test(timeout_time=4, timeout_unit="ms")
async def locality_graph(dut):
    trace = Path(os.environ["TB_LOCAL_TRACE"])
    info = json.loads((trace / "meta.json").read_text())
    cfg = info["config"]
    elf = trace / info["image"]["file"]
    local = int(os.environ["TB_LOCAL_ENABLE"])
    dut.csr_valid.value = 0
    dut.csr_we.value = 0
    dut.csr_addr.value = 0
    dut.csr_wdata.value = 0
    dut.plat_err_i.value = 0
    dut.rst_n.value = 0
    dut.rst_rv_n.value = 0
    cocotb.start_soon(Clock(dut.clk, 4000, unit="ps").start())
    await Timer(181, unit="ps")
    cocotb.start_soon(Clock(dut.clk_rv, 6000, unit="ps").start())
    mem = MemModel(dut, num_ch=4, addr_w=cfg["addr_beat_w"], seed=int(os.environ["TSR_SEED"]),
                   limit_beats=cfg["ch_bytes"] // 64, rd_outstanding=16)
    mem.start()
    await ClockCycles(dut.clk, 30)
    dut.rst_n.value = 1
    dut.rst_rv_n.value = 1
    await ClockCycles(dut.clk, 40)
    host = Host(dut)
    caps = await host.rd(TT.CSR_CAPS)
    local_caps = TT.CAPS_ATT_LOCAL | TT.CAPS_MVM_LOCAL_SCALE | TT.CAPS_MVM_LOCAL_SOFTCAP
    assert caps & local_caps == (local_caps if local else 0)

    ent = struct.unpack_from("<I", elf.read_bytes(), 24)[0]
    await host.wr(TT.CSR_RV_BOOT, ent - TT.BOOT_ENTRY_OFF)
    await host.wr(TT.CSR_CTRL, TT.CTRL_RV_RUN)
    dut._log.info("RV_RUN posted for %s", elf.name)
    lcq_rd = 0

    async def wait_record(kind, seq=None):
        nonlocal lcq_rd
        for _ in range(4000):
            await ClockCycles(dut.clk, 100)
            wr = await host.rd(TT.CSR_LCQ_WR)
            assert 0 <= wr - lcq_rd <= TT.LCQ_DEPTH, (wr, lcq_rd)
            match = None
            for k in range(lcq_rd, wr):
                rec = [await host.rd(TT.CSR_LCQ_REC(k % TT.LCQ_DEPTH, j)) for j in range(8)]
                if rec[0] & 0xff == kind and (seq is None or rec[1] == seq):
                    match = rec
            if wr != lcq_rd:
                lcq_rd = wr
                await host.wr(TT.CSR_LCQ_RD, wr)
            if match is not None:
                return match
            assert (await host.rd(TT.CSR_RT_STATE)) & 0xff != TT.RT_STATE_PHASE_TRAPPED
        raise AssertionError(f"no completion type {kind}, status {(await host.rd(TT.CSR_STATUS)):#x}")

    await wait_record(TT.LCQ_R0_TYPE_HELLO)
    dut._log.info("graph firmware booted")
    saw_attention = False
    saw_scale = False
    vu = dut.u_dp.g_vu.u_vu
    if local:
        mvm = dut.u_dp.g_mvm.u_mvm

        async def monitor():
            nonlocal saw_attention, saw_scale
            while True:
                await ClockCycles(dut.clk, 1)
                saw_attention |= bool(vu.local_busy.value)
                saw_scale |= bool(vu.scale_locked.value)
                if mvm.local_pv.value:
                    assert not mvm.spm_rd_req_valid.value, "local PV read a central P line"

        cocotb.start_soon(monitor())

    async def replay(work_trace, work, abort=False, restarted=False):
        nonlocal saw_attention, saw_scale
        saw_attention = saw_scale = False
        for region in work["expect"]:
            mem.load(region["ch"], region["offset"], bytes(region["size"]))
        for region in work["init"]:
            data = (work_trace / region["file"]).read_bytes()
            mem.load(region["ch"], region["offset"], data)
        mem.clear_tracking()
        launch = work["launch"]
        seq = await host.rd(TT.CSR_LQ_SEQ)
        await host.wr(TT.CSR_LQ_W(0), TT.DESC_TYPE_LAUNCH | (launch["index"] << TT.DESC_KERNEL_LSB))
        for i, word in enumerate(launch["args"]):
            await host.wr(TT.CSR_LQ_W(i + 1), word)
        await host.wr(TT.CSR_LQ_PUSH, 1)
        if abort:
            # The high four bits of each typed lane request are lane valid.
            # Wait for a request rather than just an acquired reservation.
            half = len(vu.bank_req) // 2
            valid_mask = (15 << (half - 4)) | (15 << (2 * half - 4))
            for _ in range(100_000):
                await ClockCycles(dut.clk, 1)
                if int(vu.bank_req.value) & valid_mask:
                    break
            else:
                raise AssertionError("no local lane work to abort")
            await host.wr(TT.CSR_CTRL, TT.CTRL_RV_RUN | TT.CTRL_QABORT)
        done = await wait_record(TT.LCQ_R0_TYPE_DONE, seq)
        status = (done[0] & TT.LCQ_R0_STATUS) >> TT.LCQ_R0_STATUS_LSB
        if abort:
            assert status == TT.LCQ_R0_STATUS_ABORTED, [hex(w) for w in done]
            assert await host.rd(TT.CSR_STATUS) & TT.STATUS_QIDLE
            assert not vu.local_busy.value and not vu.scale_locked.value
            mem.check(idle=True)
            dut._log.info("graph %s local work drained after QABORT", work_trace.name)
            return

        assert done[2] == work["done"]["ret"] and status == work["done"]["status"], [hex(w) for w in done]
        if local:
            pushes = (work_trace / work["ops"]["file"]).read_bytes()
            words0 = [words[0] for words in struct.iter_unpack("<12I", pushes) if words[-1] == 0]
            want_att = any(w & 0xff == 0x21 and w & 0x30000000 for w in words0)
            want_epi = any(w & 0xff == 0x20 and w & (TT.MVM_LOCAL_SCALE | TT.MVM_LOCAL_SOFTCAP) for w in words0)
            assert saw_attention == want_att
            assert saw_scale == want_epi
        for region in work["expect"]:
            expect = (work_trace / region["file"]).read_bytes()
            got = mem.read_bytes(region["ch"], region["offset"], len(expect))
            assert got == expect, (region, next(i for i in range(len(got)) if got[i] != expect[i]))
        mem.check(idle=True)
        dut._log.info("graph %s local=%d firmware_cycles=%d restarted=%d",
                      work_trace.name, local, done[7], restarted)

    await replay(trace, info)
    if local and os.environ.get("TB_LOCAL_ABORT") == "1":
        await replay(trace, info, abort=True)
        await replay(trace, info, restarted=True)
    for item in filter(None, os.environ.get("TB_LOCAL_MORE", "").split(os.pathsep)):
        other_trace = Path(item)
        other = json.loads((other_trace / "meta.json").read_text())
        assert other["image"]["id"] == info["image"]["id"]
        await replay(other_trace, other)


def tcm_preload(directory, image):
    """Write sparse Verilator readmemh files from ELF PT_LOAD segments."""
    files = {"itcm": [], "dtcm": []}
    for addr, seg in C.ctrl_tb.elf_segments(image):
        base = 0x100000 if addr >= 0x100000 else 0
        words = files["dtcm" if base else "itcm"]
        words.append(f"@{(addr - base) // 4:x}")
        words.extend(f"{word:08x}" for (word,) in struct.iter_unpack("<I", seg))
    args = []
    for name, words in files.items():
        path = directory / f"rtl_preload_{name}.hex"
        path.write_text("\n".join(words) + "\n")
        args.append(f"+tb_{name}={path}")
    return args


@pytest.mark.parametrize("local", [0, 1])
def test_k4_locality_graph(local, tmp_path):
    """TSR_LOCAL_TRACE points to a trace captured with the candidate simulator and firmware."""
    value = os.environ.get("TSR_LOCAL_TRACE")
    if not value:
        pytest.skip("set TSR_LOCAL_TRACE to a K4 T4 candidate graph trace")
    trace = Path(value if local else os.environ.get("TSR_LOCAL_BASELINE_TRACE", value)).resolve()
    info = json.loads((trace / "meta.json").read_text())
    cfg = info["config"]
    preload = tcm_preload(tmp_path, trace / info["image"]["file"])
    sv = Path(__file__).resolve().parent / "hdl" / "tsr_tb_local_tcm.sv"
    assert cfg["num_ch"] == 4 and cfg["t_max"] == 4
    local_enable = 0 if os.environ.get("TSR_LOCAL_NARROW_ONLY") == "1" else local
    params = dict(NUM_CH=4, T_MAX=4, K_MAX=cfg["k_max"], SPM_BYTES=cfg["spm_bytes"],
                  CH_BYTES=f"64'd{cfg['ch_bytes']}", ADDR_BEAT_W=cfg["addr_beat_w"],
                  LOCAL_INTERMEDIATES=local_enable, LOCAL_SCALE=local_enable, LOCAL_SOFTCAP=local_enable, MVM_GATHER_PAIR=1, MVM_GATHER_FLITS=1 + local,
                  HAS_TOPK=int(bool(cfg["caps"] & TT.CAPS_TOPK)),
                  HAS_DQROW=int(bool(cfg["caps"] & TT.CAPS_DQROW)),
                  DDR_LOCAL_PIPE=1, RV_LOCAL_TRANSPORT=1, RST_PIPE=3, SPM_RD_LAT=4,
                  MVM_BC_PIPE=2, MVM_RES_PIPE=2, MVM_MAC_LAT=9,
                  VU_FRONT_LANES=8, VU_EXP_LANES=8, VU_BACK_LANES=4, VU_LANE_PIPE=2)
    extra = os.environ.get("TSR_LOCAL_TRACE_MORE", "")
    if not local:
        extra = os.environ.get("TSR_LOCAL_BASELINE_MORE", extra)
    extra = os.pathsep.join(str(Path(item).resolve()) for item in filter(None, extra.split(os.pathsep)))
    run("tsr_core", __name__, parameters=params, name=f"core_intermediate_local_{local}",
        extra_sources=[sv], plusargs=preload,
        testcase=["locality_graph"], extra_env={"TB_LOCAL_TRACE": str(trace), "TB_LOCAL_ENABLE": str(local_enable),
                   "TB_LOCAL_MORE": extra, "TB_LOCAL_ABORT": os.environ.get("TSR_LOCAL_ABORT", "0")},
        timescale=("1ps", "1ps"))


def test_k4_full_graph(tmp_path):
    """TSR_K4_TRACE points to a trace of a K4 T4 graph whose records use no local kind (a legacy attention graph or a
    dense graph, captured with the simulator and the production firmware). The core is the packaged full-feature image
    (tsr_u200_k4, gen_tops.py): T_MAX 4, VU 8 / 8 / 4 behind two lane-pipe stages, attention, paired two-flit gather,
    the DDR0 / DDR3 boundaries and the RV transport, with the local-intermediate banks and epilogues off (CAPS 0x21).
    TSR_K4_TRACE_MORE adds further graphs of the same configuration, separated by the path separator."""
    value = os.environ.get("TSR_K4_TRACE")
    if not value:
        pytest.skip("set TSR_K4_TRACE to a K4 T4 graph trace without local kinds")
    trace = Path(value).resolve()
    info = json.loads((trace / "meta.json").read_text())
    cfg = info["config"]
    preload = tcm_preload(tmp_path, trace / info["image"]["file"])
    sv = Path(__file__).resolve().parent / "hdl" / "tsr_tb_local_tcm.sv"
    assert cfg["num_ch"] == 4 and cfg["t_max"] == 4
    params = dict(NUM_CH=4, T_MAX=4, K_MAX=cfg["k_max"], SPM_BYTES=cfg["spm_bytes"],
                  CH_BYTES=f"64'd{cfg['ch_bytes']}", ADDR_BEAT_W=cfg["addr_beat_w"],
                  MVM_GATHER_PAIR=1, MVM_GATHER_FLITS=2, DDR_LOCAL_PIPE=1, RV_LOCAL_TRANSPORT=1, XSLR_REG=1, RST_PIPE=3,
                  SPM_RD_LAT=4, MVM_BC_PIPE=2, MVM_RES_PIPE=2, MVM_MAC_LAT=9,
                  VU_FRONT_LANES=8, VU_EXP_LANES=8, VU_BACK_LANES=4, VU_LANE_PIPE=2)
    extra = os.pathsep.join(str(Path(item).resolve())
                            for item in filter(None, os.environ.get("TSR_K4_TRACE_MORE", "").split(os.pathsep)))
    run("tsr_core", __name__, parameters=params, name="core_k4_full_graph",
        extra_sources=[sv], plusargs=preload,
        testcase=["locality_graph"], extra_env={"TB_LOCAL_TRACE": str(trace), "TB_LOCAL_ENABLE": "0",
                   "TB_LOCAL_MORE": extra, "TB_LOCAL_ABORT": "0"},
        timescale=("1ps", "1ps"))
