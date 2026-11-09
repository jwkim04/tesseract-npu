"""tb/core first_run — the first launch after configuration on the packaged full-feature core.

The card showed `selftest` id 5 failing in two lines of the causal ATTQK scores on the first run after an image load and
passing on every later run (docs/platform-u200.md 14.5). The cause was a flop whose reset value depended on a setup flop
that is zero after configuration; the benches start every flop at a random value and never saw it. This bench starts
the core at zero (rand_reset 0), as the FPGA does.

It boots the production firmware on the RTL Ibex of the packaged K4 core (Xilinx technology wrappers, T_MAX 4, paired
two-flit gather, XSLR_REG, RV transport), optionally replays kind-ops traces through the `replay` kernel (TB_FR_PRIME),
then launches a captured `selftest` launch trace (TB_FR_TRACE: tsr-selftest --device sim with TSR_TRACE) TB_FR_REPEAT
times and compares every expect region after each launch. It reports the differing 64-byte lines of each launch
instead of stopping at the first. TB_FR_MON=1 logs, during the first launch, the ATTQK and ATTPV jobs of every cluster
and the SPM writes of the MVM unit (TB_FR_MON_LINES=lo,hi limits the lines); TB_FR_PVSTATE=1 logs the output stage of the
ATTPV units before the first launch.
"""
from __future__ import annotations

import json
import os
import struct
import subprocess
from pathlib import Path

import pytest

try:
    import cocotb
    from cocotb.clock import Clock
    from cocotb.triggers import ClockCycles, FallingEdge, Timer
except ImportError:      # collected by pytest outside the simulator
    cocotb = None

from common import tables as TT
from common.runner import REPO, RTL, mutated_sources, rtl_sources, run
from test_core_local import REPLAY_KERNEL, Host, ops_replay_args, tcm_preload

if cocotb is not None:
    from common.mem_model import MemModel

    @cocotb.test(timeout_time=400, timeout_unit="ms")
    async def first_run(dut):
        trace = Path(os.environ["TB_FR_TRACE"])
        info = json.loads((trace / "meta.json").read_text())
        cfg = info["config"]
        repeat = int(os.environ.get("TB_FR_REPEAT", "3"))
        primes = [Path(p) for p in filter(None, os.environ.get("TB_FR_PRIME", "").split(os.pathsep))]
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
        elf = trace / info["image"]["file"]
        ent = struct.unpack_from("<I", elf.read_bytes(), 24)[0]
        await host.wr(TT.CSR_RV_BOOT, ent - TT.BOOT_ENTRY_OFF)
        await host.wr(TT.CSR_CTRL, TT.CTRL_RV_RUN)
        lcq_rd = 0

        async def wait_record(kind, seq=None):
            nonlocal lcq_rd
            for _ in range(40000):
                await ClockCycles(dut.clk, 50)
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
        dut._log.info("firmware booted")

        async def launch(kernel, args):
            seq = await host.rd(TT.CSR_LQ_SEQ)
            await host.wr(TT.CSR_LQ_W(0), TT.DESC_TYPE_LAUNCH | (kernel << TT.DESC_KERNEL_LSB))
            for i, word in enumerate(args):
                await host.wr(TT.CSR_LQ_W(i + 1), word)
            await host.wr(TT.CSR_LQ_PUSH, 1)
            return await wait_record(TT.LCQ_R0_TYPE_DONE, seq)

        for p in primes:
            meta = json.loads((p / "meta.json").read_text())
            for region in meta["expect"]:
                mem.load(region["ch"], region["offset"], bytes(~b & 0xff for b in (p / region["file"]).read_bytes()))
            for region in meta["init"]:
                mem.load(region["ch"], region["offset"], (p / region["file"]).read_bytes())
            push = (p / meta["ops"]["file"]).read_bytes()
            push += bytes(-len(push) % 64)
            mem.load(meta["ops"]["load"]["ch"], meta["ops"]["load"]["offset"], push)
            mem.clear_tracking()
            done = await launch(REPLAY_KERNEL, ops_replay_args(meta, cfg["spm_bytes"] // 64))
            got = [(done[0] & TT.LCQ_R0_STATUS) >> TT.LCQ_R0_STATUS_LSB, done[2]]
            bad = []
            for region in meta["expect"]:
                expect = (p / region["file"]).read_bytes()
                if mem.read_bytes(region["ch"], region["offset"], len(expect)) != expect:
                    bad.append(region["file"])
            dut._log.info("prime %s: status/ret %s, differing regions %s", p.name, got, bad)

        mon_on = False
        if os.environ.get("TB_FR_PVSTATE") == "1":
            for c in range(4):
                pv = dut.u_dp.g_mvm.u_mvm.g_cl[c].u_cl.g_att.u_pv
                dut._log.info("PVSTATE cluster %d before any op: ob_zero_q %d o_st_q %d jv_q %d ng %d", c,
                              int(pv.ob_zero_q.value), int(pv.o_st_q.value), int(pv.jv_q.value), int(pv.ng.value))
        if os.environ.get("TB_FR_MON") == "1":
            mvm = dut.u_dp.g_mvm.u_mvm
            lo_l, hi_l = [int(x, 0) for x in os.environ.get("TB_FR_MON_LINES", "509,545").split(",")]

            async def monitor():
                cyc = 0
                spm_aw = 16
                while True:
                    await FallingEdge(dut.clk)
                    cyc += 1
                    if not mon_on:
                        continue
                    if int(mvm.spm_wr_valid.value) and int(mvm.spm_wr_ready.value):
                        line = int(mvm.spm_wr_line.value)
                        if lo_l <= line < hi_l:
                            dut._log.info("MON cyc %d SPMWR line %d be %#x d0 %#010x", cyc, line, int(mvm.spm_wr_be.value),
                                          int(mvm.spm_wr_data.value) & 0xffffffff)
                    for c in range(4):
                        ga = mvm.g_cl[c].u_cl.g_att
                        if int(ga.p_jv.value) and int(ga.p_jr.value):
                            j = int(ga.p_j.value)
                            dut._log.info("MON cyc %d PV%d job line %d own %#x tail %#x last %d", cyc, c,
                                          (j >> (16 + 16 + 512)) & ((1 << spm_aw) - 1), (j >> (16 + 512)) & 0xffff,
                                          (j >> 512) & 0xffff, j >> (spm_aw + 16 + 16 + 512))
                        aq = ga.u_aqk
                        if int(aq.job_valid.value) and int(aq.job_ready.value):
                            j = int(aq.job.value)
                            line = (j >> (16 + 16 + 512)) & ((1 << spm_aw) - 1)
                            dut._log.info("MON cyc %d AQK%d job line %d e %d last %d", cyc, c, line, (j >> 512) & 0x7f,
                                          j >> (spm_aw + 16 + 16 + 512))

            cocotb.start_soon(monitor())

        results = []
        for n in range(repeat):
            mon_on = (n == 0)
            for region in info["expect"]:
                mem.load(region["ch"], region["offset"], bytes(region["size"]))
            for region in info["init"]:
                mem.load(region["ch"], region["offset"], (trace / region["file"]).read_bytes())
            mem.clear_tracking()
            launch_info = info["launch"]
            done = await launch(launch_info["index"], launch_info["args"])
            status = (done[0] & TT.LCQ_R0_STATUS) >> TT.LCQ_R0_STATUS_LSB
            diffs = []
            for region in info["expect"]:
                expect = (trace / region["file"]).read_bytes()
                got = mem.read_bytes(region["ch"], region["offset"], len(expect))
                if got != expect:
                    base = region["offset"] // 64
                    diffs += [(region["ch"], base + i // 64) for i in range(0, len(expect), 64)
                              if got[i:i + 64] != expect[i:i + 64]]
            ok = status == info["done"]["status"] and done[2] == info["done"]["ret"] and not diffs
            dut._log.info("launch %d: status %d ret %#x, %d differing lines %s -> %s", n, status, done[2], len(diffs),
                          [(c, hex(b)) for c, b in diffs[:12]], "ok" if ok else "DIFFERS")
            results.append((ok, diffs))
        want = os.environ.get("TB_FR_EXPECT", "all_ok")
        if want == "all_ok":
            assert all(ok for ok, _ in results), [(ok, len(d)) for ok, d in results]
        else:                                  # "first_differs": the card's behaviour
            assert not results[0][0] and all(ok for ok, _ in results[1:]), [(ok, len(d)) for ok, d in results]


def _params(cfg):
    return dict(NUM_CH=4, T_MAX=4, K_MAX=cfg["k_max"], SPM_BYTES=cfg["spm_bytes"],
                CH_BYTES=f"64'd{cfg['ch_bytes']}", ADDR_BEAT_W=cfg["addr_beat_w"],
                MVM_GATHER_PAIR=1, MVM_GATHER_FLITS=2, DDR_LOCAL_PIPE=1, RV_LOCAL_TRANSPORT=1, XSLR_REG=1, RST_PIPE=3,
                SPM_RD_LAT=4, MVM_BC_PIPE=2, MVM_RES_PIPE=2, MVM_MAC_LAT=9,
                VU_FRONT_LANES=8, VU_EXP_LANES=8, VU_BACK_LANES=4, VU_LANE_PIPE=2,
                FAST_CLK=int(os.environ.get("TSR_FR_FAST", "0")))


def selftest_trace(directory):
    """The launch trace of `selftest` id 5 (the attention ops) from the simulator configured as the card image (T_MAX 4,
    no local-intermediate capabilities): tsr-selftest --device sim with TSR_TRACE. Skips the test without the tool."""
    tool = REPO / "build" / "dev" / "bin" / "tsr-selftest"
    image = REPO / "build" / "dev" / "fw" / "tsr-fw.elf"
    if not tool.exists() or not image.exists():
        pytest.skip("build the host tools and the image first: cmake --build build/dev --target tsr-selftest tsr_fw")
    env = dict(os.environ, TSR_TRACE=str(directory), TSR_SIM_TMAX="4", TSR_SIM_ATT_LOCAL="0", TSR_SIM_MVM_LOCAL_SCALE="0",
               TSR_SIM_MVM_LOCAL_SOFTCAP="0", TSR_SIM_THREADS="1")
    env.pop("TSR_SIM_SCHED", None)
    subprocess.run([str(tool), "--device", "sim", "--image", str(image), "--no-mem", "--no-shadow", "--rate-n", "4",
                    "--ids", "5"], env=env, check=True, stdout=subprocess.DEVNULL)
    found = sorted(m.parent for m in Path(directory).rglob("meta.json"))
    assert len(found) == 1, found
    return found[0]


# the output stage of tsr_mvm_pv before the fix: reset to `ng == 0`
PV_FIX = "ob_zero_q  <= rst_n & (ng == 13'd0);"
PV_OLD = "ob_zero_q  <= (ng == 13'd0);"


def _first_run(tmp_path, mutant):
    """The first launch after configuration, with every flop and RAM word at zero as on the FPGA (rand_reset 0), on the
    packaged full-feature core (Xilinx wrappers, paired two-flit gather, XSLR_REG, production firmware on the RTL Ibex).
    `selftest` id 5 runs three times; each run must equal the simulator's output bit for bit. With the output stage of
    tsr_mvm_pv reset to `ng == 0` the first run of the card image (and of this core at zero state) differed in two
    lines of the causal ATTQK scores: the zero lines of an ATTPV that had not started were offered to the gather and
    written over them (docs/platform-u200.md 14.5).
    TSR_FR_TRACE: another launch trace; TSR_FR_PRIME: ops traces replayed through the `replay` kernel first;
    TSR_FR_REPEAT: launches (default 3); TSR_FR_EXPECT=first_differs: the card's picture; TSR_RAND_RESET: the initial
    values (default 0); TB_FR_MON=1 in the environment logs the ATTQK / ATTPV jobs and the SPM writes of the MVM unit."""
    value = os.environ.get("TSR_FR_TRACE")
    trace = Path(value).resolve() if value else selftest_trace(tmp_path / "trace")
    info = json.loads((trace / "meta.json").read_text())
    primes = []
    for item in filter(None, os.environ.get("TSR_FR_PRIME", "").split(os.pathsep)):
        path = Path(item).resolve()
        primes += [path] if (path / "meta.json").exists() else sorted(m.parent for m in path.rglob("meta.json"))
    preload = tcm_preload(tmp_path, trace / info["image"]["file"])
    sv = Path(__file__).resolve().parent / "hdl" / "tsr_tb_local_tcm.sv"
    sources = rtl_sources(RTL / "filelist_xilinx.f")
    expect = os.environ.get("TSR_FR_EXPECT", "all_ok")
    if mutant:
        sources = mutated_sources(sources, "tsr_mvm_pv.sv", PV_FIX, PV_OLD, "first_run_pv_reset")
    run("tsr_core", __name__, parameters=_params(info["config"]), name="core_k4_full_ops_mut" if mutant else "core_k4_full_ops",
        sources=sources, extra_sources=[sv], plusargs=preload,
        testcase=["first_run"], rand_reset=int(os.environ.get("TSR_RAND_RESET", "0")),
        extra_env={"TB_FR_TRACE": str(trace), "TB_FR_PRIME": os.pathsep.join(str(p) for p in primes),
                   "TB_FR_REPEAT": os.environ.get("TSR_FR_REPEAT", "3"), "TB_FR_EXPECT": expect},
        timescale=("1ps", "1ps"))


def test_k4_selftest_first_run(tmp_path):
    _first_run(tmp_path, False)


def test_k4_selftest_first_run_fast(tmp_path, monkeypatch):
    """The same first launch at zero state with FAST_CLK 1 (docs/platform-u200.md §8.10): the Xilinx wrappers with the DSP
    lanes' pv_en copies, the three-stage emitters, the PV convert pipeline and the KVST placer stage."""
    monkeypatch.setenv("TSR_FR_FAST", "1")
    _first_run(tmp_path, False)


@pytest.mark.slow
def test_k4_selftest_first_run_mutant(tmp_path):
    """The old reset value of the output stage (PV_OLD) must fail the same run: the assertion of tsr_mvm_pv stops the
    simulation right after reset (an output job with no ATTPV running), and without it the first launch differs in the
    two score lines (TSR_FR_EXPECT=first_differs shows that picture on a build without the assertion)."""
    with pytest.raises(SystemExit):
        _first_run(tmp_path, True)
