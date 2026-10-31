"""tb/core — the integrated core (W4-A, docs/design/README.md §10 items 1-3): tsr_core (tsr_ctrl with
Ibex + tsr_core_dp) on Verilator through cocotb, against the simulator running the same firmware
(docs/design/06-software-verification.md "SW" §10.3; CC §15.4's full-system bench in its smallest form).

core_smoke  the host loads tb/core/fw/fw_core.c (crt0 / link.ld of tb/ctrl) and its argument block
            through the CSR port's loader (LOAD_ADDR / LOAD_DATA), checks every segment with VERIFY
            (LOAD_CRC against zlib's crc32), boots the hart (RV_BOOT 0, CTRL.RV_RUN) and waits for
            RT_STATE = DONE. The firmware, on the RTL Ibex, pushes a MUL_MAT-shaped stream through the
            control window, the issue FIFO crossing, tsr_issue / tsr_chk and the unit queues into the
            units: LD (background, x), QACT A32 and A256, MVM Q8_0 and Q4_K (T = 2, N 40 / 24, K 512,
            weights on both channels), ST of both results, FENCE, ST of the SPM image; joins on the last
            ticket, reads results through the SPM bridge and the core window, pushes log words, posts
            an LCQ record and mirrors a checksum in RT_INFO. tb/core/core_tb.py runs the same host
            sequence on the simulator (libtsr_sim.so, its built-in tsr_rv hart). Compared: the
            identification registers, the receive and VERIFY CRCs, every DRAM region the ops read or
            write (both results, the SPM image, the inputs untouched), no DRAM write outside the store
            targets, and the final CSR state (STATUS's schedule-independent bits, RT_*, LCQ / log
            pointers and contents, Q_DONE / Q_ISSUED / Q_ERR_CODE per unit, Q_PIDX, Q_FIFO).
            The memory model randomises latency, back-pressure and response interleaving.
Mutations (opt-in, TSR_CORE_MUTANTS=1, -m slow or -k mutants): test_core_mutants rebuilds the bench
with one wiring mutant of rtl/core/tsr_core.sv / tsr_core_dp.sv at a time (8); each must fail core_smoke.

Environment: TB_CORE_RV_PS (clk_rv period, default 6000 ps), TB_CORE_PS (core clock, 3620 ps), TSR_SEED,
TSR_SIM_LIB, TSR_CORE_MUT (internal: the mutant to build).
"""
from __future__ import annotations

import os
import shutil
import struct
import sys
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import core_tb as C  # noqa: E402
from common import tables as TT  # noqa: E402

try:
    import pytest
except ImportError:      # inside the simulator
    pytest = None

REPO = C.REPO
FW_OUT = REPO / "build" / "cocotb" / "core_fw"
M32 = 0xFFFFFFFF


# =====================================================================================================
# cocotb side
# =====================================================================================================
try:
    import cocotb
    from cocotb.clock import Clock
    from cocotb.triggers import ClockCycles, FallingEdge, RisingEdge, Timer
except ImportError:      # collected by pytest outside the simulator
    cocotb = None

if cocotb is not None:
    from common.mem_model import MemModel

    class Host:
        """The host CSR port of tsr_core (docs/rtl-m2.md §6 protocol, EX §5)."""

        def __init__(self, dut):
            self.d = dut

        async def wr(self, off, val):
            d = self.d
            d.csr_valid.value = 1
            d.csr_we.value = 1
            d.csr_addr.value = off & 0xFFF
            d.csr_wdata.value = val & M32
            while True:
                await FallingEdge(d.clk)
                if int(d.csr_ready.value):
                    break
            await RisingEdge(d.clk)
            d.csr_valid.value = 0

        async def rd(self, off):
            d = self.d
            d.csr_valid.value = 1
            d.csr_we.value = 0
            d.csr_addr.value = off & 0xFFF
            while True:
                await FallingEdge(d.clk)
                if int(d.csr_ready.value):
                    v = int(d.csr_rdata.value)
                    break
            await RisingEdge(d.clk)
            d.csr_valid.value = 0
            return v

    @cocotb.test()
    async def core_smoke(dut):
        seed = int(os.environ.get("TSR_SEED", "1"), 0)
        rv_ps = int(os.environ.get("TB_CORE_RV_PS", "6000"))
        core_ps = int(os.environ.get("TB_CORE_PS", "3620"))
        elf = Path(os.environ["TB_CORE_ELF"])
        sc = C.scenario(seed)
        sim = C.run_sim(sc, elf)

        dut.csr_valid.value = 0
        dut.csr_we.value = 0
        dut.csr_addr.value = 0
        dut.csr_wdata.value = 0
        dut.plat_err_i.value = 0
        dut.rst_n.value = 0
        dut.rst_rv_n.value = 0
        cocotb.start_soon(Clock(dut.clk, core_ps, unit="ps").start())
        await Timer((seed * 977) % rv_ps + 1, unit="ps")            # independent clocks, random phase
        cocotb.start_soon(Clock(dut.clk_rv, rv_ps, unit="ps").start())
        mem = MemModel(dut, num_ch=C.CFG["num_ch"], addr_w=C.CFG["addr_beat_w"], seed=seed,
                       limit_beats=C.CFG["ch_bytes"] // 64, rd_outstanding=C.CFG["rd_outstanding"])
        for ch, off, data in sc["writes"]:
            mem.load(ch, off, data)
        mem.clear_tracking()
        mem.start()
        await ClockCycles(dut.clk, 30)
        dut.rst_rv_n.value = 1
        dut.rst_n.value = 1
        await ClockCycles(dut.clk, 40)
        host = Host(dut)
        errs = []

        ids = {o: await host.rd(o) for o in C.ID_REGS}
        C.check_ids(ids, sim["ids"], errs)
        syms = C.ctrl_tb.elf_symbols(elf)
        crcs = []
        t0 = cocotb.utils.get_sim_time("ns")
        nwords = 0
        for addr, words in C.load_words(elf, syms["args"], sc["args"]):
            await host.wr(TT.CSR_LOAD_ADDR, addr)
            for w in words:
                await host.wr(TT.CSR_LOAD_DATA, w)
            nwords += len(words)
            crc_rx = await host.rd(TT.CSR_LOAD_CRC)
            await host.wr(TT.CSR_LOAD_ADDR, addr)
            await host.wr(TT.CSR_LOAD_COUNT, len(words))
            await host.wr(TT.CSR_LOAD_CTRL, 1)
            n = 0
            while (await host.rd(TT.CSR_LOAD_STATUS)) & 1:
                await ClockCycles(dut.clk, 64)
                n += 1
                assert n < 100000, "VERIFY did not finish"
            crcs.append((crc_rx, await host.rd(TT.CSR_LOAD_CRC),
                         zlib.crc32(struct.pack("<%dI" % len(words), *words))))
        if crcs != sim["crcs"]:
            errs.append("load CRCs (receive, VERIFY, zlib): RTL %s / simulator %s" % (crcs, sim["crcs"]))
        st = await host.rd(TT.CSR_STATUS)
        if (st & C.STATUS_MASK) != (sim["status_loaded"] & C.STATUS_MASK):
            errs.append("STATUS after the load: RTL %#x / simulator %#x" % (st, sim["status_loaded"]))
        dut._log.info("loaded %d words with VERIFY in %.1f us", nwords, (cocotb.utils.get_sim_time("ns") - t0) / 1e3)

        await host.wr(TT.CSR_RV_BOOT, 0)
        await host.wr(TT.CSR_CTRL, TT.CTRL_RV_RUN)
        t0 = cocotb.utils.get_sim_time("ns")
        cyc = 0
        while (await host.rd(TT.CSR_RT_STATE)) != C.RT_DONE:
            await ClockCycles(dut.clk, 500)
            cyc += 500
            if cyc > 600_000:                          # DONE comes after ~15 K cycles
                raise AssertionError("RT_STATE %#x after %d core cycles (RT_INFO %#x, STATUS %#x)" % (
                    await host.rd(TT.CSR_RT_STATE), cyc, await host.rd(TT.CSR_RT_INFO),
                    await host.rd(TT.CSR_STATUS)))
        t_run = (cocotb.utils.get_sim_time("ns") - t0) / 1e3
        n = 0
        while not ((await host.rd(TT.CSR_STATUS)) & TT.STATUS_QIDLE):
            await ClockCycles(dut.clk, 100)
            n += 1
            assert n < 10000, "QIDLE not reached"
        await ClockCycles(dut.clk, 3000)
        regs = {o: await host.rd(o) for o in C.final_regs()}
        C.compare_final(regs, sim["regs"], errs)

        for ch, off, size in sc["regions"]:
            got = mem.read_bytes(ch, off, size)
            want = sim["dram"][(ch, off)]
            if got != want:
                bad = [i // 64 for i in range(0, size, 64) if got[i:i + 64] != want[i:i + 64]]
                errs.append("DRAM ch%d [%#x, +%#x): %d beats differ (first beats %s)" % (
                    ch, off, size, len(bad), bad[:8]))
        # no DRAM write outside the three store targets
        allowed = {}
        for ch, off, size in sc["regions"][:3]:
            allowed.setdefault(ch, set()).update(range(off // 64, (off + size) // 64))
        for c in range(C.CFG["num_ch"]):
            stray = sorted(set(mem.written[c]) - allowed.get(c, set()))
            if stray:
                errs.append("DRAM ch%d: %d beats written outside the store targets (first %s)" % (c, len(stray), stray[:8]))
        mem.check(idle=True)
        dut._log.info("RTL: firmware DONE %.1f us after RV_RUN (~%d core cycles); simulator: %d instructions, "
                      "%d core cycles to DONE + idle; RT_INFO %#x, LCQ_WR %d, LOG_WR %d, Q_DONE %s",
                      t_run, cyc, sim["instret"], sim["cycles"], regs[TT.CSR_RT_INFO], regs[TT.CSR_LCQ_WR],
                      regs[TT.CSR_LOG_WR], [regs[TT.CSR_Q_DONE_BASE + 4 * u] for u in range(6)])
        assert not errs, "\n".join(errs[:40])


# =====================================================================================================
# pytest side
# =====================================================================================================
# Wiring mutants (CC §15.5 spot checks for the integration): (file, old, new, description).
MUTANTS = {
    "bridge_addr": ("tsr_core_dp.sv", ".req_waddr(spm_req_addr[SPM_WAW+1:2])", ".req_waddr(spm_req_addr[SPM_WAW:1])",
                    "SPM bridge word address off by one bit"),
    "cwin_data": ("tsr_core_dp.sv", ".cwin_req_valid, .cwin_req_off, .cwin_rsp_valid, .cwin_rsp_rdata,",
                  ".cwin_req_valid, .cwin_req_off(12'h000), .cwin_rsp_valid, .cwin_rsp_rdata,",
                  "core-window offset not passed to tsr_csr"),
    "lcq_words": ("tsr_core.sv", ".lq_req_wdata_o(lq_req_wdata), .lq_req_lcq_w_o(lq_req_lcq_w),",
                  ".lq_req_wdata_o(lq_req_wdata), .lq_req_lcq_w_o({lq_req_lcq_w[0], lq_req_lcq_w[7:1]}),",
                  "LCQ record words rotated"),
    "rt_swap": ("tsr_core.sv", ".rt_state_o(rt_state), .rt_info_o(rt_info),",
                ".rt_state_o(rt_info), .rt_info_o(rt_state),", "RT_STATE / RT_INFO mirrors swapped"),
    "qdepth_mvm": ("tsr_core.sv", "5'd0, 5'd0, 5'(QD_MVM_E), 5'(QD_QACT), 5'(QD_DMA)}",
                   "5'd0, 5'd0, 5'd0, 5'(QD_QACT), 5'(QD_DMA)}", "tsr_ctrl told the MVM queue is absent"),
    "ld_rdata": ("tsr_core_dp.sv", ".ld_rsp_valid, .ld_rsp_rdata, .ld_rsp_err, .ld_rsp_timeout,",
                 ".ld_rsp_valid, .ld_rsp_rdata(32'h0), .ld_rsp_err, .ld_rsp_timeout,",
                 "loader read data lost (VERIFY)"),
    "bridge_line": ("tsr_core.sv", ".spm_rsp_valid_i(spm_rsp_valid), .spm_rsp_line_i(spm_rsp_line),",
                    ".spm_rsp_valid_i(spm_rsp_valid), .spm_rsp_line_i({spm_rsp_line[479:0], spm_rsp_line[511:480]}),",
                    "SPM bridge response line rotated by one word"),
    "dseq_sseq": ("tsr_core.sv", ".s_seq_i(s_seq), .d_seq_i(d_seq),", ".s_seq_i(d_seq), .d_seq_i(s_seq),",
                  "S[u] / D[u] swapped at the crossing (DONE_SEQ shows accepts)"),
}


def _sources(mut=None):
    from common.runner import rtl_sources
    srcs = rtl_sources()
    if not mut:
        return srcs
    fname, old, new, _ = MUTANTS[mut]
    d = REPO / "build" / "cocotb" / "core_mut" / mut
    d.mkdir(parents=True, exist_ok=True)
    out = []
    for s in srcs:
        if s.name == fname:
            txt = s.read_text()
            assert txt.count(old) == 1, (mut, fname, txt.count(old))
            p = d / fname
            new_txt = txt.replace(old, new)
            if not p.exists() or p.read_text() != new_txt:
                p.write_text(new_txt)
            s = p
        out.append(s)
    return out


# Runs of core_smoke (one build): (seed, clk_rv period in ps; the core clock is 3620 ps): F_RV / F_CORE
# 0.6 (the planning ratio), 1.0 and 1.67 (CC §15.4's ratios), each with its own data and memory
# schedule.
RUNS = {"s1_r0.6": (1, 6000), "s2_r1.0": (2, 3620), "s3_r1.67": (3, 2170)}


def _run(mut=None, seed=1, rv_ps=6000):
    from common.runner import run
    elf = C.build_fw(FW_OUT)
    name = "core" if not mut else "core_mut_" + mut
    run("tsr_core", __name__, parameters=dict(C.PARAMS), sources=_sources(mut), name=name, seed=seed,
        testcase=["core_smoke"], extra_env={"TB_CORE_ELF": str(elf), "TB_CORE_RV_PS": str(rv_ps)},
        timescale=("1ps", "1ps"))


def test_core(cfg):
    seed, rv_ps = RUNS[cfg]
    _run(seed=seed + int(os.environ.get("TSR_SEED", "1"), 0) - 1, rv_ps=rv_ps)   # --tsr-seed shifts all


def _mutants_requested(request) -> bool:
    cfg = request.config
    return (os.environ.get("TSR_CORE_MUTANTS") == "1" or "mutants" in (cfg.getoption("keyword", "") or "")
            or "slow" in (cfg.getoption("markexpr", "") or "")
            or any("test_core_mutants" in str(a) for a in cfg.args))


def test_core_mutants(request):
    """Opt-in (TSR_CORE_MUTANTS=1, -m slow, -k mutants or its node id; about 20 min with
    TSR_BUILD_JOBS=2): every wiring mutant must fail core_smoke (seed 1, ratio 0.6)."""
    if not _mutants_requested(request):
        pytest.skip("opt-in: TSR_CORE_MUTANTS=1, -m slow or -k mutants")
    sel = os.environ.get("TSR_CORE_MUT_ONLY")
    todo = [m for m in MUTANTS if not sel or m in sel.split(",")]
    for m in todo:
        _sources(m)                           # every pattern matches exactly once (else: an error, not a kill)
    survived = []
    for m in todo:
        try:
            _run(m)
        except BaseException as e:            # cocotb failure under pytest: SystemExit / build error
            if isinstance(e, KeyboardInterrupt):
                raise
            print("mutant %-12s killed (%s)" % (m, type(e).__name__))
            continue
        survived.append(m)
        print("mutant %-12s SURVIVED" % m)
    assert not survived, "surviving mutants: %s" % survived


if pytest is not None:
    test_core = pytest.mark.parametrize("cfg", list(RUNS))(test_core)
    test_core_mutants = pytest.mark.slow(test_core_mutants)
