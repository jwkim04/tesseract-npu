"""tb/ctrl — the RISC-V control subsystem (rtl/ctrl: tsr_ctrl_rv + tsr_ctrl_cdc, W4 item 3) against
the simulator (CC §15.2, §15.4; docs/design/06-software-verification.md SW §10.3).

Bench: tb/ctrl/sv/tb_ctrl.sv (tsr_ctrl with RVFI + the core-domain stub tb_ctrl_core.sv) on
Verilator through cocotb; the simulator side is tb/ctrl/rvsim.py (tsr_rv on a firmware bus bound to
a tsr_sim, and the simulator's DM model, through libtsr_sim.so).

Tests (one simulator process each, so each starts from a hardware reset):
  lockstep    the firmware tb/ctrl/fw (fw_ctrl.c, mode 0) is loaded and read back through the load
              port, booted, and run to its end on the RTL; then on tsr_rv with the simulator's
              control window, in lockstep with the RTL's RVFI records (timing-dependent loads take
              the RTL's value, CC §15.4). Compared: every retired instruction (pc, instruction, trap,
              next pc, rd and its value, memory address, store data), every issue-FIFO entry (all
              387 bits' fields, CC §7.2), the LCQ records posted, the log words, the RT_* mirrors,
              the SPM lines written and the final DTCM image.
  dmi         DMI smoke against the simulator's DM model (tsr_dm.c): activate, halt, read x5 and
              dcsr with abstract commands, the halted / running status, resume; SBA reads of DTCM
              and of the control window, SBA writes refused for the control window (CC-33). Every
              DMI read value compared with the model's for the same sequence; RV_HALTED from the
              fetch address (CC-29) cross-checked against dmstatus.allhalted; then (RTL only) DB's
              halt request toggle (one halt, cause haltreq, pend cleared, no re-halt) and
              halt-on-boot (dpc = RV_BOOT + 0x80).
  halt_reset_pending  normal reset release discards an old DB halt request, a fresh request still
              halts once, and cancel while booting clears pending state (DB §3.3, §3.8).
  push_wait   the push safety net (CC §7.3): with the dispatcher stopped a push waits, faults after
              ISSUE_WAIT_MAX clk_rv cycles with ISSUE_STATUS.PUSH_TIMEOUT and the core's sticky
              pulse, and the count is suspended while dbg_stop = 1.
  bridge      the SPM bridge bound (CC §10.5): a silenced bridge gives a load fault and
              BRIDGE_TIMEOUT after 4,096 clk_rv cycles; its late answer is discarded.
  rv_reset    RV_RUN 1 -> 0 during a waiting push (CC §12.2): the push is dropped (no entry, no
              ticket, no ISSUE_PIDX step), RV_RESET_DONE rises, RT_* clear, dmstatus.allunavail;
              ISSUE_PIDX / ISSUE_TICKET read through SBA; SBA TCM writes refused while held; the
              loader works in reset and is refused while the hart runs or outside the TCMs.
  rv_reset_bridge  RV_RUN 1 -> 0 during an outstanding SPM-bridge access: the reset waits for
              it (bounded), then a clean reboot.
  lb          line-buffer invalidation (CC §10.5): a hit keeps old data, a control-window access
              and a rising interrupt line invalidate, an SBA DTCM write does not.
  fifo_full   a full issue FIFO (dispatcher stopped) gives every unit credit 0 and holds the next
              push although its queue has room (CC §7.3).
  err_rec     error records (CC §9.4 I2, §11.2): a record sent by the core is visible in the
              control window (read through SBA) before its acknowledge; clk_rv stopped: loader and
              DMI requests complete locally with a timeout (RV_CLK_OK = 0, CC §15.2).
Mutations: tb/ctrl/mutate_ctrl.py (CC §15.5 spot checks).

Environment: TB_CTRL_RV_PS (clk_rv period, default 6000 ps = 166.7 MHz), TB_CTRL_CORE_PS (3620 ps =
276.2 MHz), TSR_SEED, TSR_SIM_LIB, TB_CTRL_SRC (a directory replacing rtl/ctrl, for mutations).
"""
from __future__ import annotations

import os
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import ctrl_tb  # noqa: E402

try:
    import pytest
except ImportError:      # inside the simulator
    pytest = None

REPO = ctrl_tb.REPO
FW_OUT = REPO / "build" / "cocotb" / "ctrl_fw"
DONE = 0xD0E


def sources(ctrl_dir=None):
    """Sources and build arguments: the waivers, tsr_tables_pkg, the generic SRAM, rtl/ctrl/filelist.f
    (or its copy in ctrl_dir) and the bench HDL."""
    ctrl_dir = Path(ctrl_dir or os.environ.get("TB_CTRL_SRC", REPO / "rtl" / "ctrl"))
    srcs = [REPO / "rtl" / "core" / "tsr_tables_pkg.sv", REPO / "rtl" / "tech" / "generic" / "tsr_sram_1r1w.sv"]
    # the waiver files go first on the command line (cocotb_tools 2.0.1 does not classify .vlt sources)
    args = [str(REPO / "rtl" / "ctrl" / "waivers" / "vendor.vlt"), "-DRVFI"]
    for line in (REPO / "rtl" / "ctrl" / "filelist.f").read_text().splitlines():
        line = line.split("//", 1)[0].strip()
        if not line:
            continue
        if line.startswith("+incdir+"):
            args.append("+incdir+" + str((REPO / "rtl" / "ctrl" / line[len("+incdir+"):]).resolve()))
        elif line.startswith("+define+"):
            args.append(line)
        else:
            p = (REPO / "rtl" / "ctrl" / line).resolve()
            if (REPO / "rtl" / "ctrl") in p.parents:
                p = ctrl_dir / p.relative_to(REPO / "rtl" / "ctrl")
            srcs.append(p)
    srcs += [REPO / "rtl/core/tsr_rv_local_transport.sv",
             HERE / "sv" / "tb_ctrl_core.sv", HERE / "sv" / "tb_ctrl.sv"]
    return srcs, args


def periods():
    return int(os.environ.get("TB_CTRL_RV_PS", "6000")), int(os.environ.get("TB_CTRL_CORE_PS", "3620"))


def seed():
    return int(os.environ.get("TSR_SEED", "1"), 0)


# =====================================================================================================
# cocotb tests
# =====================================================================================================
try:
    import cocotb
    from cocotb.clock import Clock
    from cocotb.triggers import ClockCycles, RisingEdge
except ImportError:      # collected by pytest outside the simulator
    cocotb = None

if cocotb is not None:

    @cocotb.test()
    async def lockstep(dut):
        import rvsim
        rv_ps, core_ps = periods()
        await ctrl_tb.start(dut, rv_ps, core_ps, seed=seed(), lat=8)
        elf = ctrl_tb.build_fw(FW_OUT)
        syms = ctrl_tb.elf_symbols(elf)
        nw = await ctrl_tb.load_image(dut, elf)
        dut._log.info("loaded and verified %d words", nw)
        dut.rv_run.value = 1
        cyc = await ctrl_tb.wait_rt(dut, DONE)
        await ClockCycles(dut.clk, 3000)      # let the stub drain the FIFO and the counters settle
        rt = (int(dut.rt_state.value), int(dut.rt_info.value))
        spm_lines = {}
        dut._log.info("firmware done after ~%d core cycles; %d RVFI records, %d FIFO entries",
                      cyc, int(dut.rvfi_count.value), int(dut.stub_pops.value))
        assert int(dut.stub_errors.value) == 0, "stub saw %d protocol errors" % int(dut.stub_errors.value)
        # hold the hart, read the DTCM image back through the load port
        dut.rv_run.value = 0
        while not int(dut.rv_reset_done.value):
            await RisingEdge(dut.clk)
        n_dtcm = (syms["__bss_end"] - ctrl_tb.DTCM_BASE + 3) // 4
        dtcm = bytearray()
        for i in range(n_dtcm):
            v, err, to = await ctrl_tb.ld_access(dut, 0, ctrl_tb.DTCM_BASE + 4 * i)
            assert not err and not to
            dtcm += struct.pack("<I", v)
        run = ctrl_tb.run_dir()
        rtl = ctrl_tb.read_rvfi(run / "rvfi.log")
        # the hart spins at __fw_end after main returns; the simulator stops there
        end = next((k for k, r in enumerate(rtl) if r["pc"] == syms["__fw_end"]), len(rtl))
        assert all(r["pc"] == syms["__fw_end"] for r in rtl[end:]), "RTL ran past __fw_end"
        rtl = rtl[:end]
        rtl_if = ctrl_tb.read_ififo(run / "ififo.log")
        rtl_lcq = [[int(x, 16) for x in l.split()] for l in (run / "lq.log").read_text().split("\n") if l.strip()]
        rtl_log = [int(l, 16) for l in (run / "logw.log").read_text().split("\n") if l.strip()]
        spm_w = [l.split() for l in (run / "spm.log").read_text().split("\n") if l.startswith("W")]
        for ln in sorted({int(w[1], 16) // 64 for w in spm_w}):
            spm_lines[ln] = int(dut.u_core.spm[ln].value).to_bytes(64, "little")

        sim = rvsim.SimRun(elf)
        try:
            sim.run(ref=rtl)
            errs = []
            e = ctrl_tb.compare_records(rtl, sim.records)
            errs += ["retire: " + x for x in e]
            e = ctrl_tb.compare_pushes(rtl_if, sim.pushes)
            errs += ["issue FIFO: " + x for x in e]
            if rtl_lcq != sim.lcq_records():
                errs.append("LCQ records RTL %s / simulator %s" % (rtl_lcq, sim.lcq_records()))
            if rtl_log != sim.log_words():
                errs.append("log words RTL %s / simulator %s" % (rtl_log, sim.log_words()))
            sim_rt = (sim.csr(0x038), sim.csr(0x03C))
            if rt != sim_rt:
                errs.append("RT_STATE / RT_INFO mirrors RTL %08x %08x / simulator %08x %08x" % (rt + sim_rt))
            for ln, data in spm_lines.items():
                if data != sim.spm(64 * ln, 64):
                    errs.append("SPM line %d RTL %s / simulator %s" % (ln, data.hex(), sim.spm(64 * ln, 64).hex()))
            sim_dtcm = sim.dtcm(len(dtcm))
            if bytes(dtcm) != sim_dtcm:
                diff = [i for i in range(0, len(dtcm), 4) if dtcm[i:i + 4] != sim_dtcm[i:i + 4]]
                errs.append("DTCM differs in %d words, first at %#x: RTL %s / simulator %s" % (
                    len(diff), ctrl_tb.DTCM_BASE + diff[0], dtcm[diff[0]:diff[0] + 4].hex(),
                    sim_dtcm[diff[0]:diff[0] + 4].hex()))
            dut._log.info("lockstep: %d instructions, %d injected timing-dependent loads, %d FIFO entries, "
                          "%d LCQ records, %d log words, %d SPM writes in %d lines, %d DTCM words",
                          len(rtl), sim.injected, len(rtl_if), len(rtl_lcq), len(rtl_log), len(spm_w),
                          len(spm_lines), n_dtcm)
            for x in errs[:40]:
                dut._log.error(x)
            assert not errs, "%d mismatches against the simulator" % len(errs)
        finally:
            sim.close()


    @cocotb.test()
    async def dmi(dut):
        import rvsim
        rv_ps, core_ps = periods()
        await ctrl_tb.start(dut, rv_ps, core_ps, seed=seed(), lat=4)
        elf = ctrl_tb.build_fw(FW_OUT)
        syms = ctrl_tb.elf_symbols(elf)
        await ctrl_tb.load_image(dut, elf, patches={syms["fw_mode"]: 1})
        dut.rv_run.value = 1
        await ClockCycles(dut.clk, 20000)                # the hart is in mode 1's loop by then
        assert not int(dut.rv_halted.value)
        prt = ctrl_tb.DmiPort(dut=dut)
        rtl_out, rtl_dpc = await ctrl_tb.dmi_sequence(prt, syms["fw_mode"])
        await ClockCycles(dut.clk, 200)
        # RV_HALTED (CC-29) against dmstatus.allhalted: halt again and look at both
        p = ctrl_tb.DmiPort(dut=dut)
        await p.wr(ctrl_tb.DMI_DMCONTROL, 0x80000001)
        st = await p.poll(ctrl_tb.DMI_DMSTATUS, 0x200, 0x200)
        await ClockCycles(dut.clk, 10)
        halted_rtl = int(dut.rv_halted.value)
        await p.wr(ctrl_tb.DMI_DMCONTROL, 0x40000001)
        await p.poll(ctrl_tb.DMI_DMSTATUS, 0x800, 0x800)
        await ClockCycles(dut.clk, 20)
        running_rtl = not int(dut.rv_halted.value)
        assert halted_rtl and running_rtl, "RV_HALTED %d while halted, %d after resume (dmstatus %#x)" % (
            halted_rtl, int(dut.rv_halted.value), st)

        dm = rvsim.DmSim(elf, patches={syms["fw_mode"]: 1})
        try:
            dm.advance(20000)
            psim = ctrl_tb.DmiPort(dmsim=dm)
            sim_out, sim_dpc = await ctrl_tb.dmi_sequence(psim, syms["fw_mode"])
        finally:
            dm.close()
        loop = (syms["main"], syms["main"] + 0x1000)      # mode 1 spins inside main
        errs = []
        for (la, va), (lb, vb) in zip(rtl_out, sim_out):
            assert la == lb
            if va != vb:
                errs.append("%-28s RTL %08x / DM model %08x" % (la, va, vb))
        for x in rtl_out:
            dut._log.info("  %-28s %08x", x[0], x[1])
        dut._log.info("dpc RTL %08x, DM model %08x", rtl_dpc, sim_dpc)
        for e in errs + ["RTL: " + t for t in prt.timeouts] + ["DM model: " + t for t in psim.timeouts]:
            dut._log.error(e)
        assert not prt.timeouts and not psim.timeouts, "DMI conditions not reached"
        assert not errs, "%d DMI reads differ from the simulator's DM model" % len(errs)
        assert loop[0] <= rtl_dpc < loop[1] and loop[0] <= sim_dpc < loop[1], (hex(rtl_dpc), hex(sim_dpc))

        # DB's halt request (DB §3.3): one toggle gives one halt (cause haltreq); pend clears at
        # RV_HALTED; a resume does not re-halt
        dut.dbg_halt_tgl.value = 1
        for _ in range(4000):
            await RisingEdge(dut.clk)
            if int(dut.rv_halted.value):
                break
        else:
            raise AssertionError("DB halt request did not halt the hart")
        await ClockCycles(dut.clk, 20)
        assert not int(dut.rv_halt_pend.value), "pend not cleared by RV_HALTED"
        await p.wr(ctrl_tb.DMI_COMMAND, 0x00220000 | 0x7B0)
        await p.poll(ctrl_tb.DMI_ABSTRACTCS, 1 << 12, 0)
        dcsr = await p.rd(ctrl_tb.DMI_DATA0)
        assert (dcsr >> 6) & 7 == 3, "dcsr.cause %d after a DB halt request" % ((dcsr >> 6) & 7)
        await p.wr(ctrl_tb.DMI_DMCONTROL, 0x40000001)
        await p.poll(ctrl_tb.DMI_DMSTATUS, 0x800, 0x800)
        await p.wr(ctrl_tb.DMI_DMCONTROL, 0x1)
        await ClockCycles(dut.clk, 3000)
        assert not int(dut.rv_halted.value), "the hart halted again after the resume"
        # halt-on-boot (DB §3.8): the hart halts before its first instruction, dpc = RV_BOOT + 0x80
        dut.rv_run.value = 0
        while not int(dut.rv_reset_done.value):
            await RisingEdge(dut.clk)
        dut.dbg_halt_on_boot.value = 1
        dut.rv_run.value = 1
        for _ in range(4000):
            await RisingEdge(dut.clk)
            if int(dut.rv_halted.value):
                break
        else:
            raise AssertionError("halt-on-boot did not halt the hart")
        await p.poll(ctrl_tb.DMI_DMSTATUS, 0x200, 0x200)
        await p.wr(ctrl_tb.DMI_COMMAND, 0x00220000 | 0x7B1)
        await p.poll(ctrl_tb.DMI_ABSTRACTCS, 1 << 12, 0)
        dpc = await p.rd(ctrl_tb.DMI_DATA0)
        dut.dbg_halt_on_boot.value = 0
        assert dpc == 0x80, "halt-on-boot dpc %08x" % dpc
        assert not p.timeouts, p.timeouts

    @cocotb.test()
    async def halt_reset_pending(dut):
        """DB §§3.3,3.8: reset release discards old requests; cancel and new requests still work."""
        rv_ps, core_ps = periods()
        await ctrl_tb.start(dut, rv_ps, core_ps, seed=seed())
        _, err, timeout = await ctrl_tb.ld_access(dut, 1, 0x80, 0x0000006F)  # jal x0, 0
        assert not err and not timeout
        p = ctrl_tb.DmiPort(dut=dut)
        await p.wr(ctrl_tb.DMI_DMCONTROL, 1)  # keep the DM active before the first halt

        async def pending_request():
            dut.dbg_halt_tgl.value = 1 - int(dut.dbg_halt_tgl.value)
            await ClockCycles(dut.clk_rv, 20)
            assert int(dut.rv_halt_pend.value), "halt request did not become pending while held"

        async def boot_running():
            dut.rv_run.value = 1
            await ClockCycles(dut.clk_rv, 100)
            assert int(dut.rv_running.value), "hart did not leave reset"
            assert not int(dut.rv_halt_pend.value), "reset release retained a stale halt request"
            assert not int(dut.rv_halted.value), "old request halted HALT_ON_BOOT=0 boot"

        # The simulator has the same regression: an unconsumed request predating RESET is
        # discarded even if no RV_HALTED or RV_HALT_CANCEL ever acknowledged it.
        await pending_request()
        await boot_running()
        # A genuinely new request after reset release still halts, and resume stays running.
        dut.dbg_halt_tgl.value = 1 - int(dut.dbg_halt_tgl.value)
        for _ in range(1000):
            await RisingEdge(dut.clk_rv)
            if int(dut.rv_halted.value):
                break
        else:
            raise AssertionError("new post-reset halt request was lost")
        await ClockCycles(dut.clk_rv, 10)
        assert not int(dut.rv_halt_pend.value)
        await p.wr(ctrl_tb.DMI_DMCONTROL, 0x40000001)
        await p.poll(ctrl_tb.DMI_DMSTATUS, 0x800, 0x800)
        await p.wr(ctrl_tb.DMI_DMCONTROL, 1)
        await ClockCycles(dut.clk_rv, 100)
        assert not int(dut.rv_halted.value), "new request was not one-shot"

        dut.rv_run.value = 0
        while not int(dut.rv_reset_done.value):
            await RisingEdge(dut.clk)
        await pending_request()
        # The cancel and boot level cross together; cancel must take precedence over the old
        # request and must not interfere with the normal reset release.
        dut.dbg_cancel_tgl.value = 1
        await boot_running()
        assert not p.timeouts, p.timeouts

    @cocotb.test()
    async def push_wait(dut):
        rv_ps, core_ps = periods()
        await ctrl_tb.start(dut, rv_ps, core_ps, seed=seed(), lat=4)
        elf = ctrl_tb.build_fw(FW_OUT)
        syms = ctrl_tb.elf_symbols(elf)
        await ctrl_tb.load_image(dut, elf, patches={syms["fw_mode"]: 2})
        dut.cfg_pop_mute.value = 1                       # the dispatcher takes nothing: credits run out
        dut.rv_run.value = 1
        while ctrl_tb.vec(dut.issued_seq, 28, 5) != 4:   # four NOPs went (SYNC depth 4), the 5th waits
            await ClockCycles(dut.clk, 50)
        wait_max = int(os.environ.get("TB_CTRL_WAIT_MAX", "65536"))
        # dbg_stop suspends the ISSUE_WAIT_MAX count (CC §7.3, DB §3.9)
        # (push_timeout is a one-cycle pulse of the core clock: sample it there; n counts clk_rv
        # cycles through the core-side copy of the clk_rv cycle count)
        dut.dbg_stop.value = 1
        n, prev = 0, int(dut.rv_cyc.value)
        while n < wait_max + 20000:
            await RisingEdge(dut.clk)
            cur = int(dut.rv_cyc.value)
            n += (cur - prev) & 0xFFFF
            prev = cur
            assert not int(dut.push_timeout.value), "push timed out under dbg_stop"
        dut.dbg_stop.value = 0
        n = 0
        prev = int(dut.rv_cyc.value)
        while not int(dut.push_timeout.value):
            await RisingEdge(dut.clk)
            cur = int(dut.rv_cyc.value)
            n += (cur - prev) & 0xFFFF
            prev = cur
            assert n < wait_max + 1000, "no PUSH_TIMEOUT"
        dut._log.info("PUSH_TIMEOUT %d clk_rv cycles after dbg_stop fell (ISSUE_WAIT_MAX %d)", n, wait_max)
        assert wait_max - 2000 < n <= wait_max + 50, n
        await ctrl_tb.wait_rt(dut, DONE, max_cycles=4 * wait_max)
        assert int(dut.perf_push_stall.value) >= 2 * wait_max, int(dut.perf_push_stall.value)
        dut.rv_run.value = 0
        while not int(dut.rv_reset_done.value):
            await RisingEdge(dut.clk)
        res = [(await ctrl_tb.ld_access(dut, 0, syms["res"] + 4 * i))[0] for i in range(15)]
        # per push: ISSUE_PIDX, trap count; then ISSUE_STATUS before / after its W1C
        want = [1, 0, 2, 0, 3, 0, 4, 0, 4, 1, 4, 2, 1, 0]
        assert res[:14] == want, "results %s, expected %s" % (res[:14], want)
        dut.cfg_pop_mute.value = 0
        await ClockCycles(dut.clk, 500)
        assert int(dut.stub_pops.value) == 4 and int(dut.stub_errors.value) == 0

    @cocotb.test()
    async def bridge(dut):
        rv_ps, core_ps = periods()
        await ctrl_tb.start(dut, rv_ps, core_ps, seed=seed(), lat=4)
        elf = ctrl_tb.build_fw(FW_OUT)
        syms = ctrl_tb.elf_symbols(elf)
        await ctrl_tb.load_image(dut, elf, patches={syms["fw_mode"]: 3})
        line10 = bytes((0x30 + i) & 0xFF for i in range(64))
        dut.u_core.spm[10].value = int.from_bytes(line10, "little")
        dut.cfg_spm_mute.value = 1
        dut.rv_run.value = 1
        # the window request leaves when the bridge handshake gets busy; from there to the timeout
        while not int(dut.u_dut.u_cdc.x_bvalid.value):
            await RisingEdge(dut.clk)
        c0, n = int(dut.rv_cyc.value), 0
        while not int(dut.bridge_timeout.value):         # a core-clock pulse
            await RisingEdge(dut.clk)
            n = (int(dut.rv_cyc.value) - c0) & 0xFFFF
            assert n < 20000, "no BRIDGE_TIMEOUT"
        dut._log.info("BRIDGE_TIMEOUT ~%d clk_rv cycles after the request", n)
        assert 4090 <= n <= 4200, n
        await ClockCycles(dut.clk, 300)
        dut.cfg_spm_mute.value = 0                       # the late answer of the first load comes now
        await ctrl_tb.wait_rt(dut, DONE)
        dut.rv_run.value = 0
        while not int(dut.rv_reset_done.value):
            await RisingEdge(dut.clk)
        res = [(await ctrl_tb.ld_access(dut, 0, syms["res"] + 4 * i))[0] for i in range(4)]
        spm = (Path(ctrl_tb.run_dir()) / "spm.log").read_text().split()
        dut._log.info("results %s; SPM requests %s", [hex(x) for x in res], spm)
        assert res[1] == 1 and res[2] == int.from_bytes(line10[4:8], "little") and res[3] == 1, res
        assert int(dut.perf_bridge_acc.value) == 2

    @cocotb.test()
    async def rv_reset(dut):
        rv_ps, core_ps = periods()
        await ctrl_tb.start(dut, rv_ps, core_ps, seed=seed(), lat=4)
        elf = ctrl_tb.build_fw(FW_OUT)
        syms = ctrl_tb.elf_symbols(elf)
        await ctrl_tb.load_image(dut, elf, patches={syms["fw_mode"]: 4})
        dut.cfg_pop_mute.value = 1
        dut.rv_run.value = 1
        await ctrl_tb.wait_rt(dut, 4, step=50)
        while ctrl_tb.vec(dut.issued_seq, 28, 5) != 4:
            await ClockCycles(dut.clk, 50)
        await ClockCycles(dut.clk, 2000)                 # the 5th push waits
        assert not int(dut.rv_reset_done.value)
        dut.rv_run.value = 0                             # RISC-V reset during the waiting push
        n = 0
        while not int(dut.rv_reset_done.value):
            await RisingEdge(dut.clk)
            n += 1
            assert n < 2000
        await ClockCycles(dut.clk, 200)
        assert int(dut.rt_state.value) == 0 and int(dut.rt_info.value) == 0, "RT_* not cleared"
        assert ctrl_tb.vec(dut.issued_seq, 28, 5) == 4
        # the load port works while the hart is held, is refused while it runs
        _, err, to = await ctrl_tb.ld_access(dut, 1, 0x00100300, 0x13579BDF)
        assert not err and not to
        v, err, to = await ctrl_tb.ld_access(dut, 0, 0x00100300)
        assert v == 0x13579BDF and not err and not to
        _, err, _ = await ctrl_tb.ld_access(dut, 1, 0x00200000, 0)
        assert err, "load port outside the TCMs not refused"
        p = ctrl_tb.DmiPort(dut=dut)
        await p.wr(ctrl_tb.DMI_DMCONTROL, 1)
        pidx, cs = await p.sba_read(0x2000003C)          # ISSUE_PIDX: the dropped push took no index
        tick, _ = await p.sba_read(0x2000002C)
        st = await p.rd(ctrl_tb.DMI_DMSTATUS)
        dut._log.info("after the RISC-V reset: ISSUE_PIDX %d, ticket %08x, sbcs %08x, dmstatus %08x", pidx, tick, cs, st)
        assert pidx == 4 and tick == 0x50000003, (pidx, hex(tick))
        assert st & (1 << 13), "dmstatus.allunavail not set with the hart held in reset"
        # SBA TCM writes are refused while the hart is held (CC-33)
        cs = await p.sba_write(0x00100300, 0)
        assert (cs >> 12) & 7 == 2, "SBA TCM write while held: sbcs %08x" % cs
        dut.cfg_pop_mute.value = 0
        await ClockCycles(dut.clk, 500)
        assert int(dut.stub_pops.value) == 4 and int(dut.stub_errors.value) == 0, int(dut.stub_pops.value)
        # boot again: the load port is refused while the hart runs
        dut.rv_run.value = 1
        await ClockCycles(dut.clk, 100)
        _, err, _ = await ctrl_tb.ld_access(dut, 1, 0x00100300, 0)
        assert err, "load port write while the hart runs not refused"

    @cocotb.test()
    async def rv_reset_bridge(dut):
        # RV_RUN 1 -> 0 while an SPM-bridge access is outstanding (CC §12.2 step 2: the sequencer
        # waits for it, at most the bridge bound), then a clean reboot
        rv_ps, core_ps = periods()
        await ctrl_tb.start(dut, rv_ps, core_ps, seed=seed(), lat=4)
        elf = ctrl_tb.build_fw(FW_OUT)
        syms = ctrl_tb.elf_symbols(elf)
        await ctrl_tb.load_image(dut, elf, patches={syms["fw_mode"]: 3})
        line10 = bytes((0x60 + 3 * i) & 0xFF for i in range(64))
        dut.u_core.spm[10].value = int.from_bytes(line10, "little")
        dut.cfg_spm_mute.value = 1
        dut.rv_run.value = 1
        while not int(dut.u_dut.u_cdc.x_bvalid.value):
            await RisingEdge(dut.clk)
        await ClockCycles(dut.clk, 100)
        dut.rv_run.value = 0
        c0, n, prev = 0, 0, int(dut.rv_cyc.value)
        while not int(dut.rv_reset_done.value):
            await RisingEdge(dut.clk)
            cur = int(dut.rv_cyc.value)
            n += (cur - prev) & 0xFFFF
            prev = cur
            assert n < 6000, "RISC-V reset did not complete within the bridge bound"
        dut._log.info("RV_RESET_DONE %d clk_rv cycles after RV_RUN fell (bridge outstanding)", n)
        assert n > 3500, "the reset did not wait for the outstanding bridge access (%d cycles)" % n
        dut.cfg_spm_mute.value = 0
        await ClockCycles(dut.clk, 500)
        dut.rv_run.value = 1
        await ctrl_tb.wait_rt(dut, DONE)
        dut.rv_run.value = 0
        while not int(dut.rv_reset_done.value):
            await RisingEdge(dut.clk)
        res = [(await ctrl_tb.ld_access(dut, 0, syms["res"] + 4 * i))[0] for i in range(4)]
        assert res == [0xA5A5A5A5, 0, int.from_bytes(line10[4:8], "little"), 0], [hex(x) for x in res]

    @cocotb.test()
    async def err_rec(dut):
        rv_ps, core_ps = periods()
        rv_clk = await ctrl_tb.start(dut, rv_ps, core_ps, seed=seed(), lat=4)
        rec = (7 << 96) | (0x20 << 64) | (0x123 << 32) | (4 << 28) | 5   # pidx, w0, info, code, seq
        dut.err_rec.value = rec << (128 * 2)
        dut.err_valid.value = 1 << 2
        seen = None
        for _ in range(200):
            await RisingEdge(dut.clk)
            if int(dut.err_ack.value) >> 2 & 1:
                seen = int(dut.u_dut.u_cdc.r_err_rec.value) >> 256 & ((1 << 128) - 1)
                break
        dut.err_valid.value = 0
        assert seen == rec, "record not visible in clk_rv at its acknowledge (I2): %s" % (hex(seen) if seen else seen)
        p = ctrl_tb.DmiPort(dut=dut)
        await p.wr(ctrl_tb.DMI_DMCONTROL, 1)
        vals = [(await p.sba_read(0x20000000 + o))[0] for o in (0x0E0, 0x0E4, 0x0E8, 0x0EC, 0x188)]
        assert vals == [5, 4, 0x123, 0x20, 7], vals
        # clk_rv stopped: loader and DMI answer locally with a timeout; RV_CLK_OK falls (CC §15.2)
        rv_clk.cancel()
        await ClockCycles(dut.clk, 400)
        assert not int(dut.rv_clk_ok.value), "RV_CLK_OK still 1 with clk_rv stopped"
        _, _, to = await ctrl_tb.ld_access(dut, 0, 0x00100000)
        _, _, dto = await ctrl_tb.dmi(dut, 1, ctrl_tb.DMI_DMSTATUS)
        assert to and dto, "requests with clk_rv stopped did not time out locally"
        rec3 = (9 << 96) | (0x30 << 64) | (0x456 << 32) | (3 << 28) | 7
        dut.err_rec.value = (rec << (128 * 2)) | (rec3 << (128 * 3))
        dut.err_valid.value = 1 << 3
        for _ in range(50):
            await RisingEdge(dut.clk)
            if int(dut.err_ack.value) >> 3 & 1:
                break
        else:
            raise AssertionError("error record not acknowledged locally with clk_rv stopped")
        dut.err_valid.value = 0
        # The acknowledge was local: clk_rv's copy of unit 3 is stale. When clk_rv runs again (no reset),
        # one rec_resync pulse makes the core send every record once more (tsr_issue's reaction is
        # modelled here: err_valid[u] until err_ack[u]); afterwards clk_rv shows the records.
        assert (int(dut.u_dut.u_cdc.r_err_rec.value) >> 384) & ((1 << 128) - 1) != rec3
        pulses = []

        async def watch():
            while True:
                await RisingEdge(dut.clk)
                if int(dut.rec_resync.value):
                    pulses.append(1)
        w = cocotb.start_soon(watch())
        cocotb.start_soon(Clock(dut.clk_rv, rv_ps, unit="ps").start())
        for _ in range(3000):
            await RisingEdge(dut.clk)
            if pulses:
                break
        assert pulses, "no rec_resync pulse after clk_rv recovered"
        pending = 0x3F
        dut.err_valid.value = pending
        for _ in range(600):
            await RisingEdge(dut.clk)
            pending &= ~int(dut.err_ack.value)
            dut.err_valid.value = pending
            if not pending:
                break
        assert not pending, "resync: records %#x not acknowledged" % pending
        await ClockCycles(dut.clk, 400)
        w.cancel()
        r = int(dut.u_dut.u_cdc.r_err_rec.value)
        assert (r >> 256) & ((1 << 128) - 1) == rec and (r >> 384) & ((1 << 128) - 1) == rec3, \
            "clk_rv still shows the stale record after the resync"
        assert len(pulses) == 1, "rec_resync pulsed %d times for one clk_rv recovery" % len(pulses)


    @cocotb.test()
    async def lb(dut):
        rv_ps, core_ps = periods()
        await ctrl_tb.start(dut, rv_ps, core_ps, seed=seed(), lat=4)
        elf = ctrl_tb.build_fw(FW_OUT)
        syms = ctrl_tb.elf_symbols(elf)
        await ctrl_tb.load_image(dut, elf, patches={syms["fw_mode"]: 5})
        p = ctrl_tb.DmiPort(dut=dut)
        await p.wr(ctrl_tb.DMI_DMCONTROL, 1)
        dut.rv_run.value = 1
        await ClockCycles(dut.clk, 4000)                 # the hart spins on flag5
        pat = [0x11110000 + i for i in range(3)]
        dut.u_core.spm[8].value = pat[0]                 # behind the hart's back (a tile op would)
        await p.sba_write(syms["flag5"], 1)
        await ClockCycles(dut.clk, 2000)
        dut.u_core.spm[9].value = pat[1]
        dut.irq_lq.value = 1                             # a rising interrupt line (MIE = 0)
        await ClockCycles(dut.clk, 50)
        await p.sba_write(syms["flag5"], 2)
        await ClockCycles(dut.clk, 2000)
        dut.u_core.spm[11].value = pat[2]
        await p.sba_write(syms["flag5"], 3)
        await ctrl_tb.wait_rt(dut, DONE)
        dut.rv_run.value = 0
        while not int(dut.rv_reset_done.value):
            await RisingEdge(dut.clk)
        res = [(await ctrl_tb.ld_access(dut, 0, syms["res"] + 4 * i))[0] for i in range(7)]
        old = 0xA5A5A5A5
        want = [old, old, pat[0], old, pat[1], old, old]
        dut._log.info("line-buffer reads %s", [hex(x) for x in res])
        assert res == want, "expected %s" % [hex(x) for x in want]

    @cocotb.test()
    async def fifo_full(dut):
        rv_ps, core_ps = periods()
        await ctrl_tb.start(dut, rv_ps, core_ps, seed=seed(), lat=4)
        elf = ctrl_tb.build_fw(FW_OUT)
        syms = ctrl_tb.elf_symbols(elf)
        await ctrl_tb.load_image(dut, elf, patches={syms["fw_mode"]: 6})
        dut.cfg_pop_mute.value = 1
        dut.rv_run.value = 1
        tot = lambda: sum(ctrl_tb.vec(dut.issued_seq, 28, u) for u in range(6))
        n = 0
        while tot() < 32:
            await ClockCycles(dut.clk, 50)
            n += 50
            assert n < 200000, "32 pushes not done"
        await ClockCycles(dut.clk, 3000)
        assert tot() == 32 and ctrl_tb.vec(dut.issued_seq, 28, 5) == 0, "a push went into a full FIFO"
        p = ctrl_tb.DmiPort(dut=dut)
        await p.wr(ctrl_tb.DMI_DMCONTROL, 1)
        credit, _ = await p.sba_read(0x20000030)         # ISSUE_CREDIT: no FIFO space, no credit
        assert credit == 0, "ISSUE_CREDIT %08x with a full issue FIFO" % credit
        dut.cfg_pop_mute.value = 0
        await ctrl_tb.wait_rt(dut, DONE)
        await ClockCycles(dut.clk, 2000)
        assert int(dut.stub_pops.value) == 33 and int(dut.stub_errors.value) == 0, int(dut.stub_pops.value)


# =====================================================================================================
# pytest
# =====================================================================================================
def _run(testcase, parameters=None, name="tb_ctrl", env=None, defines=(), seed=None):
    from common.runner import run
    srcs, args = sources()
    args += ["+define+" + d for d in defines]
    e = {"PYTHONPATH": str(HERE) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    e.update(env or {})
    return run("tb_ctrl", "test_ctrl", sources=srcs, parameters=parameters or {}, build_args=args,
               testcase=testcase, name=name, extra_env=e, seed=seed)


def test_ctrl_rv_local_transport():
    _run(["lockstep", "bridge", "rv_reset", "rv_reset_bridge", "lb", "fifo_full",
          "push_wait", "halt_reset_pending"],
         parameters={"RV_LOCAL_TRANSPORT": 1}, name="tb_ctrl_rv_local",
         env={"TB_CTRL_CORE_PS": "4000", "TB_CTRL_RV_PS": "6000"})


def test_ctrl_rv_local_transport_xslr():
    # The SPM link end's XSLR_REG request stage: the bridge and the line buffer, a reset with a request.
    _run(["lockstep", "bridge", "rv_reset_bridge", "lb"],
         parameters={"RV_LOCAL_TRANSPORT": 1, "XSLR_REG": 1}, name="tb_ctrl_rv_local_xslr",
         env={"TB_CTRL_CORE_PS": "4000", "TB_CTRL_RV_PS": "6000"})


def test_ctrl_lockstep():
    _run("lockstep")


def test_ctrl_dmi():
    _run("dmi")


def test_ctrl_halt_reset_pending():
    _run("halt_reset_pending")


def test_ctrl_push_wait():
    _run("push_wait")


def test_ctrl_bridge():
    _run("bridge")


def test_ctrl_rv_reset():
    _run("rv_reset")


def test_ctrl_err_rec():
    _run("err_rec")


def test_ctrl_rv_reset_bridge():
    _run("rv_reset_bridge")


def test_ctrl_lb():
    _run("lb")


def test_ctrl_fifo_full():
    _run("fifo_full")


# Two clocks (CC §15.4): F_RV / F_CORE = 0.31, 0.6 (above), 1.0 and 1.67 with the core at 276.2 MHz,
# and other stub latency seeds.
if pytest is not None:
    @pytest.mark.parametrize("rv_ps", [11678, 3620, 2168])
    def test_ctrl_lockstep_ratio(rv_ps):
        _run("lockstep", env={"TB_CTRL_RV_PS": str(rv_ps)})

    @pytest.mark.parametrize("seed", [2, 3])
    def test_ctrl_lockstep_seed(seed):
        _run("lockstep", seed=seed, env={"TSR_SEED": str(seed)})

    # CDC jitter mode (CC-27): every synchroniser bit takes 2 or 3 stages at random.
    @pytest.mark.parametrize("case", ["lockstep", "dmi", "halt_reset_pending", "lb", "fifo_full", "err_rec"])
    def test_ctrl_jitter(case):
        _run(case, name="tb_ctrl_jitter", defines=["TSR_CDC_JITTER"])


    # CTRL_SYNC = 1 (CC §11.1: clk_rv = clk, every synchroniser one register stage)
    @pytest.mark.parametrize("case", ["lockstep", "dmi", "halt_reset_pending", "bridge"])
    def test_ctrl_sync(case):
        _run(case, parameters={"CTRL_SYNC": 1}, name="tb_ctrl_sync",
             env={"TB_CTRL_SYNC": "1", "TB_CTRL_RV_PS": "3620"})
