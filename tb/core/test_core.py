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
core_db     (W4-B) the DB block in the core: the same firmware with the DBG block configured through the host
            port before the boot: DBG_ID, the DMI gate (a DMI write with DMI_EN = 0 sets DMI_BLOCKED, W1C;
            with DMI_EN = 1 dmcontrol / dmstatus through the RTL debug module), the dispatch history, the trace
            writer (a 64-line ring in the ch0 IO region, written through the write mux of channel 0) and a tile
            breakpoint on the first MVM: the host waits for STOP_ACTIVE and DRAINED, compares the stop state
            (DBG_STOP_PIDX, NEXT_PIDX, the TBP, the units' oldest entries, Q_DONE ...) and the history with the
            simulator, RELEASEs and compares the final state, every history entry, the trace records without
            their times and the DRAM results (the DB block changes nothing).
core_m7     (W4-B) the production image tsr-fw.elf (build/dev/fw) on the RTL Ibex runs an op list of the M7
            kinds through the graph kernel: RMS_NORM, MUL_MAT (Q8_0, bias addends and residual adds), ROPE,
            SET_ROWS_KV (KVST on the DMA), FLASH_ATTN (ATTQK / ATTPV on the MVM unit, ASMAX / AMERGE / ANORM
            and the norm, RoPE and GLU ops on the VU), a decoder layer in the smallest shapes (the `tiny` group
            of sw/ctrl/tests/test_fw_m7_layer.cpp: decode T = 1 and a prefill tile T = 2), on a 2-channel core with
            1 MiB of SPM, CAPS VU | ATT. The host loads the image, boots, waits for HELLO, pushes the LAUNCH
            descriptor and waits for the DONE record, as the transport does (EX §3, §4). Compared with the
            simulator's run of the same launch (dumped to build/cocotb/core_m7/<case>): the identification
            registers, the DONE record, every output row of every record and both KV planes bit for bit, Q_DONE
            / Q_ISSUED / Q_ERR_CODE per unit (unit 3 = VU ops run), the s-register bank, the PERF event counters,
            no DRAM write outside the outputs and planes. The DBG block is on during the launch (history, the
            trace writer into a 64-line ring): every history entry (the runtime's QCLEAR and the layer's ops with
            their push PCs and dependencies), the trace records without their times and the counters equal the
            simulator's.
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

        def __init__(self, dut, mem=None):
            self.d = dut
            self.mem = mem

        async def idle(self, n):
            await ClockCycles(self.d.clk, n)

        async def mem_read(self, ch, off, size):
            return self.mem.read_bytes(ch, off, size)

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


    @cocotb.test()
    async def core_db(dut):
        """The DB block in the core: the fw_core program with the DBG block configured through the host port
        (DMI gate, history, trace writer, a tile breakpoint at the first MVM), compared with the simulator."""
        seed = int(os.environ.get("TSR_SEED", "1"), 0)
        rv_ps = int(os.environ.get("TB_CORE_RV_PS", "6000"))
        core_ps = int(os.environ.get("TB_CORE_PS", "3620"))
        elf = Path(os.environ["TB_CORE_ELF"])
        sc = C.scenario(seed)
        sim = C.run_sim_db(sc, elf)

        dut.csr_valid.value = 0
        dut.csr_we.value = 0
        dut.csr_addr.value = 0
        dut.csr_wdata.value = 0
        dut.plat_err_i.value = 0
        dut.rst_n.value = 0
        dut.rst_rv_n.value = 0
        cocotb.start_soon(Clock(dut.clk, core_ps, unit="ps").start())
        await Timer((seed * 977) % rv_ps + 1, unit="ps")
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
        rtl = await C.db_prog(Host(dut, mem), sc, elf)
        errs = []
        C.compare_db(rtl, sim, errs)
        allowed = {}
        for ch, off, size in sc["regions"][:3]:
            allowed.setdefault(ch, set()).update(range(off // 64, (off + size) // 64))
        allowed.setdefault(0, set()).update(range(C.DB_RING_BEAT, C.DB_RING_BEAT + C.DB_RING_LINES))
        for c in range(C.CFG["num_ch"]):
            stray = sorted(set(mem.written[c]) - allowed.get(c, set()))
            if stray:
                errs.append("DRAM ch%d: %d beats written outside the store targets and the trace ring (first %s)" % (c, len(stray), stray[:8]))
        mem.check(idle=True)
        dut._log.info("DB: %d history entries, %d trace records, stops at pidx %s, TBP hits %s",
                      len(rtl["hist"]), len(rtl["trw"]), [x[TT.CSR_DBG_STOP_PIDX] for x in rtl["stops"]],
                      rtl["db_final"][TT.CSR_DBG_TBP_HITS(0)])
        assert not errs, "\n".join(errs[:40])

    @cocotb.test()
    async def core_m7(dut):
        seed = int(os.environ.get("TSR_SEED", "1"), 0)
        rv_ps = int(os.environ.get("TB_CORE_RV_PS", "6000"))
        core_ps = int(os.environ.get("TB_CORE_PS", "3620"))
        elf = Path(os.environ["TB_M7_ELF"])
        sc = C.parse_m7(Path(os.environ["TB_M7_DIR"]))
        cfg = sc["cfg"]
        data, expect = sc["data"], sc["expect"]

        dut.csr_valid.value = 0
        dut.csr_we.value = 0
        dut.csr_addr.value = 0
        dut.csr_wdata.value = 0
        dut.plat_err_i.value = 0
        dut.rst_n.value = 0
        dut.rst_rv_n.value = 0
        cocotb.start_soon(Clock(dut.clk, core_ps, unit="ps").start())
        await Timer((seed * 977) % rv_ps + 1, unit="ps")
        cocotb.start_soon(Clock(dut.clk_rv, rv_ps, unit="ps").start())
        mem = MemModel(dut, num_ch=cfg["num_ch"], addr_w=cfg["addr_beat_w"], seed=seed,
                       limit_beats=cfg["ch_bytes"] // 64, rd_outstanding=16)
        for ch, beat, n, off in sc["ins"]:
            mem.load(ch, beat * 64, data[off:off + n])
        mem.clear_tracking()
        mem.start()
        await ClockCycles(dut.clk, 30)
        dut.rst_rv_n.value = 1
        dut.rst_n.value = 1
        await ClockCycles(dut.clk, 40)
        host = Host(dut)
        errs = []

        for o, want in sc["ids"]:
            got = await host.rd(o)
            if got != want:
                errs.append("CSR %#05x: RTL %#010x / simulator %#010x" % (o, got, want))
        # the image: PT_LOAD segments through the loader (the receive CRC only; VERIFY is core_smoke's)
        t0 = cocotb.utils.get_sim_time("ns")
        nwords = 0
        for addr, seg in C.ctrl_tb.elf_segments(elf):
            words = list(struct.unpack("<%dI" % (len(seg) // 4), seg))
            await host.wr(TT.CSR_LOAD_ADDR, addr)
            for w in words:
                await host.wr(TT.CSR_LOAD_DATA, w)
            nwords += len(words)
            crc_rx = await host.rd(TT.CSR_LOAD_CRC)
            want = zlib.crc32(struct.pack("<%dI" % len(words), *words))
            if crc_rx != want:
                errs.append("load CRC of the segment at %#x: RTL %#010x / zlib %#010x" % (addr, crc_rx, want))
        dut._log.info("loaded %d words in %.1f us", nwords, (cocotb.utils.get_sim_time("ns") - t0) / 1e3)
        assert not errs, "\n".join(errs[:20])

        lcq_rd = 0

        async def consume():
            nonlocal lcq_rd
            wr = await host.rd(TT.CSR_LCQ_WR)
            recs = []
            for k in range(lcq_rd, wr):
                recs.append([await host.rd(TT.CSR_LCQ_REC(k % TT.LCQ_DEPTH, j)) for j in range(8)])
            lcq_rd = wr
            await host.wr(TT.CSR_LCQ_RD, wr)
            lw = await host.rd(TT.CSR_LOG_WR)
            await host.wr(TT.CSR_LOG_RD, lw)
            return recs

        async def wait_rec(pred, what, limit):
            cyc = 0
            got = []
            while True:
                await ClockCycles(dut.clk, 500)
                cyc += 500
                got += await consume()
                for r in got:
                    if pred(r):
                        return r, got, cyc
                if (await host.rd(TT.CSR_RT_STATE)) & 0xFF == TT.RT_STATE_PHASE_TRAPPED:
                    raise AssertionError("%s: the firmware trapped (RT_INFO %#x, records %s)" % (
                        what, await host.rd(TT.CSR_RT_INFO), got))
                if cyc > limit:
                    raise AssertionError("%s: no answer in %d core cycles (RT_STATE %#x, RT_INFO %#x, STATUS %#x)" % (
                        what, cyc, await host.rd(TT.CSR_RT_STATE), await host.rd(TT.CSR_RT_INFO),
                        await host.rd(TT.CSR_STATUS)))

        ent = struct.unpack_from("<I", elf.read_bytes(), 24)[0]
        await host.wr(TT.CSR_LOG_CFG, TT.LOG_CFG_ENABLE | (5 << TT.LOG_CFG_LEVEL_LSB))
        await host.wr(TT.CSR_RV_BOOT, ent - TT.BOOT_ENTRY_OFF)
        await host.wr(TT.CSR_CTRL, TT.CTRL_RV_RUN)
        hello, _, cyc = await wait_rec(lambda r: (r[0] & 0xFF) == TT.LCQ_R0_TYPE_HELLO, "boot", 2_000_000)
        dut._log.info("HELLO %d core cycles after RV_RUN", cyc)

        # DB: history (on by default) and the trace writer on during the launch, a 64-line ring below the list
        # (the same configuration as the simulator's run; the DB block changes no result)
        trw_beat, trw_lines = sc["trw_cfg"]
        await host.wr(TT.CSR_DBG_CFG, (await host.rd(TT.CSR_DBG_CFG)) | TT.DBG_CFG_TRW_EN)
        await host.wr(TT.CSR_DBG_TRW_BASE, trw_beat)
        await host.wr(TT.CSR_DBG_TRW_LINES, trw_lines)
        kidx, a0, a1, a2 = sc["launch"]
        seq = await host.rd(TT.CSR_LQ_SEQ)
        await host.wr(TT.CSR_LQ_W(0), TT.DESC_TYPE_LAUNCH | (kidx << TT.DESC_KERNEL_LSB))
        for i, v in enumerate((a0, a1, a2, 0, 0, 0, 0)):
            await host.wr(TT.CSR_LQ_W(1 + i), v)
        t0 = cocotb.utils.get_sim_time("ns")
        await host.wr(TT.CSR_LQ_PUSH, 1)
        done, _, cyc = await wait_rec(lambda r: (r[0] & 0xFF) in (TT.LCQ_R0_TYPE_DONE, TT.LCQ_R0_TYPE_PONG,
                                                                   TT.LCQ_R0_TYPE_TRAP) and r[1] == seq,
                                      "launch", int(os.environ.get("TB_M7_LIMIT", "6000000")))
        dut._log.info("launch: DONE %s, %.1f us (~%d core cycles polled)", [hex(x) for x in done[:7]],
                      (cocotb.utils.get_sim_time("ns") - t0) / 1e3, cyc)
        if done[:7] != sc["done"]:
            errs.append("DONE record r0-r6: RTL %s / simulator %s" % ([hex(x) for x in done[:7]],
                                                                       [hex(x) for x in sc["done"]]))
        n = 0
        while not ((await host.rd(TT.CSR_STATUS)) & TT.STATUS_QIDLE):
            await ClockCycles(dut.clk, 100)
            n += 1
            assert n < 20000, "QIDLE not reached"
        await ClockCycles(dut.clk, 3000)
        for o, want in sc["csrs"]:
            got = await host.rd(o)
            if got != want:
                errs.append("CSR %#05x after the launch: RTL %#010x / simulator %#010x" % (o, got, want))
        # DB: history entries, trace records (without their times) and counters against the simulator
        await host.wr(TT.CSR_DBG_CTRL, TT.DBG_CTRL_TRW_FLUSH)
        await ClockCycles(dut.clk, 4000)
        head = await host.rd(TT.CSR_DBG_HIST_HEAD)
        wptr = await host.rd(TT.CSR_DBG_TRW_WPTR)
        drops = await host.rd(TT.CSR_DBG_TRW_DROPS)
        p61, p62 = await host.rd(TT.CSR_PERF_LO(61)), await host.rd(TT.CSR_PERF_LO(62))
        s_head, s_wptr, s_drops, s_p61, s_p62 = sc["dbg"]
        if (head, wptr, drops, p62) != (s_head, s_wptr, s_drops, s_p62):
            errs.append("DB: HIST_HEAD / TRW_WPTR / TRW_DROPS / PERF TRW_DROPS RTL %s / simulator %s" % (
                (head, wptr, drops, p62), (s_head, s_wptr, s_drops, s_p62)))
        for who, v, w in (("RTL", p61, wptr), ("simulator", s_p61, s_wptr)):
            if not 0 < v <= w or w - v > 3:          # TRW_RECORDS counts while ACTIVE: the last flush may be after it
                errs.append("DB: %s PERF TRW_RECORDS %d with %d records written" % (who, v, w))
        for p in range(min(head, len(sc["hist"]))):
            await host.wr(TT.CSR_DBG_HIST_IDX, p)
            got = [await host.rd(x) for x in C.HIST_OFFS]
            if got != sc["hist"][p]:
                errs.append("DB: history %d: RTL %s / simulator %s" % (p, [hex(x) for x in got], [hex(x) for x in sc["hist"][p]]))
        if wptr == s_wptr and 0 < wptr <= 2 * trw_lines:
            raw = mem.read_bytes(0, trw_beat * 64, (wptr * 32 + 63) // 64 * 64)
            recs = [[int.from_bytes(raw[32 * k + 4 * i:32 * k + 4 * i + 4], "little") for i in range(8)] for k in range(wptr)]
            a, b = C.trw_view(recs), C.trw_view(sc["trw"])
            for k in a:
                if a[k] != b[k]:
                    errs.append("DB: trace records (%s) differ: RTL %s / simulator %s" % (k, a[k], b[k]))
        # outputs and KV planes, bit for bit
        allowed = {0: set(range(trw_beat, trw_beat + trw_lines))}
        for ch, beat, rows, stride, lines, off in sc["outs"]:
            for r in range(rows):
                got = mem.read_bytes(ch, (beat + r * stride) * 64, lines * 64)
                want = expect[off + r * lines * 64:off + (r + 1) * lines * 64]
                if got != want:
                    bad = [i for i in range(lines) if got[64 * i:64 * i + 64] != want[64 * i:64 * i + 64]]
                    errs.append("output ch%d row %d (beat %d): lines %s differ" % (ch, r, beat + r * stride, bad[:6]))
                allowed.setdefault(ch, set()).update(range(beat + r * stride, beat + r * stride + lines))
        for ch, beat, nbytes, off in sc["planes"]:
            got = mem.read_bytes(ch, beat * 64, nbytes)
            if got != expect[off:off + nbytes]:
                bad = [i for i in range(nbytes // 64) if got[64 * i:64 * i + 64] != expect[off + 64 * i:off + 64 * i + 64]]
                errs.append("KV plane ch%d beat %d: %d lines differ (first %s)" % (ch, beat, len(bad), bad[:6]))
            allowed.setdefault(ch, set()).update(range(beat, beat + nbytes // 64))
        for c in range(cfg["num_ch"]):
            stray = sorted(set(mem.written[c]) - allowed.get(c, set()))
            if stray:
                errs.append("DRAM ch%d: %d beats written outside the outputs and planes (first %s)" % (c, len(stray), stray[:8]))
        mem.check(idle=True)
        dut._log.info("units: Q_DONE %s", [await host.rd(TT.CSR_Q_DONE_BASE + 4 * u) for u in range(6)])
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


# W4-B wiring mutants of the VU / AT / DB integration in tsr_core_dp / tsr_core: (file, old, new, description,
# the test that must fail: "m7" = core_m7 on the decode case, "db" = core_db). A hang is a kill (the launch /
# stop waits are shortened for mutant runs: TB_M7_LIMIT, TB_DB_WAIT).
XMUTANTS = {
    "vu_bank": ("tsr_core_dp.sv", "assign vu_rd_rsp_data     = sb_rd_rsp_data;",
                "assign vu_rd_rsp_data     = {sb_rd_rsp_data[0], sb_rd_rsp_data[1]};",
                "the VU's SPM read data of the two banks swapped", "m7"),
    "xc_at_cmd": ("tsr_core_dp.sv",
                  "      .at_req_valid, .at_req_insn, .at_rsp_valid, .at_rsp_code, .at_rsp_info, .at_rsp_ext, .at_rsp_cmd\n",
                  "      .at_req_valid, .at_req_insn, .at_rsp_valid, .at_rsp_code, .at_rsp_info, .at_rsp_ext, .at_rsp_cmd('0)\n",
                  "the AT checker class's payload lost at the xc mux", "m7"),
    "kvst_off": ("tsr_core_dp.sv", ".SPM_RD_LAT(SPM_RD_LAT), .HAS_KVST(HAS_ATT), .XC_CMD_W(XC_CMD_W)",
                 ".SPM_RD_LAT(SPM_RD_LAT), .HAS_KVST(1'b0), .XC_CMD_W(XC_CMD_W)",
                 "KVST left out of the DMA", "m7"),
    "mvm_xc": ("tsr_core_dp.sv", ".cmd(mvm_cmd), .cmd_xc(mvm_cmd_xc),", ".cmd(mvm_cmd), .cmd_xc(1'b0),",
               "the MVM unit never sees an attention command", "m7"),
    "qd_vu": ("tsr_core.sv", "5'd0, 5'(QD_VU_E), 5'(QD_MVM_E)", "5'd0, 5'd0, 5'(QD_MVM_E)",
              "tsr_ctrl told the VU queue is absent", "m7"),
    "perf_vu_lines": ("tsr_core_dp.sv", "perf_ext_un[PERF_X + 15][0] = vu_perf_line;",
                      "perf_ext_un[PERF_X + 14][0] = vu_perf_line;", "VU_LINES counted in the VU_HAZARD slot", "m7"),
    "dbg_hold": ("tsr_core_dp.sv", ".sr_busy, .dbg_hold, .dbg_step,", ".sr_busy, .dbg_hold('0), .dbg_step,",
                 "DB's hold never reaches the queue heads", "db"),
    "wrmux_off": ("tsr_core_dp.sv", "if (HAS_DB && HAS_TRW && c == tsr_tables_pkg::TSR_TRW_CH) begin : g_trwmux",
                  "if (1'b0 && HAS_DB && HAS_TRW && c == tsr_tables_pkg::TSR_TRW_CH) begin : g_trwmux",
                  "the trace writer's write mux left out of channel 0", "db"),
    "dmi_open": ("tsr_core_dp.sv", "    .dmi_en, .dmi_ready, .dmi_req, .dmi_addr,", "    .dmi_en(1'b1), .dmi_ready, .dmi_req, .dmi_addr,",
                 "the DMI window always open (DMI_EN not wired)", "db"),
    "perf_db": ("tsr_core_dp.sv", "assign perf_ext = perf_ext_db | perf_ext_un;", "assign perf_ext = perf_ext_un;",
                "DB's PERF slots (TRW_RECORDS ...) not added", "m7"),
    "db_hist_deps": ("tsr_core_dp.sv",
                     "      .ifq_valid, .ifq_ready, .ifq_entry(tsr_ifent_t'(ifq_entry)),\n      .tap_enq, .tap_enq_unit,",
                     "      .ifq_valid(1'b0), .ifq_ready, .ifq_entry(tsr_ifent_t'(ifq_entry)),\n      .tap_enq, .tap_enq_unit,",
                     "DB does not see the issue-FIFO pops (history dependencies as pushed)", "db"),
}


def _mutant(mut):
    return MUTANTS[mut] if mut in MUTANTS else XMUTANTS[mut][:4]


def _sources(mut=None):
    from common.runner import rtl_sources
    srcs = rtl_sources()
    if not mut:
        return srcs
    fname, old, new, _ = _mutant(mut)
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


def _run_m7(case, seed=1, rv_ps=6000, mut=None):
    """core_m7 of one case: the scenario from the simulator, the RTL core built with its configuration."""
    from common.runner import run
    if not C.M7_BIN.exists() or not C.M7_ELF.exists():
        pytest.skip("build the simulator-side test and the image first: cmake --build build/dev --target "
                    "test_fw_m7_layer tsr_fw")
    d = C.m7_scenario(case)
    cfg = C.parse_m7(d)["cfg"]
    params = dict(NUM_CH=cfg["num_ch"], T_MAX=cfg["t_max"], K_MAX=cfg["k_max"], SPM_BYTES=cfg["spm_bytes"],
                  CH_BYTES="64'd%d" % cfg["ch_bytes"], ADDR_BEAT_W=cfg["addr_beat_w"])
    env = {"TB_M7_ELF": str(C.M7_ELF), "TB_M7_DIR": str(d), "TB_CORE_RV_PS": str(rv_ps)}
    if mut:
        env["TB_M7_LIMIT"] = "300000"                 # core cycles to wait for the answer (the decode case needs ~60 K)
    run("tsr_core", __name__, parameters=params, sources=_sources(mut), name="core_m7" if not mut else "core_m7_mut_" + mut,
        seed=seed, testcase=["core_m7"], extra_env=env, timescale=("1ps", "1ps"))


def _run_db(seed=1, rv_ps=6000, mut=None):
    from common.runner import run
    elf = C.build_fw(FW_OUT)
    env = {"TB_CORE_ELF": str(elf), "TB_CORE_RV_PS": str(rv_ps)}
    if mut:
        env["TB_DB_WAIT"] = "600"                     # polls of 500 core cycles (the run needs ~100)
    run("tsr_core", __name__, parameters=dict(C.PARAMS), sources=_sources(mut),
        name="core" if not mut else "core_mut_" + mut, seed=seed, testcase=["core_db"], extra_env=env,
        timescale=("1ps", "1ps"))


def test_core_db():
    _run_db(seed=int(os.environ.get("TSR_SEED", "1"), 0))


def test_core_m7(m7case):
    _run_m7(m7case, seed=int(os.environ.get("TSR_SEED", "1"), 0))


def _mutants_requested(request) -> bool:
    cfg = request.config
    return (os.environ.get("TSR_CORE_MUTANTS") == "1" or "mutants" in (cfg.getoption("keyword", "") or "")
            or "slow" in (cfg.getoption("markexpr", "") or "")
            or any("mutants" in str(a) for a in cfg.args))


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


def test_core_w4b_mutants(request):
    """Opt-in (as test_core_mutants; about 5 min per mutant): every wiring mutant of the VU / AT / DB integration
    must fail core_m7 (tiny-decode) or core_db."""
    if not _mutants_requested(request) and not any("w4b_mutants" in str(a) for a in request.config.args):
        pytest.skip("opt-in: TSR_CORE_MUTANTS=1, -m slow or -k mutants")
    sel = os.environ.get("TSR_CORE_MUT_ONLY")
    todo = [m for m in XMUTANTS if not sel or m in sel.split(",")]
    for m in todo:
        _sources(m)
    survived = []
    for m in todo:
        try:
            if XMUTANTS[m][4] == "m7":
                _run_m7("tiny-decode", mut=m)
            else:
                _run_db(mut=m)
        except BaseException as e:
            if isinstance(e, KeyboardInterrupt):
                raise
            print("mutant %-14s killed (%s)" % (m, type(e).__name__))
            continue
        survived.append(m)
        print("mutant %-14s SURVIVED" % m)
    assert not survived, "surviving mutants: %s" % survived


if pytest is not None:
    test_core = pytest.mark.parametrize("cfg", list(RUNS))(test_core)
    test_core_w4b_mutants = pytest.mark.slow(test_core_w4b_mutants)
    test_core_m7 = pytest.mark.parametrize("m7case", list(C.M7_CASES))(test_core_m7)
    test_core_mutants = pytest.mark.slow(test_core_mutants)
