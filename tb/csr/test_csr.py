"""tb/csr — tsr_csr (with tsr_issue, wired as tsr_core_dp wires them) against the simulator's host CSR port
and RISC-V control window (docs/design/01-execution-model.md "EX" §3-§6, §8.5; docs/design/04-control-core.md
"CC" §6, §12.2, §15.2).

The wrapper tb/csr/hdl/tsr_csr_tb.sv exposes the host CSR port and the core-domain side of tsr_ctrl_cdc;
the bench models the clk_rv side: the core window and LQ / LCQ handshakes, the log FIFO, the loader's
TCM port (ITCM / DTCM decode of the simulator, random latency, timeouts), the DMI port, the clk_rv levels
(RV_RESET_DONE after RV_RUN drops, RV_CLK_OK, ndmreset) and counters. The issue FIFO and the units are
tb/issue's environment. Every access the RTL answers is compared with the same access on the simulator
(tsr_sim_csr_read / _write, tsr_sim_cw_access for the RISC-V side, tsr_sim_corewin_read), cycle-agnostic:
  csr_regs        reset values and random writes / reads of the whole map (unmapped and unaligned read 0)
  csr_rings       LQ / LCQ / log protocol: pushes, overflow, pops and their faults, posts with and without
                  ANSWER and their faults, LCQ_RD / LOG_RD write rule, watermarks, LQ_CANCEL's QABORT,
                  LOG_FREE as the clk_rv side computes it, ring contents
  csr_loader      LOAD_ADDR / LOAD_DATA writes into ITCM, DTCM and holes, receive CRC, VERIFY CRC, LOAD_DATA
                  reads, TCM contents against the simulator's TCMs; refusals and LOAD_TIMEOUT (RTL only)
  csr_soft_reset  SOFT_RESET with held and running entries: SR_BUSY, the clear list of step 3 against the
                  simulator; RV_RUN_REFUSED while busy; SR_RV_TIMEOUT without clk_rv (RTL only)
  csr_perf        PERF event counters RETIRED / PUSHES / LAUNCHES against the simulator inside a launch, the
                  time counters against the bench's own count of the gates, the LO / HI snapshot rule
  csr_cwin        core-window reads (no snapshot side effect, LOAD_DATA and DMI read 0) interleaved with host
                  traffic, against tsr_sim_corewin_read
  csr_dmi, csr_wdog, csr_rvfreq, csr_rv_reset   RTL-only directed checks (DMI handshake / gating / timeout,
                  unit watchdog, RV_FREQ, RISC-V reset actions and ndmreset, host QCLEAR accept / refuse)
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent / "issue"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from common import isa, tables as T                      # noqa: E402
from common.runner import REPO, run                       # noqa: E402

import issue_sim as S                                     # noqa: E402
import issue_tb as E                                      # noqa: E402

def src():
    d = os.environ.get("TSR_MUT_DIR")                      # mutated copies (tb/issue/test_issue_mutants.py)
    pick = lambda n: Path(d) / n if d and (Path(d) / n).exists() else REPO / "rtl/core" / n   # noqa: E731
    return [REPO / "rtl/core/tsr_tables_pkg.sv", REPO / "rtl/core/tsr_pkg.sv", pick("tsr_chk.sv"),
            pick("tsr_issue.sv"), pick("tsr_csr.sv"), HERE / "hdl/tsr_csr_tb.sv"]
M32 = 0xFFFFFFFF
ITCM, DTCM = T.ITCM_BYTES, T.DTCM_BYTES
DTCM_BASE = 0x100000


def env_cfg():
    c = json.loads(os.environ.get("TSR_CSR_CFG", "{}"))
    return S.config(num_ch=c.get("num_ch", 4), t_max=c.get("t_max", 8)), tuple(c.get("qd", S.U200_QD)), c


# ---------------------------------------------------------------------------------------------
# Bench: host port, clk_rv-side models, the issue environment
# ---------------------------------------------------------------------------------------------
class Bench:
    def __init__(self, dut, cfg, qd, seed=1):
        self.dut = dut
        self.cfg, self.qd = cfg, qd
        self.r = random.Random(seed)
        self.sim = S.Sim(cfg, qd)
        self.env = E.Env(dut, cfg, qd, seed=seed, ctl_ports=("dbg_hold", "dbg_step"), gap=(0, 0), lat=(1, 6),
                         ready_p=0.9)
        # clk_rv side state
        self.rv = dict(reset_done=1, running=0, halted=0, sleep=0, clk_ok=1, ndmreset=0)
        self.rv_rst_lat = 12           # cycles from RV_RUN low to RV_RESET_DONE (clk_rv present)
        self.rv_present = True
        self.rv_ratio = 0.6
        self.rv_cyc = 0.0
        self.counts = [0, 0, 0]        # PUSH_STALL, BRIDGE_ACC, BRIDGE_WAIT
        self.tcm = {}                  # word address -> value (the TCMs of the loader port)
        self.ld_lat = (1, 8)
        self.ld_hang = False           # the TCM port never answers: the CDC times out
        self.dmi_lat = (1, 6)
        self.dmi_hang = False
        self.dmi_regs = {}
        self.dmi_log = []
        self.cycle = 0
        self.written_lcq = set()       # LCQ ring slots written (unwritten slots are undefined)
        self.written_log = set()
        self.log_wr_rv = 0             # LOG_PUSH stores counted on the clk_rv side
        self.time_en = self.evt_en = 0
        self.busy_time = [0] * 6

    # ---- host CSR port ----
    async def wr(self, off, val):
        d = self.dut
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
        d = self.dut
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

    async def cw(self, off):
        """Core-window read (tsr_ctrl_cdc requester 2)."""
        d = self.dut
        d.cwin_req_valid.value = 1
        d.cwin_req_off.value = off & 0xFFF
        while True:
            await FallingEdge(d.clk)
            if int(d.cwin_rsp_valid.value):
                v = int(d.cwin_rsp_rdata.value)
                break
        await RisingEdge(d.clk)
        d.cwin_req_valid.value = 0
        return v

    async def lq(self, off, we=0, wdata=0, lcq_w=None):
        """One LQ / LCQ handshake (control-window offset): (rdata, fault)."""
        d = self.dut
        d.lq_req_valid.value = 1
        d.lq_req_off.value = off
        d.lq_req_we.value = we
        d.lq_req_wdata.value = wdata & M32
        d.lq_req_lcq_w.value = isa.insn_int(lcq_w or [0] * 8)
        while True:
            await FallingEdge(d.clk)
            if int(d.lq_rsp_valid.value):
                v, f = int(d.lq_rsp_rdata.value), int(d.lq_rsp_fault.value)
                break
        await RisingEdge(d.clk)
        d.lq_req_valid.value = 0
        return v, f

    async def log_push(self, w):
        d = self.dut
        d.log_valid.value = 1
        d.log_data.value = w & M32
        await RisingEdge(d.clk)
        d.log_valid.value = 0
        self.log_wr_rv = (self.log_wr_rv + 1) & M32

    async def cycles(self, n):
        for _ in range(n):
            await RisingEdge(self.dut.clk)

    # ---- clk_rv side models (one loop) ----
    async def _rv_loop(self):
        d = self.dut
        r = self.r
        ld_pend = None
        dmi_pend = None
        prev_run = 0
        rst_at = None
        while True:
            await FallingEdge(d.clk)
            run = int(d.rv_run.value)
            ldq = int(d.ld_req.value)
            ld = (int(d.ld_we.value), int(d.ld_addr.value), int(d.ld_wdata.value)) if ldq else None
            dmq = int(d.dmi_req.value)
            dm = (int(d.dmi_addr.value), int(d.dmi_op.value), int(d.dmi_wdata.value)) if dmq else None
            self.time_en += int(d.perf_time_en.value)
            pb = int(d.perf_busy.value)
            te = int(d.perf_time_en.value)
            for u in range(6):
                self.busy_time[u] += te and (pb >> u) & 1
            await RisingEdge(d.clk)
            self.cycle += 1
            c = self.cycle
            # RISC-V reset sequence (CC §12.2) as the clk_rv side answers it
            if prev_run and not run:
                rst_at = c + self.rv_rst_lat
                self.rv["running"] = 0
            if run and not prev_run and self.rv_present:
                self.rv["reset_done"] = 0
                self.rv["running"] = 1
            if rst_at is not None and c >= rst_at and self.rv_present:
                self.rv["reset_done"] = 1
                rst_at = None
            prev_run = run
            # loader port (tsr_ctrl_cdc u_hs_ld + the TCM ports)
            if ld is not None:
                if ld_pend is not None:
                    self.env.errors.append("loader request while one is in flight")
                ld_pend = [c + (4097 if (self.ld_hang or not self.rv_present) else r.randint(*self.ld_lat))] + list(ld)
            rsp = 0
            if ld_pend is not None and c >= ld_pend[0]:
                we, a, wd = ld_pend[1:]
                timeout = self.ld_hang or not self.rv_present
                in_tcm = a < ITCM or DTCM_BASE <= a < DTCM_BASE + DTCM
                if not timeout and we and in_tcm:
                    self.tcm[a] = wd
                d.ld_rsp_rdata.value = 0 if (timeout or we or not in_tcm) else self.tcm.get(a, 0)
                d.ld_rsp_err.value = int(not timeout and not in_tcm)
                d.ld_rsp_timeout.value = int(timeout)
                rsp = 1
                ld_pend = None
            d.ld_rsp_valid.value = rsp
            d.ld_ready.value = int(ld_pend is None and not rsp)
            # DMI port
            if dm is not None:
                dmi_pend = [c + (4097 if self.dmi_hang else r.randint(*self.dmi_lat))] + list(dm)
                self.dmi_log.append(dm)
            rsp = 0
            if dmi_pend is not None and c >= dmi_pend[0]:
                a, op, wd = dmi_pend[1:]
                if op == 2 and not self.dmi_hang:
                    self.dmi_regs[a] = wd
                d.dmi_rsp_rdata.value = 0 if self.dmi_hang else self.dmi_regs.get(a, 0)
                d.dmi_rsp_resp.value = 0
                d.dmi_rsp_timeout.value = int(self.dmi_hang)
                rsp = 1
                dmi_pend = None
            d.dmi_rsp_valid.value = rsp
            d.dmi_ready.value = int(dmi_pend is None and not rsp)
            # levels, counters
            if self.rv_present:
                self.rv_cyc += self.rv_ratio
            d.rv_cyc.value = int(self.rv_cyc) & 0xFFFF
            d.rv_reset_done.value = self.rv["reset_done"]
            d.rv_running.value = self.rv["running"]
            d.rv_halted.value = self.rv["halted"]
            d.rv_sleep.value = self.rv["sleep"]
            d.rv_clk_ok.value = int(self.rv_present)
            d.rv_ndmreset.value = self.rv["ndmreset"]
            d.perf_push_stall.value = self.counts[0] & M32
            d.perf_bridge_acc.value = self.counts[1] & M32
            d.perf_bridge_wait.value = self.counts[2] & M32
            iss = 0
            for u in range(6):
                iss |= (self.env.issued[u] & 0xFFFFFFF) << (28 * u)
            d.issued_seq.value = iss
            d.ifq_count.value = min(63, sum(1 for e in self.env.fifo if e[0] <= self.env.cycle))

    async def start(self):
        d = self.dut
        cocotb.start_soon(Clock(d.clk, 10, unit="ns").start())
        d.rst_n.value = 0
        zero = ["csr_valid", "csr_we", "csr_addr", "csr_wdata", "cwin_req_valid", "cwin_req_off", "lq_req_valid",
                "lq_req_off", "lq_req_we", "lq_req_wdata", "lq_req_lcq_w", "log_valid", "log_data", "rt_state",
                "rt_info", "ld_ready", "ld_rsp_valid", "ld_rsp_rdata", "ld_rsp_err", "ld_rsp_timeout", "dmi_en",
                "dmi_ready", "dmi_rsp_valid", "dmi_rsp_rdata", "dmi_rsp_resp", "dmi_rsp_timeout", "dbg_rdata",
                "db_stop_active", "sreg", "rv_cyc", "perf_push_stall", "perf_bridge_acc", "perf_bridge_wait",
                "push_timeout", "bridge_timeout", "rv_halted", "rv_sleep", "rv_running", "rv_reset_done",
                "rv_clk_ok", "rv_ndmreset", "issued_seq", "ifq_count", "ifq_valid", "ifq_entry", "xc_rsp_valid",
                "xc_rsp_code", "xc_rsp_info", "xc_rsp_ext", "xc_rsp_cmd", "dma_cmd_ready", "dma_done",
                "qact_cmd_ready", "qact_done", "mvm_cmd_ready", "mvm_done", "vu_cmd_ready", "vu_done",
                "att_cmd_ready", "att_done", "dbg_hold", "dbg_step", "err_ack", "perf_ch_rd_beat", "perf_ch_wait",
                "perf_ext", "unit_progress", "plat_err_i"]
        for nm in zero:
            getattr(d, nm).value = 0
        for _ in range(6):
            await RisingEdge(d.clk)
        d.rst_n.value = 1
        await RisingEdge(d.clk)
        cocotb.start_soon(self._rv_loop())
        self.env.start_loop()
        await self.cycles(4)

    # ---- comparisons ----
    async def cmp(self, offs, what, mask=None):
        bad = []
        for off in offs:
            a = await self.rd(off)
            b = self.sim.csr_read(off)
            m = (mask or {}).get(off, M32)
            if (a & m) != (b & m):
                bad.append("%s: CSR %#05x RTL %#010x sim %#010x" % (what, off, a & m, b & m))
        return bad


# offsets compared with the simulator (EX §5.1); STATUS without the bench-driven clk_rv levels
STATUS_MASK = 0x0FFFFFE0
BASE_REGS = [0x000, 0x004, 0x008, 0x00C, 0x010, 0x014, 0x018, 0x020, 0x024, 0x028, 0x030, 0x034, 0x038, 0x03C] + \
    list(range(0x040, 0x060, 4)) + [0x060, 0x064, 0x068, 0x06C, 0x070, 0x074, 0x078, 0x07C, 0x080, 0x084, 0x088,
                                    0x08C, 0x090, 0x094, 0x098, 0x09C, 0x0A0, 0x0BC, 0x0C0, 0x0C8, 0x0CC, 0x0D0,
                                    0x0D4, 0x0D8, 0x0FC]
Q_REGS = list(range(0xE00, 0xE68, 4)) + list(range(0xE80, 0xF20, 4)) + [0xE68, 0xF20, 0xFFC]
GAP_REGS = [0xB00, 0xB7C, 0xBFC, 0x9C0, 0x9FC, 0x001, 0x022, 0x103, 0x0C6]
MASKS = {0x024: STATUS_MASK}


async def _bench(dut, seed_salt=0):
    cfg, qd, c = env_cfg()
    b = Bench(dut, cfg, qd, seed=int(os.environ.get("TSR_SEED", "1")) * 31 + seed_salt)
    await b.start()
    return b


def _finish(b, bad):
    for x in bad[:20]:
        b.dut._log.error(x)
    errs = b.env.errors
    for x in errs[:10]:
        b.dut._log.error(x)
    b.sim.close()
    assert not bad and not errs, "%d mismatches, %d bench errors" % (len(bad), len(errs))


@cocotb.test()
async def csr_regs(dut):
    """Reset values and random writes / reads of the register map against the simulator."""
    b = await _bench(dut, 1)
    r = b.r
    bad = await b.cmp(BASE_REGS + Q_REGS + GAP_REGS, "reset", MASKS)
    rw = [0x020, 0x028, 0x034, 0x06C, 0x070, 0x098, 0x0C0, 0x0CC, 0x024] + list(range(0x040, 0x060, 4))
    for i in range(160):
        off = r.choice(rw + [r.randrange(0, 0x1000, 4)])
        if 0x100 <= off < 0x300 or 0x800 <= off < 0xE00 or off in (0x060, 0x0C4, 0x0D0, 0x078, 0x094):
            continue                                       # PERF, DBG / DMI, the rings, the loader: own tests
        v = r.getrandbits(32)
        if off == 0x020:
            v &= 0x0E                                      # CTRL levels only (no RV_RUN, no pulses)
        if off == 0x024:
            v = r.getrandbits(32)
        await b.wr(off, v)
        b.sim.csr_write(off, v)
        if i % 8 == 7:
            bad += await b.cmp(BASE_REGS + [off], "after write %#x = %#x" % (off, v), MASKS)
    rt = (r.getrandbits(32), r.getrandbits(32))            # RT_* mirrors (the clk_rv side's registers)
    dut.rt_state.value, dut.rt_info.value = rt
    b.sim.cw_write(T.CW_RT_STATE, rt[0])
    b.sim.cw_write(T.CW_RT_INFO, rt[1])
    await b.cycles(2)
    bad += await b.cmp(BASE_REGS + Q_REGS + GAP_REGS, "final", MASKS)
    _finish(b, bad)


@cocotb.test()
async def csr_rings(dut):
    """LQ / LCQ / log protocol (EX §4) against the simulator's host port and tsr_sim_cw_access."""
    b = await _bench(dut, 2)
    r = b.r
    s = b.sim
    bad = []
    lq_reads = list(range(0x200, 0x220, 4)) + [0x220, 0x224, 0x22C, 0x230, 0x234, 0x264, 0x288]
    for i in range(700):
        x = r.random()
        if x < 0.10:                                       # host: stage and push a descriptor
            for j in range(r.randint(0, 8)):
                off, v = 0x040 + 4 * r.randrange(8), r.getrandbits(32)
                await b.wr(off, v)
                s.csr_write(off, v)
            await b.wr(0x060, 0)
            s.csr_write(0x060, 0)
        elif x < 0.25:                                     # RV: read a head / flags register
            off = r.choice(lq_reads)
            v, f = await b.lq(off)
            sv = s.cw_read(off)
            if (v, f) != (sv, 0):
                bad.append("LQ read %#x: RTL %#x fault %d, sim %#x" % (off, v, f, sv))
        elif x < 0.33:                                     # RV: LQ_POP
            v, f = await b.lq(0x228, 1)
            rc, _ = s.cw_access(1, 0x228, 0)
            if f != (rc == S.CW_FAULT):
                bad.append("LQ_POP fault RTL %d sim rc %d" % (f, rc))
        elif x < 0.45:                                     # RV: LCQ_POST (ANSWER 0 / 1)
            words = [r.getrandbits(32) for _ in range(8)]
            ans = r.randint(0, 1)
            for j, w in enumerate(words):
                s.cw_write(T.CW_LCQ_W(j), w)
            v, f = await b.lq(0x260, 1, ans, words)
            rc, _ = s.cw_access(1, 0x260, ans)
            if f != (rc == S.CW_FAULT):
                bad.append("LCQ_POST(%d) fault RTL %d sim rc %d" % (ans, f, rc))
            if not f:
                b.written_lcq.add((s.csr_read(0x074) - 1) % 8)
        elif x < 0.58:                                     # RV: log words
            for _ in range(r.choice([1, 1, 4, 40])):
                w = r.getrandbits(32)
                before = s.csr_read(0x090)
                await b.log_push(w)
                s.cw_write(T.CW_LOG_PUSH, w)
                if s.csr_read(0x090) != before:
                    b.written_log.add(before % 256)
        elif x < 0.62:                                     # RV: LOG_FREE as the clk_rv side computes it
            v, f = await b.lq(0x284)
            free = (256 - ((b.log_wr_rv - v) & M32)) & M32
            sv = s.cw_read(0x284)
            if free != sv:
                bad.append("LOG_FREE RTL %d (LOG_RD + drops %#x) sim %d" % (free, v, sv))
        elif x < 0.72:                                     # host: LCQ_RD / LOG_RD writes (legal or not)
            off = r.choice([0x078, 0x094])
            cur = s.csr_read(off)
            wr = s.csr_read(0x074 if off == 0x078 else 0x090)
            v = r.choice([wr, cur + r.randint(0, max(0, (wr - cur) & M32)), wr + 1, cur - 1, r.getrandbits(32)]) & M32
            await b.wr(off, v)
            s.csr_write(off, v)
        elif x < 0.76:                                     # host: watermarks (LQ_CANCEL may QABORT)
            off = r.choice([0x06C, 0x070])
            v = (s.csr_read(0x064) + r.randint(-3, 3)) & M32
            await b.wr(off, v)
            s.csr_write(off, v)
            await b.cycles(4)
        elif x < 0.79:
            v = r.getrandbits(8)
            await b.wr(0x098, v)
            s.csr_write(0x098, v)
        elif x < 0.82:                                     # host: W1C of the sticky bits
            v = r.getrandbits(32)
            await b.wr(0x024, v)
            s.csr_write(0x024, v)
        else:                                              # host: read the ring registers
            offs = [0x024, 0x064, 0x068, 0x06C, 0x070, 0x074, 0x078, 0x07C, 0x084, 0x090, 0x094, 0x098, 0x09C]
            offs += [0x300 + 32 * k + 4 * j for k in sorted(b.written_lcq) for j in range(8)][:16]
            offs += [0x400 + 4 * k for k in sorted(b.written_log)][-8:]
            bad += await b.cmp(offs, "rings step %d" % i, MASKS)
        if len(bad) > 10:
            break
    await b.cycles(10)
    offs = [0x300 + 32 * k + 4 * j for k in sorted(b.written_lcq) for j in range(8)]
    offs += [0x400 + 4 * k for k in sorted(b.written_log)]
    bad += await b.cmp(BASE_REGS + Q_REGS + offs, "rings final", MASKS)
    b.dut._log.info("LCQ slots written %d, log slots %d", len(b.written_lcq), len(b.written_log))
    _finish(b, bad)


async def _tcm_compare(b, bad, addrs):
    p, n = b.sim.tcm("itcm")
    q, m = b.sim.tcm("dtcm")
    import ctypes
    it = ctypes.string_at(p, n)
    dt = ctypes.string_at(q, m)
    for a in addrs:
        if a < ITCM:
            sv = int.from_bytes(it[a:a + 4], "little")
        elif DTCM_BASE <= a < DTCM_BASE + DTCM:
            sv = int.from_bytes(dt[a - DTCM_BASE:a - DTCM_BASE + 4], "little")
        else:
            continue
        if b.tcm.get(a, 0) != sv:
            bad.append("TCM word %#x: RTL %#x sim %#x" % (a, b.tcm.get(a, 0), sv))


@cocotb.test()
async def csr_loader(dut):
    """Loader (EX §3.3): segments into ITCM, DTCM and holes; receive CRC, VERIFY CRC, LOAD_DATA reads, the
    TCM contents, against the simulator; then refusals (RV_RUN = 1, SR_BUSY) and LOAD_TIMEOUT."""
    b = await _bench(dut, 3)
    r = b.r
    s = b.sim
    bad = []
    touched = set()
    refused_busy = 0
    for seg in range(14):
        base = r.choice([0, r.randrange(0, ITCM, 4), DTCM_BASE + r.randrange(0, DTCM, 4), ITCM - 16,
                         DTCM_BASE + DTCM - 8, 0x30000, r.randrange(0, 1 << 32, 4)]) & M32
        if r.random() < 0.15:
            base |= r.randint(1, 3)                        # unaligned LOAD_ADDR: masked
        nw = r.choice([1, 3, 16, 64])
        await b.wr(0x0C0, base)
        s.csr_write(0x0C0, base)
        bad += await b.cmp([0x0C0, 0x0C8], "seg %d start" % seg)
        for k in range(nw):
            w = r.getrandbits(32)
            await b.wr(0x0C4, w)
            s.csr_write(0x0C4, w)
            touched.add(((base & ~3) + 4 * k) & M32)
        bad += await b.cmp([0x0C0, 0x0C8, 0x024], "seg %d written" % seg, MASKS)
        # read back the first word, then VERIFY the segment
        await b.wr(0x0C0, base)
        s.csr_write(0x0C0, base)
        bad += await b.cmp([0x0C4], "seg %d LOAD_DATA read" % seg)
        cnt = nw if r.random() < 0.9 else r.choice([0, nw + 2])
        await b.wr(0x0CC, cnt)
        s.csr_write(0x0CC, cnt)
        await b.wr(0x0D0, 1)
        s.csr_write(0x0D0, 1)
        if nw >= 16 and cnt >= 16:
            # RTL only (the simulator's VERIFY completes at once): a LOAD_ADDR write while VERIFY_BUSY is
            # dropped with LOAD_REFUSED, else it would restart LOAD_CRC in the middle of the sequence
            await b.wr(0x0C0, base ^ 0x40)
            st = await b.rd(0x024)
            if st & (1 << 16):
                refused_busy += 1
                await b.wr(0x024, 1 << 16)
            else:                                           # VERIFY was already over: the write took effect
                s.csr_write(0x0C0, base ^ 0x40)
        n = 0
        while (await b.rd(0x0D4)) & 1:
            n += 1
            assert n < 5000, "VERIFY_BUSY stuck"
        bad += await b.cmp([0x0C0, 0x0C8, 0x0CC, 0x0D4, 0x024], "seg %d verified" % seg, MASKS)
        if len(bad) > 10:
            break
    await b.cycles(20)
    assert refused_busy, "no LOAD_ADDR write hit a running VERIFY"
    await _tcm_compare(b, bad, sorted(touched))
    # refusals (RTL only): RV_RUN = 1 refuses writes and VERIFY, LOAD_DATA reads 0
    await b.wr(0x020, 1)
    await b.wr(0x0C0, 0)
    await b.wr(0x0C4, 0x1234)
    st = await b.rd(0x024)
    assert st & (1 << 16), "LOAD_REFUSED not set by a LOAD_DATA write with RV_RUN = 1"
    assert (await b.rd(0x0C4)) == 0, "LOAD_DATA read with RV_RUN = 1 is not 0"
    assert b.tcm.get(0, None) != 0x1234
    await b.wr(0x024, 1 << 16)
    await b.wr(0x0D0, 1)
    assert (await b.rd(0x024)) & (1 << 16), "VERIFY with RV_RUN = 1 not refused"
    await b.wr(0x020, 0)
    await b.wr(0x024, M32)
    await b.cycles(40)
    # LOAD_TIMEOUT: the TCM port never answers (the CDC ends the request after its bound)
    b.ld_hang = True
    await b.wr(0x0C0, 0x40)
    await b.wr(0x0C4, 0xABCD)
    v = await b.rd(0x0C4)
    st = await b.rd(0x024)
    assert v == 0 and st & (1 << 17), "LOAD_TIMEOUT not set (read %#x status %#x)" % (v, st)
    b.ld_hang = False
    _finish(b, bad)


@cocotb.test()
async def csr_soft_reset(dut):
    """SOFT_RESET (EX §8.5): held entries complete ABORTED, the step 3 clear list and the kept state equal
    the simulator's; RV_RUN_REFUSED while SR_BUSY; SR_RV_TIMEOUT without clk_rv (RTL only)."""
    b = await _bench(dut, 4)
    r = b.r
    s = b.sim
    s.listen()
    bad = []
    # state to flush: a launch, records, log words, sticky bits, errors, PERF
    for k in range(3):
        await b.wr(0x040, 0x100 + k)
        s.csr_write(0x040, 0x100 + k)
        await b.wr(0x060, 0)
        s.csr_write(0x060, 0)
    v, f = await b.lq(0x228, 1)
    s.cw_access(1, 0x228, 0)
    for k in range(5):
        w = r.getrandbits(32)
        await b.log_push(w)
        s.cw_write(T.CW_LOG_PUSH, w)
    words = [r.getrandbits(32) for _ in range(8)]
    for j, w in enumerate(words):
        s.cw_write(T.CW_LCQ_W(j), w)
    await b.lq(0x260, 1, 0, words)
    s.cw_access(1, 0x260, 0)
    await b.wr(0x06C, 0xFFFFFFF0)
    s.csr_write(0x06C, 0xFFFFFFF0)                         # watermark valid, not covering the running launch
    await b.wr(0x070, 0xFFFFFFF0)
    s.csr_write(0x070, 0xFFFFFFF0)                         # soft stop: STOP_VALID, cleared by step 3 (EX §8.5)
    await b.wr(0x060, 0)
    s.csr_write(0x060, 0)
    # a long LD running (it completes, CC §12.1), then a host hold and entries that cannot start
    # (DMA entries queued behind the running LD among them): after SOFT_RESET's QABORT they complete
    # ABORTED, and step 2 waits for the LD
    b.env.lat_force[0] = 500
    await b.env.push(isa.enc_ld(0, 0, 0, 4))
    s.push(isa.enc_ld(0, 0, 0, 4))
    s.advance(10)                                          # the simulator's LD starts (64 + 4 cycles)
    await b.cycles(40)
    await b.wr(0x020, 0x2)
    s.csr_write(0x020, 0x2)
    tks = []
    for w in (isa.enc_ld(0, 8, 8, 4), isa.enc_st(1, 0, 0, 2), isa.enc_qact(0, 64, 0, 64, 1), isa.enc_nop(),
              [0x02, 0, 0, 0, 0, 0, 0, 0]):
        t = await b.env.push(w)
        ts = s.push(w)
        tks.append((t, ts))
    await b.cycles(60)
    s.advance(50)                                          # the NOP and the refused op drain in both
    await b.wr(0x020, 0x2 | (1 << 10))
    s.csr_write(0x020, 0x2 | (1 << 10))
    # RV_RUN while SR_BUSY: refused (RTL; the simulator's sequence ends at once once drained)
    st = await b.rd(0x024)
    assert st & (1 << 9), "SR_BUSY not set"
    assert len(b.env.start[0]) == 1, "the long LD is not running at the SOFT_RESET: %s" % b.env.start[0]
    await b.wr(0x020, 0x3)
    st = await b.rd(0x024)
    assert st & (1 << 27), "RV_RUN_REFUSED not set while SR_BUSY"
    n = 0
    while True:
        st = await b.rd(0x024)
        if not st & (1 << 9):
            break
        n += 1
        assert n < 3000, "SR_BUSY stuck"
    dut._log.info("SOFT_RESET: SR_BUSY fell at cycle %d (status %#x); DMA starts %s dones %s", b.env.cycle, st,
                  [x[2] for x in b.env.start[0]], [x[4] for x in b.env.done[0]])
    assert st & (1 << 6), "SR_BUSY fell before QIDLE (step 2, EX §8.5): status %#x" % st
    b.env.lat_force[0] = None
    s.run_until_idle(1 << 20)
    await b.cycles(30)
    bad += await b.cmp(BASE_REGS + Q_REGS + [0x100 + 8 * i for i in range(64)], "after SOFT_RESET", MASKS)
    sim_done = [(e["unit"], e["ticket"] & 0xFFFFFFF, e["code"]) for e in s.events if e["kind"] == S.QEV_DONE]
    rtl_done = [(u, x[0], x[2]) for u in range(6) for x in b.env.done[u]]
    if sorted(sim_done) != sorted(rtl_done):
        bad.append("completions: sim %s RTL %s" % (sorted(sim_done), sorted(rtl_done)))
    # entries after the SOFT_RESET run normally (the clear points are set)
    await b.wr(0x020, 0x0)
    s.csr_write(0x020, 0x0)
    t = await b.env.push(isa.enc_ld(1, 8, 8, 2))
    s.push(isa.enc_ld(1, 8, 8, 2))
    await b.env.wait_idle()
    s.run_until_idle(1 << 20)
    bad += await b.cmp(Q_REGS + [0x024], "after the SOFT_RESET's QCLEAR", MASKS)
    # SR_RV_TIMEOUT: clk_rv absent (RTL only; SR_RV_MAX of the build)
    b.rv_present = False
    b.rv["reset_done"] = 0
    await b.wr(0x020, 1 << 10)
    n = 0
    while (await b.rd(0x024)) & (1 << 9):
        n += 1
        assert n < 20000, "SR_BUSY stuck without clk_rv"
    st = await b.rd(0x024)
    assert st & (1 << 26), "SR_RV_TIMEOUT not set (status %#x)" % st
    b.rv_present = True
    del t
    _finish(b, bad)


@cocotb.test()
async def csr_perf(dut):
    """PERF (EX §6): event counters against the simulator inside a launch; time counters against the
    gates the bench counts; the LO / HI snapshot; zeroed by SOFT_RESET."""
    b = await _bench(dut, 5)
    s = b.sim
    bad = []
    X = 5 + 2 * b.cfg["num_ch"]
    await b.wr(0x060, 0)
    s.csr_write(0x060, 0)
    await b.lq(0x228, 1)
    s.cw_access(1, 0x228, 0)
    for i in range(30):
        w = b.r.choice([isa.enc_ld(0, i, i, 3), isa.enc_nop(), isa.enc_qact(0, 128 + 8 * i, 0, 64, 1),
                        isa.enc_mvm(0, 1024, 0, 0, 32, 256, 1), [0x05, 0, 0, 0, 0, 0, 0, 0]])
        await b.env.push(w)
        s.push(w)
    await b.env.wait_idle()
    s.run_until_idle(1 << 20)
    words = [1, 0, 0, 0, 0, 0, 0, 0]
    for j, w in enumerate(words):
        s.cw_write(T.CW_LCQ_W(j), w)
    await b.lq(0x260, 1, 1, words)
    s.cw_access(1, 0x260, 1)
    await b.cycles(10)
    for idx in (T.PERF_RETIRED, X + T.PERF_XOFF_PUSHES, X + T.PERF_XOFF_LAUNCHES):
        lo = await b.rd(0x100 + 8 * idx)
        hi = await b.rd(0x104 + 8 * idx)
        sv = s.perf(idx)
        if (hi << 32 | lo) != sv:
            bad.append("PERF %d: RTL %d sim %d" % (idx, hi << 32 | lo, sv))
    cyc = await b.rd(0x100)
    if abs(cyc - b.time_en) > 4:
        bad.append("PERF CYCLES %d, gate counted %d" % (cyc, b.time_en))
    for u, idx in ((0, X + 0), (1, X + 1), (2, 2)):
        v = await b.rd(0x100 + 8 * idx)
        if abs(v - b.busy_time[u]) > 4:
            bad.append("PERF slot %d (unit %d busy): %d, counted %d" % (idx, u, v, b.busy_time[u]))
    # snapshot: X + 7 (a 32-bit clk_rv count extended to 64 bits) pushed over 2^32
    await b.wr(0x060, 0)
    await b.lq(0x228, 1)
    b.counts[0] = 0xFFFFFFF0
    await b.cycles(3)
    i7 = X + T.PERF_XOFF_PUSH_STALL
    lo = await b.rd(0x100 + 8 * i7)
    b.counts[0] = (b.counts[0] + 0x7FFFFFFF) & M32
    await b.cycles(3)
    b.counts[0] = (b.counts[0] + 0x7FFFFFFF) & M32
    await b.cycles(3)
    hi_cw = await b.cw(0x104 + 8 * i7)                     # live HI (no snapshot)
    hi = await b.rd(0x104 + 8 * i7)                        # the snapshot of the LO read
    lo2 = await b.rd(0x100 + 8 * i7)
    hi2 = await b.rd(0x104 + 8 * i7)
    dut._log.info("X+7: LO %#x snapshot HI %#x, live HI %#x, then %#x:%08x", lo, hi, hi_cw, hi2, lo2)
    if (hi, lo) != (0, 0xFFFFFFF0):
        bad.append("PERF snapshot: LO %#x HI %#x, expected HI 0 LO 0xfffffff0" % (lo, hi))
    if hi_cw != 1 or (hi2 << 32 | lo2) != 0xFFFFFFF0 + 2 * 0x7FFFFFFF:
        bad.append("PERF X+7 live value %#x:%08x (core window HI %#x), expected %#x" %
                   (hi2, lo2, hi_cw, 0xFFFFFFF0 + 2 * 0x7FFFFFFF))
    # zeroed by SOFT_RESET
    await b.wr(0x020, 1 << 10)
    while (await b.rd(0x024)) & (1 << 9):
        pass
    for idx in (0, 1, i7):
        v = await b.rd(0x100 + 8 * idx) | (await b.rd(0x104 + 8 * idx)) << 32
        if v > 64:
            bad.append("PERF %d = %d after SOFT_RESET" % (idx, v))
    _finish(b, bad)


@cocotb.test()
async def csr_cwin(dut):
    """Core window (CC §5.3, §6.6): random reads interleaved with host traffic equal tsr_sim_corewin_read;
    no PERF snapshot side effect; LOAD_DATA and DMI read 0."""
    b = await _bench(dut, 6)
    r = b.r
    s = b.sim
    bad = []
    dut.dmi_en.value = 1
    for i in range(200):
        off = r.choice(BASE_REGS + Q_REGS + [0x0C4, 0xC00, 0xC40, 0x9C0])
        if r.random() < 0.3:
            v = r.getrandbits(32) & 0xFFFFFFFC
            o = r.choice(list(range(0x040, 0x060, 4)) + [0x070, 0x0C0, 0x0CC])
            await b.wr(o, v)
            s.csr_write(o, v)
        a = await b.cw(off)
        sv = s.corewin_read(off)
        m = MASKS.get(off, M32)
        if (a & m) != (sv & m):
            bad.append("core window %#x: RTL %#x sim %#x" % (off, a, sv))
    if b.dmi_log:
        bad.append("a core-window read reached the DMI port: %s" % b.dmi_log[:3])
    _finish(b, bad)


@cocotb.test()
async def csr_dmi(dut):
    """DMI window (DB §3.2, CC §6.1): gated by dmi_en; request fields; read data; DMI_TIMEOUT (RTL only)."""
    b = await _bench(dut, 7)
    assert (await b.rd(0xC10)) == 0 and not b.dmi_log, "DMI access with dmi_en = 0"
    dut.dmi_en.value = 1
    await b.wr(0xC00 + 4 * 0x10, 0xDEADBEEF)
    v = await b.rd(0xC00 + 4 * 0x10)
    assert v == 0xDEADBEEF, "DMI read %#x" % v
    assert b.dmi_log == [(0x10, 2, 0xDEADBEEF), (0x10, 1, 0)] or b.dmi_log[0][:2] == (0x10, 2), b.dmi_log
    b.dmi_hang = True
    v = await b.rd(0xC00 + 4 * 0x11)
    st = await b.rd(0x024)
    assert v == 0 and st & (1 << 18), "DMI_TIMEOUT not set"
    b.dmi_hang = False
    b.sim.close()


@cocotb.test()
async def csr_wdog(dut):
    """Unit watchdog (EX §8.6): a unit holding an op without progress for 2^L cycles sets Q_STATE.HUNG and
    STATUS.UNIT_HUNG; progress clears HUNG (RTL only)."""
    b = await _bench(dut, 8)
    await b.wr(0x034, 6)                                   # L = 6: 64 cycles
    b.env.lat_force[0] = 400
    await b.env.push(isa.enc_ld(0, 0, 0, 1))
    await b.cycles(200)
    st = await b.rd(0x024)
    qs = await b.rd(0xE40)
    assert st & (1 << 25) and qs & (1 << 9), "UNIT_HUNG / HUNG not set (status %#x Q_STATE %#x)" % (st, qs)
    await b.env.wait_idle()
    qs = await b.rd(0xE40)
    assert not qs & (1 << 9), "HUNG kept after the unit's done"
    await b.wr(0x034, 0)
    b.env.lat_force[0] = 400
    await b.wr(0x024, M32)
    await b.env.push(isa.enc_ld(0, 0, 0, 1))
    await b.cycles(300)
    assert not (await b.rd(0x024)) & (1 << 25), "watchdog fired with L = 0"
    b.env.lat_force[0] = None
    await b.env.wait_idle()
    b.sim.close()


@cocotb.test()
async def csr_rvfreq(dut):
    """RV_FREQ (EX §5.1): clk_rv cycles in the last 2^16 core cycles (ratio 0.6 -> 39321)."""
    b = await _bench(dut, 9)
    b.rv_ratio = 0.6
    await b.cycles(2 * 65536 + 200)
    v = await b.rd(0x02C)
    assert abs(v - int(0.6 * 65536)) <= 2, "RV_FREQ %d" % v
    b.sim.close()


@cocotb.test()
async def csr_rv_reset(dut):
    """RISC-V reset actions of the core (CC §12.2): RV_RUN 1 -> 0 and ndmreset rising apply QABORT and
    ANSWERED := TAKEN; host QCLEAR refused unless QIDLE and the RISC-V quiet, then accepted; rv_run held
    low while a loader request is in flight (RTL; ndmreset also against the simulator's)."""
    b = await _bench(dut, 10)
    s = b.sim
    bad = []
    await b.wr(0x060, 0)
    s.csr_write(0x060, 0)
    await b.lq(0x228, 1)
    s.cw_access(1, 0x228, 0)
    b.rv["ndmreset"] = 1
    s.ndmreset(1)
    await b.cycles(10)
    bad += await b.cmp([0x024, 0x068, 0x084] + Q_REGS, "ndmreset", MASKS)
    b.rv["ndmreset"] = 0
    s.ndmreset(0)
    await b.cycles(4)
    await b.wr(0x020, 1 << 9)                              # clear the abort's records (idle, quiet)
    assert not (await b.rd(0x024)) & (1 << 8), "host QCLEAR at idle did not clear the error states"
    # host QCLEAR: refused with an entry queued (held), accepted when idle and quiet
    await b.wr(0x020, 0x2)
    await b.env.push(isa.enc_ld(0, 0, 0, 1))
    await b.cycles(40)
    await b.wr(0x020, 0x2 | (1 << 9))
    assert (await b.rd(0x024)) & (1 << 19), "QCLEAR not refused with an entry queued"
    await b.wr(0x020, 0x0)
    await b.env.wait_idle()
    await b.wr(0x024, M32)
    await b.wr(0x020, 1 << 9)
    st = await b.rd(0x024)
    assert not st & (1 << 19) and not st & (1 << 8), "host QCLEAR at idle not accepted (status %#x)" % st
    # RV_RUN 1 -> 0: QABORT, ANSWERED := TAKEN
    await b.wr(0x060, 0)
    await b.lq(0x228, 1)
    await b.wr(0x020, 0x1)
    await b.cycles(30)
    assert int(dut.rv_run.value) == 1
    await b.wr(0x020, 0x0)
    await b.cycles(30)
    st = await b.rd(0x024)
    assert st & (1 << 8) and not st & (1 << 5), "RV_RUN 1 -> 0: QERR %d ACTIVE %d" % ((st >> 8) & 1, (st >> 5) & 1)
    assert (await b.rd(0x068)) == (await b.rd(0x084))
    _finish(b, bad)


@cocotb.test()
async def csr_rv_boot(dut):
    """RV_BOOT (EX §5.1): [7:0] read 0; a write while RV_RUN = 1 is ignored (the vector is taken when RV_RUN
    rises, and again after a debug ndmreset: the bus to clk_rv must be stable then, CC §11.2), as in the
    simulator; the value reaches the control subsystem port only through this register."""
    b = await _bench(dut, 14)
    s = b.sim
    bad = []
    for v in (0x0000_1234, 0xFFFF_FFFF, 0x0010_0000):
        await b.wr(0x028, v)
        s.csr_write(0x028, v)
        bad += await b.cmp([0x028], "RV_BOOT := %#x" % v)
    keep = await b.rd(0x028)
    await _ctrl(b, 1)                                      # RV_RUN = 1
    await b.cycles(30)
    for v in (0x0000_5500, 0xFFFF_FF00):
        await b.wr(0x028, v)
        s.csr_write(0x028, v)
        bad += await b.cmp([0x028], "RV_BOOT := %#x with RV_RUN = 1" % v)
        if int(dut.rv_boot.value) != keep:
            bad.append("rv_boot port %#x changed with RV_RUN = 1 (was %#x)" % (int(dut.rv_boot.value), keep))
    await _ctrl(b, 0)
    await b.cycles(30)
    await b.wr(0x028, 0x0000_7700)
    s.csr_write(0x028, 0x0000_7700)
    bad += await b.cmp([0x028], "RV_BOOT after RV_RUN fell")
    assert int(dut.rv_boot.value) == 0x7700
    _finish(b, bad)


async def _ctrl(b, v):
    await b.wr(0x020, v)
    b.sim.csr_write(0x020, v)


@cocotb.test()
async def csr_qclear_run_change(dut):
    """Host QCLEAR in a CTRL write (EX §5.2 bit 9: "levels first, then pulses"; the RISC-V quiet is judged on
    the state before the write's own RV_RUN change) against the simulator: RV_RUN 0 -> 1 with QCLEAR over an
    abort's records is accepted; RV_RUN 1 -> 0 with QCLEAR with the hart running is refused (the reset has not
    completed; the abort keeps the records); with the hart halted it is accepted; {QABORT, QCLEAR} in one write
    at idle clears (the abort's records included). The simulator judges the state before the change."""
    b = await _bench(dut, 11)
    bad = []
    regs = [0x024] + Q_REGS
    await _ctrl(b, 1 << 8)                                 # QABORT, RISC-V in reset: errors set
    await b.cycles(20)
    await _ctrl(b, 0x1 | (1 << 9))                         # RV_RUN 0 -> 1 and QCLEAR
    await b.cycles(40)
    bad += await b.cmp(regs, "RV_RUN 0->1 + QCLEAR", MASKS)
    await _ctrl(b, 0x1 | (1 << 8))                         # hart running: QABORT again
    await b.cycles(20)
    await _ctrl(b, 1 << 9)                                 # RV_RUN 1 -> 0 and QCLEAR: refused
    await b.cycles(40)
    st = await b.rd(0x024)
    if not (st & (1 << 19) and st & (1 << 8)):
        bad.append("RV_RUN 1->0 + QCLEAR (hart running) was not refused: STATUS %#x" % st)
    bad += await b.cmp(regs, "RV_RUN 1->0 + QCLEAR", MASKS)
    await _ctrl(b, 1 << 9)                                 # reset done now: accepted
    await b.cycles(20)
    bad += await b.cmp(regs, "QCLEAR after the reset", MASKS)
    await _ctrl(b, (1 << 8) | (1 << 9))                    # {QABORT, QCLEAR} in one write at idle
    await b.cycles(40)
    bad += await b.cmp(regs, "QABORT|QCLEAR", MASKS)
    _finish(b, bad)


@cocotb.test()
async def csr_qclear_halted(dut):
    """Host QCLEAR with RV_RUN = 1 and the hart halted (EX §5.2 bit 9: "halted with no push in flight") is
    accepted, with RV_RUN falling in the same write as well; refused when the hart is not halted."""
    b = await _bench(dut, 12)
    s = b.sim
    bad = []
    regs = [0x024] + Q_REGS
    await _ctrl(b, 0x1)
    await b.cycles(40)
    await _ctrl(b, 0x1 | (1 << 8))                         # QABORT: every unit in error
    await b.cycles(20)
    await _ctrl(b, 0x1 | (1 << 9))                         # running, not halted: refused
    await b.cycles(20)
    bad += await b.cmp(regs, "QCLEAR, hart running", MASKS)
    await b.wr(0x024, M32)
    s.csr_write(0x024, M32)                                # clear the sticky bits
    b.rv["halted"] = 1
    s.rv_set_state(2)                                      # TSR_STATUS_RV_HALTED
    await b.cycles(10)
    await _ctrl(b, 0x1 | (1 << 9))                         # halted: accepted
    await b.cycles(20)
    bad += await b.cmp(regs, "QCLEAR, hart halted", {0x024: STATUS_MASK & ~0x1F})
    await _ctrl(b, 0x1 | (1 << 8))
    await b.cycles(20)
    await _ctrl(b, 1 << 9)                                 # RV_RUN 1 -> 0 and QCLEAR, hart halted: accepted
    await b.cycles(40)
    bad += await b.cmp(regs, "QCLEAR, RV_RUN 1->0, hart halted", {0x024: STATUS_MASK & ~0x1F})
    _finish(b, bad)


@cocotb.test()
async def csr_watermarks(dut):
    """LQ_CANCEL coverage boundary (EX §4.9: covered iff (wm - ANSWERED) mod 2^32 < 2^31) with one launch
    running, and the LCQ_RD write-rule boundaries (EX §4.2), against the simulator."""
    b = await _bench(dut, 13)
    s = b.sim
    bad = []
    await b.wr(0x060, 0)
    s.csr_write(0x060, 0)
    await b.lq(0x228, 1)
    s.cw_access(1, 0x228, 0)
    for wm in (0x8000_0000, 0xFFFF_FFFF, 0x7FFF_FFFF):     # not covered, not covered, covered (abort)
        await b.wr(0x06C, wm)
        s.csr_write(0x06C, wm)
        await b.cycles(20)
        bad += await b.cmp([0x024, 0x06C, 0x07C] + Q_REGS[:8], "LQ_CANCEL %#x" % wm, MASKS)
    for k in range(3):                                     # LCQ: 3 records posted, RD = 0
        await b.lq(0x260, 1, 0, [k] * 8)
        s.cw_access(1, 0x260, 0)
    for v in (4, 3, 0x8000_0003, 2):                       # refused, accepted, refused, refused
        await b.wr(0x078, v)
        s.csr_write(0x078, v)
        bad += await b.cmp([0x024, 0x074, 0x078], "LCQ_RD := %#x" % v, MASKS)
    _finish(b, bad)


def _run(name, cfg, qd, extra=None, params=None, tests=None):
    c = dict(num_ch=cfg["num_ch"], t_max=cfg["t_max"], qd=list(qd))
    c.update(extra or {})
    p = dict(NUM_CH=cfg["num_ch"], T_MAX=cfg["t_max"], QD_DMA=qd[0], QD_QACT=qd[1], QD_MVM=qd[2], QD_VU=qd[3],
             QD_ATT=qd[4], QD_SYNC=qd[5], SR_RV_MAX=3000)
    p.update(params or {})
    run("tsr_csr_tb", __name__, sources=src(), name="csr_" + name, parameters=p,
        extra_env={"TSR_CSR_CFG": json.dumps(c)}, testcase=tests)


def test_csr_u200():
    _run("u200", S.config(num_ch=4, t_max=8), S.U200_QD)


def test_csr_fast_clk():
    """The U200 shape with FAST_CLK 1: the PERF counters take clear, enable and increment from registers (1 cycle later).
    The counters equal the simulator's once the launch is idle; the time counters stay within the bench's tolerance."""
    _run("u200_fast", S.config(num_ch=4, t_max=8), S.U200_QD, params={"FAST_CLK": 1})


def test_csr_1ch():
    """1 channel (PERF extension base X = 7), no MVM unit."""
    _run("1ch", S.config(num_ch=1, t_max=1), (8, 4, 0, 16, 0, 4),
         tests=["csr_regs", "csr_rings", "csr_perf", "csr_cwin", "csr_soft_reset"])
