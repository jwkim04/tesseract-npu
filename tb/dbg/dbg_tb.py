"""cocotb environment of tb/dbg: the DB block (tsr_dbg, tsr_dbg_rv, tsr_dbg_wrmux) with tsr_csr and
tsr_issue (tb/dbg/sv/tsr_dbg_tb.sv) against the simulator's DBG model (sw/sim/tsr_dbg.c) and DM
model (sw/sim/tsr_dm.c) through libtsr_sim.so (docs/design/05-debug.md "DB" §9.1, §14 items 1-2, 6).

  DbgSim   the simulator with its DBG model (attached by tsr_sim_create_opts): steps, CSR reads, the
           trace ring in its DRAM, REC records of its control window
  Bench    the RTL side: the host CSR port, tb/issue's environment (RISC-V side of the issue FIFO,
           unit models, monitors), the clk_rv side of tsr_ctrl_cdc (as tb/csr's bench, the DMI port
           answered by a second simulator's DM model), tsr_ctrl_rv's DB halt-request register (pend,
           DB §3.3), the clk_rv DB window (REC / DBG_FLAGS), the memory behind channel TRW_CH's write
           source (random ready / latency, in-order responses) and an optional DMA write stream on
           the write mux
  Runner   the same step list on both sides with DB's stop semantics: a push without credit while a
           stop is active waits for DRAINED; if the credit does not come back, the stream is blocked
           at that step (both sides block at the same step: the credits at DRAINED depend only on
           the push-order prefix, DB §5.1); the test then compares and acts (QSTEP, RELEASE, ...)
Comparisons are cycle-agnostic: register values (model values and times masked), the per-unit
completion streams, history entries (TIME excluded), trace records per kind and per unit (times
excluded; RTL times checked against the RTL's own strobes), REC records per unit.
"""
from __future__ import annotations

import ctypes
import random
from collections import deque

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge, Timer

from common import isa, tables as T

import issue_sim as S
import issue_tb as E

M32 = 0xFFFFFFFF
HOOK_PC = 0xFFFFFFFD               # the simulator's push PC of its test hooks (TSR_SIM_PC_HOOK)
TK_NONE = S.TICKET_NONE
SEQ_MASK = E.SEQ_MASK
u32, vp = ctypes.c_uint32, ctypes.c_void_p

ST_STOP_ACTIVE = T.DBG_STATUS_STOP_ACTIVE
ST_DRAINED = T.DBG_STATUS_DRAINED
ST_STEP_BUSY = T.DBG_STATUS_STEP_BUSY
# DBG_STATUS bits that are not the stop / record state: RVWP_BUSY (the RTL's handshake in flight)
STATUS_MASK = M32 & ~T.DBG_STATUS_RVWP_BUSY

_EXTRA = False


def lib():
    global _EXTRA
    L = S.lib()
    if not _EXTRA:
        for name, res, args in (("tsr_sim_dram", vp, [vp, u32]), ("tsr_sim_dm", vp, [vp]),
                                ("tsr_dm_dmi_read", u32, [vp, u32]), ("tsr_dm_dmi_write", None, [vp, u32, u32])):
            f = getattr(L, name)
            f.restype = res
            f.argtypes = args
        _EXTRA = True
    return L


def resolve_dep(d, tickets, unit, issued):
    if d[0] == "none":
        return TK_NONE
    if d[0] == "t":
        return tickets[d[1]]
    if d[0] == "rel":
        return (d[1] << 28) | (d[2] & SEQ_MASK)
    if d[0] == "own":
        return (unit << 28) | issued[unit]
    if d[0] == "future":
        return (d[1] << 28) | ((issued[d[1]] + 1) & SEQ_MASK)
    raise ValueError(d)


def program(gen, n_ops, marks=True, qclears=True, rec_p=0.2, launch_end=False):
    """A schedule-independent push stream without holds, aborts or idle points (tb/issue's Gen ops:
    errors, BAD_DEP, word masks, QCLEAR barriers), optionally with markers (one LAUNCH_END)."""
    r = gen.r
    while len([s for s in gen.steps if s[0] == "push"]) < n_ops:
        x = r.random()
        if x < 0.04 and qclears:
            gen.qclear()
        elif x < 0.07 and marks:
            v = r.getrandbits(32) & 0x0FFFFFFF
            gen.steps.append(("mark", v | (r.choice([0, 1, 3, 4, 5, 6]) << 28)))
        else:
            pv = 0xFF00 if r.random() < 0.8 else (r.getrandbits(8) << 8)
            gen.push(gen.op(), push_val=pv | (r.random() < rec_p))
    if launch_end:
        gen.steps.append(("mark", (T.MARK_CLASS_LAUNCH_END << 28) | 0x123))
    return gen.steps


def trw_words(b: bytes):
    return [int.from_bytes(b[4 * i:4 * i + 4], "little") for i in range(8)]


def trw_view(recs):
    """Trace records without their times, grouped: OP per unit, MARK, STOP, DROP, RVHALT (DB §8.3)."""
    out = dict(op=[[] for _ in range(8)], mark=[], stop=[], drop=[], rvhalt=[], other=[])
    for w in recs:
        k = w[0] & 0xF
        if k == T.TRW_W0_KIND_OP:
            out["op"][(w[0] >> 4) & 0xF].append((w[0], w[1], w[2]))
        elif k == T.TRW_W0_KIND_MARK:
            out["mark"].append((w[1], w[2]))
        elif k == T.TRW_W0_KIND_STOP:
            out["stop"].append((w[1], w[2], w[5]))
        elif k == T.TRW_W0_KIND_DROP:
            out["drop"].append(w[1])
        elif k == T.TRW_W0_KIND_RVHALT:
            out["rvhalt"].append(w[2])
        else:
            out["other"].append(tuple(w))
    return out


def rec_view(recs):
    """REC records (DB §8.2) without times: per unit {ticket, data3} in order; markers {value, data3}."""
    out = dict(op=[[] for _ in range(8)], mark=[])
    for r in recs:
        if r[3] & T.REC_DATA3_MARKER:
            out["mark"].append((r[2], r[3]))
        else:
            out["op"][r[0] >> 28].append((r[0], r[3]))
    return out


# =================================================================================================
# Simulator side
# =================================================================================================
class DbgSim:
    """The simulator with its DBG model; the same operations as the RTL bench, synchronously."""

    def __init__(self, cfg, qd, launch=True):
        self.L = lib()
        self.cfg, self.qd = dict(cfg), tuple(qd)
        self.s = S.Sim(cfg, qd)
        self.s.listen()
        if launch:      # one launch around everything: ACTIVE gates the PERF events (EX §6)
            self.s.csr_write(T.CSR_LQ_PUSH, 0)
            rc, _ = self.s.cw_access(1, T.CW_LQ_POP, 0)
            assert rc == S.CW_OK
        self.tickets = {}
        self.issued = [0] * 6
        self.k = 0                      # next step

    def close(self):
        self.s.close()

    # ---- CSR ----
    def rd(self, off):
        return self.s.csr_read(off)

    def wr(self, off, v):
        self.s.csr_write(off, v & M32)

    def cw_rd(self, off):
        return self.s.cw_read(off)

    def cw_wr(self, off, v):
        self.s.cw_write(off, v & M32)

    def status(self):
        return self.rd(T.CSR_DBG_STATUS)

    def credit(self, u):
        return (self.s.cw_read(T.CW_ISSUE_CREDIT) >> (4 * u)) & 0xF

    def settle(self):
        """Runs until the issue FIFO is empty and a DRAINED state (if a stop is active) has settled."""
        for _ in range(4000):
            if self.rd(T.CSR_Q_FIFO) == 0:
                break
            self.s.advance(100)
        self.s.advance(2000)

    def wait_drained(self, budget=1 << 26):
        n = 0
        while not (self.status() & ST_DRAINED):
            self.s.advance(500)
            n += 500
            if n > budget:
                raise TimeoutError("simulator: no DRAINED (status %#x)" % self.status())
        self.settle()

    def step_done(self, budget=1 << 26):
        n = 0
        while self.status() & ST_STEP_BUSY:
            self.s.advance(200)
            n += 200
            if n > budget:
                raise TimeoutError("simulator: QSTEP does not end")

    def idle(self):
        assert self.s.run_until_idle(1 << 34)

    # ---- steps ----
    def _wait_credit(self, u):
        for _ in range(200000):
            if self.credit(u):
                return True
            st = self.status()
            if (st & ST_STOP_ACTIVE) and (st & ST_DRAINED):
                self.s.advance(2000)
                return self.credit(u) != 0
            self.s.advance(100)
        raise TimeoutError("simulator: no credit for unit %d" % u)

    def run(self, steps):
        """Executes steps from self.k on; returns True when all ran, False when a push is blocked."""
        while self.k < len(steps):
            st = steps[self.k]
            kind = st[0]
            if kind == "push":
                w = st[1]
                u = self.s.check(w)[0]
                if not self._wait_credit(u):
                    return False
                d0 = resolve_dep(st[2], self.tickets, u, self.issued)
                d1 = resolve_dep(st[3], self.tickets, u, self.issued)
                t = self.s.push(w, d0, d1, st[4])
                assert t != TK_NONE, "simulator push %d failed" % self.k
                self.tickets[self.k] = t
                self.issued[t >> 28] = ((t & SEQ_MASK) + 1) & SEQ_MASK
            elif kind == "qctrl":
                if st[1] & 8 and not self._wait_credit(5):
                    return False
                t = self.s.qctrl(st[1])
                if st[1] & 8:
                    self.tickets[self.k] = t
                    self.issued[5] = ((t & SEQ_MASK) + 1) & SEQ_MASK
            elif kind == "mark":
                self.s.cw_write(T.CW_TRACE_MARK, st[1])
            elif kind == "wr":
                self.settle()                    # the pushes before it are dispatched: deterministic
                self.wr(st[1], st[2])
            elif kind == "wait":
                self.s.advance(st[1])
            elif kind == "idle":
                self.idle()
            else:
                raise ValueError(st)
            self.k += 1
        return True

    # ---- observations ----
    def hist(self, pidx):
        self.wr(T.CSR_DBG_HIST_IDX, pidx)
        return [self.rd(o) for o in HIST_OFFS]

    def ring(self, base, lines, frm, to):
        """Trace records [frm, to) of the ring in channel base[31:28] of the simulator's DRAM."""
        p = self.L.tsr_sim_dram(self.s.sim, base >> 28)
        beat = base & 0x0FFFFFF0
        cap = 2 * lines
        out = []
        for k in range(frm, to):
            off = beat * 64 + (k % cap) * 32
            out.append(trw_words(ctypes.string_at(p + off, 32)))
        return out

    def recs(self):
        """Pops every REC record of the control window (DB §8.2)."""
        out = []
        while self.cw_rd(T.CW_REC_COUNT):
            out.append([self.cw_rd(T.CW_REC_DATA_BASE + 4 * i) for i in range(4)])
            self.cw_wr(T.CW_REC_POP, 0)
        return out

    def done_streams(self):
        d = [[] for _ in range(6)]
        for e in self.s.events:
            if e["kind"] == S.QEV_DONE:
                d[e["unit"]].append((e["ticket"] & SEQ_MASK, e["pidx"], e["code"], e["ran"]))
        return d


HIST_OFFS = [T.CSR_DBG_HIST_INFO, T.CSR_DBG_HIST_PIDX, T.CSR_DBG_HIST_TICKET, T.CSR_DBG_HIST_DEP0,
             T.CSR_DBG_HIST_DEP1, T.CSR_DBG_HIST_PC] + [T.CSR_DBG_HIST_W_BASE + 4 * i for i in range(8)]


def snap_regs(num_ch):
    """The registers compared at a stop (DB §10.1, EX §5.4): the stop state, the TBPs, the history
    head, the per-unit completion counts and error records, the oldest entries."""
    regs = [T.CSR_DBG_ID, T.CSR_DBG_CFG, T.CSR_DBG_STATUS, T.CSR_DBG_STOP_PIDX, T.CSR_DBG_STOP_AT, T.CSR_DBG_STEP_N,
            T.CSR_DBG_NEXT_PIDX, T.CSR_DBG_STEP_PIDX, T.CSR_DBG_MATCHQ, T.CSR_DBG_HIST_HEAD, T.CSR_DBG_HIST_IDX,
            T.CSR_DBG_TRW_BASE, T.CSR_DBG_TRW_LINES, T.CSR_DBG_TRW_RPTR, T.CSR_DBG_TRW_MASK]
    for i in range(4):
        regs += [T.CSR_DBG_TBP_CTL(i), T.CSR_DBG_TBP_A(i), T.CSR_DBG_TBP_B(i), T.CSR_DBG_TBP_HITS(i)]
    for j in range(2):
        regs += [T.CSR_DBG_RVWP_CTL(j), T.CSR_DBG_RVWP_LO(j), T.CSR_DBG_RVWP_HI(j), T.CSR_DBG_RVWP_HITS(j)]
    for u in range(8):
        regs += [T.CSR_DBG_U_OLDEST(u), T.CSR_DBG_U_OPIDX(u)]
    for u in range(6):
        regs += [T.CSR_Q_DONE(u), T.CSR_Q_ERR_SEQ(u), T.CSR_Q_ERR_CODE(u), T.CSR_Q_ERR_INFO(u),
                 T.CSR_Q_ERR_W0(u), T.CSR_Q_ERR_PIDX(u)]
    regs += [T.CSR_Q_PIDX]
    return regs


# DBG_Uu_STATE: compared at a DRAINED stop without the head-state field (the simulator derives it
# from its Q_STATE view, the RTL from tsr_issue's registered stay reason: DB §9.1 "model values")
U_STATE_MASK = T.DBG_U_STATE_ACCEPTED | T.DBG_U_STATE_HOLD | T.DBG_U_STATE_OUTSTANDING


# =================================================================================================
# RTL side
# =================================================================================================
def dma_write_mask(index):
    return (0x000F, 0x00F0, 0xF000, 0xFFFF)[index % 4]


class Bench:
    def __init__(self, dut, cfg, qd, seed=1, dm=None, mem_lat=(1, 10), mem_ready=0.8, env_kw=None,
                 aw_ready=None, w_ready=None):
        self.dut = dut
        self.cfg, self.qd = cfg, tuple(qd)
        self.r = random.Random(seed)
        kw = dict(gap=(0, 4), lat=(1, 12), ready_p=0.85)
        kw.update(env_kw or {})
        self.env = E.Env(dut, cfg, qd, seed=seed, ctl_ports=(), **kw)
        self.rv = dict(reset_done=1, running=0, halted=0, sleep=0, clk_ok=1, ndmreset=0, dmactive=0)
        self.rv_present = True
        self.rv_rst_lat = 12
        self.rv_cyc = 0.0
        self.unit_hung = 0
        # tsr_ctrl_rv's DB halt request (DB §3.3)
        self.pend = 0
        self.tgl = (0, 0)
        self.halt_reqs = 0
        # DMI: a DM model (tsr_dm of another simulator instance) or a plain register file
        self.dm = dm
        self.dmi_regs = {}
        self.dmi_log = []
        self.dmi_lat = (1, 6)
        self.dmi_resp = 0
        # memory behind channel TRW_CH's write source
        self.mem = {}
        self.mem_lat, self.mem_ready = mem_lat, mem_ready
        # ready probabilities of the request (AW) and data (W) streams; both mem_ready unless given: a low aw_ready
        # with a high w_ready is an arbiter whose request slice is full while data keeps flowing
        self.aw_ready = mem_ready if aw_ready is None else aw_ready
        self.w_ready = mem_ready if w_ready is None else w_ready
        self.wreq = deque()             # [addr, len, id, beats received]
        self.wrsp = deque()             # [cycle, id]
        self.wdat = deque()             # beats not yet matched with a request
        self.data_first = 0             # beats that arrived before their request
        self.bursts = []                # (addr, len, id) of every completed burst
        # DMA write stream on the write mux
        self.dma_bursts = deque()       # [addr, len, data list] to send
        self.dma_sent = []
        self.dma_rsp = 0
        self.dma_data_first = 0.3
        self.cycle = 0
        self.errors = self.env.errors

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

    async def lq(self, off, we=0, wdata=0):
        """One LQ / LCQ handshake of the RISC-V (control-window offset): (rdata, fault)."""
        d = self.dut
        d.lq_req_valid.value = 1
        d.lq_req_off.value = off
        d.lq_req_we.value = we
        d.lq_req_wdata.value = wdata & M32
        d.lq_req_lcq_w.value = 0
        while True:
            await FallingEdge(d.clk)
            if int(d.lq_rsp_valid.value):
                v, f = int(d.lq_rsp_rdata.value), int(d.lq_rsp_fault.value)
                break
        await RisingEdge(d.clk)
        d.lq_req_valid.value = 0
        return v, f

    async def launch(self):
        """One launch taken (LQ_PUSH, LQ_POP): ACTIVE, which gates the PERF counters (EX §6)."""
        await self.wr(T.CSR_LQ_PUSH, 0)
        await self.cycles(4)
        _, f = await self.lq(T.CW_LQ_POP, we=1)
        assert not f, "LQ_POP faulted"

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

    # ---- the clk_rv DB window (tsr_ctl_win's db_* port) ----
    async def db_rd(self, off):
        d = self.dut
        await FallingEdge(d.clk_rv)
        d.db_req.value = 1
        d.db_we.value = 0
        d.db_off.value = off
        await FallingEdge(d.clk_rv)
        d.db_req.value = 0
        await FallingEdge(d.clk_rv)
        return int(d.db_rdata.value)

    async def db_wr(self, off, val):
        d = self.dut
        await FallingEdge(d.clk_rv)
        d.db_req.value = 1
        d.db_we.value = 1
        d.db_off.value = off
        d.db_wdata.value = val & M32
        await FallingEdge(d.clk_rv)
        d.db_req.value = 0
        d.db_we.value = 0

    async def recs(self):
        out = []
        while await self.db_rd(T.CW_REC_COUNT):
            out.append([await self.db_rd(T.CW_REC_DATA_BASE + 4 * i) for i in range(4)])
            await self.db_wr(T.CW_REC_POP, 0)
            for _ in range(4):           # the FIFO's registered output refills
                await RisingEdge(self.dut.clk_rv)
        return out

    async def cycles(self, n):
        for _ in range(n):
            await RisingEdge(self.dut.clk)

    def drained(self):
        return int(self.dut.u_dbg.st_drained.value) and int(self.dut.u_dbg.st_active.value)

    def stop_active(self):
        return int(self.dut.u_dbg.st_active.value)

    # ---- the clk_rv side and the memory (one loop on clk) ----
    async def _loop(self):
        d = self.dut
        r = self.r
        ld_pend = dmi_pend = None
        prev_run = 0
        rst_at = None
        while True:
            await FallingEdge(d.clk)
            run = int(d.rv_run.value)
            ldq = int(d.ld_req.value)
            ld = (int(d.ld_we.value), int(d.ld_addr.value), int(d.ld_wdata.value)) if ldq else None
            dmq = int(d.dmi_req.value)
            dm = (int(d.dmi_addr.value), int(d.dmi_op.value), int(d.dmi_wdata.value)) if dmq else None
            tgl = (int(d.dbg_halt_tgl.value), int(d.dbg_cancel_tgl.value))
            # memory port (write source of channel TRW_CH)
            rq = int(d.s_wr_req_valid.value) and int(d.s_wr_req_ready.value)
            rq_v = (int(d.s_wr_req_addr.value), int(d.s_wr_req_len.value), int(d.s_wr_req_id.value)) if rq else None
            write_valid = int(d.s_wr_dat_valid.value)
            write_ready = int(d.s_wr_dat_ready.value)
            write_payload = (int(d.s_wr_dat_data.value), int(d.s_wr_dat_last.value),
                             int(d.s_wr_dat_mask.value)) if write_valid else None
            dq_v = write_payload if write_ready else None
            rspq = int(d.s_wr_rsp_valid.value) and int(d.s_wr_rsp_ready.value)
            dreq = int(d.d_wr_req_valid.value) and int(d.d_wr_req_ready.value)
            ddat = int(d.d_wr_dat_valid.value) and int(d.d_wr_dat_ready.value)
            drsp = int(d.d_wr_rsp_valid.value)
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
            # tsr_ctrl_rv's pend (DB §3.3): set by a request toggle, cleared by RV_HALTED and the cancel
            if tgl[1] != self.tgl[1] or self.rv["halted"]:
                self.pend = 0
            elif tgl[0] != self.tgl[0]:
                self.pend = 1
            if tgl[0] != self.tgl[0]:
                self.halt_reqs += 1
            self.tgl = tgl
            d.rv_halt_pend.value = self.pend
            # loader port: never used here (refused like tb/csr's when no TCM model)
            if ld is not None:
                ld_pend = c + r.randint(1, 4)
            rsp = 0
            if ld_pend is not None and c >= ld_pend:
                d.ld_rsp_rdata.value = 0
                d.ld_rsp_err.value = 1
                d.ld_rsp_timeout.value = 0
                rsp = 1
                ld_pend = None
            d.ld_rsp_valid.value = rsp
            d.ld_ready.value = int(ld_pend is None and not rsp)
            # DMI port: the DM model
            if dm is not None:
                dmi_pend = [c + r.randint(*self.dmi_lat)] + list(dm)
                self.dmi_log.append(dm)
            rsp = 0
            if dmi_pend is not None and c >= dmi_pend[0]:
                a, op, wd = dmi_pend[1:]
                if self.dm is not None:
                    L = lib()
                    if op == 2:
                        L.tsr_dm_dmi_write(self.dm, a, wd)
                        rd = 0
                        # the DM's own dmcontrol.dmactive, as riscv-dbg's dmactive_o drives rv_dmactive
                        self.rv["dmactive"] = L.tsr_dm_dmi_read(self.dm, 0x10) & 1
                    else:
                        rd = L.tsr_dm_dmi_read(self.dm, a)
                else:
                    if op == 2:
                        self.dmi_regs[a] = wd
                        self.rv["dmactive"] = self.dmi_regs.get(0x10, 0) & 1
                    rd = self.dmi_regs.get(a, 0)
                d.dmi_rsp_rdata.value = rd
                d.dmi_rsp_resp.value = self.dmi_resp
                d.dmi_rsp_timeout.value = 0
                rsp = 1
                dmi_pend = None
            d.dmi_rsp_valid.value = rsp
            d.dmi_ready.value = int(dmi_pend is None and not rsp)
            # memory: requests, beats, responses (in order)
            if rq_v is not None:
                self.wreq.append([rq_v[0], rq_v[1], rq_v[2], 0])
            if dq_v is not None:
                self.wdat.append(dq_v)          # data may come before its request (independent streams)
                if not self.wreq:
                    self.data_first += 1
            while self.wreq and self.wdat:
                w = self.wreq[0]
                dv, last, mask = self.wdat.popleft()
                expected_mask = 0xFFFF if w[2] == 1 else dma_write_mask(w[3])
                if mask != expected_mask:
                    self.errors.append(f"cycle {c}: write mask {mask:#06x}, expected {expected_mask:#06x}")
                bits = sum(0xFFFFFFFF << (32 * word) for word in range(16) if mask & (1 << word))
                address = w[0] + w[3]
                self.mem[address] = (self.mem.get(address, 0) & ~bits) | (dv & bits)
                w[3] += 1
                if last != (w[3] == w[1]):
                    self.errors.append("cycle %d: WLAST %d at beat %d of %d" % (c, last, w[3], w[1]))
                if last:
                    self.bursts.append(tuple(w[:3]))
                    self.wrsp.append([c + r.randint(*self.mem_lat), w[2]])
                    self.wreq.popleft()
            if rspq:
                self.wrsp.popleft()
            d.s_wr_req_ready.value = int(r.random() < self.aw_ready)
            d.s_wr_dat_ready.value = int(r.random() < self.w_ready)
            if self.wrsp and self.wrsp[0][0] <= c:
                d.s_wr_rsp_valid.value = 1
                d.s_wr_rsp_id.value = self.wrsp[0][1]
            else:
                d.s_wr_rsp_valid.value = 0
            # DMA write stream through the mux
            self.dma_rsp += drsp
            self._dma_drive(dreq, ddat)
            # levels and counters
            if self.rv_present:
                self.rv_cyc += 0.6
            d.rv_cyc.value = int(self.rv_cyc) & 0xFFFF
            d.rv_reset_done.value = self.rv["reset_done"]
            d.rv_running.value = self.rv["running"]
            d.rv_halted.value = self.rv["halted"]
            d.rv_sleep.value = self.rv["sleep"]
            d.rv_clk_ok.value = int(self.rv_present)
            d.rv_ndmreset.value = self.rv["ndmreset"]
            d.rv_dmactive.value = self.rv["dmactive"]
            d.unit_hung.value = self.unit_hung
            iss = 0
            for u in range(6):
                iss |= (self.env.issued[u] & SEQ_MASK) << (28 * u)
            d.issued_seq.value = iss
            d.ifq_count.value = min(63, sum(1 for e in self.env.fifo if e[0] <= self.env.cycle))

    # DMA stream state: [addr, len, beats, req_sent, beats_sent]
    def _dma_drive(self, dreq, ddat):
        d = self.dut
        if not hasattr(self, "_dcur"):
            self._dcur = None
            self._dpend = deque()       # requests taken whose data is not complete
        if dreq and self._dcur is not None:
            self._dcur[3] = 1
        if ddat and self._dcur is not None:
            self._dcur[4] += 1
        if self._dcur is not None and self._dcur[3] and self._dcur[4] == self._dcur[1]:
            self.dma_sent.append(tuple(self._dcur[:3]))
            self._dcur = None
        if self._dcur is None and self.dma_bursts:
            a, n, data = self.dma_bursts.popleft()
            self._dcur = [a, n, data, 0, 0]
            self._dfirst = self.r.random() < self.dma_data_first
        cur = self._dcur
        if cur is None:
            d.d_wr_req_valid.value = 0
            d.d_wr_dat_valid.value = 0
            return
        d.d_wr_req_valid.value = int(not cur[3])
        d.d_wr_req_addr.value = cur[0]
        d.d_wr_req_len.value = cur[1]
        # data before the request with probability dma_data_first (the DMA does not wait)
        can_dat = cur[4] < cur[1] and (cur[3] or self._dfirst)
        d.d_wr_dat_valid.value = int(can_dat)
        if can_dat:
            d.d_wr_dat_data.value = cur[2][cur[4]]
            d.d_wr_dat_last.value = int(cur[4] == cur[1] - 1)
            d.d_wr_dat_mask.value = dma_write_mask(cur[4])

    async def start(self, core_ps=10000, rv_ps=16000, rv_phase_ps=0):
        d = self.dut
        # Initialise both clock inputs, reset and traffic before either clock
        # runs, including the core cycles used to offset the RV clock's phase.
        d.clk.value = 0
        d.clk_rv.value = 0
        d.rst_n.value = 0
        zero = ["csr_valid", "csr_we", "csr_addr", "csr_wdata", "cwin_req_valid", "cwin_req_off", "lq_req_valid",
                "lq_req_off", "lq_req_we", "lq_req_wdata", "lq_req_lcq_w", "log_valid", "log_data", "rt_state",
                "rt_info", "ld_ready", "ld_rsp_valid", "ld_rsp_rdata", "ld_rsp_err", "ld_rsp_timeout",
                "dmi_ready", "dmi_rsp_valid", "dmi_rsp_rdata", "dmi_rsp_resp", "dmi_rsp_timeout",
                "sreg", "rv_cyc", "perf_push_stall", "perf_bridge_acc", "perf_bridge_wait",
                "push_timeout", "bridge_timeout", "rv_halted", "rv_sleep", "rv_running", "rv_reset_done",
                "rv_clk_ok", "rv_ndmreset", "issued_seq", "ifq_count", "ifq_valid", "ifq_entry", "xc_rsp_valid",
                "xc_rsp_code", "xc_rsp_info", "xc_rsp_ext", "xc_rsp_cmd", "dma_cmd_ready", "dma_done",
                "qact_cmd_ready", "qact_done", "mvm_cmd_ready", "mvm_done", "vu_cmd_ready", "vu_done",
                "att_cmd_ready", "att_done", "err_ack", "perf_ch_rd_beat", "perf_ch_wait",
                "unit_progress", "plat_err_i", "rv_halt_pend", "unit_hung", "db_req", "db_we", "db_off",
                "db_wdata", "ib_data_req", "ib_data_gnt", "ib_data_addr", "ib_data_we", "ib_data_be", "ib_cd",
                "ib_sleep", "ib_halted", "ib_in_reset", "ib_dfault", "d_wr_req_valid", "d_wr_req_addr",
                "d_wr_req_len", "d_wr_dat_valid", "d_wr_dat_data", "d_wr_dat_mask", "d_wr_dat_last", "s_wr_req_ready",
                "s_wr_dat_ready", "s_wr_rsp_valid", "s_wr_rsp_id"]
        for nm in zero:
            getattr(d, nm).value = 0
        cocotb.start_soon(Clock(d.clk, core_ps, unit="ps").start())
        if rv_phase_ps:
            await Timer(rv_phase_ps, unit="ps")
        cocotb.start_soon(Clock(d.clk_rv, rv_ps, unit="ps").start())
        for _ in range(8):
            await RisingEdge(d.clk)
        d.rst_n.value = 1
        await RisingEdge(d.clk)
        cocotb.start_soon(self._loop())
        cocotb.start_soon(self._dbmon())
        self.env.start_loop()
        await self.cycles(8)

    # ---- DB invariants on the RTL's own strobes (DB §5.1-§5.2), every cycle ----
    async def _dbmon(self):
        """M1: while DRAINED (for 3 cycles: the level is registered), every enqueued entry before T
        has completed and DBG_NEXT_PIDX is at or after T (the push-order prefix, DB §5.1).
        M2: no entry at or after T starts while the stop with that T has been active for 4 cycles
        (the hard hold; a QSTEP's X is below its new T)."""
        d = self.dut
        env = self.env
        ge = lambda a, b: ((a - b) & M32) < 0x80000000      # noqa: E731
        seen = [[0, 0, 0] for _ in range(6)]
        enq, done = {}, set()
        hist = deque(maxlen=4)
        dr_run = 0
        while True:
            await FallingEdge(d.clk)
            act = int(d.u_dbg.st_active.value)
            t = int(d.u_dbg.st_t.value)
            dr = act and int(d.u_dbg.st_drained.value)
            for u in range(6):
                for e in env.enq[u][seen[u][0]:]:
                    enq[e["pidx"]] = u
                seen[u][0] = len(env.enq[u])
                for (_, p, c) in env.start[u][seen[u][1]:]:
                    if len(hist) == 4 and all(a and ge(p, tt) for (a, tt) in hist):
                        self.errors.append("cycle %d: entry %d of unit %d started while held (T %d)" % (c, p, u, t))
                seen[u][1] = len(env.start[u])
                for (_, p, _, _, _) in env.done[u][seen[u][2]:]:
                    done.add(p)
                seen[u][2] = len(env.done[u])
            hist.append((act, t))
            dr_run = dr_run + 1 if dr else 0
            if dr_run == 3:
                miss = [p for p in enq if not ge(p, t) and p not in done]
                if miss:
                    self.errors.append("cycle %d: DRAINED at T %d with entries before T not completed: %s" %
                                       (self.cycle, t, sorted(miss)[:6]))
                if not ge(int(d.u_issue.q_pidx.value), t):
                    self.errors.append("cycle %d: DRAINED at T %d with DBG_NEXT_PIDX %d" %
                                       (self.cycle, t, int(d.u_issue.q_pidx.value)))

    # ---- the step runner (the simulator's DbgSim.run on the RTL) ----
    async def _wait_credit(self, u):
        n = 0
        dn = 0
        while self.env.credit(u) == 0:
            await RisingEdge(self.dut.clk)
            n += 1
            if self.stop_active() and self.drained() and not self.env.fifo:
                dn += 1
                if dn > 400:
                    return self.env.credit(u) != 0
            else:
                dn = 0
            if n > 400000 or (self.errors and n > 20000):
                raise TimeoutError("RTL: no credit for unit %d (%d bench errors)" % (u, len(self.errors)))
        return True

    async def run(self, steps, state):
        """Executes steps from state['k'] on; returns True when all ran, False when a push is blocked."""
        env = self.env
        while state["k"] < len(steps):
            st = steps[state["k"]]
            kind = st[0]
            if kind == "push":
                w = st[1]
                wm = (st[4] >> 8) & 0xFF
                wv = [x if (i == 0 or (wm >> i) & 1) else 0 for i, x in enumerate(w)]
                u = env.unit_of(wv)
                if not await self._wait_credit(u):
                    return False
                d0 = resolve_dep(st[2], state["tickets"], u, env.issued)
                d1 = resolve_dep(st[3], state["tickets"], u, env.issued)
                state["tickets"][state["k"]] = await env.push(w, d0, d1, st[4], pc=HOOK_PC)
            elif kind == "qctrl":
                if st[1] & 8 and not await self._wait_credit(5):
                    return False
                t = await env.qctrl(st[1], pc=HOOK_PC)
                if st[1] & 8:
                    state["tickets"][state["k"]] = t
            elif kind == "mark":
                await env.mark(st[1], pc=HOOK_PC)
            elif kind == "wr":
                await self.settle()
                await self.wr(st[1], st[2])
            elif kind == "wait":
                await self.cycles(st[1])
            elif kind == "idle":
                await env.wait_idle(limit=3000000)
            else:
                raise ValueError(st)
            state["k"] += 1
        return True

    async def settle(self):
        """Waits until the issue FIFO model is empty and the dispatcher has taken everything."""
        n = 0
        while self.env.fifo or int(self.dut.u_issue.dst_q.value) != 0:
            await RisingEdge(self.dut.clk)
            n += 1
            if n > 400000:
                raise TimeoutError("RTL: the issue FIFO does not drain")
        await self.cycles(300)

    async def wait_drained(self, limit=400000):
        n = 0
        while not self.drained():
            await RisingEdge(self.dut.clk)
            n += 1
            if n > limit:
                raise TimeoutError("RTL: no DRAINED")
        await self.settle()

    async def step_done(self, limit=400000):
        n = 0
        while int(self.dut.u_dbg.st_busy.value):
            await RisingEdge(self.dut.clk)
            n += 1
            if n > limit:
                raise TimeoutError("RTL: QSTEP does not end")

    async def idle(self):
        await self.env.wait_idle(limit=600000)
        await self.cycles(50)

    # ---- observations ----
    async def hist(self, pidx):
        await self.wr(T.CSR_DBG_HIST_IDX, pidx)
        return [await self.rd(o) for o in HIST_OFFS]

    def ring(self, base, lines, frm, to):
        beat0 = base & 0x0FFFFFF0
        cap = 2 * lines
        out = []
        for k in range(frm, to):
            pos = k % cap
            v = self.mem.get(beat0 + pos // 2)
            if v is None:
                out.append(None)
                continue
            half = (v >> (256 * (pos & 1))) & ((1 << 256) - 1)
            out.append([(half >> (32 * i)) & M32 for i in range(8)])
        return out

    def done_streams(self):
        return [[(s, p, c, ran) for (s, p, c, ran, _) in self.env.done[u]] for u in range(6)]


async def compare_regs(b: Bench, ds: DbgSim, regs, what, masks=None):
    bad = []
    for off in regs:
        a = await b.rd(off)
        e = ds.rd(off)
        m = (masks or {}).get(off, M32)
        if off == T.CSR_DBG_STATUS:
            m &= STATUS_MASK
        if (a & m) != (e & m):
            bad.append("%s: %#05x RTL %#010x sim %#010x" % (what, off, a & m, e & m))
    return bad


async def compare_units(b: Bench, ds: DbgSim, what):
    bad = []
    for u in range(8):
        a = await b.rd(T.CSR_DBG_U_STATE(u))
        e = ds.rd(T.CSR_DBG_U_STATE(u))
        if (a & U_STATE_MASK) != (e & U_STATE_MASK):
            bad.append("%s: DBG_U%d_STATE RTL %#x sim %#x" % (what, u, a & U_STATE_MASK, e & U_STATE_MASK))
    return bad


async def compare_hist(b: Bench, ds: DbgSim, lo, hi, what):
    """History entries of push indices [lo, hi): every field, PRESENT or not (a slot that is not PRESENT reads 0
    in every field, DB §5.6: the indices below the ring, past the head and after a HIST_CLEAR are not)."""
    bad = []
    for p in range(lo, hi):
        a = await b.hist(p)
        e = ds.hist(p)
        if (a[0] & 1) != (e[0] & 1):
            bad.append("%s: hist %d PRESENT RTL %d sim %d" % (what, p, a[0] & 1, e[0] & 1))
        elif a != e:
            bad.append("%s: hist %d RTL %s sim %s" % (what, p, [hex(x) for x in a], [hex(x) for x in e]))
        if len(bad) > 8:
            break
    return bad


def compare_done(b: Bench, ds: DbgSim, what):
    bad = []
    a, e = b.done_streams(), ds.done_streams()
    for u in range(6):
        if a[u] != e[u]:
            k = next((i for i in range(min(len(a[u]), len(e[u]))) if a[u][i] != e[u][i]), min(len(a[u]), len(e[u])))
            bad.append("%s: completions of unit %d differ at #%d of %d / %d: RTL %s sim %s" %
                       (what, u, k, len(a[u]), len(e[u]), a[u][k] if k < len(a[u]) else None,
                        e[u][k] if k < len(e[u]) else None))
    return bad
