"""cocotb environment of the issue-path benches (tb/issue, tb/csr): the RISC-V side of the issue port,
the unit models, the external checker class, the I2 record handshake and the monitors, all driven
from one per-cycle loop (drive after the rising edge, sample at the falling edge).

  RvIssue    the clk_rv side of the issue FIFO (CC §7.1-§7.3): tickets {u, ISSUED_SEQ[u]}, the push
             index, credits from the RTL's S[u] and the FIFO room, a 32-entry FIFO with a random
             crossing delay, QCTRL and TRACE_MARK entries (387-bit tsr_ifent_t).
  units      DMA / QACT / MVM / VU / AT models: random cmd_ready, random latency, done pulses in accept
             order, accept-ahead up to the unit's limit (tables ACCEPT_MAX); every accepted command is
             checked against the expected command of that unit's next running entry.
  xc         the external checker classes (VU / AT opcodes): answers with the simulator's check.
  records    err_ack after a random delay once err_valid is up (tsr_ctrl_cdc protocol).
  monitors   enqueue tap, start / completion strobes, TRACE_MARK, PERF events, CC §9.4 checks in Python
             (I1 from the completion cycles, per-unit completion order, D / S steps).
The program runner executes the same step list on the simulator (issue_sim.Sim) and on the RTL; the
comparison is cycle-agnostic (CC §15: schedule-independent streams).
"""
from __future__ import annotations

import random
from collections import deque

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge

from common import isa, tables as T

import issue_sim as S

NUNITS = 6
U_DMA, U_QACT, U_MVM, U_VU, U_ATT, U_SYNC = range(6)
UNIT_NAMES = ("dma", "qact", "mvm", "vu", "att", "sync")
ACCEPT_MAX = (T.ACCEPT_MAX_DMA, T.ACCEPT_MAX_QACT, T.ACCEPT_MAX_MVM, T.ACCEPT_MAX_VU, T.ACCEPT_MAX_ATT,
              T.ACCEPT_MAX_SYNC)
SEQ_MASK = (1 << 28) - 1
TK_NONE = S.TICKET_NONE
KIND_OP, KIND_CTRL = 0, 1
CTRL_QABORT, CTRL_HOLD_SET, CTRL_HOLD_CLR, CTRL_QCLEAR, CTRL_MARK = 1, 2, 3, 4, 5
EXT_W = 107


def seq_done(d, s):
    x = (d - s) & SEQ_MASK
    return 1 <= x < (1 << 27)


def ifent(kind, words, dep0=TK_NONE, dep1=TK_NONE, rec=0, ticket=0, pc=0) -> int:
    """tsr_ifent_t (CC §7.2): kind [1:0], w0..w7 [257:2], DEP0 [289:258], DEP1 [321:290], REC [322],
    ticket [354:323], push PC [386:355]."""
    v = kind & 3
    v |= isa.insn_int(words) << T.IFIFO_W_LSB
    v |= (dep0 & 0xFFFFFFFF) << T.IFIFO_DEP0_LSB
    v |= (dep1 & 0xFFFFFFFF) << T.IFIFO_DEP1_LSB
    v |= (rec & 1) << T.IFIFO_REC_LSB
    v |= (ticket & 0xFFFFFFFF) << T.IFIFO_TICKET_LSB
    v |= (pc & 0xFFFFFFFF) << T.IFIFO_PC_LSB
    return v


def unpack_ext(v: int, n: int = 4):
    """tsr_ext_t [n-1:0] -> list of (dram, wr, ch, lo, hi) for the valid slots, slot order."""
    out = []
    for i in range(n):
        e = (v >> (i * EXT_W)) & ((1 << EXT_W) - 1)
        hi = e & ((1 << 48) - 1)
        lo = (e >> 48) & ((1 << 48) - 1)
        ch = (e >> 96) & 0xFF
        wr = (e >> 104) & 1
        dram = (e >> 105) & 1
        valid = (e >> 106) & 1
        if valid:
            out.append((dram, wr, ch, lo, hi))
    return out


def pack_ext(exts) -> int:
    v = 0
    for i, (dram, wr, ch, lo, hi) in enumerate(exts[:4]):
        e = (1 << 106) | (dram << 105) | (wr << 104) | ((ch & 0xFF) << 96) | ((lo & ((1 << 48) - 1)) << 48) | \
            (hi & ((1 << 48) - 1))
        v |= e << (i * EXT_W)
    return v


def sim_ext_to_rtl(exts):
    """The simulator's tsr_extents (kind, write, ch, lo, hi) -> the RTL's tagged extents (SPM / DRAM only)."""
    return [(1 if k == S.EXT_DRAM else 0, wr, ch, lo, hi) for (k, wr, ch, lo, hi) in exts if k in (S.EXT_SPM, S.EXT_DRAM)]


# ---------------------------------------------------------------------------------------------
# Expected unit commands (docs/rtl-core.md §3.2-§3.4) of a running op, from its words
# ---------------------------------------------------------------------------------------------
def expected_cmd(unit, w, num_ch):
    op = w[0] & 0xFF
    sub = (w[0] >> 8) & 0xF
    if unit == U_DMA and op in (isa.OP_LD, isa.OP_ST):
        return ("dma", isa.DMA_CMD_T.pack(is_st=int(op == isa.OP_ST), ch=sub, spm_line=w[1], dram_beat=w[2],
                                          n_lines=w[3]))
    if unit == U_QACT and op == isa.OP_QACT:
        K = w[3]
        xl = (K >> 4) & 0xFFFFFFFF
        return ("qact", isa.QACT_CMD_T.pack(kind=sub & 1, dst_line=w[1], src_line=w[2], k=K & 0xFFFF,
                                            t=w[4] & 0xF, src_stride=w[5] if w[5] else xl,
                                            qa_lines=isa.qa_lines(sub & 1, K) & 0xFFFF))
    if unit == U_MVM and op == isa.OP_MVM:
        N, K, Tt = w[4], w[5] & 0xFFFF, (w[5] >> 16) & 0xF
        yl = ((N + 15) >> 4) & 0xFFFFFFFF
        R = (N + num_ch - 1) // num_ch
        return ("mvm", isa.MVM_CMD_T.pack(fmt=sub & 1, acc=(w[0] >> 12) & 1, dst_line=w[1], x_line=w[2],
                                          w_off=w[3], n=N, k=K, t=Tt, dst_stride=w[6] if w[6] else yl,
                                          y_lines=yl, rows_per_ch=R & 0xFFFFFFFF,
                                          row_beats=isa.row_beats(sub & 1, K), qa_lines=isa.qa_lines(sub & 1, K)))
    return ("xc", isa.insn_int(w))


class Env:
    """Everything around a tsr_issue instance (top or inside the tb wrapper). `pfx` is the signal
    prefix of the issue ports in the DUT (the wrapper exposes them unprefixed)."""

    def __init__(self, dut, cfg: dict, qd, caps=0, seed=1, sim: S.Sim | None = None, lat=(1, 12),
                 ready_p=0.85, rec_lat=(0, 6), xc_lat=(0, 4), gap=(0, 6), xing=(2, 4), seq_init=0, pidx_init=0,
                 ctl_ports=("host_hold", "halt_hold", "qabort", "rv_reset", "host_clear", "sr_busy", "dbg_hold",
                            "dbg_step", "rec_resync"), resync_p=0.0):
        self.dut = dut
        self.cfg = cfg
        self.qd = tuple(qd)
        self.caps = caps
        self.num_ch = cfg["num_ch"]
        self.r = random.Random(seed)
        self.sim = sim                       # checker oracle for the xc class (and the RV-side unit map)
        self.lat = lat
        self.ready_p = ready_p
        self.rec_lat = rec_lat
        self.xc_lat = xc_lat
        self.gap = gap
        self.xing = xing
        self.cycle = 0
        self.ctl_ports = tuple(ctl_ports)    # control inputs this env drives (a wrapper drives its own)
        # RISC-V side state (CC §5.1)
        self.issued = [seq_init] * NUNITS
        self.pidx = pidx_init
        self.fifo = deque()                  # [visible_cycle, entry int, info dict]
        self.pushed_log = []                 # every ticketed entry pushed: dict
        self.ticket_of = {}                  # step index -> ticket
        # sampled RTL state
        self.S = [seq_init] * NUNITS
        self.D = [seq_init] * NUNITS
        self.q_pidx = pidx_init
        self.qidle = 1
        # unit models
        self.u_hold = [deque() for _ in range(NUNITS)]   # accepted ops: [done_cycle, cmd]
        self.u_ready = [0] * NUNITS
        self.u_cmds = [[] for _ in range(NUNITS)]        # accepted commands (kind, value)
        self.u_last_done = [0] * NUNITS
        self.ready_force = [None] * NUNITS               # directed tests: force cmd_ready
        self.lat_force = [None] * NUNITS
        # record handshake
        self.rec_state = [0] * NUNITS        # 0 idle, 1 take scheduled, 2 in flight
        self.rec_at = [0] * NUNITS
        self.rec_seen = [None] * NUNITS      # record value at the last take (what clk_rv shows)
        self.rec_stall = False               # directed: no take / ack while set
        self.resync_p = resync_p             # per-cycle probability of a rec_resync pulse (clk_rv OK again)
        self.rec_sent = [0] * NUNITS         # record handshakes taken (err_valid seen) per unit
        # external checker
        self.xc_pending = None               # cycle at which to answer
        self.xc_answered = False
        # monitors
        self.enq = [[] for _ in range(NUNITS)]
        self.done = [[] for _ in range(NUNITS)]          # (seq, pidx, code, ran, cycle)
        self.start = [[] for _ in range(NUNITS)]         # (seq, pidx, cycle)
        self.marks = []
        self.perf = dict(retired=0, push=0, busy=[0] * 6, mvm_wait=0, mvm_empty=0, hold=0)
        self.errors = []
        self.done_cycle = {}                 # ticket -> completion cycle
        self.deps_of = {}                    # ticket -> (dep0, dep1) as dispatched
        # control inputs (tsr_issue-only bench)
        self.ctl = dict(host_hold=0, halt_hold=0, qabort=0, rv_reset=0, host_clear=0, sr_busy=0,
                        dbg_hold=0, dbg_step=0, rec_resync=0)
        self.expect_cmds = None              # per unit: list of expected commands (from the simulator)
        self.cmd_check_n = [0] * NUNITS

    # ---- RISC-V side ----
    def unit_of(self, words):
        """The unit the issue path sends an opcode to (CC §7.4: refused opcodes go to SYNC)."""
        op = words[0] & 0xFF
        u = T.OP_UNIT[op]
        need = T.OP_CAPS[op]
        if not T.OP_LEGAL[op] or (self.caps & need) != need or u >= NUNITS or self.qd[u] == 0:
            return U_SYNC
        if op not in (isa.OP_NOP, isa.OP_FENCE, isa.OP_LD, isa.OP_ST, isa.OP_QACT, isa.OP_MVM) and self.sim is None:
            return U_SYNC
        return u

    def credit(self, u):
        if self.qd[u] == 0:
            return 0
        used = (self.issued[u] - self.S[u]) & SEQ_MASK
        c = max(0, self.qd[u] - used)
        return min(c, T.ISSUE_FIFO_DEPTH - len(self.fifo), 15)

    async def wait_cycles(self, n):
        for _ in range(n):
            await RisingEdge(self.dut.clk)

    async def _gap(self):
        n = self.r.randint(*self.gap) if self.gap[1] else 0
        if n:
            await self.wait_cycles(n)

    def _put(self, entry, info):
        vis = self.cycle + self.r.randint(*self.xing)
        if self.fifo and self.fifo[-1][0] > vis:
            vis = self.fifo[-1][0]
        self.fifo.append([vis, entry, info])

    async def push(self, words, dep0=TK_NONE, dep1=TK_NONE, push_val=0xFF00, pc=None):
        """ISSUE_PUSH (CC §7.1): words masked by WMASK, waits for a credit; returns the ticket."""
        await self._gap()
        wm = (push_val >> 8) & 0xFF
        w = [x if (i == 0 or (wm >> i) & 1) else 0 for i, x in enumerate(words)]
        u = self.unit_of(w)
        n = 0
        while self.credit(u) == 0:
            await RisingEdge(self.dut.clk)
            n += 1
            if n > 200000:
                raise TimeoutError("push to unit %d: no credit for 200000 cycles" % u)
        t = (u << 28) | (self.issued[u] & SEQ_MASK)
        self.issued[u] = (self.issued[u] + 1) & SEQ_MASK
        pc = (0x1000 + 4 * len(self.pushed_log)) if pc is None else pc
        info = dict(ticket=t, pidx=self.pidx, w=w, dep=(dep0, dep1), rec=push_val & 1, pc=pc, unit=u)
        self.pushed_log.append(info)
        self.pidx = (self.pidx + 1) & 0xFFFFFFFF
        self._put(ifent(KIND_OP, w, dep0, dep1, push_val & 1, t, pc), info)
        return t

    async def qctrl(self, bits, pc=0x2000):
        """QCTRL store: one CTRL entry per set bit of [3:0] in bit order; QCLEAR takes a SYNC ticket."""
        await self._gap()
        n = bin(bits & 0xF).count("1")
        if n == 0:
            return TK_NONE
        while T.ISSUE_FIFO_DEPTH - len(self.fifo) < n or ((bits & 8) and self.credit(U_SYNC) == 0):
            await RisingEdge(self.dut.clk)
        ticket = TK_NONE
        for b, code in ((1, CTRL_QABORT), (2, CTRL_HOLD_SET), (4, CTRL_HOLD_CLR), (8, CTRL_QCLEAR)):
            if not bits & b:
                continue
            w = [code, 0, 0, 0, 0, 0, 0, 0]
            if code == CTRL_QCLEAR:
                t = (U_SYNC << 28) | self.issued[U_SYNC]
                self.issued[U_SYNC] = (self.issued[U_SYNC] + 1) & SEQ_MASK
                info = dict(ticket=t, pidx=self.pidx, w=w, dep=(TK_NONE, TK_NONE), rec=0, pc=pc, unit=U_SYNC)
                self.pushed_log.append(info)
                self.pidx = (self.pidx + 1) & 0xFFFFFFFF
                ticket = t
                self._put(ifent(KIND_CTRL, w, ticket=t, pc=pc), info)
            else:
                self._put(ifent(KIND_CTRL, w, pc=pc), dict(ctrl=code))
        return ticket

    async def mark(self, value, pc=0x3000):
        await self._gap()
        while T.ISSUE_FIFO_DEPTH - len(self.fifo) < 1:
            await RisingEdge(self.dut.clk)
        self._put(ifent(KIND_CTRL, [CTRL_MARK, value, 0, 0, 0, 0, 0, 0], pc=pc),
                  dict(mark=value, pidx=self.pidx, pc=pc))

    async def wait_idle(self, limit=400000):
        n = 0
        while self.fifo or not self.qidle or any(self.u_hold):
            await RisingEdge(self.dut.clk)
            n += 1
            if n > limit:
                raise TimeoutError("not idle after %d cycles (S %s D %s issued %s)" % (limit, self.S, self.D,
                                                                                      self.issued))
        await self.wait_cycles(3)

    # ---- the per-cycle loop ----
    def start_loop(self):
        self._task = cocotb.start_soon(self._loop())

    def stop(self):
        self._task.cancel()

    def _rd(self, name):
        return int(getattr(self.dut, name).value)

    async def _loop(self):
        d = self.dut
        r = self.r
        names = UNIT_NAMES
        has_port = [self.qd[u] != 0 and u != U_SYNC for u in range(NUNITS)]
        pulses = [k for k in ("qabort", "rv_reset", "host_clear", "rec_resync") if k in self.ctl_ports]
        levels = [k for k in ("host_hold", "halt_hold", "sr_busy", "dbg_hold", "dbg_step") if k in self.ctl_ports]
        while True:
            await FallingEdge(d.clk)
            # ---- sample ----
            ifq_ready = self._rd("ifq_ready")
            cmdv = [self._rd("%s_cmd_valid" % names[u]) if has_port[u] else 0 for u in range(5)]
            cmd = {}
            for u in range(5):
                if cmdv[u] and self.u_ready[u]:
                    cmd[u] = self._cmd_value(u)
            s_all = self._rd("seq_s")
            d_all = self._rd("seq_d")
            S_now = [(s_all >> (28 * u)) & SEQ_MASK for u in range(NUNITS)]
            D_now = [(d_all >> (28 * u)) & SEQ_MASK for u in range(NUNITS)]
            tap_enq = self._rd("tap_enq")
            enq = None
            if tap_enq:
                enq = dict(unit=self._rd("tap_enq_unit"), seq=self._rd("tap_enq_seq"), pidx=self._rd("tap_enq_pidx"),
                           code=self._rd("tap_enq_code"), info=self._rd("tap_enq_info"), kind=self._rd("tap_enq_kind"),
                           dep=(self._rd("tap_enq_dep0"), self._rd("tap_enq_dep1")), pc=self._rd("tap_enq_pc"),
                           rec=self._rd("tap_enq_rec"), ext=unpack_ext(self._rd("tap_enq_ext")),
                           w=[(self._rd("tap_enq_w") >> (32 * i)) & 0xFFFFFFFF for i in range(8)])
            tdone = self._rd("tap_done")
            dones = []
            if tdone:
                codes = self._rd("tap_done_code")
                rans = self._rd("tap_done_ran")
                pidxs = self._rd("tap_done_pidx")
                for u in range(NUNITS):
                    if (tdone >> u) & 1:
                        dones.append((u, D_now[u], (pidxs >> (32 * u)) & 0xFFFFFFFF, (codes >> (4 * u)) & 0xF,
                                      (rans >> u) & 1))
            tstart = self._rd("tap_start")
            starts = []
            if tstart:
                sp = self._rd("tap_start_pidx")
                for u in range(NUNITS):
                    if (tstart >> u) & 1:
                        starts.append((u, S_now[u], (sp >> (32 * u)) & 0xFFFFFFFF))
            tmark = self._rd("tap_mark")
            mark = (self._rd("tap_mark_value"), self._rd("tap_mark_pidx"), self._rd("tap_mark_pc")) if tmark else None
            xc_req = self._rd("xc_req_valid")
            recv = self._rd("err_valid")
            rec_hs = self._rd("err_hs") if recv else 0
            self.qidle = self._rd("qidle")
            self.q_pidx = self._rd("q_pidx")
            pr = self._rd("perf_retired")
            self.perf["retired"] += pr
            self.perf["push"] += self._rd("perf_push")
            pb = self._rd("perf_busy")
            for u in range(6):
                self.perf["busy"][u] += (pb >> u) & 1
            self.perf["mvm_wait"] += self._rd("perf_mvm_wait")
            self.perf["mvm_empty"] += self._rd("perf_mvm_empty")
            self.perf["hold"] += self._rd("perf_hold")

            await RisingEdge(d.clk)
            self.cycle += 1
            c = self.cycle
            # ---- events of this edge ----
            if ifq_ready:
                if not self.fifo or self.fifo[0][0] > c - 1:
                    self.errors.append("cycle %d: ifq_ready without a visible entry" % c)
                else:
                    self.fifo.popleft()
            for u, v in cmd.items():
                lat = self.lat_force[u] if self.lat_force[u] is not None else r.randint(*self.lat)
                dc = max(c + lat, (self.u_hold[u][-1][0] + 1) if self.u_hold[u] else 0)
                self.u_hold[u].append([dc, v])
                self.u_cmds[u].append(v)
                if len(self.u_hold[u]) > ACCEPT_MAX[u]:
                    self.errors.append("cycle %d: unit %d holds %d ops (I3)" % (c, u, len(self.u_hold[u])))
                self._check_cmd(u, v)
            for u, s, p in starts:
                self.start[u].append((s, p, c))
            for u, seq, p, code, ran in dones:
                t = (u << 28) | seq
                self.done[u].append((seq, p, code, ran, c))
                self.done_cycle[t] = c
                self._check_i1(t, c)
            if enq is not None:
                self.enq[enq["unit"]].append(enq)
                self.deps_of[(enq["unit"] << 28) | enq["seq"]] = enq["dep"]
            if mark is not None:
                self.marks.append(mark)
            for u in range(NUNITS):
                if ((S_now[u] - self.S[u]) & SEQ_MASK) > 1 or ((D_now[u] - self.D[u]) & SEQ_MASK) > 1:
                    self.errors.append("cycle %d: unit %d S / D stepped by more than 1" % (c, u))
            self.S, self.D = S_now, D_now
            # ---- drive the next cycle ----
            # issue FIFO
            if self.fifo and self.fifo[0][0] <= c:
                d.ifq_valid.value = 1
                d.ifq_entry.value = self.fifo[0][1]
            else:
                d.ifq_valid.value = 0
            # units: done pulses and ready
            for u in range(5):
                if not has_port[u]:
                    continue
                done = 0
                if self.u_hold[u] and self.u_hold[u][0][0] <= c:
                    self.u_hold[u].popleft()
                    self.u_last_done[u] = c
                    done = 1
                getattr(d, "%s_done" % names[u]).value = done
                room = len(self.u_hold[u]) < ACCEPT_MAX[u]
                if self.ready_force[u] is not None:
                    rdy = self.ready_force[u] and room
                else:
                    rdy = room and (r.random() < self.ready_p)
                # a unit that holds an op and has no accept-ahead is not ready (docs/rtl-core.md §3.0)
                self.u_ready[u] = int(rdy)
                getattr(d, "%s_cmd_ready" % names[u]).value = int(rdy)
            # external checker class
            # (the request sampled in the cycle of an answer is that answer's: it drops at this edge)
            if xc_req and self.xc_pending is None and not self.xc_answered:
                self.xc_pending = c + r.randint(*self.xc_lat)
            self.xc_answered = False
            if self.xc_pending is not None and c >= self.xc_pending and xc_req:
                words = [(self._rd("xc_req_insn") >> (32 * i)) & 0xFFFFFFFF for i in range(8)]
                unit, code, info, exts = self.sim.check(words)
                d.xc_rsp_valid.value = 1
                d.xc_rsp_code.value = code
                d.xc_rsp_info.value = info
                d.xc_rsp_ext.value = pack_ext(sim_ext_to_rtl(exts))
                d.xc_rsp_cmd.value = isa.insn_int(words)
                self.xc_pending = None
                self.xc_answered = True
            else:
                d.xc_rsp_valid.value = 0
            # record handshake (tsr_ctrl_cdc: err_valid held with err_hs until err_ack)
            ack = 0
            for u in range(NUNITS):
                if self.rec_stall:
                    break
                if self.rec_state[u] == 0 and (recv >> u) & 1:
                    self.rec_state[u] = 1
                    self.rec_sent[u] += 1
                    self.rec_at[u] = c + r.randint(*self.rec_lat)
                    self.rec_seen[u] = (rec_hs >> (128 * u)) & ((1 << 128) - 1)
                elif self.rec_state[u] == 1 and c >= self.rec_at[u]:
                    ack |= 1 << u
                    self.rec_state[u] = 2
                elif self.rec_state[u] == 2:
                    self.rec_state[u] = 0      # err_valid drops at the ack edge
            d.err_ack.value = ack
            # control (pulses last one cycle)
            if self.resync_p and "rec_resync" in pulses and r.random() < self.resync_p:
                self.ctl["rec_resync"] = 1
            for k in pulses:
                getattr(d, k).value = self.ctl[k]
                self.ctl[k] = 0
            for k in levels:
                getattr(d, k).value = self.ctl[k]

    def _cmd_value(self, u):
        d = self.dut
        if u == U_DMA:
            return ("dma", int(d.dma_cmd.value)) if not self._xcflag("dma") else ("xc", int(d.dma_xcmd.value))
        if u == U_QACT:
            return ("qact", int(d.qact_cmd.value))
        if u == U_MVM:
            return ("mvm", int(d.mvm_cmd.value)) if not self._xcflag("mvm") else ("xc", int(d.mvm_xcmd.value))
        if u == U_VU:
            return ("xc", int(d.vu_cmd.value))
        return ("xc", int(d.att_cmd.value))

    def _xcflag(self, name):
        return int(getattr(self.dut, "%s_cmd_xc" % name).value)

    def _check_cmd(self, u, v):
        if self.expect_cmds is None:
            return
        k = self.cmd_check_n[u]
        self.cmd_check_n[u] += 1
        exp = self.expect_cmds[u]
        if k >= len(exp):
            self.errors.append("unit %d accepted command #%d, the simulator ran %d" % (u, k, len(exp)))
        elif exp[k] != v:
            self.errors.append("unit %d command #%d: RTL %s %x, expected %s %x" % (u, k, v[0], v[1], exp[k][0], exp[k][1]))

    def _check_i1(self, t, c):
        """I1 (CC §9.4) from the completion cycles: every dep completed at an earlier edge."""
        deps = self.deps_of.get(t)
        if deps is None:
            return
        for dp in deps:
            if (dp >> 28) == 15:
                continue
            dc = self.done_cycle.get(dp)
            if dc is None or dc >= c:
                self.errors.append("I1: %08x completed at cycle %d before its dep %08x (%s)" % (t, c, dp, dc))


CLK_TASK = None


async def start(dut, reset_cycles=6):
    global CLK_TASK
    CLK_TASK = cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.rst_n.value = 0
    for nm, v in (("ifq_valid", 0), ("ifq_entry", 0), ("xc_rsp_valid", 0), ("xc_rsp_code", 0), ("xc_rsp_info", 0),
                  ("xc_rsp_ext", 0), ("xc_rsp_cmd", 0), ("err_ack", 0)):
        getattr(dut, nm).value = v
    for u in UNIT_NAMES[:5]:
        getattr(dut, "%s_cmd_ready" % u).value = 0
        getattr(dut, "%s_done" % u).value = 0
    for nm in ("host_hold", "halt_hold", "qabort", "rv_reset", "host_clear", "sr_busy", "dbg_hold", "dbg_step",
               "rec_resync"):
        if hasattr(dut, nm):
            getattr(dut, nm).value = 0
    for _ in range(reset_cycles):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
