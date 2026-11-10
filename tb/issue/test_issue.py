"""tb/issue — tsr_issue (dispatcher, tsr_chk, queues, heads, SYNC, errors, holds, records) against the
simulator's queue model (docs/design/04-control-core.md "CC" §7-§9, §15.2; SW §4).

Every test runs push streams through the RTL's issue-FIFO port and, unchanged, through the
simulator's issue port (sw/sim tsr_q.c via tb/issue/issue_sim.py), then compares, cycle-agnostic:
  - the ticket of every push (RISC-V side model == tsr_sim_push);
  - per unit, every enqueue {seq, pidx, code, deps as dispatched} (RTL tap vs the ENQ events);
  - per unit, every completion {seq, pidx, code, ran} in order (RTL strobes vs the DONE events);
  - per unit, every command a unit accepted (unit models) against the simulator's running ops;
  - the final error records, D[u], Q_PIDX, the TRACE_MARK stream and the PERF events RETIRED /
    PUSHES (the simulator inside one launch so that ACTIVE gates them, EX §6).
Streams: random programs that keep every outcome schedule-independent (the rules of
sw/tools/tsr_trace_gen.c: hold windows within the queue depths, QCLEAR followed by deps on its
ticket, QABORT only after a QHOLD that holds everything since, host QABORT / QCLEAR at idle
points), and the trace sets under traces/ (push.bin replayed into the issue FIFO; also checked
against expect/tickets.bin). The CC §9.4 invariants are RTL assertions (fatal) plus Python checks
(I1 from completion cycles, I3 in the unit models, one S / D step per cycle).
Directed tests: latencies (done -> dependent start <= 3, same unit <= 2), ticket wrap (SEQ_INIT /
PIDX_INIT preset near 2^28 / 2^32), hard hold and its release by QABORT and SR_BUSY, dbg_step,
holds per source, I2 with stalled record handshakes, MVM accept-ahead, credits / back-pressure.
Mutation spot checks: tb/issue/mutate.py (opt-in, TSR_ISSUE_MUTANTS=1).
"""
from __future__ import annotations

import glob
import json
import os
import random
import struct
import sys
from pathlib import Path

import cocotb
import pytest
from cocotb.triggers import FallingEdge, RisingEdge

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common import isa, tables as T                      # noqa: E402
from common.runner import REPO, run                       # noqa: E402

import issue_sim as S                                     # noqa: E402
import issue_tb as E                                      # noqa: E402

SEQ_MASK = E.SEQ_MASK
TK_NONE = S.TICKET_NONE
def rtl(name):
    """An RTL source of the issue path; $TSR_MUT_DIR overrides it with a mutated copy (test_issue_mutants)."""
    d = os.environ.get("TSR_MUT_DIR")
    if d and (Path(d) / name).exists():
        return Path(d) / name
    return REPO / "rtl/core" / name


def core_src():
    return [REPO / "rtl/core/tsr_tables_pkg.sv", REPO / "rtl/core/tsr_pkg.sv", rtl("tsr_chk.sv"), rtl("tsr_issue.sv")]


# ---------------------------------------------------------------------------------------------
# Configuration (cocotb side: from the environment the pytest launcher sets)
# ---------------------------------------------------------------------------------------------
def env_cfg():
    c = json.loads(os.environ.get("TSR_ISSUE_CFG", "{}"))
    cfg = S.config(num_ch=c.get("num_ch", 4), t_max=c.get("t_max", 8), k_max=c.get("k_max", 32768),
                   spm_bytes=c.get("spm_bytes", 4 << 20), ch_bytes=c.get("ch_bytes", 16 << 30),
                   addr_beat_w=c.get("addr_beat_w", 28), caps=c.get("caps", 0))
    qd = tuple(c.get("qd", S.U200_QD))
    return cfg, qd, c


# ---------------------------------------------------------------------------------------------
# Programs. Steps:
#   ("push", words, dep0, dep1, push_val)  deps: ("none",) | ("t", step) | ("rel", unit, rel_seq)
#                                         | ("own",)  (the op's own ticket: BAD_DEP)
#   ("qctrl", bits)   ("mark", value)   ("idle",)   ("wait", cycles)
#   ("host_hold", 0 | 1)   ("dbg_hold", mask)   ("host_abort",)   ("host_clear",)
# ---------------------------------------------------------------------------------------------
class Gen:
    """Random schedule-independent programs over the core classes (and VU ops when CAPS.VU)."""

    def __init__(self, cfg, qd, seed):
        self.cfg, self.qd = cfg, qd
        self.r = random.Random(seed)
        self.C = cfg["num_ch"]
        self.spm_lines = cfg["spm_bytes"] // 64
        self.dlim = isa.dram_limit(cfg["ch_bytes"], cfg["addr_beat_w"])
        self.steps = []
        self.good = []                    # ticketed push steps whose tickets are good dep targets
        self.barrier = None               # QCLEAR step the first entry of each unit depends on
        self.need_barrier = set()
        self.stale = []                   # ticketed steps before the last QCLEAR
        self.stale_ok = False             # an idle step since the QCLEAR: a dep on them is BAD_DEP
        self.unit_of = {}

    # ---- op encoders ----
    def _ld_st(self, bad=None):
        r = self.r
        n = r.choice([0, 1, 2, r.randint(1, 64)])
        ch = r.randrange(self.C)
        spm = r.randint(0, max(0, self.spm_lines - n))
        dram = r.randint(0, max(0, min(self.dlim, 1 << 32) - n))
        if bad == "spm":
            spm = self.spm_lines - n + r.randint(1, 3) if r.random() < 0.7 else r.randint(self.spm_lines, (1 << 32) - 1)
        if bad == "dram":
            dram = min((1 << 32) - 1, self.dlim - n + r.randint(1, 3)) if self.dlim < (1 << 32) else dram
            if self.dlim >= (1 << 32):
                n = max(n, 1)
        if bad == "ch":
            ch = r.randint(self.C, 15) if self.C < 16 else ch
        w = (isa.enc_st if r.random() < 0.4 else isa.enc_ld)(ch, spm & 0xFFFFFFFF, dram & 0xFFFFFFFF, n)
        if bad == "rsv":
            w[r.randint(4, 7)] = r.getrandbits(32) | 1
        return w

    def _qact(self, bad=None):
        r = self.r
        kind = r.randint(0, 1)
        bs = 256 if kind else 32
        K = bs * r.randint(1, max(1, min(self.cfg["k_max"], 2048) // bs))
        Tt = r.randint(1, self.cfg["t_max"])
        if bad == "k":
            K = r.choice([0, K + 16, self.cfg["k_max"] + bs])
        if bad == "t":
            Tt = r.choice([0, self.cfg["t_max"] + 1, 0x80000001])
        if bad == "kind":
            kind = r.randint(2, 15)
        ql = isa.qa_lines(kind & 1, K) if K else 1
        xl = K // 16
        stride = r.choice([0, 0, xl + r.randint(0, 8)])
        ss = stride if stride else xl
        e_dst = Tt * ql if Tt < 16 else ql
        e_src = (Tt - 1) * ss + xl if Tt < 16 else xl
        dst = r.randint(0, max(0, self.spm_lines - e_dst))
        src = r.randint(0, max(0, self.spm_lines - e_src))
        if bad == "spm":
            dst = self.spm_lines - e_dst + r.randint(1, 4)
        if bad == "spm2":
            src = self.spm_lines - e_src + r.randint(1, 4)
        w = isa.enc_qact(kind, dst & 0xFFFFFFFF, src & 0xFFFFFFFF, K, Tt, stride)
        if bad == "rsv":
            w[r.choice([0, 6, 7])] |= 1 << r.randint(16, 31)
        return w

    def _mvm(self, bad=None):
        r = self.r
        fmt = r.randint(0, 1)
        bs = 256 if fmt else 32
        K = bs * r.randint(1, max(1, min(self.cfg["k_max"], 4096) // bs))
        Tt = r.randint(1, self.cfg["t_max"])
        N = r.choice([0, 1, self.C - 1 if self.C > 1 else 1, r.randint(1, 512)])
        if bad == "fmt":
            fmt = r.randint(2, 15)
        if bad == "k":
            K = r.choice([0, K + 16, (self.cfg["k_max"] + bs) & 0xFFFF])
        if bad == "t":
            Tt = r.choice([0, min(15, self.cfg["t_max"] + 1)])
        f1 = fmt & 1
        ql = isa.qa_lines(f1, K) if K else 1
        yl = (N + 15) // 16
        dstride = r.choice([0, 0, yl + r.randint(0, 4)])
        ds = dstride if dstride else yl
        e_dst = (max(Tt, 1) - 1) * ds + yl
        e_x = max(Tt, 1) * ql
        R = (N + self.C - 1) // self.C
        e_w = R * isa.row_beats(f1, K) if K else 0
        dst = r.randint(0, max(0, self.spm_lines - e_dst))
        x = r.randint(0, max(0, self.spm_lines - e_x))
        woff = r.randint(0, max(0, min(self.dlim, 1 << 32) - e_w))
        if bad == "spm":
            dst = self.spm_lines - e_dst + r.randint(1, 4) if e_dst else self.spm_lines + 1
        if bad == "spm2":
            x = self.spm_lines - e_x + r.randint(1, 4)
        if bad == "dram" and self.dlim < (1 << 32):
            woff = min((1 << 32) - 1, self.dlim - e_w + r.randint(1, 4))
        w = isa.enc_mvm(fmt, dst & 0xFFFFFFFF, x & 0xFFFFFFFF, woff & 0xFFFFFFFF, N, K & 0xFFFF, Tt & 0xF, dstride,
                        acc=r.randint(0, 1))
        if bad == "rsv":
            j = r.choice([0, 5, 7])
            w[j] |= 1 << (r.randint(13, 31) if j == 0 else r.randint(20, 31) if j == 5 else r.randint(0, 31))
        return w

    def _sync(self, bad=None):
        w = isa.enc_fence() if self.r.random() < 0.4 else isa.enc_nop()
        if bad == "rsv":
            w[self.r.randint(0, 7)] |= 1 << (self.r.randint(8, 31))
        return w

    def _vu(self):
        r = self.r
        # VU / AT opcodes (the VU / AT classes through the xc port); legal or not, the simulator decides
        op = r.choice([0x30, 0x31, 0x32, 0x40, 0x34, 0x37, 0x12, 0x33])
        return [op | (r.getrandbits(8) << 8 if r.random() < 0.5 else 0)] + \
               [r.choice([0, r.getrandbits(8), r.getrandbits(16)]) for _ in range(7)]

    def op(self):
        r = self.r
        cls = r.choices(["ldst", "qact", "mvm", "sync", "illegal", "vu"],
                        [5, 3, 3 if self.qd[2] else 0, 2, 0.15, 2 if self.cfg["caps"] & 1 else 0.1])[0]
        bad = None
        if r.random() < 0.06:
            bad = r.choice({"ldst": ["spm", "dram", "ch", "rsv"], "qact": ["k", "t", "kind", "spm", "spm2", "rsv"],
                            "mvm": ["fmt", "k", "t", "spm", "spm2", "dram", "rsv"], "sync": ["rsv"],
                            "illegal": [None], "vu": [None]}[cls])
        if cls == "ldst":
            return self._ld_st(bad)
        if cls == "qact":
            return self._qact(bad)
        if cls == "mvm":
            return self._mvm(bad)
        if cls == "sync":
            return self._sync(bad)
        if cls == "vu":
            return self._vu()
        op = r.choice([0x01, 0x02, 0x03, 0x05, 0x09, 0x0A, 0x13, 0x1C, 0x2F, 0x48, 0x70, 0x80, 0xFF])
        return [op | (r.getrandbits(24) << 8 if r.random() < 0.3 else 0)] + [r.getrandbits(32) if r.random() < 0.3 else 0
                                                                            for _ in range(7)]

    def unit(self, words):
        """The simulator's unit for the words (refused -> SYNC)."""
        return self.sim_unit(words)

    # ---- steps ----
    def dep(self, unit, inject=True):
        r = self.r
        x = r.random()
        if inject and self.stale_ok and self.stale and x < 0.03:
            return ("t", r.choice(self.stale))          # pushed before the last QCLEAR (CC §8.3)
        if inject and x < 0.02:
            return r.choice([("own",), ("rel", r.randint(6, 14), r.getrandbits(28)),
                             ("rel", 4, 0) if self.qd[4] == 0 else ("none",),
                             ("future", r.randrange(6) if self.qd[5] else 5)])
        if x < 0.45 or not self.good:
            return ("none",)
        lo = max(0, len(self.good) - 24)
        return ("t", self.good[r.randrange(lo, len(self.good))])

    def push(self, words, d0=None, d1=None, push_val=None):
        u = self.unit(words)
        if d0 is None:
            d0 = self.dep(u)
        if d1 is None:
            d1 = self.dep(u)
        if self.barrier is not None and u in self.need_barrier:
            d0 = ("t", self.barrier)
            self.need_barrier.discard(u)
        if push_val is None:
            pv = 0xFF00 if self.r.random() < 0.8 else (self.r.getrandbits(8) << 8)
            push_val = pv | (self.r.random() < 0.2)
        self.steps.append(("push", words, d0, d1, push_val))
        idx = len(self.steps) - 1
        self.unit_of[idx] = u
        self.good.append(idx)
        return idx

    def qclear(self):
        self.steps.append(("qctrl", 8))
        idx = len(self.steps) - 1
        self.barrier = idx
        self.stale = [i for i in self.good if i != idx][-16:]
        self.stale_ok = False
        self.good = [idx]
        self.need_barrier = {u for u in range(5) if self.qd[u]}
        return idx

    def program(self, n_ops=60, aborts=True, holds=True, host=True):
        r = self.r
        if aborts and r.random() < 0.3:
            self.qabort_recipe()
        while len([s for s in self.steps if s[0] == "push"]) < n_ops:
            x = r.random()
            if x < 0.03 and holds:
                self.hold_window(r.choice(["host", "rv", "dbg"]))
            elif x < 0.075:
                self.qclear()
            elif x < 0.085 and host:
                self.steps.append(("idle",))
                self.steps.append(("host_abort",))
                self.push(self.op())
                self.steps.append(("idle",))
                self.steps.append(("host_clear",))
                self.stale = []
                self.good = []
                self.barrier = None
                self.need_barrier = set()
            elif x < 0.105:
                self.steps.append(("mark", r.getrandbits(32)))
            elif x < 0.115:
                self.steps.append(("idle",))
                self.stale_ok = True
                if self.good and r.random() < 0.5:
                    # a stale dep (before the last QCLEAR) once the QCLEAR has surely applied
                    pass
            else:
                self.push(self.op())
        self.steps.append(("idle",))
        return self.steps

    def hold_window(self, kind):
        """Hold, then at most depth entries per unit (no push waits on a held queue), then release."""
        self.steps.append(("idle",))
        if kind == "host":
            self.steps.append(("host_hold", 1))
        elif kind == "rv":
            self.steps.append(("qctrl", 2))
        else:
            self.steps.append(("dbg_hold", self.r.getrandbits(6)))
        room = list(self.qd)
        room[5] = max(0, room[5] - (1 if kind == "rv" else 0))
        for _ in range(self.r.randint(1, 12)):
            w = self.op()
            u = self.unit(w)
            if room[u] <= 0:
                continue
            room[u] -= 1
            self.push(w)
        if kind == "host":
            self.steps.append(("host_hold", 0))
        elif kind == "rv":
            self.steps.append(("qctrl", 4))
        else:
            self.steps.append(("dbg_hold", 0))

    def qabort_recipe(self):
        """QHOLD set, held entries (nothing starts), QABORT, QHOLD clear, QCLEAR (CC §9.5; the
        trace generator's recipe): every held entry completes ABORTED. Only entries that would run
        (no poisoned entry, no SYNC entry: those drain under a hold and would race the abort)."""
        self.steps.append(("idle",))
        self.steps.append(("qctrl", 2))
        room = list(self.qd)
        room[5] -= 1          # the QCLEAR
        held = []
        for _ in range(self.r.randint(1, 10)):
            w = self.op()
            u = self.unit(w)
            if room[u] <= 0 or u == 5 or self.sim_code(w) != 0:
                continue
            room[u] -= 1
            # deps only on held entries: everything since the hold fails, whatever the schedule
            d0 = ("t", self.r.choice(held)) if held and self.r.random() < 0.5 else ("none",)
            held.append(self.push(w, d0, ("none",), 0xFF00))      # unmasked: the words checked above
        if room[5] > 0 and held:
            self.push(isa.enc_nop(), ("t", held[-1]), ("none",), 0xFF00)
        self.steps.append(("qctrl", 1))
        self.steps.append(("qctrl", 4))
        self.qclear()


# ---------------------------------------------------------------------------------------------
# Trace sets (SW §5.2): push.bin -> steps; deps are relative tickets (base 0 on a fresh queue)
# ---------------------------------------------------------------------------------------------
def trace_steps(tdir):
    b = open(os.path.join(tdir, "ops", "push.bin"), "rb").read()
    steps = []
    for i in range(0, len(b), 48):
        rec = struct.unpack_from("<12I", b, i)
        w, d0, d1, pv, kind = list(rec[0:8]), rec[8], rec[9], rec[10], rec[11]
        if kind == 0:
            steps.append(("push", w, ("rel", d0 >> 28, d0 & SEQ_MASK) if (d0 >> 28) != 15 else ("none",),
                          ("rel", d1 >> 28, d1 & SEQ_MASK) if (d1 >> 28) != 15 else ("none",), pv))
        elif kind == 1:
            steps.append(("qctrl", w[0]))
        else:
            steps.append(("mark", w[0]))
    steps.append(("idle",))
    return steps


def trace_expect(tdir):
    p = os.path.join(tdir, "expect", "tickets.bin")
    if not os.path.exists(p):
        return None
    b = open(p, "rb").read()
    n = struct.unpack_from("<I", b, 0)[0]
    out = []
    for u in range(n):
        cnt, code, rseq, info, w0, rpidx = struct.unpack_from("<6I", b, 16 + 32 * u)
        out.append(dict(count=cnt, code=code, seq=rseq, info=info, w0=w0, pidx=rpidx))
    return out


def trace_config(tdir):
    m = json.load(open(os.path.join(tdir, "meta.json")))
    c = m["config"]
    return S.config(num_ch=c["num_ch"], t_max=c["t_max"], k_max=c["k_max"], spm_bytes=c["spm_bytes"],
                    ch_bytes=c["ch_bytes"], addr_beat_w=c["addr_beat_w"], caps=c["caps"]), tuple(c["q_depth"])


# ---------------------------------------------------------------------------------------------
# Running a program on the simulator
# ---------------------------------------------------------------------------------------------
def resolve_dep(d, tickets, base, unit, issued):
    if d[0] == "none":
        return TK_NONE
    if d[0] == "t":
        return tickets[d[1]]
    if d[0] == "rel":
        return (d[1] << 28) | ((base + d[2]) & SEQ_MASK)
    if d[0] == "own":
        return (unit << 28) | issued[unit]
    if d[0] == "future":
        return (d[1] << 28) | ((issued[d[1]] + 1) & SEQ_MASK)
    raise ValueError(d)


def run_sim(cfg, qd, steps, launch=True):
    sim = S.Sim(cfg, qd)
    sim.listen()
    sim.dbg_hooks()
    if launch:       # one launch around the stream so that ACTIVE gates the PERF events (EX §6)
        sim.csr_write(T.CSR_LQ_PUSH, 0)
        rc, _ = sim.cw_access(1, T.CW_LQ_POP, 0)
        assert rc == S.CW_OK
    tickets = {}
    issued = [0] * 6
    ctrl = 0
    for i, st in enumerate(steps):
        k = st[0]
        if k == "push":
            w = st[1]
            u, _, _, _ = sim.check(w)
            d0 = resolve_dep(st[2], tickets, 0, u, issued)
            d1 = resolve_dep(st[3], tickets, 0, u, issued)
            t = sim.push(w, d0, d1, st[4])
            assert t != TK_NONE, "simulator push %d timed out (no credit)" % i
            tickets[i] = t
            issued[t >> 28] = ((t & SEQ_MASK) + 1) & SEQ_MASK
        elif k == "qctrl":
            t = sim.qctrl(st[1])
            if st[1] & 8:
                tickets[i] = t
                issued[5] = ((t & SEQ_MASK) + 1) & SEQ_MASK
        elif k == "mark":
            sim.cw_write(T.CW_TRACE_MARK, st[1])
        elif k == "idle":
            assert sim.run_until_idle(1 << 32)
        elif k == "wait":
            sim.advance(st[1])
        elif k == "host_hold":
            ctrl = (ctrl & ~T.CTRL_HOST_HOLD) | (T.CTRL_HOST_HOLD if st[1] else 0)
            sim.csr_write(T.CSR_CTRL, ctrl)
        elif k == "dbg_hold":
            sim.hold_mask = st[1]
        elif k == "host_abort":
            sim.csr_write(T.CSR_CTRL, ctrl | T.CTRL_QABORT)
        elif k == "host_clear":
            sim.csr_write(T.CSR_CTRL, ctrl | T.CTRL_QCLEAR)
        else:
            raise ValueError(k)
    assert sim.run_until_idle(1 << 32)
    obs = dict(tickets=tickets, enq=[[] for _ in range(6)], done=[[] for _ in range(6)], marks=[], cmds=[[] for _ in range(6)])
    for e in sim.events:
        u = e["unit"]
        if e["kind"] == S.QEV_ENQ:
            obs["enq"][u].append((e["ticket"] & SEQ_MASK, e["pidx"], e["code"], e["dep"][0], e["dep"][1]))
        elif e["kind"] == S.QEV_DONE:
            obs["done"][u].append((e["ticket"] & SEQ_MASK, e["pidx"], e["code"], e["ran"]))
            if e["ran"]:
                w = e["w"]
                if not ((w[0] & 0xFF) in (isa.OP_LD, isa.OP_ST) and w[3] == 0):
                    obs["cmds"][u].append(E.expected_cmd(u, w, cfg["num_ch"]))
        elif e["kind"] == S.QEV_MARK:
            obs["marks"].append((e["w"][1], e["pidx"]))
    obs["rec"] = []
    for u in range(6):
        st = sim.csr_read(T.CSR_Q_STATE(u))
        err = (st >> 8) & 1
        obs["rec"].append((err, sim.csr_read(T.CSR_Q_ERR_SEQ(u)), sim.csr_read(T.CSR_Q_ERR_CODE(u)),
                           sim.csr_read(T.CSR_Q_ERR_INFO(u)), sim.csr_read(T.CSR_Q_ERR_W0(u)),
                           sim.csr_read(T.CSR_Q_ERR_PIDX(u))) if err else (0,))
    obs["D"] = [sim.csr_read(T.CSR_Q_DONE(u)) for u in range(6)]
    obs["q_pidx"] = sim.csr_read(T.CSR_Q_PIDX)
    obs["retired"] = sim.perf(T.PERF_RETIRED)
    obs["pushes"] = sim.perf(5 + 2 * cfg["num_ch"] + T.PERF_XOFF_PUSHES)
    sim.close()
    return obs


# ---------------------------------------------------------------------------------------------
# Running a program on the RTL
# ---------------------------------------------------------------------------------------------
async def run_rtl(env: E.Env, steps, seq_base=0, pidx_base=0):
    tickets = {}
    for i, st in enumerate(steps):
        k = st[0]
        if k == "push":
            w = st[1]
            u = env.unit_of(w)
            d0 = resolve_dep(st[2], tickets, seq_base, u, env.issued)
            d1 = resolve_dep(st[3], tickets, seq_base, u, env.issued)
            tickets[i] = await env.push(w, d0, d1, st[4])
        elif k == "qctrl":
            t = await env.qctrl(st[1])
            if st[1] & 8:
                tickets[i] = t
        elif k == "mark":
            await env.mark(st[1])
        elif k == "idle":
            await env.wait_idle()
        elif k == "wait":
            await env.wait_cycles(st[1])
        elif k == "host_hold":
            env.ctl["host_hold"] = st[1]
        elif k == "dbg_hold":
            env.ctl["dbg_hold"] = st[1]
        elif k == "host_abort":
            env.ctl["qabort"] = 1
            await env.wait_cycles(4)
        elif k == "host_clear":
            env.ctl["host_clear"] = 1
            await env.wait_cycles(4)
        else:
            raise ValueError(k)
    await env.wait_idle()
    await env.wait_cycles(20)       # record handshakes settle
    return tickets


def rtl_obs(env: E.Env, tickets, seq_base, pidx_base):
    rel = lambda s: (s - seq_base) & SEQ_MASK            # noqa: E731
    relp = lambda p: (p - pidx_base) & 0xFFFFFFFF         # noqa: E731
    reldep = lambda t: t if (t >> 28) >= 6 else (t & 0xF0000000) | rel(t & SEQ_MASK)   # noqa: E731
    obs = dict(tickets={i: ((t & 0xF0000000) | rel(t & SEQ_MASK)) for i, t in tickets.items()})
    obs["enq"] = [[(rel(e["seq"]), relp(e["pidx"]), e["code"], reldep(e["dep"][0]), reldep(e["dep"][1]))
                   for e in env.enq[u]] for u in range(6)]
    obs["done"] = [[(rel(s), relp(p), c, ran) for (s, p, c, ran, _) in env.done[u]] for u in range(6)]
    obs["marks"] = [(v, relp(p)) for (v, p, _) in env.marks]
    recs = int(env.dut.err_rec.value)
    obs["rec"] = []
    for u in range(6):
        r = (recs >> (129 * u)) & ((1 << 129) - 1)
        pidx = r & 0xFFFFFFFF
        w0 = (r >> 32) & 0xFFFFFFFF
        info = (r >> 64) & 0xFFFFFFFF
        code = (r >> 96) & 0xF
        seq = (r >> 100) & SEQ_MASK
        err = (r >> 128) & 1
        obs["rec"].append((1, rel(seq), code, info, w0, relp(pidx)) if err else (0,))
    obs["D"] = [rel(env.D[u]) if env.qd[u] else 0 for u in range(6)]
    obs["q_pidx"] = relp(env.q_pidx)
    obs["retired"] = env.perf["retired"]
    obs["pushes"] = env.perf["push"]
    return obs


def compare(name, sim_obs, rtl_obs_, errors, log=None):
    bad = list(errors)
    if sim_obs["tickets"] != rtl_obs_["tickets"]:
        diff = [(i, hex(sim_obs["tickets"][i]), hex(rtl_obs_["tickets"].get(i, -1))) for i in sim_obs["tickets"]
                if sim_obs["tickets"][i] != rtl_obs_["tickets"].get(i)]
        bad.append("tickets differ: %s" % diff[:5])
    for key in ("enq", "done"):
        for u in range(6):
            a, b = sim_obs[key][u], rtl_obs_[key][u]
            if a != b:
                k = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), min(len(a), len(b)))
                bad.append("%s unit %d differs at #%d of %d / %d: sim %s rtl %s" %
                           (key, u, k, len(a), len(b), a[k] if k < len(a) else None, b[k] if k < len(b) else None))
    for key in ("rec", "D", "q_pidx", "marks", "retired", "pushes"):
        if sim_obs[key] != rtl_obs_[key]:
            bad.append("%s: sim %s rtl %s" % (key, sim_obs[key], rtl_obs_[key]))
    if log:
        log.info("%s: %d enq, %d done, %s", name, sum(len(x) for x in rtl_obs_["enq"]),
                 sum(len(x) for x in rtl_obs_["done"]), "ok" if not bad else "MISMATCH")
    return bad


async def one_program(dut, cfg, qd, caps, steps, seed, name, seq_base=0, pidx_base=0, env_kw=None, launch=True):
    """Runs steps on the simulator and on a freshly reset RTL; returns the mismatch list."""
    sim_obs = run_sim(cfg, qd, steps, launch=launch)
    chk = S.Sim(cfg, qd)                                    # checker oracle for the xc class
    env = E.Env(dut, cfg, qd, caps=caps, seed=seed, sim=chk if caps else None, seq_init=seq_base,
                pidx_init=pidx_base, **(env_kw or {}))
    env.expect_cmds = sim_obs["cmds"]
    await _bring_up(dut)
    env.start_loop()
    try:
        tickets = await run_rtl(env, steps, seq_base, pidx_base)
    finally:
        env.stop()
    ro = rtl_obs(env, tickets, seq_base, pidx_base)
    for u in range(6):
        if env.cmd_check_n[u] != len(sim_obs["cmds"][u]):
            env.errors.append("unit %d accepted %d commands, the simulator ran %d" %
                              (u, env.cmd_check_n[u], len(sim_obs["cmds"][u])))
    bad = compare(name, sim_obs, ro, env.errors, dut._log)
    chk.close()
    return bad, sim_obs, ro, env


async def _bring_up(dut):
    """Clock and reset at the first call of a cocotb test (cocotb ends a test's tasks at its end, the
    clock included), a reset afterward."""
    if E.CLK_TASK is None or E.CLK_TASK.done():
        await E.start(dut)
    else:
        await _reset(dut)


async def _reset(dut):
    dut.rst_n.value = 0
    for nm in ("ifq_valid", "xc_rsp_valid", "err_ack"):
        getattr(dut, nm).value = 0
    for u in E.UNIT_NAMES[:5]:
        getattr(dut, "%s_cmd_ready" % u).value = 0
        getattr(dut, "%s_done" % u).value = 0
    for nm in ("host_hold", "halt_hold", "qabort", "rv_reset", "host_clear", "sr_busy", "dbg_hold", "dbg_step",
               "rec_resync"):
        getattr(dut, nm).value = 0
    for _ in range(4):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


def seq_bases():
    c = json.loads(os.environ.get("TSR_ISSUE_CFG", "{}"))
    return c.get("seq_init", 0), c.get("pidx_init", 0)


# ---------------------------------------------------------------------------------------------
# cocotb tests
# ---------------------------------------------------------------------------------------------
@cocotb.test()
async def issue_random(dut):
    """Random schedule-independent programs (errors, BAD_DEP, holds, QCLEAR, QABORT recipe, host
    QABORT / QCLEAR, marks, REC bits, word masks), RTL vs simulator."""
    cfg, qd, c = env_cfg()
    seed = int(os.environ.get("TSR_SEED", "1"))
    n_prog = c.get("n_prog", 12)
    sb, pb = seq_bases()
    fails = []
    for p in range(n_prog):
        g = Gen(cfg, qd, seed * 1000 + p)
        chk = S.Sim(cfg, qd)
        g.sim_unit = lambda w: chk.check(w)[0]
        g.sim_code = lambda w: chk.check(w)[1]
        steps = g.program(n_ops=c.get("n_ops", 60))
        chk.close()
        kw = dict(lat=(1, 30) if p % 3 == 0 else (1, 6), ready_p=0.6 if p % 2 else 0.95,
                  rec_lat=(0, 12) if p % 4 == 1 else (0, 2), gap=(0, 8) if p % 2 else (0, 0),
                  resync_p=0.01 if p % 5 == 2 else 0.0)      # rec_resync: records re-sent, nothing else changes
        bad, *_ = await one_program(dut, cfg, qd, cfg["caps"], steps, seed * 77 + p, "random %d" % p, sb, pb, kw)
        if bad:
            fails.append((p, bad[:6]))
            dut._log.error("program %d: %s", p, "\n  ".join(bad[:10]))
    assert not fails, "%d of %d programs differ: %s" % (len(fails), n_prog, fails[:3])


@cocotb.test()
async def issue_traces(dut):
    """The trace sets of TSR_ISSUE_TRACES (glob of trace directories) replayed into the issue FIFO:
    RTL vs the simulator's replay of the same push.bin, and vs expect/tickets.bin."""
    pat = os.environ.get("TSR_ISSUE_TRACES", "")
    if not pat:
        return
    cfg0, qd0, c = env_cfg()
    dirs = sorted(d for p in pat.split(",") for d in glob.glob(str(REPO / p)) if os.path.exists(os.path.join(d, "meta.json")))
    lim = c.get("max_traces", 0)
    if lim:
        dirs = dirs[:lim]
    assert dirs, "no trace under %s" % pat
    seed = int(os.environ.get("TSR_SEED", "1"))
    fails = []
    n_run = n_ops = 0
    for i, d in enumerate(dirs):
        cfg, qd = trace_config(d)
        if (cfg["num_ch"], cfg["t_max"], cfg["k_max"], cfg["spm_bytes"], cfg["ch_bytes"], cfg["addr_beat_w"], cfg["caps"],
                qd) != (cfg0["num_ch"], cfg0["t_max"], cfg0["k_max"], cfg0["spm_bytes"], cfg0["ch_bytes"],
                        cfg0["addr_beat_w"], cfg0["caps"], qd0):
            continue
        steps = trace_steps(d)
        n_run += 1
        n_ops += sum(1 for st in steps if st[0] == "push")
        kw = dict(lat=(1, 40) if i % 2 else (1, 4), ready_p=0.7 if i % 3 else 1.0, gap=(0, 10) if i % 2 else (0, 0))
        bad, so, ro, env = await one_program(dut, cfg, qd, cfg["caps"], steps, seed + i, os.path.basename(d), env_kw=kw)
        exp = trace_expect(d)
        if exp is not None:
            for u in range(min(6, len(exp))):
                cnt = len(ro["done"][u])
                if cnt != exp[u]["count"]:
                    bad.append("tickets.bin unit %d count %d, RTL %d" % (u, exp[u]["count"], cnt))
                if exp[u]["code"]:
                    r = ro["rec"][u]
                    if r[0] != 1 or (r[2], r[1], r[3], r[4], r[5]) != (exp[u]["code"], exp[u]["seq"], exp[u]["info"],
                                                                     exp[u]["w0"], exp[u]["pidx"]):
                        bad.append("tickets.bin unit %d record %s, RTL %s" % (u, exp[u], r))
                elif ro["rec"][u][0]:
                    bad.append("tickets.bin unit %d: no record, RTL %s" % (u, ro["rec"][u]))
        if bad:
            fails.append((d, bad[:4]))
            dut._log.error("%s: %s", d, "\n  ".join(bad[:8]))
    dut._log.info("replayed %d of %d traces (%d pushes; the others have another configuration)", n_run,
                  len(dirs), n_ops)
    assert n_run, "no trace with this configuration"
    assert not fails, "%d traces differ: %s" % (len(fails), fails[:2])


async def _fresh(dut, cfg, qd, caps=0, **kw):
    sb, pb = seq_bases()
    env = E.Env(dut, cfg, qd, caps=caps, seed=int(os.environ.get("TSR_SEED", "1")), seq_init=sb, pidx_init=pb, **kw)
    await _bring_up(dut)
    env.start_loop()
    return env


@cocotb.test()
async def issue_latency(dut):
    """CC §9.3 latency targets: done of t -> start of an idle dependent on another unit <= 3 cycles;
    done -> start of the next entry of the same unit <= 2 cycles (CC §7.6 push -> start measured)."""
    cfg, qd, _ = env_cfg()
    env = await _fresh(dut, cfg, qd, gap=(0, 0), ready_p=1.0, lat=(20, 20))
    t0 = await env.push(isa.enc_ld(0, 0, 0, 4))
    await env.push(isa.enc_ld(0, 8, 8, 4))                       # same unit, behind t0
    await env.push(isa.enc_qact(0, 64, 0, 64, 1), t0)            # QACT depends on t0
    await env.wait_idle()
    d0 = env.done[0][0][4]
    s_dma1 = env.start[0][1][2]
    s_qact = env.start[1][0][2]
    dut._log.info("done(LD0) at %d, start(LD1) at %d, start(QACT) at %d; push -> start of LD0 %d cycles",
                  d0, s_dma1, s_qact, env.start[0][0][2])
    assert s_dma1 - d0 <= 2, "same-unit done -> start %d cycles" % (s_dma1 - d0)
    assert s_qact - d0 <= 3, "done -> dependent start %d cycles" % (s_qact - d0)
    assert not env.errors, env.errors
    env.stop()


@cocotb.test()
async def issue_hard_hold(dut):
    """DB's hard hold (CC §9.2): dbg_hold blocks non-running completions too; dbg_step lets one
    entry through; a QABORT releases it (aborted entries drain), as does SR_BUSY."""
    cfg, qd, _ = env_cfg()
    env = await _fresh(dut, cfg, qd, gap=(0, 0), ready_p=1.0, lat=(2, 4))
    env.ctl["dbg_hold"] = 0x3F
    t0 = await env.push(isa.enc_nop())
    t1 = await env.push([0x01, 0, 0, 0, 0, 0, 0, 0])           # refused opcode: ILLEGAL_OP on SYNC
    await env.push(isa.enc_ld(0, 0, 0, 1))
    await env.wait_cycles(200)
    assert all(not x for x in env.done), "an entry completed under the hard hold: %s" % env.done
    env.ctl["dbg_step"] = 1 << 5                                # one SYNC entry through
    await env.wait_cycles(1)
    env.ctl["dbg_step"] = 0
    await env.wait_cycles(50)
    assert len(env.done[5]) == 1 and env.done[5][0][2] == 0, env.done[5]
    env.ctl["qabort"] = 1                                       # QABORT: the hard hold is off
    await env.wait_cycles(100)
    assert len(env.done[5]) == 2 and env.done[5][1][2] == T.ERR_ILLEGAL_OP, env.done[5]
    assert len(env.done[0]) == 1 and env.done[0][0][2] == T.ERR_ABORTED, env.done[0]
    env.ctl["host_clear"] = 1                                   # QCLEAR: the hard hold is back
    await env.wait_cycles(10)
    await env.push(isa.enc_nop())
    await env.wait_cycles(100)
    assert len(env.done[5]) == 2, "NOP completed under the hard hold after the QCLEAR"
    env.ctl["sr_busy"] = 1                                      # SOFT_RESET in progress: off
    await env.wait_cycles(50)
    assert len(env.done[5]) == 3, env.done[5]
    env.ctl["sr_busy"] = 0
    env.ctl["dbg_hold"] = 0
    await env.wait_idle()
    assert not env.errors, env.errors
    del t0, t1
    env.stop()


@cocotb.test()
async def issue_holds(dut):
    """Hold sources (CC §9.6): host, RISC-V (QHOLD in push order), debug halt; each holds running ops
    only (non-running entries drain), and setting / clearing another source does not release it."""
    cfg, qd, _ = env_cfg()
    env = await _fresh(dut, cfg, qd, gap=(0, 0), ready_p=1.0, lat=(2, 4))

    async def set_src(src, v):
        if src == "rv":
            await env.qctrl(2 if v else 4)
        else:
            env.ctl[src] = v
        await env.wait_cycles(4)

    for src in ("host_hold", "halt_hold", "rv"):
        n_start, n_sync = len(env.start[0]), len(env.done[5])
        await set_src(src, 1)
        await env.push(isa.enc_ld(0, 0, 0, 1))
        await env.push(isa.enc_nop())
        await env.wait_cycles(150)
        assert len(env.start[0]) == n_start, "an op started under %s" % src
        assert len(env.done[5]) == n_sync + 1, "the NOP did not drain under %s" % src
        for other in ("host_hold", "halt_hold", "rv"):
            if other != src:
                await set_src(other, 1)
                await set_src(other, 0)
        await env.wait_cycles(150)
        assert len(env.start[0]) == n_start, "clearing another source released %s" % src
        await set_src(src, 0)
        await env.wait_idle()
        assert len(env.start[0]) == n_start + 1, "%s: the op did not start after the release" % src
    assert not env.errors, env.errors
    env.stop()


@cocotb.test()
async def issue_i2_stall(dut):
    """I2 (CC §9.4): with the record handshake stalled, the error entry's completion waits; QCLEAR's
    own completion waits for the cleared records too."""
    cfg, qd, _ = env_cfg()
    env = await _fresh(dut, cfg, qd, gap=(0, 0), ready_p=1.0, lat=(2, 4))
    env.rec_stall = True
    await env.push([0x02, 0, 0, 0, 0, 0, 0, 0])                 # ILLEGAL_OP on SYNC
    await env.wait_cycles(200)
    assert env.done[5] == [], "error entry completed before its record handshake"
    env.rec_stall = False
    await env.wait_cycles(50)
    assert len(env.done[5]) == 1 and env.done[5][0][2] == T.ERR_ILLEGAL_OP
    env.rec_stall = True
    await env.qctrl(8)
    await env.wait_cycles(200)
    assert len(env.done[5]) == 1, "QCLEAR completed before the cleared records were shown"
    env.rec_stall = False
    await env.wait_idle()
    assert len(env.done[5]) == 2
    assert not env.errors, env.errors
    env.stop()


@cocotb.test()
async def issue_accept_ahead(dut):
    """MVM accept-ahead (CC §9.3, I3 <= 2): with cmd_ready kept high the MVM unit is offered a second op
    before its done, never a third; completions stay in accept order."""
    cfg, qd, _ = env_cfg()
    if qd[2] == 0:
        return
    env = await _fresh(dut, cfg, qd, gap=(0, 0), ready_p=1.0, lat=(60, 60))
    for i in range(4):
        await env.push(isa.enc_mvm(0, 64 * i, 1024, 0, 16, 256, 1))
    await env.wait_idle()
    st = [c for (_, _, c) in env.start[2]]
    dn = [c for (*_, c) in env.done[2]]
    assert st[1] < dn[0], "no accept-ahead: start %s done %s" % (st, dn)
    assert st[2] > dn[0], "a third op was accepted: start %s done %s" % (st, dn)
    assert not env.errors, env.errors
    env.stop()


@cocotb.test()
async def issue_backpressure(dut):
    """Credits (CC §7.3): a stream to one unit with a slow unit fills its queue; pushes wait for credits
    and the queue never overflows (RTL assertion); S[u] returns credits at accept."""
    cfg, qd, _ = env_cfg()
    env = await _fresh(dut, cfg, qd, gap=(0, 0), ready_p=1.0, lat=(400, 400))
    n = 3 * qd[0] + 2
    pushed_at = []
    for i in range(n):
        await env.push(isa.enc_ld(0, i, i, 1))
        pushed_at.append(env.cycle)
    await env.wait_idle()
    assert len(env.done[0]) == n
    # a queue slot returns at the accept (S[u]), not at the done: push QD + 1 goes in while op 0 runs
    first_done = env.done[0][0][4]
    assert pushed_at[qd[0]] < first_done, "push %d at %d, first done at %d: credit not returned at accept" % (
        qd[0], pushed_at[qd[0]], first_done)
    assert not env.errors, env.errors
    env.stop()


def _rec_of(v, u):
    """Unit u's error record from the err_rec bus (129 bits per unit: err, seq[27:0], code, w0, pidx)."""
    r = (v >> (129 * u)) & ((1 << 129) - 1)
    return dict(err=(r >> 128) & 1, seq=(r >> 100) & 0xFFFFFFF, code=(r >> 96) & 0xF, w0=(r >> 32) & 0xFFFFFFFF,
                pidx=r & 0xFFFFFFFF)


@cocotb.test()
async def issue_abort_offered(dut):
    """QABORT while a head has offered a command the unit has not taken (CC §9.5: ERR_SEQ = S[u], the
    first entry not started; the offered command counts as started, as in the simulator's q_head). The
    DMA unit takes commands slowly (ready_p 0.2), a host QABORT or a RISC-V QABORT (CTRL entry) lands at
    a random cycle of a six-entry stream, and then every entry before ERR_SEQ must have run (or ended with
    its own code) and every entry from ERR_SEQ on must have ended ABORTED without running. Kills a
    S[u] + cmd_v[u] -> S[u] mutation (test_issue_mutants: issue_abort_seq_offered), which the random
    programs, which abort only behind a QHOLD or at idle points, do not reach."""
    cfg, qd, _ = env_cfg()
    env = await _fresh(dut, cfg, qd, gap=(0, 0), ready_p=0.2, lat=(1, 4), rec_lat=(0, 2))
    r = random.Random(int(os.environ.get("TSR_SEED", "1")) * 977 + 11)
    bad, hits = [], [0, 0]
    seen = {"offered": 0}

    async def watch():       # the abort applies in the cycle abort_rq_q is set: was a command on offer?
        while True:
            await FallingEdge(dut.clk)
            if int(dut.abort_rq_q.value) and int(dut.dma_cmd_valid.value):
                seen["offered"] += 1

    mon = cocotb.start_soon(watch())
    for rnd in range(120):
        mode = rnd & 1                                   # 0 host QABORT, 1 RISC-V QABORT
        n0 = len(env.done[0])
        o0 = seen["offered"]
        for k in range(6):
            await env.push(isa.enc_ld(r.randrange(cfg["num_ch"]), 8 * k, 8 * k, 1))
        await env.wait_cycles(r.randint(0, 40))
        if mode == 0:
            env.ctl["qabort"] = 1
        else:
            await env.qctrl(1)
        await env.wait_idle()
        hits[mode] += seen["offered"] - o0
        rec = _rec_of(int(dut.err_rec.value), 0)
        for (seq, pidx, code, ran, cyc) in env.done[0][n0:]:
            before = ((seq - rec["seq"]) & 0xFFFFFFF) >= (1 << 27)
            if before and not (ran or code not in (0, T.ERR_ABORTED)):
                bad.append("round %d: seq %d < ERR_SEQ %d completed ABORTED / not run" % (rnd, seq, rec["seq"]))
            if not before and (ran or code != T.ERR_ABORTED):
                bad.append("round %d: seq %d >= ERR_SEQ %d ran %d code %d" % (rnd, seq, rec["seq"], ran, code))
        if not rec["err"] or rec["code"] != T.ERR_ABORTED:
            bad.append("round %d: DMA record %s" % (rnd, rec))
        env.ctl["host_clear"] = 1
        await env.wait_cycles(12)
        if len(bad) > 10:
            break
    mon.cancel()
    dut._log.info("abort_offered: %d host / %d RISC-V aborts hit an offered command; %d findings", hits[0],
                  hits[1], len(bad))
    for b in bad[:10]:
        dut._log.error(b)
    assert hits[0] > 0 and hits[1] > 0, "no abort hit an offered command (%s)" % hits
    assert not bad, bad[:3]
    assert not env.errors, env.errors
    env.stop()


def _hs_of(v, u):
    """The 128-bit handshake word (tsr_ctrl_pkg::err_rec_t: {pidx, w0, info, code, seq}, code 0 = no error) that
    unit u's record (err_rec bus) is sent as."""
    r = (v >> (129 * u)) & ((1 << 129) - 1)
    err = (r >> 128) & 1
    seq, code, info, w0, pidx = (r >> 100) & 0xFFFFFFF, (r >> 96) & 0xF, (r >> 64) & 0xFFFFFFFF, \
        (r >> 32) & 0xFFFFFFFF, r & 0xFFFFFFFF
    return (pidx << 96) | (w0 << 64) | (info << 32) | ((code if err else 0) << 28) | seq


@cocotb.test()
async def issue_rec_resync(dut):
    """rec_resync (tsr_ctrl_cdc: clk_rv is OK again after a stretch in which record handshakes were answered
    locally): every present unit's record goes through the handshake once more, with the current content
    (an error record, then the all-zero record after a QCLEAR), D waits at ERR_SEQ for the re-send (I2 is an
    RTL assertion), and nothing else changes (the random programs also take random pulses)."""
    cfg, qd, _ = env_cfg()
    env = await _fresh(dut, cfg, qd, gap=(0, 0), ready_p=1.0, lat=(2, 4), rec_lat=(0, 3))
    present = [u for u in range(6) if qd[u]]

    async def resync_and_check(what):
        want = [_hs_of(int(dut.err_rec.value), u) for u in range(6)]
        sent0 = list(env.rec_sent)
        env.rec_seen = [None] * 6
        env.ctl["rec_resync"] = 1
        await env.wait_cycles(60)
        for u in present:
            assert env.rec_sent[u] == sent0[u] + 1, "%s: unit %d record sent %d times" % (
                what, u, env.rec_sent[u] - sent0[u])
            assert env.rec_seen[u] == want[u], "%s: unit %d re-sent %#x, record %#x" % (what, u, env.rec_seen[u], want[u])
        for u in set(range(6)) - set(present):
            assert env.rec_sent[u] == sent0[u], "%s: absent unit %d sent a record" % (what, u)
        return want

    await resync_and_check("idle, no error")                       # the all-zero records
    await env.push(isa.enc_ld(0, 1 << 30, 0, 1))                   # SPM_RANGE on the DMA unit
    await env.push(isa.enc_ld(0, 0, 0, 1))                         # poisoned behind it
    await env.wait_idle()
    want = await resync_and_check("DMA error")
    assert want[0] & 0xFFFFFFF == 0 and (want[0] >> 28) & 0xF == T.ERR_SPM_RANGE, "no DMA record: %#x" % want[0]
    # a resync while the first record of an error entry is still in flight: the error entry's D step waits
    env.ctl["host_clear"] = 1
    await env.wait_cycles(20)
    env.rec_stall = True
    await env.push([0x02, 0, 0, 0, 0, 0, 0, 0])                    # ILLEGAL_OP on SYNC
    await env.wait_cycles(30)
    env.ctl["rec_resync"] = 1
    await env.wait_cycles(100)
    assert env.done[5] == [] or env.done[5][-1][2] != T.ERR_ILLEGAL_OP, "error entry completed under the stall"
    env.rec_stall = False
    await env.wait_idle()
    assert env.done[5][-1][2] == T.ERR_ILLEGAL_OP
    await resync_and_check("SYNC error")
    env.ctl["host_clear"] = 1                                      # QCLEAR: the records are zero again
    await env.wait_cycles(40)
    want = await resync_and_check("after QCLEAR")
    assert not any(want), "records not clear after the QCLEAR: %s" % [hex(x) for x in want]
    assert not env.errors, env.errors
    env.stop()


# ---------------------------------------------------------------------------------------------
# pytest launchers
# ---------------------------------------------------------------------------------------------
def _params(cfg, qd, caps, extra=None):
    p = dict(NUM_CH=cfg["num_ch"], T_MAX=cfg["t_max"], K_MAX=cfg["k_max"], SPM_BYTES=cfg["spm_bytes"],
             CH_BYTES="64'd%d" % cfg["ch_bytes"], ADDR_BEAT_W=cfg["addr_beat_w"], CAPS="32'd%d" % caps,
             QD_DMA=qd[0], QD_QACT=qd[1], QD_MVM=qd[2], QD_VU=qd[3], QD_ATT=qd[4], QD_SYNC=qd[5])
    p.update(extra or {})
    return p


def _run(name, cfg, qd, extra_cfg=None, extra_params=None, tests=None):
    c = dict(num_ch=cfg["num_ch"], t_max=cfg["t_max"], k_max=cfg["k_max"], spm_bytes=cfg["spm_bytes"],
             ch_bytes=cfg["ch_bytes"], addr_beat_w=cfg["addr_beat_w"], caps=cfg["caps"], qd=list(qd))
    c.update(extra_cfg or {})
    env = {"TSR_ISSUE_CFG": json.dumps(c)}
    if "traces" in c:
        env["TSR_ISSUE_TRACES"] = c["traces"]
    run("tsr_issue", __name__, parameters=_params(cfg, qd, cfg["caps"], extra_params), sources=core_src(),
        name="issue_" + name, extra_env=env, testcase=tests)


U200 = S.config(num_ch=4, t_max=8)


def test_issue_u200():
    """U200 M6 queues (DMA 8, QACT 4, MVM 4, VU 16, AT absent, SYNC 4), T_MAX 8, no VU caps."""
    _run("u200", U200, S.U200_QD, dict(n_prog=16, n_ops=70))


def test_issue_small():
    """3 channels (serial MVM divider), 256 KiB SPM, narrow DRAM, T_MAX 3, K_MAX 4096."""
    cfg = S.config(num_ch=3, t_max=3, k_max=4096, spm_bytes=256 << 10, ch_bytes=1 << 30, addr_beat_w=22)
    _run("small", cfg, S.U200_QD, dict(n_prog=12, n_ops=50))


def test_issue_nomvm():
    """No MVM unit (queue depth 0): MVM refused, deps on unit 2 are BAD_DEP; 1 channel."""
    cfg = S.config(num_ch=1, t_max=1, k_max=4096, spm_bytes=64 << 10, ch_bytes=1 << 30)
    _run("nomvm", cfg, (8, 4, 0, 16, 0, 4), dict(n_prog=8, n_ops=40))


def test_issue_depths():
    """Queue depth 1 everywhere, then 31 (CC §7.4 parameters)."""
    _run("shallow", U200, (1, 1, 1, 1, 0, 1), dict(n_prog=6, n_ops=40), tests=["issue_random", "issue_backpressure"])
    _run("deep", U200, (31, 31, 31, 31, 0, 31), dict(n_prog=6, n_ops=80), tests=["issue_random", "issue_backpressure"])


def test_issue_wrap():
    """Ticket wrap (CC §15.2): every sequence starts 40 below 2^28 and the push index 30 below 2^32."""
    sb, pb = (1 << 28) - 40, (1 << 32) - 30
    _run("wrap", U200, S.U200_QD, dict(n_prog=6, n_ops=60, seq_init=sb, pidx_init=pb),
         dict(SEQ_INIT="28'd%d" % sb, PIDX_INIT="32'd%d" % pb), tests=["issue_random"])


def test_issue_vu():
    """CAPS VU + ATT with the VU ops through the external class port (the simulator's VU / AT checks)."""
    cfg = S.config(num_ch=4, t_max=8, caps=33)
    _run("vu", cfg, S.U200_QD, dict(n_prog=8, n_ops=60), tests=["issue_random"])


def _trace_dirs(pattern):
    return [d for d in glob.glob(str(REPO / pattern)) if os.path.exists(os.path.join(d, "meta.json"))]


TRACE_SETS = ["ctrl/nocaps", "ctrl/errors", "ctrl/tmax8", "ctrl/deep", "ctrl/shallow", "m2/errors", "m2/errors8",
              "m2/tmax8", "m2/tmax1", "m2/small1ch", "m2/ch3", "m3/errors8", "m3/errors", "m3/tmax8", "m3/ch3",
              "gen/tmax8", "gen/tmax1"]


@pytest.mark.traces
@pytest.mark.parametrize("tset", TRACE_SETS, ids=[t.replace("/", "_") for t in TRACE_SETS])
def test_issue_traces(tset):
    """Trace sets (traces/<set>/*) replayed into the issue FIFO, RTL vs simulator vs tickets.bin."""
    dirs = _trace_dirs("traces/%s/*" % tset)
    if not dirs:
        pytest.skip("traces/%s not on disk (traces/README.md: regen.sh)" % tset)
    cfg, qd = trace_config(dirs[0])
    lim = int(os.environ.get("TSR_ISSUE_MAX_TRACES", "0"))
    _run("tr_" + tset.replace("/", "_"), cfg, qd, dict(traces="traces/%s/*" % tset, max_traces=lim),
         tests=["issue_traces"])
