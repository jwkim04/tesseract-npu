"""tb/dbg — the DB block (rtl/core/tsr_dbg*.sv) with tsr_csr and tsr_issue against the simulator's DBG
model (sw/sim/tsr_dbg.c) and DM model (sw/sim/tsr_dm.c) (docs/design/05-debug.md "DB" §14 items 1-2,
6-7; docs/design/06-software-verification.md "SW" §10.3).

The wrapper tb/dbg/sv/tsr_dbg_tb.sv wires the blocks as tsr_core_dp / tsr_core do; tb/dbg/dbg_tb.py has
the bench (tb/issue's environment for the issue FIFO and the units, tb/csr's clk_rv model, tsr_ctrl_rv's
halt-request register, the clk_rv DB window, the trace writer's memory, a DMA write stream) and the
step runner that executes the same push stream on both sides with DB's stop semantics.

  dbg_regs     every DBG register: reset values, access types and masks, W1C, PTR_ERR, random writes,
               unlisted offsets, the core-window view, against the simulator
  dbg_dmi      the DMI gate (DB-2): DMI_EN = 0 reads 0 and drops writes without reaching the DM
               (DMI_BLOCKED), DMI_EN = 1 reaches the DM model (two tsr_dm instances, same accesses),
               core-window reads 0, DMI_RESP_ERR
  dbg_stops    random push streams with STOP_AT stops, QSTEP (DBG_STEP_N 1 and 3), RELEASE, STOP_NOW at a
               settled point and every tile-breakpoint kind (PIDX, OPCODE, SPM, DRAM, TICKET; AFTER,
               COUNT_ONLY, HALT_RV): at every DRAINED stop the stop state, TBP hits, completion counts,
               error records, oldest entries, unit probes and the history equal the simulator's; at
               the end the per-unit completion streams (codes included), the history and the trace
               ring (records per kind and unit)
  dbg_serial   serial mode: completions in push order over all units (RTL strobes), final state and
               history equal the simulator's
  dbg_hist     the history over 3 x 512 entries (slot = pidx mod 512, PRESENT), HIST_CLEAR,
               HIST_FREEZE_ERR (256 entries after the first error), HIST_FREEZE; every register of every
               slot is compared, a slot that is not PRESENT (wrapped over, not written yet, cleared) reads 0
  dbg_trw      the trace writer: an overwrite ring with wrap, TRW_STOP_FULL drops, no ring, a ring on
               another channel than TRW_CH (all dropped), class mask,
               LAUNCH_END flush, with a DMA write stream on the same write source (the mux): ring
               records, WPTR, DROPS, PERF TRW_RECORDS / TRW_DROPS against the simulator; the RTL's own
               record times against its strobes; every DMA burst lands
  dbg_rec      completion records through the clk_rv window: REC flag and REC_CFG.ALL, markers,
               REC_COUNT, POP, irq_fast[4], 64-record overflow (REC_DROPPED, REC_DROP)
  dbg_halt     DB's halt requests (STOP_HALT_RV, a TBP's HALT_RV, RV_HALT_EN, one-shot under RV_HALTED,
               RV_HALT_CANCEL), HALT_ON_BOOT, dbg_stop, DBG_FLAGS in the window, against the simulator
               (RV_HALT_PEND) and tsr_ctrl_rv's pend rule
  dbg_abort    the hard hold against QABORT (held entries complete ABORTED) and SOFT_RESET (stop state
               cleared, configuration kept), against the simulator; directed cases whose held head is a
               NOP, a FENCE, an illegal opcode or a null LD (the first three decide before the hold, so
               SOFT_RESET must keep them held until its QABORT has applied)
  dbg_misc     RTL-only: ERR_STOP, HUNG_STOP and the history freeze, QSTEP under the debug-halt hold
               (dbg_step), RV_SAMPLE / RVWP_APPLY timeouts (no taps), MATCHQ_CLEAR
  dbg_rvtaps   RTL-only, HAS_RV_TAPS = 1 (test_dbg_rvtaps): RV watchpoints on synthetic Ibex taps (kinds,
               overlap edges, HITS, WPADDR, RVWP_HIT, halt requests), RV_SAMPLE, RV_DFAULT
Every RTL run also checks, every cycle, M1 (DRAINED implies the push-order prefix: every entry before
T completed, DBG_NEXT_PIDX at or after T) and M2 (no entry at or after T starts while held).
Mutation spot checks: test_dbg_mutants (12 mutants, opt-in, TSR_DBG_MUTANTS=1).
"""
from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import cocotb
import pytest
from cocotb.triggers import FallingEdge, RisingEdge

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent / "issue"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from common import isa, tables as T                      # noqa: E402
from common.runner import REPO, build_root, run           # noqa: E402

import issue_sim as S                                     # noqa: E402
import dbg_tb as D                                        # noqa: E402

M32 = 0xFFFFFFFF
DBG_SRC = ["tsr_dbg_stop.sv", "tsr_dbg_shq.sv", "tsr_dbg_hist.sv", "tsr_dbg_trw.sv", "tsr_dbg.sv", "tsr_dbg_rv.sv",
           "tsr_dbg_wrmux.sv"]


def rtl(name):
    """A DB source; $TSR_MUT_DIR overrides it with a mutated copy (test_dbg_mutants)."""
    d = os.environ.get("TSR_MUT_DIR")
    if d and (Path(d) / name).exists():
        return Path(d) / name
    return REPO / "rtl/core" / name


# what the wrapper instantiates (not the whole rtl/filelist.f: the bench does not depend on the
# units' sources)
BASE_SRC = ["rtl/core/tsr_tables_pkg.sv", "rtl/core/tsr_pkg.sv", "rtl/tech/generic/tsr_sram_1r1w.sv",
            "rtl/ctrl/cdc/tsr_sync.sv", "rtl/ctrl/cdc/tsr_rst_sync.sv", "rtl/ctrl/cdc/tsr_cdc_gray.sv",
            "rtl/ctrl/cdc/tsr_cdc_fifo.sv", "rtl/ctrl/cdc/tsr_cdc_hs.sv", "rtl/core/tsr_csr.sv", "rtl/core/tsr_chk.sv",
            "rtl/core/tsr_issue.sv"]


MUT_BASE = ("tsr_issue.sv",)      # base sources a mutant may replace (the hard hold lives in tsr_issue)


def sources():
    return [rtl(Path(f).name) if Path(f).name in MUT_BASE else REPO / f for f in BASE_SRC] + \
           [rtl(n) for n in DBG_SRC] + [HERE / "sv/tsr_dbg_tb.sv"]


def env_cfg():
    c = json.loads(os.environ.get("TSR_DBG_CFG", "{}"))
    cfg = S.config(num_ch=c.get("num_ch", 4), t_max=c.get("t_max", 8))
    return cfg, tuple(c.get("qd", S.U200_QD)), c


def seed():
    return int(os.environ.get("TSR_SEED", "1"), 0)


def gen(cfg, qd, s):
    from test_issue import Gen
    g = Gen(cfg, qd, s)
    chk = S.Sim(cfg, qd)
    g.sim_unit = lambda w: chk.check(w)[0]
    g.sim_code = lambda w: chk.check(w)[1]
    g._chk = chk
    return g


def fail(dut, bad, what):
    for x in bad[:25]:
        dut._log.error(x)
    assert not bad, "%s: %d mismatches" % (what, len(bad))


# =================================================================================================
# The stop scenario: one push stream on both sides, stops, actions, comparisons
# =================================================================================================
async def scenario(dut, cfg, qd, steps, name, pre=(), actions=(), seed_=1, env_kw=None, hist=True, trw=None,
                   final_regs=True, b=None, ds=None):
    """Runs steps on the simulator (DbgSim) and on the RTL (Bench) with the register writes `pre` first.
    At every blocked point (a push without credit at DRAINED) both are compared and the next action is
    applied: ("qstep", n) | ("release",) | ("stop_now",). When the stream has ended and a stop is
    still active, the same happens until no stop is left. trw = (base, lines): compare the ring."""
    own = ds is None
    if own:
        ds = D.DbgSim(cfg, qd)
        b = D.Bench(dut, cfg, qd, seed=seed_, env_kw=env_kw)
        await b.start()
    bad = []
    for off, v in pre:
        ds.wr(off, v)
        await b.wr(off, v)
    st = dict(k=0, tickets={})
    acts = list(actions)
    nstop = 0
    regs = D.snap_regs(cfg["num_ch"])
    while True:
        ok_s = ds.run(steps)
        ok_r = await b.run(steps, st)
        if ds.k != st["k"] or ok_s != ok_r:
            bad.append("%s: stream at step %d (%s, sim) / %d (%s, RTL)" % (name, ds.k, ok_s, st["k"], ok_r))
            break
        ds.settle()
        await b.settle()
        act_s, act_r = bool(ds.status() & D.ST_STOP_ACTIVE), bool(b.stop_active())
        if act_s != act_r:
            bad.append("%s: STOP_ACTIVE sim %d RTL %d at step %d" % (name, act_s, act_r, ds.k))
            break
        if not act_s:
            if ok_s:
                break
            bad.append("%s: blocked at step %d without a stop" % (name, ds.k))
            break
        nxt, t = ds.rd(T.CSR_DBG_NEXT_PIDX), ds.rd(T.CSR_DBG_STOP_PIDX)
        if ok_s and ((nxt - t) & M32) >= 0x80000000:
            # the stream ended before T: nothing to drain to; release
            ds.wr(T.CSR_DBG_CTRL, T.DBG_CTRL_RELEASE)
            await b.wr(T.CSR_DBG_CTRL, T.DBG_CTRL_RELEASE)
            continue
        if b.errors:                             # an RTL monitor fired: stop here
            break
        ds.wait_drained()
        await b.wait_drained()
        tag = "%s stop %d (step %d)" % (name, nstop, ds.k)
        bad += await D.compare_regs(b, ds, regs, tag)
        bad += await D.compare_units(b, ds, tag)
        nxt, t = ds.rd(T.CSR_DBG_NEXT_PIDX), ds.rd(T.CSR_DBG_STOP_PIDX)
        if hist:
            bad += await D.compare_hist(b, ds, max(0, nxt - 40), nxt + 6, tag)     # + slots not written yet
        if bad or nstop > 400:
            break
        act = acts.pop(0) if acts else ("release",)
        nstop += 1
        if act[0] == "qstep":
            # step only entries already dispatched (the stream waits while a QSTEP runs)
            n = min(act[1], (nxt - t) & M32)
            act = ("qstep", n) if 0 < n < 0x10000 else ("release",)
        if act[0] == "qstep":
            ds.wr(T.CSR_DBG_STEP_N, act[1])
            await b.wr(T.CSR_DBG_STEP_N, act[1])
            ds.wr(T.CSR_DBG_CTRL, T.DBG_CTRL_QSTEP)
            await b.wr(T.CSR_DBG_CTRL, T.DBG_CTRL_QSTEP)
            ds.step_done()
            await b.step_done()
        elif act[0] == "stop_now":
            ds.wr(T.CSR_DBG_CTRL, T.DBG_CTRL_RELEASE)
            await b.wr(T.CSR_DBG_CTRL, T.DBG_CTRL_RELEASE)
            ds.settle()
            await b.settle()
            ds.wr(T.CSR_DBG_CTRL, T.DBG_CTRL_STOP_NOW)
            await b.wr(T.CSR_DBG_CTRL, T.DBG_CTRL_STOP_NOW)
        else:
            ds.wr(T.CSR_DBG_CTRL, T.DBG_CTRL_RELEASE)
            await b.wr(T.CSR_DBG_CTRL, T.DBG_CTRL_RELEASE)
    bad += b.errors[:10]
    if not bad:
        ds.idle()
        await b.idle()
        tag = "%s final" % name
        bad += D.compare_done(b, ds, tag)
        if final_regs:
            bad += await D.compare_regs(b, ds, regs, tag)
        nxt = ds.rd(T.CSR_DBG_NEXT_PIDX)
        if hist:
            bad += await D.compare_hist(b, ds, max(0, nxt - 520), nxt + 6, tag)    # wrapped-over and unwritten slots too
        if trw is not None:
            bad += await compare_trw(b, ds, *trw, tag)
    dut._log.info("%s: %d steps, %d stops, %s", name, len(steps), nstop, "ok" if not bad else "MISMATCH")
    if own:
        ds.close()
    return bad, b, ds


async def compare_trw(b, ds, base, lines, tag, full=True):
    """Flushes both trace writers and compares WPTR, DROPS and the ring records (DB §8.3)."""
    bad = []
    ds.wr(T.CSR_DBG_CTRL, T.DBG_CTRL_TRW_FLUSH)
    await b.wr(T.CSR_DBG_CTRL, T.DBG_CTRL_TRW_FLUSH)
    for _ in range(200):                         # the writer's bursts and responses
        await b.cycles(20)
        if int(b.dut.u_dbg.g_trw.u_trw.ws_q.value) == 0 and int(b.dut.u_dbg.g_trw.u_trw.ob_cnt_q.value) == 0:
            break
    await b.cycles(50)
    wa, we = await b.rd(T.CSR_DBG_TRW_WPTR), ds.rd(T.CSR_DBG_TRW_WPTR)
    da, de = await b.rd(T.CSR_DBG_TRW_DROPS), ds.rd(T.CSR_DBG_TRW_DROPS)
    if (wa, da) != (we, de):
        bad.append("%s: TRW WPTR / DROPS RTL %d / %d sim %d / %d" % (tag, wa, da, we, de))
        return bad
    cap = 2 * lines
    lo = max(0, we - cap)
    ra, re_ = b.ring(base, lines, lo, wa), ds.ring(base, lines, lo, we)
    if any(x is None for x in ra):
        bad.append("%s: RTL ring has unwritten records in [%d, %d)" % (tag, lo, wa))
        return bad
    if wa > cap:
        # a wrapped overwrite ring keeps the last `cap` records of each side's own record order, a
        # schedule-dependent subset: the totals above are compared; the window is checked against
        # the RTL's own completion stream (every OP record is its entry's, units in completion order)
        bad += trw_self_check(b, ra, tag)
        return bad
    va, ve = D.trw_view(ra), D.trw_view(re_)
    for k in ("mark", "stop", "rvhalt", "other"):
        if va[k] != ve[k]:
            bad.append("%s: TRW %s records RTL %s sim %s" % (tag, k, va[k][:6], ve[k][:6]))
    if full:
        for u in range(8):
            if va["op"][u] != ve["op"][u]:
                bad.append("%s: TRW OP records of unit %d RTL %s sim %s" % (tag, u, va["op"][u][:4], ve["op"][u][:4]))
        if va["drop"] != ve["drop"]:
            bad.append("%s: TRW DROP records RTL %s sim %s" % (tag, va["drop"], ve["drop"]))
    # the RTL's times against its own strobes: RUN = completion - start of the running ops
    starts = {}
    for u in range(6):
        for (s, p, c) in b.env.start[u]:
            starts[(u, s)] = c
    dones = {}
    for u in range(6):
        for (s, p, code, ran, c) in b.env.done[u]:
            dones[(u, s)] = c
    for w in ra:
        if (w[0] & 0xF) == T.TRW_W0_KIND_OP and (w[0] >> 12) & 1:
            u, s = (w[0] >> 4) & 0xF, w[2] & D.SEQ_MASK
            if (u, s) in starts and (u, s) in dones and w[5] != dones[(u, s)] - starts[(u, s)]:
                bad.append("%s: OP record %s RUN %d, strobes say %d" % (tag, hex(w[2]), w[5],
                                                                         dones[(u, s)] - starts[(u, s)]))
                break
    return bad


def trw_self_check(b, recs, tag):
    """OP records against the RTL's monitored enqueues and completions (DB §8.3 fields, times and the
    TBP flag excluded); per unit the records are in completion order."""
    bad = []
    env = b.env
    done = {}
    for u in range(6):
        for (seq, p, code, ran, _) in env.done[u]:
            done[(u, seq)] = (p, code, ran)
    enq = {}
    for u in range(6):
        for e in env.enq[u]:
            enq[(u, e["seq"])] = e
    last = {}
    for w in recs:
        if (w[0] & 0xF) != T.TRW_W0_KIND_OP:
            continue
        u, seq = (w[0] >> 4) & 0xF, w[2] & D.SEQ_MASK
        d, e = done.get((u, seq)), enq.get((u, seq))
        if d is None or e is None or (w[2] >> 28) != u:
            bad.append("%s: OP record of unknown entry %#x" % (tag, w[2]))
            continue
        p, code, ran = d
        want = (T.TRW_W0_KIND_OP | (u << 4) | (code << 8) | (ran << 12) | (e["rec"] << 13) |
                ((e["w"][0] & 0xFF) << 16) | (((e["w"][0] >> 8) & 0xFF) << 24))
        if (w[0] & ~T.TRW_W0_TBP & M32) != want or w[1] != p:
            bad.append("%s: OP record %#x: w0 %#x pidx %d, entry says %#x / %d" % (tag, w[2], w[0], w[1], want, p))
        if u in last and ((seq - last[u]) & D.SEQ_MASK) >= (1 << 27):
            bad.append("%s: OP records of unit %d out of completion order" % (tag, u))
        last[u] = seq
        if len(bad) > 6:
            break
    return bad


# =================================================================================================
# cocotb tests
# =================================================================================================
REG_RW = [(T.CSR_DBG_CFG, 0xFFF), (T.CSR_DBG_STEP_N, 0xFFFF), (T.CSR_DBG_TRW_BASE, 0xFFFFFFF0),
          (T.CSR_DBG_TRW_MASK, 0x7FF), (T.CSR_DBG_HIST_IDX, M32)] + \
         [(T.CSR_DBG_TBP_CTL(i), 0xFFFF01FF) for i in range(4)] + [(T.CSR_DBG_TBP_A(i), M32) for i in range(4)] + \
         [(T.CSR_DBG_TBP_B(i), M32) for i in range(4)] + [(T.CSR_DBG_RVWP_CTL(j), 0xF) for j in range(2)] + \
         [(T.CSR_DBG_RVWP_LO(j), M32) for j in range(2)] + [(T.CSR_DBG_RVWP_HI(j), M32) for j in range(2)]
# model values (DB §9.1) and live values, masked in the register comparison
REG_MASK = {T.CSR_DBG_MTIME_LO: 0, T.CSR_DBG_MTIME_HI: 0, T.CSR_DBG_QSTATE: 0,
            T.CSR_DBG_RV_STATE: ~T.DBG_RV_STATE_NDMRESET & M32}
for _i in range(8):
    REG_MASK[T.CSR_DBG_U_STATE(_i)] = D.U_STATE_MASK
    REG_MASK[T.CSR_DBG_U_AGE(_i)] = 0


def all_dbg_offs():
    return [o for o in range(0x800, 0xB00, 4) if not (0x9C0 <= o < 0xA00)]


@cocotb.test()
async def dbg_regs(dut):
    """Every DBG register against the simulator: reset values, masks, W1C, PTR_ERR, unlisted offsets."""
    cfg, qd, _ = env_cfg()
    ds = D.DbgSim(cfg, qd)
    b = D.Bench(dut, cfg, qd, seed=seed())
    await b.start()
    r = random.Random(seed() * 13)
    offs = all_dbg_offs()
    bad = await D.compare_regs(b, ds, offs, "reset", REG_MASK)
    for i in range(150):
        off, m = r.choice(REG_RW)
        v = r.choice([0, M32, r.getrandbits(32), r.getrandbits(32) & m, 1 << r.randrange(32)])
        if off == T.CSR_DBG_CFG:
            v &= ~T.DBG_CFG_DMI_EN                # DMI: dbg_dmi
        ds.wr(off, v)
        await b.wr(off, v)
        if i % 10 == 9:
            bad += await D.compare_regs(b, ds, [o for o, _ in REG_RW], "rw %d" % i, REG_MASK)
    # TRW_LINES (power of two >= 16, else PTR_ERR), RPTR beyond WPTR (PTR_ERR), STATUS W1C
    for v in (16, 15, 0, 48, 1 << 20, 64, 3, 1 << 31):
        ds.wr(T.CSR_DBG_TRW_LINES, v)
        await b.wr(T.CSR_DBG_TRW_LINES, v)
        bad += await D.compare_regs(b, ds, [T.CSR_DBG_TRW_LINES, T.CSR_DBG_STATUS], "lines %#x" % v)
    for v in (0, 1, M32):
        ds.wr(T.CSR_DBG_TRW_RPTR, v)
        await b.wr(T.CSR_DBG_TRW_RPTR, v)
    for v in (0, T.DBG_STATUS_PTR_ERR, M32):
        ds.wr(T.CSR_DBG_STATUS, v)
        await b.wr(T.CSR_DBG_STATUS, v)
        bad += await D.compare_regs(b, ds, [T.CSR_DBG_STATUS, T.CSR_DBG_TRW_RPTR], "w1c %#x" % v)
    # STOP_AT / STOP_NOW / RELEASE / MATCHQ_CLEAR with an empty machine
    for v in (5, 0, 0x80000000):
        ds.wr(T.CSR_DBG_STOP_AT, v)
        await b.wr(T.CSR_DBG_STOP_AT, v)
        await b.cycles(10)
        bad += await D.compare_regs(b, ds, [T.CSR_DBG_STATUS, T.CSR_DBG_STOP_PIDX, T.CSR_DBG_STOP_AT], "at %#x" % v)
    for c in (T.DBG_CTRL_RELEASE, T.DBG_CTRL_STOP_NOW, T.DBG_CTRL_MATCHQ_CLEAR, T.DBG_CTRL_RELEASE | T.DBG_CTRL_STOP_NOW,
              T.DBG_CTRL_RELEASE, T.DBG_CTRL_HIST_CLEAR):
        ds.wr(T.CSR_DBG_CTRL, c)
        await b.wr(T.CSR_DBG_CTRL, c)
        await b.cycles(10)
        bad += await D.compare_regs(b, ds, [T.CSR_DBG_STATUS, T.CSR_DBG_STOP_PIDX, T.CSR_DBG_STOP_AT, T.CSR_DBG_CTRL,
                                            T.CSR_DBG_HIST_HEAD], "ctrl %#x" % c)
    bad += await D.compare_regs(b, ds, offs, "end", REG_MASK)
    # the core window reads the same block (CC §5.3): spot offsets
    for off in (T.CSR_DBG_ID, T.CSR_DBG_CFG, T.CSR_DBG_STOP_AT, T.CSR_DBG_TBP_A(1), T.CSR_DBG_TRW_MASK):
        a = await b.cw(off)
        e = ds.s.corewin_read(off)
        if a != e:
            bad.append("core window %#x RTL %#x sim %#x" % (off, a, e))
    ds.close()
    fail(dut, bad + b.errors, "dbg_regs")


@cocotb.test()
async def dbg_dmi(dut):
    """The DMI gate (DB §3.2, DB-2): DMI_EN = 0: reads 0, writes dropped (DMI_BLOCKED), the DM untouched;
    DMI_EN = 1: every access reaches the DM (a second DM model instance behind the RTL's DMI port, the
    simulator's own window on the first: same answers); core-window reads 0; DMI_RESP_ERR."""
    cfg, qd, _ = env_cfg()
    ds = D.DbgSim(cfg, qd)
    other = S.Sim(cfg, qd)
    L = D.lib()
    dm_b = L.tsr_sim_dm(other.sim)
    dm_a = L.tsr_sim_dm(ds.s.sim)
    assert dm_a and dm_b, "the simulator library has no DM model"
    b = D.Bench(dut, cfg, qd, seed=seed(), dm=dm_b)
    await b.start()
    r = random.Random(seed() * 7)
    bad = []
    regs = [0x04, 0x05, 0x10, 0x11, 0x12, 0x16, 0x20, 0x21, 0x27, 0x38, 0x39]
    for i in range(160):
        if i % 20 == 0:
            en = r.random() < 0.6
            v = T.DBG_CFG_DMI_EN if en else 0
            ds.wr(T.CSR_DBG_CFG, (T.CSR_DBG_CFG_RESET | v))
            await b.wr(T.CSR_DBG_CFG, (T.CSR_DBG_CFG_RESET | v))
        a7 = r.choice(regs)
        off = T.CSR_DMI(a7)
        n0 = len(b.dmi_log)
        if r.random() < 0.5:
            if a7 == 0x10:
                v = r.choice([0x1, 0x0, 0x1])            # dmactive (no ndmreset: it aborts the queues)
            elif a7 in (0x11, 0x12):
                v = r.getrandbits(32)                    # read-only
            else:
                v = r.getrandbits(32) & (0x00170000 | 0xFFF00000 if a7 == 0x38 else M32)
                if a7 == 0x38:
                    v &= ~((1 << 20) | (1 << 15))       # no SBA read on address / data
            ds.wr(off, v)
            await b.wr(off, v)
            if not en and len(b.dmi_log) != n0:
                bad.append("DMI write %#x reached the DM with DMI_EN = 0" % a7)
        else:
            ra, re_ = await b.rd(off), ds.rd(off)
            if ra != re_:
                bad.append("DMI read %#x (DMI_EN %d): RTL %#x sim %#x" % (a7, en, ra, re_))
            if not en and len(b.dmi_log) != n0:
                bad.append("DMI read %#x reached the DM with DMI_EN = 0" % a7)
        if i % 8 == 7:
            # DBG_RV_STATE.DMACTIVE is the DM's own dmcontrol.dmactive: a write that never reached it
            # (DMI_EN = 0) leaves it as it was
            bad += await D.compare_regs(b, ds, [T.CSR_DBG_STATUS, T.CSR_DBG_RV_STATE], "dmi %d" % i)
            ds.wr(T.CSR_DBG_STATUS, T.DBG_STATUS_DMI_BLOCKED)
            await b.wr(T.CSR_DBG_STATUS, T.DBG_STATUS_DMI_BLOCKED)
    # core window: DMI reads 0 (CC §5.3)
    a = await b.cw(T.CSR_DMI(0x11))
    if a != 0:
        bad.append("core-window DMI read %#x" % a)
    # a DM response code other than 0: DMI_RESP_ERR (RTL only: the DM model answers 0)
    ds_not = (await b.rd(T.CSR_DBG_STATUS)) & T.DBG_STATUS_DMI_RESP_ERR
    await b.wr(T.CSR_DBG_CFG, T.CSR_DBG_CFG_RESET | T.DBG_CFG_DMI_EN)
    b.dmi_resp = 2
    await b.rd(T.CSR_DMI(0x11))
    b.dmi_resp = 0
    st = await b.rd(T.CSR_DBG_STATUS)
    if ds_not or not st & T.DBG_STATUS_DMI_RESP_ERR:
        bad.append("DMI_RESP_ERR before %d after %#x" % (ds_not, st))
    other.close()
    ds.close()
    fail(dut, bad + b.errors, "dbg_dmi")


def tbp_pre(r, g, steps, cfg, kinds):
    """TBP register writes from the program's entries (a kind per TBP)."""
    pre = []
    pushes = [i for i, s in enumerate(steps) if s[0] == "push"]
    ticketed = [i for i, s in enumerate(steps) if s[0] == "push" or (s[0] == "qctrl" and s[1] & 8)]
    n_tk = len(ticketed)
    for i, kind in enumerate(kinds):
        flags = r.choice([0, T.DBG_TBP_CTL_AFTER, 0, T.DBG_TBP_CTL_COUNT_ONLY, T.DBG_TBP_CTL_HALT_RV])
        ctl = T.DBG_TBP_CTL_EN | (kind << 1) | flags
        if kind == T.DBG_TBP_CTL_KIND_PIDX:
            a = r.randrange(max(1, n_tk))
            bb = a + r.choice([1, 1, 2, 5])
        elif kind == T.DBG_TBP_CTL_KIND_OPCODE:
            w0 = steps[r.choice(pushes)][1][0]
            a = (w0 & 0xFF) | (0xFF << 8) | (r.choice([0, 0, 1]) * (((w0 >> 8) & 0xF) << 16 | 0xF << 20))
            bb = r.choice([0, 0, 1 << g.unit(steps[pushes[0]][1]), 0x3F])
        elif kind == T.DBG_TBP_CTL_KIND_SPM:
            a = r.randrange(cfg["spm_bytes"] // 64)
            bb = a + r.choice([1, 16, 1024, 8192])
            ctl |= r.choice([T.DBG_TBP_CTL_R, T.DBG_TBP_CTL_W, T.DBG_TBP_CTL_R | T.DBG_TBP_CTL_W])
        elif kind == T.DBG_TBP_CTL_KIND_DRAM:
            a = r.getrandbits(r.choice([16, 24, 28]))
            bb = a + r.choice([1, 1 << 16, 1 << 26])
            ctl |= r.choice([T.DBG_TBP_CTL_R, T.DBG_TBP_CTL_W, T.DBG_TBP_CTL_R | T.DBG_TBP_CTL_W])
            ctl |= r.choice([0xFFFF, 1, 2, 0x5]) << 16
        else:
            k = r.choice(ticketed) if ticketed else 0
            a = None                                     # resolved after the tickets are known: by unit / seq
            # TICKET: the n-th entry of a unit (tickets are {unit, seq} from 0)
            u = 5 if steps[k][0] == "qctrl" else g.unit(steps[k][1])
            nth = sum(1 for j in ticketed if j < k and (5 if steps[j][0] == "qctrl" else g.unit(steps[j][1])) == u)
            a = (u << 28) | nth
            bb = 0
        pre += [(T.CSR_DBG_TBP_A(i), a & M32), (T.CSR_DBG_TBP_B(i), bb & M32), (T.CSR_DBG_TBP_CTL(i), ctl)]
    return pre


@cocotb.test()
async def dbg_stops(dut):
    """Random push streams with stops (STOP_AT, QSTEP, RELEASE, STOP_NOW, tile breakpoints of every
    kind), compared with the simulator at every DRAINED stop and at the end (DB §5, invariants G / Q)."""
    cfg, qd, c = env_cfg()
    s0 = seed()
    fails = []
    for p in range(c.get("n_prog", 6)):
        r = random.Random(s0 * 101 + p)
        g = gen(cfg, qd, s0 * 1000 + p)
        steps = D.program(g, c.get("n_ops", 50))
        n_tk = sum(1 for s in steps if s[0] == "push" or (s[0] == "qctrl" and s[1] & 8))
        pre = [(T.CSR_DBG_TRW_BASE, 0x100), (T.CSR_DBG_TRW_LINES, 256),
               (T.CSR_DBG_CFG, T.CSR_DBG_CFG_RESET | T.DBG_CFG_TRW_EN)]
        mode = p % 4
        if mode == 0:            # manual stops and steps
            pre += [(T.CSR_DBG_STOP_AT, r.randrange(max(1, n_tk)))]
            acts = [r.choice([("qstep", 1), ("qstep", 3), ("qstep", 1), ("release",)]) for _ in range(8)]
        elif mode == 1:          # tile breakpoints
            kinds = r.sample([0, 1, 2, 3, 4], 4)
            pre += tbp_pre(r, g, steps, cfg, kinds)
            acts = [r.choice([("release",), ("qstep", 1), ("release",)]) for _ in range(30)]
        elif mode == 2:          # STOP_NOW at a settled point, then steps
            pre += [(T.CSR_DBG_STOP_AT, r.randrange(max(1, n_tk)))]
            acts = [("stop_now",), ("qstep", 2), ("release",)]
        else:                    # S and M together: STOP_AT, then steps over tile breakpoints
            kinds = r.sample([0, 1, 2, 3, 4], 3)
            pre += tbp_pre(r, g, steps, cfg, kinds)
            pre += [(T.CSR_DBG_STOP_AT, r.randrange(max(1, n_tk // 3)))]
            acts = [r.choice([("qstep", 1), ("qstep", 4), ("release",)]) for _ in range(30)]
        g._chk.close()
        kw = dict(lat=(1, 30) if p % 2 else (1, 6), ready_p=0.6 if p % 2 else 0.95, gap=(0, 6) if p % 3 else (0, 0))
        bad, *_ = await scenario(dut, cfg, qd, steps, "prog %d" % p, pre=pre, actions=acts, seed_=s0 * 31 + p,
                                 env_kw=kw, trw=(0x100, 256))
        if bad:
            fails.append((p, bad[:4]))
            for x in bad[:12]:
                dut._log.error(x)
    assert not fails, "%d programs differ: %s" % (len(fails), fails[:2])


@cocotb.test()
async def dbg_serial(dut):
    """Serial mode (DB §5.4): one entry in flight, completions in push order, the final state of the
    simulator (invariant S)."""
    cfg, qd, c = env_cfg()
    s0 = seed()
    fails = []
    for p in range(c.get("n_serial", 3)):
        g = gen(cfg, qd, s0 * 2000 + p)
        steps = D.program(g, c.get("n_ops", 50))
        g._chk.close()
        pre = [(T.CSR_DBG_CFG, T.CSR_DBG_CFG_RESET | T.DBG_CFG_SERIAL)]
        bad, b, ds = await scenario(dut, cfg, qd, steps, "serial %d" % p, pre=pre, seed_=s0 * 37 + p,
                                    env_kw=dict(lat=(1, 20), gap=(0, 3)))
        order = sorted((c_, pidx) for u in range(6) for (_, pidx, _, _, c_) in b.env.done[u])
        pl = [x[1] for x in order]
        if pl != sorted(pl):
            k = next(i for i in range(1, len(pl)) if pl[i] < pl[i - 1])
            bad.append("serial %d: completion order not push order at #%d: %s" % (p, k, pl[max(0, k - 3):k + 3]))
        starts = sorted((c_, u) for u in range(6) for (_, _, c_) in b.env.start[u])
        # one op in flight: every start after the previous completion
        dn = sorted(c_ for u in range(6) for (_, _, _, ran, c_) in b.env.done[u] if ran)
        if len(starts) <= len(dn):
            for i in range(1, len(starts)):
                if starts[i][0] < dn[i - 1]:
                    bad.append("serial %d: start at cycle %d before the completion at %d" % (p, starts[i][0], dn[i - 1]))
                    break
        if bad:
            fails.append((p, bad[:4]))
            for x in bad[:12]:
                dut._log.error(x)
    assert not fails, "%d programs differ: %s" % (len(fails), fails[:2])


@cocotb.test()
async def dbg_hist(dut):
    """The dispatch history (DB §5.6) over 3 x HIST_DEPTH entries, HIST_CLEAR, HIST_FREEZE_ERR, HIST_FREEZE."""
    cfg, qd, c = env_cfg()
    s0 = seed()
    bad = []
    # 1: a long stream without errors (light ops): wrap, PRESENT of the last 512
    r = random.Random(s0 * 5)
    steps = []
    for i in range(c.get("n_hist", 1300)):
        x = r.random()
        if x < 0.5:
            steps.append(("push", isa.enc_nop() if r.random() < 0.5 else isa.enc_fence(), ("none",), ("none",), 0xFF00))
        else:
            w = isa.enc_ld(r.randrange(cfg["num_ch"]), r.randrange(1 << 16), r.randrange(1 << 20), r.choice([0, 1]))
            steps.append(("push", w, ("none",), ("none",), 0xFF00 | (r.random() < 0.3)))
        if i == 700:
            steps.append(("wr", T.CSR_DBG_CTRL, T.DBG_CTRL_HIST_CLEAR))
        if i == 900:
            steps.append(("wr", T.CSR_DBG_CFG, T.CSR_DBG_CFG_RESET | T.DBG_CFG_HIST_FREEZE))
        if i == 1000:
            steps.append(("wr", T.CSR_DBG_CFG, T.CSR_DBG_CFG_RESET))
    b_, *_ = await scenario(dut, cfg, qd, steps, "hist long", env_kw=dict(lat=(1, 3), gap=(0, 0)), final_regs=True)
    bad += b_
    # 2: errors: the freeze 256 entries after the first one
    g = gen(cfg, qd, s0 * 3000)
    steps = []
    for i in range(400):
        if i == 30:
            steps.append(("push", isa.enc_ld(cfg["num_ch"] + 1, 0, 0, 1), ("none",), ("none",), 0xFF00))   # BAD channel
        else:
            steps.append(("push", isa.enc_nop(), ("none",), ("none",), 0xFF00))
    g._chk.close()
    b_, *_ = await scenario(dut, cfg, qd, steps, "hist freeze_err", env_kw=dict(lat=(1, 3), gap=(0, 0)))
    bad += b_
    fail(dut, bad, "dbg_hist")


@cocotb.test()
async def dbg_trw(dut):
    """The trace writer (DB §8.3): overwrite ring with wrap, TRW_STOP_FULL, no ring, a ring on another
    channel than TRW_CH, class mask, LAUNCH_END; a DMA write stream on the same write source; PERF
    TRW_RECORDS / TRW_DROPS."""
    cfg, qd, c = env_cfg()
    s0 = seed()
    bad = []
    cases = [
        ("wrap", 0x200, 16, T.DBG_CFG_TRW_EN, 0x7FF, True),
        ("mask", 0x400, 64, T.DBG_CFG_TRW_EN, 0x305, True),
        ("full", 0x600, 16, T.DBG_CFG_TRW_EN | T.DBG_CFG_TRW_STOP_FULL, 0x7FF, False),
        ("noring", 0x0, 0, T.DBG_CFG_TRW_EN, 0x7FF, False),
        # a ring on a channel other than TRW_CH is no ring: every record dropped (DB §8.3), in both models
        ("wrongch", (1 << 28) | 0x200, 16, T.DBG_CFG_TRW_EN, 0x7FF, False),
    ]
    for k, (name, base, lines, cfgb, mask, full) in enumerate(cases):
        g = gen(cfg, qd, s0 * 4000 + k)
        steps = D.program(g, c.get("n_trw", 60), launch_end=True)
        g._chk.close()
        ds = D.DbgSim(cfg, qd)
        b = D.Bench(dut, cfg, qd, seed=s0 * 41 + k, env_kw=dict(lat=(1, 25), gap=(0, 2)))
        await b.start()
        await b.launch()                          # ACTIVE for the PERF event counters (EX §6)
        if name == "wrap":       # a DMA write stream through the mux on the same channel
            rr = random.Random(s0)
            for i in range(40):
                a = 0x40000 + 64 * i + rr.randrange(32)
                n = rr.randint(1, 64 - (a % 64))
                b.dma_bursts.append([a, n, [rr.getrandbits(512) for _ in range(n)]])
        pre = [(T.CSR_DBG_TRW_BASE, base), (T.CSR_DBG_TRW_MASK, mask)]
        if lines:
            pre.append((T.CSR_DBG_TRW_LINES, lines))
        pre.append((T.CSR_DBG_CFG, T.CSR_DBG_CFG_RESET | cfgb))
        b_, *_ = await scenario(dut, cfg, qd, steps, "trw " + name, pre=pre, ds=ds, b=b, hist=False,
                                trw=None)
        bad += b_
        bad += await compare_trw(b, ds, base, max(lines, 16), "trw " + name, full=full or lines == 0)
        for slot in (T.PERF_TRW_RECORDS, T.PERF_TRW_DROPS):
            a = await b.rd(T.CSR_PERF_LO(slot))
            e = ds.rd(T.CSR_PERF_LO(slot))
            if a != e:
                bad.append("trw %s: PERF %d RTL %d sim %d" % (name, slot, a, e))
        if name == "wrap":
            await b.cycles(200)
            if len(b.dma_sent) != 40:
                bad.append("trw wrap: %d of 40 DMA bursts sent" % len(b.dma_sent))
            for (a, n, data) in b.dma_sent:
                for i in range(n):
                    if b.mem.get(a + i) != data[i]:
                        bad.append("trw wrap: DMA beat %#x lost or wrong" % (a + i))
                        break
            if b.dma_rsp != 40:
                bad.append("trw wrap: %d DMA write responses, 40 expected" % b.dma_rsp)
            dut._log.info("trw wrap: %d DMA bursts, %d write bursts in all, %d beats before their request",
                          len(b.dma_sent), len(b.bursts), b.data_first)
        ds.close()
        bad += b.errors
    fail(dut, bad, "dbg_trw")


@cocotb.test()
async def dbg_rec(dut):
    """Completion records to the RISC-V (DB §8.2) through the clk_rv window: REC flag, REC_CFG.ALL,
    markers, REC_COUNT / POP, irq_fast[4], the 64-record overflow."""
    cfg, qd, c = env_cfg()
    s0 = seed()
    bad = []
    for k, (n, allr) in enumerate([(30, 0), (40, 1), (90, 1)]):
        g = gen(cfg, qd, s0 * 5000 + k)
        steps = D.program(g, n, rec_p=0.4)
        g._chk.close()
        ds = D.DbgSim(cfg, qd)
        b = D.Bench(dut, cfg, qd, seed=s0 * 43 + k)
        await b.start()
        ds.wr(T.CSR_CTRL, T.CTRL_RV_RUN)
        await b.wr(T.CSR_CTRL, T.CTRL_RV_RUN)
        ds.s.advance(20000)                      # the simulator's hart leaves reset (REC restarts empty)
        await b.cycles(40)
        ds.cw_wr(T.CW_REC_CFG, allr | 2)
        await b.db_wr(T.CW_REC_CFG, allr | 2)
        await b.cycles(20)
        b_, *_ = await scenario(dut, cfg, qd, steps, "rec %d" % k, ds=ds, b=b, hist=False)
        bad += b_
        await b.cycles(100)
        ca, ce = await b.db_rd(T.CW_REC_COUNT), ds.cw_rd(T.CW_REC_COUNT)
        irq = int(dut.db_irq4.value)
        if ca != ce or irq != (ce >= 32):
            bad.append("rec %d: REC_COUNT RTL %d sim %d, irq4 %d" % (k, ca, ce, irq))
        st_a, st_e = await b.rd(T.CSR_DBG_STATUS), ds.rd(T.CSR_DBG_STATUS)
        if (st_a ^ st_e) & T.DBG_STATUS_REC_DROP:
            bad.append("rec %d: REC_DROP RTL %d sim %d" % (k, bool(st_a & T.DBG_STATUS_REC_DROP),
                                                          bool(st_e & T.DBG_STATUS_REC_DROP)))
        ra, re_ = await b.recs(), ds.recs()
        va, ve = D.rec_view(ra), D.rec_view(re_)
        if len(ra) != len(re_):
            bad.append("rec %d: %d records RTL, %d sim" % (k, len(ra), len(re_)))
        elif ce < 64 and va != ve:
            for u in range(8):
                if va["op"][u] != ve["op"][u]:
                    bad.append("rec %d: unit %d RTL %s sim %s" % (k, u, va["op"][u][:4], ve["op"][u][:4]))
            if va["mark"] != ve["mark"]:
                bad.append("rec %d: markers RTL %s sim %s" % (k, va["mark"][:4], ve["mark"][:4]))
        # DBG_FLAGS in the window
        fa, fe = await b.db_rd(T.CW_DBG_FLAGS), ds.cw_rd(T.CW_DBG_FLAGS)
        if fa != fe:
            bad.append("rec %d: DBG_FLAGS RTL %#x sim %#x" % (k, fa, fe))
        ds.close()
        bad += b.errors
    fail(dut, bad, "dbg_rec")


@cocotb.test()
async def dbg_halt(dut):
    """DB's halt requests (DB §3.3): STOP_HALT_RV and a TBP's HALT_RV with RV_HALT_EN, the cancel,
    HALT_ON_BOOT, dbg_stop and DBG_FLAGS; one request per activation, none while RV_HALTED."""
    cfg, qd, _ = env_cfg()
    ds = D.DbgSim(cfg, qd)
    b = D.Bench(dut, cfg, qd, seed=seed())
    await b.start()
    bad = []

    async def both_wr(off, v):
        ds.wr(off, v)
        await b.wr(off, v)
        await b.cycles(8)

    async def cmp(what):
        bad.extend(await D.compare_regs(b, ds, [T.CSR_DBG_STATUS, T.CSR_DBG_STOP_PIDX], what))

    base = T.CSR_DBG_CFG_RESET
    await both_wr(T.CSR_DBG_CFG, base | T.DBG_CFG_STOP_HALT_RV)            # RV_HALT_EN = 0: no request
    await both_wr(T.CSR_DBG_STOP_AT, 3)
    await cmp("no halt enable")
    n0 = b.halt_reqs
    await both_wr(T.CSR_DBG_CTRL, T.DBG_CTRL_RELEASE)
    await both_wr(T.CSR_DBG_CFG, base | T.DBG_CFG_STOP_HALT_RV | T.DBG_CFG_RV_HALT_EN)
    await both_wr(T.CSR_DBG_STOP_AT, 4)
    await cmp("halt request")
    if b.halt_reqs != n0 + 1 or not int(dut.dbg_stop.value):
        bad.append("STOP_AT with STOP_HALT_RV: %d requests, dbg_stop %d" % (b.halt_reqs - n0, int(dut.dbg_stop.value)))
    await both_wr(T.CSR_DBG_CTRL, T.DBG_CTRL_RV_HALT_CANCEL)
    await cmp("cancel")
    await both_wr(T.CSR_DBG_CTRL, T.DBG_CTRL_RELEASE)
    await cmp("release")
    if int(dut.dbg_stop.value):
        bad.append("dbg_stop after RELEASE")
    # one-shot: while RV_HALTED (RV_RUN, the hart halted) no request
    await b.wr(T.CSR_CTRL, T.CTRL_RV_RUN)
    await b.cycles(30)
    b.rv["halted"] = 1
    await b.cycles(10)
    n1 = b.halt_reqs
    await b.wr(T.CSR_DBG_STOP_AT, 9)
    await b.cycles(10)
    if b.halt_reqs != n1:
        bad.append("a halt request while RV_HALTED")
    b.rv["halted"] = 0
    await b.wr(T.CSR_DBG_CTRL, T.DBG_CTRL_RELEASE)
    await b.wr(T.CSR_DBG_CTRL, T.DBG_CTRL_STOP_NOW)
    await b.cycles(10)
    if b.halt_reqs != n1 + 1:
        bad.append("STOP_NOW after the hart resumed: %d requests" % (b.halt_reqs - n1))
    # HALT_ON_BOOT level, DBG_FLAGS (window)
    await b.wr(T.CSR_DBG_CFG, base | T.DBG_CFG_HALT_ON_BOOT | T.DBG_CFG_RV_HALT_EN | T.DBG_CFG_SERIAL |
               T.DBG_CFG_TRW_EN | T.DBG_CFG_DMI_EN)
    await b.cycles(20)
    f = await b.db_rd(T.CW_DBG_FLAGS)
    if not int(dut.dbg_halt_on_boot.value) or f != 0x1F:
        bad.append("HALT_ON_BOOT %d, DBG_FLAGS %#x (expected 0x1f)" % (int(dut.dbg_halt_on_boot.value), f))
    ds.close()
    fail(dut, bad + b.errors, "dbg_halt")


async def abort_case(dut, cfg, qd, steps, stop_at, how, tag, seed_):
    """One push stream with STOP_AT = stop_at run to the DRAINED stop, then a QABORT or a SOFT_RESET
    (how), on the simulator and on the RTL; the completion counts, error records and the completion
    streams (codes, ran flags) are compared. Returns the mismatches."""
    bad = []
    ds = D.DbgSim(cfg, qd, launch=False)
    b = D.Bench(dut, cfg, qd, seed=seed_)
    await b.start()
    n_tk = sum(1 for s in steps if s[0] == "push")
    ds.wr(T.CSR_DBG_STOP_AT, stop_at)
    await b.wr(T.CSR_DBG_STOP_AT, stop_at)
    st = dict(k=0, tickets={})
    ok_s = ds.run(steps)
    ok_r = await b.run(steps, st)
    if ds.k != st["k"] or ok_s != ok_r:
        bad.append("%s: blocked at %d / %d" % (tag, ds.k, st["k"]))
    else:
        ds.wait_drained()
        await b.wait_drained()
        reg = T.CTRL_QABORT if how == "abort" else T.CTRL_SOFT_RESET
        ds.wr(T.CSR_CTRL, reg)
        await b.wr(T.CSR_CTRL, reg)
        ds.s.advance(200000)
        await b.cycles(3000)
        bad += await D.compare_regs(b, ds, [T.CSR_DBG_STATUS, T.CSR_DBG_STOP_AT, T.CSR_DBG_MATCHQ, T.CSR_DBG_CFG] +
                                    [T.CSR_Q_DONE(u) for u in range(6)] + [T.CSR_Q_ERR_CODE(u) for u in range(6)] +
                                    [T.CSR_Q_ERR_SEQ(u) for u in range(6)], tag)
        bad += D.compare_done(b, ds, tag)
        dut._log.info("%s: %d of %d entries pushed before the abort", tag, st["k"], n_tk)
    ds.close()
    return bad + b.errors


@cocotb.test()
async def dbg_abort(dut):
    """The hard hold (DB §5.2, DB-9) yields to a QABORT (held entries complete ABORTED, blocks starts
    only until the QCLEAR) and SOFT_RESET clears the stop state (EX §8.5 step 1), against the simulator.
    Random streams, then directed ones whose held head is a NOP, a FENCE or an error-coded entry: those
    decide before the hold (CC §9.3) and complete with their own code unless the abort has been applied
    first, so SOFT_RESET (SR_BUSY and QABORT rise together, the abort applies two cycles later) must
    keep them held until it does."""
    cfg, qd, c = env_cfg()
    s0 = seed()
    bad = []
    for k in range(2):
        g = gen(cfg, qd, s0 * 6000 + k)
        steps = D.program(g, 30, qclears=False)
        g._chk.close()
        bad += await abort_case(dut, cfg, qd, steps, 10 + k * 5, "abort" if k == 0 else "reset", "abort %d" % k,
                                s0 * 47 + k)
    heads = dict(nop=isa.enc_nop(), fence=isa.enc_fence(), illegal=[0x70] + [0] * 7, nulld=isa.enc_ld(0, 0, 0, 0))
    n = 0
    for hname, hw in heads.items():
        for how in ("abort", "reset"):
            n += 1
            g = gen(cfg, qd, s0 * 7000 + n)
            D.program(g, 6, marks=False, qclears=False)
            pre = len([x for x in g.steps if x[0] == "push"])
            g.push(hw, push_val=0xFF00)
            D.program(g, pre + 4, marks=False, qclears=False)
            steps = g.steps
            g._chk.close()
            bad += await abort_case(dut, cfg, qd, steps, pre, how, "abort head=%s %s" % (hname, how), s0 * 53 + n)
    fail(dut, bad, "dbg_abort")


@cocotb.test()
async def dbg_misc(dut):
    """RTL-only checks: ERR_STOP, HUNG_STOP and the history freeze, QSTEP under the debug-halt hold
    (dbg_step), RV_SAMPLE / RVWP_APPLY timeouts without the clk_rv taps, MATCHQ_CLEAR."""
    cfg, qd, _ = env_cfg()
    b = D.Bench(dut, cfg, qd, seed=seed(), env_kw=dict(lat=(5, 10), gap=(0, 0)))
    await b.start()
    env = b.env
    bad = []
    base = T.CSR_DBG_CFG_RESET
    # ERR_STOP: a poisoned entry stops the tile at the next push index with cause ERR
    await b.wr(T.CSR_DBG_CFG, base | T.DBG_CFG_ERR_STOP)
    for i in range(4):
        await env.push(isa.enc_nop(), pc=D.HOOK_PC)
    await env.push(isa.enc_ld(cfg["num_ch"] + 2, 0, 0, 1), pc=D.HOOK_PC)      # BAD channel: poisoned
    await b.cycles(200)
    st = await b.rd(T.CSR_DBG_STATUS)
    t = await b.rd(T.CSR_DBG_STOP_PIDX)
    if not (st & T.DBG_STATUS_STOP_ACTIVE) or not (st >> 4) & T.DBG_CAUSE_ERR or t != 5:
        bad.append("ERR_STOP: status %#x T %d (expected active, cause ERR, T 5)" % (st, t))
    await b.wr(T.CSR_DBG_CTRL, T.DBG_CTRL_RELEASE)
    await b.wr(T.CSR_CTRL, T.CTRL_QCLEAR)
    await b.cycles(50)
    # HUNG_STOP + the history freeze at UNIT_HUNG
    await b.wr(T.CSR_DBG_CFG, base | T.DBG_CFG_HUNG_STOP)
    b.unit_hung = 1
    await b.cycles(20)
    st = await b.rd(T.CSR_DBG_STATUS)
    if not (st & T.DBG_STATUS_STOP_ACTIVE) or not (st >> 4) & T.DBG_CAUSE_HUNG or not st & T.DBG_STATUS_HIST_FROZEN:
        bad.append("HUNG_STOP: status %#x" % st)
    b.unit_hung = 0
    await b.wr(T.CSR_DBG_CTRL, T.DBG_CTRL_RELEASE | T.DBG_CTRL_HIST_CLEAR)
    await b.cycles(10)
    st = await b.rd(T.CSR_DBG_STATUS)
    if st & (T.DBG_STATUS_STOP_ACTIVE | T.DBG_STATUS_HIST_FROZEN):
        bad.append("after RELEASE | HIST_CLEAR: status %#x" % st)
    # QSTEP without a stop under the debug-halt hold: exactly the oldest head runs
    await b.wr(T.CSR_DBG_CFG, base)
    await b.wr(T.CSR_CTRL, T.CTRL_RV_RUN | T.CTRL_HOLD_ON_HALT)
    await b.cycles(30)
    b.rv["halted"] = 1
    await b.cycles(10)
    d0 = list(env.D)
    for i in range(3):
        await env.push(isa.enc_ld(0, i, i, 1), pc=D.HOOK_PC)
    await b.cycles(300)
    if env.D != d0:
        bad.append("HOLD_ON_HALT: entries ran while halted")
    await b.wr(T.CSR_DBG_CTRL, T.DBG_CTRL_QSTEP)
    await b.cycles(300)
    if env.D[0] != (d0[0] + 1) & D.SEQ_MASK:
        bad.append("QSTEP under the debug-halt hold: DMA D %d -> %d (one expected)" % (d0[0], env.D[0]))
    x = await b.rd(T.CSR_DBG_STEP_PIDX)
    st = await b.rd(T.CSR_DBG_STATUS)
    if st & T.DBG_STATUS_STEP_BUSY:
        bad.append("QSTEP under the hold: STEP_BUSY after the step")
    b.rv["halted"] = 0
    await b.wr(T.CSR_CTRL, T.CTRL_RV_RUN)                 # HOLD_ON_HALT off: the others run
    await env.wait_idle()
    dut._log.info("QSTEP under the hold stepped push index %d", x)
    # RV_SAMPLE / RVWP_APPLY without the clk_rv taps: timeouts after 4,096 cycles (DB §6.2, §3.7)
    await b.wr(T.CSR_DBG_CTRL, T.DBG_CTRL_RV_SAMPLE | T.DBG_CTRL_RVWP_APPLY)
    st = await b.rd(T.CSR_DBG_STATUS)
    if not st & T.DBG_STATUS_RVWP_BUSY:
        bad.append("RVWP_APPLY: RVWP_BUSY not set while in flight")
    await b.cycles(4300)
    st = await b.rd(T.CSR_DBG_STATUS)
    rs = await b.rd(T.CSR_DBG_RV_STATE)
    want = T.DBG_STATUS_RV_SAMPLE_TIMEOUT | T.DBG_STATUS_RVWP_TIMEOUT
    if (st & want) != want or st & T.DBG_STATUS_RVWP_BUSY or rs & T.DBG_RV_STATE_SAMPLE_VALID:
        bad.append("handshake timeouts: status %#x rv_state %#x" % (st, rs))
    # MATCHQ_CLEAR: a TBP stop on every NOP, cleared
    await b.wr(T.CSR_DBG_TBP_A(0), isa.OP_NOP | 0xFF00)
    await b.wr(T.CSR_DBG_TBP_B(0), 0)
    await b.wr(T.CSR_DBG_TBP_CTL(0), T.DBG_TBP_CTL_EN | (T.DBG_TBP_CTL_KIND_OPCODE << 1))
    for i in range(3):
        await env.push(isa.enc_nop(), pc=D.HOOK_PC)
    await b.cycles(200)
    mq = await b.rd(T.CSR_DBG_MATCHQ)
    await b.wr(T.CSR_DBG_TBP_CTL(0), 0)
    await b.wr(T.CSR_DBG_CTRL, T.DBG_CTRL_MATCHQ_CLEAR)
    await b.cycles(10)
    st = await b.rd(T.CSR_DBG_STATUS)
    if mq != 3 or st & T.DBG_STATUS_STOP_ACTIVE:
        bad.append("MATCHQ: %d pending (3 expected), status after the clear %#x" % (mq, st))
    await env.wait_idle()
    fail(dut, bad + b.errors, "dbg_misc")


@cocotb.test()
async def dbg_rvtaps(dut):
    """RTL-only (HAS_RV_TAPS = 1, synthetic Ibex taps in clk_rv): RVWP_APPLY, RV-watchpoint hits (kind,
    overlap edges incl. the last byte, HITS, WPADDR, RVWP_HIT, a halt request with HALT), RV_SAMPLE
    (DBG_RV_*, SAMPLE_VALID, SAMPLE_SEQ), RV_DFAULT (DB §3.7, §6.2)."""
    if not int(os.environ.get("TSR_DBG_TAPS", "0")):
        return
    cfg, qd, _ = env_cfg()
    b = D.Bench(dut, cfg, qd, seed=seed())
    await b.start()
    bad = []
    base = T.CSR_DBG_CFG_RESET

    async def access(addr, we, be):
        await RisingEdge(dut.clk_rv)
        dut.ib_data_req.value = 1
        dut.ib_data_gnt.value = 1
        dut.ib_data_addr.value = addr & ~3
        dut.ib_data_we.value = we
        dut.ib_data_be.value = be
        await RisingEdge(dut.clk_rv)
        dut.ib_data_req.value = 0
        dut.ib_data_gnt.value = 0
        await b.cycles(40)

    await b.wr(T.CSR_DBG_CFG, base | T.DBG_CFG_RV_HALT_EN)
    await b.wr(T.CSR_DBG_RVWP_CTL(0), T.DBG_RVWP_CTL_EN | T.DBG_RVWP_CTL_STORE | T.DBG_RVWP_CTL_HALT)
    await b.wr(T.CSR_DBG_RVWP_LO(0), 0x100)
    await b.wr(T.CSR_DBG_RVWP_HI(0), 0x108)
    await b.wr(T.CSR_DBG_RVWP_CTL(1), T.DBG_RVWP_CTL_EN | T.DBG_RVWP_CTL_LOAD)
    await b.wr(T.CSR_DBG_RVWP_LO(1), 0x20000000)
    await b.wr(T.CSR_DBG_RVWP_HI(1), 0x20000004)
    await b.wr(T.CSR_DBG_CTRL, T.DBG_CTRL_RVWP_APPLY)
    await b.cycles(60)
    st = await b.rd(T.CSR_DBG_STATUS)
    if st & (T.DBG_STATUS_RVWP_BUSY | T.DBG_STATUS_RVWP_TIMEOUT):
        bad.append("RVWP_APPLY: status %#x" % st)
    n0 = b.halt_reqs
    cases = [  # (addr, we, be, hit mask)
        (0x0FC, 1, 0xF, 0), (0x0FC, 1, 0xC, 0), (0x100, 1, 0x8, 1), (0x104, 1, 0x3, 1), (0x108, 1, 0x1, 0),
        (0x104, 0, 0xF, 0), (0x20000000, 0, 0xF, 2), (0x20000004, 0, 0x1, 0), (0x20000000, 1, 0xF, 0),
        (0x1FFFFFFC, 0, 0x8, 0), (0x1FFFFFFC, 0, 0xF, 0)]
    hits = [0, 0]
    for addr, we, be, mask in cases:
        await access(addr, we, be)
        for j in range(2):
            hits[j] += (mask >> j) & 1
        h = [await b.rd(T.CSR_DBG_RVWP_HITS(j)) for j in range(2)]
        if h != hits:
            bad.append("access %#x we %d be %#x: HITS %s, expected %s" % (addr, we, be, h, hits))
        if mask:
            lo = min(i for i in range(4) if (be >> i) & 1)
            wa = await b.rd(T.CSR_DBG_RV_WPADDR)
            if wa != (addr & ~3) + lo:
                bad.append("WPADDR %#x, expected %#x" % (wa, (addr & ~3) + lo))
    st = await b.rd(T.CSR_DBG_STATUS)
    if ((st >> 13) & 3) != 3:
        bad.append("RVWP_HIT %d" % ((st >> 13) & 3))
    if b.halt_reqs != n0 + hits[0]:
        bad.append("%d halt requests for %d HALT hits" % (b.halt_reqs - n0, hits[0]))
    await b.wr(T.CSR_DBG_RVWP_HITS(0), 0)
    if await b.rd(T.CSR_DBG_RVWP_HITS(0)) != 0:
        bad.append("HITS not cleared by a write")
    # RV_SAMPLE
    vals = [0x1000, 0x1004, 0x20000040, 0x80, 0xDEAD]
    dut.ib_cd.value = sum(v << (32 * i) for i, v in enumerate(vals))
    dut.ib_sleep.value = 1
    await b.wr(T.CSR_DBG_CTRL, T.DBG_CTRL_RV_SAMPLE)
    await b.cycles(60)
    got = [await b.rd(o) for o in (T.CSR_DBG_RV_PC, T.CSR_DBG_RV_NPC, T.CSR_DBG_RV_DADDR, T.CSR_DBG_RV_EPC,
                                   T.CSR_DBG_RV_EADDR)]
    rs = await b.rd(T.CSR_DBG_RV_STATE)
    if got != vals or not rs & T.DBG_RV_STATE_SAMPLE_VALID or ((rs >> 8) & 0xFF) != 1:
        bad.append("RV_SAMPLE %s rv_state %#x" % ([hex(x) for x in got], rs))
    # double fault
    await RisingEdge(dut.clk_rv)
    dut.ib_dfault.value = 1
    await RisingEdge(dut.clk_rv)
    dut.ib_dfault.value = 0
    await b.cycles(20)
    st = await b.rd(T.CSR_DBG_STATUS)
    if not st & T.DBG_STATUS_RV_DFAULT:
        bad.append("RV_DFAULT not set: %#x" % st)
    fail(dut, bad + b.errors, "dbg_rvtaps")


@cocotb.test()
async def dbg_rvwp_dense(dut):
    """DB §3.7: sustained legal one-access/RV-cycle hits keep both counts and a single HALT."""
    if not int(os.environ.get("TSR_DBG_TAPS", "0")):
        return
    cfg, qd, _ = env_cfg()
    b = D.Bench(dut, cfg, qd, seed=seed())
    core_ps, rv_ps, phase_ps = map(int, os.environ.get("TSR_DBG_CLOCKS", "3334,6000,0").split(","))
    await b.start(core_ps=core_ps, rv_ps=rv_ps, rv_phase_ps=phase_ps)
    base, n = 0x100000, 4096
    await b.wr(T.CSR_DBG_CFG, T.CSR_DBG_CFG_RESET | T.DBG_CFG_RV_HALT_EN)
    # WP0 counts the whole burst. WP1 counts and halts only its seventh access: losing that
    # event must fail even if most of a long burst still makes it through the crossing.
    for j, lo, hi, halt in ((0, base, base + 4 * n, 0), (1, base + 24, base + 28, 1)):
        await b.wr(T.CSR_DBG_RVWP_LO(j), lo)
        await b.wr(T.CSR_DBG_RVWP_HI(j), hi)
        await b.wr(T.CSR_DBG_RVWP_CTL(j), T.DBG_RVWP_CTL_EN | T.DBG_RVWP_CTL_STORE |
                   (T.DBG_RVWP_CTL_HALT if halt else 0))
    await b.wr(T.CSR_DBG_CTRL, T.DBG_CTRL_RVWP_APPLY)
    await b.cycles(100)
    assert not await b.rd(T.CSR_DBG_STATUS) & (T.DBG_STATUS_RVWP_BUSY | T.DBG_STATUS_RVWP_TIMEOUT)
    n0 = b.halt_reqs
    for i in range(n):
        await FallingEdge(dut.clk_rv)
        dut.ib_data_req.value = 1
        dut.ib_data_gnt.value = 1
        dut.ib_data_we.value = 1
        dut.ib_data_be.value = 15
        dut.ib_data_addr.value = base + 4 * i
    await FallingEdge(dut.clk_rv)
    dut.ib_data_req.value = 0
    dut.ib_data_gnt.value = 0
    await b.cycles(100)
    hits = [await b.rd(T.CSR_DBG_RVWP_HITS(j)) for j in range(2)]
    addr = await b.rd(T.CSR_DBG_RV_WPADDR)
    dut._log.info("dense RVWP clocks=%s: hits=%s halt_requests=%d last_addr=%#x",
                  (core_ps, rv_ps, phase_ps), hits, b.halt_reqs - n0, addr)
    assert hits == [n, 1], "lost dense watchpoint events: %s" % hits
    assert b.halt_reqs == n0 + 1, "lost the sole HALT-bearing event"
    assert addr == base + 4 * (n - 1), "last watchpoint address lost"
    assert not b.errors, b.errors


# =================================================================================================
# pytest launchers
# =================================================================================================
def _params(cfg, qd, extra=None):
    p = dict(NUM_CH=cfg["num_ch"], T_MAX=cfg["t_max"], K_MAX=cfg["k_max"], SPM_BYTES=cfg["spm_bytes"],
             CH_BYTES="64'd%d" % cfg["ch_bytes"], ADDR_BEAT_W=cfg["addr_beat_w"], CAPS="32'd%d" % cfg["caps"],
             QD_DMA=qd[0], QD_QACT=qd[1], QD_MVM=qd[2], QD_VU=qd[3], QD_ATT=qd[4], QD_SYNC=qd[5])
    p.update(extra or {})
    return p


def _run(name, cfg, qd, extra_cfg=None, extra_params=None, tests=None, extra_env=None, build_args=()):
    """Builds the wrapper and runs the cocotb tests (all, `tests`, or $TSR_DBG_TESTS: a comma list)."""
    if os.environ.get("TSR_DBG_TESTS") and not os.environ.get("TSR_DBG_MUT_TEST"):
        sel = os.environ["TSR_DBG_TESTS"].split(",")
        tests = [t for t in (tests or sel) if t in sel]
    c = dict(num_ch=cfg["num_ch"], t_max=cfg["t_max"], qd=list(qd))
    c.update(extra_cfg or {})
    env = {"TSR_DBG_CFG": json.dumps(c), "TSR_DBG_TAPS": str((extra_params or {}).get("HAS_RV_TAPS", 0))}
    env.update(extra_env or {})
    run("tsr_dbg_tb", __name__, parameters=_params(cfg, qd, extra_params), sources=sources(),
        name="dbg_" + name, extra_env=env, testcase=tests, build_args=build_args)


U200 = S.config(num_ch=4, t_max=8)


def test_dbg_lint():
    """Verilator -Wall lint of the DB block (every module as top, edge parameters) and the wrapper."""
    from common.runner import rtl_args
    srcs = [str(REPO / f) for f in BASE_SRC] + [str(REPO / "rtl/core" / n) for n in DBG_SRC]
    runs = [("tsr_dbg", []), ("tsr_dbg", ["-GNUM_CH=1", "-GHIST_DEPTH=0", "-GHAS_TRW=0", "-GNTBP=0", "-GNRVWP=0",
                                          "-GHAS_REC=0"]),
            ("tsr_dbg", ["-GNUM_CH=16", "-GADDR_BEAT_W=36", "-GQD_ATT=4", "-GQD_VU=31", "-GNTBP=2", "-GNRVWP=1",
                         "-GHIST_DEPTH=64"]),
            ("tsr_dbg_stop", []), ("tsr_dbg_stop", ["-GNTBP=0", "-GNUM_CH=1"]), ("tsr_dbg_shq", []),
            ("tsr_dbg_shq", ["-GQD_DMA=31", "-GQD_VU=0", "-GQD_MVM=1"]), ("tsr_dbg_hist", []),
            ("tsr_dbg_hist", ["-GDEPTH=2"]), ("tsr_dbg_trw", []), ("tsr_dbg_trw", ["-GADDR_BEAT_W=22",
                                                                                   "-GCH_BYTES=64'd268435456"]),
            ("tsr_dbg_rv", []), ("tsr_dbg_rv", ["-GHAS_RV_TAPS=1"]), ("tsr_dbg_rv", ["-GHAS_RV_TAPS=1", "-GCTRL_SYNC=1",
                                                                                   "-GREC_DEPTH=4"]),
            ("tsr_dbg_wrmux", []), ("tsr_dbg_wrmux", ["-GADDR_BEAT_W=5"]),
            ("tsr_dbg_tb", [str(HERE / "sv/tsr_dbg_tb.sv")])]
    bad = []
    for top, extra in runs:
        cmd = ["verilator", "--lint-only", "-Wall", "--top-module", top] + rtl_args() + srcs + extra
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            bad.append("%s %s:\n%s" % (top, extra, r.stderr[-2000:]))
    assert not bad, "\n".join(bad)


def test_dbg_u200():
    """U200 M6 queues, 4 channels, every test."""
    _run("u200", U200, S.U200_QD, dict(n_prog=12, n_ops=60))


def test_dbg_rvtaps():
    """tsr_dbg_rv with the Ibex taps (HAS_RV_TAPS = 1): RV watchpoints, RV_SAMPLE, double fault."""
    _run("rvtaps", U200, S.U200_QD, dict(), extra_params=dict(HAS_RV_TAPS=1), tests=["dbg_rvtaps"])


@pytest.mark.parametrize("clocks", [(3334, 6000, 0), (3334, 6000, 1777), (3620, 6000, 3599),
                                    (3620, 8000, 7999), (3620, 4000, 3999),
                                    (6000, 6000, 0), (6000, 6000, 1), (6000, 6000, 5999)])
def test_dbg_rvwp_dense(clocks):
    _run("rvtaps", U200, S.U200_QD, extra_params=dict(HAS_RV_TAPS=1), tests=["dbg_rvwp_dense"],
         extra_env={"TSR_DBG_CLOCKS": ",".join(map(str, clocks))})


@pytest.mark.parametrize("clocks", [(3334, 6000, 1777), (6000, 6000, 0), (6000, 6000, 5999)])
def test_dbg_rvwp_dense_jitter(clocks):
    _run("rvtaps", U200, S.U200_QD, extra_params=dict(HAS_RV_TAPS=1), tests=["dbg_rvwp_dense"],
         extra_env={"TSR_DBG_CLOCKS": ",".join(map(str, clocks))}, build_args=["+define+TSR_CDC_JITTER"])


def test_dbg_small():
    """1 channel, shallow queues (credits run out early: many blocked points), stops and the trace writer."""
    cfg = S.config(num_ch=1, t_max=2)
    _run("small", cfg, (2, 1, 2, 16, 0, 1), dict(n_prog=4, n_ops=40, n_serial=2, n_trw=40, n_hist=600),
         tests=["dbg_stops", "dbg_serial", "dbg_trw", "dbg_hist"])


# (id, file, original text, mutated text, test that must fail, DB §14 item 7)
MUTANTS = [
    ("hold_ge_off_by_one", "tsr_dbg_stop.sv",
     "                     ((sv_q && ge(h, s_q)) || (ev_stop_at && ge(h, ev_stop_at_val)) ||",
     "                     ((sv_q && ge(h, s_q + 32'd1)) || (ev_stop_at && ge(h, ev_stop_at_val + 32'd1)) ||",
     "dbg_stops", "start a head with pidx = T"),
    ("release_pops_all", "tsr_dbg_stop.sv",
     "      pop_head = active && m_any && !t_is_s;",
     "      pop_head = active && m_any;",
     "dbg_stops", "pop the wrong M entry on RELEASE"),
    ("drained_ignores_front", "tsr_dbg_stop.sv",
     "      if (out_v[u] && !ge(out_pidx[u], t_cur)) ok = 1'b0;",
     "      if (1'b0 && out_v[u] && !ge(out_pidx[u], t_cur)) ok = 1'b0;",
     "dbg_stops", "DRAINED ignoring a running op"),
    ("serial_lets_two", "tsr_dbg_stop.sv",
     "                      (cfg_serial && mo_v_q && h != mo_q));",
     "                      (cfg_serial && mo_v_q && h != mo_q && h != mo_q + 32'd1));",
     "dbg_serial", "serial mode allowing an overlap"),
    ("hist_slot_off", "tsr_dbg_hist.sv",
     "    wire [AW-1:0] slot = e1_pidx[AW-1:0];",
     "    wire [AW-1:0] slot = e1_pidx[AW-1:0] + AW'(1);",
     "dbg_hist", "history slot off by one"),
    ("trw_lose_record", "tsr_dbg_trw.sv",
     "  wire          put_rec   = in_v && !need_drop && !sfull;",
     "  wire          put_rec   = in_v && !need_drop && !sfull && in_rec[3:0] != 4'd1;",
     "dbg_trw", "TRW losing a record (MARK) without a DROP"),
    ("dmi_en_ignored", "tsr_dbg.sv",
     "  assign dmi_en = c_dmi_en;",
     "  assign dmi_en = 1'b1;",
     "dbg_dmi", "DMI_EN ignored"),
    ("halt_not_one_shot", "tsr_dbg.sv",
     "  wire halt_req = c_halt_en && !(rv_run && rv_halted) &&",
     "  wire halt_req = c_halt_en &&",
     "dbg_halt", "halt request not one-shot"),
    # review fixes (W4-B)
    ("hard_hold_sr_early", "tsr_issue.sv",
     "wire  hard    = dbg_hold[gu] && !abort_act_q && !sr_off_q && !dbg_step[gu];",
     "wire  hard    = dbg_hold[gu] && !abort_act_q && !sr_busy && !dbg_step[gu];",
     "dbg_abort", "hard hold dropped with SR_BUSY, before the QABORT applies"),
    ("hist_not_gated", "tsr_dbg_hist.sv",
     "      rd_pidx   <= present ? main_r.pidx : '0;",
     "      rd_pidx   <= main_r.pidx;",
     "dbg_hist", "a slot that is not PRESENT reading its stale content"),
    ("dmactive_tied", "tsr_dbg.sv",
     "rv_ndmreset, rv_dmactive, smp_valid_q,",
     "rv_ndmreset, 1'b0, smp_valid_q,",
     "dbg_dmi", "DBG_RV_STATE.DMACTIVE not the debug module's"),
    ("trw_any_channel", "tsr_dbg_trw.sv",
     "(32'(base[31:28]) == TRW_CH) &&",
     "1'b1 &&",
     "dbg_trw", "a ring on another channel than TRW_CH written"),
]


def _mutants_requested():
    if os.environ.get("TSR_DBG_MUTANTS") == "1":
        return True
    args = " ".join(sys.argv)
    return "mutants" in args or "slow" in args


@pytest.mark.slow
@pytest.mark.parametrize("m", MUTANTS, ids=[m[0] for m in MUTANTS])
def test_dbg_mutants(m, tmp_path):
    """Mutation spot checks (DB §14 item 7): each mutant must make its test fail. Opt-in:
    TSR_DBG_MUTANTS=1 (or -k mutants / -m slow)."""
    if not _mutants_requested():
        pytest.skip("opt-in: TSR_DBG_MUTANTS=1")
    mid, fname, old, new, test, what = m
    src = (REPO / "rtl/core" / fname).read_text()
    assert src.count(old) == 1, "mutant %s: original text not found once" % mid
    mdir = build_root() / "dbg_mutants" / mid
    shutil.rmtree(mdir, ignore_errors=True)
    mdir.mkdir(parents=True)
    (mdir / fname).write_text(src.replace(old, new))
    env = dict(os.environ, TSR_MUT_DIR=str(mdir), TSR_DBG_MUTANTS="0", TSR_DBG_MUT_TEST=test)
    r = subprocess.run([sys.executable, "-m", "pytest", "-x", "-q", "%s::test_dbg_mutant_target" % (HERE / "test_dbg.py")],
                       capture_output=True, text=True, env=env, cwd=str(REPO / "tb"), timeout=3600)
    shutil.rmtree(mdir, ignore_errors=True)
    assert r.returncode != 0, "mutant %s (%s) survived %s:\n%s" % (mid, what, test, r.stdout[-1500:])
    assert "FAILED" in r.stdout or "failed" in r.stdout, "mutant %s did not run:\n%s" % (mid, r.stdout[-1500:])


def test_dbg_mutant_target():
    """The test a mutant must kill (internal: run by test_dbg_mutants with TSR_MUT_DIR)."""
    t = os.environ.get("TSR_DBG_MUT_TEST")
    if not t or not os.environ.get("TSR_MUT_DIR"):
        pytest.skip("internal: run by test_dbg_mutants")
    _run("mut_" + Path(os.environ["TSR_MUT_DIR"]).name, U200, S.U200_QD, dict(n_prog=4, n_ops=40, n_serial=2, n_trw=40,
                                                                            n_hist=600), tests=[t])
