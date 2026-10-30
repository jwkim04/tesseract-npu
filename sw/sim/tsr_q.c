// tsr_q.c — Tesseract simulator, queue model (P1.6): the RISC-V side issue port and control window
// (CC §5), the issue FIFO, the dispatcher with the checker (EX §8.1, CC §8), the six unit queues
// with tickets, dependencies, SYNC (NOP, FENCE, QCLEAR), holds, QABORT, error capture and
// poisoning (EX §8.2, CC §9), invariants I1-I3, the per-unit records (Q_* for the host), and the
// time base that runs it all (SW §4.4; the P1.6 "fixed" policy, see tsr_sim_int.h §5).
//
// Model (normative sources EX §7-§8, CC §7-§9; decisions where they leave a choice):
//   - Push (CC §7.1): at the ISSUE_PUSH store the opcode's unit u is UNIT_OF(opcode) if the issue
//     path accepts it (tsr_sim_accept_op), else SYNC. Without a credit for u the store waits
//     (TSR_CW_WAIT). Else the entry {words masked by WMASK, DEP0, DEP1, REC, ticket {u,
//     ISSUED_SEQ[u]}, push index ISSUE_PIDX} enters the issue FIFO; ISSUED_SEQ[u]++, ISSUE_PIDX++,
//     ISSUE_TICKET := ticket, DEP0 / DEP1 := none. QCTRL stores put CTRL entries in the same FIFO
//     (QCLEAR takes a SYNC ticket and a push index; QABORT, QHOLD and TRACE_MARK take neither) and
//     wait for FIFO space (and a SYNC credit for QCLEAR) the same way.
//   - Credit of u (CC §7.3) = min(Q_DEPTH[u] - (ISSUED_SEQ[u] - S[u]), free FIFO entries, 15),
//     S[u] = entries of u accepted by the unit or completed without running; 0 for absent units.
//   - Dispatcher (CC §8.3): takes FIFO entries in order (crossing + 1 per cycle, tsr_sim_int.h §5),
//     runs the opcode's checker class (or ILLEGAL_OP for a refused opcode), the BAD_DEP checks on
//     its counts cnt[] and clear points, and enqueues a run entry or a poisoned entry. The check
//     order is EX §8.1's: a BAD_DEP dep is replaced by none, and BAD_DEP wins over every code after
//     it. A poisoned NOP / FENCE is an ordinary poisoned entry (no snapshot wait). FENCE and QCLEAR
//     snapshot cnt[0..5].
//   - Heads (CC §9.3), evaluated in unit order at every time point, one entry per unit at a time:
//     wait for deps (and the FENCE / QCLEAR snapshot) and DB's hard hold; then QCLEAR clears (never
//     aborted, before the unit-error rule: the QCLEAR after a QABORT must clear the SYNC unit's own
//     error state); a poisoned entry completes with its code; in an errored unit ABORTED; with a
//     failed dep DEP_FAILED; NOP / FENCE complete 0; a hold stops the rest; else the op starts.
//     A non-running completion is decided (S++, error state) at once and completes (D++) one
//     cycle later; a running op is accepted (S++) at its start and completes (exec, then D++) at
//     start + cycles(). Units serve one entry at a time, so D[u] counts in sequence order.
//   - Error state (EX §8.2): the first nonzero completion of a unit records {seq, code, info, w0,
//     pidx}; QABORT puts every built unit not already in error into error state with ERR_SEQ = S[u]
//     (the first entry not started), ERR_CODE ABORTED, ERR_W0 / ERR_PIDX of that entry if it is
//     already pushed (queue or issue FIFO), else 0 / the next ISSUE_PIDX.
//   - QCLEAR (RISC-V): after its snapshot, clears every error state and record and sets
//     clear_pt[u] = snap[u]. Host QCLEAR: only with QIDLE and the RISC-V quiet (in reset, or halted);
//     clear_pt[u] = cnt[u].

#define _GNU_SOURCE
#define TSR_SIM_CORE_INTERNAL

#include "tsr_sim.h"
#include "tsr_sim_int.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const uint8_t k_depth_u200[TSR_NUM_UNITS] = {
    TSR_QDEPTH_U200_DMA, TSR_QDEPTH_U200_QACT, TSR_QDEPTH_U200_MVM,
    TSR_QDEPTH_U200_VU,  TSR_QDEPTH_U200_ATT,  TSR_QDEPTH_U200_SYNC,
};
static const uint8_t k_accept_max[TSR_NUM_UNITS] = {
    TSR_ACCEPT_MAX_DMA, TSR_ACCEPT_MAX_QACT, TSR_ACCEPT_MAX_MVM,
    TSR_ACCEPT_MAX_VU,  TSR_ACCEPT_MAX_ATT,  TSR_ACCEPT_MAX_SYNC,
};

static void q_fatal(const tsr_sim * sim, const char * what) {
    fprintf(stderr, "tsr_sim: queue model invariant violated at MTIME %llu: %s\n",
            (unsigned long long)sim->now, what);
    abort();
}

static inline uint32_t tk_unit(uint32_t t) { return t >> 28; }
static inline uint32_t tk_seq(uint32_t t)  { return t & TSR_SEQ_MASK; }
static inline uint32_t tk_make(uint32_t u, uint32_t s) { return (u << 28) | (s & TSR_SEQ_MASK); }
static inline uint32_t seq_sub(uint32_t a, uint32_t b) { return (a - b) & TSR_SEQ_MASK; }

static int built(const tsr_sim * sim, uint32_t u) { return u < TSR_NUM_UNITS && sim->u[u].depth > 0; }

// Ticket (v, s) is done iff (D[v] - s) mod 2^28 in [1, 2^27) (CC §7.5). Unit 15 (none) is done;
// a ticket on a reserved or absent unit is never done.
static int ticket_done(const tsr_sim * sim, uint32_t t) {
    const uint32_t v = tk_unit(t);
    if (v == TSR_UNIT_NONE) return 1;
    if (!built(sim, v)) return 0;
    const uint32_t d = seq_sub(sim->u[v].D, tk_seq(t));
    return d >= 1 && d < TSR_SEQ_HALF28;
}

// ---------------------------------------------------------------------------
// Init / free
// ---------------------------------------------------------------------------
void tsr_q_init(tsr_sim * sim, const uint8_t * depth) {
    for (uint32_t u = 0; u < TSR_NUM_UNITS; ++u) {
        tsr_qunit * q = &sim->u[u];
        memset(q, 0, sizeof(*q));
        q->depth = depth ? depth[u] : k_depth_u200[u];
        q->ring = (tsr_qent *)calloc(q->depth ? q->depth : 1, sizeof(tsr_qent));
        if (q->ring == NULL) {
            fprintf(stderr, "tsr_sim: out of memory (queues)\n");
            abort();
        }
    }
    sim->iss_dep[0] = sim->iss_dep[1] = TSR_TICKET_NONE;
    sim->iss_ticket = TSR_TICKET_NONE;
    sim->wait_ticket = TSR_CW_WAIT_TICKET_RESET;
    sim->wait_credit = TSR_CW_WAIT_CREDIT_RESET;
    sim->mtimecmp = UINT64_MAX;                      // reset all ones
    sim->agent_next = TSR_SIM_AGENT_WAIT;
}

void tsr_q_free(tsr_sim * sim) {
    for (uint32_t u = 0; u < TSR_NUM_UNITS; ++u) {
        free(sim->u[u].ring);
        sim->u[u].ring = NULL;
    }
}

// ---------------------------------------------------------------------------
// Logging and events
// ---------------------------------------------------------------------------
static void q_emit(tsr_sim * sim, uint32_t kind, uint32_t unit, const tsr_qent * e, uint32_t code, uint32_t ran,
                   const uint32_t * mark_w) {
    static const char * const names[] = { "?", "enq", "start", "done", "mark" };
    if (sim->log) {
        if (kind == TSR_QEV_MARK) {
            tsr_sim_log_line(sim, "%10llu mark value=%08x next_pidx=%u", (unsigned long long)sim->now, mark_w[1],
                             sim->q_pidx);
        } else {
            tsr_sim_log_line(sim, "%10llu %-5s u%u seq=%u pidx=%u w0=%08x code=%u%s", (unsigned long long)sim->now,
                             names[kind], unit, e->seq, e->pidx, e->w[0], code,
                             kind == TSR_QEV_DONE ? (ran ? " ran" : " norun") : "");
        }
    }
    if (sim->lis == NULL) return;
    tsr_q_event ev;
    memset(&ev, 0, sizeof(ev));
    ev.kind = kind;
    ev.unit = unit;
    ev.time = sim->now;
    if (kind == TSR_QEV_MARK) {
        ev.ticket = TSR_TICKET_NONE;
        ev.pidx = sim->q_pidx;
        ev.dep[0] = ev.dep[1] = TSR_TICKET_NONE;
        ev.w = mark_w;
    } else {
        ev.ticket = tk_make(unit, e->seq);
        ev.pidx = e->pidx;
        ev.code = code;
        ev.ran = ran;
        ev.rec = e->rec;
        ev.dep[0] = e->dep[0];
        ev.dep[1] = e->dep[1];
        ev.w = e->w;
        ev.ext = &e->ext;
    }
    sim->in_cb++;
    sim->lis(sim->lis_ctx, sim, &ev);
    sim->in_cb--;
}

// ---------------------------------------------------------------------------
// Error state, QABORT, QCLEAR
// ---------------------------------------------------------------------------
static int any_err(const tsr_sim * sim) {
    for (uint32_t u = 0; u < TSR_NUM_UNITS; ++u)
        if (sim->u[u].err) return 1;
    return 0;
}

// ISSUE_STATUS.ERR_IRQ: set on the rising edge of "some unit in error state" (CC §5.1).
static void err_edge(tsr_sim * sim) {
    const int now_err = any_err(sim);
    if (now_err && !sim->err_any) sim->iss_status |= TSR_ISSUE_STATUS_ERR_IRQ;
    sim->err_any = now_err;
}

static void enter_error(tsr_sim * sim, uint32_t u, uint32_t seq, uint32_t code, uint32_t info, uint32_t w0,
                        uint32_t pidx) {
    tsr_qunit * q = &sim->u[u];
    if (q->err) return;                              // the first error since the last clear
    q->err = 1;
    q->err_seq = seq & TSR_SEQ_MASK;
    q->err_code = code;
    q->err_info = info;
    q->err_w0 = w0;
    q->err_pidx = pidx;
    err_edge(sim);
}

void tsr_q_abort(tsr_sim * sim) {
    for (uint32_t u = 0; u < TSR_NUM_UNITS; ++u) {
        tsr_qunit * q = &sim->u[u];
        if (q->depth == 0 || q->err) continue;
        // The first entry not started has seq S[u]: the queue front, else the first FIFO entry of u.
        uint32_t w0 = 0, pidx = sim->iss_pidx;
        if (q->n > 0) {
            w0 = q->ring[q->rd].w[0];
            pidx = q->ring[q->rd].pidx;
        } else {
            for (uint32_t i = 0; i < sim->f_n; ++i) {
                const tsr_ifent * f = &sim->fifo[(sim->f_rd + i) % TSR_ISSUE_FIFO_DEPTH];
                const int ticketed = f->kind == TSR_IFIFO_KIND_OP ||
                                     (f->kind == TSR_IFIFO_KIND_CTRL && f->ctrl == TSR_IFIFO_CTRL_CODE_QCLEAR);
                if (ticketed && tk_unit(f->ticket) == u) {
                    w0 = f->w[0];
                    pidx = f->pidx;
                    break;
                }
            }
        }
        enter_error(sim, u, q->S, TSR_ERR_ABORTED, 0, w0, pidx);
    }
    sim->abort_active = 1;
}

void tsr_q_clear_all(tsr_sim * sim, const uint32_t * clear_pt) {
    for (uint32_t u = 0; u < TSR_NUM_UNITS; ++u) {
        tsr_qunit * q = &sim->u[u];
        q->err = 0;
        q->err_seq = q->err_code = q->err_info = q->err_w0 = q->err_pidx = 0;
        q->clear_pt = (clear_pt ? clear_pt[u] : q->cnt) & TSR_SEQ_MASK;
    }
    sim->err_any = 0;
    sim->abort_active = 0;
}

int tsr_q_idle(const tsr_sim * sim) {
    if (sim->f_n != 0) return 0;
    for (uint32_t u = 0; u < TSR_NUM_UNITS; ++u)
        if (sim->u[u].busy || sim->u[u].n != 0) return 0;
    return 1;
}

int tsr_q_host_clear(tsr_sim * sim) {
    const int rv_quiet = !(sim->ctrl & TSR_CTRL_RV_RUN) || (sim->rv_state & TSR_STATUS_RV_HALTED);
    if (!tsr_q_idle(sim) || !rv_quiet) return 1;
    tsr_q_clear_all(sim, NULL);
    return 0;
}

// The clk_rv side of the RISC-V reset sequence (CC §12.2 step 3): staging, deps, WAIT_*, the
// RISC-V hold source; also the other clk_rv registers of the control window, ISSUE_STATUS and
// MTIMECMP (a booting hart sees no stale interrupt from the abort that the reset itself applied).
// Kept: ISSUE_PIDX, ISSUED_SEQ and ISSUE_TICKET (sequence state, hardware reset only).
void tsr_q_rv_reset(tsr_sim * sim) {
    memset(sim->iss_w, 0, sizeof(sim->iss_w));
    sim->iss_dep[0] = sim->iss_dep[1] = TSR_TICKET_NONE;
    sim->wait_ticket = TSR_CW_WAIT_TICKET_RESET;
    sim->wait_credit = TSR_CW_WAIT_CREDIT_RESET;
    sim->hold_rv = 0;
    sim->iss_status = 0;
    sim->mtimecmp = UINT64_MAX;
    sim->agent_next = TSR_SIM_AGENT_WAIT;
}

// ---------------------------------------------------------------------------
// Credits, holds, status
// ---------------------------------------------------------------------------
static uint32_t q_credit(const tsr_sim * sim, uint32_t u) {
    const tsr_qunit * q = &sim->u[u];
    if (q->depth == 0) return 0;
    const uint32_t used = seq_sub(q->issued, q->S);
    uint32_t c = used >= q->depth ? 0 : q->depth - used;
    const uint32_t ff = TSR_ISSUE_FIFO_DEPTH - sim->f_n;
    if (c > ff) c = ff;
    if (c > 15) c = 15;
    return c;
}

static uint32_t hold_src(const tsr_sim * sim) {
    uint32_t h = 0;
    if (sim->ctrl & TSR_CTRL_HOST_HOLD) h |= TSR_HOLD_SRC_HOST;
    if (sim->hold_rv) h |= TSR_HOLD_SRC_RV;
    if ((sim->ctrl & TSR_CTRL_HOLD_ON_HALT) && (sim->ctrl & TSR_CTRL_RV_RUN) && (sim->rv_state & TSR_STATUS_RV_HALTED))
        h |= TSR_HOLD_SRC_HALT;
    return h;
}

static uint32_t db_hold_mask(const tsr_sim * sim) {
    if (!sim->have_dbg || sim->dbg.unit_hold == NULL) return 0;
    return sim->dbg.unit_hold(sim->dbg.ctx);
}

uint32_t tsr_q_depth_reg(const tsr_sim * sim) {
    uint32_t v = 0;
    for (uint32_t u = 0; u < TSR_NUM_UNITS; ++u) v |= (sim->u[u].depth & 0x1Fu) << (TSR_Q_DEPTH_FIELD_W * u);
    return v;
}

// QQUIESCENT (CC §9.6): no op held by a datapath unit, the dispatcher and FIFO idle, every waiting
// head held or waiting for deps, nothing left to drain. In this model non-running completions
// are decided as soon as possible, so this is "no unit busy and the FIFO empty".
static int quiescent(const tsr_sim * sim) {
    if (sim->f_n != 0) return 0;
    for (uint32_t u = 0; u < TSR_NUM_UNITS; ++u)
        if (sim->u[u].busy) return 0;
    return 1;
}

uint32_t tsr_q_status(const tsr_sim * sim) {
    uint32_t st = 0;
    if (tsr_q_idle(sim)) st |= TSR_STATUS_QIDLE;
    if (quiescent(sim)) st |= TSR_STATUS_QQUIESCENT;
    if (any_err(sim)) st |= TSR_STATUS_QERR;
    st |= TSR_PUT(TSR_STATUS_HOLD_SRC, hold_src(sim));
    return st;
}

static uint32_t q_state(const tsr_sim * sim, uint32_t u) {
    const tsr_qunit * q = &sim->u[u];
    if (q->depth == 0) return 0;
    uint32_t v = TSR_PUT(TSR_Q_STATE_ENTRIES, q->n);
    if (q->busy && q->ran) v |= TSR_Q_STATE_RUNNING;
    if (q->n && (q->head == TSR_QH_DEP_WAIT || q->head == TSR_QH_SNAP_WAIT)) v |= TSR_Q_STATE_DEP_WAIT;
    if (q->n && (q->head == TSR_QH_HELD || q->head == TSR_QH_DBHOLD)) v |= TSR_Q_STATE_HELD;
    if (q->err) v |= TSR_Q_STATE_ERR;
    return v;
}

uint32_t tsr_q_csr_read(const tsr_sim * sim, uint32_t off) {
    if (off == TSR_CSR_Q_FIFO) return sim->f_n;
    if (off == TSR_CSR_Q_PIDX) return sim->q_pidx;
    uint32_t u;
    if (off >= TSR_CSR_Q_DONE(0) && off < TSR_CSR_Q_DONE(TSR_UNIT_SLOTS)) {
        u = (off - TSR_CSR_Q_DONE(0)) / 4;
        return built(sim, u) ? sim->u[u].D : 0;
    }
    if (off >= TSR_CSR_Q_ISSUED(0) && off < TSR_CSR_Q_ISSUED(TSR_UNIT_SLOTS)) {
        u = (off - TSR_CSR_Q_ISSUED(0)) / 4;
        return built(sim, u) ? sim->u[u].issued : 0;
    }
    if (off >= TSR_CSR_Q_STATE(0) && off < TSR_CSR_Q_STATE(TSR_UNIT_SLOTS)) {
        u = (off - TSR_CSR_Q_STATE(0)) / 4;
        return u < TSR_NUM_UNITS ? q_state(sim, u) : 0;
    }
    if (off >= TSR_CSR_Q_ERR_SEQ(0) && off < TSR_CSR_Q_ERR_SEQ(TSR_UNIT_SLOTS)) {
        u = (off - TSR_CSR_Q_ERR_SEQ(0)) / 16;
        if (!built(sim, u)) return 0;
        const tsr_qunit * q = &sim->u[u];
        switch ((off - TSR_CSR_Q_ERR_SEQ(0)) % 16) {
        case 0:  return q->err_seq;
        case 4:  return q->err_code & 0xFFu;
        case 8:  return q->err_info;
        default: return q->err_w0;
        }
    }
    if (off >= TSR_CSR_Q_ERR_PIDX(0) && off < TSR_CSR_Q_ERR_PIDX(TSR_UNIT_SLOTS)) {
        u = (off - TSR_CSR_Q_ERR_PIDX(0)) / 4;
        return built(sim, u) ? sim->u[u].err_pidx : 0;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// Issue FIFO, push, QCTRL
// ---------------------------------------------------------------------------
static tsr_ifent * fifo_push(tsr_sim * sim) {
    tsr_ifent * f = &sim->fifo[(sim->f_rd + sim->f_n) % TSR_ISSUE_FIFO_DEPTH];
    memset(f, 0, sizeof(*f));
    uint64_t t = sim->now + TSR_Q_XING;
    if (t <= sim->f_last) t = sim->f_last + 1;       // the dispatcher takes one entry per cycle
    sim->f_last = t;
    f->t_disp = t;
    sim->f_n++;
    return f;
}

static int q_push(tsr_sim * sim, uint32_t val) {
    const uint32_t opcode = sim->iss_w[0] & 0xFFu;
    const tsr_op_handler * h = tsr_sim_accept_op(sim, opcode);
    const uint32_t u = h ? h->unit : TSR_UNIT_SYNC;
    if (q_credit(sim, u) == 0) return TSR_CW_WAIT;
    tsr_ifent * f = fifo_push(sim);
    f->kind = TSR_IFIFO_KIND_OP;
    f->refused = h == NULL;
    f->rec = (val & TSR_ISSUE_PUSH_REC) ? 1 : 0;
    const uint32_t wmask = TSR_GET(val, TSR_ISSUE_PUSH_WMASK);
    for (uint32_t i = 0; i < 8; ++i) f->w[i] = (i == 0 || ((wmask >> i) & 1u)) ? sim->iss_w[i] : 0u;
    f->dep[0] = sim->iss_dep[0];
    f->dep[1] = sim->iss_dep[1];
    f->ticket = tk_make(u, sim->u[u].issued);
    f->pidx = sim->iss_pidx++;
    sim->u[u].issued = (sim->u[u].issued + 1) & TSR_SEQ_MASK;
    sim->iss_ticket = f->ticket;
    sim->iss_dep[0] = sim->iss_dep[1] = TSR_TICKET_NONE;
    tsr_sim_perf_add(sim, TSR_PERF_PUSHES(sim->num_ch), 1);
    if (sim->log)
        tsr_sim_log_line(sim, "%10llu push  u%u seq=%u pidx=%u w0=%08x dep=%08x,%08x%s", (unsigned long long)sim->now, u,
                         tk_seq(f->ticket), f->pidx, f->w[0], f->dep[0], f->dep[1], f->refused ? " (refused)" : "");
    return TSR_CW_OK;
}

// QCTRL store: each set bit of [3:0] is one CTRL entry, in bit order; all or nothing.
static int q_qctrl(tsr_sim * sim, uint32_t bits) {
    static const uint8_t codes[4] = { TSR_IFIFO_CTRL_CODE_QABORT, TSR_IFIFO_CTRL_CODE_QHOLD_SET,
                                      TSR_IFIFO_CTRL_CODE_QHOLD_CLR, TSR_IFIFO_CTRL_CODE_QCLEAR };
    bits &= 0xFu;
    uint32_t n = 0;
    for (uint32_t b = 0; b < 4; ++b) n += (bits >> b) & 1u;
    if (n == 0) return TSR_CW_OK;
    if (TSR_ISSUE_FIFO_DEPTH - sim->f_n < n) return TSR_CW_WAIT;
    if ((bits & TSR_QCTRL_QCLEAR) && q_credit(sim, TSR_UNIT_SYNC) == 0) return TSR_CW_WAIT;
    for (uint32_t b = 0; b < 4; ++b) {
        if (!((bits >> b) & 1u)) continue;
        tsr_ifent * f = fifo_push(sim);
        f->kind = TSR_IFIFO_KIND_CTRL;
        f->ctrl = codes[b];
        f->w[0] = codes[b];
        f->dep[0] = f->dep[1] = TSR_TICKET_NONE;
        if (codes[b] == TSR_IFIFO_CTRL_CODE_QCLEAR) {
            tsr_qunit * s = &sim->u[TSR_UNIT_SYNC];
            f->ticket = tk_make(TSR_UNIT_SYNC, s->issued);
            f->pidx = sim->iss_pidx++;
            s->issued = (s->issued + 1) & TSR_SEQ_MASK;
            sim->iss_ticket = f->ticket;
            tsr_sim_perf_add(sim, TSR_PERF_PUSHES(sim->num_ch), 1);
        } else {
            f->ticket = TSR_TICKET_NONE;
            f->pidx = sim->iss_pidx;                 // informational: no push index taken
        }
    }
    return TSR_CW_OK;
}

static int q_trace_mark(tsr_sim * sim, uint32_t value) {
    if (sim->f_n >= TSR_ISSUE_FIFO_DEPTH) return TSR_CW_WAIT;
    tsr_ifent * f = fifo_push(sim);
    f->kind = TSR_IFIFO_KIND_CTRL;
    f->ctrl = TSR_IFIFO_CTRL_CODE_TRACE_MARK;
    f->w[0] = TSR_IFIFO_CTRL_CODE_TRACE_MARK;
    f->w[1] = value;
    f->ticket = TSR_TICKET_NONE;
    f->pidx = sim->iss_pidx;
    return TSR_CW_OK;
}

// ---------------------------------------------------------------------------
// Dispatcher (CC §8.3)
// ---------------------------------------------------------------------------
static void q_enqueue(tsr_sim * sim, uint32_t u, const tsr_qent * e) {
    tsr_qunit * q = &sim->u[u];
    if (q->n >= q->depth) q_fatal(sim, "queue overflow (credit accounting)");
    q->ring[(q->rd + q->n) % q->depth] = *e;
    q->n++;
}

// BAD_DEP rule for dep t at the dispatcher (CC §8.3).
static int dep_bad(const tsr_sim * sim, uint32_t t) {
    const uint32_t v = tk_unit(t), s = tk_seq(t);
    if (v == TSR_UNIT_NONE) return 0;
    if (!built(sim, v)) return 1;                    // 6..14, or a unit this configuration lacks
    if (seq_sub(s, sim->u[v].cnt) < TSR_SEQ_HALF28) return 1;   // not pushed yet (incl. the op's own ticket)
    const uint32_t c = seq_sub(sim->u[v].clear_pt, s);
    if (c >= 1 && c < TSR_SEQ_HALF28) return 1;     // pushed before the last QCLEAR
    return 0;
}

// The checker's extent list against tsr_sim_int.h §1: nothing dropped, at most one DRAM entry and
// TSR_EXT_SREG_MAX SREG entries, at most TSR_EXT_MAX SPM / DRAM operands (a READ entry with the
// range of an earlier WRITE entry of the same kind and channel is the read half of a
// read-modify-write operand and counts once).
static void ext_valid(const tsr_sim * sim, const tsr_extents * x) {
    if (x->overflow != 0 || x->n > TSR_EXT_MAX + TSR_EXT_SREG_MAX) q_fatal(sim, "checker listed too many extents");
    uint32_t n_dram = 0, n_sreg = 0, n_ops = 0;
    for (uint32_t i = 0; i < x->n; ++i) {
        const tsr_extent * e = &x->e[i];
        if (e->kind == TSR_EXT_SREG) { ++n_sreg; continue; }
        if (e->kind != TSR_EXT_SPM && e->kind != TSR_EXT_DRAM) q_fatal(sim, "checker listed an extent of no kind");
        if (e->kind == TSR_EXT_DRAM) ++n_dram;
        int rmw = 0;
        for (uint32_t k = 0; k < i && !rmw; ++k) {
            const tsr_extent * w = &x->e[k];
            rmw = e->write == TSR_EXT_READ && w->write == TSR_EXT_WRITE && w->kind == e->kind && w->ch == e->ch &&
                  w->lo == e->lo && w->hi == e->hi;
        }
        if (!rmw) ++n_ops;
    }
    if (n_dram > 1) q_fatal(sim, "checker listed more than one DRAM extent");
    if (n_sreg > TSR_EXT_SREG_MAX) q_fatal(sim, "checker listed too many SREG extents");
    if (n_ops > TSR_EXT_MAX) q_fatal(sim, "checker listed more than four SPM / DRAM operands");
}

static void q_dispatch_one(tsr_sim * sim) {
    tsr_ifent f = sim->fifo[sim->f_rd];
    sim->f_rd = (sim->f_rd + 1) % TSR_ISSUE_FIFO_DEPTH;
    sim->f_n--;
    if (f.kind == TSR_IFIFO_KIND_CTRL) {
        switch (f.ctrl) {
        case TSR_IFIFO_CTRL_CODE_QABORT:    tsr_q_abort(sim); return;
        case TSR_IFIFO_CTRL_CODE_QHOLD_SET: sim->hold_rv = 1; return;
        case TSR_IFIFO_CTRL_CODE_QHOLD_CLR: sim->hold_rv = 0; return;
        case TSR_IFIFO_CTRL_CODE_TRACE_MARK: q_emit(sim, TSR_QEV_MARK, TSR_UNIT_MARKER, NULL, 0, 0, f.w); return;
        case TSR_IFIFO_CTRL_CODE_QCLEAR: break;
        default: return;
        }
    }
    const uint32_t u = tk_unit(f.ticket);
    tsr_qunit * q = &sim->u[u];
    if (tk_seq(f.ticket) != q->cnt || f.pidx != sim->q_pidx) q_fatal(sim, "dispatcher count / push index mismatch");
    tsr_qent e;
    memset(&e, 0, sizeof(e));
    memcpy(e.w, f.w, sizeof(e.w));
    e.seq = q->cnt;
    e.pidx = f.pidx;
    e.rec = f.rec;
    e.t_enq = sim->now;
    e.dep[0] = e.dep[1] = TSR_TICKET_NONE;
    if (f.kind == TSR_IFIFO_KIND_CTRL) {             // QCLEAR: a SYNC entry with a snapshot
        e.kind = TSR_QK_QCLEAR;
        for (uint32_t v = 0; v < TSR_NUM_UNITS; ++v) e.snap[v] = sim->u[v].cnt;
    } else {
        const uint32_t opcode = f.w[0] & 0xFFu;
        uint32_t code = TSR_ERR_ILLEGAL_OP, info = 0;
        const tsr_op_handler * h = f.refused ? NULL : tsr_sim_op_handler((uint8_t)opcode);
        if (h != NULL && h->check != NULL) {             // (a handler removed since the push: ILLEGAL_OP)
            code = h->check(sim, f.w, &e.ext, &info);
            ext_valid(sim, &e.ext);
        }
        int bad = 0;
        for (int i = 0; i < 2; ++i) {
            if (dep_bad(sim, f.dep[i])) bad = 1;
            else e.dep[i] = f.dep[i];
        }
        if (bad && (code == TSR_ERR_NONE || tsr_err_check_rank(code) > tsr_err_check_rank(TSR_ERR_BAD_DEP))) {
            code = TSR_ERR_BAD_DEP;
            info = 0;
        }
        e.code = code;
        e.info = code ? info : 0;
        if (code) memset(&e.ext, 0, sizeof(e.ext));
        if (code == TSR_ERR_NONE && opcode == TSR_OP_NOP) e.kind = TSR_QK_NOP;
        if (code == TSR_ERR_NONE && opcode == TSR_OP_FENCE) {
            e.kind = TSR_QK_FENCE;
            for (uint32_t v = 0; v < TSR_NUM_UNITS; ++v) e.snap[v] = sim->u[v].cnt;
        }
    }
    q->cnt = (q->cnt + 1) & TSR_SEQ_MASK;
    sim->q_pidx++;
    q_enqueue(sim, u, &e);
    q_emit(sim, TSR_QEV_ENQ, u, &e, e.code, 0, NULL);
}

// ---------------------------------------------------------------------------
// Heads and completions (CC §9)
// ---------------------------------------------------------------------------
static int deps_done(const tsr_sim * sim, const tsr_qent * e) {
    return ticket_done(sim, e->dep[0]) && ticket_done(sim, e->dep[1]);
}

// failed(t) = unit v in error state and s at or after ERR_SEQ[v] (CC §9.2); t is done here.
static int dep_failed(const tsr_sim * sim, const tsr_qent * e) {
    for (int i = 0; i < 2; ++i) {
        const uint32_t v = tk_unit(e->dep[i]);
        if (v == TSR_UNIT_NONE || !built(sim, v)) continue;
        if (sim->u[v].err && seq_sub(tk_seq(e->dep[i]), sim->u[v].err_seq) < TSR_SEQ_HALF28) return 1;
    }
    return 0;
}

static int snap_ok(const tsr_sim * sim, const tsr_qent * e) {
    for (uint32_t v = 0; v < TSR_NUM_UNITS; ++v) {
        if (sim->u[v].depth == 0) continue;
        if (seq_sub(sim->u[v].D, e->snap[v]) >= TSR_SEQ_HALF28) return 0;
    }
    return 1;
}

static int q_head(tsr_sim * sim, uint32_t u) {
    tsr_qunit * q = &sim->u[u];
    if (q->busy) return 0;
    if (q->n == 0) {
        q->head = TSR_QH_EMPTY;
        return 0;
    }
    tsr_qent * e = &q->ring[q->rd];
    if (!deps_done(sim, e)) { q->head = TSR_QH_DEP_WAIT; return 0; }
    if ((e->kind == TSR_QK_FENCE || e->kind == TSR_QK_QCLEAR) && !snap_ok(sim, e)) { q->head = TSR_QH_SNAP_WAIT; return 0; }
    const int db = (db_hold_mask(sim) >> u) & 1u;
    if (db && !sim->abort_active && !sim->sr_busy) { q->head = TSR_QH_DBHOLD; return 0; }   // hard hold
    uint32_t code = TSR_ERR_NONE;
    int run = 0;
    if (e->kind == TSR_QK_QCLEAR) {
        tsr_q_clear_all(sim, e->snap);
    } else if (e->code != TSR_ERR_NONE) {
        code = e->code;
    } else if (q->err) {
        code = TSR_ERR_ABORTED;
    } else if (dep_failed(sim, e)) {
        code = TSR_ERR_DEP_FAILED;
    } else if (e->kind == TSR_QK_NOP || e->kind == TSR_QK_FENCE) {
        code = TSR_ERR_NONE;
    } else if (hold_src(sim) || db) {
        q->head = TSR_QH_HELD;
        return 0;
    } else {
        run = 1;
    }
    if (seq_sub(q->S, q->D) != 0 || e->seq != q->S) q_fatal(sim, "head leaves out of order (I1/I3)");
    q->cur = *e;
    q->rd = (q->rd + 1) % q->depth;
    q->n--;
    q->busy = 1;
    q->ran = run;
    q->cur_code = code;
    q->head = TSR_QH_EMPTY;
    q->S = (q->S + 1) & TSR_SEQ_MASK;
    if (run) {
        const tsr_op_handler * h = tsr_sim_op_handler((uint8_t)(q->cur.w[0] & 0xFFu));
        uint64_t cyc = (h && h->cycles) ? h->cycles(sim, q->cur.w) : 1;
        if (cyc == 0) cyc = 1;
        q->cur.t_start = sim->now;
        q->t_done = sim->now + cyc;
        q_emit(sim, TSR_QEV_START, u, &q->cur, 0, 1, NULL);
    } else {
        // COMPLETE(code): the first nonzero code puts the unit in error state (record before D, I2).
        // ERR_INFO is the checker's for a poisoned entry, 0 for ABORTED / DEP_FAILED.
        if (code != TSR_ERR_NONE)
            enter_error(sim, u, q->cur.seq, code, q->cur.code != TSR_ERR_NONE ? q->cur.info : 0, q->cur.w[0], q->cur.pidx);
        q->t_done = sim->now + 1;
    }
    return 1;
}

static int q_complete(tsr_sim * sim, uint32_t u) {
    tsr_qunit * q = &sim->u[u];
    if (!q->busy || q->t_done != sim->now) return 0;
    // I1: an entry completes only at its head, after both deps (and a FENCE / QCLEAR snapshot).
    if (!deps_done(sim, &q->cur) ||
        ((q->cur.kind == TSR_QK_FENCE || q->cur.kind == TSR_QK_QCLEAR) && !snap_ok(sim, &q->cur)))
        q_fatal(sim, "I1: completion before a dependency");
    if (q->ran) {
        const tsr_op_handler * h = tsr_sim_op_handler((uint8_t)(q->cur.w[0] & 0xFFu));
        if (h && h->exec) h->exec(sim, q->cur.w);   // SW §4.1: effects at the completion time
    }
    q->busy = 0;
    q->D = (q->D + 1) & TSR_SEQ_MASK;
    tsr_sim_perf_add(sim, TSR_PERF_RETIRED, 1);
    q_emit(sim, TSR_QEV_DONE, u, &q->cur, q->ran ? 0 : q->cur_code, q->ran, NULL);
    return 1;
}

static void q_invariants(const tsr_sim * sim) {
    for (uint32_t u = 0; u < TSR_NUM_UNITS; ++u) {
        const tsr_qunit * q = &sim->u[u];
        if (q->depth == 0) continue;
        if (seq_sub(q->S, q->D) > k_accept_max[u]) q_fatal(sim, "I3: S - D above the unit's accept limit");
        if (q->n > q->depth || seq_sub(q->issued, q->S) > q->depth) q_fatal(sim, "queue occupancy above Q_DEPTH");
        if (q->busy && q->cur.seq != q->D) q_fatal(sim, "I3: completion out of order");
        if (q->n > 0 && q->ring[q->rd].seq != q->S) q_fatal(sim, "queue front is not the next entry");
        // I2 holds by construction: a record is written when its entry is decided, before its D.
    }
}

// ---------------------------------------------------------------------------
// Time base (tsr_sim_int.h §5)
// ---------------------------------------------------------------------------
uint64_t tsr_sim_now(const tsr_sim * sim) { return sim->now; }

static int agent_live(const tsr_sim * sim) { return sim->agent != NULL && (sim->ctrl & TSR_CTRL_RV_RUN); }

uint64_t tsr_sim_next_event(const tsr_sim * sim) {
    uint64_t t = UINT64_MAX;
    for (uint32_t u = 0; u < TSR_NUM_UNITS; ++u)
        if (sim->u[u].busy && sim->u[u].t_done < t) t = sim->u[u].t_done;
    if (sim->f_n != 0 && sim->fifo[sim->f_rd].t_disp < t) t = sim->fifo[sim->f_rd].t_disp;
    if (agent_live(sim)) {
        if (sim->agent_next != TSR_SIM_AGENT_WAIT) {
            const uint64_t a = sim->agent_next > sim->now ? sim->agent_next : sim->now;
            if (a < t) t = a;
        } else {
            if (sim->agent_kick && sim->now < t) t = sim->now;
            if (sim->mtimecmp > sim->now && sim->mtimecmp < t) t = sim->mtimecmp;
        }
    }
    return t;
}

// PERF time counters over (now, now + dt], the state being constant in between (EX §6).
static void q_integrate(tsr_sim * sim, uint64_t dt) {
    if (dt == 0 || !tsr_sim_active(sim)) return;
    const uint32_t C = sim->num_ch;
    const int halted = (sim->ctrl & TSR_CTRL_RV_RUN) && (sim->rv_state & TSR_STATUS_RV_HALTED);
    if (halted) sim->perf[TSR_PERF_DBG_STOP_CYCLES] += dt;
    if (halted && quiescent(sim)) return;           // QHALT
    sim->perf[TSR_PERF_CYCLES] += dt;
    // RV_WFI: the hart sleeps in WFI (the RISC-V model reports RV_SLEEP from the core time it
    // reached the WFI, tsr_bus.c)
    if ((sim->ctrl & TSR_CTRL_RV_RUN) && (sim->rv_state & TSR_STATUS_RV_SLEEP)) sim->perf[TSR_PERF_RV_WFI(C)] += dt;
    int held = 0;
    for (uint32_t u = 0; u < TSR_NUM_UNITS; ++u) {
        const tsr_qunit * q = &sim->u[u];
        if (q->n && (q->head == TSR_QH_HELD || q->head == TSR_QH_DBHOLD)) held = 1;
        if (!(q->busy && q->ran)) continue;
        switch (u) {
        case TSR_UNIT_DMA:  sim->perf[TSR_PERF_DMA_CYCLES(C)] += dt; break;
        case TSR_UNIT_QACT: sim->perf[TSR_PERF_QACT_CYCLES(C)] += dt; break;
        case TSR_UNIT_MVM: {
            sim->perf[TSR_PERF_MVM_CYCLES] += dt;
            const uint32_t op = q->cur.w[0] & 0xFFu;
            if (op == TSR_OP_ATTQK) sim->perf[TSR_PERF_ATT_QK_CYCLES(C)] += dt;
            if (op == TSR_OP_ATTPV) sim->perf[TSR_PERF_ATT_PV_CYCLES(C)] += dt;
            break;
        }
        case TSR_UNIT_VU:   sim->perf[TSR_PERF_VU_CYCLES] += dt; break;
        case TSR_UNIT_ATT:  sim->perf[TSR_PERF_ATT_CYCLES(C)] += dt; break;
        default: break;
        }
    }
    const tsr_qunit * m = &sim->u[TSR_UNIT_MVM];
    if (m->depth && !m->busy) {
        if (m->n == 0) sim->perf[TSR_PERF_MVM_EMPTY(C)] += dt;
        else sim->perf[TSR_PERF_MVM_WAIT(C)] += dt;
    }
    if (held) sim->perf[TSR_PERF_HOLD_CYCLES(C)] += dt;
}

// Everything due at sim->now (tsr_sim_int.h §5 order).
static void q_timepoint(tsr_sim * sim) {
    int act = 0;
    for (uint32_t u = 0; u < TSR_NUM_UNITS; ++u) act |= q_complete(sim, u);
    while (sim->f_n != 0 && sim->fifo[sim->f_rd].t_disp <= sim->now) {
        q_dispatch_one(sim);
        act = 1;
    }
    for (uint32_t u = 0; u < TSR_NUM_UNITS; ++u) act |= q_head(sim, u);
    tsr_sim_sr_poll(sim);
    q_invariants(sim);
    if (agent_live(sim)) {
        const int due = sim->agent_next != TSR_SIM_AGENT_WAIT
                            ? sim->agent_next <= sim->now
                            : (act || sim->agent_kick || sim->now == sim->mtimecmp);
        if (due) {
            sim->agent_kick = 0;
            // While the agent runs it is not an event of its own: tsr_sim_next_event() then
            // reports the next core event (completion, dispatch, MTIMECMP) and the agent's slice
            // runs up to it instead of one instruction (tsr_bus.c).
            sim->agent_next = TSR_SIM_AGENT_WAIT;
            sim->in_cb++;
            const uint64_t d = sim->agent(sim->agent_ctx, sim);
            sim->in_cb--;
            if (agent_live(sim)) sim->agent_next = d == TSR_SIM_AGENT_WAIT ? d : sim->now + (d ? d : 1);
        }
    }
}

// Processes the next time point if it is due by t_end (returns 1), else moves time to t_end.
static int q_step(tsr_sim * sim, uint64_t t_end) {
    uint64_t t = tsr_sim_next_event(sim);
    if (t < sim->now) t = sim->now;
    if (t > t_end) {
        q_integrate(sim, t_end - sim->now);
        sim->now = t_end;
        return 0;
    }
    q_integrate(sim, t - sim->now);
    sim->now = t;
    q_timepoint(sim);
    return 1;
}

static void no_reentry(const tsr_sim * sim, const char * what) {
    if (sim->in_cb) {
        fprintf(stderr, "tsr_sim: %s called from inside a time-base callback (use tsr_sim_cw_access)\n", what);
        abort();
    }
}

typedef struct { uint32_t lcq_wr, log_wr; int halted, reset_done, sr_busy, qidle; } host_view;

static host_view host_view_of(const tsr_sim * sim) {
    host_view h;
    const int run = (sim->ctrl & TSR_CTRL_RV_RUN) != 0;
    h.lcq_wr = sim->lcq_wr;
    h.log_wr = sim->log_wr;
    h.halted = (run && (sim->rv_state & TSR_STATUS_RV_HALTED)) ||
               TSR_GET(sim->rt_state, TSR_RT_STATE_PHASE) == TSR_RT_STATE_PHASE_TRAPPED;
    h.reset_done = !run;
    h.sr_busy = sim->sr_busy;
    h.qidle = tsr_q_idle(sim);
    return h;
}

static uint32_t host_events(const tsr_sim * sim, const host_view * h0) {
    const host_view h = host_view_of(sim);
    uint32_t ev = 0;
    if (h.lcq_wr != h0->lcq_wr) ev |= TSR_SIM_EV_LCQ;
    if (h.log_wr != h0->log_wr) ev |= TSR_SIM_EV_LOG;
    if (h.halted && !h0->halted) ev |= TSR_SIM_EV_HALT;
    if ((h.reset_done && !h0->reset_done) || (!h.sr_busy && h0->sr_busy)) ev |= TSR_SIM_EV_RESET;
    if (h.qidle != h0->qidle) ev |= TSR_SIM_EV_QIDLE;
    if (sim->stop_req) ev |= TSR_SIM_EV_HALT;
    return ev;
}

uint32_t tsr_sim_advance(tsr_sim * sim, uint64_t max_cycles) {
    no_reentry(sim, "tsr_sim_advance");
    const host_view h0 = host_view_of(sim);
    const uint64_t t_end = (UINT64_MAX - sim->now < max_cycles) ? UINT64_MAX : sim->now + max_cycles;
    sim->stop_req = 0;
    q_timepoint(sim);
    for (;;) {
        const uint32_t ev = host_events(sim, &h0);
        if (ev) {
            sim->stop_req = 0;
            return ev;
        }
        if (sim->now >= t_end) return TSR_SIM_EV_BUDGET;
        q_step(sim, t_end);
    }
}

int tsr_sim_run_until_idle(tsr_sim * sim, uint64_t max_cycles) {
    no_reentry(sim, "tsr_sim_run_until_idle");
    const uint64_t t_end = (UINT64_MAX - sim->now < max_cycles) ? UINT64_MAX : sim->now + max_cycles;
    q_timepoint(sim);
    for (;;) {
        if (tsr_q_idle(sim)) return 1;
        if (sim->now >= t_end) return 0;
        q_step(sim, t_end);
    }
}

void tsr_sim_request_stop(tsr_sim * sim) { sim->stop_req = 1; }

void tsr_sim_set_listener(tsr_sim * sim, tsr_q_listener fn, void * ctx) {
    sim->lis = fn;
    sim->lis_ctx = ctx;
}

void tsr_sim_set_rv_agent(tsr_sim * sim, tsr_sim_agent_step step, void * ctx) {
    sim->agent = step;
    sim->agent_ctx = ctx;
    sim->agent_next = sim->now;
}

void tsr_sim_set_dbg_hooks(tsr_sim * sim, const tsr_sim_dbg_hooks * hooks) {
    if (hooks == NULL) {
        memset(&sim->dbg, 0, sizeof(sim->dbg));
        sim->have_dbg = 0;
    } else {
        sim->dbg = *hooks;
        sim->have_dbg = 1;
    }
}

// ---------------------------------------------------------------------------
// Control window (CC §5)
// ---------------------------------------------------------------------------
static uint32_t q_cw_read(const tsr_sim * sim, uint32_t off) {
    if (off < TSR_CW_ISSUE_W(8)) return sim->iss_w[off / 4];
    if (off >= TSR_CW_DONE_SEQ(0) && off < TSR_CW_DONE_SEQ(TSR_UNIT_SLOTS)) {
        const uint32_t u = (off - TSR_CW_DONE_SEQ(0)) / 4;
        return built(sim, u) ? sim->u[u].D : 0;
    }
    if (off >= TSR_CW_ISSUED_SEQ(0) && off < TSR_CW_ISSUED_SEQ(TSR_UNIT_SLOTS)) {
        const uint32_t u = (off - TSR_CW_ISSUED_SEQ(0)) / 4;
        return built(sim, u) ? sim->u[u].issued : 0;
    }
    if (off >= TSR_CW_ERR_SEQ(0) && off < TSR_CW_ERR_SEQ(TSR_UNIT_SLOTS)) {
        const uint32_t u = (off - TSR_CW_ERR_SEQ(0)) / 16;
        if (!built(sim, u)) return 0;
        const tsr_qunit * q = &sim->u[u];
        switch ((off - TSR_CW_ERR_SEQ(0)) % 16) {
        case 0:  return q->err_seq;
        case 4:  return q->err_code & 0xFFu;
        case 8:  return q->err_info;
        default: return q->err_w0;
        }
    }
    if (off >= TSR_CW_ERR_PIDX(0) && off < TSR_CW_ERR_PIDX(TSR_UNIT_SLOTS)) {
        const uint32_t u = (off - TSR_CW_ERR_PIDX(0)) / 4;
        return built(sim, u) ? sim->u[u].err_pidx : 0;
    }
    switch (off) {
    case TSR_CW_ISSUE_DEP0:   return sim->iss_dep[0];
    case TSR_CW_ISSUE_DEP1:   return sim->iss_dep[1];
    case TSR_CW_ISSUE_TICKET: return sim->iss_ticket;
    case TSR_CW_ISSUE_CREDIT: {
        uint32_t v = 0;
        for (uint32_t u = 0; u < TSR_NUM_UNITS; ++u) v |= q_credit(sim, u) << (TSR_CREDIT_FIELD_W * u);
        return v;
    }
    case TSR_CW_ISSUE_UNITS: {
        uint32_t v = 0;
        for (uint32_t u = 0; u < TSR_NUM_UNITS; ++u) v |= (uint32_t)built(sim, u) << u;
        return v;
    }
    case TSR_CW_ISSUE_STATUS: return sim->iss_status;
    case TSR_CW_ISSUE_PIDX:   return sim->iss_pidx;
    case TSR_CW_QSTATUS: {
        const uint32_t h = hold_src(sim);
        uint32_t v = tsr_q_idle(sim) ? TSR_QSTATUS_QIDLE : 0u;
        if (h) v |= TSR_QSTATUS_HOLD;
        if (h & TSR_HOLD_SRC_HOST) v |= TSR_QSTATUS_HOLD_HOST;
        if (h & TSR_HOLD_SRC_RV) v |= TSR_QSTATUS_HOLD_RV;
        if (h & TSR_HOLD_SRC_HALT) v |= TSR_QSTATUS_HOLD_HALT;
        for (uint32_t u = 0; u < TSR_NUM_UNITS; ++u) {
            if (sim->u[u].err) v |= 1u << (TSR_QSTATUS_ERR_LSB + u);
            if (sim->u[u].n) v |= 1u << (TSR_QSTATUS_NONEMPTY_LSB + u);
        }
        return v;
    }
    case TSR_CW_WAIT_TICKET:  return sim->wait_ticket;
    case TSR_CW_WAIT_CREDIT:  return sim->wait_credit;
    case TSR_CW_MTIME_LO:     return (uint32_t)sim->now;
    case TSR_CW_MTIME_HI:     return (uint32_t)(sim->now >> 32);
    case TSR_CW_MTIMECMP_LO:  return (uint32_t)sim->mtimecmp;
    case TSR_CW_MTIMECMP_HI:  return (uint32_t)(sim->mtimecmp >> 32);
    default: return 0;
    }
}

static int q_cw_write(tsr_sim * sim, uint32_t off, uint32_t v) {
    if (off < TSR_CW_ISSUE_W(8)) {
        sim->iss_w[off / 4] = v;
        return TSR_CW_OK;
    }
    switch (off) {
    case TSR_CW_ISSUE_DEP0:   sim->iss_dep[0] = v; return TSR_CW_OK;
    case TSR_CW_ISSUE_DEP1:   sim->iss_dep[1] = v; return TSR_CW_OK;
    case TSR_CW_ISSUE_PUSH:   return q_push(sim, v);
    case TSR_CW_ISSUE_STATUS:
        sim->iss_status &= ~(v & (TSR_ISSUE_STATUS_PUSH_TIMEOUT | TSR_ISSUE_STATUS_ERR_IRQ));
        return TSR_CW_OK;
    case TSR_CW_QCTRL:        return q_qctrl(sim, v);
    case TSR_CW_WAIT_TICKET:  sim->wait_ticket = v; return TSR_CW_OK;
    case TSR_CW_WAIT_CREDIT:  sim->wait_credit = v & (TSR_WAIT_CREDIT_UNIT | TSR_WAIT_CREDIT_EN); return TSR_CW_OK;
    case TSR_CW_TRACE_MARK:   return q_trace_mark(sim, v);
    case TSR_CW_SPM_LB_INV:   return TSR_CW_OK;     // the line buffer is the bus model's (tsr_bus)
    case TSR_CW_MTIMECMP_LO:  sim->mtimecmp = (sim->mtimecmp & 0xFFFFFFFF00000000ull) | v; return TSR_CW_OK;
    case TSR_CW_MTIMECMP_HI:  sim->mtimecmp = (sim->mtimecmp & 0xFFFFFFFFull) | ((uint64_t)v << 32); return TSR_CW_OK;
    default: return TSR_CW_OK;                       // read-only / reserved: ignored
    }
}

int tsr_q_cw_access(tsr_sim * sim, int is_write, uint32_t off, uint32_t * val) {
    if (off % 4 != 0 || off >= TSR_RV_CTL_SIZE) {
        if (!is_write) *val = 0;
        return TSR_CW_OK;
    }
    if (off >= TSR_CW_LQ_HEAD_W(0) && off < TSR_CW_MTIME_LO) return tsr_sim_lq_cw_access(sim, is_write, off, val);
    const int db_rec = off >= TSR_CW_REC_COUNT && off <= TSR_CW_REC_CFG && off != TSR_CW_TRACE_MARK;
    const int db_win = off >= TSR_CW_DBG_FLAGS && off < 0x800u;
    if (db_rec || db_win) {
        if (is_write) {
            if (sim->have_dbg && sim->dbg.cw_write) sim->dbg.cw_write(sim->dbg.ctx, off, *val);
        } else {
            *val = 0;
            if (sim->have_dbg && sim->dbg.cw_read) sim->dbg.cw_read(sim->dbg.ctx, off, val);
        }
        return TSR_CW_OK;
    }
    if (!is_write) {
        *val = q_cw_read(sim, off);
        return TSR_CW_OK;
    }
    return q_cw_write(sim, off, *val);
}

int tsr_sim_cw_access(tsr_sim * sim, int is_write, uint32_t off, uint32_t * val) {
    return tsr_q_cw_access(sim, is_write, off, val);
}

uint32_t tsr_sim_cw_read(tsr_sim * sim, uint32_t off) {
    uint32_t v = 0;
    tsr_q_cw_access(sim, 0, off, &v);
    return v;
}

void tsr_sim_cw_push_timeout(tsr_sim * sim) {
    sim->iss_status |= TSR_ISSUE_STATUS_PUSH_TIMEOUT;
    sim->sticky |= TSR_STATUS_PUSH_TIMEOUT;
}

// A store that may wait for a credit / FIFO space, outside the time base: advances time like
// the hardware wait (CC §7.3), at most ISSUE_WAIT_MAX clk_rv cycles, then PUSH_TIMEOUT.
static int store_waiting(tsr_sim * sim, uint32_t off, uint32_t val) {
    no_reentry(sim, "a waiting control-window store");
    const uint64_t limit = (uint64_t)((double)TSR_ISSUE_WAIT_MAX / sim->rv_ratio + 0.5);
    const uint64_t t0 = sim->now;
    for (;;) {
        uint32_t v = val;
        const int rc = tsr_q_cw_access(sim, 1, off, &v);
        if (rc != TSR_CW_WAIT) {
            const uint64_t waited = sim->now - t0;
            if (waited) tsr_sim_perf_add(sim, TSR_PERF_PUSH_STALL(sim->num_ch), (uint64_t)((double)waited * sim->rv_ratio));
            return rc;
        }
        if (sim->now - t0 >= limit) {
            tsr_sim_cw_push_timeout(sim);
            return TSR_CW_FAULT;
        }
        q_step(sim, t0 + limit);
    }
}

void tsr_sim_cw_write(tsr_sim * sim, uint32_t off, uint32_t val) {
    if (off == TSR_CW_ISSUE_PUSH || off == TSR_CW_QCTRL || off == TSR_CW_TRACE_MARK) {
        (void)store_waiting(sim, off, val);
        return;
    }
    tsr_q_cw_access(sim, 1, off, &val);
}

uint32_t tsr_sim_push(tsr_sim * sim, const uint32_t w[8], uint32_t dep0, uint32_t dep1, uint32_t push_val) {
    memcpy(sim->iss_w, w, sizeof(sim->iss_w));
    sim->iss_dep[0] = dep0;
    sim->iss_dep[1] = dep1;
    return store_waiting(sim, TSR_CW_ISSUE_PUSH, push_val) == TSR_CW_OK ? sim->iss_ticket : TSR_TICKET_NONE;
}

uint32_t tsr_sim_qctrl(tsr_sim * sim, uint32_t bits) {
    const int rc = store_waiting(sim, TSR_CW_QCTRL, bits);
    return (rc == TSR_CW_OK && (bits & TSR_QCTRL_QCLEAR)) ? sim->iss_ticket : TSR_TICKET_NONE;
}

// Interrupt levels (CC §12.3) as an Ibex mip mask.
uint32_t tsr_sim_rv_irq(tsr_sim * sim) {
    uint32_t m = 0;
    if (sim->now >= sim->mtimecmp) m |= TSR_RV_MIP_MTIP;
    if (tk_unit(sim->wait_ticket) != TSR_UNIT_NONE && ticket_done(sim, sim->wait_ticket)) m |= TSR_RV_MIP_FAST(0);
    if (sim->iss_status & TSR_ISSUE_STATUS_ERR_IRQ) m |= TSR_RV_MIP_FAST(1);
    if (sim->pushed != sim->taken && !tsr_sim_active(sim)) m |= TSR_RV_MIP_FAST(2);
    if ((uint32_t)(sim->lcq_wr - sim->lcq_rd) < TSR_LCQ_DEPTH) m |= TSR_RV_MIP_FAST(3);
    if ((sim->wait_credit & TSR_WAIT_CREDIT_EN) && TSR_GET(sim->wait_credit, TSR_WAIT_CREDIT_UNIT) < TSR_NUM_UNITS &&
        q_credit(sim, TSR_GET(sim->wait_credit, TSR_WAIT_CREDIT_UNIT)) != 0)
        m |= TSR_RV_MIP_FAST(5);
    return m;
}
