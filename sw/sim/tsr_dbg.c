// tsr_dbg.c — DB simulator models (P1.13): tile-level debug, the DBG register block, DB's control-
// window registers, completion records, the trace writer, the dispatch history, RV watchpoints and
// DB's halt requests (docs/v2/05-debug.md "DB" §3.3, §3.7-§3.9, §5, §6.1-§6.2, §7.1, §8.1-§8.4,
// §9.1, §10.1-§10.2; EX §5-§6, §8.5; CC §9.2-§9.6). Interface and scope: include/tsr_dbg.h.
//
// The model works only through the simulator's hooks (tsr_sim_int.h §5): the debug hooks (DBG block
// and DMI on the host port, DB's control-window registers, dbg_hold[u] / dbg_step[u] into the queue
// heads, the REC interrupt level, the stop level for PERF DBG_STOP_CYCLES, the RISC-V state and the
// SOFT_RESET step 1), one queue listener (the dispatcher's enqueue tap, start and completion
// strobes, markers, pushes), the hart's debug events (through the DM model's event hook, or the
// hart's own hook without the DM) and its data watch (RV watchpoints). Model decisions where DB
// leaves a choice (reported in the P1.13 decision log):
//   - Shadow queues (DB §8.3) mirror each unit queue from the ENQ / START / DONE events: an entry
//     becomes the head when every earlier entry of its unit left the queue (started, or completed
//     without running); they give head_pidx[u], oldest_pidx[u], min_out, the probes and the
//     trace-writer times (QWAIT / HWAIT).
//   - T, DRAINED and dbg_hold[u] are evaluated from the current state whenever the queue model
//     asks (unit_hold, called at each head evaluation), so a threshold set at an entry's enqueue
//     (TBP match) holds it before its first evaluation (DB §5.1).
//   - STOP_NOW, ERR_STOP and HUNG_STOP keep an earlier valid S (S <- the earlier of S and
//     DBG_NEXT_PIDX); a DBG_STOP_AT write sets S unconditionally. RELEASE and SOFT_RESET also end
//     a QSTEP in progress (STEP_BUSY <- 0). During a QSTEP dbg_step[u] is armed for X whatever the
//     hold sources are (it only matters under the debug-halt hold; the host and RISC-V holds still
//     block it, DB §5.2).
//   - A stop activation (STOP_NOW, STOP_AT, a TBP match that is not COUNT_ONLY, ERR, HUNG, a QSTEP
//     acting as STOP_NOW) writes a STOP record and, with STOP_HALT_RV or the TBP's HALT_RV (and
//     RV_HALT_EN), raises the halt request; a QSTEP moving T writes a STOP record without a halt
//     request. The simulator's tsr_sim_advance returns (TSR_SIM_EV_HALT) when DRAINED rises and
//     when a QSTEP completes.
//   - Halt requests (DB §3.3): pend is cleared by RV_HALTED (the hart's debug-entry event, at
//     once), RV_HALT_CANCEL and the RISC-V reset: when the hart leaves reset (its RESET event) pend
//     becomes HALT_ON_BOOT and RV_HALT_EN, so a request from before a reset never halts the next
//     boot and halt-on-boot stops the hart before its first instruction. dbg_req_rv goes to the DM
//     model (tsr_dm_set_ext_haltreq) when it is attached, else straight to the hart's debug_req_i.
//     STATUS.RV_HALTED (the agent's report) rises at the core time of the debug entry (tsr_bus.c,
//     P1.13), which is when PERF DBG_STOP_CYCLES and the RVHALT record start.
//   - The REC FIFO and REC_CFG are clk_rv state: they restart empty when the hart leaves reset.
//     Markers always make a REC record (DB §8.4).
//   - SBA_REFUSED: an SBA access started through the DMI window that ends with sberror at an
//     address CC's rules refuse (DB §3.6); DMI accesses made directly on the DM model (tsr_dm.h)
//     bypass the window and are not seen.
//   - Trace writer: records go to a 32-record staging buffer; a full buffer, TRW_FLUSH, a
//     LAUNCH_END marker, TRW_EN falling or 65,536 cycles without a new record write it to the ring
//     at once (no DRAM timing model; the idle flush is applied when the model next looks: at a
//     record, an event or a DBG access). No PAD records (writes are byte-enabled). A ring of
//     DBG_TRW_LINES = 0 (reset) or outside the channel drops every record.
//   - History DEP0 / DEP1 are the deps as pushed (the replayable push stream, DB §5.6 export);
//     INFO.STOP marks the entry whose push index is T right after its enqueue.
//   - ERR_STOP fires at the completion of the entry that put its unit in error state (its seq and
//     code equal the unit's error record); HIST_FREEZE_ERR counts from the first completion with a
//     nonzero code since HIST_CLEAR. UNIT_HUNG (EX §8.6) is polled from STATUS.

#include "tsr_dbg.h"
#include "tsr_sim_int.h"

#include <stdlib.h>
#include <string.h>

#if defined(TSR_SIM_HAVE_RV) && TSR_SIM_HAVE_RV
#include "tsr_bus.h"      // tsr_rv_agent_rv: the built-in hart
#include "tsr_rv.h"
#define DBG_HAVE_RV 1
#else
#define DBG_HAVE_RV 0
#endif
#if defined(TSR_SIM_HAVE_DM) && TSR_SIM_HAVE_DM
#include "tsr_dm.h"
#define DBG_HAVE_DM 1
#else
#define DBG_HAVE_DM 0
#endif

#define SHQ_CAP     64u               // shadow-queue entries per unit (Q_DEPTH <= 31, + the running op)
#define PPC_N       64u               // push records waiting for their enqueue (issue FIFO: 32)
#define TRW_STAGE   32u               // DB §8.3: 32-record staging
#define TRW_IDLE    65536u            // DB §8.3: partial burst after 65,536 idle cycles
#define HIST_HALF   (TSR_HIST_DEPTH / 2u)

enum { STEP_IDLE = 0, STEP_DRAIN = 1, STEP_DONE = 2 };

#define STICKY_MASK (TSR_DBG_STATUS_RVWP_HIT | TSR_DBG_STATUS_TRW_DROP | TSR_DBG_STATUS_REC_DROP |               \
                     TSR_DBG_STATUS_SBA_REFUSED | TSR_DBG_STATUS_DMI_BLOCKED | TSR_DBG_STATUS_RV_SAMPLE_TIMEOUT | \
                     TSR_DBG_STATUS_RV_DFAULT | TSR_DBG_STATUS_PTR_ERR | TSR_DBG_STATUS_RVWP_TIMEOUT |            \
                     TSR_DBG_STATUS_MATCHQ_OVF | TSR_DBG_STATUS_DMI_RESP_ERR)
#define CFG_MASK    0x00000FFFu
#define TBP_CTL_MASK (TSR_DBG_TBP_CTL_EN | TSR_DBG_TBP_CTL_KIND | TSR_DBG_TBP_CTL_R | TSR_DBG_TBP_CTL_W |          \
                      TSR_DBG_TBP_CTL_AFTER | TSR_DBG_TBP_CTL_HALT_RV | TSR_DBG_TBP_CTL_COUNT_ONLY |              \
                      TSR_DBG_TBP_CTL_CHMASK)
#define RVWP_CTL_MASK (TSR_DBG_RVWP_CTL_EN | TSR_DBG_RVWP_CTL_LOAD | TSR_DBG_RVWP_CTL_STORE | TSR_DBG_RVWP_CTL_HALT)

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
typedef struct shent {
    uint32_t pidx, ticket, w0;
    uint64_t t_disp, t_head, t_start;
    uint8_t  started;        // accepted by the unit (START)
    uint8_t  has_head;       // t_head known
    uint8_t  rec, tbp;
} shent;

typedef struct shq {
    shent    e[SHQ_CAP];
    uint32_t rd, n;
    uint64_t t_oldest;       // time the front became the oldest outstanding entry
} shq;

typedef struct hist_ent {
    uint32_t pidx, ticket, dep[2], w[8], pc, time, info;
    uint8_t  valid;
} hist_ent;

typedef struct mq_ent { uint32_t pidx, bp; } mq_ent;

typedef struct ppc_ent { uint32_t pidx, dep[2]; uint8_t valid; } ppc_ent;

struct tsr_dbg {
    tsr_sim * sim;
    int       lis_id;
    uint32_t  num_ch;
    uint64_t  ch_bytes;

    // DBG block registers (DB §10.1)
    uint32_t  cfg, sticky, step_n, hist_idx;

    // stop logic (DB §5.2)
    uint32_t  S, s_cause;
    int       s_valid;
    mq_ent    mq[TSR_MATCHQ_DEPTH];
    uint32_t  mq_rd, mq_n;
    uint32_t  last_T;
    uint32_t  next_pidx;     // DBG_NEXT_PIDX (= Q_PIDX), tracked from the enqueue tap
    uint32_t  first_pidx;    // DBG_NEXT_PIDX at attach: earlier entries are not shadowed
    int       step_phase, step_busy, step_arm;
    uint32_t  step_left, step_x, step_arm_pidx;
    int       drained_prev;

    // tile breakpoints (DB §5.3)
    uint32_t  tbp_ctl[TSR_DBG_NTBP], tbp_a[TSR_DBG_NTBP], tbp_b[TSR_DBG_NTBP], tbp_hits[TSR_DBG_NTBP];

    // RV watchpoints (DB §3.7): core-domain registers and the copy applied in clk_rv
    uint32_t  wp_ctl[TSR_DBG_NRVWP], wp_lo[TSR_DBG_NRVWP], wp_hi[TSR_DBG_NRVWP], wp_hits[TSR_DBG_NRVWP];
    uint32_t  wpa_ctl[TSR_DBG_NRVWP], wpa_lo[TSR_DBG_NRVWP], wpa_hi[TSR_DBG_NRVWP];
    uint32_t  wp_addr;

    // halt request (DB §3.3) and the RISC-V view
    int       pend, req_level, req_driven;
    int       rv_halted;
    uint32_t  dbg_entries;
    int       freeze_level, freeze_driven;

    // RV sample (DB §6.2)
    uint32_t  smp[5], smp_seq, last_daddr;
    int       smp_valid;

    // shadow queues, push records
    shq       q[TSR_NUM_UNITS];
    ppc_ent   ppc[PPC_N];

    // dispatch history (DB §5.6)
    hist_ent  hist[TSR_HIST_DEPTH];
    uint32_t  hist_head, hist_err_pidx;
    int       hist_err, hist_hung;

    // completion records (DB §8.2)
    uint32_t  rec[TSR_REC_DEPTH][4];
    uint32_t  rec_rd, rec_n, rec_cfg;

    // trace writer (DB §8.3)
    uint32_t  trw_base, trw_lines, trw_wptr, trw_rptr, trw_drops, trw_mask;
    uint32_t  trw_st[TRW_STAGE][8];
    uint32_t  trw_ns, trw_drop_pend;
    uint64_t  trw_t_last;

    int       hung_prev;
};

static inline int ge(uint32_t a, uint32_t b) { return (uint32_t)(a - b) < 0x80000000u; }   // DB §5.1
static inline uint32_t sat32(uint64_t v) { return v > 0xFFFFFFFFull ? 0xFFFFFFFFu : (uint32_t)v; }
static inline uint32_t sat_inc(uint32_t v) { return v == 0xFFFFFFFFu ? v : v + 1u; }
static inline uint64_t now_of(const tsr_dbg * d) { return tsr_sim_now(d->sim); }
static uint32_t core_status(tsr_dbg * d) { return tsr_sim_corewin_read(d->sim, TSR_CSR_STATUS); }

#if DBG_HAVE_RV
static tsr_rv * hart(tsr_dbg * d) {
    struct tsr_rv_agent * a = tsr_sim_rv_agent(d->sim);
    return a ? tsr_rv_agent_rv(a) : NULL;
}
#endif
#if DBG_HAVE_DM
static tsr_dm * dmod(tsr_dbg * d) { return tsr_sim_dm(d->sim); }
#endif

// ---------------------------------------------------------------------------
// Shadow queues
// ---------------------------------------------------------------------------
static shent * sh_at(shq * q, uint32_t i) { return &q->e[(q->rd + i) % SHQ_CAP]; }
static const shent * sh_cat(const shq * q, uint32_t i) { return &q->e[(q->rd + i) % SHQ_CAP]; }

// The head: the first entry not accepted by the unit (NULL if none).
static const shent * sh_head(const shq * q) {
    for (uint32_t i = 0; i < q->n; ++i)
        if (!sh_cat(q, i)->started) return sh_cat(q, i);
    return NULL;
}

// Sets t_head of the first entry after index i that has none (it became the head at time t).
static void sh_next_head(shq * q, uint32_t i, uint64_t t) {
    if (i + 1 < q->n) {
        shent * n = sh_at(q, i + 1);
        if (!n->has_head) {
            n->t_head = t > n->t_disp ? t : n->t_disp;
            n->has_head = 1;
        }
    }
}

// min_out (DB §5.2): the smallest push index among dispatched, not completed entries (by age
// against DBG_NEXT_PIDX).
static int min_out(const tsr_dbg * d, uint32_t * p) {
    int have = 0;
    uint32_t best = 0, best_age = 0;
    for (uint32_t u = 0; u < TSR_NUM_UNITS; ++u) {
        if (d->q[u].n == 0) continue;
        const uint32_t x = sh_cat(&d->q[u], 0)->pidx, age = d->next_pidx - x;
        if (!have || age > best_age) { best = x; best_age = age; have = 1; }
    }
    *p = best;
    return have;
}

// ---------------------------------------------------------------------------
// Stop logic (DB §5.2)
// ---------------------------------------------------------------------------
static int stop_T(const tsr_dbg * d, uint32_t * T) {
    int have = 0;
    uint32_t t = 0;
    if (d->s_valid) { t = d->S; have = 1; }
    if (d->mq_n) {
        const uint32_t m = d->mq[d->mq_rd].pidx;
        if (!have || ge(t, m)) t = m;                    // the earlier of S and the head of M
        have = 1;
    }
    if (have && T) *T = t;
    return have;
}

static uint32_t stop_cause(const tsr_dbg * d) {
    return (d->s_valid ? d->s_cause : 0u) | (d->mq_n ? TSR_DBG_CAUSE_BP : 0u);
}

static uint32_t bp_hit(const tsr_dbg * d) {
    uint32_t m = 0;
    for (uint32_t i = 0; i < d->mq_n; ++i) m |= d->mq[(d->mq_rd + i) % TSR_MATCHQ_DEPTH].bp;
    return m;
}

static void mq_push(tsr_dbg * d, uint32_t p, uint32_t bp) {
    if (d->mq_n) {
        mq_ent * t = &d->mq[(d->mq_rd + d->mq_n - 1) % TSR_MATCHQ_DEPTH];
        if (t->pidx == p) { t->bp |= bp; return; }       // matches arrive in push order
    }
    if (d->mq_n >= TSR_MATCHQ_DEPTH) {                   // at most sum Q_DEPTH <= 44 pending (DB §5.2)
        d->sticky |= TSR_DBG_STATUS_MATCHQ_OVF;
        return;
    }
    mq_ent * e = &d->mq[(d->mq_rd + d->mq_n) % TSR_MATCHQ_DEPTH];
    e->pidx = p;
    e->bp = bp;
    d->mq_n++;
}

static void mq_pop_upto(tsr_dbg * d, uint32_t x) {       // pop every m with ge(x, m)
    while (d->mq_n && ge(x, d->mq[d->mq_rd].pidx)) {
        d->mq_rd = (d->mq_rd + 1) % TSR_MATCHQ_DEPTH;
        d->mq_n--;
    }
}

// DRAINED = STOP_ACTIVE and ge(DBG_NEXT_PIDX, T) and every unit's oldest outstanding entry is at
// or after T.
static int drained(const tsr_dbg * d) {
    uint32_t T;
    if (!stop_T(d, &T) || !ge(d->next_pidx, T)) return 0;
    for (uint32_t u = 0; u < TSR_NUM_UNITS; ++u)
        if (d->q[u].n && !ge(sh_cat(&d->q[u], 0)->pidx, T)) return 0;
    return 1;
}

// dbg_hold[u] for every unit with a head (bit u), not forced to 0.
static uint32_t hold_mask(const tsr_dbg * d) {
    uint32_t T = 0;
    const int active = stop_T(d, &T);
    const int serial = (d->cfg & TSR_DBG_CFG_SERIAL) != 0;
    if (!active && !serial) return 0;
    uint32_t mo = 0;
    const int have_mo = serial ? min_out(d, &mo) : 0;
    uint32_t m = 0;
    for (uint32_t u = 0; u < TSR_NUM_UNITS; ++u) {
        const shent * h = sh_head(&d->q[u]);
        if (h == NULL) continue;
        if ((active && ge(h->pidx, T)) || (serial && have_mo && h->pidx != mo)) m |= 1u << u;
    }
    return m;
}

// ---------------------------------------------------------------------------
// Halt requests (DB §3.3), dbg_stop (DB §3.9), the RISC-V view
// ---------------------------------------------------------------------------
static int hart_halted_now(tsr_dbg * d) {
    if (!tsr_sim_rv_run(d->sim)) return 0;
#if DBG_HAVE_DM
    if (dmod(d)) return tsr_dm_rv_halted(dmod(d)) != 0;
#endif
#if DBG_HAVE_RV
    tsr_rv * rv = hart(d);
    if (rv) return tsr_rv_debug_mode(rv) != 0;
#endif
    return (core_status(d) & TSR_STATUS_RV_HALTED) != 0;
}

// dbg_req_rv = pend and not RV_HALTED (pend is cleared whenever the hart is seen halted).
static void req_drive(tsr_dbg * d) {
    const int lvl = d->pend != 0;
    if (d->req_driven && lvl == d->req_level) return;
    d->req_level = lvl;
    d->req_driven = 1;
#if DBG_HAVE_DM
    if (dmod(d)) {
        tsr_dm_set_ext_haltreq(dmod(d), lvl);
        return;
    }
#endif
#if DBG_HAVE_RV
    tsr_rv * rv = hart(d);
    if (rv) tsr_rv_set_debug_req(rv, lvl);
#endif
}

static void halt_request(tsr_dbg * d) {
    if (!(d->cfg & TSR_DBG_CFG_RV_HALT_EN)) return;
    if (hart_halted_now(d)) return;                      // RV_HALTED clears pend: no stale request
    d->pend = 1;
    req_drive(d);
}

// dbg_stop: a DB stop or the debug-halt hold in effect. CC suspends the ISSUE_WAIT_MAX count
// while it is 1 (CC §7.3), so the hart's wait count is frozen.
// st: the core's STATUS.
static int dbg_stop_level(const tsr_dbg * d, uint32_t st) {
    return stop_T(d, NULL) || (TSR_GET(st, TSR_STATUS_HOLD_SRC) & TSR_HOLD_SRC_HALT);
}

static void freeze_update(tsr_dbg * d, uint32_t st) {
#if DBG_HAVE_RV
    const int lvl = dbg_stop_level(d, st);
    if (d->freeze_driven && lvl == d->freeze_level) return;
    d->freeze_level = lvl;
    d->freeze_driven = 1;
    tsr_rv * rv = hart(d);
    if (rv) tsr_rv_freeze_wait(rv, lvl);
#else
    (void)d; (void)st;
#endif
}

// ---------------------------------------------------------------------------
// Completion records (DB §8.2)
// ---------------------------------------------------------------------------
static void rec_push(tsr_dbg * d, uint32_t d0, uint32_t d1, uint32_t d2, uint32_t d3) {
    if (d->rec_n >= TSR_REC_DEPTH) {                     // never stalls: dropped and counted
        tsr_sim_perf_add(d->sim, TSR_PERF_REC_DROPPED(d->num_ch), 1);
        d->sticky |= TSR_DBG_STATUS_REC_DROP;
        return;
    }
    uint32_t * r = d->rec[(d->rec_rd + d->rec_n) % TSR_REC_DEPTH];
    r[0] = d0; r[1] = d1; r[2] = d2; r[3] = d3;
    d->rec_n++;
}

#if DBG_HAVE_RV
static void rec_clear(tsr_dbg * d) {
    d->rec_rd = d->rec_n = 0;
    d->rec_cfg = TSR_CW_REC_CFG_RESET;
}
#endif

// ---------------------------------------------------------------------------
// Trace writer (DB §8.3)
// ---------------------------------------------------------------------------
static uint64_t trw_cap(const tsr_dbg * d) { return (uint64_t)d->trw_lines * 2u; }   // records

static void trw_drop(tsr_dbg * d, uint32_t n) {
    d->trw_drops = (uint32_t)((uint64_t)d->trw_drops + n > 0xFFFFFFFFull ? 0xFFFFFFFFu : d->trw_drops + n);
    tsr_sim_perf_add(d->sim, TSR_PERF_TRW_DROPS, n);
    d->sticky |= TSR_DBG_STATUS_TRW_DROP;
    d->trw_drop_pend += n;
}

// One record into the ring at WPTR; 0 if it could not be written (ring full with TRW_STOP_FULL,
// no ring, or a ring outside the channel).
static int trw_put(tsr_dbg * d, const uint32_t w[8]) {
    const uint64_t cap = trw_cap(d);
    if (cap == 0) return 0;
    if ((d->cfg & TSR_DBG_CFG_TRW_STOP_FULL) && (uint64_t)(uint32_t)(d->trw_wptr - d->trw_rptr) >= cap) return 0;
    const uint32_t ch = TSR_GET(d->trw_base, TSR_HANDLE_CH);
    const uint64_t beat = TSR_GET(d->trw_base, TSR_HANDLE_BEAT) & ~0xFull;
    const uint64_t off = beat * TSR_LINE_BYTES + (uint64_t)(d->trw_wptr % cap) * TSR_TRW_RECORD_BYTES;
    if (ch >= d->num_ch || off + TSR_TRW_RECORD_BYTES > d->ch_bytes) return 0;
    uint8_t * p = tsr_sim_dram(d->sim, ch) + off;
    for (uint32_t i = 0; i < 8; ++i) {
        p[4 * i] = (uint8_t)w[i]; p[4 * i + 1] = (uint8_t)(w[i] >> 8);
        p[4 * i + 2] = (uint8_t)(w[i] >> 16); p[4 * i + 3] = (uint8_t)(w[i] >> 24);
    }
    d->trw_wptr++;
    return 1;
}

static void trw_flush(tsr_dbg * d) {
    for (uint32_t i = 0; i < d->trw_ns; ++i) {
        const uint32_t * w = d->trw_st[i];
        const int drop_rec = TSR_GET(w[0], TSR_TRW_W0_KIND) == TSR_TRW_W0_KIND_DROP;
        if (trw_put(d, w)) {
            tsr_sim_perf_add(d->sim, TSR_PERF_TRW_RECORDS, 1);
        } else if (drop_rec) {
            d->trw_drop_pend += w[1];                    // the count goes into the next DROP record
        } else {
            trw_drop(d, 1);
        }
    }
    d->trw_ns = 0;
}

static void trw_idle_check(tsr_dbg * d) {
    if (d->trw_ns && now_of(d) - d->trw_t_last >= TRW_IDLE) trw_flush(d);
}

static void trw_stage(tsr_dbg * d, const uint32_t w[8]) {
    trw_idle_check(d);
    const uint64_t t = now_of(d);
    if (d->trw_drop_pend) {                              // DROP: records dropped since the previous one
        uint32_t * r = d->trw_st[d->trw_ns++];
        memset(r, 0, 32);
        r[0] = TSR_PUT(TSR_TRW_W0_KIND, TSR_TRW_W0_KIND_DROP);
        r[1] = d->trw_drop_pend;
        r[3] = (uint32_t)t; r[4] = (uint32_t)(t >> 32);
        d->trw_drop_pend = 0;
        if (d->trw_ns == TRW_STAGE) trw_flush(d);
    }
    memcpy(d->trw_st[d->trw_ns++], w, 32);
    d->trw_t_last = t;
    if (d->trw_ns == TRW_STAGE) trw_flush(d);            // a 16-beat burst
}

static int trw_on(const tsr_dbg * d, uint32_t mask_bit) {
    return (d->cfg & TSR_DBG_CFG_TRW_EN) && (d->trw_mask & mask_bit);
}

static void trw_stop_rec(tsr_dbg * d, uint32_t T, uint32_t cause, int activated) {
    if (!trw_on(d, TSR_DBG_TRW_MASK_STOPS)) return;
    const uint64_t t = now_of(d);
    uint32_t w[8] = { TSR_PUT(TSR_TRW_W0_KIND, TSR_TRW_W0_KIND_STOP), T, cause, (uint32_t)t, (uint32_t)(t >> 32),
                      activated ? 1u : 0u, 0, 0 };
    trw_stage(d, w);
}

static void trw_rvhalt_rec(tsr_dbg * d, int halted) {
    if (!trw_on(d, TSR_DBG_TRW_MASK_RVHALTS)) return;
    const uint64_t t = now_of(d);
    uint32_t w[8] = { TSR_PUT(TSR_TRW_W0_KIND, TSR_TRW_W0_KIND_RVHALT), 0, halted ? 1u : 0u, (uint32_t)t,
                      (uint32_t)(t >> 32), 0, 0, 0 };
    trw_stage(d, w);
}

// ---------------------------------------------------------------------------
// Stop events
// ---------------------------------------------------------------------------
static void activation(tsr_dbg * d, int tbp_halt) {
    uint32_t T = 0;
    if (!stop_T(d, &T)) return;
    d->last_T = T;
    trw_stop_rec(d, T, stop_cause(d), 1);
    if (tbp_halt || (d->cfg & TSR_DBG_CFG_STOP_HALT_RV)) halt_request(d);
    freeze_update(d, core_status(d));
}

// STOP_NOW / ERR / HUNG: S <- DBG_NEXT_PIDX, an earlier valid S kept.
static void stop_now(tsr_dbg * d, uint32_t cause) {
    const uint32_t p = d->next_pidx;
    if (d->s_valid && ge(p, d->S)) {
        d->s_cause |= cause;
    } else {
        d->S = p;
        d->s_valid = 1;
        d->s_cause = cause;
    }
    activation(d, 0);
}

static void step_cancel(tsr_dbg * d) {
    d->step_phase = STEP_IDLE;
    d->step_busy = 0;
    d->step_arm = 0;
}

static void release(tsr_dbg * d) {
    uint32_t T = 0;
    step_cancel(d);
    if (!stop_T(d, &T)) {
        d->s_valid = 0;
        return;
    }
    const uint32_t cause = stop_cause(d);
    d->s_valid = 0;
    mq_pop_upto(d, T);
    trw_stop_rec(d, T, cause, 0);
    if (stop_T(d, &T)) {                                 // the next M entry becomes T
        d->last_T = T;
        trw_stop_rec(d, T, stop_cause(d), 1);
    }
    freeze_update(d, core_status(d));
}

// QSTEP with STOP_ACTIVE (DB §5.2): once DRAINED, X <- T, S <- X + 1, M entries <= X popped;
// dbg_step armed for X; repeated DBG_STEP_N times.
static void step_advance(tsr_dbg * d) {
    while (d->step_phase == STEP_DRAIN) {
        uint32_t T = 0;
        if (!stop_T(d, &T)) { step_cancel(d); return; }
        if (!drained(d)) return;
        d->S = T + 1u;
        d->s_valid = 1;
        d->s_cause = TSR_DBG_CAUSE_STEP;
        mq_pop_upto(d, T);
        d->step_x = T;
        d->step_phase = STEP_DONE;
        d->step_arm = 1;
        d->step_arm_pidx = T;
        uint32_t T2 = 0;
        if (stop_T(d, &T2)) {
            d->last_T = T2;
            trw_stop_rec(d, T2, stop_cause(d), 1);
        }
    }
}

static void qstep(tsr_dbg * d) {
    if (stop_T(d, NULL)) {
        d->step_left = TSR_GET(d->step_n, TSR_DBG_STEP_N_N);
        if (d->step_left == 0) d->step_left = 1;
        d->step_phase = STEP_DRAIN;
        d->step_busy = 1;
        step_advance(d);
        return;
    }
    if (TSR_GET(core_status(d), TSR_STATUS_HOLD_SRC) & TSR_HOLD_SRC_HALT) {
        // Hart halted with HOLD_ON_HALT: X <- the smallest head push index; one entry.
        int have = 0;
        uint32_t x = 0, best_age = 0;
        for (uint32_t u = 0; u < TSR_NUM_UNITS; ++u) {
            const shent * h = sh_head(&d->q[u]);
            if (h == NULL) continue;
            const uint32_t age = d->next_pidx - h->pidx;
            if (!have || age > best_age) { x = h->pidx; best_age = age; have = 1; }
        }
        if (!have) return;                               // nothing queued: nothing to step
        d->step_x = x;
        d->step_left = 1;
        d->step_phase = STEP_DONE;
        d->step_busy = 1;
        d->step_arm = 1;
        d->step_arm_pidx = x;
        return;
    }
    stop_now(d, TSR_DBG_CAUSE_NOW);                      // QSTEP otherwise acts as STOP_NOW
}

// ---------------------------------------------------------------------------
// RISC-V state and polling
// ---------------------------------------------------------------------------
// STATUS.RV_HALTED edges (the time base's copy, reported by the RISC-V agent: trace-writer RVHALT
// records), the one-shot rule as a safety net (the hart's ENTER event clears pend at once), the
// request level and dbg_stop.
static void rv_refresh_st(tsr_dbg * d, uint32_t st) {
    const int halted = tsr_sim_rv_run(d->sim) && (st & TSR_STATUS_RV_HALTED) && (st & TSR_STATUS_RV_RUNNING);
    if (halted != d->rv_halted) {
        d->rv_halted = halted;
        trw_rvhalt_rec(d, halted);
    }
    if (d->pend && hart_halted_now(d)) d->pend = 0;      // cleared by RV_HALTED (one request, one halt)
    req_drive(d);
    freeze_update(d, st);
}
static void rv_refresh(tsr_dbg * d) { rv_refresh_st(d, core_status(d)); }

// After every queue event and DBG write: the RISC-V view, the idle flush, the watchdog edge, a
// QSTEP waiting for DRAINED, and DRAINED's rising edge (tsr_sim_advance returns there).
static void poll(tsr_dbg * d) {
    const uint32_t st = core_status(d);
    rv_refresh_st(d, st);
    trw_idle_check(d);
    const int hung = (st & TSR_STATUS_UNIT_HUNG) != 0;
    if (hung && !d->hung_prev) {                         // EX §8.6 watchdog: history freeze, HUNG_STOP
        d->hist_hung = 1;
        if (d->cfg & TSR_DBG_CFG_HUNG_STOP) stop_now(d, TSR_DBG_CAUSE_HUNG);
    }
    d->hung_prev = hung;
    step_advance(d);
    const int dr = drained(d);
    if (dr && !d->drained_prev) tsr_sim_request_stop(d->sim);
    d->drained_prev = dr;
    freeze_update(d, st);
}

// ---------------------------------------------------------------------------
// Queue events: enqueue tap (TBPs, history), start / completion strobes (REC, TRW), markers
// ---------------------------------------------------------------------------
static int ext_overlap(const tsr_extent * e, uint64_t a, uint64_t b) { return e->lo < b && a < e->hi; }

// DB §5.3 match of TBP i against an entry at its enqueue.
static int tbp_match(const tsr_dbg * d, uint32_t i, const tsr_q_event * ev) {
    const uint32_t ctl = d->tbp_ctl[i], A = d->tbp_a[i], B = d->tbp_b[i];
    if (!(ctl & TSR_DBG_TBP_CTL_EN)) return 0;
    const uint32_t w0 = ev->w ? ev->w[0] : 0u;
    switch (TSR_GET(ctl, TSR_DBG_TBP_CTL_KIND)) {
    case TSR_DBG_TBP_CTL_KIND_PIDX:
        return (uint32_t)(ev->pidx - A) < (uint32_t)(B - A);
    case TSR_DBG_TBP_CTL_KIND_OPCODE: {
        const uint32_t units = TSR_GET(B, TSR_TBP_OPC_B_UNITS);
        return (((w0 ^ A) & TSR_GET(A, TSR_TBP_OPC_A_OPC_MASK)) & 0xFFu) == 0 &&
               ((TSR_GET(w0, TSR_W0_SUB) ^ TSR_GET(A, TSR_TBP_OPC_A_SUB)) & TSR_GET(A, TSR_TBP_OPC_A_SUB_MASK)) == 0 &&
               (units == 0 || ((units >> (ev->ticket >> 28)) & 1u));
    }
    case TSR_DBG_TBP_CTL_KIND_SPM:
    case TSR_DBG_TBP_CTL_KIND_DRAM: {
        if (ev->code != 0 || ev->ext == NULL) return 0;  // never a poisoned entry
        const int spm = TSR_GET(ctl, TSR_DBG_TBP_CTL_KIND) == TSR_DBG_TBP_CTL_KIND_SPM;
        const uint32_t chm = TSR_GET(ctl, TSR_DBG_TBP_CTL_CHMASK);
        for (uint32_t k = 0; k < ev->ext->n; ++k) {
            const tsr_extent * e = &ev->ext->e[k];
            if (e->kind != (spm ? TSR_EXT_SPM : TSR_EXT_DRAM)) continue;
            if (!(e->write ? (ctl & TSR_DBG_TBP_CTL_W) : (ctl & TSR_DBG_TBP_CTL_R))) continue;
            if (!spm) {
                const uint32_t chs = e->ch == TSR_EXT_CH_ALL ? ((1u << d->num_ch) - 1u) : (1u << e->ch);
                if (!(chs & chm)) continue;
            }
            if (ext_overlap(e, A, B)) return 1;
        }
        return 0;
    }
    case TSR_DBG_TBP_CTL_KIND_TICKET:
        return ev->ticket == A;
    default:
        return 0;                                        // kinds 5..7 never match
    }
}

static const ppc_ent * ppc_find(const tsr_dbg * d, uint32_t pidx) {
    const ppc_ent * p = &d->ppc[pidx % PPC_N];
    return (p->valid && p->pidx == pidx) ? p : NULL;
}

// The deps of a push as pushed (before the dispatcher's BAD_DEP rule), kept until its enqueue.
static void ppc_store(tsr_dbg * d, uint32_t pidx, const uint32_t * dep) {
    ppc_ent * p = &d->ppc[pidx % PPC_N];
    p->pidx = pidx;
    p->dep[0] = dep ? dep[0] : TSR_TICKET_NONE;
    p->dep[1] = dep ? dep[1] : TSR_TICKET_NONE;
    p->valid = 1;
}

static int hist_recording(const tsr_dbg * d, uint32_t pidx) {
    if (!(d->cfg & TSR_DBG_CFG_HIST_EN) || (d->cfg & TSR_DBG_CFG_HIST_FREEZE) || d->hist_hung) return 0;
    if (d->hist_err && (d->cfg & TSR_DBG_CFG_HIST_FREEZE_ERR) && ge(pidx, d->hist_err_pidx + HIST_HALF)) return 0;
    return 1;
}

static int hist_frozen(const tsr_dbg * d) {
    return (d->cfg & TSR_DBG_CFG_HIST_FREEZE) || d->hist_hung ||
           (d->hist_err && (d->cfg & TSR_DBG_CFG_HIST_FREEZE_ERR) && ge(d->hist_head, d->hist_err_pidx + HIST_HALF));
}

static void on_enq(tsr_dbg * d, const tsr_q_event * ev) {
    const uint32_t u = ev->unit;
    if (u >= TSR_NUM_UNITS) return;
    shq * q = &d->q[u];
    if (q->n >= SHQ_CAP) {
        fprintf(stderr, "tsr_dbg: shadow queue of unit %u overflowed\n", u);
        abort();
    }
    shent * e = sh_at(q, q->n);
    memset(e, 0, sizeof(*e));
    e->pidx = ev->pidx;
    e->ticket = ev->ticket;
    e->w0 = ev->w ? ev->w[0] : 0u;
    e->t_disp = ev->time;
    e->rec = (uint8_t)(ev->rec != 0);
    if (q->n == 0 || sh_at(q, q->n - 1)->started) {      // every earlier entry left the queue
        e->t_head = ev->time;
        e->has_head = 1;
    }
    if (q->n == 0) q->t_oldest = ev->time;
    q->n++;
    d->next_pidx = ev->pidx + 1u;

    // Tile breakpoints (DB §5.3), registered with the queue write: the threshold holds the entry
    // before its first head evaluation.
    uint32_t hit = 0, before = 0, after = 0;
    int halt = 0;
    for (uint32_t i = 0; i < TSR_DBG_NTBP; ++i) {
        if (!tbp_match(d, i, ev)) continue;
        hit |= 1u << i;
        d->tbp_hits[i] = sat_inc(d->tbp_hits[i]);
        if (d->tbp_ctl[i] & TSR_DBG_TBP_CTL_COUNT_ONLY) continue;
        if (d->tbp_ctl[i] & TSR_DBG_TBP_CTL_AFTER) after |= 1u << i;
        else before |= 1u << i;
        if (d->tbp_ctl[i] & TSR_DBG_TBP_CTL_HALT_RV) halt = 1;
    }
    e->tbp = (uint8_t)(hit != 0);
    if (before) mq_push(d, ev->pidx, before);
    if (after) mq_push(d, ev->pidx + 1u, after);
    if (before | after) activation(d, halt);

    // Dispatch history (DB §5.6)
    if (hist_recording(d, ev->pidx)) {
        hist_ent * h = &d->hist[ev->pidx % TSR_HIST_DEPTH];
        const ppc_ent * p = ppc_find(d, ev->pidx);
        uint32_t T = 0;
        const int stop_pt = stop_T(d, &T) && T == ev->pidx;
        h->pidx = ev->pidx;
        h->ticket = ev->ticket;
        h->dep[0] = p ? p->dep[0] : ev->dep[0];
        h->dep[1] = p ? p->dep[1] : ev->dep[1];
        if (ev->w) memcpy(h->w, ev->w, sizeof(h->w));
        else memset(h->w, 0, sizeof(h->w));
        h->pc = ev->pc;                                  // push PC (CC §7.1; TSR_SIM_PC_HOOK for the test hooks)
        h->time = (uint32_t)ev->time;
        h->info = TSR_DBG_HIST_INFO_PRESENT | TSR_PUT(TSR_DBG_HIST_INFO_UNIT, u) |
                  TSR_PUT(TSR_DBG_HIST_INFO_CODE, ev->code) | (ev->rec ? TSR_DBG_HIST_INFO_REC : 0u) |
                  TSR_PUT(TSR_DBG_HIST_INFO_TBPS, hit) | (stop_pt ? TSR_DBG_HIST_INFO_STOP : 0u);
        h->valid = 1;
        d->hist_head = ev->pidx + 1u;
    }
}

static void on_start(tsr_dbg * d, const tsr_q_event * ev) {
    const uint32_t u = ev->unit;
    if (u >= TSR_NUM_UNITS || !ge(ev->pidx, d->first_pidx)) return;
    shq * q = &d->q[u];
    for (uint32_t i = 0; i < q->n; ++i) {
        shent * e = sh_at(q, i);
        if (e->started) continue;
        if (e->ticket != ev->ticket) break;              // not the head: inconsistent, ignore
        e->started = 1;
        e->t_start = ev->time;
        if (!e->has_head) { e->t_head = ev->time; e->has_head = 1; }
        sh_next_head(q, i, ev->time);                    // the next entry is loaded as head
        break;
    }
    if (d->step_arm && ev->pidx == d->step_arm_pidx) d->step_arm = 0;   // the dbg_step pulse was taken
}

static void on_done(tsr_dbg * d, const tsr_q_event * ev) {
    const uint32_t u = ev->unit;
    if (u >= TSR_NUM_UNITS || !ge(ev->pidx, d->first_pidx)) return;   // enqueued before the attach
    shq * q = &d->q[u];
    if (q->n == 0 || sh_at(q, 0)->ticket != ev->ticket) {
        fprintf(stderr, "tsr_dbg: completion of 0x%08x is not the oldest entry of unit %u\n", ev->ticket, u);
        abort();
    }
    shent e = *sh_at(q, 0);
    const uint64_t t = ev->time;
    if (!ev->ran) sh_next_head(q, 0, t);                 // a non-running completion leaves now
    q->rd = (q->rd + 1) % SHQ_CAP;
    q->n--;
    q->t_oldest = t;
    if (!e.has_head) e.t_head = t;
    const uint32_t opcode = e.w0 & 0xFFu;

    // Trace writer OP record
    if (trw_on(d, TSR_PUT(TSR_DBG_TRW_MASK_UNITS, 1u << u))) {
        uint32_t w[8];
        w[0] = TSR_PUT(TSR_TRW_W0_KIND, TSR_TRW_W0_KIND_OP) | TSR_PUT(TSR_TRW_W0_UNIT, u) |
               TSR_PUT(TSR_TRW_W0_CODE, ev->code) | (ev->ran ? TSR_TRW_W0_RAN : 0u) | (e.rec ? TSR_TRW_W0_REC : 0u) |
               (e.tbp ? TSR_TRW_W0_TBP : 0u) | TSR_PUT(TSR_TRW_W0_OPCODE, opcode) |
               TSR_PUT(TSR_TRW_W0_W0_HI, (e.w0 >> 8) & 0xFFu);
        w[1] = e.pidx;
        w[2] = e.ticket;
        w[3] = (uint32_t)t;
        w[4] = (uint32_t)(t >> 32);
        w[5] = ev->ran ? sat32(t - e.t_start) : 0u;
        w[6] = sat32(e.t_head - e.t_disp);
        w[7] = ev->ran ? sat32(e.t_start - e.t_head) : sat32(t - e.t_head);
        trw_stage(d, w);
    }
    // Completion record to the RISC-V
    if (e.rec || (d->rec_cfg & TSR_REC_CFG_ALL))
        rec_push(d, e.ticket, (uint32_t)t, ev->ran ? sat32(t - e.t_start) : 0u,
                 TSR_PUT(TSR_REC_DATA3_CODE, ev->code) | (ev->ran ? TSR_REC_DATA3_RAN : 0u) |
                     TSR_PUT(TSR_REC_DATA3_OPCODE, opcode) | TSR_PUT(TSR_REC_DATA3_PIDX, e.pidx & 0xFFFFu));
    if (ev->code != 0) {
        if (!d->hist_err) {                              // HIST_FREEZE_ERR counts from here
            d->hist_err = 1;
            d->hist_err_pidx = e.pidx;
        }
        if ((d->cfg & TSR_DBG_CFG_ERR_STOP) &&
            tsr_sim_corewin_read(d->sim, TSR_CSR_Q_ERR_SEQ(u)) == (e.ticket & 0x0FFFFFFFu) &&
            tsr_sim_corewin_read(d->sim, TSR_CSR_Q_ERR_CODE(u)) == ev->code)
            stop_now(d, TSR_DBG_CAUSE_ERR);              // this completion put the unit in error state
    }
    if (d->step_arm && e.pidx == d->step_arm_pidx) d->step_arm = 0;
    if (d->step_phase == STEP_DONE && e.pidx == d->step_x) {
        if (--d->step_left) {
            d->step_phase = STEP_DRAIN;
        } else {
            step_cancel(d);
            tsr_sim_request_stop(d->sim);
        }
    }
}

static void on_mark(tsr_dbg * d, const tsr_q_event * ev) {
    const uint32_t value = ev->w ? ev->w[1] : 0u;
    const uint64_t t = ev->time;
    if (trw_on(d, TSR_DBG_TRW_MASK_MARKERS)) {
        uint32_t w[8] = { TSR_PUT(TSR_TRW_W0_KIND, TSR_TRW_W0_KIND_MARK), ev->pidx, value, (uint32_t)t,
                          (uint32_t)(t >> 32), 0, 0, 0 };
        trw_stage(d, w);
        if (TSR_GET(value, TSR_MARK_CLASS) == TSR_MARK_CLASS_LAUNCH_END) trw_flush(d);   // partial burst
    }
    rec_push(d, TSR_DBG_REC_MARK_TICKET, (uint32_t)t, value,
             TSR_REC_DATA3_MARKER | TSR_PUT(TSR_REC_DATA3_PIDX, ev->pidx & 0xFFFFu));
}

static void on_qev(void * ctx, tsr_sim * sim, const tsr_q_event * ev) {
    tsr_dbg * d = (tsr_dbg *)ctx;
    (void)sim;
    switch (ev->kind) {
    case TSR_QEV_PUSH:  ppc_store(d, ev->pidx, ev->dep); break;
    case TSR_QEV_QCTRL:
        if (ev->ticket != TSR_TICKET_NONE) ppc_store(d, ev->pidx, NULL);   // QCLEAR: a ticketed entry
        break;
    case TSR_QEV_MARK_PUSH: break;
    case TSR_QEV_ENQ:   on_enq(d, ev); break;
    case TSR_QEV_START: on_start(d, ev); break;
    case TSR_QEV_DONE:  on_done(d, ev); break;
    case TSR_QEV_MARK:  on_mark(d, ev); break;
    default: break;
    }
    poll(d);
}

#if DBG_HAVE_RV
// The hart's debug events (from the DM model's event hook, or from the hart itself without the DM):
// ENTER = RV_HALTED rises, which clears pend at once (one request, one halt: a resume never meets
// a stale request); RESET = the hart leaves reset: a request from before the reset is dropped and
// HALT_ON_BOOT makes one pending, so the hart halts before its first instruction (DB §3.8). The REC
// FIFO and REC_CFG are clk_rv state and restart empty with the hart.
static void on_dbg_ev(void * ctx, uint32_t ev, uint32_t arg) {
    tsr_dbg * d = (tsr_dbg *)ctx;
    (void)arg;
    if (ev == TSR_RV_DBG_EV_ENTER) {
        d->dbg_entries++;
        d->pend = 0;
        req_drive(d);
    } else if (ev == TSR_RV_DBG_EV_RESET) {
        d->pend = (d->cfg & TSR_DBG_CFG_HALT_ON_BOOT) && (d->cfg & TSR_DBG_CFG_RV_HALT_EN);
        rec_clear(d);
        req_drive(d);
    }
}

// RV watchpoints (DB §3.7) on the hart's data requests (SBA is not watched).
static void on_data(void * ctx, uint32_t addr, uint32_t size, int is_store, uint32_t pc) {
    tsr_dbg * d = (tsr_dbg *)ctx;
    (void)pc;
    d->last_daddr = addr;                                // crash_dump_o.last_data_addr (RV_SAMPLE)
    tsr_rv * rv = hart(d);
    for (uint32_t j = 0; j < TSR_DBG_NRVWP; ++j) {
        const uint32_t c = d->wpa_ctl[j];
        if (!(c & TSR_DBG_RVWP_CTL_EN)) continue;
        if (!(is_store ? (c & TSR_DBG_RVWP_CTL_STORE) : (c & TSR_DBG_RVWP_CTL_LOAD))) continue;
        if (!((uint64_t)addr < d->wpa_hi[j] && (uint64_t)addr + size > d->wpa_lo[j])) continue;
        d->wp_hits[j] = sat_inc(d->wp_hits[j]);
        d->wp_addr = addr;
        d->sticky |= TSR_PUT(TSR_DBG_STATUS_RVWP_HIT, 1u << j);
        if ((c & TSR_DBG_RVWP_CTL_HALT) && !(rv && tsr_rv_debug_mode(rv))) halt_request(d);
    }
}
#endif

// ---------------------------------------------------------------------------
// Register reads (DB §10.1, §10.2)
// ---------------------------------------------------------------------------
static uint32_t probe_state(tsr_dbg * d, uint32_t u, uint32_t hm, uint32_t hold_src) {
    const shq * q = &d->q[u];
    uint32_t acc = 0;
    for (uint32_t i = 0; i < q->n; ++i) acc += sh_cat(q, i)->started;
    const uint32_t qs = tsr_sim_corewin_read(d->sim, TSR_CSR_Q_STATE(u));
    uint32_t hs = TSR_DBG_U_STATE_HEAD_EMPTY;
    if (TSR_GET(qs, TSR_Q_STATE_ENTRIES) != 0) {
        if (qs & TSR_Q_STATE_HELD) hs = TSR_DBG_U_STATE_HEAD_HELD;
        else if (qs & TSR_Q_STATE_DEP_WAIT) hs = TSR_DBG_U_STATE_HEAD_WAIT_DEP;
        else if (qs & TSR_Q_STATE_RUNNING) hs = TSR_DBG_U_STATE_HEAD_WAIT_UNIT;
        else hs = TSR_DBG_U_STATE_HEAD_LOAD;
    }
    uint32_t hold = 0;
    if (sh_head(q) != NULL) hold = hold_src | (((hm >> u) & 1u) ? TSR_HOLD_SRC_DB : 0u);
    return TSR_PUT(TSR_DBG_U_STATE_ACCEPTED, acc > 3 ? 3 : acc) | TSR_PUT(TSR_DBG_U_STATE_HEAD, hs) |
           TSR_PUT(TSR_DBG_U_STATE_HOLD, hold) | TSR_PUT(TSR_DBG_U_STATE_OUTSTANDING, q->n > 255 ? 255 : q->n);
}

// dbg_hold as it applies: forced to 0 while SR_BUSY (EX §8.5 step 1).
static uint32_t hold_mask_eff(tsr_dbg * d) {
    if (!stop_T(d, NULL) && !(d->cfg & TSR_DBG_CFG_SERIAL)) return 0;
    if (core_status(d) & TSR_STATUS_SR_BUSY) return 0;
    return hold_mask(d);
}

static uint32_t status_read(tsr_dbg * d) {
    uint32_t v = d->sticky & STICKY_MASK;
    const int active = stop_T(d, NULL);
    if (active) v |= TSR_DBG_STATUS_STOP_ACTIVE | TSR_PUT(TSR_DBG_STATUS_CAUSE, stop_cause(d));
    if (drained(d)) v |= TSR_DBG_STATUS_DRAINED;
    if (d->step_busy) v |= TSR_DBG_STATUS_STEP_BUSY;
    if (hold_mask_eff(d)) v |= TSR_DBG_STATUS_DB_HELD;
    v |= TSR_PUT(TSR_DBG_STATUS_BP_HIT, bp_hit(d));
    if (d->pend) v |= TSR_DBG_STATUS_RV_HALT_PEND;
    if (hist_frozen(d)) v |= TSR_DBG_STATUS_HIST_FROZEN;
    const uint32_t st = core_status(d);
    if (tsr_sim_rv_run(d->sim)) {
        if (st & TSR_STATUS_RV_HALTED) v |= TSR_DBG_STATUS_RV_HALTED;
        if (st & TSR_STATUS_RV_SLEEP) v |= TSR_DBG_STATUS_RV_SLEEP;
    }
    if (d->s_valid) v |= TSR_DBG_STATUS_S_VALID;
    return v;
}

static uint32_t rv_state_read(tsr_dbg * d) {
    const uint32_t st = core_status(d);
    const int run = tsr_sim_rv_run(d->sim) != 0;
    uint32_t v = 0;
    if (run && (st & TSR_STATUS_RV_HALTED)) v |= TSR_DBG_RV_STATE_HALTED;
    if (run && (st & TSR_STATUS_RV_SLEEP)) v |= TSR_DBG_RV_STATE_SLEEP;
    if (!run) v |= TSR_DBG_RV_STATE_IN_RESET;
    if (d->smp_valid) v |= TSR_DBG_RV_STATE_SAMPLE_VALID;
#if DBG_HAVE_DM
    if (dmod(d)) {                                       // dmactive_o / ndmreset_o (dmcontrol bits 0 / 1)
        const uint32_t dc = tsr_dm_dmi_read(dmod(d), TSR_DMI_DMCONTROL);
        if (dc & 1u) v |= TSR_DBG_RV_STATE_DMACTIVE;
        if (dc & 2u) v |= TSR_DBG_RV_STATE_NDMRESET;
    }
#endif
    if (d->req_level) v |= TSR_DBG_RV_STATE_HALT_REQ;
    v |= TSR_PUT(TSR_DBG_RV_STATE_SAMPLE_SEQ, d->smp_seq) | TSR_PUT(TSR_DBG_RV_STATE_DEBUG_ENTRIES, d->dbg_entries);
    return v;
}

static uint32_t reg_read(tsr_dbg * d, uint32_t off) {
    uint32_t i;
    if (off >= TSR_CSR_DBG_RVWP_CTL(0) && off < TSR_CSR_DBG_RVWP_CTL(TSR_DBG_NRVWP)) {
        i = (off - TSR_CSR_DBG_RVWP_CTL(0)) / 16;
        switch ((off - TSR_CSR_DBG_RVWP_CTL(0)) % 16) {
        case 0:  return d->wp_ctl[i];
        case 4:  return d->wp_lo[i];
        case 8:  return d->wp_hi[i];
        default: return d->wp_hits[i];
        }
    }
    if (off >= TSR_CSR_DBG_TBP_CTL(0) && off < TSR_CSR_DBG_TBP_CTL(TSR_DBG_NTBP)) {
        i = (off - TSR_CSR_DBG_TBP_CTL(0)) / 16;
        switch ((off - TSR_CSR_DBG_TBP_CTL(0)) % 16) {
        case 0:  return d->tbp_ctl[i];
        case 4:  return d->tbp_a[i];
        case 8:  return d->tbp_b[i];
        default: return d->tbp_hits[i];
        }
    }
    if (off >= TSR_CSR_DBG_RDOUT(0) && off < TSR_CSR_DBG_WROUT(TSR_CSR_DBG_WROUT_N)) return 0;   // no arbiter model
    if (off >= TSR_CSR_DBG_U_STATE(0) && off < TSR_CSR_DBG_U_STATE(TSR_CSR_DBG_U_STATE_N)) {
        const uint32_t u = (off - TSR_CSR_DBG_U_STATE(0)) / 16;
        if (u >= TSR_NUM_UNITS) return (off - TSR_CSR_DBG_U_STATE(0)) % 16 == 4 ? TSR_TICKET_NONE : 0u;
        const shq * q = &d->q[u];
        switch ((off - TSR_CSR_DBG_U_STATE(0)) % 16) {
        case 0:  return probe_state(d, u, hold_mask_eff(d), TSR_GET(core_status(d), TSR_STATUS_HOLD_SRC));
        case 4:  return q->n ? sh_cat(q, 0)->ticket : TSR_TICKET_NONE;
        case 8:  return q->n ? sh_cat(q, 0)->pidx : 0u;
        default: return q->n ? sat32(now_of(d) - q->t_oldest) : 0u;
        }
    }
    if (off >= TSR_CSR_DBG_HIST_W(0) && off < TSR_CSR_DBG_HIST_W(8))
        return d->hist[d->hist_idx % TSR_HIST_DEPTH].w[(off - TSR_CSR_DBG_HIST_W(0)) / 4];
    const hist_ent * h = &d->hist[d->hist_idx % TSR_HIST_DEPTH];
    const uint64_t now = now_of(d);
    uint32_t T = 0;
    switch (off) {
    case TSR_CSR_DBG_ID:
        return TSR_PUT(TSR_DBG_ID_VERSION, TSR_DBG_VERSION) | TSR_DBG_ID_HIST | TSR_DBG_ID_TRW | TSR_DBG_ID_RVWP |
               TSR_DBG_ID_REC | TSR_PUT(TSR_DBG_ID_NTBP, TSR_DBG_NTBP) | TSR_PUT(TSR_DBG_ID_NRVWP, TSR_DBG_NRVWP) |
               TSR_PUT(TSR_DBG_ID_LOG2_HIST, 9u);
    case TSR_CSR_DBG_CTRL:      return 0;
    case TSR_CSR_DBG_CFG:       return d->cfg;
    case TSR_CSR_DBG_STATUS:    return status_read(d);
    case TSR_CSR_DBG_STOP_PIDX: return stop_T(d, &T) ? T : d->last_T;
    case TSR_CSR_DBG_STOP_AT:   return d->S;
    case TSR_CSR_DBG_STEP_N:    return d->step_n;
    case TSR_CSR_DBG_NEXT_PIDX: return d->next_pidx;
    case TSR_CSR_DBG_STEP_PIDX: return d->step_x;
    case TSR_CSR_DBG_MATCHQ:    return TSR_PUT(TSR_DBG_MATCHQ_PENDING, d->mq_n);
    case TSR_CSR_DBG_MTIME_LO:  return (uint32_t)now;
    case TSR_CSR_DBG_MTIME_HI:  return (uint32_t)(now >> 32);
    case TSR_CSR_DBG_QSTATE: {
        uint32_t n = 0;
        for (uint32_t u = 0; u < TSR_NUM_UNITS; ++u) n += d->q[u].n;
        const uint32_t disp = tsr_sim_corewin_read(d->sim, TSR_CSR_Q_FIFO) ? TSR_DBG_QSTATE_DISP_POP : TSR_DBG_QSTATE_DISP_IDLE;
        return TSR_PUT(TSR_DBG_QSTATE_DISP, disp) | TSR_PUT(TSR_DBG_QSTATE_OUTSTANDING, n > 255 ? 255 : n);
    }
    case TSR_CSR_DBG_RV_STATE:  return rv_state_read(d);
    case TSR_CSR_DBG_RV_PC:     return d->smp[0];
    case TSR_CSR_DBG_RV_NPC:    return d->smp[1];
    case TSR_CSR_DBG_RV_DADDR:  return d->smp[2];
    case TSR_CSR_DBG_RV_EPC:    return d->smp[3];
    case TSR_CSR_DBG_RV_EADDR:  return d->smp[4];
    case TSR_CSR_DBG_RV_WPADDR: return d->wp_addr;
    case TSR_CSR_DBG_TRW_BASE:  return d->trw_base;
    case TSR_CSR_DBG_TRW_LINES: return d->trw_lines;
    case TSR_CSR_DBG_TRW_WPTR:  return d->trw_wptr;
    case TSR_CSR_DBG_TRW_RPTR:  return d->trw_rptr;
    case TSR_CSR_DBG_TRW_DROPS: return d->trw_drops;
    case TSR_CSR_DBG_TRW_MASK:  return d->trw_mask;
    case TSR_CSR_DBG_HIST_HEAD: return d->hist_head;
    case TSR_CSR_DBG_HIST_IDX:  return d->hist_idx;
    case TSR_CSR_DBG_HIST_INFO:
        return (h->valid && h->pidx == d->hist_idx) ? h->info : (h->info & ~TSR_DBG_HIST_INFO_PRESENT);
    case TSR_CSR_DBG_HIST_PIDX:   return h->pidx;
    case TSR_CSR_DBG_HIST_TICKET: return h->ticket;
    case TSR_CSR_DBG_HIST_DEP0:   return h->dep[0];
    case TSR_CSR_DBG_HIST_DEP1:   return h->dep[1];
    case TSR_CSR_DBG_HIST_TIME:   return h->time;
    case TSR_CSR_DBG_HIST_PC:     return h->pc;
    default: return 0;                                   // unlisted offsets read 0
    }
}

// ---------------------------------------------------------------------------
// Register writes
// ---------------------------------------------------------------------------
static void rv_sample(tsr_dbg * d) {
#if DBG_HAVE_RV
    tsr_rv * rv = hart(d);
    if (rv) {                                            // model values (DB §9.1)
        uint32_t epc = 0, tval = 0;
        (void)tsr_rv_csr_read(rv, TSR_RV_CSR_MEPC, &epc);
        (void)tsr_rv_csr_read(rv, TSR_RV_CSR_MTVAL, &tval);
        d->smp[0] = tsr_rv_pc(rv);
        d->smp[1] = tsr_rv_pc(rv);
        d->smp[2] = d->last_daddr;
        d->smp[3] = epc;
        d->smp[4] = tval;
        d->smp_valid = 1;
        d->smp_seq = (d->smp_seq + 1u) & 0xFFu;
        return;
    }
#endif
    d->smp_valid = 0;                                    // no clk_rv domain: the 4,096-cycle bound
    d->sticky |= TSR_DBG_STATUS_RV_SAMPLE_TIMEOUT;
}

static void rvwp_apply(tsr_dbg * d) {
#if DBG_HAVE_RV
    if (hart(d)) {
        memcpy(d->wpa_ctl, d->wp_ctl, sizeof(d->wpa_ctl));
        memcpy(d->wpa_lo, d->wp_lo, sizeof(d->wpa_lo));
        memcpy(d->wpa_hi, d->wp_hi, sizeof(d->wpa_hi));
        return;
    }
#endif
    d->sticky |= TSR_DBG_STATUS_RVWP_TIMEOUT;
}

static void hist_clear(tsr_dbg * d) {
    for (uint32_t i = 0; i < TSR_HIST_DEPTH; ++i) d->hist[i].valid = 0;
    d->hist_err = 0;
    d->hist_hung = 0;
    d->hist_head = d->next_pidx;
}

static void ctrl_write(tsr_dbg * d, uint32_t v) {
    if (v & TSR_DBG_CTRL_QSTEP) qstep(d);                // one of bits 0..2: QSTEP > RELEASE > STOP_NOW
    else if (v & TSR_DBG_CTRL_RELEASE) release(d);
    else if (v & TSR_DBG_CTRL_STOP_NOW) stop_now(d, TSR_DBG_CAUSE_NOW);
    if (v & TSR_DBG_CTRL_HIST_CLEAR) hist_clear(d);
    if (v & TSR_DBG_CTRL_RV_SAMPLE) rv_sample(d);
    if (v & TSR_DBG_CTRL_RVWP_APPLY) rvwp_apply(d);
    if (v & TSR_DBG_CTRL_TRW_FLUSH) trw_flush(d);
    if (v & TSR_DBG_CTRL_RV_HALT_CANCEL) {
        d->pend = 0;
        req_drive(d);
    }
    if ((v & TSR_DBG_CTRL_MATCHQ_CLEAR) && d->mq_n) {
        uint32_t T = 0;
        const int was = stop_T(d, &T);
        const uint32_t cause = stop_cause(d);
        d->mq_rd = d->mq_n = 0;
        if (was && !stop_T(d, NULL)) {
            step_cancel(d);
            trw_stop_rec(d, T, cause, 0);
        }
    }
    poll(d);
}

static void reg_write(tsr_dbg * d, uint32_t off, uint32_t v) {
    uint32_t i;
    if (off >= TSR_CSR_DBG_RVWP_CTL(0) && off < TSR_CSR_DBG_RVWP_CTL(TSR_DBG_NRVWP)) {
        i = (off - TSR_CSR_DBG_RVWP_CTL(0)) / 16;
        switch ((off - TSR_CSR_DBG_RVWP_CTL(0)) % 16) {
        case 0:  d->wp_ctl[i] = v & RVWP_CTL_MASK; break;
        case 4:  d->wp_lo[i] = v; break;
        case 8:  d->wp_hi[i] = v; break;
        default: d->wp_hits[i] = 0; break;               // a write clears
        }
        return;
    }
    if (off >= TSR_CSR_DBG_TBP_CTL(0) && off < TSR_CSR_DBG_TBP_CTL(TSR_DBG_NTBP)) {
        i = (off - TSR_CSR_DBG_TBP_CTL(0)) / 16;
        switch ((off - TSR_CSR_DBG_TBP_CTL(0)) % 16) {
        case 0:  d->tbp_ctl[i] = v & TBP_CTL_MASK; break;
        case 4:  d->tbp_a[i] = v; break;
        case 8:  d->tbp_b[i] = v; break;
        default: d->tbp_hits[i] = 0; break;
        }
        return;
    }
    switch (off) {
    case TSR_CSR_DBG_CTRL: ctrl_write(d, v); return;
    case TSR_CSR_DBG_CFG: {
        const uint32_t old = d->cfg;
        d->cfg = v & CFG_MASK;
        if ((old & TSR_DBG_CFG_TRW_EN) && !(d->cfg & TSR_DBG_CFG_TRW_EN)) trw_flush(d);
        poll(d);
        return;
    }
    case TSR_CSR_DBG_STATUS: d->sticky &= ~(v & STICKY_MASK); return;   // W1C
    case TSR_CSR_DBG_STOP_AT:
        d->S = v;
        d->s_valid = 1;
        d->s_cause = TSR_DBG_CAUSE_NOW;
        activation(d, 0);
        poll(d);
        return;
    case TSR_CSR_DBG_STEP_N:  d->step_n = v & TSR_DBG_STEP_N_N; return;
    case TSR_CSR_DBG_TRW_BASE: d->trw_base = v & ~0xFu; return;          // 16-beat aligned
    case TSR_CSR_DBG_TRW_LINES:
        if (v >= 16u && (v & (v - 1u)) == 0) d->trw_lines = v;
        else d->sticky |= TSR_DBG_STATUS_PTR_ERR;
        return;
    case TSR_CSR_DBG_TRW_RPTR:
        if ((uint32_t)(v - d->trw_rptr) <= (uint32_t)(d->trw_wptr - d->trw_rptr)) d->trw_rptr = v;
        else d->sticky |= TSR_DBG_STATUS_PTR_ERR;
        return;
    case TSR_CSR_DBG_TRW_MASK: d->trw_mask = v & (TSR_DBG_TRW_MASK_UNITS | TSR_DBG_TRW_MASK_MARKERS |
                                                  TSR_DBG_TRW_MASK_STOPS | TSR_DBG_TRW_MASK_RVHALTS); return;
    case TSR_CSR_DBG_HIST_IDX: d->hist_idx = v; return;
    default: return;                                     // read-only and unlisted: ignored
    }
}

// ---------------------------------------------------------------------------
// DMI window (DB §3.2)
// ---------------------------------------------------------------------------
#if DBG_HAVE_DM
// DBG_STATUS.SBA_REFUSED (DB §3.6, CC §4.3): an SBA access started through the window (a write of
// sbdata0, a write of sbaddress0 with sbreadonaddr, a read of sbdata0 with sbreadondata) that set
// sberror at an address CC's rules refuse: the DM range, a control- or core-window write, a TCM
// write while RV_RUN = 0 (the DM model reports a refusal as a bus error, tsr_dm.h).
static uint32_t sba_err(tsr_dm * dm) { return (tsr_dm_dmi_read(dm, TSR_DMI_SBCS) >> 12) & 7u; }

static void sba_note(tsr_dbg * d, tsr_dm * dm, uint32_t addr7, int is_write, uint32_t err0) {
    if (err0 != 0 || sba_err(dm) == 0) return;
    const int wr = is_write && addr7 == TSR_DMI_SBDATA0;
    const uint64_t x = tsr_dm_dmi_read(dm, TSR_DMI_SBADDRESS0);
    uint32_t itcm = 0, dtcm = 0;
    (void)tsr_sim_itcm(d->sim, &itcm);
    (void)tsr_sim_dtcm(d->sim, &dtcm);
    int refused = x - TSR_RV_DM_BASE < TSR_RV_DM_SIZE;
    if (wr && (x - TSR_RV_CTL_BASE < TSR_RV_CTL_SIZE || x - TSR_RV_CORE_BASE < TSR_RV_CORE_SIZE)) refused = 1;
    if (wr && !tsr_sim_rv_run(d->sim) && (x - TSR_RV_ITCM_BASE < itcm || x - TSR_RV_DTCM_BASE < dtcm)) refused = 1;
    if (refused) d->sticky |= TSR_DBG_STATUS_SBA_REFUSED;
}
static int sba_access(uint32_t addr7, int is_write) {
    return addr7 == TSR_DMI_SBDATA0 || (is_write && addr7 == TSR_DMI_SBADDRESS0);
}
#endif

static uint32_t dmi_read(tsr_dbg * d, uint32_t off) {
    if (!(d->cfg & TSR_DBG_CFG_DMI_EN)) return 0;
#if DBG_HAVE_DM
    tsr_dm * dm = dmod(d);
    if (dm) {
        const uint32_t a7 = (off - TSR_CSR_DMI_BASE) / 4;
        const uint32_t err0 = sba_access(a7, 0) ? sba_err(dm) : 0u;
        const uint32_t v = tsr_dm_dmi_read(dm, a7);
        if (sba_access(a7, 0)) sba_note(d, dm, a7, 0, err0);
        rv_refresh(d);
        return v;
    }
#else
    (void)off;
#endif
    return 0;
}

static void dmi_write(tsr_dbg * d, uint32_t off, uint32_t v) {
    if (!(d->cfg & TSR_DBG_CFG_DMI_EN)) {
        d->sticky |= TSR_DBG_STATUS_DMI_BLOCKED;         // dropped without reaching the DM
        return;
    }
#if DBG_HAVE_DM
    tsr_dm * dm = dmod(d);
    if (dm) {
        const uint32_t a7 = (off - TSR_CSR_DMI_BASE) / 4;
        rv_refresh(d);                                   // a resume never meets a stale request
        const uint32_t err0 = sba_access(a7, 1) ? sba_err(dm) : 0u;
        tsr_dm_dmi_write(dm, a7, v);
        if (sba_access(a7, 1)) sba_note(d, dm, a7, 1, err0);
        rv_refresh(d);
    }
#else
    (void)off; (void)v;
#endif
}

// ---------------------------------------------------------------------------
// Debug hooks (tsr_sim_int.h §5)
// ---------------------------------------------------------------------------
static int is_dmi(uint32_t off) { return off >= TSR_CSR_DMI_BASE && off < TSR_CSR_DMI(TSR_CSR_DMI_N); }

static int h_csr_read(void * ctx, uint32_t off, uint32_t * val, int host) {
    tsr_dbg * d = (tsr_dbg *)ctx;
    if (is_dmi(off)) {
        *val = host ? dmi_read(d, off) : 0u;             // core window: 0 (CC §5.3)
        return 1;
    }
    rv_refresh(d);
    trw_idle_check(d);
    *val = reg_read(d, off);
    return 1;
}

static void h_csr_write(void * ctx, uint32_t off, uint32_t val) {
    tsr_dbg * d = (tsr_dbg *)ctx;
    if (is_dmi(off)) {
        dmi_write(d, off, val);
        return;
    }
    rv_refresh(d);
    trw_idle_check(d);
    reg_write(d, off, val);
}

static int h_cw_read(void * ctx, uint32_t off, uint32_t * val) {
    tsr_dbg * d = (tsr_dbg *)ctx;
    const uint32_t * head = d->rec_n ? d->rec[d->rec_rd] : NULL;
    switch (off) {
    case TSR_CW_REC_COUNT: *val = d->rec_n; break;
    case TSR_CW_REC_DATA(0): case TSR_CW_REC_DATA(1): case TSR_CW_REC_DATA(2): case TSR_CW_REC_DATA(3):
        *val = head ? head[(off - TSR_CW_REC_DATA(0)) / 4] : 0u;
        break;
    case TSR_CW_REC_CFG: *val = d->rec_cfg; break;
    case TSR_CW_DBG_FLAGS:
        *val = (dbg_stop_level(d, core_status(d)) ? TSR_DBG_FLAGS_DBG_STOP : 0u) | ((d->cfg & TSR_DBG_CFG_SERIAL) ? TSR_DBG_FLAGS_SERIAL : 0u) |
               ((d->cfg & TSR_DBG_CFG_TRW_EN) ? TSR_DBG_FLAGS_TRW_EN : 0u) |
               ((d->cfg & TSR_DBG_CFG_DMI_EN) ? TSR_DBG_FLAGS_DMI_EN : 0u) |
               ((d->cfg & TSR_DBG_CFG_HALT_ON_BOOT) ? TSR_DBG_FLAGS_HALT_ON_BOOT : 0u);
        break;
    default: *val = 0; break;                            // reserved for DB
    }
    return 1;
}

static void h_cw_write(void * ctx, uint32_t off, uint32_t val) {
    tsr_dbg * d = (tsr_dbg *)ctx;
    if (off == TSR_CW_REC_POP) {
        if (d->rec_n) {
            d->rec_rd = (d->rec_rd + 1) % TSR_REC_DEPTH;
            d->rec_n--;
        }
    } else if (off == TSR_CW_REC_CFG) {
        d->rec_cfg = val & (TSR_REC_CFG_ALL | TSR_REC_CFG_IRQ_HALF);
    }
}

static uint32_t h_unit_hold(void * ctx) {
    tsr_dbg * d = (tsr_dbg *)ctx;
    return hold_mask_eff(d);                             // 0 at once with no stop and no serial mode
}

static uint32_t h_unit_step(void * ctx) {
    tsr_dbg * d = (tsr_dbg *)ctx;
    if (!d->step_arm) return 0;
    if (TSR_GET(core_status(d), TSR_STATUS_HOLD_SRC) & (TSR_HOLD_SRC_HOST | TSR_HOLD_SRC_RV)) return 0;
    for (uint32_t u = 0; u < TSR_NUM_UNITS; ++u) {
        const shent * h = sh_head(&d->q[u]);
        if (h && h->pidx == d->step_arm_pidx) return 1u << u;
    }
    return 0;
}

static int h_stop_active(void * ctx) { return stop_T((const tsr_dbg *)ctx, NULL); }

static uint32_t h_rv_irq(void * ctx) {
    const tsr_dbg * d = (const tsr_dbg *)ctx;
    return ((d->rec_cfg & TSR_REC_CFG_IRQ_HALF) && d->rec_n >= TSR_REC_IRQ_LEVEL) ? TSR_RV_MIP_FAST(TSR_IRQ_FAST_REC) : 0u;
}

static void h_rv_state(void * ctx) { rv_refresh((tsr_dbg *)ctx); }

// SOFT_RESET step 1 (EX §8.5, DB §5.2): S, S_VALID and the match queue; TBPs, SERIAL and the
// evidence are kept (dbg_hold is forced to 0 while SR_BUSY by hold_mask_eff).
static void h_soft_reset(void * ctx) {
    tsr_dbg * d = (tsr_dbg *)ctx;
    uint32_t T = 0;
    const int active = stop_T(d, &T);
    const uint32_t cause = stop_cause(d);
    d->s_valid = 0;
    d->mq_rd = d->mq_n = 0;
    step_cancel(d);
    if (active) trw_stop_rec(d, T, cause, 0);
    rv_refresh(d);
}

// ---------------------------------------------------------------------------
// Attach / detach
// ---------------------------------------------------------------------------
tsr_dbg * tsr_sim_dbg(tsr_sim * sim) {
    void ** slot = tsr_sim_model_slot(sim, TSR_MODEL_DBG);
    return slot ? (tsr_dbg *)*slot : NULL;
}

void tsr_dbg_attach(tsr_sim * sim) {
    void ** slot = tsr_sim_model_slot(sim, TSR_MODEL_DBG);
    if (slot == NULL || *slot != NULL) return;
    tsr_dbg * d = (tsr_dbg *)calloc(1, sizeof(*d));
    if (d == NULL) {
        fprintf(stderr, "tsr_dbg: out of memory\n");
        abort();
    }
    d->sim = sim;
    d->num_ch = tsr_sim_num_ch(sim);
    d->ch_bytes = tsr_sim_config(sim)->ch_bytes;
    d->cfg = TSR_CSR_DBG_CFG_RESET;                      // HIST_EN, HIST_FREEZE_ERR
    d->step_n = TSR_CSR_DBG_STEP_N_RESET;
    d->trw_mask = TSR_CSR_DBG_TRW_MASK_RESET;
    d->next_pidx = tsr_sim_corewin_read(sim, TSR_CSR_Q_PIDX);
    d->first_pidx = d->next_pidx;
    d->hist_head = d->next_pidx;
    d->rec_cfg = TSR_CW_REC_CFG_RESET;
    *slot = d;
    d->lis_id = tsr_sim_add_listener(sim, on_qev, d);
    tsr_sim_dbg_hooks h;
    memset(&h, 0, sizeof(h));
    h.ctx = d;
    h.csr_read = h_csr_read;
    h.csr_write = h_csr_write;
    h.cw_read = h_cw_read;
    h.cw_write = h_cw_write;
    h.unit_hold = h_unit_hold;
    h.soft_reset = h_soft_reset;
    h.unit_step = h_unit_step;
    h.stop_active = h_stop_active;
    h.rv_irq = h_rv_irq;
    h.rv_state = h_rv_state;
    tsr_sim_set_dbg_hooks(sim, &h);
#if DBG_HAVE_RV
    tsr_rv * rv = hart(d);
    if (rv) tsr_rv_set_data_watch(rv, on_data, d);
#if DBG_HAVE_DM
    if (dmod(d)) tsr_dm_set_event_hook(dmod(d), on_dbg_ev, d);   // the DM owns the hart's debug hook
    else
#endif
    if (rv) tsr_rv_set_debug_hook(rv, on_dbg_ev, d);
#endif
}

void tsr_dbg_detach(tsr_sim * sim) {
    void ** slot = tsr_sim_model_slot(sim, TSR_MODEL_DBG);
    if (slot == NULL || *slot == NULL) return;
    tsr_dbg * d = (tsr_dbg *)*slot;
    d->pend = 0;
    req_drive(d);
#if DBG_HAVE_RV
    tsr_rv * rv = hart(d);
    if (rv) {
        tsr_rv_set_data_watch(rv, NULL, NULL);
        tsr_rv_freeze_wait(rv, 0);
    }
#if DBG_HAVE_DM
    if (dmod(d)) tsr_dm_set_event_hook(dmod(d), NULL, NULL);
    else
#endif
    if (rv) tsr_rv_set_debug_hook(rv, NULL, NULL);
#endif
    tsr_sim_set_dbg_hooks(sim, NULL);
    if (d->lis_id >= 0) tsr_sim_remove_listener(sim, d->lis_id);
    *slot = NULL;
    free(d);
}

// ---------------------------------------------------------------------------
// Format helpers (tsr_dbg.h)
// ---------------------------------------------------------------------------
uint32_t tsr_log_hdr_make(uint32_t n, uint32_t level, uint32_t source, uint32_t type, int ts, uint32_t lseq) {
    return TSR_PUT(TSR_LOG_HDR_N, n) | TSR_PUT(TSR_LOG_HDR_LEVEL, level) | TSR_PUT(TSR_LOG_HDR_SOURCE, source) |
           TSR_PUT(TSR_LOG_HDR_TYPE, type) | (ts ? TSR_LOG_HDR_TS : 0u) | TSR_PUT(TSR_LOG_HDR_LSEQ, lseq) |
           TSR_PUT(TSR_LOG_HDR_SYNC, TSR_LOG_HDR_SYNC_MAGIC);
}

void tsr_log_hdr_decode(uint32_t hdr, tsr_log_hdr * o) {
    o->n = TSR_GET(hdr, TSR_LOG_HDR_N);
    o->level = TSR_GET(hdr, TSR_LOG_HDR_LEVEL);
    o->source = TSR_GET(hdr, TSR_LOG_HDR_SOURCE);
    o->type = TSR_GET(hdr, TSR_LOG_HDR_TYPE);
    o->ts = TSR_GET(hdr, TSR_LOG_HDR_TS);
    o->lseq = TSR_GET(hdr, TSR_LOG_HDR_LSEQ);
    o->sync = TSR_GET(hdr, TSR_LOG_HDR_SYNC);
}

int tsr_log_rec_check(const uint32_t * w, size_t avail, uint32_t fmt_lo, uint32_t fmt_hi, uint32_t * words) {
    if (avail < 1) return TSR_LOG_BAD_LEN;
    tsr_log_hdr h;
    tsr_log_hdr_decode(w[0], &h);
    if (h.sync != TSR_LOG_HDR_SYNC_MAGIC) return TSR_LOG_BAD_SYNC;
    if (h.type > TSR_LOG_HDR_TYPE_BLOB) return TSR_LOG_BAD_TYPE;
    if (h.level > TSR_LOG_HDR_LEVEL_TRACE) return TSR_LOG_BAD_LEVEL;
    if ((size_t)h.n + 1u > avail) return TSR_LOG_BAD_LEN;
    if (h.type == TSR_LOG_HDR_TYPE_FMT) {
        const uint32_t at = 1u + h.ts;                   // the format address follows the MTIME word
        if (h.n < at) return TSR_LOG_BAD_FMT;
        if (fmt_hi != 0 && (w[at] < fmt_lo || w[at] >= fmt_hi)) return TSR_LOG_BAD_FMT;
    }
    if (h.type == TSR_LOG_HDR_TYPE_TRAP && h.n != TSR_CRASH_LOG_WORDS + h.ts) return TSR_LOG_BAD_TRAP;
    if (words) *words = 1u + h.n;
    return 0;
}

void tsr_dbg_rec_decode(const uint32_t r[4], tsr_dbg_rec * o) {
    o->ticket = r[0];
    o->t_end = r[1];
    o->run = r[2];
    o->code = TSR_GET(r[3], TSR_REC_DATA3_CODE);
    o->ran = TSR_GET(r[3], TSR_REC_DATA3_RAN);
    o->marker = TSR_GET(r[3], TSR_REC_DATA3_MARKER);
    o->opcode = TSR_GET(r[3], TSR_REC_DATA3_OPCODE);
    o->pidx16 = TSR_GET(r[3], TSR_REC_DATA3_PIDX);
}

void tsr_trw_decode(const uint8_t rec[32], tsr_trw_rec * o) {
    memset(o, 0, sizeof(*o));
    for (uint32_t i = 0; i < 8; ++i)
        o->w[i] = (uint32_t)rec[4 * i] | ((uint32_t)rec[4 * i + 1] << 8) | ((uint32_t)rec[4 * i + 2] << 16) |
                  ((uint32_t)rec[4 * i + 3] << 24);
    o->kind = TSR_GET(o->w[0], TSR_TRW_W0_KIND);
    if (o->kind == TSR_TRW_W0_KIND_OP) {
        o->unit = TSR_GET(o->w[0], TSR_TRW_W0_UNIT);
        o->code = TSR_GET(o->w[0], TSR_TRW_W0_CODE);
        o->ran = TSR_GET(o->w[0], TSR_TRW_W0_RAN);
        o->rec = TSR_GET(o->w[0], TSR_TRW_W0_REC);
        o->tbp = TSR_GET(o->w[0], TSR_TRW_W0_TBP);
        o->opcode = TSR_GET(o->w[0], TSR_TRW_W0_OPCODE);
        o->w0_hi = TSR_GET(o->w[0], TSR_TRW_W0_W0_HI);
        o->qwait = o->w[6];
        o->hwait = o->w[7];
    }
    o->a1 = o->w[1];
    o->a2 = o->w[2];
    o->time = (uint64_t)o->w[3] | ((uint64_t)o->w[4] << 32);
    o->run = o->w[5];
}

void tsr_dbg_hist_read(tsr_sim * sim, uint32_t pidx, tsr_dbg_hist * o) {
    tsr_sim_csr_write(sim, TSR_CSR_DBG_HIST_IDX, pidx);
    o->info = tsr_sim_csr_read(sim, TSR_CSR_DBG_HIST_INFO);
    o->present = (o->info & TSR_DBG_HIST_INFO_PRESENT) != 0;
    o->unit = TSR_GET(o->info, TSR_DBG_HIST_INFO_UNIT);
    o->code = TSR_GET(o->info, TSR_DBG_HIST_INFO_CODE);
    o->rec = TSR_GET(o->info, TSR_DBG_HIST_INFO_REC);
    o->tbps = TSR_GET(o->info, TSR_DBG_HIST_INFO_TBPS);
    o->stop_point = TSR_GET(o->info, TSR_DBG_HIST_INFO_STOP);
    o->pidx = tsr_sim_csr_read(sim, TSR_CSR_DBG_HIST_PIDX);
    o->ticket = tsr_sim_csr_read(sim, TSR_CSR_DBG_HIST_TICKET);
    o->dep[0] = tsr_sim_csr_read(sim, TSR_CSR_DBG_HIST_DEP0);
    o->dep[1] = tsr_sim_csr_read(sim, TSR_CSR_DBG_HIST_DEP1);
    o->time = tsr_sim_csr_read(sim, TSR_CSR_DBG_HIST_TIME);
    o->pc = tsr_sim_csr_read(sim, TSR_CSR_DBG_HIST_PC);
    for (uint32_t i = 0; i < 8; ++i) o->w[i] = tsr_sim_csr_read(sim, TSR_CSR_DBG_HIST_W(i));
}

void tsr_dbg_unit_read(tsr_sim * sim, uint32_t u, tsr_dbg_unit * o) {
    o->state = tsr_sim_csr_read(sim, TSR_CSR_DBG_U_STATE(u));
    o->accepted = TSR_GET(o->state, TSR_DBG_U_STATE_ACCEPTED);
    o->head_state = TSR_GET(o->state, TSR_DBG_U_STATE_HEAD);
    o->hold_src = TSR_GET(o->state, TSR_DBG_U_STATE_HOLD);
    o->outstanding = TSR_GET(o->state, TSR_DBG_U_STATE_OUTSTANDING);
    o->oldest_ticket = tsr_sim_csr_read(sim, TSR_CSR_DBG_U_OLDEST(u));
    o->oldest_pidx = tsr_sim_csr_read(sim, TSR_CSR_DBG_U_OPIDX(u));
    o->age = tsr_sim_csr_read(sim, TSR_CSR_DBG_U_AGE(u));
}

void tsr_dbg_dump(tsr_sim * sim, FILE * f, uint32_t hist_n) {
    static const struct { uint32_t off; const char * name; } regs[] = {
        { TSR_CSR_DBG_ID, "DBG_ID" }, { TSR_CSR_DBG_CFG, "DBG_CFG" }, { TSR_CSR_DBG_STATUS, "DBG_STATUS" },
        { TSR_CSR_DBG_STOP_PIDX, "DBG_STOP_PIDX" }, { TSR_CSR_DBG_STOP_AT, "DBG_STOP_AT" },
        { TSR_CSR_DBG_STEP_N, "DBG_STEP_N" }, { TSR_CSR_DBG_NEXT_PIDX, "DBG_NEXT_PIDX" },
        { TSR_CSR_DBG_STEP_PIDX, "DBG_STEP_PIDX" }, { TSR_CSR_DBG_MATCHQ, "DBG_MATCHQ" },
        { TSR_CSR_DBG_MTIME_LO, "DBG_MTIME_LO" }, { TSR_CSR_DBG_MTIME_HI, "DBG_MTIME_HI" },
        { TSR_CSR_DBG_QSTATE, "DBG_QSTATE" }, { TSR_CSR_DBG_RV_STATE, "DBG_RV_STATE" },
        { TSR_CSR_DBG_RV_PC, "DBG_RV_PC" }, { TSR_CSR_DBG_RV_NPC, "DBG_RV_NPC" }, { TSR_CSR_DBG_RV_DADDR, "DBG_RV_DADDR" },
        { TSR_CSR_DBG_RV_EPC, "DBG_RV_EPC" }, { TSR_CSR_DBG_RV_EADDR, "DBG_RV_EADDR" },
        { TSR_CSR_DBG_RV_WPADDR, "DBG_RV_WPADDR" }, { TSR_CSR_DBG_TRW_BASE, "DBG_TRW_BASE" },
        { TSR_CSR_DBG_TRW_LINES, "DBG_TRW_LINES" }, { TSR_CSR_DBG_TRW_WPTR, "DBG_TRW_WPTR" },
        { TSR_CSR_DBG_TRW_RPTR, "DBG_TRW_RPTR" }, { TSR_CSR_DBG_TRW_DROPS, "DBG_TRW_DROPS" },
        { TSR_CSR_DBG_TRW_MASK, "DBG_TRW_MASK" }, { TSR_CSR_DBG_HIST_HEAD, "DBG_HIST_HEAD" },
    };
    for (size_t k = 0; k < sizeof(regs) / sizeof(regs[0]); ++k)
        fprintf(f, "%-16s 0x%03x = 0x%08x\n", regs[k].name, regs[k].off, tsr_sim_csr_read(sim, regs[k].off));
    const uint32_t st = tsr_sim_csr_read(sim, TSR_CSR_DBG_STATUS);
    fprintf(f, "  stop_active=%u drained=%u step_busy=%u db_held=%u cause=0x%x bp_hit=0x%x pend=%u halted=%u sleep=%u\n",
            TSR_GET(st, TSR_DBG_STATUS_STOP_ACTIVE), TSR_GET(st, TSR_DBG_STATUS_DRAINED),
            TSR_GET(st, TSR_DBG_STATUS_STEP_BUSY), TSR_GET(st, TSR_DBG_STATUS_DB_HELD), TSR_GET(st, TSR_DBG_STATUS_CAUSE),
            TSR_GET(st, TSR_DBG_STATUS_BP_HIT), TSR_GET(st, TSR_DBG_STATUS_RV_HALT_PEND),
            TSR_GET(st, TSR_DBG_STATUS_RV_HALTED), TSR_GET(st, TSR_DBG_STATUS_RV_SLEEP));
    for (uint32_t i = 0; i < TSR_DBG_NTBP; ++i)
        fprintf(f, "TBP%u ctl=0x%08x a=0x%08x b=0x%08x hits=%u\n", i, tsr_sim_csr_read(sim, TSR_CSR_DBG_TBP_CTL(i)),
                tsr_sim_csr_read(sim, TSR_CSR_DBG_TBP_A(i)), tsr_sim_csr_read(sim, TSR_CSR_DBG_TBP_B(i)),
                tsr_sim_csr_read(sim, TSR_CSR_DBG_TBP_HITS(i)));
    for (uint32_t j = 0; j < TSR_DBG_NRVWP; ++j)
        fprintf(f, "RVWP%u ctl=0x%x [0x%08x, 0x%08x) hits=%u\n", j, tsr_sim_csr_read(sim, TSR_CSR_DBG_RVWP_CTL(j)),
                tsr_sim_csr_read(sim, TSR_CSR_DBG_RVWP_LO(j)), tsr_sim_csr_read(sim, TSR_CSR_DBG_RVWP_HI(j)),
                tsr_sim_csr_read(sim, TSR_CSR_DBG_RVWP_HITS(j)));
    for (uint32_t u = 0; u < TSR_NUM_UNITS; ++u) {
        tsr_dbg_unit p;
        tsr_dbg_unit_read(sim, u, &p);
        fprintf(f, "unit %u: accepted=%u head=%u hold=0x%x outstanding=%u oldest=0x%08x pidx=%u age=%u\n", u, p.accepted,
                p.head_state, p.hold_src, p.outstanding, p.oldest_ticket, p.oldest_pidx, p.age);
    }
    const uint32_t head = tsr_sim_csr_read(sim, TSR_CSR_DBG_HIST_HEAD);
    const uint32_t keep = tsr_sim_csr_read(sim, TSR_CSR_DBG_HIST_IDX);
    if (hist_n > TSR_HIST_DEPTH) hist_n = TSR_HIST_DEPTH;
    for (uint32_t k = hist_n; k > 0; --k) {
        tsr_dbg_hist h;
        tsr_dbg_hist_read(sim, head - k, &h);
        if (!h.present) continue;
        fprintf(f, "hist pidx=%u ticket=0x%08x unit=%u code=%u dep=0x%08x,0x%08x pc=0x%08x t=%u w0=0x%08x%s%s\n", h.pidx,
                h.ticket, h.unit, h.code, h.dep[0], h.dep[1], h.pc, h.time, h.w[0], h.rec ? " rec" : "",
                h.stop_point ? " stop" : "");
    }
    tsr_sim_csr_write(sim, TSR_CSR_DBG_HIST_IDX, keep);
}
