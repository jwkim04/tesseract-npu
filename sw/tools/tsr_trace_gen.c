// tsr-trace-gen (v2) — random tile programs as `ops` traces (docs/v2/06-software-verification.md
// "SW" §5.2–§5.6, P1.10): push streams of tile ops with dependencies, concurrency across the units
// and error injection, run on the simulator to fill done, expect and tickets (SW §5.5), and written
// through the trace v2 library (tsr_trace.h).
//
//   tsr-trace-gen -o DIR [-n COUNT] [-s SEED] [--prefix NAME] [--ops gen|m2|m3|vu|at|ctrl]
//                 [--t-max T] [--num-ch C] [--spm-kib S] [--ch-gib G] [--addr-beat-w W] [--k-max K]
//                 [--caps CAPS] [--q-depth D0,D1,D2,D3,D4,D5] [--max-k K] [--max-n N]
//                 [--max-groups G] [--errors PCT] [--no-special] [--max-wbeats B] [--big-pct PCT]
//                 [--check-sched N] [--no-verify] [-q]
//
// Trace i is written to DIR/<prefix>_s<SEED>_<i, 4 digits>/ and depends only on (SEED, i) and the
// options (no host time, no unordered iteration: the same seed gives byte-identical files).
//
// Modes (--ops):
//   gen (alias m1)  v1's M1 programs: LD -> QACT -> MVM (-> MVM ACC) -> ST groups (traces/gen).
//   m2              v1's M2 programs: QACT groups and LD -> ST copies, no MVM (traces/m2).
//   m3              v1's M3 programs: MVM-centred groups (shapes, special values, boundaries) mixed
//                   with the M2 groups; the 23 MVM error kinds of v1 (traces/m3).
//   vu              every VU op (VU §5.2–§5.8) and every VOP function / RED kind and post / ROPE
//                   form / SOP sub, legal and illegal encodings (OPERAND, SPM_RANGE, reserved bits,
//                   unused fields), on data loaded by LD and stored by ST (traces/vu).
//   at              KVST, ATTQK, ATTPV, ASMAX, AMASK, AMERGE, ANORM (AT §5): KV planes written by
//                   KVST or the host, flash-attention chains, legal and illegal encodings (AT §5.8)
//                   (traces/at).
//   ctrl            the issue path (CC §7–§9): credits and back-pressure, holds, FENCE, NOP joins,
//                   deps across all units, BAD_DEP / DEP_FAILED / ABORTED, QABORT / QCLEAR
//                   recovery, trace marks and REC bits (traces/ctrl).
//
// v1 -> v2 (SW §5.6). There are no registers, LI / ADDI / END or register-relative fields: every
// address is the absolute immediate the allocator chose (x16 = 0, EX §7.1). v1's error endings are
// kept as error injections (the faulting op is pushed and the stream goes on); v1-only endings
// (fetch beyond the DRAM limit, missing END, CMD_ADDR_HI) are dropped. v2 endings are added:
// BAD_DEP, DEP_FAILED chains across the units, ABORTED after an error state, QABORT followed by a
// QCLEAR recovery, and an error followed by a QCLEAR recovery (the runtime's R9, EX §4.7).
//
// Dependencies. Every op's operand extents come from its checker class (tsr_sim_int.h §2); a
// hazard tracker adds a dependency on the newest conflicting op of every other unit (RAW, WAR, WAW
// on SPM lines, DRAM beats per channel and s registers); queue order covers the op's own unit.
// More than two needed deps are folded into SYNC NOP joins or a FENCE. Groups are interleaved, so
// the units run concurrently, and random extra deps on good tickets, sprinkled NOPs / FENCEs and
// REC bits vary the push stream. The programs are therefore race-free (SW §4.6) and every SW §4.7
// observable is schedule-independent; the rules that keep error records schedule-independent are:
//   - a QCLEAR is followed by a dependency on its ticket from the first entry of every other unit,
//     and nothing after it depends on a ticket before it (BAD_DEP "below the clear point" would
//     race the QCLEAR's application, CC §8.3);
//   - a QABORT comes only after a QHOLD set that opens the trace and held ops, so no op has started
//     at the abort; then either QHOLD clear and a QCLEAR (the abort's records are cleared), or, as
//     the trace's end, with a held entry at the front of every built unit's queue (each record is
//     that entry's): the record of a unit with nothing queued would take the next push index at the
//     abort's dispatch time, which depends on the schedule (CC §9.5 leaves it open);
//   - a hold window never holds more entries of a unit than its queue depth (no push waits on a
//     held queue).
// Every entry's completion code (0, its poison code, ABORTED, DEP_FAILED) is predicted from these
// rules (EX §8.2, CC §9.3) and checked.
//
// Verification of every trace before it is written (unless --no-verify): a run on the simulator
// with a queue listener (each entry's completion code against the prediction; the per-unit error
// records and the DONE record against the predicted first error); the serial check (the ops that
// ran, executed one by one in push order with tsr_sim_op on a second simulator, must leave the same
// SPM, s bank and DRAM: the program is race-free); the host references of v1 for the gen / m2 / m3
// outputs (tsr_qact, tsr_dot_row, tsr_fadd) and zero for the stores that must not run; then
// tsr_trace2_run_ops fills the trace and must agree with that run; --check-sched N replays the
// finished trace under N random schedules (tsr_trace2_replay_sim). A mismatch fails the trace
// (exit 1: simulator or generator bug).

#define _GNU_SOURCE

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "tsr_isa.h"
#include "tsr_ref.h"
#include "tsr_ref_at.h"
#include "tsr_sim.h"
#include "tsr_sim_int.h"
#include "tsr_trace.h"

// ---------------------------------------------------------------------------
// RNG (splitmix64, as v1)
// ---------------------------------------------------------------------------
typedef struct { uint64_t s; } rng_t;
static uint64_t rng_u64(rng_t * r) {
    uint64_t z = (r->s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
static uint32_t rng_u32(rng_t * r) { return (uint32_t)(rng_u64(r) >> 32); }
static uint32_t rng_below(rng_t * r, uint32_t n) { return n ? (uint32_t)(((rng_u64(r) >> 32) * (uint64_t)n) >> 32) : 0; }
static double rng_unit(rng_t * r) { return (double)(rng_u64(r) >> 11) * 0x1.0p-53; }
static double rng_normal(rng_t * r) {
    double u1 = rng_unit(r), u2 = rng_unit(r);
    if (u1 < 1e-300) u1 = 1e-300;
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}
static void rng_bytes(rng_t * r, void * dst, size_t n) {
    uint8_t * p = (uint8_t *)dst;
    for (size_t i = 0; i < n; ++i) p[i] = (uint8_t)rng_u32(r);
}

static float u2f(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static uint32_t f2u(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static void put_f32(uint8_t * d, float f) {
    const uint32_t u = f2u(f);
    d[0] = (uint8_t)u; d[1] = (uint8_t)(u >> 8); d[2] = (uint8_t)(u >> 16); d[3] = (uint8_t)(u >> 24);
}
static void put_u32(uint8_t * d, uint32_t u) {
    d[0] = (uint8_t)u; d[1] = (uint8_t)(u >> 8); d[2] = (uint8_t)(u >> 16); d[3] = (uint8_t)(u >> 24);
}
static float get_f32(const uint8_t * p) {
    return u2f((uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24));
}
static void * xmalloc(size_t n) {
    void * p = malloc(n ? n : 1);
    if (!p) { fprintf(stderr, "tsr-trace-gen: out of memory\n"); exit(2); }
    return p;
}
static void * xcalloc(size_t n) {
    void * p = calloc(n ? n : 1, 1);
    if (!p) { fprintf(stderr, "tsr-trace-gen: out of memory\n"); exit(2); }
    return p;
}
static void * xrealloc(void * p, size_t n) {
    p = realloc(p, n ? n : 1);
    if (!p) { fprintf(stderr, "tsr-trace-gen: out of memory\n"); exit(2); }
    return p;
}
static uint32_t umin32(uint32_t a, uint32_t b) { return a < b ? a : b; }
static uint64_t umin64(uint64_t a, uint64_t b) { return a < b ? a : b; }

#define TK(u, s) ((((uint32_t)(u)) << 28) | ((uint32_t)(s) & 0x0FFFFFFFu))
#define TK_U(t)  ((t) >> 28)
#define TK_S(t)  ((t) & 0x0FFFFFFFu)
#define NOFAIL   0xFFFFFFFFu

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------
enum { OPS_M1, OPS_M2, OPS_M3, OPS_VU, OPS_AT, OPS_CTRL };
static const char * const ops_names[] = { "gen", "m2", "m3", "vu", "at", "ctrl" };

typedef struct {
    tsr_config cfg;
    uint8_t    depth[TSR_NUM_UNITS];
    int        ops;
    int        m2, m3;            // v1 flags (m2: --ops m2; m3: --ops m3)
    uint32_t   max_k, max_n, max_groups;
    uint32_t   max_wbeats;        // --max-wbeats (m3): MVM weight beats per trace, busiest channel
    unsigned   big_pct;           // --big-pct (m3): every (100/PCT)-th trace gets a 4x weight budget
    int        special;
    unsigned   err_pct;
    unsigned   check_sched;       // --check-sched N: replay every trace under N random schedules
    int        verify;
    int        quiet;
} opts_t;

// ---------------------------------------------------------------------------
// Coverage (v1 slots, the v2 additions and the error-code counts)
// ---------------------------------------------------------------------------
enum { COV_Q8, COV_Q4, COV_PARTIAL, COV_NLTC, COV_ODDN, COV_NMODC, COV_TGT1, COV_ACC, COV_CHAIN, COV_XSTRIDE,
       COV_YSTRIDE, COV_ZSTRIDE, COV_SPECIAL, COV_ERR, COV_N };
static const char * cov_names[COV_N] = { "Q8_0 MVMs", "Q4_K MVMs", "partial-chunk K", "N < C", "odd N",
                                         "N % C != 0", "T > 1", "ACC (loaded)", "ACC chains", "src_stride gaps",
                                         "dst_stride gaps", "stride-0 encodings", "special activations",
                                         "error injections" };

// M2 coverage slots (--ops m2, and the M2 groups of m3).
enum { C2_A32, C2_A256, C2_TGT1, C2_XSTRIDE, C2_ZSTRIDE, C2_KMIN, C2_KMAX, C2_QPAIR, C2_COPY, C2_ZERO,
       C2_X4K, C2_DLIM, C2_SLIM, C2_SPECIAL, C2_ERR, C2_N };
static const char * cov2_names[C2_N] = { "QACT A32", "QACT A256", "T > 1", "src_stride gaps",
                                         "stride-0 encodings", "K = block size", "K = K_MAX", "QACT pairs",
                                         "LD/ST copies", "zero-length LD/ST", "4 KiB-crossing LD/ST",
                                         "operands ending at the DRAM limit", "operands ending at the SPM end",
                                         "special activations", "error injections" };

// M3 slots follow the M2 ones (defined with the M3 generator); VU / AT / ctrl slots after them.
#define COV_MAX  160
#define COV_V2   100    // v2 stream statistics (cov2v_names)
#define COV_ERRC 128    // cov[COV_ERRC + code]: traces whose first error (DONE ERR_CODE) is `code`
#define COV_ENTC 144    // cov[COV_ENTC + code]: entries completed with `code`
enum { V2_RECS, V2_OPS, V2_DEPS, V2_HAZ, V2_RDEP, V2_FOLDNOP, V2_FOLDFENCE, V2_SNOP, V2_SFENCE, V2_REC,
       V2_MARK, V2_HOLD, V2_QABORT, V2_QCLEAR, V2_BADDEP, V2_DEPFAIL, V2_ABORTED, V2_N };
static const char * cov2v_names[V2_N] = { "records", "ops", "deps", "hazard deps", "random deps",
                                          "NOP joins", "FENCE joins", "sprinkled NOPs", "sprinkled FENCEs",
                                          "REC pushes", "trace marks", "hold windows", "QABORT recipes",
                                          "QCLEAR recoveries", "BAD_DEP injections", "DEP_FAILED chains",
                                          "ABORTED-after-error tails" };
_Static_assert(COV_V2 + V2_N <= COV_ERRC, "coverage slots overlap");

// ---------------------------------------------------------------------------
// Generator state
// ---------------------------------------------------------------------------
typedef struct {
    uint32_t kind;          // TSR_TRACE_PUSH_*
    uint32_t w[8];
    uint32_t dep[2];        // tickets as pushed (a fresh simulator starts every sequence at 0, so
                            // absolute = relative, SW §5.2)
    uint32_t push;          // ISSUE_PUSH value (kind 0)
    uint32_t ticket;        // predicted ticket, TSR_TICKET_NONE without one
    uint32_t pidx;          // predicted push index (ticketed records)
    uint32_t want;          // predicted completion code (ticketed records)
    uint32_t info;          // predicted ERR_INFO when want is the record's own poison code
    uint8_t  ran;           // predicted to run on its unit (the serial check replays it)
} rec_t;

typedef struct {            // an access of an emitted op, for the hazard tracker
    uint32_t ticket;
    uint8_t  kind, write, ch;
    uint64_t lo, hi;
} acc_t;

typedef struct {            // an op of a group waiting to be interleaved
    uint32_t w[8];
    uint32_t grp;
} pend_t;

typedef struct {            // a DRAM region written by the host before the run
    uint32_t ch;
    uint64_t lo, hi;        // beats
} inreg_t;

typedef struct {
    uint32_t  ch;
    uint64_t  beat;
    uint32_t  lines;
    uint8_t * want;         // NULL: the region must stay zero (a store that must not run)
} check_t;

typedef struct {
    const opts_t * o;
    tsr_sim *  sim;         // staging simulator: its DRAM holds the inputs, its checkers classify ops
    rng_t      r;
    uint32_t   spm_lines;
    uint64_t   lim;
    // the push stream and its prediction
    rec_t *    rec;
    uint32_t   n_rec, cap_rec;
    uint32_t   cnt[TSR_NUM_UNITS];        // next sequence per unit (= ISSUED_SEQ)
    uint32_t   pidx;                      // next push index
    uint32_t   fail_seq[TSR_NUM_UNITS];   // the unit's first failing sequence since the last clear
    uint32_t   barrier;                   // QCLEAR ticket the first entry of each unit depends on
    uint32_t   barrier_need;              // bit u: unit u has had no entry since the barrier
    uint32_t   good_lo;                   // first record after the last QCLEAR (dep targets start here)
    int        quiet;                     // no sprinkled NOP / FENCE and no random deps
    int        hold;                      // a hold window is open (QHOLD set without clear)
    uint32_t   hold_room[TSR_NUM_UNITS];  // entries each unit may still take in the window
    uint32_t   hold_cnt[TSR_NUM_UNITS];   // sequences at the QHOLD set
    uint32_t   pushed_u[TSR_NUM_UNITS];   // entries pushed per unit (queue occupancy bound)
    // hazards since the last barrier
    acc_t *    acc;
    uint32_t   n_acc, cap_acc;
    // group interleaving
    pend_t *   pend;
    uint32_t   n_pend, cap_pend;
    uint32_t   grp;                       // group the next pend() belongs to
    // host inputs and checks
    inreg_t *  in;
    uint32_t   n_in, cap_in;
    check_t *  chk;
    size_t     n_chk, cap_chk;
    // expected first error of an injection (for the report; the prediction covers every entry)
    uint32_t   inj_pidx;                  // push index of the first injected fault since the last QCLEAR,
                                          // NOFAIL if none: it must be the trace's first error
    // M1 / M2 / M3 allocation (v1 fields)
    uint32_t   spm_next;
    uint64_t   io_next[TSR_NUM_CH_MAX];
    uint64_t   heap_next;
    uint32_t   spm_hi;                       // SPM lines [spm_hi, spm_lines) used from the top
    uint64_t   top_next[TSR_NUM_CH_MAX];     // DRAM beats [top_next, lim) used from the channel end
    uint64_t   io_lim;                       // M3: IO allocations stay below (= w_lo)
    uint64_t   w_lo;                         // M3: weight heap start
    int        wlim_pending;                 // M3: next non-empty tensor ends exactly at the DRAM limit
    uint64_t   wbudget;                      // M3: weight beats allowed (busiest channel, sum over MVMs)
    uint64_t   wch0, wall;                   // M3: weight beats used: busiest channel / all channels
    uint32_t   n_mvm;                        // M3: MVMs pushed
    unsigned long cov[COV_MAX];
} gen_t;

// ---------------------------------------------------------------------------
// Push stream: tickets, prediction, hazards, emission
// ---------------------------------------------------------------------------
static rec_t * rec_new(gen_t * g) {
    if (g->n_rec == g->cap_rec) {
        g->cap_rec = g->cap_rec ? 2 * g->cap_rec : 128;
        g->rec = (rec_t *)xrealloc(g->rec, g->cap_rec * sizeof(rec_t));
    }
    rec_t * r = &g->rec[g->n_rec++];
    memset(r, 0, sizeof(*r));
    r->dep[0] = r->dep[1] = TSR_TICKET_NONE;
    r->ticket = TSR_TICKET_NONE;
    g->cov[COV_V2 + V2_RECS]++;
    return r;
}

// The unit the issue path sends opcode op to (CC §7.1, §7.4; as tsr_sim_accept_op): its unit if
// the opcode is legal, the configuration has its CAPS bits and its unit and a handler exists;
// otherwise SYNC (refused: ILLEGAL_OP). *h = the handler of an accepted opcode, else NULL.
static uint32_t op_unit(const gen_t * g, uint32_t w0, const tsr_op_handler ** hp) {
    const uint32_t op = w0 & 0xFFu;
    const tsr_op_handler * h = NULL;
    if (tsr_op_legal(op)) {
        const uint32_t need = tsr_op_caps(op), u = tsr_op_unit(op);
        if ((g->o->cfg.caps & need) == need && u < TSR_NUM_UNITS && g->o->depth[u] > 0) {
            h = tsr_sim_op_handler((uint8_t)op);
            if (h != NULL && (h->check == NULL || h->unit != u)) h = NULL;
        }
    }
    if (hp) *hp = h;
    return h ? h->unit : TSR_UNIT_SYNC;
}

// Checker result of w: the code (ILLEGAL_OP for a refused opcode), *info, and the extents.
static uint32_t classify(gen_t * g, const uint32_t w[8], uint32_t * unit, uint32_t * info, tsr_extents * x) {
    const tsr_op_handler * h;
    *unit = op_unit(g, w[0], &h);
    *info = 0;
    if (x) memset(x, 0, sizeof(*x));
    if (h == NULL) return TSR_ERR_ILLEGAL_OP;
    uint32_t i = 0;
    const uint32_t code = h->check(g->sim, w, x, &i);
    if (code != TSR_ERR_NONE) {
        if (x) memset(x, 0, sizeof(*x));
        *info = i;
    }
    return code;
}

static int unit_built(const gen_t * g, uint32_t u) { return u < TSR_NUM_UNITS && g->o->depth[u] > 0; }

// BAD_DEP at the dispatcher for a dep pushed at this point (CC §8.3): unit 6..14 or absent, or a
// ticket not pushed yet. (Below the clear point never happens: no dep reaches across a QCLEAR.)
static int dep_is_bad(const gen_t * g, uint32_t t) {
    const uint32_t v = TK_U(t);
    if (v == TSR_UNIT_NONE) return 0;
    if (!unit_built(g, v)) return 1;
    return ((TK_S(t) - g->cnt[v]) & 0x0FFFFFFFu) < 0x08000000u;
}

// Completion code of an entry of unit u (CC §9.3: poisoned, then the unit's error state, then a
// failed dependency). A dep (v, s) failed iff v's first failing sequence is <= s (CC §9.2).
static uint32_t predict(const gen_t * g, uint32_t u, uint32_t poison, const uint32_t dep[2]) {
    if (poison) return poison;
    if (g->fail_seq[u] != NOFAIL) return TSR_ERR_ABORTED;
    for (int i = 0; i < 2; ++i) {
        const uint32_t v = TK_U(dep[i]);
        if (v >= TSR_NUM_UNITS || dep_is_bad(g, dep[i])) continue;
        if (g->fail_seq[v] != NOFAIL && TK_S(dep[i]) >= g->fail_seq[v]) return TSR_ERR_DEP_FAILED;
    }
    return TSR_ERR_NONE;
}

// Records a ticketed entry of unit u with predicted code `want`.
static void take_ticket(gen_t * g, rec_t * r, uint32_t u, uint32_t want) {
    r->ticket = TK(u, g->cnt[u]);
    r->pidx = g->pidx++;
    r->want = want;
    if (want != TSR_ERR_NONE && g->fail_seq[u] == NOFAIL) g->fail_seq[u] = g->cnt[u];
    g->cnt[u]++;
    g->pushed_u[u]++;
    g->barrier_need &= ~(1u << u);
    if (g->hold) {
        if (g->hold_room[u] == 0) {
            fprintf(stderr, "tsr-trace-gen: internal: hold window overfills unit %u\n", u);
            exit(2);
        }
        g->hold_room[u]--;
    }
}

static void acc_add(gen_t * g, uint32_t ticket, const tsr_extents * x) {
    for (uint32_t i = 0; i < x->n; ++i) {
        if (g->n_acc == g->cap_acc) {
            g->cap_acc = g->cap_acc ? 2 * g->cap_acc : 256;
            g->acc = (acc_t *)xrealloc(g->acc, g->cap_acc * sizeof(acc_t));
        }
        acc_t * a = &g->acc[g->n_acc++];
        a->ticket = ticket;
        a->kind = x->e[i].kind;
        a->write = x->e[i].write;
        a->ch = x->e[i].ch;
        a->lo = x->e[i].lo;
        a->hi = x->e[i].hi;
    }
}

// Newest conflicting access per other unit (RAW, WAR, WAW). req[v] = sequence + 1, 0 = none.
static void hazards(const gen_t * g, uint32_t u, const tsr_extents * x, uint32_t req[TSR_NUM_UNITS]) {
    for (uint32_t i = 0; i < x->n; ++i) {
        const tsr_extent * e = &x->e[i];
        for (uint32_t k = 0; k < g->n_acc; ++k) {
            const acc_t * a = &g->acc[k];
            const uint32_t v = TK_U(a->ticket);
            if (v == u || a->kind != e->kind || !(e->write || a->write)) continue;
            if (e->kind == TSR_EXT_DRAM && e->ch != a->ch && e->ch != TSR_EXT_CH_ALL && a->ch != TSR_EXT_CH_ALL) continue;
            if (!(e->lo < a->hi && a->lo < e->hi)) continue;
            if (TK_S(a->ticket) + 1 > req[v]) req[v] = TK_S(a->ticket) + 1;
        }
    }
}

static uint32_t push_value(gen_t * g, const uint32_t w[8], int rec_ok) {
    uint32_t m = 0;
    for (uint32_t i = 1; i < 8; ++i) if (w[i] != 0) m |= 1u << i;
    m |= rng_below(&g->r, 4) == 0 ? (rng_u32(&g->r) & 0xFEu) : 0u;   // extra bits for zero words
    if (rng_below(&g->r, 3) == 0) m = 0xFF;
    uint32_t v = (m | 1u) << 8;                                      // w0 is always sent
    if (rec_ok && rng_below(&g->r, 8) == 0) {
        v |= TSR_ISSUE_PUSH_REC;
        g->cov[COV_V2 + V2_REC]++;
    }
    return v;
}

static int ticket_good(const gen_t * g, uint32_t i) {
    const rec_t * r = &g->rec[i];
    return r->ticket != TSR_TICKET_NONE && r->want == TSR_ERR_NONE;
}

// A random good ticket after the last QCLEAR (a dep on it changes nothing but the order).
static uint32_t random_good_ticket(gen_t * g) {
    if (g->n_rec <= g->good_lo) return TSR_TICKET_NONE;
    for (int a = 0; a < 8; ++a) {
        const uint32_t i = g->good_lo + rng_below(&g->r, g->n_rec - g->good_lo);
        if (ticket_good(g, i)) return g->rec[i].ticket;
    }
    return TSR_TICKET_NONE;
}

static void emit_raw(gen_t * g, const uint32_t w[8], uint32_t d0, uint32_t d1, uint32_t pv, uint32_t u,
                     uint32_t poison, uint32_t info, const tsr_extents * x);

// A SYNC NOP / FENCE with the given deps (used for folding and sprinkling). Returns its ticket.
static uint32_t emit_sync(gen_t * g, uint32_t op, uint32_t d0, uint32_t d1) {
    uint32_t w[8] = { op, 0, 0, 0, 0, 0, 0, 0 };
    const uint32_t pv = push_value(g, w, 1);
    emit_raw(g, w, d0, d1, pv, TSR_UNIT_SYNC, TSR_ERR_NONE, 0, NULL);
    return g->rec[g->n_rec - 1].ticket;
}

static void emit_raw(gen_t * g, const uint32_t w[8], uint32_t d0, uint32_t d1, uint32_t pv, uint32_t u,
                     uint32_t poison, uint32_t info, const tsr_extents * x) {
    uint32_t dep[2] = { d0, d1 };
    // BAD_DEP (CC §8.3) is a poison too: after ILLEGAL_OP, before every other code (EX §8.1).
    int bad = dep_is_bad(g, d0) || dep_is_bad(g, d1);
    if (bad && (poison == TSR_ERR_NONE || tsr_err_check_rank(poison) > tsr_err_check_rank(TSR_ERR_BAD_DEP))) {
        poison = TSR_ERR_BAD_DEP;
        info = 0;
    }
    const uint32_t want = predict(g, u, poison, dep);
    rec_t * r = rec_new(g);
    r->kind = TSR_TRACE_PUSH_OP;
    memcpy(r->w, w, sizeof(r->w));
    r->dep[0] = d0;
    r->dep[1] = d1;
    r->push = pv;
    r->info = poison ? info : 0;
    take_ticket(g, r, u, want);
    r->ran = want == TSR_ERR_NONE && u != TSR_UNIT_SYNC;
    if (x && poison == TSR_ERR_NONE) acc_add(g, r->ticket, x);
    if (d0 != TSR_TICKET_NONE) g->cov[COV_V2 + V2_DEPS]++;
    if (d1 != TSR_TICKET_NONE) g->cov[COV_V2 + V2_DEPS]++;
}

// Folds the dep list (tickets on other units, at most one per unit) into at most `slots` tickets
// with SYNC NOP joins (done(NOP) implies done of both its deps; a newer SYNC ticket implies every
// older one) or one FENCE (done implies every earlier entry).
static uint32_t fold(gen_t * g, uint32_t * d, uint32_t n, uint32_t slots) {
    if (n <= slots) return n;
    if (g->hold && g->hold_room[TSR_UNIT_SYNC] == 0) {
        fprintf(stderr, "tsr-trace-gen: internal: dependency folding inside a full hold window\n");
        exit(2);
    }
    if (g->hold || rng_below(&g->r, 3) == 0) {      // one SYNC entry (a hold window counts each)
        d[0] = emit_sync(g, TSR_OP_FENCE, TSR_TICKET_NONE, TSR_TICKET_NONE);
        g->cov[COV_V2 + V2_FOLDFENCE]++;
        return 1;
    }
    while (n > slots) {
        const uint32_t t = emit_sync(g, TSR_OP_NOP, d[n - 2], d[n - 1]);
        g->cov[COV_V2 + V2_FOLDNOP]++;
        n -= 2;
        uint32_t k = 0;
        for (uint32_t i = 0; i < n; ++i) if (TK_U(d[i]) != TSR_UNIT_SYNC) d[k++] = d[i];   // implied by t
        d[k++] = t;
        n = k;
    }
    return n;
}

#define EMIT_PLAIN   0u
#define EMIT_NODEPS  1u     // no random extra dep, no sprinkle (tails, recipes)
#define EMIT_NOHAZ   2u     // no hazard deps either: an op that can never run (held, then aborted)

// Pushes op w: deps = hazard deps (+ the barrier) folded into the slots left by the explicit deps
// x0 / x1 (TSR_TICKET_NONE = free), then a random good dep in a free slot; sometimes followed by a
// NOP or FENCE. Returns the record index.
static uint32_t emit_op(gen_t * g, const uint32_t w[8], uint32_t x0, uint32_t x1, uint32_t flags) {
    uint32_t u, info;
    tsr_extents x;
    const uint32_t code = classify(g, w, &u, &info, &x);
    uint32_t req[TSR_NUM_UNITS] = { 0 };
    // A poisoned entry (its checker code or a BAD_DEP dep) accesses nothing: no hazard deps. It
    // still waits for the barrier, or its record could land before the QCLEAR clears it.
    if (code == TSR_ERR_NONE && !dep_is_bad(g, x0) && !dep_is_bad(g, x1) && !(flags & EMIT_NOHAZ)) hazards(g, u, &x, req);
    if (u != TSR_UNIT_SYNC && (g->barrier_need >> u) & 1u) {
        const uint32_t bs = TK_S(g->barrier) + 1;
        if (bs > req[TSR_UNIT_SYNC]) req[TSR_UNIT_SYNC] = bs;
    }
    uint32_t d[TSR_NUM_UNITS], n = 0;
    for (uint32_t v = 0; v < TSR_NUM_UNITS; ++v) {
        if (req[v] == 0 || v == u) continue;
        d[n++] = TK(v, req[v] - 1);
        g->cov[COV_V2 + V2_HAZ]++;
    }
    const uint32_t nx = (x0 != TSR_TICKET_NONE) + (x1 != TSR_TICKET_NONE);
    if (n > 2 - nx && nx == 2) {
        fprintf(stderr, "tsr-trace-gen: internal: two explicit deps leave no slot for the needed ones\n");
        exit(2);
    }
    n = fold(g, d, n, 2 - nx);
    uint32_t dep[2] = { TSR_TICKET_NONE, TSR_TICKET_NONE };
    uint32_t k = 0;
    if (x0 != TSR_TICKET_NONE) dep[k++] = x0;
    if (x1 != TSR_TICKET_NONE) dep[k++] = x1;
    for (uint32_t i = 0; i < n; ++i) dep[k++] = d[i];
    const int plain = !(flags & EMIT_NODEPS) && !g->quiet;
    if (plain && k < 2 && rng_below(&g->r, 6) == 0) {
        const uint32_t t = random_good_ticket(g);
        if (t != TSR_TICKET_NONE) {
            dep[k++] = t;
            g->cov[COV_V2 + V2_RDEP]++;
        }
    }
    if (k == 2 && rng_below(&g->r, 2)) {                   // either slot
        const uint32_t t = dep[0];
        dep[0] = dep[1];
        dep[1] = t;
    } else if (k == 1 && rng_below(&g->r, 2)) {
        dep[1] = dep[0];
        dep[0] = TSR_TICKET_NONE;
    }
    const uint32_t pv = push_value(g, w, 1);
    emit_raw(g, w, dep[0], dep[1], pv, u, code, info, &x);
    const uint32_t idx = g->n_rec - 1;
    g->cov[COV_V2 + V2_OPS]++;
    if (plain && !g->hold) {
        const uint32_t s = rng_below(&g->r, 24);
        if (s == 0) {
            emit_sync(g, TSR_OP_NOP, random_good_ticket(g), rng_below(&g->r, 2) ? random_good_ticket(g) : TSR_TICKET_NONE);
            g->cov[COV_V2 + V2_SNOP]++;
        } else if (s == 1) {
            emit_sync(g, TSR_OP_FENCE, TSR_TICKET_NONE, TSR_TICKET_NONE);
            g->cov[COV_V2 + V2_SFENCE]++;
        }
    }
    return idx;
}

// A legal op (internal error if its checker refuses it: the generator's layout went wrong).
static uint32_t emit_insn(gen_t * g, tsr_insn i) {
    const uint32_t ri = emit_op(g, i.w, TSR_TICKET_NONE, TSR_TICKET_NONE, EMIT_PLAIN);
    const rec_t * q = &g->rec[ri];
    if (q->info != 0 || (q->want != TSR_ERR_NONE && q->want != TSR_ERR_ABORTED && q->want != TSR_ERR_DEP_FAILED)) {
        fprintf(stderr, "tsr-trace-gen: internal: an op meant to be legal is not: w0 0x%08x code %u info 0x%x\n",
                i.w[0], q->want, q->info);
        exit(2);
    }
    return ri;
}

// QCTRL records (one bit each, CC §5.1).
static void emit_qctrl(gen_t * g, uint32_t bit) {
    rec_t * r = rec_new(g);
    r->kind = TSR_TRACE_PUSH_QCTRL;
    r->w[0] = bit;
    if (bit == TSR_QCTRL_QCLEAR) {
        // CC §9.5: applied after every earlier entry; never aborted (completes 0)
        take_ticket(g, r, TSR_UNIT_SYNC, TSR_ERR_NONE);
        for (uint32_t u = 0; u < TSR_NUM_UNITS; ++u) g->fail_seq[u] = NOFAIL;
        g->barrier = r->ticket;
        g->barrier_need = 0;
        for (uint32_t u = 0; u < TSR_NUM_UNITS; ++u) if (unit_built(g, u) && u != TSR_UNIT_SYNC) g->barrier_need |= 1u << u;
        g->n_acc = 0;                    // everything before is done when the QCLEAR is
        g->good_lo = g->n_rec;           // nothing after depends on a ticket before it
        g->cov[COV_V2 + V2_QCLEAR]++;
    } else if (bit == TSR_QCTRL_QABORT) {
        // CC §9.5: every built unit not in error enters error state at S[u]; in the QHOLD-first
        // recipe nothing has started, so every entry since the hold fails: its poison code, else
        // ABORTED (the unit's error state comes before a failed dep, CC §9.3).
        for (uint32_t i = 0; i + 1 < g->n_rec; ++i) {
            rec_t * q = &g->rec[i];
            if (q->ticket == TSR_TICKET_NONE || TK_S(q->ticket) < g->hold_cnt[TK_U(q->ticket)]) continue;
            if (q->want == TSR_ERR_NONE || q->want == TSR_ERR_DEP_FAILED) {
                q->want = TSR_ERR_ABORTED;
                q->ran = 0;
            }
        }
        for (uint32_t u = 0; u < TSR_NUM_UNITS; ++u)
            if (unit_built(g, u) && g->fail_seq[u] == NOFAIL) g->fail_seq[u] = g->hold_cnt[u];
        g->cov[COV_V2 + V2_QABORT]++;
    } else if (bit == TSR_QCTRL_QHOLD_SET) {
        g->hold = 1;
        for (uint32_t u = 0; u < TSR_NUM_UNITS; ++u) {
            g->hold_room[u] = g->o->depth[u] - umin32(g->o->depth[u], g->pushed_u[u]);
            g->hold_cnt[u] = g->cnt[u];
        }
        g->cov[COV_V2 + V2_HOLD]++;
    } else if (bit == TSR_QCTRL_QHOLD_CLR) {
        g->hold = 0;
    }
}

static void emit_mark(gen_t * g, uint32_t v) {
    rec_t * r = rec_new(g);
    r->kind = TSR_TRACE_PUSH_MARK;
    r->w[0] = v;
    g->cov[COV_V2 + V2_MARK]++;
}

// ---------------------------------------------------------------------------
// Groups: ops collected per group, then interleaved (the units run them concurrently)
// ---------------------------------------------------------------------------
static void pend(gen_t * g, tsr_insn i) {
    if (g->n_pend == g->cap_pend) {
        g->cap_pend = g->cap_pend ? 2 * g->cap_pend : 64;
        g->pend = (pend_t *)xrealloc(g->pend, g->cap_pend * sizeof(pend_t));
    }
    memcpy(g->pend[g->n_pend].w, i.w, sizeof(i.w));
    g->pend[g->n_pend].grp = g->grp;
    g->n_pend++;
}
static void next_group(gen_t * g) { g->grp++; }

// Emits the pending ops: each step takes the next op of a random group (group order kept).
static void flush_groups(gen_t * g) {
    const uint32_t n = g->n_pend;
    if (n == 0) return;
    uint32_t ng = 0;
    for (uint32_t i = 0; i < n; ++i) if (g->pend[i].grp + 1 > ng) ng = g->pend[i].grp + 1;
    uint32_t * cur = (uint32_t *)xcalloc(ng * sizeof(uint32_t));   // per group: index of its next op
    uint32_t * left = (uint32_t *)xcalloc(ng * sizeof(uint32_t));
    for (uint32_t i = 0; i < n; ++i) left[g->pend[i].grp]++;
    for (uint32_t k = 0; k < ng; ++k) cur[k] = n;
    for (uint32_t i = n; i-- > 0;) cur[g->pend[i].grp] = i;
    const int mix = rng_below(&g->r, 4) != 0;              // 1/4 of the streams keep the groups apart
    uint32_t last = 0;
    for (uint32_t done = 0; done < n; ++done) {
        uint32_t k = last;
        if (left[k] == 0 || (mix && rng_below(&g->r, 2))) {
            uint32_t live = 0;
            for (uint32_t j = 0; j < ng; ++j) live += left[j] != 0;
            uint32_t pick = rng_below(&g->r, live);
            for (k = 0; k < ng; ++k) if (left[k] != 0 && pick-- == 0) break;
        }
        const uint32_t i = cur[k];
        const uint32_t ri = emit_op(g, g->pend[i].w, TSR_TICKET_NONE, TSR_TICKET_NONE, EMIT_PLAIN);
        if (g->rec[ri].info != 0 || (g->rec[ri].want != TSR_ERR_NONE && g->rec[ri].want != TSR_ERR_ABORTED &&
                                     g->rec[ri].want != TSR_ERR_DEP_FAILED)) {
            fprintf(stderr, "tsr-trace-gen: internal: a group op is not legal: w0 0x%08x code %u info 0x%x\n",
                    g->pend[i].w[0], g->rec[ri].want, g->rec[ri].info);
            exit(2);
        }
        left[k]--;
        uint32_t j = i + 1;
        while (j < n && g->pend[j].grp != k) ++j;
        cur[k] = j;
        last = k;
    }
    free(cur);
    free(left);
    g->n_pend = 0;
}

// ---------------------------------------------------------------------------
// Host inputs (DRAM written before the run) and checks
// ---------------------------------------------------------------------------
static void dram_put(gen_t * g, uint32_t ch, uint64_t beat, const void * data, size_t bytes) {
    if (bytes == 0) return;
    memcpy(tsr_sim_dram(g->sim, ch) + beat * 64, data, bytes);
    if (g->n_in == g->cap_in) {
        g->cap_in = g->cap_in ? 2 * g->cap_in : 32;
        g->in = (inreg_t *)xrealloc(g->in, g->cap_in * sizeof(inreg_t));
    }
    g->in[g->n_in].ch = ch;
    g->in[g->n_in].lo = beat;
    g->in[g->n_in].hi = beat + (bytes + 63) / 64;
    g->n_in++;
}

static void add_check(gen_t * g, uint32_t ch, uint64_t beat, uint32_t lines, uint8_t * want) {
    if (g->n_chk == g->cap_chk) {
        g->cap_chk = g->cap_chk ? 2 * g->cap_chk : 16;
        g->chk = (check_t *)xrealloc(g->chk, g->cap_chk * sizeof(check_t));
    }
    g->chk[g->n_chk].ch = ch;
    g->chk[g->n_chk].beat = beat;
    g->chk[g->n_chk].lines = lines;
    g->chk[g->n_chk].want = want;
    g->n_chk++;
}

// Turns the checks added since index `from` into "must stay zero" (their stores will not run).
static void checks_cancel(gen_t * g, size_t from) {
    for (size_t i = from; i < g->n_chk; ++i) {
        free(g->chk[i].want);
        g->chk[i].want = NULL;
    }
}

// ---------------------------------------------------------------------------
// Value generators (v1)
// ---------------------------------------------------------------------------
static uint16_t rand_fp16_scale(rng_t * r, int special) {
    const uint32_t k = rng_below(r, 64);
    const uint16_t sign = (uint16_t)((rng_u32(r) & 1u) << 15);
    if (special && k == 0) return sign;                                           // +-0
    if (special && k == 1) return (uint16_t)(sign | (1 + rng_below(r, 0x3FF)));   // subnormal
    const uint16_t e = (uint16_t)(1 + rng_below(r, 13));
    return (uint16_t)(sign | (e << 10) | rng_below(r, 1024));
}

static void gen_row_blocks(rng_t * r, uint32_t fmt, uint32_t K, uint8_t * out, int special) {
    const uint32_t nb = K / tsr_fmt_block_size(fmt);
    const size_t bsz = tsr_fmt_ggml_block_bytes(fmt);
    for (uint32_t b = 0; b < nb; ++b) {
        uint8_t * blk = out + (size_t)b * bsz;
        rng_bytes(r, blk, bsz);
        const uint16_t d = rand_fp16_scale(r, special);
        blk[0] = (uint8_t)d; blk[1] = (uint8_t)(d >> 8);
        if (fmt == TSR_FMT_Q4_K) {
            const uint16_t dm = rand_fp16_scale(r, special);
            blk[2] = (uint8_t)dm; blk[3] = (uint8_t)(dm >> 8);
        }
    }
}

static int gen_x(rng_t * r, float * x, uint32_t K, int special) {
    const double scale = ldexp(1.0, (int)rng_below(r, 16) - 8);
    for (uint32_t i = 0; i < K; ++i) x[i] = (float)(rng_normal(r) * scale);
    if (!special || rng_below(r, 3) != 0) return 0;
    for (uint32_t b = 0; b < K / 32; ++b) {
        float * xb = x + 32 * b;
        switch (rng_below(r, 16)) {
        case 0: memset(xb, 0, 32 * sizeof(float)); break;
        case 1: xb[rng_below(r, 32)] = u2f(0x00012345u | (rng_u32(r) & 0x80000000u)); break;
        case 2: { const float m = xb[0]; xb[5] = -m; xb[9] = m; } break;
        case 3: if (rng_below(r, 4) == 0) xb[rng_below(r, 32)] = (rng_u32(r) & 1) ? INFINITY : -INFINITY; break;
        case 4: if (rng_below(r, 4) == 0) xb[rng_below(r, 32)] = u2f(0x7FC00000u | rng_below(r, 1u << 22)); break;
        case 5: for (int i = 0; i < 32; ++i) xb[i] = u2f(0x00000100u * (uint32_t)(i + 1)); break;
        case 6: for (int i = 0; i < 32; ++i) xb[i] = (float)((int)rng_below(r, 255) - 127) * 0.5f; break;  // .5 ties
        default: break;
        }
    }
    return 1;
}

static uint32_t rand_K(rng_t * r, uint32_t fmt, uint32_t kmax) {
    const uint32_t bs = tsr_fmt_block_size(fmt);
    const uint32_t nmax = kmax / bs;
    switch (rng_below(r, 3)) {
    case 0: if (kmax >= 1024) return 1024 * (1 + rng_below(r, kmax / 1024)); /* fallthrough */
    case 1: return bs * (1 + rng_below(r, nmax < 8 ? nmax : 8));
    default: return bs * (1 + rng_below(r, nmax));
    }
}

// T f32 vectors of K elements, xs lines apart, into channel ch at beat `beat`.
static void put_x(gen_t * g, uint32_t ch, uint64_t beat, const float * x, uint32_t K, uint32_t T, uint32_t xs) {
    uint8_t * b = (uint8_t *)xmalloc((size_t)K * 4);
    for (uint32_t t = 0; t < T; ++t) {
        for (uint32_t k = 0; k < K; ++k) put_f32(b + 4 * k, x[(size_t)t * K + k]);
        dram_put(g, ch, beat + (uint64_t)t * xs, b, (size_t)K * 4);
    }
    free(b);
}

// Repacked rows of an N x K tensor (ggml blocks) at W_OFF in every channel (tsr_row_location).
static void put_tensor(gen_t * g, uint32_t fmt, uint32_t N, uint32_t K, uint64_t w_off, const uint8_t * blocks) {
    const uint32_t C = g->o->cfg.num_ch;
    const size_t rbytes = (size_t)(K / tsr_fmt_block_size(fmt)) * tsr_fmt_ggml_block_bytes(fmt);
    const uint32_t rb = tsr_row_beats(fmt, K);
    uint8_t * row = (uint8_t *)xmalloc((size_t)rb * 64);
    for (uint32_t n = 0; n < N; ++n) {
        uint32_t ch;
        uint64_t off;
        tsr_row_location(fmt, N, K, C, n, &ch, &off);
        tsr_repack_row(fmt, blocks + rbytes * n, K, row);
        dram_put(g, ch, w_off + off, row, (size_t)rb * 64);
    }
    free(row);
}

// ---------------------------------------------------------------------------
// M1 programs (--ops gen): LD -> QACT -> MVM (-> MVM ACC) -> ST groups (v1 gen_group)
// ---------------------------------------------------------------------------
static int gen_group(gen_t * g) {
    const tsr_config * c = &g->o->cfg;
    rng_t * r = &g->r;
    const uint32_t C = c->num_ch;
    const uint32_t kmax = g->o->max_k < c->k_max ? g->o->max_k : c->k_max;
    uint32_t fmt = 0, K = 0, N = 0, T = 0, xs = 0, ys = 0, xl = 0, ql = 0, yl = 0, rb = 0, R = 0;
    int acc = 0, chain = 0, fits = 0;
    for (int attempt = 0; attempt < 32 && !fits; ++attempt) {
        fmt = (kmax >= 256) ? rng_below(r, 2) : TSR_FMT_Q8_0;
        K = rand_K(r, fmt, kmax);
        switch (rng_below(r, 8)) {
        case 0: N = C > 1 ? 1 + rng_below(r, C - 1) : 1; break;    // N < C: some channels empty
        case 1: N = C + rng_below(r, C + 1); break;                 // C <= N <= 2C
        default: N = 1 + rng_below(r, g->o->max_n); break;
        }
        T = 1 + rng_below(r, c->t_max);
        acc = rng_below(r, 3) == 0;
        chain = rng_below(r, 4) == 0;
        xl = K / 16;
        ql = tsr_qa_lines(tsr_qa_kind_for_fmt(fmt), K);
        yl = tsr_ceil_div_u32(4 * N, 64);
        xs = xl + (rng_below(r, 3) == 0 ? 1 + rng_below(r, 3) : 0);
        ys = yl + (rng_below(r, 3) == 0 ? 1 + rng_below(r, 3) : 0);
        rb = tsr_row_beats(fmt, K);
        R = tsr_ceil_div_u32(N, C);
        const uint64_t spm_need = (uint64_t)(T - 1) * xs + xl + (uint64_t)T * ql + (uint64_t)(T - 1) * ys + yl + 8;
        const uint64_t w_need = (uint64_t)R * rb * 2 + 64;
        fits = g->spm_next + spm_need <= g->spm_lines && g->heap_next + w_need <= g->lim;
    }
    if (!fits) return 0;
    next_group(g);
    const uint32_t kind = tsr_qa_kind_for_fmt(fmt);
    g->cov[fmt ? COV_Q4 : COV_Q8] += 1 + chain;
    g->cov[COV_PARTIAL] += (K % 1024) != 0;
    g->cov[COV_NLTC] += N < C;
    g->cov[COV_ODDN] += N & 1;
    g->cov[COV_NMODC] += (N % C) != 0;
    g->cov[COV_TGT1] += T > 1;
    g->cov[COV_ACC] += acc;
    g->cov[COV_CHAIN] += chain;
    g->cov[COV_XSTRIDE] += xs != xl;
    g->cov[COV_YSTRIDE] += ys != yl;

    // Weights (one or two tensors), symmetric allocation in the WEIGHT region.
    const size_t rbytes = (size_t)(K / tsr_fmt_block_size(fmt)) * tsr_fmt_ggml_block_bytes(fmt);
    uint8_t * blocks[2] = { NULL, NULL };
    uint64_t w_off[2] = { 0, 0 };
    uint8_t * row = (uint8_t *)xmalloc((size_t)rb * 64);
    for (int w = 0; w < 1 + chain; ++w) {
        w_off[w] = g->heap_next + rng_below(r, 16);
        g->heap_next = w_off[w] + (uint64_t)R * rb + rng_below(r, 16);
        blocks[w] = (uint8_t *)xmalloc(rbytes * N);
        for (uint32_t n = 0; n < N; ++n) gen_row_blocks(r, fmt, K, blocks[w] + rbytes * n, g->o->special);
        put_tensor(g, fmt, N, K, w_off[w], blocks[w]);
    }

    // Activations in some channel's IO region.
    float * x = (float *)xmalloc(sizeof(float) * K * T);
    int sp = 0;
    for (uint32_t t = 0; t < T; ++t) sp |= gen_x(r, x + (size_t)t * K, K, g->o->special);
    g->cov[COV_SPECIAL] += sp;
    const uint32_t cx = rng_below(r, C);
    const uint64_t x_beat = g->io_next[cx];
    g->io_next[cx] += (uint64_t)(T - 1) * xs + xl + rng_below(r, 4);
    put_x(g, cx, x_beat, x, K, T, xs);

    // SPM layout (never reused within a program).
    const uint32_t spm_x = g->spm_next + rng_below(r, 2);
    const uint32_t spm_qa = spm_x + (T - 1) * xs + xl + rng_below(r, 2);
    const uint32_t spm_y = spm_qa + T * ql + rng_below(r, 2);
    const uint32_t y_span = (T - 1) * ys + yl;
    g->spm_next = spm_y + y_span + rng_below(r, 2);

    if (rng_below(r, 2)) {
        pend(g, tsr_enc_ld(cx, spm_x, (uint32_t)x_beat, (T - 1) * xs + xl));
    } else {
        for (uint32_t t = 0; t < T; ++t) pend(g, tsr_enc_ld(cx, spm_x + t * xs, (uint32_t)(x_beat + (uint64_t)t * xs), xl));
    }
    const int zx = xs == xl && rng_below(r, 2);
    const int zy = ys == yl && rng_below(r, 2);
    g->cov[COV_ZSTRIDE] += zx + zy;
    pend(g, tsr_enc_qact(kind, spm_qa, spm_x, K, T, zx ? 0 : xs));

    // Old values for ACC: loaded over the whole dst span (so every dst line is written).
    uint8_t * yspan = (uint8_t *)xmalloc((size_t)y_span * 64);
    if (acc) {
        const uint32_t cy = rng_below(r, C);
        const uint64_t yo = g->io_next[cy];
        g->io_next[cy] += y_span + rng_below(r, 4);
        for (uint32_t i = 0; i < y_span * 16; ++i) {
            float v = (float)rng_normal(r);
            if (rng_below(r, 40) == 0) v = (rng_u32(r) & 1) ? 0.0f : -0.0f;
            put_f32(yspan + 4 * i, v);
        }
        dram_put(g, cy, yo, yspan, (size_t)y_span * 64);
        pend(g, tsr_enc_ld(cy, spm_y, (uint32_t)yo, y_span));
    }
    for (int w = 0; w < 1 + chain; ++w) {
        const uint32_t fl = (acc || w) ? TSR_MVM_FLAG_ACC : 0;
        pend(g, tsr_enc_mvm(fmt, fl, spm_y, spm_qa, (uint32_t)w_off[w], N, K, T, zy ? 0 : ys));
    }
    g->n_mvm += 1 + chain;

    // Expected dst span (tokens' lines; gap lines = loaded old values when acc).
    uint8_t * qa = (uint8_t *)xmalloc((size_t)ql * 64);
    for (uint32_t t = 0; t < T; ++t) {
        tsr_qact(kind, x + (size_t)t * K, K, qa);
        uint8_t * yt = yspan + (size_t)t * ys * 64;
        for (uint32_t n = 0; n < N; ++n) {
            float y = 0.0f;
            for (int w = 0; w < 1 + chain; ++w) {
                tsr_repack_row(fmt, blocks[w] + rbytes * n, K, row);
                const float d = tsr_dot_row(fmt, row, K, qa);
                y = (w == 0 && !acc) ? d : tsr_fadd(w == 0 ? get_f32(yt + 4 * n) : y, d);
            }
            put_f32(yt + 4 * n, y);
        }
        memset(yt + 4 * N, 0, (size_t)yl * 64 - 4 * N);
    }

    // Store: whole span when every line of it was written (acc or no gaps), else per token.
    if (acc || ys == yl) {
        const uint32_t cs = rng_below(r, C);
        const uint64_t ob = g->io_next[cs];
        g->io_next[cs] += y_span + rng_below(r, 4);
        pend(g, tsr_enc_st(cs, spm_y, (uint32_t)ob, y_span));
        add_check(g, cs, ob, y_span, yspan);
    } else {
        for (uint32_t t = 0; t < T; ++t) {
            const uint32_t cs = rng_below(r, C);
            const uint64_t ob = g->io_next[cs];
            g->io_next[cs] += yl + rng_below(r, 4);
            pend(g, tsr_enc_st(cs, spm_y + t * ys, (uint32_t)ob, yl));
            uint8_t * w = (uint8_t *)xmalloc((size_t)yl * 64);
            memcpy(w, yspan + (size_t)t * ys * 64, (size_t)yl * 64);
            add_check(g, cs, ob, yl, w);
        }
        free(yspan);
    }
    free(qa);
    free(x);
    free(row);
    free(blocks[0]);
    free(blocks[1]);
    return 1;
}

// ---------------------------------------------------------------------------
// M2 programs (--ops m2): QACT groups and LD -> ST copies, no MVM (v1)
// Groups: LD -> QACT (-> QACT of the other kind on the same source) -> ST of the QA images, and
// LD -> ST copies (unaligned, 4 KiB-crossing, up to 320 lines, partial re-stores, operands ending
// exactly at the last SPM line / last addressable DRAM beat, zero-length LD/ST exactly at the
// limits).
// ---------------------------------------------------------------------------
// IO allocations grow upward from io_next; "top" allocations grow downward from the channel's
// addressable end (the first one ends exactly at the DRAM limit).
static uint64_t io_room(const gen_t * g, uint32_t c) {
    uint64_t hi = g->top_next[c];
    if (g->o->m3 && hi > g->io_lim) hi = g->io_lim;           // M3: the weight heap sits above IO
    return hi > g->io_next[c] + 128 ? hi - g->io_next[c] - 128 : 0;
}
static uint64_t heap_ceiling(const gen_t * g);
static uint64_t io_alloc(gen_t * g, uint32_t c, uint64_t n) {        // caller checked io_room >= n
    const uint64_t a = g->io_next[c] + rng_below(&g->r, 64);
    g->io_next[c] = a + n;
    return a;
}
static int top_alloc(gen_t * g, uint32_t c, uint64_t n, uint64_t * beat) {
    if (g->o->m3) {                       // M3: above everything the weight heap may still take
        if (g->wlim_pending || g->top_next[c] < heap_ceiling(g) + n + 256) return 0;
    } else if (g->top_next[c] < g->io_next[c] + n + 256) {
        return 0;
    }
    g->top_next[c] -= n;
    *beat = g->top_next[c];
    return 1;
}
static uint32_t spm_room(const gen_t * g) { return g->spm_hi > g->spm_next + 8 ? g->spm_hi - g->spm_next - 8 : 0; }

static uint32_t pick_k_m2(gen_t * g, uint32_t kind, uint32_t kcap, int * edge) {
    rng_t * r = &g->r;
    const uint32_t bs = tsr_qa_block_size(kind);
    const uint32_t nmax = kcap / bs;
    const uint32_t sel = rng_below(r, 16);
    *edge = 0;
    if (sel == 0 || nmax == 0) { *edge = 1; return bs; }
    if (sel == 1) { *edge = 2; return g->o->cfg.k_max; }
    switch (rng_below(r, 3)) {
    case 0: return bs * (1 + rng_below(r, nmax < 8 ? nmax : 8));
    case 1: if (kcap >= 1024) return 1024 * (1 + rng_below(r, kcap / 1024)); /* fallthrough */
    default: return bs * (1 + rng_below(r, nmax));
    }
}

// LD (whole span or per token) of T vectors placed xs lines apart.
static void emit_loads(gen_t * g, uint32_t ch, uint32_t spm, uint64_t beat, uint32_t T, uint32_t xs, uint32_t xl) {
    if (rng_below(&g->r, 2)) {
        pend(g, tsr_enc_ld(ch, spm, (uint32_t)beat, (T - 1) * xs + xl));
    } else {
        for (uint32_t t = 0; t < T; ++t) pend(g, tsr_enc_ld(ch, spm + t * xs, (uint32_t)(beat + (uint64_t)t * xs), xl));
    }
}

// ST of T images of `l` lines each (consecutive in the SPM), whole span or per token, to channel cs.
static void emit_stores(gen_t * g, uint32_t cs, uint32_t spm, const uint8_t * img, uint32_t T, uint32_t l) {
    if (T == 1 || rng_below(&g->r, 2)) {
        const uint64_t ob = io_alloc(g, cs, (uint64_t)T * l);
        g->cov[C2_X4K] += (ob % 64) + (uint64_t)T * l > 64;
        pend(g, tsr_enc_st(cs, spm, (uint32_t)ob, T * l));
        uint8_t * w = (uint8_t *)xmalloc((size_t)T * l * 64);
        memcpy(w, img, (size_t)T * l * 64);
        add_check(g, cs, ob, T * l, w);
    } else {
        for (uint32_t t = 0; t < T; ++t) {
            const uint64_t ob = io_alloc(g, cs, l);
            pend(g, tsr_enc_st(cs, spm + t * l, (uint32_t)ob, l));
            uint8_t * w = (uint8_t *)xmalloc((size_t)l * 64);
            memcpy(w, img + (size_t)t * l * 64, (size_t)l * 64);
            add_check(g, cs, ob, l, w);
        }
    }
}

// LD -> QACT (-> second QACT of the other kind) -> ST. Returns 0 if nothing fits any more.
static int gen_qact_group_m2(gen_t * g) {
    const tsr_config * c = &g->o->cfg;
    rng_t * r = &g->r;
    const uint32_t kcap = g->o->max_k < c->k_max ? g->o->max_k : c->k_max;
    uint32_t kind = 0, K = 0, T = 0, xl = 0, ql[2] = { 0, 0 }, xs = 0, cx = 0, cs = 0;
    int pair = 0, edge = 0, fits = 0;
    for (int attempt = 0; attempt < 32 && !fits; ++attempt) {
        kind = rng_below(r, 2);
        K = pick_k_m2(g, kind, kcap, &edge);
        T = 1 + rng_below(r, c->t_max);
        if (K > 8192 && T > 2) T = 2;                         // keeps traces small
        xl = K / 16;
        ql[0] = tsr_qa_lines(kind, K);
        xs = xl + (rng_below(r, 3) == 0 ? 1 + rng_below(r, 3) : 0);
        pair = rng_below(r, 4) == 0 && K % 256 == 0;
        ql[1] = pair ? tsr_qa_lines(!kind, K) : 0;
        cx = rng_below(r, c->num_ch);
        cs = rng_below(r, c->num_ch);
        const uint64_t span = (uint64_t)(T - 1) * xs + xl;
        const uint64_t out = (uint64_t)T * (ql[0] + ql[1] + 128);
        fits = span + (uint64_t)T * (ql[0] + ql[1]) <= spm_room(g) &&
               io_room(g, cx) >= span + 64 && io_room(g, cs) >= out + (cs == cx ? span + 64 : 0);
    }
    if (!fits) return 0;
    next_group(g);
    g->cov[kind ? C2_A256 : C2_A32]++;
    if (pair) g->cov[kind ? C2_A32 : C2_A256]++;
    g->cov[C2_QPAIR] += pair;
    g->cov[C2_TGT1] += T > 1;
    g->cov[C2_XSTRIDE] += xs != xl;
    g->cov[C2_KMIN] += edge == 1;
    g->cov[C2_KMAX] += edge == 2;

    float * x = (float *)xmalloc(sizeof(float) * K * T);
    int sp = 0;
    for (uint32_t t = 0; t < T; ++t) sp |= gen_x(r, x + (size_t)t * K, K, g->o->special);
    g->cov[C2_SPECIAL] += sp;
    const uint64_t span = (uint64_t)(T - 1) * xs + xl;
    const uint64_t x_beat = io_alloc(g, cx, span);
    put_x(g, cx, x_beat, x, K, T, xs);
    g->cov[C2_X4K] += (x_beat % 64) + span > 64;

    const uint32_t spm_x = g->spm_next + rng_below(r, 2);
    uint32_t spm_qa[2];
    spm_qa[0] = spm_x + (uint32_t)span + rng_below(r, 2);
    spm_qa[1] = spm_qa[0] + T * ql[0] + rng_below(r, 2);
    g->spm_next = spm_qa[1] + T * ql[1] + rng_below(r, 2);

    emit_loads(g, cx, spm_x, x_beat, T, xs, xl);
    for (int v = 0; v < 1 + pair; ++v) {
        const uint32_t kv = v ? !kind : kind;
        const int z = xs == xl && rng_below(r, 2);
        g->cov[C2_ZSTRIDE] += z;
        pend(g, tsr_enc_qact(kv, spm_qa[v], spm_x, K, T, z ? 0 : xs));
    }
    for (int v = 0; v < 1 + pair; ++v) {
        const uint32_t kv = v ? !kind : kind;
        uint8_t * img = (uint8_t *)xmalloc((size_t)T * ql[v] * 64);
        for (uint32_t t = 0; t < T; ++t) tsr_qact(kv, x + (size_t)t * K, K, img + (size_t)t * ql[v] * 64);
        emit_stores(g, cs, spm_qa[v], img, T, ql[v]);
        free(img);
    }
    free(x);
    return 1;
}

// LD -> ST copies, and zero-length LD / ST exactly at the SPM / DRAM limits.
static int gen_copy_group_m2(gen_t * g) {
    const tsr_config * c = &g->o->cfg;
    rng_t * r = &g->r;
    const uint32_t C = c->num_ch;
    if (rng_below(r, 8) == 0) {
        const uint32_t ch = rng_below(r, C);
        const uint32_t dlim = g->lim > 0xFFFFFFFFull ? 0xFFFFFFFFu : (uint32_t)g->lim;
        const uint32_t spm = rng_below(r, 2) ? g->spm_lines : rng_below(r, g->spm_lines + 1);
        const uint32_t dram = (spm != g->spm_lines || rng_below(r, 2)) ? dlim : rng_below(r, dlim);
        next_group(g);
        pend(g, rng_below(r, 2) ? tsr_enc_ld(ch, spm, dram, 0) : tsr_enc_st(ch, spm, dram, 0));
        g->cov[C2_ZERO]++;
        return 1;
    }
    uint32_t n = 0, ca = 0, cb = 0, spm = 0, k = 0;
    int spm_top = 0, src_top = 0, dst_top = 0, fits = 0;
    for (int attempt = 0; attempt < 32 && !fits; ++attempt) {
        n = rng_below(r, 4) == 0 ? 1 + rng_below(r, 64) : 1 + rng_below(r, 320);
        ca = rng_below(r, C);
        cb = rng_below(r, C);
        spm_top = rng_below(r, 6) == 0;
        src_top = rng_below(r, 6) == 0;
        dst_top = rng_below(r, 6) == 0;
        k = rng_below(r, 4) == 0 ? rng_below(r, n) : 0;
        fits = (spm_top ? g->spm_hi >= g->spm_next + n + 16 : spm_room(g) >= n + 2) &&
               io_room(g, ca) >= (uint64_t)n + 64 && io_room(g, cb) >= (uint64_t)n + 128 + (ca == cb ? n + 64 : 0);
    }
    if (!fits) return 0;
    next_group(g);
    g->cov[C2_COPY]++;
    if (spm_top) {
        g->spm_hi -= n;
        spm = g->spm_hi;
        g->cov[C2_SLIM] += spm + n == g->spm_lines;
    } else {
        spm = g->spm_next + rng_below(r, 2);
        g->spm_next = spm + n + rng_below(r, 2);
    }
    uint64_t src = 0, dst = 0;
    if (!src_top || !top_alloc(g, ca, n, &src)) src = io_alloc(g, ca, n);
    uint8_t * data = (uint8_t *)xmalloc((size_t)n * 64);
    rng_bytes(r, data, (size_t)n * 64);
    dram_put(g, ca, src, data, (size_t)n * 64);
    if (!dst_top || !top_alloc(g, cb, n - k, &dst)) dst = io_alloc(g, cb, n - k);
    g->cov[C2_DLIM] += (src + n == g->lim) + (dst + (n - k) == g->lim);
    g->cov[C2_X4K] += ((src % 64) + n > 64) + ((dst % 64) + (n - k) > 64);
    pend(g, tsr_enc_ld(ca, spm, (uint32_t)src, n));
    pend(g, tsr_enc_st(cb, spm + k, (uint32_t)dst, n - k));
    uint8_t * want = (uint8_t *)xmalloc((size_t)(n - k) * 64);
    memcpy(want, data + (size_t)k * 64, (size_t)(n - k) * 64);
    add_check(g, cb, dst, n - k, want);
    free(data);
    return 1;
}

static int gen_group_m2(gen_t * g) {
    if (rng_below(&g->r, 4) == 0) return gen_copy_group_m2(g) || gen_qact_group_m2(g);
    return gen_qact_group_m2(g) || gen_copy_group_m2(g);
}

// ---------------------------------------------------------------------------
// M3 programs (--ops m3): MVM-centred groups mixed with the M2 QACT and LD -> ST copy groups (v1).
// An MVM group is
//
//   LD x -> QACT (or LD of a raw QA image) -> [LD old values / guard lines] -> MVM x n -> ST
//
// with up to two outputs sharing the QA (siblings like ffn_gate / ffn_up), ACC chains of up to
// three MVMs per output, an optional N = 0 MVM in between (must write nothing) and an optional
// second stage QACT(y) -> MVM (an MVM result read back as activations). Shapes cover both formats,
// K = block size / partial last chunk / whole chunks / K_MAX (the activation buffer at full depth,
// also at T = T_MAX through raw QA), N = 0, 1, < C (empty channels), random (a y line holding
// rows of two channels), > 1000; T = 1..T_MAX with dst_stride 0 or explicit with gap lines.
// Values: special activations (gen_x), near-constant "extreme" activations, raw QA images (LD'd:
// full int8 range incl. -128, f32 Dx incl. +-0 / subnormal / Inf / NaN / max finite, garbage in
// the unused tail bytes, BS either the sums of QS or arbitrary int16), weight scales d / dmin that
// are +-0, subnormal, Inf, NaN or +-65504, Q8_0 q = -128, Q4_K 6-bit scales / mins = 63 and
// isum > 2^24 (int -> f32 rounds), ACC old values incl. +-0, subnormals, Inf, NaN payloads.
// Placement: weight streams crossing 4 KiB blocks at random offsets with unread gaps between
// tensors, one tensor per trace (1/4) ending exactly at the DRAM limit, dst / QA regions ending
// exactly at the SPM end, guard lines around dst spans (an MVM that writes outside its token lines
// changes a stored line). Error injections: 3/4 MVM faults (reserved fields, BAD_FMT, K_RANGE,
// T_RANGE, SPM_RANGE on A1 / A2, DRAM_RANGE on A3, each with lower-priority violations in the same
// op, extents that wrap 32 bits), 1/4 the M2 / v2 endings.
// ---------------------------------------------------------------------------
enum { C3_Q8 = C2_N, C3_Q4, C3_K32, C3_K256, C3_KPART, C3_KMAX, C3_KMAXT, C3_N0, C3_N1, C3_NLTC, C3_ODDN,
       C3_NMIX, C3_NBIG, C3_TGT1, C3_TMAX, C3_YZERO, C3_YGAP, C3_GUARD, C3_ACC, C3_ACCSP, C3_CHAIN,
       C3_SHARE, C3_STAGE2, C3_RAWQA, C3_RAWBS, C3_XSP, C3_XEXT, C3_DZ, C3_DSUB, C3_DINF, C3_DNAN,
       C3_DMAX, C3_Q128, C3_SC63, C3_ISUM, C3_W4K, C3_WLIM, C3_N0LIM, C3_YEND, C3_QAEND, C3_ERR, C3_WRAP,
       C3_END };
#define C3_N (C3_END - C2_N)
_Static_assert(C3_END <= COV_V2, "coverage slots overlap");
static const char * cov3_names[C3_N] = {
    "Q8_0 MVMs", "Q4_K MVMs", "K = 32", "K = 256", "partial last chunk", "K = K_MAX", "K = K_MAX at T = T_MAX",
    "N = 0", "N = 1", "N < C (empty channels)", "odd N", "y lines shared by 2 channels", "N > 1000", "T > 1",
    "T = T_MAX", "dst_stride 0", "dst_stride gaps", "guarded dst spans", "ACC (loaded old values)",
    "ACC special old values", "ACC chains", "sibling MVMs sharing a QA", "MVM -> QACT -> MVM", "raw (LD) QA",
    "raw QA with arbitrary BS", "special activations", "extreme activations", "d/dmin +-0", "d/dmin subnormal",
    "d/dmin Inf", "d/dmin NaN", "d/dmin +-65504", "Q8_0 q = -128", "Q4_K sc/m = 63",
    "Q4_K int->f32 rounded isum terms", "4 KiB-crossing weight streams", "tensors ending at the DRAM limit",
    "N = 0 with W_OFF = limit", "dst ending at the SPM end", "QA ending at the SPM end", "MVM error injections",
    "faulting operands whose 32-bit extent wraps" };
_Static_assert(sizeof(cov3_names) / sizeof(cov3_names[0]) == C3_N, "cov3_names");

enum { T3_RSV, T3_FMT, T3_K, T3_T, T3_SPM_DST, T3_SPM_X, T3_DRAM, T3_N };

static uint64_t min_top(const gen_t * g) {
    uint64_t m = g->lim;
    for (uint32_t c = 0; c < g->o->cfg.num_ch; ++c) m = g->top_next[c] < m ? g->top_next[c] : m;
    return m;
}
// Highest beat the weight heap can still reach (remaining budget + per-tensor gaps).
static uint64_t heap_ceiling(const gen_t * g) {
    const uint64_t left = g->wbudget > g->wch0 ? g->wbudget - g->wch0 : 0;
    return g->heap_next + left + 4096;
}

// Symmetric weight allocation (same W_OFF in every channel). Returns W_OFF; *at_lim when the
// tensor (channel 0: R x ROW_BEATS) ends exactly at the DRAM limit.
static uint64_t heap_take(gen_t * g, uint64_t n, int * at_lim) {
    *at_lim = 0;
    if (g->wlim_pending && n > 0) {
        g->wlim_pending = 0;
        const uint64_t w = g->lim - n;
        for (uint32_t c = 0; c < g->o->cfg.num_ch; ++c) if (g->top_next[c] > w) g->top_next[c] = w;
        *at_lim = 1;
        return w;
    }
    const uint64_t w = g->heap_next + (rng_below(&g->r, 4) ? rng_below(&g->r, 16) : 0);
    g->heap_next = w + n + (rng_below(&g->r, 4) ? 1 + rng_below(&g->r, 16) : 0);
    return w;
}

// SPM region of n lines from the bottom (spm_next, 0-1 line gap) or the top (spm_hi).
static uint32_t spm_take(gen_t * g, uint32_t n, int top) {
    if (top) {
        g->spm_hi -= n;
        return g->spm_hi;
    }
    const uint32_t a = g->spm_next + rng_below(&g->r, 2);
    g->spm_next = a + n;
    return a;
}

// fp16 weight scale with a special value: +-0, subnormal, Inf, NaN, +-65504, min normal, large.
static uint16_t fp16_special(rng_t * r) {
    const uint16_t sign = (uint16_t)((rng_u32(r) & 1u) << 15);
    switch (rng_below(r, 7)) {
    case 0:  return sign;
    case 1:  return (uint16_t)(sign | (1 + rng_below(r, 0x3FF)));
    case 2:  return (uint16_t)(sign | 0x7C00);
    case 3:  return (uint16_t)(sign | 0x7C00 | (1 + rng_below(r, 0x3FF)));
    case 4:  return (uint16_t)(sign | 0x7BFF);
    case 5:  return (uint16_t)(sign | 0x0400);
    default: return (uint16_t)(sign | ((25 + rng_below(r, 6)) << 10) | rng_below(r, 1024));
    }
}

enum { WF_DZ = 1, WF_DSUB = 2, WF_DINF = 4, WF_DNAN = 8, WF_DMAX = 16, WF_Q128 = 32, WF_SC63 = 64 };
static uint32_t fp16_class(uint16_t h) {
    const uint32_t e = (h >> 10) & 0x1Fu, m = h & 0x3FFu;
    if (e == 0) return m ? WF_DSUB : WF_DZ;
    if (e == 31) return m ? WF_DNAN : WF_DINF;
    return (h & 0x7FFF) == 0x7BFF ? WF_DMAX : 0;
}

// N rows of ggml blocks. mode 0: random bytes, fp16 scales as M1 (rare +-0 / subnormal);
// 1: a quarter of the rows get special scales in 1/8 of their blocks; 2: extreme rows (Q8_0
// quants -128 / -127 / 127, Q4_K nibbles 14 / 15 with every 6-bit scale and min = 63).
// Returns WF_* flags of what the tensor contains.
static uint32_t gen_tensor_m3(gen_t * g, uint32_t fmt, uint32_t N, uint32_t K, int mode, uint8_t * blocks) {
    rng_t * r = &g->r;
    const uint32_t nb = K / tsr_fmt_block_size(fmt);
    const size_t bsz = tsr_fmt_ggml_block_bytes(fmt);
    uint32_t fl = 0;
    for (uint32_t n = 0; n < N; ++n) {
        const int srow = mode == 1 && rng_below(r, 4) == 0;
        for (uint32_t b = 0; b < nb; ++b) {
            uint8_t * blk = blocks + ((size_t)n * nb + b) * bsz;
            rng_bytes(r, blk, bsz);
            if (mode == 2 && fmt == TSR_FMT_Q8_0) {
                for (int l = 0; l < 32; ++l) blk[2 + l] = rng_below(r, 4) ? 0x80 : (uint8_t)(0x7F + rng_below(r, 3));
            } else if (mode == 2) {
                static const uint8_t nib[4] = { 0xFF, 0xFF, 0xEF, 0xFE };
                memset(blk + 4, 0xFF, 12);
                for (int l = 0; l < 128; ++l) blk[16 + l] = nib[rng_below(r, 4)];
            }
            const int nsc = fmt == TSR_FMT_Q4_K ? 2 : 1;
            for (int k = 0; k < nsc; ++k) {
                uint16_t d = (srow && rng_below(r, 8) == 0) ? fp16_special(r)
                                                            : rand_fp16_scale(r, g->o->special && mode != 2);
                if (rng_below(r, 4) == 0 && (d & 0x7C00) != 0 && (d & 0x7C00) != 0x7C00)   // normal: exponent 1..30
                    d = (uint16_t)((d & 0x83FF) | ((1 + rng_below(r, 30)) << 10));
                blk[2 * k] = (uint8_t)d;
                blk[2 * k + 1] = (uint8_t)(d >> 8);
                fl |= fp16_class(d);
            }
            if (fmt == TSR_FMT_Q8_0) {
                for (int l = 0; l < 32; ++l) if (blk[2 + l] == 0x80) fl |= WF_Q128;
            } else {
                for (int j = 0; j < 8; ++j) {
                    uint8_t sc, m;
                    tsr_get_scale_min_k4(j, blk + 4, &sc, &m);
                    if (sc == 63 || m == 63) fl |= WF_SC63;
                }
            }
        }
    }
    return fl;
}

// Q4_K isum of ggml block blk against 256 activation int8 (ggml nibble order, written from
// dequantize_row_q4_K / get_scale_min_k4, not via tsr_ref): coverage of int -> f32 rounding.
static int32_t q4k_isum(const uint8_t * blk, const int8_t * qx) {
    const uint8_t * sc = blk + 4;
    const uint8_t * qs = blk + 16;
    int32_t isum = 0;
    for (int j = 0; j < 8; ++j) {
        const int32_t s = j < 4 ? (sc[j] & 63) : ((sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4));
        int32_t S = 0;
        for (int l = 0; l < 32; ++l) {
            const uint8_t b = qs[32 * (j / 2) + l];
            S += (int32_t)((j & 1) ? (b >> 4) : (b & 0xF)) * qx[32 * j + l];
        }
        isum += s * S;
    }
    return isum;
}

// Near-constant activations: |x| within 6% of the block maximum, one sign per 256 elements, so
// QACT gives q of one sign near +-127 (Q4_K isum up to ~3e7 > 2^24 with extreme weights).
static void gen_x_extreme(rng_t * r, float * x, uint32_t K) {
    const double scale = ldexp(1.0, (int)rng_below(r, 16) - 8);
    double s = scale;
    for (uint32_t i = 0; i < K; ++i) {
        if (i % 256 == 0) s = (rng_u32(r) & 1) ? scale : -scale;
        x[i] = (float)(s * (1.0 - 0.06 * rng_unit(r)));
    }
}

static uint32_t rand_dx_bits(rng_t * r, int * sp) {
    const uint32_t sign = rng_u32(r) & 0x80000000u;
    const uint32_t k = rng_below(r, 20);
    if (k <= 6) *sp = 1;
    switch (k) {
    case 0:  return sign;                                              // +-0
    case 1:
    case 2:  return sign | (1 + rng_below(r, 0x7FFFFF));               // subnormal (DAZ)
    case 3:  return sign | 0x7F800000u;                                // +-Inf
    case 4:  return sign | (0x7F800001u + rng_below(r, 0x7FFFFE));     // NaN, quiet or signalling
    case 5:  return sign | 0x7F7FFFFFu;                                // max finite (products overflow)
    case 6:  return sign | 0x00800000u;                                // min normal (products flush)
    default: return sign | ((107u + rng_below(r, 21)) << 23) | rng_below(r, 1u << 23);   // 2^-20 .. 2^0
    }
}

// Raw QA image (loaded with LD instead of produced by QACT): every byte random first (unused tail
// bytes stay garbage), then QS blocks of all -128 / all 127 / small / random values (extreme:
// -128..-120), Dx incl. special values, BS = sums of QS (natural) or arbitrary int16 (arb_bs).
// Returns 1 if any Dx is special.
static int gen_raw_qa(rng_t * r, uint32_t kind, uint32_t K, uint8_t * qa, int arb_bs, int extreme) {
    const uint32_t ql = tsr_qa_lines(kind, K);
    rng_bytes(r, qa, (size_t)ql * 64);
    int8_t * qs = (int8_t *)qa;
    for (uint32_t b = 0; b < K / 32; ++b) {
        int8_t * q = qs + 32 * b;
        const uint32_t m = extreme ? 99 : rng_below(r, 12);
        for (int l = 0; l < 32; ++l) {
            if (m == 99) q[l] = (int8_t)(-128 + (int)rng_below(r, 9));
            else if (m == 0) q[l] = -128;
            else if (m == 1) q[l] = 127;
            else if (m == 2) q[l] = (int8_t)((int)rng_below(r, 5) - 2);
        }
    }
    int sp = 0;
    uint8_t * dreg = qa + (size_t)tsr_qa_qs_lines(kind, K) * 64;
    const uint32_t bsz = tsr_qa_block_size(kind);
    for (uint32_t b = 0; b < K / bsz; ++b) {
        const uint32_t v = extreme ? f2u((float)ldexp(1.0, -(int)rng_below(r, 12))) : rand_dx_bits(r, &sp);
        put_f32(dreg + 4 * b, u2f(v));
    }
    if (kind == TSR_QA_A256) {
        uint8_t * bs = dreg + (size_t)tsr_qa_d_lines(kind, K) * 64;
        for (uint32_t j = 0; j < K / 16; ++j) {
            int32_t v;
            if (arb_bs) {
                const uint32_t k = rng_below(r, 8);
                v = k == 0 ? 32767 : k == 1 ? -32768 : (int32_t)(int16_t)rng_u32(r);
            } else {
                v = 0;
                for (int l = 0; l < 16; ++l) v += qs[16 * j + l];
            }
            bs[2 * j] = (uint8_t)v;
            bs[2 * j + 1] = (uint8_t)((uint32_t)v >> 8);
        }
    }
    return sp;
}

// Old value for ACC / guard lines: mostly N(0,1), 1/20 special.
static uint32_t old_bits(rng_t * r, int * sp) {
    if (rng_below(r, 20) != 0) return f2u((float)rng_normal(r));
    *sp = 1;
    const uint32_t sign = rng_u32(r) & 0x80000000u;
    switch (rng_below(r, 6)) {
    case 0:  return sign;                                              // +-0
    case 1:  return sign | (1 + rng_below(r, 0x7FFFFF));               // subnormal
    case 2:  return sign | 0x7F800000u;                                // +-Inf
    case 3:  return sign | 0x7FC00000u | rng_below(r, 1u << 22);       // quiet NaN with payload
    case 4:  return sign | (0x7F800001u + rng_below(r, 0x3FFFFE));     // signalling NaN
    default: return sign | 0x7F7FFFFFu;                                // max finite
    }
}

typedef struct {
    uint64_t  w_off;
    uint8_t * blocks;              // N rows of ggml blocks
} m3_tensor;

typedef struct {                   // one MVM output (dst span) and the MVMs that write it
    uint32_t  fmt, K, N, R, rb, yl, ys, w6, n_w, acc, pre, g0, g1, top;
    uint32_t  spm;                 // first line of the region; dst = spm + g0
    uint32_t  lines;               // g0 + span + g1
    uint32_t  span;                // (T - 1) * ys + yl
    m3_tensor w[3];
    uint8_t * img;                 // expected region content (lines * 64 bytes)
} m3_out;

static uint32_t pick_n_m3(gen_t * g, int small) {
    rng_t * r = &g->r;
    const uint32_t C = g->o->cfg.num_ch;
    if (small) return rng_below(r, 2 * C + 2);
    switch (rng_below(r, 16)) {
    case 0:  return 0;
    case 1:  return 1;
    case 2:  return C > 1 ? 1 + rng_below(r, C - 1) : 1 + rng_below(r, 3);
    case 3:  return 1001 + rng_below(r, 3000);
    default: return 1 + rng_below(r, g->o->max_n);
    }
}

static void out_shape(gen_t * g, m3_out * o, uint32_t fmt, uint32_t K, uint32_t N, uint32_t T, int may_chain) {
    rng_t * r = &g->r;
    memset(o, 0, sizeof(*o));
    o->fmt = fmt;
    o->K = K;
    o->N = N;
    o->R = tsr_ceil_div_u32(N, g->o->cfg.num_ch);
    o->rb = tsr_row_beats(fmt, K);
    o->yl = tsr_ceil_div_u32(4 * N, 64);
    o->ys = o->yl + (rng_below(r, 3) == 0 ? 1 + rng_below(r, 3) : 0);
    o->w6 = (o->ys == o->yl && rng_below(r, 2)) ? 0 : o->ys;
    o->n_w = (may_chain && rng_below(r, 4) == 0) ? 2 + rng_below(r, 2) : 1;
    o->acc = rng_below(r, 3) == 0;
    o->pre = o->acc || N == 0 || rng_below(r, 2) == 0;
    o->g0 = o->pre ? rng_below(r, 3) : 0;
    o->g1 = o->pre ? rng_below(r, 3) + (N == 0) : 0;
    o->top = rng_below(r, 8) == 0;
    o->span = (T - 1) * o->ys + o->yl;
    o->lines = o->g0 + o->span + o->g1;
}

// Allocates and fills the tensors of output o (weights written to every channel's DRAM).
static uint32_t out_tensors(gen_t * g, m3_out * o, int mode) {
    const size_t rbytes = (size_t)(o->K / tsr_fmt_block_size(o->fmt)) * tsr_fmt_ggml_block_bytes(o->fmt);
    uint32_t fl = 0;
    for (uint32_t w = 0; w < o->n_w; ++w) {
        const uint64_t n = (uint64_t)o->R * o->rb;
        int at_lim;
        o->w[w].w_off = heap_take(g, n, &at_lim);
        g->cov[C3_WLIM] += at_lim;
        g->cov[C3_W4K] += n > 0 && (o->w[w].w_off % 64) + n > 64;
        g->wch0 += n;
        g->wall += (uint64_t)o->N * o->rb;
        o->w[w].blocks = (uint8_t *)xmalloc(rbytes * o->N);
        fl |= gen_tensor_m3(g, o->fmt, o->N, o->K, mode, o->w[w].blocks);
        put_tensor(g, o->fmt, o->N, o->K, o->w[w].w_off, o->w[w].blocks);
    }
    return fl;
}

// Expected region content of output o after its MVMs, from the T QA images (stride ql bytes*64).
static void out_expect(gen_t * g, m3_out * o, const uint8_t * qa, uint32_t ql, uint32_t T) {
    const size_t rbytes = (size_t)(o->K / tsr_fmt_block_size(o->fmt)) * tsr_fmt_ggml_block_bytes(o->fmt);
    uint8_t * row = (uint8_t *)xmalloc((size_t)o->rb * 64 + 64);
    for (uint32_t t = 0; t < T; ++t) {
        const uint8_t * qt = qa + (size_t)t * ql * 64;
        uint8_t * yt = o->img + ((size_t)o->g0 + (size_t)t * o->ys) * 64;
        for (uint32_t n = 0; n < o->N; ++n) {
            float y = 0.0f;
            for (uint32_t w = 0; w < o->n_w; ++w) {
                const uint8_t * blk = o->w[w].blocks + rbytes * n;
                tsr_repack_row(o->fmt, blk, o->K, row);
                const float d = tsr_dot_row(o->fmt, row, o->K, qt);
                y = (w == 0 && !o->acc) ? d : tsr_fadd(w == 0 ? get_f32(yt + 4 * n) : y, d);
                if (o->fmt == TSR_FMT_Q4_K) {
                    for (uint32_t s = 0; s < o->K / 256; ++s) {
                        const int32_t is = q4k_isum(blk + (size_t)s * 144, (const int8_t *)qt + 256 * s);
                        if ((int64_t)(float)is != (int64_t)is) g->cov[C3_ISUM]++;
                    }
                }
            }
            put_f32(yt + 4 * n, y);
        }
        memset(yt + 4 * (size_t)o->N, 0, (size_t)o->yl * 64 - 4 * (size_t)o->N);
    }
    free(row);
}

// Region preload (guards, gap lines, ACC old values) from channel cy; image = loaded content.
static void out_preload(gen_t * g, m3_out * o, uint32_t cy) {
    int sp = 0;
    for (uint32_t i = 0; i < o->lines * 16; ++i) put_f32(o->img + 4 * i, u2f(old_bits(&g->r, &sp)));
    g->cov[C3_ACCSP] += o->acc && sp;
    g->cov[C3_GUARD] += o->g0 + o->g1 > 0;
    const uint64_t b = io_alloc(g, cy, o->lines);
    dram_put(g, cy, b, o->img, (size_t)o->lines * 64);
    pend(g, tsr_enc_ld(cy, o->spm, (uint32_t)b, o->lines));
}

static void out_store(gen_t * g, m3_out * o, uint32_t T) {
    rng_t * r = &g->r;
    const uint32_t C = g->o->cfg.num_ch;
    if (o->pre || (o->ys == o->yl && (T == 1 || rng_below(r, 2)))) {
        if (o->lines == 0) return;
        const uint32_t cs = rng_below(r, C);
        const uint64_t ob = io_alloc(g, cs, o->lines);
        pend(g, tsr_enc_st(cs, o->spm, (uint32_t)ob, o->lines));
        add_check(g, cs, ob, o->lines, o->img);
        o->img = NULL;
        return;
    }
    for (uint32_t t = 0; t < T; ++t) {
        if (o->yl == 0) break;
        const uint32_t cs = rng_below(r, C);
        const uint64_t ob = io_alloc(g, cs, o->yl);
        pend(g, tsr_enc_st(cs, o->spm + o->g0 + t * o->ys, (uint32_t)ob, o->yl));
        uint8_t * w = (uint8_t *)xmalloc((size_t)o->yl * 64);
        memcpy(w, o->img + ((size_t)o->g0 + (size_t)t * o->ys) * 64, (size_t)o->yl * 64);
        add_check(g, cs, ob, o->yl, w);
    }
}

static void emit_mvm_m3(gen_t * g, const m3_out * o, uint32_t w, uint32_t qa, uint32_t T) {
    const uint32_t fl = (o->acc || w > 0) ? TSR_MVM_FLAG_ACC : 0;
    pend(g, tsr_enc_mvm(o->fmt, fl, o->spm + o->g0, qa, (uint32_t)o->w[w].w_off, o->N, o->K, T, o->w6));
    g->n_mvm++;
}

static void out_cov(gen_t * g, const m3_out * o, uint32_t T, uint32_t fl) {
    const uint32_t C = g->o->cfg.num_ch;
    g->cov[o->fmt ? C3_Q4 : C3_Q8] += o->n_w;
    g->cov[C3_N0] += o->N == 0;
    g->cov[C3_N1] += o->N == 1;
    g->cov[C3_NLTC] += o->N > 0 && o->N < C;
    g->cov[C3_ODDN] += o->N & 1;
    g->cov[C3_NMIX] += C > 1 && o->N > o->R && o->N % 16 && o->R % 16;
    g->cov[C3_NBIG] += o->N > 1000;
    g->cov[C3_TGT1] += T > 1;
    g->cov[C3_TMAX] += T == g->o->cfg.t_max;
    g->cov[C3_YZERO] += o->w6 == 0;
    g->cov[C3_YGAP] += o->ys != o->yl && T > 1;
    g->cov[C3_ACC] += o->acc;
    g->cov[C3_CHAIN] += o->n_w > 1;
    g->cov[C3_YEND] += o->spm + o->g0 + o->span == g->spm_lines;
    g->cov[C3_DZ] += (fl & WF_DZ) != 0;
    g->cov[C3_DSUB] += (fl & WF_DSUB) != 0;
    g->cov[C3_DINF] += (fl & WF_DINF) != 0;
    g->cov[C3_DNAN] += (fl & WF_DNAN) != 0;
    g->cov[C3_DMAX] += (fl & WF_DMAX) != 0;
    g->cov[C3_Q128] += (fl & WF_Q128) != 0;
    g->cov[C3_SC63] += (fl & WF_SC63) != 0;
}

// One MVM group. Returns 0 if nothing fits any more.
static int gen_mvm_group_m3(gen_t * g) {
    const tsr_config * c = &g->o->cfg;
    rng_t * r = &g->r;
    const uint32_t C = c->num_ch;
    const uint32_t kcap = g->o->max_k < c->k_max ? g->o->max_k : c->k_max;
    const uint64_t wleft = g->wbudget > g->wch0 ? g->wbudget - g->wch0 : 0;
    const int big = g->wbudget > g->o->max_wbeats && g->wch0 == 0;   // first group of a big trace: one long stream
    uint32_t fmt = 0, K = 0, T = 0, kind = 0, ql = 0, n_out = 0, xl = 0, xs = 0, kclass = 0;
    uint32_t kind2 = 0, ql2 = 0;
    int raw = 0, st2 = 0, qa_top = 0, fits = 0;
    m3_out out[3];                      // out[2] = stage 2
    for (int attempt = 0; attempt < 64 && !fits; ++attempt) {
        fmt = (kcap >= 256 && rng_below(r, 2)) ? TSR_FMT_Q4_K : TSR_FMT_Q8_0;
        const uint32_t bs = tsr_fmt_block_size(fmt), S = tsr_fmt_chunk_blocks(fmt);
        const uint32_t nmax = kcap / bs ? kcap / bs : 1;
        T = 1 + rng_below(r, c->t_max);
        uint32_t N = pick_n_m3(g, 0);
        const uint32_t ksel = rng_below(r, 16);
        kclass = 0;
        if (N > 1000) {
            K = bs * (1 + rng_below(r, nmax < 2 ? nmax : 2));
        } else if (ksel <= 1 || nmax == 1) {
            K = bs;
            kclass = 1;
        } else if (ksel == 2) {
            K = c->k_max;                                       // activation buffer at full depth
            kclass = 2;
            N = pick_n_m3(g, 1);
            if (rng_below(r, 2)) T = c->t_max;                   // every token slot of the buffer
        } else if (ksel <= 5) {                                  // partial last chunk
            uint32_t nb = 1 + rng_below(r, nmax);
            if (nb % S == 0) nb -= 1;
            if (nb == 0) nb = 1;
            K = bs * nb;
            kclass = 3;
        } else if (ksel <= 7 && kcap >= 1024) {
            K = 1024 * (1 + rng_below(r, kcap / 1024));
        } else {
            K = bs * (1 + rng_below(r, (ksel & 1) && nmax > 8 ? 8 : nmax));
        }
        if (big) {                      // 50-90 % of the budget in one MVM, K up to 5120
            K = bs * (1 + rng_below(r, nmax < 5120 / bs ? nmax : 5120 / bs));
            const uint64_t n = (uint64_t)C * (wleft * (5 + rng_below(r, 5)) / 10 / tsr_row_beats(fmt, K));
            const uint64_t nspm = 16ull * (spm_room(g) / 4) / T;        // y span stays below a quarter of the SPM
            N = (uint32_t)(n < nspm ? n : nspm);
            kclass = 0;
        }
        kind = tsr_qa_kind_for_fmt(fmt);
        ql = tsr_qa_lines(kind, K);
        st2 = !big && kclass != 2 && N <= 1000 && c->k_max >= 512 && rng_below(r, 6) == 0;
        if (st2) N = (fmt == TSR_FMT_Q4_K || rng_below(r, 2)) ? 256 * (1 + rng_below(r, 2)) : 32 * (1 + rng_below(r, 8));
        n_out = big ? 1 : 1 + (rng_below(r, 4) == 0);
        out_shape(g, &out[0], fmt, K, N, T, !big);
        if (n_out > 1) out_shape(g, &out[1], fmt, K, pick_n_m3(g, kclass == 2), T, 1);
        if (st2) {
            kind2 = (N % 256 == 0 && rng_below(r, 2)) ? TSR_QA_A256 : TSR_QA_A32;
            ql2 = tsr_qa_lines(kind2, N);
            out_shape(g, &out[2], kind2, N, 1 + rng_below(r, g->o->max_n), T, 0);
        }
        raw = rng_below(r, 5) == 0 || (kclass == 2 && T > 2);
        qa_top = rng_below(r, 8) == 0;
        xl = K / 16;
        xs = xl + (rng_below(r, 3) == 0 ? 1 + rng_below(r, 3) : 0);
        const uint64_t xspan = (uint64_t)(T - 1) * xs + xl;
        uint64_t wneed = 0, heap = 0, spm = (raw ? 0 : xspan + 2) + (uint64_t)T * ql + 2 + 16, io = (raw ? (uint64_t)T * ql : xspan) + 1024;
        for (uint32_t k = 0; k < 3; ++k) {
            if (k == 1 && n_out < 2) continue;
            if (k == 2 && !st2) continue;
            const m3_out * o = &out[k];
            wneed += (uint64_t)o->n_w * o->R * o->rb;
            heap += (uint64_t)o->n_w * (o->R * (uint64_t)o->rb + 40);
            spm += o->lines + 2 + (k == 2 ? (uint64_t)T * ql2 + 2 : 0);
            io += 2ull * o->lines + (uint64_t)T * o->yl + 512;
        }
        uint64_t io_min = ~0ull;
        for (uint32_t ch = 0; ch < C; ++ch) io_min = io_room(g, ch) < io_min ? io_room(g, ch) : io_min;
        fits = wneed <= wleft && g->heap_next + heap + 64 <= min_top(g) && spm <= spm_room(g) && io <= io_min;
    }
    if (!fits) return 0;
    next_group(g);
    const int extreme = rng_below(r, 8) == 0;
    const int mode = extreme ? 2 : (rng_below(r, 4) == 0 ? 1 : 0);
    g->cov[C3_K32] += K == 32;
    g->cov[C3_K256] += K == 256;
    g->cov[C3_KPART] += kclass == 3 || (K / tsr_fmt_block_size(fmt)) % tsr_fmt_chunk_blocks(fmt) != 0;
    g->cov[C3_KMAX] += K == c->k_max;
    g->cov[C3_KMAXT] += K == c->k_max && T == c->t_max;
    g->cov[C3_SHARE] += n_out > 1;
    g->cov[C3_STAGE2] += st2;
    g->cov[C3_RAWQA] += raw;

    // Weights.
    uint32_t wfl[3] = { 0, 0, 0 };
    for (uint32_t k = 0; k < 3; ++k) {
        if ((k == 1 && n_out < 2) || (k == 2 && !st2)) continue;
        wfl[k] = out_tensors(g, &out[k], mode);
    }

    // Activations: LD x -> QACT, or LD of raw QA images.
    uint8_t * qa = (uint8_t *)xmalloc((size_t)T * ql * 64);
    uint32_t spm_qa;
    if (raw) {
        const int arb = kind == TSR_QA_A256 && rng_below(r, 2) == 0;
        int sp = 0;
        for (uint32_t t = 0; t < T; ++t) sp |= gen_raw_qa(r, kind, K, qa + (size_t)t * ql * 64, arb, extreme);
        g->cov[C3_RAWBS] += arb;
        g->cov[C3_XSP] += sp;
        g->cov[C3_XEXT] += extreme;
        const uint32_t cq = rng_below(r, C);
        const uint64_t qb = io_alloc(g, cq, (uint64_t)T * ql);
        dram_put(g, cq, qb, qa, (size_t)T * ql * 64);
        spm_qa = spm_take(g, T * ql, qa_top);
        emit_loads(g, cq, spm_qa, qb, T, ql, ql);
    } else {
        float * x = (float *)xmalloc(sizeof(float) * K * T);
        int sp = 0;
        for (uint32_t t = 0; t < T; ++t) {
            if (extreme) gen_x_extreme(r, x + (size_t)t * K, K);
            else sp |= gen_x(r, x + (size_t)t * K, K, g->o->special);
        }
        g->cov[C3_XSP] += sp;
        g->cov[C3_XEXT] += extreme;
        for (uint32_t t = 0; t < T; ++t) tsr_qact(kind, x + (size_t)t * K, K, qa + (size_t)t * ql * 64);
        const uint32_t cx = rng_below(r, C);
        const uint64_t span = (uint64_t)(T - 1) * xs + xl;
        const uint64_t xb = io_alloc(g, cx, span);
        put_x(g, cx, xb, x, K, T, xs);
        const uint32_t spm_x = spm_take(g, (uint32_t)span, 0);
        spm_qa = spm_take(g, T * ql, qa_top);
        emit_loads(g, cx, spm_x, xb, T, xs, xl);
        const int zx = xs == xl && rng_below(r, 2);
        g->cov[C2_ZSTRIDE] += zx;
        pend(g, tsr_enc_qact(kind, spm_qa, spm_x, K, T, zx ? 0 : xs));
        g->cov[C2_A32 + kind]++;
        free(x);
    }
    g->cov[C3_QAEND] += spm_qa + T * ql == g->spm_lines;

    // Output regions (preloaded with old values / guard lines where o->pre).
    for (uint32_t k = 0; k < 3; ++k) {
        if ((k == 1 && n_out < 2) || (k == 2 && !st2)) continue;
        m3_out * o = &out[k];
        o->spm = spm_take(g, o->lines, o->top);
        o->img = (uint8_t *)xmalloc((size_t)o->lines * 64 + 64);
        memset(o->img, 0, (size_t)o->lines * 64);
        if (o->pre) out_preload(g, o, rng_below(r, C));
        out_cov(g, o, T, wfl[k]);
    }

    // MVMs of out[0] / out[1], interleaved (each output's ACC chain stays in order).
    uint32_t done[2] = { 0, 0 };
    while (done[0] < out[0].n_w || (n_out > 1 && done[1] < out[1].n_w)) {
        uint32_t k = (n_out > 1 && done[1] < out[1].n_w && (done[0] == out[0].n_w || rng_below(r, 2))) ? 1 : 0;
        emit_mvm_m3(g, &out[k], done[k]++, spm_qa, T);
    }
    for (uint32_t k = 0; k < n_out; ++k) out_expect(g, &out[k], qa, ql, T);

    // N = 0 MVM in between: writes nothing (dst over out[0]'s span, or at the SPM end).
    if (rng_below(r, 10) == 0) {
        const int at_end = rng_below(r, 3) == 0;
        const uint32_t w6 = at_end ? 0 : out[0].w6 ? out[0].ys : 0;
        const uint32_t dst = at_end ? g->spm_lines : out[0].spm + out[0].g0;
        const int wl = rng_below(r, 2);
        const uint32_t woff = wl ? (uint32_t)g->lim : (uint32_t)(g->w_lo + rng_below(r, 1u << 20));
        pend(g, tsr_enc_mvm(fmt, rng_below(r, 2) ? TSR_MVM_FLAG_ACC : 0, dst, spm_qa, woff, 0, K, T, w6));
        g->cov[C3_N0]++;
        g->cov[C3_N0LIM] += wl;
        g->cov[fmt ? C3_Q4 : C3_Q8]++;
        g->n_mvm++;
    }

    // Stage 2: QACT of out[0]'s y (K2 = N0, src_stride = its dst stride) -> MVM -> out[2].
    if (st2) {
        const m3_out * o0 = &out[0];
        m3_out * o2 = &out[2];
        const uint32_t K2 = o0->N;
        float * x2 = (float *)xmalloc(sizeof(float) * K2 * T);
        for (uint32_t t = 0; t < T; ++t)
            for (uint32_t k = 0; k < K2; ++k) x2[(size_t)t * K2 + k] = get_f32(o0->img + ((size_t)o0->g0 + (size_t)t * o0->ys) * 64 + 4 * k);
        uint8_t * qa2 = (uint8_t *)xmalloc((size_t)T * ql2 * 64);
        for (uint32_t t = 0; t < T; ++t) tsr_qact(kind2, x2 + (size_t)t * K2, K2, qa2 + (size_t)t * ql2 * 64);
        const uint32_t spm_qa2 = spm_take(g, T * ql2, 0);
        pend(g, tsr_enc_qact(kind2, spm_qa2, o0->spm + o0->g0, K2, T, o0->w6 ? o0->ys : 0));
        g->cov[C2_A32 + kind2]++;
        emit_mvm_m3(g, o2, 0, spm_qa2, T);
        out_expect(g, o2, qa2, ql2, T);
        free(qa2);
        free(x2);
    }

    for (uint32_t k = 0; k < 3; ++k) {
        if ((k == 1 && n_out < 2) || (k == 2 && !st2)) continue;
        out_store(g, &out[k], T);
        free(out[k].img);
        for (uint32_t w = 0; w < out[k].n_w; ++w) free(out[k].w[w].blocks);
    }
    free(qa);
    return 1;
}

static int gen_group_m3(gen_t * g, uint32_t k) {
    if (k == 0) return gen_mvm_group_m3(g);
    switch (rng_below(&g->r, 8)) {
    case 0:  return gen_copy_group_m2(g) || gen_mvm_group_m3(g);
    case 1:  return gen_qact_group_m2(g) || gen_mvm_group_m3(g);
    default: return gen_mvm_group_m3(g) || gen_copy_group_m2(g);
    }
}

// ---------------------------------------------------------------------------
// Error injection (EX §8.1 codes for tile ops, CC §8.3 BAD_DEP, CC §9.3 propagation)
// ---------------------------------------------------------------------------
// Pushes a faulting op whose code and ERR_INFO the generator derived from the spec, and checks that
// the simulator's checker (with the BAD_DEP rule) agrees. Returns the record index.
static uint32_t emit_fault(gen_t * g, const uint32_t w[8], uint32_t x0, uint32_t x1, uint32_t want, int64_t want_info) {
    const uint32_t idx = emit_op(g, w, x0, x1, EMIT_NODEPS);
    const rec_t * r = &g->rec[idx];
    if (r->want != want || (want_info >= 0 && r->info != (uint32_t)want_info)) {
        fprintf(stderr, "tsr-trace-gen: injected fault w0=0x%08x deps 0x%08x,0x%08x: expected code %u (info %lld), "
                        "the checker / BAD_DEP rule gives %u (info 0x%x)\n",
                w[0], x0, x1, want, (long long)want_info, r->want, r->info);
        exit(1);
    }
    if (g->inj_pidx == NOFAIL) g->inj_pidx = r->pidx;
    return idx;
}

// A legal op of unit u that may never run (ABORTED / DEP_FAILED tails): an earlier op of that unit
// that ran, else a small constructed one. Returns 0 if the unit takes no op in this configuration.
static int some_op(gen_t * g, uint32_t u, uint32_t w[8]) {
    uint32_t cand = 0, n = 0;
    for (uint32_t i = g->good_lo; i < g->n_rec; ++i) {
        const rec_t * r = &g->rec[i];
        if (r->kind == TSR_TRACE_PUSH_OP && r->ran && TK_U(r->ticket) == u && rng_below(&g->r, ++n) == 0) cand = i + 1;
    }
    if (cand) {
        memcpy(w, g->rec[cand - 1].w, 32);
        return 1;
    }
    tsr_insn i = {{0}};
    switch (u) {
    case TSR_UNIT_DMA:  i = rng_below(&g->r, 2) ? tsr_enc_ld(0, 0, 0, 1) : tsr_enc_st(0, 0, 0, 1); break;
    case TSR_UNIT_QACT: i = tsr_enc_qact(TSR_QA_A32, 0, 4, 32, 1, 0); break;
    case TSR_UNIT_MVM:  i = tsr_enc_mvm(TSR_FMT_Q8_0, 0, 0, 4, 0, 1, 32, 1, 0); break;
    case TSR_UNIT_VU:
        i.w[0] = TSR_OP_VOP;
        TSR_INSN_SET(i.w, TSR_VOP_FUNC, TSR_VOP_FUNC_COPY);
        i.w[1] = 0; i.w[2] = 4; i.w[5] = 16; i.w[6] = 1;
        break;
    case TSR_UNIT_SYNC: i = tsr_enc_nop(); break;
    default: return 0;
    }
    uint32_t uu, info;
    if (classify(g, i.w, &uu, &info, NULL) != TSR_ERR_NONE || uu != u) return 0;
    memcpy(w, i.w, 32);
    return 1;
}

// After a fault on unit u: 1-2 more entries of u (they complete ABORTED, the unit being in error
// state, EX §8.2).
static void after_error(gen_t * g, uint32_t u) {
    const uint32_t n = 1 + rng_below(&g->r, 2);
    for (uint32_t k = 0; k < n; ++k) {
        uint32_t w[8];
        if (!some_op(g, u, w)) return;
        emit_op(g, w, TSR_TICKET_NONE, TSR_TICKET_NONE, EMIT_NODEPS);
    }
    g->cov[COV_V2 + V2_ABORTED]++;
}

// DEP_FAILED chain from the failed ticket t: each link on a unit not in error state depends on the
// previous one (CC §9.2), ending sometimes with a NOP join.
static void dep_chain(gen_t * g, uint32_t t) {
    const uint32_t len = 1 + rng_below(&g->r, 3);
    uint32_t prev = t;
    for (uint32_t k = 0; k < len; ++k) {
        uint32_t cand[TSR_NUM_UNITS], nc = 0;
        for (uint32_t v = 0; v < TSR_NUM_UNITS; ++v)
            if (v != TSR_UNIT_SYNC && unit_built(g, v) && g->fail_seq[v] == NOFAIL) cand[nc++] = v;
        if (nc == 0) break;
        uint32_t w[8];
        const uint32_t v = cand[rng_below(&g->r, nc)];
        if (!some_op(g, v, w)) break;
        const uint32_t i = emit_op(g, w, prev, TSR_TICKET_NONE, EMIT_NODEPS);
        prev = g->rec[i].ticket;
    }
    if (g->fail_seq[TSR_UNIT_SYNC] == NOFAIL && rng_below(&g->r, 2)) {
        const uint32_t w[8] = { TSR_OP_NOP, 0, 0, 0, 0, 0, 0, 0 };
        emit_op(g, w, prev, TSR_TICKET_NONE, EMIT_NODEPS);
    }
    g->cov[COV_V2 + V2_DEPFAIL]++;
}

// An op of opcode op with every used field random (in range or not).
static tsr_insn rand_shaped(gen_t * g, uint32_t op) {
    rng_t * r = &g->r;
    tsr_insn i = {{0}};
    switch (op) {
    case TSR_OP_LD:
    case TSR_OP_ST:
        i.w[0] = tsr_w0(op, rng_below(r, g->o->cfg.num_ch), 0, 0);
        for (int k = 1; k <= 3; ++k) i.w[k] = rng_below(r, 2) ? rng_u32(r) : rng_below(r, 64);
        break;
    case TSR_OP_QACT:
        i.w[0] = tsr_w0(op, rng_below(r, 2), 0, 0);
        for (int k = 1; k <= 5; ++k) i.w[k] = rng_below(r, 2) ? rng_u32(r) : 32u * rng_below(r, 64);
        break;
    default:          i.w[0] = op; break;           // NOP, FENCE
    }
    return i;
}

// QACT with a legal K (multiple of the block size, <= K_MAX, small) and T.
static void legal_kt(gen_t * g, uint32_t kind, uint32_t * K, uint32_t * T) {
    const uint32_t bs = tsr_qa_block_size(kind);
    const uint32_t kcap = g->o->max_k < g->o->cfg.k_max ? g->o->max_k : g->o->cfg.k_max;
    const uint32_t nmax = kcap / bs ? kcap / bs : 1;
    *K = bs * (1 + rng_below(&g->r, nmax < 8 ? nmax : 8));
    *T = 1 + rng_below(&g->r, g->o->cfg.t_max);
}

// The refused opcodes of this configuration (EX §7.2: reserved, retired, absent or without their
// CAPS bits): ILLEGAL_OP on SYNC.
static uint32_t refused_opcode(gen_t * g) {
    static const uint8_t ops[] = { 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0F, 0x12, 0x13,
                                   0x14, 0x15, 0x17, 0x19, 0x1B, 0x1F, 0x21, 0x22, 0x23, 0x2F, 0x30, 0x31, 0x32,
                                   0x33, 0x34, 0x37, 0x38, 0x39, 0x3F, 0x40, 0x41, 0x47, 0x48, 0x6F, 0x70, 0x7E,
                                   0x80, 0xA5, 0xFE, 0xFF };
    for (;;) {
        const uint32_t op = ops[rng_below(&g->r, sizeof(ops))];
        const tsr_op_handler * h;
        op_unit(g, op, &h);
        if (h == NULL) return op;
    }
}

// v1 M2 error kinds (fetch / END endings dropped) and the v2 endings.
enum { T2_RSV, T2_OPC, T2_CH, T2_KIND, T2_K, T2_T, T2_SPM, T2_DRAM, T2_BADDEP, T2_DEPFAIL, T2_QABORT, T2_QCLEAR,
       T2_N };
#define T2_FAULTS T2_BADDEP     // T2_RSV .. T2_DRAM: one faulting op

// Pushes the faulting op of tail `tail` (< T2_FAULTS). Returns its record index.
static uint32_t fault_m2(gen_t * g, int tail) {
    rng_t * r = &g->r;
    const tsr_config * c = &g->o->cfg;
    const uint32_t S = g->spm_lines;
    const uint64_t L = g->lim;
    switch (tail) {
    case T2_CH:
        if (c->num_ch < 16) {
            tsr_insn i = rand_shaped(g, rng_below(r, 2) ? TSR_OP_LD : TSR_OP_ST);
            i.w[0] = (i.w[0] & ~0xF00u) | ((c->num_ch + rng_below(r, 16 - c->num_ch)) << 8);
            return emit_fault(g, i.w, TSR_TICKET_NONE, TSR_TICKET_NONE, TSR_ERR_ILLEGAL_OP, 0);
        }
        /* fallthrough */
    case T2_RSV: {
        static const uint32_t ops[] = { TSR_OP_NOP, TSR_OP_FENCE, TSR_OP_LD, TSR_OP_ST, TSR_OP_QACT };
        const uint32_t op = ops[rng_below(r, 5)];
        uint32_t mask, wfree;                     // EX §7.1: x16 and flags are reserved for these ops
        switch (op) {
        case TSR_OP_LD:
        case TSR_OP_ST:   mask = 0xFFFFF000u; wfree = 4; break;
        case TSR_OP_QACT: mask = 0xFFFFF000u; wfree = 6; break;
        default:          mask = 0xFFFFFF00u; wfree = 1; break;
        }
        tsr_insn i = rand_shaped(g, op);
        const uint32_t nm = (uint32_t)__builtin_popcount(mask);
        uint32_t k = rng_below(r, nm + 32u * (8u - wfree));
        if (k < nm) {
            for (uint32_t b = 0; b < 32; ++b) {
                if (((mask >> b) & 1u) && k-- == 0) { i.w[0] |= 1u << b; break; }
            }
        } else {
            k -= nm;
            i.w[wfree + k / 32] |= 1u << (k % 32);
        }
        return emit_fault(g, i.w, TSR_TICKET_NONE, TSR_TICKET_NONE, TSR_ERR_ILLEGAL_OP, 0);
    }
    case T2_OPC: {
        tsr_insn i = {{0}};
        i.w[0] = refused_opcode(g) | (rng_below(r, 2) ? (rng_u32(r) & ~0xFFu) : 0u);
        if (rng_below(r, 2)) for (int k = 1; k < 8; ++k) i.w[k] = rng_u32(r);
        return emit_fault(g, i.w, TSR_TICKET_NONE, TSR_TICKET_NONE, TSR_ERR_ILLEGAL_OP, 0);
    }
    case T2_KIND: {
        tsr_insn i = rand_shaped(g, TSR_OP_QACT);
        i.w[0] = (i.w[0] & ~0xF00u) | ((2 + rng_below(r, 14)) << 8);
        return emit_fault(g, i.w, TSR_TICKET_NONE, TSR_TICKET_NONE, TSR_ERR_ILLEGAL_OP, 0);
    }
    case T2_K: {
        const uint32_t kind = rng_below(r, 2), bs = tsr_qa_block_size(kind);
        static const uint32_t big[] = { 0xFFFFFFFFu, 0x80000000u, 0x00010000u };
        uint32_t K, T;
        legal_kt(g, kind, &K, &T);
        switch (rng_below(r, 5)) {
        case 0:  K = 0; break;
        case 1:  K = bs / 2 + bs * rng_below(r, 4); break;                    // not a multiple
        case 2:  K = kind ? 256 * rng_below(r, 4) + 32 * (1 + rng_below(r, 7)) : 32 * rng_below(r, 4) + 16; break;
        case 3:  K = c->k_max + bs * (1 + rng_below(r, 4)); break;           // multiple, > K_MAX
        default: K = big[rng_below(r, 3)]; break;
        }
        if (rng_below(r, 2)) T = rng_below(r, 2) ? 0 : c->t_max + 1 + rng_below(r, 8);   // T also bad
        const uint32_t dst = rng_below(r, 2) ? 0 : S + rng_below(r, 100);              // SPM maybe bad
        tsr_insn i = tsr_enc_qact(kind, dst, rng_u32(r), K, T, rng_u32(r));
        return emit_fault(g, i.w, TSR_TICKET_NONE, TSR_TICKET_NONE, TSR_ERR_K_RANGE, 0);
    }
    case T2_T: {
        const uint32_t kind = rng_below(r, 2);
        uint32_t K, T;
        legal_kt(g, kind, &K, &T);
        static const uint32_t bad[] = { 0, 9, 16, 0xFFFFFFFFu, 0x80000001u };
        T = rng_below(r, 3) ? bad[rng_below(r, 5)] : c->t_max + 1 + rng_below(r, 8);
        if (T >= 1 && T <= c->t_max) T = 0;
        const uint32_t dst = rng_below(r, 2) ? 0 : S + rng_below(r, 100);
        tsr_insn i = tsr_enc_qact(kind, dst, rng_u32(r), K, T, rng_below(r, 4));
        return emit_fault(g, i.w, TSR_TICKET_NONE, TSR_TICKET_NONE, TSR_ERR_T_RANGE, 0);
    }
    case T2_SPM: {
        const uint32_t v = rng_below(r, 5);
        if (v <= 1) {                          // LD / ST SPM operand beyond the SPM
            const uint32_t n = v ? 0 : 1 + rng_below(r, 64);
            const uint32_t spm = v ? S + 1 + rng_below(r, 1000) : S - (n < S ? n : S) + 1 + rng_below(r, n);
            uint32_t dram = rng_below(r, 1u << 20);
            if (rng_below(r, 2)) dram = (uint32_t)(L > 0xFFFFFFFFull ? 0xFFFFFFFFu : L);   // DRAM also bad
            tsr_insn i = rng_below(r, 2) ? tsr_enc_ld(rng_below(r, c->num_ch), spm, dram, n)
                                         : tsr_enc_st(rng_below(r, c->num_ch), spm, dram, n);
            return emit_fault(g, i.w, TSR_TICKET_NONE, TSR_TICKET_NONE, TSR_ERR_SPM_RANGE, spm);
        }
        const uint32_t kind = rng_below(r, 2);
        uint32_t K, T;
        legal_kt(g, kind, &K, &T);
        const uint32_t xl = K / 16, ql = tsr_qa_lines(kind, K);
        tsr_insn i;
        uint32_t info;
        if (v == 2) {                          // QACT dst beyond the SPM (src maybe too)
            const uint64_t ext = (uint64_t)T * ql;
            const uint32_t dst = (uint32_t)(ext <= S ? S - ext + 1 + rng_below(r, (uint32_t)ext) : rng_below(r, S));
            const uint32_t src = rng_below(r, 2) ? S + rng_below(r, 50) : 0;
            i = tsr_enc_qact(kind, dst, src, K, T, 0);
            info = dst;
        } else if (v == 3 || c->t_max < 2) {  // QACT src beyond the SPM, dst fine
            const uint32_t st = rng_below(r, 2) ? 0 : xl + rng_below(r, 8);
            const uint64_t ext = (uint64_t)(T - 1) * (st ? st : xl) + xl;
            const uint32_t src = (uint32_t)(ext <= S ? S - ext + 1 + rng_below(r, 16) : rng_below(r, S));
            i = tsr_enc_qact(kind, 0, src, K, T, st);
            info = src;
        } else {                               // (T-1) * stride beyond 32 bits
            T = 2 + rng_below(r, c->t_max - 1);
            const uint32_t src = rng_below(r, S / 2 + 1);
            i = tsr_enc_qact(kind, 0, src, K, T, 0x80000000u | rng_u32(r));
            info = src;
        }
        return emit_fault(g, i.w, TSR_TICKET_NONE, TSR_TICKET_NONE, TSR_ERR_SPM_RANGE, info);
    }
    case T2_DRAM:
    default: {
        if (L >= 0xFFFFFF00ull) {              // limit not reachable with 32-bit addresses
            tsr_insn i = tsr_enc_nop();
            i.w[7] = 1;
            return emit_fault(g, i.w, TSR_TICKET_NONE, TSR_TICKET_NONE, TSR_ERR_ILLEGAL_OP, 0);
        }
        const uint32_t n = rng_below(r, 4) == 0 ? 0 : 1 + rng_below(r, 64);
        const uint32_t spm = rng_below(r, S - (n < S ? n : S) + 1);
        uint32_t dram;
        if (n == 0) dram = (uint32_t)L + 1 + rng_below(r, 100);
        else dram = (uint32_t)(L - n) + 1 + rng_below(r, n + 50);
        if (rng_below(r, 8) == 0) dram = 0xFFFFFFFFu - rng_below(r, 16);
        tsr_insn i = rng_below(r, 2) ? tsr_enc_ld(rng_below(r, c->num_ch), spm, dram, n)
                                     : tsr_enc_st(rng_below(r, c->num_ch), spm, dram, n);
        return emit_fault(g, i.w, TSR_TICKET_NONE, TSR_TICKET_NONE, TSR_ERR_DRAM_RANGE, dram);
    }
    }
}

// BAD_DEP (CC §8.3, EX §8.1: after ILLEGAL_OP, before every other code; ERR_INFO 0): a dep on
// unit 6..13, the marker unit 14, the absent unit 4, the op's own ticket or a ticket not pushed
// yet, on a legal op, an op with a lower-priority fault, or an illegal op (ILLEGAL_OP wins).
static uint32_t fault_baddep(gen_t * g) {
    rng_t * r = &g->r;
    uint32_t w[8] = { 0 };
    uint32_t base = TSR_ERR_NONE;
    const uint32_t sel = rng_below(r, 6);
    if (sel <= 2) {
        uint32_t v = rng_below(r, TSR_NUM_UNITS);
        if (!some_op(g, v, w) && !some_op(g, v = TSR_UNIT_DMA, w)) return fault_m2(g, T2_RSV);
    } else if (sel == 3) {
        tsr_insn i = tsr_enc_qact(rng_below(r, 2), 0, 0, 48, 1, 0);        // K_RANGE
        memcpy(w, i.w, 32);
        base = TSR_ERR_K_RANGE;
    } else if (sel == 4) {
        tsr_insn i = tsr_enc_ld(0, g->spm_lines, 0, 1);                     // SPM_RANGE
        memcpy(w, i.w, 32);
        base = TSR_ERR_SPM_RANGE;
    } else {
        tsr_insn i = tsr_enc_nop();
        i.w[1 + rng_below(r, 7)] = 1u << rng_below(r, 32);                  // ILLEGAL_OP
        memcpy(w, i.w, 32);
        base = TSR_ERR_ILLEGAL_OP;
    }
    const uint32_t u = op_unit(g, w[0], NULL);
    uint32_t bad;
    switch (rng_below(r, 6)) {
    case 0:  bad = TK(6 + rng_below(r, 8), rng_below(r, 1000)); break;
    case 1:  bad = TK(TSR_UNIT_MARKER, rng_below(r, 16)); break;
    case 2:  bad = unit_built(g, TSR_UNIT_ATT) ? TK(13, 0) : TK(TSR_UNIT_ATT, rng_below(r, 4)); break;
    case 3:  bad = TK(u, g->cnt[u]); break;                                  // its own ticket
    case 4: {
        uint32_t v;
        do v = rng_below(r, TSR_NUM_UNITS); while (!unit_built(g, v));
        bad = TK(v, g->cnt[v] + (v == u ? 0 : rng_below(r, 1000)));
        break;
    }
    default: {
        uint32_t v;
        do v = rng_below(r, TSR_NUM_UNITS); while (!unit_built(g, v));
        bad = TK(v, g->cnt[v] + 0x07FFFFFFu);                                // the farthest future ticket
        break;
    }
    }
    // the other slot: a good dep, or free for the barrier dep after a QCLEAR
    const uint32_t other = (rng_below(r, 2) && !((g->barrier_need >> u) & 1u)) ? random_good_ticket(g) : TSR_TICKET_NONE;
    const uint32_t want = base == TSR_ERR_ILLEGAL_OP ? TSR_ERR_ILLEGAL_OP : TSR_ERR_BAD_DEP;
    g->cov[COV_V2 + V2_BADDEP]++;
    return rng_below(r, 2) ? emit_fault(g, w, bad, other, want, 0) : emit_fault(g, w, other, bad, want, 0);
}

// Allocation helpers for the QABORT recipe (the v1 allocators' state).
static uint32_t take_spm(gen_t * g, uint32_t n) {
    const uint32_t a = g->spm_next + rng_below(&g->r, 2);
    g->spm_next = a + n;
    return a;
}
static uint64_t take_io(gen_t * g, uint32_t ch, uint64_t n) {
    const uint64_t a = g->io_next[ch] + rng_below(&g->r, 64);
    g->io_next[ch] = a + n;
    return a;
}

// QABORT recipe (first records of the trace): QHOLD set, ops that cannot start (held, or waiting
// for a held op), QABORT, maybe more entries (ABORTED or poisoned), QHOLD clear, QCLEAR. Nothing
// has started at the abort, so every entry since the hold completes ABORTED or with its poison
// code whatever the schedule, the held stores write nothing, and the QCLEAR clears the abort's
// records (CC §9.5). The ops after the QCLEAR depend on it.
static void qabort_recipe(gen_t * g) {
    rng_t * r = &g->r;
    const uint32_t C = g->o->cfg.num_ch;
    if (g->n_rec != 0) {
        fprintf(stderr, "tsr-trace-gen: internal: the QABORT recipe must open the trace\n");
        exit(2);
    }
    g->quiet++;
    emit_qctrl(g, TSR_QCTRL_QHOLD_SET);
    const size_t chk0 = g->n_chk;
    // LD of fresh data, ST of it (must not happen), an op of another unit on the loaded lines
    const uint32_t n = 1 + rng_below(r, 16), ca = rng_below(r, C), cb = rng_below(r, C);
    const uint32_t spm = take_spm(g, n + 8);
    const uint64_t src = take_io(g, ca, n);
    uint8_t * data = (uint8_t *)xmalloc((size_t)n * 64);
    rng_bytes(r, data, (size_t)n * 64);
    dram_put(g, ca, src, data, (size_t)n * 64);
    free(data);
    const uint32_t ld = emit_insn(g, tsr_enc_ld(ca, spm, (uint32_t)src, n));
    const uint32_t t_ld = g->rec[ld].ticket;
    if (g->hold_room[TSR_UNIT_DMA] > 0) {       // (queues of depth 1 hold the LD alone)
        const uint64_t dst = take_io(g, cb, n);
        emit_insn(g, tsr_enc_st(cb, spm, (uint32_t)dst, n));
        add_check(g, cb, dst, n, NULL);
    }
    if (n >= 2 && g->hold_room[TSR_UNIT_QACT] > 0 && rng_below(r, 2))
        emit_insn(g, tsr_enc_qact(TSR_QA_A32, spm + n, spm, 32, 1, 0));     // waits for the held LD
    const uint32_t nop[8] = { TSR_OP_NOP, 0, 0, 0, 0, 0, 0, 0 };
    emit_op(g, nop, t_ld, TSR_TICKET_NONE, EMIT_NODEPS);                    // a SYNC entry that cannot complete
    emit_qctrl(g, TSR_QCTRL_QABORT);
    emit_qctrl(g, TSR_QCTRL_QHOLD_CLR);        // the aborted entries complete ABORTED, held or not
    const uint32_t k = rng_below(r, 3);
    for (uint32_t i = 0; i < k; ++i) {
        uint32_t w[8];
        const uint32_t u = rng_below(r, TSR_NUM_UNITS);
        if (rng_below(r, 3) == 0) fault_m2(g, (int)rng_below(r, T2_FAULTS));
        else if (some_op(g, u, w)) emit_op(g, w, TSR_TICKET_NONE, TSR_TICKET_NONE, EMIT_NODEPS);
    }
    emit_qctrl(g, TSR_QCTRL_QCLEAR);
    g->inj_pidx = NOFAIL;                      // cleared: the final records come from what follows
    checks_cancel(g, chk0);
    g->quiet--;
}

// Terminal QABORT (the whole trace): QHOLD set, one held entry on every built unit (the SYNC one a
// NOP waiting for a held op), QABORT, QHOLD clear, maybe entries after it. Nothing has started at
// the abort and every unit has an entry at its queue front, so every record is that entry's
// (ERR_SEQ, ERR_W0, ERR_PIDX; CC §9.5) whatever the schedule: the trace ends with every built unit
// in error state ABORTED and DONE status ABORTED (EX §4.4). A unit with nothing queued would record
// the next push index, which depends on when the abort is dispatched; returns 0 (nothing pushed)
// when some built unit takes no op in this configuration (CAPS 0: the VU).
static int qabort_terminal(gen_t * g) {
    uint32_t w[TSR_NUM_UNITS][8];
    for (uint32_t u = 0; u < TSR_NUM_UNITS; ++u)
        if (unit_built(g, u) && u != TSR_UNIT_SYNC && !some_op(g, u, w[u])) return 0;
    if (g->n_rec != 0) {
        fprintf(stderr, "tsr-trace-gen: internal: the QABORT recipe must open the trace\n");
        exit(2);
    }
    g->quiet++;
    emit_qctrl(g, TSR_QCTRL_QHOLD_SET);
    uint32_t first = TSR_TICKET_NONE;
    for (uint32_t u = 0; u < TSR_NUM_UNITS; ++u) {
        if (!unit_built(g, u) || u == TSR_UNIT_SYNC) continue;
        const uint32_t i = emit_op(g, w[u], TSR_TICKET_NONE, TSR_TICKET_NONE, EMIT_NODEPS | EMIT_NOHAZ);
        if (first == TSR_TICKET_NONE) first = g->rec[i].ticket;
    }
    const uint32_t nop[8] = { TSR_OP_NOP, 0, 0, 0, 0, 0, 0, 0 };
    emit_op(g, nop, first, TSR_TICKET_NONE, EMIT_NODEPS);
    emit_qctrl(g, TSR_QCTRL_QABORT);
    emit_qctrl(g, TSR_QCTRL_QHOLD_CLR);
    for (uint32_t k = 0, n = rng_below(&g->r, 3); k < n; ++k) {
        uint32_t v[8];
        if (some_op(g, rng_below(&g->r, TSR_NUM_UNITS), v)) emit_op(g, v, TSR_TICKET_NONE, TSR_TICKET_NONE, EMIT_NODEPS | EMIT_NOHAZ);
    }
    g->inj_pidx = g->rec[1].pidx;              // the first held entry: the first error (ABORTED)
    g->quiet--;
    return 1;
}

// v1 M1 error kinds (case 8, v1's missing END, becomes a v2 ending).
static uint32_t fault_m1(gen_t * g, uint32_t kind) {
    rng_t * r = &g->r;
    const tsr_config * c = &g->o->cfg;
    for (;;) {
        switch (kind) {
        case 0: {   // reserved bit in a NOP
            tsr_insn i = tsr_enc_nop();
            const uint32_t bit = 8 + rng_below(r, 256 - 8);
            i.w[bit / 32] |= 1u << (bit % 32);
            return emit_fault(g, i.w, TSR_TICKET_NONE, TSR_TICKET_NONE, TSR_ERR_ILLEGAL_OP, 0);
        }
        case 1: {   // opcode the issue path refuses
            tsr_insn i = tsr_enc_nop();
            i.w[0] = refused_opcode(g);
            return emit_fault(g, i.w, TSR_TICKET_NONE, TSR_TICKET_NONE, TSR_ERR_ILLEGAL_OP, 0);
        }
        case 2:
            if (c->num_ch >= 16) break;
            {
                tsr_insn i = tsr_enc_ld(c->num_ch + rng_below(r, 16 - c->num_ch), 0, 0, 1);
                return emit_fault(g, i.w, TSR_TICKET_NONE, TSR_TICKET_NONE, TSR_ERR_ILLEGAL_OP, 0);
            }
        case 3: {
            tsr_insn i = tsr_enc_mvm(2 + rng_below(r, 14), 0, 0, 0, 0, 1, 32, 1, 0);
            return emit_fault(g, i.w, TSR_TICKET_NONE, TSR_TICKET_NONE, TSR_ERR_BAD_FMT, 0);
        }
        case 4: {
            tsr_insn i = tsr_enc_qact(rng_below(r, 2), 0, 0, 48, 1, 0);
            return emit_fault(g, i.w, TSR_TICKET_NONE, TSR_TICKET_NONE, TSR_ERR_K_RANGE, 0);
        }
        case 5:
            if (c->t_max >= 15) break;
            {
                tsr_insn i = tsr_enc_mvm(0, 0, 0, 0, 0, 1, 32, c->t_max + 1 + rng_below(r, 15 - c->t_max), 0);
                return emit_fault(g, i.w, TSR_TICKET_NONE, TSR_TICKET_NONE, TSR_ERR_T_RANGE, 0);
            }
        case 6: {
            tsr_insn i = tsr_enc_ld(0, g->spm_lines - 1, 0, 2);
            return emit_fault(g, i.w, TSR_TICKET_NONE, TSR_TICKET_NONE, TSR_ERR_SPM_RANGE, g->spm_lines - 1);
        }
        case 7: {
            tsr_insn i = tsr_enc_st(rng_below(r, c->num_ch), 0, (uint32_t)(g->lim - 1), 2);
            return emit_fault(g, i.w, TSR_TICKET_NONE, TSR_TICKET_NONE, TSR_ERR_DRAM_RANGE, (uint32_t)(g->lim - 1));
        }
        default:
            return fault_baddep(g);
        }
        kind = rng_below(r, 8);
    }
}

// ---------------------------------------------------------------------------
// M3 error kinds (v1: the 23 MVM faults with lower-priority violations and 32-bit wrapping extents)
// ---------------------------------------------------------------------------
// A legal MVM (absolute operands) to be broken by a fault.
typedef struct { uint32_t fmt, fl, dst, x, woff, N, K, T, w6; } mvm_ops;

static uint64_t mvm_dst_ext(const mvm_ops * m) {
    const uint64_t yl = tsr_ceil_div_u64(4ull * m->N, 64);
    return (uint64_t)(m->T - 1) * (m->w6 ? m->w6 : yl) + yl;
}

static void legal_mvm(gen_t * g, mvm_ops * m) {
    rng_t * r = &g->r;
    const tsr_config * c = &g->o->cfg;
    const uint32_t S = g->spm_lines;
    m->fmt = rng_below(r, 2);
    const uint32_t bs = tsr_fmt_block_size(m->fmt);
    const uint32_t nmax = c->k_max / bs;
    m->K = bs * (1 + rng_below(r, nmax < 8 ? nmax : 8));
    m->T = 1 + rng_below(r, c->t_max);
    m->N = rng_below(r, 6) == 0 ? 0 : 1 + rng_below(r, 64);
    m->fl = rng_below(r, 2) ? TSR_MVM_FLAG_ACC : 0;
    const uint32_t yl = tsr_ceil_div_u32(4 * m->N, 64);
    m->w6 = rng_below(r, 2) ? 0 : yl + rng_below(r, 4);
    const uint64_t de = mvm_dst_ext(m);
    const uint64_t xe = (uint64_t)m->T * tsr_qa_lines(tsr_qa_kind_for_fmt(m->fmt), m->K);
    m->dst = rng_below(r, (uint32_t)(S - de + 1));
    m->x = rng_below(r, (uint32_t)(S - xe + 1));
    const uint64_t we = (uint64_t)tsr_ceil_div_u32(m->N, c->num_ch) * tsr_row_beats(m->fmt, m->K);
    m->woff = rng_below(r, (uint32_t)(g->lim - we + 1));
}

// Operand values beyond the limit for SPM (A1 = dst, A2 = x; extent ext) / DRAM (A3). Variant v:
// 0 just past the limit, 1 far beyond it, 2 so close to 2^32 that a + e wraps 32 bits; -1 random.
static uint32_t past_spm(gen_t * g, uint64_t ext, int v) {
    rng_t * r = &g->r;
    const uint64_t S = g->spm_lines;
    if (v < 0) v = (int)rng_below(r, 3);
    switch (v) {
    case 0:  return ext == 0 ? (uint32_t)(S + 1 + rng_below(r, 64))
                             : (uint32_t)(S - (ext < S ? ext : S) + 1 + rng_below(r, (uint32_t)(ext < 16 ? ext : 16)));
    case 1:  return (uint32_t)(S + 1 + rng_below(r, 1u << 20));
    default:
        g->cov[C3_WRAP] += ext > 0;
        return ext == 0 ? 0xFFFFFFFFu - rng_below(r, 16)
                        : (uint32_t)(0x100000000ull - 1 - rng_below(r, (uint32_t)(ext < 64 ? ext : 64)));
    }
}
static uint32_t past_dram(gen_t * g, uint64_t ext, int v) {
    rng_t * r = &g->r;
    if (v < 0) v = (int)rng_below(r, 3);
    switch (v) {
    case 0:  return (uint32_t)(g->lim - (ext < g->lim ? ext : g->lim) + 1 + rng_below(r, 16));
    case 1:  return (uint32_t)(g->lim + 1 + rng_below(r, 1u << 16));
    default:
        g->cov[C3_WRAP] += ext > 0;
        return ext == 0 ? 0xFFFFFFFFu - rng_below(r, 16)
                        : (uint32_t)(0x100000000ull - 1 - rng_below(r, (uint32_t)(ext < 64 ? ext : 64)));
    }
}

// MVM error kinds with their variants (--errors 100 cycles through this list, then the M2 / v2 ones).
static const struct { int tail, var; } m3_tail_cycle[] = {
    { T3_RSV, 0 }, { T3_FMT, 0 }, { T3_K, 0 }, { T3_T, 0 }, { T3_SPM_DST, 0 }, { T3_SPM_X, 0 }, { T3_DRAM, 0 },
    { T3_RSV, 1 }, { T3_FMT, 1 }, { T3_K, 1 }, { T3_T, 1 }, { T3_SPM_DST, 1 }, { T3_SPM_X, 1 }, { T3_DRAM, 1 },
    { T3_RSV, 2 }, { T3_K, 2 }, { T3_SPM_DST, 2 }, { T3_SPM_X, 2 }, { T3_DRAM, 2 },
    { T3_RSV, 3 }, { T3_K, 3 }, { T3_SPM_DST, 3 }, { T3_DRAM, 3 },
};
#define M3_TAIL_CYCLE ((unsigned)(sizeof(m3_tail_cycle) / sizeof(m3_tail_cycle[0])))

// Pushes a faulting MVM (kind `tail`, T3_*; variant `var`, -1 = random). Returns its record index.
//   T3_RSV     var: reserved bit in 0 flags[3:1], 1 x16[15:12] (v1 rd), 2 w5[31:20], 3 w7
//   T3_K       var: 0 K = 0, 1 not a block multiple, 2 a multiple > K_MAX, 3 16-bit field extremes
//   T3_T       var: 0 T = 0, 1 T > T_MAX (when T_MAX < 15)
//   T3_SPM_DST var: 0 (T-1) x dst_stride = 2^32 + small (T >= 3), 1 N = k x 2^30 + small (4N wraps
//                   32 bits to a small value), 2 N > 16 x SPM lines, 3 dst past the end (just / far /
//                   32-bit wrap)
//   T3_SPM_X   var: x 0 just past the end, 1 far beyond, 2 x + T x QA_LINES wraps 32 bits
//   T3_DRAM    var: W_OFF 0 just past (W_OFF + R x ROW_BEATS = limit + 1..16), 1 far beyond,
//                   2 wraps 32 bits, 3 N = 0 with W_OFF > limit
// Lower-priority violations (later checks) are added at random.
static uint32_t fault_m3(gen_t * g, int tail, int var) {
    rng_t * r = &g->r;
    const tsr_config * c = &g->o->cfg;
    mvm_ops m;
    legal_mvm(g, &m);
    int64_t info = 0;
    uint32_t code = TSR_ERR_ILLEGAL_OP, rsv_w0 = 0, rsv_w5 = 0, rsv_w7 = 0;
    int lo_k = 0, lo_t = 0, lo_dst = 0, lo_x = 0, lo_w = 0;     // violations (lower priority ones random)
    int vk = -1, vt = -1, vd = -1, vx = -1, vw = -1;            // their variants
    switch (tail) {
    case T3_RSV: {
        const int k = var >= 0 ? var : (int)rng_below(r, 4);
        if (k == 0) rsv_w0 = 1u << (13 + rng_below(r, 3));
        else if (k == 1) rsv_w0 = 1u << (28 + rng_below(r, 4));
        else if (k == 2) rsv_w5 = 1u << (20 + rng_below(r, 12));
        else rsv_w7 = 1u << rng_below(r, 32);
        if (rng_below(r, 2)) m.fmt = 2 + rng_below(r, 14);
        lo_k = rng_below(r, 3) == 0;
        lo_t = rng_below(r, 3) == 0;
        lo_dst = rng_below(r, 3) == 0;
        lo_w = rng_below(r, 3) == 0;
        code = TSR_ERR_ILLEGAL_OP;
        break;
    }
    case T3_FMT:
        m.fmt = 2 + rng_below(r, 14);
        lo_k = rng_below(r, 3) == 0;
        lo_t = rng_below(r, 3) == 0;
        lo_dst = rng_below(r, 4) == 0;
        lo_x = rng_below(r, 4) == 0;
        lo_w = rng_below(r, 4) == 0;
        code = TSR_ERR_BAD_FMT;
        break;
    case T3_K:
        lo_k = 1;
        vk = var;
        lo_t = rng_below(r, 3) == 0;
        lo_dst = rng_below(r, 4) == 0;
        lo_w = rng_below(r, 4) == 0;
        code = TSR_ERR_K_RANGE;
        break;
    case T3_T:
        lo_t = 1;
        vt = var;
        lo_dst = rng_below(r, 3) == 0;
        lo_x = rng_below(r, 4) == 0;
        lo_w = rng_below(r, 3) == 0;
        code = TSR_ERR_T_RANGE;
        break;
    case T3_SPM_DST:
        lo_dst = 1;
        vd = var;
        lo_x = rng_below(r, 3) == 0;
        lo_w = rng_below(r, 3) == 0;
        code = TSR_ERR_SPM_RANGE;
        break;
    case T3_SPM_X:
        lo_x = 1;
        vx = var;
        lo_w = rng_below(r, 3) == 0;
        code = TSR_ERR_SPM_RANGE;
        break;
    default:
        lo_w = 1;
        vw = var;
        code = TSR_ERR_DRAM_RANGE;
        break;
    }
    const uint32_t bs = tsr_fmt_valid(m.fmt) ? tsr_fmt_block_size(m.fmt) : 32;
    if (lo_k) {
        switch (vk >= 0 ? vk : (int)rng_below(r, 4)) {
        case 0:  m.K = 0; break;
        case 1:  m.K = m.fmt == TSR_FMT_Q4_K ? 256 * rng_below(r, 4) + 32 * (1 + rng_below(r, 7))
                                             : 32 * rng_below(r, 4) + 1 + rng_below(r, 31); break;
        case 2:  m.K = c->k_max + bs * (1 + rng_below(r, 4)); break;
        default: { static const uint32_t big[] = { 0xFFFF, 0xFFE0, 0xFF00, 0xFFFE }; m.K = big[rng_below(r, 4)]; } break;
        }
        if (m.K > 0xFFFF || (m.K <= c->k_max && m.K % bs == 0 && m.K != 0)) m.K = 0;
    }
    if (lo_t) {
        const int v = vt >= 0 ? vt : (int)rng_below(r, 2);
        m.T = (c->t_max < 15 && v == 1) ? c->t_max + 1 + rng_below(r, 15 - c->t_max) : 0;
    }
    if (lo_dst) {
        int v = vd >= 0 ? vd : (int)rng_below(r, 4);
        if (v == 0 && tail == T3_SPM_DST) {
            if (c->t_max >= 3) m.T = 3 + rng_below(r, c->t_max - 2);
            else v = 3;
        }
        if (v == 0 && m.T >= 3 && m.T <= 15) {
            // (T - 1) * dst_stride >= 2^32: a 32-bit product would wrap to a small extent
            m.w6 = (uint32_t)((0x100000000ull + m.T - 2) / (m.T - 1)) + rng_below(r, 4);
            m.dst = rng_below(r, 64);
            g->cov[C3_WRAP]++;
        } else if (v == 1 || v == 2) {
            // y_lines > SPM lines; for N >= 2^30, 4N does not fit 32 bits
            m.N = v == 1 ? 0x40000000u * (1 + rng_below(r, 3)) + rng_below(r, 1024)    // 4N mod 2^32 < 4096
                         : 16 * g->spm_lines + 1 + rng_below(r, 4096);
            g->cov[C3_WRAP] += v == 1;
            m.w6 = 0;
            m.dst = rng_below(r, 64);
        } else {
            m.dst = past_spm(g, (m.T >= 1 && m.T <= 15) ? mvm_dst_ext(&m) : 0, -1);
        }
    }
    if (lo_x) {
        const uint64_t xe = (tsr_fmt_valid(m.fmt) && m.K && m.K % bs == 0 && m.T >= 1 && m.T <= 15)
                                ? (uint64_t)m.T * tsr_qa_lines(tsr_qa_kind_for_fmt(m.fmt), m.K) : 0;
        m.x = past_spm(g, xe, vx);
    }
    if (lo_w) {
        const uint64_t we = (tsr_fmt_valid(m.fmt) && m.K && m.K % bs == 0)
                                ? tsr_ceil_div_u64(m.N, c->num_ch) * tsr_row_beats(m.fmt, m.K) : 0;
        const int v = vw >= 0 ? vw : (code == TSR_ERR_DRAM_RANGE ? (int)rng_below(r, 4) : (int)rng_below(r, 3));
        if (v == 3) {
            m.N = 0;                                     // e = 0: only a > limit fails
            m.woff = (uint32_t)(g->lim + 1 + rng_below(r, 64));
        } else {
            m.woff = past_dram(g, we, v);
        }
    }
    tsr_insn i = tsr_enc_mvm(m.fmt, m.fl, m.dst, m.x, m.woff, m.N, m.K, m.T, m.w6);
    i.w[0] |= rsv_w0;
    i.w[5] |= rsv_w5;
    i.w[7] |= rsv_w7;
    if (code == TSR_ERR_SPM_RANGE) info = lo_dst ? m.dst : m.x;
    if (code == TSR_ERR_DRAM_RANGE) info = m.woff;
    g->cov[C3_ERR]++;
    return emit_fault(g, i.w, TSR_TICKET_NONE, TSR_TICKET_NONE, code, info);
}

// ---------------------------------------------------------------------------
// Program assembly for --ops gen / m2 / m3
// ---------------------------------------------------------------------------
enum { TAIL_NONE, TAIL_M1, TAIL_M2, TAIL_M3 };
typedef struct { int cls, k, var; } tail_t;

// One faulting op of the mode's kinds (M3: 3/4 MVM faults).
static uint32_t random_fault(gen_t * g) {
    if (g->o->m3 && rng_below(&g->r, 4)) return fault_m3(g, (int)rng_below(&g->r, T3_N), -1);
    if (!g->o->m2 && !g->o->m3) return fault_m1(g, rng_below(&g->r, 8));
    return fault_m2(g, (int)rng_below(&g->r, T2_FAULTS));
}

// After a fault: ABORTED entries of its unit, a DEP_FAILED chain from it (either or both).
static void fault_consequences(gen_t * g, uint32_t idx, int chain) {
    const rec_t * f = &g->rec[idx];
    const uint32_t u = TK_U(f->ticket), t = f->ticket;
    if (chain) dep_chain(g, t);
    if (!chain || rng_below(&g->r, 2)) after_error(g, u);
    else if (rng_below(&g->r, 3) == 0) dep_chain(g, t);
}

static int v1_groups(gen_t * g, unsigned idx, uint32_t n_groups) {
    uint32_t made = 0;
    for (uint32_t k = 0; k < n_groups; ++k)
        made += (uint32_t)(g->o->m3 ? gen_group_m3(g, k) : g->o->m2 ? gen_group_m2(g) : gen_group(g));
    flush_groups(g);
    (void)idx;
    return (int)made;
}

static int build_v1(gen_t * g, unsigned idx) {
    const opts_t * o = g->o;
    rng_t * r = &g->r;
    tail_t tail = { TAIL_NONE, 0, -1 };
    if (o->m3) {   // --errors 100 cycles through every kind
        if (o->err_pct == 100) {
            const unsigned j = idx % (M3_TAIL_CYCLE + T2_N);
            if (j < M3_TAIL_CYCLE) tail = (tail_t){ TAIL_M3, m3_tail_cycle[j].tail, m3_tail_cycle[j].var };
            else tail = (tail_t){ TAIL_M2, (int)(j - M3_TAIL_CYCLE), -1 };
        } else if (o->err_pct && rng_below(r, 100) < o->err_pct) {
            tail = rng_below(r, 4) ? (tail_t){ TAIL_M3, (int)rng_below(r, T3_N), -1 } : (tail_t){ TAIL_M2, (int)rng_below(r, T2_N), -1 };
        }
        g->wlim_pending = rng_below(r, 4) == 0;
    } else if (o->m2) {
        if (o->err_pct && rng_below(r, 100) < o->err_pct) tail = (tail_t){ TAIL_M2, (int)rng_below(r, T2_N), -1 };
    } else if (o->err_pct && rng_below(r, 100) < o->err_pct) {
        const uint32_t k = rng_below(r, 9);
        tail = k < 8 ? (tail_t){ TAIL_M1, (int)k, -1 } : (tail_t){ TAIL_M2, T2_BADDEP + (int)rng_below(r, 4), -1 };
    }
    if (tail.cls == TAIL_M2 && tail.k == T2_QABORT) qabort_recipe(g);
    if (v1_groups(g, idx, 1 + rng_below(r, o->max_groups)) == 0) {
        fprintf(stderr, "tsr-trace-gen: trace %u: no group fits the configuration\n", idx);
        return 1;
    }
    uint32_t f;
    switch (tail.cls) {
    case TAIL_M1:
        f = fault_m1(g, (uint32_t)tail.k);
        fault_consequences(g, f, rng_below(r, 3) == 0);
        g->cov[COV_ERR]++;
        break;
    case TAIL_M3:
        f = fault_m3(g, tail.k, tail.var);
        fault_consequences(g, f, rng_below(r, 3) == 0);
        g->cov[C2_ERR]++;
        break;
    case TAIL_M2:
        g->cov[o->m2 || o->m3 ? C2_ERR : COV_ERR]++;
        if (tail.k < T2_FAULTS) {
            f = fault_m2(g, tail.k);
            fault_consequences(g, f, rng_below(r, 3) == 0);
        } else if (tail.k == T2_BADDEP) {
            f = fault_baddep(g);
            fault_consequences(g, f, rng_below(r, 3) == 0);
        } else if (tail.k == T2_DEPFAIL) {
            f = random_fault(g);
            fault_consequences(g, f, 1);
        } else if (tail.k == T2_QABORT) {
            if (rng_below(r, 2)) fault_consequences(g, random_fault(g), rng_below(r, 3) == 0);
        } else {                                                  // T2_QCLEAR: error, then R9's recovery
            f = random_fault(g);
            fault_consequences(g, f, rng_below(r, 2));
            emit_qctrl(g, TSR_QCTRL_QCLEAR);
            g->inj_pidx = NOFAIL;
            v1_groups(g, idx, 1 + rng_below(r, o->max_groups));
        }
        break;
    default:
        break;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// Running and checking a stream (see the head comment, "Verification")
// ---------------------------------------------------------------------------
typedef struct {            // a merged host-input region
    uint32_t ch;
    uint64_t lo, hi;        // beats
} region_t;

static int region_cmp(const void * a, const void * b) {
    const region_t * x = (const region_t *)a, * y = (const region_t *)b;
    if (x->ch != y->ch) return x->ch < y->ch ? -1 : 1;
    if (x->lo != y->lo) return x->lo < y->lo ? -1 : 1;
    return 0;
}

// The host inputs as sorted, merged regions (per channel, adjacent or overlapping ones joined).
static region_t * merged_inputs(const gen_t * g, uint32_t * n_out) {
    region_t * v = (region_t *)xmalloc((g->n_in + 1) * sizeof(region_t));
    for (uint32_t i = 0; i < g->n_in; ++i) {
        v[i].ch = g->in[i].ch;
        v[i].lo = g->in[i].lo;
        v[i].hi = g->in[i].hi;
    }
    qsort(v, g->n_in, sizeof(region_t), region_cmp);
    uint32_t n = 0;
    for (uint32_t i = 0; i < g->n_in; ++i) {
        if (n && v[n - 1].ch == v[i].ch && v[i].lo <= v[n - 1].hi) {
            if (v[i].hi > v[n - 1].hi) v[n - 1].hi = v[i].hi;
        } else {
            v[n++] = v[i];
        }
    }
    *n_out = n;
    return v;
}

static tsr_sim * fresh_sim(const gen_t * g, const region_t * reg, uint32_t n_reg) {
    tsr_sim_opts so;
    memset(&so, 0, sizeof(so));
    so.q_depth = g->o->depth;
    so.sched = "fixed";
    tsr_sim * s = tsr_sim_create_opts(&g->o->cfg, &so);
    if (!s) return NULL;
    for (uint32_t i = 0; i < n_reg; ++i)
        memcpy(tsr_sim_dram(s, reg[i].ch) + reg[i].lo * 64, tsr_sim_dram(g->sim, reg[i].ch) + reg[i].lo * 64,
               (size_t)(reg[i].hi - reg[i].lo) * 64);
    return s;
}

typedef struct {
    uint32_t n;
    uint32_t * code;
    uint8_t *  ran;
    uint8_t *  seen;
} evrec_t;

static void on_event(void * ctx, tsr_sim * sim, const tsr_q_event * e) {
    (void)sim;
    evrec_t * v = (evrec_t *)ctx;
    if (e->kind != TSR_QEV_DONE || e->ticket == TSR_TICKET_NONE || e->pidx >= v->n) return;
    v->code[e->pidx] = e->code;
    v->ran[e->pidx] = (uint8_t)e->ran;
    v->seen[e->pidx]++;
}

// Pushes the stream into s (tickets checked against the prediction); 0 on success.
static int push_stream(const gen_t * g, tsr_sim * s, unsigned idx) {
    for (uint32_t i = 0; i < g->n_rec; ++i) {
        const rec_t * r = &g->rec[i];
        uint32_t t = TSR_TICKET_NONE;
        if (r->kind == TSR_TRACE_PUSH_OP) {
            t = tsr_sim_push(s, r->w, r->dep[0], r->dep[1], r->push);
        } else if (r->kind == TSR_TRACE_PUSH_QCTRL) {
            t = tsr_sim_qctrl(s, r->w[0]);
        } else {
            tsr_sim_cw_write(s, TSR_CW_TRACE_MARK, r->w[0]);
        }
        if (t != r->ticket) {
            fprintf(stderr, "tsr-trace-gen: trace %u: record %u got ticket 0x%08x, predicted 0x%08x%s\n", idx, i, t,
                    r->ticket, (tsr_sim_cw_read(s, TSR_CW_ISSUE_STATUS) & TSR_ISSUE_STATUS_PUSH_TIMEOUT) ? " (push timeout)" : "");
            return 1;
        }
    }
    return 0;
}

// The predicted per-unit records and the DONE record (the replay kernel's answer, SW §7.7, with
// EX §4.4's status from the first error: the smallest push index over the units in error).
static void predicted_records(const gen_t * g, tsr_trace_unit u[TSR_NUM_UNITS], tsr_trace_done * d) {
    memset(u, 0, sizeof(tsr_trace_unit) * TSR_NUM_UNITS);
    uint32_t last_clear = 0;
    for (uint32_t i = 0; i < g->n_rec; ++i)
        if (g->rec[i].kind == TSR_TRACE_PUSH_QCTRL && g->rec[i].w[0] == TSR_QCTRL_QCLEAR) last_clear = i + 1;
    for (uint32_t i = 0; i < g->n_rec; ++i) {
        const rec_t * r = &g->rec[i];
        if (r->ticket == TSR_TICKET_NONE) continue;
        tsr_trace_unit * q = &u[TK_U(r->ticket)];
        q->count++;
        if (i < last_clear || r->want == TSR_ERR_NONE || q->err_code != 0) continue;
        q->err_code = r->want;
        q->err_rel_seq = TK_S(r->ticket);
        q->err_info = r->info;
        q->err_w0 = r->w[0];
        q->err_rel_pidx = r->pidx;
    }
    memset(d, 0, sizeof(*d));
    d->err_ticket_rel = TSR_TICKET_NONE;
    int first = -1;
    for (uint32_t k = 0; k < TSR_NUM_UNITS; ++k)
        if (u[k].err_code && (first < 0 || u[k].err_rel_pidx < u[first].err_rel_pidx)) first = (int)k;
    if (first < 0) return;
    const tsr_trace_unit * f = &u[first];
    d->status = f->err_code == TSR_ERR_ABORTED ? TSR_LCQ_R0_STATUS_ABORTED : TSR_LCQ_R0_STATUS_TILE_ERR;
    d->ret = (int32_t)(1u + f->err_rel_pidx);
    d->err_code = f->err_code;
    d->err_ticket_rel = TK((uint32_t)first, f->err_rel_seq);
    d->err_pidx_rel = f->err_rel_pidx;
    d->err_info = f->err_info;
    d->err_w0 = f->err_w0;
}

static int same_unit(const tsr_trace_unit * a, const tsr_trace_unit * b) {
    return a->count == b->count && a->err_code == b->err_code && a->err_rel_seq == b->err_rel_seq &&
           a->err_info == b->err_info && a->err_w0 == b->err_w0 && a->err_rel_pidx == b->err_rel_pidx;
}
static int same_done(const tsr_trace_done * a, const tsr_trace_done * b) {
    return a->status == b->status && a->ret == b->ret && a->err_code == b->err_code &&
           a->err_ticket_rel == b->err_ticket_rel && a->err_pidx_rel == b->err_pidx_rel &&
           a->err_info == b->err_info && a->err_w0 == b->err_w0;
}

static int same_regions(const tsr_region * a, size_t na, const tsr_region * b, size_t nb) {
    if (na != nb) return 0;
    for (size_t i = 0; i < na; ++i) {
        if (a[i].ch != b[i].ch || a[i].byte_off != b[i].byte_off || a[i].size != b[i].size) return 0;
        if (memcmp(a[i].data, b[i].data, (size_t)a[i].size) != 0) return 0;
    }
    return 1;
}

// Verification run (fixed schedule) with the listener, the serial check and the host checks.
// On success *cap holds the run's capture (the caller frees it). Returns 0 on success.
static int verify_stream(gen_t * g, unsigned idx, const region_t * reg, uint32_t n_reg, tsr_capture * cap) {
    int rc = 0;
    memset(cap, 0, sizeof(*cap));
    tsr_sim * s = fresh_sim(g, reg, n_reg);
    tsr_sim * s2 = fresh_sim(g, reg, n_reg);
    if (!s || !s2) {
        if (s) tsr_sim_destroy(s);
        if (s2) tsr_sim_destroy(s2);
        return 1;
    }
    evrec_t ev;
    ev.n = g->pidx;
    ev.code = (uint32_t *)xcalloc((ev.n + 1) * sizeof(uint32_t));
    ev.ran = (uint8_t *)xcalloc(ev.n + 1);
    ev.seen = (uint8_t *)xcalloc(ev.n + 1);
    const int lid = tsr_sim_add_listener(s, on_event, &ev);
    if (lid < 0) {
        fprintf(stderr, "tsr-trace-gen: no queue listener slot\n");
        exit(2);
    }
    tsr_sim_capture_begin(s);
    if (push_stream(g, s, idx) != 0) rc = 1;
    if (rc == 0 && !tsr_sim_run_until_idle(s, 1ull << 40)) {
        fprintf(stderr, "tsr-trace-gen: trace %u: the run did not reach QIDLE\n", idx);
        rc = 1;
    }
    tsr_sim_capture_end(s, cap);
    tsr_sim_remove_listener(s, lid);
    // every entry's completion code against the prediction
    for (uint32_t i = 0; i < g->n_rec && rc == 0; ++i) {
        const rec_t * r = &g->rec[i];
        if (r->ticket == TSR_TICKET_NONE) continue;
        if (ev.seen[r->pidx] != 1 || ev.code[r->pidx] != r->want || ev.ran[r->pidx] != r->ran) {
            fprintf(stderr, "tsr-trace-gen: trace %u: record %u (pidx %u, w0 0x%08x, ticket 0x%08x, deps 0x%08x 0x%08x) "
                            "completed %u time(s) with code %u ran %u; predicted code %u ran %u\n",
                    idx, i, r->pidx, r->w[0], r->ticket, r->dep[0], r->dep[1], ev.seen[r->pidx], ev.code[r->pidx],
                    ev.ran[r->pidx], r->want, r->ran);
            rc = 1;
        }
    }
    // the per-unit records against the prediction
    tsr_trace_unit pu[TSR_NUM_UNITS];
    tsr_trace_done pd;
    predicted_records(g, pu, &pd);
    for (uint32_t u = 0; u < TSR_NUM_UNITS && rc == 0; ++u) {
        tsr_trace_unit q;
        memset(&q, 0, sizeof(q));
        q.count = tsr_sim_csr_read(s, TSR_CSR_Q_DONE(u));
        q.err_code = tsr_sim_csr_read(s, TSR_CSR_Q_ERR_CODE(u));
        if (q.err_code) {
            q.err_rel_seq = tsr_sim_csr_read(s, TSR_CSR_Q_ERR_SEQ(u));
            q.err_info = tsr_sim_csr_read(s, TSR_CSR_Q_ERR_INFO(u));
            q.err_w0 = tsr_sim_csr_read(s, TSR_CSR_Q_ERR_W0(u));
            q.err_rel_pidx = tsr_sim_csr_read(s, TSR_CSR_Q_ERR_PIDX(u));
        }
        if (!same_unit(&q, &pu[u])) {
            fprintf(stderr, "tsr-trace-gen: trace %u: unit %u: count %u record {code %u seq %u info 0x%x w0 0x%08x pidx %u}, "
                            "predicted count %u {code %u seq %u info 0x%x w0 0x%08x pidx %u}\n", idx, u, q.count,
                    q.err_code, q.err_rel_seq, q.err_info, q.err_w0, q.err_rel_pidx, pu[u].count, pu[u].err_code,
                    pu[u].err_rel_seq, pu[u].err_info, pu[u].err_w0, pu[u].err_rel_pidx);
            rc = 1;
        }
    }
    // serial check: the ops that ran, one by one in push order
    if (rc == 0) {
        tsr_capture cap2;
        tsr_sim_capture_begin(s2);
        for (uint32_t i = 0; i < g->n_rec && rc == 0; ++i) {
            const rec_t * r = &g->rec[i];
            if (r->kind != TSR_TRACE_PUSH_OP || r->ticket == TSR_TICKET_NONE || !ev.ran[r->pidx]) continue;
            uint32_t info = 0;
            const uint32_t e = tsr_sim_op(s2, r->w, &info);
            if (e != TSR_ERR_NONE) {
                fprintf(stderr, "tsr-trace-gen: trace %u: serial check: record %u fails with %u\n", idx, i, e);
                rc = 1;
            }
        }
        tsr_sim_capture_end(s2, &cap2);
        if (rc == 0) {
            if (memcmp(tsr_sim_spm(s), tsr_sim_spm(s2), g->o->cfg.spm_bytes) != 0 ||
                memcmp(tsr_sim_sreg(s), tsr_sim_sreg(s2), 16 * sizeof(float)) != 0 ||
                !same_regions(cap->expect, cap->n_expect, cap2.expect, cap2.n_expect) ||
                !same_regions(cap->init, cap->n_init, cap2.init, cap2.n_init)) {
                fprintf(stderr, "tsr-trace-gen: trace %u: serial check: SPM, s bank or DRAM differs from the queue run "
                                "(missing dependency?)\n", idx);
                rc = 1;
            }
        }
        tsr_capture_free(&cap2);
    }
    // host references of the outputs (v1), and stores that must not have run
    for (size_t i = 0; i < g->n_chk && rc == 0; ++i) {
        const check_t * k = &g->chk[i];
        const uint8_t * got = tsr_sim_dram(s, k->ch) + k->beat * 64;
        const size_t n = (size_t)k->lines * 64;
        size_t b = 0;
        if (k->want) {
            while (b < n && got[b] == k->want[b]) ++b;
        } else {
            while (b < n && got[b] == 0) ++b;
        }
        if (b < n) {
            fprintf(stderr, "tsr-trace-gen: trace %u: output %zu (ch %u beat %llu) %s at byte %zu\n", idx, i, k->ch,
                    (unsigned long long)k->beat, k->want ? "differs from the host reference" : "was written", b);
            rc = 1;
        }
    }
    free(ev.code);
    free(ev.ran);
    free(ev.seen);
    tsr_sim_destroy(s);
    tsr_sim_destroy(s2);
    if (rc) {
        tsr_capture_free(cap);
        memset(cap, 0, sizeof(*cap));
    }
    return rc;
}

// --ops m3 weight-traffic totals over the traces written (MVM weight beats; "busiest channel" is
// channel 0, which holds R = ceil(N/C) rows of every tensor).
static struct {
    unsigned long long ch0, all, max, over, mvms;
} m3_stats;
#define M3_BIG_BEATS 20000ull   // traces above this many weight beats on the busiest channel are reported

static struct {
    unsigned long long traces, records, bytes, init_bytes, expect_bytes;
} set_stats;

static void gen_init(gen_t * g, const opts_t * o, uint64_t seed, unsigned idx) {
    memset(g, 0, sizeof(*g));
    g->o = o;
    g->r.s = seed ^ (0xD1B54A32D192ED03ull * (idx + 1));
    rng_u64(&g->r);
    for (uint32_t u = 0; u < TSR_NUM_UNITS; ++u) g->fail_seq[u] = NOFAIL;
    g->barrier = TSR_TICKET_NONE;
    g->inj_pidx = NOFAIL;
}

static void gen_free(gen_t * g) {
    for (size_t i = 0; i < g->n_chk; ++i) free(g->chk[i].want);
    free(g->chk);
    free(g->rec);
    free(g->acc);
    free(g->pend);
    free(g->in);
    if (g->sim) tsr_sim_destroy(g->sim);
}

static int build_vu(gen_t * g, unsigned idx);
static int build_at(gen_t * g, unsigned idx);
static int build_ctrl(gen_t * g, unsigned idx);

// Generates, runs, verifies and writes trace `idx`. Returns 0 on success.
static int gen_trace(const opts_t * o, uint64_t seed, unsigned idx, const char * dir, unsigned long * cov_total) {
    gen_t g;
    gen_init(&g, o, seed, idx);
    tsr_sim_opts so;
    memset(&so, 0, sizeof(so));
    so.q_depth = o->depth;
    so.sched = "fixed";
    g.sim = tsr_sim_create_opts(&o->cfg, &so);
    if (!g.sim) return 1;
    g.spm_lines = o->cfg.spm_bytes / 64;
    g.lim = tsr_sim_dram_limit_beats(g.sim);
    const uint64_t io_base = TSR_REGION_IO_OFF / 64, w_base = TSR_REGION_WEIGHT_OFF / 64;
    if (o->ops == OPS_M1 && w_base + 1024 >= g.lim) {
        fprintf(stderr, "tsr-trace-gen: channel too small for the DRAM map (WEIGHT region at 512 MiB)\n");
        gen_free(&g);
        return 1;
    }
    if (io_base + 65536 >= g.lim) {
        fprintf(stderr, "tsr-trace-gen: addressable channel too small for the IO region (64 MiB + 4 MiB)\n");
        gen_free(&g);
        return 1;
    }
    g.spm_hi = g.spm_lines;
    for (uint32_t c = 0; c < o->cfg.num_ch; ++c) g.top_next[c] = g.lim;
    for (uint32_t c = 0; c < o->cfg.num_ch; ++c) g.io_next[c] = io_base + rng_below(&g.r, 4096);
    g.heap_next = w_base + rng_below(&g.r, 4096);
    g.spm_next = rng_below(&g.r, 64);
    if (o->m3) {   // weights in the WEIGHT region, or in the upper half of [IO, limit) if that is beyond the limit
        g.w_lo = (w_base + 65536 < g.lim) ? w_base : io_base + (g.lim - io_base) / 2;
        g.io_lim = g.w_lo;
        g.heap_next = g.w_lo + rng_below(&g.r, 4096);
        g.wbudget = o->max_wbeats;
        if ((idx * o->big_pct) % 100 < o->big_pct) g.wbudget *= 4;      // every (100 / PCT)-th trace
    }
    int rc;
    switch (o->ops) {
    case OPS_VU:   rc = build_vu(&g, idx); break;
    case OPS_AT:   rc = build_at(&g, idx); break;
    case OPS_CTRL: rc = build_ctrl(&g, idx); break;
    default:       rc = build_v1(&g, idx); break;
    }
    if (rc == 0 && g.n_pend) flush_groups(&g);
    if (rc == 0 && g.hold) {
        fprintf(stderr, "tsr-trace-gen: internal: trace %u ends inside a hold window\n", idx);
        rc = 1;
    }
    uint32_t n_reg = 0;
    region_t * reg = rc == 0 ? merged_inputs(&g, &n_reg) : NULL;
    tsr_capture vcap;
    memset(&vcap, 0, sizeof(vcap));
    if (rc == 0 && o->verify) rc = verify_stream(&g, idx, reg, n_reg, &vcap);

    // The trace: config, push stream, inputs; tsr_trace2_run_ops fills done, tickets, expect, perf
    // and trims init to what the run read.
    tsr_trace2 * t = rc == 0 ? tsr_trace2_new() : NULL;
    if (t) {
        tsr_trace2_config_from_core(&t->cfg, &o->cfg);
        memcpy(t->cfg.q_depth, o->depth, sizeof(t->cfg.q_depth));
        for (uint32_t i = 0; i < g.n_rec; ++i) {
            const rec_t * r = &g.rec[i];
            if (r->kind == TSR_TRACE_PUSH_OP) tsr_trace2_add_op(t, r->w, r->dep[0], r->dep[1], r->push);
            else if (r->kind == TSR_TRACE_PUSH_QCTRL) tsr_trace2_add_qctrl(t, r->w[0]);
            else tsr_trace2_add_mark(t, r->w[0]);
        }
        for (uint32_t i = 0; i < n_reg; ++i)
            tsr_trace2_add_init(t, reg[i].ch, reg[i].lo * 64, tsr_sim_dram(g.sim, reg[i].ch) + reg[i].lo * 64,
                                (reg[i].hi - reg[i].lo) * 64);
        tsr_trace2_replay_opts ro;
        memset(&ro, 0, sizeof(ro));
        ro.sched = "fixed";
        ro.init_from_reads = 1;
        tsr_trace2_result res;
        if (tsr_trace2_run_ops(t, &ro, &res) != 0) {
            fprintf(stderr, "tsr-trace-gen: trace %u: tsr_trace2_run_ops: %s\n", idx, res.msg);
            rc = 1;
        }
        t->cfg.ctl_id = tsr_trace2_ctl_id_full();   // W1b fix: not the build's CTL_ID (tsr_trace.h)
        if (rc == 0) {
            tsr_trace_unit pu[TSR_NUM_UNITS];
            tsr_trace_done pd;
            predicted_records(&g, pu, &pd);
            int bad = !same_done(&t->done, &pd);
            for (uint32_t u = 0; u < TSR_NUM_UNITS; ++u) bad |= !same_unit(&t->unit[u], &pu[u]);
            if (o->verify) bad |= !same_regions(t->mem.expect, t->mem.n_expect, vcap.expect, vcap.n_expect);
            // the intent: the first error is the first injected fault since the last QCLEAR, else none
            const int intent = g.inj_pidx == NOFAIL ? t->done.err_code == 0
                                                    : t->done.err_code != 0 && t->done.err_pidx_rel == g.inj_pidx;
            if (!intent) {
                fprintf(stderr, "tsr-trace-gen: trace %u: first error code %u at push index %u, the injected fault is at %u\n",
                        idx, t->done.err_code, t->done.err_pidx_rel, g.inj_pidx);
                bad = 1;
            }
            if (bad) {
                fprintf(stderr, "tsr-trace-gen: trace %u: the trace run disagrees with the prediction / verification run "
                                "(done status %u ret %d code %u; predicted status %u ret %d code %u)\n", idx,
                        t->done.status, t->done.ret, t->done.err_code, pd.status, pd.ret, pd.err_code);
                rc = 1;
            }
        }
        for (unsigned k = 0; k < o->check_sched && rc == 0; ++k) {
            char sched[64];
            snprintf(sched, sizeof(sched), "random:%llu", (unsigned long long)((seed * 1000003ull + idx) * 31ull + k + 1));
            tsr_trace2_result rr;
            if (tsr_trace2_replay_sim(t, NULL, sched, &rr) != 0) {
                fprintf(stderr, "tsr-trace-gen: trace %u: replay under %s fails: %s\n", idx, sched, rr.msg);
                rc = 1;
            }
        }
        if (rc == 0 && tsr_trace2_write(t, dir, NULL) != 0) rc = 1;
        if (rc == 0) {
            set_stats.traces++;
            set_stats.records += t->n_push;
            for (size_t i = 0; i < t->mem.n_init; ++i) set_stats.init_bytes += t->mem.init[i].size;
            for (size_t i = 0; i < t->mem.n_expect; ++i) set_stats.expect_bytes += t->mem.expect[i].size;
            g.cov[COV_ERRC + (t->done.err_code & 15u)]++;
            for (uint32_t i = 0; i < g.n_rec; ++i)
                if (g.rec[i].ticket != TSR_TICKET_NONE) g.cov[COV_ENTC + (g.rec[i].want & 15u)]++;
        }
        if (rc == 0 && !o->quiet) {
            size_t bytes = 0;
            for (size_t i = 0; i < t->mem.n_init; ++i) bytes += (size_t)t->mem.init[i].size;
            printf("%s: %u records, %zu outputs checked, %zu init regions (%.1f KiB), done status %u code %u",
                   dir, g.n_rec, g.n_chk, t->mem.n_init, (double)bytes / 1024.0, t->done.status, t->done.err_code);
            if (o->m3) printf(", %u MVMs, %llu weight beats (ch0)", g.n_mvm, (unsigned long long)g.wch0);
            printf("\n");
        }
        tsr_trace2_free(t);
    }
    if (rc == 0 && o->m3) {
        m3_stats.ch0 += g.wch0;
        m3_stats.all += g.wall;
        m3_stats.mvms += g.n_mvm;
        if (g.wch0 > m3_stats.max) m3_stats.max = g.wch0;
        m3_stats.over += g.wch0 > M3_BIG_BEATS;
    }
    if (rc == 0) for (int i = 0; i < COV_MAX; ++i) cov_total[i] += g.cov[i];
    tsr_capture_free(&vcap);
    free(reg);
    gen_free(&g);
    return rc;
}

// ---------------------------------------------------------------------------
// VU programs (--ops vu): VOP RED ROPE TOPK SOP DQROW (VU §5.2-§5.9) on data loaded by LD and
// stored by ST. Every group loads its sources (random f32 incl. ±0, subnormals, ±Inf, NaN payloads,
// max finite; I32 positions; freq-factor / A tables; repacked Q8_0 / Q4_K rows), writes the s
// registers it reads with SOP.MOVI (VU §4.4), preloads destination spans with gap lines (lines an
// op does not write keep their content), runs the op and stores what it wrote. Shapes cover the
// n = 0 rule (m = 0, rows = 0, T' = 0), tails (m mod 16 != 0), stride 0 sources (broadcast),
// destination gaps, in-place ops (source = destination exactly, VU §4.5), every VOP function and
// cmode, every RED kind / post / TO_S, every ROPE mode and form (recurrence with FF / FS, ATAB),
// TOPK k = 1..16, every SOP sub, DQROW both formats. Faults: reserved bits, undefined enumerations,
// fields the op or function does not use, every OPERAND condition of VU §5.9, SPM_RANGE on each
// operand in field order (with lower-priority violations), DQROW BAD_FMT / K_RANGE, VMASK and the
// reserved VU opcodes (ILLEGAL_OP on SYNC), TOPK / DQROW without their CAPS bits.
// ---------------------------------------------------------------------------
enum { CV_VOPF = 0, CV_REDK = CV_VOPF + 25, CV_ROPE = CV_REDK + 14, CV_TOPK = CV_ROPE + 10, CV_SOP = CV_TOPK + 1,
       CV_DQROW = CV_SOP + 15, CV_N0 = CV_DQROW + 2, CV_TAIL, CV_BCAST, CV_GAP, CV_INPL, CV_FAULT, CV_END };
_Static_assert(CV_END <= COV_V2, "VU coverage slots overlap");
#define VU_LEGAL_KINDS CV_N0    // the legal-op cycle: 25 VOP functions, 14 RED forms, 10 ROPE forms, TOPK, 15 SOP, 2 DQROW

static uint32_t vu_val(rng_t * r) {
    const uint32_t sign = rng_u32(r) & 0x80000000u;
    switch (rng_below(r, 24)) {
    case 0:  return sign;                                              // ±0
    case 1:  return sign | (1 + rng_below(r, 0x7FFFFF));               // subnormal (DAZ)
    case 2:  return sign | 0x7F800000u;                                // ±Inf
    case 3:  return sign | 0x7F800001u | rng_u32(r) >> 9;              // NaN payloads
    case 4:  return sign | 0x7F7FFFFFu;                                // max finite
    case 5:  return sign | 0x00800000u;                                // min normal
    case 6:  return f2u((float)((int)rng_below(r, 64) - 32));          // small integers (ties, equal values)
    default: return f2u((float)(rng_normal(r) * ldexp(1.0, (int)rng_below(r, 12) - 6)));
    }
}

// Lines [line, line + n) of fresh SPM (bump allocator, 0-3 line gaps). 0 lines -> a random line in
// [0, SPM lines] (an empty operand may start anywhere up to the end).
static uint32_t vu_take(gen_t * g, uint64_t n) {
    if (n == 0) return rng_below(&g->r, 2) ? g->spm_lines : rng_below(&g->r, g->spm_lines + 1);
    const uint32_t a = g->spm_next + rng_below(&g->r, 4);
    g->spm_next = a + (uint32_t)n;
    return a;
}

// An SPM source of n lines: data (mode 0 random f32, 1 I32 positions, 2 positive table values,
// 3 bounded f32 for exponentials) written to fresh DRAM and loaded by LD. Returns its line.
static uint32_t vu_input(gen_t * g, uint64_t n, int mode) {
    const uint32_t line = vu_take(g, n);
    if (n == 0) return line;
    rng_t * r = &g->r;
    uint8_t * b = (uint8_t *)xmalloc((size_t)n * 64);
    for (size_t i = 0; i < (size_t)n * 16; ++i) {
        uint32_t v;
        switch (mode) {
        case 1:  v = rng_below(r, 8) == 0 ? rng_u32(r) : rng_below(r, 40000); break;
        case 2:  v = f2u((float)(0.05 + 8.0 * rng_unit(r))); break;
        case 3:  v = rng_below(r, 16) == 0 ? vu_val(r) : f2u((float)(40.0 * rng_unit(r) - 20.0)); break;
        default: v = vu_val(r); break;
        }
        put_u32(b + 4 * i, v);
    }
    const uint32_t ch = rng_below(r, g->o->cfg.num_ch);
    const uint64_t beat = take_io(g, ch, n);
    dram_put(g, ch, beat, b, (size_t)n * 64);
    free(b);
    pend(g, tsr_enc_ld(ch, line, (uint32_t)beat, (uint32_t)n));
    return line;
}

// A destination of n lines; with `preload`, loaded first (gap lines, in place, partial writes).
static uint32_t vu_output(gen_t * g, uint64_t n, int preload) {
    return preload ? vu_input(g, n, 0) : vu_take(g, n);
}
static void vu_store(gen_t * g, uint32_t line, uint64_t n) {
    if (n == 0) return;
    const uint32_t ch = rng_below(&g->r, g->o->cfg.num_ch);
    pend(g, tsr_enc_st(ch, line, (uint32_t)take_io(g, ch, n), (uint32_t)n));
}
static void vu_movi(gen_t * g, uint32_t s, uint32_t bits) {
    tsr_insn i = {{0}};
    i.w[0] = TSR_OP_SOP;
    TSR_INSN_SET(i.w, TSR_SOP_SUB, TSR_SOP_SUB_MOVI);
    TSR_INSN_SET(i.w, TSR_SOP_SD, s);
    i.w[2] = bits;
    pend(g, i);
}
static uint32_t vu_sreg(gen_t * g) { return rng_below(&g->r, 16); }

static uint32_t vlines(uint32_t n) { return (n + 15u) / 16u; }
static uint64_t vext(uint32_t n, uint64_t s, uint64_t len) { return (n == 0 || len == 0) ? 0 : (uint64_t)(n - 1) * s + len; }

static uint32_t pick_m(rng_t * r) {
    switch (rng_below(r, 12)) {
    case 0:  return 0;
    case 1:  return 1 + rng_below(r, 15);
    case 2:  return 16 * (1 + rng_below(r, 4));
    case 3:  return 16 * (1 + rng_below(r, 16)) + 1 + rng_below(r, 15);
    default: return 1 + rng_below(r, 300);
    }
}
static uint32_t pick_rows(rng_t * r) {
    switch (rng_below(r, 10)) {
    case 0:  return 0;
    case 1:
    case 2:  return 1;
    default: return 2 + rng_below(r, 7);
    }
}
// A destination stride for rows rows of len lines (>= len when rows > 1; anything otherwise).
static uint32_t pick_dstride(rng_t * r, uint32_t rows, uint32_t len, int zero_means_len) {
    if (rows <= 1) return rng_below(r, 4) ? 0 : rng_below(r, 3 * len + 3);
    if (zero_means_len && rng_below(r, 3) == 0) return 0;
    return len + (rng_below(r, 3) == 0 ? 1 + rng_below(r, 3) : 0);
}
// A source stride: 0 (broadcast) sometimes, else >= len with gaps or overlapping rows.
static uint32_t pick_sstride(rng_t * r, uint32_t rows, uint32_t len) {
    if (rows <= 1) return rng_below(r, 4) ? 0 : rng_below(r, 3 * len + 3);
    switch (rng_below(r, 6)) {
    case 0:  return 0;
    case 1:  return len > 1 ? 1 + rng_below(r, len - 1) : len;
    default: return len + (rng_below(r, 3) == 0 ? 1 + rng_below(r, 3) : 0);
    }
}

// The VOP of function func. alloc = 1: allocate, load, pend the op and its stores; alloc = 0: only
// encode with random in-range operands (for faults). Returns the words in w.
static void vu_vop(gen_t * g, uint32_t func, int alloc, uint32_t w[8]) {
    rng_t * r = &g->r;
    const uint32_t reads = tsr_vop_func_reads(func);
    const int ra = (reads & TSR_VOP_READS_A) != 0, rb = (reads & TSR_VOP_READS_B) != 0;
    const uint32_t cmode = (reads & TSR_VOP_READS_C) ? 1 + rng_below(r, 2) : 0;
    const uint32_t m = pick_m(r), rows = pick_rows(r), L = vlines(m);
    const uint32_t ds = pick_dstride(r, rows, L, 0);
    uint32_t as = ra ? pick_sstride(r, rows, L) : 0, bsd = rb ? pick_sstride(r, rows, L) : 0;
    const int inpl = ra && rows > 0 && m > 0 && rng_below(r, 5) == 0;
    if (inpl) as = ds;
    const uint64_t ed = vext(rows, ds, L), ea = vext(rows, as, L), eb = vext(rows, bsd, L);
    const uint64_t ec = cmode == TSR_VOP_CMODE_VEC ? vlines(rows) : 0;
    uint32_t d, a = 0, b = 0, c = 0, si = 0;
    const uint32_t emode = (func == TSR_VOP_FUNC_EXP || func == TSR_VOP_FUNC_EXPSUB || func == TSR_VOP_FUNC_EXPSUBC) ? 3 : 0;
    if (alloc) {
        if (cmode == TSR_VOP_CMODE_SREG) {
            si = vu_sreg(g);
            vu_movi(g, si, vu_val(r));
        }
        const int gaps = rows > 1 && ds > L;
        d = vu_output(g, ed, gaps || inpl);
        if (ra) a = inpl ? d : vu_input(g, ea, (int)emode);
        if (rb) b = vu_input(g, eb, (int)emode);
        if (cmode == TSR_VOP_CMODE_VEC) c = vu_input(g, ec, (int)emode);
        g->cov[CV_N0] += ed == 0;
        g->cov[CV_TAIL] += (m % 16) != 0;
        g->cov[CV_BCAST] += rows > 1 && ((ra && as == 0) || (rb && bsd == 0));
        g->cov[CV_GAP] += gaps;
        g->cov[CV_INPL] += inpl;
        g->cov[CV_VOPF + func]++;
    } else {
        const uint32_t S = g->spm_lines;
        d = rng_below(r, (uint32_t)(S - ed + 1));
        if (ra) a = rng_below(r, (uint32_t)(S - ea + 1));
        if (rb) b = rng_below(r, (uint32_t)(S - eb + 1));
        if (cmode == TSR_VOP_CMODE_VEC) c = rng_below(r, (uint32_t)(S - ec + 1));
        if (cmode == TSR_VOP_CMODE_SREG) si = vu_sreg(g);
    }
    memset(w, 0, 32);
    w[0] = TSR_OP_VOP;
    TSR_INSN_SET(w, TSR_VOP_CMODE, cmode);
    TSR_INSN_SET(w, TSR_VOP_FUNC, func);
    TSR_INSN_SET(w, TSR_VOP_SI, si);
    TSR_INSN_SET(w, TSR_VOP_D, d);
    TSR_INSN_SET(w, TSR_VOP_A, a);
    TSR_INSN_SET(w, TSR_VOP_B, b);
    TSR_INSN_SET(w, TSR_VOP_C, c);
    TSR_INSN_SET(w, TSR_VOP_M, m);
    TSR_INSN_SET(w, TSR_VOP_ROWS, rows);
    TSR_INSN_SET(w, TSR_VOP_D_STRIDE, ds);
    TSR_INSN_SET(w, TSR_VOP_A_STRIDE, as);
    TSR_INSN_SET(w, TSR_VOP_B_STRIDE, bsd);
    if (alloc) {
        tsr_insn i;
        memcpy(i.w, w, 32);
        pend(g, i);
        vu_store(g, d, ed);
    }
}

// SOP STS of s[s] into a preloaded line, then the line stored (makes an s register observable).
static void vu_observe_s(gen_t * g, uint32_t s) {
    const uint32_t line = vu_input(g, 1, 0);
    tsr_insn i = {{0}};
    i.w[0] = TSR_OP_SOP;
    TSR_INSN_SET(i.w, TSR_SOP_SUB, TSR_SOP_SUB_STS);
    TSR_INSN_SET(i.w, TSR_SOP_SA, s);
    TSR_INSN_SET(i.w, TSR_SOP_ADDR, 16u * line + rng_below(&g->r, 16));
    pend(g, i);
    vu_store(g, line, 1);
}

// RED form sel (0..13): kind = sel / 6 (SUM, SUMSQ: NONE / RCP / RMS; MAX: NONE), TO_S = sel & 1.
static void vu_red(gen_t * g, uint32_t sel, int alloc, uint32_t w[8]) {
    rng_t * r = &g->r;
    static const uint8_t forms[14][2] = { {0,0},{0,0},{0,1},{0,1},{0,2},{0,2},{1,0},{1,0},{1,1},{1,1},{1,2},{1,2},{2,0},{2,0} };
    const uint32_t kind = forms[sel % 14][0], post = forms[sel % 14][1], to_s = sel & 1u;
    const uint32_t m = pick_m(r), rows = to_s ? 1 : pick_rows(r), L = vlines(m);
    const uint32_t as = pick_sstride(r, rows, L);
    const uint64_t ed = to_s ? 0 : vlines(rows), ea = vext(rows, as, L);
    uint32_t d = 0, a, sd = 0, sk = 0, se = 0;
    if (to_s) sd = vu_sreg(g);
    if (post == TSR_RED_POST_RMS) {
        sk = vu_sreg(g);
        se = vu_sreg(g);
    }
    if (alloc) {
        if (post == TSR_RED_POST_RMS) {
            vu_movi(g, sk, f2u(m ? 1.0f / (float)m : 1.0f));
            vu_movi(g, se, rng_below(r, 4) ? f2u(1e-6f) : vu_val(r));
        }
        if (!to_s) d = vu_output(g, ed, 1);
        a = vu_input(g, ea, 0);
        g->cov[CV_N0] += !to_s && ed == 0;
        g->cov[CV_REDK + sel % 14]++;
    } else {
        const uint32_t S = g->spm_lines;
        if (!to_s) d = rng_below(r, (uint32_t)(S - ed + 1));
        a = rng_below(r, (uint32_t)(S - ea + 1));
    }
    memset(w, 0, 32);
    w[0] = TSR_OP_RED;
    TSR_INSN_SET(w, TSR_RED_KIND, kind);
    TSR_INSN_SET(w, TSR_RED_TO_S, to_s);
    TSR_INSN_SET(w, TSR_RED_SD, sd);
    TSR_INSN_SET(w, TSR_RED_POST, post);
    TSR_INSN_SET(w, TSR_RED_SK, sk);
    TSR_INSN_SET(w, TSR_RED_SE, se);
    TSR_INSN_SET(w, TSR_RED_D, d);
    TSR_INSN_SET(w, TSR_RED_A, a);
    TSR_INSN_SET(w, TSR_RED_M, m);
    TSR_INSN_SET(w, TSR_RED_ROWS, rows);
    TSR_INSN_SET(w, TSR_RED_A_STRIDE, as);
    if (alloc) {
        tsr_insn i;
        memcpy(i.w, w, 32);
        pend(g, i);
        if (to_s) vu_observe_s(g, sd);
        else vu_store(g, d, ed);
    }
}

// ROPE form sel (0..9): mode = sel / 5, form = sel % 5 (0 recurrence, 1 FF, 2 FS, 3 FF + FS, 4 ATAB).
static void vu_rope(gen_t * g, uint32_t sel, int alloc, uint32_t w[8]) {
    rng_t * r = &g->r;
    const uint32_t mode = (sel / 5) & 1u, form = sel % 5;
    const uint32_t ff = form == 1 || form == 3, fs = form == 2 || form == 3, atab = form == 4;
    static const uint16_t hds[] = { 16, 32, 64, 64, 128, 128, 256, 512 };
    uint32_t hd = hds[rng_below(r, 8)];
    if (mode == TSR_ROPE_MODE_NEOX && hd < 32) hd = 32;
    const uint32_t q = mode == TSR_ROPE_MODE_NEOX ? 32 : 16;
    const uint32_t nd = rng_below(r, 3) ? hd : q * (1 + rng_below(r, hd / q));
    const uint32_t nh = rng_below(r, 12) == 0 ? 0 : 1 + rng_below(r, hd >= 256 ? 2 : 4);
    const uint32_t T = rng_below(r, 12) == 0 ? 0 : 1 + rng_below(r, 4);
    const uint32_t H = nh * hd / 16, Tp = nh ? T : 0;
    uint32_t ds = rng_below(r, 3) == 0 ? 0 : H + (rng_below(r, 3) == 0 ? 1 + rng_below(r, 3) : 0);
    uint32_t ss = rng_below(r, 3) == 0 ? 0 : H + rng_below(r, 3);
    const int inpl = Tp > 0 && rng_below(r, 5) == 0;
    if (inpl) ss = ds;
    const uint32_t dsp = ds ? ds : H, ssp = ss ? ss : H;
    const uint64_t ed = vext(Tp, dsp, H), es = vext(Tp, ssp, H), ep = vlines(Tp);
    const uint64_t et = ((ff || atab) && Tp > 0) ? (nd + 31) / 32 : 0;
    const uint32_t sm = vu_sreg(g), sts = atab ? 0 : vu_sreg(g), sfs = fs ? vu_sreg(g) : 0;
    uint32_t d, src, pos, table = 0;
    if (alloc) {
        if (Tp > 0) {
            vu_movi(g, sm, f2u(rng_below(r, 2) ? 1.0f : (float)(0.5 + rng_unit(r))));
            if (!atab) vu_movi(g, sts, f2u((float)exp(-2.0 / nd * log(rng_below(r, 2) ? 10000.0 : 1000000.0))));
            if (fs) vu_movi(g, sfs, f2u((float)(0.25 + 0.75 * rng_unit(r))));
        }
        const int gaps = Tp > 1 && dsp > H;
        d = vu_output(g, ed, gaps || inpl);
        src = inpl ? d : vu_input(g, es, 0);
        pos = vu_input(g, ep, 1);
        if (ff || atab) table = et ? vu_input(g, et, 2) : vu_take(g, 0);
        g->cov[CV_N0] += ed == 0;
        g->cov[CV_GAP] += gaps;
        g->cov[CV_INPL] += inpl;
        g->cov[CV_ROPE + sel % 10]++;
    } else {
        const uint32_t S = g->spm_lines;
        d = rng_below(r, (uint32_t)(S - ed + 1));
        src = rng_below(r, (uint32_t)(S - es + 1));
        pos = rng_below(r, (uint32_t)(S - ep + 1));
        if (ff || atab) table = rng_below(r, (uint32_t)(S - et + 1));
    }
    memset(w, 0, 32);
    w[0] = TSR_OP_ROPE;
    TSR_INSN_SET(w, TSR_ROPE_MODE, mode);
    TSR_INSN_SET(w, TSR_ROPE_FF, ff);
    TSR_INSN_SET(w, TSR_ROPE_FS, fs);
    TSR_INSN_SET(w, TSR_ROPE_ATAB, atab);
    TSR_INSN_SET(w, TSR_ROPE_SM, sm);
    TSR_INSN_SET(w, TSR_ROPE_STS, sts);
    TSR_INSN_SET(w, TSR_ROPE_SFS, sfs);
    TSR_INSN_SET(w, TSR_ROPE_D, d);
    TSR_INSN_SET(w, TSR_ROPE_SRC, src);
    TSR_INSN_SET(w, TSR_ROPE_POS, pos);
    TSR_INSN_SET(w, TSR_ROPE_TABLE, table);
    TSR_INSN_SET(w, TSR_ROPE_HEAD_DIM, hd);
    TSR_INSN_SET(w, TSR_ROPE_N_DIMS, nd);
    TSR_INSN_SET(w, TSR_ROPE_N_HEADS, nh);
    TSR_INSN_SET(w, TSR_ROPE_T, T);
    TSR_INSN_SET(w, TSR_ROPE_SRC_STRIDE, ss);
    TSR_INSN_SET(w, TSR_ROPE_D_STRIDE, ds);
    if (alloc) {
        tsr_insn i;
        memcpy(i.w, w, 32);
        pend(g, i);
        vu_store(g, d, ed);
    }
}

static void vu_topk(gen_t * g, int alloc, uint32_t w[8]) {
    rng_t * r = &g->r;
    const uint32_t k = 1 + rng_below(r, TSR_TOPK_MAX_K), kl = (k + 7) / 8;
    const uint32_t m = pick_m(r), rows = pick_rows(r), L = vlines(m);
    const uint32_t ds = pick_dstride(r, rows, kl, 1), as = pick_sstride(r, rows, L);
    const uint64_t ed = vext(rows, ds ? ds : kl, kl), ea = vext(rows, as, L);
    uint32_t d, a;
    if (alloc) {
        d = vu_output(g, ed, 1);
        a = vu_input(g, ea, 0);
        g->cov[CV_N0] += ed == 0 || m == 0;
        g->cov[CV_TOPK]++;
    } else {
        d = rng_below(r, (uint32_t)(g->spm_lines - ed + 1));
        a = rng_below(r, (uint32_t)(g->spm_lines - ea + 1));
    }
    memset(w, 0, 32);
    w[0] = TSR_OP_TOPK;
    TSR_INSN_SET(w, TSR_TOPK_K, k);
    TSR_INSN_SET(w, TSR_TOPK_D, d);
    TSR_INSN_SET(w, TSR_TOPK_A, a);
    TSR_INSN_SET(w, TSR_TOPK_M, m);
    TSR_INSN_SET(w, TSR_TOPK_ROWS, rows);
    TSR_INSN_SET(w, TSR_TOPK_D_STRIDE, ds);
    TSR_INSN_SET(w, TSR_TOPK_A_STRIDE, as);
    if (alloc) {
        tsr_insn i;
        memcpy(i.w, w, 32);
        pend(g, i);
        vu_store(g, d, ed);
    }
}

// SOP sub: operands written with MOVI first, the result made observable (STS + ST).
static void vu_sop(gen_t * g, uint32_t sub, int alloc, uint32_t w[8]) {
    rng_t * r = &g->r;
    const uint32_t uses = tsr_sop_uses(sub);
    const uint32_t sd = (uses & TSR_SOP_USES_SD) ? vu_sreg(g) : 0, sa = (uses & TSR_SOP_USES_SA) ? vu_sreg(g) : 0;
    const uint32_t sb = (uses & TSR_SOP_USES_SB) ? vu_sreg(g) : 0, imm = (uses & TSR_SOP_USES_IMM) ? vu_val(r) : 0;
    uint32_t addr = 0;
    if (alloc) {
        if (uses & TSR_SOP_USES_SA) vu_movi(g, sa, vu_val(r));
        if ((uses & TSR_SOP_USES_SB) && sb != sa) vu_movi(g, sb, vu_val(r));
        if (uses & TSR_SOP_USES_ADDR) addr = 16u * vu_input(g, 1, 0) + rng_below(r, 16);
        g->cov[CV_SOP + sub]++;
    } else if (uses & TSR_SOP_USES_ADDR) {
        addr = rng_below(r, g->o->cfg.spm_bytes / 4);
    }
    memset(w, 0, 32);
    w[0] = TSR_OP_SOP;
    TSR_INSN_SET(w, TSR_SOP_SUB, sub);
    TSR_INSN_SET(w, TSR_SOP_SD, sd);
    TSR_INSN_SET(w, TSR_SOP_SA, sa);
    TSR_INSN_SET(w, TSR_SOP_SB, sb);
    TSR_INSN_SET(w, TSR_SOP_IMM, imm);
    TSR_INSN_SET(w, TSR_SOP_ADDR, addr);
    if (alloc) {
        tsr_insn i;
        memcpy(i.w, w, 32);
        pend(g, i);
        if (sub == TSR_SOP_SUB_STS) vu_store(g, addr / 16, 1);
        else vu_observe_s(g, sd);
    }
}

static void vu_dqrow(gen_t * g, uint32_t fmt, int alloc, uint32_t w[8]) {
    rng_t * r = &g->r;
    const uint32_t bs = tsr_fmt_block_size(fmt);
    const uint32_t kcap = umin32(g->o->cfg.k_max, 4096);
    const uint32_t K = bs * (1 + rng_below(r, kcap / bs < 8 ? kcap / bs : 8));
    const uint32_t rows = rng_below(r, 8) == 0 ? 0 : 1 + rng_below(r, 4), L = vlines(K), rb = tsr_row_beats(fmt, K);
    const uint32_t ds = pick_dstride(r, rows, L, 1);
    const uint32_t ss = rng_below(r, 3) == 0 ? 0 : rb + rng_below(r, 3);
    const uint64_t ed = vext(rows, ds ? ds : L, L), es = vext(rows, ss ? ss : rb, rb);
    uint32_t d, src;
    if (alloc) {
        const int gaps = rows > 1 && ds > L;
        d = vu_output(g, ed, gaps);
        // repacked rows (as LD would bring them from a weight tensor), with stride gaps
        src = vu_take(g, es);
        if (es) {
            const size_t rbytes = (size_t)(K / bs) * tsr_fmt_ggml_block_bytes(fmt);
            uint8_t * img = (uint8_t *)xcalloc((size_t)es * 64);
            uint8_t * blocks = (uint8_t *)xmalloc(rbytes);
            rng_bytes(r, img, (size_t)es * 64);
            for (uint32_t k = 0; k < rows; ++k) {
                gen_row_blocks(r, fmt, K, blocks, g->o->special);
                tsr_repack_row(fmt, blocks, K, img + (size_t)k * (ss ? ss : rb) * 64);
            }
            const uint32_t ch = rng_below(r, g->o->cfg.num_ch);
            const uint64_t beat = take_io(g, ch, es);
            dram_put(g, ch, beat, img, (size_t)es * 64);
            pend(g, tsr_enc_ld(ch, src, (uint32_t)beat, (uint32_t)es));
            free(blocks);
            free(img);
        }
        g->cov[CV_DQROW + fmt]++;
    } else {
        d = rng_below(r, (uint32_t)(g->spm_lines - ed + 1));
        src = rng_below(r, (uint32_t)(g->spm_lines - es + 1));
    }
    memset(w, 0, 32);
    w[0] = TSR_OP_DQROW;
    TSR_INSN_SET(w, TSR_DQROW_FMT, fmt);
    TSR_INSN_SET(w, TSR_DQROW_D, d);
    TSR_INSN_SET(w, TSR_DQROW_SRC, src);
    TSR_INSN_SET(w, TSR_DQROW_K, K);
    TSR_INSN_SET(w, TSR_DQROW_ROWS, rows);
    TSR_INSN_SET(w, TSR_DQROW_SRC_STRIDE, ss);
    TSR_INSN_SET(w, TSR_DQROW_D_STRIDE, ds);
    if (alloc) {
        tsr_insn i;
        memcpy(i.w, w, 32);
        pend(g, i);
        vu_store(g, d, ed);
    }
}

// Legal op number k of the cycle (VU_LEGAL_KINDS entries); TOPK / DQROW without their CAPS bits
// are replaced by a VOP.
static void vu_legal(gen_t * g, uint32_t k, int alloc, uint32_t w[8]) {
    k %= VU_LEGAL_KINDS;
    const uint32_t caps = g->o->cfg.caps;
    if (k < CV_REDK) vu_vop(g, k, alloc, w);
    else if (k < CV_ROPE) vu_red(g, k - CV_REDK, alloc, w);
    else if (k < CV_TOPK) vu_rope(g, k - CV_ROPE, alloc, w);
    else if (k < CV_SOP) {
        if (caps & TSR_CAPS_TOPK) vu_topk(g, alloc, w);
        else vu_vop(g, rng_below(&g->r, 25), alloc, w);
    } else if (k < CV_DQROW) vu_sop(g, k - CV_SOP, alloc, w);
    else {
        if (caps & TSR_CAPS_DQROW) vu_dqrow(g, k - CV_DQROW, alloc, w);
        else vu_vop(g, rng_below(&g->r, 25), alloc, w);
    }
}

// A random set reserved bit of the op's encoding table (EX §7.1, VU §5.1).
static void set_rsv_bit(rng_t * r, uint32_t w[8], const uint32_t rsv[8]) {
    uint32_t n = 0;
    for (int i = 0; i < 8; ++i) n += (uint32_t)__builtin_popcount(rsv[i]);
    uint32_t k = rng_below(r, n);
    for (int i = 0; i < 8; ++i)
        for (uint32_t b = 0; b < 32; ++b)
            if (((rsv[i] >> b) & 1u) && k-- == 0) { w[i] |= 1u << b; return; }
}

// SPM operand start beyond the SPM for extent e (24-bit line fields): just past, far, or the field maximum.
static uint32_t past_spm24(gen_t * g, uint64_t e) {
    const uint64_t S = g->spm_lines;
    switch (rng_below(&g->r, 3)) {
    case 0:  return e == 0 ? (uint32_t)(S + 1 + rng_below(&g->r, 64)) : (uint32_t)(S - umin64(e, S) + 1 + rng_below(&g->r, (uint32_t)umin64(e, 16)));
    case 1:  return (uint32_t)(S + 1 + rng_below(&g->r, 1u << 20));
    default: return 0x00FFFFFFu - rng_below(&g->r, 16);
    }
}

#define VU_FAULTS 40
// VU fault kind k (0 .. VU_FAULTS-1). Returns the record index.
static uint32_t vu_fault(gen_t * g, uint32_t k) {
    rng_t * r = &g->r;
    const uint32_t caps = g->o->cfg.caps;
    uint32_t w[8];
    uint32_t code = TSR_ERR_ILLEGAL_OP;
    int64_t info = 0;
    g->cov[CV_FAULT]++;
    for (;;) {
        switch (k % VU_FAULTS) {
        case 0: {   // VOP reserved bit
            static const uint32_t rsv[8] = { TSR_VOP_RSV_W0, TSR_VOP_RSV_W1, TSR_VOP_RSV_W2, TSR_VOP_RSV_W3,
                                             TSR_VOP_RSV_W4, TSR_VOP_RSV_W5, TSR_VOP_RSV_W6, TSR_VOP_RSV_W7 };
            vu_vop(g, rng_below(r, 25), 0, w);
            set_rsv_bit(r, w, rsv);
            break;
        }
        case 1:     // undefined VOP function
            vu_vop(g, rng_below(r, 25), 0, w);
            TSR_INSN_SET(w, TSR_VOP_FUNC, 25 + rng_below(r, 231));
            break;
        case 2:     // undefined cmode
            vu_vop(g, rng_below(r, 25), 0, w);
            TSR_INSN_SET(w, TSR_VOP_CMODE, 3 + rng_below(r, 13));
            break;
        case 3: {   // a / a_stride of a function that reads no a (SETC)
            vu_vop(g, TSR_VOP_FUNC_SETC, 0, w);
            if (rng_below(r, 2)) TSR_INSN_SET(w, TSR_VOP_A, 1 + rng_below(r, 100));
            else TSR_INSN_SET(w, TSR_VOP_A_STRIDE, 1 + rng_below(r, 100));
            break;
        }
        case 4: {   // b / b_stride of a function that reads no b
            static const uint8_t fa[] = { 4, 5, 6, 8, 9, 0x0B, 0x10, 0x11, 0x12, 0x14, 0x15, 0x16, 0x17 };
            vu_vop(g, fa[rng_below(r, sizeof(fa))], 0, w);
            if (rng_below(r, 2)) TSR_INSN_SET(w, TSR_VOP_B, 1 + rng_below(r, 100));
            else TSR_INSN_SET(w, TSR_VOP_B_STRIDE, 1 + rng_below(r, 100));
            break;
        }
        case 5:     // c line without cmode 2
            vu_vop(g, TSR_VOP_FUNC_MULC, 0, w);
            TSR_INSN_SET(w, TSR_VOP_CMODE, TSR_VOP_CMODE_SREG);
            TSR_INSN_SET(w, TSR_VOP_C, 1 + rng_below(r, 100));
            break;
        case 6:     // si without cmode 1
            vu_vop(g, TSR_VOP_FUNC_ADDC, 0, w);
            TSR_INSN_SET(w, TSR_VOP_CMODE, TSR_VOP_CMODE_VEC);
            TSR_INSN_SET(w, TSR_VOP_SI, 1 + rng_below(r, 15));
            break;
        case 7:     // a function reading c with cmode 0
            vu_vop(g, TSR_VOP_FUNC_SUBC, 0, w);
            TSR_INSN_SET(w, TSR_VOP_CMODE, 0);
            TSR_INSN_SET(w, TSR_VOP_SI, 0);
            TSR_INSN_SET(w, TSR_VOP_C, 0);
            break;
        case 8:     // a function without c with cmode != 0
            vu_vop(g, TSR_VOP_FUNC_ADD, 0, w);
            TSR_INSN_SET(w, TSR_VOP_CMODE, 1 + rng_below(r, 2));
            break;
        case 9: {   // VOP OPERAND: rows > 1 and d_stride < L(m) (ERR_INFO d_stride), maybe SPM too
            vu_vop(g, rng_below(r, 25), 0, w);
            const uint32_t m = 17 + rng_below(r, 200), L = vlines(m);
            TSR_INSN_SET(w, TSR_VOP_M, m);
            TSR_INSN_SET(w, TSR_VOP_ROWS, 2 + rng_below(r, 6));
            const uint32_t ds = rng_below(r, L);
            TSR_INSN_SET(w, TSR_VOP_D_STRIDE, ds);
            if (rng_below(r, 2)) TSR_INSN_SET(w, TSR_VOP_D, past_spm24(g, 0));
            code = TSR_ERR_OPERAND;
            info = ds;
            break;
        }
        case 10: case 11: case 12: case 13: {   // VOP SPM_RANGE on d, a, b, c (field order)
            const uint32_t opnd = (k % VU_FAULTS) - 10;
            static const uint8_t fn[4] = { 0x09, 0x02, 0x02, 0x0D };
            uint32_t m, rows, L, ds, as, bsd;
            do {
                vu_vop(g, fn[opnd], 0, w);
                m = TSR_INSN_GET(w, TSR_VOP_M);
                rows = TSR_INSN_GET(w, TSR_VOP_ROWS);
            } while (m == 0 || rows == 0);
            if (opnd == 3 && TSR_INSN_GET(w, TSR_VOP_CMODE) != TSR_VOP_CMODE_VEC) {
                TSR_INSN_SET(w, TSR_VOP_CMODE, TSR_VOP_CMODE_VEC);
                TSR_INSN_SET(w, TSR_VOP_SI, 0);
                TSR_INSN_SET(w, TSR_VOP_C, 0);
            }
            L = vlines(m);
            ds = TSR_INSN_GET(w, TSR_VOP_D_STRIDE);
            as = TSR_INSN_GET(w, TSR_VOP_A_STRIDE);
            bsd = TSR_INSN_GET(w, TSR_VOP_B_STRIDE);
            const uint64_t e[4] = { vext(rows, ds, L), vext(rows, as, L), vext(rows, bsd, L), vlines(rows) };
            static const uint32_t fld_word[4] = { 1, 2, 3, 4 };
            uint32_t a = past_spm24(g, e[opnd]);
            w[fld_word[opnd]] = a;
            for (uint32_t j = opnd + 1; j < 4; ++j)
                if (rng_below(r, 3) == 0 && (j < 3 || TSR_INSN_GET(w, TSR_VOP_CMODE) == TSR_VOP_CMODE_VEC) &&
                    (j != 2 || (tsr_vop_func_reads(TSR_INSN_GET(w, TSR_VOP_FUNC)) & TSR_VOP_READS_B)))
                    w[fld_word[j]] = past_spm24(g, e[j]);
            code = TSR_ERR_SPM_RANGE;
            info = a;
            break;
        }
        case 14: {  // RED reserved bit
            static const uint32_t rsv[8] = { TSR_RED_RSV_W0, TSR_RED_RSV_W1, TSR_RED_RSV_W2, TSR_RED_RSV_W3,
                                             TSR_RED_RSV_W4, TSR_RED_RSV_W5, TSR_RED_RSV_W6, TSR_RED_RSV_W7 };
            vu_red(g, rng_below(r, 14), 0, w);
            set_rsv_bit(r, w, rsv);
            break;
        }
        case 15:    // undefined RED kind
            vu_red(g, rng_below(r, 14), 0, w);
            TSR_INSN_SET(w, TSR_RED_KIND, 3 + rng_below(r, 13));
            break;
        case 16:    // post 3, or MAX with a post-op
            vu_red(g, rng_below(r, 2) ? 0 : 12, 0, w);
            TSR_INSN_SET(w, TSR_RED_POST, TSR_INSN_GET(w, TSR_RED_KIND) == TSR_RED_KIND_MAX ? 1 + rng_below(r, 3) : 3);
            break;
        case 17:    // sd without TO_S
            vu_red(g, 2 * rng_below(r, 7), 0, w);
            TSR_INSN_SET(w, TSR_RED_SD, 1 + rng_below(r, 15));
            break;
        case 18:    // d with TO_S
            vu_red(g, 2 * rng_below(r, 7) + 1, 0, w);
            TSR_INSN_SET(w, TSR_RED_D, 1 + rng_below(r, 1000));
            break;
        case 19:    // sk / se without post RMS
            vu_red(g, rng_below(r, 2) ? 0 : 2, 0, w);
            if (rng_below(r, 2)) TSR_INSN_SET(w, TSR_RED_SK, 1 + rng_below(r, 15));
            else TSR_INSN_SET(w, TSR_RED_SE, 1 + rng_below(r, 15));
            break;
        case 20: {  // RED OPERAND: TO_S with rows != 1 (ERR_INFO rows)
            vu_red(g, 2 * rng_below(r, 7) + 1, 0, w);
            const uint32_t rows = rng_below(r, 2) ? 0 : 2 + rng_below(r, 100);
            TSR_INSN_SET(w, TSR_RED_ROWS, rows);
            TSR_INSN_SET(w, TSR_RED_A, 0);
            code = TSR_ERR_OPERAND;
            info = rows;
            break;
        }
        case 21: case 22: { // RED SPM_RANGE on d (not TO_S), a
            const int ond = (k % VU_FAULTS) == 21;
            uint32_t rows, m;
            do {
                vu_red(g, 2 * rng_below(r, 7), 0, w);
                rows = TSR_INSN_GET(w, TSR_RED_ROWS);
                m = TSR_INSN_GET(w, TSR_RED_M);
            } while (rows == 0 || (!ond && m == 0));
            const uint64_t ed = vlines(rows), ea = vext(rows, TSR_INSN_GET(w, TSR_RED_A_STRIDE), vlines(m));
            const uint32_t a = past_spm24(g, ond ? ed : ea);
            if (ond) {
                TSR_INSN_SET(w, TSR_RED_D, a);
                if (rng_below(r, 3) == 0) TSR_INSN_SET(w, TSR_RED_A, past_spm24(g, ea));
            } else {
                TSR_INSN_SET(w, TSR_RED_A, a);
            }
            code = TSR_ERR_SPM_RANGE;
            info = a;
            break;
        }
        case 23: {  // ROPE reserved bit
            static const uint32_t rsv[8] = { TSR_ROPE_RSV_W0, TSR_ROPE_RSV_W1, TSR_ROPE_RSV_W2, TSR_ROPE_RSV_W3,
                                             TSR_ROPE_RSV_W4, TSR_ROPE_RSV_W5, TSR_ROPE_RSV_W6, TSR_ROPE_RSV_W7 };
            vu_rope(g, rng_below(r, 10), 0, w);
            set_rsv_bit(r, w, rsv);
            break;
        }
        case 24:    // undefined ROPE mode
            vu_rope(g, rng_below(r, 10), 0, w);
            TSR_INSN_SET(w, TSR_ROPE_MODE, 2 + rng_below(r, 14));
            break;
        case 25:    // ATAB with FF, FS or sts != 0 (ILLEGAL_OP before OPERAND, even with a bad head_dim)
            vu_rope(g, rng_below(r, 2) * 5 + 4, 0, w);
            switch (rng_below(r, 3)) {
            case 0:  TSR_INSN_SET(w, TSR_ROPE_FF, 1); break;
            case 1:  TSR_INSN_SET(w, TSR_ROPE_FS, 1); break;
            default: TSR_INSN_SET(w, TSR_ROPE_STS, 1 + rng_below(r, 15)); break;
            }
            if (rng_below(r, 2)) TSR_INSN_SET(w, TSR_ROPE_HEAD_DIM, 0);
            break;
        case 26:    // sfs without FS; a table line without FF or ATAB
            vu_rope(g, rng_below(r, 2) * 5, 0, w);
            if (rng_below(r, 2)) TSR_INSN_SET(w, TSR_ROPE_SFS, 1 + rng_below(r, 15));
            else TSR_INSN_SET(w, TSR_ROPE_TABLE, 1 + rng_below(r, 1000));
            break;
        case 27: {  // ROPE OPERAND head_dim: 0, not a multiple of 16, > 1024
            vu_rope(g, rng_below(r, 10), 0, w);
            static const uint32_t bad[] = { 0, 8, 24, 100, 1040, 2048, 0xFFFF };
            const uint32_t hd = bad[rng_below(r, 7)];
            TSR_INSN_SET(w, TSR_ROPE_HEAD_DIM, hd);
            code = TSR_ERR_OPERAND;
            info = hd;
            break;
        }
        case 28: {  // ROPE OPERAND n_dims: 0, > head_dim, NEOX % 32, NORM % 16
            vu_rope(g, rng_below(r, 10), 0, w);
            const uint32_t hd = TSR_INSN_GET(w, TSR_ROPE_HEAD_DIM), neox = TSR_INSN_GET(w, TSR_ROPE_MODE);
            uint32_t nd;
            switch (rng_below(r, 3)) {
            case 0:  nd = 0; break;
            case 1:  nd = hd + 16 * (1 + rng_below(r, 4)); break;
            default: nd = neox ? 16 + 32 * rng_below(r, hd / 32) : 1 + rng_below(r, 15); break;
            }
            TSR_INSN_SET(w, TSR_ROPE_N_DIMS, nd);
            code = TSR_ERR_OPERAND;
            info = nd;
            break;
        }
        case 29: {  // ROPE OPERAND: T > 1, n_heads > 0, d_stride' < H (ERR_INFO d_stride)
            uint32_t H;
            do {
                vu_rope(g, rng_below(r, 10), 0, w);
                H = TSR_INSN_GET(w, TSR_ROPE_N_HEADS) * TSR_INSN_GET(w, TSR_ROPE_HEAD_DIM) / 16;
            } while (H < 2);
            TSR_INSN_SET(w, TSR_ROPE_T, 2 + rng_below(r, 6));
            const uint32_t ds = 1 + rng_below(r, H - 1);
            TSR_INSN_SET(w, TSR_ROPE_D_STRIDE, ds);
            code = TSR_ERR_OPERAND;
            info = ds;
            break;
        }
        case 30: {  // ROPE SPM_RANGE on d / src / pos / table
            uint32_t Tp, H, nd, ff, atab;
            do {
                vu_rope(g, rng_below(r, 10), 0, w);
                Tp = TSR_INSN_GET(w, TSR_ROPE_N_HEADS) ? TSR_INSN_GET(w, TSR_ROPE_T) : 0;
                H = TSR_INSN_GET(w, TSR_ROPE_N_HEADS) * TSR_INSN_GET(w, TSR_ROPE_HEAD_DIM) / 16;
            } while (Tp == 0);
            nd = TSR_INSN_GET(w, TSR_ROPE_N_DIMS);
            ff = TSR_INSN_GET(w, TSR_ROPE_FF);
            atab = TSR_INSN_GET(w, TSR_ROPE_ATAB);
            const uint32_t ds = TSR_INSN_GET(w, TSR_ROPE_D_STRIDE), ss = TSR_INSN_GET(w, TSR_ROPE_SRC_STRIDE);
            const uint64_t e[4] = { vext(Tp, ds ? ds : H, H), vext(Tp, ss ? ss : H, H), vlines(Tp), (nd + 31) / 32 };
            const uint32_t nop = (ff || atab) ? 4 : 3, opnd = rng_below(r, nop);
            const uint32_t a = past_spm24(g, e[opnd]);
            w[1 + opnd] = a;
            code = TSR_ERR_SPM_RANGE;
            info = a;
            break;
        }
        case 31: {  // TOPK reserved bit / OPERAND k
            if (!(caps & TSR_CAPS_TOPK)) { k = 37; continue; }
            vu_topk(g, 0, w);
            if (rng_below(r, 2)) {
                static const uint32_t rsv[8] = { TSR_TOPK_RSV_W0, TSR_TOPK_RSV_W1, TSR_TOPK_RSV_W2, TSR_TOPK_RSV_W3,
                                                 TSR_TOPK_RSV_W4, TSR_TOPK_RSV_W5, TSR_TOPK_RSV_W6, TSR_TOPK_RSV_W7 };
                set_rsv_bit(r, w, rsv);
            } else {
                const uint32_t kk = rng_below(r, 2) ? 0 : 17 + rng_below(r, 15);
                TSR_INSN_SET(w, TSR_TOPK_K, kk);
                code = TSR_ERR_OPERAND;
                info = kk;
            }
            break;
        }
        case 32: {  // TOPK OPERAND: rows > 1 and d_stride' < ceil(k/8)
            if (!(caps & TSR_CAPS_TOPK)) { k = 37; continue; }
            vu_topk(g, 0, w);
            TSR_INSN_SET(w, TSR_TOPK_K, 9 + rng_below(r, 8));
            TSR_INSN_SET(w, TSR_TOPK_ROWS, 2 + rng_below(r, 6));
            TSR_INSN_SET(w, TSR_TOPK_D_STRIDE, 1);
            code = TSR_ERR_OPERAND;
            info = 1;
            break;
        }
        case 33: {  // SOP reserved bit / sub 15 / an unused field
            vu_sop(g, rng_below(r, 15), 0, w);
            const uint32_t v = rng_below(r, 3);
            if (v == 0) {
                static const uint32_t rsv[8] = { TSR_SOP_RSV_W0, TSR_SOP_RSV_W1, TSR_SOP_RSV_W2, TSR_SOP_RSV_W3,
                                                 TSR_SOP_RSV_W4, TSR_SOP_RSV_W5, TSR_SOP_RSV_W6, TSR_SOP_RSV_W7 };
                set_rsv_bit(r, w, rsv);
            } else if (v == 1) {
                TSR_INSN_SET(w, TSR_SOP_SUB, 15);
            } else {
                const uint32_t sub = TSR_INSN_GET(w, TSR_SOP_SUB), uses = tsr_sop_uses(sub);
                if (!(uses & TSR_SOP_USES_SB)) TSR_INSN_SET(w, TSR_SOP_SB, 1 + rng_below(r, 15));
                else if (!(uses & TSR_SOP_USES_IMM)) TSR_INSN_SET(w, TSR_SOP_IMM, 1 + rng_u32(r) % 0xFFFFFFFEu);
                else TSR_INSN_SET(w, TSR_SOP_ADDR, 1 + rng_below(r, 1000));
            }
            break;
        }
        case 34: {  // SOP LDS / STS word address + 1 > SPM_BYTES / 4 (ERR_INFO the address)
            vu_sop(g, rng_below(r, 2) ? TSR_SOP_SUB_LDS : TSR_SOP_SUB_STS, 0, w);
            const uint32_t words = g->o->cfg.spm_bytes / 4;
            const uint32_t a = rng_below(r, 2) ? words + rng_below(r, 64) : 0x0FFFFFFFu - rng_below(r, 16);
            TSR_INSN_SET(w, TSR_SOP_ADDR, a);
            code = TSR_ERR_SPM_RANGE;
            info = a;
            break;
        }
        case 35: {  // DQROW reserved bit / BAD_FMT / K_RANGE
            if (!(caps & TSR_CAPS_DQROW)) { k = 38; continue; }
            vu_dqrow(g, rng_below(r, 2), 0, w);
            const uint32_t v = rng_below(r, 3);
            if (v == 0) {
                static const uint32_t rsv[8] = { TSR_DQROW_RSV_W0, TSR_DQROW_RSV_W1, TSR_DQROW_RSV_W2, TSR_DQROW_RSV_W3,
                                                 TSR_DQROW_RSV_W4, TSR_DQROW_RSV_W5, TSR_DQROW_RSV_W6, TSR_DQROW_RSV_W7 };
                set_rsv_bit(r, w, rsv);
            } else if (v == 1) {
                TSR_INSN_SET(w, TSR_DQROW_FMT, 2 + rng_below(r, 14));
                if (rng_below(r, 2)) TSR_INSN_SET(w, TSR_DQROW_K, 0);
                code = TSR_ERR_BAD_FMT;
            } else {
                const uint32_t fmt = TSR_INSN_GET(w, TSR_DQROW_FMT), bs = tsr_fmt_block_size(fmt);
                static const uint32_t big[] = { 0xFFFFFFFFu, 0x80000000u };
                uint32_t K;
                switch (rng_below(r, 4)) {
                case 0:  K = 0; break;
                case 1:  K = bs * (1 + rng_below(r, 8)) + 16; break;
                case 2:  K = g->o->cfg.k_max + bs * (1 + rng_below(r, 4)); break;
                default: K = big[rng_below(r, 2)]; break;
                }
                TSR_INSN_SET(w, TSR_DQROW_K, K);
                code = TSR_ERR_K_RANGE;
            }
            break;
        }
        case 36: {  // DQROW OPERAND d_stride (rows > 1) or SPM_RANGE on d / src
            if (!(caps & TSR_CAPS_DQROW)) { k = 38; continue; }
            uint32_t rows;
            do {
                vu_dqrow(g, rng_below(r, 2), 0, w);
                rows = TSR_INSN_GET(w, TSR_DQROW_ROWS);
            } while (rows == 0);
            const uint32_t K = TSR_INSN_GET(w, TSR_DQROW_K), L = vlines(K);
            if (rng_below(r, 2)) {
                TSR_INSN_SET(w, TSR_DQROW_ROWS, 2 + rng_below(r, 4));
                const uint32_t ds = 1 + rng_below(r, L > 1 ? L - 1 : 1);
                TSR_INSN_SET(w, TSR_DQROW_D_STRIDE, ds < L ? ds : 0);
                if (ds >= L) { k = 35; continue; }
                code = TSR_ERR_OPERAND;
                info = ds;
            } else {
                const uint32_t fmt = TSR_INSN_GET(w, TSR_DQROW_FMT), ds = TSR_INSN_GET(w, TSR_DQROW_D_STRIDE);
                const uint32_t ss = TSR_INSN_GET(w, TSR_DQROW_SRC_STRIDE), rb = tsr_row_beats(fmt, K);
                const int ond = rng_below(r, 2);
                const uint32_t a = past_spm24(g, ond ? vext(rows, ds ? ds : L, L) : vext(rows, ss ? ss : rb, rb));
                if (ond) TSR_INSN_SET(w, TSR_DQROW_D, a);
                else TSR_INSN_SET(w, TSR_DQROW_SRC, a);
                code = TSR_ERR_SPM_RANGE;
                info = a;
            }
            break;
        }
        case 37:    // TOPK without CAPS.TOPK, VMASK (M9), reserved VU opcodes: refused, ILLEGAL_OP on SYNC
        case 38:
        case 39: {
            static const uint8_t ops[] = { TSR_OP_VMASK, 0x13, 0x39, 0x3A, 0x3F, 0x41, 0x44, 0x47 };
            uint32_t op = ops[rng_below(r, sizeof(ops))];
            if (!(caps & TSR_CAPS_TOPK) && rng_below(r, 2)) op = TSR_OP_TOPK;
            if (!(caps & TSR_CAPS_DQROW) && rng_below(r, 2)) op = TSR_OP_DQROW;
            vu_vop(g, rng_below(r, 25), 0, w);
            w[0] = (w[0] & ~0xFFu) | op;
            break;
        }
        }
        break;
    }
    return emit_fault(g, w, TSR_TICKET_NONE, TSR_TICKET_NONE, code, info);
}

static int build_vu(gen_t * g, unsigned idx) {
    const opts_t * o = g->o;
    rng_t * r = &g->r;
    // error kinds: VU faults, then the v2 endings
    int tail = -1;
    if (o->err_pct == 100) tail = (int)(idx % (VU_FAULTS + 4));
    else if (o->err_pct && rng_below(r, 100) < o->err_pct) tail = rng_below(r, 4) ? (int)rng_below(r, VU_FAULTS) : VU_FAULTS + (int)rng_below(r, 4);
    if (tail == VU_FAULTS + 2) {
        if (rng_below(r, 2) && qabort_terminal(g)) return 0;     // the whole trace
        qabort_recipe(g);
    }
    const uint32_t n_groups = 1 + rng_below(r, o->max_groups);
    for (uint32_t k = 0; k < n_groups; ++k) {
        next_group(g);
        uint32_t w[8];
        const uint32_t sel = rng_below(r, 4) ? (idx * 7u + k * 13u + rng_below(r, 3)) % VU_LEGAL_KINDS : rng_below(r, VU_LEGAL_KINDS);
        vu_legal(g, sel, 1, w);
    }
    flush_groups(g);
    if (tail < 0) return 0;
    uint32_t f;
    switch (tail - VU_FAULTS) {
    case 0:   // BAD_DEP
        f = fault_baddep(g);
        fault_consequences(g, f, rng_below(r, 3) == 0);
        break;
    case 1:   // DEP_FAILED chain from a VU fault
        f = vu_fault(g, rng_below(r, VU_FAULTS));
        fault_consequences(g, f, 1);
        break;
    case 2:   // QABORT recipe at the start; maybe a fault at the end
        if (rng_below(r, 2)) fault_consequences(g, vu_fault(g, rng_below(r, VU_FAULTS)), 0);
        break;
    case 3:   // fault, QCLEAR, recovery groups
        f = vu_fault(g, rng_below(r, VU_FAULTS));
        fault_consequences(g, f, rng_below(r, 2));
        emit_qctrl(g, TSR_QCTRL_QCLEAR);
        g->inj_pidx = NOFAIL;
        for (uint32_t k = 0, n = 1 + rng_below(r, 2); k < n; ++k) {
            next_group(g);
            uint32_t w[8];
            vu_legal(g, rng_below(r, VU_LEGAL_KINDS), 1, w);
        }
        flush_groups(g);
        break;
    default:  // one VU fault
        f = vu_fault(g, (uint32_t)tail);
        fault_consequences(g, f, rng_below(r, 3) == 0);
        break;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// AT programs (--ops at): KVST, ATTQK, ATTPV, ASMAX, AMASK, AMERGE, ANORM (AT §5) and the
// composite flash-attention chain of AT §6.6 / §7.1 (QACT -> KVST K and V, QACT q -> ATTQK MAXOUT
// -> ASMAX -> ATTPV LSUM -> AMERGE INIT (then merges) -> ANORM -> ST) across the DMA, QACT, MVM and
// VU queues. KV planes are written by the host (random q8_0 rows packed by tsr_kv_pack_row, the
// backend's set_tensor) and by KVST, partial meta beats included (read-modify-write over random
// plane bytes). Shapes: E = 64..2048, D = 64..512, G, GS, hg0 (several passes), T_q, CAUSAL / SLIDE
// windows, MAXOUT, LSUM, rstride / pstride / cstride gaps, n_groups across channels and LG, INIT
// and merging AMERGE. Faults: every row of AT §5.8 (K_RANGE, T_RANGE, OPERAND conditions with
// their ERR_INFO, SPM_RANGE / DRAM_RANGE per operand), reserved bits, KVST sub >= 2, ATTQK w7
// without MAXOUT, BAD_FMT, the AT-reserved opcodes.
// ---------------------------------------------------------------------------
enum { CA_KVST = 0, CA_ATTQK = 2, CA_ATTPV, CA_ASMAX, CA_AMASK, CA_AMERGE, CA_AMERGE_INIT, CA_ANORM, CA_FA,
       CA_RMW, CA_PASSES, CA_CAUSAL, CA_SLIDE, CA_MAXOUT, CA_LSUM, CA_FAULT, CA_END };
_Static_assert(CA_END <= COV_V2, "AT coverage slots overlap");
static const char * cova_names[CA_END] = { "KVST K", "KVST V", "ATTQK", "ATTPV", "ASMAX", "AMASK", "AMERGE merge",
                                           "AMERGE INIT", "ANORM", "flash-attention chains", "KVST RMW (partial meta)",
                                           "partial passes (GS < G)", "CAUSAL", "SLIDE", "MAXOUT", "LSUM", "AT faults" };

static void in_note(gen_t * g, uint32_t ch, uint64_t lo, uint64_t hi) {
    if (hi <= lo) return;
    if (g->n_in == g->cap_in) {
        g->cap_in = g->cap_in ? 2 * g->cap_in : 32;
        g->in = (inreg_t *)xrealloc(g->in, g->cap_in * sizeof(inreg_t));
    }
    g->in[g->n_in].ch = ch;
    g->in[g->n_in].lo = lo;
    g->in[g->n_in].hi = hi;
    g->n_in++;
}

static void op_pend(gen_t * g, const uint32_t w[8]) {
    tsr_insn i;
    memcpy(i.w, w, 32);
    pend(g, i);
}

// A KV plane of LG local groups at a fresh heap base (every channel), filled by the host with
// random q8_0 rows (or left zero). Returns B.
static uint32_t at_plane(gen_t * g, uint32_t kv, uint32_t E, uint32_t LG, int fill) {
    const uint32_t C = g->o->cfg.num_ch;
    const tsr_kv_geom geo = { E, C, LG };
    const uint64_t beats = tsr_kv_plane_beats(&geo);
    const uint64_t B = g->heap_next + rng_below(&g->r, 64);
    g->heap_next = B + beats + rng_below(&g->r, 64);
    if (fill) {
        uint8_t * plane[TSR_NUM_CH_MAX];
        for (uint32_t c = 0; c < C; ++c) plane[c] = tsr_sim_dram(g->sim, c) + B * 64;
        tsr_block_q8_0 * row = (tsr_block_q8_0 *)xmalloc(sizeof(tsr_block_q8_0) * (E / 32));
        const uint64_t cells = tsr_kv_plane_cells(&geo);
        for (uint64_t cell = 0; cell < cells; ++cell) {
            for (uint32_t b = 0; b < E / 32; ++b) {
                const uint16_t d = rand_fp16_scale(&g->r, g->o->special);
                memcpy(&row[b].d, &d, 2);
                rng_bytes(&g->r, row[b].qs, 32);
            }
            tsr_kv_pack_row(kv, &geo, plane, (uint32_t)cell, row);
        }
        free(row);
        for (uint32_t c = 0; c < C; ++c) in_note(g, c, B, B + beats);
    }
    return (uint32_t)B;
}

// T f32 rows of K elements -> LD -> QACT A32 into the QA vectors at dst + t·qs lines (qs 0: dense,
// one QACT; else one QACT per row). dst = 0xFFFFFFFF: fresh dense SPM. Returns the QA line.
static uint32_t at_qa_to(gen_t * g, uint32_t K, uint32_t T, uint32_t dst, uint32_t qs) {
    const uint32_t xl = K / 16, ql = tsr_qa_lines(TSR_QA_A32, K);
    float * x = (float *)xmalloc(sizeof(float) * K * T);
    for (uint32_t t = 0; t < T; ++t) gen_x(&g->r, x + (size_t)t * K, K, g->o->special);
    const uint32_t ch = rng_below(&g->r, g->o->cfg.num_ch);
    const uint64_t beat = take_io(g, ch, (uint64_t)T * xl);
    put_x(g, ch, beat, x, K, T, xl);
    free(x);
    const uint32_t sx = vu_take(g, (uint64_t)T * xl);
    if (dst == 0xFFFFFFFFu) dst = vu_take(g, (uint64_t)T * ql);
    pend(g, tsr_enc_ld(ch, sx, (uint32_t)beat, T * xl));
    if (qs == 0 || qs == ql) {                                   // QACT takes at most T_MAX tokens
        for (uint32_t t = 0; t < T; t += g->o->cfg.t_max)
            pend(g, tsr_enc_qact(TSR_QA_A32, dst + t * ql, sx + t * xl, K, umin32(T - t, g->o->cfg.t_max), 0));
    } else {
        for (uint32_t t = 0; t < T; ++t) pend(g, tsr_enc_qact(TSR_QA_A32, dst + t * qs, sx + t * xl, K, 1, 0));
    }
    return dst;
}
static uint32_t at_qa(gen_t * g, uint32_t K, uint32_t T) { return at_qa_to(g, K, T, 0xFFFFFFFFu, 0); }

// f32 rows for the vector ops: mode 0 scores (incl. -inf, specials), 1 weights in [0, 1] with
// zeros, 2 masks (0, -inf, finite, specials).
static uint32_t at_rows(gen_t * g, uint64_t lines, int mode) {
    const uint32_t line = vu_take(g, lines);
    if (lines == 0) return line;
    rng_t * r = &g->r;
    uint8_t * b = (uint8_t *)xmalloc((size_t)lines * 64);
    for (size_t i = 0; i < (size_t)lines * 16; ++i) {
        uint32_t v;
        const uint32_t k = rng_below(r, 16);
        if (mode == 1) v = k == 0 ? 0u : k == 1 ? vu_val(r) : f2u((float)rng_unit(r));
        else if (mode == 2) v = k < 6 ? 0u : k < 10 ? 0xFF800000u : k == 10 ? vu_val(r) : f2u((float)(rng_normal(r) * 2.0));
        else v = k == 0 ? 0xFF800000u : k == 1 ? vu_val(r) : f2u((float)(rng_normal(r) * 4.0));
        put_u32(b + 4 * i, v);
    }
    const uint32_t ch = rng_below(r, g->o->cfg.num_ch);
    const uint64_t beat = take_io(g, ch, lines);
    dram_put(g, ch, beat, b, (size_t)lines * 64);
    free(b);
    pend(g, tsr_enc_ld(ch, line, (uint32_t)beat, (uint32_t)lines));
    return line;
}

static uint32_t at_scale(rng_t * r) {
    static const float sc[] = { 0.125f, 0.08838835f, 0.0625f, 1.0f, 0.04419417f };
    return rng_below(r, 4) ? f2u(sc[rng_below(r, 5)]) : f2u((float)(0.01 + rng_unit(r)));
}

// Shape of an ATTQK / ATTPV pass (AT §5.2, §5.8) within the configuration's limits.
typedef struct {
    uint32_t E, D, dcode, G, GS, hg0, T_q, n_head, R, rows, max_row;
    uint32_t u0, n, LG;
} att_t;

static void att_shape(gen_t * g, att_t * a, int full_pass) {
    rng_t * r = &g->r;
    const tsr_config * c = &g->o->cfg;
    for (;;) {
        a->dcode = rng_below(r, 4);
        a->D = 64u << a->dcode;
        a->E = a->D * (1 + rng_below(r, a->D >= 256 ? 2 : 4));
        a->G = rng_below(r, 3) ? 1 + rng_below(r, 4) : 1 + rng_below(r, 16);
        if ((uint64_t)a->G * a->E > umin32(c->k_max, 8192)) continue;
        a->GS = full_pass ? a->G : 1 + rng_below(r, umin32(a->G, 8));
        if (a->GS > 8) continue;
        a->hg0 = full_pass ? 0 : rng_below(r, a->G - a->GS + 1);
        const uint32_t tq_max = c->t_max / a->GS;
        if (tq_max == 0) continue;
        a->T_q = 1 + rng_below(r, umin32(tq_max, 4));
        a->n_head = a->G * (a->E / a->D);
        a->R = a->T_q * a->n_head;
        a->rows = (a->E / a->D) * a->GS * a->T_q;
        if (a->rows > TSR_ATT_MAX_PASS_ROWS || a->R > 256) continue;
        a->max_row = (a->T_q - 1) * a->n_head + (a->E / a->D - 1) * a->G + a->hg0 + a->GS - 1;
        a->n = 1 + rng_below(r, 8);
        a->u0 = rng_below(r, 3) ? 0 : rng_below(r, 3 * c->num_ch);
        a->LG = (a->u0 + a->n - 1) / c->num_ch + 1 + rng_below(r, 2);
        return;
    }
}

static void att_w0(uint32_t w[8], uint32_t op, const att_t * a, uint32_t flags) {
    memset(w, 0, 32);
    w[0] = op | ((flags & 7u) << 12);
    TSR_INSN_SET(w, TSR_ATTQK_G_M1, a->G - 1);
    TSR_INSN_SET(w, TSR_ATTQK_HG0, a->hg0);
    TSR_INSN_SET(w, TSR_ATTQK_E64, a->E / 64);
    TSR_INSN_SET(w, TSR_ATTQK_DCODE, a->dcode);
    TSR_INSN_SET(w, TSR_ATTQK_TQ_M1, a->T_q - 1);
    TSR_INSN_SET(w, TSR_ATTQK_GS_M1, a->GS - 1);
    TSR_INSN_SET(w, TSR_ATTQK_U0, a->u0);
    TSR_INSN_SET(w, TSR_ATTQK_N_GROUPS, a->n);
}

// KVST of n_rows QA rows into a plane (kv). alloc = 0: random in-range encoding only.
static void at_kvst(gen_t * g, int alloc, uint32_t w[8], uint32_t kv, uint32_t E, uint32_t LG, uint32_t B) {
    rng_t * r = &g->r;
    const uint32_t C = g->o->cfg.num_ch, cells = 16 * C * LG;
    const uint32_t n_rows = 1 + rng_below(r, umin32(cells, rng_below(r, 3) ? 16 : 64));
    const uint32_t c0 = rng_below(r, cells - n_rows + 1);
    const uint32_t ql = tsr_qa_lines(TSR_QA_A32, E);
    const uint32_t ss = rng_below(r, 3) ? 0 : ql + rng_below(r, 3);
    uint32_t src;
    if (alloc) {
        if (ss == 0) {
            src = at_qa(g, E, n_rows);
        } else {                                 // QA vectors with gap lines between them
            src = vu_output(g, (uint64_t)(n_rows - 1) * ss + ql, 1);
            at_qa_to(g, E, n_rows, src, ss);
        }
        g->cov[CA_KVST + kv]++;
        g->cov[CA_RMW] += kv == TSR_KV_V ? (n_rows % 16 != 0 || c0 % 16 != 0) : (E % 1024 != 0 || c0 % 16 || n_rows % 16);
    } else {
        src = rng_below(r, g->spm_lines - ((n_rows - 1) * (ss ? ss : ql) + ql) + 1);
    }
    memset(w, 0, 32);
    w[0] = TSR_OP_KVST;
    TSR_INSN_SET(w, TSR_KVST_KV, kv);
    TSR_INSN_SET(w, TSR_KVST_N_ROWS_M1, n_rows - 1);
    TSR_INSN_SET(w, TSR_KVST_SRC, src);
    TSR_INSN_SET(w, TSR_KVST_BASE, B);
    TSR_INSN_SET(w, TSR_KVST_E64, E / 64);
    TSR_INSN_SET(w, TSR_KVST_LG, LG);
    TSR_INSN_SET(w, TSR_KVST_C0, c0);
    TSR_INSN_SET(w, TSR_KVST_SRC_STRIDE, ss);
    if (alloc) op_pend(g, w);
}

// ATTQK over a K plane (KVST'd or host-filled). Returns in *dst / *mdst / *rs the scores (rows at
// rs lines) and the row maxima when alloc.
static void at_attqk(gen_t * g, int alloc, uint32_t w[8], const att_t * a, uint32_t B, uint32_t flags,
                     uint32_t * pdst, uint32_t * pmdst, uint32_t * prs) {
    rng_t * r = &g->r;
    const uint32_t rs = rng_below(r, 3) ? 0 : a->n + rng_below(r, 3), rsp = rs ? rs : a->n;
    const uint32_t c_hi = rng_below(r, 3) ? 16 * a->n : rng_below(r, 16 * a->n + 1), c_lo = rng_below(r, 4) ? 0 : rng_below(r, c_hi + 1);
    const uint64_t ed = (uint64_t)a->max_row * rsp + a->n;
    const uint64_t ex = (uint64_t)a->T_q * tsr_qa_lines(TSR_QA_A32, a->n_head * a->D);
    const uint64_t em = (flags & TSR_ATT_MAXOUT) ? (a->max_row + 1 + 15) / 16 : 0;
    uint32_t dst, x, md = 0;
    if (alloc) {
        dst = vu_output(g, ed, 1);
        x = at_qa(g, a->n_head * a->D, a->T_q);
        if (flags & TSR_ATT_MAXOUT) md = vu_output(g, em, 1);
        g->cov[CA_ATTQK]++;
        g->cov[CA_PASSES] += a->GS < a->G;
        g->cov[CA_CAUSAL] += (flags & TSR_ATT_CAUSAL) != 0;
        g->cov[CA_SLIDE] += (flags & TSR_ATT_SLIDE) != 0;
        g->cov[CA_MAXOUT] += (flags & TSR_ATT_MAXOUT) != 0;
    } else {
        dst = rng_below(r, (uint32_t)(g->spm_lines - umin64(ed, g->spm_lines) + 1));
        x = rng_below(r, (uint32_t)(g->spm_lines - umin64(ex, g->spm_lines) + 1));
        if (flags & TSR_ATT_MAXOUT) md = rng_below(r, (uint32_t)(g->spm_lines - em + 1));
    }
    att_w0(w, TSR_OP_ATTQK, a, flags);
    TSR_INSN_SET(w, TSR_ATTQK_RSTRIDE, rs);
    w[1] = dst;
    w[2] = x;
    w[3] = B;
    TSR_INSN_SET(w, TSR_ATTQK_C_LO, c_lo);
    TSR_INSN_SET(w, TSR_ATTQK_C_HI, c_hi);
    w[7] = md;
    if (alloc) {
        op_pend(g, w);
        if (pdst) { *pdst = dst; *pmdst = md; *prs = rsp; }
        else {
            vu_store(g, dst, ed);
            if (em) vu_store(g, md, em);
        }
    }
}

// ATTPV; p = NULL: fresh weight rows. Returns the partials' line and cstride in *pd / *pcs.
static void at_attpv(gen_t * g, int alloc, uint32_t w[8], const att_t * a, uint32_t B, uint32_t flags,
                     const uint32_t * p_in, uint32_t ps_in, uint32_t * pd, uint32_t * pcs) {
    rng_t * r = &g->r;
    const uint32_t C = g->o->cfg.num_ch;
    const uint32_t ps = p_in ? (ps_in == a->n ? 0 : ps_in) : (rng_below(r, 3) ? 0 : a->n + rng_below(r, 3));
    const uint32_t psp = ps ? ps : a->n;
    const uint32_t rl = a->R * a->D / 16;
    const uint32_t cs = rng_below(r, 3) ? 0 : rl + rng_below(r, 3), csp = cs ? cs : rl;
    uint64_t ed = (uint64_t)(C - 1) * csp + rl;
    if ((flags & TSR_ATT_LSUM) && (uint64_t)C * csp + (uint64_t)C * ((a->R + 15) / 16) > ed) ed = (uint64_t)C * csp + (uint64_t)C * ((a->R + 15) / 16);
    const uint64_t ep = (uint64_t)a->max_row * psp + a->n;
    uint32_t dst, p;
    if (alloc) {
        dst = vu_output(g, ed, 1);
        p = p_in ? *p_in : at_rows(g, ep, 1);
        g->cov[CA_ATTPV]++;
        g->cov[CA_LSUM] += (flags & TSR_ATT_LSUM) != 0;
    } else {
        dst = rng_below(r, (uint32_t)(g->spm_lines - umin64(ed, g->spm_lines) + 1));
        p = rng_below(r, (uint32_t)(g->spm_lines - umin64(ep, g->spm_lines) + 1));
    }
    att_w0(w, TSR_OP_ATTPV, a, flags);
    TSR_INSN_SET(w, TSR_ATTPV_PSTRIDE, ps);
    w[1] = dst;
    w[2] = p;
    w[3] = B;
    TSR_INSN_SET(w, TSR_ATTPV_LG, a->LG);
    w[7] = cs;
    if (alloc) {
        op_pend(g, w);
        if (pd) { *pd = dst; *pcs = cs; }
        else vu_store(g, dst, ed);
    }
}

static void at_asmax(gen_t * g, int alloc, uint32_t w[8]) {
    rng_t * r = &g->r;
    const uint32_t nl = 1 + rng_below(r, rng_below(r, 4) ? 8 : 40), R = 1 + rng_below(r, rng_below(r, 4) ? 16 : 70);
    const uint32_t st = rng_below(r, 3) ? 0 : nl + rng_below(r, 3), stp = st ? st : nl;
    const uint64_t e = (uint64_t)(R - 1) * stp + nl, em = (R + 15) / 16;
    const int inpl = rng_below(r, 4) == 0;
    uint32_t d, s, m;
    if (alloc) {
        d = vu_output(g, e, inpl || stp > nl);
        s = inpl ? d : at_rows(g, e, 0);
        m = at_rows(g, em, 0);
        g->cov[CA_ASMAX]++;
    } else {
        d = rng_below(r, (uint32_t)(g->spm_lines - e + 1));
        s = rng_below(r, (uint32_t)(g->spm_lines - e + 1));
        m = rng_below(r, (uint32_t)(g->spm_lines - em + 1));
    }
    memset(w, 0, 32);
    w[0] = TSR_OP_ASMAX;
    w[1] = d; w[2] = s; w[3] = m;
    TSR_INSN_SET(w, TSR_ASMAX_N_LINES, nl);
    TSR_INSN_SET(w, TSR_ASMAX_STRIDE, st);
    w[5] = R;
    w[6] = at_scale(r);
    if (alloc) {
        op_pend(g, w);
        vu_store(g, d, e);
    }
}

static void at_amask(gen_t * g, int alloc, uint32_t w[8]) {
    rng_t * r = &g->r;
    const uint32_t nl = 1 + rng_below(r, 8), R = 1 + rng_below(r, 40), rpq = 1 + rng_below(r, R);
    const uint32_t st = rng_below(r, 3) ? 0 : nl + rng_below(r, 3), stp = st ? st : nl;
    const uint32_t ms = rng_below(r, 4) == 0 ? 0 : nl + rng_below(r, 3);
    const uint64_t e = (uint64_t)(R - 1) * stp + nl, emask = (uint64_t)((R + rpq - 1) / rpq - 1) * ms + nl, em = (R + 15) / 16;
    const int inpl = rng_below(r, 4) == 0;
    uint32_t d, s, mk, md;
    if (alloc) {
        d = vu_output(g, e, inpl || stp > nl);
        s = inpl ? d : at_rows(g, e, 0);
        mk = at_rows(g, emask, 2);
        md = vu_output(g, em, 1);
        g->cov[CA_AMASK]++;
    } else {
        d = rng_below(r, (uint32_t)(g->spm_lines - e + 1));
        s = rng_below(r, (uint32_t)(g->spm_lines - e + 1));
        mk = rng_below(r, (uint32_t)(g->spm_lines - emask + 1));
        md = rng_below(r, (uint32_t)(g->spm_lines - em + 1));
    }
    memset(w, 0, 32);
    w[0] = TSR_OP_AMASK;
    w[1] = d; w[2] = s; w[3] = mk;
    TSR_INSN_SET(w, TSR_AMASK_N_LINES, nl);
    TSR_INSN_SET(w, TSR_AMASK_STRIDE, st);
    TSR_INSN_SET(w, TSR_AMASK_R, R);
    TSR_INSN_SET(w, TSR_AMASK_RPQ, rpq);
    w[6] = ms;
    w[7] = md;
    if (alloc) {
        op_pend(g, w);
        vu_store(g, d, e);
        vu_store(g, md, em);
    }
}

// AMERGE / ANORM with random operands (alloc: loaded random data); the chain builds its own.
static void at_amerge(gen_t * g, int alloc, uint32_t w[8]) {
    rng_t * r = &g->r;
    const uint32_t C = g->o->cfg.num_ch;
    const uint32_t R = 1 + rng_below(r, 40), dcode = rng_below(r, 3), D = 64u << dcode, init = rng_below(r, 3) == 0;
    const uint32_t rl = R * D / 16, cs = rng_below(r, 3) ? 0 : rl + rng_below(r, 3), csp = cs ? cs : rl;
    const uint64_t ep = (uint64_t)C * csp + (uint64_t)C * ((R + 15) / 16), em = (R + 15) / 16;
    uint32_t orun, parts, mk, st;
    if (alloc) {
        orun = init ? vu_output(g, rl, 0) : at_rows(g, rl, 0);
        parts = at_rows(g, ep, 1);
        mk = at_rows(g, em, 0);
        st = init ? vu_output(g, 2 * em, 1) : at_rows(g, 2 * em, 1);
        g->cov[init ? CA_AMERGE_INIT : CA_AMERGE]++;
    } else {
        orun = rng_below(r, g->spm_lines - rl + 1);
        parts = rng_below(r, (uint32_t)(g->spm_lines - ep + 1));
        mk = rng_below(r, (uint32_t)(g->spm_lines - em + 1));
        st = rng_below(r, (uint32_t)(g->spm_lines - 2 * em + 1));
    }
    memset(w, 0, 32);
    w[0] = TSR_OP_AMERGE | (init << 12);
    w[1] = orun; w[2] = parts; w[3] = mk; w[4] = st;
    TSR_INSN_SET(w, TSR_AMERGE_R, R);
    TSR_INSN_SET(w, TSR_AMERGE_DCODE, dcode);
    w[6] = at_scale(r);
    w[7] = cs;
    if (alloc) {
        op_pend(g, w);
        vu_store(g, orun, rl);
        vu_store(g, st, 2 * em);
    }
}

static void at_anorm(gen_t * g, int alloc, uint32_t w[8]) {
    rng_t * r = &g->r;
    const uint32_t R = 1 + rng_below(r, 40), dcode = rng_below(r, 4), D = 64u << dcode, rl = R * D / 16, el = (R + 15) / 16;
    uint32_t out, orun, l;
    if (alloc) {
        out = vu_output(g, rl, 0);
        orun = at_rows(g, rl, 0);
        l = at_rows(g, el, 1);
        g->cov[CA_ANORM]++;
    } else {
        out = rng_below(r, g->spm_lines - rl + 1);
        orun = rng_below(r, g->spm_lines - rl + 1);
        l = rng_below(r, g->spm_lines - el + 1);
    }
    memset(w, 0, 32);
    w[0] = TSR_OP_ANORM;
    w[1] = out; w[2] = orun; w[3] = l;
    TSR_INSN_SET(w, TSR_ANORM_R, R);
    TSR_INSN_SET(w, TSR_ANORM_DCODE, dcode);
    if (alloc) {
        op_pend(g, w);
        vu_store(g, out, rl);
    }
}

// The flash-attention chain of one layer over a K / V plane pair (AT §6.6): KVST of the new
// cells, then per chunk of groups ATTQK MAXOUT -> ASMAX -> ATTPV LSUM -> AMERGE, then ANORM -> ST.
static void at_fa(gen_t * g) {
    rng_t * r = &g->r;
    const uint32_t C = g->o->cfg.num_ch;
    att_t a;
    att_shape(g, &a, 1);
    a.u0 = 0;
    const uint32_t chunks = 1 + rng_below(r, 3);
    const uint32_t ng = 1 + rng_below(r, 4);                   // groups per chunk
    a.LG = (chunks * ng + C - 1) / C + rng_below(r, 2);
    const uint32_t Bk = at_plane(g, TSR_KV_K, a.E, a.LG, 1), Bv = at_plane(g, TSR_KV_V, a.E, a.LG, 1);
    uint32_t w[8];
    for (uint32_t kv = 0; kv < 2; ++kv) {                       // the new cells (one KVST each)
        at_kvst(g, 1, w, kv, a.E, a.LG, kv ? Bv : Bk);
    }
    const uint32_t scale = at_scale(r), R = a.R, D = a.D, rl = R * D / 16, em = (R + 15) / 16;
    const uint32_t orun = vu_output(g, rl, 0), st = vu_output(g, 2 * em, 1);
    for (uint32_t k = 0; k < chunks; ++k) {
        a.u0 = k * ng;
        a.n = ng;
        uint32_t sc, md, rs, pd, cs;
        at_attqk(g, 1, w, &a, Bk, TSR_ATT_MAXOUT | (rng_below(r, 4) == 0 ? TSR_ATT_CAUSAL : 0), &sc, &md, &rs);
        // ASMAX in place over the scores (rows at rs lines, n lines of 16 cells each)
        uint32_t s[8] = { TSR_OP_ASMAX, sc, sc, md, 0, R, scale, 0 };
        TSR_INSN_SET(s, TSR_ASMAX_N_LINES, a.n);
        TSR_INSN_SET(s, TSR_ASMAX_STRIDE, rs == a.n ? 0 : rs);
        op_pend(g, s);
        at_attpv(g, 1, w, &a, Bv, TSR_ATT_LSUM, &sc, rs, &pd, &cs);
        uint32_t m[8] = { TSR_OP_AMERGE | ((k == 0 ? 1u : 0u) << 12), orun, pd, md, st, 0, scale, cs };
        TSR_INSN_SET(m, TSR_AMERGE_R, R);
        TSR_INSN_SET(m, TSR_AMERGE_DCODE, a.dcode);
        op_pend(g, m);
        g->cov[k == 0 ? CA_AMERGE_INIT : CA_AMERGE]++;
    }
    const uint32_t out = vu_output(g, rl, 0);
    uint32_t n[8] = { TSR_OP_ANORM, out, orun, st + em, 0, 0, 0, 0 };
    TSR_INSN_SET(n, TSR_ANORM_R, R);
    TSR_INSN_SET(n, TSR_ANORM_DCODE, a.dcode);
    op_pend(g, n);
    vu_store(g, out, rl);
    g->cov[CA_ANORM]++;
    g->cov[CA_FA]++;
}

static void at_group(gen_t * g, uint32_t sel) {
    rng_t * r = &g->r;
    uint32_t w[8];
    next_group(g);
    switch (sel % 8) {
    case 0: {   // KVST into a plane, then maybe read back by ATTQK / ATTPV (DMA -> MVM dependency)
        att_t a;
        att_shape(g, &a, rng_below(r, 2));
        const uint32_t kv = rng_below(r, 2), B = at_plane(g, kv, a.E, a.LG, rng_below(r, 3) != 0);
        at_kvst(g, 1, w, kv, a.E, a.LG, B);
        if (rng_below(r, 2)) {                   // else the plane's written beats are the result
            if (kv == TSR_KV_K) at_attqk(g, 1, w, &a, B, rng_below(r, 8), NULL, NULL, NULL);
            else at_attpv(g, 1, w, &a, B, rng_below(r, 2), NULL, 0, NULL, NULL);
        }
        break;
    }
    case 1: {
        att_t a;
        att_shape(g, &a, rng_below(r, 2));
        at_attqk(g, 1, w, &a, at_plane(g, TSR_KV_K, a.E, a.LG, 1), rng_below(r, 8), NULL, NULL, NULL);
        break;
    }
    case 2: {
        att_t a;
        att_shape(g, &a, rng_below(r, 2));
        at_attpv(g, 1, w, &a, at_plane(g, TSR_KV_V, a.E, a.LG, 1), rng_below(r, 2), NULL, 0, NULL, NULL);
        break;
    }
    case 3: at_asmax(g, 1, w); break;
    case 4: at_amask(g, 1, w); break;
    case 5: at_amerge(g, 1, w); break;
    case 6: at_anorm(g, 1, w); break;
    default: at_fa(g); break;
    }
}

static uint32_t past_spm32(gen_t * g, uint64_t e) {
    const uint64_t S = g->spm_lines;
    switch (rng_below(&g->r, 3)) {
    case 0:  return e == 0 ? (uint32_t)(S + 1 + rng_below(&g->r, 64)) : (uint32_t)(S - umin64(e, S) + 1 + rng_below(&g->r, (uint32_t)umin64(e, 16)));
    case 1:  return (uint32_t)(S + 1 + rng_below(&g->r, 1u << 20));
    default: return e == 0 ? 0xFFFFFFFFu - rng_below(&g->r, 16) : (uint32_t)(0x100000000ull - 1 - rng_below(&g->r, (uint32_t)umin64(e, 64)));
    }
}

#define AT_FAULTS 24
static uint32_t at_fault(gen_t * g, uint32_t k) {
    rng_t * r = &g->r;
    const tsr_config * c = &g->o->cfg;
    const uint32_t C = c->num_ch;
    uint32_t w[8];
    uint32_t code = TSR_ERR_ILLEGAL_OP;
    int64_t info = 0;
    att_t a;
    g->cov[CA_FAULT]++;
    switch (k % AT_FAULTS) {
    case 0: {   // KVST reserved bit / sub >= 2
        att_shape(g, &a, 1);
        at_kvst(g, 0, w, rng_below(r, 2), a.E, a.LG, (uint32_t)g->heap_next);
        if (rng_below(r, 2)) {
            static const uint32_t rsv[8] = { TSR_KVST_RSV_W0, TSR_KVST_RSV_W1, TSR_KVST_RSV_W2, TSR_KVST_RSV_W3,
                                             TSR_KVST_RSV_W4, TSR_KVST_RSV_W5, TSR_KVST_RSV_W6, TSR_KVST_RSV_W7 };
            set_rsv_bit(r, w, rsv);
        } else {
            TSR_INSN_SET(w, TSR_KVST_KV, 2 + rng_below(r, 14));
        }
        break;
    }
    case 1:     // KVST E/64 = 0 (K_RANGE)
        att_shape(g, &a, 1);
        at_kvst(g, 0, w, rng_below(r, 2), a.E, a.LG, (uint32_t)g->heap_next);
        TSR_INSN_SET(w, TSR_KVST_E64, 0);
        if (rng_below(r, 2)) TSR_INSN_SET(w, TSR_KVST_LG, 0);
        code = TSR_ERR_K_RANGE;
        break;
    case 2:     // KVST OPERAND: LG = 0 (0), c0 + n_rows > 16·C·LG (c0)
        att_shape(g, &a, 1);
        at_kvst(g, 0, w, rng_below(r, 2), a.E, a.LG, (uint32_t)g->heap_next);
        code = TSR_ERR_OPERAND;
        if (rng_below(r, 2)) {
            TSR_INSN_SET(w, TSR_KVST_LG, 0);
            info = 0;
        } else {
            const uint32_t n_rows = TSR_INSN_GET(w, TSR_KVST_N_ROWS_M1) + 1, cells = 16 * C * a.LG;
            const uint32_t c0 = rng_below(r, 2) ? cells - n_rows + 1 + rng_below(r, n_rows) : 0xFFFFFFFFu - rng_below(r, 64);
            TSR_INSN_SET(w, TSR_KVST_C0, c0);
            info = c0;
        }
        break;
    case 3: {   // KVST SPM_RANGE src / DRAM_RANGE plane
        att_shape(g, &a, 1);
        at_kvst(g, 0, w, rng_below(r, 2), a.E, a.LG, (uint32_t)g->heap_next);
        const uint32_t n_rows = TSR_INSN_GET(w, TSR_KVST_N_ROWS_M1) + 1, ql = tsr_qa_lines(TSR_QA_A32, a.E);
        const uint32_t ss = TSR_INSN_GET(w, TSR_KVST_SRC_STRIDE);
        const uint64_t es = (uint64_t)(n_rows - 1) * (ss ? ss : ql) + ql, ep = (uint64_t)a.LG * tsr_kv_group_beats(a.E);
        if (rng_below(r, 2)) {
            const uint32_t s = past_spm32(g, es);
            TSR_INSN_SET(w, TSR_KVST_SRC, s);
            code = TSR_ERR_SPM_RANGE;
            info = s;
            if (rng_below(r, 2)) TSR_INSN_SET(w, TSR_KVST_BASE, (uint32_t)g->lim);
        } else {
            const uint32_t b = (uint32_t)(g->lim - umin64(ep, g->lim) + 1 + rng_below(r, 16));
            TSR_INSN_SET(w, TSR_KVST_BASE, b);
            code = TSR_ERR_DRAM_RANGE;
            info = b;
        }
        break;
    }
    case 4: case 5: {   // ATTQK / ATTPV reserved bits (x16[15:12] included), ATTQK w7 without MAXOUT
        att_shape(g, &a, 0);
        const int qk = (k % AT_FAULTS) == 4;
        if (qk) at_attqk(g, 0, w, &a, (uint32_t)g->heap_next, rng_below(r, 4), NULL, NULL, NULL);
        else at_attpv(g, 0, w, &a, (uint32_t)g->heap_next, rng_below(r, 2), NULL, 0, NULL, NULL);
        if (qk && rng_below(r, 3) == 0) {
            w[7] = 1 + rng_below(r, 1000);                     // MAXOUT clear (flags < 4)
        } else if (rng_below(r, 2)) {
            w[0] |= 1u << (28 + rng_below(r, 4));
        } else {
            static const uint32_t rq[8] = { TSR_ATTQK_RSV_W0, 0, 0, 0, 0, 0, 0, 0 };
            static const uint32_t rp[8] = { TSR_ATTPV_RSV_W0, 0, 0, 0, 0, 0, TSR_ATTPV_RSV_W6, 0 };
            set_rsv_bit(r, w, qk ? rq : rp);
        }
        break;
    }
    case 6:     // BAD_FMT: KV format != 0 (with lower-priority violations)
        att_shape(g, &a, 0);
        if (rng_below(r, 2)) at_attqk(g, 0, w, &a, (uint32_t)g->heap_next, 4, NULL, NULL, NULL);
        else at_attpv(g, 0, w, &a, (uint32_t)g->heap_next, 1, NULL, 0, NULL, NULL);
        TSR_INSN_SET(w, TSR_ATTQK_FMT, 1 + rng_below(r, 15));
        if (rng_below(r, 2)) TSR_INSN_SET(w, TSR_ATTQK_E64, 0);
        code = TSR_ERR_BAD_FMT;
        break;
    case 7:     // K_RANGE: E/64 = 0 or G·E > K_MAX
        att_shape(g, &a, 0);
        if (rng_below(r, 2)) at_attqk(g, 0, w, &a, (uint32_t)g->heap_next, 4, NULL, NULL, NULL);
        else at_attpv(g, 0, w, &a, (uint32_t)g->heap_next, 1, NULL, 0, NULL, NULL);
        if (rng_below(r, 2)) {
            TSR_INSN_SET(w, TSR_ATTQK_E64, 0);
        } else {
            TSR_INSN_SET(w, TSR_ATTQK_E64, 511);
            TSR_INSN_SET(w, TSR_ATTQK_G_M1, (c->k_max / (511 * 64)) + rng_below(r, 4));
            TSR_INSN_SET(w, TSR_ATTQK_HG0, 0);
            TSR_INSN_SET(w, TSR_ATTQK_GS_M1, 0);
        }
        code = TSR_ERR_K_RANGE;
        break;
    case 8:     // T_RANGE: T_q·GS > T_MAX
        if (c->t_max >= 64) { k = 9; return at_fault(g, k); }
        att_shape(g, &a, 0);
        if (rng_below(r, 2)) at_attqk(g, 0, w, &a, (uint32_t)g->heap_next, 4, NULL, NULL, NULL);
        else at_attpv(g, 0, w, &a, (uint32_t)g->heap_next, 1, NULL, 0, NULL, NULL);
        {
            uint32_t tq, gs;
            do { tq = 1 + rng_below(r, 8); gs = 1 + rng_below(r, 8); } while (tq * gs <= c->t_max);
            TSR_INSN_SET(w, TSR_ATTQK_TQ_M1, tq - 1);
            TSR_INSN_SET(w, TSR_ATTQK_GS_M1, gs - 1);
            TSR_INSN_SET(w, TSR_ATTQK_G_M1, gs - 1 + rng_below(r, 8));   // G·E = 64·G stays within K_MAX
            TSR_INSN_SET(w, TSR_ATTQK_HG0, 0);
            TSR_INSN_SET(w, TSR_ATTQK_E64, 1);
            TSR_INSN_SET(w, TSR_ATTQK_DCODE, 0);
        }
        code = TSR_ERR_T_RANGE;
        break;
    case 9: case 10: case 11: case 12: case 13: case 14: {   // ATTQK / ATTPV OPERAND conditions
        const uint32_t v = (k % AT_FAULTS) - 9;
        const int qk = v == 3 || v == 5 ? 1 : v == 4 ? 0 : (int)rng_below(r, 2);
        att_shape(g, &a, 0);
        if (qk) at_attqk(g, 0, w, &a, (uint32_t)g->heap_next, 4, NULL, NULL, NULL);
        else at_attpv(g, 0, w, &a, (uint32_t)g->heap_next, 1, NULL, 0, NULL, NULL);
        code = TSR_ERR_OPERAND;
        switch (v) {
        case 0: {   // E % D != 0 (ERR_INFO E)
            const uint32_t D = 128u << rng_below(r, 3), E = D / 2 + D * rng_below(r, 3);
            TSR_INSN_SET(w, TSR_ATTQK_DCODE, D == 128 ? 1 : D == 256 ? 2 : 3);
            TSR_INSN_SET(w, TSR_ATTQK_E64, E / 64);
            TSR_INSN_SET(w, TSR_ATTQK_G_M1, 0);
            TSR_INSN_SET(w, TSR_ATTQK_HG0, 0);
            TSR_INSN_SET(w, TSR_ATTQK_GS_M1, 0);
            TSR_INSN_SET(w, TSR_ATTQK_TQ_M1, 0);
            info = E;
            break;
        }
        case 1: {   // hg0 + GS > G (ERR_INFO hg0 + GS)
            const uint32_t G = TSR_INSN_GET(w, TSR_ATTQK_G_M1) + 1, GS = TSR_INSN_GET(w, TSR_ATTQK_GS_M1) + 1;
            const uint32_t hg0 = G - GS + 1 + rng_below(r, 64 - (G - GS + 1));
            TSR_INSN_SET(w, TSR_ATTQK_HG0, hg0);
            info = hg0 + GS;
            break;
        }
        case 2: {   // n_groups 0 or > 4095
            const uint32_t n = rng_below(r, 2) ? 0 : 4096 + rng_below(r, 60000);
            TSR_INSN_SET(w, TSR_ATTQK_N_GROUPS, n);
            info = n;
            break;
        }
        case 3: {   // ATTQK: c_lo > c_hi or c_hi > 16·n_groups (ERR_INFO c_hi)
            const uint32_t n = TSR_INSN_GET(w, TSR_ATTQK_N_GROUPS);
            uint32_t lo, hi;
            if (rng_below(r, 2)) { hi = rng_below(r, 16 * n); lo = hi + 1 + rng_below(r, 16); }
            else { hi = 16 * n + 1 + rng_below(r, 100); lo = 0; }
            TSR_INSN_SET(w, TSR_ATTQK_C_LO, lo);
            TSR_INSN_SET(w, TSR_ATTQK_C_HI, hi);
            info = hi;
            break;
        }
        case 4: {   // ATTPV: l_max >= LG (ERR_INFO u0 + n - 1), or PCAP (ERR_INFO n)
            const uint32_t u0 = TSR_INSN_GET(w, TSR_ATTPV_U0), n = TSR_INSN_GET(w, TSR_ATTPV_N_GROUPS);
            if (rng_below(r, 2)) {
                TSR_INSN_SET(w, TSR_ATTPV_LG, rng_below(r, (u0 + n - 1) / C + 1));
                info = u0 + n - 1;
            } else {
                const uint32_t pcap = tsr_qa_lines(TSR_QA_A256, c->k_max) * 16;
                const uint32_t E = TSR_INSN_GET(w, TSR_ATTQK_E64) * 64, D = 64u << TSR_INSN_GET(w, TSR_ATTQK_DCODE);
                const uint32_t per = (E / D) * 16;                 // per group-column of a channel
                uint32_t nn = C * (pcap / per + 1);
                if (nn > 4095) {                                    // not reachable: fall back to l_max
                    TSR_INSN_SET(w, TSR_ATTPV_LG, 0);
                    info = u0 + n - 1;
                    break;
                }
                TSR_INSN_SET(w, TSR_ATTPV_N_GROUPS, nn);
                TSR_INSN_SET(w, TSR_ATTPV_U0, 0);
                TSR_INSN_SET(w, TSR_ATTPV_LG, 0xFFFF);
                info = nn;
            }
            break;
        }
        default: {  // pass rows > 128 (ATTQK), else 0 < rstride / pstride < n_groups
            if (qk && rng_below(r, 2)) {
                TSR_INSN_SET(w, TSR_ATTQK_DCODE, 0);
                TSR_INSN_SET(w, TSR_ATTQK_E64, 8);              // E/D = 8
                TSR_INSN_SET(w, TSR_ATTQK_G_M1, 63);
                TSR_INSN_SET(w, TSR_ATTQK_HG0, 0);
                const uint32_t gs = umin32(8, c->t_max);
                TSR_INSN_SET(w, TSR_ATTQK_GS_M1, gs - 1);
                TSR_INSN_SET(w, TSR_ATTQK_TQ_M1, c->t_max / gs - 1);
                if (8u * gs * (c->t_max / gs) <= 128) {          // not reachable at this T_MAX
                    TSR_INSN_SET(w, TSR_ATTQK_N_GROUPS, 0);
                    info = 0;
                    break;
                }
                info = 8u * gs * (c->t_max / gs);
                if ((uint64_t)64 * 512 > c->k_max) {             // G·E = 32768 within K_MAX only on large configs
                    TSR_INSN_SET(w, TSR_ATTQK_G_M1, c->k_max / 512 - 1);
                    const uint32_t G = c->k_max / 512;
                    if (G < gs) { TSR_INSN_SET(w, TSR_ATTQK_GS_M1, G - 1); }
                    const uint32_t gs2 = TSR_INSN_GET(w, TSR_ATTQK_GS_M1) + 1;
                    info = 8u * gs2 * (TSR_INSN_GET(w, TSR_ATTQK_TQ_M1) + 1);
                    if (info <= 128) { TSR_INSN_SET(w, TSR_ATTQK_N_GROUPS, 0); info = 0; }
                }
            } else {
                uint32_t n = TSR_INSN_GET(w, TSR_ATTQK_N_GROUPS);
                if (n < 2) { n = 2 + rng_below(r, 6); TSR_INSN_SET(w, TSR_ATTQK_N_GROUPS, n); TSR_INSN_SET(w, TSR_ATTQK_C_HI, 0);
                             TSR_INSN_SET(w, TSR_ATTQK_C_LO, 0); TSR_INSN_SET(w, TSR_ATTPV_LG, 0xFFFF); }
                const uint32_t s = 1 + rng_below(r, n - 1);
                TSR_INSN_SET(w, TSR_ATTQK_RSTRIDE, s);
                info = s;
            }
            break;
        }
        }
        break;
    }
    case 15: case 16: {  // ATTQK SPM_RANGE (dst, x, mdst) / DRAM_RANGE (plane)
        att_shape(g, &a, 0);
        const uint32_t flags = TSR_ATT_MAXOUT | rng_below(r, 4);
        at_attqk(g, 0, w, &a, (uint32_t)g->heap_next, flags, NULL, NULL, NULL);
        const uint32_t rs = TSR_INSN_GET(w, TSR_ATTQK_RSTRIDE), n = a.n;
        const uint64_t ed = (uint64_t)a.max_row * (rs ? rs : n) + n, ex = (uint64_t)a.T_q * tsr_qa_lines(TSR_QA_A32, a.n_head * a.D);
        const uint64_t em = (a.max_row + 16) / 16, ep = (uint64_t)((a.u0 + n - 1) / C + 1) * tsr_kv_group_beats(a.E);
        if ((k % AT_FAULTS) == 15) {
            const uint32_t opnd = rng_below(r, 3);
            const uint32_t bad = past_spm32(g, opnd == 0 ? ed : opnd == 1 ? ex : em);
            w[opnd == 2 ? 7 : 1 + opnd] = bad;
            if (rng_below(r, 2)) w[3] = (uint32_t)g->lim;
            code = TSR_ERR_SPM_RANGE;
            info = bad;
        } else {
            const uint32_t b = rng_below(r, 2) ? (uint32_t)(g->lim - umin64(ep, g->lim) + 1 + rng_below(r, 16)) : 0xFFFFFFFFu - rng_below(r, 16);
            w[3] = b;
            code = TSR_ERR_DRAM_RANGE;
            info = b;
        }
        break;
    }
    case 17: {  // ATTPV SPM_RANGE (dst, p) / DRAM_RANGE (plane)
        att_shape(g, &a, 0);
        at_attpv(g, 0, w, &a, (uint32_t)g->heap_next, rng_below(r, 2), NULL, 0, NULL, NULL);
        const uint32_t E64 = a.E / 64, LG = a.LG, lmax = (a.u0 + a.n - 1) / C;
        const uint64_t ev = ((uint64_t)(E64 - 1) * LG + lmax + 1) * 17;
        const uint32_t v = rng_below(r, 3);
        if (v < 2) {
            const uint32_t bad = past_spm32(g, 1);
            w[1 + v] = bad;
            code = TSR_ERR_SPM_RANGE;
            info = bad;
        } else {
            const uint32_t b = (uint32_t)(g->lim - umin64(ev, g->lim) + 1 + rng_below(r, 16));
            w[3] = b;
            code = TSR_ERR_DRAM_RANGE;
            info = b;
        }
        break;
    }
    case 18: case 19: {   // ASMAX / AMASK: reserved bits, OPERAND n_lines / R / scale / rpq, SPM_RANGE
        const int sm = (k % AT_FAULTS) == 18;
        if (sm) at_asmax(g, 0, w); else at_amask(g, 0, w);
        const uint32_t v = rng_below(r, sm ? 5 : 5);
        if (v == 0) {
            if (sm) { static const uint32_t rs[8] = { TSR_ASMAX_RSV_W0, 0, 0, 0, 0, 0, 0, TSR_ASMAX_RSV_W7 }; set_rsv_bit(r, w, rs); }
            else { static const uint32_t rs[8] = { TSR_AMASK_RSV_W0, TSR_AMASK_RSV_W1, TSR_AMASK_RSV_W2, TSR_AMASK_RSV_W3,
                                                   TSR_AMASK_RSV_W4, TSR_AMASK_RSV_W5, TSR_AMASK_RSV_W6, TSR_AMASK_RSV_W7 };
                   uint32_t any = 0; for (int i = 0; i < 8; ++i) any |= rs[i];
                   if (any) set_rsv_bit(r, w, rs); else w[0] |= 1u << (8 + rng_below(r, 24)); }
        } else if (v == 1) {
            const uint32_t nl = rng_below(r, 2) ? 0 : 4096 + rng_below(r, 60000);
            TSR_INSN_SET(w, TSR_ASMAX_N_LINES, nl);
            code = TSR_ERR_OPERAND;
            info = nl;
        } else if (v == 2) {
            const uint32_t R = rng_below(r, 2) ? 0 : 1025 + rng_below(r, sm ? 100000 : 60000);
            if (sm) w[5] = R; else TSR_INSN_SET(w, TSR_AMASK_R, R);
            code = TSR_ERR_OPERAND;
            info = R;
        } else if (v == 3) {
            code = TSR_ERR_OPERAND;
            if (sm) {
                static const uint32_t bad[] = { 0, 0x80000000u, 0xBF800000u, 0x7F800000u, 0x7FC00000u, 0x00000001u, 0xFF800000u };
                w[6] = bad[rng_below(r, 7)];
                info = w[6];
            } else {
                TSR_INSN_SET(w, TSR_AMASK_RPQ, 0);
                info = 0;
            }
        } else {
            const uint32_t opnd = rng_below(r, sm ? 3 : 4);
            const uint32_t bad = past_spm32(g, 1);
            w[(!sm && opnd == 3) ? 7 : 1 + opnd] = bad;
            code = TSR_ERR_SPM_RANGE;
            info = bad;
        }
        break;
    }
    case 20: case 21: {   // AMERGE / ANORM: reserved bits, OPERAND R / scale, SPM_RANGE
        const int mg = (k % AT_FAULTS) == 20;
        if (mg) at_amerge(g, 0, w); else at_anorm(g, 0, w);
        const uint32_t v = rng_below(r, 4);
        if (v == 0) {
            static const uint32_t rm[8] = { TSR_AMERGE_RSV_W0, TSR_AMERGE_RSV_W1, TSR_AMERGE_RSV_W2, TSR_AMERGE_RSV_W3,
                                            TSR_AMERGE_RSV_W4, TSR_AMERGE_RSV_W5, TSR_AMERGE_RSV_W6, TSR_AMERGE_RSV_W7 };
            static const uint32_t rn[8] = { TSR_ANORM_RSV_W0, TSR_ANORM_RSV_W1, TSR_ANORM_RSV_W2, TSR_ANORM_RSV_W3,
                                            TSR_ANORM_RSV_W4, TSR_ANORM_RSV_W5, TSR_ANORM_RSV_W6, TSR_ANORM_RSV_W7 };
            set_rsv_bit(r, w, mg ? rm : rn);
        } else if (v == 1) {
            const uint32_t R = rng_below(r, 2) ? 0 : 1025 + rng_below(r, 60000);
            if (mg) TSR_INSN_SET(w, TSR_AMERGE_R, R); else TSR_INSN_SET(w, TSR_ANORM_R, R);
            code = TSR_ERR_OPERAND;
            info = R;
        } else if (v == 2 && mg) {
            static const uint32_t bad[] = { 0, 0x80000000u, 0xC0000000u, 0x7F800000u, 0xFFC00000u, 0x00400000u };
            w[6] = bad[rng_below(r, 6)];
            code = TSR_ERR_OPERAND;
            info = w[6];
        } else {
            const uint32_t opnd = rng_below(r, mg ? 4 : 3);
            const uint32_t bad = past_spm32(g, 1);
            w[1 + opnd] = bad;
            code = TSR_ERR_SPM_RANGE;
            info = bad;
        }
        break;
    }
    default: {  // the AT-reserved opcodes (DMA, quantize and MVM classes): ILLEGAL_OP on SYNC
        static const uint8_t ops[] = { 0x15, 0x16, 0x17, 0x19, 0x1A, 0x1B, 0x23 };
        at_asmax(g, 0, w);
        w[0] = ops[rng_below(r, sizeof(ops))];
        break;
    }
    }
    return emit_fault(g, w, TSR_TICKET_NONE, TSR_TICKET_NONE, code, info);
}

static int build_at(gen_t * g, unsigned idx) {
    const opts_t * o = g->o;
    rng_t * r = &g->r;
    int tail = -1;
    if (o->err_pct == 100) tail = (int)(idx % (AT_FAULTS + 4));
    else if (o->err_pct && rng_below(r, 100) < o->err_pct) tail = rng_below(r, 4) ? (int)rng_below(r, AT_FAULTS) : AT_FAULTS + (int)rng_below(r, 4);
    if (tail == AT_FAULTS + 2) {
        if (rng_below(r, 2) && qabort_terminal(g)) return 0;     // the whole trace
        qabort_recipe(g);
    }
    // KV planes and weights-like data from the heap (WEIGHT region)
    g->heap_next = TSR_REGION_WEIGHT_OFF / 64 + rng_below(r, 4096);
    const uint32_t n_groups = 1 + rng_below(r, o->max_groups);
    for (uint32_t k = 0; k < n_groups; ++k) at_group(g, rng_below(r, 4) ? idx * 3u + k : rng_below(r, 8));
    flush_groups(g);
    if (tail < 0) return 0;
    uint32_t f;
    switch (tail - AT_FAULTS) {
    case 0:
        f = fault_baddep(g);
        fault_consequences(g, f, rng_below(r, 3) == 0);
        break;
    case 1:
        f = at_fault(g, rng_below(r, AT_FAULTS));
        fault_consequences(g, f, 1);
        break;
    case 2:
        if (rng_below(r, 2)) fault_consequences(g, at_fault(g, rng_below(r, AT_FAULTS)), 0);
        break;
    case 3:
        f = at_fault(g, rng_below(r, AT_FAULTS));
        fault_consequences(g, f, rng_below(r, 2));
        emit_qctrl(g, TSR_QCTRL_QCLEAR);
        g->inj_pidx = NOFAIL;
        at_group(g, rng_below(r, 8));
        flush_groups(g);
        break;
    default:
        f = at_fault(g, (uint32_t)tail);
        fault_consequences(g, f, rng_below(r, 3) == 0);
        break;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// ctrl programs (--ops ctrl): the issue path (CC §7-§9). Each trace is a few segments:
//   burst    long runs of independent ops on one unit (queue and issue-FIFO back-pressure: pushes
//            wait for credits, CC §7.3)
//   dag      a data-flow graph over every unit: values (SPM f32 rows) made by LD, copied / added /
//            scaled by VOP, quantized by QACT and multiplied by MVM, stored by ST; fan-out, fan-in
//            (an op with deps on two other units), in-place updates (WAR / WAW hazards), NOP joins
//            of random good tickets, FENCE barriers, trace marks, REC pushes
//   hold     QHOLD set ... QHOLD clear around ops that fill the queues to their depth (never more:
//            no push waits on a held queue), with deps inside the window
//   errors   (with --errors) a fault followed by a DEP_FAILED chain through every unit, ABORTED
//            entries after it, BAD_DEP variants, the QABORT recipe, a QCLEAR recovery
// ---------------------------------------------------------------------------
enum { CC_BURST = 0, CC_DAG, CC_HOLD, CC_FANIN, CC_INPLACE, CC_JOIN, CC_FENCE, CC_CHAIN_ALL, CC_END };
_Static_assert(CC_END <= COV_V2, "ctrl coverage slots overlap");
static const char * covc_names[CC_END] = { "bursts", "dag segments", "hold windows (filled)", "fan-in ops",
                                           "in-place updates", "NOP join trees", "FENCE barriers",
                                           "DEP_FAILED chains through every unit" };

typedef struct { uint32_t line, lines; } val_t;

typedef struct {
    val_t    v[64];
    uint32_t n;
} vals_t;

static void vals_add(vals_t * s, uint32_t line, uint32_t lines) {
    if (s->n < 64) { s->v[s->n].line = line; s->v[s->n].lines = lines; s->n++; }
    else { s->v[s->n - 1].line = line; s->v[s->n - 1].lines = lines; }
}

static int has_vu(const gen_t * g) { return (g->o->cfg.caps & TSR_CAPS_VU) && unit_built(g, TSR_UNIT_VU); }

static void vop2(gen_t * g, uint32_t func, uint32_t d, uint32_t a, uint32_t b, uint32_t lines) {
    tsr_insn i = {{0}};
    i.w[0] = TSR_OP_VOP;
    TSR_INSN_SET(i.w, TSR_VOP_FUNC, func);
    TSR_INSN_SET(i.w, TSR_VOP_D, d);
    TSR_INSN_SET(i.w, TSR_VOP_A, a);
    if (tsr_vop_func_reads(func) & TSR_VOP_READS_B) TSR_INSN_SET(i.w, TSR_VOP_B, b);
    TSR_INSN_SET(i.w, TSR_VOP_M, 16 * lines);
    TSR_INSN_SET(i.w, TSR_VOP_ROWS, 1);
    emit_insn(g, i);
}

// One dag step on the value set (emitted at once: the hazard tracker gives the deps).
// The ctrl segments reuse the SPM: near its end the allocator starts again at the bottom and the
// values are forgotten (new ops overwrite old lines: WAR / WAW hazards with their readers).
static void ctrl_spm_wrap(gen_t * g, vals_t * s) {
    if (g->spm_next + 1024 < g->spm_lines) return;
    g->spm_next = rng_below(&g->r, 64);
    if (s) s->n = 0;
}

static int room(const gen_t * g, uint32_t u) { return !g->hold || g->hold_room[u] > 0; }

static void ctrl_step(gen_t * g, vals_t * s) {
    rng_t * r = &g->r;
    const uint32_t C = g->o->cfg.num_ch;
    ctrl_spm_wrap(g, s);
    uint32_t k = s->n == 0 ? 0 : rng_below(r, 10);
    if (g->hold) {      // in a hold window: only ops whose queue has room (CC §7.3: no waiting push),
                        // and a SYNC slot left for a FENCE that folds their deps
        if (g->hold_room[TSR_UNIT_SYNC] < 2) return;
        static const uint8_t unit_of_step[10] = { TSR_UNIT_DMA, TSR_UNIT_DMA, TSR_UNIT_DMA, TSR_UNIT_MVM, TSR_UNIT_MVM,
                                                  TSR_UNIT_VU, TSR_UNIT_VU, TSR_UNIT_VU, TSR_UNIT_VU, TSR_UNIT_SYNC };
        if (k == 3 || k == 4) {
            if (!room(g, TSR_UNIT_QACT)) return;
        }
        if (!room(g, unit_of_step[k])) return;
    }
    const val_t a = s->n ? s->v[rng_below(r, s->n)] : (val_t){ 0, 0 };
    switch (k) {
    case 0:
    case 1: {   // a new value (DMA)
        const uint32_t lines = 2 * (1 + rng_below(r, 4));
        uint8_t * b = (uint8_t *)xmalloc((size_t)lines * 64);
        for (uint32_t i = 0; i < lines * 16; ++i) put_u32(b + 4 * i, vu_val(r));
        const uint32_t ch = rng_below(r, C);
        const uint64_t beat = take_io(g, ch, lines);
        dram_put(g, ch, beat, b, (size_t)lines * 64);
        free(b);
        const uint32_t line = vu_take(g, lines);
        emit_insn(g, tsr_enc_ld(ch, line, (uint32_t)beat, lines));
        vals_add(s, line, lines);
        break;
    }
    case 2: {   // store (DMA)
        const uint32_t ch = rng_below(r, C);
        emit_insn(g, tsr_enc_st(ch, a.line, (uint32_t)take_io(g, ch, a.lines), a.lines));
        break;
    }
    case 3:
    case 4: {   // QACT -> MVM (QACT, MVM): y = W x with fresh Q8_0 weights
        if (a.lines % 2) break;
        const uint32_t K = 16 * a.lines, N = 16 * (1 + rng_below(r, 4)), ql = tsr_qa_lines(TSR_QA_A32, K);
        const uint32_t qa = vu_take(g, ql), y = vu_take(g, N / 16);
        const size_t rbytes = (size_t)(K / 32) * tsr_fmt_ggml_block_bytes(TSR_FMT_Q8_0);
        uint8_t * blocks = (uint8_t *)xmalloc(rbytes * N);
        for (uint32_t n = 0; n < N; ++n) gen_row_blocks(r, TSR_FMT_Q8_0, K, blocks + rbytes * n, g->o->special);
        const uint64_t woff = g->heap_next + rng_below(r, 8);
        g->heap_next = woff + (uint64_t)tsr_ceil_div_u32(N, C) * tsr_row_beats(TSR_FMT_Q8_0, K) + rng_below(r, 8);
        put_tensor(g, TSR_FMT_Q8_0, N, K, woff, blocks);
        free(blocks);
        emit_insn(g, tsr_enc_qact(TSR_QA_A32, qa, a.line, K, 1, 0));
        emit_insn(g, tsr_enc_mvm(TSR_FMT_Q8_0, 0, y, qa, (uint32_t)woff, N, K, 1, 0));
        vals_add(s, y, N / 16);
        break;
    }
    case 5:
    case 6: {   // VOP copy / scale (VU)
        if (!has_vu(g)) break;
        const uint32_t d = vu_take(g, a.lines);
        vop2(g, rng_below(r, 2) ? TSR_VOP_FUNC_COPY : TSR_VOP_FUNC_EXP, d, a.line, 0, a.lines);
        vals_add(s, d, a.lines);
        break;
    }
    case 7:
    case 8: {   // VOP add of two values (fan-in: deps on up to two other units)
        if (!has_vu(g)) break;
        const val_t b = s->v[rng_below(r, s->n)];
        const uint32_t lines = umin32(a.lines, b.lines);
        const int inpl = rng_below(r, 3) == 0;            // in place over a: WAR / WAW with a's readers
        const uint32_t d = inpl ? a.line : vu_take(g, lines);
        vop2(g, rng_below(r, 2) ? TSR_VOP_FUNC_ADD : TSR_VOP_FUNC_MAX, d, a.line, b.line, lines);
        if (!inpl) vals_add(s, d, lines);
        g->cov[CC_FANIN]++;
        g->cov[CC_INPLACE] += inpl;
        break;
    }
    default: {  // NOP join of good tickets, or a FENCE, or a trace mark
        const uint32_t v = rng_below(r, 3);
        if (v == 0) {
            emit_sync(g, TSR_OP_NOP, random_good_ticket(g), random_good_ticket(g));
            g->cov[CC_JOIN]++;
        } else if (v == 1) {
            emit_sync(g, TSR_OP_FENCE, TSR_TICKET_NONE, rng_below(r, 2) ? random_good_ticket(g) : TSR_TICKET_NONE);
            g->cov[CC_FENCE]++;
        } else {
            emit_mark(g, rng_u32(r));
        }
        break;
    }
    }
}

// A run of n independent ops on one unit (fresh regions): queue and FIFO back-pressure.
static void ctrl_burst(gen_t * g, uint32_t u, uint32_t n) {
    rng_t * r = &g->r;
    const uint32_t C = g->o->cfg.num_ch;
    uint32_t qa = 0;
    if (u == TSR_UNIT_MVM) {     // one QA vector (LD + QACT) that the MVMs of the burst read
        if (g->hold && (g->hold_room[TSR_UNIT_DMA] == 0 || g->hold_room[TSR_UNIT_QACT] == 0)) return;
        ctrl_spm_wrap(g, NULL);
        uint8_t b[256];
        for (int k = 0; k < 64; ++k) put_u32(b + 4 * k, vu_val(r));
        const uint32_t ch = rng_below(r, C);
        const uint64_t beat = take_io(g, ch, 4);
        dram_put(g, ch, beat, b, sizeof(b));
        const uint32_t x = vu_take(g, 4);
        qa = vu_take(g, tsr_qa_lines(TSR_QA_A32, 64));
        emit_insn(g, tsr_enc_ld(ch, x, (uint32_t)beat, 4));
        emit_insn(g, tsr_enc_qact(TSR_QA_A32, qa, x, 64, 1, 0));
    }
    for (uint32_t i = 0; i < n; ++i) {
        const int dma = u == TSR_UNIT_DMA || (u == TSR_UNIT_VU && !has_vu(g));   // the op's queue
        if (!room(g, dma ? TSR_UNIT_DMA : u) || (u == TSR_UNIT_QACT && !room(g, TSR_UNIT_DMA))) return;
        ctrl_spm_wrap(g, NULL);
        const uint32_t lines = 1 + rng_below(r, 4);
        if (dma) {
            uint8_t * b = (uint8_t *)xmalloc((size_t)lines * 64);
            rng_bytes(r, b, (size_t)lines * 64);
            const uint32_t ch = rng_below(r, C);
            const uint64_t beat = take_io(g, ch, lines);
            dram_put(g, ch, beat, b, (size_t)lines * 64);
            free(b);
            emit_insn(g, tsr_enc_ld(ch, vu_take(g, lines), (uint32_t)beat, lines));
        } else if (u == TSR_UNIT_SYNC) {
            emit_sync(g, rng_below(r, 4) ? TSR_OP_NOP : TSR_OP_FENCE, TSR_TICKET_NONE, TSR_TICKET_NONE);
        } else if (u == TSR_UNIT_VU) {
            tsr_insn op = {{0}};
            op.w[0] = TSR_OP_SOP;
            TSR_INSN_SET(op.w, TSR_SOP_SUB, TSR_SOP_SUB_MOVI);
            TSR_INSN_SET(op.w, TSR_SOP_SD, rng_below(r, 16));
            op.w[2] = vu_val(r);
            emit_insn(g, op);
        } else if (u == TSR_UNIT_QACT || u == TSR_UNIT_MVM) {
            // QACT / MVM of zero-width data would read unwritten SPM: an N = 0 MVM or a QACT over
            // a freshly loaded row (the LD is a DMA op, so this burst also interleaves units)
            if (u == TSR_UNIT_MVM) {    // N = 0 (writes nothing) or N = 16 rows of fresh weights
                const uint32_t N = rng_below(r, 3) ? 0 : 16;
                uint64_t woff = g->heap_next;
                if (N) {
                    const size_t rbytes = 2 * tsr_fmt_ggml_block_bytes(TSR_FMT_Q8_0);
                    uint8_t * blocks = (uint8_t *)xmalloc(rbytes * N);
                    for (uint32_t k = 0; k < N; ++k) gen_row_blocks(r, TSR_FMT_Q8_0, 64, blocks + rbytes * k, g->o->special);
                    woff = g->heap_next + rng_below(r, 8);
                    g->heap_next = woff + (uint64_t)tsr_ceil_div_u32(N, C) * tsr_row_beats(TSR_FMT_Q8_0, 64) + 1;
                    put_tensor(g, TSR_FMT_Q8_0, N, 64, woff, blocks);
                    free(blocks);
                }
                emit_insn(g, tsr_enc_mvm(TSR_FMT_Q8_0, 0, vu_take(g, N / 16), qa, (uint32_t)woff, N, 64, 1, 0));
            } else {
                uint8_t b[128];
                for (int k = 0; k < 32; ++k) put_u32(b + 4 * k, vu_val(r));
                const uint32_t ch = rng_below(r, C);
                const uint64_t beat = take_io(g, ch, 2);
                dram_put(g, ch, beat, b, 128);
                const uint32_t x = vu_take(g, 2);
                if (!g->hold || g->hold_room[TSR_UNIT_DMA] > 0) {
                    emit_insn(g, tsr_enc_ld(ch, x, (uint32_t)beat, 2));
                    emit_insn(g, tsr_enc_qact(TSR_QA_A32, vu_take(g, tsr_qa_lines(TSR_QA_A32, 32)), x, 32, 1, 0));
                }
            }
        }
    }
    g->cov[CC_BURST]++;
}

// A DEP_FAILED chain that visits every other unit once (CC §9.2), from the failed ticket t.
static void chain_all(gen_t * g, uint32_t t) {
    uint32_t order[TSR_NUM_UNITS], n = 0;
    for (uint32_t v = 0; v < TSR_NUM_UNITS; ++v)
        if (v != TSR_UNIT_SYNC && unit_built(g, v) && g->fail_seq[v] == NOFAIL) order[n++] = v;
    for (uint32_t i = n; i > 1; --i) {
        const uint32_t j = rng_below(&g->r, i);
        const uint32_t x = order[i - 1];
        order[i - 1] = order[j];
        order[j] = x;
    }
    uint32_t prev = t;
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t w[8];
        if (!some_op(g, order[i], w)) continue;
        prev = g->rec[emit_op(g, w, prev, TSR_TICKET_NONE, EMIT_NODEPS)].ticket;
    }
    if (g->fail_seq[TSR_UNIT_SYNC] == NOFAIL) {
        const uint32_t nop[8] = { TSR_OP_NOP, 0, 0, 0, 0, 0, 0, 0 };
        emit_op(g, nop, prev, TSR_TICKET_NONE, EMIT_NODEPS);
    }
    g->cov[CC_CHAIN_ALL]++;
    g->cov[COV_V2 + V2_DEPFAIL]++;
}

static uint32_t ctrl_fault(gen_t * g) {
    rng_t * r = &g->r;
    switch (rng_below(r, 4)) {
    case 0:  return fault_m2(g, (int)rng_below(r, T2_FAULTS));
    case 1:  return fault_m3(g, (int)rng_below(r, T3_N), -1);
    case 2:  return has_vu(g) ? vu_fault(g, rng_below(r, VU_FAULTS)) : fault_m1(g, rng_below(r, 8));
    default: return fault_baddep(g);
    }
}

static int build_ctrl(gen_t * g, unsigned idx) {
    const opts_t * o = g->o;
    rng_t * r = &g->r;
    vals_t s = { .n = 0 };
    int tail = -1;
    if (o->err_pct == 100) tail = (int)(idx % 6);
    else if (o->err_pct && rng_below(r, 100) < o->err_pct) tail = (int)rng_below(r, 6);
    // opening: the QABORT recipe, or a hold window filling the queues to their depth
    if (tail == 2) {
        if (((idx / 6) & 1) && qabort_terminal(g)) return 0;     // the whole trace
        qabort_recipe(g);
    } else if (idx % 3 == 0 || rng_below(r, 4) == 0) {
        emit_qctrl(g, TSR_QCTRL_QHOLD_SET);
        for (uint32_t u = 0; u < TSR_NUM_UNITS; ++u)
            if (unit_built(g, u) && rng_below(r, 3)) ctrl_burst(g, u, g->hold_room[u] / (1 + rng_below(r, 2)));
        for (uint32_t k = 0, n = rng_below(r, 4); k < n && g->hold_room[TSR_UNIT_DMA] > 1 && g->hold_room[TSR_UNIT_VU] > 0; ++k)
            ctrl_step(g, &s);
        emit_qctrl(g, TSR_QCTRL_QHOLD_CLR);
        g->cov[CC_HOLD]++;
    }
    const uint32_t segs = 1 + rng_below(r, o->max_groups);
    for (uint32_t k = 0; k < segs; ++k) {
        const uint32_t kind = (idx + k) % 3;
        if (kind == 0) {
            uint32_t u;
            do u = rng_below(r, TSR_NUM_UNITS); while (!unit_built(g, u));
            ctrl_burst(g, u, 8 + rng_below(r, 3 * g->o->depth[u] + 8));
        } else {
            for (uint32_t i = 0, n = 6 + rng_below(r, 24); i < n; ++i) ctrl_step(g, &s);
            g->cov[CC_DAG]++;
        }
        if (rng_below(r, 3) == 0) {
            emit_sync(g, TSR_OP_FENCE, TSR_TICKET_NONE, TSR_TICKET_NONE);
            g->cov[CC_FENCE]++;
        }
        if (rng_below(r, 4) == 0) emit_mark(g, (uint32_t)(idx << 16 | k));
    }
    uint32_t f;
    switch (tail) {
    case 0:     // fault, chain through every unit, ABORTED entries after the fault
        f = ctrl_fault(g);
        chain_all(g, g->rec[f].ticket);
        after_error(g, TK_U(g->rec[f].ticket));
        break;
    case 1:     // BAD_DEP, then more work that runs (the poisoned entry blocks nothing)
        fault_baddep(g);
        for (uint32_t i = 0, n = rng_below(r, 6); i < n; ++i) ctrl_step(g, &s);
        break;
    case 3:     // fault, consequences, QCLEAR recovery, more work
    case 4:
        f = ctrl_fault(g);
        fault_consequences(g, f, rng_below(r, 2));
        emit_qctrl(g, TSR_QCTRL_QCLEAR);
        g->inj_pidx = NOFAIL;
        s.n = 0;                                 // nothing after the QCLEAR depends on what came before
        for (uint32_t i = 0, n = 4 + rng_below(r, 12); i < n; ++i) ctrl_step(g, &s);
        if (tail == 4) {
            f = ctrl_fault(g);
            fault_consequences(g, f, 1);
        }
        break;
    case 5:     // two faults on different units: the first error is the smaller push index
        f = ctrl_fault(g);
        fault_consequences(g, f, 0);
        f = ctrl_fault(g);
        fault_consequences(g, f, rng_below(r, 2));
        break;
    default:
        break;
    }
    return 0;
}

static void print_cov_mode(const opts_t * o, const unsigned long * cov) {
    if (o->ops == OPS_VU) {
        static const char * const red[14] = { "SUM", "SUM TO_S", "SUM RCP", "SUM RCP TO_S", "SUM RMS", "SUM RMS TO_S",
                                              "SUMSQ", "SUMSQ TO_S", "SUMSQ RCP", "SUMSQ RCP TO_S", "SUMSQ RMS",
                                              "SUMSQ RMS TO_S", "MAX", "MAX TO_S" };
        printf(" VOP functions");
        for (int f = 0; f < 25; ++f) printf(" %02x:%lu", f, cov[CV_VOPF + f]);
        printf(", RED");
        for (int k = 0; k < 14; ++k) printf("%s %s %lu", k ? "," : "", red[k], cov[CV_REDK + k]);
        printf(", ROPE NORM/NEOX x rec/FF/FS/FF+FS/ATAB");
        for (int k = 0; k < 10; ++k) printf(" %lu", cov[CV_ROPE + k]);
        printf(", TOPK %lu, SOP subs", cov[CV_TOPK]);
        for (int k = 0; k < 15; ++k) printf(" %d:%lu", k, cov[CV_SOP + k]);
        printf(", DQROW Q8_0 %lu Q4_K %lu, n = 0 ops %lu, tails %lu, broadcast %lu, dst gaps %lu, in place %lu, faults %lu",
               cov[CV_DQROW], cov[CV_DQROW + 1], cov[CV_N0], cov[CV_TAIL], cov[CV_BCAST], cov[CV_GAP], cov[CV_INPL], cov[CV_FAULT]);
    } else if (o->ops == OPS_AT) {
        for (int k = 0; k < CA_END; ++k) printf("%s %s %lu", k ? "," : "", cova_names[k], cov[k]);
    } else {
        for (int k = 0; k < CC_END; ++k) printf("%s %s %lu", k ? "," : "", covc_names[k], cov[k]);
    }
}


// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
static int mkdir_p(const char * path) {
    char * p = strdup(path);
    if (!p) return -1;
    const size_t n = strlen(p);
    for (size_t i = 1; i <= n; ++i) {
        if (p[i] == '/' || p[i] == '\0') {
            const char ch = p[i];
            p[i] = '\0';
            if (mkdir(p, 0755) != 0 && errno != EEXIST) {
                fprintf(stderr, "tsr-trace-gen: mkdir %s: %s\n", p, strerror(errno));
                free(p);
                return -1;
            }
            p[i] = ch;
        }
    }
    free(p);
    return 0;
}

static void usage(void) {
    fprintf(stderr,
            "usage: tsr-trace-gen -o DIR [-n COUNT] [-s SEED] [--prefix NAME] [--ops gen|m2|m3|vu|at|ctrl]\n"
            "                     [--t-max T] [--num-ch C] [--spm-kib S] [--ch-gib G] [--addr-beat-w W]\n"
            "                     [--k-max K] [--caps CAPS] [--q-depth D0,D1,D2,D3,D4,D5] [--max-k K] [--max-n N]\n"
            "                     [--max-groups G] [--errors PCT] [--no-special] [--max-wbeats B] [--big-pct PCT]\n"
            "                     [--check-sched N] [--no-verify] [-q]\n"
            "defaults: -n 20 -s 1 --prefix gen --ops gen, U200 config (T_MAX 1, 4 channels, 4 MiB SPM,\n"
            "          16 GiB/ch, ADDR_BEAT_W 28, U200 M6 queue depths 8,4,4,16,0,4), --max-k 4096 --max-n 256\n"
            "          --max-groups 3 --errors 0; --caps: a number, none (0), m6 (VU|ATT), m8 (m6|TOPK|DQROW),\n"
            "          all (everything the simulator implements); default none for gen / m2 / m3, m8 for vu,\n"
            "          m6 for at / ctrl\n"
            "--ops gen (alias m1): LD -> QACT -> MVM -> ST groups; m2: QACT groups and LD -> ST copies (no MVM);\n"
            "          m3: MVM-centred groups (--max-wbeats B, default 16384, bounds the MVM weight beats per trace\n"
            "          on the busiest channel; --big-pct PCT gives every (100/PCT)-th trace a 4x budget);\n"
            "          vu / at: every VU / AT op, legal and illegal; ctrl: the issue path (credits, holds,\n"
            "          FENCE, QABORT / QCLEAR, deps across all units, BAD_DEP / DEP_FAILED)\n"
            "--errors PCT: PCT%% of the traces get an error injection (faulting op, BAD_DEP, DEP_FAILED chain,\n"
            "          QABORT or QCLEAR recovery); --check-sched N: replay each trace under N random schedules\n");
}

static int parse_u(const char * s, unsigned long lo, unsigned long hi, unsigned long * out) {
    char * end;
    errno = 0;
    const unsigned long v = strtoul(s, &end, 0);
    if (errno || *end || end == s || v < lo || v > hi) return -1;
    *out = v;
    return 0;
}

#define CAPS_M6 (TSR_CAPS_VU | TSR_CAPS_ATT)
#define CAPS_M8 (CAPS_M6 | TSR_CAPS_TOPK | TSR_CAPS_DQROW)

int main(int argc, char ** argv) {
    opts_t o;
    memset(&o, 0, sizeof(o));
    const tsr_config u200 = TSR_CONFIG_U200_V1;
    o.cfg = u200;
    const uint8_t depth_u200[TSR_NUM_UNITS] = { TSR_QDEPTH_U200_DMA, TSR_QDEPTH_U200_QACT, TSR_QDEPTH_U200_MVM,
                                                TSR_QDEPTH_U200_VU,  TSR_QDEPTH_U200_ATT,  TSR_QDEPTH_U200_SYNC };
    memcpy(o.depth, depth_u200, sizeof(o.depth));
    o.max_k = 4096;
    o.max_n = 256;
    o.max_groups = 3;
    o.max_wbeats = 16384;
    o.special = 1;
    o.verify = 1;
    const char * out = NULL;
    const char * prefix = "gen";
    long caps = -1;
    unsigned long count = 20, seed = 1, v;
    for (int i = 1; i < argc; ++i) {
        const char * a = argv[i];
        const char * val = (i + 1 < argc) ? argv[i + 1] : NULL;
#define OPT(name) (strcmp(a, name) == 0 && val && ++i)
        if (OPT("-o")) out = val;
        else if (OPT("-n")) { if (parse_u(val, 1, 1000000, &count)) goto bad; }
        else if (OPT("-s")) { if (parse_u(val, 0, ~0ul, &seed)) goto bad; }
        else if (OPT("--prefix")) prefix = val;
        else if (OPT("--t-max")) { if (parse_u(val, 1, 8, &v)) goto bad; o.cfg.t_max = (uint32_t)v; }
        else if (OPT("--num-ch")) { if (parse_u(val, 1, TSR_NUM_CH_MAX, &v)) goto bad; o.cfg.num_ch = (uint32_t)v; }
        else if (OPT("--spm-kib")) { if (parse_u(val, 64, 65536, &v)) goto bad; o.cfg.spm_bytes = (uint32_t)v * 1024; }
        else if (OPT("--ch-gib")) { if (parse_u(val, 1, 16, &v)) goto bad; o.cfg.ch_bytes = (uint64_t)v << 30; }
        else if (OPT("--addr-beat-w")) { if (parse_u(val, 1, 40, &v)) goto bad; o.cfg.addr_beat_w = (uint32_t)v; }
        else if (OPT("--ops")) {
            int k;
            for (k = 0; k < 6; ++k) if (strcmp(val, ops_names[k]) == 0) break;
            if (k == 6 && strcmp(val, "m1") == 0) k = OPS_M1;
            if (k == 6) goto bad;
            o.ops = k;
        }
        else if (OPT("--caps")) {
            if (strcmp(val, "none") == 0) caps = 0;
            else if (strcmp(val, "m6") == 0) caps = CAPS_M6;
            else if (strcmp(val, "m8") == 0) caps = CAPS_M8;
            else if (strcmp(val, "all") == 0) caps = (long)TSR_SIM_CAPS_IMPLEMENTED;
            else if (parse_u(val, 0, 0xFFFFFFFFul, &v) == 0) caps = (long)v;
            else goto bad;
        }
        else if (OPT("--q-depth")) {
            const char * p = val;
            for (int u = 0; u < (int)TSR_NUM_UNITS; ++u) {
                char * end;
                errno = 0;
                const unsigned long d = strtoul(p, &end, 10);
                if (errno || end == p || d > 31 || (u == (int)TSR_UNIT_SYNC && d == 0)) goto bad;
                o.depth[u] = (uint8_t)d;
                if (u + 1 < (int)TSR_NUM_UNITS) { if (*end != ',') goto bad; p = end + 1; }
                else if (*end) goto bad;
            }
        }
        else if (OPT("--k-max")) { if (parse_u(val, 256, 65280, &v) || v % 256) goto bad; o.cfg.k_max = (uint32_t)v; }
        else if (OPT("--max-k")) { if (parse_u(val, 32, 65280, &v)) goto bad; o.max_k = (uint32_t)v; }
        else if (OPT("--max-n")) { if (parse_u(val, 1, 1u << 20, &v)) goto bad; o.max_n = (uint32_t)v; }
        else if (OPT("--max-groups")) { if (parse_u(val, 1, 64, &v)) goto bad; o.max_groups = (uint32_t)v; }
        else if (OPT("--errors")) { if (parse_u(val, 0, 100, &v)) goto bad; o.err_pct = (unsigned)v; }
        else if (OPT("--max-wbeats")) { if (parse_u(val, 1, 1u << 24, &v)) goto bad; o.max_wbeats = (uint32_t)v; }
        else if (OPT("--big-pct")) { if (parse_u(val, 0, 100, &v)) goto bad; o.big_pct = (unsigned)v; }
        else if (OPT("--check-sched")) { if (parse_u(val, 0, 64, &v)) goto bad; o.check_sched = (unsigned)v; }
        else if (strcmp(a, "--no-special") == 0) o.special = 0;
        else if (strcmp(a, "--no-verify") == 0) o.verify = 0;
        else if (strcmp(a, "-q") == 0) o.quiet = 1;
        else if (strcmp(a, "-h") == 0 || strcmp(a, "--help") == 0) { usage(); return 0; }
        else goto bad;
#undef OPT
    }
    if (!out) goto bad;
    o.m2 = o.ops == OPS_M2;
    o.m3 = o.ops == OPS_M3;
    if (caps < 0) caps = o.ops == OPS_VU ? CAPS_M8 : (o.ops == OPS_AT || o.ops == OPS_CTRL) ? CAPS_M6 : 0;
    o.cfg.caps = (uint32_t)caps & TSR_SIM_CAPS_IMPLEMENTED;
    if ((uint32_t)caps != o.cfg.caps) {
        fprintf(stderr, "tsr-trace-gen: --caps 0x%lx asks for more than the simulator implements (0x%x)\n", caps,
                TSR_SIM_CAPS_IMPLEMENTED);
        return 2;
    }
    if ((o.ops == OPS_VU && !(o.cfg.caps & TSR_CAPS_VU)) || (o.ops == OPS_AT && (o.cfg.caps & CAPS_M6) != CAPS_M6)) {
        fprintf(stderr, "tsr-trace-gen: --ops %s needs CAPS %s\n", ops_names[o.ops], o.ops == OPS_VU ? "VU" : "VU and ATT");
        return 2;
    }
    if (mkdir_p(out)) return 2;
    {
        unsigned long cov[COV_MAX] = {0};
        unsigned long fails = 0;
        for (unsigned long i = 0; i < count; ++i) {
            const size_t n = strlen(out) + strlen(prefix) + 64;
            char * dir = (char *)xmalloc(n);
            snprintf(dir, n, "%s/%s_s%lu_%04lu", out, prefix, seed, i);
            if (gen_trace(&o, (uint64_t)seed, (unsigned)i, dir, cov) != 0) {
                fprintf(stderr, "tsr-trace-gen: FAILED %s\n", dir);
                ++fails;
            }
            free(dir);
        }
        printf("tsr-trace-gen: %lu traces in %s (%lu failed), --ops %s; coverage:", count - fails, out, fails, ops_names[o.ops]);
        if (o.m3) {
            for (int i = 0; i < C3_N; ++i) printf("%s %s %lu", i ? "," : "", cov3_names[i], cov[C2_N + i]);
            printf("; M2 groups and QACTs:");
            for (int i = 0; i < C2_N; ++i) printf("%s %s %lu", i ? "," : "", cov2_names[i], cov[i]);
        } else if (o.m2) {
            for (int i = 0; i < C2_N; ++i) printf("%s %s %lu", i ? "," : "", cov2_names[i], cov[i]);
        } else if (o.ops == OPS_M1) {
            for (int i = 0; i < COV_N; ++i) printf("%s %s %lu", i ? "," : "", cov_names[i], cov[i]);
        } else {
            print_cov_mode(&o, cov);
        }
        printf("; stream:");
        for (int i = 0; i < V2_N; ++i) printf("%s %s %lu", i ? "," : "", cov2v_names[i], cov[COV_V2 + i]);
        printf("\ntsr-trace-gen: first error (DONE ERR_CODE) per trace:");
        for (int e = 0; e < 16; ++e) if (cov[COV_ERRC + e]) printf(" %d:%lu", e, cov[COV_ERRC + e]);
        printf("; entry completion codes:");
        for (int e = 0; e < 16; ++e) if (cov[COV_ENTC + e]) printf(" %d:%lu", e, cov[COV_ENTC + e]);
        printf("; %llu records, init %.1f KiB, expect %.1f KiB\n", set_stats.records, (double)set_stats.init_bytes / 1024.0,
               (double)set_stats.expect_bytes / 1024.0);
        if (o.m3)
            printf("tsr-trace-gen: %llu MVMs; MVM weight beats: %llu on the busiest channel (sum over traces), "
                   "%llu on all channels, max %llu per trace, %llu traces above %llu\n", m3_stats.mvms, m3_stats.ch0,
                   m3_stats.all, m3_stats.max, m3_stats.over, M3_BIG_BEATS);
        return fails ? 1 : 0;
    }
bad:
    usage();
    return 2;
}
