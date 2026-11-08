// tsr_race.c — the simulator's race detector and the launch-stateless check (P1.9;
// docs/design/06-software-verification.md "SW" §4.6, SW-3 / SW-4). Public API: tsr_race.h.
// It is an optional model (tsr_sim_int.h "Optional models"): tsr_race_attach registers a queue
// listener and an access listener when tsr_sim_opt_race() != 0, and keeps its state in the
// TSR_MODEL_RACE slot. Nothing here changes the simulation: the detector only observes.
//
// Model (memory model docs/ctrl-riscv.md §5.5 rules 2 and 4, restated by SW §4.6; EX §2.3):
//   Agents: units 0-5, the RISC-V (6), the host (7). Each agent has a vector clock over the eight;
//   an op's clock is computed at its completion (effects apply atomically then, SW §4.1), with
//   epoch = its unit's previous epoch + 1. The RISC-V and the host advance their own epoch at every
//   release (a push, a QCLEAR, an LCQ post; an LQ_PUSH, a CTRL write that starts the hart).
//   Happens-before edges, exactly SW §4.6's list:
//   E1 queue order: an op joins the clock of its unit's previous completion;
//   E2 dependencies: an op joins the clocks of its deps as dispatched (a BAD_DEP dep is none);
//   E3 RISC-V program order: every RISC-V access carries the RISC-V's current clock;
//   E4 push after a completed store: an op joins the RISC-V clock at its ISSUE_PUSH store (CC §7.1:
//      no SPM-window access is outstanding at a push, so every earlier access has completed;
//      loads included, which also orders a RISC-V read before a later op's write);
//   E5 the RISC-V observing completions: a DONE_SEQ[u] read returning v (control window, or Q_DONE[u]
//      through the core window) joins the clock of u's completion v - 1, which by E1 covers every
//      completion of u below v; the WAIT interrupt joins the clock of WAIT_TICKET's ticket (rule
//      below); a QSTATUS read with QIDLE = 1 (or a core-window STATUS read with QIDLE) joins every
//      unit's latest completion — QIDLE = every pushed entry is done, an observation of all of
//      them (CC §5.1, §10.5 counts QSTATUS among the observations);
//   E6 FENCE and QCLEAR: such an entry joins, per unit v, the clock of the last entry of v pushed
//      before it (its dispatch snapshot, CC §8.3); entries that depend on it, or follow it on SYNC,
//      inherit that by E1 / E2;
//   E7 H1 (EX §2.3): the host clock at the accepted LQ_PUSH of launch s is joined into the RISC-V
//      clock at the LQ_POP of s, hence (E4) into every op the RISC-V pushes while s, or a later
//      launch, runs;
//   E8 H2 (EX §2.3): a host read of an LCQ record slot (0x300 + 32 k) joins the RISC-V clock at
//      that record's post (the record is a message: EX §4.4, ctrl-riscv rule 5), and, for the
//      answer record of launch s, the clocks of every op of s completed by then — an op of s still
//      running when the host reads its answer (a runtime that did not join, EX §4.7 R6) stays
//      unordered, so its later accesses race with the host's;
//   E9 the hart starts: a host CTRL write that raises RV_RUN joins the host clock into the RISC-V's
//      (the host loaded the image and data before it). Subsumed today by the harness rule below
//      (every host action while RV_RUN = 0 already merges the two clocks), so no test can tell it
//      apart; kept so that the edge does not depend on that rule.
//   The host's own observations, by symmetry with E5: a Q_DONE[u] read and a STATUS read with
//   QIDLE = 1 (EX §5.3-§5.4).
//   WAIT interrupt (E5): irq_fast[0] is pending while WAIT_TICKET's ticket is done (CC §12.3); the
//   hart can see it at any instant (taking the interrupt, waking from or falling through WFI,
//   reading mip). The model has no hook on those instants, so at every event of the hart the
//   armed WAIT_TICKET is treated as observed once its ticket is done (in the core's counts; the
//   random schedule's RISC-V-side lag of a few cycles is ignored). A program that arms
//   WAIT_TICKET and then neither waits, nor enables, nor reads mip gets the edge anyway: the one
//   case where the detector may miss a race (documented, not otherwise exempted).
//   Test hooks (tsr_sim_push / _qctrl / _cw_*: pc TSR_SIM_PC_HOOK) and anything while CTRL.RV_RUN = 0 are the
//   test harness: one thread acting for the RISC-V and the host, so their clocks are merged at each
//   such action, and when no ticketed entry is outstanding (the harness ran the queues to idle:
//   tsr_sim_run_until_idle, QIDLE) every unit's latest completion is joined as well (E5's QIDLE
//   rule). The hart's accesses (a real pc, 0 included, while RV_RUN = 1) never get that.
//   Access sets: an op's tagged extents from its ENQ event (tsr_extents: SPM lines, s-register
//   indices; a read-modify-write operand is both), applied when it ran (DONE with ran = 1, code 0);
//   its DRAM set is the beats its exec reported through the capture hooks (tsr_sim_cap_read /
//   _write, which every handler's exec calls for each DRAM range it touches, tsr_sim_int.h §2),
//   clipped to its DRAM extent — exact where the checker's extent is a range-check span (AT
//   §5.8: a KVST's extent is its whole plane but it writes only the new rows, without reading
//   metadata; ATTQK / ATTPV extents start at the plane base; an MVM's weight extent is
//   the largest channel's), and the DRAM extent itself for an exec that reports nothing; the
//   RISC-V's SPM window bytes (a line-buffer fill reads its whole line), control-window and
//   core-window registers that carry the edges above, and the s registers through the core
//   window (SREG0-15); the host's DRAM through tsr_sim_dram_read / _write / _note_write and the s
//   registers at SREG0-15. Not visible: DRAM or SPM touched through the tsr_sim_dram() /
//   tsr_sim_spm() pointers without a note, and tsr_sim_op (outside the queues).
//   Race (rule 4): same SPM byte (ops: whole lines), DRAM beat or s register, at least one write,
//   two agents, neither access ordered before the other. Same-agent accesses are always ordered
//   (queue order, program order).
//   Debug-module system-bus accesses (SBA, DB §3.6) are not race-checked (the debugger reads and
//   writes SPM at quiescent points by its own protocol); its SPM writes count as writes for
//   SPM_STALE.
// Shadow state (FastTrack-style, exact for the vector clocks above): per location the last write
// {agent, epoch, ticket or pc, time} and the latest read per agent; a write clears the reads. SPM: one cell
// per line (ops access whole lines; the RISC-V is the only byte-granular agent and never races with
// itself, so a line cell is exact for races), DRAM: per channel a sorted interval map of beat
// ranges whose cells merge again when equal, s registers: 16 cells. Per unit a history of
// completion clocks indexed by sequence (HIST_MAX entries; a dep on a ticket older than that
// joins the oldest kept clock instead, which by E1 is later: conservative, counted in
// tsr_race_stats.old_refs).
// SPM_STALE (SW-4), "popped" defined: at every accepted LQ_POP the SPM write generation
// advances, and every SPM byte counts as unwritten for that launch. A write (op extent line, RISC-V
// store byte, SBA store byte) marks the bytes written in the current generation. While the launch
// runs — from its LQ_POP until its answer post or any RISC-V reset (host CTRL, SOFT_RESET, DB's
// ndmreset: the core's ANSWERED := TAKEN, checked when it matters) — an op pushed during
// it that reads a line with no byte written since the pop, a RISC-V exact-window read of a byte not
// written since the pop, or a line-buffer fill of a line with no byte written since the pop, is
// reported. Nothing is exempt: an argument block staged by the runtime is a write of the launch;
// state a kernel keeps in SPM across launches (EX §2.3 H4 allows it) is reported. Before the first
// pop and between launches nothing is checked, nor during a launch popped by a test hook (no
// runtime: the harness fixtures SPM through the tsr_sim_spm() pointer, which is invisible).
// Reports: once per (class, unordered pair of sites), class = write-write or read-write (RAW and
// WAR between the same two sites are one report); SPM_STALE once per reading site. A site is an
// op's (unit, opcode, push pc, ra at the push: the call site of a push helper), or (unit, push
// index) for a test-hook push; the RISC-V's pc; the host's direction. A report keeps its first
// occurrence (both sides, the conflicting range, extended over the contiguous lines / beats of
// that access) and counts the accesses that hit it again. Mode fatal: the first report is printed
// and the process aborts after the event that found it.

#define _GNU_SOURCE

#include "tsr_race.h"
#include "tsr_local_tile.h"
#include "tsr_sim_int.h"

#if defined(TSR_SIM_HAVE_RV) && TSR_SIM_HAVE_RV
#include "tsr_bus.h"     // the built-in hart (ra at a push)
#include "tsr_rv.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NA        TSR_RACE_AGENTS
#define A_RV      TSR_RACE_AGENT_RV
#define A_HOST    TSR_RACE_AGENT_HOST
#define SEQ_MASK  0x0FFFFFFFu
#define SEQ_HALF  0x08000000u
#define NO_LAUNCH 0xFFFFFFFFu
#define INFL      64u               // in flight per unit: Q_DEPTH (<= 31) + issue FIFO (32)
#define PQ        64u               // pushes between the issue port and the dispatcher (FIFO 32)
#define LRING     64u               // launches remembered (LQ_DEPTH ahead + answers not yet read)
#define HIST_MIN  1024u
#define HIST_MAX  (1u << 17)        // completion clocks kept per unit
#define SREGS     16u
#define PRINT_MAX 32u               // reports printed by tsr_sim_destroy (all stay readable)

static inline uint32_t tk_unit(uint32_t t) { return t >> 28; }
static inline uint32_t tk_seq(uint32_t t) { return t & SEQ_MASK; }
static inline uint32_t tk_make(uint32_t u, uint32_t s) { return (u << 28) | (s & SEQ_MASK); }

// ---------------------------------------------------------------------------
// Vector clocks and shadow cells
// ---------------------------------------------------------------------------
typedef struct vc { uint32_t c[NA]; } vc;

typedef struct acc {          // one access: epoch of its agent (0 = none), info, time
    uint32_t ep;
    uint32_t info;            // op: its ticket; RISC-V: pc; host: TSR_RACE side detail
    uint64_t time;
} acc;

typedef struct cell {         // 152 bytes, no padding (compared with memcmp)
    acc      w;
    acc      r[NA];
    uint32_t wa;              // writer's agent
    uint32_t pad;
} cell;

typedef struct seg { uint64_t lo, hi; cell c; } seg;
typedef struct segmap { seg * v; size_t n, cap; } segmap;

// Completion history of one unit.
typedef struct hent {
    vc       v;
    uint32_t seq, ok;
    uint32_t pidx, pc, ra, opcode, launch;
    uint64_t t_done;
} hent;
typedef struct hist {
    hent *   e;
    uint32_t cap;
    uint32_t D;               // next sequence to complete
    uint64_t n_done;          // completions seen
} hist;

// Entries between ENQ and DONE.
enum { K_OP = 0, K_NOP = 1, K_FENCE = 2, K_QCLEAR = 3 };
typedef struct infl {
    vc          push;
    uint32_t    ok, seq, kind, code;
    uint32_t    pidx, pc, ra, opcode, launch, gen, harness;
    uint32_t    dep[2];
    uint32_t    snap_ok;                 // bit v: snap[v] names an entry
    uint32_t    snap[TSR_NUM_UNITS];
    tsr_extents ext;
} infl;

// Pushes between PUSH / QCTRL and ENQ.
typedef struct prec { vc v; uint32_t pidx, pc, ra, launch, gen, harness, qclear; } prec;

typedef struct note { uint32_t ch, write; uint64_t lo, hi; } note;   // an exec's DRAM beats
typedef struct skey { uint64_t k0, k1; } skey;                    // a site
typedef struct dkey { skey a, b; uint32_t cls, idx; } dkey;       // dedup entry (idx + 1; 0 empty)

typedef struct race {
    tsr_sim *  sim;
    uint32_t   mode;              // 1 report, 2 fatal
    int        lis_id, acc_id;
    uint32_t   spm_lines, extent_lines, num_ch;

    vc         rv, host;
    vc         ulast[TSR_NUM_UNITS];
    hist       h[TSR_NUM_UNITS];
    infl       inf[TSR_NUM_UNITS][INFL];
    uint32_t   enq[TSR_NUM_UNITS];
    uint64_t   enq_total[TSR_NUM_UNITS];
    uint64_t   open;              // ticketed entries pushed, not completed
    prec       pq[PQ];
    uint32_t   pq_rd, pq_n;

    // RISC-V side
    uint32_t   wait_ticket;
    int        wait_joined;
    int        rv_run;            // CTRL.RV_RUN as last seen by a host CTRL write

    // launches (EX §4)
    uint32_t   pushed;            // accepted LQ_PUSHes (= LQ_SEQ)
    int        active;
    int        stale_on;          // the running launch was popped by the hart (SPM_STALE window)
    uint32_t   act_s;
    uint32_t   gen;               // SPM write generation: +1 per LQ_POP (0 = before the first)
    uint64_t   pop_time;
    vc         hpush[LRING];  uint32_t hpush_s[LRING];  uint8_t hpush_ok[LRING];
    vc         lclk[LRING];   uint32_t lclk_s[LRING];   uint8_t lclk_ok[LRING];
    vc         post[TSR_LCQ_DEPTH];
    uint32_t   post_ans[TSR_LCQ_DEPTH];
    uint8_t    post_ok[TSR_LCQ_DEPTH];

    // DRAM beats reported by the running exec (tsr_race_cap_note), for the DONE that follows
    note *     notes;
    size_t     n_notes, cap_notes;

    // shadow
    cell *     spm;
    uint32_t * sgen;
    uint64_t * smask;
    cell       sreg[SREGS];
    segmap *   dram;

    // reports
    tsr_race_report * rep;
    uint64_t * rep_scan;          // last scan that counted it
    uint64_t * rep_new;           // scan that created it
    size_t     n_rep, cap_rep;
    dkey *     ht;
    size_t     ht_cap, ht_n;
    uint64_t   scan;
    int        fatal_pending;

    tsr_race_stats st;
} race;

static race * race_of(const tsr_sim * sim);

static void * xcalloc(size_t n, size_t sz) {
    void * p = calloc(n ? n : 1, sz ? sz : 1);
    if (p == NULL) {
        fprintf(stderr, "tsr_race: out of memory (%zu x %zu bytes)\n", n, sz);
        abort();
    }
    return p;
}
static void * xrealloc(void * p, size_t n) {
    void * q = realloc(p, n ? n : 1);
    if (q == NULL) {
        fprintf(stderr, "tsr_race: out of memory (%zu bytes)\n", n);
        abort();
    }
    return q;
}

static inline void vc_join(race * R, vc * a, const vc * b) {
    for (uint32_t i = 0; i < NA; ++i)
        if (b->c[i] > a->c[i]) a->c[i] = b->c[i];
    R->st.joins++;
}

// ---------------------------------------------------------------------------
// Completion history
// ---------------------------------------------------------------------------
static void hist_grow(hist * h) {
    const uint32_t ncap = h->cap ? 2u * h->cap : HIST_MIN;
    hent * ne = (hent *)xcalloc(ncap, sizeof(hent));
    for (uint32_t i = 0; i < h->cap; ++i)
        if (h->e[i].ok) ne[h->e[i].seq & (ncap - 1u)] = h->e[i];
    free(h->e);
    h->e = ne;
    h->cap = ncap;
}

static hent * hist_put(hist * h, uint32_t seq) {
    if (h->cap == 0) hist_grow(h);
    for (;;) {
        hent * e = &h->e[seq & (h->cap - 1u)];
        if (!e->ok || e->seq == seq || h->cap >= HIST_MAX) {
            memset(e, 0, sizeof(*e));
            e->ok = 1;
            e->seq = seq;
            return e;
        }
        hist_grow(h);
    }
}

// The entry of (u, seq) if kept, else NULL.
static const hent * hist_find(const race * R, uint32_t u, uint32_t seq) {
    const hist * h = &R->h[u];
    if (h->cap == 0) return NULL;
    const hent * e = &h->e[seq & (h->cap - 1u)];
    return (e->ok && e->seq == seq) ? e : NULL;
}

// Completion `seq` of unit u is known (it completed: (D - seq) mod 2^28 in [1, n_done]).
static int hist_done(const race * R, uint32_t u, uint32_t seq) {
    const hist * h = &R->h[u];
    const uint32_t d = (h->D - seq) & SEQ_MASK;
    return d >= 1u && d < SEQ_HALF && (uint64_t)d <= h->n_done;
}

// Joins the clock of completion (u, seq) into *v. A completion older than the kept history joins
// the oldest kept one (later by queue order: conservative).
static void join_completion(race * R, vc * v, uint32_t u, uint32_t seq) {
    if (u >= TSR_NUM_UNITS || !hist_done(R, u, seq)) return;
    const hent * e = hist_find(R, u, seq);
    if (e == NULL) {
        const hist * h = &R->h[u];
        const uint64_t kept = h->n_done < h->cap ? h->n_done : h->cap;
        e = hist_find(R, u, (h->D - (uint32_t)kept) & SEQ_MASK);
        if (e == NULL) e = hist_find(R, u, (h->D - 1u) & SEQ_MASK);
        R->st.old_refs++;
        if (e == NULL) return;
    }
    vc_join(R, v, &e->v);
}

static void join_ticket(race * R, vc * v, uint32_t t) { join_completion(R, v, tk_unit(t), tk_seq(t)); }

// A read of DONE_SEQ[u] / Q_DONE[u] returning `val`: every completion below it (E5).
static void observe_unit(race * R, vc * v, uint32_t u, uint32_t val) {
    if (u >= TSR_NUM_UNITS) return;
    join_completion(R, v, u, (val - 1u) & SEQ_MASK);
}

static void observe_all(race * R, vc * v) {
    for (uint32_t u = 0; u < TSR_NUM_UNITS; ++u)
        if (R->h[u].n_done) vc_join(R, v, &R->ulast[u]);
}

// ---------------------------------------------------------------------------
// Reports
// ---------------------------------------------------------------------------
static const char * unit_name(uint32_t u) {
    static const char * const n[TSR_NUM_UNITS] = { "DMA", "QACT", "MVM", "VU", "ATT", "SYNC" };
    return u < TSR_NUM_UNITS ? n[u] : "?";
}

static const char * op_name(uint32_t op) {
    switch (op) {
    case TSR_OP_NOP:    return "NOP";    case TSR_OP_FENCE:  return "FENCE";
    case TSR_OP_LD:     return "LD";     case TSR_OP_ST:     return "ST";     case TSR_OP_KVST:   return "KVST";
    case TSR_OP_QACT:   return "QACT";   case TSR_OP_MVM:    return "MVM";    case TSR_OP_ATTQK:  return "ATTQK";
    case TSR_OP_ATTPV:  return "ATTPV";  case TSR_OP_VOP:    return "VOP";    case TSR_OP_RED:    return "RED";
    case TSR_OP_ROPE:   return "ROPE";   case TSR_OP_TOPK:   return "TOPK";   case TSR_OP_ASMAX:  return "ASMAX";
    case TSR_OP_AMASK:  return "AMASK";  case TSR_OP_AMERGE: return "AMERGE"; case TSR_OP_ANORM:  return "ANORM";
    case TSR_OP_VMASK:  return "VMASK";  case TSR_OP_SOP:    return "SOP";    case TSR_OP_DQROW:  return "DQROW";
    case TSR_OP_GDN:    return "GDN";
    default:            return "op";
    }
}

// The report side of an access in the shadow.
static void side_of(const race * R, tsr_race_side * s, uint32_t agent, const acc * x, uint32_t write) {
    memset(s, 0, sizeof(*s));
    s->agent = agent;
    s->write = write;
    s->ticket = TSR_TICKET_NONE;
    s->launch = NO_LAUNCH;
    s->time = x->time;
    if (agent < TSR_NUM_UNITS) {
        s->ticket = x->info;
        const hent * e = hist_find(R, tk_unit(x->info), tk_seq(x->info));
        if (e != NULL) {
            s->pidx = e->pidx;
            s->pc = e->pc;
            s->ra = e->ra;
            s->opcode = e->opcode;
            s->launch = e->launch;
        } else {
            s->detail = 1;                               // left the history: known by its ticket only
        }
    } else if (agent == A_RV) {
        s->pc = x->info;
    } else {
        s->detail = x->info;
    }
}

// Site key of a side (dedup): op (unit, opcode, push pc, ra) or (unit, push index) for a test-hook
// push; RISC-V pc; host direction and detail.
static skey site_key(const tsr_race_side * s) {
    const uint64_t a = (uint64_t)s->agent << 56;
    skey k;
    k.k1 = 0;
    if (s->agent < TSR_NUM_UNITS) {
        if (s->detail) {
            k.k0 = a | (3ull << 48);
            k.k1 = s->ticket;
        } else if (s->pc != 0) {
            k.k0 = a | (1ull << 48) | ((uint64_t)s->opcode << 32) | s->pc;
            k.k1 = s->ra;
        } else {
            k.k0 = a | (2ull << 48);
            k.k1 = s->pidx;
        }
    } else if (s->agent == A_RV) {
        k.k0 = a | s->pc;
    } else {
        k.k0 = a | ((uint64_t)s->write << 8) | s->detail;
    }
    return k;
}
static int skey_lt(skey x, skey y) { return x.k0 < y.k0 || (x.k0 == y.k0 && x.k1 < y.k1); }
static int skey_eq(skey x, skey y) { return x.k0 == y.k0 && x.k1 == y.k1; }

static uint64_t mix64(uint64_t x) {
    x ^= x >> 33; x *= 0xFF51AFD7ED558CCDull;
    x ^= x >> 33; x *= 0xC4CEB9FE1A85EC53ull;
    return x ^ (x >> 33);
}

// Index + 1 of the report with this key, or 0; *slot = where to insert it.
static size_t ht_find(const race * R, uint32_t cls, skey a, skey b, size_t * slot) {
    const uint64_t h = mix64(mix64(a.k0 ^ (a.k1 << 7)) + mix64(b.k0 ^ (b.k1 << 7)) * 31u + cls);
    size_t i = (size_t)h & (R->ht_cap - 1u);
    for (;;) {
        const dkey * k = &R->ht[i];
        if (k->idx == 0) { *slot = i; return 0; }
        if (k->cls == cls && skey_eq(k->a, a) && skey_eq(k->b, b)) { *slot = i; return k->idx; }
        i = (i + 1u) & (R->ht_cap - 1u);
    }
}

static void ht_grow(race * R) {
    dkey * old = R->ht;
    const size_t oc = R->ht_cap;
    R->ht_cap = oc ? 2u * oc : 256u;
    R->ht = (dkey *)xcalloc(R->ht_cap, sizeof(dkey));
    for (size_t i = 0; i < oc; ++i) {
        if (old[i].idx == 0) continue;
        size_t s;
        (void)ht_find(R, old[i].cls, old[i].a, old[i].b, &s);
        R->ht[s] = old[i];
    }
    free(old);
}

// Finds or creates the report of (cls, a, b). Returns its index; *created = 1 if new.
static size_t rep_get(race * R, uint32_t cls, skey a, skey b, int * created) {
    if (2u * (R->ht_n + 1u) > R->ht_cap) ht_grow(R);
    size_t slot;
    const size_t idx = ht_find(R, cls, a, b, &slot);
    *created = idx == 0;
    if (idx != 0) return idx - 1u;
    if (R->n_rep == R->cap_rep) {
        R->cap_rep = R->cap_rep ? 2u * R->cap_rep : 16u;
        R->rep = (tsr_race_report *)xrealloc(R->rep, R->cap_rep * sizeof(tsr_race_report));
        R->rep_scan = (uint64_t *)xrealloc(R->rep_scan, R->cap_rep * sizeof(uint64_t));
        R->rep_new = (uint64_t *)xrealloc(R->rep_new, R->cap_rep * sizeof(uint64_t));
    }
    const size_t n = R->n_rep++;
    memset(&R->rep[n], 0, sizeof(R->rep[n]));
    R->rep_scan[n] = R->rep_new[n] = 0;
    R->ht[slot].cls = cls;
    R->ht[slot].a = a;
    R->ht[slot].b = b;
    R->ht[slot].idx = (uint32_t)(n + 1u);
    R->ht_n++;
    return n;
}

// One occurrence of report idx in the current scan over [lo, hi): the first one counts; a
// contiguous continuation of the report's first occurrence extends its range.
static void rep_hit(race * R, size_t idx, uint64_t lo, uint64_t hi) {
    tsr_race_report * r = &R->rep[idx];
    if (R->rep_scan[idx] == R->scan) {
        if (R->rep_new[idx] == R->scan && lo == r->hi) r->hi = hi;
        return;
    }
    R->rep_scan[idx] = R->scan;
    r->count++;
    R->st.reports++;
}

// The current access (the later side of a conflict).
typedef struct actx {
    uint32_t       agent, write;
    const vc *     clk;
    acc            me;
    uint32_t       space, ch;
    tsr_race_side  side;
    skey           site;
} actx;

static void conflict(race * R, const actx * a, uint32_t kind, uint32_t e_agent, const acc * e, uint32_t e_write,
                     uint64_t lo, uint64_t hi) {
    tsr_race_side first;
    side_of(R, &first, e_agent, e, e_write);
    const skey ka = site_key(&first), kb = a->site;
    const uint32_t cls = kind == TSR_RACE_WAW ? 1u : 0u;
    int created;
    const size_t idx = rep_get(R, cls, skey_lt(ka, kb) ? ka : kb, skey_lt(ka, kb) ? kb : ka, &created);
    if (created) {
        tsr_race_report * r = &R->rep[idx];
        r->kind = kind;
        r->space = a->space;
        r->ch = a->ch;
        r->lo = lo;
        r->hi = hi;
        r->first = first;
        r->second = a->side;
        r->launch = a->side.launch;
        R->rep_new[idx] = R->scan;
        if (R->mode >= 2) R->fatal_pending = 1;
    }
    rep_hit(R, idx, lo, hi);
}

static void stale(race * R, const actx * a, uint64_t lo, uint64_t hi) {
    int created;
    const skey none = { 0, 0 };
    const size_t idx = rep_get(R, 2u, a->site, none, &created);
    if (created) {
        tsr_race_report * r = &R->rep[idx];
        r->kind = TSR_RACE_SPM_STALE;
        r->space = TSR_RACE_SPACE_SPM;
        r->lo = lo;
        r->hi = hi;
        r->first = a->side;
        memset(&r->second, 0, sizeof(r->second));
        r->second.agent = TSR_RACE_AGENT_NONE;
        r->second.ticket = TSR_TICKET_NONE;
        r->second.launch = NO_LAUNCH;
        r->launch = R->act_s;
        r->since = R->pop_time;
        R->rep_new[idx] = R->scan;
        if (R->mode >= 2) R->fatal_pending = 1;
    }
    rep_hit(R, idx, lo, hi);
}

// Checks and records one access on one shadow cell covering [lo, hi) (report units).
static void cell_do(race * R, cell * c, const actx * a, uint64_t lo, uint64_t hi) {
    const vc * C = a->clk;
    if (c->w.ep != 0 && c->wa != a->agent && c->w.ep > C->c[c->wa])
        conflict(R, a, a->write ? TSR_RACE_WAW : TSR_RACE_RAW, c->wa, &c->w, 1, lo, hi);
    if (a->write) {
        for (uint32_t g = 0; g < NA; ++g)
            if (g != a->agent && c->r[g].ep != 0 && c->r[g].ep > C->c[g]) conflict(R, a, TSR_RACE_WAR, g, &c->r[g], 0, lo, hi);
        c->w = a->me;
        c->wa = a->agent;
        memset(c->r, 0, sizeof(c->r));
    } else {
        c->r[a->agent] = a->me;
    }
}

// ---------------------------------------------------------------------------
// DRAM interval maps (beats, per channel)
// ---------------------------------------------------------------------------
static size_t seg_first(const segmap * m, uint64_t x) {      // first i with v[i].hi > x
    size_t a = 0, b = m->n;
    while (a < b) {
        const size_t mid = a + (b - a) / 2u;
        if (m->v[mid].hi <= x) a = mid + 1u;
        else b = mid;
    }
    return a;
}

static seg * seg_insert(race * R, segmap * m, size_t i) {
    if (m->n == m->cap) {
        m->cap = m->cap ? 2u * m->cap : 64u;
        m->v = (seg *)xrealloc(m->v, m->cap * sizeof(seg));
    }
    memmove(&m->v[i + 1u], &m->v[i], (m->n - i) * sizeof(seg));
    m->n++;
    if (++R->st.dram_segments > R->st.dram_segments_max) R->st.dram_segments_max = R->st.dram_segments;
    return &m->v[i];
}

// Merges equal neighbours among v[a, b).
static void seg_merge(race * R, segmap * m, size_t a, size_t b) {
    if (b > m->n) b = m->n;
    if (b <= a + 1u) return;
    size_t w = a;
    for (size_t r = a + 1u; r < b; ++r) {
        if (m->v[w].hi == m->v[r].lo && memcmp(&m->v[w].c, &m->v[r].c, sizeof(cell)) == 0) m->v[w].hi = m->v[r].hi;
        else m->v[++w] = m->v[r];
    }
    const size_t gone = (b - 1u) - w;
    if (gone == 0) return;
    memmove(&m->v[w + 1u], &m->v[b], (m->n - b) * sizeof(seg));
    m->n -= gone;
    R->st.dram_segments -= gone;
}

static void dram_access(race * R, uint32_t ch, uint64_t lo, uint64_t hi, const actx * a) {
    if (ch >= R->num_ch || hi <= lo) return;
    segmap * m = &R->dram[ch];
    size_t i = seg_first(m, lo);
    const size_t i0 = i;
    uint64_t p = lo;
    while (p < hi) {
        if (i < m->n && m->v[i].lo <= p) {
            if (m->v[i].lo < p) {                       // split at p
                seg * s = seg_insert(R, m, i + 1u);
                *s = m->v[i];
                s->lo = p;
                m->v[i].hi = p;
                ++i;
            }
            if (m->v[i].hi > hi) {                      // split at hi
                seg * s = seg_insert(R, m, i + 1u);
                *s = m->v[i];
                s->lo = hi;
                m->v[i].hi = hi;
            }
        } else {                                        // gap: a fresh segment
            const uint64_t e = (i < m->n && m->v[i].lo < hi) ? m->v[i].lo : hi;
            seg * s = seg_insert(R, m, i);
            memset(s, 0, sizeof(*s));
            s->lo = p;
            s->hi = e;
        }
        R->st.op_lines++;
        cell_do(R, &m->v[i].c, a, m->v[i].lo, m->v[i].hi);
        p = m->v[i].hi;
        ++i;
    }
    seg_merge(R, m, i0 ? i0 - 1u : 0u, i + 1u);
}

// ---------------------------------------------------------------------------
// SPM (line cells, SPM_STALE generations)
// ---------------------------------------------------------------------------
static inline uint64_t byte_mask(uint32_t off, uint32_t n) {
    return n >= 64u ? ~0ull : (((1ull << n) - 1u) << off);
}

// Marks bytes of a line written in the current generation.
static void spm_mark(race * R, uint32_t line, uint64_t mask) {
    if (R->sgen[line] != R->gen) {
        R->sgen[line] = R->gen;
        R->smask[line] = 0;
    }
    R->smask[line] |= mask;
}

// An op's SPM extent [l0, l1) (lines). gen: the op's launch generation (0: not in a launch).
static void spm_op(race * R, uint64_t l0, uint64_t l1, const actx * a, uint32_t gen) {
    if (l1 > R->extent_lines) l1 = R->extent_lines;
    for (uint64_t l = l0; l < l1; ++l) {
        R->st.op_lines++;
        if (!a->write && gen != 0 && R->sgen[l] < gen) stale(R, a, l * TSR_LINE_BYTES, (l + 1u) * TSR_LINE_BYTES);
        cell_do(R, &R->spm[l], a, l * TSR_LINE_BYTES, (l + 1u) * TSR_LINE_BYTES);
        if (a->write) spm_mark(R, (uint32_t)l, ~0ull);
    }
}

// ---------------------------------------------------------------------------
// Agents' synchronisation
// ---------------------------------------------------------------------------
// The harness (test hooks, or anything while the hart is in reset) acts for the RISC-V and the host.
static void harness_sync(race * R) {
    vc m = R->rv;
    vc_join(R, &m, &R->host);
    if (R->open == 0) observe_all(R, &m);
    R->rv = m;
    R->host = m;
}

// Before every event of the hart: the WAIT interrupt edge (header comment).
static void rv_sync(race * R) {
    const uint32_t t = R->wait_ticket;
    if (R->wait_joined || tk_unit(t) >= TSR_NUM_UNITS || !hist_done(R, tk_unit(t), tk_seq(t))) return;
    join_ticket(R, &R->rv, t);
    R->wait_joined = 1;
}

static int is_harness(const race * R, uint32_t pc) { return pc == TSR_SIM_PC_HOOK || !tsr_sim_rv_run(R->sim); }

// x1 of the built-in hart (called during its push store: the caller of a push helper).
static uint32_t hart_ra(const race * R) {
#if defined(TSR_SIM_HAVE_RV) && TSR_SIM_HAVE_RV
    struct tsr_rv_agent * ag = tsr_sim_rv_agent(R->sim);
    return ag ? tsr_rv_reg(tsr_rv_agent_rv(ag), 1) : 0u;
#else
    (void)R;
    return 0u;
#endif
}

// A release by the RISC-V or the host: the agent advances its epoch, so that its later accesses
// are not ordered before what it released. The harness is both, so both advance.
static void release(race * R, uint32_t agent, int harness) {
    if (agent == A_RV || harness) R->rv.c[A_RV]++;
    if (agent == A_HOST || harness) R->host.c[A_HOST]++;
}

static void rv_side(const race * R, actx * a, uint32_t write, uint32_t pc) {
    memset(a, 0, sizeof(*a));
    a->agent = A_RV;
    a->write = write;
    a->clk = &R->rv;
    a->me.ep = R->rv.c[A_RV];
    a->me.info = pc;
    a->me.time = tsr_sim_now(R->sim);
    a->side.agent = A_RV;
    a->side.write = write;
    a->side.ticket = TSR_TICKET_NONE;
    a->side.pc = pc;
    a->side.launch = R->active ? R->act_s : NO_LAUNCH;
    a->side.time = a->me.time;
    a->site = site_key(&a->side);
}

static void host_side(const race * R, actx * a, uint32_t write, uint32_t detail) {
    memset(a, 0, sizeof(*a));
    a->agent = A_HOST;
    a->write = write;
    a->clk = &R->host;
    a->me.ep = R->host.c[A_HOST];
    a->me.info = detail;
    a->me.time = tsr_sim_now(R->sim);
    a->side.agent = A_HOST;
    a->side.write = write;
    a->side.ticket = TSR_TICKET_NONE;
    a->side.detail = detail;
    a->side.launch = NO_LAUNCH;
    a->side.time = a->me.time;
    a->site = site_key(&a->side);
}

static void launch_end(race * R) {
    R->active = 0;
}

// The running launch is still ACTIVE in the core (TAKEN != ANSWERED): a RISC-V reset that no host
// CTRL write announced (DB's ndmreset, CC §12.2) answers it too.
static int launch_running(race * R) {
    if (R->active && tsr_sim_corewin_read(R->sim, TSR_CSR_LQ_TAKEN) == tsr_sim_corewin_read(R->sim, TSR_CSR_LCQ_ANSWERED))
        launch_end(R);
    return R->active;
}

static void fatal_check(race * R) {
    if (!R->fatal_pending) return;
    fprintf(stderr, "tsr_sim: race detector (TSR_SIM_RACE=fatal): %zu report(s) at MTIME %llu\n", R->n_rep,
            (unsigned long long)tsr_sim_now(R->sim));
    char line[768];
    for (size_t i = 0; i < R->n_rep; ++i) {
        tsr_race_format(&R->rep[i], line, sizeof(line));
        fprintf(stderr, "  %s\n", line);
    }
    fflush(stderr);
    abort();
}

// ---------------------------------------------------------------------------
// Queue events
// ---------------------------------------------------------------------------
static void push_rec(race * R, const tsr_q_event * ev, int qclear) {
    const int harness = is_harness(R, ev->pc);
    if (harness) harness_sync(R);
    else rv_sync(R);
    if (R->pq_n == PQ) {                                 // cannot happen (issue FIFO depth 32)
        R->pq_rd = (R->pq_rd + 1u) % PQ;
        R->pq_n--;
    }
    prec * p = &R->pq[(R->pq_rd + R->pq_n++) % PQ];
    p->v = R->rv;
    p->pidx = ev->pidx;
    p->pc = ev->pc;
    p->ra = harness ? 0u : hart_ra(R);
    p->launch = launch_running(R) ? R->act_s : NO_LAUNCH;
    p->gen = (R->active && R->stale_on) ? R->gen : 0u;
    p->harness = (uint32_t)harness;
    p->qclear = (uint32_t)qclear;
    release(R, A_RV, harness);                           // E4
    R->open++;
}

static void on_enq(race * R, const tsr_q_event * ev) {
    const uint32_t u = ev->unit;
    if (u >= TSR_NUM_UNITS) return;
    const uint32_t seq = tk_seq(ev->ticket);
    prec p;
    int found = 0;
    while (R->pq_n) {
        prec * q = &R->pq[R->pq_rd];
        R->pq_rd = (R->pq_rd + 1u) % PQ;
        R->pq_n--;
        if (q->pidx == ev->pidx) {
            p = *q;
            found = 1;
            break;
        }
    }
    if (!found) {                                        // no PUSH seen (cannot happen): as a harness push now
        memset(&p, 0, sizeof(p));
        p.v = R->rv;
        p.pidx = ev->pidx;
        p.pc = ev->pc;
        p.launch = R->active ? R->act_s : NO_LAUNCH;
        p.gen = (R->active && R->stale_on) ? R->gen : 0u;
        p.harness = 1;
        release(R, A_RV, 1);
        R->open++;
    }
    infl * f = &R->inf[u][seq % INFL];
    memset(f, 0, sizeof(*f));
    f->ok = 1;
    f->seq = seq;
    f->push = p.v;
    f->pidx = ev->pidx;
    f->pc = p.pc ? p.pc : ev->pc;
    f->ra = p.ra;
    f->launch = p.launch;
    f->gen = p.gen;
    f->harness = p.harness;
    f->code = ev->code;
    f->dep[0] = ev->dep[0];
    f->dep[1] = ev->dep[1];
    f->opcode = ev->w ? (ev->w[0] & 0xFFu) : 0u;
    f->kind = K_OP;
    if (p.qclear) f->kind = K_QCLEAR, f->opcode = 0xFFu;
    else if (ev->code == 0 && f->opcode == TSR_OP_FENCE) f->kind = K_FENCE;
    else if (ev->code == 0 && f->opcode == TSR_OP_NOP) f->kind = K_NOP;
    if (f->kind == K_FENCE || f->kind == K_QCLEAR) {     // the dispatch snapshot (E6)
        for (uint32_t v = 0; v < TSR_NUM_UNITS; ++v) {
            if (R->enq_total[v] == 0) continue;
            f->snap[v] = (R->enq[v] - 1u) & SEQ_MASK;
            f->snap_ok |= 1u << v;
        }
    }
    if (ev->code == 0 && ev->ext != NULL) f->ext = *ev->ext;
    R->enq[u] = (R->enq[u] + 1u) & SEQ_MASK;
    R->enq_total[u]++;
}

static void op_accesses(race * R, uint32_t u, const infl * f, const vc * v, uint32_t ticket, uint64_t now) {
    actx a;
    memset(&a, 0, sizeof(a));
    a.agent = u;
    a.clk = v;
    a.me.ep = v->c[u];
    a.me.info = ticket;
    a.me.time = now;
    a.side.agent = u;
    a.side.ticket = ticket;
    a.side.pidx = f->pidx;
    a.side.opcode = f->opcode;
    a.side.pc = f->harness ? 0u : f->pc;
    a.side.ra = f->harness ? 0u : f->ra;
    a.side.launch = f->launch;
    a.side.time = now;
    a.site = site_key(&a.side);
    // reads first, then writes (a read-modify-write operand is listed both ways)
    for (int pass = 0; pass < 2; ++pass) {
        for (uint32_t i = 0; i < f->ext.n; ++i) {
            const tsr_extent * e = &f->ext.e[i];
            if ((uint32_t)e->write != (uint32_t)pass) continue;
            a.write = (uint32_t)pass;
            a.side.write = (uint32_t)pass;
            switch (e->kind) {
            case TSR_EXT_SPM:
                a.space = TSR_RACE_SPACE_SPM;
                a.ch = 0;
                spm_op(R, e->lo, e->hi, &a, f->gen);
                break;
            case TSR_EXT_ATT_LOCAL:
                a.space = TSR_RACE_SPACE_ATT_LOCAL;
                a.ch = 0;
                spm_op(R, e->lo + R->spm_lines, e->hi + R->spm_lines, &a, f->gen);
                break;
            case TSR_EXT_DRAM:
                a.space = TSR_RACE_SPACE_DRAM;
                if (R->n_notes != 0) break;          // the exec's exact beats instead (below)
                if (e->ch == TSR_EXT_CH_ALL) {
                    for (uint32_t c = 0; c < R->num_ch; ++c) {
                        a.ch = c;
                        dram_access(R, c, e->lo, e->hi, &a);
                    }
                } else {
                    a.ch = e->ch;
                    dram_access(R, e->ch, e->lo, e->hi, &a);
                }
                break;
            case TSR_EXT_SREG:
                a.space = TSR_RACE_SPACE_SREG;
                a.ch = 0;
                for (uint64_t s = e->lo; s < e->hi && s < SREGS; ++s) {
                    R->st.op_lines++;
                    cell_do(R, &R->sreg[s], &a, s, s + 1u);
                }
                break;
            default:
                break;
            }
        }
        // DRAM: the beats the exec reported (tsr_race_cap_note), clipped to the op's DRAM extent.
        for (size_t k = 0; k < R->n_notes; ++k) {
            const note * nt = &R->notes[k];
            if (nt->write != (uint32_t)pass) continue;
            for (uint32_t i = 0; i < f->ext.n; ++i) {
                const tsr_extent * e = &f->ext.e[i];
                if (e->kind != TSR_EXT_DRAM || (e->ch != TSR_EXT_CH_ALL && e->ch != nt->ch)) continue;
                const uint64_t lo = nt->lo > e->lo ? nt->lo : e->lo, hi = nt->hi < e->hi ? nt->hi : e->hi;
                if (lo >= hi) continue;
                a.write = (uint32_t)pass;
                a.side.write = (uint32_t)pass;
                a.space = TSR_RACE_SPACE_DRAM;
                a.ch = nt->ch;
                dram_access(R, nt->ch, lo, hi, &a);
                break;
            }
        }
    }
}

static void on_done(race * R, const tsr_q_event * ev) {
    const uint32_t u = ev->unit;
    if (u >= TSR_NUM_UNITS) return;
    const uint32_t seq = tk_seq(ev->ticket);
    infl * f = &R->inf[u][seq % INFL];
    infl tmp;
    if (!f->ok || f->seq != seq) {                       // no ENQ seen (cannot happen): no edges but E1
        memset(&tmp, 0, sizeof(tmp));
        tmp.seq = seq;
        tmp.pidx = ev->pidx;
        tmp.launch = NO_LAUNCH;
        tmp.dep[0] = tmp.dep[1] = TSR_TICKET_NONE;
        f = &tmp;
    }
    R->st.ops++;
    vc v = f->push;
    vc_join(R, &v, &R->ulast[u]);                        // E1
    for (int i = 0; i < 2; ++i)
        if (tk_unit(f->dep[i]) < TSR_NUM_UNITS) join_ticket(R, &v, f->dep[i]);   // E2
    for (uint32_t w = 0; w < TSR_NUM_UNITS; ++w)
        if (f->snap_ok & (1u << w)) join_completion(R, &v, w, f->snap[w]);      // E6
    v.c[u] = R->ulast[u].c[u] + 1u;
    // History first (a report about this op looks it up).
    hist * h = &R->h[u];
    hent * e = hist_put(h, seq);
    e->v = v;
    e->pidx = f->pidx;
    e->pc = f->harness ? 0u : f->pc;
    e->ra = f->harness ? 0u : f->ra;
    e->opcode = f->opcode;
    e->launch = f->launch;
    e->t_done = ev->time;
    h->D = (seq + 1u) & SEQ_MASK;
    h->n_done++;
    R->ulast[u] = v;
    if (R->open) R->open--;
    if (f->launch != NO_LAUNCH) {
        const uint32_t k = f->launch % LRING;
        if (R->lclk_ok[k] && R->lclk_s[k] == f->launch) vc_join(R, &R->lclk[k], &v);
    }
    if (ev->ran && ev->code == 0 && f->code == 0 && f->ext.n != 0) {
        R->scan++;
        op_accesses(R, u, f, &v, tk_make(u, seq), ev->time);
    }
    R->n_notes = 0;
    f->ok = 0;
}

static void q_listener(void * ctx, tsr_sim * sim, const tsr_q_event * ev) {
    (void)sim;
    race * R = (race *)ctx;
    if (ev->kind != TSR_QEV_DONE) R->n_notes = 0;   // notes belong to the exec just before a DONE
    switch (ev->kind) {
    case TSR_QEV_PUSH:  push_rec(R, ev, 0); break;
    case TSR_QEV_QCTRL:
        if (ev->w != NULL && (ev->w[0] & TSR_QCTRL_QCLEAR)) push_rec(R, ev, 1);
        else if (is_harness(R, ev->pc)) harness_sync(R);
        else rv_sync(R);
        break;
    case TSR_QEV_ENQ:   on_enq(R, ev); break;
    case TSR_QEV_DONE:  on_done(R, ev); break;
    default: break;
    }
    fatal_check(R);
}

// ---------------------------------------------------------------------------
// Access events
// ---------------------------------------------------------------------------
static void sreg_access(race * R, actx * a, uint32_t i) {
    if (i >= SREGS) return;
    a->space = TSR_RACE_SPACE_SREG;
    a->ch = 0;
    R->scan++;
    cell_do(R, &R->sreg[i], a, i, i + 1u);
}


static void rv_spm(race * R, const tsr_sim_acc * x, uint32_t write) {
    const uint64_t addr = x->addr;
    const uint32_t n = x->size ? x->size : (uint32_t)x->len;
    if (n == 0 || addr >= (uint64_t)R->spm_lines * TSR_LINE_BYTES) return;
    const uint32_t line = (uint32_t)(addr / TSR_LINE_BYTES), off = (uint32_t)(addr % TSR_LINE_BYTES);
    const uint32_t nb = off + n > TSR_LINE_BYTES ? TSR_LINE_BYTES - off : n;
    const uint64_t mask = byte_mask(off, nb);
    actx a;
    rv_side(R, &a, write, x->pc);
    a.space = TSR_RACE_SPACE_SPM;
    R->scan++;
    if (!write && R->stale_on && launch_running(R)) {
        // A line-buffer fill (64 bytes) is checked like an op's line; an exact read per byte.
        const int st = nb >= TSR_LINE_BYTES ? R->sgen[line] != R->gen
                                            : (R->sgen[line] != R->gen || (R->smask[line] & mask) != mask);
        if (st) stale(R, &a, addr, addr + nb);
    }
    cell_do(R, &R->spm[line], &a, addr, addr + nb);
    if (write) spm_mark(R, line, mask);
}

// A CSR offset read by the RISC-V through the core window, or by the host.
static void csr_observe(race * R, vc * clk, actx * a, uint32_t off, uint32_t val) {
    if (off >= TSR_CSR_Q_DONE(0) && off < TSR_CSR_Q_DONE(TSR_NUM_UNITS)) observe_unit(R, clk, (off - TSR_CSR_Q_DONE(0)) / 4u, val);
    else if (off == TSR_CSR_STATUS && (val & TSR_STATUS_QIDLE)) observe_all(R, clk);
    else if (off >= TSR_CSR_SREG(0) && off < TSR_CSR_SREG(SREGS)) sreg_access(R, a, (off - TSR_CSR_SREG(0)) / 4u);
}

static void on_rv(race * R, const tsr_sim_acc * x) {
    if (is_harness(R, x->pc)) harness_sync(R);
    else rv_sync(R);
    R->st.rv_accesses++;
    switch (x->kind) {
    case TSR_ACC_RV_SPM_RD: rv_spm(R, x, 0); break;
    case TSR_ACC_RV_SPM_WR: rv_spm(R, x, 1); break;
    case TSR_ACC_RV_CW_RD:
        if (x->off >= TSR_CW_DONE_SEQ(0) && x->off < TSR_CW_DONE_SEQ(TSR_NUM_UNITS))
            observe_unit(R, &R->rv, (x->off - TSR_CW_DONE_SEQ(0)) / 4u, x->val);
        else if (x->off == TSR_CW_QSTATUS && (x->val & TSR_QSTATUS_QIDLE))
            observe_all(R, &R->rv);
        break;
    case TSR_ACC_RV_CORE_RD: {
        actx a;
        rv_side(R, &a, 0, x->pc);
        csr_observe(R, &R->rv, &a, x->off, x->val);
        break;
    }
    case TSR_ACC_RV_CW_WR:
        switch (x->off) {
        case TSR_CW_WAIT_TICKET:
            R->wait_ticket = x->val;
            R->wait_joined = 0;
            break;
        case TSR_CW_LQ_POP: {                            // accepted (faults are not reported)
            const uint32_t s = tsr_sim_corewin_read(R->sim, TSR_CSR_LQ_TAKEN) - 1u;
            const uint32_t k = s % LRING;
            if (R->hpush_ok[k] && R->hpush_s[k] == s) vc_join(R, &R->rv, &R->hpush[k]);   // E7
            // A pop by the harness (test hooks) has no runtime behind it and fixtures its SPM through
            // the tsr_sim_spm() pointer: launch tracking (H1 / H2) without an SPM_STALE window.
            R->active = 1;
            R->stale_on = !is_harness(R, x->pc);
            R->act_s = s;
            R->gen++;
            R->pop_time = tsr_sim_now(R->sim);
            memset(&R->lclk[k], 0, sizeof(vc));
            R->lclk_s[k] = s;
            R->lclk_ok[k] = 1;
            break;
        }
        case TSR_CW_LCQ_POST: {                          // accepted
            const uint32_t kk = tsr_sim_corewin_read(R->sim, TSR_CSR_LCQ_WR) - 1u;
            const uint32_t slot = kk % TSR_LCQ_DEPTH;
            R->post[slot] = R->rv;
            R->post_ok[slot] = 1;
            R->post_ans[slot] = ((x->val & TSR_LCQ_POST_ANSWER) && R->active) ? R->act_s : NO_LAUNCH;
            if (x->val & TSR_LCQ_POST_ANSWER) launch_end(R);
            release(R, A_RV, is_harness(R, x->pc));      // E8
            break;
        }
        default:
            break;
        }
        break;
    default:
        break;
    }
}

static void host_dram(race * R, const tsr_sim_acc * x, uint32_t write) {
    if (x->len == 0) return;
    const uint64_t lo = x->addr / TSR_BEAT_BYTES, hi = (x->addr + x->len + TSR_BEAT_BYTES - 1u) / TSR_BEAT_BYTES;
    actx a;
    host_side(R, &a, write, 1u);
    a.space = TSR_RACE_SPACE_DRAM;
    a.ch = x->ch;
    R->scan++;
    dram_access(R, x->ch, lo, hi, &a);
}

static void on_host(race * R, const tsr_sim_acc * x) {
    if (!tsr_sim_rv_run(R->sim)) harness_sync(R);
    R->st.host_accesses++;
    switch (x->kind) {
    case TSR_ACC_HOST_DRAM_WR: host_dram(R, x, 1); break;
    case TSR_ACC_HOST_DRAM_RD: host_dram(R, x, 0); break;
    case TSR_ACC_HOST_CSR_RD: {
        if (x->off >= TSR_CSR_LCQ_REC_BASE && x->off < TSR_CSR_LCQ_REC_BASE + TSR_CSR_LCQ_REC_STRIDE * TSR_LCQ_DEPTH) {
            const uint32_t slot = (x->off - TSR_CSR_LCQ_REC_BASE) / TSR_CSR_LCQ_REC_STRIDE;
            if (R->post_ok[slot]) {
                vc_join(R, &R->host, &R->post[slot]);                         // E8: the record
                const uint32_t s = R->post_ans[slot], k = s % LRING;
                if (s != NO_LAUNCH && R->lclk_ok[k] && R->lclk_s[k] == s) vc_join(R, &R->host, &R->lclk[k]);   // H2
            }
            break;
        }
        actx a;
        host_side(R, &a, 0, 2u);
        csr_observe(R, &R->host, &a, x->off, x->val);
        break;
    }
    case TSR_ACC_HOST_CSR_WR:
        if (x->off == TSR_CSR_LQ_PUSH) {
            const uint32_t seq = tsr_sim_corewin_read(R->sim, TSR_CSR_LQ_SEQ);
            if (seq != R->pushed) {                      // accepted: launch R->pushed
                const uint32_t s = R->pushed, k = s % LRING;
                R->hpush[k] = R->host;
                R->hpush_s[k] = s;
                R->hpush_ok[k] = 1;
                R->pushed = seq;
                release(R, A_HOST, !tsr_sim_rv_run(R->sim));   // E7
            }
        } else if (x->off == TSR_CSR_CTRL) {
            const int run = tsr_sim_rv_run(R->sim) != 0;
            if (run && !R->rv_run) {                     // E9: the hart starts
                vc_join(R, &R->rv, &R->host);
                release(R, A_HOST, 0);
            }
            if (!run || (x->val & TSR_CTRL_SOFT_RESET)) {   // RISC-V reset sequence: ANSWERED := TAKEN
                launch_end(R);
                R->wait_ticket = TSR_TICKET_NONE;
                R->wait_joined = 0;
            }
            R->rv_run = run;
        }
        break;
    default:
        break;
    }
}

// Debug-module system-bus stores into the SPM windows: written bytes for SPM_STALE (not checked).
static void on_sba(race * R, const tsr_sim_acc * x) {
    if (x->kind != TSR_ACC_SBA_WR) return;
    const uint64_t spm = (uint64_t)R->spm_lines * TSR_LINE_BYTES;
    uint64_t off;
    if (x->addr >= TSR_RV_SPM_BASE && x->addr - TSR_RV_SPM_BASE < spm) off = x->addr - TSR_RV_SPM_BASE;
    else if (x->addr >= TSR_RV_SPM_LB_BASE && x->addr - TSR_RV_SPM_LB_BASE < spm) off = x->addr - TSR_RV_SPM_LB_BASE;
    else return;
    const uint32_t n = x->size ? x->size : 4u, o = (uint32_t)(off % TSR_LINE_BYTES);
    spm_mark(R, (uint32_t)(off / TSR_LINE_BYTES), byte_mask(o, o + n > TSR_LINE_BYTES ? TSR_LINE_BYTES - o : n));
}

static void acc_listener(void * ctx, tsr_sim * sim, const tsr_sim_acc * x) {
    (void)sim;
    race * R = (race *)ctx;
    R->n_notes = 0;
    switch (x->kind) {
    case TSR_ACC_RV_SPM_RD: case TSR_ACC_RV_SPM_WR: case TSR_ACC_RV_CW_RD: case TSR_ACC_RV_CW_WR:
    case TSR_ACC_RV_CORE_RD:
        on_rv(R, x);
        break;
    case TSR_ACC_HOST_CSR_RD: case TSR_ACC_HOST_CSR_WR: case TSR_ACC_HOST_DRAM_RD: case TSR_ACC_HOST_DRAM_WR:
        on_host(R, x);
        break;
    case TSR_ACC_SBA_RD: case TSR_ACC_SBA_WR:
        on_sba(R, x);
        break;
    default:
        break;
    }
    fatal_check(R);
}

// The exec of a running op reports its DRAM beats through tsr_sim_cap_read / _write (tsr_sim_int.h
// §2: exec "captures" every DRAM range it reads or writes); tsr_sim.c passes each call here. They
// are the op's exact DRAM access set (an AT plane op touches a few groups of the plane its extent
// spans, AT §4, §5.8), used at the DONE event that follows the exec in the same time point. Notes
// left by tsr_sim_op (no DONE follows) are dropped by the next other event, and in any case
// clipped to the completing op's DRAM extent.
void tsr_race_cap_note(tsr_sim * sim, uint32_t ch, uint64_t lo, uint64_t n, int write) {
    race * R = race_of(sim);
    if (R == NULL || n == 0) return;
    if (R->n_notes == R->cap_notes) {
        R->cap_notes = R->cap_notes ? 2u * R->cap_notes : 64u;
        R->notes = (note *)xrealloc(R->notes, R->cap_notes * sizeof(note));
    }
    note * nt = &R->notes[R->n_notes++];
    nt->ch = ch;
    nt->write = write ? 1u : 0u;
    nt->lo = lo;
    nt->hi = lo + n;
}

// ---------------------------------------------------------------------------
// Attach / detach, public API
// ---------------------------------------------------------------------------
static race * race_of(const tsr_sim * sim) {
    void ** slot = tsr_sim_model_slot((tsr_sim *)sim, TSR_MODEL_RACE);
    return slot ? (race *)*slot : NULL;
}

void tsr_race_attach(tsr_sim * sim) {
    const uint32_t mode = tsr_sim_opt_race(sim);
    void ** slot = tsr_sim_model_slot(sim, TSR_MODEL_RACE);
    if (mode == 0 || slot == NULL) return;
    race * R = (race *)xcalloc(1, sizeof(race));
    R->sim = sim;
    R->mode = mode;
    R->spm_lines = tsr_sim_spm_lines(sim);
    R->num_ch = tsr_sim_num_ch(sim);
    R->extent_lines = R->spm_lines;
    if (tsr_sim_config(sim)->caps & TSR_CAPS_ATT_LOCAL)
        R->extent_lines += TSR_LOCAL_AT_SLOTS * TSR_LOCAL_AT_SLOT_LINES;
    R->spm = (cell *)xcalloc(R->extent_lines, sizeof(cell));
    R->sgen = (uint32_t *)xcalloc(R->extent_lines, sizeof(uint32_t));
    R->smask = (uint64_t *)xcalloc(R->extent_lines, sizeof(uint64_t));
    R->dram = (segmap *)xcalloc(R->num_ch, sizeof(segmap));
    R->rv.c[A_RV] = 1;
    R->host.c[A_HOST] = 1;
    R->wait_ticket = TSR_TICKET_NONE;
    R->rv_run = tsr_sim_rv_run(sim) != 0;
    R->pushed = tsr_sim_corewin_read(sim, TSR_CSR_LQ_SEQ);
    ht_grow(R);
    R->lis_id = tsr_sim_add_listener(sim, q_listener, R);
    R->acc_id = tsr_sim_add_acc_listener(sim, acc_listener, R);
    if (R->lis_id < 0 || R->acc_id < 0) {
        fprintf(stderr, "tsr_race: no free listener slot; the race detector is off\n");
        if (R->lis_id >= 0) tsr_sim_remove_listener(sim, R->lis_id);
        if (R->acc_id >= 0) tsr_sim_remove_acc_listener(sim, R->acc_id);
        free(R->spm); free(R->sgen); free(R->smask); free(R->dram); free(R->ht);
        free(R);
        return;
    }
    *slot = R;
}

static void race_free(race * R) {
    for (uint32_t u = 0; u < TSR_NUM_UNITS; ++u) free(R->h[u].e);
    for (uint32_t c = 0; c < R->num_ch; ++c) free(R->dram[c].v);
    free(R->dram);
    free(R->spm);
    free(R->sgen);
    free(R->smask);
    free(R->rep);
    free(R->rep_scan);
    free(R->rep_new);
    free(R->ht);
    free(R->notes);
    free(R);
}

void tsr_race_detach(tsr_sim * sim) {
    void ** slot = tsr_sim_model_slot(sim, TSR_MODEL_RACE);
    race * R = slot ? (race *)*slot : NULL;
    if (R == NULL) return;
    tsr_sim_remove_listener(sim, R->lis_id);
    tsr_sim_remove_acc_listener(sim, R->acc_id);
    if (R->n_rep) {                                      // report mode: print what was collected
        fprintf(stderr, "tsr_sim: race detector: %zu report(s)\n", R->n_rep);
        char line[768];
        for (size_t i = 0; i < R->n_rep && i < PRINT_MAX; ++i) {
            tsr_race_format(&R->rep[i], line, sizeof(line));
            fprintf(stderr, "  %s\n", line);
        }
        if (R->n_rep > PRINT_MAX) fprintf(stderr, "  ... and %zu more (tsr_sim_race_reports)\n", R->n_rep - PRINT_MAX);
    }
    *slot = NULL;
    race_free(R);
}

size_t tsr_sim_race_reports(const tsr_sim * sim, const tsr_race_report ** out) {
    const race * R = race_of(sim);
    if (out) *out = (R && R->n_rep) ? R->rep : NULL;
    return R ? R->n_rep : 0u;
}

void tsr_sim_race_clear(tsr_sim * sim) {
    race * R = race_of(sim);
    if (R == NULL) return;
    R->n_rep = 0;
    memset(R->ht, 0, R->ht_cap * sizeof(dkey));
    R->ht_n = 0;
}

int tsr_sim_race_active(const tsr_sim * sim) { return race_of(sim) != NULL; }

int tsr_sim_race_stats(const tsr_sim * sim, tsr_race_stats * out) {
    const race * R = race_of(sim);
    if (R == NULL) return -1;
    if (out) *out = R->st;
    return 0;
}

// ---------------------------------------------------------------------------
// Formatting
// ---------------------------------------------------------------------------
static int fmt_side(const tsr_race_side * s, char * b, size_t n) {
    const char * rw = s->write ? "write" : "read";
    if (s->agent < TSR_NUM_UNITS) {
        char lch[24] = "";
        if (s->launch != NO_LAUNCH) snprintf(lch, sizeof(lch), " launch %u", s->launch);
        if (s->pc != 0)
            return snprintf(b, n, "%s %s ticket %u:%u pidx %u pushed at pc 0x%08x (ra 0x%08x)%s %s @%llu",
                            unit_name(s->agent), op_name(s->opcode), tk_unit(s->ticket), tk_seq(s->ticket), s->pidx,
                            s->pc, s->ra, lch, rw, (unsigned long long)s->time);
        return snprintf(b, n, "%s %s ticket %u:%u pidx %u (test-hook push)%s %s @%llu", unit_name(s->agent),
                        op_name(s->opcode), tk_unit(s->ticket), tk_seq(s->ticket), s->pidx, lch, rw,
                        (unsigned long long)s->time);
    }
    if (s->agent == A_RV) {
        char lch[24] = "";
        if (s->launch != NO_LAUNCH) snprintf(lch, sizeof(lch), " launch %u", s->launch);
        return snprintf(b, n, "RISC-V pc 0x%08x%s %s @%llu", s->pc, lch, rw, (unsigned long long)s->time);
    }
    return snprintf(b, n, "host %s %s @%llu", s->detail == 1u ? "DRAM" : "CSR", rw, (unsigned long long)s->time);
}

int tsr_race_format(const tsr_race_report * r, char * buf, size_t n) {
    static const char * const kinds[] = { "?", "RAW", "WAR", "WAW", "SPM_STALE" };
    char where[64], a[256], b[256];
    switch (r->space) {
    case TSR_RACE_SPACE_SPM:
        snprintf(where, sizeof(where), "SPM bytes [0x%llx, 0x%llx)", (unsigned long long)r->lo, (unsigned long long)r->hi);
        break;
    case TSR_RACE_SPACE_ATT_LOCAL:
        snprintf(where, sizeof(where), "local attention [%llx, %llx)", (unsigned long long)r->lo,
                 (unsigned long long)r->hi);
        break;
    case TSR_RACE_SPACE_DRAM:
        snprintf(where, sizeof(where), "DRAM ch%u beats [0x%llx, 0x%llx)", r->ch, (unsigned long long)r->lo,
                 (unsigned long long)r->hi);
        break;
    default:
        snprintf(where, sizeof(where), "s%llu..s%llu", (unsigned long long)r->lo, (unsigned long long)(r->hi - 1u));
        break;
    }
    fmt_side(&r->first, a, sizeof(a));
    const char * kn = r->kind <= TSR_RACE_SPM_STALE ? kinds[r->kind] : "?";
    if (r->kind == TSR_RACE_SPM_STALE)
        return snprintf(buf, n, "%s %s: %s; not written since the LQ_POP of launch %u @%llu (x%u)", kn, where, a,
                        r->launch, (unsigned long long)r->since, r->count);
    fmt_side(&r->second, b, sizeof(b));
    return snprintf(buf, n, "%s %s: first %s; then %s, unordered (x%u)", kn, where, a, b, r->count);
}
