// test_sim_dbg.c — the DB simulator models (tsr_dbg.c, P1.13; docs/design/05-debug.md "DB" §14 item 1,
// §5.8 invariants G / Q / S / H on the simulator): every DBG register's access type and reset
// value, the DMI gate, stop points in push order against prefix runs, RELEASE, STOP_NOW, QSTEP
// (with DBG_STEP_N, and through the debug-halt hold with dbg_step), serial mode, tile breakpoints
// of every kind, the dispatch history, REC records and irq_fast[4], TRW records and ring rules,
// ERR_STOP, the hard hold against QABORT and SOFT_RESET, RV watchpoints, halt-on-boot,
// STOP_HALT_RV and the one-shot rule, the halt's exact time, dbg_stop and ISSUE_WAIT_MAX,
// DBG_STOP_CYCLES, and the comparability checks (the model attached and unconfigured changes
// nothing: tile programs and the tests/rv_issue firmware scenario, fixed and random schedules).
// Tile programs are race-free chains LD -> QACT -> MVM -> ST interleaved over the DMA, QACT, MVM and
// SYNC queues, pushed through the test hook with credit checks like firmware. Usage: sim_dbg
// [fw_issue.elf] (without the image the firmware comparability run is skipped).
#include "test_sim_util.h"
#include "tsr_sim_int.h"
#include "tsr_dbg.h"
#if defined(TSR_SIM_HAVE_RV) && TSR_SIM_HAVE_RV
#include "test_sim_rvhost.h"     // the W1a firmware scenario (sim_rv_issue), for the comparability run
#include "tsr_bus.h"
#include "tsr_rv.h"
#define HAVE_RV 1
#else
#define HAVE_RV 0
#endif

static const uint32_t ALL = 0xFF00u;   // ISSUE_PUSH: all eight words
static const uint32_t NONE = TSR_TICKET_NONE;

static uint32_t RD(tsr_sim * s, uint32_t off) { return tsr_sim_csr_read(s, off); }
static void WR(tsr_sim * s, uint32_t off, uint32_t v) { tsr_sim_csr_write(s, off, v); }
static uint32_t ST(tsr_sim * s) { return RD(s, TSR_CSR_DBG_STATUS); }
static int ge32(uint32_t a, uint32_t b) { return (uint32_t)(a - b) < 0x80000000u; }
// Advances n core cycles (tsr_sim_advance returns at host-visible events).
static void adv(tsr_sim * s, uint64_t n) {
    const uint64_t t_end = tsr_sim_now(s) + n;
    while (tsr_sim_now(s) < t_end) tsr_sim_advance(s, t_end - tsr_sim_now(s));
}

// The schedule of the simulators the tests create (NULL: the environment's, "fixed" by default);
// main() reruns the stop / step / serial tests under a random schedule (SW §4.4).
static const char * g_sched = NULL;

static tsr_sim * mk(void) {
    tsr_config c = cfg_u200(1);
    tsr_sim_opts o;
    memset(&o, 0, sizeof(o));
    o.sched = g_sched;
    tsr_sim * s = tsr_sim_create_opts(&c, &o);
    CHECK(s != NULL, "create");
    if (s == NULL) exit(1);
    if (tsr_sim_dbg(s) == NULL) {
        fprintf(stderr, "sim_dbg: the simulator library has no DBG model\n");
        exit(1);
    }
    return s;
}
static int have_dm(tsr_sim * s) {
    void ** slot = tsr_sim_model_slot(s, TSR_MODEL_DM);
    return slot != NULL && *slot != NULL;
}

// ---------------------------------------------------------------------------
// Event log (a second listener next to the model's)
// ---------------------------------------------------------------------------
typedef struct { uint32_t kind, unit, ticket, pidx, code, ran, rec, w0, w1, pc; uint64_t time; } ev_t;
typedef struct { ev_t * v; size_t n, cap; uint64_t hash; } evlog;

static void ev_cb(void * ctx, tsr_sim * s, const tsr_q_event * e) {
    (void)s;
    evlog * L = (evlog *)ctx;
    if (L->n == L->cap) {
        L->cap = L->cap ? 2 * L->cap : 512;
        L->v = (ev_t *)realloc(L->v, L->cap * sizeof(ev_t));
    }
    ev_t * r = &L->v[L->n++];
    memset(r, 0, sizeof(*r));
    r->kind = e->kind; r->unit = e->unit; r->ticket = e->ticket; r->pidx = e->pidx; r->code = e->code;
    r->ran = e->ran; r->rec = e->rec; r->time = e->time; r->pc = e->pc;
    if (e->w) { r->w0 = e->w[0]; r->w1 = e->w[1]; }
    const uint64_t f[10] = { e->kind, e->unit, e->ticket, e->pidx, e->code, e->ran, e->time, r->w0, e->dep[0], e->pc };
    for (int i = 0; i < 10; ++i) { L->hash ^= f[i]; L->hash *= 1099511628211ull; }
}
static void ev_init(evlog * L, tsr_sim * s) {
    memset(L, 0, sizeof(*L));
    L->hash = 1469598103934665603ull;
    CHECK(tsr_sim_add_listener(s, ev_cb, L) >= 0, "listener");
}
static const ev_t * ev_find(const evlog * L, uint32_t kind, uint32_t pidx) {
    for (size_t i = 0; i < L->n; ++i)
        if (L->v[i].kind == kind && L->v[i].pidx == pidx) return &L->v[i];
    return NULL;
}

// ---------------------------------------------------------------------------
// Race-free tile programs: chains LD -> QACT -> MVM -> ST with disjoint SPM / output regions,
// inputs and weights read-only; extra joins (NOP with deps) and FENCEs.
// ---------------------------------------------------------------------------
#define K_ 256u
#define N_ 16u
#define W_OFF 20000u
#define X_BEAT 1000u
#define Y_BEAT 500u
#define MAXOPS 512u

typedef struct { tsr_insn in; int dep[2]; uint32_t push; } pop_t;
typedef struct { pop_t v[MAXOPS]; uint32_t n, chains; } prog_t;

static uint32_t chain_line(uint32_t c) { return 100u + 64u * c; }
static uint32_t qa_lines(void) { return tsr_qa_lines(TSR_QA_A32, K_); }
static uint32_t y_line(uint32_t c) { return chain_line(c) + 16u + qa_lines(); }
static uint32_t y_ch(uint32_t c) { return 1u + c % 3u; }

static void gen_prog(prog_t * p, uint64_t seed, uint32_t chains) {
    rng_t r = { seed };
    uint32_t step[32] = { 0 }, left[32], nleft = chains;
    int last[32];
    for (uint32_t c = 0; c < chains; ++c) { left[c] = c; last[c] = -1; }
    p->n = 0;
    p->chains = chains;
    while (nleft) {
        const uint32_t li = rng_below(&r, nleft), c = left[li];
        pop_t * o = &p->v[p->n];
        memset(o, 0, sizeof(*o));
        const uint32_t base = chain_line(c);
        switch (step[c]) {
        case 0: o->in = tsr_enc_ld(0, base, X_BEAT + 16u * c, 16); break;
        case 1: o->in = tsr_enc_qact(TSR_QA_A32, base + 16u, base, K_, 1, 0); break;
        case 2: o->in = tsr_enc_mvm(TSR_FMT_Q8_0, 0, y_line(c), base + 16u, W_OFF, N_, K_, 1, 0); break;
        default: o->in = tsr_enc_st(y_ch(c), y_line(c), Y_BEAT + c, 1); break;
        }
        o->dep[0] = last[c];
        o->dep[1] = (p->n && rng_below(&r, 4) == 0) ? (int)rng_below(&r, p->n) : -1;
        o->push = ALL | (rng_below(&r, 3) == 0 ? TSR_ISSUE_PUSH_REC : 0u);
        last[c] = (int)p->n++;
        if (++step[c] == 4) left[li] = left[--nleft];
        const uint32_t x = rng_below(&r, 16);
        if (x < 2 && p->n >= 2) {                        // a join point
            pop_t * j = &p->v[p->n];
            memset(j, 0, sizeof(*j));
            j->in = tsr_enc_nop();
            j->dep[0] = (int)rng_below(&r, p->n);
            j->dep[1] = (int)rng_below(&r, p->n);
            j->push = ALL;
            p->n++;
        } else if (x == 2) {
            pop_t * f = &p->v[p->n];
            memset(f, 0, sizeof(*f));
            f->in = tsr_enc_fence();
            f->dep[0] = f->dep[1] = -1;
            f->push = ALL;
            p->n++;
        }
    }
}

static void init_mem(tsr_sim * s, uint32_t chains) {
    rng_t r = { 77 };
    const size_t row_bytes = (size_t)(K_ / tsr_fmt_block_size(TSR_FMT_Q8_0)) * tsr_fmt_ggml_block_bytes(TSR_FMT_Q8_0);
    uint8_t * blocks = (uint8_t *)malloc(row_bytes * N_);
    for (uint32_t row = 0; row < N_; ++row) gen_row_blocks(&r, TSR_FMT_Q8_0, K_, blocks + row * row_bytes);
    put_weights(s, TSR_FMT_Q8_0, N_, K_, W_OFF, blocks);
    free(blocks);
    float x[K_];
    for (uint32_t c = 0; c < chains; ++c) {
        gen_f32(&r, x, K_);
        put_f32_le(tsr_sim_dram(s, 0) + (uint64_t)(X_BEAT + 16u * c) * 64u, x, K_);
    }
}

static uint32_t credit(tsr_sim * s, uint32_t u) { return (tsr_sim_cw_read(s, TSR_CW_ISSUE_CREDIT) >> (4u * u)) & 0xFu; }
static uint32_t unit_of(const tsr_insn * in) { return tsr_op_unit(in->w[0] & 0xFFu); }

// Pushes ops from *k on while their units have credits (never waits).
static void drive(tsr_sim * s, const prog_t * p, uint32_t * k, uint32_t end, uint32_t * tk) {
    while (*k < end) {
        const pop_t * o = &p->v[*k];
        if (credit(s, unit_of(&o->in)) == 0) return;
        const uint32_t d0 = o->dep[0] >= 0 ? tk[o->dep[0]] : NONE, d1 = o->dep[1] >= 0 ? tk[o->dep[1]] : NONE;
        tk[*k] = tsr_sim_push(s, o->in.w, d0, d1, o->push);
        (*k)++;
    }
}

// Pushes ops [*k, end) and runs until they are all pushed and QIDLE (or the budget).
static int run_prog(tsr_sim * s, const prog_t * p, uint32_t * k, uint32_t end, uint32_t * tk, uint64_t budget) {
    const uint64_t t_end = tsr_sim_now(s) + budget;
    for (;;) {
        drive(s, p, k, end, tk);
        if (*k >= end && tsr_sim_run_until_idle(s, 0)) return 1;
        if (tsr_sim_now(s) >= t_end) return 0;
        tsr_sim_advance(s, 32);
    }
}

typedef struct { uint64_t spm, dram; uint32_t done[TSR_NUM_UNITS], err[TSR_NUM_UNITS]; } snap_t;

static void snap(tsr_sim * s, uint32_t chains, snap_t * o) {
    o->spm = fnv64(tsr_sim_spm(s), tsr_sim_config(s)->spm_bytes);
    o->dram = 0;
    for (uint32_t ch = 1; ch < 4; ++ch)
        o->dram = o->dram * 31u + fnv64(tsr_sim_dram(s, ch) + (uint64_t)Y_BEAT * 64u, (size_t)chains * 64u);
    for (uint32_t u = 0; u < TSR_NUM_UNITS; ++u) {
        o->done[u] = RD(s, TSR_CSR_Q_DONE(u));
        o->err[u] = RD(s, TSR_CSR_Q_ERR_CODE(u));
    }
}
static int snap_eq(const snap_t * a, const snap_t * b) {
    return a->spm == b->spm && a->dram == b->dram && memcmp(a->done, b->done, sizeof(a->done)) == 0 &&
           memcmp(a->err, b->err, sizeof(a->err)) == 0;
}

// The sequential reference: ops [0, m) alone on a fresh simulator (SW §4.7: a valid program's
// state after a push-order prefix is unique).
static void prefix_snap(const prog_t * p, uint32_t m, snap_t * o) {
    tsr_sim * s = mk();
    init_mem(s, p->chains);
    uint32_t tk[MAXOPS], k = 0;
    CHECK(run_prog(s, p, &k, m, tk, 50000000ull), "prefix %u runs", m);
    snap(s, p->chains, o);
    tsr_sim_destroy(s);
}

// Advances (pushing what credits allow) until DBG_STATUS.DRAINED, checking at every step that
// DRAINED is exactly "every entry before T completed and DBG_NEXT_PIDX >= T" (DB §5.2).
static int settle_drained(tsr_sim * s, const prog_t * p, uint32_t * k, uint32_t * tk, const evlog * L, uint64_t budget) {
    const uint64_t t_end = tsr_sim_now(s) + budget;
    for (;;) {
        drive(s, p, k, p->n, tk);
        const uint32_t st = ST(s);
        if (st & TSR_DBG_STATUS_STOP_ACTIVE) {
            const uint32_t T = RD(s, TSR_CSR_DBG_STOP_PIDX), next = RD(s, TSR_CSR_DBG_NEXT_PIDX);
            uint32_t done_before = 0;
            for (size_t i = 0; i < L->n; ++i)
                if (L->v[i].kind == TSR_QEV_DONE && !ge32(L->v[i].pidx, T)) ++done_before;
            const int expect = ge32(next, T) && done_before == T;   // pidx from 0: T entries before T
            CHECK(((st & TSR_DBG_STATUS_DRAINED) != 0) == expect, "DRAINED %u, expected %d (T %u, next %u, done %u)",
                  (st & TSR_DBG_STATUS_DRAINED) != 0, expect, T, next, done_before);
            if (st & TSR_DBG_STATUS_DRAINED) return 1;
        }
        if (tsr_sim_now(s) >= t_end) return 0;
        if (*k >= p->n && (RD(s, TSR_CSR_STATUS) & TSR_STATUS_QIDLE)) return 0;   // finished without a stop
        tsr_sim_advance(s, 16);
    }
}

// No entry at or after T started or completed (the stop holds them, DB §5.1).
static void check_held(const evlog * L, uint32_t T, size_t from, const char * what) {
    for (size_t i = from; i < L->n; ++i) {
        const ev_t * e = &L->v[i];
        if ((e->kind == TSR_QEV_START || e->kind == TSR_QEV_DONE) && ge32(e->pidx, T))
            CHECK(0, "%s: entry %u %s at or after T = %u", what, e->pidx, e->kind == TSR_QEV_START ? "started" : "completed", T);
    }
}

// ---------------------------------------------------------------------------
// 1. Registers: access types, reset values, side-effect-free reads (DB §10.1, §7.1)
// ---------------------------------------------------------------------------
static const uint32_t k_ro[] = {
    TSR_CSR_DBG_ID, TSR_CSR_DBG_STOP_PIDX, TSR_CSR_DBG_NEXT_PIDX, TSR_CSR_DBG_STEP_PIDX, TSR_CSR_DBG_MATCHQ,
    TSR_CSR_DBG_QSTATE, TSR_CSR_DBG_RV_STATE, TSR_CSR_DBG_RV_PC, TSR_CSR_DBG_RV_NPC, TSR_CSR_DBG_RV_DADDR,
    TSR_CSR_DBG_RV_EPC, TSR_CSR_DBG_RV_EADDR, TSR_CSR_DBG_RV_WPADDR, TSR_CSR_DBG_TRW_WPTR, TSR_CSR_DBG_TRW_DROPS,
    TSR_CSR_DBG_HIST_HEAD, TSR_CSR_DBG_HIST_INFO, TSR_CSR_DBG_HIST_PIDX, TSR_CSR_DBG_HIST_TICKET, TSR_CSR_DBG_HIST_DEP0,
    TSR_CSR_DBG_HIST_DEP1, TSR_CSR_DBG_HIST_TIME, TSR_CSR_DBG_HIST_PC,
};

static void test_regs(void) {
    tsr_sim * s = mk();
    const uint32_t id = RD(s, TSR_CSR_DBG_ID);
    CHECK(TSR_GET(id, TSR_DBG_ID_VERSION) == 1 && (id & TSR_DBG_ID_HIST) && (id & TSR_DBG_ID_TRW) && (id & TSR_DBG_ID_RVWP) &&
          (id & TSR_DBG_ID_REC) && TSR_GET(id, TSR_DBG_ID_NTBP) == 4 && TSR_GET(id, TSR_DBG_ID_NRVWP) == 2 &&
          TSR_GET(id, TSR_DBG_ID_LOG2_HIST) == 9, "DBG_ID 0x%08x", id);
    // CTL_ID.DM needs the DM model behind DBG's DMI window too (W1b fix; test_dmi_gate checks both ways)
    CHECK(((RD(s, TSR_CSR_CTL_ID) & TSR_CTL_ID_DM) != 0) == (have_dm(s) != 0), "CTL_ID.DM = DM and DBG attached");
    // reset values
    CHECK(RD(s, TSR_CSR_DBG_CFG) == TSR_CSR_DBG_CFG_RESET, "DBG_CFG reset 0x30");
    CHECK(RD(s, TSR_CSR_DBG_STEP_N) == 1, "DBG_STEP_N reset 1");
    CHECK(RD(s, TSR_CSR_DBG_TRW_MASK) == 0x7FF, "DBG_TRW_MASK reset 0x7FF");
    CHECK(ST(s) == 0, "DBG_STATUS 0 after create (0x%08x)", ST(s));
    const uint32_t zero_regs[] = { TSR_CSR_DBG_STOP_AT, TSR_CSR_DBG_TRW_BASE, TSR_CSR_DBG_TRW_LINES, TSR_CSR_DBG_TRW_RPTR,
                                   TSR_CSR_DBG_HIST_IDX, TSR_CSR_DBG_TBP_CTL(0), TSR_CSR_DBG_TBP_A(3), TSR_CSR_DBG_RVWP_CTL(1),
                                   TSR_CSR_DBG_RVWP_HI(0), TSR_CSR_DBG_MATCHQ, TSR_CSR_DBG_NEXT_PIDX, TSR_CSR_DBG_HIST_HEAD };
    for (size_t i = 0; i < sizeof(zero_regs) / 4; ++i) CHECK(RD(s, zero_regs[i]) == 0, "0x%03x reads 0 after create", zero_regs[i]);
    CHECK(RD(s, TSR_CSR_DBG_U_OLDEST(0)) == NONE && RD(s, TSR_CSR_DBG_U_OLDEST(7)) == NONE, "DBG_Uu_OLDEST none");
    // DBG_CTRL: write-only pulses, reads 0
    WR(s, TSR_CSR_DBG_CTRL, TSR_DBG_CTRL_HIST_CLEAR | TSR_DBG_CTRL_TRW_FLUSH);
    CHECK(RD(s, TSR_CSR_DBG_CTRL) == 0, "DBG_CTRL reads 0");
    // RW masks
    WR(s, TSR_CSR_DBG_STEP_N, 0xFFFFFFFFu);
    CHECK(RD(s, TSR_CSR_DBG_STEP_N) == 0xFFFF, "DBG_STEP_N [15:0]");
    WR(s, TSR_CSR_DBG_STEP_N, 1);
    for (uint32_t i = 0; i < 4; ++i) {
        WR(s, TSR_CSR_DBG_TBP_CTL(i), 0xFFFFFFFEu);           // all but EN: no breakpoint armed
        WR(s, TSR_CSR_DBG_TBP_A(i), 0x12345670u + i);
        WR(s, TSR_CSR_DBG_TBP_B(i), 0xFEDCBA90u + i);
        CHECK(RD(s, TSR_CSR_DBG_TBP_CTL(i)) == 0xFFFF01FEu, "TBP%u_CTL mask (0x%08x)", i, RD(s, TSR_CSR_DBG_TBP_CTL(i)));
        CHECK(RD(s, TSR_CSR_DBG_TBP_A(i)) == 0x12345670u + i && RD(s, TSR_CSR_DBG_TBP_B(i)) == 0xFEDCBA90u + i, "TBP%u A / B", i);
        WR(s, TSR_CSR_DBG_TBP_CTL(i), 0);
    }
    for (uint32_t j = 0; j < 2; ++j) {
        WR(s, TSR_CSR_DBG_RVWP_CTL(j), 0xFFFFFFFEu);
        WR(s, TSR_CSR_DBG_RVWP_LO(j), 0x100u + j);
        WR(s, TSR_CSR_DBG_RVWP_HI(j), 0x200u + j);
        CHECK(RD(s, TSR_CSR_DBG_RVWP_CTL(j)) == 0xE, "RVWP%u_CTL bits 0..3", j);
        CHECK(RD(s, TSR_CSR_DBG_RVWP_LO(j)) == 0x100u + j && RD(s, TSR_CSR_DBG_RVWP_HI(j)) == 0x200u + j, "RVWP%u LO / HI", j);
        WR(s, TSR_CSR_DBG_RVWP_CTL(j), 0);
    }
    WR(s, TSR_CSR_DBG_TRW_BASE, 0xFFFFFFFFu);
    CHECK(RD(s, TSR_CSR_DBG_TRW_BASE) == 0xFFFFFFF0u, "TRW_BASE beat [3:0] read 0");
    WR(s, TSR_CSR_DBG_TRW_MASK, 0xFFFFFFFFu);
    CHECK(RD(s, TSR_CSR_DBG_TRW_MASK) == 0x7FF, "TRW_MASK [10:0]");
    WR(s, TSR_CSR_DBG_HIST_IDX, 0xDEADBEEFu);
    CHECK(RD(s, TSR_CSR_DBG_HIST_IDX) == 0xDEADBEEFu, "HIST_IDX RW");
    WR(s, TSR_CSR_DBG_CFG, 0xFFFFFFFFu);
    CHECK(RD(s, TSR_CSR_DBG_CFG) == 0xFFF, "DBG_CFG [11:0]");
    WR(s, TSR_CSR_DBG_CFG, TSR_CSR_DBG_CFG_RESET);
    // TRW_LINES: power of two >= 16, else ignored with PTR_ERR; TRW_RPTR beyond WPTR: PTR_ERR
    WR(s, TSR_CSR_DBG_TRW_LINES, 15);
    CHECK(RD(s, TSR_CSR_DBG_TRW_LINES) == 0 && (ST(s) & TSR_DBG_STATUS_PTR_ERR), "TRW_LINES 15 ignored, PTR_ERR");
    WR(s, TSR_CSR_DBG_STATUS, 0);
    CHECK(ST(s) & TSR_DBG_STATUS_PTR_ERR, "W1C: writing 0 keeps a sticky bit");
    WR(s, TSR_CSR_DBG_STATUS, TSR_DBG_STATUS_PTR_ERR);
    CHECK(!(ST(s) & TSR_DBG_STATUS_PTR_ERR), "W1C clears PTR_ERR");
    WR(s, TSR_CSR_DBG_TRW_LINES, 48);
    CHECK(RD(s, TSR_CSR_DBG_TRW_LINES) == 0 && (ST(s) & TSR_DBG_STATUS_PTR_ERR), "TRW_LINES 48 ignored");
    WR(s, TSR_CSR_DBG_TRW_LINES, 64);
    CHECK(RD(s, TSR_CSR_DBG_TRW_LINES) == 64, "TRW_LINES 64");
    WR(s, TSR_CSR_DBG_STATUS, 0xFFFFFFFFu);
    CHECK(ST(s) == 0, "W1C of every bit clears the sticky bits and leaves the levels");
    WR(s, TSR_CSR_DBG_TRW_RPTR, 5);
    CHECK(RD(s, TSR_CSR_DBG_TRW_RPTR) == 0 && (ST(s) & TSR_DBG_STATUS_PTR_ERR), "TRW_RPTR beyond WPTR ignored");
    WR(s, TSR_CSR_DBG_STATUS, TSR_DBG_STATUS_PTR_ERR);
    // read-only registers ignore writes
    uint32_t before[sizeof(k_ro) / 4];
    for (size_t i = 0; i < sizeof(k_ro) / 4; ++i) before[i] = RD(s, k_ro[i]);
    for (size_t i = 0; i < sizeof(k_ro) / 4; ++i) WR(s, k_ro[i], 0xA5A5A5A5u);
    for (size_t i = 0; i < sizeof(k_ro) / 4; ++i) CHECK(RD(s, k_ro[i]) == before[i], "RO 0x%03x ignores writes", k_ro[i]);
    for (uint32_t u = 0; u < 8; ++u) {
        WR(s, TSR_CSR_DBG_U_STATE(u), 0xFFFFFFFFu);
        WR(s, TSR_CSR_DBG_U_OPIDX(u), 0xFFFFFFFFu);
        CHECK(RD(s, TSR_CSR_DBG_U_STATE(u)) == 0 && RD(s, TSR_CSR_DBG_U_OPIDX(u)) == 0 && RD(s, TSR_CSR_DBG_U_AGE(u)) == 0,
              "unit probe %u RO", u);
    }
    for (uint32_t i = 0; i < 4; ++i) {
        WR(s, TSR_CSR_DBG_RDOUT(i), 0xFFFFFFFFu);
        CHECK(RD(s, TSR_CSR_DBG_RDOUT(i)) == 0 && RD(s, TSR_CSR_DBG_WROUT(i)) == 0, "RDOUT / WROUT %u", i);
    }
    // unlisted offsets read 0 and ignore writes
    const uint32_t unl[] = { 0x830, 0x838, 0x83C, 0x85C, 0x8F0, 0x8E0, 0x8FC, 0x998, 0x9BC, 0xA24, 0xA3C, 0xA60, 0xAFC };
    for (size_t i = 0; i < sizeof(unl) / 4; ++i) {
        WR(s, unl[i], 0xFFFFFFFFu);
        CHECK(RD(s, unl[i]) == 0, "unlisted 0x%03x reads 0", unl[i]);
    }
    // reads have no side effects; the core window reads the same (DMI excepted)
    const uint64_t t0 = tsr_sim_now(s);
    for (uint32_t off = 0x800; off < 0xB00; off += 4) {
        if (off >= 0x9C0 && off < 0xA00) continue;           // SREG: the VU's
        const uint32_t a = RD(s, off), b = RD(s, off), c = tsr_sim_corewin_read(s, off);
        if (off == TSR_CSR_DBG_MTIME_LO || off == TSR_CSR_DBG_MTIME_HI) continue;
        CHECK(a == b && a == c, "0x%03x: repeated / core-window read differs (0x%x 0x%x 0x%x)", off, a, b, c);
    }
    CHECK(tsr_sim_now(s) == t0, "reads take no simulated time (DB §7.1)");
    CHECK(RD(s, TSR_CSR_DBG_MTIME_LO) == (uint32_t)t0 && RD(s, TSR_CSR_DBG_MTIME_HI) == 0, "DBG_MTIME");
    tsr_sim_advance(s, 1234);
    CHECK(RD(s, TSR_CSR_DBG_MTIME_LO) == (uint32_t)tsr_sim_now(s), "DBG_MTIME follows MTIME");
    // HITS registers: a write clears (checked with a count below)
    WR(s, TSR_CSR_DBG_TBP_CTL(1), TSR_DBG_TBP_CTL_EN | TSR_PUT(TSR_DBG_TBP_CTL_KIND, TSR_DBG_TBP_CTL_KIND_OPCODE) |
                                     TSR_DBG_TBP_CTL_COUNT_ONLY);
    WR(s, TSR_CSR_DBG_TBP_A(1), TSR_OP_NOP | (0xFFu << 8));
    WR(s, TSR_CSR_DBG_TBP_B(1), 0);
    for (int i = 0; i < 3; ++i) tsr_sim_push(s, tsr_enc_nop().w, NONE, NONE, ALL);
    CHECK(tsr_sim_run_until_idle(s, 100000), "idle");
    CHECK(RD(s, TSR_CSR_DBG_TBP_HITS(1)) == 3, "COUNT_ONLY TBP counts 3 NOPs (%u)", RD(s, TSR_CSR_DBG_TBP_HITS(1)));
    CHECK(!(ST(s) & TSR_DBG_STATUS_STOP_ACTIVE), "COUNT_ONLY never stops");
    WR(s, TSR_CSR_DBG_TBP_HITS(1), 0x1234);
    CHECK(RD(s, TSR_CSR_DBG_TBP_HITS(1)) == 0, "TBP HITS: a write clears");
    // RV sample and RVWP_APPLY: complete at once (with clk_rv), never busy
    WR(s, TSR_CSR_DBG_CTRL, TSR_DBG_CTRL_RV_SAMPLE);
    WR(s, TSR_CSR_DBG_CTRL, TSR_DBG_CTRL_RVWP_APPLY);
    const uint32_t rs = RD(s, TSR_CSR_DBG_RV_STATE);
    CHECK(!(ST(s) & TSR_DBG_STATUS_RVWP_BUSY), "RVWP_BUSY 0 after the apply");
    if (tsr_sim_rv_agent(s) != NULL) {
        CHECK((rs & TSR_DBG_RV_STATE_SAMPLE_VALID) && TSR_GET(rs, TSR_DBG_RV_STATE_SAMPLE_SEQ) == 1 &&
              (rs & TSR_DBG_RV_STATE_IN_RESET), "RV_SAMPLE: valid, SAMPLE_SEQ 1, in reset (0x%08x)", rs);
        CHECK(RD(s, TSR_CSR_DBG_RV_PC) == 0x80u, "a hart that never ran: its reset pc");
        CHECK(!(ST(s) & (TSR_DBG_STATUS_RV_SAMPLE_TIMEOUT | TSR_DBG_STATUS_RVWP_TIMEOUT)), "no timeouts with clk_rv");
    } else {
        CHECK(!(rs & TSR_DBG_RV_STATE_SAMPLE_VALID) && (ST(s) & TSR_DBG_STATUS_RV_SAMPLE_TIMEOUT) &&
              (ST(s) & TSR_DBG_STATUS_RVWP_TIMEOUT), "no clk_rv domain: both handshakes time out");
    }
    // STOP_AT read back, S_VALID
    WR(s, TSR_CSR_DBG_STOP_AT, 77);
    CHECK(RD(s, TSR_CSR_DBG_STOP_AT) == 77 && (ST(s) & TSR_DBG_STATUS_S_VALID) && (ST(s) & TSR_DBG_STATUS_STOP_ACTIVE) &&
          RD(s, TSR_CSR_DBG_STOP_PIDX) == 77 && TSR_GET(ST(s), TSR_DBG_STATUS_CAUSE) == TSR_DBG_CAUSE_NOW, "STOP_AT 77");
    WR(s, TSR_CSR_DBG_CTRL, TSR_DBG_CTRL_RELEASE | TSR_DBG_CTRL_STOP_NOW);   // RELEASE wins over STOP_NOW
    CHECK(!(ST(s) & (TSR_DBG_STATUS_S_VALID | TSR_DBG_STATUS_STOP_ACTIVE)), "RELEASE > STOP_NOW");
    WR(s, TSR_CSR_DBG_CTRL, TSR_DBG_CTRL_STOP_NOW);
    CHECK((ST(s) & TSR_DBG_STATUS_STOP_ACTIVE) && RD(s, TSR_CSR_DBG_STOP_PIDX) == RD(s, TSR_CSR_DBG_NEXT_PIDX), "STOP_NOW: S = NEXT_PIDX");
    WR(s, TSR_CSR_DBG_STOP_AT, 1000);
    WR(s, TSR_CSR_DBG_CTRL, TSR_DBG_CTRL_STOP_NOW);                         // an earlier valid S is kept
    CHECK(RD(s, TSR_CSR_DBG_STOP_AT) == RD(s, TSR_CSR_DBG_NEXT_PIDX), "STOP_NOW keeps the earlier S");
    WR(s, TSR_CSR_DBG_CTRL, TSR_DBG_CTRL_RELEASE);
    // An S below NEXT_PIDX is kept too (DB §5.2: S := NEXT_PIDX only without an earlier valid S):
    // 6 pushes held by HOST_HOLD, STOP_AT 2, then STOP_NOW.
    {
        tsr_sim * h = mk();
        WR(h, TSR_CSR_CTRL, TSR_CTRL_HOST_HOLD);
        for (uint32_t i = 0; i < 6u; ++i) {
            const tsr_insn in = tsr_enc_ld(0, 200u + i, 64u * i, 1);
            tsr_sim_push(h, in.w, NONE, NONE, ALL);
        }
        adv(h, 100);
        const uint32_t np = RD(h, TSR_CSR_DBG_NEXT_PIDX);
        WR(h, TSR_CSR_DBG_STOP_AT, 2);
        WR(h, TSR_CSR_DBG_CTRL, TSR_DBG_CTRL_STOP_NOW);
        CHECK(np == 6u && RD(h, TSR_CSR_DBG_STOP_AT) == 2u && RD(h, TSR_CSR_DBG_STOP_PIDX) == 2u,
              "STOP_NOW keeps an earlier S below NEXT_PIDX (next %u, S %u, T %u)", np, RD(h, TSR_CSR_DBG_STOP_AT),
              RD(h, TSR_CSR_DBG_STOP_PIDX));
        WR(h, TSR_CSR_DBG_CTRL, TSR_DBG_CTRL_RELEASE);
        WR(h, TSR_CSR_CTRL, 0);
        CHECK(tsr_sim_run_until_idle(h, 1000000), "released: idle");
        tsr_sim_destroy(h);
    }
    // DBG_FLAGS (control window 0x400) and the reserved DB window
    CHECK(tsr_sim_cw_read(s, TSR_CW_DBG_FLAGS) == 0, "DBG_FLAGS 0");
    WR(s, TSR_CSR_DBG_CFG, TSR_CSR_DBG_CFG_RESET | TSR_DBG_CFG_SERIAL | TSR_DBG_CFG_DMI_EN | TSR_DBG_CFG_TRW_EN |
                              TSR_DBG_CFG_HALT_ON_BOOT);
    WR(s, TSR_CSR_DBG_STOP_AT, 5000);
    const uint32_t fl = tsr_sim_cw_read(s, TSR_CW_DBG_FLAGS);
    CHECK(fl == (TSR_DBG_FLAGS_DBG_STOP | TSR_DBG_FLAGS_SERIAL | TSR_DBG_FLAGS_TRW_EN | TSR_DBG_FLAGS_DMI_EN |
                 TSR_DBG_FLAGS_HALT_ON_BOOT), "DBG_FLAGS 0x%x", fl);
    tsr_sim_cw_write(s, TSR_CW_DBG_FLAGS, 0);
    CHECK(tsr_sim_cw_read(s, TSR_CW_DBG_FLAGS) == fl && tsr_sim_cw_read(s, 0x404) == 0 && tsr_sim_cw_read(s, 0x7FC) == 0,
          "DBG_FLAGS RO, reserved words read 0");
    tsr_sim_destroy(s);
}

// ---------------------------------------------------------------------------
// 2. DMI gate (DB §3.2)
// ---------------------------------------------------------------------------
static void test_dmi_gate(void) {
    tsr_sim * s = mk();
    const int dm = have_dm(s);
    // CTL_ID.DM (EX §5.1): the host can reach a debug module (the DM model behind this DMI window).
    CHECK(((RD(s, TSR_CSR_CTL_ID) & TSR_CTL_ID_DM) != 0) == (dm != 0), "CTL_ID.DM %d with the DM model %d",
          (RD(s, TSR_CSR_CTL_ID) & TSR_CTL_ID_DM) != 0, dm);
    CHECK(RD(s, TSR_CSR_DMI(TSR_DMI_DMSTATUS)) == 0, "DMI_EN 0: dmstatus reads 0");
    WR(s, TSR_CSR_DMI(TSR_DMI_DMCONTROL), 0x00000001u);           // dmactive: dropped
    CHECK(ST(s) & TSR_DBG_STATUS_DMI_BLOCKED, "a write with DMI_EN 0 sets DMI_BLOCKED");
    WR(s, TSR_CSR_DBG_STATUS, TSR_DBG_STATUS_DMI_BLOCKED);
    WR(s, TSR_CSR_DBG_CFG, TSR_CSR_DBG_CFG_RESET | TSR_DBG_CFG_DMI_EN);
    CHECK(!(ST(s) & TSR_DBG_STATUS_DMI_BLOCKED), "cleared");
    if (dm) {
        CHECK((RD(s, TSR_CSR_DMI(TSR_DMI_DMCONTROL)) & 1u) == 0, "the blocked write did not reach the DM");
        CHECK((RD(s, TSR_CSR_DMI(TSR_DMI_DMSTATUS)) & 0xFu) == 2, "DMI_EN 1: dmstatus.version 2");
        WR(s, TSR_CSR_DMI(TSR_DMI_DMCONTROL), 0x00000001u);
        CHECK((RD(s, TSR_CSR_DMI(TSR_DMI_DMCONTROL)) & 1u) == 1, "dmactive written through the window");
        CHECK(RD(s, TSR_CSR_DBG_RV_STATE) & TSR_DBG_RV_STATE_DMACTIVE, "DBG_RV_STATE.dmactive");
        CHECK(RD(s, TSR_CSR_DMI(TSR_DMI_HARTINFO)) == 0x00212380u, "hartinfo");
        // SBA (DB §3.6): an SPM write goes through; a control-window write is refused (SBA_REFUSED)
        WR(s, TSR_CSR_DMI(TSR_DMI_SBCS), 0x00457000u);
        WR(s, TSR_CSR_DMI(TSR_DMI_SBADDRESS0), 0x40000100u);
        WR(s, TSR_CSR_DMI(TSR_DMI_SBDATA0), 0x12345678u);
        CHECK(!(ST(s) & TSR_DBG_STATUS_SBA_REFUSED) && get_u32_le(tsr_sim_spm(s) + 0x100) == 0x12345678u, "SBA write to SPM");
        WR(s, TSR_CSR_DMI(TSR_DMI_SBADDRESS0), 0x20000028u);              // ISSUE_PUSH (D15)
        WR(s, TSR_CSR_DMI(TSR_DMI_SBDATA0), 0xFF00u);
        CHECK(ST(s) & TSR_DBG_STATUS_SBA_REFUSED, "SBA write to the control window refused: SBA_REFUSED");
        CHECK(tsr_sim_cw_read(s, TSR_CW_ISSUE_PIDX) == 0, "nothing pushed");
        // A refused read through the window (the DM range): sberror 2, sbaddress0 kept (no
        // autoincrement), SBA_REFUSED again after its W1C.
        WR(s, TSR_CSR_DBG_STATUS, TSR_DBG_STATUS_SBA_REFUSED);
        CHECK(!(ST(s) & TSR_DBG_STATUS_SBA_REFUSED), "SBA_REFUSED W1C");
        WR(s, TSR_CSR_DMI(TSR_DMI_SBCS), 0x00557000u);   // 32-bit, autoincrement, read on address / data
        WR(s, TSR_CSR_DMI(TSR_DMI_SBADDRESS0), TSR_RV_DM_BASE + 0x100u);
        CHECK(((RD(s, TSR_CSR_DMI(TSR_DMI_SBCS)) >> 12) & 7u) == 2u &&
              RD(s, TSR_CSR_DMI(TSR_DMI_SBADDRESS0)) == TSR_RV_DM_BASE + 0x100u && (ST(s) & TSR_DBG_STATUS_SBA_REFUSED),
              "refused SBA read: sberror 2, sbaddress0 unchanged, SBA_REFUSED");
    } else {
        CHECK(RD(s, TSR_CSR_DMI(TSR_DMI_DMSTATUS)) == 0, "no DM model: the window reads 0");
    }
    CHECK(tsr_sim_corewin_read(s, TSR_CSR_DMI(TSR_DMI_DMSTATUS)) == 0, "core window: DMI reads 0 (CC §5.3)");
    CHECK(!(ST(s) & TSR_DBG_STATUS_DMI_BLOCKED), "no DMI_BLOCKED with DMI_EN 1");
    tsr_sim_destroy(s);
}

// ---------------------------------------------------------------------------
// 3-4. Stop points, RELEASE, QSTEP (DB §5.2; invariants G and Q)
// ---------------------------------------------------------------------------
static void full_snap(const prog_t * p, snap_t * o, uint64_t * t_done) {
    tsr_sim * s = mk();
    init_mem(s, p->chains);
    uint32_t tk[MAXOPS], k = 0;
    CHECK(run_prog(s, p, &k, p->n, tk, 50000000ull), "full run");
    snap(s, p->chains, o);
    if (t_done) *t_done = tsr_sim_now(s);
    for (uint32_t u = 0; u < TSR_NUM_UNITS; ++u) CHECK(o->err[u] == 0, "the program has no errors (unit %u)", u);
    tsr_sim_destroy(s);
}

static void test_stop_at(const prog_t * p, uint32_t T, const snap_t * full) {
    tsr_sim * s = mk();
    init_mem(s, p->chains);
    evlog L;
    ev_init(&L, s);
    WR(s, TSR_CSR_DBG_STOP_AT, T);
    uint32_t tk[MAXOPS], k = 0;
    CHECK(settle_drained(s, p, &k, tk, &L, 10000000ull), "STOP_AT %u: DRAINED", T);
    check_held(&L, T, 0, "STOP_AT");
    // DRAINED rose with the last completion before T (tsr_sim_advance returned at it)
    uint64_t last = 0;
    for (size_t i = 0; i < L.n; ++i)
        if (L.v[i].kind == TSR_QEV_DONE && !ge32(L.v[i].pidx, T) && L.v[i].time > last) last = L.v[i].time;
    CHECK(T == 0 || tsr_sim_now(s) <= last + 16, "DRAINED seen at %llu, last completion before T at %llu",
          (unsigned long long)tsr_sim_now(s), (unsigned long long)last);
    const uint32_t st = ST(s);
    CHECK((st & TSR_DBG_STATUS_STOP_ACTIVE) && (st & TSR_DBG_STATUS_S_VALID) &&
          TSR_GET(st, TSR_DBG_STATUS_CAUSE) == TSR_DBG_CAUSE_NOW && RD(s, TSR_CSR_DBG_STOP_PIDX) == T,
          "STOP_AT %u: status 0x%08x", T, st);
    if (RD(s, TSR_CSR_DBG_NEXT_PIDX) != T) CHECK(st & TSR_DBG_STATUS_DB_HELD, "a head held by DB");
    // invariant G: the state equals the sequential prefix [0, T)
    snap_t got, ref;
    snap(s, p->chains, &got);
    prefix_snap(p, T, &ref);
    CHECK(snap_eq(&got, &ref), "STOP_AT %u: state equals the prefix run", T);
    // stays stopped
    const size_t n0 = L.n;
    for (int i = 0; i < 20; ++i) { drive(s, p, &k, p->n, tk); tsr_sim_advance(s, 5000); }
    check_held(&L, T, n0, "STOP_AT later");
    // RELEASE: the run completes with the unstopped result
    WR(s, TSR_CSR_DBG_CTRL, TSR_DBG_CTRL_RELEASE);
    CHECK(!(ST(s) & TSR_DBG_STATUS_STOP_ACTIVE), "released");
    CHECK(run_prog(s, p, &k, p->n, tk, 50000000ull), "run after RELEASE");
    snap(s, p->chains, &got);
    CHECK(snap_eq(&got, full), "STOP_AT %u + RELEASE: final state equals the unstopped run", T);
    tsr_sim_destroy(s);
    free(L.v);
}

static void test_qstep(const prog_t * p, uint32_t T) {
    tsr_sim * s = mk();
    init_mem(s, p->chains);
    evlog L;
    ev_init(&L, s);
    WR(s, TSR_CSR_DBG_STOP_AT, T);
    uint32_t tk[MAXOPS], k = 0;
    CHECK(settle_drained(s, p, &k, tk, &L, 10000000ull), "QSTEP: DRAINED at %u", T);
    for (uint32_t j = 0; j < 6; ++j) {
        const size_t n0 = L.n;
        WR(s, TSR_CSR_DBG_CTRL, TSR_DBG_CTRL_QSTEP);
        CHECK(ST(s) & TSR_DBG_STATUS_STEP_BUSY, "STEP_BUSY after QSTEP");
        CHECK(RD(s, TSR_CSR_DBG_STEP_PIDX) == T + j && RD(s, TSR_CSR_DBG_STOP_PIDX) == T + j + 1,
              "QSTEP %u: X = %u, T = %u", j, RD(s, TSR_CSR_DBG_STEP_PIDX), RD(s, TSR_CSR_DBG_STOP_PIDX));
        CHECK(settle_drained(s, p, &k, tk, &L, 10000000ull), "QSTEP %u: DRAINED again", j);
        CHECK(!(ST(s) & TSR_DBG_STATUS_STEP_BUSY) && TSR_GET(ST(s), TSR_DBG_STATUS_CAUSE) == TSR_DBG_CAUSE_STEP,
              "QSTEP %u: STEP_BUSY 0, cause STEP", j);
        uint32_t started = 0, done = 0;
        for (size_t i = n0; i < L.n; ++i) {
            if (L.v[i].kind == TSR_QEV_START) { ++started; CHECK(L.v[i].pidx == T + j, "QSTEP %u started %u", j, L.v[i].pidx); }
            if (L.v[i].kind == TSR_QEV_DONE) { ++done; CHECK(L.v[i].pidx == T + j, "QSTEP %u completed %u", j, L.v[i].pidx); }
        }
        CHECK(done == 1 && started <= 1, "QSTEP %u: exactly one entry (%u started, %u done)", j, started, done);
        snap_t got, ref;
        snap(s, p->chains, &got);
        prefix_snap(p, T + j + 1, &ref);
        CHECK(snap_eq(&got, &ref), "invariant Q: QSTEP %u equals the prefix [0, %u)", j, T + j + 1);
    }
    // DBG_STEP_N = 3: one QSTEP moves T by 3
    const uint32_t T2 = RD(s, TSR_CSR_DBG_STOP_PIDX);
    WR(s, TSR_CSR_DBG_STEP_N, 3);
    WR(s, TSR_CSR_DBG_CTRL, TSR_DBG_CTRL_QSTEP);
    for (int i = 0; i < 2000 && (ST(s) & TSR_DBG_STATUS_STEP_BUSY); ++i) { drive(s, p, &k, p->n, tk); tsr_sim_advance(s, 16); }
    CHECK(settle_drained(s, p, &k, tk, &L, 10000000ull), "STEP_N 3: DRAINED");
    CHECK(RD(s, TSR_CSR_DBG_STOP_PIDX) == T2 + 3 && RD(s, TSR_CSR_DBG_STEP_PIDX) == T2 + 2, "STEP_N 3: T %u -> %u",
          T2, RD(s, TSR_CSR_DBG_STOP_PIDX));
    snap_t got, ref;
    snap(s, p->chains, &got);
    prefix_snap(p, T2 + 3, &ref);
    CHECK(snap_eq(&got, &ref), "STEP_N 3 equals the prefix [0, %u)", T2 + 3);
    free(L.v);
    tsr_sim_destroy(s);
}

// QSTEP without a stop acts as STOP_NOW; MATCHQ_CLEAR.
static void test_qstep_as_stop_now(void) {
    tsr_sim * s = mk();
    for (int i = 0; i < 3; ++i) tsr_sim_push(s, tsr_enc_nop().w, NONE, NONE, ALL);
    CHECK(tsr_sim_run_until_idle(s, 100000), "idle");
    WR(s, TSR_CSR_DBG_CTRL, TSR_DBG_CTRL_QSTEP);
    CHECK((ST(s) & TSR_DBG_STATUS_STOP_ACTIVE) && RD(s, TSR_CSR_DBG_STOP_PIDX) == 3 &&
          TSR_GET(ST(s), TSR_DBG_STATUS_CAUSE) == TSR_DBG_CAUSE_NOW && !(ST(s) & TSR_DBG_STATUS_STEP_BUSY),
          "QSTEP without a stop = STOP_NOW");
    CHECK(ST(s) & TSR_DBG_STATUS_DRAINED, "drained at once");
    WR(s, TSR_CSR_DBG_CTRL, TSR_DBG_CTRL_RELEASE);
    // Push indices compare modularly (DB §5.1): T = 2^32 - 16 lies 19 entries before NEXT_PIDX 3,
    // so it holds every new entry and is drained at once.
    WR(s, TSR_CSR_DBG_STOP_AT, 0xFFFFFFF0u);
    tsr_sim_push(s, tsr_enc_nop().w, NONE, NONE, ALL);
    adv(s, 1000);
    CHECK(RD(s, TSR_CSR_Q_DONE(TSR_UNIT_SYNC)) == 3 && (ST(s) & TSR_DBG_STATUS_DRAINED) && (ST(s) & TSR_DBG_STATUS_DB_HELD),
          "T across the wrap holds pidx 3");
    WR(s, TSR_CSR_DBG_STOP_AT, 4);
    adv(s, 1000);
    CHECK(RD(s, TSR_CSR_Q_DONE(TSR_UNIT_SYNC)) == 4 && (ST(s) & TSR_DBG_STATUS_DRAINED), "T = 4 lets pidx 3 through");
    // The hard hold (DB-9): a held poisoned entry neither completes nor sets the error state
    tsr_sim_push(s, tsr_enc_qact(0, 0, 0, 33, 1, 0).w, NONE, NONE, ALL);   // pidx 4: K_RANGE
    adv(s, 1000);
    CHECK(RD(s, TSR_CSR_Q_DONE(TSR_UNIT_QACT)) == 0 && RD(s, TSR_CSR_Q_ERR_CODE(TSR_UNIT_QACT)) == 0, "poisoned entry held");
    WR(s, TSR_CSR_DBG_CTRL, TSR_DBG_CTRL_RELEASE);
    adv(s, 1000);
    CHECK(RD(s, TSR_CSR_Q_DONE(TSR_UNIT_QACT)) == 1 && RD(s, TSR_CSR_Q_ERR_CODE(TSR_UNIT_QACT)) == TSR_ERR_K_RANGE,
          "released: it completes with its code");
    tsr_sim_destroy(s);
}

// Detach and attach again while entries are in flight: they are not shadowed (no records), later
// ones are, nothing breaks.
static void test_reattach(void) {
    tsr_sim * s = mk();
    for (int i = 0; i < 3; ++i) tsr_sim_push(s, tsr_enc_ld(0, 100 + 10 * i, 0, 8).w, NONE, NONE, ALL | TSR_ISSUE_PUSH_REC);
    adv(s, 30);
    tsr_dbg_detach(s);
    CHECK(tsr_sim_dbg(s) == NULL && RD(s, TSR_CSR_DBG_ID) == 0, "detached: DBG_ID 0");
    tsr_dbg_attach(s);
    CHECK(tsr_sim_dbg(s) != NULL && RD(s, TSR_CSR_DBG_NEXT_PIDX) == 3, "attached again");
    tsr_sim_push(s, tsr_enc_nop().w, NONE, NONE, ALL | TSR_ISSUE_PUSH_REC);
    CHECK(tsr_sim_run_until_idle(s, 100000), "idle");
    CHECK(tsr_sim_cw_read(s, TSR_CW_REC_COUNT) == 1, "only the entry enqueued after the attach has a record");
    tsr_sim_destroy(s);
}

// ---------------------------------------------------------------------------
// 5. Serial mode (DB §5.4; invariant S)
// ---------------------------------------------------------------------------
static void test_serial(const prog_t * p, const snap_t * full, uint64_t t_full) {
    tsr_sim * s = mk();
    init_mem(s, p->chains);
    evlog L;
    ev_init(&L, s);
    WR(s, TSR_CSR_DBG_CFG, TSR_CSR_DBG_CFG_RESET | TSR_DBG_CFG_SERIAL);
    uint32_t tk[MAXOPS], k = 0;
    CHECK(run_prog(s, p, &k, p->n, tk, 50000000ull), "serial run");
    snap_t got;
    snap(s, p->chains, &got);
    CHECK(snap_eq(&got, full), "invariant S: serial final state equals normal mode");
    uint32_t next_done = 0, outstanding = 0, max_out = 0;
    for (size_t i = 0; i < L.n; ++i) {
        const ev_t * e = &L.v[i];
        if (e->kind == TSR_QEV_START) { ++outstanding; if (outstanding > max_out) max_out = outstanding; }
        if (e->kind == TSR_QEV_DONE) {
            CHECK(e->pidx == next_done, "serial: completion %u in push order (expected %u)", e->pidx, next_done);
            next_done = e->pidx + 1;
            if (e->ran) --outstanding;
        }
    }
    CHECK(next_done == p->n && max_out == 1, "serial: every entry, one running at a time (max %u)", max_out);
    if (g_sched == NULL) CHECK(tsr_sim_now(s) >= t_full, "serial mode is not faster (%llu vs %llu)", (unsigned long long)tsr_sim_now(s),
          (unsigned long long)t_full);
    CHECK(tsr_sim_cw_read(s, TSR_CW_DBG_FLAGS) == TSR_DBG_FLAGS_SERIAL, "DBG_FLAGS.SERIAL without a stop");
    free(L.v);
    tsr_sim_destroy(s);
}

// ---------------------------------------------------------------------------
// 6. Tile breakpoints (DB §5.3)
// ---------------------------------------------------------------------------
static uint32_t find_op(const prog_t * p, uint32_t opcode, uint32_t nth) {
    for (uint32_t i = 0; i < p->n; ++i)
        if ((p->v[i].in.w[0] & 0xFFu) == opcode && nth-- == 0) return i;
    return UINT32_MAX;
}
static uint32_t find_chain_op(const prog_t * p, uint32_t opcode, uint32_t line) {   // MVM / ST of the chain with y at line
    for (uint32_t i = 0; i < p->n; ++i) {
        const uint32_t * w = p->v[i].in.w;
        if ((w[0] & 0xFFu) == opcode && w[1] == line) return i;
    }
    return UINT32_MAX;
}

// Runs with TBP 0 configured; returns T of the first stop after DRAINED (or UINT32_MAX).
static uint32_t tbp_run(const prog_t * p, uint32_t ctl, uint32_t a, uint32_t b, uint32_t expect_T, const char * what) {
    tsr_sim * s = mk();
    init_mem(s, p->chains);
    evlog L;
    ev_init(&L, s);
    WR(s, TSR_CSR_DBG_TBP_A(0), a);
    WR(s, TSR_CSR_DBG_TBP_B(0), b);
    WR(s, TSR_CSR_DBG_TBP_CTL(0), ctl | TSR_DBG_TBP_CTL_EN);
    uint32_t tk[MAXOPS], k = 0;
    uint32_t T = UINT32_MAX;
    if (settle_drained(s, p, &k, tk, &L, 10000000ull)) {
        T = RD(s, TSR_CSR_DBG_STOP_PIDX);
        const uint32_t st = ST(s);
        CHECK(T == expect_T, "%s: stopped at %u, expected %u", what, T, expect_T);
        CHECK(TSR_GET(st, TSR_DBG_STATUS_CAUSE) == TSR_DBG_CAUSE_BP && (TSR_GET(st, TSR_DBG_STATUS_BP_HIT) & 1u) &&
              !(st & TSR_DBG_STATUS_S_VALID), "%s: cause BP, BP_HIT TBP0 (0x%08x)", what, st);
        CHECK(RD(s, TSR_CSR_DBG_TBP_HITS(0)) >= 1 && RD(s, TSR_CSR_DBG_MATCHQ) >= 1, "%s: HITS, MATCHQ", what);
        check_held(&L, T, 0, what);
        snap_t got, ref;
        snap(s, p->chains, &got);
        prefix_snap(p, T, &ref);
        CHECK(snap_eq(&got, &ref), "%s: state equals the prefix [0, %u)", what, T);
        // the matched entry's history record
        const uint32_t hit_pidx = (ctl & TSR_DBG_TBP_CTL_AFTER) ? T - 1 : T;
        tsr_dbg_hist h;
        tsr_dbg_hist_read(s, hit_pidx, &h);
        CHECK(h.present && (h.tbps & 1u), "%s: history INFO.TBPS of entry %u", what, hit_pidx);
        if (!(ctl & TSR_DBG_TBP_CTL_AFTER)) CHECK(h.stop_point, "%s: INFO.STOP of entry %u", what, hit_pidx);
    } else {
        CHECK(expect_T == UINT32_MAX, "%s: no stop", what);
        CHECK(k == p->n && (RD(s, TSR_CSR_STATUS) & TSR_STATUS_QIDLE), "%s: the program completed", what);
    }
    free(L.v);
    tsr_sim_destroy(s);
    return T;
}

static void test_tbp(const prog_t * p) {
    const uint32_t kind_lsb = TSR_DBG_TBP_CTL_KIND_LSB;
    // OPCODE: the first MVM, BEFORE
    const uint32_t m0 = find_op(p, TSR_OP_MVM, 0), m1 = find_op(p, TSR_OP_MVM, 1);
    tbp_run(p, TSR_DBG_TBP_CTL_KIND_OPCODE << kind_lsb, TSR_OP_MVM | (0xFFu << 8), 0, m0, "OPCODE MVM");
    // OPCODE with a unit mask that excludes MVM's unit: no match
    tbp_run(p, TSR_DBG_TBP_CTL_KIND_OPCODE << kind_lsb, TSR_OP_MVM | (0xFFu << 8), 1u << TSR_UNIT_DMA, UINT32_MAX, "OPCODE unit mask");
    // OPCODE with a sub mask: LD on channel 0 matches the first LD
    tbp_run(p, TSR_DBG_TBP_CTL_KIND_OPCODE << kind_lsb, TSR_OP_LD | (0xFFu << 8) | (0u << 16) | (0xFu << 20),
            1u << TSR_UNIT_DMA, find_op(p, TSR_OP_LD, 0), "OPCODE LD sub 0");
    // SPM W AFTER on chain 2's y line: stops after the MVM that writes it
    const uint32_t mvm2 = find_chain_op(p, TSR_OP_MVM, y_line(2));
    tbp_run(p, (TSR_DBG_TBP_CTL_KIND_SPM << kind_lsb) | TSR_DBG_TBP_CTL_W | TSR_DBG_TBP_CTL_AFTER, y_line(2), y_line(2) + 1,
            mvm2 + 1, "SPM W AFTER");
    // SPM R on the same line: the ST that reads it (BEFORE)
    const uint32_t st2 = find_chain_op(p, TSR_OP_ST, y_line(2));
    tbp_run(p, (TSR_DBG_TBP_CTL_KIND_SPM << kind_lsb) | TSR_DBG_TBP_CTL_R, y_line(2), y_line(2) + 1, st2, "SPM R");
    // SPM, neither R nor W: never matches
    tbp_run(p, TSR_DBG_TBP_CTL_KIND_SPM << kind_lsb, 0, 0xFFFFFFFFu, UINT32_MAX, "SPM no mode");
    // DRAM W on chain 2's output beat, its channel / another channel
    tbp_run(p, (TSR_DBG_TBP_CTL_KIND_DRAM << kind_lsb) | TSR_DBG_TBP_CTL_W | TSR_PUT(TSR_DBG_TBP_CTL_CHMASK, 1u << y_ch(2)),
            Y_BEAT + 2, Y_BEAT + 3, st2, "DRAM W");
    tbp_run(p, (TSR_DBG_TBP_CTL_KIND_DRAM << kind_lsb) | TSR_DBG_TBP_CTL_W | TSR_PUT(TSR_DBG_TBP_CTL_CHMASK, 1u << 0),
            Y_BEAT + 2, Y_BEAT + 3, UINT32_MAX, "DRAM W other channel");
    // DRAM R on the weights through channel 3 only: MVM weights are on every channel
    tbp_run(p, (TSR_DBG_TBP_CTL_KIND_DRAM << kind_lsb) | TSR_DBG_TBP_CTL_R | TSR_PUT(TSR_DBG_TBP_CTL_CHMASK, 1u << 3), W_OFF,
            W_OFF + 1, m0, "DRAM R weights");
    // PIDX range, also wrapping through 2^32: [0xFFFFFFF0, 3) holds pidx 0..2 -> T = 0
    tbp_run(p, TSR_DBG_TBP_CTL_KIND_PIDX << kind_lsb, 9, 10, 9, "PIDX 9");
    tbp_run(p, TSR_DBG_TBP_CTL_KIND_PIDX << kind_lsb, 0xFFFFFFF0u, 3, 0, "PIDX wrap");
    tbp_run(p, (TSR_DBG_TBP_CTL_KIND_PIDX << kind_lsb) | TSR_DBG_TBP_CTL_AFTER, 0xFFFFFFF0u, 1, 1, "PIDX wrap AFTER");
    // TICKET: the second MVM's ticket {MVM, 1}
    tbp_run(p, TSR_DBG_TBP_CTL_KIND_TICKET << kind_lsb, (TSR_UNIT_MVM << 28) | 1u, 0, m1, "TICKET");
    // kinds 5..7 never match
    tbp_run(p, 5u << kind_lsb, 0, 0xFFFFFFFFu, UINT32_MAX, "KIND 5");
    // RELEASE pops M up to T; the next M entry (the second MVM) becomes T
    tsr_sim * s = mk();
    init_mem(s, p->chains);
    evlog L;
    ev_init(&L, s);
    WR(s, TSR_CSR_DBG_TBP_A(1), TSR_OP_MVM | (0xFFu << 8));
    WR(s, TSR_CSR_DBG_TBP_CTL(1), TSR_DBG_TBP_CTL_EN | (TSR_DBG_TBP_CTL_KIND_OPCODE << kind_lsb));
    uint32_t tk[MAXOPS], k = 0;
    CHECK(settle_drained(s, p, &k, tk, &L, 10000000ull) && RD(s, TSR_CSR_DBG_STOP_PIDX) == m0, "first MVM");
    CHECK(TSR_GET(ST(s), TSR_DBG_STATUS_BP_HIT) == 2u, "BP_HIT TBP1");
    WR(s, TSR_CSR_DBG_CTRL, TSR_DBG_CTRL_RELEASE);
    CHECK(settle_drained(s, p, &k, tk, &L, 10000000ull) && RD(s, TSR_CSR_DBG_STOP_PIDX) == m1, "RELEASE: next match %u (T %u)",
          m1, RD(s, TSR_CSR_DBG_STOP_PIDX));
    snap_t got, ref;
    snap(s, p->chains, &got);
    prefix_snap(p, m1, &ref);
    CHECK(snap_eq(&got, &ref), "second MVM stop equals the prefix [0, %u)", m1);
    WR(s, TSR_CSR_DBG_TBP_CTL(1), 0);
    WR(s, TSR_CSR_DBG_CTRL, TSR_DBG_CTRL_MATCHQ_CLEAR | TSR_DBG_CTRL_RELEASE);
    CHECK(RD(s, TSR_CSR_DBG_MATCHQ) == 0 && !(ST(s) & TSR_DBG_STATUS_STOP_ACTIVE), "MATCHQ_CLEAR + RELEASE");
    CHECK(run_prog(s, p, &k, p->n, tk, 50000000ull), "completes");
    free(L.v);
    tsr_sim_destroy(s);
    // A range breakpoint never matches a poisoned entry; PIDX does
    s = mk();
    WR(s, TSR_CSR_DBG_TBP_CTL(0), TSR_DBG_TBP_CTL_EN | (TSR_DBG_TBP_CTL_KIND_SPM << kind_lsb) | TSR_DBG_TBP_CTL_W |
                                     TSR_DBG_TBP_CTL_COUNT_ONLY);
    WR(s, TSR_CSR_DBG_TBP_A(0), 0);
    WR(s, TSR_CSR_DBG_TBP_B(0), 0xFFFFFFFFu);
    WR(s, TSR_CSR_DBG_TBP_CTL(1), TSR_DBG_TBP_CTL_EN | (TSR_DBG_TBP_CTL_KIND_PIDX << kind_lsb) | TSR_DBG_TBP_CTL_COUNT_ONLY);
    WR(s, TSR_CSR_DBG_TBP_A(1), 0);
    WR(s, TSR_CSR_DBG_TBP_B(1), 1);
    tsr_sim_push(s, tsr_enc_ld(0, 70000, 0, 4).w, NONE, NONE, ALL);   // SPM_RANGE
    CHECK(tsr_sim_run_until_idle(s, 100000), "idle");
    CHECK(RD(s, TSR_CSR_Q_ERR_CODE(TSR_UNIT_DMA)) == TSR_ERR_SPM_RANGE, "poisoned");
    CHECK(RD(s, TSR_CSR_DBG_TBP_HITS(0)) == 0 && RD(s, TSR_CSR_DBG_TBP_HITS(1)) == 1, "SPM range kind skips the poisoned entry");
    tsr_sim_destroy(s);
}

// ---------------------------------------------------------------------------
// 7. Dispatch history (DB §5.6; invariant H)
// ---------------------------------------------------------------------------
static void test_history(void) {
    tsr_sim * s = mk();
    evlog L;
    ev_init(&L, s);
    const uint32_t N = 3 * TSR_HIST_DEPTH + 37;
    uint32_t * tk = (uint32_t *)malloc(N * 4), * dep0 = (uint32_t *)malloc(N * 4);
    rng_t r = { 5 };
    for (uint32_t i = 0; i < N; ++i) {
        tsr_insn in = (i % 3 == 0) ? tsr_enc_ld(i % 4, 10 + i % 1000, i, 1) : tsr_enc_nop();
        dep0[i] = (i && rng_below(&r, 2)) ? tk[rng_below(&r, i)] : NONE;
        while (credit(s, unit_of(&in)) == 0) tsr_sim_advance(s, 8);
        tk[i] = tsr_sim_push(s, in.w, dep0[i], NONE, ALL | (i % 5 == 0 ? TSR_ISSUE_PUSH_REC : 0u));
    }
    CHECK(tsr_sim_run_until_idle(s, 10000000), "idle");
    CHECK(RD(s, TSR_CSR_DBG_HIST_HEAD) == N, "HIST_HEAD %u", RD(s, TSR_CSR_DBG_HIST_HEAD));
    uint32_t ok = 0;
    for (uint32_t p = N - TSR_HIST_DEPTH; p < N; ++p) {
        tsr_dbg_hist h;
        tsr_dbg_hist_read(s, p, &h);
        const ev_t * e = ev_find(&L, TSR_QEV_ENQ, p);
        const uint32_t w0 = (p % 3 == 0) ? tsr_enc_ld(p % 4, 10 + p % 1000, p, 1).w[0] : tsr_enc_nop().w[0];
        const int good = h.present && h.pidx == p && h.ticket == tk[p] && h.dep[0] == dep0[p] && h.dep[1] == NONE &&
                         h.w[0] == w0 && h.unit == (tk[p] >> 28) && h.code == 0 && h.rec == (p % 5 == 0) && e != NULL &&
                         h.time == (uint32_t)e->time && h.pc == TSR_SIM_PC_HOOK && (p % 3 != 0 || (h.w[2] == p && h.w[3] == 1));
        if (good) ++ok;
        else CHECK(0, "history entry %u (present %d pidx %u ticket 0x%08x/0x%08x)", p, h.present, h.pidx, h.ticket, tk[p]);
    }
    CHECK(ok == TSR_HIST_DEPTH, "the last 512 entries, in order");
    for (uint32_t p = 0; p < N - TSR_HIST_DEPTH; p += 97) {
        tsr_dbg_hist h;
        tsr_dbg_hist_read(s, p, &h);
        CHECK(!h.present, "entry %u overwritten: not PRESENT", p);
    }
    // HIST_CLEAR empties and keeps recording
    WR(s, TSR_CSR_DBG_CTRL, TSR_DBG_CTRL_HIST_CLEAR);
    tsr_dbg_hist h;
    tsr_dbg_hist_read(s, N - 1, &h);
    CHECK(!h.present && RD(s, TSR_CSR_DBG_HIST_HEAD) == N, "HIST_CLEAR");
    for (int i = 0; i < 3; ++i) tsr_sim_push(s, tsr_enc_nop().w, NONE, NONE, ALL);
    CHECK(tsr_sim_run_until_idle(s, 100000), "idle");
    tsr_dbg_hist_read(s, N + 2, &h);
    CHECK(h.present && RD(s, TSR_CSR_DBG_HIST_HEAD) == N + 3, "recording after the clear");
    // HIST_FREEZE_ERR: 256 entries after the first error
    const uint32_t E = N + 3;
    tsr_sim_push(s, tsr_enc_ld(9, 0, 0, 1).w, NONE, NONE, ALL);            // channel >= NUM_CH: ILLEGAL_OP
    for (uint32_t i = 0; i < 300; ++i) {
        while (credit(s, TSR_UNIT_SYNC) == 0) tsr_sim_advance(s, 8);
        tsr_sim_push(s, tsr_enc_nop().w, NONE, NONE, ALL);
    }
    CHECK(tsr_sim_run_until_idle(s, 1000000), "idle");
    tsr_dbg_hist_read(s, E, &h);
    CHECK(h.present && h.code == TSR_ERR_ILLEGAL_OP, "the erroring entry recorded with its code");
    tsr_dbg_hist_read(s, E + 255, &h);
    CHECK(h.present, "E + 255 recorded");
    tsr_dbg_hist_read(s, E + 256, &h);
    CHECK(!h.present, "E + 256 not recorded");
    CHECK((ST(s) & TSR_DBG_STATUS_HIST_FROZEN) && RD(s, TSR_CSR_DBG_HIST_HEAD) == E + 256, "HIST_FROZEN, HEAD %u",
          RD(s, TSR_CSR_DBG_HIST_HEAD));
    WR(s, TSR_CSR_DBG_CTRL, TSR_DBG_CTRL_HIST_CLEAR);
    CHECK(!(ST(s) & TSR_DBG_STATUS_HIST_FROZEN), "HIST_CLEAR unfreezes");
    // HIST_FREEZE and HIST_EN = 0 stop recording
    WR(s, TSR_CSR_DBG_CFG, TSR_CSR_DBG_CFG_RESET | TSR_DBG_CFG_HIST_FREEZE);
    const uint32_t P = RD(s, TSR_CSR_DBG_NEXT_PIDX);
    tsr_sim_push(s, tsr_enc_nop().w, NONE, NONE, ALL);
    CHECK(tsr_sim_run_until_idle(s, 100000), "idle");
    WR(s, TSR_CSR_DBG_CFG, TSR_DBG_CFG_HIST_FREEZE_ERR);
    tsr_sim_push(s, tsr_enc_nop().w, NONE, NONE, ALL);
    CHECK(tsr_sim_run_until_idle(s, 100000), "idle");
    tsr_dbg_hist_read(s, P, &h);
    CHECK(!h.present, "HIST_FREEZE: not recorded");
    tsr_dbg_hist_read(s, P + 1, &h);
    CHECK(!h.present, "HIST_EN 0: not recorded");
    free(tk);
    free(dep0);
    free(L.v);
    tsr_sim_destroy(s);
}

// ---------------------------------------------------------------------------
// 8. Completion records to the RISC-V (DB §8.2)
// ---------------------------------------------------------------------------
static void test_rec(const prog_t * p) {
    tsr_sim * s = mk();
    init_mem(s, p->chains);
    evlog L;
    ev_init(&L, s);
    uint32_t tk[MAXOPS], k = 0;
    const uint32_t m = 24;
    CHECK(run_prog(s, p, &k, m, tk, 50000000ull), "run");
    tsr_sim_cw_write(s, TSR_CW_TRACE_MARK, 0x50000007u);                 // LAYER 7
    CHECK(tsr_sim_run_until_idle(s, 100000), "marker dispatched");
    uint32_t nrec = 0;
    for (uint32_t i = 0; i < m; ++i) nrec += (p->v[i].push & TSR_ISSUE_PUSH_REC) != 0;
    CHECK(tsr_sim_cw_read(s, TSR_CW_REC_COUNT) == nrec + 1, "REC_COUNT %u, expected %u", tsr_sim_cw_read(s, TSR_CW_REC_COUNT),
          nrec + 1);
    // the records, in completion order, against the event log
    uint32_t got = 0;
    for (size_t i = 0; i < L.n; ++i) {
        const ev_t * e = &L.v[i];
        if (!((e->kind == TSR_QEV_DONE && e->rec) || e->kind == TSR_QEV_MARK)) continue;
        uint32_t d[4];
        for (uint32_t j = 0; j < 4; ++j) d[j] = tsr_sim_cw_read(s, TSR_CW_REC_DATA(j));
        tsr_dbg_rec r;
        tsr_dbg_rec_decode(d, &r);
        if (e->kind == TSR_QEV_MARK) {
            CHECK(r.ticket == TSR_DBG_REC_MARK_TICKET && r.marker && r.run == 0x50000007u && r.t_end == (uint32_t)e->time &&
                  r.pidx16 == (m & 0xFFFFu) && r.code == 0 && !r.ran, "marker record");
        } else {
            const ev_t * st = ev_find(&L, TSR_QEV_START, e->pidx);
            const uint32_t run = (e->ran && st) ? (uint32_t)(e->time - st->time) : 0u;
            CHECK(r.ticket == e->ticket && r.t_end == (uint32_t)e->time && r.run == run && r.code == e->code &&
                  r.ran == e->ran && !r.marker && r.opcode == (e->w0 & 0xFFu) && r.pidx16 == (e->pidx & 0xFFFFu),
                  "record of entry %u: ticket 0x%08x/0x%08x t %u/%llu run %u/%u", e->pidx, r.ticket, e->ticket, r.t_end,
                  (unsigned long long)e->time, r.run, run);
        }
        tsr_sim_cw_write(s, TSR_CW_REC_POP, 0);
        ++got;
    }
    CHECK(got == nrec + 1 && tsr_sim_cw_read(s, TSR_CW_REC_COUNT) == 0 && tsr_sim_cw_read(s, TSR_CW_REC_DATA(0)) == 0,
          "every record popped; an empty FIFO reads 0");
    tsr_sim_cw_write(s, TSR_CW_REC_POP, 0);
    CHECK(tsr_sim_cw_read(s, TSR_CW_REC_COUNT) == 0, "a pop of an empty FIFO is ignored");
    // REC_CFG.ALL: every entry; IRQ_HALF: irq_fast[4] while >= 32 (a launch active: PERF counts)
    WR(s, TSR_CSR_LQ_PUSH, 0);
    tsr_sim_cw_write(s, TSR_CW_LQ_POP, 0);
    tsr_sim_cw_write(s, TSR_CW_REC_CFG, 0xFFFFFFFFu);
    CHECK(tsr_sim_cw_read(s, TSR_CW_REC_CFG) == 3, "REC_CFG [1:0]");
    for (uint32_t i = 0; i < 31; ++i) {
        while (credit(s, TSR_UNIT_SYNC) == 0) tsr_sim_advance(s, 8);
        tsr_sim_push(s, tsr_enc_nop().w, NONE, NONE, ALL);
    }
    CHECK(tsr_sim_run_until_idle(s, 100000), "idle");
    CHECK(tsr_sim_cw_read(s, TSR_CW_REC_COUNT) == 31 && !(tsr_sim_rv_irq(s) & TSR_RV_MIP_FAST(4)), "31 records: no irq_fast[4]");
    tsr_sim_push(s, tsr_enc_nop().w, NONE, NONE, ALL);
    CHECK(tsr_sim_run_until_idle(s, 100000), "idle");
    CHECK(tsr_sim_rv_irq(s) & TSR_RV_MIP_FAST(4), "32 records: irq_fast[4]");
    tsr_sim_cw_write(s, TSR_CW_REC_CFG, TSR_REC_CFG_ALL);
    CHECK(!(tsr_sim_rv_irq(s) & TSR_RV_MIP_FAST(4)), "IRQ_HALF off: no irq");
    for (uint32_t i = 0; i < 40; ++i) {
        while (credit(s, TSR_UNIT_SYNC) == 0) tsr_sim_advance(s, 8);
        tsr_sim_push(s, tsr_enc_nop().w, NONE, NONE, ALL);
    }
    CHECK(tsr_sim_run_until_idle(s, 100000), "idle");
    CHECK(tsr_sim_cw_read(s, TSR_CW_REC_COUNT) == 64 && (ST(s) & TSR_DBG_STATUS_REC_DROP), "full: 64 kept, REC_DROP");
    CHECK(tsr_sim_perf(s, TSR_PERF_REC_DROPPED(4)) == 8, "PERF REC_DROPPED %llu",
          (unsigned long long)tsr_sim_perf(s, TSR_PERF_REC_DROPPED(4)));
    free(L.v);
    tsr_sim_destroy(s);
}

// ---------------------------------------------------------------------------
// 9. Trace writer (DB §8.3)
// ---------------------------------------------------------------------------
#define TRW_BEAT 0x40000u
static void trw_read(tsr_sim * s, uint32_t k, uint32_t lines, tsr_trw_rec * o) {
    const uint8_t * base = tsr_sim_dram(s, 0) + (uint64_t)TRW_BEAT * 64u;
    tsr_trw_decode(base + (uint64_t)(k % (2u * lines)) * 32u, o);
}

static void test_trw(const prog_t * p) {
    tsr_sim * s = mk();
    init_mem(s, p->chains);
    evlog L;
    ev_init(&L, s);
    const uint32_t lines = 128;                                          // 256 records
    WR(s, TSR_CSR_DBG_TRW_BASE, TSR_PUT(TSR_HANDLE_CH, 0) | TRW_BEAT);
    WR(s, TSR_CSR_DBG_TRW_LINES, lines);
    WR(s, TSR_CSR_DBG_CFG, TSR_CSR_DBG_CFG_RESET | TSR_DBG_CFG_TRW_EN);
    uint32_t tk[MAXOPS], k = 0;
    const uint32_t m = 30;
    CHECK(run_prog(s, p, &k, 15, tk, 50000000ull), "run 1");
    tsr_sim_cw_write(s, TSR_CW_TRACE_MARK, 0x30000004u);                 // NODE_BEGIN 4
    WR(s, TSR_CSR_DBG_CTRL, TSR_DBG_CTRL_STOP_NOW);
    tsr_sim_advance(s, 100);
    WR(s, TSR_CSR_DBG_CTRL, TSR_DBG_CTRL_RELEASE);
    CHECK(run_prog(s, p, &k, m, tk, 50000000ull), "run 2");
    WR(s, TSR_CSR_DBG_CTRL, TSR_DBG_CTRL_TRW_FLUSH);
    const uint32_t wptr = RD(s, TSR_CSR_DBG_TRW_WPTR);
    CHECK(wptr == m + 3, "WPTR %u = %u OP + MARK + 2 STOP", wptr, m + 3);
    uint64_t t_prev = 0;
    uint32_t n_op = 0, n_mark = 0, n_stop = 0;
    for (uint32_t i = 0; i < wptr; ++i) {
        tsr_trw_rec r;
        trw_read(s, i, lines, &r);
        CHECK(r.time >= t_prev, "record %u: monotonic time (%llu < %llu)", i, (unsigned long long)r.time, (unsigned long long)t_prev);
        t_prev = r.time;
        if (r.kind == TSR_TRW_W0_KIND_OP) {
            ++n_op;
            const ev_t * d = ev_find(&L, TSR_QEV_DONE, r.a1), * e = ev_find(&L, TSR_QEV_ENQ, r.a1), * st = ev_find(&L, TSR_QEV_START, r.a1);
            CHECK(d && e && r.a2 == d->ticket && r.unit == d->unit && r.code == d->code && r.ran == d->ran &&
                  r.opcode == (d->w0 & 0xFFu) && r.w0_hi == ((d->w0 >> 8) & 0xFFu) && r.time == d->time &&
                  r.rec == ((p->v[r.a1].push & TSR_ISSUE_PUSH_REC) != 0) && !r.tbp, "OP record %u fields", r.a1);
            if (d && e) {
                const uint64_t t_head = e->time + r.qwait;
                // t_head: the enqueue, or the time the unit's previous entry left the queue (its
                // start, or its completion when it did not run), whichever is later (DB §8.3)
                uint64_t want = e->time;
                const ev_t * prev = NULL;
                for (size_t j = 0; j < L.n; ++j)
                    if (L.v[j].kind == TSR_QEV_DONE && L.v[j].unit == d->unit && L.v[j].pidx < r.a1 &&
                        (prev == NULL || L.v[j].pidx > prev->pidx)) prev = &L.v[j];
                if (prev) {
                    const ev_t * ps = prev->ran ? ev_find(&L, TSR_QEV_START, prev->pidx) : prev;
                    if (ps && ps->time > want) want = ps->time;
                }
                CHECK(t_head == want, "OP %u: t_head %llu, expected %llu", r.a1, (unsigned long long)t_head,
                      (unsigned long long)want);
                if (r.ran) CHECK(st && r.run == d->time - st->time && t_head + r.hwait == st->time, "OP %u: RUN / QWAIT / HWAIT", r.a1);
                else CHECK(r.run == 0 && t_head + r.hwait == d->time, "OP %u (not run): HWAIT to the end", r.a1);
            }
        } else if (r.kind == TSR_TRW_W0_KIND_MARK) {
            ++n_mark;
            const ev_t * mk_ev = NULL;
            for (size_t j = 0; j < L.n; ++j) if (L.v[j].kind == TSR_QEV_MARK) mk_ev = &L.v[j];
            CHECK(mk_ev && r.a2 == 0x30000004u && r.a1 == 15 && r.time == mk_ev->time, "MARK record");
        } else if (r.kind == TSR_TRW_W0_KIND_STOP) {
            CHECK(r.a1 == 15 && r.a2 == TSR_DBG_CAUSE_NOW && r.run == (n_stop == 0 ? 1u : 0u), "STOP record %u", n_stop);
            ++n_stop;
        } else {
            CHECK(0, "unexpected record kind %u", r.kind);
        }
    }
    CHECK(n_op == m && n_mark == 1 && n_stop == 2, "records: %u OP, %u MARK, %u STOP", n_op, n_mark, n_stop);
    // TRW_MASK: no DMA records; LAUNCH_END flushes without TRW_FLUSH
    WR(s, TSR_CSR_DBG_TRW_MASK, 0x7FFu & ~(1u << TSR_UNIT_DMA));
    const uint32_t w0 = RD(s, TSR_CSR_DBG_TRW_WPTR);
    tsr_sim_push(s, tsr_enc_ld(0, 5000, 0, 1).w, NONE, NONE, ALL);
    tsr_sim_push(s, tsr_enc_nop().w, NONE, NONE, ALL);
    tsr_sim_cw_write(s, TSR_CW_TRACE_MARK, 0x20000001u);                 // LAUNCH_END
    CHECK(tsr_sim_run_until_idle(s, 100000), "idle");
    CHECK(RD(s, TSR_CSR_DBG_TRW_WPTR) == w0 + 2, "LAUNCH_END flushed NOP + MARK (no DMA record): %u", RD(s, TSR_CSR_DBG_TRW_WPTR) - w0);
    // idle flush after 65,536 cycles
    tsr_sim_push(s, tsr_enc_nop().w, NONE, NONE, ALL);
    CHECK(tsr_sim_run_until_idle(s, 100000), "idle");
    CHECK(RD(s, TSR_CSR_DBG_TRW_WPTR) == w0 + 2, "staged, not written yet");
    tsr_sim_advance(s, 70000);
    CHECK(RD(s, TSR_CSR_DBG_TRW_WPTR) == w0 + 3, "idle flush");
    free(L.v);
    tsr_sim_destroy(s);

    // Ring wrap (flight recorder) and TRW_STOP_FULL drops with a DROP record; a launch is active so
    // that the PERF event counters TRW_RECORDS / TRW_DROPS count (EX §6)
    s = mk();
    WR(s, TSR_CSR_LQ_PUSH, 0);
    tsr_sim_cw_write(s, TSR_CW_LQ_POP, 0);
    WR(s, TSR_CSR_DBG_TRW_BASE, TRW_BEAT);
    WR(s, TSR_CSR_DBG_TRW_LINES, 16);                                     // 32 records
    WR(s, TSR_CSR_DBG_CFG, TSR_CSR_DBG_CFG_RESET | TSR_DBG_CFG_TRW_EN);
    for (uint32_t i = 0; i < 40; ++i) tsr_sim_cw_write(s, TSR_CW_TRACE_MARK, i);
    CHECK(tsr_sim_run_until_idle(s, 100000), "idle");
    WR(s, TSR_CSR_DBG_CTRL, TSR_DBG_CTRL_TRW_FLUSH);
    CHECK(RD(s, TSR_CSR_DBG_TRW_WPTR) == 40, "40 records written over a 32-record ring");
    for (uint32_t i = 8; i < 40; ++i) {
        tsr_trw_rec r;
        trw_read(s, i, 16, &r);
        CHECK(r.kind == TSR_TRW_W0_KIND_MARK && r.a2 == i, "ring slot %u holds record %u (value %u)", i % 32, i, r.a2);
    }
    WR(s, TSR_CSR_DBG_TRW_RPTR, 40);
    WR(s, TSR_CSR_DBG_CFG, TSR_CSR_DBG_CFG_RESET | TSR_DBG_CFG_TRW_EN | TSR_DBG_CFG_TRW_STOP_FULL);
    for (uint32_t i = 0; i < 40; ++i) tsr_sim_cw_write(s, TSR_CW_TRACE_MARK, 100 + i);
    CHECK(tsr_sim_run_until_idle(s, 100000), "idle");
    WR(s, TSR_CSR_DBG_CTRL, TSR_DBG_CTRL_TRW_FLUSH);
    CHECK(RD(s, TSR_CSR_DBG_TRW_WPTR) == 72 && RD(s, TSR_CSR_DBG_TRW_DROPS) == 8 && (ST(s) & TSR_DBG_STATUS_TRW_DROP),
          "STOP_FULL: 32 written, 8 dropped (WPTR %u, DROPS %u)", RD(s, TSR_CSR_DBG_TRW_WPTR), RD(s, TSR_CSR_DBG_TRW_DROPS));
    WR(s, TSR_CSR_DBG_TRW_RPTR, 72);
    tsr_sim_cw_write(s, TSR_CW_TRACE_MARK, 999);
    CHECK(tsr_sim_run_until_idle(s, 100000), "idle");
    WR(s, TSR_CSR_DBG_CTRL, TSR_DBG_CTRL_TRW_FLUSH);
    tsr_trw_rec r0, r1;
    trw_read(s, 72, 16, &r0);
    trw_read(s, 73, 16, &r1);
    CHECK(r0.kind == TSR_TRW_W0_KIND_DROP && r0.a1 == 8 && r1.kind == TSR_TRW_W0_KIND_MARK && r1.a2 == 999,
          "a DROP record (8) before the next record");
    CHECK(tsr_sim_perf(s, TSR_PERF_TRW_RECORDS) == 74 && tsr_sim_perf(s, TSR_PERF_TRW_DROPS) == 8,
          "PERF TRW_RECORDS %llu (= WPTR 74), TRW_DROPS %llu", (unsigned long long)tsr_sim_perf(s, TSR_PERF_TRW_RECORDS),
          (unsigned long long)tsr_sim_perf(s, TSR_PERF_TRW_DROPS));
    tsr_sim_destroy(s);
}

// ---------------------------------------------------------------------------
// 10. ERR_STOP, the hard hold against QABORT and SOFT_RESET (DB §5.2, DB-9)
// ---------------------------------------------------------------------------
static void test_err_stop(void) {
    tsr_sim * s = mk();
    evlog L;
    ev_init(&L, s);
    WR(s, TSR_CSR_DBG_CFG, TSR_CSR_DBG_CFG_RESET | TSR_DBG_CFG_ERR_STOP);
    for (int i = 0; i < 3; ++i) tsr_sim_push(s, tsr_enc_ld(0, 100 + 10 * i, 0, 8).w, NONE, NONE, ALL);
    tsr_sim_push(s, tsr_enc_qact(0, 0, 0, 33, 1, 0).w, NONE, NONE, ALL);   // K_RANGE on QACT, pidx 3
    for (int i = 0; i < 3; ++i) tsr_sim_push(s, tsr_enc_ld(0, 200 + 10 * i, 0, 8).w, NONE, NONE, ALL);
    for (int i = 0; i < 400 && !(ST(s) & TSR_DBG_STATUS_DRAINED); ++i) tsr_sim_advance(s, 16);
    const uint32_t st = ST(s), T = RD(s, TSR_CSR_DBG_STOP_PIDX);
    CHECK((st & TSR_DBG_STATUS_STOP_ACTIVE) && TSR_GET(st, TSR_DBG_STATUS_CAUSE) == TSR_DBG_CAUSE_ERR && T >= 4 &&
          (st & TSR_DBG_STATUS_DRAINED), "ERR_STOP at %u (0x%08x)", T, st);
    check_held(&L, T, 0, "ERR_STOP");
    WR(s, TSR_CSR_DBG_CTRL, TSR_DBG_CTRL_RELEASE);
    CHECK(tsr_sim_run_until_idle(s, 100000), "idle after RELEASE");
    free(L.v);
    tsr_sim_destroy(s);
}

static void test_qabort_soft_reset(const prog_t * p) {
    // QABORT: held entries drain ABORTED (the hard hold blocks starts only), the stop stays
    tsr_sim * s = mk();
    init_mem(s, p->chains);
    evlog L;
    ev_init(&L, s);
    const uint32_t T = 10;
    WR(s, TSR_CSR_DBG_STOP_AT, T);
    uint32_t tk[MAXOPS], k = 0;
    CHECK(settle_drained(s, p, &k, tk, &L, 10000000ull), "stopped");
    const size_t n0 = L.n;
    WR(s, TSR_CSR_CTRL, TSR_CTRL_QABORT);
    CHECK(tsr_sim_run_until_idle(s, 1000000), "aborted entries drain under the stop");
    uint32_t drained = 0;
    for (size_t i = n0; i < L.n; ++i) {
        if (L.v[i].kind == TSR_QEV_START) CHECK(0, "entry %u started after the QABORT", L.v[i].pidx);
        if (L.v[i].kind == TSR_QEV_DONE) { ++drained; CHECK(!L.v[i].ran && L.v[i].code == TSR_ERR_ABORTED, "ABORTED"); }
    }
    CHECK(drained == k - T && (ST(s) & TSR_DBG_STATUS_STOP_ACTIVE), "%u held entries completed ABORTED; still stopped", drained);
    WR(s, TSR_CSR_CTRL, TSR_CTRL_QCLEAR);
    CHECK(!(RD(s, TSR_CSR_STATUS) & TSR_STATUS_QCLEAR_REFUSED), "host QCLEAR");
    tsr_sim_push(s, tsr_enc_nop().w, NONE, NONE, ALL);
    tsr_sim_advance(s, 1000);
    CHECK(!(RD(s, TSR_CSR_STATUS) & TSR_STATUS_QIDLE), "after the QCLEAR the hard hold is back: the NOP is held");
    WR(s, TSR_CSR_DBG_CTRL, TSR_DBG_CTRL_RELEASE);
    CHECK(tsr_sim_run_until_idle(s, 100000), "released");
    free(L.v);
    tsr_sim_destroy(s);

    // SOFT_RESET: step 1 clears S, S_VALID and M; SERIAL, TBPs, history and TRW pointers are kept
    s = mk();
    init_mem(s, p->chains);
    ev_init(&L, s);
    WR(s, TSR_CSR_DBG_CFG, TSR_CSR_DBG_CFG_RESET | TSR_DBG_CFG_SERIAL | TSR_DBG_CFG_TRW_EN);
    WR(s, TSR_CSR_DBG_TRW_BASE, TRW_BEAT);
    WR(s, TSR_CSR_DBG_TRW_LINES, 64);
    WR(s, TSR_CSR_DBG_TBP_A(2), TSR_OP_QACT | (0xFFu << 8));
    WR(s, TSR_CSR_DBG_TBP_CTL(2), TSR_DBG_TBP_CTL_EN | (TSR_DBG_TBP_CTL_KIND_OPCODE << TSR_DBG_TBP_CTL_KIND_LSB));
    k = 0;
    CHECK(settle_drained(s, p, &k, tk, &L, 10000000ull), "TBP stop");
    WR(s, TSR_CSR_DBG_STOP_AT, RD(s, TSR_CSR_DBG_STOP_PIDX) + 2);
    WR(s, TSR_CSR_DBG_CTRL, TSR_DBG_CTRL_TRW_FLUSH);
    const uint32_t wptr = RD(s, TSR_CSR_DBG_TRW_WPTR), head = RD(s, TSR_CSR_DBG_HIST_HEAD);
    CHECK(RD(s, TSR_CSR_DBG_MATCHQ) >= 1 && (ST(s) & TSR_DBG_STATUS_S_VALID), "S and M set");
    WR(s, TSR_CSR_CTRL, TSR_CTRL_SOFT_RESET);
    const uint32_t st = ST(s);
    CHECK(!(st & (TSR_DBG_STATUS_STOP_ACTIVE | TSR_DBG_STATUS_S_VALID | TSR_DBG_STATUS_DB_HELD)) && RD(s, TSR_CSR_DBG_MATCHQ) == 0,
          "SOFT_RESET step 1 cleared the stop state (0x%08x)", st);
    for (int i = 0; i < 1000 && (RD(s, TSR_CSR_STATUS) & TSR_STATUS_SR_BUSY); ++i) tsr_sim_advance(s, 64);
    CHECK(!(RD(s, TSR_CSR_STATUS) & TSR_STATUS_SR_BUSY), "SOFT_RESET completes (held entries drained)");
    CHECK((RD(s, TSR_CSR_DBG_CFG) & TSR_DBG_CFG_SERIAL) && RD(s, TSR_CSR_DBG_TBP_A(2)) == (TSR_OP_QACT | (0xFFu << 8)) &&
          (RD(s, TSR_CSR_DBG_TBP_CTL(2)) & TSR_DBG_TBP_CTL_EN), "SERIAL and TBPs kept");
    CHECK(RD(s, TSR_CSR_DBG_TRW_WPTR) >= wptr && RD(s, TSR_CSR_DBG_TRW_LINES) == 64 && RD(s, TSR_CSR_DBG_TRW_BASE) == TRW_BEAT,
          "TRW ring and pointers kept");
    tsr_dbg_hist h;
    tsr_dbg_hist_read(s, head - 1, &h);
    CHECK(h.present && RD(s, TSR_CSR_DBG_HIST_HEAD) == head, "history kept");
    free(L.v);
    tsr_sim_destroy(s);
}

// ---------------------------------------------------------------------------
// 11. DBG_STOP_CYCLES (EX §6, DB §8.1)
// ---------------------------------------------------------------------------
static void test_stop_cycles(void) {
    tsr_sim * s = mk();
    WR(s, TSR_CSR_LQ_PUSH, 0);                                            // a launch, popped as the RISC-V would
    tsr_sim_cw_write(s, TSR_CW_LQ_POP, 0);
    CHECK(RD(s, TSR_CSR_STATUS) & TSR_STATUS_ACTIVE, "ACTIVE");
    tsr_sim_advance(s, 500);
    CHECK(tsr_sim_perf(s, TSR_PERF_DBG_STOP_CYCLES) == 0, "no stop: 0");
    WR(s, TSR_CSR_DBG_CTRL, TSR_DBG_CTRL_STOP_NOW);
    tsr_sim_advance(s, 1000);
    CHECK(tsr_sim_perf(s, TSR_PERF_DBG_STOP_CYCLES) == 1000, "STOP_ACTIVE: 1000 (%llu)",
          (unsigned long long)tsr_sim_perf(s, TSR_PERF_DBG_STOP_CYCLES));
    WR(s, TSR_CSR_DBG_CTRL, TSR_DBG_CTRL_RELEASE);
    tsr_sim_advance(s, 700);
    CHECK(tsr_sim_perf(s, TSR_PERF_DBG_STOP_CYCLES) == 1000, "released: unchanged");
    CHECK(tsr_sim_cw_read(s, TSR_CW_DBG_FLAGS) == 0, "dbg_stop 0");
    tsr_sim_destroy(s);
}

// ---------------------------------------------------------------------------
// 12. RISC-V side: watchpoints, halt-on-boot, STOP_HALT_RV, dbg_step (DB §3.3, §3.7, §3.8, §5.2)
// ---------------------------------------------------------------------------
#if HAVE_RV
static uint32_t rv_lui(uint32_t rd, uint32_t imm20) { return (imm20 << 12) | (rd << 7) | 0x37u; }
static uint32_t rv_addi(uint32_t rd, uint32_t rs1, int32_t imm) { return ((uint32_t)imm & 0xFFFu) << 20 | rs1 << 15 | rd << 7 | 0x13u; }
static uint32_t rv_sw(uint32_t rs2, uint32_t rs1, int32_t imm) {
    const uint32_t i = (uint32_t)imm & 0xFFFu;
    return (i >> 5) << 25 | rs2 << 20 | rs1 << 15 | 2u << 12 | (i & 0x1Fu) << 7 | 0x23u;
}
static uint32_t rv_lw(uint32_t rd, uint32_t rs1, int32_t imm) { return ((uint32_t)imm & 0xFFFu) << 20 | rs1 << 15 | 2u << 12 | rd << 7 | 0x03u; }
static uint32_t rv_jal(uint32_t rd, int32_t off) {
    const uint32_t i = (uint32_t)off;
    return ((i >> 20) & 1u) << 31 | ((i >> 1) & 0x3FFu) << 21 | ((i >> 11) & 1u) << 20 | ((i >> 12) & 0xFFu) << 12 | rd << 7 | 0x6Fu;
}

// 0x80 lui x5,0x100; 0x84 addi x6,x0,0x55; 0x88 sw x6,16(x5); 0x8C lw x7,32(x5); 0x90 addi x8,x8,1;
// 0x94 sw x8,64(x5); 0x98 j 0x90
static void load_prog(tsr_sim * s) {
    const uint32_t code[] = { rv_lui(5, 0x100), rv_addi(6, 0, 0x55), rv_sw(6, 5, 16), rv_lw(7, 5, 32), rv_addi(8, 8, 1),
                              rv_sw(8, 5, 64), rv_jal(0, -8) };
    WR(s, TSR_CSR_LOAD_ADDR, 0x80);
    for (size_t i = 0; i < sizeof(code) / 4; ++i) WR(s, TSR_CSR_LOAD_DATA, code[i]);
    WR(s, TSR_CSR_RV_BOOT, 0);
}
static tsr_rv * hart_of(tsr_sim * s) { return tsr_rv_agent_rv(tsr_sim_rv_agent(s)); }
static uint32_t dtcm_word(tsr_sim * s, uint32_t off) { uint32_t n; return get_u32_le(tsr_sim_dtcm(s, &n) + off); }
static int run_until_halt(tsr_sim * s, uint64_t budget) {
    const uint64_t t_end = tsr_sim_now(s) + budget;
    while (tsr_sim_now(s) < t_end) {
        if (RD(s, TSR_CSR_STATUS) & TSR_STATUS_RV_HALTED) return 1;
        tsr_sim_advance(s, 200);
    }
    return (RD(s, TSR_CSR_STATUS) & TSR_STATUS_RV_HALTED) != 0;
}
// dpc of the halted hart: through the DM's abstract command (DB §3.5) when the DM is attached.
static uint32_t rv_dpc(tsr_sim * s, int via_dm) {
    if (via_dm) {
        WR(s, TSR_CSR_DMI(TSR_DMI_COMMAND), 0x00220000u | TSR_RV_CSR_DPC);
        CHECK(((RD(s, TSR_CSR_DMI(TSR_DMI_ABSTRACTCS)) >> 8) & 7u) == 0, "abstract command ok");
        return RD(s, TSR_CSR_DMI(TSR_DMI_DATA0));
    }
    uint32_t v = 0;
    tsr_rv_csr_read(hart_of(s), TSR_RV_CSR_DPC, &v);
    return v;
}
static void rv_resume(tsr_sim * s, int via_dm) {
    if (via_dm) {
        WR(s, TSR_CSR_DMI(TSR_DMI_DMCONTROL), 0x40000001u);
        WR(s, TSR_CSR_DMI(TSR_DMI_DMCONTROL), 0x00000001u);
    } else {
        CHECK(tsr_rv_debug_resume(hart_of(s)) == 0, "resume");
        WR(s, TSR_CSR_STATUS, 0);                                         // a host action: the agent runs again
    }
}

// Rebuilds the model without the DM (the "stub" path: halt requests go to the hart directly).
static void drop_dm(tsr_sim * s) {
#if defined(TSR_SIM_HAVE_DM) && TSR_SIM_HAVE_DM
    if (!have_dm(s)) return;
    tsr_dbg_detach(s);
    tsr_dm_detach(s);
    tsr_dbg_attach(s);
    CHECK(!(RD(s, TSR_CSR_CTL_ID) & TSR_CTL_ID_DM), "CTL_ID.DM 0: DBG without the DM model");
#else
    (void)s;
#endif
}

// The DM (when used): DMI_EN and dmactive.
static void dm_on(tsr_sim * s, int via_dm) {
    if (!via_dm) {
        drop_dm(s);
        return;
    }
    WR(s, TSR_CSR_DBG_CFG, RD(s, TSR_CSR_DBG_CFG) | TSR_DBG_CFG_DMI_EN);
    WR(s, TSR_CSR_DMI(TSR_DMI_DMCONTROL), 1u);
}

static void test_rvwp(int via_dm) {
    tsr_sim * s = mk();
    dm_on(s, via_dm);
    load_prog(s);
    WR(s, TSR_CSR_DBG_CFG, RD(s, TSR_CSR_DBG_CFG) | TSR_DBG_CFG_RV_HALT_EN);
    // WP0: store to [0x100010, 0x100014) with HALT; WP1: load whose last byte is 0x100023, no HALT
    WR(s, TSR_CSR_DBG_RVWP_LO(0), 0x100010u);
    WR(s, TSR_CSR_DBG_RVWP_HI(0), 0x100014u);
    WR(s, TSR_CSR_DBG_RVWP_CTL(0), TSR_DBG_RVWP_CTL_EN | TSR_DBG_RVWP_CTL_STORE | TSR_DBG_RVWP_CTL_HALT);
    WR(s, TSR_CSR_DBG_RVWP_LO(1), 0x100023u);
    WR(s, TSR_CSR_DBG_RVWP_HI(1), 0x100024u);
    WR(s, TSR_CSR_DBG_RVWP_CTL(1), TSR_DBG_RVWP_CTL_EN | TSR_DBG_RVWP_CTL_LOAD);
    WR(s, TSR_CSR_DBG_CTRL, TSR_DBG_CTRL_RVWP_APPLY);
    WR(s, TSR_CSR_CTRL, TSR_CTRL_RV_RUN);
    CHECK(run_until_halt(s, 100000), "watchpoint halt (%s)", via_dm ? "DM" : "no DM");
    CHECK(RD(s, TSR_CSR_DBG_RV_WPADDR) == 0x100010u && RD(s, TSR_CSR_DBG_RVWP_HITS(0)) == 1 &&
          (TSR_GET(ST(s), TSR_DBG_STATUS_RVWP_HIT) & 1u), "WP0 hit: WPADDR, HITS, RVWP_HIT");
    CHECK(dtcm_word(s, 0x10) == 0x55, "the watched store completed");
    CHECK(rv_dpc(s, via_dm) == 0x8Cu, "halted right after the store (dpc 0x%08x)", rv_dpc(s, via_dm));
    CHECK(!(ST(s) & TSR_DBG_STATUS_RV_HALT_PEND) && (ST(s) & TSR_DBG_STATUS_RV_HALTED), "pend cleared by RV_HALTED");
    CHECK(TSR_GET(RD(s, TSR_CSR_DBG_RV_STATE), TSR_DBG_RV_STATE_DEBUG_ENTRIES) == 1, "one debug-mode entry");
    // resume: the load at 0x8C hits WP1 (no HALT); no re-halt from the old request (one-shot)
    rv_resume(s, via_dm);
    for (int i = 0; i < 20; ++i) tsr_sim_advance(s, 500);
    CHECK(!(RD(s, TSR_CSR_STATUS) & TSR_STATUS_RV_HALTED), "no re-halt after the resume");
    CHECK(RD(s, TSR_CSR_DBG_RVWP_HITS(1)) == 1 && RD(s, TSR_CSR_DBG_RV_WPADDR) == 0x100020u, "WP1: the load's last byte hits");
    // a store watchpoint without HALT counts the loop's stores; the configuration acts only after APPLY
    WR(s, TSR_CSR_DBG_RVWP_LO(1), 0x100040u);
    WR(s, TSR_CSR_DBG_RVWP_HI(1), 0x100041u);
    WR(s, TSR_CSR_DBG_RVWP_CTL(1), TSR_DBG_RVWP_CTL_EN | TSR_DBG_RVWP_CTL_STORE);
    WR(s, TSR_CSR_DBG_RVWP_HITS(1), 0);
    tsr_sim_advance(s, 2000);
    CHECK(RD(s, TSR_CSR_DBG_RVWP_HITS(1)) == 0, "not applied yet");
    WR(s, TSR_CSR_DBG_CTRL, TSR_DBG_CTRL_RVWP_APPLY);
    tsr_sim_advance(s, 2000);
    const uint32_t hits = RD(s, TSR_CSR_DBG_RVWP_HITS(1));
    CHECK(hits > 10 && dtcm_word(s, 0x40) >= hits, "loop stores counted (%u)", hits);
    WR(s, TSR_CSR_DBG_STATUS, TSR_DBG_STATUS_RVWP_HIT);
    CHECK(TSR_GET(ST(s), TSR_DBG_STATUS_RVWP_HIT) == 0, "RVWP_HIT W1C");
    // RV_SAMPLE while running: the pc is in the loop, the last data address its store
    WR(s, TSR_CSR_DBG_CTRL, TSR_DBG_CTRL_RV_SAMPLE);
    const uint32_t pc = RD(s, TSR_CSR_DBG_RV_PC);
    CHECK(pc >= 0x90u && pc <= 0x98u && RD(s, TSR_CSR_DBG_RV_DADDR) == 0x100040u, "RV sample pc 0x%x daddr 0x%x", pc,
          RD(s, TSR_CSR_DBG_RV_DADDR));
    tsr_sim_destroy(s);
}

static void test_halt_on_boot(int via_dm) {
    tsr_sim * s = mk();
    dm_on(s, via_dm);
    load_prog(s);
    WR(s, TSR_CSR_DBG_CFG, RD(s, TSR_CSR_DBG_CFG) | TSR_DBG_CFG_HALT_ON_BOOT | TSR_DBG_CFG_RV_HALT_EN);
    WR(s, TSR_CSR_CTRL, TSR_CTRL_RV_RUN);
    CHECK(run_until_halt(s, 10000), "halt on boot");
    CHECK(tsr_sim_rv_instret(s) == 0 && dtcm_word(s, 0x10) == 0, "no instruction executed");
    CHECK(rv_dpc(s, via_dm) == 0x80u, "dpc = RV_BOOT + 0x80");
    if (via_dm) CHECK(RD(s, TSR_CSR_DMI(TSR_DMI_DMSTATUS)) & (1u << 9), "dmstatus.allhalted");
    rv_resume(s, via_dm);
    tsr_sim_advance(s, 5000);
    CHECK(!(RD(s, TSR_CSR_STATUS) & TSR_STATUS_RV_HALTED) && dtcm_word(s, 0x10) == 0x55, "runs after the resume");
    // a reset without HALT_ON_BOOT boots without a halt; a request left from before the reset is dropped
    WR(s, TSR_CSR_DBG_CFG, RD(s, TSR_CSR_DBG_CFG) & ~TSR_DBG_CFG_HALT_ON_BOOT);
    WR(s, TSR_CSR_CTRL, 0);
    WR(s, TSR_CSR_DBG_CFG, RD(s, TSR_CSR_DBG_CFG) | TSR_DBG_CFG_STOP_HALT_RV);
    WR(s, TSR_CSR_DBG_CTRL, TSR_DBG_CTRL_STOP_NOW);                      // request while in reset: pending
    CHECK(ST(s) & TSR_DBG_STATUS_RV_HALT_PEND, "pend while the hart is in reset");
    WR(s, TSR_CSR_DBG_CTRL, TSR_DBG_CTRL_RV_HALT_CANCEL);
    CHECK(!(ST(s) & TSR_DBG_STATUS_RV_HALT_PEND), "RV_HALT_CANCEL");
    WR(s, TSR_CSR_DBG_CTRL, TSR_DBG_CTRL_RELEASE | TSR_DBG_CTRL_STOP_NOW);
    WR(s, TSR_CSR_DBG_CTRL, TSR_DBG_CTRL_STOP_NOW);
    WR(s, TSR_CSR_DBG_CTRL, TSR_DBG_CTRL_RELEASE);
    CHECK(ST(s) & TSR_DBG_STATUS_RV_HALT_PEND, "pend again");
    WR(s, TSR_CSR_CTRL, TSR_CTRL_RV_RUN);
    for (int i = 0; i < 10; ++i) tsr_sim_advance(s, 500);
    CHECK(!(RD(s, TSR_CSR_STATUS) & TSR_STATUS_RV_HALTED) && !(ST(s) & TSR_DBG_STATUS_RV_HALT_PEND),
          "the reset dropped the old request: the hart runs");
    tsr_sim_destroy(s);
}

// The halt's time (DB §8.1, EX §6): STATUS.RV_HALTED, DBG_STOP_CYCLES and the RVHALT record start at
// the core time the hart entered debug mode, not at the start of the agent's slice: a delay loop of
// ~800 RV cycles runs in one slice before the watched store.
static uint32_t rv_bne(uint32_t rs1, uint32_t rs2, int32_t off) {
    const uint32_t i = (uint32_t)off;
    return ((i >> 12) & 1u) << 31 | ((i >> 5) & 0x3Fu) << 25 | rs2 << 20 | rs1 << 15 | 1u << 12 | ((i >> 1) & 0xFu) << 8 |
           ((i >> 11) & 1u) << 7 | 0x63u;
}
static void test_halt_time(void) {
    tsr_sim * s = mk();
    const uint32_t code[] = { rv_lui(5, 0x100), rv_addi(9, 0, 200), rv_addi(9, 9, -1), rv_bne(9, 0, -4), rv_sw(6, 5, 16),
                              rv_jal(0, 0) };
    WR(s, TSR_CSR_LOAD_ADDR, 0x80);
    for (size_t i = 0; i < sizeof(code) / 4; ++i) WR(s, TSR_CSR_LOAD_DATA, code[i]);
    WR(s, TSR_CSR_DBG_CFG, RD(s, TSR_CSR_DBG_CFG) | TSR_DBG_CFG_RV_HALT_EN | TSR_DBG_CFG_TRW_EN);
    WR(s, TSR_CSR_DBG_TRW_BASE, TRW_BEAT);
    WR(s, TSR_CSR_DBG_TRW_LINES, 16);
    WR(s, TSR_CSR_DBG_TRW_MASK, TSR_DBG_TRW_MASK_RVHALTS);
    WR(s, TSR_CSR_DBG_RVWP_LO(0), 0x100010u);
    WR(s, TSR_CSR_DBG_RVWP_HI(0), 0x100014u);
    WR(s, TSR_CSR_DBG_RVWP_CTL(0), TSR_DBG_RVWP_CTL_EN | TSR_DBG_RVWP_CTL_STORE | TSR_DBG_RVWP_CTL_HALT);
    WR(s, TSR_CSR_DBG_CTRL, TSR_DBG_CTRL_RVWP_APPLY);
    WR(s, TSR_CSR_LQ_PUSH, 0);
    tsr_sim_cw_write(s, TSR_CW_LQ_POP, 0);                              // ACTIVE
    const uint64_t t0 = tsr_sim_now(s);
    WR(s, TSR_CSR_CTRL, TSR_CTRL_RV_RUN);
    uint64_t t_h = 0;
    for (int i = 0; i < 5000 && !t_h; ++i) {
        tsr_sim_advance(s, 1);
        if (RD(s, TSR_CSR_STATUS) & TSR_STATUS_RV_HALTED) t_h = tsr_sim_now(s);
    }
    const uint64_t c = tsr_rv_cycles(hart_of(s));
    CHECK(t_h > t0 + 1000, "RV_HALTED rose at +%llu (the debug entry after ~%llu RV cycles), not at the slice start",
          (unsigned long long)(t_h - t0), (unsigned long long)c);
    CHECK(t_h <= t0 + (uint64_t)((double)c / tsr_sim_rv_ratio(s)) + 2, "no later than the hart's own time");
    tsr_sim_advance(s, 3000);
    const uint64_t t1 = tsr_sim_now(s);
    CHECK(tsr_sim_perf(s, TSR_PERF_DBG_STOP_CYCLES) == t1 - t_h, "DBG_STOP_CYCLES %llu = halted time %llu",
          (unsigned long long)tsr_sim_perf(s, TSR_PERF_DBG_STOP_CYCLES), (unsigned long long)(t1 - t_h));
    WR(s, TSR_CSR_DBG_CTRL, TSR_DBG_CTRL_TRW_FLUSH);
    tsr_trw_rec r;
    trw_read(s, 0, 16, &r);
    CHECK(RD(s, TSR_CSR_DBG_TRW_WPTR) == 1 && r.kind == TSR_TRW_W0_KIND_RVHALT && r.a2 == 1 && r.time == t_h,
          "RVHALT record at the halt's time");
    tsr_sim_destroy(s);
}

// dbg_stop (DB §3.9, CC §7.3): a push that waits for a credit during a DB stop never times out; the
// same wait under another hold source ends with PUSH_TIMEOUT and a store fault after ISSUE_WAIT_MAX.
// Firmware: LD ch0 SPM 0 <- beat 0, 1 line, pushed in a loop without a credit check (WMASK 0xFF).
static void wait_freeze_run(int db_stop) {
    tsr_sim * s = mk();
    const uint32_t code[] = { rv_lui(5, 0x20000), rv_addi(6, 0, TSR_OP_LD), rv_sw(6, 5, 0), rv_addi(6, 0, 1), rv_sw(6, 5, 12),
                              rv_lui(7, 0x10), rv_addi(7, 7, -256), rv_sw(7, 5, 0x28), rv_jal(0, -4) };
    WR(s, TSR_CSR_LOAD_ADDR, 0x80);
    for (size_t i = 0; i < sizeof(code) / 4; ++i) WR(s, TSR_CSR_LOAD_DATA, code[i]);
    if (db_stop) WR(s, TSR_CSR_DBG_STOP_AT, 0);
    else WR(s, TSR_CSR_CTRL, TSR_CTRL_HOST_HOLD);
    WR(s, TSR_CSR_CTRL, TSR_CTRL_RV_RUN | (db_stop ? 0u : TSR_CTRL_HOST_HOLD));
    adv(s, 2000);
    const uint32_t pushed = tsr_sim_cw_read(s, TSR_CW_ISSUE_PIDX);
    CHECK(pushed == TSR_QDEPTH_U200_DMA && tsr_rv_waiting(hart_of(s)), "%s: %u pushes, then the store waits",
          db_stop ? "DB stop" : "host hold", pushed);
    CHECK(((tsr_sim_cw_read(s, TSR_CW_DBG_FLAGS) & TSR_DBG_FLAGS_DBG_STOP) != 0) == db_stop, "DBG_FLAGS.dbg_stop");
    adv(s, 400000);                // > ISSUE_WAIT_MAX / 0.603 core cycles
    const int timed_out = (RD(s, TSR_CSR_STATUS) & TSR_STATUS_PUSH_TIMEOUT) != 0;
    if (db_stop) {
        CHECK(!timed_out && tsr_rv_waiting(hart_of(s)), "DB stop: the push still waits, no PUSH_TIMEOUT");
        WR(s, TSR_CSR_DBG_CTRL, TSR_DBG_CTRL_RELEASE);
        adv(s, 2000);
        CHECK(tsr_sim_cw_read(s, TSR_CW_ISSUE_PIDX) > pushed + 8, "after RELEASE the loop pushes on");
    } else {
        CHECK(timed_out, "host hold: PUSH_TIMEOUT after ISSUE_WAIT_MAX");
    }
    tsr_sim_destroy(s);
}

// STOP_HALT_RV: a stop activation halts the hart, once; dbg_step through the debug-halt hold.
static void test_stop_halt_rv(int via_dm) {
    // DB §3.3: DB's halt requests act only while DBG_CFG.RV_HALT_EN = 1. STOP_HALT_RV without it: the
    // stop happens, the hart keeps running and nothing is left pending.
    {
        tsr_sim * s = mk();
        dm_on(s, via_dm);
        load_prog(s);
        WR(s, TSR_CSR_DBG_CFG, (RD(s, TSR_CSR_DBG_CFG) | TSR_DBG_CFG_STOP_HALT_RV) & ~TSR_DBG_CFG_RV_HALT_EN);
        WR(s, TSR_CSR_CTRL, TSR_CTRL_RV_RUN);
        tsr_sim_advance(s, 3000);
        WR(s, TSR_CSR_DBG_CTRL, TSR_DBG_CTRL_STOP_NOW);
        const uint64_t n0 = tsr_sim_rv_instret(s);
        for (int i = 0; i < 10; ++i) tsr_sim_advance(s, 500);
        CHECK((ST(s) & TSR_DBG_STATUS_STOP_ACTIVE) && !(ST(s) & TSR_DBG_STATUS_RV_HALT_PEND) &&
              !(RD(s, TSR_CSR_STATUS) & TSR_STATUS_RV_HALTED) && tsr_sim_rv_instret(s) > n0,
              "RV_HALT_EN 0 (%s): STOP_HALT_RV stops the queues only, no halt, no pend", via_dm ? "DM" : "no DM");
        WR(s, TSR_CSR_DBG_CTRL, TSR_DBG_CTRL_RELEASE);
        tsr_sim_destroy(s);
    }
    tsr_sim * s = mk();
    dm_on(s, via_dm);
    load_prog(s);
    WR(s, TSR_CSR_DBG_CFG, RD(s, TSR_CSR_DBG_CFG) | TSR_DBG_CFG_STOP_HALT_RV | TSR_DBG_CFG_RV_HALT_EN);
    WR(s, TSR_CSR_CTRL, TSR_CTRL_RV_RUN | TSR_CTRL_HOLD_ON_HALT);
    tsr_sim_advance(s, 3000);
    CHECK(!(RD(s, TSR_CSR_STATUS) & TSR_STATUS_RV_HALTED), "running");
    WR(s, TSR_CSR_DBG_CTRL, TSR_DBG_CTRL_STOP_NOW);
    CHECK(run_until_halt(s, 10000), "STOP_HALT_RV halts the hart");
    CHECK(!(ST(s) & TSR_DBG_STATUS_RV_HALT_PEND), "one request, one halt");
    CHECK(tsr_sim_cw_read(s, TSR_CW_DBG_FLAGS) & TSR_DBG_FLAGS_DBG_STOP, "dbg_stop");
    WR(s, TSR_CSR_DBG_CTRL, TSR_DBG_CTRL_RELEASE);
    // the debug-halt hold (HOLD_ON_HALT) holds the queue heads; QSTEP releases one entry at a time
    evlog L;
    ev_init(&L, s);
    uint32_t tk[3];
    for (int i = 0; i < 3; ++i) tk[i] = tsr_sim_push(s, tsr_enc_ld(0, 100 + 10 * i, 0, 4).w, NONE, NONE, ALL);
    tsr_sim_advance(s, 2000);
    CHECK(RD(s, TSR_CSR_Q_DONE(TSR_UNIT_DMA)) == 0 && (TSR_GET(RD(s, TSR_CSR_STATUS), TSR_STATUS_HOLD_SRC) & TSR_HOLD_SRC_HALT),
          "held by the debug-halt hold");
    const uint32_t p0 = L.v[0].pidx;
    WR(s, TSR_CSR_DBG_CTRL, TSR_DBG_CTRL_QSTEP);                          // no stop: X = the oldest head
    CHECK((ST(s) & TSR_DBG_STATUS_STEP_BUSY) && RD(s, TSR_CSR_DBG_STEP_PIDX) == p0, "QSTEP with HOLD_ON_HALT: X %u",
          RD(s, TSR_CSR_DBG_STEP_PIDX));
    tsr_sim_advance(s, 2000);
    CHECK(RD(s, TSR_CSR_Q_DONE(TSR_UNIT_DMA)) == 1 && !(ST(s) & TSR_DBG_STATUS_STEP_BUSY), "exactly one entry ran");
    WR(s, TSR_CSR_DBG_STOP_AT, p0 + 1);                                   // with a stop: X = T
    CHECK(ST(s) & TSR_DBG_STATUS_DRAINED, "drained at p0 + 1");
    WR(s, TSR_CSR_DBG_CTRL, TSR_DBG_CTRL_QSTEP);
    tsr_sim_advance(s, 2000);
    CHECK(RD(s, TSR_CSR_Q_DONE(TSR_UNIT_DMA)) == 2 && RD(s, TSR_CSR_DBG_STOP_PIDX) == p0 + 2 && (ST(s) & TSR_DBG_STATUS_DRAINED),
          "QSTEP through the debug-halt hold (dbg_step)");
    CHECK(RD(s, TSR_CSR_STATUS) & TSR_STATUS_RV_HALTED, "the hart is still halted");
    (void)tk;
    // resume: the stop holds the third entry; RELEASE runs it
    WR(s, TSR_CSR_DBG_CTRL, TSR_DBG_CTRL_RELEASE);
    rv_resume(s, via_dm);
    tsr_sim_advance(s, 3000);
    CHECK(RD(s, TSR_CSR_Q_DONE(TSR_UNIT_DMA)) == 3 && !(RD(s, TSR_CSR_STATUS) & TSR_STATUS_RV_HALTED), "resumed, all ran, no re-halt");
    // a TBP with HALT_RV (STOP_HALT_RV off): the matching enqueue halts the hart
    WR(s, TSR_CSR_DBG_CFG, RD(s, TSR_CSR_DBG_CFG) & ~TSR_DBG_CFG_STOP_HALT_RV);
    WR(s, TSR_CSR_DBG_TBP_A(3), TSR_OP_NOP | (0xFFu << 8));
    WR(s, TSR_CSR_DBG_TBP_CTL(3), TSR_DBG_TBP_CTL_EN | (TSR_DBG_TBP_CTL_KIND_OPCODE << TSR_DBG_TBP_CTL_KIND_LSB) |
                                      TSR_DBG_TBP_CTL_HALT_RV);
    const uint32_t pn = tsr_sim_cw_read(s, TSR_CW_ISSUE_PIDX);
    tsr_sim_push(s, tsr_enc_nop().w, NONE, NONE, ALL);
    CHECK(run_until_halt(s, 10000), "TBP HALT_RV halts the hart");
    CHECK((ST(s) & TSR_DBG_STATUS_STOP_ACTIVE) && RD(s, TSR_CSR_DBG_STOP_PIDX) == pn && (TSR_GET(ST(s), TSR_DBG_STATUS_BP_HIT) & 8u),
          "stopped at the NOP by TBP3");
    // the hang-dump helper reads everything through the host port and restores DBG_HIST_IDX
    WR(s, TSR_CSR_DBG_HIST_IDX, 12345);
    FILE * df = tmpfile();
    if (df != NULL) {
        tsr_dbg_dump(s, df, 8);
        CHECK(ftell(df) > 1000 && RD(s, TSR_CSR_DBG_HIST_IDX) == 12345, "tsr_dbg_dump (%ld bytes)", ftell(df));
        fclose(df);
    }
    free(L.v);
    tsr_sim_destroy(s);
}

// The W1a firmware scenario (sim_rv_issue: launches, kernels with tile ops and deps, a TILE_ERR
// kernel with QABORT / QCLEAR, logs, WFI) with the model attached and detached: every observable,
// MTIME, instructions retired, PERF, Q_* and the queue events identical (DB §10.3: HAS_DBG 0 and 1
// give identical results and cycle counts with debugging off).
typedef struct { int detach; evlog L; uint64_t perf[TSR_PERF_SLOTS]; uint32_t q[72]; } fw_ctx;
static void fw_create(tsr_sim * s, void * ctx) {
    fw_ctx * c = (fw_ctx *)ctx;
    if (c->detach) tsr_dbg_detach(s);
    ev_init(&c->L, s);
}
static void fw_end(tsr_sim * s, void * ctx) {
    fw_ctx * c = (fw_ctx *)ctx;
    for (uint32_t i = 0; i < TSR_PERF_SLOTS; ++i) c->perf[i] = tsr_sim_perf(s, i);
    for (uint32_t i = 0; i < 72; ++i) c->q[i] = RD(s, 0xE00 + 4 * i);
}
static void test_fw_comparability(const char * elf, const char * sched) {
    FILE * f = elf ? fopen(elf, "rb") : NULL;
    if (f == NULL) {
        printf("sim_dbg: no firmware image (%s): the firmware comparability run is skipped\n", elf ? elf : "none given");
        return;
    }
    fclose(f);
    static observed a, b;
    static fw_ctx ca, cb;
    memset(&a, 0, sizeof(a)); memset(&b, 0, sizeof(b));
    memset(&ca, 0, sizeof(ca)); memset(&cb, 0, sizeof(cb));
    cb.detach = 1;
    const rvh_run ra = { 0.0, sched, fw_create, fw_end, &ca, NULL }, rb = { 0.0, sched, fw_create, fw_end, &cb, NULL };
    scenario(elf, &ra, &a);
    scenario(elf, &rb, &b);
    CHECK(a.n_rec == b.n_rec && memcmp(a.rec, b.rec, sizeof(a.rec)) == 0 && a.n_log == b.n_log &&
          memcmp(a.log, b.log, sizeof(a.log)) == 0 && a.out_hash == b.out_hash && a.s0 == b.s0, "firmware: observables");
    CHECK(a.mtime == b.mtime && a.instret == b.instret, "firmware: MTIME %llu / %llu, instret %llu / %llu",
          (unsigned long long)a.mtime, (unsigned long long)b.mtime, (unsigned long long)a.instret, (unsigned long long)b.instret);
    CHECK(ca.L.n > 100 && ca.L.n == cb.L.n && ca.L.hash == cb.L.hash, "firmware: queue events (%zu / %zu)", ca.L.n, cb.L.n);
    CHECK(memcmp(ca.perf, cb.perf, sizeof(ca.perf)) == 0 && memcmp(ca.q, cb.q, sizeof(ca.q)) == 0, "firmware: PERF and Q_*");
    free(ca.L.v);
    free(cb.L.v);
}
#endif

// ---------------------------------------------------------------------------
// 13. Comparability (DB §5.8, §10.3): attached and unconfigured, nothing observable changes
// ---------------------------------------------------------------------------
typedef struct { uint64_t ev_hash, t_end; snap_t sn; uint64_t perf[TSR_PERF_SLOTS]; uint32_t q[64]; } obs_t;

static void observe(const prog_t * p, int detach, obs_t * o) {
    tsr_sim * s = mk();
    if (detach) tsr_dbg_detach(s);
    CHECK((tsr_sim_dbg(s) == NULL) == (detach != 0), "model %s", detach ? "detached" : "attached");
    init_mem(s, p->chains);
    evlog L;
    ev_init(&L, s);
    WR(s, TSR_CSR_LQ_PUSH, 0);
    tsr_sim_cw_write(s, TSR_CW_LQ_POP, 0);                              // ACTIVE: PERF counts
    uint32_t tk[MAXOPS], k = 0;
    CHECK(run_prog(s, p, &k, p->n / 2, tk, 50000000ull), "run");
    tsr_sim_cw_write(s, TSR_CW_TRACE_MARK, 0x10000001u);
    CHECK(run_prog(s, p, &k, p->n, tk, 50000000ull), "run");
    o->ev_hash = L.hash;
    o->t_end = tsr_sim_now(s);
    snap(s, p->chains, &o->sn);
    for (uint32_t i = 0; i < TSR_PERF_SLOTS; ++i) o->perf[i] = tsr_sim_perf(s, i);
    for (uint32_t i = 0; i < 64; ++i) o->q[i] = RD(s, 0xE00 + 4 * i);
    free(L.v);
    tsr_sim_destroy(s);
}

static void test_comparability(const prog_t * p) {
    obs_t a, b;
    observe(p, 0, &a);
    observe(p, 1, &b);
    CHECK(a.ev_hash == b.ev_hash && a.t_end == b.t_end, "queue events and times identical");
    CHECK(snap_eq(&a.sn, &b.sn), "memories, completion counts and error records identical");
    CHECK(memcmp(a.perf, b.perf, sizeof(a.perf)) == 0, "PERF identical");
    CHECK(memcmp(a.q, b.q, sizeof(a.q)) == 0, "Q_* identical");
    CHECK(a.perf[TSR_PERF_DBG_STOP_CYCLES] == 0 && a.perf[TSR_PERF_TRW_RECORDS] == 0, "no stop, no trace");
}

// ---------------------------------------------------------------------------
// 14. Format helpers (DB §6.1, §8.2, §8.3)
// ---------------------------------------------------------------------------
static void test_formats(void) {
    const uint32_t h = tsr_log_hdr_make(3, TSR_LOG_HDR_LEVEL_WARN, TSR_LOG_HDR_SOURCE_KERNEL, TSR_LOG_HDR_TYPE_FMT, 1, 0x5A);
    tsr_log_hdr d;
    tsr_log_hdr_decode(h, &d);
    CHECK(d.n == 3 && d.level == 2 && d.source == 1 && d.type == 0 && d.ts == 1 && d.lseq == 0x5A && d.sync == 0xA, "log header");
    uint32_t rec[4] = { h, 1234, 0xF0000010u, 7 }, n = 0;
    CHECK(tsr_log_rec_check(rec, 4, 0xF0000000u, 0xF0001000u, &n) == 0 && n == 4, "valid FMT record");
    CHECK(tsr_log_rec_check(rec, 3, 0, 0, NULL) == TSR_LOG_BAD_LEN, "torn");
    rec[2] = 0x100;
    CHECK(tsr_log_rec_check(rec, 4, 0xF0000000u, 0xF0001000u, NULL) == TSR_LOG_BAD_FMT, "format address outside .tsr_logfmt");
    rec[0] = h & ~TSR_LOG_HDR_SYNC;
    CHECK(tsr_log_rec_check(rec, 4, 0, 0, NULL) == TSR_LOG_BAD_SYNC, "sync");
    rec[0] = tsr_log_hdr_make(0, 6, 0, TSR_LOG_HDR_TYPE_TEXT, 0, 0);
    CHECK(tsr_log_rec_check(rec, 4, 0, 0, NULL) == TSR_LOG_BAD_LEVEL, "level 6");
    rec[0] = tsr_log_hdr_make(0, 1, 0, 5, 0, 0);
    CHECK(tsr_log_rec_check(rec, 4, 0, 0, NULL) == TSR_LOG_BAD_TYPE, "type 5");
    rec[0] = tsr_log_hdr_make(3, 0, TSR_LOG_HDR_SOURCE_TRAP, TSR_LOG_HDR_TYPE_TRAP, 0, 0);
    CHECK(tsr_log_rec_check(rec, 4, 0, 0, NULL) == TSR_LOG_BAD_TRAP, "TRAP needs 44 words");
    uint32_t trap[46] = { tsr_log_hdr_make(45, 0, TSR_LOG_HDR_SOURCE_TRAP, TSR_LOG_HDR_TYPE_TRAP, 1, 3), 99, TSR_CRASH_MAGIC };
    CHECK(tsr_log_rec_check(trap, 46, 0, 0, &n) == 0 && n == 46, "TRAP with TS: 45 payload words");
    rec[0] = tsr_log_hdr_make(2, TSR_LOG_HDR_LEVEL_INFO, TSR_LOG_HDR_SOURCE_RUNTIME, TSR_LOG_HDR_TYPE_EVENT, 0, 1);
    rec[1] = 0x10000001u;
    CHECK(tsr_log_rec_check(rec, 3, 0xF0000000u, 0xF0001000u, &n) == 0 && n == 3, "EVENT");
    rec[0] = tsr_log_hdr_make(1, TSR_LOG_HDR_LEVEL_DEBUG, TSR_LOG_HDR_SOURCE_DEBUG, TSR_LOG_HDR_TYPE_BLOB, 0, 0);
    CHECK(tsr_log_rec_check(rec, 2, 0, 0, &n) == 0 && n == 2, "BLOB");
    rec[0] = tsr_log_hdr_make(0, TSR_LOG_HDR_LEVEL_TRACE, TSR_LOG_HDR_SOURCE_KERNEL, TSR_LOG_HDR_TYPE_FMT, 0, 0);
    CHECK(tsr_log_rec_check(rec, 1, 0, 0, NULL) == TSR_LOG_BAD_FMT, "FMT without its format address");
}

int main(int argc, char ** argv) {
    test_formats();
    test_regs();
    test_dmi_gate();
    prog_t * p = (prog_t *)calloc(1, sizeof(prog_t));
    gen_prog(p, 1234, 12);
    CHECK(p->n >= 48 && p->n < MAXOPS, "program of %u entries", p->n);
    snap_t full;
    uint64_t t_full = 0;
    full_snap(p, &full, &t_full);
    const uint32_t stops[] = { 0, 1, 7, 23, p->n / 2, p->n - 1 };
    for (size_t i = 0; i < sizeof(stops) / 4; ++i) test_stop_at(p, stops[i], &full);
    test_qstep(p, 11);
    test_qstep_as_stop_now();
    test_reattach();
    test_serial(p, &full, t_full);
    test_tbp(p);
    test_history();
    test_rec(p);
    test_trw(p);
    test_err_stop();
    test_qabort_soft_reset(p);
    test_stop_cycles();
#if HAVE_RV
    {
        tsr_sim * s = mk();
        const int dm = have_dm(s);
        tsr_sim_destroy(s);
        for (int via = dm; via >= 0; --via) {
            test_rvwp(via);
            test_halt_on_boot(via);
            test_stop_halt_rv(via);
        }
        test_halt_time();
        wait_freeze_run(1);
        wait_freeze_run(0);
        test_fw_comparability(argc > 1 ? argv[1] : NULL, "fixed");
        test_fw_comparability(argc > 1 ? argv[1] : NULL, "random:7");
    }
#endif
    (void)argc;
    (void)argv;
    test_comparability(p);
    // Invariants G, Q, S under a random schedule (durations x [0.25, 4], crossings 2-8)
    g_sched = "random:7";
    test_stop_at(p, 7, &full);
    test_stop_at(p, p->n / 2, &full);
    test_qstep(p, 13);
    test_serial(p, &full, t_full);
    test_comparability(p);
    g_sched = NULL;
    free(p);
    return test_finish("sim_dbg");
}
