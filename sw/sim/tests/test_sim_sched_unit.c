// test_sim_sched_unit.c — P1.8 unit tests (ctest sim_sched_unit; docs/v2/06-software-verification.md
// "SW" §4.4, §4.7-§4.9; the interface is sw/sim/tsr_sim_int.h §5 / §5b):
//   1. xoshiro128** against the reference vectors (state {1, 2, 3, 4}) and the splitmix64 stream
//      seeding pinned (a seed must reproduce the same schedule in every version);
//   2. TSR_SIM_SCHED parsing and the canonical form;
//   3. options from tsr_sim_opts and from the environment (TSR_SIM_SCHED / _RACE / _LOG / TSR_TRACE),
//      refused values;
//   4. the listener list (registration order, removal, inside a callback, the set_listener form,
//      the limit) and the PUSH / QCTRL / MARK_PUSH events;
//   5. tie-breaking at one MTIME: completions in unit order, then the dispatch, then head starts in
//      unit order (stub handlers with chosen durations);
//   6. random durations (x [min, max] both bounds reached, >= 1, rounded half up, exact for factor 1)
//      and crossings (2-8; 2 when fixed),
//      same seed same trace, other seed other trace;
//   7. the CDC lag of the RISC-V-side copies (random): DONE_SEQ never leads and trails by 2-8
//      cycles, the error record is visible no later than the DONE_SEQ that passes it (I2),
//      ERR_IRQ, credits; fixed reads the core state;
//   8. access events (host CSR, host DRAM API, control window), model slots;
//   9. TSR_SIM_LOG lines (queue events, RV retire lines) and their determinism;
//  10. ndmreset (STATUS RV_RESET_DONE / RV_RUNNING, QABORT, restart);
//  11. the hart under the schedule: the random CPI delta (-1 / 0 / +1, both seen, cost >= 1), the
//      spin rule (3 identical reads no jump, the 4th jumps exactly to the next core event).
// Usage: sim_sched_unit <scratch dir>.
#include "test_sim_util.h"
#include "tsr_sim_int.h"

#include <stdarg.h>

static const char * g_scratch = ".";

// ---------------------------------------------------------------------------
// Stub handlers: LD (DMA), QACT (QACT), MVM (MVM) with no operands; cycles = w[1] (>= 1). The
// process registers them before its first simulator (tsr_sim_int.h §2), so every op is a pure
// timing object here.
// ---------------------------------------------------------------------------
static uint32_t stub_check(tsr_sim * sim, const uint32_t w[8], tsr_extents * ext, uint32_t * info) {
    (void)sim; (void)w; (void)ext; (void)info;
    return TSR_ERR_NONE;
}
static void stub_exec(tsr_sim * sim, const uint32_t w[8]) { (void)sim; (void)w; }
static uint64_t stub_cycles(const tsr_sim * sim, const uint32_t w[8]) { (void)sim; return w[1] ? w[1] : 1u; }

static void register_stubs(void) {
    const tsr_op_handler ld = { TSR_UNIT_DMA, stub_check, stub_exec, stub_cycles };
    const tsr_op_handler qa = { TSR_UNIT_QACT, stub_check, stub_exec, stub_cycles };
    const tsr_op_handler mv = { TSR_UNIT_MVM, stub_check, stub_exec, stub_cycles };
    tsr_sim_register_op(TSR_OP_LD, &ld);
    tsr_sim_register_op(TSR_OP_QACT, &qa);
    tsr_sim_register_op(TSR_OP_MVM, &mv);
}

static uint32_t op(tsr_sim * s, uint32_t opcode, uint32_t cycles, uint32_t dep0, uint32_t dep1) {
    const uint32_t w[8] = { opcode, cycles, 0, 0, 0, 0, 0, 0 };
    return tsr_sim_push(s, w, dep0, dep1, TSR_ISSUE_PUSH_WMASK);
}

static tsr_sim * mk(const char * sched) {
    tsr_config cfg = TSR_CONFIG_U200_V1;
    cfg.num_ch = 2;
    cfg.ch_bytes = 1ull << 24;
    tsr_sim_opts o;
    memset(&o, 0, sizeof(o));
    o.sched = sched ? sched : "fixed";
    tsr_sim * s = tsr_sim_create_opts(&cfg, &o);
    if (s == NULL) {
        fprintf(stderr, "create failed (%s)\n", o.sched);
        exit(1);
    }
    return s;
}

// Event recorder.
typedef struct evrec { uint32_t kind, unit, ticket, pidx, code, ran, w0, w1, push, pc, dep0, dep1; uint64_t time; } evrec;
typedef struct evlog { evrec v[4096]; uint32_t n; } evlog;
static void ev_cb(void * ctx, tsr_sim * sim, const tsr_q_event * ev) {
    (void)sim;
    evlog * l = (evlog *)ctx;
    if (l->n >= 4096u) return;
    evrec * r = &l->v[l->n++];
    r->kind = ev->kind; r->unit = ev->unit; r->ticket = ev->ticket; r->pidx = ev->pidx; r->code = ev->code;
    r->ran = ev->ran; r->w0 = ev->w ? ev->w[0] : 0; r->w1 = ev->w ? ev->w[1] : 0; r->push = ev->push; r->pc = ev->pc;
    r->dep0 = ev->dep[0]; r->dep1 = ev->dep[1]; r->time = ev->time;
}

// ---------------------------------------------------------------------------
// 1. RNG
// ---------------------------------------------------------------------------
static void test_rng(void) {
    // xoshiro128** reference outputs from state {1, 2, 3, 4} (the reference C implementation; the
    // same vector is in rand_xoshiro's Xoshiro128StarStar test).
    static const uint32_t ref[10] = { 11520u, 0u, 5927040u, 70819200u, 2031721883u, 1637235492u, 1287239034u,
                                      3734860849u, 3729100597u, 4258142804u };
    tsr_rng r = { { 1u, 2u, 3u, 4u } };
    for (int i = 0; i < 10; ++i) {
        const uint32_t v = tsr_rng_next(&r);
        CHECK(v == ref[i], "xoshiro128** output %d: %u, reference %u", i, v, ref[i]);
    }
    // Stream seeding (tsr_sched.c): splitmix64 from seed + (stream + 1) * 0x632BE59BD9B4E019.
    tsr_rng_seed(&r, 7u, 0u);
    CHECK(r.s[0] == 0x378fa476u && r.s[1] == 0xdb5e8bdeu && r.s[2] == 0xc290ae44u && r.s[3] == 0x29363312u,
          "seed 7 stream 0 state %08x %08x %08x %08x", r.s[0], r.s[1], r.s[2], r.s[3]);
    static const uint32_t s7[4] = { 3477800004u, 3253420555u, 3469757153u, 2337741503u };
    for (int i = 0; i < 4; ++i) CHECK(tsr_rng_next(&r) == s7[i], "seed 7 stream 0 output %d", i);
    tsr_rng_seed(&r, 0x1234567890ABCDEFull, 9u);
    static const uint32_t s9[4] = { 3314149379u, 3396136536u, 228287818u, 3623191406u };
    for (int i = 0; i < 4; ++i) CHECK(tsr_rng_next(&r) == s9[i], "seed 0x1234567890ABCDEF stream 9 output %d", i);
    // Streams of one seed are distinct; below() stays in range and reaches every value.
    uint32_t first[TSR_RS_COUNT];
    for (uint32_t k = 0; k < TSR_RS_COUNT; ++k) {
        tsr_rng_seed(&r, 5u, k);
        first[k] = tsr_rng_next(&r);
        for (uint32_t j = 0; j < k; ++j) CHECK(first[j] != first[k], "streams %u and %u start alike", j, k);
    }
    uint32_t seen = 0;
    tsr_rng_seed(&r, 99u, 3u);
    for (int i = 0; i < 1000; ++i) {
        const uint32_t v = tsr_rng_below(&r, 7u);
        CHECK(v < 7u, "below(7) = %u", v);
        seen |= 1u << v;
        CHECK(tsr_rng_below(&r, 1u) == 0u, "below(1)");
    }
    CHECK(seen == 0x7Fu, "below(7) reaches 0..6 (0x%x)", seen);
}

// ---------------------------------------------------------------------------
// 2. TSR_SIM_SCHED parsing
// ---------------------------------------------------------------------------
static void test_parse(void) {
    tsr_sched_spec s;
    char err[160], buf[80];
    CHECK(tsr_sched_parse(NULL, &s, err, sizeof(err)) == 0 && !s.random, "NULL = fixed");
    CHECK(tsr_sched_parse("", &s, err, sizeof(err)) == 0 && !s.random, "\"\" = fixed");
    CHECK(tsr_sched_parse("fixed", &s, err, sizeof(err)) == 0 && !s.random, "fixed");
    tsr_sched_format(&s, buf, sizeof(buf));
    CHECK(strcmp(buf, "fixed") == 0, "format fixed: %s", buf);
    CHECK(tsr_sched_parse("random:7", &s, err, sizeof(err)) == 0 && s.random && s.seed == 7u &&
          s.fmin == 16384u && s.fmax == 262144u, "random:7 defaults 0.25-4");
    tsr_sched_format(&s, buf, sizeof(buf));
    CHECK(strcmp(buf, "random:7,0.25,4") == 0, "canonical %s", buf);
    CHECK(tsr_sched_parse("random:0xFFFFFFFFFFFFFFFF,0.5,2", &s, err, sizeof(err)) == 0 && s.seed == UINT64_MAX &&
          s.fmin == 32768u && s.fmax == 131072u, "hex seed, bounds");
    tsr_sched_format(&s, buf, sizeof(buf));
    CHECK(strcmp(buf, "random:18446744073709551615,0.5,2") == 0, "canonical %s", buf);
    CHECK(tsr_sched_parse("random:3,1,1", &s, err, sizeof(err)) == 0 && s.fmin == 65536u && s.fmax == 65536u, "factor 1");
    // A leading 0 is decimal, never octal; hex needs 0x / 0X.
    CHECK(tsr_sched_parse("random:08", &s, err, sizeof(err)) == 0 && s.seed == 8u, "random:08 = seed 8 (%s)", err);
    CHECK(tsr_sched_parse("random:010", &s, err, sizeof(err)) == 0 && s.seed == 10u, "random:010 = seed 10");
    CHECK(tsr_sched_parse("random:0X1f,1,2", &s, err, sizeof(err)) == 0 && s.seed == 31u, "random:0X1f = seed 31");
    CHECK(tsr_sched_parse("random:18446744073709551615", &s, err, sizeof(err)) == 0 && s.seed == UINT64_MAX, "2^64 - 1");
    // The canonical form parses back to the same schedule (bounds not representable in decimal).
    tsr_sched_spec s2;
    CHECK(tsr_sched_parse("random:5,0.3,1000.7", &s, err, sizeof(err)) == 0, "random:5,0.3,1000.7");
    tsr_sched_format(&s, buf, sizeof(buf));
    CHECK(tsr_sched_parse(buf, &s2, err, sizeof(err)) == 0 && s2.random && s2.seed == s.seed && s2.fmin == s.fmin &&
          s2.fmax == s.fmax, "round trip of %s", buf);
    static const char * const bad[] = {
        "random", "random:", "random:-1", "random:+1", "random:1,", "random:1,0.5", "random:1,0,1", "random:1,2,1",
        "random:1,0.5,2000", "random:1,nan,2", "random:x", "random:1x", "Fixed", "fixed ", "random:1,0.5,2,3",
        "random:99999999999999999999", "random:18446744073709551616", "random:0x", "random:0x0x5", "random:0x-5",
        "random:0x10000000000000000", "random:1 ", "random:0b1",
    };
    for (size_t i = 0; i < sizeof(bad) / sizeof(bad[0]); ++i) {
        err[0] = '\0';
        CHECK(tsr_sched_parse(bad[i], &s, err, sizeof(err)) != 0 && err[0] != '\0', "\"%s\" refused", bad[i]);
    }
}

// ---------------------------------------------------------------------------
// 3. Options and environment (SW §4.8, §4.9)
// ---------------------------------------------------------------------------
static void test_opts(void) {
    tsr_config cfg = TSR_CONFIG_U200_V1;
    cfg.num_ch = 1;
    cfg.ch_bytes = 1ull << 20;
    tsr_sim_opts o;
    memset(&o, 0, sizeof(o));
    unsetenv("TSR_SIM_SCHED");
    unsetenv("TSR_SIM_RACE");
    unsetenv("TSR_SIM_LOG");
    unsetenv("TSR_TRACE");
    tsr_sim * s = tsr_sim_create_opts(&cfg, &o);
    CHECK(s && strcmp(tsr_sim_opt_sched(s), "fixed") == 0 && !tsr_sim_sched_random(s) && tsr_sim_opt_race(s) == 0 &&
          tsr_sim_opt_trace_dir(s) == NULL && tsr_sim_fast_forward(s), "defaults");
    tsr_sim_destroy(s);
    setenv("TSR_SIM_SCHED", "random:42,0.5,3", 1);
    setenv("TSR_TRACE", "/nonexistent/trace-dir", 1);
    s = tsr_sim_create_opts(&cfg, &o);
    CHECK(s && strcmp(tsr_sim_opt_sched(s), "random:42,0.5,3") == 0 && tsr_sim_sched_random(s), "TSR_SIM_SCHED read");
    CHECK(s && tsr_sim_opt_trace_dir(s) && strcmp(tsr_sim_opt_trace_dir(s), "/nonexistent/trace-dir") == 0, "TSR_TRACE read");
    tsr_sim_destroy(s);
    o.sched = "fixed";                                   // the option wins over the environment
    o.trace_dir = "/other";
    s = tsr_sim_create_opts(&cfg, &o);
    CHECK(s && !tsr_sim_sched_random(s) && strcmp(tsr_sim_opt_trace_dir(s), "/other") == 0, "opts win");
    tsr_sim_destroy(s);
    o.sched = "random:1,4,0.5";
    CHECK(tsr_sim_create_opts(&cfg, &o) == NULL, "bad sched refused");
    o.sched = NULL;
    setenv("TSR_SIM_SCHED", "bogus", 1);
    CHECK(tsr_sim_create_opts(&cfg, &o) == NULL, "bad TSR_SIM_SCHED refused");
    unsetenv("TSR_SIM_SCHED");
    unsetenv("TSR_TRACE");
    o.trace_dir = NULL;
    setenv("TSR_SIM_RACE", "sometimes", 1);
    CHECK(tsr_sim_create_opts(&cfg, &o) == NULL, "bad TSR_SIM_RACE refused");
    unsetenv("TSR_SIM_RACE");
    o.race = 3;
    CHECK(tsr_sim_create_opts(&cfg, &o) == NULL, "race 3 refused");
    o.race = 0;
    o.log_flags = 2;
    CHECK(tsr_sim_create_opts(&cfg, &o) == NULL, "unknown log flag refused");
    o.log_flags = 0;
    setenv("TSR_SIM_LOG", "/nonexistent-dir/x.log", 1);
    CHECK(tsr_sim_create_opts(&cfg, &o) == NULL, "unopenable TSR_SIM_LOG refused");
    unsetenv("TSR_SIM_LOG");
    // The switch and the model slots.
    s = tsr_sim_create_opts(&cfg, &o);
    tsr_sim_set_fast_forward(s, 0);
    CHECK(!tsr_sim_fast_forward(s), "fast-forward off");
    void ** sl[TSR_MODEL_COUNT];
    for (int i = 0; i < TSR_MODEL_COUNT; ++i) {
        sl[i] = tsr_sim_model_slot(s, i);
        CHECK(sl[i] != NULL, "slot %d", i);
        for (int j = 0; j < i; ++j) CHECK(sl[i] != sl[j], "slots %d / %d distinct", i, j);
    }
    CHECK(tsr_sim_model_slot(s, -1) == NULL && tsr_sim_model_slot(s, TSR_MODEL_COUNT) == NULL, "bad slot ids");
    tsr_sim_destroy(s);
}

// ---------------------------------------------------------------------------
// 4. Listener list, PUSH / QCTRL / MARK_PUSH
// ---------------------------------------------------------------------------
typedef struct order_ctx { int tag; int * out; int * n; } order_ctx;
static int g_rm_id = -1;
static void order_cb(void * ctx, tsr_sim * sim, const tsr_q_event * ev) {
    order_ctx * c = (order_ctx *)ctx;
    if (ev->kind != TSR_QEV_PUSH) return;
    c->out[(*c->n)++] = c->tag;
    if (c->tag == 100 && g_rm_id >= 0) {                 // removes a later listener during the event
        tsr_sim_remove_listener(sim, g_rm_id);
        g_rm_id = -1;
    }
}

static void test_listeners(void) {
    tsr_sim * s = mk(NULL);
    int out[64], n = 0;
    order_ctx c[20];
    for (int i = 0; i < 20; ++i) { c[i].tag = i; c[i].out = out; c[i].n = &n; }
    const int id0 = tsr_sim_add_listener(s, order_cb, &c[0]);
    const int id1 = tsr_sim_add_listener(s, order_cb, &c[1]);
    const int id2 = tsr_sim_add_listener(s, order_cb, &c[2]);
    CHECK(id0 >= 0 && id1 >= 0 && id2 >= 0 && id0 != id1 && id1 != id2, "ids");
    CHECK(tsr_sim_add_listener(s, NULL, NULL) < 0, "NULL fn refused");
    op(s, TSR_OP_LD, 5, TSR_TICKET_NONE, TSR_TICKET_NONE);
    CHECK(n == 3 && out[0] == 0 && out[1] == 1 && out[2] == 2, "registration order");
    tsr_sim_remove_listener(s, id1);
    tsr_sim_remove_listener(s, 12345);                    // unknown: ignored
    n = 0;
    op(s, TSR_OP_LD, 5, TSR_TICKET_NONE, TSR_TICKET_NONE);
    CHECK(n == 2 && out[0] == 0 && out[1] == 2, "after removal");
    // The W1a form owns one entry: appended, replaced in place for the same fn, removed by NULL.
    tsr_sim_set_listener(s, order_cb, &c[3]);
    n = 0;
    op(s, TSR_OP_LD, 5, TSR_TICKET_NONE, TSR_TICKET_NONE);
    CHECK(n == 3 && out[2] == 3, "set_listener appended");
    tsr_sim_set_listener(s, order_cb, &c[4]);
    n = 0;
    op(s, TSR_OP_LD, 5, TSR_TICKET_NONE, TSR_TICKET_NONE);
    CHECK(n == 3 && out[2] == 4, "set_listener: new context in place");
    tsr_sim_set_listener(s, NULL, NULL);
    n = 0;
    op(s, TSR_OP_LD, 5, TSR_TICKET_NONE, TSR_TICKET_NONE);
    CHECK(n == 2, "set_listener NULL removes it");
    // Removal from inside a callback: the removed listener is not called for the same event.
    order_ctx rm = { 100, out, &n };
    tsr_sim_remove_listener(s, id0);
    tsr_sim_remove_listener(s, id2);
    tsr_sim_add_listener(s, order_cb, &rm);
    g_rm_id = tsr_sim_add_listener(s, order_cb, &c[5]);
    n = 0;
    op(s, TSR_OP_LD, 5, TSR_TICKET_NONE, TSR_TICKET_NONE);
    CHECK(n == 1 && out[0] == 100, "removed during the event: not called (%d calls)", n);
    // Capacity (models attached at create may hold entries too): the list fills up before
    // TSR_SIM_LISTENERS_MAX more, and a removal makes room for exactly one.
    int full = -1, last = -1;
    for (int i = 0; i < 40; ++i) {
        const int id = tsr_sim_add_listener(s, order_cb, &c[6]);
        if (id < 0) { full = i; break; }
        last = id;
    }
    CHECK(full >= 0 && full <= (int)TSR_SIM_LISTENERS_MAX - 1, "list full after %d more", full);
    tsr_sim_remove_listener(s, last);
    CHECK(tsr_sim_add_listener(s, order_cb, &c[6]) >= 0 && tsr_sim_add_listener(s, order_cb, &c[6]) < 0, "one free entry");
    tsr_sim_destroy(s);

    // Event content: PUSH (masked words, deps and value as pushed, REC), ENQ (BAD_DEP dep reads
    // none), QCTRL (one event per write, QCLEAR ticket), MARK_PUSH / MARK.
    s = mk(NULL);
    static evlog L;
    memset(&L, 0, sizeof(L));
    tsr_sim_add_listener(s, ev_cb, &L);
    const uint32_t t0 = op(s, TSR_OP_LD, 7, TSR_TICKET_NONE, TSR_TICKET_NONE);
    const uint32_t w[8] = { TSR_OP_QACT, 3, 0xAA, 0xBB, 0, 0, 0, 0 };
    const uint32_t bad_dep = TSR_PUT(TSR_TICKET_UNIT, TSR_UNIT_MVM) | 5u;   // never pushed: BAD_DEP
    const uint32_t pv = (0x03u << 8) | TSR_ISSUE_PUSH_REC;                  // words 0, 1 only; REC
    const uint32_t t1 = tsr_sim_push(s, w, t0, bad_dep, pv);
    const uint32_t tq = tsr_sim_qctrl(s, TSR_QCTRL_QHOLD_SET | TSR_QCTRL_QHOLD_CLR | TSR_QCTRL_QCLEAR);
    tsr_sim_cw_write(s, TSR_CW_TRACE_MARK, 0x5151u);
    tsr_sim_run_until_idle(s, 100000);
    int got_push = 0, got_enq = 0, got_q = 0, got_mp = 0, got_m = 0;
    for (uint32_t i = 0; i < L.n; ++i) {
        const evrec * e = &L.v[i];
        if (e->kind == TSR_QEV_PUSH && e->ticket == t1) {
            got_push = e->w0 == TSR_OP_QACT && e->w1 == 3 && e->dep0 == t0 && e->dep1 == bad_dep && e->push == pv &&
                       e->unit == TSR_UNIT_QACT && e->time == 0 && e->pc == TSR_SIM_PC_HOOK;
        }
        if (e->kind == TSR_QEV_ENQ && e->ticket == t1) got_enq = e->code == TSR_ERR_BAD_DEP && e->dep1 == TSR_TICKET_NONE && e->push == pv;
        if (e->kind == TSR_QEV_QCTRL)
            got_q = e->w0 == (TSR_QCTRL_QHOLD_SET | TSR_QCTRL_QHOLD_CLR | TSR_QCTRL_QCLEAR) && e->ticket == tq &&
                    e->unit == TSR_UNIT_SYNC && e->pidx == 2u;
        if (e->kind == TSR_QEV_MARK_PUSH) got_mp = e->w0 == 0x5151u && e->unit == TSR_UNIT_MARKER && e->pidx == 3u;
        if (e->kind == TSR_QEV_MARK) got_m = e->w1 == 0x5151u;
    }
    CHECK(got_push, "PUSH event: words, deps, value as pushed");
    CHECK(got_enq, "ENQ: BAD_DEP, the dep reads none");
    CHECK(got_q, "QCTRL event: value, QCLEAR ticket and push index");
    CHECK(got_mp && got_m, "MARK_PUSH (w0 = value) and MARK (w1 = value)");
    // Every PUSH / QCTRL / MARK_PUSH precedes the dispatch of its entry, in push order.
    tsr_sim_destroy(s);
}

// ---------------------------------------------------------------------------
// 5. Tie-breaking at one MTIME (fixed schedule)
// ---------------------------------------------------------------------------
static void test_ties(void) {
    tsr_sim * s = mk(NULL);
    static evlog L;
    memset(&L, 0, sizeof(L));
    tsr_sim_add_listener(s, ev_cb, &L);
    // A (DMA, 100) dispatched at 2, B (MVM, 99) at 3: both done at 102. C (QACT, dep B) starts at
    // 102; D (DMA, queued behind A) starts at 102; E (DMA) pushed at 100 is dispatched at 102.
    const uint32_t tA = op(s, TSR_OP_LD, 100, TSR_TICKET_NONE, TSR_TICKET_NONE);
    const uint32_t tB = op(s, TSR_OP_MVM, 99, TSR_TICKET_NONE, TSR_TICKET_NONE);
    const uint32_t tC = op(s, TSR_OP_QACT, 10, tB, TSR_TICKET_NONE);
    const uint32_t tD = op(s, TSR_OP_LD, 5, TSR_TICKET_NONE, TSR_TICKET_NONE);
    tsr_sim_advance(s, 100);
    CHECK(tsr_sim_now(s) == 100, "at 100");
    const uint32_t tE = op(s, TSR_OP_LD, 1, TSR_TICKET_NONE, TSR_TICKET_NONE);
    tsr_sim_run_until_idle(s, 100000);
    uint32_t seq[16], n = 0;
    for (uint32_t i = 0; i < L.n && n < 16; ++i)
        if (L.v[i].time == 102 && L.v[i].kind != TSR_QEV_PUSH) seq[n++] = i;
    CHECK(n == 5, "%u events at MTIME 102", n);
    if (n == 5) {
        const evrec * e = L.v;
        CHECK(e[seq[0]].kind == TSR_QEV_DONE && e[seq[0]].ticket == tA, "1st: DONE A (unit 0)");
        CHECK(e[seq[1]].kind == TSR_QEV_DONE && e[seq[1]].ticket == tB, "2nd: DONE B (unit 2)");
        CHECK(e[seq[2]].kind == TSR_QEV_ENQ && e[seq[2]].ticket == tE, "3rd: dispatch of E");
        CHECK(e[seq[3]].kind == TSR_QEV_START && e[seq[3]].ticket == tD, "4th: START D (unit 0)");
        CHECK(e[seq[4]].kind == TSR_QEV_START && e[seq[4]].ticket == tC, "5th: START C (unit 1)");
    }
    tsr_sim_destroy(s);
}

// ---------------------------------------------------------------------------
// 6. Random durations and crossings
// ---------------------------------------------------------------------------
typedef struct dstat { uint64_t dmin, dmax, xmin, xmax; uint32_t n_dur, n_x, below, above; uint64_t hash; } dstat;

// 120 ops of `base` cycles on three units, each pushed after the previous completed (so the FIFO is
// empty and the crossing is visible as ENQ - PUSH).
static void run_series(const char * sched, uint32_t base, dstat * st) {
    tsr_sim * s = mk(sched);
    static evlog L;
    memset(&L, 0, sizeof(L));
    tsr_sim_add_listener(s, ev_cb, &L);
    static const uint32_t ops3[3] = { TSR_OP_LD, TSR_OP_QACT, TSR_OP_MVM };
    for (int i = 0; i < 120; ++i) {
        op(s, ops3[i % 3], base, TSR_TICKET_NONE, TSR_TICKET_NONE);
        tsr_sim_run_until_idle(s, 1000000);
        tsr_sim_advance(s, 10);
    }
    memset(st, 0, sizeof(*st));
    st->dmin = st->xmin = UINT64_MAX;
    uint64_t h = 1469598103934665603ull;
    for (uint32_t i = 0; i < L.n; ++i) {
        const evrec * e = &L.v[i];
        h = (h ^ e->time) * 1099511628211ull;
        h = (h ^ ((uint64_t)e->kind << 32 | e->ticket)) * 1099511628211ull;
        if (e->kind == TSR_QEV_START) {
            for (uint32_t j = i + 1; j < L.n; ++j) {
                if (L.v[j].kind == TSR_QEV_DONE && L.v[j].ticket == e->ticket) {
                    const uint64_t d = L.v[j].time - e->time;
                    if (d < st->dmin) st->dmin = d;
                    if (d > st->dmax) st->dmax = d;
                    st->below += d < base;
                    st->above += d > base;
                    st->n_dur++;
                    break;
                }
            }
        }
        if (e->kind == TSR_QEV_PUSH) {
            for (uint32_t j = i + 1; j < L.n; ++j) {
                if (L.v[j].kind == TSR_QEV_ENQ && L.v[j].ticket == e->ticket) {
                    const uint64_t x = L.v[j].time - e->time;
                    if (x < st->xmin) st->xmin = x;
                    if (x > st->xmax) st->xmax = x;
                    st->n_x++;
                    break;
                }
            }
        }
    }
    st->hash = h;
    tsr_sim_destroy(s);
}

static void test_random_timing(void) {
    dstat f, r, r2, r3, one, tiny;
    run_series(NULL, 1000, &f);
    CHECK(f.n_dur == 120 && f.dmin == 1000 && f.dmax == 1000 && f.xmin == 2 && f.xmax == 2, "fixed: durations 1000, crossings 2");
    run_series("random:11", 1000, &r);
    CHECK(r.n_dur == 120 && r.dmin >= 250 && r.dmax <= 4000, "random: durations in [250, 4000] (%llu..%llu)",
          (unsigned long long)r.dmin, (unsigned long long)r.dmax);
    CHECK(r.below > 10 && r.above > 10, "random: shorter and longer than the cycle model (%u / %u)", r.below, r.above);
    CHECK(r.xmin == 2 && r.xmax == 8 && r.n_x == 120, "random: crossings 2..8 (%llu..%llu)", (unsigned long long)r.xmin,
          (unsigned long long)r.xmax);
    run_series("random:11", 1000, &r2);
    CHECK(r2.hash == r.hash, "same seed: same event trace");
    run_series("random:12", 1000, &r3);
    CHECK(r3.hash != r.hash, "other seed: other trace");
    run_series("random:5,1,1", 1000, &one);
    CHECK(one.dmin == 1000 && one.dmax == 1000, "factor 1: durations exact (%llu..%llu)", (unsigned long long)one.dmin,
          (unsigned long long)one.dmax);
    run_series("random:6,0.25,0.25", 1, &tiny);
    CHECK(tiny.dmin == 1 && tiny.dmax == 1, "round(0.25) -> 1 cycle minimum");
    // Rounding is half up (3 x 0.5 = 1.5 -> 2), and f is uniform over [min, max] in Q16.16 steps
    // with both bounds reached (65536 x [1, 1 + 2^-16]: 65536 or 65537 cycles).
    dstat half, top;
    run_series("random:7,0.5,0.5", 3, &half);
    CHECK(half.dmin == 2 && half.dmax == 2, "round(1.5) = 2 (%llu..%llu)", (unsigned long long)half.dmin,
          (unsigned long long)half.dmax);
    run_series("random:8,1,1.0000152587890625", 65536, &top);
    CHECK(top.dmin == 65536 && top.dmax == 65537 && top.below == 0, "factor bounds [min, max] both reached (%llu..%llu)",
          (unsigned long long)top.dmin, (unsigned long long)top.dmax);
    printf("sim_sched_unit: random:11 durations %llu..%llu (%u below, %u above 1000), crossings %llu..%llu\n",
           (unsigned long long)r.dmin, (unsigned long long)r.dmax, r.below, r.above, (unsigned long long)r.xmin,
           (unsigned long long)r.xmax);
}

// ---------------------------------------------------------------------------
// 7. CDC lag of the RISC-V-side copies
// ---------------------------------------------------------------------------
static void test_lag(void) {
    // Fixed: the control window reads the core state.
    tsr_sim * s = mk(NULL);
    op(s, TSR_OP_MVM, 50, TSR_TICKET_NONE, TSR_TICKET_NONE);
    tsr_sim_run_until_idle(s, 100000);
    CHECK(tsr_sim_cw_read(s, TSR_CW_DONE_SEQ(TSR_UNIT_MVM)) == 1u, "fixed: DONE_SEQ at once");
    tsr_sim_destroy(s);

    for (uint32_t seed = 1; seed <= 6; ++seed) {
        char spec[32];
        snprintf(spec, sizeof(spec), "random:%u", seed);
        s = mk(spec);
        // An op, then a poisoned NOP on SYNC (ILLEGAL_OP): its error record and DONE_SEQ.
        op(s, TSR_OP_MVM, 50, TSR_TICKET_NONE, TSR_TICKET_NONE);
        const tsr_insn bad = insn_raw(TSR_OP_NOP, 1, 0, 0, 0, 0, 0, 0);
        tsr_sim_push(s, bad.w, TSR_TICKET_NONE, TSR_TICKET_NONE, TSR_ISSUE_PUSH_WMASK);
        uint64_t t_core = UINT64_MAX, t_rv = UINT64_MAX, t_err_core = UINT64_MAX, t_irq = UINT64_MAX;
        int leads = 0, i2 = 0, credit_over = 0;
        for (int i = 0; i < 4000 && t_rv == UINT64_MAX; ++i) {
            tsr_sim_advance(s, 1);
            const uint64_t now = tsr_sim_now(s);
            const uint32_t d_core = tsr_sim_csr_read(s, TSR_CSR_Q_DONE(TSR_UNIT_MVM));
            const uint32_t d_rv = tsr_sim_cw_read(s, TSR_CW_DONE_SEQ(TSR_UNIT_MVM));
            if (d_rv > d_core) leads = 1;
            if (d_core == 1u && t_core == UINT64_MAX) t_core = now;
            if (d_rv == 1u && t_rv == UINT64_MAX) t_rv = now;
            const uint32_t ds_core = tsr_sim_csr_read(s, TSR_CSR_Q_DONE(TSR_UNIT_SYNC));
            const uint32_t ds_rv = tsr_sim_cw_read(s, TSR_CW_DONE_SEQ(TSR_UNIT_SYNC));
            if (ds_rv > ds_core) leads = 1;
            if (tsr_sim_csr_read(s, TSR_CSR_Q_ERR_CODE(TSR_UNIT_SYNC)) != 0u && t_err_core == UINT64_MAX) t_err_core = now;
            // I2 on the RISC-V side: DONE_SEQ past ERR_SEQ implies the record is visible.
            if (ds_rv >= 1u && tsr_sim_cw_read(s, TSR_CW_ERR_CODE(TSR_UNIT_SYNC)) != TSR_ERR_ILLEGAL_OP) i2 = 1;
            if ((tsr_sim_cw_read(s, TSR_CW_ISSUE_STATUS) & TSR_ISSUE_STATUS_ERR_IRQ) && t_irq == UINT64_MAX) t_irq = now;
            const uint32_t cr = tsr_sim_cw_read(s, TSR_CW_ISSUE_CREDIT);
            for (uint32_t u = 0; u < TSR_NUM_UNITS; ++u) {
                const uint32_t depth = (tsr_sim_csr_read(s, TSR_CSR_Q_DEPTH) >> (TSR_Q_DEPTH_FIELD_W * u)) & 0x1Fu;
                if (((cr >> (TSR_CREDIT_FIELD_W * u)) & 0xFu) > depth) credit_over = 1;
            }
        }
        CHECK(!leads, "%s: DONE_SEQ never leads the core", spec);
        CHECK(!i2, "%s: I2 on the RISC-V side", spec);
        CHECK(!credit_over, "%s: credits within the depths", spec);
        CHECK(t_core != UINT64_MAX && t_rv != UINT64_MAX && t_rv >= t_core + 2 && t_rv <= t_core + 8,
              "%s: DONE_SEQ visible %llu cycles after the core (2..8)", spec, (unsigned long long)(t_rv - t_core));
        CHECK(t_irq != UINT64_MAX && t_irq >= t_err_core + 2, "%s: ERR_IRQ seen after its crossing", spec);
        tsr_sim_destroy(s);
    }
}

// ---------------------------------------------------------------------------
// 8. Access events
// ---------------------------------------------------------------------------
typedef struct acclog { tsr_sim_acc v[256]; uint32_t n; } acclog;
static void acc_cb(void * ctx, tsr_sim * sim, const tsr_sim_acc * a) {
    (void)sim;
    acclog * l = (acclog *)ctx;
    if (l->n < 256u) l->v[l->n++] = *a;
}
static const tsr_sim_acc * acc_find_kind(const acclog * l, uint32_t kind, uint32_t off) {
    for (uint32_t i = 0; i < l->n; ++i)
        if (l->v[i].kind == kind && (off == ~0u || l->v[i].off == off)) return &l->v[i];
    return NULL;
}

static void test_acc(void) {
    tsr_sim * s = mk(NULL);
    static acclog L;
    memset(&L, 0, sizeof(L));
    const int listened0 = tsr_sim_acc_listened(s);   // models attached at create may listen
    const int id = tsr_sim_add_acc_listener(s, acc_cb, &L);
    CHECK(id >= 0 && tsr_sim_acc_listened(s), "listener");
    tsr_sim_advance(s, 7);
    tsr_sim_csr_write(s, TSR_CSR_LQ_PUSH, 1);
    const uint32_t st = tsr_sim_csr_read(s, TSR_CSR_STATUS);
    const tsr_sim_acc * a = acc_find_kind(&L, TSR_ACC_HOST_CSR_WR, TSR_CSR_LQ_PUSH);
    CHECK(a && a->val == 1u && a->time == 7u, "HOST_CSR_WR LQ_PUSH at MTIME 7");
    a = acc_find_kind(&L, TSR_ACC_HOST_CSR_RD, TSR_CSR_STATUS);
    CHECK(a && a->val == st, "HOST_CSR_RD STATUS with its value");
    uint8_t buf[256], back[256];
    for (int i = 0; i < 256; ++i) buf[i] = (uint8_t)(i * 7 + 1);
    CHECK(tsr_sim_dram_write(s, 1, 4096, buf, 256) == 0 && memcmp(tsr_sim_dram(s, 1) + 4096, buf, 256) == 0, "dram_write");
    a = acc_find_kind(&L, TSR_ACC_HOST_DRAM_WR, ~0u);
    CHECK(a && a->ch == 1u && a->addr == 4096u && a->len == 256u, "HOST_DRAM_WR ch 1 [4096, +256)");
    CHECK(tsr_sim_dram_read(s, 1, 4096, back, 256) == 0 && memcmp(back, buf, 256) == 0, "dram_read");
    a = acc_find_kind(&L, TSR_ACC_HOST_DRAM_RD, ~0u);
    CHECK(a && a->ch == 1u && a->addr == 4096u && a->len == 256u, "HOST_DRAM_RD");
    const uint32_t n0 = L.n;
    CHECK(tsr_sim_dram_write(s, 2, 0, buf, 1) == -1 && tsr_sim_dram_read(s, 0, (1ull << 24) - 8, back, 16) == -1,
          "out-of-range DRAM accesses refused");
    tsr_sim_dram_note_write(s, 0, 1ull << 30, 64);       // out of range: ignored
    CHECK(L.n == n0, "no event for refused accesses");
    tsr_sim_dram_note_write(s, 0, 640, 64);
    CHECK(L.n == n0 + 1 && L.v[n0].kind == TSR_ACC_HOST_DRAM_WR && L.v[n0].addr == 640u && L.v[n0].len == 64u, "note_write");
    tsr_sim_cw_read(s, TSR_CW_ISSUE_PIDX);
    a = acc_find_kind(&L, TSR_ACC_RV_CW_RD, TSR_CW_ISSUE_PIDX);
    CHECK(a && a->pc == TSR_SIM_PC_HOOK, "RV_CW_RD (test form, pc TSR_SIM_PC_HOOK)");
    op(s, TSR_OP_LD, 3, TSR_TICKET_NONE, TSR_TICKET_NONE);
    a = acc_find_kind(&L, TSR_ACC_RV_CW_WR, TSR_CW_ISSUE_PUSH);
    CHECK(a && a->val == TSR_ISSUE_PUSH_WMASK, "RV_CW_WR ISSUE_PUSH");
    uint32_t v = 0;
    CHECK(tsr_sim_cw_access_pc(s, 0, TSR_CW_ISSUE_TICKET, &v, 0x1234u) == TSR_CW_OK, "access with a pc");
    a = acc_find_kind(&L, TSR_ACC_RV_CW_RD, TSR_CW_ISSUE_TICKET);
    CHECK(a && a->pc == 0x1234u && a->val == v, "RV_CW_RD carries the pc");
    const uint32_t n1 = L.n;
    tsr_sim_cw_access_pc(s, 0, TSR_CW_ISSUE_TICKET, &v, TSR_SIM_PC_SBA);
    CHECK(L.n == n1, "an SBA access through the bus is not an RV_CW event");
    tsr_sim_acc e;
    memset(&e, 0, sizeof(e));
    e.kind = TSR_ACC_SBA_RD;
    e.addr = 0x40000010u;
    e.size = 4;
    e.val = 77;
    tsr_sim_acc_emit(s, &e);
    CHECK(L.n == n1 + 1 && L.v[n1].kind == TSR_ACC_SBA_RD && L.v[n1].val == 77u && L.v[n1].time == tsr_sim_now(s), "acc_emit");
    tsr_sim_remove_acc_listener(s, id);
    CHECK(tsr_sim_acc_listened(s) == listened0, "removed");
    tsr_sim_destroy(s);
}

// ---------------------------------------------------------------------------
// 9. TSR_SIM_LOG
// ---------------------------------------------------------------------------
static char * slurp(const char * path, size_t * n) {
    FILE * f = fopen(path, "rb");
    if (f == NULL) return NULL;
    fseek(f, 0, SEEK_END);
    const long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char * b = (char *)calloc((size_t)sz + 1, 1);
    *n = fread(b, 1, (size_t)sz, f);
    fclose(f);
    return b;
}

// A small run with TSR_SIM_LOG=rv,<path>: queue events and a hart program (two instructions, a
// park loop) for RV retire lines.
static char * log_run(const char * path, const char * sched, size_t * n) {
    remove(path);
    char env[512];
    snprintf(env, sizeof(env), "rv,%s", path);
    setenv("TSR_SIM_LOG", env, 1);
    tsr_sim * s = mk(sched);
    unsetenv("TSR_SIM_LOG");
    const uint32_t t = op(s, TSR_OP_LD, 20, TSR_TICKET_NONE, TSR_TICKET_NONE);
    op(s, TSR_OP_MVM, 30, t, TSR_TICKET_NONE);
    tsr_sim_qctrl(s, TSR_QCTRL_QCLEAR);
    uint8_t * itcm = tsr_sim_itcm(s, NULL);
    const uint32_t prog[3] = { 0x00500093u /* addi x1, x0, 5 */, 0x00108113u /* addi x2, x1, 1 */, 0x0000006Fu /* j . */ };
    for (int i = 0; i < 3; ++i) for (int b = 0; b < 4; ++b) itcm[0x80 + 4 * i + b] = (uint8_t)(prog[i] >> (8 * b));
    tsr_sim_csr_write(s, TSR_CSR_RV_BOOT, 0);
    tsr_sim_csr_write(s, TSR_CSR_CTRL, TSR_CTRL_RV_RUN);
    tsr_sim_advance(s, 400);
    tsr_sim_destroy(s);
    return slurp(path, n);
}

static void test_log(void) {
    char path[512];
    // Per process: sim_sched_unit and sim_sched_unit_ubsan share the scratch directory and run in
    // parallel under ctest -j (W1b integration: a shared name made both fail now and then).
    snprintf(path, sizeof(path), "%s/sim_sched_unit.%ld.log", g_scratch, (long)getpid());
    size_t n1 = 0, n2 = 0;
    char * a = log_run(path, "random:3", &n1);
    char * b = log_run(path, "random:3", &n2);
    CHECK(a && b && n1 > 0 && n1 == n2 && memcmp(a, b, n1) == 0, "TSR_SIM_LOG content is deterministic (%zu bytes)", n1);
    CHECK(a && strstr(a, " push  u0 seq=0 pidx=0 w0=00000010") && strstr(a, " enq ") && strstr(a, " start ") &&
          strstr(a, " done ") && strstr(a, " qctrl value=00000008"), "push / enq / start / done / qctrl lines");
    // W1b integration: RV retire lines need the built-in hart (rv.cmake; absent in a bare-core build).
    tsr_sim * probe = mk(NULL);
    const int have_rv = tsr_sim_rv_agent(probe) != NULL;
    tsr_sim_destroy(probe);
    if (have_rv) {
        CHECK(a && strstr(a, " rv    pc=00000080 insn=00500093 x1=00000005") && strstr(a, " rv    pc=00000084 insn=00108113 x2=00000006"),
              "RV retire lines");
    }
    free(a);
    free(b);
    remove(path);
}

// ---------------------------------------------------------------------------
// 10. ndmreset
// ---------------------------------------------------------------------------
static void test_ndmreset(void) {
    tsr_sim * s = mk(NULL);
    uint8_t * itcm = tsr_sim_itcm(s, NULL);
    const uint32_t jself = 0x0000006Fu;
    for (int b = 0; b < 4; ++b) itcm[0x80 + b] = (uint8_t)(jself >> (8 * b));
    tsr_sim_csr_write(s, TSR_CSR_CTRL, TSR_CTRL_RV_RUN);
    tsr_sim_advance(s, 100);
    uint32_t st = tsr_sim_csr_read(s, TSR_CSR_STATUS);
    CHECK((st & TSR_STATUS_RV_RUNNING) && !(st & TSR_STATUS_RV_RESET_DONE), "running");
    const uint64_t gen = tsr_sim_rv_boot_gen(s);
    tsr_sim_rv_ndmreset(s, 1);
    st = tsr_sim_csr_read(s, TSR_CSR_STATUS);
    CHECK(!(st & TSR_STATUS_RV_RUNNING) && (st & TSR_STATUS_RV_RESET_DONE) && (st & TSR_STATUS_QERR) &&
          (tsr_sim_csr_read(s, TSR_CSR_CTRL) & TSR_CTRL_RV_RUN), "ndmreset: held in reset, QABORT applied, RV_RUN kept");
    const uint64_t pc_held = tsr_sim_rv_instret(s);
    tsr_sim_advance(s, 1000);
    CHECK(tsr_sim_rv_instret(s) == pc_held, "the hart does not run while held");
    tsr_sim_rv_ndmreset(s, 0);
    CHECK(tsr_sim_rv_boot_gen(s) == gen + 1u, "release restarts the hart");
    tsr_sim_advance(s, 100);
    st = tsr_sim_csr_read(s, TSR_CSR_STATUS);
    if (tsr_sim_rv_agent(s) != NULL) {   // W1b integration: a bare-core build has no hart to run
        CHECK((st & TSR_STATUS_RV_RUNNING) && tsr_sim_rv_instret(s) > pc_held, "running again");
    }
    tsr_sim_destroy(s);
}

// ---------------------------------------------------------------------------
// 11. The hart under the schedule (built-in RISC-V agent; skipped without it): the random RV CPI
//     delta, the spin rule's 4th read and the jump target (SW §4.4, tsr_sim_int.h §5b)
// ---------------------------------------------------------------------------
typedef struct rvline { uint64_t t; uint32_t pc; } rvline;

// Runs prog (words at ITCM 0x80, RV_BOOT 0) at rv_ratio 1.0, so that RV and core cycles are equal,
// for `cycles` core cycles after `pre` (may push stub ops and advance); returns the retire lines
// (TSR_SIM_LOG_RV: the core time each instruction completed, effective RV cycles), at most max.
static uint32_t rv_lines(const char * sched, const uint32_t * prog, uint32_t nw, void (*pre)(tsr_sim *), uint64_t cycles,
                         rvline * out, uint32_t max) {
    tsr_config cfg = TSR_CONFIG_U200_V1;
    cfg.num_ch = 2;
    cfg.ch_bytes = 1ull << 24;
    FILE * f = tmpfile();
    CHECK(f != NULL, "tmpfile");
    if (f == NULL) return 0;
    tsr_sim_opts o;
    memset(&o, 0, sizeof(o));
    o.sched = sched;
    o.rv_ratio = 1.0;
    o.log = f;
    o.log_flags = TSR_SIM_LOG_RV;
    tsr_sim * s = tsr_sim_create_opts(&cfg, &o);
    CHECK(s != NULL, "create (%s)", sched);
    if (s == NULL) { fclose(f); return 0; }
    uint8_t * itcm = tsr_sim_itcm(s, NULL);
    for (uint32_t i = 0; i < nw; ++i)
        for (int b = 0; b < 4; ++b) itcm[0x80 + 4 * i + (uint32_t)b] = (uint8_t)(prog[i] >> (8 * b));
    if (pre) pre(s);
    tsr_sim_csr_write(s, TSR_CSR_RV_BOOT, 0);
    tsr_sim_csr_write(s, TSR_CSR_CTRL, TSR_CTRL_RV_RUN);
    tsr_sim_advance(s, cycles);
    tsr_sim_destroy(s);                                   // flushes the log; o.log stays open
    rewind(f);
    char line[256];
    uint32_t n = 0;
    while (n < max && fgets(line, sizeof(line), f)) {
        unsigned long long t;
        unsigned pc;
        if (strstr(line, " rv    pc=") && sscanf(line, "%llu rv    pc=%x", &t, &pc) == 2) {
            out[n].t = t;
            out[n].pc = pc;
            ++n;
        }
    }
    fclose(f);
    return n;
}

static void spin_pre(tsr_sim * s) {
    // MVM stub of 1000 cycles (dispatched and started at 2, done at 1002), running when the hart
    // boots at MTIME 100: its completion is the next core event.
    op(s, TSR_OP_MVM, 1000, TSR_TICKET_NONE, TSR_TICKET_NONE);
    tsr_sim_advance(s, 100);
}

static void test_rv_sched(void) {
    tsr_sim * probe = mk(NULL);
    const int have_rv = tsr_sim_rv_agent(probe) != NULL;
    tsr_sim_destroy(probe);
    if (!have_rv) {
        printf("sim_sched_unit: no RISC-V model in this library: part 11 skipped\n");
        return;
    }
    // (a) CPI: a DTCM-only loop (lw / add / sw / mul / div / taken branch: costs 1 to 37) has the
    // same instruction stream under every schedule; random adds -1, 0 or +1 per instruction, never
    // below 1 cycle (tsr_sched_rv_cpi), fixed nothing.
    static const uint32_t loop[8] = { 0x001002b7u /* lui t0, 0x100 (DTCM) */, 0x00300313u /* li t1, 3 */,
                                      0x0002a383u /* 1: lw t2, 0(t0) */, 0x006383b3u /* add t2, t2, t1 */,
                                      0x0072a023u /* sw t2, 0(t0) */, 0x02638433u /* mul s0, t2, t1 */,
                                      0x026444b3u /* div s1, s0, t1 */, 0xfe0006e3u /* beqz zero, 1b */ };
    static rvline fx[600], rn[600];
    const uint32_t nf = rv_lines("fixed", loop, 8, NULL, 3000, fx, 600);
    for (uint32_t seed = 1; seed <= 3; ++seed) {
        char spec[32];
        snprintf(spec, sizeof(spec), "random:%u", seed);
        const uint32_t nr = rv_lines(spec, loop, 8, NULL, 3000, rn, 600);
        const uint32_t n = nf < nr ? nf : nr;
        uint32_t plus = 0, minus = 0, bad = 0, below1 = 0, pcs = 0;
        for (uint32_t i = 1; i < n; ++i) {
            if (fx[i].pc != rn[i].pc) ++pcs;
            const int64_t cf = (int64_t)(fx[i].t - fx[i - 1].t), cr = (int64_t)(rn[i].t - rn[i - 1].t), d = cr - cf;
            if (d == 1) ++plus;
            else if (d == -1) ++minus;
            else if (d != 0) ++bad;
            if (cr < 1) ++below1;
            if (d == -1 && cf < 2) ++below1;
        }
        CHECK(n >= 200 && pcs == 0, "%s: %u retire lines, same instruction stream as fixed (%u differ)", spec, n, pcs);
        CHECK(bad == 0 && below1 == 0, "%s: per-instruction CPI delta in {-1, 0, +1}, cost >= 1 (%u out of range, %u below 1)",
              spec, bad, below1);
        CHECK(plus > 10 && minus > 10, "%s: the CPI varies both ways (%u +1, %u -1 of %u)", spec, plus, minus, n - 1);
    }
    // (b) The spin rule: the same DONE_SEQ register read with the same value 4 times in a row (no
    // other MMIO or store between) makes the hart jump to the next core event, the MVM completion
    // at 1002; 3 reads do not. Then the marker addi retires 1 cycle after the hart resumes there
    // (the jump neither stops early nor overshoots the event).
    for (uint32_t reads = 3; reads <= 4; ++reads) {
        uint32_t prog[8], nw = 0;
        prog[nw++] = 0x200002b7u;                                       // lui t0, 0x20000 (control window)
        for (uint32_t k = 0; k < reads; ++k) prog[nw++] = 0x0482a303u;  // lw t1, DONE_SEQ(MVM)(t0)
        prog[nw++] = 0x00700393u;                                       // li t2, 7 (the marker)
        prog[nw++] = 0x0000006fu;                                       // j .
        static rvline sl[64];
        const uint32_t n = rv_lines("fixed", prog, nw, spin_pre, 2000, sl, 64);
        const uint32_t im = 1u + reads;                                 // the marker's retire line
        CHECK(n > im && sl[im].pc == 0x80u + 4u * im, "%u reads: marker retired (%u lines)", reads, n);
        if (n <= im) continue;
        if (reads == 3) {
            CHECK(sl[im].t < 200u, "3 identical reads: no jump (marker at MTIME %llu)", (unsigned long long)sl[im].t);
        } else {
            CHECK(sl[im].t == 1003u, "4 identical reads: jump to the completion at 1002, marker at MTIME %llu (1003)",
                  (unsigned long long)sl[im].t);
        }
    }
}

int main(int argc, char ** argv) {
    if (argc > 1) {
        g_scratch = argv[1];
        mkdir(g_scratch, 0755);
    }
    unsetenv("TSR_SIM_SCHED");
    unsetenv("TSR_SIM_LOG");
    register_stubs();
    test_rng();
    test_parse();
    test_opts();
    test_listeners();
    test_ties();
    test_random_timing();
    test_lag();
    test_acc();
    test_log();
    test_ndmreset();
    test_rv_sched();
    return test_finish("sim_sched_unit");
}
