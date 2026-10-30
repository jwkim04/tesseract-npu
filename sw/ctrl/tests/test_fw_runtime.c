// test_fw_runtime.c — fw_runtime (docs/v2/06-software-verification.md "SW" §10.2; P2.2): the SDK
// runtime (sw/ctrl/rt: crt0, R1-R10 of docs/v2/01-execution-model.md "EX" §4.7, traps and panic of
// docs/v2/05-debug.md "DB" §6.3, log records of DB §6.1) as the RISC-V binary on the simulator
// (tsr_rv executing the real image is the reference, SW §6.8), driven through the host CSR port as
// the transport will drive it (EX §3.3-§3.5, §4.8-§4.9).
//
// Image: fw_rt (sw/ctrl/tests/runtime.cmake) = sw/ctrl/rt + sw/ctrl/tests/fw/rt_*.cpp (NO_KERNELS);
// kernel indices are looked up by name in the ELF (tsr_elf), the image id is tsr-elf's.
//
// One scenario, run under the fixed and 7 random schedules with TSR_SIM_RACE=fatal (and twice more at
// F_RV / F_CORE 1.5 and 0.25); every run makes every check below, and the runs must agree on SW §4.7's
// observables: LCQ records r0-r6, the
// per-unit op streams, the RISC-V's SPM store stream, the final written DRAM / SPM (test_sim_obs.h),
// and the log words (TS words and the TRAP payload masked: log content is not an SW §4.7
// observable, but every record here is deterministic). Phases, in launch order:
//   A  cold load (load_image of test_sim_rvhost.h: receive CRC and VERIFY), 8 descriptors pushed
//      before boot (the LQ full, no LQ_OVERFLOW): HELLO first (image id = tsr-elf's stored and
//      computed id, SDK ABI | runtime version, kernel count), then PING -> PONG and rt_args0..6,
//      with the host consuming only when the LCQ is full (R8 back-pressure);
//   B  rt_args7 (0-7 arguments: args beyond nargs read 0), ARGBLK of 1 and 16 lines (R5 staging:
//      blk = the top 16 SPM lines, content = the DRAM block), BAD_DESC variants (unknown TYPE,
//      reserved bit, PING with KERNEL != 0, KERNEL beyond the table, ARGBLK channel >= NUM_CH,
//      beat + lines > 2^28, a block beyond the channel: the staging LD's DRAM_RANGE in r3-r6),
//      PING, a copy kernel;
//   C  a tile error: TILE_ERR / DRAM_RANGE with ticket, ERR_INFO, ERR_W0, ERR_PIDX of the first
//      error and the kernel's RET; the next launch is clean (R9);
//   D  RET < 0 -> KERNEL_ERR, RET >= 0 -> OK; D2 R8 back-pressure: 8 DONE records fill the LCQ, the
//      9th launch waits in R8 (RT_STATE ANSWER, the hart asleep) until the host drains; a cancel of
//      that launch while it waits (QABORT after R7) is cleared by R9 (ERR_CODE[SYNC]), so the next
//      launch is clean;
//   E  PROF: REC_CFG.ALL during the launch only, tsr_args flags; LAUNCH_BEGIN / END markers around
//      every kernel (DB §8.4);
//   F  logs (SW §6.6): every record type and conversion (FMT strings from .tsr_logfmt, %s from
//      .rodata), the LOG_CFG.LEVEL filter, ENABLE = 0, the compile-time filter (DEBUG in a release
//      image), drop counting with the "dropped n" EVENT, LOG_CFG.BLOCK (the host holds the full
//      ring: the kernel waits, no DONE, no word; then drained: no drop);
//   G  cancel and stop watermarks (EX §4.9): before start (ABORTED, ERR_CODE 0, nothing runs; a
//      covered descriptor that is also invalid answers BAD_DESC: R4 before R3, EX §4.4) and
//      during a long kernel (stop: polls stop_requested() -> STOPPED; cancel: QABORT of its held
//      ops -> ABORTED with the first aborted op in r3-r6), then clean launches;
//   H  a debugger (DM model through the DMI window): attached + launch_break, the hart halts at
//      __tsr_launch_halt (dcsr.cause ebreak, a0 = kernel, a1 = SEQ, a2 -> tsr_args) and resumes;
//   I  an illegal instruction in a kernel with a debugger attached (the hart stops at
//      __tsr_trap_halt with the faulting context restored; detached, resumed at the mret, it traps
//      again), the TRAP record (answer: ACTIVE; __tsr_crash, the TRAP log record, RT_STATE
//      TRAPPED), warm restart (EX §3.5: SOFT_RESET, DTCM reload, ITCM VERIFY, HELLO), panic as a
//      TRAP with mcause 24 and mtval = code after its FATAL record, a load access fault, an event
//      of a bad type (the runtime's panic), warm restarts, then PING and a copy;
//   J  a trap outside a launch (R1's kernel-table check, a corrupted entry loaded by the host): the
//      TRAP record as an event, then a warm restart;
//   K  an RV reset (RV_RUN 1 -> 0: QABORT, every unit left in error state, CC §12.1), the DTCM
//      reloaded, RV_RUN: R1's QCLEAR makes the next launch clean.
// Afterwards (fixed schedule): the launch costs of EX §4.11 in core cycles: descriptor push ->
// kernel entry (the LAUNCH_BEGIN marker) and the empty-launch round trip (push -> DONE in the LCQ),
// one at a time and 8 outstanding.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#include "test_sim_rvhost.h"      // first (_GNU_SOURCE): load_image, crc32_zlib (EX §3.3)
#include "test_sim_obs.h"         // SW §4.7 observables
#pragma GCC diagnostic pop

#include "tsr_dbg.h"
#include "tsr_dm.h"
#include "tsr_elf.h"
#include "tsr_race.h"

#define N_SCHED 10
static const char * const k_sched[N_SCHED] = {
    "fixed", "random:1", "random:2", "random:3", "random:4", "random:5", "random:6", "random:7", "fixed", "random:8",
};
// F_RV / F_CORE per run (0 = the default 166.7 / 276.3): the last two runs at other clock ratios.
static const double k_ratio[N_SCHED] = { 0, 0, 0, 0, 0, 0, 0, 0, 1.5, 0.25 };

#define EV_ARGS    0xC1u
#define EV_BLK     0xC2u
#define EV_STARTED 0xC3u
#define EV_TILE    0xC4u
#define EV_PROF    0xC5u

#define CH_BEATS   (1u << 24)                     // 1 GiB channels (min with 2^ADDR_BEAT_W)
#define BLK1_BEAT  0x1000u                        // ch1: argument blocks
#define BLK16_BEAT 0x1010u
#define SRC_BEAT   0x2000u                        // ch0: copy sources
#define DST_BEAT   0x3000u                        // ch1: copy destinations
#define H(ch, beat) (((uint32_t)(ch) << 28) | (uint32_t)(beat))

// ---------------------------------------------------------------------------------------------
// The image
// ---------------------------------------------------------------------------------------------
typedef struct img {
    const char * path;
    tsr_elf *    e;
    tsr_rv_elf   rv;
    uint64_t     id;
    uint32_t     n_kernels;
    uint32_t     k_args[8], k_blk1, k_blk16, k_nop, k_ret, k_copy, k_tile, k_stop, k_cancel, k_prof, k_log, k_trap;
    uint32_t     s_launch_halt, s_trap_halt, s_panic_point, s_trap_insn, s_dbg, s_crash, s_stack_bottom, s_stack_top;
    uint32_t     logfmt_lo, logfmt_hi;
    int          debug_build;           // TSR_LOG_MAX_LEVEL >= 4: rt_log's DEBUG record is compiled in
} img;
static img G;

static uint32_t sym(const char * name) {
    uint32_t v = 0;
    CHECK(tsr_elf_symbol(G.e, name, &v) == 0, "symbol %s in %s", name, G.path);
    return v;
}
static uint32_t kidx(const char * name) {
    const int i = tsr_elf_kernel_find(G.e, name);
    CHECK(i >= 0, "kernel %s in the table", name);
    return i < 0 ? 0xFFFFu : (uint32_t)i;
}

static int img_open(const char * path) {
    char err[256];
    G.path = path;
    if (tsr_elf_read(path, &G.e, err, sizeof(err)) != 0) {
        CHECK(0, "tsr_elf_read %s: %s", path, err);
        return 0;
    }
    if (tsr_rv_elf_read(path, &G.rv) != 0) {
        CHECK(0, "tsr_rv_elf_read %s", path);
        return 0;
    }
    uint64_t stored = 0;
    CHECK(tsr_elf_stored_id(G.e, &stored) == 0, "image: .tsr_id present");
    G.id = tsr_elf_id(G.e);
    CHECK(stored == G.id && G.id != 0u, "image: the stamped id 0x%016llx equals the computed one 0x%016llx",
          (unsigned long long)stored, (unsigned long long)G.id);
    char rep[4096];
    rep[0] = '\0';
    CHECK(tsr_elf_check(G.e, NULL, TSR_ELF_CHECK_ID, rep, sizeof(rep)) == 0, "tsr_elf_check: %s", rep);
    G.n_kernels = tsr_elf_n_kernels(G.e);
    char nm[32];
    for (uint32_t k = 0; k < 8u; ++k) {
        snprintf(nm, sizeof(nm), "rt_args%u", k);
        G.k_args[k] = kidx(nm);
    }
    G.k_blk1 = kidx("rt_blk1");
    G.k_blk16 = kidx("rt_blk16");
    G.k_nop = kidx("rt_nop");
    G.k_ret = kidx("rt_ret");
    G.k_copy = kidx("rt_copy");
    G.k_tile = kidx("rt_tile_err");
    G.k_stop = kidx("rt_stop");
    G.k_cancel = kidx("rt_cancel");
    G.k_prof = kidx("rt_prof");
    G.k_log = kidx("rt_log");
    G.k_trap = kidx("rt_trap");
    G.s_launch_halt = sym("__tsr_launch_halt");
    G.s_trap_halt = sym("__tsr_trap_halt");
    G.s_panic_point = sym("__tsr_panic_point");
    G.s_trap_insn = sym("rt_trap_insn");
    G.s_dbg = sym("__tsr_dbg");
    G.s_crash = sym("__tsr_crash");
    G.s_stack_bottom = sym("__stack_bottom");
    G.s_stack_top = sym("__stack_top");
    G.logfmt_lo = sym("__tsr_logfmt_start");
    G.logfmt_hi = sym("__tsr_logfmt_end");
    // A debug firmware build (no NDEBUG) keeps DEBUG records (SW §6.6): its format is in .tsr_logfmt.
    for (uint32_t i = 0; i < tsr_elf_logfmt_count(G.e); ++i) {
        const char * f = tsr_elf_logfmt_at(G.e, i, NULL);
        if (f != NULL && strcmp(f, "rt_log debug %u") == 0) G.debug_build = 1;
    }
    return !g_fail;
}

// ---------------------------------------------------------------------------------------------
// One run: the simulator, the host side of EX §4.8, the observables
// ---------------------------------------------------------------------------------------------
typedef struct rec8 { uint32_t r[8]; } rec8;

typedef struct run {
    const char * sched;
    double       ratio;
    tsr_sim *    sim;
    uint32_t     lcq_rd, log_rd, pushed;
    int          log_hold;              // do not drain the log ring (drop test)
    rec8 *       rec;
    uint32_t     n_rec, cap_rec;
    uint32_t *   logw;
    uint32_t     n_log, cap_log;
    uint32_t     marks[256];            // TRACE_MARK values pushed (MARK_PUSH)
    uint64_t     mark_time[256];
    uint32_t     n_marks;
    uint32_t     max_lag;               // records left in the LCQ at most
    obs          ob;
    // measurements (fixed schedule)
    uint64_t     mtime_end, instret;
} run;

static void run_free(run * r) {
    free(r->rec);
    free(r->logw);
    obs_free(&r->ob);
    r->rec = NULL;
    r->logw = NULL;
}

static void q_listener(void * ctx, tsr_sim * sim, const tsr_q_event * ev) {
    (void)sim;
    run * r = (run *)ctx;
    if (ev->kind == TSR_QEV_MARK_PUSH && r->n_marks < 256u) {
        r->mark_time[r->n_marks] = ev->time;
        r->marks[r->n_marks++] = ev->w[0];
    }
}

static void fw_consume(run * r) {
    const uint32_t wr = tsr_sim_csr_read(r->sim, TSR_CSR_LCQ_WR);
    if (wr - r->lcq_rd > r->max_lag) r->max_lag = wr - r->lcq_rd;
    for (uint32_t k = r->lcq_rd; k != wr; ++k) {
        if (r->n_rec == r->cap_rec) {
            r->cap_rec = r->cap_rec ? 2u * r->cap_rec : 128u;
            r->rec = (rec8 *)realloc(r->rec, r->cap_rec * sizeof(rec8));
        }
        for (uint32_t j = 0; j < 8u; ++j) r->rec[r->n_rec].r[j] = tsr_sim_csr_read(r->sim, TSR_CSR_LCQ_REC(k % TSR_LCQ_DEPTH, j));
        ++r->n_rec;
    }
    r->lcq_rd = wr;
    tsr_sim_csr_write(r->sim, TSR_CSR_LCQ_RD, wr);
    if (r->log_hold) return;
    const uint32_t lwr = tsr_sim_csr_read(r->sim, TSR_CSR_LOG_WR);
    for (uint32_t k = r->log_rd; k != lwr; ++k) {
        if (r->n_log == r->cap_log) {
            r->cap_log = r->cap_log ? 2u * r->cap_log : 1024u;
            r->logw = (uint32_t *)realloc(r->logw, r->cap_log * sizeof(uint32_t));
        }
        r->logw[r->n_log++] = tsr_sim_csr_read(r->sim, TSR_CSR_LOG_RING(k % TSR_LOG_DEPTH));
    }
    r->log_rd = lwr;
    tsr_sim_csr_write(r->sim, TSR_CSR_LOG_RD, lwr);
}

static uint32_t push(run * r, const uint32_t d[8]) {
    for (uint32_t k = 0; k < 8u; ++k) tsr_sim_csr_write(r->sim, TSR_CSR_LQ_W(k), d[k]);
    tsr_sim_csr_write(r->sim, TSR_CSR_LQ_PUSH, 1u);
    return r->pushed++;
}
static uint32_t launch(run * r, uint32_t kernel, uint32_t flags, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                       uint32_t a4, uint32_t a5, uint32_t a6) {
    const uint32_t d[8] = { TSR_DESC_TYPE_LAUNCH | flags | TSR_PUT(TSR_DESC_KERNEL, kernel), a0, a1, a2, a3, a4, a5, a6 };
    return push(r, d);
}
static uint32_t ping(run * r, uint32_t a0) {
    const uint32_t d[8] = { TSR_DESC_TYPE_PING, a0, 0, 0, 0, 0, 0, 0 };
    return push(r, d);
}

static uint32_t rt_phase(run * r) { return TSR_GET(tsr_sim_csr_read(r->sim, TSR_CSR_RT_STATE), TSR_RT_STATE_PHASE); }

// Advances until `want` records were consumed. lag: consume only when the LCQ is full or nothing
// happens (R8 back-pressure); trap_ok: a TRAPPED runtime ends the wait instead of failing.
static int fw_run_until_n(run * r, uint32_t want, int lag, int trap_ok, const char * what, int max_it) {
    for (int it = 0; it < max_it && r->n_rec < want; ++it) {
        const uint32_t ev = tsr_sim_advance(r->sim, 200000);
        const uint32_t pending = tsr_sim_csr_read(r->sim, TSR_CSR_LCQ_WR) - r->lcq_rd;
        if (!lag || pending >= TSR_LCQ_DEPTH || (ev & TSR_SIM_EV_BUDGET)) fw_consume(r);
        if (rt_phase(r) == TSR_RT_STATE_PHASE_TRAPPED && !trap_ok) {
            fw_consume(r);
            CHECK(0, "%s: %s: the runtime trapped (pc 0x%08x)", r->sched, what, tsr_sim_rv_pc(r->sim));
            return 0;
        }
    }
    fw_consume(r);
    CHECK(r->n_rec >= want, "%s: %s: %u records, expected %u", r->sched, what, r->n_rec, want);
    return r->n_rec >= want;
}
static int fw_run_until(run * r, uint32_t want, int lag, int trap_ok, const char * what) {
    return fw_run_until_n(r, want, lag, trap_ok, what, 20000);
}

// Waits until every pushed launch is answered and the runtime is back in R2 with the queues idle.
static void quiesce(run * r, const char * what) {
    int ok = 0;
    for (int it = 0; it < 20000; ++it) {
        fw_consume(r);
        const uint32_t st = tsr_sim_csr_read(r->sim, TSR_CSR_STATUS);
        if (tsr_sim_csr_read(r->sim, TSR_CSR_LCQ_ANSWERED) == r->pushed && rt_phase(r) == TSR_RT_STATE_PHASE_IDLE &&
            (st & TSR_STATUS_QIDLE) && (st & TSR_STATUS_RV_SLEEP)) {
            ok = 1;
            break;
        }
        tsr_sim_advance(r->sim, 2000);
    }
    fw_consume(r);
    CHECK(ok, "%s: %s: quiescent (answered %u of %u, phase %u)", r->sched, what,
          tsr_sim_csr_read(r->sim, TSR_CSR_LCQ_ANSWERED), r->pushed, rt_phase(r));
}

// ---------------------------------------------------------------------------------------------
// Record checks
// ---------------------------------------------------------------------------------------------
static uint32_t rtype(const rec8 * x) { return TSR_GET(x->r[0], TSR_LCQ_R0_TYPE); }
static uint32_t rstatus(const rec8 * x) { return TSR_GET(x->r[0], TSR_LCQ_R0_STATUS); }
static uint32_t rerr(const rec8 * x) { return TSR_GET(x->r[0], TSR_LCQ_R0_ERR_CODE); }

static const rec8 * at(run * r, uint32_t i) {
    static const rec8 none = {{ 0xFFFFFFFFu, 0xFFFFFFFFu, 0, 0, 0, 0, 0, 0 }};
    return i < r->n_rec ? &r->rec[i] : &none;
}

static void chk_done(run * r, uint32_t i, uint32_t seq, uint32_t status, uint32_t err, uint32_t ret, const char * what) {
    const rec8 * x = at(r, i);
    CHECK(rtype(x) == TSR_LCQ_R0_TYPE_DONE && rstatus(x) == status && rerr(x) == err &&
          TSR_GET(x->r[0], TSR_LCQ_R0_SEQ) == (seq & 0xFFu) && x->r[1] == seq && x->r[2] == ret,
          "%s: %s: record %u = DONE seq %u status %u err %u ret 0x%08x (got r0 0x%08x r1 %u r2 0x%08x)", r->sched, what, i,
          seq, status, err, ret, x->r[0], x->r[1], x->r[2]);
    if (err == 0u)
        CHECK(x->r[3] == TSR_TICKET_NONE && x->r[4] == 0u && x->r[5] == 0u && x->r[6] == 0u,
              "%s: %s: no error record (r3 0x%08x r4 0x%08x r5 0x%08x r6 %u)", r->sched, what, x->r[3], x->r[4], x->r[5],
              x->r[6]);
}
static void chk_err(run * r, uint32_t i, uint32_t ticket, uint32_t info, uint32_t w0, uint32_t pidx, const char * what) {
    const rec8 * x = at(r, i);
    CHECK(x->r[3] == ticket && x->r[4] == info && x->r[5] == w0 && x->r[6] == pidx,
          "%s: %s: r3-r6 = 0x%08x 0x%08x 0x%08x %u, expected 0x%08x 0x%08x 0x%08x %u", r->sched, what, x->r[3], x->r[4],
          x->r[5], x->r[6], ticket, info, w0, pidx);
}
static void chk_event(run * r, uint32_t i, uint32_t type, uint32_t seq, const char * what) {
    const rec8 * x = at(r, i);
    CHECK(rtype(x) == type && TSR_GET(x->r[0], TSR_LCQ_R0_SEQ) == (seq & 0xFFu) && x->r[1] == seq,
          "%s: %s: record %u = event 0x%02x seq %u (got r0 0x%08x r1 %u)", r->sched, what, i, type, seq, x->r[0], x->r[1]);
}
static void chk_hello(run * r, uint32_t i, uint32_t taken, const char * what) {
    const rec8 * x = at(r, i);
    CHECK(x->r[0] == TSR_LCQ_R0_TYPE_HELLO && x->r[1] == taken && x->r[2] == (uint32_t)G.id &&
          x->r[3] == (uint32_t)(G.id >> 32) && x->r[4] == (1u | (1u << 16)) && x->r[5] == G.n_kernels && x->r[6] == 0u &&
          x->r[7] > 0u,
          "%s: %s: HELLO r0-r7 = %08x %u %08x %08x %08x %u %u %u (id 0x%016llx, %u kernels, TAKEN %u)", r->sched, what,
          x->r[0], x->r[1], x->r[2], x->r[3], x->r[4], x->r[5], x->r[6], x->r[7], (unsigned long long)G.id, G.n_kernels,
          taken);
}

// ---------------------------------------------------------------------------------------------
// Log records (DB §6.1)
// ---------------------------------------------------------------------------------------------
typedef struct lrec { uint32_t at, hdr, n, level, source, type, ts, lseq; const uint32_t * p; } lrec;

// Parses the log words from index `from` into records; every record must pass DB §6.1's decoder
// check (tsr_log_rec_check, with .tsr_logfmt's bounds). Returns the count.
static uint32_t log_parse(run * r, uint32_t from, lrec * out, uint32_t max) {
    uint32_t n = 0;
    for (uint32_t i = from; i < r->n_log && n < max;) {
        uint32_t words = 0;
        const int bad = tsr_log_rec_check(r->logw + i, r->n_log - i, G.logfmt_lo, G.logfmt_hi, &words);
        CHECK(bad == 0, "%s: log word %u: decoder check %d (header 0x%08x)", r->sched, i, bad, r->logw[i]);
        if (bad) break;
        tsr_log_hdr h;
        tsr_log_hdr_decode(r->logw[i], &h);
        lrec * x = &out[n++];
        x->at = i;
        x->hdr = r->logw[i];
        x->n = h.n;
        x->level = h.level;
        x->source = h.source;
        x->type = h.type;
        x->ts = h.ts;
        x->lseq = h.lseq;
        x->p = r->logw + i + 1u + (h.ts ? 1u : 0u);   // payload after the TS word
        i += words;
    }
    return n;
}

static void chk_lrec(run * r, const lrec * x, uint32_t level, uint32_t source, uint32_t type, uint32_t n_payload,
                     uint32_t seq, const char * what) {
    CHECK(x->level == level && x->source == source && x->type == type && x->ts == 1u && x->n == n_payload + 1u &&
          x->lseq == (seq & 0xFFu),
          "%s: %s: log header 0x%08x (level %u source %u type %u ts %u n %u lseq %u; expected %u %u %u 1 %u %u)", r->sched,
          what, x->hdr, x->level, x->source, x->type, x->ts, x->n, x->lseq, level, source, type, n_payload + 1u,
          seq & 0xFFu);
}
static void chk_fmt(run * r, const lrec * x, const char * fmt, const char * what) {
    const char * s = tsr_elf_logfmt(G.e, x->p[0]);
    CHECK(s != NULL && strcmp(s, fmt) == 0, "%s: %s: format at 0x%08x is \"%s\", expected \"%s\"", r->sched, what, x->p[0],
          s ? s : "(none)", fmt);
}

// ---------------------------------------------------------------------------------------------
// DMI (DB §3.2-§3.6) through the host CSR window, as the debug server
// ---------------------------------------------------------------------------------------------
static tsr_sim * DS;
static uint32_t dR(uint32_t a) { return tsr_sim_csr_read(DS, TSR_CSR_DMI(a)); }
static void dW(uint32_t a, uint32_t v) { tsr_sim_csr_write(DS, TSR_CSR_DMI(a), v); }
static int dpoll(uint32_t bit, uint64_t max_cycles) {
    for (uint64_t c = 0; c <= max_cycles; c += 100) {
        if (dR(TSR_DMI_DMSTATUS) & bit) return 1;
        tsr_sim_advance(DS, 100);
    }
    return (dR(TSR_DMI_DMSTATUS) & bit) != 0;
}
static uint32_t dacs(void) {
    for (int k = 0; k < 1000; ++k) {
        const uint32_t v = dR(TSR_DMI_ABSTRACTCS);
        if (!(v & TSR_DM_ABSTRACTCS_BUSY)) return (v & TSR_DM_ABSTRACTCS_CMDERR) >> TSR_DM_ABSTRACTCS_CMDERR_LSB;
        tsr_sim_advance(DS, 100);
    }
    return 0xFFu;
}
static uint32_t dreg(uint32_t regno) {
    dW(TSR_DMI_COMMAND, 0x00220000u + regno);
    const uint32_t e = dacs();
    CHECK(e == 0u, "DMI: read regno 0x%04x: cmderr %u", regno, e);
    return dR(TSR_DMI_DATA0);
}
static void dregw(uint32_t regno, uint32_t v) {
    dW(TSR_DMI_DATA0, v);
    dW(TSR_DMI_COMMAND, 0x00230000u + regno);
    const uint32_t e = dacs();
    CHECK(e == 0u, "DMI: write regno 0x%04x: cmderr %u", regno, e);
}
static void dsba_write(uint32_t a, uint32_t v) {          // DB §3.6: 32-bit SBA write
    dW(TSR_DMI_SBCS, 2u << TSR_DM_SBCS_SBACCESS_LSB);
    dW(TSR_DMI_SBADDRESS0, a);
    dW(TSR_DMI_SBDATA0, v);
    CHECK((dR(TSR_DMI_SBCS) & (TSR_DM_SBCS_SBERROR | TSR_DM_SBCS_SBBUSYERROR)) == 0u, "SBA write 0x%08x", a);
}
static uint32_t dsba_read(uint32_t a) {
    dW(TSR_DMI_SBCS, (2u << TSR_DM_SBCS_SBACCESS_LSB) | TSR_DM_SBCS_SBREADONADDR);
    dW(TSR_DMI_SBADDRESS0, a);
    const uint32_t v = dR(TSR_DMI_SBDATA0);
    CHECK((dR(TSR_DMI_SBCS) & (TSR_DM_SBCS_SBERROR | TSR_DM_SBCS_SBBUSYERROR)) == 0u, "SBA read 0x%08x", a);
    return v;
}
static int dhalt(void) {
    dW(TSR_DMI_DMCONTROL, 0x80000001u);
    const int ok = dpoll(TSR_DM_DMSTATUS_ALLHALTED, 200000);
    dW(TSR_DMI_DMCONTROL, 0x00000001u);
    return ok;
}
static int dresume(void) {
    dW(TSR_DMI_DMCONTROL, 0x40000001u);
    const int ok = dpoll(TSR_DM_DMSTATUS_ALLRESUMEACK, 2000);
    dW(TSR_DMI_DMCONTROL, 0x00000001u);
    return ok;
}

// ---------------------------------------------------------------------------------------------
// Warm restart (EX §3.5): SOFT_RESET, DTCM segments reloaded, ITCM segments VERIFYed, boot, HELLO
// ---------------------------------------------------------------------------------------------
// With the hart in reset (RV_RUN = 0): DTCM segments reloaded, ITCM segments VERIFYed (reloaded only
// on a mismatch), then RV_RUN and HELLO (r1 = TAKEN).
static void reload_and_boot(run * r, const char * what) {
    tsr_sim * sim = r->sim;
    uint32_t n_itcm_ok = 0;
    for (uint32_t s = 0; s < G.rv.nseg; ++s) {
        const tsr_rv_elf_seg * g = &G.rv.seg[s];
        if (g->filesz == 0) continue;
        const uint32_t bytes = (g->filesz + 3u) & ~3u;
        uint8_t * buf = (uint8_t *)calloc(bytes, 1);
        memcpy(buf, g->data, g->filesz);
        const uint32_t crc = crc32_zlib(buf, bytes);
        int load = g->paddr >= TSR_RV_DTCM_BASE;          // DTCM: always reloaded (.data may have changed)
        if (!load) {                                       // ITCM: VERIFY, reload only on a mismatch
            tsr_sim_csr_write(sim, TSR_CSR_LOAD_ADDR, g->paddr);
            tsr_sim_csr_write(sim, TSR_CSR_LOAD_COUNT, bytes / 4u);
            tsr_sim_csr_write(sim, TSR_CSR_LOAD_CTRL, TSR_LOAD_CTRL_VERIFY);
            for (int i = 0; i < 1000 && (tsr_sim_csr_read(sim, TSR_CSR_LOAD_STATUS) & TSR_LOAD_STATUS_VERIFY_BUSY); ++i)
                tsr_sim_advance(sim, 1000);
            load = tsr_sim_csr_read(sim, TSR_CSR_LOAD_CRC) != crc;
            n_itcm_ok += !load;
        }
        if (load) {
            tsr_sim_csr_write(sim, TSR_CSR_LOAD_ADDR, g->paddr);
            for (uint32_t i = 0; i < bytes; i += 4) tsr_sim_csr_write(sim, TSR_CSR_LOAD_DATA, get_u32_le(buf + i));
            CHECK(tsr_sim_csr_read(sim, TSR_CSR_LOAD_CRC) == crc, "%s: %s: reload 0x%08x: receive CRC", r->sched, what,
                  g->paddr);
        }
        free(buf);
    }
    CHECK(n_itcm_ok >= 1u, "%s: %s: ITCM VERIFY matched (no ITCM rewrite)", r->sched, what);
    const uint32_t n0 = r->n_rec;
    tsr_sim_csr_write(sim, TSR_CSR_CTRL, tsr_sim_csr_read(sim, TSR_CSR_CTRL) | TSR_CTRL_RV_RUN);
    if (fw_run_until(r, n0 + 1u, 0, 0, what)) chk_hello(r, n0, r->pushed, what);
}

static void warm_restart(run * r, const char * what) {
    tsr_sim * sim = r->sim;
    tsr_sim_csr_write(sim, TSR_CSR_CTRL, tsr_sim_csr_read(sim, TSR_CSR_CTRL) | TSR_CTRL_SOFT_RESET);
    for (int it = 0; it < 1000 && (tsr_sim_csr_read(sim, TSR_CSR_STATUS) & TSR_STATUS_SR_BUSY); ++it)
        tsr_sim_advance(sim, 10000);
    const uint32_t st = tsr_sim_csr_read(sim, TSR_CSR_STATUS);
    CHECK(!(st & TSR_STATUS_SR_BUSY) && !(tsr_sim_csr_read(sim, TSR_CSR_CTRL) & TSR_CTRL_RV_RUN),
          "%s: %s: SOFT_RESET done, RV_RUN 0 (STATUS 0x%08x)", r->sched, what, st);
    // The rings were flushed (EX §8.5): the host's read pointers follow.
    r->lcq_rd = tsr_sim_csr_read(sim, TSR_CSR_LCQ_RD);
    r->log_rd = tsr_sim_csr_read(sim, TSR_CSR_LOG_RD);
    CHECK(r->lcq_rd == tsr_sim_csr_read(sim, TSR_CSR_LCQ_WR) && tsr_sim_csr_read(sim, TSR_CSR_LQ_TAKEN) == r->pushed &&
          tsr_sim_csr_read(sim, TSR_CSR_LCQ_ANSWERED) == r->pushed, "%s: %s: rings flushed", r->sched, what);
    reload_and_boot(r, what);
}

// ---------------------------------------------------------------------------------------------
// The scenario
// ---------------------------------------------------------------------------------------------
static uint32_t ld_w0(uint32_t ch) { return TSR_OP_LD | (ch << 8); }

static void fw_scenario(run * r) {
    char what[160];
    tsr_config cfg = TSR_CONFIG_U200_V1;
    cfg.num_ch = 2;
    cfg.ch_bytes = 1ull << 30;
    cfg.caps = TSR_CAPS_VU;
    tsr_sim_opts o;
    memset(&o, 0, sizeof(o));
    o.sched = r->sched;
    o.rv_ratio = r->ratio;
    o.race = 2;                                            // TSR_SIM_RACE=fatal
    tsr_sim * sim = tsr_sim_create_opts(&cfg, &o);
    CHECK(sim != NULL, "%s: create", r->sched);
    if (sim == NULL) return;
    r->sim = sim;
    DS = sim;
    CHECK(tsr_sim_dram_limit_beats(sim) == CH_BEATS, "channel beats");
    const uint32_t L = cfg.spm_bytes / 64u;
    obs_attach(&r->ob, sim);
    tsr_sim_add_listener(sim, q_listener, r);

    // DRAM inputs (host writes before the pushes: EX §2.3 H1).
    uint8_t blk[16 * 64];
    for (uint32_t i = 0; i < sizeof(blk); ++i) blk[i] = (uint8_t)(i * 7u + 3u);
    tsr_sim_dram_write(sim, 1, (uint64_t)BLK1_BEAT * 64u, blk, 64);
    tsr_sim_dram_write(sim, 1, (uint64_t)BLK16_BEAT * 64u, blk, sizeof(blk));
    uint8_t src[32 * 64];
    for (uint32_t i = 0; i < sizeof(src); ++i) src[i] = (uint8_t)(i * 13u + 1u);
    tsr_sim_dram_write(sim, 0, (uint64_t)SRC_BEAT * 64u, src, sizeof(src));
    uint8_t ee[8 * 64];
    memset(ee, 0xEE, sizeof(ee));
    for (uint32_t k = 0; k < 8u; ++k) tsr_sim_dram_write(sim, 1, (uint64_t)(DST_BEAT + 0x100u * k) * 64u, ee, sizeof(ee));

    // ======== A: cold load, 8 descriptors before boot, HELLO, R8 back-pressure ========
    load_image(sim, &G.rv);
    tsr_sim_csr_write(sim, TSR_CSR_LOG_CFG, TSR_LOG_CFG_ENABLE | TSR_PUT(TSR_LOG_CFG_LEVEL, TSR_LOG_HDR_LEVEL_INFO));
    const uint32_t A0 = ping(r, 0xC0FFEE00u);
    for (uint32_t k = 0; k < 7u; ++k)
        launch(r, G.k_args[k], 0, 0xA0u + k, 0xA1u + k, 0xA2u + k, 0xA3u + k, 0xA4u + k, 0xA5u + k, 0xA6000000u | (0x1110u * k));
    CHECK(tsr_sim_csr_read(sim, TSR_CSR_LQ_SEQ) == 8u && !(tsr_sim_csr_read(sim, TSR_CSR_STATUS) & TSR_STATUS_LQ_OVERFLOW),
          "%s: A: 8 launches queued, no overflow", r->sched);
    tsr_sim_csr_write(sim, TSR_CSR_RV_BOOT, 0u);
    tsr_sim_csr_write(sim, TSR_CSR_CTRL, TSR_CTRL_RV_RUN);
    if (!fw_run_until(r, 16, 1, 0, "A")) goto out;
    chk_hello(r, 0, 0, "A cold boot");
    {
        const rec8 * p = at(r, 1);
        CHECK(p->r[0] == (TSR_LCQ_R0_TYPE_PONG | TSR_PUT(TSR_LCQ_R0_SEQ, A0)) && p->r[1] == A0 && p->r[2] == 0xC0FFEE00u &&
              p->r[3] == 0u && p->r[4] == 0u && p->r[5] == 0u && p->r[6] == 0u,
              "%s: A: PONG (r0 0x%08x r2 0x%08x r3-r6 0x%08x 0x%08x 0x%08x 0x%08x; EX §4.4: r3-r6 = 0)", r->sched,
              p->r[0], p->r[2], p->r[3], p->r[4], p->r[5], p->r[6]);
        for (uint32_t k = 0; k < 7u; ++k) {
            const uint32_t seq = 1u + k, i = 2u + 2u * k;
            snprintf(what, sizeof(what), "A rt_args%u", k);
            chk_event(r, i, EV_ARGS, seq, what);
            const rec8 * e = at(r, i);
            CHECK(TSR_GET(e->r[0], TSR_LCQ_R0_STATUS) == k && TSR_GET(e->r[0], TSR_LCQ_R0_ERR_CODE) == 0x10u,
                  "%s: %s: info 0x%06x (nargs, no blk, no PROF, seq)", r->sched, what, (e->r[0] >> 8) & 0xFFFFu);
            for (uint32_t j = 0; j < 6u; ++j)
                CHECK(e->r[2 + j] == (j < k ? 0xA0u + k + j : 0u), "%s: %s: a[%u] = 0x%08x", r->sched, what, j, e->r[2 + j]);
            chk_done(r, i + 1u, seq, TSR_LCQ_R0_STATUS_OK, 0, k << 16, what);   // a[6] is beyond nargs (k <= 6)
        }
    }
    CHECK(r->max_lag == TSR_LCQ_DEPTH, "%s: A: the LCQ was full (max %u records pending)", r->sched, r->max_lag);
    quiesce(r, "A");

    // ======== B: 7 arguments, argument blocks, BAD_DESC ========
    {
        const uint32_t n0 = r->n_rec;
        const uint32_t s_a7 = launch(r, G.k_args[7], 0, 1, 2, 3, 4, 5, 6, 0x00077777u);
        const uint32_t s_b1 = launch(r, G.k_blk1, 0, H(1, BLK1_BEAT), 0x11u, 0x22u, 0x33u, 0x44u, 0x55u, 0x66u);
        const uint32_t s_b16 = launch(r, G.k_blk16, 0, H(1, BLK16_BEAT), 0x111u, 0x222u, 0x333u, 0x444u, 0x555u, 0x666u);
        if (!fw_run_until(r, n0 + 6u, 0, 0, "B args")) goto out;
        chk_event(r, n0, EV_ARGS, s_a7, "B rt_args7");
        for (uint32_t j = 0; j < 6u; ++j) CHECK(at(r, n0)->r[2 + j] == 1u + j, "%s: B rt_args7: a[%u]", r->sched, j);
        chk_done(r, n0 + 1u, s_a7, TSR_LCQ_R0_STATUS_OK, 0, 0x00077777u, "B rt_args7");
        const uint32_t blk_addr = TSR_RV_SPM_BASE + 64u * (L - 16u);
        for (uint32_t b = 0; b < 2u; ++b) {
            const uint32_t lines = b ? 16u : 1u, i = n0 + 2u + 2u * b, seq = b ? s_b16 : s_b1;
            uint32_t h = 2166136261u;
            for (uint32_t k = 0; k < 64u * lines; ++k) h = (h ^ blk[k]) * 16777619u;
            snprintf(what, sizeof(what), "B rt_blk%u", lines);
            chk_event(r, i, EV_BLK, seq, what);
            const rec8 * e = at(r, i);
            CHECK(TSR_GET(e->r[0], TSR_LCQ_R0_STATUS) == lines && e->r[2] == blk_addr && e->r[3] == h &&
                  e->r[4] == get_u32_le(blk) && e->r[5] == get_u32_le(blk + 64u * lines - 4u) &&
                  e->r[6] == (b ? 0x111u : 0x11u) && e->r[7] == (b ? 0x222u : 0x22u),
                  "%s: %s: blk 0x%08x (0x%08x) hash 0x%08x (0x%08x) words 0x%08x 0x%08x", r->sched, what, e->r[2], blk_addr,
                  e->r[3], h, e->r[4], e->r[5]);
            chk_done(r, i + 1u, seq, TSR_LCQ_R0_STATUS_OK, 0, 0, what);
        }
        quiesce(r, "B args");
        // BAD_DESC variants (R4), then a staging LD beyond the channel.
        const uint32_t n1 = r->n_rec;
        uint32_t d[8] = { 0x07u, 1, 2, 3, 4, 5, 6, 7 };
        const uint32_t s_bad0 = push(r, d);                                                       // unknown TYPE
        d[0] = TSR_DESC_TYPE_LAUNCH | 0x200u | TSR_PUT(TSR_DESC_KERNEL, G.k_nop);
        const uint32_t s_bad1 = push(r, d);                                                       // reserved bit 9
        d[0] = TSR_DESC_TYPE_PING | TSR_PUT(TSR_DESC_KERNEL, 1u);
        const uint32_t s_bad2 = push(r, d);                                                       // PING, KERNEL != 0
        d[0] = TSR_DESC_TYPE_LAUNCH | TSR_PUT(TSR_DESC_KERNEL, G.n_kernels);
        const uint32_t s_bad3 = push(r, d);                                                       // beyond the table
        const uint32_t s_bad4 = launch(r, G.k_blk1, 0, H(3, BLK1_BEAT), 0, 0, 0, 0, 0, 0);       // ch 3 >= NUM_CH 2
        const uint32_t s_bad5 = launch(r, G.k_blk16, 0, H(1, 0x0FFFFFF8u), 0, 0, 0, 0, 0, 0);    // beat + 16 > 2^28
        if (!fw_run_until(r, n1 + 6u, 0, 0, "B BAD_DESC")) goto out;
        const uint32_t sb[6] = { s_bad0, s_bad1, s_bad2, s_bad3, s_bad4, s_bad5 };
        for (uint32_t k = 0; k < 6u; ++k) {
            snprintf(what, sizeof(what), "B BAD_DESC %u", k);
            chk_done(r, n1 + k, sb[k], TSR_LCQ_R0_STATUS_BAD_DESC, 0, 0, what);
        }
        quiesce(r, "B BAD_DESC");
        const uint32_t pidx = tsr_sim_cw_read(sim, TSR_CW_ISSUE_PIDX);
        const uint32_t dma = tsr_sim_cw_read(sim, TSR_CW_ISSUED_SEQ(TSR_UNIT_DMA)) & 0x0FFFFFFFu;
        const uint32_t n2 = r->n_rec;
        const uint32_t s_bad6 = launch(r, G.k_blk16, 0, H(1, CH_BEATS - 8u), 0, 0, 0, 0, 0, 0);  // LD DRAM_RANGE
        const uint32_t s_ping = ping(r, 0x12345678u);
        const uint32_t s_copy = launch(r, G.k_copy, 0, H(0, SRC_BEAT), H(1, DST_BEAT), 4, 100, 0, 0, 0);
        if (!fw_run_until(r, n2 + 3u, 0, 0, "B staging")) goto out;
        chk_done(r, n2, s_bad6, TSR_LCQ_R0_STATUS_BAD_DESC, TSR_ERR_DRAM_RANGE, 0, "B staging LD beyond the channel");
        chk_err(r, n2, TSR_PUT(TSR_TICKET_UNIT, TSR_UNIT_DMA) | dma, CH_BEATS - 8u, ld_w0(1), pidx, "B staging LD");
        CHECK(rtype(at(r, n2 + 1u)) == TSR_LCQ_R0_TYPE_PONG && at(r, n2 + 1u)->r[1] == s_ping &&
              at(r, n2 + 1u)->r[2] == 0x12345678u, "%s: B: PONG after BAD_DESC", r->sched);
        chk_done(r, n2 + 2u, s_copy, TSR_LCQ_R0_STATUS_OK, 0, 0, "B copy after the failed staging (R9)");
        quiesce(r, "B");
        uint8_t got[4 * 64];
        tsr_sim_dram_read(sim, 1, (uint64_t)DST_BEAT * 64u, got, sizeof(got));
        CHECK(memcmp(got, src, sizeof(got)) == 0, "%s: B: copy output", r->sched);
    }

    // ======== C: tile error, then a clean launch ========
    {
        const uint32_t n0 = r->n_rec;
        const uint32_t s_t = launch(r, G.k_tile, 0, H(0, CH_BEATS - 2u), H(1, DST_BEAT + 0x100u), 4, 200, 0, 0, 0);
        const uint32_t s_c = launch(r, G.k_copy, 0, H(0, SRC_BEAT + 4u), H(1, DST_BEAT + 0x200u), 4, 201, 0, 0, 0);
        if (!fw_run_until(r, n0 + 3u, 0, 0, "C")) goto out;
        chk_event(r, n0, EV_TILE, s_t, "C rt_tile_err");
        const rec8 * e = at(r, n0);
        chk_done(r, n0 + 1u, s_t, TSR_LCQ_R0_STATUS_TILE_ERR, TSR_ERR_DRAM_RANGE, (uint32_t)-9, "C TILE_ERR (RET < 0)");
        chk_err(r, n0 + 1u, e->r[3], CH_BEATS - 2u, ld_w0(0), e->r[2], "C first error = the LD");
        CHECK(TSR_GET(e->r[3], TSR_TICKET_UNIT) == TSR_UNIT_DMA && e->r[4] == e->r[3] + 1u, "%s: C: tickets", r->sched);
        chk_done(r, n0 + 2u, s_c, TSR_LCQ_R0_STATUS_OK, 0, 0, "C clean launch after the error (R9)");
        quiesce(r, "C");
        uint8_t got[4 * 64];
        tsr_sim_dram_read(sim, 1, (uint64_t)(DST_BEAT + 0x100u) * 64u, got, sizeof(got));
        CHECK(memcmp(got, ee, sizeof(got)) == 0, "%s: C: the ST behind the failed LD wrote nothing", r->sched);
        tsr_sim_dram_read(sim, 1, (uint64_t)(DST_BEAT + 0x200u) * 64u, got, sizeof(got));
        CHECK(memcmp(got, src + 4u * 64u, sizeof(got)) == 0, "%s: C: clean copy output", r->sched);
        CHECK(!(tsr_sim_csr_read(sim, TSR_CSR_STATUS) & TSR_STATUS_QERR), "%s: C: no unit in error after R9", r->sched);
    }

    // ======== D: KERNEL_ERR ========
    {
        const uint32_t n0 = r->n_rec;
        const uint32_t s1 = launch(r, G.k_ret, 0, (uint32_t)-5, 0, 0, 0, 0, 0, 0);
        const uint32_t s2 = launch(r, G.k_ret, 0, 3, 0, 0, 0, 0, 0, 0);
        if (!fw_run_until(r, n0 + 2u, 0, 0, "D")) goto out;
        chk_done(r, n0, s1, TSR_LCQ_R0_STATUS_KERNEL_ERR, 0, (uint32_t)-5, "D RET -5");
        chk_done(r, n0 + 1u, s2, TSR_LCQ_R0_STATUS_OK, 0, 3, "D RET 3");
        quiesce(r, "D");
    }

    // ======== D2 (fix W2a): R8 back-pressure; R9 after a cancel that lands between R7 and the post ========
    {
        const uint32_t n0 = r->n_rec;
        uint32_t sq[9];
        for (uint32_t k = 0; k < 8u; ++k) sq[k] = launch(r, G.k_ret, 0, 10u + k, 0, 0, 0, 0, 0, 0);
        int full = 0;                                      // 8 DONE records fill the LCQ (not consumed)
        for (int it = 0; it < 2000 && !full; ++it) {
            tsr_sim_advance(sim, 2000);
            full = tsr_sim_csr_read(sim, TSR_CSR_LCQ_WR) - r->lcq_rd == TSR_LCQ_DEPTH && rt_phase(r) == TSR_RT_STATE_PHASE_IDLE;
        }
        CHECK(full, "%s: D2: 8 DONE records fill the LCQ, the runtime idle", r->sched);
        sq[8] = launch(r, G.k_ret, 0, 18u, 0, 0, 0, 0, 0, 0);
        int waiting = 0;                                   // the 9th: R8 waits (WFI on irq_fast[3])
        for (int it = 0; it < 2000 && !waiting; ++it) {
            tsr_sim_advance(sim, 2000);
            waiting = rt_phase(r) == TSR_RT_STATE_PHASE_ANSWER && (tsr_sim_csr_read(sim, TSR_CSR_STATUS) & TSR_STATUS_RV_SLEEP);
        }
        CHECK(waiting && tsr_sim_csr_read(sim, TSR_CSR_LCQ_WR) - r->lcq_rd == TSR_LCQ_DEPTH &&
                  tsr_sim_csr_read(sim, TSR_CSR_LCQ_ANSWERED) == r->pushed - 1u && r->n_rec == n0,
              "%s: D2: the 9th launch waits in R8 (RT_STATE ANSWER, asleep, LCQ full, not answered)", r->sched);
        // EX §4.9: a cancel covering the running launch applies QABORT at once (ACTIVE until the post):
        // after R7 found no error, so only R9's ERR_CODE[SYNC] check sees it and clears it.
        tsr_sim_csr_write(sim, TSR_CSR_LQ_CANCEL, sq[8]);
        const uint32_t s_copy = launch(r, G.k_copy, 0, H(0, SRC_BEAT + 24u), H(1, DST_BEAT + 0x700u), 4, 600, 0, 0, 0);
        if (!fw_run_until(r, n0 + 10u, 0, 0, "D2")) goto out;
        for (uint32_t k = 0; k < 9u; ++k) {
            snprintf(what, sizeof(what), "D2 rt_ret %u", k);
            chk_done(r, n0 + k, sq[k], TSR_LCQ_R0_STATUS_OK, 0, 10u + k, what);
        }
        chk_done(r, n0 + 9u, s_copy, TSR_LCQ_R0_STATUS_OK, 0, 0, "D2 copy after the late cancel (R9 cleared the QABORT)");
        quiesce(r, "D2");
        uint8_t got[4 * 64];
        tsr_sim_dram_read(sim, 1, (uint64_t)(DST_BEAT + 0x700u) * 64u, got, sizeof(got));
        CHECK(memcmp(got, src + 24u * 64u, sizeof(got)) == 0, "%s: D2: copy output", r->sched);
        CHECK(!(tsr_sim_csr_read(sim, TSR_CSR_STATUS) & TSR_STATUS_QERR), "%s: D2: no unit in error", r->sched);
    }

    // ======== E: PROF ========
    {
        const uint32_t n0 = r->n_rec;
        const uint32_t s1 = launch(r, G.k_prof, TSR_DESC_PROF, 0, 0, 0, 0, 0, 0, 0);
        const uint32_t s2 = launch(r, G.k_prof, 0, 0, 0, 0, 0, 0, 0, 0);
        if (!fw_run_until(r, n0 + 4u, 0, 0, "E")) goto out;
        chk_event(r, n0, EV_PROF, s1, "E PROF");
        CHECK((at(r, n0)->r[2] & TSR_REC_CFG_ALL) && at(r, n0)->r[3] == 1u && at(r, n0)->r[4] == 1u && at(r, n0)->r[5] == 1u,
              "%s: E: PROF launch: REC_CFG 0x%x flags %u pidx0 %u base %u", r->sched, at(r, n0)->r[2], at(r, n0)->r[3],
              at(r, n0)->r[4], at(r, n0)->r[5]);
        chk_done(r, n0 + 1u, s1, TSR_LCQ_R0_STATUS_OK, 0, 0, "E PROF");
        chk_event(r, n0 + 2u, EV_PROF, s2, "E no PROF");
        CHECK(!(at(r, n0 + 2u)->r[2] & TSR_REC_CFG_ALL) && at(r, n0 + 2u)->r[3] == 0u,
              "%s: E: plain launch: REC_CFG 0x%x flags %u (restored)", r->sched, at(r, n0 + 2u)->r[2], at(r, n0 + 2u)->r[3]);
        chk_done(r, n0 + 3u, s2, TSR_LCQ_R0_STATUS_OK, 0, 0, "E no PROF");
        quiesce(r, "E");
    }

    // ======== F: logs ========
    {
        static lrec lr[512];
        const uint32_t v = 0x00C0FFEEu;
        // F1: every record type at LEVEL 5
        tsr_sim_csr_write(sim, TSR_CSR_LOG_CFG, TSR_LOG_CFG_ENABLE | TSR_PUT(TSR_LOG_CFG_LEVEL, 5u));
        uint32_t n0 = r->n_rec, l0 = r->n_log;
        uint32_t s = launch(r, G.k_log, 0, 2, 0, v, 0, 0, 0, 0);
        if (!fw_run_until(r, n0 + 1u, 0, 0, "F1")) goto out;
        chk_done(r, n0, s, TSR_LCQ_R0_STATUS_OK, 0, 0, "F1");
        quiesce(r, "F1");
        uint32_t nl = log_parse(r, l0, lr, 512);
        const uint32_t n_f1 = G.debug_build ? 6u : 5u;     // FMT, TEXT, EVENT, BLOB, [DEBUG,] WARN
        CHECK(nl == n_f1, "%s: F1: %u log records, expected %u (DEBUG %s)", r->sched, nl, n_f1,
              G.debug_build ? "kept: debug build" : "compiled out: release build");
        if (nl == n_f1) {
            chk_lrec(r, &lr[0], 3, 1, TSR_LOG_HDR_TYPE_FMT, 10, s, "F1 FMT");
            chk_fmt(r, &lr[0], "rt_log all %d %u %x %08x %f %t %llx %s", "F1 FMT");
            const uint32_t * p = lr[0].p;
            CHECK(p[1] == 0xFFFFFFFBu && p[2] == 7u && p[3] == v && p[4] == 0x1234u && p[5] == 0x3F800000u &&
                  p[6] == (TSR_PUT(TSR_TICKET_UNIT, TSR_UNIT_DMA) | 9u) && p[7] == 0x55667788u && p[8] == 0x11223344u,
                  "%s: F1: FMT arguments", r->sched);
            const char * str = tsr_elf_string(G.e, p[9]);
            CHECK(str != NULL && strcmp(str, "rt-rodata-string") == 0, "%s: F1: %%s argument 0x%08x -> \"%s\"", r->sched, p[9],
                  str ? str : "(none)");
            chk_lrec(r, &lr[1], 3, 1, TSR_LOG_HDR_TYPE_TEXT, 4, s, "F1 TEXT");
            CHECK(memcmp(lr[1].p, "rt_log text line", 16) == 0, "%s: F1: TEXT payload", r->sched);
            chk_lrec(r, &lr[2], 3, 1, TSR_LOG_HDR_TYPE_EVENT, 3, s, "F1 EVENT");
            CHECK(lr[2].p[0] == 0xABCDu && lr[2].p[1] == v && lr[2].p[2] == 2u, "%s: F1: EVENT payload", r->sched);
            chk_lrec(r, &lr[3], 3, 1, TSR_LOG_HDR_TYPE_BLOB, 4, s, "F1 BLOB");
            CHECK(lr[3].p[0] == 0xB10Bu && lr[3].p[1] == 1u && lr[3].p[2] == 2u && lr[3].p[3] == 3u, "%s: F1: BLOB payload",
                  r->sched);
            if (G.debug_build) {
                chk_lrec(r, &lr[4], 4, 1, TSR_LOG_HDR_TYPE_FMT, 2, s, "F1 DEBUG");
                chk_fmt(r, &lr[4], "rt_log debug %u", "F1 DEBUG");
            }
            const lrec * w = &lr[n_f1 - 1u];
            chk_lrec(r, w, 2, 1, TSR_LOG_HDR_TYPE_FMT, 2, s, "F1 WARN");
            chk_fmt(r, w, "rt_log warn %u", "F1 WARN");
            CHECK(w->p[1] == v, "%s: F1: WARN argument", r->sched);
        }
        // F2: LOG_CFG.LEVEL 2 keeps only the WARN record; F3: ENABLE = 0 keeps nothing
        tsr_sim_csr_write(sim, TSR_CSR_LOG_CFG, TSR_LOG_CFG_ENABLE | TSR_PUT(TSR_LOG_CFG_LEVEL, 2u));
        n0 = r->n_rec;
        l0 = r->n_log;
        s = launch(r, G.k_log, 0, 2, 0, v + 1u, 0, 0, 0, 0);
        if (!fw_run_until(r, n0 + 1u, 0, 0, "F2")) goto out;
        quiesce(r, "F2");
        nl = log_parse(r, l0, lr, 512);
        CHECK(nl == 1u && lr[0].type == TSR_LOG_HDR_TYPE_FMT && lr[0].level == 2u && lr[0].p[1] == v + 1u,
              "%s: F2: LEVEL 2: only the WARN record (%u records)", r->sched, nl);
        tsr_sim_csr_write(sim, TSR_CSR_LOG_CFG, TSR_PUT(TSR_LOG_CFG_LEVEL, 5u));
        n0 = r->n_rec;
        l0 = r->n_log;
        s = launch(r, G.k_log, 0, 2, 0, v + 2u, 0, 0, 0, 0);
        if (!fw_run_until(r, n0 + 1u, 0, 0, "F3")) goto out;
        quiesce(r, "F3");
        CHECK(r->n_log == l0, "%s: F3: ENABLE 0: no log words (%u)", r->sched, r->n_log - l0);
        // F4: drop: 60 records of 6 words into an undrained 256-word ring: 42 fit, 18 are dropped and
        // reported by the next record that fits
        tsr_sim_csr_write(sim, TSR_CSR_LOG_CFG, TSR_LOG_CFG_ENABLE | TSR_PUT(TSR_LOG_CFG_LEVEL, 3u));
        CHECK(tsr_sim_csr_read(sim, TSR_CSR_LOG_WR) == r->log_rd, "%s: F4: the log ring is empty", r->sched);
        n0 = r->n_rec;
        l0 = r->n_log;
        r->log_hold = 1;
        const uint32_t s4 = launch(r, G.k_log, 0, 1, 60, v + 4u, 0, 0, 0, 0);
        if (!fw_run_until(r, n0 + 1u, 0, 0, "F4")) goto out;
        chk_done(r, n0, s4, TSR_LCQ_R0_STATUS_OK, 0, 60, "F4");
        quiesce(r, "F4");
        r->log_hold = 0;
        fw_consume(r);
        nl = log_parse(r, l0, lr, 512);
        CHECK(nl == 42u, "%s: F4: %u records in the ring, expected 42", r->sched, nl);
        for (uint32_t k = 0; k < nl && k < 42u; ++k) {
            snprintf(what, sizeof(what), "F4 record %u", k);
            chk_lrec(r, &lr[k], 3, 1, TSR_LOG_HDR_TYPE_FMT, 4, s4, what);
            CHECK(lr[k].p[1] == k && lr[k].p[2] == 60u && lr[k].p[3] == v + 4u, "%s: %s: arguments", r->sched, what);
        }
        if (nl > 0u) chk_fmt(r, &lr[0], "rt_log %u of %u: 0x%x", "F4");
        CHECK(!(tsr_sim_csr_read(sim, TSR_CSR_STATUS) & TSR_STATUS_LOG_DROP), "%s: F4: no word dropped by the ring",
              r->sched);
        n0 = r->n_rec;
        l0 = r->n_log;
        const uint32_t s5 = launch(r, G.k_log, 0, 1, 1, v + 5u, 0, 0, 0, 0);
        if (!fw_run_until(r, n0 + 1u, 0, 0, "F4b")) goto out;
        quiesce(r, "F4b");
        nl = log_parse(r, l0, lr, 512);
        CHECK(nl == 2u, "%s: F4b: %u records, expected the drop EVENT and one record", r->sched, nl);
        if (nl == 2u) {
            chk_lrec(r, &lr[0], 2, 0, TSR_LOG_HDR_TYPE_EVENT, 2, s5, "F4b dropped n");
            CHECK(lr[0].p[0] == 0xF0000001u && lr[0].p[1] == 18u, "%s: F4b: \"dropped %u\" (code 0x%08x)", r->sched,
                  lr[0].p[1], lr[0].p[0]);
            chk_lrec(r, &lr[1], 3, 1, TSR_LOG_HDR_TYPE_FMT, 4, s5, "F4b record");
        }
        // F5: LOG_CFG.BLOCK: the kernel waits for the host to drain; nothing is dropped. The host holds
        // the ring until it is full and the kernel has waited (no DONE, no word), then drains (fix W2a).
        tsr_sim_csr_write(sim, TSR_CSR_LOG_CFG, TSR_LOG_CFG_ENABLE | TSR_LOG_CFG_BLOCK | TSR_PUT(TSR_LOG_CFG_LEVEL, 3u));
        n0 = r->n_rec;
        l0 = r->n_log;
        r->log_hold = 1;
        const uint32_t s6 = launch(r, G.k_log, 0, 1, 60, v + 6u, 0, 0, 0, 0);
        int lfull = 0;
        for (int it = 0; it < 2000 && !lfull; ++it) {
            tsr_sim_advance(sim, 2000);
            fw_consume(r);
            lfull = tsr_sim_csr_read(sim, TSR_CSR_LOG_WR) - r->log_rd == 42u * 6u;   // 42 whole records: 4 words free
        }
        {
            const uint32_t lw = tsr_sim_csr_read(sim, TSR_CSR_LOG_WR);
            tsr_sim_advance(sim, 100000);
            fw_consume(r);
            CHECK(lfull && r->n_rec == n0 && rt_phase(r) == TSR_RT_STATE_PHASE_KERNEL &&
                      tsr_sim_csr_read(sim, TSR_CSR_LOG_WR) == lw,
                  "%s: F5: BLOCK: with the ring full the kernel waits (full %d, %u records, phase %u)", r->sched, lfull,
                  r->n_rec - n0, rt_phase(r));
        }
        r->log_hold = 0;
        if (!fw_run_until(r, n0 + 1u, 0, 0, "F5")) goto out;
        quiesce(r, "F5");
        nl = log_parse(r, l0, lr, 512);
        CHECK(nl == 60u, "%s: F5: BLOCK: %u records, expected 60", r->sched, nl);
        for (uint32_t k = 0; k < nl; ++k)
            CHECK(lr[k].type == TSR_LOG_HDR_TYPE_FMT && lr[k].p[1] == k && lr[k].lseq == (s6 & 0xFFu),
                  "%s: F5: record %u", r->sched, k);
        tsr_sim_csr_write(sim, TSR_CSR_LOG_CFG, TSR_LOG_CFG_ENABLE | TSR_PUT(TSR_LOG_CFG_LEVEL, 3u));
    }

    // ======== G: cancel and stop ========
    {
        // G1: cancel before start (the runtime is idle: no QABORT); a covered descriptor that is also
        // invalid answers BAD_DESC (R4 validates before R3: BAD_DESC above ABORTED, EX §4.4)
        uint32_t n0 = r->n_rec;
        uint32_t q = r->pushed;
        tsr_sim_csr_write(sim, TSR_CSR_LQ_CANCEL, q + 2u);
        const uint32_t s1 = launch(r, G.k_copy, 0, H(0, SRC_BEAT + 8u), H(1, DST_BEAT + 0x300u), 4, 300, 0, 0, 0);
        const uint32_t s2 = ping(r, 0x55u);
        const uint32_t dbad[8] = { 0x07u, 0, 0, 0, 0, 0, 0, 0 };
        const uint32_t s2b = push(r, dbad);                                                      // unknown TYPE
        const uint32_t s3 = launch(r, G.k_nop, 0, 0, 0, 0, 0, 0, 0, 0);
        if (!fw_run_until(r, n0 + 4u, 0, 0, "G1")) goto out;
        chk_done(r, n0, s1, TSR_LCQ_R0_STATUS_ABORTED, 0, 0, "G1 cancelled launch");
        chk_done(r, n0 + 1u, s2, TSR_LCQ_R0_STATUS_ABORTED, 0, 0, "G1 cancelled PING");
        chk_done(r, n0 + 2u, s2b, TSR_LCQ_R0_STATUS_BAD_DESC, 0, 0, "G1 cancelled bad descriptor: BAD_DESC (R4 before R3)");
        chk_done(r, n0 + 3u, s3, TSR_LCQ_R0_STATUS_OK, 0, 0, "G1 launch after the watermark");
        quiesce(r, "G1");
        uint8_t got[4 * 64];
        tsr_sim_dram_read(sim, 1, (uint64_t)(DST_BEAT + 0x300u) * 64u, got, sizeof(got));
        CHECK(memcmp(got, ee, sizeof(got)) == 0, "%s: G1: the cancelled copy did not run", r->sched);
        // G2: stop before start
        n0 = r->n_rec;
        q = r->pushed;
        tsr_sim_csr_write(sim, TSR_CSR_LQ_STOP, q);
        const uint32_t s4 = launch(r, G.k_nop, 0, 0, 0, 0, 0, 0, 0, 0);
        const uint32_t s5 = launch(r, G.k_nop, 0, 0, 0, 0, 0, 0, 0, 0);
        if (!fw_run_until(r, n0 + 2u, 0, 0, "G2")) goto out;
        chk_done(r, n0, s4, TSR_LCQ_R0_STATUS_ABORTED, 0, 0, "G2 stopped before start");
        chk_done(r, n0 + 1u, s5, TSR_LCQ_R0_STATUS_OK, 0, 0, "G2 launch after the watermark");
        quiesce(r, "G2");
        // G3: soft stop during a long kernel polling stop_requested()
        n0 = r->n_rec;
        const uint32_t s6 = launch(r, G.k_stop, 0, 50000000u, (uint32_t)-7, 0, 0, 0, 0, 0);
        if (!fw_run_until(r, n0 + 1u, 0, 0, "G3 start")) goto out;
        chk_event(r, n0, EV_STARTED, s6, "G3 started");
        tsr_sim_csr_write(sim, TSR_CSR_LQ_STOP, s6);
        if (!fw_run_until_n(r, n0 + 2u, 0, 0, "G3", 1000)) goto out;   // bounded: a missed stop fails fast
        chk_done(r, n0 + 1u, s6, TSR_LCQ_R0_STATUS_STOPPED, 0, (uint32_t)-7, "G3 STOPPED (RET < 0)");
        quiesce(r, "G3");
        // G4: hard cancel during a long kernel: its held ops complete ABORTED
        n0 = r->n_rec;
        const uint32_t s7 = launch(r, G.k_cancel, 0, H(0, SRC_BEAT + 12u), H(1, DST_BEAT + 0x400u), 4, 400, 0, 0, 0);
        if (!fw_run_until(r, n0 + 1u, 0, 0, "G4 start")) goto out;
        chk_event(r, n0, EV_STARTED, s7, "G4 started");
        const rec8 e = *at(r, n0);
        tsr_sim_csr_write(sim, TSR_CSR_LQ_CANCEL, s7);
        const uint32_t s8 = launch(r, G.k_copy, 0, H(0, SRC_BEAT + 16u), H(1, DST_BEAT + 0x500u), 4, 401, 0, 0, 0);
        if (!fw_run_until_n(r, n0 + 3u, 0, 0, "G4", 1000)) goto out;   // bounded
        chk_done(r, n0 + 1u, s7, TSR_LCQ_R0_STATUS_ABORTED, TSR_ERR_ABORTED, 0, "G4 cancelled while running");
        chk_err(r, n0 + 1u, e.r[3], 0u, ld_w0(0), e.r[2], "G4 first error = the held LD");
        chk_done(r, n0 + 2u, s8, TSR_LCQ_R0_STATUS_OK, 0, 0, "G4 next launch clean (R9)");
        quiesce(r, "G4");
        tsr_sim_dram_read(sim, 1, (uint64_t)(DST_BEAT + 0x400u) * 64u, got, sizeof(got));
        CHECK(memcmp(got, ee, sizeof(got)) == 0, "%s: G4: the aborted ST wrote nothing", r->sched);
        tsr_sim_dram_read(sim, 1, (uint64_t)(DST_BEAT + 0x500u) * 64u, got, sizeof(got));
        CHECK(memcmp(got, src + 16u * 64u, sizeof(got)) == 0, "%s: G4: copy output", r->sched);
    }

    // ======== H: debugger: halt at __tsr_launch_halt, resume ========
    {
        tsr_sim_csr_write(sim, TSR_CSR_DBG_CFG, tsr_sim_csr_read(sim, TSR_CSR_DBG_CFG) | TSR_DBG_CFG_DMI_EN);
        dW(TSR_DMI_DMCONTROL, 0x00000001u);
        CHECK(dhalt(), "%s: H: halt the idle hart", r->sched);
        dregw(TSR_RV_CSR_DCSR, dreg(TSR_RV_CSR_DCSR) | (1u << 15));        // ebreakm
        dsba_write(G.s_dbg + 4u * TSR_DBG_INFO_ATTACHED_WORD, 1u);
        dsba_write(G.s_dbg + 4u * TSR_DBG_INFO_LAUNCH_BREAK_WORD, G.k_nop + 1u);
        CHECK(dsba_read(G.s_dbg) == TSR_DBG_INFO_MAGIC && dsba_read(G.s_dbg + 8u) == (uint32_t)G.id,
              "%s: H: __tsr_dbg magic and build id (= image id [31:0])", r->sched);
        CHECK(dresume(), "%s: H: resume", r->sched);
        uint32_t n0 = r->n_rec;
        const uint32_t s1 = launch(r, G.k_args[2], 0, 7, 8, 0, 0, 0, 0, 0);   // not selected by launch_break
        const uint32_t s2 = launch(r, G.k_nop, 0, 0, 0, 0, 0, 0, 0, 0);
        if (!fw_run_until(r, n0 + 2u, 0, 0, "H args2")) goto out;
        chk_done(r, n0 + 1u, s1, TSR_LCQ_R0_STATUS_OK, 0, 2u << 16, "H args2 (no break)");
        int halted = dpoll(TSR_DM_DMSTATUS_ALLHALTED, 2000000);
        CHECK(halted, "%s: H: the hart halted at the launch break", r->sched);
        if (halted) {
            const uint32_t dpc = dreg(TSR_RV_CSR_DPC);
            CHECK(dpc == G.s_launch_halt && ((dreg(TSR_RV_CSR_DCSR) >> 6) & 7u) == 1u,
                  "%s: H: dpc 0x%08x = __tsr_launch_halt 0x%08x, cause ebreak", r->sched, dpc, G.s_launch_halt);
            CHECK(dreg(TSR_DM_REGNO_GPR(10)) == G.k_nop && dreg(TSR_DM_REGNO_GPR(11)) == s2, "%s: H: a0 = kernel, a1 = SEQ",
                  r->sched);
            CHECK(r->n_rec == n0 + 2u && TSR_GET(tsr_sim_csr_read(sim, TSR_CSR_RT_STATE), TSR_RT_STATE_PHASE) ==
                  TSR_RT_STATE_PHASE_KERNEL, "%s: H: halted before the kernel ran", r->sched);
            dregw(TSR_RV_CSR_DPC, dpc + 4u);
            CHECK(dresume(), "%s: H: resume after the launch break", r->sched);
        }
        if (!fw_run_until(r, n0 + 3u, 0, 0, "H nop")) goto out;
        chk_done(r, n0 + 2u, s2, TSR_LCQ_R0_STATUS_OK, 0, 0, "H rt_nop after the break");
        quiesce(r, "H nop");
        // launch_break = all: the arguments through a2
        dsba_write(G.s_dbg + 4u * TSR_DBG_INFO_LAUNCH_BREAK_WORD, 0xFFFFFFFFu);
        n0 = r->n_rec;
        const uint32_t s3 = launch(r, G.k_args[1], 0, 0xBEEF0001u, 0xBEEF0002u, 0, 0, 0, 0, 0);
        halted = dpoll(TSR_DM_DMSTATUS_ALLHALTED, 2000000);
        CHECK(halted, "%s: H: halted (launch_break all)", r->sched);
        if (halted) {
            const uint32_t a2 = dreg(TSR_DM_REGNO_GPR(12));
            CHECK(dreg(TSR_DM_REGNO_GPR(10)) == G.k_args[1] && dsba_read(a2) == 0xBEEF0001u && dsba_read(a2 + 4u) == 0u &&
                  dsba_read(a2 + 4u * 8u) == s3, "%s: H: a2 -> tsr_args {a[0], a[1] = 0 beyond nargs, ..., seq}", r->sched);
            dregw(TSR_RV_CSR_DPC, dreg(TSR_RV_CSR_DPC) + 4u);
            CHECK(dresume(), "%s: H: resume", r->sched);
        }
        if (!fw_run_until(r, n0 + 2u, 0, 0, "H args1")) goto out;
        chk_done(r, n0 + 1u, s3, TSR_LCQ_R0_STATUS_OK, 0, 1u << 16, "H args1");
        quiesce(r, "H");
        // Detach: attached = 0, ebreakm off, DMI off.
        dsba_write(G.s_dbg + 4u * TSR_DBG_INFO_ATTACHED_WORD, 0u);
        dsba_write(G.s_dbg + 4u * TSR_DBG_INFO_LAUNCH_BREAK_WORD, 0u);
        CHECK(dhalt(), "%s: H: halt to detach", r->sched);
        dregw(TSR_RV_CSR_DCSR, dreg(TSR_RV_CSR_DCSR) & ~(1u << 15));
        CHECK(dresume(), "%s: H: resume to detach", r->sched);
        tsr_sim_csr_write(sim, TSR_CSR_DBG_CFG, tsr_sim_csr_read(sim, TSR_CSR_DBG_CFG) & ~TSR_DBG_CFG_DMI_EN);
    }

    // ======== I: TRAP, panic, warm restarts ========
    {
        uint32_t dtcm_bytes = 0;
        const uint8_t * dtcm = tsr_sim_dtcm(sim, &dtcm_bytes);
        const uint32_t crash_off = G.s_crash - TSR_RV_DTCM_BASE, dbg_off = G.s_dbg - TSR_RV_DTCM_BASE;
        static lrec lr[64];
        static const struct { uint32_t mode, mcause, info, mtval; } tc[4] = {
            { 0, 2u, 0x1234u, 0u }, { 1, 24u, 0x2345u, 0xDEAD0001u }, { 2, 5u, 0x3456u, 0x30000000u },
            { 4, 24u, 0x4567u, 0x52540001u },
        };
        for (uint32_t t = 0; t < 4u; ++t) {
            const uint32_t n0 = r->n_rec, l0 = r->n_log;
            uint32_t dbg_sp = 0, dbg_ra = 0, traps = 1;
            snprintf(what, sizeof(what), "I trap %u", t);
            if (t == 0u) {
                // Debugger attached (DB §6.3 step 2): the handler restores the faulting context and
                // stops at __tsr_trap_halt; the debugger detaches and resumes at its mret, so the
                // instruction traps again, now without a debugger (steps 3-4).
                tsr_sim_csr_write(sim, TSR_CSR_DBG_CFG, tsr_sim_csr_read(sim, TSR_CSR_DBG_CFG) | TSR_DBG_CFG_DMI_EN);
                CHECK(dhalt(), "%s: %s: halt to attach", r->sched, what);
                dregw(TSR_RV_CSR_DCSR, dreg(TSR_RV_CSR_DCSR) | (1u << 15));
                dsba_write(G.s_dbg + 4u * TSR_DBG_INFO_ATTACHED_WORD, 1u);
                CHECK(dresume(), "%s: %s: resume after attach", r->sched, what);
            }
            const uint32_t s = launch(r, G.k_trap, 0, tc[t].mode, tc[t].info, 0, 0, 0, 0, 0);
            if (t == 0u) {
                const int halted = dpoll(TSR_DM_DMSTATUS_ALLHALTED, 2000000);
                CHECK(halted, "%s: %s: the hart stopped at __tsr_trap_halt", r->sched, what);
                if (halted) {
                    const uint32_t dpc = dreg(TSR_RV_CSR_DPC);
                    CHECK(dpc == G.s_trap_halt && ((dreg(TSR_RV_CSR_DCSR) >> 6) & 7u) == 1u && dreg(0x342u) == 2u &&
                          dreg(0x341u) == G.s_trap_insn, "%s: %s: dpc 0x%08x (__tsr_trap_halt 0x%08x), mcause, mepc", r->sched,
                          what, dpc, G.s_trap_halt);
                    dbg_sp = dreg(TSR_DM_REGNO_GPR(2));
                    dbg_ra = dreg(TSR_DM_REGNO_GPR(1));
                    CHECK(dbg_sp >= G.s_stack_bottom && dbg_sp < G.s_stack_top, "%s: %s: the restored sp 0x%08x is the kernel's",
                          r->sched, what, dbg_sp);
                    const uint8_t * c = dtcm + crash_off;
                    CHECK(get_u32_le(c) == 0u && get_u32_le(c + 4u * TSR_CRASH_NESTED_WORD) == 0u &&
                          get_u32_le(dtcm + dbg_off + 4u * TSR_DBG_INFO_TRAP_COUNT_WORD) == 1u && r->n_rec == n0 &&
                          rt_phase(r) == TSR_RT_STATE_PHASE_KERNEL, "%s: %s: no crash record, no TRAP record yet", r->sched, what);
                    dsba_write(G.s_dbg + 4u * TSR_DBG_INFO_ATTACHED_WORD, 0u);
                    dregw(TSR_RV_CSR_DCSR, dreg(TSR_RV_CSR_DCSR) & ~(1u << 15));
                    dregw(TSR_RV_CSR_DPC, dpc + 4u);                      // the mret after the ebreak
                    CHECK(dresume(), "%s: %s: resume", r->sched, what);
                }
                tsr_sim_csr_write(sim, TSR_CSR_DBG_CFG, tsr_sim_csr_read(sim, TSR_CSR_DBG_CFG) & ~TSR_DBG_CFG_DMI_EN);
                traps = 2;
            }
            if (!fw_run_until(r, n0 + 1u, 0, 1, what)) goto out;
            for (int it = 0; it < 100 && rt_phase(r) != TSR_RT_STATE_PHASE_TRAPPED; ++it) tsr_sim_advance(sim, 1000);
            fw_consume(r);
            const rec8 * x = at(r, n0);
            const uint32_t mepc = t == 0 ? G.s_trap_insn : (t == 1 || t == 3) ? G.s_panic_point : x->r[3];
            const uint32_t mtval = tc[t].mtval;
            const uint32_t rts = TSR_RT_STATE_PHASE_KERNEL | TSR_PUT(TSR_RT_STATE_KERNEL, G.k_trap);
            CHECK(x->r[0] == (TSR_LCQ_R0_TYPE_TRAP | TSR_PUT(TSR_LCQ_R0_STATUS, TSR_LCQ_R0_STATUS_TRAP) |
                              TSR_PUT(TSR_LCQ_R0_SEQ, s)) && x->r[1] == s && x->r[2] == tc[t].mcause && x->r[3] == mepc &&
                  x->r[4] == mtval && x->r[5] == rts && x->r[6] == tc[t].info,
                  "%s: %s: TRAP record %08x %u %u %08x %08x %08x %08x", r->sched, what, x->r[0], x->r[1], x->r[2], x->r[3],
                  x->r[4], x->r[5], x->r[6]);
            CHECK(tsr_sim_csr_read(sim, TSR_CSR_LCQ_ANSWERED) == r->pushed, "%s: %s: TRAP answered the launch", r->sched, what);
            CHECK(tsr_sim_csr_read(sim, TSR_CSR_RT_STATE) == (TSR_RT_STATE_PHASE_TRAPPED | TSR_PUT(TSR_RT_STATE_KERNEL, G.k_trap)),
                  "%s: %s: RT_STATE TRAPPED (0x%08x)", r->sched, what, tsr_sim_csr_read(sim, TSR_CSR_RT_STATE));
            // __tsr_crash (DB §6.3) and the debug info
            const uint8_t * c = dtcm + crash_off;
            const uint32_t sp = get_u32_le(c + 4u * (TSR_CRASH_X1_WORD + 1u));
            CHECK(get_u32_le(c) == TSR_CRASH_MAGIC && get_u32_le(c + 4) == 1u &&
                  get_u32_le(c + 4u * TSR_CRASH_MCAUSE_WORD) == tc[t].mcause && get_u32_le(c + 4u * TSR_CRASH_MEPC_WORD) == mepc &&
                  get_u32_le(c + 4u * TSR_CRASH_MTVAL_WORD) == mtval && get_u32_le(c + 4u * TSR_CRASH_SEQ_WORD) == s &&
                  get_u32_le(c + 4u * TSR_CRASH_KERNEL_WORD) == G.k_trap &&
                  get_u32_le(c + 4u * TSR_CRASH_RT_STATE_WORD) == rts &&
                  get_u32_le(c + 4u * TSR_CRASH_RT_INFO_WORD) == tc[t].info &&
                  get_u32_le(c + 4u * TSR_CRASH_NESTED_WORD) == 1u && sp >= G.s_stack_bottom && sp < G.s_stack_top,
                  "%s: %s: __tsr_crash", r->sched, what);
            CHECK(get_u32_le(dtcm + dbg_off + 4u * TSR_DBG_INFO_TRAP_COUNT_WORD) == traps, "%s: %s: trap_count", r->sched, what);
            if (t == 0u)
                CHECK(sp == dbg_sp && get_u32_le(c + 4u * TSR_CRASH_X1_WORD) == dbg_ra,
                      "%s: %s: the crash record holds the context the debugger saw", r->sched, what);
            // Log: (panic: its FATAL FMT record), then the TRAP record = MTIME + words 0..43 of __tsr_crash
            const uint32_t nl = log_parse(r, l0, lr, 64);
            const uint32_t want = (t == 1 || t == 3) ? 2u : 1u;
            CHECK(nl == want, "%s: %s: %u log records, expected %u", r->sched, what, nl, want);
            if (nl == want) {
                if (want == 2u) {
                    chk_lrec(r, &lr[0], 0, 0, TSR_LOG_HDR_TYPE_FMT, 5, s, "I panic FATAL");
                    chk_fmt(r, &lr[0], "panic 0x%08x (0x%08x, 0x%08x, 0x%08x)", "I panic FATAL");
                    const uint32_t a0 = t == 1 ? 1u : 0x7Fu, a1 = t == 1 ? 2u : 0u, a2 = t == 1 ? 3u : 0u;
                    CHECK(lr[0].p[1] == mtval && lr[0].p[2] == a0 && lr[0].p[3] == a1 && lr[0].p[4] == a2,
                          "%s: %s: panic arguments", r->sched, what);
                }
                const lrec * tr = &lr[want - 1u];
                chk_lrec(r, tr, 0, 2, TSR_LOG_HDR_TYPE_TRAP, 44, s, "I TRAP log record");
                int same = 1;
                for (uint32_t k = 0; k < 44u; ++k) same &= tr->p[k] == get_u32_le(c + 4u * k);
                CHECK(same, "%s: %s: the TRAP log record carries __tsr_crash", r->sched, what);
            }
            warm_restart(r, what);
        }
        // After the restarts: PING and a copy
        const uint32_t n0 = r->n_rec;
        const uint32_t s1 = ping(r, 0x77u);
        const uint32_t s2 = launch(r, G.k_copy, 0, H(0, SRC_BEAT + 20u), H(1, DST_BEAT + 0x600u), 4, 500, 0, 0, 0);
        if (!fw_run_until(r, n0 + 2u, 0, 0, "I after")) goto out;
        CHECK(rtype(at(r, n0)) == TSR_LCQ_R0_TYPE_PONG && at(r, n0)->r[1] == s1, "%s: I: PONG after warm restart", r->sched);
        chk_done(r, n0 + 1u, s2, TSR_LCQ_R0_STATUS_OK, 0, 0, "I copy after warm restart");
        quiesce(r, "I");
        uint8_t got[4 * 64];
        tsr_sim_dram_read(sim, 1, (uint64_t)(DST_BEAT + 0x600u) * 64u, got, sizeof(got));
        CHECK(memcmp(got, src + 20u * 64u, sizeof(got)) == 0, "%s: I: copy output", r->sched);
    }

    // ======== J: a trap outside a launch (R1's kernel-table check panics at boot): EX's TRAP record
    // as an event (r1 = TAKEN, LCQ_ANSWERED unchanged), then a warm restart with the intact image ========
    {
        tsr_sim_csr_write(sim, TSR_CSR_CTRL, tsr_sim_csr_read(sim, TSR_CSR_CTRL) | TSR_CTRL_SOFT_RESET);
        for (int it = 0; it < 1000 && (tsr_sim_csr_read(sim, TSR_CSR_STATUS) & TSR_STATUS_SR_BUSY); ++it)
            tsr_sim_advance(sim, 10000);
        r->lcq_rd = tsr_sim_csr_read(sim, TSR_CSR_LCQ_RD);
        r->log_rd = tsr_sim_csr_read(sim, TSR_CSR_LOG_RD);
        uint32_t kt = 0;
        CHECK(tsr_elf_symbol(G.e, "__tsr_kernels_start", &kt) == 0, "J: __tsr_kernels_start");
        tsr_sim_csr_write(sim, TSR_CSR_LOAD_ADDR, kt + 8u);              // entry 0: nargs | flags | argblk_lines
        tsr_sim_csr_write(sim, TSR_CSR_LOAD_DATA, 9u);                     // nargs 9 > 7
        const uint32_t n0 = r->n_rec, l0 = r->n_log, answered = tsr_sim_csr_read(sim, TSR_CSR_LCQ_ANSWERED);
        tsr_sim_csr_write(sim, TSR_CSR_CTRL, tsr_sim_csr_read(sim, TSR_CSR_CTRL) | TSR_CTRL_RV_RUN);
        if (!fw_run_until(r, n0 + 1u, 0, 1, "J")) goto out;
        for (int it = 0; it < 100 && rt_phase(r) != TSR_RT_STATE_PHASE_TRAPPED; ++it) tsr_sim_advance(sim, 1000);
        fw_consume(r);
        const rec8 * x = at(r, n0);
        const uint32_t taken = r->pushed;
        CHECK(x->r[0] == (TSR_LCQ_R0_TYPE_TRAP | TSR_PUT(TSR_LCQ_R0_STATUS, TSR_LCQ_R0_STATUS_TRAP) |
                          TSR_PUT(TSR_LCQ_R0_SEQ, taken)) && x->r[1] == taken && x->r[2] == 24u && x->r[3] == G.s_panic_point &&
              x->r[4] == 0x52540003u && x->r[5] == TSR_RT_STATE_PHASE_BOOT && x->r[6] == 0u,
              "%s: J: TRAP event %08x %u %u %08x %08x %08x %08x", r->sched, x->r[0], x->r[1], x->r[2], x->r[3], x->r[4],
              x->r[5], x->r[6]);
        CHECK(tsr_sim_csr_read(sim, TSR_CSR_LCQ_ANSWERED) == answered && rt_phase(r) == TSR_RT_STATE_PHASE_TRAPPED,
              "%s: J: an event (nothing answered), RT_STATE TRAPPED", r->sched);
        static lrec lj[8];
        const uint32_t nl = log_parse(r, l0, lj, 8);
        CHECK(nl == 2u && lj[0].type == TSR_LOG_HDR_TYPE_FMT && lj[0].lseq == 0u && lj[0].p[1] == 0x52540003u &&
              lj[0].p[2] == 0u && lj[1].type == TSR_LOG_HDR_TYPE_TRAP && lj[1].lseq == 0u,
              "%s: J: FATAL panic + TRAP log records outside a launch (LSEQ 0)", r->sched);
        warm_restart(r, "J");
        const uint32_t n1 = r->n_rec;
        const uint32_t s1 = launch(r, G.k_nop, 0, 0, 0, 0, 0, 0, 0, 0);
        if (!fw_run_until(r, n1 + 1u, 0, 0, "J after")) goto out;
        chk_done(r, n1, s1, TSR_LCQ_R0_STATUS_OK, 0, 0, "J launch after the restart");
        quiesce(r, "J");
    }

    // ======== K (fix W2a): an RV reset leaves every unit in error state (RV_RUN 1 -> 0 applies QABORT;
    // error states are kept, CC §12.1); R1's QCLEAR at boot makes the next launch clean ========
    {
        tsr_sim_csr_write(sim, TSR_CSR_CTRL, tsr_sim_csr_read(sim, TSR_CSR_CTRL) & ~TSR_CTRL_RV_RUN);
        for (int it = 0; it < 100 && !(tsr_sim_csr_read(sim, TSR_CSR_STATUS) & TSR_STATUS_RV_RESET_DONE); ++it)
            tsr_sim_advance(sim, 1000);
        CHECK((tsr_sim_csr_read(sim, TSR_CSR_STATUS) & TSR_STATUS_QERR) != 0u,
              "%s: K: after RV_RUN 1 -> 0 the units are in error state (STATUS 0x%08x)", r->sched,
              tsr_sim_csr_read(sim, TSR_CSR_STATUS));
        reload_and_boot(r, "K");
        const uint32_t n1 = r->n_rec;
        const uint32_t s1 = launch(r, G.k_copy, 0, H(0, SRC_BEAT + 28u), H(1, DST_BEAT + 0x800u), 4, 700, 0, 0, 0);
        if (!fw_run_until(r, n1 + 1u, 0, 0, "K")) goto out;
        chk_done(r, n1, s1, TSR_LCQ_R0_STATUS_OK, 0, 0, "K copy after the RV reset (R1's QCLEAR)");
        quiesce(r, "K");
        uint8_t got[4 * 64];
        tsr_sim_dram_read(sim, 1, (uint64_t)(DST_BEAT + 0x800u) * 64u, got, sizeof(got));
        CHECK(memcmp(got, src + 28u * 64u, sizeof(got)) == 0, "%s: K: copy output", r->sched);
    }

    // ======== markers: LAUNCH_BEGIN / END around every kernel call (DB §8.4); the four trapped
    // launches have no END ========
    {
        uint32_t begins = 0, ends = 0, ok = 1;
        for (uint32_t i = 0; i < r->n_marks; ++i) {
            const uint32_t m = r->marks[i], cls = TSR_GET(m, TSR_MARK_CLASS);
            if (cls == TSR_MARK_CLASS_LAUNCH_BEGIN) {
                ++begins;
            } else {
                ++ends;
                ok &= cls == TSR_MARK_CLASS_LAUNCH_END && i > 0u &&
                      r->marks[i - 1u] == ((m & ~TSR_MARK_CLASS) | TSR_PUT(TSR_MARK_CLASS, TSR_MARK_CLASS_LAUNCH_BEGIN));
            }
        }
        CHECK(ok && begins == 48u && ends == 44u, "%s: markers: %u BEGIN, %u END (expected 48, 44), pairs %s", r->sched,
              begins, ends, ok ? "ok" : "broken");
        const uint32_t m0 = r->n_marks ? r->marks[0] : 0u;
        CHECK(TSR_GET(m0, TSR_MARK_LAUNCH_KERNEL) == G.k_args[0] && TSR_GET(m0, TSR_MARK_LAUNCH_SEQ) == 1u,
              "%s: first marker 0x%08x = LAUNCH_BEGIN rt_args0 seq 1", r->sched, m0);
    }
    {
        const uint32_t st = tsr_sim_csr_read(sim, TSR_CSR_STATUS);
        CHECK(!(st & (TSR_STATUS_LQ_OVERFLOW | TSR_STATUS_RING_RD_BAD | TSR_STATUS_LOG_DROP | TSR_STATUS_QERR)),
              "%s: no sticky errors at the end (STATUS 0x%08x)", r->sched, st);
    }
    {
        tsr_race_stats rs;
        CHECK(tsr_sim_race_active(sim) && tsr_sim_race_stats(sim, &rs) == 0 && rs.reports == 0u && rs.rv_accesses > 0u &&
              rs.ops > 0u, "%s: the race detector ran (fatal) and found nothing", r->sched);
    }
    obs_finish(&r->ob, sim);
    r->mtime_end = tsr_sim_now(sim);
    r->instret = tsr_sim_rv_instret(sim);
out:
    tsr_sim_destroy(sim);
    r->sim = NULL;
}

// The log words with what is timing (TS words) or machine state (the TRAP payload) zeroed.
static void log_masked(const run * r, uint32_t * out) {
    memcpy(out, r->logw, r->n_log * sizeof(uint32_t));
    for (uint32_t i = 0; i < r->n_log;) {
        tsr_log_hdr h;
        tsr_log_hdr_decode(r->logw[i], &h);
        if (h.sync != TSR_LOG_HDR_SYNC_MAGIC) break;
        if (h.ts && i + 1u < r->n_log) out[i + 1u] = 0u;
        if (h.type == TSR_LOG_HDR_TYPE_TRAP)
            for (uint32_t k = 1; k <= h.n && i + k < r->n_log; ++k) out[i + k] = 0u;
        i += 1u + h.n;
    }
}

static void compare(const run * a, const run * b) {
    char what[96];
    snprintf(what, sizeof(what), "%s vs %s", b->sched, a->sched);
    int ok = a->n_rec == b->n_rec;
    for (uint32_t i = 0; ok && i < a->n_rec; ++i)
        for (uint32_t j = 0; j < 7u; ++j) ok &= a->rec[i].r[j] == b->rec[i].r[j];
    CHECK(ok, "%s: LCQ records r0-r6 differ (%u vs %u records)", what, a->n_rec, b->n_rec);
    obs_equal(&a->ob, &b->ob, what);
    if (a->n_log == b->n_log) {
        uint32_t * x = (uint32_t *)malloc(a->n_log * sizeof(uint32_t) + 4u);
        uint32_t * y = (uint32_t *)malloc(b->n_log * sizeof(uint32_t) + 4u);
        log_masked(a, x);
        log_masked(b, y);
        CHECK(memcmp(x, y, a->n_log * sizeof(uint32_t)) == 0, "%s: log words differ", what);
        free(x);
        free(y);
    } else {
        CHECK(0, "%s: %u vs %u log words", what, a->n_log, b->n_log);
    }
    CHECK(a->n_marks == b->n_marks && memcmp(a->marks, b->marks, a->n_marks * sizeof(uint32_t)) == 0,
          "%s: TRACE_MARK values differ", what);
}

// ---------------------------------------------------------------------------------------------
// Launch costs (EX §4.11), fixed schedule, default F_RV / F_CORE
// ---------------------------------------------------------------------------------------------
static void measure(void) {
    tsr_config cfg = TSR_CONFIG_U200_V1;
    cfg.num_ch = 2;
    cfg.ch_bytes = 1ull << 30;
    cfg.caps = TSR_CAPS_VU;
    tsr_sim_opts o;
    memset(&o, 0, sizeof(o));
    o.sched = "fixed";
    tsr_sim * sim = tsr_sim_create_opts(&cfg, &o);
    if (sim == NULL) return;
    run r;
    memset(&r, 0, sizeof(r));
    r.sched = "measure";
    r.sim = sim;
    tsr_sim_add_listener(sim, q_listener, &r);
    load_image(sim, &G.rv);
    tsr_sim_csr_write(sim, TSR_CSR_CTRL, TSR_CTRL_RV_RUN);
    fw_run_until(&r, 1, 0, 0, "measure boot");
    quiesce(&r, "measure boot");
    enum { N = 16 };
    uint64_t entry = 0, rtt = 0, dur = 0, entry_min = ~0ull, rtt_min = ~0ull;
    for (int k = 0; k < N; ++k) {
        const uint32_t n0 = r.n_rec, m0 = r.n_marks;
        const uint64_t t0 = tsr_sim_now(sim);
        const uint32_t s = launch(&r, G.k_nop, 0, 0, 0, 0, 0, 0, 0, 0);
        uint64_t t_done = 0;
        for (int it = 0; it < 100000 && tsr_sim_csr_read(sim, TSR_CSR_LCQ_WR) == r.lcq_rd; ++it) tsr_sim_advance(sim, 100000);
        t_done = tsr_sim_now(sim);
        fw_consume(&r);
        CHECK(r.n_rec == n0 + 1u && r.rec[n0].r[1] == s && r.n_marks >= m0 + 1u, "measure: launch %d", k);
        if (r.n_rec != n0 + 1u || r.n_marks < m0 + 1u) break;
        const uint64_t e = r.mark_time[m0] - t0, d = t_done - t0;
        entry += e;
        rtt += d;
        dur += r.rec[n0].r[7];
        if (e < entry_min) entry_min = e;
        if (d < rtt_min) rtt_min = d;
        quiesce(&r, "measure");
    }
    // 8 outstanding
    const uint32_t n0 = r.n_rec;
    const uint64_t t0 = tsr_sim_now(sim);
    for (int k = 0; k < 8; ++k) launch(&r, G.k_nop, 0, 0, 0, 0, 0, 0, 0, 0);
    fw_run_until(&r, n0 + 8u, 0, 0, "measure pipelined");
    const uint64_t t8 = tsr_sim_now(sim) - t0;
    const double f = 276.3;   // core MHz (U200)
    printf("fw_runtime launch costs (fixed schedule, F_RV/F_CORE %.4f, core cycles at %.1f MHz):\n", tsr_sim_rv_ratio(sim), f);
    printf("  descriptor push -> kernel entry (LAUNCH_BEGIN): mean %.0f (min %llu) = %.2f us   [EX §4.11 target <= 5 us]\n",
           (double)entry / N, (unsigned long long)entry_min, (double)entry / N / f);
    printf("  empty launch, push -> DONE in the LCQ:         mean %.0f (min %llu) = %.2f us   [device part of the <= 10 us round trip]\n",
           (double)rtt / N, (unsigned long long)rtt_min, (double)rtt / N / f);
    printf("  DONE r7 DURATION (pop -> post):                mean %.0f = %.2f us\n", (double)dur / N, (double)dur / N / f);
    printf("  8 outstanding empty launches:                  %.0f per launch = %.2f us (%.0f K launches/s)\n", (double)t8 / 8.0,
           (double)t8 / 8.0 / f, 8.0 * f * 1e3 / (double)t8);
    run_free(&r);
    tsr_sim_destroy(sim);
}

// fw_runtime_stale: the fw_rt image built with RT_MUTANT_STALE (rt_blk1 reads all 16 lines of the
// argument range, the runtime staged one): the race detector (report mode) must name exactly that
// read as SPM_STALE in its launch, and nothing else.
static int stale_main(const char * path) {
    if (!img_open(path)) return test_finish("fw_runtime_stale");
    tsr_config cfg = TSR_CONFIG_U200_V1;
    cfg.num_ch = 2;
    cfg.ch_bytes = 1ull << 30;
    cfg.caps = TSR_CAPS_VU;
    tsr_sim_opts o;
    memset(&o, 0, sizeof(o));
    o.sched = "fixed";
    o.race = 1;                                            // report: collect, do not stop
    tsr_sim * sim = tsr_sim_create_opts(&cfg, &o);
    CHECK(sim != NULL, "stale: create");
    if (sim == NULL) return test_finish("fw_runtime_stale");
    run r;
    memset(&r, 0, sizeof(r));
    r.sched = "stale";
    r.sim = sim;
    const uint32_t L = cfg.spm_bytes / 64u;
    uint8_t blk[64];
    memset(blk, 0x5A, sizeof(blk));
    tsr_sim_dram_write(sim, 1, (uint64_t)BLK1_BEAT * 64u, blk, sizeof(blk));
    load_image(sim, &G.rv);
    tsr_sim_csr_write(sim, TSR_CSR_CTRL, TSR_CTRL_RV_RUN);
    fw_run_until(&r, 1, 0, 0, "stale boot");
    const uint32_t s = launch(&r, G.k_blk1, 0, H(1, BLK1_BEAT), 0, 0, 0, 0, 0, 0);
    fw_run_until(&r, 3, 0, 0, "stale launch");
    quiesce(&r, "stale");
    const tsr_race_report * rep = NULL;
    const size_t n = tsr_sim_race_reports(sim, &rep);
    char line[512] = "";
    if (n > 0u) tsr_race_format(&rep[0], line, sizeof(line));
    CHECK(n == 1u && rep[0].kind == TSR_RACE_SPM_STALE && rep[0].space == TSR_RACE_SPACE_SPM &&
          rep[0].first.agent == TSR_RACE_AGENT_RV && rep[0].launch == s && rep[0].lo >= 64ull * (L - 15u) &&
          rep[0].hi <= 64ull * L, "stale: one SPM_STALE report of the RISC-V read beyond the staged line (%zu: %s)", n, line);
    printf("fw_runtime_stale: %s\n", line);
    run_free(&r);
    tsr_sim_destroy(sim);
    tsr_rv_elf_free(&G.rv);
    tsr_elf_free(G.e);
    return test_finish("fw_runtime_stale");
}

int main(int argc, char ** argv) {
    if (argc == 3 && strcmp(argv[1], "--stale") == 0) return stale_main(argv[2]);
    if (argc < 2) {
        fprintf(stderr, "usage: test_fw_runtime <fw_rt.elf> | --stale <fw_rt_stale.elf>\n");
        return 2;
    }
    unsetenv("TSR_SIM_SCHED");
    unsetenv("TSR_SIM_LOG");
    unsetenv("TSR_SIM_RV_RATIO");
    unsetenv("TSR_TRACE");
    setenv("TSR_SIM_RACE", "fatal", 1);
    if (!img_open(argv[1])) return test_finish("fw_runtime");
    printf("fw_runtime: %s: image id 0x%016llx, %u kernels, %s build\n", argv[1], (unsigned long long)G.id, G.n_kernels,
           G.debug_build ? "debug" : "release");

    static run R[N_SCHED];
    for (int k = 0; k < N_SCHED; ++k) {
        memset(&R[k], 0, sizeof(R[k]));
        R[k].sched = k_sched[k];
        R[k].ratio = k_ratio[k];
        fw_scenario(&R[k]);
        printf("fw_runtime %-9s F_RV/F_CORE %-6s: %u records, %u log words, %u markers, MTIME %llu, %llu RV instructions\n",
               k_sched[k], k_ratio[k] == 1.5 ? "1.5" : k_ratio[k] == 0.25 ? "0.25" : "def", R[k].n_rec, R[k].n_log,
               R[k].n_marks, (unsigned long long)R[k].mtime_end, (unsigned long long)R[k].instret);
    }
    int distinct = 0;
    for (int k = 0; k < N_SCHED; ++k) {
        if (k > 0) compare(&R[0], &R[k]);
        int same = 0;
        for (int j = 0; j < k; ++j) same |= R[j].mtime_end == R[k].mtime_end;
        distinct += !same;
    }
    CHECK(distinct >= 8, "the schedules change the timing (%d distinct final MTIMEs of %d)", distinct, N_SCHED);
    for (int k = 0; k < N_SCHED; ++k) run_free(&R[k]);
    measure();
    tsr_rv_elf_free(&G.rv);
    tsr_elf_free(G.e);
    return test_finish("fw_runtime");
}
