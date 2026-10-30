// test_sim_w1b_e2e.c — ctest sim_w1b_e2e (W1b integration, docs/design/README.md §10 W1): the W1b
// models together on the sim_rv_issue firmware (tests/rv_issue/fw_issue.c, host side and every
// check in test_sim_rvhost.h). Every run has the DM, DBG, RACE (TSR_SIM_RACE=fatal: a race
// aborts) and TRACE models attached and captures its launches (TSR_TRACE=<dir> and
// tsr_sim_capture_launch).
//
//   A. Determinism (SW §4.4, §4.7): fixed + random:1..7. Equal across the 8 runs: the LCQ records
//      r0-r6, the log, the DRAM outputs and s0, the per-unit op streams (relative tickets), the
//      RISC-V's SPM store stream, the final written DRAM beats and SPM lines (test_sim_obs.h), and
//      the captured launch traces (push records, DONE, tickets.bin, init / expect regions, image);
//      no race report; the schedules change the timing (distinct final MTIMEs).
//   B. Trace round trip (SW §5.4, §5.5, §10.1 "launch traces"): the fixed run's traces read back
//      from its TSR_TRACE directory equal the in-memory captures and replay on fresh simulators
//      under fixed, random:11 and random:12,0.5,3; a random run's traces replay under fixed.
//   C. A GDB-style DMI session (DB §3.4-§3.7, §5.8, §9.1) through host CSR 0xC00 + 4r with
//      DBG_CFG.DMI_EN and CTRL.HOLD_ON_HALT set at attach, as the debug server does: halt in the
//      middle of the first kernel launch (after its LD and QACT pushes), read x1-x31 (against the
//      simulator's debug view), dpc, dcsr, and the code at dpc (ITCM), SPM (both windows) and the
//      stack (DTCM) through SBA (against the memories), single-step 5 instructions (one retirement
//      each, cause step), set hardware trigger 0 at the kernel's first push (the call site of the
//      push helper for the LD, located in the undisturbed run), resume, hit it at the next launch's
//      kernel before its first push, remove it, resume to completion. The session's run passes
//      every sim_rv_issue check and its observables (A's list, traces included) equal the
//      undisturbed fixed run; a second identical session is cycle-for-cycle reproducible (DB §9.1:
//      a GDB session on sim with a fixed schedule is reproducible).
//
// Usage: sim_w1b_e2e <fw_issue.elf> <scratch dir>. "Skipped" (77) when the ELF was not built (no
// RISC-V compiler at configure time); CMake adds the test only when all four models are built in.
#include "test_sim_rvhost.h"      // first: test_sim_util.h defines _GNU_SOURCE
#include "test_sim_obs.h"

#include <dirent.h>

#include "tsr_dm.h"
#include "tsr_race.h"
#include "tsr_trace.h"

#define N_SCHED   8
#define MAX_TR    8u
#define N_STEP    5
static const char * const k_sched[N_SCHED] = {
    "fixed", "random:1", "random:2", "random:3", "random:4", "random:5", "random:6", "random:7",
};
static const char * const k_replay_sched[3] = { "fixed", "random:11", "random:12,0.5,3" };

static const char * g_scratch = "/tmp";

// ---------------------------------------------------------------------------
// One run: observables, captured traces, push tracking, an optional debugger session
// ---------------------------------------------------------------------------
typedef struct dbg_session dbg_session;

typedef struct e2e_run {
    const char * sched;
    observed    rec;                    // records, log, outputs, s0 (test_sim_rvhost.h)
    obs         ob;                     // SW §4.7 observables
    tsr_trace2  tr[MAX_TR];             // captured launches, in answer order
    uint32_t    n_tr;
    tsr_trace_capture_stats cs;
    int         models;                 // bit i: model i attached (TSR_MODEL_*), race fatal
    size_t      n_race;
    uint64_t    mtime, instret, ff_jumps;
    // pushes seen (PUSH queue events)
    int         any_push;
    uint32_t    last_pidx;
    // the reference run: the push PC and ra (call site + insn size) at the PUSH of push index want_pidx
    uint32_t    want_pidx;
    int         got_want;
    uint32_t    push_pc, push_ra;
    dbg_session * dbg;
} e2e_run;

static void e2e_q(void * ctx, tsr_sim * sim, const tsr_q_event * ev) {
    e2e_run * r = (e2e_run *)ctx;
    if (ev->kind != TSR_QEV_PUSH) return;
    r->any_push = 1;
    r->last_pidx = ev->pidx;
    if (r->want_pidx != 0u && ev->pidx == r->want_pidx && !r->got_want) {
        r->got_want = 1;
        r->push_pc = ev->pc;
        r->push_ra = tsr_rv_reg(tsr_rv_agent_rv(tsr_sim_rv_agent(sim)), 1);   // inside the leaf push helper
    }
}

static void e2e_create(tsr_sim * sim, void * ctx) {
    e2e_run * r = (e2e_run *)ctx;
    r->models = 0;
    if (tsr_sim_dm(sim) != NULL && *tsr_sim_model_slot(sim, TSR_MODEL_DM) != NULL) r->models |= 1;
    if (tsr_sim_csr_read(sim, TSR_CSR_DBG_ID) != 0u && *tsr_sim_model_slot(sim, TSR_MODEL_DBG) != NULL) r->models |= 2;
    if (tsr_sim_opt_race(sim) == 2u && *tsr_sim_model_slot(sim, TSR_MODEL_RACE) != NULL) r->models |= 4;
    if (tsr_sim_opt_trace_dir(sim) != NULL && *tsr_sim_model_slot(sim, TSR_MODEL_TRACE) != NULL) r->models |= 8;
    obs_attach(&r->ob, sim);
    tsr_sim_add_listener(sim, e2e_q, r);
    tsr_sim_capture_launch(sim, 1);
}

static void e2e_end(tsr_sim * sim, void * ctx) {
    e2e_run * r = (e2e_run *)ctx;
    obs_finish(&r->ob, sim);
    for (r->n_tr = 0; r->n_tr < MAX_TR; ++r->n_tr) {
        tsr_trace2_init(&r->tr[r->n_tr]);
        if (tsr_sim_capture_take(sim, &r->tr[r->n_tr]) != 0) {
            tsr_trace2_clear(&r->tr[r->n_tr]);
            break;
        }
    }
    tsr_sim_capture_stats(sim, &r->cs);
    const tsr_race_report * rr = NULL;
    r->n_race = tsr_sim_race_reports(sim, &rr);
    r->mtime = tsr_sim_now(sim);
    r->instret = tsr_sim_rv_instret(sim);
    r->ff_jumps = tsr_rv_agent_ff_jumps(tsr_sim_rv_agent(sim));
}

static void e2e_free(e2e_run * r) {
    obs_free(&r->ob);
    for (uint32_t i = 0; i < r->n_tr; ++i) tsr_trace2_clear(&r->tr[i]);
    r->n_tr = 0;
}

// ---------------------------------------------------------------------------
// Trace comparison: everything SW §5.4 compares and the trace carries (not sched, rv_ratio, perf)
// ---------------------------------------------------------------------------
static int regions_equal(const tsr_region * a, size_t na, const tsr_region * b, size_t nb) {
    if (na != nb) return 0;
    for (size_t i = 0; i < na; ++i) {
        if (a[i].ch != b[i].ch || a[i].byte_off != b[i].byte_off || a[i].size != b[i].size) return 0;
        if (a[i].size != 0 && memcmp(a[i].data, b[i].data, (size_t)a[i].size) != 0) return 0;
    }
    return 1;
}

static int trace_equal(const tsr_trace2 * a, const tsr_trace2 * b, const char * what) {
    const char * diff = NULL;
    if (a->kind != b->kind) diff = "kind";
    else if (a->image_id != b->image_id || a->image_size != b->image_size ||
             (a->image_size && memcmp(a->image, b->image, a->image_size) != 0)) diff = "image";
    else if (a->kernel_index != b->kernel_index || a->prof != b->prof || memcmp(a->args, b->args, sizeof(a->args)) != 0)
        diff = "launch descriptor";
    else if (a->n_push != b->n_push || (a->n_push && memcmp(a->push, b->push, a->n_push * sizeof(tsr_trace_push)) != 0))
        diff = "push records";
    else if (memcmp(&a->done, &b->done, sizeof(a->done)) != 0) diff = "DONE";
    else if (a->n_units != b->n_units || memcmp(a->unit, b->unit, a->n_units * sizeof(tsr_trace_unit)) != 0) diff = "tickets";
    else if (!regions_equal(a->mem.init, a->mem.n_init, b->mem.init, b->mem.n_init)) diff = "init regions";
    else if (!regions_equal(a->mem.expect, a->mem.n_expect, b->mem.expect, b->mem.n_expect)) diff = "expect regions";
    else if (a->cfg.core.num_ch != b->cfg.core.num_ch || a->cfg.core.t_max != b->cfg.core.t_max ||
             a->cfg.core.spm_bytes != b->cfg.core.spm_bytes || a->cfg.core.ch_bytes != b->cfg.core.ch_bytes ||
             a->cfg.core.caps != b->cfg.core.caps || a->cfg.ctl_id != b->cfg.ctl_id || a->cfg.cfg3 != b->cfg.cfg3 ||
             a->cfg.itcm_bytes != b->cfg.itcm_bytes || a->cfg.dtcm_bytes != b->cfg.dtcm_bytes ||
             memcmp(a->cfg.q_depth, b->cfg.q_depth, sizeof(a->cfg.q_depth)) != 0) diff = "config";
    CHECK(diff == NULL, "%s: captured traces differ (%s)", what, diff ? diff : "");
    return diff == NULL;
}

// Everything a run is compared on: records, log, outputs, SW §4.7 observables, traces.
static void run_equal(const e2e_run * a, const e2e_run * b, const char * what) {
    CHECK(b->rec.n_rec == 13u && b->rec.n_rec == a->rec.n_rec && memcmp(b->rec.rec, a->rec.rec, sizeof(a->rec.rec)) == 0,
          "%s: LCQ records r0-r6 equal", what);
    CHECK(b->rec.n_log == a->rec.n_log && memcmp(b->rec.log, a->rec.log, a->rec.n_log * sizeof(uint32_t)) == 0,
          "%s: log equal", what);
    CHECK(b->rec.out_hash == a->rec.out_hash && b->rec.s0 == a->rec.s0, "%s: outputs and s0 equal", what);
    obs_equal(&a->ob, &b->ob, what);
    CHECK(b->n_tr == a->n_tr, "%s: %u captured traces, reference %u", what, b->n_tr, a->n_tr);
    for (uint32_t i = 0; i < a->n_tr && i < b->n_tr; ++i) {
        char w[160];
        snprintf(w, sizeof(w), "%s, trace %u", what, i);
        trace_equal(&a->tr[i], &b->tr[i], w);
    }
}

static void run_checks(const e2e_run * r, const char * what) {
    CHECK(r->models == 15, "%s: DM, DBG, RACE (fatal) and TRACE attached (mask 0x%x)", what, r->models);
    CHECK(r->n_race == 0u, "%s: no race report (%zu)", what, r->n_race);
    // 8 descriptors: 2 PINGs and the unknown TYPE are not captured; 4 kernel launches and the
    // launch of a kernel beyond the table (a DONE BAD_DESC answer) are.
    CHECK(r->cs.captured == 5u && r->cs.written == 5u && r->cs.skipped == 3u && r->cs.failed == 0u && r->n_tr == 5u,
          "%s: captures %u written %u skipped %u failed %u taken %u", what, r->cs.captured, r->cs.written, r->cs.skipped,
          r->cs.failed, r->n_tr);
}

static void run_one(const char * elf, e2e_run * r, const char * sched, const char * trace_dir, dbg_session * dbg,
                    void (*on_advance)(tsr_sim *, void *)) {
    memset(&r->rec, 0, sizeof(r->rec));
    r->sched = sched;
    r->dbg = dbg;
    rm_rf(trace_dir);
    setenv("TSR_TRACE", trace_dir, 1);                // read by tsr_sim_create_opts
    const rvh_run run = { 0.0, sched, e2e_create, e2e_end, r, on_advance };
    scenario(elf, &run, &r->rec);
    unsetenv("TSR_TRACE");
}

// ---------------------------------------------------------------------------
// C. The debugger session (DMI through host CSR 0xC00 + 4r, DB §3.2, §3.5, §3.6)
// ---------------------------------------------------------------------------
struct dbg_session {
    e2e_run * run;
    uint32_t  bp;                        // trigger address: the kernel's first push (call site)
    uint32_t  bp_pidx;                   // push index of that push in the next kernel launch
    int       state;                     // 0 waiting, 1 done, -1 failed
    // what the session saw
    uint32_t  halt_pidx, halt_dpc, step_dpc[N_STEP], hit_dpc, hit_pidx, hit_answered;
    uint64_t  halt_time, hit_time;
};

static tsr_sim * DS;                     // the simulator of the hook in progress
static uint32_t dR(uint32_t a) { return tsr_sim_csr_read(DS, TSR_CSR_DMI(a)); }
static void dW(uint32_t a, uint32_t v) { tsr_sim_csr_write(DS, TSR_CSR_DMI(a), v); }
static int dpoll(uint32_t bit, uint64_t max_cycles, uint64_t chunk) {
    for (uint64_t c = 0; c <= max_cycles; c += chunk) {
        if (dR(TSR_DMI_DMSTATUS) & bit) return 1;
        tsr_sim_advance(DS, chunk);
    }
    return (dR(TSR_DMI_DMSTATUS) & bit) != 0;
}
static uint32_t dacs(void) {             // R 0x16 until busy = 0; cmderr
    for (int k = 0; k < 1000; ++k) {
        const uint32_t v = dR(TSR_DMI_ABSTRACTCS);
        if (!(v & TSR_DM_ABSTRACTCS_BUSY)) return (v & TSR_DM_ABSTRACTCS_CMDERR) >> TSR_DM_ABSTRACTCS_CMDERR_LSB;
        tsr_sim_advance(DS, 100);
    }
    return 0xFFu;
}
static uint32_t dreg(uint32_t regno) {   // Access Register read (DB §3.5)
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
static void dsba_read(uint32_t a, uint32_t n, uint32_t * out) {   // DB §3.6 n-word read
    dW(TSR_DMI_SBCS, 0x0055F000u);
    dW(TSR_DMI_SBADDRESS0, a);
    for (uint32_t i = 0; i + 1u < n; ++i) out[i] = dR(TSR_DMI_SBDATA0);
    dW(TSR_DMI_SBCS, 0x00457000u);
    out[n - 1u] = dR(TSR_DMI_SBDATA0);
}
static uint32_t dsba_err(void) { return dR(TSR_DMI_SBCS) & (TSR_DM_SBCS_SBBUSYERROR | TSR_DM_SBCS_SBERROR); }
static uint32_t dcause(void) { return (dreg(TSR_RV_CSR_DCSR) >> 6) & 7u; }
static int dresume(void) {
    dW(TSR_DMI_DMCONTROL, 0x40000001u);
    const int ok = dpoll(TSR_DM_DMSTATUS_ALLRESUMEACK, 1000, 50);
    dW(TSR_DMI_DMCONTROL, 0x00000001u);
    return ok;
}

static int words_equal(const uint32_t * w, const uint8_t * mem, uint32_t n) {
    for (uint32_t i = 0; i < n; ++i) if (w[i] != get_u32_le(mem + 4u * i)) return 0;
    return 1;
}

static void dbg_hook(tsr_sim * sim, void * ctx) {
    e2e_run * r = (e2e_run *)ctx;
    dbg_session * d = r->dbg;
    if (d == NULL || d->state != 0) return;
    if (tsr_sim_csr_read(sim, TSR_CSR_LQ_SEQ) < N_LAUNCH) return;   // the host has not pushed every launch yet
    DS = sim;
    d->state = -1;
    tsr_rv * hart = tsr_rv_agent_rv(tsr_sim_rv_agent(sim));
    // Mid-launch: the first kernel launch has pushed its LD and QACT (push indices 1, 2).
    for (int k = 0; k < 100000 && !(r->any_push && r->last_pidx >= 2u); ++k) tsr_sim_advance(sim, 50);
    CHECK(r->any_push && r->last_pidx >= 2u && r->last_pidx < 7u, "session: inside the first kernel (last push index %u)", r->last_pidx);
    // Attach (the server: DMI_EN, HOLD_ON_HALT), dmactive, halt.
    tsr_sim_csr_write(sim, TSR_CSR_DBG_CFG, tsr_sim_csr_read(sim, TSR_CSR_DBG_CFG) | TSR_DBG_CFG_DMI_EN);
    tsr_sim_csr_write(sim, TSR_CSR_CTRL, tsr_sim_csr_read(sim, TSR_CSR_CTRL) | TSR_CTRL_HOLD_ON_HALT);
    dW(TSR_DMI_DMCONTROL, 0x00000001u);
    dW(TSR_DMI_DMCONTROL, 0x80000001u);
    const int halted = dpoll(TSR_DM_DMSTATUS_ALLHALTED, 200000, 50);
    dW(TSR_DMI_DMCONTROL, 0x00000001u);
    CHECK(halted, "session: halted");
    if (!halted) return;
    d->halt_time = tsr_sim_now(sim);
    d->halt_pidx = r->last_pidx;
    CHECK(dcause() == TSR_RV_DBG_CAUSE_HALTREQ, "session: dcsr.cause haltreq");
    CHECK(tsr_sim_csr_read(sim, TSR_CSR_LCQ_ANSWERED) == 1u &&
          TSR_GET(tsr_sim_csr_read(sim, TSR_CSR_RT_STATE), TSR_RT_STATE_PHASE) == TSR_RT_STATE_PHASE_KERNEL,
          "session: halted mid-launch (only the PING answered, RT_STATE KERNEL)");
    // Registers (GDB 'g'): x1-x31 against the simulator's debug view; dpc in ITCM.
    int regs_ok = 1;
    for (unsigned i = 1; i < 32u; ++i) regs_ok &= dreg(TSR_DM_REGNO_GPR(i)) == tsr_sim_rv_reg(sim, i);
    CHECK(regs_ok, "session: x1-x31 through Access Register = the hart's registers");
    uint32_t itcm_bytes = 0;
    const uint8_t * itcm = tsr_sim_itcm(sim, &itcm_bytes);
    d->halt_dpc = dreg(TSR_RV_CSR_DPC);
    CHECK(d->halt_dpc >= TSR_BOOT_ENTRY_OFF && d->halt_dpc < itcm_bytes, "session: dpc 0x%08x in ITCM", d->halt_dpc);
    // Memory through SBA: the code at dpc in ITCM, SPM (exact and line-buffered windows), the stack in DTCM.
    uint32_t w[32];
    const uint32_t code = (d->halt_dpc & ~3u) < itcm_bytes - 16u ? (d->halt_dpc & ~3u) : 0u;
    dsba_read(TSR_RV_ITCM_BASE + code, 4, w);
    CHECK(dsba_err() == 0u && words_equal(w, itcm + code, 4), "session: the code at dpc through SBA (ITCM)");
    const uint8_t * spm = tsr_sim_spm(sim);
    dsba_read(TSR_RV_SPM_BASE, 32, w);
    CHECK(dsba_err() == 0u && words_equal(w, spm, 32), "session: SPM lines 0-1 through SBA (0x4000_0000)");
    dsba_read(TSR_RV_SPM_LB_BASE + 64u * 8u, 32, w);
    CHECK(dsba_err() == 0u && words_equal(w, spm + 64u * 8u, 32), "session: SPM lines 8-9 through SBA (line-buffered alias)");
    const uint32_t sp = dreg(TSR_DM_REGNO_GPR(2));
    uint32_t dtcm_bytes = 0;
    const uint8_t * dtcm = tsr_sim_dtcm(sim, &dtcm_bytes);
    CHECK(sp >= TSR_RV_DTCM_BASE + 32u && sp <= TSR_RV_DTCM_BASE + dtcm_bytes, "session: sp 0x%08x in DTCM", sp);
    if (sp >= TSR_RV_DTCM_BASE && sp + 64u <= TSR_RV_DTCM_BASE + dtcm_bytes) {
        dsba_read(sp, 16, w);
        CHECK(dsba_err() == 0u && words_equal(w, dtcm + (sp - TSR_RV_DTCM_BASE), 16), "session: stack words through SBA");
    }
    // Single step 5 instructions (dcsr.step; DB §3.4).
    dregw(TSR_RV_CSR_DCSR, dreg(TSR_RV_CSR_DCSR) | 4u);
    for (int k = 0; k < N_STEP; ++k) {
        const uint64_t n0 = tsr_rv_retired(hart);
        dW(TSR_DMI_DMCONTROL, 0x40000001u);
        const int ok = dpoll(TSR_DM_DMSTATUS_ALLHALTED, 200000, 50);
        dW(TSR_DMI_DMCONTROL, 0x00000001u);
        d->step_dpc[k] = dreg(TSR_RV_CSR_DPC);
        CHECK(ok && dcause() == TSR_RV_DBG_CAUSE_STEP && tsr_rv_retired(hart) == n0 + 1u,
              "session: step %d retires one instruction (dpc 0x%08x)", k, d->step_dpc[k]);
    }
    CHECK(d->step_dpc[0] != d->halt_dpc, "session: the steps move dpc");
    dregw(TSR_RV_CSR_DCSR, dreg(TSR_RV_CSR_DCSR) & ~4u);
    // Hardware breakpoint (DB §3.7): trigger 0 at the kernel's first push.
    dregw(TSR_RV_CSR_TSELECT, 0);
    dregw(TSR_RV_CSR_TDATA1, 0x28001044u);
    dregw(TSR_RV_CSR_TDATA2, d->bp);
    CHECK(dreg(TSR_RV_CSR_TDATA2) == d->bp, "session: tdata2 = 0x%08x", d->bp);
    CHECK(dresume(), "session: resume");
    const int hit = dpoll(TSR_DM_DMSTATUS_ALLHALTED, 5000000, 500);
    CHECK(hit, "session: breakpoint hit");
    if (!hit) return;
    d->hit_time = tsr_sim_now(sim);
    d->hit_dpc = dreg(TSR_RV_CSR_DPC);
    d->hit_pidx = r->last_pidx;
    d->hit_answered = tsr_sim_csr_read(sim, TSR_CSR_LCQ_ANSWERED);
    CHECK(dcause() == TSR_RV_DBG_CAUSE_TRIGGER && d->hit_dpc == d->bp, "session: trigger at 0x%08x (dpc 0x%08x)", d->bp, d->hit_dpc);
    CHECK(d->hit_pidx + 1u == d->bp_pidx && d->hit_answered == 2u &&
          TSR_GET(tsr_sim_csr_read(sim, TSR_CSR_RT_STATE), TSR_RT_STATE_PHASE) == TSR_RT_STATE_PHASE_KERNEL,
          "session: before the second kernel launch's first push (last push %u, answered %u)", d->hit_pidx, d->hit_answered);
    // Remove it, resume to completion, detach.
    dregw(TSR_RV_CSR_TDATA1, 0);
    CHECK(dresume(), "session: resume to completion");
    tsr_sim_csr_write(sim, TSR_CSR_CTRL, tsr_sim_csr_read(sim, TSR_CSR_CTRL) & ~TSR_CTRL_HOLD_ON_HALT);
    tsr_sim_csr_write(sim, TSR_CSR_DBG_CFG, tsr_sim_csr_read(sim, TSR_CSR_DBG_CFG) & ~TSR_DBG_CFG_DMI_EN);
    d->state = 1;
}

// The call site of the push helper from the PUSH event's (pc, ra): ra - 2 (c.jal / c.jalr) or ra - 4
// (jal / jalr), whichever is a call whose target can hold the push store at pc.
static uint32_t call_site(const uint8_t * itcm, uint32_t itcm_bytes, uint32_t pc, uint32_t ra) {
    uint32_t found = 0;
    int n = 0;
    if (ra >= 4u && ra <= itcm_bytes) {
        const uint32_t i = get_u32_le(itcm + ra - 4u);
        if ((i & 0xFFFu) == 0x0EFu) {                                            // jal ra, imm
            const uint32_t imm = ((i >> 31) & 1u) << 20 | ((i >> 21) & 0x3FFu) << 1 | ((i >> 20) & 1u) << 11 | ((i >> 12) & 0xFFu) << 12;
            const uint32_t t = ra - 4u + (imm | (0u - (imm & 0x100000u)));
            if (t <= pc && pc - t < 1024u) { found = ra - 4u; ++n; }
        } else if ((i & 0xFFFu) == 0x0E7u) {                                     // jalr ra, ...
            found = ra - 4u;
            ++n;
        }
    }
    if (ra >= 2u && ra <= itcm_bytes) {
        const uint32_t h = (uint32_t)itcm[ra - 2u] | (uint32_t)itcm[ra - 1u] << 8;
        if ((h & 0xE003u) == 0x2001u) {                                          // c.jal
            const uint32_t imm = ((h >> 12) & 1u) << 11 | ((h >> 11) & 1u) << 4 | ((h >> 9) & 3u) << 8 | ((h >> 8) & 1u) << 10 |
                                 ((h >> 7) & 1u) << 6 | ((h >> 6) & 1u) << 7 | ((h >> 3) & 7u) << 1 | ((h >> 2) & 1u) << 5;
            const uint32_t t = ra - 2u + (imm | (0u - (imm & 0x800u)));
            if (t <= pc && pc - t < 1024u) { found = ra - 2u; ++n; }
        } else if ((h & 0xF07Fu) == 0x9002u && (h & 0x0F80u) != 0u) {           // c.jalr
            found = ra - 2u;
            ++n;
        }
    }
    return n == 1 ? found : 0u;
}

// ---------------------------------------------------------------------------
// Trace directories written by TSR_TRACE (<dir>/<seq>/ or <dir>/sim<k>/<seq>/), in name order
// ---------------------------------------------------------------------------
static int name_cmp(const void * a, const void * b) { return strcmp(*(char * const *)a, *(char * const *)b); }

static uint32_t find_traces(const char * dir, char ** out, uint32_t max) {
    uint32_t n = 0;
    DIR * dd = opendir(dir);
    if (dd == NULL) return 0;
    char * names[64];
    uint32_t nn = 0;
    for (struct dirent * e; (e = readdir(dd)) != NULL;) {
        if (e->d_name[0] == '.' || strcmp(e->d_name, "images") == 0 || nn == 64u) continue;
        names[nn++] = strdup(e->d_name);
    }
    closedir(dd);
    qsort(names, nn, sizeof(char *), name_cmp);
    for (uint32_t i = 0; i < nn; ++i) {
        char p[1024], m[1100];
        snprintf(p, sizeof(p), "%s/%s", dir, names[i]);
        snprintf(m, sizeof(m), "%s/meta.json", p);
        struct stat st;
        if (stat(m, &st) == 0) {
            if (n < max) out[n++] = strdup(p);
        } else if (stat(p, &st) == 0 && S_ISDIR(st.st_mode)) {
            n += find_traces(p, out + n, max - n);
        }
        free(names[i]);
    }
    return n;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: sim_w1b_e2e <fw_issue.elf> [scratch]\n");
        return 2;
    }
    const char * elf = argv[1];
    if (argc > 2) g_scratch = argv[2];
    FILE * f = fopen(elf, "rb");
    if (f == NULL) {
        printf("sim_w1b_e2e: %s not built (no RISC-V compiler at configure time); skipped\n", elf);
        return 77;
    }
    fclose(f);
    char * const base = make_scratch(g_scratch, "sim_w1b_e2e");   // a fresh directory per run
    unsetenv("TSR_SIM_SCHED");
    unsetenv("TSR_SIM_LOG");
    unsetenv("TSR_SIM_RV_RATIO");
    setenv("TSR_SIM_RACE", "fatal", 1);

    // ---- A. fixed + 7 random schedules ----
    static e2e_run R[N_SCHED];
    char dir[N_SCHED][800];
    for (int k = 0; k < N_SCHED; ++k) {
        memset(&R[k], 0, sizeof(R[k]));
        if (k == 0) R[k].want_pidx = 8u;            // the second kernel launch's first push (its LD)
        snprintf(dir[k], sizeof(dir[k]), "%s/run%d", base, k);
        run_one(elf, &R[k], k_sched[k], dir[k], NULL, NULL);
        run_checks(&R[k], k_sched[k]);
    }
    int distinct = 0;
    for (int k = 0; k < N_SCHED; ++k) {
        char what[96];
        snprintf(what, sizeof(what), "A, %s vs fixed", k_sched[k]);
        if (k > 0) run_equal(&R[0], &R[k], what);
        int same = 0;
        for (int j = 0; j < k; ++j) same |= R[j].mtime == R[k].mtime;
        distinct += !same;
        printf("sim_w1b_e2e A %-9s: MTIME %8llu, %6llu RV instructions, %3llu fast-forward jumps, %u traces\n", k_sched[k],
               (unsigned long long)R[k].mtime, (unsigned long long)R[k].instret, (unsigned long long)R[k].ff_jumps, R[k].n_tr);
    }
    CHECK(distinct >= 6, "A: the schedules change the timing (%d distinct final MTIMEs of %d)", distinct, N_SCHED);

    // ---- B. the captured traces on disk, replayed on fresh simulators ----
    char * tdir[MAX_TR];
    const uint32_t nd = find_traces(dir[0], tdir, MAX_TR);
    CHECK(nd == R[0].n_tr, "B: %u trace directories under %s (captured %u)", nd, dir[0], R[0].n_tr);
    uint32_t n_replay = 0, n_ok = 0;
    for (uint32_t i = 0; i < nd; ++i) {
        tsr_trace2 * t = NULL;
        char err[512];
        const int rd = tsr_trace2_read(tdir[i], &t, err, sizeof(err));
        CHECK(rd == 0, "B: read %s: %s", tdir[i], rd == 0 ? "" : err);
        if (rd != 0) continue;
        CHECK(t->kind == TSR_TRACE2_KIND_LAUNCH, "B: %s is a launch trace", tdir[i]);
        if (i < R[0].n_tr) {
            char what[1100];
            snprintf(what, sizeof(what), "B, %s vs its in-memory capture", tdir[i]);
            trace_equal(&R[0].tr[i], t, what);
        }
        for (int s = 0; s < 3; ++s) {
            tsr_trace2_result res;
            memset(&res, 0, sizeof(res));
            const int rc = tsr_trace2_replay_sim(t, NULL, k_replay_sched[s], &res);
            ++n_replay;
            n_ok += rc == 0 && res.ok;
            CHECK(rc == 0 && res.ok, "B: replay %s under %s: %s", tdir[i], k_replay_sched[s], res.msg);
        }
        tsr_trace2_free(t);
        free(tdir[i]);
    }
    for (uint32_t i = 0; i < R[3].n_tr; ++i) {               // captured under random:3, replayed under fixed
        tsr_trace2_result res;
        memset(&res, 0, sizeof(res));
        const int rc = tsr_trace2_replay_sim(&R[3].tr[i], NULL, "fixed", &res);
        ++n_replay;
        n_ok += rc == 0 && res.ok;
        CHECK(rc == 0 && res.ok, "B: random:3 trace %u replayed under fixed: %s", i, res.msg);
    }
    printf("sim_w1b_e2e B: %u of %u replays on fresh simulators passed\n", n_ok, n_replay);

    // ---- C. the GDB-style session, twice, against the undisturbed fixed run ----
    uint32_t itcm_bytes = TSR_ITCM_BYTES;
    tsr_rv_elf img;
    uint8_t * itcm = (uint8_t *)calloc(itcm_bytes, 1);
    if (tsr_rv_elf_read(elf, &img) == 0) {                   // the image's ITCM bytes, for call_site
        for (uint32_t s = 0; s < img.nseg; ++s)
            if (img.seg[s].paddr < itcm_bytes && img.seg[s].filesz <= itcm_bytes - img.seg[s].paddr)
                memcpy(itcm + img.seg[s].paddr, img.seg[s].data, img.seg[s].filesz);
        tsr_rv_elf_free(&img);
    }
    const uint32_t bp = R[0].got_want ? call_site(itcm, itcm_bytes, R[0].push_pc, R[0].push_ra) : 0u;
    free(itcm);
    CHECK(R[0].got_want && bp != 0u, "C: the kernel's first push: call site 0x%08x (push pc 0x%08x, ra 0x%08x)", bp,
          R[0].push_pc, R[0].push_ra);
    static e2e_run D[2];
    dbg_session ds[2];
    for (int k = 0; k < 2 && bp != 0u; ++k) {
        memset(&D[k], 0, sizeof(D[k]));
        memset(&ds[k], 0, sizeof(ds[k]));
        ds[k].run = &D[k];
        ds[k].bp = bp;
        ds[k].bp_pidx = 8u;
        char ddir[800];
        snprintf(ddir, sizeof(ddir), "%s/debug%d", base, k);
        run_one(elf, &D[k], "fixed", ddir, &ds[k], dbg_hook);
        run_checks(&D[k], "C, session run");
        CHECK(ds[k].state == 1, "C: session %d completed", k);
        run_equal(&R[0], &D[k], "C, session run vs the undisturbed run");
        printf("sim_w1b_e2e C session %d: halted at MTIME %llu after push %u (dpc 0x%08x), steps to", k,
               (unsigned long long)ds[k].halt_time, ds[k].halt_pidx, ds[k].halt_dpc);
        for (int s = 0; s < N_STEP; ++s) printf(" 0x%x", ds[k].step_dpc[s]);
        printf(", trigger 0x%08x hit at MTIME %llu; end MTIME %llu (undisturbed %llu)\n", bp,
               (unsigned long long)ds[k].hit_time, (unsigned long long)D[k].mtime, (unsigned long long)R[0].mtime);
    }
    if (bp != 0u) {
        CHECK(ds[0].halt_time == ds[1].halt_time && ds[0].halt_dpc == ds[1].halt_dpc &&
              memcmp(ds[0].step_dpc, ds[1].step_dpc, sizeof(ds[0].step_dpc)) == 0 && ds[0].hit_time == ds[1].hit_time &&
              D[0].mtime == D[1].mtime && D[0].instret == D[1].instret,
              "C: the session is reproducible under the fixed schedule");
        CHECK(D[0].mtime != R[0].mtime, "C: the session changed the timing (MTIME %llu vs %llu)",
              (unsigned long long)D[0].mtime, (unsigned long long)R[0].mtime);
    }
    for (int k = 0; k < N_SCHED; ++k) e2e_free(&R[k]);
    for (int k = 0; k < 2; ++k) e2e_free(&D[k]);
    if (!g_fail) rm_rf(base);
    else fprintf(stderr, "sim_w1b_e2e: scratch kept in %s\n", base);
    free(base);
    return test_finish("sim_w1b_e2e");
}
