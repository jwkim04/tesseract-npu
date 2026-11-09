// test_sim_race.c — the race detector and SPM_STALE (SW §4.6, SW-3 / SW-4; tsr_race.c).
//
// Modes (sw/sim/race.cmake):
//   sim_race tile                      ctest sim_race: directed tile programs through the test hooks
//                                      (no RISC-V): RAW / WAR / WAW across units without deps
//                                      reported (SPM, DRAM), the same with a dep, a FENCE or a QCLEAR
//                                      not, same-unit queue order not, s-register hazards against
//                                      the host, host DRAM against an op, the harness's
//                                      run-until-idle edge (keyed on TSR_SIM_PC_HOOK: a hart access at
//                                      pc 0 is no harness access), a 1000-op random program (determinism of
//                                      the reports), fatal mode in a child (abort with the report)
//   sim_race rv <fw_issue.elf> <fw_issue_nodep.elf> <fw_race.elf>
//                                      ctest sim_race_rv: firmware on the built-in hart. (1) the
//                                      sim_rv_issue firmware under fatal: no report; (2) its mutant
//                                      without the ST -> VOP dependency: reported, with outputs
//                                      bit-identical to (1)'s (only the detector catches it);
//                                      fw_race.c's kernels: (4) store then push ordered, op write
//                                      then RISC-V read without a wait reported, with a DONE_SEQ
//                                      wait or the WAIT interrupt not, a store after a push
//                                      reported; (5) SPM_STALE for an op and for the RISC-V, not
//                                      after a write in the launch, and across launches; (6) H1 / H2
//                                      host DRAM writes and reads before / after LQ_PUSH and the
//                                      answer, and H2's own join (kernel 12 answers without the R6
//                                      join: an ST completed before the answer read is ordered, one
//                                      still running at it is not). Exit 77 when an ELF is missing
//                                      (no RISC-V compiler).
//   sim_race bench <n_ops> <race>      overhead measurement (not a ctest): an n-op random program,
//                                      wall time with race 0 / 1 / 2.
#include "test_sim_util.h"

#include "tsr_bus.h"
#include "tsr_race.h"
#include "tsr_sim_int.h"

#include <fcntl.h>
#include <signal.h>
#include <sys/wait.h>
#include <time.h>

static const uint32_t ALL = 0xFF00u;
static const uint32_t NONE = TSR_TICKET_NONE;

static tsr_sim * mk(uint32_t race) {
    tsr_config c = cfg_u200(1);
    c.num_ch = 2;
    c.ch_bytes = 1ull << 28;
    c.caps = TSR_CAPS_VU;
    tsr_sim_opts o;
    memset(&o, 0, sizeof(o));
    o.race = race;
    tsr_sim * s = tsr_sim_create_opts(&c, &o);
    CHECK(s != NULL, "create (race %u)", race);
    if (s == NULL) exit(1);
    CHECK(race == 0 || tsr_sim_race_active(s), "the detector is active");
    return s;
}

static uint32_t push(tsr_sim * s, tsr_insn i, uint32_t d0, uint32_t d1) { return tsr_sim_push(s, i.w, d0, d1, ALL); }
static void idle(tsr_sim * s) { CHECK(tsr_sim_run_until_idle(s, 100000000ull), "run until idle"); }

static void dump(const tsr_sim * s, const char * tag) {
    const tsr_race_report * r;
    const size_t n = tsr_sim_race_reports(s, &r);
    char line[768];
    for (size_t i = 0; i < n; ++i) {
        tsr_race_format(&r[i], line, sizeof(line));
        printf("  [%s] %s\n", tag, line);
    }
}

// Reports of a kind mask (bit kind) between the agents a and b (either order; b = NONE for stale).
static size_t count_rep(const tsr_sim * s, uint32_t kinds, uint32_t space, uint32_t a, uint32_t b) {
    const tsr_race_report * r;
    const size_t n = tsr_sim_race_reports(s, &r);
    size_t k = 0;
    for (size_t i = 0; i < n; ++i) {
        if (!((kinds >> r[i].kind) & 1u) || (space && r[i].space != space)) continue;
        const uint32_t x = r[i].first.agent, y = r[i].second.agent;
        if ((x == a && y == b) || (x == b && y == a)) ++k;
    }
    return k;
}
#define K_RACE  ((1u << TSR_RACE_RAW) | (1u << TSR_RACE_WAR) | (1u << TSR_RACE_WAW))
#define K_RW    ((1u << TSR_RACE_RAW) | (1u << TSR_RACE_WAR))
#define K_WAW   (1u << TSR_RACE_WAW)
#define K_STALE (1u << TSR_RACE_SPM_STALE)
#define A_RV    TSR_RACE_AGENT_RV
#define A_HOST  TSR_RACE_AGENT_HOST
#define A_NONE  TSR_RACE_AGENT_NONE

static size_t n_rep(const tsr_sim * s) { return tsr_sim_race_reports(s, NULL); }

static tsr_insn enc_red_s(uint32_t sd, uint32_t a, uint32_t m) {
    tsr_insn i = {{0}};
    i.w[0] = TSR_OP_RED;
    TSR_INSN_SET(i.w, TSR_RED_KIND, TSR_RED_KIND_SUM);
    TSR_INSN_SET(i.w, TSR_RED_TO_S, 1u);
    TSR_INSN_SET(i.w, TSR_RED_SD, sd);
    TSR_INSN_SET(i.w, TSR_RED_POST, TSR_RED_POST_NONE);
    TSR_INSN_SET(i.w, TSR_RED_A, a);
    TSR_INSN_SET(i.w, TSR_RED_M, m);
    TSR_INSN_SET(i.w, TSR_RED_ROWS, 1u);
    return i;
}

static tsr_insn enc_vop1(uint32_t d, uint32_t a, uint32_t m) {
    tsr_insn i = {{0}};
    i.w[0] = TSR_OP_VOP;
    TSR_INSN_SET(i.w, TSR_VOP_CMODE, TSR_VOP_CMODE_NONE);
    TSR_INSN_SET(i.w, TSR_VOP_FUNC, TSR_VOP_FUNC_SILU);
    TSR_INSN_SET(i.w, TSR_VOP_D, d);
    TSR_INSN_SET(i.w, TSR_VOP_A, a);
    TSR_INSN_SET(i.w, TSR_VOP_M, m);
    TSR_INSN_SET(i.w, TSR_VOP_ROWS, 1u);
    return i;
}

// ---------------------------------------------------------------------------
// (3) Directed tile programs (the test hooks act for the RISC-V; RV_RUN = 0)
// ---------------------------------------------------------------------------
// A pair of ops on two units touching the same SPM lines, pushed back to back (nothing has
// completed when the second is pushed, so the harness gets no idle edge).
typedef enum { ORD_NONE, ORD_DEP, ORD_FENCE, ORD_QCLEAR } ordering;

static void pair_case(const char * name, tsr_insn a, tsr_insn b, ordering ord, uint32_t ua, uint32_t ub,
                      uint32_t kinds, uint32_t space) {
    tsr_sim * s = mk(1);
    const uint32_t ta = push(s, a, NONE, NONE);
    uint32_t dep = NONE;
    if (ord == ORD_DEP) dep = ta;
    if (ord == ORD_FENCE) dep = push(s, tsr_enc_fence(), NONE, NONE);
    if (ord == ORD_QCLEAR) dep = tsr_sim_qctrl(s, TSR_QCTRL_QCLEAR);
    CHECK(ord == ORD_NONE || dep != NONE, "%s: ordering entry pushed", name);
    (void)push(s, b, dep, NONE);
    idle(s);
    const size_t want = ord == ORD_NONE ? 1u : 0u;
    CHECK(n_rep(s) == want && count_rep(s, kinds, space, ua, ub) == want, "%s (%s): %zu reports, expected %zu", name,
          ord == ORD_NONE ? "no order" : ord == ORD_DEP ? "dep" : ord == ORD_FENCE ? "FENCE" : "QCLEAR", n_rep(s), want);
    if (n_rep(s) != want) dump(s, name);
    tsr_sim_destroy(s);
}

static void test_tile_pairs(void) {
    // QACT reads 4 f32 lines (K = 64) and writes QA_LINES(A32, 64) lines.
    const uint32_t qal = tsr_qa_lines(TSR_QA_A32, 64);
    for (int o = ORD_NONE; o <= ORD_QCLEAR; ++o) {
        // RAW / WAR across DMA and QACT: LD writes 100..103, QACT reads them.
        pair_case("LD -> QACT reads", tsr_enc_ld(0, 100, 0, 4), tsr_enc_qact(TSR_QA_A32, 200, 100, 64, 1, 0), (ordering)o,
                  TSR_UNIT_DMA, TSR_UNIT_QACT, K_RW, TSR_RACE_SPACE_SPM);
        // WAR: ST reads 300.., QACT overwrites them with its QA output.
        pair_case("ST reads -> QACT writes", tsr_enc_st(0, 300, 64, qal), tsr_enc_qact(TSR_QA_A32, 300, 400, 64, 1, 0),
                  (ordering)o, TSR_UNIT_DMA, TSR_UNIT_QACT, K_RW, TSR_RACE_SPACE_SPM);
        // WAW: LD and QACT both write 500..
        pair_case("LD writes, QACT writes", tsr_enc_ld(0, 500, 128, qal), tsr_enc_qact(TSR_QA_A32, 500, 600, 64, 1, 0),
                  (ordering)o, TSR_UNIT_DMA, TSR_UNIT_QACT, K_WAW, TSR_RACE_SPACE_SPM);
        // VU vs DMA: VOP writes 700, ST reads it.
        pair_case("VOP writes -> ST reads", enc_vop1(700, 710, 16), tsr_enc_st(1, 700, 256, 1), (ordering)o,
                  TSR_UNIT_VU, TSR_UNIT_DMA, K_RW, TSR_RACE_SPACE_SPM);
        // DRAM: ST writes ch0 beat 1000; MVM reads its weights from beat 1000 on every channel.
        pair_case("ST writes DRAM -> MVM weights", tsr_enc_st(0, 800, 1000, 1),
                  tsr_enc_mvm(TSR_FMT_Q8_0, 0, 900, 810, 1000, 16, 256, 1, 0), (ordering)o, TSR_UNIT_DMA, TSR_UNIT_MVM,
                  K_RW, TSR_RACE_SPACE_DRAM);
    }
    // Same unit, no dep: queue order (E1).
    {
        tsr_sim * s = mk(1);
        (void)push(s, tsr_enc_ld(0, 100, 0, 4), NONE, NONE);
        (void)push(s, tsr_enc_st(1, 100, 0, 4), NONE, NONE);         // reads what the LD wrote
        (void)push(s, tsr_enc_ld(0, 100, 64, 4), NONE, NONE);        // overwrites what the ST read
        (void)push(s, enc_vop1(120, 100, 64), NONE, NONE);           // VU: unordered with the 2nd LD
        (void)push(s, enc_vop1(100, 130, 16), NONE, NONE);           // VU writes 100 (WAW with LD 2)
        idle(s);
        CHECK(count_rep(s, K_RACE, 0, TSR_UNIT_DMA, TSR_UNIT_DMA) == 0, "same-unit queue order: no DMA-DMA report");
        CHECK(count_rep(s, K_RW, TSR_RACE_SPACE_SPM, TSR_UNIT_DMA, TSR_UNIT_VU) >= 1 &&
              count_rep(s, K_WAW, TSR_RACE_SPACE_SPM, TSR_UNIT_DMA, TSR_UNIT_VU) >= 1, "VU vs DMA reported");
        CHECK(count_rep(s, K_RACE, 0, TSR_UNIT_VU, TSR_UNIT_VU) == 0, "same-unit queue order: no VU-VU report");
        tsr_sim_destroy(s);
    }
    // A chain through a third unit orders transitively: LD -> QACT (dep) -> VOP (dep on QACT).
    {
        tsr_sim * s = mk(1);
        const uint32_t t1 = push(s, tsr_enc_ld(0, 100, 0, 4), NONE, NONE);
        const uint32_t t2 = push(s, tsr_enc_qact(TSR_QA_A32, 200, 300, 64, 1, 0), t1, NONE);
        (void)push(s, enc_vop1(400, 100, 64), t2, NONE);             // reads the LD's lines
        idle(s);
        CHECK(n_rep(s) == 0, "transitive order: %zu reports", n_rep(s));
        tsr_sim_destroy(s);
    }
}

// s registers (16 locations): RED TO_S writes s0 on the VU; the host reads SREG0 (EX §7.4).
static void test_sreg(void) {
    tsr_sim * s = mk(1);
    (void)push(s, enc_red_s(0, 100, 16), NONE, NONE);
    (void)tsr_sim_csr_read(s, TSR_CSR_SREG(0));                     // before the RED completed
    idle(s);
    CHECK(count_rep(s, K_RW, TSR_RACE_SPACE_SREG, TSR_UNIT_VU, A_HOST) == 1, "SREG: host read vs RED write reported");
    const tsr_race_report * r;
    if (tsr_sim_race_reports(s, &r) == 1) CHECK(r[0].lo == 0 && r[0].hi == 1 && r[0].kind == TSR_RACE_WAR, "SREG: s0, WAR");
    tsr_sim_race_clear(s);
    (void)push(s, enc_red_s(1, 100, 16), NONE, NONE);
    idle(s);                                                         // the harness saw QIDLE
    (void)tsr_sim_csr_read(s, TSR_CSR_SREG(1));
    (void)push(s, enc_red_s(1, 100, 16), NONE, NONE);                // after the read: ordered
    idle(s);
    CHECK(n_rep(s) == 0, "SREG after run-until-idle: %zu reports", n_rep(s));
    // SOP reads s registers, RED writes them: both on the VU, ordered by the queue.
    tsr_sim_destroy(s);
}

// Host DRAM against ops in harness mode (RV_RUN = 0: the test is the RISC-V and the host).
// Masked KVST preserves replay preimages without a physical metadata read. Both capture modes
// must visit only the QA source lines and two written DRAM beats in the race model.
static void test_kvst_capture_accesses(void) {
    for (int capturing = 0; capturing < 2; ++capturing) {
        tsr_config c = cfg_u200(1);
        c.num_ch = 2;
        c.ch_bytes = 1ull << 28;
        c.caps = TSR_CAPS_VU | TSR_CAPS_ATT;
        tsr_sim_opts o = {0};
        o.race = 1;
        tsr_sim * s = tsr_sim_create_opts(&c, &o);
        CHECK(s != NULL, "masked KVST race simulator");
        if (s == NULL) continue;
        // Direct initialization creates no host access or shadow segmentation.
        memset(tsr_sim_dram(s, 0) + 1000u * 64u, 0xA5, 128u);
        if (capturing) tsr_sim_capture_begin(s);
        tsr_race_stats before, after;
        CHECK(tsr_sim_race_stats(s, &before) == 0, "KVST race stats before");
        tsr_insn i = {{0}};
        i.w[0] = TSR_OP_KVST;
        TSR_INSN_SET(i.w, TSR_KVST_KV, TSR_KVST_KV_V);
        TSR_INSN_SET(i.w, TSR_KVST_SRC, 100u);
        TSR_INSN_SET(i.w, TSR_KVST_BASE, 1000u);
        TSR_INSN_SET(i.w, TSR_KVST_E64, 1u);
        TSR_INSN_SET(i.w, TSR_KVST_LG, 1u);
        push(s, i, NONE, NONE);
        idle(s);
        CHECK(tsr_sim_csr_read(s, TSR_CSR_Q_ERR_CODE(TSR_UNIT_DMA)) == 0u, "KVST ran");
        CHECK(tsr_sim_race_stats(s, &after) == 0, "KVST race stats after");
        CHECK(after.op_lines - before.op_lines == tsr_qa_lines(TSR_QA_A32, 64u) + 2u,
              "KVST capture %d: QA reads and two DRAM writes only, got %llu visits", capturing,
              (unsigned long long)(after.op_lines - before.op_lines));
        CHECK(n_rep(s) == 0, "KVST has no race");
        if (capturing) {
            tsr_capture cap;
            tsr_sim_capture_end(s, &cap);
            CHECK(cap.n_init == 1 && cap.init[0].ch == 0 && cap.init[0].byte_off == 64000u &&
                  cap.init[0].size == 64u, "partial META preimage remains captured");
            CHECK(cap.n_expect == 1 && cap.expect[0].ch == 0 && cap.expect[0].byte_off == 64000u &&
                  cap.expect[0].size == 128u, "META and quant output beats remain captured");
            tsr_capture_free(&cap);
        }
        tsr_sim_destroy(s);
    }
}

static void test_host_harness(void) {
    uint8_t buf[256];
    memset(buf, 0x3C, sizeof(buf));
    tsr_sim * s = mk(1);
    CHECK(tsr_sim_dram_write(s, 0, 64 * 64, buf, 256) == 0, "dram_write");
    (void)push(s, tsr_enc_ld(0, 100, 64, 4), NONE, NONE);            // after the write: ordered
    idle(s);
    CHECK(n_rep(s) == 0, "host write, then push: %zu reports", n_rep(s));
    (void)push(s, tsr_enc_ld(0, 100, 64, 4), NONE, NONE);
    CHECK(tsr_sim_dram_write(s, 0, 64 * 64, buf, 256) == 0, "dram_write");   // while the LD is pending
    idle(s);
    CHECK(count_rep(s, K_RW, TSR_RACE_SPACE_DRAM, A_HOST, TSR_UNIT_DMA) == 1, "push, then host write: reported");
    tsr_sim_race_clear(s);
    (void)push(s, tsr_enc_st(1, 100, 32, 4), NONE, NONE);
    CHECK(tsr_sim_dram_read(s, 1, 32 * 64, buf, 256) == 0, "dram_read");     // the ST has not run
    idle(s);
    CHECK(count_rep(s, K_RW, TSR_RACE_SPACE_DRAM, A_HOST, TSR_UNIT_DMA) == 1, "host read before the ST: reported");
    tsr_sim_race_clear(s);
    (void)push(s, tsr_enc_st(1, 100, 32, 4), NONE, NONE);
    idle(s);
    CHECK(tsr_sim_dram_read(s, 1, 32 * 64, buf, 256) == 0, "dram_read");
    memset(tsr_sim_dram(s, 1) + 32 * 64, 0, 64);                     // through the pointer: invisible
    tsr_sim_dram_note_write(s, 1, 32 * 64, 64);                      // ... unless noted
    idle(s);
    CHECK(n_rep(s) == 0, "after run-until-idle: %zu reports", n_rep(s));
    tsr_sim_destroy(s);
}

// The harness rule keys on TSR_SIM_PC_HOOK, not on pc 0: 0 is a hart pc (the ITCM base, vector slot 0
// with RV_BOOT = 0). With RV_RUN = 1 (a stub agent: no hart runs) an LD pushed at pc 0 and an SPM
// read at pc 0 of the line it wrote, with no observation between, are the hart's: a race. The
// same accesses from the test hooks' pc are the harness's (merged clocks, the idle join): none.
static uint64_t stub_step(void * ctx, tsr_sim * sim) { (void)ctx; (void)sim; return TSR_SIM_AGENT_WAIT; }
static void pc_case(uint32_t pc, size_t want) {
    tsr_sim * s = mk(1);
    tsr_sim_set_rv_agent(s, stub_step, NULL);
    tsr_sim_csr_write(s, TSR_CSR_CTRL, TSR_CTRL_RV_RUN);
    const tsr_insn ld = tsr_enc_ld(0, 100, 64, 1);
    uint32_t v;
    for (uint32_t k = 0; k < 8; ++k) {
        v = ld.w[k];
        (void)tsr_sim_cw_access_pc(s, 1, TSR_CW_ISSUE_W(k), &v, pc);
    }
    v = NONE;
    (void)tsr_sim_cw_access_pc(s, 1, TSR_CW_ISSUE_DEP0, &v, pc);
    v = NONE;
    (void)tsr_sim_cw_access_pc(s, 1, TSR_CW_ISSUE_DEP1, &v, pc);
    v = ALL;
    CHECK(tsr_sim_cw_access_pc(s, 1, TSR_CW_ISSUE_PUSH, &v, pc) == TSR_CW_OK, "push at pc 0x%08x", pc);
    idle(s);
    tsr_sim_acc a;
    memset(&a, 0, sizeof(a));
    a.kind = TSR_ACC_RV_SPM_RD;
    a.addr = 100u * 64u;
    a.size = 4;
    a.len = 4;
    a.pc = pc;
    tsr_sim_acc_emit(s, &a);
    CHECK(count_rep(s, K_RW, TSR_RACE_SPACE_SPM, A_RV, TSR_UNIT_DMA) == want && n_rep(s) == want,
          "pc 0x%08x: %zu reports, expected %zu", pc, n_rep(s), want);
    tsr_sim_destroy(s);
}
static void test_pc_sentinel(void) {
    pc_case(0u, 1);
    pc_case(TSR_SIM_PC_HOOK, 0);
}

// Partial overlaps: the report range is the overlap, folded hits count once per access.
static void test_ranges(void) {
    tsr_sim * s = mk(1);
    (void)push(s, tsr_enc_ld(0, 100, 0, 8), NONE, NONE);                    // writes 100..107
    (void)push(s, tsr_enc_qact(TSR_QA_A32, 200, 104, 64, 1, 0), NONE, NONE); // reads 104..107
    idle(s);
    const tsr_race_report * r;
    CHECK(tsr_sim_race_reports(s, &r) == 1, "one report for the overlap");
    if (tsr_sim_race_reports(s, &r) == 1) {
        CHECK(r[0].lo == 104u * 64u && r[0].hi == 108u * 64u, "overlap [0x%llx, 0x%llx)", (unsigned long long)r[0].lo,
              (unsigned long long)r[0].hi);
        CHECK(r[0].count == 1, "count %u", r[0].count);
        CHECK(r[0].first.ticket != NONE && r[0].second.ticket != NONE && r[0].first.pc == 0 && r[0].second.pc == 0,
              "both sides name their tickets (test-hook pushes: pc 0)");
        CHECK(r[0].first.time <= r[0].second.time, "first is the earlier access");
    }
    tsr_sim_destroy(s);
}

// A random tile program; returns a hash of its reports (and their order).
static uint64_t random_program(uint32_t n_ops, uint32_t race, uint64_t seed, int print) {
    tsr_sim * s = mk(race);
    rng_t r = { seed };
    uint32_t last[16];
    for (int i = 0; i < 16; ++i) last[i] = NONE;
    const uint32_t qal = tsr_qa_lines(TSR_QA_A32, 256);
    for (uint32_t k = 0; k < n_ops; ++k) {
        const uint32_t base = 64u * rng_below(&r, 32), base2 = 64u * rng_below(&r, 32);
        tsr_insn i;
        switch (rng_below(&r, 6)) {
        case 0:  i = tsr_enc_ld(rng_below(&r, 2), base, 64u * rng_below(&r, 64), 16); break;
        case 1:  i = tsr_enc_st(rng_below(&r, 2), base, 4096u + 64u * rng_below(&r, 64), 16); break;
        case 2:  i = tsr_enc_qact(TSR_QA_A32, base, base2 + 16u, 256, 1, 0); (void)qal; break;
        case 3:  i = enc_vop1(base, base2, 256); break;
        case 4:  i = enc_red_s(rng_below(&r, 16), base, 256); break;
        default: i = tsr_enc_mvm(TSR_FMT_Q8_0, 0, base, base2, 8192u + 64u * rng_below(&r, 16), 16, 256, 1, 0); break;
        }
        const uint32_t d0 = rng_below(&r, 3) ? last[rng_below(&r, 16)] : NONE;
        const uint32_t d1 = rng_below(&r, 4) == 0 ? last[rng_below(&r, 16)] : NONE;
        last[k % 16] = push(s, i, d0, d1);
    }
    idle(s);
    uint64_t h = 1469598103934665603ull;
    const tsr_race_report * rep;
    const size_t n = tsr_sim_race_reports(s, &rep);
    for (size_t i = 0; i < n; ++i) h = (h ^ fnv64(&rep[i], sizeof(rep[i]))) * 1099511628211ull;
    tsr_race_stats st;
    if (print && tsr_sim_race_stats(s, &st) == 0)
        printf("random program: %u ops, %zu reports (%llu hits), %llu op cells, %llu joins, %llu DRAM segments max, "
               "MTIME %llu\n", n_ops, n, (unsigned long long)st.reports, (unsigned long long)st.op_lines,
               (unsigned long long)st.joins, (unsigned long long)st.dram_segments_max, (unsigned long long)tsr_sim_now(s));
    tsr_sim_destroy(s);
    return h;
}

static void test_random(void) {
    const uint64_t a = random_program(1000, 1, 0x1234u, 1), b = random_program(1000, 1, 0x1234u, 0);
    CHECK(a == b, "reports are deterministic");
    // With every op depending on the previous one (a serial chain), nothing races.
    tsr_sim * s = mk(1);
    uint32_t t = NONE;
    rng_t r = { 7 };
    for (uint32_t k = 0; k < 300; ++k) {
        const uint32_t base = 64u * rng_below(&r, 8), base2 = 64u * rng_below(&r, 8);
        tsr_insn i;
        switch (k % 4) {
        case 0:  i = tsr_enc_ld(0, base, 64u * rng_below(&r, 8), 16); break;
        case 1:  i = enc_vop1(base, base2, 256); break;
        case 2:  i = tsr_enc_st(1, base, 64u * rng_below(&r, 8), 16); break;
        default: i = enc_red_s(rng_below(&r, 16), base, 256); break;
        }
        t = push(s, i, t, NONE);
    }
    idle(s);
    CHECK(n_rep(s) == 0, "serial chain: %zu reports", n_rep(s));
    tsr_sim_destroy(s);
}

// Fatal mode aborts with the report printed (in a child process).
static void test_fatal(void) {
    fflush(stdout);
    const pid_t pid = fork();
    if (pid == 0) {
        const int devnull = open("/dev/null", O_WRONLY);
        if (devnull >= 0) dup2(devnull, 2);
        tsr_sim * s = mk(2);
        (void)push(s, tsr_enc_ld(0, 100, 0, 4), NONE, NONE);
        (void)push(s, tsr_enc_qact(TSR_QA_A32, 200, 100, 64, 1, 0), NONE, NONE);
        idle(s);
        _exit(0);
    }
    int st = 0;
    waitpid(pid, &st, 0);
    CHECK(WIFSIGNALED(st) && WTERMSIG(st) == SIGABRT, "fatal mode aborts on a race (status 0x%x)", st);
    // Fatal without a race completes.
    tsr_sim * s = mk(2);
    const uint32_t t = push(s, tsr_enc_ld(0, 100, 0, 4), NONE, NONE);
    (void)push(s, tsr_enc_qact(TSR_QA_A32, 200, 100, 64, 1, 0), t, NONE);
    idle(s);
    CHECK(n_rep(s) == 0, "fatal, ordered: no report");
    tsr_sim_destroy(s);
    // Off (opts.race = 0 takes TSR_SIM_RACE, so only when the environment leaves it off): nothing
    // tracked.
    const char * env = getenv("TSR_SIM_RACE");
    if (env != NULL && env[0] != '\0' && strcmp(env, "off") != 0) {
        printf("sim_race: TSR_SIM_RACE=%s in the environment: the race-off case is skipped\n", env);
        return;
    }
    s = mk(0);
    (void)push(s, tsr_enc_ld(0, 100, 0, 4), NONE, NONE);
    (void)push(s, tsr_enc_qact(TSR_QA_A32, 200, 100, 64, 1, 0), NONE, NONE);
    idle(s);
    const tsr_race_report * r = (const tsr_race_report *)1;
    CHECK(tsr_sim_race_reports(s, &r) == 0 && r == NULL && !tsr_sim_race_active(s), "race off: no detector");
    tsr_sim_destroy(s);
}

// ---------------------------------------------------------------------------
// Firmware on the built-in hart
// ---------------------------------------------------------------------------
static uint32_t crc32_zlib(const uint8_t * p, size_t n) {
    uint32_t c = 0xFFFFFFFFu;
    for (size_t i = 0; i < n; ++i) {
        c ^= p[i];
        for (int k = 0; k < 8; ++k) c = (c >> 1) ^ (TSR_CRC32_POLY & (0u - (c & 1u)));
    }
    return ~c;
}

static int load_elf(tsr_sim * sim, const char * path) {
    tsr_rv_elf elf;
    if (tsr_rv_elf_read(path, &elf) != 0) return -1;
    for (uint32_t i = 0; i < elf.nseg; ++i) {
        const tsr_rv_elf_seg * g = &elf.seg[i];
        if (g->filesz == 0) continue;
        const uint32_t bytes = (g->filesz + 3u) & ~3u;
        uint8_t * buf = (uint8_t *)calloc(bytes, 1);
        memcpy(buf, g->data, g->filesz);
        tsr_sim_csr_write(sim, TSR_CSR_LOAD_ADDR, g->paddr);
        for (uint32_t k = 0; k < bytes; k += 4) tsr_sim_csr_write(sim, TSR_CSR_LOAD_DATA, get_u32_le(buf + k));
        CHECK(tsr_sim_csr_read(sim, TSR_CSR_LOAD_CRC) == crc32_zlib(buf, bytes), "load CRC of 0x%08x", g->paddr);
        free(buf);
    }
    tsr_rv_elf_free(&elf);
    return 0;
}

typedef struct host {
    tsr_sim * sim;
    uint32_t  lcq_rd;
    uint32_t  rec[64][8];
    uint32_t  n_rec;
} host;

static void consume(host * h) {
    const uint32_t wr = tsr_sim_csr_read(h->sim, TSR_CSR_LCQ_WR);
    for (uint32_t k = h->lcq_rd; k != wr; ++k) {
        for (uint32_t j = 0; j < 8u; ++j) {
            const uint32_t v = tsr_sim_csr_read(h->sim, TSR_CSR_LCQ_REC(k % TSR_LCQ_DEPTH, j));
            if (h->n_rec < 64u) h->rec[h->n_rec][j] = v;
        }
        if (h->n_rec < 64u) ++h->n_rec;
    }
    h->lcq_rd = wr;
    tsr_sim_csr_write(h->sim, TSR_CSR_LCQ_RD, wr);
}

static int run_until(host * h, uint32_t n_want) {
    for (int it = 0; it < 4000 && h->n_rec < n_want; ++it) {
        (void)tsr_sim_advance(h->sim, 200000);
        if (TSR_GET(tsr_sim_csr_read(h->sim, TSR_CSR_RT_STATE), TSR_RT_STATE_PHASE) == TSR_RT_STATE_PHASE_TRAPPED) {
            CHECK(0, "the firmware trapped: mcause %u", tsr_sim_csr_read(h->sim, TSR_CSR_RT_INFO));
            return 0;
        }
        consume(h);
    }
    CHECK(h->n_rec >= n_want, "%u records, expected %u", h->n_rec, n_want);
    return h->n_rec >= n_want;
}

static void push_desc(host * h, const uint32_t d[8]) {
    for (uint32_t k = 0; k < 8u; ++k) tsr_sim_csr_write(h->sim, TSR_CSR_LQ_W(k), d[k]);
    tsr_sim_csr_write(h->sim, TSR_CSR_LQ_PUSH, 1u);
}

// Boots an image and consumes its HELLO. sched NULL: TSR_SIM_SCHED or fixed.
static tsr_sim * boot_sched(const char * elf, uint32_t race, double ratio, const char * sched, host * h) {
    tsr_config c = cfg_u200(1);
    c.num_ch = 2;
    c.ch_bytes = 1ull << 30;
    c.caps = TSR_CAPS_VU;
    tsr_sim_opts o;
    memset(&o, 0, sizeof(o));
    o.race = race;
    o.rv_ratio = ratio;
    o.sched = sched;
    tsr_sim * s = tsr_sim_create_opts(&c, &o);
    CHECK(s != NULL && tsr_sim_rv_agent(s) != NULL, "create with the RISC-V model");
    if (s == NULL) exit(1);
    memset(h, 0, sizeof(*h));
    h->sim = s;
    CHECK(load_elf(s, elf) == 0, "load %s", elf);
    tsr_sim_csr_write(s, TSR_CSR_RV_BOOT, 0u);
    tsr_sim_csr_write(s, TSR_CSR_CTRL, TSR_CTRL_RV_RUN);
    if (run_until(h, 1)) CHECK(TSR_GET(h->rec[0][0], TSR_LCQ_R0_TYPE) == TSR_LCQ_R0_TYPE_HELLO, "HELLO");
    return s;
}
static tsr_sim * boot(const char * elf, uint32_t race, double ratio, host * h) { return boot_sched(elf, race, ratio, NULL, h); }

static uint32_t handle(uint32_t ch, uint32_t beat) { return TSR_PUT(TSR_TICKET_UNIT, ch) | beat; }

// ---- (1) / (2): the sim_rv_issue firmware and its mutant ----
typedef struct { uint64_t y_hash; uint32_t s0, n_rec; uint32_t rec[16][7]; } issue_out;

static void issue_case(const char * elf, uint32_t race, const char * sched, issue_out * out) {
    host h;
    tsr_sim * s = boot_sched(elf, race, 0.0, sched, &h);
    const uint32_t N = 40, K = 128, fmt = TSR_FMT_Q8_0, kind = TSR_QA_A32;
    const uint32_t qal = tsr_qa_lines(kind, K), yl = (N + 15u) / 16u;
    rng_t r = { 0xACE5u };
    // Inputs through the visible API: x on ch0, the repacked weights on every channel, y region.
    float x[128];
    gen_f32(&r, x, K);
    uint8_t xb[512];
    put_f32_le(xb, x, K);
    const uint64_t x_beat = TSR_REGION_IO_OFF / 64u, y_beat = TSR_REGION_IO_OFF / 64u + 5000u;
    const uint64_t w_off = TSR_REGION_WEIGHT_OFF / 64u;
    CHECK(tsr_sim_dram_write(s, 0, x_beat * 64u, xb, 4u * K) == 0, "x");
    const size_t row_bytes = (size_t)(K / tsr_fmt_block_size(fmt)) * tsr_fmt_ggml_block_bytes(fmt);
    uint8_t * blocks = (uint8_t *)malloc(row_bytes * N);
    for (uint32_t n = 0; n < N; ++n) gen_row_blocks(&r, fmt, K, blocks + row_bytes * n);
    put_weights(s, fmt, N, K, w_off, blocks);
    for (uint32_t n = 0; n < N; ++n) {
        uint32_t ch;
        uint64_t off;
        tsr_row_location(fmt, N, K, 2, n, &ch, &off);
        tsr_sim_dram_note_write(s, ch, (w_off + off) * 64u, (uint64_t)tsr_row_beats(fmt, K) * 64u);
    }
    free(blocks);
    uint8_t ee[64 * 4];
    memset(ee, 0xEE, sizeof(ee));
    CHECK(tsr_sim_dram_write(s, 1, y_beat * 64u, ee, (uint64_t)yl * 64u) == 0, "y");
    uint32_t d[8] = { TSR_DESC_TYPE_LAUNCH, handle(0, (uint32_t)x_beat), (uint32_t)w_off, handle(1, (uint32_t)y_beat), N, K,
                      fmt | qal << 16, TSR_VOP_FUNC_SILU };
    push_desc(&h, d);
    push_desc(&h, d);                                     // twice: the second reruns on the same lines
    const uint32_t ping[8] = { TSR_DESC_TYPE_PING, 0xC0FFEEu, 0, 0, 0, 0, 0, 0 };
    push_desc(&h, ping);
    if (run_until(&h, 6)) {                              // HELLO, EV + DONE twice, PONG
        CHECK(TSR_GET(h.rec[2][0], TSR_LCQ_R0_TYPE) == TSR_LCQ_R0_TYPE_DONE &&
              TSR_GET(h.rec[2][0], TSR_LCQ_R0_STATUS) == TSR_LCQ_R0_STATUS_OK && h.rec[2][2] == 6u, "first launch DONE OK");
        CHECK(TSR_GET(h.rec[5][0], TSR_LCQ_R0_TYPE) == TSR_LCQ_R0_TYPE_PONG, "PONG last");
    }
    uint8_t y[256];
    CHECK(tsr_sim_dram_read(s, 1, y_beat * 64u, y, (uint64_t)yl * 64u) == 0, "read y after the answers");
    out->y_hash = fnv64(y, (size_t)yl * 64u);
    out->s0 = f2u(tsr_sim_sreg(s)[0]);
    out->n_rec = h.n_rec;
    for (uint32_t i = 0; i < h.n_rec && i < 16u; ++i) memcpy(out->rec[i], h.rec[i], 7 * sizeof(uint32_t));
    if (race == 1) {
        printf("mutant (no ST -> VOP dependency), schedule %s: %zu report(s)\n", sched ? sched : "default", n_rep(s));
        dump(s, "mutant");
        CHECK(count_rep(s, K_RW, TSR_RACE_SPACE_SPM, TSR_UNIT_VU, TSR_UNIT_DMA) >= 1, "mutant: VOP write vs ST read reported");
        const tsr_race_report * rep;
        const size_t n = tsr_sim_race_reports(s, &rep);
        for (size_t i = 0; i < n; ++i) {
            const tsr_race_side * vu = rep[i].first.agent == TSR_UNIT_VU ? &rep[i].first : &rep[i].second;
            const tsr_race_side * dm = rep[i].first.agent == TSR_UNIT_DMA ? &rep[i].first : &rep[i].second;
            if (vu->agent != TSR_UNIT_VU || dm->agent != TSR_UNIT_DMA) continue;
            CHECK(vu->opcode == TSR_OP_VOP && vu->write && dm->opcode == TSR_OP_ST && !dm->write && vu->pc != 0 &&
                  dm->pc != 0 && vu->launch == dm->launch, "mutant: the sides are the VOP (write) and the ST (read)");
        }
    } else {
        CHECK(n_rep(s) == 0, "original firmware: %zu reports", n_rep(s));
        if (n_rep(s)) dump(s, "original");
    }
    tsr_sim_destroy(s);
}

static void test_issue(const char * orig, const char * mutant) {
    static issue_out a, b;
    memset(&a, 0, sizeof(a));
    memset(&b, 0, sizeof(b));
    issue_case(orig, 2, "fixed", &a);                     // (1) fatal: any report aborts
    issue_case(mutant, 1, "fixed", &b);                   // (2)
    CHECK(a.n_rec == b.n_rec && memcmp(a.rec, b.rec, sizeof(a.rec)) == 0 && a.y_hash == b.y_hash && a.s0 == b.s0,
          "the mutant's records, outputs and s0 equal the original's (only the race detector catches it)");
    // Detection does not depend on the schedule (SW §4.6, ctrl-riscv §9.2): the same verdicts under
    // random op durations, crossings and RV CPI.
    static const char * const scheds[] = { "random:1", "random:2,0.1,10", "random:3" };
    for (int i = 0; i < 3; ++i) {
        issue_case(orig, 2, scheds[i], &a);
        issue_case(mutant, 1, scheds[i], &b);
    }
}

// ---- (4) - (6): fw_race.c kernels ----
enum { L_A = 100, L_B = 110, L_C = 120, L_D = 130, L_E = 140, L_F = 150, L_G = 160, L_H = 170 };
#define X_BEAT 2048u
#define Y_BEAT 4096u

static uint32_t run_kernel(host * h, uint32_t kernel, const uint32_t a[7]) {
    uint32_t d[8] = { TSR_DESC_TYPE_LAUNCH | TSR_PUT(TSR_DESC_KERNEL, kernel), 0, 0, 0, 0, 0, 0, 0 };
    for (int i = 0; i < 7; ++i) d[1 + i] = a[i];
    const uint32_t n0 = h->n_rec;
    push_desc(h, d);
    if (!run_until(h, n0 + 1)) return 0xFFFFFFFFu;
    CHECK(TSR_GET(h->rec[n0][0], TSR_LCQ_R0_TYPE) == TSR_LCQ_R0_TYPE_DONE &&
          TSR_GET(h->rec[n0][0], TSR_LCQ_R0_STATUS) == TSR_LCQ_R0_STATUS_OK, "kernel %u: DONE OK (0x%08x)", kernel,
          h->rec[n0][0]);
    return h->rec[n0][2];
}

typedef struct { const char * name; uint32_t kernel; uint32_t args[7]; uint32_t kinds, space, a, b; size_t want, total; } kcase;

static void test_kernels(const char * elf) {
    uint8_t data[64 * 8];
    for (uint32_t i = 0; i < sizeof(data); ++i) data[i] = (uint8_t)(i * 7u + 1u);
    const kcase cs[] = {
        { "store then push (E4)", 0, { L_A, handle(1, Y_BEAT) }, K_RACE | K_STALE, 0, A_RV, TSR_UNIT_DMA, 0, 0 },
        { "LD then RISC-V read, no wait", 1, { L_B, handle(0, X_BEAT) }, K_RW, TSR_RACE_SPACE_SPM, A_RV, TSR_UNIT_DMA, 1, 2 },
        { "LD, DONE_SEQ wait, read", 2, { L_B, handle(0, X_BEAT) }, K_RACE | K_STALE, 0, A_RV, TSR_UNIT_DMA, 0, 0 },
        { "LD, WAIT interrupt, read", 3, { L_B, handle(0, X_BEAT) }, K_RACE | K_STALE, 0, A_RV, TSR_UNIT_DMA, 0, 0 },
        { "ST of a line not written in the launch", 4, { L_C, handle(1, Y_BEAT) }, K_STALE, TSR_RACE_SPACE_SPM,
          TSR_UNIT_DMA, A_NONE, 1, 1 },
        { "LD then ST of the line", 5, { L_C, handle(0, X_BEAT), handle(1, Y_BEAT) }, K_RACE | K_STALE, 0, TSR_UNIT_DMA,
          A_NONE, 0, 0 },
        { "RISC-V read of a line not written in the launch", 7, { L_E }, K_STALE, TSR_RACE_SPACE_SPM, A_RV, A_NONE, 1, 1 },
        { "RISC-V store after the push of an ST reading it", 8, { L_F, handle(1, Y_BEAT) }, K_RW, TSR_RACE_SPACE_SPM, A_RV,
          TSR_UNIT_DMA, 1, 1 },
        { "line-buffer fills: written line fresh, unwritten line stale", 9, { L_G, L_H }, K_STALE, TSR_RACE_SPACE_SPM, A_RV,
          A_NONE, 1, 1 },
        { "RED to s0, core-window SREG0 read without a wait", 10, { L_G }, K_RW, TSR_RACE_SPACE_SREG, A_RV, TSR_UNIT_VU, 1, 1 },
        { "LD, QSTATUS.QIDLE wait, read", 11, { L_B, handle(0, X_BEAT) }, K_RACE | K_STALE, 0, A_RV, TSR_UNIT_DMA, 0, 0 },
    };
    for (size_t c = 0; c < sizeof(cs) / sizeof(cs[0]); ++c) {
        host h;
        tsr_sim * s = boot(elf, 1, 0.0, &h);
        CHECK(tsr_sim_dram_write(s, 0, X_BEAT * 64u, data, sizeof(data)) == 0, "x");
        (void)run_kernel(&h, cs[c].kernel, cs[c].args);
        const size_t got = count_rep(s, cs[c].kinds, cs[c].space, cs[c].a, cs[c].b);
        CHECK(got == cs[c].want && n_rep(s) == cs[c].total, "%s: %zu matching of %zu reports, expected %zu of %zu",
              cs[c].name, got, n_rep(s), cs[c].want, cs[c].total);
        if (got != cs[c].want || n_rep(s) != cs[c].total || cs[c].total) dump(s, cs[c].name);
        if (cs[c].kernel == 1) CHECK(count_rep(s, K_STALE, TSR_RACE_SPACE_SPM, A_RV, A_NONE) == 1, "no wait: also stale");
        const tsr_race_report * r;
        if (cs[c].kernel == 9 && tsr_sim_race_reports(s, &r) == 1)
            CHECK(r[0].lo == L_H * 64u && r[0].hi == L_H * 64u + 64u, "the stale fill is line H, whole");
        if (cs[c].kernel == 10 && tsr_sim_race_reports(s, &r) == 1)
            CHECK(r[0].lo == 0 && r[0].hi == 1, "the s-register race is on s0");
        tsr_sim_destroy(s);
    }
    // SPM_STALE across launches: launch 1 writes C (LD), launch 2 stores it without writing it.
    {
        host h;
        tsr_sim * s = boot(elf, 1, 0.0, &h);
        const uint32_t a5[7] = { L_C, handle(0, X_BEAT), handle(1, Y_BEAT) }, a4[7] = { L_C, handle(1, Y_BEAT) };
        (void)run_kernel(&h, 5, a5);
        CHECK(n_rep(s) == 0, "launch 1: fresh");
        (void)run_kernel(&h, 4, a4);
        CHECK(count_rep(s, K_STALE, TSR_RACE_SPACE_SPM, TSR_UNIT_DMA, A_NONE) == 1 && n_rep(s) == 1,
              "launch 2: the line written by launch 1 is stale (%zu reports)", n_rep(s));
        const tsr_race_report * r;
        if (tsr_sim_race_reports(s, &r) == 1)
            CHECK(r[0].launch == 1u && r[0].lo == L_C * 64u && r[0].hi == L_C * 64u + 64u && r[0].since > 0,
                  "stale report: launch %u, [0x%llx, 0x%llx)", r[0].launch, (unsigned long long)r[0].lo,
                  (unsigned long long)r[0].hi);
        tsr_sim_destroy(s);
    }
    // (6) H1 / H2 with kernel 6 (LD x -> D, ST D -> y).
    const uint32_t a6[7] = { handle(0, X_BEAT), handle(1, Y_BEAT), L_D, 4 };
    uint8_t y[256];
    {
        host h;
        tsr_sim * s = boot(elf, 1, 0.0, &h);
        CHECK(tsr_sim_dram_write(s, 0, X_BEAT * 64u, data, 256) == 0, "x before LQ_PUSH");
        (void)run_kernel(&h, 6, a6);
        CHECK(tsr_sim_dram_read(s, 1, Y_BEAT * 64u, y, 256) == 0 && memcmp(y, data, 256) == 0, "y = x after the answer");
        CHECK(tsr_sim_dram_write(s, 0, X_BEAT * 64u, data, 256) == 0, "x rewritten after the answer");
        (void)run_kernel(&h, 6, a6);
        CHECK(n_rep(s) == 0, "H1 / H2 respected: %zu reports", n_rep(s));
        if (n_rep(s)) dump(s, "H1/H2");
        tsr_sim_destroy(s);
    }
    {
        // H1 violated: x written after LQ_PUSH while the launch reads it.
        host h;
        tsr_sim * s = boot(elf, 1, 0.0, &h);
        uint32_t d[8] = { TSR_DESC_TYPE_LAUNCH | TSR_PUT(TSR_DESC_KERNEL, 6u), a6[0], a6[1], a6[2], a6[3], 0, 0, 0 };
        push_desc(&h, d);
        CHECK(tsr_sim_dram_write(s, 0, X_BEAT * 64u, data, 256) == 0, "x after LQ_PUSH");
        (void)run_until(&h, 2);
        CHECK(count_rep(s, K_RW, TSR_RACE_SPACE_DRAM, A_HOST, TSR_UNIT_DMA) == 1 && n_rep(s) == 1,
              "H1: host write after LQ_PUSH vs the LD reported (%zu reports)", n_rep(s));
        dump(s, "H1");
        tsr_sim_destroy(s);
    }
    {
        // H2 violated: y read before the answer was read.
        host h;
        tsr_sim * s = boot(elf, 1, 0.0, &h);
        uint32_t d[8] = { TSR_DESC_TYPE_LAUNCH | TSR_PUT(TSR_DESC_KERNEL, 6u), a6[0], a6[1], a6[2], a6[3], 0, 0, 0 };
        push_desc(&h, d);
        (void)tsr_sim_advance(s, 200);
        CHECK(tsr_sim_dram_read(s, 1, Y_BEAT * 64u, y, 256) == 0, "y while the launch runs");
        (void)run_until(&h, 2);
        CHECK(count_rep(s, K_RW, TSR_RACE_SPACE_DRAM, A_HOST, TSR_UNIT_DMA) == 1 && n_rep(s) == 1,
              "H2: host read before the answer vs the ST reported (%zu reports)", n_rep(s));
        dump(s, "H2");
        // After the answer, the same read is ordered (no new report).
        tsr_sim_race_clear(s);
        CHECK(tsr_sim_dram_read(s, 1, Y_BEAT * 64u, y, 256) == 0, "y after the answer");
        CHECK(n_rep(s) == 0, "H2: read after the answer: %zu reports", n_rep(s));
        tsr_sim_destroy(s);
    }
    // H2 itself (E8's join of the launch's completed ops at the answer read). Kernel 12 answers
    // without the R6 join, so the RISC-V clock at its post does not cover its ST: (a) the host
    // waits for the post and then for the ST's completion, both silently (core-window accessor:
    // no access event, no Q_DONE / STATUS observation), reads the answer, then y: only H2 orders
    // the ST before that read, 0 reports; (b) reading the answer while the ST still runs leaves
    // the ST unordered with the later y read (the runtime broke R6): 1 report. Fixed schedule:
    // the timing of (b) is part of the assertion.
    for (int late = 0; late < 2; ++late) {
        enum { N12 = 1024 };                             // lines: LD and ST 1088 cycles each
        static uint8_t x64[N12 * 64], y64[N12 * 64];
        for (uint32_t i = 0; i < sizeof(x64); ++i) x64[i] = (uint8_t)(i * 13u + 5u);
        const uint32_t a12[7] = { handle(0, X_BEAT), handle(1, Y_BEAT), L_D, N12 };
        host h;
        tsr_sim * s = boot_sched(elf, 1, 0.0, "fixed", &h);
        CHECK(tsr_sim_dram_write(s, 0, X_BEAT * 64u, x64, sizeof(x64)) == 0, "x before LQ_PUSH");
        const uint32_t d0 = tsr_sim_corewin_read(s, TSR_CSR_Q_DONE(TSR_UNIT_DMA));
        uint32_t d[8] = { TSR_DESC_TYPE_LAUNCH | TSR_PUT(TSR_DESC_KERNEL, 12u), a12[0], a12[1], a12[2], a12[3], 0, 0, 0 };
        push_desc(&h, d);
        for (int it = 0; it < 100000 && tsr_sim_corewin_read(s, TSR_CSR_LCQ_WR) == h.lcq_rd; ++it) (void)tsr_sim_advance(s, 8);
        const uint32_t at_post = (tsr_sim_corewin_read(s, TSR_CSR_Q_DONE(TSR_UNIT_DMA)) - d0) & TSR_TICKET_SEQ;
        CHECK(tsr_sim_corewin_read(s, TSR_CSR_LCQ_WR) != h.lcq_rd && at_post < 2u,
              "kernel 12 answered with its ST still running (%u of 2 DMA ops done)", at_post);
        if (late) consume(&h);                           // (b) the answer while the ST runs
        for (int it = 0; it < 100000 && ((tsr_sim_corewin_read(s, TSR_CSR_Q_DONE(TSR_UNIT_DMA)) - d0) & TSR_TICKET_SEQ) < 2u; ++it)
            (void)tsr_sim_advance(s, 8);
        CHECK(((tsr_sim_corewin_read(s, TSR_CSR_Q_DONE(TSR_UNIT_DMA)) - d0) & TSR_TICKET_SEQ) == 2u, "the ST completed");
        if (!late) consume(&h);                          // (a) the answer after the ST completed
        CHECK(h.n_rec == 2 && TSR_GET(h.rec[1][0], TSR_LCQ_R0_TYPE) == TSR_LCQ_R0_TYPE_DONE, "kernel 12: DONE");
        CHECK(tsr_sim_dram_read(s, 1, Y_BEAT * 64u, y64, sizeof(y64)) == 0 && memcmp(y64, x64, sizeof(x64)) == 0,
              "kernel 12: y = x");
        if (!late) {
            CHECK(n_rep(s) == 0, "H2: answer read after the ST completed orders it (%zu reports)", n_rep(s));
            if (n_rep(s)) dump(s, "H2 join");
        } else {
            CHECK(count_rep(s, K_RW, TSR_RACE_SPACE_DRAM, A_HOST, TSR_UNIT_DMA) == 1 && n_rep(s) == 1,
                  "H2: answer read while the ST ran: the later y read is unordered (%zu reports)", n_rep(s));
            dump(s, "H2 no join");
        }
        tsr_sim_destroy(s);
    }
    // The RISC-V scenarios at the 1:1 clock ratio give the same verdicts.
    {
        host h;
        tsr_sim * s = boot(elf, 1, 1.0, &h);
        CHECK(tsr_sim_dram_write(s, 0, X_BEAT * 64u, data, 256) == 0, "x");
        const uint32_t a2[7] = { L_B, handle(0, X_BEAT) }, a3[7] = { L_B, handle(0, X_BEAT) };
        (void)run_kernel(&h, 2, a2);
        (void)run_kernel(&h, 3, a3);
        CHECK(n_rep(s) == 0, "ratio 1.0: DONE_SEQ and WAIT waits ordered (%zu reports)", n_rep(s));
        tsr_sim_destroy(s);
    }
}

// ---------------------------------------------------------------------------
static int file_exists(const char * p) {
    FILE * f = fopen(p, "rb");
    if (f == NULL) return 0;
    fclose(f);
    return 1;
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

int main(int argc, char ** argv) {
    const char * mode = argc > 1 ? argv[1] : "tile";
    if (strcmp(mode, "tile") == 0) {
        test_tile_pairs();
        test_sreg();
        test_host_harness();
        test_kvst_capture_accesses();
        test_pc_sentinel();
        test_ranges();
        test_random();
        test_fatal();
        return test_finish("sim_race");
    }
    if (strcmp(mode, "rv") == 0) {
        if (argc < 5) {
            fprintf(stderr, "usage: sim_race rv <fw_issue.elf> <fw_issue_nodep.elf> <fw_race.elf>\n");
            return 2;
        }
        for (int i = 2; i < 5; ++i) {
            if (!file_exists(argv[i])) {
                printf("sim_race_rv: %s not built (no RISC-V compiler at configure time); skipped\n", argv[i]);
                return 77;
            }
        }
        tsr_sim * probe = mk(0);
        const int have_rv = tsr_sim_rv_agent(probe) != NULL;
        tsr_sim_destroy(probe);
        if (!have_rv) {
            printf("sim_race_rv: the simulator library has no RISC-V model; skipped\n");
            return 77;
        }
        test_issue(argv[2], argv[3]);
        test_kernels(argv[4]);
        return test_finish("sim_race_rv");
    }
    if (strcmp(mode, "bench") == 0) {
        const uint32_t n = argc > 2 ? (uint32_t)strtoul(argv[2], NULL, 0) : 1000u;
        const uint32_t race = argc > 3 ? (uint32_t)strtoul(argv[3], NULL, 0) : 1u;
        const int reps = argc > 4 ? atoi(argv[4]) : 5;
        double best = 1e30;
        for (int i = 0; i < reps; ++i) {
            const double t0 = now_s();
            (void)random_program(n, race, 0x1234u, i == 0 && race != 0);
            const double t = now_s() - t0;
            if (t < best) best = t;
        }
        printf("bench: %u ops, race %u: best of %d %.4f s\n", n, race, reps, best);
        return 0;
    }
    fprintf(stderr, "usage: sim_race tile | rv <elfs...> | bench [n] [race] [reps]\n");
    return 2;
}
