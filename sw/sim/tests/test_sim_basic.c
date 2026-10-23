// test_sim_basic.c — ISA simulator: CSRs, control flow, registers, LD/ST, every M1-reachable
// error code (with ERR_PC / ERR_INFO and "no state change"), reserved-field checks, abort.
#include "test_sim_util.h"

#include <pthread.h>
#include <time.h>

#define DONE    TSR_STATUS_DONE
#define DONEERR (TSR_STATUS_DONE | TSR_STATUS_ERR)

static const uint64_t IO_BEAT = TSR_REGION_IO_OFF / TSR_BEAT_BYTES;   // 64 MiB / 64

// ---------------------------------------------------------------------------
// State snapshot for "a failing instruction changes no state"
// ---------------------------------------------------------------------------
typedef struct {
    uint8_t * spm;          // full SPM copy
    uint8_t * dram[4];      // first MiB of each channel's IO region
    int32_t   R[16];
} snap_t;

#define SNAP_DRAM_BYTES (1u << 20)

static void snap_take(tsr_sim * s, snap_t * z) {
    const uint32_t spm_bytes = tsr_sim_config(s)->spm_bytes;
    if (z->spm == NULL) {
        z->spm = (uint8_t *)malloc(4u << 20);
        for (int c = 0; c < 4; ++c) z->dram[c] = (uint8_t *)malloc(SNAP_DRAM_BYTES);
    }
    memcpy(z->spm, tsr_sim_spm(s), spm_bytes);
    for (uint32_t c = 0; c < 4 && c < tsr_sim_config(s)->num_ch; ++c) {
        memcpy(z->dram[c], tsr_sim_dram(s, c) + TSR_REGION_IO_OFF, SNAP_DRAM_BYTES);
    }
    for (int i = 0; i < 16; ++i) z->R[i] = tsr_sim_ireg(s, (uint32_t)i);
}
static int snap_same_mem(tsr_sim * s, const snap_t * z) {
    if (memcmp(z->spm, tsr_sim_spm(s), tsr_sim_config(s)->spm_bytes) != 0) return 0;
    for (uint32_t c = 0; c < 4 && c < tsr_sim_config(s)->num_ch; ++c) {
        if (memcmp(z->dram[c], tsr_sim_dram(s, c) + TSR_REGION_IO_OFF, SNAP_DRAM_BYTES) != 0) return 0;
    }
    return 1;
}
static snap_t g_snap;

// Runs prog (which must fail at err_pc with code/info). Checks that SPM and the first MiB of
// every channel's IO region are unchanged and that exactly the err_pc instructions before the
// failing one retired (the failing instruction itself has no effect).
static void expect_error(tsr_sim * s, const prog_t * p, uint32_t code, uint32_t err_pc, uint32_t info,
                         const char * what) {
    prog_load(s, p, 0);
    snap_take(s, &g_snap);
    const uint64_t ret0 = tsr_sim_perf(s, TSR_PERF_RETIRED);
    const uint32_t st = tsr_sim_exec(s, 0, p->n, NULL);
    CHECK(st == DONEERR, "%s: status %u", what, st);
    CHECK(sim_err_code(s) == code, "%s: err_code %u, expected %u", what, sim_err_code(s), code);
    CHECK(sim_err_pc(s) == err_pc, "%s: err_pc %u, expected %u", what, sim_err_pc(s), err_pc);
    CHECK(sim_err_info(s) == info, "%s: err_info 0x%x, expected 0x%x", what, sim_err_info(s), info);
    CHECK(snap_same_mem(s, &g_snap), "%s: SPM or DRAM changed", what);
    CHECK(tsr_sim_perf(s, TSR_PERF_RETIRED) - ret0 == err_pc, "%s: retired %llu", what,
          (unsigned long long)(tsr_sim_perf(s, TSR_PERF_RETIRED) - ret0));
}

// One-instruction program (+ optional register setup before it) that must fail.
static void expect_insn_error(tsr_sim * s, const tsr_insn * setup, uint32_t n_setup, tsr_insn bad, uint32_t code,
                              uint32_t info, const char * what) {
    prog_t p = {0};
    for (uint32_t i = 0; i < n_setup; ++i) prog_add(&p, setup[i]);
    prog_add(&p, bad);
    prog_add(&p, tsr_enc_end());
    expect_error(s, &p, code, n_setup, info, what);
    prog_free(&p);
}

static void expect_ok(tsr_sim * s, tsr_insn insn, const char * what) {
    prog_t p = {0};
    prog_add(&p, insn);
    prog_add(&p, tsr_enc_end());
    const uint32_t st = prog_run(s, &p, 0, NULL);
    CHECK(st == DONE, "%s: status %u err %u info 0x%x", what, st, sim_err_code(s), sim_err_info(s));
    prog_free(&p);
}

// ---------------------------------------------------------------------------
static void test_csr(void) {
    tsr_config c = cfg_u200(1);
    tsr_sim * s = tsr_sim_create(&c, NULL);
    CHECK(s != NULL, "create");
    CHECK(tsr_sim_csr_read(s, TSR_CSR_ID) == 0x54535231u, "ID");
    CHECK(tsr_sim_csr_read(s, TSR_CSR_VERSION) == 0x0100u, "VERSION 0x%x", tsr_sim_csr_read(s, TSR_CSR_VERSION));
    CHECK(tsr_sim_csr_read(s, TSR_CSR_CFG0) == (4u | (128u << 8) | (1u << 20)), "CFG0 0x%x", tsr_sim_csr_read(s, TSR_CSR_CFG0));
    CHECK(tsr_sim_csr_read(s, TSR_CSR_CFG1) == 65536u, "CFG1");
    CHECK(tsr_sim_csr_read(s, TSR_CSR_CFG2) == (16u | (16u << 8) | (128u << 16)), "CFG2 0x%x", tsr_sim_csr_read(s, TSR_CSR_CFG2));
    CHECK(tsr_sim_csr_read(s, TSR_CSR_CAPS) == 0, "CAPS");
    CHECK(tsr_sim_csr_read(s, TSR_CSR_STATUS) == 0, "STATUS at reset");
    CHECK(tsr_sim_csr_read(s, TSR_CSR_CTRL) == 0, "CTRL reads 0");
    CHECK(tsr_sim_csr_read(s, 0x1C) == 0 && tsr_sim_csr_read(s, 0x26) == 0, "unmapped reads 0");
    // RW registers
    tsr_sim_csr_write(s, TSR_CSR_CMD_ADDR_LO, 0x12345678u);
    tsr_sim_csr_write(s, TSR_CSR_CMD_ADDR_HI, 0x9u);
    tsr_sim_csr_write(s, TSR_CSR_CMD_COUNT, 77);
    for (uint32_t i = 0; i < 8; ++i) tsr_sim_csr_write(s, TSR_CSR_ARG0 + 4 * i, 1000 + i);
    CHECK(tsr_sim_csr_read(s, TSR_CSR_CMD_ADDR_LO) == 0x12345678u && tsr_sim_csr_read(s, TSR_CSR_CMD_ADDR_HI) == 9u, "CMD_ADDR");
    CHECK(tsr_sim_csr_read(s, TSR_CSR_CMD_COUNT) == 77, "CMD_COUNT");
    for (uint32_t i = 0; i < 8; ++i) CHECK(tsr_sim_csr_read(s, TSR_CSR_ARG0 + 4 * i) == 1000 + i, "ARG%u", i);
    // read-only registers ignore writes
    tsr_sim_csr_write(s, TSR_CSR_ID, 0);
    tsr_sim_csr_write(s, TSR_CSR_ERR_CODE, 5);
    tsr_sim_csr_write(s, TSR_CSR_CFG0, 0);
    CHECK(tsr_sim_csr_read(s, TSR_CSR_ID) == TSR_ID_MAGIC && tsr_sim_csr_read(s, TSR_CSR_ERR_CODE) == 0 &&
          tsr_sim_csr_read(s, TSR_CSR_CFG0) != 0, "RO writes ignored");

    // A run: LD 3 lines from ch1, NOP, END at cmd beat 0 -> perf exact.
    prog_t p = {0};
    prog_add(&p, tsr_enc_ld(1, 10, IO_BEAT, 3));
    prog_add(&p, tsr_enc_nop());
    prog_add(&p, tsr_enc_end());
    uint32_t st = prog_run(s, &p, 0, NULL);
    CHECK(st == DONE, "run status %u", st);
    CHECK(tsr_sim_perf(s, TSR_PERF_RETIRED) == 3, "retired %llu", (unsigned long long)tsr_sim_perf(s, TSR_PERF_RETIRED));
    CHECK(tsr_sim_perf(s, TSR_PERF_CYCLES) == (1 + 64 + 3) + 1 + 1, "cycles %llu", (unsigned long long)tsr_sim_perf(s, TSR_PERF_CYCLES));
    CHECK(tsr_sim_perf(s, tsr_perf_ch_rd_beats(0)) == 2, "ch0 fetch beats %llu", (unsigned long long)tsr_sim_perf(s, tsr_perf_ch_rd_beats(0)));
    CHECK(tsr_sim_perf(s, tsr_perf_ch_rd_beats(1)) == 3, "ch1 beats");
    CHECK(tsr_sim_perf(s, tsr_perf_ch_rd_beats(2)) == 0 && tsr_sim_perf(s, tsr_perf_ch_wait(4, 0)) == 0, "other counters");
    CHECK(tsr_sim_perf(s, TSR_PERF_MVM_CYCLES) == 0 && tsr_sim_perf(s, TSR_PERF_VU_CYCLES) == 0, "mvm/vu cycles");
    // PERF CSR LO/HI view
    CHECK(tsr_sim_csr_read(s, TSR_CSR_PERF_BASE + 8 * TSR_PERF_RETIRED) == 3 &&
          tsr_sim_csr_read(s, TSR_CSR_PERF_BASE + 8 * TSR_PERF_RETIRED + 4) == 0, "PERF CSR");
    CHECK(tsr_sim_csr_read(s, TSR_CSR_PERF_BASE + 8 * tsr_perf_ch_rd_beats(1)) == 3, "PERF ch1 CSR");
    CHECK(tsr_sim_csr_read(s, TSR_CSR_PERF_BASE + 8 * tsr_perf_count(4)) == 0, "PERF past end reads 0");
    CHECK(tsr_sim_csr_read(s, TSR_CSR_PC) == 2, "PC = index of END (%u)", tsr_sim_csr_read(s, TSR_CSR_PC));
    // STATUS W1C: BUSY not writable, DONE cleared by writing 1.
    tsr_sim_csr_write(s, TSR_CSR_STATUS, TSR_STATUS_BUSY);
    CHECK(tsr_sim_csr_read(s, TSR_CSR_STATUS) == DONE, "BUSY write ignored");
    tsr_sim_csr_write(s, TSR_CSR_STATUS, TSR_STATUS_DONE);
    CHECK(tsr_sim_csr_read(s, TSR_CSR_STATUS) == 0, "DONE W1C");
    // tsr_sim_run when not BUSY is a no-op
    CHECK(tsr_sim_run(s) == 0, "run while idle");

    // Error run: ERR + DONE, ERR W1C keeps ERR_CODE; next START clears ERR_*.
    prog_reset(&p);
    prog_add(&p, insn_raw(0xEEu, 0, 0, 0, 0, 0, 0, 0));
    st = prog_run(s, &p, 0, NULL);
    CHECK(st == DONEERR && sim_err_code(s) == TSR_ERR_ILLEGAL_OP, "illegal opcode");
    tsr_sim_csr_write(s, TSR_CSR_STATUS, TSR_STATUS_ERR);
    CHECK(tsr_sim_csr_read(s, TSR_CSR_STATUS) == DONE && sim_err_code(s) == TSR_ERR_ILLEGAL_OP, "ERR W1C");
    prog_reset(&p);
    prog_add(&p, tsr_enc_end());
    st = prog_run(s, &p, 0, NULL);
    CHECK(st == DONE && sim_err_code(s) == 0 && sim_err_pc(s) == 0 && sim_err_info(s) == 0, "START clears ERR_*");

    // START while BUSY is ignored: the first latched CMD_COUNT is used.
    prog_reset(&p);
    prog_add(&p, tsr_enc_nop());
    prog_add(&p, tsr_enc_nop());
    prog_add(&p, tsr_enc_end());
    prog_load(s, &p, 0);
    tsr_sim_csr_write(s, TSR_CSR_CMD_ADDR_LO, 0);
    tsr_sim_csr_write(s, TSR_CSR_CMD_ADDR_HI, 0);
    tsr_sim_csr_write(s, TSR_CSR_CMD_COUNT, 2);           // would fall off the end
    tsr_sim_csr_write(s, TSR_CSR_CTRL, TSR_CTRL_START);
    CHECK(tsr_sim_csr_read(s, TSR_CSR_STATUS) == TSR_STATUS_BUSY, "BUSY after START");
    tsr_sim_csr_write(s, TSR_CSR_CMD_COUNT, 3);
    tsr_sim_csr_write(s, TSR_CSR_CTRL, TSR_CTRL_START);   // ignored
    tsr_sim_csr_write(s, TSR_CSR_STATUS, TSR_STATUS_BUSY | TSR_STATUS_DONE | TSR_STATUS_ERR);   // W1C: BUSY stays
    CHECK(tsr_sim_csr_read(s, TSR_CSR_STATUS) == TSR_STATUS_BUSY, "BUSY is not W1C");
    st = tsr_sim_run(s);
    CHECK(st == DONEERR && sim_err_code(s) == TSR_ERR_ILLEGAL_OP && sim_err_pc(s) == 2, "START while busy ignored");

    // SOFT_RESET: clears status, ERR_*, perf; keeps CMD_* and ARG*.
    tsr_sim_csr_write(s, TSR_CSR_ARG0 + 4, 1001);
    tsr_sim_csr_write(s, TSR_CSR_CTRL, TSR_CTRL_SOFT_RESET);
    CHECK(tsr_sim_csr_read(s, TSR_CSR_STATUS) == 0 && sim_err_code(s) == 0 && sim_err_pc(s) == 0, "soft reset status");
    CHECK(tsr_sim_perf(s, TSR_PERF_RETIRED) == 0 && tsr_sim_perf(s, TSR_PERF_CYCLES) == 0 &&
          tsr_sim_perf(s, tsr_perf_ch_rd_beats(1)) == 0, "soft reset perf");
    CHECK(tsr_sim_csr_read(s, TSR_CSR_CMD_COUNT) == 3 && tsr_sim_csr_read(s, TSR_CSR_ARG0 + 4) == 1001, "soft reset keeps CSRs");
    prog_free(&p);
    tsr_sim_destroy(s);

    // CFG encodings of a non-default config
    tsr_config c2 = c;
    c2.num_ch = 3; c2.lanes = 64; c2.t_max = 8; c2.spm_bytes = 256 * 1024; c2.vl = 8; c2.nsreg = 12; c2.k_max = 4096;
    s = tsr_sim_create(&c2, NULL);
    CHECK(s != NULL, "create custom");
    CHECK(tsr_sim_csr_read(s, TSR_CSR_CFG0) == (3u | (64u << 8) | (8u << 20)), "CFG0 custom");
    CHECK(tsr_sim_csr_read(s, TSR_CSR_CFG1) == 4096u, "CFG1 custom");
    CHECK(tsr_sim_csr_read(s, TSR_CSR_CFG2) == (8u | (12u << 8) | (16u << 16)), "CFG2 custom");
    // SPM starts as 0xA5
    int all_a5 = 1;
    for (uint32_t i = 0; i < c2.spm_bytes; ++i) all_a5 &= tsr_sim_spm(s)[i] == 0xA5;
    CHECK(all_a5, "SPM initial 0xA5");
    tsr_sim_destroy(s);

    // Unsupported configs are rejected.
    tsr_config bad;
    bad = c; bad.beat_bits = 256;  CHECK(tsr_sim_create(&bad, NULL) == NULL, "beat_bits");
    bad = c; bad.t_max = 9;        CHECK(tsr_sim_create(&bad, NULL) == NULL, "t_max 9");
    bad = c; bad.t_max = 0;        CHECK(tsr_sim_create(&bad, NULL) == NULL, "t_max 0");
    bad = c; bad.num_ch = 0;       CHECK(tsr_sim_create(&bad, NULL) == NULL, "num_ch 0");
    bad = c; bad.k_max = 1000;     CHECK(tsr_sim_create(&bad, NULL) == NULL, "k_max");
    bad = c; bad.spm_bytes = 100;  CHECK(tsr_sim_create(&bad, NULL) == NULL, "spm_bytes");
    bad = c; bad.caps = TSR_CAPS_VU; CHECK(tsr_sim_create(&bad, NULL) == NULL, "caps VU");
    bad = c; bad.nireg = 8;        CHECK(tsr_sim_create(&bad, NULL) == NULL, "nireg");
    // NUM_CH <= 16: LD/ST name the channel in the 4-bit w0 sub field (spec 구성 파라미터).
    bad = c; bad.num_ch = TSR_NUM_CH_MAX + 1; CHECK(tsr_sim_create(&bad, NULL) == NULL, "num_ch 17");
    bad = c; bad.num_ch = 32;      CHECK(tsr_sim_create(&bad, NULL) == NULL, "num_ch 32");
    bad = c; bad.num_ch = 255;     CHECK(tsr_sim_create(&bad, NULL) == NULL, "num_ch 255");
}

// ---------------------------------------------------------------------------
static void test_regs_and_flow(void) {
    tsr_config c = cfg_u200(1);
    tsr_sim * s = tsr_sim_create(&c, NULL);
    prog_t p = {0};
    // LI / ADDI with wrap, R0 discard, ARG -> R8..R15.
    prog_add(&p, tsr_enc_li(1, 0x7FFFFFFFu));
    prog_add(&p, tsr_enc_addi(2, 1, 1));                 // 0x80000000
    prog_add(&p, tsr_enc_addi(3, 2, 0xFFFFFFFFu));       // 0x7FFFFFFF (-1)
    prog_add(&p, tsr_enc_li(0, 1234));                   // discarded
    prog_add(&p, tsr_enc_addi(0, 1, 5));                 // discarded
    prog_add(&p, tsr_enc_addi(4, 0, 77));                // R0 reads 0
    prog_add(&p, tsr_enc_addi(5, 8, 1));                 // ARG0 + 1
    prog_add(&p, tsr_enc_addi(6, 15, 0));                // ARG7
    prog_add(&p, tsr_enc_fence());
    prog_add(&p, tsr_enc_nop());
    prog_add(&p, tsr_enc_end());
    prog_add(&p, tsr_enc_li(7, 99));                     // after END: never executed
    const uint32_t args[8] = { 10, 11, 12, 13, 14, 15, 16, 0xFFFFFFF0u };
    uint32_t st = prog_run(s, &p, 0, args);
    CHECK(st == DONE, "flow status %u (err %u at %u)", st, sim_err_code(s), sim_err_pc(s));
    CHECK((uint32_t)tsr_sim_ireg(s, 1) == 0x7FFFFFFFu, "R1");
    CHECK((uint32_t)tsr_sim_ireg(s, 2) == 0x80000000u, "R2");
    CHECK((uint32_t)tsr_sim_ireg(s, 3) == 0x7FFFFFFFu, "R3");
    CHECK(tsr_sim_ireg(s, 0) == 0, "R0");
    CHECK(tsr_sim_ireg(s, 4) == 77, "R4");
    CHECK(tsr_sim_ireg(s, 5) == 11, "R5");
    CHECK((uint32_t)tsr_sim_ireg(s, 6) == 0xFFFFFFF0u, "R6");
    CHECK(tsr_sim_ireg(s, 7) == 0, "R7 (after END)");
    for (int i = 0; i < 8; ++i) CHECK((uint32_t)tsr_sim_ireg(s, 8 + (uint32_t)i) == args[i], "R%d = ARG%d", 8 + i, i);
    CHECK(tsr_sim_perf(s, TSR_PERF_RETIRED) == 11, "retired incl. END");
    // Next run: R1..R7 are zero again, R8.. = new ARGs (NULL -> 0).
    prog_reset(&p);
    prog_add(&p, tsr_enc_end());
    st = prog_run(s, &p, 0, NULL);
    CHECK(st == DONE, "end only");
    for (uint32_t i = 0; i < 16; ++i) CHECK(tsr_sim_ireg(s, i) == 0, "R%u zero at start", i);

    // Fetch from a non-zero, odd-aligned command address; odd instruction count.
    prog_reset(&p);
    for (int i = 0; i < 6; ++i) prog_add(&p, tsr_enc_addi(1, 1, 1));
    prog_add(&p, tsr_enc_end());
    const uint64_t perf0 = tsr_sim_perf(s, tsr_perf_ch_rd_beats(0));
    st = prog_run(s, &p, 12345, NULL);
    CHECK(st == DONE && tsr_sim_ireg(s, 1) == 6, "cmd at beat 12345");
    CHECK(tsr_sim_perf(s, tsr_perf_ch_rd_beats(0)) - perf0 == 4, "7 insns = 4 fetch beats");

    // Running past CMD_COUNT -> ILLEGAL_OP, ERR_PC = CMD_COUNT; count 0 -> ERR_PC 0.
    prog_reset(&p);
    prog_add(&p, tsr_enc_li(1, 5));
    prog_add(&p, tsr_enc_nop());
    expect_error(s, &p, TSR_ERR_ILLEGAL_OP, 2, 0, "past CMD_COUNT");
    CHECK(tsr_sim_ireg(s, 1) == 5, "insns before the end executed");
    st = tsr_sim_exec(s, 0, 0, NULL);
    CHECK(st == DONEERR && sim_err_code(s) == TSR_ERR_ILLEGAL_OP && sim_err_pc(s) == 0, "CMD_COUNT 0");

    // Command fetch beyond the DRAM limit: DRAM_RANGE at the first unfetchable pc.
    const uint64_t lim = tsr_sim_dram_limit_beats(s);
    CHECK(lim == (16ull << 30) / 64, "limit beats");
    prog_reset(&p);
    prog_add(&p, tsr_enc_nop());
    prog_add(&p, tsr_enc_nop());
    prog_load(s, &p, lim - 1);
    st = tsr_sim_exec(s, lim - 1, 3, NULL);
    CHECK(st == DONEERR && sim_err_code(s) == TSR_ERR_DRAM_RANGE && sim_err_pc(s) == 2 &&
          sim_err_info(s) == (uint32_t)lim, "fetch past DRAM end: code %u pc %u info 0x%x", sim_err_code(s),
          sim_err_pc(s), sim_err_info(s));
    st = tsr_sim_exec(s, lim + 5, 1, NULL);
    CHECK(st == DONEERR && sim_err_code(s) == TSR_ERR_DRAM_RANGE && sim_err_pc(s) == 0 &&
          sim_err_info(s) == (uint32_t)(lim + 5), "cmd addr beyond DRAM");
    st = tsr_sim_exec(s, (1ull << 40) + 3, 1, NULL);   // 64-bit CMD_ADDR, ERR_INFO = low 32 bits
    CHECK(st == DONEERR && sim_err_code(s) == TSR_ERR_DRAM_RANGE && sim_err_info(s) == 3, "cmd addr >= 2^32");
    prog_free(&p);
    tsr_sim_destroy(s);
}

// ---------------------------------------------------------------------------
static void test_ldst(void) {
    tsr_config c = cfg_u200(1);
    tsr_sim * s = tsr_sim_create(&c, NULL);
    rng_t r = { 42 };
    // fill IO areas of all channels with random bytes
    for (uint32_t ch = 0; ch < 4; ++ch) rng_bytes(&r, tsr_sim_dram(s, ch) + TSR_REGION_IO_OFF, 64 * 1024);
    prog_t p = {0};
    // LD 5 lines ch2 -> SPM 100; ST SPM 100..104 -> ch3 IO+512; register-relative copy back.
    prog_add(&p, tsr_enc_ld(2, 100, IO_BEAT + 7, 5));
    prog_add(&p, tsr_enc_st(3, 100, IO_BEAT + 512, 5));
    prog_add(&p, tsr_enc_li(1, 0xFFFFFFFFu));            // -1
    prog_add(&p, tsr_enc_li(2, (uint32_t)(IO_BEAT + 600)));
    tsr_insn ld = tsr_enc_ld(0, 201, 0xFFFFFFF0u, 2);   // SPM = R1 + 201 = 200 (wrap); DRAM = R2 - 16
    tsr_insn_set_base(&ld, 1, 2, 0);
    prog_add(&p, ld);
    tsr_insn st_ = tsr_enc_st(1, 0, 5, 2);              // SPM = R4(=200) + 0; DRAM = R2 + 5
    tsr_insn_set_base(&st_, 4, 2, 0);
    prog_add(&p, tsr_enc_li(4, 200));
    prog_add(&p, st_);
    prog_add(&p, tsr_enc_ld(0, 300, IO_BEAT, 0));       // zero-length: no-op
    prog_add(&p, tsr_enc_end());
    const uint32_t st = prog_run(s, &p, 0, NULL);
    CHECK(st == DONE, "ldst status %u err %u pc %u", st, sim_err_code(s), sim_err_pc(s));
    CHECK(memcmp(tsr_sim_spm(s) + 100 * 64, tsr_sim_dram(s, 2) + (IO_BEAT + 7) * 64, 5 * 64) == 0, "LD data");
    CHECK(memcmp(tsr_sim_dram(s, 3) + (IO_BEAT + 512) * 64, tsr_sim_dram(s, 2) + (IO_BEAT + 7) * 64, 5 * 64) == 0, "ST data");
    CHECK(memcmp(tsr_sim_spm(s) + 200 * 64, tsr_sim_dram(s, 0) + (IO_BEAT + 600 - 16) * 64, 2 * 64) == 0, "reg-relative LD");
    CHECK(memcmp(tsr_sim_dram(s, 1) + (IO_BEAT + 605) * 64, tsr_sim_dram(s, 0) + (IO_BEAT + 600 - 16) * 64, 2 * 64) == 0,
          "reg-relative ST");
    CHECK(tsr_sim_spm(s)[300 * 64] == 0xA5, "zero-length LD wrote nothing");

    // Exact-fit boundaries pass: SPM last line, DRAM last beat.
    const uint32_t spm_lines = c.spm_bytes / 64;
    const uint64_t lim = tsr_sim_dram_limit_beats(s);
    expect_ok(s, tsr_enc_ld(0, spm_lines - 2, (uint32_t)(lim - 2), 2), "LD exact fit");
    expect_ok(s, tsr_enc_st(3, spm_lines - 1, (uint32_t)(lim - 1), 1), "ST exact fit");
    expect_ok(s, tsr_enc_ld(0, spm_lines, (uint32_t)lim, 0), "zero length at the limits");
    prog_free(&p);
    tsr_sim_destroy(s);

    // ADDR_BEAT_W smaller than the channel: limit = 2^addr_beat_w beats.
    tsr_config c2 = c;
    c2.addr_beat_w = 20;
    s = tsr_sim_create(&c2, NULL);
    CHECK(tsr_sim_dram_limit_beats(s) == (1ull << 20), "limit = 2^ADDR_BEAT_W");
    expect_ok(s, tsr_enc_ld(0, 0, (1u << 20) - 1, 1), "LD last addressable beat");
    expect_insn_error(s, NULL, 0, tsr_enc_ld(0, 0, (1u << 20) - 1, 2), TSR_ERR_DRAM_RANGE, (1u << 20) - 1, "LD past 2^ADDR_BEAT_W");
    tsr_sim_destroy(s);
    // CH_BYTES smaller than 2^ADDR_BEAT_W beats
    c2 = c;
    c2.ch_bytes = 1ull << 30;
    s = tsr_sim_create(&c2, NULL);
    CHECK(tsr_sim_dram_limit_beats(s) == (1ull << 24), "limit = CH_BYTES/64");
    expect_insn_error(s, NULL, 0, tsr_enc_st(1, 0, (1u << 24), 1), TSR_ERR_DRAM_RANGE, 1u << 24, "ST past CH_BYTES");
    tsr_sim_destroy(s);

    // NUM_CH = 16 (the maximum): every channel, including ch15, is reachable by LD/ST and
    // channels are distinct (no aliasing through the 4-bit field).
    c2 = c;
    c2.num_ch = TSR_NUM_CH_MAX;
    c2.ch_bytes = 1ull << 30;
    s = tsr_sim_create(&c2, NULL);
    CHECK(s != NULL, "create num_ch 16");
    if (s != NULL) {
        for (uint32_t ch = 0; ch < TSR_NUM_CH_MAX; ++ch) memset(tsr_sim_dram(s, ch) + TSR_REGION_IO_OFF, 0x40 + (int)ch, 64);
        prog_t q = {0};
        for (uint32_t ch = 0; ch < TSR_NUM_CH_MAX; ++ch) prog_add(&q, tsr_enc_ld(ch, ch, IO_BEAT, 1));
        prog_add(&q, tsr_enc_st(15, 0, IO_BEAT + 1, 1));     // ch0's line -> ch15
        prog_add(&q, tsr_enc_end());
        const uint32_t st16 = prog_run(s, &q, 0, NULL);
        CHECK(st16 == DONE, "num_ch 16 LD/ST status %u err %u", st16, sim_err_code(s));
        for (uint32_t ch = 0; ch < TSR_NUM_CH_MAX; ++ch) {
            CHECK(tsr_sim_spm(s)[ch * 64] == 0x40 + ch, "num_ch 16: LD ch%u got 0x%02x", ch, tsr_sim_spm(s)[ch * 64]);
            if (ch > 0) {   // ch0 also counts command fetch beats
                CHECK(tsr_sim_perf(s, tsr_perf_ch_rd_beats(ch)) == 1, "num_ch 16: rd beats ch%u", ch);
            }
        }
        CHECK(tsr_sim_dram(s, 15)[(IO_BEAT + 1) * 64] == 0x40, "num_ch 16: ST to ch15");
        CHECK(tsr_sim_dram(s, 0)[(IO_BEAT + 1) * 64] == 0, "num_ch 16: ST did not alias ch0");
        prog_free(&q);
        tsr_sim_destroy(s);
    }
}

// ---------------------------------------------------------------------------
// Reserved fields: flipping any bit an instruction does not use -> ILLEGAL_OP.
// ---------------------------------------------------------------------------
typedef struct {
    const char * name;
    tsr_insn     base;        // valid instruction
    uint32_t     used[8];     // bits that belong to fields the instruction uses
} rsv_case;

static void test_reserved_fields(void) {
    tsr_config c = cfg_u200(8);
    tsr_sim * s = tsr_sim_create(&c, NULL);
    const uint32_t OPC = 0xFFu, SUB = 0xF00u, RA = 0xF0000u, RB = 0xF00000u, RC = 0xF000000u, RD = 0xF0000000u;
    const uint32_t ALL = 0xFFFFFFFFu;
    rsv_case cases[] = {
        { "NOP",   tsr_enc_nop(),   { OPC, 0, 0, 0, 0, 0, 0, 0 } },
        { "END",   tsr_enc_end(),   { OPC, 0, 0, 0, 0, 0, 0, 0 } },
        { "FENCE", tsr_enc_fence(), { OPC, 0, 0, 0, 0, 0, 0, 0 } },
        { "LI",    tsr_enc_li(1, 5),      { OPC | RD, ALL, 0, 0, 0, 0, 0, 0 } },
        { "ADDI",  tsr_enc_addi(1, 2, 5), { OPC | RA | RD, ALL, 0, 0, 0, 0, 0, 0 } },
        { "LD",    tsr_enc_ld(0, 0, (uint32_t)IO_BEAT, 1), { OPC | SUB | RA | RB, ALL, ALL, ALL, 0, 0, 0, 0 } },
        { "ST",    tsr_enc_st(0, 0, (uint32_t)IO_BEAT, 1), { OPC | SUB | RA | RB, ALL, ALL, ALL, 0, 0, 0, 0 } },
        { "QACT",  tsr_enc_qact(TSR_QA_A32, 0, 100, 32, 1, 0), { OPC | SUB | RA | RB, ALL, ALL, ALL, ALL, ALL, 0, 0 } },
        { "MVM",   tsr_enc_mvm(TSR_FMT_Q8_0, 0, 0, 100, (uint32_t)(TSR_REGION_WEIGHT_OFF / 64), 4, 32, 1, 0),
                   { OPC | SUB | (1u << 12) | RA | RB | RC, ALL, ALL, ALL, ALL, 0x000FFFFFu, ALL, 0 } },
    };
    for (size_t k = 0; k < sizeof(cases) / sizeof(cases[0]); ++k) {
        const rsv_case * rc = &cases[k];
        if (strcmp(rc->name, "END") != 0) expect_ok(s, rc->base, rc->name);
        int n_flips = 0;
        for (uint32_t wi = 0; wi < 8; ++wi) {
            for (uint32_t bit = 0; bit < 32; ++bit) {
                if (rc->used[wi] & (1u << bit)) continue;
                tsr_insn bad = rc->base;
                bad.w[wi] ^= 1u << bit;
                char what[96];
                snprintf(what, sizeof(what), "%s reserved w%u bit %u", rc->name, wi, bit);
                expect_insn_error(s, NULL, 0, bad, TSR_ERR_ILLEGAL_OP, 0, what);
                ++n_flips;
            }
        }
        CHECK(n_flips > 0, "%s has reserved bits", rc->name);
    }
    // Sub-code values: LD/ST channel >= NUM_CH, QACT kind >= 2 -> ILLEGAL_OP; MVM fmt >= 2 -> BAD_FMT.
    for (uint32_t sub = 0; sub < 16; ++sub) {
        char what[64];
        tsr_insn ld = tsr_enc_ld(sub, 0, (uint32_t)IO_BEAT, 1);
        tsr_insn st = tsr_enc_st(sub, 0, (uint32_t)IO_BEAT, 1);
        tsr_insn qa = tsr_enc_qact(sub, 0, 100, 256, 1, 0);
        tsr_insn mv = tsr_enc_mvm(sub, 0, 0, 100, (uint32_t)(TSR_REGION_WEIGHT_OFF / 64), 4, 256, 1, 0);
        snprintf(what, sizeof(what), "LD ch %u", sub);
        if (sub < 4) expect_ok(s, ld, what); else expect_insn_error(s, NULL, 0, ld, TSR_ERR_ILLEGAL_OP, 0, what);
        snprintf(what, sizeof(what), "ST ch %u", sub);
        if (sub < 4) expect_ok(s, st, what); else expect_insn_error(s, NULL, 0, st, TSR_ERR_ILLEGAL_OP, 0, what);
        snprintf(what, sizeof(what), "QACT kind %u", sub);
        if (sub < 2) expect_ok(s, qa, what); else expect_insn_error(s, NULL, 0, qa, TSR_ERR_ILLEGAL_OP, 0, what);
        snprintf(what, sizeof(what), "MVM fmt %u", sub);
        if (sub < 2) expect_ok(s, mv, what); else expect_insn_error(s, NULL, 0, mv, TSR_ERR_BAD_FMT, 0, what);
    }
    // Every opcode outside the M1 set is ILLEGAL_OP (this simulator has CAPS = 0).
    const uint32_t m1[] = { TSR_OP_NOP, TSR_OP_END, TSR_OP_LI, TSR_OP_ADDI, TSR_OP_LD, TSR_OP_ST,
                            TSR_OP_QACT, TSR_OP_MVM, TSR_OP_FENCE };
    for (uint32_t op = 0; op < 256; ++op) {
        int is_m1 = 0;
        for (size_t i = 0; i < sizeof(m1) / sizeof(m1[0]); ++i) is_m1 |= (op == m1[i]);
        if (is_m1) continue;
        char what[32];
        snprintf(what, sizeof(what), "opcode 0x%02x", op);
        expect_insn_error(s, NULL, 0, insn_raw(op, 0, 0, 0, 0, 0, 0, 0), TSR_ERR_ILLEGAL_OP, 0, what);
    }
    // A plausible later-milestone encoding is still illegal (e.g. MULI R1 = R2 * 3, SOP MOVI).
    expect_insn_error(s, NULL, 0, insn_raw(tsr_w0(TSR_OP_MULI, 0, 0, 2, 0, 0, 1), 3, 0, 0, 0, 0, 0, 0),
                      TSR_ERR_ILLEGAL_OP, 0, "MULI");
    expect_insn_error(s, NULL, 0, insn_raw(tsr_w0(TSR_OP_SOP, TSR_SOP_MOVI, 0, 0, 0, 0, 0), 1, 0x3F800000u, 0, 0, 0, 0, 0),
                      TSR_ERR_ILLEGAL_OP, 0, "SOP");
    tsr_sim_destroy(s);
}

// ---------------------------------------------------------------------------
// Error codes, priorities, ERR_INFO and no state change.
// ---------------------------------------------------------------------------
static void test_errors(void) {
    tsr_config c = cfg_u200(2);
    tsr_sim * s = tsr_sim_create(&c, NULL);
    rng_t r = { 7 };
    rng_bytes(&r, tsr_sim_spm(s), c.spm_bytes);   // arbitrary SPM content to detect changes
    for (uint32_t ch = 0; ch < 4; ++ch) rng_bytes(&r, tsr_sim_dram(s, ch) + TSR_REGION_IO_OFF, 1u << 20);
    const uint32_t SL = c.spm_bytes / 64;                   // SPM lines
    const uint32_t LIM = (uint32_t)tsr_sim_dram_limit_beats(s);
    const uint32_t WOFF = (uint32_t)(TSR_REGION_WEIGHT_OFF / 64);
    tsr_insn setup[3] = { tsr_enc_li(1, SL - 3), tsr_enc_li(2, 0xFFFFFFFFu), tsr_enc_li(3, LIM - 10) };

    // LD / ST
    expect_insn_error(s, NULL, 0, tsr_enc_ld(0, SL - 1, (uint32_t)IO_BEAT, 2), TSR_ERR_SPM_RANGE, SL - 1, "LD SPM");
    expect_insn_error(s, NULL, 0, tsr_enc_ld(1, 0, LIM - 1, 2), TSR_ERR_DRAM_RANGE, LIM - 1, "LD DRAM");
    expect_insn_error(s, NULL, 0, tsr_enc_st(2, SL, (uint32_t)IO_BEAT, 1), TSR_ERR_SPM_RANGE, SL, "ST SPM");
    expect_insn_error(s, NULL, 0, tsr_enc_st(3, 0, 0xFFFFFFFFu, 1), TSR_ERR_DRAM_RANGE, 0xFFFFFFFFu, "ST DRAM");
    expect_insn_error(s, NULL, 0, tsr_enc_st(3, 0, (uint32_t)IO_BEAT, 0xFFFFFFFFu), TSR_ERR_SPM_RANGE, 0, "ST huge n");
    expect_insn_error(s, NULL, 0, tsr_enc_ld(0, SL + 1, (uint32_t)LIM, 0), TSR_ERR_SPM_RANGE, SL + 1, "LD n=0 start past SPM end");
    // priorities: ILLEGAL (ch) > SPM > DRAM
    expect_insn_error(s, NULL, 0, tsr_enc_ld(4, SL, LIM, 1), TSR_ERR_ILLEGAL_OP, 0, "LD ch+SPM+DRAM");
    expect_insn_error(s, NULL, 0, tsr_enc_ld(0, SL, LIM, 1), TSR_ERR_SPM_RANGE, SL, "LD SPM+DRAM");
    // register-relative: address = R + imm, ERR_INFO = computed address
    {
        tsr_insn ld = tsr_enc_ld(0, 2, 0, 2);          // SPM = (SL-3) + 2 = SL-1, 2 lines -> overflow
        tsr_insn_set_base(&ld, 1, 0, 0);
        expect_insn_error(s, setup, 3, ld, TSR_ERR_SPM_RANGE, SL - 1, "LD reg SPM");
        tsr_insn st = tsr_enc_st(0, 0, 9, 2);          // DRAM = (LIM-10) + 9 = LIM-1, 2 beats
        tsr_insn_set_base(&st, 0, 3, 0);
        expect_insn_error(s, setup, 3, st, TSR_ERR_DRAM_RANGE, LIM - 1, "ST reg DRAM");
        tsr_insn ok = tsr_enc_st(0, 0, 8, 2);          // LIM-2 .. LIM: fits
        tsr_insn_set_base(&ok, 0, 3, 0);
        prog_t p = {0};
        for (int i = 0; i < 3; ++i) prog_add(&p, setup[i]);
        prog_add(&p, ok);
        prog_add(&p, tsr_enc_end());
        CHECK(prog_run(s, &p, 0, NULL) == DONE, "ST reg exact fit");
        prog_free(&p);
        tsr_insn wrap = tsr_enc_ld(0, 0, 0, 1);        // SPM = 0xFFFFFFFF + 0 -> out of range
        tsr_insn_set_base(&wrap, 2, 0, 0);
        expect_insn_error(s, setup, 3, wrap, TSR_ERR_SPM_RANGE, 0xFFFFFFFFu, "LD SPM 0xFFFFFFFF");
    }

    // QACT
    expect_insn_error(s, NULL, 0, tsr_enc_qact(0, 0, 100, 0, 1, 0), TSR_ERR_K_RANGE, 0, "QACT K=0");
    expect_insn_error(s, NULL, 0, tsr_enc_qact(0, 0, 100, 48, 1, 0), TSR_ERR_K_RANGE, 0, "QACT A32 K%32");
    expect_insn_error(s, NULL, 0, tsr_enc_qact(1, 0, 100, 128, 1, 0), TSR_ERR_K_RANGE, 0, "QACT A256 K%256");
    expect_insn_error(s, NULL, 0, tsr_enc_qact(0, 0, 100, 32768 + 32, 1, 0), TSR_ERR_K_RANGE, 0, "QACT K>K_MAX");
    expect_ok(s, tsr_enc_qact(1, 0, 1000, 32768, 2, 0), "QACT K=K_MAX T=T_MAX");
    expect_insn_error(s, NULL, 0, tsr_enc_qact(0, 0, 100, 32, 0, 0), TSR_ERR_T_RANGE, 0, "QACT T=0");
    expect_insn_error(s, NULL, 0, tsr_enc_qact(0, 0, 100, 32, 3, 0), TSR_ERR_T_RANGE, 0, "QACT T>T_MAX");
    expect_insn_error(s, NULL, 0, tsr_enc_qact(0, SL - 1, 0, 32, 1, 0), TSR_ERR_SPM_RANGE, SL - 1, "QACT dst");
    expect_insn_error(s, NULL, 0, tsr_enc_qact(0, 0, SL - 1, 32, 1, 0), TSR_ERR_SPM_RANGE, SL - 1, "QACT src");
    expect_insn_error(s, NULL, 0, tsr_enc_qact(0, 0, SL - 3, 32, 2, 3), TSR_ERR_SPM_RANGE, SL - 3, "QACT src stride");
    expect_ok(s, tsr_enc_qact(0, 0, SL - 4, 32, 2, 2), "QACT src stride exact fit");
    expect_ok(s, tsr_enc_qact(0, SL - 4, 0, 32, 2, 0), "QACT dst exact fit (2 x 2 lines)");
    // priorities
    expect_insn_error(s, NULL, 0, tsr_enc_qact(5, SL, SL, 33, 0, 0), TSR_ERR_ILLEGAL_OP, 0, "QACT kind+K+T+SPM");
    expect_insn_error(s, NULL, 0, tsr_enc_qact(0, SL, SL, 33, 0, 0), TSR_ERR_K_RANGE, 0, "QACT K+T+SPM");
    expect_insn_error(s, NULL, 0, tsr_enc_qact(0, SL, SL, 32, 0, 0), TSR_ERR_T_RANGE, 0, "QACT T+SPM");
    expect_insn_error(s, NULL, 0, tsr_enc_qact(0, SL, SL + 7, 32, 1, 0), TSR_ERR_SPM_RANGE, SL, "QACT dst before src");

    // MVM: N = 9, K = 64 (Q8_0) / 256 (Q4_K), T = 2
    const uint32_t rb8 = tsr_row_beats(TSR_FMT_Q8_0, 64);
    const uint32_t R9 = 3;   // ceil(9/4)
    expect_ok(s, tsr_enc_mvm(0, 0, 0, 100, WOFF, 9, 64, 2, 0), "MVM ok");
    expect_insn_error(s, NULL, 0, tsr_enc_mvm(0, 2, 0, 100, WOFF, 9, 64, 2, 0), TSR_ERR_ILLEGAL_OP, 0, "MVM flag bit1");
    expect_insn_error(s, NULL, 0, tsr_enc_mvm(3, 0, 0, 100, WOFF, 9, 64, 2, 0), TSR_ERR_BAD_FMT, 0, "MVM fmt 3");
    expect_insn_error(s, NULL, 0, tsr_enc_mvm(0, 0, 0, 100, WOFF, 9, 0, 2, 0), TSR_ERR_K_RANGE, 0, "MVM K=0");
    expect_insn_error(s, NULL, 0, tsr_enc_mvm(0, 0, 0, 100, WOFF, 9, 48, 2, 0), TSR_ERR_K_RANGE, 0, "MVM Q8_0 K%32");
    expect_insn_error(s, NULL, 0, tsr_enc_mvm(1, 0, 0, 100, WOFF, 9, 64, 2, 0), TSR_ERR_K_RANGE, 0, "MVM Q4_K K%256");
    expect_insn_error(s, NULL, 0, tsr_enc_mvm(1, 0, 0, 100, WOFF, 9, 32768 + 256, 2, 0), TSR_ERR_K_RANGE, 0, "MVM K>K_MAX");
    expect_insn_error(s, NULL, 0, tsr_enc_mvm(0, 0, 0, 100, WOFF, 9, 64, 0, 0), TSR_ERR_T_RANGE, 0, "MVM T=0");
    expect_insn_error(s, NULL, 0, tsr_enc_mvm(0, 0, 0, 100, WOFF, 9, 64, 3, 0), TSR_ERR_T_RANGE, 0, "MVM T>T_MAX");
    expect_insn_error(s, NULL, 0, tsr_enc_mvm(0, 0, 0, 100, WOFF, 9, 64, 15, 0), TSR_ERR_T_RANGE, 0, "MVM T=15");
    // dst: 2 tokens x ceil(36/64)=1 line, stride 0 -> 1 line: needs dst + 2 <= SL
    expect_insn_error(s, NULL, 0, tsr_enc_mvm(0, 0, SL - 1, 100, WOFF, 9, 64, 2, 0), TSR_ERR_SPM_RANGE, SL - 1, "MVM dst");
    expect_ok(s, tsr_enc_mvm(0, 0, SL - 2, 100, WOFF, 9, 64, 2, 0), "MVM dst exact fit");
    expect_insn_error(s, NULL, 0, tsr_enc_mvm(0, 0, SL - 10, 100, WOFF, 9, 64, 2, 10), TSR_ERR_SPM_RANGE, SL - 10, "MVM dst stride");
    expect_ok(s, tsr_enc_mvm(0, 0, SL - 10, 100, WOFF, 9, 64, 2, 9), "MVM dst stride exact fit");
    const uint32_t qa8 = tsr_qa_lines(TSR_QA_A32, 64);
    expect_insn_error(s, NULL, 0, tsr_enc_mvm(0, 0, 0, SL - 2 * qa8 + 1, WOFF, 9, 64, 2, 0), TSR_ERR_SPM_RANGE,
                      SL - 2 * qa8 + 1, "MVM x");
    expect_ok(s, tsr_enc_mvm(0, 0, 0, SL - 2 * qa8, WOFF, 9, 64, 2, 0), "MVM x exact fit");
    // DRAM: channel 0 holds R = 3 rows -> W_OFF + 3*ROW_BEATS <= LIM
    expect_insn_error(s, NULL, 0, tsr_enc_mvm(0, 0, 0, 100, LIM - R9 * rb8 + 1, 9, 64, 2, 0), TSR_ERR_DRAM_RANGE,
                      LIM - R9 * rb8 + 1, "MVM W_OFF");
    expect_ok(s, tsr_enc_mvm(0, 0, 0, 100, LIM - R9 * rb8, 9, 64, 2, 0), "MVM W_OFF exact fit");
    expect_insn_error(s, NULL, 0, tsr_enc_mvm(0, 0, 0, 100, LIM + 1, 0, 64, 1, 0), TSR_ERR_DRAM_RANGE, LIM + 1, "MVM N=0 W_OFF past end");
    expect_insn_error(s, NULL, 0, tsr_enc_mvm(0, 0, 0, 100, WOFF, 0xFFFFFFFFu, 64, 1, 0), TSR_ERR_SPM_RANGE, 0, "MVM huge N");
    // priorities
    expect_insn_error(s, NULL, 0, tsr_enc_mvm(7, 8, SL, SL, LIM, 9, 33, 0, 0), TSR_ERR_ILLEGAL_OP, 0, "MVM all");
    expect_insn_error(s, NULL, 0, tsr_enc_mvm(7, 0, SL, SL, LIM, 9, 33, 0, 0), TSR_ERR_BAD_FMT, 0, "MVM fmt+K+T+SPM+DRAM");
    expect_insn_error(s, NULL, 0, tsr_enc_mvm(0, 0, SL, SL, LIM, 9, 33, 0, 0), TSR_ERR_K_RANGE, 0, "MVM K+T+SPM+DRAM");
    expect_insn_error(s, NULL, 0, tsr_enc_mvm(0, 0, SL, SL, LIM, 9, 32, 0, 0), TSR_ERR_T_RANGE, 0, "MVM T+SPM+DRAM");
    expect_insn_error(s, NULL, 0, tsr_enc_mvm(0, 0, SL, SL + 1, LIM, 9, 32, 1, 0), TSR_ERR_SPM_RANGE, SL, "MVM dst first");
    expect_insn_error(s, NULL, 0, tsr_enc_mvm(0, 0, 0, SL + 1, LIM, 9, 32, 1, 0), TSR_ERR_SPM_RANGE, SL + 1, "MVM x before DRAM");
    {
        tsr_insn m = tsr_enc_mvm(0, 0, 0, 100, 10, 9, 64, 1, 0);   // W_OFF = (LIM-10) + 10 = LIM -> DRAM
        tsr_insn_set_base(&m, 0, 0, 3);
        expect_insn_error(s, setup, 3, m, TSR_ERR_DRAM_RANGE, LIM, "MVM reg W_OFF");
        tsr_insn m2 = tsr_enc_mvm(0, 1, 3, 100, WOFF, 9, 64, 1, 0);   // dst = R1 + 3 = SL, ACC
        tsr_insn_set_base(&m2, 1, 0, 0);
        expect_insn_error(s, setup, 3, m2, TSR_ERR_SPM_RANGE, SL, "MVM reg dst (ACC)");
        // ACC with dst exactly fitting passes
        tsr_insn m3 = tsr_enc_mvm(0, 1, 2, 100, WOFF, 9, 64, 1, 0);   // dst = SL - 1, 1 line
        tsr_insn_set_base(&m3, 1, 0, 0);
        prog_t p = {0};
        for (int i = 0; i < 3; ++i) prog_add(&p, setup[i]);
        prog_add(&p, m3);
        prog_add(&p, tsr_enc_end());
        CHECK(prog_run(s, &p, 0, NULL) == DONE, "MVM reg dst ACC fits");
        prog_free(&p);
    }
    tsr_sim_destroy(s);
}

// ---------------------------------------------------------------------------
// ABORT
// ---------------------------------------------------------------------------
typedef struct { tsr_sim * s; int delay_ms; } abort_arg;

static void * abort_thread(void * a) {
    abort_arg * aa = (abort_arg *)a;
    struct timespec ts = { aa->delay_ms / 1000, (long)(aa->delay_ms % 1000) * 1000000L };
    nanosleep(&ts, NULL);
    tsr_sim_csr_write(aa->s, TSR_CSR_CTRL, TSR_CTRL_ABORT);
    return NULL;
}

static void test_abort(void) {
    tsr_config c = cfg_u200(1);
    tsr_sim * s = tsr_sim_create(&c, NULL);
    prog_t p = {0};
    prog_add(&p, tsr_enc_li(1, 9));
    prog_add(&p, tsr_enc_end());
    prog_load(s, &p, 0);
    // ABORT before the first instruction: ABORTED at pc 0, nothing retires.
    tsr_sim_csr_write(s, TSR_CSR_CMD_ADDR_LO, 0);
    tsr_sim_csr_write(s, TSR_CSR_CMD_ADDR_HI, 0);
    tsr_sim_csr_write(s, TSR_CSR_CMD_COUNT, 2);
    tsr_sim_csr_write(s, TSR_CSR_CTRL, TSR_CTRL_START);
    tsr_sim_csr_write(s, TSR_CSR_CTRL, TSR_CTRL_ABORT);
    uint32_t st = tsr_sim_run(s);
    CHECK(st == DONEERR && sim_err_code(s) == TSR_ERR_ABORTED && sim_err_pc(s) == 0 && sim_err_info(s) == 0,
          "abort before start: status %u code %u pc %u", st, sim_err_code(s), sim_err_pc(s));
    CHECK(tsr_sim_ireg(s, 1) == 0 && tsr_sim_perf(s, TSR_PERF_RETIRED) == 0, "nothing executed");
    // ABORT while idle is ignored by the next START.
    tsr_sim_csr_write(s, TSR_CSR_CTRL, TSR_CTRL_ABORT);
    st = tsr_sim_exec(s, 0, 2, NULL);
    CHECK(st == DONE && tsr_sim_ireg(s, 1) == 9, "ABORT while idle ignored");

    // ABORT from another thread during a long program (MVMs of 512 x 4096 Q8_0).
    const uint32_t N = 512, K = 4096, n_mvm = 4000;
    const uint32_t WOFF = (uint32_t)(TSR_REGION_WEIGHT_OFF / 64);
    prog_reset(&p);
    prog_add(&p, tsr_enc_qact(0, 0, 1000, K, 1, 0));
    for (uint32_t i = 0; i < n_mvm; ++i) prog_add(&p, tsr_enc_mvm(0, 0, 2000, 0, WOFF, N, K, 1, 0));
    prog_add(&p, tsr_enc_end());
    const uint64_t ret0 = tsr_sim_perf(s, TSR_PERF_RETIRED);
    abort_arg aa = { s, 30 };
    prog_load(s, &p, 0);
    tsr_sim_csr_write(s, TSR_CSR_CMD_COUNT, p.n);
    tsr_sim_csr_write(s, TSR_CSR_CTRL, TSR_CTRL_START);
    pthread_t th;
    pthread_create(&th, NULL, abort_thread, &aa);
    st = tsr_sim_run(s);
    pthread_join(th, NULL);
    const uint32_t epc = sim_err_pc(s);
    CHECK(st == DONEERR && sim_err_code(s) == TSR_ERR_ABORTED, "threaded abort: status %u code %u", st, sim_err_code(s));
    CHECK(epc > 0 && epc < p.n, "aborted mid-program at %u of %u", epc, p.n);
    CHECK(tsr_sim_perf(s, TSR_PERF_RETIRED) - ret0 == epc, "retired = ERR_PC (%llu vs %u)",
          (unsigned long long)(tsr_sim_perf(s, TSR_PERF_RETIRED) - ret0), epc);
    prog_free(&p);
    tsr_sim_destroy(s);
}

int main(void) {
    test_csr();
    test_regs_and_flow();
    test_ldst();
    test_reserved_fields();
    test_errors();
    test_abort();
    return test_finish("sim_basic");
}
