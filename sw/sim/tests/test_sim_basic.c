// test_sim_basic.c — ISA simulator op-level core: CSRs, LD/ST, every reachable op error code
// (with ERR_INFO and "no state change"), reserved-field checks including x16 (EX §7.1), the
// reserved and retired opcodes, perf counters.
#include "test_sim_util.h"

static const uint64_t IO_BEAT = TSR_REGION_IO_OFF / TSR_BEAT_BYTES;   // 64 MiB / 64

// ---------------------------------------------------------------------------
// State snapshot for "a failing op changes no state"
// ---------------------------------------------------------------------------
typedef struct {
    uint8_t * spm;          // full SPM copy
    uint8_t * dram[4];      // first MiB of each channel's IO region
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
}
static int snap_same_mem(tsr_sim * s, const snap_t * z) {
    if (memcmp(z->spm, tsr_sim_spm(s), tsr_sim_config(s)->spm_bytes) != 0) return 0;
    for (uint32_t c = 0; c < 4 && c < tsr_sim_config(s)->num_ch; ++c) {
        if (memcmp(z->dram[c], tsr_sim_dram(s, c) + TSR_REGION_IO_OFF, SNAP_DRAM_BYTES) != 0) return 0;
    }
    return 1;
}
static void snap_free(snap_t * z) {
    free(z->spm);
    for (int c = 0; c < 4; ++c) free(z->dram[c]);
    memset(z, 0, sizeof(*z));
}
static snap_t g_snap;

// One op that must fail with code / info. Checks that SPM and the first MiB of every channel's
// IO region are unchanged, that no cycles were counted and that the op still counts as retired.
static void expect_insn_error(tsr_sim * s, tsr_insn bad, uint32_t code, uint32_t info, const char * what) {
    snap_take(s, &g_snap);
    const uint64_t ret0 = tsr_sim_perf(s, TSR_PERF_RETIRED), cyc0 = tsr_sim_perf(s, TSR_PERF_CYCLES);
    uint32_t got_info = 0xDEADBEEFu;
    const uint32_t err = tsr_sim_op(s, bad.w, &got_info);
    CHECK(err == code, "%s: err_code %u, expected %u", what, err, code);
    CHECK(got_info == info, "%s: err_info 0x%x, expected 0x%x", what, got_info, info);
    CHECK(snap_same_mem(s, &g_snap), "%s: SPM or DRAM changed", what);
    CHECK(tsr_sim_perf(s, TSR_PERF_RETIRED) - ret0 == 1 && tsr_sim_perf(s, TSR_PERF_CYCLES) == cyc0,
          "%s: retired %llu, cycles +%llu", what, (unsigned long long)(tsr_sim_perf(s, TSR_PERF_RETIRED) - ret0),
          (unsigned long long)(tsr_sim_perf(s, TSR_PERF_CYCLES) - cyc0));
}

static void expect_ok(tsr_sim * s, tsr_insn insn, const char * what) {
    uint32_t info = 0xDEADBEEFu;
    const uint32_t err = tsr_sim_op(s, insn.w, &info);
    CHECK(err == TSR_ERR_NONE && info == 0, "%s: err %u info 0x%x", what, err, info);
}

// ---------------------------------------------------------------------------
static void test_csr(void) {
    tsr_config c = cfg_u200(1);
    tsr_sim * s = tsr_sim_create(&c, NULL);
    CHECK(s != NULL, "create");
    CHECK(tsr_sim_csr_read(s, TSR_CSR_ID) == 0x54535231u, "ID");
    CHECK(tsr_sim_csr_read(s, TSR_CSR_VERSION) == 0x0100u, "VERSION 0x%x (Tesseract Core capability 1.0)",
          tsr_sim_csr_read(s, TSR_CSR_VERSION));
    CHECK(tsr_sim_csr_read(s, TSR_CSR_CFG0) == (4u | (128u << 8) | (1u << 20)), "CFG0 0x%x", tsr_sim_csr_read(s, TSR_CSR_CFG0));
    CHECK(tsr_sim_csr_read(s, TSR_CSR_CFG1) == 65536u, "CFG1");
    CHECK(tsr_sim_csr_read(s, TSR_CSR_CFG2) == (16u | (16u << 8) | (128u << 16)), "CFG2 0x%x", tsr_sim_csr_read(s, TSR_CSR_CFG2));
    CHECK(tsr_sim_csr_read(s, TSR_CSR_CAPS) == 0, "CAPS");
    // The rest of the map (EX §5) is sim_csr's. Gaps of the map read 0, unaligned offsets too.
    for (uint32_t off = 0x08C; off < TSR_CSR_PERF_BASE; off += 4) {
        if (off == 0x08C || (off >= 0x0A0 && off < TSR_CSR_LOAD_ADDR) || off > TSR_CSR_LOAD_STATUS)
            CHECK(tsr_sim_csr_read(s, off) == 0, "offset 0x%03x reads 0", off);
    }
    CHECK(tsr_sim_csr_read(s, 0x26) == 0, "unaligned reads 0");
    // Writes to the identity and configuration registers are ignored.
    for (uint32_t off = 0; off < TSR_CSR_CTRL; off += 4) tsr_sim_csr_write(s, off, 0xFFFFFFFFu);
    CHECK(tsr_sim_csr_read(s, TSR_CSR_ID) == TSR_ID_MAGIC && tsr_sim_csr_read(s, TSR_CSR_CFG0) == (4u | (128u << 8) | (1u << 20)) &&
          tsr_sim_csr_read(s, TSR_CSR_CAPS) == 0 && tsr_sim_csr_read(s, TSR_CSR_PERF_BASE) == 0, "writes ignored");

    // LD 3 lines from ch1, NOP, FENCE: perf exact (no command fetch beats on ch0).
    expect_ok(s, tsr_enc_ld(1, 10, IO_BEAT, 3), "LD");
    expect_ok(s, tsr_enc_nop(), "NOP");
    expect_ok(s, tsr_enc_fence(), "FENCE");
    CHECK(tsr_sim_perf(s, TSR_PERF_RETIRED) == 3, "retired %llu", (unsigned long long)tsr_sim_perf(s, TSR_PERF_RETIRED));
    CHECK(tsr_sim_perf(s, TSR_PERF_CYCLES) == (1 + 64 + 3) + 1 + 1, "cycles %llu", (unsigned long long)tsr_sim_perf(s, TSR_PERF_CYCLES));
    CHECK(tsr_sim_perf(s, tsr_perf_ch_rd_beats(0)) == 0, "ch0 beats %llu", (unsigned long long)tsr_sim_perf(s, tsr_perf_ch_rd_beats(0)));
    CHECK(tsr_sim_perf(s, tsr_perf_ch_rd_beats(1)) == 3, "ch1 beats");
    CHECK(tsr_sim_perf(s, tsr_perf_ch_rd_beats(2)) == 0 && tsr_sim_perf(s, tsr_perf_ch_wait(4, 0)) == 0, "other counters");
    CHECK(tsr_sim_perf(s, TSR_PERF_MVM_CYCLES) == 0 && tsr_sim_perf(s, TSR_PERF_VU_CYCLES) == 0 && tsr_sim_perf(s, 4) == 0,
          "mvm/vu cycles, reserved slot 4");
    // PERF CSR LO/HI view
    CHECK(tsr_sim_csr_read(s, TSR_CSR_PERF_BASE + 8 * TSR_PERF_RETIRED) == 3 &&
          tsr_sim_csr_read(s, TSR_CSR_PERF_BASE + 8 * TSR_PERF_RETIRED + 4) == 0, "PERF CSR");
    CHECK(tsr_sim_csr_read(s, TSR_CSR_PERF_BASE + 8 * tsr_perf_ch_rd_beats(1)) == 3, "PERF ch1 CSR");
    // tsr_sim_op counts the op-level set only: no unit busy time in the extension slots.
    CHECK(tsr_sim_csr_read(s, TSR_CSR_PERF_BASE + 8 * TSR_PERF_DMA_CYCLES(4)) == 0, "DMA_CYCLES not counted by tsr_sim_op");
    // A failing op counts as retired (EX §6: any outcome) and adds no cycles.
    uint32_t info = 1;
    CHECK(tsr_sim_op(s, insn_raw(0xEEu, 0, 0, 0, 0, 0, 0, 0).w, &info) == TSR_ERR_ILLEGAL_OP && info == 0, "illegal opcode");
    CHECK(tsr_sim_op(s, insn_raw(0xEEu, 0, 0, 0, 0, 0, 0, 0).w, NULL) == TSR_ERR_ILLEGAL_OP, "NULL err_info");
    CHECK(tsr_sim_perf(s, TSR_PERF_RETIRED) == 5 && tsr_sim_perf(s, TSR_PERF_CYCLES) == 70, "error ops: retired, no cycles");
    tsr_sim_destroy(s);

    // CFG encodings of a non-default config
    tsr_config c2 = c;
    c2.num_ch = 3; c2.lanes = 64; c2.t_max = 8; c2.spm_bytes = 256 * 1024; c2.vl = 8; c2.nsreg = 12; c2.k_max = 4096;
    s = tsr_sim_create(&c2, NULL);
    CHECK(s != NULL, "create custom");
    CHECK(tsr_sim_csr_read(s, TSR_CSR_CFG0) == (3u | (64u << 8) | (8u << 20)), "CFG0 custom");
    CHECK(tsr_sim_csr_read(s, TSR_CSR_CFG1) == 4096u, "CFG1 custom");
    CHECK(tsr_sim_csr_read(s, TSR_CSR_CFG2) == (8u | (12u << 8) | (16u << 16)), "CFG2 custom");
    // SPM starts as 0xA5, s0..s15 as +0.0
    int all_a5 = 1;
    for (uint32_t i = 0; i < c2.spm_bytes; ++i) all_a5 &= tsr_sim_spm(s)[i] == 0xA5;
    CHECK(all_a5, "SPM initial 0xA5");
    for (uint32_t i = 0; i < 16; ++i) CHECK(f2u(tsr_sim_sreg(s)[i]) == 0, "s%u = +0.0", i);
    tsr_sim_destroy(s);

    // Unsupported configs are rejected.
    tsr_config bad;
    bad = c; bad.beat_bits = 256;  CHECK(tsr_sim_create(&bad, NULL) == NULL, "beat_bits");
    bad = c; bad.t_max = 9;        CHECK(tsr_sim_create(&bad, NULL) == NULL, "t_max 9");
    bad = c; bad.t_max = 0;        CHECK(tsr_sim_create(&bad, NULL) == NULL, "t_max 0");
    bad = c; bad.num_ch = 0;       CHECK(tsr_sim_create(&bad, NULL) == NULL, "num_ch 0");
    bad = c; bad.k_max = 1000;     CHECK(tsr_sim_create(&bad, NULL) == NULL, "k_max");
    bad = c; bad.spm_bytes = 100;  CHECK(tsr_sim_create(&bad, NULL) == NULL, "spm_bytes");
    // CAPS beyond the registered handlers: VMASK has none before M9 (VU and ATT do once
    // tsr_vu.c / tsr_at.c are linked, P1.4 / P1.5).
    CHECK(!(tsr_sim_caps_implemented() & TSR_CAPS_VMASK), "VMASK not implemented");
    bad = c; bad.caps = TSR_CAPS_VMASK; CHECK(tsr_sim_create(&bad, NULL) == NULL, "caps VMASK");
    bad = c; bad.caps = tsr_sim_caps_implemented();
    s = tsr_sim_create(&bad, NULL);
    CHECK(s != NULL, "caps = tsr_sim_caps_implemented() 0x%x accepted", bad.caps);
    tsr_sim_destroy(s);
    bad = c; bad.nsreg = 17;       CHECK(tsr_sim_create(&bad, NULL) == NULL, "nsreg");
    // NUM_CH <= 16: LD/ST name the channel in the 4-bit w0 sub field (spec 구성 파라미터).
    bad = c; bad.num_ch = TSR_NUM_CH_MAX + 1; CHECK(tsr_sim_create(&bad, NULL) == NULL, "num_ch 17");
    bad = c; bad.num_ch = 32;      CHECK(tsr_sim_create(&bad, NULL) == NULL, "num_ch 32");
    bad = c; bad.num_ch = 255;     CHECK(tsr_sim_create(&bad, NULL) == NULL, "num_ch 255");
}

// ---------------------------------------------------------------------------
static void test_ldst(void) {
    tsr_config c = cfg_u200(1);
    tsr_sim * s = tsr_sim_create(&c, NULL);
    rng_t r = { 42 };
    // fill IO areas of all channels with random bytes
    for (uint32_t ch = 0; ch < 4; ++ch) rng_bytes(&r, tsr_sim_dram(s, ch) + TSR_REGION_IO_OFF, 64 * 1024);
    ops_t p = {0};
    // LD 5 lines ch2 -> SPM 100; ST SPM 100..104 -> ch3 IO+512; a copy ch0 -> SPM 200 -> ch1.
    ops_add(&p, tsr_enc_ld(2, 100, IO_BEAT + 7, 5));
    ops_add(&p, tsr_enc_st(3, 100, IO_BEAT + 512, 5));
    ops_add(&p, tsr_enc_ld(0, 200, (uint32_t)(IO_BEAT + 600 - 16), 2));
    ops_add(&p, tsr_enc_st(1, 200, (uint32_t)(IO_BEAT + 605), 2));
    ops_add(&p, tsr_enc_ld(0, 300, IO_BEAT, 0));       // zero-length: no-op
    uint32_t at = 0;
    const uint32_t err = ops_run(s, &p, &at, NULL);
    CHECK(err == TSR_ERR_NONE, "ldst err %u at op %u", err, at);
    CHECK(memcmp(tsr_sim_spm(s) + 100 * 64, tsr_sim_dram(s, 2) + (IO_BEAT + 7) * 64, 5 * 64) == 0, "LD data");
    CHECK(memcmp(tsr_sim_dram(s, 3) + (IO_BEAT + 512) * 64, tsr_sim_dram(s, 2) + (IO_BEAT + 7) * 64, 5 * 64) == 0, "ST data");
    CHECK(memcmp(tsr_sim_spm(s) + 200 * 64, tsr_sim_dram(s, 0) + (IO_BEAT + 600 - 16) * 64, 2 * 64) == 0, "LD ch0");
    CHECK(memcmp(tsr_sim_dram(s, 1) + (IO_BEAT + 605) * 64, tsr_sim_dram(s, 0) + (IO_BEAT + 600 - 16) * 64, 2 * 64) == 0,
          "ST ch1");
    CHECK(tsr_sim_spm(s)[300 * 64] == 0xA5, "zero-length LD wrote nothing");

    // Exact-fit boundaries pass: SPM last line, DRAM last beat.
    const uint32_t spm_lines = c.spm_bytes / 64;
    const uint64_t lim = tsr_sim_dram_limit_beats(s);
    expect_ok(s, tsr_enc_ld(0, spm_lines - 2, (uint32_t)(lim - 2), 2), "LD exact fit");
    expect_ok(s, tsr_enc_st(3, spm_lines - 1, (uint32_t)(lim - 1), 1), "ST exact fit");
    expect_ok(s, tsr_enc_ld(0, spm_lines, (uint32_t)lim, 0), "zero length at the limits");
    ops_free(&p);
    tsr_sim_destroy(s);

    // ADDR_BEAT_W smaller than the channel: limit = 2^addr_beat_w beats.
    tsr_config c2 = c;
    c2.addr_beat_w = 20;
    s = tsr_sim_create(&c2, NULL);
    CHECK(tsr_sim_dram_limit_beats(s) == (1ull << 20), "limit = 2^ADDR_BEAT_W");
    expect_ok(s, tsr_enc_ld(0, 0, (1u << 20) - 1, 1), "LD last addressable beat");
    expect_insn_error(s, tsr_enc_ld(0, 0, (1u << 20) - 1, 2), TSR_ERR_DRAM_RANGE, (1u << 20) - 1, "LD past 2^ADDR_BEAT_W");
    tsr_sim_destroy(s);
    // CH_BYTES smaller than 2^ADDR_BEAT_W beats
    c2 = c;
    c2.ch_bytes = 1ull << 30;
    s = tsr_sim_create(&c2, NULL);
    CHECK(tsr_sim_dram_limit_beats(s) == (1ull << 24), "limit = CH_BYTES/64");
    expect_insn_error(s, tsr_enc_st(1, 0, (1u << 24), 1), TSR_ERR_DRAM_RANGE, 1u << 24, "ST past CH_BYTES");
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
        ops_t q = {0};
        for (uint32_t ch = 0; ch < TSR_NUM_CH_MAX; ++ch) ops_add(&q, tsr_enc_ld(ch, ch, IO_BEAT, 1));
        ops_add(&q, tsr_enc_st(15, 0, IO_BEAT + 1, 1));     // ch0's line -> ch15
        const uint32_t err16 = ops_run(s, &q, NULL, NULL);
        CHECK(err16 == TSR_ERR_NONE, "num_ch 16 LD/ST err %u", err16);
        for (uint32_t ch = 0; ch < TSR_NUM_CH_MAX; ++ch) {
            CHECK(tsr_sim_spm(s)[ch * 64] == 0x40 + ch, "num_ch 16: LD ch%u got 0x%02x", ch, tsr_sim_spm(s)[ch * 64]);
            CHECK(tsr_sim_perf(s, tsr_perf_ch_rd_beats(ch)) == 1, "num_ch 16: rd beats ch%u", ch);
        }
        CHECK(tsr_sim_dram(s, 15)[(IO_BEAT + 1) * 64] == 0x40, "num_ch 16: ST to ch15");
        CHECK(tsr_sim_dram(s, 0)[(IO_BEAT + 1) * 64] == 0, "num_ch 16: ST did not alias ch0");
        ops_free(&q);
        tsr_sim_destroy(s);
    }
}

// ---------------------------------------------------------------------------
// Reserved fields: flipping any bit an op does not use -> ILLEGAL_OP. That includes every bit
// of x16 (w0[31:16], the former register fields; EX §7.1).
// ---------------------------------------------------------------------------
typedef struct {
    const char * name;
    tsr_insn     base;        // valid op
    uint32_t     used[8];     // bits that belong to fields the op uses
} rsv_case;

static void test_reserved_fields(void) {
    tsr_config c = cfg_u200(8);
    tsr_sim * s = tsr_sim_create(&c, NULL);
    const uint32_t OPC = 0xFFu, SUB = 0xF00u;
    const uint32_t ALL = 0xFFFFFFFFu;
    rsv_case cases[] = {
        { "NOP",   tsr_enc_nop(),   { OPC, 0, 0, 0, 0, 0, 0, 0 } },
        { "FENCE", tsr_enc_fence(), { OPC, 0, 0, 0, 0, 0, 0, 0 } },
        { "LD",    tsr_enc_ld(0, 0, (uint32_t)IO_BEAT, 1), { OPC | SUB, ALL, ALL, ALL, 0, 0, 0, 0 } },
        { "ST",    tsr_enc_st(0, 0, (uint32_t)IO_BEAT, 1), { OPC | SUB, ALL, ALL, ALL, 0, 0, 0, 0 } },
        { "QACT",  tsr_enc_qact(TSR_QA_A32, 0, 100, 32, 1, 0), { OPC | SUB, ALL, ALL, ALL, ALL, ALL, 0, 0 } },
        { "MVM",   tsr_enc_mvm(TSR_FMT_Q8_0, 0, 0, 100, (uint32_t)(TSR_REGION_WEIGHT_OFF / 64), 4, 32, 1, 0),
                   { OPC | SUB | (1u << 12), ALL, ALL, ALL, ALL, 0x000FFFFFu, ALL, 0 } },
    };
    for (size_t k = 0; k < sizeof(cases) / sizeof(cases[0]); ++k) {
        const rsv_case * rc = &cases[k];
        CHECK(tsr_w0_x16(rc->base.w[0]) == 0, "%s encodes x16 = 0", rc->name);
        expect_ok(s, rc->base, rc->name);
        int n_flips = 0;
        for (uint32_t wi = 0; wi < 8; ++wi) {
            for (uint32_t bit = 0; bit < 32; ++bit) {
                if (rc->used[wi] & (1u << bit)) continue;
                tsr_insn bad = rc->base;
                bad.w[wi] ^= 1u << bit;
                char what[96];
                snprintf(what, sizeof(what), "%s reserved w%u bit %u", rc->name, wi, bit);
                expect_insn_error(s, bad, TSR_ERR_ILLEGAL_OP, 0, what);
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
        if (sub < 4) expect_ok(s, ld, what); else expect_insn_error(s, ld, TSR_ERR_ILLEGAL_OP, 0, what);
        snprintf(what, sizeof(what), "ST ch %u", sub);
        if (sub < 4) expect_ok(s, st, what); else expect_insn_error(s, st, TSR_ERR_ILLEGAL_OP, 0, what);
        snprintf(what, sizeof(what), "QACT kind %u", sub);
        if (sub < 2) expect_ok(s, qa, what); else expect_insn_error(s, qa, TSR_ERR_ILLEGAL_OP, 0, what);
        snprintf(what, sizeof(what), "MVM fmt %u", sub);
        if (sub < 2) expect_ok(s, mv, what); else expect_insn_error(s, mv, TSR_ERR_BAD_FMT, 0, what);
    }
    // Every opcode outside {NOP, LD, ST, QACT, MVM, FENCE} is ILLEGAL_OP (this simulator has CAPS = 0).
    const uint32_t impl[] = { TSR_OP_NOP, TSR_OP_LD, TSR_OP_ST, TSR_OP_QACT, TSR_OP_MVM, TSR_OP_FENCE };
    for (uint32_t op = 0; op < 256; ++op) {
        int is_impl = 0;
        for (size_t i = 0; i < sizeof(impl) / sizeof(impl[0]); ++i) is_impl |= (op == impl[i]);
        if (is_impl) continue;
        char what[32];
        snprintf(what, sizeof(what), "opcode 0x%02x", op);
        expect_insn_error(s, insn_raw(op, 0, 0, 0, 0, 0, 0, 0), TSR_ERR_ILLEGAL_OP, 0, what);
    }
    // The reserved END, LI and ADDI (0x01-0x03) with the fields they used to have, a
    // retired MULI, and a later-milestone SOP MOVI: all ILLEGAL_OP, and none changes state.
    expect_insn_error(s, insn_raw(0x00000001u, 0, 0, 0, 0, 0, 0, 0), TSR_ERR_ILLEGAL_OP, 0, "END");
    expect_insn_error(s, insn_raw(0x10000002u, 5, 0, 0, 0, 0, 0, 0), TSR_ERR_ILLEGAL_OP, 0, "LI R1 = 5");
    expect_insn_error(s, insn_raw(0x10020003u, 5, 0, 0, 0, 0, 0, 0), TSR_ERR_ILLEGAL_OP, 0, "ADDI R1 = R2 + 5");
    expect_insn_error(s, insn_raw(0x10020004u, 3, 0, 0, 0, 0, 0, 0), TSR_ERR_ILLEGAL_OP, 0, "MULI");
    expect_insn_error(s, insn_raw(tsr_w0(TSR_OP_SOP, TSR_SOP_SUB_MOVI, 0, 0), 1, 0x3F800000u, 0, 0, 0, 0, 0),
                      TSR_ERR_ILLEGAL_OP, 0, "SOP");
    // Register-relative operands (a base register in x16) are refused, not reinterpreted.
    tsr_insn rel = tsr_enc_ld(0, 0, (uint32_t)IO_BEAT, 1);
    rel.w[0] |= 8u << 20;                                   // would have meant A2 = R8 + w2
    expect_insn_error(s, rel, TSR_ERR_ILLEGAL_OP, 0, "LD with a base register");
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

    // LD / ST
    expect_insn_error(s, tsr_enc_ld(0, SL - 1, (uint32_t)IO_BEAT, 2), TSR_ERR_SPM_RANGE, SL - 1, "LD SPM");
    expect_insn_error(s, tsr_enc_ld(1, 0, LIM - 1, 2), TSR_ERR_DRAM_RANGE, LIM - 1, "LD DRAM");
    expect_insn_error(s, tsr_enc_st(2, SL, (uint32_t)IO_BEAT, 1), TSR_ERR_SPM_RANGE, SL, "ST SPM");
    expect_insn_error(s, tsr_enc_st(3, 0, 0xFFFFFFFFu, 1), TSR_ERR_DRAM_RANGE, 0xFFFFFFFFu, "ST DRAM");
    expect_insn_error(s, tsr_enc_st(3, 0, (uint32_t)IO_BEAT, 0xFFFFFFFFu), TSR_ERR_SPM_RANGE, 0, "ST huge n");
    expect_insn_error(s, tsr_enc_ld(0, SL + 1, (uint32_t)LIM, 0), TSR_ERR_SPM_RANGE, SL + 1, "LD n=0 start past SPM end");
    // priorities: ILLEGAL (ch) > SPM > DRAM
    expect_insn_error(s, tsr_enc_ld(4, SL, LIM, 1), TSR_ERR_ILLEGAL_OP, 0, "LD ch+SPM+DRAM");
    expect_insn_error(s, tsr_enc_ld(0, SL, LIM, 1), TSR_ERR_SPM_RANGE, SL, "LD SPM+DRAM");
    // ERR_INFO = the operand's start address (the immediate itself)
    expect_insn_error(s, tsr_enc_st(0, 0, LIM - 1, 2), TSR_ERR_DRAM_RANGE, LIM - 1, "ST DRAM last beat + 1");
    expect_ok(s, tsr_enc_st(0, 0, LIM - 2, 2), "ST exact fit");
    expect_insn_error(s, tsr_enc_ld(0, 0xFFFFFFFFu, (uint32_t)IO_BEAT, 1), TSR_ERR_SPM_RANGE, 0xFFFFFFFFu, "LD SPM 0xFFFFFFFF");

    // QACT
    expect_insn_error(s, tsr_enc_qact(0, 0, 100, 0, 1, 0), TSR_ERR_K_RANGE, 0, "QACT K=0");
    expect_insn_error(s, tsr_enc_qact(0, 0, 100, 48, 1, 0), TSR_ERR_K_RANGE, 0, "QACT A32 K%32");
    expect_insn_error(s, tsr_enc_qact(1, 0, 100, 128, 1, 0), TSR_ERR_K_RANGE, 0, "QACT A256 K%256");
    expect_insn_error(s, tsr_enc_qact(0, 0, 100, 32768 + 32, 1, 0), TSR_ERR_K_RANGE, 0, "QACT K>K_MAX");
    expect_ok(s, tsr_enc_qact(1, 0, 1000, 32768, 2, 0), "QACT K=K_MAX T=T_MAX");
    expect_insn_error(s, tsr_enc_qact(0, 0, 100, 32, 0, 0), TSR_ERR_T_RANGE, 0, "QACT T=0");
    expect_insn_error(s, tsr_enc_qact(0, 0, 100, 32, 3, 0), TSR_ERR_T_RANGE, 0, "QACT T>T_MAX");
    expect_insn_error(s, tsr_enc_qact(0, SL - 1, 0, 32, 1, 0), TSR_ERR_SPM_RANGE, SL - 1, "QACT dst");
    expect_insn_error(s, tsr_enc_qact(0, 0, SL - 1, 32, 1, 0), TSR_ERR_SPM_RANGE, SL - 1, "QACT src");
    expect_insn_error(s, tsr_enc_qact(0, 0, SL - 3, 32, 2, 3), TSR_ERR_SPM_RANGE, SL - 3, "QACT src stride");
    expect_ok(s, tsr_enc_qact(0, 0, SL - 4, 32, 2, 2), "QACT src stride exact fit");
    expect_ok(s, tsr_enc_qact(0, SL - 4, 0, 32, 2, 0), "QACT dst exact fit (2 x 2 lines)");
    // priorities
    expect_insn_error(s, tsr_enc_qact(5, SL, SL, 33, 0, 0), TSR_ERR_ILLEGAL_OP, 0, "QACT kind+K+T+SPM");
    expect_insn_error(s, tsr_enc_qact(0, SL, SL, 33, 0, 0), TSR_ERR_K_RANGE, 0, "QACT K+T+SPM");
    expect_insn_error(s, tsr_enc_qact(0, SL, SL, 32, 0, 0), TSR_ERR_T_RANGE, 0, "QACT T+SPM");
    expect_insn_error(s, tsr_enc_qact(0, SL, SL + 7, 32, 1, 0), TSR_ERR_SPM_RANGE, SL, "QACT dst before src");

    // MVM: N = 9, K = 64 (Q8_0) / 256 (Q4_K), T = 2
    const uint32_t rb8 = tsr_row_beats(TSR_FMT_Q8_0, 64);
    const uint32_t R9 = 3;   // ceil(9/4)
    expect_ok(s, tsr_enc_mvm(0, 0, 0, 100, WOFF, 9, 64, 2, 0), "MVM ok");
    expect_insn_error(s, tsr_enc_mvm(0, 2, 0, 100, WOFF, 9, 64, 2, 0), TSR_ERR_ILLEGAL_OP, 0, "MVM flag bit1");
    expect_insn_error(s, tsr_enc_mvm(3, 0, 0, 100, WOFF, 9, 64, 2, 0), TSR_ERR_BAD_FMT, 0, "MVM fmt 3");
    expect_insn_error(s, tsr_enc_mvm(0, 0, 0, 100, WOFF, 9, 0, 2, 0), TSR_ERR_K_RANGE, 0, "MVM K=0");
    expect_insn_error(s, tsr_enc_mvm(0, 0, 0, 100, WOFF, 9, 48, 2, 0), TSR_ERR_K_RANGE, 0, "MVM Q8_0 K%32");
    expect_insn_error(s, tsr_enc_mvm(1, 0, 0, 100, WOFF, 9, 64, 2, 0), TSR_ERR_K_RANGE, 0, "MVM Q4_K K%256");
    expect_insn_error(s, tsr_enc_mvm(1, 0, 0, 100, WOFF, 9, 32768 + 256, 2, 0), TSR_ERR_K_RANGE, 0, "MVM K>K_MAX");
    expect_insn_error(s, tsr_enc_mvm(0, 0, 0, 100, WOFF, 9, 64, 0, 0), TSR_ERR_T_RANGE, 0, "MVM T=0");
    expect_insn_error(s, tsr_enc_mvm(0, 0, 0, 100, WOFF, 9, 64, 3, 0), TSR_ERR_T_RANGE, 0, "MVM T>T_MAX");
    expect_insn_error(s, tsr_enc_mvm(0, 0, 0, 100, WOFF, 9, 64, 15, 0), TSR_ERR_T_RANGE, 0, "MVM T=15");
    // dst: 2 tokens x ceil(36/64)=1 line, stride 0 -> 1 line: needs dst + 2 <= SL
    expect_insn_error(s, tsr_enc_mvm(0, 0, SL - 1, 100, WOFF, 9, 64, 2, 0), TSR_ERR_SPM_RANGE, SL - 1, "MVM dst");
    expect_ok(s, tsr_enc_mvm(0, 0, SL - 2, 100, WOFF, 9, 64, 2, 0), "MVM dst exact fit");
    expect_insn_error(s, tsr_enc_mvm(0, 0, SL - 10, 100, WOFF, 9, 64, 2, 10), TSR_ERR_SPM_RANGE, SL - 10, "MVM dst stride");
    expect_ok(s, tsr_enc_mvm(0, 0, SL - 10, 100, WOFF, 9, 64, 2, 9), "MVM dst stride exact fit");
    const uint32_t qa8 = tsr_qa_lines(TSR_QA_A32, 64);
    expect_insn_error(s, tsr_enc_mvm(0, 0, 0, SL - 2 * qa8 + 1, WOFF, 9, 64, 2, 0), TSR_ERR_SPM_RANGE,
                      SL - 2 * qa8 + 1, "MVM x");
    expect_ok(s, tsr_enc_mvm(0, 0, 0, SL - 2 * qa8, WOFF, 9, 64, 2, 0), "MVM x exact fit");
    // DRAM: channel 0 holds R = 3 rows -> W_OFF + 3*ROW_BEATS <= LIM
    expect_insn_error(s, tsr_enc_mvm(0, 0, 0, 100, LIM - R9 * rb8 + 1, 9, 64, 2, 0), TSR_ERR_DRAM_RANGE,
                      LIM - R9 * rb8 + 1, "MVM W_OFF");
    expect_ok(s, tsr_enc_mvm(0, 0, 0, 100, LIM - R9 * rb8, 9, 64, 2, 0), "MVM W_OFF exact fit");
    expect_insn_error(s, tsr_enc_mvm(0, 0, 0, 100, LIM + 1, 0, 64, 1, 0), TSR_ERR_DRAM_RANGE, LIM + 1, "MVM N=0 W_OFF past end");
    expect_insn_error(s, tsr_enc_mvm(0, 0, 0, 100, WOFF, 0xFFFFFFFFu, 64, 1, 0), TSR_ERR_SPM_RANGE, 0, "MVM huge N");
    // priorities
    expect_insn_error(s, tsr_enc_mvm(7, 8, SL, SL, LIM, 9, 33, 0, 0), TSR_ERR_ILLEGAL_OP, 0, "MVM all");
    expect_insn_error(s, tsr_enc_mvm(7, 0, SL, SL, LIM, 9, 33, 0, 0), TSR_ERR_BAD_FMT, 0, "MVM fmt+K+T+SPM+DRAM");
    expect_insn_error(s, tsr_enc_mvm(0, 0, SL, SL, LIM, 9, 33, 0, 0), TSR_ERR_K_RANGE, 0, "MVM K+T+SPM+DRAM");
    expect_insn_error(s, tsr_enc_mvm(0, 0, SL, SL, LIM, 9, 32, 0, 0), TSR_ERR_T_RANGE, 0, "MVM T+SPM+DRAM");
    expect_insn_error(s, tsr_enc_mvm(0, 0, SL, SL + 1, LIM, 9, 32, 1, 0), TSR_ERR_SPM_RANGE, SL, "MVM dst first");
    expect_insn_error(s, tsr_enc_mvm(0, 0, 0, SL + 1, LIM, 9, 32, 1, 0), TSR_ERR_SPM_RANGE, SL + 1, "MVM x before DRAM");
    expect_insn_error(s, tsr_enc_mvm(0, 0, 0, 100, LIM, 9, 64, 1, 0), TSR_ERR_DRAM_RANGE, LIM, "MVM W_OFF = limit");
    expect_insn_error(s, tsr_enc_mvm(0, 1, SL, 100, WOFF, 9, 64, 1, 0), TSR_ERR_SPM_RANGE, SL, "MVM dst (ACC)");
    expect_ok(s, tsr_enc_mvm(0, 1, SL - 1, 100, WOFF, 9, 64, 1, 0), "MVM dst ACC exact fit");
    tsr_sim_destroy(s);
}

int main(void) {
    test_csr();
    test_ldst();
    test_reserved_fields();
    test_errors();
    snap_free(&g_snap);
    return test_finish("sim_basic");
}
