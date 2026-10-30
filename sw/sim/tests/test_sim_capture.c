// test_sim_capture.c — access capture semantics of the simulator (the part of the trace dumps that
// lives in tsr_sim; the trace format builds on it, docs/design/06-software-verification.md §5.5): init =
// DRAM read before it was written, copied at first read; expect = DRAM written, taken at the
// end; merging of adjacent and overlapping ranges, sorting by (channel, offset), empty
// captures, and the capture of a large MVM. Built a second time with UBSan (sim_capture_ubsan).
#include "test_sim_util.h"

#include <time.h>

static const uint64_t IO = TSR_REGION_IO_OFF / 64;
static const uint64_t WOFF = TSR_REGION_WEIGHT_OFF / 64;

static int region_is(const tsr_region * r, uint32_t ch, uint64_t beat, uint64_t beats) {
    return r->ch == ch && r->byte_off == beat * 64 && r->size == beats * 64;
}

static uint8_t * dup_bytes(const void * p, size_t n) {
    uint8_t * d = (uint8_t *)malloc(n ? n : 1);
    memcpy(d, p, n);
    return d;
}

static void test_capture(void) {
    tsr_config cfg = cfg_u200(1);
    tsr_sim * s = tsr_sim_create(&cfg, NULL);
    rng_t r = { 11 };
    for (uint32_t c = 0; c < 4; ++c) rng_bytes(&r, tsr_sim_dram(s, c) + IO * 64, 1000 * 64);
    rng_bytes(&r, tsr_sim_dram(s, 0) + 99 * 64, 64);
    // ch0 IO+10.. holds 64 f32 activations (finite)
    float x[64];
    gen_f32(&r, x, 64);
    put_f32_le(tsr_sim_dram(s, 0) + (IO + 10) * 64, x, 64);
    // Q8_0 N=6 K=64 -> R=2: ch0 rows 0-1, ch1 2-3, ch2 4-5, ch3 none
    const uint32_t N = 6, K = 64, rb = tsr_row_beats(TSR_FMT_Q8_0, K);
    uint8_t blocks[6 * 2 * 34];
    for (uint32_t n = 0; n < N; ++n) gen_row_blocks(&r, TSR_FMT_Q8_0, K, blocks + n * 68);
    put_weights(s, TSR_FMT_Q8_0, N, K, WOFF, blocks);

    ops_t p = {0};
    ops_add(&p, tsr_enc_ld(0, 0, (uint32_t)(IO + 10), 4));      // read [IO+10, IO+14)
    ops_add(&p, tsr_enc_st(0, 0, (uint32_t)(IO + 200), 2));     // write [IO+200, IO+202)
    ops_add(&p, tsr_enc_ld(0, 8, (uint32_t)(IO + 200), 2));     // read of written data: not init
    ops_add(&p, tsr_enc_ld(0, 16, (uint32_t)(IO + 12), 4));     // [IO+12, IO+16): only +14..16 new
    ops_add(&p, tsr_enc_ld(2, 24, (uint32_t)(IO + 50), 1));     // ch2 read ...
    ops_add(&p, tsr_enc_st(2, 0, (uint32_t)(IO + 50), 1));      // ... then written: init + expect
    ops_add(&p, tsr_enc_qact(TSR_QA_A32, 40, 0, K, 1, 0));
    ops_add(&p, tsr_enc_mvm(TSR_FMT_Q8_0, 0, 60, 40, (uint32_t)WOFF, N, K, 1, 0));
    ops_add(&p, tsr_enc_st(1, 60, (uint32_t)(IO + 300), 1));
    ops_add(&p, tsr_enc_st(0, 60, (uint32_t)(IO + 202), 1));    // right-adjacent to [IO+200, IO+202)
    ops_add(&p, tsr_enc_st(0, 60, (uint32_t)(IO + 199), 1));    // left-adjacent -> [IO+199, IO+203)
    ops_add(&p, tsr_enc_ld(0, 30, 99, 1));                       // a single beat below IO
    ops_add(&p, tsr_enc_nop());
    uint8_t * pre_ch2 = dup_bytes(tsr_sim_dram(s, 2) + (IO + 50) * 64, 64);
    uint8_t * pre_ch0 = dup_bytes(tsr_sim_dram(s, 0) + (IO + 10) * 64, 6 * 64);

    tsr_sim_capture_begin(s);
    uint32_t at = 0;
    const uint32_t err = ops_run(s, &p, &at, NULL);
    tsr_capture cap;
    tsr_sim_capture_end(s, &cap);
    CHECK(err == TSR_ERR_NONE, "capture ops: err %u at op %u", err, at);

    // init: ch0 beat 99, IO [10,16), W [0,2rb); ch1 W; ch2 IO+50, W. Sorted by (ch, off).
    CHECK(cap.n_init == 6, "n_init %zu", cap.n_init);
    if (cap.n_init == 6) {
        CHECK(region_is(&cap.init[0], 0, 99, 1), "init0 single beat: ch %u off %llu size %llu", cap.init[0].ch,
              (unsigned long long)cap.init[0].byte_off, (unsigned long long)cap.init[0].size);
        CHECK(region_is(&cap.init[1], 0, IO + 10, 6), "init1 merged IO read");
        CHECK(region_is(&cap.init[2], 0, WOFF, 2 * rb), "init2 ch0 weights");
        CHECK(region_is(&cap.init[3], 1, WOFF, 2 * rb), "init3 ch1 weights");
        CHECK(region_is(&cap.init[4], 2, IO + 50, 1), "init4 ch2 read-then-written");
        CHECK(region_is(&cap.init[5], 2, WOFF, 2 * rb), "init5 ch2 weights");
        CHECK(memcmp(cap.init[0].data, tsr_sim_dram(s, 0) + 99 * 64, 64) == 0, "beat 99 bytes");
        CHECK(memcmp(cap.init[1].data, pre_ch0, 6 * 64) == 0, "IO pre-content");
        CHECK(memcmp(cap.init[4].data, pre_ch2, 64) == 0, "ch2 pre-content (before the ST)");
        CHECK(memcmp(cap.init[5].data, tsr_sim_dram(s, 2) + WOFF * 64, 2 * rb * 64) == 0, "ch2 weights");
    }
    CHECK(cap.n_expect == 3, "n_expect %zu", cap.n_expect);
    if (cap.n_expect == 3) {
        CHECK(region_is(&cap.expect[0], 0, IO + 199, 4), "expect0 merged adjacent writes");
        CHECK(region_is(&cap.expect[1], 1, IO + 300, 1), "expect1");
        CHECK(region_is(&cap.expect[2], 2, IO + 50, 1), "expect2");
        CHECK(memcmp(cap.expect[0].data, tsr_sim_dram(s, 0) + (IO + 199) * 64, 4 * 64) == 0, "expect0 post-content");
        CHECK(memcmp(cap.expect[2].data, tsr_sim_dram(s, 2) + (IO + 50) * 64, 64) == 0, "expect post-content");
        CHECK(memcmp(cap.expect[2].data, pre_ch2, 64) != 0, "ch2 beat really changed");
    }
    tsr_capture_free(&cap);
    CHECK(cap.n_init == 0 && cap.n_expect == 0 && cap.init == NULL && cap.expect == NULL, "capture_free resets");

    // Failing ops capture nothing (they read and write nothing).
    tsr_sim_capture_begin(s);
    CHECK(tsr_sim_op(s, tsr_enc_ld(0, 0, (uint32_t)tsr_sim_dram_limit_beats(s), 1).w, NULL) == TSR_ERR_DRAM_RANGE, "fault");
    CHECK(tsr_sim_op(s, insn_raw(0x00000002u, 0, 0, 0, 0, 0, 0, 0).w, NULL) == TSR_ERR_ILLEGAL_OP, "reserved opcode");
    tsr_sim_capture_end(s, &cap);
    CHECK(cap.n_init == 0 && cap.n_expect == 0, "failing ops: empty capture");
    tsr_capture_free(&cap);

    // A capture that was never begun is empty; begin/end without ops is empty; ops outside a
    // capture are not recorded by the next one.
    tsr_capture empty;
    tsr_sim_capture_end(s, &empty);
    CHECK(empty.n_init == 0 && empty.n_expect == 0, "capture_end without begin");
    tsr_sim_capture_begin(s);
    tsr_sim_capture_end(s, &empty);
    CHECK(empty.n_init == 0 && empty.n_expect == 0, "empty capture");
    CHECK(ops_run(s, &p, NULL, NULL) == TSR_ERR_NONE, "ops outside a capture");
    tsr_sim_capture_begin(s);
    CHECK(tsr_sim_op(s, tsr_enc_ld(3, 0, (uint32_t)(IO + 400), 2).w, NULL) == TSR_ERR_NONE, "one LD");
    tsr_sim_capture_end(s, &cap);
    CHECK(cap.n_init == 1 && cap.n_expect == 0 && region_is(&cap.init[0], 3, IO + 400, 2), "only the captured LD");
    tsr_capture_free(&cap);

    free(pre_ch0);
    free(pre_ch2);
    ops_free(&p);
    tsr_sim_destroy(s);
}

static void test_big_capture(void) {
    tsr_config cfg = cfg_u200(1);
    tsr_sim * s = tsr_sim_create(&cfg, NULL);
    tsr_sim_set_threads(s, 4);
    const uint32_t fmt = TSR_FMT_Q4_K, N = 8192, K = 32768;
    const uint64_t rb = tsr_row_beats(fmt, K), per_ch = tsr_tensor_ch_beats(fmt, N, K, 4);
    rng_t r = { 8 };
    // Any bytes are a valid Q4_K row image; fill the four channel ranges directly.
    for (uint32_t c = 0; c < 4; ++c) {
        uint64_t * w = (uint64_t *)(void *)(tsr_sim_dram(s, c) + WOFF * 64);
        for (uint64_t i = 0; i < per_ch * 8; ++i) w[i] = rng_u64(&r) & 0xFBFFFBFFFBFFFBFFull;   // keep fp16 finite-ish
    }
    float * x = (float *)malloc(sizeof(float) * K);
    gen_f32(&r, x, K);
    put_f32_le(tsr_sim_spm(s), x, K);
    ops_t p = {0};
    ops_add(&p, tsr_enc_qact(TSR_QA_A256, 3000, 0, K, 1, 0));
    ops_add(&p, tsr_enc_mvm(fmt, 0, 4000, 3000, (uint32_t)WOFF, N, K, 1, 0));
    ops_add(&p, tsr_enc_st(3, 4000, (uint32_t)IO, N * 4 / 64));
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    tsr_sim_capture_begin(s);
    const uint32_t err = ops_run(s, &p, NULL, NULL);
    tsr_capture cap;
    tsr_sim_capture_end(s, &cap);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    const double secs = (double)(t1.tv_sec - t0.tv_sec) + 1e-9 * (double)(t1.tv_nsec - t0.tv_nsec);
    printf("big capture: %.1f MB of weights, ops + capture %.2f s\n", (double)(per_ch * 4 * 64) / 1e6, secs);
    CHECK(err == TSR_ERR_NONE, "big MVM err %u", err);
    CHECK(cap.n_init == 4, "big: n_init %zu (4 weight ranges)", cap.n_init);
    for (uint32_t c = 0; c < 4 && cap.n_init == 4; ++c) {
        const tsr_region * g = &cap.init[c];
        CHECK(region_is(g, c, WOFF, N / 4 * rb), "big: ch%u weight range", c);
        CHECK(memcmp(g->data, tsr_sim_dram(s, c) + WOFF * 64, (size_t)g->size) == 0, "big: ch%u data", c);
    }
    CHECK(cap.n_expect == 1 && region_is(&cap.expect[0], 3, IO, N * 4 / 64), "big: expect");
    tsr_capture_free(&cap);
    free(x);
    ops_free(&p);
    tsr_sim_destroy(s);
}

int main(void) {
    test_capture();
    test_big_capture();
    return test_finish("sim_capture");
}
