// test_sim_mvm.c — QACT and MVM through complete programs (LD -> QACT -> MVM -> ST) checked
// against a direct tsr_ref computation that never looks at where the simulator put things:
// expected rows are repacked from the ggml blocks independently, expected QA images come from
// tsr_qact on the host floats. Covers both fmts, partial chunks, N < C, odd N, N not a multiple
// of C, T > 1 (T_MAX = 8), ACC, dst_stride / src_stride 0 and explicit, register-relative
// addressing (LI/ADDI/ARG registers), 1..4 channels, zeroed tails, untouched stride gaps,
// per-channel beat counters, MVM cycle model, and bit-identical results for 1 vs 4 threads.
#include "test_sim_util.h"

#define DONE TSR_STATUS_DONE

typedef struct {
    uint32_t num_ch;
    uint32_t fmt, N, K, T;
    int      acc;
    uint32_t x_slack;      // extra lines between f32 source tokens (0 -> src_stride may be 0)
    uint32_t y_slack;      // extra lines between dst tokens (0 -> dst_stride may be 0)
    int      zero_strides; // encode stride 0 when the slack is 0
    int      regrel;       // register-relative addressing for every address field
    int      special;      // special activation values (zero blocks, subnormals, Inf, NaN)
    unsigned threads;
    uint64_t seed;
} mvm_case;

// Register file model used to build register-relative encodings.
typedef struct { uint32_t R[16]; } regs_t;

// Rewrites the address fields w1..w(nf) of insn (absolute targets) as base register + imm.
static void relocate(tsr_insn * insn, int nf, const regs_t * rg, rng_t * r) {
    static const uint32_t bases[] = { 0, 1, 2, 3, 4, 8, 9, 15 };
    uint32_t b[3] = { 0, 0, 0 };
    for (int f = 0; f < nf; ++f) {
        b[f] = bases[rng_below(r, sizeof(bases) / sizeof(bases[0]))];
        insn->w[1 + f] = insn->w[1 + f] - rg->R[b[f]];
    }
    tsr_insn_set_base(insn, b[0], b[1], b[2]);
}

static void gen_x(rng_t * r, float * x, uint32_t K, int special) {
    gen_f32(r, x, K);
    if (!special) return;
    for (uint32_t b = 0; b < K / 32; ++b) {
        float * xb = x + 32 * b;
        switch (rng_below(r, 12)) {
        case 0: memset(xb, 0, 32 * sizeof(float)); break;                                  // zero block
        case 1: xb[rng_below(r, 32)] = u2f(0x00012345u | (rng_u32(r) & 0x80000000u)); break; // subnormal
        case 2: { const float m = xb[0]; xb[5] = -m; xb[9] = m; } break;                 // ties for max
        case 3: xb[rng_below(r, 32)] = INFINITY; break;
        case 4: xb[rng_below(r, 32)] = u2f(0x7FC01234u); break;                           // NaN
        case 5: for (int i = 0; i < 32; ++i) xb[i] = u2f(0x00000100u * (uint32_t)(i + 1)); break; // all subnormal
        default: break;
        }
    }
}

// Runs one case; returns a hash of the output (for thread-count comparisons).
static uint64_t run_case(const mvm_case * mc) {
    rng_t r = { mc->seed };
    tsr_config cfg = cfg_u200(8);
    cfg.num_ch = mc->num_ch;
    tsr_sim * s = tsr_sim_create(&cfg, NULL);
    CHECK(s != NULL, "create");
    tsr_sim_set_threads(s, mc->threads);
    const uint32_t fmt = mc->fmt, N = mc->N, K = mc->K, T = mc->T, C = mc->num_ch;
    const uint32_t kind = tsr_qa_kind_for_fmt(fmt);
    const uint32_t x_lines = K * 4 / 64, qa_lines = tsr_qa_lines(kind, K);
    const uint32_t y_lines = tsr_ceil_div_u32(4 * N, 64);
    const uint32_t xs = x_lines + mc->x_slack, ys = y_lines + mc->y_slack;
    const uint32_t row_beats = tsr_row_beats(fmt, K);
    char tag[160];
    snprintf(tag, sizeof(tag), "C=%u fmt=%u N=%u K=%u T=%u acc=%d xs=%u ys=%u z=%d rr=%d sp=%d th=%u", C, fmt, N, K, T,
             mc->acc, mc->x_slack, mc->y_slack, mc->zero_strides, mc->regrel, mc->special, mc->threads);

    // Host data
    const size_t row_bytes = (size_t)(K / tsr_fmt_block_size(fmt)) * tsr_fmt_ggml_block_bytes(fmt);
    uint8_t * blocks = (uint8_t *)malloc(row_bytes * N + 1);
    for (uint32_t n = 0; n < N; ++n) gen_row_blocks(&r, fmt, K, blocks + row_bytes * n);
    const uint64_t w_off = TSR_REGION_WEIGHT_OFF / 64 + rng_below(&r, 100000);
    put_weights(s, fmt, N, K, w_off, blocks);

    float * x = (float *)malloc(sizeof(float) * K * T);
    for (uint32_t t = 0; t < T; ++t) gen_x(&r, x + (size_t)t * K, K, mc->special);
    const uint64_t x_beat = TSR_REGION_IO_OFF / 64 + rng_below(&r, 1000);
    for (uint32_t t = 0; t < T; ++t) put_f32_le(tsr_sim_dram(s, 0) + (x_beat + (uint64_t)t * xs) * 64, x + (size_t)t * K, K);
    const uint32_t y_span = (T - 1) * ys + y_lines;
    const uint64_t yo_beat = x_beat + (uint64_t)T * xs + 7;
    uint8_t * y_old = (uint8_t *)malloc((size_t)y_span * 64 + 64);
    rng_bytes(&r, y_old, (size_t)y_span * 64);
    for (uint32_t i = 0; i < y_span * 16; ++i) {           // finite old values (some are -0/+0)
        float v = (float)rng_normal(&r);
        if (rng_below(&r, 50) == 0) v = (rng_u32(&r) & 1) ? 0.0f : -0.0f;
        put_f32_le(y_old + 4 * i, &v, 1);
    }
    memcpy(tsr_sim_dram(s, 0) + yo_beat * 64, y_old, (size_t)y_span * 64);

    // SPM layout (line addresses), with random gaps
    const uint32_t spm_x = rng_below(&r, 64);
    const uint32_t spm_qa = spm_x + (T - 1) * xs + x_lines + rng_below(&r, 4);
    const uint32_t spm_y = spm_qa + T * qa_lines + rng_below(&r, 4);
    CHECK(spm_y + y_span <= cfg.spm_bytes / 64, "%s: SPM layout fits", tag);
    // Pre-fill the dst span with a pattern so untouched gap lines can be checked.
    memset(tsr_sim_spm(s) + (size_t)spm_y * 64, 0x5A, (size_t)y_span * 64);

    // Program
    regs_t rg;
    memset(&rg, 0, sizeof(rg));
    uint32_t args[8];
    for (int i = 0; i < 8; ++i) { args[i] = rng_u32(&r); rg.R[8 + i] = args[i]; }
    prog_t p = {0};
    if (mc->regrel) {
        rg.R[1] = rng_u32(&r);
        prog_add(&p, tsr_enc_li(1, rg.R[1]));
        rg.R[2] = rng_below(&r, 1u << 20);
        prog_add(&p, tsr_enc_li(2, rg.R[2]));
        const uint32_t d3 = rng_u32(&r);
        rg.R[3] = rg.R[2] + d3;
        prog_add(&p, tsr_enc_addi(3, 2, d3));
        const uint32_t d4 = rng_u32(&r);
        rg.R[4] = rg.R[8] + d4;
        prog_add(&p, tsr_enc_addi(4, 8, d4));
    }
    const uint32_t st_ch = rng_below(&r, C);
    const uint64_t y_beat = TSR_REGION_IO_OFF / 64 + 50000 + rng_below(&r, 1000);
    tsr_insn i_ldx = tsr_enc_ld(0, spm_x, (uint32_t)x_beat, (T - 1) * xs + x_lines);
    tsr_insn i_qact = tsr_enc_qact(kind, spm_qa, spm_x, K, T, (mc->x_slack == 0 && mc->zero_strides) ? 0 : xs);
    tsr_insn i_ldy = tsr_enc_ld(0, spm_y, (uint32_t)yo_beat, y_span);
    tsr_insn i_mvm = tsr_enc_mvm(fmt, mc->acc ? TSR_MVM_FLAG_ACC : 0, spm_y, spm_qa, (uint32_t)w_off, N, K, T,
                                 (mc->y_slack == 0 && mc->zero_strides) ? 0 : ys);
    tsr_insn i_st = tsr_enc_st(st_ch, spm_y, (uint32_t)y_beat, y_span);
    if (mc->regrel) {
        relocate(&i_ldx, 2, &rg, &r);
        relocate(&i_qact, 2, &rg, &r);
        relocate(&i_ldy, 2, &rg, &r);
        relocate(&i_mvm, 3, &rg, &r);
        relocate(&i_st, 2, &rg, &r);
    }
    prog_add(&p, i_ldx);
    prog_add(&p, tsr_enc_fence());
    prog_add(&p, i_qact);
    if (mc->acc) prog_add(&p, i_ldy);
    const uint32_t mvm_pc = prog_add(&p, i_mvm);
    prog_add(&p, i_st);
    prog_add(&p, tsr_enc_end());
    (void)mvm_pc;

    uint64_t rd0[8] = {0};
    for (uint32_t c = 0; c < C; ++c) rd0[c] = tsr_sim_perf(s, tsr_perf_ch_rd_beats(c));
    const uint64_t mvm0 = tsr_sim_perf(s, TSR_PERF_MVM_CYCLES);
    const uint32_t st = prog_run(s, &p, 0, args);
    CHECK(st == DONE, "%s: status %u err %u pc %u info 0x%x", tag, st, sim_err_code(s), sim_err_pc(s), sim_err_info(s));

    // Expected QA images
    uint8_t * qa = (uint8_t *)malloc((size_t)qa_lines * 64 * T);
    for (uint32_t t = 0; t < T; ++t) tsr_qact(kind, x + (size_t)t * K, K, qa + (size_t)t * qa_lines * 64);
    CHECK(memcmp(tsr_sim_spm(s) + (size_t)spm_qa * 64, qa, (size_t)qa_lines * 64 * T) == 0, "%s: QA image", tag);

    // Expected dst span image (SPM) = pre-image with the MVM writes applied.
    uint8_t * want = (uint8_t *)malloc((size_t)y_span * 64);
    if (mc->acc) memcpy(want, y_old, (size_t)y_span * 64);
    else memset(want, 0x5A, (size_t)y_span * 64);
    uint8_t * rowbuf = (uint8_t *)malloc((size_t)row_beats * 64);
    for (uint32_t n = 0; n < N; ++n) {
        tsr_repack_row(fmt, blocks + row_bytes * n, K, rowbuf);
        for (uint32_t t = 0; t < T; ++t) {
            float y = tsr_dot_row(fmt, rowbuf, K, qa + (size_t)t * qa_lines * 64);
            uint8_t * e = want + ((size_t)t * ys) * 64 + 4 * n;
            if (mc->acc) y = tsr_fadd(u2f(get_u32_le(e)), y);
            put_f32_le(e, &y, 1);
        }
    }
    for (uint32_t t = 0; t < T; ++t) memset(want + ((size_t)t * ys) * 64 + 4 * N, 0, (size_t)y_lines * 64 - 4 * N);
    const uint8_t * got_spm = tsr_sim_spm(s) + (size_t)spm_y * 64;
    const uint8_t * got_dram = tsr_sim_dram(s, st_ch) + y_beat * 64;
    if (memcmp(got_spm, want, (size_t)y_span * 64) != 0) {
        size_t k = 0;
        while (got_spm[k] == want[k]) ++k;
        CHECK(0, "%s: SPM dst span differs at byte %zu (token %zu, element %zu)", tag, k, k / 64 / ys,
              (k - (k / 64 / ys) * ys * 64) / 4);
    }
    CHECK(memcmp(got_dram, want, (size_t)y_span * 64) == 0, "%s: ST output", tag);

    // Beat counters: channel c streams rows_c * ROW_BEATS; ch0 also fetches and LDs.
    const uint32_t R = tsr_ceil_div_u32(N, C);
    uint64_t maxb = 0;
    for (uint32_t c = 0; c < C; ++c) {
        const uint32_t rows = (N > c * R) ? ((N - c * R < R) ? N - c * R : R) : 0;
        const uint64_t mb = (uint64_t)rows * row_beats;
        if (mb > maxb) maxb = mb;
        uint64_t want_b = mb;
        if (c == 0) want_b += tsr_ceil_div_u32(p.n, 2) + ((T - 1) * xs + x_lines) + (mc->acc ? y_span : 0);
        CHECK(tsr_sim_perf(s, tsr_perf_ch_rd_beats(c)) - rd0[c] == want_b, "%s: ch%u beats %llu, expected %llu", tag, c,
              (unsigned long long)(tsr_sim_perf(s, tsr_perf_ch_rd_beats(c)) - rd0[c]), (unsigned long long)want_b);
    }
    CHECK(tsr_sim_perf(s, TSR_PERF_MVM_CYCLES) - mvm0 == 64 + (uint64_t)T * qa_lines + maxb, "%s: MVM cycles", tag);

    const uint64_t h = fnv64(got_spm, (size_t)y_span * 64);
    free(blocks); free(x); free(y_old); free(qa); free(want); free(rowbuf);
    prog_free(&p);
    tsr_sim_destroy(s);
    return h;
}

static uint32_t rand_K(rng_t * r, uint32_t fmt, uint32_t kmax) {
    const uint32_t bs = tsr_fmt_block_size(fmt);
    switch (rng_below(r, 3)) {
    case 0: return 1024 * (1 + rng_below(r, kmax / 1024));                    // whole chunks
    case 1: return bs * (1 + rng_below(r, 8));                                // small, partial
    default: return bs * (1 + rng_below(r, kmax / bs));                       // anything
    }
}

static void test_directed(void) {
    // {C, fmt, N, K, T, acc, x_slack, y_slack, zero_strides, regrel, special}
    const uint32_t d[][11] = {
        { 4, 0,   1,   32, 1, 0, 0, 0, 1, 0, 0 },
        { 4, 0,   3,   32, 1, 0, 0, 0, 1, 0, 0 },     // N < C
        { 4, 0,   4,   64, 1, 1, 0, 0, 1, 0, 0 },
        { 4, 0,   5, 1056, 2, 0, 0, 0, 1, 0, 0 },     // one full chunk + 1 block
        { 4, 0,   7, 1024, 3, 1, 2, 3, 0, 1, 0 },
        { 4, 0,  13, 2144, 8, 0, 0, 0, 1, 0, 1 },     // T = T_MAX, partial chunk, odd N
        { 4, 0,  67, 4096, 1, 1, 0, 0, 0, 1, 0 },
        { 4, 0, 100,  992, 4, 0, 1, 1, 0, 0, 1 },     // rem-only row (31 blocks)
        { 4, 0, 257, 1056, 2, 1, 0, 5, 1, 1, 0 },
        { 4, 1,   1,  256, 1, 0, 0, 0, 1, 0, 0 },
        { 4, 1,   2,  512, 1, 1, 0, 0, 1, 0, 0 },     // N < C
        { 4, 1,   5, 1280, 2, 0, 0, 0, 1, 0, 1 },     // 1 chunk + 1 sb
        { 4, 1,   9,  768, 3, 1, 1, 0, 0, 1, 0 },     // 3 sb: partial chunk only
        { 4, 1,  31, 4096, 1, 0, 0, 2, 0, 0, 0 },
        { 4, 1,  64, 1024, 8, 1, 0, 0, 1, 1, 1 },
        { 4, 1, 130, 3328, 2, 0, 3, 0, 1, 0, 0 },
        { 1, 0,  33,  160, 2, 1, 0, 0, 1, 1, 0 },     // single channel
        { 1, 1,  17,  512, 1, 0, 0, 0, 1, 0, 0 },
        { 3, 0,  10,  320, 5, 0, 0, 1, 1, 0, 0 },     // C = 3: rows 4/4/2
        { 3, 1,   7, 2304, 2, 1, 0, 0, 1, 1, 0 },
        { 2, 0,   1,   32, 7, 1, 1, 1, 0, 0, 1 },     // N = 1, C = 2: channel 1 empty
        { 4, 0,  16,   32, 8, 0, 0, 0, 1, 0, 0 },     // y exactly one line per token
        { 4, 1,  17, 32768, 1, 1, 0, 0, 1, 0, 0 },    // K = K_MAX
    };
    for (size_t i = 0; i < sizeof(d) / sizeof(d[0]); ++i) {
        mvm_case mc = { d[i][0], d[i][1], d[i][2], d[i][3], d[i][4], (int)d[i][5], d[i][6], d[i][7], (int)d[i][8],
                        (int)d[i][9], (int)d[i][10], 1, 1000 + i };
        run_case(&mc);
    }
}

static void test_random(int n_cases) {
    rng_t r = { 2024 };
    for (int i = 0; i < n_cases; ++i) {
        mvm_case mc;
        mc.num_ch = 1 + rng_below(&r, 4);
        mc.fmt = rng_below(&r, 2);
        mc.K = rand_K(&r, mc.fmt, 4608);
        mc.N = rng_below(&r, 4) == 0 ? 1 + rng_below(&r, 2 * mc.num_ch) : 1 + rng_below(&r, 300);
        mc.T = 1 + rng_below(&r, 8);
        mc.acc = (int)rng_below(&r, 2);
        mc.x_slack = rng_below(&r, 3) == 0 ? rng_below(&r, 5) : 0;
        mc.y_slack = rng_below(&r, 3) == 0 ? rng_below(&r, 5) : 0;
        mc.zero_strides = (int)rng_below(&r, 2);
        mc.regrel = (int)rng_below(&r, 2);
        mc.special = rng_below(&r, 4) == 0;
        mc.threads = rng_below(&r, 3) == 0 ? 3 : 1;
        mc.seed = rng_u64(&r);
        run_case(&mc);
    }
}

static void test_threads(void) {
    // Same big MVM with 1 and 4 threads: bit-identical outputs.
    for (uint32_t fmt = 0; fmt < 2; ++fmt) {
        mvm_case mc = { 4, fmt, 999, fmt ? 4096u : 4160u, 3, 1, 0, 0, 1, 0, 1, 1, 77 + fmt };
        const uint64_t h1 = run_case(&mc);
        mc.threads = 4;
        const uint64_t h4 = run_case(&mc);
        CHECK(h1 == h4, "fmt %u: 1-thread and 4-thread outputs differ", fmt);
    }
}

// QACT alone: A32 / A256, T up to 8, src_stride 0 / explicit; outputs bit-exact with tsr_qact,
// tails zero; lines between QA vectors are not touched (dst stride is exactly QA_LINES).
static void test_qact_only(void) {
    rng_t r = { 99 };
    tsr_config cfg = cfg_u200(8);
    tsr_sim * s = tsr_sim_create(&cfg, NULL);
    for (int it = 0; it < 60; ++it) {
        const uint32_t kind = rng_below(&r, 2);
        const uint32_t bs = tsr_qa_block_size(kind);
        const uint32_t K = bs * (1 + rng_below(&r, kind ? 20 : 150));
        const uint32_t T = 1 + rng_below(&r, 8);
        const uint32_t xl = K * 4 / 64, ql = tsr_qa_lines(kind, K);
        const uint32_t stride = rng_below(&r, 2) ? 0 : xl + rng_below(&r, 3);
        const uint32_t xs = stride ? stride : xl;
        float * x = (float *)malloc(sizeof(float) * K * T);
        for (uint32_t t = 0; t < T; ++t) gen_x(&r, x + (size_t)t * K, K, (int)rng_below(&r, 2));
        const uint32_t src = 10, dst = src + T * xs + 5;
        memset(tsr_sim_spm(s), 0xA5, cfg.spm_bytes);
        for (uint32_t t = 0; t < T; ++t) put_f32_le(tsr_sim_spm(s) + (size_t)(src + t * xs) * 64, x + (size_t)t * K, K);
        prog_t p = {0};
        prog_add(&p, tsr_enc_qact(kind, dst, src, K, T, stride));
        prog_add(&p, tsr_enc_end());
        const uint32_t st = prog_run(s, &p, 0, NULL);
        CHECK(st == DONE, "qact status");
        uint8_t * want = (uint8_t *)malloc((size_t)ql * 64);
        for (uint32_t t = 0; t < T; ++t) {
            tsr_qact(kind, x + (size_t)t * K, K, want);
            CHECK(memcmp(tsr_sim_spm(s) + (size_t)(dst + t * ql) * 64, want, (size_t)ql * 64) == 0,
                  "QACT kind %u K %u T %u stride %u token %u", kind, K, T, stride, t);
        }
        CHECK(tsr_sim_spm(s)[(size_t)(dst + T * ql) * 64] == 0xA5 && tsr_sim_spm(s)[(size_t)dst * 64 - 1] == 0xA5,
              "QACT wrote outside its dst span");
        free(want);
        free(x);
        prog_free(&p);
    }
    tsr_sim_destroy(s);
}

// N = 0: legal, writes nothing, reads no weights.
static void test_n_zero_and_chain(void) {
    tsr_config cfg = cfg_u200(1);
    tsr_sim * s = tsr_sim_create(&cfg, NULL);
    const uint32_t WOFF = (uint32_t)(TSR_REGION_WEIGHT_OFF / 64);
    prog_t p = {0};
    prog_add(&p, tsr_enc_mvm(0, 1, 500, 0, WOFF, 0, 64, 1, 0));
    prog_add(&p, tsr_enc_end());
    const uint64_t rd1 = tsr_sim_perf(s, tsr_perf_ch_rd_beats(1));
    CHECK(prog_run(s, &p, 0, NULL) == DONE, "N = 0 ok");
    CHECK(tsr_sim_spm(s)[500 * 64] == 0xA5, "N = 0 writes nothing");
    CHECK(tsr_sim_perf(s, tsr_perf_ch_rd_beats(1)) == rd1, "N = 0 reads no weights");

    // Two MVMs sharing one QA, the second with ACC: y = W1 x + W2 x (tsr_fadd of the two dots).
    rng_t r = { 5 };
    const uint32_t N = 37, K = 512, fmt = TSR_FMT_Q4_K;
    const size_t rbytes = (K / 256) * 144;
    uint8_t * b1 = (uint8_t *)malloc(rbytes * N), * b2 = (uint8_t *)malloc(rbytes * N);
    for (uint32_t n = 0; n < N; ++n) { gen_row_blocks(&r, fmt, K, b1 + rbytes * n); gen_row_blocks(&r, fmt, K, b2 + rbytes * n); }
    const uint64_t w1 = WOFF, w2 = WOFF + tsr_tensor_ch_beats(fmt, N, K, 4) + 3;
    put_weights(s, fmt, N, K, w1, b1);
    put_weights(s, fmt, N, K, w2, b2);
    float x[512];
    gen_f32(&r, x, K);
    put_f32_le(tsr_sim_spm(s) + 100 * 64, x, K);
    prog_reset(&p);
    prog_add(&p, tsr_enc_qact(TSR_QA_A256, 200, 100, K, 1, 0));
    prog_add(&p, tsr_enc_mvm(fmt, 0, 300, 200, (uint32_t)w1, N, K, 1, 0));
    prog_add(&p, tsr_enc_mvm(fmt, TSR_MVM_FLAG_ACC, 300, 200, (uint32_t)w2, N, K, 1, 0));
    prog_add(&p, tsr_enc_end());
    CHECK(prog_run(s, &p, 0, NULL) == DONE, "chain ok");
    uint8_t qa[64 * 16];
    tsr_qact(TSR_QA_A256, x, K, qa);
    uint8_t row[64 * 32];
    for (uint32_t n = 0; n < N; ++n) {
        tsr_repack_row(fmt, b1 + rbytes * n, K, row);
        const float y1 = tsr_dot_row(fmt, row, K, qa);
        tsr_repack_row(fmt, b2 + rbytes * n, K, row);
        const float y2 = tsr_dot_row(fmt, row, K, qa);
        const uint32_t want = f2u(tsr_fadd(y1, y2));
        const uint32_t got = get_u32_le(tsr_sim_spm(s) + 300 * 64 + 4 * n);
        CHECK(got == want, "chain row %u: 0x%08x vs 0x%08x", n, got, want);
    }
    free(b1);
    free(b2);
    prog_free(&p);
    tsr_sim_destroy(s);
}

int main(int argc, char ** argv) {
    const int n_random = argc > 1 ? atoi(argv[1]) : 150;
    test_qact_only();
    test_directed();
    test_n_zero_and_chain();
    test_threads();
    test_random(n_random);
    return test_finish("sim_mvm");
}
