// fw_selftest_cases.h — the `selftest` ids of SW §7.6 as host-side cases (data set-up, launch, every
// output checked against tsr_ref / tsr_ref_vu / tsr_ref_at): factored out of test_fw_selftest_sim.c
// (P2.5) by the W2a integration so that fw_m6_e2e (test_fw_m6_e2e.cpp) runs the same cases in its
// sessions. Plain C, also valid C++. The includer provides test_sim_obs.h (CHECK, rng, put_weights)
// first; a session is one booted image (fw_sim_host.hpp) with `idx` = the selftest kernel's index.
//   st_run_ids   ids 1-10 (id 2 twice) and the misuses, in fw_selftest_sim's order
#pragma once

#include <math.h>

#include "../ref/fw_sim_host.hpp"
#include "../ref/selftest_abi.hpp"
#include "tsr_ref_at.h"
#include "tsr_ref_math.h"
#include "tsr_ref_vu.h"

#define NCH       4u
#define AREA0     (TSR_REGION_IO_OFF / 64u)          // ch0 beat of the first data area
#define AREA_STEP 0x10000u                            // beats between the ids' areas (4 MiB)
#define W_BASE    (TSR_REGION_WEIGHT_OFF / 64u)       // weights, every channel
#define S1_BEAT   0x400000u                           // id 1 scratch, every channel (256 MiB)
#define KV_BASE   0xC00000u                           // id 5 planes, every channel
#define LMAX      4096u

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------
static uint8_t * D(tsr_sim * sim, uint32_t ch, uint64_t beat) { return tsr_sim_dram(sim, ch) + beat * 64u; }
static uint32_t rd(const uint8_t * p, uint32_t i) { return get_u32_le(p + 4u * i); }
static void wr(uint8_t * p, uint32_t i, uint32_t v) {
    p[4 * i] = (uint8_t)v; p[4 * i + 1] = (uint8_t)(v >> 8); p[4 * i + 2] = (uint8_t)(v >> 16); p[4 * i + 3] = (uint8_t)(v >> 24);
}
static void put_params(tsr_sim * sim, uint64_t area, uint32_t id, const uint32_t * p, uint32_t n) {
    uint8_t * q = D(sim, 0, area);
    memset(q, 0, 64);
    wr(q, 0, TSR_ST_MAGIC);
    wr(q, 1, id);
    for (uint32_t i = 0; i < n; ++i) wr(q, 2 + i, p[i]);
}
static void rand_f32(rng_t * r, uint8_t * dst, uint32_t n, double scale) {
    for (uint32_t i = 0; i < n; ++i) wr(dst, i, f2u((float)(rng_normal(r) * scale)));
}
static uint64_t g_cmp_lines, g_cmp_n;   // lines / regions compared (reported)
static int cmp_lines(const char * what, const uint8_t * got, const uint8_t * want, size_t lines) {
    g_cmp_lines += lines;
    g_cmp_n++;
    if (memcmp(got, want, lines * 64u) == 0) return 1;
    size_t k = 0;
    while (got[k] == want[k]) ++k;
    CHECK(0, "%s: differs at byte %zu (word 0x%08x, expected 0x%08x)", what, k, get_u32_le(got + (k & ~(size_t)3)),
          get_u32_le(want + (k & ~(size_t)3)));
    return 0;
}
static uint32_t weights(tsr_sim * sim, rng_t * r, uint32_t fmt, uint32_t N, uint32_t K, uint64_t w_off) {
    const size_t rb = (size_t)(K / tsr_fmt_block_size(fmt)) * tsr_fmt_ggml_block_bytes(fmt);
    uint8_t * blocks = (uint8_t *)malloc(rb * N);
    for (uint32_t n = 0; n < N; ++n) gen_row_blocks(r, fmt, K, blocks + rb * n);
    put_weights(sim, fmt, N, K, w_off, blocks);
    free(blocks);
    return (uint32_t)w_off;
}

typedef struct session { tsr_sim * sim; fwh h; int idx; uint32_t done[32][7]; uint32_t n_done; const char * sched; } session;

static int launch(session * s, uint32_t id, uint64_t area, uint32_t lines, uint32_t seed, int32_t want_ret, uint32_t want_status) {
    const uint32_t a[7] = { id, (uint32_t)area, lines, seed, 0, 0, 0 };
    if (fwh_launch(&s->h, (uint32_t)s->idx, a, 1ull << 34) != 0) {
        CHECK(0, "[%s] selftest %u: no answer", s->sched, id);
        return 0;
    }
    const uint32_t st = fwh_status(&s->h);
    CHECK(TSR_GET(s->h.done[0], TSR_LCQ_R0_TYPE) == TSR_LCQ_R0_TYPE_DONE && st == want_status && fwh_ret(&s->h) == want_ret,
          "[%s] selftest %u: DONE status %u RET 0x%08x (expected %u, 0x%08x; first error ticket 0x%08x code %u info 0x%08x "
          "w0 0x%08x)", s->sched, id, st, (uint32_t)fwh_ret(&s->h), want_status, (uint32_t)want_ret, s->h.done[3],
          TSR_GET(s->h.done[0], TSR_LCQ_R0_ERR_CODE), s->h.done[4], s->h.done[5]);
    if (s->n_done < 32u) memcpy(s->done[s->n_done++], s->h.done, 7 * sizeof(uint32_t));
    return st == want_status && fwh_ret(&s->h) == want_ret;
}

// ---------------------------------------------------------------------------
// 1 LDST
// ---------------------------------------------------------------------------
static void id1(session * s) {
    tsr_sim * sim = s->sim;
    const uint64_t area = AREA0 + 1u * AREA_STEP;
    const uint32_t p[2] = { LMAX, S1_BEAT };
    put_params(sim, area, 1, p, 2);
    rng_t r = { 0x1D1 };
    rng_bytes(&r, D(sim, 0, area + 1), (size_t)LMAX * 64u);
    if (!launch(s, 1, area, 1 + LMAX, 11, 0, 0)) return;
    uint32_t lens[5];
    const uint32_t n = tsr_st1_lens(LMAX, lens);
    uint32_t off = 0;
    char what[96];
    const uint8_t * src = D(sim, 0, area + 1);
    for (uint32_t k = 0; k < n; ++k) {
        for (uint32_t c = 0; c < NCH; ++c) {
            snprintf(what, sizeof(what), "[%s] id 1: length %u channel %u", s->sched, lens[k], c);
            cmp_lines(what, D(sim, c, S1_BEAT + off), src, lens[k]);
        }
        off += lens[k];
    }
    for (uint32_t c = 0; c < NCH; ++c) {
        snprintf(what, sizeof(what), "[%s] id 1: cross copy into channel %u", s->sched, c);
        cmp_lines(what, D(sim, c, S1_BEAT + off), src, LMAX);
    }
}

// ---------------------------------------------------------------------------
// 2 QACT
// ---------------------------------------------------------------------------
static void id2(session * s, uint32_t K, uint32_t T, uint32_t slot) {
    tsr_sim * sim = s->sim;
    const uint64_t area = AREA0 + (uint64_t)slot * AREA_STEP;
    const uint32_t p[2] = { K, T };
    put_params(sim, area, 2, p, 2);
    const tsr_st2_layout L = tsr_st2(K, T);
    rng_t r = { 0x2D2 + K };
    rand_f32(&r, D(sim, 0, area + L.x), T * K, 3.0);
    if (!launch(s, 2, area, L.end, 0, 0, 0)) return;
    const uint32_t q32 = tsr_st_qa_lines(0, K), q256 = tsr_st_qa_lines(1, K);
    float * x = (float *)malloc(K * sizeof(float));
    uint8_t * qa = (uint8_t *)malloc((size_t)q256 * 64u + q32 * 64u);
    char what[96];
    for (uint32_t t = 0; t < T; ++t) {
        for (uint32_t k = 0; k < K; ++k) x[k] = u2f(rd(D(sim, 0, area + L.x + t * (K / 16u)), k));
        tsr_qact(TSR_QA_A32, x, K, qa);
        snprintf(what, sizeof(what), "[%s] id 2: K %u token %u A32", s->sched, K, t);
        cmp_lines(what, D(sim, 0, area + L.qa32 + t * q32), qa, q32);
        tsr_qact(TSR_QA_A256, x, K, qa);
        snprintf(what, sizeof(what), "[%s] id 2: K %u token %u A256", s->sched, K, t);
        cmp_lines(what, D(sim, 0, area + L.qa256 + t * q256), qa, q256);
    }
    free(x);
    free(qa);
}

// ---------------------------------------------------------------------------
// 3 MVM
// ---------------------------------------------------------------------------
static void mvm_ref(tsr_sim * sim, uint32_t fmt, uint32_t N, uint32_t K, uint32_t w_off, const uint8_t * xrow, float * y) {
    const uint32_t kind = tsr_qa_kind_for_fmt(fmt);
    float * x = (float *)malloc(K * sizeof(float));
    uint8_t * qa = (uint8_t *)malloc((size_t)tsr_qa_lines(kind, K) * 64u);
    uint8_t * row = (uint8_t *)malloc((size_t)tsr_row_beats(fmt, K) * 64u);
    for (uint32_t k = 0; k < K; ++k) x[k] = u2f(rd(xrow, k));
    tsr_qact(kind, x, K, qa);
    for (uint32_t n = 0; n < N; ++n) {
        uint32_t ch;
        uint64_t off;
        tsr_row_location(fmt, N, K, NCH, n, &ch, &off);
        y[n] = tsr_dot_row(fmt, D(sim, ch, w_off + off), K, qa);
    }
    free(x); free(qa); free(row);
}

static void put_y(uint8_t * dst, const float * y, uint32_t N) {
    const uint32_t yl = (N + 15u) / 16u;
    memset(dst, 0, (size_t)yl * 64u);
    for (uint32_t n = 0; n < N; ++n) wr(dst, n, f2u(y[n]));
}

static void id3(session * s) {
    tsr_sim * sim = s->sim;
    const uint32_t N = 200, K = 512, TM = tsr_sim_config(sim)->t_max;
    const uint64_t area = AREA0 + 3u * AREA_STEP;
    rng_t r = { 0x3D3 };
    const uint32_t w8 = weights(sim, &r, TSR_FMT_Q8_0, N, K, W_BASE);
    const uint32_t w4 = weights(sim, &r, TSR_FMT_Q4_K, N, K, W_BASE + 0x10000u);
    const uint32_t p[4] = { N, K, w8, w4 };
    put_params(sim, area, 3, p, 4);
    const tsr_st3_layout L = tsr_st3(N, K, TM);
    rand_f32(&r, D(sim, 0, area + L.x), TM * K, 1.0);
    if (!launch(s, 3, area, L.end, 0, 0, 0)) return;
    const uint32_t yl = (N + 15u) / 16u;
    float y8[200], y4[200], ya[200];
    uint8_t want[13 * 64];
    char what[96];
    for (uint32_t t = 0; t < TM; ++t) {
        const uint8_t * xr = D(sim, 0, area + L.x + t * (K / 16u));
        mvm_ref(sim, TSR_FMT_Q8_0, N, K, w8, xr, y8);
        mvm_ref(sim, TSR_FMT_Q4_K, N, K, w4, xr, y4);
        for (uint32_t n = 0; n < N; ++n) ya[n] = tsr_fadd(y8[n], y4[n]);
        if (t == 0) {
            put_y(want, y8, N);
            snprintf(what, sizeof(what), "[%s] id 3: Q8_0 T=1", s->sched);
            cmp_lines(what, D(sim, 0, area + L.y8_1), want, yl);
            put_y(want, y4, N);
            snprintf(what, sizeof(what), "[%s] id 3: Q4_K T=1", s->sched);
            cmp_lines(what, D(sim, 0, area + L.y4_1), want, yl);
        }
        put_y(want, y8, N);
        snprintf(what, sizeof(what), "[%s] id 3: Q8_0 T=%u token %u", s->sched, TM, t);
        cmp_lines(what, D(sim, 0, area + L.y8_t + t * yl), want, yl);
        put_y(want, y4, N);
        snprintf(what, sizeof(what), "[%s] id 3: Q4_K T=%u token %u", s->sched, TM, t);
        cmp_lines(what, D(sim, 0, area + L.y4_t + t * yl), want, yl);
        put_y(want, ya, N);
        snprintf(what, sizeof(what), "[%s] id 3: ACC token %u", s->sched, t);
        cmp_lines(what, D(sim, 0, area + L.yacc + t * yl), want, yl);
    }
}

// ---------------------------------------------------------------------------
// 4 VU
// ---------------------------------------------------------------------------
static void id4(session * s) {
    tsr_sim * sim = s->sim;
    const uint32_t M = 40, R = 5;
    const uint64_t area = AREA0 + 4u * AREA_STEP;
    const tsr_st4_layout L = tsr_st4(M, R);
    const uint32_t ts = f2u(powf(10000.0f, -2.0f / 64.0f)), mrope = f2u(1.0f), fsc = f2u(0.5f);
    const uint32_t sk = f2u(1.0f / (float)M), se = f2u(1e-6f);
    const uint32_t p[7] = { M, R, ts, mrope, fsc, sk, se };
    put_params(sim, area, 4, p, 7);
    rng_t r = { 0x4D4 };
    uint8_t * A = D(sim, 0, area);
    memset(A + 64, 0, (size_t)(L.end - 1u) * 64u);
    for (uint32_t row = 0; row < R; ++row) {
        rand_f32(&r, A + (size_t)(L.a + row * L.L) * 64u, M, 2.0);
        rand_f32(&r, A + (size_t)(L.b + row * L.L) * 64u, M, 2.0);
    }
    rand_f32(&r, A + (size_t)L.c * 64u, R, 1.0);
    rand_f32(&r, A + (size_t)L.sv * 64u, 16, 1.5);
    wr(A + (size_t)L.sv * 64u, 10, f2u(2.25f));                       // SOP sa: a positive value (SQRT, RSQRT)
    rand_f32(&r, A + (size_t)L.rope_x * 64u, 256, 1.0);
    wr(A + (size_t)L.rope_pos * 64u, 0, 7u);
    wr(A + (size_t)L.rope_pos * 64u, 1, 300u);
    for (uint32_t i = 0; i < 16u; ++i) wr(A + (size_t)L.rope_ff * 64u, i, f2u(1.0f + (float)rng_unit(&r)));
    for (uint32_t rr = 0; rr < 2u; ++rr) {
        uint8_t blk[8 * 34 + 144];
        gen_row_blocks(&r, TSR_FMT_Q8_0, 256, blk);
        tsr_repack_row(TSR_FMT_Q8_0, blk, 256, A + (size_t)(L.dq8 + rr * L.dq8_beats) * 64u);
        gen_row_blocks(&r, TSR_FMT_Q4_K, 256, blk);
        tsr_repack_row(TSR_FMT_Q4_K, blk, 256, A + (size_t)(L.dq4 + rr * L.dq4_beats) * 64u);
    }
    // Copy the inputs (the launch overwrites only outputs, but keep them apart anyway).
    const size_t in_bytes = (size_t)L.in_end * 64u;
    uint8_t * in = (uint8_t *)malloc(in_bytes);
    memcpy(in, A, in_bytes);
    if (!launch(s, 4, area, L.end, 0, 0, 0)) { free(in); return; }
    const uint32_t * a = (const uint32_t *)(in + (size_t)L.a * 64u), * b = (const uint32_t *)(in + (size_t)L.b * 64u);
    const uint32_t * c = (const uint32_t *)(in + (size_t)L.c * 64u), * sv = (const uint32_t *)(in + (size_t)L.sv * 64u);
    uint32_t * w = (uint32_t *)calloc((size_t)(L.end + 64u) * 16u, 4);
    char what[96];
    for (uint32_t f = 0; f < TSR_ST4_N_FUNC; ++f) {
        const uint32_t reads = tsr_vop_func_reads(f);
        const int rc = (reads & 4u) != 0;
        memset(w, 0, (size_t)R * L.L * 64u);
        tsr_ref_vop(f, rc ? TSR_VOP_CMODE_VEC : TSR_VOP_CMODE_NONE, R, M, w, L.L, (reads & 1u) ? a : NULL, L.L,
                    (reads & 2u) ? b : NULL, L.L, rc ? c : NULL);
        snprintf(what, sizeof(what), "[%s] id 4: VOP func %u", s->sched, f);
        cmp_lines(what, A + (size_t)(L.vop + f * R * L.L) * 64u, (const uint8_t *)w, R * L.L);
    }
    {
        const uint32_t s3 = sv[3];
        memset(w, 0, (size_t)R * L.L * 64u);
        tsr_ref_vop(TSR_VOP_FUNC_MULC, TSR_VOP_CMODE_SREG, R, M, w, L.L, a, L.L, NULL, 0, &s3);
        snprintf(what, sizeof(what), "[%s] id 4: VOP MULC SREG", s->sched);
        cmp_lines(what, A + (size_t)L.vop_sreg * 64u, (const uint8_t *)w, R * L.L);
    }
    static const uint32_t reds[TSR_ST4_N_RED][2] = {
        {TSR_RED_KIND_SUM, TSR_RED_POST_NONE}, {TSR_RED_KIND_SUM, TSR_RED_POST_RCP}, {TSR_RED_KIND_SUM, TSR_RED_POST_RMS},
        {TSR_RED_KIND_SUMSQ, TSR_RED_POST_NONE}, {TSR_RED_KIND_SUMSQ, TSR_RED_POST_RCP}, {TSR_RED_KIND_SUMSQ, TSR_RED_POST_RMS},
        {TSR_RED_KIND_MAX, TSR_RED_POST_NONE} };
    for (uint32_t j = 0; j < TSR_ST4_N_RED; ++j) {
        memset(w, 0, 64);
        tsr_ref_red(reds[j][0], reds[j][1], R, M, w, a, L.L, sk, se);
        snprintf(what, sizeof(what), "[%s] id 4: RED kind %u post %u", s->sched, reds[j][0], reds[j][1]);
        cmp_lines(what, A + (size_t)(L.red + j) * 64u, (const uint8_t *)w, 1);
    }
    {   // SOP and the s registers (LDS, MOVI, RED TO_S, STS)
        uint32_t sr[16];
        memcpy(sr, sv, sizeof(sr));
        sr[1] = sk; sr[2] = se; sr[4] = mrope; sr[5] = ts; sr[6] = fsc;
        sr[7] = tsr_ref_red_row(TSR_RED_KIND_SUM, TSR_RED_POST_NONE, a, M, 0, 0);
        static const uint32_t subs[12] = { TSR_SOP_SUB_ADD, TSR_SOP_SUB_MUL, TSR_SOP_SUB_ADDI, TSR_SOP_SUB_MULI,
                                           TSR_SOP_SUB_SQRT, TSR_SOP_SUB_RECIP, TSR_SOP_SUB_SUB, TSR_SOP_SUB_DIV,
                                           TSR_SOP_SUB_RSQRT, TSR_SOP_SUB_MAX, TSR_SOP_SUB_MIN, TSR_SOP_SUB_EXP };
        memset(w, 0, 64);
        for (uint32_t j = 0; j < 12u; ++j) w[j] = tsr_ref_sop(subs[j], sr[10], sr[11], sv[12], 0);
        w[12] = sr[7];
        w[13] = sr[0];
        w[14] = sk;
        w[15] = sr[15];
        snprintf(what, sizeof(what), "[%s] id 4: SOP", s->sched);
        cmp_lines(what, A + (size_t)L.sop * 64u, (const uint8_t *)w, 1);
    }
    {   // ROPE
        tsr_ref_rope_cfg rc;
        memset(&rc, 0, sizeof(rc));
        rc.mode = TSR_ROPE_MODE_NORM; rc.m = mrope; rc.ts = ts; rc.head_dim = 64; rc.n_dims = 64; rc.n_heads = 2; rc.T = 2;
        const uint32_t * pos = (const uint32_t *)(in + (size_t)L.rope_pos * 64u);
        const uint32_t * x = (const uint32_t *)(in + (size_t)L.rope_x * 64u);
        memset(w, 0, TSR_ST4_ROPE_LINES * 64u);
        tsr_ref_rope(&rc, w, 0, x, 0, pos, NULL);
        snprintf(what, sizeof(what), "[%s] id 4: ROPE NORM", s->sched);
        cmp_lines(what, A + (size_t)L.rope_norm * 64u, (const uint8_t *)w, TSR_ST4_ROPE_LINES);
        rc.mode = TSR_ROPE_MODE_NEOX; rc.ff = 1; rc.fs = 1; rc.freq_scale = fsc; rc.n_dims = 32;
        memset(w, 0, TSR_ST4_ROPE_LINES * 64u);
        tsr_ref_rope(&rc, w, 0, x, 0, pos, (const uint32_t *)(in + (size_t)L.rope_ff * 64u));
        snprintf(what, sizeof(what), "[%s] id 4: ROPE NEOX FF FS", s->sched);
        cmp_lines(what, A + (size_t)L.rope_neox * 64u, (const uint8_t *)w, TSR_ST4_ROPE_LINES);
    }
    const uint32_t caps = tsr_sim_config(sim)->caps;
    if (caps & TSR_CAPS_TOPK) {
        memset(w, 0, (size_t)R * 64u);
        tsr_ref_topk(4, R, M, w, 0, a, L.L);
        snprintf(what, sizeof(what), "[%s] id 4: TOPK", s->sched);
        cmp_lines(what, A + (size_t)L.topk * 64u, (const uint8_t *)w, R);
    }
    if (caps & TSR_CAPS_DQROW) {
        memset(w, 0, 32u * 64u);
        tsr_ref_dqrow(TSR_FMT_Q8_0, 256, 2, w, 0, in + (size_t)L.dq8 * 64u, 0);
        snprintf(what, sizeof(what), "[%s] id 4: DQROW Q8_0", s->sched);
        cmp_lines(what, A + (size_t)L.dq8_out * 64u, (const uint8_t *)w, 32);
        memset(w, 0, 32u * 64u);
        tsr_ref_dqrow(TSR_FMT_Q4_K, 256, 2, w, 0, in + (size_t)L.dq4 * 64u, 0);
        snprintf(what, sizeof(what), "[%s] id 4: DQROW Q4_K", s->sched);
        cmp_lines(what, A + (size_t)L.dq4_out * 64u, (const uint8_t *)w, 32);
    }
    free(w);
    free(in);
}

// ---------------------------------------------------------------------------
// 5 AT
// ---------------------------------------------------------------------------
static void id5(session * s) {
    tsr_sim * sim = s->sim;
    const uint32_t n_kv = 40, Tq = 4, C = NCH;
    const float scale = 0.125f;
    const uint64_t area = AREA0 + 5u * AREA_STEP;
    const uint32_t BK = KV_BASE, BV = KV_BASE + 0x100u;
    const uint32_t p[5] = { n_kv, Tq, f2u(scale), BK, BV };
    put_params(sim, area, 5, p, 5);
    const tsr_st5_layout L = tsr_st5(n_kv, Tq, C);
    uint8_t * A = D(sim, 0, area);
    memset(A + 64, 0, (size_t)(L.end - 1u) * 64u);
    rng_t r = { 0x5D5 };
    rand_f32(&r, A + (size_t)L.k * 64u, n_kv * 64u, 1.0);
    rand_f32(&r, A + (size_t)L.v * 64u, n_kv * 64u, 1.0);
    rand_f32(&r, A + (size_t)L.q * 64u, Tq * 64u, 1.0);
    for (uint32_t q = 0; q < Tq; ++q)
        for (uint32_t j = 0; j < 16u * L.n_groups; ++j) {
            const uint32_t k = rng_below(&r, 4);
            const float m = k == 0 ? -INFINITY : k == 1 ? 0.0f : k == 2 ? -0.0f : (float)(rng_normal(&r) * 0.5);
            wr(A + (size_t)(L.mask + q * L.n_groups) * 64u, j, f2u(m));
        }
    for (uint32_t c = 0; c < C; ++c) {   // the planes start zeroed (KVST read-modify-writes partial meta beats)
        memset(D(sim, c, BK), 0, 17u * 64u);
        memset(D(sim, c, BV), 0, 17u * 64u);
    }
    if (!launch(s, 5, area, L.end, 0, 0, 0)) return;
    char what[96];
    // QACT of the rows, then the planes (KVST).
    const tsr_kv_geom g = { 64, C, 1 };
    uint8_t * kqa = (uint8_t *)calloc(n_kv, 128), * vqa = (uint8_t *)calloc(n_kv, 128), * qqa = (uint8_t *)calloc(Tq, 128);
    float x[64];
    for (uint32_t i = 0; i < n_kv; ++i) {
        for (uint32_t k = 0; k < 64u; ++k) x[k] = u2f(rd(A + (size_t)(L.k + 4u * i) * 64u, k));
        tsr_qact(TSR_QA_A32, x, 64, kqa + 128u * i);
        for (uint32_t k = 0; k < 64u; ++k) x[k] = u2f(rd(A + (size_t)(L.v + 4u * i) * 64u, k));
        tsr_qact(TSR_QA_A32, x, 64, vqa + 128u * i);
    }
    for (uint32_t i = 0; i < Tq; ++i) {
        for (uint32_t k = 0; k < 64u; ++k) x[k] = u2f(rd(A + (size_t)(L.q + 4u * i) * 64u, k));
        tsr_qact(TSR_QA_A32, x, 64, qqa + 128u * i);
    }
    snprintf(what, sizeof(what), "[%s] id 5: QACT K", s->sched);
    cmp_lines(what, A + (size_t)L.kqa * 64u, kqa, 2u * n_kv);
    snprintf(what, sizeof(what), "[%s] id 5: QACT V", s->sched);
    cmp_lines(what, A + (size_t)L.vqa * 64u, vqa, 2u * n_kv);
    snprintf(what, sizeof(what), "[%s] id 5: QACT Q", s->sched);
    cmp_lines(what, A + (size_t)L.qqa * 64u, qqa, 2u * Tq);
    uint8_t * hk[NCH], * hv[NCH];
    const uint8_t * dk[NCH], * dv[NCH];
    for (uint32_t c = 0; c < C; ++c) {
        hk[c] = (uint8_t *)calloc(17, 64);
        hv[c] = (uint8_t *)calloc(17, 64);
        dk[c] = D(sim, c, BK);
        dv[c] = D(sim, c, BV);
    }
    tsr_ref_kvst(TSR_KV_K, &g, hk, 0, n_kv, kqa, 0, NULL);
    tsr_ref_kvst(TSR_KV_V, &g, hv, 0, n_kv, vqa, 0, NULL);
    for (uint32_t c = 0; c < C; ++c) {
        snprintf(what, sizeof(what), "[%s] id 5: KVST K plane channel %u", s->sched, c);
        cmp_lines(what, dk[c], hk[c], 17);
        snprintf(what, sizeof(what), "[%s] id 5: KVST V plane channel %u", s->sched, c);
        cmp_lines(what, dv[c], hv[c], 17);
    }
    // Each op from the device's own inputs.
    const tsr_att_shape sh = { 64, 64, 1, 0, 1, Tq };
    const uint32_t ng = L.n_groups, R = L.R;
    const size_t outb = (size_t)(L.end - L.kqa) * 64u;
    float * h = (float *)malloc(outb), * dev = (float *)malloc(outb);
    memcpy(dev, A + (size_t)L.kqa * 64u, outb);
    memset(h, 0, outb);
#define AT_LINE(field) ((L.field - L.kqa) * 16u)
    tsr_ref_attqk(&sh, C, dk, 0, ng, 0, n_kv, TSR_ATT_MAXOUT, A + (size_t)L.qqa * 64u, h + AT_LINE(scores), 0, h + AT_LINE(mvec));
    snprintf(what, sizeof(what), "[%s] id 5: ATTQK", s->sched);
    cmp_lines(what, (const uint8_t *)(dev + AT_LINE(scores)), (const uint8_t *)(h + AT_LINE(scores)), R * ng + 1u);
    tsr_ref_attqk(&sh, C, dk, 0, ng, 0, n_kv, TSR_ATT_MAXOUT | TSR_ATT_CAUSAL, A + (size_t)L.qqa * 64u, h + AT_LINE(scores_c),
                  0, h + AT_LINE(mvec_c));
    snprintf(what, sizeof(what), "[%s] id 5: ATTQK CAUSAL", s->sched);
    cmp_lines(what, (const uint8_t *)(dev + AT_LINE(scores_c)), (const uint8_t *)(h + AT_LINE(scores_c)), R * ng + 1u);
    tsr_ref_asmax(h + AT_LINE(p), dev + AT_LINE(scores), dev + AT_LINE(mvec), ng, 0, R, scale);
    snprintf(what, sizeof(what), "[%s] id 5: ASMAX", s->sched);
    cmp_lines(what, (const uint8_t *)(dev + AT_LINE(p)), (const uint8_t *)(h + AT_LINE(p)), R * ng);
    tsr_ref_amask(h + AT_LINE(sm), dev + AT_LINE(scores), (const float *)(A + (size_t)L.mask * 64u), ng, 0, R, 1, ng,
                  h + AT_LINE(mvec_m));
    snprintf(what, sizeof(what), "[%s] id 5: AMASK", s->sched);
    cmp_lines(what, (const uint8_t *)(dev + AT_LINE(sm)), (const uint8_t *)(h + AT_LINE(sm)), R * ng + 1u);
    tsr_ref_attpv(&sh, C, 1, dv, 0, ng, TSR_ATT_LSUM, dev + AT_LINE(p), 0, h + AT_LINE(parts), 0);
    snprintf(what, sizeof(what), "[%s] id 5: ATTPV LSUM", s->sched);
    cmp_lines(what, (const uint8_t *)(dev + AT_LINE(parts)), (const uint8_t *)(h + AT_LINE(parts)), C * 4u * R + C);
    tsr_ref_amerge(h + AT_LINE(orun1), dev + AT_LINE(parts), dev + AT_LINE(mvec), h + AT_LINE(st1), R, 64, scale, C, 0,
                   TSR_ATT_INIT);
    snprintf(what, sizeof(what), "[%s] id 5: AMERGE INIT", s->sched);
    cmp_lines(what, (const uint8_t *)(dev + AT_LINE(orun1)), (const uint8_t *)(h + AT_LINE(orun1)), 4u * R + 2u);
    tsr_ref_amerge(h + AT_LINE(orun2), dev + AT_LINE(parts), dev + AT_LINE(mvec), h + AT_LINE(st2), R, 64, scale, C, 0,
                   TSR_ATT_INIT);
    tsr_ref_amerge(h + AT_LINE(orun2), dev + AT_LINE(parts), dev + AT_LINE(mvec_m), h + AT_LINE(st2), R, 64, scale, C, 0, 0);
    snprintf(what, sizeof(what), "[%s] id 5: AMERGE merge", s->sched);
    cmp_lines(what, (const uint8_t *)(dev + AT_LINE(orun2)), (const uint8_t *)(h + AT_LINE(orun2)), 4u * R + 2u);
    tsr_ref_anorm(h + AT_LINE(out), dev + AT_LINE(orun2), dev + AT_LINE(st2) + 16u, R, 64);
    snprintf(what, sizeof(what), "[%s] id 5: ANORM", s->sched);
    cmp_lines(what, (const uint8_t *)(dev + AT_LINE(out)), (const uint8_t *)(h + AT_LINE(out)), 4u * R);
#undef AT_LINE
    for (uint32_t c = 0; c < C; ++c) { free(hk[c]); free(hv[c]); }
    free(kqa); free(vqa); free(qqa); free(h); free(dev);
}

// ---------------------------------------------------------------------------
// 6 CONC
// ---------------------------------------------------------------------------
static void id6(session * s) {
    tsr_sim * sim = s->sim;
    const uint32_t N = 96, K = 512, J = 9, fmt = TSR_FMT_Q4_K;
    const uint64_t area = AREA0 + 6u * AREA_STEP;
    rng_t r = { 0x6D6 };
    const uint32_t w = weights(sim, &r, fmt, N, K, W_BASE + 0x20000u);
    const uint32_t p[5] = { N, K, w, fmt, J };
    put_params(sim, area, 6, p, 5);
    const tsr_st6_layout L = tsr_st6(N, K, J);
    rand_f32(&r, D(sim, 0, area + L.x), J * K, 1.0);
    if (!launch(s, 6, area, L.end, 0, 0, 0)) return;
    const uint32_t yl = (N + 15u) / 16u;
    float y[96];
    uint32_t yw[16 * 6], zw[16 * 6], sl[16];
    char what[96];
    for (uint32_t j = 0; j < J; ++j) {
        mvm_ref(sim, fmt, N, K, w, D(sim, 0, area + L.x + j * (K / 16u)), y);
        memset(yw, 0, sizeof(yw));
        for (uint32_t n = 0; n < N; ++n) yw[n] = f2u(y[n]);
        tsr_ref_vop(TSR_VOP_FUNC_SILU, TSR_VOP_CMODE_NONE, 1, N, zw, yl, yw, yl, NULL, 0, NULL);
        snprintf(what, sizeof(what), "[%s] id 6: job %u y", s->sched, j);
        cmp_lines(what, D(sim, 0, area + L.y + j * yl), (const uint8_t *)zw, yl);
        memset(sl, 0, sizeof(sl));
        tsr_ref_red(TSR_RED_KIND_SUM, TSR_RED_POST_NONE, 1, N, sl, zw, yl, 0, 0);
        snprintf(what, sizeof(what), "[%s] id 6: job %u sum", s->sched, j);
        cmp_lines(what, D(sim, 0, area + L.s + j), (const uint8_t *)sl, 1);
    }
}

// ---------------------------------------------------------------------------
// 7, 8, 10 (device-checked) and 9 (log)
// ---------------------------------------------------------------------------
static void id7(session * s) {
    tsr_sim * sim = s->sim;
    const uint64_t area = AREA0 + 7u * AREA_STEP;
    rng_t r = { 0x7D7 };
    const uint32_t w = weights(sim, &r, TSR_FMT_Q8_0, 16, 32, W_BASE + 0x30000u);
    const uint32_t p[2] = { 0xFFFFFFF0u, w };
    put_params(sim, area, 7, p, 2);
    const tsr_st7_layout L = tsr_st7();
    rng_bytes(&r, D(sim, 0, area + L.in), 128);
    if (!launch(s, 7, area, L.end, 0, 0, 0)) return;
    char what[64];
    snprintf(what, sizeof(what), "[%s] id 7: recovery round trip", s->sched);
    cmp_lines(what, D(sim, 0, area + L.out), D(sim, 0, area + L.in), 2);
}

static void id8(session * s) {
    tsr_sim * sim = s->sim;
    const uint64_t area = AREA0 + 8u * AREA_STEP;
    const uint32_t p[1] = { 1000u };
    put_params(sim, area, 8, p, 1);
    const uint32_t seed = 0x8D8A11u;
    if (!launch(s, 8, area, tsr_st8().end, seed, 0, 0)) return;
    uint8_t want[128];
    tsr_st8_pattern(seed, want);
    char what[64];
    snprintf(what, sizeof(what), "[%s] id 8: stored pattern", s->sched);
    cmp_lines(what, D(sim, 0, area + tsr_st8().out), want, 2);
}

// The NUL-terminated string at VMA va of the ELF's .tsr_logfmt (address 0xF000_0000 + offset).
static const char * logfmt(const fwh * h, uint32_t va) {
    uint32_t off = 0, size = 0;
    if (fwh_elf_section(h->image, h->image_n, ".tsr_logfmt", &off, &size)) return NULL;
    if (va < 0xF0000000u || va - 0xF0000000u >= size) return NULL;
    return (const char *)h->image + off + (va - 0xF0000000u);
}

static void id9(session * s) {
    tsr_sim * sim = s->sim;
    const uint64_t area = AREA0 + 9u * AREA_STEP;
    put_params(sim, area, 9, NULL, 0);
    const uint32_t seed = 0x9D9C0DEu;
    if (!launch(s, 9, area, 1, seed, 0, 0)) return;
    const fwh * h = &s->h;
    int ev = 0;
    for (uint32_t i = 0; i < h->n_ev; ++i) {
        const uint32_t * e = h->ev[i];
        if (TSR_GET(e[0], TSR_LCQ_R0_TYPE) != TSR_ST9_EVENT) continue;
        ev = 1;
        CHECK(TSR_GET(e[0], TSR_LCQ_R0_STATUS) == (seed & 0xFFu) && TSR_GET(e[0], TSR_LCQ_R0_ERR_CODE) == ((seed >> 8) & 0xFFu) &&
              e[1] == h->done[1] && e[2] == seed && e[3] == ~seed, "[%s] id 9: event 0x%08x 0x%08x 0x%08x 0x%08x", s->sched,
              e[0], e[1], e[2], e[3]);
    }
    CHECK(ev, "[%s] id 9: no kernel event 0x%02x", s->sched, TSR_ST9_EVENT);
    // The two FMT records of SOURCE kernel.
    int found = 0;
    for (uint32_t i = 0; i < h->n_log;) {
        const uint32_t hd = h->log[i], n = TSR_GET(hd, TSR_LOG_HDR_N);
        if (TSR_GET(hd, TSR_LOG_HDR_SYNC) != TSR_LOG_HDR_SYNC_MAGIC || i + 1u + n > h->n_log) {
            CHECK(0, "[%s] id 9: log word %u 0x%08x is not a record header", s->sched, i, hd);
            break;
        }
        const uint32_t * pl = h->log + i + 1;
        if (TSR_GET(hd, TSR_LOG_HDR_SOURCE) == TSR_LOG_HDR_SOURCE_KERNEL && TSR_GET(hd, TSR_LOG_HDR_TYPE) == TSR_LOG_HDR_TYPE_FMT) {
            const uint32_t ts = TSR_GET(hd, TSR_LOG_HDR_TS) ? 1u : 0u;
            const char * f = n > ts ? logfmt(h, pl[ts]) : NULL;
            CHECK(TSR_GET(hd, TSR_LOG_HDR_LSEQ) == (h->done[1] & 0xFFu), "[%s] id 9: log LSEQ", s->sched);
            if (found == 0) {
                CHECK(f && !strcmp(f, "selftest 9: seed %x, id %u") && TSR_GET(hd, TSR_LOG_HDR_LEVEL) == TSR_LOG_HDR_LEVEL_INFO &&
                      n == ts + 3u && pl[ts + 1] == seed && pl[ts + 2] == 9u, "[%s] id 9: first record (%s)", s->sched, f ? f : "?");
            } else if (found == 1) {
                CHECK(f && !strcmp(f, "selftest 9: %u %x") && TSR_GET(hd, TSR_LOG_HDR_LEVEL) == TSR_LOG_HDR_LEVEL_WARN &&
                      n == ts + 3u && pl[ts + 1] == ~seed && pl[ts + 2] == seed + 1u, "[%s] id 9: second record (%s)", s->sched,
                      f ? f : "?");
            }
            ++found;
        }
        i += 1u + n;
    }
    CHECK(found == 2, "[%s] id 9: %d kernel FMT records, expected 2", s->sched, found);
}

static void id10(session * s) {
    tsr_sim * sim = s->sim;
    const uint64_t area = AREA0 + 10u * AREA_STEP;
    rng_t r = { 0xAD10 };
    const uint32_t w = weights(sim, &r, TSR_FMT_Q8_0, 16, 32, W_BASE + 0x40000u);
    const uint32_t p[1] = { w };
    put_params(sim, area, 10, p, 1);
    rand_f32(&r, D(sim, 0, area + 1), 32, 1.0);
    launch(s, 10, area, tsr_st10().end, 0, 0, 0);
}

// Misuses: unknown id, wrong magic, an area too small (KERNEL_ERR: RET negative).
static void misuse(session * s) {
    const uint64_t area = AREA0 + 11u * AREA_STEP;
    launch(s, 11, area, 1, 0, (int32_t)TSR_ST_RET_BAD_ID, TSR_LCQ_R0_STATUS_KERNEL_ERR);
    put_params(s->sim, area, 3, NULL, 0);
    launch(s, 2, area, 100, 0, (int32_t)TSR_ST_RET_BAD_PARAMS, TSR_LCQ_R0_STATUS_KERNEL_ERR);
    const uint32_t p[2] = { 1024, 8 };
    put_params(s->sim, area, 2, p, 2);
    launch(s, 2, area, 10, 0, (int32_t)TSR_ST_RET_AREA, TSR_LCQ_R0_STATUS_KERNEL_ERR);
    // A device-side check failing: id 7 with an in-range "bad" beat finds no DRAM_RANGE (RET 1, OK).
    const uint32_t p7[2] = { (uint32_t)AREA0, W_BASE + 0x30000u };
    put_params(s->sim, area, 7, p7, 2);
    launch(s, 7, area, tsr_st7().end, 0, 1, TSR_LCQ_R0_STATUS_OK);
    // id 8 with its SPM lines in the runtime's argument range.
    const uint32_t p8[1] = { 65530u };
    put_params(s->sim, area, 8, p8, 1);
    launch(s, 8, area, tsr_st8().end, 0, (int32_t)TSR_ST_RET_BAD_PARAMS, TSR_LCQ_R0_STATUS_KERNEL_ERR);
}

// Every id in fw_selftest_sim's order: 1, 2 (K 1024 T 8; K 4096 T 1), 3-10, then the misuses.
static void st_run_ids(session * s) {
    id1(s);
    id2(s, 1024, 8, 2);
    id2(s, 4096, 1, 12);
    id3(s);
    id4(s);
    id5(s);
    id6(s);
    id7(s);
    id8(s);
    id9(s);
    id10(s);
    misuse(s);
}
