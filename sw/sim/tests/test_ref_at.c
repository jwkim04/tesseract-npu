// test_ref_at.c — the AT references (sw/sim/tsr_ref_at.c; docs/v2/03-attention-kv.md §13.2).
//
// Usage: ref_at self | ggml [-v]
//   self  layout round trips and placement, KVST (bytes, RMW counts; with ggml: against
//         quantize_row_q8_0_ref), ATTQK (SEG vs tsr_dot_row and the short tree, bounds, GQA rows,
//         MAXOUT), ATTPV (independent __int128 / exact-rational model, directed cases, a
//         double-precision bound), the vector ops, MSUM, the chunk planner, and tsr_ref_fa against
//         an independent double-precision model with an error bound derived from AT §6
//   ggml  tsr_ref_fa against ggml-cpu's FLASH_ATTN_EXT on the same q8_0 cache (AT §6.7: KL-level
//         agreement, not bit identity): NMSE (gate 5e-4, the test-backend-ops tolerance AT §13.2
//         names), max abs / rel error, softmax-KL of the output rows, and both against the double
//         model; the device must stay within its derived bound and be no less precise than the CPU
//
// Independence: the models here do not call tsr_ref_at.c except for the function under test.
// f32 arithmetic in the models is double-rounded-once (innocuous for + - * / with 53 >= 2·24+2,
// as in test_ref_dot.c), exponents come from frexp, integer rounding from nearbyint (default
// round-to-nearest-even), sums from __int128, fp16 from ldexp. EXP is tsr_m_exp (VU §6.3 defines
// it; ref_math_* test it).

#include <float.h>

#include "test_util.h"
#include "tsr_ref.h"
#include "tsr_ref_at.h"
#include "tsr_ref_math.h"

#ifdef TSR_REF_AT_GGML
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"
_Static_assert(sizeof(block_q8_0) == sizeof(tsr_block_q8_0), "q8_0 layout");
#endif

static int g_verbose = 0;
#define U24 0x1p-24

// ---------------------------------------------------------------------------
// Independent f32 arithmetic with the spec rules (DAZ, RNE, FTZ, canonical NaN)
// ---------------------------------------------------------------------------
static float ind_in(float x) { return fpclassify(x) == FP_SUBNORMAL ? copysignf(0.0f, x) : x; }
static float ind_out(double r) {
    if (isnan(r)) return u2f(0x7FC00000u);
    const float f = (float)r;
    return fpclassify(f) == FP_SUBNORMAL ? copysignf(0.0f, f) : f;
}
static float ind_add(float a, float b) { return ind_out((double)ind_in(a) + (double)ind_in(b)); }
static float ind_sub(float a, float b) { return ind_out((double)ind_in(a) - (double)ind_in(b)); }
static float ind_mul(float a, float b) { return ind_out((double)ind_in(a) * (double)ind_in(b)); }
static float ind_rcp(float a) { return ind_out(1.0 / (double)ind_in(a)); }
static double ind_h2d(uint16_t h) {
    const int e = (h >> 10) & 31, m = h & 1023;
    double v;
    if (e == 0) v = ldexp((double)m, -24);
    else if (e == 31) v = m ? NAN : INFINITY;
    else v = ldexp((double)(1024 + m), e - 25);
    return (h & 0x8000) ? -v : v;
}
static float ind_h2f(uint16_t h) { return (float)ind_h2d(h); }
static float ind_maxt(float a, float b) {   // total order: -inf < ... < -0 < +0 < ... < +inf < NaN
    a = ind_in(a); b = ind_in(b);
    if (isnan(a) || isnan(b)) return u2f(0x7FC00000u);
    if (a == b) return signbit(a) ? b : a;
    return a > b ? a : b;
}
static int bits_eq(float a, float b) { return f2u(a) == f2u(b); }

// ---------------------------------------------------------------------------
// Random data
// ---------------------------------------------------------------------------
static uint16_t rand_fp16_scale(rng_t * r, int specials) {
    const uint32_t k = rng_below(r, 100);
    if (specials && k < 2) return 0x0000;
    if (specials && k < 3) return 0x8000;
    if (specials && k < 4) return 0x7C00;                        // +inf
    if (specials && k < 5) return 0x7E00;                        // NaN
    if (specials && k < 7) return (uint16_t)(1 + rng_below(r, 1023)); // fp16 subnormal
    const float v = (float)(ldexp(1.0 + rng_unit(r), (int)rng_below(r, 16) - 12));
    return (uint16_t)(tsr_f32_to_fp16(v) | ((rng_u32(r) & 1u) ? 0x8000u : 0u));
}
static void rand_q8_row(rng_t * r, tsr_block_q8_0 * row, uint32_t nb, int specials) {
    for (uint32_t b = 0; b < nb; ++b) {
        row[b].d = rand_fp16_scale(r, specials);
        for (int l = 0; l < 32; ++l) row[b].qs[l] = (int8_t)(rng_u32(r) >> 24);
        if (specials && rng_below(r, 20) == 0) for (int l = 0; l < 32; ++l) row[b].qs[l] = -128;
    }
}
static void rand_f32_row(rng_t * r, float * x, uint32_t n) {
    for (uint32_t b = 0; b < n / 32u; ++b) {
        const uint32_t kind = rng_below(r, 16);
        const double sc = ldexp(1.0, (int)rng_below(r, 40) - 20);
        for (uint32_t l = 0; l < 32; ++l) {
            float v = (float)(rng_normal(r) * sc);
            if (kind == 0) v = 0.0f;
            x[32u * b + l] = v;
        }
        if (kind == 1) x[32u * b + rng_below(r, 32)] = (float)(sc * 1000.0);
    }
}

// ---------------------------------------------------------------------------
// Planes
// ---------------------------------------------------------------------------
typedef struct planes {
    tsr_kv_geom g;
    size_t bytes;            // per channel
    uint8_t * ch[TSR_NUM_CH_MAX];
} planes;
static void planes_init(planes * p, uint32_t E, uint32_t C, uint32_t LG) {
    p->g = (tsr_kv_geom){ E, C, LG };
    p->bytes = (size_t)tsr_kv_plane_beats(&p->g) * 64u;
    for (uint32_t c = 0; c < C; ++c) p->ch[c] = (uint8_t *)calloc(p->bytes, 1);
}
static void planes_free(planes * p) { for (uint32_t c = 0; c < p->g.C; ++c) free(p->ch[c]); }
static void planes_copy(planes * dst, const planes * src) {
    planes_init(dst, src->g.E, src->g.C, src->g.LG);
    for (uint32_t c = 0; c < src->g.C; ++c) memcpy(dst->ch[c], src->ch[c], src->bytes);
}
static int planes_eq(const planes * a, const planes * b) {
    for (uint32_t c = 0; c < a->g.C; ++c) if (memcmp(a->ch[c], b->ch[c], a->bytes) != 0) return 0;
    return 1;
}

// AT §4.3 / §4.4 transcribed from the document text.
static void doc_addr(uint32_t kv, uint32_t E, uint32_t C, uint32_t LG, uint32_t j, uint32_t b,
                     uint32_t * ch, uint64_t * meta, uint64_t * quant) {
    const uint64_t u = j / 16u, r = j % 16u, l = u / C, GB = 17u * E / 64u;
    *ch = (uint32_t)(u % C);
    if (kv == TSR_KV_K) {
        const uint64_t beta = r * (E / 32u) + b, kappa = beta >> 5, o = beta & 31u;
        *meta  = (l * GB + kappa * 17u) * 64u + 2u * o;
        *quant = (l * GB + kappa * 17u + 1u + (o >> 1)) * 64u + 32u * (o & 1u);
    } else {
        const uint64_t s = b / 2u, e = b % 2u, chunk = (s * LG + l) * 17u;
        *meta  = chunk * 64u + 4u * r + 2u * e;
        *quant = (chunk + 1u + r) * 64u + 32u * e;
    }
}

// ---------------------------------------------------------------------------
// 1. Layout (AT §4, §13.2 "Layout")
// ---------------------------------------------------------------------------
static void test_layout_shape(rng_t * rng, uint32_t E, uint32_t kv_size, uint32_t C, uint32_t extra_lg) {
    const uint32_t LG = tsr_kv_lg_min(kv_size, C) + extra_lg;
    const uint32_t nb = E / 32u, rowb = nb * 34u;
    CHECK(LG >= 1 && 16ull * C * LG >= kv_size, "lg_min E %u kv %u C %u", E, kv_size, C);
    for (uint32_t kv = 0; kv < 2; ++kv) {
        planes P;
        planes_init(&P, E, C, LG);
        CHECK(P.bytes == (size_t)LG * 17u * E, "plane bytes per channel");
        const uint64_t cells = tsr_kv_plane_cells(&P.g);
        // Placement: every byte of the plane belongs to exactly one (cell, block, byte); the V
        // slice-s region of a channel holds only slice s.
        uint16_t * own[TSR_NUM_CH_MAX];
        for (uint32_t c = 0; c < C; ++c) own[c] = (uint16_t *)calloc(P.bytes, sizeof(uint16_t));
        long bad = 0;
        for (uint32_t j = 0; j < cells; ++j) {
            for (uint32_t b = 0; b < nb; ++b) {
                tsr_kv_loc loc;
                if (kv == TSR_KV_K) tsr_kv_addr_k(&P.g, j, b, &loc); else tsr_kv_addr_v(&P.g, j, b, &loc);
                uint32_t dch; uint64_t dm, dq;
                doc_addr(kv, E, C, LG, j, b, &dch, &dm, &dq);
                if (loc.ch != dch || loc.meta != dm || loc.quant != dq) ++bad;
                const uint16_t tag = (uint16_t)(kv == TSR_KV_V ? b / 2u + 1u : 1u);
                for (int k = 0; k < 34; ++k) {
                    const uint64_t off = k < 2 ? loc.meta + (uint64_t)k : loc.quant + (uint64_t)(k - 2);
                    if (off >= P.bytes || own[loc.ch][off] != 0) { ++bad; continue; }
                    own[loc.ch][off] = tag;
                }
            }
        }
        CHECK(bad == 0, "E %u kv %u C %u kv-plane %u: %ld placement errors", E, kv_size, C, kv, bad);
        long unc = 0, wrong_slice = 0;
        for (uint32_t c = 0; c < C; ++c) {
            for (size_t i = 0; i < P.bytes; ++i) {
                if (own[c][i] == 0) ++unc;
                if (kv == TSR_KV_V && own[c][i] != 0 && (uint32_t)own[c][i] != (uint32_t)(i / 64u / (17u * LG)) + 1u) ++wrong_slice;
            }
            free(own[c]);
        }
        CHECK(unc == 0, "E %u C %u: %ld plane bytes not covered by any cell", E, C, unc);
        CHECK(wrong_slice == 0, "E %u C %u: %ld V bytes outside their slice region", E, C, wrong_slice);
        // Round trip of kv_size rows of random bytes (incl. NaN / inf scales: a byte permutation).
        uint8_t * src = (uint8_t *)malloc((size_t)kv_size * rowb + 1u);
        uint8_t * back = (uint8_t *)malloc((size_t)cells * rowb + 1u);
        rng_bytes(rng, src, (size_t)kv_size * rowb);
        for (uint32_t j = 0; j < kv_size; ++j) tsr_kv_pack_row(kv, &P.g, P.ch, j, (const tsr_block_q8_0 *)(src + (size_t)j * rowb));
        for (uint32_t j = 0; j < cells; ++j) tsr_kv_unpack_row(kv, &P.g, (const uint8_t * const *)P.ch, j, (tsr_block_q8_0 *)(back + (size_t)j * rowb));
        CHECK(memcmp(src, back, (size_t)kv_size * rowb) == 0, "E %u kv %u C %u: pack/unpack round trip", E, kv_size, C);
        long nz = 0;
        for (size_t i = (size_t)kv_size * rowb; i < (size_t)cells * rowb; ++i) nz += back[i] != 0;
        CHECK(nz == 0, "cells past kv_size not zero after pack (%ld)", nz);
        // set_bytes in random pieces equals pack_row; get_bytes of random ranges equals the source.
        planes Q;
        planes_init(&Q, E, C, LG);
        for (size_t off = 0; off < (size_t)kv_size * rowb;) {
            size_t n = 1u + rng_below(rng, 3u * rowb);
            if (n > (size_t)kv_size * rowb - off) n = (size_t)kv_size * rowb - off;
            tsr_kv_set_bytes(kv, &Q.g, Q.ch, src + off, off, n);
            off += n;
        }
        CHECK(planes_eq(&P, &Q), "E %u kv %u C %u: set_bytes differs from pack_row", E, kv_size, C);
        for (int it = 0; it < 50; ++it) {
            const size_t off = rng_below(rng, kv_size * rowb);
            size_t n = 1u + rng_below(rng, 2u * rowb + 7u);
            if (n > (size_t)kv_size * rowb - off) n = (size_t)kv_size * rowb - off;
            memset(back, 0x5A, n);
            tsr_kv_get_bytes(kv, &P.g, (const uint8_t * const *)P.ch, back, off, n);
            CHECK(memcmp(back, src + off, n) == 0, "get_bytes [%zu, +%zu)", off, n);
        }
        planes_free(&Q);
        free(src); free(back);
        planes_free(&P);
    }
}

static void test_layout(rng_t * rng) {
    // Model shapes (AT §4.5): Qwen2.5-0.5B (D 64, 2 KV heads), Qwen2.5-7B (128, 4), Qwen2.5-32B
    // (128, 8: one chunk per row), Gemma 4 12B full layers (512, 1: two rows per chunk) and SWA
    // layers (256, 8; 1,536-cell cache).
    static const struct { uint32_t E, kv; } shapes[] = {
        { 128, 4096 }, { 512, 4096 }, { 1024, 4096 }, { 512, 2048 }, { 2048, 1536 },
    };
    static const uint32_t Cs[] = { 1, 2, 3, 4, 16 };
    for (size_t i = 0; i < sizeof shapes / sizeof shapes[0]; ++i) {
        for (size_t c = 0; c < sizeof Cs / sizeof Cs[0]; ++c) {
            if (Cs[c] == 16 && shapes[i].E > 1024) continue;   // keep the run short
            test_layout_shape(rng, shapes[i].E, shapes[i].kv, Cs[c], c == 2 ? 1u : 0u);
        }
    }
    for (int it = 0; it < 24; ++it) {
        const uint32_t E = 64u * (1u + rng_below(rng, 64));
        const uint32_t kv = 1u + rng_below(rng, 1500);
        test_layout_shape(rng, E, kv, Cs[rng_below(rng, 5)], rng_below(rng, 2));
    }
    // Directed: E = 1024 -> a cell row is exactly one chunk at beat l·GB + 17 r; E = 512 -> two
    // rows share a chunk; V slice s of group l at (s·LG + l)·17.
    {
        const tsr_kv_geom g = { 1024, 4, 3 };
        tsr_kv_loc loc;
        tsr_kv_addr_k(&g, 16u * 9u + 5u, 0, &loc);   // u = 9: ch 1, l = 2
        CHECK(loc.ch == 1 && loc.meta == (2u * 17u * 16u + 17u * 5u) * 64u, "K E=1024 row chunk");
        tsr_kv_addr_k(&g, 16u * 9u + 5u, 31, &loc);
        CHECK(loc.meta == (2u * 17u * 16u + 17u * 5u) * 64u + 62u && loc.quant == (2u * 17u * 16u + 17u * 5u + 16u) * 64u + 32u, "K last block");
        const tsr_kv_geom h = { 512, 4, 3 };
        tsr_kv_addr_k(&h, 1, 0, &loc);
        CHECK(loc.meta == 32u && loc.quant == 9u * 64u, "K E=512: cell 1 shares chunk 0 (meta byte 32, quant beat 9)");
        tsr_kv_addr_v(&h, 16u * 6u + 7u, 5, &loc);   // u = 6: ch 2, l = 1; slice 2
        CHECK(loc.ch == 2 && loc.meta == ((2u * 3u + 1u) * 17u) * 64u + 28u + 2u && loc.quant == ((2u * 3u + 1u) * 17u + 8u) * 64u + 32u, "V slice placement");
    }
}

// ---------------------------------------------------------------------------
// 2. KVST (AT §5.1)
// ---------------------------------------------------------------------------
static void test_kvst_case(rng_t * rng, uint32_t kv, uint32_t E, uint32_t C, uint32_t LG, uint32_t c0, uint32_t n_rows,
                           uint32_t stride_extra) {
    planes P, X;
    planes_init(&P, E, C, LG);
    for (uint32_t c = 0; c < C; ++c) rng_bytes(rng, P.ch[c], P.bytes);
    planes_copy(&X, &P);
    const uint32_t ql = tsr_qa_lines(TSR_QA_A32, E);
    const uint32_t stride = stride_extra ? ql + stride_extra : 0u;
    const size_t sb = (size_t)(stride ? stride : ql) * 64u;
    uint8_t * src = (uint8_t *)calloc((size_t)n_rows * sb + 64u, 1);
    float * x = (float *)malloc((size_t)E * sizeof(float));
    const uint32_t nb = E / 32u;
    tsr_block_q8_0 * row = (tsr_block_q8_0 *)malloc((size_t)nb * sizeof *row);
    long ggml_mismatch = 0, d_inexact = 0;
    for (uint32_t i = 0; i < n_rows; ++i) {
        rand_f32_row(rng, x, E);
        tsr_qact(TSR_QA_A32, x, E, src + (size_t)i * sb);
        const uint8_t * qa = src + (size_t)i * sb;
        const float * D = tsr_qa_d(TSR_QA_A32, E, qa);
        for (uint32_t b = 0; b < nb; ++b) {     // the expected bytes: fp16(D) and QS
            const uint16_t h = tsr_f32_to_fp16(D[b]);
            if (!bits_eq(ind_h2f(h), D[b])) ++d_inexact;
            row[b].d = h;
            memcpy(row[b].qs, qa + 32u * b, 32);
        }
#ifdef TSR_REF_AT_GGML
        block_q8_0 * gr = (block_q8_0 *)malloc((size_t)nb * sizeof(block_q8_0));
        quantize_row_q8_0_ref(x, gr, E);
        if (memcmp(gr, row, (size_t)nb * sizeof(block_q8_0)) != 0) ++ggml_mismatch;
        free(gr);
#endif
        tsr_kv_pack_row(kv, &X.g, X.ch, c0 + i, row);
    }
    tsr_kvst_stats st;
    tsr_ref_kvst(kv, &P.g, P.ch, c0, n_rows, src, stride, &st);
    CHECK(planes_eq(&P, &X), "KVST kv %u E %u C %u c0 %u n %u: plane bytes", kv, E, C, c0, n_rows);
    CHECK(d_inexact == 0, "fp16(D) not exact for QACT output (%ld)", d_inexact);
    CHECK(ggml_mismatch == 0, "KVST rows differ from quantize_row_q8_0_ref (%ld rows)", ggml_mismatch);
    // RMW counts by brute force: slots written per meta beat (each meta beat has 32 slots).
    uint8_t * cnt[TSR_NUM_CH_MAX];
    uint8_t * qw[TSR_NUM_CH_MAX];
    const size_t beats = P.bytes / 64u;
    for (uint32_t c = 0; c < C; ++c) { cnt[c] = (uint8_t *)calloc(beats, 1); qw[c] = (uint8_t *)calloc(beats, 1); }
    for (uint32_t i = 0; i < n_rows; ++i) {
        for (uint32_t b = 0; b < nb; ++b) {
            uint32_t ch; uint64_t m, q;
            doc_addr(kv, E, C, LG, c0 + i, b, &ch, &m, &q);
            cnt[ch][m / 64u] += 1u;
            qw[ch][q / 64u] = 1u;
        }
    }
    for (uint32_t c = 0; c < C; ++c) {
        uint32_t mb = 0, rmw = 0, qb = 0;
        for (size_t k = 0; k < beats; ++k) {
            if (cnt[c][k]) { ++mb; if (cnt[c][k] < 32u) ++rmw; }
            qb += qw[c][k];
        }
        CHECK(st.meta_beats[c] == mb && st.meta_rmw[c] == rmw && st.quant_beats[c] == qb,
              "KVST stats ch %u: meta %u/%u rmw %u/%u quant %u/%u (kv %u E %u c0 %u n %u)", c, st.meta_beats[c], mb,
              st.meta_rmw[c], rmw, st.quant_beats[c], qb, kv, E, c0, n_rows);
        free(cnt[c]); free(qw[c]);
    }
    free(row); free(x); free(src);
    planes_free(&P); planes_free(&X);
}

static void test_kvst(rng_t * rng) {
    // Directed: E = 1024 K (full meta beats, no RMW), E = 512 K (RMW unless both rows of a chunk),
    // V aligned 16-cell groups (no RMW) and partial groups (RMW), rows spanning chunks (E = 1536).
    test_kvst_case(rng, TSR_KV_K, 1024, 4, 2, 37, 9, 0);
    test_kvst_case(rng, TSR_KV_K, 512, 4, 2, 32, 2, 3);
    test_kvst_case(rng, TSR_KV_K, 512, 4, 2, 33, 1, 0);
    test_kvst_case(rng, TSR_KV_V, 256, 3, 2, 48, 16, 0);
    test_kvst_case(rng, TSR_KV_V, 256, 3, 3, 50, 64, 1);
    test_kvst_case(rng, TSR_KV_K, 1536, 2, 2, 3, 5, 0);
    test_kvst_case(rng, TSR_KV_K, 2048, 4, 1, 0, 64, 0);
    for (int it = 0; it < 30; ++it) {
        const uint32_t E = 64u * (1u + rng_below(rng, 40));
        const uint32_t C = 1u + rng_below(rng, 5);
        const uint32_t LG = 1u + rng_below(rng, 3);
        const uint32_t cells = 16u * C * LG;
        const uint32_t n = 1u + rng_below(rng, cells < 64u ? cells : 64u);
        const uint32_t c0 = rng_below(rng, cells - n + 1u);
        test_kvst_case(rng, rng_below(rng, 2), E, C, LG, c0, n, rng_below(rng, 3));
    }
    // Single-row counts from the formulas: one decode token writes E/64 quant beats; K E = 1024
    // writes one full meta beat; V writes E/64 meta beats, all RMW.
    tsr_kvst_stats st;
    const tsr_kv_geom g = { 1024, 4, 4 };
    tsr_kvst_count(TSR_KV_K, &g, 70, 1, &st);
    CHECK(st.meta_beats[0] == 1 && st.meta_rmw[0] == 0 && st.quant_beats[0] == 16, "K decode token counts");
    tsr_kvst_count(TSR_KV_V, &g, 70, 1, &st);
    CHECK(st.meta_beats[0] == 16 && st.meta_rmw[0] == 16 && st.quant_beats[0] == 16, "V decode token counts");
}

// ---------------------------------------------------------------------------
// 3. ATTQK (AT §5.2, §6.2)
// ---------------------------------------------------------------------------
// SEG through tsr_dot_row on the segment repacked as a standalone Q8_0 row (AT §13.2).
static float seg_via_dot_row(uint32_t D, const tsr_block_q8_0 * kseg, const int8_t * qx, const float * dx) {
    uint8_t row[17u * 64u];
    uint8_t qa[(8u + 1u) * 64u];
    memset(qa, 0, sizeof qa);
    tsr_repack_row(TSR_FMT_Q8_0, kseg, D, row);
    memcpy(qa, qx, D);
    memcpy(qa + (size_t)tsr_qa_qs_lines(TSR_QA_A32, D) * 64u, dx, (D / 32u) * sizeof(float));
    return tsr_dot_row(TSR_FMT_Q8_0, row, D, qa);
}
// z(pairwise tree of t_0 .. t_(n-1)) for n = D/32 = 2^m (AT §6.2's short tree), independent ops.
static float seg_short_tree(uint32_t D, const tsr_block_q8_0 * kseg, const int8_t * qx, const float * dx) {
    float t[16];
    const uint32_t n = D / 32u;
    for (uint32_t k = 0; k < n; ++k) {
        int32_t s = 0;
        for (int l = 0; l < 32; ++l) s += kseg[k].qs[l] * qx[32u * k + (uint32_t)l];
        t[k] = ind_mul(ind_mul(ind_h2f(kseg[k].d), dx[k]), (float)s);
    }
    for (uint32_t w = n; w > 1u; w /= 2u) for (uint32_t k = 0; k < w / 2u; ++k) t[k] = ind_add(t[2u * k], t[2u * k + 1u]);
    return f2u(t[0]) == 0x80000000u ? 0.0f : t[0];
}

static void test_seg(rng_t * rng) {
    tsr_block_q8_0 k[16];
    int8_t qx[512];
    float dx[16];
    static const uint32_t dir[] = { 0x00000000u, 0x80000000u, 0x7F800000u, 0xFF800000u, 0x7FC00000u, 0x00800000u,
                                    0x80800000u, 0x80C00000u, 0x00400000u, 0x7F7FFFFFu, 0xFF7FFFFFu, 0x3F800000u,
                                    0x00800001u, 0x80800001u, 0x33800000u, 0xB3800000u };
    long bad1 = 0, bad2 = 0;
    for (int it = 0; it < 40000; ++it) {
        const uint32_t D = 64u << rng_below(rng, 4), n = D / 32u;
        const int directed = it % 2;
        rand_q8_row(rng, k, n, it % 3 == 0);
        for (uint32_t i = 0; i < D; ++i) qx[i] = (int8_t)(rng_u32(rng) >> 24);
        for (uint32_t b = 0; b < n; ++b) {
            dx[b] = (float)ldexp(1.0 + rng_unit(rng), (int)rng_below(rng, 60) - 30) * ((rng_u32(rng) & 1u) ? -1.0f : 1.0f);
            if (directed) {
                // t_b = directed value exactly: d = 1, isum = 1, Dx = the value (DAZ applies)
                k[b].d = 0x3C00;
                memset(k[b].qs, 0, 32);
                memset(qx + 32u * b, 0, 32);
                k[b].qs[0] = 1;
                qx[32u * b] = 1;
                dx[b] = u2f(dir[rng_below(rng, 16)]);
                if (rng_below(rng, 4) == 0) dx[b] = u2f(rng_u32(rng));
            }
        }
        const float a = tsr_att_seg(D, k, qx, dx);
        const float b1 = seg_via_dot_row(D, k, qx, dx);
        const float b2 = seg_short_tree(D, k, qx, dx);
        if (!bits_eq(a, b1)) { if (bad1++ < 5) fprintf(stderr, "SEG vs dot_row D %u: %08x %08x\n", D, f2u(a), f2u(b1)); }
        if (!bits_eq(a, b2)) { if (bad2++ < 5) fprintf(stderr, "SEG vs short tree D %u: %08x %08x\n", D, f2u(a), f2u(b2)); }
    }
    CHECK(bad1 == 0, "SEG differs from tsr_dot_row on the repacked segment (%ld)", bad1);
    CHECK(bad2 == 0, "SEG differs from z(pairwise tree) (%ld)", bad2);
}

#define SENT 0xDEADBEEFu

static void test_attqk_case(rng_t * rng, const tsr_att_shape * sh, uint32_t C, uint32_t u0, uint32_t n_groups,
                            uint32_t c_lo, uint32_t c_hi, uint32_t flags, uint32_t rstride) {
    const uint32_t E = sh->E, D = sh->D, nb = E / 32u, n_head = tsr_att_n_head(sh), R = tsr_att_R(sh), K = n_head * D;
    const uint32_t LG = (u0 + n_groups - 1u) / C + 1u;
    planes P;
    planes_init(&P, E, C, LG);
    const uint32_t cells = 16u * C * LG;
    tsr_block_q8_0 * rows = (tsr_block_q8_0 *)malloc((size_t)cells * nb * sizeof *rows);
    for (uint32_t j = 0; j < cells; ++j) {
        rand_q8_row(rng, rows + (size_t)j * nb, nb, rng_below(rng, 8) == 0);
        tsr_kv_pack_row(TSR_KV_K, &P.g, P.ch, j, rows + (size_t)j * nb);
    }
    const size_t xl = (size_t)tsr_qa_lines(TSR_QA_A32, K) * 64u;
    uint8_t * x = (uint8_t *)calloc((size_t)sh->T_q * xl, 1);
    float * xf = (float *)malloc((size_t)K * sizeof(float));
    for (uint32_t q = 0; q < sh->T_q; ++q) { rand_f32_row(rng, xf, K); tsr_qact(TSR_QA_A32, xf, K, x + q * xl); }
    const uint32_t rs = rstride ? rstride : n_groups;
    const size_t dn = (size_t)R * rs * 16u;
    float * dst = (float *)malloc(dn * sizeof(float));
    float mdst[1024 + 16];
    for (size_t i = 0; i < dn; ++i) dst[i] = u2f(SENT);
    for (size_t i = 0; i < sizeof mdst / sizeof mdst[0]; ++i) mdst[i] = u2f(SENT);
    tsr_ref_attqk(sh, C, (const uint8_t * const *)P.ch, u0, n_groups, c_lo, c_hi, flags, x, dst, rstride, mdst);
    // Expected.
    uint8_t * pass = (uint8_t *)calloc(R, 1);
    long bad = 0, badm = 0;
    for (uint32_t q = 0; q < sh->T_q; ++q) {
        const int64_t dq = (int64_t)sh->T_q - 1 - q;
        int64_t lo = (int64_t)c_lo - ((flags & TSR_ATT_SLIDE) ? dq : 0);
        if (lo < 0) lo = 0;
        const int64_t hi = (int64_t)c_hi - ((flags & TSR_ATT_CAUSAL) ? dq : 0);
        const int8_t * qs = (const int8_t *)(x + q * xl);
        const float * qd = tsr_qa_d(TSR_QA_A32, K, x + q * xl);
        for (uint32_t g = 0; g < E / D; ++g) {
            for (uint32_t i = 0; i < sh->GS; ++i) {
                const uint32_t r = q * n_head + g * sh->G + sh->hg0 + i, h = g * sh->G + sh->hg0 + i;
                pass[r] = 1;
                float mx = u2f(TSR_M_FLOOR);
                for (uint32_t j = 0; j < 16u * n_groups; ++j) {
                    float e = u2f(0xFF800000u);
                    if ((int64_t)j >= lo && (int64_t)j < hi) {
                        const tsr_block_q8_0 * kr = rows + (size_t)(16u * u0 + j) * nb + (size_t)g * (D / 32u);
                        e = seg_via_dot_row(D, kr, qs + (size_t)h * D, qd + (size_t)h * (D / 32u));
                    }
                    mx = ind_maxt(mx, e);
                    if (!bits_eq(dst[(size_t)r * rs * 16u + j], e)) ++bad;
                }
                for (uint32_t j = 16u * n_groups; j < 16u * rs; ++j) bad += f2u(dst[(size_t)r * rs * 16u + j]) != SENT;
                if (flags & TSR_ATT_MAXOUT) badm += !bits_eq(mdst[r], mx);
            }
        }
    }
    for (uint32_t r = 0; r < R; ++r) {
        if (!pass[r]) for (uint32_t j = 0; j < 16u * rs; ++j) bad += f2u(dst[(size_t)r * rs * 16u + j]) != SENT;
        if (!pass[r] || !(flags & TSR_ATT_MAXOUT)) badm += f2u(mdst[r]) != SENT;
    }
    for (uint32_t r = R; r < sizeof mdst / sizeof mdst[0]; ++r) badm += f2u(mdst[r]) != SENT;
    CHECK(bad == 0 && badm == 0, "ATTQK E %u D %u G %u hg0 %u GS %u Tq %u C %u u0 %u n %u [%u,%u) fl %u: %ld score, %ld max errors",
          E, D, sh->G, sh->hg0, sh->GS, sh->T_q, C, u0, n_groups, c_lo, c_hi, flags, bad, badm);
    free(pass); free(dst); free(xf); free(x); free(rows);
    planes_free(&P);
}

static void test_attqk(rng_t * rng) {
    // GQA heads as slots; prefill bounds; head subsets (G = 16 > T_MAX: two passes).
    const tsr_att_shape s1 = { 128, 64, 7, 0, 7, 1 };   // Qwen2.5-0.5B decode
    test_attqk_case(rng, &s1, 4, 3, 5, 0, 80, TSR_ATT_MAXOUT, 0);
    const tsr_att_shape s2 = { 512, 512, 16, 8, 8, 1 }; // Gemma 4 12B full layer, second pass
    test_attqk_case(rng, &s2, 4, 0, 2, 3, 29, TSR_ATT_MAXOUT, 3);
    const tsr_att_shape s3 = { 512, 256, 2, 0, 2, 4 };  // Gemma SWA prefill tile, CAUSAL + SLIDE
    test_attqk_case(rng, &s3, 3, 5, 4, 20, 50, TSR_ATT_MAXOUT | TSR_ATT_CAUSAL | TSR_ATT_SLIDE, 0);
    const tsr_att_shape s4 = { 256, 128, 1, 0, 1, 8 };  // T_q 8, CAUSAL bounds below zero
    test_attqk_case(rng, &s4, 1, 0, 1, 0, 4, TSR_ATT_MAXOUT | TSR_ATT_CAUSAL, 0);
    test_attqk_case(rng, &s4, 2, 1, 2, 0, 0, TSR_ATT_MAXOUT, 0);   // empty bounds: all -inf, max M_FLOOR
    for (int it = 0; it < 60; ++it) {
        tsr_att_shape s;
        s.D = 64u << rng_below(rng, 4);
        s.E = s.D * (1u + rng_below(rng, s.D >= 256 ? 2u : 4u));
        s.G = 1u + rng_below(rng, it % 5 == 0 ? 20u : 8u);
        s.GS = 1u + rng_below(rng, s.G < 8u ? s.G : 8u);
        s.hg0 = rng_below(rng, s.G - s.GS + 1u);
        s.T_q = 1u + rng_below(rng, 8u / s.GS);
        const uint32_t C = 1u + rng_below(rng, 4), n = 1u + rng_below(rng, 5), u0 = rng_below(rng, 9);
        const uint32_t c_hi = rng_below(rng, 16u * n + 1u), c_lo = rng_below(rng, c_hi + 1u);
        test_attqk_case(rng, &s, C, u0, n, c_lo, c_hi, rng_below(rng, 8), rng_below(rng, 2) ? n + rng_below(rng, 3) : 0u);
    }
}

// ---------------------------------------------------------------------------
// 4. ATTPV (AT §5.3, §6.4, §6.5): independent model
// ---------------------------------------------------------------------------
static float ind_F(__int128 A, int k) {   // RNE24(A)·2^k, |result| < 2^-126 -> signed zero, >= 2^128 -> inf
    if (A == 0) return 0.0f;
    const double ad = (double)A;            // exact: |A| < 2^53
    const float f = (float)ad;              // one RNE to 24 bits
    const double v = ldexp((double)f, k);
    if (fabs(v) < 0x1p-126) return A < 0 ? -0.0f : 0.0f;
    if (fabs(v) >= 0x1p128) return A < 0 ? -INFINITY : INFINITY;
    return (float)v;
}
static int ind_eps(float w) { int e; frexp((double)w, &e); return e - 1; }

// Lanes over AT §13.2's stated bound (segments + 1)·2^-24·sum|p·v| (reported, not gated: the
// bound lacks the 24-bit alignment term 2^(E_S-24)·sum|q| of a lane whose large-weight cells have
// small quants; ind_attpv's bound includes it and is gated).
static long g_doc_bound_over = 0, g_doc_bound_lanes = 0;

// O[c][r][d] and l[c][r] for the pass rows (other entries untouched), plus a per-element bound
// against the exact sum (bnd[c][r][d]: error bound; exact[c][r][d]).
static void ind_attpv(const tsr_att_shape * sh, uint32_t C, uint32_t u0, uint32_t n_groups, int lsum,
                      const tsr_block_q8_0 * vrows, const float * p, uint32_t ps,
                      float * O, float * l, double * exact, double * bnd) {
    const uint32_t D = sh->D, nb = sh->E / 32u, n_head = tsr_att_n_head(sh), R = tsr_att_R(sh);
    for (uint32_t c = 0; c < C; ++c) {
        uint32_t U[4096];
        uint32_t nu = 0;
        for (uint32_t u = u0; u < u0 + n_groups; ++u) if (u % C == c) U[nu++] = u;
        for (uint32_t q = 0; q < sh->T_q; ++q) for (uint32_t g = 0; g < sh->E / D; ++g) for (uint32_t i = 0; i < sh->GS; ++i) {
            const uint32_t r = q * n_head + g * sh->G + sh->hg0 + i;
            const float * pr = p + (size_t)r * ps * 16u;
            for (uint32_t d = 0; d < D; ++d) {
                const uint32_t b = g * (D / 32u) + d / 32u;     // block of the cell row
                const uint32_t s = b / 2u;                      // slice; the other block of the slice is b ^ 1
                float o = 0.0f;
                double ex = 0.0, eb = 0.0, pvtot = 0.0;
                for (uint32_t k0 = 0; k0 < nu; k0 += 4u) {
                    int nonfin[2] = { 0, 0 }, any = 0, es = -100000;
                    double wmax = 0.0;
                    for (uint32_t k = k0; k < nu && k < k0 + 4u; ++k) for (uint32_t rr = 0; rr < 16u; ++rr) {
                        const uint32_t j = 16u * U[k] + rr;
                        const float pj = ind_in(pr[j - 16u * u0]);
                        if (pj == 0.0f) continue;
                        for (uint32_t e = 0; e < 2; ++e) {
                            const float w = ind_mul(pj, ind_h2f(vrows[(size_t)(j - 16u * u0) * nb + 2u * s + e].d));
                            if (!isfinite(w)) nonfin[e] = 1;
                            else if (w != 0.0f) { any = 1; if (ind_eps(w) > es) es = ind_eps(w); if (fabs((double)w) > wmax) wmax = fabs((double)w); }
                        }
                    }
                    const uint32_t e = b & 1u;
                    float cval;
                    __int128 A = 0;
                    double qabs = 0.0, pvabs = 0.0;
                    for (uint32_t k = k0; k < nu && k < k0 + 4u; ++k) for (uint32_t rr = 0; rr < 16u; ++rr) {
                        const uint32_t j = 16u * U[k] + rr;
                        const float pj = ind_in(pr[j - 16u * u0]);
                        if (pj == 0.0f) continue;
                        const tsr_block_q8_0 * vb = &vrows[(size_t)(j - 16u * u0) * nb + b];
                        const float w = ind_mul(pj, ind_h2f(vb->d));
                        const int qv = vb->qs[d % 32u];
                        if (any && !nonfin[e]) A += (__int128)(int64_t)nearbyint(ldexp((double)w, 23 - es)) * qv;
                        ex += (double)pj * ind_h2d(vb->d) * qv;
                        qabs += fabs((double)qv);
                        pvabs += fabs((double)pj * ind_h2d(vb->d) * qv);
                    }
                    if (nonfin[e]) cval = u2f(0x7FC00000u);
                    else if (!any) cval = 0.0f;
                    else cval = ind_F(A, es - 23);
                    o = ind_add(o, cval);
                    pvtot += pvabs;
                    // per segment: w rounding, alignment (half a unit of 2^(E_S-23) per cell), RNE24, FTZ
                    eb += U24 * pvabs + U24 * 2.0 * wmax * qabs + U24 * fabs((double)cval) + 0x1p-126;
                }
                const uint32_t nseg = (nu + 3u) / 4u;
                eb += (double)nseg * U24 * fabs((double)o) * 2.0;
                const size_t ix = ((size_t)c * R + r) * D + d;
                if (isfinite(o) && isfinite(ex)) {
                    ++g_doc_bound_lanes;
                    g_doc_bound_over += fabs((double)o - ex) > ((double)nseg + 1.0) * U24 * pvtot;
                }
                O[ix] = o;
                exact[ix] = ex;
                bnd[ix] = eb;
            }
            if (lsum) {
                float lv = 0.0f;
                for (uint32_t k0 = 0; k0 < nu; k0 += 4u) {
                    int nonfin = 0, any = 0, ep = -100000;
                    for (uint32_t k = k0; k < nu && k < k0 + 4u; ++k) for (uint32_t rr = 0; rr < 16u; ++rr) {
                        const float pj = ind_in(pr[16u * (U[k] - u0) + rr]);
                        if (!isfinite(pj)) nonfin = 1;
                        else if (pj != 0.0f) { any = 1; if (ind_eps(pj) > ep) ep = ind_eps(pj); }
                    }
                    float cl;
                    if (nonfin) cl = u2f(0x7FC00000u);
                    else if (!any) cl = 0.0f;
                    else {
                        __int128 S = 0;
                        for (uint32_t k = k0; k < nu && k < k0 + 4u; ++k) for (uint32_t rr = 0; rr < 16u; ++rr) {
                            const float pj = ind_in(pr[16u * (U[k] - u0) + rr]);
                            S += (__int128)(int64_t)nearbyint(ldexp((double)pj, 23 - ep));
                        }
                        cl = ind_F(S, ep - 23);
                    }
                    lv = ind_add(lv, cl);
                }
                l[(size_t)c * R + r] = lv;
            }
        }
    }
}

typedef enum { P_SOFT, P_SPECIAL, P_ZERO, P_EXTREME } p_kind;

static void test_attpv_case(rng_t * rng, const tsr_att_shape * sh, uint32_t C, uint32_t u0, uint32_t n_groups,
                            uint32_t flags, p_kind pk, int vspecial, uint32_t pstride_extra, uint32_t cstride_extra) {
    const uint32_t E = sh->E, D = sh->D, nb = E / 32u, R = tsr_att_R(sh), Rl = (R + 15u) / 16u;
    const uint32_t LG = (u0 + n_groups - 1u) / C + 1u + rng_below(rng, 2);
    planes P;
    planes_init(&P, E, C, LG);
    const uint32_t cells = 16u * C * LG;
    tsr_block_q8_0 * rows = (tsr_block_q8_0 *)malloc((size_t)cells * nb * sizeof *rows);
    for (uint32_t j = 0; j < cells; ++j) {
        rand_q8_row(rng, rows + (size_t)j * nb, nb, vspecial && rng_below(rng, 4) == 0);
        if (pk == P_EXTREME) for (uint32_t b = 0; b < nb; ++b) rows[(size_t)j * nb + b].d = (uint16_t)(rng_below(rng, 2) ? 0x7BFF : 0x0001);
        tsr_kv_pack_row(TSR_KV_V, &P.g, P.ch, j, rows + (size_t)j * nb);
    }
    const uint32_t ps = n_groups + pstride_extra;
    float * p = (float *)malloc((size_t)R * ps * 16u * sizeof(float));
    for (size_t i = 0; i < (size_t)R * ps * 16u; ++i) {
        float v = tsr_m_exp((float)(-rng_unit(rng) * 12.0));
        const uint32_t k = rng_below(rng, 100);
        if (pk == P_ZERO) v = (k < 50) ? 0.0f : -0.0f;
        else if (k < 20) v = 0.0f;                                  // masked cells (p = EXP(-inf))
        else if (pk == P_SPECIAL && k < 22) v = u2f(0x00400000u);   // subnormal: skipped (DAZ)
        else if (pk == P_SPECIAL && k < 23) v = u2f(rng_below(rng, 2) ? 0x7F800000u : 0x7FC00000u);
        else if (pk == P_SPECIAL && k < 26) v = -v;
        else if (pk == P_EXTREME) v = rng_below(rng, 2) ? u2f(0x00800000u + rng_below(rng, 1000)) : 1.0f;
        p[i] = v;
    }
    const uint32_t cs = R * D / 16u + cstride_extra;
    const size_t dn = ((size_t)C * cs + (size_t)C * Rl) * 16u;
    float * dst = (float *)malloc(dn * sizeof(float));
    for (size_t i = 0; i < dn; ++i) dst[i] = u2f(SENT);
    tsr_ref_attpv(sh, C, LG, (const uint8_t * const *)P.ch, u0, n_groups, flags, p, pstride_extra ? ps : 0u, dst,
                  cstride_extra ? cs : 0u);
    float * O = (float *)malloc((size_t)C * R * D * sizeof(float));
    float * l = (float *)malloc((size_t)C * R * sizeof(float));
    double * ex = (double *)malloc((size_t)C * R * D * sizeof(double));
    double * eb = (double *)malloc((size_t)C * R * D * sizeof(double));
    uint8_t * pass = (uint8_t *)calloc(R, 1);
    ind_attpv(sh, C, u0, n_groups, (flags & TSR_ATT_LSUM) != 0, rows + (size_t)16u * u0 * nb, p, ps, O, l, ex, eb);
    long bad = 0, badb = 0, badl = 0, bads = 0;
    double worst = 0.0;
    for (uint32_t q = 0; q < sh->T_q; ++q) for (uint32_t g = 0; g < E / D; ++g) for (uint32_t i = 0; i < sh->GS; ++i)
        pass[tsr_att_row(sh, q, g, i)] = 1;
    for (uint32_t c = 0; c < C; ++c) {
        for (uint32_t r = 0; r < R; ++r) {
            for (uint32_t d = 0; d < D; ++d) {
                const float got = dst[((size_t)c * cs + (size_t)r * D / 16u) * 16u + d];
                const size_t ix = ((size_t)c * R + r) * D + d;
                if (!pass[r]) { bads += f2u(got) != SENT; continue; }
                if (!bits_eq(got, O[ix])) { if (bad++ < 3) fprintf(stderr, "  O c%u r%u d%u: %08x model %08x\n", c, r, d, f2u(got), f2u(O[ix])); }
                if (isfinite(got) && isfinite(ex[ix])) {
                    const double e = fabs((double)got - ex[ix]);
                    if (e > eb[ix]) ++badb;
                    if (eb[ix] > 0 && e / eb[ix] > worst) worst = e / eb[ix];
                }
            }
            const float lg = dst[((size_t)C * cs + (size_t)c * Rl) * 16u + r];
            if (!pass[r] || !(flags & TSR_ATT_LSUM)) bads += f2u(lg) != SENT;
            else badl += !bits_eq(lg, l[(size_t)c * R + r]);
        }
        for (uint32_t r = R; r < 16u * Rl; ++r) bads += f2u(dst[((size_t)C * cs + (size_t)c * Rl) * 16u + r]) != SENT;
    }
    CHECK(bad == 0 && badl == 0 && bads == 0 && badb == 0,
          "ATTPV E %u D %u G %u hg0 %u GS %u Tq %u C %u u0 %u n %u lsum %u pk %d: %ld O, %ld l, %ld untouched, %ld bound (worst %.3f)",
          E, D, sh->G, sh->hg0, sh->GS, sh->T_q, C, u0, n_groups, flags, (int)pk, bad, badl, bads, badb, worst);
    if (g_verbose) printf("  ATTPV case worst err/bound %.3f\n", worst);
    free(pass); free(eb); free(ex); free(l); free(O); free(dst); free(p); free(rows);
    planes_free(&P);
}

static void test_bf_helpers(rng_t * rng) {
    // F(A, k) vs the exact-rational model over random A and k, incl. carries, FTZ and overflow edges.
    long bad = 0;
    for (int it = 0; it < 400000; ++it) {
        int64_t A = (int64_t)(rng_u64(rng) >> (27 + rng_below(rng, 37)));
        if (rng_below(rng, 2)) A = -A;
        if (it % 7 == 0) A = (int64_t)((1ull << (24 + rng_below(rng, 13))) - 1u - rng_below(rng, 3)) * (rng_below(rng, 2) ? -1 : 1);
        const int k = (int)rng_below(rng, 300) - 190;
        if (!bits_eq(tsr_att_bf_f(A, k), ind_F(A, k))) { if (bad++ < 5) fprintf(stderr, "F(%lld, %d)\n", (long long)A, k); }
    }
    CHECK(bad == 0, "tsr_att_bf_f vs exact rational model (%ld)", bad);
    CHECK(f2u(tsr_att_bf_f((1ll << 24) - 1, -150)) == 0u, "F just below 2^-126 -> +0");
    CHECK(f2u(tsr_att_bf_f(-((1ll << 25) - 1), -151)) == 0x80800000u, "F rounding up to -2^-126 is kept");
    CHECK(f2u(tsr_att_bf_f(1, 128)) == 0x7F800000u && f2u(tsr_att_bf_f(-1, 127)) == 0xFF000000u, "F overflow edge");
    // RNE_int(w·2^(23-E_S)) vs nearbyint.
    bad = 0;
    for (int it = 0; it < 400000; ++it) {
        const float w = u2f((rng_u32(rng) & 0x80000000u) | ((1u + rng_below(rng, 253)) << 23) | (rng_u32(rng) & 0x7FFFFFu));
        const int es = ind_eps(w) + (int)rng_below(rng, 30);
        const int32_t a = tsr_att_bf_int(w, es);
        const int64_t b = (int64_t)nearbyint(ldexp((double)w, 23 - es));
        if (a != b) { if (bad++ < 5) fprintf(stderr, "bf_int(%08x, %d) %d vs %lld\n", f2u(w), es, a, (long long)b); }
    }
    CHECK(bad == 0, "tsr_att_bf_int vs nearbyint (%ld)", bad);
    CHECK(tsr_att_bf_int(u2f(0x00400000u), 0) == 0 && tsr_att_bf_int(-0.0f, 5) == 0, "bf_int of zero / subnormal");
    CHECK(tsr_att_ilog2(1.0f) == 0 && tsr_att_ilog2(u2f(0x00800000u)) == -126 && tsr_att_ilog2(-3.0f) == 1, "ilog2");
}

static void test_attpv(rng_t * rng) {
    test_bf_helpers(rng);
    const tsr_att_shape s1 = { 128, 64, 7, 0, 7, 1 };
    test_attpv_case(rng, &s1, 4, 0, 16, TSR_ATT_LSUM, P_SOFT, 0, 0, 0);           // 4 full segments per channel
    test_attpv_case(rng, &s1, 4, 3, 21, TSR_ATT_LSUM, P_SOFT, 0, 2, 5);           // partial segments, u0 % C != 0
    test_attpv_case(rng, &s1, 3, 1, 2, TSR_ATT_LSUM, P_SOFT, 0, 0, 0);            // n_groups < C: empty channel
    test_attpv_case(rng, &s1, 4, 0, 9, TSR_ATT_LSUM, P_ZERO, 1, 0, 0);            // all p = +-0 (inf scales skipped)
    test_attpv_case(rng, &s1, 2, 2, 13, TSR_ATT_LSUM, P_SPECIAL, 1, 0, 0);        // NaN / inf / subnormal p, specials in V
    test_attpv_case(rng, &s1, 1, 0, 11, TSR_ATT_LSUM, P_EXTREME, 0, 0, 0);        // w near 2^-126 and near 2^16; q = -128
    const tsr_att_shape s2 = { 512, 512, 16, 8, 8, 1 };
    test_attpv_case(rng, &s2, 4, 1, 6, TSR_ATT_LSUM, P_SOFT, 0, 0, 0);            // second head subset
    const tsr_att_shape s3 = { 512, 256, 2, 1, 1, 4 };
    test_attpv_case(rng, &s3, 3, 0, 7, 0, P_SOFT, 0, 1, 0);                       // no LSUM, T_q 4
    for (int it = 0; it < 40; ++it) {
        tsr_att_shape s;
        s.D = 64u << rng_below(rng, 3);
        s.E = s.D * (1u + rng_below(rng, 3));
        if (it == 39) printf("  ATTPV lanes over AT §13.2's (segments+1)·2^-24·sum|p·v| bound: %ld of %ld (not gated)\n",
                             g_doc_bound_over, g_doc_bound_lanes);
        s.G = 1u + rng_below(rng, 8);
        s.GS = 1u + rng_below(rng, s.G);
        s.hg0 = rng_below(rng, s.G - s.GS + 1u);
        s.T_q = 1u + rng_below(rng, 8u / s.GS);
        const uint32_t C = 1u + rng_below(rng, 5);
        test_attpv_case(rng, &s, C, rng_below(rng, 7), 1u + rng_below(rng, 24), rng_below(rng, 2),
                        (p_kind)rng_below(rng, 4), (int)rng_below(rng, 2), rng_below(rng, 2), rng_below(rng, 2) * 3u);
    }
}

// ---------------------------------------------------------------------------
// 5. Vector ops (AT §5.4–§5.7)
// ---------------------------------------------------------------------------
static float ind_ctree(const float * x, uint32_t n) {   // levels, left to right; unpaired element passes
    float v[16];
    for (uint32_t i = 0; i < n; ++i) v[i] = x[i];
    while (n > 1) {
        float w[16];
        uint32_t m = 0;
        for (uint32_t k = 0; k + 1 < n; k += 2) w[m++] = ind_add(v[k], v[k + 1]);
        if (n & 1u) w[m++] = v[n - 1];
        for (uint32_t i = 0; i < m; ++i) v[i] = w[i];
        n = m;
    }
    return v[0];
}

static float rand_score(rng_t * rng) {
    const uint32_t k = rng_below(rng, 64);
    if (k == 0) return u2f(0xFF800000u);
    if (k == 1) return u2f(0x7FC00000u);
    if (k == 2) return u2f(TSR_M_FLOOR);
    if (k == 3) return u2f(0x00400000u);
    if (k == 4) return -0.0f;
    return (float)(rng_normal(rng) * 8.0);
}

static void test_vecops(rng_t * rng) {
    long bad = 0;
    // CTREE orders: C = 3 -> ((x0 + x1) + x2), C = 4 -> ((x0 + x1) + (x2 + x3)).
    {
        const float a[4] = { 1e8f, 1.0f, -1e8f, 1.0f };
        CHECK(bits_eq(tsr_att_ctree(a, 3), ind_add(ind_add(a[0], a[1]), a[2])), "CTREE C=3");
        CHECK(bits_eq(tsr_att_ctree(a, 4), ind_add(ind_add(a[0], a[1]), ind_add(a[2], a[3]))), "CTREE C=4");
        CHECK(bits_eq(tsr_att_ctree(a, 1), a[0]), "CTREE C=1");
    }
    for (int it = 0; it < 20000; ++it) {
        float x[16];
        const uint32_t n = 1u + rng_below(rng, 16);
        for (uint32_t i = 0; i < n; ++i) x[i] = (float)(rng_normal(rng) * ldexp(1.0, (int)rng_below(rng, 60) - 30));
        bad += !bits_eq(tsr_att_ctree(x, n), ind_ctree(x, n));
    }
    CHECK(bad == 0, "CTREE vs model (%ld)", bad);
    // MAXT
    bad = 0;
    for (int it = 0; it < 100000; ++it) {
        const float a = rng_below(rng, 4) ? rand_score(rng) : u2f(rng_u32(rng)), b = rng_below(rng, 4) ? rand_score(rng) : u2f(rng_u32(rng));
        bad += !bits_eq(tsr_att_maxt(a, b), ind_maxt(a, b));
    }
    CHECK(bad == 0, "MAXT vs model (%ld)", bad);
    CHECK(f2u(tsr_att_maxt(-0.0f, 0.0f)) == 0u && f2u(tsr_att_maxt(0.0f, -0.0f)) == 0u, "MAXT -0 < +0");
    // ASMAX / AMASK in place, random rows, against the formulas.
    for (int it = 0; it < 40; ++it) {
        const uint32_t nl = 1u + rng_below(rng, 5), st = nl + rng_below(rng, 3), R = 1u + rng_below(rng, 40), rpq = 1u + rng_below(rng, 7);
        const uint32_t nq = (R + rpq - 1u) / rpq, mst = nl + rng_below(rng, 2);
        const size_t n = (size_t)R * st * 16u;
        float * s = (float *)malloc(n * sizeof(float)), * s0 = (float *)malloc(n * sizeof(float));
        float * mk = (float *)malloc(((size_t)R + 16u) * sizeof(float)), * mask = (float *)malloc((size_t)nq * mst * 16u * sizeof(float));
        for (size_t i = 0; i < n; ++i) s[i] = s0[i] = rand_score(rng);
        for (size_t i = 0; i < (size_t)nq * mst * 16u; ++i) {
            const uint32_t k = rng_below(rng, 6);
            mask[i] = k < 2 ? u2f(0xFF800000u) : k == 2 ? -0.0f : k == 3 ? 0.0f : (float)(rng_normal(rng));
        }
        for (uint32_t r = 0; r < R + 16u; ++r) mk[r] = u2f(SENT);
        tsr_ref_amask(s, s, mask, nl, st, R, rpq, mst, mk);
        long b1 = 0;
        for (uint32_t r = 0; r < R; ++r) {
            float mx = u2f(TSR_M_FLOOR);
            for (uint32_t k = 0; k < 16u * nl; ++k) {
                const float m = mask[(size_t)(r / rpq) * mst * 16u + k];
                const float e = f2u(m) == 0xFF800000u ? u2f(0xFF800000u) : ind_add(s0[(size_t)r * st * 16u + k], m);
                b1 += !bits_eq(s[(size_t)r * st * 16u + k], e);
                mx = ind_maxt(mx, e);
            }
            for (uint32_t k = 16u * nl; k < 16u * st; ++k) b1 += !bits_eq(s[(size_t)r * st * 16u + k], s0[(size_t)r * st * 16u + k]);
            b1 += !bits_eq(mk[r], mx);
        }
        for (uint32_t r = R; r < R + 16u; ++r) b1 += f2u(mk[r]) != SENT;
        CHECK(b1 == 0, "AMASK nl %u st %u R %u rpq %u: %ld", nl, st, R, rpq, b1);
        // ASMAX on the AMASK output, scale random normal > 0
        const float scale = (float)ldexp(1.0 + rng_unit(rng), (int)rng_below(rng, 10) - 6);
        for (size_t i = 0; i < n; ++i) s0[i] = s[i];
        tsr_ref_asmax(s, s, mk, nl, st, R, scale);
        long b2 = 0;
        for (uint32_t r = 0; r < R; ++r) for (uint32_t k = 0; k < 16u * st; ++k) {
            const size_t ix = (size_t)r * st * 16u + k;
            const float e = k < 16u * nl ? tsr_m_exp(ind_mul(ind_sub(s0[ix], mk[r]), scale)) : s0[ix];
            b2 += !bits_eq(s[ix], e);
        }
        CHECK(b2 == 0, "ASMAX nl %u st %u R %u: %ld", nl, st, R, b2);
        free(s); free(s0); free(mk); free(mask);
    }
    // AMERGE (INIT then merges) and ANORM against the formulas; C in 1..16.
    for (int it = 0; it < 40; ++it) {
        const uint32_t C = 1u + rng_below(rng, 16), D = 64u << rng_below(rng, 2), R = 1u + rng_below(rng, 37), Rl = (R + 15u) / 16u;
        const uint32_t cs = R * D / 16u + rng_below(rng, 2) * 2u;
        const float scale = (float)ldexp(1.0 + rng_unit(rng), (int)rng_below(rng, 6) - 4);
        const size_t pn = ((size_t)C * cs + (size_t)C * Rl) * 16u;
        float * parts = (float *)malloc(pn * sizeof(float));
        float * orun = (float *)malloc((size_t)R * D * sizeof(float)), * om = (float *)malloc((size_t)R * D * sizeof(float));
        float * st = (float *)malloc((size_t)2u * Rl * 16u * sizeof(float)), * sm = (float *)malloc((size_t)2u * Rl * 16u * sizeof(float));
        float * mk = (float *)malloc((size_t)Rl * 16u * sizeof(float));
        for (size_t i = 0; i < (size_t)2u * Rl * 16u; ++i) st[i] = sm[i] = u2f(SENT);
        long b = 0;
        for (int k = 0; k < 4; ++k) {
            for (size_t i = 0; i < pn; ++i) parts[i] = (float)(rng_unit(rng) * 3.0);
            for (uint32_t r = 0; r < R; ++r) {
                mk[r] = rng_below(rng, 6) == 0 ? u2f(TSR_M_FLOOR) : (float)(rng_normal(rng) * 10.0);
                if (rng_below(rng, 20) == 0) for (uint32_t c = 0; c < C; ++c) parts[((size_t)C * cs + (size_t)c * Rl) * 16u + r] = 0.0f;
            }
            tsr_ref_amerge(orun, parts, mk, st, R, D, scale, C, cs == R * D / 16u ? 0u : cs, k == 0 ? TSR_ATT_INIT : 0u);
            for (uint32_t r = 0; r < R; ++r) {
                float xv[16];
                for (uint32_t c = 0; c < C; ++c) xv[c] = parts[((size_t)C * cs + (size_t)c * Rl) * 16u + r];
                const float lk = ind_ctree(xv, C);
                float a = 0.0f, bb = 0.0f;
                if (k == 0) { sm[r] = mk[r]; sm[Rl * 16u + r] = lk; }
                else {
                    const float mn = ind_maxt(sm[r], mk[r]);
                    a = tsr_m_exp(ind_mul(ind_sub(sm[r], mn), scale));
                    bb = tsr_m_exp(ind_mul(ind_sub(mk[r], mn), scale));
                    sm[Rl * 16u + r] = ind_add(ind_mul(sm[Rl * 16u + r], a), ind_mul(lk, bb));
                    sm[r] = mn;
                }
                for (uint32_t d = 0; d < D; ++d) {
                    for (uint32_t c = 0; c < C; ++c) xv[c] = parts[(size_t)c * cs * 16u + (size_t)r * D + d];
                    const float ok = ind_ctree(xv, C);
                    om[(size_t)r * D + d] = k == 0 ? ok : ind_add(ind_mul(om[(size_t)r * D + d], a), ind_mul(ok, bb));
                }
            }
            for (size_t i = 0; i < (size_t)R * D; ++i) b += !bits_eq(orun[i], om[i]);
            for (size_t i = 0; i < (size_t)2u * Rl * 16u; ++i) b += !bits_eq(st[i], sm[i]);
        }
        CHECK(b == 0, "AMERGE C %u D %u R %u: %ld", C, D, R, b);
        // ANORM, incl. an empty row (l = 0, O = 0 -> 0) and a NaN row.
        st[Rl * 16u] = 0.0f;
        for (uint32_t d = 0; d < D; ++d) orun[d] = 0.0f;
        if (R > 1) st[Rl * 16u + 1] = u2f(0x7FC00000u);
        float * out = (float *)malloc((size_t)R * D * sizeof(float));
        tsr_ref_anorm(out, orun, st + Rl * 16u, R, D);
        long bn = 0;
        for (uint32_t r = 0; r < R; ++r) {
            const float ri = ind_rcp(ind_maxt(st[Rl * 16u + r], u2f(0x00800000u)));
            for (uint32_t d = 0; d < D; ++d) bn += !bits_eq(out[(size_t)r * D + d], ind_mul(orun[(size_t)r * D + d], ri));
        }
        CHECK(bn == 0, "ANORM R %u D %u: %ld", R, D, bn);
        CHECK(f2u(out[0]) == 0u, "ANORM empty row -> +0");
        free(out); free(parts); free(orun); free(om); free(st); free(sm); free(mk);
    }
    // M_FLOOR merges: two floors -> a = b = 1; floor against a real maximum -> a = 0.
    CHECK(f2u(tsr_m_exp(tsr_fmul(tsr_fsub(u2f(TSR_M_FLOOR), u2f(TSR_M_FLOOR)), 0.125f))) == 0x3F800000u, "EXP(0) = 1 between floors");
    CHECK(f2u(tsr_m_exp(tsr_fmul(tsr_fsub(u2f(TSR_M_FLOOR), 3.0f), 0.125f))) == 0u, "EXP((M_FLOOR - m)·scale) = +0");
    CHECK(f2u(tsr_m_exp(tsr_fmul(tsr_fsub(u2f(0xFF800000u), u2f(TSR_M_FLOOR)), 1.0f))) == 0u, "EXP(-inf - M_FLOOR) = +0");
}

// ---------------------------------------------------------------------------
// FA cases: llama.cpp-style masks, data, double model
// ---------------------------------------------------------------------------
enum { M_CAUSAL, M_SWA, M_RING, M_FINITE, M_HOLES, M_EMPTYROW };

typedef struct fa_case {
    const char * name;
    uint32_t D, n_kv_h, G, T, n_kv;     // n_kv = kv_size (cells of the cache view)
    uint32_t pos0;                      // position of query token 0
    uint32_t n_swa;                     // M_SWA / M_RING window
    int mask;
    uint32_t C, t_max, pcap;
    float scale, qmul;
} fa_case;

typedef struct fa_data {
    uint32_t E, n_head, LG;
    planes K, V;
    uint8_t * kc, * vc;                 // ggml cache bytes [n_kv][E/32 blocks]
    float * q;                          // [T][n_head][D]
    uint16_t * mask;                    // [T][n_kv]
    uint32_t * msum_words;
    tsr_msum ms;
    float * mask_f32;
    uint32_t new_runs[2][2], n_new;
} fa_data;

static void fa_build(rng_t * rng, const fa_case * fc, fa_data * fd) {
    const uint32_t D = fc->D, E = D * fc->n_kv_h, n_kv = fc->n_kv, T = fc->T, nb = E / 32u;
    fd->E = E;
    fd->n_head = fc->G * fc->n_kv_h;
    fd->LG = tsr_kv_lg_min(n_kv, fc->C);
    planes_init(&fd->K, E, fc->C, fd->LG);
    planes_init(&fd->V, E, fc->C, fd->LG);
    // K / V rows: QACT A32 + KVST in runs of 64 cells (the SET_ROWS sequence).
    const size_t ql = (size_t)tsr_qa_lines(TSR_QA_A32, E) * 64u;
    uint8_t * qa = (uint8_t *)calloc(64u * ql, 1);
    float * x = (float *)malloc((size_t)E * sizeof(float));
    for (uint32_t kv = 0; kv < 2; ++kv) {
        for (uint32_t c0 = 0; c0 < n_kv; c0 += 64u) {
            const uint32_t n = n_kv - c0 < 64u ? n_kv - c0 : 64u;
            for (uint32_t i = 0; i < n; ++i) {
                for (uint32_t e = 0; e < E; ++e) x[e] = (float)rng_normal(rng);
                tsr_qact(TSR_QA_A32, x, E, qa + i * ql);
            }
            tsr_ref_kvst(kv, kv ? &fd->V.g : &fd->K.g, kv ? fd->V.ch : fd->K.ch, c0, n, qa, 0, NULL);
        }
    }
    free(x); free(qa);
    const size_t cb = (size_t)n_kv * nb * 34u;
    fd->kc = (uint8_t *)malloc(cb);
    fd->vc = (uint8_t *)malloc(cb);
    tsr_kv_get_bytes(TSR_KV_K, &fd->K.g, (const uint8_t * const *)fd->K.ch, fd->kc, 0, cb);
    tsr_kv_get_bytes(TSR_KV_V, &fd->V.g, (const uint8_t * const *)fd->V.ch, fd->vc, 0, cb);
    fd->q = (float *)malloc((size_t)T * fd->n_head * D * sizeof(float));
    for (size_t i = 0; i < (size_t)T * fd->n_head * D; ++i) fd->q[i] = (float)(rng_normal(rng) * (double)fc->qmul);
    // Mask: cell j holds position cpos[j] (-1 empty); llama.cpp's causal / SWA-standard rules.
    int64_t * cpos = (int64_t *)malloc((size_t)n_kv * sizeof(int64_t));
    const int64_t npos = (int64_t)fc->pos0 + T;
    for (uint32_t j = 0; j < n_kv; ++j) cpos[j] = -1;
    for (int64_t p = 0; p < npos; ++p) {
        if (fc->mask == M_RING) cpos[p % n_kv] = p;
        else if (p < n_kv) cpos[p] = p;
    }
    fd->mask = (uint16_t *)malloc((size_t)T * n_kv * sizeof(uint16_t));
    for (uint32_t t = 0; t < T; ++t) {
        const int64_t pq = (int64_t)fc->pos0 + t;
        for (uint32_t j = 0; j < n_kv; ++j) {
            const int64_t p = cpos[j];
            int vis = p >= 0 && p <= pq;
            if ((fc->mask == M_SWA || fc->mask == M_RING) && pq - p >= (int64_t)fc->n_swa) vis = 0;
            if (fc->mask == M_HOLES && j % 7u == 3u) vis = 0;
            if (fc->mask == M_EMPTYROW && t == 0) vis = 0;
            uint16_t h = vis ? 0x0000 : 0xFC00;
            if (vis && fc->mask == M_FINITE) h = tsr_f32_to_fp16(-0.25f * (float)((pq - p) % 13));
            fd->mask[(size_t)t * n_kv + j] = h;
        }
    }
    // Cells written in this launch: those of positions pos0 .. pos0 + T - 1.
    fd->n_new = 0;
    for (int64_t p = fc->pos0; p < npos; ++p) {
        const uint32_t j = (uint32_t)(fc->mask == M_RING ? p % n_kv : p);
        if (fd->n_new > 0 && fd->new_runs[fd->n_new - 1][1] == j) fd->new_runs[fd->n_new - 1][1] = j + 1u;
        else { fd->new_runs[fd->n_new][0] = j; fd->new_runs[fd->n_new][1] = j + 1u; ++fd->n_new; }
    }
    free(cpos);
    fd->msum_words = (uint32_t *)malloc(tsr_msum_words(T) * sizeof(uint32_t));
    const uint32_t fl = tsr_msum_build_f16(fd->mask, n_kv, n_kv, T, 0x1234u, fd->msum_words);
    tsr_msum_view(fd->msum_words, tsr_msum_words(T), &fd->ms);
    fd->mask_f32 = NULL;
    if (fl & TSR_MSUM_GENERAL) {   // the backend writes the f32 copy only for GENERAL masks (AT §7.3)
        fd->mask_f32 = (float *)malloc((size_t)T * 16u * ((n_kv + 15u) / 16u) * sizeof(float));
        tsr_msum_mask_f32(fd->mask, n_kv, n_kv, T, fc->scale, fd->mask_f32);
    }
}

static void fa_free(fa_data * fd) {
    planes_free(&fd->K); planes_free(&fd->V);
    free(fd->kc); free(fd->vc); free(fd->q); free(fd->mask); free(fd->msum_words); free(fd->mask_f32);
}

static int fa_run_dev(const fa_case * fc, fa_data * fd, float * out, tsr_fa_stats * st, int with_f32) {
    const tsr_att_cfg cfg = { fc->C, fc->t_max, fc->pcap, 0 };
    float * mf = NULL;
    if (with_f32 && fd->mask_f32 == NULL) {
        mf = (float *)malloc((size_t)fc->T * 16u * ((fc->n_kv + 15u) / 16u) * sizeof(float));
        tsr_msum_mask_f32(fd->mask, fc->n_kv, fc->n_kv, fc->T, fc->scale, mf);
    }
    const tsr_fa_args a = { fc->T, fc->D, fc->G, fc->scale, fd->q, fd->E, fd->LG,
                            (const uint8_t * const *)fd->K.ch, (const uint8_t * const *)fd->V.ch, &fd->ms,
                            mf ? mf : fd->mask_f32, (const uint32_t (*)[2])fd->new_runs, fd->n_new, out, st };
    const int rc = tsr_ref_fa(&cfg, &a);
    free(mf);
    return rc;
}

// Independent double model on the ggml cache bytes, with Q quantized as QACT A32 (the device's
// and, up to rare ties, the CPU's Q). exact[T][n_head][D]; bound[...] = the error bound of the
// device result derived from AT §6 (see fa_bound below); W = sum p|v| / sum p.
static void fa_double(const fa_case * fc, const fa_data * fd, double * exact, double * bound) {
    const uint32_t D = fc->D, E = fd->E, nb = E / 32u, n_head = fd->n_head, n_kv = fc->n_kv, K = n_head * D;
    const tsr_att_cfg cfg = { fc->C, fc->t_max, fc->pcap, 0 };
    uint32_t GS, Tq0;
    tsr_fa_pass_rule(&cfg, fc->G, &GS, &Tq0);
    uint8_t * qa = (uint8_t *)malloc((size_t)tsr_qa_lines(TSR_QA_A32, K) * 64u);
    double * qd = (double *)malloc((size_t)D * sizeof(double));
    double * s = (double *)malloc((size_t)n_kv * sizeof(double)), * t1 = (double *)malloc((size_t)n_kv * sizeof(double));
    double * pj = (double *)malloc((size_t)n_kv * sizeof(double));
    tsr_fa_chunk * ch = (tsr_fa_chunk *)malloc(4096u * sizeof *ch);
    const double u = U24;
    for (uint32_t t = 0; t < fc->T; ++t) {
        // the tile's chunk list (structure only: which cells share a segment)
        const uint32_t q0 = t / Tq0 * Tq0, Tq = fc->T - q0 < Tq0 ? fc->T - q0 : Tq0;
        const long nch = tsr_fa_plan(&cfg, E, D, &fd->ms, q0, Tq, (const uint32_t (*)[2])fd->new_runs, fd->n_new, ch, 4096u);
        tsr_qact(TSR_QA_A32, fd->q + (size_t)t * K, K, qa);
        const int8_t * qs = (const int8_t *)qa;
        const float * dx = tsr_qa_d(TSR_QA_A32, K, qa);
        for (uint32_t h = 0; h < n_head; ++h) {
            const uint32_t kh = h / fc->G;
            for (uint32_t d = 0; d < D; ++d) qd[d] = (double)dx[(h * D + d) / 32u] * qs[h * D + d];
            double M = -INFINITY, smax = 0.0, t1max = 0.0;
            for (uint32_t j = 0; j < n_kv; ++j) {
                const uint16_t mh = fd->mask[(size_t)t * n_kv + j];
                s[j] = -INFINITY;
                if (mh == 0xFC00u) continue;
                const uint8_t * kr = fd->kc + ((size_t)j * nb + (size_t)kh * (D / 32u)) * 34u;
                double acc = 0.0, a1 = 0.0;
                for (uint32_t b = 0; b < D / 32u; ++b) {
                    uint16_t dh;
                    memcpy(&dh, kr + 34u * b, 2);
                    const int8_t * kq = (const int8_t *)(kr + 34u * b + 2u);
                    double blk = 0.0;
                    for (int l = 0; l < 32; ++l) blk += (double)kq[l] * qd[32u * b + (uint32_t)l];
                    acc += ind_h2d(dh) * blk;
                    a1 += fabs(ind_h2d(dh) * blk);
                }
                t1[j] = a1;
                // the CPU's form scale·s + m; the device adds m / scale (the f32 mask copy) to the
                // raw score and scales the sum, so its rounding scale is |s| + |m| / scale
                s[j] = acc * (double)fc->scale + ind_h2d(mh);
                if (s[j] > M) M = s[j];
                const double sm = fabs(acc) + fabs(ind_h2d(mh)) / (double)fc->scale;
                if (sm > smax) smax = sm;
                if (a1 > t1max) t1max = a1;
            }
            double l = 0.0;
            for (uint32_t j = 0; j < n_kv; ++j) { pj[j] = isinf(s[j]) ? 0.0 : exp(s[j] - M); l += pj[j]; }
            // relative error of each p (score rounding, the fsub / fmul before EXP, EXP itself, the
            // merge factors) and the per-chunk and per-segment roundings (AT §6.2-§6.6)
            const double eps_p = u * (8.0 * (double)fc->scale * t1max + 6.0 * (double)fc->scale * smax + 6.0);
            const double ncst = (double)nch * 3.0 + 10.0;
            for (uint32_t d = 0; d < D; ++d) {
                const uint32_t b = kh * (D / 32u) + d / 32u;
                double num = 0.0, W = 0.0, align = 0.0;
                uint32_t nseg = 0;
                for (uint32_t j = 0; j < n_kv; ++j) {
                    if (pj[j] == 0.0) continue;
                    const uint8_t * vr = fd->vc + ((size_t)j * nb + b) * 34u;
                    uint16_t dh;
                    memcpy(&dh, vr, 2);
                    const double v = ind_h2d(dh) * ((const int8_t *)(vr + 2))[d % 32u];
                    num += pj[j] * v;
                    W += pj[j] * fabs(v);
                }
                // alignment of the 24-bit weights (AT §6.4 step 5): half a unit of 2^(E_S-23) per
                // cell and segment; segments from the chunk list, channels and runs of 4 groups.
                for (long k = 0; k < nch; ++k) {
                    for (uint32_t c = 0; c < fc->C; ++c) {
                        uint32_t cnt = 0;
                        double wm = 0.0, qsum = 0.0;
                        for (uint32_t uu = ch[k].u0; uu <= ch[k].u0 + ch[k].n_groups; ++uu) {
                            const int last = uu == ch[k].u0 + ch[k].n_groups;
                            if (!last && uu % fc->C != c) continue;
                            if (last || cnt == 4u) {
                                if (wm > 0.0) align += 2.0 * u * wm * qsum;
                                if (cnt) ++nseg;
                                cnt = 0; wm = 0.0; qsum = 0.0;
                                if (last) break;
                            }
                            ++cnt;
                            for (uint32_t rr = 0; rr < 16u; ++rr) {
                                const uint32_t j = 16u * uu + rr;
                                if (j >= n_kv || pj[j] == 0.0) continue;
                                const uint8_t * vr = fd->vc + ((size_t)j * nb + b) * 34u;
                                uint16_t dh;
                                memcpy(&dh, vr, 2);
                                const double w = pj[j] * fabs(ind_h2d(dh));
                                if (w > wm) wm = w;
                                qsum += fabs((double)((const int8_t *)(vr + 2))[d % 32u]);
                            }
                        }
                    }
                }
                const size_t ix = ((size_t)t * n_head + h) * D + d;
                exact[ix] = l > 0.0 ? num / l : 0.0;
                bound[ix] = l > 0.0 ? (2.0 * eps_p + ((double)nseg + ncst) * u) * (W / l) + align / l : 0.0;
            }
        }
    }
    free(ch); free(pj); free(t1); free(s); free(qd); free(qa);
}

typedef struct fa_err {
    double max_ratio;        // max |dev - exact| / bound
    double rms_rel;          // rms(dev - exact) / rms(exact)
    long over;               // elements over the bound
} fa_err;

static fa_err fa_vs_double(const float * dev, const double * exact, const double * bound, size_t n) {
    fa_err e = { 0.0, 0.0, 0 };
    double se = 0.0, sx = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double err = fabs((double)dev[i] - exact[i]);
        if (!(err <= bound[i])) ++e.over;
        if (bound[i] > 0.0 && err / bound[i] > e.max_ratio) e.max_ratio = err / bound[i];
        if (bound[i] == 0.0 && err > 0.0) e.max_ratio = INFINITY;
        se += err * err;
        sx += exact[i] * exact[i];
    }
    e.rms_rel = sx > 0.0 ? sqrt(se / sx) : sqrt(se);
    return e;
}

static const fa_case k_fa_self[] = {
    // name                 D   nkvh G  T   n_kv pos0 nswa mask       C  tmax pcap  scale     qmul
    { "decode-d64-g7",      64, 2,   7, 1,  256, 199, 0,   M_CAUSAL,  4, 8,   9344, 0.125f,   1.0f },
    { "prefill-d128-tq1",  128, 2,   5, 5,  160, 120, 0,   M_CAUSAL,  3, 8,   9344, 0.0883883f, 1.5f },
    { "prefill-swa-tq4",   256, 2,   2, 7,  200, 150, 64,  M_SWA,     4, 8,   9344, 1.0f,     0.25f },
    { "prefill-causal-c1",  64, 1,   1, 13, 300, 250, 0,   M_CAUSAL,  1, 8,   9344, 0.125f,   2.0f },
    { "decode-ring",       128, 1,   4, 1,  256, 300, 128, M_RING,    4, 8,   9344, 0.0883883f, 1.0f },
    { "decode-g16-2pass",  512, 1,  16, 1,  128, 100, 0,   M_CAUSAL,  4, 8,   9344, 1.0f,     0.05f },
    { "prefill-finite",    128, 1,   2, 6,  100, 90,  0,   M_FINITE,  2, 8,   9344, 1.0f,     0.1f },
    { "prefill-finite-scl",128, 1,   2, 6,  100, 90,  0,   M_FINITE,  2, 8,   9344, 0.0883883f, 1.0f },   // mask copy / scale
    { "prefill-holes",      64, 2,   3, 4,  120, 100, 0,   M_HOLES,   4, 8,   9344, 0.125f,   1.0f },
    { "emptyrow-tmax1",     64, 1,   2, 3,   64, 40,  0,   M_EMPTYROW,4, 1,   9344, 0.125f,   1.0f },
    { "small-pcap",        128, 2,   3, 1,  512, 511, 0,   M_CAUSAL,  4, 8,   512,  0.0883883f, 3.0f },
    { "prefill-ring-wrap",  64, 1,   2, 4,  128, 260, 64,  M_RING,    4, 8,   9344, 0.125f,   1.0f },   // E_MASK first
};

static void test_fa_double(rng_t * rng) {
    for (size_t i = 0; i < sizeof k_fa_self / sizeof k_fa_self[0]; ++i) {
        const fa_case * fc = &k_fa_self[i];
        fa_data fd;
        fa_build(rng, fc, &fd);
        const size_t n = (size_t)fc->T * fd.n_head * fc->D;
        float * out = (float *)malloc(n * sizeof(float));
        double * ex = (double *)malloc(n * sizeof(double)), * bd = (double *)malloc(n * sizeof(double));
        tsr_fa_stats st;
        int rc = fa_run_dev(fc, &fd, out, &st, 0);
        // A prefill tile over a wrapped SWA ring has two runs per row: the GENERAL path, but the
        // MSUM is not GENERAL, so the backend wrote no f32 copy (AT §7.3 open item). tsr_ref_fa
        // reports it; with the copy supplied the composite is well defined.
        CHECK((rc == TSR_FA_E_MASK) == (fc->mask == M_RING && fc->T > 1), "%s: E_MASK expectation (rc %d)", fc->name, rc);
        if (rc == TSR_FA_E_MASK) rc = fa_run_dev(fc, &fd, out, &st, 1);
        CHECK(rc == TSR_FA_OK, "%s: tsr_ref_fa rc %d", fc->name, rc);
        fa_double(fc, &fd, ex, bd);
        const fa_err e = fa_vs_double(out, ex, bd, n);
        CHECK(e.over == 0, "%s: %ld elements over the derived bound (max err/bound %.3f)", fc->name, e.over, e.max_ratio);
        if (fc->mask == M_EMPTYROW) {
            long nz = 0;
            for (size_t k = 0; k < (size_t)fd.n_head * fc->D; ++k) nz += f2u(out[k]) != 0u;
            CHECK(nz == 0, "%s: a fully masked row must give +0 (%ld)", fc->name, nz);
        }
        printf("  %-18s chunks %3u (general %u) ops qk %4u pv %4u: rms rel err vs double %.2e, max err/bound %.3f\n",
               fc->name, st.chunks, st.general_chunks, st.ops_attqk, st.ops_attpv, e.rms_rel, e.max_ratio);
        free(out); free(ex); free(bd);
        fa_free(&fd);
    }
}

// ---------------------------------------------------------------------------
// 6. MSUM and 7. the chunk planner
// ---------------------------------------------------------------------------
static void test_msum(void) {
    enum { NKV = 70 };
    uint16_t m[4][NKV];
    for (int j = 0; j < NKV; ++j) {
        m[0][j] = (j >= 5 && j < 40) ? 0x0000 : 0xFC00;                          // one run
        m[1][j] = (j < 3 || (j >= 60 && j < 70)) ? ((j & 1) ? 0x8000 : 0) : 0xFC00;  // two runs, -0 counts as 0
        m[2][j] = 0xFC00;                                                          // empty row
        m[3][j] = (j >= 10 && j < 20) ? 0x0000 : 0xFC00;
    }
    uint32_t w[8 + 4 * 4];
    CHECK(tsr_msum_build_f16(&m[0][0], NKV, NKV, 4, 99, w) == 0, "MSUM not GENERAL");
    CHECK(w[0] == TSR_MSUM_MAGIC && w[1] == 4 && w[2] == NKV && w[3] == 0 && w[4] == 0, "MSUM header");
    CHECK(w[8] == 5 && w[9] == 40 && w[10] == 0 && w[11] == 0, "row 0 runs");
    CHECK(w[12] == 0 && w[13] == 3 && w[14] == 60 && w[15] == 70, "row 1 runs");
    CHECK(w[16] == 0 && w[17] == 0 && w[18] == 0 && w[19] == 0, "row 2 empty");
    tsr_msum ms;
    CHECK(tsr_msum_view(w, 24, &ms) == 0 && ms.n_q == 4 && ms.rows[3].lo0 == 10 && ms.rows[3].hi0 == 20, "MSUM view");
    CHECK(tsr_msum_view(w, 23, &ms) != 0, "MSUM view size check");
    m[3][30] = 0; m[3][50] = 0;                                                    // three runs -> GENERAL
    CHECK(tsr_msum_build_f16(&m[0][0], NKV, NKV, 4, 99, w) == TSR_MSUM_GENERAL && w[4] == 99, "three runs -> GENERAL");
    CHECK(w[20] == 10 && w[21] == 51, "GENERAL rows hold the hull");
    m[3][30] = 0xFC00; m[3][50] = 0xFC00;
    m[0][7] = 0x3C00;                                                              // finite nonzero -> GENERAL
    CHECK(tsr_msum_build_f16(&m[0][0], NKV, NKV, 4, 0, w) == TSR_MSUM_GENERAL, "finite value -> GENERAL");
    float f[4 * 80];
    tsr_msum_mask_f32(&m[0][0], NKV, NKV, 4, 1.0f, f);
    CHECK(f2u(f[7]) == 0x3F800000u && f2u(f[0]) == 0xFF800000u && f2u(f[70]) == 0xFF800000u && f2u(f[79]) == 0xFF800000u &&
          f2u(f[80 + 1]) == 0x80000000u, "f32 copy and -inf tail");
    // finite values divided by the FA scale (AT §5.5 decision); +-0 and -inf unchanged
    m[0][8] = 0xBC00;                                                              // -1
    tsr_msum_mask_f32(&m[0][0], NKV, NKV, 4, 0.125f, f);
    CHECK(f2u(f[7]) == 0x41000000u && f2u(f[8]) == 0xC1000000u && f2u(f[5]) == 0u && f2u(f[0]) == 0xFF800000u &&
          f2u(f[80 + 1]) == 0x80000000u && f2u(f[79]) == 0xFF800000u, "f32 copy / scale 0.125");
    tsr_msum_mask_f32(&m[0][0], NKV, NKV, 4, 0.0883883f, f);
    CHECK(f2u(f[7]) == f2u(tsr_fdiv(1.0f, 0.0883883f)) && f2u(f[8]) == f2u(tsr_fdiv(-1.0f, 0.0883883f)), "f32 copy / 1/sqrt(128)");
}

// Checks one tile's chunk list against the planner's invariants.
static void check_plan(const tsr_att_cfg * cfg, uint32_t E, uint32_t D, const tsr_msum * ms, uint32_t q0, uint32_t Tq,
                       const uint32_t (*nr)[2], uint32_t n_new, const char * what) {
    tsr_fa_chunk ch[512];
    const long n = tsr_fa_plan(cfg, E, D, ms, q0, Tq, nr, n_new, ch, 512);
    CHECK(n >= 1 && n <= 512, "%s: plan count %ld", what, n);
    if (n < 1 || n > 512) return;
    const uint32_t h = E / D;
    // sub-range groups for NCG
    uint32_t ncg_max = 0;
    long bad = 0;
    int seen_new = 0;
    uint8_t * vis = (uint8_t *)calloc((size_t)Tq * (ms->n_kv + 16u), 1);
    int any_general = 0;
    for (long k = 0; k < n; ++k) {
        const tsr_fa_chunk * c = &ch[k];
        if (c->n_groups > ncg_max) ncg_max = c->n_groups;
        bad += c->n_groups == 0 || c->n_groups > 128u || c->c_lo > c->c_hi || c->c_hi > 16u * c->n_groups;
        bad += (uint64_t)h * 16u * ((c->n_groups + cfg->C - 1u) / cfg->C) > cfg->pcap;   // ATTPV PCAP check
        int has_new = 0;
        for (uint32_t i = 0; i < n_new; ++i) has_new |= nr[i][0] < 16u * (c->u0 + c->n_groups) && nr[i][1] > 16u * c->u0;
        if (seen_new && !has_new) ++bad;                                                   // newest last
        seen_new |= has_new;
        if (c->general) { any_general = 1; continue; }
        for (uint32_t q = 0; q < Tq; ++q) {
            const int64_t dq = (int64_t)Tq - 1 - q;
            int64_t lo = (int64_t)c->c_lo - ((c->flags & TSR_ATT_SLIDE) ? dq : 0);
            if (lo < 0) lo = 0;
            const int64_t hi = (int64_t)c->c_hi - ((c->flags & TSR_ATT_CAUSAL) ? dq : 0);
            for (int64_t j = lo; j < hi && j < 16 * (int64_t)c->n_groups; ++j) {
                const int64_t cell = 16 * (int64_t)c->u0 + j;
                if (cell >= ms->n_kv) { ++bad; continue; }
                if (vis[(size_t)q * ms->n_kv + (size_t)cell]++) ++bad;                    // each cell once
            }
        }
    }
    if (!any_general && !(ms->flags & TSR_MSUM_GENERAL)) {   // FAST: exactly the runs
        for (uint32_t q = 0; q < Tq; ++q) {
            const tsr_msum_row * w = &ms->rows[q0 + q];
            for (uint32_t j = 0; j < ms->n_kv; ++j) {
                const int in = (j >= w->lo0 && j < w->hi0) || (j >= w->lo1 && j < w->hi1);
                if (in != (vis[(size_t)q * ms->n_kv + j] != 0)) ++bad;
            }
        }
    }
    free(vis);
    CHECK(bad == 0, "%s: %ld plan invariant violations (%ld chunks)", what, bad, n);
}

static void test_planner(rng_t * rng) {
    // NCG formula (AT §6.6 step 3) at U200 values.
    const tsr_att_cfg u200 = { 4, 8, 9344, 32768 };
    CHECK(tsr_fa_ncg(&u200, 1024, 128, 256) == 128 && tsr_fa_ncg(&u200, 1024, 128, 64) == 32 &&
          tsr_fa_ncg(&u200, 1024, 128, 10) == 16 && tsr_fa_ncg(&u200, 1024, 128, 4000) == 128, "NCG Qwen2.5-32B");
    const tsr_att_cfg c3 = { 3, 8, 9344, 32768 };
    CHECK(tsr_fa_ncg(&c3, 128, 64, 100) == 60 && tsr_fa_ncg(&c3, 128, 64, 1000) == 128, "NCG C = 3");
    const tsr_att_cfg small = { 4, 8, 16u * 8u * 5u, 0 };
    CHECK(tsr_fa_ncg(&small, 1024, 128, 1000) == 20, "NCG PCAP limit");
    uint32_t GS, Tq;
    tsr_fa_pass_rule(&u200, 5, &GS, &Tq); CHECK(GS == 5 && Tq == 1, "pass rule G 5");
    tsr_fa_pass_rule(&u200, 2, &GS, &Tq); CHECK(GS == 2 && Tq == 4, "pass rule G 2");
    tsr_fa_pass_rule(&u200, 16, &GS, &Tq); CHECK(GS == 8 && Tq == 1, "pass rule G 16");
    // >= 2 chunks for >= 8C groups; one decode run.
    {
        tsr_msum_row row = { 0, 16u * 32u, 0, 0 };
        const tsr_msum ms = { 1, 16u * 32u, 0, &row };
        tsr_fa_chunk ch[8];
        CHECK(tsr_fa_plan(&u200, 1024, 128, &ms, 0, 1, NULL, 0, ch, 8) == 2 && ch[0].n_groups == 16 && ch[1].u0 == 16, "two chunks at 8C groups");
        const uint32_t nw[1][2] = { { 3, 4 } };
        CHECK(tsr_fa_plan(&u200, 1024, 128, &ms, 0, 1, nw, 1, ch, 8) == 2 && ch[0].u0 == 16 && ch[1].u0 == 0, "newest chunk last");
        const tsr_msum_row e = { 0, 0, 0, 0 };
        const tsr_msum me = { 1, 512, 0, &e };
        CHECK(tsr_fa_plan(&u200, 1024, 128, &me, 0, 1, NULL, 0, ch, 8) == 1 && ch[0].n_groups == 1 && ch[0].c_hi == 0, "empty row: one empty chunk");
    }
    // A causal prefill tile whose upper bounds straddle a chunk cut (rule 5): 8 queries at cells
    // 60..67, C = 1, NCG = 4 groups -> the cut at group 4 (cell 64) moves to groups 3 and 5.
    {
        const tsr_att_cfg c1 = { 1, 8, 9344, 0 };
        tsr_msum_row rows[8];
        for (uint32_t q = 0; q < 8; ++q) rows[q] = (tsr_msum_row){ 0, 61u + q, 0, 0 };
        const tsr_msum ms = { 8, 80, 0, rows };
        tsr_fa_chunk ch[8];
        const long n = tsr_fa_plan(&c1, 64, 64, &ms, 0, 8, NULL, 0, ch, 8);
        CHECK(n == 2 && ch[0].u0 == 0 && ch[0].n_groups == 3 && ch[0].c_hi == 48 && ch[0].flags == 0 &&
              ch[1].u0 == 3 && ch[1].n_groups == 2 && ch[1].c_lo == 0 && ch[1].c_hi == 20 && ch[1].flags == TSR_ATT_CAUSAL,
              "straddling cut split (n %ld)", n);
        check_plan(&c1, 64, 64, &ms, 0, 8, NULL, 0, "straddle");
        tsr_msum_row bad = { 5, 90, 0, 0 };   // run beyond n_kv
        const tsr_msum mb = { 1, 80, 0, &bad };
        CHECK(tsr_fa_plan(&c1, 64, 64, &mb, 0, 1, NULL, 0, ch, 8) == -2, "malformed MSUM rejected");
    }
    // Groups per channel (cycle model helper).
    CHECK(tsr_att_ch_groups(4, 3, 9, 3) == 3 && tsr_att_ch_groups(4, 3, 9, 0) == 2 && tsr_att_ch_groups(4, 3, 1, 1) == 0 &&
          tsr_att_ch_groups(1, 7, 5, 0) == 5, "tsr_att_ch_groups");
    // Random llama.cpp-style tiles: causal / SWA / ring / holes / finite, prefill tiles that straddle
    // chunk boundaries (small PCAP -> small NCG), C 1..5.
    for (int it = 0; it < 400; ++it) {
        fa_case fc;
        memset(&fc, 0, sizeof fc);
        fc.name = "plan";
        fc.D = 64u << rng_below(rng, 3);
        fc.n_kv_h = 1u + rng_below(rng, 2);
        fc.G = 1u + rng_below(rng, 4);
        fc.mask = (int)rng_below(rng, 6);
        fc.T = fc.mask == M_RING ? 1u : 1u + rng_below(rng, 12);
        fc.n_kv = 16u + rng_below(rng, 900);
        fc.pos0 = fc.mask == M_RING ? fc.n_kv + rng_below(rng, 3u * fc.n_kv) : rng_below(rng, fc.n_kv - fc.T + 1u);
        if (fc.mask != M_RING && fc.pos0 + fc.T > fc.n_kv) fc.pos0 = fc.n_kv - fc.T;
        fc.n_swa = 1u + rng_below(rng, fc.n_kv - 1u);
        fc.C = 1u + rng_below(rng, 5);
        fc.t_max = 1u + rng_below(rng, 8);
        fc.pcap = 16u * fc.n_kv_h * (1u + rng_below(rng, 12));
        // masks only (no data)
        const uint32_t n_kv = fc.n_kv, T = fc.T;
        uint16_t * mask = (uint16_t *)malloc((size_t)T * n_kv * sizeof(uint16_t));
        int64_t * cpos = (int64_t *)malloc((size_t)n_kv * sizeof(int64_t));
        for (uint32_t j = 0; j < n_kv; ++j) cpos[j] = -1;
        for (int64_t p = 0; p < (int64_t)fc.pos0 + T; ++p) {
            if (fc.mask == M_RING) cpos[p % n_kv] = p; else if (p < n_kv) cpos[p] = p;
        }
        for (uint32_t t = 0; t < T; ++t) for (uint32_t j = 0; j < n_kv; ++j) {
            const int64_t pq = (int64_t)fc.pos0 + t, p = cpos[j];
            int vis = p >= 0 && p <= pq;
            if ((fc.mask == M_SWA || fc.mask == M_RING) && pq - p >= (int64_t)fc.n_swa) vis = 0;
            if (fc.mask == M_HOLES && j % 7u == 3u) vis = 0;
            if (fc.mask == M_EMPTYROW && t == 0) vis = 0;
            mask[(size_t)t * n_kv + j] = vis ? (fc.mask == M_FINITE ? 0xBC00 : 0x0000) : 0xFC00;
        }
        uint32_t * w = (uint32_t *)malloc(tsr_msum_words(T) * sizeof(uint32_t));
        tsr_msum_build_f16(mask, n_kv, n_kv, T, 0, w);
        tsr_msum ms;
        tsr_msum_view(w, tsr_msum_words(T), &ms);
        const tsr_att_cfg cfg = { fc.C, fc.t_max, fc.pcap, 0 };
        tsr_fa_pass_rule(&cfg, fc.G, &GS, &Tq);
        const uint32_t nw[2][2] = { { fc.pos0 % n_kv, fc.pos0 % n_kv + 1u }, { 0, 0 } };
        for (uint32_t q0 = 0; q0 < T; q0 += Tq) {
            char what[96];
            snprintf(what, sizeof what, "plan it %d mask %d T %u q0 %u Tq %u C %u pcap %u", it, fc.mask, T, q0, Tq, fc.C, fc.pcap);
            check_plan(&cfg, fc.D * fc.n_kv_h, fc.D, &ms, q0, T - q0 < Tq ? T - q0 : Tq, nw, 1, what);
        }
        free(w); free(cpos); free(mask);
    }
}

// ---------------------------------------------------------------------------
// ggml: tsr_ref_fa against ggml-cpu FLASH_ATTN_EXT
// ---------------------------------------------------------------------------
#ifdef TSR_REF_AT_GGML
static const fa_case k_fa_ggml[] = {
    // name                       D   nkvh G   T   n_kv  pos0  nswa  mask       C  tmax pcap  scale       qmul
    { "qwen05b-decode",           64, 2,   7,  1,  1024, 1000, 0,    M_CAUSAL,  4, 8,   9344, 0.125f,     1.0f },
    { "qwen05b-prefill",          64, 2,   7,  12, 512,  300,  0,    M_CAUSAL,  4, 8,   9344, 0.125f,     1.0f },
    { "qwen32b-decode",          128, 8,   5,  1,  2048, 2000, 0,    M_CAUSAL,  4, 8,   9344, 0.0883883f, 1.0f },
    { "qwen32b-prefill",         128, 8,   5,  6,  1024, 700,  0,    M_CAUSAL,  4, 8,   9344, 0.0883883f, 1.0f },
    { "qwen-d128-c3",            128, 4,   7,  3,  1024, 900,  0,    M_CAUSAL,  3, 8,   9344, 0.0883883f, 2.0f },
    { "gemma-swa-prefill",       256, 8,   2,  11, 1024, 600,  256,  M_SWA,     4, 8,   9344, 1.0f,       0.08f },
    { "gemma-swa-ring-decode",   256, 8,   2,  1,  512,  1300, 384,  M_RING,    4, 8,   9344, 1.0f,       0.08f },
    { "gemma-full-decode",       512, 1,   16, 1,  1024, 1000, 0,    M_CAUSAL,  4, 8,   9344, 1.0f,       0.06f },
    { "gemma-full-prefill",      512, 1,   16, 3,  512,  400,  0,    M_CAUSAL,  4, 8,   9344, 1.0f,       0.06f },
    { "d256-prefill-tq8-c1",     256, 1,   1,  16, 768,  700,  0,    M_CAUSAL,  1, 8,   9344, 0.0625f,    1.0f },
    { "d128-general-finite",     128, 2,   4,  4,  512,  400,  0,    M_FINITE,  4, 8,   9344, 1.0f,       0.1f },
    { "finite-mask-scale!=1",    128, 2,   4,  4,  512,  400,  0,    M_FINITE,  4, 8,   9344, 0.0883883f, 1.0f },   // mask copy / scale
    { "d64-general-holes",        64, 2,   4,  5,  512,  450,  0,    M_HOLES,   4, 8,   9344, 0.125f,     1.0f },
    { "d128-peaked-smallpcap",   128, 2,   3,  2,  4096, 4000, 0,    M_CAUSAL,  4, 8,   512,  0.0883883f, 4.0f },
    { "d64-tmax1-swa",            64, 2,   3,  4,  256,  200,  64,   M_SWA,     4, 1,   9344, 0.125f,     1.0f },
    { "qwen32b-decode-16k",      128, 8,   5,  1,  16384, 16000, 0,  M_CAUSAL,  4, 8,   9344, 0.0883883f, 1.0f },
    { "d512-g1-prefill-tq8",     512, 1,   1,  9,  640,  500,  0,    M_CAUSAL,  4, 8,   9344, 0.0441942f, 1.0f },
    { "d64-c16-swa",              64, 4,   2,  7,  2048, 1500, 512,  M_SWA,     16, 8,  9344, 0.125f,     1.0f },
};

typedef struct fa_cmp { double nmse, max_abs, max_rel, kl_max, kl_mean; } fa_cmp;

// Q blocks where the CPU's x86 quantize_row_q8_0 (FA's vec_dot_type conversion, ops.cpp q_to_vec_dot)
// differs from QACT A32 = quantize_row_q8_0_ref: AT §6.7 source 1 (rare ties).
static long q_tie_blocks(const fa_data * fd, uint32_t T, uint32_t D) {
    const uint32_t K = fd->n_head * D, nb = K / 32u;
    block_q8_0 * a = (block_q8_0 *)malloc((size_t)nb * sizeof *a), * b = (block_q8_0 *)malloc((size_t)nb * sizeof *b);
    const ggml_from_float_t x86 = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0)->from_float;
    long n = 0;
    for (uint32_t t = 0; t < T; ++t) {
        x86(fd->q + (size_t)t * K, a, K);
        quantize_row_q8_0_ref(fd->q + (size_t)t * K, b, K);
        for (uint32_t k = 0; k < nb; ++k) n += memcmp(&a[k], &b[k], sizeof a[k]) != 0;
    }
    free(a); free(b);
    return n;
}

static fa_cmp fa_compare(const float * dev, const float * cpu, size_t rows, uint32_t D) {
    fa_cmp r = { 0, 0, 0, 0, 0 };
    double se = 0.0, sc = 0.0;
    for (size_t i = 0; i < rows; ++i) {
        const float * a = dev + i * D, * b = cpu + i * D;
        double rmax = 0.0;
        for (uint32_t d = 0; d < D; ++d) if (fabs((double)b[d]) > rmax) rmax = fabs((double)b[d]);
        double ma = -INFINITY, mb = -INFINITY;
        for (uint32_t d = 0; d < D; ++d) {
            const double e = fabs((double)a[d] - (double)b[d]);
            se += e * e;
            sc += (double)b[d] * (double)b[d];
            if (e > r.max_abs) r.max_abs = e;
            const double den = fabs((double)b[d]) > 1e-3 * rmax ? fabs((double)b[d]) : 1e-3 * rmax;
            if (den > 0.0 && e / den > r.max_rel) r.max_rel = e / den;
            if ((double)a[d] > ma) ma = (double)a[d];
            if ((double)b[d] > mb) mb = (double)b[d];
        }
        // KL(softmax(cpu row) || softmax(dev row)) over the D outputs (a logit-like view of the row)
        double za = 0.0, zb = 0.0;
        for (uint32_t d = 0; d < D; ++d) { za += exp((double)a[d] - ma); zb += exp((double)b[d] - mb); }
        double kl = 0.0;
        for (uint32_t d = 0; d < D; ++d) {
            const double pb = exp((double)b[d] - mb) / zb, pa = exp((double)a[d] - ma) / za;
            if (pb > 0.0) kl += pb * log(pb / pa);
        }
        if (kl > r.kl_max) r.kl_max = kl;
        r.kl_mean += kl / (double)rows;
    }
    r.nmse = sc > 0.0 ? se / sc : se;
    return r;
}

static int fa_run_cpu(const fa_case * fc, const fa_data * fd, float * out) {
    const uint32_t D = fc->D, E = fd->E, n_kv = fc->n_kv, T = fc->T, n_head = fd->n_head;
    const size_t cache_b = (size_t)n_kv * (E / 32u) * 34u;
    const size_t mem = 2u * cache_b + 3u * (size_t)T * n_head * D * 4u + (size_t)T * n_kv * 2u + (64u << 20);
    struct ggml_init_params ip = { mem, NULL, false };
    struct ggml_context * ctx = ggml_init(ip);
    if (ctx == NULL) return -1;
    struct ggml_tensor * kc = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, E, n_kv);
    struct ggml_tensor * vc = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, E, n_kv);
    memcpy(kc->data, fd->kc, cache_b);
    memcpy(vc->data, fd->vc, cache_b);
    // llama.cpp's FA views of the cache (src/llama-kv-cache.cpp get_k / get_v): [D, n_kv, n_head_kv]
    struct ggml_tensor * k = ggml_view_4d(ctx, kc, D, n_kv, fc->n_kv_h, 1, kc->nb[1], ggml_row_size(GGML_TYPE_Q8_0, D), kc->nb[1] * n_kv, 0);
    struct ggml_tensor * v = ggml_view_4d(ctx, vc, D, n_kv, fc->n_kv_h, 1, vc->nb[1], ggml_row_size(GGML_TYPE_Q8_0, D), vc->nb[1] * n_kv, 0);
    struct ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, T, n_head, 1);
    float * qd = (float *)q->data;
    for (uint32_t t = 0; t < T; ++t) for (uint32_t h = 0; h < n_head; ++h)
        memcpy(qd + ((size_t)h * T + t) * D, fd->q + ((size_t)t * n_head + h) * D, (size_t)D * sizeof(float));
    struct ggml_tensor * m = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_kv, T, 1, 1);
    memcpy(m->data, fd->mask, (size_t)T * n_kv * sizeof(uint16_t));
    struct ggml_tensor * fa = ggml_flash_attn_ext(ctx, q, k, v, m, fc->scale, 0.0f, 0.0f);
    ggml_prec_set_acc(fa, GGML_PREC_F32);   // as build_attn_mha (src/llama-graph.cpp)
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, fa);
    const enum ggml_status st = ggml_graph_compute_with_ctx(ctx, gf, 4);
    if (st == GGML_STATUS_SUCCESS) memcpy(out, fa->data, (size_t)T * n_head * D * sizeof(float));   // [T][n_head][D]
    ggml_free(ctx);
    return st == GGML_STATUS_SUCCESS ? 0 : -1;
}

static void test_fa_ggml(rng_t * rng) {
    ggml_cpu_init();
    printf("  %-24s %9s %9s %9s %9s %9s | %9s %9s %7s %6s\n", "case", "NMSE", "max_abs", "max_rel", "KL_max", "KL_mean",
           "dev_rms", "cpu_rms", "bound", "Qties");
    for (size_t i = 0; i < sizeof k_fa_ggml / sizeof k_fa_ggml[0]; ++i) {
        const fa_case * fc = &k_fa_ggml[i];
        fa_data fd;
        fa_build(rng, fc, &fd);
        const size_t rows = (size_t)fc->T * fd.n_head, n = rows * fc->D;
        float * dev = (float *)malloc(n * sizeof(float)), * cpu = (float *)malloc(n * sizeof(float));
        double * ex = (double *)malloc(n * sizeof(double)), * bd = (double *)malloc(n * sizeof(double));
        tsr_fa_stats st;
        int rc = fa_run_dev(fc, &fd, dev, &st, 0);
        if (rc == TSR_FA_E_MASK) rc = fa_run_dev(fc, &fd, dev, &st, 1);   // see test_fa_double
        CHECK(rc == TSR_FA_OK, "%s: tsr_ref_fa rc %d", fc->name, rc);
        CHECK(fa_run_cpu(fc, &fd, cpu) == 0, "%s: ggml compute failed", fc->name);
        fa_double(fc, &fd, ex, bd);
        const fa_cmp c = fa_compare(dev, cpu, rows, fc->D);
        const fa_err ed = fa_vs_double(dev, ex, bd, n);
        double sc = 0.0, sx = 0.0;
        for (size_t k = 0; k < n; ++k) { sc += ((double)cpu[k] - ex[k]) * ((double)cpu[k] - ex[k]); sx += ex[k] * ex[k]; }
        const double cpu_rms = sqrt(sc / sx);
        const long ties = q_tie_blocks(&fd, fc->T, fc->D);
        printf("  %-24s %9.2e %9.2e %9.2e %9.2e %9.2e | %9.2e %9.2e %7.3f %6ld  (chunks %u, general %u)\n", fc->name,
               c.nmse, c.max_abs, c.max_rel, c.kl_max, c.kl_mean, ed.rms_rel, cpu_rms, ed.max_ratio, ties, st.chunks,
               st.general_chunks);
        CHECK(sx > 0.0, "%s: the case has no visible cell", fc->name);
        CHECK(c.nmse <= 5e-4, "%s: NMSE %.3e above the test-backend-ops tolerance 5e-4", fc->name, c.nmse);
        // KL-level agreement: f32-level differences give KL ~ 1e-15 .. 1e-11 (the CPU's sequential
        // f32 sums grow with n_kv); one Q tie on the CPU (AT §6.7 source 1) moves a row by ~1e-4
        // relative, KL ~ 1e-9. Device precision itself is gated against the double model below.
        CHECK(c.kl_max <= (ties ? 1e-6 : 1e-8), "%s: softmax-KL of output rows %.3e (Q ties %ld)", fc->name, c.kl_max, ties);
        CHECK(ed.over == 0, "%s: %ld device elements over the derived bound (max %.3f)", fc->name, ed.over, ed.max_ratio);
        // AT §6.4 / §6.7: the device is no less precise than the CPU (rms vs the exact model)
        CHECK(ed.rms_rel <= 2.0 * cpu_rms + 1e-9, "%s: device rms error %.3e vs CPU %.3e", fc->name, ed.rms_rel, cpu_rms);
        free(dev); free(cpu); free(ex); free(bd);
        fa_free(&fd);
    }
}
#endif

int main(int argc, char ** argv) {
    const char * mode = argc > 1 ? argv[1] : "self";
    for (int i = 2; i < argc; ++i) if (strcmp(argv[i], "-v") == 0) g_verbose = 1;
    rng_t rng = { 0x5EED0A7ull };
    if (strcmp(mode, "self") == 0) {
        test_layout(&rng);      printf("layout done (%ld checks, %d failed)\n", g_checks, g_fail);
        test_kvst(&rng);        printf("kvst done (%ld, %d)\n", g_checks, g_fail);
        test_seg(&rng);         printf("seg done (%ld, %d)\n", g_checks, g_fail);
        test_attqk(&rng);       printf("attqk done (%ld, %d)\n", g_checks, g_fail);
        test_attpv(&rng);       printf("attpv done (%ld, %d)\n", g_checks, g_fail);
        test_vecops(&rng);      printf("vecops done (%ld, %d)\n", g_checks, g_fail);
        test_msum();
        test_planner(&rng);     printf("msum / planner done (%ld, %d)\n", g_checks, g_fail);
        test_fa_double(&rng);
        return test_finish("ref_at self");
    }
#ifdef TSR_REF_AT_GGML
    if (strcmp(mode, "ggml") == 0) {
        test_fa_ggml(&rng);
        return test_finish("ref_at ggml");
    }
#endif
    fprintf(stderr, "usage: ref_at self | ggml [-v]\n");
    return 2;
}
