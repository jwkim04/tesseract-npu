// test_ref_ggml.c — tsr_ref against ggml-base at the llama.cpp pin.
//
//   fp16     tsr_fp16_to_f32 / tsr_f32_to_fp16 vs ggml_fp16_to_fp32 / ggml_fp32_to_fp16
//            (GGML_FP16_TO_FP32 / GGML_FP32_TO_FP16), exhaustive over all inputs.
//   qact     tsr_qact A32 vs quantize_row_q8_0_ref, A256 vs quantize_row_q8_K_ref, bit-exact
//            on inputs inside the documented domain (no subnormals, no NaN/Inf, block amax
//            >= 2^-119), plus the documented behaviour outside it.
//   dequant  tsr_dequant_row (repacked) vs dequantize_row_q8_0 / dequantize_row_q4_K.
//
// Why dequant is bit-exact with ggml for every finite d/dmin (result of this test):
//   Q8_0: ggml y = q * d, tsr y = d (*) q — the same single rounded product.
//   Q4_K: ggml d1 = d*sc, m1 = dmin*m, y = d1*q - m1 (ggml-base is built without -march, so
//         no FMA contraction); tsr computes exactly (d (*) sc) (*) q (-) (dmin (*) m).
//   DAZ/FTZ never trigger: fp16 values are >= 2^-24 in magnitude (normal in f32), d*sc and
//   (d*sc)*q are exact (<= 21 significant bits), dmin*m is exact, and the difference is a
//   multiple of 2^-24, so it is 0 or >= 2^-24 — never subnormal. (For the same reason
//   d*(sc*q) gives identical results; the hardware may use either product order.)
//   Only NaN encodings differ when d/dmin are Inf/NaN: ggml yields the x86 default NaN
//   0xFFC00000 or a payload-carrying NaN, tsr yields canonical 0x7FC00000.
//
// Usage: test_ref_ggml fp16|qact|dequant

#include <float.h>

#include "test_util.h"
#include "tsr_ref.h"

#include "ggml.h"
#include "ggml-quants.h"   // ggml/src: block_q8_0, block_q4_K, block_q8_K and the *_ref functions

_Static_assert(sizeof(block_q8_0) == sizeof(tsr_block_q8_0), "q8_0 layout");
_Static_assert(sizeof(block_q4_K) == sizeof(tsr_block_q4_K), "q4_K layout");

// ---------------------------------------------------------------------------
// fp16
// ---------------------------------------------------------------------------
static void test_fp16(void) {
    for (uint32_t h = 0; h < 65536; ++h) {
        const uint32_t a = f2u(tsr_fp16_to_f32((uint16_t)h));
        const uint32_t b = f2u(ggml_fp16_to_fp32((ggml_fp16_t)h));
        CHECK(a == b, "fp16->f32 0x%04x: tsr 0x%08x ggml 0x%08x", h, a, b);
    }
    enum { CH = 1 << 20 };
    float * xs = (float *)malloc(CH * sizeof(float));
    ggml_fp16_t * hs = (ggml_fp16_t *)malloc(CH * sizeof(ggml_fp16_t));
    uint64_t bad = 0;
    for (uint64_t base = 0; base < (1ull << 32); base += CH) {
        for (uint32_t i = 0; i < CH; ++i) xs[i] = u2f((uint32_t)(base + i));
        ggml_fp32_to_fp16_row(xs, hs, CH);
        for (uint32_t i = 0; i < CH; ++i) {
            const uint16_t t = tsr_f32_to_fp16(xs[i]);
            if (t != hs[i]) {
                if (bad < 5) fprintf(stderr, "f32->fp16 0x%08x: tsr 0x%04x ggml 0x%04x\n", (uint32_t)(base + i), t, hs[i]);
                ++bad;
            }
        }
    }
    ++g_checks;
    if (bad) { fprintf(stderr, "FAIL: %llu f32->fp16 mismatches\n", (unsigned long long)bad); ++g_fail; }
    free(xs); free(hs);
}

// ---------------------------------------------------------------------------
// qact
// ---------------------------------------------------------------------------
static int is_subnormal_bits(uint32_t u) { return (u & 0x7F800000u) == 0 && (u & 0x007FFFFFu) != 0; }

// Random normal-or-zero float with exponent in [emin, emax] (biased), random sign.
static float rand_float_exp(rng_t * r, uint32_t emin, uint32_t emax) {
    const uint32_t e = emin + rng_below(r, emax - emin + 1);
    return u2f((rng_u32(r) & 0x80000000u) | (e << 23) | (rng_u32(r) & 0x007FFFFFu));
}

// Fill one block per a "style" chosen to hit the interesting cases.
static void fill_block(rng_t * r, float * x, uint32_t n, uint32_t style, int a256) {
    const double scale = ldexp(1.0, (int)rng_below(r, 160) - 80);
    switch (style) {
    case 0:  // gaussian, random scale
        for (uint32_t j = 0; j < n; ++j) x[j] = (float)(rng_normal(r) * scale);
        break;
    case 1:  // random bits over a wide exponent range (many products underflow to 0)
        for (uint32_t j = 0; j < n; ++j) x[j] = rand_float_exp(r, 1, 254);
        break;
    case 2:  // all zero, mixed signs of zero
        for (uint32_t j = 0; j < n; ++j) x[j] = (rng_u32(r) & 1) ? -0.0f : 0.0f;
        break;
    case 3: {  // ties for max |x|: +-M at several positions, first occurrence decides (A256)
        const float M = (float)(1.0 + rng_unit(r)) * (float)scale;
        for (uint32_t j = 0; j < n; ++j) x[j] = (float)(rng_normal(r) * 0.3) * M;
        for (int t = 0; t < 4; ++t) x[rng_below(r, n)] = (rng_u32(r) & 1) ? -M : M;
        for (uint32_t j = 0; j < n; ++j) if (fabsf(x[j]) > M) x[j] = M * 0.5f;
        break;
    }
    case 4: {  // exact .5 rounding boundaries
        // A32: amax = 127*2^k -> d = 2^k exactly, id = 2^-k, x*id = m + 0.5 exactly.
        // A256: max = -127*2^k -> iscale = 2^-k, iscale*x = m + 0.5 exactly (ties to even);
        //       max = +127*2^k -> iscale = -2^-k.
        const int k = (int)rng_below(r, 40) - 20;
        const float p2 = (float)ldexp(1.0, k);
        const float sgn = (a256 && (rng_u32(r) & 1)) ? -1.0f : 1.0f;
        for (uint32_t j = 0; j < n; ++j) {
            const int m = (int)rng_below(r, 254) - 127;           // -127 .. 126
            x[j] = ((float)m + 0.5f) * p2;
        }
        x[rng_below(r, n)] = sgn * 127.0f * p2;
        break;
    }
    case 5:  // negative max: one dominant negative value
        for (uint32_t j = 0; j < n; ++j) x[j] = (float)(rng_normal(r) * scale);
        x[rng_below(r, n)] = (float)(-8.0 * scale);
        break;
    case 6:  // large magnitudes up to FLT_MAX (fp16(d) overflows to Inf for A32)
        for (uint32_t j = 0; j < n; ++j) x[j] = rand_float_exp(r, 200, 254);
        if (rng_u32(r) & 1) x[rng_below(r, n)] = (rng_u32(r) & 1) ? FLT_MAX : -FLT_MAX;
        break;
    case 7:  // mostly zeros with a few values; tiny-but-normal block amax near the domain edge
        for (uint32_t j = 0; j < n; ++j) x[j] = 0.0f;
        for (int t = 0; t < 3; ++t) x[rng_below(r, n)] = (float)(rng_normal(r) * ldexp(1.0, -118));
        x[rng_below(r, n)] = (float)ldexp(1.0, -118);
        break;
    default: // small integers (many exact ties in A256 after scaling)
        for (uint32_t j = 0; j < n; ++j) x[j] = (float)((int)rng_below(r, 255) - 127);
        break;
    }
    // Keep inside the documented bit-exact domain: no subnormals.
    for (uint32_t j = 0; j < n; ++j) if (is_subnormal_bits(f2u(x[j]))) x[j] = 0.0f;
}

// Block amax >= 2^-119 or exactly zero (then both sides produce zeros).
static int block_in_domain(const float * x, uint32_t n) {
    float amax = 0.0f;
    for (uint32_t j = 0; j < n; ++j) if (fabsf(x[j]) > amax) amax = fabsf(x[j]);
    return amax == 0.0f || amax >= 0x1.0p-119f;
}

static uint64_t g_qact_blocks[2], g_qact_ties[2];

static void qact_compare(uint32_t kind, const float * x, uint32_t K) {
    uint8_t * qa = (uint8_t *)aligned_alloc(64, (size_t)tsr_qa_lines(kind, K) * 64);
    const int8_t * qs = tsr_qa_qs(kind, K, qa);
    const float * D = tsr_qa_d(kind, K, qa);
    memset(qa, 0xA5, (size_t)tsr_qa_lines(kind, K) * 64);   // like the simulator's initial SPM
    tsr_qact(kind, x, K, qa);
    // Tail bytes of every region (QS, D and, for A256, BS) must be zero (구현 명확화 꼬리 바이트).
    const size_t qs_end = K, d_end = (size_t)tsr_qa_qs_lines(kind, K) * 64 + 4 * (K / tsr_qa_block_size(kind));
    for (size_t o = qs_end; o < (size_t)tsr_qa_qs_lines(kind, K) * 64; ++o) CHECK(qa[o] == 0, "qs tail");
    for (size_t o = d_end; o < (size_t)(tsr_qa_qs_lines(kind, K) + tsr_qa_d_lines(kind, K)) * 64; ++o) CHECK(qa[o] == 0, "d tail");
    if (kind == TSR_QA_A256) {   // BS region: int16 x K/16, then zeros to the end of the image
        const size_t bs_end = (size_t)(tsr_qa_qs_lines(kind, K) + tsr_qa_d_lines(kind, K)) * 64 + 2 * (K / 16);
        CHECK(tsr_qa_bs_lines(kind, K) * 64 >= 2 * (K / 16), "bs region size");
        for (size_t o = bs_end; o < (size_t)tsr_qa_lines(kind, K) * 64; ++o) CHECK(qa[o] == 0, "bs tail K=%u o=%zu", K, o);
    }
    if (kind == TSR_QA_A32) {
        const uint32_t nb = K / 32;
        block_q8_0 * y = (block_q8_0 *)calloc(nb, sizeof(block_q8_0));
        quantize_row_q8_0_ref(x, y, K);
        for (uint32_t b = 0; b < nb; ++b) {
            const float gd = ggml_fp16_to_fp32(y[b].d);
            CHECK(f2u(D[b]) == f2u(gd), "A32 K=%u blk %u D tsr 0x%08x ggml 0x%08x", K, b, f2u(D[b]), f2u(gd));
            CHECK(memcmp(qs + 32 * b, y[b].qs, 32) == 0, "A32 K=%u blk %u qs differ", K, b);
            for (uint32_t j = 0; j < 32; ++j) {
                if (qs[32 * b + j] != y[b].qs[j]) {
                    fprintf(stderr, "   j=%u x=%a tsr %d ggml %d\n", j, (double)x[32 * b + j], qs[32 * b + j], y[b].qs[j]);
                    break;
                }
            }
            ++g_qact_blocks[0];
        }
        free(y);
    } else {
        const uint32_t nb = K / 256;
        const int16_t * bs = tsr_qa_bs(kind, K, qa);
        block_q8_K * y = (block_q8_K *)calloc(nb, sizeof(block_q8_K));   // zeroed: ggml skips bsums for amax == 0
        quantize_row_q8_K_ref(x, y, K);
        for (uint32_t b = 0; b < nb; ++b) {
            CHECK(f2u(D[b]) == f2u(y[b].d), "A256 K=%u blk %u d tsr %a ggml %a", K, b, (double)D[b], (double)y[b].d);
            CHECK(memcmp(qs + 256 * b, y[b].qs, 256) == 0, "A256 K=%u blk %u qs differ", K, b);
            CHECK(memcmp(bs + 16 * b, y[b].bsums, 32) == 0, "A256 K=%u blk %u bsums differ", K, b);
            for (uint32_t j = 0; j < 256; ++j) {
                if (qs[256 * b + j] != y[b].qs[j]) {
                    fprintf(stderr, "   j=%u x=%a tsr %d ggml %d\n", j, (double)x[256 * b + j], qs[256 * b + j], y[b].qs[j]);
                    break;
                }
            }
            for (uint32_t j = 0; j < 256; ++j) {
                if (qs[256 * b + j] < -127) CHECK(0, "A256 q=-128 should not occur in-domain");
            }
            ++g_qact_blocks[1];
        }
        free(y);
    }
    free(qa);
}

static void test_qact(void) {
    rng_t rng = { 0x0AC7ull };
    for (uint32_t kind = 0; kind < 2; ++kind) {
        const uint32_t bs = tsr_qa_block_size(kind);
        for (int it = 0; it < 3000; ++it) {
            const uint32_t nb = 1 + rng_below(&rng, kind == TSR_QA_A32 ? 40 : 6);
            const uint32_t K = nb * bs;
            float * x = (float *)malloc(K * sizeof(float));
            for (uint32_t b = 0; b < nb; ++b) {
                do {
                    fill_block(&rng, x + b * bs, bs, rng_below(&rng, 9), kind == TSR_QA_A256);
                } while (!block_in_domain(x + b * bs, bs));
            }
            qact_compare(kind, x, K);
            free(x);
        }
        // Long rows.
        const uint32_t K = kind == TSR_QA_A32 ? 32768 : 27648;
        float * x = (float *)malloc(K * sizeof(float));
        for (uint32_t j = 0; j < K; ++j) x[j] = (float)rng_normal(&rng);
        qact_compare(kind, x, K);
        free(x);
    }

    // Count exact .5 ties actually exercised (sanity that style 4 hits the rounding rule).
    for (uint32_t kind = 0; kind < 2; ++kind) {
        const uint32_t n = tsr_qa_block_size(kind);
        float x[256];
        x[0] = (kind == TSR_QA_A32) ? 127.0f : -127.0f;          // d = 1 / iscale = 1
        for (uint32_t j = 1; j < n; ++j) x[j] = (float)((int)j % 200 - 100) + 0.5f;
        qact_compare(kind, x, n);
        uint8_t qa[64 * 8];
        tsr_qact(kind, x, n, qa);
        const int8_t * q = tsr_qa_qs(kind, n, qa);
        for (uint32_t j = 1; j < n; ++j) {
            const int m = (int)j % 200 - 100;                     // x = m + 0.5
            const int want = (kind == TSR_QA_A32) ? (m >= 0 ? m + 1 : m)       // half away
                                                  : ((m % 2 == 0) ? m : m + 1); // half even
            CHECK(q[j] == want, "tie kind %u j %u: q %d want %d", kind, j, q[j], want);
            ++g_qact_ties[kind];
        }
    }

    // Ties for max |x| (A256): first occurrence decides the sign of max.
    {
        float x[256];
        for (int j = 0; j < 256; ++j) x[j] = 1.0f;
        x[10] = -4.0f; x[20] = 4.0f;
        uint8_t qa[64 * 6];
        tsr_qact(TSR_QA_A256, x, 256, qa);
        CHECK(f2u(*tsr_qa_d(TSR_QA_A256, 256, qa)) == f2u(tsr_frecip(tsr_fdiv(-127.0f, -4.0f))), "first max -4");
        CHECK(tsr_qa_qs(TSR_QA_A256, 256, qa)[10] == -127 && tsr_qa_qs(TSR_QA_A256, 256, qa)[20] == 127, "q of +-max");
        qact_compare(TSR_QA_A256, x, 256);
        x[10] = 4.0f; x[20] = -4.0f;
        tsr_qact(TSR_QA_A256, x, 256, qa);
        CHECK(*tsr_qa_d(TSR_QA_A256, 256, qa) < 0.0f, "first max +4 -> d < 0");
        qact_compare(TSR_QA_A256, x, 256);
    }

    // Outside the ggml-identical domain: spec-defined behaviour (구현 명확화).
    {
        float x[256];
        uint8_t qa[64 * 6];
        // Subnormal inputs are DAZ'ed: they never set amax and quantize to 0.
        for (int j = 0; j < 32; ++j) x[j] = u2f(0x00400000u + (uint32_t)j);
        tsr_qact(TSR_QA_A32, x, 32, qa);
        CHECK(f2u(*tsr_qa_d(TSR_QA_A32, 32, qa)) == 0, "A32 all-subnormal -> D = 0");
        for (int j = 0; j < 32; ++j) CHECK(tsr_qa_qs(TSR_QA_A32, 32, qa)[j] == 0, "A32 subnormal q");
        // (The all-subnormal block above passes even without DAZ, because d = amax/127 is
        // flushed anyway. The cases below fail if QACT skips DAZ in the amax/max search or
        // in x*id / iscale*x — spec 구현 명확화 "QACT 입력".)
        // A32, normal amax 2^-119 next to subnormals: d = 2^-119/127 stays normal, id ~ 2^126,
        // and 0x007FFFFF * id ~ 0.99 would round to +-1 without DAZ; with DAZ it is +-0 -> 0.
        for (int j = 0; j < 32; ++j) x[j] = 0.0f;
        x[0] = u2f(0x04000000u);          // 2^-119
        x[1] = u2f(0x007FFFFFu);          // largest subnormal
        x[2] = u2f(0x807FFFFFu);
        tsr_qact(TSR_QA_A32, x, 32, qa);
        {
            const int8_t * q = tsr_qa_qs(TSR_QA_A32, 32, qa);
            CHECK(q[0] == 127 && q[1] == 0 && q[2] == 0, "A32 DAZ x*id: q = %d %d %d, want 127 0 0", q[0], q[1], q[2]);
            const float d = tsr_fdiv(x[0], 127.0f);
            CHECK((f2u(d) & 0x7F800000u) != 0, "A32 DAZ case: d must be normal");
            CHECK(f2u(*tsr_qa_d(TSR_QA_A32, 32, qa)) == f2u(tsr_fp16_to_f32(tsr_f32_to_fp16(d))), "A32 DAZ D");
        }
        // A256, max = -2^-119 next to +-largest subnormal: iscale = 127*2^119, iscale*x ~ +-0.99
        // would give q = +-1 (BS[0] = -126, BS[1] = -1) without DAZ.
        for (int j = 0; j < 256; ++j) x[j] = 0.0f;
        x[0] = -0x1.0p-119f;
        x[1] = u2f(0x007FFFFFu);
        x[17] = u2f(0x807FFFFFu);
        tsr_qact(TSR_QA_A256, x, 256, qa);
        {
            const int8_t * q = tsr_qa_qs(TSR_QA_A256, 256, qa);
            const int16_t * bsum = tsr_qa_bs(TSR_QA_A256, 256, qa);
            CHECK(q[0] == -127 && q[1] == 0 && q[17] == 0, "A256 DAZ iscale*x: q = %d %d %d, want -127 0 0", q[0], q[1], q[17]);
            CHECK(bsum[0] == -127 && bsum[1] == 0, "A256 DAZ BS = %d %d, want -127 0", bsum[0], bsum[1]);
            CHECK(f2u(*tsr_qa_d(TSR_QA_A256, 256, qa)) == f2u(tsr_frecip(tsr_fdiv(-127.0f, x[0]))), "A256 DAZ d");
        }
        // A256 block of zeros and subnormals only: after DAZ amax = 0 -> d = +0, q = 0, BS = 0.
        // The first largest |x| is a positive subnormal; without DAZ in the max search it would
        // become max, giving iscale = -127/+0 = -Inf and d = -0 (and q = -128 without DAZ in
        // iscale*x).
        for (int j = 0; j < 256; ++j) x[j] = 0.0f;
        x[5] = u2f(0x00400000u);
        x[9] = u2f(0x80300000u);
        tsr_qact(TSR_QA_A256, x, 256, qa);
        CHECK(f2u(*tsr_qa_d(TSR_QA_A256, 256, qa)) == 0x00000000u, "A256 subnormal-only d = 0x%08x, want +0",
              f2u(*tsr_qa_d(TSR_QA_A256, 256, qa)));
        for (int j = 0; j < 256; ++j) CHECK(tsr_qa_qs(TSR_QA_A256, 256, qa)[j] == 0, "A256 subnormal-only q[%d]", j);
        for (int j = 0; j < 16; ++j) CHECK(tsr_qa_bs(TSR_QA_A256, 256, qa)[j] == 0, "A256 subnormal-only bs[%d]", j);
        // Tiny amax: d = amax/127 is subnormal -> FTZ -> d = 0, id = 0, all q = 0 (ggml would
        // keep a subnormal d and produce q = +-127).
        for (int j = 0; j < 32; ++j) x[j] = 0.0f;
        x[3] = 0x1.0p-122f; x[4] = -0x1.0p-123f;
        tsr_qact(TSR_QA_A32, x, 32, qa);
        CHECK(f2u(*tsr_qa_d(TSR_QA_A32, 32, qa)) == 0 && tsr_qa_qs(TSR_QA_A32, 32, qa)[3] == 0, "A32 tiny amax");
        // NaN is ignored by the amax search and quantizes to 0; Inf amax -> d = Inf, id = 0.
        for (int j = 0; j < 32; ++j) x[j] = (float)(j - 16);
        x[5] = u2f(0x7FC00000);
        tsr_qact(TSR_QA_A32, x, 32, qa);
        CHECK(*tsr_qa_d(TSR_QA_A32, 32, qa) == tsr_fp16_to_f32(tsr_f32_to_fp16(16.0f / 127.0f)), "A32 NaN ignored in amax");
        CHECK(tsr_qa_qs(TSR_QA_A32, 32, qa)[5] == 0 && tsr_qa_qs(TSR_QA_A32, 32, qa)[0] == -127, "A32 NaN q");
        x[5] = u2f(0x7F800000);
        tsr_qact(TSR_QA_A32, x, 32, qa);
        CHECK(f2u(*tsr_qa_d(TSR_QA_A32, 32, qa)) == 0x7F800000, "A32 Inf amax -> D = Inf");
        for (int j = 0; j < 32; ++j) CHECK(tsr_qa_qs(TSR_QA_A32, 32, qa)[j] == 0, "A32 Inf -> q = 0");
        // A256 with only NaN: amax stays 0 -> d = 0, q = 0, BS = 0.
        for (int j = 0; j < 256; ++j) x[j] = u2f(0xFFC00000);
        tsr_qact(TSR_QA_A256, x, 256, qa);
        CHECK(f2u(*tsr_qa_d(TSR_QA_A256, 256, qa)) == 0, "A256 all NaN d");
        for (int j = 0; j < 16; ++j) CHECK(tsr_qa_bs(TSR_QA_A256, 256, qa)[j] == 0, "A256 all NaN bs");
        // A256 Inf max: iscale = -127/Inf = -0, d = 1/-0 = -Inf, q = 0.
        for (int j = 0; j < 256; ++j) x[j] = 1.0f;
        x[7] = u2f(0x7F800000);
        tsr_qact(TSR_QA_A256, x, 256, qa);
        CHECK(f2u(*tsr_qa_d(TSR_QA_A256, 256, qa)) == 0xFF800000, "A256 Inf max -> d = -Inf");
        for (int j = 0; j < 256; ++j) CHECK(tsr_qa_qs(TSR_QA_A256, 256, qa)[j] == 0, "A256 Inf -> q = 0");
        // A256 |max| so small that -127/max overflows: iscale = -Inf, q saturates, d = -0.
        for (int j = 0; j < 256; ++j) x[j] = 0.0f;
        x[0] = 0x1.0p-125f; x[1] = -0x1.0p-126f;
        tsr_qact(TSR_QA_A256, x, 256, qa);
        CHECK(f2u(*tsr_qa_d(TSR_QA_A256, 256, qa)) == 0x80000000, "A256 tiny max d = -0");
        CHECK(tsr_qa_qs(TSR_QA_A256, 256, qa)[0] == -128 && tsr_qa_qs(TSR_QA_A256, 256, qa)[1] == 127 &&
              tsr_qa_qs(TSR_QA_A256, 256, qa)[2] == 0, "A256 tiny max q saturates");
        CHECK(tsr_qa_bs(TSR_QA_A256, 256, qa)[0] == -1, "A256 tiny max bs");
    }
    printf("qact: %llu A32 blocks, %llu A256 blocks compared bit-exact; %llu/%llu .5 ties checked\n",
           (unsigned long long)g_qact_blocks[0], (unsigned long long)g_qact_blocks[1],
           (unsigned long long)g_qact_ties[0], (unsigned long long)g_qact_ties[1]);
}

// ---------------------------------------------------------------------------
// dequant
// ---------------------------------------------------------------------------
static uint16_t rand_fp16_finite(rng_t * r) {
    for (;;) {
        const uint16_t h = (uint16_t)rng_u32(r);
        if ((h & 0x7C00) != 0x7C00) return h;
    }
}
static uint16_t rand_fp16_typical(rng_t * r) {   // positive, like real model scales
    return tsr_f32_to_fp16((float)ldexp(1.0 + rng_unit(r), -(int)rng_below(r, 16) - 2));
}

static void dequant_compare(uint32_t fmt, const uint8_t * blocks, uint32_t K, int allow_nan_diff,
                            uint64_t * nan_bits_diff) {
    uint8_t * row = (uint8_t *)malloc((size_t)tsr_row_beats(fmt, K) * 64);
    float * a = (float *)malloc(K * sizeof(float));
    float * b = (float *)malloc(K * sizeof(float));
    tsr_repack_row(fmt, blocks, K, row);
    tsr_dequant_row(fmt, row, K, a);
    if (fmt == TSR_FMT_Q8_0) dequantize_row_q8_0((const block_q8_0 *)blocks, b, K);
    else dequantize_row_q4_K((const block_q4_K *)blocks, b, K);
    for (uint32_t k = 0; k < K; ++k) {
        if (f2u(a[k]) == f2u(b[k])) continue;
        if (allow_nan_diff && a[k] != a[k] && b[k] != b[k]) {   // both NaN, different encodings
            CHECK(f2u(a[k]) == 0x7FC00000u, "tsr NaN not canonical");
            ++*nan_bits_diff;
            continue;
        }
        CHECK(0, "dequant fmt %u K=%u k=%u: tsr 0x%08x ggml 0x%08x", fmt, K, k, f2u(a[k]), f2u(b[k]));
    }
    free(row); free(a); free(b);
}

static void test_dequant(void) {
    rng_t rng = { 0xDE0ull };
    uint64_t n_vals = 0, nan_diff = 0;
    for (int it = 0; it < 2000; ++it) {
        // Q8_0
        {
            const uint32_t nb = 1 + rng_below(&rng, 80), K = nb * 32;
            block_q8_0 * blk = (block_q8_0 *)malloc(nb * sizeof(block_q8_0));
            rng_bytes(&rng, blk, nb * sizeof(block_q8_0));                      // qs incl. -128
            for (uint32_t i = 0; i < nb; ++i) blk[i].d = (it & 1) ? rand_fp16_finite(&rng) : rand_fp16_typical(&rng);
            dequant_compare(TSR_FMT_Q8_0, (const uint8_t *)blk, K, 0, NULL);
            n_vals += K;
            free(blk);
        }
        // Q4_K
        {
            const uint32_t nb = 1 + rng_below(&rng, 12), K = nb * 256;
            block_q4_K * blk = (block_q4_K *)malloc(nb * sizeof(block_q4_K));
            rng_bytes(&rng, blk, nb * sizeof(block_q4_K));                      // all scale bits random
            for (uint32_t i = 0; i < nb; ++i) {
                uint16_t d = (it & 1) ? rand_fp16_finite(&rng) : rand_fp16_typical(&rng);
                uint16_t m = (it & 1) ? rand_fp16_finite(&rng) : rand_fp16_typical(&rng);
                memcpy((uint8_t *)&blk[i] + 0, &d, 2);
                memcpy((uint8_t *)&blk[i] + 2, &m, 2);
            }
            dequant_compare(TSR_FMT_Q4_K, (const uint8_t *)blk, K, 0, NULL);
            n_vals += K;
            free(blk);
        }
    }
    // Non-finite scales: identical except for NaN encodings (ggml: x86 default NaN 0xFFC00000
    // or payload-propagating NaN; tsr: canonical 0x7FC00000).
    for (int it = 0; it < 200; ++it) {
        const uint32_t nb = 4;
        block_q4_K blk[4];
        block_q8_0 b8[4];
        rng_bytes(&rng, blk, sizeof blk);
        rng_bytes(&rng, b8, sizeof b8);
        const uint16_t specials[] = { 0x7C00, 0xFC00, 0x7E00, 0x7C01, 0xFE00, 0x0000, 0x8000, 0x0001 };
        for (uint32_t i = 0; i < nb; ++i) {
            uint16_t d = specials[rng_below(&rng, 8)], m = specials[rng_below(&rng, 8)];
            memcpy((uint8_t *)&blk[i], &d, 2);
            memcpy((uint8_t *)&blk[i] + 2, &m, 2);
            b8[i].d = specials[rng_below(&rng, 8)];
        }
        dequant_compare(TSR_FMT_Q4_K, (const uint8_t *)blk, nb * 256, 1, &nan_diff);
        dequant_compare(TSR_FMT_Q8_0, (const uint8_t *)b8, nb * 32, 1, &nan_diff);
    }
    printf("dequant: %llu finite-scale values bit-exact; non-finite scales: %llu values differ only in NaN encoding\n",
           (unsigned long long)n_vals, (unsigned long long)nan_diff);
}

int main(int argc, char ** argv) {
    const char * what = argc > 1 ? argv[1] : "all";
    int all = strcmp(what, "all") == 0;
    if (all || !strcmp(what, "fp16")) test_fp16();
    if (all || !strcmp(what, "qact")) test_qact();
    if (all || !strcmp(what, "dequant")) test_dequant();
    char name[64];
    snprintf(name, sizeof name, "test_ref_ggml %s", what);
    return test_finish(name);
}
