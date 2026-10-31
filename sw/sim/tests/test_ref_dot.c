// test_ref_dot.c — tsr_dot_row / tsr_term_* (spec "Numeric rules").
//
// 1. Bit-exact against an independent, straightforward re-implementation of the spec formula
//    written here. It shares no code with tsr_ref.c: it reads the original ggml blocks (not
//    the repacked row), takes activations from ggml's own quantize_row_q8_0_ref /
//    quantize_row_q8_K_ref (not from tsr_qact), decodes scales with explicit bit slicing, and
//    does f32 arithmetic through double (exact products; one RNE rounding to f32, which is
//    innocuous for +,-,* since 53 >= 2*24+2) with DAZ/FTZ/NaN rules via fpclassify.
// 2. Close to a double-precision reference sum_k w_k * x_k (w = dequantised weight in double,
//    x = Dx * q_x): |y - ref| <= 1e-5 * sum_k (|w-terms| * |x_k|).

#include <float.h>

#include "test_util.h"
#include "tsr_ref.h"

#include "ggml.h"
#include "ggml-quants.h"

// ---------------------------------------------------------------------------
// Independent f32 arithmetic with the spec rules
// ---------------------------------------------------------------------------
static float ind_in(float x) {                        // DAZ
    return fpclassify(x) == FP_SUBNORMAL ? copysignf(0.0f, x) : x;
}
static float ind_out(double r) {                      // one RNE rounding, then FTZ / canonical NaN
    if (isnan(r)) return u2f(0x7FC00000u);
    const float f = (float)r;
    return fpclassify(f) == FP_SUBNORMAL ? copysignf(0.0f, f) : f;
}
static float ind_add(float a, float b) { return ind_out((double)ind_in(a) + (double)ind_in(b)); }
static float ind_sub(float a, float b) { return ind_out((double)ind_in(a) - (double)ind_in(b)); }
static float ind_mul(float a, float b) { return ind_out((double)ind_in(a) * (double)ind_in(b)); }
static float ind_i2f(int32_t v) { return (float)(double)v; }
static float ind_h2f(uint16_t h) {                    // exact fp16 decode via ldexp
    const int e = (h >> 10) & 31, m = h & 1023;
    double v;
    if (e == 0) v = ldexp((double)m, -24);
    else if (e == 31) v = m ? NAN : INFINITY;
    else v = ldexp((double)(1024 + m), e - 25);
    return (float)((h & 0x8000) ? -v : v);
}

// 6-bit scales/mins of Q4_K, written from the bit layout (not from get_scale_min_k4):
//   j < 4 : sc = scales[j] & 63,                 m = scales[j+4] & 63
//   j >= 4: sc = low nibble of scales[j+4] | top 2 bits of scales[j-4] << 4,
//           m  = high nibble of scales[j+4] | top 2 bits of scales[j]   << 4
static void ind_scale_min(const uint8_t * s, int j, int * sc, int * m) {
    if (j < 4) { *sc = s[j] & 0x3F; *m = s[j + 4] & 0x3F; }
    else {
        *sc = (s[j + 4] & 0x0F) + 16 * (s[j - 4] >> 6);
        *m  = (s[j + 4] >> 4)   + 16 * (s[j] >> 6);
    }
}

static float ind_tree(const float * p) {
    return ind_add(ind_add(ind_add(ind_add(p[0], p[1]), ind_add(p[2], p[3])),
                           ind_add(ind_add(p[4], p[5]), ind_add(p[6], p[7]))),
                   ind_add(ind_add(ind_add(p[8], p[9]), ind_add(p[10], p[11])),
                           ind_add(ind_add(p[12], p[13]), ind_add(p[14], p[15]))));
}

static float ind_term_q8_0(const block_q8_0 * w, const block_q8_0 * x) {
    int32_t isum = 0;
    for (int l = 0; l < 32; ++l) isum += w->qs[l] * x->qs[l];
    const float Dx = ind_h2f(x->d);                   // A32 D = f32(fp16(d))
    return ind_mul(ind_mul(ind_h2f(w->d), Dx), ind_i2f(isum));
}

static float ind_term_q4_K(const block_q4_K * w, const block_q8_K * x) {
    uint16_t hd, hm;
    memcpy(&hd, (const uint8_t *)w, 2);
    memcpy(&hm, (const uint8_t *)w + 2, 2);
    int32_t isum = 0, msum = 0;
    for (int j = 0; j < 8; ++j) {
        int sc, m;
        ind_scale_min(w->scales, j, &sc, &m);
        int32_t S = 0;
        for (int l = 0; l < 32; ++l) {
            const int k = 32 * j + l;                 // weight index inside the superblock
            const int byte = w->qs[32 * (k / 64) + (k % 32)];
            const int qw = ((k % 64) < 32) ? (byte & 15) : (byte >> 4);
            S += qw * x->qs[k];
        }
        isum += sc * S;
        msum += m * (x->bsums[2 * j] + x->bsums[2 * j + 1]);
    }
    const float t1 = ind_mul(ind_mul(ind_h2f(hd), x->d), ind_i2f(isum));
    const float t2 = ind_mul(ind_mul(ind_h2f(hm), x->d), ind_i2f(msum));
    return ind_sub(t1, t2);
}

static float ind_dot(uint32_t fmt, const void * wblocks, const void * xblocks, uint32_t K, float * terms) {
    float p[16];
    for (int i = 0; i < 16; ++i) p[i] = 0.0f;
    const uint32_t nb = K / (fmt == TSR_FMT_Q8_0 ? 32 : 256);
    for (uint32_t i = 0; i < nb; ++i) {
        const float t = fmt == TSR_FMT_Q8_0
            ? ind_term_q8_0((const block_q8_0 *)wblocks + i, (const block_q8_0 *)xblocks + i)
            : ind_term_q4_K((const block_q4_K *)wblocks + i, (const block_q8_K *)xblocks + i);
        terms[i] = t;
        p[i % 16] = ind_add(p[i % 16], t);
    }
    return ind_tree(p);
}

// Double-precision reference and error scale.
static double ref_dot(uint32_t fmt, const void * wb, const void * xb, uint32_t K, double * scale) {
    double s = 0.0, a = 0.0;
    if (fmt == TSR_FMT_Q8_0) {
        for (uint32_t i = 0; i < K / 32; ++i) {
            const block_q8_0 * w = (const block_q8_0 *)wb + i, * x = (const block_q8_0 *)xb + i;
            const double dw = ind_h2f(w->d), dx = ind_h2f(x->d);
            for (int l = 0; l < 32; ++l) {
                const double v = dw * w->qs[l] * dx * x->qs[l];
                s += v; a += fabs(v);
            }
        }
    } else {
        for (uint32_t i = 0; i < K / 256; ++i) {
            const block_q4_K * w = (const block_q4_K *)wb + i;
            const block_q8_K * x = (const block_q8_K *)xb + i;
            uint16_t hd, hm;
            memcpy(&hd, (const uint8_t *)w, 2);
            memcpy(&hm, (const uint8_t *)w + 2, 2);
            const double d = ind_h2f(hd), dmin = ind_h2f(hm);
            for (int k = 0; k < 256; ++k) {
                int sc, m;
                ind_scale_min(w->scales, k / 32, &sc, &m);
                const int byte = w->qs[32 * (k / 64) + (k % 32)];
                const int qw = ((k % 64) < 32) ? (byte & 15) : (byte >> 4);
                const double xv = (double)x->d * x->qs[k];
                s += (d * sc * qw - dmin * m) * xv;
                a += (fabs(d * sc * qw) + fabs(dmin * m)) * fabs(xv);
            }
        }
    }
    *scale = a;
    return s;
}

// ---------------------------------------------------------------------------
static uint64_t g_rows, g_terms;
static double g_max_rel[2];

// Build weights + activations, run both implementations, compare.
static void run_case(uint32_t fmt, const uint8_t * wblocks, const float * xf, uint32_t K, int check_ref) {
    const uint32_t kind = tsr_qa_kind_for_fmt(fmt);
    const uint32_t nb = K / tsr_fmt_block_size(fmt);
    uint8_t * row = (uint8_t *)malloc((size_t)tsr_row_beats(fmt, K) * 64);
    uint8_t * qa = (uint8_t *)aligned_alloc(64, (size_t)tsr_qa_lines(kind, K) * 64);
    tsr_repack_row(fmt, wblocks, K, row);
    tsr_qact(kind, xf, K, qa);

    void * xb;
    if (fmt == TSR_FMT_Q8_0) { xb = calloc(nb, sizeof(block_q8_0)); quantize_row_q8_0_ref(xf, (block_q8_0 *)xb, K); }
    else                     { xb = calloc(nb, sizeof(block_q8_K)); quantize_row_q8_K_ref(xf, (block_q8_K *)xb, K); }

    float * terms = (float *)malloc(nb * sizeof(float));
    const float yi = ind_dot(fmt, wblocks, xb, K, terms);
    const float yt = tsr_dot_row(fmt, row, K, qa);
    for (uint32_t i = 0; i < nb; ++i) {
        const float t = fmt == TSR_FMT_Q8_0 ? tsr_term_q8_0(row, K, i, qa) : tsr_term_q4_K(row, K, i, qa);
        CHECK(f2u(t) == f2u(terms[i]), "fmt %u K=%u term %u: tsr 0x%08x ind 0x%08x", fmt, K, i, f2u(t), f2u(terms[i]));
    }
    g_terms += nb;
    CHECK(f2u(yt) == f2u(yi), "fmt %u K=%u dot: tsr 0x%08x (%a) ind 0x%08x (%a)", fmt, K, f2u(yt), (double)yt, f2u(yi), (double)yi);
    ++g_rows;

    if (check_ref) {
        double scale;
        const double ref = ref_dot(fmt, wblocks, xb, K, &scale);
        const double err = fabs((double)yt - ref);
        const double rel = scale > 0 ? err / scale : err;
        if (rel > g_max_rel[fmt]) g_max_rel[fmt] = rel;
        CHECK(rel <= 1e-5, "fmt %u K=%u: y=%.9g ref=%.17g err/scale=%.3g", fmt, K, (double)yt, ref, rel);
    }
    free(row); free(qa); free(xb); free(terms);
}

static uint16_t h_typical(rng_t * r, int lo, int hi) {
    return tsr_f32_to_fp16((float)ldexp(1.0 + rng_unit(r), -(lo + (int)rng_below(r, (uint32_t)(hi - lo + 1)))));
}

static void make_weights(rng_t * r, uint32_t fmt, uint32_t K, uint8_t * blocks, int wild) {
    const uint32_t nb = K / tsr_fmt_block_size(fmt);
    rng_bytes(r, blocks, nb * tsr_fmt_ggml_block_bytes(fmt));            // qs/scales random (q8 incl. -128)
    for (uint32_t i = 0; i < nb; ++i) {
        uint8_t * b = blocks + i * tsr_fmt_ggml_block_bytes(fmt);
        uint16_t d = h_typical(r, 3, 12), m = h_typical(r, 4, 14);
        if (wild) {                                                     // any finite fp16, any sign
            do { d = (uint16_t)rng_u32(r); } while ((d & 0x7C00) == 0x7C00);
            do { m = (uint16_t)rng_u32(r); } while ((m & 0x7C00) == 0x7C00);
        }
        memcpy(b, &d, 2);
        if (fmt == TSR_FMT_Q4_K) memcpy(b + 2, &m, 2);
    }
}

static void make_acts(rng_t * r, float * x, uint32_t K, int style) {
    const double sc = ldexp(1.0, (int)rng_below(r, 24) - 12);
    for (uint32_t k = 0; k < K; ++k) {
        double v = rng_normal(r) * sc;
        if (style == 1 && (rng_u32(r) & 3) == 0) v *= 50.0;      // outliers
        if (style == 2) v = fabs(v);                            // one-signed (less cancellation)
        x[k] = (float)v;
    }
    if (style == 3) for (uint32_t k = 0; k < K; ++k) if (rng_u32(r) & 1) x[k] = 0.0f;
}

static void test_random(void) {
    rng_t rng = { 0xD07ull };
    for (uint32_t fmt = 0; fmt < 2; ++fmt) {
        const uint32_t bs = tsr_fmt_block_size(fmt);
        for (int it = 0; it < 1500; ++it) {
            uint32_t nb;
            if (it < 64) nb = 1 + (uint32_t)it;                                  // every small size
            else nb = 1 + rng_below(&rng, fmt == TSR_FMT_Q8_0 ? 1024 : 128);    // up to K_MAX
            const uint32_t K = nb * bs;
            uint8_t * w = (uint8_t *)malloc(nb * tsr_fmt_ggml_block_bytes(fmt));
            float * x = (float *)malloc(K * sizeof(float));
            const int wild = (it % 5) == 4;
            make_weights(&rng, fmt, K, w, wild);
            make_acts(&rng, x, K, it % 4);
            run_case(fmt, w, x, K, !wild);
            free(w); free(x);
        }
        // Model shapes.
        const uint32_t Ks[] = { 896, 4864, 5120, 27648, 32768 };
        for (size_t i = 0; i < sizeof Ks / sizeof Ks[0]; ++i) {
            const uint32_t K = Ks[i];
            if (K % bs) continue;
            uint8_t * w = (uint8_t *)malloc((K / bs) * tsr_fmt_ggml_block_bytes(fmt));
            float * x = (float *)malloc(K * sizeof(float));
            make_weights(&rng, fmt, K, w, 0);
            make_acts(&rng, x, K, 0);
            run_case(fmt, w, x, K, 1);
            free(w); free(x);
        }
    }
}

// Special values through the whole term/accumulate path: FTZ of d*Dx, Inf/NaN scales,
// all-zero activations, signed zeros. Only the bit-exact comparison applies here.
static void test_special(void) {
    rng_t rng = { 0x5BEC1A1ull };
    const uint16_t hspec[] = { 0x7C00, 0xFC00, 0x7E00, 0x0000, 0x8000, 0x0001, 0x8001, 0x7BFF, 0x3C00 };
    for (uint32_t fmt = 0; fmt < 2; ++fmt) {
        for (int it = 0; it < 400; ++it) {
            const uint32_t nb = 1 + rng_below(&rng, fmt == TSR_FMT_Q8_0 ? 70 : 20);
            const uint32_t K = nb * tsr_fmt_block_size(fmt);
            uint8_t * w = (uint8_t *)malloc(nb * tsr_fmt_ggml_block_bytes(fmt));
            float * x = (float *)malloc(K * sizeof(float));
            make_weights(&rng, fmt, K, w, 0);
            for (uint32_t i = 0; i < nb; ++i) {
                if (rng_below(&rng, 3) == 0) {
                    uint8_t * b = w + i * tsr_fmt_ggml_block_bytes(fmt);
                    const uint16_t d = hspec[rng_below(&rng, 9)], m = hspec[rng_below(&rng, 9)];
                    memcpy(b, &d, 2);
                    if (fmt == TSR_FMT_Q4_K) memcpy(b + 2, &m, 2);
                }
            }
            const int style = it % 4;
            for (uint32_t k = 0; k < K; ++k) {
                switch (style) {
                case 0: x[k] = 0.0f; break;                                   // all-zero QA
                case 1: x[k] = (float)(rng_normal(&rng) * 0x1.0p-117); break; // d*Dx -> subnormal -> FTZ
                case 2: x[k] = (float)(rng_normal(&rng) * 0x1.0p-110); break; // partial FTZ
                default: x[k] = (float)(rng_normal(&rng) * 0x1.0p100); break; // large: Inf terms possible
                }
                if (fpclassify(x[k]) == FP_SUBNORMAL) x[k] = 0.0f;
            }
            run_case(fmt, w, x, K, 0);
            free(w); free(x);
        }
    }
}

int main(void) {
    test_random();
    test_special();
    printf("dot: %llu rows / %llu terms bit-exact vs independent spec implementation; "
           "max |y-ref|/scale: Q8_0 %.3g, Q4_K %.3g\n",
           (unsigned long long)g_rows, (unsigned long long)g_terms, g_max_rel[0], g_max_rel[1]);
    return test_finish("test_ref_dot");
}
