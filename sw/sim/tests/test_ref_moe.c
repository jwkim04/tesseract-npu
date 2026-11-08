// test_ref_moe.c — the MoE routing reference tsr_ref_moe_route (sw/sim/tsr_ref_vu.c; docs/spec.md §13,
// docs/design/06-software-verification.md "SW" §7.5 MOE_ROUTE, SW-61 to SW-63).
//
// Checks (one function each):
//   directed   a row with a known top-8 of 128; the weights sum to 1 within rounding; equal logits
//              select the smaller index first (the tie rule); a NaN logit makes every probability NaN
//              and the selection 0 .. k-1 (every candidate ties); k = 1 and k = n; probs output
//   random     random rows (n = 8 .. 512, k = 1 .. 16) against an independent evaluation: the
//              probabilities and weights against double precision (relative ~1e-6), the selected
//              set against a sort of the reference's own probabilities (value descending, index
//              ascending), the weights summing to 1 (relative 1e-6)
//   rows       the per-row VU ops (tsr_ref_red, tsr_ref_vop on T rows with the c vectors a MOE_ROUTE
//              tile uses) give the same bits as the one-token function, T = 1 .. 20
// With TSR_REF_VU_GGML (TSR_WITH_LLAMA builds; links the pinned ggml-cpu):
//   ggml_moe   the graph build_moe_ffn emits for norm_w (soft_max, argsort(DESC) + view, get_rows,
//              sum_rows, clamp, div) on ggml-cpu against the reference. Random rows: the selected ids
//              equal in every row whose k-th and (k+1)-th probabilities differ by more than 4e-6
//              relative (the VU's exp is not ggml's expf), weights within 8 ulp. Ties (equal logits
//              at the k boundary, and a duplicated block) and NaN rows are run and counted: ggml
//              sorts with std::sort and `>`, so equal values and NaN come out in an order the standard
//              does not define; the reference's rule is the VU TOPK's (index ascending, NaN first).
//              The count of rows where the two disagree is printed. The clamp is shown to be the identity
//              for the rows of n probabilities (the k largest sum to at least k / n).
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "test_util.h"
#include "tsr_ref.h"
#include "tsr_ref_math.h"
#include "tsr_ref_vu.h"

#ifdef TSR_REF_VU_GGML
#include "ggml-cpu.h"
#include "ggml.h"
#endif

#define CANON 0x7FC00000u

static uint32_t fb(float f) { return f2u(f); }
static float bf(uint32_t u) { return u2f(u); }

static void eqw(const char * what, uint32_t got, uint32_t want, int line) {
    ++g_checks;
    if (got != want) {
        fprintf(stderr, "FAIL line %d: %s = 0x%08x, want 0x%08x\n", line, what, got, want);
        if (++g_fail >= 40) exit(1);
    }
}
#define EQ(what, got, want) eqw(what, (got), (want), __LINE__)

// Logits that are an integer multiple of 1/8: exact in f32, so equal logits give equal probabilities.
static void fill_logits(rng_t * r, uint32_t n, uint32_t * x, double scale) {
    for (uint32_t i = 0; i < n; ++i) x[i] = fb((float)(rng_normal(r) * scale));
}

static void test_directed(void) {
    uint32_t x[128], ids[8], w[8], p[128];
    for (uint32_t i = 0; i < 128; ++i) x[i] = fb(-4.0f);
    // eight distinct high logits at scattered positions, in no particular order
    const uint32_t pos[8] = { 90, 3, 127, 64, 17, 33, 5, 100 };
    for (uint32_t j = 0; j < 8; ++j) x[pos[j]] = fb(1.0f + 0.25f * (float)j);
    tsr_ref_moe_route(128, 8, x, ids, w, p);
    for (uint32_t j = 0; j < 8; ++j) EQ("directed id", ids[j], pos[7 - j]);       // descending logit
    double sum = 0;
    for (uint32_t j = 0; j < 8; ++j) sum += (double)bf(w[j]);
    CHECK(fabs(sum - 1.0) < 1e-6, "weights sum %.9g", sum);
    for (uint32_t j = 1; j < 8; ++j) CHECK(bf(w[j - 1]) > bf(w[j]), "weights descend");
    for (uint32_t j = 0; j < 8; ++j) EQ("probs output", p[ids[j]] != 0u, 1u);

    // equal logits at the boundary: positions 20, 40, 60 tie for the last two places
    for (uint32_t i = 0; i < 128; ++i) x[i] = fb(-4.0f);
    for (uint32_t j = 0; j < 6; ++j) x[10 + j] = fb(2.0f + (float)j);
    x[60] = x[20] = x[40] = fb(1.0f);
    tsr_ref_moe_route(128, 8, x, ids, w, NULL);
    EQ("tie id 6", ids[6], 20u);
    EQ("tie id 7", ids[7], 40u);                                                   // 60 is left out
    EQ("tie weight", w[6], w[7]);
    // all equal: the first k indices
    for (uint32_t i = 0; i < 128; ++i) x[i] = fb(0.5f);
    tsr_ref_moe_route(128, 8, x, ids, w, NULL);
    for (uint32_t j = 0; j < 8; ++j) EQ("all equal id", ids[j], j);
    EQ("all equal weight", w[0], fb(0.125f));                                       // 8 equal weights: exactly 1/8
    // NaN anywhere: every probability is NaN, every weight too, the ids are 0 .. k-1
    x[77] = 0x7FC00001u;
    tsr_ref_moe_route(128, 8, x, ids, w, p);
    for (uint32_t i = 0; i < 128; ++i) EQ("nan prob", p[i], CANON);
    for (uint32_t j = 0; j < 8; ++j) {
        EQ("nan id", ids[j], j);
        EQ("nan weight", w[j], CANON);
    }
    // +inf logit: x - max is NaN for that element and 0 for none: the max is +inf, every x - inf is -inf
    // except inf - inf = NaN; the sum is NaN
    for (uint32_t i = 0; i < 128; ++i) x[i] = fb((float)i * 0.01f);
    x[5] = 0x7F800000u;
    tsr_ref_moe_route(128, 8, x, ids, w, p);
    EQ("inf logit prob", p[5], CANON);
    // k = 1 and k = n (n = 8)
    for (uint32_t i = 0; i < 8; ++i) x[i] = fb((float)i);
    tsr_ref_moe_route(8, 1, x, ids, w, NULL);
    EQ("k = 1 id", ids[0], 7u);
    CHECK(w[0] == fb(1.0f) || w[0] == fb(1.0f) - 1u, "k = 1 weight 0x%08x", w[0]);   // FM(q, RC(q)) is within 1 ulp of 1
    tsr_ref_moe_route(8, 8, x, ids, w, NULL);
    for (uint32_t j = 0; j < 8; ++j) EQ("k = n id", ids[j], 7 - j);
    // denormal logits are zero (DAZ); a negative zero equals a positive zero
    for (uint32_t i = 0; i < 16; ++i) x[i] = 0u;
    x[3] = 0x00000001u;
    x[4] = 0x80000000u;
    tsr_ref_moe_route(16, 4, x, ids, w, NULL);
    for (uint32_t j = 0; j < 4; ++j) EQ("daz ties id", ids[j], j);
}

static void test_random(void) {
    rng_t r = { 0x30E5ull };
    for (int it = 0; it < 3000; ++it) {
        const uint32_t n = 8 + rng_below(&r, 505), k = 1 + rng_below(&r, n < 16 ? n : 16);
        uint32_t x[TSR_MOE_MAX_EXPERTS], ids[16], w[16], p[TSR_MOE_MAX_EXPERTS];
        fill_logits(&r, n, x, 0.25 + 4.0 * rng_unit(&r));
        tsr_ref_moe_route(n, k, x, ids, w, p);
        // double precision softmax of the same logits
        double mx = -1e300;
        for (uint32_t i = 0; i < n; ++i) if ((double)bf(x[i]) > mx) mx = (double)bf(x[i]);
        double s = 0, d[TSR_MOE_MAX_EXPERTS];
        for (uint32_t i = 0; i < n; ++i) { d[i] = exp((double)bf(x[i]) - mx); s += d[i]; }
        for (uint32_t i = 0; i < n; ++i) {
            const double dp = d[i] / s;
            const double e = fabs((double)bf(p[i]) - dp);
            // x - max is rounded to f32 before the exp: the relative error of the exp is about |x - max| * 2^-24
            const double tol = (fabs((double)bf(x[i]) - mx) * 6.0e-8 + 5e-7) * dp;
            CHECK(e <= tol || bf(p[i]) == 0.0f, "prob %u of %u: %.9g vs %.9g", i, n, (double)bf(p[i]), dp);
        }
        // the selection against a sort of the reference's own probabilities
        uint32_t order[TSR_MOE_MAX_EXPERTS];
        for (uint32_t i = 0; i < n; ++i) order[i] = i;
        for (uint32_t a = 1; a < n; ++a) {               // insertion sort, value descending then index ascending
            const uint32_t v = order[a];
            uint32_t b = a;
            while (b > 0 && (bf(p[order[b - 1]]) < bf(p[v]))) { order[b] = order[b - 1]; --b; }
            order[b] = v;
        }
        double ssel = 0, wsum = 0;
        for (uint32_t j = 0; j < k; ++j) {
            EQ("random id", ids[j], order[j]);
            ssel += (double)bf(p[ids[j]]);
        }
        for (uint32_t j = 0; j < k; ++j) {
            const double dw = (double)bf(p[ids[j]]) / ssel;
            CHECK(fabs((double)bf(w[j]) - dw) <= 1e-6 * dw, "weight %u: %.9g vs %.9g", j, (double)bf(w[j]), dw);
            wsum += (double)bf(w[j]);
        }
        CHECK(fabs(wsum - 1.0) < 1e-6, "weights sum %.9g (n %u k %u)", wsum, n, k);
    }
}

// The per-row VU ops of a MOE_ROUTE tile (T rows, the c vectors RED writes) against the one-token function.
static void test_rows(void) {
    rng_t r = { 0x7A11ull };
    for (uint32_t T = 1; T <= 20; ++T) {
        const uint32_t n = 128, k = 8, LN = n / 16;
        uint32_t L[20 * 128], V[32], P[20 * 128], E[20 * 128], sel[20 * 16], SW[20 * 16], W[20 * 16];
        for (uint32_t t = 0; t < T; ++t) fill_logits(&r, n, L + t * n, 1.5);
        // stage 1: RED.MAX -> V; VOP.EXPSUBC (cmode VEC) -> E; RED.SUM post RCP -> V; VOP.MULC (VEC) -> P
        memset(V, 0, sizeof V);
        tsr_ref_red(TSR_RED_KIND_MAX, TSR_RED_POST_NONE, T, n, V, L, LN, 0, 0);
        tsr_ref_vop(TSR_VOP_FUNC_EXPSUBC, TSR_VOP_CMODE_VEC, T, n, E, LN, L, LN, NULL, 0, V);
        tsr_ref_red(TSR_RED_KIND_SUM, TSR_RED_POST_RCP, T, n, V, E, LN, 0, 0);
        tsr_ref_vop(TSR_VOP_FUNC_MULC, TSR_VOP_CMODE_VEC, T, n, P, LN, E, LN, NULL, 0, V);
        // the RISC-V part: TOPK per token into a line per token (value words), ids on the side
        uint32_t ids_rv[20][16];
        for (uint32_t t = 0; t < T; ++t) {
            uint32_t d[32];
            tsr_ref_topk(k, 1, n, d, 0, P + t * n, 0);
            memset(SW + t * 16, 0, 16 * sizeof(uint32_t));
            for (uint32_t j = 0; j < k; ++j) { ids_rv[t][j] = d[2 * j]; SW[t * 16 + j] = d[2 * j + 1]; }
        }
        // stage 2: RED.SUM post RCP over the k selected -> V; VOP.MULC (VEC) -> W
        tsr_ref_red(TSR_RED_KIND_SUM, TSR_RED_POST_RCP, T, k, V, SW, 1, 0, 0);
        tsr_ref_vop(TSR_VOP_FUNC_MULC, TSR_VOP_CMODE_VEC, T, k, W, 1, SW, 1, NULL, 0, V);
        (void)sel;
        for (uint32_t t = 0; t < T; ++t) {
            uint32_t ids[16], w[16];
            tsr_ref_moe_route(n, k, L + t * n, ids, w, NULL);
            for (uint32_t j = 0; j < k; ++j) {
                EQ("rows id", ids_rv[t][j], ids[j]);
                EQ("rows weight", W[t * 16 + j], w[j]);
            }
        }
    }
}

#ifdef TSR_REF_VU_GGML
// The graph of llama.cpp's build_moe_ffn for norm_w (softmax gating): returns ids [k, rows] and weights [k, rows].
static void ggml_route(uint32_t n, uint32_t k, uint32_t rows, const float * logits, int32_t * ids, float * w) {
    struct ggml_init_params ip = { (size_t)8 * n * rows * sizeof(float) + 32u * 1024u * 1024u, NULL, false };
    struct ggml_context * ctx = ggml_init(ip);
    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, rows);
    memcpy(a->data, logits, (size_t)n * rows * sizeof(float));
    struct ggml_tensor * probs = ggml_soft_max(ctx, a);
    struct ggml_tensor * sel = ggml_argsort_top_k(ctx, probs, (int)k);
    struct ggml_tensor * probs3 = ggml_reshape_3d(ctx, probs, 1, n, rows);
    struct ggml_tensor * wt = ggml_get_rows(ctx, probs3, sel);
    wt = ggml_reshape_2d(ctx, wt, k, rows);
    struct ggml_tensor * sum = ggml_sum_rows(ctx, wt);
    sum = ggml_clamp(ctx, sum, 6.103515625e-5f, INFINITY);
    wt = ggml_div(ctx, wt, sum);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, wt);
    ggml_build_forward_expand(gf, sel);
    ggml_graph_compute_with_ctx(ctx, gf, 4);
    for (uint32_t t = 0; t < rows; ++t) {
        for (uint32_t j = 0; j < k; ++j) {
            ids[t * k + j] = *(const int32_t *)((const char *)sel->data + t * sel->nb[1] + j * sel->nb[0]);
            w[t * k + j] = ((const float *)wt->data)[t * k + j];
        }
    }
    ggml_free(ctx);
}

static uint32_t ulp_dist(float a, float b) {
    const uint32_t x = fb(a), y = fb(b);
    return x > y ? x - y : y - x;
}

static void test_ggml_moe(void) {
    rng_t r = { 0x66F1ull };
    const uint32_t n = 128, k = 8, rows = 512;
    float * lg = malloc((size_t)n * rows * sizeof(float));
    int32_t * gid = malloc((size_t)k * rows * sizeof(int32_t));
    float * gw = malloc((size_t)k * rows * sizeof(float));
    // 1. random rows (the continuous case the model produces)
    for (uint32_t t = 0; t < rows; ++t)
        for (uint32_t i = 0; i < n; ++i) lg[t * n + i] = (float)(rng_normal(&r) * (0.5 + 3.0 * rng_unit(&r)));
    ggml_route(n, k, rows, lg, gid, gw);
    long ids_same = 0, near = 0, ids_diff_far = 0, w_ok = 0;
    uint32_t max_ulp = 0;
    for (uint32_t t = 0; t < rows; ++t) {
        uint32_t ids[16], w[16], p[128];
        tsr_ref_moe_route(n, k, (const uint32_t *)(lg + t * n), ids, w, p);
        int same = 1;
        for (uint32_t j = 0; j < k; ++j) same &= (int32_t)ids[j] == gid[t * k + j];
        ids_same += same;
        // is the k-th / (k+1)-th probability gap large enough that both exps must agree?
        float pk = bf(p[ids[k - 1]]), pn = 0;
        for (uint32_t i = 0; i < n; ++i) {
            int in = 0;
            for (uint32_t j = 0; j < k; ++j) in |= ids[j] == i;
            if (!in && bf(p[i]) > pn) pn = bf(p[i]);
        }
        const int gap_ok = (pk - pn) > 4e-6f * pk;
        near += !gap_ok;
        if (!same && gap_ok) ++ids_diff_far;
        if (same) {
            for (uint32_t j = 0; j < k; ++j) {
                const uint32_t u = ulp_dist(bf(w[j]), gw[t * k + j]);
                if (u > max_ulp) max_ulp = u;
            }
            ++w_ok;
        }
    }
    printf("  ggml_moe random: ids identical in %ld of %u rows (%ld rows had a k-th gap below 4e-6 relative); "
           "weights of the identical rows within %u ulp\n", ids_same, rows, near, max_ulp);
    CHECK(ids_diff_far == 0, "%ld rows disagree with a large k-th gap", ids_diff_far);
    CHECK(max_ulp <= 8, "weights differ by %u ulp", max_ulp);
    CHECK(w_ok > rows * 95 / 100, "only %ld rows share the ids", w_ok);

    // 2. ties: groups of equal logits straddling the k boundary; the reference takes the smaller index
    long tie_rows = 0, tie_same = 0, tie_set_same = 0;
    for (uint32_t t = 0; t < rows; ++t) {
        for (uint32_t i = 0; i < n; ++i) lg[t * n + i] = -3.0f + 0.125f * (float)rng_below(&r, 4);
        const uint32_t hi = 3 + rng_below(&r, 5);                 // 3 .. 7 clear winners
        for (uint32_t j = 0; j < hi; ++j) lg[t * n + rng_below(&r, n)] = 4.0f + 0.5f * (float)j;
    }
    ggml_route(n, k, rows, lg, gid, gw);
    for (uint32_t t = 0; t < rows; ++t) {
        uint32_t ids[16], w[16];
        tsr_ref_moe_route(n, k, (const uint32_t *)(lg + t * n), ids, w, NULL);
        int same = 1, set_same = 1;
        for (uint32_t j = 0; j < k; ++j) {
            same &= (int32_t)ids[j] == gid[t * k + j];
            int found = 0;
            for (uint32_t m = 0; m < k; ++m) found |= (int32_t)ids[j] == gid[t * k + m];
            set_same &= found;
        }
        ++tie_rows;
        tie_same += same;
        tie_set_same += set_same;
    }
    printf("  ggml_moe ties (many equal logits at the boundary): selection order identical in %ld of %ld rows, "
           "selected set identical in %ld (ggml: std::sort, unspecified order of equal values; reference: smaller index first)\n",
           tie_same, tie_rows, tie_set_same);

    // 2b. one pair of equal logits at the boundary (rank k and k + 1 of otherwise distinct values): the
    //     reference takes the smaller index; count how often ggml's std::sort agrees, by pair position
    long pair_rows = 0, pair_lo_ggml = 0, pair_lo_ref = 0;
    for (uint32_t t = 0; t < rows; ++t) {
        for (uint32_t i = 0; i < n; ++i) lg[t * n + i] = -8.0f + 0.001f * (float)i;      // distinct, ascending with the index
        uint32_t a, b;
        do {                                       // two positions that are not among the seven winners' (j * 13 + 5) % n
            a = rng_below(&r, n);
            b = rng_below(&r, n);
        } while (a == b || (a % 13 == 5 % 13 && a < 7 * 13 + 5) || (b % 13 == 5 % 13 && b < 7 * 13 + 5));
        const uint32_t lo = a < b ? a : b, hi = a < b ? b : a;
        for (uint32_t j = 0; j < k - 1; ++j) lg[t * n + (j * 13 + 5) % n] = 3.0f + (float)j;  // seven clear winners
        lg[t * n + lo] = lg[t * n + hi] = 1.5f;                                             // the pair takes place 8 and 9
    }
    ggml_route(n, k, rows, lg, gid, gw);
    for (uint32_t t = 0; t < rows; ++t) {
        uint32_t ids[16], w[16];
        tsr_ref_moe_route(n, k, (const uint32_t *)(lg + t * n), ids, w, NULL);
        // which member of the pair holds place 8 (the last place)?
        uint32_t pair[2], np = 0;
        for (uint32_t i = 0; i < n; ++i) if (lg[t * n + i] == 1.5f && np < 2) pair[np++] = i;
        if (np < 2) continue;
        ++pair_rows;
        pair_lo_ref += (int32_t)ids[k - 1] == (int32_t)pair[0];
        pair_lo_ggml += gid[t * k + k - 1] == (int32_t)pair[0];
    }
    printf("  ggml_moe one tied pair at places 8 and 9: the smaller index is selected in %ld of %ld rows by the reference, "
           "in %ld by ggml's std::sort\n", pair_lo_ref, pair_rows, pair_lo_ggml);
    CHECK(pair_lo_ref == pair_rows, "reference ties");

    // 3. NaN rows: every probability is NaN in both; the reference selects 0 .. k-1
    long nan_rows = 0, nan_same = 0, nan_w_nan = 0;
    for (uint32_t t = 0; t < rows; ++t) {
        for (uint32_t i = 0; i < n; ++i) lg[t * n + i] = (float)rng_normal(&r);
        lg[t * n + rng_below(&r, n)] = NAN;
    }
    ggml_route(n, k, rows, lg, gid, gw);
    for (uint32_t t = 0; t < rows; ++t) {
        uint32_t ids[16], w[16];
        tsr_ref_moe_route(n, k, (const uint32_t *)(lg + t * n), ids, w, NULL);
        int same = 1, wn = 1;
        for (uint32_t j = 0; j < k; ++j) {
            same &= (int32_t)ids[j] == gid[t * k + j];
            wn &= isnan(bf(w[j])) && isnan(gw[t * k + j]);
        }
        ++nan_rows;
        nan_same += same;
        nan_w_nan += wn;
    }
    printf("  ggml_moe NaN logit: all weights NaN in both for %ld of %ld rows; ids identical in %ld "
           "(ggml's order of equal-comparing NaNs is unspecified; reference: 0 .. k-1)\n", nan_w_nan, nan_rows, nan_same);
    CHECK(nan_w_nan == nan_rows, "NaN weights");

    // 4. the clamp is the identity: the k largest of n probabilities sum to at least k / n >= 6.1e-5
    CHECK((double)k / (double)TSR_MOE_MAX_EXPERTS >= 6.103515625e-5, "clamp bound");
    free(lg); free(gid); free(gw);
}
#endif

int main(void) {
    test_directed();
    test_random();
    test_rows();
#ifdef TSR_REF_VU_GGML
    test_ggml_moe();
#else
    printf("  (ggml cross-check not built: TSR_WITH_LLAMA is off)\n");
#endif
    return test_finish("ref_moe");
}
