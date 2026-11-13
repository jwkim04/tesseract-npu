// test-tsr-headcap.cpp: the logit soft cap of a head stored in row blocks (SW-88, SW-97).
//
// The graph is Gemma's head: y = mul_mat(w, x), then SCALE (1 / cap), TANH, SCALE (cap), on the transport TSR_DEVICE selects (default sim).
// A head of N = 131072 or 262144 rows is stored in 2 or 4 row blocks at T_MAX 4. With TSR_MM_DEFER on (the default) the backend lowers the
// blocks as MUL_MAT_BEGIN / MUL_MAT_END and the cap per block slice (SW-97); the ctests run the same cases with TSR_MM_DEFER=0 and with
// TSR_HEAD_SPLIT=0 (the whole matrix, the cap whole) and compare the logits byte for byte (argv[1]: the file the logits are appended to,
// tagged as test-ggml-tesseract's TSR_TEST_DUMP) and check the records of each launch (check_headcap.py, run on TSR_DEBUG=2).
// Each case also compares the logits with the ggml CPU backend by NMSE (the two tanh differ in the last bits), and checks that a second
// run (the cached plan) and a run with new x give the same bits as a fresh run of the same inputs.
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-tesseract.h"

static int g_fail = 0;
#define CHECK(cond, ...)                                                          \
    do {                                                                          \
        if (!(cond)) {                                                            \
            fprintf(stderr, "  FAIL %s:%d: %s: ", __FILE__, __LINE__, #cond);     \
            fprintf(stderr, __VA_ARGS__);                                         \
            fprintf(stderr, "\n");                                                \
            g_fail++;                                                             \
        }                                                                         \
    } while (0)

static std::mt19937 g_rng(97);
static ggml_backend_t g_tsr = nullptr, g_cpu = nullptr;

static std::vector<float> rand_f32(size_t n, float lo, float hi) {
    std::uniform_real_distribution<float> u(lo, hi);
    std::vector<float> v(n);
    for (auto & x : v) x = u(g_rng);
    return v;
}

static void dump(FILE * f, const char * tag, const std::vector<float> & v) {
    if (f == nullptr) return;
    const uint32_t n = (uint32_t) strlen(tag);
    const uint64_t b = v.size() * 4;
    fwrite(&n, 4, 1, f);
    fwrite(tag, 1, n, f);
    fwrite(&b, 8, 1, f);
    fwrite(v.data(), 1, (size_t) b, f);
    fflush(f);
}

static double nmse(const std::vector<float> & a, const std::vector<float> & b) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        num += ((double) a[i] - b[i]) * ((double) a[i] - b[i]);
        den += (double) b[i] * b[i];
    }
    return den == 0 ? num : num / den;
}

static size_t mismatches(const std::vector<float> & a, const std::vector<float> & b) {
    size_t bad = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        uint32_t x, y;
        memcpy(&x, &a[i], 4);
        memcpy(&y, &b[i], 4);
        bad += x != y;
    }
    return bad;
}

// The logits of the graph on backend `be`.
// alloc: 0 one buffer allocation per tensor, 1 ggml's graph allocator (the chain in place on y, as llama.cpp), 2 x and the cap's first
// node share their bytes (x is dead after the head: a legal reuse), 3 no cap (the logits are the head's y).
struct headcap_run {
    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t bw = nullptr, bx = nullptr;
    ggml_gallocr_t        ga = nullptr;
    ggml_tensor *         w = nullptr, *x = nullptr, *y = nullptr, *s = nullptr, *t = nullptr, *out = nullptr;
    ggml_cgraph *         graph = nullptr;
};

static void build(headcap_run * r, ggml_backend_t be, int64_t N, int64_t K, int64_t T, bool weights, int alloc) {
    ggml_init_params ip = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true };
    r->ctx = ggml_init(ip);
    ggml_context * cw = weights ? ggml_init(ip) : r->ctx;
    r->w = ggml_new_tensor_2d(cw, GGML_TYPE_Q8_0, K, N);
    r->x = ggml_new_tensor_2d(r->ctx, GGML_TYPE_F32, K, T);
    ggml_tensor * y = ggml_mul_mat(r->ctx, r->w, r->x);
    ggml_tensor * s = ggml_scale(r->ctx, y, 1.0f / 30.0f);
    ggml_tensor * t = ggml_tanh(r->ctx, s);
    r->y = y;
    r->s = s;
    r->t = t;
    r->out = alloc == 3 ? y : ggml_scale(r->ctx, t, 30.0f);
    ggml_set_name(r->w, "output.weight");
    ggml_set_name(r->x, "result_norm");
    ggml_set_name(y, "result_output_pre");
    ggml_set_name(s, "cap_scale_in");
    ggml_set_name(t, "cap_tanh");
    ggml_set_name(r->out, alloc == 3 ? "result_output_pre" : "result_output");
    if (weights) {
        r->bw = ggml_backend_alloc_ctx_tensors(cw, be);
        ggml_backend_buffer_set_usage(r->bw, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }
    r->graph = ggml_new_graph(r->ctx);
    ggml_build_forward_expand(r->graph, r->out);
    if (!weights || alloc == 0 || alloc == 3) {
        r->bx = ggml_backend_alloc_ctx_tensors(r->ctx, be);
    } else if (alloc == 1) {
        r->ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
        CHECK(ggml_gallocr_alloc_graph(r->ga, r->graph), "gallocr");
    } else {
        const size_t big = (size_t) (N * T * 4);
        r->bx = ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(be), 4 * big + (1 << 20));
        char * base = (char *) ggml_backend_buffer_get_base(r->bx);
        CHECK(ggml_backend_tensor_alloc(r->bx, r->y, base) == GGML_STATUS_SUCCESS, "alloc y");
        CHECK(ggml_backend_tensor_alloc(r->bx, r->s, base + big) == GGML_STATUS_SUCCESS, "alloc s");
        CHECK(ggml_backend_tensor_alloc(r->bx, r->x, base + big) == GGML_STATUS_SUCCESS, "alloc x");   // x shares s's first bytes
        CHECK(ggml_backend_tensor_alloc(r->bx, r->t, base + 2 * big) == GGML_STATUS_SUCCESS, "alloc t");
        CHECK(ggml_backend_tensor_alloc(r->bx, r->out, base + 3 * big) == GGML_STATUS_SUCCESS, "alloc out");
    }
}

static std::vector<float> logits(headcap_run & r, ggml_backend_t be, int64_t N, int64_t T, const std::vector<float> & xf) {
    ggml_backend_tensor_set(r.x, xf.data(), 0, xf.size() * 4);
    std::vector<float> poison((size_t) (N * T), 3.0e38f);     // a stale logit cannot pass
    ggml_backend_tensor_set(r.out, poison.data(), 0, poison.size() * 4);
    CHECK(ggml_backend_graph_compute(be, r.graph) == GGML_STATUS_SUCCESS, "compute");
    std::vector<float> o((size_t) (N * T));
    ggml_backend_tensor_get(r.out, o.data(), 0, o.size() * 4);
    return o;
}

static void run_case(FILE * df, int64_t N, int64_t K, int64_t T, int alloc) {
    char tag[64];
    snprintf(tag, sizeof(tag), "headcap N=%" PRId64 " K=%" PRId64 " T=%" PRId64 " alloc=%d", N, K, T, alloc);
    fprintf(stderr, "headcap case: N=%" PRId64 " K=%" PRId64 " T=%" PRId64 " alloc=%d\n", N, K, T, alloc);
    const std::vector<float> wf = rand_f32((size_t) (N * K), -1.0f, 1.0f);
    const std::vector<float> xf = rand_f32((size_t) (K * T), -2.0f, 2.0f);   // logits up to about +-60: the cap saturates
    const std::vector<float> xf2 = rand_f32((size_t) (K * T), -1.0f, 1.0f);
    std::vector<uint8_t> wb(ggml_row_size(GGML_TYPE_Q8_0, K) * (size_t) N);
    ggml_quantize_chunk(GGML_TYPE_Q8_0, wf.data(), wb.data(), 0, N, K, nullptr);

    headcap_run d, c;
    build(&d, g_tsr, N, K, T, true, alloc);
    build(&c, g_cpu, N, K, T, false, alloc == 3 ? 3 : 0);
    ggml_backend_tensor_set(d.w, wb.data(), 0, wb.size());
    ggml_backend_tensor_set(c.w, wb.data(), 0, wb.size());

    const std::vector<float> od = logits(d, g_tsr, N, T, xf);
    dump(df, tag, od);
    const std::vector<float> oc = logits(c, g_cpu, N, T, xf);
    const double e = nmse(od, oc);
    CHECK(e <= 1e-8, "%s: NMSE vs the CPU backend %.3g", tag, e);
    const std::vector<float> od_again = logits(d, g_tsr, N, T, xf);
    CHECK(mismatches(od, od_again) == 0, "%s: the second (cached plan) run differs", tag);
    const std::vector<float> od2 = logits(d, g_tsr, N, T, xf2);
    dump(df, tag, od2);
    CHECK(nmse(od2, logits(c, g_cpu, N, T, xf2)) <= 1e-8, "%s: new x: NMSE vs the CPU backend", tag);

    if (d.ga != nullptr) ggml_gallocr_free(d.ga);
    if (d.bx != nullptr) ggml_backend_buffer_free(d.bx);
    ggml_backend_buffer_free(d.bw);
    ggml_backend_buffer_free(c.bx);
    ggml_free(d.ctx);
    ggml_free(c.ctx);
    printf("  %-34s NMSE(cpu)=%.2e  %s\n", tag, e, g_fail == 0 ? "ok" : "FAILED");
}

int main(int argc, char ** argv) {
    g_tsr = ggml_backend_tesseract_init();
    g_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (g_tsr == nullptr || g_cpu == nullptr) {
        fprintf(stderr, "test-tsr-headcap: no Tesseract or CPU backend\n");
        return 1;
    }
    FILE * df = argc > 1 ? fopen(argv[1], "wb") : nullptr;
    run_case(df, 131072, 256, 1, 0);      // 2 row blocks
    run_case(df, 131072, 256, 4, 0);      // T > 1: the slice is a set of strided rows
    run_case(df, 262144, 256, 1, 0);      // 4 row blocks
    run_case(df, 262144, 256, 3, 0);
    run_case(df, 262144, 256, 4, 0);      // T = T_MAX: a block holds half the ring
    run_case(df, 131072, 256, 8, 0);      // T > T_MAX: a block needs two T-tiles; the head and the cap stay whole-lowered
    run_case(df, 262144, 256, 2, 1);      // the chain in place on y (llama.cpp's allocator)
    run_case(df, 262144, 256, 3, 2);      // x shares bytes with the cap's first node: the cap stays whole
    run_case(df, 262144, 256, 3, 3);      // no cap: the blocks are not deferred
    if (df != nullptr) fclose(df);
    ggml_backend_free(g_cpu);
    ggml_backend_free(g_tsr);
    if (g_fail) {
        printf("test-tsr-headcap: %d check(s) FAILED\n", g_fail);
        return 1;
    }
    printf("test-tsr-headcap: all cases OK\n");
    return 0;
}
