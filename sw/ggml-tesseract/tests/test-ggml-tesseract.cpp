// test-ggml-tesseract.cpp — end-to-end tests of the ggml-tesseract backend through the public
// ggml API, on whatever transport TSR_DEVICE selects (default: sim).
//
// For every case the device output is compared
//   (1) BIT-EXACTLY with the Tesseract reference built from tsr_ref: repack each weight row
//       (after the spec's load-time conversion for non-native types), tsr_qact each src1 row,
//       tsr_dot_row — i.e. exactly what LD/QACT/MVM/ST must produce;
//   (2) with the ggml CPU backend, by NMSE (<= 5e-4, test-backend-ops' MUL_MAT tolerance).
// It also checks guard tensors around the output (no stray writes), weight read-back,
// split uploads, the program cache, chained MUL_MATs, an F32 src0 produced on the device,
// supports_op rejections and acceptances (llama.cpp's 512-token placement probe for output
// heads), host-memory buffers with staged MUL_MATs, and a resident weight with T > IO.
//
// TSR_TEST_DUMP=<file> appends every device output the test reads back (tagged records: u32 tag
// length, tag, u64 byte count, bytes) to <file>, so that two backends can be compared byte for byte
// on the same inputs (the inputs come from a fixed-seed generator): e.g. this binary against the v1
// libraries of build/v1 (LD_LIBRARY_PATH=build/v1/bin TSR_DEVICE=sim) and against this build's.
#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-quants.h"      // quantize_row_q8_0_ref (ggml/src)
#include "ggml-tesseract.h"

#include "tsr_isa.h"
#include "tsr_ref.h"

static int g_fail = 0;
static int g_cases = 0;

#define CHECK(cond, ...)                                                          \
    do {                                                                          \
        if (!(cond)) {                                                            \
            fprintf(stderr, "  FAIL %s:%d: %s: ", __FILE__, __LINE__, #cond);     \
            fprintf(stderr, __VA_ARGS__);                                         \
            fprintf(stderr, "\n");                                                \
            g_fail++;                                                             \
        }                                                                         \
    } while (0)

static ggml_backend_t g_tsr = nullptr;
static ggml_backend_t g_cpu = nullptr;
static std::mt19937 g_rng(42);

static void dump(const char * tag, const std::vector<float> & v) {
    static FILE * f = [] {
        const char * p = getenv("TSR_TEST_DUMP");
        return p && *p ? fopen(p, "ab") : nullptr;
    }();
    if (f == nullptr) return;
    const uint32_t n = (uint32_t) strlen(tag);
    const uint64_t b = v.size() * 4;
    fwrite(&n, 4, 1, f);
    fwrite(tag, 1, n, f);
    fwrite(&b, 8, 1, f);
    fwrite(v.data(), 1, (size_t) b, f);
    fflush(f);
}

static std::vector<float> rand_f32(size_t n, float lo = -1.0f, float hi = 1.0f) {
    std::uniform_real_distribution<float> u(lo, hi);
    std::vector<float> v(n);
    for (auto & x : v) x = u(g_rng);
    return v;
}

// f32 data -> bytes of `type` (N rows of K)
static std::vector<uint8_t> encode(ggml_type type, const std::vector<float> & f, int64_t N, int64_t K) {
    std::vector<uint8_t> out(ggml_row_size(type, K) * N);
    if (type == GGML_TYPE_F32) {
        memcpy(out.data(), f.data(), out.size());
    } else if (type == GGML_TYPE_F16) {
        ggml_fp32_to_fp16_row(f.data(), (ggml_fp16_t *) out.data(), N * K);
    } else if (type == GGML_TYPE_BF16) {
        ggml_fp32_to_bf16_row_ref(f.data(), (ggml_bf16_t *) out.data(), N * K);
    } else {
        ggml_quantize_chunk(type, f.data(), out.data(), 0, N, K, nullptr);
    }
    return out;
}

static uint32_t dev_fmt(ggml_type t) { return t == GGML_TYPE_Q4_K ? TSR_FMT_Q4_K : TSR_FMT_Q8_0; }

// Device-format repacked rows of a weight given in ggml bytes of `type` (spec 로드 시 변환).
static std::vector<std::vector<uint8_t>> ref_rows(ggml_type type, const uint8_t * a, int64_t N, int64_t K) {
    const uint32_t fmt = dev_fmt(type);
    const size_t rb = ggml_row_size(type, K);
    std::vector<std::vector<uint8_t>> rows(N);
    std::vector<float> f(K);
    std::vector<block_q8_0> q8(K / 32 + 1);
    for (int64_t n = 0; n < N; ++n) {
        const void * blk = a + n * rb;
        if (!(type == GGML_TYPE_Q8_0 || type == GGML_TYPE_Q4_K)) {
            if (type == GGML_TYPE_F32) {
                memcpy(f.data(), blk, K * 4);
            } else {
                ggml_get_type_traits(type)->to_float(blk, f.data(), K);
            }
            quantize_row_q8_0_ref(f.data(), q8.data(), K);
            blk = q8.data();
        }
        rows[n].resize((size_t) tsr_row_beats(fmt, (uint32_t) K) * 64);
        tsr_repack_row(fmt, blk, (uint32_t) K, rows[n].data());
    }
    return rows;
}

// y[t*N + n] per the spec's MVM semantics on QACT(x_t).
static std::vector<float> ref_mm(ggml_type type, const uint8_t * a, int64_t N, int64_t K, const float * x, int64_t T) {
    const uint32_t fmt = dev_fmt(type);
    const uint32_t kind = tsr_qa_kind_for_fmt(fmt);
    auto rows = ref_rows(type, a, N, K);
    std::vector<uint8_t> qa((size_t) tsr_qa_lines(kind, (uint32_t) K) * 64);
    std::vector<float> y(N * T);
    for (int64_t t = 0; t < T; ++t) {
        tsr_qact(kind, x + t * K, (uint32_t) K, qa.data());
        for (int64_t n = 0; n < N; ++n) {
            y[t * N + n] = tsr_dot_row(fmt, rows[n].data(), (uint32_t) K, qa.data());
        }
    }
    return y;
}

static double nmse(const std::vector<float> & a, const std::vector<float> & b) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        num += ((double) a[i] - b[i]) * ((double) a[i] - b[i]);
        den += (double) b[i] * b[i];
    }
    return den == 0 ? num : num / den;
}

static size_t count_bit_mismatch(const std::vector<float> & a, const std::vector<float> & b, size_t * first) {
    size_t bad = 0;
    *first = SIZE_MAX;
    for (size_t i = 0; i < a.size(); ++i) {
        uint32_t x, y;
        memcpy(&x, &a[i], 4);
        memcpy(&y, &b[i], 4);
        if (x != y) {
            if (bad == 0) *first = i;
            bad++;
        }
    }
    return bad;
}

// CPU result of mul_mat(a, x) with the same bytes.
static std::vector<float> cpu_mm(ggml_type type, const std::vector<uint8_t> & a, int64_t N, int64_t K,
                                 const std::vector<float> & x, int64_t T) {
    ggml_init_params ip = { ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * ta = ggml_new_tensor_2d(ctx, type, K, N);
    ggml_tensor * tx = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, T);
    ggml_tensor * ty = ggml_mul_mat(ctx, ta, tx);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, g_cpu);
    ggml_backend_tensor_set(ta, a.data(), 0, a.size());
    ggml_backend_tensor_set(tx, x.data(), 0, x.size() * 4);
    ggml_cgraph * g = ggml_new_graph(ctx);
    ggml_build_forward_expand(g, ty);
    ggml_backend_graph_compute(g_cpu, g);
    std::vector<float> y(N * T);
    ggml_backend_tensor_get(ty, y.data(), 0, y.size() * 4);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return y;
}

struct mm_case {
    const char * name;
    ggml_type type;
    int64_t N, K;
    int64_t t1, t2, t3;     // src1 ne[1..3]; T = t1*t2*t3
    bool weights;           // src0 in a WEIGHTS-usage buffer (eager placement) vs ANY (lazy)
    int split;              // upload src0 in this many shuffled pieces (0/1 = one set_tensor)
};

static void upload(ggml_tensor * t, const std::vector<uint8_t> & bytes, int split) {
    if (split <= 1) {
        ggml_backend_tensor_set(t, bytes.data(), 0, bytes.size());
        return;
    }
    // uneven pieces in shuffled order
    std::vector<size_t> cuts = { 0, bytes.size() };
    for (int i = 1; i < split; ++i) cuts.push_back(g_rng() % bytes.size());
    std::sort(cuts.begin(), cuts.end());
    cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());
    std::vector<size_t> order(cuts.size() - 1);
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::shuffle(order.begin(), order.end(), g_rng);
    for (size_t i : order) {
        ggml_backend_tensor_set(t, bytes.data() + cuts[i], cuts[i], cuts[i + 1] - cuts[i]);
    }
}

static void run_case(const mm_case & c) {
    g_cases++;
    const int fail0 = g_fail;
    const int64_t T = c.t1 * c.t2 * c.t3;
    ggml_init_params ip = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx   = ggml_init(ip);
    ggml_context * ctx_w = c.weights ? ggml_init(ip) : ctx;

    ggml_tensor * g0  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 48);
    ggml_tensor * a   = ggml_new_tensor_2d(ctx_w, c.type, c.K, c.N);
    ggml_tensor * x   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.K, c.t1, c.t2, c.t3);
    ggml_tensor * y   = ggml_mul_mat(ctx, a, x);
    ggml_tensor * g1  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 48);
    ggml_set_name(a, "a");
    ggml_set_name(x, "x");
    ggml_set_name(y, "y");

    CHECK(ggml_backend_supports_op(g_tsr, y), "%s: supports_op(unallocated) is false", c.name);

    ggml_backend_buffer_t buf_w = nullptr;
    if (c.weights) {
        buf_w = ggml_backend_alloc_ctx_tensors(ctx_w, g_tsr);
        ggml_backend_buffer_set_usage(buf_w, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, g_tsr);
    CHECK(ggml_backend_supports_op(g_tsr, y), "%s: supports_op(allocated) is false", c.name);

    const std::vector<float>   af = rand_f32(c.N * c.K);
    const std::vector<uint8_t> ab = encode(c.type, af, c.N, c.K);
    const std::vector<float>   xf = rand_f32(c.K * T);
    std::vector<uint8_t> guard(48 * 4, 0xC3);

    upload(a, ab, c.split);
    ggml_backend_tensor_set(x, xf.data(), 0, xf.size() * 4);
    ggml_backend_tensor_set(g0, guard.data(), 0, guard.size());
    ggml_backend_tensor_set(g1, guard.data(), 0, guard.size());
    // poison the output so stale data cannot pass
    std::vector<uint8_t> poison(ggml_nbytes(y), 0x7F);
    ggml_backend_tensor_set(y, poison.data(), 0, poison.size());

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    CHECK(ggml_backend_graph_compute(g_tsr, graph) == GGML_STATUS_SUCCESS, "%s: compute", c.name);

    std::vector<float> yd(c.N * T);
    ggml_backend_tensor_get(y, yd.data(), 0, yd.size() * 4);
    dump(c.name, yd);

    const std::vector<float> yr = ref_mm(c.type, ab.data(), c.N, c.K, xf.data(), T);
    size_t first = 0;
    const size_t bad = count_bit_mismatch(yd, yr, &first);
    CHECK(bad == 0, "%s: %zu/%zu outputs differ from the tsr_ref oracle (first at %zu: dev %.9g ref %.9g)",
          c.name, bad, yd.size(), first, bad ? yd[first] : 0.0, bad ? yr[first] : 0.0);

    const std::vector<float> yc = cpu_mm(c.type, ab, c.N, c.K, xf, T);
    const double e = nmse(yd, yc);
    CHECK(e <= 5e-4, "%s: NMSE vs CPU %.3g > 5e-4", c.name, e);

    std::vector<uint8_t> gback(guard.size());
    ggml_backend_tensor_get(g0, gback.data(), 0, gback.size());
    CHECK(gback == guard, "%s: guard before the output was modified", c.name);
    ggml_backend_tensor_get(g1, gback.data(), 0, gback.size());
    CHECK(gback == guard, "%s: guard after the output was modified", c.name);

    // weight read-back: exact unless the weight was converted at load (WEIGHTS buffer, non-native type)
    std::vector<uint8_t> aback(ab.size());
    ggml_backend_tensor_get(a, aback.data(), 0, aback.size());
    const bool native = c.type == GGML_TYPE_Q8_0 || c.type == GGML_TYPE_Q4_K;
    if (native || !c.weights) {
        CHECK(aback == ab, "%s: src0 read-back differs", c.name);
    }
    // partial read-back of a slice
    if (ab.size() > 100 && (native || !c.weights)) {
        std::vector<uint8_t> part(37);
        ggml_backend_tensor_get(a, part.data(), 50, part.size());
        CHECK(memcmp(part.data(), ab.data() + 50, part.size()) == 0, "%s: partial src0 read-back differs", c.name);
    }

    // second run: program cache (and lazily converted src0 cache) must give identical bits
    ggml_backend_tensor_set(y, poison.data(), 0, poison.size());
    ggml_backend_graph_compute(g_tsr, graph);
    std::vector<float> yd2(yd.size());
    ggml_backend_tensor_get(y, yd2.data(), 0, yd2.size() * 4);
    CHECK(count_bit_mismatch(yd2, yd, &first) == 0, "%s: second (cached) run differs", c.name);

    // re-upload different src1 data: the cached program must read the new data
    const std::vector<float> xf2 = rand_f32(c.K * T);
    ggml_backend_tensor_set(x, xf2.data(), 0, xf2.size() * 4);
    ggml_backend_graph_compute(g_tsr, graph);
    ggml_backend_tensor_get(y, yd2.data(), 0, yd2.size() * 4);
    dump(c.name, yd2);
    const std::vector<float> yr2 = ref_mm(c.type, ab.data(), c.N, c.K, xf2.data(), T);
    CHECK(count_bit_mismatch(yd2, yr2, &first) == 0, "%s: run with new src1 differs from the oracle", c.name);

    // re-upload src0 (ANY buffer: the lazy conversion must be invalidated)
    if (!c.weights) {
        const std::vector<float>   af3 = rand_f32(c.N * c.K);
        const std::vector<uint8_t> ab3 = encode(c.type, af3, c.N, c.K);
        ggml_backend_tensor_set(a, ab3.data(), 0, ab3.size());
        ggml_backend_graph_compute(g_tsr, graph);
        ggml_backend_tensor_get(y, yd2.data(), 0, yd2.size() * 4);
        dump(c.name, yd2);
        const std::vector<float> yr3 = ref_mm(c.type, ab3.data(), c.N, c.K, xf2.data(), T);
        CHECK(count_bit_mismatch(yd2, yr3, &first) == 0, "%s: run after src0 re-upload differs (stale conversion?)", c.name);
    }

    ggml_backend_buffer_free(buf);
    if (buf_w) ggml_backend_buffer_free(buf_w);
    if (ctx_w != ctx) ggml_free(ctx_w);
    ggml_free(ctx);
    printf("  %-40s %-5s N=%-6" PRId64 " K=%-6" PRId64 " T=%-4" PRId64 " %-7s NMSE(cpu)=%.2e  %s\n", c.name,
           ggml_type_name(c.type), c.N, c.K, T, c.weights ? "WEIGHTS" : "ANY", e, g_fail == fail0 ? "ok" : "FAILED");
}

// h = a1 * x (device), y = a2 * reshape(h) (device, same graph): src1 produced on the device.
// With N1 % 16 != 0 and T > 1, h goes through the line-padded scratch + host compaction, and
// the second MUL_MAT must see the compacted rows (the backend flushes before it).
static void run_chain(int64_t K, int64_t N1, int64_t T, int64_t K2, int64_t N2, ggml_type t2) {
    g_cases++;
    const int fail0 = g_fail;
    GGML_ASSERT((N1 * T) % K2 == 0);
    const int64_t T2 = N1 * T / K2;
    ggml_init_params ip = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * a1 = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, K, N1);
    ggml_tensor * a2 = ggml_new_tensor_2d(ctx, t2, K2, N2);
    ggml_tensor * x  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, T);
    ggml_tensor * h  = ggml_mul_mat(ctx, a1, x);                    // [N1, T]
    ggml_tensor * hr = ggml_reshape_2d(ctx, h, K2, T2);             // [K2, T2]
    ggml_tensor * y  = ggml_mul_mat(ctx, a2, hr);                   // [N2, T2]
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, g_tsr);
    CHECK(ggml_backend_supports_op(g_tsr, y), "chain: second MUL_MAT not supported");
    const auto ab1 = encode(GGML_TYPE_Q8_0, rand_f32(N1 * K), N1, K);
    const auto ab2 = encode(t2, rand_f32(N2 * K2), N2, K2);
    const auto xf  = rand_f32(K * T);
    ggml_backend_tensor_set(a1, ab1.data(), 0, ab1.size());
    ggml_backend_tensor_set(a2, ab2.data(), 0, ab2.size());
    ggml_backend_tensor_set(x, xf.data(), 0, xf.size() * 4);
    ggml_cgraph * g = ggml_new_graph(ctx);
    ggml_build_forward_expand(g, y);
    ggml_backend_graph_compute(g_tsr, g);
    std::vector<float> hd(N1 * T), yd(N2 * T2);
    ggml_backend_tensor_get(h, hd.data(), 0, hd.size() * 4);
    ggml_backend_tensor_get(y, yd.data(), 0, yd.size() * 4);
    dump("chain h", hd);
    dump("chain y", yd);
    const auto href = ref_mm(GGML_TYPE_Q8_0, ab1.data(), N1, K, xf.data(), T);
    const auto yref = ref_mm(t2, ab2.data(), N2, K2, href.data(), T2);
    size_t first;
    CHECK(count_bit_mismatch(hd, href, &first) == 0, "chain: h differs");
    CHECK(count_bit_mismatch(yd, yref, &first) == 0, "chain: y differs");
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    printf("  %-40s K=%" PRId64 " N1=%" PRId64 " T=%" PRId64 " -> %s K2=%" PRId64 " N2=%" PRId64 " T2=%" PRId64 "  %s\n",
           "chain (src1 produced on the device)", K, N1, T, ggml_type_name(t2), K2, N2, T2, g_fail == fail0 ? "ok" : "FAILED");
}

// h = a1 * x (device, F32 [N1, T]); y = reshape(h) * z: the device-produced F32 h is src0
// (converted lazily at graph time, after the producing program ran).
static void run_src0_from_device(int64_t K, int64_t N1, int64_t T, int64_t K2, int64_t T2) {
    g_cases++;
    const int fail0 = g_fail;
    GGML_ASSERT((N1 * T) % K2 == 0);
    const int64_t N2 = N1 * T / K2;
    ggml_init_params ip = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * a1 = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, K, N1);
    ggml_tensor * x  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, T);
    ggml_tensor * z  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K2, T2);
    ggml_tensor * h  = ggml_mul_mat(ctx, a1, x);                    // [N1, T]
    ggml_tensor * hr = ggml_reshape_2d(ctx, h, K2, N2);             // src0 [K2, N2]
    ggml_tensor * y  = ggml_mul_mat(ctx, hr, z);                    // [N2, T2]
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, g_tsr);
    CHECK(ggml_backend_supports_op(g_tsr, y), "src0-from-device: not supported");
    const auto ab1 = encode(GGML_TYPE_Q8_0, rand_f32(N1 * K), N1, K);
    ggml_backend_tensor_set(a1, ab1.data(), 0, ab1.size());
    ggml_cgraph * g = ggml_new_graph(ctx);
    ggml_build_forward_expand(g, y);
    for (int rep = 0; rep < 2; ++rep) {
        const auto xf = rand_f32(K * T);       // new data each repetition: the cached conversion must be redone
        const auto zf = rand_f32(K2 * T2);
        ggml_backend_tensor_set(x, xf.data(), 0, xf.size() * 4);
        ggml_backend_tensor_set(z, zf.data(), 0, zf.size() * 4);
        ggml_backend_graph_compute(g_tsr, g);
        std::vector<float> yd(N2 * T2);
        ggml_backend_tensor_get(y, yd.data(), 0, yd.size() * 4);
        dump("src0 from device y", yd);
        const auto href = ref_mm(GGML_TYPE_Q8_0, ab1.data(), N1, K, xf.data(), T);
        std::vector<uint8_t> hb(href.size() * 4);
        memcpy(hb.data(), href.data(), hb.size());
        const auto yref = ref_mm(GGML_TYPE_F32, hb.data(), N2, K2, zf.data(), T2);
        size_t first;
        CHECK(count_bit_mismatch(yd, yref, &first) == 0, "src0-from-device rep %d: y differs", rep);
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    printf("  %-40s K=%" PRId64 " N1=%" PRId64 " T=%" PRId64 " -> K2=%" PRId64 " N2=%" PRId64 " T2=%" PRId64 "  %s\n",
           "F32 src0 produced on the device", K, N1, T, K2, N2, T2, g_fail == fail0 ? "ok" : "FAILED");
}

// memset_tensor / clear on LINEAR (ANY) and PLACED (WEIGHTS) buffers, partial-row update of a
// complete native weight (read-modify-write), and that the device uses the updated weight.
static void run_memset_clear() {
    g_cases++;
    const int fail0 = g_fail;
    const int64_t K = 256, N = 24, T = 2;
    ggml_init_params ip = { ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true };
    ggml_context * cw = ggml_init(ip);
    ggml_context * ca = ggml_init(ip);
    ggml_tensor * w  = ggml_new_tensor_2d(cw, GGML_TYPE_Q8_0, K, N);
    ggml_tensor * v  = ggml_new_tensor_1d(cw, GGML_TYPE_F32, 100);        // RAW (AUX) in the weight buffer
    ggml_tensor * x  = ggml_new_tensor_2d(ca, GGML_TYPE_F32, K, T);
    ggml_tensor * y  = ggml_mul_mat(ca, w, x);
    ggml_backend_buffer_t bw = ggml_backend_alloc_ctx_tensors(cw, g_tsr);
    ggml_backend_buffer_set_usage(bw, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_buffer_t ba = ggml_backend_alloc_ctx_tensors(ca, g_tsr);

    // LINEAR: memset a sub-range
    const auto xf = rand_f32(K * T);
    ggml_backend_tensor_set(x, xf.data(), 0, xf.size() * 4);
    ggml_backend_tensor_memset(x, 0x00, 100 * 4, 50 * 4);
    std::vector<float> xb(K * T);
    ggml_backend_tensor_get(x, xb.data(), 0, xb.size() * 4);
    auto xe = xf;
    std::fill(xe.begin() + 100, xe.begin() + 150, 0.0f);
    CHECK(memcmp(xb.data(), xe.data(), xb.size() * 4) == 0, "LINEAR memset");

    // PLACED: whole-tensor memset of a never-written weight, then read back
    ggml_backend_tensor_memset(w, 0x11, 0, ggml_nbytes(w));
    std::vector<uint8_t> wb(ggml_nbytes(w));
    ggml_backend_tensor_get(w, wb.data(), 0, wb.size());
    CHECK(std::all_of(wb.begin(), wb.end(), [](uint8_t c) { return c == 0x11; }), "PLACED whole memset");

    // upload real data, then a partial-row update (read-modify-write path) and a whole-row update
    auto wbytes = encode(GGML_TYPE_Q8_0, rand_f32(N * K), N, K);
    ggml_backend_tensor_set(w, wbytes.data(), 0, wbytes.size());
    const size_t rb = ggml_row_size(GGML_TYPE_Q8_0, K);
    const auto patch = encode(GGML_TYPE_Q8_0, rand_f32(K), 1, K);
    ggml_backend_tensor_set(w, patch.data() + 5, 3 * rb + 5, 40);          // inside row 3
    memcpy(wbytes.data() + 3 * rb + 5, patch.data() + 5, 40);
    ggml_backend_tensor_set(w, patch.data(), 7 * rb, rb);                  // whole row 7
    memcpy(wbytes.data() + 7 * rb, patch.data(), rb);
    ggml_backend_tensor_memset(w, 0x00, 11 * rb + 2, 20);                  // memset inside row 11 (keeps d)
    memset(wbytes.data() + 11 * rb + 2, 0, 20);
    ggml_backend_tensor_get(w, wb.data(), 0, wb.size());
    CHECK(wb == wbytes, "PLACED partial updates read back");

    ggml_cgraph * g = ggml_new_graph(ca);
    ggml_build_forward_expand(g, y);
    ggml_backend_graph_compute(g_tsr, g);
    std::vector<float> yd(N * T);
    ggml_backend_tensor_get(y, yd.data(), 0, yd.size() * 4);
    dump("memset/clear y", yd);
    const auto yr = ref_mm(GGML_TYPE_Q8_0, wbytes.data(), N, K, xe.data(), T);
    size_t first;
    CHECK(count_bit_mismatch(yd, yr, &first) == 0, "MUL_MAT after partial weight updates");

    // RAW tensor in the weight buffer
    const auto vf = rand_f32(100);
    ggml_backend_tensor_set(v, vf.data(), 0, 400);
    std::vector<float> vb(100);
    ggml_backend_tensor_get(v, vb.data(), 0, 400);
    CHECK(vb == vf, "RAW round trip");

    // clear: every placed tensor reads back as zero
    ggml_backend_buffer_clear(bw, 0);
    ggml_backend_tensor_get(w, wb.data(), 0, wb.size());
    CHECK(std::all_of(wb.begin(), wb.end(), [](uint8_t c) { return c == 0; }), "PLACED clear (weight)");
    ggml_backend_tensor_get(v, vb.data(), 0, 400);
    CHECK(std::all_of(vb.begin(), vb.end(), [](float f) { return f == 0.0f; }), "PLACED clear (raw)");
    ggml_backend_buffer_clear(ba, 0xFF);
    ggml_backend_tensor_get(x, xb.data(), 0, xb.size() * 4);
    std::vector<uint8_t> xr(xb.size() * 4);
    memcpy(xr.data(), xb.data(), xr.size());
    CHECK(std::all_of(xr.begin(), xr.end(), [](uint8_t c) { return c == 0xFF; }), "LINEAR clear");

    ggml_backend_buffer_free(ba);
    ggml_backend_buffer_free(bw);
    ggml_free(ca);
    ggml_free(cw);
    printf("  %-40s %s\n", "memset / clear / partial weight updates", g_fail == fail0 ? "ok" : "FAILED");
}

static void check_supports_op() {
    g_cases++;
    const int fail0 = g_fail;
    ggml_init_params ip = { ggml_tensor_overhead() * 32 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    auto mm = [&](ggml_tensor * a, ggml_tensor * b) { return ggml_mul_mat(ctx, a, b); };
    ggml_tensor * w  = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, 256, 64);
    ggml_tensor * x  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 4);
    ggml_tensor * xh = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 256, 4);
    ggml_tensor * w3 = ggml_new_tensor_3d(ctx, GGML_TYPE_Q8_0, 256, 64, 2);
    ggml_tensor * w48 = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 48, 64);
    ggml_tensor * x48 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 48, 4);
    ggml_tensor * wbig = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, 32768 + 32, 4);
    ggml_tensor * xbig = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32768 + 32, 1);
    ggml_tensor * xt = ggml_transpose(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 256));   // non-contiguous
    ggml_tensor * wi = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 256, 64);

    CHECK(ggml_backend_supports_op(g_tsr, mm(w, x)), "plain Q4_K");
    CHECK(!ggml_backend_supports_op(g_tsr, mm(w, xh)), "F16 src1 must be rejected");
    CHECK(!ggml_backend_supports_op(g_tsr, mm(w3, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 256, 4, 2))), "3D src0 must be rejected");
    CHECK(!ggml_backend_supports_op(g_tsr, mm(w48, x48)), "K=48 must be rejected (Q8_0 conversion needs K%%32)");
    CHECK(!ggml_backend_supports_op(g_tsr, mm(wbig, xbig)), "K > K_MAX must be rejected");
    CHECK(!ggml_backend_supports_op(g_tsr, mm(w, xt)), "non-contiguous src1 must be rejected");
    CHECK(!ggml_backend_supports_op(g_tsr, ggml_add(ctx, x, x)), "ADD must be rejected");
    (void) wi;

    // src0 in a CPU buffer: rejected (weights must live on the device)
    ggml_context * cctx = ggml_init(ip);
    ggml_tensor * wc = ggml_new_tensor_2d(cctx, GGML_TYPE_Q8_0, 256, 64);
    ggml_backend_buffer_t cbuf = ggml_backend_alloc_ctx_tensors(cctx, g_cpu);
    CHECK(!ggml_backend_supports_op(g_tsr, mm(wc, x)), "src0 in a CPU buffer must be rejected");

    // view with offset of a WEIGHTS tensor: rejected; a src1 view at a non-64 offset: rejected
    ggml_context * wctx = ggml_init(ip);
    ggml_tensor * ww = ggml_new_tensor_2d(wctx, GGML_TYPE_Q8_0, 256, 64);
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(wctx, g_tsr);
    ggml_backend_buffer_set_usage(wbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_tensor * wv = ggml_view_2d(ctx, ww, 256, 32, ww->nb[1], 32 * ww->nb[1]);
    CHECK(ggml_backend_supports_op(g_tsr, mm(ww, x)), "WEIGHTS tensor");
    CHECK(!ggml_backend_supports_op(g_tsr, mm(wv, x)), "row slice of a WEIGHTS tensor must be rejected");
    ggml_tensor * xs = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256 * 4 + 8);
    ggml_tensor * xv = ggml_view_2d(ctx, xs, 256, 4, 256 * 4, 8 * 4);
    CHECK(!ggml_backend_supports_op(g_tsr, mm(w, xv)), "src1 view at offset 32 B must be rejected");

    // src0 that is not a weight (backend review #0). Before allocation (test-backend-ops) a plain
    // leaf is accepted; a graph input (llama.cpp: attn_inp_k_rot, the Q rotation of a quantized
    // KV cache) or the result of another op is an activation and must stay on the CPU.
    ggml_tensor * rot = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
    ggml_set_input(rot);
    ggml_tensor * q64 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 14);
    CHECK(!ggml_backend_supports_op(g_tsr, mm(rot, q64)), "unallocated graph-input src0 must be rejected");
    ggml_tensor * leaf = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
    CHECK(ggml_backend_supports_op(g_tsr, mm(leaf, q64)), "unallocated leaf src0 (test-backend-ops) must be accepted");
    ggml_tensor * act = ggml_cont(ctx, ggml_transpose(ctx, leaf));
    CHECK(!ggml_backend_supports_op(g_tsr, mm(act, q64)), "unallocated op-produced src0 must be rejected");
    ggml_tensor * actv = ggml_reshape_2d(ctx, mm(leaf, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64)), 64, 64);
    CHECK(!ggml_backend_supports_op(g_tsr, mm(actv, q64)), "unallocated view of an op result as src0 must be rejected");

    ggml_backend_buffer_free(cbuf);
    ggml_backend_buffer_free(wbuf);
    ggml_free(cctx);
    ggml_free(wctx);
    ggml_free(ctx);
    printf("  %-40s %s\n", "supports_op rejections", g_fail == fail0 ? "ok" : "FAILED");
}

// llama.cpp's weight placement probe (weight_buft_supported, src/llama-model-loader.cpp:961-965
// and :1065-1069 at the pin): the GGUF meta tensor gets a size-0 buffer of the candidate buffer
// type and the op is built with an unallocated F32 src1 of 512 tokens, whatever -ub is.
static bool probe_like_llama(ggml_type type, int64_t K, int64_t N, int64_t tokens) {
    ggml_init_params ip = { ggml_tensor_overhead() * 8, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * w = ggml_new_tensor_2d(ctx, type, K, N);
    ggml_tensor * b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, w->ne[0], tokens, w->ne[2], w->ne[3]);
    ggml_tensor * op = ggml_mul_mat(ctx, w, b);
    w->buffer = ggml_backend_buft_alloc_buffer(ggml_backend_tesseract_buffer_type(), 0);
    const bool ok = ggml_backend_dev_supports_op(ggml_backend_get_device(g_tsr), op);
    ggml_backend_buffer_free(w->buffer);
    w->buffer = nullptr;
    ggml_free(ctx);
    return ok;
}

// Output heads (backend review #1 / contract #0): the placement probe must accept them, so that
// output.weight goes to the device; and a resident weight must not be rejected at run time for
// a large token count (#2), which would make the scheduler read it back for the CPU.
static void check_output_head_placement() {
    g_cases++;
    const int fail0 = g_fail;
    // Qwen2.5-0.5B head: output.weight Q8_0 [896 x 151936]: dst at 512 tokens = 296.75 MiB
    CHECK(probe_like_llama(GGML_TYPE_Q8_0, 896, 151936, 512), "Qwen2.5-0.5B output.weight probe rejected");
    // Qwen2.5-32B head: output.weight Q6_K [5120 x 152064] (converted to Q8_0 at load)
    CHECK(probe_like_llama(GGML_TYPE_Q6_K, 5120, 152064, 512), "Qwen2.5-32B output.weight probe rejected");
    // Qwen2.5-32B with a Q4_K head (Q4_K_S-like files) and a Llama-3 vocabulary
    CHECK(probe_like_llama(GGML_TYPE_Q4_K, 5120, 152064, 512), "Q4_K head probe rejected");
    CHECK(probe_like_llama(GGML_TYPE_Q8_0, 4096, 128256, 512), "Llama-3-8B head probe rejected");
    // the probe is token-count independent
    CHECK(probe_like_llama(GGML_TYPE_Q6_K, 5120, 152064, 8192), "32B head probe at 8192 tokens rejected");

    // ... and the 32B head would fit: converted to Q8_0 it takes R x ROW_BEATS beats per channel,
    // one token of it fits the SPM (x + QA + y lines) and the IO stage (x + line-padded y).
    const uint32_t N = 152064, K = 5120;
    size_t free_b = 0, total_b = 0;
    ggml_backend_dev_memory(ggml_backend_get_device(g_tsr), &free_b, &total_b);
    const uint64_t num_ch   = 4;    // U200 v1 / sim default (spec 구성 파라미터)
    const uint64_t per_ch   = tsr_tensor_ch_beats(TSR_FMT_Q8_0, N, K, (uint32_t) num_ch) * 64;
    const uint64_t spm_need = (uint64_t) K * 4 / 64 + tsr_qa_lines(TSR_QA_A32, K) + ((uint64_t) N * 4 + 63) / 64;
    CHECK(per_ch == (uint64_t) 38016 * 85 * 64, "32B head per-channel size %" PRIu64, per_ch);
    CHECK(per_ch * num_ch <= free_b, "32B head (%.1f MiB/ch) does not fit the WEIGHT heap (%.1f MiB free)",
          per_ch / 1048576.0, free_b / 1048576.0);
    CHECK(spm_need * 64 <= 4u * 1024 * 1024, "32B head: one token needs %" PRIu64 " SPM lines", spm_need);
    CHECK((uint64_t) K * 4 + ((uint64_t) N * 4 + 63) / 64 * 64 <= 16u << 20, "32B head: one token does not fit the IO stage");

    // resident weight (WEIGHTS buffer), 16384 tokens: src1 64 MiB + dst 256 MiB > IO region
    ggml_init_params ip = { ggml_tensor_overhead() * 8, nullptr, true };
    ggml_context * wctx = ggml_init(ip);
    ggml_context * ctx  = ggml_init(ip);
    ggml_tensor * w = ggml_new_tensor_2d(wctx, GGML_TYPE_IQ2_XXS, 1024, 4096);
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(wctx, g_tsr);
    ggml_backend_buffer_set_usage(wbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1024, 16384);
    CHECK(ggml_backend_supports_op(g_tsr, ggml_mul_mat(ctx, w, x)), "resident weight rejected at T=16384");
    ggml_backend_buffer_free(wbuf);
    ggml_free(wctx);
    ggml_free(ctx);
    printf("  %-40s %s\n", "output head placement probe (0.5B/32B)", g_fail == fail0 ? "ok" : "FAILED");
}

// A resident weight with more tokens than the IO region holds (backend review #2), through the
// real scheduler: the MUL_MAT must stay on Tesseract (before, supports_op rejected it and the
// scheduler copied the weight back to the CPU on every compute, aborting for IQ2_XXS), and the
// 256 MiB dst chunk is kept in host memory and filled in stage tiles.
static void run_sched_large_T() {
    g_cases++;
    const int fail0 = g_fail;
    const int64_t K = 32, N = 4096, T = 16384;           // dst 256 MiB > 240 MiB of IO windows
    ggml_init_params ip = { ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true };
    ggml_context * cw = ggml_init(ip);
    ggml_tensor * w = ggml_new_tensor_2d(cw, GGML_TYPE_Q8_0, K, N);
    ggml_backend_buffer_t bw = ggml_backend_alloc_ctx_tensors(cw, g_tsr);
    ggml_backend_buffer_set_usage(bw, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    const auto wb = encode(GGML_TYPE_Q8_0, rand_f32(N * K), N, K);
    ggml_backend_tensor_set(w, wb.data(), 0, wb.size());

    ggml_context * ca = ggml_init(ip);
    ggml_tensor * x = ggml_new_tensor_2d(ca, GGML_TYPE_F32, K, T);
    ggml_set_input(x);
    ggml_tensor * y = ggml_mul_mat(ca, w, x);
    ggml_set_output(y);
    ggml_cgraph * g = ggml_new_graph(ca);
    ggml_build_forward_expand(g, y);
    ggml_backend_t backends[2] = { g_tsr, g_cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 2, 64, false, true);
    CHECK(ggml_backend_sched_alloc_graph(sched, g), "large T: sched alloc");
    CHECK(ggml_backend_sched_get_tensor_backend(sched, y) == g_tsr, "large T: MUL_MAT not assigned to Tesseract");
    const auto xf = rand_f32(K * T);
    ggml_backend_tensor_set(x, xf.data(), 0, xf.size() * 4);
    CHECK(ggml_backend_sched_graph_compute(sched, g) == GGML_STATUS_SUCCESS, "large T: compute");
    // spot-check token ranges at the start, around stage-tile boundaries and at the end
    for (int64_t t0 : { (int64_t) 0, (int64_t) 1020, T / 2 - 3, T - 7 }) {
        const int64_t nt = 7;
        std::vector<float> yd(N * nt);
        ggml_backend_tensor_get(y, yd.data(), (size_t) (t0 * N * 4), yd.size() * 4);
        dump("large T y", yd);
        const auto yr = ref_mm(GGML_TYPE_Q8_0, wb.data(), N, K, xf.data() + t0 * K, nt);
        size_t first;
        CHECK(count_bit_mismatch(yd, yr, &first) == 0, "large T: tokens [%" PRId64 ", +7) differ", t0);
    }
    ggml_backend_sched_free(sched);
    ggml_backend_buffer_free(bw);
    ggml_free(ca);
    ggml_free(cw);
    printf("  %-40s N=%" PRId64 " K=%" PRId64 " T=%" PRId64 "  %s\n", "resident weight, dst > IO (scheduler)", N, K, T,
           g_fail == fail0 ? "ok" : "FAILED");
}

typedef const char * (*buffer_storage_fn)(ggml_backend_buffer_t);

static const char * storage_of(ggml_backend_buffer_t b) {
    static buffer_storage_fn fn = (buffer_storage_fn) ggml_backend_reg_get_proc_address(
        ggml_backend_dev_backend_reg(ggml_backend_get_device(g_tsr)), "ggml_backend_tesseract_buffer_storage");
    return fn ? fn(b) : "?";
}

// Buffers that do not get an IO window (backend review #1/#2/#3):
//   - an activation buffer that does not fit the free IO windows lives in host memory and its
//     MUL_MATs are staged through IO in T tiles (here forced by filling IO first);
//   - an ANY buffer of leaf tensors only (LoRA adapter, control vector) lives in host memory
//     and takes no IO; a MUL_MAT src0 in it is converted into the WEIGHT heap.
static void run_hosted_buffers() {
    g_cases++;
    const int fail0 = g_fail;
    const bool forced = getenv("TSR_FORCE_HOST") != nullptr && atoi(getenv("TSR_FORCE_HOST")) != 0;
    ggml_init_params ip = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true };

    // 1. a LoRA-like adapter: ANY usage, only leaves (lora_a [K, r], lora_b [r, N])
    const int64_t K = 512, R = 32, N = 96, T = 40;
    ggml_context * lctx = ggml_init(ip);
    ggml_tensor * la = ggml_new_tensor_2d(lctx, GGML_TYPE_F16, K, R);
    ggml_tensor * lb = ggml_new_tensor_2d(lctx, GGML_TYPE_F32, R, N);
    ggml_backend_buffer_t lbuf = ggml_backend_alloc_ctx_tensors(lctx, g_tsr);
    const auto lab = encode(GGML_TYPE_F16, rand_f32(K * R), R, K);
    const auto lbb = encode(GGML_TYPE_F32, rand_f32(R * N), N, R);
    ggml_backend_tensor_set(la, lab.data(), 0, lab.size());
    ggml_backend_tensor_set(lb, lbb.data(), 0, lbb.size());
    CHECK(strcmp(storage_of(lbuf), "host") == 0, "leaf-only ANY buffer: storage %s, expected host", storage_of(lbuf));
    std::vector<uint8_t> back(lab.size());
    ggml_backend_tensor_get(la, back.data(), 0, back.size());
    CHECK(back == lab, "leaf-only ANY buffer: exact read-back");

    // 2. fill the IO windows with a non-leaf ANY buffer, so that the next one cannot get a window
    size_t io_cap = ggml_backend_buft_get_max_size(ggml_backend_tesseract_buffer_type());
    ggml_context * fctx = ggml_init(ip);
    ggml_tensor * filler = ggml_new_tensor_1d(fctx, GGML_TYPE_F32, (int64_t) (io_cap / 4) - 4096);
    ggml_tensor * fop = ggml_scale(fctx, ggml_new_tensor_1d(fctx, GGML_TYPE_F32, 16), 2.0f);   // not a leaf
    ggml_backend_buffer_t fbuf = ggml_backend_alloc_ctx_tensors(fctx, g_tsr);
    std::vector<float> small(16, 1.0f);
    ggml_backend_tensor_set(fop, small.data(), 0, 64);          // first access: takes an IO window
    CHECK(strcmp(storage_of(fbuf), forced ? "host" : "io") == 0, "filler: storage %s", storage_of(fbuf));
    (void) filler;

    // 3. activations that no longer fit IO: h = la * x (src0 from the leaf buffer), y = lb * h
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, T);
    ggml_tensor * h = ggml_mul_mat(ctx, la, x);       // [R, T]
    ggml_tensor * y = ggml_mul_mat(ctx, lb, h);       // [N, T]
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, g_tsr);
    CHECK(ggml_backend_supports_op(g_tsr, h) && ggml_backend_supports_op(g_tsr, y), "hosted: supports_op");
    const auto xf = rand_f32(K * T);
    ggml_backend_tensor_set(x, xf.data(), 0, xf.size() * 4);
    CHECK(strcmp(storage_of(buf), "host") == 0, "activations without a free IO window: storage %s, expected host", storage_of(buf));
    ggml_cgraph * g = ggml_new_graph(ctx);
    ggml_build_forward_expand(g, y);
    for (int rep = 0; rep < 2; ++rep) {
        CHECK(ggml_backend_graph_compute(g_tsr, g) == GGML_STATUS_SUCCESS, "hosted: compute");
        std::vector<float> hd(R * T), yd(N * T);
        ggml_backend_tensor_get(h, hd.data(), 0, hd.size() * 4);
        ggml_backend_tensor_get(y, yd.data(), 0, yd.size() * 4);
        dump("hosted h", hd);
        dump("hosted y", yd);
        const auto href = ref_mm(GGML_TYPE_F16, lab.data(), R, K, xf.data(), T);
        const auto yref = ref_mm(GGML_TYPE_F32, lbb.data(), N, R, href.data(), T);
        size_t first;
        CHECK(count_bit_mismatch(hd, href, &first) == 0, "hosted rep %d: h differs", rep);
        CHECK(count_bit_mismatch(yd, yref, &first) == 0, "hosted rep %d: y differs", rep);
    }
    ggml_backend_buffer_free(buf);
    ggml_backend_buffer_free(fbuf);
    ggml_backend_buffer_free(lbuf);
    ggml_free(ctx);
    ggml_free(fctx);
    ggml_free(lctx);
    printf("  %-40s %s\n", "host-memory buffers (no IO window)", g_fail == fail0 ? "ok" : "FAILED");
}

int main() {
    g_tsr = ggml_backend_tesseract_init();
    if (g_tsr == nullptr) {
        fprintf(stderr, "test-ggml-tesseract: no Tesseract backend (TSR_DEVICE=%s)\n", getenv("TSR_DEVICE") ? getenv("TSR_DEVICE") : "");
        return 1;
    }
    g_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (g_cpu == nullptr) {
        fprintf(stderr, "test-ggml-tesseract: no CPU backend\n");
        return 1;
    }
    printf("device: %s\n", ggml_backend_dev_description(ggml_backend_get_device(g_tsr)));

    const mm_case cases[] = {
        // name                               type             N      K    t1 t2 t3 weights split
        { "q8_0 ANY",                         GGML_TYPE_Q8_0,   64,   256,   1, 1, 1, false, 0 },
        { "q8_0 WEIGHTS",                     GGML_TYPE_Q8_0,   64,   256,   3, 1, 1, true,  0 },
        { "q4_K WEIGHTS N%16!=0 T>1 (scratch)", GGML_TYPE_Q4_K, 40,   512,   3, 1, 1, true,  0 },
        { "q4_K ANY split upload",            GGML_TYPE_Q4_K,   48,  1024,   2, 1, 1, false, 5 },
        { "q4_K WEIGHTS split upload",        GGML_TYPE_Q4_K,   37,   768,   1, 1, 1, true,  7 },
        { "f16 ANY N%16!=0 T=1 (tail pad)",   GGML_TYPE_F16,    33,    96,   1, 1, 1, false, 0 },
        { "f16 WEIGHTS converted",            GGML_TYPE_F16,    96,   160,   4, 1, 1, true,  0 },
        { "f32 ANY lazy, fixup",              GGML_TYPE_F32,    17,    64,   5, 1, 1, false, 0 },
        { "f32 WEIGHTS 1-row (raw -> lazy)",  GGML_TYPE_F32,     1,   512,  64, 1, 1, true,  0 },
        { "bf16 ANY",                         GGML_TYPE_BF16,   16,    32,   7, 1, 1, false, 0 },
        { "q6_K WEIGHTS converted, split",    GGML_TYPE_Q6_K,   48,   256,   2, 1, 1, true,  3 },
        { "q5_K ANY converted",               GGML_TYPE_Q5_K,   80,   512,   1, 1, 1, false, 0 },
        { "q4_0 ANY converted",               GGML_TYPE_Q4_0,   32,   128,   2, 1, 1, false, 0 },
        { "q8_0 batched src1 [K,3,2,2]",      GGML_TYPE_Q8_0,  130,  4096,   3, 2, 2, false, 0 },
        { "q4_K larger",                      GGML_TYPE_Q4_K, 1000,  2048,   2, 1, 1, true,  0 },
        { "q8_0 many tokens",                 GGML_TYPE_Q8_0,   64,   128,  37, 1, 1, true,  0 },
        { "q8_0 N=1",                         GGML_TYPE_Q8_0,    1,    32,   4, 1, 1, true,  0 },
        { "q8_0 N<C (3 rows on 4 channels)",  GGML_TYPE_Q8_0,    3,    64,   2, 1, 1, false, 0 },
    };
    for (const mm_case & c : cases) {
        run_case(c);
    }
    run_chain(256, 64, 3, 64, 48, GGML_TYPE_Q8_0);    // N1 % 16 == 0: both nodes in one program
    run_chain(256, 256, 1, 256, 40, GGML_TYPE_Q4_K);
    run_chain(128, 17, 32, 32, 24, GGML_TYPE_Q8_0);   // h via scratch + host fixup, then reshaped as src1
    run_src0_from_device(256, 64, 5, 64, 3);
    run_src0_from_device(128, 17, 32, 32, 3);         // h via scratch + fixup, then src0
    run_memset_clear();
    check_supports_op();
    check_output_head_placement();
    run_hosted_buffers();
    run_sched_large_T();

    ggml_backend_free(g_cpu);
    ggml_backend_free(g_tsr);
    if (g_fail) {
        printf("test-ggml-tesseract: %d check(s) FAILED in %d cases\n", g_fail, g_cases);
        return 1;
    }
    printf("test-ggml-tesseract: all %d cases OK\n", g_cases);
    return 0;
}
