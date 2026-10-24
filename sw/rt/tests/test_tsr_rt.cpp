// test_tsr_rt.cpp — tsr_rt on the transport TSR_DEVICE selects (sim, mock, vkernel, xrt).
//
// Every case is checked against a model built here independently of the runtime's upload and
// lowering code:
//   weights: decode the host bytes with ggml (memcpy / ggml_fp16_to_fp32_row / ggml_bf16_to_fp32_row),
//            quantize the WHOLE row with ggml's quantize_row_q8_0_ref (native Q8_0 / Q4_K bytes as
//            they are), then cut the blocks of each K part (weight_info.k_part) and repack them
//            with tsr_repack_row;
//   y:       tsr_qact per token and K part, tsr_dot_row per row and K part, K parts added in order
//            with tsr_fadd (MVM.ACC), i.e. the spec's numerics.
// Checks per case: read_blocks() == the ggml blocks byte for byte; linear() and linear_ref() ==
// the model bit for bit; y entries outside [0, N) of strided rows untouched; NMSE against an f64
// product of the dequantized weights and the exact x (sanity: catches an error shared by the model
// and the device, e.g. a transposition). Plus: program cache, counters, IO split (a device opened
// with a small IO area), free / re-upload, unsupported shapes, two threads, and the measured
// comparison of libtsr_ref's tsr_qact (A32) with quantize_row_q8_0_ref that decided the conversion
// path (the runtime uses ggml's function; tsr_qact differs on subnormal / tiny blocks).
//
// Options: --quick (3 small shapes, one device; vkernel smoke), --big (adds an N = 130000 head that
// is split in N at T_MAX 8).
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "ggml.h"
#include "ggml-quants.h"   // quantize_row_q8_0_ref, dequantize_row_* (ggml/src)

#include "tsr_isa.h"
#include "tsr_ref.h"
#include "tsr_rt.h"

using namespace tsr::rt;

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

static std::mt19937 g_rng(1234);

static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------------------
// Data
// ---------------------------------------------------------------------------

// f32 weight values with structure: per-row scales over 2^-4..2^4, all-zero rows and blocks.
static std::vector<float> make_w_f32(uint32_t N, uint32_t K) {
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<float> w((size_t) N * K);
    for (uint32_t n = 0; n < N; ++n) {
        const float s = std::ldexp(1.0f, (int) (n % 9) - 4);
        for (uint32_t k = 0; k < K; ++k) {
            float v = u(g_rng) * s;
            if (n % 11 == 3 && N > 4) v = 0.0f;                    // zero row
            if ((n + k / 32) % 13 == 0) v = 0.0f;                  // zero 32-blocks
            w[(size_t) n * K + k] = v;
        }
    }
    return w;
}

static std::vector<float> make_x(uint32_t T, uint32_t K, size_t xs) {
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<float> x((size_t) T * xs, NAN);                    // padding NaN: must not be read
    for (uint32_t t = 0; t < T; ++t) {
        const float s = std::ldexp(1.0f, (int) (t % 5) - 2);
        for (uint32_t k = 0; k < K; ++k) x[(size_t) t * xs + k] = u(g_rng) * s;
    }
    return x;
}

// Host bytes of a weight in type t, row r at r * rs (rs >= packed row bytes).
static std::vector<uint8_t> encode(dtype t, const std::vector<float> & w, uint32_t N, uint32_t K, size_t rs) {
    const size_t rb = dtype_row_bytes(t, K);
    std::vector<uint8_t> out((size_t) N * rs + 16, 0xEE);
    std::vector<uint8_t> row(rb);
    for (uint32_t n = 0; n < N; ++n) {
        const float * src = w.data() + (size_t) n * K;
        switch (t) {
            case dtype::f32:  memcpy(row.data(), src, rb); break;
            case dtype::f16:  ggml_fp32_to_fp16_row(src, (ggml_fp16_t *) row.data(), K); break;
            case dtype::bf16: ggml_fp32_to_bf16_row_ref(src, (ggml_bf16_t *) row.data(), K); break;
            case dtype::q8_0: quantize_row_q8_0_ref(src, (block_q8_0 *) row.data(), K); break;
            case dtype::q4_k: ggml_quantize_chunk(GGML_TYPE_Q4_K, src, row.data(), 0, 1, K, nullptr); break;
        }
        memcpy(out.data() + (size_t) n * rs, row.data(), rb);
    }
    return out;
}

// The model's device-format ggml blocks of every row (whole rows, ggml's conversion).
static std::vector<uint8_t> model_blocks(dtype t, const uint8_t * bytes, uint32_t N, uint32_t K, size_t rs) {
    const uint32_t fmt = dtype_device_format(t);
    const size_t brb = tsr_fmt_ggml_block_bytes(fmt) * (K / tsr_fmt_block_size(fmt));
    std::vector<uint8_t> out((size_t) N * brb);
    std::vector<float> f(K);
    std::vector<uint8_t> row(dtype_row_bytes(t, K));
    for (uint32_t n = 0; n < N; ++n) {
        memcpy(row.data(), bytes + (size_t) n * rs, row.size());   // aligned copy
        uint8_t * dst = out.data() + (size_t) n * brb;
        switch (t) {
            case dtype::f32:  memcpy(f.data(), row.data(), (size_t) K * 4); break;
            case dtype::f16:  ggml_fp16_to_fp32_row((const ggml_fp16_t *) row.data(), f.data(), K); break;
            case dtype::bf16: ggml_bf16_to_fp32_row((const ggml_bf16_t *) row.data(), f.data(), K); break;
            case dtype::q8_0:
            case dtype::q4_k: memcpy(dst, row.data(), brb); continue;
        }
        quantize_row_q8_0_ref(f.data(), (block_q8_0 *) dst, K);
    }
    return out;
}

struct model_y {
    std::vector<float>  y;     // [T, N]
    std::vector<double> y64;   // f64 sanity product of the dequantized blocks and exact x
};

static model_y model_linear(uint32_t fmt, const std::vector<uint8_t> & blocks, uint32_t N, uint32_t K,
                            uint32_t k_part, const float * x, uint32_t T, size_t xs) {
    const uint32_t kind = tsr_qa_kind_for_fmt(fmt);
    const uint32_t bs = tsr_fmt_block_size(fmt);
    const size_t bb = tsr_fmt_ggml_block_bytes(fmt);
    const size_t brb = bb * (K / bs);
    std::vector<uint32_t> k0s, ks;
    for (uint32_t k0 = 0; k0 < K; k0 += k_part) {
        k0s.push_back(k0);
        ks.push_back(std::min(k_part, K - k0));
    }
    const size_t kp = k0s.size();
    // QA per token and part
    std::vector<std::vector<uint8_t>> qa((size_t) T * kp);
    for (uint32_t t = 0; t < T; ++t) {
        for (size_t j = 0; j < kp; ++j) {
            qa[t * kp + j].resize((size_t) tsr_qa_lines(kind, ks[j]) * 64);
            tsr_qact(kind, x + (size_t) t * xs + k0s[j], ks[j], qa[t * kp + j].data());
        }
    }
    model_y m;
    m.y.assign((size_t) T * N, 0.0f);
    m.y64.assign((size_t) T * N, 0.0);
    auto work = [&](uint32_t lo, uint32_t hi) {
        std::vector<uint8_t> rep;
        std::vector<float> deq(K);
        for (uint32_t n = lo; n < hi; ++n) {
            const uint8_t * rb = blocks.data() + (size_t) n * brb;
            if (fmt == TSR_FMT_Q8_0) dequantize_row_q8_0((const block_q8_0 *) rb, deq.data(), K);
            else                     dequantize_row_q4_K((const block_q4_K *) rb, deq.data(), K);
            std::vector<std::vector<uint8_t>> parts(kp);
            for (size_t j = 0; j < kp; ++j) {
                parts[j].resize((size_t) tsr_row_beats(fmt, ks[j]) * 64);
                tsr_repack_row(fmt, rb + (size_t) (k0s[j] / bs) * bb, ks[j], parts[j].data());
            }
            for (uint32_t t = 0; t < T; ++t) {
                float acc = 0.0f;
                for (size_t j = 0; j < kp; ++j) {
                    const float v = tsr_dot_row(fmt, parts[j].data(), ks[j], qa[t * kp + j].data());
                    acc = j == 0 ? v : tsr_fadd(acc, v);
                }
                m.y[(size_t) t * N + n] = acc;
                double s = 0;
                for (uint32_t k = 0; k < K; ++k) s += (double) deq[k] * x[(size_t) t * xs + k];
                m.y64[(size_t) t * N + n] = s;
            }
        }
    };
    const unsigned nth = (uint64_t) N * K * T > (1u << 22) ? 4u : 1u;
    std::vector<std::thread> th;
    const uint32_t step = (N + nth - 1) / nth;
    for (unsigned i = 0; i < nth; ++i) {
        const uint32_t lo = std::min<uint32_t>(N, i * step), hi = std::min<uint32_t>(N, lo + step);
        if (lo < hi) th.emplace_back(work, lo, hi);
    }
    for (auto & t : th) t.join();
    return m;
}

static uint32_t f32_bits(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    return u;
}

static const uint32_t Y_GUARD = 0x7FBADBADu;   // a NaN payload linear() must leave alone

// ---------------------------------------------------------------------------
// One case
// ---------------------------------------------------------------------------

struct tcase {
    dtype    t;
    uint32_t N, K, T;
    uint32_t max_rows = 0, max_k = 0;      // weight_options
    uint32_t x_pad = 0, y_pad = 0;         // extra floats per x / y row
    uint32_t rs_pad = 0;                   // extra bytes per weight row
    int      expect_np = -1, expect_kp = -1;
};

static void run_case(device & dev, const tcase & c) {
    g_cases++;
    const double t0 = now_s();
    const size_t rs = dtype_row_bytes(c.t, c.K) + c.rs_pad;
    const std::vector<float> wf = make_w_f32(c.N, c.K);
    const std::vector<uint8_t> wb = encode(c.t, wf, c.N, c.K, rs);
    weight_options o;
    o.max_part_rows = c.max_rows;
    o.max_part_k = c.max_k;
    char name[160];
    snprintf(name, sizeof(name), "%-4s N=%-6u K=%-5u T=%-2u%s%s%s", dtype_name(c.t), c.N, c.K, c.T,
             c.max_rows ? " rows<=" : "", c.max_rows ? std::to_string(c.max_rows).c_str() : "",
             c.max_k ? (" k<=" + std::to_string(c.max_k)).c_str() : "");
    // (the string temporaries above live until the end of the full expression)

    weight_handle h = 0;
    try {
        h = dev.upload(c.t, wb.data(), c.N, c.K, rs, o);
    } catch (const std::exception & e) {
        CHECK(false, "%s: upload threw: %s", name, e.what());
        return;
    }
    const weight_info wi = dev.info(h);
    CHECK(wi.N == c.N && wi.K == c.K && wi.fmt == dtype_device_format(c.t), "%s: info", name);
    if (c.expect_np >= 0) CHECK((int) wi.n_parts == c.expect_np, "%s: n_parts %u, expected %d", name, wi.n_parts, c.expect_np);
    if (c.expect_kp >= 0) CHECK((int) wi.k_parts == c.expect_kp, "%s: k_parts %u, expected %d", name, wi.k_parts, c.expect_kp);
    if (c.max_rows) CHECK(wi.n_part <= c.max_rows, "%s: n_part %u", name, wi.n_part);
    if (c.max_k) CHECK(wi.k_part <= c.max_k, "%s: k_part %u", name, wi.k_part);

    // stored blocks == ggml's conversion of the whole rows
    const std::vector<uint8_t> mb = model_blocks(c.t, wb.data(), c.N, c.K, rs);
    std::vector<uint8_t> db(mb.size(), 0x5A);
    dev.read_blocks(h, 0, c.N, db.data());
    size_t bad_blk = 0;
    for (size_t i = 0; i < mb.size(); ++i) bad_blk += mb[i] != db[i];
    CHECK(bad_blk == 0, "%s: read_blocks differs from ggml's blocks in %zu bytes", name, bad_blk);

    const size_t xs = c.K + c.x_pad, ys = c.N + c.y_pad;
    const std::vector<float> x = make_x(c.T, c.K, xs);
    std::vector<float> y((size_t) c.T * ys), yr((size_t) c.T * ys);
    for (auto * v : { &y, &yr }) {
        for (auto & e : *v) memcpy(&e, &Y_GUARD, 4);
    }
    const counters k0 = dev.get_counters();
    try {
        dev.linear(h, x.data(), c.T, y.data(), xs, ys);
        dev.linear_ref(h, x.data(), c.T, yr.data(), xs, ys);
    } catch (const std::exception & e) {
        CHECK(false, "%s: linear threw: %s", name, e.what());
        dev.free_weight(h);
        return;
    }
    const counters k1 = dev.get_counters();
    const model_y m = model_linear(wi.fmt, mb, c.N, c.K, wi.k_part, x.data(), c.T, xs);

    size_t bad = 0, bad_ref = 0, guard = 0, first = SIZE_MAX;
    double num = 0, den = 0;
    for (uint32_t t = 0; t < c.T; ++t) {
        for (uint32_t n = 0; n < ys; ++n) {
            const float d = y[(size_t) t * ys + n], r = yr[(size_t) t * ys + n];
            if (n >= c.N) {
                guard += f32_bits(d) != Y_GUARD || f32_bits(r) != Y_GUARD;
                continue;
            }
            const float e = m.y[(size_t) t * c.N + n];
            if (f32_bits(d) != f32_bits(e)) {
                if (bad == 0) first = (size_t) t * c.N + n;
                bad++;
            }
            bad_ref += f32_bits(r) != f32_bits(e);
            const double ex = m.y64[(size_t) t * c.N + n];
            num += ((double) d - ex) * ((double) d - ex);
            den += ex * ex;
        }
    }
    const double nmse = den > 0 ? num / den : num;
    if (bad != 0) {
        const uint32_t ft = (uint32_t) (first / c.N), fn = (uint32_t) (first % c.N);
        CHECK(false, "%s: device != model in %zu of %zu elements (first t=%u n=%u: dev %.9g model %.9g)", name, bad,
              (size_t) c.T * c.N, ft, fn, y[(size_t) ft * ys + fn], m.y[first]);
    }
    CHECK(bad_ref == 0, "%s: linear_ref != model in %zu elements", name, bad_ref);
    CHECK(guard == 0, "%s: %zu y entries outside [0, N) were written", name, guard);
    CHECK(nmse < 1e-3, "%s: NMSE vs the f64 product %.3g", name, nmse);
    CHECK(k1.linears == k0.linears + 1 && k1.execs > k0.execs, "%s: counters", name);
    printf("  %-44s parts %ux%u  progs %" PRIu64 "  NMSE %.2e  %s  %.2fs\n", name, wi.n_parts, wi.k_parts,
           k1.execs - k0.execs, nmse, bad == 0 && bad_ref == 0 && bad_blk == 0 && guard == 0 ? "ok" : "FAIL", now_s() - t0);
    dev.free_weight(h);
}

// ---------------------------------------------------------------------------
// Other checks
// ---------------------------------------------------------------------------

// Program cache and counters: the second identical call reuses the program; host traffic is
// exactly x in and y out; the device read at least the weight.
static void test_cache_counters(device & dev) {
    g_cases++;
    const uint32_t N = 64, K = 896, T = 3;
    const std::vector<float> wf = make_w_f32(N, K);
    const weight_handle h = dev.upload(dtype::f32, wf.data(), N, K);
    const std::vector<float> x = make_x(T, K, K);
    std::vector<float> y1((size_t) T * N), y2((size_t) T * N);
    dev.linear(h, x.data(), T, y1.data());
    dev.reset_counters();
    dev.linear(h, x.data(), T, y2.data());
    const counters c = dev.get_counters();
    CHECK(memcmp(y1.data(), y2.data(), y1.size() * 4) == 0, "repeated linear differs");
    CHECK(c.prog_hits >= 1 && c.prog_misses == 0, "cache: hits %" PRIu64 " misses %" PRIu64, c.prog_hits, c.prog_misses);
    CHECK(c.h2d_bytes == (uint64_t) T * K * 4, "h2d %" PRIu64 " != x bytes %u", c.h2d_bytes, T * K * 4);
    CHECK(c.d2h_bytes == (uint64_t) T * N * 4, "d2h %" PRIu64 " != y bytes %u", c.d2h_bytes, T * N * 4);
    CHECK(c.h2d_weight_bytes == 0, "weight bytes moved during linear");
    const uint64_t wbytes = (uint64_t) N * tsr_row_beats(TSR_FMT_Q8_0, K) * 64;
    CHECK(c.perf_cycles > 0, "PERF_CYCLES delta 0");
    CHECK(c.dev_read_bytes >= wbytes, "device read %" PRIu64 " < weight bytes %" PRIu64, c.dev_read_bytes, wbytes);
    printf("  cache + counters: execs %" PRIu64 " hits %" PRIu64 " PERF_CYCLES %" PRIu64 " MVM %" PRIu64
           " dev read %" PRIu64 " B (weight %" PRIu64 " B) h2d %" PRIu64 " d2h %" PRIu64 "  %s\n",
           c.execs, c.prog_hits, c.perf_cycles, c.perf_mvm_cycles, c.dev_read_bytes, wbytes, c.h2d_bytes, c.d2h_bytes,
           g_fail == 0 ? "ok" : "(see failures)");
    dev.free_weight(h);
}

// Free / re-upload: memory comes back, a new weight of the same shape reuses it and is right.
static void test_free_reupload(device & dev) {
    g_cases++;
    const memory_info m0 = dev.memory();
    const uint32_t N = 200, K = 512;
    std::vector<float> a = make_w_f32(N, K), b = make_w_f32(N, K);
    const weight_handle ha = dev.upload(dtype::f32, a.data(), N, K);
    const memory_info m1 = dev.memory();
    CHECK(m1.weight_free < m0.weight_free && m1.n_weights == m0.n_weights + 1, "upload did not take memory");
    dev.free_weight(ha);
    const memory_info m2 = dev.memory();
    CHECK(m2.weight_free == m0.weight_free && m2.n_weights == m0.n_weights, "free did not return memory");
    bool threw = false;
    try {
        dev.free_weight(ha);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    CHECK(threw, "double free not rejected");
    const weight_handle hb = dev.upload(dtype::f32, b.data(), N, K);
    CHECK(hb != ha, "handle reused");
    const std::vector<float> x = make_x(2, K, K);
    std::vector<float> y(2 * N), yr(2 * N);
    dev.linear(hb, x.data(), 2, y.data());
    dev.linear_ref(hb, x.data(), 2, yr.data());
    CHECK(memcmp(y.data(), yr.data(), y.size() * 4) == 0, "re-uploaded weight: device != ref");
    dev.free_weight(hb);
    printf("  free / re-upload  %s\n", g_fail == 0 ? "ok" : "(see failures)");
}

static void test_unsupported(device & dev) {
    g_cases++;
    CHECK(dev.unsupported_reason(dtype::f32, 4, 48) != nullptr, "K=48 accepted");
    CHECK(dev.unsupported_reason(dtype::q4_k, 4, 288) != nullptr, "Q4_K K=288 accepted");
    CHECK(dev.unsupported_reason(dtype::f32, 0, 64) != nullptr, "N=0 accepted");
    CHECK(dev.unsupported_reason(dtype::f32, 4, 64, weight_options{ 0, 48 }) != nullptr, "max_part_k=48 accepted");
    CHECK(dev.unsupported_reason(dtype::f32, 4, 64) == nullptr, "K=64 rejected");
    CHECK(dev.unsupported_reason(dtype::f32, 4, 33024) == nullptr, "K=33024 (K split) rejected");
    float w[48] = {};
    bool threw = false;
    try {
        dev.upload(dtype::f32, w, 1, 48);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    CHECK(threw, "upload of K=48 did not throw invalid_argument");
    threw = false;
    try {
        dev.linear(12345678, w, 1, w);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    CHECK(threw, "linear on an unknown handle did not throw");
    printf("  unsupported shapes / handles  %s\n", g_fail == 0 ? "ok" : "(see failures)");
}

// Two threads on two weights, interleaved calls; every result bit-exact.
static void test_threads(device & dev) {
    g_cases++;
    const uint32_t K = 512;
    const uint32_t Ns[2] = { 96, 40 };
    weight_handle hs[2];
    std::vector<float> ws[2];
    for (int i = 0; i < 2; ++i) {
        ws[i] = make_w_f32(Ns[i], K);
        hs[i] = dev.upload(dtype::f32, ws[i].data(), Ns[i], K);
    }
    std::atomic<int> bad{ 0 };
    auto body = [&](int i) {
        std::mt19937 rng(77 + i);
        std::uniform_real_distribution<float> u(-1.0f, 1.0f);
        for (int it = 0; it < 12; ++it) {
            const uint32_t T = 1 + (uint32_t) (it % 4);
            std::vector<float> x((size_t) T * K);
            for (auto & v : x) v = u(rng);
            std::vector<float> y((size_t) T * Ns[i]), yr((size_t) T * Ns[i]);
            dev.linear(hs[i], x.data(), T, y.data());
            dev.linear_ref(hs[i], x.data(), T, yr.data());
            if (memcmp(y.data(), yr.data(), y.size() * 4) != 0) bad++;
        }
    };
    std::thread a(body, 0), b(body, 1);
    a.join();
    b.join();
    CHECK(bad == 0, "%d threaded calls differ from the reference", bad.load());
    for (int i = 0; i < 2; ++i) dev.free_weight(hs[i]);
    printf("  two threads x 12 calls  %s\n", bad == 0 ? "ok" : "FAIL");
}

// A device whose IO area holds 17 tokens of this shape: one call runs as several programs.
static void test_io_split(const char * kind) {
    g_cases++;
    open_options oo;
    oo.kind = kind;
    oo.io_bytes = 64 * 1024;
    device dev(oo);
    const uint32_t N = 64, K = 896, T = 40;   // per token 3584 + 256 bytes -> 17 tokens per program
    const std::vector<float> wf = make_w_f32(N, K);
    const weight_handle h = dev.upload(dtype::f32, wf.data(), N, K);
    const std::vector<float> x = make_x(T, K, K);
    std::vector<float> y((size_t) T * N), yr((size_t) T * N);
    dev.linear(h, x.data(), T, y.data());
    const counters c = dev.get_counters();
    dev.linear_ref(h, x.data(), T, yr.data());
    CHECK(c.execs == 3, "IO split: %" PRIu64 " programs, expected 3", c.execs);
    CHECK(memcmp(y.data(), yr.data(), y.size() * 4) == 0, "IO split: device != ref");
    printf("  IO area 64 KiB, T=%u: %" PRIu64 " programs  %s\n", T, c.execs, g_fail == 0 ? "ok" : "(see failures)");
}

// libtsr_ref's QACT A32 as a weight quantizer? Measured against quantize_row_q8_0_ref: equal on
// ordinary blocks, different on blocks with subnormals or a tiny amax (DAZ / FTZ, spec 구현 명확화).
// The runtime therefore converts weights with ggml's function, like the ggml backend.
static void test_qact_vs_ggml() {
    g_cases++;
    auto differs = [](const float * x) {
        block_q8_0 g;
        quantize_row_q8_0_ref(x, &g, 32);
        alignas(16) uint8_t qa[256];
        tsr_qact(TSR_QA_A32, x, 32, qa);
        const int8_t * qs = tsr_qa_qs(TSR_QA_A32, 32, qa);
        const float d = tsr_qa_d(TSR_QA_A32, 32, qa)[0];
        if (tsr_f32_to_fp16(d) != g.d) return true;
        return memcmp(qs, g.qs, 32) != 0;
    };
    std::normal_distribution<float> nd(0.0f, 1.0f);
    size_t n_rand = 200000, bad_rand = 0;
    float x[32];
    for (size_t i = 0; i < n_rand; ++i) {
        const float s = std::ldexp(1.0f, (int) (i % 61) - 30);
        for (auto & v : x) v = nd(g_rng) * s;
        bad_rand += differs(x);
    }
    // edge blocks: a subnormal element next to a normal amax near the FTZ range, all subnormal,
    // amax so small that d = amax / 127 is subnormal
    size_t n_edge = 0, bad_edge = 0;
    for (int e = -149; e <= -110; ++e) {
        for (int v = 0; v < 4; ++v) {
            for (auto & q : x) q = std::ldexp(nd(g_rng), e);
            if (v == 1) x[3] = 1e-40f;
            if (v == 2) x[7] = std::ldexp(1.0f, -126);
            n_edge++;
            bad_edge += differs(x);
        }
    }
    CHECK(bad_rand == 0, "tsr_qact A32 differs from quantize_row_q8_0_ref on %zu ordinary blocks", bad_rand);
    printf("  tsr_qact(A32) vs quantize_row_q8_0_ref: %zu / %zu ordinary blocks differ, %zu / %zu edge blocks "
           "(subnormal / tiny amax) differ -> weights use ggml's function\n", bad_rand, n_rand, bad_edge, n_edge);
}

int main(int argc, char ** argv) {
    bool quick = false, big = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--quick")) quick = true;
        else if (!strcmp(argv[i], "--big")) big = true;
        else {
            fprintf(stderr, "usage: %s [--quick] [--big]\n", argv[0]);
            return 2;
        }
    }
    const double t_start = now_s();
    std::unique_ptr<device> devp;
    try {
        devp.reset(new device());
    } catch (const std::exception & e) {
        fprintf(stderr, "test-tsr-rt: %s\n", e.what());
        return 1;
    }
    device & dev = *devp;
    const tsr_config & cfg = dev.config();
    printf("test-tsr-rt on %s\n", dev.description());

    const uint32_t tmax = cfg.t_max;
    std::vector<tcase> cases;
    if (quick) {
        cases = {
            { dtype::f32,  64, 896, 2 },
            { dtype::q4_k, 33, 512, 3 },
            { dtype::bf16, 100, 896, 2, 16, 0, 0, 0, 0, 7, 1 },
        };
    } else {
        cases = {
            // Q8_0 from f32 / f16 / bf16 and native Q8_0 / Q4_K over K in {32 .. 4864}, N, T 1..20
            { dtype::f32,  1, 32, 1 },
            { dtype::f32,  3, 64, 2 },
            { dtype::f32,  16, 96, 3 },
            { dtype::f32,  17, 256, 5 },
            { dtype::f32,  64, 896, 1 },
            { dtype::f32,  100, 896, 9 },
            { dtype::f32,  896, 896, 20 },
            { dtype::f32,  2, 4864, 20 },
            { dtype::f32,  136, 4864, 3 },
            { dtype::f32,  4864, 896, 2 },
            { dtype::f16,  37, 896, 7 },
            { dtype::f16,  128, 4864, 1 },
            { dtype::f16,  5, 32, 20 },
            { dtype::bf16, 50, 512, 8 },
            { dtype::bf16, 896, 896, 16 },
            { dtype::bf16, 3, 4864, 11 },
            { dtype::q8_0, 130, 1024, 3 },
            { dtype::q8_0, 896, 4864, 1 },
            { dtype::q4_k, 1, 256, 1 },
            { dtype::q4_k, 33, 512, 9 },
            { dtype::q4_k, 128, 4864, 2 },
            { dtype::q4_k, 5, 4864, 20 },
            { dtype::q4_k, 300, 2048, 8 },
            // N parts (forced): bit-identical to unsplit rows
            { dtype::f32,  100, 896, 9, 16, 0, 0, 0, 0, 7, 1 },
            { dtype::q8_0, 77, 4864, 3, 40, 0, 0, 0, 0, 2, 1 },
            { dtype::q4_k, 90, 512, 10, 31, 0, 0, 0, 0, 3, 1 },
            { dtype::bf16, 1000, 256, 17, 999, 0, 0, 0, 0, 2, 1 },
            // K parts (forced) with MVM.ACC, and both
            { dtype::f32,  40, 896, 5, 0, 256, 0, 0, 0, 1, 4 },
            { dtype::q4_k, 20, 2048, 4, 0, 512, 0, 0, 0, 1, 4 },
            { dtype::f16,  50, 896, 9, 16, 288, 0, 0, 0, 4, 4 },
            { dtype::q8_0, 7, 4864, 20, 0, 1024, 0, 0, 0, 1, 5 },
            // K > K_MAX (32768): natural K split
            { dtype::f32,  24, 33024, 3, 0, 0, 0, 0, 0, 1, 2 },
            { dtype::q4_k, 9, 33280, 2, 0, 0, 0, 0, 0, 1, 2 },
            // strided x / y, padded and misaligned weight rows
            { dtype::f32,  33, 512, 6, 0, 0, 48, 5, 6 },
            { dtype::f16,  20, 896, 4, 0, 0, 16, 1, 1 },
            { dtype::q8_0, 10, 256, 3, 0, 0, 0, 3, 1 },
            { dtype::q4_k, 6, 512, 2, 0, 0, 32, 0, 3 },
        };
        if (big) {
            // an output head with N = 130000, K = 896: at T_MAX 8 x + QA + y of 8 tokens need more
            // than the SPM (65536 lines), so the weight is split in two N parts; at T_MAX 1 not.
            cases.push_back({ dtype::q8_0, 130000, 896, 9, 0, 0, 0, 0, 0, tmax >= 8 ? 2 : 1, 1 });
        }
    }
    printf("cases (T_MAX %u):\n", tmax);
    for (const tcase & c : cases) run_case(dev, c);
    if (!quick) {
        test_cache_counters(dev);
        test_free_reupload(dev);
        test_unsupported(dev);
        test_threads(dev);
        test_io_split(nullptr);
        test_qact_vs_ggml();
    } else {
        test_cache_counters(dev);
    }
    const counters c = dev.get_counters();
    printf("totals: %" PRIu64 " linear calls, %" PRIu64 " programs (%" PRIu64 " cached hits), PERF_CYCLES %" PRIu64
           ", device read %.1f MiB, h2d %.1f MiB (weights %.1f), d2h %.1f MiB, %.1f s\n",
           c.linears, c.execs, c.prog_hits, c.perf_cycles, c.dev_read_bytes / 1048576.0, c.h2d_bytes / 1048576.0,
           c.h2d_weight_bytes / 1048576.0, c.d2h_bytes / 1048576.0, now_s() - t_start);
    if (g_fail) {
        printf("test-tsr-rt: %d check(s) FAILED in %d cases\n", g_fail, g_cases);
        return 1;
    }
    printf("test-tsr-rt: all %d cases OK\n", g_cases);
    return 0;
}
