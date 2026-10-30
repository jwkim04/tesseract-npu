// test-tsr-opl-launch.cpp — the op-list path's launcher on a device (ctest ggml_tesseract_opl_launch;
// docs/v2/06-software-verification.md §9.3 steps 3-5, §7.3): tsr_opl_dev boots the transport selected
// by TSR_DEVICE (default sim) with the build's firmware image and runs lists built by tsr_opl_builder
// through the `graph` kernel. No ggml.
//
//   A. A list whose dependencies need a JOIN and an ACC_DST record: three MUL_MATs from x, a fourth
//      reading all three outputs (three predecessors: one JOIN), a fifth adding a K part into the
//      fourth's y (ACC_DST, ordered after it by RAW / WAW on y). Every y row equals the spec model
//      (tsr_qact + tsr_dot_row, tsr_fadd for ACC) bit for bit; DONE STATUS OK, RET 0.
//   B. The same list again: an OPL cache hit (nothing written), the same results.
//   C. A two-record list whose second record's x runs past the end of channel 0: DONE STATUS
//      TILE_ERR, RET = 2 (op 1), the first error DRAM_RANGE; tsr_opl_describe names op 1.
//   C2. The same fault in a record behind a JOIN (index 5, tag 104): RET 6, and
//      tsr_opl_failed_op / _tag (what the backend and sw/rt name the node with) give 5 / 104.
//   D. A MUL_MAT with K = 48 (not a multiple of 32): the executor refuses it before pushing,
//      KERNEL_ERR with RET 0x8000_0004 (BAD_RECORD), named by tsr_opl_describe.
//   E. A valid launch after the errors gives the results of A again (the device recovered).
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "tsr-opl.h"
#include "tsr-wlayout.h"
#include "tsr_isa.h"
#include "tsr_ref.h"
#include "tsr_transport.h"

namespace opl = tsr::opl;
namespace mm  = tsr::opl::mm;

static int g_fail = 0;
#define CHECK(cond, ...)                                                              \
    do {                                                                              \
        if (!(cond)) {                                                                \
            fprintf(stderr, "  FAIL %s:%d: %s: ", __FILE__, __LINE__, #cond);         \
            fprintf(stderr, __VA_ARGS__);                                             \
            fprintf(stderr, "\n");                                                    \
            g_fail++;                                                                 \
        }                                                                             \
    } while (0)

static std::mt19937 g_rng(1234);

static float rnd() { return std::uniform_real_distribution<float>(-1.0f, 1.0f)(g_rng); }

// A random Q8_0 weight [N x K]: its ggml blocks and its device copy at W_OFF w_off (bytes).
struct weight {
    uint32_t N, K;
    uint64_t w_off;
    std::vector<tsr_block_q8_0> blk;   // N * K / 32
};

static weight make_weight(tsr_transport * tp, uint32_t num_ch, uint32_t N, uint32_t K, uint64_t w_off) {
    weight w{ N, K, w_off, std::vector<tsr_block_q8_0>((size_t) N * K / 32) };
    for (auto & b : w.blk) {
        b.d = (uint16_t) (0x2000u + (g_rng() & 0x3FFu));   // positive f16 scales around 2^-7
        for (int i = 0; i < 32; ++i) b.qs[i] = (int8_t) (g_rng() & 0xFF);
    }
    const size_t rb = (size_t) K / 32 * sizeof(tsr_block_q8_0);
    tsr_weight_store_rows(num_ch, TSR_FMT_Q8_0, N, K, w_off, 0, N, 1u << 20,
        [&](uint32_t r, void *) -> const void * { return (const uint8_t *) w.blk.data() + (size_t) r * rb; },
        [&](uint32_t ch, uint64_t off, const void * p, size_t n) {
            CHECK(tsr_transport_write(tp, ch, off, p, n) == 0, "weight write");
        });
    return w;
}

// y[t][n] of the spec model for x rows (K floats, x_stride floats apart).
static std::vector<float> ref_mm(const weight & w, const float * x, uint32_t T, size_t x_stride) {
    std::vector<float> y((size_t) T * w.N);
    std::vector<uint8_t> qa((size_t) tsr_qa_lines(TSR_QA_A32, w.K) * 64);
    std::vector<uint8_t> row((size_t) tsr_row_beats(TSR_FMT_Q8_0, w.K) * 64);
    for (uint32_t t = 0; t < T; ++t) {
        tsr_qact(TSR_QA_A32, x + t * x_stride, w.K, qa.data());
        for (uint32_t n = 0; n < w.N; ++n) {
            tsr_repack_row(TSR_FMT_Q8_0, w.blk.data() + (size_t) n * w.K / 32, w.K, row.data());
            y[(size_t) t * w.N + n] = tsr_dot_row(TSR_FMT_Q8_0, row.data(), w.K, qa.data());
        }
    }
    return y;
}

static void mm_args(uint32_t * a, uint64_t x, uint32_t xs, uint64_t y, uint32_t ys, const weight & w, uint32_t K, uint32_t T,
                    uint32_t flags = 0) {
    a[mm::X] = tsr_dram_handle(0, x);
    a[mm::X_STRIDE] = xs;
    a[mm::Y] = tsr_dram_handle(0, y);
    a[mm::Y_STRIDE] = ys;
    a[mm::W_OFF] = (uint32_t) (w.w_off / 64);
    a[mm::FMT] = TSR_FMT_Q8_0;
    a[mm::N] = w.N;
    a[mm::K] = K;
    a[mm::T] = T;
    a[mm::ADD] = opl::HANDLE_NONE;
    a[mm::ADD_STRIDE] = 0;
    a[mm::FLAGS] = flags;
}

static std::vector<float> read_f32(tsr_transport * tp, uint64_t off, size_t n) {
    std::vector<float> v(n);
    CHECK(tsr_transport_read(tp, 0, off, v.data(), n * 4) == 0, "read");
    return v;
}

static size_t bit_diff(const std::vector<float> & a, const std::vector<float> & b) {
    size_t bad = 0;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) bad += memcmp(&a[i], &b[i], 4) != 0;
    return bad + (a.size() != b.size() ? 1 : 0);
}

int main() {
    tsr_transport * tp = tsr_transport_open(nullptr);
    if (tp == nullptr) {
        fprintf(stderr, "test-tsr-opl-launch: cannot open the transport (TSR_DEVICE=%s)\n", getenv("TSR_DEVICE") ? getenv("TSR_DEVICE") : "");
        return 1;
    }
    const tsr_config cfg = *tsr_transport_config(tp);
    tsr_opl_dev dev;
    std::string err;
    if (!dev.open(tp, &err)) {
        fprintf(stderr, "test-tsr-opl-launch: %s\n", err.c_str());
        tsr_transport_close(tp);
        return 1;
    }
    printf("device %s, image %s (%016" PRIx64 "), graph = kernel %d\n", tsr_transport_name(tp), dev.image_path(), dev.image_id(),
           dev.graph_kernel());
    const tsr_dram_write_fn write = [tp](uint32_t ch, uint64_t off, const void * p, size_t n) {
        CHECK(tsr_transport_write(tp, ch, off, p, n) == 0, "write");
    };

    // ---- A: three projections, a JOIN-ed consumer, an ACC_DST K part ----
    const uint32_t T = 3, K = 64, N = 64;
    const uint64_t IO = TSR_REGION_IO_OFF;
    const uint64_t X = IO, A = IO + 0x10000, Y4 = IO + 0x20000;   // A: [a | b | c] per token, 3 x 64 floats
    uint64_t woff = TSR_REGION_WEIGHT_OFF;
    auto next_w = [&](uint32_t n, uint32_t k) {
        const weight w = make_weight(tp, cfg.num_ch, n, k, woff);
        woff += (tsr_tensor_ch_beats(TSR_FMT_Q8_0, n, k, cfg.num_ch) * 64 + 4095) / 4096 * 4096;
        return w;
    };
    const weight w0 = next_w(N, K), w1 = next_w(N, K), w2 = next_w(N, K);
    const weight w3 = next_w(32, 3 * N), w4 = next_w(32, K);
    std::vector<float> x((size_t) T * K);
    for (float & v : x) v = rnd();
    CHECK(tsr_transport_write(tp, 0, X, x.data(), x.size() * 4) == 0, "x");
    tsr_opl_builder b(cfg.num_ch);
    uint32_t a[mm::NARGS];
    const uint32_t a_stride = 3 * N / 16;                         // lines between tokens of [a | b | c]
    uint32_t idx[5];
    for (uint32_t i = 0; i < 3; ++i) {
        const weight & w = i == 0 ? w0 : i == 1 ? w1 : w2;
        mm_args(a, X, K / 16, A + (uint64_t) i * N * 4, a_stride, w, K, T);
        idx[i] = b.mul_mat(a, (int32_t) i);
    }
    mm_args(a, A, a_stride, Y4, 2, w3, 3 * N, T);                 // reads [a | b | c]: 3 predecessors
    idx[3] = b.mul_mat(a, 3);
    mm_args(a, X, K / 16, Y4, 2, w4, K, T, mm::FLAG_ACC_DST);     // y4 += W4 x
    idx[4] = b.mul_mat(a, 4);
    CHECK(b.n_joins() == 1 && idx[3] == 4 && idx[4] == 5 && b.op(5).dep0 == 4, "list shape: %u joins, idx %u %u",
          b.n_joins(), idx[3], idx[4]);
    const std::vector<uint8_t> blob = b.finish();
    tsr_opl_run_result r = dev.run(blob, 0, write);
    CHECK(r.rc == 0 && r.res.type == TSR_LCQ_R0_TYPE_DONE && r.res.status == TSR_LCQ_R0_STATUS_OK && r.res.ret == 0 && !r.cache_hit,
          "A: rc %d type %u status %u ret %d: %s", r.rc, r.res.type, r.res.status, r.res.ret, tsr_opl_describe(r, nullptr).c_str());
    std::vector<float> want_abc((size_t) T * 3 * N);
    for (uint32_t i = 0; i < 3; ++i) {
        const std::vector<float> yi = ref_mm(i == 0 ? w0 : i == 1 ? w1 : w2, x.data(), T, K);
        for (uint32_t t = 0; t < T; ++t) memcpy(&want_abc[(size_t) t * 3 * N + i * N], &yi[(size_t) t * N], N * 4);
    }
    const std::vector<float> got_abc = read_f32(tp, A, want_abc.size());
    CHECK(bit_diff(got_abc, want_abc) == 0, "A: [a | b | c] differ from the spec model in %zu values", bit_diff(got_abc, want_abc));
    std::vector<float> want4 = ref_mm(w3, want_abc.data(), T, 3 * N);
    const std::vector<float> acc = ref_mm(w4, x.data(), T, K);
    for (size_t i = 0; i < want4.size(); ++i) want4[i] = tsr_fadd(want4[i], acc[i]);
    const std::vector<float> got4 = read_f32(tp, Y4, want4.size());
    CHECK(bit_diff(got4, want4) == 0, "A: y4 (JOIN + ACC_DST) differs from the spec model in %zu values", bit_diff(got4, want4));
    printf("  A: 5 MUL_MAT + 1 JOIN records, %zu B, %.3f ms  %s\n", blob.size(), r.wall_ns * 1e-6, g_fail == 0 ? "ok" : "FAILED");

    // ---- B: the same list again: a cache hit ----
    const uint64_t w0_bytes = dev.cache().get_stats().bytes_written;
    std::vector<float> zero(want4.size(), 0.0f);
    CHECK(tsr_transport_write(tp, 0, Y4, zero.data(), zero.size() * 4) == 0, "clear y4");
    r = dev.run(blob, 0, write);
    CHECK(r.rc == 0 && r.res.status == TSR_LCQ_R0_STATUS_OK && r.cache_hit && dev.cache().get_stats().bytes_written == w0_bytes,
          "B: rc %d status %u hit %d", r.rc, r.res.status, (int) r.cache_hit);
    CHECK(bit_diff(read_f32(tp, Y4, want4.size()), want4) == 0, "B: y4 differs on the cached list");
    printf("  B: cached list (hits %" PRIu64 ", misses %" PRIu64 ")  %s\n", dev.cache().get_stats().hits,
           dev.cache().get_stats().misses, g_fail == 0 ? "ok" : "FAILED");

    // ---- C: a DRAM_RANGE error in op 1 -> TILE_ERR, RET 2 ----
    {
        tsr_opl_builder c(cfg.num_ch);
        mm_args(a, X, K / 16, IO + 0x30000, 4, w0, K, 1);
        c.mul_mat(a, 0);
        mm_args(a, cfg.ch_bytes - 64, K / 16, IO + 0x40000, 4, w0, K, 1);   // x: 4 lines from the last beat
        c.mul_mat(a, 1);
        const tsr_opl_run_result rc = dev.run(c.finish(), 0, write);
        const std::string msg = tsr_opl_describe(rc, "op-one");
        CHECK(rc.rc == 0 && rc.res.type == TSR_LCQ_R0_TYPE_DONE && rc.res.status == TSR_LCQ_R0_STATUS_TILE_ERR && rc.res.ret == 2 &&
                  rc.res.err_code == TSR_ERR_DRAM_RANGE && c.tag_of((uint32_t) rc.res.ret - 1) == 1,
              "C: rc %d status %u ret %d err %u: %s", rc.rc, rc.res.status, rc.res.ret, rc.res.err_code, msg.c_str());
        CHECK(msg.find("TILE_ERR at op 1 (op-one)") != std::string::npos && msg.find("DRAM_RANGE") != std::string::npos,
              "C: description '%s'", msg.c_str());
        printf("  C: %s  %s\n", msg.c_str(), g_fail == 0 ? "ok" : "FAILED");
    }

    // ---- C2: the failing record behind a JOIN (index 5, tag 104): RET 6 names tag 104 ----
    {
        tsr_opl_builder c(cfg.num_ch);
        const uint64_t A2 = IO + 0x50000;
        for (uint32_t i = 0; i < 3; ++i) {
            mm_args(a, X, K / 16, A2 + (uint64_t) i * N * 4, a_stride, i == 0 ? w0 : i == 1 ? w1 : w2, K, 1);
            c.mul_mat(a, (int32_t) (100 + i));
        }
        mm_args(a, A2, a_stride, IO + 0x60000, 2, w3, 3 * N, 1);              // 3 predecessors: JOIN at 3, op 4
        const uint32_t i4 = c.mul_mat(a, 103);
        mm_args(a, cfg.ch_bytes - 64, K / 16, IO + 0x70000, 4, w0, K, 1);   // op 5: x past the end of ch0
        const uint32_t i5 = c.mul_mat(a, 104);
        CHECK(c.n_joins() == 1 && i4 == 4 && i5 == 5, "C2: list shape: %u joins, idx %u %u", c.n_joins(), i4, i5);
        const tsr_opl_run_result rc = dev.run(c.finish(), 0, write);
        const std::string msg = tsr_opl_describe(rc, "op-five");
        CHECK(rc.rc == 0 && rc.res.status == TSR_LCQ_R0_STATUS_TILE_ERR && rc.res.ret == 6 && rc.res.err_code == TSR_ERR_DRAM_RANGE,
              "C2: rc %d status %u ret %d err %u: %s", rc.rc, rc.res.status, rc.res.ret, rc.res.err_code, msg.c_str());
        CHECK(tsr_opl_failed_op(rc, c.n_ops()) == 5 && tsr_opl_failed_tag(c, rc) == 104, "C2: failing op %d, tag %d (want 5, 104)",
              tsr_opl_failed_op(rc, c.n_ops()), tsr_opl_failed_tag(c, rc));
        CHECK(msg.find("TILE_ERR at op 5 (op-five)") != std::string::npos, "C2: description '%s'", msg.c_str());
        printf("  C2: failing op 5 behind a JOIN -> tag %d  %s\n", tsr_opl_failed_tag(c, rc), g_fail == 0 ? "ok" : "FAILED");
    }

    // ---- D: K = 48: BAD_RECORD -> KERNEL_ERR ----
    {
        tsr_opl_builder c(cfg.num_ch);
        mm_args(a, X, 3, IO + 0x30000, 4, w0, 48, 1);
        c.mul_mat(a, 0);
        const tsr_opl_run_result rc = dev.run(c.finish(), 0, write);
        const std::string msg = tsr_opl_describe(rc, nullptr);
        CHECK(rc.rc == 0 && rc.res.status == TSR_LCQ_R0_STATUS_KERNEL_ERR && (uint32_t) rc.res.ret == opl::RET_BAD_RECORD,
              "D: rc %d status %u ret 0x%08x: %s", rc.rc, rc.res.status, (uint32_t) rc.res.ret, msg.c_str());
        CHECK(msg.find("BAD_RECORD") != std::string::npos, "D: description '%s'", msg.c_str());
        printf("  D: %s  %s\n", msg.c_str(), g_fail == 0 ? "ok" : "FAILED");
    }

    // ---- E: the device still runs lists correctly ----
    CHECK(tsr_transport_write(tp, 0, Y4, zero.data(), zero.size() * 4) == 0, "clear y4");
    r = dev.run(blob, 0, write);
    CHECK(r.rc == 0 && r.res.status == TSR_LCQ_R0_STATUS_OK && r.res.ret == 0, "E: rc %d status %u", r.rc, r.res.status);
    CHECK(bit_diff(read_f32(tp, Y4, want4.size()), want4) == 0, "E: y4 differs after the failed launches");
    printf("  E: valid list after the errors  %s\n", g_fail == 0 ? "ok" : "FAILED");
    printf("  launches %" PRIu64 ", op records %" PRIu64 "\n", dev.launches, dev.ops);

    tsr_transport_close(tp);
    if (g_fail) {
        printf("test-tsr-opl-launch: %d check(s) FAILED\n", g_fail);
        return 1;
    }
    printf("test-tsr-opl-launch: OK\n");
    return 0;
}
