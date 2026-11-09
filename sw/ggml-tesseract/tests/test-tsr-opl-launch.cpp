// test-tsr-opl-launch.cpp — the op-list path's launcher on a device (ctest ggml_tesseract_opl_launch;
// docs/design/06-software-verification.md §9.3 steps 3-5, §7.3): tsr_opl_dev boots the transport selected
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
//   M7 (SW §9.3; lists as the backend's builder makes them: kind_ranges, preload entries):
//   F. Fusion is bit-identical to the unfused device sequence: MUL_MAT with a bias row / the residual
//      as its addend vs MUL_MAT then ADD (and ADD = one f32 addition of the spec model); RMS_NORM with
//      its weight vs RMS_NORM then MUL, for a decode row, 3 rows, per-head rows and per-row weights.
//   G. A chain of GET_ROWS (ids from the preload table), CPY, SCALE, MUL by per-row scalars, ADD of a
//      broadcast row, UNARY, GLU, ROPE (positions from the preload table) and a fused RMS_NORM: the
//      exact kinds equal the spec model, every output equals a SERIAL run of the same records; then
//      new positions and ids through the same list (an OPL cache hit) give the new results.
//   H. A GET_ROWS with a negative id answers TILE_ERR DRAM_RANGE at its record, named by its tag.
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

// =============================================================================================
// M7: lists of the M7 kinds as the backend's builder makes them (tsr_opl_builder: kind_ranges
// for the dependencies, preload entries), on the device
// =============================================================================================
static uint32_t H0(uint64_t off) { return tsr_dram_handle(0, off); }

static uint32_t add_kind(tsr_opl_builder & b, uint32_t kind, const uint32_t * a, int32_t tag, uint64_t aux = 0) {
    std::vector<tsr_dram_range> rd, wr;
    b.kind_ranges(kind, a, aux, rd, wr);
    const uint32_t i = b.add(kind, a, opl::kind_nargs(kind), rd, wr, tag);
    CHECK(i != tsr_opl_builder::NO_ROOM && i != tsr_opl_builder::FAILED, "add %u: %s", kind, b.error());
    return i;
}
template <class A> static uint32_t add_m7(tsr_opl_builder & b, uint32_t kind, const A & args, int32_t tag, uint64_t aux = 0) {
    uint32_t w[opl::fa::NARGS] = {};
    opl::put_args(args, w);
    return add_kind(b, kind, w, tag, aux);
}

static bool run_list(tsr_opl_dev & dev, const std::vector<uint8_t> & blob, const tsr_dram_write_fn & write, const char * what) {
    const tsr_opl_run_result r = dev.run(blob, 0, write);
    const bool ok = r.rc == 0 && r.res.type == TSR_LCQ_R0_TYPE_DONE && r.res.status == TSR_LCQ_R0_STATUS_OK && r.res.ret == 0;
    CHECK(ok, "%s: %s", what, tsr_opl_describe(r, nullptr).c_str());
    return ok;
}

static void write_f32(tsr_transport * tp, uint64_t off, const std::vector<float> & v) {
    CHECK(tsr_transport_write(tp, 0, off, v.data(), v.size() * 4) == 0, "write");
}

static std::vector<float> rnd_vec(size_t n, float scale = 1.0f) {
    std::vector<float> v(n);
    for (float & x : v) x = rnd() * scale;
    return v;
}

static uint32_t fb(float f) { return opl::writer::f32_bits(f); }

template <class F>
static void test_m7(tsr_transport * tp, const tsr_config & cfg, tsr_opl_dev & dev, const tsr_dram_write_fn & write, F next_w) {
    const uint64_t IO = TSR_REGION_IO_OFF + 0x100000;
    const int fail0 = g_fail;

    // ---- F: fusion is bit-identical to the unfused device pair (SW §7.4, the builder's pass 3) ----
    {
        // MUL_MAT + ADD of a bias row (addend stride 0) and of the residual (addend stride N/16)
        const uint32_t N = 64, K = 64, T = 3;
        const weight w = next_w(N, K);
        const uint64_t X = IO, BIAS = IO + 0x1000, RES = IO + 0x2000, Y1 = IO + 0x3000, Y2 = IO + 0x4000, Y3 = IO + 0x5000, Y4 = IO + 0x6000,
                       Y5 = IO + 0x7000, Y6 = IO + 0x8000;
        write_f32(tp, X, rnd_vec(T * K));
        const std::vector<float> bias = rnd_vec(N), res = rnd_vec(T * N);
        write_f32(tp, BIAS, bias);
        write_f32(tp, RES, res);
        uint32_t a[mm::NARGS];
        tsr_opl_builder u(cfg.num_ch);
        mm_args(a, X, K / 16, Y1, N / 16, w, K, T);
        u.mul_mat(a, 0);
        add_m7(u, opl::K_ADD, opl::ew_args{ H0(Y2), N / 16, H0(Y1), N / 16, H0(BIAS), 0, N, T, 1, 0 }, 1);
        mm_args(a, X, K / 16, Y4, N / 16, w, K, T);
        u.mul_mat(a, 2);
        add_m7(u, opl::K_ADD, opl::ew_args{ H0(Y5), N / 16, H0(Y4), N / 16, H0(RES), N / 16, N, T, T, 0 }, 3);
        tsr_opl_builder f(cfg.num_ch);
        mm_args(a, X, K / 16, Y3, N / 16, w, K, T);
        a[mm::ADD] = H0(BIAS);
        a[mm::ADD_STRIDE] = 0;
        f.mul_mat(a, 0);
        mm_args(a, X, K / 16, Y6, N / 16, w, K, T);
        a[mm::ADD] = H0(RES);
        a[mm::ADD_STRIDE] = N / 16;
        f.mul_mat(a, 1);
        run_list(dev, u.finish(), write, "F unfused MUL_MAT + ADD");
        run_list(dev, f.finish(), write, "F fused MUL_MAT + addend");
        const std::vector<float> y2 = read_f32(tp, Y2, T * N), y3 = read_f32(tp, Y3, T * N), y5 = read_f32(tp, Y5, T * N),
                                 y6 = read_f32(tp, Y6, T * N), y1 = read_f32(tp, Y1, T * N);
        CHECK(bit_diff(y2, y3) == 0, "F: bias: fused differs from MUL_MAT + ADD in %zu values", bit_diff(y2, y3));
        CHECK(bit_diff(y5, y6) == 0, "F: residual: fused differs from MUL_MAT + ADD in %zu values", bit_diff(y5, y6));
        std::vector<float> want(T * N);
        for (uint32_t i = 0; i < T * N; ++i) want[i] = tsr_fadd(y1[i], bias[i % N]);
        CHECK(bit_diff(y2, want) == 0, "F: the device ADD is not one f32 addition");
        // RMS_NORM + MUL by a weight: one row of 896, per-head rows (14 heads x 2 tokens of 64, period 1),
        // per-row weights (period = rows)
        struct norm_case { uint32_t n, rows, period; const char * what; };
        const norm_case nc[] = { { 896, 1, 1, "0.5B decode row" }, { 896, 3, 1, "3 rows" }, { 64, 28, 1, "per-head rows" },
                                 { 128, 3, 3, "per-row weights" } };
        for (const norm_case & c : nc) {
            const uint32_t L = (c.n + 15) / 16;
            const uint64_t XN = IO + 0x10000, W = IO + 0x20000, Z1 = IO + 0x30000, Z2 = IO + 0x40000, Z3 = IO + 0x50000;
            write_f32(tp, XN, rnd_vec(c.rows * L * 16, 3.0f));
            write_f32(tp, W, rnd_vec(c.period * L * 16));
            const uint32_t eps = fb(1e-6f), inv_n = fb(1.0f / (float) c.n);
            tsr_opl_builder un(cfg.num_ch), fu(cfg.num_ch);
            add_m7(un, opl::K_RMS_NORM, opl::rms_norm_args{ H0(Z1), L, H0(XN), L, c.n, c.rows, eps, inv_n, opl::HANDLE_NONE, 1 }, 0);
            add_m7(un, opl::K_MUL, opl::ew_args{ H0(Z2), L, H0(Z1), L, H0(W), c.period > 1 ? L : 0u, c.n, c.rows, c.period, 0 }, 1);
            add_m7(fu, opl::K_RMS_NORM, opl::rms_norm_args{ H0(Z3), L, H0(XN), L, c.n, c.rows, eps, inv_n, H0(W), c.period }, 0);
            CHECK(un.op(1).dep0 == 0, "F: MUL after RMS_NORM (RAW)");
            run_list(dev, un.finish(), write, "F unfused RMS_NORM + MUL");
            run_list(dev, fu.finish(), write, "F fused RMS_NORM");
            const std::vector<float> z2 = read_f32(tp, Z2, c.rows * L * 16), z3 = read_f32(tp, Z3, c.rows * L * 16);
            std::vector<float> z2n, z3n;   // the rows' n values (the line tails are not part of the result)
            for (uint32_t r = 0; r < c.rows; ++r) {
                z2n.insert(z2n.end(), z2.begin() + r * L * 16, z2.begin() + r * L * 16 + c.n);
                z3n.insert(z3n.end(), z3.begin() + r * L * 16, z3.begin() + r * L * 16 + c.n);
            }
            CHECK(bit_diff(z2n, z3n) == 0, "F: RMS_NORM + MUL (%s): fused differs in %zu values", c.what, bit_diff(z2n, z3n));
        }
        printf("  F: fusion bit-identical: MUL_MAT + ADD (bias, residual), RMS_NORM + MUL (4 shapes)  %s\n",
               g_fail == fail0 ? "ok" : "FAILED");
    }

    // ---- G: a chain of every short kind with preload entries, its inputs changed between launches ----
    {
        const int fail1 = g_fail;
        const uint32_t n = 64, L = 4, SR = 8;
        const uint64_t PP = IO + 0x60000, PI = IO + 0x60040, S = IO + 0x61000, V = IO + 0x62000, BR = IO + 0x63000, Q = IO + 0x64000,
                       W64 = IO + 0x65000, G1 = IO + 0x70000, C1 = IO + 0x71000, S1 = IO + 0x72000, M1 = IO + 0x73000, A1 = IO + 0x74000,
                       U1 = IO + 0x75000, GL = IO + 0x76000, RQ = IO + 0x77000, RN = IO + 0x78000;
        const std::vector<float> src = rnd_vec(SR * n), vs = rnd_vec(16), brow = rnd_vec(n);
        write_f32(tp, S, src);
        write_f32(tp, V, vs);
        write_f32(tp, BR, brow);
        write_f32(tp, Q, rnd_vec(2 * 2 * n));
        write_f32(tp, W64, rnd_vec(n));
        auto set_inputs = [&](const int32_t (&pos)[2], const int32_t (&ids)[3]) {
            CHECK(tsr_transport_write(tp, 0, PP, pos, sizeof pos) == 0 && tsr_transport_write(tp, 0, PI, ids, sizeof ids) == 0, "inputs");
        };
        auto build = [&](uint32_t flags) {
            tsr_opl_builder b(cfg.num_ch, flags);
            const uint32_t pp = b.preload(H0(PP), 8), pi = b.preload(H0(PI), 12);
            CHECK(pp == 0 && pi == 64 && b.preload(H0(PP), 8) == 0, "preload entries %u %u", pp, pi);
            add_m7(b, opl::K_GET_ROWS, opl::get_rows_args{ H0(G1), L, H0(S), L, pi, 3, L }, 0, SR);
            add_m7(b, opl::K_CPY, opl::cpy_args{ H0(C1), L, H0(G1), L, L, 3 }, 1);
            add_m7(b, opl::K_SCALE, opl::scale_args{ H0(S1), L, H0(C1), L, n, 3, fb(0.5f), fb(0.25f) }, 2);
            add_m7(b, opl::K_MUL, opl::ew_args{ H0(M1), L, H0(S1), L, H0(V), 0, n, 3, 3, opl::ew::FLAG_B_SCALAR }, 3);
            add_m7(b, opl::K_ADD, opl::ew_args{ H0(A1), L, H0(M1), L, H0(BR), 0, n, 3, 1, 0 }, 4);
            add_m7(b, opl::K_UNARY, opl::unary_args{ H0(U1), L, H0(A1), L, n, 3, opl::unary::TANH }, 5);
            add_m7(b, opl::K_GLU, opl::glu_args{ H0(GL), L, H0(A1), L, H0(U1), L, n, 3, opl::glu::SWIGLU }, 6);
            add_m7(b, opl::K_ROPE, opl::rope_args{ H0(RQ), 2 * L, H0(Q), 2 * L, pp, opl::HANDLE_NONE, fb(powf(10000.0f, -2.0f / 64.0f)),
                                                   fb(1.0f), fb(1.0f), 64, 64, 2, 2, opl::rope::MODE_NEOX }, 7);
            add_m7(b, opl::K_RMS_NORM, opl::rms_norm_args{ H0(RN), L, H0(RQ), L, n, 4, fb(1e-6f), fb(1.0f / 64.0f), H0(W64), 1 }, 8);
            CHECK(b.op(1).dep0 == 0 && b.op(2).dep0 == 1 && b.op(6).dep0 == 5 && b.op(6).dep1 == opl::DEP_NONE && b.op(8).dep0 == 7 &&
                  b.op(7).dep0 == opl::DEP_NONE, "G: dependencies");
            return b.finish();
        };
        const std::vector<uint8_t> blob = build(opl::F_STOP_ON_ERROR), serial = build(opl::F_STOP_ON_ERROR | opl::F_SERIAL);
        const uint64_t outs[] = { G1, C1, S1, M1, A1, U1, GL, RQ, RN };
        auto read_all = [&]() {
            std::vector<std::vector<float>> o;
            for (uint64_t off : outs) o.push_back(read_f32(tp, off, off == RQ || off == RN ? 4 * n : 3 * n));
            return o;
        };
        auto check_exact = [&](const int32_t (&ids)[3], const std::vector<std::vector<float>> & o, const char * when) {
            std::vector<float> g(3 * n), s1(3 * n), m1(3 * n), a1(3 * n);
            for (uint32_t r = 0; r < 3; ++r) {
                for (uint32_t i = 0; i < n; ++i) {
                    g[r * n + i] = src[(size_t) ids[r] * n + i];
                    s1[r * n + i] = tsr_fadd(tsr_fmul(g[r * n + i], 0.5f), 0.25f);
                    m1[r * n + i] = tsr_fmul(s1[r * n + i], vs[r]);
                    a1[r * n + i] = tsr_fadd(m1[r * n + i], brow[i]);
                }
            }
            CHECK(bit_diff(o[0], g) == 0 && bit_diff(o[1], g) == 0, "G (%s): GET_ROWS / CPY are not the source rows", when);
            CHECK(bit_diff(o[2], s1) == 0, "G (%s): SCALE x·0.5 + 0.25", when);
            CHECK(bit_diff(o[3], m1) == 0, "G (%s): MUL by per-row scalars", when);
            CHECK(bit_diff(o[4], a1) == 0, "G (%s): ADD of a broadcast row", when);
        };
        const int32_t pos0[2] = { 5, 6 }, ids0[3] = { 7, 0, 3 };
        set_inputs(pos0, ids0);
        run_list(dev, serial, write, "G serial");
        const auto ref0 = read_all();
        run_list(dev, blob, write, "G");
        const auto got0 = read_all();
        check_exact(ids0, got0, "first inputs");
        for (size_t i = 0; i < got0.size(); ++i) {
            CHECK(bit_diff(got0[i], ref0[i]) == 0, "G: output %zu differs from the SERIAL run", i);
        }
        // the next "token": new positions and ids through the same list (an OPL cache hit, no rebuild)
        const int32_t pos1[2] = { 900, 17 }, ids1[3] = { 2, 2, 6 };
        set_inputs(pos1, ids1);
        const uint64_t hits = dev.cache().get_stats().hits;
        run_list(dev, blob, write, "G again");
        CHECK(dev.cache().get_stats().hits == hits + 1, "G: the list was not a cache hit");
        const auto got1 = read_all();
        check_exact(ids1, got1, "second inputs");
        CHECK(bit_diff(got1[7], got0[7]) != 0, "G: ROPE did not read the new positions");
        run_list(dev, serial, write, "G serial again");
        const auto ref1 = read_all();
        for (size_t i = 0; i < got1.size(); ++i) {
            CHECK(bit_diff(got1[i], ref1[i]) == 0, "G: output %zu differs from the SERIAL run (second inputs)", i);
        }
        printf("  G: GET_ROWS, CPY, SCALE, MUL (per-row scalars), ADD, UNARY, GLU, ROPE, RMS_NORM with preload entries; new "
               "inputs through the cached list  %s\n", g_fail == fail1 ? "ok" : "FAILED");
    }

    // ---- H: a TILE_ERR in an M7 record names its tag (SW §9.3 step 5 for every kind) ----
    {
        const int fail2 = g_fail;
        const uint64_t PI = IO + 0x80000, S = IO + 0x61000, G2 = IO + 0x81000, A2 = IO + 0x82000;
        const int32_t ids[2] = { 1, -1 };      // a negative id: LD beat 0xFFFF_FFFF, DRAM_RANGE (SW §7.2)
        CHECK(tsr_transport_write(tp, 0, PI, ids, sizeof ids) == 0, "ids");
        tsr_opl_builder b(cfg.num_ch);
        const uint32_t pi = b.preload(H0(PI), 8);
        add_m7(b, opl::K_ADD, opl::ew_args{ H0(A2), 4, H0(S), 4, H0(S), 4, 64, 2, 2, 0 }, 20);
        add_m7(b, opl::K_GET_ROWS, opl::get_rows_args{ H0(G2), 4, H0(S), 4, pi, 2, 4 }, 21, 8);
        const tsr_opl_run_result r = dev.run(b.finish(), 0, write);
        CHECK(r.rc == 0 && r.res.status == TSR_LCQ_R0_STATUS_TILE_ERR && r.res.err_code == TSR_ERR_DRAM_RANGE && tsr_opl_failed_op(r, b.n_ops()) == 1 &&
                  tsr_opl_failed_tag(b, r) == 21,
              "H: status %u ret %d err %u: %s", r.res.status, r.res.ret, r.res.err_code, tsr_opl_describe(r, "get_rows").c_str());
        printf("  H: %s  %s\n", tsr_opl_describe(r, "GET_ROWS (tag 21)").c_str(), g_fail == fail2 ? "ok" : "FAILED");
    }
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

    // ---- M7 (SW §9.3): the builder's fusion, preload table, kind ranges and error mapping ----
    test_m7(tp, cfg, dev, write, next_w);
    printf("  launches %" PRIu64 ", op records %" PRIu64 "\n", dev.launches, dev.ops);

    tsr_transport_close(tp);
    if (g_fail) {
        printf("test-tsr-opl-launch: %d check(s) FAILED\n", g_fail);
        return 1;
    }
    printf("test-tsr-opl-launch: OK\n");
    return 0;
}
