// test-tsr-kinds.cpp — the framework-free encoders of tsr-kinds.h and the reference executor of sw/ctrl/ref/opl_ref.hpp
// (ctest ggml_tesseract_kinds; docs/design/06-software-verification.md §7.2, §9.3). No ggml.
//
// Every list is built with the encoders through tsr_opl_builder (tsr_k_emit) and run through tsr_opl_dev on the transport selected
// by TSR_DEVICE (default sim; the build's firmware image). Three images of DRAM are kept: the device's, A (tsr::ref::opl_exec
// run on the blob) and B (the per-kind reference called directly with argument words written here by hand, not by the encoders).
//
//   For each kind (MUL_MAT as MUL_MAT, as MUL_MAT_BEGIN / MUL_MAT_END, with a bias addend, with a residual addend, with a K part
//   in ACC_DST mode, Q8_0 and Q4_K; ADD, MUL with the three forms of b; SCALE; RMS_NORM with and without a weight; GLU; UNARY of
//   every function; ROPE with an angle table, both pairings; CPY; SET_ROWS_KV + FLASH_ATTN with causal, sliding, random and
//   no mask, GQA, decode and prefill, with and without cells before the new ones) and T in {1, 2, 5, 8, 17}:
//     (a) every DRAM range the list writes (the builder's write ranges) equals opl_exec's image, bit for bit;
//     (b) for MUL_MAT, RMS_NORM, ROPE and the SET_ROWS_KV + FLASH_ATTN pair the outputs equal the direct reference's.
//   Then one decoder layer of Qwen2.5-0.5B's shape (896, 14 / 2 heads of 64, n_ff 4864) with random Q8_0 weights as one list, for
//   several T: (a).
//   Negative controls (the comparison must fail): a wrong inv_n word, ROPE with the other pairing, FLASH_ATTN without FA_NEW.
//   Refusals: a kind the image does not have, shapes the planners refuse.
#include <sys/mman.h>

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "opl_ref.hpp"
#include "elem_ref.hpp"
#include "kvfa_ref.hpp"
#include "mulmat_ref.hpp"
#include "rmsnorm_ref.hpp"
#include "rope_ref.hpp"
#include "tsr-kinds.h"
#include "tsr-wlayout.h"
#include "tsr_ref.h"

namespace opl = tsr::opl;

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

static std::mt19937 g_rng(20240607);
static float rnd(float scale = 1.0f) { return std::uniform_real_distribution<float>(-scale, scale)(g_rng); }
static float rnd_n(float sigma) { return std::normal_distribution<float>(0.0f, sigma)(g_rng); }
static uint32_t rnd_u(uint32_t n) { return (uint32_t) (g_rng() % n); }
static uint32_t f2b(float f) { return tsr_f32_bits(f); }

static const uint32_t TS[] = { 1, 2, 5, 8, 17 };

// ---------------------------------------------------------------------------------------------
// The device and the two host images
// ---------------------------------------------------------------------------------------------
static constexpr uint64_t IMG_BYTES = TSR_REGION_WEIGHT_OFF + (96ull << 20);   // per channel: the regions below the heap + 96 MiB of it

struct world {
    tsr_transport * tp = nullptr;
    tsr_config cfg = {};
    tsr_opl_dev dev;
    tsr_kenv kenv;
    uint32_t C = 1;
    uint8_t * img[2][TSR_NUM_CH_MAX] = {};      // 0: A (opl_exec), 1: B (direct references)
    uint8_t * const * pa() const { return img[0]; }
    tsr::ref::dram_view view(int w) const { return tsr::ref::dram_view{ img[w], C, IMG_BYTES }; }
    uint64_t io = TSR_REGION_IO_OFF, aux = TSR_REGION_AUX_OFF, heap = TSR_REGION_WEIGHT_OFF;
    tsr::ref::opl_exec_cfg xcfg;

    // channel-0 byte offsets (64-byte aligned bump allocators; every case gets fresh memory)
    uint64_t alloc_io(uint64_t bytes) { const uint64_t r = io; io += (bytes + 63u) / 64u * 64u + 64u; return r; }
    uint64_t alloc_aux(uint64_t bytes) { const uint64_t r = aux; aux += (bytes + 63u) / 64u * 64u + 64u; return r; }
    uint64_t alloc_heap(uint64_t bytes) { const uint64_t r = heap; heap += (bytes + 4095u) / 4096u * 4096u; return r; }

    // writes to the device and to both images
    void put(uint32_t ch, uint64_t off, const void * p, size_t n) {
        if (n == 0) return;
        CHECK(off + n <= IMG_BYTES, "put beyond the image (ch %u, off 0x%" PRIx64 ")", ch, off);
        CHECK(tsr_transport_write(tp, ch, off, p, n) == 0, "device write");
        memcpy(img[0][ch] + off, p, n);
        memcpy(img[1][ch] + off, p, n);
    }
    void put_f32(uint64_t off, const std::vector<float> & v) { put(0, off, v.data(), v.size() * 4); }
    std::vector<uint8_t> dev_read(uint32_t ch, uint64_t off, size_t n) {
        std::vector<uint8_t> v(n);
        CHECK(tsr_transport_read(tp, ch, off, v.data(), n) == 0, "device read");
        return v;
    }
};

static world * W = nullptr;

static uint32_t H0(uint64_t byte_off) { return tsr_dram_handle(0, byte_off); }

// f32 rows [rows][n], row r at stride lines from `off`: written to the device and the images; the padding of every line is zero.
static tsr_k_rows put_rows(uint64_t off, const std::vector<float> & v, uint32_t rows, uint32_t n, uint32_t stride) {
    std::vector<float> img((size_t) rows * stride * 16u, 0.0f);
    for (uint32_t r = 0; r < rows; ++r) memcpy(&img[(size_t) r * stride * 16u], &v[(size_t) r * n], (size_t) n * 4);
    W->put_f32(off, img);
    return { H0(off), stride };
}

// A fresh area for `rows` output rows of n f32 (stride lines), filled with a pattern so that an unwritten byte is noticed.
static tsr_k_rows new_out(uint32_t rows, uint32_t stride) {
    const uint64_t off = W->alloc_io((uint64_t) rows * stride * 64u);
    std::vector<uint8_t> junk((size_t) rows * stride * 64u, 0xA5);
    W->put(0, off, junk.data(), junk.size());
    return { H0(off), stride };
}

static std::vector<float> rnd_vec(size_t n, float scale) {
    std::vector<float> v(n);
    for (float & x : v) x = rnd(scale);
    return v;
}

// Device output rows against values computed on the host here (independently of the encoders and of the references):
// rtol = 0: the same bits; else |d - e| <= rtol |e| + atol. Returns the number of differing values.
static size_t cmp_f32(const tsr_k_rows & dst, uint32_t rows, uint32_t n, const std::vector<float> & want, double rtol, double atol) {
    size_t bad = 0;
    for (uint32_t r = 0; r < rows; ++r) {
        const std::vector<uint8_t> raw = W->dev_read(0, (uint64_t) (dst.h & 0x0FFFFFFFu) * 64 + (uint64_t) r * dst.stride * 64, (size_t) n * 4);
        for (uint32_t i = 0; i < n; ++i) {
            float d, e = want[(size_t) r * n + i];
            memcpy(&d, raw.data() + 4u * i, 4);
            if (rtol == 0.0) bad += memcmp(&d, &e, 4) != 0;
            else bad += !(std::fabs((double) d - (double) e) <= rtol * std::fabs((double) e) + atol);
        }
    }
    return bad;
}

// ---------------------------------------------------------------------------------------------
// Running a list and comparing
// ---------------------------------------------------------------------------------------------
struct run_result {
    bool dev_ok = false;
    size_t mism_exec = 0;      // bytes where the device differs from opl_exec's image in a range the list writes
    uint32_t records = 0, joins = 0, begins = 0;
};

static size_t cmp_bytes(uint32_t ch, uint64_t off, uint64_t n, int w) {
    const std::vector<uint8_t> d = W->dev_read(ch, off, (size_t) n);
    return memcmp(d.data(), W->img[w][ch] + off, (size_t) n) != 0 ? 1u : 0u;
}

// 1 per differing contiguous run (a row of a strided range, or the whole range)
static size_t cmp_range(const tsr_dram_range & r, int w) {
    if (r.hi <= r.lo) return 0;
    size_t bad = 0;
    if (r.row != 0 && r.stride != 0 && r.row < r.stride) {
        for (uint64_t off = r.lo; off + r.row <= r.hi; off += r.stride) bad += cmp_bytes(r.ch, off, r.row, w);
    } else {
        bad += cmp_bytes(r.ch, r.lo, r.hi - r.lo, w);
    }
    return bad;
}

// Runs the builder's list on the device and through opl_exec on image A; compares every write range.
static run_result run_list(tsr_opl_builder & b, const char * what) {
    run_result rr;
    CHECK(b.settle(), "%s: settle: %s", what, b.error());
    const std::vector<uint8_t> blob = b.finish();
    CHECK(!blob.empty(), "%s: finish: %s", what, b.error());
    if (blob.empty()) return rr;
    rr.records = b.n_ops();
    rr.joins = b.n_joins();
    rr.begins = b.n_of_kind(opl::K_MUL_MAT_BEGIN);
    const tsr_opl_run_result r = W->dev.run(blob, 0, [](uint32_t ch, uint64_t off, const void * p, size_t n) {
        CHECK(tsr_transport_write(W->tp, ch, off, p, n) == 0, "write");
    });
    rr.dev_ok = r.rc == 0 && r.res.type == TSR_LCQ_R0_TYPE_DONE && r.res.status == TSR_LCQ_R0_STATUS_OK && r.res.ret == 0;
    CHECK(rr.dev_ok, "%s: device: %s", what, tsr_opl_describe(r, nullptr).c_str());
    std::string why;
    const int xrc = tsr::ref::opl_exec(blob, W->view(0), W->xcfg, &why);
    CHECK(xrc == 0, "%s: opl_exec: %s", what, why.c_str());
    if (!rr.dev_ok || xrc != 0) return rr;
    for (uint32_t i = 0; i < b.n_ops(); ++i) {
        for (const tsr_dram_range & w : b.op(i).wr) rr.mism_exec += cmp_range(w, 0);
    }
    CHECK(rr.mism_exec == 0, "%s: %zu write ranges differ between the device and opl_exec", what, rr.mism_exec);
    return rr;
}

// Compares `rows` rows of row_bytes bytes (stride bytes apart) at ch0 off: device against image B. Returns the differing rows.
static size_t cmp_b(uint64_t off, uint32_t rows, uint64_t row_bytes, uint64_t stride) {
    size_t bad = 0;
    for (uint32_t r = 0; r < rows; ++r) {
        const std::vector<uint8_t> d = W->dev_read(0, off + r * stride, (size_t) row_bytes);
        bad += memcmp(d.data(), W->img[1][0] + off + r * stride, (size_t) row_bytes) != 0 ? 1u : 0u;
    }
    return bad;
}

static bool emit(tsr_opl_builder & b, tsr_rec & r, int32_t tag, tsr_fanew_map * fm = nullptr, bool defer = false) {
    uint32_t i = tsr_k_emit(b, r, tag, fm, defer);
    if (i == tsr_opl_builder::NO_ROOM || i == tsr_opl_builder::FAILED) {
        CHECK(false, "emit tag %d (%s): %s", tag, opl::kind_name(r.kind), b.error());
        return false;
    }
    return true;
}

#define ENC(call)                                                                                  \
    do {                                                                                           \
        std::string _why;                                                                          \
        const bool _ok = (call);                                                                   \
        CHECK(_ok, "encoder refused: %s", _why.c_str());                                           \
        if (!_ok) return;                                                                          \
    } while (0)

static tsr_opl_builder new_builder() { return tsr_opl_builder(W->C, opl::F_STOP_ON_ERROR); }

// ---------------------------------------------------------------------------------------------
// Weights
// ---------------------------------------------------------------------------------------------
struct weight_t {
    uint32_t fmt, N, K;
    uint64_t w_off;   // bytes
};

static weight_t make_weight(uint32_t fmt, uint32_t N, uint32_t K, float d_scale = 1.0f) {
    const uint32_t bs = tsr_fmt_block_size(fmt);
    const size_t row_bytes = fmt == TSR_FMT_Q8_0 ? (size_t) K / 32 * sizeof(tsr_block_q8_0) : (size_t) K / 256 * sizeof(tsr_block_q4_K);
    std::vector<uint8_t> blocks((size_t) N * row_bytes);
    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t b = 0; b < K / bs; ++b) {
            uint8_t * p = blocks.data() + (size_t) n * row_bytes + (size_t) b * (fmt == TSR_FMT_Q8_0 ? sizeof(tsr_block_q8_0) : sizeof(tsr_block_q4_K));
            if (fmt == TSR_FMT_Q8_0) {
                tsr_block_q8_0 q;
                q.d = (uint16_t) (0x1C00u + (d_scale < 1.0f ? 0 : 0x400u) + rnd_u(0x300u));   // positive fp16 scales
                for (int i = 0; i < 32; ++i) q.qs[i] = (int8_t) (rnd_u(256u));
                memcpy(p, &q, sizeof(q));
            } else {
                tsr_block_q4_K q;
                q.d = (uint16_t) (0x2400u + rnd_u(0x200u));
                q.dmin = (uint16_t) (0x2000u + rnd_u(0x200u));
                for (auto & s : q.scales) s = (uint8_t) rnd_u(256u);
                for (auto & s : q.qs) s = (uint8_t) rnd_u(256u);
                memcpy(p, &q, sizeof(q));
            }
        }
    }
    weight_t w{ fmt, N, K, W->alloc_heap(tsr_tensor_ch_beats(fmt, N, K, W->C) * 64u) };
    tsr_weight_store_rows(W->C, fmt, N, K, w.w_off, 0, N, 1u << 20,
        [&](uint32_t r, void *) -> const void * { return blocks.data() + (size_t) r * row_bytes; },
        [](uint32_t ch, uint64_t off, const void * p, size_t n) { W->put(ch, off, p, n); });
    return w;
}

// ---------------------------------------------------------------------------------------------
// MUL_MAT
// ---------------------------------------------------------------------------------------------
enum { MM_PLAIN = 0, MM_BIAS = 1, MM_RESID = 2, MM_ACC = 4, MM_DEFER = 8 };

static void test_mul_mat(uint32_t fmt, uint32_t N, uint32_t K, uint32_t T, uint32_t mode) {
    char what[160];
    snprintf(what, sizeof(what), "MUL_MAT %s N %u K %u T %u mode %u", fmt == TSR_FMT_Q8_0 ? "Q8_0" : "Q4_K", N, K, T, mode);
    const bool acc = (mode & MM_ACC) != 0;
    const uint32_t Kp = acc ? K / 2 : K;                   // a K part
    const weight_t w0 = make_weight(fmt, N, Kp), w1 = acc ? make_weight(fmt, N, Kp) : w0;
    const uint32_t xs = K / 16, ys = tsr_k_lines(N);
    const tsr_k_rows x = put_rows(W->alloc_io((uint64_t) T * xs * 64), rnd_vec((size_t) T * K, 1.0f), T, K, xs);
    const tsr_k_rows y = new_out(T, ys);
    tsr_k_rows add;
    if (mode & MM_BIAS) add = put_rows(W->alloc_aux((uint64_t) ys * 64), rnd_vec(N, 1.0f), 1, N, ys), add.stride = 0;
    if (mode & MM_RESID) add = put_rows(W->alloc_io((uint64_t) T * ys * 64), rnd_vec((size_t) T * N, 1.0f), T, N, ys);

    tsr_opl_builder b = new_builder();
    tsr_rec r0, r1;
    tsr_kd_mm d{};
    d.x = x; d.y = y; d.add = add; d.w_off = (uint32_t) (w0.w_off / 64); d.fmt = fmt; d.N = N; d.K = Kp; d.T = T;
    ENC(tsr_k_mul_mat(W->kenv, d, &r0, &_why));
    const bool defer = (mode & MM_DEFER) != 0 && tsr_k_mm_deferrable(W->kenv, r0);
    if ((mode & MM_DEFER) && !defer) return;               // this shape does not defer (more than one T tile): not a case
    b.set_defer(defer);
    if (!emit(b, r0, 0, nullptr, defer)) return;
    if (defer) {
        // The builder lowers a pending MUL_MAT as BEGIN only when the next record leaves y alone, and gives it its END at the first
        // record that touches y: a copy of x, then a copy of y.
        tsr_kd_cpy cx{}, cy{};
        cx.src = x; cx.dst = new_out(T, xs); cx.row_lines = xs; cx.rows = T;
        cy.src = y; cy.dst = new_out(T, ys); cy.row_lines = ys; cy.rows = T;
        tsr_rec rx, ry;
        ENC(tsr_k_cpy(W->kenv, cx, &rx, &_why));
        ENC(tsr_k_cpy(W->kenv, cy, &ry, &_why));
        if (!emit(b, rx, 1) || !emit(b, ry, 2)) return;
    }
    if (acc) {
        d.x.h = x.h + Kp / 16;                             // the second K part's columns
        d.add = tsr_k_rows();
        d.acc_dst = true;
        d.w_off = (uint32_t) (w1.w_off / 64);
        ENC(tsr_k_mul_mat(W->kenv, d, &r1, &_why));
        if (!emit(b, r1, 1, nullptr, false)) return;
    }
    const run_result rr = run_list(b, what);
    CHECK(!defer || rr.begins >= 1, "%s: deferral did not produce a MUL_MAT_BEGIN", what);

    // (b) the direct reference with hand-written words
    const tsr::ref::dram_view vb = W->view(1);
    const bool has_add = (mode & (MM_BIAS | MM_RESID)) != 0;
    uint32_t a[12] = { x.h, xs, y.h, ys, (uint32_t) (w0.w_off / 64), fmt, N, Kp, T, has_add ? add.h : opl::HANDLE_NONE,
                       has_add ? add.stride : 0u, 0u };
    CHECK(tsr::ref::mulmat_ref(vb, a) == 0, "%s: direct reference", what);
    if (acc) {
        uint32_t a2[12] = { x.h + Kp / 16, xs, y.h, ys, (uint32_t) (w1.w_off / 64), fmt, N, Kp, T, opl::HANDLE_NONE, 0u, opl::mm::FLAG_ACC_DST };
        CHECK(tsr::ref::mulmat_ref(vb, a2) == 0, "%s: direct reference (K part 2)", what);
    }
    const size_t bad = cmp_b((uint64_t) (y.h & 0x0FFFFFFFu) * 64, T, (uint64_t) ys * 64, (uint64_t) ys * 64);
    CHECK(bad == 0, "%s: %zu output rows differ from the direct reference", what, bad);
}

static void test_mul_mats() {
    printf("MUL_MAT\n");
    for (uint32_t T : TS) {
        test_mul_mat(TSR_FMT_Q8_0, 128, 64, T, MM_PLAIN);
        test_mul_mat(TSR_FMT_Q8_0, 300, 96, T, MM_PLAIN | MM_DEFER);
        test_mul_mat(TSR_FMT_Q8_0, 96, 256, T, MM_BIAS);
        test_mul_mat(TSR_FMT_Q8_0, 96, 256, T, MM_BIAS | MM_DEFER);
        test_mul_mat(TSR_FMT_Q8_0, 128, 128, T, MM_RESID);
        test_mul_mat(TSR_FMT_Q8_0, 128, 128, T, MM_RESID | MM_DEFER);
        test_mul_mat(TSR_FMT_Q8_0, 128, 256, T, MM_ACC);
        test_mul_mat(TSR_FMT_Q4_K, 64, 256, T, MM_PLAIN);
        test_mul_mat(TSR_FMT_Q4_K, 64, 512, T, MM_BIAS | MM_DEFER);
    }
}

// ---------------------------------------------------------------------------------------------
// Element-wise kinds
// ---------------------------------------------------------------------------------------------
static void test_ew(uint32_t kind, uint32_t n, uint32_t T, int form) {   // form 0: row b, 1: one row broadcast, 2: scalar
    char what[96];
    snprintf(what, sizeof(what), "%s n %u T %u b form %d", opl::kind_name(kind), n, T, form);
    const uint32_t L = tsr_k_lines(n), sa = L + (T > 1 ? 1u : 0u);   // a with padded rows
    const std::vector<float> av = rnd_vec((size_t) T * n, 3.0f);
    const tsr_k_rows a = put_rows(W->alloc_io((uint64_t) T * sa * 64), av, T, n, sa);
    const tsr_k_rows dst = new_out(T, sa);
    tsr_kd_ew d{};
    d.kind = kind; d.dst = dst; d.a = a; d.n = n; d.rows = T;
    std::vector<float> bv;                                   // b as the rows see it
    if (form == 0) {
        bv = rnd_vec((size_t) T * n, 3.0f);
        d.b = put_rows(W->alloc_io((uint64_t) T * L * 64), bv, T, n, L);
        d.b_period = T;
    } else if (form == 1) {
        const std::vector<float> row = rnd_vec(n, 3.0f);
        d.b = put_rows(W->alloc_aux((uint64_t) L * 64), row, 1, n, L);
        d.b_period = 1;
        for (uint32_t t = 0; t < T; ++t) bv.insert(bv.end(), row.begin(), row.end());
    } else {
        std::vector<float> sc(16, 0.0f);
        sc[0] = rnd(2.0f);
        d.b = put_rows(W->alloc_aux(64), sc, 1, 16, 1);
        d.b_scalar = true;
        bv.assign((size_t) T * n, sc[0]);
    }
    tsr_opl_builder b = new_builder();
    tsr_rec r;
    ENC(tsr_k_add(W->kenv, d, &r, &_why));
    if (!emit(b, r, 0)) return;
    run_list(b, what);
    std::vector<float> want((size_t) T * n);
    for (size_t i = 0; i < want.size(); ++i) want[i] = kind == opl::K_ADD ? av[i] + bv[i] : av[i] * bv[i];
    const size_t bad = cmp_f32(dst, T, n, want, 0.0, 0.0);
    CHECK(bad == 0, "%s: %zu values differ from the host's f32 %s", what, bad, kind == opl::K_ADD ? "sum" : "product");
}

static void test_scale(uint32_t n, uint32_t T, float s, float bias) {
    char what[96];
    snprintf(what, sizeof(what), "SCALE n %u T %u s %g b %g", n, T, (double) s, (double) bias);
    const uint32_t L = tsr_k_lines(n);
    const std::vector<float> xv = rnd_vec((size_t) T * n, 4.0f);
    tsr_kd_scale d{};
    d.src = put_rows(W->alloc_io((uint64_t) T * L * 64), xv, T, n, L);
    d.dst = new_out(T, L);
    d.n = n; d.rows = T; d.s = s; d.b = bias;
    tsr_opl_builder b = new_builder();
    tsr_rec r;
    ENC(tsr_k_scale(W->kenv, d, &r, &_why));
    if (!emit(b, r, 0)) return;
    run_list(b, what);
    std::vector<float> want(xv.size());
    for (size_t i = 0; i < want.size(); ++i) {
        volatile float m = xv[i] * s;
        want[i] = bias != 0.0f ? (float) m + bias : (float) m;
    }
    const size_t bad = cmp_f32(d.dst, T, n, want, 0.0, 0.0);
    CHECK(bad == 0, "%s: %zu values differ from the host's x * s (+ b)", what, bad);
}

static double h_silu(double x) { return x / (1.0 + std::exp(-x)); }
static double h_gelu(double x) { return 0.5 * x * (1.0 + std::tanh(0.7978845608028654 * (x + 0.044715 * x * x * x))); }

static void test_glu(uint32_t n, uint32_t T, uint32_t variant) {
    char what[96];
    snprintf(what, sizeof(what), "GLU variant %u n %u T %u", variant, n, T);
    const uint32_t L = tsr_k_lines(n);
    const std::vector<float> gv = rnd_vec((size_t) T * n, 4.0f), uv = rnd_vec((size_t) T * n, 2.0f);
    tsr_kd_glu d{};
    d.gate = put_rows(W->alloc_io((uint64_t) T * L * 64), gv, T, n, L);
    d.up = put_rows(W->alloc_io((uint64_t) T * L * 64), uv, T, n, L);
    d.dst = new_out(T, L);
    d.n = n; d.rows = T; d.variant = variant;
    tsr_opl_builder b = new_builder();
    tsr_rec r;
    ENC(tsr_k_glu(W->kenv, d, &r, &_why));
    if (!emit(b, r, 0)) return;
    run_list(b, what);
    std::vector<float> want(gv.size());
    for (size_t i = 0; i < want.size(); ++i) want[i] = (float) ((variant == opl::glu::SWIGLU ? h_silu(gv[i]) : h_gelu(gv[i])) * (double) uv[i]);
    const size_t bad = cmp_f32(d.dst, T, n, want, 2e-3, 1e-4);   // the vector unit's functions: a few ulp
    CHECK(bad == 0, "%s: %zu values differ from the host's act(gate) * up", what, bad);
}

static void test_unary(uint32_t n, uint32_t T, uint32_t func) {
    char what[96];
    snprintf(what, sizeof(what), "UNARY func %u n %u T %u", func, n, T);
    const uint32_t L = tsr_k_lines(n);
    const std::vector<float> xv = rnd_vec((size_t) T * n, func == opl::unary::EXP ? 6.0f : 8.0f);
    tsr_kd_unary d{};
    d.src = put_rows(W->alloc_io((uint64_t) T * L * 64), xv, T, n, L);
    d.dst = new_out(T, L);
    d.n = n; d.rows = T; d.func = func;
    tsr_opl_builder b = new_builder();
    tsr_rec r;
    ENC(tsr_k_unary(W->kenv, d, &r, &_why));
    if (!emit(b, r, 0)) return;
    run_list(b, what);
    std::vector<float> want(xv.size());
    for (size_t i = 0; i < want.size(); ++i) {
        const double x = xv[i];
        double y = 0;
        switch (func) {
            case opl::unary::TANH: y = std::tanh(x); break;
            case opl::unary::SILU: y = h_silu(x); break;
            case opl::unary::EXP: y = std::exp(x); break;
            case opl::unary::GELU: y = h_gelu(x); break;
            default: y = 1.0 / (1.0 + std::exp(-x)); break;
        }
        want[i] = (float) y;
    }
    const size_t bad = cmp_f32(d.dst, T, n, want, 2e-3, 1e-4);
    CHECK(bad == 0, "%s: %zu values differ from the host's function", what, bad);
}

static void test_cpy(uint32_t row_lines, uint32_t T, uint32_t pad) {
    char what[96];
    snprintf(what, sizeof(what), "CPY %u lines T %u pad %u", row_lines, T, pad);
    const uint32_t n = row_lines * 16u, ss = row_lines + pad;
    const std::vector<float> xv = rnd_vec((size_t) T * n, 5.0f);
    tsr_kd_cpy d{};
    d.src = put_rows(W->alloc_io((uint64_t) T * ss * 64), xv, T, n, ss);
    d.dst = new_out(T, row_lines);
    d.row_lines = row_lines; d.rows = T;
    tsr_opl_builder b = new_builder();
    tsr_rec r;
    ENC(tsr_k_cpy(W->kenv, d, &r, &_why));
    if (!emit(b, r, 0)) return;
    run_list(b, what);
    const size_t bad = cmp_f32(d.dst, T, n, xv, 0.0, 0.0);
    CHECK(bad == 0, "%s: %zu values differ from the source", what, bad);
}

static void test_scalar_kinds() {
    printf("ADD MUL SCALE GLU UNARY CPY\n");
    for (uint32_t T : TS) {
        for (uint32_t kind : { opl::K_ADD, opl::K_MUL }) {
            for (int form = 0; form < 3; ++form) {
                test_ew(kind, 64, T, form);
                test_ew(kind, 200, T, form);
            }
        }
        test_scale(96, T, 0.37f, 0.0f);
        test_scale(130, T, -1.5f, 0.25f);
        test_glu(64, T, opl::glu::SWIGLU);
        test_glu(256, T, opl::glu::SWIGLU);
        test_glu(96, T, opl::glu::GEGLU);
        for (uint32_t f : { opl::unary::TANH, opl::unary::SILU, opl::unary::EXP, opl::unary::GELU, opl::unary::SIGMOID }) {
            test_unary(64, T, f);
            test_unary(100, T, f);
        }
        test_cpy(4, T, 0);
        test_cpy(9, T, 3);
    }
}

// ---------------------------------------------------------------------------------------------
// RMS_NORM
// ---------------------------------------------------------------------------------------------
struct rms_setup {
    tsr_kd_rms d;
    uint32_t y_off_rows = 0;
};

static tsr_kd_rms rms_data(uint32_t n, uint32_t T, bool weight, float eps) {
    const uint32_t L = tsr_k_lines(n);
    tsr_kd_rms d{};
    d.src = put_rows(W->alloc_io((uint64_t) T * L * 64), rnd_vec((size_t) T * n, 2.0f), T, n, L);
    d.dst = new_out(T, L);
    d.n = n; d.rows = T; d.eps = eps;
    if (weight) {
        d.weight = put_rows(W->alloc_aux((uint64_t) L * 64), rnd_vec(n, 1.5f), 1, n, L).h;
        d.w_period = 1;
    }
    return d;
}

// The direct reference: words written here (inv_n = 1.0f / n as a host division).
static int rms_direct(const tsr_kd_rms & d, uint32_t inv_n_bits) {
    const uint32_t a[10] = { d.dst.h, d.dst.stride, d.src.h, d.src.stride, d.n, d.rows, f2b(d.eps), inv_n_bits, d.weight, 1u };
    return tsr::ref::rmsnorm_ref(W->view(1), a);
}

static void test_rms(uint32_t n, uint32_t T, bool weight) {
    char what[96];
    snprintf(what, sizeof(what), "RMS_NORM n %u T %u weight %d", n, T, (int) weight);
    const tsr_kd_rms d = rms_data(n, T, weight, 1e-6f);
    tsr_opl_builder b = new_builder();
    tsr_rec r;
    ENC(tsr_k_rms_norm(W->kenv, d, &r, &_why));
    CHECK(r.args[opl::rms::INV_N] == f2b(1.0f / (float) n), "%s: inv_n word", what);
    if (!emit(b, r, 0)) return;
    run_list(b, what);
    CHECK(rms_direct(d, f2b(1.0f / (float) n)) == 0, "%s: direct reference", what);
    const size_t bad = cmp_b((uint64_t) (d.dst.h & 0x0FFFFFFFu) * 64, T, (uint64_t) tsr_k_lines(n) * 64, (uint64_t) d.dst.stride * 64);
    CHECK(bad == 0, "%s: %zu rows differ from the direct reference", what, bad);
}

static void test_rms_kinds() {
    printf("RMS_NORM\n");
    for (uint32_t T : TS) {
        test_rms(64, T, true);
        test_rms(128, T, false);
        test_rms(896, T, true);
        test_rms(1000, T, true);
    }
}

// ---------------------------------------------------------------------------------------------
// ROPE
// ---------------------------------------------------------------------------------------------
struct rope_case {
    uint32_t n_heads, head_dim, n_dims, T;
    bool neox;
    tsr_kd_rope d;
    std::vector<int32_t> pos;
};

static rope_case rope_data(uint32_t nh, uint32_t hd, uint32_t nd, uint32_t T, bool neox) {
    rope_case c{};
    c.n_heads = nh; c.head_dim = hd; c.n_dims = nd; c.T = T; c.neox = neox;
    const uint32_t H = nh * hd / 16u;
    c.d.src = put_rows(W->alloc_io((uint64_t) T * H * 64), rnd_vec((size_t) T * nh * hd, 2.0f), T, nh * hd, H);
    c.d.dst = new_out(T, H);
    c.d.n_heads = nh; c.d.head_dim = hd; c.d.n_dims = nd; c.d.T = T; c.d.neox = neox;
    c.pos.resize(T);
    for (uint32_t t = 0; t < T; ++t) c.pos[t] = (int32_t) (rnd_u(1500u) + t);
    std::vector<float> pf(((T + 15u) / 16u) * 16u, 0.0f);
    memcpy(pf.data(), c.pos.data(), (size_t) T * 4);
    c.d.pos_src = put_rows(W->alloc_io(pf.size() * 4), pf, 1, (uint32_t) pf.size(), (uint32_t) pf.size() / 16u).h;
    // the model's inv_freq: base^(-2 i / n_dims) as float32
    std::vector<float> tab(nd / 2u);
    for (uint32_t i = 0; i < nd / 2u; ++i) tab[i] = (float) std::pow(1000000.0, -2.0 * (double) i / (double) nd);
    c.d.table = put_rows(W->alloc_aux((uint64_t) tsr_k_lines(nd / 2u) * 64), tab, 1, nd / 2u, tsr_k_lines(nd / 2u)).h;
    return c;
}

// The direct reference of a ROPE with NEOX | ATAB words written here; pre: the positions at byte 0.
static int rope_direct(const rope_case & c, bool neox) {
    uint32_t a[14] = { c.d.dst.h, c.d.dst.stride, c.d.src.h, c.d.src.stride, 0u, c.d.table, 0u, f2b(1.0f), f2b(1.0f), c.head_dim, c.n_dims,
                       c.n_heads, c.T, (neox ? opl::rope::MODE_NEOX : 0u) | opl::rope::MODE_ATAB };
    std::vector<uint8_t> pre(64u * ((c.T * 4u + 63u) / 64u), 0);
    memcpy(pre.data(), c.pos.data(), (size_t) c.T * 4);
    return tsr::ref::rope_ref(W->view(1), a, pre.data(), (uint32_t) pre.size());
}

static void test_rope(uint32_t nh, uint32_t hd, uint32_t nd, uint32_t T, bool neox) {
    char what[96];
    snprintf(what, sizeof(what), "ROPE %u x %u (%u rotated) T %u %s", nh, hd, nd, T, neox ? "NEOX" : "pairs");
    rope_case c = rope_data(nh, hd, nd, T, neox);
    tsr_opl_builder b = new_builder();
    tsr_rec r;
    ENC(tsr_k_rope(W->kenv, c.d, &r, &_why));
    if (!emit(b, r, 0)) return;
    run_list(b, what);
    CHECK(rope_direct(c, neox) == 0, "%s: direct reference", what);
    const size_t bad = cmp_b((uint64_t) (c.d.dst.h & 0x0FFFFFFFu) * 64, T, (uint64_t) nh * hd / 16u * 64, (uint64_t) c.d.dst.stride * 64);
    CHECK(bad == 0, "%s: %zu rows differ from the direct reference", what, bad);
}

static void test_rope_kinds() {
    printf("ROPE\n");
    for (uint32_t T : TS) {
        test_rope(2, 64, 64, T, true);
        test_rope(14, 64, 64, T, true);
        test_rope(4, 128, 128, T, true);
        test_rope(2, 64, 32, T, true);       // the tail of every head is copied
        test_rope(2, 64, 64, T, false);
    }
}

// ---------------------------------------------------------------------------------------------
// SET_ROWS_KV + FLASH_ATTN
// ---------------------------------------------------------------------------------------------
enum { MK_CAUSAL, MK_SLIDING, MK_RANDOM, MK_NONE };

struct fa_params {
    uint32_t n_head, n_head_kv, D, T;
    uint32_t kv_size;         // cells of a plane (= n_kv)
    bool history;             // random rows in the cells the new tokens do not write
    bool wrap;                // the new tokens write the first cells of the plane (history behind them)
    int mask;
    bool fa_new = true;       // give the FLASH_ATTN the FA_NEW map
};

struct fa_out {
    uint64_t att_off = 0;
    uint32_t token_lines = 0;
    uint32_t k_plane = 0, v_plane = 0;
    tsr_k_kv kv;
    size_t mism_direct = 0;
    size_t mism_exec = 0;
    bool built = false;
};

static fa_out run_fa(const fa_params & p, const char * what, bool check_direct = true) {
    fa_out o;
    const uint32_t E = p.n_head_kv * p.D, G = p.n_head / p.n_head_kv, T = p.T, C = W->C;
    tsr_k_kv kv;
    std::string why;
    if (!tsr_k_kv_geom(E, p.kv_size, C, &kv, &why)) {
        CHECK(false, "%s: %s", what, why.c_str());
        return o;
    }
    const uint32_t kp = (uint32_t) (W->alloc_heap(kv.bytes) / 64), vp = (uint32_t) (W->alloc_heap(kv.bytes) / 64);
    // the new cells and the planes' initial contents (history: A32 + KVST, the device image)
    std::vector<uint32_t> cells(T);
    for (uint32_t t = 0; t < T; ++t) cells[t] = p.wrap ? t : (p.history ? p.kv_size - T + t : t);
    std::vector<std::vector<uint8_t>> kb(C, std::vector<uint8_t>(kv.bytes, 0)), vb(C, std::vector<uint8_t>(kv.bytes, 0));
    std::vector<uint8_t *> kpp(C), vpp(C);
    for (uint32_t c = 0; c < C; ++c) { kpp[c] = kb[c].data(); vpp[c] = vb[c].data(); }
    const tsr_kv_geom geo = { E, C, kv.LG };
    std::vector<bool> is_new(p.kv_size, false);
    for (uint32_t c : cells) is_new[c] = true;
    if (p.history) {
        std::vector<uint8_t> qa((size_t) tsr_qa_lines(TSR_QA_A32, E) * 64);
        std::vector<float> h(E);
        for (uint32_t j = 0; j < p.kv_size; ++j) {
            if (is_new[j]) continue;
            for (uint32_t w = 0; w < 2; ++w) {
                for (auto & v : h) v = rnd_n(1.5f);
                tsr_qact(TSR_QA_A32, h.data(), E, qa.data());
                tsr_ref_kvst(w, &geo, w ? vpp.data() : kpp.data(), j, 1u, qa.data(), 0u, nullptr);
            }
        }
    }
    for (uint32_t c = 0; c < C; ++c) {
        W->put(c, (uint64_t) kp * 64, kb[c].data(), kb[c].size());
        W->put(c, (uint64_t) vp * 64, vb[c].data(), vb[c].size());
    }
    // activations
    const uint32_t ql = p.n_head * p.D / 16u, el = E / 16u;
    const tsr_k_rows q = put_rows(W->alloc_io((uint64_t) T * ql * 64), rnd_vec((size_t) T * p.n_head * p.D, 2.0f), T, p.n_head * p.D, ql);
    const tsr_k_rows kr = put_rows(W->alloc_io((uint64_t) T * el * 64), rnd_vec((size_t) T * E, 2.0f), T, E, el);
    const tsr_k_rows vr = put_rows(W->alloc_io((uint64_t) T * el * 64), rnd_vec((size_t) T * E, 2.0f), T, E, el);
    const tsr_k_rows att = new_out(T, ql);
    std::vector<uint64_t> idx64(((T + 7u) / 8u) * 8u, 0);
    for (uint32_t t = 0; t < T; ++t) idx64[t] = cells[t];
    const uint64_t idx_off = W->alloc_io(idx64.size() * 8);
    W->put(0, idx_off, idx64.data(), idx64.size() * 8);
    // the mask
    const uint32_t n_kv = p.kv_size;
    std::vector<uint16_t> m16((size_t) T * n_kv, 0xFC00u);
    std::vector<int> tok_of(n_kv, -1);
    for (uint32_t t = 0; t < T; ++t) tok_of[cells[t]] = (int) t;
    for (uint32_t t = 0; t < T; ++t) {
        for (uint32_t j = 0; j < n_kv; ++j) {
            bool vis = false;
            if (tok_of[j] >= 0) vis = (uint32_t) tok_of[j] <= t;
            else vis = p.history;
            if (p.mask == MK_NONE) vis = tok_of[j] >= 0 || p.history;
            if (p.mask == MK_SLIDING && vis && tok_of[j] >= 0 && (uint32_t) tok_of[j] + 3u <= t) vis = false;
            if (p.mask == MK_RANDOM && vis && tok_of[j] != (int) t && rnd_u(10u) < 3u) vis = false;
            if (vis) m16[(size_t) t * n_kv + j] = 0u;
        }
    }
    const float scale = 1.0f / std::sqrt((float) p.D);
    const uint64_t f32_off = W->alloc_io((uint64_t) T * tsr_k_lines(n_kv) * 64);
    const uint32_t f32_h = H0(f32_off);
    tsr_k_fa_mask m;
    const bool plain_causal = !p.history && !p.wrap && p.mask == MK_CAUSAL;
    if (!tsr_k_fa_mask_build(W->kenv, E, p.D, G, scale, T, n_kv, f32_h, plain_causal ? nullptr : m16.data(), n_kv, 0, &m, &why)) {
        CHECK(false, "%s: mask: %s", what, why.c_str());
        return o;
    }
    const uint64_t msum_off = W->alloc_io(m.msum.size() * 4);
    W->put(0, msum_off, m.msum.data(), m.msum.size() * 4);
    if (m.need_f32) W->put(0, f32_off, m.f32.data(), m.f32.size() * 4);

    tsr_kd_set_rows sk{}, sv{};
    sk.src = kr; sk.n_rows = T; sk.kv = kv; sk.which = opl::setrows::WHICH_K; sk.plane = kp; sk.idx_src = H0(idx_off);
    sv = sk; sv.src = vr; sv.which = opl::setrows::WHICH_V; sv.plane = vp;
    tsr_kd_flash_attn fa{};
    fa.dst = att; fa.q = q; fa.n_kv = n_kv; fa.D = p.D; fa.n_head = p.n_head; fa.n_head_kv = p.n_head_kv; fa.T = T; fa.scale = scale;
    fa.kv = kv; fa.k_plane = kp; fa.v_plane = vp; fa.msum_h = H0(msum_off); fa.mask_h = m.need_f32 ? f32_h : opl::HANDLE_NONE;
    tsr_rec rk, rv, rf;
    if (!tsr_k_set_rows_kv(W->kenv, sk, &rk, &why) || !tsr_k_set_rows_kv(W->kenv, sv, &rv, &why) ||
        !tsr_k_flash_attn(W->kenv, fa, &rf, &why)) {
        CHECK(false, "%s: encoder refused: %s", what, why.c_str());
        return o;
    }
    tsr_opl_builder b = new_builder();
    tsr_fanew_map fm, none;
    if (!emit(b, rk, 0, &fm) || !emit(b, rv, 1, &fm) || !emit(b, rf, 2, p.fa_new ? &fm : &none)) return o;
    // tsr_encode put FA_NEW into the FLASH_ATTN record exactly when it was given the map of the SET_ROWS_KV records
    const uint32_t fl = rf.args[opl::fa::FLAGS];
    CHECK(p.fa_new ? ((fl & tsr::plan::attn::FA_NEW) != 0u && tsr::plan::attn::fa_n_new(fl) == T) : fl == 0u,
          "%s: FLASH_ATTN flags word 0x%08x", what, fl);
    const run_result rr = run_list(b, what);
    o.mism_exec = rr.mism_exec;
    o.built = true;
    o.att_off = (uint64_t) (att.h & 0x0FFFFFFFu) * 64;
    o.token_lines = ql;
    o.k_plane = kp; o.v_plane = vp; o.kv = kv;
    if (!check_direct) return o;

    // (b) SET_ROWS_KV x 2 and FLASH_ATTN with words written here; the preload area by hand: indices at 0, the summary behind them
    const uint32_t msum_pre = (uint32_t) ((idx64.size() * 8 + 63u) / 64u * 64u);
    std::vector<uint8_t> pre(msum_pre + (m.msum.size() * 4 + 63u) / 64u * 64u, 0);
    memcpy(pre.data(), idx64.data(), (size_t) T * 8);
    memcpy(pre.data() + msum_pre, m.msum.data(), m.msum.size() * 4);
    const uint32_t geom = opl::geom_word(E / 64u, kv.LG);
    const uint32_t ska[12] = { H0((uint64_t) kp * 64), 0u, kr.h, kr.stride, 0u, T, E, opl::setrows::WHICH_K, kp, geom, 0u, 0u };
    const uint32_t sva[12] = { H0((uint64_t) vp * 64), 0u, vr.h, vr.stride, 0u, T, E, opl::setrows::WHICH_V, vp, geom, 0u, 0u };
    const tsr::ref::dram_view vb2 = W->view(1);
    std::vector<uint8_t *> kpB(C), vpB(C);
    std::vector<const uint8_t *> kpC(C), vpC(C);
    for (uint32_t c = 0; c < C; ++c) {
        kpB[c] = W->img[1][c] + (uint64_t) kp * 64;
        vpB[c] = W->img[1][c] + (uint64_t) vp * 64;
        kpC[c] = kpB[c];
        vpC[c] = vpB[c];
    }
    CHECK(tsr::ref::set_rows_kv_ref(vb2, ska, pre.data(), pre.size(), kpB.data(), C) == 0, "%s: direct SET_ROWS_KV K", what);
    CHECK(tsr::ref::set_rows_kv_ref(vb2, sva, pre.data(), pre.size(), vpB.data(), C) == 0, "%s: direct SET_ROWS_KV V", what);
    const uint32_t faa[22] = { att.h, att.stride, q.h, p.D / 16u, ql, H0((uint64_t) kp * 64), H0((uint64_t) vp * 64), n_kv, p.D, p.n_head,
                               p.n_head_kv, T, msum_pre, m.need_f32 ? f32_h : opl::HANDLE_NONE, f2b(scale), 0u, kp, vp, geom, p.D, G,
                               tsr::plan::attn::fa_flags(0u, T) };
    CHECK(tsr::ref::flash_attn_ref(vb2, faa, pre.data(), pre.size(), kpC.data(), vpC.data(), W->kenv.att) == 0, "%s: direct FLASH_ATTN", what);
    o.mism_direct = cmp_b(o.att_off, T, (uint64_t) ql * 64, (uint64_t) ql * 64);
    for (uint32_t c = 0; c < C; ++c) {
        o.mism_direct += memcmp(W->dev_read(c, (uint64_t) kp * 64, kv.bytes).data(), W->img[1][c] + (uint64_t) kp * 64, kv.bytes) != 0;
        o.mism_direct += memcmp(W->dev_read(c, (uint64_t) vp * 64, kv.bytes).data(), W->img[1][c] + (uint64_t) vp * 64, kv.bytes) != 0;
    }
    return o;
}

static void test_fa(uint32_t nh, uint32_t nkv, uint32_t D, uint32_t T, uint32_t kv_size, bool history, bool wrap, int mask) {
    char what[160];
    snprintf(what, sizeof(what), "SET_ROWS_KV + FLASH_ATTN %u / %u heads of %u T %u cells %u hist %d wrap %d mask %d", nh, nkv, D, T, kv_size,
             (int) history, (int) wrap, mask);
    fa_params p{ nh, nkv, D, T, kv_size, history, wrap, mask };
    const fa_out o = run_fa(p, what);
    CHECK(o.built && o.mism_direct == 0, "%s: %zu outputs / planes differ from the direct references", what, o.mism_direct);
}

static void test_fa_kinds() {
    printf("SET_ROWS_KV + FLASH_ATTN\n");
    for (uint32_t T : TS) {
        test_fa(2, 1, 64, T, T, false, false, MK_CAUSAL);               // the graph runtime's form: cells 0 .. T-1, n_kv = T
        test_fa(14, 2, 64, T, T, false, false, MK_CAUSAL);
        test_fa(4, 2, 128, T, T, false, false, MK_CAUSAL);
        test_fa(2, 1, 64, T, T, false, false, MK_NONE);
        test_fa(2, 1, 64, T, T + 24, true, false, MK_CAUSAL);           // cells behind the new ones
        test_fa(2, 1, 64, T, T + 40, true, true, MK_CAUSAL);            // the new cells before the old ones
        test_fa(4, 2, 64, T, T + 24, true, false, MK_SLIDING);
        test_fa(4, 2, 64, T, T + 24, true, false, MK_RANDOM);           // holes: the GENERAL path with the f32 copy
        test_fa(2, 1, 64, T, 1500, true, true, MK_CAUSAL);              // several chunks, the new cells first in the plane
        test_fa(4, 2, 128, T, 700, true, false, MK_CAUSAL);             // several chunks, the new cells last
    }
}

// ---------------------------------------------------------------------------------------------
// One decoder layer of Qwen2.5-0.5B's shape (test_fw_m7_layer.cpp build(), the layer without ggml)
// ---------------------------------------------------------------------------------------------
static void test_layer(uint32_t T) {
    char what[64];
    snprintf(what, sizeof(what), "Qwen2.5-0.5B layer T %u", T);
    const uint32_t n_embd = 896, n_head = 14, n_kv_head = 2, D = 64, n_ff = 4864, E = n_kv_head * D, Qd = n_head * D;
    const float eps = 1e-6f, scale = 0.125f;
    const uint32_t xs = tsr_k_lines(n_embd), qs = tsr_k_lines(Qd), ks = tsr_k_lines(E), fs = tsr_k_lines(n_ff);
    const weight_t wq = make_weight(TSR_FMT_Q8_0, Qd, n_embd), wk = make_weight(TSR_FMT_Q8_0, E, n_embd), wv = make_weight(TSR_FMT_Q8_0, E, n_embd);
    const weight_t wo = make_weight(TSR_FMT_Q8_0, n_embd, Qd), wg = make_weight(TSR_FMT_Q8_0, n_ff, n_embd), wu = make_weight(TSR_FMT_Q8_0, n_ff, n_embd);
    const weight_t wd = make_weight(TSR_FMT_Q8_0, n_embd, n_ff);
    auto aux_row = [&](uint32_t n, float sc, float mean) {
        std::vector<float> v(n);
        for (auto & x : v) x = mean + rnd(sc);
        return put_rows(W->alloc_aux((uint64_t) tsr_k_lines(n) * 64), v, 1, n, tsr_k_lines(n)).h;
    };
    const uint32_t h_an = aux_row(n_embd, 0.3f, 1.0f), h_fn = aux_row(n_embd, 0.3f, 1.0f);
    const uint32_t h_bq = aux_row(Qd, 0.5f, 0.0f), h_bk = aux_row(E, 0.5f, 0.0f), h_bv = aux_row(E, 0.5f, 0.0f);
    // RoPE: positions and the angle table
    std::vector<float> pf(((T + 15u) / 16u) * 16u, 0.0f);
    std::vector<int32_t> pos(T);
    for (uint32_t t = 0; t < T; ++t) pos[t] = (int32_t) t;
    memcpy(pf.data(), pos.data(), (size_t) T * 4);
    const uint32_t pos_h = put_rows(W->alloc_io(pf.size() * 4), pf, 1, (uint32_t) pf.size(), (uint32_t) pf.size() / 16u).h;
    std::vector<float> tab(D / 2u);
    for (uint32_t i = 0; i < D / 2u; ++i) tab[i] = (float) std::pow(1000000.0, -2.0 * (double) i / (double) D);
    const uint32_t tab_h = put_rows(W->alloc_aux(256), tab, 1, D / 2u, tsr_k_lines(D / 2u)).h;
    // KV planes (cells 0 .. T-1), indices, causal mask summary
    tsr_k_kv kv;
    std::string why;
    CHECK(tsr_k_kv_geom(E, T, W->C, &kv, &why), "%s", why.c_str());
    const uint32_t kp = (uint32_t) (W->alloc_heap(kv.bytes) / 64), vp = (uint32_t) (W->alloc_heap(kv.bytes) / 64);
    {
        const std::vector<uint8_t> zero(kv.bytes, 0);
        for (uint32_t c = 0; c < W->C; ++c) {
            W->put(c, (uint64_t) kp * 64, zero.data(), zero.size());
            W->put(c, (uint64_t) vp * 64, zero.data(), zero.size());
        }
    }
    std::vector<uint64_t> idx64(((T + 7u) / 8u) * 8u, 0);
    for (uint32_t t = 0; t < T; ++t) idx64[t] = t;
    const uint64_t idx_off = W->alloc_io(idx64.size() * 8);
    W->put(0, idx_off, idx64.data(), idx64.size() * 8);
    const uint32_t f32_h = H0(W->alloc_io((uint64_t) T * tsr_k_lines(T) * 64));
    tsr_k_fa_mask m;
    CHECK(tsr_k_fa_mask_build(W->kenv, E, D, n_head / n_kv_head, scale, T, T, f32_h, nullptr, 0, 0, &m, &why), "%s", why.c_str());
    const uint64_t msum_off = W->alloc_io(m.msum.size() * 4);
    W->put(0, msum_off, m.msum.data(), m.msum.size() * 4);
    if (m.need_f32) W->put(0, (uint64_t) (f32_h & 0x0FFFFFFFu) * 64, m.f32.data(), m.f32.size() * 4);

    const tsr_k_rows X = put_rows(W->alloc_io((uint64_t) T * xs * 64), rnd_vec((size_t) T * n_embd, 1.0f), T, n_embd, xs);
    const tsr_k_rows Xn = new_out(T, xs), Q = new_out(T, qs), K = new_out(T, ks), V = new_out(T, ks), Qr = new_out(T, qs), Kr = new_out(T, ks);
    const tsr_k_rows Att = new_out(T, qs), Hh = new_out(T, xs), Hn = new_out(T, xs), G = new_out(T, fs), U = new_out(T, fs), GL = new_out(T, fs), Y = new_out(T, xs);

    tsr_opl_builder b = new_builder();
    tsr_fanew_map fm;
    int32_t tag = 0;
    const bool defer_ok = true;
    auto mm = [&](const weight_t & w, tsr_k_rows x, tsr_k_rows y, uint32_t addh, uint32_t adds, bool * ok) {
        tsr_kd_mm d{};
        d.x = x; d.y = y; d.add = { addh, adds }; d.w_off = (uint32_t) (w.w_off / 64); d.fmt = w.fmt; d.N = w.N; d.K = w.K; d.T = T;
        tsr_rec r;
        if (!tsr_k_mul_mat(W->kenv, d, &r, &why)) { CHECK(false, "%s: mul_mat: %s", what, why.c_str()); *ok = false; return; }
        const bool defer = defer_ok && tsr_k_mm_deferrable(W->kenv, r);
        b.set_defer(defer);
        if (!emit(b, r, tag++, nullptr, defer)) *ok = false;
    };
    auto push = [&](bool ok, tsr_rec & r, bool * all) {
        if (!ok) { CHECK(false, "%s: %s", what, why.c_str()); *all = false; return; }
        if (!emit(b, r, tag++, &fm)) *all = false;
    };
    bool ok = true;
    {
        tsr_kd_rms d{}; d.dst = Xn; d.src = X; d.n = n_embd; d.rows = T; d.eps = eps; d.weight = h_an;
        tsr_rec r; push(tsr_k_rms_norm(W->kenv, d, &r, &why), r, &ok);
    }
    mm(wq, Xn, Q, h_bq, 0, &ok);
    mm(wk, Xn, K, h_bk, 0, &ok);
    mm(wv, Xn, V, h_bv, 0, &ok);
    for (int w = 0; w < 2; ++w) {
        tsr_kd_rope d{}; d.dst = w ? Kr : Qr; d.src = w ? K : Q; d.n_heads = w ? n_kv_head : n_head; d.head_dim = D; d.n_dims = D; d.T = T;
        d.pos_src = pos_h; d.table = tab_h; d.neox = true;
        tsr_rec r; push(tsr_k_rope(W->kenv, d, &r, &why), r, &ok);
    }
    {
        tsr_kd_set_rows sk{}; sk.src = Kr; sk.n_rows = T; sk.kv = kv; sk.which = opl::setrows::WHICH_K; sk.plane = kp; sk.idx_src = H0(idx_off);
        tsr_kd_set_rows sv = sk; sv.src = V; sv.which = opl::setrows::WHICH_V; sv.plane = vp;
        tsr_rec r1, r2;
        push(tsr_k_set_rows_kv(W->kenv, sk, &r1, &why), r1, &ok);
        push(tsr_k_set_rows_kv(W->kenv, sv, &r2, &why), r2, &ok);
    }
    {
        tsr_kd_flash_attn fa{}; fa.dst = Att; fa.q = Qr; fa.n_kv = T; fa.D = D; fa.n_head = n_head; fa.n_head_kv = n_kv_head; fa.T = T; fa.scale = scale;
        fa.kv = kv; fa.k_plane = kp; fa.v_plane = vp; fa.msum_h = H0(msum_off); fa.mask_h = m.need_f32 ? f32_h : opl::HANDLE_NONE;
        tsr_rec r; push(tsr_k_flash_attn(W->kenv, fa, &r, &why), r, &ok);
    }
    mm(wo, Att, Hh, X.h, X.stride, &ok);                       // attn_out + the residual
    {
        tsr_kd_rms d{}; d.dst = Hn; d.src = Hh; d.n = n_embd; d.rows = T; d.eps = eps; d.weight = h_fn;
        tsr_rec r; push(tsr_k_rms_norm(W->kenv, d, &r, &why), r, &ok);
    }
    mm(wg, Hn, G, opl::HANDLE_NONE, 0, &ok);
    mm(wu, Hn, U, opl::HANDLE_NONE, 0, &ok);
    {
        tsr_kd_glu d{}; d.dst = GL; d.gate = G; d.up = U; d.n = n_ff; d.rows = T; d.variant = opl::glu::SWIGLU;
        tsr_rec r; push(tsr_k_glu(W->kenv, d, &r, &why), r, &ok);
    }
    mm(wd, GL, Y, Hh.h, Hh.stride, &ok);                       // ffn_out + the residual
    if (!ok) return;
    const run_result rr = run_list(b, what);
    printf("  %s: %u records, %u JOINs, %u MUL_MAT_BEGIN\n", what, rr.records, rr.joins, rr.begins);
    // the layer's output must not be the untouched pattern (a layer that wrote nothing would also equal opl_exec's image)
    const std::vector<uint8_t> yb = W->dev_read(0, (uint64_t) (Y.h & 0x0FFFFFFFu) * 64, (size_t) xs * 64);
    size_t pat = 0;
    for (uint8_t c : yb) pat += c == 0xA5;
    CHECK(pat < yb.size() / 2, "%s: the output rows are still the pattern", what);
}

// ---------------------------------------------------------------------------------------------
// Negative controls: the comparison against the independent reference must fail
// ---------------------------------------------------------------------------------------------
static void neg_inv_n() {
    const uint32_t n = 896, T = 5;
    const tsr_kd_rms d = rms_data(n, T, true, 1e-6f);
    tsr_opl_builder b = new_builder();
    tsr_rec r;
    ENC(tsr_k_rms_norm(W->kenv, d, &r, &_why));
    r.args[opl::rms::INV_N] += 1u;                       // one ulp off 1 / n
    if (!emit(b, r, 0)) return;
    const run_result rr = run_list(b, "negative control: wrong inv_n");
    CHECK(rr.mism_exec == 0, "negative control: the device and opl_exec agree on the mutated record");
    CHECK(rms_direct(d, f2b(1.0f / (float) n)) == 0, "direct reference");
    const size_t bad = cmp_b((uint64_t) (d.dst.h & 0x0FFFFFFFu) * 64, T, (uint64_t) tsr_k_lines(n) * 64, (uint64_t) d.dst.stride * 64);
    CHECK(bad > 0, "negative control: a wrong inv_n was not caught by the comparison with the direct reference");
    printf("  wrong inv_n: %zu of %u rows differ from the direct reference\n", bad, T);
}

static void neg_pairing() {
    rope_case c = rope_data(14, 64, 64, 5, false);      // the encoder gets the pairs (2 i, 2 i + 1) ...
    tsr_opl_builder b = new_builder();
    tsr_rec r;
    ENC(tsr_k_rope(W->kenv, c.d, &r, &_why));
    if (!emit(b, r, 0)) return;
    const run_result rr = run_list(b, "negative control: non-NEOX pairing");
    CHECK(rr.mism_exec == 0, "negative control: the device and opl_exec agree on the other pairing");
    CHECK(rope_direct(c, true) == 0, "direct reference");  // ... the direct reference rotates (i, i + n_dims / 2)
    const size_t bad = cmp_b((uint64_t) (c.d.dst.h & 0x0FFFFFFFu) * 64, 5, (uint64_t) 14 * 64 / 16u * 64, (uint64_t) c.d.dst.stride * 64);
    CHECK(bad > 0, "negative control: the non-NEOX pairing was not caught by the comparison with the NEOX reference");
    printf("  non-NEOX pairing: %zu of 5 rows differ from the NEOX reference\n", bad);
}

static void neg_fa_new() {
    // New cells before many old ones (the plane wrapped, more than one chunk of groups): the planner puts the chunks of the new cells
    // last when it knows them (FA_NEW). Whether that changes the bits depends on the device: with T_MAX 1 the merge order of the
    // chunks shows in the result; with T_MAX >= 2 the tiles are GENERAL or one FAST range and the merge does not depend on the order
    // (the explored shapes: every mask kind, wrapped or not, 600 to 4000 cells, T up to 40, all equal). The comparison is required to
    // fail where the device is order-sensitive; everywhere the record's FA_NEW flag is checked (run_fa: the flags word).
    size_t bad_max = 0;
    for (uint32_t T : { 5u, 8u, 17u }) {
        for (uint32_t cells : { 600u, 1500u }) {
            fa_params p{ 2, 1, 64, T, cells, true, true, MK_CAUSAL, false };
            const fa_out o = run_fa(p, "negative control: no FA_NEW");
            CHECK(o.built && o.mism_exec == 0, "negative control: the device and opl_exec agree without FA_NEW");
            bad_max = std::max(bad_max, o.mism_direct);
            printf("  no FA_NEW, T %u, %u cells: %zu outputs / planes differ from the reference with FA_NEW\n", T, cells, o.mism_direct);
        }
    }
    if (W->cfg.t_max == 1) {
        CHECK(bad_max > 0, "negative control: a missing FA_NEW was not caught by the comparison with the direct reference");
    } else {
        printf("  (T_MAX %u: the FLASH_ATTN result does not depend on the chunk order; the FA_NEW flag word is checked in every case)\n",
               W->cfg.t_max);
    }
    fa_params p{ 2, 1, 64, 17, 1500, true, true, MK_CAUSAL, true };
    const fa_out o = run_fa(p, "control: FA_NEW");
    CHECK(o.built && o.mism_direct == 0 && o.mism_exec == 0, "control: with FA_NEW the wrapped plane equals the direct reference");
}

// ---------------------------------------------------------------------------------------------
// Refusals
// ---------------------------------------------------------------------------------------------
static void test_refusals() {
    printf("refusals\n");
    std::string why;
    tsr_rec r;
    tsr_kd_unary u{};
    u.src = { H0(W->alloc_io(256)), 1 };
    u.dst = { H0(W->alloc_io(256)), 1 };
    u.n = 64; u.rows = 1; u.func = opl::unary::SILU;
    CHECK(tsr_k_unary(W->kenv, u, &r, &why), "a valid UNARY: %s", why.c_str());
    tsr_kenv e2 = W->kenv;
    e2.kinds &= ~(1ull << opl::K_UNARY);
    CHECK(!tsr_k_unary(e2, u, &r, &why) && why.find("lacks the UNARY") != std::string::npos, "a kind the image lacks: %s", why.c_str());
    CHECK(r.n_args == 0, "a refused record is empty");
    u.func = 9;
    CHECK(!tsr_k_unary(W->kenv, u, &r, &why), "UNARY func 9");
    tsr_kd_mm m{};
    m.x = { H0(W->alloc_io(256)), 4 }; m.y = { H0(W->alloc_io(256)), 4 }; m.N = 64; m.K = 100; m.T = 1;   // K not a multiple of 32
    CHECK(!tsr_k_mul_mat(W->kenv, m, &r, &why), "MUL_MAT K = 100");
    m.K = 64; m.add = { H0(W->alloc_io(256)), 0 }; m.acc_dst = true;
    CHECK(!tsr_k_mul_mat(W->kenv, m, &r, &why), "MUL_MAT with an addend and ACC_DST");
    tsr_kd_rope ro{};
    ro.src = { H0(W->alloc_io(1024)), 4 }; ro.dst = { H0(W->alloc_io(1024)), 4 }; ro.n_heads = 1; ro.head_dim = 64; ro.n_dims = 48; ro.T = 1;
    ro.pos_src = H0(W->alloc_io(64)); ro.table = H0(W->alloc_aux(64));
    CHECK(!tsr_k_rope(W->kenv, ro, &r, &why), "NEOX ROPE with n_dims 48");
    tsr_k_kv kv;
    CHECK(!tsr_k_kv_geom(100, 16, W->C, &kv, &why), "KV plane with E = 100");
    CHECK(tsr_k_kv_geom(128, 16, W->C, &kv, &why) && kv.bytes > 0, "KV plane geometry");
    CHECK(!tsr_k_kv_geom(128, 16u * W->C * 65536u + 1u, W->C, &kv, &why) && why.find("does not fit") != std::string::npos,
          "KV plane with LG above 16 bits: %s", why.c_str());
    CHECK(tsr_k_kv_geom(128, 16u * W->C * 65535u, W->C, &kv, &why) && kv.LG == 65535u, "KV plane with LG 65535");
}

// ---------------------------------------------------------------------------------------------
int main() {
    W = new world;
    W->tp = tsr_transport_open(nullptr);
    if (W->tp == nullptr) {
        fprintf(stderr, "test-tsr-kinds: cannot open the transport (TSR_DEVICE=%s)\n", getenv("TSR_DEVICE") ? getenv("TSR_DEVICE") : "");
        return 1;
    }
    W->cfg = *tsr_transport_config(W->tp);
    std::string err;
    if (!W->dev.open(W->tp, &err)) {
        fprintf(stderr, "test-tsr-kinds: %s\n", err.c_str());
        return 1;
    }
    W->C = W->cfg.num_ch;
    const uint32_t cfg3 = tsr_transport_csr_read(W->tp, TSR_CSR_CFG3);
    const uint32_t pcap = ((cfg3 & TSR_CFG3_PCAP16) >> TSR_CFG3_PCAP16_LSB) * 16u;
    W->kenv = tsr_kenv_make(W->cfg, pcap, tsr_kinds_mask(W->dev.versions()), W->dev.image_path());
    W->xcfg.t_max = W->cfg.t_max;
    W->xcfg.pcap = pcap;
    W->xcfg.k_max = W->cfg.k_max;
    W->xcfg.mm_threads = 4;
    printf("device %s, image %s, C %u T_MAX %u K_MAX %u PCAP %u\n", tsr_transport_name(W->tp), W->dev.image_path(), W->C, W->cfg.t_max,
           W->cfg.k_max, pcap);
    if (W->C > TSR_NUM_CH_MAX) return 1;
    for (uint32_t w = 0; w < 2; ++w) {
        for (uint32_t c = 0; c < W->C; ++c) {
            void * p = mmap(nullptr, IMG_BYTES, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
            if (p == MAP_FAILED) {
                fprintf(stderr, "test-tsr-kinds: cannot map the host image\n");
                return 1;
            }
            W->img[w][c] = (uint8_t *) p;
        }
    }
    const char * only = getenv("TSR_KINDS_ONLY");
    auto want = [&](const char * n) { return only == nullptr || strstr(only, n) != nullptr; };
    if (want("mm")) test_mul_mats();
    if (want("ew")) test_scalar_kinds();
    if (want("rms")) test_rms_kinds();
    if (want("rope")) test_rope_kinds();
    if (want("fa")) test_fa_kinds();
    if (want("layer")) {
        printf("layer\n");
        for (uint32_t T : { 1u, 5u, 17u }) test_layer(T);
    }
    if (want("neg")) {
        printf("negative controls\n");
        neg_inv_n();
        neg_pairing();
        neg_fa_new();
    }
    if (want("refuse")) test_refusals();
    tsr_transport_close(W->tp);
    printf("%s (%d failures)\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
