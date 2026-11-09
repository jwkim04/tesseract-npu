// test-tsr-attn.cpp — ctest ggml_tesseract_attn (docs/design/06-software-verification.md §10.2, §10.4 M7
// gate item 4; docs/design/03-attention-kv.md §13.6): the KV cache and attention on the
// Tesseract backend through the public ggml API, against the CPU backend and the AT references, on the
// transport TSR_DEVICE selects (default sim).
//
//   1. KV buffers (SW §9.3 M7, SW-17, AT §4.6): a ggml context of cache_k_l<n> / cache_v_l<n> q8_0
//      tensors allocated on the device becomes a KV buffer at its clear (storage "kv"), planes
//      top-down (the first tensor's highest); set / get / memset of random byte ranges equal a host
//      mirror (partial blocks and rows, whole tensors, two streams); llama.cpp's state save / load
//      pattern (row ranges out, clear, row ranges back) restores every byte; views resolve to their
//      plane and cell; freeing the buffer gives the device memory back.
//   2. Mask summaries (AT §7.3): F16 masks named attn_inp_kq_mask (causal, SWA, a wrapped SWA ring,
//      holes, finite values, empty rows) written with ggml_backend_tensor_set; the MSUM the device
//      holds equals tsr_msum_build_f16 word for word, the f32 copy (when a tile reads it) equals
//      tsr_msum_mask_f32 with the FA's scale, get_alloc_size reserves both.
//   3. SET_ROWS into the cache + FLASH_ATTN_EXT reading it, the graph of llama.cpp's build_attn
//      (cpy_k / cpy_v, get_k / get_v views, permutes, FA with an F16 kq_mask and a scale, F32
//      precision), q8_0 K / V, GQA, D 64 / 128 / 256, decode (T = 1) and prefill (T > 1), causal,
//      SWA, a wrapped SWA ring (two runs per row; GENERAL tiles in prefill), holes and finite mask
//      values (GENERAL MSUM, AMASK with the f32 copy), on the Tesseract backend vs the CPU backend:
//        - every node is accepted by supports_op (the graph runs as one Tesseract graph_compute);
//        - the device cache after SET_ROWS equals the history plus quantize_row_q8_0_ref of the new
//          rows bit for bit (QACT A32, AT §6.1), and the CPU's up to its rare x86 rounding ties;
//        - the attention output equals tsr_ref_fa (AT §6.6, the bit-exact oracle) run on the device's
//          own cache, MSUM, f32 copy and new cells, bit for bit;
//        - NMSE against the CPU backend <= 5e-4 (AT §13.2: the test-backend-ops tolerance) and, with
//          masks of +-0 / -inf only, <= 1e-6 (the ties and accumulation order measure ~1e-12 to
//          1e-9; a one-cell error is far above it);
//      a wrapped ring of 2048 cells with the new cell in the low run and the older run over two
//      chunks makes the bits depend on FA_NEW (AT §6.6 rule 7); ggml_tesseract_attn_split1 runs the
//      test with TSR_OPL_SPLIT=1, so FA_NEW crosses a list cut (its own preload entry).
//   4. supports_op and placement: a CPY into a view whose rows end inside a line is refused,
//      one with rows of whole lines runs and leaves the viewed tensor's other bytes alone; an op
//      reading leaf data in host memory (an ANY buffer of leaves only, llama.cpp's control vectors) is
//      refused (SW §8 R6 as built: only MUL_MAT is staged).
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
#include "ggml-cpu.h"
#include "ggml-quants.h"      // quantize_row_q8_0_ref (ggml/src)
#include "ggml-tesseract.h"

#include "tsr_isa.h"
#include "tsr_ref.h"
#include "tsr_ref_at.h"

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

namespace {

ggml_backend_t g_tsr = nullptr;
ggml_backend_t g_cpu = nullptr;
std::mt19937_64 g_rng(20261031);

using storage_fn      = const char * (*) (ggml_backend_buffer_t);
using kv_plane_fn     = int (*) (const ggml_tensor *, uint32_t *);
using mask_summary_fn = size_t (*) (const ggml_tensor *, uint32_t *, size_t, float *, size_t);
using att_cfg_fn      = int (*) (uint32_t *);
storage_fn      h_storage = nullptr;
kv_plane_fn     h_kv_plane = nullptr;
mask_summary_fn h_mask = nullptr;
att_cfg_fn      h_att_cfg = nullptr;

constexpr uint16_t NINF = 0xFC00u;

float urand(float lo, float hi) {
    return lo + (hi - lo) * (float) ((g_rng() >> 11) * (1.0 / 9007199254740992.0));
}

ggml_context * new_ctx(size_t n_tensors) {
    ggml_init_params p = { ggml_tensor_overhead() * n_tensors + ggml_graph_overhead() * 2, nullptr, true };
    return ggml_init(p);
}

double nmse(const std::vector<float> & a, const std::vector<float> & b) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        num += ((double) a[i] - b[i]) * ((double) a[i] - b[i]);
        den += (double) b[i] * b[i];
    }
    return den == 0 ? num : num / den;
}

// ggml q8_0 rows of random f32 rows (the reference rounding the device's QACT A32 uses, AT §6.1).
std::vector<uint8_t> q8_rows(const float * x, size_t rows, uint32_t E) {
    std::vector<uint8_t> out(rows * (E / 32) * sizeof(block_q8_0));
    for (size_t r = 0; r < rows; ++r) {
        quantize_row_q8_0_ref(x + r * E, (block_q8_0 *) (out.data() + r * (E / 32) * sizeof(block_q8_0)), E);
    }
    return out;
}

// =================================================================================================
// 1. KV buffers through the ggml API
// =================================================================================================
void test_kv_buffer() {
    printf("kv buffer\n");
    size_t free0 = 0, total0 = 0;
    ggml_backend_dev_memory(ggml_backend_get_device(g_tsr), &free0, &total0);

    ggml_context * ctx = new_ctx(8);
    struct spec { const char * name; int64_t E, kv, ns; };
    const spec sp[] = { { "cache_k_l0", 128, 300, 1 }, { "cache_v_l0", 128, 300, 1 },
                        { "cache_k_l1", 256, 200, 2 }, { "cache_v_l1", 256, 200, 2 } };
    std::vector<ggml_tensor *> t;
    for (const spec & s : sp) {
        ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_Q8_0, s.E, s.kv, s.ns);
        ggml_set_name(x, s.name);
        t.push_back(x);
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, g_tsr);
    CHECK(buf != nullptr, "alloc");
    CHECK(strcmp(h_storage(buf), "undecided") == 0, "storage before the clear: %s", h_storage(buf));
    ggml_backend_buffer_clear(buf, 0);
    CHECK(strcmp(h_storage(buf), "kv") == 0, "storage after the clear: %s", h_storage(buf));

    // planes: top-down in tensor order, two per two-stream tensor
    uint32_t prev = UINT32_MAX;
    for (size_t i = 0; i < t.size(); ++i) {
        for (int64_t s = 0; s < sp[i].ns; ++s) {
            ggml_tensor * v = ggml_view_2d(ctx, t[i], sp[i].E, sp[i].kv, t[i]->nb[1], (size_t) s * t[i]->nb[2]);
            uint32_t info[9];
            CHECK(h_kv_plane(v, info) == 0, "plane of %s stream %lld", sp[i].name, (long long) s);
            CHECK(info[0] == (uint32_t) (i % 2) && info[1] == (uint32_t) s && info[2] == 0, "%s: which %u stream %u cell %u",
                  sp[i].name, info[0], info[1], info[2]);
            CHECK(info[3] < prev, "%s stream %lld: base beat 0x%x not below 0x%x", sp[i].name, (long long) s, info[3], prev);
            CHECK(info[5] == (uint32_t) sp[i].E && info[7] == (uint32_t) sp[i].kv && info[6] == tsr_kv_lg_min((uint32_t) sp[i].kv, info[8]),
                  "%s: E %u kv %u LG %u", sp[i].name, info[5], info[7], info[6]);
            prev = info[3];
        }
    }
    // a view at a cell offset
    ggml_tensor * vv = ggml_view_2d(ctx, t[2], 256, 10, t[2]->nb[1], t[2]->nb[2] + 37 * t[2]->nb[1]);
    uint32_t info[9];
    CHECK(h_kv_plane(vv, info) == 0 && info[1] == 1 && info[2] == 37, "view: stream %u cell %u", info[1], info[2]);

    // contents: zero after the clear, then random ranges against a mirror
    std::vector<std::vector<uint8_t>> mirror;
    for (ggml_tensor * x : t) {
        mirror.emplace_back(ggml_nbytes(x), 0);
        std::vector<uint8_t> got(ggml_nbytes(x), 0xEE);
        ggml_backend_tensor_get(x, got.data(), 0, got.size());
        CHECK(got == mirror.back(), "%s not zero after the clear", x->name);
    }
    for (size_t i = 0; i < t.size(); ++i) {
        for (auto & b : mirror[i]) b = (uint8_t) g_rng();
        ggml_backend_tensor_set(t[i], mirror[i].data(), 0, mirror[i].size());
    }
    int n_ops = 0;
    for (int op = 0; op < 300; ++op) {
        const size_t i = g_rng() % t.size();
        const size_t nb = mirror[i].size(), rb = t[i]->nb[1];
        size_t off = g_rng() % nb, n = 0;
        switch (g_rng() % 4) {
            case 0: n = 1 + g_rng() % std::min<size_t>(nb - off, 3 * rb); break;
            case 1: off = (g_rng() % (nb / rb)) * rb; n = std::min(nb - off, (1 + g_rng() % 50) * rb); break;
            case 2: n = nb - off; break;
            default: n = 1 + g_rng() % std::min<size_t>(nb - off, 40); break;
        }
        const int kind = (int) (g_rng() % 3);
        if (kind == 0) {
            std::vector<uint8_t> d(n);
            for (auto & b : d) b = (uint8_t) g_rng();
            ggml_backend_tensor_set(t[i], d.data(), off, n);
            memcpy(mirror[i].data() + off, d.data(), n);
        } else if (kind == 1) {
            const uint8_t v = (uint8_t) g_rng();
            ggml_backend_tensor_memset(t[i], v, off, n);
            memset(mirror[i].data() + off, v, n);
        } else {
            std::vector<uint8_t> got(n);
            ggml_backend_tensor_get(t[i], got.data(), off, n);
            CHECK(memcmp(got.data(), mirror[i].data() + off, n) == 0, "%s: get [%zu, +%zu) differs", t[i]->name, off, n);
        }
        ++n_ops;
    }
    // state save / load (llama_kv_cache::state_write_data / state_read_data: row ranges per tensor)
    std::vector<std::vector<uint8_t>> saved(t.size());
    for (size_t i = 0; i < t.size(); ++i) {
        const size_t rb = t[i]->nb[1], rows = mirror[i].size() / rb;
        saved[i].resize(mirror[i].size());
        for (size_t r0 = 0; r0 < rows;) {
            const size_t k = std::min(rows - r0, (size_t) (1 + g_rng() % 90));
            ggml_backend_tensor_get(t[i], saved[i].data() + r0 * rb, r0 * rb, k * rb);
            r0 += k;
        }
        CHECK(saved[i] == mirror[i], "%s: saved state differs", t[i]->name);
    }
    ggml_backend_buffer_clear(buf, 0);
    for (size_t i = 0; i < t.size(); ++i) {
        std::vector<uint8_t> got(mirror[i].size(), 0xEE);
        ggml_backend_tensor_get(t[i], got.data(), 0, got.size());
        CHECK(std::all_of(got.begin(), got.end(), [](uint8_t b) { return b == 0; }), "%s not zero after the clear", t[i]->name);
        const size_t rb = t[i]->nb[1], rows = mirror[i].size() / rb;
        for (size_t r0 = 0; r0 < rows;) {
            const size_t k = std::min(rows - r0, (size_t) (1 + g_rng() % 70));
            ggml_backend_tensor_set(t[i], saved[i].data() + r0 * rb, r0 * rb, k * rb);
            r0 += k;
        }
        ggml_backend_tensor_get(t[i], got.data(), 0, got.size());
        CHECK(got == mirror[i], "%s: state load differs", t[i]->name);
    }
    size_t free1 = 0, total1 = 0;
    ggml_backend_dev_memory(ggml_backend_get_device(g_tsr), &free1, &total1);
    ggml_backend_buffer_free(buf);
    size_t free2 = 0, total2 = 0;
    ggml_backend_dev_memory(ggml_backend_get_device(g_tsr), &free2, &total2);
    CHECK(free1 < free0 && free2 == free0, "device memory: free %zu -> %zu (planes) -> %zu (freed)", free0, free1, free2);
    printf("  %d random set / memset / get operations, state save / load of %zu tensors: %s; planes took %.2f MiB\n", n_ops,
           t.size(), g_fail ? "FAILED" : "ok", (free0 - free1) / 1048576.0);

    // a buffer with another tensor is not a KV buffer
    ggml_context * ctx2 = new_ctx(4);
    ggml_tensor * a = ggml_new_tensor_3d(ctx2, GGML_TYPE_Q8_0, 128, 64, 1);
    ggml_set_name(a, "cache_k_l0");
    ggml_tensor * b = ggml_new_tensor_1d(ctx2, GGML_TYPE_F32, 64);
    ggml_set_name(b, "other");
    ggml_backend_buffer_t buf2 = ggml_backend_alloc_ctx_tensors(ctx2, g_tsr);
    ggml_backend_buffer_clear(buf2, 0);
    CHECK(strcmp(h_storage(buf2), "kv") != 0, "a mixed buffer became %s", h_storage(buf2));
    ggml_backend_buffer_free(buf2);
    ggml_free(ctx2);
    ggml_free(ctx);
}

// =================================================================================================
// Masks
// =================================================================================================
enum mask_mode { M_CAUSAL, M_SWA, M_RING, M_HOLES, M_FINITE, M_EMPTY };
const char * mode_name(int m) {
    static const char * n[] = { "causal", "swa", "ring", "holes", "finite", "empty" };
    return n[m];
}

// Positions of the cells (-1: empty) and the mask of T query tokens at positions p0 .. p0+T-1, the
// way llama.cpp builds them (AT §2.4): a cell is visible to a token when it holds a position <= the
// token's and, with a window, within it. kv_size cells; the ring keeps position p in cell p mod
// kv_size (the iSWA cache); otherwise cell p holds position p.
struct mask_case {
    std::vector<int64_t> pos;      // per cell (n_kv cells)
    std::vector<uint16_t> m;       // [T][n_kv]
};
mask_case make_mask(int mode, uint32_t n_kv, uint32_t T, int64_t p0, int64_t swa) {
    mask_case c;
    c.pos.assign(n_kv, -1);
    const int64_t P = p0 + T - 1;                      // the last token
    for (uint32_t j = 0; j < n_kv; ++j) {
        if (mode == M_RING) {
            c.pos[j] = P - ((P - (int64_t) j) % (int64_t) n_kv + n_kv) % n_kv;
        } else if ((int64_t) j <= P) {
            c.pos[j] = j;
        }
    }
    c.m.assign((size_t) T * n_kv, NINF);
    for (uint32_t t = 0; t < T; ++t) {
        const int64_t pq = p0 + t;
        for (uint32_t j = 0; j < n_kv; ++j) {
            const int64_t pj = c.pos[j];
            bool vis = pj >= 0 && pj <= pq;
            if (mode == M_SWA || mode == M_RING) vis = vis && pq - pj < swa;
            if (mode == M_HOLES && (j * 7 + t) % 11 == 3) vis = false;
            if (mode == M_EMPTY && t == 0) vis = false;
            uint16_t h = vis ? 0x0000u : NINF;
            if (vis && mode == M_FINITE && (j % 5) == 1) h = ggml_fp32_to_fp16(urand(-2.0f, 0.5f));
            if (vis && (j % 13) == 4 && mode != M_FINITE) h = 0x8000u;   // -0 cells
            c.m[(size_t) t * n_kv + j] = h;
        }
    }
    return c;
}

// Copies the device must hold: the MSUM GENERAL, or a query tile of this reader planned GENERAL.
bool copy_needed(const tsr_att_cfg & cfg, const std::vector<uint32_t> & words, uint32_t E, uint32_t D, uint32_t G) {
    tsr_msum ms;
    tsr_msum_view(words.data(), words.size(), &ms);
    if (ms.flags & TSR_MSUM_GENERAL) return true;
    uint32_t GS, Tq;
    tsr_fa_pass_rule(&cfg, G, &GS, &Tq);
    std::vector<tsr_fa_chunk> ch(4096);
    for (uint32_t q0 = 0; q0 < ms.n_q; q0 += Tq) {
        const long n = tsr_fa_plan(&cfg, E, D, &ms, q0, std::min(Tq, ms.n_q - q0), nullptr, 0, ch.data(), ch.size());
        for (long k = 0; k < n && (size_t) k < ch.size(); ++k) {
            if (ch[(size_t) k].general) return true;
        }
    }
    return false;
}

// =================================================================================================
// 2. Mask summaries through the ggml API
// =================================================================================================
void test_mask_summaries(const tsr_att_cfg & cfg) {
    printf("mask summaries\n");
    int n = 0, n_general = 0, n_copies = 0;
    for (int it = 0; it < 36; ++it) {
        const int mode = it % 6;
        const uint32_t T = it % 3 == 0 ? 1u : (it % 3 == 1 ? 3u : 8u);
        const uint32_t n_kv = 256u * (1 + (uint32_t) (g_rng() % 3)) + (it % 4 == 3 ? 40u : 0u);
        const int64_t p0 = mode == M_RING ? (int64_t) (n_kv + g_rng() % (2 * n_kv)) : (int64_t) (g_rng() % (n_kv - T));
        const mask_case mc = make_mask(mode, n_kv, T, p0, 64 + (int64_t) (g_rng() % 100));
        const uint32_t D = 64u << (g_rng() % 3), nkvh = 1 + (uint32_t) (g_rng() % 4), G = 1 + (uint32_t) (g_rng() % 4);
        const float scale = it % 2 ? 1.0f / sqrtf((float) D) : 1.0f;

        ggml_context * cm = new_ctx(2);
        ggml_tensor * mask = ggml_new_tensor_4d(cm, GGML_TYPE_F16, n_kv, T, 1, 1);
        ggml_set_name(mask, "attn_inp_kq_mask");
        ggml_set_input(mask);
        ggml_backend_buffer_t mb = ggml_backend_alloc_ctx_tensors(cm, g_tsr);
        const size_t want_alloc = (ggml_nbytes(mask) + 63) / 64 * 64 + (4 * tsr_msum_words(T) + 63) / 64 * 64 +
                                  (size_t) T * 64 * ((n_kv + 15) / 16);
        CHECK(ggml_backend_buft_get_alloc_size(ggml_backend_get_default_buffer_type(g_tsr), mask) == want_alloc,
              "get_alloc_size %zu, want %zu", ggml_backend_buft_get_alloc_size(ggml_backend_get_default_buffer_type(g_tsr), mask),
              want_alloc);
        ggml_backend_tensor_set(mask, mc.m.data(), 0, ggml_nbytes(mask));
        CHECK(strcmp(h_storage(mb), "io") == 0, "mask buffer storage %s", h_storage(mb));

        ggml_context * cg = new_ctx(8);
        ggml_tensor * q = ggml_new_tensor_4d(cg, GGML_TYPE_F32, D, T, nkvh * G, 1);
        ggml_tensor * k = ggml_new_tensor_4d(cg, GGML_TYPE_Q8_0, D, n_kv, nkvh, 1);
        ggml_tensor * v = ggml_new_tensor_4d(cg, GGML_TYPE_Q8_0, D, n_kv, nkvh, 1);
        ggml_tensor * fa = ggml_flash_attn_ext(cg, q, k, v, mask, scale, 0.0f, 0.0f);

        std::vector<uint32_t> words(tsr_msum_words(T) + 8, 0xDEADBEEF);
        std::vector<float> copy((size_t) T * 16 * ((n_kv + 15) / 16), -1.0f);
        const size_t nw = h_mask(fa, words.data(), words.size(), copy.data(), copy.size());
        CHECK(nw == tsr_msum_words(T), "%s T=%u n_kv=%u: %zu MSUM words", mode_name(mode), T, n_kv, nw);
        words.resize(tsr_msum_words(T));
        std::vector<uint32_t> want(words.size());
        const uint32_t fl = tsr_msum_build_f16(mc.m.data(), n_kv, n_kv, T, words[4], want.data());
        if ((fl & TSR_MSUM_GENERAL) == 0 && (words[3] & TSR_MSUM_GENERAL)) {
            // promoted (a refused tile, AT §6.6 rule 5 / 6): the rows keep the runs
            want[3] = words[3];
            want[4] = words[4];
        }
        CHECK(words == want, "%s T=%u n_kv=%u: the device MSUM differs from tsr_msum_build_f16", mode_name(mode), T, n_kv);
        n_general += (words[3] & TSR_MSUM_GENERAL) != 0;
        if (copy_needed(cfg, want, D * nkvh, D, G)) {
            std::vector<float> wc(copy.size());
            tsr_msum_mask_f32(mc.m.data(), n_kv, n_kv, T, scale, wc.data());
            CHECK(memcmp(wc.data(), copy.data(), wc.size() * 4) == 0, "%s T=%u n_kv=%u scale %g: the f32 copy differs",
                  mode_name(mode), T, n_kv, scale);
            ++n_copies;
        }
        ++n;
        ggml_free(cg);
        ggml_backend_buffer_free(mb);
        ggml_free(cm);
    }
    printf("  %d masks: MSUM equal to tsr_msum_build_f16, %d GENERAL, %d f32 copies equal to tsr_msum_mask_f32\n", n, n_general,
           n_copies);
    // T_MAX 1: one-row tiles are FAST (AT §6.6 rule 2), so only GENERAL masks need a copy; with
    // larger T_MAX, rows with two runs or holes put prefill tiles on the GENERAL path (rule 3)
    CHECK(n_general > 0 && n_copies >= n_general && (cfg.t_max == 1 || n_copies > n_general),
          "coverage: %d GENERAL, %d copies", n_general, n_copies);
}

// =================================================================================================
// 3. SET_ROWS + FLASH_ATTN_EXT: device vs CPU vs tsr_ref_fa
// =================================================================================================
struct fa_case {
    const char * name;
    uint32_t D, nkvh, G;      // head dim, KV heads, query heads per KV head
    uint32_t kv_size, n_kv;   // cache cells, cells FA reads (llama.cpp pads to 256)
    uint32_t T;               // query tokens
    int mode;
    int64_t p0;               // position of the first new token
    int64_t swa;
    float scale;
    uint32_t steps;           // ubatches run on the same graph (the second and later reuse its op list)
};

// The attention graph of llama.cpp's build_attn with a KV cache (llm_graph_context::build_attn,
// build_attn_mha; llama_kv_cache::cpy_k / cpy_v / get_k / get_v; FA path, V not transposed).
struct attn_graph {
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_tensor * q_cur, * k_cur, * v_cur, * k_idxs, * v_idxs, * mask, * out, * fa, * sk, * sv;
};
attn_graph build_attn(const fa_case & c, ggml_tensor * cache_k, ggml_tensor * cache_v) {
    attn_graph g;
    g.ctx = new_ctx(64);
    const uint32_t n_head = c.nkvh * c.G, E = c.D * c.nkvh;
    g.q_cur = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F32, c.D, n_head, c.T);
    g.k_cur = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F32, c.D, c.nkvh, c.T);
    g.v_cur = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F32, c.D, c.nkvh, c.T);
    g.k_idxs = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I64, c.T);
    g.v_idxs = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I64, c.T);
    g.mask = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F16, c.n_kv, c.T, 1, 1);
    ggml_set_name(g.q_cur, "Qcur");
    ggml_set_name(g.k_cur, "Kcur");
    ggml_set_name(g.v_cur, "Vcur");
    ggml_set_name(g.k_idxs, "attn_inp_k_idxs");
    ggml_set_name(g.v_idxs, "attn_inp_v_idxs");
    ggml_set_name(g.mask, "attn_inp_kq_mask");
    for (ggml_tensor * x : { g.q_cur, g.k_cur, g.v_cur, g.k_idxs, g.v_idxs, g.mask }) ggml_set_input(x);

    g.sk = ggml_set_rows(g.ctx, cache_k, ggml_view_2d(g.ctx, g.k_cur, E, c.T, g.k_cur->nb[2], 0), g.k_idxs);
    g.sv = ggml_set_rows(g.ctx, cache_v, ggml_view_2d(g.ctx, g.v_cur, E, c.T, g.v_cur->nb[2], 0), g.v_idxs);
    ggml_tensor * k = ggml_view_4d(g.ctx, cache_k, c.D, c.nkvh, c.n_kv, 1, ggml_row_size(GGML_TYPE_Q8_0, c.D),
                                   ggml_row_size(GGML_TYPE_Q8_0, E), ggml_row_size(GGML_TYPE_Q8_0, (int64_t) E * c.kv_size), 0);
    ggml_tensor * v = ggml_view_4d(g.ctx, cache_v, c.D, c.nkvh, c.n_kv, 1, ggml_row_size(GGML_TYPE_Q8_0, c.D),
                                   ggml_row_size(GGML_TYPE_Q8_0, E), ggml_row_size(GGML_TYPE_Q8_0, (int64_t) E * c.kv_size), 0);
    ggml_tensor * q = ggml_view_4d(g.ctx, g.q_cur, c.D, n_head, c.T, 1, g.q_cur->nb[1], g.q_cur->nb[2], g.q_cur->nb[2] * c.T, 0);
    q = ggml_permute(g.ctx, q, 0, 2, 1, 3);
    k = ggml_permute(g.ctx, k, 0, 2, 1, 3);
    v = ggml_permute(g.ctx, v, 0, 2, 1, 3);
    g.fa = ggml_flash_attn_ext(g.ctx, q, k, v, g.mask, c.scale, 0.0f, 0.0f);
    ggml_prec_set_acc(g.fa, GGML_PREC_F32);
    g.out = ggml_reshape_2d(g.ctx, g.fa, g.fa->ne[0] * g.fa->ne[1], g.fa->ne[2] * g.fa->ne[3]);
    ggml_set_name(g.out, "kqv_out");
    ggml_set_output(g.out);
    g.gf = ggml_new_graph(g.ctx);
    ggml_build_forward_expand(g.gf, g.sk);
    ggml_build_forward_expand(g.gf, g.sv);
    ggml_build_forward_expand(g.gf, g.out);
    return g;
}

struct fa_stats {
    double worst_nmse_binary = 0, worst_nmse_values = 0;
    int cases = 0, runs = 0, cache_tie_rows = 0;
};

void run_fa_case(const fa_case & c, const tsr_att_cfg & cfg, fa_stats * st) {
    const uint32_t n_head = c.nkvh * c.G, E = c.D * c.nkvh;
    const size_t rb = ggml_row_size(GGML_TYPE_Q8_0, E);

    // caches on both backends (b = 0 the device, 1 the CPU), with the same history: the cells of
    // positions before the first new token (a cell without a position stays zero, as in a fresh
    // llama.cpp cache)
    ggml_context * ckv[2];
    ggml_tensor * ck[2], * cv[2];
    ggml_backend_buffer_t kvbuf[2];
    for (int b = 0; b < 2; ++b) {
        ckv[b] = new_ctx(4);
        ck[b] = ggml_new_tensor_3d(ckv[b], GGML_TYPE_Q8_0, E, c.kv_size, 1);
        cv[b] = ggml_new_tensor_3d(ckv[b], GGML_TYPE_Q8_0, E, c.kv_size, 1);
        ggml_set_name(ck[b], "cache_k_l0");
        ggml_set_name(cv[b], "cache_v_l0");
        kvbuf[b] = ggml_backend_alloc_ctx_tensors(ckv[b], b == 0 ? g_tsr : g_cpu);
        ggml_backend_buffer_clear(kvbuf[b], 0);
    }
    CHECK(strcmp(h_storage(kvbuf[0]), "kv") == 0, "%s: cache storage %s", c.name, h_storage(kvbuf[0]));
    std::vector<float> hist((size_t) c.kv_size * E);
    for (auto & x : hist) x = urand(-1.0f, 1.0f);
    std::vector<uint8_t> wk = q8_rows(hist.data(), c.kv_size, E);   // the expected cache contents
    for (auto & x : hist) x = urand(-1.0f, 1.0f);
    std::vector<uint8_t> wv = q8_rows(hist.data(), c.kv_size, E);
    {
        const mask_case m0 = make_mask(c.mode, c.n_kv, c.T, c.p0, c.swa);
        for (uint32_t j = 0; j < c.kv_size; ++j) {
            if (!(j < c.n_kv && m0.pos[j] >= 0 && m0.pos[j] < c.p0)) {
                memset(wk.data() + j * rb, 0, rb);
                memset(wv.data() + j * rb, 0, rb);
            }
        }
    }
    for (int b = 0; b < 2; ++b) {
        ggml_backend_tensor_set(ck[b], wk.data(), 0, wk.size());
        ggml_backend_tensor_set(cv[b], wv.data(), 0, wv.size());
    }

    // one graph per backend, allocated once and run for every step (a decode loop: the device reuses
    // the op list, the mask and indices change through set_tensor and the preload table)
    attn_graph g[2];
    ggml_gallocr_t ga[2] = { nullptr, nullptr };
    bool ok = true;
    for (int b = 0; b < 2; ++b) {
        g[b] = build_attn(c, ck[b], cv[b]);
        if (b == 0) {
            for (int i = 0; i < ggml_graph_n_nodes(g[0].gf); ++i) {
                ggml_tensor * node = ggml_graph_node(g[0].gf, i);
                if (!ggml_backend_supports_op(g_tsr, node)) {
                    CHECK(false, "%s: the device does not support %s (%s)", c.name, ggml_op_desc(node), node->name);
                    ok = false;
                }
            }
        }
        ga[b] = ggml_gallocr_new(ggml_backend_get_default_buffer_type(b == 0 ? g_tsr : g_cpu));
        CHECK(ggml_gallocr_alloc_graph(ga[b], g[b].gf), "%s: gallocr", c.name);
    }

    // the composite reference (AT §6.6) on a cache image (ggml q8_0 bytes), an MSUM and an f32 copy
    auto oracle = [&](const std::vector<float> & q, const std::vector<int64_t> & idx, const std::vector<uint8_t> & kimg,
                      const std::vector<uint8_t> & vimg, const std::vector<uint32_t> & words, const std::vector<float> & copy,
                      uint32_t C, uint32_t LG, std::vector<float> * res) {
        const tsr_kv_geom geo = { E, C, LG };
        const uint64_t pb = tsr_kv_plane_beats(&geo) * 64;
        std::vector<std::vector<uint8_t>> kp(C, std::vector<uint8_t>(pb)), vp(C, std::vector<uint8_t>(pb));
        std::vector<uint8_t *> kpp(C), vpp(C);
        for (uint32_t ch = 0; ch < C; ++ch) {
            kpp[ch] = kp[ch].data();
            vpp[ch] = vp[ch].data();
        }
        tsr_kv_set_bytes(TSR_KV_K, &geo, kpp.data(), kimg.data(), 0, kimg.size());
        tsr_kv_set_bytes(TSR_KV_V, &geo, vpp.data(), vimg.data(), 0, vimg.size());
        tsr_msum ms;
        tsr_msum_view(words.data(), words.size(), &ms);
        std::vector<uint32_t> runs(2 * c.T);
        for (uint32_t t = 0; t < c.T; ++t) {
            runs[2 * t] = (uint32_t) idx[t];
            runs[2 * t + 1] = (uint32_t) idx[t] + 1;
        }
        res->assign((size_t) c.T * n_head * c.D, NAN);
        tsr_fa_args a = {};
        a.T = c.T;
        a.D = c.D;
        a.G = c.G;
        a.scale = c.scale;
        a.q = q.data();
        a.E = E;
        a.LG = LG;
        a.kplane = (const uint8_t * const *) kpp.data();
        a.vplane = (const uint8_t * const *) vpp.data();
        a.msum = &ms;
        a.mask_f32 = copy.data();
        a.new_runs = (const uint32_t (*)[2]) runs.data();
        a.n_new = c.T;
        a.out = res->data();
        return tsr_ref_fa(&cfg, &a);
    };

    for (uint32_t step = 0; ok && step < c.steps; ++step) {
        const int64_t p = c.p0 + (int64_t) step * c.T;
        const mask_case mc = make_mask(c.mode, c.n_kv, c.T, p, c.swa);
        std::vector<float> q((size_t) c.T * n_head * c.D), kc((size_t) c.T * E), vc((size_t) c.T * E);
        for (auto & x : q) x = urand(-1.0f, 1.0f);
        for (auto & x : kc) x = urand(-1.0f, 1.0f);
        for (auto & x : vc) x = urand(-1.0f, 1.0f);
        std::vector<int64_t> idx(c.T);
        for (uint32_t t = 0; t < c.T; ++t) {
            idx[t] = c.mode == M_RING ? (p + t) % c.kv_size : p + t;
        }
        const std::vector<uint8_t> nk = q8_rows(kc.data(), c.T, E), nv = q8_rows(vc.data(), c.T, E);
        for (uint32_t t = 0; t < c.T; ++t) {
            memcpy(wk.data() + idx[t] * rb, nk.data() + t * rb, rb);
            memcpy(wv.data() + idx[t] * rb, nv.data() + t * rb, rb);
        }

        // CPU first; then the reference on the CPU's cache and the host's MSUM (checks the harness
        // and the reference against ggml, independent of the device); then the device
        std::vector<float> out[2];
        std::vector<uint8_t> cpu_k(wk.size()), cpu_v(wv.size());
        double e_host = 0;
        for (int b = 1; b >= 0; --b) {
            const attn_graph & gb = g[b];
            ggml_backend_tensor_set(gb.q_cur, q.data(), 0, q.size() * 4);
            ggml_backend_tensor_set(gb.k_cur, kc.data(), 0, kc.size() * 4);
            ggml_backend_tensor_set(gb.v_cur, vc.data(), 0, vc.size() * 4);
            ggml_backend_tensor_set(gb.k_idxs, idx.data(), 0, idx.size() * 8);
            ggml_backend_tensor_set(gb.v_idxs, idx.data(), 0, idx.size() * 8);
            ggml_backend_tensor_set(gb.mask, mc.m.data(), 0, mc.m.size() * 2);
            CHECK(ggml_backend_graph_compute(b == 0 ? g_tsr : g_cpu, gb.gf) == GGML_STATUS_SUCCESS, "%s: compute", c.name);
            out[b].resize((size_t) c.T * n_head * c.D);
            ggml_backend_tensor_get(gb.out, out[b].data(), 0, out[b].size() * 4);
            if (b == 1) {
                ggml_backend_tensor_get(ck[1], cpu_k.data(), 0, cpu_k.size());
                ggml_backend_tensor_get(cv[1], cpu_v.data(), 0, cpu_v.size());
                std::vector<uint32_t> words(tsr_msum_words(c.T));
                tsr_msum_build_f16(mc.m.data(), c.n_kv, c.n_kv, c.T, 0, words.data());
                std::vector<float> copy((size_t) c.T * 16 * ((c.n_kv + 15) / 16));
                tsr_msum_mask_f32(mc.m.data(), c.n_kv, c.n_kv, c.T, c.scale, copy.data());
                std::vector<float> ref;
                const int rc = oracle(q, idx, cpu_k, cpu_v, words, copy, cfg.C, tsr_kv_lg_min(c.kv_size, cfg.C), &ref);
                CHECK(rc == TSR_FA_OK, "%s: tsr_ref_fa on the CPU cache %d", c.name, rc);
                e_host = nmse(ref, out[1]);
                continue;
            }
            // the device cache: the history and every step's new rows as quantize_row_q8_0_ref writes them
            std::vector<uint8_t> gk(wk.size()), gv(wv.size());
            ggml_backend_tensor_get(ck[0], gk.data(), 0, gk.size());
            ggml_backend_tensor_get(cv[0], gv.data(), 0, gv.size());
            CHECK(gk == wk && gv == wv, "%s step %u: the device cache differs from history + quantize_row_q8_0_ref", c.name, step);
            // the CPU's cache: up to x86 quantize_row_q8_0's rare rounding ties (AT §6.7 source 1)
            for (uint32_t r = 0; r < c.kv_size; ++r) {
                st->cache_tie_rows += memcmp(gk.data() + r * rb, cpu_k.data() + r * rb, rb) != 0;
                st->cache_tie_rows += memcmp(gv.data() + r * rb, cpu_v.data() + r * rb, rb) != 0;
            }
            // the oracle: tsr_ref_fa on the device's own cache, MSUM, f32 copy and new cells
            uint32_t info[9];
            CHECK(h_kv_plane(ck[0], info) == 0, "%s: plane", c.name);
            std::vector<uint32_t> words(tsr_msum_words(c.T));
            std::vector<float> copy((size_t) c.T * 16 * ((c.n_kv + 15) / 16));
            CHECK(h_mask(gb.fa, words.data(), words.size(), copy.data(), copy.size()) == words.size(), "%s: MSUM", c.name);
            std::vector<float> ref;
            const int rc = oracle(q, idx, gk, gv, words, copy, info[8], info[6], &ref);
            CHECK(rc == TSR_FA_OK, "%s: tsr_ref_fa %d", c.name, rc);
            size_t diff = 0;
            for (size_t i = 0; i < ref.size(); ++i) {
                uint32_t x, y;
                memcpy(&x, &ref[i], 4);
                memcpy(&y, &out[0][i], 4);
                diff += x != y;
            }
            CHECK(diff == 0, "%s step %u: %zu of %zu outputs differ from tsr_ref_fa (the bit-exact oracle)", c.name, step, diff,
                  ref.size());
        }
        const double e = nmse(out[0], out[1]);
        const bool values = c.mode == M_FINITE;
        CHECK(e <= 5e-4 && e_host <= 5e-4, "%s step %u: NMSE %.3e (device), %.3e (reference on the CPU cache) vs the CPU above "
              "5e-4", c.name, step, e, e_host);
        CHECK(values || (e <= 1e-6 && e_host <= 1e-6), "%s step %u: NMSE %.3e (device), %.3e (reference) vs the CPU above 1e-6 "
              "(+-0 / -inf mask)", c.name, step, e, e_host);
        if (values) st->worst_nmse_values = std::max(st->worst_nmse_values, e);
        else        st->worst_nmse_binary = std::max(st->worst_nmse_binary, e);
        st->runs++;
        if (step == 0 || step + 1 == c.steps) {
            printf("  %-28s D=%3u heads %2u/%u T=%u n_kv=%4u %-6s step %u: NMSE vs CPU %.3e (reference on the CPU cache %.3e)\n",
                   c.name, c.D, n_head, c.nkvh, c.T, c.n_kv, mode_name(c.mode), step, e, e_host);
        }
    }
    st->cases++;
    for (int b = 0; b < 2; ++b) {
        if (ga[b] != nullptr) ggml_gallocr_free(ga[b]);
        ggml_free(g[b].ctx);
        ggml_backend_buffer_free(kvbuf[b]);
        ggml_free(ckv[b]);
    }
}

void test_fa(const tsr_att_cfg & cfg) {
    printf("SET_ROWS + FLASH_ATTN_EXT (device vs CPU vs tsr_ref_fa)\n");
    const float s64 = 0.125f, s128 = 1.0f / sqrtf(128.0f), s256 = 1.0f / 16.0f;
    const fa_case cases[] = {
        // name                          D    kvh  G  kv_size n_kv  T  mode      p0    swa  scale steps
        { "decode causal (Qwen 0.5B)",    64,  2,  7,  512,   256, 1, M_CAUSAL,  100,   0, s64,  3 },
        { "prefill causal",              128,  4,  2,  512,   256, 8, M_CAUSAL,   60,   0, s128, 2 },
        { "prefill causal T=5",           64,  2,  3,  512,   512, 5, M_CAUSAL,  300,   0, s64,  1 },
        { "decode long (Qwen 32B shape)", 128, 8,  5, 4096,  2048, 1, M_CAUSAL, 1900,   0, s128, 2 },
        { "decode swa (Gemma SWA)",      256,  8,  2,  512,   256, 1, M_SWA,     200, 100, 1.0f, 3 },
        { "prefill swa",                 256,  2,  2,  512,   256, 5, M_SWA,     150,  64, 1.0f, 2 },
        { "decode D=512 (Gemma full)",   512,  1, 16,  512,   256, 1, M_CAUSAL,  200,   0, 1.0f, 2 },
        { "decode wrapped ring",         128,  2,  4,  256,   256, 1, M_RING,    300, 128, s128, 3 },
        { "prefill wrapped ring",         64,  2,  2,  256,   256, 4, M_RING,    290, 100, s64,  2 },
        // the newest cell in the low run of a wrapped ring whose older run spans two chunks
        // (NCG 64 groups at C = 4): three chunks, ordered (low, high, high) by u0 and (high, high, low)
        // by AT §6.6 rule 7, so the output bits depend on FA_NEW naming the new cell (with two chunks,
        // or the new cell already last, they would not: AMERGE of two is symmetric)
        { "decode ring, newest first",   128,  2,  4, 2048,  2048, 1, M_RING,   2088, 2000, s128, 3 },
        { "prefill holes (GENERAL)",     128,  2,  2,  512,   256, 3, M_HOLES,    90,   0, s128, 2 },
        { "prefill holes, 4 chunks",      64,  2,  2, 1024,  1024, 4, M_HOLES,   900,   0, s64,  1 },
        { "decode finite values",         64,  4,  2,  512,   256, 1, M_FINITE,  120,   0, s64,  2 },
        { "prefill finite values",       256,  1,  4,  512,   512, 2, M_FINITE,  260,   0, s256, 1 },
    };
    fa_stats st;
    for (const fa_case & c : cases) {
        run_fa_case(c, cfg, &st);
    }
    printf("  %d cases, %d runs: worst NMSE %.3e (+-0 / -inf masks), %.3e (finite mask values); cache rows differing from the "
           "CPU's (x86 rounding ties) %d\n", st.cases, st.runs, st.worst_nmse_binary, st.worst_nmse_values, st.cache_tie_rows);
}

// =================================================================================================
// 4. supports_op and where operands live (SW §8 R6 and §9.3 as built, SW-41)
// =================================================================================================
void test_placement_rules() {
    printf("placement rules of supports_op\n");
    // (a) a result that is a view whose rows end inside a line (the kinds store whole lines): refused;
    //     rows of whole lines: run, and the viewed tensor's bytes outside the view keep their values
    for (int64_t n : { 17, 16 }) {
        ggml_context * c = new_ctx(8);
        ggml_tensor * A = ggml_new_tensor_2d(c, GGML_TYPE_F32, 32, 4);
        ggml_tensor * S = ggml_new_tensor_2d(c, GGML_TYPE_F32, 32, 4);
        ggml_tensor * vA = ggml_view_2d(c, A, n, 4, A->nb[1], 0);
        ggml_tensor * vS = ggml_view_2d(c, S, n, 4, S->nb[1], 0);
        ggml_tensor * out = ggml_cpy(c, vS, vA);
        ggml_cgraph * gf = ggml_new_graph(c);
        ggml_build_forward_expand(gf, out);
        const bool sup = ggml_backend_supports_op(g_tsr, out);
        CHECK(sup == (n == 16), "CPY into a [%lld x 4] view of [32 x 4]: supports_op %d", (long long) n, (int) sup);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(c, g_tsr);
        std::vector<float> a(128, 0.25f), sv(128), r(128);
        for (auto & x : sv) x = urand(-1.0f, 1.0f);
        ggml_backend_tensor_set(A, a.data(), 0, 512);
        ggml_backend_tensor_set(S, sv.data(), 0, 512);
        if (sup) {
            CHECK(ggml_backend_graph_compute(g_tsr, gf) == GGML_STATUS_SUCCESS, "CPY into a view: compute");
            ggml_backend_tensor_get(A, r.data(), 0, 512);
            int bad = 0;
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 32; ++j)
                    bad += r[i * 32 + j] != (j < n ? sv[i * 32 + j] : 0.25f);
            CHECK(bad == 0, "CPY into a [%lld x 4] view: %d values of the viewed tensor wrong", (long long) n, bad);
        }
        ggml_backend_buffer_free(buf);
        ggml_free(c);
    }
    // (b) leaf data in host memory (llama.cpp's control vectors and LoRA adapters: an ANY buffer holding
    //     only leaf tensors is HOSTED): an op reading it stays on the CPU, before and after the buffer's
    //     storage is decided; the M7 kinds address operands in place only
    {
        ggml_context * cl = new_ctx(2);
        ggml_tensor * dir = ggml_new_tensor_1d(cl, GGML_TYPE_F32, 64);
        ggml_set_name(dir, "direction.1");
        ggml_backend_buffer_t lb = ggml_backend_alloc_ctx_tensors(cl, g_tsr);
        ggml_context * cg = new_ctx(4);
        ggml_tensor * cur = ggml_new_tensor_2d(cg, GGML_TYPE_F32, 64, 2);
        ggml_tensor * add = ggml_add(cg, cur, dir);
        ggml_tensor * add2 = ggml_add(cg, cur, cur);
        CHECK(!ggml_backend_supports_op(g_tsr, add), "ADD reading a leaf-only buffer (undecided) is accepted");
        CHECK(ggml_backend_supports_op(g_tsr, add2), "ADD of compute tensors is refused");
        std::vector<float> d(64, 0.5f);
        ggml_backend_tensor_set(dir, d.data(), 0, 256);
        CHECK(strcmp(h_storage(lb), "host") == 0, "leaf-only buffer storage %s", h_storage(lb));
        CHECK(!ggml_backend_supports_op(g_tsr, add), "ADD reading a host-memory buffer is accepted");
        ggml_free(cg);
        ggml_backend_buffer_free(lb);
        ggml_free(cl);
    }
    printf("  results that are views, leaf data in host memory: ok\n");
}

}  // namespace

int main() {
    g_tsr = ggml_backend_tesseract_init();
    if (g_tsr == nullptr) {
        fprintf(stderr, "test-tsr-attn: no Tesseract device (TSR_DEVICE=%s)\n", getenv("TSR_DEVICE") ? getenv("TSR_DEVICE") : "");
        return 1;
    }
    g_cpu = ggml_backend_cpu_init();
    ggml_backend_reg_t reg = ggml_backend_tesseract_reg();
    h_storage  = (storage_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_tesseract_buffer_storage");
    h_kv_plane = (kv_plane_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_tesseract_kv_plane");
    h_mask     = (mask_summary_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_tesseract_mask_summary");
    h_att_cfg  = (att_cfg_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_tesseract_att_cfg");
    if (!h_storage || !h_kv_plane || !h_mask || !h_att_cfg) {
        fprintf(stderr, "test-tsr-attn: the backend lacks the test hooks\n");
        return 1;
    }
    uint32_t ac[4];
    if (h_att_cfg(ac) != 0) {
        fprintf(stderr, "test-tsr-attn: no AT configuration\n");
        return 1;
    }
    const tsr_att_cfg cfg = { ac[0], ac[1], ac[2], ac[3] };
    printf("test-tsr-attn on %s: C=%u T_MAX=%u PCAP=%u K_MAX=%u\n", ggml_backend_dev_description(ggml_backend_get_device(g_tsr)),
           cfg.C, cfg.t_max, cfg.pcap, cfg.k_max);

    test_kv_buffer();
    test_mask_summaries(cfg);
    test_fa(cfg);
    test_placement_rules();

    ggml_backend_free(g_cpu);
    ggml_backend_free(g_tsr);
    if (g_fail) {
        fprintf(stderr, "test-tsr-attn: %d check(s) FAILED\n", g_fail);
        return 1;
    }
    printf("test-tsr-attn: OK\n");
    return 0;
}
