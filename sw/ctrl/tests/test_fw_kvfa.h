// test_fw_kvfa.h — the cases of fw_kvfa_* (sw/ctrl/tests/test_fw_kvfa_rv.cpp, test_fw_kvfa_host.cpp):
// one OPL list per case, SET_ROWS_KV of the new tokens' K rows, SET_ROWS_KV of their V
// rows, then FLASH_ATTN over the cache (docs/design/06-software-verification.md "SW" §7.2, §7.5;
// docs/design/03-attention-kv.md "AT" §6.6, §7). Header-only C++20, host side.
//
// A case is a model shape (D, n_head_kv, G), a cache of n_kv cells holding positions (llama.cpp's
// layout: cell p for position p, or p mod n_kv for a wrapped SWA ring), T new tokens at positions
// pos0 .. pos0 + T - 1 (their cells are the SET_ROWS indices), and llama.cpp's mask rules for those
// queries (causal; standard SWA p_q - p >= n_swa masked; a finite mask; holes; an empty row). The
// data are generated once per case: the planes' old cells (random rows through QACT A32 + KVST, the
// device image), the new K / V source rows, Q, the F16 mask and its MSUM (tsr_msum_build_f16) and,
// when a tile needs the GENERAL path, the f32 copy (tsr_msum_mask_f32 with the op's scale). The
// expected results come from sw/ctrl/ref/kvfa_ref.hpp on host copies of the planes: the planes after
// both SET_ROWS (tsr_qact + tsr_ref_kvst per row) and the FA output rows (tsr_ref_fa).
#pragma once

#include "test_sim_obs.h"   // first: it sets _GNU_SOURCE (the includer #undefs it before)

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "../ref/kvfa_ref.hpp"
#include "tsr/opl.hpp"
#include "tsr/plan.hpp"

namespace kvfa {

constexpr uint64_t LINE = 64u;
inline uint32_t H(uint32_t ch, uint64_t beat) { return (ch << 28) | uint32_t(beat & 0x0FFFFFFFu); }
inline uint32_t f2b(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

enum mask_kind { M_CAUSAL, M_SWA, M_RING, M_FINITE, M_HOLES, M_EMPTYROW };

struct kcase {
    const char * name;
    const char * group;     // ctest group (test_fw_kvfa_rv suite <group>)
    uint32_t D, nkvh, G;    // head dimension, KV heads, query heads per KV head
    uint32_t T;             // new tokens = query rows = SET_ROWS rows
    uint32_t n_kv;          // cells of the cache view (the ring's size for M_RING)
    uint32_t pos0;          // position of the first new token
    uint32_t n_swa;         // window (M_SWA, M_RING)
    int mask;
    float scale, qmul;
    uint32_t pad;           // extra lines in the q / dst / src row strides (0: dense rows)
    uint32_t kmax = 0;      // the device's K_MAX (0: the U200's 32,768); a small one gives a small
                            // PCAP, so small chunks: many per tile (buffer rotation, rule 5)
    uint32_t scatter = 0;   // > 0: FLASH_ATTN's new cells (rule 7) are this many cells 32 j + 5 (every
                            // other group, all in the first chunks) of their own preload entry
                            // instead of the SET_ROWS indices: those chunks move to the end; more
                            // than 8 group runs take the kernel's scan of the indices
};

// Qwen2.5-0.5B (D 64, n_head_kv 2, G 7), Qwen2.5-32B (D 128, 8, 5), Gemma 4 12B SWA layers (D 256,
// 8, 2, window 1024, 1,536-cell ring) and global layers (D 512, 1, 16: two head passes); n_kv 16,
// 256, 4096 (16K: group "slow"); decode and prefill; causal, SWA, the wrapped ring, GENERAL masks.
inline const std::vector<kcase> & cases() {
    static const float s64 = 0.125f, s128 = 0.08838834764831845f;
    static const std::vector<kcase> v = {
        // name                       group        D    kvh G   T   n_kv   pos0   nswa  mask        scale qmul pad
        { "q05b-dec-16",              "qwen05b",   64,  2, 7,  1,  16,    11,    0,    M_CAUSAL,   s64,  1.0f, 0 },
        { "q05b-dec-256",             "qwen05b",   64,  2, 7,  1,  256,   200,   0,    M_CAUSAL,   s64,  1.0f, 1 },
        { "q05b-dec-4096",            "qwen05b",   64,  2, 7,  1,  4096,  4000,  0,    M_CAUSAL,   s64,  1.0f, 0 },
        { "q05b-pre-16",              "qwen05b",   64,  2, 7,  9,  16,    3,     0,    M_CAUSAL,   s64,  1.0f, 0 },
        { "q05b-pre-256",             "qwen05b",   64,  2, 7,  6,  256,   120,   0,    M_CAUSAL,   s64,  1.0f, 2 },
        { "q05b-pre-4096",            "qwen05b",   64,  2, 7,  5,  4096,  2047,  0,    M_CAUSAL,   s64,  1.0f, 0 },
        { "q05b-dec-4096-pcap304",    "qwen05b",   64,  2, 7,  1,  4096,  4090,  0,    M_CAUSAL,   s64,  1.0f, 0, 1024 },
        { "q05b-dec-4096-new5",       "qwen05b",   64,  2, 7,  1,  4096,  4090,  0,    M_CAUSAL,   s64,  1.0f, 0, 1024, 5 },
        { "q05b-pre-4096-new12-scan", "qwen05b",   64,  2, 7,  3,  4096,  4000,  0,    M_CAUSAL,   s64,  1.0f, 0, 1024, 12 },
        { "q32b-dec-16",              "qwen32b",   128, 8, 5,  1,  16,    15,    0,    M_CAUSAL,   s128, 1.0f, 0 },
        { "q32b-dec-256",             "qwen32b",   128, 8, 5,  1,  256,   255,   0,    M_CAUSAL,   s128, 1.0f, 0 },
        { "q32b-dec-4096",            "qwen32b",   128, 8, 5,  1,  4096,  3999,  0,    M_CAUSAL,   s128, 1.0f, 3 },
        { "q32b-pre-256",             "qwen32b",   128, 8, 5,  3,  256,   60,    0,    M_CAUSAL,   s128, 1.0f, 0 },
        { "q32b-pre-4096",            "qwen32b",   128, 8, 5,  2,  4096,  3001,  0,    M_CAUSAL,   s128, 1.0f, 0 },
        { "g12swa-dec-16",            "gemma_swa", 256, 8, 2,  1,  16,    7,     1024, M_SWA,      1.0f, 0.08f, 0 },
        { "g12swa-dec-256",           "gemma_swa", 256, 8, 2,  1,  256,   250,   1024, M_SWA,      1.0f, 0.08f, 0 },
        { "g12swa-dec-4096",          "gemma_swa", 256, 8, 2,  1,  4096,  3500,  1024, M_SWA,      1.0f, 0.08f, 0 },
        { "g12swa-pre-16",            "gemma_swa", 256, 8, 2,  6,  16,    5,     1024, M_SWA,      1.0f, 0.08f, 0 },
        { "g12swa-pre-4096-slide",    "gemma_swa", 256, 8, 2,  8,  4096,  2000,  1024, M_SWA,      1.0f, 0.08f, 1 },
        { "g12swa-ring-dec",          "gemma_swa", 256, 8, 2,  1,  1536,  3200,  1024, M_RING,     1.0f, 0.08f, 0 },
        { "g12swa-ring-pre-wrap",     "gemma_swa", 256, 8, 2,  5,  1536,  3070,  1024, M_RING,     1.0f, 0.08f, 0 },
        { "g12swa-pre-causal-pcap1216", "gemma_swa", 256, 8, 2, 8, 4096,  2297,  0,    M_CAUSAL,   1.0f, 0.08f, 0, 4096 },
        { "g12glb-dec-16",            "gemma_glb", 512, 1, 16, 1,  16,    9,     0,    M_CAUSAL,   1.0f, 0.06f, 0 },
        { "g12glb-dec-256",           "gemma_glb", 512, 1, 16, 1,  256,   222,   0,    M_CAUSAL,   1.0f, 0.06f, 0 },
        { "g12glb-dec-4096",          "gemma_glb", 512, 1, 16, 1,  4096,  4095,  0,    M_CAUSAL,   1.0f, 0.06f, 0 },
        { "g12glb-pre-256",           "gemma_glb", 512, 1, 16, 3,  256,   100,   0,    M_CAUSAL,   1.0f, 0.06f, 1 },
        { "general-finite-dec-256",   "general",   128, 8, 5,  1,  256,   200,   0,    M_FINITE,   s128, 1.0f, 0 },
        { "general-finite-pre-4096",  "general",   128, 8, 5,  4,  4096,  1500,  0,    M_FINITE,   s128, 1.0f, 0 },
        // GENERAL tiles of four chunks (K_MAX 8,192: PCAP 2,336, NCG 72): the mask LD into M[k mod 2]
        // after the AMASK of chunk k - 2 (k_fattn.cpp's dep) is exercised from chunk 2 on.
        { "general-finite-pre-4096-pcap", "general", 128, 8, 5, 4,  4096,  1500,  0,    M_FINITE,   s128, 1.0f, 0, 8192 },
        { "general-finite-g12swa",    "general",   256, 8, 2,  4,  256,   100,   0,    M_FINITE,   0.5f, 0.08f, 0 },
        { "general-holes-q05b",       "general",   64,  2, 7,  3,  256,   150,   0,    M_HOLES,    s64,  1.0f, 0 },
        { "empty-row-q05b",           "general",   64,  2, 7,  2,  256,   40,    0,    M_EMPTYROW, s64,  1.0f, 0 },
        { "q05b-dec-16k",             "slow",      64,  2, 7,  1,  16384, 16000, 0,    M_CAUSAL,   s64,  1.0f, 0 },
        { "q32b-dec-16k",             "slow",      128, 8, 5,  1,  16384, 16383, 0,    M_CAUSAL,   s128, 1.0f, 0 },
        { "q32b-pre-16k",             "slow",      128, 8, 5,  2,  16384, 9000,  0,    M_CAUSAL,   s128, 1.0f, 0 },
        { "g12glb-dec-16k",           "slow",      512, 1, 16, 1,  16384, 12000, 0,    M_CAUSAL,   1.0f, 0.06f, 0 },
        { "g12swa-dec-16k-slide",     "slow",      256, 8, 2,  1,  16384, 15000, 1024, M_SWA,      1.0f, 0.08f, 0 },
    };
    return v;
}

inline tsr_config sim_cfg(uint32_t kmax = 0) {
    tsr_config c = TSR_CONFIG_U200;
    c.t_max = 8;
    if (kmax) c.k_max = kmax;
    c.ch_bytes = 256ull << 20;
    c.caps = TSR_SIM_CAPS_IMPLEMENTED;
    return c;
}
inline tsr_att_cfg att_cfg(const tsr_config & c) {
    return tsr_att_cfg{ c.num_ch, c.t_max, tsr_qa_lines(TSR_QA_A256, c.k_max) * 16u, c.k_max };
}

// One case's data and list.
struct kdata {
    const kcase * c = nullptr;
    kcase own{};                                  // build_setrows's case (c points at it)
    bool has_fa = true;                           // the list ends with FLASH_ATTN
    tsr_config cfg{};
    uint32_t E = 0, n_head = 0, K = 0, LG = 0, C = 0;
    uint32_t q_h = 0, q_stride = 0, dst_h = 0, dst_stride = 0, srck_h = 0, srcv_h = 0, src_stride = 0, mask_h = 0;
    uint32_t idx_h = 0, msum_h = 0, opl_h = 0;
    uint32_t kB = 0, vB = 0;
    uint64_t plane_beats = 0;
    std::vector<std::vector<uint8_t>> kp, vp;     // the planes before the list (per channel image)
    std::vector<float> q, srck, srcv;             // q rows [T][K], new K / V rows [T][E]
    std::vector<uint8_t> idx;                     // I64 cells of the new tokens
    std::vector<uint8_t> idx_new;                 // kcase::scatter's cells (FLASH_ATTN's new cells)
    uint32_t idx_new_h = 0, pre_new = 0;
    std::vector<uint32_t> msum;
    std::vector<float> mask_f32;                  // empty: no f32 copy
    std::vector<uint8_t> pre;                     // the preload area's image
    uint32_t pre_idx = 0, pre_msum = 0;
    tsr::opl::set_rows_kv_args sk{}, sv{};
    tsr::opl::flash_attn_args fa{};
    std::vector<uint8_t> blob;
    // expected (set by expect())
    std::vector<std::vector<uint8_t>> kp1, vp1;
    std::vector<float> out;
    tsr_fa_stats st{};
    bool have_expect = false;
};

inline void gen_normal(rng_t * r, float * x, size_t n, float mul) {
    for (size_t i = 0; i < n; ++i) x[i] = float(rng_normal(r) * double(mul));
}

// The list of the case's records: preload entries (the indices, then the MSUM), [SET_ROWS_KV K,
// SET_ROWS_KV V,] [FLASH_ATTN after both]; and the preload area's image (as the executor loads it).
inline void make_blob(kdata & k, bool sets = true, bool fa = true) {
    tsr::opl::writer w(tsr::opl::F_STOP_ON_ERROR, 0x6b7666u);
    const uint32_t pi = w.add_preload(k.idx_h, uint32_t(k.idx.size()));
    const uint32_t pm = w.add_preload(k.msum_h, uint32_t(k.msum.size() * 4u));
    CHECK(pi == k.pre_idx && pm == k.pre_msum, "%s: preload offsets %u / %u", k.c->name, pi, pm);
    if (!k.idx_new.empty()) {
        const uint32_t pn = w.add_preload(k.idx_new_h, uint32_t(k.idx_new.size()));
        CHECK(pn == k.pre_new, "%s: preload offset %u", k.c->name, pn);
    }
    std::vector<uint32_t> deps;
    if (sets) {
        deps.push_back(w.add_set_rows_kv(k.sk));
        deps.push_back(w.add_set_rows_kv(k.sv));
    }
    if (fa) w.add_flash_attn(k.fa, deps.data(), uint32_t(deps.size()));
    k.blob = w.finish();
    CHECK(w.ok() && !k.blob.empty(), "%s: OPL writer: %s", k.c->name, w.error());
    k.pre.assign(size_t(w.pre_lines()) * LINE, 0);
    memcpy(k.pre.data() + k.pre_idx, k.idx.data(), k.idx.size());
    memcpy(k.pre.data() + k.pre_msum, k.msum.data(), k.msum.size() * 4u);
    if (!k.idx_new.empty()) memcpy(k.pre.data() + k.pre_new, k.idx_new.data(), k.idx_new.size());
}

// Generates the case's data and list (deterministic per case name).
inline void build(const kcase & c, kdata & k) {
    k.c = &c;
    k.cfg = sim_cfg(c.kmax);
    k.C = k.cfg.num_ch;
    k.E = c.D * c.nkvh;
    k.n_head = c.G * c.nkvh;
    k.K = k.n_head * c.D;
    k.LG = tsr_kv_lg_min(c.n_kv, k.C);
    const tsr_kv_geom g = { k.E, k.C, k.LG };
    k.plane_beats = tsr_kv_plane_beats(&g);
    uint64_t seed = 0x6b766661u;
    for (const char * p = c.name; *p; ++p) seed = seed * 131u + uint8_t(*p);
    rng_t r = { seed };
    // DRAM placement (channel bytes 256 MiB = 4 Mi beats).
    k.kB = 0x200000u;
    k.vB = k.kB + uint32_t((k.plane_beats + 63u) & ~63ull);
    k.q_stride = k.K / 16u + c.pad;
    k.dst_stride = k.K / 16u + c.pad;
    k.src_stride = k.E / 16u + c.pad;
    k.q_h = H(1, 0x10000u);
    k.dst_h = H(2, 0x10000u);
    k.srck_h = H(3, 0x10000u);
    k.srcv_h = H(3, 0x40000u);
    k.mask_h = H(1, 0x80000u);
    k.idx_h = H(0, 0x20000u);
    k.msum_h = H(0, 0x20100u);
    k.opl_h = H(0, 0x40000u);
    // Cells and positions (llama.cpp: cell p, or p mod n_kv in a ring).
    std::vector<int64_t> cpos(c.n_kv, -1);
    const int64_t npos = int64_t(c.pos0) + c.T;
    for (int64_t p = 0; p < npos; ++p) {
        if (c.mask == M_RING) cpos[size_t(p % c.n_kv)] = p;
        else if (p < int64_t(c.n_kv)) cpos[size_t(p)] = p;
    }
    std::vector<uint32_t> cell(c.T);
    for (uint32_t t = 0; t < c.T; ++t) cell[t] = uint32_t(c.mask == M_RING ? (c.pos0 + t) % c.n_kv : c.pos0 + t);
    // The planes: every cell but the new ones gets a random row (QACT A32 + KVST: the device image).
    k.kp.assign(k.C, std::vector<uint8_t>(size_t(k.plane_beats) * LINE, 0));
    k.vp.assign(k.C, std::vector<uint8_t>(size_t(k.plane_beats) * LINE, 0));
    {
        std::vector<uint8_t *> kpp(k.C), vpp(k.C);
        for (uint32_t i = 0; i < k.C; ++i) { kpp[i] = k.kp[i].data(); vpp[i] = k.vp[i].data(); }
        std::vector<bool> is_new(c.n_kv, false);
        for (uint32_t t = 0; t < c.T; ++t) is_new[cell[t]] = true;
        const size_t ql = size_t(tsr_qa_lines(TSR_QA_A32, k.E)) * LINE;
        std::vector<uint8_t> qa(ql);
        std::vector<float> x(k.E);
        for (uint32_t j = 0; j < c.n_kv; ++j) {
            if (is_new[j]) continue;
            for (int kv = 0; kv < 2; ++kv) {
                gen_normal(&r, x.data(), k.E, 1.0f);
                tsr_qact(TSR_QA_A32, x.data(), k.E, qa.data());
                tsr_ref_kvst(uint32_t(kv), &g, kv ? vpp.data() : kpp.data(), j, 1u, qa.data(), 0u, nullptr);
            }
        }
    }
    k.srck.resize(size_t(c.T) * k.E);
    k.srcv.resize(size_t(c.T) * k.E);
    gen_normal(&r, k.srck.data(), k.srck.size(), 1.0f);
    gen_normal(&r, k.srcv.data(), k.srcv.size(), 1.0f);
    k.q.resize(size_t(c.T) * k.K);
    gen_normal(&r, k.q.data(), k.q.size(), c.qmul);
    k.idx.assign(size_t(c.T) * 8u, 0);
    for (uint32_t t = 0; t < c.T; ++t) memcpy(k.idx.data() + 8u * t, &cell[t], 4);
    // The F16 mask [T][n_kv] (llama.cpp's rules) and its MSUM.
    std::vector<uint16_t> mask(size_t(c.T) * c.n_kv);
    for (uint32_t t = 0; t < c.T; ++t) {
        const int64_t pq = int64_t(c.pos0) + t;
        for (uint32_t j = 0; j < c.n_kv; ++j) {
            const int64_t p = cpos[j];
            bool vis = p >= 0 && p <= pq;
            if ((c.mask == M_SWA || c.mask == M_RING) && pq - p >= int64_t(c.n_swa)) vis = false;
            if (c.mask == M_HOLES && j % 7u == 3u) vis = false;
            if (c.mask == M_EMPTYROW && t == 0) vis = false;
            uint16_t h = vis ? 0x0000 : 0xFC00;
            if (vis && c.mask == M_FINITE) h = tsr_f32_to_fp16(-0.25f * float((pq - p) % 13));
            mask[size_t(t) * c.n_kv + j] = h;
        }
    }
    k.msum.resize(tsr_msum_words(c.T));
    const uint32_t mflags = tsr_msum_build_f16(mask.data(), c.n_kv, c.n_kv, c.T, k.mask_h, k.msum.data());
    // The f32 copy when a tile needs GENERAL: a GENERAL MSUM, or the planner's GENERAL for a tile.
    bool need_f32 = (mflags & TSR_MSUM_GENERAL) != 0u;
    if (!need_f32) {
        tsr_msum ms;
        tsr_msum_view(k.msum.data(), k.msum.size(), &ms);
        const tsr_att_cfg acfg = att_cfg(k.cfg);
        uint32_t GS = 0, Tq = 0;
        tsr_fa_pass_rule(&acfg, c.G, &GS, &Tq);
        for (uint32_t q0 = 0; q0 < c.T && !need_f32; q0 += Tq) {
            const uint32_t tq = std::min(Tq, c.T - q0);
            std::vector<tsr_fa_chunk> ch(4096);
            const long n = tsr_fa_plan(&acfg, k.E, c.D, &ms, q0, tq, nullptr, 0, ch.data(), ch.size());
            for (long i = 0; i < n && size_t(i) < ch.size(); ++i) need_f32 = need_f32 || ch[size_t(i)].general;
        }
    }
    if (need_f32) {
        k.mask_f32.resize(size_t(c.T) * 16u * ((c.n_kv + 15u) / 16u));
        tsr_msum_mask_f32(mask.data(), c.n_kv, c.n_kv, c.T, c.scale, k.mask_f32.data());
    }
    // The records: SET_ROWS_KV K and V, FLASH_ATTN after both (make_blob).
    const uint32_t geom = tsr::opl::geom_word(k.E / 64u, k.LG);
    k.pre_idx = 0u;          // the writer's offsets for the two entries (make_blob checks them)
    k.pre_msum = uint32_t((k.idx.size() + 63u) & ~size_t(63));
    k.sk = tsr::opl::set_rows_kv_args{ H(0, k.kB), 0u, k.srck_h, k.src_stride, k.pre_idx, c.T, k.E, 0u, k.kB, geom, 0u, 0u };
    k.sv = tsr::opl::set_rows_kv_args{ H(0, k.vB), 0u, k.srcv_h, k.src_stride, k.pre_idx, c.T, k.E, 1u, k.vB, geom, 0u, 0u };
    uint32_t fl = tsr::plan::attn::fa_flags(k.pre_idx, c.T);
    if (c.scatter != 0u) {
        k.idx_new.assign(size_t(c.scatter) * 8u, 0);
        for (uint32_t j = 0; j < c.scatter; ++j) {
            const uint32_t cell_j = (32u * j + 5u) % c.n_kv;
            memcpy(k.idx_new.data() + 8u * j, &cell_j, 4);
        }
        k.idx_new_h = H(0, 0x20200u);
        k.pre_new = k.pre_msum + uint32_t((k.msum.size() * 4u + 63u) & ~size_t(63));
        fl = tsr::plan::attn::fa_flags(k.pre_new, c.scatter);
    }
    k.fa = tsr::opl::flash_attn_args{ k.dst_h, k.dst_stride, k.q_h, c.D / 16u, k.q_stride, H(0, k.kB), H(0, k.vB), c.n_kv, c.D,
                                      k.n_head, c.nkvh, c.T, k.pre_msum,
                                      need_f32 ? k.mask_h : tsr::opl::HANDLE_NONE, f2b(c.scale), 0u, k.kB, k.vB, geom, c.D,
                                      c.G, fl };
    make_blob(k);
}

// SET_ROWS_KV alone (K and V), with rows the kernel splits: `cells` in this order (runs of
// consecutive cells, cut at 64 rows; repeats: the later row wins), source rows `pad` lines apart
// beyond E/16. The planes' other cells hold random rows. The shape is c's (D, n_head_kv, n_kv).
inline void build_setrows(kdata & k, const kcase & c, const std::vector<uint32_t> & cells, uint32_t pad) {
    k.own = c;
    k.own.T = uint32_t(cells.size());
    k.own.pad = pad;
    k.c = &k.own;
    k.has_fa = false;
    k.cfg = sim_cfg();
    k.C = k.cfg.num_ch;
    k.E = c.D * c.nkvh;
    k.n_head = c.G * c.nkvh;
    k.K = k.n_head * c.D;
    k.LG = tsr_kv_lg_min(c.n_kv, k.C);
    const tsr_kv_geom g = { k.E, k.C, k.LG };
    k.plane_beats = tsr_kv_plane_beats(&g);
    rng_t r = { 0x5e7b0u + cells.size() };
    k.kB = 0x200000u;
    k.vB = k.kB + uint32_t((k.plane_beats + 63u) & ~63ull);
    k.q_stride = k.K / 16u;
    k.dst_stride = k.K / 16u;
    k.src_stride = k.E / 16u + pad;
    k.q_h = H(1, 0x10000u);
    k.dst_h = H(2, 0x10000u);
    k.srck_h = H(3, 0x10000u);
    k.srcv_h = H(3, 0x40000u);
    k.mask_h = H(1, 0x80000u);
    k.idx_h = H(0, 0x20000u);
    k.msum_h = H(0, 0x20100u);
    k.opl_h = H(0, 0x40000u);
    k.kp.assign(k.C, std::vector<uint8_t>(size_t(k.plane_beats) * LINE, 0));
    k.vp.assign(k.C, std::vector<uint8_t>(size_t(k.plane_beats) * LINE, 0));
    {
        std::vector<uint8_t *> kpp(k.C), vpp(k.C);
        for (uint32_t i = 0; i < k.C; ++i) { kpp[i] = k.kp[i].data(); vpp[i] = k.vp[i].data(); }
        std::vector<uint8_t> qa(size_t(tsr_qa_lines(TSR_QA_A32, k.E)) * LINE);
        std::vector<float> x(k.E);
        for (uint32_t j = 0; j < c.n_kv; ++j) {
            for (int kv = 0; kv < 2; ++kv) {
                gen_normal(&r, x.data(), k.E, 1.0f);
                tsr_qact(TSR_QA_A32, x.data(), k.E, qa.data());
                tsr_ref_kvst(uint32_t(kv), &g, kv ? vpp.data() : kpp.data(), j, 1u, qa.data(), 0u, nullptr);
            }
        }
    }
    const uint32_t n = k.own.T;
    k.srck.resize(size_t(n) * k.E);
    k.srcv.resize(size_t(n) * k.E);
    gen_normal(&r, k.srck.data(), k.srck.size(), 1.0f);
    gen_normal(&r, k.srcv.data(), k.srcv.size(), 1.0f);
    k.q.assign(size_t(n) * k.K, 0.0f);
    k.idx.assign(size_t(n) * 8u, 0);
    for (uint32_t t = 0; t < n; ++t) memcpy(k.idx.data() + 8u * t, &cells[t], 4);
    k.msum.assign(TSR_MSUM_HDR_WORDS, 0u);   // unused (no FLASH_ATTN): a header for the preload entry
    k.msum[0] = TSR_MSUM_MAGIC;
    const uint32_t geom = tsr::opl::geom_word(k.E / 64u, k.LG);
    k.pre_idx = 0u;
    k.pre_msum = uint32_t((k.idx.size() + 63u) & ~size_t(63));
    k.sk = tsr::opl::set_rows_kv_args{ H(0, k.kB), 0u, k.srck_h, k.src_stride, k.pre_idx, n, k.E, 0u, k.kB, geom, 0u, 0u };
    k.sv = tsr::opl::set_rows_kv_args{ H(0, k.vB), 0u, k.srcv_h, k.src_stride, k.pre_idx, n, k.E, 1u, k.vB, geom, 0u, 0u };
    make_blob(k, true, false);
}

inline uint8_t * at(tsr_sim * sim, uint32_t h, uint64_t beats = 0) {
    return tsr_sim_dram(sim, h >> 28) + (uint64_t(h & 0x0FFFFFFFu) + beats) * LINE;
}

// Writes the case's inputs into a fresh simulator's DRAM.
inline void fill(tsr_sim * sim, const kdata & k) {
    const kcase & c = *k.c;
    for (uint32_t i = 0; i < k.C; ++i) {
        memcpy(tsr_sim_dram(sim, i) + uint64_t(k.kB) * LINE, k.kp[i].data(), k.kp[i].size());
        memcpy(tsr_sim_dram(sim, i) + uint64_t(k.vB) * LINE, k.vp[i].data(), k.vp[i].size());
    }
    for (uint32_t t = 0; t < c.T; ++t) {
        memcpy(at(sim, k.q_h, uint64_t(t) * k.q_stride), k.q.data() + size_t(t) * k.K, size_t(k.K) * 4u);
        memcpy(at(sim, k.srck_h, uint64_t(t) * k.src_stride), k.srck.data() + size_t(t) * k.E, size_t(k.E) * 4u);
        memcpy(at(sim, k.srcv_h, uint64_t(t) * k.src_stride), k.srcv.data() + size_t(t) * k.E, size_t(k.E) * 4u);
    }
    if (!k.mask_f32.empty()) memcpy(at(sim, k.mask_h), k.mask_f32.data(), k.mask_f32.size() * 4u);
    memcpy(at(sim, k.idx_h), k.idx.data(), k.idx.size());
    memcpy(at(sim, k.msum_h), k.msum.data(), k.msum.size() * 4u);
    if (!k.idx_new.empty()) memcpy(at(sim, k.idx_new_h), k.idx_new.data(), k.idx_new.size());
    memcpy(at(sim, k.opl_h), k.blob.data(), k.blob.size());
}

inline tsr::ref::dram_view view_of(tsr_sim * sim, std::vector<uint8_t *> & chp) {
    const tsr_config * c = tsr_sim_config(sim);
    chp.resize(c->num_ch);
    for (uint32_t i = 0; i < c->num_ch; ++i) chp[i] = tsr_sim_dram(sim, i);
    return tsr::ref::dram_view{ chp.data(), c->num_ch, c->ch_bytes };
}

// The expected planes and output rows (the references on the filled simulator's DRAM as input and
// host copies of the planes).
inline void expect(tsr_sim * sim, kdata & k) {
    std::vector<uint8_t *> chp;
    const tsr::ref::dram_view v = view_of(sim, chp);
    k.kp1 = k.kp;
    k.vp1 = k.vp;
    std::vector<uint8_t *> kpp(k.C), vpp(k.C);
    for (uint32_t i = 0; i < k.C; ++i) { kpp[i] = k.kp1[i].data(); vpp[i] = k.vp1[i].data(); }
    uint32_t a[22];
    tsr::opl::put_args(k.sk, a);
    CHECK(tsr::ref::set_rows_kv_ref(v, a, k.pre.data(), k.pre.size(), kpp.data(), k.C) == 0, "%s: set_rows_kv_ref K", k.c->name);
    tsr::opl::put_args(k.sv, a);
    CHECK(tsr::ref::set_rows_kv_ref(v, a, k.pre.data(), k.pre.size(), vpp.data(), k.C) == 0, "%s: set_rows_kv_ref V", k.c->name);
    k.have_expect = true;
    if (!k.has_fa) return;
    tsr::opl::put_args(k.fa, a);
    k.out.assign(size_t(k.c->T) * k.K, 0.0f);
    const tsr_att_cfg acfg = att_cfg(k.cfg);
    std::vector<const uint8_t *> kc(kpp.begin(), kpp.end()), vc(vpp.begin(), vpp.end());
    const int rc = tsr::ref::flash_attn_ref_out(v, a, k.pre.data(), k.pre.size(), kc.data(), vc.data(), acfg, k.out.data(), &k.st);
    CHECK(rc == 0, "%s: flash_attn_ref rc %d", k.c->name, rc);
    if (k.c->scatter != 0u) {
        // The case must discriminate rule 7: without the new cells the chunk order, hence the
        // output bits, differ.
        a[tsr::opl::fa::FLAGS] = 0u;
        std::vector<float> o2(k.out.size());
        tsr::ref::flash_attn_ref_out(v, a, k.pre.data(), k.pre.size(), kc.data(), vc.data(), acfg, o2.data(), nullptr);
        CHECK(memcmp(o2.data(), k.out.data(), o2.size() * 4u) != 0, "%s: the output does not depend on the chunk order",
              k.c->name);
    }
    k.have_expect = true;
}

// After a run: the planes and the output rows bit for bit, and the written DRAM inside the planes and
// the dst rows. Returns false on a difference.
inline bool check_outputs(tsr_sim * sim, const kdata & k, const char * tag, const obs * o) {
    bool ok = true;
    for (uint32_t i = 0; i < k.C; ++i) {
        for (int kv = 0; kv < 2; ++kv) {
            const std::vector<uint8_t> & want = kv ? k.vp1[i] : k.kp1[i];
            const uint8_t * got = tsr_sim_dram(sim, i) + uint64_t(kv ? k.vB : k.kB) * LINE;
            if (memcmp(got, want.data(), want.size()) != 0) {
                size_t b = 0;
                while (got[b] == want[b]) ++b;
                CHECK(0, "%s [%s]: %c plane ch%u differs at byte %zu (beat %zu)", k.c->name, tag, kv ? 'V' : 'K', i, b, b / 64u);
                ok = false;
            }
        }
    }
    for (uint32_t t = 0; t < k.c->T && ok && k.has_fa; ++t) {
        const uint8_t * got = at(sim, k.dst_h, uint64_t(t) * k.dst_stride);
        const uint8_t * want = reinterpret_cast<const uint8_t *>(k.out.data() + size_t(t) * k.K);
        if (memcmp(got, want, size_t(k.K) * 4u) != 0) {
            size_t b = 0;
            while (got[b] == want[b]) ++b;
            uint32_t g = 0, w = 0;
            memcpy(&g, got + (b & ~size_t(3)), 4);
            memcpy(&w, want + (b & ~size_t(3)), 4);
            CHECK(0, "%s [%s]: output token %u differs at value %zu (head %zu): 0x%08x, expected 0x%08x", k.c->name, tag, t,
                  b / 4u, b / 4u / k.c->D, g, w);
            ok = false;
        }
    }
    if (o != nullptr) {
        // Every written DRAM beat lies in a plane (KVST's write extent is the whole plane, AT §5.8) or
        // in the dst rows.
        for (uint32_t i = 0; i < o->n_dram; ++i) {
            const ival_t & x = o->dram_wr[i];
            const bool in_plane = x.lo >= k.kB && x.hi <= k.vB + k.plane_beats;
            // inside the dst rows: every beat of [lo, hi) in some row (an ST may cover several dense rows)
            bool in_dst = x.ch == (k.dst_h >> 28) && k.has_fa;
            const uint64_t d0 = k.dst_h & 0x0FFFFFFFu;
            for (uint64_t b = x.lo; b < x.hi && in_dst; ++b) {
                const uint64_t r = b >= d0 ? (b - d0) / k.dst_stride : ~0ull;
                in_dst = r < k.c->T && b - d0 - r * k.dst_stride < k.K / 16u;
            }
            if (!in_plane && !in_dst) {
                CHECK(0, "%s [%s]: a write outside the planes and the dst rows: ch%u beats [%llu, %llu)", k.c->name, tag, x.ch,
                      (unsigned long long)x.lo, (unsigned long long)x.hi);
                ok = false;
                break;
            }
        }
    }
    return ok;
}

}  // namespace kvfa
