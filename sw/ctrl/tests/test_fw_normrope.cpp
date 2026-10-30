// test_fw_normrope.cpp — the RMS_NORM (0x14) and ROPE (0x16) kinds of W2b P2.6 (sw/ctrl/kernels/
// k_norm.cpp, k_rope.cpp; docs/design/06-software-verification.md "SW" §7.2, §7.5, §10.2: per kind, 8
// schedules, race fatal, outputs bit-exact against the C++ references sw/ctrl/ref/rmsnorm_ref.cpp and
// rope_ref.cpp). One source, two programs:
//
//   test_fw_normrope_rv IMAGE suite [--scheds N] [--only K]      (fw_kernels_normrope_rv)
//       Op lists through the `graph` executor of the RISC-V image on tsr_rv, each under the fixed
//       schedule and 7 random ones with TSR_SIM_RACE fatal: suite "decode" (U200 SPM): the decode
//       records of Qwen2.5-0.5B / 32B (RMS_NORM n 896 / 5120 with weight; ROPE NEOX theta 1e6, head_dim
//       64 / 128, q and k heads) and Gemma 4 12B (RMS_NORM 3840; per-head q / k norms with weight and
//       the v norm without, rows = n_head, n = 256 / 512; ROPE SWA head_dim 256 theta 1e4, global
//       head_dim 512 with the rope_freqs table), q_norm -> ROPE chains, positions sampled from
//       0 .. 32767, plus the other forms: NORM mode, freq_scale != 1 (FS), the A table (ATAB),
//       partial rotary (n_dims < head_dim: the raw tail copy), n % 16 != 0 (tail words), weight row
//       period > 1, strided rows, in place, a record with three predecessors (JOIN); suite "prefill"
//       (512 KiB SPM, planning budget 3,376 lines): multi-tile RMS_NORM / ROPE (double buffer, single
//       buffer, the store-first fallback), period blocks, strided tokens, a row of more than half the
//       budget, 200 ROPE tokens at positions over 0 .. 32767. Every written row is compared bit for bit
//       with the references applied in list order; DONE must be OK / RET 0; the written DRAM must be
//       exactly the destination rows; SW §4.7's observables (per-unit op streams, RISC-V SPM stores,
//       final written DRAM / SPM, DONE r0-r6) must be equal under every schedule.
//   test_fw_normrope_rv MUTANT suite --only 0 --scheds 1 --race report --expect-fail
//       (fw_kernels_normrope_rv_mutant_race) the image whose RED lacks its dep on the LD
//       (TSR_KNORM_MUTANT 1): the race detector's reports must fail the run.
//   test_fw_normrope_rv IMAGE refuse                                (fw_kernels_normrope_rv_refuse)
//       Records the kinds refuse: RET = BAD_RECORD / SCRATCH (KERNEL_ERR), nothing of the op pushed.
//   test_fw_normrope_rv IMAGE perf                                  (fw_kernels_normrope_perf)
//       RISC-V cycles of work per decode record (SW §7.8; W2b requirement <= 350 on Ibex maxperf, the
//       simulator default): lists of 16 vs 48 records of one decode shape (chained like a layer),
//       executor work up to the last op push with wait-loop iterations excluded (fw_prof.h
//       spin_filter, fw_m6_perf's metric), deep queues and U200 queues, with a per-function profile;
//       PERF PUSH_STALL must be 0. TSR_FW_PROF=1 prints the profiles.
//   test_fw_normrope_host                                            (fw_kernels_normrope_host)
//       The kinds compiled for the host (SW §6.8, ASan + UBSan): the suites' records called directly
//       (kind_ctx by hand, the preload area loaded and joined) against the references; the decode
//       fast path against the general path on twin simulators (every push, deps, push value, ticket
//       and ring state equal; deps on every unit); refused records push nothing; the 2-op RMS form
//       against the spec's 7-op lowering (docs/spec.md:311-321, VU §6.2) on edge and random rows.
// Exit 0 when everything passed, 1 on a failure, 77 (skipped) when the image is missing.
#undef _GNU_SOURCE
#include "test_sim_obs.h"

#include <sys/stat.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <vector>

#include "../ref/fw_sim_host.hpp"
#include "../ref/rmsnorm_ref.hpp"
#include "../ref/rope_ref.hpp"
#include "tsr/opl.hpp"
#include "tsr/plan.hpp"
#include "tsr_race.h"
#include "tsr_ref.h"
#include "tsr_ref_vu.h"

#if defined(TSR_FW_HOST)
#include "tsr/exec.hpp"
#include "tsr/kernel.hpp"
#include "tsr/rt.hpp"
namespace tsr::nr {
extern uint32_t rms_fast_taken, rope_fast_taken;
ticket rms_norm_slow(const volatile uint32_t * a, uint32_t n_args, uint32_t kflags, deps d, kind_ctx & ctx);
ticket rope_slow(const volatile uint32_t * a, uint32_t n_args, uint32_t kflags, deps d, kind_ctx & ctx);
}  // namespace tsr::nr
#else
#include "fw_prof.h"
#include "tsr_rv.h"
#endif

namespace {

namespace opl = tsr::opl;
namespace rr = tsr::opl::rms;
namespace ro = tsr::opl::rope;
constexpr uint32_t NONE = opl::HANDLE_NONE;
constexpr uint64_t LINE = 64u;

uint32_t H(uint32_t ch, uint64_t beat) { return (ch << 28) | uint32_t(beat & 0x0FFFFFFFu); }
uint32_t fbits(float f) { return f2u(f); }
uint32_t L16(uint32_t n) { return (n + 15u) / 16u; }

#if !defined(TSR_FW_HOST)
const char * g_image = nullptr;
uint32_t g_race = 2;                   // TSR_SIM_RACE fatal; --race report for the mutant image
#endif

// ---------------------------------------------------------------------------
// Records, suites and their data
// ---------------------------------------------------------------------------
struct rec {
    uint32_t kind;                    // opl::K_RMS_NORM / K_ROPE
    uint32_t a[14];
    std::vector<int> deps;            // indices of earlier records
    uint32_t index = 0;               // op index in the list
    std::vector<int32_t> pos;         // ROPE positions (preload entry)
    uint32_t pos_h = NONE;            // their DRAM copy
    const char * what = "";
};

struct init_blob {
    uint32_t h;
    std::vector<uint32_t> w;
};

struct suite {
    const char * name;
    tsr_config cfg;
    std::vector<rec> recs;
    std::vector<init_blob> init;
    std::vector<uint8_t> list;        // the op list
    uint32_t opl_h = H(0, 0x40000u);  // ch0 16 MiB (OPL region)
    std::vector<uint8_t> pre;         // the preload area as the executor loads it
    uint32_t list_flags = opl::F_STOP_ON_ERROR;
};

struct alloc {
    uint64_t io[4] = { 0x80000u, 0x10000u, 0x10000u, 0x10000u };   // next free beat per channel
    uint64_t take(uint32_t ch, uint64_t beats) { const uint64_t b = io[ch]; io[ch] += (beats + 63u) & ~63ull; return b; }
};

// f32 activations: rows of n values (scale per row, rare outliers), garbage in the rest of each row's
// last line (the kernels must ignore it), `stride` lines apart.
init_blob gen_rows(rng_t * r, uint32_t h, uint32_t rows, uint32_t stride, uint32_t n, int kind) {
    init_blob b{h, std::vector<uint32_t>(size_t(rows - 1u) * stride * 16u + size_t(L16(n)) * 16u, 0u)};
    for (uint32_t i = 0; i < rows; ++i) {
        const double sc = kind == 1 ? 0.1 : std::pow(10.0, double(rng_below(r, 7)) - 3.0);
        for (uint32_t j = 0; j < L16(n) * 16u; ++j) {
            uint32_t v;
            if (j >= n) v = rng_u32(r);                            // tail garbage
            else if (kind == 1) v = fbits(float(1.0 + sc * rng_normal(r)));   // weights near 1
            else v = fbits(float(rng_normal(r) * sc * (rng_below(r, 100) == 0 ? 30.0 : 1.0)));
            b.w[size_t(i) * stride * 16u + j] = v;
        }
    }
    return b;
}

// RMS_NORM record: rows of n at a new source (stride L + xpad), a new destination (L + ypad, or the
// source itself when in_place), an optional weight of `wrows` rows (period p).
rec make_rms(suite & s, alloc & al, rng_t * r, const char * what, uint32_t n, uint32_t rows, bool w, uint32_t p,
             float eps, uint32_t xch, uint32_t ych, uint32_t xpad = 0, uint32_t ypad = 0, bool in_place = false) {
    rec q{};
    q.kind = opl::K_RMS_NORM;
    q.what = what;
    const uint32_t L = L16(n);
    q.a[rr::SRC_STRIDE] = L + xpad;
    q.a[rr::SRC] = H(xch, al.take(xch, uint64_t(rows) * (L + xpad)));
    s.init.push_back(gen_rows(r, q.a[rr::SRC], rows, L + xpad, n, 0));
    if (in_place) {
        q.a[rr::DST] = q.a[rr::SRC];
        q.a[rr::DST_STRIDE] = q.a[rr::SRC_STRIDE];
    } else {
        q.a[rr::DST_STRIDE] = L + ypad;
        q.a[rr::DST] = H(ych, al.take(ych, uint64_t(rows) * (L + ypad)));
    }
    q.a[rr::N] = n;
    q.a[rr::ROWS] = rows;
    q.a[rr::EPS] = fbits(eps);
    q.a[rr::INV_N] = fbits(tsr_fdiv(1.0f, float(n)));   // SW §7.2: tsr_fdiv(1, n)
    q.a[rr::W] = NONE;
    q.a[rr::W_PERIOD] = 0u;
    if (w) {
        const uint32_t wrows = p == 1u ? 1u : (p < rows ? p : rows);
        q.a[rr::W] = H(ych, al.take(ych, uint64_t(wrows) * L));
        q.a[rr::W_PERIOD] = p;
        s.init.push_back(gen_rows(r, q.a[rr::W], wrows, L, n, 1));
    }
    return q;
}

// ROPE models (VU §6.7; test_ref_vu.c's table): base, head_dim, n_dims, NEOX, ff kind (0 none, 1
// Gemma 4 proportional: 1 for the first quarter, 1e30 after; 2 ramp 1 .. 8; 3 YaRN-like A table).
struct rope_model {
    const char * name;
    float base;
    uint32_t hd, nd;
    bool neox;
    int ff;
    float freq_scale, attn;
};
const rope_model QWEN05 = { "Qwen2.5-0.5B", 1e6f, 64, 64, true, 0, 1.0f, 1.0f };
const rope_model QWEN32 = { "Qwen2.5-32B", 1e6f, 128, 128, true, 0, 1.0f, 1.0f };
const rope_model GEMMA_SWA = { "Gemma 4 SWA", 1e4f, 256, 256, true, 0, 1.0f, 1.0f };
const rope_model GEMMA_GLOBAL = { "Gemma 4 global", 1e6f, 512, 512, true, 1, 1.0f, 1.0f };
const rope_model NORM_RAMP = { "NORM ramp ff", 5e5f, 128, 128, false, 2, 1.0f, 1.0f };
const rope_model LINEAR = { "linear freq_scale 0.25", 1e6f, 128, 128, true, 0, 0.25f, 1.0f };
const rope_model YARN = { "A table (YaRN form)", 1e6f, 128, 128, true, 3, 1.0f, 1.1f };
const rope_model PARTIAL = { "partial NEOX 64 of 128", 1e4f, 128, 64, true, 0, 1.0f, 1.0f };
const rope_model PARTIAL_NORM = { "partial NORM 16 of 64", 1e4f, 64, 16, false, 0, 0.5f, 0.75f };

std::vector<uint32_t> rope_table(const rope_model & m) {
    const uint32_t half = m.nd / 2u;
    std::vector<uint32_t> t(half);
    for (uint32_t i = 0; i < half; ++i) {
        float v = 1.0f;
        if (m.ff == 1) v = i < half / 4u ? 1.0f : 1e30f;            // conversion/gemma.py, partial_rotary 0.25
        else if (m.ff == 2) v = 1.0f + 7.0f * float(i) / float(half);
        else if (m.ff == 3) {                                         // A_i in double, one rounding (VU §6.7)
            const double th = std::pow(double(m.base), -2.0 * i / double(m.nd));
            const double ramp = std::min(1.0, std::max(0.0, (double(i) - 8.0) / 24.0));
            v = float(th * (ramp * 0.25 + (1.0 - ramp)));
        }
        t[i] = fbits(v);
    }
    return t;
}

// ROPE record: T tokens of nh heads; positions as given (empty: random over 0 .. 32767);
// src_from: an earlier record whose destination rows are this record's source (a chain).
rec make_rope(suite & s, alloc & al, rng_t * r, const char * what, const rope_model & m, uint32_t nh, uint32_t T,
              uint32_t xch, uint32_t ych, uint32_t xpad = 0, uint32_t ypad = 0, std::vector<int32_t> pos = {},
              int src_from = -1, bool in_place = false, uint32_t pos_h = NONE) {
    rec q{};
    q.kind = opl::K_ROPE;
    q.what = what;
    const uint32_t Hl = nh * m.hd / 16u;
    if (src_from >= 0) {
        const rec & p = s.recs[size_t(src_from)];
        q.a[ro::SRC] = p.a[rr::DST];
        q.a[ro::SRC_STRIDE] = p.a[rr::DST_STRIDE];
        q.deps.push_back(src_from);
    } else {
        q.a[ro::SRC_STRIDE] = Hl + xpad;
        q.a[ro::SRC] = H(xch, al.take(xch, uint64_t(T) * (Hl + xpad)));
        s.init.push_back(gen_rows(r, q.a[ro::SRC], T, Hl + xpad, Hl * 16u, 0));
    }
    if (in_place) {
        q.a[ro::DST] = q.a[ro::SRC];
        q.a[ro::DST_STRIDE] = q.a[ro::SRC_STRIDE];
    } else {
        q.a[ro::DST_STRIDE] = Hl + ypad;
        q.a[ro::DST] = H(ych, al.take(ych, uint64_t(T) * (Hl + ypad)));
    }
    q.a[ro::TABLE] = NONE;
    if (m.ff != 0) {
        q.a[ro::TABLE] = H(ych, al.take(ych, L16(m.nd / 2u)));
        s.init.push_back(init_blob{q.a[ro::TABLE], rope_table(m)});
    }
    q.a[ro::TS] = fbits(powf(m.base, -2.0f / float(m.nd)));          // host libm, as ggml-cpu (VU §6.7)
    q.a[ro::M] = fbits(m.attn);
    q.a[ro::FREQ_SCALE] = fbits(m.freq_scale);
    q.a[ro::HEAD_DIM] = m.hd;
    q.a[ro::N_DIMS] = m.nd;
    q.a[ro::N_HEADS] = nh;
    q.a[ro::N_TOKENS] = T;
    q.a[ro::MODE] = (m.neox ? ro::MODE_NEOX : 0u) | (m.ff == 3 ? ro::MODE_ATAB : 0u);
    if (pos.empty()) {
        for (uint32_t t = 0; t < T; ++t) pos.push_back(int32_t(rng_below(r, 32768u)));
    }
    q.pos = pos;
    if (pos_h != NONE) {                   // an earlier record's positions (one preload entry, as a decode list)
        q.pos_h = pos_h;
        return q;
    }
    q.pos_h = H(0, al.take(0, L16(T)));
    std::vector<uint32_t> pw(size_t(L16(T)) * 16u, 0u);
    for (uint32_t t = 0; t < T; ++t) pw[t] = uint32_t(pos[t]);
    s.init.push_back(init_blob{q.pos_h, pw});
    return q;
}

// The op list (positions as preload entries, in record order) and the preload image.
bool build_list(suite & s) {
    opl::writer w(s.list_flags, 0x4E52);
    std::vector<uint32_t> pre_byte(s.recs.size(), 0u);
    for (size_t i = 0; i < s.recs.size(); ++i) {
        if (s.recs[i].kind != opl::K_ROPE) continue;
        size_t k = 0;                      // records sharing a positions copy share its entry
        while (k < i && !(s.recs[k].kind == opl::K_ROPE && s.recs[k].pos_h == s.recs[i].pos_h)) ++k;
        pre_byte[i] = k < i ? pre_byte[k] : w.add_preload(s.recs[i].pos_h, 4u * uint32_t(s.recs[i].pos.size()));
    }
    s.pre.assign(size_t(w.pre_lines()) * LINE, 0u);
    for (size_t i = 0; i < s.recs.size(); ++i) {
        rec & q = s.recs[i];
        std::vector<uint32_t> d;
        for (int k : q.deps) d.push_back(s.recs[size_t(k)].index);
        if (q.kind == opl::K_ROPE) {
            q.a[ro::POS] = pre_byte[i];
            for (size_t t = 0; t < q.pos.size(); ++t) memcpy(s.pre.data() + pre_byte[i] + 4u * t, &q.pos[t], 4u);
            q.index = w.add(opl::K_ROPE, q.a, ro::NARGS, d.data(), uint32_t(d.size()));
        } else {
            q.index = w.add(opl::K_RMS_NORM, q.a, rr::NARGS, d.data(), uint32_t(d.size()));
        }
    }
    s.list = w.finish();
    CHECK(w.ok() && !s.list.empty(), "%s: OPL writer: %s", s.name, w.error());
    return w.ok();
}

void fill(tsr_sim * sim, const suite & s) {
    for (const auto & b : s.init)
        tsr_sim_dram_write(sim, b.h >> 28, uint64_t(b.h & 0x0FFFFFFFu) * LINE, b.w.data(), b.w.size() * 4u);
    if (!s.list.empty())
        tsr_sim_dram_write(sim, s.opl_h >> 28, uint64_t(s.opl_h & 0x0FFFFFFFu) * LINE, s.list.data(), s.list.size());
}

tsr::ref::dram_view view_of(tsr_sim * sim, std::vector<uint8_t *> & chp) {
    const tsr_config * c = tsr_sim_config(sim);
    chp.resize(c->num_ch);
    for (uint32_t i = 0; i < c->num_ch; ++i) chp[i] = tsr_sim_dram(sim, i);
    return tsr::ref::dram_view{chp.data(), c->num_ch, c->ch_bytes};
}

// The references applied in list order to sim's DRAM.
bool apply_refs(tsr_sim * sim, const suite & s) {
    std::vector<uint8_t *> chp;
    const tsr::ref::dram_view v = view_of(sim, chp);
    bool ok = true;
    for (const auto & q : s.recs) {
        const int rc = q.kind == opl::K_ROPE ? tsr::ref::rope_ref(v, q.a, s.pre.data(), uint32_t(s.pre.size()))
                                             : tsr::ref::rmsnorm_ref(v, q.a);
        CHECK(rc == tsr::ref::NR_OK, "%s: reference refused record %u (%s): %d", s.name, q.index, q.what, rc);
        ok = ok && rc == tsr::ref::NR_OK;
    }
    return ok;
}

// The destination rows of a record: (handle, rows, stride, lines per row).
void dst_rows(const rec & q, uint32_t * h, uint32_t * rows, uint32_t * stride, uint32_t * lines) {
    if (q.kind == opl::K_ROPE) {
        *h = q.a[ro::DST]; *rows = q.a[ro::N_TOKENS]; *stride = q.a[ro::DST_STRIDE];
        *lines = q.a[ro::N_HEADS] * q.a[ro::HEAD_DIM] / 16u;
    } else {
        *h = q.a[rr::DST]; *rows = q.a[rr::ROWS]; *stride = q.a[rr::DST_STRIDE]; *lines = L16(q.a[rr::N]);
    }
}

// Every destination row of the suite equal in the two simulators' DRAM; returns the mismatches.
uint32_t compare_rows(const suite & s, tsr_sim * got_sim, tsr_sim * want_sim, const char * tag) {
    uint32_t bad = 0;
    for (const auto & q : s.recs) {
        uint32_t h, rows, stride, lines;
        dst_rows(q, &h, &rows, &stride, &lines);
        for (uint32_t i = 0; i < rows; ++i) {
            const uint64_t off = (uint64_t(h & 0x0FFFFFFFu) + uint64_t(i) * stride) * LINE;
            const uint8_t * g = tsr_sim_dram(got_sim, h >> 28) + off;
            const uint8_t * w = tsr_sim_dram(want_sim, h >> 28) + off;
            if (memcmp(g, w, size_t(lines) * LINE) != 0) {
                size_t k = 0;
                while (g[k] == w[k]) ++k;
                CHECK(0, "%s [%s]: op %u (%s) row %u differs at byte %zu: 0x%08x, reference 0x%08x", s.name, tag, q.index, q.what,
                      i, k, get_u32_le(g + (k & ~size_t(3))), get_u32_le(w + (k & ~size_t(3))));
                ++bad;
                break;
            }
        }
    }
    return bad;
}

std::vector<ival_t> merge(std::vector<ival_t> v) {
    std::sort(v.begin(), v.end(), [](const ival_t & a, const ival_t & b) { return a.ch != b.ch ? a.ch < b.ch : a.lo < b.lo; });
    std::vector<ival_t> m;
    for (const auto & x : v) {
        if (!m.empty() && m.back().ch == x.ch && x.lo <= m.back().hi) m.back().hi = std::max(m.back().hi, x.hi);
        else m.push_back(x);
    }
    return m;
}
std::vector<ival_t> written_rows(const suite & s) {
    std::vector<ival_t> v;
    for (const auto & q : s.recs) {
        uint32_t h, rows, stride, lines;
        dst_rows(q, &h, &rows, &stride, &lines);
        for (uint32_t i = 0; i < rows; ++i) {
            const uint64_t lo = (h & 0x0FFFFFFFu) + uint64_t(i) * stride;
            v.push_back(ival_t{h >> 28, lo, lo + lines});
        }
    }
    return merge(v);
}

tsr_config base_cfg(uint32_t spm_bytes) {
    tsr_config c = TSR_CONFIG_U200_V1;
    c.t_max = 8;
    c.spm_bytes = spm_bytes;
    c.ch_bytes = 1ull << 30;
    c.caps = TSR_SIM_CAPS_IMPLEMENTED;
    return c;
}

// ---------------------------------------------------------------------------
// Suites
// ---------------------------------------------------------------------------
suite suite_decode() {
    suite s{};
    s.name = "decode";
    s.cfg = base_cfg(4u << 20);
    rng_t r = { 0xDEC0DEull };
    alloc al;
    auto add = [&](rec q) { s.recs.push_back(std::move(q)); return int(s.recs.size() - 1u); };
    const float eps6 = 1e-6f;
    // Qwen2.5-0.5B: attn_norm, then ROPE of q (14 heads) and k (2 heads).
    const int q05 = add(make_rms(s, al, &r, "qwen05 attn_norm", 896, 1, true, 1, eps6, 0, 1));
    { rec q = make_rope(s, al, &r, "qwen05 rope q", QWEN05, 14, 1, 1, 2, 0, 0, {0}); q.deps = {q05}; add(q); }
    add(make_rope(s, al, &r, "qwen05 rope k", QWEN05, 2, 1, 2, 3, 0, 0, {32767}));
    add(make_rms(s, al, &r, "qwen05 ffn_norm", 896, 1, true, 1, eps6, 3, 0));
    // Qwen2.5-32B.
    add(make_rms(s, al, &r, "qwen32 attn_norm", 5120, 1, true, 1, eps6, 1, 2));
    add(make_rope(s, al, &r, "qwen32 rope q", QWEN32, 40, 1, 2, 3, 0, 0, {1}));
    add(make_rope(s, al, &r, "qwen32 rope k", QWEN32, 8, 1, 3, 0, 0, 0, {12345}));
    // Gemma 4 12B: attn_norm; SWA layer: q / k norms per head with weight, v norm without, ROPE of the
    // normed q / k (chains); global layer: head_dim 512 norms, ROPE with rope_freqs.
    add(make_rms(s, al, &r, "gemma attn_norm", 3840, 1, true, 1, eps6, 0, 1));
    const int qn = add(make_rms(s, al, &r, "gemma swa q_norm (16 heads)", 256, 16, true, 1, eps6, 1, 2));
    const int kn = add(make_rms(s, al, &r, "gemma swa k_norm (8 heads)", 256, 8, true, 1, eps6, 2, 3));
    add(make_rms(s, al, &r, "gemma swa v_norm (8 heads, no weight)", 256, 8, false, 0, eps6, 3, 0));
    add(make_rope(s, al, &r, "gemma swa rope q", GEMMA_SWA, 16, 1, 0, 1, 0, 0, {4097}, qn));
    add(make_rope(s, al, &r, "gemma swa rope k (in place)", GEMMA_SWA, 8, 1, 0, 1, 0, 0, {4097}, kn, true));
    const int gq = add(make_rms(s, al, &r, "gemma global q_norm (16 x 512)", 512, 16, true, 1, eps6, 1, 2));
    const int gk = add(make_rms(s, al, &r, "gemma global k_norm (1 x 512)", 512, 1, true, 1, eps6, 2, 3));
    add(make_rope(s, al, &r, "gemma global rope q (rope_freqs)", GEMMA_GLOBAL, 16, 1, 0, 0, 0, 0, {30000}, gq));
    add(make_rope(s, al, &r, "gemma global rope k (rope_freqs)", GEMMA_GLOBAL, 1, 1, 0, 0, 0, 0, {30000}, gk));
    // Other forms.
    add(make_rope(s, al, &r, "NORM mode, ramp freq factors", NORM_RAMP, 8, 1, 1, 2));
    add(make_rope(s, al, &r, "freq_scale 0.25 (FS)", LINEAR, 8, 1, 2, 3));
    add(make_rope(s, al, &r, "A table (ATAB), M 1.1", YARN, 4, 1, 3, 0));
    add(make_rope(s, al, &r, "partial NEOX (tail copy)", PARTIAL, 6, 1, 0, 1));
    add(make_rope(s, al, &r, "partial NORM, FS, M 0.75", PARTIAL_NORM, 3, 1, 1, 2));
    add(make_rope(s, al, &r, "decode q, three tokens dense", QWEN05, 14, 3, 2, 3));
    add(make_rms(s, al, &r, "n 100 (tail words)", 100, 3, true, 1, 1e-5f, 0, 2));
    add(make_rms(s, al, &r, "n 16, eps 1e-5, no weight", 16, 1, false, 0, 1e-5f, 1, 1));
    add(make_rms(s, al, &r, "weight period 3, 7 rows", 64, 7, true, 3, eps6, 2, 0));
    add(make_rms(s, al, &r, "weight period 8 >= rows", 64, 5, true, 8, eps6, 3, 1));
    add(make_rms(s, al, &r, "strided rows (general path)", 896, 4, true, 1, eps6, 0, 1, 3, 2));
    add(make_rms(s, al, &r, "in place", 896, 2, true, 1, eps6, 1, 1, 0, 0, true));
    add(make_rope(s, al, &r, "strided tokens (general path)", QWEN32, 8, 3, 2, 0, 5, 1));
    {   // three predecessors: JOIN records
        rec q = make_rms(s, al, &r, "three predecessors (JOIN)", 896, 1, true, 1, eps6, 2, 3);
        q.deps = { 0, 4, 7 };
        add(q);
    }
    return s;
}

suite suite_prefill() {
    suite s{};
    s.name = "prefill";
    s.cfg = base_cfg(512u << 10);                      // 8,192 lines: planning budget 3,376
    rng_t r = { 0x9BEF11ull };
    alloc al;
    auto add = [&](rec q) { s.recs.push_back(std::move(q)); return int(s.recs.size() - 1u); };
    add(make_rms(s, al, &r, "rms 896 x 200 (tiles of 33)", 896, 200, true, 1, 1e-6f, 0, 1));
    add(make_rms(s, al, &r, "rms 896 x 70 strided", 896, 70, true, 1, 1e-6f, 1, 2, 2, 1));
    add(make_rms(s, al, &r, "rms period 16, 70 rows", 896, 70, true, 16, 1e-6f, 2, 3));
    add(make_rms(s, al, &r, "rms period 50, 120 rows (tiles inside blocks)", 896, 120, true, 50, 1e-6f, 3, 0));
    add(make_rms(s, al, &r, "per-head 256 x 320", 256, 320, true, 1, 1e-6f, 0, 2));
    add(make_rms(s, al, &r, "no weight 5120 x 9", 5120, 9, false, 0, 1e-5f, 1, 3));
    add(make_rms(s, al, &r, "rows of 2,000 lines (single buffer)", 32000, 3, false, 0, 1e-6f, 2, 0));
    {
        std::vector<int32_t> pos;
        for (uint32_t t = 0; t < 200u; ++t) pos.push_back(t < 3u ? int32_t(t) : t == 199u ? 32767 : int32_t(rng_below(&r, 32768u)));
        add(make_rope(s, al, &r, "rope 14 x 64, 200 tokens over 0 .. 32767", QWEN05, 14, 200, 0, 1, 0, 0, pos));
    }
    add(make_rope(s, al, &r, "rope strided tokens, 50", QWEN05, 14, 50, 1, 2, 3, 5));
    add(make_rope(s, al, &r, "rope table 4 x 512, 40 tokens (single buffer)", GEMMA_GLOBAL, 4, 40, 2, 3));
    add(make_rope(s, al, &r, "rope NORM 8 x 128, 33 tokens", NORM_RAMP, 8, 33, 3, 0));
    add(make_rope(s, al, &r, "rope ATAB 8 x 128, 64 tokens", YARN, 8, 64, 0, 2));
    {   // positions 0 .. 32767 in steps: 16 tokens each of a stride
        std::vector<int32_t> pos;
        for (uint32_t t = 0; t < 64u; ++t) pos.push_back(int32_t(t * 512u + (t & 7u)));
        add(make_rope(s, al, &r, "rope Gemma SWA 2 x 256, 64 tokens", GEMMA_SWA, 2, 64, 1, 3, 0, 0, pos));
    }
    return s;
}

#if !defined(TSR_FW_HOST)
// ---------------------------------------------------------------------------
// RISC-V image: suites under 8 schedules
// ---------------------------------------------------------------------------
struct run_obs {
    obs o;
    uint32_t done[7];
};

bool run_suite(suite & s, const char * sched, tsr_sim * want, run_obs * out) {
    tsr_sim_opts so;
    memset(&so, 0, sizeof(so));
    so.sched = sched;
    so.race = g_race;
    so.threads = 4;
    tsr_sim * sim = tsr_sim_create_opts(&s.cfg, &so);
    CHECK(sim != nullptr, "%s: simulator", s.name);
    if (sim == nullptr) return false;
    fill(sim, s);
    obs_attach(&out->o, sim);
    fwh h;
    bool ok = fwh_open(&h, sim, g_image) == 0;
    CHECK(ok, "%s [%s]: boot", s.name, sched);
    const int gi = ok ? fwh_kernel(&h, "graph") : -1;
    if (ok) {
        const uint32_t a[7] = { s.opl_h, uint32_t(s.list.size()), opl::LF_VERIFY_CRC, 0, 0, 0, 0 };
        ok = gi >= 0 && fwh_launch(&h, uint32_t(gi), a, 0) == 0;
        CHECK(ok, "%s [%s]: launch", s.name, sched);
    }
    if (ok) {
        CHECK(TSR_GET(h.done[0], TSR_LCQ_R0_TYPE) == TSR_LCQ_R0_TYPE_DONE && fwh_status(&h) == TSR_LCQ_R0_STATUS_OK &&
                  fwh_ret(&h) == 0,
              "%s [%s]: DONE 0x%08x RET 0x%08x (err ticket 0x%08x info 0x%08x w0 0x%08x pidx %u)", s.name, sched, h.done[0],
              h.done[2], h.done[3], h.done[4], h.done[5], h.done[6]);
        for (int i = 0; i < 7; ++i) out->done[i] = h.done[i];
        compare_rows(s, sim, want, sched);
        obs_finish(&out->o, sim);
        std::vector<ival_t> got = merge(std::vector<ival_t>(out->o.dram_wr, out->o.dram_wr + out->o.n_dram));
        const std::vector<ival_t> wr = written_rows(s);
        bool same = got.size() == wr.size();
        for (size_t i = 0; same && i < got.size(); ++i) same = got[i].ch == wr[i].ch && got[i].lo == wr[i].lo && got[i].hi == wr[i].hi;
        CHECK(same, "%s [%s]: the written DRAM (%zu ranges) is not exactly the destination rows (%zu ranges)", s.name, sched,
              got.size(), wr.size());
        const tsr_race_report * rp = nullptr;
        const size_t nr = tsr_sim_race_reports(sim, &rp);
        CHECK(nr == 0, "%s [%s]: %zu race reports", s.name, sched, nr);
        printf("%s [%s]: %zu records, %llu core cycles, %llu RV instructions\n", s.name, sched, s.recs.size(),
               (unsigned long long)h.cycles, (unsigned long long)out->o.instret);
    }
    fwh_close(&h);
    tsr_sim_destroy(sim);
    return ok;
}

void run_scheds(suite & s, uint32_t n_sched) {
    if (!build_list(s)) return;
    {   // The plans of the suite's records on its device (plan_norm.hpp): tiles per record.
        const tsr::plan::dev_cfg dc{s.cfg.spm_bytes / 64u, s.cfg.t_max, s.cfg.k_max, s.cfg.num_ch, 0u, 16u,
                                    tsr::plan::CAPS_VU};
        const tsr::plan::plan_limits lim = tsr::plan::plan_limits_of(dc);
        printf("%s: budget %u lines; tiles per record:", s.name, lim.budget);
        uint32_t multi = 0;
        for (const auto & q : s.recs) {
            tsr::kind_plan kp{};
            const bool ok = q.kind == opl::K_ROPE ? tsr::plan_rope(q.a, ro::NARGS, lim, &kp)
                                                  : tsr::plan_rms_norm(q.a, rr::NARGS, lim, &kp);
            CHECK(ok, "%s: record %u (%s) does not plan (status 0x%08x)", s.name, q.index, q.what, kp.status);
            printf(" %u", ok ? kp.n_tiles : 0u);
            multi += ok && kp.n_tiles > 1u;
        }
        printf(" (%u multi-tile)\n", multi);
    }
    // The expected DRAM: the same inputs, the references in list order.
    tsr_sim_opts so;
    memset(&so, 0, sizeof(so));
    so.sched = "fixed";
    tsr_sim * want = tsr_sim_create_opts(&s.cfg, &so);
    fill(want, s);
    if (!apply_refs(want, s)) {
        tsr_sim_destroy(want);
        return;
    }
    std::vector<run_obs> runs(n_sched);
    for (uint32_t k = 0; k < n_sched; ++k) {
        char sched[64];
        if (k == 0) snprintf(sched, sizeof(sched), "fixed");
        else snprintf(sched, sizeof(sched), "random:%u", 1000u + k);
        const bool ok = run_suite(s, sched, want, &runs[k]);
        if (k > 0 && ok) {
            char what[128];
            snprintf(what, sizeof(what), "%s: %s vs fixed", s.name, sched);
            obs_equal(&runs[0].o, &runs[k].o, what);
            CHECK(memcmp(runs[0].done, runs[k].done, sizeof(runs[0].done)) == 0, "%s: DONE r0-r6 differ", what);
        }
    }
    for (auto & x : runs) obs_free(&x.o);
    tsr_sim_destroy(want);
}

// Refused records through the executor: RET = the code (KERNEL_ERR), and nothing of the op pushed
// (the op's pidx range is empty: the next push index after the list's start is the FENCE / nothing).
void run_refusals() {
    struct bad { const char * what; uint32_t kind; int field; uint32_t value; int field2; uint32_t value2; uint32_t want; };
    const bad bads[] = {
        { "RMS_NORM n 0", opl::K_RMS_NORM, rr::N, 0u, -1, 0u, opl::RET_BAD_RECORD },
        { "RMS_NORM n 2^24", opl::K_RMS_NORM, rr::N, 1u << 24, rr::ROWS, 1u, opl::RET_BAD_RECORD },
        { "RMS_NORM rows 0", opl::K_RMS_NORM, rr::ROWS, 0u, -1, 0u, opl::RET_BAD_RECORD },
        { "RMS_NORM weight period 0", opl::K_RMS_NORM, rr::W_PERIOD, 0u, -1, 0u, opl::RET_BAD_RECORD },
        { "RMS_NORM dst stride < L, 2 rows", opl::K_RMS_NORM, rr::DST_STRIDE, 55u, -1, 0u, opl::RET_BAD_RECORD },
        { "RMS_NORM a row beyond the budget", opl::K_RMS_NORM, rr::N, 16u * 70000u, rr::ROWS, 1u, opl::RET_SCRATCH },
        { "ROPE head_dim 24", opl::K_ROPE, ro::HEAD_DIM, 24u, -1, 0u, opl::RET_BAD_RECORD },
        { "ROPE head_dim 2048", opl::K_ROPE, ro::HEAD_DIM, 2048u, -1, 0u, opl::RET_BAD_RECORD },
        { "ROPE NEOX n_dims 48", opl::K_ROPE, ro::N_DIMS, 48u, -1, 0u, opl::RET_BAD_RECORD },
        { "ROPE mode bit 2", opl::K_ROPE, ro::MODE, 5u, -1, 0u, opl::RET_BAD_RECORD },
        { "ROPE ATAB without table", opl::K_ROPE, ro::MODE, 3u, -1, 0u, opl::RET_BAD_RECORD },
        { "ROPE positions beyond the preload area", opl::K_ROPE, ro::N_TOKENS, 17u, -1, 0u, opl::RET_BAD_RECORD },
        { "ROPE position offset not line-aligned", opl::K_ROPE, ro::POS, 4u, -1, 0u, opl::RET_BAD_RECORD },
        { "ROPE n_heads 0", opl::K_ROPE, ro::N_HEADS, 0u, -1, 0u, opl::RET_BAD_RECORD },
        { "kind flags", opl::K_RMS_NORM, -2, 1u, -1, 0u, opl::RET_BAD_RECORD },
    };
    for (const auto & b : bads) {
        suite s{};
        s.name = b.what;
        s.cfg = base_cfg(4u << 20);
        rng_t r = { 0xBADull };
        alloc al;
        // A good first record (it runs), then the refused one.
        s.recs.push_back(make_rms(s, al, &r, "good", 896, 1, true, 1, 1e-6f, 0, 1));
        rec q = b.kind == opl::K_ROPE ? make_rope(s, al, &r, "refused", QWEN05, 2, 16, 1, 2)
                                      : make_rms(s, al, &r, "refused", 896, 2, true, 1, 1e-6f, 1, 2);
        if (b.field >= 0) q.a[b.field] = b.value;
        if (b.field2 >= 0) q.a[b.field2] = b.value2;
        s.recs.push_back(q);
        // Build by hand: kind flags and the preload area of exactly 16 positions.
        opl::writer w(s.list_flags, 0xBAD);
        const uint32_t pb = s.recs[1].kind == opl::K_ROPE ? w.add_preload(s.recs[1].pos_h, 64u) : 0u;
        w.add(opl::K_RMS_NORM, s.recs[0].a, rr::NARGS);
        if (s.recs[1].kind == opl::K_ROPE) {
            if (b.field != ro::POS) s.recs[1].a[ro::POS] = pb;
            w.add(opl::K_ROPE, s.recs[1].a, ro::NARGS);
        } else {
            w.add(opl::K_RMS_NORM, s.recs[1].a, rr::NARGS, nullptr, 0, b.field == -2 ? b.value : 0u);
        }
        s.list = w.finish();
        tsr_sim_opts so;
        memset(&so, 0, sizeof(so));
        so.sched = "fixed";
        so.race = 2;
        tsr_sim * sim = tsr_sim_create_opts(&s.cfg, &so);
        fill(sim, s);
        fwh h;
        bool ok = fwh_open(&h, sim, g_image) == 0;
        const int gi = ok ? fwh_kernel(&h, "graph") : -1;
        uint32_t ndma = 0, nvu = 0;
        struct cnt { uint32_t * d, * v; } c{&ndma, &nvu};
        tsr_sim_add_listener(sim, [](void * ctx, tsr_sim *, const tsr_q_event * ev) {
            if (ev->kind != TSR_QEV_PUSH) return;
            cnt * k = static_cast<cnt *>(ctx);
            if (ev->unit == TSR_UNIT_DMA) ++*k->d;
            if (ev->unit == TSR_UNIT_VU) ++*k->v;
        }, &c);
        if (ok) {
            const uint32_t a[7] = { s.opl_h, uint32_t(s.list.size()), 0, 0, 0, 0, 0 };
            ok = gi >= 0 && fwh_launch(&h, uint32_t(gi), a, 0) == 0;
        }
        // The good record pushes LD x, LD w, 2 MOVI, RED, VOP, ST (3 DMA + 4 VU); the executor's list
        // LD (+ the preload LD) are DMA too.
        const uint32_t want_dma = 3u + 1u + (s.recs[1].kind == opl::K_ROPE ? 1u : 0u);
        CHECK(ok && fwh_status(&h) == TSR_LCQ_R0_STATUS_KERNEL_ERR && uint32_t(fwh_ret(&h)) == b.want && ndma == want_dma &&
                  nvu == 4u,
              "refuse %s: status %u RET 0x%08x (expected KERNEL_ERR 0x%08x), %u DMA / %u VU pushes (expected %u / 4)", b.what,
              fwh_status(&h), uint32_t(fwh_ret(&h)), b.want, ndma, nvu, want_dma);
        fwh_close(&h);
        tsr_sim_destroy(sim);
    }
    printf("refuse: %zu refused records checked\n", sizeof(bads) / sizeof(bads[0]));
}

// ---------------------------------------------------------------------------
// Perf: RISC-V cycles of work per decode record (fw_m6_perf's metric)
// ---------------------------------------------------------------------------
struct perf_ctx {
    tsr_rv * rv = nullptr;
    uint64_t last = 0, busy = 0, work = 0, work_insns = 0, work_at_push = 0, insns_at_push = 0;
    uint32_t last_w0 = 0;
    fwprof::spin_filter spin;
    fwprof::prof * prof = nullptr;
};
int perf_retire(void * ctx, const tsr_rv_retire * r) {
    perf_ctx * p = static_cast<perf_ctx *>(ctx);
    const uint64_t now = tsr_rv_cycles(p->rv);
    const uint64_t d = now - p->last;
    p->last = now;
    const bool spin = p->spin.step(r);
    if (d <= 256u) {
        p->busy += d;
        if (!spin) {
            p->work += d;
            p->work_insns++;
        }
        if (p->prof) p->prof->add(r, d, spin);
    }
    if (r->mem_write && r->mem_size == 4u && r->mem_addr == 0x20000000u + TSR_CW_ISSUE_W(0)) p->last_w0 = r->mem_val;
    if (r->mem_write && r->mem_size == 4u && r->mem_addr == 0x20000000u + TSR_CW_ISSUE_PUSH &&
        (p->last_w0 & 0xFFu) != TSR_OP_FENCE) {
        p->work_at_push = p->work;
        p->insns_at_push = p->work_insns;
    }
    return 0;
}

struct perf_shape {
    const char * name;
    bool rope;
    uint32_t n, rows;      // RMS_NORM
    bool w;
    const rope_model * m;  // ROPE
    uint32_t nh;
};

// One launch of `count` records of the shape (each depending on the previous one, as in a layer
// chain); returns {work cycles, instructions} up to the last op push, the launch's core cycles
// (descriptor push to DONE) and F_RV / F_CORE.
std::array<double, 4> perf_run(const perf_shape & sh, uint32_t count, bool deep, fwprof::prof * prof, bool * stall_ok) {
    suite s{};
    s.name = sh.name;
    s.cfg = base_cfg(4u << 20);
    rng_t r = { 0x9E7Full };
    alloc al;
    uint32_t pos_h = NONE;                 // one positions entry for the list (a decode token's position)
    for (uint32_t i = 0; i < count; ++i) {
        rec q = sh.rope ? make_rope(s, al, &r, "perf", *sh.m, sh.nh, 1, i & 3u, (i + 1u) & 3u, 0, 0, {1234}, -1, false, pos_h)
                        : make_rms(s, al, &r, "perf", sh.n, sh.rows, sh.w, 1, 1e-6f, i & 3u, (i + 1u) & 3u);
        if (i > 0) q.deps = { int(i) - 1 };
        if (sh.rope) pos_h = q.pos_h;
        s.recs.push_back(q);
    }
    if (!build_list(s)) return {0, 0, 0, 1};
    tsr_sim_opts so;
    memset(&so, 0, sizeof(so));
    so.sched = "fixed";
    so.threads = 4;
    static const uint8_t deep_q[6] = { 31, 31, 31, 31, 0, 31 };
    if (deep) so.q_depth = deep_q;
    tsr_sim * sim = tsr_sim_create_opts(&s.cfg, &so);
    fill(sim, s);
    fwh h;
    if (fwh_open(&h, sim, g_image) != 0) {
        CHECK(0, "perf: boot");
        tsr_sim_destroy(sim);
        return {0, 0, 0, 1};
    }
    perf_ctx p;
    p.rv = tsr_rv_agent_rv(tsr_sim_rv_agent(sim));
    p.last = tsr_rv_cycles(p.rv);
    p.prof = prof;
    tsr_rv_agent_set_retire_hook(tsr_sim_rv_agent(sim), perf_retire, &p);
    const int gi = fwh_kernel(&h, "graph");
    const uint32_t a[7] = { s.opl_h, uint32_t(s.list.size()), 0, 0, 0, 0, 0 };
    const uint64_t x0 = p.work, i0 = p.work_insns;
    const bool ok = gi >= 0 && fwh_launch(&h, uint32_t(gi), a, 0) == 0 && fwh_status(&h) == TSR_LCQ_R0_STATUS_OK && fwh_ret(&h) == 0;
    CHECK(ok, "perf %s: launch DONE 0x%08x RET 0x%08x", sh.name, h.done[0], h.done[2]);
    tsr_rv_agent_set_retire_hook(tsr_sim_rv_agent(sim), nullptr, nullptr);
    const uint64_t stall = tsr_sim_perf(sim, TSR_PERF_PUSH_STALL(tsr_sim_config(sim)->num_ch));
    if (stall != 0u) *stall_ok = false;
    const double ratio = tsr_sim_rv_ratio(sim);
    fwh_close(&h);
    tsr_sim_destroy(sim);
    return { double(p.work_at_push - x0), double(p.insns_at_push - i0), double(h.cycles), ratio };
}

int perf_main() {
    const char * core = getenv("TSR_SIM_RV_CORE");
    printf("fw_kernels_normrope_perf: RISC-V timing model Ibex %s; work = RV cycles up to the last op push, wait-loop "
           "iterations after their first test excluded (fw_prof.h); per record = (48 - 16 records) / 32\n",
           core && core[0] ? core : tsr_rv_core_name(TSR_RV_CORE_DEFAULT));
    const perf_shape shapes[] = {
        { "RMS_NORM Qwen2.5-0.5B 896 x 1, weight", false, 896, 1, true, nullptr, 0 },
        { "RMS_NORM Qwen2.5-32B 5120 x 1, weight", false, 5120, 1, true, nullptr, 0 },
        { "RMS_NORM Gemma q_norm 256 x 16, weight", false, 256, 16, true, nullptr, 0 },
        { "RMS_NORM Gemma v_norm 256 x 8, no weight", false, 256, 8, false, nullptr, 0 },
        { "ROPE Qwen2.5-0.5B q 14 x 64", true, 0, 0, false, &QWEN05, 14 },
        { "ROPE Qwen2.5-32B k 8 x 128", true, 0, 0, false, &QWEN32, 8 },
        { "ROPE Gemma global q 16 x 512, rope_freqs", true, 0, 0, false, &GEMMA_GLOBAL, 16 },
        { "ROPE freq_scale 0.25 (3 MOVI)", true, 0, 0, false, &LINEAR, 8 },
    };
    const bool want_prof = getenv("TSR_FW_PROF") && getenv("TSR_FW_PROF")[0] == '1';
    bool stall_ok = true;
    double worst = 0;
    for (int deep = 1; deep >= 0; --deep) {
        printf("  %s:\n", deep ? "deep queues (31)" : "U200 queue depths");
        for (size_t si = 0; si < sizeof(shapes) / sizeof(shapes[0]); ++si) {
            const perf_shape & sh = shapes[si];
            fwprof::prof pr[2];
            if (want_prof) {
                pr[0].load_symbols(g_image);
                pr[1].load_symbols(g_image);
            }
            const auto a = perf_run(sh, 16, deep != 0, want_prof ? &pr[0] : nullptr, &stall_ok);
            const auto b = perf_run(sh, 48, deep != 0, want_prof ? &pr[1] : nullptr, &stall_ok);
            const double cyc = (b[0] - a[0]) / 32.0, ins = (b[1] - a[1]) / 32.0;
            // The launch's time per record against the RISC-V's work per record, both at F_CORE
            // 276.3 MHz: the RISC-V sets the pace when its work exceeds the device time per record.
            const double dev_us = (b[2] - a[2]) / 32.0 / 276.3, rv_us = cyc / (b[3] * 276.3);
            printf("    %-46s %6.1f RV cycles of work per record (%5.1f instructions)%s; per record %.2f us of "
                   "launch, %.2f us of RISC-V work (%s)\n", sh.name, cyc, ins, cyc <= 350.0 ? "" : " > 350", dev_us, rv_us,
                   rv_us >= 0.95 * dev_us ? "RISC-V-bound" : "unit-bound");
            if (deep) worst = std::max(worst, cyc);
            if (want_prof) {
                char title[160];
                snprintf(title, sizeof(title), "    profile per record: %s (%s)", sh.name, deep ? "deep" : "U200");
                char tag[32];
                snprintf(tag, sizeof(tag), ".s%zu%s", si, deep ? "_deep" : "_u200");
                fwprof::prof::diff(pr[1], pr[0]).report(stdout, title, 32.0, 12, tag);
            }
        }
    }
    CHECK(stall_ok, "perf: PERF PUSH_STALL != 0: a push waited for a credit in hardware (CC §7.3)");
    printf("fw_kernels_normrope_perf: worst decode shape %.1f RV cycles of work per record (deep queues); W2b target <= 350: %s\n",
           worst, worst <= 350.0 ? "met" : "NOT met");
    // Regression ceiling, holding on both Ibex tables (measured 2026-10-30, deep queues: worst 488
    // maxperf / 608 small); the target is judged on maxperf and printed above.
    CHECK(worst <= 700.0, "perf: regression ceiling 700 RV cycles per decode record (%.1f)", worst);
    return test_finish("fw_kernels_normrope_perf");
}
#endif  // !TSR_FW_HOST

#if defined(TSR_FW_HOST)
// ---------------------------------------------------------------------------
// Host build (SW §6.8)
// ---------------------------------------------------------------------------
tsr_sim * make_sim(const tsr_config & c, const char * sched, uint32_t race) {
    tsr_sim_opts so;
    memset(&so, 0, sizeof(so));
    so.sched = sched;
    so.race = race;
    so.threads = 2;
    return tsr_sim_create_opts(&c, &so);
}

// The executor's contract by hand: ring, ctx, the preload area loaded from the list's preload
// entries' DRAM copies and joined (as graph.cpp step 1).
struct host_exec {
    tsr::plan::exec_layout lay{};
    tsr::plan::dev_cfg dc{};
    tsr::spm_ring ring;
    tsr::kind_ctx ctx{};
    void init(const tsr_config & cfg, const suite & s) {
        const uint32_t spm_lines = cfg.spm_bytes / 64u;
        const uint32_t pl = uint32_t(s.pre.size() / LINE);
        tsr::plan::layout(spm_lines, pl, &lay);
        dc = tsr::plan::dev_cfg{spm_lines, cfg.t_max, cfg.k_max, cfg.num_ch, 0u, 16u, tsr::plan::CAPS_VU};
        ring.init(lay.ring_line, lay.ring_lines);
        ctx = tsr::kind_ctx{};
        ctx.ring = &ring;
        ctx.cfg = &dc;
        ctx.mml = tsr::plan::mm_limits_of(dc);
        ctx.pl = tsr::plan::plan_limits_of(dc);
        ctx.one = tsr::spm{lay.one_line};      // the one-tile block (graph.cpp's setup; W2b integration)
        ctx.one_lines = lay.one_lines;
        ctx.pre_lines = pl;
        ctx.pre = tsr::spm{lay.pre_line};
        ctx.preload = pl ? tsr::spm_ptr<uint8_t>(ctx.pre) : nullptr;
        // The preload area: each ROPE record's positions at its pre_byte.
        tsr::ticket t{};
        for (const auto & q : s.recs)
            if (q.kind == opl::K_ROPE)
                t = tsr::issue(tsr::LD(tsr::spm{lay.pre_line + q.a[ro::POS] / 64u}, tsr::from_handle(q.pos_h), L16(uint32_t(q.pos.size()))));
        tsr::join(t);
    }
};

// 1. The suites' records called directly against the references.
void host_suite(suite & s, const char * sched) {
    if (!build_list(s)) return;
    tsr_sim * sim = make_sim(s.cfg, sched, 2), * want = make_sim(s.cfg, "fixed", 0);
    fill(sim, s);
    fill(want, s);
    apply_refs(want, s);
    tsr::host_bind(sim);
    host_exec hx;
    hx.init(s.cfg, s);
    std::vector<tsr::ticket> tk;
    for (size_t i = 0; i < s.recs.size(); ++i) {
        const rec & q = s.recs[i];
        tsr::ticket d0{}, d1{};
        if (q.deps.size() > 0) d0 = tk[size_t(q.deps[0])];
        if (q.deps.size() > 1) {   // more than two: fold (the executor's JOIN records)
            tsr::group g;
            for (int k : q.deps) g.add(tk[size_t(k)]);
            const tsr::deps dd = g.as_deps();
            d0 = dd.a;
            d1 = dd.b;
        }
        hx.ctx.ret = 0;
        hx.ctx.op_index = uint32_t(i);
        const tsr::ticket t = q.kind == opl::K_ROPE ? tsr::kind_rope(q.a, ro::NARGS, 0, tsr::after(d0, d1), hx.ctx)
                                                    : tsr::kind_rms_norm(q.a, rr::NARGS, 0, tsr::after(d0, d1), hx.ctx);
        CHECK(hx.ctx.ret == 0 && !t.none(), "host %s [%s]: op %zu (%s): ret 0x%08x", s.name, sched, i, q.what, hx.ctx.ret);
        tk.push_back(t);
    }
    tsr::join_all();
    CHECK(tsr::err_units() == 0, "host %s [%s]: a unit in error state", s.name, sched);
    CHECK(tsr::host().no_credit_pushes == 0u, "host %s [%s]: %llu pushes without a credit", s.name, sched,
          (unsigned long long)tsr::host().no_credit_pushes);
    const uint32_t bad = compare_rows(s, sim, want, sched);
    printf("host %s [%s]: %zu records against the references: %s\n", s.name, sched, s.recs.size(), bad ? "DIFFER" : "bit-exact");
    tsr::host_bind(nullptr);
    tsr_sim_destroy(want);
    tsr_sim_destroy(sim);
}

// 2. Fast path vs general path on twin simulators.
struct plog {
    std::vector<std::array<uint32_t, 12>> v;
};
void plog_q(void * ctx, tsr_sim *, const tsr_q_event * ev) {
    if (ev->kind != TSR_QEV_PUSH && ev->kind != TSR_QEV_QCTRL) return;
    std::array<uint32_t, 12> e{};
    for (int k = 0; k < 8; ++k) e[size_t(k)] = ev->w[k];
    e[8] = ev->dep[0];
    e[9] = ev->dep[1];
    e[10] = ev->push;
    e[11] = ev->kind;
    static_cast<plog *>(ctx)->v.push_back(e);
}

// Runs record q of suite s after a history (two records of the same suite through the handlers, so
// the ring reuses released blocks) with deps on units (d0, d1: -1 none, 0 SYNC, 1 DMA, 2 QACT, 3 VU,
// 4 MVM), through the fast path (the handler) or the general path. Returns the ticket.
uint32_t twin_run(tsr_sim * sim, suite & s, size_t qi, bool fast, int d0, int d1, bool rec_flag, plog * log,
                  uint32_t ring_state[3]) {
    fill(sim, s);
    tsr::host_bind(sim);
    host_exec hx;
    hx.init(s.cfg, s);
    hx.ctx.push_flags = rec_flag ? tsr::PUSH_REC : 0u;
    // A ring of 2.5 blocks of the record's size (as fw_kernels_host 1b): the history's blocks are
    // reused by the record.
    const rec & q = s.recs[qi];
    tsr::kind_plan kp{};
    const bool planned = q.kind == opl::K_ROPE ? tsr::plan_rope(q.a, ro::NARGS, tsr::plan::plan_limits_of(hx.dc), &kp)
                                               : tsr::plan_rms_norm(q.a, rr::NARGS, tsr::plan::plan_limits_of(hx.dc), &kp);
    CHECK(planned, "twin %s: plan", q.what);
    const uint32_t b = kp.scratch_lines;
    hx.ring.init(hx.lay.ring_line, 2u * b + b / 2u);
    const int lid = tsr_sim_add_listener(sim, plog_q, log);
    tsr::ticket prev{};
    for (uint32_t i = 0; i < 3u; ++i) {        // history: the record itself three times (its own data)
        hx.ctx.ret = 0;
        prev = q.kind == opl::K_ROPE ? tsr::kind_rope(q.a, ro::NARGS, 0, tsr::deps{prev}, hx.ctx)
                                     : tsr::kind_rms_norm(q.a, rr::NARGS, 0, tsr::deps{prev}, hx.ctx);
    }
    // Predecessors on the units the case names (scratch in the OPL area, outside the ring).
    const uint32_t S0 = hx.lay.opl_line;
    const tsr::ticket ld = tsr::issue(tsr::LD(tsr::spm{S0}, tsr::dram{1, 0x40000u}, 56));
    const tsr::ticket qt = tsr::issue(tsr::QACT(tsr::qa::a32, tsr::spm{S0 + 100}, tsr::spm{S0}, 896, 1), tsr::deps{ld});
    const tsr::ticket vu = tsr::issue(tsr::VOP({.func = tsr::VOP_FUNC_COPY, .d = S0 + 300, .a = S0, .m = 16u * 56u, .rows = 1}),
                                      tsr::deps{ld});
    const tsr::ticket sy = tsr::issue(tsr::NOP(), tsr::deps{qt});
    auto pred = [&](int u) {
        switch (u) {
        case 0: return sy;
        case 1: return ld;
        case 2: return qt;
        case 3: return vu;
        default: return tsr::ticket{};
        }
    };
    hx.ctx.ret = 0;
    const tsr::deps dd = tsr::after(pred(d0), pred(d1));
    tsr::ticket t;
    const uint32_t f0 = tsr::nr::rms_fast_taken + tsr::nr::rope_fast_taken;
    if (fast) t = q.kind == opl::K_ROPE ? tsr::kind_rope(q.a, ro::NARGS, 0, dd, hx.ctx) : tsr::kind_rms_norm(q.a, rr::NARGS, 0, dd, hx.ctx);
    else t = q.kind == opl::K_ROPE ? tsr::nr::rope_slow(q.a, ro::NARGS, 0, dd, hx.ctx) : tsr::nr::rms_norm_slow(q.a, rr::NARGS, 0, dd, hx.ctx);
    CHECK(hx.ctx.ret == 0, "twin %s: ret 0x%08x (%s path)", q.what, hx.ctx.ret, fast ? "fast" : "general");
    tsr::join_all();
    CHECK(tsr::err_units() == 0, "twin %s: a unit in error state (%s path)", q.what, fast ? "fast" : "general");
    tsr_sim_remove_listener(sim, lid);
    ring_state[0] = hx.ring.head();
    ring_state[1] = hx.ring.blocks();
    ring_state[2] = tsr::nr::rms_fast_taken + tsr::nr::rope_fast_taken - f0;   // 1: the fast path took it
    tsr::host_bind(nullptr);
    return t.v;
}

void host_twins() {
    suite s = suite_decode();
    build_list(s);
    const int cases[][3] = { { 0, -1, 0 }, { 0, 1, 1 }, { 1, -1, 0 }, { -1, -1, 1 }, { 2, 3, 0 }, { 3, 0, 1 } };
    uint32_t n = 0, n_ok = 0;
    for (size_t qi = 0; qi < s.recs.size(); ++qi) {
        const rec & q = s.recs[qi];
        // Decode records only (the fast paths': ROPE one token; RMS_NORM one tile, dense rows, period 1).
        const bool dense = q.kind == opl::K_ROPE ? q.a[ro::N_TOKENS] == 1u
                                                 : (q.a[rr::ROWS] == 1u || q.a[rr::SRC_STRIDE] == L16(q.a[rr::N]));
        const bool period1 = q.kind == opl::K_ROPE || q.a[rr::W] == NONE || q.a[rr::W_PERIOD] == 1u;
        if (!dense || !period1 || q.a[rr::DST] == q.a[rr::SRC]) continue;
        for (const auto & c : cases) {
            tsr_sim * sa = make_sim(s.cfg, "fixed", 2), * sb = make_sim(s.cfg, "fixed", 2);
            plog la, lb;
            uint32_t ra[3], rb[3];
            const uint32_t ta = twin_run(sa, s, qi, true, c[0], c[1], c[2] != 0, &la, ra);
            const uint32_t tb = twin_run(sb, s, qi, false, c[0], c[1], c[2] != 0, &lb, rb);
            size_t f = 0;
            while (f < la.v.size() && f < lb.v.size() && la.v[f] == lb.v[f]) ++f;
            const bool same = la.v.size() == lb.v.size() && f == la.v.size();
            const bool took_fast = ra[2] == 1u && rb[2] == 0u;
            CHECK(same, "twin %s deps %d %d: pushes differ (%zu / %zu, first at %zu: w0 0x%08x / 0x%08x deps 0x%08x 0x%08x / "
                  "0x%08x 0x%08x)", q.what, c[0], c[1], la.v.size(), lb.v.size(), f, f < la.v.size() ? la.v[f][0] : 0u,
                  f < lb.v.size() ? lb.v[f][0] : 0u, f < la.v.size() ? la.v[f][8] : 0u, f < la.v.size() ? la.v[f][9] : 0u,
                  f < lb.v.size() ? lb.v[f][8] : 0u, f < lb.v.size() ? lb.v[f][9] : 0u);
            CHECK(ta == tb && ra[0] == rb[0] && ra[1] == rb[1], "twin %s: ticket 0x%08x / 0x%08x, ring head %u / %u, blocks %u / %u",
                  q.what, ta, tb, ra[0], rb[0], ra[1], rb[1]);
            CHECK(took_fast, "twin %s deps %d %d: the fast path declined", q.what, c[0], c[1]);
            ++n;
            n_ok += same && ta == tb && took_fast;
            tsr_sim_destroy(sb);
            tsr_sim_destroy(sa);
        }
    }
    printf("host twins: %u of %u decode records x deps pushed the same stream through the fast and the general path\n", n_ok, n);
}

// 3. Refused records: ctx.ret set, nothing pushed (both kinds, through the handler).
void host_refusals() {
    suite s = suite_decode();
    build_list(s);
    tsr_sim * sim = make_sim(s.cfg, "fixed", 2);
    fill(sim, s);
    tsr::host_bind(sim);
    host_exec hx;
    hx.init(s.cfg, s);
    struct bad { const char * what; int rec; int field; uint32_t value; uint32_t nargs, kflags, want; };
    const bad bads[] = {
        { "RMS_NORM n 0", 0, rr::N, 0u, rr::NARGS, 0, opl::RET_BAD_RECORD },
        { "RMS_NORM rows 0", 0, rr::ROWS, 0u, rr::NARGS, 0, opl::RET_BAD_RECORD },
        { "RMS_NORM weight period 0", 0, rr::W_PERIOD, 0u, rr::NARGS, 0, opl::RET_BAD_RECORD },
        { "RMS_NORM n_args 9", 0, rr::N, 896u, 9u, 0, opl::RET_BAD_RECORD },
        { "RMS_NORM kind flags", 0, rr::N, 896u, rr::NARGS, 1u, opl::RET_BAD_RECORD },
        { "RMS_NORM beyond the budget", 0, rr::N, 16u * 70000u, rr::NARGS, 0, opl::RET_SCRATCH },
        { "ROPE n_dims 0", 1, ro::N_DIMS, 0u, ro::NARGS, 0, opl::RET_BAD_RECORD },
        { "ROPE positions outside the preload area", 1, ro::POS, 64u * 300u, ro::NARGS, 0, opl::RET_BAD_RECORD },
        { "ROPE ATAB without table", 1, ro::MODE, 3u, ro::NARGS, 0, opl::RET_BAD_RECORD },
        { "ROPE kind flags", 1, ro::N_DIMS, 64u, ro::NARGS, 2u, opl::RET_BAD_RECORD },
        { "ROPE n_args 13", 1, ro::N_DIMS, 64u, 13u, 0, opl::RET_BAD_RECORD },
    };
    for (const auto & b : bads) {
        uint32_t a[14];
        memcpy(a, s.recs[size_t(b.rec)].a, sizeof(a));
        a[b.field] = b.value;
        const uint32_t p0 = tsr::pidx_now();
        hx.ctx.ret = 0;
        const tsr::ticket t = s.recs[size_t(b.rec)].kind == opl::K_ROPE ? tsr::kind_rope(a, b.nargs, b.kflags, tsr::deps{}, hx.ctx)
                                                                         : tsr::kind_rms_norm(a, b.nargs, b.kflags, tsr::deps{}, hx.ctx);
        CHECK(hx.ctx.ret == b.want && t.none() && tsr::pidx_now() == p0, "host refuse %s: ret 0x%08x (expected 0x%08x), %u pushes",
              b.what, hx.ctx.ret, b.want, tsr::pidx_now() - p0);
    }
    tsr::host_bind(nullptr);
    tsr_sim_destroy(sim);
    printf("host refuse: %zu refused records pushed nothing\n", sizeof(bads) / sizeof(bads[0]));
}

// 4. The 2-op RMS form against the spec's 7-op lowering (VU §6.2, docs/spec.md:311-321).
void host_seven_op() {
    rng_t r = { 0x7091ull };
    const uint32_t ns[] = { 1, 7, 16, 64, 100, 128, 129, 256, 512, 896, 3840, 5120 };
    long rows_total = 0, rows_same = 0;
    for (uint32_t n : ns) {
        const uint32_t rows = 64, L = L16(n);
        std::vector<uint32_t> x(size_t(rows) * L * 16u), w(size_t(L) * 16u), y2(x.size()), y7(x.size());
        for (uint32_t i = 0; i < rows; ++i) {
            const int kind = int(i % 8u);
            for (uint32_t j = 0; j < n; ++j) {
                float v = float(rng_normal(&r));
                if (kind == 1) v *= 1e-20f;                 // tiny: sumsq underflows towards FTZ
                if (kind == 2) v *= 1e18f;                  // huge: sumsq overflows to +inf
                if (kind == 3) v = 0.0f;                    // zero row: RC(SQ(eps))
                if (kind == 4 && j == 0) v = INFINITY;
                if (kind == 5 && j == 1u % n) v = NAN;
                if (kind == 6) v = (j & 1u) ? -0.0f : 1e-40f;   // subnormal inputs (DAZ)
                x[size_t(i) * L * 16u + j] = fbits(v);
            }
        }
        for (uint32_t j = 0; j < n; ++j) w[j] = fbits(float(1.0 + 0.2 * rng_normal(&r)));
        for (const float eps : { 1e-6f, 1e-5f, 0.0f }) {
            const uint32_t inv_n = fbits(tsr_fdiv(1.0f, float(n)));
            tsr::ref::rmsnorm_ref_rows(n, rows, x.data(), L * 16u, y2.data(), L * 16u, fbits(eps), inv_n, w.data(), 0, 1);
            tsr::ref::rmsnorm_ref_7op(n, rows, x.data(), L * 16u, y7.data(), L * 16u, fbits(eps), inv_n, w.data(), 0, 1);
            for (uint32_t i = 0; i < rows; ++i) {
                const bool same = memcmp(&y2[size_t(i) * L * 16u], &y7[size_t(i) * L * 16u], size_t(L) * 64u) == 0;
                rows_same += same;
                ++rows_total;
                CHECK(same, "7-op: n %u row %u eps %g: the 2-op form differs from the spec's 7-op lowering", n, i, double(eps));
            }
        }
    }
    printf("host 7-op: %ld of %ld rows bit-identical (2-op RED.SUMSQ post RMS + VOP.MULCB vs docs/spec.md:311-321)\n", rows_same,
           rows_total);
}
#endif  // TSR_FW_HOST

}  // namespace

int main(int argc, char ** argv) {
#if defined(TSR_FW_HOST)
    (void)argc;
    (void)argv;
    {
        suite a = suite_decode(), b = suite_prefill();
        host_suite(a, "fixed");
        host_suite(b, "fixed");
        suite c = suite_decode();
        host_suite(c, "random:7");
    }
    host_twins();
    host_refusals();
    host_seven_op();
    return test_finish("fw_kernels_normrope_host");
#else
    if (argc < 3) {
        fprintf(stderr, "usage: %s IMAGE suite|refuse|perf [--scheds N] [--only K]\n", argv[0]);
        return 2;
    }
    g_image = argv[1];
    struct stat st;
    if (stat(g_image, &st) != 0) {
        printf("%s: no image %s: skipped\n", argv[0], g_image);
        return 77;
    }
    const std::string mode = argv[2];
    uint32_t n_sched = 8;
    int only = -1;
    bool expect_fail = false;          // a mutant image: the run must find a difference
    for (int i = 3; i < argc; ++i) {
        if (!strcmp(argv[i], "--scheds") && i + 1 < argc) n_sched = uint32_t(atoi(argv[++i]));
        else if (!strcmp(argv[i], "--only") && i + 1 < argc) only = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--race") && i + 1 < argc) g_race = !strcmp(argv[++i], "report") ? 1u : 2u;
        else if (!strcmp(argv[i], "--expect-fail")) expect_fail = true;
    }
    if (mode == "perf") return perf_main();
    if (mode == "suite") {
        if (only < 0 || only == 0) { suite s = suite_decode(); run_scheds(s, n_sched); }
        if (only < 0 || only == 1) { suite s = suite_prefill(); run_scheds(s, n_sched); }
        if (only < 0 || only == 2) run_refusals();
    } else if (mode == "refuse") {
        run_refusals();
    } else {
        fprintf(stderr, "unknown mode %s\n", mode.c_str());
        return 2;
    }
    if (expect_fail) {
        printf("fw_kernels_normrope_rv: mutant image: %d check(s) failed (%s)\n", g_fail, g_fail ? "caught, as required" : "NOT CAUGHT");
        return g_fail ? 0 : 1;
    }
    return test_finish("fw_kernels_normrope_rv");
#endif
}
