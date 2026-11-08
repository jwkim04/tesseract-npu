// test_fw_moe_rv.cpp — fw_kernels_moe_rv*: the MoE kinds MOE_ROUTE (0x1E) and MUL_MAT_ID (0x1F)
// (sw/ctrl/kernels/k_moe.cpp, SW §7.5, SW-60 to SW-65) in the RISC-V image on tsr_rv, launched through the
// `graph` executor (docs/design/06-software-verification.md §10.2: per kind, 8 schedules, race fatal,
// outputs bit-exact against the references).
//
//   test_fw_moe_rv IMAGE suite [--scheds N]
//       op lists launched under the fixed schedule and 7 random ones with TSR_SIM_RACE fatal:
//       route    MOE_ROUTE of 128 experts / 8 used and of 64 / 4, T = 1 .. 140 (tiles of 64, a partial
//                last tile, dense and strided ids rows, strided logits rows), logits with equal values
//                (the tie rule), a NaN row, an inf row, one expert far above the rest
//       mmid     MUL_MAT_ID: the gate-up form (one x row shared by the slots, Q4_K) and the down form (one
//                x row per slot, Q8_0, scale table), 16 experts, 4 and 8 used, T = 1 .. 9, ids written by
//                the host and by a MOE_ROUTE of the same list (the dependency), K and N not multiples of 16
//                lines, more tokens than one ids chunk, a 512 KiB SPM
//       Every output (ids lines, packed weights, dst rows) is compared bit for bit with the references
//       (sw/ctrl/ref/moe_ref.cpp on a twin simulator), DONE must be OK / RET 0, the SW §4.7 observables
//       must be equal under every schedule, and the written DRAM must be exactly the output ranges.
//   test_fw_moe_rv IMAGE errors
//       an expert number outside [0, N_EXPERT) or negative ends the launch with TILE_ERR on the
//       MUL_MAT_ID (RET = op + 1), a refused MOE_ROUTE / MUL_MAT_ID record ends it with BAD_RECORD.
// Exit 0 when everything passed, 1 on a failure, 77 (skipped) when the image is missing.
#undef _GNU_SOURCE
#include "test_sim_obs.h"

#include <sys/stat.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <string>
#include <vector>

#include "../ref/fw_sim_host.hpp"
#include "../ref/moe_ref.hpp"
#include "tsr/opl.hpp"
#include "tsr/plan.hpp"
#include "tsr_race.h"
#include "tsr_ref_vu.h"

namespace {

namespace mr = tsr::opl::moer;
namespace mi = tsr::opl::mmid;
constexpr uint32_t NONE = tsr::opl::HANDLE_NONE;
constexpr uint64_t LINE = 64u;

uint32_t H(uint32_t ch, uint64_t beat) { return (ch << 28) | uint32_t(beat & 0x0FFFFFFFu); }

const char * g_image = nullptr;

// ---------------------------------------------------------------------------
// The list: records, host data, expected written ranges
// ---------------------------------------------------------------------------
struct rec {
    uint32_t kind;
    uint32_t a[17];
    std::vector<uint32_t> deps;
    uint32_t index = 0;
};

struct suite {
    const char * name;
    tsr_config cfg;
    std::vector<rec> recs;
    std::vector<std::function<void(tsr_sim *, rng_t *)>> fills;   // host data, in order
    uint64_t seed;
    std::vector<uint8_t> blob;
    uint32_t opl_h = H(0, 0x40000u);
};

struct alloc {
    uint64_t io[4] = { 0x4000000u / 64u, 0x1000000u / 64u, 0x1000000u / 64u, 0x1000000u / 64u };
    uint64_t w = 0x20000000u / 64u;
    uint64_t take(uint32_t ch, uint64_t beats) { const uint64_t b = io[ch]; io[ch] += (beats + 63u) & ~63ull; return b; }
};

tsr_config base_cfg(uint32_t spm_bytes) {
    tsr_config c = TSR_CONFIG_U200;
    c.t_max = 8;
    c.spm_bytes = spm_bytes;
    c.ch_bytes = 1ull << 30;
    c.caps = TSR_SIM_CAPS_IMPLEMENTED;
    return c;
}

// n_expert experts of N x K weights at w_off, expert e at w_off + e * stride (the per-channel expert stride).
void put_experts(tsr_sim * sim, rng_t * r, uint32_t fmt, uint32_t N, uint32_t K, uint64_t w_off, uint32_t n_expert) {
    const size_t row_bytes = size_t(K / tsr_fmt_block_size(fmt)) * tsr_fmt_ggml_block_bytes(fmt);
    std::vector<uint8_t> blk(row_bytes);
    const uint32_t C = tsr_sim_config(sim)->num_ch;
    const uint64_t stride = tsr_tensor_ch_beats(fmt, N, K, C);
    for (uint32_t e = 0; e < n_expert; ++e)
        for (uint32_t n = 0; n < N; ++n) {
            gen_row_blocks(r, fmt, K, blk.data());
            uint32_t ch;
            uint64_t off;
            tsr_row_location(fmt, N, K, C, n, &ch, &off);
            tsr_repack_row(fmt, blk.data(), K, tsr_sim_dram(sim, ch) + (w_off + e * stride + off) * LINE);
        }
}

void put_f32_rows(tsr_sim * sim, rng_t * r, uint32_t h, uint32_t rows, uint32_t stride, uint32_t n, double scale) {
    std::vector<float> x(n);
    for (uint32_t t = 0; t < rows; ++t) {
        for (uint32_t i = 0; i < n; ++i) x[i] = float(rng_normal(r) * scale);
        put_f32_le(tsr_sim_dram(sim, h >> 28) + (uint64_t(h & 0x0FFFFFFFu) + uint64_t(t) * stride) * LINE, x.data(), n);
    }
}
void put_words(tsr_sim * sim, uint32_t h, const std::vector<uint32_t> & w) {
    std::vector<uint8_t> b(w.size() * 4);
    for (size_t i = 0; i < w.size(); ++i)
        for (int k = 0; k < 4; ++k) b[4 * i + size_t(k)] = uint8_t(w[i] >> (8 * k));
    tsr_sim_dram_write(sim, h >> 28, uint64_t(h & 0x0FFFFFFFu) * LINE, b.data(), b.size());
}

tsr::ref::dram_view view_of(tsr_sim * sim, std::vector<uint8_t *> & chp) {
    const tsr_config * c = tsr_sim_config(sim);
    chp.resize(c->num_ch);
    for (uint32_t i = 0; i < c->num_ch; ++i) chp[i] = tsr_sim_dram(sim, i);
    return tsr::ref::dram_view{chp.data(), c->num_ch, c->ch_bytes};
}

// The DRAM beats a record writes (what its STs cover).
void add_ranges(std::vector<ival_t> & v, const rec & r) {
    auto line = [&](uint32_t h, uint64_t off, uint64_t n) {
        const uint64_t lo = (h & 0x0FFFFFFFu) + off;
        v.push_back(ival_t{h >> 28, lo, lo + n});
    };
    if (r.kind == tsr::opl::K_MOE_ROUTE) {
        const uint32_t k = r.a[mr::N_USED], T = r.a[mr::T];
        for (uint32_t t = 0; t < T; ++t) line(r.a[mr::IDS], uint64_t(t) * r.a[mr::IDS_STRIDE], 1);
        line(r.a[mr::W], 0, tsr::plan::vu_lines(T * k));
    } else {
        const uint32_t nu = r.a[mi::N_USED], T = r.a[mi::T], yl = (r.a[mi::N] + 15u) / 16u;
        for (uint32_t i = 0; i < T * nu; ++i) line(r.a[mi::DST], uint64_t(i) * r.a[mi::DST_STRIDE], yl);
    }
}
std::vector<ival_t> merged(std::vector<ival_t> v) {
    std::sort(v.begin(), v.end(), [](const ival_t & a, const ival_t & b) { return a.ch != b.ch ? a.ch < b.ch : a.lo < b.lo; });
    std::vector<ival_t> m;
    for (const auto & x : v) {
        if (!m.empty() && m.back().ch == x.ch && x.lo <= m.back().hi) m.back().hi = std::max(m.back().hi, x.hi);
        else m.push_back(x);
    }
    return m;
}

void build_list(suite & s) {
    tsr::opl::writer w(tsr::opl::F_STOP_ON_ERROR, 0x300E);
    for (auto & r : s.recs) {
        if (r.kind == tsr::opl::K_MOE_ROUTE) {
            r.index = w.add(r.kind, r.a, mr::NARGS, r.deps.data(), uint32_t(r.deps.size()));
        } else {
            r.index = w.add(r.kind, r.a, mi::NARGS, r.deps.data(), uint32_t(r.deps.size()));
        }
    }
    s.blob = w.finish();
    CHECK(w.ok() && !s.blob.empty(), "%s: OPL writer: %s", s.name, w.error());
}

void fill(tsr_sim * sim, suite & s) {
    rng_t r = { s.seed };
    for (auto & f : s.fills) f(sim, &r);
    tsr_sim_dram_write(sim, s.opl_h >> 28, uint64_t(s.opl_h & 0x0FFFFFFFu) * LINE, s.blob.data(), s.blob.size());
}

// The outputs of the references: a twin simulator filled the same way, the records applied in order.
// Returns the bytes of every output range of every record (in the order of add_ranges).
bool expected(suite & s, std::vector<std::vector<uint8_t>> * out) {
    tsr_sim_opts so;
    memset(&so, 0, sizeof(so));
    so.sched = "fixed";
    tsr_sim * sim = tsr_sim_create_opts(&s.cfg, &so);
    if (sim == nullptr) return false;
    fill(sim, s);
    std::vector<uint8_t *> chp;
    const tsr::ref::dram_view v = view_of(sim, chp);
    bool ok = true;
    for (auto & r : s.recs) {
        const int st = r.kind == tsr::opl::K_MOE_ROUTE ? tsr::ref::moe_route_ref(v, r.a) : tsr::ref::mul_mat_id_ref(v, r.a, 4);
        CHECK(st == tsr::ref::MM_OK, "%s: reference of op %u failed (%d)", s.name, r.index, st);
        ok = ok && st == tsr::ref::MM_OK;
        std::vector<ival_t> rg;
        add_ranges(rg, r);
        for (const ival_t & x : rg) out->push_back(std::vector<uint8_t>(v.ch[x.ch] + x.lo * LINE, v.ch[x.ch] + x.hi * LINE));
    }
    tsr_sim_destroy(sim);
    return ok;
}

struct run_obs {
    obs o;
    uint32_t done[7];
};

bool run_suite(suite & s, const char * sched, const std::vector<std::vector<uint8_t>> & want, run_obs * out) {
    tsr_sim_opts so;
    memset(&so, 0, sizeof(so));
    so.sched = sched;
    so.race = 2;
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
    CHECK(!ok || gi >= 0, "%s: no graph kernel in the image", s.name);
    if (ok && gi >= 0) {
        const uint32_t a[7] = { s.opl_h, uint32_t(s.blob.size()), tsr::opl::LF_VERIFY_CRC, 0, 0, 0, 0 };
        ok = fwh_launch(&h, uint32_t(gi), a, 0) == 0;
        CHECK(ok, "%s [%s]: launch", s.name, sched);
    }
    if (ok) {
        CHECK(TSR_GET(h.done[0], TSR_LCQ_R0_TYPE) == TSR_LCQ_R0_TYPE_DONE && fwh_status(&h) == TSR_LCQ_R0_STATUS_OK &&
              fwh_ret(&h) == 0, "%s [%s]: DONE 0x%08x RET 0x%08x (err ticket 0x%08x info 0x%08x w0 0x%08x pidx %u)", s.name,
              sched, h.done[0], h.done[2], h.done[3], h.done[4], h.done[5], h.done[6]);
        for (int i = 0; i < 7; ++i) out->done[i] = h.done[i];
        size_t wi = 0;
        std::vector<ival_t> all;
        for (const auto & r : s.recs) {
            std::vector<ival_t> rg;
            add_ranges(rg, r);
            for (const ival_t & x : rg) {
                const uint8_t * got = tsr_sim_dram(sim, x.ch) + x.lo * LINE;
                const std::vector<uint8_t> & w = want[wi++];
                if (memcmp(got, w.data(), w.size()) != 0) {
                    size_t k = 0;
                    while (got[k] == w[k]) ++k;
                    CHECK(0, "%s [%s]: op %u (kind 0x%x): output differs at ch%u beat %llu byte %zu (0x%08x, expected 0x%08x)",
                          s.name, sched, r.index, r.kind, x.ch, (unsigned long long)x.lo, k, get_u32_le(got + (k & ~size_t(3))),
                          get_u32_le(w.data() + (k & ~size_t(3))));
                    break;
                }
            }
            add_ranges(all, r);
        }
        obs_finish(&out->o, sim);
        std::vector<ival_t> wr(out->o.dram_wr, out->o.dram_wr + out->o.n_dram);
        const std::vector<ival_t> got = merged(wr), exp = merged(all);
        bool same = got.size() == exp.size();
        for (size_t i = 0; same && i < got.size(); ++i) same = got[i].ch == exp[i].ch && got[i].lo == exp[i].lo && got[i].hi == exp[i].hi;
        CHECK(same, "%s [%s]: the written DRAM (%zu ranges) is not exactly the output ranges (%zu)", s.name, sched, got.size(),
              exp.size());
        const tsr_race_report * rr = nullptr;
        const size_t reports = tsr_sim_race_reports(sim, &rr);
        CHECK(reports == 0, "%s [%s]: %zu race reports", s.name, sched, reports);
        printf("%s [%s]: %zu ops, %llu core cycles, %llu RV instructions\n", s.name, sched, s.recs.size(),
               (unsigned long long)h.cycles, (unsigned long long)out->o.instret);
    }
    fwh_close(&h);
    tsr_sim_destroy(sim);
    return ok;
}

int g_only = -1;                       // --only N: a list of record N alone (deps dropped), for debugging

bool run_scheds(suite & s, uint32_t n_sched) {
    if (g_only >= 0 && size_t(g_only) < s.recs.size()) {
        rec r = s.recs[size_t(g_only)];
        r.deps.clear();
        s.recs.assign(1, r);
    }
    build_list(s);
    std::vector<std::vector<uint8_t>> want;
    if (!expected(s, &want)) return false;
    std::vector<run_obs> runs(n_sched);
    bool ok = true;
    for (uint32_t k = 0; k < n_sched; ++k) {
        char sched[64];
        if (k == 0) snprintf(sched, sizeof(sched), "fixed");
        else snprintf(sched, sizeof(sched), "random:%u", 1000u + k);
        ok = run_suite(s, sched, want, &runs[k]) && ok;
        if (k > 0 && ok) {
            char what[128];
            snprintf(what, sizeof(what), "%s: %s vs fixed", s.name, sched);
            obs_equal(&runs[0].o, &runs[k].o, what);
            CHECK(memcmp(runs[0].done, runs[k].done, sizeof(runs[0].done)) == 0, "%s: DONE r0-r6 differ", what);
        }
    }
    for (auto & r : runs) obs_free(&r.o);
    return ok;
}

// ---------------------------------------------------------------------------
// Record makers
// ---------------------------------------------------------------------------
enum logit_mode { L_RANDOM, L_TIES, L_SPECIAL, L_PEAK };

// A MOE_ROUTE record with host logits (T rows, stride lines apart), ids rows ids_stride lines apart.
rec make_route(suite & s, alloc & al, uint32_t n, uint32_t k, uint32_t T, uint32_t lstride_pad, uint32_t istride,
               logit_mode mode, uint32_t lch, uint32_t ich, uint32_t wch) {
    rec r{};
    r.kind = tsr::opl::K_MOE_ROUTE;
    const uint32_t ln = (n + 15u) / 16u, ls = ln + lstride_pad;
    r.a[mr::LOGITS] = H(lch, al.take(lch, uint64_t(T) * ls));
    r.a[mr::LOGITS_STRIDE] = ls;
    r.a[mr::IDS] = H(ich, al.take(ich, uint64_t(T) * istride + 8));
    r.a[mr::IDS_STRIDE] = istride;
    r.a[mr::W] = H(wch, al.take(wch, tsr::plan::vu_lines(T * k) + 1));
    r.a[mr::N_EXPERT] = n;
    r.a[mr::N_USED] = k;
    r.a[mr::T] = T;
    r.a[mr::FLAGS] = 0;
    const uint32_t h = r.a[mr::LOGITS];
    s.fills.push_back([=](tsr_sim * sim, rng_t * rg) {
        put_f32_rows(sim, rg, h, T, ls, n, 2.0);
        if (mode == L_RANDOM) return;
        for (uint32_t t = 0; t < T; ++t) {
            uint8_t * row = tsr_sim_dram(sim, h >> 28) + (uint64_t(h & 0x0FFFFFFFu) + uint64_t(t) * ls) * LINE;
            auto set = [&](uint32_t i, float v) { put_f32_le(row + 4u * i, &v, 1); };
            if (mode == L_TIES) {                               // few distinct values: many ties across the boundary
                for (uint32_t i = 0; i < n; ++i) set(i, -2.0f + 0.5f * float(rng_below(rg, 5)));
            } else if (mode == L_SPECIAL) {
                if (t % 3 == 0) set(rng_below(rg, n), NAN);
                else if (t % 3 == 1) set(rng_below(rg, n), INFINITY);
                else set(rng_below(rg, n), -INFINITY);
            } else if (mode == L_PEAK) {                        // one logit so large that every other exp is 0
                set(rng_below(rg, n), 200.0f);
            }
        }
    });
    return r;
}

// A MUL_MAT_ID record. ids: host-written random expert numbers (ids_h == 0 allocates and fills them) or the
// ids of the given MOE_ROUTE record (route_ids handle and stride in bytes).
rec make_mmid(suite & s, alloc & al, uint32_t fmt, uint32_t N, uint32_t K, uint32_t T, uint32_t nu, uint32_t ne,
              bool shared_x, bool scale, uint32_t ids_h, uint32_t ids_stride_bytes, uint32_t xch, uint32_t ych,
              uint32_t x_pad = 0, uint32_t y_pad = 0, uint32_t bad_id = 0xFFFFFFFFu, bool bad_neg = false) {
    rec r{};
    r.kind = tsr::opl::K_MUL_MAT_ID;
    const uint32_t C = s.cfg.num_ch, xl = K / 16u, yl = (N + 15u) / 16u;
    const uint64_t wb = tsr_tensor_ch_beats(fmt, N, K, C);
    const uint64_t w_off = al.w;
    al.w += ((wb * ne) + 63u) & ~63ull;
    s.fills.push_back([=](tsr_sim * sim, rng_t * rg) { put_experts(sim, rg, fmt, N, K, w_off, ne); });
    r.a[mi::W_OFF] = uint32_t(w_off);
    r.a[mi::EXP_STRIDE] = uint32_t(wb);
    r.a[mi::FMT] = fmt;
    r.a[mi::N] = N;
    r.a[mi::K] = K;
    r.a[mi::T] = T;
    r.a[mi::N_USED] = nu;
    r.a[mi::N_EXPERT] = ne;
    r.a[mi::FLAGS] = 0;
    const uint32_t xslot = shared_x ? 0u : xl + x_pad;
    const uint32_t xtok = (shared_x ? xl : nu * (xl + x_pad)) + x_pad;
    r.a[mi::X_SLOT_STRIDE] = xslot;
    r.a[mi::X_TOK_STRIDE] = xtok;
    r.a[mi::X] = H(xch, al.take(xch, uint64_t(T) * xtok + 8));
    const uint32_t xh = r.a[mi::X];
    s.fills.push_back([=](tsr_sim * sim, rng_t * rg) {
        for (uint32_t t = 0; t < T; ++t)
            put_f32_rows(sim, rg, xh + t * xtok, shared_x ? 1u : nu, xslot ? xslot : xl, K, 1.0);
    });
    r.a[mi::DST_STRIDE] = yl + y_pad;
    r.a[mi::DST] = H(ych, al.take(ych, uint64_t(T) * nu * (yl + y_pad) + 8));
    r.a[mi::SCALE] = NONE;
    if (scale) {
        r.a[mi::SCALE] = H(xch, al.take(xch, (ne + 15u) / 16u + 1));
        const uint32_t sh = r.a[mi::SCALE];
        s.fills.push_back([=](tsr_sim * sim, rng_t * rg) { put_f32_rows(sim, rg, sh, 1, 1, ne, 1.0); });
    }
    if (ids_h != 0u) {
        r.a[mi::IDS] = ids_h;
        r.a[mi::IDS_STRIDE] = ids_stride_bytes;
    } else {
        r.a[mi::IDS_STRIDE] = ids_stride_bytes;
        r.a[mi::IDS] = H(xch, al.take(xch, (uint64_t(T) * ids_stride_bytes + 63u) / 64u + 2));
        const uint32_t ih = r.a[mi::IDS], ib = ids_stride_bytes;
        s.fills.push_back([=](tsr_sim * sim, rng_t * rg) {
            std::vector<uint32_t> w((uint64_t(T) * ib) / 4u + nu, 0u);
            for (uint32_t t = 0; t < T; ++t)
                for (uint32_t j = 0; j < nu; ++j) w[t * (ib / 4u) + j] = rng_below(rg, ne);
            if (bad_id != 0xFFFFFFFFu) w[(T - 1u) * (ib / 4u)] = bad_neg ? 0xFFFFFFFEu : bad_id;
            put_words(sim, ih, w);
        });
    }
    return r;
}

// ---------------------------------------------------------------------------
// Suites
// ---------------------------------------------------------------------------
suite suite_route() {
    suite s{};
    s.name = "route";
    s.cfg = base_cfg(4u << 20);
    s.seed = 0x40E;
    alloc al;
    // T = 1: the decode shape; dense ids (stride 1), strided ids rows as the argsort tensor (8 lines)
    s.recs.push_back(make_route(s, al, 128, 8, 1, 0, 8, L_RANDOM, 0, 1, 2));
    s.recs.push_back(make_route(s, al, 128, 8, 3, 0, 8, L_RANDOM, 0, 1, 2));
    s.recs.push_back(make_route(s, al, 128, 8, 16, 0, 1, L_RANDOM, 1, 2, 3));        // dense ids rows
    s.recs.push_back(make_route(s, al, 128, 8, 64, 0, 8, L_RANDOM, 2, 3, 0));        // one full tile
    s.recs.push_back(make_route(s, al, 128, 8, 65, 0, 8, L_RANDOM, 3, 0, 1));        // 64 + 1
    s.recs.push_back(make_route(s, al, 128, 8, 140, 2, 8, L_RANDOM, 0, 1, 2));       // 64 + 64 + 12, strided logits
    s.recs.push_back(make_route(s, al, 64, 4, 5, 0, 4, L_RANDOM, 0, 1, 2));
    s.recs.push_back(make_route(s, al, 128, 8, 9, 0, 8, L_TIES, 1, 2, 3));
    s.recs.push_back(make_route(s, al, 128, 8, 7, 0, 8, L_SPECIAL, 2, 3, 0));
    s.recs.push_back(make_route(s, al, 128, 8, 4, 0, 8, L_PEAK, 3, 0, 1));
    s.recs.push_back(make_route(s, al, 100, 6, 6, 1, 7, L_RANDOM, 0, 1, 2));         // n % 16 != 0, k = 6, odd strides
    s.recs.push_back(make_route(s, al, 16, 16, 3, 0, 2, L_TIES, 1, 2, 3));           // k = n = 16
    s.recs.push_back(make_route(s, al, 512, 16, 2, 0, 33, L_RANDOM, 2, 3, 0));       // 512 experts, k = 16
    s.recs.push_back(make_route(s, al, 8, 2, 33, 0, 1, L_RANDOM, 3, 0, 1));          // few experts, k = 2, T = 33
    return s;
}

suite suite_mmid() {
    suite s{};
    s.name = "mmid";
    s.cfg = base_cfg(4u << 20);
    s.seed = 0x1D;
    alloc al;
    const uint32_t Q8 = TSR_FMT_Q8_0, Q4 = TSR_FMT_Q4_K;
    // host ids, 16 experts
    s.recs.push_back(make_mmid(s, al, Q4, 256, 512, 1, 4, 16, true, false, 0, 16, 0, 1));          // gate-up form, decode
    s.recs.push_back(make_mmid(s, al, Q8, 192, 128, 1, 4, 16, false, true, 0, 16, 1, 2));          // down form with scale
    s.recs.push_back(make_mmid(s, al, Q4, 320, 768, 3, 8, 16, true, false, 0, 32, 2, 3));          // T = 3, k = 8
    s.recs.push_back(make_mmid(s, al, Q8, 100, 96, 5, 8, 12, false, true, 0, 36, 3, 0, 2, 1));     // N % 16 != 0, padded rows, row stride 36 B
    s.recs.push_back(make_mmid(s, al, Q8, 64, 64, 9, 2, 5, true, true, 0, 8, 0, 1));               // 9 tokens, 2 used
    s.recs.push_back(make_mmid(s, al, Q4, 128, 256, 70, 3, 16, true, false, 0, 520, 1, 2));        // ids rows 520 B apart: several chunks
    // ids from a MOE_ROUTE of the list (the dependency): 16 experts, 4 used, T = 4
    {
        rec rt = make_route(s, al, 16, 4, 4, 0, 8, L_RANDOM, 2, 3, 0);
        rt.deps = {};
        s.recs.push_back(rt);
        const uint32_t ri = uint32_t(s.recs.size()) - 1u;
        rec m1 = make_mmid(s, al, Q4, 256, 256, 4, 4, 16, true, false, rt.a[mr::IDS], rt.a[mr::IDS_STRIDE] * 64u, 3, 0);
        m1.deps = { ri };
        s.recs.push_back(m1);
        rec m2 = make_mmid(s, al, Q8, 128, 64, 4, 4, 16, false, true, rt.a[mr::IDS], rt.a[mr::IDS_STRIDE] * 64u, 0, 1);
        m2.deps = { ri };
        s.recs.push_back(m2);
    }
    return s;
}

// A small SPM (512 KiB): token blocks of two or three per ring, wrapping.
suite suite_mmid_small() {
    suite s{};
    s.name = "mmid_small_spm";
    s.cfg = base_cfg(512u << 10);
    s.seed = 0x5A;
    alloc al;
    s.recs.push_back(make_mmid(s, al, TSR_FMT_Q8_0, 512, 256, 40, 8, 16, true, true, 0, 32, 0, 1));
    s.recs.push_back(make_mmid(s, al, TSR_FMT_Q4_K, 256, 1024, 12, 8, 16, false, false, 0, 32, 1, 2));
    return s;
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------
struct err_case {
    const char * what;
    std::function<suite()> make;
    uint32_t want_ret;      // RET of the DONE record
    uint32_t want_status;
};

bool run_err(const err_case & c) {
    suite s = c.make();
    build_list(s);
    tsr_sim_opts so;
    memset(&so, 0, sizeof(so));
    so.sched = "fixed";
    so.race = 0;
    tsr_sim * sim = tsr_sim_create_opts(&s.cfg, &so);
    CHECK(sim != nullptr, "%s: simulator", c.what);
    if (sim == nullptr) return false;
    fill(sim, s);
    fwh h;
    bool ok = fwh_open(&h, sim, g_image) == 0;
    CHECK(ok, "%s: boot", c.what);
    if (ok) {
        const int gi = fwh_kernel(&h, "graph");
        const uint32_t a[7] = { s.opl_h, uint32_t(s.blob.size()), 0, 0, 0, 0, 0 };
        ok = gi >= 0 && fwh_launch(&h, uint32_t(gi), a, 0) == 0;
        CHECK(ok, "%s: launch", c.what);
        if (ok) {
            CHECK(fwh_status(&h) == c.want_status && uint32_t(fwh_ret(&h)) == c.want_ret,
                  "%s: status %u RET 0x%08x, wanted %u / 0x%08x", c.what, fwh_status(&h), fwh_ret(&h), c.want_status, c.want_ret);
            printf("%s: status %u RET 0x%08x (ERR_INFO 0x%08x)\n", c.what, fwh_status(&h), fwh_ret(&h), h.done[5]);
        }
    }
    fwh_close(&h);
    tsr_sim_destroy(sim);
    return ok;
}

suite err_suite(int which) {
    suite s{};
    s.name = "errors";
    s.cfg = base_cfg(4u << 20);
    s.seed = 0xE44;
    alloc al;
    switch (which) {
    case 0:   // an expert number beyond the tensor (last token, first slot)
        s.recs.push_back(make_mmid(s, al, TSR_FMT_Q8_0, 64, 64, 3, 2, 8, true, false, 0, 8, 0, 1, 0, 0, 8));
        break;
    case 1:   // a negative expert number
        s.recs.push_back(make_mmid(s, al, TSR_FMT_Q8_0, 64, 64, 2, 2, 8, false, true, 0, 8, 0, 1, 0, 0, 0, true));
        break;
    case 2: { // MOE_ROUTE with FLAGS != 0
        rec r = make_route(s, al, 128, 8, 1, 0, 8, L_RANDOM, 0, 1, 2);
        r.a[mr::FLAGS] = 1;
        s.recs.push_back(r);
        break;
    }
    case 3: { // MOE_ROUTE with k > n
        rec r = make_route(s, al, 8, 4, 1, 0, 1, L_RANDOM, 0, 1, 2);
        r.a[mr::N_USED] = 9;
        s.recs.push_back(r);
        break;
    }
    case 4: { // MUL_MAT_ID with an unsupported format
        rec r = make_mmid(s, al, TSR_FMT_Q8_0, 64, 64, 1, 2, 8, true, false, 0, 8, 0, 1);
        r.a[mi::FMT] = 2;
        s.recs.push_back(r);
        break;
    }
    default: { // MUL_MAT_ID with a scale table and an ids stride that is not a multiple of 4
        rec r = make_mmid(s, al, TSR_FMT_Q8_0, 64, 64, 2, 2, 8, true, true, 0, 8, 0, 1);
        r.a[mi::IDS_STRIDE] = 9;
        s.recs.push_back(r);
        break;
    }
    }
    return s;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s IMAGE suite|errors [--scheds N]\n", argv[0]);
        return 2;
    }
    g_image = argv[1];
    struct stat st;
    if (stat(g_image, &st) != 0) {
        fprintf(stderr, "image %s missing: skipped\n", g_image);
        return 77;
    }
    uint32_t n_sched = 8;
    for (int i = 3; i + 1 < argc; ++i) {
        if (strcmp(argv[i], "--scheds") == 0) n_sched = uint32_t(atoi(argv[i + 1]));
        if (strcmp(argv[i], "--only") == 0) g_only = atoi(argv[i + 1]);
    }
    const std::string mode = argv[2];
    if (mode == "suite") {
        suite a = suite_route(), b = suite_mmid(), c = suite_mmid_small();
        run_scheds(a, n_sched);
        run_scheds(b, n_sched);
        run_scheds(c, 2u);
    } else if (mode == "errors") {
        const uint32_t BAD = tsr::opl::RET_BAD_RECORD;
        const err_case cases[] = {
            { "id beyond the tensor", [] { return err_suite(0); }, 1u, TSR_LCQ_R0_STATUS_TILE_ERR },
            { "negative id", [] { return err_suite(1); }, 1u, TSR_LCQ_R0_STATUS_TILE_ERR },
            { "ROUTE flags", [] { return err_suite(2); }, BAD, TSR_LCQ_R0_STATUS_KERNEL_ERR },
            { "ROUTE k > n", [] { return err_suite(3); }, BAD, TSR_LCQ_R0_STATUS_KERNEL_ERR },
            { "MUL_MAT_ID fmt", [] { return err_suite(4); }, BAD, TSR_LCQ_R0_STATUS_KERNEL_ERR },
            { "MUL_MAT_ID ids stride", [] { return err_suite(5); }, BAD, TSR_LCQ_R0_STATUS_KERNEL_ERR },
        };
        for (const err_case & c : cases) run_err(c);
    } else {
        fprintf(stderr, "unknown mode %s\n", mode.c_str());
        return 2;
    }
    return g_fail ? 1 : (printf("test_fw_moe_rv %s: all %ld checks passed\n", mode.c_str(), g_checks), 0);
}
