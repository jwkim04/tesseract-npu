// test_fw_gdn_rv.cpp — fw_gdn_rv*: the Gated DeltaNet kinds SOFTPLUS (0x2D), SSM_CONV (0x2E) and
// GATED_DELTA_NET (0x2F) (sw/ctrl/kernels/gdn_kinds.cpp; docs/design/06-software-verification.md "SW"
// §7.9) in the RISC-V image on tsr_rv, launched through the `graph` executor with op lists (the
// form of test_fw_elem_rv.cpp: fixed schedule and random ones with TSR_SIM_RACE fatal, outputs
// bit-exact against sw/ctrl/ref/gdn_ref.cpp, SW §4.7 observables equal across schedules, the written
// DRAM exactly the declared output regions).
//
//   test_fw_gdn_rv IMAGE suite [--scheds N]
//       softplus  rows, strides, a tile boundary on a small SPM, the VU specials (inf, NaN, +-0,
//                 subnormals, +-88, 1e30), a chain from another op's output (dep);
//       sconv     the 2B / 27B shapes (C 6144 / 10240, d_conv 4, T 1 and 5), several sequences, strided
//                 token rows, in-place state, a small SPM that tiles the channels, with and without SILU;
//       gdn       S_v 128 with H_k = H_v = 16 and H_v = 48 / H_k = 16, T 1 and 3, two sequences, strided q / k / v
//                 token rows, S_v 16 / 32 / 64, in-place state, a small SPM;
//       chain     SSM_CONV -> SOFTPLUS-free conv output feeding nothing, GDN -> SOFTPLUS (deps across kinds).
//   test_fw_gdn_rv IMAGE errors
//       refused records answer BAD_RECORD / SCRATCH and write nothing.
// Exit 0 when everything passed, 1 on a failure, 77 (skipped) when the image is missing.
#undef _GNU_SOURCE
#include "test_sim_obs.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <string>
#include <vector>

#include "../ref/fw_sim_host.hpp"
#include "../ref/gdn_ref.hpp"
#include "tsr/opl.hpp"
#include "tsr/plan.hpp"
#include "tsr_race.h"

namespace {

namespace opl = tsr::opl;
constexpr uint64_t LINE = 64u;

uint32_t H(uint32_t ch, uint64_t beat) { return (ch << 28) | uint32_t(beat & 0x0FFFFFFFu); }
uint32_t L16(uint32_t n) { return (n + 15u) / 16u; }

const char * g_image = nullptr;

// ---------------------------------------------------------------------------
// Data
// ---------------------------------------------------------------------------
uint32_t special_word(rng_t * r) {
    static const uint32_t sp[] = {
        0x00000000u, 0x80000000u, 0x7F800000u, 0xFF800000u, 0x7FC00000u, 0xFFC00001u, 0x7F800001u, 0x00000001u,
        0x807FFFFFu, 0x00400000u, 0x7F7FFFFFu, 0xFF7FFFFFu, 0x00800000u, 0x80800000u, 0x42B00000u /* 88 */,
        0xC2B00000u /* -88 */, 0x41A00000u /* 20 */, 0xC1A00000u, 0x7149F2CAu /* 1e30 */, 0xF149F2CAu, 0x3F800000u,
        0xBF800000u, 0x3F000000u};
    return sp[rng_below(r, uint32_t(sizeof(sp) / sizeof(sp[0])))];
}

struct fill_t {
    uint32_t h, rows, stride, n;   // rows of `stride` lines; the first n words of each row get data
    int mode;                      // 0: normal * a, 1: uniform in [a, b], 2: normal * a with specials
    float a, b;
};
struct region {
    uint32_t h, rows, stride, lines;
};
struct op_t {
    uint32_t kind;
    std::vector<uint32_t> a;
    std::vector<uint32_t> deps;
    std::vector<region> outs;
    uint32_t index = 0;
    const char * what = "";
};
struct suite {
    const char * name;
    tsr_config cfg;
    uint64_t seed;
    std::vector<fill_t> fills;
    std::vector<op_t> ops;
    std::vector<uint8_t> blob;
    uint32_t opl_h = H(0, 0x40000u);
    uint64_t io[4] = {0x100000u, 0x40000u, 0x40000u, 0x40000u};
    uint32_t next_ch = 0;
    uint32_t take(uint32_t ch, uint64_t lines) {
        const uint64_t b = io[ch];
        io[ch] += (lines + 7u) & ~7ull;
        return H(ch, b);
    }
    uint32_t ch() { return next_ch++ & 3u; }
};

tsr_config base_cfg(uint32_t spm_bytes) {
    tsr_config c = TSR_CONFIG_U200;
    c.t_max = 8;
    c.spm_bytes = spm_bytes;
    c.ch_bytes = 1ull << 30;
    c.caps = TSR_SIM_CAPS_IMPLEMENTED;
    return c;
}

// A filled tensor of rows x `stride` lines; n words of each row carry data.
uint32_t tensor(suite & s, uint32_t rows, uint32_t stride, uint32_t n, int mode, float a, float b = 0.0f) {
    const uint32_t h = s.take(s.ch(), uint64_t(rows) * stride);
    s.fills.push_back(fill_t{h, rows, stride, n, mode, a, b});
    return h;
}
// An output tensor (not filled; pre-filled with garbage so that a missing write shows).
uint32_t outtensor(suite & s, uint32_t rows, uint32_t stride, uint32_t n = 0) {
    const uint32_t h = s.take(s.ch(), uint64_t(rows) * stride);
    s.fills.push_back(fill_t{h, rows, stride, n, 1, 0.0f, 0.0f});
    return h;
}

int add_op(suite & s, uint32_t kind, std::vector<uint32_t> a, std::vector<region> outs, std::vector<uint32_t> deps, const char * what) {
    op_t o;
    o.kind = kind;
    o.a = std::move(a);
    o.outs = std::move(outs);
    o.deps = std::move(deps);
    o.what = what;
    s.ops.push_back(o);
    return int(s.ops.size()) - 1;
}

// ---------------------------------------------------------------------------
// Record builders
// ---------------------------------------------------------------------------
// SOFTPLUS rows x n; from >= 0: the source is that op's first output (rows of its stride).
int softplus(suite & s, uint32_t n, uint32_t rows, uint32_t pad = 0, uint32_t dpad = 0, bool special = false, int from = -1,
             const char * what = "") {
    uint32_t hs, ss;
    std::vector<uint32_t> deps;
    if (from >= 0) {
        const region & g = s.ops[size_t(from)].outs[0];
        hs = g.h;
        ss = g.stride;
        deps.push_back(uint32_t(from));
    } else {
        ss = L16(n) + pad;
        hs = tensor(s, rows, ss, n, special ? 2 : 0, 3.0f);
    }
    const uint32_t ds = L16(n) + dpad, hd = outtensor(s, rows, ds);
    return add_op(s, opl::K_SOFTPLUS, {hd, ds, hs, ss, n, rows}, {region{hd, rows, ds, L16(n)}}, deps, what);
}

// SSM_CONV
int sconv(suite & s, uint32_t C, uint32_t dc, uint32_t T, uint32_t ns, bool silu, uint32_t xpad = 0, uint32_t dpad = 0,
          bool inplace = false, const char * what = "") {
    const uint32_t Lc = C / 16u;
    const uint32_t hw = tensor(s, dc, Lc, C, 0, 0.5f);
    const uint32_t hsi = tensor(s, ns * (dc - 1u), Lc, C, 0, 1.0f);
    const uint32_t hso = inplace ? hsi : outtensor(s, ns * (dc - 1u), Lc);
    const uint32_t xs = Lc + xpad, ds = Lc + dpad;
    const uint32_t hx = tensor(s, ns * T, xs, C, 0, 1.0f);
    const uint32_t hd = outtensor(s, ns * T, ds);
    std::vector<region> outs = {region{hd, ns * T, ds, Lc}, region{hso, ns * (dc - 1u), Lc, Lc}};
    return add_op(s, opl::K_SSM_CONV, {hd, ds, hx, xs, hsi, hso, hw, C, dc, T, ns, silu ? opl::sconv::FLAG_SILU : 0u}, outs, {}, what);
}

// GATED_DELTA_NET: sv, hk, hv, T tokens, ns sequences; token-row padding of q / k / v (lines), v head stride
// (L = dense), in-place state.
struct gdn_opts {
    uint32_t qpad = 0, vpad = 0, gpad = 0, vhs_extra = 0;
    bool inplace = false;
    bool scale_nontrivial = true;
};
int gdn(suite & s, uint32_t sv, uint32_t hk, uint32_t hv, uint32_t T, uint32_t ns, const gdn_opts & o = {}, const char * what = "") {
    const uint32_t L = sv / 16u, nt = ns * T, Lg = L16(hv);
    const float qs = 1.0f / std::sqrt(float(sv));
    const uint32_t qts = hk * L + o.qpad, vhs = L + o.vhs_extra, vts = hv * vhs + o.vpad, gts = Lg + o.gpad;
    const uint32_t hq = tensor(s, nt, qts, hk * L * 16u, 0, qs), hk_ = tensor(s, nt, qts, hk * L * 16u, 0, qs);
    const uint32_t hv_ = tensor(s, nt, vts, hv * vhs * 16u, 0, 1.0f);
    const uint32_t hg = tensor(s, nt, gts, hv, 1, -3.0f, -0.001f), hb = tensor(s, nt, gts, hv, 1, 0.0f, 1.0f);
    const uint32_t hsi = tensor(s, ns * hv * sv, L, sv, 0, 0.3f);
    const uint32_t hso = o.inplace ? hsi : outtensor(s, ns * hv * sv, L);
    const uint32_t hd = outtensor(s, nt * hv, L);
    const uint32_t scale = o.scale_nontrivial ? 0x3db504f3u /* 1/sqrt(128) */ : 0x3f800000u;
    std::vector<uint32_t> a(opl::gdn::NARGS);
    namespace g = opl::gdn;
    a[g::DST] = hd; a[g::STATE_OUT] = hso; a[g::STATE_IN] = hsi;
    a[g::Q] = hq; a[g::Q_HEAD_STRIDE] = L; a[g::Q_TOKEN_STRIDE] = qts;
    a[g::K] = hk_; a[g::K_HEAD_STRIDE] = L; a[g::K_TOKEN_STRIDE] = qts;
    a[g::V] = hv_; a[g::V_HEAD_STRIDE] = vhs; a[g::V_TOKEN_STRIDE] = vts;
    a[g::G] = hg; a[g::G_TOKEN_STRIDE] = gts; a[g::BETA] = hb; a[g::BETA_TOKEN_STRIDE] = gts;
    a[g::S_V] = sv; a[g::H_K] = hk; a[g::H_V] = hv; a[g::N_TOKENS] = T; a[g::N_SEQS] = ns; a[g::SCALE] = scale;
    return add_op(s, opl::K_GATED_DELTA_NET, a, {region{hd, nt * hv, L, L}, region{hso, ns * hv * sv, L, L}}, {}, what);
}

// ---------------------------------------------------------------------------
// The list, fill, mirror, run
// ---------------------------------------------------------------------------
void build(suite & s) {
    opl::writer w(opl::F_STOP_ON_ERROR, 0x6D0Eu);
    for (auto & o : s.ops) o.index = w.add(o.kind, o.a.data(), uint32_t(o.a.size()), o.deps.data(), uint32_t(o.deps.size()), 0u);
    s.blob = w.finish();
    CHECK(w.ok() && !s.blob.empty(), "%s: OPL writer: %s", s.name, w.error());
}

void fill(tsr_sim * sim, const suite & s) {
    rng_t r = {s.seed};
    for (const fill_t & g : s.fills) {
        for (uint32_t row = 0; row < g.rows; ++row) {
            uint8_t * p = tsr_sim_dram(sim, g.h >> 28) + (uint64_t(g.h & 0x0FFFFFFFu) + uint64_t(row) * g.stride) * LINE;
            const uint32_t words = 16u * g.stride;
            for (uint32_t j = 0; j < words; ++j) {
                uint32_t v;
                if (j >= g.n) v = rng_u32(&r);
                else if (g.mode == 1) v = f2u(g.a + float(rng_unit(&r)) * (g.b - g.a));
                else if (g.mode == 2 && rng_below(&r, 4) == 0) v = special_word(&r);
                else v = f2u(float(rng_normal(&r)) * g.a);
                memcpy(p + 4u * j, &v, 4);
            }
        }
    }
    tsr_sim_dram_write(sim, s.opl_h >> 28, uint64_t(s.opl_h & 0x0FFFFFFFu) * LINE, s.blob.data(), s.blob.size());
}

struct mirror {
    std::vector<std::vector<uint8_t>> ch;
    std::vector<uint8_t *> chp;
    uint64_t bytes = 0;
    tsr::ref::dram_view view() { return tsr::ref::dram_view{chp.data(), uint32_t(chp.size()), bytes}; }
};
void make_mirror(tsr_sim * sim, const suite & s, mirror & m) {
    uint64_t top = 0;
    for (uint32_t c = 0; c < 4u; ++c) top = std::max(top, s.io[c]);
    m.bytes = top * LINE;
    m.ch.assign(4, std::vector<uint8_t>(m.bytes));
    m.chp.resize(4);
    for (uint32_t c = 0; c < 4u; ++c) {
        memcpy(m.ch[c].data(), tsr_sim_dram(sim, c), m.bytes);
        m.chp[c] = m.ch[c].data();
    }
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
    for (const auto & o : s.ops)
        for (const region & g : o.outs)
            for (uint32_t r = 0; r < g.rows; ++r) {
                const uint64_t lo = (g.h & 0x0FFFFFFFu) + uint64_t(r) * g.stride;
                v.push_back(ival_t{g.h >> 28, lo, lo + g.lines});
            }
    return merge(v);
}

const char * kind_nm(uint32_t k) { return opl::kind_name(k); }

int apply_ref(const tsr::ref::dram_view & v, const op_t & o) {
    switch (o.kind) {
        case opl::K_SOFTPLUS: return tsr::ref::softplus_ref(v, o.a.data());
        case opl::K_SSM_CONV: return tsr::ref::sconv_ref(v, o.a.data());
        case opl::K_GATED_DELTA_NET: return tsr::ref::gdn_ref(v, o.a.data());
        default: return -100;
    }
}

struct run_obs {
    obs o;
    uint32_t done[7];
    uint64_t cycles = 0;
};

bool run_suite(suite & s, const char * sched, mirror & m, bool first, run_obs * out) {
    tsr_sim_opts so;
    memset(&so, 0, sizeof(so));
    so.sched = sched;
    so.race = 2;
    so.threads = 2;
    tsr_sim * sim = tsr_sim_create_opts(&s.cfg, &so);
    CHECK(sim != nullptr, "%s: simulator", s.name);
    if (sim == nullptr) return false;
    fill(sim, s);
    if (first) {
        make_mirror(sim, s, m);
        const tsr::ref::dram_view v = m.view();
        for (const auto & o : s.ops) {
            const int rc = apply_ref(v, o);
            CHECK(rc == tsr::ref::GD_OK, "%s: the reference refused op %u (%s %s): %d", s.name, o.index, kind_nm(o.kind), o.what, rc);
        }
    }
    obs_attach(&out->o, sim);
    fwh h;
    bool ok = fwh_open(&h, sim, g_image) == 0;
    CHECK(ok, "%s [%s]: boot", s.name, sched);
    const int gi = ok ? fwh_kernel(&h, "graph") : -1;
    if (ok && gi >= 0) {
        const uint32_t a[7] = {s.opl_h, uint32_t(s.blob.size()), opl::LF_VERIFY_CRC, 0, 0, 0, 0};
        ok = fwh_launch(&h, uint32_t(gi), a, 0) == 0;
        CHECK(ok, "%s [%s]: launch", s.name, sched);
    }
    if (ok) {
        CHECK(TSR_GET(h.done[0], TSR_LCQ_R0_TYPE) == TSR_LCQ_R0_TYPE_DONE && fwh_status(&h) == TSR_LCQ_R0_STATUS_OK &&
                  fwh_ret(&h) == 0,
              "%s [%s]: DONE 0x%08x RET 0x%08x (err ticket 0x%08x info 0x%08x w0 0x%08x pidx %u)", s.name, sched, h.done[0],
              h.done[2], h.done[3], h.done[4], h.done[5], h.done[6]);
        for (int i = 0; i < 7; ++i) out->done[i] = h.done[i];
        for (const auto & o : s.ops) {
            for (size_t k = 0; k < o.outs.size(); ++k) {
                const region & g = o.outs[k];
                bool bad = false;
                for (uint32_t r = 0; r < g.rows && !bad; ++r) {
                    const uint64_t off = (uint64_t(g.h & 0x0FFFFFFFu) + uint64_t(r) * g.stride) * LINE;
                    const uint8_t * got = tsr_sim_dram(sim, g.h >> 28) + off;
                    const uint8_t * want = m.ch[g.h >> 28].data() + off;
                    if (memcmp(got, want, size_t(g.lines) * LINE) != 0) {
                        size_t b = 0;
                        while (got[b] == want[b]) ++b;
                        CHECK(0, "%s [%s]: op %u (%s %s) output %zu row %u: word %zu is 0x%08x, expected 0x%08x", s.name, sched,
                              o.index, kind_nm(o.kind), o.what, k, r, b / 4u, get_u32_le(got + (b & ~size_t(3))),
                              get_u32_le(want + (b & ~size_t(3))));
                        bad = true;
                    }
                }
            }
        }
        obs_finish(&out->o, sim);
        const std::vector<ival_t> want = written_rows(s);
        const std::vector<ival_t> got = merge(std::vector<ival_t>(out->o.dram_wr, out->o.dram_wr + out->o.n_dram));
        bool same = got.size() == want.size();
        for (size_t i = 0; same && i < got.size(); ++i)
            same = got[i].ch == want[i].ch && got[i].lo == want[i].lo && got[i].hi == want[i].hi;
        CHECK(same, "%s [%s]: the written DRAM (%zu ranges) is not exactly the output regions (%zu ranges)", s.name, sched,
              got.size(), want.size());
        const tsr_race_report * rr = nullptr;
        const size_t reports = tsr_sim_race_reports(sim, &rr);
        CHECK(reports == 0, "%s [%s]: %zu race reports", s.name, sched, reports);
        out->cycles = h.cycles;
        printf("%s [%s]: %zu ops, %llu core cycles, %llu RV instructions\n", s.name, sched, s.ops.size(),
               (unsigned long long)h.cycles, (unsigned long long)out->o.instret);
    }
    fwh_close(&h);
    tsr_sim_destroy(sim);
    return ok;
}

bool run_scheds(suite & s, uint32_t n_sched) {
    build(s);
    mirror m;
    std::vector<run_obs> runs(n_sched);
    bool ok = true;
    for (uint32_t k = 0; k < n_sched; ++k) {
        char sched[64];
        if (k == 0) snprintf(sched, sizeof(sched), "fixed");
        else snprintf(sched, sizeof(sched), "random:%u", 3000u + k);
        ok = run_suite(s, sched, m, k == 0, &runs[k]) && ok;
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
// Suites
// ---------------------------------------------------------------------------
suite suite_softplus() {
    suite s{};
    s.name = "softplus";
    s.cfg = base_cfg(4u << 20);
    s.seed = 0x50F7;
    softplus(s, 16, 1, 0, 0, false, -1, "H_v = 16, one token");
    softplus(s, 48, 1, 0, 0, false, -1, "H_v = 48, one token");
    softplus(s, 48, 7, 1, 2, false, -1, "7 tokens, strided");
    softplus(s, 100, 5, 0, 0, true, -1, "n = 100 (tail), specials");
    softplus(s, 1000, 33, 0, 0, true, -1, "33 rows, specials");
    softplus(s, 4096, 1, 0, 0, false, -1, "one long row");
    return s;
}
suite suite_softplus_small() {
    suite s{};
    s.name = "softplus_small";
    s.cfg = base_cfg(512u << 10);   // ring budget 3,376 lines: tiles of rows
    s.seed = 0x50F8;
    softplus(s, 48, 300, 0, 0, false, -1, "300 rows (tiles)");
    softplus(s, 256, 200, 1, 1, true, -1, "200 rows of 16 lines, strided (tiles)");
    softplus(s, 8192, 2, 0, 0, false, -1, "2 rows of 512 lines");
    return s;
}
suite suite_sconv() {
    suite s{};
    s.name = "sconv";
    s.cfg = base_cfg(4u << 20);
    s.seed = 0x5C07;
    sconv(s, 6144, 4, 1, 1, true, 0, 0, false, "2B decode (C 6144, SILU)");
    sconv(s, 10240, 4, 1, 1, true, 0, 0, false, "27B decode (C 10240, SILU)");
    sconv(s, 6144, 4, 1, 1, false, 0, 0, false, "no SILU");
    sconv(s, 6144, 4, 5, 1, true, 3, 2, false, "T = 5, strided token rows");
    sconv(s, 1024, 4, 3, 2, true, 0, 0, false, "T = 3, two sequences");
    sconv(s, 512, 4, 1, 3, true, 1, 1, true, "three sequences, in-place state");
    sconv(s, 160, 3, 2, 2, true, 0, 0, false, "d_conv 3, C 160");
    sconv(s, 96, 2, 4, 1, false, 0, 0, true, "d_conv 2, in place");
    sconv(s, 6144, 4, 33, 1, true, 0, 0, false, "T = 33");
    return s;
}
suite suite_sconv_small() {
    suite s{};
    s.name = "sconv_small";
    s.cfg = base_cfg(512u << 10);
    s.seed = 0x5C08;
    sconv(s, 6144, 4, 1, 1, true, 0, 0, false, "decode, channel tiles");
    sconv(s, 6144, 4, 9, 2, true, 1, 2, false, "T = 9, two sequences, channel tiles, strided");
    sconv(s, 2000 - 2000 % 16, 4, 20, 1, false, 0, 0, true, "C 1984, T = 20, in place");
    return s;
}
suite suite_gdn() {
    suite s{};
    s.name = "gdn";
    s.cfg = base_cfg(4u << 20);
    s.seed = 0x6D01;
    gdn_opts o;
    gdn(s, 128, 16, 16, 1, 1, o, "2B decode (S_v 128, 16 heads)");
    gdn(s, 128, 16, 48, 1, 1, o, "27B decode (48 v heads, 16 k heads)");
    o.qpad = 3; o.vpad = 5; o.gpad = 1;
    gdn(s, 128, 16, 16, 3, 1, o, "T = 3, strided token rows");
    o = gdn_opts{};
    o.inplace = true;
    gdn(s, 64, 4, 8, 2, 2, o, "S_v 64, two sequences, in place");
    gdn(s, 32, 2, 6, 5, 1, o, "S_v 32, 6 heads over 2 k heads, T = 5, in place");
    gdn(s, 16, 1, 3, 1, 3, gdn_opts{}, "S_v 16, 3 sequences");
    gdn_opts p;
    p.vhs_extra = 2;
    p.scale_nontrivial = false;
    gdn(s, 128, 4, 4, 2, 1, p, "v head stride 10 lines, scale 1");
    gdn(s, 256, 2, 2, 1, 1, gdn_opts{}, "S_v 256");
    return s;
}
suite suite_gdn_small() {
    suite s{};
    s.name = "gdn_small";
    s.cfg = base_cfg(512u << 10);   // budget half 1,688 lines: S_v 64 (812 lines) fits
    s.seed = 0x6D02;
    gdn(s, 64, 4, 8, 1, 1, gdn_opts{}, "S_v 64 decode");
    gdn(s, 64, 4, 8, 6, 1, gdn_opts{}, "S_v 64, T = 6");
    gdn_opts o;
    o.inplace = true;
    gdn(s, 32, 4, 4, 3, 2, o, "S_v 32, two sequences, in place");
    return s;
}
suite suite_chain() {
    suite s{};
    s.name = "chain";
    s.cfg = base_cfg(4u << 20);
    s.seed = 0xC4A1;
    const int g = gdn(s, 128, 16, 16, 1, 1, gdn_opts{}, "gdn");
    softplus(s, 128, 16, 0, 0, false, g, "softplus of the gdn output (dep)");
    return s;
}

// --- refused records and errors ------------------------------------------------------------------
struct refuse_case {
    const char * what;
    uint32_t kind;
    std::vector<uint32_t> a;
    uint32_t kflags;
    uint32_t want_ret;
};

bool run_errors() {
    bool ok = true;
    // A valid GDN record to patch.
    auto base_gdn = [] {
        suite s{};
        s.name = "err";
        s.cfg = base_cfg(4u << 20);
        s.seed = 1;
        gdn(s, 64, 4, 4, 1, 1, gdn_opts{}, "ok");
        return s;
    };
    struct patch {
        const char * what;
        uint32_t word;
        uint32_t value;
    };
    const patch ps[] = {
        {"S_v not a multiple of 16", opl::gdn::S_V, 60}, {"S_v 0", opl::gdn::S_V, 0}, {"S_v 272", opl::gdn::S_V, 272},
        {"H_k 0", opl::gdn::H_K, 0}, {"H_v not a multiple of H_k", opl::gdn::H_V, 6}, {"T 0", opl::gdn::N_TOKENS, 0},
        {"sequences 0", opl::gdn::N_SEQS, 0}, {"q head stride below L", opl::gdn::Q_HEAD_STRIDE, 3},
    };
    for (const patch & p : ps) {
        suite s = base_gdn();
        s.ops[0].a[p.word] = p.value;
        build(s);
        tsr_sim_opts so;
        memset(&so, 0, sizeof(so));
        so.sched = "fixed";
        so.race = 2;
        so.threads = 2;
        tsr_sim * sim = tsr_sim_create_opts(&s.cfg, &so);
        CHECK(sim != nullptr, "errors: simulator");
        fill(sim, s);
        fwh h;
        const bool booted = fwh_open(&h, sim, g_image) == 0;
        CHECK(booted, "errors: boot");
        if (booted) {
            const int gi = fwh_kernel(&h, "graph");
            const uint32_t a[7] = {s.opl_h, uint32_t(s.blob.size()), opl::LF_VERIFY_CRC, 0, 0, 0, 0};
            fwh_launch(&h, uint32_t(gi), a, 0);
            const bool refused = fwh_ret(&h) == int32_t(opl::RET_BAD_RECORD);
            CHECK(refused, "errors: GDN with %s: RET 0x%08x, expected BAD_RECORD", p.what, uint32_t(fwh_ret(&h)));
            ok = ok && refused;
        }
        fwh_close(&h);
        tsr_sim_destroy(sim);
    }
    // The reference refuses what the planner refuses.
    {
        suite s = base_gdn();
        s.ops[0].a[opl::gdn::S_V] = 60;
        mirror m;
        tsr_sim * sim = tsr_sim_create(&s.cfg, nullptr);
        make_mirror(sim, s, m);
        CHECK(tsr::ref::gdn_ref(m.view(), s.ops[0].a.data()) == tsr::ref::GD_BAD_RECORD, "errors: gdn_ref BAD_RECORD");
        tsr_sim_destroy(sim);
    }
    return ok;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: test_fw_gdn_rv IMAGE suite|errors [--scheds N]\n");
        return 2;
    }
    g_image = argv[1];
    if (FILE * f = fopen(g_image, "rb")) fclose(f);
    else {
        printf("SKIP: %s not found\n", g_image);
        return 77;
    }
    const std::string mode = argv[2];
    uint32_t n_sched = 8;
    for (int i = 3; i + 1 < argc; ++i)
        if (!strcmp(argv[i], "--scheds")) n_sched = uint32_t(atoi(argv[i + 1]));
    if (mode == "suite") {
        suite (*suites[])() = {suite_softplus, suite_softplus_small, suite_sconv, suite_sconv_small,
                               suite_gdn, suite_gdn_small, suite_chain};
        for (auto mk : suites) {
            suite s = mk();
            run_scheds(s, n_sched);
        }
        return test_finish("fw_gdn_rv");
    }
    if (mode == "errors") {
        run_errors();
        return test_finish("fw_gdn_rv errors");
    }
    fprintf(stderr, "unknown mode %s\n", mode.c_str());
    return 2;
}
