// test_fw_elem_rv.cpp — fw_elem_rv*: the element-wise and row kinds ADD, MUL, SCALE, GLU, UNARY,
// CPY, GET_ROWS (sw/ctrl/kernels/k_elem.cpp, k_glu.cpp, k_rows.cpp; docs/design/06-software-
// verification.md "SW" §7.2, §7.5) in the RISC-V image on tsr_rv, launched through the `graph`
// executor with op lists (SW §10.2 "fw_kernels_rv": per kind, 8 schedules, race fatal, outputs
// bit-exact against the references).
//
//   test_fw_elem_rv IMAGE suite [--scheds N] [--only K]
//       Op lists launched under the fixed schedule and 7 random ones with TSR_SIM_RACE fatal:
//         decode   decode shapes of Qwen2.5-0.5B / 32B and Gemma 4 12B (T = 1): residual ADD, norm-
//                  weight MUL (broadcast), QKV bias, per-head rows, one scalar, a per-row scalar,
//                  SCALE (softcap over 262,144 logits, with bias), GLU (SWIGLU, GEGLU16, GEGLU),
//                  UNARY (all five), CPY, GET_ROWS (one id), chains with deps (fast paths);
//         prefill  prefill shapes (T up to 512): dense and strided rows, b periods (blocks), the
//                  scalar forms (one scalar over many rows: SOP.LDS; per-row; periods), multi-tile
//                  ops, GET_ROWS runs, repeated and reversed ids, strided CPY;
//         small    a 512 KiB SPM (budget 3,376 lines): tile boundaries of plan_elem.hpp for every
//                  operand-b form, single-buffer tiles, chunked CPY / GET_ROWS rows, the ring
//                  fallback (no room beside a live tile);
//         special  NaN (payloads, signs), +-inf, +-0, subnormals, FLT_MAX, the GELU16 clamps and
//                  the TANH small path in every element-wise kind (VU §4.1 rules).
//       Every dst row is compared bit for bit with sw/ctrl/ref/elem_ref.cpp applied to the same DRAM
//       in list order; DONE must be OK / RET 0; the SW §4.7 observables (per-unit op streams, the
//       RISC-V SPM store stream, the final written DRAM and SPM state, DONE r0-r6) must be equal
//       under every schedule; the written DRAM must be exactly the dst rows; no race report.
//   test_fw_elem_rv IMAGE shapes [--scheds N]
//       (slow) The full model shapes: prefill ubatch 512 of Qwen2.5-0.5B (QKV bias, norm weight,
//       SWIGLU 4864), Qwen2.5-32B (residual, QKV bias 7168, norm weight, SWIGLU 27648, CPY) and Gemma
//       4 12B (inp_scale_rows, out_scale, per-head norm weight, residual, GEGLU16 15360, GET_ROWS of 512
//       ids) and the 262,144-logit softcap chain; fixed + 7 random schedules (~30 s: label slow).
//   test_fw_elem_rv IMAGE errors
//       Refused records (BAD_RECORD: bad arguments, kind flags, GET_ROWS ids outside the preload
//       area; SCRATCH: a row larger than the ring budget) end the launch with that code after the
//       earlier ops ran; a GET_ROWS id that is negative or beyond the channel ends it with TILE_ERR
//       DRAM_RANGE on that op (RET = op + 1).
//   test_fw_elem_rv IMAGE perf [--u200]
//       RISC-V cycles of work per record of each kind at decode shapes (fw_prof.h: busy cycles up to
//       the launch's last op push, wait-loop iterations after their first test excluded; two list
//       lengths, the difference per record), Ibex maxperf by default (TSR_SIM_RV_CORE), deep queues
//       (U200 queue depths with --u200); target <= 350 (W2b). Prints a profile per kind.
// Exit 0 when everything passed, 1 on a failure, 77 (skipped) when the image is missing.
#undef _GNU_SOURCE
#include "test_sim_obs.h"

#include <algorithm>
#include <functional>
#include <string>
#include <vector>

#include "../ref/elem_ref.hpp"
#include "../ref/fw_sim_host.hpp"
#include "fw_prof.h"
#include "tsr/opl.hpp"
#include "tsr/plan.hpp"
#include "tsr_race.h"

namespace {

namespace opl = tsr::opl;
constexpr uint64_t LINE = 64u;
constexpr uint32_t NONE = opl::HANDLE_NONE;

uint32_t H(uint32_t ch, uint64_t beat) { return (ch << 28) | uint32_t(beat & 0x0FFFFFFFu); }
uint32_t L16(uint32_t n) { return (n + 15u) / 16u; }

const char * g_image = nullptr;

// ---------------------------------------------------------------------------
// Data
// ---------------------------------------------------------------------------
// A mix of finite values and the VU §4.1 specials (special != 0: about 1 in 4 elements).
uint32_t special_word(rng_t * r) {
    static const uint32_t sp[] = {
        0x00000000u, 0x80000000u, 0x7F800000u, 0xFF800000u, 0x7FC00000u, 0xFFC00001u, 0x7F800001u, 0xFFFFFFFFu,
        0x00000001u, 0x807FFFFFu, 0x00400000u, 0x80000010u, 0x7F7FFFFFu, 0xFF7FFFFFu, 0x00800000u, 0x80800000u,
        0x41200000u /* 10 */, 0xC1200000u /* -10 */, 0x41300000u /* 11 */, 0xC1100000u /* -9 */, 0x42B00000u /* 88 */,
        0xC2B00000u /* -88 */, 0x3C7FFFFFu /* just below 2^-6 */, 0x3C800000u, 0xBC7F0000u, 0x7149F2CAu /* 1e30 */,
        0xF149F2CAu, 0x3F800000u, 0xBF800000u};
    return sp[rng_below(r, uint32_t(sizeof(sp) / sizeof(sp[0])))];
}
uint32_t data_word(rng_t * r, bool special, float scale) {
    if (special && rng_below(r, 4) == 0) return special_word(r);
    return f2u(float(rng_normal(r)) * scale);
}

// ---------------------------------------------------------------------------
// One op list (a suite), its inputs and its expected outputs
// ---------------------------------------------------------------------------
struct region {                         // DRAM rows the list reads (filled) or writes (checked)
    uint32_t h, rows, stride, lines;    // rows of `lines` lines, `stride` lines apart
};
struct op_t {
    uint32_t kind;
    std::vector<uint32_t> a;
    std::vector<uint32_t> deps;         // op indices of the list (writer.add folds > 2 into JOINs)
    uint32_t kflags = 0;
    uint32_t index = 0;                 // its op index in the list
    region out{};                       // its dst rows
    const char * what = "";
};
struct suite {
    const char * name;
    tsr_config cfg;
    uint64_t seed;
    bool special = false;
    std::vector<region> in;             // filled with data (rows' first n words; the rest random bits)
    std::vector<uint32_t> in_n;         // words per row of each input (data), the rest raw bits
    std::vector<op_t> ops;
    std::vector<uint8_t> ids;           // every GET_ROWS op's ids: one preload entry (a graph's ids tensor)
    uint32_t ids_src = 0;
    std::vector<uint8_t> blob;
    uint32_t opl_h = H(0, 0x40000u);    // ch0 16 MiB: the OPL region (EX §11)
    std::vector<uint8_t> pre_area;      // the preload area as the executor lays it out (for elem_ref)
    uint64_t io[4] = {0x100000u, 0x40000u, 0x40000u, 0x40000u};   // next free beat per channel
    uint32_t next_ch = 0;
    uint32_t take(uint32_t ch, uint64_t lines) {
        const uint64_t b = io[ch];
        io[ch] += (lines + 7u) & ~7ull;
        return H(ch, b);
    }
    uint32_t ch() { return next_ch++ & 3u; }
};

tsr_config base_cfg(uint32_t spm_bytes) {
    tsr_config c = TSR_CONFIG_U200_V1;
    c.t_max = 8;
    c.spm_bytes = spm_bytes;
    c.ch_bytes = 1ull << 30;
    c.caps = TSR_SIM_CAPS_IMPLEMENTED;
    return c;
}

// An input tensor of rows x (n words) with `pad` extra lines per row (stride = L(n) + pad); filled.
uint32_t input(suite & s, uint32_t n, uint32_t rows, uint32_t pad, uint32_t * stride, uint32_t lines = 0) {
    const uint32_t l = lines ? lines : L16(n);
    *stride = l + pad;
    const uint32_t h = s.take(s.ch(), uint64_t(rows) * (l + pad));
    s.in.push_back(region{h, rows, l + pad, l});
    s.in_n.push_back(n);
    return h;
}
uint32_t output(suite & s, uint32_t lines, uint32_t rows, uint32_t pad, uint32_t * stride) {
    *stride = lines + pad;
    return s.take(s.ch(), uint64_t(rows) * (lines + pad));
}

// --- record builders (each returns the op's position in s.ops) ---
struct src_ref { int op = -1; };       // an input that is an earlier op's output (dep)

int add_op(suite & s, uint32_t kind, std::vector<uint32_t> a, region out, std::vector<uint32_t> deps, const char * what) {
    op_t o;
    o.kind = kind;
    o.a = std::move(a);
    o.out = out;
    o.deps = std::move(deps);
    o.what = what;
    s.ops.push_back(o);
    return int(s.ops.size()) - 1;
}

// ADD / MUL: a rows x n (+ a_pad), b by period p (scalar: f32 vector of max(p, 1) words; else p rows,
// or one row when p = 1 / b_stride0), dst (+ d_pad). from >= 0: a = that op's dst rows (dep).
int ew(suite & s, uint32_t kind, uint32_t n, uint32_t rows, uint32_t p, bool scalar, uint32_t a_pad = 0, uint32_t b_pad = 0,
       uint32_t d_pad = 0, bool b_stride0 = false, int from = -1, const char * what = "") {
    uint32_t as, bs = 0, ds;
    uint32_t ha;
    std::vector<uint32_t> deps;
    if (from >= 0) {
        ha = s.ops[size_t(from)].out.h;
        as = s.ops[size_t(from)].out.stride;
        deps.push_back(uint32_t(from));
    } else {
        ha = input(s, n, rows, a_pad, &as);
    }
    uint32_t hb;
    if (scalar) {
        hb = input(s, std::max(p, 1u), 1u, 0u, &bs);
        bs = 0;
    } else {
        const uint32_t brows = (p == 1u || b_stride0) ? 1u : std::min(p, rows);
        hb = input(s, n, brows, b_pad, &bs);
        if (b_stride0) bs = 0;
    }
    const uint32_t hd = output(s, L16(n), rows, d_pad, &ds);
    return add_op(s, kind, {hd, ds, ha, as, hb, bs, n, rows, p, scalar ? opl::ew::FLAG_B_SCALAR : 0u},
                  region{hd, rows, ds, L16(n)}, deps, what);
}
int scale(suite & s, uint32_t n, uint32_t rows, float sv, float bv, uint32_t pad = 0, int from = -1, const char * what = "") {
    uint32_t ss, ds, hs;
    std::vector<uint32_t> deps;
    if (from >= 0) {
        hs = s.ops[size_t(from)].out.h;
        ss = s.ops[size_t(from)].out.stride;
        deps.push_back(uint32_t(from));
    } else {
        hs = input(s, n, rows, pad, &ss);
    }
    const uint32_t hd = output(s, L16(n), rows, pad, &ds);
    return add_op(s, opl::K_SCALE, {hd, ds, hs, ss, n, rows, f2u(sv), f2u(bv)}, region{hd, rows, ds, L16(n)}, deps, what);
}
int glu(suite & s, uint32_t variant, uint32_t n, uint32_t rows, uint32_t pad = 0, const char * what = "") {
    uint32_t gs, us, ds;
    const uint32_t hg = input(s, n, rows, pad, &gs), hu = input(s, n, rows, pad, &us);
    const uint32_t hd = output(s, L16(n), rows, pad, &ds);
    return add_op(s, opl::K_GLU, {hd, ds, hg, gs, hu, us, n, rows, variant}, region{hd, rows, ds, L16(n)}, {}, what);
}
int unary(suite & s, uint32_t func, uint32_t n, uint32_t rows, uint32_t pad = 0, int from = -1, const char * what = "") {
    uint32_t ss, ds, hs;
    std::vector<uint32_t> deps;
    if (from >= 0) {
        hs = s.ops[size_t(from)].out.h;
        ss = s.ops[size_t(from)].out.stride;
        deps.push_back(uint32_t(from));
    } else {
        hs = input(s, n, rows, pad, &ss);
    }
    const uint32_t hd = output(s, L16(n), rows, pad, &ds);
    return add_op(s, opl::K_UNARY, {hd, ds, hs, ss, n, rows, func}, region{hd, rows, ds, L16(n)}, deps, what);
}
int cpy(suite & s, uint32_t rl, uint32_t rows, uint32_t s_pad = 0, uint32_t d_pad = 0, int from = -1, const char * what = "") {
    uint32_t ss, ds, hs;
    std::vector<uint32_t> deps;
    if (from >= 0) {
        hs = s.ops[size_t(from)].out.h;
        ss = s.ops[size_t(from)].out.stride;
        deps.push_back(uint32_t(from));
    } else {
        hs = input(s, 16u * rl, rows, s_pad, &ss, rl);
    }
    const uint32_t hd = output(s, rl, rows, d_pad, &ds);
    return add_op(s, opl::K_CPY, {hd, ds, hs, ss, rl, rows}, region{hd, rows, ds, rl}, deps, what);
}
// GET_ROWS of ids (into a src tensor of src_rows rows of rl lines); ids_off is set by build().
int get_rows(suite & s, uint32_t rl, uint32_t src_rows, const std::vector<int32_t> & ids, uint32_t s_pad = 0,
             uint32_t d_pad = 0, const char * what = "") {
    uint32_t ss, ds;
    const uint32_t hs = input(s, 16u * rl, src_rows, s_pad, &ss, rl);
    const uint32_t n = uint32_t(ids.size());
    const uint32_t hd = output(s, rl, n, d_pad, &ds);
    const uint32_t off = uint32_t(s.ids.size());
    s.ids.resize(s.ids.size() + 4u * n);
    for (uint32_t i = 0; i < n; ++i) memcpy(s.ids.data() + off + 4u * i, &ids[i], 4);
    // IDS: the byte offset in the ids entry; build() adds the entry's pre_byte.
    return add_op(s, opl::K_GET_ROWS, {hd, ds, hs, ss, off, n, rl}, region{hd, n, ds, rl}, {}, what);
}

// The list (one preload entry holding every GET_ROWS op's ids; its pre_byte added to the ids words,
// once: build() may run again after a test patched a record).
void build(suite & s) {
    opl::writer w(opl::F_STOP_ON_ERROR, 0xE1E3u);
    uint32_t pre_byte = 0;
    if (!s.ids.empty()) {
        if (s.ids_src == 0u) s.ids_src = s.take(0, (s.ids.size() + 63u) / 64u);
        pre_byte = w.add_preload(s.ids_src, uint32_t(s.ids.size()));
    }
    s.pre_area.assign(size_t(w.pre_lines()) * 64u, 0u);
    if (!s.ids.empty()) memcpy(s.pre_area.data() + pre_byte, s.ids.data(), s.ids.size());
    for (auto & o : s.ops)
        o.index = w.add(o.kind, o.a.data(), uint32_t(o.a.size()), o.deps.data(), uint32_t(o.deps.size()), o.kflags);
    s.blob = w.finish();
    CHECK(w.ok() && !s.blob.empty(), "%s: OPL writer: %s", s.name, w.error());
}

// Fills the inputs (deterministic for the seed), the preload sources and the list.
void fill(tsr_sim * sim, const suite & s) {
    rng_t r = {s.seed};
    for (size_t i = 0; i < s.in.size(); ++i) {
        const region & g = s.in[i];
        const float scale = float(0.25 + 4.0 * rng_unit(&r));
        for (uint32_t row = 0; row < g.rows; ++row) {
            uint8_t * p = tsr_sim_dram(sim, g.h >> 28) + (uint64_t(g.h & 0x0FFFFFFFu) + uint64_t(row) * g.stride) * LINE;
            const uint32_t words = 16u * g.stride;
            for (uint32_t j = 0; j < words; ++j) {
                const uint32_t v = j < s.in_n[i] ? data_word(&r, s.special, scale) : rng_u32(&r);
                memcpy(p + 4u * j, &v, 4);
            }
        }
    }
    if (!s.ids.empty())
        tsr_sim_dram_write(sim, s.ids_src >> 28, uint64_t(s.ids_src & 0x0FFFFFFFu) * LINE, s.ids.data(), s.ids.size());
    tsr_sim_dram_write(sim, s.opl_h >> 28, uint64_t(s.opl_h & 0x0FFFFFFFu) * LINE, s.blob.data(), s.blob.size());
}

// A host copy of the used DRAM (per channel [0, io[ch]) beats): the reference runs on it.
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

// The DRAM beats the list writes: every op's dst rows.
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
        for (uint32_t r = 0; r < o.out.rows; ++r) {
            const uint64_t lo = (o.out.h & 0x0FFFFFFFu) + uint64_t(r) * o.out.stride;
            v.push_back(ival_t{o.out.h >> 28, lo, lo + o.out.lines});
        }
    return merge(v);
}

struct run_obs {
    obs o;
    uint32_t done[7];
};

const char * kind_name(uint32_t k) {
    switch (k) {
        case opl::K_ADD: return "ADD";
        case opl::K_MUL: return "MUL";
        case opl::K_SCALE: return "SCALE";
        case opl::K_GLU: return "GLU";
        case opl::K_GET_ROWS: return "GET_ROWS";
        case opl::K_CPY: return "CPY";
        case opl::K_UNARY: return "UNARY";
        default: return "?";
    }
}

// One launch of the suite's list under `sched` on a fresh simulator; expected rows from mirror m
// (computed on the first run).
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
            const int rc = tsr::ref::elem_ref(v, o.kind, o.a.data(), uint32_t(o.a.size()), s.pre_area.data(), s.pre_area.size());
            CHECK(rc == tsr::ref::EL_OK, "%s: elem_ref refused op %u (%s %s): %d", s.name, o.index, kind_name(o.kind), o.what, rc);
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
            for (uint32_t r = 0; r < o.out.rows; ++r) {
                const uint64_t off = (uint64_t(o.out.h & 0x0FFFFFFFu) + uint64_t(r) * o.out.stride) * LINE;
                const uint8_t * got = tsr_sim_dram(sim, o.out.h >> 28) + off;
                const uint8_t * want = m.ch[o.out.h >> 28].data() + off;
                if (memcmp(got, want, size_t(o.out.lines) * LINE) != 0) {
                    size_t k = 0;
                    while (got[k] == want[k]) ++k;
                    CHECK(0, "%s [%s]: op %u (%s %s) row %u: word %zu is 0x%08x, expected 0x%08x", s.name, sched, o.index,
                          kind_name(o.kind), o.what, r, k / 4u, get_u32_le(got + (k & ~size_t(3))),
                          get_u32_le(want + (k & ~size_t(3))));
                    break;
                }
            }
        }
        obs_finish(&out->o, sim);
        const std::vector<ival_t> want = written_rows(s);
        const std::vector<ival_t> got = merge(std::vector<ival_t>(out->o.dram_wr, out->o.dram_wr + out->o.n_dram));
        bool same = got.size() == want.size();
        for (size_t i = 0; same && i < got.size(); ++i)
            same = got[i].ch == want[i].ch && got[i].lo == want[i].lo && got[i].hi == want[i].hi;
        CHECK(same, "%s [%s]: the written DRAM (%zu ranges) is not exactly the dst rows (%zu ranges)", s.name, sched,
              got.size(), want.size());
        const tsr_race_report * rr = nullptr;
        const size_t reports = tsr_sim_race_reports(sim, &rr);
        CHECK(reports == 0, "%s [%s]: %zu race reports", s.name, sched, reports);
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
        else snprintf(sched, sizeof(sched), "random:%u", 2000u + k);
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
using opl::K_ADD;
using opl::K_MUL;
namespace un = opl::unary;
namespace gl = opl::glu;

// Decode shapes (T = 1): the fast paths.
suite suite_decode() {
    suite s{};
    s.name = "decode";
    s.cfg = base_cfg(4u << 20);
    s.seed = 0xDEC0DE;
    // Qwen2.5-0.5B (896 / 4864, 14 heads x 64), 32B (5120 / 27648, 40 x 128), Gemma 4 12B (3840 / 15360,
    // 16 x 256 SWA, 16 x 512 global; vocab 262,144).
    const int q05 = ew(s, K_ADD, 896, 1, 1, false, 0, 0, 0, false, -1, "0.5B residual");
    ew(s, K_MUL, 896, 1, 1, false, 0, 0, 0, false, q05, "0.5B norm weight (chain)");
    ew(s, K_ADD, 1152, 1, 1, false, 0, 0, 0, false, -1, "0.5B QKV bias");
    ew(s, K_ADD, 5120, 1, 1, false, 0, 0, 0, false, -1, "32B residual");
    ew(s, K_ADD, 7168, 1, 1, false, 0, 0, 0, false, -1, "32B QKV bias");
    ew(s, K_ADD, 3840, 1, 1, false, 0, 0, 0, false, -1, "12B residual");
    ew(s, K_MUL, 3840, 1, 1, true, 0, 0, 0, false, -1, "12B inp_scale_rows (per-row scalar, T = 1)");
    ew(s, K_MUL, 3840, 1, 1, true, 0, 0, 0, false, -1, "12B out_scale (one scalar)");
    ew(s, K_ADD, 896, 1, 1, true, 0, 0, 0, false, -1, "scalar ADD");
    ew(s, K_MUL, 256, 16, 1, false, 0, 0, 0, false, -1, "12B per-head weight (16 rows, broadcast)");
    ew(s, K_MUL, 512, 16, 16, false, 0, 0, 0, false, -1, "per-head rows form (p = rows)");
    ew(s, K_ADD, 64, 14, 14, true, 0, 0, 0, false, -1, "per-row scalars, 14 rows");
    ew(s, K_MUL, 128, 40, 1, false, 0, 0, 0, true, -1, "b stride 0 (broadcast)");
    const int sc = scale(s, 262144, 1, 1.0f / 30.0f, 0.0f, 0, -1, "12B softcap 1/cap");
    const int th = unary(s, un::TANH, 262144, 1, 0, sc, "12B softcap tanh (chain)");
    scale(s, 262144, 1, 30.0f, 0.0f, 0, th, "12B softcap cap (chain)");
    scale(s, 5120, 1, 0.125f, -1.5f, 0, -1, "SCALE with bias");
    glu(s, gl::SWIGLU, 4864, 1, 0, "0.5B SWIGLU");
    glu(s, gl::SWIGLU, 27648, 1, 0, "32B SWIGLU");
    glu(s, gl::GEGLU16, 15360, 1, 0, "12B GEGLU16");
    glu(s, gl::GEGLU, 15360, 1, 0, "GEGLU (exact)");
    for (uint32_t f = 0; f <= un::FUNC_MAX; ++f) unary(s, f, 896 + 16u * f + f, 1, 0, -1, "unary");
    cpy(s, 56, 1, 0, 0, -1, "CPY 0.5B row");
    cpy(s, 240, 1, 0, 0, q05 >= 0 ? -1 : -1, "CPY 12B row");
    get_rows(s, 56, 7, {5}, 0, 0, "GET_ROWS 0.5B inp_out_ids");
    get_rows(s, 320, 3, {0}, 0, 0, "GET_ROWS 32B");
    get_rows(s, 240, 4, {3}, 2, 0, "GET_ROWS 12B (src stride 242)");
    // W2b fix (review): the s-register shadow's clobber rule (exec.hpp kind_ctx::sreg_valid): a SCALE
    // sets s12 (SOP.MOVI); a MUL / ADD with one scalar for several rows loads s12 with SOP.LDS (ew_run's
    // sreg_clobber); the next SCALE with the same s (and b in s13) must push its MOVI again, else its
    // MULC reads the scalar. elem_ref's values catch a missing clobber.
    scale(s, 896, 1, 0.75f, 0.0f, 0, -1, "shadow: SCALE s 0.75 (MOVI s12)");
    ew(s, K_MUL, 896, 4, 1, true, 0, 0, 0, false, -1, "shadow: one scalar, 4 rows (SOP.LDS s12)");
    scale(s, 896, 1, 0.75f, 0.0f, 0, -1, "shadow: SCALE s 0.75 again (MOVI after the LDS)");
    scale(s, 896, 1, 0.75f, 1.0f, 0, -1, "shadow: SCALE s 0.75 b 1 (s12, s13)");
    ew(s, K_ADD, 896, 3, 1, true, 0, 0, 0, false, -1, "shadow: one scalar ADD, 3 rows (SOP.LDS s12)");
    scale(s, 896, 1, 0.75f, 1.0f, 0, -1, "shadow: SCALE s 0.75 b 1 again");
    return s;
}

// Prefill shapes (T up to 512) and the general forms.
suite suite_prefill() {
    suite s{};
    s.name = "prefill";
    s.cfg = base_cfg(4u << 20);
    s.seed = 0x9EF111;
    ew(s, K_ADD, 896, 512, 512, false, 0, 0, 0, false, -1, "0.5B residual T = 512");
    ew(s, K_ADD, 1152, 512, 1, false, 0, 0, 0, false, -1, "0.5B QKV bias T = 512 (broadcast)");
    ew(s, K_MUL, 3840, 128, 1, true, 0, 0, 0, false, -1, "one scalar, 128 rows (SOP.LDS)");
    ew(s, K_MUL, 3840, 512, 512, true, 0, 0, 0, false, -1, "inp_scale_rows T = 512 (per-row, tiles of 16k)");
    ew(s, K_ADD, 64, 14 * 37, 14, false, 0, 0, 0, false, -1, "period 14 (blocks)");
    ew(s, K_MUL, 128, 40 * 9, 40, true, 0, 0, 0, false, -1, "scalar period 40");
    ew(s, K_ADD, 300, 33, 33, false, 3, 2, 1, false, -1, "strided a / b / dst rows");
    ew(s, K_MUL, 300, 33, 1, false, 5, 0, 4, false, -1, "strided, broadcast b");
    ew(s, K_ADD, 5120, 64, 64, false, 0, 0, 0, false, -1, "32B residual T = 64 (2 tiles)");
    ew(s, K_ADD, 17, 70, 5, false, 1, 1, 1, false, -1, "n = 17, period 5, strided");
    scale(s, 896, 300, -0.5f, 0.0f, 0, -1, "SCALE T = 300");
    scale(s, 896, 31, 2.0f, 0.75f, 3, -1, "SCALE bias, strided");
    glu(s, gl::SWIGLU, 4864, 64, 0, "0.5B SWIGLU T = 64");
    glu(s, gl::GEGLU16, 15360, 8, 0, "12B GEGLU16 T = 8");
    glu(s, gl::GEGLU, 100, 50, 2, "GEGLU strided");
    for (uint32_t f = 0; f <= un::FUNC_MAX; ++f) unary(s, f, 33 + 50u * f, 40 + f, f & 1u, -1, "unary rows");
    cpy(s, 56, 512, 0, 0, -1, "CPY dense 512 rows (merged)");
    cpy(s, 13, 40, 3, 1, -1, "CPY strided");
    cpy(s, 7, 9, 0, 2, -1, "CPY dst strided");
    get_rows(s, 56, 512, [] { std::vector<int32_t> v; for (int i = 0; i < 300; ++i) v.push_back(i + 100); return v; }(), 0, 0,
             "GET_ROWS run of 300");
    get_rows(s, 20, 64, {63, 62, 1, 2, 3, 3, 3, 0, 10, 11}, 0, 0, "GET_ROWS mixed ids");
    get_rows(s, 20, 64, {7, 8, 9, 40, 41}, 4, 3, "GET_ROWS strided src / dst");
    return s;
}

// A small SPM (512 KiB: ring budget 65,536 / 8 - 4,368 - 448 (the one-tile block, plan.hpp) = 3,376 lines,
// half 1,688): tile boundaries.
suite suite_small() {
    suite s{};
    s.name = "small";
    s.cfg = base_cfg(512u << 10);
    s.seed = 0x5A11;
    // L = 56: rows form 112 lines per row -> 15 rows per tile; broadcast 56 per row + 56 -> 29 rows.
    ew(s, K_ADD, 896, 60, 60, false, 0, 0, 0, false, -1, "rows form, 4 tiles");
    ew(s, K_MUL, 896, 60, 1, false, 0, 0, 0, false, -1, "broadcast, 2 tiles");
    ew(s, K_ADD, 896, 60, 7, false, 0, 0, 0, false, -1, "period 7: tiles of 28 rows");
    ew(s, K_MUL, 896, 100, 1, true, 0, 0, 0, false, -1, "one scalar, 4 tiles (s12 from tile 0)");
    ew(s, K_MUL, 896, 100, 100, true, 0, 0, 0, false, -1, "per-row scalars, tiles of 32 rows");
    ew(s, K_ADD, 896, 50, 9, true, 0, 0, 0, false, -1, "scalar period 9");
    ew(s, K_ADD, 16000, 3, 3, false, 0, 0, 0, false, -1, "L = 1000: one row per tile, single buffer");
    ew(s, K_MUL, 26000, 1, 1, false, 0, 0, 0, false, -1, "L = 1625: 3,250 lines, one tile > half");
    scale(s, 896, 70, 3.0f, 1.0f, 0, -1, "SCALE 3 tiles");
    glu(s, gl::GEGLU16, 896, 40, 0, "GLU 3 tiles");
    unary(s, un::SILU, 896, 70, 1, -1, "UNARY strided, 3 tiles");
    cpy(s, 1000, 7, 0, 0, -1, "CPY merged 7,000 lines: chunks of 1,688");
    cpy(s, 3000, 2, 1, 1, -1, "CPY strided rows of 3,000 lines: chunked");
    cpy(s, 500, 9, 1, 0, -1, "CPY strided 3 rows per tile");
    get_rows(s, 2500, 4, {3, 0, 2}, 0, 0, "GET_ROWS chunked rows");
    get_rows(s, 300, 20, {1, 2, 3, 4, 5, 6, 7, 19}, 0, 0, "GET_ROWS 6 rows per tile");
    return s;
}

// Every element-wise kind on the special values (VU §4.1).
suite suite_special() {
    suite s{};
    s.name = "special";
    s.cfg = base_cfg(4u << 20);
    s.seed = 0x5EC1A1;
    s.special = true;
    ew(s, K_ADD, 1000, 3, 3, false, 0, 0, 0, false, -1, "ADD specials");
    ew(s, K_MUL, 1000, 3, 1, false, 0, 0, 0, false, -1, "MUL specials");
    ew(s, K_ADD, 1000, 3, 3, true, 0, 0, 0, false, -1, "ADDC specials");
    ew(s, K_MUL, 1000, 3, 1, true, 0, 0, 0, false, -1, "MULC specials");
    scale(s, 1000, 3, 1e-30f, 0.0f, 0, -1, "SCALE tiny s");
    scale(s, 1000, 3, -3.0e38f, 2.0f, 0, -1, "SCALE huge s, bias");
    scale(s, 1000, 2, 1.0f, -0.0f, 0, -1, "SCALE bias -0 (bits != 0: ADDC)");
    for (uint32_t v = 0; v <= gl::VARIANT_MAX; ++v) glu(s, v, 1000, 3, 0, "GLU specials");
    for (uint32_t f = 0; f <= un::FUNC_MAX; ++f) unary(s, f, 1000, 3, 0, -1, "UNARY specials");
    cpy(s, 63, 3, 0, 0, -1, "CPY specials (bits)");
    return s;
}

// The full model shapes (slow): prefill ubatch 512 of Qwen2.5-0.5B, Qwen2.5-32B and Gemma 4 12B, and
// the decode vocab-sized softcap.
suite suite_shapes() {
    suite s{};
    s.name = "shapes";
    s.cfg = base_cfg(4u << 20);
    s.seed = 0x5BA9E5;
    const uint32_t T = 512;
    // Qwen2.5-0.5B
    ew(s, K_ADD, 1152, T, 1, false, 0, 0, 0, false, -1, "0.5B QKV bias T 512");
    ew(s, K_MUL, 896, T, 1, false, 0, 0, 0, false, -1, "0.5B norm weight T 512");
    glu(s, gl::SWIGLU, 4864, T, 0, "0.5B SWIGLU T 512");
    // Qwen2.5-32B
    ew(s, K_ADD, 5120, T, T, false, 0, 0, 0, false, -1, "32B residual T 512");
    ew(s, K_ADD, 7168, T, 1, false, 0, 0, 0, false, -1, "32B QKV bias T 512");
    ew(s, K_MUL, 5120, T, 1, false, 0, 0, 0, false, -1, "32B norm weight T 512");
    glu(s, gl::SWIGLU, 27648, T, 0, "32B SWIGLU T 512");
    cpy(s, 320, T, 0, 0, -1, "32B CPY T 512");
    // Gemma 4 12B
    ew(s, K_MUL, 3840, T, T, true, 0, 0, 0, false, -1, "12B inp_scale_rows T 512");
    ew(s, K_MUL, 3840, T, 1, true, 0, 0, 0, false, -1, "12B out_scale T 512 (one scalar)");
    ew(s, K_MUL, 256, 16 * T, 1, false, 0, 0, 0, false, -1, "12B q norm weight, 16 heads x 512");
    ew(s, K_ADD, 3840, T, T, false, 0, 0, 0, false, -1, "12B residual T 512");
    glu(s, gl::GEGLU16, 15360, T, 0, "12B GEGLU16 T 512");
    {
        std::vector<int32_t> ids;
        for (int i = 0; i < int(T); ++i) ids.push_back(i);
        get_rows(s, 240, T, ids, 0, 0, "12B inp_out_ids T 512");
    }
    const int sc = scale(s, 262144, 1, 1.0f / 30.0f, 0.0f, 0, -1, "12B softcap 1/cap (decode)");
    const int th = unary(s, un::TANH, 262144, 1, 0, sc, "12B softcap tanh");
    scale(s, 262144, 1, 30.0f, 0.0f, 0, th, "12B softcap cap");
    return s;
}

// ---------------------------------------------------------------------------
// errors
// ---------------------------------------------------------------------------
struct err_case {
    const char * what;
    std::function<void(suite &)> make;   // builds ops; the last op is the one that fails
    uint32_t ret;                        // expected RET (KERNEL_ERR code, or op index + 1)
    uint32_t err_code;                   // expected first error (TILE_ERR), 0 for KERNEL_ERR
};

bool run_error(const err_case & c) {
    suite s{};
    s.name = c.what;
    s.cfg = base_cfg(512u << 10);
    s.seed = 0xE44;
    c.make(s);
    build(s);
    tsr_sim_opts so;
    memset(&so, 0, sizeof(so));
    so.sched = "fixed";
    so.race = 2;
    tsr_sim * sim = tsr_sim_create_opts(&s.cfg, &so);
    if (sim == nullptr) return false;
    fill(sim, s);
    fwh h;
    bool ok = fwh_open(&h, sim, g_image) == 0;
    if (ok) {
        const uint32_t a[7] = {s.opl_h, uint32_t(s.blob.size()), opl::LF_VERIFY_CRC, 0, 0, 0, 0};
        ok = fwh_launch(&h, uint32_t(fwh_kernel(&h, "graph")), a, 0) == 0;
    }
    CHECK(ok, "errors: %s: launch", c.what);
    if (ok) {
        const uint32_t st = fwh_status(&h), ret = uint32_t(fwh_ret(&h));
        if (c.err_code != 0u) {
            const uint32_t code = TSR_GET(h.done[0], TSR_LCQ_R0_ERR_CODE);
            CHECK(st == TSR_LCQ_R0_STATUS_TILE_ERR && ret == c.ret && code == c.err_code,
                  "errors: %s: DONE status %u RET 0x%08x code %u (expected TILE_ERR, RET %u, code %u)", c.what, st, ret, code,
                  c.ret, c.err_code);
        } else {
            CHECK(st == TSR_LCQ_R0_STATUS_KERNEL_ERR && ret == c.ret, "errors: %s: DONE status %u RET 0x%08x (expected 0x%08x)",
                  c.what, st, ret, c.ret);
        }
        // The ops before the failing one ran: their rows are the reference's.
        mirror m;
        if (s.ops.size() > 1u) {
            tsr_sim * ref = tsr_sim_create_opts(&s.cfg, &so);
            fill(ref, s);
            make_mirror(ref, s, m);
            tsr_sim_destroy(ref);
            for (size_t i = 0; i + 1u < s.ops.size(); ++i) {
                const op_t & o = s.ops[i];
                tsr::ref::elem_ref(m.view(), o.kind, o.a.data(), uint32_t(o.a.size()), s.pre_area.data(), s.pre_area.size());
                for (uint32_t r = 0; r < o.out.rows; ++r) {
                    const uint64_t off = (uint64_t(o.out.h & 0x0FFFFFFFu) + uint64_t(r) * o.out.stride) * LINE;
                    CHECK(memcmp(tsr_sim_dram(sim, o.out.h >> 28) + off, m.ch[o.out.h >> 28].data() + off,
                                 size_t(o.out.lines) * LINE) == 0,
                          "errors: %s: op %zu (before the failing op) row %u differs", c.what, i, r);
                }
            }
        }
        const tsr_race_report * rr = nullptr;
        CHECK(tsr_sim_race_reports(sim, &rr) == 0, "errors: %s: race reports", c.what);
        printf("errors: %s: status %u RET 0x%08x ERR_INFO 0x%08x\n", c.what, st, ret, h.done[4]);
    }
    fwh_close(&h);
    tsr_sim_destroy(sim);
    return ok;
}

int errors_main() {
    const uint32_t BAD = opl::RET_BAD_RECORD, SCR = opl::RET_SCRATCH;
    auto good = [](suite & s) { ew(s, K_ADD, 64, 2, 2, false); };
    auto patch = [](suite & s, uint32_t w, uint32_t v) { s.ops.back().a[w] = v; };
    std::vector<err_case> cases = {
        {"ADD n = 0", [&](suite & s) { good(s); ew(s, K_ADD, 64, 1, 1, false); patch(s, opl::ew::N, 0); }, BAD, 0},
        {"ADD rows = 0", [&](suite & s) { good(s); ew(s, K_ADD, 64, 1, 1, false); patch(s, opl::ew::ROWS, 0); }, BAD, 0},
        {"ADD period 0", [&](suite & s) { good(s); ew(s, K_ADD, 64, 1, 1, false); patch(s, opl::ew::B_PERIOD, 0); }, BAD, 0},
        {"MUL unknown flag", [&](suite & s) { good(s); ew(s, K_MUL, 64, 1, 1, false); patch(s, opl::ew::FLAGS, 2); }, BAD, 0},
        {"ADD n > 1,048,560", [&](suite & s) { good(s); ew(s, K_ADD, 64, 1, 1, false); patch(s, opl::ew::N, 0x100000u); }, BAD, 0},
        {"ADD overlapping dst rows", [&](suite & s) { good(s); ew(s, K_ADD, 64, 2, 2, false); patch(s, opl::ew::DST_STRIDE, 3); }, BAD, 0},
        {"ADD kind flags", [&](suite & s) { good(s); ew(s, K_ADD, 64, 1, 1, false); s.ops.back().kflags = 1; }, BAD, 0},
        {"SCALE kind flags (fast shape)", [&](suite & s) { good(s); scale(s, 64, 1, 2.0f, 0.0f); s.ops.back().kflags = 0x80; }, BAD, 0},
        {"GLU variant 3", [&](suite & s) { good(s); glu(s, 0, 64, 1); patch(s, gl::VARIANT, 3); }, BAD, 0},
        {"UNARY func 5", [&](suite & s) { good(s); unary(s, 0, 64, 1); patch(s, un::FUNC, 5); }, BAD, 0},
        {"CPY row lines 0", [&](suite & s) { good(s); cpy(s, 4, 1); patch(s, opl::cpy::ROW_LINES, 0); }, BAD, 0},
        {"CPY merged beyond beat[27:0]", [&](suite & s) { good(s); cpy(s, 4, 1); patch(s, opl::cpy::ROWS, 0x08000000u); patch(s, opl::cpy::ROW_LINES, 4); }, BAD, 0},
        {"GET_ROWS ids misaligned", [&](suite & s) { good(s); get_rows(s, 4, 4, {1}); patch(s, opl::getrows::IDS, 2); }, BAD, 0},
        {"GET_ROWS ids beyond the preload area", [&](suite & s) { good(s); get_rows(s, 4, 4, {1, 2}); patch(s, opl::getrows::IDS, 64); }, BAD, 0},
        {"ADD row larger than the ring (SCRATCH)", [&](suite & s) { good(s); ew(s, K_ADD, 16u * 3000u, 1, 1, false); }, SCR, 0},
        {"UNARY row larger than the ring (SCRATCH)", [&](suite & s) { good(s); unary(s, 0, 16u * 3900u, 1); }, SCR, 0},
        {"GET_ROWS negative id (fast)", [&](suite & s) { good(s); get_rows(s, 4, 4, {-1}); }, 2u, TSR_ERR_DRAM_RANGE},
        {"GET_ROWS id beyond the channel", [&](suite & s) { good(s); get_rows(s, 4, 4, {1, 0x03FFFFFF}); }, 2u, TSR_ERR_DRAM_RANGE},
        {"GET_ROWS id beyond 2^32 beats (chunked)", [&](suite & s) { good(s); get_rows(s, 2500, 1, {0x7FFFFFFF}); }, 2u, TSR_ERR_DRAM_RANGE},
    };
    for (const auto & c : cases) run_error(c);
    return test_finish("fw_elem_rv errors");
}

// ---------------------------------------------------------------------------
// perf: RV cycles of work per record (fw_prof.h)
// ---------------------------------------------------------------------------
struct perf_ctx {
    tsr_rv * rv = nullptr;
    uint64_t last = 0;
    fwprof::spin_filter spin;
    fwprof::prof * prof = nullptr;
    uint32_t last_w0 = 0;
    uint64_t work = 0, work_insns = 0, work_at_push = 0, insns_at_push = 0;
};
int perf_retire(void * ctx, const tsr_rv_retire * r) {
    perf_ctx * p = static_cast<perf_ctx *>(ctx);
    const uint64_t now = tsr_rv_cycles(p->rv);
    const uint64_t d = now - p->last;
    p->last = now;
    const bool spin = p->spin.step(r);
    if (d <= 256u) {
        if (!spin) {
            p->work += d;
            p->work_insns++;
        }
        if (p->prof) p->prof->add(r, d, spin);
    }
    if (r->mem_write && r->mem_size == 4u && r->mem_addr == 0x20000000u + TSR_CW_ISSUE_W(0)) p->last_w0 = r->mem_val;
    if (r->mem_write && r->mem_size == 4u && r->mem_addr == 0x20000000u + TSR_CW_ISSUE_PUSH && (p->last_w0 & 0xFFu) != TSR_OP_FENCE) {
        p->work_at_push = p->work;
        p->insns_at_push = p->work_insns;
    }
    return 0;
}

// A chain of n records of one shape (each reads the previous one's output, so deps chain them as in
// a layer), built by mk(s, from) returning the op.
using mk_fn = std::function<int(suite &, int)>;

// Work (cycles, instructions) of the launch up to its last op push.
bool perf_run(const mk_fn & mk, uint32_t n, bool u200, double * cyc, double * ins, fwprof::prof * prof) {
    suite s{};
    s.name = "perf";
    s.cfg = base_cfg(4u << 20);
    s.seed = 0x9E4F;
    int prev = -1;
    for (uint32_t i = 0; i < n; ++i) prev = mk(s, prev);
    build(s);
    tsr_sim_opts so;
    memset(&so, 0, sizeof(so));
    so.sched = "fixed";
    static const uint8_t deep_q[6] = {31, 31, 31, 31, 0, 31};
    if (!u200) so.q_depth = deep_q;
    tsr_sim * sim = tsr_sim_create_opts(&s.cfg, &so);
    if (sim == nullptr) return false;
    fill(sim, s);
    fwh h;
    if (fwh_open(&h, sim, g_image) != 0) {
        tsr_sim_destroy(sim);
        return false;
    }
    perf_ctx p;
    p.rv = tsr_rv_agent_rv(tsr_sim_rv_agent(sim));
    p.last = tsr_rv_cycles(p.rv);
    p.prof = prof;
    tsr_rv_agent_set_retire_hook(tsr_sim_rv_agent(sim), perf_retire, &p);
    const uint32_t a[7] = {s.opl_h, uint32_t(s.blob.size()), 0, 0, 0, 0, 0};
    const bool ok = fwh_launch(&h, uint32_t(fwh_kernel(&h, "graph")), a, 0) == 0 && fwh_status(&h) == TSR_LCQ_R0_STATUS_OK;
    CHECK(ok, "perf: launch (status %u RET 0x%08x)", fwh_status(&h), uint32_t(fwh_ret(&h)));
    tsr_rv_agent_set_retire_hook(tsr_sim_rv_agent(sim), nullptr, nullptr);
    const uint64_t stall = tsr_sim_perf(sim, TSR_PERF_PUSH_STALL(tsr_sim_config(sim)->num_ch));
    CHECK(stall == 0u, "perf: PERF PUSH_STALL %llu: a push waited for a credit in hardware (CC §7.3)", (unsigned long long)stall);
    *cyc = double(p.work_at_push);
    *ins = double(p.insns_at_push);
    fwh_close(&h);
    tsr_sim_destroy(sim);
    return ok;
}

int perf_main(bool u200, bool verbose) {
    {
        const char * c = getenv("TSR_SIM_RV_CORE");
        printf("fw_elem_perf: RISC-V timing model: Ibex %s; %s queues\n", c && c[0] ? c : tsr_rv_core_name(TSR_RV_CORE_DEFAULT),
               u200 ? "U200" : "deep (31)");
    }
    struct pk {
        const char * name;
        mk_fn mk;
    };
    // n = 896 (Qwen2.5-0.5B, as fw_exec_perf's short records): the units keep up with the RISC-V, so
    // the figure is the RISC-V's own work; the model shapes after them move enough data that the DMA
    // queue fills and the batched credit check falls back to issue()'s per-push wait (still work by
    // fw_prof.h's metric: each wait's first test).
    const std::vector<pk> kinds = {
        {"ADD residual (n 896)", [](suite & s, int f) { return ew(s, K_ADD, 896, 1, 1, false, 0, 0, 0, false, f); }},
        {"MUL norm weight (n 896, broadcast)", [](suite & s, int f) { return ew(s, K_MUL, 896, 1, 1, false, 0, 0, 0, false, f); }},
        {"MUL one scalar (n 896)", [](suite & s, int f) { return ew(s, K_MUL, 896, 1, 1, true, 0, 0, 0, false, f); }},
        {"SCALE (n 896)", [](suite & s, int f) { return scale(s, 896, 1, 0.5f, 0.0f, 0, f); }},
        {"SCALE with bias (n 896)", [](suite & s, int f) { return scale(s, 896, 1, 0.5f, 1.0f, 0, f); }},
        {"GLU SWIGLU (n 896)", [](suite & s, int) { return glu(s, gl::SWIGLU, 896, 1); }},
        {"UNARY SILU (n 896)", [](suite & s, int f) { return unary(s, un::SILU, 896, 1, 0, f); }},
        {"CPY (896 f32 = 56 lines)", [](suite & s, int f) { return cpy(s, 56, 1, 0, 0, f); }},
        {"GET_ROWS 1 id (56 lines)", [](suite & s, int) { return get_rows(s, 56, 4, {2}); }},
        {"ADD residual 32B (n 5120)", [](suite & s, int f) { return ew(s, K_ADD, 5120, 1, 1, false, 0, 0, 0, false, f); }},
        {"MUL norm weight 12B (n 3840)", [](suite & s, int f) { return ew(s, K_MUL, 3840, 1, 1, false, 0, 0, 0, false, f); }},
        {"MUL per-head weight 12B (16 x 256)", [](suite & s, int f) { return ew(s, K_MUL, 256, 16, 1, false, 0, 0, 0, false, f); }},
        {"MUL inp_scale_rows 12B (n 3840)", [](suite & s, int f) { return ew(s, K_MUL, 3840, 1, 1, true, 0, 0, 0, false, f); }},
        {"GLU SWIGLU 0.5B (n 4864)", [](suite & s, int) { return glu(s, gl::SWIGLU, 4864, 1); }},
        {"GLU GEGLU16 12B (n 15360)", [](suite & s, int) { return glu(s, gl::GEGLU16, 15360, 1); }},
        {"UNARY TANH 12B (n 3840)", [](suite & s, int f) { return unary(s, un::TANH, 3840, 1, 0, f); }},
    };
    const uint32_t n0 = 8, n1 = 40;
    printf("fw_elem_perf: RV cycles of work per decode record (%u vs %u records; up to the last op push, spin-wait "
           "iterations excluded, fw_prof.h)\n", n1, n0);
    bool all = true;
    for (const auto & k : kinds) {
        double c0 = 0, c1 = 0, i0 = 0, i1 = 0;
        fwprof::prof pr[2];
        if (verbose) {
            pr[0].load_symbols(g_image);
            pr[1].load_symbols(g_image);
        }
        if (!perf_run(k.mk, n0, u200, &c0, &i0, verbose ? &pr[0] : nullptr) ||
            !perf_run(k.mk, n1, u200, &c1, &i1, verbose ? &pr[1] : nullptr)) {
            CHECK(0, "perf: %s: run failed", k.name);
            continue;
        }
        const double per = (c1 - c0) / double(n1 - n0), pin = (i1 - i0) / double(n1 - n0);
        const bool met = per <= 350.0;
        all = all && met;
        printf("  %-44s %6.1f RV cycles (%5.1f instructions) per record: %s\n", k.name, per, pin, met ? "<= 350" : "ABOVE 350");
        if (verbose) {
            char tag[16];
            snprintf(tag, sizeof(tag), ".%zu", size_t(&k - kinds.data()));
            fwprof::prof::diff(pr[1], pr[0]).report(stdout, k.name, double(n1 - n0), 12, tag);
        }
        // Regression ceiling (well above the target: a broken fast path costs > 1,000).
        CHECK(per <= 600.0, "perf: %s: %.0f RV cycles per record above the regression ceiling 600", k.name, per);
    }
    printf("fw_elem_perf: target <= 350 RV cycles per short record (W2b): %s\n", all ? "met by every kind" : "NOT met by every kind");
    return test_finish("fw_elem_rv perf");
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s IMAGE suite|errors|perf [--scheds N] [--only K] [--u200] [--verbose]\n", argv[0]);
        return 2;
    }
    g_image = argv[1];
    if (FILE * f = fopen(g_image, "rb")) fclose(f);
    else {
        printf("%s: no image %s: skipped\n", argv[0], g_image);
        return 77;
    }
    const std::string mode = argv[2];
    uint32_t n_sched = 8;
    int only = -1;
    bool u200 = false, verbose = false;
    for (int i = 3; i < argc; ++i) {
        if (!strcmp(argv[i], "--scheds") && i + 1 < argc) n_sched = uint32_t(atoi(argv[++i]));
        else if (!strcmp(argv[i], "--only") && i + 1 < argc) only = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--u200")) u200 = true;
        else if (!strcmp(argv[i], "--verbose")) verbose = true;
    }
    if (mode == "errors") return errors_main();
    if (mode == "shapes") {
        suite sh = suite_shapes();
        run_scheds(sh, n_sched);
        return test_finish("fw_elem_rv shapes");
    }
    if (mode == "perf") return perf_main(u200, verbose);
    if (mode != "suite") {
        fprintf(stderr, "unknown mode %s\n", mode.c_str());
        return 2;
    }
    suite (*suites[])() = {suite_decode, suite_prefill, suite_small, suite_special};
    for (int k = 0; k < 4; ++k) {
        if (only >= 0 && only != k) continue;
        suite s = suites[k]();
        run_scheds(s, n_sched);
    }
    return test_finish("fw_elem_rv");
}
