// test_fw_elem_host.cpp — fw_kernels_elem_host: the element-wise and row kinds (ADD, MUL, SCALE, GLU,
// UNARY, CPY, GET_ROWS; sw/ctrl/kernels/k_elem.cpp, k_glu.cpp, k_rows.cpp) compiled for the host
// (docs/design/06-software-verification.md "SW" §6.8, SW-25: tsr_fw_host under ASan + UBSan), driven
// through the simulator's test hooks (tsr/backend_host.hpp). The reference stays the RV binary on
// tsr_rv (fw_kernels_elem_rv runs the same sources there).
//
//   1. the planners (tsr/plan_elem.hpp): the refusals of their header comment, and on a grid of
//      shapes (every operand-b form, rows 1 .. 70,000, n 1 .. 1,048,560, the U200 budget and a small
//      one) the tiling invariants: tiles cover the rows (n_tiles = ceil(rows / t)), aligned (period
//      p, 16 rows for per-row scalars), a block within the budget (half of it unless single-buffered
//      one unit), one tile whenever the rows fit half the budget, at most 65,535 rows; RET_SCRATCH
//      exactly when one aligned unit exceeds the budget; copies likewise; plan_op agrees;
//   2. every decode fast path against the general path on twin simulators (as fw_kernels_host's
//      MUL_MAT test): each kind and form at decode shapes, after the same history (records of the
//      kind through the handlers, so that the ring reuses blocks released by STs), with deps on
//      no unit, DMA, SYNC, VU and SYNC + VU, with and without REC: the fast path takes the record,
//      and every push (eight words, both deps, the ISSUE_PUSH value), the ticket, the ring state and
//      the outputs are equal; race detector fatal on both;
//   3. records through the handlers (fast path first, else general) with their own ring and
//      kind_ctx (the executor's contract, tsr/exec.hpp): decode, prefill and small-ring shapes,
//      chains with deps; every dst row bit-exact against sw/ctrl/ref/elem_ref.cpp;
//   4. refused records (bad arguments, kind flags, no VU in caps, GET_ROWS ids outside the preload
//      area, a row beyond the ring): ctx.ret set, nothing pushed.
// Exit 0 / 1.
#undef _GNU_SOURCE
#include "test_sim_util.h"

#include <array>
#include <functional>
#include <vector>

#include "tsr/exec.hpp"
#include "tsr/rt.hpp"

#include "../kernels/k_elem.hpp"
#include "../ref/elem_ref.hpp"
#include "tsr_race.h"
#include "tsr_sim_int.h"

namespace {

namespace opl = tsr::opl;
namespace pl = tsr::plan;
constexpr uint64_t LINE = 64u;
uint32_t H(uint32_t ch, uint64_t beat) { return (ch << 28) | uint32_t(beat & 0x0FFFFFFFu); }
uint32_t L16(uint32_t n) { return (n + 15u) / 16u; }

tsr_config cfg_u200(uint32_t spm_bytes = 4u << 20) {
    tsr_config c = TSR_CONFIG_U200;
    c.t_max = 8;
    c.spm_bytes = spm_bytes;
    c.ch_bytes = 1ull << 28;
    c.caps = TSR_SIM_CAPS_IMPLEMENTED;
    return c;
}
tsr_sim * make_sim(const tsr_config & c, const char * sched = "fixed") {
    tsr_sim_opts so;
    memset(&so, 0, sizeof(so));
    so.sched = sched;
    so.race = 2;
    so.threads = 1;
    return tsr_sim_create_opts(&c, &so);
}

// ---------------------------------------------------------------------------
// 1. Planners
// ---------------------------------------------------------------------------
pl::plan_limits limits(uint32_t budget) {
    pl::plan_limits l = pl::LIMITS_U200;
    l.budget = budget;
    return l;
}

void check_ew_plan(uint32_t kind, const uint32_t * a, uint32_t n_args, const pl::plan_limits & l, uint32_t bm_want,
                   const char * what) {
    tsr::kind_plan kp{};
    const bool ok = pl::plan_kind(l, kind, a, n_args, &kp);
    uint32_t n = 0, rows = 0, p = 1u;
    switch (kind) {
        case opl::K_ADD: case opl::K_MUL: n = a[opl::ew::N]; rows = a[opl::ew::ROWS]; p = a[opl::ew::B_PERIOD]; break;
        case opl::K_SCALE: n = a[opl::scl::N]; rows = a[opl::scl::ROWS]; break;
        case opl::K_GLU: n = a[opl::glu::N]; rows = a[opl::glu::ROWS]; break;
        default: n = a[opl::unary::N]; rows = a[opl::unary::ROWS]; break;
    }
    const uint32_t L = L16(n);
    const uint32_t pe = (bm_want == pl::EB_PERIOD || bm_want == pl::EB_CPERIOD) ? p : 1u;
    const uint32_t align = pe != 1u ? pe : bm_want == pl::EB_CROWS ? 16u : 1u;
    const uint64_t unit = pl::ew_tile_lines(bm_want, L, std::min(align, rows), pe);
    if (!ok) {
        CHECK(kp.status == opl::RET_SCRATCH && unit > l.budget, "plan %s: status 0x%08x (unit %llu lines, budget %u)", what,
              kp.status, (unsigned long long)unit, l.budget);
        CHECK(kp.scratch_lines == pl::SCRATCH_UNSUPPORTED, "plan %s: scratch of a refused plan", what);
        return;
    }
    const uint32_t t = kp.tile_rows;
    CHECK(unit <= l.budget, "plan %s: planned although one unit (%llu lines) exceeds the budget %u", what,
          (unsigned long long)unit, l.budget);
    CHECK(pl::ew_bmode(kp) == bm_want, "plan %s: form %u, expected %u", what, pl::ew_bmode(kp), bm_want);
    CHECK(kp.row_lines == L && kp.aux[0] == pe, "plan %s: row lines %u / period %u", what, kp.row_lines, kp.aux[0]);
    CHECK(t >= 1u && t <= 0xFFFFu && t <= rows, "plan %s: tile rows %u", what, t);
    CHECK(kp.n_tiles == (rows + t - 1u) / t, "plan %s: %u tiles of %u rows for %u rows", what, kp.n_tiles, t, rows);
    CHECK(((kp.flags & pl::KP_ONE_TILE) != 0u) == (kp.n_tiles == 1u), "plan %s: KP_ONE_TILE", what);
    CHECK(kp.scratch_lines == pl::ew_tile_lines(bm_want, L, t, pe) && kp.scratch_lines <= l.budget, "plan %s: scratch %u", what,
          kp.scratch_lines);
    if (kp.n_tiles > 1u) {
        CHECK(t % align == 0u, "plan %s: tile of %u rows not aligned to %u", what, t, align);
        CHECK(kp.scratch_lines <= l.budget / 2u || t == align, "plan %s: a multi-tile op's tile (%u lines) above half the budget",
              what, kp.scratch_lines);
        // The largest such tile: one more aligned unit would not fit half the budget (or the field).
        const uint32_t t2 = t + align;
        CHECK(t2 > 0xFFFFu || pl::ew_tile_lines(bm_want, L, t2, pe) > l.budget / 2u, "plan %s: tile of %u rows not the largest",
              what, t);
    }
    if (rows <= 0xFFFFu && pl::ew_tile_lines(bm_want, L, rows, pe) <= l.budget / 2u)
        CHECK(kp.n_tiles == 1u, "plan %s: rows fit half the budget but %u tiles", what, kp.n_tiles);
    const pl::op_plan op = pl::plan_op(tsr::plan::U200_K4, kind, a, n_args);
    if (l.budget == pl::LIMITS_U200.budget)
        CHECK(op.status == opl::RET_OK && op.scratch_lines == kp.scratch_lines, "plan %s: plan_op disagrees", what);
}

void test_planners() {
    const uint32_t budgets[2] = {pl::LIMITS_U200.budget, 3824u};
    const uint32_t ns[] = {1, 15, 16, 17, 64, 896, 1152, 3840, 5120, 15360, 27648, 262144, 1048560};
    const uint32_t rows_[] = {1, 2, 14, 16, 17, 40, 333, 512, 4096, 70000};
    uint32_t cnt = 0;
    for (uint32_t budget : budgets) {
        const pl::plan_limits l = limits(budget);
        for (uint32_t n : ns)
            for (uint32_t rows : rows_) {
                char what[160];
                for (uint32_t form = 0; form < 7; ++form) {
                    // ADD: rows (p = rows), bcast (p = 1), period (p = 7 / 40), b stride 0; scalar 1, rows, period.
                    uint32_t p = 1, fl = 0, bs = L16(n), bm = 0;
                    switch (form) {
                        case 0: p = rows; bm = rows == 1u ? pl::EB_BCAST : pl::EB_ROWS; break;
                        case 1: p = 1; bm = pl::EB_BCAST; break;
                        case 2: p = rows > 7u ? 7u : rows; bm = p == 1u ? pl::EB_BCAST : p >= rows ? pl::EB_ROWS : pl::EB_PERIOD; break;
                        case 3: p = 40; bs = 0; bm = pl::EB_BCAST; break;
                        case 4: p = 1; fl = 1; bm = pl::EB_C1; break;
                        case 5: p = rows; fl = 1; bm = rows == 1u ? pl::EB_C1 : pl::EB_CROWS; break;
                        default: p = 40; fl = 1; bm = 40u >= rows ? pl::EB_CROWS : pl::EB_CPERIOD; break;
                    }
                    const uint32_t a[10] = {H(1, 0), L16(n), H(0, 0), L16(n), H(2, 0), bs, n, rows, p, fl};
                    snprintf(what, sizeof(what), "ADD n %u rows %u form %u budget %u", n, rows, form, budget);
                    check_ew_plan(opl::K_ADD, a, 10, l, bm, what);
                    ++cnt;
                }
                const uint32_t sa[8] = {H(1, 0), L16(n), H(0, 0), L16(n), n, rows, 0x3F000000u, 0u};
                snprintf(what, sizeof(what), "SCALE n %u rows %u budget %u", n, rows, budget);
                check_ew_plan(opl::K_SCALE, sa, 8, l, pl::EB_NONE, what);
                const uint32_t ga[9] = {H(1, 0), L16(n), H(0, 0), L16(n), H(2, 0), L16(n), n, rows, 1u};
                snprintf(what, sizeof(what), "GLU n %u rows %u budget %u", n, rows, budget);
                check_ew_plan(opl::K_GLU, ga, 9, l, pl::EB_ROWS, what);
                const uint32_t ua[7] = {H(1, 0), L16(n), H(0, 0), L16(n), n, rows, 2u};
                snprintf(what, sizeof(what), "UNARY n %u rows %u budget %u", n, rows, budget);
                check_ew_plan(opl::K_UNARY, ua, 7, l, pl::EB_NONE, what);
                cnt += 3;
            }
        // Copies.
        const uint32_t rls[] = {1, 7, 56, 1000, 1911, 1912, 1913, 3000, 40000};
        for (uint32_t rl : rls)
            for (uint32_t rows : rows_) {
                for (int dense = 0; dense < 2; ++dense) {
                    const uint32_t ss = dense ? rl : rl + 3u, ds = dense ? rl : rl + 1u;
                    const uint32_t ca[6] = {H(1, 0), ds, H(0, 0), ss, rl, rows};
                    tsr::kind_plan kp{};
                    const bool ok = tsr::plan_cpy(ca, 6, l, &kp);
                    const bool merged = rows == 1u || dense;
                    const uint64_t tot = merged ? uint64_t(rows) * rl : rl;
                    if (merged && tot > 0x0FFFFFFFu) {
                        CHECK(!ok && kp.status == opl::RET_BAD_RECORD, "plan CPY rl %u rows %u: beyond beat[27:0]", rl, rows);
                        continue;
                    }
                    CHECK(ok, "plan CPY rl %u rows %u dense %d: 0x%08x", rl, rows, dense, kp.status);
                    if (!ok) continue;
                    const uint32_t half = budget / 2u;
                    const bool chunked = ((kp.flags >> pl::KP_KIND_LSB) & pl::CP_CHUNKED) != 0u;
                    CHECK(kp.row_lines == tot && kp.aux[0] == (merged ? 1u : rows) && chunked == (tot > half),
                          "plan CPY rl %u rows %u: row lines %u rows %u chunked %d", rl, rows, kp.row_lines, kp.aux[0], chunked);
                    CHECK(kp.scratch_lines <= half && kp.scratch_lines >= 1u, "plan CPY rl %u rows %u: scratch %u", rl, rows,
                          kp.scratch_lines);
                    if (chunked)
                        CHECK(kp.aux[1] == half && kp.n_tiles == kp.aux[0] * ((kp.row_lines + half - 1u) / half),
                              "plan CPY rl %u rows %u: chunks", rl, rows);
                    else
                        CHECK(kp.n_tiles == (kp.aux[0] + kp.tile_rows - 1u) / kp.tile_rows && kp.scratch_lines == kp.tile_rows * kp.row_lines &&
                                  (kp.n_tiles == 1u || (kp.tile_rows + 1u) * kp.row_lines > half),
                              "plan CPY rl %u rows %u: tiles", rl, rows);
                    ++cnt;
                }
            }
    }
    // Refusals (BAD_RECORD) of the header comment.
    const pl::plan_limits l = pl::LIMITS_U200;
    auto bad = [&](uint32_t kind, std::vector<uint32_t> a, const char * what, uint32_t want = opl::RET_BAD_RECORD,
                   const pl::plan_limits * lp = nullptr) {
        tsr::kind_plan kp{};
        const bool ok = pl::plan_kind(lp ? *lp : l, kind, a.data(), uint32_t(a.size()), &kp);
        CHECK(!ok && kp.status == want && kp.scratch_lines == pl::SCRATCH_UNSUPPORTED, "plan refusal %s: ok %d status 0x%08x", what,
              ok, kp.status);
    };
    const std::vector<uint32_t> add = {H(1, 0), 56, H(0, 0), 56, H(2, 0), 56, 896, 4, 1, 0};
    auto with = [](std::vector<uint32_t> v, uint32_t i, uint32_t x) { v[i] = x; return v; };
    bad(opl::K_ADD, with(add, opl::ew::N, 0), "ADD n 0");
    bad(opl::K_ADD, with(add, opl::ew::N, pl::EW_N_MAX + 1u), "ADD n beyond 1,048,560");
    bad(opl::K_ADD, with(add, opl::ew::ROWS, 0), "ADD rows 0");
    bad(opl::K_ADD, with(add, opl::ew::B_PERIOD, 0), "ADD period 0");
    bad(opl::K_ADD, with(add, opl::ew::FLAGS, 2), "ADD flag bit 1");
    bad(opl::K_ADD, with(add, opl::ew::DST_STRIDE, 55), "ADD dst rows overlap");
    bad(opl::K_ADD, std::vector<uint32_t>(add.begin(), add.end() - 1), "ADD 9 args");
    bad(opl::K_MUL, with(add, opl::ew::N, 0), "MUL n 0");
    {
        pl::plan_limits nv = l;
        nv.caps = 0;
        bad(opl::K_ADD, add, "ADD without VU caps", opl::RET_BAD_RECORD, &nv);
        tsr::kind_plan kp{};
        const uint32_t ca[6] = {H(1, 0), 4, H(0, 0), 4, 4, 2};
        CHECK(tsr::plan_cpy(ca, 6, nv, &kp), "plan CPY without VU caps (DMA only) refused");
    }
    bad(opl::K_SCALE, {H(1, 0), 56, H(0, 0), 56, 0, 1, 0, 0}, "SCALE n 0");
    bad(opl::K_GLU, {H(1, 0), 56, H(0, 0), 56, H(2, 0), 56, 896, 1, 3}, "GLU variant 3");
    bad(opl::K_UNARY, {H(1, 0), 56, H(0, 0), 56, 896, 1, 6}, "UNARY func 6");
    bad(opl::K_UNARY, {H(1, 0), 56, H(0, 0), 56, 896, 0, 0}, "UNARY rows 0");
    bad(opl::K_CPY, {H(1, 0), 4, H(0, 0), 4, 0, 1}, "CPY rl 0");
    bad(opl::K_CPY, {H(1, 0), 3, H(0, 0), 5, 4, 2}, "CPY dst rows overlap");
    bad(opl::K_GET_ROWS, {H(1, 0), 4, H(0, 0), 4, 2, 1, 4}, "GET_ROWS ids misaligned");
    bad(opl::K_GET_ROWS, {H(1, 0), 4, H(0, 0), 4, 16380, 2, 4}, "GET_ROWS ids beyond 256 lines");
    bad(opl::K_GET_ROWS, {H(1, 0), 4, H(0, 0), 4, 0, 0, 4}, "GET_ROWS 0 ids");
    bad(opl::K_ADD, {H(1, 0), 4000, H(0, 0), 4000, H(2, 0), 4000, 64000, 1, 1, 0}, "ADD 8,000 lines on a 3,824 budget",
        opl::RET_SCRATCH, [] { static pl::plan_limits s = pl::LIMITS_U200; s.budget = 3824; return &s; }());
    printf("planners: %u plans checked\n", cnt);
}

// ---------------------------------------------------------------------------
// Records, data and the reference
// ---------------------------------------------------------------------------
struct rec {
    uint32_t kind;
    std::vector<uint32_t> a;
    uint32_t dst, ds, rows, lines;   // dst rows (for the comparison)
    std::vector<int32_t> ids;        // GET_ROWS (written to the preload area)
};

struct alloc {
    uint64_t io[4] = {0x10000u, 0x10000u, 0x10000u, 0x10000u};
    uint32_t next = 0;
    uint64_t bytes() const { return std::max(std::max(io[0], io[1]), std::max(io[2], io[3])) * LINE; }
    uint32_t take(uint64_t lines) {
        const uint32_t ch = next++ & 3u;
        const uint64_t b = io[ch];
        io[ch] += (lines + 7u) & ~7ull;
        return H(ch, b);
    }
};

void fill_rows(tsr_sim * sim, rng_t * r, uint32_t h, uint32_t rows, uint32_t stride, uint32_t lines) {
    for (uint32_t row = 0; row < rows; ++row) {
        uint8_t * p = tsr_sim_dram(sim, h >> 28) + (uint64_t(h & 0x0FFFFFFFu) + uint64_t(row) * stride) * LINE;
        for (uint32_t j = 0; j < 16u * lines; ++j) {
            const uint32_t v = rng_below(r, 16) == 0 ? rng_u32(r) : f2u(float(rng_normal(r)) * 2.0f);
            memcpy(p + 4u * j, &v, 4);
        }
    }
}

// Builders: inputs allocated and filled on sim (and every simulator of `twins`).
struct maker {
    std::vector<tsr_sim *> sims;
    alloc al;
    rng_t r{0x31337};
    uint32_t in(uint32_t rows, uint32_t stride, uint32_t lines) {
        const uint32_t h = al.take(uint64_t(rows) * stride);
        const rng_t r0 = r;
        for (tsr_sim * s : sims) {
            r = r0;
            fill_rows(s, &r, h, rows, stride, lines);
        }
        return h;
    }
    rec ew(uint32_t kind, uint32_t n, uint32_t rows, uint32_t p, bool scalar, uint32_t pad = 0, bool bs0 = false) {
        const uint32_t L = L16(n), st = L + pad;
        const uint32_t ha = in(rows, st, L);
        uint32_t hb, bs;
        if (scalar) {
            hb = in(1, L16(std::max(p, 1u)), L16(std::max(p, 1u)));
            bs = 0;
        } else {
            const uint32_t br = (p == 1u || bs0) ? 1u : std::min(p, rows);
            hb = in(br, st, L);
            bs = bs0 ? 0u : st;
        }
        const uint32_t hd = al.take(uint64_t(rows) * st);
        return rec{kind, {hd, st, ha, st, hb, bs, n, rows, p, scalar ? 1u : 0u}, hd, st, rows, L, {}};
    }
    rec scale(uint32_t n, uint32_t rows, uint32_t s, uint32_t b, uint32_t pad = 0) {
        const uint32_t L = L16(n), st = L + pad;
        const uint32_t hs = in(rows, st, L), hd = al.take(uint64_t(rows) * st);
        return rec{opl::K_SCALE, {hd, st, hs, st, n, rows, s, b}, hd, st, rows, L, {}};
    }
    rec glu(uint32_t var, uint32_t n, uint32_t rows, uint32_t pad = 0) {
        const uint32_t L = L16(n), st = L + pad;
        const uint32_t hg = in(rows, st, L), hu = in(rows, st, L), hd = al.take(uint64_t(rows) * st);
        return rec{opl::K_GLU, {hd, st, hg, st, hu, st, n, rows, var}, hd, st, rows, L, {}};
    }
    rec unary(uint32_t f, uint32_t n, uint32_t rows, uint32_t pad = 0) {
        const uint32_t L = L16(n), st = L + pad;
        const uint32_t hs = in(rows, st, L), hd = al.take(uint64_t(rows) * st);
        return rec{opl::K_UNARY, {hd, st, hs, st, n, rows, f}, hd, st, rows, L, {}};
    }
    rec cpy(uint32_t rl, uint32_t rows, uint32_t spad = 0, uint32_t dpad = 0) {
        const uint32_t hs = in(rows, rl + spad, rl), hd = al.take(uint64_t(rows) * (rl + dpad));
        return rec{opl::K_CPY, {hd, rl + dpad, hs, rl + spad, rl, rows}, hd, rl + dpad, rows, rl, {}};
    }
    rec get_rows(uint32_t rl, uint32_t src_rows, std::vector<int32_t> ids, uint32_t ids_off, uint32_t spad = 0) {
        const uint32_t hs = in(src_rows, rl + spad, rl), n = uint32_t(ids.size()), hd = al.take(uint64_t(n) * rl);
        return rec{opl::K_GET_ROWS, {hd, rl, hs, rl + spad, ids_off, n, rl}, hd, rl, n, rl, ids};
    }
};

// The executor's contract by hand: ring, ctx with dev_cfg / limits, the preload area.
struct env {
    tsr_sim * sim = nullptr;
    tsr::spm_ring ring;
    tsr::plan::dev_cfg dc{};
    tsr::kind_ctx ctx{};
    tsr::plan::exec_layout lay{};
    void init(tsr_sim * s, uint32_t ring_lines = 0, uint32_t caps = TSR_SIM_CAPS_IMPLEMENTED) {
        sim = s;
        const tsr_config & c = *tsr_sim_config(s);
        const uint32_t spm_lines = c.spm_bytes / 64u;
        tsr::plan::layout(spm_lines, 16u, &lay);
        ring.init(lay.ring_line, ring_lines ? ring_lines : lay.ring_lines);
        dc = tsr::plan::dev_cfg{spm_lines, c.t_max, c.k_max, c.num_ch, 9344u, 16u, caps};
        ctx = tsr::kind_ctx{};
        ctx.ring = &ring;
        ctx.cfg = &dc;
        ctx.mml = tsr::plan::mm_limits_of(dc);
        ctx.pl = tsr::plan::plan_limits_of(dc);
        ctx.one = tsr::spm{lay.one_line};      // the one-tile block (graph.cpp's setup)
        ctx.one_lines = lay.one_lines;
        ctx.pre_lines = 16u;
        ctx.pre = tsr::spm{lay.pre_line};
        ctx.preload = tsr::spm_ptr<uint8_t>(ctx.pre);
    }
    // GET_ROWS ids at byte off of the preload area (the RISC-V's view; raw SPM writes).
    void put_ids(const std::vector<int32_t> & ids, uint32_t off) {
        if (!ids.empty()) memcpy(tsr_sim_spm(sim) + 64ull * lay.pre_line + off, ids.data(), 4u * ids.size());
    }
};

// The reference on a DRAM copy of sim (records applied in order); returns the copy.
struct mirror {
    std::vector<std::vector<uint8_t>> ch;
    std::vector<uint8_t *> chp;
    tsr::ref::dram_view v{};
};
void ref_apply(tsr_sim * sim, const std::vector<rec> & rs, const std::vector<uint8_t> & pre, mirror & m, uint64_t bytes) {
    m.ch.assign(4, std::vector<uint8_t>(bytes));
    m.chp.resize(4);
    for (uint32_t c = 0; c < 4u; ++c) {
        memcpy(m.ch[c].data(), tsr_sim_dram(sim, c), bytes);
        m.chp[c] = m.ch[c].data();
    }
    m.v = tsr::ref::dram_view{m.chp.data(), 4u, bytes};
    for (const auto & r : rs) {
        const int rc = tsr::ref::elem_ref(m.v, r.kind, r.a.data(), uint32_t(r.a.size()), pre.data(), pre.size());
        CHECK(rc == tsr::ref::EL_OK, "elem_ref refused a %u record: %d", r.kind, rc);
    }
}
bool rows_equal(tsr_sim * sim, const mirror & m, const rec & r) {
    for (uint32_t row = 0; row < r.rows; ++row) {
        const uint64_t off = (uint64_t(r.dst & 0x0FFFFFFFu) + uint64_t(row) * r.ds) * LINE;
        if (memcmp(tsr_sim_dram(sim, r.dst >> 28) + off, m.ch[r.dst >> 28].data() + off, size_t(r.lines) * LINE) != 0) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// 2. Fast path vs general path on twin simulators
// ---------------------------------------------------------------------------
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

using rec_fn = std::function<rec(maker &, uint32_t /*ids_off*/)>;

// One twin: history (3 records of the shape through the handlers, in a ring of 2.5 blocks), the
// predecessors (deps), then the record by the fast path (fast) or the general path.
struct twin_out {
    plog log;
    uint32_t ticket = 0, ring_head = 0, ring_blocks = 0, fast = 0;
    bool rows_ok = false;
};
void twin_run(const rec_fn & mk, int deps_form, bool recf, bool fast, twin_out & o) {
    const tsr_config cfg = cfg_u200();
    tsr_sim * sim = make_sim(cfg);
    tsr::host_bind(sim);
    maker mkr;
    mkr.sims = {sim};
    env e;
    e.init(sim);
    std::vector<rec> hist;
    for (int k = 0; k < 4; ++k) hist.push_back(mk(mkr, 64u * uint32_t(k)));
    for (const auto & r : hist) e.put_ids(r.ids, r.a.size() == 7u && r.kind == opl::K_GET_ROWS ? r.a[opl::getrows::IDS] : 0u);
    // The block size of the record (its fast path's one tile).
    tsr::kind_plan kp{};
    tsr::plan::plan_kind(e.ctx.pl, hist[0].kind, hist[0].a.data(), uint32_t(hist[0].a.size()), &kp);
    const uint32_t b = kp.scratch_lines;
    e.ring.init(e.lay.ring_line, 2u * b + b / 2u);
    e.ctx.push_flags = recf ? tsr::PUSH_REC : 0u;
    tsr::ticket prev{};
    for (int k = 0; k < 3; ++k) {
        const rec & r = hist[size_t(k)];
        e.ctx.ret = 0;
        prev = tsr::kind_handler(r.kind)(r.a.data(), uint32_t(r.a.size()), 0u, tsr::deps{prev}, e.ctx);
        CHECK(e.ctx.ret == 0 && !prev.none(), "twin history: ret 0x%08x", e.ctx.ret);
    }
    // Predecessors of the record: tickets of the chosen units.
    tsr::deps d{};
    const tsr::ticket t_sync = deps_form == 2 || deps_form == 4 ? tsr::issue(tsr::NOP(), tsr::deps{prev}) : tsr::ticket{};
    const tsr::ticket t_vu = deps_form == 3 || deps_form == 4 ? tsr::issue(tsr::kf::sop_movi(3u, 0x3F800000u), tsr::deps{prev}) : tsr::ticket{};
    if (deps_form == 1) d = tsr::deps{prev};                    // DMA (the last history ST)
    else if (deps_form == 2) d = tsr::after(t_sync);
    else if (deps_form == 3) d = tsr::after(prev, t_vu);
    else if (deps_form == 4) d = tsr::after(t_vu, t_sync);
    const rec & r = hist[3];
    const int lid = tsr_sim_add_listener(sim, plog_q, &o.log);
    e.ctx.ret = 0;
    if (fast) {
        o.ticket = tsr::ek::host_fast(r.kind, r.a.data(), 0u, d, e.ctx);
        o.fast = o.ticket != tsr::ek::NOT_FAST;
    } else {
        o.ticket = tsr::ek::host_general(r.kind, r.a.data(), uint32_t(r.a.size()), 0u, d, e.ctx).v;
    }
    tsr_sim_remove_listener(sim, lid);
    CHECK(e.ctx.ret == 0, "twin: record ret 0x%08x", e.ctx.ret);
    o.ring_head = e.ring.head();
    o.ring_blocks = e.ring.blocks();
    tsr::join_all();
    CHECK(tsr::err_units() == 0, "twin: a unit in error state");
    std::vector<uint8_t> pre(64u * 16u, 0u);
    memcpy(pre.data(), tsr_sim_spm(sim) + 64ull * e.lay.pre_line, pre.size());
    // The reference for the record alone, on the state after the history (its inputs are its own).
    mirror m;
    ref_apply(sim, {r}, pre, m, mkr.al.bytes());
    o.rows_ok = rows_equal(sim, m, r);
    const tsr_race_report * rr = nullptr;
    CHECK(tsr_sim_race_reports(sim, &rr) == 0, "twin: race reports");
    tsr::host_bind(nullptr);
    tsr_sim_destroy(sim);
}

void test_fast_vs_general() {
    struct tc {
        const char * what;
        rec_fn mk;
        uint32_t pushes;   // pushes of the fast path
    };
    namespace un = opl::unary;
    const std::vector<tc> cases = {
        {"ADD residual 896", [](maker & m, uint32_t) { return m.ew(opl::K_ADD, 896, 1, 1, false); }, 4},
        {"ADD residual 5120", [](maker & m, uint32_t) { return m.ew(opl::K_ADD, 5120, 1, 1, false); }, 4},
        {"ADD rows form 14 x 64 (p = 14)", [](maker & m, uint32_t) { return m.ew(opl::K_ADD, 64, 14, 14, false); }, 4},
        {"MUL broadcast 16 x 256", [](maker & m, uint32_t) { return m.ew(opl::K_MUL, 256, 16, 1, false); }, 4},
        {"MUL b stride 0, p 40", [](maker & m, uint32_t) { return m.ew(opl::K_MUL, 128, 40, 40, false, 0, true); }, 4},
        {"MUL one scalar, one row", [](maker & m, uint32_t) { return m.ew(opl::K_MUL, 3840, 1, 1, true); }, 4},
        {"ADD per-row scalars 14 rows", [](maker & m, uint32_t) { return m.ew(opl::K_ADD, 64, 14, 14, true); }, 4},
        {"MUL per-row scalar, one row, p 512", [](maker & m, uint32_t) { return m.ew(opl::K_MUL, 3840, 1, 512, true); }, 4},
        // SCALE: the history's equal SCALEs leave s12 / s13 in the s-register shadow: no SOP.MOVI.
        {"SCALE", [](maker & m, uint32_t) { return m.scale(3840, 1, 0x3D088889u, 0u); }, 3},
        {"SCALE bias", [](maker & m, uint32_t) { return m.scale(896, 1, 0x3E000000u, 0xBFC00000u); }, 4},
        {"SCALE 8 dense rows", [](maker & m, uint32_t) { return m.scale(64, 8, 0x40000000u, 0u); }, 3},
        {"GLU SWIGLU 4864", [](maker & m, uint32_t) { return m.glu(0, 4864, 1); }, 4},
        {"GLU GEGLU16 15360", [](maker & m, uint32_t) { return m.glu(1, 15360, 1); }, 4},
        {"GLU GEGLU 3 rows", [](maker & m, uint32_t) { return m.glu(2, 100, 3); }, 4},
        {"UNARY TANH", [](maker & m, uint32_t) { return m.unary(un::TANH, 3840, 1); }, 3},
        {"UNARY GELU16", [](maker & m, uint32_t) { return m.unary(un::GELU16, 900, 1); }, 3},
        {"UNARY SILU", [](maker & m, uint32_t) { return m.unary(un::SILU, 17, 1); }, 3},
        {"UNARY EXP", [](maker & m, uint32_t) { return m.unary(un::EXP, 896, 2); }, 3},
        {"UNARY GELU", [](maker & m, uint32_t) { return m.unary(un::GELU, 896, 1); }, 3},
        {"CPY one row", [](maker & m, uint32_t) { return m.cpy(56, 1); }, 2},
        {"CPY 6 dense rows", [](maker & m, uint32_t) { return m.cpy(7, 6); }, 2},
        {"GET_ROWS one id", [](maker & m, uint32_t off) { return m.get_rows(56, 9, {7}, off); }, 2},
        {"GET_ROWS one id, src stride 60", [](maker & m, uint32_t off) { return m.get_rows(56, 9, {2}, off, 4); }, 2},
    };
    uint32_t n = 0;
    for (const auto & c : cases)
        for (int df = 0; df <= 4; ++df)
            for (int recf = 0; recf < 2; ++recf) {
                if (recf && df != 3) continue;
                twin_out f, g;
                twin_run(c.mk, df, recf != 0, true, f);
                twin_run(c.mk, df, recf != 0, false, g);
                CHECK(f.fast, "fast/general %s deps %d: the fast path did not take the record", c.what, df);
                CHECK(f.log.v.size() == c.pushes, "fast/general %s deps %d: %zu fast pushes, expected %u", c.what, df,
                      f.log.v.size(), c.pushes);
                CHECK(f.log.v == g.log.v, "fast/general %s deps %d rec %d: pushes differ (%zu vs %zu)", c.what, df, recf,
                      f.log.v.size(), g.log.v.size());
                CHECK(f.ticket == g.ticket && f.ring_head == g.ring_head && f.ring_blocks == g.ring_blocks,
                      "fast/general %s deps %d: ticket 0x%08x / 0x%08x, ring %u.%u / %u.%u", c.what, df, f.ticket, g.ticket,
                      f.ring_head, f.ring_blocks, g.ring_head, g.ring_blocks);
                CHECK(f.rows_ok && g.rows_ok, "fast/general %s deps %d: rows differ from elem_ref (fast %d general %d)", c.what, df,
                      f.rows_ok, g.rows_ok);
                ++n;
            }
    printf("fast vs general: %u twin runs\n", n);
}

// ---------------------------------------------------------------------------
// 3. Records through the handlers against elem_ref; 4. refusals
// ---------------------------------------------------------------------------
void run_list(const char * name, uint32_t spm_bytes, const std::function<std::vector<rec>(maker &)> & mk, bool chain) {
    const tsr_config cfg = cfg_u200(spm_bytes);
    tsr_sim * sim = make_sim(cfg);
    tsr::host_bind(sim);
    maker m;
    m.sims = {sim};
    env e;
    e.init(sim);
    std::vector<rec> rs = mk(m);
    std::vector<uint8_t> pre(64u * 16u, 0u);
    for (const auto & r : rs)
        if (r.kind == opl::K_GET_ROWS && !r.ids.empty()) memcpy(pre.data() + r.a[opl::getrows::IDS], r.ids.data(), 4u * r.ids.size());
    memcpy(tsr_sim_spm(sim) + 64ull * e.lay.pre_line, pre.data(), pre.size());
    mirror mr;
    ref_apply(sim, rs, pre, mr, m.al.bytes());
    tsr::ticket prev{};
    for (size_t i = 0; i < rs.size(); ++i) {
        e.ctx.ret = 0;
        e.ctx.op_index = uint32_t(i);
        prev = tsr::kind_handler(rs[i].kind)(rs[i].a.data(), uint32_t(rs[i].a.size()), 0u, chain ? tsr::deps{prev} : tsr::deps{},
                                             e.ctx);
        CHECK(e.ctx.ret == 0 && !prev.none(), "%s: record %zu (kind 0x%02x): ret 0x%08x", name, i, rs[i].kind, e.ctx.ret);
    }
    tsr::join_all();
    CHECK(tsr::err_units() == 0, "%s: a unit in error state", name);
    for (size_t i = 0; i < rs.size(); ++i)
        CHECK(rows_equal(sim, mr, rs[i]), "%s: record %zu (kind 0x%02x) differs from elem_ref", name, i, rs[i].kind);
    const tsr_race_report * rr = nullptr;
    CHECK(tsr_sim_race_reports(sim, &rr) == 0, "%s: race reports", name);
    CHECK(tsr::host().no_credit_pushes == 0u, "%s: a push without a credit", name);
    printf("%s: %zu records, ring: %u software joins, %u full allocations\n", name, rs.size(), e.ring.joins(), e.ring.slow_allocs());
    tsr::host_bind(nullptr);
    tsr_sim_destroy(sim);
}

void test_lists() {
    namespace un = opl::unary;
    run_list("handlers decode", 4u << 20, [](maker & m) {
        return std::vector<rec>{m.ew(opl::K_ADD, 896, 1, 1, false), m.ew(opl::K_MUL, 3840, 1, 1, true), m.scale(262144, 1, 0x3D088889u, 0u),
                                m.glu(1, 15360, 1), m.unary(un::GELU16, 3840, 1), m.cpy(320, 1), m.get_rows(240, 5, {4}, 0)};
    }, true);
    run_list("handlers prefill", 4u << 20, [](maker & m) {
        return std::vector<rec>{m.ew(opl::K_ADD, 896, 512, 512, false), m.ew(opl::K_ADD, 64, 14 * 20, 14, false, 3),
                                m.ew(opl::K_MUL, 3840, 512, 512, true), m.ew(opl::K_MUL, 3840, 100, 1, true),
                                m.ew(opl::K_ADD, 128, 200, 40, true), m.scale(896, 300, 0xBF000000u, 0x3F400000u, 2),
                                m.glu(0, 4864, 64), m.unary(un::SILU, 1000, 33, 1), m.cpy(13, 40, 3, 1), m.cpy(56, 512),
                                m.get_rows(20, 64, {63, 62, 1, 2, 3, 3, 0}, 0), m.get_rows(20, 64, {5, 6, 7, 8}, 64, 2)};
    }, false);
    // The s-register shadow's clobber rule (exec.hpp kind_ctx::sreg_valid). SCALE sets
    // s12 = 0.75 (SOP.MOVI); a MUL with one scalar for four rows loads s12 with SOP.LDS (general path,
    // ew_run's sreg_clobber); the next SCALE with s = 0.75 must push its MOVI again, else its MULC reads
    // the scalar of the MUL. elem_ref computes 0.75 x: the comparison catches a missing clobber.
    run_list("handlers s12 shadow clobber", 4u << 20, [](maker & m) {
        return std::vector<rec>{m.scale(896, 1, 0x3F400000u, 0u), m.ew(opl::K_MUL, 896, 4, 1, true), m.scale(896, 1, 0x3F400000u, 0u),
                                m.scale(896, 1, 0x3F400000u, 0x3F800000u), m.ew(opl::K_ADD, 896, 3, 1, true),
                                m.scale(896, 1, 0x3F400000u, 0x3F800000u)};
    }, false);
    run_list("handlers small ring", 512u << 10, [](maker & m) {
        return std::vector<rec>{m.ew(opl::K_ADD, 896, 60, 60, false), m.ew(opl::K_MUL, 896, 60, 7, false), m.ew(opl::K_MUL, 896, 100, 1, true),
                                m.ew(opl::K_ADD, 896, 100, 100, true), m.ew(opl::K_ADD, 896, 50, 9, true), m.ew(opl::K_ADD, 26000, 1, 1, false),
                                m.scale(896, 70, 0x40400000u, 0x3F800000u), m.glu(1, 896, 40), m.cpy(1000, 7), m.cpy(3000, 2, 1, 1),
                                m.get_rows(2500, 4, {3, 0, 2}, 0), m.get_rows(300, 20, {1, 2, 3, 4, 5, 6, 7, 19}, 64)};
    }, true);
}

void test_refusals() {
    const tsr_config cfg = cfg_u200(512u << 10);
    tsr_sim * sim = make_sim(cfg);
    tsr::host_bind(sim);
    maker m;
    m.sims = {sim};
    env e;
    e.init(sim);
    struct bad {
        const char * what;
        rec r;
        uint32_t field, value, kflags, want;
        uint32_t caps;
    };
    const uint32_t BAD = opl::RET_BAD_RECORD, CAPS = TSR_SIM_CAPS_IMPLEMENTED;
    const rec add = m.ew(opl::K_ADD, 896, 1, 1, false), sc = m.scale(896, 1, 0x3F800000u, 0u), gl = m.glu(0, 64, 1);
    const rec un = m.unary(0, 64, 1), cp = m.cpy(4, 1), gr = m.get_rows(4, 4, {1}, 0);
    const std::vector<bad> bads = {
        {"ADD n 0", add, opl::ew::N, 0, 0, BAD, CAPS},
        {"ADD flag bit 1", add, opl::ew::FLAGS, 2, 0, BAD, CAPS},
        {"ADD kind flags", add, opl::ew::FLAGS, 0, 1, BAD, CAPS},
        {"ADD without VU", add, opl::ew::FLAGS, 0, 0, BAD, 0},
        {"ADD period 0", add, opl::ew::B_PERIOD, 0, 0, BAD, CAPS},
        {"ADD beyond the ring", add, opl::ew::N, 16u * 2000u, 0, opl::RET_SCRATCH, CAPS},
        {"SCALE kind flags", sc, opl::scl::B, 0, 0x10, BAD, CAPS},
        {"SCALE without VU", sc, opl::scl::B, 0, 0, BAD, 0},
        {"GLU variant 3", gl, opl::glu::VARIANT, 3, 0, BAD, CAPS},
        {"UNARY func 7", un, opl::unary::FUNC, 7, 0, BAD, CAPS},
        {"UNARY rows 0", un, opl::unary::ROWS, 0, 0, BAD, CAPS},
        {"CPY kind flags", cp, opl::cpy::ROWS, 1, 2, BAD, CAPS},
        {"CPY rl 0", cp, opl::cpy::ROW_LINES, 0, 0, BAD, CAPS},
        {"GET_ROWS ids beyond the list's preload area", gr, opl::getrows::IDS, 16u * 64u, 0, BAD, CAPS},
        {"GET_ROWS ids misaligned", gr, opl::getrows::IDS, 1, 0, BAD, CAPS},
        {"GET_ROWS 0 ids", gr, opl::getrows::N_IDS, 0, 0, BAD, CAPS},
    };
    for (const auto & b : bads) {
        std::vector<uint32_t> a = b.r.a;
        a[b.field] = b.value;
        e.dc.caps = b.caps;
        e.ctx.pl = tsr::plan::plan_limits_of(e.dc);
        const uint32_t p0 = tsr::pidx_now();
        e.ctx.ret = 0;
        const tsr::ticket t = tsr::kind_handler(b.r.kind)(a.data(), uint32_t(a.size()), b.kflags, tsr::deps{}, e.ctx);
        CHECK(e.ctx.ret == b.want && t.none() && tsr::pidx_now() == p0, "refusal %s: ret 0x%08x (expected 0x%08x), %u pushes", b.what,
              e.ctx.ret, b.want, tsr::pidx_now() - p0);
    }
    tsr::join_all();
    tsr::host_bind(nullptr);
    tsr_sim_destroy(sim);
}

}  // namespace

int main() {
    test_planners();
    test_fast_vs_general();
    test_lists();
    test_refusals();
    return test_finish("fw_kernels_elem_host");
}
