// test_fw_moe_host.cpp — fw_kernels_moe_host: the MoE kinds (sw/ctrl/kernels/k_moe.cpp) compiled for the host
// (docs/design/06-software-verification.md "SW" §6.8, SW-25: tsr_fw_host under ASan + UBSan). The records
// themselves run on tsr_rv in fw_kernels_moe_rv; this test holds what the host build can reach directly:
//
//   1. the planners (tsr/plan_moe.hpp): the refusals of their header comment, the tiling of MOE_ROUTE
//      (tiles of 64 tokens, ring block lines) and of MUL_MAT_ID (token block, head block) on a grid of
//      shapes, RET_SCRATCH exactly when the block exceeds the budget, plan_kind / plan_op dispatch,
//      opl::check_args;
//   2. the RISC-V selection (the kernel's integer top-k) against tsr_ref_topk on random words, on words from
//      the special set (NaN of every payload, ±inf, ±0, subnormals), on rows of few distinct values (ties)
//      and on every n 1 .. 70 with every k 1 .. min(n, 16): the expert numbers and the DAZ'd values equal.
//   3. the grouping of MUL_MAT_ID (SW-62, SW-66): the planner's group and head block on a grid of limits against
//      a recomputation; the exact push sequence of a small grouped record (the ops, their deps, their order:
//      the double buffering across tiles, and "the back of the held tile before the allocation" when the ring
//      has no room beside it); random routings through the kernel under ASan + UBSan, whose MVMs must be the
//      tiles of the streaming rule (full tiles per expert, one partial tile per expert at the end, experts
//      beyond 256 alone) and whose STs must cover every (token, slot) row once.
// Exit 0 / 1.
#undef _GNU_SOURCE
#include "test_sim_util.h"

#include <algorithm>
#include <deque>
#include <map>
#include <vector>

#include "tsr/exec.hpp"
#include "tsr/plan_moe.hpp"
#include "tsr/rt.hpp"
#include "tsr_race.h"
#include "tsr_ref_vu.h"
#include "tsr_sim_int.h"

namespace tsr {
void host_moe_select_topk(const uint32_t * p, uint32_t n, uint32_t k, uint32_t * ids, uint32_t * vals);
}

namespace {

namespace opl = tsr::opl;
namespace pl = tsr::plan;
namespace mr = tsr::opl::moer;
namespace mi = tsr::opl::mmid;

void test_route_plans() {
    uint32_t a[mr::NARGS] = {0x10000000u, 8u, 0x20000000u, 8u, 0x30000000u, 128u, 8u, 1u, 0u};
    tsr::kind_plan kp{};
    CHECK(tsr::plan_moe_route(a, mr::NARGS, pl::LIMITS_U200, &kp) && kp.status == opl::RET_OK, "route plans");
    CHECK(kp.scratch_lines == 8u + 1u + 2u && kp.n_tiles == 1u && kp.tile_rows == 1u && kp.row_lines == 8u, "route T=1 block");
    for (uint32_t T : { 2u, 15u, 16u, 17u, 63u, 64u, 65u, 129u, 512u, 4096u }) {
        a[mr::T] = T;
        a[mr::LOGITS_STRIDE] = 8u;
        a[mr::IDS_STRIDE] = 8u;
        CHECK(tsr::plan_moe_route(a, mr::NARGS, pl::LIMITS_U200, &kp), "route T=%u plans", T);
        const uint32_t tr = T < 64u ? T : 64u;
        CHECK(kp.tile_rows == tr && kp.n_tiles == (T + 63u) / 64u, "route T=%u: %u tiles of %u", T, kp.n_tiles, kp.tile_rows);
        CHECK(kp.scratch_lines == tr * 8u + pl::vu_lines(tr) + 2u * tr + pl::vu_lines(tr * 8u), "route T=%u: block %u", T, kp.scratch_lines);
        CHECK(tsr::plan::plan_op(pl::U200_K4, opl::K_MOE_ROUTE, a, mr::NARGS).status == opl::RET_OK, "plan_op route");
    }
    // refusals
    auto refuse = [&](uint32_t word, uint32_t v, uint32_t status, const char * what) {
        uint32_t b[mr::NARGS];
        for (uint32_t i = 0; i < mr::NARGS; ++i) b[i] = a[i];
        b[word] = v;
        tsr::kind_plan k2{};
        const bool ok = tsr::plan_moe_route(b, mr::NARGS, pl::LIMITS_U200, &k2);
        CHECK(!ok && k2.status == status && k2.scratch_lines == pl::SCRATCH_UNSUPPORTED, "route %s: ok %d status 0x%x", what, ok,
              k2.status);
    };
    a[mr::T] = 4u;
    refuse(mr::FLAGS, 1u, opl::RET_BAD_RECORD, "flags");
    refuse(mr::N_USED, 0u, opl::RET_BAD_RECORD, "k = 0");
    refuse(mr::N_USED, 17u, opl::RET_BAD_RECORD, "k > 16");
    refuse(mr::N_USED, 129u, opl::RET_BAD_RECORD, "k > n");
    refuse(mr::N_EXPERT, 513u, opl::RET_BAD_RECORD, "n > 512");
    refuse(mr::T, 0u, opl::RET_BAD_RECORD, "T = 0");
    refuse(mr::LOGITS_STRIDE, 7u, opl::RET_BAD_RECORD, "overlapping logits rows");
    refuse(mr::IDS_STRIDE, 0u, opl::RET_BAD_RECORD, "overlapping ids rows");
    {   // no VU, and a budget that cannot hold one tile
        tsr::kind_plan k2{};
        pl::plan_limits l = pl::LIMITS_U200;
        l.caps = 0;
        CHECK(!tsr::plan_moe_route(a, mr::NARGS, l, &k2) && k2.status == opl::RET_BAD_RECORD, "route without VU");
        l = pl::LIMITS_U200;
        l.budget = 10u;
        CHECK(!tsr::plan_moe_route(a, mr::NARGS, l, &k2) && k2.status == opl::RET_SCRATCH, "route over the budget");
    }
    CHECK(opl::check_args(opl::K_MOE_ROUTE, a, mr::NARGS, 0u) == opl::RET_OK, "check_args route");
    a[mr::FLAGS] = 2u;
    CHECK(opl::check_args(opl::K_MOE_ROUTE, a, mr::NARGS, 0u) == opl::RET_BAD_RECORD, "check_args route flags");
}

void test_mmid_plans() {
    auto rec = [](uint32_t fmt, uint32_t N, uint32_t K, uint32_t T, uint32_t nu, uint32_t slot_stride, bool scale) {
        std::vector<uint32_t> a(mi::NARGS, 0u);
        const uint32_t xl = K / 16u;
        a[mi::DST] = 0x10000000u; a[mi::DST_STRIDE] = (N + 15u) / 16u; a[mi::X] = 0x20000000u; a[mi::X_SLOT_STRIDE] = slot_stride;
        a[mi::X_TOK_STRIDE] = slot_stride == 0u ? xl : nu * slot_stride; a[mi::IDS] = 0x30000000u; a[mi::IDS_STRIDE] = 512u;
        a[mi::W_OFF] = 0u; a[mi::EXP_STRIDE] = 4096u; a[mi::FMT] = fmt; a[mi::N] = N; a[mi::K] = K; a[mi::T] = T;
        a[mi::N_USED] = nu; a[mi::N_EXPERT] = 128u; a[mi::SCALE] = scale ? 0x40000000u : opl::HANDLE_NONE; a[mi::FLAGS] = 0u;
        return a;
    };
    tsr::kind_plan kp{};
    // Gemma 4 26B-A4B: gate-up (Q4_K 1408 x 2816, shared x) and down (Q8_0 2816 x 704, x per slot, scale)
    {
        auto a = rec(pl::FMT_Q4_K, 1408u, 2816u, 1u, 8u, 0u, false);
        CHECK(tsr::plan_mul_mat_id(a.data(), mi::NARGS, pl::LIMITS_U200, &kp), "gate-up plans");
        const uint32_t xl = 176u, ql = pl::qa_lines(pl::QA_A256, 2816u), yl = 88u;
        CHECK(kp.aux[0] == xl + ql + 8u * yl && kp.aux[1] == pl::ID_CHUNK_LINES && kp.aux[2] == 1u && kp.n_tiles == 1u,
              "gate-up block %u head %u xs %u", kp.aux[0], kp.aux[1], kp.aux[2]);
        CHECK(kp.scratch_lines == kp.aux[0] + kp.aux[1], "gate-up scratch");
    }
    {   // T = 512 on LIMITS_U200 (T_MAX 8): tiles of 8 rows of one expert; the head block adds the pending table
        auto a = rec(pl::FMT_Q8_0, 2816u, 704u, 512u, 8u, 44u, true);
        CHECK(tsr::plan_mul_mat_id(a.data(), mi::NARGS, pl::LIMITS_U200, &kp), "down plans");
        const uint32_t xl = 44u, ql = pl::qa_lines(pl::QA_A32, 704u), yl = 176u;
        CHECK(kp.aux[3] == 8u && kp.tile_rows == 8u && kp.aux[0] == 8u * (xl + ql + yl) &&
                  kp.aux[1] == pl::ID_CHUNK_LINES + 8u + pl::vu_lines(128u * 8u) && kp.aux[2] == 8u && kp.n_tiles == 512u,
              "down block %u head %u xs %u group %u tiles %u", kp.aux[0], kp.aux[1], kp.aux[2], kp.aux[3], kp.n_tiles);
        CHECK(kp.scratch_lines == kp.aux[0] + kp.aux[1], "down scratch");
        CHECK(tsr::plan::plan_op(pl::U200_K4, opl::K_MUL_MAT_ID, a.data(), mi::NARGS).status == opl::RET_OK, "plan_op mmid");
        a[mi::T] = 1u;   // a decode token keeps the per-token geometry
        CHECK(tsr::plan_mul_mat_id(a.data(), mi::NARGS, pl::LIMITS_U200, &kp) && kp.aux[3] == 0u && kp.tile_rows == 1u &&
                  kp.n_tiles == 1u && kp.aux[0] == 8u * (xl + ql) + 8u * yl && kp.aux[1] == pl::ID_CHUNK_LINES + 8u,
              "down T = 1: per-token geometry");
    }
    // refusals
    auto bad = [&](std::vector<uint32_t> a, uint32_t word, uint32_t v, uint32_t status, const char * what) {
        a[word] = v;
        tsr::kind_plan k2{};
        const bool ok = tsr::plan_mul_mat_id(a.data(), mi::NARGS, pl::LIMITS_U200, &k2);
        CHECK(!ok && k2.status == status, "mmid %s: ok %d status 0x%x", what, ok, k2.status);
    };
    const auto base = rec(pl::FMT_Q8_0, 64u, 64u, 2u, 2u, 4u, true);
    bad(base, mi::FLAGS, 1u, opl::RET_BAD_RECORD, "flags");
    bad(base, mi::FMT, 2u, opl::RET_BAD_RECORD, "format");
    bad(base, mi::K, 48u, opl::RET_BAD_RECORD, "K % 32 != 0");
    bad(base, mi::N_USED, 0u, opl::RET_BAD_RECORD, "k = 0");
    bad(base, mi::N_USED, 17u, opl::RET_BAD_RECORD, "k > 16");
    bad(base, mi::N_EXPERT, 0u, opl::RET_BAD_RECORD, "no experts");
    bad(base, mi::N_EXPERT, 4097u, opl::RET_BAD_RECORD, "too many experts");
    bad(base, mi::T, 0u, opl::RET_BAD_RECORD, "T = 0");
    bad(base, mi::EXP_STRIDE, 0u, opl::RET_BAD_RECORD, "no expert stride");
    bad(base, mi::IDS_STRIDE, 6u, opl::RET_BAD_RECORD, "ids stride");
    bad(base, mi::IDS_STRIDE, 4u, opl::RET_BAD_RECORD, "ids rows overlap");
    bad(base, mi::X_SLOT_STRIDE, 3u, opl::RET_BAD_RECORD, "slot stride below x_lines");
    bad(base, mi::DST_STRIDE, 1u, opl::RET_BAD_RECORD, "dst rows overlap");
    {
        tsr::kind_plan k2{};
        pl::plan_limits l = pl::LIMITS_U200;
        auto a = base;
        l.budget = 100u;
        CHECK(!tsr::plan_mul_mat_id(a.data(), mi::NARGS, l, &k2) && k2.status == opl::RET_SCRATCH, "mmid over the budget");
        l = pl::LIMITS_U200;
        l.caps = 0;
        CHECK(!tsr::plan_mul_mat_id(a.data(), mi::NARGS, l, &k2) && k2.status == opl::RET_BAD_RECORD, "mmid scale without VU");
        a[mi::SCALE] = opl::HANDLE_NONE;
        CHECK(tsr::plan_mul_mat_id(a.data(), mi::NARGS, l, &k2), "mmid without scale needs no VU");
    }
    CHECK(opl::check_args(opl::K_MUL_MAT_ID, base.data(), mi::NARGS, 0u) == opl::RET_OK, "check_args mmid");
}


// ---------------------------------------------------------------------------
// 3a. The planner's grouping against a recomputation
// ---------------------------------------------------------------------------
void test_mmid_group_plans() {
    struct shape { uint32_t fmt, N, K; };
    const shape shapes[] = { { pl::FMT_Q8_0, 64u, 64u }, { pl::FMT_Q4_K, 256u, 512u }, { pl::FMT_Q8_0, 2816u, 704u },
                             { pl::FMT_Q4_K, 1408u, 2816u }, { pl::FMT_Q8_0, 4096u, 8192u } };
    const uint32_t spms[] = { 65536u, 8192u, 4400u };
    long n_group = 0, n_token = 0;
    for (const shape & sh : shapes)
        for (const uint32_t spm_lines : spms)
            for (const uint32_t t_max : { 1u, 2u, 3u, 4u, 8u })
                for (const uint32_t ne : { 1u, 8u, 128u, 256u, 257u, 4096u })
                    for (const bool scale : { false, true })
                        for (const uint32_t T : { 1u, 2u, 3u, 5u, 9u, 512u, (1u << 28), (1u << 28) + 1u }) {
                            const pl::dev_cfg dc = {spm_lines, t_max, 32768u, 4u, 9344u, 16u, pl::CAPS_VU | pl::CAPS_ATT};
                            const pl::plan_limits l = pl::plan_limits_of(dc);
                            if (l.budget == 0u) continue;
                            const uint32_t nu = 8u, xl = sh.K / 16u, yl = (sh.N + 15u) / 16u;
                            std::vector<uint32_t> a(mi::NARGS, 0u);
                            a[mi::DST] = 0x10000000u; a[mi::DST_STRIDE] = yl; a[mi::X] = 0x20000000u; a[mi::X_SLOT_STRIDE] = 0u;
                            a[mi::X_TOK_STRIDE] = xl; a[mi::IDS] = 0x30000000u; a[mi::IDS_STRIDE] = 32u; a[mi::EXP_STRIDE] = 4096u;
                            a[mi::FMT] = sh.fmt; a[mi::N] = sh.N; a[mi::K] = sh.K; a[mi::T] = T; a[mi::N_USED] = nu;
                            a[mi::N_EXPERT] = ne; a[mi::SCALE] = scale ? 0x40000000u : opl::HANDLE_NONE;
                            tsr::kind_plan kp{};
                            const bool ok = tsr::plan_mul_mat_id(a.data(), mi::NARGS, l, &kp);
                            pl::mm_tiling tl{};
                            const uint32_t pr = pl::mm_plan(pl::mm_limits{l.k_max, l.t_max, l.budget}, sh.fmt, sh.N, sh.K, 1u, &tl);
                            if (pr != opl::RET_OK) { CHECK(!ok, "grid: refused shape plans"); continue; }
                            // the recomputation: a tile of g rows (as many as t_max allows with two tiles in the budget)
                            uint32_t g = (l.budget >> 1) / tl.lpt;
                            if (g == 0u) g = 1u;
                            if (g > t_max) g = t_max;
                            if (g > T) g = T;
                            const uint32_t ng = ne < 256u ? ne : 256u;
                            uint32_t pw = 1u;
                            while (pw < g - 1u) pw *= 2u;   // g = 1 never reaches the grouped case
                            const uint32_t table = scale ? (ne + 15u) / 16u : 0u;
                            const uint32_t head = 128u + table + (ng * pw + 15u) / 16u;
                            const bool grouped = T > 1u && g >= 2u && T <= (1u << 28) && head <= pl::one_block_lines(spm_lines);
                            const bool fits = (grouped ? uint64_t(g) * tl.lpt + head : uint64_t(1u) * (tl.x_lines + tl.qa_lines) + nu * yl + 128u + table) <= l.budget;
                            if (!fits) { CHECK(!ok && kp.status == opl::RET_SCRATCH, "grid: over the budget"); continue; }
                            CHECK(ok, "grid: shape %u/%u/%u spm %u t_max %u ne %u T %u plans", sh.fmt, sh.N, sh.K, spm_lines, t_max, ne, T);
                            if (!ok) continue;
                            CHECK(kp.aux[3] == (grouped ? g : 0u) && kp.tile_rows == (grouped ? g : 1u),
                                  "grid: shape %u/%u/%u spm %u t_max %u ne %u scale %d T %u: group %u (expected %u)", sh.fmt, sh.N,
                                  sh.K, spm_lines, t_max, ne, scale, T, kp.aux[3], grouped ? g : 0u);
                            if (grouped) {
                                const uint64_t pairs = uint64_t(T) * nu;
                                CHECK(kp.aux[0] == g * tl.lpt && kp.aux[1] == head && kp.n_tiles == (pairs + g - 1u) / g &&
                                          kp.scratch_lines == kp.aux[0] + kp.aux[1] && kp.row_lines == yl,
                                      "grid: grouped block %u head %u (expected %u) tiles %u (expected %llu)", kp.aux[0], kp.aux[1], head,
                                      kp.n_tiles, (unsigned long long)((pairs + g - 1u) / g));
                                CHECK((kp.flags & pl::KP_ONE_TILE) == 0u, "grid: a grouped record has KP_ONE_TILE clear (tiles %u)", kp.n_tiles);
                                ++n_group;
                            } else {
                                CHECK(kp.aux[1] == 128u + table && kp.n_tiles == T && kp.aux[2] == 1u, "grid: per-token head %u tiles %u",
                                      kp.aux[1], kp.n_tiles);
                                ++n_token;
                            }
                        }
    printf("  grouping plans: %ld grouped and %ld per-token records against the recomputation\n", n_group, n_token);
}


// ---------------------------------------------------------------------------
// 3b. The pushes of a grouped record
// ---------------------------------------------------------------------------
constexpr uint64_t LINE = 64u;
uint32_t HND(uint32_t ch, uint64_t beat) { return (ch << 28) | uint32_t(beat & 0x0FFFFFFFu); }

struct pev {
    uint32_t unit, ticket, w[8], dep[2];
};
void cap_pushes(void * c, tsr_sim *, const tsr_q_event * ev) {
    if (ev->kind != TSR_QEV_PUSH) return;
    pev e{};
    e.unit = ev->unit;
    e.ticket = ev->ticket;
    for (int k = 0; k < 8; ++k) e.w[k] = ev->w[k];
    e.dep[0] = ev->dep[0];
    e.dep[1] = ev->dep[1];
    static_cast<std::vector<pev> *>(c)->push_back(e);
}

struct mmid_case {
    uint32_t fmt, N, K, T, nu, ne, t_max;
    bool shared, scale;
    std::vector<uint32_t> ids;          // T rows of nu expert numbers
    uint32_t ids_stride = 0;            // bytes, 0: 4 nu
    uint32_t ring_lines = 0;            // 0: the executor's ring
};
// The record's addresses (the test's DRAM layout): channel, first beat.
constexpr uint32_t W_OFF = 0x100000u, X_BEAT = 0x4000u, DST_BEAT = 0x4000u, IDS_BEAT = 0x4000u, SCALE_BEAT = 0x4000u;

struct mmid_out {
    std::vector<pev> pushes;
    uint32_t a[mi::NARGS];
    uint32_t ret = 0, err = 0, one_line = 0, x_lines = 0, y_lines = 0, exp_stride = 0, xslot = 0, xtok = 0;
    uint32_t group = 0, qa_lines = 0, ret_ticket = 0;
};

// Runs kind_mul_mat_id on a fresh simulator (host backend): zero weights and activations (the pushes do not depend on
// them), the scale table 1.5, 2.5, ... (distinct bits), the ids of the case.
void run_mmid(const mmid_case & c, mmid_out * o) {
    tsr_config cfg = TSR_CONFIG_U200;
    cfg.t_max = c.t_max;
    cfg.ch_bytes = 1ull << 30;
    cfg.caps = TSR_SIM_CAPS_IMPLEMENTED;
    tsr_sim_opts so;
    memset(&so, 0, sizeof(so));
    so.sched = "fixed";
    so.race = 2;
    so.threads = 2;
    tsr_sim * sim = tsr_sim_create_opts(&cfg, &so);
    CHECK(sim != nullptr, "mmid: simulator");
    if (!sim) return;
    tsr::host_bind(sim);
    const uint32_t spm_lines = cfg.spm_bytes / 64u;
    tsr::host_spm_lines = spm_lines;
    tsr::plan::exec_layout lay{};
    CHECK(tsr::plan::layout(spm_lines, 0, &lay), "mmid: layout");
    const pl::dev_cfg dc = {spm_lines, cfg.t_max, cfg.k_max, cfg.num_ch, 9344u, 16u, pl::CAPS_VU | pl::CAPS_ATT};
    tsr::kind_ctx ctx{};
    static tsr::spm_ring ring;
    ring.init(lay.ring_line, c.ring_lines != 0u ? c.ring_lines : lay.ring_lines);
    ctx.ring = &ring;
    ctx.cfg = &dc;
    ctx.mml = pl::mm_limits_of(dc);
    ctx.pl = pl::plan_limits_of(dc);
    ctx.one = tsr::spm{lay.one_line};
    ctx.one_lines = lay.one_lines;
    const uint32_t xl = c.K / 16u, yl = (c.N + 15u) / 16u;
    const uint32_t xslot = c.shared ? 0u : xl, xtok = c.shared ? xl : c.nu * xl;
    const uint32_t ib = c.ids_stride != 0u ? c.ids_stride : 4u * c.nu;
    const uint32_t exp_stride = uint32_t(tsr_tensor_ch_beats(c.fmt, c.N, c.K, cfg.num_ch));
    uint32_t * a = o->a;
    memset(a, 0, sizeof(o->a));
    a[mi::DST] = HND(1, DST_BEAT);
    a[mi::DST_STRIDE] = yl;
    a[mi::X] = HND(0, X_BEAT);
    a[mi::X_SLOT_STRIDE] = xslot;
    a[mi::X_TOK_STRIDE] = xtok;
    a[mi::IDS] = HND(2, IDS_BEAT);
    a[mi::IDS_STRIDE] = ib;
    a[mi::W_OFF] = W_OFF;
    a[mi::EXP_STRIDE] = exp_stride;
    a[mi::FMT] = c.fmt;
    a[mi::N] = c.N;
    a[mi::K] = c.K;
    a[mi::T] = c.T;
    a[mi::N_USED] = c.nu;
    a[mi::N_EXPERT] = c.ne;
    a[mi::SCALE] = c.scale ? HND(3, SCALE_BEAT) : opl::HANDLE_NONE;
    a[mi::FLAGS] = 0u;
    std::vector<uint8_t> ib_bytes(size_t(c.T) * ib + 64u, 0u);
    for (uint32_t t = 0; t < c.T; ++t)
        for (uint32_t j = 0; j < c.nu; ++j) memcpy(&ib_bytes[size_t(t) * ib + 4u * j], &c.ids[size_t(t) * c.nu + j], 4);
    tsr_sim_dram_write(sim, 2, uint64_t(IDS_BEAT) * LINE, ib_bytes.data(), ib_bytes.size());
    if (c.scale) {
        std::vector<float> sc(c.ne + 16u);
        for (uint32_t i = 0; i < sc.size(); ++i) sc[i] = 1.5f + float(i);
        tsr_sim_dram_write(sim, 3, uint64_t(SCALE_BEAT) * LINE, sc.data(), sc.size() * 4u);
    }
    tsr::kind_plan kp{};
    CHECK(tsr::plan_mul_mat_id(a, mi::NARGS, ctx.pl, &kp), "mmid: plans (status 0x%x)", kp.status);
    o->group = kp.aux[3];
    pl::mm_tiling tl1{};
    CHECK(pl::mm_plan(ctx.mml, c.fmt, c.N, c.K, 1u, &tl1) == 0u, "mmid: tiling");
    o->qa_lines = tl1.qa_lines;
    o->one_line = lay.one_line;
    o->x_lines = xl;
    o->y_lines = yl;
    o->exp_stride = exp_stride;
    o->xslot = xslot;
    o->xtok = xtok;
    const int lid = tsr_sim_add_listener(sim, cap_pushes, &o->pushes);
    const tsr::ticket t = tsr::kind_mul_mat_id(a, mi::NARGS, 0u, tsr::deps{}, ctx);
    o->ret = ctx.ret;
    o->ret_ticket = t.v;
    tsr_sim_remove_listener(sim, lid);   // before join_all, which pushes a FENCE
    tsr::join_all();
    o->err = tsr::err_units();
    CHECK(!t.none() || ctx.ret != 0u, "mmid: no ticket");
    // the record returns the ticket of the last ST pushed (the executor joins it in F_SERIAL mode and a K_JOIN record depends on it)
    if (ctx.ret == 0u)
        CHECK(!o->pushes.empty() && (o->pushes.back().w[0] & 0xFFu) == tsr::OP_ST && t.v == o->pushes.back().ticket,
              "mmid: the returned ticket 0x%08x is not the last ST's (0x%08x)", t.v, o->pushes.empty() ? 0u : o->pushes.back().ticket);
    const tsr_race_report * rr = nullptr;
    CHECK(tsr_sim_race_reports(sim, &rr) == 0u, "mmid: race reports");
    tsr::host_bind(nullptr);
    tsr_sim_destroy(sim);
}

// What the streaming rule (k_moe.cpp, mmid_grouped) pushes, in tile order.
struct mtile {
    uint32_t e;
    std::vector<uint32_t> pairs;        // (token << 4) | slot
};
std::vector<mtile> model_tiles(const mmid_case & c, uint32_t group) {
    const uint32_t ng = c.ne < 256u ? c.ne : 256u;
    std::vector<mtile> out;
    std::vector<std::vector<uint32_t>> pend(ng);
    for (uint32_t t = 0; t < c.T; ++t)
        for (uint32_t j = 0; j < c.nu; ++j) {
            const uint32_t e = c.ids[size_t(t) * c.nu + j], pair = (t << 4) | j;
            if (e >= ng) {
                out.push_back({e, {pair}});
            } else if (pend[e].size() + 1u < group) {
                pend[e].push_back(pair);
            } else {
                pend[e].push_back(pair);
                out.push_back({e, pend[e]});
                pend[e].clear();
            }
        }
    for (uint32_t e = 0; e < ng; ++e)
        if (!pend[e].empty()) out.push_back({e, pend[e]});
    return out;
}

const char * opname(uint32_t w0) {
    switch (w0 & 0xFFu) {
    case tsr::OP_LD: return "LD";
    case tsr::OP_ST: return "ST";
    case tsr::OP_QACT: return "QACT";
    case tsr::OP_MVM: return "MVM";
    case tsr::OP_VOP: return "VOP";
    case tsr::OP_SOP: return "SOP";
    default: return "?";
    }
}

// One expected push: the op, the index of the pushes its deps name (-1: none), and one word that tells which data it
// moves (LD: the DRAM beat; ST: the DRAM beat; QACT / MVM: the rows, MVM also the weights; VOP: the rows; SOP: the value).
struct xpush {
    const char * op;
    int d0, d1;
    uint32_t a, b;
};

void check_spm(const char * what, const mmid_out & o, bool scale);

// Compares the captured pushes of a record with the expected ones: head pushes (the scale table, the ids chunk) are
// listed by the caller in `exp`, too.
void check_sequence(const char * what, const mmid_out & o, const std::vector<xpush> & exp, bool scale) {
    check_spm(what, o, scale);
    std::map<uint32_t, int> idx;
    for (size_t i = 0; i < o.pushes.size(); ++i) idx[o.pushes[i].ticket] = int(i);
    CHECK(o.pushes.size() == exp.size(), "%s: %zu pushes, expected %zu", what, o.pushes.size(), exp.size());
    for (size_t i = 0; i < o.pushes.size() && i < exp.size(); ++i) {
        const pev & p = o.pushes[i];
        const xpush & x = exp[i];
        auto dep = [&](uint32_t t) {
            if ((t >> 28) == 0xFu) return -1;   // TSR_TICKET_NONE: unit 15
            const auto it = idx.find(t);
            return it == idx.end() ? -2 : it->second;
        };
        int d0 = dep(p.dep[0]), d1 = dep(p.dep[1]);
        if (d0 < d1) std::swap(d0, d1);   // the two deps in either slot
        int e0 = x.d0, e1 = x.d1;
        if (e0 < e1) std::swap(e0, e1);
        uint32_t a = 0, b = 0;
        switch (p.w[0] & 0xFFu) {
        case tsr::OP_LD: case tsr::OP_ST: a = p.w[2]; break;
        case tsr::OP_QACT: a = p.w[4]; break;
        case tsr::OP_MVM: a = (p.w[5] >> 16) & 0xFu; b = p.w[3]; break;
        case tsr::OP_VOP: a = p.w[6] & 0xFFFFu; break;
        case tsr::OP_SOP: a = p.w[2]; break;
        default: break;
        }
        CHECK(strcmp(opname(p.w[0]), x.op) == 0 && d0 == e0 && d1 == e1 && a == x.a && b == x.b,
              "%s: push %zu: %s deps (%d, %d) word %u/%u, expected %s deps (%d, %d) word %u/%u", what, i, opname(p.w[0]), d0, d1, a, b,
              x.op, e0, e1, x.a, x.b);
    }
}

// The SPM lines of the pushes of a grouped record: the ids chunk at the start of the one-tile block, the scale table after
// it, and every tile block [x_0 .. x_(m-1) | QA_0 .. QA_(m-1) | y_0 .. y_(m-1)] (the block starts at the tile's first LD).
void check_spm(const char * what, const mmid_out & o, bool scale) {
    struct tb { uint32_t blk, m; };
    std::deque<tb> held;             // tiles whose MVM is pushed and whose back is not
    tb cur{0, 0};                    // the tile whose front is being pushed
    uint32_t st_k = 0;
    for (size_t i = 0; i < o.pushes.size(); ++i) {
        const pev & p = o.pushes[i];
        const uint32_t ch = (p.w[0] >> tsr::LD_CH_LSB) & 0xFu;
        switch (p.w[0] & 0xFFu) {
        case tsr::OP_LD:
            if (ch == 2u) {
                CHECK(p.w[1] == o.one_line, "%s: push %zu: the ids chunk at line %u, expected %u", what, i, p.w[1], o.one_line);
            } else if (ch == 3u) {
                CHECK(p.w[1] == o.one_line + pl::ID_CHUNK_LINES, "%s: push %zu: the scale table at line %u", what, i, p.w[1]);
            } else {
                if (cur.m == 0u) cur.blk = p.w[1];
                CHECK(p.w[1] == cur.blk + cur.m * o.x_lines, "%s: push %zu: x row %u at line %u, expected %u", what, i, cur.m, p.w[1],
                      cur.blk + cur.m * o.x_lines);
                ++cur.m;
            }
            break;
        case tsr::OP_QACT:
            CHECK(p.w[4] == cur.m && p.w[2] == cur.blk && p.w[1] == cur.blk + cur.m * o.x_lines && p.w[5] == o.x_lines,
                  "%s: push %zu: QACT of %u rows from line %u to %u stride %u (block %u, %u rows)", what, i, p.w[4], p.w[2], p.w[1], p.w[5],
                  cur.blk, cur.m);
            break;
        case tsr::OP_MVM:
            CHECK(((p.w[5] >> 16) & 0xFu) == cur.m && p.w[2] == cur.blk + cur.m * o.x_lines &&
                      p.w[1] == cur.blk + cur.m * (o.x_lines + o.qa_lines) && p.w[6] == o.y_lines,
                  "%s: push %zu: MVM of %u rows, x at line %u, y at line %u stride %u (block %u, %u rows)", what, i, (p.w[5] >> 16) & 0xFu,
                  p.w[2], p.w[1], p.w[6], cur.blk, cur.m);
            held.push_back(cur);
            cur = tb{0, 0};
            break;
        case tsr::OP_VOP: {
            CHECK(!held.empty(), "%s: push %zu: a VOP with no tile held", what, i);
            if (held.empty()) break;
            const tb & h = held.front();
            const uint32_t Y = h.blk + h.m * (o.x_lines + o.qa_lines);
            CHECK(scale && p.w[1] == Y && p.w[2] == Y && (p.w[6] & 0xFFFFu) == h.m && (p.w[6] >> 16) == o.y_lines && (p.w[7] & 0xFFFFu) == o.y_lines,
                  "%s: push %zu: VOP at lines %u / %u (expected %u) rows %u", what, i, p.w[1], p.w[2], Y, p.w[6] & 0xFFFFu);
            break;
        }
        case tsr::OP_ST: {
            CHECK(!held.empty(), "%s: push %zu: an ST with no tile held", what, i);
            if (held.empty()) break;
            const tb & h = held.front();
            const uint32_t Y = h.blk + h.m * (o.x_lines + o.qa_lines);
            CHECK(p.w[1] == Y + st_k * o.y_lines, "%s: push %zu: ST of row %u from line %u, expected %u", what, i, st_k, p.w[1],
                  Y + st_k * o.y_lines);
            if (++st_k == h.m) {
                held.pop_front();
                st_k = 0;
            }
            break;
        }
        default: break;
        }
    }
    CHECK(held.empty() && st_k == 0u && cur.m == 0u, "%s: a tile is left without its back (%zu held)", what, held.size());
}

// The expected pushes of a case with a scale table, written as the double-buffered tile loop (the tiles in emission
// order; `order` lists the events: 'F' i = the front of tile i, 'B' i = the back of tile i).
struct seq_builder {
    const mmid_case & c;
    const mmid_out & o;
    std::vector<mtile> tiles;
    std::vector<xpush> v;
    std::vector<int> mv;                      // index of tile i's MVM
    std::vector<uint32_t> last_scale;
    uint32_t sreg = 0;                        // s14 as the pushes leave it (0: unknown)
    seq_builder(const mmid_case & cc, const mmid_out & oo, std::vector<mtile> t) : c(cc), o(oo), tiles(std::move(t)), mv(tiles.size(), -1) {}
    uint32_t scale_bits(uint32_t e) const {
        const float f = 1.5f + float(e);
        uint32_t u;
        memcpy(&u, &f, 4);
        return u;
    }
    void head() {
        if (c.scale) v.push_back({"LD", -1, -1, SCALE_BEAT, 0});
        v.push_back({"LD", c.scale ? -1 : -1, -1, IDS_BEAT, 0});
    }
    void front(size_t i) {
        const mtile & t = tiles[i];
        for (uint32_t pr : t.pairs) {
            const uint32_t tok = pr >> 4, slot = pr & 15u;
            v.push_back({"LD", -1, -1, X_BEAT + tok * o.xtok + slot * o.xslot, 0});
        }
        const int ld = int(v.size()) - 1;
        v.push_back({"QACT", ld, -1, uint32_t(t.pairs.size()), 0});
        v.push_back({"MVM", int(v.size()) - 1, -1, uint32_t(t.pairs.size()), W_OFF + t.e * o.exp_stride});
        mv[i] = int(v.size()) - 1;
    }
    void back(size_t i) {
        const mtile & t = tiles[i];
        int dep = mv[i];
        if (c.scale) {
            const uint32_t sb = scale_bits(t.e);
            if (sreg != sb) {
                v.push_back({"SOP", -1, -1, sb, 0});
                sreg = sb;
            }
            v.push_back({"VOP", mv[i], -1, uint32_t(t.pairs.size()), 0});
            dep = int(v.size()) - 1;
        }
        for (size_t k = 0; k < t.pairs.size(); ++k) {
            const uint32_t tok = t.pairs[k] >> 4, slot = t.pairs[k] & 15u;
            v.push_back({"ST", k == 0 ? dep : -1, -1, DST_BEAT + (tok * c.nu + slot) * o.y_lines, 0});
        }
    }
};

void test_mmid_sequence() {
    // T = 2, k = 2, 4 experts, tiles of 2 rows (T_MAX 2). Pairs in order: (0,0) e0, (0,1) e1, (1,0) e0, (1,1) e2.
    // e0 fills a tile at (1,0); the experts 1 and 2 hold one pair each until the flush.
    mmid_case c{TSR_FMT_Q8_0, 32u, 64u, 2u, 2u, 4u, 2u, false, true, {0u, 1u, 0u, 2u}};
    {
        mmid_out o;
        run_mmid(c, &o);
        CHECK(o.ret == 0u && o.err == 0u && o.group == 2u, "sequence: ret 0x%x, error %u, group %u", o.ret, o.err, o.group);
        const std::vector<mtile> tiles = model_tiles(c, o.group);
        CHECK(tiles.size() == 3u && tiles[0].e == 0u && tiles[0].pairs.size() == 2u && tiles[1].e == 1u && tiles[2].e == 2u,
              "sequence: the model's tiles");
        seq_builder b(c, o, tiles);
        b.head();
        b.front(0);          // the full tile of e0
        b.front(1);          // the flush: e1's tile, then the back of the one before
        b.back(0);
        b.front(2);
        b.back(1);
        b.back(2);           // the back of the last tile at the end
        check_sequence("sequence (room for two tiles)", o, b.v, c.scale);
        printf("  push sequence of a record of three tiles: %zu pushes\n", o.pushes.size());
    }
    {   // a ring that holds a tile of two rows and a tile of one row but not both beside each other (8 lines per row)
        mmid_case r = c;
        r.ring_lines = 20u;
        mmid_out o;
        run_mmid(r, &o);
        CHECK(o.ret == 0u && o.err == 0u, "sequence (small ring): ret 0x%x, error %u", o.ret, o.err);
        const std::vector<mtile> tiles = model_tiles(r, o.group);
        seq_builder b(r, o, tiles);
        b.head();
        b.front(0);
        b.back(0);           // no room beside the held tile: its back first
        b.front(1);
        b.front(2);          // room again (the first tile's block is released)
        b.back(1);
        b.back(2);
        check_sequence("sequence (back before the allocation)", o, b.v, r.scale);
    }
    {   // without a scale table and with a shared x row: no SOP / VOP, the STs depend on the MVM
        mmid_case r = c;
        r.scale = false;
        r.shared = true;
        mmid_out o;
        run_mmid(r, &o);
        CHECK(o.ret == 0u && o.err == 0u, "sequence (no table): ret 0x%x, error %u", o.ret, o.err);
        seq_builder b(r, o, model_tiles(r, o.group));
        b.head();
        b.front(0);
        b.front(1);
        b.back(0);
        b.front(2);
        b.back(1);
        b.back(2);
        check_sequence("sequence (no scale table, shared x)", o, b.v, r.scale);
    }
    for (int tiles = 1; tiles <= 2; ++tiles) {
        // T = 2, k = 1: the two pairs on one expert make one tile (the record's only ST is the first and last of the tile),
        // on two experts they make two tiles of one row
        mmid_case r = c;
        r.nu = 1u;
        r.ids = tiles == 1 ? std::vector<uint32_t>{3u, 3u} : std::vector<uint32_t>{3u, 1u};
        mmid_out o;
        run_mmid(r, &o);
        CHECK(o.ret == 0u && o.err == 0u && o.group == 2u, "sequence (%d tiles): ret 0x%x, error %u, group %u", tiles, o.ret, o.err, o.group);
        const std::vector<mtile> t = model_tiles(r, o.group);
        CHECK(t.size() == size_t(tiles), "sequence (%d tiles): the model's tiles %zu", tiles, t.size());
        seq_builder b(r, o, t);
        b.head();
        for (size_t i = 0; i < t.size(); ++i) {
            b.front(i);
            if (i > 0u) b.back(i - 1u);
        }
        b.back(t.size() - 1u);
        check_sequence(tiles == 1 ? "sequence (one tile)" : "sequence (two tiles of one row)", o, b.v, r.scale);
    }
}

// ---------------------------------------------------------------------------
// 3c. Random routings through the kernel (ASan + UBSan): the tiles of the streaming rule
// ---------------------------------------------------------------------------
void test_mmid_routings() {
    rng_t r = { 0x6B0A };
    struct routing { const char * name; uint32_t T, nu, ne, t_max, ids_stride, mode; bool scale, shared; };
    // mode: 0 uniform, 1 skewed (a few experts), 2 every pair on one expert, 3 every pair on its own expert (ne large),
    //       4 the same expert twice in a token
    const routing cases[] = {
        { "uniform", 37u, 4u, 16u, 4u, 0u, 0u, true, false },
        { "skewed", 40u, 8u, 64u, 8u, 0u, 1u, false, true },
        { "one expert", 9u, 8u, 16u, 4u, 0u, 2u, true, true },
        { "own expert", 5u, 8u, 64u, 4u, 0u, 3u, false, false },
        { "twice in a token", 21u, 4u, 8u, 3u, 0u, 4u, true, false },
        { "chunks (ids rows 520 B apart)", 70u, 3u, 16u, 4u, 520u, 0u, true, true },
        { "T = 2", 2u, 8u, 128u, 4u, 0u, 0u, true, false },
        { "T = 3", 3u, 8u, 128u, 4u, 0u, 0u, false, false },
        { "experts above 256", 24u, 8u, 300u, 4u, 0u, 5u, true, false },
        { "t_max 1", 7u, 4u, 16u, 1u, 0u, 0u, true, false },
    };
    long n_tiles = 0;
    for (const routing & rc : cases) {
        mmid_case c{rc.scale ? TSR_FMT_Q8_0 : TSR_FMT_Q4_K, 32u, rc.scale ? 64u : 256u, rc.T, rc.nu, rc.ne, rc.t_max, rc.shared, rc.scale, {}};
        c.ids_stride = rc.ids_stride;
        c.ids.resize(size_t(rc.T) * rc.nu);
        for (uint32_t t = 0; t < rc.T; ++t)
            for (uint32_t j = 0; j < rc.nu; ++j) {
                uint32_t e = rng_below(&r, rc.ne);
                if (rc.mode == 1) e = rng_below(&r, 5u) < 4u ? rng_below(&r, 3u) : e;
                if (rc.mode == 2) e = 3u % rc.ne;
                if (rc.mode == 3) e = (t * rc.nu + j) % rc.ne;
                if (rc.mode == 4) e = j % 2u == 1u ? c.ids[size_t(t) * rc.nu + j - 1u] : e;
                if (rc.mode == 5) e = rng_below(&r, 4u) == 0u ? 256u + rng_below(&r, rc.ne - 256u) : e % 256u;
                c.ids[size_t(t) * rc.nu + j] = e;
            }
        mmid_out o;
        run_mmid(c, &o);
        CHECK(o.ret == 0u && o.err == 0u, "routing %s: ret 0x%x, error %u", rc.name, o.ret, o.err);
        const bool grouped = o.group != 0u;
        CHECK(grouped == (rc.t_max >= 2u && rc.T > 1u), "routing %s: group %u", rc.name, o.group);
        // the pushes by unit
        std::vector<uint32_t> ld_x, st, mvm_e, mvm_t, qact_t, vop_rows;
        for (const pev & p : o.pushes) {
            switch (p.w[0] & 0xFFu) {
            case tsr::OP_LD: if (p.w[1] < o.one_line) ld_x.push_back(p.w[2]); break;
            case tsr::OP_ST: st.push_back(p.w[2]); break;
            case tsr::OP_QACT: qact_t.push_back(p.w[4]); break;
            case tsr::OP_MVM: mvm_e.push_back((p.w[3] - W_OFF) / o.exp_stride); mvm_t.push_back((p.w[5] >> 16) & 0xFu); break;
            case tsr::OP_VOP: vop_rows.push_back(p.w[6] & 0xFFFFu); break;
            default: break;
            }
        }
        std::vector<uint32_t> w_ld, w_st, w_e, w_t, w_vop;
        if (grouped) {
            for (const mtile & t : model_tiles(c, o.group)) {
                w_e.push_back(t.e);
                w_t.push_back(uint32_t(t.pairs.size()));
                w_vop.push_back(uint32_t(t.pairs.size()));
                for (uint32_t pr : t.pairs) {
                    w_ld.push_back(X_BEAT + (pr >> 4) * o.xtok + (pr & 15u) * o.xslot);
                    w_st.push_back(DST_BEAT + ((pr >> 4) * rc.nu + (pr & 15u)) * o.y_lines);
                }
            }
        } else {   // per token: one MVM of one row per pair, in order
            for (uint32_t t = 0; t < rc.T; ++t)
                for (uint32_t j = 0; j < rc.nu; ++j) {
                    w_e.push_back(c.ids[size_t(t) * rc.nu + j]);
                    w_t.push_back(1u);
                    w_vop.push_back(1u);
                    w_st.push_back(DST_BEAT + (t * rc.nu + j) * o.y_lines);
                }
        }
        CHECK(mvm_e == w_e && mvm_t == w_t && qact_t == (grouped ? w_t : qact_t) && st == w_st, "routing %s: the MVMs / STs differ from the model (%zu / %zu MVMs, %zu / %zu STs)",
              rc.name, mvm_e.size(), w_e.size(), st.size(), w_st.size());
        if (grouped) check_spm(rc.name, o, rc.scale);
        if (grouped) CHECK(ld_x == w_ld, "routing %s: the x LDs differ from the model (%zu / %zu)", rc.name, ld_x.size(), w_ld.size());
        if (rc.scale) CHECK(vop_rows == w_vop, "routing %s: the scale VOPs (%zu / %zu)", rc.name, vop_rows.size(), w_vop.size());
        // every pair is stored once
        std::vector<uint32_t> sorted = st;
        std::sort(sorted.begin(), sorted.end());
        CHECK(std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end() && st.size() == size_t(rc.T) * rc.nu,
              "routing %s: %zu STs for %u pairs", rc.name, st.size(), rc.T * rc.nu);
        n_tiles += long(mvm_e.size());
        printf("  routing %-30s group %u: %zu tiles for %u pairs\n", rc.name, o.group, mvm_e.size(), rc.T * rc.nu);
    }
    (void)n_tiles;
}

// ---------------------------------------------------------------------------
void test_select() {
    rng_t r = { 0x5E1Eull };
    const uint32_t specials[] = { 0x00000000u, 0x80000000u, 0x00000001u, 0x807FFFFFu, 0x7F800000u, 0xFF800000u, 0x7FC00000u,
                                  0xFFC00001u, 0x7F800001u, 0xFFFFFFFFu, 0x3F800000u, 0xBF800000u, 0x00800000u, 0x80800000u,
                                  0x41200000u, 0xC1200000u };
    long rows = 0;
    for (uint32_t n = 1; n <= 70; ++n)
        for (uint32_t k = 1; k <= (n < 16u ? n : 16u); ++k)
            for (int mode = 0; mode < 3; ++mode) {
                std::vector<uint32_t> x(n);
                for (uint32_t i = 0; i < n; ++i)
                    x[i] = mode == 0 ? rng_u32(&r) : mode == 1 ? specials[rng_below(&r, 16)] : f2u(float(rng_below(&r, 4)) * 0.25f);
                uint32_t ids[16], vals[16], d[32];
                tsr::host_moe_select_topk(x.data(), n, k, ids, vals);
                tsr_ref_topk(k, 1, n, d, 0, x.data(), 0);
                for (uint32_t j = 0; j < k; ++j) {
                    CHECK(ids[j] == d[2 * j] && vals[j] == d[2 * j + 1], "select n %u k %u mode %d slot %u: (%u, 0x%08x) vs (%u, 0x%08x)",
                          n, k, mode, j, ids[j], vals[j], d[2 * j], d[2 * j + 1]);
                }
                ++rows;
            }
    for (int it = 0; it < 4000; ++it) {                 // the production size: 128 experts, 8 used, and 512 / 16
        const uint32_t n = it % 2 ? 128u : 512u, k = it % 2 ? 8u : 16u;
        std::vector<uint32_t> x(n);
        for (uint32_t i = 0; i < n; ++i) x[i] = (it % 5 == 0) ? specials[rng_below(&r, 16)] : (it % 5 == 1) ? f2u(float(rng_below(&r, 3))) : rng_u32(&r);
        uint32_t ids[16], vals[16], d[32];
        tsr::host_moe_select_topk(x.data(), n, k, ids, vals);
        tsr_ref_topk(k, 1, n, d, 0, x.data(), 0);
        for (uint32_t j = 0; j < k; ++j) CHECK(ids[j] == d[2 * j] && vals[j] == d[2 * j + 1], "select n %u it %d slot %u", n, it, j);
        ++rows;
    }
    printf("  select: %ld rows equal to tsr_ref_topk\n", rows);
}

}  // namespace

int main() {
    test_route_plans();
    test_mmid_plans();
    test_mmid_group_plans();
    test_mmid_sequence();
    test_mmid_routings();
    test_select();
    return test_finish("fw_kernels_moe_host");
}
