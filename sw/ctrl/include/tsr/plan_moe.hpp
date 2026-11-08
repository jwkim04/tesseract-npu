// tsr/plan_moe.hpp — planners of the MoE kinds MOE_ROUTE (0x1E) and MUL_MAT_ID (0x1F)
// (docs/design/06-software-verification.md "SW" §7.2, §7.5, SW-60 to SW-65; docs/spec.md §13).
// Freestanding, C++17-compatible, constexpr-friendly: the firmware's handlers (sw/ctrl/kernels/
// k_moe.cpp) and the host's supports_op call the same functions (SW-20). The planner contract is
// tsr/plan.hpp's ("M7 planning"), which declares plan_moe_route / plan_mul_mat_id and includes this
// file after the declarations.
//
// MOE_ROUTE. Per tile of tr = min(T, 64) tokens (the c vector of a tile is up to four lines, VU §4.3) one ring
// block, in this order:
//     P    tr x L(n_expert)   the logits, then the softmax probabilities (in place)
//     V    L(tr)              the c vector: max, then 1 / sum, then 1 / (sum of the selected)
//     SEL  tr                 one line per token: the k selected probabilities, then their weights (in place)
//     IDL  tr                 one line per token: the k expert numbers, 0 in the rest of the line
//     PK   L(tr k)            the weights packed, k words per token (not used when tr = 1: SEL is the packed form)
// kind_plan: scratch_lines = the block of a full tile, row_lines = L(n_expert), tile_rows = tr, n_tiles,
// aux[0] = k, aux[1] = block lines.
// BAD_RECORD: no VU (l.caps without CAPS_VU); a wrong n_args; FLAGS != 0; k = 0 or k > 16 or k > n_expert;
// n_expert > 512; T = 0; logits or ids rows (T > 1) that overlap (strides below L(n_expert) / 1).
// RET_SCRATCH: the block does not fit the budget.
//
// MUL_MAT_ID. Per token one ring block (a MUL_MAT's [x | QA | y] per slot, tsr/plan.hpp mm_plan with
// T = 1):  [x_0 .. x_(xs-1) | QA_0 .. QA_(xs-1) | y_0 .. y_(k-1)]  with xs = 1 when the x row is shared by
// the slots (X_SLOT_STRIDE 0: the gate and up projections) and xs = k otherwise, plus one head block per
// record: the chunk of ids the RISC-V reads (ID_CHUNK_LINES lines) and the scale table (L(n_expert) lines
// with a SCALE table). kind_plan: scratch_lines = token block + head block, row_lines = y lines of one
// row, tile_rows = 1, n_tiles = T, aux[0] = token block lines, aux[1] = head block lines, aux[2] = xs.
// BAD_RECORD: a wrong n_args; FLAGS != 0; an MVM shape mm_plan refuses (fmt, K, N); k = 0 or > 16;
// n_expert = 0 or > 4096; T = 0; EXP_STRIDE = 0 with more than one expert; a slot stride that is neither
// 0 nor at least x_lines; a token stride below the rows of a token; dst stride below y_lines; an ids
// stride that is not a multiple of 4 or is below 4 k (T > 1); a SCALE table without a VU (l.caps) or
// with a handle that is not a line. RET_SCRATCH: the token block plus the head block do not fit the budget.
#pragma once

#include <stdint.h>

#include "tsr/plan.hpp"

namespace tsr {
namespace plan {

constexpr uint32_t ID_CHUNK_LINES = 128u;     // SPM lines of the ids the RISC-V reads per chunk (MUL_MAT_ID)

// MOE_ROUTE geometry.
struct route_geom {
    uint32_t ln;        // L(n_expert)
    uint32_t tr;        // tokens per tile
    uint32_t n_tiles;
    uint32_t lines;     // ring block of a full tile
    uint32_t pk_lines;  // L(tr k)
};
constexpr route_geom route_geometry(uint32_t n_expert, uint32_t k, uint32_t T) {
    route_geom g{};
    g.ln = vu_lines(n_expert);
    g.tr = T < opl::moer::TILE_TOKENS ? T : opl::moer::TILE_TOKENS;
    g.n_tiles = ceil_div(T, opl::moer::TILE_TOKENS);
    g.pk_lines = vu_lines(g.tr * k);
    g.lines = g.tr * g.ln + vu_lines(g.tr) + 2u * g.tr + (g.tr > 1u ? g.pk_lines : 0u);
    return g;
}

}  // namespace plan

TSR_PLAN_HOT constexpr bool plan_moe_route(const uint32_t * args, uint32_t n_args, const plan_limits & l, kind_plan * out) {
    namespace r = opl::moer;
    using namespace plan;
    if (n_args != r::NARGS || (l.caps & plan::CAPS_VU) == 0u) return plan_fail(out, opl::RET_BAD_RECORD);
    const uint32_t n = args[r::N_EXPERT], k = args[r::N_USED], T = args[r::T];
    if (args[r::FLAGS] != 0u || k == 0u || k > r::MAX_USED || n > r::MAX_EXPERTS || k > n || T == 0u)
        return plan_fail(out, opl::RET_BAD_RECORD);
    const route_geom g = route_geometry(n, k, T);
    if (T > 1u && (args[r::LOGITS_STRIDE] < g.ln || args[r::IDS_STRIDE] < 1u)) return plan_fail(out, opl::RET_BAD_RECORD);
    if (g.lines > l.budget) return plan_fail(out, opl::RET_SCRATCH);
    out->status = opl::RET_OK;
    out->scratch_lines = g.lines;
    out->row_lines = g.ln;
    out->tile_rows = g.tr;
    out->n_tiles = g.n_tiles;
    out->flags = g.n_tiles == 1u ? KP_ONE_TILE : 0u;
    out->aux[0] = k;
    out->aux[1] = g.lines;
    out->aux[2] = 0u;
    out->aux[3] = 0u;
    return true;
}

namespace plan {
// MUL_MAT_ID geometry (valid fmt / K / N only: the caller has run mm_plan).
struct mmid_geom {
    mm_tiling tl;       // the MUL_MAT tiling of one (token, slot) pair
    uint32_t xs;        // x rows per token block
    uint32_t tok_lines; // token block
    uint32_t head_lines;// head block
};
TSR_PLAN_HOT constexpr uint32_t mmid_geometry(const mm_limits & ml, uint32_t fmt, uint32_t N, uint32_t K, uint32_t k,
                                              bool shared_x, uint32_t n_expert, bool scale, mmid_geom * g) {
    const uint32_t st = mm_plan(ml, fmt, N, K, 1u, &g->tl);
    if (st != opl::RET_OK) return st;
    g->xs = shared_x ? 1u : k;
    g->tok_lines = g->xs * (g->tl.x_lines + g->tl.qa_lines) + k * g->tl.y_lines;
    g->head_lines = ID_CHUNK_LINES + (scale ? vu_lines(n_expert) : 0u);
    return opl::RET_OK;
}

}  // namespace plan

TSR_PLAN_HOT constexpr bool plan_mul_mat_id(const uint32_t * args, uint32_t n_args, const plan_limits & l, kind_plan * out) {
    namespace m = opl::mmid;
    using namespace plan;
    if (n_args != m::NARGS || args[m::FLAGS] != 0u) return plan_fail(out, opl::RET_BAD_RECORD);
    const uint32_t k = args[m::N_USED], ne = args[m::N_EXPERT], T = args[m::T];
    if (k == 0u || k > m::MAX_USED || ne == 0u || ne > m::MAX_EXPERTS || T == 0u ||
        (ne > 1u && args[m::EXP_STRIDE] == 0u))
        return plan_fail(out, opl::RET_BAD_RECORD);
    const bool has_scale = args[m::SCALE] != opl::HANDLE_NONE;
    if (has_scale && (l.caps & plan::CAPS_VU) == 0u) return plan_fail(out, opl::RET_BAD_RECORD);
    mmid_geom g{};
    const uint32_t st = mmid_geometry(mm_limits{l.k_max, l.t_max, l.budget}, args[m::FMT], args[m::N], args[m::K], k,
                                      args[m::X_SLOT_STRIDE] == 0u, ne, has_scale, &g);
    if (st != opl::RET_OK) return plan_fail(out, st == opl::RET_SCRATCH ? opl::RET_SCRATCH : opl::RET_BAD_RECORD);
    if ((args[m::X_SLOT_STRIDE] != 0u && args[m::X_SLOT_STRIDE] < g.tl.x_lines) ||
        (T > 1u && args[m::X_TOK_STRIDE] < (args[m::X_SLOT_STRIDE] == 0u ? g.tl.x_lines : k * args[m::X_SLOT_STRIDE])) ||
        (k * T > 1u && args[m::DST_STRIDE] < g.tl.y_lines) || (args[m::IDS_STRIDE] & 3u) != 0u ||
        (T > 1u && args[m::IDS_STRIDE] < 4u * k))
        return plan_fail(out, opl::RET_BAD_RECORD);
    if (uint64_t(g.tok_lines) + g.head_lines > l.budget) return plan_fail(out, opl::RET_SCRATCH);
    out->status = opl::RET_OK;
    out->scratch_lines = g.tok_lines + g.head_lines;
    out->row_lines = g.tl.y_lines;
    out->tile_rows = 1u;
    out->n_tiles = T;
    out->flags = T == 1u ? KP_ONE_TILE : 0u;
    out->aux[0] = g.tok_lines;
    out->aux[1] = g.head_lines;
    out->aux[2] = g.xs;
    out->aux[3] = 0u;
    return true;
}

// The shapes of Gemma 4 26B-A4B on the U200 plan: the router for one token and a prefill tile of 16, the
// gate-up projection (Q4_K, 1408 x 2816, the x row shared by 8 slots) and the down projection (Q8_0,
// 2816 x 704, one x row per slot).
namespace plan {
namespace moe_check {
constexpr kind_plan route(uint32_t T) {
    const uint32_t a[opl::moer::NARGS] = {0x10000000u, 8u, 0x20000000u, 8u, 0x30000000u, 128u, 8u, T, 0u};
    kind_plan k{};
    tsr::plan_moe_route(a, opl::moer::NARGS, LIMITS_U200, &k);
    return k;
}
constexpr kind_plan mmid(uint32_t fmt, uint32_t N, uint32_t K, uint32_t slot_stride, uint32_t T, bool scale) {
    const uint32_t xl = K / 16u;
    const uint32_t a[opl::mmid::NARGS] = {0x10000000u, N / 16u, 0x20000000u, slot_stride, 8u * xl, 0x30000000u, 512u, 0u,
                                          1u << 20, fmt, N, K, T, 8u, 128u, scale ? 0x40000000u : opl::HANDLE_NONE, 0u};
    kind_plan k{};
    tsr::plan_mul_mat_id(a, opl::mmid::NARGS, LIMITS_U200, &k);
    return k;
}
}  // namespace moe_check
}  // namespace plan
static_assert(plan::moe_check::route(1u).status == opl::RET_OK && plan::moe_check::route(1u).scratch_lines == 8u + 1u + 2u &&
                  plan::moe_check::route(64u).status == opl::RET_OK && plan::moe_check::route(64u).n_tiles == 1u &&
                  plan::moe_check::route(65u).n_tiles == 2u && plan::moe_check::route(512u).n_tiles == 8u,
              "plan_moe.hpp: MOE_ROUTE of 128 experts, 8 used, plans on the U200");
static_assert(plan::moe_check::mmid(plan::FMT_Q4_K, 1408u, 2816u, 0u, 1u, false).status == opl::RET_OK &&
                  plan::moe_check::mmid(plan::FMT_Q4_K, 1408u, 2816u, 0u, 1u, false).aux[2] == 1u &&
                  plan::moe_check::mmid(plan::FMT_Q8_0, 2816u, 704u, 44u, 1u, true).status == opl::RET_OK &&
                  plan::moe_check::mmid(plan::FMT_Q8_0, 2816u, 704u, 44u, 1u, true).aux[2] == 8u &&
                  plan::moe_check::mmid(plan::FMT_Q8_0, 2816u, 704u, 44u, 512u, true).n_tiles == 512u,
              "plan_moe.hpp: the Gemma 4 26B-A4B expert projections plan on the U200");

}  // namespace tsr
