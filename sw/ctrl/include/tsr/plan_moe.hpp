// tsr/plan_moe.hpp — planners of the MoE kinds MOE_ROUTE (0x1E) and MUL_MAT_ID (0x1F)
// (docs/design/06-software-verification.md "SW" §7.2, §7.5, SW-60 to SW-66; docs/spec.md §13).
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
// MUL_MAT_ID. The MVM of one expert reads its weights once per tile of rows, so a record with T > 1 groups the
// (token, slot) pairs by expert (SW-62, SW-66). Two geometries, chosen by mmid_geometry (per token) and mmid_grouping:
//   grouped (T > 1, a tile of at least two rows plans): one ring block per tile of m <= group rows of one expert,
//       [x_0 .. x_(m-1) | QA_0 .. QA_(m-1) | y_0 .. y_(m-1)]   (a MUL_MAT's tile, tsr/plan.hpp mm_plan)
//     with group = the T-tile of mm_plan (at most T_MAX, at most T, two tiles in the ring budget), and a head block:
//     the chunk of ids the RISC-V reads (ID_CHUNK_LINES lines), the scale table (L(n_expert) lines with a SCALE
//     table) and the pending table, for each of the first ng = min(n_expert, GROUP_MAX_EXPERTS) experts a slot of pw
//     words (pw = the power of two >= group - 1, so a slot stays inside one SPM line) that holds the pairs
//     (token << 4 | slot) of the expert that wait for a full tile (L(ng pw) lines). It must fit the one-tile block.
//   per token (T = 1, a tile of one row, or a head block that does not fit the one-tile block): one ring block per
//     token, [x_0 .. x_(xs-1) | QA_0 .. QA_(xs-1) | y_0 .. y_(k-1)]  with xs = 1 when the x row is shared by the
//     slots (X_SLOT_STRIDE 0: the gate and up projections) and xs = k otherwise, and a head block of the ids chunk and
//     the scale table.
// kind_plan, grouped: scratch_lines = tile block + head block, row_lines = y lines of one row, tile_rows = group,
// n_tiles = ceil(T k / group) (the least; the routing decides the real count, so KP_ONE_TILE stays clear), aux[0] = tile
// block lines (group x lines of a row), aux[1] = head block lines, aux[2] = xs (as per token), aux[3] = group.
// kind_plan, per token: scratch_lines = token block + head block, row_lines, tile_rows = 1, n_tiles = T,
// aux[0] = token block lines, aux[1] = head block lines, aux[2] = xs, aux[3] = 0.
// BAD_RECORD: a wrong n_args; FLAGS != 0; an MVM shape mm_plan refuses (fmt, K, N); k = 0 or > 16;
// n_expert = 0 or > 4096; T = 0; EXP_STRIDE = 0 with more than one expert; a slot stride that is neither
// 0 nor at least x_lines; a token stride below the rows of a token; dst stride below y_lines; an ids
// stride that is not a multiple of 4 or is below 4 k (T > 1); a SCALE table without a VU (l.caps) or
// with a handle that is not a line. RET_SCRATCH: the token (or tile) block plus the head block do not fit the budget.
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
constexpr uint32_t GROUP_MAX_EXPERTS = 256u;    // experts with a pending slot (the handler's count array); others run alone
constexpr uint32_t GROUP_MAX_TOKENS = 1u << 28;  // a pending entry is (token << 4) | slot

// Words of a pending slot: the power of two >= group - 1 (group >= 2).
TSR_PLAN_HOT constexpr uint32_t pend_slot_words(uint32_t group) {
    uint32_t w = 1u;
    while (w < group - 1u) w <<= 1;
    return w;
}

// MUL_MAT_ID geometry of the per-token path (valid fmt / K / N only: the caller has run mm_plan). The decode token's
// record (T = 1) runs this alone: nothing of the grouping below is evaluated for it beyond the compare of T (the
// grouping costs RISC-V cycles on the token's critical path).
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

// The grouped geometry of a record with T > 1: group = the T-tile of mm_plan for T tokens (as many rows as T_MAX
// allows with two tiles in the ring budget, at most T) when it is at least 2 and the head block (the ids chunk, the
// scale table and the pending table) fits the one-tile block (one_lines lines); else group = 0 and the per-token
// geometry stands. tok_lines / head_lines are the grouped block sizes (group rows of the tile, the head block).
struct mmid_grp {
    uint32_t group;     // rows of a full tile; 0: the per-token path
    uint32_t pw;        // words of a pending slot
    uint32_t tok_lines; // ring block of a full tile: group x lines of a row
    uint32_t head_lines;
};
TSR_PLAN_HOT constexpr mmid_grp mmid_grouping(const mm_limits & ml, uint32_t fmt, uint32_t N, uint32_t K, uint32_t n_expert,
                                              bool scale, uint32_t T, uint32_t one_lines) {
    mmid_grp r{};
    mm_tiling tl{};
    if (T <= 1u || T > GROUP_MAX_TOKENS || mm_plan(ml, fmt, N, K, T, &tl) != opl::RET_OK || tl.t_tile < 2u) return r;
    const uint32_t ng = n_expert < GROUP_MAX_EXPERTS ? n_expert : GROUP_MAX_EXPERTS;
    const uint32_t pw = pend_slot_words(tl.t_tile);
    const uint32_t head = ID_CHUNK_LINES + (scale ? vu_lines(n_expert) : 0u) + vu_lines(ng * pw);
    if (head > one_lines) return r;
    r.group = tl.t_tile;
    r.pw = pw;
    r.tok_lines = tl.t_tile * tl.lpt;
    r.head_lines = head;
    return r;
}

}  // namespace plan

namespace plan {
// The checks of a MUL_MAT_ID record and its per-token geometry, shared by the T = 1 planner and the batch tail.
TSR_PLAN_HOT constexpr bool mmid_checks(const uint32_t * args, uint32_t n_args, const plan_limits & l, kind_plan * out,
                                        mmid_geom * g) {
    namespace m = opl::mmid;
    if (n_args != m::NARGS || args[m::FLAGS] != 0u) return plan_fail(out, opl::RET_BAD_RECORD);
    const uint32_t k = args[m::N_USED], ne = args[m::N_EXPERT], T = args[m::T];
    if (k == 0u || k > m::MAX_USED || ne == 0u || ne > m::MAX_EXPERTS || T == 0u ||
        (ne > 1u && args[m::EXP_STRIDE] == 0u))
        return plan_fail(out, opl::RET_BAD_RECORD);
    const bool has_scale = args[m::SCALE] != opl::HANDLE_NONE;
    if (has_scale && (l.caps & plan::CAPS_VU) == 0u) return plan_fail(out, opl::RET_BAD_RECORD);
    const uint32_t st = mmid_geometry(mm_limits{l.k_max, l.t_max, l.budget}, args[m::FMT], args[m::N], args[m::K], k,
                                      args[m::X_SLOT_STRIDE] == 0u, ne, has_scale, g);
    if (st != opl::RET_OK) return plan_fail(out, st == opl::RET_SCRATCH ? opl::RET_SCRATCH : opl::RET_BAD_RECORD);
    if ((args[m::X_SLOT_STRIDE] != 0u && args[m::X_SLOT_STRIDE] < g->tl.x_lines) ||
        (T > 1u && args[m::X_TOK_STRIDE] < (args[m::X_SLOT_STRIDE] == 0u ? g->tl.x_lines : k * args[m::X_SLOT_STRIDE])) ||
        (k * T > 1u && args[m::DST_STRIDE] < g->tl.y_lines) || (args[m::IDS_STRIDE] & 3u) != 0u ||
        (T > 1u && args[m::IDS_STRIDE] < 4u * k))
        return plan_fail(out, opl::RET_BAD_RECORD);
    return true;
}

// The plan of a record with T > 1: the grouped geometry when mmid_grouping finds one, else the per-token one.
// Out of line (the firmware calls it once per batch record), and it fills its own kind_plan, which it returns: a
// kind_plan whose address the T = 1 planner passed on would not stay in registers there (the decode token pays for it).
[[gnu::noinline]] constexpr kind_plan plan_mmid_batch(const uint32_t * args, const plan_limits & l, const mmid_geom & g0) {
    namespace m = opl::mmid;
    kind_plan o{};
    kind_plan * const out = &o;
    const uint32_t k = args[m::N_USED], T = args[m::T];
    mmid_geom g = g0;
    const mmid_grp gr = mmid_grouping(mm_limits{l.k_max, l.t_max, l.budget}, args[m::FMT], args[m::N], args[m::K], args[m::N_EXPERT],
                                      args[m::SCALE] != opl::HANDLE_NONE, T, one_block_lines(l.spm_lines));
    if (gr.group != 0u) {
        g.tok_lines = gr.tok_lines;
        g.head_lines = gr.head_lines;
    }
    if (uint64_t(g.tok_lines) + g.head_lines > l.budget) {
        plan_fail(out, opl::RET_SCRATCH);
        return o;
    }
    out->status = opl::RET_OK;
    out->scratch_lines = g.tok_lines + g.head_lines;
    out->row_lines = g.tl.y_lines;
    out->tile_rows = 1u;
    out->n_tiles = T;
    if (gr.group != 0u) {   // ceil(T k / group) without a 64-bit divide: T = q group + r
        const uint32_t q = T / gr.group, r = T % gr.group;
        out->tile_rows = gr.group;
        out->n_tiles = q * k + (r * k + gr.group - 1u) / gr.group;
    }
    out->flags = 0u;   // KP_ONE_TILE stays clear: the routing can run more tiles than the planned least
    out->aux[0] = g.tok_lines;
    out->aux[1] = g.head_lines;
    out->aux[2] = g.xs;
    out->aux[3] = gr.group;
    return o;
}
}  // namespace plan

TSR_PLAN_HOT constexpr bool plan_mul_mat_id(const uint32_t * args, uint32_t n_args, const plan_limits & l, kind_plan * out) {
    using namespace plan;
    mmid_geom g{};
    if (!mmid_checks(args, n_args, l, out, &g)) return false;
    if (args[opl::mmid::T] > 1u) [[unlikely]] {
        const kind_plan r = plan_mmid_batch(args, l, g);
        *out = r;
        return r.status == opl::RET_OK;
    }
    if (uint64_t(g.tok_lines) + g.head_lines > l.budget) return plan_fail(out, opl::RET_SCRATCH);
    out->status = opl::RET_OK;
    out->scratch_lines = g.tok_lines + g.head_lines;
    out->row_lines = g.tl.y_lines;
    out->tile_rows = 1u;
    out->n_tiles = 1u;   // T = 1
    out->flags = KP_ONE_TILE;
    out->aux[0] = g.tok_lines;
    out->aux[1] = g.head_lines;
    out->aux[2] = g.xs;
    out->aux[3] = 0u;
    return true;
}

// The shapes of Gemma 4 26B-A4B on the U200 plan: the router for one token and a prefill tile of 16, the
// gate-up projection (Q4_K, 1408 x 2816, the x row shared by 8 slots) and the down projection (Q8_0,
// 2816 x 704, one x row per slot), and the grouping of their pairs by expert.
namespace plan {
namespace moe_check {
constexpr kind_plan route(uint32_t T) {
    const uint32_t a[opl::moer::NARGS] = {0x10000000u, 8u, 0x20000000u, 8u, 0x30000000u, 128u, 8u, T, 0u};
    kind_plan k{};
    tsr::plan_moe_route(a, opl::moer::NARGS, LIMITS_U200, &k);
    return k;
}
constexpr kind_plan mmid(uint32_t fmt, uint32_t N, uint32_t K, uint32_t slot_stride, uint32_t T, bool scale,
                         uint32_t n_expert = 128u, const plan_limits & l = LIMITS_U200) {
    const uint32_t xl = K / 16u;
    const uint32_t a[opl::mmid::NARGS] = {0x10000000u, N / 16u, 0x20000000u, slot_stride, 8u * xl, 0x30000000u, 512u, 0u,
                                          1u << 20, fmt, N, K, T, 8u, n_expert, scale ? 0x40000000u : opl::HANDLE_NONE, 0u};
    kind_plan k{};
    tsr::plan_mul_mat_id(a, opl::mmid::NARGS, l, &k);
    return k;
}
// The release image: T_MAX 4 (two weight streams for at most two tokens); a 512 KiB SPM with T_MAX 8.
constexpr plan_limits limits_t4() {
    plan_limits l = LIMITS_U200;
    l.t_max = 4u;
    return l;
}
constexpr plan_limits limits_small_spm() {
    const dev_cfg c = {8192u, 8u, 32768u, 4u, 9344u, 16u, CAPS_VU | CAPS_ATT};
    return plan_limits_of(c);
}
constexpr kind_plan gate_up(uint32_t T) { return mmid(FMT_Q4_K, 1408u, 2816u, 0u, T, false, 128u, limits_t4()); }
constexpr kind_plan down(uint32_t T) { return mmid(FMT_Q8_0, 2816u, 704u, 44u, T, true, 128u, limits_t4()); }
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
// A decode token (T = 1) takes the per-token path; a batch takes tiles of 4 rows on the release image, with the
// head block of 128 experts (the ids chunk, the scale table, the pending table: 128 slots of 4 words = 32 lines).
static_assert(plan::moe_check::gate_up(1u).aux[3] == 0u && plan::moe_check::down(1u).aux[3] == 0u &&
                  plan::moe_check::gate_up(1u).n_tiles == 1u && plan::moe_check::gate_up(1u).tile_rows == 1u,
              "plan_moe.hpp: a decode token of the Gemma 4 26B-A4B expert projections takes the per-token path");
static_assert(plan::moe_check::gate_up(2u).aux[3] == 2u && plan::moe_check::gate_up(3u).aux[3] == 3u &&
                  plan::moe_check::gate_up(512u).status == opl::RET_OK && plan::moe_check::gate_up(512u).aux[3] == 4u &&
                  plan::moe_check::gate_up(512u).tile_rows == 4u && plan::moe_check::gate_up(512u).n_tiles == 1024u &&
                  plan::moe_check::gate_up(512u).aux[1] == plan::ID_CHUNK_LINES + 32u &&
                  plan::moe_check::gate_up(512u).aux[0] == 4u * (176u + plan::qa_lines(plan::QA_A256, 2816u) + 88u) &&
                  plan::moe_check::down(512u).status == opl::RET_OK && plan::moe_check::down(512u).aux[3] == 4u &&
                  plan::moe_check::down(512u).aux[1] == plan::ID_CHUNK_LINES + 8u + 32u &&
                  plan::moe_check::down(512u).aux[2] == 8u && plan::moe_check::down(512u).n_tiles == 1024u,
              "plan_moe.hpp: the Gemma 4 26B-A4B expert projections group by expert in tiles of 4 rows on the release image");
// A table of 4,096 experts on a 512 KiB SPM keeps the per-token path (the pending table does not fit the one-tile
// block); every record that planned with the per-token geometry still plans.
static_assert(plan::moe_check::mmid(plan::FMT_Q8_0, 64u, 64u, 4u, 9u, true, 4096u, plan::moe_check::limits_small_spm()).status ==
                      opl::RET_OK &&
                  plan::moe_check::mmid(plan::FMT_Q8_0, 64u, 64u, 4u, 9u, true, 4096u, plan::moe_check::limits_small_spm()).aux[3] == 0u &&
                  plan::moe_check::mmid(plan::FMT_Q8_0, 64u, 64u, 4u, 9u, true, 4096u, plan::moe_check::limits_small_spm()).n_tiles == 9u &&
                  plan::moe_check::mmid(plan::FMT_Q8_0, 64u, 64u, 4u, 9u, true, 4096u).aux[3] != 0u,
              "plan_moe.hpp: 4,096 experts with a table on a small SPM plan on the per-token path");

}  // namespace tsr
