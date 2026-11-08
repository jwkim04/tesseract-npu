// tsr/plan_gdn.hpp — planners of the Gated DeltaNet kinds SOFTPLUS (0x2D), SSM_CONV (0x2E) and
// GATED_DELTA_NET (0x2F) (docs/design/06-software-verification.md "SW" §7.10). Freestanding,
// C++17-compatible, constexpr-friendly: the firmware's handlers (sw/ctrl/kernels/gdn_kinds.cpp) and the
// host's supports_op call the same functions (SW-20). The contract and the declarations are
// tsr/plan.hpp's ("M7 planning"), which includes this file after them.
//
// All three kinds need the VU (l.caps CAPS_VU, else BAD_RECORD) and plan against l.budget.
//
// SOFTPLUS. Rows of n f32 are L = ceil(n / 16) lines. A row tile of t rows takes one block of five
// t L-line buffers (the input, -|x|, and three temporaries of the composition, gdn_kinds.cpp); every row
// in one tile when 5 rows L lines fit half the budget (two tiles in flight: the next tile's LD
// overlaps this tile's VOPs), else the largest t whose block fits half the budget, else one row in
// the whole budget (single buffer), else RET_SCRATCH. BAD_RECORD: n_args, n = 0 or L > 65,535, rows
// = 0 or above 65,535 per tile (a tile is at most VU_ROWS_MAX rows), dst rows that overlap.
// kind_plan: scratch_lines = block, row_lines = L, tile_rows = t, n_tiles.
//
// SSM_CONV. C channels (a multiple of 16, L_c = C / 16 lines per row), d_conv taps (2 .. 8), T tokens
// per sequence. A channel tile of lt lines (16 lt channels) takes one block of
// (d_conv - 1 + T) window rows + d_conv tap rows + 2 T accumulator rows of lt lines each. lt is the
// largest value with block <= half the budget, at most L_c and 65,535; the op is ceil(L_c / lt) tiles
// per sequence. RET_SCRATCH when lt would be 0 (even a one-line tile does not fit). BAD_RECORD:
// n_args, a channel count that is not a positive multiple of 16, d_conv outside 2 .. 8, T = 0 or
// above 65,535 - d_conv, no sequences, unknown flags, a destination token stride below L_c when
// there is more than one token.
// kind_plan: scratch_lines = block, row_lines = L_c, tile_rows = T, n_tiles = tiles per sequence x
// sequences, aux[0] = lt, aux[1] = tiles per sequence.
//
// GATED_DELTA_NET. S_v (a multiple of 16 in [16, 256]), H_k key heads, H_v value heads (a multiple
// of H_k), T tokens per sequence, n_seqs sequences. L = S_v / 16. One (sequence, head) takes one
// ring block: the state tile S (S_v L lines), the product tile P (S_v L lines) and two token slots
// of 5 L + 2 lines (k, q, v, the g and beta lines, the delta vector and the output vector), so
// block = 2 S_v L + 2 (5 L + 2). RET_SCRATCH when two blocks do not fit the budget (one is enough
// for correctness; the planner asks for two so that the next head's LD overlaps). BAD_RECORD:
// n_args, S_v not a multiple of 16 or outside [16, 256], H_k = 0, H_v not a multiple of H_k, T or
// n_seqs 0, T n_seqs above 2^24, a head stride below L with more than one head, a g / beta token
// stride below ceil(H_v / 16) lines with more than one token, a stride above 2^24 - 1.
// kind_plan: scratch_lines = block, row_lines = L, tile_rows = S_v, n_tiles = n_seqs H_v (heads),
// aux[0] = token slots (2), aux[1] = lines of one token slot (5 L + 2).
#pragma once

#include <stdint.h>

#include "tsr/plan.hpp"

namespace tsr {
namespace plan {
namespace gdn {

constexpr uint32_t STRIDE_MAX = 0x00FFFFFFu;   // an SPM / DRAM line stride field of a record: 24 bits
constexpr uint32_t SP_BUFFERS = 5u;            // SOFTPLUS: buffers of the composition (gdn_kinds.cpp)

constexpr uint32_t sp_block(uint32_t t, uint32_t L) { return SP_BUFFERS * t * L; }

}  // namespace gdn
}  // namespace plan

// SOFTPLUS (0x2D): opl::softplus words.
TSR_PLAN_HOT constexpr bool plan_softplus(const uint32_t * a, uint32_t n_args, const plan_limits & l, kind_plan * out) {
    namespace s = opl::softplus;
    if (n_args != s::NARGS || (l.caps & plan::CAPS_VU) == 0u) return plan::plan_fail(out, opl::RET_BAD_RECORD);
    const uint32_t n = a[s::N], rows = a[s::ROWS];
    if (n - 1u >= plan::EW_N_MAX || rows == 0u) return plan::plan_fail(out, opl::RET_BAD_RECORD);
    const uint32_t L = plan::vu_lines(n);
    if (rows > 1u && a[s::DST_STRIDE] < L) return plan::plan_fail(out, opl::RET_BAD_RECORD);
    const uint32_t half = l.budget >> 1;
    uint32_t t = 0;
    if (rows <= plan::VU_ROWS_MAX && plan::gdn::sp_block(rows, L) <= half) {
        t = rows;
    } else {
        t = half / (plan::gdn::SP_BUFFERS * L);
        if (t > plan::VU_ROWS_MAX) t = plan::VU_ROWS_MAX;
        if (t == 0u) {
            if (plan::gdn::sp_block(1u, L) > l.budget) return plan::plan_fail(out, opl::RET_SCRATCH);
            t = 1u;
        }
    }
    out->status = opl::RET_OK;
    out->scratch_lines = plan::gdn::sp_block(t, L);
    out->row_lines = L;
    out->tile_rows = t;
    out->n_tiles = t >= rows ? 1u : plan::ceil_div(rows, t);
    out->flags = out->n_tiles == 1u ? plan::KP_ONE_TILE : 0u;
    out->aux[0] = out->aux[1] = out->aux[2] = out->aux[3] = 0u;
    return true;
}

// SSM_CONV (0x2E): opl::sconv words.
TSR_PLAN_HOT constexpr bool plan_ssm_conv(const uint32_t * a, uint32_t n_args, const plan_limits & l, kind_plan * out) {
    namespace c = opl::sconv;
    if (n_args != c::NARGS || (l.caps & plan::CAPS_VU) == 0u) return plan::plan_fail(out, opl::RET_BAD_RECORD);
    const uint32_t C = a[c::N_CH], dc = a[c::D_CONV], T = a[c::N_TOKENS], ns = a[c::N_SEQS];
    if (C == 0u || (C & 15u) != 0u || dc < 2u || dc > c::D_CONV_MAX || T == 0u || T > plan::VU_ROWS_MAX - c::D_CONV_MAX ||
        ns == 0u || (a[c::FLAGS] & ~c::FLAGS_KNOWN) != 0u)
        return plan::plan_fail(out, opl::RET_BAD_RECORD);
    const uint32_t Lc = C >> 4;
    if (Lc > plan::VU_STRIDE_MAX || (T > 1u && a[c::DST_TOKEN_STRIDE] < Lc)) return plan::plan_fail(out, opl::RET_BAD_RECORD);
    const uint32_t rows = (dc - 1u) + T + dc + 2u * T;      // window + taps + accumulator + product rows
    uint32_t lt = (l.budget >> 1) / rows;
    if (lt > Lc) lt = Lc;
    if (lt > plan::VU_STRIDE_MAX) lt = plan::VU_STRIDE_MAX;
    if (lt == 0u) return plan::plan_fail(out, opl::RET_SCRATCH);
    const uint32_t per_seq = plan::ceil_div(Lc, lt);
    out->status = opl::RET_OK;
    out->scratch_lines = rows * lt;
    out->row_lines = Lc;
    out->tile_rows = T;
    out->n_tiles = per_seq * ns;
    out->flags = out->n_tiles == 1u ? plan::KP_ONE_TILE : 0u;
    out->aux[0] = lt;
    out->aux[1] = per_seq;
    out->aux[2] = out->aux[3] = 0u;
    return true;
}

// GATED_DELTA_NET (0x2F, two slots): opl::gdn words.
TSR_PLAN_HOT constexpr bool plan_gated_delta_net(const uint32_t * a, uint32_t n_args, const plan_limits & l,
                                                 kind_plan * out) {
    namespace g = opl::gdn;
    if (n_args != g::NARGS || (l.caps & plan::CAPS_VU) == 0u) return plan::plan_fail(out, opl::RET_BAD_RECORD);
    const uint32_t sv = a[g::S_V], hk = a[g::H_K], hv = a[g::H_V], T = a[g::N_TOKENS], ns = a[g::N_SEQS];
    if ((sv & 15u) != 0u || sv < 16u || sv > g::S_V_MAX || hk == 0u || hv < hk || hv % hk != 0u || T == 0u || ns == 0u ||
        uint64_t(T) * ns > 0x00FFFFFFull || uint64_t(ns) * hv > 0x00FFFFFFull)
        return plan::plan_fail(out, opl::RET_BAD_RECORD);
    const uint32_t L = sv >> 4, Lg = plan::vu_lines(hv);
    static_assert(g::S_V_MAX / 16u <= 16u, "L <= 16");
    if ((hk > 1u && (a[g::Q_HEAD_STRIDE] < L || a[g::K_HEAD_STRIDE] < L)) || (hv > 1u && a[g::V_HEAD_STRIDE] < L) ||
        (T * ns > 1u && (a[g::G_TOKEN_STRIDE] < Lg || a[g::BETA_TOKEN_STRIDE] < Lg)))
        return plan::plan_fail(out, opl::RET_BAD_RECORD);
    const uint32_t strides[7] = {a[g::Q_HEAD_STRIDE], a[g::Q_TOKEN_STRIDE], a[g::K_HEAD_STRIDE], a[g::K_TOKEN_STRIDE],
                                 a[g::V_HEAD_STRIDE], a[g::V_TOKEN_STRIDE], a[g::G_TOKEN_STRIDE] | a[g::BETA_TOKEN_STRIDE]};
    for (uint32_t i = 0; i < 7u; ++i)
        if (strides[i] > plan::gdn::STRIDE_MAX) return plan::plan_fail(out, opl::RET_BAD_RECORD);
    const uint32_t block = 3u * sv * L + 2u * (5u * L + 2u);
    if (block > (l.budget >> 1)) return plan::plan_fail(out, opl::RET_SCRATCH);
    out->status = opl::RET_OK;
    out->scratch_lines = block;
    out->row_lines = L;
    out->tile_rows = sv;
    out->n_tiles = ns * hv;
    out->flags = out->n_tiles == 1u ? plan::KP_ONE_TILE : 0u;
    out->aux[0] = 2u;
    out->aux[1] = 5u * L + 2u;
    out->aux[2] = out->aux[3] = 0u;
    return true;
}

}  // namespace tsr
