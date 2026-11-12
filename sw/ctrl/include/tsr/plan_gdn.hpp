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
// of H_k), T tokens per sequence, n_seqs sequences. L = S_v / 16. The head by head pipeline (the
// composition, and a fused record that the token block pipeline below does not take) takes one ring
// block per record: two state tiles (S_v L lines each), the product tile P (S_v L lines) and two
// token slots of 5 L + 2 lines (k, q, v, the g and beta lines, the delta vector and the output
// vector), so block = 3 S_v L + 2 (5 L + 2). RET_SCRATCH when two blocks do not fit the budget (one is
// enough for correctness; the planner asks for two so that the next head's LD overlaps). BAD_RECORD:
// n_args, S_v not a multiple of 16 or outside [16, 256], H_k = 0, H_v not a multiple of H_k, T or
// n_seqs 0, T n_seqs above 2^24, a head stride below L with more than one head, a g / beta token
// stride below ceil(H_v / 16) lines with more than one token, a stride above 2^24 - 1.
// kind_plan: scratch_lines = block, row_lines = L, tile_rows = S_v, n_tiles = n_seqs H_v (heads),
// aux[0] = token slots (2), aux[1] = lines of one token slot (5 L + 2), aux[2] = aux[3] = 0.
// A record on the fused path (plan::gdn::use_fused: the device has CAPS_GDN and S_v >= 64, SW-78) takes the token
// block pipeline (gdn_kinds.cpp gdn_run_tok) when its block fits the budget. The operands of all the heads of a
// group of Gs heads load together, one token block per (sequence, token) step, Lg = ceil(H_v / 16): g, beta and alpha
// (Lg lines each), the k and q rows of the H_k key heads (L lines each), the v and o rows of the group's heads (L
// lines each): tok_lines = 3 Lg + 2 (H_k + Gs) L. Two token blocks alternate. The state tiles are R tiles of S_v L
// lines. With T = 1 R = 2 and Gs = H_v (one group per sequence): the heads of the sequences, in order, use the two
// tiles in turn. With T > 1 R = Gs, one tile per head of a group, which stay in the SPM for all T tokens (tokens
// outer, heads inner); Gs is the largest value at most H_v whose block fits the budget. The block is
// R S_v L + 2 tok_lines(Gs) lines: the whole budget may be used (the record's ring block is released when its last
// ST is done). A fused record whose block does not fit (T = 1 with Gs = H_v; T > 1 not even with Gs = 1) takes the
// head by head pipeline above, and RET_SCRATCH when that block does not fit half the budget.
// kind_plan: scratch_lines = block, row_lines = L, tile_rows = S_v, n_tiles = n_seqs H_v, aux[0] = token blocks (2),
// aux[1] = lines of one token block, aux[2] = R, aux[3] = Gs.
// The plan depends on l.caps through plan::gdn::use_fused (CAPS bit 8): the host and the firmware plan one way
// for a device. Without the bit the plan is the head by head plan. A record that the head by head plan takes
// is never refused by the token block plan: it falls back to the head by head plan.
#pragma once

#include <stdint.h>

#include "tsr/plan.hpp"

namespace tsr {
namespace plan {
namespace gdn {

constexpr uint32_t STRIDE_MAX = 0x00FFFFFFu;   // an SPM / DRAM line stride field of a record: 24 bits
constexpr uint32_t SP_BUFFERS = 5u;            // SOFTPLUS: buffers of the composition (gdn_kinds.cpp)

constexpr uint32_t sp_block(uint32_t t, uint32_t L) { return SP_BUFFERS * t * L; }

// GATED_DELTA_NET takes the fused path (the VU's GDN op, VU §7.12; numerics revision 2, SW-78) when the device
// has the VU and its CAPS bit 8 and S_v is in the op's range of m (64 to 256, VU §7.12.4; S_v is at most 256 for
// every record). Otherwise it runs the composition of revision 1. The firmware's handler and the references
// choose with this function and nothing else, so a record's output depends on (record, CAPS) only.
constexpr uint32_t FUSED_SV_MIN = 64u;
constexpr bool use_fused(uint32_t caps, uint32_t s_v) {
    return (caps & CAPS_VU) != 0u && (caps & CAPS_GDN) != 0u && s_v >= FUSED_SV_MIN;
}

// The token block pipeline of the fused path: lines of one token block of a group of gs heads (Lg lines of g, beta
// and alpha each; the k and q rows of hk key heads, the v and o rows of gs heads, L lines per row).
constexpr uint32_t TOK_BLOCKS = 2u;
constexpr uint32_t tok_lines(uint32_t Lg, uint32_t hk, uint32_t gs, uint32_t L) { return 3u * Lg + 2u * (hk + gs) * L; }

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
    if (plan::gdn::use_fused(l.caps, sv)) {
        // The token block pipeline: R state tiles (2 with one token, else one per head of a group of gs heads). The
        // sums stay below 2^32: hv, hk < 2^24, L <= 16, and gs (SL + 4 L) <= the budget (< 2^24) when T > 1.
        const uint32_t SL = sv * L;
        uint32_t R = 2u, gs = hv;
        if (T > 1u) {
            const uint32_t fixed = 2u * plan::gdn::tok_lines(Lg, hk, 0u, L);   // two token blocks without v and o rows
            gs = l.budget > fixed ? (l.budget - fixed) / (SL + 4u * L) : 0u;
            if (gs > hv) gs = hv;
            R = gs;
        }
        const uint32_t tb = plan::gdn::tok_lines(Lg, hk, gs, L), blk = R * SL + 2u * tb;
        if (gs != 0u && blk <= l.budget) {
            out->status = opl::RET_OK;
            out->scratch_lines = blk;
            out->row_lines = L;
            out->tile_rows = sv;
            out->n_tiles = ns * hv;
            out->flags = out->n_tiles == 1u ? plan::KP_ONE_TILE : 0u;
            out->aux[0] = plan::gdn::TOK_BLOCKS;
            out->aux[1] = tb;
            out->aux[2] = R;
            out->aux[3] = gs;
            return true;
        }
    }
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

namespace plan {
namespace gdn_check {
// A record of S_v sv with a dense layout (head strides L, token strides one token's rows), planned for a device of
// `caps` and a ring budget.
constexpr kind_plan probe(uint32_t caps, uint32_t budget, uint32_t sv, uint32_t hk, uint32_t hv, uint32_t T, uint32_t ns) {
    const uint32_t L = sv / 16u, Lg = (hv + 15u) / 16u;
    uint32_t a[opl::gdn::NARGS] = {};
    a[opl::gdn::Q_HEAD_STRIDE] = a[opl::gdn::K_HEAD_STRIDE] = a[opl::gdn::V_HEAD_STRIDE] = L;
    a[opl::gdn::Q_TOKEN_STRIDE] = a[opl::gdn::K_TOKEN_STRIDE] = hk * L;
    a[opl::gdn::V_TOKEN_STRIDE] = hv * L;
    a[opl::gdn::G_TOKEN_STRIDE] = a[opl::gdn::BETA_TOKEN_STRIDE] = Lg;
    a[opl::gdn::S_V] = sv;
    a[opl::gdn::H_K] = hk;
    a[opl::gdn::H_V] = hv;
    a[opl::gdn::N_TOKENS] = T;
    a[opl::gdn::N_SEQS] = ns;
    plan_limits l = LIMITS_U200;
    l.caps = caps;
    l.budget = budget;
    kind_plan k{};
    plan_gated_delta_net(a, opl::gdn::NARGS, l, &k);
    return k;
}
constexpr uint32_t FUSED = CAPS_VU | CAPS_ATT | CAPS_GDN, COMPOSED = CAPS_VU | CAPS_ATT;
}  // namespace gdn_check
}  // namespace plan
// The 27B (S_v 128, H_k 16, H_v 48) on the U200 budget: a decode token is one group of 48 heads over two state tiles, a
// batch keeps the 48 tiles; the token block is 3 · 3 + 2 (16 + 48) 8 lines.
static_assert(plan::gdn_check::probe(plan::gdn_check::FUSED, 60144u, 128u, 16u, 48u, 1u, 1u).aux[2] == 2u &&
                  plan::gdn_check::probe(plan::gdn_check::FUSED, 60144u, 128u, 16u, 48u, 1u, 1u).aux[3] == 48u &&
                  plan::gdn_check::probe(plan::gdn_check::FUSED, 60144u, 128u, 16u, 48u, 1u, 1u).aux[1] == 9u + 1024u &&
                  plan::gdn_check::probe(plan::gdn_check::FUSED, 60144u, 128u, 16u, 48u, 1u, 1u).scratch_lines == 2u * 1024u + 2u * 1033u &&
                  plan::gdn_check::probe(plan::gdn_check::FUSED, 60144u, 128u, 16u, 48u, 78u, 1u).aux[2] == 48u &&
                  plan::gdn_check::probe(plan::gdn_check::FUSED, 60144u, 128u, 16u, 48u, 78u, 1u).aux[3] == 48u &&
                  plan::gdn_check::probe(plan::gdn_check::FUSED, 60144u, 128u, 16u, 48u, 78u, 1u).scratch_lines == 48u * 1024u + 2u * 1033u,
              "plan_gdn.hpp: the 27B plans on the token block pipeline");
// A device without CAPS_GDN, and S_v 16 and 32 on one with it, get the plan of the head by head pipeline (block
// 3 S_v L + 2 (5 L + 2), no state tiles or groups in aux).
static_assert(plan::gdn_check::probe(plan::gdn_check::COMPOSED, 60144u, 128u, 16u, 48u, 1u, 1u).scratch_lines == 3u * 1024u + 2u * 42u &&
                  plan::gdn_check::probe(plan::gdn_check::COMPOSED, 60144u, 128u, 16u, 48u, 1u, 1u).aux[2] == 0u &&
                  plan::gdn_check::probe(plan::gdn_check::FUSED, 60144u, 32u, 2u, 6u, 3u, 1u).aux[2] == 0u &&
                  plan::gdn_check::probe(plan::gdn_check::FUSED, 60144u, 32u, 2u, 6u, 3u, 1u).scratch_lines == 3u * 32u * 2u + 2u * 12u,
              "plan_gdn.hpp: no CAPS_GDN or S_v below 64 plans the head by head block");
// A budget that holds eight tiles of S_v 128 beside the token blocks makes groups of eight heads; one that holds no tile and
// the blocks beside it, or two tiles with one token, takes the head by head pipeline when that block fits half the
// budget, and is refused when it does not.
static_assert(plan::gdn_check::probe(plan::gdn_check::FUSED, 9300u, 128u, 16u, 48u, 5u, 1u).aux[3] == 8u &&
                  plan::gdn_check::probe(plan::gdn_check::FUSED, 9300u, 128u, 16u, 48u, 5u, 1u).aux[2] == 8u &&
                  plan::gdn_check::probe(plan::gdn_check::FUSED, 3376u, 64u, 128u, 128u, 1u, 1u).aux[2] == 0u &&
                  plan::gdn_check::probe(plan::gdn_check::FUSED, 3376u, 64u, 128u, 128u, 1u, 1u).status == opl::RET_OK &&
                  plan::gdn_check::probe(plan::gdn_check::FUSED, 3376u, 128u, 2u, 4u, 1u, 1u).aux[2] == 2u &&
                  plan::gdn_check::probe(plan::gdn_check::FUSED, 3376u, 128u, 16u, 48u, 1u, 1u).status == opl::RET_SCRATCH,
              "plan_gdn.hpp: groups on a small budget, the head by head pipeline when the block does not fit");

}  // namespace tsr
