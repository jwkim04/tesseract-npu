// tsr/plan_norm.hpp — planners of RMS_NORM (0x14) and ROPE (0x16) (docs/v2/06-software-verification.md
// "SW" §7.2, §7.5; W2b P2.6). Freestanding, C++17-compatible, constexpr-friendly: the firmware's kind
// handlers (sw/ctrl/kernels/k_norm.cpp, k_rope.cpp) and the host's supports_op (W3b) call the same
// functions (V2-SW-20).
//
// The planner contract and the declarations are tsr/plan.hpp's ("M7 planning"), which includes this
// file after them; including this file first works too (it includes plan.hpp).
// Owner: the W2b normrope work package.
//
// The planners plan against l.budget (decision P2.4) and need the VU (l.caps CAPS_VU, else
// BAD_RECORD). Besides plan_rms_norm / plan_rope (the contract's form, on argument words) the
// field-level functions below are what the decode fast paths evaluate on the words they read in
// place (no array copy): the same rules, shifts / adds / compares only for a single-tile op.
//
// RMS_NORM (SW §7.5 row; VU §6.2, §8.1). Rows are L(n) = ceil(n / 16) lines (every row of a tensor
// starts on a line, SW §7.2). A row tile of R rows takes one ring block [x rows | weight | vector]:
//   x       R L lines (the rows, dense; the VOP writes its result in place, VU §4.5);
//   weight  none, L lines (weight period 1: one row, VOP b stride 0), or R L lines (period p > 1:
//           the tile's weight rows, contiguous because a tile never crosses a period block);
//   vector  ceil(R / 16) lines of RED's per-row results when R > 1 (one row: RED TO_S, s register).
// Tiles (plan.hpp plan_rows' rule, with the block's fixed weight row and vector lines): every row in
// one tile when block_lines(rows) <= budget (rows <= 65,535, rows <= p for p > 1, L <= 65,535 unless
// one row); else R = the largest count with block_lines(R) <= budget / 2 (two tiles live: the next
// tile's LD overlaps this tile's VU work), at most 65,535, at most p for p > 1 (a tile never crosses a
// period block: rows [k p, (k + 1) p) are tiled on their own), 1 when L > 65,535 (the stride fields
// are 16 bits); a single buffer (R = 1 in the budget) when even one row exceeds budget / 2.
// Argument rules (BAD_RECORD): n_args 10; n in [1, 2^24) (the VU m field); rows >= 1; a weight's
// row period >= 1 (the word is ignored without a weight); dst stride >= L when rows > 1 (output rows
// may not overlap; the source stride is free: rows are loaded one LD per row unless dense). The
// weight tensor's rows are L lines apart in DRAM (dense; W2b decision: SW §7.2 has no weight
// stride word), row r mod p for output row r.
// kind_plan: row_lines = L, tile_rows = R, n_tiles, scratch_lines = block_lines(R), flags
// KP_ONE_TILE | RMS_KP_WEIGHT (a weight) | RMS_KP_WROWS (period > 1: per-row weight, b stride L),
// aux[0] = weight lines of a full tile.
//
// ROPE (SW §7.5 row; VU §5.4, §6.7). One token is H = n_heads head_dim / 16 lines (heads contiguous
// within a token: the backend's supports_op checks nb1 = 4 head_dim); the OPL strides are lines
// between tokens. A token tile of Tt tokens takes one block [x tokens | table]: Tt H lines (ROPE works
// in place: same start and token stride) plus ceil(n_dims / 32) lines of the freq-factor / A table
// when there is one (loaded per tile; every decode op is one tile). Tt: every token when they fit
// the budget (n_tokens <= 65,535); else a multiple of 16 (a tile's positions must start on a line of
// the preload area: the VU reads the I32 position vector as an SPM operand) from budget / 2, else
// from the whole budget (single buffer), else RET_SCRATCH. Argument rules (BAD_RECORD): n_args 14;
// mode bits other than NEOX / ATAB; head_dim a multiple of 16 in [16, 1024]; n_dims in [1, head_dim],
// a multiple of 32 (NEOX) or 16 (NORM); ATAB needs a table; n_heads in [1, 65,535]; n_tokens >= 1;
// H <= 65,535 and dst stride >= H when n_tokens > 1; the position byte offset a multiple of 64 (the
// handler also checks that the n_tokens positions lie inside the list's preload area). With ATAB
// the TS and freq_scale words are ignored (the host folded them into the A table, VU §6.7);
// freq_scale = 1.0f (0x3F800000) means no FS multiply (exact: FM(1, phi) = phi for every VU angle).
// kind_plan: row_lines = H, tile_rows = Tt (tokens), n_tiles, scratch_lines = Tt H + table lines,
// aux[0] = table lines.
#pragma once

#include <stdint.h>

#include "tsr/plan.hpp"

namespace tsr {
namespace plan {
namespace norm {

// ---------------------------------------------------------------------------------------------
// RMS_NORM
// ---------------------------------------------------------------------------------------------
constexpr uint32_t RMS_KP_WEIGHT = 1u << KP_KIND_LSB;          // fused weight
constexpr uint32_t RMS_KP_WROWS  = 1u << (KP_KIND_LSB + 1u);   // weight period > 1 (b stride L)

// Lines of a tile block of R rows (header comment); 64-bit (R < 2^16, L < 2^20).
TSR_PLAN_HOT constexpr uint64_t rms_block_lines(uint32_t L, uint32_t R, bool has_w, uint32_t period) {
    const uint64_t w = !has_w ? 0u : period == 1u ? uint64_t(L) : uint64_t(R) * L;
    return uint64_t(R) * L + w + (R > 1u ? vu_lines(R) : 0u);
}

// The argument rules (no budget).
TSR_PLAN_HOT constexpr uint32_t rms_check(uint32_t n, uint32_t rows, uint32_t dst_stride, bool has_w, uint32_t period) {
    if (n == 0u || n > VU_M_MAX || rows == 0u || (has_w && period == 0u)) return opl::RET_BAD_RECORD;
    if (rows > 1u && dst_stride < vu_lines(n)) return opl::RET_BAD_RECORD;
    return opl::RET_OK;
}

// Whether every row fits one tile (the fast path's question, no divide).
TSR_PLAN_HOT constexpr bool rms_one_tile(uint32_t budget, uint32_t L, uint32_t rows, bool has_w, uint32_t period) {
    if (rows > VU_ROWS_MAX || (rows > 1u && L > VU_STRIDE_MAX)) return false;
    if (has_w && period > 1u && rows > period) return false;
    return rms_block_lines(L, rows, has_w, period) <= budget;
}

struct rms_tiling {
    uint32_t L;           // lines per row
    uint32_t R;           // rows per tile (a period block's or the op's last tile may be shorter)
    uint32_t n_tiles;
    uint32_t lines;       // block_lines(R)
};

constexpr uint32_t rms_plan(uint32_t budget, uint32_t n, uint32_t rows, uint32_t dst_stride, bool has_w,
                            uint32_t period, rms_tiling * out) {
    const uint32_t chk = rms_check(n, rows, dst_stride, has_w, period);
    if (chk != opl::RET_OK) return chk;
    if (budget > SPM_LINE_MAX) budget = SPM_LINE_MAX;     // ring lines are 24-bit SPM addresses
    const uint32_t L = vu_lines(n);
    if (!has_w) period = 1u;
    uint32_t R = 0;
    if (rms_one_tile(budget, L, rows, has_w, period)) {
        R = rows;
    } else {
        uint32_t cap = rows < VU_ROWS_MAX ? rows : VU_ROWS_MAX;
        if (period > 1u && period < cap) cap = period;
        if (L > VU_STRIDE_MAX) cap = 1u;
        const uint32_t half = budget >> 1;
        if (rms_block_lines(L, 1u, has_w, period) <= half) {
            // From the linear bound (k L + 1/16 lines per row, f fixed), then down to the largest fit
            // (the vector's ceil makes block_lines not exactly linear). 32-bit: half - f <= half <
            // 2^24 (SPM lines are 24-bit fields), so 16 (half - f) + 15 < 2^28, 16 k L + 1 < 2^30.
            const uint32_t k = (has_w && period != 1u) ? 2u : 1u, f = (has_w && period == 1u) ? L : 0u;
            uint32_t r = (16u * (half - f) + 15u) / (16u * k * L + 1u) + 1u;
            if (r > cap) r = cap;
            while (r > 1u && rms_block_lines(L, r, has_w, period) > half) --r;
            R = r;
        } else if (rms_block_lines(L, 1u, has_w, period) <= budget) {
            R = 1u;                                     // single buffer
        } else {
            return opl::RET_SCRATCH;
        }
    }
    if (out != nullptr) {
        out->L = L;
        out->R = R;
        if (R == rows) {
            out->n_tiles = 1u;
        } else if (period <= 1u) {
            out->n_tiles = ceil_div(rows, R);
        } else {                                        // tiles per period block, then the last block
            const uint32_t fb = rows / period, rem = rows - fb * period;
            out->n_tiles = fb * ceil_div(period, R) + (rem != 0u ? ceil_div(rem, R) : 0u);
        }
        out->lines = uint32_t(rms_block_lines(L, R, has_w, period));
    }
    return opl::RET_OK;
}

// ---------------------------------------------------------------------------------------------
// ROPE
// ---------------------------------------------------------------------------------------------
TSR_PLAN_HOT constexpr uint32_t rope_check(uint32_t mode, bool has_table, uint32_t head_dim, uint32_t n_dims,
                                           uint32_t n_heads, uint32_t n_tokens, uint32_t dst_stride, uint32_t pos_off) {
    const bool neox = (mode & opl::rope::MODE_NEOX) != 0u;
    if ((mode & ~opl::rope::MODE_KNOWN) != 0u || head_dim == 0u || (head_dim & 15u) != 0u || head_dim > 1024u ||
        n_dims == 0u || n_dims > head_dim || (n_dims & (neox ? 31u : 15u)) != 0u ||
        ((mode & opl::rope::MODE_ATAB) != 0u && !has_table) || n_heads == 0u || n_heads > VU_ROWS_MAX ||
        n_tokens == 0u || (pos_off & 63u) != 0u)
        return opl::RET_BAD_RECORD;
    const uint32_t H = (n_heads * head_dim) >> 4;    // < 2^16 2^10 / 16: no overflow
    if (n_tokens > 1u && (H > VU_STRIDE_MAX || dst_stride < H)) return opl::RET_BAD_RECORD;
    return opl::RET_OK;
}
TSR_PLAN_HOT constexpr uint32_t rope_tab_lines(bool has_table, uint32_t n_dims) {
    return has_table ? vu_lines(n_dims >> 1) : 0u;
}
// Whether every token fits one tile (the fast path's question; H and the table from rope_check'd fields).
TSR_PLAN_HOT constexpr bool rope_one_tile(uint32_t budget, uint32_t H, uint32_t tab, uint32_t n_tokens) {
    return n_tokens <= VU_ROWS_MAX && uint64_t(n_tokens) * H + tab <= budget;
}

struct rope_tiling {
    uint32_t H;            // lines per token
    uint32_t tab;          // table lines (0: none)
    uint32_t Tt;           // tokens per tile (n_tokens, or a multiple of 16)
    uint32_t n_tiles;
    uint32_t lines;        // Tt H + tab
};

constexpr uint32_t rope_plan(uint32_t budget, uint32_t mode, bool has_table, uint32_t head_dim, uint32_t n_dims,
                             uint32_t n_heads, uint32_t n_tokens, uint32_t dst_stride, uint32_t pos_off,
                             rope_tiling * out) {
    const uint32_t chk = rope_check(mode, has_table, head_dim, n_dims, n_heads, n_tokens, dst_stride, pos_off);
    if (chk != opl::RET_OK) return chk;
    const uint32_t H = (n_heads * head_dim) >> 4, tab = rope_tab_lines(has_table, n_dims);
    uint32_t tt = 0;
    if (rope_one_tile(budget, H, tab, n_tokens)) {
        tt = n_tokens;
    } else {
        const uint32_t bs[2] = {budget >> 1, budget};   // double buffer, else single
        for (uint32_t i = 0; i < 2u && tt == 0u; ++i) {
            if (bs[i] < tab + H) continue;
            uint32_t fit = (bs[i] - tab) / H;            // tokens that fit
            if (fit > VU_ROWS_MAX) fit = VU_ROWS_MAX;
            tt = fit & ~15u;
        }
        if (tt == 0u) return opl::RET_SCRATCH;
    }
    if (out != nullptr) {
        out->H = H;
        out->tab = tab;
        out->Tt = tt;
        out->n_tiles = tt == n_tokens ? 1u : ceil_div(n_tokens, tt);
        out->lines = tt * H + tab;
    }
    return opl::RET_OK;
}

}  // namespace norm
}  // namespace plan

// ---------------------------------------------------------------------------------------------
// The contract's planners (plan.hpp "M7 planning")
// ---------------------------------------------------------------------------------------------
// RMS_NORM (0x14).
TSR_PLAN_HOT constexpr bool plan_rms_norm(const uint32_t * a, uint32_t n_args, const plan_limits & l, kind_plan * out) {
    namespace r = opl::rms;
    if (n_args != r::NARGS || (l.caps & plan::CAPS_VU) == 0u) return plan::plan_fail(out, opl::RET_BAD_RECORD);
    const bool has_w = a[r::W] != opl::HANDLE_NONE;
    const uint32_t p = has_w ? a[r::W_PERIOD] : 1u;
    plan::norm::rms_tiling t{};
    const uint32_t st = plan::norm::rms_plan(l.budget, a[r::N], a[r::ROWS], a[r::DST_STRIDE], has_w, p, &t);
    if (st != opl::RET_OK) return plan::plan_fail(out, st);
    out->status = opl::RET_OK;
    out->scratch_lines = t.lines;
    out->row_lines = t.L;
    out->tile_rows = t.R;
    out->n_tiles = t.n_tiles;
    out->flags = (t.n_tiles == 1u ? plan::KP_ONE_TILE : 0u) | (has_w ? plan::norm::RMS_KP_WEIGHT : 0u) |
                 (has_w && p > 1u ? plan::norm::RMS_KP_WROWS : 0u);
    out->aux[0] = !has_w ? 0u : p == 1u ? t.L : t.R * t.L;
    out->aux[1] = out->aux[2] = out->aux[3] = 0u;
    return true;
}

// ROPE (0x16).
TSR_PLAN_HOT constexpr bool plan_rope(const uint32_t * a, uint32_t n_args, const plan_limits & l, kind_plan * out) {
    namespace r = opl::rope;
    if (n_args != r::NARGS || (l.caps & plan::CAPS_VU) == 0u) return plan::plan_fail(out, opl::RET_BAD_RECORD);
    plan::norm::rope_tiling t{};
    const uint32_t st = plan::norm::rope_plan(l.budget, a[r::MODE], a[r::TABLE] != opl::HANDLE_NONE, a[r::HEAD_DIM],
                                              a[r::N_DIMS], a[r::N_HEADS], a[r::N_TOKENS], a[r::DST_STRIDE], a[r::POS], &t);
    if (st != opl::RET_OK) return plan::plan_fail(out, st);
    out->status = opl::RET_OK;
    out->scratch_lines = t.lines;
    out->row_lines = t.H;
    out->tile_rows = t.Tt;
    out->n_tiles = t.n_tiles;
    out->flags = t.n_tiles == 1u ? plan::KP_ONE_TILE : 0u;
    out->aux[0] = t.tab;
    out->aux[1] = out->aux[2] = out->aux[3] = 0u;
    return true;
}

// ---------------------------------------------------------------------------------------------
// Static checks (U200: budget 65,536 - 16 - 4,096 - 256 = 61,168 lines)
// ---------------------------------------------------------------------------------------------
namespace plan {
namespace norm {
namespace detail {
constexpr uint32_t B_U200 = 61168u;
constexpr uint32_t rms_R(uint32_t n, uint32_t rows, bool w, uint32_t p) {
    rms_tiling t{};
    return rms_plan(B_U200, n, rows, vu_lines(n), w, p, &t) == opl::RET_OK ? t.R : 0u;
}
// R is the largest legal tile: one tile, or block_lines(R + 1) > budget / 2 or a cap was reached.
constexpr bool rms_R_is_max(uint32_t n, uint32_t rows, bool w, uint32_t p) {
    const uint32_t R = rms_R(n, rows, w, p), L = vu_lines(n);
    if (R == 0u) return false;
    if (R == rows) return rms_block_lines(L, R, w, p) <= B_U200;
    if (rms_block_lines(L, R, w, p) > B_U200 / 2u) return R == 1u;
    return R == VU_ROWS_MAX || (p > 1u && w && R == p) || rms_block_lines(L, R + 1u, w, p) > B_U200 / 2u;
}
constexpr uint32_t rope_T(uint32_t hd, uint32_t nh, uint32_t T, bool tab) {
    rope_tiling t{};
    return rope_plan(B_U200, opl::rope::MODE_NEOX, tab, hd, hd, nh, T, (nh * hd) >> 4, 0u, &t) == opl::RET_OK ? t.Tt : 0u;
}
constexpr bool plans(uint32_t kind, const uint32_t * a) {
    kind_plan kp{};
    return kind == opl::K_ROPE ? plan_rope(a, opl::rope::NARGS, LIMITS_U200, &kp)
                               : plan_rms_norm(a, opl::rms::NARGS, LIMITS_U200, &kp);
}
constexpr uint32_t QWEN_NORM[10] = {0, 56, 0x100, 56, 896, 1, 0x358637BDu, 0x3A924925u, 0x200, 1};
constexpr uint32_t QWEN_ROPE[14] = {0, 56, 0x100, 56, 0, opl::HANDLE_NONE, 0x3F2B5A7Bu, 0x3F800000u, 0x3F800000u,
                                    64, 64, 14, 1, opl::rope::MODE_NEOX};
}  // namespace detail
static_assert(detail::rms_R(896u, 1u, true, 1u) == 1u && detail::rms_R(5120u, 512u, true, 1u) == 94u &&
                  detail::rms_R(5120u, 190u, true, 1u) == 190u && detail::rms_R_is_max(5120u, 512u, true, 1u) &&
                  detail::rms_R_is_max(896u, 4096u, false, 1u) && detail::rms_R_is_max(256u, 100000u, true, 7u) &&
                  detail::rms_R(256u, 32u, true, 1u) == 32u && detail::rms_R(16u * 40000u, 3u, true, 1u) == 0u &&
                  detail::rms_R(16u * 20000u, 3u, true, 1u) == 1u && detail::rms_R(16u * 70000u, 1u, false, 1u) == 0u,
              "plan_norm: RMS_NORM row tiles (one tile when it fits; else tiles of at most half the budget)");
static_assert(rms_check(896u, 2u, 55u, true, 1u) == opl::RET_BAD_RECORD && rms_check(0u, 1u, 0u, false, 0u) != 0u &&
                  rms_check(896u, 1u, 0u, true, 0u) == opl::RET_BAD_RECORD && rms_check(896u, 1u, 0u, false, 0u) == 0u,
              "plan_norm: RMS_NORM argument rules");
static_assert(detail::rope_T(64u, 14u, 1u, false) == 1u && detail::rope_T(128u, 40u, 512u, false) == 80u &&
                  detail::rope_T(512u, 16u, 7u, true) == 7u && detail::rope_T(128u, 48u, 32768u, false) == 64u &&
                  detail::rope_T(128u, 40u, 190u, false) == 190u,
              "plan_norm: ROPE token tiles");
static_assert(detail::plans(opl::K_RMS_NORM, detail::QWEN_NORM) && detail::plans(opl::K_ROPE, detail::QWEN_ROPE),
              "plan_norm: Qwen2.5-0.5B decode RMS_NORM / ROPE records plan on the U200");
}  // namespace norm
}  // namespace plan

}  // namespace tsr
