// tsr/plan_elem.hpp — planners of the element-wise kinds: ADD, MUL, SCALE, GLU, UNARY, CPY, GET_ROWS
// (docs/v2/06-software-verification.md "SW" §7.2, §7.5; W2b P2.6). Freestanding, C++17-compatible,
// constexpr-friendly: the firmware's kind handlers (sw/ctrl/kernels/k_elem.cpp, k_glu.cpp,
// k_rows.cpp) and the host's supports_op (W3b) call the same functions (V2-SW-20).
//
// The planner contract and the declarations are tsr/plan.hpp's ("M7 planning"), which includes this
// file after them; including this file first works too (it includes plan.hpp).
// Owner: the W2b elem work package.
//
// Element-wise kinds (ADD 0x11, MUL 0x12, SCALE 0x13, GLU 0x15, UNARY 0x1B). Every one runs per row
// tile: LD the tile's a rows (and its b part) into one ring block, the VOP(s) in place on the a rows
// (VU §4.5: d = a exactly), ST the rows. A row of n f32 is L = L(n) = ceil(n/16) lines (VU §4.2), a
// tile of t rows holds t L lines of a plus its b part (kind_plan flags >> KP_KIND_LSB = EB_*):
//   EB_NONE     SCALE, UNARY: no b
//   EB_ROWS     b row r for a row r (ADD / MUL with b period p >= rows; GLU's up rows): t L lines,
//               VOP b stride L
//   EB_BCAST    one b row for every a row (p = 1, or a b stride of 0): L lines, VOP b stride 0
//   EB_PERIOD   b rows 0 .. p-1 (1 < p < rows): min(p, t) L lines; tiles are multiples of p and one
//               VOP covers each period block of p rows (VU §9: the VU has strides, not periods)
//   EB_C1       FLAG_B_SCALAR, p = 1: one f32 at b: one line; a single-row op reads it as a one-word
//               c vector (cmode 2); otherwise tile 0 loads the line and SOP.LDS puts it into s12
//               (cmode 1 in every tile; later tiles have no b part)
//   EB_CROWS    FLAG_B_SCALAR, p >= rows: a per-row scalar, word r of the f32 vector at b (e.g.
//               Gemma's inp_scale_rows, SW §7.5): ceil(t/16) lines of the vector from line r0/16,
//               cmode 2; tiles are multiples of 16 rows (a c vector starts on a line, VU §4.3)
//   EB_CPERIOD  FLAG_B_SCALAR, 1 < p < rows: words 0 .. p-1: ceil(min(p, t)/16) lines, cmode 2;
//               tiles are multiples of p, one VOP per period block
// FLAG_B_SCALAR with p > 1 (decision W2b-elem-2): SW §7.2 defines the flag as "b is one f32
// scalar"; with the row period the same words describe a vector of per-row scalars (word r mod p),
// which is SW §7.5's per-row c vector (cmode 2) for Gemma's [1, T] inp_scale_rows; p = 1 is the one
// scalar of SW §7.2. The b stride is not used with the flag.
// Tile size (MUL_MAT's T-tile rule, SW §7.4; decision W2b-elem-3): all rows in one tile when its
// block fits half the planning budget (l.budget = ring_budget, decision P2.4; room for the next
// tile or the neighbour ops' blocks), else the largest aligned row count (p for the period forms,
// 16 for EB_CROWS, else 1) whose block fits half the budget (two tiles in flight: the LD of tile k+1
// overlaps the VOP of tile k), else one aligned unit if its block fits the whole budget (single
// buffer), else RET_SCRATCH. At most 65,535 rows per tile (VOP rows[15:0]). (plan.hpp's plan_rows
// takes one tile up to the whole budget; these kinds keep MUL_MAT's half so that a prefill op's
// LD / VOP / ST overlap: identical for every decode shape, where one tile is far below half.)
// kind_plan: scratch_lines = tile 0's block (the largest), row_lines = L, tile_rows = t, n_tiles,
// flags = KP_ONE_TILE | EB_* << KP_KIND_LSB, aux[0] = the period / alignment p (1 unless EB_PERIOD /
// EB_CPERIOD).
// BAD_RECORD: no VU (l.caps without CAPS_VU, also when unknown: 0); a wrong n_args; n = 0 or
// L(n) > 65,535 (the VOP's 16-bit strides; n <= 1,048,560); rows = 0; dst rows that overlap (rows > 1
// with dst stride < L); ADD / MUL: b period 0 or an unknown flag; GLU variant > 2; UNARY func > 4.
// (Empty ops: llama.cpp emits none; the backend does not either.)
//
// Row copies (CPY 0x1A, GET_ROWS 0x17; SW §7.5: LD rows -> ST rows through one ring block per tile;
// bit copies, no VU op). A row is rl = "row lines" lines. A CPY whose rows are dense on both sides
// (or that has one row) is one run of rows x rl lines (CP_MERGED). A tile is t rows, a block of t rl
// lines, when a row fits half the budget (t = min(rows, half / rl)); else each row moves in chunks of
// half the budget (CP_CHUNKED: one chunk per tile). kind_plan: scratch_lines = the largest block,
// row_lines = rl (CP_MERGED: rows x rl), tile_rows = t (CP_CHUNKED: 1), n_tiles = tiles (CP_CHUNKED:
// rows x chunks per row), flags = KP_ONE_TILE | CP_* << KP_KIND_LSB, aux[0] = rows after merging,
// aux[1] = chunk lines (CP_CHUNKED, else 0), aux[2] = chunks per row (CP_CHUNKED, else 1).
// BAD_RECORD: a wrong n_args; rl = 0; rows / n_ids = 0; dst rows that overlap (more than one row
// with dst stride < rl); a merged CPY beyond beat[27:0]; GET_ROWS: ids not 4-byte aligned or beyond
// the largest preload area (the handler checks them against the list's preload area, kind_ctx::
// pre_ok; the host reader with opl::check_args). GET_ROWS ids are I32 (SW §7.2); an id that is
// negative, or puts its row beyond 2^32 beats or the channel, is a data error the device reports
// (DRAM_RANGE, k_rows.cpp).
#pragma once

#include <stdint.h>

#include "tsr/plan.hpp"

namespace tsr {
namespace plan {

constexpr uint32_t EB_NONE = 0u, EB_ROWS = 1u, EB_BCAST = 2u, EB_PERIOD = 3u, EB_C1 = 4u, EB_CROWS = 5u,
                   EB_CPERIOD = 6u;
constexpr uint32_t EW_N_MAX = VU_STRIDE_MAX * 16u;   // L(n) <= 65,535: the VOP's 16-bit strides (VU §5.2)
constexpr uint32_t CP_MERGED = 1u, CP_CHUNKED = 2u;  // copy flags (kind_plan flags >> KP_KIND_LSB)

// The b part of a tile of t rows (t, L <= 65,535, so t L < 2^32; 32-bit arithmetic, no libgcc call).
constexpr uint32_t ew_b_lines(uint32_t bm, uint32_t L, uint32_t t, uint32_t p) {
    switch (bm) {
        case EB_ROWS:    return t * L;
        case EB_BCAST:   return L;
        case EB_PERIOD:  return (p < t ? p : t) * L;
        case EB_C1:      return 1u;
        case EB_CROWS:   return vu_lines(t);
        case EB_CPERIOD: return vu_lines(p < t ? p : t);
        default:         return 0u;
    }
}
// Lines of a tile's block, saturating at 2^32 - 1 (it is only compared with the budget, < 2^24).
constexpr uint32_t ew_tile_lines(uint32_t bm, uint32_t L, uint32_t t, uint32_t p) {
    const uint32_t a = t * L, s = a + ew_b_lines(bm, L, t, p);
    return s < a ? 0xFFFFFFFFu : s;
}

// Rows per tile (header comment), 0 if not even one aligned unit fits the budget.
constexpr uint32_t ew_rows_per_tile(uint32_t bm, uint32_t L, uint32_t p, uint32_t rows, uint32_t budget) {
    const uint32_t half = budget >> 1;
    if (rows <= VU_ROWS_MAX && ew_tile_lines(bm, L, rows, p) <= half) return rows;
    const uint32_t align = (bm == EB_PERIOD || bm == EB_CPERIOD) ? p : bm == EB_CROWS ? 16u : 1u;
    if (align > VU_ROWS_MAX) return 0u;
    // The largest t with t L + b(t) <= half, rounded down to the alignment (b(t) grows with t for
    // EB_ROWS, 2 t L, and EB_CROWS, t L + t / 16 for t a multiple of 16; else it is fixed).
    uint32_t t = 0;
    if (bm == EB_ROWS) t = half / (2u * L);
    else if (bm == EB_CROWS) t = 16u * (half / (16u * L + 1u));
    else {
        const uint32_t fixed = ew_b_lines(bm, L, align, p);
        t = half >= fixed ? (half - fixed) / L : 0u;
    }
    t = t / align * align;
    const uint32_t cap = VU_ROWS_MAX / align * align;
    if (t > cap) t = cap;
    if (t == 0u) {                                     // one aligned unit, single-buffered
        t = align < rows ? align : rows;
        if (ew_tile_lines(bm, L, t, p) > budget) return 0u;
    }
    return t;
}

TSR_PLAN_HOT constexpr bool ew_shape_ok(uint32_t n, uint32_t rows) { return n - 1u < EW_N_MAX && rows != 0u; }

// The common tail of the element-wise planners.
constexpr bool ew_finish(uint32_t bm, uint32_t n, uint32_t rows, uint32_t dst_stride, uint32_t p, const plan_limits & l,
                         kind_plan * out) {
    if ((l.caps & CAPS_VU) == 0u) return plan_fail(out, opl::RET_BAD_RECORD);   // no VU (VOP, SOP)
    const uint32_t budget = l.budget;
    const uint32_t L = vu_lines(n);
    if (rows > 1u && dst_stride < L) return plan_fail(out, opl::RET_BAD_RECORD);   // output rows overlap
    const uint32_t pe = (bm == EB_PERIOD || bm == EB_CPERIOD) ? p : 1u;
    const uint32_t t = ew_rows_per_tile(bm, L, pe, rows, budget);
    if (t == 0u) return plan_fail(out, opl::RET_SCRATCH);
    out->status = opl::RET_OK;
    out->scratch_lines = ew_tile_lines(bm, L, t, pe);
    out->row_lines = L;
    out->tile_rows = t;
    out->n_tiles = t == rows ? 1u : ceil_div(rows, t);
    out->flags = (out->n_tiles == 1u ? KP_ONE_TILE : 0u) | (bm << KP_KIND_LSB);
    out->aux[0] = pe;
    out->aux[1] = out->aux[2] = out->aux[3] = 0u;
    return true;
}
constexpr uint32_t ew_bmode(const kind_plan & k) { return k.flags >> KP_KIND_LSB; }

constexpr bool ew_plan_addmul(const uint32_t * a, uint32_t n_args, const plan_limits & l, kind_plan * out) {
    namespace e = opl::ew;
    if (n_args != e::NARGS) return plan_fail(out, opl::RET_BAD_RECORD);
    const uint32_t n = a[e::N], rows = a[e::ROWS], p = a[e::B_PERIOD], fl = a[e::FLAGS];
    if (!ew_shape_ok(n, rows) || p == 0u || (fl & ~e::FLAGS_KNOWN) != 0u) return plan_fail(out, opl::RET_BAD_RECORD);
    const uint32_t bm = (fl & e::FLAG_B_SCALAR) ? (p == 1u ? EB_C1 : p >= rows ? EB_CROWS : EB_CPERIOD)
                        : (p == 1u || a[e::B_STRIDE] == 0u) ? EB_BCAST : p >= rows ? EB_ROWS : EB_PERIOD;
    return ew_finish(bm, n, rows, a[e::DST_STRIDE], p, l, out);
}

// ---------------------------------------------------------------------------------------------
// Row copies
// ---------------------------------------------------------------------------------------------
constexpr bool cp_finish(uint32_t rl, uint32_t rows, uint32_t merged, uint32_t budget, kind_plan * out) {
    const uint32_t half = budget >> 1;
    if (half == 0u) return plan_fail(out, opl::RET_SCRATCH);
    out->status = opl::RET_OK;
    out->row_lines = rl;
    out->aux[0] = rows;
    out->aux[3] = 0u;
    uint32_t f = merged ? CP_MERGED : 0u;
    if (rl <= half) {
        const uint32_t t = half / rl < rows ? half / rl : rows;
        out->tile_rows = t;
        out->n_tiles = t == rows ? 1u : ceil_div(rows, t);
        out->scratch_lines = t * rl;
        out->aux[1] = 0u;
        out->aux[2] = 1u;
    } else {
        const uint32_t cpr = ceil_div(rl, half);
        const uint64_t nt = uint64_t(rows) * cpr;
        if (nt > 0xFFFFFFFFu) return plan_fail(out, opl::RET_BAD_RECORD);
        f |= CP_CHUNKED;
        out->tile_rows = 1u;
        out->n_tiles = uint32_t(nt);
        out->scratch_lines = half;
        out->aux[1] = half;
        out->aux[2] = cpr;
    }
    out->flags = (out->n_tiles == 1u ? KP_ONE_TILE : 0u) | (f << KP_KIND_LSB);
    return true;
}

}  // namespace plan

// ADD (0x11), MUL (0x12): opl::ew words.
TSR_PLAN_HOT constexpr bool plan_add(const uint32_t * args, uint32_t n_args, const plan_limits & l, kind_plan * out) {
    return plan::ew_plan_addmul(args, n_args, l, out);
}
TSR_PLAN_HOT constexpr bool plan_mul(const uint32_t * args, uint32_t n_args, const plan_limits & l, kind_plan * out) {
    return plan::ew_plan_addmul(args, n_args, l, out);
}
// SCALE (0x13): opl::scl words.
TSR_PLAN_HOT constexpr bool plan_scale(const uint32_t * args, uint32_t n_args, const plan_limits & l, kind_plan * out) {
    namespace s = opl::scl;
    if (n_args != s::NARGS || !plan::ew_shape_ok(args[s::N], args[s::ROWS])) return plan::plan_fail(out, opl::RET_BAD_RECORD);
    return plan::ew_finish(plan::EB_NONE, args[s::N], args[s::ROWS], args[s::DST_STRIDE], 1u, l, out);
}
// GLU (0x15): opl::glu words; the up rows are operand b (EB_ROWS).
TSR_PLAN_HOT constexpr bool plan_glu(const uint32_t * args, uint32_t n_args, const plan_limits & l, kind_plan * out) {
    namespace g = opl::glu;
    if (n_args != g::NARGS || !plan::ew_shape_ok(args[g::N], args[g::ROWS]) || args[g::VARIANT] > g::VARIANT_MAX)
        return plan::plan_fail(out, opl::RET_BAD_RECORD);
    return plan::ew_finish(plan::EB_ROWS, args[g::N], args[g::ROWS], args[g::DST_STRIDE], 1u, l, out);
}
// UNARY (0x1B): opl::unary words.
TSR_PLAN_HOT constexpr bool plan_unary(const uint32_t * args, uint32_t n_args, const plan_limits & l, kind_plan * out) {
    namespace u = opl::unary;
    if (n_args != u::NARGS || !plan::ew_shape_ok(args[u::N], args[u::ROWS]) || args[u::FUNC] > u::FUNC_MAX)
        return plan::plan_fail(out, opl::RET_BAD_RECORD);
    return plan::ew_finish(plan::EB_NONE, args[u::N], args[u::ROWS], args[u::DST_STRIDE], 1u, l, out);
}
// CPY (0x1A): opl::cpy words.
TSR_PLAN_HOT constexpr bool plan_cpy(const uint32_t * args, uint32_t n_args, const plan_limits & l, kind_plan * out) {
    namespace c = opl::cpy;
    if (n_args != c::NARGS) return plan::plan_fail(out, opl::RET_BAD_RECORD);
    const uint32_t rl = args[c::ROW_LINES], rows = args[c::ROWS];
    if (rl == 0u || rows == 0u || (rows > 1u && args[c::DST_STRIDE] < rl)) return plan::plan_fail(out, opl::RET_BAD_RECORD);
    if (rows == 1u || (args[c::SRC_STRIDE] == rl && args[c::DST_STRIDE] == rl)) {
        const uint64_t tot = uint64_t(rows) * rl;
        if (tot > 0x0FFFFFFFu) return plan::plan_fail(out, opl::RET_BAD_RECORD);   // beyond beat[27:0]
        return plan::cp_finish(uint32_t(tot), 1u, 1u, l.budget, out);
    }
    return plan::cp_finish(rl, rows, 0u, l.budget, out);
}
// GET_ROWS (0x17): opl::getrows words.
TSR_PLAN_HOT constexpr bool plan_get_rows(const uint32_t * args, uint32_t n_args, const plan_limits & l, kind_plan * out) {
    namespace g = opl::getrows;
    if (n_args != g::NARGS) return plan::plan_fail(out, opl::RET_BAD_RECORD);
    const uint32_t rl = args[g::ROW_LINES], n = args[g::N_IDS], ids = args[g::IDS];
    if (rl == 0u || n == 0u || (ids & (opl::PRE_ID_ALIGN - 1u)) != 0u ||
        uint64_t(ids) + uint64_t(opl::PRE_ID_BYTES) * n > uint64_t(plan::PRE_MAX_LINES) * plan::LINE_BYTES ||
        (n > 1u && args[g::DST_STRIDE] < rl))
        return plan::plan_fail(out, opl::RET_BAD_RECORD);
    return plan::cp_finish(rl, n, 0u, l.budget, out);
}

// The decode shapes of the target models plan as one tile on the U200 (the fast paths' case), the
// prefill shapes as the tile rules above say (fw_kernels_elem_host checks the rules on a grid).
namespace plan {
namespace elem_check {
constexpr kind_plan add(uint32_t n, uint32_t rows, uint32_t p, uint32_t fl) {
    const uint32_t a[opl::ew::NARGS] = {0x10000000u, vu_lines(n), 0u, vu_lines(n), 0x20000000u, vu_lines(n), n, rows, p, fl};
    kind_plan k{};
    plan_add(a, opl::ew::NARGS, LIMITS_U200, &k);
    return k;
}
constexpr kind_plan glu(uint32_t n, uint32_t rows) {
    const uint32_t a[opl::glu::NARGS] = {0x10000000u, vu_lines(n), 0u, vu_lines(n), 0x20000000u, vu_lines(n), n, rows, 1u};
    kind_plan k{};
    plan_glu(a, opl::glu::NARGS, LIMITS_U200, &k);
    return k;
}
constexpr kind_plan unary(uint32_t n, uint32_t rows) {
    const uint32_t a[opl::unary::NARGS] = {0x10000000u, vu_lines(n), 0u, vu_lines(n), n, rows, 0u};
    kind_plan k{};
    plan_unary(a, opl::unary::NARGS, LIMITS_U200, &k);
    return k;
}
}  // namespace elem_check
}  // namespace plan
static_assert(plan::elem_check::add(896, 1, 1, 0).n_tiles == 1u && plan::elem_check::add(896, 1, 1, 0).scratch_lines == 112u &&
                  plan::elem_check::add(5120, 1, 1, 0).n_tiles == 1u && plan::elem_check::add(3840, 1, 1, 1).scratch_lines == 241u &&
                  plan::elem_check::glu(27648, 1).n_tiles == 1u && plan::elem_check::glu(15360, 1).scratch_lines == 1920u &&
                  plan::elem_check::unary(262144, 1).n_tiles == 1u,
              "plan_elem.hpp: the decode shapes (Qwen2.5 0.5B / 32B, Gemma 4 12B, its 262,144 logits) are one tile");
static_assert(plan::elem_check::add(5120, 512, 512, 0).tile_rows == 46u && plan::elem_check::add(5120, 512, 512, 0).n_tiles == 12u &&
                  plan::elem_check::add(3840, 512, 512, 1).tile_rows == 112u &&
                  plan::elem_check::add(3840, 512, 512, 1).flags >> plan::KP_KIND_LSB == plan::EB_CROWS &&
                  plan::elem_check::add(64, 14 * 37, 14, 0).tile_rows == 14u * 37u &&
                  plan::elem_check::add(896, 1, 1, 0x2u).status == opl::RET_BAD_RECORD,
              "plan_elem.hpp: prefill tiles (32B residual: 46-row tiles in the U200 budget of 60,144 lines; 12B inp_scale_rows: 112 rows, a multiple of 16)");

}  // namespace tsr
