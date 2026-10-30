// tsr/plan.hpp — planning code shared by the firmware and the host backend (docs/v2/
// 06-software-verification.md "SW" §6.7 last paragraph, §7.3, §7.4, §7.5; V2-SW-20).
//
// Freestanding and C++17-compatible: no libstdc++, no heap, no float. The firmware's kind handlers
// plan their tiles with it, and the host's supports_op asks the same functions, so the host
// rejects exactly what the firmware cannot plan (V2-SW-20: one source, true by construction).
//
// Contents:
//   - device parameters that planning depends on (dev_cfg: SPM lines, T_MAX, K_MAX), read by the
//     executor from CFG0 / CFG1 / CFG2 through the core window, by the host from the CSRs;
//   - the executor's SPM layout (SW §7.3): argument range, OPL area, preload area, scratch ring,
//     and the ring's planning budget;
//   - format geometry: QA_LINES (QA kinds A32 / A256), ROW_BEATS (repacked weight rows), f32 lines;
//   - MUL_MAT tiling (v1's tsr_mm_lines_per_token / tsr_mm_shape_ok / tsr_mm_plan_tiling, tag
//     v1-m4:sw/ggml-tesseract/tsr-lower.cpp, constraints spec "supports_op (v1)" and "lowering
//     (v1)"), with the T-tile chosen for the ring budget instead of the whole SPM;
//   - the per-kind plan (plan_op / check_op): RET_OK, RET_SCRATCH (cannot be planned in the ring)
//     or RET_BAD_RECORD (arguments the device cannot run), and the scratch lines per tile. M7 kinds
//     are declared stubs (W2b).
//
// Planning budget (decision P2.4): every plan uses ring_budget(cfg) = SPM lines - 16 (argument
// range) - 4096 (OPL area) - 256 (largest preload area), not the ring of the actual list. The
// actual ring is never smaller, so a plan made by the host for any op list is valid in the firmware,
// and the tiles (hence the pushed op stream) do not depend on the list's preload size.
#pragma once

#include <stdint.h>

#include "tsr/opl.hpp"

namespace tsr {
namespace plan {

constexpr uint32_t LINE_BYTES  = 64u;
constexpr uint32_t T_MAX_ARCH  = 8u;        // spec 구성 파라미터: architecture bound of T_MAX
constexpr uint32_t K_FIELD_MAX = 0xFFFFu;   // MVM w5 K[15:0]
constexpr uint32_t T_FIELD_MAX = 15u;       // MVM w5 T[19:16]
constexpr uint32_t FMT_Q8_0 = 0u, FMT_Q4_K = 1u;   // spec ISA weight formats (mmio.hpp FMT_*)
constexpr uint32_t QA_A32 = 0u, QA_A256 = 1u;      // QACT kinds (mmio.hpp QA_*)

// ---------------------------------------------------------------------------------------------
// Device parameters
// ---------------------------------------------------------------------------------------------
struct dev_cfg {
    uint32_t spm_lines;   // CFG1: SPM_BYTES / 64 (U200: 65,536)
    uint32_t t_max;       // CFG0.T_MAX (U200 k4: 8, k1: 1)
    uint32_t k_max;       // CFG2.K_MAX_256 x 256 (U200: 32,768)
};
constexpr dev_cfg U200_K4 = {65536u, 8u, 32768u};

// ---------------------------------------------------------------------------------------------
// Executor SPM layout (SW §7.3; L = spm_lines):
//   [L - 16, L)                 runtime argument range (SW §6.3 R5)
//   [L - 16 - 4096, L - 16)     OPL area (256 KiB); a larger list streams in two 128 KiB halves
//   [opl_line - pre_lines, opl_line)  preload area (header pre_lines <= 256)
//   [0, pre_line)               scratch ring (spm_ring)
// ---------------------------------------------------------------------------------------------
constexpr uint32_t ARG_LINES      = 16u;
constexpr uint32_t OPL_AREA_LINES = 4096u;
constexpr uint32_t OPL_HALF_LINES = 2048u;
constexpr uint32_t PRE_MAX_LINES  = opl::PRE_MAX_LINES;

struct exec_layout {
    uint32_t arg_line;     // first line of the argument range
    uint32_t opl_line;     // first line of the OPL area
    uint32_t pre_line;     // first line of the preload area
    uint32_t pre_lines;
    uint32_t ring_line;    // 0
    uint32_t ring_lines;   // = pre_line
};
// Layout for a list with `pre_lines` preload lines; false if the SPM is too small (the ring
// would be smaller than the planning budget's minimum of one line) or pre_lines > 256.
constexpr bool layout(uint32_t spm_lines, uint32_t pre_lines, exec_layout * out) {
    if (pre_lines > PRE_MAX_LINES || spm_lines < ARG_LINES + OPL_AREA_LINES + PRE_MAX_LINES + 1u) return false;
    exec_layout l{};
    l.arg_line = spm_lines - ARG_LINES;
    l.opl_line = l.arg_line - OPL_AREA_LINES;
    l.pre_line = l.opl_line - pre_lines;
    l.pre_lines = pre_lines;
    l.ring_line = 0u;
    l.ring_lines = l.pre_line;
    *out = l;
    return true;
}
// The planning budget: the ring of a list with the largest preload area (0 if the SPM is too
// small for the executor).
constexpr uint32_t ring_budget(const dev_cfg & c) {
    return c.spm_lines >= ARG_LINES + OPL_AREA_LINES + PRE_MAX_LINES + 1u
               ? c.spm_lines - ARG_LINES - OPL_AREA_LINES - PRE_MAX_LINES
               : 0u;
}

// ---------------------------------------------------------------------------------------------
// Format geometry (spec "양자화 활성값(QA) 레이아웃", "가중치 레이아웃"; tsr_ref.c is the reference)
// ---------------------------------------------------------------------------------------------
constexpr uint32_t ceil_div(uint32_t a, uint32_t b) { return a / b + (a % b != 0u ? 1u : 0u); }
constexpr bool     fmt_valid(uint32_t fmt) { return fmt == FMT_Q8_0 || fmt == FMT_Q4_K; }
constexpr uint32_t fmt_block(uint32_t fmt) { return fmt == FMT_Q8_0 ? 32u : fmt == FMT_Q4_K ? 256u : 0u; }
constexpr uint32_t qa_kind_for_fmt(uint32_t fmt) { return fmt == FMT_Q8_0 ? QA_A32 : QA_A256; }
constexpr uint32_t qa_block(uint32_t kind) { return kind == QA_A32 ? 32u : kind == QA_A256 ? 256u : 0u; }
// QA_LINES(kind, K) = QS lines + D lines + BS lines (A256 only); 0 for an invalid kind or a K
// that is not a multiple of the block.
constexpr uint32_t qa_lines(uint32_t kind, uint32_t K) {
    const uint32_t b = qa_block(kind);
    if (b == 0u || K % b != 0u) return 0u;
    const uint32_t qs = ceil_div(K, LINE_BYTES);
    const uint32_t d = ceil_div(4u * (K / b), LINE_BYTES);
    const uint32_t bs = kind == QA_A256 ? ceil_div(2u * (K / 16u), LINE_BYTES) : 0u;
    return qs + d + bs;
}
// ROW_BEATS(fmt, K): beats of one repacked weight row (chunks of 32 Q8_0 blocks / 4 Q4_K
// superblocks, each a meta beat plus its quant beats); 0 if invalid.
constexpr uint32_t row_beats(uint32_t fmt, uint32_t K) {
    const uint32_t b = fmt_block(fmt);
    if (b == 0u || K % b != 0u) return 0u;
    const uint32_t S = fmt == FMT_Q8_0 ? 32u : 4u;        // blocks per chunk
    const uint32_t qb = fmt == FMT_Q8_0 ? 32u : 128u;     // quant bytes per block
    const uint32_t nb = K / b, full = nb / S, rem = nb % S;
    return full * (1u + ceil_div(S * qb, LINE_BYTES)) + (rem ? 1u + ceil_div(rem * qb, LINE_BYTES) : 0u);
}
constexpr uint32_t f32_lines(uint32_t n) { return ceil_div(4u * n, LINE_BYTES); }

// ---------------------------------------------------------------------------------------------
// MUL_MAT (SW §7.4): per T tile LD x, (LD addend into y), QACT, MVM (ACC with an addend or
// ACC_DST), ST y. Scratch per tile: x (t_tile x x_lines), QA (t_tile x qa_lines), y (t_tile x
// y_lines); the kernel double-buffers tiles in the ring.
// ---------------------------------------------------------------------------------------------
constexpr uint32_t mm_t_max(const dev_cfg & c) {
    uint32_t t = c.t_max < T_MAX_ARCH ? c.t_max : T_MAX_ARCH;
    return t < T_FIELD_MAX ? t : T_FIELD_MAX;
}
// Lines one token needs (x + QA + y); 0 if fmt / K cannot be lowered (v1: K % 32 == 0).
constexpr uint32_t mm_lines_per_token(uint32_t fmt, uint32_t N, uint32_t K) {
    if (!fmt_valid(fmt) || K == 0u || K % 32u != 0u || K % fmt_block(fmt) != 0u || N == 0u) return 0u;
    const uint32_t q = qa_lines(qa_kind_for_fmt(fmt), K);
    const uint64_t s = uint64_t(K / 16u) + q + ceil_div(N, 16u);
    return s > 0xFFFFFFFFull ? 0u : uint32_t(s);
}

struct mm_tiling {
    uint32_t kind;       // QA kind for fmt (Q8_0 -> A32, Q4_K -> A256)
    uint32_t x_lines;    // one f32 x row: 4K / 64 (exact: K % 32 == 0)
    uint32_t qa_lines;   // QA_LINES(kind, K)
    uint32_t y_lines;    // ceil(4N / 64): MVM dst_stride and the per-token ST length
    uint32_t lpt;        // x_lines + qa_lines + y_lines
    uint32_t t_tile;     // tokens per tile: 1 <= t_tile <= min(T, T_MAX)
    uint32_t n_tiles;    // ceil(T / t_tile)
};

// Argument checks (RET_BAD_RECORD: the device cannot run it at all) then the ring check
// (RET_SCRATCH: one token does not fit the budget). T-tile: as many tokens as T_MAX allows while
// two tiles fit the budget (double buffering), else 1 (a single buffer; still plannable while one
// token fits). T itself never makes a shape unplannable (spec "supports_op (v1)": no condition on
// T), so the host checks T = 1.
constexpr uint32_t mm_plan(const dev_cfg & c, uint32_t fmt, uint32_t N, uint32_t K, uint32_t T, mm_tiling * out) {
    const uint32_t kmax = c.k_max < K_FIELD_MAX ? c.k_max : K_FIELD_MAX;
    if (!fmt_valid(fmt) || N == 0u || T == 0u || K < 32u || K % 32u != 0u || K % fmt_block(fmt) != 0u || K > kmax ||
        mm_t_max(c) == 0u)
        return opl::RET_BAD_RECORD;
    const uint32_t lpt = mm_lines_per_token(fmt, N, K);
    const uint32_t budget = ring_budget(c);
    if (lpt == 0u || lpt > budget) return opl::RET_SCRATCH;
    mm_tiling t{};
    t.kind = qa_kind_for_fmt(fmt);
    t.x_lines = K / 16u;
    t.qa_lines = qa_lines(t.kind, K);
    t.y_lines = ceil_div(N, 16u);
    t.lpt = lpt;
    uint32_t tt = (budget / 2u) / lpt;
    if (tt == 0u) tt = 1u;
    if (tt > mm_t_max(c)) tt = mm_t_max(c);
    if (tt > T) tt = T;
    t.t_tile = tt;
    t.n_tiles = ceil_div(T, tt);
    if (out != nullptr) *out = t;
    return opl::RET_OK;
}
// supports_op's MUL_MAT question (independent of T).
constexpr bool mm_supported(const dev_cfg & c, uint32_t fmt, uint32_t N, uint32_t K) {
    return mm_plan(c, fmt, N, K, 1u, nullptr) == opl::RET_OK;
}

// ---------------------------------------------------------------------------------------------
// Per-kind plan (SW §7.5 scratch table): whether op `kind` with these OPL argument words can be
// planned (RET_OK, RET_SCRATCH or RET_BAD_RECORD) and the scratch lines one tile needs in the ring.
// NOP / JOIN need nothing.
// ---------------------------------------------------------------------------------------------
constexpr uint32_t SCRATCH_UNSUPPORTED = 0xFFFFFFFFu;
struct op_plan {
    uint32_t status;          // opl::RET_OK / RET_SCRATCH / RET_BAD_RECORD
    uint32_t scratch_lines;   // lines of one tile (MUL_MAT: t_tile x (x + QA + y)); SCRATCH_UNSUPPORTED
                              // when status != RET_OK
};

// M7 kinds (SW §7.5 scratch column). TODO(W2b, P2.6 / P2.7): replace each stub by the kind's real
// requirement (ADD / MUL / SCALE 2-3 x row lines, RMS_NORM / GLU / GET_ROWS / UNARY 2 x row lines,
// ROPE 2 x head_dim x n_heads / 16, CPY 2 x block, SET_ROWS_KV and FLASH_ATTN per AT §7.4) and its
// argument checks. Until then they are unsupported: the host must not emit them and the executor
// has no handler for them (BAD_RECORD).
constexpr op_plan m7_unsupported(const dev_cfg &, const uint32_t *, uint32_t) {
    return op_plan{opl::RET_BAD_RECORD, SCRATCH_UNSUPPORTED};
}
constexpr op_plan add_plan(const dev_cfg & c, const uint32_t * a, uint32_t n)         { return m7_unsupported(c, a, n); }
constexpr op_plan mul_plan(const dev_cfg & c, const uint32_t * a, uint32_t n)         { return m7_unsupported(c, a, n); }
constexpr op_plan scale_plan(const dev_cfg & c, const uint32_t * a, uint32_t n)       { return m7_unsupported(c, a, n); }
constexpr op_plan rms_norm_plan(const dev_cfg & c, const uint32_t * a, uint32_t n)    { return m7_unsupported(c, a, n); }
constexpr op_plan glu_plan(const dev_cfg & c, const uint32_t * a, uint32_t n)         { return m7_unsupported(c, a, n); }
constexpr op_plan rope_plan(const dev_cfg & c, const uint32_t * a, uint32_t n)        { return m7_unsupported(c, a, n); }
constexpr op_plan get_rows_plan(const dev_cfg & c, const uint32_t * a, uint32_t n)    { return m7_unsupported(c, a, n); }
constexpr op_plan set_rows_kv_plan(const dev_cfg & c, const uint32_t * a, uint32_t n) { return m7_unsupported(c, a, n); }
constexpr op_plan flash_attn_plan(const dev_cfg & c, const uint32_t * a, uint32_t n)  { return m7_unsupported(c, a, n); }
constexpr op_plan cpy_plan(const dev_cfg & c, const uint32_t * a, uint32_t n)         { return m7_unsupported(c, a, n); }
constexpr op_plan unary_plan(const dev_cfg & c, const uint32_t * a, uint32_t n)       { return m7_unsupported(c, a, n); }

constexpr op_plan mul_mat_plan(const dev_cfg & c, const uint32_t * a, uint32_t) {
    mm_tiling t{};
    const uint32_t st = mm_plan(c, a[opl::mm::FMT], a[opl::mm::N], a[opl::mm::K], a[opl::mm::T], &t);
    return st == opl::RET_OK ? op_plan{st, t.t_tile * t.lpt} : op_plan{st, SCRATCH_UNSUPPORTED};
}

constexpr op_plan plan_op(const dev_cfg & c, uint32_t kind, const uint32_t * args, uint32_t n_args) {
    if (opl::kind_nargs(kind) == opl::NARGS_UNDEF || n_args != opl::kind_nargs(kind))
        return op_plan{opl::RET_BAD_RECORD, SCRATCH_UNSUPPORTED};
    switch (kind) {
        case opl::K_NOP:
        case opl::K_JOIN:        return op_plan{opl::RET_OK, 0u};
        case opl::K_MUL_MAT:     return mul_mat_plan(c, args, n_args);
        case opl::K_ADD:         return add_plan(c, args, n_args);
        case opl::K_MUL:         return mul_plan(c, args, n_args);
        case opl::K_SCALE:       return scale_plan(c, args, n_args);
        case opl::K_RMS_NORM:    return rms_norm_plan(c, args, n_args);
        case opl::K_GLU:         return glu_plan(c, args, n_args);
        case opl::K_ROPE:        return rope_plan(c, args, n_args);
        case opl::K_GET_ROWS:    return get_rows_plan(c, args, n_args);
        case opl::K_SET_ROWS_KV: return set_rows_kv_plan(c, args, n_args);
        case opl::K_FLASH_ATTN:  return flash_attn_plan(c, args, n_args);
        case opl::K_CPY:         return cpy_plan(c, args, n_args);
        case opl::K_UNARY:       return unary_plan(c, args, n_args);
        default:                 return op_plan{opl::RET_BAD_RECORD, SCRATCH_UNSUPPORTED};
    }
}
// The status alone (the host's supports_op on a built record, the executor's tests).
constexpr uint32_t check_op(const dev_cfg & c, uint32_t kind, const uint32_t * args, uint32_t n_args) {
    return plan_op(c, kind, args, n_args).status;
}

static_assert(plan_op(U200_K4, opl::K_NOP, nullptr, 0).status == opl::RET_OK, "NOP plans");
static_assert(mm_supported(U200_K4, FMT_Q8_0, 151936u, 896u) && mm_supported(U200_K4, FMT_Q4_K, 5120u, 27648u) &&
                  mm_supported(U200_K4, FMT_Q8_0, 262144u, 3840u) && !mm_supported(U200_K4, FMT_Q4_K, 64u, 32u) &&
                  !mm_supported(U200_K4, FMT_Q8_0, 2000000u, 32u),
              "plan.hpp: Qwen2.5 / Gemma 4 heads plan on the U200; K < block and a too-wide N do not");

}  // namespace plan
}  // namespace tsr
