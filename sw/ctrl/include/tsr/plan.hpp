// tsr/plan.hpp — planning code shared by the firmware and the host backend (docs/design/
// 06-software-verification.md "SW" §6.7 last paragraph, §7.3, §7.4, §7.5; SW-20).
//
// Freestanding and C++17-compatible: no libstdc++, no heap, no float. The firmware's kind handlers
// plan their tiles with it, and the host's supports_op asks the same functions, so the host
// rejects exactly what the firmware cannot plan (SW-20: one source, true by construction).
//
// Contents:
//   - device parameters that planning depends on (dev_cfg: SPM lines, T_MAX, K_MAX), read by the
//     executor from CFG0 / CFG1 / CFG2 through the core window, by the host from the CSRs;
//   - the executor's SPM layout (SW §7.3): argument range, OPL area, preload area, scratch ring,
//     and the ring's planning budget;
//   - format geometry: QA_LINES (QA kinds A32 / A256), ROW_BEATS (repacked weight rows), f32 lines;
//   - MUL_MAT tiling (tsr_mm_lines_per_token / tsr_mm_shape_ok / tsr_mm_plan_tiling, constraints spec
//     "supports_op" and "lowering"), with the T-tile chosen for the ring budget instead of the whole SPM;
//   - the per-kind plan (plan_op / check_op): RET_OK, RET_SCRATCH (cannot be planned in the ring)
//     or RET_BAD_RECORD (arguments the device cannot run), and the scratch lines per tile;
//   - M7 planning: plan_limits / kind_plan, the planners tsr::plan_<kind> (declared here,
//     defined in tsr/plan_elem.hpp, plan_norm.hpp, plan_attn.hpp, plan_moe.hpp), plan_kind, and the shared rules
//     plan_rows (row tiles) and for_period_segments (row periods).
//
// Planning budget: every plan uses ring_budget(cfg) = SPM lines - 16 (argument
// range) - 4096 (OPL area) - 256 (largest preload area) - the one-tile block (1,024 on the U200;
// 60,144 lines left), not the ring of the actual list. The
// actual ring is never smaller, so a plan made by the host for any op list is valid in the firmware,
// and the tiles (hence the pushed op stream) do not depend on the list's preload size.
#pragma once

#include <stdint.h>

#include "tsr/opl.hpp"

namespace tsr {
namespace plan {

// The functions a kind handler evaluates per op are forced inline (the firmware is built with -Os,
// which would otherwise call them): with T = 1 known at the call site, mm_plan folds to a few
// shifts, adds and compares.
#if defined(__GNUC__)
#define TSR_PLAN_HOT [[gnu::always_inline]]
#else
#define TSR_PLAN_HOT
#endif

constexpr uint32_t LINE_BYTES  = 64u;
constexpr uint32_t T_MAX_ARCH  = 8u;        // spec "Configuration parameters": architecture bound of T_MAX
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
    // M7 kinds: read once per launch by the executor like the three above, by the host from
    // the CSRs. 0 = unknown: a kind that needs the value plans nothing (BAD_RECORD), so a host that
    // fills only the M6 fields keeps rejecting the M7 kinds.
    uint32_t num_ch = 0;  // CFG0.NUM_CH: DRAM channels C (AT planes, FA chunk size; U200: 4)
    uint32_t pcap = 0;    // AT PCAP = CFG3[31:16] x 16 (U200: 9,344; 0: no AT ops)
    uint32_t vl = 0;      // CFG2.VL (16)
    uint32_t caps = 0;    // CAPS (EX §5.1): bit0 VU (VOP, RED, ROPE, SOP), bit5 AT ops, bit6 DQROW
};
constexpr uint32_t CAPS_VU = 1u << 0, CAPS_ATT = 1u << 5, CAPS_DQROW = 1u << 6;   // = mmio.hpp CAPS_*
constexpr uint32_t CAPS_GDN = 1u << 8;   // the fused Gated DeltaNet state update (VU §7.12; mmio.hpp CAPS_GDN)
constexpr uint32_t CAPS_MVM_LOCAL_SCALE = 1u << 13, CAPS_MVM_LOCAL_SOFTCAP = 1u << 14;
constexpr dev_cfg U200_K4 = {65536u, 8u, 32768u, 4u, 9344u, 16u, CAPS_VU | CAPS_ATT};

// ---------------------------------------------------------------------------------------------
// Executor SPM layout (SW §7.3; L = spm_lines):
//   [L - 16, L)                 runtime argument range (SW §6.3 R5)
//   [L - 16 - 4096, L - 16)     OPL area (256 KiB); a larger list streams in two 128 KiB halves
//   [opl_line - pre_lines, opl_line)  preload area (header pre_lines <= 256)
//   [pre_line - B, pre_line)    the one-tile block (B = one_block_lines(L), below)
//   [0, pre_line - B)           scratch ring (spm_ring)
//
// The one-tile block (SW §7.3 "M7 kinds in the executor"): one fixed
// block that every single-tile record of a kind whose tile is DMA-bracketed uses instead of a ring
// block. A tile is DMA-bracketed when its first write to the block is a DMA op (LD), its last read
// is a DMA op (ST, KVST), and every other op on the block lies between them by deps (the first non-
// DMA op depends on the tile's last LD, the first closing DMA op on the tile's last non-DMA op):
// ADD, MUL, SCALE, RMS_NORM, GLU, ROPE, GET_ROWS, CPY, UNARY and SET_ROWS_KV (LD, QACT, KVST). The
// DMA unit runs its queue in order, one op after the other (a dep on the op's own unit is implied,
// CC §8.3), so the next such tile's first LD starts after this tile's last DMA op is done, which
// itself followed every op of this tile: reusing the same lines needs no dep and no ring record,
// and costs no concurrency (consecutive DMA-bracketed tiles are serialized by the DMA queue
// whatever their SPM lines). MUL_MAT (whose blocks the M6 traces pin) and FLASH_ATTN keep the ring.
//
// Its size (SW §7.3 / SW-36): the block comes out of the ring's planning budget, which
// sets MUL_MAT's prefill T-tile (two tiles in the budget). 1,024 lines (64 KiB) keeps the T-tile of
// every Qwen2.5 0.5B-32B output head at 3 tokens (budget 60,144; 4,096 lines made it 2: +50 % head
// weight traffic in an all-logits prefill) and still holds every decode tile of Qwen2.5 0.5B / 32B
// and Gemma 4 12B but their GLU (32B 3,456 lines, 12B 1,920: those records take the ring, about 70
// RISC-V cycles more, once per layer). The static_asserts after mm_plan pin these T-tiles.
// ---------------------------------------------------------------------------------------------
constexpr uint32_t ARG_LINES      = 16u;
constexpr uint32_t OPL_AREA_LINES = 4096u;
constexpr uint32_t OPL_HALF_LINES = 2048u;
constexpr uint32_t PRE_MAX_LINES  = opl::PRE_MAX_LINES;
constexpr uint32_t ONE_BLOCK_MAX  = 1024u;   // 64 KiB (above)

// Lines of the one-tile block: ONE_BLOCK_MAX when that leaves the ring at least 7/8 of the space
// below the preload area, else that eighth rounded down to 64 lines (small test SPMs); 0 when the
// SPM is too small for the executor.
TSR_PLAN_HOT constexpr uint32_t one_block_lines(uint32_t spm_lines) {
    if (spm_lines < ARG_LINES + OPL_AREA_LINES + PRE_MAX_LINES + 1u) return 0u;
    const uint32_t b = (spm_lines - ARG_LINES - OPL_AREA_LINES - PRE_MAX_LINES) >> 3;
    return b >= ONE_BLOCK_MAX ? ONE_BLOCK_MAX : (b & ~63u);
}

struct exec_layout {
    uint32_t arg_line;     // first line of the argument range
    uint32_t opl_line;     // first line of the OPL area
    uint32_t pre_line;     // first line of the preload area
    uint32_t pre_lines;
    uint32_t ring_line;    // 0
    uint32_t ring_lines;   // = one_line
    uint32_t one_line;     // first line of the one-tile block (= pre_line - one_lines)
    uint32_t one_lines;    // one_block_lines(spm_lines)
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
    l.one_lines = one_block_lines(spm_lines);
    l.one_line = l.pre_line - l.one_lines;
    l.ring_line = 0u;
    l.ring_lines = l.one_line;
    *out = l;
    return true;
}
// The planning budget: the ring of a list with the largest preload area (0 if the SPM is too
// small for the executor).
TSR_PLAN_HOT constexpr uint32_t ring_budget(const dev_cfg & c) {
    return c.spm_lines >= ARG_LINES + OPL_AREA_LINES + PRE_MAX_LINES + 1u
               ? c.spm_lines - ARG_LINES - OPL_AREA_LINES - PRE_MAX_LINES - one_block_lines(c.spm_lines)
               : 0u;
}
static_assert(one_block_lines(65536u) == 1024u && one_block_lines(8192u) == 448u && one_block_lines(4368u) == 0u &&
                  ring_budget(U200_K4) == 60144u,
              "plan.hpp: the one-tile block on the U200 and on a 512 KiB test SPM; the U200 planning budget");

// ---------------------------------------------------------------------------------------------
// Format geometry (spec "Quantized activation (QA) layout", "Device weight layout"; tsr_ref.c is the reference)
// ---------------------------------------------------------------------------------------------
// Divides on the control core cost 38 cycles (Ibex, CC §3.2; SW §4.2), so everything a kind handler
// evaluates per op uses shifts: every block size here is a power of two (Q8_0 / A32 32 = 2^5,
// Q4_K / A256 256 = 2^8, a line 64 = 2^6, 16 f32 per line). Only ceil_div with a run-time divisor
// (the T-tile of a multi-tile MUL_MAT, T > 1) divides. Results equal the plain formulas
// (static_asserts below; fw_exec / fw_hdr compare them with tsr_ref's geometry).
constexpr uint32_t ceil_div(uint32_t a, uint32_t b) { return a / b + (a % b != 0u ? 1u : 0u); }
TSR_PLAN_HOT constexpr uint32_t ceil_shr(uint32_t a, uint32_t sh) { return (a >> sh) + ((a & ((1u << sh) - 1u)) != 0u ? 1u : 0u); }
TSR_PLAN_HOT constexpr bool     fmt_valid(uint32_t fmt) { return fmt == FMT_Q8_0 || fmt == FMT_Q4_K; }
TSR_PLAN_HOT constexpr uint32_t fmt_block(uint32_t fmt) { return fmt == FMT_Q8_0 ? 32u : fmt == FMT_Q4_K ? 256u : 0u; }
constexpr uint32_t fmt_block_log2(uint32_t fmt) { return fmt == FMT_Q8_0 ? 5u : 8u; }   // valid fmt only
TSR_PLAN_HOT constexpr uint32_t qa_kind_for_fmt(uint32_t fmt) { return fmt == FMT_Q8_0 ? QA_A32 : QA_A256; }
constexpr uint32_t qa_block(uint32_t kind) { return kind == QA_A32 ? 32u : kind == QA_A256 ? 256u : 0u; }
// QA_LINES(kind, K) = QS lines + D lines + BS lines (A256 only); 0 for an invalid kind or a K
// that is not a multiple of the block. QS = ceil(K / 64), D = ceil(4 (K / b) / 64), BS = ceil(2
// (K / 16) / 64), each by shifts (b = 2^5 or 2^8).
TSR_PLAN_HOT constexpr uint32_t qa_lines(uint32_t kind, uint32_t K) {
    if (kind > QA_A256) return 0u;
    const uint32_t lb = kind == QA_A32 ? 5u : 8u;
    if ((K & ((1u << lb) - 1u)) != 0u) return 0u;
    const uint32_t qs = ceil_shr(K, 6u);
    const uint32_t d = ceil_shr((K >> lb) << 2, 6u);
    const uint32_t bs = kind == QA_A256 ? ceil_shr((K >> 4) << 1, 6u) : 0u;
    return qs + d + bs;
}
// The plain formulas (spec "Quantized activation (QA) layout"), for the static_asserts.
constexpr uint32_t qa_lines_div(uint32_t kind, uint32_t K) {
    const uint32_t b = qa_block(kind);
    if (b == 0u || K % b != 0u) return 0u;
    return ceil_div(K, LINE_BYTES) + ceil_div(4u * (K / b), LINE_BYTES) +
           (kind == QA_A256 ? ceil_div(2u * (K / 16u), LINE_BYTES) : 0u);
}
static_assert(qa_lines(QA_A32, 896u) == qa_lines_div(QA_A32, 896u) && qa_lines(QA_A32, 32u) == qa_lines_div(QA_A32, 32u) &&
                  qa_lines(QA_A32, 4864u) == qa_lines_div(QA_A32, 4864u) &&
                  qa_lines(QA_A32, 27648u) == qa_lines_div(QA_A32, 27648u) &&
                  qa_lines(QA_A256, 256u) == qa_lines_div(QA_A256, 256u) &&
                  qa_lines(QA_A256, 4864u) == qa_lines_div(QA_A256, 4864u) &&
                  qa_lines(QA_A256, 27648u) == qa_lines_div(QA_A256, 27648u) &&
                  qa_lines(QA_A256, 5376u) == qa_lines_div(QA_A256, 5376u) && qa_lines(QA_A32, 100u) == 0u &&
                  qa_lines(QA_A256, 896u) == 0u && qa_lines(2u, 256u) == 0u,
              "plan.hpp: qa_lines by shifts = the plain formula");
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
TSR_PLAN_HOT constexpr uint32_t mm_t_max(const dev_cfg & c) {
    uint32_t t = c.t_max < T_MAX_ARCH ? c.t_max : T_MAX_ARCH;
    return t < T_FIELD_MAX ? t : T_FIELD_MAX;
}
// Lines one token needs (x + QA + y); 0 if fmt / K cannot be lowered (K % 32 == 0).
// (No overflow: K / 16 + QA_LINES + ceil(N / 16) < 2^28 + 2^27 + 2^28 for 32-bit K and N.)
TSR_PLAN_HOT constexpr uint32_t mm_lines_per_token(uint32_t fmt, uint32_t N, uint32_t K) {
    if (!fmt_valid(fmt) || K == 0u || (K & 31u) != 0u || (K & (fmt_block(fmt) - 1u)) != 0u || N == 0u) return 0u;
    return (K >> 4) + qa_lines(qa_kind_for_fmt(fmt), K) + ceil_shr(N, 4u);
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
// token fits). T itself never makes a shape unplannable (spec "supports_op": no condition on
// T), so the host checks T = 1.
// The device limits mm_plan checks against (K field / K_MAX, T_MAX, the ring budget), derived from
// dev_cfg once: the executor computes them once per launch (kind_ctx::mml).
struct mm_limits {
    uint32_t kmax;     // min(K_MAX, K field max)
    uint32_t t_max;    // mm_t_max: 0 = no MUL_MAT can run
    uint32_t budget;   // ring_budget
};
TSR_PLAN_HOT constexpr mm_limits mm_limits_of(const dev_cfg & c) {
    return mm_limits{c.k_max < K_FIELD_MAX ? c.k_max : K_FIELD_MAX, mm_t_max(c), ring_budget(c)};
}

// No divide when T = 1 (a decode token: t_tile = n_tiles = 1 whatever the budget) or when the
// T-tile is 1; otherwise two (the T-tile for the budget, n_tiles), once per op.
TSR_PLAN_HOT constexpr uint32_t mm_plan(const mm_limits & l, uint32_t fmt, uint32_t N, uint32_t K, uint32_t T,
                                        mm_tiling * out) {
    if (!fmt_valid(fmt) || N == 0u || T == 0u || K < 32u || (K & (fmt_block(fmt) - 1u)) != 0u || K > l.kmax ||
        l.t_max == 0u)
        return opl::RET_BAD_RECORD;
    // = mm_lines_per_token(fmt, N, K), with the QA lines evaluated once (the checks above hold:
    // K is a multiple of the block, which is a multiple of 32, and K <= 0xFFFF, so K + 4095 does
    // not overflow): QS = ceil(K / 64), D = ceil(4 (K / b) / 64) = ceil(K / 512) for A32 and
    // ceil(K / 4096) for A256, BS = ceil(2 (K / 16) / 64) = ceil(K / 512) (A256 only).
    const uint32_t kind = qa_kind_for_fmt(fmt);
    const uint32_t xl = K >> 4;
    const uint32_t q512 = (K + 511u) >> 9;
    const uint32_t ql = ((K + 63u) >> 6) + (kind == QA_A32 ? q512 : ((K + 4095u) >> 12) + q512);
    const uint32_t yl = ceil_shr(N, 4u);
    const uint32_t lpt = xl + ql + yl;
    if (lpt > l.budget) return opl::RET_SCRATCH;
    if (out == nullptr) return opl::RET_OK;
    mm_tiling & t = *out;
    t.kind = kind;
    t.x_lines = xl;
    t.qa_lines = ql;
    t.y_lines = yl;
    t.lpt = lpt;
    uint32_t tt = 1u;
    if (T > 1u) {
        tt = (l.budget >> 1) / lpt;
        if (tt == 0u) tt = 1u;
        if (tt > l.t_max) tt = l.t_max;
        if (tt > T) tt = T;
    }
    t.t_tile = tt;
    t.n_tiles = tt == 1u ? T : ceil_div(T, tt);
    return opl::RET_OK;
}
TSR_PLAN_HOT constexpr uint32_t mm_plan(const dev_cfg & c, uint32_t fmt, uint32_t N, uint32_t K, uint32_t T,
                                        mm_tiling * out) {
    return mm_plan(mm_limits_of(c), fmt, N, K, T, out);
}
// supports_op's MUL_MAT question (independent of T).
constexpr bool mm_supported(const dev_cfg & c, uint32_t fmt, uint32_t N, uint32_t K) {
    return mm_plan(c, fmt, N, K, 1u, nullptr) == opl::RET_OK;
}
// The prefill T-tiles of the model shapes on the U200: the ring budget (SPM layout, the
// one-tile block) decides them, so a change there that alters MUL_MAT's tiling (its weight traffic:
// a weight matrix streams once per T-tile) fails here instead of silently. Output heads are the
// shapes at the edge: Qwen2.5 0.5B (151,936 x 896) and 32B (152,064 x 5,120) hold 3 tokens per
// tile, Gemma 4 12B (262,144 x 3,840) 1; the layers' matrices T_MAX (8).
namespace detail {
constexpr uint32_t mm_t_tile_u200(uint32_t fmt, uint32_t N, uint32_t K) {
    mm_tiling t{};
    return mm_plan(U200_K4, fmt, N, K, 512u, &t) == opl::RET_OK ? t.t_tile : 0u;
}
}  // namespace detail
static_assert(detail::mm_t_tile_u200(FMT_Q8_0, 151936u, 896u) == 3u && detail::mm_t_tile_u200(FMT_Q8_0, 152064u, 5120u) == 3u &&
                  detail::mm_t_tile_u200(FMT_Q4_K, 152064u, 5120u) == 3u &&
                  detail::mm_t_tile_u200(FMT_Q8_0, 262144u, 3840u) == 1u &&
                  detail::mm_t_tile_u200(FMT_Q8_0, 4864u, 896u) == 8u && detail::mm_t_tile_u200(FMT_Q8_0, 896u, 4864u) == 8u &&
                  detail::mm_t_tile_u200(FMT_Q4_K, 27648u, 5120u) == 8u && detail::mm_t_tile_u200(FMT_Q4_K, 5120u, 27648u) == 8u &&
                  detail::mm_t_tile_u200(FMT_Q8_0, 15360u, 3840u) == 8u && detail::mm_t_tile_u200(FMT_Q8_0, 3840u, 15360u) == 8u,
              "plan.hpp: MUL_MAT prefill T-tiles of the Qwen2.5 0.5B / 32B and Gemma 4 12B shapes on the U200");

// ---------------------------------------------------------------------------------------------
// M7 planning (SW §7.5): the device limits every M7 kind plans against and the
// plan one returns. The per-kind planners are freestanding constexpr-friendly functions in
// namespace tsr, one header per kernel group (tsr/plan_elem.hpp: ADD, MUL, SCALE, GLU, UNARY, CPY,
// GET_ROWS; tsr/plan_norm.hpp: RMS_NORM, ROPE; tsr/plan_attn.hpp: SET_ROWS_KV, FLASH_ATTN), all of
// the form
//     bool tsr::plan_<kind>(const uint32_t * args, uint32_t n_args, const plan_limits & l, kind_plan * out)
// args: the record's argument words (SW §7.2; the firmware passes a DTCM / register copy of the
// words it read in place), n_args: their count. Contract:
//   - out is never nullptr; out->status is always written: RET_OK, RET_BAD_RECORD (arguments the
//     device cannot run on any ring: a wrong n_args, a bad enumeration or flag, a shape outside the
//     ops' fields or the kind's tile composition, a device without the unit: caps, num_ch or pcap
//     0) or RET_SCRATCH (the smallest tile does not fit l.budget); the other fields only with RET_OK;
//   - returns out->status == RET_OK;
//   - depends only on (args, n_args, l): not on the list's preload size or the ring's state, so
//     the host's supports_op and the firmware agree (SW-20);
//   - the preload references are checked against the list's preload area by the caller (the
//     handler: kind_ctx::pre_ok, exec.hpp; the host reader: opl::check_args), not here;
//   - per-op work is shifts, adds and compares when the op has one tile (decode shapes, T = 1): a
//     divide only where a multi-tile op needs its tile count (as mm_plan, SW §4.2).
// The thirteen planners are declared below (after plan_limits / kind_plan) and defined by the three
// headers, which this one includes after the declarations; plan_op / plan_kind call them. A
// planner header either starts with #pragma once and #include "tsr/plan.hpp", or is a fragment
// (#if !defined(TSR_PLAN_CORE_HPP) #include "tsr/plan.hpp" #elif !defined(<its guard>) ...), then
// defines its planners (constexpr, the declared signature); any include order works either way. A
// kind without a source keeps a stub that returns plan_fail(out, RET_BAD_RECORD) (the
// device runs nothing it cannot plan).
// ---------------------------------------------------------------------------------------------
struct plan_limits {
    uint32_t budget;      // ring_budget: ring lines one op's live tiles may take
    uint32_t t_max;       // mm_t_max: QACT / MVM tokens per op, FA pass size (1..8; 0: no MVM work)
    uint32_t k_max;       // min(CFG2 K_MAX, 0xFFFF): QACT / MVM K (and FA's G E <= K_MAX)
    uint32_t num_ch;      // dev_cfg num_ch (C; 0 = unknown)
    uint32_t pcap;        // dev_cfg pcap (0 = no AT ops)
    uint32_t vl;          // dev_cfg vl
    uint32_t caps;        // dev_cfg caps (CAPS_VU, CAPS_ATT, ...)
    uint32_t spm_lines;   // CFG1
};
TSR_PLAN_HOT constexpr plan_limits plan_limits_of(const dev_cfg & c) {
    return plan_limits{ring_budget(c), mm_t_max(c), c.k_max < K_FIELD_MAX ? c.k_max : K_FIELD_MAX, c.num_ch, c.pcap,
                       c.vl, c.caps, c.spm_lines};
}
constexpr plan_limits LIMITS_U200 = plan_limits_of(U200_K4);

constexpr uint32_t SCRATCH_UNSUPPORTED = 0xFFFFFFFFu;

// One op's plan (scratch per tile as SW §7.5's table counts it, the tiling, kind-specific values).
struct kind_plan {
    uint32_t status;          // opl::RET_OK / RET_SCRATCH / RET_BAD_RECORD (always written)
    uint32_t scratch_lines;   // ring lines of one tile, all its blocks (the op_plan / supports_op
                              // figure; SCRATCH_UNSUPPORTED unless RET_OK)
    uint32_t row_lines;       // lines of one row of the kind's unit (f32 rows: vu_lines(n))
    uint32_t tile_rows;       // rows per tile (<= VU_ROWS_MAX: one VU op's rows)
    uint32_t n_tiles;         // tiles of the op (1: a single-tile op, the fast path's case)
    uint32_t flags;           // KP_* below; bits [31:16] kind-specific (the kind's header lists them)
    uint32_t aux[4];          // kind-specific planned values (the kind's header lists them)
};
constexpr uint32_t KP_ONE_TILE = 1u << 0;    // n_tiles == 1
constexpr uint32_t KP_KIND_LSB = 16u;        // first kind-specific flag bit

// Field limits of the VU / AT encodings (VU §5.1): element counts 24 bits, rows / strides / heads /
// T 16 bits, SPM lines 24 bits.
constexpr uint32_t VU_M_MAX = 0xFFFFFFu, VU_ROWS_MAX = 0xFFFFu, VU_STRIDE_MAX = 0xFFFFu, SPM_LINE_MAX = 0xFFFFFFu;
// L(n) = ceil(n / 16): lines of n f32 values (VU §4.2), by a shift.
TSR_PLAN_HOT constexpr uint32_t vu_lines(uint32_t n) { return (n >> 4) + ((n & 15u) != 0u ? 1u : 0u); }

// The shared row-tiling rule of the row-wise kinds (as mm_plan's T-tile, SW §7.4): a tile of
// tile_rows rows takes tile_rows x lpr ring lines (lpr: the lines of every block of one row); all
// rows in one tile when they fit the budget (at most VU_ROWS_MAX); else as many rows as two tiles in
// the budget allow (double buffering), at least 1 (a single buffer), at most VU_ROWS_MAX.
// RET_SCRATCH when one row does not fit (lpr > budget), RET_BAD_RECORD for lpr = 0. No divide for a
// single-tile op (decode shapes): one 32 x 32 -> 64 multiply and compares.
struct row_tiling {
    uint32_t tile_rows, n_tiles;
};
TSR_PLAN_HOT constexpr uint32_t plan_rows(uint32_t rows, uint32_t lpr, uint32_t budget, row_tiling * out) {
    if (lpr == 0u) return opl::RET_BAD_RECORD;
    if (lpr > budget) return opl::RET_SCRATCH;
    if (rows <= VU_ROWS_MAX && uint64_t(rows) * lpr <= budget) {
        out->tile_rows = rows;
        out->n_tiles = rows != 0u ? 1u : 0u;
        return opl::RET_OK;
    }
    uint32_t tr = (budget >> 1) / lpr;
    if (tr == 0u) tr = 1u;
    if (tr > VU_ROWS_MAX) tr = VU_ROWS_MAX;
    out->tile_rows = tr;
    out->n_tiles = ceil_div(rows, tr);
    return opl::RET_OK;
}

// Row periods (SW §7.2 ADD / MUL "b row period", RMS_NORM "weight row period"; VU §9: the VU has
// strides, not periods). Rows [r0, r0 + n) of an op whose operand b has row period p read b row
// r mod p. for_period_segments calls f(row, len, b_row, bcast) for the segments that read
// consecutive b rows, in row order, and returns their count: one segment with bcast = true (b
// stride 0: every row reads b row 0) for p <= 1; else the rows split where r mod p wraps to 0 (one
// segment when r0 + n <= p, e.g. a decode op whose b has a row per head), each one VU op whose b
// operand starts at b row b_row with b's stride. A remainder only when r0 >= p (a later tile).
template <class F> constexpr uint32_t for_period_segments(uint32_t r0, uint32_t n, uint32_t p, F && f) {
    if (n == 0u) return 0u;
    if (p <= 1u) {
        f(r0, n, 0u, true);
        return 1u;
    }
    uint32_t m = r0 < p ? r0 : r0 % p;
    uint32_t r = r0, left = n, k = 0;
    while (left != 0u) {
        const uint32_t len = left < p - m ? left : p - m;
        f(r, len, m, false);
        r += len;
        left -= len;
        m = 0u;
        ++k;
    }
    return k;
}
constexpr uint32_t period_segments(uint32_t r0, uint32_t n, uint32_t p) {
    return for_period_segments(r0, n, p, [](uint32_t, uint32_t, uint32_t, bool) {});
}
static_assert(period_segments(0, 8, 1) == 1 && period_segments(0, 8, 8) == 1 && period_segments(0, 8, 64) == 1 &&
                  period_segments(0, 8, 4) == 2 && period_segments(0, 9, 4) == 3 && period_segments(6, 4, 4) == 2 &&
                  period_segments(0, 0, 4) == 0,
              "row period segments");

// Every planner's failure exit: status, the unsupported scratch figure, false.
TSR_PLAN_HOT constexpr bool plan_fail(kind_plan * out, uint32_t status) {
    out->status = status;
    out->scratch_lines = SCRATCH_UNSUPPORTED;
    return false;
}

}  // namespace plan
using plan::kind_plan;
using plan::plan_limits;

// The M7 planners (contract above; defined in tsr/plan_elem.hpp, plan_norm.hpp, plan_attn.hpp).
TSR_PLAN_HOT constexpr bool plan_add(const uint32_t * args, uint32_t n_args, const plan_limits & l, kind_plan * out);
TSR_PLAN_HOT constexpr bool plan_mul(const uint32_t * args, uint32_t n_args, const plan_limits & l, kind_plan * out);
TSR_PLAN_HOT constexpr bool plan_scale(const uint32_t * args, uint32_t n_args, const plan_limits & l, kind_plan * out);
TSR_PLAN_HOT constexpr bool plan_rms_norm(const uint32_t * args, uint32_t n_args, const plan_limits & l, kind_plan * out);
TSR_PLAN_HOT constexpr bool plan_glu(const uint32_t * args, uint32_t n_args, const plan_limits & l, kind_plan * out);
TSR_PLAN_HOT constexpr bool plan_rope(const uint32_t * args, uint32_t n_args, const plan_limits & l, kind_plan * out);
TSR_PLAN_HOT constexpr bool plan_get_rows(const uint32_t * args, uint32_t n_args, const plan_limits & l, kind_plan * out);
TSR_PLAN_HOT constexpr bool plan_set_rows_kv(const uint32_t * args, uint32_t n_args, const plan_limits & l, kind_plan * out);
TSR_PLAN_HOT constexpr bool plan_flash_attn(const uint32_t * args, uint32_t n_args, const plan_limits & l, kind_plan * out);
TSR_PLAN_HOT constexpr bool plan_cpy(const uint32_t * args, uint32_t n_args, const plan_limits & l, kind_plan * out);
TSR_PLAN_HOT constexpr bool plan_unary(const uint32_t * args, uint32_t n_args, const plan_limits & l, kind_plan * out);
TSR_PLAN_HOT constexpr bool plan_moe_route(const uint32_t * args, uint32_t n_args, const plan_limits & l, kind_plan * out);
TSR_PLAN_HOT constexpr bool plan_mul_mat_id(const uint32_t * args, uint32_t n_args, const plan_limits & l, kind_plan * out);
// The Gated DeltaNet kinds (SW §7.10; defined in tsr/plan_gdn.hpp).
TSR_PLAN_HOT constexpr bool plan_softplus(const uint32_t * args, uint32_t n_args, const plan_limits & l, kind_plan * out);
TSR_PLAN_HOT constexpr bool plan_ssm_conv(const uint32_t * args, uint32_t n_args, const plan_limits & l, kind_plan * out);
TSR_PLAN_HOT constexpr bool plan_gated_delta_net(const uint32_t * args, uint32_t n_args, const plan_limits & l, kind_plan * out);
}  // namespace tsr

#define TSR_PLAN_CORE_HPP 1   // the planner headers' fragment guard (either form above works)
#include "tsr/plan_elem.hpp"
#include "tsr/plan_norm.hpp"
#include "tsr/plan_attn.hpp"
#include "tsr/plan_moe.hpp"
#include "tsr/plan_gdn.hpp"

namespace tsr {
namespace plan {

// ---------------------------------------------------------------------------------------------
// Per-kind plan (SW §7.5 scratch table): whether op `kind` with these OPL argument words can be
// planned (RET_OK, RET_SCRATCH or RET_BAD_RECORD) and the scratch lines one tile needs in the ring.
// NOP / JOIN need nothing. The M7 kinds go to their planners (above).
// ---------------------------------------------------------------------------------------------
struct op_plan {
    uint32_t status;          // opl::RET_OK / RET_SCRATCH / RET_BAD_RECORD
    uint32_t scratch_lines;   // lines of one tile (MUL_MAT: t_tile x (x + QA + y)); SCRATCH_UNSUPPORTED
                              // when status != RET_OK
};

constexpr op_plan mul_mat_plan(const dev_cfg & c, const uint32_t * a, uint32_t) {
    mm_tiling t{};
    const uint32_t st = mm_plan(c, a[opl::mm::FMT], a[opl::mm::N], a[opl::mm::K], a[opl::mm::T], &t);
    return st == opl::RET_OK ? op_plan{st, t.t_tile * t.lpt} : op_plan{st, SCRATCH_UNSUPPORTED};
}

// The M7 planner of `kind` (false with status BAD_RECORD for a kind that is not M7's).
constexpr bool plan_kind(const plan_limits & l, uint32_t kind, const uint32_t * args, uint32_t n_args, kind_plan * out) {
    switch (kind) {
        case opl::K_ADD:         return plan_add(args, n_args, l, out);
        case opl::K_MUL:         return plan_mul(args, n_args, l, out);
        case opl::K_SCALE:       return plan_scale(args, n_args, l, out);
        case opl::K_RMS_NORM:    return plan_rms_norm(args, n_args, l, out);
        case opl::K_GLU:         return plan_glu(args, n_args, l, out);
        case opl::K_ROPE:        return plan_rope(args, n_args, l, out);
        case opl::K_GET_ROWS:    return plan_get_rows(args, n_args, l, out);
        case opl::K_SET_ROWS_KV: return plan_set_rows_kv(args, n_args, l, out);
        case opl::K_FLASH_ATTN:  return plan_flash_attn(args, n_args, l, out);
        case opl::K_CPY:         return plan_cpy(args, n_args, l, out);
        case opl::K_UNARY:       return plan_unary(args, n_args, l, out);
        case opl::K_MOE_ROUTE:   return plan_moe_route(args, n_args, l, out);
        case opl::K_MUL_MAT_ID:  return plan_mul_mat_id(args, n_args, l, out);
        case opl::K_SOFTPLUS:    return plan_softplus(args, n_args, l, out);
        case opl::K_SSM_CONV:    return plan_ssm_conv(args, n_args, l, out);
        case opl::K_GATED_DELTA_NET: return plan_gated_delta_net(args, n_args, l, out);
        default:                 return plan_fail(out, opl::RET_BAD_RECORD);
    }
}

constexpr op_plan plan_op(const dev_cfg & c, uint32_t kind, const uint32_t * args, uint32_t n_args) {
    if (opl::kind_nargs(kind) == opl::NARGS_UNDEF || n_args != opl::kind_nargs(kind))
        return op_plan{opl::RET_BAD_RECORD, SCRATCH_UNSUPPORTED};
    switch (kind) {
        case opl::K_NOP:
        case opl::K_JOIN:    return op_plan{opl::RET_OK, 0u};
        case opl::K_MUL_MAT: return mul_mat_plan(c, args, n_args);
        case opl::K_MUL_MAT_SCALE:
        case opl::K_MUL_MAT_SOFTCAP: {
            const uint32_t cap = kind == opl::K_MUL_MAT_SCALE ? CAPS_MVM_LOCAL_SCALE : CAPS_MVM_LOCAL_SOFTCAP;
            if (!(c.caps & cap) || c.num_ch != 4u ||
                !opl::mm_scale::operands_ok(args[opl::mm::ADD], args[opl::mm::ADD_STRIDE], args[opl::mm::FLAGS]))
                return op_plan{opl::RET_BAD_RECORD, SCRATCH_UNSUPPORTED};
            return mul_mat_plan(c, args, n_args);
        }
        default: {
            kind_plan kp{};
            return plan_kind(plan_limits_of(c), kind, args, n_args, &kp) ? op_plan{opl::RET_OK, kp.scratch_lines}
                                                                          : op_plan{kp.status, SCRATCH_UNSUPPORTED};
        }
    }
}
// The status alone (the host's supports_op on a built record, the executor's tests).
constexpr uint32_t check_op(const dev_cfg & c, uint32_t kind, const uint32_t * args, uint32_t n_args) {
    return plan_op(c, kind, args, n_args).status;
}
static_assert(plan_op(U200_K4, opl::K_NOP, nullptr, 0).status == opl::RET_OK, "NOP plans");
constexpr uint32_t mm_lpt_of(const dev_cfg & c, uint32_t fmt, uint32_t N, uint32_t K) {
    mm_tiling t{};
    return mm_plan(c, fmt, N, K, 1u, &t) == opl::RET_OK ? t.lpt : 0u;
}
constexpr bool mm_qa_matches(uint32_t fmt, uint32_t k_step) {
    for (uint32_t K = k_step; K <= 0xFFFFu; K += k_step) {
        mm_tiling t{};
        if (mm_plan(mm_limits{0xFFFFu, 8u, 0xFFFFFFFFu}, fmt, 16u, K, 1u, &t) != opl::RET_OK) return false;
        if (t.qa_lines != qa_lines_div(qa_kind_for_fmt(fmt), K) || t.lpt != mm_lines_per_token(fmt, 16u, K)) return false;
    }
    return true;
}
static_assert(mm_qa_matches(FMT_Q8_0, 32u) && mm_qa_matches(FMT_Q4_K, 256u),
              "mm_plan's QA lines = QA_LINES for every K of the K field (Q8_0: K % 32 = 0, Q4_K: K % 256 = 0)");
static_assert(mm_lpt_of(U200_K4, FMT_Q8_0, 896u, 896u) == mm_lines_per_token(FMT_Q8_0, 896u, 896u) &&
                  mm_lpt_of(U200_K4, FMT_Q4_K, 896u, 4864u) == mm_lines_per_token(FMT_Q4_K, 896u, 4864u) &&
                  mm_lpt_of(U200_K4, FMT_Q8_0, 151936u, 896u) == mm_lines_per_token(FMT_Q8_0, 151936u, 896u),
              "mm_plan's lines per token = mm_lines_per_token");
static_assert(mm_supported(U200_K4, FMT_Q8_0, 151936u, 896u) && mm_supported(U200_K4, FMT_Q4_K, 5120u, 27648u) &&
                  mm_supported(U200_K4, FMT_Q8_0, 262144u, 3840u) && !mm_supported(U200_K4, FMT_Q4_K, 64u, 32u) &&
                  !mm_supported(U200_K4, FMT_Q8_0, 2000000u, 32u),
              "plan.hpp: Qwen2.5 / Gemma 4 heads plan on the U200; K < block and a too-wide N do not");

}  // namespace plan
}  // namespace tsr
