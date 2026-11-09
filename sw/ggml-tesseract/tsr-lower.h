// tsr-lower.h — SPM tiling of one MUL_MAT (spec "ggml backend integration", lowering).
//
// The plan a MUL_MAT of N x K weights and T f32 tokens follows on the device, per T tile:
// LD the x tile, QACT it, MVM against W_OFF, ST the y tile. The SPM is partitioned
// [x tile | QA tile | y tile]; every operand starts on a line and no two operands share a line
// (spec "SPM rules"). The backend's supports_op and weight placement and the runtime's weight split
// use the shape check. No ggml dependency.
//
// The firmware MUL_MAT kind emits the ops with the shared, freestanding plan.hpp
// (docs/design/06-software-verification.md §9.3, SW-20).
#pragma once

#include <cstdint>

#include "tsr_isa.h"

// SPM tiling of one MUL_MAT.
struct tsr_mm_tiling {
    uint32_t kind;       // QA kind matching fmt (Q8_0 -> A32, Q4_K -> A256)
    uint32_t x_lines;    // one f32 src1 row: 4K/64 lines (K % 32 == 0, so exact)
    uint32_t qa_lines;   // QA_LINES(kind, K)
    uint32_t y_lines;    // ceil(4N/64): MVM dst_stride and the per-token ST length
    uint32_t t_tile;     // tokens per tile, 1 <= t_tile <= min(T, T_MAX)
    uint32_t spm_x;      // SPM line bases of the three regions
    uint32_t spm_qa;
    uint32_t spm_y;
    uint32_t spm_end;    // first line after the y region (<= SPM lines)
};

// Effective per-MVM token limit: min(cfg.t_max, TSR_T_MAX_ARCH) (and the 4-bit w5 T field).
uint32_t tsr_mm_t_max(const tsr_config & cfg);

// Lines one token needs in SPM (x + QA + y). 0 if the shape is not lowerable.
uint64_t tsr_mm_lines_per_token(uint32_t fmt, uint32_t N, uint32_t K);

// Can a MUL_MAT with this device format and shape be lowered at all (independent of T)?
// Checks fmt, K (multiple of the block size and of 32, <= K_MAX, fits the 16-bit K field),
// N >= 1, and that one token fits in SPM.
bool tsr_mm_shape_ok(const tsr_config & cfg, uint32_t fmt, uint64_t N, uint64_t K);

// Compute the tiling for T tokens. Returns false if the shape is not lowerable.
bool tsr_mm_plan_tiling(const tsr_config & cfg, uint32_t fmt, uint32_t N, uint32_t K, uint32_t T,
                        tsr_mm_tiling * out);

// Staging of one MUL_MAT through the IO stage (spec "lowering constraints"; SW §9.3 step 2): whether
// src1 / dst must pass through the stage, and the stage tiles. A MUL_MAT is staged when src1 or
// dst is not in an IO window (host memory), or when its dst rows do not start on a line
// (N*4 % 64 != 0) with T > 1. Each stage tile of <= tc_max tokens holds the staged x rows
// (x_row bytes each) at x_off and the staged, line-padded y rows (y_stride bytes each) at y_off.
struct tsr_stage_plan {
    bool     stage_x, stage_y;
    uint64_t per_token;   // stage bytes one token needs
    uint64_t tc_max;      // tokens per stage tile: min(T, stage_size / per_token)
    uint64_t x_off;       // stage offsets (byte offsets of the channel)
    uint64_t y_off;
    uint64_t n_tiles;     // ceil(T / tc_max)
};
// x_in_io / y_in_io: src1 / dst live in IO windows. Returns false if nothing is staged (*out then
// has stage_x = stage_y = false) or if one token does not fit the stage (tc_max = 0).
bool tsr_plan_stage(uint64_t stage_off, uint64_t stage_size, uint32_t N, uint32_t T, uint64_t x_row, uint64_t y_stride,
                    bool x_in_io, bool y_in_io, tsr_stage_plan * out);

