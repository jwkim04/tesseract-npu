// tsr-lower.cpp — see tsr-lower.h
#include "tsr-lower.h"

#include <algorithm>

#include "tsr_ref.h"

uint32_t tsr_mm_t_max(const tsr_config & cfg) {
    uint32_t t = cfg.t_max;
    t = std::min<uint32_t>(t, TSR_T_MAX_ARCH);
    t = std::min<uint32_t>(t, 15u);            // MVM w5 T field is 4 bits
    return t;
}

uint64_t tsr_mm_lines_per_token(uint32_t fmt, uint32_t N, uint32_t K) {
    if (!tsr_fmt_valid(fmt) || K == 0 || K % 32 != 0) {
        return 0;
    }
    const uint32_t kind = tsr_qa_kind_for_fmt(fmt);
    const uint64_t x  = (uint64_t) K * 4 / TSR_LINE_BYTES;
    const uint64_t qa = tsr_qa_lines(kind, K);
    const uint64_t y  = tsr_ceil_div_u64((uint64_t) N * 4, TSR_LINE_BYTES);
    return x + qa + y;
}

bool tsr_mm_shape_ok(const tsr_config & cfg, uint32_t fmt, uint64_t N, uint64_t K) {
    if (cfg.beat_bits != 512) {
        return false;                          // v1 software assumes 64-byte beats/lines
    }
    if (!tsr_fmt_valid(fmt)) {
        return false;
    }
    if (N < 1 || N > UINT32_MAX || K < 32 || K > 0xFFFFu || K % 32 != 0) {
        return false;
    }
    if (!tsr_fmt_k_valid(fmt, (uint32_t) K, cfg.k_max)) {
        return false;
    }
    if (tsr_mm_t_max(cfg) < 1) {
        return false;
    }
    const uint64_t spm_lines = cfg.spm_bytes / TSR_LINE_BYTES;
    const uint64_t lpt = tsr_mm_lines_per_token(fmt, (uint32_t) N, (uint32_t) K);
    return lpt > 0 && lpt <= spm_lines;
}

bool tsr_mm_plan_tiling(const tsr_config & cfg, uint32_t fmt, uint32_t N, uint32_t K, uint32_t T,
                        tsr_mm_tiling * out) {
    if (T < 1 || !tsr_mm_shape_ok(cfg, fmt, N, K)) {
        return false;
    }
    const uint64_t spm_lines = cfg.spm_bytes / TSR_LINE_BYTES;
    tsr_mm_tiling t = {};
    t.kind     = tsr_qa_kind_for_fmt(fmt);
    t.x_lines  = K * 4 / TSR_LINE_BYTES;
    t.qa_lines = tsr_qa_lines(t.kind, K);
    t.y_lines  = (uint32_t) tsr_ceil_div_u64((uint64_t) N * 4, TSR_LINE_BYTES);
    const uint64_t lpt = (uint64_t) t.x_lines + t.qa_lines + t.y_lines;
    uint64_t tt = std::min<uint64_t>(T, tsr_mm_t_max(cfg));
    tt = std::min<uint64_t>(tt, spm_lines / lpt);
    if (tt < 1) {
        return false;
    }
    t.t_tile  = (uint32_t) tt;
    t.spm_x   = 0;
    t.spm_qa  = t.spm_x  + t.t_tile * t.x_lines;
    t.spm_y   = t.spm_qa + t.t_tile * t.qa_lines;
    t.spm_end = t.spm_y  + t.t_tile * t.y_lines;
    *out = t;
    return true;
}

bool tsr_plan_stage(uint64_t stage_off, uint64_t stage_size, uint32_t N, uint32_t T, uint64_t x_row, uint64_t y_stride,
                    bool x_in_io, bool y_in_io, tsr_stage_plan * out) {
    tsr_stage_plan p = {};
    p.stage_x = !x_in_io;
    p.stage_y = !y_in_io || (((uint64_t) N * 4) % TSR_LINE_BYTES != 0 && T > 1);
    if (!p.stage_x && !p.stage_y) {
        *out = p;
        return false;
    }
    p.per_token = (p.stage_x ? x_row : 0) + (p.stage_y ? y_stride : 0);
    p.tc_max    = std::min<uint64_t>(T, stage_size / p.per_token);
    p.x_off     = stage_off;
    p.y_off     = stage_off + (p.stage_x ? p.tc_max * x_row : 0);
    p.n_tiles   = p.tc_max ? (T + p.tc_max - 1) / p.tc_max : 0;
    *out = p;
    return p.tc_max > 0;
}

