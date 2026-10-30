// rmsnorm_ref.cpp — see rmsnorm_ref.hpp (SW §7.5: the RMS_NORM kind's tile-op sequence as tsr_ref_vu
// calls; VU §6.2 RED.SUMSQ with post RMS, VU §5.2 MULCB / MULC).
#include "rmsnorm_ref.hpp"

#include <cstring>
#include <vector>

#include "tsr/opl.hpp"
#include "tsr/plan.hpp"
#include "tsr_ref.h"
#include "tsr_ref_vu.h"

namespace tsr {
namespace ref {

namespace {

constexpr uint64_t LINE = 64u;

uint8_t * at(const dram_view & d, uint32_t h, uint64_t lines, uint64_t bytes) {
    const uint32_t ch = h >> 28;
    const uint64_t off = (uint64_t(h & 0x0FFFFFFFu) + lines) * LINE;
    if (ch >= d.num_ch || off + bytes > d.ch_bytes) return nullptr;
    return d.ch[ch] + off;
}

uint32_t rd32(const uint8_t * p) {
    return uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24;
}
void wr32(uint8_t * p, uint32_t v) {
    p[0] = uint8_t(v); p[1] = uint8_t(v >> 8); p[2] = uint8_t(v >> 16); p[3] = uint8_t(v >> 24);
}

uint32_t lines_of(uint32_t n) { return (n >> 4) + ((n & 15u) != 0u ? 1u : 0u); }

// The kernel's VOP for rows [r0, r0 + nr) of one weight-period block: MULCB with the c vector (or
// MULC without a weight). The tsr_ref_vop strides are in lines.
void scale_rows(uint32_t n, uint32_t nr, const uint32_t * x, size_t xs, uint32_t * y, size_t ys, const uint32_t * c,
                const uint32_t * w, size_t ws) {
    const uint32_t cm = nr == 1u ? TSR_VOP_CMODE_SREG : TSR_VOP_CMODE_VEC;
    if (w != nullptr)
        tsr_ref_vop(TSR_VOP_FUNC_MULCB, cm, nr, n, y, uint32_t(ys / 16u), x, uint32_t(xs / 16u), w, uint32_t(ws / 16u), c);
    else
        tsr_ref_vop(TSR_VOP_FUNC_MULC, cm, nr, n, y, uint32_t(ys / 16u), x, uint32_t(xs / 16u), nullptr, 0u, c);
}

}  // namespace

void rmsnorm_ref_rows(uint32_t n, uint32_t rows, const uint32_t * x, size_t x_stride, uint32_t * y, size_t y_stride,
                      uint32_t eps, uint32_t inv_n, const uint32_t * w, size_t w_stride, uint32_t w_period) {
    if (n == 0u || rows == 0u) return;
    // RED.SUMSQ, post RMS, sk = inv_n, se = eps: one result per row (the TO_S result of a one-row tile
    // is the same tsr_ref_red_row value).
    std::vector<uint32_t> s(size_t(lines_of(rows)) * 16u, 0u);
    tsr_ref_red(TSR_RED_KIND_SUMSQ, TSR_RED_POST_RMS, rows, n, s.data(), x, uint32_t(x_stride / 16u), inv_n, eps);
    // VOP: one op when the weight is one row (stride 0) or has a row per row; else one per period block.
    if (w == nullptr || w_period == 1u || w_period >= rows) {
        const size_t ws = (w != nullptr && w_period != 1u) ? w_stride : 0u;
        scale_rows(n, rows, x, x_stride, y, y_stride, s.data(), w, ws);
        return;
    }
    for (uint32_t r0 = 0; r0 < rows; r0 += w_period) {
        const uint32_t nr = rows - r0 < w_period ? rows - r0 : w_period;
        // c for rows r0 ...: the per-row results from word r0 (the kernel's per-tile vectors).
        std::vector<uint32_t> c(s.begin() + r0, s.begin() + r0 + nr);
        scale_rows(n, nr, x + size_t(r0) * x_stride, x_stride, y + size_t(r0) * y_stride, y_stride, c.data(), w, w_stride);
    }
}

void rmsnorm_ref_7op(uint32_t n, uint32_t rows, const uint32_t * x, size_t x_stride, uint32_t * y, size_t y_stride,
                     uint32_t eps, uint32_t inv_n, const uint32_t * w, size_t w_stride, uint32_t w_period) {
    for (uint32_t r = 0; r < rows; ++r) {
        const uint32_t * xr = x + size_t(r) * x_stride;
        uint32_t * yr = y + size_t(r) * y_stride;
        // RED.SUMSQ (TO_S, post NONE) -> s1; SOP.MULI s1 *= inv_n; ADDI += eps; SQRT; RECIP.
        uint32_t s1 = tsr_ref_red_row(TSR_RED_KIND_SUMSQ, TSR_RED_POST_NONE, xr, n, 0u, 0u);
        s1 = tsr_ref_sop(TSR_SOP_SUB_MULI, s1, 0u, inv_n, 0u);
        s1 = tsr_ref_sop(TSR_SOP_SUB_ADDI, s1, 0u, eps, 0u);
        s1 = tsr_ref_sop(TSR_SOP_SUB_SQRT, s1, 0u, 0u, 0u);
        s1 = tsr_ref_sop(TSR_SOP_SUB_RECIP, s1, 0u, 0u, 0u);
        // VOP.MULS (MULC with cmode 1), then VOP.MUL by the weight row.
        tsr_ref_vop(TSR_VOP_FUNC_MULC, TSR_VOP_CMODE_SREG, 1u, n, yr, 0u, xr, 0u, nullptr, 0u, &s1);
        if (w != nullptr) {
            const uint32_t * wr = w + size_t(r % w_period) * w_stride;
            tsr_ref_vop(TSR_VOP_FUNC_MUL, TSR_VOP_CMODE_NONE, 1u, n, yr, 0u, yr, 0u, wr, 0u, nullptr);
        }
    }
}

int rmsnorm_ref(const dram_view & d, const uint32_t a[10]) {
    namespace r = opl::rms;
    const uint32_t n = a[r::N], rows = a[r::ROWS], L = lines_of(n);
    const bool has_w = a[r::W] != opl::HANDLE_NONE;
    const uint32_t p = a[r::W_PERIOD];
    // The kernel's argument rules (plan_norm.hpp; the budget is the kernel's business).
    if (plan::norm::rms_check(n, rows, a[r::DST_STRIDE], has_w, p) != opl::RET_OK) return NR_BAD_RECORD;
    const size_t rw = size_t(L) * 16u;                 // words per row in the SPM image
    std::vector<uint32_t> x(size_t(rows) * rw, 0u), y(size_t(rows) * rw, 0u);
    for (uint32_t i = 0; i < rows; ++i) {
        const uint8_t * p8 = at(d, a[r::SRC], uint64_t(i) * a[r::SRC_STRIDE], uint64_t(L) * LINE);
        if (p8 == nullptr) return NR_RANGE;
        for (uint32_t j = 0; j < n; ++j) x[size_t(i) * rw + j] = rd32(p8 + 4u * j);
    }
    std::vector<uint32_t> w;
    const uint32_t wrows = has_w ? (p < rows ? p : rows) : 0u;
    if (has_w) {
        w.assign(size_t(wrows) * rw, 0u);
        for (uint32_t k = 0; k < wrows; ++k) {
            const uint8_t * p8 = at(d, a[r::W], uint64_t(k) * L, uint64_t(L) * LINE);
            if (p8 == nullptr) return NR_RANGE;
            for (uint32_t j = 0; j < n; ++j) w[size_t(k) * rw + j] = rd32(p8 + 4u * j);
        }
    }
    rmsnorm_ref_rows(n, rows, x.data(), rw, y.data(), rw, a[r::EPS], a[r::INV_N], has_w ? w.data() : nullptr, rw,
                     has_w ? p : 1u);
    for (uint32_t i = 0; i < rows; ++i) {
        uint8_t * p8 = at(d, a[r::DST], uint64_t(i) * a[r::DST_STRIDE], uint64_t(L) * LINE);
        if (p8 == nullptr) return NR_RANGE;
        for (uint32_t j = 0; j < 16u * L; ++j) wr32(p8 + 4u * j, y[size_t(i) * rw + j]);
    }
    return NR_OK;
}

}  // namespace ref
}  // namespace tsr
