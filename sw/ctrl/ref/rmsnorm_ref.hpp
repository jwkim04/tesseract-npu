// rmsnorm_ref.hpp — host C++ reference of the firmware's RMS_NORM kind (op list kind 0x14; docs/design/
// 06-software-verification.md "SW" §7.2, §7.5 and its last paragraph): it composes the tsr_ref_vu calls
// the kernel's tile ops execute, in the kernel's order, so the kernel tests compare the firmware
// against it bit for bit.
//
// The kernel (sw/ctrl/kernels/k_norm.cpp) runs, per row tile of a record (tsr::opl::rms argument
// words): LD x rows (the tile's rows, L(n) = ceil(n/16) lines each, dense in SPM), [LD the weight
// row(s)], SOP.MOVI s[sk] = inv_n and s[se] = eps (once per op), RED.SUMSQ with post RMS (TO_S for a
// one-row tile, else a per-row vector), VOP.MULCB (c = that s register / vector, b = the weight row,
// stride 0 for weight period 1) or VOP.MULC without a weight, in place, then ST of the rows. Per row
// r that is, with the VU's fp rules (VU §4.1, §6.2):
//   s_r = RC(SQ(FA(FM(tree128(partials of FM(x_j, x_j)), inv_n), eps)))   (RED.SUMSQ post RMS)
//   y_rj = FM(FM(x_rj, s_r), w_(r mod p),j)   with a weight,  FM(x_rj, s_r) without   (VOP)
//   words [n, 16 L(n)) of each output row are 0 (the VOP tail rule; the ST writes whole lines).
// Row tiles do not change any value (every row is reduced and scaled on its own), so the reference
// runs one RED over all rows and one VOP per weight-period block (one VOP when p = 1 or p >= rows).
//
// rmsnorm_ref_7op is the spec's lowering (docs/spec.md:311-321: RED.SUMSQ, SOP.MULI inv_n,
// SOP.ADDI eps, SOP.SQRT, SOP.RECIP, VOP.MULS, VOP.MUL) on the same rows; VU §6.2 states the two are
// bit-identical when s[sk] holds the MULI immediate, and the kernel tests check it.
#pragma once

#include <cstddef>
#include <cstdint>

#include "mulmat_ref.hpp"   // dram_view

namespace tsr {
namespace ref {

// Status of rmsnorm_ref / rope_ref (the kernels' codes: tsr::opl::RET_BAD_RECORD for NR_BAD_RECORD).
constexpr int NR_OK = 0;
constexpr int NR_BAD_RECORD = -1;   // arguments the kernel rejects (plan_norm.hpp)
constexpr int NR_RANGE = -2;        // a row outside the DRAM view (the device would fail with DRAM_RANGE)

// rows x n words: row r of x at x + r * x_stride words, of y at y + r * y_stride words (strides in
// words, multiples of 16 in the kernel's SPM image; y may equal x with the same stride: in place).
// w: weight rows (row k at w + k * w_stride words) used as row r mod w_period, or nullptr. Each y row
// gets its zero tail up to 16 L(n) words. eps / inv_n are f32 bit patterns.
void rmsnorm_ref_rows(uint32_t n, uint32_t rows, const uint32_t * x, size_t x_stride, uint32_t * y, size_t y_stride,
                      uint32_t eps, uint32_t inv_n, const uint32_t * w, size_t w_stride, uint32_t w_period);

// The 7-op lowering of docs/spec.md:311-321 (RED.SUMSQ post NONE, then SOP MULI / ADDI / SQRT / RECIP
// on the row result, VOP.MULC by it, VOP.MUL by the weight row), same arguments and layout.
void rmsnorm_ref_7op(uint32_t n, uint32_t rows, const uint32_t * x, size_t x_stride, uint32_t * y, size_t y_stride,
                     uint32_t eps, uint32_t inv_n, const uint32_t * w, size_t w_stride, uint32_t w_period);

// Applies one RMS_NORM record (args = the 10 OPL argument words, SW §7.2, tsr::opl::rms) to the DRAM
// view as the kernel does: every source row is read before any output row is written (the kernel's
// in-place DRAM case dst == src, same stride, is exact); output rows are L(n) lines with a zero tail.
// Weight rows are L(n) lines apart (dense; the weight tensor's rows start on lines, SW §7.2).
int rmsnorm_ref(const dram_view & d, const uint32_t args[10]);

}  // namespace ref
}  // namespace tsr
