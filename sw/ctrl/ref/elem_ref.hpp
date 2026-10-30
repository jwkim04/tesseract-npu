// elem_ref.hpp — host C++ reference of the firmware's element-wise and row kinds: ADD 0x11, MUL 0x12,
// SCALE 0x13, GLU 0x15, GET_ROWS 0x17, CPY 0x1A, UNARY 0x1B (docs/v2/06-software-verification.md
// "SW" §7.2, §7.5 and its last paragraph): it composes the tsr_ref_vu calls the kernels' tile ops
// execute, in the kernels' order, so the kernel tests compare the firmware against it bit for bit.
//
// The kernels (sw/ctrl/kernels/k_elem.cpp, k_glu.cpp, k_rows.cpp; tile-op sequence in their header
// comments) run, per row tile, LD a rows (+ the b part), the VOP(s) in place, ST. Per row r with
// n elements and the VU's fp rules (VU §4.1: DAZ inputs, RNE, FTZ, canonical NaN; VU §6):
//   ADD / MUL   y = VOP.ADD / MUL (a_r, b_(r mod p))                     (b row; p = 1: one row)
//               y = VOP.ADDC / MULC (a_r, c = b word (r mod p))          (FLAG_B_SCALAR)
//   SCALE       y = VOP.MULC (x_r, c = s), then VOP.ADDC (y, c = b) when b != 0 (bits)
//   GLU         y = VOP.SWIGLU / GEGLU16 / GEGLU (gate_r, up_r)          (variant 0 / 1 / 2)
//   UNARY       y = VOP.TANH / GELU16 / SILU / EXP / GELU (x_r)          (func 0 .. 4)
//   CPY         dst row r = the src row's rl lines, bit for bit
//   GET_ROWS    dst row i = src row ids[i] (I32 in the preload area), bit for bit
// An f32 output row is L(n) = ceil(n/16) lines with words [n, 16 L) zero (the VOP tail rule; the ST
// writes whole lines). Row tiles, period blocks and the c-vector / s-register forms of the scalar
// never change a value (every element is computed on its own), so the reference runs one tsr_ref_vop
// per row with the row's b / c. Every source row is read before any output row is written (the
// kernels' in-place DRAM case, dst = a with the same stride, is exact).
#pragma once

#include <cstddef>
#include <cstdint>

#include "mulmat_ref.hpp"   // dram_view

namespace tsr {
namespace ref {

constexpr int EL_OK = 0;
constexpr int EL_BAD_RECORD = -1;   // arguments the kernel rejects (tsr/plan_elem.hpp, kind flags aside)
constexpr int EL_RANGE = -2;        // a row outside the DRAM view, or a GET_ROWS id the device reports
                                    // as DRAM_RANGE (negative, or beyond 2^32 beats)

// The VOP function of an element-wise record (kind ADD / MUL / GLU / UNARY / SCALE: MULC), or ~0u
// for an undefined variant / func.
uint32_t elem_vop_func(uint32_t kind, const uint32_t * args);

// One f32 output row of an element-wise kind: n words of y (and the zero tail up to 16 L(n) words)
// from the row's a (n words) and b (n words; nullptr when not read) and the scalar c (the c form's
// b word; SCALE's s). SCALE with bias: c2 = b bits (applied when nonzero).
void elem_ref_row(uint32_t kind, uint32_t func, uint32_t n, uint32_t * y, const uint32_t * a, const uint32_t * b,
                  uint32_t c, bool scalar, uint32_t c2 = 0u);

// Applies one record (args = the kind's OPL argument words, SW §7.2) to the DRAM view, as the
// kernel does. pre / pre_bytes: the list's preload area (GET_ROWS ids; nullptr / 0 otherwise).
int elem_ref(const dram_view & d, uint32_t kind, const uint32_t * args, uint32_t n_args, const uint8_t * pre = nullptr,
             size_t pre_bytes = 0);

}  // namespace ref
}  // namespace tsr
