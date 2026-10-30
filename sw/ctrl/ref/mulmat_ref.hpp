// mulmat_ref.hpp — host C++ reference of the firmware's MUL_MAT kind (docs/design/06-software-verification.md
// "SW" §7.4, §7.5 last paragraph): it composes the tsr_ref calls the kernel's tile ops execute, in the
// kernel's order, so the kernel tests compare the firmware against it bit for bit.
//
// The kernel (sw/ctrl/kernels/k_mulmat.cpp) runs, per T tile of a MUL_MAT record (op list kind 0x10,
// SW §7.2 argument words, tsr::opl::mm):
//   LD x rows -> QACT (A32 for Q8_0, A256 for Q4_K; one QA vector per token) -> [LD the addend rows,
//   or the old y rows for ACC_DST, into the y block] -> MVM (ACC when a row was loaded) -> ST y.
// Per token t that is, with the spec's fp rules (tsr_ref.h; DAZ / RNE / FTZ / canonical NaN):
//   qa    = tsr_qact(kind, x_t, K)                                   (QACT)
//   d_n   = tsr_dot_row(fmt, W row n, K, qa)                         (MVM, 16 partials + tree)
//   y_t,n = tsr_fadd(old_t,n, d_n)  with an addend / ACC_DST, else d_n   (MVM ACC: one f32 add)
//   old_t = addend row t (row 0 when the addend stride is 0), or the y row t before the op (ACC_DST)
//   the y row holds N words and a zero tail up to ceil(N/16) lines (the ST writes whole lines).
// Rows: x_t at handle X + t * x_stride lines, y_t at Y + t * y_stride, addend_t at A + t * a_stride
// (a_stride 0: row 0 for every token); weights at W_OFF in the symmetric layout (tsr_row_location).
// A record that sets both an addend and ACC_DST is rejected (the kernel's decision P2.5: the y block
// can hold only one old row; RET BAD_RECORD), as is everything tsr::plan::mm_plan rejects.
// The reference reads every input row of token t before it writes y_t, tokens in order: the kernel
// reads tile t+1's rows before it stores tile t, so the two agree whenever no x / addend row of a
// later token overlaps an earlier token's y row (the backend's contract: MUL_MAT never aliases its
// sources with other tokens' outputs).
#pragma once

#include <cstddef>
#include <cstdint>

namespace tsr {
namespace ref {

// Device DRAM as the host sees it: byte 0 of each channel (tsr_sim_dram), the channel count and size.
struct dram_view {
    uint8_t * const * ch;    // ch[c], c < num_ch
    uint32_t num_ch;
    uint64_t ch_bytes;
};

// Status of mulmat_ref.
constexpr int MM_OK = 0;
constexpr int MM_BAD_RECORD = -1;   // arguments the kernel rejects (tsr::opl::RET_BAD_RECORD)
constexpr int MM_RANGE = -2;        // a row outside the DRAM view (the device would fail with DRAM_RANGE)

// Applies one MUL_MAT record (args = the 12 OPL argument words, SW §7.2) to the DRAM view, as the
// kernel does. threads > 1 splits the rows of each token over host threads (results identical).
int mulmat_ref(const dram_view & d, const uint32_t args[12], uint32_t threads = 1);

// One token: y[n] = (add ? tsr_fadd(add[n], dot_n) : dot_n) for n < N, the dot products of x (K f32)
// against the N weight rows at W_OFF of the view. y holds N words (no tail).
void mulmat_ref_token(const dram_view & d, uint32_t fmt, uint32_t N, uint32_t K, uint64_t w_off, const float * x,
                      const float * add, float * y, uint32_t threads = 1);

}  // namespace ref
}  // namespace tsr
