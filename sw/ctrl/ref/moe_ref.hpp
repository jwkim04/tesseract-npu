// moe_ref.hpp — host C++ references of the firmware's MoE kinds MOE_ROUTE (0x1E) and MUL_MAT_ID (0x1F)
// (docs/design/06-software-verification.md "SW" §7.5, SW-60 to SW-65): they compose the tsr_ref calls the
// kernels' tile ops execute, in the kernels' order, so the kernel tests compare the firmware against
// them bit for bit (the one-token selection is tsr_ref_moe_route of sw/sim/tsr_ref_vu.c).
//
// moe_route_ref, per token t of the record (tsr::opl::moer words): the n_expert logits at
// LOGITS + t * LOGITS_STRIDE lines give the k expert numbers and weights of tsr_ref_moe_route. The ids
// go to the first line of row t of the IDS tensor (IDS + t * IDS_STRIDE lines): words 0 .. k-1 the
// numbers, words k .. 15 zero (the rest of the row is untouched). The weights go packed to W: word
// t * k + j. Tiles of 64 tokens store whole lines, so the words after the last token's weights up to
// the end of its line are zero.
//
// mul_mat_id_ref, per (token t, slot j) of the record (tsr::opl::mmid words): e = the I32 at IDS + t *
// IDS_STRIDE bytes + 4 j; y = the MUL_MAT product of the x row (X + t * X_TOK_STRIDE + j * X_SLOT_STRIDE
// lines) with the weights at W_OFF + e * EXP_STRIDE beats (mulmat_ref_token), then y = FM(y, SCALE[e]) with
// a scale table; the y row (N words, zero tail to a whole line) is row t * N_USED + j of the dst tensor
// (DST + row * DST_STRIDE lines). An id outside [0, N_EXPERT) is MM_RANGE (the device fails with
// DRAM_RANGE on that MVM).
#pragma once

#include <cstdint>

#include "mulmat_ref.hpp"

namespace tsr {
namespace ref {

// Applies one MOE_ROUTE record (9 words) to the DRAM view. MM_OK, MM_BAD_RECORD or MM_RANGE.
int moe_route_ref(const dram_view & d, const uint32_t args[9]);
// Applies one MUL_MAT_ID record (17 words) to the DRAM view. threads as mulmat_ref.
int mul_mat_id_ref(const dram_view & d, const uint32_t args[17], uint32_t threads = 1);

}  // namespace ref
}  // namespace tsr
