// rope_ref.hpp — host C++ reference of the firmware's ROPE kind (OPL v1 kind 0x16; docs/v2/
// 06-software-verification.md "SW" §7.2, §7.5 and its last paragraph): the tsr_ref_vu ROPE call the
// kernel's tile op executes (VU §5.4, §6.7), on the same operands, so the kernel tests compare the
// firmware against it bit for bit.
//
// The kernel (sw/ctrl/kernels/k_rope.cpp) runs, per token tile of a record (tsr::opl::rope argument
// words): LD x (the tile's tokens, H = n_heads * head_dim / 16 lines each, dense in SPM), [LD the
// freq-factor or A table, ceil(n_dims / 32) lines], SOP.MOVI of TS (recurrence form), M and
// freq_scale (only when it is not 1.0f), one ROPE over the tile's tokens in place (positions: the I32
// words at the record's preload byte offset, read by the VU), ST. ROPE's word-0 flags: NEOX = mode
// bit0, ATAB = mode bit1, FF = a table without ATAB, FS = freq_scale != 0x3F800000 without ATAB
// (FM(1.0f, phi) = phi for every VU angle, so FS off is exact). Token tiles do not change any value.
#pragma once

#include <cstddef>
#include <cstdint>

#include "mulmat_ref.hpp"     // dram_view
#include "rmsnorm_ref.hpp"    // NR_* status codes
#include "tsr_ref_vu.h"

namespace tsr {
namespace ref {

// The tsr_ref_rope configuration of a ROPE record (args = the 14 OPL words) for T tokens.
tsr_ref_rope_cfg rope_cfg_of(const uint32_t args[14], uint32_t T);

// Applies one ROPE record to the DRAM view as the kernel does. pre / pre_bytes: the list's preload
// area as the executor loaded it (the positions are I32 words at args[POS]). Every source token is
// read before any output token is written; output tokens are H lines (heads are whole lines).
int rope_ref(const dram_view & d, const uint32_t args[14], const uint8_t * pre, uint32_t pre_bytes);

}  // namespace ref
}  // namespace tsr
