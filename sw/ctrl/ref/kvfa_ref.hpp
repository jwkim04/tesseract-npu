// kvfa_ref.hpp — host C++ references of the firmware's SET_ROWS_KV and FLASH_ATTN kinds (docs/v2/
// 06-software-verification.md "SW" §7.5 last paragraph; docs/v2/03-attention-kv.md "AT" §5.1, §6.6,
// §13.1; W2b P2.7): they compose the tsr_ref calls the kernels' tile ops execute, in the kernels'
// order, from an OPL record's argument words, so the kernel tests compare the firmware against them
// bit for bit. sw/ctrl/ref/kv_ref.cpp and fattn_ref.cpp; the backend's tests (W3b) can use them too.
//
// SET_ROWS_KV (sw/ctrl/kernels/k_kv.cpp): for each row i in order, x = the f32 row at SRC + i·SRC
// stride lines (E values), qa = tsr_qact(A32, x, E), tsr_ref_kvst(which, {E, C, LG}, plane, idx[i],
// 1, qa) (the kernel's runs of consecutive cells, QACT grouping and KVST n_rows place the same
// bytes; a cell written twice keeps the later row, as the DMA unit's queue order).
//
// FLASH_ATTN (sw/ctrl/kernels/k_fattn.cpp): tsr_ref_fa (the normative sequence of AT §6.6, W1a's
// head-subsets-inside-chunks order and the chunk planner sw/sim/include/tsr_fa_plan.h) with the
// record's inputs: T = n_tokens, D, G, scale = word 14's bits; q rows (n_head·D f32 each) at Q +
// t·q token stride lines; the K / V planes of the record; the MSUM of the preload area; the f32
// mask copy at word 13 (rows of ceil(n_kv/16) lines; NULL when word 13 is 0xFFFF_FFFF); new_runs =
// the FA_NEW indices as runs of consecutive cells in index order (plan_attn.hpp); C, T_MAX, PCAP and
// K_MAX of the device; the output rows (n_head·D f32 each) written at DST + t·dst stride lines.
// The chunk planner is shared, not independent: tsr_ref_fa and k_fattn.cpp compile the same
// tsr_fa_plan.h, so the bit-exact FA tests (fw_kvfa_rv_*, fw_kvfa_host, fw_m7_layer_*) cannot see a
// planner error (a chunk bound or new-cell rule off by one gives the same wrong result on both
// sides). Its independent checks are fw_kvfa_plan (W1a's frozen planner as the oracle, AT §6.6 rules
// 1-8) and fw_kvfa_kl (ggml-cpu's flash_attn_ext, LLAMA builds, gates near the numerical floor).
//
// Memory: the record's DRAM rows are read from / written to a dram_view (mulmat_ref.hpp); the planes
// are passed as one pointer per channel to their image (byte 64·B of the channel, AT §4.2;
// tsr_ref_at.h conventions), so a test can apply the records to host copies of the planes; the
// preload area is a host image of its bytes (pre[0] = the area's first byte).
#pragma once

#include <cstddef>
#include <cstdint>

#include "mulmat_ref.hpp"

extern "C" {
#include "tsr_ref_at.h"
}

namespace tsr {
namespace ref {

constexpr int KV_OK = 0;
constexpr int KV_BAD_RECORD = -1;   // arguments or indices the kernel refuses (RET_BAD_RECORD)

// SET_ROWS_KV (12 argument words) onto plane[c] (the record's plane: K or V as word 7 says). src rows
// from d. pre / pre_bytes: the preload area. C = NUM_CH.
int set_rows_kv_ref(const dram_view & d, const uint32_t args[12], const uint8_t * pre, size_t pre_bytes,
                    uint8_t * const plane[], uint32_t C);

// FLASH_ATTN (22 argument words): the output rows into d (DST rows), from q rows / the mask copy in d,
// the planes kplane / vplane and the preload area. Returns tsr_ref_fa's code (TSR_FA_OK = 0, ...), or
// KV_BAD_RECORD for words the kernel refuses before the reference could (softcap, head_dim vs D,
// n_head vs G, strides, MSUM header, flags). stats may be NULL.
int flash_attn_ref(const dram_view & d, const uint32_t args[22], const uint8_t * pre, size_t pre_bytes,
                   const uint8_t * const kplane[], const uint8_t * const vplane[], const tsr_att_cfg & cfg,
                   tsr_fa_stats * stats = nullptr);

// The f32 output only (T rows of n_head·D), without touching d (out: T·n_head·D floats).
int flash_attn_ref_out(const dram_view & d, const uint32_t args[22], const uint8_t * pre, size_t pre_bytes,
                       const uint8_t * const kplane[], const uint8_t * const vplane[], const tsr_att_cfg & cfg,
                       float * out, tsr_fa_stats * stats = nullptr);

}  // namespace ref
}  // namespace tsr
