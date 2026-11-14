// tsr-rec.h — the record of the op-list builder and its encoding, without ggml
// (docs/design/06-software-verification.md "SW" §7.2 OPL kinds and their argument words, §9.3 builder passes 4-6).
//
//   tsr_rec         one lowered record before encoding: its kind, argument words, preload references, the
//                   KV plane and the mask words it uses
//   tsr_fanew_map   the FA_NEW source of each KV plane written so far (K plane base beat -> I64 indices)
//   tsr_note_kv_write  fills the map from a SET_ROWS_KV record
//   tsr_encode      passes 4-6 for one record: preload entries (deduplicated, the FA_NEW word of a FLASH_ATTN),
//                   DRAM ranges (tsr_opl_builder::kind_ranges), RAW / WAR / WAW dependencies with transitive
//                   reduction and JOINs (tsr_opl_builder::add)
//
// The ggml lowering (tsr-build.h) and the framework-free encoders (tsr-kinds.h) both produce tsr_rec.
#pragma once

#include <cstdint>
#include <map>
#include <utility>
#include <vector>

#include "tsr-msum.h"
#include "tsr-opl.h"

// One preload reference of a record: argument word `word` holds the preload byte offset of `bytes`
// bytes from DRAM handle `src` (SW §7.2 "Preload entry").
struct tsr_pre_ref {
    uint32_t word;
    uint32_t src;
    uint32_t bytes;
};

// A lowered record (before encoding).
struct tsr_rec {
    uint32_t kind = tsr::opl::K_NOP;
    uint32_t n_args = 0;
    uint32_t args[tsr::opl::fa::NARGS] = {};   // the longest kind (FLASH_ATTN, 22 words)
    std::vector<tsr_pre_ref> pre;
    uint64_t aux = 0;                          // kind_ranges' aux: GET_ROWS source rows
    // SET_ROWS_KV: its plane and the indices it writes (the FA_NEW of a FLASH_ATTN reading the plane).
    // FLASH_ATTN: its K plane; FA_NEW is resolved by tsr_encode.
    uint32_t kv_plane = 0;
    uint32_t kv_idx_src = 0, kv_idx_n = 0;
    tsr_msum_ref mask;                         // FLASH_ATTN: the mask words used
    // Further records of the same node (one per token of an RMS_NORM whose source tokens are not spaced like its
    // heads, tsr_lower_rms); run in order after this one. Probe mode fills none: all share the first one's plan.
    std::vector<tsr_rec> more;
};

// The FA_NEW source of each KV plane written in this graph_compute: K plane base beat -> (handle of
// the I64 indices, count). Filled by tsr_note_kv_write from the SET_ROWS_KV records in program order.
using tsr_fanew_map = std::map<uint32_t, std::pair<uint32_t, uint32_t>>;
void tsr_note_kv_write(const tsr_rec & r, tsr_fanew_map * m);

// Passes 4-6 for one record: its preload entries (deduplicated; FA_NEW from fanew for a FLASH_ATTN),
// their offsets into its words, its DRAM ranges and its dependencies (b.add). Returns the op index,
// tsr_opl_builder::NO_ROOM (the record, its JOINs or its preload entries do not fit this list:
// nothing was added; run the list and encode into a new one) or FAILED (writer error).
uint32_t tsr_encode(tsr_opl_builder & b, tsr_rec & r, int32_t tag, const tsr_fanew_map & fanew);
