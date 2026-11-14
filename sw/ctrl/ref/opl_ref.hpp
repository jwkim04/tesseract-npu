// opl_ref.hpp — host reference executor of an op-list blob (docs/design/06-software-verification.md "SW" §7.2, §7.3).
//
// opl_exec runs a list on a host DRAM image the way the firmware's `graph` kernel defines its result: the preload entries are
// copied from DRAM into the preload area first, then every record runs in list order through the per-kind reference that the
// firmware tests compare against bit for bit (mulmat_ref, elem_ref, rmsnorm_ref, rope_ref, set_rows_kv_ref, flash_attn_ref).
// JOIN and MUL_MAT_END do nothing (the order of the list already holds every dependency) and MUL_MAT_BEGIN is MUL_MAT. A kind
// without a reference here is an error, not a skip.
//
// It uses no simulator, no firmware and no device: the result is what the composition of the references gives, so a device (or the
// simulator) that equals it bit for bit ran the list as it is defined. The references are in this library (tsr_opl_ref).
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "mulmat_ref.hpp"   // dram_view

namespace tsr {
namespace ref {

constexpr int OX_OK = 0;
constexpr int OX_BAD_BLOB = -100;     // the blob or one of its records fails the host reader's checks (opl::reader)
constexpr int OX_NO_REF = -101;       // a kind with no reference here
constexpr int OX_RECORD = -102;       // the kind's reference refuses the record's words (its status is in the text)
constexpr int OX_RANGE = -103;        // a handle or a preload source outside the DRAM view

// The device parameters the references take: the planners' T_MAX, PCAP and K_MAX for the attention chunk planner (the channel
// count is the view's). mm_threads: threads of mulmat_ref (the result does not depend on it).
struct opl_exec_cfg {
    uint32_t t_max = 8;
    uint32_t pcap = 9344;
    uint32_t k_max = 32768;
    uint32_t mm_threads = 1;
};

struct opl_exec_stats {
    uint32_t records = 0;                 // records run (JOIN and MUL_MAT_END included)
    uint32_t by_kind[64] = {};            // per kind
};

// Runs the list. d: the DRAM image (every handle of the list is a {channel, beat} of it). Returns OX_OK or an OX_* code; *why
// names the record (index and kind) and the reason. The image is written as the device would write it. *st: optional counts.
int opl_exec(const uint8_t * blob, size_t bytes, const dram_view & d, const opl_exec_cfg & cfg, std::string * why = nullptr,
             opl_exec_stats * st = nullptr);
inline int opl_exec(const std::vector<uint8_t> & blob, const dram_view & d, const opl_exec_cfg & cfg, std::string * why = nullptr,
                    opl_exec_stats * st = nullptr) {
    return opl_exec(blob.data(), blob.size(), d, cfg, why, st);
}

}  // namespace ref
}  // namespace tsr
