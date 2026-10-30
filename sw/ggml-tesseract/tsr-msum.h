// tsr-msum.h — the mask-summary producer of the ggml-tesseract backend (docs/v2/03-attention-kv.md
// "AT" §7.3 MSUM, §5.5 AMASK and the f32 mask copy, §6.6 the chunk planner; docs/v2/06-software-
// verification.md "SW" §9.3 M7 "Mask summaries", V2-SW-32, V2-SW-15; W3b P3.5).
//
// The device never reads ggml's F16 KQ mask (the VU has no f16, V2-AT-14). For each mask the
// scheduler writes into a Tesseract buffer (set_tensor of a tensor whose name contains "kq_mask":
// llama.cpp's attn_inp_kq_mask and the scheduler's input copies "<backend>#attn_inp_kq_mask#<n>",
// src/llama-graph.cpp:30-46, ggml-backend.cpp:1529), the backend writes next to the mask, inside
// the bytes the buffer type's get_alloc_size reserved for it (tsr_msum_layout):
//   [0, mask_bytes)              the F16 mask (ggml's bytes, unchanged)
//   [msum_off, +msum_bytes)      its MSUM image (AT §7.3: header {'MSUM', n_q, n_kv, flags, f32-mask
//                                handle, 0, 0, 0}, then {lo0, hi0, lo1, hi1} per query row), built by
//                                AT's reference tsr_msum_build_f16 (the runs exactly as it computes
//                                them); the FLASH_ATTN record refers to it through the preload table
//                                (word 12), so the cached op list stays valid while the mask changes
//   [copy_off, +copy_bytes)      the f32 copy AMASK reads (word 13): fdiv(m, scale) per cell, rows of
//                                ceil(n_kv/16) lines, padding -inf (tsr_msum_mask_f32)
//
// Decisions (W3b, recorded in SW §9.3 / §13 by the integration):
//   - Scales: the producer learns the FA scales from the builder. For every FLASH_ATTN that reads a
//     mask, the builder calls attach(mask, {scale, E, D, G}) while it builds (and again on every
//     cached reuse; a known reader costs a lookup). The set of readers of a mask only grows (until
//     the mask's shape changes or its buffer is freed). set_tensor then writes one f32 copy per
//     distinct registered scale that needs one; the first scale's copy is the reserved slot next to
//     the mask, a further scale's copy is a side allocation (alloc_fn). A mask whose values are all
//     +-0 / -inf (every llama.cpp mask, AT §2.4) gives the same copy for every scale (fdiv(+-0, s) =
//     +-0, fdiv(-inf, s) = -inf); only finite nonzero values make the copies differ. A reader
//     attached after the mask was written (the first launch of a new graph: the scheduler writes
//     the inputs before graph_compute) re-runs the producer from the host copy of the mask, so the
//     order does not matter.
//   - When a copy is written: the MSUM is GENERAL, or the shared chunk planner (tsr_fa_plan, the
//     firmware's own source) puts some query tile of some registered reader on the GENERAL path
//     (AT §6.6 rule 3, §7.3: rows with two runs, holes, a wrapped SWA ring in prefill). Otherwise the
//     copy is not written (FAST chunks never read it) and word 13 still names the slot (constant
//     words, so the cached list holds).
//   - Rule-5 refusals (AT §6.6 rule 5 / 6: a FAST tile with T_q' > 1 whose cut cannot be encoded,
//     data-dependent, invisible to supports_op): the producer runs the planner over every query tile
//     of every registered reader; if any tile is refused it marks the MSUM GENERAL (flags bit0; the
//     rows keep the runs, informative) and writes the copy, so every tile plans on the GENERAL path
//     (rule 1) and the executor never refuses a tile for mask data. GENERAL is the AMASK path the
//     firmware and tsr_ref_fa both implement, so the simulator stays the bit-exact oracle (it reads the
//     same MSUM); only the chunk list (hence the f32 rounding) differs from the FAST plan.
//   - A FLASH_ATTN without a mask gets an all-visible MSUM (one run [0, n_kv) per row) from
//     nomask(), a side allocation cached per (n_q, n_kv).
//
// Interface to the builder (ggml-tesseract.cpp, the "W3b kv" sections):
//   static tsr_msum_ref tsr_fa_mask_ref(tsr_device * d, const ggml_tensor * fa);
//       for a FLASH_ATTN_EXT node: attach (scale = op_params[0], E, D, G from its operands) and the
//       words / ranges to use; ref.valid == false when the mask is not summarized (not named
//       *kq_mask*, not F16 [n_kv, n_q, 1, 1], or not in device memory).
//   static bool tsr_fa_mask_supported(const ggml_tensor * fa);
//       supports_op: the FLASH_ATTN_EXT node's mask (src[3]) is one the producer summarizes, with
//       n_kv = K's cells and n_q = Q's tokens (the MSUM's n_kv / n_q are FLASH_ATTN words 7 / 11),
//       or there is none.
//   preload: writer.pre_msum(ref.msum_handle, ref.n_q); FLASH_ATTN word 12 = that pre_byte,
//   word 13 = ref.mask_handle; hazards: the record reads ref.copy_range (the preload reads
//   ref.msum_range at launch start, before every op).
//
// No ggml dependency (part of tsr_lower; unit test tests/test-tsr-kvbuf.cpp). Not thread-safe: the
// backend calls it with its device mutex held.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <vector>

#include "tsr-opl.h"        // tsr_dram_range, tsr_dram_handle
#include "tsr-wlayout.h"    // tsr_dram_write_fn
#include "tsr_ref_at.h"     // tsr_att_cfg, tsr_msum_*

// A tensor name the producer summarizes: contains "kq_mask".
bool tsr_msum_is_mask_name(const char * name);

// The bytes of a mask's allocation (all offsets relative to the mask's first byte, 64-aligned).
struct tsr_msum_layout {
    uint32_t n_kv = 0, n_q = 0;    // mask ne[0], ne[1]
    size_t   row_stride = 0;       // f16 elements between rows (nb[1] / 2)
    uint64_t mask_bytes = 0;       // pad64(ggml nbytes)
    uint64_t msum_off = 0;         // = mask_bytes
    uint64_t msum_bytes = 0;       // pad64(4 * tsr_msum_words(n_q)) (opl::msum_bytes(n_q) rounded up)
    uint64_t copy_off = 0;         // = msum_off + msum_bytes
    uint64_t copy_bytes = 0;       // n_q * 64 * ceil(n_kv / 16)
    uint64_t total = 0;            // copy_off + copy_bytes: the buffer type's get_alloc_size
};
// A summarized mask: tsr_msum_is_mask_name, F16 (is_f16), [n_kv, n_q, 1, 1] with n_kv, n_q >= 1,
// n_kv <= 2^20 (FLASH_ATTN's limit), rows of contiguous f16 (nb[0] = 2, nb[1] >= 2 n_kv, even).
bool tsr_msum_layout_of(const char * name, bool is_f16, const int64_t ne[4], const size_t nb[4], size_t nbytes,
                        tsr_msum_layout * out);

// One FLASH_ATTN that reads a mask: its scale (ggml op_params[0], as f32 bits) and the shape the
// chunk planner needs (E = n_head_kv * D of K, D, G = n_head / n_head_kv).
struct tsr_msum_reader {
    uint32_t scale_bits = 0;
    uint32_t E = 0, D = 0, G = 0;
    bool operator==(const tsr_msum_reader & o) const {
        return scale_bits == o.scale_bits && E == o.E && D == o.D && G == o.G;
    }
};

// What the builder writes into a FLASH_ATTN record.
struct tsr_msum_ref {
    bool     valid = false;
    uint32_t n_q = 0, n_kv = 0;
    uint32_t msum_handle = tsr::opl::HANDLE_NONE;   // preload src (writer.pre_msum(msum_handle, n_q))
    uint32_t mask_handle = tsr::opl::HANDLE_NONE;   // FLASH_ATTN word 13 (always a valid handle)
    tsr_dram_range msum_range = {};                  // the MSUM image's bytes
    tsr_dram_range copy_range = {};                  // this scale's f32 copy
    uint32_t flags = 0;                              // the MSUM written last (GENERAL bit0; informative)
};

class tsr_msum_store {
public:
    // A side allocation of `bytes` (64-aligned) in device memory: channel and byte offset.
    using alloc_fn = std::function<bool (uint64_t bytes, uint32_t * ch, uint64_t * off)>;
    using free_fn  = std::function<void (uint32_t ch, uint64_t off)>;

    // cfg: the device's AT parameters (C = NUM_CH, T_MAX, PCAP = CFG3[31:16] * 16, K_MAX) for the
    // planner; write: host -> device; alloc / release: side allocations (further scales, nomask()).
    void init(const tsr_att_cfg & cfg, tsr_dram_write_fn write, alloc_fn alloc, free_fn release);
    bool ready() const { return (bool) write_; }

    // set_tensor of a mask: `key` identifies the tensor (its virtual address), (ch, off) is the
    // device byte offset of its first byte (the reservation follows it), f16 the whole mask (row q
    // at f16 + q * L.row_stride). Writes the MSUM and the copies the registered readers need. A
    // different layout at the same key starts a new mask (readers and side copies dropped).
    void on_set(uint64_t key, uint32_t ch, uint64_t off, const tsr_msum_layout & L, const uint16_t * f16);

    // The builder: register a reader of the mask at `key` and get its words. Re-produces from the
    // host copy when the reader is new. valid = false if no mask was ever written at key.
    tsr_msum_ref attach(uint64_t key, const tsr_msum_reader & r);

    // An all-visible MSUM of n_q rows and n_kv cells (FLASH_ATTN without a mask).
    tsr_msum_ref nomask(uint32_t n_q, uint32_t n_kv);

    // Forget every mask whose key lies in [lo, hi) (its buffer is freed or reset); side copies are
    // released.
    void forget(uint64_t lo, uint64_t hi);
    void clear();

    // Tests / debug: the last MSUM words written for key (host copy), and which copies are valid.
    bool last_msum(uint64_t key, std::vector<uint32_t> * words) const;
    size_t n_masks() const { return masks_.size(); }

    // Statistics: productions (set_tensor + re-productions), f32 copies written, GENERAL by values /
    // runs, GENERAL by promotion (rule-5 refusals), re-productions triggered by attach.
    uint64_t st_produced = 0, st_copies = 0, st_general = 0, st_promoted = 0, st_reattach = 0;

private:
    struct copy_slot {
        uint32_t scale_bits;
        uint32_t ch;
        uint64_t off;
        bool     side;          // a side allocation (released with the mask)
    };
    struct mask {
        tsr_msum_layout L;
        uint32_t ch = 0;
        uint64_t off = 0;
        std::vector<uint16_t> f16;              // host copy of the mask (rows at L.row_stride)
        std::vector<tsr_msum_reader> readers;   // registered readers (grows)
        std::vector<copy_slot> slots;           // one per distinct scale; slot 0 is the reservation
        std::vector<uint32_t> words;            // the MSUM written last
        bool have_data = false;
    };
    void produce(mask & m);
    const copy_slot & slot_for(mask & m, uint32_t scale_bits);
    tsr_msum_ref ref_of(const mask & m, const copy_slot & s) const;
    void drop_side(mask & m);

    tsr_att_cfg cfg_ = {};
    tsr_dram_write_fn write_;
    alloc_fn alloc_;
    free_fn release_;
    std::map<uint64_t, mask> masks_;
    struct nomask_entry { uint32_t ch; uint64_t off; };
    std::map<std::pair<uint32_t, uint32_t>, nomask_entry> nomask_;
};
