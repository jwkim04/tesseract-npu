// tsr-lower.h — MUL_MAT lowering v1 (spec "ggml 백엔드 통합", lowering (v1)).
//
// Pure instruction generation: given resolved DRAM addresses (IO region on ch0 for the f32
// activations, symmetric W_OFF for the weight) this emits, per T tile,
//
//     LD    ch0  spm_x  <- x_beat + t0*x_lines,  tt*x_lines lines
//     QACT  kind spm_qa <- spm_x, K, tt, src_stride = x_lines
//     MVM   fmt  spm_y  <- spm_qa, W_OFF, N, K, tt, dst_stride = y_lines
//     ST    ch0  y_beat + t0*y_lines <- spm_y, tt*y_lines lines
//
// The SPM is partitioned [x tile | QA tile | y tile]; every operand starts on a line and no
// two operands share a line (spec SPM 규칙). Also the byte-bounded program cache
// (tsr_prog_cache). No ggml dependency.
#pragma once

#include <cstddef>
#include <cstdint>
#include <list>
#include <string>
#include <unordered_map>
#include <vector>

#include "tsr_isa.h"

// SPM tiling of one MUL_MAT.
struct tsr_mm_tiling {
    uint32_t kind;       // QA kind matching fmt (Q8_0 -> A32, Q4_K -> A256)
    uint32_t x_lines;    // one f32 src1 row: 4K/64 lines (K % 32 == 0, so exact)
    uint32_t qa_lines;   // QA_LINES(kind, K)
    uint32_t y_lines;    // ceil(4N/64): MVM dst_stride and the per-token ST length
    uint32_t t_tile;     // tokens per tile, 1 <= t_tile <= min(T, T_MAX)
    uint32_t spm_x;      // SPM line bases of the three regions
    uint32_t spm_qa;
    uint32_t spm_y;
    uint32_t spm_end;    // first line after the y region (<= SPM lines)
};

// Effective per-MVM token limit: min(cfg.t_max, TSR_T_MAX_ARCH) (and the 4-bit w5 T field).
uint32_t tsr_mm_t_max(const tsr_config & cfg);

// Lines one token needs in SPM (x + QA + y). 0 if the shape is not lowerable.
uint64_t tsr_mm_lines_per_token(uint32_t fmt, uint32_t N, uint32_t K);

// Can a MUL_MAT with this device format and shape be lowered at all (independent of T)?
// Checks fmt, K (multiple of the block size and of 32, <= K_MAX, fits the 16-bit K field),
// N >= 1, and that one token fits in SPM.
bool tsr_mm_shape_ok(const tsr_config & cfg, uint32_t fmt, uint64_t N, uint64_t K);

// Compute the tiling for T tokens. Returns false if the shape is not lowerable.
bool tsr_mm_plan_tiling(const tsr_config & cfg, uint32_t fmt, uint32_t N, uint32_t K, uint32_t T,
                        tsr_mm_tiling * out);

// One lowered MUL_MAT with resolved addresses.
struct tsr_mm_desc {
    uint32_t fmt;
    uint32_t N;
    uint32_t K;
    uint32_t T;
    uint64_t w_off_beat;   // W_OFF in beats (same on every channel, symmetric allocation)
    uint64_t x_beat;       // ch0 beat of src1 token 0; token t at x_beat + t * x_lines
    uint64_t y_beat;       // ch0 beat of the output for token 0; token t at y_beat + t * y_lines
};

// Append the instructions of one MUL_MAT (no END). After each tile the instruction count is
// appended to `cuts`: execution may be split at those points (no SPM state crosses a tile).
// Aborts if the descriptor is not lowerable (callers check with tsr_mm_shape_ok first).
void tsr_mm_emit(const tsr_config & cfg, const tsr_mm_desc & d,
                 std::vector<tsr_insn> & insns, std::vector<size_t> & cuts);

// One-line disassembly for TSR_DEBUG output.
std::string tsr_insn_str(const tsr_insn & in);

// A lowered program (one segment of MUL_MATs, without END) and where its last copy sits in
// the CMD region.
struct tsr_prog {
    std::vector<tsr_insn> insns;  // without END
    std::vector<size_t>   cuts;   // safe split points (instruction counts)
    int      half = -1;           // CMD half holding this program, if resident
    uint64_t seq  = 0;            // write sequence number of that copy
    uint64_t uses = 0;
};

// Program cache (spec lowering (v1): "그래프 모양과 텐서 주소가 같으면 ... 캐시해 재사용").
// Keyed by the resolved descriptors of a segment. Bounded by the bytes the programs (and their
// keys) occupy and by an entry count; when an insertion would exceed either bound the least
// recently used entries are evicted. A program larger than max_entry_bytes is never cached:
// with T_MAX = 1 a prefill program grows with the token count (4 instructions per token per
// MUL_MAT), is rarely reused, and re-emitting it costs nothing next to running it.
class tsr_prog_cache {
public:
    tsr_prog_cache(size_t max_bytes, size_t max_entry_bytes, size_t max_entries);

    // The cached program for `key` (marked most recently used), or nullptr.
    tsr_prog * find(const std::string & key);

    // Accounted size of a program under `key`.
    static size_t cost(const std::string & key, const tsr_prog & p);

    // Whether insert() would keep this program (cost <= max_entry_bytes).
    bool cacheable(const std::string & key, const tsr_prog & p) const;

    // Insert (replacing an entry with the same key), evicting LRU entries first so that the
    // byte and entry bounds hold afterwards. Requires cacheable(key, p). Returns the entry.
    tsr_prog * insert(const std::string & key, tsr_prog && p);

    void     clear();
    size_t   bytes()     const { return bytes_; }
    size_t   size()      const { return map_.size(); }
    uint64_t evictions() const { return evictions_; }
    size_t   max_bytes() const { return max_bytes_; }

private:
    struct entry {
        tsr_prog prog;
        size_t   cost = 0;
        std::list<std::string>::iterator lru;
    };
    void evict_one();

    size_t   max_bytes_;
    size_t   max_entry_bytes_;
    size_t   max_entries_;
    size_t   bytes_ = 0;
    uint64_t evictions_ = 0;
    std::unordered_map<std::string, entry> map_;
    std::list<std::string> lru_;   // front = most recently used
};
