// tsr-opl.h — the host side of the op-list path (docs/design/06-software-verification.md "SW" §9.3
// M6 (P3.2), §9.4; op list format SW §7.2; the `graph` kernel SW §7.1, §7.3; launch protocol EX §4,
// transport API EX §10.1 + SW §9.1). Part of the tsr_lower library, shared by the ggml backend
// (ggml-tesseract.cpp) and the ggml-free runtime (sw/rt). No ggml dependency.
//
//   tsr_opl_builder   one op list record per op, appended in program order, with dependencies from
//                     a hazard analysis over DRAM byte ranges (SW §9.3 step 1): op k depends on an
//                     earlier op i iff k reads what i writes (RAW), k writes what i reads (WAR) or
//                     k writes what i writes (WAW). The conflicting set is transitively reduced
//                     (an op already implied through another kept predecessor is dropped) and more
//                     than two predecessors are folded by JOIN records, as SW §7.2 prescribes and
//                     tsr::opl::writer::add does. Encoding is tsr/opl.hpp's writer (the records,
//                     header, CRC); the header's graph_id is the list's CRC, so equal lists are
//                     equal blobs.
//   tsr_opl_cache     an LRU of op lists in the OPL region (ch0 [0, 64 MiB), EX §11) keyed by
//                     content (64-bit FNV-1a, confirmed by a byte compare against the host copy):
//                     a list seen before is not written again (SW §9.3 step 3, §9.4).
//   tsr_opl_dev       the `graph` kernel on an open transport: boot (warm if possible, EX §3.5) with
//                     $TSR_IMAGE or the build's image unless the transport is booted already, the
//                     kernel index, and run = place in the cache, LAUNCH {a0 handle, a1 bytes, a2
//                     flags}, wait with the transport's debug-aware timeout (SW §9.3 step 4). The
//                     caller maps the answer to its ops (tsr_opl_describe; step 5).
//
// MUL_MAT records (SW §7.2 kind 0x10): the args are tsr::opl::mm's words; handles are {ch[31:28],
// beat[27:0]} of 64-byte beats; strides are in lines. Tiling is the firmware's (plan.hpp, SW-20),
// so a record carries the whole T; the host only checks with the same plan.hpp that the firmware can
// plan it (supports_op). In an M6 list (MUL_MAT records only) every DRAM access of a record but the
// weight stream is an LD or ST on the one DMA queue, whose order already orders the records; the
// dependencies are still part of the list (the OPL contract, SW §7.2) and order the units once M7's
// kinds put VU / AT work between them.
//
// M7 records (W3b, SW §7.2 kinds 0x11-0x1B; the backend's tsr-build.h lowers ggml nodes to them):
// kind_ranges gives the DRAM ranges each kind reads and writes from its argument words (whole lines of
// every row operand, row periods, KV planes on every channel, FLASH_ATTN's f32 mask copy), which feed
// the same hazard analysis; preload() adds the list's preload entries (SW §7.2 "Preload entry",
// SW-15: positions, ids, KV indices, mask summaries), one per (source, size), and refuses (NO_ROOM)
// an entry that would take the preload area beyond 256 lines, so the caller cuts the list there.
//
// Files: tsr-opl.cpp (ranges, builder, cache; no device calls, so the unit tests link without one)
// and tsr-opl-dev.cpp (tsr_opl_dev, tsr_opl_describe: the transport calls).
#pragma once

#include <cstddef>
#include <cstdint>
#include <list>
#include <string>
#include <unordered_map>
#include <vector>

#include "tsr/opl.hpp"
#include "tsr/plan.hpp"

#include "tsr-alloc.h"
#include "tsr-wlayout.h"     // tsr_dram_write_fn
#include "tsr_isa.h"
#include "tsr_transport.h"

// ---------------------------------------------------------------------------------------------
// DRAM ranges and handles
// ---------------------------------------------------------------------------------------------
// Bytes of channel ch: the extent [lo, hi), either all of it (row = 0) or rows of `row` bytes
// every `stride` bytes from lo (row < stride, the last row ending at hi): T token rows with a
// stride larger than the row, e.g. per-token rows interleaved with another op's (tsr_rows_range).
struct tsr_dram_range {
    uint32_t ch;
    uint64_t lo, hi;
    uint64_t row = 0;
    uint64_t stride = 0;
};

// n rows of `row` bytes, `stride` bytes apart, from byte lo of channel ch (stride 0: the same row
// n times). Contiguous when the rows touch or overlap.
tsr_dram_range tsr_rows_range(uint32_t ch, uint64_t lo, uint64_t n, uint64_t row, uint64_t stride);

// Do a and b share a byte? Exact for contiguous and strided ranges (O(1) against a contiguous
// one, O(rows of the shorter) between two strided ones; above 65,536 rows the extents decide,
// which can only add a dependency).
bool tsr_ranges_overlap(const tsr_dram_range & a, const tsr_dram_range & b);

// DRAM handle of a 64-byte-aligned byte offset (EX §4.3: {ch[31:28], beat[27:0]}).
uint32_t tsr_dram_handle(uint32_t ch, uint64_t byte_off);
inline uint32_t tsr_handle_ch(uint32_t h)        { return h >> 28; }
inline uint64_t tsr_handle_byte(uint32_t h)      { return (uint64_t) (h & 0x0FFFFFFFu) * TSR_BEAT_BYTES; }

// The planner's view of the device (plan.hpp dev_cfg from the CFG CSRs the transport read).
tsr::plan::dev_cfg tsr_plan_cfg(const tsr_config & cfg);

// ---------------------------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------------------------
class tsr_opl_builder {
public:
    static constexpr uint32_t NO_ROOM = 0xFFFFFFFEu;   // add(): the list is full, flush and retry
    static constexpr uint32_t FAILED  = 0xFFFFFFFFu;   // add(): writer error (error())

    struct op_info {
        uint32_t kind;
        int32_t  tag;                      // caller's tag (the backend: node index); -1 for JOIN
        uint32_t dep0, dep1;               // as encoded (op indices or DEP_NONE)
        std::vector<tsr_dram_range> rd, wr;
    };

    // num_ch: channels the weight of a MUL_MAT record spans (its read range on every channel).
    explicit tsr_opl_builder(uint32_t num_ch = 1, uint32_t flags = tsr::opl::F_STOP_ON_ERROR);

    void reset();                         // empty list, same flags
    void set_flags(uint32_t f)            { flags_ = f; }
    uint32_t flags() const                { return flags_; }

    // One op reading rd and writing wr (byte ranges). Returns its op index, NO_ROOM if the record
    // and its JOINs would exceed 4096 ops (SW-29; nothing is added), or FAILED.
    uint32_t add(uint32_t kind, const uint32_t * args, uint32_t n_args, const std::vector<tsr_dram_range> & rd,
                 const std::vector<tsr_dram_range> & wr, int32_t tag);

    // A MUL_MAT record (args: tsr::opl::mm words); its ranges come from the args (mm_ranges).
    uint32_t mul_mat(const uint32_t * args, int32_t tag, uint32_t kind = tsr::opl::K_MUL_MAT);

    // The ranges a MUL_MAT record reads and writes: x (T rows, x_stride lines apart, K/16 lines
    // each), the addend (stride 0: one row) or, with ACC_DST, y; the weight's rows on every channel
    // (W_OFF + the per-channel size); y written (T rows of ceil(N/16) lines, y_stride apart: the ST
    // writes whole lines, so a T = 1 row's tail line padding is included). Rows further apart than
    // their length are strided ranges (the gaps are not accessed).
    void mm_ranges(const uint32_t * args, std::vector<tsr_dram_range> & rd, std::vector<tsr_dram_range> & wr) const;

    // M7 (W3b, SW §9.3 builder pass 4): the DRAM ranges a record of any kind reads and writes, from
    // its argument words (SW §7.2) the way its kernel accesses them (SW §7.5): every row operand as
    // rows of whole lines (L(n) = ceil(n/16) lines; the kinds LD / ST whole lines, so a row's tail
    // padding is included), a row stride of 0 as one row, a row period p as min(p, rows) rows; ADD /
    // MUL's FLAG_B_SCALAR vector as its words' lines; RMS_NORM's weight rows (dense, L(n) lines
    // apart, SW §7.2); ROPE's table (L(n_dims/2) lines); SET_ROWS_KV writes, and FLASH_ATTN reads, the
    // whole KV planes on every channel (the cells are data: AT §4.5's plane, tsr_kv_plane_beats of
    // {E, C, LG} from GEOM); FLASH_ATTN reads its f32 mask copy (word 13; n_tokens rows of
    // L(n_kv) lines) when there is one. The preload area is not a DRAM access of a record (the
    // executor loads it before op 0, SW §7.3). aux: GET_ROWS's source rows (its ids are data, so
    // every source row may be read; 0: unknown, the source is taken as one row per id, which the
    // caller must not rely on). MUL_MAT: mm_ranges. NOP / JOIN / unknown kinds: nothing.
    void kind_ranges(uint32_t kind, const uint32_t * args, uint64_t aux, std::vector<tsr_dram_range> & rd,
                     std::vector<tsr_dram_range> & wr) const;

    // Preload entries (SW §7.2, SW-15): the pre_byte of an entry of `bytes` bytes from DRAM handle
    // `src`, added unless an entry with the same (src, bytes) is in the list already (positions,
    // KV indices and a mask's MSUM are shared by every record that reads them). NO_ROOM when a new
    // entry would take the preload area beyond 256 lines (nothing is added; the caller runs the
    // list and retries in a new one); FAILED for an empty entry.
    uint32_t preload(uint32_t src, uint32_t bytes);
    // Lines the preload area would have after adding the entries (src, bytes) not in it yet.
    uint32_t preload_lines_with(const std::vector<std::pair<uint32_t, uint32_t>> & entries) const;
    uint32_t pre_lines() const            { return w_.pre_lines(); }
    uint32_t n_pre() const                { return (uint32_t) pre_.size(); }

    bool        empty() const             { return ops_.empty(); }
    uint32_t    n_ops() const             { return (uint32_t) ops_.size(); }
    uint32_t    n_joins() const           { return n_joins_; }
    uint32_t    n_deps() const            { return n_deps_; }        // predecessors kept (after reduction)
    uint32_t    n_dropped() const         { return n_dropped_; }     // conflicts implied transitively
    const op_info & op(uint32_t i) const  { return ops_[i]; }
    int32_t     tag_of(uint32_t i) const  { return i < ops_.size() ? ops_[i].tag : -1; }
    bool        ok() const                { return w_.ok(); }
    const char * error() const            { return w_.error(); }

    // Does a pending op write (any_access: or read) bytes of r?
    bool touches(const tsr_dram_range & r, bool any_access) const;

    // The blob (header, op area, CRC; graph_id = the CRC of bytes [64, total)). Empty if !ok() or
    // the list is empty.
    std::vector<uint8_t> finish() const;

private:
    bool is_ancestor(uint32_t a, uint32_t b) const;   // a is an ancestor of b (a < b)
    uint32_t push_record(uint32_t kind, const uint32_t * args, uint32_t n_args, uint32_t d0, uint32_t d1, int32_t tag);

    uint32_t num_ch_;
    uint32_t flags_;
    tsr::opl::writer w_;
    std::vector<std::pair<std::pair<uint32_t, uint32_t>, uint32_t>> pre_;   // ((src, bytes), pre_byte)
    std::vector<op_info> ops_;
    std::vector<std::vector<uint64_t>> anc_;   // ancestor bitsets (bit i: op i), per op
    uint32_t n_joins_ = 0, n_deps_ = 0, n_dropped_ = 0;
};

// ---------------------------------------------------------------------------------------------
// OPL cache
// ---------------------------------------------------------------------------------------------
class tsr_opl_cache {
public:
    struct stats {
        uint64_t hits = 0, misses = 0, evictions = 0, bytes_written = 0;
    };

    // [base, base + size) of channel `ch`, 64-byte aligned; at most max_entries lists.
    void init(uint32_t ch, uint64_t base, uint64_t size, size_t max_entries = 4096);

    // The DRAM byte offset of `blob` in the region: found (hit) or written now with `write`
    // (evicting the least recently used lists until it fits). False if the blob is larger than the
    // region. write may throw / abort; the cache is then unchanged.
    bool place(const std::vector<uint8_t> & blob, const tsr_dram_write_fn & write, uint64_t * off, bool * hit);

    void   clear();                      // forget every list (the DRAM contents are no longer known)
    size_t entries() const               { return lru_.size(); }
    uint64_t used_bytes() const          { return alloc_.used(); }
    uint64_t region_bytes() const        { return alloc_.total(); }
    uint32_t channel() const             { return ch_; }
    const stats & get_stats() const      { return st_; }
    void   reset_stats()                 { st_ = stats(); }

    static uint64_t hash(const uint8_t * p, size_t n);   // FNV-1a 64

private:
    struct entry {
        uint64_t key;
        uint64_t off;
        std::vector<uint8_t> blob;
    };
    void evict_lru();

    uint32_t ch_ = 0;
    size_t max_entries_ = 4096;
    tsr_range_alloc alloc_;
    std::list<entry> lru_;                                            // front = most recently used
    std::unordered_multimap<uint64_t, std::list<entry>::iterator> map_;
    stats st_;
};

// ---------------------------------------------------------------------------------------------
// The `graph` kernel on a transport
// ---------------------------------------------------------------------------------------------
struct tsr_opl_run_result {
    int        rc = 0;          // tsr_transport_call's return value (0: answered; TSR_TE_*)
    tsr_result res = {};        // decoded answer (valid when rc == 0)
    bool       too_big = false; // the list does not fit the OPL region (rc = TSR_TE_INVAL, not launched)
    bool       cache_hit = false;
    uint64_t   opl_off = 0;     // ch0 byte offset of the list
    uint64_t   wall_ns = 0;     // launch + wait
};

class tsr_opl_dev {
public:
    tsr_opl_dev() = default;
    tsr_opl_dev(const tsr_opl_dev &) = delete;
    tsr_opl_dev & operator=(const tsr_opl_dev &) = delete;

    // Boots tp (warm if possible) with $TSR_IMAGE / the build's image unless it already runs an image,
    // finds the `graph` kernel, reads the versions of the device and the booted image (SW §6.9) and
    // sets up the OPL cache over the OPL region. False with *err.
    bool open(tsr_transport * tp, std::string * err);
    bool is_open() const                 { return tp_ != nullptr; }

    // Places the list in the cache and runs it through `graph` (a2 = launch flags, tsr::opl::LF_*).
    // r.rc != 0: the transport failed or timed out (the device is diagnosed by the transport).
    tsr_opl_run_result run(const std::vector<uint8_t> & blob, uint32_t launch_flags, const tsr_dram_write_fn & write);

    tsr_opl_cache & cache()              { return cache_; }
    const tsr_opl_cache & cache() const  { return cache_; }
    uint64_t image_id() const            { return image_id_; }
    const char * image_path() const      { return image_path_.c_str(); }
    int      graph_kernel() const        { return k_graph_; }
    // Tesseract Core capability and CAPS, the image's Tesseract Basis version, ABI and kinds (SW §6.9).
    const tsr_versions & versions() const { return versions_; }

    uint64_t launches = 0;               // graph launches answered or failed
    uint64_t launch_ns = 0;              // their wall time (launch + wait)
    uint64_t ops = 0;                    // op records launched (JOINs included)

private:
    tsr_transport * tp_ = nullptr;
    int k_graph_ = -1;
    uint64_t image_id_ = 0;
    std::string image_path_;
    tsr_versions versions_ = {};
    tsr_opl_cache cache_;
};

// SW §9.3 step 5: the op a failed launch names. A TILE_ERR answer of `graph` carries RET = the
// failing record's index + 1 (SW §6.4, §7.3): tsr_opl_failed_op gives that index, or -1 when r is
// not an answered TILE_ERR or RET is not a record of a list of n_ops; tsr_opl_failed_tag gives the
// tag b recorded for it (the backend: its node index), -1 for a JOIN or no op. The one place this
// arithmetic lives (the backend and sw/rt both name the failing op with it).
int32_t tsr_opl_failed_op(const tsr_opl_run_result & r, uint32_t n_ops);
int32_t tsr_opl_failed_tag(const tsr_opl_builder & b, const tsr_opl_run_result & r);

// One line describing a failed graph launch for the user: the status, the executor's return
// code or the failing op (ret - 1, with `op_name` when the caller knows it), the first error's
// code and ERR_INFO / ticket / w0 / push index (EX §4.4).
std::string tsr_opl_describe(const tsr_opl_run_result & r, const char * op_name);
const char * tsr_opl_ret_name(uint32_t ret);      // executor codes 0x8000_000x (SW §7.3)
const char * tsr_err_code_name(uint32_t code);    // op error codes (EX §8.1)
