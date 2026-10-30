// tsr/opl.hpp — op list format OPL v1 (docs/v2/06-software-verification.md "SW" §7.2, V2-SW-12,
// V2-SW-15, V2-SW-29): constants, the header / preload / record layouts, the graph executor's
// return codes (SW §7.3), CRC-32/ISO-HDLC, a freestanding record reader and a header-only host
// writer.
//
// Shared by the firmware (freestanding C++20, V2-SW-8: graph.cpp and the kind handlers), the host
// build of the kernels (SW §6.8), the backend's OPL builder (P3.2, C++17) and the tests. Everything
// above the "host writer" section is freestanding and C++17-compatible (no libstdc++, no heap);
// the writer is compiled only in hosted builds (__STDC_HOSTED__ = 1) and uses the standard library.
//
// Blob layout (little-endian, one 64-byte-aligned blob in device DRAM, SW §7.2):
//
//     [0, 64)               header (struct header)
//     [pre_off, +16 n_pre)  preload table (struct pre_entry), 64-aligned start (the writer puts it
//                           right after the header; the executor requires it inside the first
//                           OPL_AREA_LINES lines, SW §7.3)
//     [ops_off, +64 n_slots) op area: records in 64-byte slots, record i starting on a slot
//     total_bytes           = ops_off + 64 n_slots (a multiple of 64)
//
// Record (one or more slots):
//     word0: [7:0] kind, [11:8] n_slots - 1, [15:12] 0, [23:16] kind flags, [31:24] n_args
//     word1: [15:0] dep0, [31:16] dep1   (op indices < own index; 0xFFFF = none)
//     word2..: args[n_args]              (n_args <= 16 n_slots - 2)
// The op index is the record's position in the list (JOIN and NOP records count). An op with more
// than two predecessors is preceded by JOIN records (the writer emits them, add()).
#pragma once

#include <stddef.h>
#include <stdint.h>

namespace tsr {
namespace opl {

// ---------------------------------------------------------------------------------------------
// Constants (SW §7.2, §7.3)
// ---------------------------------------------------------------------------------------------
constexpr uint32_t MAGIC          = 0x4C504F54u;  // "TOPL" as a little-endian u32
constexpr uint32_t VERSION        = 1u;
constexpr uint32_t HDR_BYTES      = 64u;
constexpr uint32_t SLOT_BYTES     = 64u;
constexpr uint32_t SLOT_WORDS     = 16u;
constexpr uint32_t MAX_OPS        = 4096u;        // V2-SW-29: records per list (DTCM tables)
constexpr uint32_t MAX_REC_SLOTS  = 16u;          // word0 [11:8] = n_slots - 1
constexpr uint32_t PRE_ENTRY_BYTES = 16u;
constexpr uint32_t PRE_ALIGN      = 64u;          // pre_byte alignment
constexpr uint32_t PRE_MAX_LINES  = 256u;         // SW §7.3: preload area <= 256 lines
constexpr uint32_t DEP_NONE       = 0xFFFFu;
constexpr uint32_t HANDLE_NONE    = 0xFFFFFFFFu;  // "no tensor" in an argument word (addend, weight, ...)

// Header flags (SW §7.2).
constexpr uint32_t F_REC           = 1u << 0;     // completion records (ISSUE_PUSH.REC) on every push
constexpr uint32_t F_MARK          = 1u << 1;     // TRACE_MARK NODE_BEGIN (op index) before each op
constexpr uint32_t F_SERIAL        = 1u << 2;     // join after every op (debugging)
constexpr uint32_t F_STOP_ON_ERROR = 1u << 3;     // stop pushing once an error is seen (backend: always)
constexpr uint32_t F_KNOWN         = 0xFu;        // other bits must be 0 (else BAD_VERSION, see graph.cpp)

// `graph` kernel launch arguments (SW §7.1): a0 OPL handle, a1 OPL bytes, a2 flags.
constexpr uint32_t LF_VERIFY_CRC = 1u << 0;

// Executor return codes (SW §7.3; RET of the DONE record, SW §6.4). 1 ... 4096: the failing op is
// RET - 1 (DONE STATUS TILE_ERR). Codes >= 0x8000_0000 are negative as int32: KERNEL_ERR.
constexpr uint32_t RET_OK             = 0u;
constexpr uint32_t RET_BAD_MAGIC      = 0x80000001u;
constexpr uint32_t RET_BAD_VERSION    = 0x80000002u;
constexpr uint32_t RET_TOO_MANY_OPS   = 0x80000003u;
constexpr uint32_t RET_BAD_RECORD     = 0x80000004u;  // unknown kind, wrong n_args, slot overflow, layout
constexpr uint32_t RET_BAD_DEP        = 0x80000005u;  // dep >= own index
constexpr uint32_t RET_SCRATCH        = 0x80000006u;  // the op cannot be planned in the ring
constexpr uint32_t RET_CRC            = 0x80000007u;

// ---------------------------------------------------------------------------------------------
// Kinds (SW §7.2). M6 implements NOP, JOIN and MUL_MAT; M7 the rest (W2b); 0x20-0x3F are M8's.
// ---------------------------------------------------------------------------------------------
constexpr uint32_t K_NOP         = 0x00u;   // no tile ops; its ticket is "none"
constexpr uint32_t K_JOIN        = 0x01u;   // SYNC NOP over dep0, dep1
constexpr uint32_t K_MUL_MAT     = 0x10u;
constexpr uint32_t K_ADD         = 0x11u;
constexpr uint32_t K_MUL         = 0x12u;
constexpr uint32_t K_SCALE       = 0x13u;
constexpr uint32_t K_RMS_NORM    = 0x14u;
constexpr uint32_t K_GLU         = 0x15u;
constexpr uint32_t K_ROPE        = 0x16u;
constexpr uint32_t K_GET_ROWS    = 0x17u;
constexpr uint32_t K_SET_ROWS_KV = 0x18u;
constexpr uint32_t K_FLASH_ATTN  = 0x19u;
constexpr uint32_t K_CPY         = 0x1Au;
constexpr uint32_t K_UNARY       = 0x1Bu;
constexpr uint32_t N_KINDS       = 0x40u;   // kind table size (0x00-0x3F); a larger kind is BAD_RECORD

// Argument count of each defined kind (the table of SW §7.2), or NARGS_UNDEF for a kind without a
// definition (the executor answers BAD_RECORD). A record must carry exactly this many arguments.
constexpr uint32_t NARGS_UNDEF = 0xFFu;
constexpr uint32_t kind_nargs(uint32_t kind) {
    switch (kind) {
        case K_NOP:         return 0u;
        case K_JOIN:        return 0u;
        case K_MUL_MAT:     return 12u;
        case K_ADD:         return 10u;
        case K_MUL:         return 10u;
        case K_SCALE:       return 8u;
        case K_RMS_NORM:    return 10u;
        case K_GLU:         return 9u;
        case K_ROPE:        return 14u;
        case K_GET_ROWS:    return 7u;
        case K_SET_ROWS_KV: return 12u;
        case K_FLASH_ATTN:  return 22u;
        case K_CPY:         return 6u;
        case K_UNARY:       return 7u;
        default:            return NARGS_UNDEF;
    }
}
// Smallest slot count holding n_args argument words.
constexpr uint32_t slots_for_nargs(uint32_t n_args) { return (n_args + 2u + SLOT_WORDS - 1u) / SLOT_WORDS; }

// MUL_MAT (0x10) argument words (SW §7.2). Handles are {ch[31:28], beat[27:0]}; strides in lines.
namespace mm {
constexpr uint32_t X = 0, X_STRIDE = 1, Y = 2, Y_STRIDE = 3, W_OFF = 4, FMT = 5, N = 6, K = 7, T = 8,
                   ADD = 9, ADD_STRIDE = 10, FLAGS = 11, NARGS = 12;
constexpr uint32_t FLAG_ACC_DST = 1u << 0;   // y += W x (y is read and written)
}  // namespace mm

// ---------------------------------------------------------------------------------------------
// Layouts
// ---------------------------------------------------------------------------------------------
struct header {
    uint32_t magic;        // MAGIC
    uint16_t version;      // 1
    uint16_t hdr_bytes;    // 64
    uint32_t n_ops;        // 1 ... 4096
    uint32_t n_slots;      // 64-byte slots in the op area
    uint32_t ops_off;      // byte offset of the op area (64-aligned)
    uint32_t flags;        // F_*
    uint32_t n_pre;        // preload entries
    uint32_t pre_off;      // byte offset of the preload table (64-aligned when n_pre > 0)
    uint32_t pre_lines;    // SPM lines of the preload area (<= 256)
    uint32_t graph_id;     // host hash, logged by the executor
    uint32_t total_bytes;  // blob size (a multiple of 64)
    uint32_t crc32;        // CRC-32/ISO-HDLC of bytes [64, total_bytes)
    uint32_t rsvd[4];      // 0
};
static_assert(sizeof(header) == HDR_BYTES, "OPL header is 64 bytes");

struct pre_entry {
    uint32_t src;          // DRAM handle (64-byte-aligned beat)
    uint32_t bytes;        // size in bytes (> 0)
    uint32_t pre_byte;     // byte offset inside the preload area, 64-aligned
    uint32_t rsvd;         // 0
};
static_assert(sizeof(pre_entry) == PRE_ENTRY_BYTES, "OPL preload entry is 16 bytes");

// Header word offsets (for readers that load words, e.g. through the SPM window).
namespace hw {
constexpr uint32_t MAGIC = 0, VER = 1, N_OPS = 2, N_SLOTS = 3, OPS_OFF = 4, FLAGS = 5, N_PRE = 6,
                   PRE_OFF = 7, PRE_LINES = 8, GRAPH_ID = 9, TOTAL = 10, CRC = 11;
}  // namespace hw

// Record word0 / word1 fields.
constexpr uint32_t w0_kind(uint32_t w0)   { return w0 & 0xFFu; }
constexpr uint32_t w0_slots(uint32_t w0)  { return ((w0 >> 8) & 0xFu) + 1u; }
constexpr uint32_t w0_rsvd(uint32_t w0)   { return (w0 >> 12) & 0xFu; }
constexpr uint32_t w0_kflags(uint32_t w0) { return (w0 >> 16) & 0xFFu; }
constexpr uint32_t w0_nargs(uint32_t w0)  { return w0 >> 24; }
constexpr uint32_t w1_dep0(uint32_t w1)   { return w1 & 0xFFFFu; }
constexpr uint32_t w1_dep1(uint32_t w1)   { return w1 >> 16; }
constexpr uint32_t make_w0(uint32_t kind, uint32_t n_slots, uint32_t kflags, uint32_t n_args) {
    return (kind & 0xFFu) | (((n_slots - 1u) & 0xFu) << 8) | ((kflags & 0xFFu) << 16) | ((n_args & 0xFFu) << 24);
}
constexpr uint32_t make_w1(uint32_t dep0, uint32_t dep1) { return (dep0 & 0xFFFFu) | ((dep1 & 0xFFFFu) << 16); }

// Summary of the last `graph` launch, kept by the executor in DTCM (__tsr_exec_state, exec.hpp)
// for the debugger and the tests (host tools read it from the image's DTCM by symbol). Not ABI
// beyond this SDK version.
struct exec_state {
    uint32_t n_ops;          // ops of the list (header n_ops; 0 if the header was refused)
    uint32_t n_done;         // ops whose pushes started (< n_ops after a stop or a refusal)
    uint32_t stopped;        // 1: pushing stopped (STOP_ON_ERROR saw an error, or an executor load failed)
    uint32_t graph_id;       // header graph_id
    uint32_t flags;          // header flags (| F_MARK for a PROF launch)
    uint32_t pidx0;          // ISSUE_PIDX of op 0's first push (pidx_end if no op started)
    uint32_t pidx_end;       // ISSUE_PIDX after the last pushed op (before join_all)
    uint32_t chunks;         // 128 KiB chunks of a streamed list (0: not streamed)
    uint32_t late_chunks;    // chunk loads pushed only when their first record was needed (not
                             // prefetched; 0 unless the prefetch rule is broken)
    uint32_t ret;            // the kernel's return value
};

// ---------------------------------------------------------------------------------------------
// Validation helpers (pure; used by the executor and by the reader below)
// ---------------------------------------------------------------------------------------------
// Header checks of SW §7.3 step 1, in this order: magic (BAD_MAGIC); version, hdr_bytes, unknown
// flag bits and nonzero reserved words 48-63 (BAD_VERSION: fields a v1 executor does not know); n_ops in [1, 4096] (TOO_MANY_OPS; 0 ops is malformed: BAD_RECORD);
// layout (BAD_RECORD): total_bytes a multiple of 64, at least 64 and at most `blob_bytes` (the
// launch's a1), ops_off 64-aligned and >= 64, the op area [ops_off, ops_off + 64 n_slots) equal to
// the rest of the blob, n_ops <= n_slots <= 4096 x 16, the preload table 64-aligned inside [64, ops_off),
// pre_lines <= 256 (0 iff n_pre = 0). The CRC is the caller's (it needs the bytes).
constexpr uint32_t check_header(const header & h, uint32_t blob_bytes) {
    if (h.magic != MAGIC) return RET_BAD_MAGIC;
    if (h.version != VERSION || h.hdr_bytes != HDR_BYTES || (h.flags & ~F_KNOWN) != 0u ||
        (h.rsvd[0] | h.rsvd[1] | h.rsvd[2] | h.rsvd[3]) != 0u)
        return RET_BAD_VERSION;
    if (h.n_ops > MAX_OPS) return RET_TOO_MANY_OPS;
    if (h.n_ops == 0u) return RET_BAD_RECORD;
    const uint32_t tb = h.total_bytes;
    if (tb % SLOT_BYTES != 0u || tb < HDR_BYTES || tb > blob_bytes) return RET_BAD_RECORD;
    if (h.ops_off % SLOT_BYTES != 0u || h.ops_off < HDR_BYTES || h.ops_off > tb) return RET_BAD_RECORD;
    if (h.n_slots != (tb - h.ops_off) / SLOT_BYTES || h.n_slots < h.n_ops || h.n_slots > MAX_OPS * MAX_REC_SLOTS)
        return RET_BAD_RECORD;
    if (h.n_pre != 0u) {
        if (h.pre_off % SLOT_BYTES != 0u || h.pre_off < HDR_BYTES || h.pre_off > h.ops_off ||
            h.n_pre > (h.ops_off - h.pre_off) / PRE_ENTRY_BYTES)
            return RET_BAD_RECORD;
        if (h.pre_lines == 0u || h.pre_lines > PRE_MAX_LINES) return RET_BAD_RECORD;
    } else if (h.pre_lines != 0u) {
        return RET_BAD_RECORD;
    }
    return RET_OK;
}

// One preload entry against the header's preload area: bytes > 0, pre_byte 64-aligned, the entry
// inside [0, 64 pre_lines), rsvd 0. BAD_RECORD otherwise.
constexpr uint32_t check_pre(const pre_entry & e, uint32_t pre_lines) {
    if (e.bytes == 0u || e.pre_byte % PRE_ALIGN != 0u || e.rsvd != 0u) return RET_BAD_RECORD;
    const uint64_t end = uint64_t(e.pre_byte) + e.bytes;
    return end <= uint64_t(pre_lines) * SLOT_BYTES ? RET_OK : RET_BAD_RECORD;
}

// Record checks for op `idx` whose first slot is slot `slot` of an op area of `n_slots` slots:
// word0 [15:12] = 0, the record inside the op area (slot overflow), n_args <= 16 n_slots - 2
// (BAD_RECORD); a defined kind with exactly kind_nargs() arguments (BAD_RECORD); deps < idx or
// DEP_NONE (BAD_DEP). Kinds defined but not built into this image are the executor's business
// (no handler: BAD_RECORD).
// check_record_n takes the required argument count (NARGS_UNDEF: no such kind) from the caller
// (the executor's test kinds, exec.hpp).
constexpr uint32_t check_record_n(uint32_t w0, uint32_t w1, uint32_t idx, uint32_t slot, uint32_t n_slots,
                                  uint32_t want) {
    const uint32_t ns = w0_slots(w0);
    if (w0_rsvd(w0) != 0u || slot >= n_slots || ns > n_slots - slot || w0_nargs(w0) + 2u > ns * SLOT_WORDS)
        return RET_BAD_RECORD;
    if (want == NARGS_UNDEF || w0_nargs(w0) != want) return RET_BAD_RECORD;
    const uint32_t d0 = w1_dep0(w1), d1 = w1_dep1(w1);
    if ((d0 != DEP_NONE && d0 >= idx) || (d1 != DEP_NONE && d1 >= idx)) return RET_BAD_DEP;
    return RET_OK;
}
constexpr uint32_t check_record(uint32_t w0, uint32_t w1, uint32_t idx, uint32_t slot, uint32_t n_slots) {
    return check_record_n(w0, w1, idx, slot, n_slots, kind_nargs(w0_kind(w0)));
}

// ---------------------------------------------------------------------------------------------
// CRC-32/ISO-HDLC (= zlib crc32: reflected 0xEDB88320, init and final XOR 0xFFFFFFFF; check value
// crc32("123456789") = 0xCBF43926). Nibble table: 64 bytes of .rodata in the firmware.
// Use: c = CRC_INIT; c = crc_bytes(c, p, n) / crc_word(c, w) ...; result = crc_final(c).
// ---------------------------------------------------------------------------------------------
constexpr uint32_t CRC_INIT = 0xFFFFFFFFu;
struct crc_nibble_table { uint32_t t[16]; };
constexpr crc_nibble_table make_crc_nibble_table() {
    crc_nibble_table r{};
    for (uint32_t i = 0; i < 16u; ++i) {
        uint32_t c = i;
        for (int k = 0; k < 4; ++k) c = (c >> 1) ^ (0xEDB88320u & (0u - (c & 1u)));
        r.t[i] = c;
    }
    return r;
}
inline constexpr crc_nibble_table CRC_NIBBLE = make_crc_nibble_table();

constexpr uint32_t crc_byte(uint32_t c, uint8_t b) {
    c ^= b;
    c = (c >> 4) ^ CRC_NIBBLE.t[c & 0xFu];
    c = (c >> 4) ^ CRC_NIBBLE.t[c & 0xFu];
    return c;
}
// Four bytes of a little-endian word, low byte first.
constexpr uint32_t crc_word(uint32_t c, uint32_t w) {
    c ^= w;
    for (int k = 0; k < 8; ++k) c = (c >> 4) ^ CRC_NIBBLE.t[c & 0xFu];
    return c;
}
constexpr uint32_t crc_bytes(uint32_t c, const uint8_t * p, size_t n) {
    for (size_t i = 0; i < n; ++i) c = crc_byte(c, p[i]);
    return c;
}
constexpr uint32_t crc_final(uint32_t c) { return ~c; }
static_assert(crc_final(crc_word(crc_word(crc_byte(CRC_INIT, '1'), 0x35343332u), 0x39383736u)) == 0xCBF43926u,
              "CRC-32/ISO-HDLC check value");

// ---------------------------------------------------------------------------------------------
// Freestanding reader over a blob in addressable memory (host tools, tests, the backend's dumps;
// the executor reads through the SPM window with the same checks, graph.cpp). Copies, no heap.
// ---------------------------------------------------------------------------------------------
constexpr uint32_t rd32(const uint8_t * p) {
    return uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24;
}

struct record {
    uint32_t idx;          // op index
    uint32_t slot;         // first slot in the op area
    uint32_t kind, n_slots, kflags, n_args, dep0, dep1;
    const uint8_t * args;  // n_args little-endian words
    uint32_t arg(uint32_t i) const { return rd32(args + 4u * i); }
};

class reader {
public:
    // Parses and checks the header (check_header with blob_bytes = bytes) and, if verify_crc, the
    // CRC. Returns RET_OK or the executor's code for the same blob.
    uint32_t open(const void * blob, uint32_t bytes, bool verify_crc) {
        p_ = static_cast<const uint8_t *>(blob);
        next_idx_ = 0;
        next_slot_ = 0;
        if (bytes < HDR_BYTES) return RET_BAD_MAGIC;
        h_.magic = rd32(p_ + 0);
        h_.version = uint16_t(rd32(p_ + 4) & 0xFFFFu);
        h_.hdr_bytes = uint16_t(rd32(p_ + 4) >> 16);
        h_.n_ops = rd32(p_ + 8);
        h_.n_slots = rd32(p_ + 12);
        h_.ops_off = rd32(p_ + 16);
        h_.flags = rd32(p_ + 20);
        h_.n_pre = rd32(p_ + 24);
        h_.pre_off = rd32(p_ + 28);
        h_.pre_lines = rd32(p_ + 32);
        h_.graph_id = rd32(p_ + 36);
        h_.total_bytes = rd32(p_ + 40);
        h_.crc32 = rd32(p_ + 44);
        for (uint32_t i = 0; i < 4u; ++i) h_.rsvd[i] = rd32(p_ + 48u + 4u * i);
        const uint32_t e = check_header(h_, bytes);
        if (e != RET_OK) return e;
        for (uint32_t i = 0; i < h_.n_pre; ++i) {
            const uint32_t pe = check_pre(pre(i), h_.pre_lines);
            if (pe != RET_OK) return pe;
        }
        if (verify_crc &&
            crc_final(crc_bytes(CRC_INIT, p_ + HDR_BYTES, h_.total_bytes - HDR_BYTES)) != h_.crc32)
            return RET_CRC;
        return RET_OK;
    }
    const header & hdr() const { return h_; }
    pre_entry pre(uint32_t i) const {
        const uint8_t * q = p_ + h_.pre_off + PRE_ENTRY_BYTES * i;
        pre_entry e{rd32(q), rd32(q + 4), rd32(q + 8), rd32(q + 12)};
        return e;
    }
    // Next record with the executor's record checks. Returns RET_OK, a check_record code, or
    // RET_BAD_RECORD after the last op (idx == n_ops). any_kind: a kind without a definition in
    // SW §7.2 (e.g. the executor tests' kinds 0x30-0x3F) is accepted with any argument count.
    uint32_t next(record & r, bool any_kind = false) {
        if (next_idx_ >= h_.n_ops) return RET_BAD_RECORD;
        const uint8_t * q = p_ + h_.ops_off + SLOT_BYTES * next_slot_;
        if (next_slot_ >= h_.n_slots) return RET_BAD_RECORD;
        const uint32_t w0 = rd32(q), w1 = rd32(q + 4);
        const uint32_t want = kind_nargs(w0_kind(w0));
        const uint32_t e = check_record_n(w0, w1, next_idx_, next_slot_, h_.n_slots,
                                          (want == NARGS_UNDEF && any_kind) ? w0_nargs(w0) : want);
        r.idx = next_idx_;
        r.slot = next_slot_;
        r.kind = w0_kind(w0);
        r.n_slots = w0_slots(w0);
        r.kflags = w0_kflags(w0);
        r.n_args = w0_nargs(w0);
        r.dep0 = w1_dep0(w1);
        r.dep1 = w1_dep1(w1);
        r.args = q + 8;
        if (e != RET_OK) return e;
        next_idx_ += 1u;
        next_slot_ += r.n_slots;
        return RET_OK;
    }

private:
    const uint8_t * p_ = nullptr;
    header h_{};
    uint32_t next_idx_ = 0, next_slot_ = 0;
};

}  // namespace opl
}  // namespace tsr

// =============================================================================================
// Host writer (hosted builds only): the backend's OPL builder (P3.2) and the tests.
// =============================================================================================
#if defined(__STDC_HOSTED__) && __STDC_HOSTED__ && !defined(TSR_OPL_NO_WRITER)
#include <cstring>
#include <vector>

namespace tsr {
namespace opl {

// Builds one blob. Records are appended in order; add() returns the op index of the new record
// (after any JOIN records it inserted). Layout by finish(): header, preload table at 64 (rounded
// to a slot), op area, CRC over [64, total). Preload entries get 64-aligned pre_byte offsets in
// the order they are added; pre_lines covers the last one. Errors (more than 4096 ops, more than
// 256 preload lines, a dep that is not an earlier op) set ok() = false with a message.
class writer {
public:
    explicit writer(uint32_t flags = F_STOP_ON_ERROR, uint32_t graph_id = 0) : flags_(flags), graph_id_(graph_id) {}

    void set_flags(uint32_t f) { flags_ = f; }
    void set_graph_id(uint32_t id) { graph_id_ = id; }
    bool ok() const { return err_ == nullptr; }
    const char * error() const { return err_ ? err_ : ""; }
    uint32_t n_ops() const { return n_ops_; }
    uint32_t n_slots() const { return uint32_t(ops_.size() / SLOT_WORDS); }
    uint32_t pre_lines() const { return (pre_end_ + SLOT_BYTES - 1u) / SLOT_BYTES; }

    // A preload entry: returns its pre_byte (offset inside the preload area), or HANDLE_NONE.
    uint32_t add_preload(uint32_t src_handle, uint32_t bytes) {
        const uint32_t at = (pre_end_ + PRE_ALIGN - 1u) / PRE_ALIGN * PRE_ALIGN;
        if (bytes == 0u || uint64_t(at) + bytes > uint64_t(PRE_MAX_LINES) * SLOT_BYTES) {
            fail("preload area exceeds 256 lines (or an empty entry)");
            return HANDLE_NONE;
        }
        pre_.push_back(pre_entry{src_handle, bytes, at, 0u});
        pre_end_ = at + bytes;
        return at;
    }

    // Low level: one record with explicit word0 fields and deps (no JOIN folding, no checks beyond
    // the slot count; tests use it for malformed records). n_slots = 0 takes the minimum.
    uint32_t add_raw(uint32_t kind, const uint32_t * args, uint32_t n_args, uint32_t dep0, uint32_t dep1,
                     uint32_t kflags = 0, uint32_t n_slots = 0, uint32_t rsvd_bits = 0) {
        if (n_slots == 0u) n_slots = slots_for_nargs(n_args);
        if (n_slots == 0u || n_slots > MAX_REC_SLOTS || n_args > 255u) {
            fail("record slot count out of range");
            return HANDLE_NONE;
        }
        const size_t base = ops_.size();
        ops_.resize(base + size_t(n_slots) * SLOT_WORDS, 0u);
        ops_[base] = make_w0(kind, n_slots, kflags, n_args) | ((rsvd_bits & 0xFu) << 12);
        ops_[base + 1] = make_w1(dep0, dep1);
        for (uint32_t i = 0; i < n_args && 2u + i < n_slots * SLOT_WORDS; ++i) ops_[base + 2 + i] = args[i];
        return n_ops_++;
    }

    // One op with any number of predecessors (op indices; DEP_NONE and duplicates are dropped).
    // More than two are folded by JOIN records emitted before the op (SW §7.2): the first two
    // predecessors go into a JOIN, whose index replaces them at the end of the list, until two
    // remain. Returns the op's index.
    uint32_t add(uint32_t kind, const uint32_t * args, uint32_t n_args, const uint32_t * deps = nullptr,
                 uint32_t n_deps = 0, uint32_t kflags = 0, uint32_t n_slots = 0) {
        std::vector<uint32_t> q;
        for (uint32_t i = 0; i < n_deps; ++i) {
            const uint32_t d = deps[i];
            if (d == DEP_NONE) continue;
            if (d >= n_ops_) {
                fail("dependency on an op that is not earlier");
                return HANDLE_NONE;
            }
            bool dup = false;
            for (uint32_t x : q) dup = dup || x == d;
            if (!dup) q.push_back(d);
        }
        while (q.size() > 2u) {
            const uint32_t j = add_raw(K_JOIN, nullptr, 0, q[0], q[1]);
            q.erase(q.begin(), q.begin() + 2);
            q.push_back(j);
        }
        const uint32_t d0 = q.size() > 0 ? q[0] : DEP_NONE, d1 = q.size() > 1 ? q[1] : DEP_NONE;
        return add_raw(kind, args, n_args, d0, d1, kflags, n_slots);
    }
    uint32_t add(uint32_t kind, std::initializer_list<uint32_t> args, std::initializer_list<uint32_t> deps = {},
                 uint32_t kflags = 0) {
        std::vector<uint32_t> a(args), d(deps);
        return add(kind, a.data(), uint32_t(a.size()), d.data(), uint32_t(d.size()), kflags);
    }

    // MUL_MAT record (SW §7.2 argument words, mm::*).
    uint32_t mul_mat(uint32_t x, uint32_t x_stride, uint32_t y, uint32_t y_stride, uint32_t w_off, uint32_t fmt,
                     uint32_t N, uint32_t K, uint32_t T, uint32_t addend = HANDLE_NONE, uint32_t addend_stride = 0,
                     uint32_t mm_flags = 0, const uint32_t * deps = nullptr, uint32_t n_deps = 0) {
        const uint32_t a[mm::NARGS] = {x, x_stride, y, y_stride, w_off, fmt, N, K, T, addend, addend_stride, mm_flags};
        return add(K_MUL_MAT, a, mm::NARGS, deps, n_deps);
    }

    // The blob (header, preload table, op area; CRC filled). Empty if !ok() or there are no ops
    // (callers that want malformed headers patch the returned bytes and call patch_crc()).
    std::vector<uint8_t> finish() const {
        std::vector<uint8_t> out;
        if (!ok() || n_ops_ == 0u) return out;
        if (n_ops_ > MAX_OPS) {
            const_cast<writer *>(this)->fail("more than 4096 ops (the builder must split, V2-SW-29)");
            return out;
        }
        const uint32_t pre_off = pre_.empty() ? 0u : HDR_BYTES;
        const uint32_t pre_bytes = uint32_t(pre_.size()) * PRE_ENTRY_BYTES;
        const uint32_t ops_off = HDR_BYTES + (pre_bytes + SLOT_BYTES - 1u) / SLOT_BYTES * SLOT_BYTES;
        const uint32_t total = ops_off + uint32_t(ops_.size()) * 4u;
        out.assign(total, 0u);
        header h{};
        h.magic = MAGIC;
        h.version = uint16_t(VERSION);
        h.hdr_bytes = uint16_t(HDR_BYTES);
        h.n_ops = n_ops_;
        h.n_slots = n_slots();
        h.ops_off = ops_off;
        h.flags = flags_;
        h.n_pre = uint32_t(pre_.size());
        h.pre_off = pre_off;
        h.pre_lines = pre_.empty() ? 0u : pre_lines();
        h.graph_id = graph_id_;
        h.total_bytes = total;
        h.crc32 = 0u;
        put_header(out.data(), h);
        for (size_t i = 0; i < pre_.size(); ++i) {
            uint8_t * q = out.data() + pre_off + PRE_ENTRY_BYTES * i;
            wr32(q, pre_[i].src); wr32(q + 4, pre_[i].bytes); wr32(q + 8, pre_[i].pre_byte); wr32(q + 12, 0u);
        }
        for (size_t i = 0; i < ops_.size(); ++i) wr32(out.data() + ops_off + 4u * i, ops_[i]);
        patch_crc(out);
        return out;
    }

    // Recomputes header.crc32 over [64, total_bytes) (total_bytes as stored, clamped to the blob).
    static void patch_crc(std::vector<uint8_t> & blob) {
        if (blob.size() < HDR_BYTES) return;
        uint32_t tb = rd32(blob.data() + 4u * hw::TOTAL);
        if (tb > blob.size()) tb = uint32_t(blob.size());
        const uint32_t c = tb > HDR_BYTES ? crc_final(crc_bytes(CRC_INIT, blob.data() + HDR_BYTES, tb - HDR_BYTES))
                                          : crc_final(CRC_INIT);
        wr32(blob.data() + 4u * hw::CRC, c);
    }
    static void wr32(uint8_t * p, uint32_t v) {
        p[0] = uint8_t(v); p[1] = uint8_t(v >> 8); p[2] = uint8_t(v >> 16); p[3] = uint8_t(v >> 24);
    }
    static void put_header(uint8_t * p, const header & h) {
        wr32(p + 0, h.magic);
        wr32(p + 4, uint32_t(h.version) | uint32_t(h.hdr_bytes) << 16);
        wr32(p + 8, h.n_ops);
        wr32(p + 12, h.n_slots);
        wr32(p + 16, h.ops_off);
        wr32(p + 20, h.flags);
        wr32(p + 24, h.n_pre);
        wr32(p + 28, h.pre_off);
        wr32(p + 32, h.pre_lines);
        wr32(p + 36, h.graph_id);
        wr32(p + 40, h.total_bytes);
        wr32(p + 44, h.crc32);
        for (uint32_t i = 0; i < 4u; ++i) wr32(p + 48u + 4u * i, h.rsvd[i]);
    }

private:
    void fail(const char * m) {
        if (err_ == nullptr) err_ = m;
    }
    uint32_t flags_, graph_id_;
    uint32_t n_ops_ = 0;
    std::vector<uint32_t> ops_;       // op area words
    std::vector<pre_entry> pre_;
    uint32_t pre_end_ = 0;            // bytes used in the preload area
    const char * err_ = nullptr;
};

}  // namespace opl
}  // namespace tsr
#endif  // hosted writer
