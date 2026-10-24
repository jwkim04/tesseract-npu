// tsr_bw_regs.h — software view of the M0 DMA bandwidth kernel tsr_bw_k4 (docs/m0-bw.md §2, §3):
// the s_axi_control register map, the ctrl word, and the reference functions of the write pattern,
// the read checksum and the burst sequence. Header-only C++ (no dependencies), shared by the
// host tool (sw/tools/tsr_bw.cpp) and the Verilator bench (tb/platform/bw/bw_bench.cpp), which
// checks the RTL against exactly these functions.
//
// Owner: m0bw agent.
#pragma once

#include <cstdint>

namespace tsr_bw {

// ---- s_axi_control (byte offsets; range 0x1000) ----
constexpr uint32_t REG_RANGE = 0x1000;
constexpr uint32_t AP_CTRL = 0x000, GIE = 0x004, IER = 0x008, ISR = 0x00C;
constexpr uint32_t AP_START = 1, AP_DONE = 2, AP_IDLE = 4, AP_READY = 8;
constexpr uint32_t SEED = 0x010;                         // pattern seed (all channels)
constexpr uint32_t CH_CFG0 = 0x020, CH_CFG_STRIDE = 0x020;
constexpr uint32_t CFG_CTRL = 0x00, CFG_BYTES = 0x08, CFG_RD_OFF = 0x10, CFG_WR_OFF = 0x18;
constexpr uint32_t CH_BUF0 = 0x0A0;                      // + 8c: ch<c> buffer address (u64)
constexpr uint32_t BW_VERSION = 0x0F0, BW_INFO = 0x0F4, TOTAL_CYCLES = 0x0F8;
constexpr uint32_t RES0 = 0x100, RES_STRIDE = 0x040;     // channel result blocks
constexpr uint32_t R_STATUS = 0x00, R_ERRS = 0x04, R_RD_CYCLES = 0x08, R_WR_CYCLES = 0x10, R_RD_BYTES = 0x18,
                   R_WR_BYTES = 0x20, R_CHECKSUM = 0x28, R_RD_LAT = 0x30, R_WR_LAT = 0x34, R_RD_BURSTS = 0x38,
                   R_WR_BURSTS = 0x3C;
constexpr uint32_t ST_RAN = 1, ST_CFG_ERR = 2, ST_RRESP = 4, ST_BRESP = 8, ST_UNEXPECTED = 16, ST_DONE = 32;
constexpr uint32_t VERSION_MAGIC = 0x4257;               // BW_VERSION[31:16] ("BW")
constexpr uint32_t VERSION_MAJOR = 1;                    // BW_VERSION[15:8]; [7:0] = NUM_CH
constexpr unsigned MAX_CH = 4;

inline uint32_t cfg_reg(unsigned ch, uint32_t field) { return CH_CFG0 + CH_CFG_STRIDE * ch + field; }
inline uint32_t buf_reg(unsigned ch) { return CH_BUF0 + 8 * ch; }
inline uint32_t res_reg(unsigned ch, uint32_t field) { return RES0 + RES_STRIDE * ch + field; }

// ctrl word: bit0 enable, bit1 read, bit2 write, [14:8] burst beats (1..64), [22:16] outstanding
// bursts per direction (1..64), [26:24] id_log2 (0..4: IDs 0 .. 2^id_log2 - 1 round robin)
inline uint32_t ctrl_word(bool en, bool rd, bool wr, unsigned burst, unsigned outstanding, unsigned id_log2) {
    return (en ? 1u : 0u) | (rd ? 2u : 0u) | (wr ? 4u : 0u) | (burst & 0x7Fu) << 8 | (outstanding & 0x7Fu) << 16 |
           (id_log2 & 7u) << 24;
}

// ---- write pattern: 64-bit word at byte offset o (8-aligned, from the buffer address) ----
inline uint64_t mix64(uint64_t x) {   // one xorshift64 step, linear over GF(2)
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return x;
}
inline uint64_t salt(uint32_t seed, unsigned ch) { return (uint64_t)seed << 32 | (ch & 0xFFu); }
inline uint64_t pattern_word(uint32_t seed, unsigned ch, uint64_t off) { return mix64(off ^ salt(seed, ch)); }
// The 8 words of the beat at byte offset off (64-aligned), fast form: mix64(off) ^ mix64(8i) ^ mix64(salt).
inline void pattern_beat(uint32_t seed, unsigned ch, uint64_t off, uint64_t w[8]) {
    const uint64_t m = mix64(off) ^ mix64(salt(seed, ch));
    for (unsigned i = 0; i < 8; ++i) w[i] = m ^ mix64(8ull * i);
}

// ---- read checksum: sum over beats of sum_i rotl64(w_i, 8i) (mod 2^64; order independent) ----
inline uint64_t rotl64(uint64_t x, unsigned n) { return n ? (x << n) | (x >> (64 - n)) : x; }
inline uint64_t beat_hash(const uint64_t w[8]) {
    uint64_t h = 0;
    for (unsigned i = 0; i < 8; ++i) h += rotl64(w[i], 8 * i);
    return h;
}
inline uint64_t beat_hash_bytes(const uint8_t* p) {   // 64 bytes, little-endian words
    uint64_t w[8];
    for (unsigned i = 0; i < 8; ++i) {
        uint64_t v = 0;
        for (unsigned b = 0; b < 8; ++b) v |= (uint64_t)p[8 * i + b] << (8 * b);
        w[i] = v;
    }
    return beat_hash(w);
}

// ---- burst sequence of a region [beat0, beat0 + nbeats): len = min(B, to 4 KiB, left) ----
inline unsigned burst_len(uint64_t beat, uint64_t left, unsigned b) {
    uint64_t l = 64 - (beat & 63);
    if (left < l) l = left;
    return (unsigned)(b < l ? b : l);
}
// Number of bursts the kernel issues for the region (closed form over 4 KiB segments).
inline uint64_t burst_count(uint64_t beat0, uint64_t nbeats, unsigned b) {
    if (nbeats == 0 || b == 0) return 0;
    const uint64_t end = beat0 + nbeats;
    auto ceil_div = [](uint64_t x, uint64_t y) { return (x + y - 1) / y; };
    const uint64_t blk0 = beat0 >> 6, blk1 = (end - 1) >> 6;
    if (blk0 == blk1) return ceil_div(nbeats, b);
    const uint64_t first = 64 - (beat0 & 63), last = end - (blk1 << 6);
    return ceil_div(first, b) + (blk1 - blk0 - 1) * ceil_div(64, b) + ceil_div(last, b);
}

}  // namespace tsr_bw
