// log.cpp — log records and tsr::panic (docs/v2/06-software-verification.md "SW" §6.6;
// docs/v2/05-debug.md "DB" §6.1, §6.3; docs/v2/01-execution-model.md "EX" §4.10). The record format
// and the API are described in tsr/log.hpp.
//
// Writer (rt::log_write), in this order:
//   1. LOG_CFG (the RISC-V copy, 0x288): ENABLE = 0 or LEVEL > LOG_CFG.LEVEL drops the record
//      silently (filtered, not counted);
//   2. the record needs 1 + N words, N = 1 (TS: MTIME[31:0]) + n; with records dropped earlier and
//      not yet reported, 4 more words for the "dropped n" EVENT record (header, MTIME,
//      LOG_EV_DROPPED, n; LEVEL WARN, SOURCE runtime). If LOG_FREE is smaller: with LOG_CFG.BLOCK
//      (and allow_block) re-read LOG_FREE until it fits; else drop the whole record and count it;
//   3. the "dropped n" EVENT record first (count reset), then the header and the payload.
// LOG_FREE may under-report (EX §4.1), never over-report, so the ring never holds a partial
// record. Its value only decides whether a record is written (log content is not an observable of
// SW §4.7). LSEQ is SEQ[7:0] of the running launch, 0 outside launches (DB §6.1).
#include "rt_int.hpp"
#include "tsr/log.hpp"

using namespace tsr;
using tsr::rt::cw;

extern "C" [[noreturn]] void tsr_rt_panic_trap(uint32_t code);   // crt0.S

namespace {
uint32_t g_dropped;     // records dropped since the last "dropped n" EVENT (DTCM, SW §6.6)

inline uint32_t hdr(uint32_t n, uint32_t level, uint32_t source, uint32_t type) {
    const uint32_t lseq = tsr_rt_launch.active ? (tsr_rt_launch.seq & 0xFFu) : 0u;
    return (n << LOG_HDR_N_LSB) | (level << LOG_HDR_LEVEL_LSB) | (source << LOG_HDR_SOURCE_LSB) |
           (type << LOG_HDR_TYPE_LSB) | LOG_HDR_TS | (lseq << LOG_HDR_LSEQ_LSB) |
           (LOG_HDR_SYNC_MAGIC << LOG_HDR_SYNC_LSB);
}
}  // namespace

void tsr::rt::log_write2(uint32_t level, uint32_t source, uint32_t type, uint32_t pre, uint32_t npre,
                         const volatile uint32_t * payload, uint32_t n, int allow_block) {
    const uint32_t cfg = cw(CW_LOG_CFG);
    if (!(cfg & LOG_CFG_ENABLE) || level > ((cfg & LOG_CFG_LEVEL) >> LOG_CFG_LEVEL_LSB)) return;
    if (npre + n > 254u) n = 254u - npre;               // N = 1 + npre + n <= 255 (header [7:0])
    const uint32_t need = 2u + npre + n;                 // header, MTIME, [pre], payload
    for (;;) {
        const uint32_t want = need + (g_dropped ? 4u : 0u);
        if (cw(CW_LOG_FREE) >= want) break;
        if (!(allow_block && (cfg & LOG_CFG_BLOCK))) {
            ++g_dropped;
            return;
        }
    }
    volatile uint32_t & push = cw(CW_LOG_PUSH);
    if (g_dropped) {
        push = hdr(3u, LOG_HDR_LEVEL_WARN, LOG_HDR_SOURCE_RUNTIME, LOG_HDR_TYPE_EVENT);
        push = cw(CW_MTIME_LO);
        push = LOG_EV_DROPPED;
        push = g_dropped;
        g_dropped = 0;
    }
    push = hdr(1u + npre + n, level, source, type);
    push = cw(CW_MTIME_LO);
    if (npre) push = pre;
    for (uint32_t i = 0; i < n; ++i) push = payload[i];
}

void tsr::rt::log_write(uint32_t level, uint32_t source, uint32_t type, const volatile uint32_t * payload, uint32_t n,
                        int allow_block) {
    log_write2(level, source, type, 0u, 0u, payload, n, allow_block);
}

void tsr::log_record(uint32_t level, uint32_t source, uint32_t type, const uint32_t * payload, uint32_t n) {
    rt::log_write2(level, source, type, 0u, 0u, payload, n, 1);
}

// TEXT: UTF-8, little-endian in words, zero-padded; one line per record (at most 64 words here).
void tsr::log_text(uint32_t level, uint32_t source, const char * s) {
    uint32_t w[64];
    uint32_t n = 0, k = 0;
    for (; s[k] != '\0' && n < 64u; ++k) {
        if ((k & 3u) == 0u) w[n] = 0;
        w[n] |= uint32_t(uint8_t(s[k])) << (8u * (k & 3u));
        if ((k & 3u) == 3u) ++n;
    }
    if ((k & 3u) != 0u && n < 64u) ++n;
    rt::log_write2(level, source, LOG_HDR_TYPE_TEXT, 0u, 0u, w, n, 1);
}

// EVENT: event code (marker encoding, DB §8.4), then the arguments.
void tsr::log_event(uint32_t level, uint32_t source, uint32_t code, const uint32_t * a, uint32_t n) {
    rt::log_write2(level, source, LOG_HDR_TYPE_EVENT, code, 1u, a, n, 1);
}

// BLOB: tag, then raw words.
void tsr::log_blob(uint32_t level, uint32_t source, uint32_t tag, const uint32_t * x, uint32_t n) {
    rt::log_write2(level, source, LOG_HDR_TYPE_BLOB, tag, 1u, x, n, 1);
}

uint32_t tsr::log_dropped() { return g_dropped; }

// DB §6.3, SW §6.6: FATAL record with the arguments, then the trap handler with mcause 24.
[[noreturn]] void tsr::panic(uint32_t code, uint32_t a0, uint32_t a1, uint32_t a2) {
    TSR_LOG_SRC(TSR_LOG_FATAL, LOG_HDR_SOURCE_RUNTIME, "panic 0x%08x (0x%08x, 0x%08x, 0x%08x)", code, a0, a1, a2);
    tsr_rt_panic_trap(code);
}
