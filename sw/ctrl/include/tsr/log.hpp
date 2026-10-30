// tsr/log.hpp — firmware logging and panic (docs/design/06-software-verification.md "SW" §6.6;
// docs/design/05-debug.md "DB" §6.1, §6.3; docs/design/01-execution-model.md "EX" §4.10).
//
// Records go to EX's log ring (LOG_PUSH) with DB's header (the only definition, DB §6.1):
//   [7:0] N payload words, [11:8] LEVEL (0 FATAL .. 5 TRACE), [15:12] SOURCE (0 runtime, 1 kernel,
//   2 trap handler, 3 debug kernels), [18:16] TYPE (0 FMT, 1 TEXT, 2 TRAP, 3 EVENT, 4 BLOB),
//   [19] TS (payload word 0 is MTIME[31:0]), [27:20] LSEQ (SEQ[7:0] of the running launch, 0
//   outside launches), [31:28] SYNC 0xA.
// Every record the SDK writes carries TS. A FMT record's next payload word is the address of its
// format string in .tsr_logfmt (VMA 0xF000_0000, not loaded: no TCM, not in the image id), then
// the arguments as 32-bit words.
//
//   TSR_LOG(level, "fmt", args...)               SOURCE kernel
//   TSR_LOG_SRC(level, source, "fmt", args...)   explicit SOURCE (the runtime uses 0)
// level must be a constant expression: records above TSR_LOG_MAX_LEVEL (default 3 INFO; 4 DEBUG
// in a debug build: no NDEBUG, or TSR_FW_DEBUG) are compiled out. Conversions (decoded on the host
// from the ELF): %d %u %x %08x, %f (argument = f32 bits, pass a uint32_t; firmware has no float,
// SW-8), %t (a ticket: pass tsr::ticket or its uint32_t), 64-bit integers as two words (lo, hi;
// %lld / %llu / %llx), %s for strings in loaded sections (the word is the address). The argument
// word count is fixed at compile time: an integer of 8 bytes takes two words, everything else one.
// The header's LEVEL (0..5) is the log level; FMT records are at most 1 + 253 argument words.
//
// Writing (log.cpp): LOG_CFG.ENABLE = 0 or LEVEL > LOG_CFG.LEVEL drops the record silently (the
// only runtime filter). Otherwise the record is written only when LOG_FREE >= 1 + N: if it does not
// fit it is dropped whole and counted in DTCM, and the next record that fits is preceded by an
// EVENT record "dropped n" (LEVEL WARN, payload MTIME, LOG_EV_DROPPED, n); with LOG_CFG.BLOCK the
// writer waits for space instead. The ring never holds a partial record. No logging from
// interrupt handlers (the runtime has none: it waits in WFI with mstatus.MIE = 0).
//
// tsr::panic(code, a0, a1, a2) logs a FATAL record with its arguments, then enters the trap
// handler with mcause = 24 and mtval = code (DB §6.3): a TRAP record, never a DONE (SW §6.4).
//
// Host build (TSR_FW_HOST, SW §6.8): TSR_LOG formats the record on host_log_sink (default:
// stderr, "[tsr L<level> S<source>] text"); tsr::panic prints its arguments and aborts.
#ifndef TSR_LOG_HPP
#define TSR_LOG_HPP

#include <stddef.h>
#include <stdint.h>

#include "tsr/mmio.hpp"

#ifndef TSR_LOG_MAX_LEVEL
#if !defined(NDEBUG) || (defined(TSR_FW_DEBUG) && TSR_FW_DEBUG)
#define TSR_LOG_MAX_LEVEL 4
#else
#define TSR_LOG_MAX_LEVEL 3
#endif
#endif

#define TSR_LOG_FATAL 0
#define TSR_LOG_ERROR 1
#define TSR_LOG_WARN  2
#define TSR_LOG_INFO  3
#define TSR_LOG_DEBUG 4
#define TSR_LOG_TRACE 5

#ifdef TSR_FW_HOST
#include <stdio.h>
#include <stdlib.h>
#define TSR_LOGFMT_ATTR_
#else
#define TSR_LOGFMT_ATTR_ __attribute__((section(".tsr_logfmt"), aligned(1)))
#endif

namespace tsr {

// ---------------------------------------------------------------------------------------------
// Record writers (log.cpp; host: inline below). payload excludes the TS word, which the writer
// adds. n <= 254 (FMT: 1 + argument words).
// ---------------------------------------------------------------------------------------------
#ifndef TSR_FW_HOST
void log_record(uint32_t level, uint32_t source, uint32_t type, const uint32_t * payload, uint32_t n);
void log_text(uint32_t level, uint32_t source, const char * s);                       // TEXT
void log_event(uint32_t level, uint32_t source, uint32_t code, const uint32_t * a, uint32_t n);  // EVENT
void log_blob(uint32_t level, uint32_t source, uint32_t tag, const uint32_t * w, uint32_t n);    // BLOB
uint32_t log_dropped();          // records dropped and not yet reported by a "dropped n" EVENT
[[noreturn]] void panic(uint32_t code, uint32_t a0 = 0, uint32_t a1 = 0, uint32_t a2 = 0);
#else
inline void (*host_log_sink)(uint32_t level, uint32_t source, uint32_t type, const char * fmt,
                             const uint32_t * a, uint32_t n) = nullptr;
namespace log_detail {
// Minimal host formatter of the conversions above (the device decoder is tsr-log / the transport).
inline void host_print(uint32_t level, uint32_t source, const char * fmt, const uint32_t * a, uint32_t n) {
    fprintf(stderr, "[tsr L%u S%u] ", (unsigned)level, (unsigned)source);
    uint32_t k = 0;
    for (const char * p = fmt; *p; ++p) {
        if (*p != '%') { fputc(*p, stderr); continue; }
        ++p;
        if (*p == '%') { fputc('%', stderr); continue; }
        if (*p == '\0') break;
        int zero8 = 0, ll = 0;
        if (p[0] == '0' && p[1] == '8') { zero8 = 1; p += 2; }
        if (p[0] == 'l' && p[1] == 'l') { ll = 1; p += 2; }
        const uint32_t lo = k < n ? a[k] : 0u;
        const uint32_t hi = (ll && k + 1 < n) ? a[k + 1] : 0u;
        k += ll ? 2u : 1u;
        const unsigned long long v = ((unsigned long long)hi << 32) | lo;
        switch (*p) {
        case 'd': ll ? fprintf(stderr, "%lld", (long long)v) : fprintf(stderr, "%d", (int)(int32_t)lo); break;
        case 'u': fprintf(stderr, "%llu", v); break;
        case 'x': zero8 ? fprintf(stderr, "%08x", (unsigned)lo) : fprintf(stderr, "%llx", v); break;
        case 'f': { union { uint32_t u; float f; } c; c.u = lo; fprintf(stderr, "%g", (double)c.f); } break;
        case 't': fprintf(stderr, "t(%u,%u)", (unsigned)(lo >> 28), (unsigned)(lo & 0x0FFFFFFFu)); break;
        case 's': fprintf(stderr, "<str 0x%08x>", (unsigned)lo); break;
        default: fprintf(stderr, "%%%c", *p); break;
        }
    }
    fputc('\n', stderr);
}
}  // namespace log_detail
inline void log_record(uint32_t level, uint32_t source, uint32_t type, const uint32_t * payload, uint32_t n) {
    if (host_log_sink) host_log_sink(level, source, type, nullptr, payload, n);
}
inline void log_text(uint32_t level, uint32_t source, const char * s) {
    if (host_log_sink) host_log_sink(level, source, LOG_HDR_TYPE_TEXT, s, nullptr, 0);
    else fprintf(stderr, "[tsr L%u S%u] %s\n", (unsigned)level, (unsigned)source, s);
}
inline void log_event(uint32_t level, uint32_t source, uint32_t code, const uint32_t * a, uint32_t n) {
    uint32_t w[256];
    w[0] = code;
    for (uint32_t i = 0; i < n && i < 255u; ++i) w[1 + i] = a[i];
    log_record(level, source, LOG_HDR_TYPE_EVENT, w, n + 1u);
}
inline void log_blob(uint32_t level, uint32_t source, uint32_t tag, const uint32_t * x, uint32_t n) {
    uint32_t w[256];
    w[0] = tag;
    for (uint32_t i = 0; i < n && i < 255u; ++i) w[1 + i] = x[i];
    log_record(level, source, LOG_HDR_TYPE_BLOB, w, n + 1u);
}
inline uint32_t log_dropped() { return 0; }
[[noreturn]] inline void panic(uint32_t code, uint32_t a0 = 0, uint32_t a1 = 0, uint32_t a2 = 0) {
    fprintf(stderr, "tsr::panic(0x%08x, 0x%08x, 0x%08x, 0x%08x) in a host build of the firmware\n",
            (unsigned)code, (unsigned)a0, (unsigned)a1, (unsigned)a2);
    abort();
}
#endif

// ---------------------------------------------------------------------------------------------
// FMT records: argument words (compile-time count)
// ---------------------------------------------------------------------------------------------
namespace log_detail {

template <class T> constexpr uint32_t words_of() {
    if constexpr (requires(T x) { x.v; }) return 1u;                    // tsr::ticket (%t)
    else if constexpr (requires(T x) { static_cast<const volatile void *>(x); }) return 1u;  // %s, pointers
    else return sizeof(T) > 4u ? 2u : 1u;                               // integers, enums
}

template <class T> inline void put_one(uint32_t *& p, T x) {
    if constexpr (requires { x.v; }) {
        *p++ = static_cast<uint32_t>(x.v);
    } else if constexpr (requires { static_cast<const volatile void *>(x); }) {
        *p++ = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(x));
    } else if constexpr (sizeof(T) > 4u) {
        const unsigned long long v = static_cast<unsigned long long>(x);
        *p++ = static_cast<uint32_t>(v);
        *p++ = static_cast<uint32_t>(v >> 32);
    } else {
        *p++ = static_cast<uint32_t>(x);
    }
}

template <class... A> inline void fmt_record(uint32_t level, uint32_t source, const char * fmt, A... a) {
    constexpr uint32_t n = 1u + (0u + ... + words_of<A>());
    static_assert(n <= 254u, "TSR_LOG: too many argument words");
    uint32_t w[n];
    uint32_t * p = w;
    *p++ = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(fmt));
    (put_one(p, a), ...);
#ifdef TSR_FW_HOST
    if (host_log_sink) host_log_sink(level, source, LOG_HDR_TYPE_FMT, fmt, w + 1, n - 1u);
    else host_print(level, source, fmt, w + 1, n - 1u);
#else
    log_record(level, source, LOG_HDR_TYPE_FMT, w, n);
#endif
}

}  // namespace log_detail
}  // namespace tsr

// The format literal lives in .tsr_logfmt (one array per call site); fmt must be a string literal.
#define TSR_LOG_SRC(level, source, fmt, ...)                                                        \
    do {                                                                                            \
        if constexpr ((level) <= TSR_LOG_MAX_LEVEL) {                                               \
            static const char tsr_logfmt_[] TSR_LOGFMT_ATTR_ = fmt;                                 \
            ::tsr::log_detail::fmt_record(uint32_t(level), uint32_t(source),                        \
                                          tsr_logfmt_ __VA_OPT__(, ) __VA_ARGS__);                  \
        }                                                                                           \
    } while (0)

#define TSR_LOG(level, fmt, ...) TSR_LOG_SRC(level, ::tsr::LOG_HDR_SOURCE_KERNEL, fmt __VA_OPT__(, ) __VA_ARGS__)

#endif  // TSR_LOG_HPP
