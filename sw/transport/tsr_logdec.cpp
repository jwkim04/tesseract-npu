// tsr_logdec.cpp — the log and TRAP-record decoder of tsr_logdec.h (DB §6.1, §6.3; EX §4.4, §4.10;
// SW §6.6). Formats of the firmware side: sw/ctrl/include/tsr/log.hpp (conversions %d %u %x %08x
// %f %t %lld %llu %llx %s; a 64-bit argument takes two words, lo then hi), sw/ctrl/rt/log.cpp.

#include "tsr_logdec.h"

#include "tsr_elf.h"
#include "tsr_image_int.h"

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <string>

namespace tsr {

namespace {

const char * const k_level[6] = { "FATAL", "ERROR", "WARN", "INFO", "DEBUG", "TRACE" };
const char * const k_source[4] = { "runtime", "kernel", "trap", "debug" };

std::string hex32(uint32_t v) {
    char b[16];
    snprintf(b, sizeof(b), "0x%08x", v);
    return b;
}

float f32_of(uint32_t u) {
    float f;
    memcpy(&f, &u, 4);
    return f;
}

// FMT: the format string at addr in .tsr_logfmt, arguments a[0..n).
bool format_fmt(const tsr_image * img, uint32_t addr, const uint32_t * a, size_t n, std::string * out) {
    const tsr_elf * e = img ? image_elf(img) : nullptr;
    const char * fmt = e ? tsr_elf_logfmt(e, addr) : nullptr;
    if (fmt == nullptr) {
        if (e != nullptr) return false;   // DB §6.1: the address must lie in .tsr_logfmt
        *out += "<fmt " + hex32(addr) + ">";
        for (size_t i = 0; i < n; ++i) *out += " " + hex32(a[i]);
        return true;
    }
    size_t k = 0;
    char buf[64];
    for (const char * p = fmt; *p; ++p) {
        if (*p != '%') {
            *out += *p;
            continue;
        }
        ++p;
        if (*p == '%') {
            *out += '%';
            continue;
        }
        if (*p == '\0') break;
        bool zero8 = false, ll = false;
        if (p[0] == '0' && p[1] == '8') {
            zero8 = true;
            p += 2;
        }
        if (p[0] == 'l' && p[1] == 'l') {
            ll = true;
            p += 2;
        }
        const uint32_t lo = k < n ? a[k] : 0u;
        const uint32_t hi = (ll && k + 1 < n) ? a[k + 1] : 0u;
        k += ll ? 2u : 1u;
        const unsigned long long v = ((unsigned long long) hi << 32) | lo;
        switch (*p) {
        case 'd':
            if (ll) snprintf(buf, sizeof(buf), "%lld", (long long) v);
            else snprintf(buf, sizeof(buf), "%d", (int) (int32_t) lo);
            break;
        case 'u': snprintf(buf, sizeof(buf), "%llu", ll ? v : (unsigned long long) lo); break;
        case 'x':
            if (zero8) snprintf(buf, sizeof(buf), "%08x", (unsigned) lo);
            else snprintf(buf, sizeof(buf), "%llx", ll ? v : (unsigned long long) lo);
            break;
        case 'f': snprintf(buf, sizeof(buf), "%.9g", (double) f32_of(lo)); break;
        case 't':
            if (lo == TSR_TICKET_NONE) snprintf(buf, sizeof(buf), "t(none)");
            else snprintf(buf, sizeof(buf), "t(%u,%u)", (unsigned) (lo >> 28), (unsigned) (lo & 0x0FFFFFFFu));
            break;
        case 's': {
            const char * s = e ? tsr_elf_string(e, lo) : nullptr;
            if (s) *out += s;
            else *out += "<str " + hex32(lo) + ">";
            buf[0] = '\0';
            break;
        }
        default: snprintf(buf, sizeof(buf), "%%%s%c", ll ? "ll" : "", *p); break;
        }
        *out += buf;
    }
    if (k < n) {   // more argument words than conversions: show them, do not hide data
        *out += " [+";
        for (; k < n; ++k) *out += " " + hex32(a[k]);
        *out += "]";
    }
    return true;
}

std::string marker_text(uint32_t code) {
    static const char * const cls[7] = { "user", "launch-begin", "launch-end", "node-begin", "node-end", "layer", "phase" };
    const uint32_t c = TSR_GET(code, TSR_MARK_CLASS), pl = TSR_GET(code, TSR_MARK_PAYLOAD);
    char b[96];
    if (c < 7) snprintf(b, sizeof(b), "event %s 0x%07x", cls[c], pl);
    else snprintf(b, sizeof(b), "event 0x%08x", code);
    return b;
}

} // namespace

const char * rt_phase_name(uint32_t phase) {
    static const char * const n[8] = { "RESET", "BOOT", "IDLE", "DISPATCH", "KERNEL", "JOIN", "ANSWER", "TRAPPED" };
    return phase < 8 ? n[phase] : "?";
}

std::string log_decoder::format(const tsr_image * img, const uint32_t * rec, size_t n, bool * ok) {
    if (ok) *ok = true;
    const uint32_t h = rec[0];
    const uint32_t N = TSR_GET(h, TSR_LOG_HDR_N), level = TSR_GET(h, TSR_LOG_HDR_LEVEL), source = TSR_GET(h, TSR_LOG_HDR_SOURCE),
                   type = TSR_GET(h, TSR_LOG_HDR_TYPE), lseq = TSR_GET(h, TSR_LOG_HDR_LSEQ);
    const bool ts = (h & TSR_LOG_HDR_TS) != 0;
    char head[96];
    if (TSR_GET(h, TSR_LOG_HDR_SYNC) != TSR_LOG_HDR_SYNC_MAGIC || type > TSR_LOG_HDR_TYPE_BLOB || level > TSR_LOG_HDR_LEVEL_TRACE ||
        n != 1u + N || (ts && N == 0)) {
        if (ok) *ok = false;
        return "[tsr log] firmware log bug: bad record header " + hex32(h);
    }
    const uint32_t * p = rec + 1;
    size_t m = N;
    uint32_t t = 0;
    if (ts) {
        t = *p++;
        --m;
    }
    snprintf(head, sizeof(head), "[tsr %s %s seq %u", k_level[level], source < 4 ? k_source[source] : "?", lseq);
    std::string s = head;
    if (ts) {
        snprintf(head, sizeof(head), " t %u", t);
        s += head;
    }
    s += "] ";
    switch (type) {
    case TSR_LOG_HDR_TYPE_FMT:
        if (m == 0 || !format_fmt(img, p[0], p + 1, m - 1, &s)) {
            if (ok) *ok = false;
            return "[tsr log] firmware log bug: FMT record " + hex32(h) + " with format address " +
                   (m ? hex32(p[0]) : std::string("(none)")) + " outside .tsr_logfmt";
        }
        break;
    case TSR_LOG_HDR_TYPE_TEXT:
        for (size_t i = 0; i < 4 * m; ++i) {
            const char c = (char) (p[i / 4] >> (8 * (i % 4)));
            if (c == '\0') break;
            s += c;
        }
        break;
    case TSR_LOG_HDR_TYPE_TRAP: {
        // The 44 words of __tsr_crash (DB §6.3).
        if (m < TSR_CRASH_LOG_WORDS || p[TSR_CRASH_MAGIC_WORD] != TSR_CRASH_MAGIC) {
            s += "TRAP record (malformed, " + std::to_string(m) + " words)";
            break;
        }
        const uint32_t mc = p[TSR_CRASH_MCAUSE_WORD], epc = p[TSR_CRASH_MEPC_WORD];
        char b[256];
        snprintf(b, sizeof(b), "TRAP %s (mcause %u) at ", tsr_mcause_name(mc), mc);
        s += b;
        s += image_symbolize(img, epc);
        snprintf(b, sizeof(b), ", mtval 0x%08x, launch seq %u kernel %u, RT_STATE %s, RT_INFO 0x%08x, ra 0x%08x sp 0x%08x%s",
                 p[TSR_CRASH_MTVAL_WORD], p[TSR_CRASH_SEQ_WORD], p[TSR_CRASH_KERNEL_WORD],
                 rt_phase_name(TSR_GET(p[TSR_CRASH_RT_STATE_WORD], TSR_RT_STATE_PHASE)), p[TSR_CRASH_RT_INFO_WORD],
                 p[TSR_CRASH_X1_WORD], p[TSR_CRASH_X1_WORD + 1], p[TSR_CRASH_NESTED_WORD] > 1 ? " (nested)" : "");
        s += b;
        // ra points after the call; ra - 1 lies in the calling instruction (as addr2line users do).
        if (img && p[TSR_CRASH_X1_WORD] != 0) s += "; called from " + image_symbolize(img, p[TSR_CRASH_X1_WORD] - 1);
        break;
    }
    case TSR_LOG_HDR_TYPE_EVENT:
        if (m == 0) {
            s += "event (no code)";
        } else if (p[0] == LOG_EV_DROPPED) {
            s += "dropped " + std::to_string(m > 1 ? p[1] : 0) + " log record(s)";
        } else {
            s += marker_text(p[0]);
            for (size_t i = 1; i < m; ++i) s += " " + hex32(p[i]);
        }
        break;
    default:   // BLOB
        s += "blob";
        if (m) s += " tag " + hex32(p[0]);
        for (size_t i = 1; i < m; ++i) s += " " + hex32(p[i]);
        break;
    }
    return s;
}

void log_decoder::feed(const uint32_t * w, size_t n, const std::function<void(const std::string &)> & sink) {
    pend_.insert(pend_.end(), w, w + n);
    size_t i = 0;
    while (i < pend_.size()) {
        const uint32_t h = pend_[i];
        const uint32_t N = TSR_GET(h, TSR_LOG_HDR_N);
        if (TSR_GET(h, TSR_LOG_HDR_SYNC) != TSR_LOG_HDR_SYNC_MAGIC || TSR_GET(h, TSR_LOG_HDR_TYPE) > TSR_LOG_HDR_TYPE_BLOB ||
            TSR_GET(h, TSR_LOG_HDR_LEVEL) > TSR_LOG_HDR_LEVEL_TRACE) {
            // DB §6.1: report and skip to LOG_WR (everything held now).
            bad++;
            if (sink) sink("[tsr log] firmware log bug: bad record header " + hex32(h) + "; skipping " +
                           std::to_string(pend_.size() - i) + " word(s)");
            pend_.clear();
            return;
        }
        if (pend_.size() - i < 1u + N) break;   // partial record: wait for the rest
        bool ok = true;
        const std::string line = format(img_, &pend_[i], 1u + N, &ok);
        records++;
        if (!ok) bad++;
        const uint32_t tsw = (h & TSR_LOG_HDR_TS) ? 1u : 0u;
        if (TSR_GET(h, TSR_LOG_HDR_TYPE) == TSR_LOG_HDR_TYPE_EVENT && N >= 2u + tsw && pend_[i + 1 + tsw] == LOG_EV_DROPPED)
            dropped_reported += pend_[i + 2 + tsw];
        last_lseq = TSR_GET(h, TSR_LOG_HDR_LSEQ);
        if (sink) sink(line);
        i += 1u + N;
    }
    pend_.erase(pend_.begin(), pend_.begin() + (long) i);
}

std::string trap_report(const tsr_image * img, const tsr_record & rec) {
    const uint32_t * r = rec.w;
    const uint32_t mc = r[TSR_LCQ_TRAP_MCAUSE_WORD], epc = r[TSR_LCQ_TRAP_MEPC_WORD], rt = r[TSR_LCQ_TRAP_RT_STATE_WORD];
    const uint32_t kidx = TSR_GET(rt, TSR_RT_STATE_KERNEL), phase = TSR_GET(rt, TSR_RT_STATE_PHASE);
    char b[320];
    std::string s;
    snprintf(b, sizeof(b), "TRAP: %s (mcause %u) at ", tsr_mcause_name(mc), mc);
    s = b;
    s += image_symbolize(img, epc);
    if (mc == TSR_PANIC_MCAUSE) snprintf(b, sizeof(b), ", panic code 0x%08x", r[TSR_LCQ_TRAP_MTVAL_WORD]);
    else snprintf(b, sizeof(b), ", mtval 0x%08x", r[TSR_LCQ_TRAP_MTVAL_WORD]);
    s += b;
    snprintf(b, sizeof(b), "; RT_STATE %s", rt_phase_name(phase));
    s += b;
    if (phase == TSR_RT_STATE_PHASE_KERNEL || phase == TSR_RT_STATE_PHASE_JOIN || phase == TSR_RT_STATE_PHASE_DISPATCH) {
        const char * kn = img ? image_kernel_name(img, kidx) : "";
        snprintf(b, sizeof(b), " kernel %u%s%s%s", kidx, kn[0] ? " (" : "", kn, kn[0] ? ")" : "");
        s += b;
    }
    snprintf(b, sizeof(b), ", RT_INFO 0x%08x, MTIME %u", r[TSR_LCQ_TRAP_RT_INFO_WORD], r[TSR_LCQ_TRAP_MTIME_WORD]);
    s += b;
    return s;
}

} // namespace tsr

extern "C" const char * tsr_mcause_name(uint32_t mc) {
    if (mc & 0x80000000u) return "interrupt";
    switch (mc) {
    case 0:  return "instruction address misaligned";
    case 1:  return "instruction access fault";
    case 2:  return "illegal instruction";
    case 3:  return "breakpoint (ebreak)";
    case 4:  return "load address misaligned";
    case 5:  return "load access fault";
    case 6:  return "store address misaligned";
    case 7:  return "store access fault";
    case 8:  return "ecall from U-mode";
    case 11: return "ecall from M-mode";
    case TSR_PANIC_MCAUSE: return "tsr::panic";
    default: return "unknown cause";
    }
}

extern "C" long tsr_log_decode(const tsr_image * img, const uint32_t * words, size_t n, FILE * out) {
    tsr::log_decoder d(img);
    d.feed(words, n, [out](const std::string & l) {
        if (out) fprintf(out, "%s\n", l.c_str());
    });
    if (d.pending_words() && out) fprintf(out, "[tsr log] %zu word(s) of an incomplete record at the end\n", d.pending_words());
    return (long) d.records;
}
