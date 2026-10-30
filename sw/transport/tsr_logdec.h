// tsr_logdec.h — decoder of the device log (docs/design/05-debug.md "DB" §6.1 records on EX's log ring,
// docs/design/01-execution-model.md "EX" §4.10; docs/design/06-software-verification.md "SW" §6.6) and of
// TRAP records (EX §4.4, DB §6.3). The transport decodes its log with it (TSR_LOG); tools that hold
// raw words (tsr_transport_log_drain, saved dumps; SW §9.5 tsr-log) use the same functions.
//
// One line per record: "[tsr <LEVEL> <source> seq <LSEQ> t <MTIME[31:0]>] <text>". FMT strings
// come from the image's .tsr_logfmt and %s arguments from its loaded sections (tsr-elf); without an
// image they are shown as addresses. A record whose header fails DB §6.1's checks (SYNC 0xA, TYPE
// 0-4, LEVEL 0-5, a FMT address inside .tsr_logfmt) is reported as a firmware log bug and the
// decoder skips the rest of the words it holds (the transport then resumes at LOG_WR).
#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "tsr_transport.h"

#ifdef __cplusplus
extern "C" {
#endif

// Decodes n raw log words (whole records; a trailing partial record is reported) to out, one line
// per record. img may be NULL. Returns the number of records decoded (bad records included).
long tsr_log_decode(const tsr_image * img, const uint32_t * words, size_t n, FILE * out);

// RISC-V mcause as text: "illegal instruction", "load access fault", "tsr::panic" (24, EX §4.7), ...
const char * tsr_mcause_name(uint32_t mcause);

#ifdef __cplusplus
}

#include <functional>
#include <string>
#include <vector>

namespace tsr {

// LOG_EV_DROPPED (sw/ctrl/include/tsr/rt.hpp, DB §6.1): the EVENT "dropped n" that precedes the next
// record that fits after records were dropped.
constexpr uint32_t LOG_EV_DROPPED = 0xF0000001u;

class log_decoder {
public:
    explicit log_decoder(const tsr_image * img = nullptr) : img_(img) {}
    void set_image(const tsr_image * img) { img_ = img; }   // not owned; may be NULL
    // Appends words; every complete record is decoded and passed to sink (one line, no newline).
    // A partial record stays pending until its remaining words arrive.
    void   feed(const uint32_t * w, size_t n, const std::function<void(const std::string &)> & sink);
    size_t pending_words() const { return pend_.size(); }
    void   reset() { pend_.clear(); }
    uint64_t records = 0;            // records decoded
    uint64_t bad = 0;                // header / format failures (firmware log bugs)
    uint64_t dropped_reported = 0;   // sum of n over "dropped n" EVENT records
    uint32_t last_lseq = 0;          // LSEQ of the record last passed to the sink
    // One complete record (header + N payload words) as text.
    static std::string format(const tsr_image * img, const uint32_t * rec, size_t n, bool * ok = nullptr);

private:
    const tsr_image *     img_;
    std::vector<uint32_t> pend_;
};

// The decoded report of a TRAP record (EX §4.4: r2 mcause, r3 mepc, r4 mtval, r5 RT_STATE, r6
// RT_INFO, r7 MTIME): cause, mepc -> function and line from the image, the kernel.
std::string trap_report(const tsr_image * img, const tsr_record & rec);

// RT_STATE phase name (EX §4.5): "RESET", "BOOT", "IDLE", ... "TRAPPED".
const char * rt_phase_name(uint32_t phase);

} // namespace tsr
#endif
