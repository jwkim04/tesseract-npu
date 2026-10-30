// rt_int.hpp — internals shared by the runtime sources (crt0.S's C side, runtime.cpp, trap.cpp,
// log.cpp; docs/v2/06-software-verification.md "SW" §6.3-§6.6). Not part of the SDK API.
//
// Register access goes straight to the control window (CC §5) and the core window (CC §5.3)
// through the generated offsets of tsr/mmio.hpp. Waits use WFI with mstatus.MIE = 0: the runtime
// never takes an interrupt, it only selects wake-up lines in mie (CC §12.3, SW §4.2), so there are
// no interrupt handlers (and no logging from one, SW §6.6).
#ifndef TSR_RT_INT_HPP
#define TSR_RT_INT_HPP

#include <stdint.h>

#include "tsr/mmio.hpp"
#include "tsr/rt.hpp"

namespace tsr::rt {

inline volatile uint32_t & cw(uint32_t off) {           // control window, word access only
    return *reinterpret_cast<volatile uint32_t *>(uintptr_t(RV_CTL_BASE + off));
}
inline uint32_t core(uint32_t off) {                     // core window: word loads of the host map
    return *reinterpret_cast<const volatile uint32_t *>(uintptr_t(RV_CORE_BASE + off));
}

constexpr uint32_t mie_fast(uint32_t i) { return 1u << (16u + i); }   // irq_fast[i] in mie
inline uint32_t mie_swap(uint32_t v) {
    uint32_t old;
    __asm__ volatile("csrrw %0, mie, %1" : "=r"(old) : "r"(v) : "memory");
    return old;
}
inline void wfi() { __asm__ volatile("wfi" : : : "memory"); }

// MTIME (CC §5.2): 64-bit read HI, LO, HI.
inline uint64_t mtime64() {
    for (;;) {
        const uint32_t hi = cw(CW_MTIME_HI), lo = cw(CW_MTIME_LO);
        if (cw(CW_MTIME_HI) == hi) return (uint64_t(hi) << 32) | lo;
    }
}

// RT_STATE (EX §4.5): [7:0] phase, [31:16] kernel index.
inline void set_state(uint32_t phase, uint32_t kernel = 0) {
    cw(CW_RT_STATE) = phase | (kernel << RT_STATE_KERNEL_LSB);
}

// Posts staged words r0..r7 to the LCQ (EX §4.5): waits with WFI on irq_fast[3] while LCQ_FREE = 0
// (R8; the value only ends the wait, SW §4.7), then LCQ_POST with ANSWER.
void post(const uint32_t r[8], uint32_t answer);

// Log writer shared by log.cpp's API and the trap handler (allow_block = 0: never wait, even with
// LOG_CFG.BLOCK: a trap must reach its LCQ record).
void log_write(uint32_t level, uint32_t source, uint32_t type, const volatile uint32_t * payload, uint32_t n,
               int allow_block);
// The same with npre (0 or 1) words `pre` in front of the payload (EVENT code, BLOB tag).
void log_write2(uint32_t level, uint32_t source, uint32_t type, uint32_t pre, uint32_t npre,
                const volatile uint32_t * payload, uint32_t n, int allow_block);

}  // namespace tsr::rt

// Debug contract (DB §6.3): __tsr_dbg (.tsr_dbg, 16 words), __tsr_crash (.tsr_crash, 48 words).
extern "C" volatile uint32_t __tsr_dbg[tsr::DBG_INFO_NWORDS];
extern "C" volatile uint32_t __tsr_crash[tsr::CRASH_NWORDS];
// .tsr_id (EX §3.2): the image id, written into the ELF by tsr-elf after linking.
extern "C" const volatile uint32_t __tsr_image_id[2];

#endif  // TSR_RT_INT_HPP
