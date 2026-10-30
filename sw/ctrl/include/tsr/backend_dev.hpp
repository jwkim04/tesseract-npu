// tsr/backend_dev.hpp — the device backend of the tile/VU header library (SW §6.7): namespace
// tsr::hw on the control core's own address space (CC §4.1). Included by kernel.hpp when
// TSR_FW_HOST is not defined; backend_host.hpp is the simulator-hook twin (SW §6.8) with the
// same functions and semantics.
//
//   cw_read / cw_write   control window 0x2000_0000 + off (CC §5; word accesses only)
//   core_read            core window 0x2000_1000 + CSR offset (CC §5.3; word loads)
//   push                 one ISSUE_PUSH (CC §5.1, §7.1): the staging words of op::used (w0 always;
//                        words with a clear bit are sent as 0 by the WMASK, so stale staging
//                        contents never matter), the deps that are not none (DEP0 / DEP1 are reset
//                        to none by every push), the push, then ISSUE_TICKET. The caller has checked
//                        the credit; the store itself completes at once then
//   qctrl                one QCTRL write; a QCLEAR's SYNC ticket (ISSUE_TICKET), else none
//   pause                one step of a wait loop (nothing on the device: the loop re-reads)
//   mie_set / mie_write / wfi   Ibex mie (csrrs / csrw) and WFI (CC §12.3); mstatus.MIE is never
//                        set by the SDK, so WFI wakes on a pending enabled line without a trap
//   spm_addr / spm_lb_addr      SPM windows: exact 0x4000_0000, line-buffered 0x5000_0000
//   stop_requested       EX §4.9 from LQ_FLAGS / LQ_HEAD_SEQ / LQ_STOP (below)
#ifndef TSR_BACKEND_DEV_HPP
#define TSR_BACKEND_DEV_HPP

#include <stdint.h>

#include "mmio.hpp"

#if defined(TSR_FW_HOST)
#error "backend_dev.hpp is the RISC-V backend; host builds use backend_host.hpp (TSR_FW_HOST)"
#endif

namespace tsr::hw {

inline volatile uint32_t * cw() { return reinterpret_cast<volatile uint32_t *>(uintptr_t(RV_CTL_BASE)); }
[[gnu::always_inline]] inline uint32_t cw_read(uint32_t off) { return cw()[off / 4u]; }
[[gnu::always_inline]] inline void cw_write(uint32_t off, uint32_t v) { cw()[off / 4u] = v; }
inline uint32_t core_read(uint32_t csr_off) {
    return reinterpret_cast<volatile uint32_t *>(uintptr_t(RV_CORE_BASE))[csr_off / 4u];
}

[[gnu::always_inline]] inline uint32_t push(const op & o, uint32_t dep0, uint32_t dep1, uint32_t push_val) {
    volatile uint32_t * w = cw();
    const uint32_t used = o.used;
    w[CW_ISSUE_W(0) / 4u] = o.w[0];
    if (used & 0x02u) w[CW_ISSUE_W(1) / 4u] = o.w[1];
    if (used & 0x04u) w[CW_ISSUE_W(2) / 4u] = o.w[2];
    if (used & 0x08u) w[CW_ISSUE_W(3) / 4u] = o.w[3];
    if (used & 0x10u) w[CW_ISSUE_W(4) / 4u] = o.w[4];
    if (used & 0x20u) w[CW_ISSUE_W(5) / 4u] = o.w[5];
    if (used & 0x40u) w[CW_ISSUE_W(6) / 4u] = o.w[6];
    if (used & 0x80u) w[CW_ISSUE_W(7) / 4u] = o.w[7];
    if ((dep0 >> 28) != UNIT_NONE) w[CW_ISSUE_DEP0 / 4u] = dep0;
    if ((dep1 >> 28) != UNIT_NONE) w[CW_ISSUE_DEP1 / 4u] = dep1;
    w[CW_ISSUE_PUSH / 4u] = push_val;
    return w[CW_ISSUE_TICKET / 4u];
}

inline uint32_t qctrl(uint32_t bits) {
    cw_write(CW_QCTRL, bits);
    return (bits & QCTRL_QCLEAR) ? cw_read(CW_ISSUE_TICKET) : TICKET_NONE;
}

inline void pause() {}

inline uint32_t mie_set(uint32_t bits) {
    uint32_t old;
    __asm__ volatile("csrrs %0, mie, %1" : "=r"(old) : "r"(bits) : "memory");
    return old;
}
inline void mie_write(uint32_t v) { __asm__ volatile("csrw mie, %0" : : "r"(v) : "memory"); }
inline void wfi() { __asm__ volatile("wfi" : : : "memory"); }

inline uintptr_t spm_addr(uint32_t byte) { return uintptr_t(RV_SPM_BASE) + byte; }
inline uintptr_t spm_lb_addr(uint32_t byte) { return uintptr_t(RV_SPM_LB_BASE) + byte; }

// EX §4.9: STOP_VALID and the running launch covered by LQ_STOP ((LQ_STOP - q) mod 2^32 < 2^31).
// While ACTIVE the running launch is TAKEN - 1 = LQ_HEAD_SEQ - 1 (EX §4.2: TAKEN - ANSWERED is 0
// or 1 and only the RISC-V's own LQ_POP moves TAKEN), so no runtime state is needed. One LQ_FLAGS
// read when no stop is pending.
inline bool stop_requested() {
    const uint32_t f = cw_read(CW_LQ_FLAGS);
    if (!(f & LQ_FLAGS_STOP_VALID) || !(f & LQ_FLAGS_ACTIVE)) return false;
    const uint32_t q = cw_read(CW_LQ_HEAD_SEQ) - 1u;
    return cw_read(CW_LQ_STOP) - q < SEQ_HALF;
}

}  // namespace tsr::hw

#endif  // TSR_BACKEND_DEV_HPP
