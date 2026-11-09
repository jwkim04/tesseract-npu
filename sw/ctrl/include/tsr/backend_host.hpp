// tsr/backend_host.hpp — the host backend of the tile/VU header library (SW §6.8, SW-25):
// namespace tsr::hw on the simulator's test hooks, for kernels compiled with TSR_FW_HOST into the
// host library tsr_fw_host (sw/ctrl/host/backend_host.cpp, sw/ctrl/tests/hdr.cmake). Same
// functions and semantics as backend_dev.hpp, so a kernel pushes the same op stream on both:
//
//   cw_read / cw_write   the RISC-V side control window through tsr_sim_cw_read / _write (the same
//                        registers, including the random schedule's lagging RISC-V-side copies of
//                        DONE_SEQ, credits, QSTATUS and the error records, tsr_sim_int.h §5b)
//   core_read            tsr_sim_corewin_read (CC §5.3)
//   push                 tsr_sim_push (the staging stores + ISSUE_PUSH; the words of a clear WMASK
//                        bit are passed as 0, as the hardware sends them); returns the ticket
//   qctrl                tsr_sim_qctrl
//   pause                advances the simulator to its next event (tsr_sim_next_event), so the
//                        wait loops of issue() / join() / join_irq() / spm_ring progress; a wait
//                        with nothing pending in the simulator is a deadlock: abort with a message
//   mie_set / mie_write  no hart: no-ops; wfi = pause
//   spm_addr             tsr_sim_spm() + byte (both windows; no line buffer on the host)
//   stop_requested       host_ctx::stop (there is no running launch on the host; with rt.hpp,
//                        tsr::stop_requested() is rt_stop_requested(), driven by host_run's stop)
//
// The pushes are test-hook pushes (push PC TSR_SIM_PC_HOOK): the race detector treats them as the
// harness (tsr_race.c), and SPM accesses through spm_ptr are invisible to it (the raw tsr_sim_spm
// pointer). The reference stays the RV binary on tsr_rv (SW §6.8).
//
// A test binds a simulator with host_bind(sim) before calling kernels (it also drops the cached
// ISSUE_UNITS / CAPS of the previous simulator), and runs a kernel with host_run (the shim:
// kernel call, then join_all as the runtime's R6, then the first error as R7, no other R-steps).
#ifndef TSR_BACKEND_HOST_HPP
#define TSR_BACKEND_HOST_HPP

#include <stdint.h>

#include "mmio.hpp"

#if !defined(TSR_FW_HOST)
#error "backend_host.hpp is the host backend: define TSR_FW_HOST (host builds of the kernels)"
#endif

extern "C" {
typedef struct tsr_sim tsr_sim;   // sw/sim/include/tsr_sim.h
}

namespace tsr {

struct host_ctx {
    tsr_sim * sim = nullptr;
    bool stop = false;             // what hw::stop_requested() returns (rt.hpp's host_stop drives
                                   // tsr::stop_requested() when the runtime header is present)
    uint64_t pauses = 0;           // wait-loop steps (diagnostics)
    uint64_t pushes = 0;           // ISSUE_PUSH stores
    uint64_t no_credit_pushes = 0; // pushes (and QCLEARs, which take a SYNC credit) made while the
                                   // target unit had no credit (CC §7.3: the library must never do
                                   // that; the tests assert 0)
};

host_ctx & host();                 // the bound context
void host_bind(tsr_sim * sim);     // binds sim (nullptr unbinds); resets stop and the hardware cache

// The host shim (sw/ctrl/host/shim_host.cpp; SW §6.8): runs one kernel on the bound
// simulator as the runtime would call it, without the R-steps: fills tsr::host_launch (seq,
// kernel, flags, pidx0 = ISSUE_PIDX, base[u] = ISSUED_SEQ[u]) and tsr::host_stop, optionally
// stages the argument block (ki with TSR_KFLAG_ARGBLK: one LD of argblk_lines lines from a[0] into
// the top 16 SPM lines and a join; args.blk then points there), calls the kernel, join_all() (R6),
// and reads the first error relative to pidx0 (R7's rule). Needs rt.hpp (tsr_args).
struct host_result {
    int32_t ret;                   // the kernel's return value
    uint32_t err_code;             // first error (smallest push index since pidx0), 0 = none
    uint32_t err_ticket, err_info, err_w0, err_pidx;
    uint32_t pidx0, pidx_end;      // ISSUE_PIDX before the kernel and after the join_all FENCE
    uint32_t stop_seen;            // the kernel saw stop_requested() true
};

namespace hw {
uint32_t cw_read(uint32_t off);
void cw_write(uint32_t off, uint32_t v);
uint32_t core_read(uint32_t csr_off);
uint32_t push(const op & o, uint32_t dep0, uint32_t dep1, uint32_t push_val);
uint32_t qctrl(uint32_t bits);
void pause();
inline uint32_t mie_set(uint32_t) { return 0u; }
inline void mie_write(uint32_t) {}
inline void wfi() { pause(); }
uintptr_t spm_addr(uint32_t byte);
inline uintptr_t spm_lb_addr(uint32_t byte) { return spm_addr(byte); }
bool stop_requested();
}  // namespace hw

}  // namespace tsr

#if __has_include("rt.hpp")
#include "rt.hpp"
namespace tsr {
host_result host_run(tsr_kernel_fn kernel, const tsr_args & args, const kernel_info * ki = nullptr,
                     uint32_t kernel_index = 0, bool stop = false);
}  // namespace tsr
#endif

#endif  // TSR_BACKEND_HOST_HPP
