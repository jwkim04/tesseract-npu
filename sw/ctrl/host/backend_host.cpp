// backend_host.cpp — the host backend of the tile/VU header library (SW §6.8, SW-25):
// tsr::hw on the simulator's test hooks (tsr/backend_host.hpp has the contract). Part of the host
// library tsr_fw_host (sw/ctrl/tests/hdr.cmake), compiled with TSR_FW_HOST.
//
// Every register access is the RISC-V-side control window of the bound simulator
// (tsr_sim_cw_read / _write, tsr_sim_int.h §4), so the library above runs the same code paths as
// on the device: credits, DONE_SEQ, the error records, ISSUE_PIDX, ISSUED_SEQ, QSTATUS, with the
// random schedule's RISC-V-side lag. A wait loop step (pause) advances the simulator to its next
// event; the pushes and QCTRL writes are the test hooks tsr_sim_push / tsr_sim_qctrl.
#include "tsr/kernel.hpp"

#include <stdio.h>
#include <stdlib.h>

#include "tsr_sim.h"

// The simulator-internal calls used here (sw/sim/tsr_sim_int.h §4, §5: exported by libtsr_sim, not
// part of its public header). Declared locally so that every tsr_fw_host variant compiles with the
// public include directory alone.
extern "C" {
uint32_t tsr_sim_cw_read(tsr_sim * sim, uint32_t off);
void     tsr_sim_cw_write(tsr_sim * sim, uint32_t off, uint32_t val);
uint32_t tsr_sim_corewin_read(tsr_sim * sim, uint32_t csr_off);
uint64_t tsr_sim_next_event(const tsr_sim * sim);
}

namespace tsr {

namespace {
host_ctx g_host;

tsr_sim * bound(const char * what) {
    if (g_host.sim == nullptr) {
        fprintf(stderr, "tsr host backend: %s without a bound simulator (tsr::host_bind)\n", what);
        abort();
    }
    return g_host.sim;
}
}  // namespace

host_ctx & host() { return g_host; }

void host_bind(tsr_sim * sim) {
    g_host.sim = sim;
    g_host.stop = false;
    g_host.pauses = 0;
    g_host.pushes = 0;
    g_host.no_credit_pushes = 0;
    detail::g_hw_units = 0;          // ISSUE_UNITS / CAPS are per simulator: reload on first use
    detail::g_hw_caps = 0;
    for (uint8_t & u : detail::g_unit_tab) u = 0;
}

namespace hw {

uint32_t cw_read(uint32_t off) { return tsr_sim_cw_read(bound("cw_read"), off); }
void cw_write(uint32_t off, uint32_t v) { tsr_sim_cw_write(bound("cw_write"), off, v); }
uint32_t core_read(uint32_t csr_off) { return tsr_sim_corewin_read(bound("core_read"), csr_off); }

uint32_t push(const op & o, uint32_t dep0, uint32_t dep1, uint32_t push_val) {
    tsr_sim * sim = bound("push");
    uint32_t w[8];
    w[0] = o.w[0];
    for (uint32_t i = 1; i < 8u; ++i) w[i] = (o.used & (1u << i)) ? o.w[i] : 0u;
    // Diagnostic: the credit of the unit the push lands on, as the RISC-V side sees it now (the
    // push itself would wait like the hardware's safety net, CC §7.3, if it were 0).
    const uint32_t credits = tsr_sim_cw_read(sim, CW_ISSUE_CREDIT);
    const uint32_t t = tsr_sim_push(sim, w, dep0, dep1, push_val);
    ++g_host.pushes;
    if ((t >> 28) < NUM_UNITS && ((credits >> (CREDIT_FIELD_W * (t >> 28))) & 0xFu) == 0u) ++g_host.no_credit_pushes;
    return t;
}

uint32_t qctrl(uint32_t bits) {
    tsr_sim * sim = bound("qctrl");
    // A QCLEAR takes a SYNC credit like a push (CC §5.1, §7.3): the same diagnostic as push() (fix W2a).
    if (bits & QCTRL_QCLEAR) {
        const uint32_t credits = tsr_sim_cw_read(sim, CW_ISSUE_CREDIT);
        if (((credits >> (CREDIT_FIELD_W * UNIT_SYNC)) & 0xFu) == 0u) ++g_host.no_credit_pushes;
    }
    return tsr_sim_qctrl(sim, bits);
}

void pause() {
    tsr_sim * sim = bound("pause");
    ++g_host.pauses;
    const uint64_t now = tsr_sim_now(sim);
    const uint64_t next = tsr_sim_next_event(sim);
    if (next == UINT64_MAX) {
        fprintf(stderr,
                "tsr host backend: a wait loop with nothing pending in the simulator (deadlock: a hold, or a "
                "ticket that can never complete) at MTIME %llu\n",
                (unsigned long long)now);
        abort();
    }
    tsr_sim_advance(sim, next > now ? next - now : 1u);
}

uintptr_t spm_addr(uint32_t byte) { return reinterpret_cast<uintptr_t>(tsr_sim_spm(bound("spm_ptr"))) + byte; }

bool stop_requested() { return g_host.stop; }

}  // namespace hw
}  // namespace tsr
