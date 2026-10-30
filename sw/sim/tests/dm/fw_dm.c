// fw_dm.c — the RISC-V program of sim_dm_fw (sw/sim/tests/test_sim_dm.c --fw): plain C that a
// debugger session works on (P1.12, DB §3.5-§3.7). Built by the pinned xPack riscv-none-elf-gcc
// with tests/rv_issue's crt0.S and link.ld (sw/sim/dm.cmake); loaded straight into the TCMs.
//
// The loop squares the volatile input, stores the result, reports it (RT_INFO and a 16-entry
// trace ring) and counts. The host sets a hardware breakpoint at dm_square (argument in a0), a
// software breakpoint at dm_report written through SBA, steps, and reads / writes the globals
// through SBA while the hart runs. noipa keeps both functions out of line with their own symbols
// (no inlining, cloning or IPA register changes).
#include <stdint.h>

#include "tsr_isa.h"

#define CW(o) (*(volatile uint32_t *)(uintptr_t)(TSR_RV_CTL_BASE + (o)))   // control window, CC §5

volatile uint32_t g_input = 5u;
volatile uint32_t g_result;
volatile uint32_t g_count;
volatile uint32_t g_trace[16];

static __attribute__((noipa, used)) uint32_t dm_square(uint32_t x) { return x * x; }

static __attribute__((noipa, used)) void dm_report(uint32_t v) {
    g_trace[g_count & 15u] = v;
    CW(TSR_CW_RT_INFO) = v;
}

int main(void);
int main(void) {
    for (;;) {
        const uint32_t r = dm_square(g_input);
        g_result = r;
        dm_report(r);
        g_count = g_count + 1u;
    }
}
