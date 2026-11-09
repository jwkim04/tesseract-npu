// test_sim_rv_issue.c — integration (ctest sim_rv_issue): a RISC-V program on tsr_rv drives the
// core model through the control window, launched by the host through the CSR port. The scenario
// (host side, firmware, every check) is test_sim_rvhost.h, shared with sim_sched_determinism.
//
// Co-stepping (SW §4.4): the core's time base (tsr_q.c) and the built-in RISC-V agent (tsr_bus.c)
// advance together, deterministically. The agent maps core time to RV cycles by F_RV / F_CORE
// (tsr_sim_opts.rv_ratio) and runs the hart in slices that end before the next core event; a
// control-window, core-window or SPM access ends its slice, so each such access sees the core
// state of its own instant. The scenario runs at the k4 clocks (166.7 / 276.3 MHz, the default)
// and at 1:1 under the default schedule (TSR_SIM_SCHED, fixed unless the environment says
// otherwise); SW §4.7's schedule-independent observables (records r0-r6, DRAM outputs, s0, log
// words) must be identical, only r7 (DURATION) and MTIME may differ.
//
// Usage: sim_rv_issue <fw_issue.elf>. Exit 77 (ctest "Skipped") when the ELF does not exist (the
// RISC-V compiler was not found at configure time; sw/ctrl/tools/install-toolchain.sh).
#include "test_sim_rvhost.h"

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: sim_rv_issue <fw_issue.elf>\n");
        return 2;
    }
    FILE * f = fopen(argv[1], "rb");
    if (f == NULL) {
        printf("sim_rv_issue: %s not built (no RISC-V compiler at configure time); skipped. "
               "Install it with sw/ctrl/tools/install-toolchain.sh and reconfigure.\n", argv[1]);
        return 77;
    }
    fclose(f);
    static observed a, b;
    memset(&a, 0, sizeof(a));
    memset(&b, 0, sizeof(b));
    const rvh_run ra = { 0.0, NULL, NULL, NULL, NULL, NULL };   // default ratio: 166.7 / 276.3 (SW §4.9)
    const rvh_run rb = { 1.0, NULL, NULL, NULL, NULL, NULL };
    scenario(argv[1], &ra, &a);
    scenario(argv[1], &rb, &b);
    CHECK(a.n_rec == b.n_rec && memcmp(a.rec, b.rec, sizeof(a.rec)) == 0, "records r0-r6 equal at both clock ratios");
    CHECK(a.n_log == b.n_log && memcmp(a.log, b.log, sizeof(a.log)) == 0, "log words equal at both clock ratios");
    CHECK(a.out_hash == b.out_hash && a.s0 == b.s0, "DRAM outputs and s0 equal at both clock ratios");
    return test_finish("sim_rv_issue");
}
