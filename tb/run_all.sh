#!/usr/bin/env bash
# tb/run_all.sh [--no-lint] [pytest args...] — RTL lint (rtl/lint.sh, which also runs the Xilinx
# technology, XRT platform and M0 bandwidth-kernel lint scripts), then every testbench under tb/:
# the block tests (common, dma, fp, qact, spm; M3: mvmfront, mvmback, mvm), tb/tech (Xilinx
# technology wrappers against the generic ones), tb/fastreplay (the C++ memory model self-test),
# tb/platform (the adapter lint, hw/u200/build_xclbin.sh: k1 / k4 disabled until Y5, platform
# lookup) and tb/platform/bw (the M0 bandwidth kernel and tsr_bw on the mock / XRT no-op shim).
# The core-level benches (tb/core, tb/fastreplay trace replay, the tb/platform kernel bench and
# tb/platform/vkernel) come with the control core (docs/design SW §10.3), as do tb/ctrl, tb/vu,
# tb/at, tb/rv and tb/dbg. Until then tb/mvm compares with common.mvm_model.mvm_reference, and
# tb/common test_mvm_model_vs_simulator ties mvm_reference to the simulator's op-level API
# (tsr_sim_op in build/dev/sw/sim/libtsr_sim.so, target tsr_sim_shared).
# Slow tests (marker `slow`) are opt-in. tb/pytest.ini does not deselect the marker; each slow test
# skips itself unless it was asked for, by tb/fp/test_fp.py's rule (slow_requested): an -m
# expression naming `slow` (all of them), -k with the test's name, its node id, or its directory's
# switch: TSR_FP_SLOW=1 (tb/fp: the full sweeps; the Vivado OOC runs also need TSR_VIVADO),
# TSR_PLATFORM_SLOW=1 (tb/platform/bw test_bw_mutants: the 57 mutants of mutate_bw.py, 235 s;
# 2026-10-24, cold builds, pass when opted in).
#   TSR_PLATFORM_SLOW=1 tb/run_all.sh --no-lint platform/bw -k mutants     the bw mutant test
# Opt-in campaigns (not run here, docs/rtl-m3.md §13, docs/platform-u200.md): tb/mvmback/mutate_back.py,
# Vivado netlist simulation of the technology wrappers (TSR_TECH_VIVADO=1).
#   tb/run_all.sh                       lint + all tests: lint all clean (9 min); 141 passed, 25 opt-in
#                                       skipped (fp slow sweeps, Vivado runs, the bw mutant test),
#                                       0 failed, 25 min wall with TSR_BUILD_JOBS=2 and every
#                                       Verilator build cold on a loaded dev server (2026-10-30); 142 passed with test_mvm_model_vs_simulator
#                                       back, 9.7 min with the Verilator builds warm (same day)
#   tb/run_all.sh -k dma                lint + tests matching "dma"
#   tb/run_all.sh --no-lint common      only tb/common (self-test)
#   tb/run_all.sh mvm --tsr-seed 7      only the MVM unit tests, another seed
# Builds use make -j${TSR_BUILD_JOBS:-4}; build dirs under ${TSR_SIM_BUILD_ROOT:-build/cocotb}.
# Opt-in extras (not run here): fp slow sweeps (-m slow or TSR_FP_SLOW=1), the bw mutant test
# (-m slow or TSR_PLATFORM_SLOW=1), Vivado out-of-context runs (tb/qact TSR_QACT_VIVADO=1,
# tb/core/viv/core_ooc.sh (no work in the core until Y4), tb/platform/viv/kernel_ooc.tcl).
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
if [ "${1:-}" = "--no-lint" ]; then
    shift
else
    "$ROOT/rtl/lint.sh" || { echo "run_all.sh: lint failed"; exit 1; }
fi
cd "$ROOT/tb" || exit 1
exec python3 -m pytest "$@"
