#!/usr/bin/env bash
# tb/run_all.sh [--no-lint] [pytest args...] — RTL lint (rtl/lint.sh, which also runs the Xilinx
# technology, XRT platform and M0 bandwidth-kernel lint scripts), then every testbench under tb/:
# the block tests (common, dma, fp, qact, spm; M3: mvmfront, mvmback, mvm), tb/tech (Xilinx
# technology wrappers against the generic ones), tb/fastreplay (the C++ memory model self-test),
# tb/platform (the adapter lint, hw/u200/build_xclbin.sh: k1 / k4 disabled until Y5, platform
# lookup) and tb/platform/bw (the M0 bandwidth kernel and tsr_bw on the mock / XRT no-op shim).
# X0 (docs/v2 EX §12, SW §10.3) deleted the benches of the v1 command path: tb/cmdproc; tb/core
# test_core.py (M2 suite) and test_core_m3.py (M3 core replays, rewritten in Y4 as tile-only
# replays of v2 push.bin traces) and mutate_core.py; the cmd.bin replay of tb/fastreplay (run.py,
# fastreplay.cpp, fr_common.h, mutate.py: test_fastreplay_m2 / _impl / _detects_corrupt /
# _detects_mutants); the v1 parts of tb/common (csr_driver, proggen, pycore, replay, simlib,
# trace: test_traces_load, test_replay_compare_detects_mismatches, loop_csr,
# test_core_skeleton_sim, test_pycore_traces, test_proggen_*, test_simlib_keyed_build); the
# tb/platform kernel bench with its start / done tests
# (test_kernel_directed, test_kernel_replay, test_kernel_error_injection, test_packaged_k4 / _k1,
# test_adapter_mutants) and tb/platform/vkernel (v1 vkernel protocol). tb/mvm compares with
# common.mvm_model.mvm_reference instead of the ISA simulator's program runner;
# tb/common test_mvm_model_vs_simulator ties mvm_reference to the simulator's op-level API
# (tsr_sim_op in build/dev/sw/sim/libtsr_sim.so, target tsr_sim_shared). They return with
# their v2 replacements (tb/core v2, tb/fastreplay v2, tb/platform v2, tb/platform/vkernel two
# clocks, and the new tb/ctrl, tb/vu, tb/at, tb/rv, tb/dbg); until then build/v1 and tag v1-m4 are
# the comparison path.
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
#                                       Verilator build cold on a loaded dev server (X0 tree,
#                                       2026-10-30); 142 passed with test_mvm_model_vs_simulator
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
