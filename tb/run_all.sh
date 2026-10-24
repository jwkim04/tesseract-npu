#!/usr/bin/env bash
# tb/run_all.sh [--no-lint] [pytest args...] — RTL lint (rtl/lint.sh, which also runs the Xilinx
# technology, XRT platform and M0 bandwidth-kernel lint scripts), then every testbench under tb/:
# the block tests (cmdproc, common, dma, fp, qact, spm; M3: mvmfront, mvmback, mvm), tb/fastreplay
# (memory model self-test, M2 replay, corruption detection), the core tests: tb/core/test_core.py
# (M2 suite, HAS_MVM = 0: traces/m2, implementation variants, directed, random) and
# tb/core/test_core_m3.py (HAS_MVM = 1: traces/m3 + gen + m2 per configuration in every memory
# regime, ProgGen m3 lockstep programs, implementation variants, PERF / throughput / dependency /
# ABORT / SOFT_RESET / reset tests, also on the U200 bitstream build with RST_PIPE 3 (u200_rst3),
# and a tb/fastreplay smoke of traces/m3 + gen in all regimes),
# and the M3 part C tests: tb/tech (Xilinx technology wrappers against the generic ones),
# tb/platform (the XRT kernel: lint, directed tests, replays through the adapter, error
# injection, the packaged tops, build_xclbin.sh without a platform), tb/platform/bw (the M0
# bandwidth kernel and tsr_bw on the mock / XRT no-op shim) and tb/platform/vkernel (the host
# protocol layer and tsr_xrt_selftest against the Verilated kernel over TSR_DEVICE=vkernel in
# three control modes, adapter error injection, and the kernel-level fastreplay smoke at the U200
# bitstream parameters on rtl/filelist_xilinx.f, T_MAX 1 and 8).
# Slow tests (marker `slow`) are opt-in. tb/pytest.ini does not deselect the marker; each slow test
# skips itself unless it was asked for, by tb/fp/test_fp.py's rule (slow_requested): an -m
# expression naming `slow` (all of them), -k with the test's name, its node id, or its directory's
# switch: TSR_FP_SLOW=1 (tb/fp: the full sweeps; the Vivado OOC runs also need TSR_VIVADO),
# TSR_PLATFORM_SLOW=1 (tb/platform test_adapter_mutants: 5 mutants of mutate_adapter.py, 222 s;
# tb/platform/bw test_bw_mutants: the 57 mutants of mutate_bw.py, 235 s; 2026-10-24, cold builds,
# both pass when opted in) and TSR_FASTREPLAY_SLOW=1 (tb/fastreplay test_fastreplay_detects_mutants:
# 4 M2 RTL mutants of mutate.py, all killed, 21-26 s; 2026-10-24, cold builds).
#   TSR_PLATFORM_SLOW=1 tb/run_all.sh --no-lint platform -k mutants     the two mutant tests
#   TSR_FASTREPLAY_SLOW=1 tb/run_all.sh --no-lint fastreplay -k mutants the fastreplay mutant test
# tb/common also replays traces/gen and traces/m3 through its Python core model
# (docs/rtl-m3.md §10.1). Opt-in campaigns (not run here, docs/rtl-m3.md §13, docs/platform-u200.md):
# tb/fastreplay over traces/model and traces/tmp, tb/core/mutate_core.py, tb/mvmback/mutate_back.py,
# the full tb/platform/mutate_adapter.py campaign, Vivado netlist simulation of the technology
# wrappers (TSR_TECH_VIVADO=1), the model runs over TSR_DEVICE=vkernel (vkernel.py run).
#   tb/run_all.sh                       lint + all tests: 209 passed, 27 opt-in skipped (fp slow
#                                       sweeps, Vivado runs, the three mutant tests), 0 failed,
#                                       1 h 41 min wall on the dev server with every Verilator build
#                                       cold (final M3 part C tree, 2026-10-24; the first part C run,
#                                       with the mutant tests still included: 211 passed, 109 min;
#                                       part B: 170 passed, 87 min; docs/rtl-m3.md §13.4)
#   tb/run_all.sh -k dma                lint + tests matching "dma"
#   tb/run_all.sh --no-lint common      only tb/common (self-test)
#   tb/run_all.sh core --tsr-seed 7     only the core tests, another seed
# Builds use make -j${TSR_BUILD_JOBS:-4}; build dirs under ${TSR_SIM_BUILD_ROOT:-build/cocotb}.
# Opt-in extras (not run here): fp slow sweeps (-m slow or TSR_FP_SLOW=1), the platform mutant
# tests (-m slow or TSR_PLATFORM_SLOW=1), the fastreplay mutant test (-m slow or
# TSR_FASTREPLAY_SLOW=1), Vivado out-of-context
# runs (tb/qact TSR_QACT_VIVADO=1, tb/core/viv/core_ooc.sh, tb/platform/viv/kernel_ooc.tcl).
# tb/platform test_packaged_k1 needs build/dev/bin/tsr-trace-gen (skipped without it).
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
if [ "${1:-}" = "--no-lint" ]; then
    shift
else
    "$ROOT/rtl/lint.sh" || { echo "run_all.sh: lint failed"; exit 1; }
fi
cd "$ROOT/tb" || exit 1
exec python3 -m pytest "$@"
