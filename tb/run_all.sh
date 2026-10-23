#!/usr/bin/env bash
# tb/run_all.sh [--no-lint] [pytest args...] — RTL lint, then every testbench under tb/: the block
# tests (cmdproc, common, dma, fp, qact, spm; M3: mvmfront, mvmback, mvm), tb/fastreplay (memory
# model self-test, M2 replay, corruption detection) and the core tests: tb/core/test_core.py (M2
# suite, HAS_MVM = 0: traces/m2, implementation variants, directed, random) and
# tb/core/test_core_m3.py (HAS_MVM = 1: traces/m3 + gen + m2 per configuration in every memory
# regime, ProgGen m3 lockstep programs, implementation variants, PERF / throughput / dependency /
# ABORT / SOFT_RESET / reset tests, and a tb/fastreplay smoke of traces/m3 + gen in all regimes).
# tb/common also replays traces/gen and traces/m3 through its Python core model
# (docs/rtl-m3.md §10.1). Opt-in campaigns (not run here, docs/rtl-m3.md §13): tb/fastreplay over
# traces/model and traces/tmp, tb/core/mutate_core.py, tb/mvmback/mutate_back.py.
#   tb/run_all.sh                       lint + all tests: 170 passed, 5 opt-in skipped, 87 min wall
#                                       (5206 s) on the dev server with every Verilator build cold, one
#                                       simulation at a time (M3 integration, 2026-10-23)
#   tb/run_all.sh -k dma                lint + tests matching "dma"
#   tb/run_all.sh --no-lint common      only tb/common (self-test)
#   tb/run_all.sh core --tsr-seed 7     only the core tests, another seed
# Builds use make -j${TSR_BUILD_JOBS:-4}; build dirs under ${TSR_SIM_BUILD_ROOT:-build/cocotb}.
# Opt-in extras (not run here): fp slow sweeps (-m slow), Vivado out-of-context runs
# (tb/qact TSR_QACT_VIVADO=1, tb/core/viv/core_ooc.tcl).
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
if [ "${1:-}" = "--no-lint" ]; then
    shift
else
    "$ROOT/rtl/lint.sh" || { echo "run_all.sh: lint failed"; exit 1; }
fi
cd "$ROOT/tb" || exit 1
exec python3 -m pytest "$@"
