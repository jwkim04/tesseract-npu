#!/usr/bin/env bash
# make_report.sh CARD_DIR SIM_DIR OUT_DIR [MODEL_NAME]: the profile report (docs/prof-dbg.md "The profile report") from the dump
# directories that capture_card.sh and capture_sim.sh wrote. The scheduler's splits are taken from the simulator capture's log.
#
#   EXPECT_CYCLES   cycles of the decode launch measured earlier, for a check (default 16316000: Gemma 4 26B-A4B, docs/performance.md)
#   REF_TOKS        tok/s of an earlier run with the profiler off, shown beside the speed of this run
#   LAUNCH          the decode launch to show (default: the middle one both runs have)
#   REF_RUN         directory of the same run without the profiler (capture_card.sh with NO_DUMP=1)
set -euo pipefail
card=$1; sim=$2; out=$3; name=${4:-}
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PY=${TSR_PYTHON:-/home/limagin-dev/miniconda3/bin/python}
for d in "$card" "$sim"; do
  if [ ! -s "$d/sched.json" ] && [ -s "$d/sched_and_run.log" ]; then "$PY" "$here/prof_report.py" sched "$d/sched_and_run.log" "$d/sched.json"; fi
done
cmds="# commands (git $(git -C "$here" rev-parse --short HEAD 2>/dev/null || echo ?))
sw/tools/prof/capture_card.sh MODEL DIR_CARD 33
TSR_SIM_DRAM_DIR=build/simdram sw/tools/prof/capture_sim.sh MODEL DIR_SIM 33
sw/tools/prof/make_report.sh DIR_CARD DIR_SIM OUT_DIR \"NAME\""
"$PY" "$here/prof_report.py" report --card "$card" ${sim:+--sim "$sim"} --out "$out" ${name:+--model-name "$name"} \
  ${EXPECT_CYCLES:+--expect-cycles "$EXPECT_CYCLES"} ${REF_TOKS:+--ref-toks "$REF_TOKS"} ${LAUNCH:+--launch "$LAUNCH"} ${REF_RUN:+--ref-run "$REF_RUN"} --commands "$cmds"
