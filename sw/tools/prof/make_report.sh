#!/usr/bin/env bash
# make_report.sh CARD_DIR SIM_DIR OUT_DIR [MODEL_NAME]: the profile report (docs/prof-dbg.md "The profile report") from the dump
# directories that capture_card.sh and capture_sim.sh wrote. The scheduler's splits are taken from the simulator capture's log.
#
#   EXPECT_CYCLES   cycles of the decode launch measured earlier, for a check (no default: unset skips the check)
#   REF_TOKS        tok/s of an earlier run with the profiler off, shown beside the speed of this run
#   LAUNCH          the decode launch to show (default: the middle one both runs have)
#   REF_RUN         directory of the same run without the profiler (capture_card.sh with NO_DUMP=1)
set -euo pipefail
card=$1; sim=$2; out=$3; name=${4:-}
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PY=${TSR_PYTHON:-python3}
for d in "$card" "$sim"; do
  if [ ! -s "$d/sched.json" ] && [ -s "$d/sched_and_run.log" ]; then "$PY" "$here/prof_report.py" sched "$d/sched_and_run.log" "$d/sched.json"; fi
done
# The timing bits of CAPS (16 MVM_DUAL, 17 QACT_PIPE) must be the same on both sides, or the simulator's times do not belong to the card's.
if [ -s "$card/meta.json" ] && [ -s "$sim/meta.json" ]; then
  "$PY" -I - "$card/meta.json" "$sim/meta.json" <<'EOF2' || true
import json, sys
c, s = (json.load(open(f)).get("caps") for f in sys.argv[1:3])
if c is not None and s is not None and (c ^ s) & 0x30000:
    print("make_report.sh: warning: CAPS of the card run 0x%x and of the simulator run 0x%x differ in the timing bits 16 and 17: "
          "capture_sim.sh with SIM_CARD=k4dual or k4qpipe, as the card image has them" % (c, s), file=sys.stderr)
EOF2
fi
cmds="# commands (git $(git -C "$here" rev-parse --short HEAD 2>/dev/null || echo ?))
sw/tools/prof/capture_card.sh MODEL DIR_CARD 33
TSR_SIM_DRAM_DIR=build/simdram sw/tools/prof/capture_sim.sh MODEL DIR_SIM 33
sw/tools/prof/make_report.sh DIR_CARD DIR_SIM OUT_DIR \"NAME\""
"$PY" "$here/prof_report.py" report --card "$card" ${sim:+--sim "$sim"} --out "$out" ${name:+--model-name "$name"} \
  ${EXPECT_CYCLES:+--expect-cycles "$EXPECT_CYCLES"} ${REF_TOKS:+--ref-toks "$REF_TOKS"} ${LAUNCH:+--launch "$LAUNCH"} ${REF_RUN:+--ref-run "$REF_RUN"} --commands "$cmds"
