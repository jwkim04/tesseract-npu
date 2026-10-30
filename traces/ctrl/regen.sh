#!/usr/bin/env bash
# traces/ctrl/regen.sh [BUILD_DIR] — issue-path regression traces (trace v2 `ops`, docs/v2/06-software-
# verification.md §5.6 "new"; docs/v2/04-control-core.md §7-§9): tsr-trace-gen v2 --ops ctrl, then
# every trace replayed under the fixed and two random schedules (race detector fatal).
#
#   tmax8/    64 traces, U200 config with T_MAX 8, CAPS VU|ATT, U200 depths (8, 4, 4, 16, 0, 4):
#             bursts beyond the queue depth on every unit (credits, back-pressure, issue-FIFO full),
#             data-flow graphs over every unit (fan-out, fan-in with deps on two units, in-place
#             WAR / WAW, NOP joins, FENCE barriers), hold windows filling the queues to their depth,
#             trace marks, REC pushes; ~30% error segments
#   errors/   36 traces, every error segment six times: a fault with a DEP_FAILED chain through every
#             unit and ABORTED entries after it, BAD_DEP variants, the QABORT recipe, QCLEAR
#             recoveries (with and without a later fault), two faults on different units
#   deep/     16 traces, queue depths 31 (the Q_DEPTH field maximum, SYNC included)
#   shallow/  16 traces, queue depths 1 (every push after the first waits for a credit)
#   nocaps/   16 traces, CAPS 0: VU and AT opcodes refused (ILLEGAL_OP on SYNC), the VU queue idle
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
. "$ROOT/traces/v2_tools.sh"
v2_tools "${1:-$ROOT/build/dev}"
GEN=$V2_GEN
rm -rf "$HERE"/tmax8 "$HERE"/errors "$HERE"/deep "$HERE"/shallow "$HERE"/nocaps
"$GEN" -q --ops ctrl --prefix ctrl -o "$HERE/tmax8"   -n 64 -s 61 --t-max 8 --errors 30
"$GEN" -q --ops ctrl --prefix ctrl -o "$HERE/errors"  -n 36 -s 62 --t-max 8 --errors 100
"$GEN" -q --ops ctrl --prefix ctrl -o "$HERE/deep"    -n 16 -s 63 --t-max 8 --errors 30 --q-depth 31,31,31,31,0,31
"$GEN" -q --ops ctrl --prefix ctrl -o "$HERE/shallow" -n 16 -s 64 --t-max 8 --errors 30 --q-depth 1,1,1,1,0,1
"$GEN" -q --ops ctrl --prefix ctrl -o "$HERE/nocaps"  -n 16 -s 65 --t-max 8 --errors 30 --caps none
v2_replay "$HERE"
