#!/usr/bin/env bash
# traces/gen/regen.sh [BUILD_DIR] — the M1 generator's LD -> QACT -> MVM (-> MVM ACC) -> ST programs
# as trace v2 `ops` traces (docs/v2/06-software-verification.md §5.6): tsr-trace-gen v2 --ops gen with
# v1's seeds and configurations (tag v1-m4: the tsr-traces target of sw/tools/CMakeLists.txt), then
# every trace replayed under the fixed and two random schedules (race detector fatal),
# tile-only and through the replay kernel (one launch per trace: SPM_STALE is fatal, SW §4.6).
#
#   tmax1/  24 traces, U200 config (4 ch, T_MAX 1, 4 MiB SPM, 16 GiB/ch, CAPS 0), ~15% error injections
#   tmax8/   8 traces, U200 config with T_MAX 8
# Groups are interleaved, so the DMA, QACT and MVM queues run concurrently with deps between them.
# Every SPM line an op reads is written earlier in its trace (2026-10-30: a strided QACT source loaded per
# token also gets its stride gaps loaded; 5 traces changed, none read the power-on SPM fill).
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
. "$ROOT/traces/v2_tools.sh"
v2_tools "${1:-$ROOT/build/dev}"
GEN=$V2_GEN
rm -rf "$HERE"/tmax1 "$HERE"/tmax8
"$GEN" -q --ops gen --prefix gen -o "$HERE/tmax1" -n 24 -s 1 --max-k 2048 --max-n 96 --errors 15
"$GEN" -q --ops gen --prefix gen -o "$HERE/tmax8" -n 8  -s 2 --t-max 8 --max-k 2048 --max-n 96
v2_replay "$HERE"
v2_replay_kernel "$HERE"
