#!/usr/bin/env bash
# traces/m2/regen.sh [BUILD_DIR] — the M2 regression traces as `ops` traces (docs/design/
# 06-software-verification.md §5.6): tsr-trace-gen --ops m2 (QACT groups, LD -> ST copies; no MVM),
# with fixed seeds and configurations, then every trace
# replayed under the fixed and two random schedules (race detector fatal),
# tile-only and through the replay kernel (one launch per trace: SPM_STALE is fatal, SW §4.6).
#
#   tmax1/    60 traces, U200 config (4 ch, T_MAX 1, 4 MiB SPM, 16 GiB/ch, CAPS 0), ~15% error injections
#   tmax8/    16 traces, U200 config with T_MAX 8, ~15% error injections
#   small1ch/ 16 traces, 1 channel, 64 KiB SPM, 1 GiB channel with ADDR_BEAT_W 22 (DRAM limit
#             256 MiB < channel size), K_MAX 4096, T_MAX 2, ~20% error injections
#   ch3/       8 traces, 3 channels, T_MAX 3, 256 KiB SPM, ~25% error injections
#   errors/   40 short programs that all have an error injection, U200 config: the M2 fault kinds
#             (reserved fields, refused opcodes, channel >= NUM_CH, QACT kind, K_RANGE, T_RANGE,
#             SPM_RANGE on each operand with 64-bit extents, DRAM_RANGE) and the dependency endings (BAD_DEP,
#             DEP_FAILED chains, ABORTED after an error state, QABORT recipe, QCLEAR recovery)
#   errors8/  32 like errors/, T_MAX 8 (multi-token range checks, (T-1) x src_stride beyond 32 bits)
# Every program is a push stream with deps (no LI / ADDI / END, absolute immediates, x16 = 0). Each set uses one configuration (meta.json "config").
# Every SPM line an op reads is written earlier in its trace (2026-10-30: a strided QACT source loaded per
# token also gets its stride gaps loaded; 13 traces changed, none read the power-on SPM fill).
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
. "$ROOT/traces/v2_tools.sh"
v2_tools "${1:-$ROOT/build/dev}"
GEN=$V2_GEN
rm -rf "$HERE"/tmax1 "$HERE"/tmax8 "$HERE"/small1ch "$HERE"/ch3 "$HERE"/errors "$HERE"/errors8
"$GEN" -q --ops m2 --prefix m2 -o "$HERE/tmax1"    -n 60 -s 21 --errors 15 --max-k 2048
"$GEN" -q --ops m2 --prefix m2 -o "$HERE/tmax8"    -n 16 -s 22 --errors 15 --max-k 2048 --t-max 8
"$GEN" -q --ops m2 --prefix m2 -o "$HERE/small1ch" -n 16 -s 27 --errors 20 --num-ch 1 --spm-kib 64 \
       --ch-gib 1 --addr-beat-w 22 --k-max 4096 --t-max 2
"$GEN" -q --ops m2 --prefix m2 -o "$HERE/ch3"      -n 8  -s 24 --errors 25 --num-ch 3 --t-max 3 \
       --spm-kib 256 --max-k 1024
"$GEN" -q --ops m2 --prefix m2 -o "$HERE/errors"   -n 40 -s 25 --errors 100 --max-groups 1 --max-k 512
"$GEN" -q --ops m2 --prefix m2 -o "$HERE/errors8"  -n 32 -s 35 --errors 100 --max-groups 1 --max-k 512 --t-max 8
v2_replay "$HERE"
v2_replay_kernel "$HERE"
