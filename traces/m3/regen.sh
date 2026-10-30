#!/usr/bin/env bash
# traces/m3/regen.sh [BUILD_DIR] — the M3 regression traces as trace v2 `ops` traces (docs/v2/
# 06-software-verification.md §5.6): tsr-trace-gen v2 --ops m3 (MVM-centred groups mixed with the M2
# QACT / copy groups), with v1's seeds and configurations (tag v1-m4, traces/m3/regen.sh there), then
# every trace replayed under the fixed and two random schedules (race detector fatal),
# tile-only and through the replay kernel (one launch per trace: SPM_STALE is fatal, SW §4.6). The
# generator prints the MVM count and weight traffic per set (v1's trace_stats.py reads v1 traces only;
# it stays for traces/model).
#
#   tmax1/    64 traces, U200 config (4 ch, T_MAX 1, 4 MiB SPM, 16 GiB/ch, CAPS 0), ~10% error
#             injections, traces 0, 25, 50 "big" (one long weight stream, 4x the per-trace weight budget)
#   tmax8/    32 traces, U200 config with T_MAX 8, ~10% error injections, traces 0 and 25 big
#   small1ch/ 24 traces, 1 channel, 64 KiB SPM, 1 GiB channel with ADDR_BEAT_W 22 (DRAM limit
#             256 MiB < channel size: the weight heap sits in the upper half of [64 MiB, limit)),
#             K_MAX 4096, T_MAX 2, ~15% error injections
#   ch3/      16 traces, 3 channels, T_MAX 3, 256 KiB SPM, ~15% error injections
#   ch16/     12 traces, 16 channels, T_MAX 2, 256 KiB SPM, 1 GiB/ch, K_MAX 8192, ~10% error injections
#   errors/   68 short programs (one group each) with an error injection, U200 config: two cycles
#             through the 23 MVM fault kinds of v1 (reserved fields, BAD_FMT fmt 2..15, K_RANGE incl.
#             the 16-bit K field, T_RANGE, SPM_RANGE on A1 / A2, DRAM_RANGE on A3, with lower-priority
#             violations and 32-bit-wrapping extents) and the 12 M2 / v2 kinds
#   errors8/  68 like errors/, T_MAX 8 (multi-token extents, (T-1) x dst_stride >= 2^32)
# Not in git (.gitignore): regenerate with this script. Weight traffic: --max-wbeats 16384 MVM weight
# beats per trace on the busiest channel (channel 0; 4096 in the error sets). Every SPM line an op
# reads is written earlier in its trace (2026-10-30: a strided QACT source loaded per token also gets its
# stride gaps loaded; 23 traces changed, none read the power-on SPM fill).
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
. "$ROOT/traces/v2_tools.sh"
v2_tools "${1:-$ROOT/build/dev}"
GEN=$V2_GEN
rm -rf "$HERE"/tmax1 "$HERE"/tmax8 "$HERE"/small1ch "$HERE"/ch3 "$HERE"/ch16 "$HERE"/errors "$HERE"/errors8
for s in "tmax1    -n 64 -s 31 --errors 10 --big-pct 4" \
         "tmax8    -n 32 -s 32 --errors 10 --big-pct 4 --t-max 8" \
         "small1ch -n 24 -s 33 --errors 15 --num-ch 1 --spm-kib 64 --ch-gib 1 --addr-beat-w 22 --k-max 4096 --t-max 2" \
         "ch3      -n 16 -s 34 --errors 15 --num-ch 3 --t-max 3 --spm-kib 256" \
         "ch16     -n 12 -s 36 --errors 10 --num-ch 16 --t-max 2 --spm-kib 256 --ch-gib 1 --k-max 8192" \
         "errors   -n 68 -s 35 --errors 100 --max-groups 1 --max-k 1024 --max-wbeats 4096" \
         "errors8  -n 68 -s 37 --errors 100 --max-groups 1 --max-k 1024 --max-wbeats 4096 --t-max 8"; do
    set -- $s
    name=$1
    shift
    "$GEN" -q --ops m3 --prefix m3 -o "$HERE/$name" "$@" > "$HERE/$name.log"
    sed -n "s|^tsr-trace-gen: \([0-9]* MVMs\)|$name: \1|p" "$HERE/$name.log"
    rm -f "$HERE/$name.log"
done
v2_replay "$HERE"
v2_replay_kernel "$HERE"
