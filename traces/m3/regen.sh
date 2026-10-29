#!/usr/bin/env bash
# traces/m3/regen.sh [BUILD_DIR [OUT_DIR]] — regenerate the v1 M3 RTL regression traces
# (tsr-trace-gen --ops m3: MVM-centred programs mixed with the M2 QACT / copy groups), replay every
# one on the ISA simulator and print the MVM shape / weight-traffic summary (trace_stats.py).
#
#   tmax1/    64 traces, U200 v1 config (4 ch, T_MAX 1, 4 MiB SPM, 16 GiB/ch), ~10% error endings,
#             traces 0, 25, 50 "big" (one long weight stream, 4x the per-trace weight budget)
#   tmax8/    32 traces, U200 v1 config with T_MAX 8, ~10% error endings, traces 0 and 25 big
#   small1ch/ 24 traces, 1 channel, 64 KiB SPM, 1 GiB channel with ADDR_BEAT_W 22 (DRAM limit
#             256 MiB < channel size: the weight heap sits in the upper half of [64 MiB, limit)),
#             K_MAX 4096, T_MAX 2, ~15% error endings
#   ch3/      16 traces, 3 channels, T_MAX 3, 256 KiB SPM, ~15% error endings
#   ch16/     12 traces, 16 channels, T_MAX 2, 256 KiB SPM, 1 GiB/ch, K_MAX 8192, ~10% error endings
#   errors/   68 short programs (one group each) that all end in an error, U200 v1 config: two cycles
#             through every MVM fault (reserved fields, BAD_FMT fmt 2..15, K_RANGE incl. the 16-bit
#             K field, T_RANGE, SPM_RANGE on A1 / A2, DRAM_RANGE on A3, with lower-priority
#             violations and 32-bit-wrapping extents) and every M2 error ending
#   errors8/  68 like errors/, T_MAX 8 (multi-token extents, (T-1) x dst_stride >= 2^32)
# Each set uses one configuration (meta.json "config"): build the RTL core with
# trace.Trace.core_parameters() of any trace in the set. Weight traffic: --max-wbeats 16384 MVM
# weight beats per trace on the busiest channel (channel 0; 4096 in the error sets), see the
# "MVM weight beats" line printed per set and trace_stats.py for per-set totals.
# X0 (docs/v2 EX §12, SW §5.6): the v1 sets are deleted from disk; this script is kept as the record
# of their seeds and configurations, which Y2 reuses to regenerate traces/m3 as v2 `ops` traces
# with tsr-trace-gen v2 (P1.10). It still reproduces the v1 sets with the v1 tools: BUILD_DIR
# (default build/v1, a binary copy of tag v1-m4) is checked by traces/v1_tools.sh before anything
# is deleted. The v1 sets go to OUT_DIR (default build/v1-traces/m3), not into traces/m3, which
# holds the v2 sets from Y2 on.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
. "$ROOT/traces/v1_tools.sh"
v1_tools "${1:-$ROOT/build/v1}"
OUT=${2:-$ROOT/build/v1-traces/m3}
mkdir -p "$OUT"
GEN=$V1_GEN
REPLAY=$V1_REPLAY
rm -rf "$OUT"/tmax1 "$OUT"/tmax8 "$OUT"/small1ch "$OUT"/ch3 "$OUT"/ch16 "$OUT"/errors "$OUT"/errors8
"$GEN" -q --ops m3 --prefix m3 -o "$OUT/tmax1"    -n 64 -s 31 --errors 10 --big-pct 4
"$GEN" -q --ops m3 --prefix m3 -o "$OUT/tmax8"    -n 32 -s 32 --errors 10 --big-pct 4 --t-max 8
"$GEN" -q --ops m3 --prefix m3 -o "$OUT/small1ch" -n 24 -s 33 --errors 15 --num-ch 1 --spm-kib 64 \
       --ch-gib 1 --addr-beat-w 22 --k-max 4096 --t-max 2
"$GEN" -q --ops m3 --prefix m3 -o "$OUT/ch3"      -n 16 -s 34 --errors 15 --num-ch 3 --t-max 3 --spm-kib 256
"$GEN" -q --ops m3 --prefix m3 -o "$OUT/ch16"     -n 12 -s 36 --errors 10 --num-ch 16 --t-max 2 --spm-kib 256 \
       --ch-gib 1 --k-max 8192
"$GEN" -q --ops m3 --prefix m3 -o "$OUT/errors"   -n 68 -s 35 --errors 100 --max-groups 1 --max-k 1024 --max-wbeats 4096
"$GEN" -q --ops m3 --prefix m3 -o "$OUT/errors8"  -n 68 -s 37 --errors 100 --max-groups 1 --max-k 1024 --max-wbeats 4096 \
       --t-max 8
"$REPLAY" -q "$OUT"
python3 "$HERE/trace_stats.py" "$OUT"
