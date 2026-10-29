#!/usr/bin/env bash
# traces/m2/regen.sh [BUILD_DIR [OUT_DIR]] — regenerate the v1 M2 RTL regression traces
# (tsr-trace-gen --ops m2: NOP END LI ADDI LD ST QACT FENCE only) and replay every one on the ISA
# simulator.
#
#   tmax1/    60 traces, U200 v1 config (4 ch, T_MAX 1, 4 MiB SPM, 16 GiB/ch), ~15% error endings
#   tmax8/    16 traces, U200 v1 config with T_MAX 8, ~15% error endings
#   small1ch/ 16 traces, 1 channel, 64 KiB SPM, 1 GiB channel with ADDR_BEAT_W 22 (DRAM limit
#             256 MiB < channel size), K_MAX 4096, T_MAX 2, ~20% error endings
#   ch3/       8 traces, 3 channels, T_MAX 3, 256 KiB SPM, ~25% error endings
#   errors/   40 short programs that all end in an error (every M2-reachable error ending,
#             incl. fetch beyond the DRAM limit and CMD_ADDR_HI != 0), U200 v1 config
#   errors8/  32 short programs that all end in an error, U200 v1 config with T_MAX 8 (multi-token
#             range checks, incl. (T-1)*src_stride beyond 32 bits)
# Each set uses one configuration (meta.json "config"): build the RTL core with
# trace.Trace.core_parameters() of any trace in the set.
# X0 (docs/v2 EX §12, SW §5.6): the v1 sets are deleted from git; this script is kept as the record
# of their seeds and configurations, which Y2 reuses to regenerate traces/m2 as v2 `ops` traces
# with tsr-trace-gen v2 (P1.10). It still reproduces the v1 sets with the v1 tools: BUILD_DIR
# (default build/v1, a binary copy of tag v1-m4) is checked by traces/v1_tools.sh before anything
# is deleted. The v1 sets go to OUT_DIR (default build/v1-traces/m2, outside git), never back into
# traces/m2, which holds the v2 sets from Y2 on.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
. "$ROOT/traces/v1_tools.sh"
v1_tools "${1:-$ROOT/build/v1}"
OUT=${2:-$ROOT/build/v1-traces/m2}
mkdir -p "$OUT"
GEN=$V1_GEN
REPLAY=$V1_REPLAY
rm -rf "$OUT"/tmax1 "$OUT"/tmax8 "$OUT"/small1ch "$OUT"/ch3 "$OUT"/errors "$OUT"/errors8
"$GEN" -q --ops m2 --prefix m2 -o "$OUT/tmax1"    -n 60 -s 21 --errors 15 --max-k 2048
"$GEN" -q --ops m2 --prefix m2 -o "$OUT/tmax8"    -n 16 -s 22 --errors 15 --max-k 2048 --t-max 8
"$GEN" -q --ops m2 --prefix m2 -o "$OUT/small1ch" -n 16 -s 27 --errors 20 --num-ch 1 --spm-kib 64 \
       --ch-gib 1 --addr-beat-w 22 --k-max 4096 --t-max 2
"$GEN" -q --ops m2 --prefix m2 -o "$OUT/ch3"      -n 8  -s 24 --errors 25 --num-ch 3 --t-max 3 \
       --spm-kib 256 --max-k 1024
"$GEN" -q --ops m2 --prefix m2 -o "$OUT/errors"   -n 40 -s 25 --errors 100 --max-groups 1 --max-k 512
"$GEN" -q --ops m2 --prefix m2 -o "$OUT/errors8"  -n 32 -s 35 --errors 100 --max-groups 1 --max-k 512 --t-max 8
"$REPLAY" -q "$OUT"
