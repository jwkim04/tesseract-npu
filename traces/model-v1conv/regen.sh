#!/usr/bin/env bash
# traces/model-v1conv/regen.sh [BUILD_DIR] — the v1 real-model traces (traces/model) converted to trace
# v2 (docs/v2/06-software-verification.md §5.6, V2-SW-6): the regression check that the tile units'
# results did not change from v1 to v2.
#
# tsr-trace-conv (BUILD_DIR/bin, default build/dev) turns each v1 trace of traces/model (2 models x 145
# splits, cmd.bin programs with R0 only and no errors) into a kind-ops trace: each v1 instruction becomes
# a push that depends on the previous one (a serial chain: v1's sequential semantics), END is dropped,
# init and expect are kept byte for byte. tsr-trace-replay then replays every converted trace on the
# simulator under the fixed schedule and one random schedule — tile-only through the issue port until
# the `replay` kernel exists (SW §7.7; P2.5 adds the kernel route with flag SERIAL) — and every one must
# match v1's expect bit for bit. Each model's set is written to <set>.new and replaces the old set only
# after all its replays passed. traces/model itself is never modified (it is regenerated only by the v1
# tools, traces/model/regen.sh).
#
#   qwen05b-q8_0/              145 traces    (sizes and times: see the last run's summary lines)
#   qwen05b-q4_k_m-requant/    145 traces
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
B=$(cd "${1:-$ROOT/build/dev}" 2>/dev/null && pwd) || { echo "$0: no build directory ${1:-$ROOT/build/dev}" >&2; exit 2; }
CONV=$B/bin/tsr-trace-conv
REPLAY=$B/bin/tsr-trace-replay
for f in "$CONV" "$REPLAY"; do
    [ -x "$f" ] || { echo "$0: $f not built (cmake --build $B --target tsr-trace-conv tsr-trace-replay)" >&2; exit 2; }
done
export TSR_SIM_THREADS=${TSR_SIM_THREADS:-4}
unset TSR_TRACE TSR_SIM_SCHED
for m in qwen05b-q8_0 qwen05b-q4_k_m-requant; do
    src=$ROOT/traces/model/$m
    [ -d "$src" ] || { echo "$0: $src not on disk (traces/model/regen.sh with the v1 tools)" >&2; exit 2; }
    d=$HERE/$m
    n=$d.new
    rm -rf "$n"
    "$CONV" -q -o "$n" "$src" || { echo "conversion of $src failed ($d is unchanged)"; rm -rf "$n"; exit 1; }
    "$REPLAY" -q --sched fixed --sched random:1 "$n" || { echo "replay of $n failed ($d is unchanged)"; exit 1; }
    rm -rf "$d"
    mv "$n" "$d"
    echo "$m: $(find "$d" -name meta.json | wc -l) traces, $(du -sh "$d" | cut -f1)"
done
