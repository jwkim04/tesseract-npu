#!/usr/bin/env bash
# test_trace_replay_device.sh GEN REPLAY IMAGE OUT [SELFTEST] — ctest tools_trace_replay_device (docs/design/
# 06-software-verification.md §9.5: tsr-trace-replay --device KIND): a small generated set per --ops
# mode (m3, vu, at, ctrl; errors on; T_MAX 8 like the device below) replayed through transports:
#   1. --device sim --shadow: every trace through the `replay` kernel of IMAGE, answer and expect
#      regions against the trace, device against the shadow simulator;
#   2. --device mock: the same through the adapter model (DMA bounce and alignment rules on);
#   3. a copy with one corrupted expect byte must fail on the device route (exit 1, one FAIL line);
#   4. with SELFTEST: kind `launch` traces captured on the sim kind (TSR_TRACE, SW §5.5) while
#      tsr-selftest runs `selftest` ids 1, 2, 3, 7 and 9, replayed as full launches through mock with
#      the shadow (the capture's synthesized image loads the same words as IMAGE, tsr_devrun.h).
set -u
GEN=$1
REPLAY=$2
IMG=$3
OUT=$4
ST=${5:-}
fail() { echo "FAIL: $*"; exit 1; }
rm -rf "$OUT"
mkdir -p "$OUT"
unset TSR_SIM_SCHED TSR_TRACE
for m in "m3 -n 3 -s 35 --errors 50 --max-k 1024 --max-wbeats 2048 --t-max 2" \
         "vu -n 4 -s 43 --errors 50 --t-max 8 --max-groups 2" \
         "at -n 3 -s 53 --errors 50 --t-max 8 --max-groups 2" \
         "ctrl -n 4 -s 62 --errors 100 --t-max 8 --max-groups 2"; do
    set -- $m
    name=$1
    shift
    "$GEN" -q --ops "$name" "$@" -o "$OUT/sets/$name" > "$OUT/gen_$name.log" 2>&1 || { cat "$OUT/gen_$name.log"; fail "tsr-trace-gen --ops $name"; }
done
n=$(find "$OUT/sets" -name meta.json | wc -l)
TSR_SIM_TMAX=8 "$REPLAY" -q --device sim --shadow --replay-image "$IMG" "$OUT/sets" > "$OUT/sim.log" 2>&1 ||
    { tail -20 "$OUT/sim.log"; fail "--device sim --shadow"; }
tail -1 "$OUT/sim.log"
grep -q " $n traces on sim (+ shadow): [1-9][0-9]* passed, 0 failed" "$OUT/sim.log" || fail "sim: not every trace ran"
TSR_MOCK_TMAX=8 TSR_MOCK_DMA_ALIGN=64 "$REPLAY" -q --device mock --replay-image "$IMG" "$OUT/sets" > "$OUT/mock.log" 2>&1 ||
    { tail -20 "$OUT/mock.log"; fail "--device mock"; }
tail -1 "$OUT/mock.log"
t=$(find "$OUT/sets/vu" -name meta.json | head -1 | xargs dirname)
cp -r "$t" "$OUT/bad"
f=$(ls "$OUT"/bad/expect/ch*.bin | head -1)
printf '\xA5' | dd of="$f" bs=1 seek=7 conv=notrunc status=none
if TSR_SIM_TMAX=8 "$REPLAY" -q --device sim --replay-image "$IMG" "$OUT/bad" > "$OUT/bad.log" 2>&1; then
    cat "$OUT/bad.log"
    fail "a corrupted expect byte was not detected"
fi
[ "$(grep -c '^FAIL' "$OUT/bad.log")" = 1 ] || { cat "$OUT/bad.log"; fail "expected exactly one FAIL line"; }
grep -q "expect ch" "$OUT/bad.log" || { cat "$OUT/bad.log"; fail "the failure does not name the expect region"; }
echo "test_trace_replay_device: corrupted copy rejected: $(grep '^FAIL' "$OUT/bad.log" | sed 's/.*: device: //' | cut -c1-80)"
if [ -n "$ST" ]; then
    TSR_TRACE="$OUT/cap" "$ST" --device sim --image "$IMG" --no-mem --rate-n 0 --no-shadow --ids 1,2,3,7,9 > "$OUT/cap.log" 2>&1 ||
        { tail -20 "$OUT/cap.log"; fail "tsr-selftest with TSR_TRACE"; }
    nl=$(find "$OUT/cap" -name meta.json | wc -l)
    [ "$nl" -ge 6 ] || fail "only $nl launch traces captured"
    "$REPLAY" -q --device mock --shadow --replay-image "$IMG" "$OUT/cap" > "$OUT/launch.log" 2>&1 ||
        { tail -20 "$OUT/launch.log"; fail "kind launch traces through mock"; }
    tail -1 "$OUT/launch.log"
    grep -q " $nl traces on mock (+ shadow): $nl passed, 0 failed" "$OUT/launch.log" || fail "launch: not every trace passed"
fi
echo "test_trace_replay_device: PASS"
