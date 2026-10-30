#!/usr/bin/env bash
# test_trace_gen.sh GEN OUT_DIR [REPLAY] — ctest tools_trace_gen_replay (docs/v2/06-software-
# verification.md §10.2, P1.10): tsr-trace-gen v2 per --ops mode with a fixed seed.
#   1. a small set per mode (gen, m2, m3, vu, at, ctrl; errors on), each trace verified by the
#      generator (prediction of every completion, serial check, host references) and replayed under
#      two random schedules (--check-sched 2);
#   2. the same options again without verification: the sets must be byte-identical (determinism:
#      the checks change nothing, no host-time dependence);
#   3. with REPLAY (tsr-trace-replay): every trace replayed under the fixed and two random schedules
#      (with the race detector when TSR_SIM_RACE is set), then a copy with one corrupted expect byte
#      must fail.
# Exit 0 when everything passed.
set -eu
GEN=$1
OUT=$2
REPLAY=${3:-}
rm -rf "$OUT"
mkdir -p "$OUT/a" "$OUT/b"

gen() {
    local name=$1
    shift
    if ! "$GEN" -q "$@" --check-sched 2 -o "$OUT/a/$name" > "$OUT/$name.log" 2>&1; then
        cat "$OUT/$name.log" >&2
        echo "test_trace_gen: --ops $name: generation failed" >&2
        exit 1
    fi
    "$GEN" -q "$@" --no-verify -o "$OUT/b/$name" > /dev/null
    grep '^tsr-trace-gen: first error' "$OUT/$name.log" | sed "s|^tsr-trace-gen:|  $name:|"
}

gen gen  --ops gen  -n 4 -s 1  --max-k 1024 --max-n 64 --errors 50
gen m2   --ops m2   -n 8 -s 25 --errors 100 --max-groups 1 --max-k 512 --t-max 2
gen m3   --ops m3   -n 4 -s 35 --errors 50 --max-k 1024 --max-wbeats 2048 --t-max 2
gen vu   --ops vu   -n 6 -s 43 --errors 50 --t-max 8 --max-groups 2
gen at   --ops at   -n 4 -s 53 --errors 50 --t-max 8 --max-groups 2
gen ctrl --ops ctrl -n 6 -s 62 --errors 100 --t-max 8 --max-groups 2

if ! diff -r -q "$OUT/a" "$OUT/b" > "$OUT/diff.log"; then
    cat "$OUT/diff.log" >&2
    echo "test_trace_gen: the same options gave different traces" >&2
    exit 1
fi
n=$(find "$OUT/a" -name meta.json | wc -l)
echo "test_trace_gen: $n traces generated twice, byte-identical"

if [ -n "$REPLAY" ]; then
    "$REPLAY" -q --sched fixed --sched random:1 --sched random:2 "$OUT/a"
    t=$(find "$OUT/a/vu" -name meta.json | head -1 | xargs dirname)
    cp -r "$t" "$OUT/bad"
    f=$(ls "$OUT"/bad/expect/ch*.bin | head -1)
    printf '\xA5' | dd of="$f" bs=1 seek=7 conv=notrunc status=none
    if "$REPLAY" -q "$OUT/bad" > "$OUT/bad.log" 2>&1; then
        echo "test_trace_gen: a corrupted expect byte was not detected" >&2
        exit 1
    fi
    echo "test_trace_gen: corrupted copy rejected"
fi
echo "test_trace_gen: PASS"
