#!/usr/bin/env bash
# test_trace_gen.sh GEN OUT_DIR [REPLAY [IMAGE]] — ctest tools_trace_gen_replay (docs/design/06-software-
# verification.md §10.2): tsr-trace-gen per --ops mode with a fixed seed.
#   1. a small set per mode (gen, m2, m3, vu, at, ctrl; errors on), each trace verified by the
#      generator (prediction of every completion, serial check, SPM_STALE check, host references)
#      and replayed under two random schedules (--check-sched 2);
#   2. the same options again without verification: the sets must be byte-identical (determinism:
#      the checks change nothing, no host-time dependence);
#   3. with REPLAY (tsr-trace-replay): every trace replayed under the fixed and two random schedules
#      tile-only (race detector fatal unless TSR_SIM_RACE says otherwise), then a copy with one
#      corrupted expect byte must fail;
#   4. with IMAGE (a firmware image with the `replay` kernel, SW §7.7; given by the build when it has
#      the race detector): every trace replayed through the kernel under the same three schedules
#      with TSR_SIM_RACE=fatal — a trace is one launch there, so a read of SPM that nothing wrote in
#      the trace is an SPM_STALE abort (SW §4.6, SW-4);
#   5. the SPM_STALE mutant (--mutant-stale: per-token loads leave the stride gaps of a strided QACT
#      source unwritten, as an earlier generator did): the generator's verification must refuse the
#      trace, and with IMAGE the race detector must report SPM_STALE on its kernel-route replay
#      (report mode: under fatal the same report aborts the process).
# Exit 0 when everything passed.
set -eu
GEN=$1
OUT=$2
REPLAY=${3:-}
IMAGE=${4:-}
rm -rf "$OUT"
mkdir -p "$OUT/a" "$OUT/b"
export TSR_SIM_RACE=${TSR_SIM_RACE:-fatal}
unset TSR_SIM_SCHED TSR_TRACE

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

if [ -n "$REPLAY" ] && [ -n "$IMAGE" ]; then
    if [ ! -f "$IMAGE" ]; then
        echo "test_trace_gen: firmware image $IMAGE not built (cmake --build <dir> --target tsr_fw)" >&2
        exit 1
    fi
    TSR_SIM_RACE=fatal "$REPLAY" -q --sched fixed --sched random:1 --sched random:2 --replay-image "$IMAGE" "$OUT/a"
    echo "test_trace_gen: kernel route (replay kernel, race detector fatal): no SPM_STALE"
fi

# 5. SPM_STALE mutant: trace 2 of this m2 configuration has a strided QACT source loaded per token.
mut="--ops m2 -n 3 -s 22 --t-max 8 --max-k 2048 --errors 15 --mutant-stale"
if "$GEN" -q $mut -o "$OUT/stale" > "$OUT/stale.log" 2>&1 || ! grep -q 'SPM_STALE' "$OUT/stale.log"; then
    cat "$OUT/stale.log" >&2
    echo "test_trace_gen: the generator did not refuse the SPM_STALE mutant" >&2
    exit 1
fi
echo "test_trace_gen: SPM_STALE mutant refused by the generator: $(grep -m1 'SPM_STALE: record' "$OUT/stale.log" | sed 's/^tsr-trace-gen: //')"
if [ -n "$REPLAY" ] && [ -n "$IMAGE" ]; then
    "$GEN" -q $mut --no-verify -o "$OUT/stale_nv" > /dev/null
    # report mode (fatal would end in abort()): the replay itself passes, the detector must print the read
    TSR_SIM_RACE=report "$REPLAY" -q --replay-image "$IMAGE" "$OUT/stale_nv/gen_s22_0002" > "$OUT/stale_nv.log" 2>&1 || true
    if ! grep -q '^  SPM_STALE SPM bytes' "$OUT/stale_nv.log"; then
        cat "$OUT/stale_nv.log" >&2
        echo "test_trace_gen: the race detector did not report the SPM_STALE mutant on the kernel route" >&2
        exit 1
    fi
    echo "test_trace_gen: SPM_STALE mutant reported by the race detector on the kernel route:" \
         "$(grep -m1 '^  SPM_STALE' "$OUT/stale_nv.log" | sed 's/^ *//; s/ pushed at.*//')"
fi
echo "test_trace_gen: PASS"
