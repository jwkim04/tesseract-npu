#!/usr/bin/env bash
# test_selftest_k4min.sh SELFTEST IMAGE OUT — ctest tools_selftest_sim_k4min: the checklist on the simulator
# configured as the reduced card image tsr_u200_k4 (TSR_SIM_VU=0 TSR_SIM_TMAX=8, docs/platform-u200.md §8.7):
# the core is built without the vector unit and attention, so CAPS reads 0, Q_DEPTH has no VU queue and T_MAX is 8.
# The identity registers must say so, and the checklist must pass on the sim and on the mock kind.
set -eu
SELFTEST=$1
IMAGE=$2
OUT=$3
rm -rf "$OUT"
mkdir -p "$OUT"

fail() { echo "test_selftest_k4min: FAIL: $*" >&2; exit 1; }

for kind in sim mock; do
    log="$OUT/$kind.log"
    rc=0
    env TSR_SIM_VU=0 TSR_SIM_TMAX=8 TSR_MOCK_TMAX=8 TSR_SIM_THREADS=4 "$SELFTEST" --device "$kind" --image "$IMAGE" \
        --mem-mib 8 -v >"$log" 2>&1 || rc=$?
    [ "$rc" -eq 0 ] || { tail -20 "$log" >&2; fail "$kind: tsr-selftest exit $rc"; }
    grep -q "CAPS 0x00000000" "$log" || fail "$kind: CAPS is not 0"
    grep -q "T_MAX 8" "$log" || fail "$kind: T_MAX is not 8"
    grep -q "VU 0, ATT 0" "$log" || fail "$kind: Q_DEPTH has a VU queue"
    grep -q " -> PASS" "$log" || fail "$kind: no PASS line"
    echo "$kind: CAPS 0, T_MAX 8, no VU queue, checklist passed"
done
