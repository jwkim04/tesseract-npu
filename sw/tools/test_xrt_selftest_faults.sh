#!/usr/bin/env bash
# test_xrt_selftest_faults.sh SELFTEST REPO — ctest driver: tsr_xrt_selftest on the mock kdev must
# fail (exit 1) whenever the platform adapter flagged an error, also when the core's results are
# right (the protocol layer reports ADAPTER_STATUS only through the exec return code and clears
# the register after every run; docs/xrt-host.md §5):
#   1. a trace that ends in DONE, with a BRESP error injected into its run (TSR_MOCK_INJECT)
#   2. a trace that ends in DONE | ERR (DRAM_RANGE: exec returns -1 anyway), with an RRESP error
#   3. ADAPTER_STATUS left by an earlier process (the open clears it): the identity check fails
#   4. controls: the same runs without injection pass (exit 0), with both start sequences
#      (TSR_MOCK_CTRL=regs and kds, the xrt::run path)
set -u
ST=$1
REPO=$2
OKT=$REPO/traces/m2/tmax1/m2_s21_0000      # ends in DONE
ERRT=$REPO/traces/m2/errors/m2_s25_0000    # ends in DONE | ERR (DRAM_RANGE)
fail() { echo "FAIL: $*"; exit 1; }
for t in "$OKT" "$ERRT"; do [ -f "$t/meta.json" ] || fail "trace $t missing"; done
export TSR_SIM_THREADS=2

run() {   # run EXPECTED_EXIT NAME [env...] -- selftest args...
    local want=$1 name=$2; shift 2
    local envs=()
    while [ "$1" != "--" ]; do envs+=("$1"); shift; done
    shift
    out=$(env "${envs[@]}" "$ST" --device mock "$@" 2>&1)
    local rc=$?
    if [ $rc -ne "$want" ]; then
        echo "$out" | tail -20
        fail "$name: exit $rc, expected $want"
    fi
    echo "ok   $name (exit $rc)"
}

for ctrl in regs kds; do
    run 0 "$ctrl: clean trace" TSR_MOCK_CTRL=$ctrl -- --no-mem --no-programs --traces "$OKT"
    run 0 "$ctrl: clean error-ending trace" TSR_MOCK_CTRL=$ctrl -- --no-mem --no-programs --traces "$ERRT"
    run 1 "$ctrl: BRESP error in a DONE trace" TSR_MOCK_CTRL=$ctrl TSR_MOCK_INJECT=1:0x00010002 -- \
        --no-mem --no-programs --traces "$OKT"
    echo "$out" | grep -q "FAIL  .*m2_s21_0000: .*ADAPTER_STATUS 0x00010002 during the run" || fail "$ctrl: trace failure not reported"
    echo "$out" | grep -q "FAIL  ADAPTER_STATUS clean in every run and at the end" || fail "$ctrl: final check passed"
    run 1 "$ctrl: RRESP error in an error-ending trace" TSR_MOCK_CTRL=$ctrl TSR_MOCK_INJECT=1:0x00030001 -- \
        --no-mem --no-programs --traces "$ERRT"
    echo "$out" | grep -q "FAIL  .*m2_s25_0000: .*ADAPTER_STATUS 0x00030001 during the run" || fail "$ctrl: error-ending trace passed"
    run 1 "$ctrl: ADAPTER_STATUS left by an earlier process" TSR_MOCK_CTRL=$ctrl TSR_MOCK_INJECT=0:0x00020004 -- \
        --no-mem --no-programs --traces "$OKT"
    echo "$out" | grep -q "FAIL  ADAPTER_STATUS clean at open" || fail "$ctrl: leftover not reported"
    echo "$out" | grep -q "PASS\|traces: 1 passed" || fail "$ctrl: the trace itself should pass"
done
# a program run (not a trace) with an adapter error: the END program is the first program run
run 1 "program with a BRESP error" TSR_MOCK_INJECT=1:0x00010002 -- --no-mem
echo "$out" | grep -q "FAIL  END: .*ADAPTER_STATUS 0x00010002" || fail "program failure not reported"
echo "test_xrt_selftest_faults: all passed"
