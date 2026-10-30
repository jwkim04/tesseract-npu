#!/usr/bin/env bash
# test_selftest_faults.sh SELFTEST IMAGE OUT BASIS CAP TSR_ELF — ctest tools_selftest_mock_faults (docs/design/
# 06-software-verification.md §10.2; was tools_xrt_selftest_mock_faults): tsr-selftest on the mock kdev
# must fail (exit 1, the failing check named) whenever the protocol layer reports a fault, also when
# every result it could compare is right, and must pass without one. Faults (sw/transport/
# tsr_kdev_mock.cpp): TSR_MOCK_INJECT N:V (ADAPTER_STATUS value V at the N-th LQ_PUSH: an AXI error
# during that launch, PLAT_ERR, EX §9.3; N = 0: left by an earlier process), TSR_MOCK_TCM_FAULT
# (a TCM write fault: VERIFY mismatch, EX §3.3), TSR_MOCK_HANG (a hung MVM unit: the recovery ladder,
# DB §7.3, "card reset needed", EX §8.6).
# Launch numbering with "--rate-n 2": 1 PING + 2 sequential + 2 pipelined PINGs, then section 5's PING
# (push 6) and the first id given to --ids (push 7).
# BASIS and CAP are the Tesseract Basis version (tsr/version.h) and the Tesseract Core capability
# (tsr_tables.h VERSION) of the build, "X.Y.Z" and "X.Y"; the kinds the boot lists are read from the
# image with TSR_ELF (tsr-elf kinds), so a release or a kind revision bump does not edit this file.
set -u
ST=$1
IMG=$2
OUT=$3
BASIS=$4
CAP=$5
ELFTOOL=$6
mkdir -p "$OUT"
fail() { echo "FAIL: $*"; exit 1; }
[ -f "$IMG" ] || fail "no firmware image $IMG"
COMMON=(--device mock --image "$IMG" --no-mem --rate-n 2)

run() {   # run EXPECTED_EXIT NAME [env...] -- selftest args...
    local want=$1 name=$2; shift 2
    local envs=()
    while [ "$1" != "--" ]; do envs+=("$1"); shift; done
    shift
    out=$(env "${envs[@]}" "$ST" "${COMMON[@]}" "$@" 2>&1)
    local rc=$?
    printf '%s\n' "$out" > "$OUT/$(echo "$name" | tr -c 'a-zA-Z0-9\n' '_').log"
    if [ $rc -ne "$want" ]; then
        printf '%s\n' "$out" | tail -25
        fail "$name: exit $rc, expected $want"
    fi
    echo "ok   $name (exit $rc)"
}
expect() {   # expect PATTERN WHAT: the last run's output must contain PATTERN
    printf '%s\n' "$out" | grep -q -- "$1" || { printf '%s\n' "$out" | tail -25; fail "$2"; }
}

run 0 "control: no fault" -- --ids 1,7
expect "PASS$" "the control run did not pass"
# The versions (SW §6.9): the device's Tesseract Core capability, the image's Tesseract Basis version.
dots() { printf '%s' "${1//./\\.}"; }
expect "Tesseract Core capability $(dots "$CAP") (VERSION 0x" "the identity block does not name the capability"
expect "Tesseract Basis $(dots "$BASIS"), Basis ABI [0-9][0-9]*, needs capability $(dots "$CAP"); host Tesseract Basis $(dots "$BASIS")" \
       "the boot does not name the versions"
kinds_out=$("$ELFTOOL" kinds "$IMG") || fail "tsr-elf kinds $IMG"
nk=$(printf '%s\n' "$kinds_out" | grep -c '^0x')
sig=$(printf '%s\n' "$kinds_out" | sed -n 's/^signature=//p')
first3=$(printf '%s\n' "$kinds_out" | grep '^0x' | head -3 | awk -F'\t' '{ sub("rev=", "", $3); printf "%s%s r%s", (NR > 1 ? " " : ""), $2, $3 }')
[ "$nk" -gt 0 ] && [ -n "$sig" ] && [ -n "$first3" ] || fail "no kinds from tsr-elf kinds $IMG"
expect "kinds ($nk, signature $sig): $first3" "the boot does not list the kinds"

run 1 "ADAPTER_STATUS left by an earlier process" TSR_MOCK_INJECT=0:0x00020004 -- --no-kernel
expect "FAIL  ADAPTER_STATUS clean at open" "the leftover was not reported"

# An AXI error during a launch: the transport reports it (STATUS.PLAT_ERR seen while waiting, or a
# DONE with status PLAT_ERR) and counts one event; it clears ADAPTER_STATUS only after every launch
# in flight has answered (EX §8.6), so the launch answers DONE PLAT_ERR (R7) and the selftest fails
# on that answer and on the count.
run 1 "BRESP error during the id 1 launch (PLAT_ERR)" TSR_MOCK_INJECT=7:0x00010002 -- --ids 1,7
expect "FAIL  id 1 .*status PLAT_ERR" "the launch did not answer DONE PLAT_ERR"
expect "FAIL  ADAPTER_STATUS clean at the end, no PLAT_ERR during the run: .*1 PLAT_ERR event" "the PLAT_ERR was not reported"

# A PONG carries no status (EX §4.4): an AXI error during a PING stays in ADAPTER_STATUS and
# STATUS.PLAT_ERR until the final checks see it.
run 1 "RRESP error during a PING" TSR_MOCK_INJECT=2:0x00030001 -- --no-kernel
expect "FAIL  ADAPTER_STATUS clean at the end, no PLAT_ERR during the run: RRESP error" "the AXI error was not reported"
expect "FAIL  STATUS: no timeout, overflow, ring, hang or platform error bit" "STATUS.PLAT_ERR was not reported"

run 1 "TCM write fault (VERIFY)" TSR_MOCK_TCM_FAULT=0x100:0x1 -- --no-kernel
expect "FAIL  boot 1" "the boot did not fail"

run 1 "hung MVM unit during id 3" TSR_MOCK_HANG=7 TSR_LAUNCH_TIMEOUT_MS=200 TSR_ABORT_MS=50 TSR_RESET_MS=50 -- --ids 3
expect "FAIL  id 3" "the hung launch was not reported"
run 1 "shadow inputs mutated (test hook): device = shadow must fail" TSR_SELFTEST_MUTATE_SHADOW=1 -- --ids 3,7
expect "FAIL  id 3 .*every output byte = shadow" "the output comparison with the shadow did not fail"
printf '%s\n' "$out" | grep -q "FAIL  id 3: Q8_0" && fail "the device's reference check failed (only the shadow was mutated)"
echo "test_selftest_faults: all passed"
