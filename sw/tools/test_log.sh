#!/usr/bin/env bash
# test_log.sh TSR_LOG SELFTEST IMAGE OUT — ctest tools_log_selftest (docs/design/06-software-verification.md
# §9.5; DB §6.1): the `selftest` kernel's id 9 on the sim kind writes two FMT records; tsr-selftest saves
# every drained log word (--log-out), tsr-log decodes them with the ELF (binary and --hex input give the
# same text), then `tsr-log selftest` checks every record type, the malformed ones and the resync.
set -u
LOG=$1
ST=$2
IMG=$3
OUT=$4
fail() { echo "FAIL: $*"; exit 1; }
rm -rf "$OUT"
mkdir -p "$OUT"
"$ST" --device sim --image "$IMG" --no-mem --ids 9 --rate-n 0 --no-shadow --log-out "$OUT/sim.log" > "$OUT/selftest.out" 2>&1 ||
    { tail -20 "$OUT/selftest.out"; fail "tsr-selftest --ids 9 failed"; }
[ -s "$OUT/sim.log" ] || fail "no log words saved"
"$LOG" --elf "$IMG" "$OUT/sim.log" > "$OUT/decoded.txt" 2> "$OUT/decoded.err" || { cat "$OUT/decoded.txt" "$OUT/decoded.err"; fail "tsr-log reported malformed records"; }
grep -q "\[tsr INFO kernel seq [0-9]* t [0-9]*\] selftest 9: seed 9d9c0de, id 9$" "$OUT/decoded.txt" || { cat "$OUT/decoded.txt"; fail "INFO record of id 9 not decoded"; }
grep -q "\[tsr WARN kernel seq [0-9]* t [0-9]*\] selftest 9: 4129701665 9d9c0df$" "$OUT/decoded.txt" || { cat "$OUT/decoded.txt"; fail "WARN record of id 9 not decoded"; }
od -An -tx4 -v "$OUT/sim.log" > "$OUT/sim.hex"
"$LOG" --hex --elf "$IMG" "$OUT/sim.hex" > "$OUT/decoded_hex.txt" 2>/dev/null || fail "tsr-log --hex failed"
cmp -s "$OUT/decoded.txt" "$OUT/decoded_hex.txt" || fail "binary and --hex input decode differently"
echo "test_log: $(wc -l < "$OUT/decoded.txt") records decoded from $(($(stat -c %s "$OUT/sim.log") / 4)) words"
"$LOG" selftest "$IMG" || fail "tsr-log selftest"
echo "test_log: PASS"
