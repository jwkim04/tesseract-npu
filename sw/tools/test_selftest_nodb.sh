#!/usr/bin/env bash
# test_selftest_nodb.sh SELFTEST TSR_DBG IMAGE OUT — ctest tools_selftest_sim_nodb: the host tools on the simulator
# configured as the maximum-clock card image tsr_u200_k4, built without the DB block (HAS_DB 0, TSR_SIM_DB=0,
# docs/platform-u200.md §8.10): DBG_ID reads 0, the DBG and DMI windows read 0 and CTL_ID.DM reads 0.
#   - the checklist passes on sim and mock (T_MAX 4, CAPS with the GDN bit as the image reads it);
#   - the debug-only parts fail cleanly with a message that names the missing DB block: tsr-selftest --debug and --trw,
#     tsr-dbg tile / trw / hist / rv sample / rv halt (the attach a GDB session makes);
#   - what does not need the DB block works: tsr-dbg status / units / perf / sreg.
set -eu
SELFTEST=$1
TSRDBG=$2
IMAGE=$3
OUT=$4
rm -rf "$OUT"
mkdir -p "$OUT"

fail() { echo "test_selftest_nodb: FAIL: $*" >&2; exit 1; }

for kind in sim mock; do
    log="$OUT/$kind.log"
    rc=0
    env TSR_SIM_DB=0 TSR_SIM_TMAX=4 TSR_MOCK_TMAX=4 TSR_SIM_ATT_LOCAL=0 TSR_SIM_MVM_LOCAL_SCALE=0 TSR_SIM_MVM_LOCAL_SOFTCAP=0 TSR_SIM_THREADS=4 "$SELFTEST" --device "$kind" --image "$IMAGE" \
        --mem-mib 8 -v >"$log" 2>&1 || rc=$?
    [ "$rc" -eq 0 ] || { tail -20 "$log" >&2; fail "$kind: tsr-selftest exit $rc"; }
    grep -q " -> PASS" "$log" || fail "$kind: no PASS line"
    grep -q "CTL_ID 0x" "$log" || fail "$kind: no CTL_ID line"
    if grep "CTL_ID 0x" "$log" | grep -q "debug module"; then fail "$kind: CTL_ID reports a debug module"; fi
    echo "$kind: checklist passed, CTL_ID without the debug module"

    # --debug and --trw were asked for and the image cannot do them: a failed check that says why, not a hang or a crash
    for opt in "--debug" "--trw 1"; do
        rc=0
        env TSR_SIM_DB=0 TSR_SIM_TMAX=4 TSR_MOCK_TMAX=4 TSR_SIM_ATT_LOCAL=0 TSR_SIM_MVM_LOCAL_SCALE=0 TSR_SIM_MVM_LOCAL_SOFTCAP=0 TSR_SIM_THREADS=4 "$SELFTEST" --device "$kind" --image "$IMAGE" \
            --no-mem --no-kernel --no-shadow $opt >"$OUT/$kind.opt.log" 2>&1 || rc=$?
        [ "$rc" -eq 1 ] || { tail -20 "$OUT/$kind.opt.log" >&2; fail "$kind: tsr-selftest $opt exit $rc, expected 1"; }
        grep -q "no DB block" "$OUT/$kind.opt.log" || { cat "$OUT/$kind.opt.log" >&2; fail "$kind: tsr-selftest $opt does not say the DB block is missing"; }
        echo "$kind: tsr-selftest $opt fails with the DB block named"
    done
done

# tsr-dbg: what needs the DB block fails (exit 1) and says so; what does not, works.
run_dbg() {
    env TSR_SIM_DB=0 TSR_SIM_TMAX=4 TSR_SIM_THREADS=4 "$TSRDBG" --device sim --elf "$IMAGE" --attach "$@"
}
for cmd in "status" "units" "perf" "sreg"; do
    rc=0
    run_dbg -ex "$cmd" >"$OUT/dbg_$cmd.log" 2>&1 || rc=$?
    [ "$rc" -eq 0 ] || { cat "$OUT/dbg_$cmd.log" >&2; fail "tsr-dbg $cmd exit $rc, expected 0"; }
done
for cmd in "tile show" "trw status" "hist" "rv sample" "rv halt"; do
    rc=0
    run_dbg -ex "$cmd" >"$OUT/dbg_bad.log" 2>&1 || rc=$?
    [ "$rc" -eq 1 ] || { cat "$OUT/dbg_bad.log" >&2; fail "tsr-dbg $cmd exit $rc, expected 1"; }
    grep -qE "no DB block|has none \(DBG_ID = 0" "$OUT/dbg_bad.log" || { cat "$OUT/dbg_bad.log" >&2; fail "tsr-dbg $cmd does not say the DB block is missing"; }
    echo "tsr-dbg $cmd: fails with the DB block named"
done

# a GDB session cannot attach: the server says so and exits 1 before it listens
rc=0
env TSR_SIM_DB=0 TSR_SIM_TMAX=4 TSR_SIM_THREADS=4 "$TSRDBG" gdb --device sim --elf "$IMAGE" --attach --port 0 >"$OUT/dbg_gdb.log" 2>&1 || rc=$?
[ "$rc" -eq 1 ] || { cat "$OUT/dbg_gdb.log" >&2; fail "tsr-dbg gdb exit $rc, expected 1"; }
grep -q "no DB block" "$OUT/dbg_gdb.log" || { cat "$OUT/dbg_gdb.log" >&2; fail "tsr-dbg gdb does not say the DB block is missing"; }
echo "tsr-dbg gdb: fails with the DB block named"

# with the DB model the same commands do not complain (the check is about the image, not the tool)
rc=0
env TSR_SIM_TMAX=4 TSR_SIM_THREADS=4 "$TSRDBG" --device sim --elf "$IMAGE" --attach -ex "hist" >"$OUT/dbg_db.log" 2>&1 || rc=$?
[ "$rc" -eq 0 ] || { cat "$OUT/dbg_db.log" >&2; fail "tsr-dbg hist with the DB model exit $rc, expected 0"; }
echo "tsr-dbg hist with the DB model: ok"
