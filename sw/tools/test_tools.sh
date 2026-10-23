#!/usr/bin/env bash
# test_tools.sh GEN REPLAY DIR — ctest driver for tsr-trace-gen and tsr-trace-replay.
#   1. generate 31 M1 traces over several configs (T_MAX 1/3/8, 1/3/4/16 channels, small SPM,
#      error endings, with and without special values), 14 M2 traces (--ops m2: no MVM;
#      every error ending, narrow ADDR_BEAT_W) and 30 M3 traces (--ops m3: MVM-centred, T_MAX 4,
#      1 channel with ADDR_BEAT_W 22, every MVM error ending); --num-ch 17, --ops m9 and
#      --max-wbeats 0 are rejected; every M3 trace executes an MVM
#   2. generation is deterministic (same seed -> identical files), M1, M2 and M3
#   3. tsr-trace-replay passes all of them (exit code 0)
#   4. one corrupted expect byte -> exit code 1; a path without traces -> exit code 1
set -u
GEN=$1
REPLAY=$2
DIR=$3
fail() { echo "FAIL: $*"; exit 1; }

rm -rf "$DIR"
mkdir -p "$DIR"
"$GEN" -q -o "$DIR/a" -n 12 -s 7                                            || fail "gen a"
"$GEN" -q -o "$DIR/b" -n 8 -s 8 --t-max 8 --errors 30                       || fail "gen b"
"$GEN" -q -o "$DIR/c" -n 4 -s 9 --num-ch 1 --t-max 3 --spm-kib 256 --ch-gib 1 --max-k 1024 || fail "gen c"
"$GEN" -q -o "$DIR/d" -n 4 -s 10 --num-ch 3 --t-max 2 --errors 50 --no-special || fail "gen d"
"$GEN" -q -o "$DIR/e" -n 3 -s 11 --num-ch 16 --ch-gib 1 --max-n 96 --max-k 512 --errors 30 || fail "gen e"
"$GEN" -q --ops m2 -o "$DIR/m" -n 8 -s 12 --t-max 4 --errors 60                  || fail "gen m"
"$GEN" -q --ops m2 -o "$DIR/mc1" -n 6 -s 13 --num-ch 1 --spm-kib 64 --ch-gib 1 --addr-beat-w 22 \
       --k-max 1024 --t-max 2 --errors 100                                      || fail "gen mc1"
"$GEN" -q --ops m3 -o "$DIR/v" -n 8 -s 14 --t-max 4 --errors 30 --max-wbeats 4096 --big-pct 50 > /dev/null \
                                                                                || fail "gen v"
"$GEN" -q --ops m3 -o "$DIR/vc1" -n 8 -s 15 --num-ch 1 --spm-kib 64 --ch-gib 1 --addr-beat-w 22 \
       --k-max 1024 --t-max 2 --errors 20 --max-wbeats 2048 > /dev/null          || fail "gen vc1"
"$GEN" -q --ops m3 -o "$DIR/ve" -n 14 -s 16 --t-max 3 --errors 100 --max-groups 1 --max-k 512 \
       --max-wbeats 1024 > /dev/null                                            || fail "gen ve"
"$GEN" -q --ops m3 -o "$DIR/x0" -n 1 --max-wbeats 0 2> /dev/null && fail "--max-wbeats 0 accepted"
"$GEN" -q -o "$DIR/x17" -n 1 --num-ch 17 2> /dev/null && fail "--num-ch 17 accepted"
"$GEN" -q -o "$DIR/x17" -n 1 --ops m9 2> /dev/null && fail "--ops m9 accepted"
rm -rf "$DIR/x17" "$DIR/x0"
for t in "$DIR"/m/*/ "$DIR"/mc1/*/; do        # M2 traces contain no MVM instruction
    od -An -tx1 -w32 -v "$t/cmd.bin" | awk '{ if ($1 == "20") bad = 1 } END { exit bad }' \
        || fail "M2 trace $t contains an MVM"
done
for t in "$DIR"/v/*/ "$DIR"/vc1/*/ "$DIR"/ve/*/; do   # M3 traces contain an MVM instruction
    od -An -tx1 -w32 -v "$t/cmd.bin" | awk '{ if ($1 == "20") ok = 1 } END { exit !ok }' \
        || fail "M3 trace $t contains no MVM"
done

"$GEN" -q -o "$DIR/a2" -n 12 -s 7 || fail "gen a2"
diff -r "$DIR/a" "$DIR/a2" > /dev/null || fail "generation is not deterministic"
rm -rf "$DIR/a2"
"$GEN" -q --ops m2 -o "$DIR/m2" -n 8 -s 12 --t-max 4 --errors 60 || fail "gen m2"
diff -r "$DIR/m" "$DIR/m2" > /dev/null || fail "M2 generation is not deterministic"
rm -rf "$DIR/m2"
"$GEN" -q --ops m3 -o "$DIR/v2" -n 8 -s 14 --t-max 4 --errors 30 --max-wbeats 4096 --big-pct 50 > /dev/null \
                                                                                || fail "gen v2"
diff -r "$DIR/v" "$DIR/v2" > /dev/null || fail "M3 generation is not deterministic"
rm -rf "$DIR/v2"

n=$(find "$DIR" -name meta.json | wc -l)
[ "$n" -eq 75 ] || fail "expected 75 traces, found $n"

"$REPLAY" -q "$DIR"
rc=$?
[ "$rc" -eq 0 ] || fail "replay of intact traces: exit code $rc"

f=$(ls "$DIR"/a/gen_s7_0003/expect/*.bin | head -n 1)
[ -n "$f" ] || fail "no expect file to corrupt"
b=$(od -An -tu1 -j 5 -N 1 "$f" | tr -d ' ')
nb=$(( (b ^ 0x5A) & 255 ))
printf "$(printf '\\%03o' "$nb")" | dd of="$f" bs=1 seek=5 conv=notrunc status=none
"$REPLAY" -q "$DIR/a" 2> "$DIR/replay_err.txt"
rc=$?
[ "$rc" -eq 1 ] || fail "corrupted trace: expected exit code 1, got $rc"
grep -q "expect mismatch at ch" "$DIR/replay_err.txt" || fail "no mismatch report"

"$REPLAY" -q "$DIR/does-not-exist" 2> /dev/null
rc=$?
[ "$rc" -eq 1 ] || fail "missing path: expected exit code 1, got $rc"

rm -rf "$DIR"
echo "PASS: 75 generated traces (31 M1, 14 M2, 30 M3) replayed; corruption detected"
