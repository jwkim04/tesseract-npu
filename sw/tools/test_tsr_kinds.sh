#!/usr/bin/env bash
# test_tsr_kinds.sh klcheck|numcheck TOOL MODEL OUT — ctests tools_klcheck_kinds_qwen05b and
# tools_numcheck_kinds_qwen05b (docs/design/06-software-verification.md SW-49): a device run takes its
# TSR_KINDS from the phase. --phase m6 sets m6 whatever the caller exported; --phase m7 keeps the
# caller's list, and without one the device takes every kind. The tool must exit 0, the backend's
# TSR_DEBUG line must name the kinds the device took, and the tool must record them (klcheck: the
# "TSR_KINDS" line of the run; tsr-numcheck: info.TSR_KINDS of meta.json).
set -eu
WHAT=$1
TOOL=$2
MODEL=$3
OUT=$4
rm -rf "$OUT"
mkdir -p "$OUT"

fail() { echo "test_tsr_kinds: FAIL: $*" >&2; exit 1; }

# run PHASE KINDS (a list, or "-" for TSR_KINDS unset) NAME
run() {
    local phase=$1 kinds=$2 name=$3
    local log="$OUT/$name.log"
    local -a env_kinds=(env -u TSR_KINDS)
    [ "$kinds" = "-" ] || env_kinds=(env "TSR_KINDS=$kinds")
    local rc=0
    if [ "$WHAT" = klcheck ]; then
        TSR_DEBUG=1 "${env_kinds[@]}" "$TOOL" run -m "$MODEL" --device sim --phase "$phase" --n-prompts 1 -n 2 \
            --out "$OUT/$name.klc" >"$log" 2>&1 || rc=$?
    else
        TSR_DEBUG=1 "${env_kinds[@]}" "$TOOL" run -m "$MODEL" --device sim --phase "$phase" -T 4 --nodes result_output \
            --run dev -o "$OUT/$name" >"$log" 2>&1 || rc=$?
    fi
    [ "$rc" = 0 ] || { cat "$log"; fail "$name: the run exited with $rc"; }
}

# expect NAME KINDS BACKEND_TAIL: the backend's line "TSR_KINDS=KINDS: the device takes BACKEND_TAIL" and
# the tool's record of KINDS
expect() {
    local name=$1 kinds=$2 tail=$3
    grep -Fq "[tesseract] TSR_KINDS=$kinds: the device takes $tail" "$OUT/$name.log" ||
        { cat "$OUT/$name.log"; fail "$name: no backend line for TSR_KINDS=$kinds"; }
    if [ "$WHAT" = klcheck ]; then
        grep -Fxq "klcheck [sim]: TSR_KINDS $kinds" "$OUT/$name.log" || { cat "$OUT/$name.log"; fail "$name: TSR_KINDS not recorded as $kinds"; }
    else
        grep -Fq "\"TSR_KINDS\": \"$kinds\"" "$OUT/$name/meta.json" || { cat "$OUT/$name/meta.json"; fail "$name: meta.json TSR_KINDS is not $kinds"; }
    fi
}

# M7 keeps the caller's list
run m7 "all,-RMS_NORM" m7_list
expect m7_list "all,-RMS_NORM" "NOP JOIN MUL_MAT ADD MUL SCALE GLU ROPE GET_ROWS SET_ROWS_KV FLASH_ATTN CPY UNARY"

# M6 sets m6,+MUL_MAT_ID (the matrix products) over the caller's list
run m6 "all" m6_forced
expect m6_forced "m6,+MUL_MAT_ID" "NOP JOIN MUL_MAT MUL_MAT_ID"

# M7 without a list: every kind of the booted image
run m7 - m7_unset
if grep -Fq "[tesseract] TSR_KINDS=" "$OUT/m7_unset.log"; then
    fail "m7_unset: the backend reports a TSR_KINDS list"
fi
if [ "$WHAT" = klcheck ]; then
    grep -Eq "^klcheck \[sim\]: TSR_KINDS unset" "$OUT/m7_unset.log" || fail "m7_unset: not recorded as unset"
else
    grep -Eq '"TSR_KINDS": null' "$OUT/m7_unset/meta.json" || fail "m7_unset: meta.json TSR_KINDS is not null"
fi
echo "test_tsr_kinds: ok ($WHAT)"
