#!/bin/sh
# ref_math_np_cmp.sh — compare the independent numpy model (ref_math_np.py) with the C reference
# (ref_math_dump --check) bit for bit (VU §10 item 1, SW §10.1, V2-SW-23). Used by the
# ref_math_np_* ctests of sw/sim/ref_math.cmake.
#
#   ref_math_np_cmp.sh PYTHON NP_PY DUMP strided FUNC [B...]
#       2^24 inputs at stride 256 from 0 (every sign / exponent / top-15-bit mantissa) and 2^24 at
#       the odd stride 257 from 0 for every B, then the special windows below (stride 1) with the
#       first B.
#   ref_math_np_cmp.sh PYTHON NP_PY DUMP full FUNC [B]
#       all 2^32 inputs in 256 chunks of 2^24; runs only with TSR_REF_MATH_FULL=1 (else exit 77,
#       which the slow ctests treat as "skipped").
#
# FUNC is a name both CLIs know. A binary function (and expk, whose B is the κ index) takes one
# or more fixed second operands B (hex); strided runs every B, full only the first. Exit status:
# 0 = bit-identical everywhere, 1 = a mismatch or a failing command, 77 = skipped.
set -u
PY=$1 NP=$2 DUMP=$3 MODE=$4 FN=$5
shift 5

# Special windows (start count): signed zeros and subnormals, smallest normals, largest finites,
# infinities and NaNs; TANH's small-path edge 2^-6, GELU16's +-10 clamps, EXP's +-88.7 overflow /
# underflow and the -87.3 .. -91.9 SILU flush region, EXPK's Ev = 156 saturation, SIN/COS's
# E = 115 identity edge, the fp16 subnormal / normal / max-finite edges, 1, pi, pi/2, 2^24 and
# the GELU -0 region (-10.0 .. -10.1) and GELU16's 2-ulp table region (about -5.2 .. -4.7).
WINDOWS="00000000:8192 80000000:8192 007FE000:16384 807FE000:16384 7F7FE000:16384 FF7FE000:16384
7F800000:4096 FF800000:4096 7FBFF000:8192 FFBFF000:8192 FFFFF000:4096 3C7FF000:8192 BC7FF000:8192
411FF000:8192 C11FF000:8192 42B0F000:16384 C2B0F000:16384 C2AEF000:16384 4DFFF000:8192
CDFFF000:8192 4E7FF000:8192 397FF000:8192 B97FF000:8192 337FF000:12288 387FF000:8192 477FE000:8192
C77FE000:8192 3F7FF000:16384 BF7FF000:8192 40490000:8192 3FC90000:8192 4B7FF000:8192
C120A000:65536 C0A50000:65536"

fails=0
cmp_one() {   # cmp_one start count stride [b]
    if ! "$PY" "$NP" "$FN" "$@" | "$DUMP" --check "$FN" "$@"; then
        echo "MISMATCH or failure: $FN $*"
        fails=$((fails + 1))
    fi
}

case $MODE in
strided)
    for b in ${@:-""}; do
        cmp_one 0 16777216 256 $b
        cmp_one 0 16777216 257 $b
    done
    for w in $WINDOWS; do
        cmp_one "${w%%:*}" "${w##*:}" 1 ${1:-}
    done
    ;;
full)
    if [ "${TSR_REF_MATH_FULL:-0}" != 1 ]; then
        echo "skipped (set TSR_REF_MATH_FULL=1)"
        exit 77
    fi
    k=0
    while [ $k -lt 256 ]; do
        cmp_one "$(printf '%08x' $((k * 16777216)))" 16777216 1 ${1:-}
        k=$((k + 1))
    done
    ;;
*)
    echo "usage: ref_math_np_cmp.sh PYTHON NP_PY DUMP strided|full FUNC [B...]" >&2
    exit 2
    ;;
esac
[ $fails -eq 0 ] && echo "ref_math_np_cmp.sh $MODE $FN: bit-identical" && exit 0
echo "ref_math_np_cmp.sh $MODE $FN: $fails failing comparison(s)"
exit 1
