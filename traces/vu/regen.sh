#!/usr/bin/env bash
# traces/vu/regen.sh [BUILD_DIR] — VU regression traces (`ops` traces, docs/design/06-software-
# verification.md §5.6 "new"): tsr-trace-gen --ops vu, every VU op of docs/design/02-vector-unit.md
# §5 with legal and illegal encodings, then every trace replayed under the fixed and two random
# schedules (race detector fatal),
# tile-only and through the replay kernel (one launch per trace: SPM_STALE is fatal, SW §4.6).
#
#   m8/      96 traces, U200 config with T_MAX 8 and CAPS VU|ATT|TOPK|DQROW (the simulator's M8 set):
#            each group one op of the 67-entry cycle (25 VOP functions with cmode 0 / 1 / 2, 14 RED
#            kind / post / TO_S forms, 10 ROPE mode x form (recurrence, FF, FS, FF + FS, ATAB), TOPK,
#            15 SOP subs, DQROW Q8_0 / Q4_K) on loaded data with special values; n = 0 ops, tails,
#            broadcast sources, destination gaps, in-place ops; ~25% error injections
#   m6/      32 traces, CAPS VU|ATT (the M6 bitstream: TOPK and DQROW are refused, ILLEGAL_OP on SYNC)
#   errors/  88 traces with an error injection: two cycles through the 40 VU fault kinds (reserved
#            bits, undefined enumerations, unused fields, every OPERAND condition of VU §5.9 with its
#            ERR_INFO, SPM_RANGE per operand in field order, SOP LDS / STS addresses, DQROW BAD_FMT /
#            K_RANGE, VMASK and reserved VU opcodes) and the dependency endings (BAD_DEP, DEP_FAILED chain,
#            QABORT recipe, QCLEAR recovery)
#   small1ch/ 24 traces, 1 channel, 64 KiB SPM, 1 GiB channel with ADDR_BEAT_W 22, T_MAX 1, CAPS m8
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
. "$ROOT/traces/trace_tools.sh"
trace_tools "${1:-$ROOT/build/dev}"
GEN=$TT_GEN
rm -rf "$HERE"/m8 "$HERE"/m6 "$HERE"/errors "$HERE"/small1ch
"$GEN" -q --ops vu --prefix vu -o "$HERE/m8"       -n 96 -s 41 --t-max 8 --caps m8 --errors 25
"$GEN" -q --ops vu --prefix vu -o "$HERE/m6"       -n 32 -s 42 --t-max 8 --caps m6 --errors 25
"$GEN" -q --ops vu --prefix vu -o "$HERE/errors"   -n 88 -s 43 --t-max 8 --caps m8 --errors 100 --max-groups 2
"$GEN" -q --ops vu --prefix vu -o "$HERE/small1ch" -n 24 -s 44 --t-max 1 --caps m8 --errors 20 --num-ch 1 \
       --spm-kib 64 --ch-gib 1 --addr-beat-w 22
trace_replay "$HERE"
trace_replay_kernel "$HERE"
