#!/usr/bin/env bash
# traces/at/regen.sh [BUILD_DIR] — AT regression traces (`ops` traces, docs/design/06-software-
# verification.md §5.6 "new"): tsr-trace-gen --ops at, the seven AT ops of docs/design/03-attention-
# kv.md §5 and the flash-attention chain of §6.6 across the DMA, QACT, MVM and VU queues, then every
# trace replayed under the fixed and two random schedules (race detector fatal),
# tile-only and through the replay kernel (one launch per trace: SPM_STALE is fatal, SW §4.6).
#
#   tmax8/   64 traces, U200 config (4 ch, 4 MiB SPM) with T_MAX 8, CAPS VU|ATT: KVST into host-filled
#            or zero planes (partial meta beats read-modify-written), ATTQK (CAUSAL / SLIDE / MAXOUT,
#            several passes per head group, rstride gaps), ATTPV (LSUM, pstride / cstride gaps, LG),
#            ASMAX / AMASK (in place, masks with -inf), AMERGE (INIT and merges), ANORM, FA chains;
#            ~20% error injections
#   tmax1/   24 traces, T_MAX 1 (decode only: T_q = GS = 1)
#   errors/  56 traces with an error injection: two cycles through the 24 AT fault kinds (every row
#            of AT §5.8 with its ERR_INFO, reserved bits, KVST sub, ATTQK w7 without MAXOUT, BAD_FMT,
#            the AT-reserved opcodes) and the dependency endings
#   ch3/     16 traces, 3 channels (groups round-robin over an odd channel count), T_MAX 2, 512 KiB SPM
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
. "$ROOT/traces/v2_tools.sh"
v2_tools "${1:-$ROOT/build/dev}"
GEN=$V2_GEN
rm -rf "$HERE"/tmax8 "$HERE"/tmax1 "$HERE"/errors "$HERE"/ch3
"$GEN" -q --ops at --prefix at -o "$HERE/tmax8"  -n 64 -s 51 --t-max 8 --errors 20
"$GEN" -q --ops at --prefix at -o "$HERE/tmax1"  -n 24 -s 52 --t-max 1 --errors 20
"$GEN" -q --ops at --prefix at -o "$HERE/errors" -n 56 -s 53 --t-max 8 --errors 100 --max-groups 1
"$GEN" -q --ops at --prefix at -o "$HERE/ch3"    -n 16 -s 54 --t-max 2 --num-ch 3 --spm-kib 512 --errors 25
v2_replay "$HERE"
v2_replay_kernel "$HERE"
