#!/usr/bin/env bash
# test_selftest_traces.sh SELFTEST IMAGE TRACES [MAX] — ctest tools_selftest_mock_traces and
# tools_selftest_mock_traces_tmax8 (docs/design/06-software-verification.md §10.2, §10.4 M6 gate item 1):
# tsr-selftest --traces over the sets through the mock kdev with the shadow simulator: every trace
# of traces/m3, traces/vu, traces/at and traces/ctrl through the `replay` kernel, and traces/
# model-v1conv with the replay kernel's SERIAL flag (SW §5.4). The mock's configuration comes from
# the environment (TSR_MOCK_TMAX, TSR_MOCK_NUM_CH, ...); a trace made for another configuration is
# skipped (tsr_devrun.h) and counted. MAX > 0 caps the replays per set. The sets are not in git
# (traces/*/regen.sh): the test is skipped (77) when none is on disk.
set -u
ST=$1
IMG=$2
TR=$3
MAX=${4:-0}
args=()
for s in m3 vu at ctrl; do
    [ -d "$TR/$s" ] && args+=(--traces "$TR/$s")
done
[ -d "$TR/model-v1conv" ] && args+=(--traces-serial "$TR/model-v1conv")
if [ ${#args[@]} -eq 0 ]; then
    echo "test_selftest_traces: no trace set under $TR (traces/*/regen.sh): skipped"
    exit 77
fi
[ "$MAX" -gt 0 ] && args+=(--max-traces "$MAX")
exec "$ST" --device mock --image "$IMG" --no-mem --no-kernel --rate-n 0 "${args[@]}"
