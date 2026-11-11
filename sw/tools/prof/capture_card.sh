#!/usr/bin/env bash
# capture_card.sh MODEL OUTDIR [N_TOKENS]: one card session that records a decode run for the profile report
# (docs/prof-dbg.md "The profile report"): waits until nobody has the card open, optionally replays the first-ATTPV trace (WARM=1), runs
# llama-completion with TSR_DUMP_PLAN (the plan, the launches with PERF deltas and the trace writer's records), and clears the
# trace writer afterward. Never kills another process.
#
#   TSR_BUILD     build tree with the XRT backend (default build/xrt)
#   TSR_XCLBIN    the card image (default: build/u200/k4/tsr_u200_k4.xclbin, what hw/u200/build_xclbin.sh k4 writes)
#   WARM_TRACE    trace directory of the first-ATTPV replay (default traces/at/errors/at_s53_0009 of this checkout)
#   WARM=1        replay the first-ATTPV trace first (for an image with that defect; off by default)
#   NO_DUMP=1     no TSR_DUMP_PLAN: the speed of the same run without the profiler
#   PROMPT, CTX, EXTRA, TIMEOUT as in capture_sim.sh
set -uo pipefail
model=$1; out=$2; ntok=${3:-33}
here=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
build=${TSR_BUILD:-$here/build/xrt}
source /opt/xilinx/xrt/setup.sh > /dev/null 2>&1
export TSR_XCLBIN=${TSR_XCLBIN:-$here/build/u200/k4/tsr_u200_k4.xclbin}
export TSR_IMAGE=${TSR_IMAGE:-$build/fw/tsr-fw.elf}
warm=${WARM_TRACE:-$here/traces/at/errors/at_s53_0009}
mkdir -p "$out"
rm -f "$out"/plan_*.json "$out"/launches.jsonl "$out"/trw.raw "$out"/meta.json "$out"/cpu_fallback.json

card_idle() { ! lsof -w /dev/dri/by-path/pci-0000:01:00.1-render 2>/dev/null | awk 'NR>1' | grep -q .; }
n=0
until card_idle; do n=$((n + 1)); [ $((n % 15)) -eq 1 ] && echo "$(date +%T) card busy, waiting" >&2; sleep 4; done

df_kb=$(df --output=avail -k / | tail -1)
[ "$df_kb" -lt $((40 * 1024 * 1024)) ] && { echo "root disk below 40 GB free, stopping" >&2; exit 1; }

cleanup() {   # the trace writer's configuration survives the process: leave it off
  "$build/bin/tsr-dbg" --device xrt --attach -ex "trw stop" > "$out/trw_stop.txt" 2>&1 || true
}
trap cleanup EXIT

if [ -n "${WARM:-}" ] && [ -z "${SKIP_WARM:-}" ]; then
  timeout 300 "$build/bin/tsr-selftest" --device xrt --no-mem --no-kernel --rate-n 10 --traces "$warm" -v > "$out/warm.log" 2>&1
  echo "warm rc=$?" >> "$out/warm.log"
  tail -2 "$out/warm.log"
fi

{
  echo "{\"date\":\"$(date -Is)\",\"host\":\"$(hostname)\",\"model\":\"$model\",\"model_bytes\":$(stat -c %s "$model"),\"n_tokens\":$ntok,"
  echo " \"xclbin\":\"$TSR_XCLBIN\",\"xclbin_sha256\":\"$(sha256sum "$TSR_XCLBIN" | cut -d' ' -f1)\",\"firmware\":\"$TSR_IMAGE\","
  echo " \"git\":\"$(git -C "$here" rev-parse --short HEAD)\",\"loadavg\":\"$(cut -d' ' -f1-3 /proc/loadavg)\"}"
} > "$out/capture.json"

export LLAMA_ATTN_ROT_DISABLE=1 TSR_STATS=1 TSR_DEVICE=xrt
[ -n "${NO_DUMP:-}" ] || export TSR_DUMP_PLAN=$out   # NO_DUMP=1: the same run with the profiler off (llama.cpp's eval time is the reference speed)
timeout "${TIMEOUT:-3000}" "$build/bin/llama-completion" -m "$model" -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -c "${CTX:-4096}" --temp 0 -no-cnv \
    -p "${PROMPT:-The history of the printing press begins}" -n "$ntok" -t 8 ${EXTRA:-} < /dev/null > "$out/run.out" 2> "$out/run.log"
echo "rc=$?" >> "$out/run.log"
"$build/bin/tsr-dbg" --device xrt --attach -ex perf -ex status > "$out/after.txt" 2>&1 || true
echo "capture_card: $out"
