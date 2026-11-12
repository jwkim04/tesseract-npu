#!/usr/bin/env bash
# capture_sim.sh MODEL OUTDIR [N_TOKENS]: one decode run on the simulator with the profile dump on (docs/prof-dbg.md
# "The profile report"). Writes OUTDIR/{plan_*.json,launches.jsonl,trw.raw,meta.json,cpu_fallback.json,run.log,sched.log}.
#
#   TSR_BUILD        build tree with llama-completion (default build/dev)
#   SIM_CARD         TSR_SIM_CARD value (default k4dual: the release image, with the dual weight stream); "" for the plain simulator
#   TSR_SIM_GDN      the fused GDN op of the release image (CAPS bit 8): 1 with a card configuration unless set, 0 for the composed path
#   TSR_SIM_DRAM_DIR directory for the file-backed simulator DRAM (a 26B model needs about 20 GB there)
#   PROMPT           the prompt (default "The history of the printing press begins")
#   CTX, EXTRA       context size (default 4096) and extra llama-completion arguments
set -euo pipefail
model=$1; out=$2; ntok=${3:-4}
here=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
build=${TSR_BUILD:-$here/build/dev}
mkdir -p "$out"
rm -f "$out"/plan_*.json "$out"/launches.jsonl "$out"/trw.raw "$out"/meta.json "$out"/cpu_fallback.json
export TSR_IMAGE=${TSR_IMAGE:-$build/fw/tsr-fw.elf}
export LLAMA_ATTN_ROT_DISABLE=1 TSR_DEVICE=sim TSR_STATS=1
[ -n "${SIM_CARD-k4dual}" ] && export TSR_SIM_CARD=${SIM_CARD-k4dual} TSR_SIM_GDN=${TSR_SIM_GDN-1}
export TSR_DUMP_PLAN=$out
timeout "${TIMEOUT:-7200}" env GGML_SCHED_DEBUG=2 "$build/bin/llama-completion" -m "$model" -ngl 99 -fa on -ctk q8_0 -ctv q8_0 \
    -c "${CTX:-4096}" --temp 0 -no-cnv -p "${PROMPT:-The history of the printing press begins}" -n "$ntok" -t 8 -lv 5 ${EXTRA:-} \
    < /dev/null > "$out/run.out" 2> "$out/sched_and_run.log"
grep -v "^[0-9.]* [DIWE] \(node\|  \)" "$out/sched_and_run.log" | grep -a "SPLIT\|common_perf_print\|\[tesseract\]" > "$out/run.log" || true
echo "capture_sim: $out"
