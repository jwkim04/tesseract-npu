#!/usr/bin/env bash
# traces/model/regen.sh [BUILD_DIR] — real-model RTL regression traces: ONE minimal llama.cpp run per
# model on the simulated device (sim transport, U200 v1 config: 4 ch, T_MAX 1, 4 MiB SPM) with
# TSR_TRACE, which dumps every executed Tesseract program (one per split) as a trace directory; then
# every trace is replayed on the ISA simulator and summarised.
#
#   llama-completion -m MODEL -ngl 99 -nkvo -c 256 -t 4 -p "Hi" -n 1 --no-warmup -no-cnv --temp 0 -s 1
#
# "Hi" is a single token for the Qwen2.5 tokenizer (no BOS), so the run evaluates exactly one graph
# with one token — the same graph as one decode step — and samples one token; -n 1 ends the run
# before a second graph, --no-warmup skips the warm-up graph, -nkvo keeps the KV cache on the CPU
# (v1 requirement). A Qwen2.5-0.5B graph has 145 Tesseract splits (6 per layer x 24 layers + the
# output head), so each model gives 145 traces 000000..000144 (T = 1 MVMs only, no ACC, explicit
# dst_stride, K = 896 or 4864):
#
#   qwen05b-q8_0/            models/qwen2.5-0.5b-instruct-q8_0.gguf             145 traces, 507 MiB
#                            169 MVMs, all Q8_0
#   qwen05b-q4_k_m-requant/  models/qwen2.5-0.5b-instruct-q4_k_m-requant.gguf   145 traces, 482 MiB
#                            169 MVMs: 12 Q4_K (ffn_down of 12 layers, K 4864 = 19 superblocks, partial
#                            chunk), the rest Q5_0 / Q6_K converted to Q8_0 at load
#
# Per-trace weight beats on channel 0 (= cycles of the busiest cluster at 1 beat / cycle):
#   attn_q / attn_output N 896  3360 | attn_k / attn_v N 128  480 | ffn_gate + ffn_up (one split,
#   2 x N 4864) 36480 | ffn_down Q8_0 18144, Q4_K 9632 | output head N 151936 569760 (140 MiB trace).
# Nothing was dropped (both models together ~1 GB). Shapes per trace:
#   python3 traces/m3/trace_stats.py --classes traces/model
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
BUILD=${1:-$ROOT/build/dev}
for m in q8_0 q4_k_m-requant; do
    d=$HERE/qwen05b-$m
    rm -rf "$d"
    TSR_TRACE=$d TSR_SIM_THREADS=4 "$BUILD/bin/llama-completion" -m "$ROOT/models/qwen2.5-0.5b-instruct-$m.gguf" \
        -ngl 99 -nkvo -c 256 -t 4 -p "Hi" -n 1 --no-warmup -no-cnv --temp 0 -s 1 < /dev/null > /dev/null 2> "$d.log" \
        || { echo "llama-completion failed for $m, see $d.log"; exit 1; }
    rm -f "$d.log"
    echo "$m: $(find "$d" -name meta.json | wc -l) traces, $(du -sh "$d" | cut -f1)"
done
"$BUILD/bin/tsr-trace-replay" -q "$HERE"
python3 "$ROOT/traces/m3/trace_stats.py" --classes "$HERE"
