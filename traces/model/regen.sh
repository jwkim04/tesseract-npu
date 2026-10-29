#!/usr/bin/env bash
# traces/model/regen.sh [BUILD_DIR] — v1 real-model RTL regression traces: ONE minimal llama.cpp run per
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
# X0 (docs/v2 EX §12, SW §5.6): these v1 traces stay on disk only as the input of tsr-trace-conv
# (traces/model-v1conv, Y2), and only the v1 tools can reproduce them: BUILD_DIR (default build/v1,
# a binary copy of tag v1-m4) is checked by traces/v1_tools.sh first (v1 tools present, v1 traces,
# loader pointed at BUILD_DIR/bin, TSR_DEVICE=sim). Each model's set is written to <set>.new and
# replayed there; only then does it replace the old set, so a failed run leaves traces/model as it
# was.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
. "$ROOT/traces/v1_tools.sh"
v1_tools "${1:-$ROOT/build/v1}" llama-completion libggml-tesseract.so.0
for m in q8_0 q4_k_m-requant; do
    [ -e "$ROOT/models/qwen2.5-0.5b-instruct-$m.gguf" ] || { echo "$0: models/qwen2.5-0.5b-instruct-$m.gguf not found"; exit 2; }
done
for m in q8_0 q4_k_m-requant; do
    d=$HERE/qwen05b-$m
    n=$d.new
    rm -rf "$n"
    TSR_TRACE=$n TSR_SIM_THREADS=4 "$V1_BUILD/bin/llama-completion" -m "$ROOT/models/qwen2.5-0.5b-instruct-$m.gguf" \
        -ngl 99 -nkvo -c 256 -t 4 -p "Hi" -n 1 --no-warmup -no-cnv --temp 0 -s 1 < /dev/null > /dev/null 2> "$n.log" \
        || { echo "llama-completion failed for $m, see $n.log ($d is unchanged)"; exit 1; }
    "$V1_REPLAY" -q "$n" || { echo "replay of $n failed ($d is unchanged)"; exit 1; }
    rm -f "$n.log"
    rm -rf "$d"
    mv "$n" "$d"
    echo "$m: $(find "$d" -name meta.json | wc -l) traces, $(du -sh "$d" | cut -f1)"
done
python3 "$ROOT/traces/m3/trace_stats.py" --classes "$HERE"
