#!/usr/bin/env python3
"""Generate text with a Hugging Face causal LM through torch.compile(backend="tesseract") and time it.

    PYTHONPATH= build/tools/torchvenv/bin/python -I sw/torch/examples/hf_generate.py \
        models/hf/Qwen2.5-0.5B-Instruct [--kv-cache on|off | --eager] [--new 64] [--reps 3] [--json OUT]

(-I: the model directory is data; this script puts sw/torch and the directory of the extension, TSR_TORCH_EXT_DIR
or else build/torch/python, on sys.path itself.)
The model is loaded from a local directory: safetensors only, no remote code.

The forward of the model is compiled in full mode (model.forward = tt.compile(model.forward)) and model.generate
runs greedy with Hugging Face's static KV cache (docs/torch-backend.md §1). The prompt is one user turn in the
model's chat template, or the plain text when the tokenizer has no template.

  --kv-cache on   (default) cached attention: the attention of every call runs on the device over the K and V rows
                  that the device keeps, so for a Qwen2.5 decoder a decode step is one region call and one launch
  --kv-cache off  the option kv_cache off: the cache update and the attention stay on the CPU and end the regions
  --eager         no backend: PyTorch on the host CPU in float32, for comparison

The first generation of a process compiles and uploads the weights. It is printed apart and not counted. Each of
the --reps generations after it prints
  prefill   the time from the call of generate to the first new token, and the prompt tokens per second of it
  decode    the mean time between two new tokens after the first, and its inverse in tokens per second
and the counters of that generation: region calls, launches, calls that fell back to the CPU, host to device and
device to host transfers, device cycles. The time of a token is the time at which generate hands it to its
streamer. min_new_tokens keeps a generation going past the end of sequence token, so that every repetition times
--new tokens; the text after that token means nothing. The last line gives the median over the repetitions.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "sw" / "torch"), os.environ.get("TSR_TORCH_EXT_DIR") or str(REPO / "build" / "torch" / "python")]

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from transformers.generation.streamers import BaseStreamer  # noqa: E402

PROMPT = "Name the capital of France in one word."

# name in the output -> the counters of tt.counters() that add up to it
COUNTERS = {
    "region_calls": ("region_calls",),
    "kv_region_calls": ("kv_region_calls",),
    "launches": ("rt_execs",),
    "cpu_fallback_calls": ("region_cpu_fallback_calls", "cpu_fallback_calls"),
    "h2d_writes": ("rt_h2d_writes",),
    "d2h_reads": ("rt_d2h_reads",),
    "device_cycles": ("rt_perf_cycles",),
    "uploads": ("uploads",),
    "plan_misses": ("rt_graph_plan_misses",),
}


class Clock(BaseStreamer):
    """Records the time of every call of generate: first the prompt, then each new token."""

    def __init__(self) -> None:
        self.t: list[float] = []

    def put(self, value) -> None:
        self.t.append(time.perf_counter())

    def end(self) -> None:
        pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir")
    ap.add_argument("--prompt", default=PROMPT, help="the user turn (or the plain text without a chat template)")
    ap.add_argument("--kv-cache", choices=["on", "off"], default="on", help="the option kv_cache of the backend")
    ap.add_argument("--eager", action="store_true", help="no backend: PyTorch on the host CPU")
    ap.add_argument("--new", type=int, default=64, help="new tokens of every generation (at least 2)")
    ap.add_argument("--reps", type=int, default=3, help="timed generations after the first")
    ap.add_argument("--cache-len", type=int, default=256, help="max_cache_len of the static cache")
    ap.add_argument("--json", help="write the results here")
    args = ap.parse_args()
    if args.new < 2 or args.reps < 1:
        ap.error("--new must be at least 2 and --reps at least 1")
    if args.eager and args.kv_cache == "off":
        ap.error("--eager does not use the backend: --kv-cache has no meaning with it")
    mode = "eager" if args.eager else f"kv_cache {args.kv_cache}"

    tok = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=False,
                                                 use_safetensors=True, dtype=torch.float32).eval()
    if tok.chat_template:
        enc = tok.apply_chat_template([{"role": "user", "content": args.prompt}], add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True)
    else:
        enc = tok(args.prompt, return_tensors="pt")
    enc = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
    T = int(enc["input_ids"].shape[1])
    if T + args.new > args.cache_len:
        ap.error(f"the prompt ({T} tokens) and --new {args.new} do not fit --cache-len {args.cache_len}")

    tt = None
    device = "PyTorch eager on the host CPU"
    if not args.eager:
        import tesseract_torch as tt  # noqa: PLC0415
        model.forward = tt.compile(model.forward, mode="full", kv_cache=args.kv_cache == "on")
        device = tt.device().description
    print(f"{args.model_dir}: {mode}; {device}; torch {torch.__version__}, {torch.get_num_threads()} threads", flush=True)

    def counters() -> dict:
        if tt is None:
            return dict.fromkeys(COUNTERS, 0)
        c = tt.counters()
        return {name: sum(c.get(k, 0) for k in keys) for name, keys in COUNTERS.items()}

    def one(tag: str) -> dict:
        clk = Clock()
        c0 = counters()
        t0 = time.perf_counter()
        out = model.generate(**enc, min_new_tokens=args.new, max_new_tokens=args.new, do_sample=False,
                             cache_implementation="static", max_cache_len=args.cache_len, streamer=clk)
        t1 = time.perf_counter()
        c1 = counters()
        ts = clk.t                                    # ts[0]: the prompt, ts[1]: the first new token, ...
        steps = len(ts) - 2                           # decode steps: every new token after the first
        prefill = ts[1] - t0
        decode = (ts[-1] - ts[1]) / steps
        d = {k: c1[k] - c0[k] for k in COUNTERS}
        r = {"tag": tag, "prompt_tokens": T, "new_tokens": int(out.shape[1]) - T, "decode_steps": steps,
             "total_s": t1 - t0, "prefill_s": prefill, "prefill_tok_s": T / prefill, "decode_ms_per_token": decode * 1e3,
             "decode_tok_s": 1.0 / decode, "counters": d, "text": tok.decode(out[0, T:])}
        line = (f"{tag}: prefill {prefill * 1e3:.1f} ms ({T / prefill:.1f} tok/s over {T} prompt tokens), decode "
                f"{decode * 1e3:.2f} ms per token ({1.0 / decode:.2f} tok/s over {steps} steps)")
        if tt is not None:
            line += (f"; region calls {d['region_calls']} ({d['kv_region_calls']} with cached attention), launches "
                     f"{d['launches']}, CPU fallbacks {d['cpu_fallback_calls']}, transfers {d['h2d_writes']} to and "
                     f"{d['d2h_reads']} from the device, device cycles {d['device_cycles']}, uploads {d['uploads']}, "
                     f"plan misses {d['plan_misses']}")
        print(line, flush=True)
        return r

    res = {"model_dir": args.model_dir, "mode": mode, "device": device, "torch": torch.__version__,
           "torch_threads": torch.get_num_threads(), "prompt": args.prompt, "prompt_tokens": T, "new_tokens": args.new,
           "cache_len": args.cache_len, "load_before": os.getloadavg()}
    res["first"] = one("first generation (not counted)" if tt is None else "first generation (compiles, uploads; not counted)")
    res["runs"] = [one(f"generation {i + 2}") for i in range(args.reps)]
    res["load_after"] = os.getloadavg()
    dec = [r["decode_tok_s"] for r in res["runs"]]
    pre = [r["prefill_tok_s"] for r in res["runs"]]
    res["decode_tok_s_median"] = statistics.median(dec)
    res["prefill_tok_s_median"] = statistics.median(pre)
    print(f"text: {res['runs'][-1]['text'][:80]!r}")
    print(f"{mode}: decode {res['decode_tok_s_median']:.2f} tok/s (median of {len(dec)}; {min(dec):.2f} to {max(dec):.2f}), "
          f"prefill {res['prefill_tok_s_median']:.1f} tok/s (median; {min(pre):.1f} to {max(pre):.1f})", flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
