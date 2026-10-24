#!/usr/bin/env python3
"""Run a Hugging Face causal LM with torch.compile(backend="tesseract") and compare it with eager CPU.

    PYTHONPATH= build/tools/torchvenv/bin/python -I sw/torch/examples/hf_compare.py \
        models/hf/Qwen2.5-0.5B-Instruct [--gguf models/qwen2.5-0.5b-instruct-q8_0.gguf]

(-I: the model directory is data; this script puts sw/torch and build/torch/python on sys.path itself.)
The model is loaded from a local directory: safetensors only, no remote code.

1. The compiled model greedy-generates --gen tokens after the prompt (full recompute per step, no KV
   cache). Every step's logits must equal, bit for bit, eager execution of the same model on the same
   input with every nn.Linear on the device's reference model (tesseract_torch.reference_linears):
   the device arithmetic, the partition and the CPU ops together.
2. Eager CPU (the unquantized model) scores the device's sequence in one forward (klcheck's method,
   docs/klcheck.md): per position KL(cpu || dev) and top-1 agreement. For scale: the same for eager
   with every Linear weight replaced by its dequantized Q8_0 copy (the weight conversion alone), and
   for eager with the input embeddings nudged by one f32 ulp (klcheck's floor; with an unquantized
   reference this is ~0, so the spec's "KL <= 3 x floor", defined against the CPU backend on the same
   quantized weights, does not apply here).
3. --gguf: the device's Q8_0 blocks of every weight against that GGUF's Q8_0 tensors byte for byte,
   with the relative error of both dequantized copies against the checkpoint's weights (a GGUF made
   from other weights shows up as a large GGUF error).
Exit status 0 if (1) holds on every step and top-1 >= 80 %.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "sw" / "torch"), str(REPO / "build" / "torch" / "python")]

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import tesseract_torch as tt  # noqa: E402

PROMPT = "The quick brown fox jumps over the lazy dog. Tesseract is an open-source inference accelerator that"


def nudge(e: torch.Tensor, seed: int) -> torch.Tensor:
    """Every element moved by exactly one f32 ulp up or down (random sign per element)."""
    g = torch.Generator().manual_seed(seed)
    up = torch.rand(e.shape, generator=g) < 0.5
    return torch.where(up, torch.nextafter(e, torch.full_like(e, math.inf)), torch.nextafter(e, torch.full_like(e, -math.inf)))


def kl_rows(p_logits: torch.Tensor, q_logits: torch.Tensor) -> np.ndarray:
    lp = F.log_softmax(p_logits.double(), -1)
    lq = F.log_softmax(q_logits.double(), -1)
    return (lp.exp() * (lp - lq)).sum(-1).numpy()


GGUF_NAMES = {"self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v",
              "self_attn.o_proj": "attn_output", "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up",
              "mlp.down_proj": "ffn_down"}


def _dequant_q8_0(blocks: np.ndarray, N: int, K: int) -> torch.Tensor:
    b = blocks.reshape(N, K // 32, 34)
    d = b[:, :, 0:2].copy().view(np.float16).astype(np.float32)
    q = b[:, :, 2:34].view(np.int8).astype(np.float32)
    return torch.from_numpy((q * d).reshape(N, K))


def gguf_compare(model, path: str) -> dict:
    sys.path.insert(0, str(REPO / "third_party" / "llama.cpp" / "gguf-py"))
    import gguf  # noqa: PLC0415

    rd = gguf.GGUFReader(path)
    tensors = {t.name: t for t in rd.tensors}
    dev = tt.device()
    out = {"tensors": 0, "identical": 0, "blocks": 0, "blocks_differ": 0, "other_weights": 0, "per_tensor": [],
           "not_q8_0": [], "missing": []}
    tied = getattr(model.config, "tie_word_embeddings", False)
    seen = set()
    for name, p in model.named_parameters():
        if not name.endswith(".weight") or p.dim() != 2:
            continue
        if name == "lm_head.weight" or (tied and name == "model.embed_tokens.weight"):
            gname = "output.weight"
        elif ".layers." in name:
            i = name.split(".layers.")[1].split(".")[0]
            sub = name.split(f".layers.{i}.")[1].rsplit(".weight", 1)[0]
            if sub not in GGUF_NAMES:
                continue
            gname = f"blk.{i}.{GGUF_NAMES[sub]}.weight"
        else:
            continue
        if gname in seen:
            continue
        seen.add(gname)
        h = tt.weight_cache().find(p)
        if h is None:
            out["missing"].append(name)
            continue
        t = tensors.get(gname)
        if t is None or t.tensor_type != gguf.GGMLQuantizationType.Q8_0:
            out["not_q8_0"].append(gname)
            continue
        N, K = p.shape
        a = dev.read_blocks(h).numpy().reshape(N, -1)
        b = np.asarray(t.data).reshape(N, -1)
        nd = int((a.reshape(-1, 34) != b.reshape(-1, 34)).any(axis=1).sum())
        w = p.detach().float()
        nrm = float(w.norm())
        e_dev = float((_dequant_q8_0(a, N, K) - w).norm()) / nrm
        e_gguf = float((_dequant_q8_0(b, N, K) - w).norm()) / nrm
        out["tensors"] += 1
        out["blocks"] += a.size // 34
        out["blocks_differ"] += nd
        out["identical"] += nd == 0
        out["other_weights"] += e_gguf > 10 * e_dev
        out["per_tensor"].append({"gguf": gname, "hf": name, "blocks_differ": nd, "blocks": a.size // 34,
                                  "rel_err_dev": e_dev, "rel_err_gguf": e_gguf})
    return out


class dequant_linears:
    """Eager nn.Linear with the weight replaced by its dequantized device Q8_0 copy (float32)."""

    def __enter__(self):
        self.orig = torch.nn.Linear.forward

        def forward(mod, x):                    # dequantized per call: no second copy of the model in memory
            h = tt.weight_cache().find(mod.weight)
            if h is None:
                return self.orig(mod, x)
            N, K = mod.weight.shape
            return F.linear(x, _dequant_q8_0(tt.device().read_blocks(h).numpy(), N, K).to(x.dtype), mod.bias)

        torch.nn.Linear.forward = forward
        return self

    def __exit__(self, *a):
        torch.nn.Linear.forward = self.orig


def _f32_bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and torch.equal(a.float().view(torch.int32), b.float().view(torch.int32))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir")
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--tokens", type=int, default=16, help="prompt tokens (truncated)")
    ap.add_argument("--gen", type=int, default=8, help="greedy tokens generated by the device")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"], help="model / activation dtype")
    ap.add_argument("--seed", type=int, default=1, help="floor nudge seed")
    ap.add_argument("--gguf", help="GGUF of the same model: compare Q8_0 blocks byte for byte")
    ap.add_argument("--no-ref-check", action="store_true", help="skip the bit-exact reference check (1)")
    ap.add_argument("--json", help="write the results here")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    dtype = getattr(torch, args.dtype)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=False,
                                                 use_safetensors=True, dtype=dtype).eval()
    print(f"loaded {args.model_dir} ({sum(p.numel() for p in model.parameters()) / 1e6:.1f} M params, {args.dtype}) "
          f"in {time.time() - t0:.1f} s; device: {tt.device().description}", flush=True)
    ids = tok(args.prompt, return_tensors="pt").input_ids[:, :args.tokens]
    T = ids.shape[1]

    # (1) device generation, every step's logits kept
    compiled = tt.compile(model)
    steps = []
    seq = ids
    t0 = time.time()
    with torch.no_grad():
        for _ in range(args.gen):
            lg = compiled(input_ids=seq, use_cache=False).logits[0]
            steps.append((seq, lg))
            seq = torch.cat([seq, lg[-1].float().argmax().view(1, 1)], dim=1)
    t_dev = time.time() - t0
    cnt = tt.counters()
    print(f"device: {args.gen} forwards of {T}..{T + args.gen - 1} tokens in {t_dev:.1f} s (incl. compilation)", flush=True)

    ref_equal = None
    t_ref = 0.0
    if not args.no_ref_check:
        t0 = time.time()
        ref_equal = []
        with torch.no_grad(), tt.reference_linears():
            for s_in, lg in steps:
                ref_equal.append(_f32_bits_equal(model(input_ids=s_in, use_cache=False).logits[0], lg))
        t_ref = time.time() - t0

    # (2) eager CPU on the device's sequence
    dev_logits = torch.cat([steps[0][1].float()] + [lg[-1:].float() for _, lg in steps[1:]])
    ctx = seq[:, :-1]
    with torch.no_grad():
        cpu_logits = model(input_ids=ctx, use_cache=False).logits[0].float()
        with dequant_linears():
            wq_logits = model(input_ids=ctx, use_cache=False).logits[0].float()
        emb = model.get_input_embeddings()(ctx)
        floor_logits = model(inputs_embeds=nudge(emb.float(), args.seed).to(emb.dtype), use_cache=False).logits[0].float()
    kl_dev = kl_rows(cpu_logits, dev_logits)
    kl_wq = kl_rows(cpu_logits, wq_logits)
    kl_floor = kl_rows(cpu_logits, floor_logits)
    top1 = (cpu_logits.argmax(-1) == dev_logits.argmax(-1)).double().numpy()
    top1_wq = (cpu_logits.argmax(-1) == wq_logits.argmax(-1)).double().numpy()
    res = {
        "model": args.model_dir, "dtype": args.dtype, "device": tt.device().description, "prompt_tokens": T,
        "generated": args.gen, "positions": int(len(kl_dev)),
        "ref_bit_exact_steps": None if ref_equal is None else f"{sum(ref_equal)}/{len(ref_equal)}",
        "kl_dev_mean": float(kl_dev.mean()), "kl_dev_median": float(np.median(kl_dev)), "kl_dev_max": float(kl_dev.max()),
        "top1": float(top1.mean()), "top1_generated": float(top1[T - 1:].mean()),
        "kl_weights_only_mean": float(kl_wq.mean()), "top1_weights_only": float(top1_wq.mean()),
        "kl_floor_1ulp_mean": float(kl_floor.mean()),
        "max_abs_dlogit": float((dev_logits - cpu_logits).abs().max()),
        "dev_text": tok.decode(seq[0, T:]), "cpu_argmax_on_dev_ctx": tok.decode(cpu_logits[T - 1:].argmax(-1)),
        "device_seconds": round(t_dev, 1), "reference_seconds": round(t_ref, 1),
        "weights_on_device": len(tt.weight_cache()), "counters": dict(cnt),
    }
    res["pass"] = bool((ref_equal is None or all(ref_equal)) and res["top1"] >= 0.8)
    if args.gguf:
        g = gguf_compare(model, args.gguf)
        res["gguf"] = {k: v for k, v in g.items() if k != "per_tensor"}
        res["gguf_tensors"] = g["per_tensor"]
    print(json.dumps({k: v for k, v in res.items() if k not in ("counters", "gguf_tensors")}, indent=1, ensure_ascii=False))
    print("counters:\n" + tt.format_counters(cnt))
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
