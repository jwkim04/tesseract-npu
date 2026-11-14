#!/usr/bin/env python3
"""hf_check.py - end-to-end numeric check of a Hugging Face causal LM through torch.compile(backend="tesseract").

    build/tools/torchvenv/bin/python -I sw/torch/tools/hf_check.py models/hf/Qwen2.5-0.5B-Instruct \
        [--download Qwen/Qwen2.5-0.5B-Instruct] [--tokens 32] [--gen 16] [--gguf GGUF] [--json OUT]
    build/tools/torchvenv/bin/python -I sw/torch/tools/hf_check.py MODEL_DIR --forward-only --tokens 8 \
        --save-logits sim.npy                                  # then on another device: --expect sim.npy

Run it with -I: the model directory is data, and this script sets sys.path itself (sw/torch, the
directory of the extension: TSR_TORCH_EXT_DIR, else build/torch/python, and sw/tools). The model is loaded from a local directory: safetensors only,
trust_remote_code=False, float32 (a bf16 checkpoint is widened exactly). The device is TSR_DEVICE
(sim by default). With --download REPO the directory is first filled from the Hugging Face Hub with
huggingface_hub (weights as *.safetensors, config and tokenizer files only; a repository without
safetensors is refused).

Full check (one prompt; default: numcheck's fixed English sentence, truncated to --tokens):
  1. CPU eager (float32, unquantized) runs the prompt once (teacher forcing: logits at every position),
     and once more from the input embeddings, which must give the same bits (embd identity, as klcheck).
  2. Floor (docs/klcheck.md): the same CPU run with every element of the input embedding moved by
     exactly one f32 ulp, nextafterf(x, +-inf) with the direction from klcheck's splitmix64 hash of
     (seed, seq 0, position, element).
  3. Device: every nn.Linear weight is uploaded (Q8_0 by quantize_row_q8_0_ref), the compiled model
     runs the prompt twice (first call: Dynamo + AOT compilation; second call: steady state, every device
     call timed with its PERF cycles). The two must be bit-identical.
     Reported: KL(cpu || dev) per position, top-1, ratio = mean KL(cpu || dev) / mean KL(cpu || floor).
  4. Bit-exact against the reference: eager execution with every Linear on the device's reference
     model (tesseract_torch.reference_linears) must give the device's logits bit for bit.
  5. Same 8-bit operands on the CPU ("cpu-q8"): eager with every Linear computed by torch's f32 GEMM
     from the dequantized device weight blocks and the A32-quantized activations (QACT semantics, numpy
     model of sw/tools/numcheck.py). This is the CPU-backend comparison of klcheck (same quantized model,
     different accumulation order), with its own 1-ulp floor. Also "weights only": dequantized Q8_0
     weights times f32 activations.
  6. Greedy generation of --gen tokens by the CPU and by the device (full recompute per step, no KV
     cache): the first position where they differ; the device's last step is also checked bit for bit
     against the reference (4).
  7. The llama.cpp path's oracle (docs/numcheck.md B1, B1b), from build/numcheck: for --sample linears,
     the input x and output y of the actual device call are captured; the device's weight blocks must
     equal ggml's quantize_row_q8_0_ref of the weight byte for byte, and y must equal, bit for bit,
     libtsr_ref (nc_model_mm: tsr_qact + tsr_repack_row + tsr_dot_row, as numcheck checks llama.cpp
     runs) and the numpy model of the spec text. --blocks all (default) compares the blocks of every
     weight with ggml's function. --gguf: the output head's device blocks against that GGUF's Q8_0
     output.weight (llama.cpp's own file; informational).
PASS (exit 0): every Linear weight ran on the device (no CPU fallback), the device is repeatable and
bit-identical to the reference (4, 6), the llama.cpp-oracle checks (7) hold, top-1 vs CPU >= 80 %.
The KL ratios are reported, not gated. Exit 1 FAIL, 2 usage / input error.

--forward-only: one compiled forward of the first --tokens prompt tokens; --save-logits writes the f32
logits (.npy), --expect compares them bit for bit with a saved file (e.g. TSR_DEVICE=vkernel vs sim).
A second forward of the same prompt gives the steady state (no compilation, every op list cached).
--full-mode: full mode (docs/torch-backend.md): one compiled forward of the first --tokens prompt tokens
must be ONE region and ONE launch (no matmul node left), checked bit for bit against the reference executor
(TSR_TORCH_VERIFY on for this forward), then a second forward (steady state, bit-identical, one more
launch, no upload); the logits are compared with eager float32 (KL per position, top-1 >= 80 %) and with an
eager run whose attention quantizes q, k and v as q8_0 (reported, not gated). With --full-mode,
--save-logits and --expect work as with --forward-only: --expect is a check (the logits must be bit-identical
to the file; "expect" in the JSON counts the identical rows), e.g. a card against the simulator.
--full-mode --attention cpu: the same with options={"attention": "cpu"}: attention stays on the CPU in float32 and
the forward is several regions (one per attention boundary). "one region" and "one launch" become "every region ran
on the device" (no CPU fallback, every region call verified) and the steady forward launches once per region; the
matmuls between the regions stay device matmuls, so "no matmul node left" is only checked with attention on the device. The JSON
records the region and launch counts. --attention device (the default) keeps the checks above.
--speed 1,8,32: timing only. Steady-state forwards of the first N prompt tokens for each N, with the PERF
counters read after every launch and with them off.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "sw" / "torch"), os.environ.get("TSR_TORCH_EXT_DIR") or str(REPO / "build" / "torch" / "python"),
                str(REPO / "sw" / "tools")]

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import tesseract_torch as tt  # noqa: E402
import tesseract_torch.runtime as tt_rt  # noqa: E402

# numcheck's fixed English sentence (sw/tools/numcheck.cpp k_prompt; 55 tokens with the Qwen2.5 tokenizer)
PROMPT = ("The history of the printing press begins in the fifteenth century, when Johannes Gutenberg "
          "combined movable metal type, an oil-based ink and a wooden screw press, so that a small "
          "workshop could produce hundreds of identical books in the time a scribe needed to copy one.")
HF_ALLOW = ["*.safetensors", "*.safetensors.index.json", "config.json", "generation_config.json",
            "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json",
            "added_tokens.json"]
PICKLE_SUFFIXES = (".bin", ".pt", ".pth", ".pkl", ".ckpt")
CATEGORIES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head")


def die(msg: str) -> None:
    print(f"hf_check.py: error: {msg}", file=sys.stderr)
    sys.exit(2)


def log(msg: str) -> None:
    print(msg, flush=True)


# ------------------------------------------------------------------------------------------
# model files
# ------------------------------------------------------------------------------------------
def download(repo: str, dest: Path, revision: str | None) -> str:
    """Fill dest from the Hub: safetensors + config + tokenizer files only. Returns the commit."""
    from huggingface_hub import HfApi, snapshot_download  # noqa: PLC0415

    info = HfApi().model_info(repo, revision=revision)
    names = [s.rfilename for s in info.siblings]
    if not any(n.endswith(".safetensors") for n in names):
        die(f"{repo}: no *.safetensors in the repository (pickle checkpoints are not loaded)")
    t0 = time.time()
    snapshot_download(repo_id=repo, revision=info.sha, local_dir=str(dest), allow_patterns=HF_ALLOW)
    log(f"download {repo}@{info.sha[:8]} -> {dest} ({time.time() - t0:.1f} s)")
    return info.sha


def model_files(model_dir: Path) -> dict:
    """Weight files, hub revision and sha256 (checked against the hub's LFS etag when it is recorded)."""
    st = sorted(model_dir.glob("*.safetensors"))
    if not st:
        die(f"{model_dir}: no *.safetensors")
    pick = [p.name for p in model_dir.iterdir() if p.suffix in PICKLE_SUFFIXES]
    if pick:
        die(f"{model_dir}: pickle files present ({', '.join(pick)}); only safetensors directories are loaded")
    out = {"safetensors": [], "revision": None}
    meta_dir = model_dir / ".cache" / "huggingface" / "download"
    for p in st:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        rec = {"file": p.name, "bytes": p.stat().st_size, "sha256": h.hexdigest()}
        m = meta_dir / (p.name + ".metadata")
        if m.exists():
            lines = m.read_text().split("\n")
            out["revision"] = lines[0].strip() or None
            rec["hub_etag_match"] = len(lines) > 1 and lines[1].strip() == rec["sha256"]
        out["safetensors"].append(rec)
    return out


def load_model(model_dir: Path):
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False,
                                                 use_safetensors=True, dtype=torch.float32).eval()
    return tok, model


# ------------------------------------------------------------------------------------------
# klcheck's 1-ulp floor (sw/tools/klcheck.cpp splitmix64 / embd_nudge_key / embd_nudge)
# ------------------------------------------------------------------------------------------
M64 = (1 << 64) - 1


def splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & M64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & M64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & M64
    return x ^ (x >> 31)


def splitmix64_np(x: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        x = x + np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return x ^ (x >> np.uint64(31))


def nudge_key(seed: int, seq: int, pos: int) -> int:
    return splitmix64(splitmix64(splitmix64(seed) ^ seq) ^ ((pos << 1) & M64))


def nudge_embeds(e: np.ndarray, seed: int, seq: int = 0) -> np.ndarray:
    """Element j of token index pos -> nextafterf(x, -inf) if the top bit of splitmix64(key + j) is set,
    else nextafterf(x, +inf); key = nudge_key(seed, seq, pos)."""
    e = np.ascontiguousarray(e, dtype=np.float32)
    out = np.empty_like(e)
    j = np.arange(e.shape[1], dtype=np.uint64)
    up, dn = np.float32(np.inf), np.float32(-np.inf)
    for pos in range(e.shape[0]):
        with np.errstate(over="ignore"):
            down = (splitmix64_np(np.uint64(nudge_key(seed, seq, pos)) + j) >> np.uint64(63)).astype(bool)
        out[pos] = np.where(down, np.nextafter(e[pos], dn), np.nextafter(e[pos], up))
    # every element moved by exactly one ulp: one step back toward the original recovers it
    assert np.all(out != e) and np.all(np.nextafter(out, e) == e)
    return out


# ------------------------------------------------------------------------------------------
# logits comparison (klcheck's statistics, float64)
# ------------------------------------------------------------------------------------------
def compare(ref: torch.Tensor, x: torch.Tensor) -> dict:
    lr = F.log_softmax(ref.double(), -1)
    lx = F.log_softmax(x.double(), -1)
    kl = (lr.exp() * (lr - lx)).sum(-1).numpy()
    top1 = (ref.argmax(-1) == x.argmax(-1)).numpy()
    ident = (ref.contiguous().view(torch.int32) == x.contiguous().view(torch.int32)).all(-1).numpy()
    dmax = (x.double() - ref.double()).abs().max(-1).values.numpy()
    cos = F.cosine_similarity(ref.double(), x.double(), dim=-1).numpy()
    return {"positions": int(kl.size), "kl_mean": float(kl.mean()), "kl_median": float(np.median(kl)),
            "kl_max": float(kl.max()), "top1": float(top1.mean()), "top1_count": int(top1.sum()),
            "bit_identical_rows": int(ident.sum()), "max_abs_dlogit": float(dmax.max()), "min_cosine": float(cos.min()),
            "kl_per_position": [float(v) for v in kl], "top1_per_position": [bool(v) for v in top1]}


def q8_0_roundtrip(x: torch.Tensor) -> torch.Tensor:
    """x quantized as q8_0 and back: blocks of 32 values along the last dimension, d = max|x| / 127 stored as
    fp16, q = round(x / d) with ties away from zero, value q * fp16(d). The device does this to Q, K and V."""
    shape = x.shape
    b = x.float().reshape(-1, 32)
    d = b.abs().amax(-1, keepdim=True) / 127.0
    inv = torch.where(d > 0, 1.0 / d, torch.zeros_like(d))
    t = b * inv
    q = torch.sign(t) * torch.floor(t.abs() + 0.5)
    return (q * d.half().float()).reshape(shape).to(x.dtype)


class Q8Attention:
    """While active, torch.nn.functional.scaled_dot_product_attention quantizes q, k and v as q8_0 first
    (the device's attention: q8 scores, exact P.V). Used for the eager reference only."""

    def __enter__(self):
        self.orig = F.scaled_dot_product_attention
        orig = self.orig

        def sdpa(q, k, v, *a, **kw):
            assert q.shape[-1] % 32 == 0, tuple(q.shape)
            return orig(q8_0_roundtrip(q), q8_0_roundtrip(k), q8_0_roundtrip(v), *a, **kw)

        F.scaled_dot_product_attention = sdpa
        return self

    def __exit__(self, *exc):
        F.scaled_dot_product_attention = self.orig


def bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and torch.equal(a.float().contiguous().view(torch.int32), b.float().contiguous().view(torch.int32))


def ratio(a: float, b: float):
    return a / b if b > 0 else None


# ------------------------------------------------------------------------------------------
# device tap: every device call of the compiled graph (time, PERF cycles, optional capture of x, y)
# ------------------------------------------------------------------------------------------
class DeviceTap:
    def __init__(self, capture: set[int] | None = None):
        self.capture = capture or set()
        self.calls: list[dict] = []
        self.captured: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def __enter__(self):
        self.orig = tt_rt.device_linear_2d
        tt_rt.device_linear_2d = self._call
        return self

    def __exit__(self, *a):
        tt_rt.device_linear_2d = self.orig

    def _call(self, handle, x2, *, ref=False, name="", dev=None):
        if ref:
            return self.orig(handle, x2, ref=True, name=name, dev=dev)
        d = tt.device() if dev is None else dev
        c0 = d.counters()
        t0 = time.perf_counter()
        y = self.orig(handle, x2, ref=ref, name=name, dev=dev)
        dt = time.perf_counter() - t0
        c1 = d.counters()
        self.calls.append({"handle": int(handle), "T": int(y.shape[0]), "N": int(y.shape[1]), "sec": dt,
                           "cycles": c1["perf_cycles"] - c0["perf_cycles"],
                           "mvm_cycles": c1["perf_mvm_cycles"] - c0["perf_mvm_cycles"]})
        if handle in self.capture and handle not in self.captured:
            self.captured[handle] = (x2.float().contiguous().clone(), y.clone())
        return y


# ------------------------------------------------------------------------------------------
# eager references with the same 8-bit operands
# ------------------------------------------------------------------------------------------
def dequant_q8_0(blocks: np.ndarray, N: int, K: int) -> torch.Tensor:
    b = blocks.reshape(N, K // 32, 34)
    d = b[:, :, 0:2].copy().view(np.float16).astype(np.float32)
    q = b[:, :, 2:34].view(np.int8).astype(np.float32)
    return torch.from_numpy((q * d).reshape(N, K))                    # exact in f32


class q8_linears:
    """Eager nn.Linear from the device's Q8_0 blocks, dequantized (exact in f32), times either the
    A32-quantized activation (QACT semantics; also exact once dequantized) or, weights_only, the f32
    activation. Products and sums: torch's f32 GEMM. Bias added in f32."""

    def __init__(self, nc, weights_only: bool = False):
        self.nc = nc
        self.weights_only = weights_only

    def __enter__(self):
        self.orig = torch.nn.Linear.forward
        nc, wo, orig = self.nc, self.weights_only, self.orig

        def forward(mod, x):
            h = tt.weight_cache().find(mod.weight)
            if h is None:
                return orig(mod, x)
            N, K = mod.weight.shape
            w = dequant_q8_0(tt.device().read_blocks(h).numpy(), N, K)
            lead = x.shape[:-1]
            x2 = x.reshape(-1, K).float()
            if not wo:
                q, D, _ = nc.np_qact(nc.QA_A32, x2.numpy())
                xq = q.astype(np.float32).reshape(-1, K // 32, 32) * D[..., None]
                x2 = torch.from_numpy(np.ascontiguousarray(xq.reshape(-1, K), dtype=np.float32))
            y = F.linear(x2, w)
            if mod.bias is not None:
                y = y + mod.bias.float()
            return y.to(x.dtype).reshape(*lead, N)

        torch.nn.Linear.forward = forward
        return self

    def __exit__(self, *a):
        torch.nn.Linear.forward = self.orig


# ------------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------------
def linear_modules(model) -> list[tuple[str, torch.nn.Linear]]:
    return [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]


def category(name: str) -> str:
    for c in CATEGORIES:
        if name.endswith(c):
            return c
    return "other"


def upload_all(model) -> dict:
    """Upload every supported Linear weight before compiling (the compiled graph then hits the cache)."""
    t0 = time.time()
    c0 = tt.counters()
    skipped = {}
    for name, m in linear_modules(model):
        why = tt.supported_weight(m.weight)
        if why is not None:
            skipped[name] = why
            continue
        tt.weight_cache().handle(m.weight, name + ".weight")
    c1 = tt.counters()
    return {"uploads": c1["uploads"] - c0["uploads"], "upload_bytes_host": c1["upload_bytes"] - c0["upload_bytes"],
            "h2d_weight_bytes": c1.get("rt_h2d_weight_bytes", 0) - c0.get("rt_h2d_weight_bytes", 0),
            "seconds": round(time.time() - t0, 2), "not_supported": skipped}


def handle_names(model) -> dict[int, list[str]]:
    out: dict[int, list[str]] = {}
    for name, m in linear_modules(model):
        h = tt.weight_cache().find(m.weight)
        if h is not None:
            out.setdefault(int(h), []).append(name)
    return out


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def tokenize(tok, prompt: str, n: int) -> torch.Tensor:
    ids = tok(prompt, return_tensors="pt").input_ids
    if ids.shape[1] < n:
        die(f"the prompt has {ids.shape[1]} tokens, --tokens {n} needs more")
    return ids[:, :n]


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, torch.dtype):
        return str(o).replace("torch.", "")
    return o


# ------------------------------------------------------------------------------------------
# llama.cpp-path oracle checks (numcheck's native helpers)
# ------------------------------------------------------------------------------------------
def load_numcheck(build: Path):
    import numcheck as nc  # noqa: PLC0415  (sw/tools/numcheck.py)

    return nc, nc.Native(build)


def check_blocks_all(model, nat) -> dict:
    """Device Q8_0 blocks of every uploaded weight vs ggml's quantize_row_q8_0_ref, byte for byte."""
    dev = tt.device()
    t0 = time.time()
    res = {"tensors": 0, "identical": 0, "blocks": 0, "blocks_differ": 0, "differ": []}
    seen = set()
    for name, m in linear_modules(model):
        h = tt.weight_cache().find(m.weight)
        if h is None or h in seen:
            continue
        seen.add(h)
        N = m.weight.shape[0]
        ref = nat.quantize_q8_0(m.weight.detach().float().contiguous().numpy()).reshape(N, -1)
        got = dev.read_blocks(h).numpy()
        nd = int((ref.reshape(-1, 34) != got.reshape(-1, 34)).any(axis=1).sum()) if ref.shape == got.shape else ref.size // 34
        res["tensors"] += 1
        res["blocks"] += ref.size // 34
        res["blocks_differ"] += nd
        res["identical"] += nd == 0
        if nd:
            res["differ"].append({"weight": name, "blocks_differ": nd})
    res["seconds"] = round(time.time() - t0, 1)
    return res


def check_sample(nc, nat, name: str, mod, x: torch.Tensor, y: torch.Tensor, threads: int, gguf_blocks=None) -> dict:
    """One captured device call: x [T, K] f32 as the device got it, y [T, N] f32 as the device returned it."""
    dev = tt.device()
    h = tt.weight_cache().find(mod.weight)
    N, K = mod.weight.shape
    T = x.shape[0]
    t0 = time.time()
    blocks_ggml = nat.quantize_q8_0(mod.weight.detach().float().contiguous().numpy()).reshape(N, -1)
    blocks_dev = dev.read_blocks(h).numpy()
    xn = x.numpy()
    yd = y.numpy().view(np.uint32)
    y_ref = nat.model_mm(nc.FMT_Q8_0, blocks_ggml, N, K, xn, threads)        # libtsr_ref (numcheck B1)
    qa = nc.np_qact(nc.QA_A32, xn)
    y_spec = np.empty((T, N), np.float32)
    step = max(1, (1 << 21) // max(1, T * (K // 32)))                       # numpy model in row chunks
    for r0 in range(0, N, step):
        r1 = min(N, r0 + step)
        wp = nc.parse_q8_0(np.ascontiguousarray(blocks_ggml[r0:r1]), r1 - r0, K)
        y_spec[:, r0:r1] = nc.spec_model_y(nc.FMT_Q8_0, wp, qa, T)
    y_ext = dev.linear_ref(h, x).numpy()                                     # the extension's reference
    info = dev.info(h)
    out = {"linear": name, "N": int(N), "K": int(K), "T": int(T), "bias": mod.bias is not None,
           "device_parts": f"{info['n_parts']} x {info['k_parts']}",
           "blocks_equal_ggml_ref": bool(np.array_equal(blocks_dev, blocks_ggml)),
           "y_elements": int(yd.size),
           "y_mismatch_libtsr_ref": int((yd != y_ref.view(np.uint32)).sum()),
           "y_mismatch_numpy_spec": int((yd != y_spec.view(np.uint32)).sum()),
           "y_mismatch_ext_linear_ref": int((yd != y_ext.view(np.uint32)).sum())}
    if gguf_blocks is not None:
        g = np.asarray(gguf_blocks).reshape(N, -1)
        out["blocks_equal_gguf"] = bool(g.shape == blocks_dev.shape and np.array_equal(g, blocks_dev))
        y_g = nat.model_mm(nc.FMT_Q8_0, np.ascontiguousarray(g), N, K, xn, threads)
        out["y_mismatch_libtsr_ref_gguf_blocks"] = int((yd != y_g.view(np.uint32)).sum())
    out["ok"] = (out["blocks_equal_ggml_ref"] and out["y_mismatch_libtsr_ref"] == 0
                 and out["y_mismatch_numpy_spec"] == 0 and out["y_mismatch_ext_linear_ref"] == 0)
    out["seconds"] = round(time.time() - t0, 1)
    return out


def gguf_output_blocks(path: str):
    sys.path.insert(0, str(REPO / "third_party" / "llama.cpp" / "gguf-py"))
    import gguf  # noqa: PLC0415

    tensors = {t.name: t for t in gguf.GGUFReader(path).tensors}
    cand = tensors.get("output.weight") or tensors.get("token_embd.weight")   # tied models may store only the latter
    if cand is None:
        return None, "no output.weight / token_embd.weight"
    if cand.tensor_type != gguf.GGMLQuantizationType.Q8_0:
        return None, f"{cand.name} is {cand.tensor_type.name}, not Q8_0"
    return cand.data, cand.name


# ------------------------------------------------------------------------------------------
# modes
# ------------------------------------------------------------------------------------------
def save_and_expect(args, raw: np.ndarray, res: dict) -> bool:
    """--save-logits writes the f32 logits (.npy); --expect compares them bit for bit with a saved file and
    records the count of identical rows in res["expect"]. False if --expect is given and they differ."""
    if args.save_logits:
        np.save(args.save_logits, raw)
    if not args.expect:
        return True
    ref = np.load(args.expect)
    same_shape = ref.shape == raw.shape and ref.dtype == raw.dtype
    rows = int((ref.view(np.uint32) == raw.view(np.uint32)).all(-1).sum()) if same_shape else 0
    res["expect"] = {"file": args.expect, "rows": int(raw.shape[0]), "bit_identical_rows": rows,
                     "sha256_expected": hashlib.sha256(ref.tobytes()).hexdigest()}
    return same_shape and rows == raw.shape[0]


def run_forward_only(args, tok, model, res: dict) -> int:
    ids = tokenize(tok, args.prompt, args.tokens)
    res["prompt_ids"] = ids[0].tolist()
    res["upload"] = upload_all(model)
    compiled = tt.compile(model)
    tt.reset_counters()
    t0 = time.time()
    with torch.no_grad(), DeviceTap() as tap:
        lg = compiled(input_ids=ids, use_cache=False).logits[0].float().contiguous()
    wall = time.time() - t0
    c = tt.counters()
    T = ids.shape[1]
    res["forward"] = {"tokens": T, "wall_s_incl_compile": round(wall, 2), "device_calls": len(tap.calls),
                      "device_s": round(sum(r["sec"] for r in tap.calls), 2),
                      "perf_cycles": c.get("rt_perf_cycles"), "perf_cycles_per_token": c.get("rt_perf_cycles", 0) / T,
                      "perf_mvm_cycles_per_token": c.get("rt_perf_mvm_cycles", 0) / T, "execs": c.get("rt_execs"),
                      "nodes_device": c["nodes_device"], "nodes_cpu": c["nodes_cpu"],
                      "cpu_fallback_calls": c["cpu_fallback_calls"]}
    raw = lg.numpy()
    res["logits_sha256"] = hashlib.sha256(raw.tobytes()).hexdigest()
    ok = c["cpu_fallback_calls"] == 0
    # a second forward of the same prompt: no compilation, every op list in the cache (steady state)
    tt.reset_counters()
    t0 = time.time()
    with torch.no_grad(), DeviceTap() as tap2:
        lg2 = compiled(input_ids=ids, use_cache=False).logits[0].float().contiguous()
    wall2 = time.time() - t0
    c2 = tt.counters()
    res["forward_steady"] = {"wall_s": round(wall2, 3), "device_s": round(sum(r["sec"] for r in tap2.calls), 3),
                             "device_calls": len(tap2.calls), "execs": c2.get("rt_execs"),
                             "prog_hits": c2.get("rt_prog_hits"), "prog_misses": c2.get("rt_prog_misses"),
                             "perf_cycles_per_token": c2.get("rt_perf_cycles", 0) / T,
                             "h2d_bytes": c2.get("rt_h2d_bytes"), "d2h_bytes": c2.get("rt_d2h_bytes"),
                             "repeatable": bits_equal(lg, lg2)}
    ok = ok and res["forward_steady"]["repeatable"]
    ok = save_and_expect(args, raw, res) and ok
    res["pass"] = bool(ok)
    return 0 if ok else 1


def run_regions(args, tok, model, res: dict) -> int:
    """--full-mode: one forward is one region and one launch, verified against the reference executor;
    logits against eager float32."""
    ids = tokenize(tok, args.prompt, args.tokens)
    T = int(ids.shape[1])
    res["prompt_ids"] = ids[0].tolist()
    with torch.no_grad():
        ref = model(input_ids=ids, use_cache=False).logits[0].float().contiguous()
        with Q8Attention():
            ref_q8 = model(input_ids=ids, use_cache=False).logits[0].float().contiguous()
    tt.set_verify(True)
    compiled = tt.compile(model, mode="full", attention=args.attention)
    tt.reset_counters()
    t0 = time.time()
    with torch.no_grad():
        lg = compiled(input_ids=ids, use_cache=False).logits[0].float().contiguous()
    wall = time.time() - t0
    c = tt.counters()
    tt.set_verify(False)
    regs = tt.regions()
    res["forward"] = {"tokens": T, "attention": args.attention, "wall_s_incl_compile": round(wall, 2), "regions": c["regions"],
                      "region_calls": c["region_calls"], "launches": c["rt_graph_launches"],
                      "records": c["rt_graph_records"], "verified_regions": c["verified_regions"],
                      "region_cpu_fallback_calls": c["region_cpu_fallback_calls"], "matmul_nodes": c["nodes_device"],
                      "uploads": c["uploads"], "consts_uploaded": c["consts_uploaded"],
                      "perf_cycles": c.get("rt_perf_cycles"), "perf_cycles_per_token": c.get("rt_perf_cycles", 0) / T,
                      "ops": regs[-1].region.op_counts() if regs else {}}
    tt.reset_counters()
    t0 = time.time()
    with torch.no_grad():
        lg2 = compiled(input_ids=ids, use_cache=False).logits[0].float().contiguous()
    c2 = tt.counters()
    raw = lg.numpy()
    res["logits_sha256"] = hashlib.sha256(raw.tobytes()).hexdigest()
    res["forward_steady"] = {"wall_s": round(time.time() - t0, 3), "launches": c2["rt_graph_launches"],
                             "uploads": c2["uploads"] + c2["consts_uploaded"], "plan_hits": c2["rt_graph_plan_hits"],
                             "repeatable": bits_equal(lg, lg2)}
    cmp = compare(ref, lg)
    cmp_q8 = compare(ref_q8, lg)
    cmp_ref_q8 = compare(ref, ref_q8)
    res["kl"] = {"cpu_vs_device": cmp, "cpu_q8_attention_vs_device": cmp_q8, "cpu_vs_cpu_q8_attention": cmp_ref_q8}
    nreg = c["regions"]                       # 1 with attention on the device; one per attention boundary otherwise
    res["checks"] = {}
    if args.attention == "device":
        res["checks"]["one region"] = nreg == 1
        res["checks"]["one launch"] = c["region_calls"] == 1 and c["rt_graph_launches"] == 1
        res["checks"]["no matmul node left"] = c["nodes_device"] == 0 and c["device_calls"] == 0
    else:
        res["checks"]["every region ran on the device"] = (
            nreg >= 1 and c["region_calls"] == nreg and c["rt_graph_launches"] == nreg
            and c["region_cpu_fallback_calls"] == 0 and c["verified_regions"] == nreg)
    res["checks"].update({
        "no CPU fallback": c["region_cpu_fallback_calls"] == 0,
        "verified bit for bit": c["verified_regions"] == nreg,
        "steady state": res["forward_steady"]["launches"] == nreg and res["forward_steady"]["uploads"] == 0
        and res["forward_steady"]["repeatable"],
        "top-1 >= 80 %": cmp["top1"] >= 0.8,
    })
    if args.expect:
        res["checks"]["bit-identical to --expect"] = save_and_expect(args, raw, res)
    else:
        save_and_expect(args, raw, res)
    res["pass"] = all(res["checks"].values())
    log("KL(cpu || device) per position: " + " ".join(f"{v:.2e}" for v in cmp["kl_per_position"]))
    for name, c_ in (("KL(eager f32 || device)", cmp), ("KL(eager q8 attention || device)", cmp_q8),
                     ("KL(eager f32 || eager q8 attention)", cmp_ref_q8)):
        log(f"{name}: mean {c_['kl_mean']:.2e} median {c_['kl_median']:.2e} max {c_['kl_max']:.2e} top-1 {c_['top1']:.3f}")
    return 0 if res["pass"] else 1


def run_speed(args, tok, model, res: dict) -> int:
    """--speed 1,8,32: steady-state compiled forwards of the first N prompt tokens, for each N in the list, with the
    PERF counters read after every launch (the default) and again with them off (`open_device(read_perf=False)`: the 12
    CSR reads per launch are not made). Each N is run once to compile and to cache its op lists, then timed 3 times;
    the fastest is reported. Nothing is compared here: the bits are checked by the other modes."""
    lens = [int(v) for v in args.speed.split(",") if v.strip()]
    ids_all = tokenize(tok, args.prompt, max(lens))
    res["prompt_ids"] = ids_all[0].tolist()
    runs = []
    for read_perf in (True, False):
        torch._dynamo.reset()
        tt.close_device()
        tt.open_device(read_perf=read_perf)
        up = upload_all(model)
        compiled = tt.compile(model)
        rows = []
        for n in lens:
            ids = ids_all[:, :n]
            with torch.no_grad():
                compiled(input_ids=ids, use_cache=False)                   # compile (new length) and fill the op-list cache
                best = None
                for _ in range(3):
                    tt.reset_counters()
                    with DeviceTap() as tap:
                        t0 = time.time()
                        compiled(input_ids=ids, use_cache=False)
                        wall = time.time() - t0
                    c = tt.counters()
                    row = {"tokens": n, "wall_s": round(wall, 4), "device_s": round(sum(r["sec"] for r in tap.calls), 4),
                           "device_calls": len(tap.calls), "execs": c.get("rt_execs"), "prog_hits": c.get("rt_prog_hits"),
                           "prog_misses": c.get("rt_prog_misses"), "h2d_bytes": c.get("rt_h2d_bytes"),
                           "d2h_bytes": c.get("rt_d2h_bytes"), "cpu_fallback_calls": c["cpu_fallback_calls"]}
                    if read_perf:
                        row.update({"perf_cycles": c.get("rt_perf_cycles"), "perf_cycles_per_token": c.get("rt_perf_cycles", 0) / n,
                                    "perf_mvm_cycles_per_token": c.get("rt_perf_mvm_cycles", 0) / n,
                                    "dev_read_bytes_per_token": c.get("rt_dev_read_bytes", 0) / n})
                    if best is None or row["wall_s"] < best["wall_s"]:
                        best = row
            best["ms_per_token"] = round(1000 * best["wall_s"] / n, 3)
            best["device_ms_per_token"] = round(1000 * best["device_s"] / n, 3)
            rows.append(best)
            log(f"speed read_perf={read_perf} T={n}: wall {best['wall_s']} s ({best['ms_per_token']} ms/token), device calls "
                f"{best['device_s']} s, {best['execs']} execs" + (f", {best['perf_cycles_per_token']:.0f} PERF cycles/token" if read_perf else ""))
        runs.append({"read_perf": read_perf, "upload_s": up["seconds"], "rows": rows})
    res["speed"] = runs
    res["pass"] = all(r["cpu_fallback_calls"] == 0 and r["device_calls"] == 169 for run in runs for r in run["rows"])
    return 0 if res["pass"] else 1


def run_full(args, tok, model, res: dict) -> int:
    nc, nat = load_numcheck(Path(args.numcheck_build))
    threads = args.threads or min(8, os.cpu_count() or 1)
    ids = tokenize(tok, args.prompt, args.tokens)
    T = ids.shape[1]
    res["prompt_ids"] = ids[0].tolist()
    res["prompt_text"] = tok.decode(ids[0])
    times = res.setdefault("seconds", {})

    # 1-2. CPU eager, embd identity, floor
    emb_mod = model.get_input_embeddings()
    with torch.no_grad():
        t0 = time.time()
        cpu = model(input_ids=ids, use_cache=False).logits[0].float().contiguous()
        times["cpu_forward"] = round(time.time() - t0, 2)
        emb = emb_mod(ids)
        cpu_embd = model(inputs_embeds=emb, use_cache=False).logits[0].float()
        emb_n = torch.from_numpy(nudge_embeds(emb[0].numpy(), args.seed))[None]
        floor = model(inputs_embeds=emb_n, use_cache=False).logits[0].float()
    res["embd_identity"] = bits_equal(cpu, cpu_embd)
    log(f"cpu: forward of {T} tokens {times['cpu_forward']} s; embd path identical: {res['embd_identity']}")

    # 3. device
    res["upload"] = upload_all(model)
    log(f"upload: {res['upload']['uploads']} weights, {res['upload']['h2d_weight_bytes'] / 2**20:.1f} MiB to the device "
        f"in {res['upload']['seconds']} s")
    compiled = tt.compile(model)
    hn = handle_names(model)
    samples = {}
    for s in args.sample.split(","):
        s = s.strip()
        mod = dict(linear_modules(model)).get(s)
        if mod is None:
            die(f"--sample {s}: no such nn.Linear (e.g. model.layers.0.self_attn.q_proj, lm_head)")
        h = tt.weight_cache().find(mod.weight)
        if h is None:
            die(f"--sample {s}: weight not on the device ({tt.supported_weight(mod.weight)})")
        samples[s] = (int(h), mod)
    up0 = tt.counters()["uploads"]
    with torch.no_grad():
        t0 = time.time()
        dev1 = compiled(input_ids=ids, use_cache=False).logits[0].float().contiguous()
        times["device_first_call_incl_compile"] = round(time.time() - t0, 2)
        c_first = tt.counters()             # the first call compiled (and partitioned) the graph
        up_first = c_first["uploads"] - up0
        tt.reset_counters()
        with DeviceTap({h for h, _ in samples.values()}) as tap:
            t0 = time.time()
            dev = compiled(input_ids=ids, use_cache=False).logits[0].float().contiguous()
            times["device_forward"] = round(time.time() - t0, 2)
        c_fw = tt.counters()
    res["device_repeatable"] = bits_equal(dev1, dev)
    log(f"device: first call {times['device_first_call_incl_compile']} s (compile + run), forward {times['device_forward']} s; "
        f"repeatable: {res['device_repeatable']}")

    # partition / offload report from the tapped forward
    calls_per_handle: dict[int, int] = {}
    for r in tap.calls:
        calls_per_handle[r["handle"]] = calls_per_handle.get(r["handle"], 0) + 1
    by_cat: dict[str, dict] = {}
    for h, names in hn.items():
        for n in names:
            cat = category(n)
            e = by_cat.setdefault(cat, {"linears": 0, "on_device": 0})
            e["linears"] += 1
            e["on_device"] += calls_per_handle.get(h, 0) >= 1
    n_linear = len(linear_modules(model))
    lm = getattr(model, "lm_head", None)
    tied = lm is not None and lm.weight.data_ptr() == emb_mod.weight.data_ptr()
    lm_h = tt.weight_cache().find(lm.weight) if lm is not None else None
    cyc_cat: dict[str, dict] = {}
    for r in tap.calls:
        cat = category(hn.get(r["handle"], ["?"])[0])
        e = cyc_cat.setdefault(cat, {"calls": 0, "cycles": 0, "mvm_cycles": 0, "sec": 0.0})
        e["calls"] += 1
        e["cycles"] += r["cycles"]
        e["mvm_cycles"] += r["mvm_cycles"]
        e["sec"] += r["sec"]
    for e in cyc_cat.values():
        e["cycles_per_token"] = e["cycles"] / T
        e["sec"] = round(e["sec"], 3)
    dev_sec = sum(r["sec"] for r in tap.calls)
    res["offload"] = {
        "nn_linear_modules": n_linear, "device_weights": len(hn),
        "device_calls_per_forward": len(tap.calls), "distinct_weights_called": len(calls_per_handle),
        "every_linear_on_device_once": all(calls_per_handle.get(h, 0) == 1 for h in hn) and len(calls_per_handle) == len(hn)
                                      and sum(len(v) for v in hn.values()) == n_linear,
        "by_category": by_cat, "graph0_matmul_nodes_device": c_first["nodes_device"],
        "graph0_matmul_nodes_cpu": c_first["nodes_cpu"], "graphs": None, "cpu_fallback_calls": c_fw["cpu_fallback_calls"],
        "lm_head_tied_to_embed_tokens": tied,
        "lm_head_on_device": lm_h is not None and calls_per_handle.get(int(lm_h), 0) == 1,
        "lm_head_shares_embed_upload": lm_h is not None and lm_h == tt.weight_cache().find(emb_mod.weight),
    }
    res["perf"] = {
        "tokens": T, "perf_cycles": c_fw.get("rt_perf_cycles"), "perf_cycles_per_token": c_fw.get("rt_perf_cycles", 0) / T,
        "perf_mvm_cycles_per_token": c_fw.get("rt_perf_mvm_cycles", 0) / T,
        "dev_read_bytes_per_token": c_fw.get("rt_dev_read_bytes", 0) / T, "execs": c_fw.get("rt_execs"),
        "prog_hits": c_fw.get("rt_prog_hits"), "prog_misses": c_fw.get("rt_prog_misses"),
        "h2d_bytes": c_fw.get("rt_h2d_bytes"), "d2h_bytes": c_fw.get("rt_d2h_bytes"),
        "by_category": cyc_cat, "device_call_s": round(dev_sec, 2), "cpu_ops_s": round(times["device_forward"] - dev_sec, 2),
        "device_forward_s_per_token": times["device_forward"] / T,
    }

    # 6. greedy generation (full recompute per step), CPU and device
    def greedy(fn, last_row: torch.Tensor):
        toks = [int(last_row.argmax())]
        seq, lg = ids, None
        for _ in range(1, args.gen):
            seq = torch.cat([seq, torch.tensor([[toks[-1]]], dtype=ids.dtype)], dim=1)
            lg = fn(seq)
            toks.append(int(lg[-1].argmax()))
        return toks, seq, lg

    with torch.no_grad():
        t0 = time.time()
        g_cpu, _, _ = greedy(lambda s: model(input_ids=s, use_cache=False).logits[0].float(), cpu[-1])
        times["cpu_generate"] = round(time.time() - t0, 2)
        t0 = time.time()
        g_dev, last_in, last_lg = greedy(lambda s: compiled(input_ids=s, use_cache=False).logits[0].float().contiguous(), dev[-1])
        times["device_generate"] = round(time.time() - t0, 2)
    div = next((i for i, (a, b) in enumerate(zip(g_cpu, g_dev)) if a != b), None)
    gen_tokens_processed = sum(range(T + 1, T + args.gen)) if args.gen > 1 else 0
    res["generate"] = {"n": args.gen, "cpu_tokens": g_cpu, "device_tokens": g_dev, "cpu_text": tok.decode(g_cpu),
                       "device_text": tok.decode(g_dev), "identical": g_cpu == g_dev,
                       "first_divergence": div, "first_divergence_abs_position": None if div is None else T + div,
                       "device_forwards": args.gen - 1, "device_token_passes": gen_tokens_processed}
    log(f"generate {args.gen}: cpu {times['cpu_generate']} s, device {times['device_generate']} s; "
        f"first divergence: {div}")
    c_all = tt.counters()
    res["offload"]["graphs"] = c_first["graphs"] + c_all["graphs"]       # recompiles (dynamic length) included
    res["offload"]["uploads_in_compiled_calls"] = up_first + c_all["uploads"]   # 0: every call hit the cache
    res["offload"]["cpu_fallback_calls"] = c_first["cpu_fallback_calls"] + c_all["cpu_fallback_calls"]

    # 4. bit-exact vs eager with reference linears (teacher-forced forward and the last generation step)
    ref_ok = {}
    if not args.no_ref_check:
        with torch.no_grad(), tt.reference_linears():
            t0 = time.time()
            ref_tf = model(input_ids=ids, use_cache=False).logits[0].float()
            times["reference_forward"] = round(time.time() - t0, 2)
            ref_ok["teacher_forced"] = bits_equal(ref_tf, dev)
            if last_lg is not None:
                ref_ok["last_generate_step"] = bits_equal(model(input_ids=last_in, use_cache=False).logits[0].float(), last_lg)
        log(f"reference linears: {ref_ok}")
    res["ref_bit_exact"] = ref_ok

    # 7. llama.cpp-path oracle
    gblocks = None
    if args.gguf:
        gb, gname = gguf_output_blocks(args.gguf)
        res["gguf"] = {"file": args.gguf, "tensor": gname}
        gblocks = gb
    res["samples"] = []
    for s, (h, mod) in samples.items():
        if h not in tap.captured:
            res["samples"].append({"linear": s, "ok": False, "error": "not called on the device"})
            continue
        x, y = tap.captured[h]
        r = check_sample(nc, nat, s, mod, x, y, threads, gblocks if (s == "lm_head" and gblocks is not None) else None)
        res["samples"].append(r)
        log(f"sample {s} [N={r['N']} K={r['K']} T={r['T']}]: blocks == ggml ref {r['blocks_equal_ggml_ref']}, "
            f"y mismatches libtsr_ref {r['y_mismatch_libtsr_ref']} / numpy spec {r['y_mismatch_numpy_spec']} / "
            f"ext {r['y_mismatch_ext_linear_ref']} of {r['y_elements']}"
            + (f", blocks == GGUF {r['blocks_equal_gguf']}" if "blocks_equal_gguf" in r else ""))
    if args.blocks == "all":
        res["blocks_all"] = check_blocks_all(model, nat)
        b = res["blocks_all"]
        log(f"blocks: {b['identical']}/{b['tensors']} weights byte-identical to ggml quantize_row_q8_0_ref "
            f"({b['blocks_differ']} of {b['blocks']} blocks differ, {b['seconds']} s)")

    # 5. cpu-q8 and weights-only references
    with torch.no_grad():
        t0 = time.time()
        with q8_linears(nc):
            cpuq = model(input_ids=ids, use_cache=False).logits[0].float()
            cpuq_floor = model(inputs_embeds=emb_n, use_cache=False).logits[0].float()
        with q8_linears(nc, weights_only=True):
            wq = model(input_ids=ids, use_cache=False).logits[0].float()
        times["q8_references"] = round(time.time() - t0, 2)

    # comparisons
    k_dev = compare(cpu, dev)
    k_floor = compare(cpu, floor)
    k_q8 = compare(cpu, cpuq)
    k_wq = compare(cpu, wq)
    kq_dev = compare(cpuq, dev)
    kq_floor = compare(cpuq, cpuq_floor)
    res["kl"] = {
        "cpu_vs_device": k_dev, "cpu_vs_floor": k_floor,
        "ratio_device_over_floor": ratio(k_dev["kl_mean"], k_floor["kl_mean"]),
        "cpu_vs_cpu_q8": k_q8, "cpu_vs_weights_only_q8": k_wq,
        "cpu_q8_vs_device": kq_dev, "cpu_q8_vs_its_floor": kq_floor,
        "ratio_q8_device_over_floor": ratio(kq_dev["kl_mean"], kq_floor["kl_mean"]),
    }
    # positions where the floor moved at all (numcheck: the 8-bit activation quantization can absorb the
    # 1-ulp nudge, so the floor KL is exactly 0 there)
    for key, d, f in (("", k_dev, k_floor), ("q8_", kq_dev, kq_floor)):
        moved = [i for i, v in enumerate(f["kl_per_position"]) if v > 0]
        res["kl"][f"{key}floor_moved_positions"] = len(moved)
        if moved:
            md = float(np.mean([d["kl_per_position"][i] for i in moved]))
            mf = float(np.mean([f["kl_per_position"][i] for i in moved]))
            res["kl"][f"{key}ratio_floor_moved"] = ratio(md, mf)
    checks = {
        "every_linear_on_device": bool(res["offload"]["every_linear_on_device_once"] and res["offload"]["lm_head_on_device"]
                                       and res["offload"]["cpu_fallback_calls"] == 0),
        "device_repeatable": res["device_repeatable"],
        "ref_bit_exact": bool(args.no_ref_check or (ref_ok and all(ref_ok.values()))),
        "samples_llama_oracle": all(r.get("ok") for r in res["samples"]),
        "blocks_all": args.blocks != "all" or res["blocks_all"]["identical"] == res["blocks_all"]["tensors"],
        "top1_ge_80": k_dev["top1"] >= 0.8,
    }
    res["checks"] = checks
    res["pass"] = all(checks.values())
    return 0 if res["pass"] else 1


def summary(res: dict) -> str:
    L = []
    if "kl" in res:
        k = res["kl"]
        for key, label in (("cpu_vs_device", "KL(cpu || device)"), ("cpu_vs_floor", "KL(cpu || cpu+1ulp)"),
                           ("cpu_vs_cpu_q8", "KL(cpu || cpu-q8)"), ("cpu_vs_weights_only_q8", "KL(cpu || weights-only q8)"),
                           ("cpu_q8_vs_device", "KL(cpu-q8 || device)"), ("cpu_q8_vs_its_floor", "KL(cpu-q8 || cpu-q8+1ulp)")):
            if key not in k:
                continue
            s = k[key]
            L.append(f"  {label:<28} mean {s['kl_mean']:.4e}  median {s['kl_median']:.4e}  max {s['kl_max']:.4e}  "
                     f"top-1 {s['top1_count']}/{s['positions']}  identical rows {s['bit_identical_rows']}  "
                     f"max|dlogit| {s['max_abs_dlogit']:.4g}")
        for lab, rk, mk, pk in (("cpu f32", "ratio_device_over_floor", "ratio_floor_moved", "floor_moved_positions"),
                                ("cpu-q8 ", "ratio_q8_device_over_floor", "q8_ratio_floor_moved", "q8_floor_moved_positions")):
            if rk not in k:
                continue
            r, rm = k[rk], k.get(mk)
            L.append(f"  ratio device/floor ({lab}): " + (f"{r:.4g}" if r is not None else "floor 0")
                     + f"; over the {k[pk]} positions where the floor moved: " + (f"{rm:.4g}" if rm is not None else "-"))
    if "forward" in res and "regions" in res["forward"]:
        f = res["forward"]
        L.append(f"  forward T={f['tokens']}: {f['regions']} region, {f['launches']} launch(es), {f['records']} records, "
                 f"{f['perf_cycles_per_token']:.0f} PERF cycles / token, ops {f['ops']}")
    if "perf" in res:
        p = res["perf"]
        L.append(f"  PERF cycles / token {p['perf_cycles_per_token']:.0f} (MVM {p['perf_mvm_cycles_per_token']:.0f}), "
                 f"{p['execs']} execs, device calls {p['device_call_s']} s + CPU ops {p['cpu_ops_s']} s per forward")
    if "checks" in res:
        L.append("  checks: " + ", ".join(f"{k} {'OK' if v else 'FAIL'}" for k, v in res["checks"].items()))
    L.append(f"  {'PASS' if res.get('pass') else 'FAIL'}")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir")
    ap.add_argument("--download", metavar="REPO", help="fill MODEL_DIR from the Hub first (safetensors, config, tokenizer)")
    ap.add_argument("--revision", help="hub revision for --download")
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--tokens", type=int, default=32, help="prompt tokens (teacher-forced positions)")
    ap.add_argument("--gen", type=int, default=16, help="greedy tokens generated by CPU and device")
    ap.add_argument("--seed", type=int, default=1, help="floor nudge seed (klcheck --perturb-ulp)")
    ap.add_argument("--sample", default="model.layers.0.self_attn.q_proj,model.layers.12.mlp.down_proj,lm_head",
                    help="linears checked against the llama.cpp path's oracle")
    ap.add_argument("--blocks", choices=["all", "sample"], default="all", help="weights whose blocks are compared with ggml")
    ap.add_argument("--gguf", help="GGUF whose Q8_0 output.weight is compared with the device's head blocks")
    nc_default = REPO / "build" / "numcheck"
    if not nc_default.is_dir():
        nc_default = REPO / "build" / "dev"
    ap.add_argument("--numcheck-build", default=str(nc_default),
                    help="build tree with libtsr_ref.so + libnumcheck_helper.so (default build/numcheck, else build/dev)")
    ap.add_argument("--threads", type=int, default=0, help="torch / oracle threads (default: torch's, oracle 8)")
    ap.add_argument("--no-ref-check", action="store_true", help="skip the bit-exact reference forwards")
    ap.add_argument("--forward-only", action="store_true", help="one compiled forward; --save-logits / --expect")
    ap.add_argument("--full-mode", action="store_true", help="full mode: one forward = one region = one launch, verified; "
                    "KL and top-1 against eager f32")
    ap.add_argument("--attention", choices=["device", "cpu"], default="device",
                    help="(--full-mode) where attention runs: device (q8_0, one region, one launch) or cpu (float32; "
                    "several regions, every one on the device)")
    ap.add_argument("--speed", metavar="N,N,...", help="timing mode: steady-state forwards of the first N prompt tokens "
                    "(e.g. 1,8,32), PERF counters on and off; 169 device calls each, no comparison")
    ap.add_argument("--save-logits", help="(--forward-only, --full-mode) write the logits as .npy")
    ap.add_argument("--expect", help="(--forward-only, --full-mode) logits .npy that must be bit-identical")
    ap.add_argument("--json", help="write the results here")
    args = ap.parse_args()
    if args.tokens < 1 or args.gen < 1:
        die("--tokens and --gen must be >= 1")
    # the other modes tap the per-matmul device calls: they test matmul mode
    os.environ["TSR_TORCH_MODE"] = "full" if args.full_mode else "matmul"
    if args.threads:
        torch.set_num_threads(args.threads)

    model_dir = Path(args.model_dir)
    t_start = time.time()
    res: dict = {"model_dir": str(model_dir), "mode": "speed" if args.speed else ("forward-only" if args.forward_only else ("full-mode" if args.full_mode else "full"))}
    if args.download:
        model_dir.mkdir(parents=True, exist_ok=True)
        res["downloaded"] = download(args.download, model_dir, args.revision)
    res["files"] = model_files(model_dir)
    t0 = time.time()
    tok, model = load_model(model_dir)
    res["load_s"] = round(time.time() - t0, 2)
    res["params_M"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 1)
    res["device"] = tt.device().description
    res["torch"] = torch.__version__
    res["torch_threads"] = torch.get_num_threads()
    import transformers  # noqa: PLC0415

    res["transformers"] = transformers.__version__
    log(f"{model_dir}: {res['params_M']} M params (float32), loaded in {res['load_s']} s; {res['device']}")
    if args.speed:
        rc = run_speed(args, tok, model, res)
    else:
        rc = (run_regions if args.full_mode else run_forward_only if args.forward_only else run_full)(args, tok, model, res)
    res["wall_s"] = round(time.time() - t_start, 1)
    res["peak_rss_gb"] = round(rss_gb(), 2)
    res["counters_end"] = tt.counters()
    out = jsonable(res)
    brief = {k: v for k, v in out.items() if k not in ("kl", "counters_end", "prompt_ids")}
    log(json.dumps(brief, indent=1, ensure_ascii=False))
    log(summary(res))
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    return rc


if __name__ == "__main__":
    sys.exit(main())
