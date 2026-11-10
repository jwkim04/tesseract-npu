#!/usr/bin/env python3
"""Tiny synthetic Qwen2 and Gemma 4 GGUFs for the M7 graph tests (SW-21).

docs/design/06-software-verification.md §10.2 (ggml_tesseract_graph_qwen_tiny / _gemma_tiny), §13
SW-21: the smallest real dense Gemma 4 is 12B, too slow for ctest on the simulator, so the
build generates small models with random weights that keep the structure llama.cpp builds its
graphs from (SW §8 "Ops in the two model families"):

  qwen2-tiny-q8_0.gguf    2 layers; Q/K/V with bias, GQA 4:2, head_dim 64, SWIGLU, separate head
  qwen2-tiny-q4_k.gguf    the same weights, every 2D weight Q4_K (the device's native K-quant;
                          no Q6_K as in a real Q4_K_M: the backend converts Q6_K to Q8_0 at load,
                          so the device would compute with other weights than the CPU, a weight
                          difference that a tiny model's 1-ulp floor cannot absorb, see
                          tiny.cmake)
  gemma4-tiny-q8_0.gguf   4 layers, 3 SWA + 1 full (the 12B's 5:1 pattern with its last layer
                          full); SWA head_dim 64 with GQA 16:8, full head_dim 128 with 16:1 (the
                          12B's G = 2 / 16), no attn_v on the full layer (V = K projection),
                          per-head Q/K norms, rope_freqs on the full layer (25 % rotated, the rest
                          1e30 as in the 12B), layer_output_scale, GEGLU, tied output head,
                          final logit softcap 30, embedding x sqrt(n_embd)
  qwen35-tiny-q8_0.gguf   4 layers, 3 Gated DeltaNet + 1 full attention (Qwen3.5 / 3.8's 3 : 1 pattern):
                          S_v = 32, 4 key heads for 16 value heads (a 4x tiling like the 27B's 16 : 48), conv kernel 4,
                          the full layer with the joint Q|gate projection, per-head Q/K norms and partial
                          IMROPE (32 of 64 dims, sections 8 / 4 / 4 / 0)
  qwen35-tiny-mtp-q8_0.gguf, qwen35-tiny-s128-mtp-q8_0.gguf
                          the two models above with the multi-token prediction block after the layers (llama.cpp's
                          nextn layer: eh_proj, enorm, hnorm, shared_head_norm and a full attention layer with its FFN;
                          the head is the model's own output matrix), as the real Qwen3.8 GGUFs carry it (SW-83)
  qwen3moe-tiny-q8_0.gguf, qwen3moe-tiny-q4_k.gguf
                          2 layers of Qwen3-30B-A3B's kind (llama.cpp "qwen3moe"): head_dim 128 with GQA 16:2 (G = 8) and a
                          Q wider than the hidden size, per-head Q/K norms, NEOX rope, a softmax router over 64 experts
                          (F32 [E, 64]), 8 used with renormalized weights, separate gate / up / down expert tensors
                          (3D, Q8_0 or Q4_K), SWIGLU, no shared expert and no per-expert scale
  tiny-prompts.txt        prompts in the vocabularies' alphabet (klcheck --prompts)

Every model has a 64-row vocabulary with valid tokenizer metadata: Qwen2 a GPT-2 byte-level BPE
(tokenizer.ggml.model "gpt2", pre "qwen2", ChatML specials), Gemma 4 the SentencePiece-style BPE
("gemma4": "▁" for spaces, <pad>/<eos>/<bos>/<unk>/<mask>). Characters outside the alphabet are
dropped by llama.cpp's BPE, so any prompt tokenizes.

Weights are drawn from numpy's legacy RandomState (MT19937, frozen by NEP 19, unlike Generator
streams) with a fixed seed per model: the files are identical on every run, host and numpy version.
Scales are chosen so that every sub-layer adds O(1) to the residual and the head's logits spread
over a few units (top-1 well separated from top-2 most of the time).

Q8_0 uses gguf-py's quantizer. gguf-py has no Q4_K quantizer, so the encoder below is written
from the block layout (ggml-common.h) and checked against gguf-py's dequantizer on every run: the
files are valid for any reader whatever the encoder's rounding choices.

Usage: gen_tiny_models.py --out-dir DIR [--seed N] [--only NAME ...] [--check]
       (PYTHONPATH must reach third_party/llama.cpp/gguf-py; numpy needed)
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import zlib
from pathlib import Path

import numpy as np

try:
    import gguf
    from gguf import GGMLQuantizationType as QT
    from gguf import GGUFValueType as VT
except ImportError:  # the build passes PYTHONPATH; a manual run may not
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "third_party" / "llama.cpp" / "gguf-py"))
    import gguf
    from gguf import GGMLQuantizationType as QT
    from gguf import GGUFValueType as VT

QK_K = 256

# ---------------------------------------------------------------------------------------------
# Q4_K encoder (block layout of ggml-common.h; validated against gguf.quants.dequantize)


def _f16(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float16)


def quantize_q4_k(x: np.ndarray) -> np.ndarray:
    """f32 rows (ne0 % 256 == 0) -> block_q4_K bytes: d, dmin (f16), scales[12], qs[128].

    Per 32-value sub-block j: x ~ d*sc[j]*q - dmin*m[j], q in [0, 15], sc, m 6-bit."""
    rows, n = x.shape
    nb = rows * n // QK_K
    b = x.reshape(nb, 8, 32).astype(np.float32)
    lo = np.minimum(b.min(axis=2), 0.0)                     # m_j = -lo >= 0
    hi = b.max(axis=2)
    step = (hi - lo) / 15.0                                  # sub-block scale
    mins = -lo
    d = _f16(step.max(axis=1) / 63.0)
    dmin = _f16(mins.max(axis=1) / 63.0)
    d32, dmin32 = d.astype(np.float32), dmin.astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        sc = np.where(d32[:, None] > 0, np.rint(step / d32[:, None]), 0).clip(0, 63).astype(np.uint8)
        m = np.where(dmin32[:, None] > 0, np.rint(mins / dmin32[:, None]), 0).clip(0, 63).astype(np.uint8)
        s = d32[:, None] * sc.astype(np.float32)             # effective step per sub-block
        off = dmin32[:, None] * m.astype(np.float32)
        q = np.where(s[:, :, None] > 0, np.rint((b + off[:, :, None]) / s[:, :, None]), 0).clip(0, 15).astype(np.uint8)
    # scales: bytes 0-3 sc[0:4] | (sc[4:8] >> 4) << 6; bytes 4-7 m[0:4] | (m[4:8] >> 4) << 6;
    #         bytes 8-11 (sc[4:8] & 15) | (m[4:8] & 15) << 4   (get_scale_min_k4)
    scales = np.empty((nb, 12), np.uint8)
    scales[:, 0:4] = (sc[:, 0:4] & 63) | ((sc[:, 4:8] >> 4) << 6)
    scales[:, 4:8] = (m[:, 0:4] & 63) | ((m[:, 4:8] >> 4) << 6)
    scales[:, 8:12] = (sc[:, 4:8] & 15) | ((m[:, 4:8] & 15) << 4)
    # qs: 4 chunks of 32 bytes; byte l of chunk c = sub-block 2c value l (low) | 2c+1 value l (high)
    q = q.reshape(nb, 4, 2, 32)
    qs = (q[:, :, 0, :] | (q[:, :, 1, :] << 4)).reshape(nb, 128)
    blk = np.concatenate([d.view(np.uint8).reshape(nb, 2), dmin.view(np.uint8).reshape(nb, 2), scales, qs], axis=1)
    return blk.reshape(rows, n // QK_K * 144)


def encode(name: str, w: np.ndarray, qtype: QT) -> tuple[np.ndarray, float]:
    """Quantized bytes and the max |dequantized - w| (checked against the type's step)."""
    w2 = w.reshape(-1, w.shape[-1]).astype(np.float32)
    if qtype == QT.F32:
        return w.astype(np.float32), 0.0
    if qtype == QT.Q8_0:
        q = gguf.quants.quantize(w2, QT.Q8_0)
    elif qtype == QT.Q4_K:
        q = quantize_q4_k(w2)
    else:
        raise ValueError(f"{name}: no encoder for {qtype.name}")
    back = gguf.quants.dequantize(q, qtype).reshape(w2.shape)
    err = float(np.abs(back - w2).max())
    # bound: Q8_0 half a step (amax/254) at most, Q4_K one step of the widest sub-block
    # (2*amax/15) at most, which covers the half step plus the 6-bit scale and min rounding
    amax = float(np.abs(w2).max())
    bound = {QT.Q8_0: amax / 127.0, QT.Q4_K: 2.0 * amax / 15.0 * 1.05}[qtype]
    if not np.isfinite(back).all() or err > bound:
        raise RuntimeError(f"{name}: {qtype.name} round trip error {err:.3g} > bound {bound:.3g}")
    return q.reshape(*w.shape[:-1], q.shape[-1]), err


# ---------------------------------------------------------------------------------------------
# vocabularies (64 rows each)

LETTERS = [chr(c) for c in range(ord("a"), ord("z") + 1)]

# merges shared by both vocabularies; "_" stands for the space symbol of each tokenizer
MERGES = [("_", "t"), ("h", "e"), ("_t", "he"), ("i", "n"), ("e", "r"), ("a", "n"), ("o", "n"),
          ("_", "a"), ("r", "e"), ("_", "s"), ("e", "n"), ("a", "t"), ("o", "r"), ("e", "s"),
          ("_", "w"), ("_", "o"), ("i", "t"), ("a", "l"), ("_", "c"), ("i", "s"), ("_", "f"),
          ("a", "r"), ("_", "b"), ("o", "u"), ("_", "m"), ("l", "e"), ("_", "p"), ("in", "g"),
          ("_", "d"), ("_", "l"), ("_", "r")]

# The graph tests use the first four (klcheck --n-prompts 4, -ub 16, 16 decode steps each). The two
# long ones are 251 / 256 (Qwen2 / Gemma 4) and 334 / 343 tokens: the first fills the 256-cell KV
# bucket and crosses it while decoding (a new decode graph and op list for n_kv 512, SW §8 R7), the
# second crosses it in prefill; with Gemma 4's SWA ring of pad(16 + 16, 256) = 256 cells (klcheck's
# default --swa-full off) both wrap the ring, in decode and in prefill (two-run and holed SWA masks).
PROMPTS = [
    "the cat sat on the mat and the dog ran in the park, while the bird sang a little song.",
    "the river starts high in the hills where the snow melts in the spring, and it runs down through the "
    "forest past the old mill and the stone bridge. in the valley the farmers use its water for the fields "
    "of wheat and corn, and the children swim in the deep pools on the long summer days. further down the "
    "river grows wide and slow, and the boats carry wood and grain to the town.",
    "once upon a time there was a small robot who wanted to learn about the stars and the sea.",
    "at the market the bakers sell warm bread and the fishers sell their catch from the morning, while the "
    "musicians play on the steps of the old hall and the people stop to listen. when the sun goes down the "
    "lamps are lit one by one, the shops close their doors, and the families walk home along the river, "
    "talking about the day and the plans they have for the next one, and the long winter to come. later the "
    "stars came out over the hills and the wind turned cold, so they lit the fire and told old stories until "
    "late at night.",
    "water flows down the river to the ocean, where the waves meet the shore at the end of the day.",
    "a farmer planted seeds in the spring, and in the autumn the fields were full of golden wheat.",
    "in the city the trains run all night, and the lights of the towers shine over the quiet streets.",
    "she wrote a letter to her friend about the mountains, the snow and the long walk to the lake.",
    "the old clock in the hall struck nine, and the children ran up the stairs to their rooms.",
    "under the bridge a small boat waited for the tide, its sail folded and its rope tied to a post.",
]


def _expand(sym: str, space: str) -> str:
    return sym.replace("_", space)


def vocab_qwen2() -> dict:
    """GPT-2 byte-level BPE: Ġ = space (0x20), Ċ = newline; ChatML specials at the end."""
    sp, nl = "Ġ", "Ċ"
    base = LETTERS + [sp, ".", ",", nl]
    merges = [(_expand(a, sp), _expand(b, sp)) for a, b in MERGES]
    toks = base + [a + b for a, b in merges]
    specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    toks += specials
    assert len(toks) == 64 and len(set(toks)) == 64, len(toks)
    types = [1] * (64 - len(specials)) + [3] * len(specials)     # NORMAL, CONTROL
    eot = toks.index("<|endoftext|>")
    return {
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.pre": "qwen2",
        "tokens": toks,
        "types": types,
        "merges": [f"{a} {b}" for a, b in merges],
        "scores": None,
        "ids": {"bos": eot, "eos": toks.index("<|im_end|>"), "padding": eot},
        "add_bos": False,
    }


def vocab_gemma4() -> dict:
    """Gemma 4's SentencePiece-style BPE (no byte encoding, ▁ = space), its control tokens first."""
    sp = "▁"
    specials = ["<pad>", "<eos>", "<bos>", "<unk>", "<mask>"]
    base = ["\n", sp] + LETTERS + [".", ","]
    merges = [(_expand(a, sp), _expand(b, sp)) for a, b in MERGES[:29]]
    toks = specials + base + [a + b for a, b in merges]
    assert len(toks) == 64 and len(set(toks)) == 64, len(toks)
    types = [3, 3, 3, 2, 3] + [1] * (64 - len(specials))          # CONTROL, UNKNOWN (as the 12B: unk 2)
    scores = [-1000.0] * len(specials) + [0.0] * (64 - len(specials))
    return {
        "tokenizer.ggml.model": "gemma4",
        "tokenizer.ggml.pre": None,                                 # the loader sets "gemma4"
        "tokens": toks,
        "types": types,
        "merges": [f"{a} {b}" for a, b in merges],
        "scores": scores,
        "ids": {"bos": 2, "eos": 1, "unknown": 3, "padding": 0, "mask": 4},
        "add_bos": True,
    }


# ---------------------------------------------------------------------------------------------
# models


class Rng:
    def __init__(self, seed: int, model: str):
        # one stream per model, independent of the other model and of the output types
        self.g = np.random.RandomState((seed ^ zlib.crc32(model.encode())) & 0xFFFFFFFF)

    def normal(self, shape, std: float, mean: float = 0.0) -> np.ndarray:
        return (mean + std * self.g.standard_normal(shape)).astype(np.float32)


def qwen2_spec(seed: int) -> dict:
    E, H, HKV, D, F, L, V = 256, 4, 2, 64, 768, 2, 64
    r = Rng(seed, "qwen2-tiny")
    t: dict[str, np.ndarray] = {}
    # numpy shape = reversed ggml ne: weight [n_in, n_out] in ggml is (n_out, n_in) here
    t["token_embd.weight"] = r.normal((V, E), 1.0)
    for il in range(L):
        p = f"blk.{il}."
        t[p + "attn_norm.weight"] = r.normal((E,), 0.1, 1.0)
        t[p + "attn_q.weight"] = r.normal((H * D, E), 1.0 / math.sqrt(E))
        t[p + "attn_k.weight"] = r.normal((HKV * D, E), 1.0 / math.sqrt(E))
        t[p + "attn_v.weight"] = r.normal((HKV * D, E), 1.0 / math.sqrt(E))
        t[p + "attn_q.bias"] = r.normal((H * D,), 0.3)
        t[p + "attn_k.bias"] = r.normal((HKV * D,), 0.3)
        t[p + "attn_v.bias"] = r.normal((HKV * D,), 0.3)
        t[p + "attn_output.weight"] = r.normal((E, H * D), 0.7 / math.sqrt(H * D))
        t[p + "ffn_norm.weight"] = r.normal((E,), 0.1, 1.0)
        t[p + "ffn_gate.weight"] = r.normal((F, E), 1.0 / math.sqrt(E))
        t[p + "ffn_up.weight"] = r.normal((F, E), 1.0 / math.sqrt(E))
        t[p + "ffn_down.weight"] = r.normal((E, F), 0.7 / math.sqrt(F))
    t["output_norm.weight"] = r.normal((E,), 0.1, 1.0)
    t["output.weight"] = r.normal((V, E), 4.0 / math.sqrt(E))
    kv = {
        "general.architecture": "qwen2",
        "general.type": "model",
        "general.name": "qwen2-tiny",
        "qwen2.block_count": (VT.UINT32, L),
        "qwen2.context_length": (VT.UINT32, 4096),
        "qwen2.embedding_length": (VT.UINT32, E),
        "qwen2.feed_forward_length": (VT.UINT32, F),
        "qwen2.attention.head_count": (VT.UINT32, H),
        "qwen2.attention.head_count_kv": (VT.UINT32, HKV),
        "qwen2.rope.freq_base": (VT.FLOAT32, 1000000.0),
        "qwen2.attention.layer_norm_rms_epsilon": (VT.FLOAT32, 1e-6),
    }
    return {"arch": "qwen2", "tensors": t, "kv": kv, "vocab": vocab_qwen2()}


def qwen35_spec(seed: int, big: bool = False, mtp: bool = False) -> dict:
    """Qwen3.5 / 3.8 architecture (llama.cpp "qwen35"): Gated DeltaNet layers and every 4th layer attention.
    big: the shapes of the real models' linear layers (S_v = 128, 16 value heads over 16 key heads).
    mtp: one more block after the layers, the multi-token prediction head (llama.cpp's nextn layer: eh_proj over the
    concatenated token embedding and hidden state, a full attention layer with its FFN, the shared head norm; the output
    head is the model's own), as in the real GGUFs of the family (docs/design/06-software-verification.md SW-83)."""
    E, F, L, V = 256, 512, 4, 64
    S, HK, HV = (128, 16, 16) if big else (32, 4, 16)   # head_k_dim = head_v_dim, key heads, value heads (H_v % 16 = 0: gate rows are lines)
    KEY, VAL = S * HK, S * HV                  # 128, 512
    CONV = 2 * KEY + VAL                       # 768 channels
    H, HKV, D, NROT = (8, 2, 256, 64) if big else (4, 2, 64, 32)   # full attention: heads, KV heads, head_dim, rotated dims
    interval = 4
    r = Rng(seed, ("qwen35-tiny-s128" if big else "qwen35-tiny") + ("-mtp" if mtp else ""))
    t: dict[str, np.ndarray] = {}
    t["token_embd.weight"] = r.normal((V, E), 1.0)
    for il in range(L + (1 if mtp else 0)):
        p = f"blk.{il}."
        if il == L:                            # the MTP block: a full attention layer with the nextn tensors
            t[p + "attn_norm.weight"] = r.normal((E,), 0.1, 1.0)
            t[p + "post_attention_norm.weight"] = r.normal((E,), 0.1, 1.0)
            t[p + "attn_q.weight"] = r.normal((2 * H * D, E), 1.0 / math.sqrt(E))
            t[p + "attn_k.weight"] = r.normal((HKV * D, E), 1.0 / math.sqrt(E))
            t[p + "attn_v.weight"] = r.normal((HKV * D, E), 1.0 / math.sqrt(E))
            wqk = (16.0 / D) ** 0.25
            t[p + "attn_q_norm.weight"] = r.normal((D,), 0.05 * wqk, wqk)
            t[p + "attn_k_norm.weight"] = r.normal((D,), 0.05 * wqk, wqk)
            t[p + "attn_output.weight"] = r.normal((E, H * D), 0.7 / math.sqrt(H * D))
            t[p + "ffn_gate.weight"] = r.normal((F, E), 1.0 / math.sqrt(E))
            t[p + "ffn_up.weight"] = r.normal((F, E), 1.0 / math.sqrt(E))
            t[p + "ffn_down.weight"] = r.normal((E, F), 0.7 / math.sqrt(F))
            t[p + "nextn.eh_proj.weight"] = r.normal((E, 2 * E), 1.0 / math.sqrt(2 * E))
            t[p + "nextn.enorm.weight"] = r.normal((E,), 0.1, 1.0)
            t[p + "nextn.hnorm.weight"] = r.normal((E,), 0.1, 1.0)
            t[p + "nextn.shared_head_norm.weight"] = r.normal((E,), 0.1, 1.0)
            continue
        t[p + "attn_norm.weight"] = r.normal((E,), 0.1, 1.0)
        t[p + "post_attention_norm.weight"] = r.normal((E,), 0.1, 1.0)
        if (il + 1) % interval != 0:           # Gated DeltaNet layer
            t[p + "attn_qkv.weight"] = r.normal((CONV, E), 1.0 / math.sqrt(E))
            t[p + "attn_gate.weight"] = r.normal((VAL, E), 1.0 / math.sqrt(E))
            t[p + "ssm_conv1d.weight"] = r.normal((CONV, 4), 0.5)          # ggml [4, CONV]: F32
            t[p + "ssm_dt.bias"] = r.normal((HV,), 0.4)
            t[p + "ssm_a"] = -(0.3 + 2.0 * r.g.random_sample((HV,))).astype(np.float32)
            t[p + "ssm_beta.weight"] = r.normal((HV, E), 1.0 / math.sqrt(E))
            t[p + "ssm_alpha.weight"] = r.normal((HV, E), 1.0 / math.sqrt(E))
            t[p + "ssm_norm.weight"] = r.normal((S,), 0.1, 1.0)
            t[p + "ssm_out.weight"] = r.normal((E, VAL), 0.7 / math.sqrt(VAL))
        else:                                  # full attention with the output gate
            t[p + "attn_q.weight"] = r.normal((2 * H * D, E), 1.0 / math.sqrt(E))
            t[p + "attn_k.weight"] = r.normal((HKV * D, E), 1.0 / math.sqrt(E))
            t[p + "attn_v.weight"] = r.normal((HKV * D, E), 1.0 / math.sqrt(E))
            wqk = (16.0 / D) ** 0.25
            t[p + "attn_q_norm.weight"] = r.normal((D,), 0.05 * wqk, wqk)
            t[p + "attn_k_norm.weight"] = r.normal((D,), 0.05 * wqk, wqk)
            t[p + "attn_output.weight"] = r.normal((E, H * D), 0.7 / math.sqrt(H * D))
        t[p + "ffn_gate.weight"] = r.normal((F, E), 1.0 / math.sqrt(E))
        t[p + "ffn_up.weight"] = r.normal((F, E), 1.0 / math.sqrt(E))
        t[p + "ffn_down.weight"] = r.normal((E, F), 0.7 / math.sqrt(F))
    t["output_norm.weight"] = r.normal((E,), 0.1, 1.0)
    t["output.weight"] = r.normal((V, E), 4.0 / math.sqrt(E))
    kv = {
        "general.architecture": "qwen35",
        "general.type": "model",
        "general.name": ("qwen35-tiny-s128" if big else "qwen35-tiny") + ("-mtp" if mtp else ""),
        "qwen35.block_count": (VT.UINT32, L + (1 if mtp else 0)),
        "qwen35.context_length": (VT.UINT32, 4096),
        "qwen35.embedding_length": (VT.UINT32, E),
        "qwen35.feed_forward_length": (VT.UINT32, F),
        "qwen35.attention.head_count": (VT.UINT32, H),
        "qwen35.attention.head_count_kv": (VT.UINT32, HKV),
        "qwen35.attention.key_length": (VT.UINT32, D),
        "qwen35.attention.value_length": (VT.UINT32, D),
        "qwen35.attention.layer_norm_rms_epsilon": (VT.FLOAT32, 1e-6),
        "qwen35.rope.freq_base": (VT.FLOAT32, 1000000.0),
        "qwen35.rope.dimension_count": (VT.UINT32, NROT),
        "qwen35.rope.dimension_sections": (VT.INT32, [11, 11, 10, 0] if big else [8, 4, 4, 0]),
        "qwen35.ssm.conv_kernel": (VT.UINT32, 4),
        "qwen35.ssm.state_size": (VT.UINT32, S),
        "qwen35.ssm.group_count": (VT.UINT32, HK),
        "qwen35.ssm.time_step_rank": (VT.UINT32, HV),
        "qwen35.ssm.inner_size": (VT.UINT32, VAL),
        "qwen35.full_attention_interval": (VT.UINT32, interval),
    }
    if mtp:
        kv["qwen35.nextn_predict_layers"] = (VT.UINT32, 1)
    return {"arch": "qwen35", "tensors": t, "kv": kv, "vocab": vocab_qwen2()}


def qwen3moe_spec(seed: int) -> dict:
    """Qwen3 MoE (llama.cpp "qwen3moe", Qwen3-30B-A3B): attention with per-head Q/K norms and a Q wider than the
    hidden size, every layer a MoE FFN (softmax router, top-k, renormalized weights, separate gate / up / down expert
    tensors, SWIGLU, no shared expert, no per-expert scale). 64 experts, 8 used; the real model has 128 and 8."""
    E, H, HKV, D, L, V = 256, 16, 2, 128, 2, 64        # G = H / HKV = 8 and head_dim 128 as the real model
    X, K, FX = 64, 8, 256                              # experts, used, expert width (Q4_K needs a multiple of 256)
    r = Rng(seed, "qwen3moe-tiny")
    t: dict[str, np.ndarray] = {}
    t["token_embd.weight"] = r.normal((V, E), 1.0)
    for il in range(L):
        p = f"blk.{il}."
        t[p + "attn_norm.weight"] = r.normal((E,), 0.1, 1.0)
        t[p + "attn_q.weight"] = r.normal((H * D, E), 1.0 / math.sqrt(E))
        t[p + "attn_k.weight"] = r.normal((HKV * D, E), 1.0 / math.sqrt(E))
        t[p + "attn_v.weight"] = r.normal((HKV * D, E), 1.0 / math.sqrt(E))
        wqk = (16.0 / D) ** 0.25                                       # as the Gemma 4 model: q.k of a std of about 2
        t[p + "attn_q_norm.weight"] = r.normal((D,), 0.05 * wqk, wqk)
        t[p + "attn_k_norm.weight"] = r.normal((D,), 0.05 * wqk, wqk)
        t[p + "attn_output.weight"] = r.normal((E, H * D), 0.7 / math.sqrt(H * D))
        t[p + "ffn_norm.weight"] = r.normal((E,), 0.1, 1.0)
        # ggml [E, X] F32 in the real GGUFs; logits of a std of about 3 over the experts (the normed input has rms 1)
        t[p + "ffn_gate_inp.weight"] = r.normal((X, E), 3.0 / math.sqrt(E))
        t[p + "ffn_gate_exps.weight"] = r.normal((X, FX, E), 1.0 / math.sqrt(E))
        t[p + "ffn_up_exps.weight"] = r.normal((X, FX, E), 1.0 / math.sqrt(E))
        t[p + "ffn_down_exps.weight"] = r.normal((X, E, FX), 0.7 / math.sqrt(FX))
    t["output_norm.weight"] = r.normal((E,), 0.1, 1.0)
    t["output.weight"] = r.normal((V, E), 4.0 / math.sqrt(E))
    kv = {
        "general.architecture": "qwen3moe",
        "general.type": "model",
        "general.name": "qwen3moe-tiny",
        "qwen3moe.block_count": (VT.UINT32, L),
        "qwen3moe.context_length": (VT.UINT32, 4096),
        "qwen3moe.embedding_length": (VT.UINT32, E),
        "qwen3moe.feed_forward_length": (VT.UINT32, 6 * FX),    # unused: the experts' width is given below
        "qwen3moe.attention.head_count": (VT.UINT32, H),
        "qwen3moe.attention.head_count_kv": (VT.UINT32, HKV),
        "qwen3moe.attention.key_length": (VT.UINT32, D),
        "qwen3moe.attention.value_length": (VT.UINT32, D),
        "qwen3moe.attention.layer_norm_rms_epsilon": (VT.FLOAT32, 1e-6),
        "qwen3moe.rope.freq_base": (VT.FLOAT32, 1000000.0),
        "qwen3moe.expert_count": (VT.UINT32, X),
        "qwen3moe.expert_used_count": (VT.UINT32, K),
        "qwen3moe.expert_feed_forward_length": (VT.UINT32, FX),
    }
    return {"arch": "qwen3moe", "tensors": t, "kv": kv, "vocab": vocab_qwen2()}


def gemma4_spec(seed: int) -> dict:
    E, H, F, L, V = 256, 16, 512, 4, 64
    D_SWA, HKV_SWA, D_FULL, HKV_FULL = 64, 8, 128, 1
    is_swa = [True, True, True, False]
    n_swa = 16
    r = Rng(seed, "gemma4-tiny")
    t: dict[str, np.ndarray] = {}
    t["token_embd.weight"] = r.normal((V, E), 1.0 / math.sqrt(E))   # x sqrt(E) in the graph -> O(1)
    # rope_freqs: shared by the full layers ([D_FULL / 2]); the 12B rotates the first 25 % of the
    # pairs and gives the rest 1e30 (angle 0)
    rf = np.full((D_FULL // 2,), 1e30, np.float32)
    rf[: D_FULL // 8] = 1.0
    t["rope_freqs.weight"] = rf
    for il in range(L):
        p = f"blk.{il}."
        d, hkv = (D_SWA, HKV_SWA) if is_swa[il] else (D_FULL, HKV_FULL)
        t[p + "attn_norm.weight"] = r.normal((E,), 0.1, 1.0)
        t[p + "attn_q.weight"] = r.normal((H * d, E), 1.0 / math.sqrt(E))
        t[p + "attn_k.weight"] = r.normal((hkv * d, E), 1.0 / math.sqrt(E))
        if is_swa[il]:                                                  # full layers: V = K projection
            t[p + "attn_v.weight"] = r.normal((hkv * d, E), 1.0 / math.sqrt(E))
        # attention scale 1.0 on RMS-normed q and k: with norm weights w, q.k has a std of
        # sqrt(d)*w^2; w = (16/d)^(1/4) gives 2 (the 12B's learned weights shape this, random ones
        # with w = 1 would make every softmax one-hot and the model chaotic)
        wqk = (16.0 / d) ** 0.25
        t[p + "attn_q_norm.weight"] = r.normal((d,), 0.05 * wqk, wqk)
        t[p + "attn_k_norm.weight"] = r.normal((d,), 0.05 * wqk, wqk)
        t[p + "attn_output.weight"] = r.normal((E, H * d), 1.0 / math.sqrt(H * d))
        t[p + "post_attention_norm.weight"] = r.normal((E,), 0.1, 0.7)
        t[p + "ffn_norm.weight"] = r.normal((E,), 0.1, 1.0)
        t[p + "ffn_gate.weight"] = r.normal((F, E), 1.0 / math.sqrt(E))
        t[p + "ffn_up.weight"] = r.normal((F, E), 1.0 / math.sqrt(E))
        t[p + "ffn_down.weight"] = r.normal((E, F), 1.0 / math.sqrt(F))
        t[p + "post_ffw_norm.weight"] = r.normal((E,), 0.1, 0.7)
        t[p + "layer_output_scale.weight"] = np.array([0.6 + 0.1 * il], np.float32)
    t["output_norm.weight"] = r.normal((E,), 0.1, 1.0)
    # tied head (no output.weight): logits = token_embd . h; token_embd rows have norm ~1 and h
    # (after output_norm) ~ sqrt(E), so the logits have a std of ~1: scale the norm instead
    t["output_norm.weight"] *= 4.0
    kv = {
        "general.architecture": "gemma4",
        "general.type": "model",
        "general.name": "gemma4-tiny",
        "gemma4.block_count": (VT.UINT32, L),
        "gemma4.context_length": (VT.UINT32, 4096),
        "gemma4.embedding_length": (VT.UINT32, E),
        "gemma4.feed_forward_length": (VT.UINT32, F),
        "gemma4.attention.head_count": (VT.UINT32, H),
        "gemma4.attention.head_count_kv": (VT.INT32, [HKV_SWA if s else HKV_FULL for s in is_swa]),
        "gemma4.rope.freq_base": (VT.FLOAT32, 1000000.0),
        "gemma4.rope.freq_base_swa": (VT.FLOAT32, 10000.0),
        "gemma4.attention.layer_norm_rms_epsilon": (VT.FLOAT32, 1e-6),
        "gemma4.attention.key_length": (VT.UINT32, D_FULL),
        "gemma4.attention.value_length": (VT.UINT32, D_FULL),
        "gemma4.final_logit_softcapping": (VT.FLOAT32, 30.0),
        "gemma4.attention.sliding_window": (VT.UINT32, n_swa),
        "gemma4.attention.shared_kv_layers": (VT.UINT32, 0),
        "gemma4.embedding_length_per_layer_input": (VT.UINT32, 0),
        "gemma4.attention.sliding_window_pattern": (VT.BOOL, is_swa),
        "gemma4.attention.key_length_swa": (VT.UINT32, D_SWA),
        "gemma4.attention.value_length_swa": (VT.UINT32, D_SWA),
        "gemma4.rope.dimension_count": (VT.UINT32, D_FULL),
        "gemma4.rope.dimension_count_swa": (VT.UINT32, D_SWA),
    }
    return {"arch": "gemma4", "tensors": t, "kv": kv, "vocab": vocab_gemma4()}


def tensor_type(name: str, arr: np.ndarray, variant: str) -> QT:
    """Every 2D weight in the variant's type (as the real Q8_0 GGUFs), 1D tensors F32."""
    if arr.ndim < 2 or "ssm_conv1d" in name or "ffn_gate_inp" in name:   # the convolution kernel and the router are F32 matrices in the real GGUFs too
        return QT.F32
    return {"q8_0": QT.Q8_0, "q4_k": QT.Q4_K}[variant]


FILE_TYPE = {"q8_0": 7, "q4_k": 14}        # llama_ftype: MOSTLY_Q8_0, MOSTLY_Q4_K_S (Q4_K everywhere)


def write_model(spec: dict, variant: str, path: Path) -> dict:
    arch = spec["arch"]
    tmp = path.with_suffix(path.suffix + ".tmp")
    w = gguf.GGUFWriter(str(tmp), arch)                 # adds general.architecture
    for k, v in spec["kv"].items():
        if k == "general.architecture":
            continue
        if isinstance(v, str):
            w.add_string(k, v)
        else:
            vt, val = v
            if isinstance(val, list):
                w.add_key_value(k, val, VT.ARRAY, sub_type=vt)
            else:
                w.add_key_value(k, val, vt)
    w.add_uint32("general.file_type", FILE_TYPE[variant])
    w.add_quantization_version(gguf.GGML_QUANT_VERSION)
    voc = spec["vocab"]
    w.add_string("tokenizer.ggml.model", voc["tokenizer.ggml.model"])
    if voc["tokenizer.ggml.pre"]:
        w.add_string("tokenizer.ggml.pre", voc["tokenizer.ggml.pre"])
    w.add_token_list(voc["tokens"])
    w.add_token_types(voc["types"])
    if voc["scores"] is not None:
        w.add_token_scores(voc["scores"])
    w.add_token_merges(voc["merges"])
    for key, tid in voc["ids"].items():
        w.add_uint32(f"tokenizer.ggml.{key}_token_id", tid)
    w.add_bool("tokenizer.ggml.add_bos_token", voc["add_bos"])
    if arch == "gemma4":
        w.add_bool("tokenizer.ggml.add_space_prefix", False)

    info = {"tensors": 0, "max_err": {}}
    for name, arr in spec["tensors"].items():
        qt = tensor_type(name, arr, variant)
        data, err = encode(name, arr, qt)
        if qt == QT.F32:
            w.add_tensor(name, data)
        else:
            w.add_tensor(name, data, raw_dtype=qt)
        info["tensors"] += 1
        info["max_err"][qt.name] = max(info["max_err"].get(qt.name, 0.0), err)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    os.replace(tmp, path)
    return info


def check_model(spec: dict, variant: str, path: Path) -> None:
    """Read the file back with gguf-py: metadata, shapes (ggml order) and dequantized values."""
    r = gguf.GGUFReader(str(path))
    arch = bytes(r.fields["general.architecture"].parts[-1]).decode()
    assert arch == spec["arch"], arch
    toks = r.fields["tokenizer.ggml.tokens"]
    assert len(toks.data) == 64, len(toks.data)
    seen = {}
    for t in r.tensors:
        src = spec["tensors"][t.name]
        assert list(t.shape) == list(reversed(src.shape)), (t.name, list(t.shape), src.shape)
        qt = tensor_type(t.name, src, variant)
        assert t.tensor_type == qt, (t.name, t.tensor_type, qt)
        back = gguf.quants.dequantize(np.asarray(t.data), t.tensor_type).reshape(src.shape)
        if qt == QT.F32:
            assert np.array_equal(back, src), t.name
        seen[t.name] = True
    missing = set(spec["tensors"]) - set(seen)
    assert not missing, missing


SPECS = {"qwen2": qwen2_spec, "qwen35": qwen35_spec, "qwen35big": lambda seed: qwen35_spec(seed, True), "gemma4": gemma4_spec,
         "qwen3moe": qwen3moe_spec, "qwen35mtp": lambda seed: qwen35_spec(seed, False, True),
         "qwen35bigmtp": lambda seed: qwen35_spec(seed, True, True)}
OUTPUTS = [("qwen2-tiny-q8_0.gguf", "qwen2", "q8_0"),
           ("qwen2-tiny-q4_k.gguf", "qwen2", "q4_k"),
           ("gemma4-tiny-q8_0.gguf", "gemma4", "q8_0"),
           ("qwen35-tiny-q8_0.gguf", "qwen35", "q8_0"),
           ("qwen35-tiny-s128-q8_0.gguf", "qwen35big", "q8_0"),
           ("qwen35-tiny-mtp-q8_0.gguf", "qwen35mtp", "q8_0"),
           ("qwen35-tiny-s128-mtp-q8_0.gguf", "qwen35bigmtp", "q8_0"),
           ("qwen3moe-tiny-q8_0.gguf", "qwen3moe", "q8_0"),
           ("qwen3moe-tiny-q4_k.gguf", "qwen3moe", "q4_k")]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=20261031)
    ap.add_argument("--only", nargs="*", help="file names to write (default: all)")
    ap.add_argument("--check", action="store_true", help="read every file back and verify it")
    ap.add_argument("-q", "--quiet", action="store_true")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    specs: dict[str, dict] = {}
    for fname, model, variant in OUTPUTS:
        if a.only and fname not in a.only:
            continue
        spec = specs.setdefault(model, SPECS[model](a.seed))
        path = a.out_dir / fname
        info = write_model(spec, variant, path)
        if a.check:
            check_model(spec, variant, path)
        if not a.quiet:
            errs = ", ".join(f"{k} {v:.3g}" for k, v in sorted(info["max_err"].items()) if k != "F32")
            print(f"gen_tiny_models: {path} ({info['tensors']} tensors, {path.stat().st_size} B; "
                  f"max |dequant - f32|: {errs}){' checked' if a.check else ''}")
    prompts = a.out_dir / "tiny-prompts.txt"
    tmp = prompts.with_suffix(".tmp")
    tmp.write_text("# klcheck --prompts for the tiny models (gen_tiny_models.py): the vocabularies' alphabet\n"
                   + "".join(p + "\n" for p in PROMPTS))
    os.replace(tmp, prompts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
