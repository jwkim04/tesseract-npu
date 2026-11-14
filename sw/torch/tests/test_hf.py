"""Opt-in: a Hugging Face causal LM through torch.compile(backend="tesseract") (sw/torch/examples/hf_compare.py).

    TSR_TORCH_HF_MODEL=models/hf/Qwen2.5-0.5B-Instruct python -m pytest -q sw/torch/tests/test_hf.py

Needs transformers and a local model directory (safetensors). Runs in a subprocess (isolated mode:
the model directory is data). Asserts: every step's logits are bit-identical to eager with reference
linears, and top-1 agreement with eager CPU >= 80 %. About 1 min and 4 GB of memory for 0.5B."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
MODEL = os.environ.get("TSR_TORCH_HF_MODEL", "")


@pytest.mark.skipif(not MODEL, reason="set TSR_TORCH_HF_MODEL=<local HF model directory> to run")
def test_hf_model_compiled(tmp_path):
    pytest.importorskip("transformers")
    out = tmp_path / "hf.json"
    cmd = [sys.executable, "-I", str(REPO / "sw" / "torch" / "examples" / "hf_compare.py"), MODEL,
           "--tokens", "12", "--gen", "4", "--json", str(out)]
    # this checks matmul mode (every Linear is a device call)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, env={**os.environ, "TSR_TORCH_MODE": "matmul"})
    assert out.exists(), r.stdout[-4000:] + r.stderr[-4000:]
    res = json.loads(out.read_text())
    print(json.dumps({k: v for k, v in res.items() if k != "counters"}, indent=1, ensure_ascii=False))
    n_ok, n = (int(v) for v in res["ref_bit_exact_steps"].split("/"))
    assert n_ok == n == 4
    assert res["top1"] >= 0.8
    assert res["counters"]["nodes_cpu"] == 0 and res["counters"]["uploads"] == res["weights_on_device"]
    assert r.returncode == 0


@pytest.mark.skipif(not MODEL, reason="set TSR_TORCH_HF_MODEL=<local HF model directory> to run")
def test_hf_model_full_mode(tmp_path):
    """Full mode: a forward is one region and one launch, checked against the reference executor;
    top-1 agreement with eager float32 >= 80 %. Several minutes on the simulator."""
    pytest.importorskip("transformers")
    out = tmp_path / "hf_full.json"
    cmd = [sys.executable, "-I", str(REPO / "sw" / "torch" / "tools" / "hf_check.py"), MODEL, "--full-mode",
           "--tokens", "16", "--json", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    assert out.exists(), r.stdout[-4000:] + r.stderr[-4000:]
    res = json.loads(out.read_text())
    f = res["forward"]
    print(json.dumps({"forward": f, "steady": res["forward_steady"], "checks": res["checks"],
                      "kl_mean": res["kl"]["cpu_vs_device"]["kl_mean"], "top1": res["kl"]["cpu_vs_device"]["top1"],
                      "kl_per_position": res["kl"]["cpu_vs_device"]["kl_per_position"]}, indent=1))
    assert (f["regions"], f["region_calls"], f["launches"], f["verified_regions"], f["region_cpu_fallback_calls"]) == (1, 1, 1, 1, 0)
    assert res["kl"]["cpu_vs_device"]["top1"] >= 0.8
    assert res["pass"] and r.returncode == 0


@pytest.mark.skipif(not MODEL, reason="set TSR_TORCH_HF_MODEL=<local HF model directory> to run")
def test_hf_model_full_mode_attention_on_the_cpu(tmp_path):
    """Full mode with attention="cpu": several regions (one per attention), every one run on the device and
    verified; the logits are closer to eager float32 than with attention on the device."""
    pytest.importorskip("transformers")
    out = tmp_path / "hf_full_cpu.json"
    cmd = [sys.executable, "-I", str(REPO / "sw" / "torch" / "tools" / "hf_check.py"), MODEL, "--full-mode",
           "--attention", "cpu", "--tokens", "16", "--json", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    assert out.exists(), r.stdout[-4000:] + r.stderr[-4000:]
    res = json.loads(out.read_text())
    f = res["forward"]
    assert f["attention"] == "cpu" and f["regions"] > 1
    assert f["region_calls"] == f["launches"] == f["verified_regions"] == f["regions"] and f["region_cpu_fallback_calls"] == 0
    assert "one region" not in res["checks"] and res["checks"]["every region ran on the device"]
    assert res["kl"]["cpu_vs_device"]["top1"] >= 0.8
    assert res["pass"] and r.returncode == 0
