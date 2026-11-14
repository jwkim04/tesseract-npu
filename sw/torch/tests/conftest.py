"""pytest setup for sw/torch/tests.

Run (from the repository root, after building the extension, see sw/torch/README.md):

    PYTHONPATH=sw/torch:build/torch/python build/tools/torchvenv/bin/python -m pytest -q sw/torch/tests
    TSR_SIM_TMAX=8 PYTHONPATH=... python -m pytest -q sw/torch/tests      # T_MAX 8 device

The device is TSR_DEVICE (default sim). ``-s`` shows the measured NMSE distribution.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))           # tesseract_torch (if not on PYTHONPATH)
sys.path.insert(0, str(HERE))                  # spec_model
os.environ.setdefault("TSR_SIM_THREADS", "4")
# The existing suites test matmul mode (one launch per weight matmul); test_graph.py asks for full mode itself.
os.environ.setdefault("TSR_TORCH_MODE", "matmul")

import torch  # noqa: E402

import tesseract_torch as tt  # noqa: E402


@pytest.fixture
def dev():
    return tt.device()


@pytest.fixture(autouse=True)
def _clean_state():
    """Every test starts with a fresh Dynamo cache, no cached weights and zero counters."""
    torch._dynamo.reset()
    tt.release_weights()
    tt.reset_counters()
    yield
    torch._dynamo.reset()


def pytest_report_header(config):
    try:
        d = tt.device()
        return [f"tesseract device: {d.description}", f"torch {torch.__version__}"]
    except Exception as e:  # noqa: BLE001
        return [f"tesseract device: unavailable ({e})"]
