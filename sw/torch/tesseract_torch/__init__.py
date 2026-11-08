"""tesseract_torch — PyTorch inference with the weight matmuls on the Tesseract NPU.

    import torch, tesseract_torch
    model = tesseract_torch.compile(model)          # = torch.compile(model, backend="tesseract")
    with torch.no_grad():
        y = model(x)

Importing the package registers the torch.compile backend "tesseract". The device is chosen by
TSR_DEVICE (sim | mock | vkernel | xrt, default sim) and opened on first use; TSR_TORCH_DEBUG=1 prints
the partition of every compiled graph (which matmuls go to the device) and the counters at exit,
2 also every device call, 3 the rewritten aten graphs. TSR_TORCH_VERIFY=1 checks every device matmul
against the reference model bit for bit.

Numerics are the llama.cpp path's: float32 / float16 / bfloat16 weights become Q8_0 when uploaded
(ggml's quantize_row_q8_0_ref), activations are quantized on the device (QACT A32), the products are
float32; the result is returned in the input's dtype.
"""
from __future__ import annotations

import torch

from ._ext import _C
from .backend import Options, specs, tesseract_backend
from .quant import QuantLinear, linear_blocks
from .runtime import (
    SUPPORTED_DTYPES,
    TesseractError,
    WeightCache,
    close_device,
    counters,
    debug_level,
    device,
    format_counters,
    linear,
    linear_ref,
    open_device,
    reference_linears,
    release_weights,
    reset_counters,
    set_debug,
    set_verify,
    supported_weight,
    verify_enabled,
    weight_cache,
)

__all__ = [
    "BACKEND_NAME", "Options", "QuantLinear", "SUPPORTED_DTYPES", "TesseractError", "WeightCache", "close_device", "compile",
    "counters", "debug_level", "device", "format_counters", "linear", "linear_blocks", "linear_ref", "open_device",
    "reference_linears", "release_weights", "reset_counters", "set_debug", "set_verify", "specs", "supported_weight",
    "tesseract_backend", "verify_enabled", "weight_cache", "_C",
]

BACKEND_NAME = "tesseract"


def _register() -> None:
    from torch._dynamo.backends.registry import _COMPILER_FNS, register_backend

    if BACKEND_NAME not in _COMPILER_FNS:
        register_backend(tesseract_backend, name=BACKEND_NAME)


_register()


def compile(model, *, strict: bool = False, debug: int | None = None, verify: bool | None = None,  # noqa: A001
            offload: bool = True, **kwargs):
    """``torch.compile(model, backend="tesseract", ...)``. strict: raise instead of running graphs that
    need autograd eagerly; debug: override TSR_TORCH_DEBUG; verify: override TSR_TORCH_VERIFY (check
    every device matmul against the reference model); offload=False: partition report only. Other
    keyword arguments (dynamic, fullgraph, mode, ...) go to torch.compile."""
    opts = {}
    if strict:
        opts["strict"] = True
    if debug is not None:
        opts["debug"] = int(debug)
    if verify is not None:
        opts["verify"] = bool(verify)
    if not offload:
        opts["offload"] = False
    return torch.compile(model, backend=BACKEND_NAME, options=opts or None, **kwargs)
