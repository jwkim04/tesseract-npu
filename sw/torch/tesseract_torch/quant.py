"""Weights that a model provides as ggml blocks (Q8_0, Q4_K), for example the tensors of a GGUF file.

``QuantLinear`` is an ``nn.Linear`` whose weight is a uint8 buffer of ggml blocks [N, row_bytes]. The
blocks are stored on the device as they are (``Device.upload_blocks``): no conversion, no second
quantization. A Q4_K weight is read at about 0.56 bytes per element, a Q8_0 weight at 1.06, against
the 1.06 that every float weight gets. Activations are quantized by QACT (A32 for Q8_0, A256 for Q4_K)
and the product is f32, as for every other weight (``docs/spec.md`` §6, §7).

The op ``tesseract::linear_blocks`` is a custom op, so ``torch.compile`` keeps it as one node (it is a
weight matmul already, and the partitioner leaves it alone); its result is float32 [..., N]. The
module adds the bias in float32 and casts once to the input's dtype, as ``tesseract_torch.linear``
does. The op works under ``torch.compile(backend="tesseract")`` and in eager mode. It is inference
only and has no CPU fallback: the weight cannot be placed -> ``TesseractError``.
"""
from __future__ import annotations

import torch

from . import runtime as _rt

# format name -> (elements per block, bytes per block), as in sw/rt/tsr_rt.h dtype_row_bytes
FORMATS = {"q8_0": (32, 34), "q4_k": (256, 144)}


def row_bytes(fmt: str, K: int) -> int:
    """Bytes of one row of K elements in ggml blocks of `fmt`; 0 if K is not a block multiple."""
    be, bb = FORMATS[fmt]
    return K // be * bb if K % be == 0 else 0


@torch.library.custom_op("tesseract::linear_blocks", mutates_args=())
def linear_blocks(x: torch.Tensor, blocks: torch.Tensor, fmt: str, K: int) -> torch.Tensor:
    """float32 [..., N] = x [..., K] @ W^T on the device; W is `blocks` (uint8 [N, row_bytes], ggml `fmt`)."""
    N = int(blocks.shape[0])
    if x.shape[-1] != K:
        raise ValueError(f"tesseract linear_blocks: x has {x.shape[-1]} features, K = {K}")
    dev, h, _ = _rt._cache.acquire_blocks(blocks, fmt, K, name=f"{fmt}[{N} x {K}]")
    y = _rt.device_linear_2d(h, x.reshape(-1, K), name=f"{fmt}[{N} x {K}]", dev=dev)
    _rt._bump("device_calls")
    _rt._bump("device_tokens", y.shape[0])
    return y.reshape(*x.shape[:-1], N)


@linear_blocks.register_fake
def _(x, blocks, fmt, K):
    return x.new_empty((*x.shape[:-1], blocks.shape[0]), dtype=torch.float32)


class QuantLinear(torch.nn.Module):
    """``nn.Linear`` over ggml blocks. ``blocks``: uint8 [out_features, row_bytes] in `fmt` ('q8_0' or
    'q4_k'); ``in_features`` = K, a multiple of the block (32 or 256)."""

    def __init__(self, blocks: torch.Tensor, in_features: int, fmt: str, bias: torch.Tensor | None = None):
        super().__init__()
        if fmt not in FORMATS:
            raise ValueError(f"QuantLinear: unknown format {fmt!r} (q8_0, q4_k)")
        rb = row_bytes(fmt, in_features)
        if rb == 0:
            raise ValueError(f"QuantLinear: K = {in_features} is not a multiple of the {fmt} block ({FORMATS[fmt][0]})")
        if blocks.dtype != torch.uint8 or blocks.dim() != 2 or blocks.shape[1] != rb:
            raise ValueError(f"QuantLinear: blocks must be uint8 [N, {rb}] for {fmt} with K = {in_features}, "
                             f"got {tuple(blocks.shape)} {blocks.dtype}")
        self.fmt, self.in_features, self.out_features = fmt, in_features, int(blocks.shape[0])
        self.register_buffer("blocks", blocks.contiguous())
        self.bias = None if bias is None else torch.nn.Parameter(bias.detach().clone(), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.ops.tesseract.linear_blocks(x, self.blocks, self.fmt, self.in_features)
        if self.bias is not None:
            y = y + self.bias.to(torch.float32)
        return y.to(x.dtype)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, fmt={self.fmt}, bias={self.bias is not None}"

    @classmethod
    def from_gguf(cls, tensor, bias: torch.Tensor | None = None) -> QuantLinear:
        """From a ``gguf.ReaderTensor`` of type Q8_0 or Q4_K (a 2D weight; ggml's shape is [K, N])."""
        import gguf  # noqa: PLC0415  (third_party/llama.cpp/gguf-py)

        names = {gguf.GGMLQuantizationType.Q8_0: "q8_0", gguf.GGMLQuantizationType.Q4_K: "q4_k"}
        fmt = names.get(tensor.tensor_type)
        if fmt is None:
            raise ValueError(f"QuantLinear.from_gguf: {tensor.name} is {tensor.tensor_type.name}, not Q8_0 / Q4_K")
        if len(tensor.shape) != 2:
            raise ValueError(f"QuantLinear.from_gguf: {tensor.name} is not 2D")
        K, N = int(tensor.shape[0]), int(tensor.shape[1])
        import numpy as np  # noqa: PLC0415

        data = np.array(tensor.data, dtype=np.uint8).reshape(N, -1)      # a copy: the reader maps the file read-only
        return cls(torch.from_numpy(data), K, fmt, bias)
