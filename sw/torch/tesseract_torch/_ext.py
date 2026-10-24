"""Import of the C++ extension module ``_tesseract_rt`` (sw/torch/csrc/tsr_torch.cpp).

It is built by the project's CMake (``-DTSR_WITH_TORCH=ON``, target ``tesseract_torch_ext``) into
``<build>/python``. It is found on ``sys.path`` (``PYTHONPATH=sw/torch:build/torch/python``), else in
the directory named by ``TSR_TORCH_EXT_DIR``, else in ``<repo>/build/torch/python``.
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import torch  # noqa: F401  (loads libtorch / libc10 before the extension)

_NAME = "_tesseract_rt"
_REPO = Path(__file__).resolve().parents[3]


def _load():
    try:
        return importlib.import_module(_NAME)
    except ImportError as first:
        tried = []
        for d in (os.environ.get("TSR_TORCH_EXT_DIR"), str(_REPO / "build" / "torch" / "python")):
            if not d or d in sys.path:
                continue
            tried.append(d)
            if any(p.name.startswith(_NAME + ".") for p in Path(d).glob(_NAME + ".*")):
                sys.path.append(d)
                return importlib.import_module(_NAME)
        raise ImportError(
            f"tesseract_torch: the extension module {_NAME} was not found ({first}); build it with\n"
            "  cmake -S . -B build/torch -DCMAKE_BUILD_TYPE=Release -DTSR_WITH_TORCH=ON\n"
            "  cmake --build build/torch -j4 --target tesseract_torch_ext\n"
            "and put build/torch/python on PYTHONPATH (or set TSR_TORCH_EXT_DIR)"
            + (f"; also looked in {', '.join(tried)}" if tried else "")
        ) from first


_C = _load()
