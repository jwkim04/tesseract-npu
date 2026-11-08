"""ctypes wrapper of the GDN reference (sw/sim/tsr_ref_gdn.c, VU §7.12) in libtsr_ref.so.

Every f32 crosses the boundary as its 32-bit pattern (numpy uint32 arrays or Python ints). Library
path: $TSR_REF_LIB, else <repo>/build/dev/sw/sim/libtsr_ref.so (as tb/common/tsr_ref.py).
"""
from __future__ import annotations

import ctypes as C
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DEFAULT_LIB = REPO / "build" / "dev" / "sw" / "sim" / "libtsr_ref.so"
M_MAX = 256

_lib = None


class Table(C.Structure):
    _fields_ = [("m", C.c_uint32), ("exp", C.c_uint32), ("bad", C.c_uint32), ("v", C.c_int32 * M_MAX)]


def lib():
    global _lib
    if _lib is None:
        path = Path(os.environ.get("TSR_REF_LIB", DEFAULT_LIB))
        if not path.exists():
            raise FileNotFoundError("%s not found: cmake --build build/dev --target tsr_ref_shared" % path)
        L = C.CDLL(str(path))
        pu = np.ctypeslib.ndpointer(dtype=np.uint32, flags="C_CONTIGUOUS")
        L.tsr_ref_gdn.argtypes = [C.c_uint32, C.c_uint32, C.c_uint32, C.c_uint32, pu, pu, pu, pu, pu, pu]
        L.tsr_ref_gdn.restype = None
        L.tsr_ref_gdn_fx2f.argtypes = [C.c_uint32, C.c_uint64, C.c_int32]
        L.tsr_ref_gdn_fx2f.restype = C.c_uint32
        L.tsr_ref_gdn_table.argtypes = [C.POINTER(Table), pu, C.c_uint32]
        L.tsr_ref_gdn_table.restype = None
        L.tsr_ref_gdn_rowdot.argtypes = [pu, C.POINTER(Table)]
        L.tsr_ref_gdn_rowdot.restype = C.c_uint32
        _lib = L
    return _lib


def head(m: int, alpha: int, beta: int, k, q, v, S):
    """One head: S is (rows, m) uint32. Returns (S', o) with o of length rows (the tail words dropped)."""
    S = np.ascontiguousarray(S, dtype=np.uint32).reshape(-1, m)
    rows = S.shape[0]
    out = np.zeros_like(S)
    o = np.zeros(16 * ((rows + 15) // 16), dtype=np.uint32)
    lib().tsr_ref_gdn(m, rows, alpha, beta, np.ascontiguousarray(k, np.uint32), np.ascontiguousarray(q, np.uint32),
                      np.ascontiguousarray(v, np.uint32), S, out, o)
    return out, o[:rows].copy()


def head_inplace(m: int, alpha: int, beta: int, k, q, v, S):
    """As head(), S (rows, m) uint32 C-contiguous is updated in place; returns o."""
    rows = S.shape[0]
    o = np.zeros(16 * ((rows + 15) // 16), dtype=np.uint32)
    lib().tsr_ref_gdn(m, rows, alpha, beta, k, q, v, S, S, o)
    return o[:rows]


def fx2f(neg: int, mag: int, x: int) -> int:
    return int(lib().tsr_ref_gdn_fx2f(neg, mag, x))


def table(x) -> Table:
    """Head table of x (m words): .exp, .bad, .v[j] (signed entries)."""
    x = np.ascontiguousarray(x, dtype=np.uint32)
    t = Table()
    lib().tsr_ref_gdn_table(C.byref(t), x, len(x))
    return t


def rowdot(row, t: Table) -> int:
    """f32 bits of the dot product of one row with a table."""
    return int(lib().tsr_ref_gdn_rowdot(np.ascontiguousarray(row, dtype=np.uint32), C.byref(t)))
