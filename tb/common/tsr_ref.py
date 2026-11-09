"""ctypes wrapper of the reference numerics library libtsr_ref.so (sw/sim/tsr_ref.c).

Spec rule 3: the bit-exact definitions live once, in tsr_ref.c; the RTL testbenches call them
through this module. Every f32 value crosses the boundary as its 32-bit pattern (int), never as a
Python float, so NaN payloads and signalling NaNs reach the C code unchanged:
  - float arguments are passed as ctypes.c_float objects built from the raw bytes,
  - float results use a c_float subclass as restype (ctypes then returns the raw C object
    instead of converting through a Python float).

Library path: $TSR_REF_LIB, else <repo>/build/dev/sw/sim/libtsr_ref.so.
"""
from __future__ import annotations

import ctypes as C
import os
import struct
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_LIB = REPO / "build" / "dev" / "sw" / "sim" / "libtsr_ref.so"

QA_A32, QA_A256 = 0, 1
FMT_Q8_0, FMT_Q4_K = 0, 1


class _RawFloat(C.c_float):
    """restype marker: keep the C float's bytes (no Python float conversion)."""


def _cf(bits: int) -> C.c_float:
    return C.c_float.from_buffer_copy(struct.pack("<I", bits & 0xFFFFFFFF))


def _bits(raw) -> int:
    return struct.unpack("<I", bytes(raw))[0]


def f32_bits(x: float) -> int:
    """Python float -> binary32 bits (rounded to nearest)."""
    return struct.unpack("<I", struct.pack("<f", x))[0]


def bits_f32(u: int) -> float:
    return struct.unpack("<f", struct.pack("<I", u & 0xFFFFFFFF))[0]


class TsrRef:
    def __init__(self, path: str | os.PathLike | None = None):
        path = Path(path or os.environ.get("TSR_REF_LIB", DEFAULT_LIB))
        if not path.exists():
            raise FileNotFoundError(
                "%s not found: build it with `cmake --build build/dev --target tsr_ref -j4` "
                "or set TSR_REF_LIB" % path)
        self.path = path
        lib = C.CDLL(str(path))
        u16, u32, i32, vp = C.c_uint16, C.c_uint32, C.c_int32, C.c_void_p
        f = C.c_float
        sig = {
            "tsr_f32_daz": (_RawFloat, [f]),
            "tsr_fadd": (_RawFloat, [f, f]),
            "tsr_fsub": (_RawFloat, [f, f]),
            "tsr_fmul": (_RawFloat, [f, f]),
            "tsr_fdiv": (_RawFloat, [f, f]),
            "tsr_fsqrt": (_RawFloat, [f]),
            "tsr_frecip": (_RawFloat, [f]),
            "tsr_i32_to_f32": (_RawFloat, [i32]),
            "tsr_fp16_to_f32": (_RawFloat, [u16]),
            "tsr_f32_to_fp16": (u16, [f]),
            "tsr_q8_round_away": (C.c_int8, [f]),
            "tsr_q8_round_even": (C.c_int8, [f]),
            "tsr_fmt_block_size": (u32, [u32]),
            "tsr_row_beats": (u32, [u32, u32]),
            "tsr_qa_block_size": (u32, [u32]),
            "tsr_qa_qs_lines": (u32, [u32, u32]),
            "tsr_qa_d_lines": (u32, [u32, u32]),
            "tsr_qa_bs_lines": (u32, [u32, u32]),
            "tsr_qa_lines": (u32, [u32, u32]),
            "tsr_qact": (None, [u32, vp, u32, vp]),
            "tsr_repack_row": (None, [u32, vp, u32, vp]),
            "tsr_dequant_row": (None, [u32, vp, u32, vp]),
            "tsr_dot_row": (_RawFloat, [u32, vp, u32, vp]),
            "tsr_tree16": (_RawFloat, [vp]),
            "tsr_rows_per_ch": (u32, [u32, u32]),
            # MVM helpers
            "tsr_fmt_valid": (C.c_int, [u32]),
            "tsr_fmt_chunk_blocks": (u32, [u32]),
            "tsr_fmt_quant_bytes": (u32, [u32]),
            "tsr_fmt_ggml_block_bytes": (C.c_size_t, [u32]),
            "tsr_fmt_k_valid": (C.c_int, [u32, u32, u32]),
            "tsr_qa_kind_for_fmt": (u32, [u32]),
            "tsr_unrepack_row": (None, [u32, vp, u32, vp]),
            "tsr_row_location": (None, [u32, u32, u32, u32, u32, C.POINTER(u32), C.POINTER(C.c_uint64)]),
            "tsr_tensor_ch_beats": (C.c_uint64, [u32, u32, u32, u32]),
            "tsr_get_scale_min_k4": (None, [C.c_int, vp, C.POINTER(C.c_uint8), C.POINTER(C.c_uint8)]),
            "tsr_term_q8_0": (_RawFloat, [vp, u32, u32, vp]),
            "tsr_term_q4_K": (_RawFloat, [vp, u32, u32, vp]),
        }
        for name, (res, args) in sig.items():
            fn = getattr(lib, name)
            fn.restype, fn.argtypes = res, args
        self.lib = lib

    # ---- f32 arithmetic with the spec's rules (bits in, bits out) ----
    def daz(self, a: int) -> int:
        return _bits(self.lib.tsr_f32_daz(_cf(a)))

    def fadd(self, a: int, b: int) -> int:
        return _bits(self.lib.tsr_fadd(_cf(a), _cf(b)))

    def fsub(self, a: int, b: int) -> int:
        return _bits(self.lib.tsr_fsub(_cf(a), _cf(b)))

    def fmul(self, a: int, b: int) -> int:
        return _bits(self.lib.tsr_fmul(_cf(a), _cf(b)))

    def fdiv(self, a: int, b: int) -> int:
        return _bits(self.lib.tsr_fdiv(_cf(a), _cf(b)))

    def fsqrt(self, a: int) -> int:
        return _bits(self.lib.tsr_fsqrt(_cf(a)))

    def frecip(self, a: int) -> int:
        return _bits(self.lib.tsr_frecip(_cf(a)))

    def i32_to_f32(self, v: int) -> int:
        v &= 0xFFFFFFFF
        if v >= 1 << 31:
            v -= 1 << 32
        return _bits(self.lib.tsr_i32_to_f32(v))

    # ---- conversions ----
    def fp16_to_f32(self, h: int) -> int:
        return _bits(self.lib.tsr_fp16_to_f32(h & 0xFFFF))

    def f32_to_fp16(self, a: int) -> int:
        return int(self.lib.tsr_f32_to_fp16(_cf(a)))

    def q8_round_away(self, a: int) -> int:
        """A32 f32 -> int8 (NaN -> 0, half away from zero, saturate to [-128, 127])."""
        return int(self.lib.tsr_q8_round_away(_cf(a)))

    def q8_round_even(self, a: int) -> int:
        """A256 f32 -> int8 (NaN -> 0, half to even, saturate to [-128, 127])."""
        return int(self.lib.tsr_q8_round_even(_cf(a)))

    # ---- layouts ----
    def qa_block_size(self, kind: int) -> int:
        return int(self.lib.tsr_qa_block_size(kind))

    def qa_qs_lines(self, kind: int, K: int) -> int:
        return int(self.lib.tsr_qa_qs_lines(kind, K))

    def qa_d_lines(self, kind: int, K: int) -> int:
        return int(self.lib.tsr_qa_d_lines(kind, K))

    def qa_bs_lines(self, kind: int, K: int) -> int:
        return int(self.lib.tsr_qa_bs_lines(kind, K))

    def qa_lines(self, kind: int, K: int) -> int:
        return int(self.lib.tsr_qa_lines(kind, K))

    def row_beats(self, fmt: int, K: int) -> int:
        return int(self.lib.tsr_row_beats(fmt, K))

    def rows_per_ch(self, N: int, num_ch: int) -> int:
        return int(self.lib.tsr_rows_per_ch(N, num_ch))

    # ---- QACT ----
    def qact(self, kind: int, x, K: int) -> bytes:
        """Quantize one token: x = K f32 bit patterns (list of int) or 4*K little-endian bytes.
        Returns the QA image (qa_lines(kind, K) * 64 bytes: QS, D, BS; tails zero)."""
        if isinstance(x, (bytes, bytearray, memoryview)):
            xb = bytes(x)
        else:
            xb = struct.pack("<%dI" % K, *[v & 0xFFFFFFFF for v in x])
        if len(xb) != 4 * K:
            raise ValueError("qact: need %d bytes of x, got %d" % (4 * K, len(xb)))
        bs = self.qa_block_size(kind)
        if bs == 0 or K == 0 or K % bs:
            raise ValueError("qact: K=%d is not a positive multiple of %d" % (K, bs))
        xbuf = C.create_string_buffer(xb, len(xb))
        out = C.create_string_buffer(self.qa_lines(kind, K) * 64)
        self.lib.tsr_qact(kind, xbuf, K, out)
        return out.raw

    def qa_split(self, kind: int, K: int, qa: bytes) -> dict:
        """Views of a QA image: qs (list of int8), d (list of f32 bits), bs (list of int16)."""
        qs_n = self.qa_qs_lines(kind, K) * 64
        d_n = self.qa_d_lines(kind, K) * 64
        nb = K // self.qa_block_size(kind)
        qs = list(struct.unpack_from("<%db" % K, qa, 0))
        d = list(struct.unpack_from("<%dI" % nb, qa, qs_n))
        bs = list(struct.unpack_from("<%dh" % (K // 16), qa, qs_n + d_n)) if kind == QA_A256 else []
        return dict(qs=qs, d=d, bs=bs)

    # ---- MVM helpers: weight formats and the device layout ----
    def fmt_valid(self, fmt: int) -> bool:
        return bool(self.lib.tsr_fmt_valid(fmt))

    def fmt_block_size(self, fmt: int) -> int:
        """Weights per block: Q8_0 32, Q4_K 256."""
        return int(self.lib.tsr_fmt_block_size(fmt))

    def fmt_chunk_blocks(self, fmt: int) -> int:
        """S: blocks per chunk (Q8_0 32, Q4_K 4)."""
        return int(self.lib.tsr_fmt_chunk_blocks(fmt))

    def fmt_quant_bytes(self, fmt: int) -> int:
        """QB: quant bytes per block (Q8_0 32, Q4_K 128)."""
        return int(self.lib.tsr_fmt_quant_bytes(fmt))

    def fmt_ggml_block_bytes(self, fmt: int) -> int:
        """ggml block size in bytes (Q8_0 34, Q4_K 144)."""
        return int(self.lib.tsr_fmt_ggml_block_bytes(fmt))

    def fmt_k_valid(self, fmt: int, K: int, k_max: int) -> bool:
        return bool(self.lib.tsr_fmt_k_valid(fmt, K & 0xFFFFFFFF, k_max))

    def qa_kind_for_fmt(self, fmt: int) -> int:
        return int(self.lib.tsr_qa_kind_for_fmt(fmt))

    def repack_row(self, fmt: int, ggml_blocks: bytes, K: int) -> bytes:
        """ggml blocks of one row -> the repacked row image (ROW_BEATS * 64 bytes)."""
        need = (K // self.fmt_block_size(fmt)) * self.fmt_ggml_block_bytes(fmt)
        if len(ggml_blocks) != need:
            raise ValueError("repack_row: need %d bytes of ggml blocks, got %d" % (need, len(ggml_blocks)))
        out = C.create_string_buffer(self.row_beats(fmt, K) * 64)
        self.lib.tsr_repack_row(fmt, C.create_string_buffer(bytes(ggml_blocks), len(ggml_blocks)), K, out)
        return out.raw

    def unrepack_row(self, fmt: int, row: bytes, K: int) -> bytes:
        """Inverse of repack_row: the ggml blocks of one repacked row."""
        if len(row) < self.row_beats(fmt, K) * 64:
            raise ValueError("unrepack_row: row image too short")
        out = C.create_string_buffer((K // self.fmt_block_size(fmt)) * self.fmt_ggml_block_bytes(fmt))
        self.lib.tsr_unrepack_row(fmt, C.create_string_buffer(bytes(row), len(row)), K, out)
        return out.raw

    def row_location(self, fmt: int, N: int, K: int, num_ch: int, r: int):
        """Row r -> (channel, beat offset relative to W_OFF) (spec Symmetric allocation)."""
        ch, off = C.c_uint32(), C.c_uint64()
        self.lib.tsr_row_location(fmt, N, K, num_ch, r, C.byref(ch), C.byref(off))
        return int(ch.value), int(off.value)

    def tensor_ch_beats(self, fmt: int, N: int, K: int, num_ch: int) -> int:
        """R * ROW_BEATS: beats a tensor occupies in each channel (EXP_STRIDE)."""
        return int(self.lib.tsr_tensor_ch_beats(fmt, N, K, num_ch))

    def get_scale_min_k4(self, j: int, scales: bytes):
        """ggml get_scale_min_k4 on the 12 scale bytes of a Q4_K block -> (sc_j, m_j)."""
        if len(scales) < 12:
            raise ValueError("get_scale_min_k4: need 12 scale bytes")
        sc, m = C.c_uint8(), C.c_uint8()
        self.lib.tsr_get_scale_min_k4(j, C.create_string_buffer(bytes(scales[:12]), 12), C.byref(sc), C.byref(m))
        return int(sc.value), int(m.value)

    def dequant_row(self, fmt: int, row: bytes, K: int) -> list:
        """DQROW semantics: one repacked row -> K f32 bit patterns."""
        out = (C.c_uint32 * K)()
        self.lib.tsr_dequant_row(fmt, C.create_string_buffer(bytes(row), len(row)), K, out)
        return [int(v) for v in out]

    # ---- MVM helpers: dot products (spec Numeric rules) ----
    def term_q8_0(self, row: bytes, K: int, b: int, qa: bytes) -> int:
        """Term of Q8_0 block b of a repacked row against an A32 QA image (f32 bits)."""
        return _bits(self.lib.tsr_term_q8_0(C.create_string_buffer(bytes(row), len(row)), K, b,
                                            C.create_string_buffer(bytes(qa), len(qa))))

    def term_q4_K(self, row: bytes, K: int, s: int, qa: bytes) -> int:
        """Term of Q4_K superblock s of a repacked row against an A256 QA image (f32 bits)."""
        return _bits(self.lib.tsr_term_q4_K(C.create_string_buffer(bytes(row), len(row)), K, s,
                                            C.create_string_buffer(bytes(qa), len(qa))))

    def term(self, fmt: int, row: bytes, K: int, i: int, qa: bytes) -> int:
        return self.term_q4_K(row, K, i, qa) if fmt == FMT_Q4_K else self.term_q8_0(row, K, i, qa)

    def dot_row(self, fmt: int, row: bytes, K: int, qa: bytes) -> int:
        """Full row: 16 partials + fixed tree (no ACC). f32 bits."""
        return _bits(self.lib.tsr_dot_row(fmt, C.create_string_buffer(bytes(row), len(row)), K,
                                          C.create_string_buffer(bytes(qa), len(qa))))

    def tree16(self, p) -> int:
        buf = C.create_string_buffer(struct.pack("<16I", *[x & 0xFFFFFFFF for x in p]), 64)
        return _bits(self.lib.tsr_tree16(buf))


_default: TsrRef | None = None


def ref() -> TsrRef:
    """Process-wide shared instance."""
    global _default
    if _default is None:
        _default = TsrRef()
    return _default
