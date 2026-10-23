#!/usr/bin/env python3
"""Smoke test of the libtsr_ref.so C API through ctypes (what cocotb testbenches will use).

Usage: test_ref_ctypes.py <path/to/libtsr_ref.so>

Checks the API is callable from Python and that results agree with small independent Python
computations (f32 arithmetic emulated with struct round trips: one double operation followed
by one rounding to binary32 is correctly rounded for +, -, *).
"""
import ctypes as C
import random
import struct
import sys

TSR_FMT_Q8_0, TSR_FMT_Q4_K = 0, 1
TSR_QA_A32, TSR_QA_A256 = 0, 1


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def load(path):
    lib = C.CDLL(path)
    u32, i32, f, vp, bp = C.c_uint32, C.c_int32, C.c_float, C.c_void_p, C.c_char_p
    sig = {
        "tsr_row_beats": (u32, [u32, u32]),
        "tsr_qa_lines": (u32, [u32, u32]),
        "tsr_qa_qs_lines": (u32, [u32, u32]),
        "tsr_fp16_to_f32": (f, [C.c_uint16]),
        "tsr_f32_to_fp16": (C.c_uint16, [f]),
        "tsr_fadd": (f, [f, f]),
        "tsr_fmul": (f, [f, f]),
        "tsr_i32_to_f32": (f, [i32]),
        "tsr_f32_bits": (u32, [f]),
        "tsr_q8_round_even": (C.c_int8, [f]),
        "tsr_repack_row": (None, [u32, vp, u32, vp]),
        "tsr_unrepack_row": (None, [u32, vp, u32, vp]),
        "tsr_dequant_row": (None, [u32, vp, u32, vp]),
        "tsr_qact": (None, [u32, vp, u32, vp]),
        "tsr_term_q8_0": (f, [vp, u32, u32, vp]),
        "tsr_dot_row": (f, [u32, vp, u32, vp]),
        "tsr_tree16": (f, [vp]),
    }
    for name, (res, args) in sig.items():
        fn = getattr(lib, name)
        fn.restype, fn.argtypes = res, args
    return lib


def main():
    lib = load(sys.argv[1])
    fails = 0

    def check(cond, msg):
        nonlocal fails
        if not cond:
            fails += 1
            print("FAIL:", msg)

    # Sizes from the spec examples.
    check(lib.tsr_row_beats(TSR_FMT_Q4_K, 27648) == 243, "ROW_BEATS Q4_K 27648")
    check(lib.tsr_row_beats(TSR_FMT_Q8_0, 5120) == 85, "ROW_BEATS Q8_0 5120")
    check(lib.tsr_qa_lines(TSR_QA_A256, 27648) == 493, "QA_LINES A256 27648")

    # fp helpers.
    check(lib.tsr_fp16_to_f32(0x3C00) == 1.0, "fp16 1.0")
    check(lib.tsr_f32_to_fp16(65520.0) == 0x7C00, "fp16 overflow")
    check(lib.tsr_f32_bits(lib.tsr_fmul(float("inf"), 0.0)) == 0x7FC00000, "canonical NaN")
    check(lib.tsr_f32_bits(lib.tsr_fadd(2.0 ** -126 * 1.5, -(2.0 ** -126))) == 0, "FTZ")
    # FTZ window (spec 구현 명확화 "FTZ 판정 시점"): exact 2^-126 - 2^-150 rounds to 2^-126 and is
    # kept (an IEEE tininess-after-rounding / x86 FTZ datapath gives 0); 2^-126 - 2^-149 -> 0.
    check(lib.tsr_f32_bits(lib.tsr_fmul(1.0 - 2.0 ** -24, 2.0 ** -126)) == 0x00800000, "FTZ window edge")
    check(lib.tsr_f32_bits(lib.tsr_fmul(1.0 - 2.0 ** -23, 2.0 ** -126)) == 0, "FTZ below window")
    check(lib.tsr_i32_to_f32(16777217) == 16777216.0, "i32->f32 RNE")
    check(lib.tsr_q8_round_even(2.5) == 2, "round even")

    rnd = random.Random(1234)

    # Q8_0 row: repack -> unrepack round trip, dequant vs Python decode.
    K = 32 * 37
    nb = K // 32
    blocks = bytearray()
    for _ in range(nb):
        d = lib.tsr_f32_to_fp16(rnd.uniform(0.001, 0.02))
        blocks += struct.pack("<H", d) + bytes(rnd.randrange(256) for _ in range(32))
    blocks = bytes(blocks)
    rb = lib.tsr_row_beats(TSR_FMT_Q8_0, K)
    row = C.create_string_buffer(rb * 64)
    lib.tsr_repack_row(TSR_FMT_Q8_0, blocks, K, row)
    back = C.create_string_buffer(len(blocks))
    lib.tsr_unrepack_row(TSR_FMT_Q8_0, row, K, back)
    check(back.raw == blocks, "repack round trip")
    # Spec placement: meta bytes [2b, 2b+2) of chunk 0 = d of block b.
    check(row.raw[2 * 5:2 * 5 + 2] == blocks[34 * 5:34 * 5 + 2], "meta placement")

    deq = (C.c_float * K)()
    lib.tsr_dequant_row(TSR_FMT_Q8_0, row, K, deq)
    for i in range(nb):
        d = lib.tsr_fp16_to_f32(struct.unpack_from("<H", blocks, 34 * i)[0])
        for j in range(32):
            q = struct.unpack_from("<b", blocks, 34 * i + 2 + j)[0]
            if bits(deq[32 * i + j]) != bits(f32(d * q)):
                check(False, "dequant %d %d" % (i, j))

    # QACT A32 + dot: terms and the 16-partial tree recomputed in Python.
    x = (C.c_float * K)(*[rnd.gauss(0.0, 1.0) for _ in range(K)])
    qa = C.create_string_buffer(lib.tsr_qa_lines(TSR_QA_A32, K) * 64)
    lib.tsr_qact(TSR_QA_A32, x, K, qa)
    qs_off = 0
    d_off = lib.tsr_qa_qs_lines(TSR_QA_A32, K) * 64
    p = [0.0] * 16
    for i in range(nb):
        t_lib = lib.tsr_term_q8_0(row, K, i, qa)
        dw = lib.tsr_fp16_to_f32(struct.unpack_from("<H", blocks, 34 * i)[0])
        dx = struct.unpack_from("<f", qa.raw, d_off + 4 * i)[0]
        isum = sum(struct.unpack_from("<b", blocks, 34 * i + 2 + j)[0] *
                   struct.unpack_from("<b", qa.raw, qs_off + 32 * i + j)[0] for j in range(32))
        t_py = f32(f32(dw * dx) * f32(float(isum)))
        check(bits(t_lib) == bits(t_py), "term %d" % i)
        p[i % 16] = f32(p[i % 16] + t_lib)
    l1 = [f32(p[2 * k] + p[2 * k + 1]) for k in range(8)]
    l2 = [f32(l1[2 * k] + l1[2 * k + 1]) for k in range(4)]
    l3 = [f32(l2[2 * k] + l2[2 * k + 1]) for k in range(2)]
    y_py = f32(l3[0] + l3[1])
    y_lib = lib.tsr_dot_row(TSR_FMT_Q8_0, row, K, qa)
    check(bits(y_lib) == bits(y_py), "dot row %r vs %r" % (y_lib, y_py))
    pa = (C.c_float * 16)(*p)
    check(bits(lib.tsr_tree16(pa)) == bits(y_py), "tree16")

    if fails:
        print("test_ref_ctypes: %d FAILED" % fails)
        return 1
    print("test_ref_ctypes: all checks passed (y = %r)" % y_lib)
    return 0


if __name__ == "__main__":
    sys.exit(main())
