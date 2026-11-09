"""fp library testbench — block (c), docs/rtl-core.md §8.2 (rtl/core/tsr_fp_pkg.sv, tsr_fp_mul.sv,
tsr_fp_div.sv) and block (fp), docs/rtl-mvm.md §8 (rtl/core/tsr_fp_add.sv). Oracle: libtsr_ref.so
(sw/sim/tsr_ref.c) through common.tsr_ref, bit-exact.

pytest functions (test_*):
  test_fp_directed_vectors_match_ref
                        the hand-derived expectations of MUL_DIRECTED / DIV_DIRECTED / ADD_DIRECTED
                        (spec mandatory vectors) agree with libtsr_ref; fadd commutes and fsub(a, b)
                        = fadd(a, -b) on the generated add vectors
  test_fp_lint          verilator --lint-only -Wall: tsr_fp_mul and tsr_fp_add at LAT 1..8,
                        tsr_fp_div, the testbench wrappers (rtl/lint.sh covers LAT 1 / 8 / default)
                        incl. tsr_fp_defaults_tb (with tsr_pkg.sv)
  test_fp_funcs         cocotb on tb/fp/hdl/tsr_fp_funcs_tb.sv (16 lanes): fp16->f32 exhaustive
                        (65536), f32->fp16 / q8 away / q8 even / i32->f32 / daz / is_nan / amag /
                        amax_cand on directed vectors, every exponent x random mantissas, every
                        fp16 rounding midpoint +-1 ulp, every q8 half-integer +-1 ulp; amax/max
                        tree + chain reductions vs the reference scan and through tsr_qact
  test_fp_mul[LAT]      cocotb on tsr_fp_mul, LAT 1..6: stage allocation (fp_mul_stages: [IN] holds
                        the raw operands from LAT 2, product registers 1/2/2/3 at LAT 3/4/5/6,
                        [NORM] from LAT 5); spec mandatory
                        vectors, test_ref_fp.c edge cases, random classes (FTZ window, underflow,
                        overflow, ties, recip, QACT operands); random ce stalls and in_valid gaps
                        against a cycle model
  test_fp_div           cocotb on tsr_fp_div: same vector classes; random input gaps and
                        out_ready backpressure; checks R2 (out_valid / y hold), latency 14,
                        in_ready timing, operand latching
  test_fp_add[LAT]      cocotb on tsr_fp_add, LAT 1..8: fp_add_stages (cut registers per LAT,
                        [IN] = raw operands), fp_add_vectors (ADD_DIRECTED + add_vectors: every
                        exponent difference 0..40 / 64 / 126 / 150 / 253, RNE ties at every
                        alignment distance 1..24 with +-1 ulp neighbours, cancellation, FTZ /
                        2^-126 boundary, overflow, carry, specials; ~32 k results; random ce incl.
                        long ce = 0 bursts and in_valid gaps against a LAT-deep cycle model,
                        out_valid / y hold while ce = 0), fp_add_throughput (512 back-to-back
                        operations: first result after exactly LAT edges, then one per cycle),
                        fp_add_reset (reset with a full pipeline,
                        with ce 0 or 1, clears every valid stage)
  test_fp_add_dsp[LAT,NORM_DSP]
                        the same four cocotb tests on tsr_fp_add_dsp (rtl/core/tsr_fp_add_dsp.sv, the
                        measurement prototype with the far-path alignment shift, and with NORM_DSP the
                        near-path left shift, as DSP48E2 multiplications by a one-hot power of two): LAT 5..8,
                        NORM_DSP 0, and LAT 5 / 8 with NORM_DSP 1; bit-exact against libtsr_ref
  test_fp_default_lats  cocotb on tb/fp/hdl/tsr_fp_defaults_tb.sv: the measured latency of
                        tsr_fp_mul / tsr_fp_add at their default LAT equals MUL_LAT_U200 /
                        ADD_LAT_U200 (the LATs swept and run out of context below) and
                        tsr_pkg::TSR_DEF_FMUL_LAT / TSR_DEF_FADD_LAT (what the MVM blocks pass);
                        the only fp test that also reads rtl/core/tsr_pkg.sv
  test_fp_sweep_quick   Verilator C++ harness (tb/fp/cpp/fp_sweep.cpp, linked with libtsr_ref):
                        6 M inputs through every conversion (all 65536 halves among them),
                        int32 -> f32 exhaustively on [-2^27, 2^27], 200 k amax blocks, 1.3 M
                        products at LAT 4 and at LAT 5, 0.65 M quotients, 2.6 M sums at the
                        default adder LAT and 0.65 M at LAT 1 (10 vector classes each, see
                        fp_sweep.cpp; the add mode also checks coverage floors), ~30 s
  test_fp_sweep_add_dsp Verilator C++ harness with +define+TSR_FP_ADD_DSP=1 / 2 (tsr_fp_add becomes a wrapper of
                        tsr_fp_add_dsp): 2 x 2.6 M sums at LAT 5, 0.65 M at LAT 8, 3.9 M with NORM_DSP, ~30 s
  test_fp_slow_gate     the opt-in rule of the slow tests below (fast; no simulation)
  test_fp_sweep_full    (slow, opt-in) exhaustive: all 2^32 f32 / int32 patterns through every
                        conversion; 2^23 products (LAT 4) and 2^23 quotients per class (84 M each),
                        so the edge classes walk every mantissa at the FTZ / min-normal decision,
                        the overflow boundary and the reciprocal; 4 M amax blocks; 10^8 sums /
                        differences at the default adder LAT (its four runs walk 2^21 * 3 + 2.5 M
                        consecutive indices: every 23-bit delta of the walk class) and 2 x 10^7 at
                        LAT 1 and 3; ~6 min
                        with TSR_FP_SWEEP_PROCS (default 4) processes
  test_fp_ooc_timing[]  (slow, opt-in, needs Vivado: TSR_VIVADO=<path to vivado>) out-of-context
                        implementation at 300 MHz on the U200 part (tb/fp/viv/fp_ooc.tcl): WNS >= 0
                        and the logic-depth budget of docs/rtl-core.md §1 (LUT levels per path) for
                        tsr_fp_mul and tsr_fp_add at their default LAT and tsr_fp_div; ~2 min and
                        ~3 GB per run
Slow tests are opt-in because tb/pytest.ini does not deselect the `slow` marker: they run only
with `-m slow` (any -m expression naming `slow`), with TSR_FP_SLOW=1, or when named on the
command line (node id, or -k with the name, e.g. -k sweep_full); otherwise they are reported as
skipped (no build, no Vivado).
cocotb tests (fp_*) are selected by the pytest functions through common.runner.run(testcase=).
Builds use only the fp sources (not the whole filelist), so work in progress elsewhere in rtl/
cannot break this testbench; the exception is test_fp_default_lats (and its lint run), which reads
rtl/core/tsr_pkg.sv on purpose.
"""
from __future__ import annotations

import os
import random
import re
import struct
import subprocess
import sys
from collections import deque
from pathlib import Path

import cocotb
import pytest
from cocotb.triggers import FallingEdge, RisingEdge, Timer

from common.runner import REPO, build_root, run
from common.tbutil import rng, start_clock_and_reset
from common.tsr_ref import QA_A32, QA_A256, ref

CANON = 0x7FC00000
FP_SOURCES = [REPO / "rtl" / "core" / f
              for f in ("tsr_fp_pkg.sv", "tsr_fp_mul.sv", "tsr_fp_div.sv", "tsr_fp_add.sv")]
FP_DSP_SOURCES = FP_SOURCES + [REPO / "rtl" / "core" / "tsr_fp_add_dsp.sv"]
HDL = REPO / "tb" / "fp" / "hdl"
FUNCS_SV = HDL / "tsr_fp_funcs_tb.sv"
SWEEP_SV = HDL / "tsr_fp_sweep_tb.sv"
DEFAULTS_SV = HDL / "tsr_fp_defaults_tb.sv"
PKG_SV = REPO / "rtl" / "core" / "tsr_pkg.sv"   # only for the default-LAT check
TABLES_SV = REPO / "rtl" / "core" / "tsr_tables_pkg.sv"   # tsr_pkg takes the VERSION fields from it
SWEEP_CPP = REPO / "tb" / "fp" / "cpp" / "fp_sweep.cpp"
OOC_TCL = REPO / "tb" / "fp" / "viv" / "fp_ooc.tcl"
MUL_LAT_U200 = 4          # tsr_fp_mul default LAT: the 300 MHz U200 configuration
ADD_LAT_U200 = 5          # tsr_fp_add default LAT: the 300 MHz U200 configuration
ADD_LATS = list(range(1, 9))
ADD_DSP_LATS = list(range(5, 9))   # tsr_fp_add_dsp: the DSP sits between [C1] and [C2]
ADD_DSP_CFGS = [(lat, 0) for lat in ADD_DSP_LATS] + [(5, 1), (8, 1)]   # (LAT, NORM_DSP)
LANES = 16
DIV_LATENCY = 14          # out_valid after edge n + 14 for an operation accepted at edge n

LO = 2.0 ** -126 - 2.0 ** -150   # FTZ threshold: |exact| < LO -> signed zero


# =============================================================================================
# Vectors
# =============================================================================================
def f32(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def f32val(u: int) -> float:
    return struct.unpack("<f", struct.pack("<I", u & 0xFFFFFFFF))[0]


def mk(s: int, e: int, m: int) -> int:
    return ((s & 1) << 31) | ((e & 0xFF) << 23) | (m & 0x7FFFFF)


# (a, b, expected) — expected from the spec (Implementation clarifications) / sw/sim/tests/test_ref_fp.c; every one
# is also checked against libtsr_ref (test_fp_directed_vectors_match_ref).
MUL_DIRECTED = [
    (0x3F7FFFFF, 0x00800000, 0x00800000),   # spec mandatory: window edge rounds up to 2^-126
    (0x3F1FA0BA, 0x00CD4708, 0x00800000),   # spec mandatory: window interior
    (0x3F7FFFFE, 0x00800000, 0x00000000),   # spec mandatory: exactly subnormal -> flushed
    (0xBF7FFFFF, 0x00800000, 0x80800000),
    (0xBF7FFFFE, 0x00800000, 0x80000000),
    (0x00000001, 0x7F000000, 0x00000000),   # DAZ: sub * huge
    (0x80000001, 0x7F000000, 0x80000000),
    (0x1F800000, 0x20000000, 0x00000000),   # 2^-64 * 2^-63 -> 2^-127 flushed
    (0x9F800000, 0x20000000, 0x80000000),
    (0x20000000, 0x20000000, 0x00800000),   # exactly 2^-126
    (0x00000000, 0xBF800000, 0x80000000),   # +0 * -1
    (0xFF800000, 0x40000000, 0xFF800000),
    (0x7F7FFFFF, 0x40000000, 0x7F800000),   # max * 2
    (0xFF7FFFFF, 0x40000000, 0xFF800000),
    (0x00000000, 0x7F800000, CANON),        # 0 * inf
    (0x00000001, 0x7F800000, CANON),        # DAZ makes it 0 * inf
    (0xFFC12345, 0x40000000, CANON),        # -qNaN payload
    (0x7F800001, 0x3F800000, CANON),        # sNaN
    (0x3F800000, 0x7FBFFFFF, CANON),
    (0x7F800000, 0xFF800000, 0xFF800000),
    (0x80000000, 0x80000000, 0x00000000),
    (0x3F800000, 0x3F800000, 0x3F800000),
    (0x3F800001, 0x3F800001, 0x3F800002),   # (1+u)^2 = 1+2u+u^2 -> rounds down to 1+2u
    (0x3FFFFFFF, 0x3FFFFFFF, 0x407FFFFE),   # (2-u)^2 -> 4 - 4u + u^2 -> 0x407FFFFE
    (0x7F000000, 0x40000000, 0x7F800000),   # 2^127 * 2 -> Inf
    (0x7F000000, 0x3FFFFFFF, 0x7F7FFFFF),   # 2^127 * (2 - 2^-23) = FLT_MAX exactly
]
DIV_DIRECTED = [
    (0x00FFFFFF, 0x40000000, 0x00800000),   # spec mandatory: (2^-125 - 2^-149) / 2 -> 2^-126
    (0x00FFFFFE, 0x40000000, 0x00000000),   # one ulp less -> flushed
    (0x3F800000, 0x00000001, 0x7F800000),   # 1 / sub = 1 / +0
    (0x3F800000, 0x80000001, 0xFF800000),
    (0x3F800000, 0x7F000000, 0x00000000),   # recip 2^127 -> 2^-127 flushed
    (0x3F800000, 0xFF000000, 0x80000000),
    (0x3F800000, 0x7E800000, 0x00800000),   # recip 2^126
    (0x00800000, 0x40000000, 0x00000000),   # min normal / 2
    (0x3F800000, 0x40400000, 0x3EAAAAAB),   # 1/3
    (0x3F800000, 0x7F800000, 0x00000000),   # 1/inf
    (0x3F800000, 0xFF800000, 0x80000000),
    (0x3F800000, 0x00000000, 0x7F800000),   # 1/0
    (0xBF800000, 0x00000000, 0xFF800000),
    (0x00000000, 0x00000000, CANON),        # 0/0
    (0x7F800000, 0xFF800000, CANON),        # inf/inf
    (0x3F800000, 0xFFFFFFFF, CANON),        # recip NaN
    (0x7FC00001, 0x3F800000, CANON),
    (0x3F800000, 0x40800000, 0x3E800000),   # 1/4
    (0x7F800000, 0x00000000, 0x7F800000),   # inf/0
    (0xFF800000, 0x3F800000, 0xFF800000),
    (0x00000000, 0x7F800000, 0x00000000),   # 0/inf
    (0x80000000, 0x3F800000, 0x80000000),
    (0x00000001, 0x00000001, CANON),        # DAZ: 0/0
    (0x42FE0000, 0x42FE0000, 0x3F800000),   # 127/127
    (0xC2FE0000, 0x3F800000, 0xC2FE0000),   # -127/1
    (0x7F7FFFFF, 0x3F7FFFFF, 0x7F800000),   # FLT_MAX / (1 - 2^-24) = 2^128 -> Inf
    (0x7F7FFFFF, 0x3F800000, 0x7F7FFFFF),
    (0x00800000, 0x3F800001, 0x00000000),   # 2^-126 / (1 + 2^-23) < 2^-126 - 2^-150 -> 0
    (0x00800001, 0x3F800001, 0x00800000),   # exactly 2^-126
]

# (a, b, sub, expected): y = a + b (sub = 0) / a - b (sub = 1); expectations derived by hand from
# the spec rules (docs/rtl-mvm.md §8.1 mandatory vectors) and checked against libtsr_ref below.
ADD_DIRECTED = [
    # exact subnormal results are flushed with their sign (spec Implementation clarifications, "FTZ decision point")
    (0x00C00000, 0x00800000, 1, 0x00000000),   # mandatory: 1.5 * 2^-126 - 2^-126 -> +0
    (0x80C00000, 0x80800000, 1, 0x80000000),   # mandatory: its negation -> -0
    (0x00800000, 0x00C00000, 1, 0x80000000),
    (0x80800000, 0x00C00000, 0, 0x00000000),
    (0x00FFFFFF, 0x00800000, 1, 0x00000000),   # (1 - 2^-23) * 2^-126
    (0x80FFFFFF, 0x00800000, 0, 0x80000000),
    (0x01000000, 0x00FFFFFF, 1, 0x00000000),   # 2^-149
    (0x00800001, 0x00800000, 1, 0x00000000),
    (0x00800000, 0x00800001, 1, 0x80000000),
    # results at the normal boundary
    (0x01000000, 0x00800000, 1, 0x00800000),   # 2^-125 - 2^-126 = 2^-126 exactly
    (0x01000001, 0x00800000, 1, 0x00800002),
    (0x01400000, 0x00800000, 1, 0x01000000),
    (0x00800000, 0x00800000, 0, 0x01000000),
    # exact cancellation: x - x = +0 under RNE, whatever the signs
    (0x3F800000, 0x3F800000, 1, 0x00000000),
    (0xBF800000, 0xBF800000, 1, 0x00000000),
    (0x3F800000, 0xBF800000, 0, 0x00000000),
    (0xBF800000, 0x3F800000, 0, 0x00000000),
    (0x7F7FFFFF, 0xFF7FFFFF, 0, 0x00000000),
    (0x00800000, 0x80800000, 0, 0x00000000),
    # signed zeros
    (0x80000000, 0x80000000, 0, 0x80000000),   # mandatory: -0 + -0 = -0
    (0x00000000, 0x80000000, 0, 0x00000000),
    (0x80000000, 0x00000000, 0, 0x00000000),
    (0x80000000, 0x00000000, 1, 0x80000000),   # -0 - +0 = -0
    (0x00000000, 0x00000000, 1, 0x00000000),
    (0x80000000, 0x80000000, 1, 0x00000000),
    (0x00000000, 0x80000000, 1, 0x00000000),
    # DAZ: subnormal operands are signed zeros (their sign is kept)
    (0x80000001, 0x80000002, 0, 0x80000000),
    (0x80000001, 0x007FFFFF, 1, 0x80000000),
    (0x00000001, 0x80000001, 0, 0x00000000),
    (0x807FFFFF, 0x3F800000, 0, 0x3F800000),
    (0x3F800000, 0x807FFFFF, 1, 0x3F800000),
    (0x00000001, 0xBF800000, 1, 0x3F800000),
    (0x00400000, 0x00800000, 0, 0x00800000),
    # Inf
    (0x7F800000, 0x7F800000, 1, CANON),        # mandatory: Inf - Inf
    (0xFF800000, 0x7F800000, 0, CANON),
    (0x7F800000, 0xFF800000, 0, CANON),
    (0xFF800000, 0xFF800000, 1, CANON),
    (0x7F800000, 0x7F800000, 0, 0x7F800000),
    (0xFF800000, 0xFF800000, 0, 0xFF800000),
    (0x7F800000, 0xFF800000, 1, 0x7F800000),
    (0xFF800000, 0x3F800000, 0, 0xFF800000),
    (0x3F800000, 0xFF800000, 1, 0x7F800000),
    (0x00000001, 0x7F800000, 1, 0xFF800000),
    (0x7F7FFFFF, 0x7F800000, 1, 0xFF800000),
    # NaN payloads -> canonical NaN
    (0x7FC12345, 0x3F800000, 0, CANON),
    (0xFFFFFFFF, 0x00000000, 1, CANON),
    (0x3F800000, 0x7F800001, 0, CANON),
    (0x7F800001, 0x7F800000, 0, CANON),
    (0xFF812345, 0xFF800000, 1, CANON),
    (0x7FC00000, 0x7FC00000, 0, CANON),
    (0x00000000, 0xFFC00001, 1, CANON),
    (0x7FBFFFFF, 0xFF800000, 0, CANON),
    # overflow to Inf (RNE), FLT_MAX ties
    (0x7F7FFFFF, 0x7F7FFFFF, 0, 0x7F800000),
    (0xFF7FFFFF, 0xFF7FFFFF, 0, 0xFF800000),
    (0x7F7FFFFF, 0xFF7FFFFF, 1, 0x7F800000),
    (0x7F000000, 0x7F000000, 0, 0x7F800000),   # 2^127 + 2^127 = 2^128 (exact carry)
    (0x7F7FFFFF, 0x73000000, 0, 0x7F800000),   # FLT_MAX + half ulp: tie, odd -> up -> Inf
    (0x7F7FFFFF, 0x72FFFFFF, 0, 0x7F7FFFFF),   # just below the tie
    (0x7F7FFFFE, 0x73000000, 0, 0x7F7FFFFE),   # tie, even -> stays
    (0x7F000000, 0x7EFFFFFF, 0, 0x7F800000),   # 2^128 - 2^103: tie above FLT_MAX -> Inf
    # carry into the next binade, ties at 1.0 and below
    (0x3FFFFFFF, 0x34000000, 0, 0x40000000),   # 2 - 2^-23 + 2^-23 = 2 (exact)
    (0x3FFFFFFF, 0x33800000, 0, 0x40000000),   # tie, odd -> up into the next binade
    (0x3FFFFFFE, 0x33800000, 0, 0x3FFFFFFE),   # tie, even -> stays
    (0x3F800000, 0x33800000, 0, 0x3F800000),   # 1 + 2^-24: tie -> 1
    (0x3F800001, 0x33800000, 0, 0x3F800002),   # tie, odd -> up
    (0x3F800000, 0x33800001, 0, 0x3F800001),   # above the tie
    (0x3F800000, 0x33800000, 1, 0x3F7FFFFF),   # 1 - 2^-24 exact (one-bit left normalisation)
    (0x3F800000, 0x33000000, 1, 0x3F800000),   # 1 - 2^-25: tie below 1 -> 1 (even)
    (0x3F800000, 0x33000001, 1, 0x3F7FFFFF),   # just above the tie -> down
    # operands 2^-126 / 2^126 apart, d >= 26 (everything in the sticky bit)
    (0x3F800000, 0x00800000, 0, 0x3F800000),
    (0x3F800000, 0x00800000, 1, 0x3F800000),
    (0x00800000, 0x3F800000, 1, 0xBF800000),
    (0x7F000000, 0x00800000, 1, 0x7F000000),   # d = 253
    (0x3F800001, 0x00800000, 1, 0x3F800001),
    (0x4B800000, 0x3F800000, 1, 0x4B7FFFFF),   # 2^24 - 1 exact
    (0x4B800000, 0x3F000000, 1, 0x4B800000),   # 2^24 - 0.5: tie -> 2^24 (even)
    # massive cancellation (near path)
    (0x3F800001, 0x3F800000, 1, 0x34000000),   # 2^-23
    (0x4B000001, 0x4B000000, 1, 0x3F800000),
    (0x40000000, 0x3FFFFFFF, 1, 0x34000000),   # d = 1: 2 - (2 - 2^-23)
    (0x3FC00000, 0x3FA00000, 1, 0x3E800000),   # 1.5 - 1.25 = 0.25
    (0x40000000, 0x3FC00000, 1, 0x3F000000),   # d = 1: 2 - 1.5 = 0.5
    (0x40400000, 0x3F800000, 1, 0x40000000),   # d = 1, no cancellation: 3 - 1 = 2
]


def _rnd_mant(r: random.Random) -> int:
    k = r.randrange(8)
    return [0, 0x7FFFFF, 0x7FFFFF - r.randrange(4), r.randrange(8), 1 << r.randrange(23),
            0x7FFFFF ^ (1 << r.randrange(23)), r.getrandbits(23), r.getrandbits(23)][k]


def _rnd_any(r: random.Random) -> int:
    k = r.randrange(4)
    if k == 0:
        return r.getrandbits(32)
    if k == 1:
        return mk(r.getrandbits(1), r.choice([0, 1, 2, 126, 127, 128, 253, 254, 255]), _rnd_mant(r))
    return mk(r.getrandbits(1), r.randrange(256), _rnd_mant(r))


def _rnd_normal(r: random.Random, lo: int = 1, hi: int = 254) -> int:
    return mk(r.getrandbits(1), r.randint(lo, hi), _rnd_mant(r) if r.getrandbits(1) else r.getrandbits(23))


def _near(r: random.Random, target: float, div: bool):
    """(a, b): the exact a*b (mul) or a/b (div) lies within a few ulps of `target`."""
    for _ in range(100):
        b = _rnd_normal(r, 127, 250) if div else _rnd_normal(r, 1, 126)
        bv = abs(f32val(b))
        av = target * bv if div else target / bv
        if 2.0 ** -126 <= av < 2.0 ** 127:
            a = (f32(av) + r.randint(-1, 1)) | (r.getrandbits(1) << 31)
            return a, b
    return _rnd_normal(r), _rnd_normal(r)


def pair_vectors(r: random.Random, n_per_class: int, div: bool):
    out = []
    for _ in range(n_per_class):
        out.append((_rnd_any(r), _rnd_any(r)))
        out.append((_rnd_normal(r), _rnd_normal(r)))
        out.append(_near(r, LO * (1 + r.randint(-4, 4) * 2.0 ** -27), div))           # FTZ window
        out.append(_near(r, 2.0 ** r.randint(-128, -124) * (1 + r.random()), div))    # underflow
        out.append(_near(r, 2.0 ** 128 * (1 + r.randint(-6, 2) * 2.0 ** -25), div))   # overflow
        out.append((r.choice([0x3F800000, 0xC2FE0000, 0x42FE0000]), _rnd_any(r)))   # recip / QACT
        x = _rnd_any(r)
        if div:
            out.append((x & 0x7FFFFFFF, 0x42FE0000))                                # amax / 127
        else:
            out.append((x, ref().frecip(ref().fdiv(_rnd_normal(r) & 0x7FFFFFFF, 0x42FE0000))))  # x * id
        if not div:   # short significands: exact products and exact ties
            out.append((mk(r.getrandbits(1), r.randint(64, 191), r.getrandbits(12) << 11),
                        mk(r.getrandbits(1), r.randint(64, 191), r.getrandbits(13) << 10)))
    return out


def add_vectors(r: random.Random, n_per_class: int):
    """(a, b, sub) triples for tsr_fp_add, the classes of fp_sweep.cpp `add` in small numbers plus
    every exponent difference 0..30 / 31..40 / 64 / 126 / 253 and RNE ties at every alignment
    distance 1..24 (with +-1 ulp neighbours), for both effective operations."""
    out = []

    def push(a, b, sub=None):
        out.append((a & 0xFFFFFFFF, b & 0xFFFFFFFF, r.getrandbits(1) if sub is None else sub))

    def eff(a, b, want_sub):
        return ((a >> 31) ^ (b >> 31) ^ int(want_sub)) & 1

    for d in list(range(0, 41)) + [64, 126, 150, 253]:
        for _ in range(max(1, n_per_class // 50)):
            ea = r.randint(d + 1, 254) if d < 254 else 254
            a = mk(r.getrandbits(1), ea, _rnd_mant(r) if r.getrandbits(1) else r.getrandbits(23))
            b = mk(r.getrandbits(1), ea - d, _rnd_mant(r) if r.getrandbits(1) else r.getrandbits(23))
            push(*((a, b) if r.getrandbits(1) else (b, a)))
    for d in range(1, 25):
        for _ in range(max(1, n_per_class // 100)):
            ea = r.randint(d + 1, 254)
            mb = (((r.getrandbits(24) << d) | (1 << (d - 1))) & 0xFFFFFF) | 0x800000
            a = mk(r.getrandbits(1), ea, r.getrandbits(23))
            b = mk(r.getrandbits(1), ea - d, mb)
            for nb in (b, b + 1, b - 1):
                for ws in (0, 1):
                    push(a, nb, eff(a, nb, ws))
    for _ in range(n_per_class):
        push(r.getrandbits(32), r.getrandbits(32))
        push(_rnd_any(r), _rnd_any(r))
        # near-equal magnitudes, mostly effective subtraction (cancellation of every depth)
        a = mk(r.getrandbits(1), r.randint(1, 254), r.getrandbits(23))
        m = (a & 0x7FFFFFFF) + r.choice([-1, 1]) * (((1 + (r.getrandbits(32) >> r.randrange(32)))) >> r.randrange(24))
        if 0x00800000 <= m <= 0x7F7FFFFF:
            b = m | (r.getrandbits(1) << 31)
            push(a, b, eff(a, b, r.random() < 0.875))
        # tiny exponents: exact subnormal differences (FTZ) and the 2^-126 boundary
        a = mk(r.getrandbits(1), r.randint(1, 4), _rnd_mant(r))
        b = mk(r.getrandbits(1), r.randint(1, 4), _rnd_mant(r))
        push(a, b, eff(a, b, r.random() < 0.75))
        # overflow: exponents 254 / 225..254, mostly effective addition
        a = mk(r.getrandbits(1), 254, 0x7FFFFF - r.randrange(8) if r.getrandbits(1) else r.getrandbits(23))
        b = mk(r.getrandbits(1), 254 - r.randrange(30), _rnd_mant(r))
        push(a, b, eff(a, b, r.random() < 0.125))
        # carry: significands near all ones plus a small addend
        ea = r.randint(27, 254)
        a = mk(r.getrandbits(1), ea, 0x7FFFFF - r.randrange(64))
        b = mk(r.getrandbits(1), ea - r.randrange(27), _rnd_mant(r))
        push(a, b, eff(a, b, r.random() < 0.125))
    return out


def test_fp_directed_vectors_match_ref():
    """The hard-coded expectations above agree with the reference library."""
    r = ref()
    for a, b, want in MUL_DIRECTED:
        assert r.fmul(a, b) == want, ("fmul", hex(a), hex(b))
    for a, b, want in DIV_DIRECTED:
        assert r.fdiv(a, b) == want, ("fdiv", hex(a), hex(b))
    for a, b, sub, want in ADD_DIRECTED:
        got = r.fsub(a, b) if sub else r.fadd(a, b)
        assert got == want, ("fsub" if sub else "fadd", hex(a), hex(b), hex(got), hex(want))
    # fadd is commutative bit-exactly and fsub(a, b) = fadd(a, -b) for non-NaN b (both used by
    # the MVM unit: operand order of ACC / tree adds, Q4_K fsub)
    rr = rng(0x7A1)
    for a, b, _ in add_vectors(rr, 200):
        assert r.fadd(a, b) == r.fadd(b, a), (hex(a), hex(b))
        if (b & 0x7FFFFFFF) <= 0x7F800000:
            assert r.fsub(a, b) == r.fadd(a, b ^ 0x80000000), (hex(a), hex(b))


# =============================================================================================
# cocotb: tsr_fp_mul
# =============================================================================================
def _has_scope(dut, name: str) -> bool:
    try:
        getattr(dut, name)
        return True
    except AttributeError:
        return False


@cocotb.test()
async def fp_mul_stages(dut):
    """Stage allocation of tsr_fp_mul (its header, "Pipeline"), the structure behind its measured
    timing (review finding: a 10-level round stage at LAT 3 and 4): [IN] holds the raw operands
    from LAT 2 (no logic in front of the module's input flops; the old classification in front of
    [IN] fails here), product registers (DSP P, then M) 1 / 2 / 2 / 3 at LAT 3 / 4 / 5 / 6,
    [NORM] from LAT 5. The logic depth itself needs Vivado: test_fp_ooc_timing (opt-in)."""
    lat = int(os.environ["TSR_FP_LAT"])
    r = rng(0x311 + lat)
    want = {"g_in": lat >= 2, "g_mul": lat >= 3, "g_norm": lat >= 5}
    got = {k: _has_scope(dut, k) for k in want}
    assert got == want, "LAT %d: stage registers %s, expected %s" % (lat, got, want)
    if lat >= 3:
        n_mul = {3: 1, 4: 2, 5: 2}.get(lat, lat - 3)
        assert len(dut.g_mul.p_q) == n_mul, "LAT %d: %d product registers, expected %d" % (
            lat, len(dut.g_mul.p_q), n_mul)
    dut.ce.value = 1
    dut.in_valid.value = 0
    dut.a.value = 0
    dut.b.value = 0
    await start_clock_and_reset(dut)
    if lat < 2:
        return
    for i in range(64):
        a, b = (_rnd_any(r), _rnd_any(r)) if i else (0xFFFFFFFF, 0x7F800000)
        ce = r.random() < 0.7
        dut.ce.value = int(ce)
        dut.a.value = a
        dut.b.value = b
        prev = (int(dut.g_in.a_q.value), int(dut.g_in.b_q.value)) if i else None
        await RisingEdge(dut.clk)
        await FallingEdge(dut.clk)
        got_ab = (int(dut.g_in.a_q.value), int(dut.g_in.b_q.value))
        if ce:
            assert got_ab == (a, b), "[IN] is not the raw operands: %s vs %s" % (
                [hex(v) for v in got_ab], [hex(a), hex(b)])
        elif prev is not None:
            assert got_ab == prev, "[IN] changed while ce = 0"
    dut.ce.value = 1


@cocotb.test()
async def fp_mul_vectors(dut):
    lat = int(os.environ["TSR_FP_LAT"])
    r = rng(0x301 + lat)
    rf = ref()
    vecs = [(a, b) for a, b, _ in MUL_DIRECTED] + pair_vectors(r, 1500, div=False)
    dut.ce.value = 1
    dut.in_valid.value = 0
    dut.a.value = 0
    dut.b.value = 0
    await start_clock_and_reset(dut)
    pipe = deque()           # inputs taken at the last `lat` advancing edges: (valid, a, b)
    idx = checked = 0
    held = None
    while idx < len(vecs) or any(v for v, _, _ in pipe):
        ce = r.random() < 0.8
        iv = idx < len(vecs) and r.random() < 0.85
        a, b = vecs[idx] if iv else (r.getrandbits(32), r.getrandbits(32))
        dut.ce.value = int(ce)
        dut.in_valid.value = int(iv)
        dut.a.value = a
        dut.b.value = b
        await RisingEdge(dut.clk)
        if ce:
            pipe.append((iv, a, b))
            if len(pipe) > lat:
                pipe.popleft()
            idx += int(iv)
        await FallingEdge(dut.clk)
        mv = len(pipe) == lat and pipe[0][0]
        assert int(dut.out_valid.value) == int(mv), "out_valid mismatch at vector %d" % idx
        if mv:
            _, ea, eb = pipe[0]
            got, want = int(dut.y.value), rf.fmul(ea, eb)
            assert got == want, "fmul(0x%08x, 0x%08x) = 0x%08x, want 0x%08x (LAT %d)" % (ea, eb, got, want, lat)
            if ce:
                checked += 1
            elif held is not None:
                assert got == held, "y changed while ce = 0"
            held = got
    assert checked == len(vecs), (checked, len(vecs))
    dut._log.info("tsr_fp_mul LAT=%d: %d products checked" % (lat, checked))


# =============================================================================================
# cocotb: tsr_fp_add
# =============================================================================================
ADD_CUTS = ("g_in", "g_c1", "g_c2", "g_c3", "g_xout")


def add_cuts(lat: int) -> dict:
    """Cut registers of tsr_fp_add at LAT (its header, "Pipeline")."""
    return {"g_in": lat >= 2, "g_c1": lat >= 4, "g_c2": lat >= 3, "g_c3": lat >= 5, "g_xout": lat >= 6}


def _add_ref(a: int, b: int, sub: int) -> int:
    return ref().fsub(a, b) if sub else ref().fadd(a, b)


async def _add_idle(dut):
    dut.ce.value = 1
    dut.in_valid.value = 0
    dut.a.value = 0
    dut.b.value = 0
    dut.sub.value = 0
    await start_clock_and_reset(dut)


@cocotb.test()
async def fp_add_stages(dut):
    """Stage allocation of tsr_fp_add (cut registers per LAT; [IN] holds the raw operands, so no
    logic sits in front of the module's input flops; LAT 6..8 add output registers). The logic
    depth itself needs Vivado: test_fp_ooc_timing (opt-in)."""
    lat = int(os.environ["TSR_FP_LAT"])
    r = rng(0x711 + lat)
    want = add_cuts(lat)
    got = {k: _has_scope(dut, k) for k in ADD_CUTS}
    assert got == want, "LAT %d: cut registers %s, expected %s" % (lat, got, want)
    if lat >= 6:
        assert len(dut.g_xout.y_q) == lat - 5, "LAT %d: %d output registers" % (lat, len(dut.g_xout.y_q))
    await _add_idle(dut)
    if lat < 2:
        return
    prev = None
    for i in range(64):
        a, b, sub = (0xFFFFFFFF, 0x7F800000, 1) if i == 0 else (_rnd_any(r), _rnd_any(r), r.getrandbits(1))
        ce = r.random() < 0.7
        dut.ce.value = int(ce)
        dut.a.value = a
        dut.b.value = b
        dut.sub.value = sub
        await RisingEdge(dut.clk)
        await FallingEdge(dut.clk)
        q = int(dut.g_in.d_q.value)                     # packed {a, b, sub}
        got_in = (q >> 33, (q >> 1) & 0xFFFFFFFF, q & 1)
        if ce:
            assert got_in == (a, b, sub), "[IN] is not the raw operands: %s vs %s" % (
                [hex(v) for v in got_in], [hex(a), hex(b), sub])
        elif prev is not None:
            assert got_in == prev, "[IN] changed while ce = 0"
        prev = got_in
    dut.ce.value = 1


@cocotb.test()
async def fp_add_vectors(dut):
    """Directed + generated vectors, bit-exact against libtsr_ref, with random ce (incl. long
    ce = 0 bursts) and in_valid gaps against a LAT-deep cycle model: out_valid in every cycle,
    y at every fresh result, out_valid / y unchanged while ce = 0."""
    lat = int(os.environ["TSR_FP_LAT"])
    r = rng(0x701 + lat)
    vecs = [(a, b, s) for a, b, s, _ in ADD_DIRECTED] + add_vectors(r, 3000)
    await _add_idle(dut)
    pipe = deque()           # inputs taken at the last `lat` advancing edges: (valid, a, b, sub)
    idx = checked = 0
    burst = 0
    prev = None
    while idx < len(vecs) or any(v for v, _, _, _ in pipe):
        if burst == 0 and r.random() < 0.01:
            burst = r.randint(3, 25)
        if burst:
            burst -= 1
            ce = False
        else:
            ce = r.random() < 0.85
        iv = idx < len(vecs) and r.random() < 0.85
        a, b, sub = vecs[idx] if iv else (r.getrandbits(32), r.getrandbits(32), r.getrandbits(1))
        dut.ce.value = int(ce)
        dut.in_valid.value = int(iv)
        dut.a.value = a
        dut.b.value = b
        dut.sub.value = sub
        await RisingEdge(dut.clk)
        if ce:
            pipe.append((iv, a, b, sub))
            if len(pipe) > lat:
                pipe.popleft()
            idx += int(iv)
        await FallingEdge(dut.clk)
        ov, y = int(dut.out_valid.value), int(dut.y.value)
        if not ce and prev is not None:
            assert (ov, y) == prev, "out_valid / y changed while ce = 0"
        prev = (ov, y)
        mv = len(pipe) == lat and pipe[0][0]
        assert ov == int(mv), "out_valid mismatch at vector %d (LAT %d)" % (idx, lat)
        if mv and ce:
            _, ea, eb, es = pipe[0]
            want = _add_ref(ea, eb, es)
            assert y == want, "%s(0x%08x, 0x%08x) = 0x%08x, want 0x%08x (LAT %d)" % (
                "fsub" if es else "fadd", ea, eb, y, want, lat)
            checked += 1
    assert checked == len(vecs), (checked, len(vecs))
    dut._log.info("tsr_fp_add LAT=%d: %d results checked" % (lat, checked))


@cocotb.test()
async def fp_add_throughput(dut):
    """Measured latency and throughput with ce = 1: 512 operations on consecutive edges give
    512 results on consecutive cycles, the first one LAT edges after the first operation was
    taken (out_valid at the falling edge after edge n + LAT - 1)."""
    lat = int(os.environ["TSR_FP_LAT"])
    r = rng(0x731 + lat)
    vecs = add_vectors(r, 100)[:512]
    assert len(vecs) == 512
    await _add_idle(dut)
    await FallingEdge(dut.clk)
    edge = 0                 # rising edges since the first operation was presented
    got = []                 # (edge index of the result, y)
    for i in range(512 + lat + 4):
        if i < len(vecs):
            dut.in_valid.value = 1
            dut.a.value, dut.b.value, dut.sub.value = vecs[i]
        else:
            dut.in_valid.value = 0
        await RisingEdge(dut.clk)
        edge += 1
        await FallingEdge(dut.clk)
        if int(dut.out_valid.value):
            got.append((edge, int(dut.y.value)))
    assert len(got) == 512, "%d results for 512 operations" % len(got)
    first = got[0][0]
    assert first == lat, "first result after %d edges, LAT %d" % (first, lat)
    assert [e for e, _ in got] == list(range(first, first + 512)), "results not on consecutive cycles"
    for (a, b, sub), (_, y) in zip(vecs, got):
        assert y == _add_ref(a, b, sub), (hex(a), hex(b), sub, hex(y))
    dut._log.info("tsr_fp_add LAT=%d: latency %d cycles, 512 results in 512 consecutive cycles" % (lat, first))


@cocotb.test()
async def fp_add_reset(dut):
    """A synchronous reset with a full pipeline (and ce = 0 during the reset) clears every valid
    stage: no result of an operation taken before the reset ever appears; operations after the
    reset run normally."""
    lat = int(os.environ["TSR_FP_LAT"])
    r = rng(0x721 + lat)
    await _add_idle(dut)
    for rep in range(6):
        for _ in range(lat + r.randrange(3)):          # fill the pipeline
            dut.ce.value = 1
            dut.in_valid.value = 1
            dut.a.value, dut.b.value, dut.sub.value = _rnd_any(r), _rnd_any(r), r.getrandbits(1)
            await RisingEdge(dut.clk)
        dut.in_valid.value = 0
        dut.ce.value = rep % 2                          # reset must not depend on ce
        dut.rst_n.value = 0
        await RisingEdge(dut.clk)
        await FallingEdge(dut.clk)
        assert int(dut.out_valid.value) == 0, "out_valid after reset"
        dut.rst_n.value = 1
        dut.ce.value = 1
        for _ in range(lat + 2):
            await RisingEdge(dut.clk)
            await FallingEdge(dut.clk)
            assert int(dut.out_valid.value) == 0, "a pre-reset operation survived the reset"
        a, b, sub = _rnd_any(r), _rnd_any(r), r.getrandbits(1)
        dut.in_valid.value = 1
        dut.a.value, dut.b.value, dut.sub.value = a, b, sub
        await RisingEdge(dut.clk)
        dut.in_valid.value = 0
        for i in range(lat - 1):
            await FallingEdge(dut.clk)
            assert int(dut.out_valid.value) == 0, "early out_valid"
            await RisingEdge(dut.clk)
        await FallingEdge(dut.clk)
        assert int(dut.out_valid.value) == 1 and int(dut.y.value) == _add_ref(a, b, sub)


# =============================================================================================
# cocotb: default latencies vs tsr_pkg (tsr_fp_defaults_tb)
# =============================================================================================
@cocotb.test()
async def fp_defaults(dut):
    """Regression (review finding: tsr_pkg::TSR_DEF_FADD_LAT was 6 while tsr_fp_add's default,
    the LAT swept and timing-closed here, is 5). Measures the latency of tsr_fp_mul / tsr_fp_add
    instantiated with their default LAT (one operation, counted like fp_add_throughput) and checks
    default = the U200 LAT this file sweeps and runs out of context (MUL_LAT_U200 / ADD_LAT_U200)
    = tsr_pkg::TSR_DEF_FMUL_LAT / TSR_DEF_FADD_LAT, the value every MVM block passes."""
    dut.in_valid.value = 0
    dut.a.value = 0
    dut.b.value = 0
    await start_clock_and_reset(dut)
    a, b = 0x3FC00000, 0x40100000          # 1.5, 2.25: product 3.375, sum 3.75
    await FallingEdge(dut.clk)
    seen = {}                              # unit -> (edge of the first out_valid, y)
    for edge in range(1, 16):              # the operation is taken at edge 1
        dut.in_valid.value = 1 if edge == 1 else 0
        dut.a.value, dut.b.value = a, b
        await RisingEdge(dut.clk)
        await FallingEdge(dut.clk)
        for unit in ("mul", "add"):
            if int(getattr(dut, unit + "_out_valid").value) and unit not in seen:
                seen[unit] = (edge, int(getattr(dut, unit + "_y").value))
    assert set(seen) == {"mul", "add"}, "no result within 15 cycles: %s" % seen
    assert seen["mul"][1] == ref().fmul(a, b) == 0x40580000, hex(seen["mul"][1])
    assert seen["add"][1] == ref().fadd(a, b) == 0x40700000, hex(seen["add"][1])
    lat_mul, lat_add = seen["mul"][0], seen["add"][0]
    pkg_mul, pkg_add = int(dut.pkg_fmul_lat.value), int(dut.pkg_fadd_lat.value)
    dut._log.info("default LAT: tsr_fp_mul %d, tsr_fp_add %d; tsr_pkg TSR_DEF_FMUL_LAT %d, "
                  "TSR_DEF_FADD_LAT %d" % (lat_mul, lat_add, pkg_mul, pkg_add))
    errs = []
    if lat_mul != MUL_LAT_U200:
        errs.append("tsr_fp_mul default LAT %d != MUL_LAT_U200 %d" % (lat_mul, MUL_LAT_U200))
    if lat_add != ADD_LAT_U200:
        errs.append("tsr_fp_add default LAT %d != ADD_LAT_U200 %d" % (lat_add, ADD_LAT_U200))
    if pkg_mul != lat_mul:
        errs.append("rtl/core/tsr_pkg.sv TSR_DEF_FMUL_LAT = %d, but tsr_fp_mul's default LAT is %d"
                    % (pkg_mul, lat_mul))
    if pkg_add != lat_add:
        errs.append("rtl/core/tsr_pkg.sv TSR_DEF_FADD_LAT = %d, but tsr_fp_add's default LAT is %d"
                    % (pkg_add, lat_add))
    assert not errs, "; ".join(errs) + " (the MVM blocks default MUL_LAT / ADD_LAT to tsr_pkg)"


# =============================================================================================
# cocotb: tsr_fp_div
# =============================================================================================
@cocotb.test()
async def fp_div_vectors(dut):
    r = rng(0x401)
    rf = ref()
    vecs = [(a, b) for a, b, _ in DIV_DIRECTED] + pair_vectors(r, 400, div=True)
    dut.in_valid.value = 0
    dut.out_ready.value = 0
    dut.a.value = 0
    dut.b.value = 0
    await start_clock_and_reset(dut)
    # Cycle numbering: cyc = number of rising edges so far; at a falling edge cyc is the index of
    # the edge just passed, so a transfer seen there happens at edge cyc + 1.
    cyc = [0]
    expect = deque()         # (a, b, accepting edge)
    done = [0]
    cover = {"idle_accept": 0, "blocked": 0}

    async def counter():
        while True:
            await RisingEdge(dut.clk)
            cyc[0] += 1

    async def driver():
        await FallingEdge(dut.clk)
        for a, b in vecs:
            # gaps: back-to-back, short, and long enough for the divider to sit idle
            for _ in range(r.choice([0, 0, 0, 1, 3, 20])):
                await FallingEdge(dut.clk)
            dut.in_valid.value = 1
            dut.a.value = a
            dut.b.value = b
            # at a falling edge: in_ready (a register) is stable until the next rising edge
            if int(dut.in_ready.value):
                cover["idle_accept"] += 1       # offered to an already idle divider
            while not int(dut.in_ready.value):
                await FallingEdge(dut.clk)
            expect.append((a, b, cyc[0] + 1))   # transfer at the next rising edge
            await FallingEdge(dut.clk)
            assert int(dut.in_ready.value) == 0, "in_ready right after an accept"
            dut.in_valid.value = 0
            dut.a.value = r.getrandbits(32)     # operands must have been latched
            dut.b.value = r.getrandbits(32)

    async def monitor():
        held = None                             # y presented and not taken in the last cycle
        freed = 0                               # edge at which the output register was last freed
        stall = 0                               # long out_ready-low bursts block the engine
        while done[0] < len(vecs):
            await FallingEdge(dut.clk)
            ov = int(dut.out_valid.value)
            if held is not None:
                assert ov == 1 and int(dut.y.value) == held, "R2: out_valid / y not held"
            if stall == 0 and r.random() < 0.02:
                stall = r.randint(10, 40)
            if stall:
                stall -= 1
                rdy = False
            else:
                rdy = r.random() < 0.7
            dut.out_ready.value = int(rdy)
            if not ov:
                held = None
                continue
            assert expect, "out_valid without an accepted operation"
            a, b, acc = expect[0]
            got, want = int(dut.y.value), rf.fdiv(a, b)
            assert got == want, "fdiv(0x%08x, 0x%08x) = 0x%08x, want 0x%08x" % (a, b, got, want)
            if held is None:                    # first cycle of this result
                exp_edge = max(acc + DIV_LATENCY, freed)
                cover["blocked"] += int(freed > acc + DIV_LATENCY)
                assert cyc[0] == exp_edge, "result at edge %d, expected %d (accepted %d)" % (
                    cyc[0], exp_edge, acc)
            if rdy:
                expect.popleft()
                done[0] += 1
                held = None
                freed = cyc[0] + 1
            else:
                held = got
        await RisingEdge(dut.clk)               # the last transfer
        dut.out_ready.value = 0

    cocotb.start_soon(counter())
    cocotb.start_soon(driver())
    await monitor()
    for _ in range(40):
        await FallingEdge(dut.clk)
        assert int(dut.out_valid.value) == 0
    dut._log.info("tsr_fp_div: %d quotients checked, coverage %s" % (done[0], cover))
    assert cover["idle_accept"] > 20 and cover["blocked"] > 20, cover


# =============================================================================================
# cocotb: tsr_fp_pkg functions (tsr_fp_funcs_tb)
# =============================================================================================
def _pack(vals, w):
    out = 0
    for i, v in enumerate(vals):
        out |= (v & ((1 << w) - 1)) << (w * i)
    return out


def _unpack(v, w, n):
    return [(v >> (w * i)) & ((1 << w) - 1) for i in range(n)]


def _s8(v):
    return v - 256 if v >= 128 else v


async def _funcs_batch(dut, xs, hs=None):
    xs = list(xs) + [0] * (LANES - len(xs))
    hs = list(hs if hs is not None else [x & 0xFFFF for x in xs]) + [0] * (LANES - len(hs or xs))
    dut.x.value = _pack(xs, 32)
    dut.h.value = _pack(hs, 16)
    await Timer(1, "ns")
    return xs, hs


def _amax_scan(xs):
    """tsr_ref.c tsr_qact amax_candidate() scan (transcribed; also checked through tsr_qact)."""
    r = ref()
    amax, mx = 0, 0
    for x in xs:
        d = r.daz(x)
        a = d & 0x7FFFFFFF
        if a <= 0x7F800000 and a > amax:
            amax, mx = a, d
    return amax, mx


async def _check_conversions(dut, values):
    r = ref()
    n = 0
    for i in range(0, len(values), LANES):
        xs, _ = await _funcs_batch(dut, values[i:i + LANES])
        f2h = _unpack(int(dut.f2h.value), 16, LANES)
        q8a = _unpack(int(dut.q8a.value), 8, LANES)
        q8e = _unpack(int(dut.q8e.value), 8, LANES)
        i2f = _unpack(int(dut.i2f.value), 32, LANES)
        daz = _unpack(int(dut.daz.value), 32, LANES)
        nan = _unpack(int(dut.is_nan.value), 1, LANES)
        amag = _unpack(int(dut.amag.value), 31, LANES)
        cand = _unpack(int(dut.cand.value), 32, LANES)
        for j, x in enumerate(xs):
            assert f2h[j] == r.f32_to_fp16(x), "f32_to_fp16(0x%08x) = 0x%04x" % (x, f2h[j])
            assert _s8(q8a[j]) == r.q8_round_away(x), "q8_round_away(0x%08x) = %d" % (x, _s8(q8a[j]))
            assert _s8(q8e[j]) == r.q8_round_even(x), "q8_round_even(0x%08x) = %d" % (x, _s8(q8e[j]))
            assert i2f[j] == r.i32_to_f32(x), "i32_to_f32(0x%08x) = 0x%08x" % (x, i2f[j])
            assert daz[j] == r.daz(x), "daz(0x%08x)" % x
            assert nan[j] == int((x & 0x7FFFFFFF) > 0x7F800000), "is_nan(0x%08x)" % x
            assert amag[j] == r.daz(x) & 0x7FFFFFFF, "amag(0x%08x)" % x
            am, mx = _amax_scan([x])
            assert cand[j] == (am << 1) | (mx >> 31), "amax_cand(0x%08x)" % x
            n += 1
    return n


@cocotb.test()
async def fp_funcs_fp16_exhaustive(dut):
    r = ref()
    for base in range(0, 65536, LANES):
        hs = list(range(base, base + LANES))
        await _funcs_batch(dut, [0] * LANES, hs)
        got = _unpack(int(dut.h2f.value), 32, LANES)
        for j, h in enumerate(hs):
            assert got[j] == r.fp16_to_f32(h), "fp16_to_f32(0x%04x) = 0x%08x" % (h, got[j])
    dut._log.info("fp16_to_f32: all 65536 halves checked")


@cocotb.test()
async def fp_funcs_conversions(dut):
    r = ref()
    rr = rng(0x501)
    vals = [0x00000000, 0x80000000, 0x00000001, 0x807FFFFF, 0x00800000, 0x7F800000, 0xFF800000,
            0x7FC00000, 0x7F800001, 0xFFC12345, 0xFFFFFFFF, 0x7F7FFFFF, 0xFF7FFFFF,
            # f32 -> fp16 (test_ref_fp.c)
            0x477FE000, 0x477FEFFF, 0x477FF000, 0x47800000, 0x33800000, 0x33000000, 0x33000001,
            0xB3000001, 0x33C00000, 0x34200000, 0x387FE000, 0x80400000, 0x3F801000, 0x3F803000,
            0x3F801001,
            # int32 (test_ref_fp.c)
            16777216, 16777217, 16777218, 16777219, 33554435, 33554434, 33554438, 30965760,
            0x7FFFFFFF, 0x80000000, 2147483583, 2147483584, (-16777217) & 0xFFFFFFFF, 1, 0xFFFFFFFF]
    # q8 (test_ref_fp.c) and every half-integer of [-256, 256] +- 1 ulp, both rounding modes
    for x in (0.49999997, 0.5, -0.5, 1.5, 2.5, -2.5, -1.5, 3.5, 126.5, -126.5, 127.0, 127.5,
              127.49998, -127.5, -128.0, -128.5, 128.0, 1e10, -1e10, 0.50000006, 2.4999998, 255.5, 300.0):
        vals.append(f32(x))
    for k in range(-512, 513):
        u = f32(k / 2.0)
        vals += [u, (u + 1) & 0xFFFFFFFF, (u - 1) & 0xFFFFFFFF]
    # every exponent, both signs, special + random mantissas
    for e in range(256):
        for s in (0, 1):
            vals += [mk(s, e, _rnd_mant(rr)) for _ in range(12)]
    # every fp16 rounding midpoint (between consecutive finite halves) and +-1 f32 ulp
    for h in range(0, 0x7C00):
        lo, hi = r.fp16_to_f32(h), r.fp16_to_f32(h + 1)
        mid = f32((f32val(lo) + f32val(hi)) / 2)          # exact: midpoints fit in binary32
        s = rr.getrandbits(1) << 31
        vals += [mid | s, (mid + 1) | s, (mid - 1) | s]
    # int32 boundaries
    for k in range(32):
        for d in (-2, -1, 0, 1, 2):
            vals += [((1 << k) + d) & 0xFFFFFFFF, (-(1 << k) + d) & 0xFFFFFFFF]
    vals += [rr.getrandbits(32) >> rr.randrange(32) for _ in range(2000)]
    n = await _check_conversions(dut, vals)
    dut._log.info("conversions: %d values checked" % n)


@cocotb.test()
async def fp_funcs_amax(dut):
    r = ref()
    rr = rng(0x601)
    sp = [0x00000000, 0x80000000, 0x00000001, 0x807FFFFF, 0x7F800000, 0xFF800000, 0x7FC00000,
          0xFFFFFFFF, 0x7F800001, 0x00800000, 0x80800000, 0x7F7FFFFF]
    blocks = [[0] * LANES, [0x80000000] * LANES, [0x7FC00000] * LANES, [0x807FFFFF] * LANES,
              [f32(-4.0), f32(4.0)] + [f32(1.0)] * (LANES - 2),        # spec example: first of the tie
              [f32(4.0), f32(-4.0)] + [f32(1.0)] * (LANES - 2),
              [0x7FC00000, 0xFF800000, 0x7F800000] + [0] * (LANES - 3)]
    for _ in range(600):
        mode = rr.randrange(4)
        pool = [_rnd_normal(rr, 100, 140) & 0x7FFFFFFF for _ in range(rr.randint(1, 4))]
        blk = []
        for _ in range(LANES):
            if mode == 0:
                v = _rnd_any(rr)
            elif mode == 1:
                v = rr.choice(sp)
            elif mode == 2:
                v = rr.choice(pool) | (rr.getrandbits(1) << 31)
            else:
                v = rr.choice(sp) if rr.randrange(3) else rr.choice(pool) | (rr.getrandbits(1) << 31)
            blk.append(v)
        blocks.append(blk)
    for blk in blocks:
        await _funcs_batch(dut, blk)
        amax, mx = _amax_scan(blk)
        for pre in ("tree", "chain"):
            ga, gm = int(getattr(dut, pre + "_amax").value), int(getattr(dut, pre + "_max").value)
            assert (ga, gm) == (amax, mx), "%s: got (0x%08x, 0x%08x) want (0x%08x, 0x%08x) for %s" % (
                pre, ga, gm, amax, mx, [hex(v) for v in blk])
        # Through the reference QACT itself (zero padding never changes amax / max):
        ga, gm = int(dut.tree_amax.value), int(dut.tree_max.value)
        v = r.qa_split(QA_A32, 32, r.qact(QA_A32, blk + [0] * (32 - LANES), 32))
        assert v["d"][0] == r.fp16_to_f32(r.f32_to_fp16(r.fdiv(ga, 0x42FE0000))), "A32 D"
        v = r.qa_split(QA_A256, 256, r.qact(QA_A256, blk + [0] * (256 - LANES), 256))
        want_d = 0 if ga == 0 else r.frecip(r.fdiv(0xC2FE0000, gm))
        assert v["d"][0] == want_d, "A256 d: block %s" % [hex(x) for x in blk]
    dut._log.info("amax: %d blocks checked" % len(blocks))


# =============================================================================================
# pytest: simulations
# =============================================================================================
def test_fp_lint():
    vl = os.environ.get("VERILATOR", "verilator")
    runs = [("tsr_fp_mul", ["-GLAT=%d" % lat]) for lat in range(1, 9)]
    runs += [("tsr_fp_add", ["-GLAT=%d" % lat]) for lat in ADD_LATS]
    runs += [("tsr_fp_add_dsp", ["-GLAT=%d" % lat, "-GNORM_DSP=%d" % norm]) for lat, norm in ADD_DSP_CFGS]
    runs += [("tsr_fp_div", []), ("tsr_fp_funcs_tb", ["-GLANES=4"]), ("tsr_fp_funcs_tb", []),
             ("tsr_fp_sweep_tb", ["-GMUL_LAT=1", "-GADD_LAT=1"]),
             ("tsr_fp_sweep_tb", ["-GMUL_LAT=5", "-GADD_LAT=8", "-GALANES=3"])]
    srcs = [str(s) for s in FP_DSP_SOURCES + [FUNCS_SV, SWEEP_SV]]
    for top, args in runs:
        p = subprocess.run([vl, "--lint-only", "-Wall", "--top-module", top] + args + srcs,
                           capture_output=True, text=True)
        assert p.returncode == 0, "lint %s %s:\n%s" % (top, args, p.stderr)
    p = subprocess.run([vl, "--lint-only", "-Wall", "--top-module", "tsr_fp_defaults_tb",
                        str(TABLES_SV), str(PKG_SV)] + [str(f) for f in FP_SOURCES] + [str(DEFAULTS_SV)],
                       capture_output=True, text=True)
    assert p.returncode == 0, "lint tsr_fp_defaults_tb:\n%s" % p.stderr


def test_fp_funcs():
    run("tsr_fp_funcs_tb", __name__, sources=FP_SOURCES + [FUNCS_SV], parameters={"LANES": LANES},
        testcase=["fp_funcs_fp16_exhaustive", "fp_funcs_conversions", "fp_funcs_amax"])


@pytest.mark.parametrize("lat", [1, 2, 3, 4, 5, 6])
def test_fp_mul(lat):
    run("tsr_fp_mul", __name__, sources=FP_SOURCES, parameters={"LAT": lat},
        testcase=["fp_mul_stages", "fp_mul_vectors"], extra_env={"TSR_FP_LAT": str(lat)})


def test_fp_div():
    run("tsr_fp_div", __name__, sources=FP_SOURCES, testcase="fp_div_vectors")


@pytest.mark.parametrize("lat", ADD_LATS)
def test_fp_add(lat):
    run("tsr_fp_add", __name__, sources=FP_SOURCES, parameters={"LAT": lat},
        testcase=["fp_add_stages", "fp_add_vectors", "fp_add_throughput", "fp_add_reset"],
        extra_env={"TSR_FP_LAT": str(lat)})


@pytest.mark.parametrize("lat,norm", ADD_DSP_CFGS)
def test_fp_add_dsp(lat, norm):
    """tsr_fp_add_dsp (the DSP alignment prototype, area run E1c of docs/design/02-vector-unit.md §7.10;
    norm: NORM_DSP, the near-path shift in a DSP as well): the same cocotb tests as tsr_fp_add, bit-exact
    against libtsr_ref."""
    run("tsr_fp_add_dsp", __name__, sources=FP_DSP_SOURCES, parameters={"LAT": lat, "NORM_DSP": norm},
        testcase=["fp_add_stages", "fp_add_vectors", "fp_add_throughput", "fp_add_reset"],
        extra_env={"TSR_FP_LAT": str(lat)})


def test_fp_default_lats():
    """tsr_pkg::TSR_DEF_FMUL_LAT / TSR_DEF_FADD_LAT = the measured default LAT of tsr_fp_mul /
    tsr_fp_add = MUL_LAT_U200 / ADD_LAT_U200 (cocotb fp_defaults)."""
    run("tsr_fp_defaults_tb", __name__, sources=[TABLES_SV, PKG_SV] + FP_SOURCES + [DEFAULTS_SV],
        testcase="fp_defaults")


# =============================================================================================
# Opt-in rule of the slow tests (tb/pytest.ini does not deselect the `slow` marker, so a plain
# `pytest tb` / tb/run_all.sh would otherwise run them)
# =============================================================================================
SLOW_ENV = "TSR_FP_SLOW"


def slow_requested(config, name: str, env=None) -> bool:
    """True if the slow test function `name` (test_fp_<short>) was asked for: TSR_FP_SLOW=1, an
    -m expression naming the `slow` marker (pytest has already deselected the test if the
    expression rejects it), a -k expression containing <short> (e.g. -k sweep_full), or a node id
    argument naming the test."""
    env = os.environ if env is None else env
    if env.get(SLOW_ENV, "0") not in ("", "0"):
        return True
    if re.search(r"\bslow\b", config.getoption("markexpr", "") or ""):
        return True
    short = name[len("test_fp_"):] if name.startswith("test_fp_") else name
    if short in (config.getoption("keyword", "") or ""):
        return True
    return any(("::" + name) in str(a) for a in config.args)


def require_slow(request):
    if not slow_requested(request.config, request.node.originalname):
        pytest.skip("slow test, opt-in: -m slow, %s=1, -k <name> or its node id" % SLOW_ENV)


SLOW_ONLY_PLUGIN = """
def pytest_collection_modifyitems(config, items):
    drop = [i for i in items if i.get_closest_marker("slow") is None]
    if drop:
        config.hook.pytest_deselected(items=drop)
        items[:] = [i for i in items if i.get_closest_marker("slow") is not None]
"""


def test_fp_slow_gate():
    """Regression (review finding): `pytest tb` / tb/run_all.sh must not start the slow tests
    (the full sweep is 4 processes for ~5 min; the Vivado runs need ~3 GB each)."""
    class Cfg:
        def __init__(self, markexpr="", keyword="", args=("tb",)):
            self.opts = {"markexpr": markexpr, "keyword": keyword}
            self.args = list(args)

        def getoption(self, key, default=None):
            return self.opts.get(key, default)

    for n in ("test_fp_sweep_full", "test_fp_ooc_timing"):
        short = n[len("test_fp_"):]
        assert not slow_requested(Cfg(), n, env={})
        assert not slow_requested(Cfg(markexpr="not traces"), n, env={})
        assert not slow_requested(Cfg(keyword="fp"), n, env={})
        assert not slow_requested(Cfg(keyword="sweep"), n, env={})
        assert not slow_requested(Cfg(args=["fp/test_fp.py", "fp"]), n, env={})
        assert not slow_requested(Cfg(), n, env={SLOW_ENV: "0"})
        assert slow_requested(Cfg(markexpr="slow"), n, env={})
        assert slow_requested(Cfg(markexpr="slow and not traces"), n, env={})
        assert slow_requested(Cfg(keyword=short), n, env={})
        assert slow_requested(Cfg(keyword=n), n, env={})
        assert slow_requested(Cfg(args=["fp/test_fp.py::%s" % n]), n, env={})
        assert slow_requested(Cfg(args=["fp/test_fp.py::%s[mul]" % n]), n, env={})
        assert slow_requested(Cfg(), n, env={SLOW_ENV: "1"})
    # End to end: run this file the way `pytest tb` does (no -m / -k / node id), keeping only
    # the slow tests (plugin below): every one of them must be reported as skipped, quickly.
    pdir = build_root() / "fp_slow_gate"
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "fp_slow_only.py").write_text(SLOW_ONLY_PLUGIN)
    env = {k: v for k, v in os.environ.items() if k not in (SLOW_ENV, "PYTEST_ADDOPTS")}
    env["PYTHONPATH"] = os.pathsep.join([str(pdir)] + [p for p in [env.get("PYTHONPATH")] if p])
    p = subprocess.run([sys.executable, "-m", "pytest", "-p", "fp_slow_only", "-p", "no:cacheprovider",
                        "-q", "-rs", "fp/test_fp.py"], cwd=str(REPO / "tb"), env=env,
                       capture_output=True, text=True, timeout=300)
    out = p.stdout + p.stderr
    assert p.returncode == 0, out[-3000:]
    m = re.search(r"(\d+) skipped", out)
    n_slow = 1 + len(OOC_RUNS)                   # test_fp_sweep_full + one OOC run per module
    assert m and int(m.group(1)) == n_slow and " passed" not in out and " failed" not in out, out[-3000:]


# =============================================================================================
# pytest: Verilator C++ sweep harness
# =============================================================================================
def build_sweep(mul_lat: int = MUL_LAT_U200, add_lat: int = ADD_LAT_U200, add_dsp: int = 0) -> Path:
    """Build tb/fp/cpp/fp_sweep.cpp against libtsr_ref.so; returns the executable. add_dsp: the adders are
    tsr_fp_add_dsp (+define+TSR_FP_ADD_DSP=<add_dsp> makes tsr_fp_add a wrapper of it; 2: NORM_DSP)."""
    lib = Path(ref().path).resolve()
    vl = os.environ.get("VERILATOR", "verilator")
    bdir = build_root() / ("fp_sweep_mul%d_add%d%s" % (mul_lat, add_lat, "_dsp%d" % add_dsp if add_dsp else ""))
    bdir.mkdir(parents=True, exist_ok=True)
    jobs = os.environ.get("TSR_BUILD_JOBS", "4")
    cmd = [vl, "--cc", "--exe", "--build", "-O3", "-j", jobs, "--top-module", "tsr_fp_sweep_tb",
           "-GLANES=%d" % LANES, "-GMLANES=16", "-GDLANES=16", "-GMUL_LAT=%d" % mul_lat,
           "-GALANES=32", "-GADD_LAT=%d" % add_lat,
           "-CFLAGS", "-O2 -I%s -DMUL_LAT_DEF=%d -DADD_LAT_DEF=%d" % (
               REPO / "sw" / "sim" / "include", mul_lat, add_lat),
           "-LDFLAGS", "-L%s -ltsr_ref -Wl,-rpath,%s" % (lib.parent, lib.parent),
           "--Mdir", str(bdir), "-o", "fp_sweep"]
    if add_dsp:
        cmd += ["+define+TSR_FP_ADD_DSP=%d" % add_dsp]
    cmd += [str(s) for s in (FP_DSP_SOURCES if add_dsp else FP_SOURCES) + [FUNCS_SV, SWEEP_SV, SWEEP_CPP]]
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode == 0, "fp_sweep build failed:\n%s\n%s" % (p.stdout[-4000:], p.stderr[-4000:])
    return bdir / "fp_sweep"


def run_sweeps(exe: Path, jobs):
    """Run [(mode, a, b), ...] concurrently (TSR_FP_SWEEP_PROCS at a time); all must pass."""
    procs = int(os.environ.get("TSR_FP_SWEEP_PROCS", "4"))
    pending, running, failed = list(jobs), [], []
    while pending or running:
        while pending and len(running) < procs:
            job = pending.pop(0)
            running.append((job, subprocess.Popen([str(exe)] + [str(j) for j in job],
                                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)))
        job, p = running.pop(0)
        out = p.communicate()[0]
        print("fp_sweep %s:\n%s" % (" ".join(str(j) for j in job), out.strip()))
        if p.returncode != 0:
            failed.append((job, out[-3000:]))
    assert not failed, "fp_sweep failures:\n%s" % "\n".join("%s:\n%s" % f for f in failed)


# int32 -> f32 exhaustively on [-2^27, 2^27] (funcs mode: every pattern u of the range, as int32),
# in four chunks; the rest of the int32 range is covered by funcs_rand and the full sweep.
I2F_EXHAUSTIVE = [("funcs", 0, 1 << 26), ("funcs", 1 << 26, (1 << 27) + 1),
                  ("funcs", (1 << 32) - (1 << 27), (1 << 32) - (1 << 26)), ("funcs", (1 << 32) - (1 << 26), 1 << 32)]


def test_fp_sweep_quick():
    seed = int(os.environ.get("TSR_SEED", "1"), 0)
    exe = build_sweep(MUL_LAT_U200, ADD_LAT_U200)
    run_sweeps(exe, [("funcs", 0, 1 << 21), ("funcs", 0x3F000000, 0x3F000000 + (1 << 20)),
                     ("funcs", 0xB3000000, 0xB3000000 + (1 << 20)),
                     ("funcs_rand", 1 << 21, seed), ("amax", 200000, seed),
                     ("mul", 1 << 17, seed), ("div", 1 << 16, seed), ("add", 1 << 18, seed)]
               + I2F_EXHAUSTIVE)
    exe5 = build_sweep(5, 1)   # tsr_fp_mul with the [NORM] register; tsr_fp_add fully combinational
    run_sweeps(exe5, [("mul", 1 << 17, seed + 1), ("add", 1 << 16, seed + 1)])


def test_fp_sweep_add_dsp():
    """tsr_fp_add_dsp (through the TSR_FP_ADD_DSP define of tsr_fp_add) against libtsr_ref: 2.6 M sums at LAT 5
    for two seeds and 0.65 M at LAT 8, then 3.9 M with the near-path shift in a DSP too (NORM_DSP); the add
    classes of fp_sweep.cpp."""
    seed = int(os.environ.get("TSR_SEED", "1"), 0)
    run_sweeps(build_sweep(MUL_LAT_U200, 5, add_dsp=1), [("add", 1 << 18, seed), ("add", 1 << 18, seed + 31)])
    run_sweeps(build_sweep(MUL_LAT_U200, 8, add_dsp=1), [("add", 1 << 16, seed + 7)])
    run_sweeps(build_sweep(MUL_LAT_U200, 5, add_dsp=2), [("add", 1 << 18, seed + 63), ("add", 1 << 17, seed + 95)])


@pytest.mark.slow
def test_fp_sweep_full(request):
    require_slow(request)
    seed = int(os.environ.get("TSR_SEED", "1"), 0)
    exe = build_sweep(MUL_LAT_U200)
    chunk = 1 << 28
    jobs = [("funcs", k * chunk, (k + 1) * chunk) for k in range((1 << 32) // chunk)]
    jobs += [("mul", 1 << 23, seed), ("div", 1 << 23, seed), ("amax", 1 << 22, seed),
             ("funcs_rand", 1 << 26, seed)]
    # tsr_fp_add: 10^8 sums / differences at the default LAT (4 x 10 classes x 2.5 M; consecutive
    # seeds, so the walk class covers > 2^23 consecutive indices, see fp_sweep.cpp), and
    # 2 x 10^7 at LAT 1 (no cut register) and LAT 3 (S1 + S2 and S3 + S4 merged)
    jobs += [("add", 2500000, seed + 100 + k) for k in range(4)]
    run_sweeps(exe, jobs)
    for lat in (1, 3):
        run_sweeps(build_sweep(MUL_LAT_U200, lat), [("add", 1000000, seed + 200 + lat + k) for k in range(2)])


# =============================================================================================
# pytest: Vivado out-of-context timing / logic depth (opt-in)
# =============================================================================================
OOC_RUNS = {"mul": ("tsr_fp_mul", "LAT=%d" % MUL_LAT_U200), "div": ("tsr_fp_div", "-"),
            "add": ("tsr_fp_add", "LAT=%d" % ADD_LAT_U200)}
OOC_MAX_LUT = 6           # docs/rtl-core.md §1: keep logic depth at ~6 LUT levels (300 MHz)


def parse_ooc(log: str) -> dict:
    """The FPOOC summary lines of tb/fp/viv/fp_ooc.tcl."""
    res = {"dsp": [], "stages": {}}
    for line in log.splitlines():
        f = line.split()
        if not f or f[0] != "FPOOC":
            continue
        if f[1] == "wns":
            res["wns"], res["whs"] = float(f[2]), float(f[4])
        elif f[1] == "depth":
            res["max_lut"], res["max_fabric"], res["paths"] = int(f[3]), int(f[5]), int(f[7])
        elif f[1] == "fabric_wns":
            res["fabric_wns"] = float(f[2])
        elif f[1] in ("lut_path", "fabric_path"):
            res[f[1]] = " ".join(f[2:])
        elif f[1] == "dsp":
            res["dsp"].append(" ".join(f[2:]))
        elif f[1] == "stage":
            res["stages"][f[2]] = {"slack": float(f[4]), "max_lut": int(f[6]), "max_fabric": int(f[8])}
    return res


@pytest.mark.slow
@pytest.mark.parametrize("which", sorted(OOC_RUNS))
def test_fp_ooc_timing(request, which):
    """Regression for the review finding on tsr_fp_mul's round stage (P register -> y: 7 LUTs +
    3 CARRY8 at LAT 3 and 4 before the restructure; this test reports max_lut 7 on that design):
    out-of-context place and route at 3.333 ns on xcu200-fsgd2104-2-e must meet timing with at
    most OOC_MAX_LUT LUTs on any path."""
    require_slow(request)
    vivado = os.environ.get("TSR_VIVADO")
    if not vivado:
        pytest.skip("needs TSR_VIVADO=<path to the vivado executable>")
    top, gen = OOC_RUNS[which]
    out = build_root() / ("fp_ooc_%s" % which)
    out.mkdir(parents=True, exist_ok=True)
    p = subprocess.run([vivado, "-mode", "batch", "-nojournal", "-nolog", "-source", str(OOC_TCL),
                        "-tclargs", str(REPO), top, gen, str(out / "reports")],
                       cwd=str(out), capture_output=True, text=True)
    log = p.stdout + p.stderr
    (out / "vivado.log").write_text(log)
    assert p.returncode == 0, "vivado failed (%s):\n%s" % (out / "vivado.log", log[-3000:])
    res = parse_ooc(log)
    print("fp_ooc %s %s: %s" % (top, gen, res))
    assert "wns" in res and "max_lut" in res, "no FPOOC summary in %s" % (out / "vivado.log")
    assert res["wns"] >= 0 and res["whs"] >= 0, "timing not met: %s" % res
    assert res["max_lut"] <= OOC_MAX_LUT, "LUT levels %d > %d on %s" % (
        res["max_lut"], OOC_MAX_LUT, res.get("lut_path"))
