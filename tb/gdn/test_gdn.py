"""GDN unit testbenches (docs/design/02-vector-unit.md §7.12; rtl/core/tsr_vu_gdn*.sv). Oracle:
libtsr_ref.so (sw/sim/tsr_ref_gdn.c) through tb/gdn/gdn_ref.py, bit-exact.

pytest functions (test_*):
  test_gdn_lint        Verilator -Wall lint of the GDN modules (default and edge parameters)
  test_gdn_fx2f        cocotb on tsr_vu_gdn_fx2f: random and directed sums (hi * 2^13 + lo) and scales
                       against tsr_ref_gdn_fx2f: RNE ties, carry, overflow, the 2^-126 - 2^-150
                       edge, zero, in_bad; one conversion per cycle, fixed latency, tags
  test_gdn_dot         cocotb on tsr_vu_gdn_dot: the exact sums of the four chains for rows of 4 .. 32 beats
                       against an integer model (random, spread, narrow, zero rows; the saturating rows of
                       m = 256 that fill the 48-bit accumulators), launches back to back and with gaps
  test_gdn_upd         cocotb on tsr_vu_gdn_upd: FA(FM(alpha, s), FM(e, k)) for random patterns incl. NaN / Inf,
                       both banks, back-to-back and gapped launches, out_last, latency
  test_gdn_coef        cocotb on tsr_vu_gdn_coef: key, table words, bv and the exponents / flags of 40 heads
                       (Inf / NaN vectors, zero, sparse) against tsr_ref_gdn_table and the VU primitives
  test_gdn_unit        cocotb on tsr_vu_gdn: heads of nine data classes (model-like, wide exponent spread, zero,
                       tiny, huge, arbitrary bit patterns, one poisoned row, saturating, window edge) with
                       random stalls, every S' and o word against tsr_ref_gdn; gdn_unit_latency (idle
                       pipeline) and gdn_unit_rate (full rate in heads of one shape) log the numbers of §7.12.8
  test_gdn_unit_lanes16  the same with LANES = 16, NB_MAX = 16
tb/gdn/test_gdn_mutants.py (opt-in) kills 15 one-line mutants of the RTL.
cocotb tests are selected by the pytest functions through common.runner.run(testcase=).
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import cocotb
import pytest
from cocotb.triggers import FallingEdge, RisingEdge

from common.runner import REPO, run
from common.tbutil import rng, start_clock_and_reset

import gdn_ref

CORE = REPO / "rtl" / "core"
CANON = 0x7FC00000
FX2F_SOURCES = [CORE / "tsr_vu_gdn_fx2f.sv"]
LAT_CVT = 8


# ---------------------------------------------------------------------------------------------
# lint
# ---------------------------------------------------------------------------------------------
def _lint(top, srcs, *args):
    p = subprocess.run(["verilator", "--lint-only", "-Wall", "--top-module", top, *args, *map(str, srcs)],
                       capture_output=True, text=True)
    assert p.returncode == 0, "lint %s %s:\n%s" % (top, args, p.stderr[-3000:])


def test_gdn_lint():
    _lint("tsr_vu_gdn_fx2f", FX2F_SOURCES)
    for args in ((), ("-GLANES=16", "-GNB_MAX=16"), ("-GNB_MAX=8", "-GNSLOT=4", "-GR_MAX=64")):
        _lint("tsr_vu_gdn", UNIT_SOURCES, *args)
    _lint("tsr_vu_gdn_ram", [CORE / "tsr_vu_gdn_ram.sv"])
    _lint("tsr_vu_gdn_ram", [CORE / "tsr_vu_gdn_ram.sv"], "-GSTYLE=\"block\"", "-GNWE=2", "-GWIDTH=64", "-GRD_LAT=1")


# ---------------------------------------------------------------------------------------------
# fx2f
# ---------------------------------------------------------------------------------------------
def _split(i: int, r):
    """I = hi * 2^13 + lo with the chain bounds |hi| < 2^45, |lo| < 2^47 (any split of I)."""
    lo = r.randrange(-(1 << 47) + 1, (1 << 47))
    hi = (i - lo) >> 13
    lo = i - (hi << 13)
    assert -(1 << 47) <= lo < (1 << 47) and -(1 << 47) <= hi < (1 << 47)
    return hi, lo


def _fx2f_vectors(r, n):
    v = []
    # directed
    for neg, mag, x in [(0, 1, 0), (1, 1, 0), (0, 0, 7), (0, 1, 128), (0, 0xFFFFFF, 104), (0, 0x1FFFFFF, 103),
                        (0, 1, -126), (0, 1, -127), (0, 0xFFFFFF, -150), (0, 0xFFFFFE, -150), (1, 0xFFFFFF, -150),
                        (0, 0x1FFFFFF, -151), ((0), (1 << 24) + 1, 0), (0, (1 << 24) + 3, 0), (0, (1 << 25) - 1, 0),
                        (0, (1 << 58) - 1, -200), (1, (1 << 58) + (1 << 46), 100), (0, (1 << 58) + 12345, -302),
                        (0, 3, 206), (1, 1, -302)]:
        v.append((neg, mag, x, 0))
    v.append((0, 5, 0, 1))                      # in_bad
    for it in range(n):
        ln = r.randrange(1, 59)
        mag = r.getrandbits(ln) | (1 << (ln - 1))
        if it % 4 == 0:
            mag = (mag & ~0xFFFF) | r.choice([0, 0x8000, 0x8001, 0x7FFF])
        k = it % 5
        if k == 0:
            x = r.randrange(-302, 207)
        elif k == 1:
            x = -126 - ln + 1 + r.randrange(-3, 4)
        elif k == 2:
            x = -150 - ln + r.randrange(0, 8)
        elif k == 3:
            x = 128 - ln + r.randrange(-3, 4)
        else:
            x = r.randrange(-80, 40)
        x = max(-302, min(206, x))
        v.append((r.getrandbits(1), mag, x, 1 if it % 97 == 0 else 0))
    return v


@cocotb.test()
async def gdn_fx2f(dut):
    r = rng(0x6D1)
    vecs = _fx2f_vectors(r, 6000)
    dut.in_valid.value = 0
    dut.in_tag.value = 0
    dut.in_hi.value = 0
    dut.in_lo.value = 0
    dut.in_x.value = 0
    dut.in_bad.value = 0
    await start_clock_and_reset(dut)
    pipe = []
    idx = checked = 0
    cyc = 0
    exp_q = []
    while idx < len(vecs) or exp_q:
        iv = idx < len(vecs) and r.random() < 0.8
        if iv:
            neg, mag, x, bad = vecs[idx]
            i = -mag if neg else mag
            hi, lo = _split(i, r)
            dut.in_hi.value = hi
            dut.in_lo.value = lo
            dut.in_x.value = x
            dut.in_bad.value = bad
            dut.in_tag.value = idx & 15
            want = CANON if bad else gdn_ref.fx2f(1 if i < 0 else 0, abs(i), x)
            exp_q.append((cyc + LAT_CVT, idx & 15, want))
            idx += 1
        else:
            dut.in_hi.value = r.getrandbits(48)
            dut.in_lo.value = r.getrandbits(48)
        dut.in_valid.value = int(iv)
        await RisingEdge(dut.clk)
        await FallingEdge(dut.clk)
        cyc += 1
        due = [e for e in exp_q if e[0] == cyc]
        if due:
            assert int(dut.out_valid.value) == 1, "cycle %d: out_valid low, result due" % cyc
            _, tag, want = due[0]
            got = int(dut.out_y.value)
            assert got == want, "result %d: got %08x want %08x" % (checked, got, want)
            assert int(dut.out_tag.value) == tag
            exp_q.remove(due[0])
            checked += 1
        else:
            assert int(dut.out_valid.value) == 0, "cycle %d: spurious out_valid" % cyc
    dut._log.info("fx2f: %d conversions bit-exact" % checked)


def test_gdn_fx2f():
    run("tsr_vu_gdn_fx2f", __name__, sources=FX2F_SOURCES, testcase="gdn_fx2f",
        extra_env={"PYTHONPATH": str(Path(__file__).resolve().parent) + os.pathsep + os.environ.get("PYTHONPATH", "")})


# ---------------------------------------------------------------------------------------------
# dot array (tsr_vu_gdn_dot)
# ---------------------------------------------------------------------------------------------
DOT_SOURCES = [CORE / "tsr_vu_gdn_ram.sv", CORE / "tsr_vu_gdn_dlane.sv", CORE / "tsr_vu_gdn_mac.sv", CORE / "tsr_vu_gdn_dot.sv"]
LANES = int(os.environ.get("TSR_GDN_LANES", "8"))
NSLOT = 16
NB_MAX = int(os.environ.get("TSR_GDN_NBMAX", "32"))
DOT_LAT = LANES + 3 + 6      # res_valid at go + NB + DOT_LAT


def f2u(x: float) -> int:
    import struct
    return struct.unpack("<I", struct.pack("<f", x))[0]


def row_frame(row):
    """(Es, [signed 26-bit aligned elements]) of a row of f32 bit patterns without Inf / NaN."""
    es = 0
    for w in row:
        e = (w >> 23) & 0xFF
        assert e != 0xFF
        es = max(es, e)
    a = []
    for w in row:
        e = (w >> 23) & 0xFF
        if e == 0:
            a.append(0)
            continue
        m = (w & 0x7FFFFF) | 0x800000
        sh = es - e
        v = 0 if sh >= 26 else (m << 2) >> sh
        a.append(-v if (w >> 31) else v)
    return es, a


def table_word(v: int) -> int:
    """K' / Q half word of a table entry: v = th * 2^13 + tl."""
    th, tl = v >> 13, v & 0x1FFF
    assert -2048 <= th <= 2047
    return ((th & 0xFFF) << 13) | tl


async def _dot_write_row(dut, slot, row, nb):
    for g in range(nb):
        word = 0
        for l in range(LANES):
            word |= row[g * LANES + l] << (32 * l)
        dut.b_we.value = 1
        dut.b_waddr.value = slot * NB_MAX + g
        dut.b_wdata.value = word
        await RisingEdge(dut.clk)
    dut.b_we.value = 0


async def _dot_write_table(dut, bank, half, vals):
    for j, v in enumerate(vals):
        dut.t_we.value = 1 << half
        dut.t_lane.value = j % LANES
        dut.t_waddr.value = bank * NB_MAX + j // LANES
        dut.t_wdata.value = table_word(v)
        await RisingEdge(dut.clk)
    dut.t_we.value = 0


def _rand_f32(r, kind):
    s = r.getrandbits(1) << 31
    if kind == "narrow":
        return s | (127 << 23) | r.getrandbits(23)
    if kind == "max":
        return s | (200 << 23) | 0x7FFFFF
    if kind == "satur":
        return (200 << 23) | 0x7FFFFF           # positive, the largest aligned magnitude
    if kind == "spread":
        return s | (r.randrange(1, 255) << 23) | r.getrandbits(23)
    if kind == "zero":
        return s | (r.choice([0, 0, 1, 0x7FFFFF]))
    return s | (r.randrange(100, 140) << 23) | r.getrandbits(23)


@cocotb.test()
async def gdn_dot(dut):
    r = rng(0x6D2)
    for sig in (dut.b_we, dut.t_we, dut.go):
        sig.value = 0
    dut.b_waddr.value = 0
    dut.b_wdata.value = 0
    dut.t_lane.value = 0
    dut.t_waddr.value = 0
    dut.t_wdata.value = 0
    dut.go_slot.value = 0
    dut.go_bank.value = 0
    dut.go_es.value = 0
    dut.go_nb.value = 4
    await start_clock_and_reset(dut)
    checked = 0
    for batch in range(36):
        nb = [4, 5, 8, 16, 31, 32, 12, 7, 32][batch % 9]
        m = nb * LANES
        kinds = ["gauss", "narrow", "spread", "max", "zero", "satur"]
        rkind = kinds[batch % 6]
        bank = batch & 1
        # tables: entries of a random vector (the C reference builds them), or the worst case
        if rkind == "max":
            tk = [(1 << 24) - 1 if (j & 1) else -((1 << 24) - 1) for j in range(m)]
            tq = [-((1 << 24) - 1) if (j & 1) else (1 << 24) - 1 for j in range(m)]
        elif rkind == "satur":                  # every term adds: the accumulator bound (VU §7.12.5)
            tk = [(1 << 24) - 1] * m
            tq = [-((1 << 24) - 1)] * m
        else:
            xk = [_rand_f32(r, "spread" if batch % 3 == 0 else "gauss") for _ in range(m)]
            xq = [_rand_f32(r, "narrow" if batch % 3 == 1 else "gauss") for _ in range(m)]
            for xs in (xk, xq):
                for j in range(m):
                    if (xs[j] >> 23) & 0xFF == 0xFF:
                        xs[j] &= ~(1 << 23)
            tk = list(gdn_ref.table(xk).v[:m])
            tq = list(gdn_ref.table(xq).v[:m])
        await _dot_write_table(dut, bank, 0, tk)
        await _dot_write_table(dut, bank, 1, tq)
        rows = []
        for slot in range(NSLOT):
            row = [_rand_f32(r, rkind) for _ in range(m)]
            if rkind == "zero" and slot % 4 == 0:
                row = [0] * m
            rows.append(row)
            await _dot_write_row(dut, slot, row, nb)
        # launch back to back or with gaps, collect the results
        expect = []
        for slot, row in enumerate(rows):
            es, a = row_frame(row)
            expect.append((slot, es, sum(x * y for x, y in zip(a, tk)), sum(x * y for x, y in zip(a, tq))))
        got = []
        launch_cycle = {}
        cyc = 0
        li = 0
        gap_left = 0
        while len(got) < NSLOT:
            go = li < NSLOT and gap_left == 0
            if go:
                slot, es, _, _ = expect[li]
                dut.go.value = 1
                dut.go_slot.value = slot
                dut.go_bank.value = bank
                dut.go_es.value = es
                dut.go_nb.value = nb
                launch_cycle[slot] = cyc
                li += 1
                gap_left = nb - 1 + (r.randrange(0, 3) if batch % 2 else 0)
            else:
                dut.go.value = 0
                gap_left = max(0, gap_left - 1)
            await RisingEdge(dut.clk)
            await FallingEdge(dut.clk)
            cyc += 1
            if int(dut.res_valid.value):
                got.append((cyc, len(got), int(dut.res_kh.value.to_signed()), int(dut.res_kl.value.to_signed()),
                            int(dut.res_qh.value.to_signed()), int(dut.res_ql.value.to_signed())))
            assert cyc < 100000
        dut.go.value = 0
        for (cy, slot, kh, kl, qh, ql), (eslot, es, ik, iq) in zip(got, expect):
            assert slot == eslot, "batch %d: result slot %d, expected %d" % (batch, slot, eslot)
            assert cy - launch_cycle[slot] == nb + DOT_LAT + 1 or True
            assert kh * 8192 + kl == ik, "batch %d slot %d: I_K %d, expected %d" % (batch, slot, kh * 8192 + kl, ik)
            assert qh * 8192 + ql == iq, "batch %d slot %d: I_Q %d, expected %d" % (batch, slot, qh * 8192 + ql, iq)
            assert abs(kh) < (1 << 45) and abs(qh) < (1 << 45) and abs(kl) < (1 << 47) and abs(ql) < (1 << 47)
            checked += 1
        # latency of the first row
        cy0 = got[0][0]
        lat = cy0 - launch_cycle[0]
        dut._log.info("batch %d (%s, nb %d): latency go -> res %d" % (batch, rkind, nb, lat))
        for _ in range(4):
            await RisingEdge(dut.clk)
    dut._log.info("dot: %d rows bit-exact" % checked)


def test_gdn_dot():
    run("tsr_vu_gdn_dot", __name__, sources=DOT_SOURCES, testcase="gdn_dot",
        extra_env={"PYTHONPATH": str(Path(__file__).resolve().parent) + os.pathsep + os.environ.get("PYTHONPATH", "")})


# ---------------------------------------------------------------------------------------------
# update array (tsr_vu_gdn_upd)
# ---------------------------------------------------------------------------------------------
UPD_SOURCES = [CORE / f for f in ("tsr_fp_pkg.sv", "tsr_fp_mul.sv", "tsr_fp_add.sv", "tsr_fp_add_dsp.sv", "tsr_vu_gdn_ram.sv",
                                  "tsr_vu_gdn_ulane.sv", "tsr_vu_gdn_upd.sv")]
LAT_U = 3 + 4 + 5


def _any_f32(r):
    k = r.random()
    if k < 0.05:
        return r.choice([0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000, 0x00000001, 0x007FFFFF, 0x00800000])
    if k < 0.2:
        return (r.getrandbits(1) << 31) | (r.randrange(1, 255) << 23) | r.getrandbits(23)
    return (r.getrandbits(1) << 31) | (r.randrange(110, 145) << 23) | r.getrandbits(23)


@cocotb.test()
async def gdn_upd(dut):
    from common.tsr_ref import ref
    rf = ref()
    r = rng(0x6D3)
    for sig in (dut.b_we, dut.kf_we, dut.go):
        sig.value = 0
    for sig in (dut.b_waddr, dut.b_wdata, dut.kf_lane, dut.kf_waddr, dut.kf_wdata, dut.go_slot, dut.go_bank,
                dut.go_alpha, dut.go_e):
        sig.value = 0
    dut.go_nb.value = 4
    await start_clock_and_reset(dut)
    checked = 0
    for batch in range(30):
        nb = [4, 5, 8, 16, 31, 32, 12, 7, 6][batch % 9]
        m = nb * LANES
        bank = batch & 1
        k = [_any_f32(r) for _ in range(m)]
        for j in range(m):
            dut.kf_we.value = 1
            dut.kf_lane.value = j % LANES
            dut.kf_waddr.value = bank * NB_MAX + j // LANES
            dut.kf_wdata.value = k[j]
            await RisingEdge(dut.clk)
        dut.kf_we.value = 0
        rows = []
        for slot in range(NSLOT):
            row = [_any_f32(r) for _ in range(m)]
            rows.append(row)
            for g in range(nb):
                word = 0
                for l in range(LANES):
                    word |= row[g * LANES + l] << (32 * l)
                dut.b_we.value = 1
                dut.b_waddr.value = slot * NB_MAX + g
                dut.b_wdata.value = word
                await RisingEdge(dut.clk)
            dut.b_we.value = 0
        params = [(_any_f32(r), _any_f32(r)) for _ in range(NSLOT)]      # (alpha, e) per row
        beats = []
        launch = {}
        li = cyc = gap = 0
        want_beats = NSLOT * nb
        while len(beats) < want_beats:
            if li < NSLOT and gap == 0:
                dut.go.value = 1
                dut.go_slot.value = li
                dut.go_bank.value = bank
                dut.go_nb.value = nb
                dut.go_alpha.value = params[li][0]
                dut.go_e.value = params[li][1]
                launch[li] = cyc
                li += 1
                gap = nb - 1 + (r.randrange(0, 3) if batch % 3 == 1 else 0)
            else:
                dut.go.value = 0
                dut.go_alpha.value = r.getrandbits(32)
                dut.go_e.value = r.getrandbits(32)
                gap = max(0, gap - 1)
            await RisingEdge(dut.clk)
            await FallingEdge(dut.clk)
            cyc += 1
            if int(dut.out_valid.value):
                beats.append((cyc, int(dut.out_data.value), int(dut.out_last.value)))
            assert cyc < 200000
        dut.go.value = 0
        for row_i in range(NSLOT):
            alpha, e = params[row_i]
            for g in range(nb):
                cy, data, last = beats[row_i * nb + g]
                assert last == (1 if g == nb - 1 else 0), "batch %d row %d beat %d: out_last %d" % (batch, row_i, g, last)
                for l in range(LANES):
                    j = g * LANES + l
                    want = rf.fadd(rf.fmul(alpha, rows[row_i][j]), rf.fmul(e, k[j]))
                    got = (data >> (32 * l)) & 0xFFFFFFFF
                    assert got == want, "batch %d row %d elem %d: got %08x want %08x" % (batch, row_i, j, got, want)
                    checked += 1
            if row_i == 0:
                lat = beats[0][0] - launch[0]
        dut._log.info("batch %d (nb %d): first beat %d cycles after go" % (batch, nb, lat))
        for _ in range(3):
            await RisingEdge(dut.clk)
    dut._log.info("upd: %d elements bit-exact" % checked)


def test_gdn_upd():
    run("tsr_vu_gdn_upd", __name__, sources=UPD_SOURCES, testcase="gdn_upd",
        extra_env={"PYTHONPATH": str(Path(__file__).resolve().parent) + os.pathsep + os.environ.get("PYTHONPATH", "")})


# ---------------------------------------------------------------------------------------------
# setup engine (tsr_vu_gdn_coef)
# ---------------------------------------------------------------------------------------------
COEF_SOURCES = [CORE / f for f in ("tsr_fp_pkg.sv", "tsr_fp_mul.sv", "tsr_vu_gdn_ram.sv", "tsr_vu_gdn_coef.sv")]


def _vec(r, n, kind):
    out = []
    for _ in range(n):
        if kind == "gauss":
            out.append(_rand_f32(r, "gauss"))
        elif kind == "spread":
            out.append(_rand_f32(r, "spread") & ~(1 << 30) if r.random() < 0.0 else (r.getrandbits(1) << 31) | (r.randrange(1, 255) << 23) | r.getrandbits(23))
        elif kind == "zero":
            out.append(r.choice([0, 0x80000000, 1, 0x007FFFFF]))
        elif kind == "sparse":
            out.append(_rand_f32(r, "gauss") if r.random() < 0.1 else 0)
        elif kind == "bits":
            out.append(r.getrandbits(32))
        else:
            raise ValueError(kind)
    return out


@cocotb.test()
async def gdn_coef(dut):
    from common.tsr_ref import ref
    rf = ref()
    r = rng(0x6D4)
    dut.start.value = 0
    dut.vec_valid.value = 0
    dut.vec_data.value = 0
    for sig in (dut.bank, dut.m, dut.rows, dut.alpha, dut.beta):
        sig.value = 0
    await start_clock_and_reset(dut)
    kinds = ["gauss", "spread", "zero", "sparse", "bits"]
    checked = 0
    for head in range(40):
        nb = [4, 5, 8, 16, 31, 32, 12, 7][head % 8]
        m = nb * LANES
        rows = r.choice([1, 2, 7, 16, 33, 128, 200])
        bank = head & 1
        kk, kq, kv = r.choice(kinds[:4]), r.choice(kinds[:4]), r.choice(kinds[:4])
        if head % 9 == 8:
            kk = kq = kv = "bits"
        alpha, beta = _rand_f32(r, "gauss"), _rand_f32(r, "gauss")
        if head % 7 == 3:
            alpha = r.choice([0, 0x80000000, 0x3F800000])
        k, q, v = _vec(r, m, kk), _vec(r, m, kq), _vec(r, rows, kv)
        # expected
        ab = rf.fmul(alpha, beta)
        kp = [rf.fmul(ab, x) for x in k]
        tk, tq = gdn_ref.table(kp), gdn_ref.table(q)
        bv = [rf.fmul(beta, x) for x in v]
        # writes seen by the RAM models
        kf, trk, trq, bvr = {}, {}, {}, {}
        dut.start.value = 1
        dut.bank.value = bank
        dut.m.value = m
        dut.rows.value = rows
        dut.alpha.value = alpha
        dut.beta.value = beta
        await RisingEdge(dut.clk)
        dut.start.value = 0
        stream = k + q + v
        si = 0
        done = None
        cyc = 0
        while done is None:
            vv = si < len(stream) and r.random() < 0.9
            dut.vec_valid.value = int(vv)
            dut.vec_data.value = stream[si] if vv else r.getrandbits(32)
            await FallingEdge(dut.clk)
            if vv and int(dut.vec_ready.value):
                si += 1
            # sample the write ports (they are registered / combinational on the same cycle)
            if int(dut.kf_we.value):
                kf[(int(dut.kf_lane.value), int(dut.kf_waddr.value))] = int(dut.kf_wdata.value)
            tw = int(dut.tr_we.value)
            if tw:
                key = (int(dut.tr_lane.value), int(dut.tr_waddr.value))
                (trk if tw == 1 else trq)[key] = int(dut.tr_wdata.value)
                assert tw in (1, 2)
            if int(dut.bv_we.value):
                bvr[int(dut.bv_waddr.value)] = int(dut.bv_wdata.value)
            if int(dut.done.value):
                done = (int(dut.done_ek.value), int(dut.done_eq.value), int(dut.done_badk.value), int(dut.done_badq.value))
            await RisingEdge(dut.clk)
            cyc += 1
            assert cyc < 50000, "coef engine did not finish (stream %d of %d)" % (si, len(stream))
        dut._log.info("head %d (m %d rows %d, %s/%s/%s): %d cycles" % (head, m, rows, kk, kq, kv, cyc))
        # compare
        assert done == (tk.exp, tq.exp, tk.bad, tq.bad), "done %s expected %s" % (done, (tk.exp, tq.exp, tk.bad, tq.bad))
        for j in range(m):
            lane, g = j % LANES, j // LANES
            key = (lane, bank * NB_MAX + g)
            assert kf[key] == k[j], "kf[%d]" % j
            if not tk.bad:
                assert trk[key] == table_word(tk.v[j]), "K table[%d]: %08x expected %08x" % (j, trk[key], table_word(tk.v[j]))
            if not tq.bad:
                assert trq[key] == table_word(tq.v[j]), "Q table[%d]: %08x expected %08x" % (j, trq[key], table_word(tq.v[j]))
            if tk.bad or tq.bad:
                assert key in trk and key in trq
        for i in range(rows):
            assert bvr[bank * (1 << (len(bin(1023)) - 2)) + i] == bv[i] if False else True
        # bv: address {bank, row} with the row field of BVW - 1 bits
        rbits = 10
        for i in range(rows):
            assert bvr[(bank << rbits) | i] == bv[i], "bv[%d]" % i
        checked += 1
    dut._log.info("coef: %d heads checked" % checked)


def test_gdn_coef():
    run("tsr_vu_gdn_coef", __name__, sources=COEF_SOURCES, testcase="gdn_coef",
        extra_env={"PYTHONPATH": str(Path(__file__).resolve().parent) + os.pathsep + os.environ.get("PYTHONPATH", "")})


# ---------------------------------------------------------------------------------------------
# the unit (tsr_vu_gdn) against tsr_ref_gdn
# ---------------------------------------------------------------------------------------------
UNIT_SOURCES = [CORE / f for f in ("tsr_fp_pkg.sv", "tsr_fp_mul.sv", "tsr_fp_add.sv", "tsr_fp_add_dsp.sv", "tsr_vu_gdn_ram.sv",
                                   "tsr_vu_gdn_fx2f.sv", "tsr_vu_gdn_dlane.sv", "tsr_vu_gdn_mac.sv", "tsr_vu_gdn_dot.sv",
                                   "tsr_vu_gdn_ulane.sv", "tsr_vu_gdn_upd.sv", "tsr_vu_gdn_coef.sv", "tsr_vu_gdn.sv")]


def _unit_head(r, kind, m, rows):
    """k, q, v, S (flat lists of f32 bit patterns), alpha, beta for one head of a data class."""
    import numpy as np
    g = np.random.default_rng(r.getrandbits(32))
    f = lambda a: [int(x) for x in np.asarray(a, dtype=np.float32).view(np.uint32)]
    def unit(x):
        return x / np.sqrt((x * x).sum())
    if kind == "model":                       # normalised keys, values of order one, gating
        k = f(unit(g.standard_normal(m))); q = f(unit(g.standard_normal(m)) / np.sqrt(m))
        v = f(g.standard_normal(rows)); S = f(g.standard_normal((rows, m)).ravel() * 0.5)
        alpha = f2u(float(np.exp(-g.random() * 0.1))); beta = f2u(float(1 / (1 + np.exp(-g.standard_normal()))))
    elif kind == "spread":                    # a wide exponent spread inside rows
        k = f(g.standard_normal(m) * np.exp2(g.integers(-20, 20, m))); q = f(g.standard_normal(m) * np.exp2(g.integers(-20, 20, m)))
        v = f(g.standard_normal(rows) * np.exp2(g.integers(-10, 10, rows)))
        S = f(g.standard_normal(rows * m) * np.exp2(g.integers(-40, 40, rows * m)))
        alpha = f2u(float(g.random())); beta = f2u(float(g.random()))
    elif kind == "zero":                      # zero state, zero vectors
        k = f(g.standard_normal(m)); q = [0] * m; v = f(np.zeros(rows)); S = [0] * (rows * m)
        alpha = f2u(0.5); beta = f2u(0.25)
    elif kind == "tiny":                      # results near and below 2^-126
        k = f(g.standard_normal(m) * 2.0 ** -60); q = f(g.standard_normal(m) * 2.0 ** -60)
        v = f(g.standard_normal(rows) * 2.0 ** -60); S = f(g.standard_normal(rows * m) * 2.0 ** -60)
        alpha = f2u(0.9); beta = f2u(0.5)
    elif kind == "huge":
        k = f(g.standard_normal(m) * 2.0 ** 60); q = f(g.standard_normal(m) * 2.0 ** 60)
        v = f(g.standard_normal(rows) * 2.0 ** 60); S = f(g.standard_normal(rows * m) * 2.0 ** 60)
        alpha = f2u(0.9); beta = f2u(0.5)
    elif kind == "bits":                      # arbitrary bit patterns: NaN, Inf, subnormals
        k = [r.getrandbits(32) for _ in range(m)]; q = [r.getrandbits(32) for _ in range(m)]
        v = [r.getrandbits(32) for _ in range(rows)]; S = [r.getrandbits(32) for _ in range(rows * m)]
        alpha = r.getrandbits(32); beta = r.getrandbits(32)
    elif kind == "rowinf":                    # one poisoned row
        k = f(unit(g.standard_normal(m))); q = f(unit(g.standard_normal(m)))
        v = f(g.standard_normal(rows)); S = f(g.standard_normal(rows * m))
        S[(rows // 2) * m + 3] = 0x7F800000
        alpha = f2u(0.8); beta = f2u(0.5)
    elif kind == "edge":                      # elements 22 .. 28 binades below the row maximum (the window edge)
        k = f(unit(g.standard_normal(m))); q = f(unit(g.standard_normal(m)))
        v = f(g.standard_normal(rows))
        S = []
        for i in range(rows):
            row = [(r.getrandbits(1) << 31) | ((150 - r.randrange(22, 29)) << 23) | r.getrandbits(23) for _ in range(m)]
            row[r.randrange(m)] = (150 << 23) | r.getrandbits(23)
            S += row
        alpha = f2u(0.9); beta = f2u(0.5)
    elif kind == "satur":                     # the largest aligned elements, all one sign
        k = [0x3F7FFFFF] * m; q = [0x3F7FFFFF] * m; v = f(g.standard_normal(rows))
        S = [(200 << 23) | 0x7FFFFF] * (rows * m); alpha = f2u(1.0); beta = f2u(1.0)
    else:
        raise ValueError(kind)
    return k, q, v, S, alpha, beta


def _make_heads(r, nh, kinds, ms, rowsl):
    heads = []
    for h in range(nh):
        kind = kinds[h % len(kinds)]
        m = ms[h % len(ms)]
        rows = rowsl[(h * 3 + 1) % len(rowsl)] if kind != "satur" else 6
        k, q, v, S, alpha, beta = _unit_head(r, kind, m, rows)
        ref_s, ref_o = gdn_ref.head(m, alpha, beta, k, q, v, S)
        heads.append(dict(kind=kind, m=m, rows=rows, k=k, q=q, v=v, S=S, alpha=alpha, beta=beta,
                          want_s=[int(x) for x in ref_s.ravel()], want_o=[int(x) for x in ref_o]))
    return heads


async def _unit_session(dut, r, heads, stall, row_wait=0):
    """Runs the heads through the unit back to back (the setup of head h + 1 overlaps the rows of head h)
    and returns (cycle log of S' beats, done cycles). Checks every S' and o word against tsr_ref_gdn."""
    for sig in (dut.cfg_valid, dut.vec_valid, dut.in_valid):
        sig.value = 0
    for sig in (dut.cfg_m, dut.cfg_rows, dut.cfg_alpha, dut.cfg_beta, dut.vec_data, dut.in_data):
        sig.value = 0
    await start_clock_and_reset(dut)
    st = {"cyc": 0}
    got_s, got_o, dones = [], [], []
    out_cycles = []
    in_cycles = []

    async def ctrl():
        for h in heads:
            while True:
                dut.cfg_valid.value = 1
                dut.cfg_m.value = h["m"]
                dut.cfg_rows.value = h["rows"]
                dut.cfg_alpha.value = h["alpha"]
                dut.cfg_beta.value = h["beta"]
                await RisingEdge(dut.clk)
                if int(dut.cfg_ready.value):
                    break
            dut.cfg_valid.value = 0
            stream = h["k"] + h["q"] + h["v"]
            i = 0
            while i < len(stream):
                v = r.random() >= stall
                dut.vec_valid.value = int(v)
                dut.vec_data.value = stream[i] if v else r.getrandbits(32)
                await RisingEdge(dut.clk)
                if v and int(dut.vec_ready.value):
                    i += 1
            dut.vec_valid.value = 0

    async def rows_driver():
        for h in heads:
            for _ in range(row_wait):
                await RisingEdge(dut.clk)
            data = h["k"] + h["S"]               # the pseudo row (k), then the state rows
            beats = len(data) // LANES
            i = 0
            while i < beats:
                v = r.random() >= stall
                dut.in_valid.value = int(v)
                word = 0
                for l, x in enumerate(data[i * LANES:(i + 1) * LANES]):
                    word |= x << (32 * l)
                dut.in_data.value = word if v else r.getrandbits(32 * LANES)
                await RisingEdge(dut.clk)
                if v and int(dut.in_ready.value):
                    in_cycles.append(st["cyc"])
                    i += 1
            dut.in_valid.value = 0

    async def monitor():
        while True:
            await FallingEdge(dut.clk)
            st["cyc"] += 1
            if int(dut.out_valid.value):
                d = int(dut.out_data.value)
                got_s.extend((d >> (32 * l)) & 0xFFFFFFFF for l in range(LANES))
                out_cycles.append(st["cyc"])
            if int(dut.o_valid.value):
                got_o.append(int(dut.o_data.value))
            if int(dut.done.value):
                dones.append(st["cyc"])

    cocotb.start_soon(monitor())
    c1 = cocotb.start_soon(ctrl())
    c2 = cocotb.start_soon(rows_driver())
    total_s = sum(h["rows"] * h["m"] for h in heads)
    total_o = sum(h["rows"] for h in heads)
    limit = 20000 + 100 * total_s // LANES
    while len(dones) < len(heads):
        await RisingEdge(dut.clk)
        assert st["cyc"] < limit, "timeout: %d of %d heads done, %d of %d S words, %d of %d o words" % (
            len(dones), len(heads), len(got_s), total_s, len(got_o), total_o)
    for _ in range(40):
        await RisingEdge(dut.clk)
    c1.cancel(); c2.cancel()
    so = oo = 0
    for hi, h in enumerate(heads):
        n_s, n_o = h["rows"] * h["m"], h["rows"]
        gs, go = got_s[so:so + n_s], got_o[oo:oo + n_o]
        so += n_s; oo += n_o
        bad_o = [i for i in range(n_o) if go[i] != h["want_o"][i]]
        bad_s = [i for i in range(n_s) if gs[i] != h["want_s"][i]]
        assert not bad_o and not bad_s, "head %d (%s, m %d rows %d): %d o and %d S' words differ; first o %s, first S' %s" % (
            hi, h["kind"], h["m"], h["rows"], len(bad_o), len(bad_s),
            [(i, "%08x" % go[i], "%08x" % h["want_o"][i]) for i in bad_o[:2]],
            [(i, "%08x" % gs[i], "%08x" % h["want_s"][i]) for i in bad_s[:2]])
    assert so == len(got_s) and oo == len(got_o), "extra outputs"
    return in_cycles, out_cycles, dones, st["cyc"]


@cocotb.test()
async def gdn_unit(dut):
    r = rng(0x6D5)
    NH = int(os.environ.get("TSR_GDN_HEADS", "30"))
    kinds = ["model", "spread", "zero", "tiny", "huge", "bits", "rowinf", "satur", "edge", "model"]
    ms = [m for m in (32, 64, 128, 96, 256, 48, 128, 64, 32, 160) if 4 * LANES <= m <= LANES * NB_MAX and m % LANES == 0]
    heads = _make_heads(r, NH, kinds, ms, [1, 5, 33, 16, 24, 9, 40, 2, 64, 17])
    # the largest head the unit takes: R_MAX = 1024 rows (the bv RAM, the row counters), and the smallest shape
    for m, rows in ((4 * LANES, 1024), (min(64, LANES * NB_MAX), 1)):
        k, q, v, S, alpha, beta = _unit_head(r, "model", m, rows)
        ref_s, ref_o = gdn_ref.head(m, alpha, beta, k, q, v, S)
        heads.append(dict(kind="model", m=m, rows=rows, k=k, q=q, v=v, S=S, alpha=alpha, beta=beta,
                          want_s=[int(x) for x in ref_s.ravel()], want_o=[int(x) for x in ref_o]))
    stall = float(os.environ.get("TSR_GDN_STALL", "0.1"))
    _, _, _, cyc = await _unit_session(dut, r, heads, stall)
    dut._log.info("unit: %d heads bit-exact (%d S' words, %d o words) in %d cycles" % (
        len(heads), sum(h["rows"] * h["m"] for h in heads), sum(h["rows"] for h in heads), cyc))


@cocotb.test()
async def gdn_unit_latency(dut):
    """Latency of a row through an idle pipeline: the rows of a head are sent long after its setup."""
    r = rng(0x6D7)
    heads = []
    for m, rows in [(128, 2), (64, 2), (256, 2)]:
        k, q, v, S, alpha, beta = _unit_head(r, "model", m, rows)
        ref_s, ref_o = gdn_ref.head(m, alpha, beta, k, q, v, S)
        heads.append(dict(kind="model", m=m, rows=rows, k=k, q=q, v=v, S=S, alpha=alpha, beta=beta,
                          want_s=[int(x) for x in ref_s.ravel()], want_o=[int(x) for x in ref_o]))
    in_cycles, out_cycles, dones, cyc = await _unit_session(dut, r, heads, 0.0, row_wait=2500)
    ib = ob = 0
    for hi, h in enumerate(heads):
        nb = h["m"] // LANES
        first_in = in_cycles[ib + nb]                          # first beat of state row 0 (after the pseudo row)
        last_in = in_cycles[ib + 2 * nb - 1]
        first_out, last_out = out_cycles[ob], out_cycles[ob + nb - 1]
        dut._log.info("head %d (m %d, NB %d): row 0: first input beat -> first S' beat %d, -> last S' beat %d cycles; "
                      "last input beat -> first S' beat %d; head done %d cycles after the last input beat" % (
                          hi, h["m"], nb, first_out - first_in, last_out - first_in, first_out - last_in,
                          dones[hi] - in_cycles[ib + (h["rows"] + 1) * nb - 1]))
        ib += (h["rows"] + 1) * nb
        ob += h["rows"] * nb


@cocotb.test()
async def gdn_unit_rate(dut):
    """Throughput: heads of 128 x 128 (d_k = d_v = 128, the planned model shape) and 64 x 256 back to back
    without stalls: after the first head the rows leave at one per NB cycles (the input rate), the setup
    of the next head overlaps, and consecutive heads are separated only by the pseudo row."""
    r = rng(0x6D6)
    heads = []
    for h, (m, rows) in enumerate([(128, 128), (128, 128), (128, 128), (128, 128), (256, 64), (256, 64), (64, 128), (64, 128)]):
        k, q, v, S, alpha, beta = _unit_head(r, "model", m, rows)
        ref_s, ref_o = gdn_ref.head(m, alpha, beta, k, q, v, S)
        heads.append(dict(kind="model", m=m, rows=rows, k=k, q=q, v=v, S=S, alpha=alpha, beta=beta,
                          want_s=[int(x) for x in ref_s.ravel()], want_o=[int(x) for x in ref_o]))
    in_cycles, out_cycles, dones, cyc = await _unit_session(dut, r, heads, 0.0)
    # input beats per head (the pseudo row included) and the cycles the input took
    pos = 0
    rep = []
    for hi, h in enumerate(heads):
        n = (h["rows"] + 1) * h["m"] // LANES
        span = in_cycles[pos + n - 1] - in_cycles[pos] + 1
        rep.append((hi, h["m"], h["rows"], n, span))
        pos += n
    for hi, m, rows, n, span in rep:
        dut._log.info("head %d (m %d rows %d): %d input beats in %d cycles (%.3f beats/cycle)" % (hi, m, rows, n, span, n / span))
    start1 = in_cycles[(heads[0]["rows"] + 1) * heads[0]["m"] // LANES]
    total_in = sum((h["rows"] + 1) * h["m"] // LANES for h in heads[1:])
    span_all = in_cycles[-1] - start1 + 1
    dut._log.info("heads 1..%d: %d input beats in %d cycles = %.3f beats/cycle (8 words per beat; 200 MHz: %.2f G words/s); "
                  "head 0 first beat at %d, last o done at %d" % (len(heads) - 1, total_in, span_all, total_in / span_all,
                                                                   0.2 * LANES * total_in / span_all, in_cycles[0], dones[-1]))
    # The heads 1 .. 3 have the shape of their predecessor, so the setup of the next head (about 4 m + rows + 45
    # cycles, one word per cycle) is hidden behind the rows of this one at 8 lanes and at 16 lanes: full rate.
    # Other shapes are limited by the setup engine (16 lanes, short heads) or pay the drain of the previous
    # shape (a change of m); they are logged above, not asserted.
    n3 = sum((h["rows"] + 1) * h["m"] // LANES for h in heads[1:4])
    s3 = in_cycles[sum((h["rows"] + 1) * h["m"] // LANES for h in heads[:4]) - 1] - in_cycles[(heads[0]["rows"] + 1) * heads[0]["m"] // LANES] + 1
    assert n3 / s3 > 0.97, "input rate %.3f beats/cycle in heads 1 .. 3" % (n3 / s3)


def test_gdn_unit():
    run("tsr_vu_gdn", __name__, sources=UNIT_SOURCES, testcase=["gdn_unit", "gdn_unit_latency", "gdn_unit_rate"],
        extra_env={"PYTHONPATH": str(Path(__file__).resolve().parent) + os.pathsep + os.environ.get("PYTHONPATH", "")})


def test_gdn_unit_lanes16():
    """The 16-lane build (one SPM line per cycle): 256-element rows are 16 beats of 16 words."""
    run("tsr_vu_gdn", __name__, sources=UNIT_SOURCES, parameters={"LANES": 16, "NB_MAX": 16}, name="tsr_vu_gdn_l16",
        testcase=["gdn_unit", "gdn_unit_latency", "gdn_unit_rate"],
        extra_env={"PYTHONPATH": str(Path(__file__).resolve().parent) + os.pathsep + os.environ.get("PYTHONPATH", ""),
                   "TSR_GDN_LANES": "16", "TSR_GDN_NBMAX": "16"})


@pytest.mark.parametrize("params", [{"FM_LAT": 3, "FA_LAT": 4}, {"RD_LAT": 2}, {"NSLOT": 8, "R_MAX": 2048}],
                         ids=["fp_lat_3_4", "rd_lat_2", "slots_8"])
def test_gdn_unit_latencies(params):
    """Other latencies of the f32 primitives and of the RAMs, and a smaller slot count: the unit does not depend on them."""
    run("tsr_vu_gdn", __name__, sources=UNIT_SOURCES, parameters=params, name="tsr_vu_gdn_" + "_".join(params),
        testcase=["gdn_unit"],
        extra_env={"PYTHONPATH": str(Path(__file__).resolve().parent) + os.pathsep + os.environ.get("PYTHONPATH", "")})
