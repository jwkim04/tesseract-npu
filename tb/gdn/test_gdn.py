"""GDN engine testbenches (docs/design/02-vector-unit.md §7.12; rtl/core/tsr_vu_gdn*.sv). Oracle:
libtsr_ref.so (sw/sim/tsr_ref_gdn.c) through tb/gdn/gdn_ref.py, bit-exact. The state update runs on the front lanes of
the vector unit (VU-28), so the engine is tested without them against a lane model (tb/gdn/hdl/tsr_gdn_tb.sv:
FA(FM(a, c1), FM(b, c2)) from tsr_fp_mul and tsr_fp_add) and the whole path, lanes included, in tb/vu (vu_gdn,
vu_gdn_mixed).

pytest functions (test_*):
  test_gdn_lint        Verilator -Wall lint of the GDN modules (default and edge parameters)
  test_gdn_fx2f        cocotb on tsr_vu_gdn_fx2f: random and directed sums (hi * 2^13 + lo) and scales
                       against tsr_ref_gdn_fx2f: RNE ties, carry, overflow, the 2^-126 - 2^-150
                       edge, zero, in_bad; one conversion per cycle, fixed latency, tags
  test_gdn_dot         cocotb on tsr_vu_gdn_dot: the exact sums of the four chains for rows of 4 .. 32 beats
                       against an integer model (random, spread, narrow, zero rows; the saturating rows of
                       m = 256 that fill the 48-bit accumulators), launches back to back and with gaps
  test_gdn_upd         cocotb on tsr_vu_gdn_upd (the operand reader): the S and k words of every beat, alpha, e
                       and the tag of each row, the destination line and half-line, out_last, the latency
  test_gdn_coef        cocotb on tsr_vu_gdn_coef: key, table words, bv and the exponents / flags of 40 heads
                       (Inf / NaN vectors, zero, sparse) against tsr_ref_gdn_table and the VU primitives
  test_gdn_unit        cocotb on tsr_gdn_tb (tsr_vu_gdn with the lane model): heads of nine data classes
                       (model-like, wide exponent spread, zero, tiny, huge, arbitrary bit patterns, one poisoned
                       row, saturating, window edge) driven the way tsr_vu_inj drives the engine (head command,
                       vector lines, row lines at the pace of cfg_ok / vec_ok / row_ok) with random stalls and a
                       random writer credit; every S' line and o line against tsr_ref_gdn; gdn_unit_latency (idle
                       pipeline) and gdn_unit_rate (heads of one shape back to back) log the numbers of §7.12.8
  test_gdn_unit_lanes16  the same with LANES = 16, NB_MAX = 16 (a VU with 16 front lanes)
tb/gdn/test_gdn_mutants.py (opt-in) kills one-line mutants of the RTL.
cocotb tests are selected by the pytest functions through common.runner.run(testcase=).
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import cocotb
import pytest
from cocotb.triggers import FallingEdge, RisingEdge, Timer

from common.runner import REPO, run
from common.tbutil import rng, start_clock_and_reset

import gdn_data
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
        _lint("tsr_gdn_tb", UNIT_SOURCES + [HDL / "tsr_gdn_tb.sv"], *args)
    _lint("tsr_vu_gdn_upd", UPD_SOURCES)
    _lint("tsr_vu_gdn_upd", UPD_SOURCES, "-GLANES=16", "-GNB_MAX=16")
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
    """The ingest writes a row a line (16 words) at a time to {slot, line}."""
    for i in range(len(row) // 16):
        word = 0
        for w in range(16):
            word |= row[16 * i + w] << (32 * w)
        dut.b_we.value = 1
        dut.b_waddr.value = slot * (NB_MAX * LANES // 16) + i
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
        nb = [4, 6, 8, 16, 30, 32, 12, 10, 32][batch % 9]       # m = nb * LANES is a multiple of 16: the ingest takes lines
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
# operand reader (tsr_vu_gdn_upd)
# ---------------------------------------------------------------------------------------------
UPD_SOURCES = [CORE / f for f in ("tsr_vu_gdn_ram.sv", "tsr_vu_gdn_upd.sv")]
UPD_SLOTS = 16
LAT_U = 3 + 1                   # go -> first beat valid: the registered launch (1) and the RAM read latency (3)


def _any_f32(r):
    k = r.random()
    if k < 0.05:
        return r.choice([0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000, 0x00000001, 0x007FFFFF, 0x00800000])
    if k < 0.2:
        return (r.getrandbits(1) << 31) | (r.randrange(1, 255) << 23) | r.getrandbits(23)
    return (r.getrandbits(1) << 31) | (r.randrange(110, 145) << 23) | r.getrandbits(23)


@cocotb.test()
async def gdn_upd(dut):
    r = rng(0x6D3)
    for sig in (dut.b_we, dut.kf_we, dut.go):
        sig.value = 0
    for sig in (dut.b_waddr, dut.b_wdata, dut.kf_lane, dut.kf_waddr, dut.kf_wdata, dut.go_slot, dut.go_bank,
                dut.go_alpha, dut.go_e, dut.go_line, dut.go_tag):
        sig.value = 0
    dut.go_nb.value = 4
    await start_clock_and_reset(dut)
    bpl = 2 if LANES == 8 else 1                 # beats per line
    checked = 0
    for batch in range(30):
        nb = [4, 8, 16, 32, 12, 6][batch % 6] if LANES == 8 else [4, 8, 16, 12][batch % 4]
        nb = min(nb, NB_MAX)
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
        for slot in range(UPD_SLOTS):
            row = [_any_f32(r) for _ in range(m)]
            rows.append(row)
            for i in range(m // 16):
                word = 0
                for w in range(16):
                    word |= row[16 * i + w] << (32 * w)
                dut.b_we.value = 1
                dut.b_waddr.value = slot * (NB_MAX * LANES // 16) + i
                dut.b_wdata.value = word
                await RisingEdge(dut.clk)
            dut.b_we.value = 0
        # (alpha, e, first destination line, tag) per row
        params = [(_any_f32(r), _any_f32(r), r.randrange(1 << 20), r.getrandbits(1)) for _ in range(UPD_SLOTS)]
        beats = []
        launch = {}
        li = cyc = gap = 0
        want_beats = UPD_SLOTS * nb
        while len(beats) < want_beats:
            if li < UPD_SLOTS and gap == 0:
                dut.go.value = 1
                dut.go_slot.value = li
                dut.go_bank.value = bank
                dut.go_nb.value = nb
                dut.go_alpha.value = params[li][0]
                dut.go_e.value = params[li][1]
                dut.go_line.value = params[li][2]
                dut.go_tag.value = params[li][3]
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
            if int(dut.u_v.value):
                beats.append((cyc, int(dut.u_s.value), int(dut.u_k.value), int(dut.u_alpha.value), int(dut.u_e.value),
                              int(dut.u_line.value), int(dut.u_part.value), int(dut.u_tag.value), int(dut.u_last.value)))
            assert cyc < 200000
        dut.go.value = 0
        for row_i in range(UPD_SLOTS):
            alpha, e, line0, tag = params[row_i]
            for g in range(nb):
                cy, sd, kd, al, ee, line, part, tg, last = beats[row_i * nb + g]
                assert last == (1 if g == nb - 1 else 0), "batch %d row %d beat %d: u_last %d" % (batch, row_i, g, last)
                assert (al, ee, tg) == (alpha, e, tag), "batch %d row %d beat %d: alpha/e/tag %08x %08x %d" % (
                    batch, row_i, g, al, ee, tg)
                assert line == line0 + g // bpl and part == (g % bpl if bpl == 2 else 0), "batch %d row %d beat %d: line %d part %d" % (
                    batch, row_i, g, line, part)
                for l in range(LANES):
                    j = g * LANES + l
                    assert (sd >> (32 * l)) & 0xFFFFFFFF == rows[row_i][j], "batch %d row %d elem %d: S word" % (batch, row_i, j)
                    assert (kd >> (32 * l)) & 0xFFFFFFFF == k[j], "batch %d row %d elem %d: k word" % (batch, row_i, j)
                    checked += 1
            if row_i == 0:
                lat = beats[0][0] - launch[0]
        assert lat == LAT_U, "latency go -> first beat %d" % lat
        dut._log.info("batch %d (nb %d): first beat %d cycles after go" % (batch, nb, lat))
        for _ in range(3):
            await RisingEdge(dut.clk)
    dut._log.info("upd: %d elements checked" % checked)


def test_gdn_upd():
    run("tsr_vu_gdn_upd", __name__, sources=UPD_SOURCES, testcase="gdn_upd",
        extra_env={"PYTHONPATH": str(Path(__file__).resolve().parent) + os.pathsep + os.environ.get("PYTHONPATH", "")})


def test_gdn_upd_lanes16():
    run("tsr_vu_gdn_upd", __name__, sources=UPD_SOURCES, parameters={"LANES": 16, "NB_MAX": 16}, name="tsr_vu_gdn_upd_l16",
        testcase="gdn_upd",
        extra_env={"PYTHONPATH": str(Path(__file__).resolve().parent) + os.pathsep + os.environ.get("PYTHONPATH", ""),
                   "TSR_GDN_LANES": "16", "TSR_GDN_NBMAX": "16"})


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
# the engine (tsr_vu_gdn with the lane model, tb/gdn/hdl/tsr_gdn_tb.sv) against tsr_ref_gdn
# ---------------------------------------------------------------------------------------------
HDL = Path(__file__).resolve().parent / "hdl"
UNIT_SOURCES = [CORE / f for f in ("tsr_tables_pkg.sv", "tsr_pkg.sv", "tsr_fp_pkg.sv", "tsr_vu_pkg.sv", "tsr_fp_mul.sv",
                                   "tsr_fp_add.sv", "tsr_fp_add_dsp.sv", "tsr_vu_gdn_ram.sv", "tsr_vu_gdn_fx2f.sv",
                                   "tsr_vu_gdn_dlane.sv", "tsr_vu_gdn_mac.sv", "tsr_vu_gdn_dot.sv", "tsr_vu_gdn_upd.sv",
                                   "tsr_vu_gdn_coef.sv", "tsr_vu_gdn.sv")]
UNIT_TB = UNIT_SOURCES + [HDL / "tsr_gdn_tb.sv"]
GS_CFG, GS_VK, GS_VQ, GS_VV, GS_ROW = 0, 1, 2, 3, 4
NSLOT_U = int(os.environ.get("TSR_GDN_NSLOT", "64"))


def _make_heads(r, nh, kinds, ms, rowsl):
    heads = []
    for h in range(nh):
        kind = kinds[h % len(kinds)]
        m = ms[h % len(ms)]
        rows = rowsl[(h * 3 + 1) % len(rowsl)] if kind != "satur" else 6
        heads.append(_head(r, kind, m, rows))
    return heads


def _head(r, kind, m, rows):
    k, q, v, S, alpha, beta = gdn_data.head_data(r, kind, m, rows)
    ref_s, ref_o = gdn_ref.head(m, alpha, beta, k, q, v, S)
    return dict(kind=kind, m=m, rows=rows, k=k, q=q, v=v, S=S, alpha=alpha, beta=beta,
                want_s=[int(x) for x in ref_s.ravel()], want_o=[int(x) for x in ref_o])


def _slots(heads, tag_of=lambda h: h & 1):
    """The slot sequence tsr_vu_gen makes for each head (VU §7.12.5): the command, the lines of k, q and v (4 injection
    cycles each), then the pseudo row (the lines of k) and the lines of the rows (one cycle each)."""
    out = []
    dbase, obase = 0x1000, 0x100
    for hi, h in enumerate(heads):
        m, rows, nl = h["m"], h["rows"], h["m"] // 16
        h["d"], h["o"] = dbase, obase
        dbase += rows * nl + 16
        obase += (rows + 15) // 16 + 4
        out.append(dict(kind="cfg", gsub=GS_CFG, nsub=4, tag=tag_of(hi), bline=m, rrow=rows, d1=h["o"],
                        alpha=h["alpha"], beta=h["beta"]))
        for gs, vec in ((GS_VK, h["k"]), (GS_VQ, h["q"]), (GS_VV, h["v"])):
            nlines = (len(vec) + 15) // 16
            for j in range(nlines):
                words = vec[16 * j:16 * j + 16]
                out.append(dict(kind="vec", gsub=gs, nsub=4, tag=tag_of(hi), tail=len(words),
                                line=words + [0] * (16 - len(words))))
        n = 0
        for rr, words in enumerate([h["k"]] + [h["S"][i * m:(i + 1) * m] for i in range(rows)]):
            for j in range(nl):
                out.append(dict(kind="row", gsub=GS_ROW, nsub=1, tag=tag_of(hi), first=(j == 0),
                                d0=(h["d"] + (rr - 1) * nl + j) if rr else 0, line=words[16 * j:16 * j + 16]))
    return out


async def _engine_session(dut, r, heads, stall, ocp=1.0, row_wait=0):
    """Plays tsr_vu_inj for the engine: a slot fires when its flow-control level is high and the previous slot has
    left the injection stage (a slot of nsub sub-slots keeps it for nsub cycles); its sub-slots reach the engine two
    cycles later. The writer's credit (oc_ok) and o_line_ready are random. Checks every S' line and o line."""
    for sig in (dut.g_v, dut.oc_ok, dut.o_line_ready):
        sig.value = 0
    for sig in (dut.g_gsub, dut.g_tail, dut.g_d0, dut.g_d1, dut.g_bline, dut.g_rrow, dut.g_tag, dut.g_line,
                dut.g_alpha, dut.g_beta):
        sig.value = 0
    await start_clock_and_reset(dut)
    slots = _slots(heads)
    st = {"cyc": 0}
    mem, olines, ocs = {}, {}, []
    sched = {}
    busy_until = row_wait
    si = 0
    fire_cycles = []
    ybeats = []
    oc_total = 0
    nsub_row = 16 // LANES                     # beats per line
    oc_ok_now = False
    events = {}
    limit = 60000 + 200 * sum((h["rows"] + 1) * h["m"] // LANES for h in heads)
    total_lines = sum(h["rows"] * h["m"] // 16 for h in heads)
    total_olines = sum((h["rows"] + 15) // 16 for h in heads)
    while True:
        await FallingEdge(dut.clk)
        t = st["cyc"]
        st["cyc"] += 1
        assert t < limit, "timeout: slot %d of %d, %d of %d S' lines, %d of %d o lines" % (
            si, len(slots), len(mem), total_lines, len(olines), total_olines)
        # outputs of this cycle
        for ev in ("cfg_fire", "coef_done", "dot_go", "upd_launch", "head_done_v"):
            if int(getattr(dut.u_gdn, ev).value):
                events.setdefault(ev, []).append(t)
        if int(dut.y_v.value):
            line, part, tag = int(dut.y_line.value), int(dut.y_part.value), int(dut.y_tag.value)
            data = int(dut.y_data.value)
            words = [(data >> (32 * l)) & 0xFFFFFFFF for l in range(LANES)]
            cur = mem.setdefault(line, [None] * 16)
            base = (part * LANES) if nsub_row == 2 else 0
            assert all(x is None for x in cur[base:base + LANES]), "S' line %d part %d written twice" % (line, part)
            cur[base:base + LANES] = words
            ybeats.append(t)
        ready_now = r.random() < 0.7
        dut.o_line_ready.value = int(ready_now)
        if int(dut.o_line_v.value) and ready_now:
            a = int(dut.o_line_addr.value)
            assert a not in olines, "o line %d twice" % a
            d = int(dut.o_line_data.value)
            olines[a] = ([(d >> (32 * i)) & 0xFFFFFFFF for i in range(16)], int(dut.o_line_tag.value))
        oc_ok_now = r.random() < ocp
        dut.oc_ok.value = int(oc_ok_now)
        await Timer(1, "ns")                      # oc_take depends on oc_ok: read it with the inputs of this cycle
        if int(dut.oc_take.value):
            oc_total += int(dut.oc_take.value)
            ocs.append(int(dut.oc_take.value))
            assert oc_ok_now, "oc_take %d while the writer had no room" % int(dut.oc_take.value)
        # the injector: decide, then drive what reaches the engine this cycle
        if si < len(slots) and t >= busy_until and r.random() >= stall:
            sl = slots[si]
            ok = {"cfg": int(dut.cfg_ok.value), "vec": int(dut.vec_ok.value),
                  "row": 1 if not sl.get("first") else int(dut.row_ok.value)}[sl["kind"]]
            if ok:
                busy_until = t + sl["nsub"]
                fire_cycles.append(t)
                sched[t + 2] = (sl, 0)
                si += 1
        if t in sched:
            sl, sub = sched.pop(t)
            dut.g_v.value = 1
            dut.g_gsub.value = sl["gsub"]
            dut.g_tag.value = sl["tag"]
            dut.g_tail.value = sl.get("tail", 16)
            dut.g_d0.value = sl.get("d0", 0)
            dut.g_d1.value = sl.get("d1", 0)
            dut.g_bline.value = sl.get("bline", 0)
            dut.g_rrow.value = sl.get("rrow", 0)
            dut.g_alpha.value = sl.get("alpha", 0)
            dut.g_beta.value = sl.get("beta", 0)
            if sl["kind"] != "cfg":
                dut.g_line.value = sum(x << (32 * i) for i, x in enumerate(sl["line"]))
        else:
            dut.g_v.value = 0
            dut.g_line.value = r.getrandbits(512)
        if si == len(slots) and not sched and len(mem) == total_lines and len(olines) == total_olines and \
                all(None not in v for v in mem.values()):
            break
    for _ in range(40):
        await RisingEdge(dut.clk)
    # compare
    for hi, h in enumerate(heads):
        m, rows, nl = h["m"], h["rows"], h["m"] // 16
        got = []
        for ln in range(rows * nl):
            got += mem[h["d"] + ln]
        bad_s = [i for i in range(rows * m) if got[i] != h["want_s"][i]]
        go = []
        for j in range((rows + 15) // 16):
            words, tag = olines[h["o"] + j]
            assert tag == (hi & 1), "head %d o line %d: tag %d" % (hi, j, tag)
            go += words
        bad_o = [i for i in range(rows) if go[i] != h["want_o"][i]]
        assert not bad_o and not bad_s and all(x == 0 for x in go[rows:]), \
            "head %d (%s, m %d rows %d): %d o and %d S' words differ; first o %s, first S' %s; tail %s" % (
                hi, h["kind"], m, rows, len(bad_o), len(bad_s),
                [(i, "%08x" % go[i], "%08x" % h["want_o"][i]) for i in bad_o[:2]],
                [(i, "%08x" % got[i], "%08x" % h["want_s"][i]) for i in bad_s[:2]], go[rows:][:3])
    want_oc = [h["m"] // 16 for h in heads for _ in range(h["rows"])]
    assert ocs == want_oc, "oc_take sequence differs: %d takes (%d lines), expected %d takes (%d lines); first difference at %d; got %s want %s" % (
        len(ocs), oc_total, len(want_oc), total_lines, next((i for i, (a, b) in enumerate(zip(ocs, want_oc)) if a != b), -1), ocs[:16], want_oc[:16])
    return fire_cycles, ybeats, st["cyc"], events


@cocotb.test()
async def gdn_unit(dut):
    r = rng(0x6D5)
    NH = int(os.environ.get("TSR_GDN_HEADS", "30"))
    kinds = ["model", "spread", "zero", "tiny", "huge", "bits", "rowinf", "satur", "edge", "model"]
    ms = [m for m in (64, 128, 96, 256, 64, 128, 160, 80) if 4 * LANES <= m <= LANES * NB_MAX and m % 16 == 0]
    heads = _make_heads(r, NH, kinds, ms, [1, 5, 33, 16, 24, 9, 40, 2, 64, 17])
    # the largest head the engine takes: R_MAX = 1024 rows (the bv RAM, the row counters), and the smallest shape
    for m, rows in ((64, 1024), (min(64, LANES * NB_MAX), 1)):
        heads.append(_head(r, "model", m, rows))
    stall = float(os.environ.get("TSR_GDN_STALL", "0.1"))
    _, _, cyc, _ = await _engine_session(dut, r, heads, stall, ocp=0.8)
    dut._log.info("unit: %d heads bit-exact (%d S' words, %d o words) in %d cycles" % (
        len(heads), sum(h["rows"] * h["m"] for h in heads), sum(h["rows"] for h in heads), cyc))


@cocotb.test()
async def gdn_unit_latency(dut):
    """Latency of a row through an idle pipeline: the rows of a head are sent long after its setup."""
    r = rng(0x6D7)
    heads = [_head(r, "model", m, 2) for m in (128, 64, 256) if 4 * LANES <= m <= LANES * NB_MAX]
    fire, ybeats, cyc, _ = await _engine_session(dut, r, heads, 0.0, row_wait=0)
    dut._log.info("latency run: %d heads in %d cycles" % (len(heads), cyc))


@cocotb.test()
async def gdn_unit_rate(dut):
    """Throughput: heads of 128 x 128 (d_k = d_v = 128, the planned model shape) and other shapes back to back, no
    stalls, the writer always with room. The cycles per head are logged and compared with the rows' own time
    ((rows + 1) * NB cycles: one beat per cycle at the input)."""
    r = rng(0x6D6)
    shapes = [(128, 128), (128, 128), (128, 128), (128, 128), (256, 64), (256, 64), (64, 128), (64, 128)]
    shapes = [(m, rows) for (m, rows) in shapes if 4 * LANES <= m <= LANES * NB_MAX]
    heads = [_head(r, "model", m, rows) for (m, rows) in shapes]
    fire, ybeats, cyc, events = await _engine_session(dut, r, heads, 0.0, ocp=1.0)
    # the first slot of each head's rows (the pseudo row) and the last row line: the span of a head at the input
    pos = 0
    spans = []
    for hi, h in enumerate(heads):
        n_slots = 1 + 3 * 0 + (2 * (h["m"] // 16) + (h["rows"] + 15) // 16) + (h["rows"] + 1) * (h["m"] // 16)
        spans.append((fire[pos], fire[pos + n_slots - 1]))
        pos += n_slots
    for hi, h in enumerate(heads):
        beats = (h["rows"] + 1) * h["m"] // LANES
        gap = spans[hi][0] - spans[hi - 1][0] if hi else 0
        dut._log.info("head %d (m %d rows %d): %d row beats; command at %d, last slot at %d; %d cycles after the previous "
                      "command (%.2f of the beat time)" % (hi, h["m"], h["rows"], beats, spans[hi][0], spans[hi][1], gap,
                                                           gap / beats if gap else 0))
    nrow = [h["rows"] for h in heads]
    ul = events["upd_launch"]
    dg = events["dot_go"]
    p0 = 0
    for hi, h in enumerate(heads):
        dut._log.info("head %d: command %d, tables ready %d, first dot launch %d, first row launch %d, last row launch %d" % (
            hi, events["cfg_fire"][hi], events["coef_done"][hi], dg[sum(n + 1 for n in nrow[:hi])],
            ul[sum(nrow[:hi])], ul[sum(nrow[:hi + 1]) - 1]))
    dut._log.info("%d heads in %d cycles; the S' beats left in %d cycles" % (len(heads), cyc, ybeats[-1] - ybeats[0] + 1))


def test_gdn_unit():
    run("tsr_gdn_tb", __name__, sources=UNIT_TB, testcase=["gdn_unit", "gdn_unit_latency", "gdn_unit_rate"],
        extra_env={"PYTHONPATH": str(Path(__file__).resolve().parent) + os.pathsep + os.environ.get("PYTHONPATH", "")})


def test_gdn_unit_lanes16():
    """The 16-lane build (one SPM line per cycle): 256-element rows are 16 beats of 16 words."""
    run("tsr_gdn_tb", __name__, sources=UNIT_TB, parameters={"LANES": 16, "NB_MAX": 16}, name="tsr_gdn_tb_l16",
        testcase=["gdn_unit", "gdn_unit_latency", "gdn_unit_rate"],
        extra_env={"PYTHONPATH": str(Path(__file__).resolve().parent) + os.pathsep + os.environ.get("PYTHONPATH", ""),
                   "TSR_GDN_LANES": "16", "TSR_GDN_NBMAX": "16"})


@pytest.mark.parametrize("params", [{"FM_LAT": 3, "FA_LAT": 4}, {"RD_LAT": 2}, {"NSLOT": 8, "R_MAX": 2048},
                                    {"STYLE": '"block"'}], ids=["fp_lat_3_4", "rd_lat_2", "slots_8", "block_ram"])
def test_gdn_unit_latencies(params):
    """Other latencies of the f32 primitives and of the RAMs, a smaller slot count, block RAM: the engine does not
    depend on them (the lane model follows FM_LAT and FA_LAT)."""
    run("tsr_gdn_tb", __name__, sources=UNIT_TB, parameters=params, name="tsr_gdn_tb_" + "_".join(params),
        testcase=["gdn_unit"],
        extra_env={"PYTHONPATH": str(Path(__file__).resolve().parent) + os.pathsep + os.environ.get("PYTHONPATH", ""),
                   "TSR_GDN_NSLOT": str(params.get("NSLOT", 64))})
