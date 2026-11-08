"""Mutation spot checks of the GDN unit (docs/design/02-vector-unit.md §7.12.10; the style of
tb/vu/test_vu_mutants.py): each mutant is a one-line change of one rtl/core/tsr_vu_gdn*.sv file written to a
private copy (build/cocotb/gdn_mutants/<id>/); the bench of that block runs on the copy and must fail.

Benches: "fx2f", "dot", "upd", "coef" the block benches of tb/gdn/test_gdn.py, "unit" the whole unit against
tsr_ref_gdn (gdn_unit: 30 heads of nine data classes).

Opt-in (about 3 minutes): TSR_GDN_MUTANTS=1, or -m slow, or -k mutants.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from common.runner import REPO, build_root, run

import test_gdn as T

CORE = REPO / "rtl" / "core"

# (id, file, old text (unique in the file), new text, bench, what)
MUTANTS = [
    ("control", "tsr_vu_gdn_dlane.sv", "(sh9[7:0] > 8'd25)", "(sh9[7:0] > 8'd25)", "unit",
     "no change: the harness must pass (a failing build or import would otherwise look like a kill)"),
    ("dl_window", "tsr_vu_gdn_dlane.sv", "(sh9[7:0] > 8'd25)", "(sh9[7:0] > 8'd24)", "unit",
     "an element 25 binades below the row maximum is dropped"),
    ("dl_sign", "tsr_vu_gdn_dlane.sv", "assign d3_s   = d2_sign & ~d2_zero;", "assign d3_s   = 1'b0;", "dot",
     "the sign of the row element is lost"),
    ("mac_lo_sign", "tsr_vu_gdn_mac.sv", "kl_1[l] <= signed'({1'b0, tw[64*l +: 13]});",
     "kl_1[l] <= 14'(signed'(tw[64*l +: 13]));", "dot", "the low table piece is taken as signed"),
    ("mac_restart", "tsr_vu_gdn_mac.sv", "acc_qh <= (acc_first ? 48'sd0 : acc_qh) + pqh[LANES-1];",
     "acc_qh <= acc_qh + pqh[LANES-1];", "dot", "the Q accumulator is not restarted for a new row"),
    ("cvt_tie", "tsr_vu_gdn_fx2f.sv", "assign inc_c = nm5[MW-25] & ((|nm5[MW-26:0]) | q_c[0]);",
     "assign inc_c = nm5[MW-25] & (|nm5[MW-26:0]);", "fx2f", "ties are not rounded to even"),
    ("cvt_ftz", "tsr_vu_gdn_fx2f.sv", "end else if (e0_6 == 12'sd0 && allones6) begin",
     "end else if (e0_6 == 12'sd0 && 1'b0) begin", "fx2f", "2^-126 - 2^-150 flushes to zero"),
    ("cvt_ovf", "tsr_vu_gdn_fx2f.sv", "if (e1_c >= 12'sd255) y_c = {sgn6, 8'hFF, 23'h0};",
     "if (e1_c >= 12'sd256) y_c = {sgn6, 8'hFF, 23'h0};", "fx2f", "overflow by the rounding carry is not Inf"),
    ("coef_round", "tsr_vu_gdn_coef.sv", "t2_mag <= 24'((t1_m + 25'd1) >> 1);", "t2_mag <= 24'(t1_m >> 1);", "coef",
     "table entries truncate instead of rounding half up"),
    ("coef_sign", "tsr_vu_gdn_coef.sv", "t3_w <= (t2_s && t2_mag != 24'd0) ? t3_neg : {1'b0, t2_mag};",
     "t3_w <= {1'b0, t2_mag};", "coef", "table entries lose their sign"),
    ("upd_alpha", "tsr_vu_gdn_upd.sv", "if (go_d[RD_LAT-1]) begin", "if (go_d[RD_LAT-2]) begin", "upd",
     "alpha and e reach the multipliers one beat early"),
    ("ul_sub", "tsr_vu_gdn_ulane.sv", ".sub(1'b0), .out_valid, .y);", ".sub(1'b1), .out_valid, .y);", "upd",
     "the update subtracts"),
    ("top_fs", "tsr_vu_gdn.sv", "assign fa_a = fs_issue ? bv_rdata : x2_q[x2_rp[2:0]];",
     "assign fa_a = fs_issue ? ud_u[RD_LAT-1] : x2_q[x2_rp[2:0]];", "unit", "e = u' - bv instead of bv - u'"),
    ("top_kappa", "tsr_vu_gdn.sv", "assign fm_b = fm_issueB ? bk_kappa[eb_bank] : wa_y;",
     "assign fm_b = fm_issueB ? bk_kappa[~eb_bank] : wa_y;", "unit", "kappa of the other bank"),
    ("top_pseudo", "tsr_vu_gdn.sv",
     "if (cvo_v & cvo_kind & cvo_pseudo) e_valid[cvo_slot] <= 1'b1;     // the pseudo row is over: skip it",
     "if (dot_go & ma_pseudo) e_valid[dot_slot] <= 1'b1;", "unit", "the pseudo row's slot is freed while its dot reads"),
    ("top_emax", "tsr_vu_gdn.sv", "always_ff @(posedge clk) lvl[i] <= (a_e > b_e) ? a_e : b_e;",
     "always_ff @(posedge clk) lvl[i] <= (a_e < b_e) ? a_e : b_e;", "unit", "the row exponent is the smallest"),
]

BENCH = {
    "fx2f": (T.FX2F_SOURCES, "tsr_vu_gdn_fx2f", "gdn_fx2f"),
    "dot": (T.DOT_SOURCES, "tsr_vu_gdn_dot", "gdn_dot"),
    "upd": (T.UPD_SOURCES, "tsr_vu_gdn_upd", "gdn_upd"),
    "coef": (T.COEF_SOURCES, "tsr_vu_gdn_coef", "gdn_coef"),
    "unit": (T.UNIT_SOURCES, "tsr_vu_gdn", "gdn_unit"),
}


def _wanted():
    return os.environ.get("TSR_GDN_MUTANTS") == "1" or "mutants" in " ".join(sys.argv) or "slow" in " ".join(sys.argv)


@pytest.mark.slow
@pytest.mark.parametrize("mid", [m[0] for m in MUTANTS])
def test_gdn_mutants(mid):
    """The mutant is killed: the bench of its block fails on it (the unmutated RTL passes them in test_gdn.py)."""
    if not _wanted():
        pytest.skip("opt-in: TSR_GDN_MUTANTS=1 or -k mutants")
    _, fname, old, new, bench, what = next(x for x in MUTANTS if x[0] == mid)
    src = (CORE / fname).read_text()
    assert src.count(old) == 1, "mutant %s: the original text is not unique in %s (RTL changed?)" % (mid, fname)
    d = build_root() / "gdn_mutants" / mid
    d.mkdir(parents=True, exist_ok=True)
    (d / fname).write_text(src.replace(old, new))
    srcs, top, case = BENCH[bench]
    srcs = [d / fname if Path(s).name == fname else s for s in srcs]
    killed = False
    try:
        run(top, "test_gdn", sources=srcs, testcase=case, name="mut_" + mid,
            extra_env={"PYTHONPATH": str(Path(T.__file__).resolve().parent) + os.pathsep + os.environ.get("PYTHONPATH", "")})
    except (SystemExit, RuntimeError, AssertionError, TimeoutError):
        killed = True
    print("MUTANT %s (%s): %s" % (mid, what, "killed by " + bench if killed else "SURVIVED"))
    if mid == "control":
        assert not killed, "the unmutated copy fails the %s bench: the harness is broken" % bench
        return
    assert killed, "mutant %s (%s) survived the %s bench" % (mid, what, bench)
