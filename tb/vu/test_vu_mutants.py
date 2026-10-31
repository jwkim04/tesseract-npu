"""Mutation spot checks of the vector unit (docs/design/02-vector-unit.md "VU" §10; the style of
tb/issue/test_issue_mutants.py): each mutant is a one-line change of one VU file written to a private
copy (build/cocotb/vu_mutants/<id>/); the bench of that block runs on it ($TSR_MUT_DIR) and must fail. A
mutant that does not build counts as not killed (the build error is printed).

Benches: "chk" the checker class (tb/vu/test_vu.py, 3000 random + the directed ops, U200 and small);
"arith" the EXPK / RC / SQ / SIN-COS cores (vu_sweep, reduced sweeps); "lane" the lane functions
(vu_lane_sweep, reduced); "unit:<tests>" the whole unit against the simulator (tb/vu/test_vu_unit.py,
the named cocotb tests at half size).

Opt-in (about 1.5 hours with TSR_BUILD_JOBS=2): TSR_VU_MUTANTS=1, or -m slow, or -k mutants.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common.runner import REPO, build_root   # noqa: E402

# (id, file, original text, mutated text, bench, the rule it breaks)
MUTANTS = [
    # ---- tsr_vu_chk (checker class) ----
    ("chk_topk_k16", "tsr_vu_chk.sv", "if (kk == 5'd0 || kk > 5'd16) begin", "if (kk == 5'd0 || kk >= 5'd16) begin",
     "chk", "TOPK k = 16 is legal"),
    ("chk_dq_ss0", "tsr_vu_chk.sv", "ssp = (w_q[7][15:0] != 16'd0) ? 32'(w_q[7][15:0]) : rb_q;",
     "ssp = 32'(w_q[7][15:0]);", "chk", "DQROW src_stride 0 = ROW_BEATS"),
    ("chk_vop_dstride", "tsr_vu_chk.sv", "if (rows > 16'd1 && 32'(w_q[6][31:16]) < lm_q) begin",
     "if (rows > 16'd2 && 32'(w_q[6][31:16]) < lm_q) begin", "chk", "VOP rows > 1 and d_stride < L(m): OPERAND"),
    ("chk_rope_table", "tsr_vu_chk.sv", "c2_len[3] = AW'(32'((32'(nd) + 32'd31) >> 5));",
     "c2_len[3] = AW'(32'((32'(nd) + 32'd31) >> 6));", "chk", "ROPE table extent ceil(n_dims / 32)"),
    ("chk_scale_sign", "tsr_vu_chk.sv", "return se[8] || se[7:0] == 8'd0 || se[7:0] == 8'hFF;",
     "return se[7:0] == 8'd0 || se[7:0] == 8'hFF;", "chk", "AT scale > 0 (AT §5.4)"),
    # ---- arithmetic cores ----
    ("exp_p2", "tsr_vu_exp.sv", "localparam logic [31:0] C_P2F = 32'h3D7F7BFF;",
     "localparam logic [31:0] C_P2F = 32'h3D7F7BFE;", "arith", "EXPK polynomial coefficient (1 ulp of P2)"),
    ("rcp_ftz", "tsr_vu_rcp.sv", "else if (fz ? (e >= 8'd254) : (e >= 8'd253)) r1_val = {x0_q[31], 31'd0};",
     "else if (fz ? (e >= 8'd254) : (e >= 8'd254)) r1_val = {x0_q[31], 31'd0};", "arith",
     "RC flushes results below 2^-126 (x >= 2^126·m)"),
    ("rcp_seed", "tsr_vu_rcp.sv", "(32'd513 + 32'(k) * 32'd2)) + 32'd1) >> 1);", "(32'd515 + 32'(k) * 32'd2)) + 32'd1) >> 1);",
     "arith", "RC seed off by one segment (Newton converges, the correction must not hide it)"),
    ("sqrt_ge", "tsr_vu_sqrt.sv", "      if (r >= t) begin", "      if (r > t) begin", "arith",
     "restoring square root digit test"),
    ("sincos_sn2", "tsr_vu_sincos.sv", "SN2 = 32'h3C08839E", "SN2 = 32'h3C08839F", "arith",
     "SIN polynomial coefficient (1 ulp)"),
    # ---- tsr_vu_lane ----
    ("lane_tanh_small", "tsr_vu_lane.sv", "fl1_q.tsmall <= (ad[30:0] < 31'h3C80_0000);",
     "fl1_q.tsmall <= (ad[30:0] < 31'h3D00_0000);", "lane", "TANH small path |x| < 2^-6"),
    ("lane_le10", "tsr_vu_lane.sv", "fl1_q.le10  <= !tsr_f32_is_nan(a0_q) && ad[31] && (ad[30:0] >= TSR_VU_TEN[30:0]);",
     "fl1_q.le10  <= !tsr_f32_is_nan(a0_q) && ad[31] && (ad[30:0] > TSR_VU_TEN[30:0]);", "lane",
     "GELU16 clamp x <= -10 -> +0 (GELU(-10) itself rounds to -0; x >= 10 is an equivalent mutant: GELU(10) = 10)"),
    ("lane_rot_neg", "tsr_vu_lane.sv", "U_ROT:              begin fa_x = fma_y; fa_y_in = fmb_y; fa_sub = neg6; end",
     "U_ROT:              begin fa_x = fma_y; fa_y_in = fmb_y; fa_sub = 1'b0; end", "lane", "ROT subtracts when neg"),
    ("lane_rmax_init", "tsr_vu_lane.sv", "(u11 == U_RMAX) ? TSR_VU_NINF : 32'd0;", "(u11 == U_RMAX) ? 32'd0 : 32'd0;",
     "lane", "RMAX partials start at -inf"),
    ("lane_expsub_inf", "tsr_vu_lane.sv", "U_EXPSUB, U_EXPSUBC:    f_y <= fl41.aninf ? 32'd0 : e_y;",
     "U_EXPSUB, U_EXPSUBC:    f_y <= e_y;", "lane", "EXPSUB -inf rule (VU §6.5)"),
    # ---- tsr_vu_red ----
    ("red_unpaired", "tsr_vu_red.sv", "assign ib = h_v ? tx[k] : (tm[k] ? tx[k] : 32'd0);",
     "assign ib = h_v ? tx[k] : (tm[k] ? tx[k] : 32'h8000_0000);", "unit:vu_red",
     "the unpaired result passes z(x) = FA(x, +0)"),
    ("red_rms_mode", "tsr_vu_red.sv", "2'd2:    p_m = 4'b1111;               // RMS",
     "2'd2:    p_m = 4'b0111;               // RMS", "unit:vu_red", "RMS = RC(SQ(FA(FM(y, sk), se)))"),
    ("red_rsqrt_mode", "tsr_vu_red.sv", "(opk[sopx_tag].post == 2'd1) ? 4'b0010 : 4'b0011;",
     "(opk[sopx_tag].post == 2'd1) ? 4'b0010 : 4'b0010;", "unit:vu_sop", "SOP RSQRT = RC(SQ(x))"),
    # ---- tsr_vu_gen ----
    ("gen_dq_chunk", "tsr_vu_gen.sv", "cb_q      <= cb_q + (dqf_q ? TSR_VU_LW'(9) : TSR_VU_LW'(17));",
     "cb_q      <= cb_q + (dqf_q ? TSR_VU_LW'(9) : TSR_VU_LW'(18));", "unit:vu_dqrow", "Q8_0 chunk = 17 lines"),
    ("gen_amask_query", "tsr_vu_gen.sv", "if (qc_q == rpq_q - 16'd1) begin", "if (qc_q == rpq_q) begin", "unit:vu_at",
     "AMASK mask row floor(r / rpq)"),
    ("gen_amerge_bases", "tsr_vu_gen.sv", "wire bases_ok = (bset_q == 2'd0) && (amp_d_q == amp_q);",
     "wire bases_ok = 1'b1;", "unit:vu_at", "AMERGE pass bases settle before use"),
    # ---- tsr_vu_inj ----
    ("inj_exit_order", "tsr_vu_inj.sv", "if (has_exit && lat < ex_rem_q) ops_in = 1'b0;",
     "if (has_exit && lat + 8'd1 < ex_rem_q) ops_in = 1'b0;", "unit:vu_vop_funcs|vu_stream",
     "exit order: a slot's lines leave after every earlier one"),
    ("inj_dq_wait", "tsr_vu_inj.sv", "if (dq_line && dq_prod <= 6'(s.dqb)) ops_in = 1'b0;",
     "if (1'b0) ops_in = 1'b0;", "unit:vu_dqrow", "a DQROW line waits for its constant-table entry"),
    ("inj_dq_nibble", "tsr_vu_inj.sv", "o_a[l]   = tsr_vu_u4_to_f32(i_s.dq[1] ? qb[7:4] : qb[3:0]);",
     "o_a[l]   = tsr_vu_u4_to_f32(i_s.dq[1] ? qb[3:0] : qb[7:4]);", "unit:vu_dqrow", "Q4_K nibble order"),
    ("inj_norm_sign", "tsr_vu_inj.sv", "        o_neg[l] = (l % 2 == 0);\n      end\n      // DQROW",
     "        o_neg[l] = (l % 2 == 1);\n      end\n      // DQROW", "unit:vu_rope", "ROPE NORM rotation signs"),
    # ---- tsr_vu_rd / tsr_vu_wr ----
    ("rd_bank_b", "tsr_vu_rd.sv", "if (wb) qm[b][qwp[b] + 2'(wa)] <= {1'b1, AW'(s_slot.bline)};",
     "if (wb) qm[b][qwp[b] + 2'(wa)] <= {1'b1, AW'(s_slot.aline)};", "unit:vu_smoke", "operand b's line"),
    ("wr_keep", "tsr_vu_wr.sv", "for (int i = 0; i < 16; i++) o_be[4*i +: 4] <= (32'(i) < 32'(xd.tail)) ? 4'hF : 4'h0;",
     "for (int i = 0; i < 16; i++) o_be[4*i +: 4] <= (32'(i) <= 32'(xd.tail)) ? 4'hF : 4'h0;", "unit:vu_at",
     "AT row vectors keep the words past R"),
    # ---- tsr_vu_angle / tsr_vu_topk / tsr_vu_dq ----
    ("ang_cs_swap", "tsr_vu_angle.sv", "mem[{wr_h, wr_i[8:4]}] <= {mc, ms};", "mem[{wr_h, wr_i[8:4]}] <= {ms, mc};",
     "unit:vu_rope", "angle buffer {c, s}"),
    ("ang_pos_line", "tsr_vu_angle.sv", "wire        pos_ok = posv_q[t_q[4]] && (posl_q[t_q[4]] == {4'd0, t_q[15:4]});",
     "wire        pos_ok = posv_q[t_q[4]];", "unit:vu_rope", "a token waits for its position line"),
    ("topk_tie", "tsr_vu_topk.sv", "    return ja < jb;", "    return ja > jb;", "unit:vu_topk",
     "TOPK ties to the smaller index"),
    ("topk_tail", "tsr_vu_topk.sv", "      if (5'(i) >= k_q) begin", "      if (5'(i) > k_q) begin", "unit:vu_topk",
     "the words past 2k are 0"),
    ("dq_scale_hi", "tsr_vu_dq.sv", "sc_q <= {q[j - 4][7:6], q[j + 4][3:0]};", "sc_q <= {q[j - 4][7:6], q[j + 4][7:4]};",
     "unit:vu_dqrow", "tsr_get_scale_min_k4 for j >= 4"),
    ("dq_no_d1", "tsr_vu_dq.sv", "tab[ew] <= fmt_q ? {d1, mw, dmw} : {dw, 64'd0};",
     "tab[ew] <= fmt_q ? {dw, mw, dmw} : {dw, 64'd0};", "unit:vu_dqrow", "Q4_K d1 = FM(d, sc)"),
    # ---- tsr_vu (held ops, hazards) ----
    ("top_sopx_barrier", "tsr_vu.sv", "h = ((a.sw & (b.sr | b.sw)) != 16'd0) || a.sopx || b.sopx || (a.rope && b.rope);",
     "h = ((a.sw & (b.sr | b.sw)) != 16'd0) || b.sopx || (a.rope && b.rope);", "unit:vu_sop|vu_stream",
     "an SOP SQRT / RSQRT / DIV is a barrier on both sides"),
    ("top_raw_b", "tsr_vu.sv", "for (int j = 0; j < 4; j++) h |= ovl(a.wlo[i], a.whi[i], b.rlo[j], b.rhi[j]);",
     "for (int j = 0; j < 1; j++) h |= ovl(a.wlo[i], a.whi[i], b.rlo[j], b.rhi[j]);", "unit:vu_stream|vu_decode",
     "RAW hazard on every read extent"),
    ("top_dq_extent", "tsr_vu.sv", "dc1_len[0] <= EXW'(lk); dc1_nz[0] <= rows != 16'd0;",
     "dc1_len[0] <= EXW'(lk); dc1_nz[0] <= 1'b0;", "unit:vu_dqrow", "DQROW's write extent in the hazard check"),
]


def _wanted():
    return os.environ.get("TSR_VU_MUTANTS") == "1" or "mutants" in " ".join(sys.argv) or "slow" in " ".join(sys.argv)


def _benches(kind):
    """The bench runs of a block: (label, callable)."""
    if kind == "chk":
        import test_vu as t
        return [("chk u200", lambda: t._run_chk("mut", 4 << 20, 32768, 3000)),
                ("chk small", lambda: t._run_chk("mut_small", 64 << 10, 4096, 3000, num_ch=3))]
    if kind == "arith":
        import test_vu_arith as t
        seed = int(os.environ.get("TSR_SEED", "1"), 0)

        def arith():
            exe = t.arith_exe()
            # exp_rand 2^22 per kappa: a 1-ulp change of P2 (2^-32) moves about one result in 2^24
            t.run_jobs(exe, [("specials",), ("exp_rand", 1 << 22, seed), ("rcp", *t.EDGE_EXPS[:4]),
                             ("rcp", *t.EDGE_EXPS[4:]), ("sqrt", "1", "127", "128", "254"), ("rand", 1 << 20, seed),
                             ("sincos_rand", 1 << 18, seed)])
        return [("arith", arith)]
    if kind == "lane":
        import test_vu_arith as t
        seed = int(os.environ.get("TSR_SEED", "1"), 0)

        def lane():
            exe = t.lane_exe()
            t.run_jobs(exe, [("vop", 1 << 15, seed), ("gelu16", seed), ("at", 1 << 15, seed), ("red", 1000, seed)])
        return [("lane", lane)]
    assert kind.startswith("unit:"), kind
    import test_vu_unit as t
    tests = kind[5:]
    return [("unit " + tests, lambda: t._run("mut", n=0.5, testcase=tests))]


def _fails(fn):
    try:
        fn()
    except (SystemExit, RuntimeError, AssertionError, TimeoutError):
        return True
    return False


@pytest.mark.slow
@pytest.mark.parametrize("mid", [m[0] for m in MUTANTS])
def test_vu_mutants(mid):
    """The mutant is killed: some bench of its block fails on it (the unmutated RTL passes these benches
    in tb/vu/test_vu.py, test_vu_arith.py and test_vu_unit.py)."""
    if not _wanted():
        pytest.skip("opt-in: TSR_VU_MUTANTS=1 or -k mutants")
    m = next(x for x in MUTANTS if x[0] == mid)
    _, fname, old, new, kind, what = m
    src = (REPO / "rtl/core" / fname).read_text()
    assert src.count(old) == 1, "mutant %s: the original text is not unique in %s (RTL changed?)" % (mid, fname)
    d = build_root() / "vu_mutants" / mid
    d.mkdir(parents=True, exist_ok=True)
    (d / fname).write_text(src.replace(old, new))
    os.environ["TSR_MUT_DIR"] = str(d)
    try:
        killers = [label for label, fn in _benches(kind) if _fails(fn)]
    finally:
        del os.environ["TSR_MUT_DIR"]
    print("MUTANT %s (%s): killed by %s" % (mid, what, killers or "nothing"))
    assert killers, "mutant %s (%s) survived %s" % (mid, what, kind)
