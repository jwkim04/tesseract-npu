#!/usr/bin/env python3
"""Mutation campaign for the MVM back end (tsr_mvm_back, tsr_mvm_actbuf, tsr_mac_lane) and its tests.

Every mutant is one textual change (it must match exactly once) applied to a scratch copy of rtl/;
the tests of tb/mvmback run against the copy (TSR_MVMBACK_RTL = the copy's filelist) in the order
given per mutant and stop at the first failing pytest item. A mutant is KILLED when an item fails
(wrong result, protocol / throughput / latency check, RTL assertion, timeout), SURVIVED otherwise.
The repository is never modified.

    python3 tb/mvmback/mutate_back.py <scratch dir> [mutant id ...]
    python3 tb/mvmback/mutate_back.py --check          (patterns only)

Results: <scratch dir>/results.json and a table on stdout. One simulation build at a time;
the per-mutant build directory is deleted afterwards.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BACK = "core/tsr_mvm_back.sv"
ACT = "core/tsr_mvm_actbuf.sv"
MAC = "tech/generic/tsr_mac_lane.sv"

CFG_ORDER = ["test_mvm_back[lat1]", "test_mvm_back[lat8]", "test_mvm_back[t8]", "test_mvm_back[mix]",
             "test_mvm_back[t1]", "test_mvm_back[k768]", "test_mvm_back[k256]"]
MAC_ITEMS = ["test_mvm_back_mac_lane"]
I2F_ITEMS = ["test_mvm_back_i2f_sweep"]

# (id, file, old, new, description, extra items run first, expected: "kill" or "equiv")
MUTANTS = [
    ("first_touch", BACK, "wire [31:0] opnd = lfirst ? 32'd0 :", "wire [31:0] opnd = 1'b0 ? 32'd0 :",
     "first term of a partial reads the slot instead of +0.0 (matters for ADD_LAT >= 7)", [], "kill"),
    ("bypass", BACK, "(lbyp ? acc_y[l] : pr_q[l][lset][lslot])", "(1'b0 ? acc_y[l] : pr_q[l][lset][lslot])",
     "no forwarding of the adder output (ADD_LAT 8 recurrence)", [], "kill"),
    ("no_clear", BACK, "if (!rst_n || tc_clr_q[pr_s][pr_k]) begin",
     "if (!rst_n) begin", "partial slots not cleared when level 1 reads them", [], "kill"),
    ("no_reset_pr", BACK, "if (!rst_n || tc_clr_q[pr_s][pr_k]) begin",
     "if (tc_clr_q[pr_s][pr_k]) begin", "partial slots not reset", [], "kill"),
    ("l1_slot", BACK, ".b(pr_q[1][tc_l1_q[1][3]][tc_l1_q[1][2:0]])", ".b(pr_q[1][tc_l1_q[1][3]][~tc_l1_q[1][2:0]])",
     "level 1 pairs p[2k] with the wrong odd partial", [], "kill"),
    ("l1_set", BACK, ".a(pr_q[0][tc_l1_q[0][3]][tc_l1_q[0][2:0]])", ".a(pr_q[0][~tc_l1_q[0][3]][tc_l1_q[0][2:0]])",
     "level 1 reads the even partial from the other set", [], "kill"),
    ("l2_operand", BACK, ".in_valid(i2v), .a(h1_q), .b(y1)", ".in_valid(i2v), .a(y1), .b(y1)",
     "level 2 adds the right operand twice", [], "kill"),
    ("hold_parity", BACK, "if (tc_hen_q[0]) h1_q <= y1;", "if (t1v && tg1_q) h1_q <= y1;",
     "level-1 hold register captures the odd instead of the even result", [], "kill"),
    ("hold_equiv", BACK, "if (tc_hen_q[0]) h1_q <= y1;", "if (t1v) h1_q <= y1;",
     "hold register also written at the odd result (sampled by level 2 at that edge: equivalent)", [],
     "equiv"),
    ("l4_parity", BACK, "wire  i4v = t3v & tg3_q;", "wire  i4v = t3v & ~tg3_q;",
     "level 4 issues on the wrong (first) level-3 result", [], "kill"),
    ("hs_bit", BACK, "wire hs_p = ~cfg_fmt & cd[C_M0 - 1].b4[1];", "wire hs_p = ~cfg_fmt & cd[C_M0 - 1].b4[2];",
     "Q8_0 weight placement by the wrong QS-line parity bit", [], "kill"),
    ("hs_align", BACK, "wire hs_p = ~cfg_fmt & cd[C_M0 - 1].b4[1];", "wire hs_p = ~cfg_fmt & cd[C_M0].b4[1];",
     "Q8_0 weight placement by the previous beat's QS-line parity (tag one cycle off)", [], "kill"),
    ("w_align", BACK, ".N(C_M0 - 1)) u_dly_w", ".N(C_M0)) u_dly_w",
     "weights one cycle late at the placement register / MACs", [], "kill"),
    ("q8_sum_sel", BACK, "is1_q[0] <= 23'(hs_g ? s2 : s0);", "is1_q[0] <= 23'(s0);",
     "Q8_0 block-b sum always from group 0", [], "kill"),
    ("sc_index", BACK, "is1_q[0] <= 23'($signed({1'b0, sc_g[0]}) * $signed(s0[16:0]));",
     "is1_q[0] <= 23'($signed({1'b0, sc_g[1]}) * $signed(s0[16:0]));", "Q4_K sc_0 replaced by sc_1", [], "kill"),
    ("sc_width", BACK, "is1_q[3] <= 23'($signed({1'b0, sc_g[3]}) * $signed(s3[16:0]));",
     "is1_q[3] <= 23'($signed({1'b0, sc_g[3][4:0]}) * $signed(s3[16:0]));", "Q4_K scale bit 5 dropped", [], "kill"),
    ("part0_cond", BACK, "if (cv[C_G + 2] && cfg_fmt && !cd[C_G + 2].half) part0_q <= is2_q[0];",
     "if (cv[C_G + 2] && cfg_fmt && cd[C_G + 2].half) part0_q <= is2_q[0];", "Q4_K part0 held from half 1", [],
     "kill"),
    ("isum_carry", BACK, "is3_q[0] <= cfg_fmt ? (26'(part0_q) + 26'(is2_q[0])) : 26'(is2_q[0]);",
     "is3_q[0] <= cfg_fmt ? 26'(25'(part0_q + is2_q[0])) : 26'(is2_q[0]);",
     "Q4_K isum computed in 25 bits (overflow above 2^24)", [], "kill"),
    ("bs_sel", BACK, "wire bsel = cd[C_M0].b4[0];", "wire bsel = 1'b0;", "BS half of the line ignores s % 2", [],
     "kill"),
    ("bs_pair", BACK, "+ 17'($signed(dbs_data[t][256*bsel + 32*ms_k + 16 +: 16]));",
     "+ 17'($signed(dbs_data[t][256*bsel + 32*ms_k +: 16]));", "BS pair uses BS[2j] twice", [], "kill"),
    ("bs_pair_w", BACK, "pair_q[ms_k] <= 17'($signed(dbs_data[t][256*bsel + 32*ms_k +: 16]))",
     "pair_q[ms_k] <= 17'(dbs_data[t][256*bsel + 32*ms_k +: 16])", "BS element zero-extended (unsigned)", [],
     "kill"),
    ("m_index", BACK, "mp_q[mp_k] <= 23'($signed({1'b0, m_m1[mp_k]}) * pair_q[mp_k]);",
     "mp_q[mp_k] <= 23'($signed({1'b0, m_m1[7 - mp_k]}) * pair_q[mp_k]);", "m_k reversed", [], "kill"),
    ("msum_align", BACK, ".N(C_CV - C_M0 - MS_LAT)) u_dly_ms", ".N(C_CV - C_M0 - MS_LAT + 1)) u_dly_ms",
     "msum one cycle late at the converter", [], "kill"),
    ("dx_odd", BACK, "wire [31:0] dx_odd  = dline[64*cd[C_M0].b4[3:1] + 32 +: 32];",
     "wire [31:0] dx_odd  = dline[64*cd[C_M0].b4[3:1] +: 32];", "Q8_0 Dx of block b+1 = Dx of b", [], "kill"),
    ("dxh_sel", BACK, "dxh_q <= cd[C_M0].b4[0] ? dx_odd : dx_even;", "dxh_q <= dx_even;",
     "Q4_K Dx of odd superblocks from the even element", [], "kill"),
    ("dxh_cond", BACK, "if (cv[C_M0] && cfg_fmt && !cd[C_M0].half) dxh_q", "if (cv[C_M0] && cfg_fmt && cd[C_M0].half) dxh_q",
     "Q4_K Dx hold captured from the half-1 beat (a BS element)", [], "kill"),
    ("dxh_equiv", BACK, "if (cv[C_M0] && cfg_fmt && !cd[C_M0].half) dxh_q", "if (cv[C_M0] && cfg_fmt) dxh_q",
     "Dx hold also written by the half-1 beat (its multiplier samples the old value: equivalent)", [],
     "equiv"),
    ("d_b", BACK, ".a(db_m0), .b(cfg_fmt ? dxh_q : dx_odd)", ".a(da_m0), .b(cfg_fmt ? dxh_q : dx_odd)",
     "second scale (Q8_0 d of b+1 / Q4_K dmin) replaced by d_a", [], "kill"),
    ("fsub", BACK, ".a(pb0), .b(pb1), .sub(1'b1)", ".a(pb0), .b(pb1), .sub(1'b0)",
     "Q4_K term t1 + t2 instead of t1 - t2", [], "kill"),
    ("f_align", BACK, ".N(C_F - C_FI)) u_dly_f", ".N(C_F - C_FI + 1)) u_dly_f",
     "f32 sums one cycle late at the second products", [], "kill"),
    ("bs_addr", BACK, "? DBS_AW'(11'(cfg_d_lines) + 11'(beat.blk[TSR_MVM_BLK_W-1:1]))",
     "? DBS_AW'(11'(beat.blk[TSR_MVM_BLK_W-1:1]))", "BS line without the D-line offset", [], "kill"),
    ("qs_addr", BACK, "QS_AW'({beat.blk, beat.half})", "QS_AW'({beat.blk, 1'b0})",
     "Q4_K QS word ignores the half", [], "kill"),
    ("q4_lane", BACK, "      islot  = cd[C_A4].b4[3:1];", "      islot  = cd[C_A4].b4[3:1] + 3'd1;",
     "Q4_K accumulate writes slot s/2 + 1 (the read slot comes from the per-token tag copies)", [], "kill"),
    ("q8_two", BACK, "      iv[1]  = cv[C_A8] & cd[C_A8].two;", "      iv[1]  = cv[C_A8];",
     "Q8_0 adds the absent block b+1 of an odd partial chunk", [], "kill"),
    ("q4_half", BACK, "      iv[0]  = cv[C_A4] & cd[C_A4].half & ~cd[C_A4].b4[0];",
     "      iv[0]  = cv[C_A4] & ~cd[C_A4].b4[0];", "Q4_K half-0 beats issue a term on lane 0", [], "kill"),
    ("first_def", BACK, "first: (beat.blk[TSR_MVM_BLK_W-1:4] == '0)", "first: (beat.blk[TSR_MVM_BLK_W-1:5] == '0)",
     "terms 16..31 treated as first terms", [], "kill"),
    ("set_toggle", BACK, "rset_q     <= rset_q ^ acc_last;", "rset_q     <= rset_q;",
     "every row uses partial set 0", [], "kill"),
    ("l1_free", BACK, "wire        l1_free = ~l1_act_q | (l1_k_q == 3'd7);", "wire        l1_free = 1'b1;",
     "level 1 restarts while reading a set", [], "kill"),
    ("row_gap", BACK, "localparam int unsigned ROW_GAP   = 8;", "localparam int unsigned ROW_GAP   = 7;",
     "row starts 7 cycles apart allowed", [], "kill"),
    ("res_depth", BACK, "localparam int unsigned RES_DEPTH = (HOLD + 7) / 8 + 1;",
     "localparam int unsigned RES_DEPTH = (HOLD + 7) / 8;", "result FIFO one entry short (RB_MIN drop)", [],
     "kill"),
    ("no_credit", BACK, "ready_q    <= (resv_n < (RES_AW+1)'(RES_DEPTH)) &", "ready_q    <= 1'b1 &",
     "no result-space reservation", [], "kill"),
    ("credit_obo", BACK, "ready_q    <= (resv_n < (RES_AW+1)'(RES_DEPTH)) &",
     "ready_q    <= (resv_n <= (RES_AW+1)'(RES_DEPTH)) &", "one reservation too many (FIFO overflow by one)",
     [], "kill"),
    ("rf_wrap", BACK, "return (32'(p) == RES_DEPTH - 1) ? '0 : p + 1'b1;", "return p + 1'b1;",
     "result FIFO pointers do not wrap at a non-power-of-2 depth", [], "kill"),
    ("res_valid", BACK, "res_valid_q <= (rf_cnt_n != '0);", "res_valid_q <= (rf_cnt_q != '0);",
     "res_valid from the old count", [], "kill"),
    ("idle", BACK, "idle_q     <= at_start_n & (resv_n == '0);", "idle_q     <= (resv_n == '0);",
     "idle ignores a row in progress", [], "kill"),
    ("q4_mask", BACK, "{wlo_p[wp_l][7:4] & {4{~cfg_fmt}}, wlo_p[wp_l][3:0]}", "{wlo_p[wp_l][7:4], wlo_p[wp_l][3:0]}",
     "Q4_K upper nibbles not masked (front end guarantees 0: expected equivalent)", [], "equiv"),
    ("i2f_rne", BACK, "wire        inc = g & (st | m23[0]);", "wire        inc = g & st;",
     "int->f32: round half down instead of to even", I2F_ITEMS, "kill"),
    ("i2f_exp", BACK, "wire [7:0]  ex  = 8'(126 + IW) - 8'(lz2) + {7'd0, fr[23]};",
     "wire [7:0]  ex  = 8'(126 + IW) - 8'(lz2);", "int->f32: rounding carry not added to the exponent",
     I2F_ITEMS, "kill"),
    ("i2f_neg", BACK, "mag1 <= v[IW-1] ? (~v + IW'(1)) : v;", "mag1 <= v[IW-1] ? (~v) : v;",
     "int->f32: magnitude of negatives off by one", I2F_ITEMS, "kill"),
    ("act_bank", ACT, "& ~wr_line.dbs & ~wr_line.addr[0];", "& ~wr_line.dbs & 1'b1;",
     "actbuf: odd QS lines also written to bank 0", [], "kill"),
    ("act_tok", ACT, "dbs_we_q[we_t]   <= wr_valid & (32'(wr_line.tok) == 32'(we_t)) &  wr_line.dbs;",
     "dbs_we_q[we_t]   <= wr_valid &  wr_line.dbs;", "actbuf: D/BS lines written into every token", [], "kill"),
    ("act_waddr", ACT, "qs_wa_q  <= QS_AW'(wr_line.addr[9:1]);", "qs_wa_q  <= QS_AW'(wr_line.addr[9:0]);",
     "actbuf: QS word address = line address", [], "kill"),
    ("mac_sext_w", MAC, "assign d[t][k] = $signed({{8{w[k][7]}}, w[k]})",
     "assign d[t][k] = $signed({8'd0, w[k]})", "MAC: weights zero-extended (-128 .. -1 wrong)", MAC_ITEMS,
     "kill"),
    ("mac_sext_t", MAC, "assign d[t][k] = $signed({g_l[i-1].q[t][2*k][WI-2], g_l[i-1].q[t][2*k]})",
     "assign d[t][k] = $signed({1'b0, g_l[i-1].q[t][2*k]})", "MAC: tree operand not sign-extended", MAC_ITEMS,
     "kill"),
    ("mac_ce", MAC, "        else if (ce) vq <= vd;", "        else vq <= vd;",
     "MAC: cut-register valid ignores ce (only visible with ce stalls)", MAC_ITEMS, "kill"),
    ("mac_ce_data", MAC, "        if (ce) q <= d;", "        q <= d;",
     "MAC: cut-register data ignores ce (only visible with ce stalls)", MAC_ITEMS, "kill"),
    ("mac_cut", MAC, "if (i < LV && cut_after(i)) begin : g_cut", "if (i < LV && cut_after(i) && i > 0) begin : g_cut",
     "MAC: the product register dropped (latency LAT - 1 for LAT >= 2)", MAC_ITEMS, "kill"),
    ("mac_prod_late", MAC, "if (i == 0) return NCUT > 0;", "if (i == 1) return NCUT > 0;",
     "MAC: product register moved behind adder level 1 (LAT 2 / 3: latency and sums unchanged; "
     "only mac_lane_structure sees it)", MAC_ITEMS, "kill"),
    ("mac_spread", MAC, "if (cut_c * int'(LV) / int'(NCUT) == i) return 1'b1;",
     "if (cut_c * int'(LV) / int'(NCUT) + 1 == i) return 1'b1;",
     "MAC: adder cuts one level late (LAT 3: products | 3 | 2 levels; latency and sums unchanged up to LAT 5)",
     MAC_ITEMS, "kill"),
]


def apply(src_root: Path, dst_root: Path, rel: str, old: str, new: str):
    shutil.copytree(src_root / "rtl", dst_root / "rtl")
    # rtl/ctrl/filelist.f names the vendored IP as ../../third_party (W4-A)
    (dst_root / "third_party").symlink_to(src_root / "third_party")
    p = dst_root / "rtl" / rel
    s = p.read_text()
    n = s.count(old)
    if n != 1:
        raise SystemExit("mutant pattern matches %d times in %s: %r" % (n, rel, old))
    p.write_text(s.replace(old, new))


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    if sys.argv[1] == "--check":           # every pattern matches exactly once
        for mid, rel, old, new, *_ in MUTANTS:
            n = (REPO / "rtl" / rel).read_text().count(old)
            print("%-12s %s" % (mid, "ok" if n == 1 else "MATCHES %d" % n))
        return
    scr = Path(sys.argv[1]).resolve()
    only = set(sys.argv[2:])
    scr.mkdir(parents=True, exist_ok=True)
    resf = scr / "results.json"
    results = json.loads(resf.read_text()) if resf.exists() else {}
    for mid, rel, old, new, desc, extra, expect in MUTANTS:
        if only and mid not in only:
            continue
        d = scr / mid
        if d.exists():
            shutil.rmtree(d)
        d.mkdir()
        apply(REPO, d, rel, old, new)
        env = dict(os.environ, TSR_MVMBACK_RTL=str(d / "rtl" / "filelist.f"), TSR_SIM_BUILD_ROOT=str(d / "build"))
        killed_by, t0 = None, time.time()
        for item in extra + CFG_ORDER:
            p = subprocess.run([sys.executable, "-m", "pytest", "-x", "-q", "mvmback/test_mvm_back.py::" + item],
                               cwd=REPO / "tb", env=env, capture_output=True, text=True, timeout=3600)
            if p.returncode != 0:
                tail = [ln for ln in (p.stdout + p.stderr).splitlines()
                        if "Error" in ln or "assert" in ln.lower() or "FAIL" in ln][:4]
                killed_by = (item, tail)
                break
        dt = time.time() - t0
        status = "KILLED" if killed_by else "SURVIVED"
        results[mid] = dict(desc=desc, file=rel, status=status, expect=expect, seconds=round(dt),
                            killed_by=killed_by[0] if killed_by else None,
                            why=killed_by[1] if killed_by else None)
        resf.write_text(json.dumps(results, indent=1))
        print("%-12s %-8s %-28s %4ds  %s" % (mid, status, killed_by[0] if killed_by else "-", dt, desc), flush=True)
        shutil.rmtree(d / "build", ignore_errors=True)
    k = sum(1 for r in results.values() if r["status"] == "KILLED")
    print("killed %d / %d" % (k, len(results)))


if __name__ == "__main__":
    main()
