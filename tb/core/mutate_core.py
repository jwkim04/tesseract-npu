#!/usr/bin/env python3
"""tb/core/mutate_core.py — mutation campaign of the integrated core (HAS_MVM = 1): every mutant is
applied to a scratch copy of rtl/ (never to the repository) and the core-level tests must fail.

    python3 tb/core/mutate_core.py WORK_DIR [--only NAME[,NAME]] [--noassert] [--stage2-all] [--list]

Mutants: wiring of tsr_core's MVM hookup (W*) and unit mutants seen through the core (M*: tsr_mvm,
tsr_mvm_cluster, tsr_mvm_gather, tsr_mvm_ws, tsr_mvm_front, tsr_mvm_back, tsr_cmdproc).
Kill stages (the first failing stage kills; later stages are skipped):
  1  tb/fastreplay (C++ replay, --wno-fatal) of traces/m3/ch3 + traces/m3/tmax1 +
     traces/gen/tmax1 under the rand, fast and slow regimes (every check of
     fastreplay.cpp: results, written / read sets, PERF, per-source beats, quiescence, RTL
     assertions incl. tsr_core I1..I4)
  2  tb/core test_core_m3_directed[c2] (cocotb: m3_perf exact PERF_MVM_CYCLES / SPM / WS counts,
     m3_mvm_st dependencies, m3_random lockstep incl. the whole SPM, m3_abort / resets ...)
--noassert: the scratch copy starts tsr_pkg.sv with `define SYNTHESIS, which removes every
simulation-only check of the RTL (block assertions and tsr_core I1..I4): shows which mutants the
testbenches alone kill. Results: WORK_DIR/results.json (name, stage, killed, first error).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

# (name, [(file under rtl/, old (exactly once), new), ...], what it breaks, expected outcome)
MUTANTS = [
    # ---- tsr_core wiring of the MVM unit ----
    ("W_ws_req_ch_rot", [("core/tsr_core.sv",
                          "    assign s_rd_req_valid[c][TSR_SRC_WS]  = w_rd_req_valid[c];\n"
                          "    assign s_rd_req_addr[c][TSR_SRC_WS]   = w_rd_req_addr[c];\n"
                          "    assign s_rd_req_len[c][TSR_SRC_WS]    = w_rd_req_len[c];\n"
                          "    assign s_rd_req_id[c][TSR_SRC_WS]     = w_rd_req_id[c];\n"
                          "    assign s_rd_rsp_ready[c][TSR_SRC_WS]  = w_rd_rsp_ready[c];\n"
                          "    assign w_rd_req_ready[c]              = s_rd_req_ready[c][TSR_SRC_WS];\n"
                          "    assign w_rd_rsp_valid[c]              = s_rd_rsp_valid[c][TSR_SRC_WS];\n",
                          "    localparam int CR = (c + 1) % NUM_CH;\n"
                          "    assign s_rd_req_valid[c][TSR_SRC_WS]  = w_rd_req_valid[CR];\n"
                          "    assign s_rd_req_addr[c][TSR_SRC_WS]   = w_rd_req_addr[CR];\n"
                          "    assign s_rd_req_len[c][TSR_SRC_WS]    = w_rd_req_len[CR];\n"
                          "    assign s_rd_req_id[c][TSR_SRC_WS]     = w_rd_req_id[CR];\n"
                          "    assign s_rd_rsp_ready[c][TSR_SRC_WS]  = w_rd_rsp_ready[c];\n"
                          "    assign w_rd_req_ready[CR]             = s_rd_req_ready[c][TSR_SRC_WS];\n"
                          "    assign w_rd_rsp_valid[c]              = s_rd_rsp_valid[c][TSR_SRC_WS];\n")],
     "cluster c's weight requests go to channel c+1's arbiter (responses still to cluster c)", "killed"),
    ("W_ws_rsp_data_rev", [("core/tsr_core.sv", ".ws_rd_rsp_data(s_rd_rsp_data)",
                            ".ws_rd_rsp_data({<<BEAT_BITS{s_rd_rsp_data}})")],
     "weight response data of the channels in reverse order", "killed"),
    ("W_ws_len_msb", [("core/tsr_core.sv", "    assign s_rd_req_len[c][TSR_SRC_WS]    = w_rd_req_len[c];",
                       "    assign s_rd_req_len[c][TSR_SRC_WS]    = {1'b0, w_rd_req_len[c][5:0]};")],
     "WS burst length loses its MSB (64 -> 0)", "killed"),
    ("W_perf_mvm_tied0", [("core/tsr_core.sv", ".perf_retire, .perf_mvm_busy, .perf_ch_rd_beat",
                           ".perf_retire, .perf_mvm_busy(1'b0), .perf_ch_rd_beat")],
     "PERF_MVM_CYCLES input not connected", "killed"),
    ("W_spm_rsp_client", [("core/tsr_core.sv",
                           ".spm_rd_rsp_valid(spm_rd_rsp_valid[TSR_SPM_CL_MVM]),\n      .spm_rd_rsp_data(spm_rd_rsp_data),\n      .spm_wr_valid(spm_wr_valid[TSR_SPM_CL_MVM])",
                           ".spm_rd_rsp_valid(spm_rd_rsp_valid[TSR_SPM_CL_QACT]),\n      .spm_rd_rsp_data(spm_rd_rsp_data),\n      .spm_wr_valid(spm_wr_valid[TSR_SPM_CL_MVM])")],
     "MVM sees the QACT client's SPM read responses", "killed"),
    ("W_mvm_spm_rd_lat", [("core/tsr_core.sv",
                           ".SPM_BYTES(SPM_BYTES), .SPM_RD_LAT(SPM_RD_LAT),\n      .ADDR_BEAT_W(ADDR_BEAT_W), .BEAT_BITS(BEAT_BITS), .WS_FIFO_BEATS(MVM_WS_FIFO)",
                           ".SPM_BYTES(SPM_BYTES), .SPM_RD_LAT(SPM_RD_LAT + 1),\n      .ADDR_BEAT_W(ADDR_BEAT_W), .BEAT_BITS(BEAT_BITS), .WS_FIFO_BEATS(MVM_WS_FIFO)")],
     "MVM unit told a wrong SPM read latency", "killed"),
    ("W_mvm_done_late_ready", [("core/tsr_core.sv", ".mvm_cmd_valid, .mvm_cmd_ready, .mvm_cmd, .mvm_done",
                                ".mvm_cmd_valid, .mvm_cmd_ready(mvm_cmd_ready & ~mvm_done), .mvm_cmd, .mvm_done")],
     "cmdproc sees cmd_ready masked in the done cycle (timing only)", "equivalent"),
    ("W_ws_rsp_ready_1", [("core/tsr_core.sv", "    assign s_rd_rsp_ready[c][TSR_SRC_WS]  = w_rd_rsp_ready[c];",
                           "    assign s_rd_rsp_ready[c][TSR_SRC_WS]  = 1'b1;")],
     "WS rd_rsp_ready tied 1 (the streamer's is constant 1 by contract)", "equivalent"),
    ("W_bc_pipe_param", [("core/tsr_core.sv", ".BC_PIPE(MVM_BC_PIPE), .RES_PIPE(MVM_RES_PIPE)",
                          ".BC_PIPE(MVM_BC_PIPE + 1), .RES_PIPE(MVM_RES_PIPE)")],
     "broadcast pipe one stage deeper than the parameter (results identical by contract)", "equivalent"),
    # ---- tsr_mvm (dispatch, broadcast) ----
    ("M_bc_stride", [("core/tsr_mvm.sv", "    else if (rq_fire) rq_line_q <= rq_line_q + 1'b1;",
                      "    else if (rq_fire) rq_line_q <= rq_line_q + (rq_l_end ? 2'd2 : 2'd1);")],
     "broadcast skips a line at every token boundary (QA image stride + 1)", "killed"),
    ("M_bc_last_tok0", [("core/tsr_mvm.sv", "bc_d_q <= {rs_l_end & rs_t_end, rs_tok_q",
                         "bc_d_q <= {rs_l_end, rs_tok_q")],
     "bcline.last after token 0 (activations released before the other tokens arrived)", "killed"),
    ("M_expect_ch_le", [("core/tsr_mvm.sv", "ex_q[ex_c_q] <= (ex_acc_q < 36'(c_q.n));",
                         "ex_q[ex_c_q] <= (ex_acc_q <= 36'(c_q.n));")],
     "gather expects a cluster with c*R == N (no rows): hang", "killed"),
    ("M_busy_from_accept", [("core/tsr_mvm.sv", "  assign perf_mvm_busy = busy_q;",
                             "  assign perf_mvm_busy = busy_q | fire;")],
     "perf_mvm_busy also in the accepting cycle (PERF_MVM_CYCLES + 1 per MVM)", "killed"),
    # ---- tsr_mvm_cluster ----
    ("M_rows_c_plus1", [("core/tsr_mvm_cluster.sv",
                         "rows_q  <= !gt_q ? 32'd0 : ((left_q < 36'(r_q)) ? left_q[31:0] : r_q);",
                         "rows_q  <= !gt_q ? 32'd0 : ((left_q < 36'(r_q)) ? left_q[31:0] : r_q + 32'd1);")],
     "rows_c one too many for full channels (over-fetch, extra row)", "killed"),
    ("M_rows_c_minus1", [("core/tsr_mvm_cluster.sv",
                          "rows_q  <= !gt_q ? 32'd0 : ((left_q < 36'(r_q)) ? left_q[31:0] : r_q);",
                          "rows_q  <= !gt_q ? 32'd0 : ((left_q < 36'(r_q)) ? left_q[31:0] - 32'd1 : r_q);")],
     "rows_c one short for the partial (last) channel", "killed"),
    ("M_act_gate_off", [("core/tsr_mvm_cluster.sv", "  assign f_beat_ready = b_beat_ready & act_ready_q;",
                         "  assign f_beat_ready = b_beat_ready;"),
                        ("core/tsr_mvm_cluster.sv", ".beat_valid(f_beat_valid & act_ready_q)",
                         ".beat_valid(f_beat_valid)")],
     "weight beats reach the back end before the activations are in the buffer", "killed"),
    # ---- tsr_mvm_ws / front ----
    ("M_ws_overfetch", [("core/tsr_mvm_ws.sv", "        rq_left_q <= cmd_beats;",
                         "        rq_left_q <= cmd_beats + 1'b1;")],
     "weight streamer requests one beat more than rows_c x ROW_BEATS", "killed"),
    ("M_q8_two_always", [("core/tsr_mvm_front.sv", "    p_in.two      = ~cfg_fmt_q & ~(row_end & cfg_rem_odd_q);",
                          "    p_in.two      = ~cfg_fmt_q;")],
     "Q8_0 odd last block count: the absent block b+1 is used (DRAM garbage)", "killed"),
    # ---- tsr_mvm_back ----
    ("M_tree_l1_order", [("core/tsr_mvm_back.sv",
                          ".a(pr_q[0][l1_set_q][l1_k_q]), .b(pr_q[1][l1_set_q][l1_k_q]), .sub(1'b0),",
                          ".a(pr_q[0][l1_set_q][{l1_k_q[0], l1_k_q[2:1]}]), .b(pr_q[1][l1_set_q][{l1_k_q[0], l1_k_q[2:1]}]), .sub(1'b0),")],
     "tree level 1 reads the pairs in another order (tsr_tree16 grouping broken)", "killed"),
    ("M_slot_perm_q8", [("core/tsr_mvm_back.sv", "      islot  = cd[C_A8].b4[3:1];",
                         "      islot  = {cd[C_A8].b4[1], cd[C_A8].b4[3:2]};")],
     "Q8_0 terms accumulate into permuted partials (p[i % 16] order broken)", "killed"),
    ("M_lane_swap_q8", [("core/tsr_mvm_back.sv", "      wire [31:0] term = cfg_fmt ? qd : ((l == 0) ? pb0 : pb1);",
                         "      wire [31:0] term = cfg_fmt ? qd : ((l == 1) ? pb0 : pb1);")],
     "Q8_0 even / odd block terms swap partial lanes (p[2k] <-> p[2k+1]; the level-1 add commutes)",
     "equivalent"),
    ("M_msum_23b", [("core/tsr_mvm_back.sv",
                     "      msum_q <= (26'(m4_q[0]) + 26'(m4_q[1])) + (26'(m4_q[2]) + 26'(m4_q[3]));",
                     "      msum_q <= 26'($signed(23'((26'(m4_q[0]) + 26'(m4_q[1])) + (26'(m4_q[2]) + 26'(m4_q[3])))));")],
     "Q4_K msum wraps at 23 bits (enough only for QACT-made BS)", "killed"),
    ("M_q8_half_sel", [("core/tsr_mvm_back.sv", "    wire hs_g = ~cfg_fmt & cd[C_G].b4[1];",
                        "    wire hs_g = ~cfg_fmt & cd[C_G].b4[2];")],
     "Q8_0 odd-QS-line half select uses the wrong block bit", "killed"),
    # ---- tsr_mvm_gather ----
    ("M_acc_tail_sum", [("core/tsr_mvm_gather.sv",
                         "push_d = {ap_line, lane_be(ap_own | ap_tail), lane_mask(sum_y, ap_own)};",
                         "push_d = {ap_line, lane_be(ap_own | ap_tail), lane_mask(sum_y, ap_own | ap_tail)};")],
     "ACC: tail lanes written with old + 0 instead of 0", "killed"),
    ("M_acc_tail_kept", [("core/tsr_mvm_gather.sv",
                          "push_d = {ap_line, lane_be(ap_own | ap_tail), lane_mask(sum_y, ap_own)};",
                          "push_d = {ap_line, lane_be(ap_own), lane_mask(sum_y, ap_own)};")],
     "ACC: tail bytes not written (old tail kept instead of 0)", "killed"),
    ("M_be_no_tail", [("core/tsr_mvm_gather.sv", "push_d = {j_line, lane_be(j_own | j_tail), lane_mask(j_y, j_own)};",
                       "push_d = {j_line, lane_be(j_own), lane_mask(j_y, j_own)};")],
     "plain MVM: tail bytes not written", "killed"),
    ("M_be_extra_lane", [("core/tsr_mvm_gather.sv",
                          "push_d = {j_line, lane_be(j_own | j_tail), lane_mask(j_y, j_own)};",
                          "push_d = {j_line, lane_be(j_own | j_tail | (j_own >> 1)), lane_mask(j_y, j_own)};")],
     "byte enables cover the lane below a job's lanes (clobbers another cluster's / old lane)", "killed"),
    ("M_done_early", [("core/tsr_mvm_gather.sv", "  wire all_empty = ~|full_q & ~|fillv_q & (cnt_q == '0);",
                       "  wire all_empty = ~|full_q & ~|fillv_q & (cnt_q <= CNT_W'(1));")],
     "gather done while the last SPM write is still pending", "killed"),
    ("M_acc_old_lane", [("core/tsr_mvm_gather.sv", ".in_valid(rp_v), .a(old_q[l]), .b(rp_y[l]), .sub(1'b0),",
                         ".in_valid(rp_v), .a(old_q[l ^ 1]), .b(rp_y[l]), .sub(1'b0),")],
     "ACC adds the neighbouring lane's old value", "killed"),
    ("M_acc_rd_align", [("core/tsr_mvm_gather.sv",
                         "tsr_vpipe #(.VW(1), .W(IS_W), .DEPTH(SPM_RD_LAT + 1)) u_rdp (",
                         "tsr_vpipe #(.VW(1), .W(IS_W), .DEPTH(SPM_RD_LAT + 2)) u_rdp (")],
     "ACC job paired with the old line one cycle late", "killed"),
    # ---- tsr_cmdproc (MVM dispatch fields) ----
    ("M_cmdproc_R_plus1", [("core/tsr_cmdproc.sv", "      mvm_cmd_q.rows_per_ch <= mx_rows;",
                            "      mvm_cmd_q.rows_per_ch <= mx_rows + 32'd1;")],
     "dispatched rows_per_ch = ceil(N / NUM_CH) + 1 (rows land on the wrong channels)", "killed"),
]

STAGE1_TRACES = ["traces/m3/ch3", "traces/m3/tmax1", "traces/gen/tmax1"]   # two builds: 3 ch T_MAX 3, U200


def sh(cmd, log: Path, env=None, timeout=None) -> int:
    with open(log, "a") as f:
        f.write("$ %s\n" % " ".join(str(c) for c in cmd))
        f.flush()
        try:
            r = subprocess.run([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT, env=env, timeout=timeout,
                               cwd=REPO)
            return r.returncode
        except subprocess.TimeoutExpired:
            f.write("TIMEOUT after %s s\n" % timeout)
            return 124


def make_copy(work: Path, name: str, edits, noassert: bool) -> Path:
    d = work / name
    if d.exists():
        shutil.rmtree(d)
    shutil.copytree(REPO / "rtl", d / "rtl", ignore=shutil.ignore_patterns("platform"))
    for f, old, new in edits:
        p = d / "rtl" / f
        s = p.read_text()
        n = s.count(old)
        if n != 1:
            raise SystemExit("%s: '%s...' occurs %d times in %s" % (name, old[:60], n, f))
        p.write_text(s.replace(old, new))
    if noassert:
        p = d / "rtl" / "core" / "tsr_pkg.sv"
        p.write_text("`define SYNTHESIS\n" + p.read_text())
    return d / "rtl" / "filelist.f"


def stage1(name, fl: Path, d: Path) -> tuple:
    out = d / "fr.json"
    env = dict(os.environ, TSR_FASTREPLAY_BUILD=str(d / "fr"))
    rc = sh([sys.executable, REPO / "tb/fastreplay/run.py"] + STAGE1_TRACES +
            ["--regimes", "rand,fast,slow", "--filelist", fl, "--wno-fatal", "--procs", "3", "-q",
             "--timeout-mult", "2", "--stop-on-fail", "--json", out], d / "stage1.log", env=env, timeout=3600)
    if not out.is_file():
        tail = "\n".join((d / "stage1.log").read_text(errors="replace").splitlines()[-6:])
        return (rc != 0, "fastreplay rc %d (no report; build failure?): %s" % (rc, tail))
    rep = json.loads(out.read_text())
    if rep["failed"]:
        fails = [f for g in rep["groups"] for f in g["failures"]]
        first = fails[0]
        kinds = {("assertion" if all("RTL $stop/$error" in e for e in f["errors"]) else "testbench") for f in fails}
        return (True, "%d failing runs (%s); first: %s [%s] %s" % (
            rep["failed"], "+".join(sorted(kinds)), Path(first["trace"]).name, first["regime"],
            " | ".join(first["errors"][:2])[:400]))
    return (False, "%d runs passed" % rep["runs"])


def stage2(name, fl: Path, d: Path) -> tuple:
    env = dict(os.environ, TSR_CORE_FILELIST=str(fl), TSR_SIM_BUILD_ROOT=str(d / "cocotb"),
               TSR_CORE_TESTS="m3_perf,m3_mvm_st,m3_random,m3_abort", TSR_CORE_M3_INTERRUPTS="10")
    log = d / "stage2.log"
    rc = sh([sys.executable, "-m", "pytest", "tb/core/test_core_m3.py::test_core_m3_directed[c2]", "-x", "-q", "-s",
             "-p", "no:cacheprovider"], log, env=env, timeout=3600)
    txt = log.read_text(errors="replace")
    if rc == 0:
        return (False, "c2 directed passed")
    lines = [l.strip() for l in txt.splitlines() if "Error" in l or "AssertionError" in l or "FAIL " in l]
    return (True, "c2 directed failed: %s" % (" | ".join(lines[:3])[:400]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("work")
    ap.add_argument("--only", default="")
    ap.add_argument("--noassert", action="store_true")
    ap.add_argument("--stage2-all", action="store_true", help="run stage 2 even when stage 1 killed")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for m in MUTANTS:
            print("%-24s %-11s %s" % (m[0], m[3], m[2]))
        return 0
    work = Path(a.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    only = set(x for x in a.only.split(",") if x)
    res_file = work / ("results_noassert.json" if a.noassert else "results.json")
    results = json.loads(res_file.read_text()) if res_file.is_file() else {}
    os.chdir(REPO / "tb")
    for name, edits, what, expect in MUTANTS:
        if only and name not in only:
            continue
        t0 = time.time()
        fl = make_copy(work, name, edits, a.noassert)
        d = work / name
        killed, msg = stage1(name, fl, d)
        stage = 1 if killed else None
        msg2 = ""
        if not killed or a.stage2_all:
            k2, msg2 = stage2(name, fl, d)
            if k2 and not killed:
                killed, stage = True, 2
        r = dict(name=name, what=what, expected=expect, killed=killed, stage=stage, stage1=msg, stage2=msg2,
                 seconds=round(time.time() - t0))
        results[name] = r
        res_file.write_text(json.dumps(results, indent=1))
        print("%-24s %-8s stage %s (expected %s) %4d s | %s %s" % (
            name, "KILLED" if killed else "SURVIVED", stage, expect, r["seconds"], msg[:200],
            ("| " + msg2[:200]) if msg2 else ""), flush=True)
        for sub in ("rtl", "fr", "cocotb"):         # builds are large; logs and reports stay
            shutil.rmtree(d / sub, ignore_errors=True)
    k = sum(1 for r in results.values() if r["killed"])
    print("%d of %d mutants killed" % (k, len(results)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
