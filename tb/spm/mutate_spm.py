"""Mutation spot-checks of the SPM work of W4-A (tsr_spm, tsr_spm_cport, tsr_spm_bridge, tsr_dma's LD
reservation; CC §15.5): each mutant is a copy of one RTL file with one edit, written under
build/cocotb/spm_mut/<id>/ (the tree is never edited) and swapped in through $TSR_MUT
(tb/spm/spm_sim.py mut_sources, which tb/spm and tb/dma pass their source lists through); a mutant
is killed when one of its pytest nodes fails. The two *_na mutants also remove the RTL assertions,
so the bench monitors alone must catch them.

    python tb/spm/mutate_spm.py            every mutant (22; 2.5 min with TSR_BUILD_JOBS=2, 2026-10-31)
    python tb/spm/mutate_spm.py ID...      only these
    python tb/spm/mutate_spm.py --check    every pattern matches exactly once (no simulation)

Moved into the tree by the W4-A integration from the SPM builder's runner; 22 of 22 killed (2026-10-31).
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROOT = str(REPO / "rtl" / "core") + "/"
OUT = REPO / "build" / "cocotb" / "spm_mut"
TB = REPO / "tb"

M = {
  # tsr_spm
  "spm_fixed_prio_rd1": ("tsr_spm.sv",
      "        if (|rd_req_b[b]) rd_gnt_q[b] <= rr_next(rd_req_b[b], rd_gnt_q[b]);",
      "        if (|rd_req_b[b]) rd_gnt_q[b] <= (b == 1) ? lowest_set(rd_req_b[b]) : rr_next(rd_req_b[b], rd_gnt_q[b]);"),
  "spm_no_keep_grant": ("tsr_spm.sv",
      "        if (|wr_req_b[b]) wr_gnt_q[b] <= rr_next(wr_req_b[b], wr_gnt_q[b]);",
      "        wr_gnt_q[b] <= (|wr_req_b[b]) ? rr_next(wr_req_b[b], wr_gnt_q[b]) : '0;"),
  "spm_wrong_depth_bank": ("tsr_spm.sv",
      "    wire wr_hit = wr_fire[B] && (wr_dsel[B] == DSEL_W'(D));",
      "    wire wr_hit = wr_fire[B] && (wr_dsel[B] == DSEL_W'(D)) || (wr_fire[1-B] && D == 0 && B == 1 && wr_dsel[1-B] == '0);"),
  "spm_rsp_early_o": ("tsr_spm.sv",
      "      for (int unsigned b = 0; b < 2; b++) rd_rsp_valid[c][b] = rsp_pipe[RD_LAT-1][b][c];",
      "      for (int unsigned b = 0; b < 2; b++) rd_rsp_valid[c][b] = (b == 1 && RD_LAT > 1) ? rsp_pipe[RD_LAT-2][b][c] : rsp_pipe[RD_LAT-1][b][c];"),
  "spm_be_slice": ("tsr_spm.sv",
      "          be_q <= wr_be_sel[B][k*COL_BE +: COL_BE];",
      "          be_q <= wr_be_sel[B][((k+1)%NCOL)*COL_BE +: COL_BE];"),
  "spm_owner_first": ("tsr_spm.sv",
      "    return (above != '0) ? lowest_set(above) : lowest_set(req);",
      "    return ((req & gnt) != '0) ? gnt : (above != '0) ? lowest_set(above) : lowest_set(req);"),
  # tsr_spm_cport
  "cport_rsp_sel": ("tsr_spm_cport.sv",
      "  assign rd_rsp_data    = s_rd_rsp_valid[1] ? s_rd_rsp_data[1] : s_rd_rsp_data[0];",
      "  assign rd_rsp_data    = s_rd_rsp_valid[0] ? s_rd_rsp_data[1] : s_rd_rsp_data[0];"),
  "cport_ready_other": ("tsr_spm_cport.sv",
      "  assign wr_ready       = s_wr_ready[wr_b];",
      "  assign wr_ready       = s_wr_ready[wr_b] & s_wr_ready[~wr_b] | s_wr_ready[wr_b] & ~wr_valid;"),
  "cport_steer": ("tsr_spm_cport.sv",
      "  assign s_rd_req_valid = {rd_req_valid & rd_b, rd_req_valid & ~rd_b};",
      "  assign s_rd_req_valid = {rd_req_valid & rd_b, rd_req_valid & ~rd_b & ~(rd_req_line[1] & rd_req_line[2])};"),
  # tsr_spm_bridge
  "br_store_early": ("tsr_spm_bridge.sv",
      "          end else if (req_we) begin\n            wr_v_q <= 1'b1;\n            st_q   <= S_WR;",
      "          end else if (req_we) begin\n            wr_v_q <= 1'b1;\n            rsp_v_q <= 1'b1;\n            st_q   <= S_WR;"),
  "br_be_shift": ("tsr_spm_bridge.sv",
      "      req_be_line[4*w +: 4] = (req_waddr[3:0] == 4'(w)) ? req_be : 4'b0;",
      "      req_be_line[4*w +: 4] = (req_waddr[3:0] == 4'(w)) ? {req_be[2:0], req_be[3]} : 4'b0;"),
  "br_no_capture": ("tsr_spm_bridge.sv",
      "    if (st_q == S_RWAIT && spm_rd_rsp_valid) rline_q <= spm_rd_rsp_data;",
      "    if (st_q == S_RWAIT && spm_rd_rsp_valid && line_q[0]) rline_q <= spm_rd_rsp_data;"),
  "br_no_range": ("tsr_spm_bridge.sv",
      "  wire          req_bad = 32'(req_lw) >= LINES;",
      "  wire          req_bad = 1'b0;"),
  # tsr_dma
  "dma_no_reserve": ("tsr_dma.sv",
      "  wire rq_load = ~idle_q & (~rq_busy | rq_fire) & (g_left_q != '0) & (st_q | ld_room);",
      "  wire rq_load = ~idle_q & (~rq_busy | rq_fire) & (g_left_q != '0);"),
  "dma_reserve_off1": ("tsr_dma.sv",
      "  wire ld_room = (QAW+1)'(blen) <= ld_free_q;",
      "  wire ld_room = (QAW+1)'(blen) <= ld_free_q + 1'b1;"),
  "dma_credit_leak": ("tsr_dma.sv",
      "        ld_free_q <= ld_free_q - ((rq_load) ? (QAW+1)'(blen) : '0) + (QAW+1)'(q_pop);",
      "        ld_free_q <= ld_free_q - ((rq_load) ? (QAW+1)'(blen) : '0) + (QAW+1)'(q_pop && o_line_q[1:0] != 2'b11);"),
  "dma_old_flow": ("tsr_dma.sv", [
      ("  assign rd_rsp_ready = '1;                       // burst reservation (CC §10.4)",
       "  assign rd_rsp_ready = {NUM_CH{q_cnt_q < (QAW+1)'(QD - 1)}};"),
      ("  wire rq_load = ~idle_q & (~rq_busy | rq_fire) & (g_left_q != '0) & (st_q | ld_room);",
       "  wire rq_load = ~idle_q & (~rq_busy | rq_fire) & (g_left_q != '0);"),
      ("  wire           ld_push = ~idle_q & ~st_q & (|(rd_rsp_valid & ch_oh_q));   // rd_rsp_ready = 1",
       "  wire           ld_push = ~idle_q & ~st_q & (|(rd_rsp_valid & ch_oh_q)) & (q_cnt_q < (QAW+1)'(QD - 1));")]),
  "dma_st_credit": ("tsr_dma.sv",
      "      if (!st_q) begin\n        ld_free_q <=",
      "      if (1'b1) begin\n        ld_free_q <="),
}

NA = [("      assert (ld_free_q <= (QAW+1)'(QD)", "      if (1'b0) assert (ld_free_q <= (QAW+1)'(QD)"),
      ("        assert (ld_free_q == (QAW+1)'(QD) && q_cnt_q == '0)", "        if (1'b0) assert (ld_free_q == (QAW+1)'(QD) && q_cnt_q == '0)"),
      ("      assert (!(q_push && !q_pop && q_cnt_q == (QAW+1)'(QD)))", "      if (1'b0) assert (!(q_push && !q_pop && q_cnt_q == (QAW+1)'(QD)))")]
M["dma_no_reserve_na"] = ("tsr_dma.sv", [M["dma_no_reserve"][1:]] + NA)
M["dma_old_flow_na"] = ("tsr_dma.sv", M["dma_old_flow"][1] + NA)
M["dma_ro_skid_order"] = ("tsr_dma.sv", "      if (sk_pop)  sk_rp_q <= ~sk_rp_q;", "      if (1'b0)    sk_rp_q <= ~sk_rp_q;")
M["dma_ro_occ"] = ("tsr_dma.sv", "  wire q_pop    = (q_cnt_q != '0) & (ro_occ < 3'(QRL + 1));",
                   "  wire q_pop    = (q_cnt_q != '0) & (ro_occ < 3'(QRL + 2));")


SPM1 = "spm/test_spm.py::test_spm[small]"
SPM2 = "spm/test_spm.py::test_spm_sys[small]"
SPM2_ODD = "spm/test_spm.py::test_spm_sys[odd_lat4]"
DMA_U = "dma/test_dma.py::test_dma_unit[q512]"
DMA_R64 = "dma/test_dma.py::test_dma_rtl_spm[q64]"
NODES = {
    "spm_fixed_prio_rd1": [SPM1], "spm_no_keep_grant": [SPM1], "spm_wrong_depth_bank": [SPM1],
    "spm_rsp_early_o": [SPM1], "spm_be_slice": [SPM1], "spm_owner_first": [SPM1, SPM2],
    "cport_rsp_sel": [SPM2], "cport_ready_other": [SPM2], "cport_steer": [SPM2],
    "br_store_early": [SPM2], "br_be_shift": [SPM2], "br_no_capture": [SPM2], "br_no_range": [SPM2_ODD],
    "dma_no_reserve": [DMA_U], "dma_reserve_off1": [DMA_U], "dma_credit_leak": [DMA_U],
    "dma_old_flow": [DMA_U, DMA_R64], "dma_st_credit": [DMA_U],
    "dma_no_reserve_na": [DMA_U], "dma_old_flow_na": [DMA_R64, DMA_U], "dma_ro_skid_order": [DMA_U],
    "dma_ro_occ": [DMA_U],
}


def edits_of(spec):
    return spec[1] if isinstance(spec[1], list) else [(spec[1], spec[2])]


def write_mutant(mid):
    spec = M[mid]
    f = spec[0]
    src = open(ROOT + f).read()
    for old, new in edits_of(spec):
        assert src.count(old) == 1, (mid, src.count(old))
        src = src.replace(old, new)
    d = OUT / mid
    d.mkdir(parents=True, exist_ok=True)
    (d / f).write_text(src)
    return d / f


def main():
    args = sys.argv[1:]
    if args[:1] == ["--check"]:
        bad = 0
        for mid, spec in M.items():
            src = open(ROOT + spec[0]).read()
            n = [src.count(old) for old, _ in edits_of(spec)]
            ok = all(k == 1 for k in n)
            bad += not ok
            print("%-20s %s" % (mid, "ok" if ok else "MATCHES %s" % n))
        sys.exit(1 if bad else 0)
    only = args or list(NODES)
    res = {}
    for mid in only:
        p = write_mutant(mid)
        env = dict(os.environ, TSR_MUT="%s=%s" % (p.name, p), TSR_BUILD_JOBS=os.environ.get("TSR_BUILD_JOBS", "2"))
        out, killed, t0 = [], False, time.time()
        for node in NODES[mid]:
            r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", node], cwd=TB, env=env,
                               capture_output=True, text=True, timeout=3000)
            txt = r.stdout + r.stderr
            why = [ln.strip() for ln in txt.splitlines()
                   if "AssertionError" in ln or "Assertion failed" in ln or "TimeoutError" in ln][:2]
            out.append((node, r.returncode, why))
            if r.returncode != 0:
                killed = True
                break
        res[mid] = dict(killed=killed, runs=out, secs=round(time.time() - t0))
        print(mid, "KILLED" if killed else "SURVIVED", out[-1][2][:1], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(res, indent=1))
    sys.exit(0 if all(r["killed"] for r in res.values()) else 1)


if __name__ == "__main__":
    main()
