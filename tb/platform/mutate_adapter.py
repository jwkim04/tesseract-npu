#!/usr/bin/env python3
"""tb/platform/mutate_adapter.py — mutation campaign of the Alveo XRT platform adapter: every mutant
is applied to a scratch copy of rtl/platform/alveo_xrt (never to the repository) and the kernel
tests must fail.

    python3 tb/platform/mutate_adapter.py WORK_DIR [--only NAME[,NAME]] [--list] [--procs N]

Kill stages (the first failing stage kills; later stages are skipped):
  1  tb/fastreplay --top kernel (kernel_bench.cpp replay, --wno-fatal, --stop-on-fail) of
     traces/m3/ch3 + traces/m3/small1ch + traces/m2/small1ch under the rand, fast and slow regimes
     (every replay check
     + the kernel checks: AXI4 / AXI4-Lite protocol, AP_CTRL / ISR / interrupt after each run,
     ADAPTER_STATUS == injected, PERF_CYCLES vs the host-side cycles)
  2  the directed tests (kernel_bench --directed all, seed 1) on the stage-1 build of traces/m3/ch3
A mutant may name a kernel parameter override (e.g. AXI_SLICES=1) for both stages.
Results: WORK_DIR/results.json (name, stage, killed, first error) and a table on stdout.
Builds go to WORK_DIR/build (TSR_FASTREPLAY_BUILD); make -j$TSR_BUILD_JOBS (default 4).
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
PDIR = REPO / "rtl" / "platform" / "alveo_xrt"
STAGE1_TRACES = [REPO / "traces/m3/ch3", REPO / "traces/m3/small1ch", REPO / "traces/m2/small1ch"]
DIRECTED_TRACES = sorted((REPO / "traces/m3/ch3").iterdir())[:3]

# (name, [(file, old (exactly once), new), ...], what it breaks, params, expected)
MUTANTS = [
    ("NONE", [], "no mutation: the baseline must survive both stages", {}, "survived"),
    ("B_len_no_minus1", [("tsr_xrt_axi_bridge.sv", "return {b + off, 8'(l - TSR_LEN_W'(1)), id};",
                          "return {b + off, 8'(l), id};")],
     "AxLEN = len instead of len - 1", {}, "killed"),
    ("B_no_base", [("tsr_xrt_axi_bridge.sv", "return {b + off, 8'(l - TSR_LEN_W'(1)), id};",
                    "return {off, 8'(l - TSR_LEN_W'(1)), id};")],
     "channel base not added", {}, "killed"),
    ("B_base_low32", [("tsr_xrt_axi_bridge.sv", "return {b + off, 8'(l - TSR_LEN_W'(1)), id};",
                       "return {{32'd0, b[31:0] + off[31:0]}, 8'(l - TSR_LEN_W'(1)), id};")],
     "32-bit address adder (no carry into the upper half)", {}, "killed"),
    ("B_wstrb", [("tsr_xrt_axi_bridge.sv", "assign m_axi_wstrb   = '1;",
                  "assign m_axi_wstrb   = {1'b0, {(BEAT_BITS/8-1){1'b1}}};")],
     "WSTRB misses byte 63", {}, "killed"),
    ("B_cache", [("tsr_xrt_axi_bridge.sv", "assign m_axi_arcache = 4'b0011;", "assign m_axi_arcache = 4'b0000;")],
     "ARCACHE 0000 instead of 0011", {}, "killed"),
    ("B_rid_swap", [("tsr_xrt_axi_bridge.sv", ".s_valid(r_fwd), .s_ready(ro_ready), .s_data({rh_data, rh_id, rh_last}),",
                     ".s_valid(r_fwd), .s_ready(ro_ready), .s_data({rh_data, rh_id ^ 4'h4, rh_last}),")],
     "read response id bit 2 flipped towards the core", {}, "killed"),
    ("B_rlast_drop", [("tsr_xrt_axi_bridge.sv", ".s_valid(r_fwd), .s_ready(ro_ready), .s_data({rh_data, rh_id, rh_last}),",
                       ".s_valid(r_fwd), .s_ready(ro_ready), .s_data({rh_data, rh_id, rh_last & (rh_id != 4'd4)}),")],
     "RLAST of DMA reads (id 4) not forwarded", {}, "killed"),
    ("B_filter_r_off", [("tsr_xrt_axi_bridge.sv", "wire r_exp  = rd_nz_q[rh_id];", "wire r_exp  = 1'b1;")],
     "stale R beats forwarded to the core", {}, "killed"),
    ("B_filter_b_off", [("tsr_xrt_axi_bridge.sv", "wire b_exp  = wr_nz_q[bh_id];", "wire b_exp  = 1'b1;")],
     "stale B responses forwarded to the core", {}, "killed"),
    ("B_cnt_no_ws", [("tsr_xrt_axi_bridge.sv", "rn = rd_cnt_q[i] + RD_CNT_W'(ar_hs && ar_id == IDW'(i))",
                      "rn = rd_cnt_q[i] + RD_CNT_W'(ar_hs && ar_id == IDW'(i) && i != 8)")],
     "weight-stream bursts (id 8) not counted: their beats are dropped", {}, "killed"),
    ("B_hs_reg_slices1", [("tsr_xrt_axi_bridge.sv", "if (SLICES >= 2) begin : g_hs_q", "if (SLICES >= 1) begin : g_hs_q")],
     "registered handshake with a single slice", {"AXI_SLICES": 1},
     "equivalent (count updated exactly when the earliest beat reaches the filter: zero margin, still correct)"),
    ("B_hs_late", [("tsr_xrt_axi_bridge.sv", "        ar_hs_q <= m_axi_arvalid & m_axi_arready;",
                    "        ar_hs_q <= ar_hs_q2;\n        ar_hs_q2 <= ar_hs_q3;\n        ar_hs_q3 <= m_axi_arvalid & m_axi_arready;"),
                   ("tsr_xrt_axi_bridge.sv", "    logic           ar_hs_q, aw_hs_q;", "    logic           ar_hs_q, aw_hs_q, ar_hs_q2, ar_hs_q3;"),
                   ("tsr_xrt_axi_bridge.sv", "      ar_id_q <= m_axi_arid;", "      ar_id_q <= ar_id_q2;\n      ar_id_q2 <= ar_id_q3;\n      ar_id_q3 <= m_axi_arid;"),
                   ("tsr_xrt_axi_bridge.sv", "    logic [IDW-1:0] ar_id_q, aw_id_q;", "    logic [IDW-1:0] ar_id_q, aw_id_q, ar_id_q2, ar_id_q3;")],
     "read handshake registered three times: a beat right after its AR reaches the filter before its count",
     {}, "killed"),
    ("B_hs_late2", [("tsr_xrt_axi_bridge.sv", "        ar_hs_q <= m_axi_arvalid & m_axi_arready;",
                     "        ar_hs_q <= ar_hs_q2;\n        ar_hs_q2 <= m_axi_arvalid & m_axi_arready;"),
                    ("tsr_xrt_axi_bridge.sv", "    logic           ar_hs_q, aw_hs_q;", "    logic           ar_hs_q, aw_hs_q, ar_hs_q2;"),
                    ("tsr_xrt_axi_bridge.sv", "      ar_id_q <= m_axi_arid;", "      ar_id_q <= ar_id_q2;\n      ar_id_q2 <= m_axi_arid;"),
                    ("tsr_xrt_axi_bridge.sv", "    logic [IDW-1:0] ar_id_q, aw_id_q;", "    logic [IDW-1:0] ar_id_q, aw_id_q, ar_id_q2;")],
     "read handshake registered twice", {},
     "equivalent (with 2 slices the count is updated exactly when the earliest beat reaches the filter)"),
    ("B_rresp_ev_off", [("tsr_xrt_axi_bridge.sv", "ev_rresp_err  <= r_take & (rh_resp != 2'b00);",
                         "ev_rresp_err  <= 1'b0;")],
     "RRESP errors not reported", {}, "killed"),
    ("C_idle_stays", [("tsr_xrt_ctrl.sv", "            ap_idle_q <= 1'b0;\n            step_q    <= '0;",
                       "            step_q    <= '0;")],
     "ap_idle not dropped when a run starts", {}, "killed"),
    ("C_no_cor", [("tsr_xrt_ctrl.sv", "      if (ctrl_rd) begin\n        ap_done_q  <= 1'b0;\n        ap_ready_q <= 1'b0;\n      end\n", "")],
     "ap_done / ap_ready not cleared on read", {}, "killed"),
    ("C_ready_missing", [("tsr_xrt_ctrl.sv", "          ap_ready_q  <= 1'b1;\n", "")],
     "ap_ready never set", {}, "killed"),
    ("C_isr_no_ier", [("tsr_xrt_ctrl.sv", "          if (ier_q) isr_q <= 1'b1;", "          isr_q <= 1'b1;")],
     "ISR set although IER = 0", {}, "killed"),
    ("C_irq_no_gie", [("tsr_xrt_ctrl.sv", "irq_q  <= gie_q & ier_q & isr_q;", "irq_q  <= ier_q & isr_q;")],
     "interrupt ignores GIE", {}, "killed"),
    ("C_isr_tow_set", [("tsr_xrt_ctrl.sv", "if (isr_tow) isr_q <= ~isr_q;", "if (isr_tow) isr_q <= 1'b0;")],
     "ISR write clears instead of toggling", {}, "killed"),
    ("C_misaligned_off", [("tsr_xrt_ctrl.sv", "misaligned |= (base_reg_q[c][11:0] != 12'd0);",
                           "misaligned |= 1'b0;")],
     "misaligned base not refused", {}, "killed"),
    ("C_srst_poll_off", [("tsr_xrt_ctrl.sv", "if (rs != RS_IDLE && srst_wr) srst_seen_q <= 1'b1;", "")],
     "SOFT_RESET during a run: no STATUS polling (hang)", {}, "killed"),
    ("C_no_w1c", [("tsr_xrt_ctrl.sv", "4'd11:   begin a = TSR_CSR_STATUS;      d = ST_DONE | ST_ERR;  end",
                   "4'd11:   begin a = TSR_CSR_STATUS;      d = 32'd0;  end")],
     "start sequence does not clear STATUS.DONE / ERR", {},
     "equivalent (the core's START clears DONE / ERR itself: STATUS := BUSY; the W1C step is contractual)"),
    ("C_args_rev", [("tsr_xrt_ctrl.sv", "d = arg_q[3'(k - 4'd3)]; end", "d = arg_q[3'(4'd10 - k)]; end")],
     "ARG0..7 written in reverse order", {}, "killed"),
    ("C_cmd_hi_lo", [("tsr_xrt_ctrl.sv", "4'd1:    begin a = TSR_CSR_CMD_ADDR_HI; d = cmd_addr_q[63:32]; end",
                      "4'd1:    begin a = TSR_CSR_CMD_ADDR_HI; d = cmd_addr_q[31:0]; end")],
     "CMD_ADDR_HI gets the low word", {}, "killed"),
    ("C_base_live", [("tsr_xrt_ctrl.sv", "  assign ch_base   = base_q;", "  assign ch_base   = base_reg_q;")],
     "bases not latched at the start (live registers)", {}, "killed"),
    ("C_win_strb", [("tsr_xrt_ctrl.sv", "if (ax_win && (!ax_wr_q || ax_wstrb_q == 4'hF)) begin",
                     "if (ax_win) begin")],
     "partial-strobe window writes performed", {}, "killed"),
    ("C_cnt_wrap", [("tsr_xrt_ctrl.sv", "acnt_d = sum[16] ? 16'hFFFF : sum[15:0];", "acnt_d = sum[15:0];")],
     "error count wraps instead of saturating", {}, "killed"),
    ("C_w1c_count_any", [("tsr_xrt_ctrl.sv", "if ((ax_wstrb_q[2] && ax_wdata_q[23:16] != 8'd0) || (ax_wstrb_q[3] && ax_wdata_q[31:24] != 8'd0))",
                          "if (ax_wstrb_q[0])")],
     "any ADAPTER_STATUS write clears the count", {}, "killed"),
    ("C_b_early", [("tsr_xrt_ctrl.sv", "            awready_q <= 1'b1;\n            wready_q  <= 1'b1;\n",
                    "            awready_q <= 1'b1;\n            wready_q  <= 1'b1;\n            bvalid_q  <= 1'b1;\n")],
     "B before the AW / W handshakes", {}, "killed"),
    ("C_rdata_live", [("tsr_xrt_ctrl.sv", "    if (reg_rd) rdata_q <= reg_rdata;", "    if (1'b1) rdata_q <= reg_rdata;")],
     "RDATA not held while RVALID waits for RREADY", {}, "killed"),
    ("C_done_early", [("tsr_xrt_ctrl.sv", "          end else if (done_q) begin\n            rs <= RS_FIN;",
                       "          end else if (1'b1) begin\n            rs <= RS_FIN;")],
     "completion reported at once (before the core's done)", {}, "killed"),
    ("R_core_first", [("tsr_xrt_reset.sv", "  assign rst_core_n = core_q[CORE_STAGES-2];", "  assign rst_core_n = rst1_q;")],
     "core leaves reset one cycle before the bridges", {}, "equivalent? (the core issues no request "
     "until a START, many cycles after reset)"),
    ("R_ch_copy_late", [("tsr_xrt_reset.sv", "    ch_q   <= {(N_CH * CH_COPIES){rst1_q}};",
                         "    for (int unsigned c = 0; c < N_CH; c++) ch_q[c] <= {ctrl_q, {(CH_COPIES - 1){rst1_q}}};")],
     "the last reset copy of every channel (bridge write counters of IDs 12..15) one cycle late", {},
     "killed (the bridge's reset-copy check; functionally the parts then leave reset one cycle apart)"),
]


def sh(cmd, log: Path, env=None, timeout=None) -> int:
    with open(log, "a") as f:
        f.write("$ %s\n" % " ".join(str(c) for c in cmd))
        f.flush()
        try:
            r = subprocess.run([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT, env=env, timeout=timeout)
            return r.returncode
        except subprocess.TimeoutExpired:
            f.write("TIMEOUT\n")
            return 124


def make_copy(work: Path, name: str, edits) -> Path:
    d = work / "src" / name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    for f in PDIR.iterdir():
        if f.suffix in (".sv", ".v", ".f"):
            shutil.copy(f, d / f.name)
    for fname, old, new in edits:
        p = d / fname
        t = p.read_text()
        n = t.count(old)
        if n != 1:
            raise SystemExit("mutant %s: pattern found %d times in %s:\n%s" % (name, n, fname, old))
        p.write_text(t.replace(old, new))
    return d / "filelist.f"


def first_error(log: Path) -> str:
    """The first failure of the log: a replay FAIL with its first error, a directed FAIL with its
    first finding, an RTL error or a run.py error."""
    lines = [l.strip() for l in log.read_text(errors="replace").splitlines()]
    for i, s in enumerate(lines):
        if s.startswith("FAIL") or "%Error" in s or s.startswith("run.py:"):
            nxt = lines[i + 1] if i + 1 < len(lines) and not s.startswith("run.py:") else ""
            return (s + (" :: " + nxt if nxt else ""))[:400]
    return ""


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("work")
    ap.add_argument("--only", default="")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    if a.list:
        for m in MUTANTS:
            print("%-20s %s" % (m[0], m[2]))
        return 0
    work = Path(a.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, TSR_FASTREPLAY_BUILD=str(work / "build"))
    only = set(x for x in a.only.split(",") if x)
    results = []
    for name, edits, what, params, expected in MUTANTS:
        if only and name not in only:
            continue
        t0 = time.time()
        fl = make_copy(work, name, edits)
        log = work / ("%s.log" % name)
        log.write_text("")
        pargs = []
        for k, v in params.items():
            pargs += ["--param", "%s=%s" % (k, v)]
        js = work / ("%s.json" % name)
        rc = sh([sys.executable, REPO / "tb/fastreplay/run.py"] + STAGE1_TRACES +
                ["--top", "kernel", "--has-mvm", "1", "--platform-filelist", fl, "--wno-fatal", "--regimes",
                 "rand,fast,slow", "--procs", str(a.procs), "--stop-on-fail", "-q", "--json", js] + pargs, log,
                env=env, timeout=3600)
        stage, killed = 1, rc != 0
        if not killed:
            # the ch3 build of this mutant: the newest kern_c3_* under the work build root
            cands = sorted((work / "build").glob("kern_c3_t3_m1_*/obj/fastreplay"), key=lambda p: p.stat().st_mtime)
            binary = cands[-1] if cands else None
            if binary is None:
                raise SystemExit("mutant %s: no ch3 kernel build found" % name)
            rc = sh([binary, "--directed", "all", "--seed", "1"] + DIRECTED_TRACES, log, env=env, timeout=1800)
            stage, killed = 2, rc != 0
        err = first_error(log) if killed else ""
        dt = time.time() - t0
        results.append(dict(name=name, what=what, params=params, expected=expected, killed=killed,
                            stage=stage if killed else None, first_error=err, seconds=round(dt, 1)))
        print("%-20s %-9s stage %s  %5.0f s  %s" % (name, "KILLED" if killed else "survived",
                                                   stage if killed else "-", dt, err[:150]), flush=True)
        (work / "results.json").write_text(json.dumps(results, indent=1))
    n_k = sum(r["killed"] for r in results)
    print("%d mutants, %d killed, %d survived" % (len(results), n_k, len(results) - n_k))
    return 0


if __name__ == "__main__":
    sys.exit(main())
