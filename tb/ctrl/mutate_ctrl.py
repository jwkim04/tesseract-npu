"""tb/ctrl mutation spot checks (CC §15.5): each mutant is one textual change of rtl/ctrl, built in
a copy of the tree (TB_CTRL_SRC) and run against the tb/ctrl tests that must kill it. A mutant
survives if all of its tests pass; the run fails if any mutant survives.

    "$TSR_PYTHON" tb/ctrl/mutate_ctrl.py [name ...]

Opt-in (not in tb/run_all.sh: one Verilator build per mutant, about 1 min each with
TSR_BUILD_JOBS=2). The mutated tree and its build go to $TSR_SIM_BUILD_ROOT/ctrl_mut_src_<name>[_build]
and are removed afterward.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PY = sys.executable

# (name, file in rtl/ctrl, old text, new text, tests that must fail, CC §15.5 item)
MUTANTS = [
    ("pidx_on_qhold", "tsr_ctl_win.sv",
     "        end else if (ik_q == IK_QCTRL && emit_code == 3'(TSR_IFIFO_CTRL_CODE_QCLEAR)) begin\n"
     "          issued_q[USYNC] <= issued_q[USYNC] + 1'b1;\n"
     "          pidx_q   <= pidx_q + 1'b1;",
     "        end else if (ik_q == IK_QCTRL) begin\n"
     "          if (emit_code == 3'(TSR_IFIFO_CTRL_CODE_QCLEAR)) issued_q[USYNC] <= issued_q[USYNC] + 1'b1;\n"
     "          pidx_q   <= pidx_q + 1'b1;",
     ["lockstep"], "ISSUE_PIDX advanced by a QHOLD"),
    ("lb_no_inv_cw", "tsr_ctl_win.sv",
     "          if (hart && (w_region == RG_CW || w_region == RG_CORE)) lb_valid_q <= 1'b0;",
     "          if (hart && (w_region == RG_CORE)) lb_valid_q <= 1'b0;",
     ["lb"], "the line buffer not invalidated on one event (control-window access)"),
    ("dep_kept", "tsr_ctl_win.sv",
     "          dep0_q   <= TICKET_NONE;\n",
     "",
     ["lockstep"], "DEP0 not reset by a push (CC §5.1)"),
    ("wmask_ignored", "tsr_ctl_win.sv",
     "(i == 0 || ival_q[TSR_ISSUE_PUSH_WMASK_LSB + i]) ? iss_w_q[i] : '0;",
     "iss_w_q[i];",
     ["lockstep"], "WMASK not applied to the issue-FIFO entry (CC §7.1)"),
    ("sba_cw_write", "tsr_rv_bus.sv",
     "((a_rg == RG_CW || a_rg == RG_CORE) && (a_we || !a_word)) ||",
     "((a_rg == RG_CW || a_rg == RG_CORE) && !a_word) || (a_rg == RG_CORE && a_we) ||",
     ["dmi"], "an SBA write to the control window not refused (CC-33)"),
    ("wait_no_suspend", "tsr_ctl_win.sv",
     "            if (!dbg_stop) begin\n              if (32'(wait_q) + 1 >= ISSUE_WAIT_MAX) begin",
     "            if (1'b1) begin\n              if (32'(wait_q) + 1 >= ISSUE_WAIT_MAX) begin",
     ["push_wait"], "the ISSUE_WAIT_MAX count not suspended under dbg_stop (CC §7.3)"),
    ("late_answer_taken", "cdc/tsr_cdc_hs.sv",
     "          end else if (a_ack) begin\n            a_st_q <= A_IDLE;\n          end\n        end\n        A_DRAIN_PEND: begin\n"
     "          if (a_ack) begin\n            req_q     <= pend_q;\n            req_tgl_q <= ~req_tgl_q;\n            a_st_q    <= A_BUSY;",
     "          end else if (a_ack) begin\n            a_st_q <= A_IDLE;\n          end\n        end\n        A_DRAIN_PEND: begin\n"
     "          if (a_ack) begin\n            a_done     <= 1'b1;\n            a_rsp_data <= rsp_q;\n            a_st_q     <= A_IDLE;",
     ["bridge"], "a late handshake answer not discarded (taken for the next request)"),
    ("push_not_dropped", "tsr_ctrl_rv.sv",
     "  assign rv_kill       = sq_q == SQ_DRAIN;",
     "  assign rv_kill       = 1'b0;",
     ["rv_reset"], "a push waiting for a credit not dropped by the RISC-V reset (CC §12.2)"),
    ("err_rec_lost", "tsr_ctrl_cdc.sv",
     "      else if (b_valid) r_err_rec[u] <= b_req;",
     "      else if (b_valid && r_err_rec[u] == '0) r_err_rec[u] <= '0;",
     ["err_rec"], "an error record acknowledged without clk_rv showing it (I2)"),
    ("dtcm_no_fwd", "tsr_rv_bus.sv",
     "      fwd_q <= a_dtcm_rd && dw_q && dw_addr_q == a_addr[DAW+1:2];",
     "      fwd_q <= 1'b0;",
     ["lockstep"], "a DTCM load right after a store of the same word misses the store (bus forwarding)"),
    ("resync_lost", "tsr_ctrl_cdc.sv",
     "  assign rec_resync_o = rv_ok && !rv_ok_d_q;",
     "  assign rec_resync_o = 1'b0;",
     ["err_rec"], "no record re-send when clk_rv is OK again after local acknowledges (clk_rv's copy stays stale)"),
    ("credit_no_fifo", "tsr_ctl_win.sv",
     "      if (c > 32'(if_free)) c = 32'(if_free);\n",
     "",
     ["fifo_full"], "credits without the issue-FIFO free count (CC §7.3)"),
]


def run_mutant(name, fname, old, new, tests):
    src = REPO / "rtl" / "ctrl"
    dst = Path(os.environ.get("TSR_SIM_BUILD_ROOT", REPO / "build" / "cocotb")) / ("ctrl_mut_src_" + name)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    p = dst / fname
    text = p.read_text()
    if text.count(old) != 1:
        return "BADSITE"
    p.write_text(text.replace(old, new))
    env = dict(os.environ)
    env["TB_CTRL_SRC"] = str(dst)
    env["TSR_SIM_BUILD_ROOT"] = str(dst) + "_build"     # removed after the mutant
    killed = []
    for t in tests:
        r = subprocess.run([PY, "-m", "pytest", "-q", "-x", str(HERE / "test_ctrl.py"), "-k", "test_ctrl_" + t],
                           cwd=REPO / "tb", env=env, capture_output=True, text=True)
        killed.append(r.returncode != 0)
        if r.returncode not in (0, 1):
            print(r.stdout[-2000:], r.stderr[-2000:])
    shutil.rmtree(dst, ignore_errors=True)
    shutil.rmtree(str(dst) + "_build", ignore_errors=True)
    return "KILLED" if any(killed) else "SURVIVED"


def main(argv):
    names = set(argv)
    bad = 0
    for name, fname, old, new, tests, what in MUTANTS:
        if names and name not in names:
            continue
        res = run_mutant(name, fname, old, new, tests)
        print("%-18s %-9s %s (tests: %s)" % (name, res, what, ", ".join(tests)), flush=True)
        bad += res != "KILLED"
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
