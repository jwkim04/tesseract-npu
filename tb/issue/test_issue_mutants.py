"""Mutation spot checks of the issue path (docs/design/04-control-core.md "CC" §15.5): each mutant is a
one-line change of tsr_chk, tsr_issue or tsr_csr written to a private copy (build/cocotb/mutants/<id>/),
then the bench of that block runs on it ($TSR_MUT_DIR) and must fail. A mutant that does not build is
reported as such, not as killed.

Opt-in (about 15 minutes with TSR_BUILD_JOBS=2): TSR_ISSUE_MUTANTS=1, or -m slow, or -k mutants.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent / "chk", HERE.parent / "csr"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from common.runner import REPO, build_root   # noqa: E402

# (id, file, original text, mutated text, bench, what CC §15.5 / the checker contract it breaks)
MUTANTS = [
    # ---- tsr_chk ----
    ("chk_spm_off_by_one", "tsr_chk.sv", "oor_q[0] <= end_q[0] > SPM_LIM;", "oor_q[0] <= end_q[0] >= SPM_LIM;",
     "chk", "range rule a + e <= limit"),
    ("chk_order_k_fmt", "tsr_chk.sv", "end else if (fmt_bad_q) begin\n      c7_code = TSR_ERR_BAD_FMT;",
     "end else if (fmt_bad_q && !krng_q) begin\n      c7_code = TSR_ERR_BAD_FMT;", "chk", "check order BAD_FMT -> K_RANGE"),
    ("chk_mvm_r_floor", "tsr_chk.sv", "c2_ndiv = 33'(w4) + 33'(NUM_CH - 1);", "c2_ndiv = 33'(w4);", "chk",
     "R = ceil(N / NUM_CH)"),
    ("chk_qact_stride0", "tsr_chk.sv", "c2_ss   = (w5 != 32'd0) ? w5 : c2_xl;", "c2_ss   = w5;", "chk",
     "QACT stride 0 = K/16"),
    ("chk_ch_limit", "tsr_chk.sv", "32'(sub) >= 32'(NUM_CH))", "32'(sub) > 32'(NUM_CH))", "chk",
     "LD / ST channel < NUM_CH"),
    ("chk_acc_extent", "tsr_chk.sv", "cand[3] = '{v: acc_q && e_q[0] != '0,", "cand[3] = '{v: 1'b0,", "chk",
     "MVM ACC lists dst as a read"),
    ("chk_q4k_kblock", "tsr_chk.sv", "(kind_q ? 16'd255 : 16'd31)", "(kind_q ? 16'd127 : 16'd31)", "chk",
     "Q4_K / A256 K must be a multiple of 256 (not of 128)"),
    ("chk_e_extent_w", "tsr_chk.sv", "e_q[2] <= TSR_EXT_AW'(rr_q) * TSR_EXT_AW'(rbp_q);",
     "e_q[2] <= TSR_EXT_AW'(rr_q[13:0]) * TSR_EXT_AW'(rbp_q);", "chk",
     "weights extent R x ROW_BEATS truncated below 2^14 beats (the multi-driven e_q class)"),
    # ---- tsr_issue ----
    ("issue_drop_dep1", "tsr_issue.sv", "dep_ok_q <= tk_done(hd_q.dep0, d_q) && tk_done(hd_q.dep1, d_q);",
     "dep_ok_q <= tk_done(hd_q.dep0, d_q);", "issue", "drop a dep compare"),
    ("issue_poison_early", "tsr_issue.sv", "if (!dep_ok_q || (is_snap && !snap_ok_q)) begin",
     "if ((!dep_ok_q || (is_snap && !snap_ok_q)) && hd_q.code == TSR_ERR_NONE) begin", "issue",
     "complete a non-running entry before its deps"),
    ("issue_skip_poison", "tsr_issue.sv", "end else if (hd_q.code != TSR_ERR_NONE) begin\n          d_nr   = drained;",
     "end else if (1'b0) begin\n          d_nr   = drained;", "issue", "skip a poison"),
    ("issue_d_before_ack", "tsr_issue.sv", "wire nr_fire  = nr_q && !nr_block;", "wire nr_fire  = nr_q;", "issue",
     "advance D[u] before the record acknowledge"),
    ("issue_qclear_oob", "tsr_issue.sv", "wire clear_now = clear_rq_q || (dec_clear[U_SYNC]);",
     "wire clear_now = clear_rq_q || (dst_q == D_POP && pop_qclr);", "issue", "apply a RISC-V QCLEAR out of band"),
    ("issue_credit_at_done", "tsr_issue.sv", "assign s_inc[gu]     = xfer || d_nr;",
     "assign s_inc[gu]     = u_done[gu] || d_nr;", "issue", "credit returned at done instead of accept"),
    ("issue_start_on_load", "tsr_issue.sv", "wire  can     = hd_v_q && ev_v_q && !cmd_v_q && !nr_q && !no_dec;",
     "wire  can     = hd_v_q && !cmd_v_q && !nr_q && !no_dec;", "issue", "a unit started in the cycle it becomes head"),
    ("issue_hard_after_abort", "tsr_issue.sv", "wire  hard    = dbg_hold[gu] && !abort_act_q && !sr_busy && !dbg_step[gu];",
     "wire  hard    = dbg_hold[gu] && !sr_busy && !dbg_step[gu];", "issue", "the hard hold kept after a QABORT"),
    ("issue_pidx_qhold", "tsr_issue.sv", "      pidx_q           <= pidx_q + 32'd1;\n    end\n  end",
     "      pidx_q           <= pidx_q + 32'd1;\n    end\n    if (dst_q == D_POP && pop_ctrl && pop_code == 4'd2) pidx_q <= pidx_q + 32'd1;\n  end",
     "issue", "the push index advanced by a QHOLD"),
    ("issue_abort_seq_offered", "tsr_issue.sv", "seq: s_q[u] + 28'(cmd_v[u]), code: TSR_ERR_ABORTED",
     "seq: s_q[u], code: TSR_ERR_ABORTED", "issue",
     "QABORT's ERR_SEQ forgets the command offered and not yet taken (CC §9.5)"),
    ("issue_resync_dropped", "tsr_issue.sv", "if (rec_resync) rec_dirty_q <= rec_dirty_q | UNITS;",
     "if (rec_resync) rec_dirty_q <= rec_dirty_q;", "issue",
     "rec_resync does not re-send the records (clk_rv's copy stays stale)"),
    ("issue_rv_hold_lost", "tsr_issue.sv", "wire hold_any   = host_hold || hold_rv_q || halt_hold;",
     "wire hold_any   = host_hold || halt_hold;", "issue", "one hold source missing"),
    # ---- tsr_csr ----
    ("csr_sr_skip_drain", "tsr_csr.sv", "SR_DRAIN: if (qidle && !qabort_q) sr_q <= SR_CLR;", "SR_DRAIN: sr_q <= SR_CLR;",
     "csr", "the SR sequencer skipping step 2"),
    ("csr_lcq_rd_rule", "tsr_csr.sv", "if ((a_wdata_q - lcq_rd_q) <= lcq_used) lcq_rd_q <= a_wdata_q;",
     "if ((a_wdata_q - lcq_rd_q) < lcq_used) lcq_rd_q <= a_wdata_q;", "csr", "LCQ_RD write rule"),
    ("csr_pop_while_active", "tsr_csr.sv", "wire         pop_ok  = (avail != 32'd0) && !active;",
     "wire         pop_ok  = (avail != 32'd0);", "csr", "LQ_POP fault while ACTIVE"),
    ("csr_no_snapshot", "tsr_csr.sv", "a_rdata_q <= (a_host_q && snap_v_q && snap_idx_q == p_idx) ? snap_hi_q : p_val[63:32];",
     "a_rdata_q <= p_val[63:32];", "csr", "PERF LO / HI snapshot"),
    ("csr_crc_no_restart", "tsr_csr.sv", "load_addr_q <= {a_wdata_q[31:2], 2'b00};\n                  load_crc_q  <= '1;",
     "load_addr_q <= {a_wdata_q[31:2], 2'b00};", "csr", "LOAD_ADDR restarts LOAD_CRC"),
    ("csr_load_addr_busy", "tsr_csr.sv", "if (vf_busy_q) sticky_q[S_LOAD_REFUSED] <= 1'b1;\n                else begin\n                  load_addr_q",
     "if (1'b0) sticky_q[S_LOAD_REFUSED] <= 1'b1;\n                else begin\n                  load_addr_q", "csr",
     "LOAD_ADDR refused while VERIFY_BUSY"),
    ("csr_boot_while_run", "tsr_csr.sv", "12'h028: if (!ctrl_q[C_RV_RUN]) boot_q <= a_wdata_q[31:8];",
     "12'h028: boot_q <= a_wdata_q[31:8];", "csr", "RV_BOOT writes ignored while RV_RUN = 1"),
    ("csr_qclear_halted", "tsr_csr.sv", "if (qidle && (rv_reset_done || rv_halted)) host_clear_q <= 1'b1;",
     "if (qidle && rv_reset_done) host_clear_q <= 1'b1;", "csr", "host QCLEAR with the hart halted (EX §5.2 bit 9)"),
    ("csr_sr_stop_valid", "tsr_csr.sv", "          stop_v_q      <= 1'b0;\n          host_clear_q  <= 1'b1;",
     "          host_clear_q  <= 1'b1;", "csr", "SOFT_RESET step 3 clears the LQ_STOP VALID bit (EX §8.5)"),
    ("csr_qclear_not_idle", "tsr_csr.sv", "if (qidle && (rv_reset_done || rv_halted)) host_clear_q <= 1'b1;",
     "if (rv_reset_done || rv_halted) host_clear_q <= 1'b1;", "csr", "host QCLEAR accepted only when QIDLE"),
    ("csr_log_free", "tsr_csr.sv", "12'h284: lq_rd_v = log_rd_q + log_drops_q;", "12'h284: lq_rd_v = log_rd_q;", "csr",
     "LOG_FREE base for the clk_rv side"),
]


def _wanted():
    return os.environ.get("TSR_ISSUE_MUTANTS") == "1" or "mutants" in " ".join(sys.argv) or "slow" in " ".join(sys.argv)


def _benches(kind):
    """The bench runs of a block: (label, callable)."""
    if kind == "chk":
        import test_chk as t
        return [("chk odd", lambda: t._run("mut", t.S.config(num_ch=3, t_max=3, k_max=4096, spm_bytes=256 << 10,
                                                               ch_bytes=1 << 30, addr_beat_w=22), n_ops=3000)),
                ("chk u200", lambda: t._run("mut4", t.S.config(num_ch=4, t_max=8), n_ops=3000))]
    if kind == "issue":
        import test_issue as t
        cfg, qd = t.trace_config(t._trace_dirs("traces/ctrl/nocaps/*")[0])
        assert (cfg["caps"], qd, cfg["t_max"], cfg["num_ch"]) == (0, t.S.U200_QD, 8, 4), "traces/ctrl/nocaps config"
        return [("issue random + directed", lambda: t._run("mut", t.U200, t.S.U200_QD, dict(n_prog=10, n_ops=60))),
                ("issue traces ctrl/nocaps", lambda: t._run("mut_tr", t.U200, t.S.U200_QD,
                                                            dict(traces="traces/ctrl/nocaps/*"), tests=["issue_traces"])),
                ("issue traces m2/errors", lambda: t._run("mut_tr2", *t.trace_config(t._trace_dirs("traces/m2/errors/*")[0]),
                                                          dict(traces="traces/m2/errors/*"), tests=["issue_traces"]))]
    import test_csr as t
    return [("csr u200", lambda: t._run("mut", t.S.config(num_ch=4, t_max=8), t.S.U200_QD))]


def _fails(fn):
    try:
        fn()
    except (SystemExit, RuntimeError, AssertionError):
        return True
    return False


_BASELINE = {}


def _baseline(kind):
    """Every bench of the block passes on the unmutated RTL (else a 'kill' would mean nothing)."""
    if kind not in _BASELINE:
        _BASELINE[kind] = [label for label, fn in _benches(kind) if _fails(fn)]
    return _BASELINE[kind]


@pytest.mark.slow
@pytest.mark.parametrize("mid", [m[0] for m in MUTANTS])
def test_issue_mutants(mid):
    if not _wanted():
        pytest.skip("opt-in: TSR_ISSUE_MUTANTS=1 or -k mutants")
    m = next(x for x in MUTANTS if x[0] == mid)
    _, fname, old, new, kind, what = m
    src = (REPO / "rtl/core" / fname).read_text()
    assert src.count(old) == 1, "mutant %s: the original text is not unique in %s (RTL changed?)" % (mid, fname)
    d = build_root() / "mutants" / mid
    d.mkdir(parents=True, exist_ok=True)
    (d / fname).write_text(src.replace(old, new))
    base = _baseline(kind)
    assert not base, "the unmutated RTL fails %s: mutants of %s are not meaningful" % (base, kind)
    os.environ["TSR_MUT_DIR"] = str(d)
    try:
        killers = [label for label, fn in _benches(kind) if _fails(fn)]
    finally:
        del os.environ["TSR_MUT_DIR"]
    print("MUTANT %s (%s): killed by %s" % (mid, what, killers or "nothing"))
    assert killers, "mutant %s survived: %s (%s)" % (mid, what, kind)
