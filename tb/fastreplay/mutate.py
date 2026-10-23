#!/usr/bin/env python3
"""tb/fastreplay/mutate.py — mutation campaign for the fast replayer: every mutant is applied to a
scratch copy (never to the repository), and the replay / self-test must fail on it.

    tb/fastreplay/mutate.py [--set rtl|model|all] [--only NAME[,NAME]] [--work DIR] [--keep]

  rtl    M2 RTL mutants of tsr_core blocks (scratch copy of rtl/, its own filelist.f): replayed
         with run.py on traces/m2 (all regimes) and, if that does not kill it, on the small1ch /
         ch3 sets with implementation parameters RD_OUTSTANDING 2, SPM_RD_LAT 1, FETCH_BEATS 2.
         Killed = at least one failing run (the first error is reported).
  model  mutants of the memory model (memmodel.h, scratch copy) against selftest.cpp.
Work directory: --work, else $TSR_FASTREPLAY_BUILD/../mutants, else build/fastreplay_mutants.
Exit status 0 when every mutant was killed (survivors are listed).
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import run as fr  # noqa: E402  (tb/fastreplay/run.py)

# (name, file under rtl/, old text (must occur exactly once), new text, what it breaks)
RTL_MUTANTS = [
    ("qact_qs_tail", "core/tsr_qact.sv",
     "else if (w_sq == 2'd0)  qs_asm[128*s +: 128] <= '0;", "else if (1'b0)  qs_asm[128*s +: 128] <= '0;",
     "QACT QS line tail not zeroed (spec 꼬리 바이트)"),
    ("qact_round_kind", "core/tsr_qact.sv",
     "else                    q2_q[j] <= tsr_f32_to_q8(m_y[j], c_kind);",
     "else                    q2_q[j] <= tsr_f32_to_q8(m_y[j], 1'b1);",
     "A32 rounds half-even instead of half away from zero"),
    ("arb_outstanding_plus1", "core/tsr_mem_arb.sv",
     "wire rd_cnt_ok = (rd_cnt_q < CNT_W'(RD_OUTSTANDING));", "wire rd_cnt_ok = (rd_cnt_q <= CNT_W'(RD_OUTSTANDING));",
     "arbiter allows RD_OUTSTANDING + 1 reads (rtl-m2 §4 M2)"),
    ("arb_perf_beat_last", "core/tsr_mem_arb.sv",
     "perf_rd_beat <= rd_beat;", "perf_rd_beat <= rd_beat & ~m_rd_rsp_last;",
     "PERF_CH_RD_BEATS misses the last beat of each burst"),
    ("rslice_payload", "core/tsr_mem_arb.sv",
     "if (m_free && (!rdy_q || s_valid)) main_q <= rdy_q ? s_data : skid_q;",
     "if ((!rdy_q || s_valid)) main_q <= rdy_q ? s_data : skid_q;",
     "register slice overwrites a stalled output payload (R2)"),
    ("dma_st_done_early", "core/tsr_dma.sv",
     "& (o_left_q == '0) & (wr_pend_q == '0);", "& (o_left_q == '0);",
     "ST done before its write responses (DONE => quiescent)"),
    ("dma_burst_cross", "core/tsr_dma.sv",
     "wire [TSR_LEN_W-1:0] to_blk_end = TSR_LEN_W'(7'd64 - {1'b0, 6'(g_addr_q)});",
     "wire [TSR_LEN_W-1:0] to_blk_end = TSR_LEN_W'(7'd64);",
     "DMA bursts cross 4 KiB blocks (M1)"),
    ("dma_wr_last", "core/tsr_dma.sv",
     "o_last_q  <= (p_lo_q == 6'h3F) || (p_left_q == NW'(1));", "o_last_q  <= (p_left_q == NW'(1));",
     "wr_dat_last missing at 4 KiB burst ends (M4)"),
    ("dma_wr_req_waits_dat_ready", "core/tsr_dma.sv",
     "assign wr_req_valid = rq_wr_v_q;", "assign wr_req_valid = rq_wr_v_q & wr_dat_ready;",
     "write request waits for write-data ready (M5, R2)"),
    ("dma_wr_pend_no_inc", "core/tsr_dma.sv",
     "wr_pend_q <= wr_pend_q + NW'(rq_fire) - NW'(wr_rsp_fire);", "wr_pend_q <= wr_pend_q - NW'(wr_rsp_fire);",
     "ST write-pending count not incremented (RTL assertion + early done)"),
    ("fetch_overread", "core/tsr_fetch.sv",
     "nbeats_q <= {1'b0, count_q[31:1]} + {31'd0, count_q[0]};", "nbeats_q <= {1'b0, count_q[31:1]} + 32'd1;",
     "fetch reads one beat past the command buffer"),
    ("cmd_finish_before_fetch_idle", "core/tsr_cmdproc.sv",
     "if (fetch_idle && !fetch_stop_q) begin", "if (!fetch_stop_q) begin",
     "run_finish without waiting for fetch idle (F4)"),
    ("cmd_dram_err_info", "core/tsr_cmdproc.sv",
     "ex_info = op_mvm_q ? a3_q : a2_q;", "ex_info = a1_q;",
     "DRAM_RANGE ERR_INFO from the wrong operand"),
    ("csr_perf_wait_swap", "core/tsr_csr.sv",
     "perf_inc[TSR_PERF_CH_RD_BEATS + c]          = perf_ch_rd_beat[c];",
     "perf_inc[TSR_PERF_CH_RD_BEATS + c]          = perf_ch_wait[c];",
     "PERF_CH_RD_BEATS counts wait cycles"),
    ("csr_perf_cycles_always", "core/tsr_csr.sv",
     "perf_inc[TSR_PERF_CYCLES]     = busy_q;", "perf_inc[TSR_PERF_CYCLES]     = 1'b1;",
     "PERF_CYCLES counts while not BUSY"),
]

# Memory model mutants (memmodel.h): the self-test must fail.
MODEL_MUTANTS = [
    ("mm_no_len_check", "if (len < 1 || len > 64) fail(", "if (false) fail(", "P1 length"),
    ("mm_no_cross_check", "if ((addr % 64) + len > 64)", "if (false)", "P1 4 KiB block"),
    ("mm_no_limit_check", "if (addr + len > cfg_.limit)\n", "if (false)\n", "P1 limit"),
    ("mm_outstanding_plus1", "if (ch.outstanding >= cfg_.rd_outstanding)", "if (ch.outstanding > cfg_.rd_outstanding)",
     "P2"),
    ("mm_no_payload_check", "(o.rd_req_addr != ch.prq_addr || o.rd_req_len != ch.prq_len || o.rd_req_id != ch.prq_id)",
     "(false)", "P3 rd_req"),
    ("mm_no_last_check", "if (d.last != (i == w.len - 1))", "if (false)", "P4"),
    ("mm_fetch_any_channel", "(src == SRC_FETCH && c != 0)", "(false)", "P5"),
    ("mm_no_inactive_check", "if (!active_ && (o.rd_req_valid", "if (false && (o.rd_req_valid", "P6"),
    ("mm_no_poison", "memset(in.rd_rsp_data, 0xE5, 64);", "", "poison"),
    ("mm_interleave_always", "const long k = pick(ch, ch.rd, !reg_.interleave);", "const long k = pick(ch, ch.rd, false);",
     "ordered regime"),
    ("mm_same_id_reorder", "if (seen & (1u << id)) continue;", "", "same-id order"),
    ("mm_data_first_broken", "ch.wr_req_ready = !stalled && need < 0 && ch.rng.chance(thr_wr_req_);",
     "ch.wr_req_ready = !stalled && ch.rng.chance(thr_wr_req_);", "M5 data_first"),
    ("mm_bad_read_untracked", "if (beat < cfg_.limit && !ch.written.get(beat) && !ch.allowed.get(beat)) {",
     "if (false) {", "read tracking"),
    ("mm_written_not_tracked", "ch.written.set(beat);\n", "\n", "written set"),
    ("mm_wrsp_never", "ch.wrsp.push_back(WRsp{", "if (false) ch.wrsp.push_back(WRsp{", "write response"),
]


def patch(path: Path, old: str, new: str):
    s = path.read_text()
    n = s.count(old)
    if n != 1:
        raise SystemExit("mutate.py: %s: pattern occurs %d times: %r" % (path, n, old))
    path.write_text(s.replace(old, new))


def rtl_campaign(work: Path, only, log=print) -> list:
    out = []
    specs = [
        dict(paths=[str(REPO / "traces" / "m2")], regimes=["all"], params={}),
        dict(paths=[str(REPO / "traces" / "m2" / "small1ch"), str(REPO / "traces" / "m2" / "ch3")], regimes=["all"],
             params=dict(RD_OUTSTANDING=2, SPM_RD_LAT=1, FETCH_BEATS=2)),
    ]
    saved = os.environ.get("TSR_FASTREPLAY_BUILD")
    try:
        for name, f, old, new, what in RTL_MUTANTS:
            if only and name not in only:
                continue
            out.append(_rtl_one(work, name, f, old, new, what, specs, log))
    finally:
        if saved is None:
            os.environ.pop("TSR_FASTREPLAY_BUILD", None)
        else:
            os.environ["TSR_FASTREPLAY_BUILD"] = saved
    return out


def _rtl_one(work, name, f, old, new, what, specs, log):
    mdir = work / name
    if mdir.exists():
        shutil.rmtree(mdir)
    shutil.copytree(REPO / "rtl", mdir / "rtl")
    patch(mdir / "rtl" / f, old, new)
    os.environ["TSR_FASTREPLAY_BUILD"] = str(mdir / "build")
    t0 = time.time()
    verdict, first = "survived", ""
    try:
        for sp in specs:
            rep = fr.replay(sp["paths"], regimes=sp["regimes"], params=sp["params"], has_mvm="0",
                            filelist=mdir / "rtl" / "filelist.f", stop_on_fail=True, wno_fatal=True,
                            log=lambda *_: None, echo=None)
            if rep.failed:
                fails = [f for g in rep.groups for f in g.failures]
                verdict = "killed"
                first = "%s [%s%s]: %s" % (Path(fails[0]["trace"]).name, fails[0]["regime"],
                                           (" " + " ".join("%s=%s" % kv for kv in sp["params"].items()))
                                           if sp["params"] else "", fails[0]["errors"][0] if fails[0]["errors"] else "?")
                break
    except (RuntimeError, SystemExit) as e:
        verdict, first = "killed (build/elaboration)", str(e).splitlines()[0][:200]
    dt = time.time() - t0
    log("%-30s %-10s %5.0f s  %s | %s" % (name, verdict, dt, what, first))
    shutil.rmtree(mdir, ignore_errors=True)
    return (name, verdict, what, first)


def model_campaign(work: Path, only, log=print) -> list:
    out = []
    cxx = os.environ.get("CXX", "g++")
    for name, old, new, what in [(None, None, None, "unmutated")] + MODEL_MUTANTS:
        if only and name is not None and name not in only:
            continue
        mdir = work / (name or "model_base")
        if mdir.exists():
            shutil.rmtree(mdir)
        mdir.mkdir(parents=True)
        shutil.copy(HERE / "memmodel.h", mdir / "memmodel.h")
        shutil.copy(HERE / "selftest.cpp", mdir / "selftest.cpp")
        if name:
            patch(mdir / "memmodel.h", old, new)
        exe = mdir / "selftest"
        r = subprocess.run([cxx, "-std=c++17", "-O1", "-I", str(mdir), str(mdir / "selftest.cpp"), "-o", str(exe)],
                           capture_output=True, text=True)
        if r.returncode:
            verdict, first = "killed (compile)", r.stderr.splitlines()[0] if r.stderr else "?"
        else:
            t = subprocess.run([str(exe)], capture_output=True, text=True, timeout=300)
            fails = [l for l in t.stdout.splitlines() if l.startswith("FAIL ")]
            if name is None:
                verdict = "ok" if t.returncode == 0 else "BASE FAILS"
            else:
                verdict = "killed" if t.returncode else "survived"
            first = fails[0][5:] if fails else ""
        log("%-30s %-10s %s | %s" % (name or "(unmutated)", verdict, what, first))
        if name:
            out.append((name, verdict, what, first))
        shutil.rmtree(mdir, ignore_errors=True)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--set", default="all", choices=["rtl", "model", "all"])
    ap.add_argument("--only", default="", help="comma list of mutant names")
    ap.add_argument("--work", default=None)
    a = ap.parse_args(argv)
    only = set(x for x in a.only.split(",") if x)
    if a.work:
        work = Path(a.work)
    elif os.environ.get("TSR_FASTREPLAY_BUILD"):
        work = Path(os.environ["TSR_FASTREPLAY_BUILD"]).parent / "mutants"
    else:
        work = REPO / "build" / "fastreplay_mutants"
    work.mkdir(parents=True, exist_ok=True)
    res = []
    if a.set in ("model", "all"):
        res += model_campaign(work, only)
    if a.set in ("rtl", "all"):
        res += rtl_campaign(work, only)
    surv = [r for r in res if r[1] == "survived"]
    print("mutate.py: %d mutants, %d killed, %d survived%s" % (
        len(res), len(res) - len(surv), len(surv), (": " + ", ".join(r[0] for r in surv)) if surv else ""))
    return 1 if surv else 0


if __name__ == "__main__":
    sys.exit(main())
