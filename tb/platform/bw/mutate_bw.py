#!/usr/bin/env python3
"""tb/platform/bw/mutate_bw.py — mutation campaign of the M0 bandwidth kernel RTL against its bench
(docs/m0-bw.md §5): every mutant is one textual change of rtl/platform/alveo_xrt/bw/*.sv applied to
a scratch copy, built with run_bw.py (-Wno-fatal) and run through a reduced test set; a mutant is
killed when the bench (or the build's elaboration checks) fails.

    mutate_bw.py --work DIR [--only NAME,...] [--procs N] [--seed N] [--param K=V]...

Results: DIR/results.json ({mutant: {"killed": bool, "first": first failing line, "seconds": s}}).
The unmutated copy (NONE) must pass. Exit status 0 when NONE passes (the survivors are listed).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RTL_BW = REPO / "rtl" / "platform" / "alveo_xrt" / "bw"

_spec = importlib.util.spec_from_file_location("tsr_run_bw", HERE / "run_bw.py")
rb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rb)

E, C = "tsr_bw_engine.sv", "tsr_bw_ctrl.sv"
# (name, file, old, new, what it breaks)
MUTANTS = [
    ("E_len_no_4k", E, "lim = z ? ({1'b0, end_lo} - {1'b0, lo}) : (7'd64 - {1'b0, lo});",
     "lim = z ? ({1'b0, end_lo} - {1'b0, lo}) : 7'd64;", "bursts cross 4 KiB"),
    ("E_len_no_end", E, "lim = z ? ({1'b0, end_lo} - {1'b0, lo}) : (7'd64 - {1'b0, lo});",
     "lim = (7'd64 - {1'b0, lo});", "bursts run past the region end"),
    ("E_len_4k_off1", E, "lim = z ? ({1'b0, end_lo} - {1'b0, lo}) : (7'd64 - {1'b0, lo});",
     "lim = z ? ({1'b0, end_lo} - {1'b0, lo}) : (7'd63 - {1'b0, lo});", "4 KiB limit one beat short"),
    ("E_min_swapped", E, "return (b < lim) ? b : lim;", "return (b < lim) ? lim : b;", "max instead of min"),
    ("E_ar_id_nomask", E, "ar_id_q   <= rg_idc_q & id_mask;", "ar_id_q   <= rg_idc_q;", "read IDs ignore id_log2"),
    ("E_aw_id_nomask", E, "aw_id_q   <= wg_idc_q & id_mask;", "aw_id_q   <= wg_idc_q;", "write IDs ignore id_log2"),
    ("E_idmask", E, "4'((5'd1 << (c_idl > 3'd4 ? 3'd4 : c_idl)) - 5'd1)", "4'((5'd1 << (c_idl > 3'd4 ? 3'd4 : c_idl)))",
     "ID mask = 2^id_log2"),
    ("E_rd_mo_le", E, "& (rd_out_q < c_mo);", "& (rd_out_q <= c_mo);", "one read burst more in flight"),
    ("E_wr_mo_le", E, "& (wr_out_q < c_mo);", "& (wr_out_q <= c_mo);", "one write burst more in flight"),
    ("E_rd_out_beat", E, "rd_out_q <= rd_out_q + 7'(ar_ld) - 7'(r_end);", "rd_out_q <= rd_out_q + 7'(ar_ld) - 7'(r_beat);",
     "in-flight count decremented per beat"),
    ("E_blk_carry", E, "rg_blk_q <= rg_blk_q + HW'(1);", "rg_blk_q <= rg_blk_q;", "read address does not leave the 4 KiB block"),
    ("E_z_early", E, "rg_z_q   <= (rg_nb_q == (HW+1)'(1));", "rg_z_q   <= (rg_nb_q == (HW+1)'(2));", "end block detected one block early"),
    ("E_wlast", E, "dlast_q <= (dg_len == 7'd1);", "dlast_q <= (dg_len == 7'd2);", "WLAST one beat early"),
    ("E_w_no_ahead", E, "(~dg_new | (ahead_q != 7'd0))", "1'b1", "W beats before their AW (legal AXI: may be equivalent)"),
    ("E_dg_carry", E, "if (dg_lo_q == 6'd63) begin", "if (dg_lo_q == 6'd62) begin", "W block carry one beat early"),
    ("E_pat_addr", E, "dmix_q  <= mix64({{(58 - AW){1'b0}}, dg_blk_q, dg_lo_q, 6'd0});",
     "dmix_q  <= mix64({{(58 - AW){1'b0}}, dg_blk_q, dg_lo_q + 6'd1, 6'd0});", "pattern of the next beat"),
    ("E_pat_salt_ch", E, "salt_mix_q <= mix64({seed_q, 24'd0, 8'(CH)});", "salt_mix_q <= mix64({seed_q, 32'd0});",
     "pattern ignores the channel"),
    ("E_pat_word", E, "dmix_q ^ mix64(64'(8 * i)) ^ salt_mix_q;", "dmix_q ^ mix64(64'(8 * (7 - i))) ^ salt_mix_q;",
     "words of a beat reversed"),
    ("E_mix_shift", E, "z = y ^ (y >> 7);", "z = y ^ (y >> 6);", "mix64 differs from the reference"),
    ("E_cs_rot", E, "rotl64(rd_rsp_data[128*j + 64 +: 64], 16 * j + 8)", "rotl64(rd_rsp_data[128*j + 64 +: 64], 16 * j + 4)",
     "checksum rotation"),
    ("E_cs_tree", E, "cs2_q[1] <= cs1_q[2] + cs1_q[3];", "cs2_q[1] <= cs1_q[2] + cs1_q[2];", "checksum adder tree"),
    ("E_cs_nodrain", E, "if (rd_done && wr_done && !cs_busy && !dv_q) st <= S_FIN;",
     "if (rd_done && wr_done && !dv_q) st <= S_FIN;", "done before the checksum pipeline drained"),
    ("E_done_no_wr", E, "if (rd_done && wr_done && !cs_busy && !dv_q) st <= S_FIN;",
     "if (rd_done && !cs_busy && !dv_q) st <= S_FIN;", "done without waiting for the writes"),
    ("E_rd_beats_last", E, "rd_beats_q <= rd_beats_q + NBW'(1);", "rd_beats_q <= rd_beats_q + NBW'(rd_rsp_last);",
     "read bytes count bursts"),
    ("E_wr_beats_stall", E, "if (dv_q && wr_dat_ready) wr_beats_q", "if (dv_q) wr_beats_q", "write bytes count stalled cycles"),
    ("E_rd_cyc_off1", E, "rd_cyc_q   <= cnt1_q;", "rd_cyc_q   <= cnt1_q - CNT_W'(1);", "read cycles one short"),
    ("E_rd_lat_every", E, "if (!rd_first_q) begin\n          rd_first_q <= 1'b1;\n          rd_lat_q",
     "begin\n          rd_first_q <= 1'b1;\n          rd_lat_q", "first-beat latency overwritten by every beat"),
    ("E_wr_cyc_none", E, "wr_cyc_q    <= cnt1_q;", "wr_cyc_q    <= wr_cyc_q;", "write cycles never recorded"),
    ("E_rresp_off", E, "if (ev_rresp_err) begin", "if (1'b0) begin", "RRESP errors not counted"),
    ("E_bresp_off", E, "if (ev_bresp_err) begin", "if (1'b0) begin", "BRESP errors not counted"),
    ("E_unexp_off", E, "if (ev_unexpected) unexp_q <= 1'b1;", "if (1'b0) unexp_q <= 1'b1;", "unexpected responses not flagged"),
    ("E_cfg_bmax", E, "| (c_b == 7'd0) | (c_b > 7'd64)", "| (c_b == 7'd0)", "burst > 64 accepted"),
    ("E_cfg_mo0", E, "| (c_mo == 7'd0) | (c_mo > 7'd64)", "| (c_mo > 7'd64)", "outstanding 0 accepted"),
    ("E_cfg_idl", E, "| (c_idl > 3'd4)", "", "id_log2 > 4 accepted"),
    ("E_cfg_base", E, "| (base_q[11:0] != 12'd0)", "", "misaligned base accepted"),
    ("E_cfg_bytes64", E, "| (bytes_q[5:0] != 6'd0)", "", "bytes % 64 accepted"),
    ("E_cfg_lim_ge", E, "(c_rd & (rd_end_q > LIM))", "(c_rd & (rd_end_q >= LIM))", "a region ending at the limit refused"),
    ("E_cfg_wr_lim", E, "(c_wr & (wr_end_q > LIM))", "(c_wr & 1'b0)", "write region beyond the limit accepted"),
    ("E_rend_disabled", E, "rend = (c_en & c_rd & ~err_q & nz_q) ? rd_end_q[AW+6:6] : {1'b0, ra0};",
     "rend = (c_en & ~err_q & nz_q) ? rd_end_q[AW+6:6] : {1'b0, ra0};", "reads on a write-only channel"),
    ("E_rend_bytes0", E, "rend = (c_en & c_rd & ~err_q & nz_q) ? rd_end_q[AW+6:6] : {1'b0, ra0};",
     "rend = (c_en & c_rd & ~err_q) ? rd_end_q[AW+6:6] : {1'b0, ra0};",
     "bytes 0 with the read offset at the limit walks the whole range (review finding)"),
    ("E_wend_bytes0", E, "wend = (c_en & c_wr & ~err_q & nz_q) ? wr_end_q[AW+6:6] : {1'b0, wa0};",
     "wend = (c_en & c_wr & ~err_q) ? wr_end_q[AW+6:6] : {1'b0, wa0};",
     "bytes 0 with the write offset at the limit walks the whole range (review finding)"),
    ("E_status_ran", E, "ran_q     <= c_en & ~err_q;", "ran_q     <= ~err_q;", "RAN set on a disabled channel"),
    ("E_res_csum", E, "4'd10:   res_rdata <= csum_q[31:0];", "4'd10:   res_rdata <= csum_q[63:32];", "checksum low word"),
    ("E_res_bytes", E, "wire [63:0] rd_byt64 = 64'(rd_beats_q) << 6", "wire [63:0] rd_byt64 = 64'(rd_beats_q) << 5",
     "read bytes scaled"),
    ("C_no_cor", C, "if (ctrl_rd) begin", "if (1'b0) begin", "ap_done / ap_ready not clear on read"),
    ("C_idle_stays", C, "ap_idle_q <= 1'b0;\n            go_q", "go_q", "ap_idle not dropped at the start"),
    ("C_done_ch0", C, "wire all_done = (done_in_q == {NUM_CH{go_q}});", "wire all_done = (done_in_q[0] == go_q);",
     "completion when channel 0 is done"),
    ("C_isr_no_ier", C, "if (ier_q) isr_q <= 1'b1;", "isr_q <= 1'b1;", "ISR set without IER"),
    ("C_irq_no_gie", C, "irq_q      <= gie_q & ier_q & isr_q;", "irq_q      <= ier_q & isr_q;", "interrupt without GIE"),
    ("C_res_sel", C, "res_sel <= ax_off[5:2];", "res_sel <= ax_off[5:2] + 4'd1;", "result word select"),
    ("C_rch", C, "rch_q   <= 2'(ax_off[7:6]);", "rch_q   <= 2'(ax_off[7:6] + 2'd1);", "result channel select"),
    ("C_wstrb", C, "r[8*b +: 8] = be[b] ? nw[8*b +: 8] : old[8*b +: 8];", "r[8*b +: 8] = nw[8*b +: 8];", "byte strobes ignored"),
    ("C_bytes_hi", C, "if (ax_off == A_CH0 + 12'(32 * c + 12))   bytes_q[c][63:32]",
     "if (ax_off == A_CH0 + 12'(32 * c + 16))   bytes_q[c][63:32]", "bytes hi word decoded at rd_off lo"),
    ("C_snapshot_live", C, "assign ch_ctrl   = ctl_s;", "assign ch_ctrl   = ctl_q;",
     "ctrl word not snapshotted (engines latch it CTRL_PIPE cycles after the start: expected equivalent in practice)"),
    ("C_tot_stop", C, "tot_q <= tot_q + 64'd1;", "tot_q <= tot_q + 64'd0;", "TOTAL_CYCLES not counted"),
    ("C_seed_snap", C, "seed_s  <= seed_q;", "seed_s  <= 32'd0;", "seed lost"),
    ("C_base_off", C, "assign ch_base   = buf_s;", "assign ch_base   = buf_s + 64'h1000;", "buffer address + 4 KiB"),
]
TESTS = ["--test", "regmap,single,all4,combos,ideal,cfgerr,errors,apctrl,reset", "--regime", "ideal,rand,slow", "--runs", "15",
         "--stop-on-fail"]


def make_copy(work: Path, name: str, mut) -> Path:
    d = work / ("m_" + name) / "bw"
    if d.parent.exists():
        shutil.rmtree(d.parent)
    d.mkdir(parents=True)
    for f in RTL_BW.iterdir():
        if f.suffix in (".sv", ".v", ".h"):
            shutil.copy2(f, d / f.name)
    # filelist with the shared sources as absolute paths
    lines = []
    for line in (RTL_BW / "filelist.f").read_text().splitlines():
        t = line.split("//", 1)[0].strip()
        if not t:
            lines.append(line)
        elif t.startswith(".."):
            lines.append(str((RTL_BW / t).resolve()))
        else:
            lines.append(t)
    (d / "filelist.f").write_text("\n".join(lines) + "\n")
    if mut is not None:
        _, f, old, new, _ = mut
        src = (d / f).read_text()
        n = src.count(old)
        if n != 1:
            raise SystemExit("mutant %s: pattern found %d times in %s" % (name, n, f))
        (d / f).write_text(src.replace(old, new))
    return d


def run_one(work: Path, name: str, mut, seed: int, params: dict | None = None) -> dict:
    t0 = time.time()
    d = make_copy(work, name, mut)
    log = []
    try:
        binary = rb.build("body", params or {}, jobs=2, filelist=d / "filelist.f", wno_fatal=True, log=log.append,
                          root=d.parent / "build")
    except (RuntimeError, SystemExit) as e:
        msg = str(e).splitlines()
        first = next((l for l in msg if "Error" in l or "error" in l), msg[-1] if msg else "build failed")
        return {"killed": True, "first": "build: " + first.strip()[:200], "seconds": round(time.time() - t0, 1)}
    r = subprocess.run([str(binary), "--seed", str(seed)] + TESTS, capture_output=True, text=True, timeout=3600)
    lines = r.stdout.splitlines()
    first = next((l.strip() for l in lines if "(stop on fail)" in l), "")
    for i, l in enumerate(lines if not first else []):
        if l.startswith("FAIL"):
            first = l.strip()
            if i + 1 < len(lines) and lines[i + 1].startswith("      ") and not lines[i + 1].strip().startswith("ch"):
                first += " | " + lines[i + 1].strip()
            elif i + 2 < len(lines) and lines[i + 2].startswith("      "):
                first += " | " + lines[i + 2].strip()
            break
    out = {"killed": r.returncode != 0, "first": first[:300], "seconds": round(time.time() - t0, 1)}
    shutil.rmtree(d.parent / "build", ignore_errors=True)
    return out


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True)
    ap.add_argument("--only", default="")
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--param", action="append", default=[], help="K=V: kernel parameter of every build")
    a = ap.parse_args(argv)
    params = {k: int(v, 0) for k, v in (kv.split("=", 1) for kv in a.param)}
    work = Path(a.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    sel = [m for m in MUTANTS if not a.only or m[0] in a.only.split(",")]
    res_file = work / "results.json"
    results = json.loads(res_file.read_text()) if res_file.is_file() else {}
    base = run_one(work, "NONE", None, a.seed, params)
    results["NONE"] = base
    print("NONE: %s (%.0f s)" % ("FAILS (bench or RTL broken)" if base["killed"] else "passes", base["seconds"]), flush=True)
    if base["killed"]:
        print("  " + base["first"])
        res_file.write_text(json.dumps(results, indent=1))
        return 1
    with cf.ThreadPoolExecutor(max_workers=a.procs) as ex:
        futs = {ex.submit(run_one, work, m[0], m, a.seed, params): m for m in sel}
        for f in cf.as_completed(futs):
            m = futs[f]
            r = f.result()
            r["what"] = m[4]
            results[m[0]] = r
            print("%-18s %s  %s" % (m[0], "killed  " if r["killed"] else "SURVIVED", r["first"][:150]), flush=True)
            res_file.write_text(json.dumps(results, indent=1))
    killed = sum(1 for m in sel if results[m[0]]["killed"])
    print("%d mutants: %d killed, %d survived: %s" % (len(sel), killed, len(sel) - killed,
                                                       ", ".join(m[0] for m in sel if not results[m[0]]["killed"])))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
