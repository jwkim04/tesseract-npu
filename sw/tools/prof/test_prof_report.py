#!/usr/bin/env python3
"""test_prof_report.py: the plan dump (TSR_DUMP_PLAN) and the report generator on a tiny model on the simulator.

  - llama-completion with TSR_DUMP_PLAN writes the plan, launch, trace and meta files, and its output text is the same as without it
  - prof_report.py makes report.html, trace.perfetto.json and summary.json from them
  - the record cycles add up to the launch span, the join found every record, the HTML follows the page rules (a title of two to four
    words, tokens on :root with a dark set under both selectors, no external script or stylesheet, under 16 MB), the Perfetto file
    has a slice id on every slice and a flow arrow from a record to each of its ops
  - the ideal MVM cycles follow the weight streams of the core (docs/prof-dbg.md §3.1): runs of the card image's simulator with the dual
    weight stream (TSR_SIM_CARD=k4dual) and with TSR_MVM_DUAL=0, and of the plain simulator, against the rule, the simulated run time of
    every MVM op, meta.json's mvm_dual and the E and F of the utilization split

Standard library only. Exit 0 on success."""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FAILS = []


def check(ok, text):
    print(("ok: " if ok else "FAIL: ") + text)
    if not ok:
        FAILS.append(text)


def run_llama(build, model, out, dump, fw, card=None):
    """One decode run on the simulator: the plain one, or `card` (environment variables) for the card image's configuration."""
    env = dict(os.environ)
    for k in ("TSR_DUMP_PLAN", "TSR_SIM_CARD", "TSR_SIM_MVM_DUAL", "TSR_MVM_DUAL", "TSR_SIM_MEM_RATE"):
        env.pop(k, None)
    env.update({"TSR_IMAGE": fw, "LLAMA_ATTN_ROT_DISABLE": "1", "TSR_DEVICE": "sim", "TSR_SIM_RACE": "fatal"})
    env.update(card or {})
    if dump:
        env["TSR_DUMP_PLAN"] = out
    cmd = [os.path.join(build, "bin", "llama-completion"), "-m", model, "-ngl", "99", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "-c", "512",
           "--temp", "0", "-no-cnv", "-p", "The history of the printing press begins", "-n", "5", "-t", "4"]
    # llama.cpp prints through a log thread that the process does not wait for at exit: on a loaded machine (ctest -j8) a run
    # can end before the last token and the timing report are written (also with the CPU backend alone, TSR_MON=0, no
    # device). Such a run is incomplete, not different, so a run whose stderr has no timing report is repeated.
    for _ in range(6):
        p = subprocess.run(cmd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600)
        err = p.stderr.decode(errors="replace")
        if p.returncode != 0 or "total time" in err:
            break
    return p.returncode, p.stdout.decode(errors="replace"), err


CAPS_MVM_DUAL = 1 << 16        # CAPS bit 16: a cluster reads two weight streams for an MVM of T <= T_MAX / 2 (docs/rtl-mvm.md §15)


def fresh_dir(path):
    """An empty directory: the plan files of an earlier dump would be read as this run's."""
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path)
    return path


def plan_files(dump):
    return [os.path.join(dump, f) for f in sorted(os.listdir(dump)) if re.match(r"plan_\d+\.json$", f)]


def load_dump(dump):
    """meta.json of a dump and its weight-MVM records from every plan: (meta, [(kind name, argument words by name, record)])."""
    meta = json.load(open(os.path.join(dump, "meta.json")))
    kinds = {int(k): (v["name"], [x.split(":")[0] for x in v["args"]]) for k, v in meta["kinds"].items()}
    recs = []
    for f in plan_files(dump):
        for lst in json.load(open(f))["lists"]:
            for r in lst["recs"]:
                name, spec = kinds[r["kind"]]
                if name.startswith("MUL_MAT") and r["w_bytes"]:
                    recs.append((name, dict(zip(spec, r["a"])), r))
    return meta, recs


def pass_tokens(name, arg, r):
    """The tokens of each MVM of a record in issue order. MUL_MAT: one MVM per T-tile of the firmware's plan (SW §7.4), the last tile with
    the rest of the tokens. MUL_MAT_ID: one MVM of one token per token and used expert (SW §7.5)."""
    if name == "MUL_MAT_ID":
        return [1] * (arg["T"] * arg["N_USED"])
    return [r["tile_rows"]] * (r["tiles"] - 1) + [arg["T"] - r["tile_rows"] * (r["tiles"] - 1)]


def check_ideal(tag, dump, dual):
    """The ideal cycles of every weight-MVM record of a dump against the rule: an MVM streams the weights once, and the slowest channel
    has ceil(N / num_ch) rows of ROW_BEATS beats, one beat per cycle and stream. With two streams (`dual`, T <= T_MAX / 2) it reads half
    the rows, rounded up. ROW_BEATS comes from the weight bytes, which do not depend on the streams."""
    meta, recs = load_dump(dump)
    half = meta["t_max"] // 2
    bad, n_two = [], 0
    for name, arg, r in recs:
        toks = pass_tokens(name, arg, r)
        rows = -(-arg["N"] // meta["num_ch"])
        one = r["w_bytes"] // (len(toks) * meta["num_ch"] * 64)          # cycles of one MVM on one stream: rows x ROW_BEATS
        want = [-(-rows // 2) * (one // rows) if dual and t <= half else one for t in toks]
        got = [c for n, c in r.get("mvm_passes", []) for _ in range(n)]
        n_two += sum(1 for t in toks if dual and t <= half)
        if r["ideal"] != sum(want) or got != want:
            bad.append("%s %s: ideal %s, per MVM %s, want %s" % (name, r["name"], r["ideal"], got[:3], want[:3]))
    rule = "two-stream rule (%d MVMs with T <= %d on two streams)" % (n_two, half) if dual else "one-stream rule: the weight bytes over num_ch x 64 B"
    check(recs and not bad and (n_two > 0 or not dual),
          "%s: the ideal of %d MVM records follows the %s%s" % (tag, len(recs), rule, "; failing: " + "; ".join(bad[:3]) if bad else ""))


def report_of(dump, out, launch=None):
    """prof_report.py report of a dump (its decode token, or `launch`): (summary.json, the MVM op slices of the Perfetto file by id)."""
    cmd = [sys.executable, os.path.join(HERE, "prof_report.py"), "report", "--card", dump, "--out", out, "--model-name", "tiny"]
    if launch is not None:
        cmd += ["--launch", str(launch)]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600)
    check(p.returncode == 0, "prof_report.py report of %s%s exits 0" % (os.path.basename(dump), "" if launch is None else ", launch %d" % launch))
    if p.returncode != 0:
        print(p.stdout.decode(), p.stderr.decode())
        return {"summary": {}}, {}
    summ = json.load(open(os.path.join(out, "summary.json")))
    ev = json.load(open(os.path.join(out, "trace.perfetto.json")))["traceEvents"]
    return summ, {e["args"]["id"]: e["args"] for e in ev if e["ph"] == "X" and e["args"].get("op") == "MVM"}


def check_ops(what, two, one, every_op_on_two):
    """The MVM op slices of one launch on two streams (`two`) and on one (`one`), by slice id: no op's ideal is above its simulated run
    time, and the run time of an op falls by what its ideal falls (the fill and the broadcast do not depend on the streams)."""
    ids = [k for k in one if one[k].get("ideal_cycles")] if two.keys() == one.keys() else []
    check(ids, "%s: %d MVM ops with an ideal, the same ops on one and on two streams" % (what, len(ids)))
    over = [k for ops in (two, one) for k in ids if ops[k].get("ideal_cycles", 0) > ops[k]["cycles"]]
    check(not over, "%s: no MVM op's ideal exceeds its simulated run time (%d ops on each)" % (what, len(ids)))
    bad = [k for k in ids if abs((one[k]["cycles"] - two[k]["cycles"]) - (one[k]["ideal_cycles"] - two[k].get("ideal_cycles", 0))) > 0.01]
    check(not bad, "%s: the simulated run time of an MVM op falls by the fall of its ideal on two streams (%d ops differ)" % (what, len(bad)))
    n_two = sum(1 for k in ids if two[k].get("ideal_cycles", 0) < one[k]["ideal_cycles"])
    check(n_two == len(ids) if every_op_on_two else 0 < n_two < len(ids),
          "%s: %d of %d MVM ops are on two streams%s" % (what, n_two, len(ids), "" if every_op_on_two else ", the ops of more than T_MAX / 2 tokens stay on one"))
    wb = [sum(x.get("bytes_read", 0) for x in ops.values()) for ops in (two, one)]
    check(wb[0] == wb[1] > 0, "%s: the weight bytes of the MVM ops are the same on one and on two streams (%d B)" % (what, wb[0]))


def dual_checks(a, plain, plain_summ):
    """The ideal follows the weight streams of the core. The plain simulator has one stream. The card image's simulator (TSR_SIM_CARD=k4dual,
    CAPS bit 16, T_MAX 4) has two for an MVM of T <= T_MAX / 2, and one with TSR_MVM_DUAL=0, which changes the cycles and nothing else."""
    meta, _ = load_dump(plain)
    check(meta.get("mvm_dual") == 0 and not meta["caps"] & CAPS_MVM_DUAL, "plain simulator: CAPS bit 16 clear, meta.json mvm_dual 0")
    check_ideal("plain simulator", plain, False)
    dumps, texts = {}, {}
    for dual, tag, card in ((True, "two streams", {"TSR_SIM_CARD": "k4dual"}), (False, "TSR_MVM_DUAL=0", {"TSR_SIM_CARD": "k4dual", "TSR_MVM_DUAL": "0"})):
        dumps[dual] = fresh_dir(os.path.join(a.out_dir, "dual" if dual else "single"))
        rc, texts[dual], _ = run_llama(a.build_dir, a.model, dumps[dual], True, a.fw, card)
        meta, _ = load_dump(dumps[dual])
        check(rc == 0 and meta["caps"] & CAPS_MVM_DUAL and meta["t_max"] >= 2 and meta.get("mvm_dual") == int(dual),
              "%s: llama-completion exits 0 (%d), CAPS bit 16 set, T_MAX %d, meta.json mvm_dual %s" % (tag, rc, meta["t_max"], meta.get("mvm_dual")))
        check_ideal(tag, dumps[dual], dual)
    check(texts[True] == texts[False] and len(texts[True]) > 10, "the generated text is the same on one and on two weight streams")
    # the decode token, and the prompt launch (launch 1: tiles of T_MAX tokens and a shorter last one)
    sums = {}
    for launch, what in ((None, "decode token"), (1, "prompt")):
        (sum2, ops2), (sum1, ops1) = [report_of(dumps[v], os.path.join(a.out_dir, "report_%s_%s" % (what.split()[0], "dual" if v else "single")), launch) for v in (True, False)]
        check_ops(what, ops2, ops1, launch is None)
        sums[what] = (sum2["summary"], sum1["summary"])
    s2, s1 = sums["decode token"]
    check(s2.get("F", -1) >= 0 and s1.get("F", -1) >= 0,
          "decode token: the MVM cycles are above the ideal on two streams and on one (F %.3f, %.3f)" % (s2.get("F", -1), s1.get("F", -1)))
    check(s2.get("mvm_dual") == 1 and s1.get("mvm_dual") == 0 and s2.get("t_max") == s1.get("t_max") == meta["t_max"],
          "decode token: summary.json names the ideal's streams (mvm_dual %s and %s, T_MAX %s)" % (s2.get("mvm_dual"), s1.get("mvm_dual"), s2.get("t_max")))
    e1 = plain_summ["summary"]["E_clock"]
    check(abs(s1.get("E_clock", 0) - e1) < 1e-9 and e1 < s2.get("E_clock", 0) <= 2 * e1 + 1e-9,
          "decode token: the stream rate ceiling is %.3f on one stream, as on the plain simulator (%.3f), and %.3f on two" % (s1.get("E_clock", 0), e1, s2.get("E_clock", 0)))
    # E is the DDR read rate over the ideal MVM time (docs/prof-dbg.md §3.1): the stream rate ceiling times the bytes read per weight byte. On two
    # streams the ceiling is above the tsr_bw bandwidth, so E is above 100 % and the MVM time above the ideal (F) holds what the DDR cannot follow.
    check(all(abs(s.get("E", -1) - s.get("E_clock", 0) * s.get("rd_meas", 0) / max(s.get("w_bytes", 0), 1)) < 1e-6 for s in (s2, s1)) and s2.get("E", 0) > 1,
          "decode token: E is the stream rate ceiling times the bytes read per weight byte, above 100 %% on two streams (E %.3f on two, %.3f on one)" % (s2.get("E", 0), s1.get("E", 0)))
    # an older dump (no mvm_dual in meta.json, no mvm_passes in the plans) loads as one stream
    old = fresh_dir(os.path.join(a.out_dir, "older"))
    shutil.copytree(plain, old, dirs_exist_ok=True)
    m = json.load(open(os.path.join(old, "meta.json")))
    m.pop("mvm_dual", None)
    json.dump(m, open(os.path.join(old, "meta.json"), "w"))
    for f in plan_files(old):
        pl = json.load(open(f))
        for lst in pl["lists"]:
            for r in lst["recs"]:
                r.pop("mvm_passes", None)
        json.dump(pl, open(f, "w"))
    so = report_of(old, os.path.join(a.out_dir, "report_older"))[0]["summary"]
    check(so.get("mvm_dual") == 0 and so.get("ideal_cycles") == plain_summ["summary"]["ideal_cycles"],
          "an older dump gives the same report on one stream (mvm_dual %s, ideal %s cycles)" % (so.get("mvm_dual"), so.get("ideal_cycles")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--fw", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    dump = os.path.join(a.out_dir, "dump")
    os.makedirs(dump, exist_ok=True)
    for f in os.listdir(dump):
        os.remove(os.path.join(dump, f))

    rc0, out0, _ = run_llama(a.build_dir, a.model, dump, False, a.fw)
    rc1, out1, err1 = run_llama(a.build_dir, a.model, dump, True, a.fw)
    check(rc0 == 0 and rc1 == 0, "llama-completion exits 0 with and without TSR_DUMP_PLAN (%d, %d)" % (rc0, rc1))
    check(out0 == out1 and len(out0) > 10, "the generated text is the same with the dump on and off (%d bytes)" % len(out0))
    for f in ("launches.jsonl", "meta.json", "trw.raw", "cpu_fallback.json"):
        check(os.path.exists(os.path.join(dump, f)) and os.path.getsize(os.path.join(dump, f)) > 0, "%s written" % f)
    plans = [f for f in os.listdir(dump) if re.match(r"plan_\d+\.json$", f)]
    check(len(plans) >= 2, "plan files: %d" % len(plans))
    launches = [json.loads(l) for l in open(os.path.join(dump, "launches.jsonl")) if l.strip()]
    check(len(launches) >= 4 and all(l.get("perf") for l in launches), "%d launches, each with PERF deltas" % len(launches))
    plan = json.load(open(os.path.join(dump, "plan_%d.json" % launches[-1]["plan"])))
    recs = plan["lists"][0]["recs"]
    check(plan["lists"][0]["n_ops"] == len(recs) and all(r["a"] is not None for r in recs), "the decode plan lists %d records with argument words" % len(recs))
    check(any(n["place"] == "absorbed" for n in plan["nodes"]) and len(plan["groups"]) > 0, "nodes carry a placement; %d fusion groups are listed" % len(plan["groups"]))
    check(plan["lists"][0]["end"], "the list's end reason is recorded: %s" % plan["lists"][0]["end"])

    # the report
    outdir = os.path.join(a.out_dir, "report")
    p = subprocess.run([sys.executable, os.path.join(HERE, "prof_report.py"), "report", "--card", dump, "--sim", dump, "--out", outdir,
                        "--model-name", "tiny"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600)
    print(p.stdout.decode(), p.stderr.decode())
    check(p.returncode == 0, "prof_report.py report exits 0")
    html = open(os.path.join(outdir, "report.html")).read()
    outdir2 = os.path.join(a.out_dir, "report2")
    p2 = subprocess.run([sys.executable, os.path.join(HERE, "prof_report.py"), "report", "--card", dump, "--sim", dump, "--out", outdir2,
                         "--model-name", "tiny"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600)
    check(p2.returncode == 0 and open(os.path.join(outdir2, "report.html"), "rb").read() == open(os.path.join(outdir, "report.html"), "rb").read(),
          "the same inputs give a byte-identical report.html")
    summ = json.load(open(os.path.join(outdir, "summary.json")))
    check(len(html) < 16 * 1024 * 1024, "report.html is %.1f MB (under 16 MB)" % (len(html) / 1e6))
    m = re.search(r"<title>(.*?)</title>", html)
    check(m is not None and 2 <= len(m.group(1).split()) <= 4, "title is two to four words: %r" % (m.group(1) if m else None))
    css = re.search(r"<style>(.*?)</style>", html, re.S).group(1)
    check("@media (prefers-color-scheme: dark)" in css and ':root:not([data-theme="light"])' in css and ':root[data-theme="dark"]' in css,
          "dark colors under the media query guarded by :root:not([data-theme=\"light\"]) and under :root[data-theme=\"dark\"]")
    check(re.search(r"body\s*\{[^}]*background:", css) is not None, "body has an explicit background")
    check(not re.search(r"<script[^>]*\ssrc=|<link[^>]*stylesheet|@import|url\(\s*['\"]?https?:", html), "no external script, stylesheet or import")
    js = re.search(r"<script>(.*?)</script>", html, re.S).group(1)
    ls_ok = True
    lines = js.split("\n")
    for i, l in enumerate(lines):
        if "localStorage" in l and "try" not in l and "try" not in lines[i - 1]:
            ls_ok = False
    check(ls_ok, "every localStorage access is inside try")
    data = json.loads(re.search(r'<script type="application/json" id="data">(.*?)</script>', html, re.S).group(1).replace("<\\/", "</"))
    rows0 = data["tl"]["runs"][0]["rows"]
    n_ops = sum(len(r["ops"]) for r in rows0)
    check(len(data["records"]) == len(recs) and n_ops > 100, "page data: %d records, %d timeline slices on the card" % (len(data["records"]), n_ops))
    check(rows0[0]["unit"] == -1 and any(r["label"] == "MVM" for r in rows0) and any(r["label"] == "VU" for r in rows0),
          "timeline rows: the RISC-V issue band and a row for the MVM and the VU")
    check(all(('id="%s"' % i) in html for i in ("toolbar", "tree", "tl", "bottom", "status", "launch", "theme")) and html.count('role="tab"') == 4,
          "page regions: toolbar, tree, timeline, bottom pane with four tabs, status bar")
    rules = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    check(not re.search(r"box-shadow|gradient|@font-face", rules), "no shadows, gradients or web fonts in the style sheet")
    check(all(x <= 13 for x in map(int, re.findall(r"--fs[a-z-]*:\s*(\d+)px", rules))), "no type above 13 px")
    ids = re.findall(r'\sid="([^"]+)"', re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.S))
    check(len(ids) == len(set(ids)), "element ids are unique (%d)" % len(ids))
    check("viewport" in html and "overflow-x: auto" in css, "viewport meta and scrolling boxes for wide content")
    bad = [c["text"] for c in summ["checks"] if c["ok"] is False]
    check(not bad, "the report's own checks pass (%d checks)%s" % (len(summ["checks"]), ("; failing: " + "; ".join(bad)) if bad else ""))
    s = summ["summary"]
    check(s["cycles"] > 0 and abs(s["span"] - s["cycles"]) < 0.01 * s["cycles"], "launch span %d against PERF_CYCLES %d" % (s["span"], s["cycles"]))
    check(sum(k["cycles"] for k in summ["kinds"]) + summ["other_cycles"]["setup"] + summ["other_cycles"]["tail"] == s["span"],
          "per-kind cycles plus setup and tail add up to the launch span")
    tr = json.load(open(os.path.join(outdir, "trace.perfetto.json")))
    ev = tr["traceEvents"]
    xs = [e for e in ev if e["ph"] == "X"]
    check(len(xs) > 100 and all("id" in e["args"] for e in xs if e["tid"] not in (10,)), "Perfetto: %d slices, each with args.id" % len(xs))
    starts = {e["id"] for e in ev if e["ph"] == "s"}
    ends = {e["id"] for e in ev if e["ph"] == "f"}
    check(starts == ends and len(starts) > 50, "Perfetto: %d flow arrows record -> op, each with both ends" % len(starts))
    check(any(e["ph"] == "C" and e["name"].startswith("PERF ") for e in ev) and any(e["ph"] == "M" and e["name"] == "thread_name" and e["args"]["name"] == "MVM" for e in ev),
          "Perfetto: PERF counter tracks and a track per unit")
    bad = 0
    per = {}
    for e in xs:
        per.setdefault((e["pid"], e["tid"]), []).append((e["ts"], e["ts"] + e["dur"]))
    for key, lst in per.items():
        lst.sort(key=lambda x: (x[0], -x[1]))
        stack = []
        for t0, t1 in lst:
            while stack and stack[-1] <= t0 + 1e-9:
                stack.pop()
            if stack and t1 > stack[-1] + 1e-6:
                bad += 1
            stack.append(t1)
    check(bad == 0, "Perfetto: slices of a track never overlap without nesting (%d tracks)" % len(per))
    sl = [e for e in xs if e["args"].get("kind") == "MUL_MAT" and "ideal_cycles" in e["args"]]
    check(len(sl) > 0 and all("efficiency" in e["args"] for e in sl if e["args"].get("cycles")), "Perfetto: MVM slices carry the ideal cycles and the efficiency")
    dual_checks(a, dump, summ)
    if FAILS:
        print("%d check(s) failed" % len(FAILS))
        return 1
    print("test_prof_report: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
