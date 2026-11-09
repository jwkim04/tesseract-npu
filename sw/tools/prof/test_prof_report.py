#!/usr/bin/env python3
"""test_prof_report.py: the plan dump (TSR_DUMP_PLAN) and the report generator on a tiny model on the simulator.

  - llama-completion with TSR_DUMP_PLAN writes the plan, launch, trace and meta files, and its output text is the same as without it
  - prof_report.py makes report.html, trace.perfetto.json and summary.json from them
  - the record cycles add up to the launch span, the join found every record, the HTML follows the page rules (a title of two to four
    words, tokens on :root with a dark set under both selectors, no external script or stylesheet, under 16 MB), the Perfetto file
    has a slice id on every slice and a flow arrow from a record to each of its ops

Standard library only. Exit 0 on success."""
import argparse
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FAILS = []


def check(ok, text):
    print(("ok: " if ok else "FAIL: ") + text)
    if not ok:
        FAILS.append(text)


def run_llama(build, model, out, dump, fw):
    env = dict(os.environ)
    env.update({"TSR_IMAGE": fw, "LLAMA_ATTN_ROT_DISABLE": "1", "TSR_DEVICE": "sim", "TSR_SIM_RACE": "fatal"})
    env.pop("TSR_DUMP_PLAN", None)
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
    summ = json.load(open(os.path.join(outdir, "summary.json")))
    check(len(html) < 16 * 1024 * 1024, "report.html is %.1f MB (under 16 MB)" % (len(html) / 1e6))
    m = re.search(r"<title>(.*?)</title>", html)
    check(m is not None and 2 <= len(m.group(1).split()) <= 4, "title is two to four words: %r" % (m.group(1) if m else None))
    css = re.search(r"<style>(.*?)</style>", html, re.S).group(1)
    check("@media (prefers-color-scheme: dark)" in css and ':root:not([data-theme="light"])' in css and ':root[data-theme="dark"]' in css,
          "dark colours under the media query guarded by :root:not([data-theme=\"light\"]) and under :root[data-theme=\"dark\"]")
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
    check(len(data["records"]) == len(recs) and len(data["ops"]["card"]) > 100, "page data: %d records, %d ops" % (len(data["records"]), len(data["ops"]["card"])))
    ids = re.findall(r'\bid="([^"]+)"', html)
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
        for a, b in lst:
            while stack and stack[-1] <= a + 1e-9:
                stack.pop()
            if stack and b > stack[-1] + 1e-6:
                bad += 1
            stack.append(b)
    check(bad == 0, "Perfetto: slices of a track never overlap without nesting (%d tracks)" % len(per))
    sl = [e for e in xs if e["args"].get("kind") == "MUL_MAT" and "ideal_cycles" in e["args"]]
    check(len(sl) > 0 and all("efficiency" in e["args"] for e in sl if e["args"].get("cycles")), "Perfetto: MVM slices carry the ideal cycles and the efficiency")
    if FAILS:
        print("%d check(s) failed" % len(FAILS))
        return 1
    print("test_prof_report: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
