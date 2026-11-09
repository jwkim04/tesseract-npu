#!/usr/bin/env python3
"""prof_report.py: the profile report of one decode token (docs/prof-dbg.md "The profile report").

  prof_report.py report --card DIR [--sim DIR] --out DIR [--launch N] [--model-name S] [--expect-cycles N] [--ref-toks X]
      reads a dump directory of the card (TSR_DUMP_PLAN, tools/prof/capture_card.sh) and, optionally, of the simulator
      (capture_sim.sh), and writes OUT/report.html (one self-contained page), OUT/trace.perfetto.json (the same slices for
      Perfetto / chrome://tracing) and OUT/summary.json (the numbers of the page).
  prof_report.py sched LOG OUT.json
      the scheduler's splits out of a GGML_SCHED_DEBUG=2 log (llama-completion -lv 5): which nodes the CPU runs.

Standard library only.
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prof_html as H      # noqa: E402
import prof_lib as P       # noqa: E402
import prof_model as M     # noqa: E402
import prof_perfetto as PF  # noqa: E402


def _size(s):
    m = re.match(r"([\d.]+)([KMG]?)", s)
    if not m:
        return 0.0
    return float(m.group(1)) * {"": 1, "K": 1e3, "M": 1e6, "G": 1e9}[m.group(2)]


def parse_sched(path):
    """The splits of the decode graph in a GGML_SCHED_DEBUG=2 log: [{idx, backend, n_nodes, inputs, node_names, nodes}]."""
    text = open(path, errors="replace").read()
    text = re.sub(r"\d+\.\d\d\.\d\d\d\.\d\d\d [DIWEI] ", "", text)
    tok = re.compile(r"## SPLIT #(\d+): (\S+) # (\d+) inputs|node # *(\d+) \( *([A-Z_0-9]+)\): *(.*?) \( *([\d.]+[KMG]?)\) \[ *([A-Za-z0-9_ ]*?) *\]|\[([^\]\[\n]*?) \( *([\d.]+[KMG]?)\)\]")
    graphs, cur, split = [], None, None
    for m in tok.finditer(text):
        if m.group(1) is not None:
            if int(m.group(1)) == 0:
                cur = []
                graphs.append(cur)
            if cur is None:
                continue
            split = {"idx": int(m.group(1)), "backend": m.group(2), "n_inputs": int(m.group(3)), "inputs": [], "input_sizes": [], "nodes": []}
            cur.append(split)
        elif m.group(4) is not None and split is not None:
            split["nodes"].append({"idx": int(m.group(4)), "op": m.group(5), "name": m.group(6).strip(), "backend": m.group(8)})
        elif m.group(9) is not None and split is not None and not split["nodes"]:
            split["inputs"].append(m.group(9).strip())
            split["input_sizes"].append(_size(m.group(10)))
    if not graphs:
        return None

    def key(g):
        for sp in g:
            if not sp["backend"].startswith("CPU") and sp["input_sizes"]:
                return sp["input_sizes"][0]
        return 1e18
    best = min(range(len(graphs)), key=lambda i: (key(graphs[i]), -i))
    out = []
    for sp in graphs[best]:
        out.append({"idx": sp["idx"], "backend": sp["backend"], "n_nodes": len(sp["nodes"]), "inputs": sp["inputs"],
                    "node_names": [n["name"] for n in sp["nodes"]],
                    "nodes": sp["nodes"] if sp["backend"].startswith("CPU") else []})
    return {"splits": out, "graphs_seen": len(graphs)}


def cmd_sched(a):
    s = parse_sched(a.log)
    if s is None:
        print("prof_report: no scheduler splits in %s" % a.log, file=sys.stderr)
        return 1
    with open(a.out, "w") as f:
        json.dump(s, f, separators=(",", ":"))
    print("sched: %d splits of the decode graph (%d graphs in the log) -> %s" % (len(s["splits"]), s["graphs_seen"], a.out))
    return 0


def cmd_report(a):
    card = P.Run(a.card)
    sim = P.Run(a.sim) if a.sim else None
    # a launch both runs have, in the middle of the decode tokens
    launch = a.launch
    if launch is None and sim is not None:
        cd = [k for k in card.decode_launches() if card.launch(k)[1] is not None]
        sd = {k for k in sim.decode_launches() if sim.launch(k)[1] is not None}
        both = [k for k in cd if k in sd]
        if both:
            launch = both[len(both) // 2]
    model = M.build_model(card, sim, launch=launch, expect_cycles=a.expect_cycles, ref_toks=a.ref_toks, model_name=a.model_name, ref_run=a.ref_run)
    os.makedirs(a.out, exist_ok=True)
    cmds = a.commands or ""
    n = H.build_html(model, os.path.join(a.out, "report.html"), {"commands": cmds})
    views = []
    cv = M.LaunchView(card, model["launch"], "card")
    views.append((cv, 1, "Tesseract card (%s, %.0f MHz)" % (model["run"]["card"]["image_id"], (cv.f or 0) / 1e6)))
    if sim is not None and model.get("sim") is not None:
        sv = M.LaunchView(sim, model["launch"], "sim")
        views.append((sv, 2, "Tesseract simulator (card configuration)"))
    nev = PF.write_perfetto(os.path.join(a.out, "trace.perfetto.json"), views, model["records"], cv.f)
    summ = {k: v for k, v in model.items() if k in ("summary", "kinds", "group_totals", "checks", "run", "launch", "other_cycles", "sim", "per_token")}
    summ["layers"] = [{"layer": L["layer"], "groups": L["groups"]} for L in model["layers"]]
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summ, f, indent=1, default=lambda o: o if not isinstance(o, float) else round(o, 6))
    print("report: %s (%.1f MB), %s (%d events), %s" % (os.path.join(a.out, "report.html"), n / 1e6, os.path.join(a.out, "trace.perfetto.json"), nev,
                                                          os.path.join(a.out, "summary.json")))
    bad = [c for c in model["checks"] if c["ok"] is False]
    for c in bad:
        print("check differs: " + c["text"], file=sys.stderr)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report")
    r.add_argument("--card", required=True)
    r.add_argument("--sim")
    r.add_argument("--out", required=True)
    r.add_argument("--launch", type=int)
    r.add_argument("--model-name")
    r.add_argument("--expect-cycles", type=float, help="cycles of the decode launch measured earlier (for the check)")
    r.add_argument("--ref-toks", type=float, help="tok/s of an earlier run with the profiler off")
    r.add_argument("--ref-run", help="run.log directory of the same run with the profiler off (capture_card.sh with NO_DUMP=1): llama.cpp's speed and PERF total")
    r.add_argument("--commands", help="text of the commands that made the run, shown in the Method section")
    s = sub.add_parser("sched")
    s.add_argument("log")
    s.add_argument("out")
    a = ap.parse_args()
    sys.exit({"report": cmd_report, "sched": cmd_sched}[a.cmd](a))


if __name__ == "__main__":
    main()
