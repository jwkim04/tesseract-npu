"""prof_html.py: the one-file HTML report (docs/prof-dbg.md "The profile report"). The page is a profiler window: a toolbar, a tree of the
launch, a timeline with one row per unit and a bottom pane with tabs. This module prepares the data (one JSON block) and the page frame
and joins them with report.css and report.js; the script fills the panes. No external script, stylesheet or font. The output depends only
on the model, so equal inputs give equal bytes. Standard library only."""
import html
import json
import os

import prof_graph as G
import prof_lib as P

HERE = os.path.dirname(os.path.abspath(__file__))
PERF_KEYS = ("CYCLES", "RETIRED", "MVM_CYCLES", "VU_CYCLES", "DMA_CYCLES", "QACT_CYCLES", "ATT_CYCLES", "MVM_WAIT", "MVM_EMPTY", "HOLD_CYCLES",
             "RV_WFI", "VU_SPM_WAIT", "VU_HAZARD", "SPM_RD_WAIT", "SPM_WR_WAIT")
TRACK_ORDER = [2, 3, 0, 1, 4, 5]     # MVM, VU, DMA, QACT, ATT, SYNC


def esc(s):
    return html.escape(str(s), quote=True)


def _lanes(spans):
    """Greedy lane of each span (start, end), sorted by start: spans of a unit that overlap in time go to further lanes."""
    ends, out = [], []
    for a, b in spans:
        for k, e in enumerate(ends):
            if a >= e:
                ends[k] = b
                out.append(k)
                break
        else:
            ends.append(b)
            out.append(len(ends) - 1)
    return out


def timeline_rows(tl, span, tag, label):
    """One run of the timeline: the issue band (record bands from the NODE_BEGIN markers) and one row per unit lane.
    Op row: [start, duration, record, opcode, sub, queue wait, head wait, ideal, ordinal]; issue row: [start, duration, record]."""
    marks = sorted(tl["marks"], key=lambda m: m[1])
    issue = []
    for k, (i, t) in enumerate(marks):
        t2 = marks[k + 1][1] if k + 1 < len(marks) else span
        issue.append([t, max(t2 - t, 0), i])
    rows = [{"id": "%s.issue" % tag, "unit": -1, "label": "RISC-V issue", "ops": issue}]
    by_unit = {}
    for o in tl["ops"]:
        if o[3] > 0:
            by_unit.setdefault(o[1], []).append([o[2], o[3], o[0], o[6], o[7], o[4], o[5], o[8], o[9]])
    for u in TRACK_ORDER:
        ops = sorted(by_unit.get(u, []), key=lambda o: (o[0], o[0] + o[1]))
        if not ops:
            continue
        lanes = _lanes([(o[0], o[0] + o[1]) for o in ops])
        for k in range(max(lanes) + 1):
            rows.append({"id": "%s.%s.%d" % (tag, P.UNITS[u], k), "unit": u, "label": P.UNITS[u] if k == 0 else "%s, overlap %d" % (P.UNITS[u], k),
                         "ops": [o for o, l in zip(ops, lanes) if l == k]})
    return {"tag": tag, "label": label, "rows": rows}


def page_data(model, extra):
    s = model["summary"]
    run = model["run"]["card"]
    capture = model.get("capture") or {}
    gm = G.GraphModel(model)
    tl = model["timeline"]
    runs = [timeline_rows(tl["card"], tl["span"], "card", "Card")]
    if tl.get("sim"):
        runs.append(timeline_rows(tl["sim"], tl["sim_span"], "sim", "Simulator"))
    recs = []
    for r in model["records"]:
        d = {"i": r["i"], "kn": r["kn"], "name": r["name"], "node": r["node"], "layer": r["layer"], "group": r["group"], "shape": r["shape"],
             "dep": r["dep"], "tiles": r["tiles"], "tile_rows": r["tile_rows"], "scratch": r["scratch"], "rd": r["rd"], "wr": r["wr"], "w": r["w"],
             "ideal": r["ideal"], "cost": r["cost"], "n_ops": r["n_ops"], "first": r["first"], "last": r["last"], "args": r["args"],
             "rdr": r["rdr"], "wrr": r["wrr"]}
        if "sim" in r:
            d["sim"] = r["sim"]["cost"]
            d["sim_ops"] = r["sim"]["n_ops"]
        recs.append(d)
    nodes = [{"i": x["i"], "name": x["name"], "op": x["op"], "type": x["type"], "ne": x["ne"], "place": x["place"], "grp": x["grp"], "layer": x["layer"],
              "g": x["g"], "recs": x["recs"], "cost": x["cost"], "src": [[a[0], a[1]] for a in x["src"]]} for x in model["graph"]["nodes"]]
    sim = model.get("sim")
    summary = {k: s.get(k) for k in ("cycles", "span", "core_hz", "t_dev_ms", "tok_s_dev", "launch_wall_ms", "token_period_ms", "tok_s", "n_decode",
                                     "ref_toks", "mvm_cycles", "ideal_cycles", "w_bytes", "rd_model", "rd_meas", "setup", "tail", "H", "G", "F", "E", "U",
                                     "E_clock", "empty", "wait", "mvm_dual", "t_max")}
    summary["llama_on"] = s.get("llama_on")
    summary["llama_off"] = s.get("llama_off")
    return {
        "meta": {"model": model.get("model") or "", "image": run["image_id"], "commit": capture.get("git", ""), "date": (capture.get("date") or "")[:10],
                 "xclbin": (capture.get("xclbin_sha256") or "")[:8], "launch": model["launch"], "n_launches": run["n_launches"],
                 "description": run["description"], "sim_description": (model["run"].get("sim") or {}).get("description", ""),
                 "f": tl["f"], "span": int(tl["span"]) + 1, "sim_span": tl.get("sim_span"), "has_sim": bool(tl.get("sim")),
                 "sim_note": model.get("sim_note") or "", "commands": (extra or {}).get("commands", "")},
        "groups": model["groups"], "group_totals": model["group_totals"], "layers": [L["layer"] for L in model["layers"]],
        "summary": summary, "perf": {k: s["perf"].get(k, 0) for k in PERF_KEYS},
        "sim": ({"span": sim["span"], "cycles": sim["cycles"], "perf": {k: sim["perf"].get(k, 0) for k in PERF_KEYS}} if sim else None),
        "checks": model["checks"], "per_token": model["per_token"], "kinds": model["kinds"], "records": recs, "nodes": nodes,
        "fusion": model["graph"]["groups"], "lists": model["graph"]["lists"], "dynamic": model["graph"].get("dynamic") or [],
        "sched": (model["graph"].get("sched") or {}).get("splits"), "cpu": gm.cpu, "launches": model["launches"], "launch_groups": model["launch_groups"],
        "spm_lines": model.get("spm_lines") or 65536, "units": P.UNITS, "opnames": {str(k): v for k, v in P.OPNAME.items()},
        "tl": {"runs": runs},
    }


def frame(D):
    """The static frame of the page: toolbar, tree, timeline, bottom pane. The script fills the lists and tables."""
    m = D["meta"]
    opts = []
    for l in D["launches"]:
        what = "decode" if l["decode"] else "other"
        sel = " selected" if l["k"] == m["launch"] else ""
        dis = "" if (l["k"] == m["launch"]) else " disabled"
        opts.append('<option value="%d"%s%s>%d  %s  %s cycles</option>' % (l["k"], sel, dis, l["k"], what, "{:,}".format(int(l["cycles"]))))
    kv = [("Model", m["model"] or "-"), ("Image", m["image"] or "-"), ("Commit", m["commit"] or "-"), ("Clock", "%.1f MHz" % (m["f"] / 1e6))]
    if m["xclbin"]:
        kv.insert(2, ("Bitstream", m["xclbin"]))
    kvs = "".join('<span class="kv"><span class="k">%s</span><span class="v">%s</span></span>' % (esc(k), esc(v)) for k, v in kv)
    return """<div id="app">
<header id="toolbar" role="toolbar" aria-label="Report">
<span class="title">Tesseract Profile</span>
%s
<label class="kv"><span class="k">Launch</span><select id="launch" aria-label="Launch">%s</select></label>
<span class="spacer"></span>
<button id="treebtn" type="button" aria-controls="tree">Tree</button>
<button id="theme" type="button" title="Switch between the light and the dark theme">Theme</button>
</header>
<div id="body">
<aside id="tree" aria-label="Launch tree"><div class="ph"><span>Launch</span><input id="treefilter" type="search" placeholder="Filter" aria-label="Filter the tree"></div><div id="treelist" role="tree" tabindex="0"></div></aside>
<div id="main">
<section id="tl" aria-label="Timeline">
<div class="ph"><span>Timeline</span><span id="tlrange" class="num"></span><span class="spacer"></span>
<button id="zin" type="button">Zoom in</button><button id="zout" type="button">Zoom out</button><button id="zsel" type="button">Selection</button><button id="zfit" type="button">Fit</button></div>
<div id="tlscroll" tabindex="0"><div id="tlgrid"><div id="corner"></div><div id="rulerw"><canvas id="ruler"></canvas></div><div id="labels"></div><div id="rowsw"><canvas id="rows"></canvas></div></div></div>
</section>
<div id="splitter" role="separator" aria-orientation="horizontal" aria-label="Resize the panes"></div>
<section id="bottom" aria-label="Details">
<div id="tabs" role="tablist"><button role="tab" data-tab="summary" type="button">Summary</button><button role="tab" data-tab="records" type="button">Records</button><button role="tab" data-tab="graph" type="button">Graph</button><button role="tab" data-tab="details" type="button">Details</button></div>
<div id="panes"><div class="pane" id="p-summary"></div><div class="pane" id="p-records"></div><div class="pane" id="p-graph"></div><div class="pane" id="p-details"></div></div>
</section>
</div>
</div>
<footer id="status"><span id="st-sel"></span><span class="spacer"></span><span id="legend"></span><span id="st-cur" class="num"></span></footer>
<div id="tip" role="tooltip" hidden></div>
</div>""" % (kvs, "".join(opts))


def build_html(model, out_path, extra=None):
    D = page_data(model, extra)
    css = open(os.path.join(HERE, "report.css")).read()
    js = open(os.path.join(HERE, "report.js")).read()
    data = json.dumps(D, separators=(",", ":")).replace("</", "<\\/")
    doc = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tesseract Profile</title>
<style>%s</style>
</head>
<body>
%s
<script type="application/json" id="data">%s</script>
<script>%s</script>
</body>
</html>
""" % (css, frame(D), data, js)
    with open(out_path, "w") as fh:
        fh.write(doc)
    return len(doc)
