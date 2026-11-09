"""prof_html.py: the one-file HTML report (docs/prof-dbg.md "The profile report"). The charts, the graph and the timeline are SVG made
here; the page's script only switches layers, shows details and zooms the timeline. No external script or stylesheet.
Standard library only."""
import html
import json
import os

import prof_graph as G
import prof_model as M
import prof_lib as P

HERE = os.path.dirname(os.path.abspath(__file__))
CYCLE_COLOR = {"attention": "var(--g-attention)", "router": "var(--g-router)", "experts": "var(--g-experts)", "dense": "var(--g-dense)",
               "residual": "var(--g-residual)", "head": "var(--g-head)"}


def esc(s):
    return html.escape(str(s), quote=True)


def fnum(n, d=0):
    return ("{:,.%df}" % d).format(n)


def fcyc(n):
    return G.fmt_cyc(n)


def pct(x, d=1):
    return ("%." + str(d) + "f %%") % (100.0 * x)


def nice_max(v):
    """A round axis maximum at or above v and about five ticks."""
    import math
    if v <= 0:
        return 1.0, 1.0
    e = 10 ** math.floor(math.log10(v))
    for m in (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if m * e >= v:
            top = m * e
            break
    step = top / 5.0 if top / e in (5, 10, 2.5, 1.5) else top / 4.0
    for st in (e * 0.1, e * 0.2, e * 0.25, e * 0.5, e, e * 2, e * 2.5, e * 5):
        if top / st <= 6:
            step = st
            break
    return top, step


# ---------------------------------------------------------------------------------------------
# summary charts
# ---------------------------------------------------------------------------------------------
def layer_chart(model):
    """Measured cycles of each layer stacked by kind group, beside the ideal (weight bytes / 256 B per cycle); the simulator's
    total as a tick on the measured bar. The embedding and head have a panel of their own."""
    layers = model["layers"]
    mid = [L for L in layers if L["layer"] not in ("in", "out")]
    side = [L for L in layers if L["layer"] in ("in", "out")]
    has_sim = model.get("sim") is not None

    def tot(L, key):
        return sum(g[key] for g in L["groups"].values())
    out = []
    # panel geometry
    pitch = 34
    bw = 11
    left, top, bottom = 52, 14, 34
    H = 300
    plot_h = H - top - bottom
    W1 = left + len(mid) * pitch + 10
    ymax, step = nice_max(max([tot(L, "cycles") for L in mid] + [tot(L, "ideal") for L in mid] + [1]) * 1.04)
    out.append('<svg width="%d" height="%d" viewBox="0 0 %d %d" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Measured and ideal cycles per layer">' % (W1, H, W1, H))
    v = 0.0
    while v <= ymax + 1e-9:
        y = top + plot_h * (1 - v / ymax)
        out.append('<line class="gl" x1="%d" x2="%d" y1="%.1f" y2="%.1f"/>' % (left, W1 - 6, y, y))
        out.append('<text x="%d" y="%.1f" text-anchor="end">%s</text>' % (left - 6, y + 4, ("%.2f M" % (v / 1e6)) if ymax >= 1e6 else ("%d K" % (v / 1e3))))
        v += step
    out.append('<line class="ax" x1="%d" x2="%d" y1="%d" y2="%d"/>' % (left, W1 - 6, top + plot_h, top + plot_h))
    for k, L in enumerate(mid):
        x0 = left + k * pitch + 4
        out.append('<g class="bar" data-layer="%s">' % esc(L["layer"]))
        for j, key in enumerate(("cycles", "ideal")):
            y = top + plot_h
            for g in M.GROUPS:
                val = L["groups"][g][key]
                if val <= 0:
                    continue
                h = plot_h * val / ymax
                gap = 2 if h > 4 else 0
                y -= h
                out.append('<rect class="seg%s g-%s" x="%d" y="%.1f" width="%d" height="%.1f" style="fill:var(--g-%s)"><title>%s</title></rect>' %
                           (" ideal" if key == "ideal" else "", g, x0 + j * (bw + 2), y + gap, bw, max(h - gap, 0.5), g,
                            esc("Layer %s, %s, %s: %s cycles" % (L["layer"], M.GROUP_LABEL[g], "ideal" if key == "ideal" else "measured", fnum(val)))))
        if has_sim:
            ys = top + plot_h * (1 - tot(L, "sim") / ymax)
            out.append('<line class="simtick" x1="%d" x2="%d" y1="%.1f" y2="%.1f"><title>%s</title></line>' % (x0 - 1, x0 + bw + 1, ys, ys,
                       esc("Layer %s, simulator: %s cycles" % (L["layer"], fnum(tot(L, "sim"))))))
        out.append('<rect class="hit" x="%d" y="%d" width="%d" height="%d" fill-opacity="0"><title>%s</title></rect>' %
                   (left + k * pitch, top, pitch, plot_h + 20, esc("Layer %s: %s measured, %s ideal; click to open the layer in the graph" % (L["layer"], fcyc(tot(L, "cycles")), fcyc(tot(L, "ideal"))))))
        out.append('</g>')
        if k % 2 == 0 or len(mid) < 12:
            out.append('<text x="%d" y="%d" text-anchor="middle">%s</text>' % (x0 + bw, top + plot_h + 14, esc(L["layer"])))
    out.append('<text x="%d" y="%d" text-anchor="middle">layer</text>' % (left + len(mid) * pitch / 2, H - 4))
    out.append('</svg>')
    panel1 = "".join(out)
    # the side panel (embedding, head) has its own scale
    out = []
    pitch2, left2 = 54, 52
    W2 = left2 + len(side) * pitch2 + 12
    ymax2, step2 = nice_max(max([tot(L, "cycles") for L in side] + [tot(L, "ideal") for L in side] + [1]) * 1.04)
    out.append('<svg width="%d" height="%d" viewBox="0 0 %d %d" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Embedding and head cycles (own scale)">' % (W2, H, W2, H))
    v = 0.0
    while v <= ymax2 + 1e-9:
        y = top + plot_h * (1 - v / ymax2)
        out.append('<line class="gl" x1="%d" x2="%d" y1="%.1f" y2="%.1f"/>' % (left2, W2 - 4, y, y))
        out.append('<text x="%d" y="%.1f" text-anchor="end">%s</text>' % (left2 - 6, y + 4, ("%.1f M" % (v / 1e6)) if ymax2 >= 1e6 else ("%d K" % (v / 1e3))))
        v += step2
    out.append('<line class="ax" x1="%d" x2="%d" y1="%d" y2="%d"/>' % (left2, W2 - 4, top + plot_h, top + plot_h))
    for k, L in enumerate(side):
        x0 = left2 + k * pitch2 + 8
        out.append('<g class="bar" data-layer="%s">' % esc(L["layer"]))
        for j, key in enumerate(("cycles", "ideal")):
            y = top + plot_h
            for g in M.GROUPS:
                val = L["groups"][g][key]
                if val <= 0:
                    continue
                h = plot_h * val / ymax2
                gap = 2 if h > 4 else 0
                y -= h
                out.append('<rect class="seg%s g-%s" x="%d" y="%.1f" width="%d" height="%.1f" style="fill:var(--g-%s)"><title>%s</title></rect>' %
                           (" ideal" if key == "ideal" else "", g, x0 + j * (bw + 2), y + gap, bw, max(h - gap, 0.5), g,
                            esc("%s, %s: %s cycles" % ({"in": "Embedding", "out": "Head"}[L["layer"]], "ideal" if key == "ideal" else "measured", fnum(val)))))
        if has_sim:
            ys = top + plot_h * (1 - tot(L, "sim") / ymax2)
            out.append('<line class="simtick" x1="%d" x2="%d" y1="%.1f" y2="%.1f"/>' % (x0 - 1, x0 + bw + 1, ys, ys))
        out.append('<rect class="hit" x="%d" y="%d" width="%d" height="%d" fill-opacity="0"><title>%s</title></rect>' %
                   (left2 + k * pitch2, top, pitch2, plot_h + 20, esc("%s: %s measured; click to open" % ({"in": "Embedding", "out": "Output head"}[L["layer"]], fcyc(tot(L, "cycles"))))))
        out.append('</g>')
        out.append('<text x="%d" y="%d" text-anchor="middle">%s</text>' % (x0 + bw, top + plot_h + 14, {"in": "embed", "out": "head"}[L["layer"]]))
    out.append('<text x="%d" y="%d" text-anchor="middle">own scale</text>' % (left2 + len(side) * pitch2 / 2, H - 4))
    out.append('</svg>')
    panel2 = "".join(out)
    return panel1, panel2


def token_chart(model):
    pts = [p for p in model.get("per_token", []) if p["card"]]
    if len(pts) < 3:
        return ""
    W, H, left, top, bottom = 560, 190, 56, 12, 28
    lo = min([p["card"] for p in pts] + [p["sim"] for p in pts if p["sim"]])
    hi = max([p["card"] for p in pts] + [p["sim"] for p in pts if p["sim"]])
    pad = max((hi - lo) * 0.15, 1000)
    lo -= pad
    hi += pad
    xs = lambda i: left + (W - left - 12) * i / max(len(pts) - 1, 1)
    ys = lambda v: top + (H - top - bottom) * (1 - (v - lo) / (hi - lo))
    o = ['<svg width="%d" height="%d" viewBox="0 0 %d %d" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Cycles of each decode token, card and simulator">' % (W, H, W, H)]
    for k in range(5):
        v = lo + (hi - lo) * k / 4
        y = ys(v)
        o.append('<line class="gl" x1="%d" x2="%d" y1="%.1f" y2="%.1f"/><text x="%d" y="%.1f" text-anchor="end">%.3f M</text>' % (left, W - 8, y, y, left - 6, y + 4, v / 1e6))
    for series, col, nm in (("card", "var(--g-attention)", "card"), ("sim", "var(--g-router)", "simulator")):
        sel = [(xs(i), ys(p[series])) for i, p in enumerate(pts) if p[series]]
        if len(sel) > 1:
            o.append('<polyline fill="none" stroke="%s" stroke-width="2" stroke-linejoin="round" points="%s"/>' % (col, " ".join("%.1f,%.1f" % q for q in sel)))
            for q, p in zip(sel, [p for p in pts if p[series]]):
                o.append('<circle cx="%.1f" cy="%.1f" r="3.5" fill="%s" stroke="var(--surface)" stroke-width="2"><title>%s</title></circle>' % (q[0], q[1], col,
                         esc("launch %d, %s: %s cycles" % (p["k"], nm, fnum(p[series])))))
    o.append('<text x="%d" y="%d" text-anchor="middle">decode token (launch number)</text>' % ((left + W) // 2, H - 6))
    o.append('<text x="%d" y="%d" text-anchor="middle">%d</text><text x="%d" y="%d" text-anchor="middle">%d</text>' % (left, H - 16, pts[0]["k"], W - 12, H - 16, pts[-1]["k"]))
    o.append('</svg>')
    return "".join(o)


# ---------------------------------------------------------------------------------------------
# timeline
# ---------------------------------------------------------------------------------------------
TRACKS = [("MVM", 2), ("VU", 3), ("DMA", 0), ("QACT", 1), ("ATT", 4)]


def timeline_block(tag, label, tl, rows, span, show_sync=False):
    """The tracks of one run: a record band (issue slices) and one track per unit that has work, as SVG in cycle units."""
    ops = tl["ops"]
    th, gap = 18, 2
    bins = 600
    tracks = [("records", None)] + [(n, u) for n, u in TRACKS if any(o[1] == u and o[3] > 0 for o in ops)]
    if show_sync and any(o[1] == 5 for o in ops):
        tracks.append(("SYNC", 5))
    out = ['<div class="lab run">%s</div><div></div>' % esc(label)]
    for name, u in tracks:
        h = th if u is not None else 12
        parts = ['<svg class="trk" data-tag="%s" data-track="%s" viewBox="0 0 %d %d" preserveAspectRatio="none" height="%d" xmlns="http://www.w3.org/2000/svg">' % (tag, name, span, h, h)]
        if u is None:
            marks = tl["marks"]
            for k, (i, t) in enumerate(marks):
                t2 = marks[k + 1][1] if k + 1 < len(marks) else span
                r = rows[i]
                parts.append('<rect class="s g-%s" x="%d" y="1" width="%d" height="%d" data-rec="%d" style="fill:var(--g-%s)"/>' % (r["group"], t, max(t2 - t - 0, 1), h - 2, i, r["group"]))
        else:
            dens = [0.0] * bins
            for o in ops:
                if o[1] != u or o[3] <= 0:
                    continue
                a, b = o[2], o[2] + o[3]
                ia, ib = int(a * bins / span), int(min(b, span - 1) * bins / span)
                for bi in range(max(ia, 0), min(ib, bins - 1) + 1):
                    lo, hi = bi * span / bins, (bi + 1) * span / bins
                    dens[bi] += max(0.0, min(b, hi) - max(a, lo)) / (span / bins)
            d = ["M0 %d" % h]
            for bi, v in enumerate(dens):
                d.append("L%.0f %.1f" % (bi * span / bins, h - min(v, 1.0) * (h - 2)))
                d.append("L%.0f %.1f" % ((bi + 1) * span / bins, h - min(v, 1.0) * (h - 2)))
            d.append("L%d %d Z" % (span, h))
            parts.append('<path d="%s" style="fill:var(--ink);opacity:.14"/>' % " ".join(d))
            for k, o in enumerate(ops):
                if o[1] != u or o[3] <= 0:
                    continue
                r = rows[o[0]] if o[0] >= 0 else None
                grp = r["group"] if r else "head"
                parts.append('<rect id="%s%d" class="s g-%s" x="%d" y="1" width="%d" height="%d" data-k="%d" style="fill:var(--g-%s)"/>' %
                             (tag[0], k, grp, o[2], max(o[3], 1), h - 2, k, grp))
        parts.append('</svg>')
        out.append('<div class="lab">%s</div>%s' % (esc(name if name != "records" else "records (issue)"), "".join(parts)))
    return "".join(out)


# ---------------------------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------------------------
def kind_table(model):
    sim = model.get("sim") is not None
    rows = []
    rows.append('<table><thead><tr><th>Record kind</th><th class="num">Records</th><th class="num">Cycles</th><th class="num">Share</th><th class="num">Ideal</th>'
                '<th class="num">Efficiency</th><th class="num">GB/s</th>%s<th>Open</th></tr></thead><tbody>' %
                ('<th class="num">Simulator</th><th class="num">Sim / card</th>' if sim else ""))
    for d in model["kinds"]:
        eff = ("%d %%" % round(100 * d["eff"])) if d["eff"] else "–"
        ideal = fcyc(d["ideal"]) if d["ideal"] else "–"
        r = model["records"][d["best"]] if d["best"] is not None else None
        link = ""
        if r is not None and r.get("slice"):
            link = '<a class="lk" data-slice="%s" title="the longest %s record">slice</a> · <a class="lk" data-rec="%d">record</a>' % (esc(r["slice"]), esc(d["kind"]), d["best"])
            if r["node"] >= 0:
                link += ' · <a class="lk" data-node="%d">node</a>' % r["node"]
        cells = '<td>%s</td><td class="num">%d</td><td class="num">%s</td><td class="num">%s</td><td class="num">%s</td><td class="num">%s</td><td class="num">%s</td>' % (
            esc(d["kind"]), d["count"], fnum(d["cycles"]), pct(d["share"]), ideal, eff, ("%.1f" % d["gbs"]) if d["gbs"] and d["bytes"] else "–")
        if sim:
            cells += '<td class="num">%s</td><td class="num">%s</td>' % (fnum(d["sim"]), ("%.3f" % d["ratio"]) if d["ratio"] else "–")
        rows.append("<tr>%s<td>%s</td></tr>" % (cells, link))
    oc = model["other_cycles"]
    rows.append('<tr><td>launch setup and tail</td><td class="num">–</td><td class="num">%s</td><td class="num">%s</td><td></td><td></td><td></td>%s<td></td></tr>' %
                (fnum(oc["setup"] + oc["tail"]), pct((oc["setup"] + oc["tail"]) / model["summary"]["span"]), "<td></td><td></td>" if sim else ""))
    rows.append("</tbody></table>")
    return "".join(rows)


def group_table(model):
    sim = model.get("sim") is not None
    t = ['<table><thead><tr><th>Kind group</th><th class="num">Cycles</th><th class="num">Share</th><th class="num">Ideal</th><th class="num">Efficiency</th>%s</tr></thead><tbody>' %
         ('<th class="num">Simulator</th><th class="num">Sim / card</th>' if sim else "")]
    span = model["summary"]["span"]
    for g in M.GROUPS:
        d = model["group_totals"][g]
        eff = ("%d %%" % round(100 * d["ideal"] / d["cycles"])) if d["ideal"] and d["cycles"] else "–"
        t.append('<tr><td><span class="sw g-%s"></span>%s</td><td class="num">%s</td><td class="num">%s</td><td class="num">%s</td><td class="num">%s</td>%s</tr>' %
                 (g, esc(M.GROUP_LABEL[g]), fnum(d["cycles"]), pct(d["cycles"] / span), fcyc(d["ideal"]) if d["ideal"] else "–", eff,
                  ('<td class="num">%s</td><td class="num">%.3f</td>' % (fnum(d["sim"]), d["cycles"] / d["sim"] if d["sim"] else 0)) if sim else ""))
    t.append("</tbody></table>")
    return "".join(t)


def records_table(model):
    sim = model.get("sim") is not None
    t = ['<table id="rectab"><thead><tr><th class="num">#</th><th>Kind</th><th>Record</th><th>Layer</th><th>Group</th><th>Shape</th><th class="num">Cycles</th>'
         '<th class="num">Ideal</th><th class="num">Eff</th>%s<th>Open</th></tr></thead><tbody>' % ('<th class="num">Sim</th>' if sim else "")]
    for r in model["records"]:
        eff = ("%d %%" % round(100 * r["ideal"] / r["cost"])) if r["ideal"] and r["cost"] else "–"
        t.append('<tr class="cl" data-rec="%d" data-kind="%s" data-group="%s" data-layer="%s"><td class="num">%d</td><td>%s</td><td>%s</td><td>%s</td><td><span class="sw g-%s"></span>%s</td>'
                 '<td>%s</td><td class="num">%s</td><td class="num">%s</td><td class="num">%s</td>%s<td>%s</td></tr>' %
                 (r["i"], esc(r["kn"]), r["group"], esc(r["layer"]), r["i"], esc(r["kn"]), esc(G.trunc(r["name"], 48)), esc(r["layer"]), r["group"], esc(r["group"]),
                  esc(G.trunc(r["shape"], 44)), fnum(r["cost"]), fcyc(r["ideal"]) if r["ideal"] else "–", eff,
                  ('<td class="num">%s</td>' % fnum(r["sim"]["cost"])) if sim else "",
                  ('<a class="lk" data-slice="%s">slice</a>' % esc(r["slice"])) if r.get("slice") else ""))
    t.append("</tbody></table>")
    return "".join(t)


def checks_list(model):
    li = []
    for c in model["checks"]:
        st = {True: ('ok', 'ok'), False: ('bad', 'differs'), None: ('info', 'info')}[c["ok"]]
        li.append('<li><span class="st %s">%s</span>%s</li>' % (st[0], ("✓ " if c["ok"] is True else ("✗ " if c["ok"] is False else "· ")) + st[1], esc(c["text"])))
    return '<ul class="checks">%s</ul>' % "".join(li)


def factor_rows(model):
    s = model["summary"]
    if "H" not in s:
        return "<p class=\"note\">The utilisation split needs the host timestamps of two consecutive decode launches.</p>"
    rows = [("(1 − H)  device busy", 1 - s["H"], "H = %s: wall time with no launch active (host work, the PCIe round trip, sampling)" % pct(s["H"])),
            ("(1 − G)  MVM busy in the launch", 1 - s["G"], "G = %s: device time outside the MVM (LD, QACT, VU, ST, issue gaps)" % pct(s["G"])),
            ("(1 − F)  weight stream in MVM time", 1 - s["F"], "F = %s: MVM fill, broadcast, tail and bubbles inside the MVM time" % pct(s["F"])),
            ("E  stream rate / M0 bandwidth", min(s["E"], 1.0), "E = %s: bytes read per ideal weight-stream cycle × core clock ÷ 72.05 GB/s; at 256 B per cycle it is %s (the clock bound)" % (pct(s["E"]), pct(s["E_clock"])))]
    out = []
    for lab, v, tip in rows:
        out.append('<div class="fac"><div>%s</div><div class="bar" title="%s"><i style="width:%.1f%%"></i></div><div class="v">%s</div></div>' % (esc(lab), esc(tip), 100 * v, pct(v)))
    out.append('<div class="fac"><div><b>U = product</b></div><div class="bar"><i style="width:%.1f%%"></i></div><div class="v"><b>%s</b></div></div>' % (100 * s["U"], pct(s["U"])))
    return "".join(out)


# ---------------------------------------------------------------------------------------------
# the page
# ---------------------------------------------------------------------------------------------
def build_html(model, out_path, extra=None):
    s = model["summary"]
    run = model["run"]["card"]
    has_sim = model.get("sim") is not None
    gm = G.GraphModel(model)
    # layer SVGs
    layer_svgs = []
    for L in model["layers"]:
        r = G.render_layer(gm, L["layer"], 1)
        if r is not None:
            layer_svgs.append((L["layer"], r[0], r[1]))
    overview = G.render_overview(gm, model)
    panel1, panel2 = layer_chart(model)
    tchart = token_chart(model)
    # counts for the graph section
    nodes = model["graph"]["nodes"]
    cnt = {"device": 0, "absorbed": 0, "noop": 0, "skipped": 0, "empty": 0}
    for x in nodes:
        cnt[x["place"]] = cnt.get(x["place"], 0) + 1
    n_cpu = len(gm.cpu)
    n_groups = len(model["graph"]["groups"])
    sched = model["graph"].get("sched")
    # timeline
    tl = model["timeline"]
    span = int(tl["span"]) + 1
    f = tl["f"]
    tl_html = timeline_block("card", "card", tl["card"], model["records"], span)
    if tl.get("sim"):
        tl_html += timeline_block("sim", "simulator", tl["sim"], model["records"], span)
    title = "Decode Token Profile"
    model_name = model.get("model") or ""
    capture = model.get("capture") or {}
    bit = (" bitstream %s," % esc(capture["xclbin_sha256"][:8])) if capture.get("xclbin_sha256") else ""
    sub = "%s · U200 card,%s firmware image %s, core clock %.1f MHz · decode launch %d of %d launches (%d decode tokens captured)%s" % (
        esc(model_name), bit, esc(run["image_id"]), (run["core_hz"] or 0) / 1e6, model["launch"], run["n_launches"], s["n_decode"],
        (" · captured " + esc(capture.get("date", "")[:10])) if capture.get("date") else "")
    # tiles
    tok = s.get("tok_s")
    on, off = s.get("llama_on") or {}, s.get("llama_off") or {}
    tiles = []
    hero = s.get("ref_toks") or tok
    note = []
    if off.get("toks"):
        note.append("llama.cpp eval time, profiler off, same session (%d decode tokens)" % (s["n_decode"] - 1))
        if on.get("toks"):
            note.append("profiler on: %.2f tok/s" % on["toks"])
        if tok:
            note.append("%.2f tok/s (%.1f ms per token) from the timestamps of consecutive launches, which U and H use" % (tok, s["token_period_ms"]))
        if off.get("perf_cycles") and on.get("perf_cycles"):
            note.append("PERF cycles of the whole run: %s on, %s off" % (fnum(on["perf_cycles"]), fnum(off["perf_cycles"])))
    else:
        note.append("profiler on; %.1f ms per token from the host timestamps of %d consecutive decode launches" % (s["token_period_ms"] or 0, s["n_decode"] - 1) if tok else "")
        if s.get("ref_toks"):
            note.append("profiler off: %.2f tok/s (earlier measurement)" % s["ref_toks"])
    tiles.append('<div class="tile hero"><div class="l">Decode speed</div><div class="v">%s tok/s</div><div class="n">%s</div></div>' % (
        ("%.2f" % hero) if hero else "–", esc("; ".join(n for n in note if n))))
    tiles.append('<div class="tile"><div class="l">Device time per token</div><div class="v">%s cycles</div><div class="n">%.2f ms at %.1f MHz, %.2f tok/s from device time alone</div></div>' % (
        fnum(s["cycles"]), s["t_dev_ms"], s["core_hz"] / 1e6, s["tok_s_dev"]))
    if "U" in s:
        tiles.append('<div class="tile"><div class="l">DDR utilisation U</div><div class="v">%s</div><div class="n">%.2f GB read per token (PERF) · ceiling at this clock %s</div></div>' % (
            pct(s["U"]), s["rd_meas"] / 1e9, pct(s["E_clock"])))
    if has_sim:
        sm = model["sim"]
        tiles.append('<div class="tile"><div class="l">Simulator, same token</div><div class="v">%s cycles</div><div class="n">%+.2f %% against the card (TSR_SIM_CARD=k4full)</div></div>' % (
            fnum(sm["span"]), 100 * (sm["ratio"] - 1)))
    legend = '<div class="legend">%s<span><i class="sw" style="background:var(--ink);height:2px;width:14px;vertical-align:3px"></i>simulator total</span><span>left bar: measured, right bar: ideal</span></div>' % "".join(
        '<span><i class="sw g-%s"></i>%s</span>' % (g, esc(M.GROUP_LABEL[g])) for g in M.GROUPS)
    # graph section: split chain
    splits_html = []
    if sched and sched.get("splits"):
        for sp in sched["splits"]:
            cls = "cpu" if sp["backend"].startswith("CPU") else "dev"
            ins = sp.get("inputs") or []
            splits_html.append('<div class="b %s"><b>Split %d on %s</b><br>%d compute nodes, %d inputs copied in from other backends%s</div>' % (
                cls, sp["idx"], esc(sp["backend"]), sp["n_nodes"], len(ins), ("<br><span class=\"note\">" + esc(", ".join(sp.get("node_names", [])[:6])) + "</span>") if cls == "cpu" else ""))
            splits_html.append('<div class="arrow">→</div>')
    else:
        splits_html.append('<div class="b cpu"><b>CPU</b><br>%d nodes stay on the CPU (refused by supports_op)</div><div class="arrow">→</div>' % n_cpu)
        splits_html.append('<div class="b dev"><b>Tesseract0</b><br>%d graph nodes</div><div class="arrow">→</div>' % len(nodes))
    ln = model["graph"]["lists"]
    launch_box = '<div class="b dev"><b>Launch of the Tesseract split</b><br>%s<br><span class="note">list ended by: %s</span></div>' % (
        "; ".join("list %d: %d records (%d JOIN), %s op-list bytes, %d preload entries (%d lines)" % (l["idx"], l["n_ops"], l["joins"], fnum(l["blob"]), l["pre"][0], l["pre"][1]) for l in ln),
        esc("; ".join(l["end"] for l in ln)))
    splits_html.append(launch_box)
    dyn = model["graph"].get("dynamic") or []
    cpu_rows = ""
    cpu_list = gm.cpu
    plan_names = {x["name"] for x in nodes}
    refusals = [c for c in model["graph"].get("cpu", []) if c["name"].strip()]
    other = [c for c in refusals if c["name"] not in {x["name"] for x in cpu_list} and c["name"] not in plan_names]
    if cpu_list:
        cpu_rows = '<div class="scroll"><table><thead><tr><th>CPU node</th><th>Op</th><th>Why</th></tr></thead><tbody>%s</tbody></table></div>' % "".join(
            "<tr><td>%s</td><td>%s</td><td>%s</td></tr>" % (esc(c["name"]), esc(c["op"]), esc(c["why"])) for c in cpu_list)
    if other:
        cpu_rows += '<p class="note">Also refused by supports_op while the scheduler placed other graphs (not in this one): %s.</p>' % esc("; ".join("%s %s (%s)" % (c["op"], c["name"], c["why"]) for c in other))
    inputs = model["graph"]["inputs"]
    stage_btns = "".join('<button data-layer="%s">%s</button>' % (esc(L["layer"]), esc(G.layer_label(L["layer"]))) for L in model["layers"])
    layer_blocks = "".join('<div class="gvl" data-layer="%s" hidden><div class="gv">%s</div></div>' % (esc(l), svg) for l, svg, info in layer_svgs)
    # embedded data (what the script shows in the detail drawer)
    D = {
        "records": model["records"], "nodes": nodes, "groups_g": model["graph"]["groups"], "launch": model["launch"], "f": f,
        "span": span, "sim_span": tl.get("sim_span"), "has_sim": bool(tl.get("sim")), "ring_lines": 60144,
        "ops": {"card": tl["card"]["ops"], "sim": tl["sim"]["ops"] if tl.get("sim") else None},
        "units": P.UNITS, "opnames": {str(k): v for k, v in P.OPNAME.items()}, "kind_labels": M.GROUP_LABEL,
        "layers": [L["layer"] for L in model["layers"]], "cpu": gm.cpu,
    }
    data_json = json.dumps(D, separators=(",", ":")).replace("</", "<\\/")
    css = open(os.path.join(HERE, "report.css")).read()
    js = open(os.path.join(HERE, "report.js")).read()
    commands = extra.get("commands", "") if extra else ""
    doc = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>%(title)s</title>
<style>%(css)s</style>
</head>
<body>
<header class="top"><div class="wrap">
<h1>%(h1)s</h1>
<div class="sub">%(sub)s</div>
<nav><a href="#summary">Summary</a><a href="#graph">Graph</a><a href="#timeline">Timeline</a><a href="#records">Records</a><a href="#method">Method</a><button id="theme" title="Switch between the light and the dark theme">Theme</button></nav>
</div></header>
<main class="wrap">
<section id="summary">
<h2>Summary</h2>
<p class="note">One decode token of %(model)s on the U200 card (launch %(launch)d), with the same token from the simulator in the card configuration. Cycles are core cycles of the card's trace writer; a record's cycles are the increase of the launch's completion frontier over its ops, so they add up to the launch.</p>
<div class="tiles">%(tiles)s</div>
<div class="card"><h3>Where the DDR time goes: U ≈ (1 − H)(1 − G)(1 − F) E</h3>
%(factors)s
<p class="note">U is the bytes the card read per token (PERF read beats × 64 B) times tokens per second over B_M0 = 72.05 GB/s (docs/spec.md §14). MVM share of the launch: %(mvmshare)s; MVM_EMPTY %(empty)s and MVM_WAIT %(wait)s of the cycles. Ideal MVM cycles are weight bytes ÷ 256 B per cycle (four channels × 64 B): %(ideal)s cycles for this token against %(mvm)s MVM busy cycles.</p></div>
<div class="card"><h3>Cycles per layer, measured against ideal</h3>
%(legend)s
<div style="display:flex;flex-wrap:wrap;gap:12px;align-items:flex-start"><div class="scroll" style="max-width:100%%">%(panel1)s</div><div class="scroll">%(panel2)s</div></div>
<p class="note">Each layer has two bars: the measured cycles (the records' frontier cycles by kind group) and the ideal cycles of its weight streams. Click a layer to open its graph. The embedding and the output head are on a scale of their own.</p>
%(grouptab)s</div>
<div class="card"><h3>By record kind</h3><div class="scroll">%(kindtab)s</div>
<p class="note">Efficiency is ideal cycles over measured cycles for the kinds that stream weights; GB/s is the bytes a kind's records move (formulas of the record kinds) over their cycles. The links open the longest record of the kind in the timeline, its record and its graph node.</p></div>
%(simcard)s
<div class="card"><h3>Checks</h3>%(checks)s</div>
</section>

<section id="graph">
<h2>The compiled graph</h2>
<p class="note">ggml nodes → fusion groups → op-list records → firmware kinds, for the decode graph the backend built (plan %(planid)s). One scheduler pass per token: the splits below, one launch.</p>
<div class="card"><h3>Splits and the launch</h3>
<div class="split">%(splits)s</div>
<p>Nodes of the Tesseract split: <span class="pill">%(n_dev)d on the device (one record or more)</span><span class="pill">%(n_abs)d absorbed by %(n_groups)d fusion groups</span><span class="pill">%(n_noop)d views and reshapes (no record)</span>
<span class="pill">%(n_cpu)d on the CPU</span><span class="pill">%(n_in)d tensors copied in from the host or the CPU each token</span><span class="pill">%(n_rec)d records, %(n_launch)d launch per token</span>%(dyn)s</p>
%(cpurows)s
%(lgtab)s
%(spm)s
<p class="note">A launch ends where the op list ends: at the end of the split, when a list is full (4,096 records or 256 preload lines), or at a node whose operands the device cannot address in place (a staged MUL_MAT). %(cutnote)s %(refnote)s</p></div>
<div class="card"><h3>Layers</h3><div class="gv" style="max-height:none">%(overview)s</div>
<p class="note">Click a layer to open it. Boxes are records; a dashed box is a fusion group (its member nodes are listed in the details). Dashed red boxes are nodes that run on the CPU.</p></div>
<div class="card"><h3>Layer <span id="layername"></span></h3>
<div class="rowfilter" id="layerbtns">%(stagebtns)s</div>
%(layerblocks)s
<p class="note">Edges are data flow between records; the grey pills are tensors that come from, or go to, another layer or the host. Click a box for its node, record, arguments, DRAM ranges and SPM tiles.</p></div>
</section>

<section id="timeline">
<h2>Execution timeline of the token</h2>
<p class="note">One track per unit, the colours are the kind groups of the graph. The grey area under a track is its busy share. Click a slice for its details; the Perfetto file has the same slices with their arguments, flow arrows from the record to its ops and the PERF counter tracks.</p>
<div class="card">
<div class="tlbar"><button id="zin">Zoom in</button><button id="zout">Zoom out</button><button id="zreset">Whole token</button><span id="tlrange" class="note"></span><input type="text" id="slicesearch" placeholder="slice id, for example card:r412.MVM.0" aria-label="slice id"><button id="slicego">Find</button></div>
<div class="legend">%(legendg)s</div>
<div id="tlinfo" class="note" style="min-height:1.5em"></div>
<div class="scroll"><div class="tl tlin"><div></div><div class="axis" id="axis"></div>%(tl)s</div></div>
<p class="note">Slice ids are <code>card:r&lt;record&gt;.&lt;UNIT&gt;&lt;n&gt;</code> and <code>sim:...</code>; the same ids are in the Perfetto file (<code>args.id</code>). Perfetto cannot link into a local file, so search the id there with the omnibox (&ldquo;:&rdquo; then the id).</p>
</div>
</section>

<section id="records">
<h2>Records</h2>
<div class="rowfilter"><input type="text" id="recfilter" placeholder="filter by kind, name, layer, group" aria-label="filter records"><select id="grpfilter"><option value="">all groups</option>%(grpopts)s</select></div>
<div class="card scroll" style="max-height:70vh;overflow:auto">%(rectab)s</div>
</section>

<section id="method">
<h2>Method</h2>
<div class="card">
<p><b>Capture.</b> <code>TSR_DUMP_PLAN=&lt;dir&gt;</code> makes the ggml backend write, without changing a record or a bit, the plan of each graph it builds (nodes, fusion groups, records with argument words, dependencies, DRAM ranges and the firmware's SPM tile plan), one line per launch with the PERF deltas, the cause of each launch boundary, the nodes supports_op refused, and the trace writer's records of the whole run. The descriptor's PROF bit puts a NODE_BEGIN marker before every record; the marker's push index joins a trace op to its record.</p>
<p><b>Cycles.</b> A record's cycles are the increase of the launch's completion frontier (the latest end of any op so far, starting at LAUNCH_BEGIN) over its ops. Setup and tail are the cycles before the first record and after the last op. Ideal cycles of an MVM record are its weight bytes over 256 B per cycle (the rows are shared out over the four channels); an MVM slice carries its record's ideal split evenly over the record's MVM ops.</p>
<p><b>Utilisation.</b> H is the share of the token's wall time with no launch active (host timestamps of consecutive decode launches against PERF_CYCLES at the core clock), G = 1 − MVM_CYCLES / CYCLES, F = 1 − ideal / MVM_CYCLES, and E is the rest of the product: bytes read per ideal weight-stream cycle × core clock ÷ B_M0. At 256 B per cycle E is the clock bound 256 B × f ÷ 72.05 GB/s.</p>
<p><b>Profiler cost.</b> The markers add a RISC-V trace write per record and the PERF counters are read twice per launch outside the launch, so the speed here is a little below an unprofiled run; the cycle counts agree to the figures in the checks.</p>
<p><b>Layers and groups</b> come from the node names and weights (attention, router, experts, dense branch, residual and norms, embedding and head); they are for reading only.</p>
%(commands)s
</div>
</section>
</main>
<aside id="detail" hidden><div class="hd"><h3 id="dtitle"></h3><button id="dclose" aria-label="close">Close</button></div><div id="dbody"></div></aside>
<script type="application/json" id="data">%(data)s</script>
<script>%(js)s</script>
</body>
</html>
""" % {
        "title": title, "css": css, "js": js, "h1": esc(model_name + ": one decode token") if model_name else "One decode token", "sub": sub, "model": esc(model_name or "the model"),
        "launch": model["launch"], "tiles": "".join(tiles), "factors": factor_rows(model), "legend": legend, "panel1": panel1, "panel2": panel2,
        "grouptab": group_table(model), "kindtab": kind_table(model),
        "simcard": sim_card(model, tchart), "checks": checks_list(model),
        "planid": esc("of launch %d" % model["launch"]),
        "splits": "".join(splits_html), "n_dev": cnt["device"], "n_abs": cnt["absorbed"], "n_groups": n_groups, "n_noop": cnt["noop"] + cnt.get("skipped", 0) + cnt.get("empty", 0),
        "n_cpu": n_cpu, "n_in": (len(sched["splits"][-1]["inputs"]) if sched and sched.get("splits") else len(inputs)), "n_rec": len(model["records"]), "n_launch": 1,
        "dyn": ('<span class="pill">plan rebuilt for every launch: %s</span>' % esc("; ".join(dyn))) if dyn else "",
        "cpurows": cpu_rows, "lgtab": launch_table(model), "spm": spm_note(model),
        "refnote": "All refusals with their reasons are in cpu_fallback.json.",
        "cutnote": ("In this graph every list ends at the end of its split." if all(l["end"].startswith("end of the graph") for l in ln) else
                    "Lists of this graph end because: %s." % esc("; ".join(sorted({l["end"] for l in ln})))),
        "overview": overview, "stagebtns": stage_btns, "layerblocks": layer_blocks,
        "legendg": "".join('<span><i class="sw g-%s"></i>%s</span>' % (g, esc(M.GROUP_LABEL[g])) for g in M.GROUPS),
        "tl": tl_html, "grpopts": "".join('<option value="%s">%s</option>' % (g, esc(M.GROUP_LABEL[g])) for g in M.GROUPS), "rectab": records_table(model),
        "commands": ("<p><b>Rerun.</b></p><pre class=\"mono scroll\">%s</pre>" % esc(commands)) if commands else "",
        "mvmshare": pct(1 - s["G"]) if "G" in s else "–", "empty": pct(s.get("empty", 0)), "wait": pct(s.get("wait", 0)), "ideal": fnum(s["ideal_cycles"]), "mvm": fnum(s["mvm_cycles"]),
        "data": data_json,
    }
    with open(out_path, "w") as fh:
        fh.write(doc)
    return len(doc)


def spm_note(model):
    """The SPM layout of the executor for this list (SW §7.3) and the largest tile plan of a record."""
    spm = model.get("spm_lines") or 65536
    lists = model["graph"]["lists"]
    pre = lists[0]["pre"][1] if lists else 0
    one = 1024
    ring = spm - 16 - 4096 - 256 - one
    big = max(model["records"], key=lambda r: r["scratch"]) if model["records"] else None
    txt = ("<p><b>SPM</b> (%s lines of 64 B = %.1f MiB): runtime arguments 16 lines, op-list area 4,096, preload area up to 256 (%d in this list), one-tile block %s, "
           "scratch ring %s lines (the planning budget of every record's tiles)." % (fnum(spm), spm * 64 / 2 ** 20, pre, fnum(one), fnum(ring)))
    if big is not None and big["scratch"]:
        txt += " Largest tile plan: %s, %d tile(s) of %s lines (%.0f %% of the ring)." % (esc("%s %s" % (big["kn"], big["name"])), big["tiles"], fnum(big["scratch"]), 100.0 * big["scratch"] / ring)
    return txt + "</p>"


def launch_table(model):
    rows = []
    for g in model.get("launch_groups", []):
        ks = g["launches"]
        what = "decode token" if g["decode"] else (("warm-up pass of %d tokens" if ks[0] == 0 else "prompt of %d tokens") % g["tokens"])
        nums = ("%d" % ks[0]) if len(ks) == 1 else ("%d to %d" % (ks[0], ks[-1]))
        cyc = g["cycles"]
        rows.append("<tr><td>%s</td><td>%s</td><td class=\"num\">%d</td><td class=\"num\">%d</td><td class=\"num\">%s</td><td>%s</td><td>%s</td></tr>" % (
            esc(nums), esc(what), len(ks), g["n_ops"], (fnum(min(cyc)) if min(cyc) == max(cyc) else fnum(min(cyc)) + " to " + fnum(max(cyc))), esc(g["end"]),
            ("built the plan once, then %d launches from the plan cache" % g["hits"]) if g["hits"] else "built the plan"))
    return ('<div class="scroll"><table><thead><tr><th>Launches</th><th>What</th><th class="num">Count</th><th class="num">Records</th><th class="num">PERF cycles</th><th>List ended by</th>'
            '<th>Plan</th></tr></thead><tbody>%s</tbody></table></div>' % "".join(rows)) if rows else ""


def sim_card(model, tchart):
    if model.get("sim") is None:
        note = model.get("sim_note")
        return '<div class="card"><h3>Card and simulator</h3><p class="note">No simulator run was given%s.</p></div>' % ((": " + esc(note)) if note else "")
    sm = model["sim"]
    s = model["summary"]
    return ('<div class="card"><h3>Card and simulator</h3><div class="scroll"><table><thead><tr><th></th><th class="num">Card</th><th class="num">Simulator</th><th class="num">Sim / card</th></tr></thead><tbody>'
            '<tr><td>Launch span (cycles)</td><td class="num">%s</td><td class="num">%s</td><td class="num">%.4f</td></tr>'
            '<tr><td>PERF_CYCLES of the launch</td><td class="num">%s</td><td class="num">%s</td><td class="num">%.4f</td></tr>'
            '<tr><td>MVM_CYCLES</td><td class="num">%s</td><td class="num">%s</td><td class="num">%.4f</td></tr>'
            '<tr><td>VU_CYCLES</td><td class="num">%s</td><td class="num">%s</td><td class="num">%.4f</td></tr>'
            '<tr><td>DMA_CYCLES</td><td class="num">%s</td><td class="num">%s</td><td class="num">%.4f</td></tr>'
            '<tr><td>QACT_CYCLES</td><td class="num">%s</td><td class="num">%s</td><td class="num">%.4f</td></tr></tbody></table></div>'
            '<p class="note">The simulator ran the same model and prompt with <code>TSR_SIM_CARD=k4full</code> (T_MAX 4, CAPS 0x21, the card image\'s VU lanes and register stages, clk_rv at half the core clock): %s. Per-kind and per-layer simulator cycles are in the tables above.</p>%s</div>' % (
        fnum(s["span"]), fnum(sm["span"]), sm["span"] / s["span"], fnum(s["cycles"]), fnum(sm["cycles"]), sm["cycles"] / s["cycles"],
        fnum(s["perf"].get("MVM_CYCLES", 0)), fnum(sm["perf"].get("MVM_CYCLES", 0)), sm["perf"].get("MVM_CYCLES", 0) / max(s["perf"].get("MVM_CYCLES", 1), 1),
        fnum(s["perf"].get("VU_CYCLES", 0)), fnum(sm["perf"].get("VU_CYCLES", 0)), sm["perf"].get("VU_CYCLES", 0) / max(s["perf"].get("VU_CYCLES", 1), 1),
        fnum(s["perf"].get("DMA_CYCLES", 0)), fnum(sm["perf"].get("DMA_CYCLES", 0)), sm["perf"].get("DMA_CYCLES", 0) / max(s["perf"].get("DMA_CYCLES", 1), 1),
        fnum(s["perf"].get("QACT_CYCLES", 0)), fnum(sm["perf"].get("QACT_CYCLES", 0)), sm["perf"].get("QACT_CYCLES", 0) / max(s["perf"].get("QACT_CYCLES", 1), 1),
        esc(model["run"]["sim"]["description"][:120]) if model["run"].get("sim") else "",
        ('<h3 style="margin-top:12px">Cycles of every decode token</h3><div class="scroll">%s</div><p class="note">Launch cycles (PERF_CYCLES) of the captured decode tokens, card against simulator; the axis does not start at zero.</p>' % tchart) if tchart else ""))
