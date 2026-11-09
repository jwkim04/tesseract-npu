"""prof_graph.py: the compiled graph of the report as layered SVG (no graph library): boxes for the device records and the
fusion groups, edges for the data flow, stubs for what comes from or goes to another layer, an overview of the layers.
Standard library only."""
import collections
import html
import re

import prof_model as M

BOX_W = 244
ROW_H = 14
GAP_X = 46
GAP_Y = 12
STUB_W = 150


def esc(s):
    return html.escape(str(s), quote=True)


def trunc(s, n):
    s = str(s)
    return s if len(s) <= n else s[:n - 1] + "…"


def fmt_cyc(n):
    if n >= 1e6:
        return "%.2f M" % (n / 1e6)
    if n >= 1e3:
        return "%.1f K" % (n / 1e3)
    return "%d" % n


class GraphModel:
    def __init__(self, model):
        g = model["graph"]
        self.nodes = g["nodes"]
        self.groups = g["groups"]
        self.rows = model["records"]
        self.cpu = self._cpu_nodes(g)
        n = len(self.nodes)
        self.group_nodes = collections.defaultdict(list)
        for x in self.nodes:
            if x["grp"] >= 0:
                self.group_nodes[x["grp"]].append(x["i"])
        self.consumers = collections.defaultdict(set)
        for x in self.nodes:
            for s in x["src"]:
                u = self.resolve(s[1])
                if u is not None:
                    self.consumers[u].add(x["i"])

    @staticmethod
    def _cpu_nodes(g):
        out = []
        sc = g.get("sched")
        if sc:
            for sp in sc.get("splits", []):
                if sp["backend"].startswith("CPU"):
                    for nd in sp.get("nodes", []):
                        out.append({"name": nd["name"], "op": nd["op"], "why": "runs in the scheduler's CPU split"})
        # refusals recorded by supports_op: reasons for the nodes that stayed on the CPU
        why = {c["name"]: c["why"] for c in g.get("cpu", [])}
        for o in out:
            if o["name"] in why:
                o["why"] = why[o["name"]]
        return out

    def resolve(self, i):
        """The node that produces the data of node i (a view or reshape passes it on); None when it is not a node."""
        seen = 0
        while i is not None and i >= 0 and self.nodes[i]["place"] == "noop" and seen < 64:
            nxt = None
            for s in self.nodes[i]["src"]:
                if s[1] >= 0:
                    nxt = s[1]
                    break
            if nxt is None:
                return None
            i = nxt
            seen += 1
        return i if i is not None and i >= 0 else None

    def sn_key(self, i):
        x = self.nodes[i]
        if x["grp"] >= 0:
            return ("g", x["grp"])
        return ("n", i)

    def head_of(self, key):
        if key[0] == "g":
            return self.groups[key[1]]["head"]
        return key[1]


def layer_label(l):
    return {"in": "in", "out": "out"}.get(l, "L%s" % l)


def build_layer(gm, layer):
    """Supernodes (a record's node, or a fusion group), edges and stubs of one layer."""
    nodes = gm.nodes
    keys = []
    seen = set()
    for x in nodes:
        if x["place"] in ("noop", "skipped", "empty"):
            continue
        k = gm.sn_key(x["i"])
        head = gm.head_of(k)
        if nodes[head]["layer"] != layer:
            continue
        if k not in seen:
            seen.add(k)
            keys.append(k)
    sns = []
    sn_index = {}
    for k in keys:
        members = [k[1]] if k[0] == "n" else sorted(gm.group_nodes[k[1]])
        head = gm.head_of(k)
        hn = nodes[head]
        recs = []
        for m in members:
            recs += nodes[m]["recs"]
        recs = sorted(set(recs))
        cost = sum(gm.rows[r]["cost"] for r in recs)
        ideal = sum(gm.rows[r]["ideal"] for r in recs)
        kinds = []
        for r in recs:
            kn = gm.rows[r]["kn"]
            if kn not in kinds:
                kinds.append(kn)
        sn = {"id": len(sns), "key": k, "members": members, "head": head, "recs": recs, "cost": cost, "ideal": ideal, "kinds": kinds,
              "g": hn["g"], "fused": k[0] == "g", "name": gm.rows[recs[0]]["name"] if recs else hn["name"]}
        for m in members:
            sn_index[m] = sn["id"]
        sns.append(sn)
    edges = {}
    stubs = {}
    for sn in sns:
        for m in sn["members"]:
            for s in nodes[m]["src"]:
                u = gm.resolve(s[1]) if s[1] >= 0 else None
                if u is not None and u in sn_index:
                    a = sn_index[u]
                    if a != sn["id"]:
                        edges[(("n", a), ("n", sn["id"]))] = 1
                    continue
                if u is not None:                                  # a node of another layer
                    lab = "%s (%s)" % (nodes[u]["name"], layer_label(nodes[u]["layer"]))
                    stubs[("in", lab)] = lab
                    edges[(("s", lab), ("n", sn["id"]))] = 1
                elif s[2] == "input":
                    nm = re.sub(r"^[^#]*#|#\d+$", "", s[0])
                    nm = re.sub(r"blk\.\d+\.", "", nm)
                    lab = "host input: " + nm
                    stubs[("in", lab)] = lab
                    edges[(("s", lab), ("n", sn["id"]))] = 1
                elif s[2] == "kv":
                    lab = "KV cache"
                    stubs[("in", lab)] = lab
                    edges[(("s", lab), ("n", sn["id"]))] = 1
    # outputs to other layers
    for sn in sns:
        outs = set()
        for m in sn["members"]:
            for c in gm.consumers.get(m, ()):
                if nodes[c]["layer"] != layer and gm.sn_key(c) != sn["key"]:
                    outs.add(layer_label(nodes[c]["layer"]))
        for o in sorted(outs):
            lab = "to " + o
            stubs[("out", lab)] = lab
            edges[(("n", sn["id"]), ("s", lab))] = 1
    return sns, list(edges), stubs


def sn_lines(gm, sn):
    """The text rows of a box: kind, name, shape or members, cycles."""
    rows = gm.rows
    kinds = "+".join(sn["kinds"]) if sn["kinds"] else gm.nodes[sn["head"]]["op"]
    l1 = trunc(kinds, 38)
    l2 = trunc(sn["name"], 38)
    if sn["fused"] and len(sn["members"]) > 2:
        l3 = trunc(" · ".join(gm.nodes[m]["op"] for m in sn["members"]), 40)
    elif sn["recs"]:
        l3 = trunc(rows[sn["recs"][0]]["shape"] or gm.nodes[sn["head"]]["op"], 40)
    else:
        l3 = gm.nodes[sn["head"]]["op"]
    if sn["recs"]:
        l4 = "%s cyc" % fmt_cyc(sn["cost"])
        if sn["ideal"] and sn["cost"]:
            l4 += " · %d %% of ideal" % round(100.0 * sn["ideal"] / sn["cost"])
        l4 += " · %d rec" % len(sn["recs"])
    else:
        l4 = "no record"
    return [l1, l2, l3, l4]


def layout(sns, edges, stubs):
    """Longest-path ranks, barycentre order, columns top to bottom. Returns positions {('n', id): (x, y, w, h)}."""
    ids = [("n", s["id"]) for s in sns] + [("s", lab) for (d, lab) in stubs]
    out_edges = collections.defaultdict(list)
    in_edges = collections.defaultdict(list)
    for a, b in edges:
        out_edges[a].append(b)
        in_edges[b].append(a)
    rank = {}
    state = {}

    def visit(v):
        if v in rank:
            return rank[v]
        if state.get(v) == 1:                 # a cycle: ignore the back edge
            return 0
        state[v] = 1
        r = 0
        for u in in_edges[v]:
            if u[0] == "s" and u[1].startswith("to "):
                continue
            r = max(r, visit(u) + 1)
        state[v] = 2
        rank[v] = r
        return r

    for v in ids:
        if v[0] == "s" and v[1].startswith("to "):
            continue
        visit(v)
    maxr = max(rank.values()) if rank else 0
    for v in ids:                              # output stubs one column right of their producer
        if v[0] == "s" and v[1].startswith("to "):
            rank[v] = max((rank.get(u, 0) for u in in_edges[v]), default=0) + 1
    # input stubs sit just before their first consumer, not all in column 0
    for v in ids:
        if v[0] == "s" and not v[1].startswith("to "):
            cons = [rank[c] for c in out_edges[v] if c in rank]
            rank[v] = max(0, (min(cons) - 1) if cons else 0)
    cols = collections.defaultdict(list)
    for v in ids:
        cols[rank[v]].append(v)
    order = {}
    for r in sorted(cols):
        for k, v in enumerate(sorted(cols[r], key=lambda v: (v[0] != "n", v[1] if v[0] == "n" else 0))):
            order[v] = float(k)

    def bary(v, nbrs):
        p = [order[u] for u in nbrs if u in order]
        return sum(p) / len(p) if p else order[v]

    for it in range(6):
        rs = sorted(cols)
        seq = rs if it % 2 == 0 else rs[::-1]
        for r in seq:
            ref = in_edges if it % 2 == 0 else out_edges
            cols[r].sort(key=lambda v: (bary(v, ref[v]), order[v]))
            for k, v in enumerate(cols[r]):
                order[v] = float(k)
    return rank, cols, order


def render_layer(gm, layer, kmax_cost):
    sns, edges, stubs = build_layer(gm, layer)
    cpu_boxes = []
    if layer == "in":
        for c in gm.cpu:
            cpu_boxes.append(c)
    if not sns and not cpu_boxes:
        return None
    rank, cols, order = layout(sns, edges, stubs)
    lines = {sn["id"]: sn_lines(gm, sn) for sn in sns}

    def height(v):
        if v[0] == "s":
            return 22
        return 12 + ROW_H * 4

    def width(v):
        return STUB_W if v[0] == "s" else BOX_W
    # the ranks are wrapped into bands of BAND columns, so that a long chain does not make a page-wide drawing
    BAND = 6
    ranks = sorted(cols)
    colx = {}
    for r in ranks:
        colx[r] = 10 + (r % BAND) * (BOX_W + GAP_X)
    bands = collections.defaultdict(list)
    for r in ranks:
        bands[r // BAND].append(r)
    pos = {}
    band_y, band_h = {}, {}
    y_cursor = 10
    for bi in sorted(bands):
        colh = {}
        for r in bands[bi]:
            y = 0
            for v in cols[r]:
                pos[v] = [colx[r], y, width(v), height(v)]
                y += height(v) + GAP_Y
            colh[r] = y - GAP_Y
        hmax = max(colh.values())
        for r in bands[bi]:
            off = (hmax - colh[r]) / 2.0
            for v in cols[r]:
                pos[v][1] += y_cursor + off
        band_y[bi] = y_cursor
        band_h[bi] = hmax
        y_cursor += hmax + 44
    total_h = y_cursor - 44 + 10
    band_of = {v: rank[v] // BAND for v in rank}
    cpu_h = 0
    if cpu_boxes:                              # the CPU split's nodes below the device boxes
        y0 = total_h + 20
        for k, c in enumerate(cpu_boxes):
            pos[("c", k)] = [10 + (k % 3) * (BOX_W + GAP_X), y0 + (k // 3) * (12 + ROW_H * 4 + GAP_Y), BOX_W, 12 + ROW_H * 4]
        cpu_h = y0 + ((len(cpu_boxes) + 2) // 3) * (12 + ROW_H * 4 + GAP_Y)
    W = max([p[0] + p[2] for p in pos.values()] + [200]) + 10
    H = max(total_h + 10, cpu_h + 10)
    svg = ['<svg data-layer="%s" width="%d" height="%d" viewBox="0 0 %d %d" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Graph of layer %s">' %
           (esc(layer), W, H, W, H, esc(layer))]
    # edges first
    for a, b in edges:
        if a not in pos or b not in pos:
            continue
        pa, pb = pos[a], pos[b]
        x1, y1 = pa[0] + pa[2], pa[1] + pa[3] / 2.0
        x2, y2 = pb[0], pb[1] + pb[3] / 2.0
        if x2 > x1 + 4:
            dx = max(20, (x2 - x1) / 2.0)
            d = "M%.1f %.1f C%.1f %.1f %.1f %.1f %.1f %.1f" % (x1, y1, x1 + dx, y1, x2 - dx, y2, x2, y2)
        else:                                  # the next band: out to the right, down between the bands, in from the left
            yg = band_y[band_of[a]] + band_h[band_of[a]] + 22
            d = "M%.1f %.1f H%.1f V%.1f H%.1f V%.1f H%.1f" % (x1, y1, x1 + 16, yg, x2 - 16, y2, x2)
        svg.append('<path class="ge" data-a="%s" data-b="%s" d="%s"/>' % (esc(a[0] + str(a[1])), esc(b[0] + str(b[1])), d))
    for (d, lab) in stubs:
        v = ("s", lab)
        px, py, pw, ph = pos[v]
        svg.append('<g class="gx"><rect x="%.1f" y="%.1f" width="%d" height="%d" rx="11"/><text x="%.1f" y="%.1f">%s</text></g>' %
                   (px, py, pw, ph, px + 8, py + 15, esc(trunc(lab, 24))))
    for sn in sns:
        px, py, pw, ph = pos[("n", sn["id"])]
        ln = lines[sn["id"]]
        cls = "gn g-%s%s" % (sn["g"], " fused" if sn["fused"] else "")
        svg.append('<g class="%s" data-sn="%d" data-nodes="%s" data-rec="%s" transform="translate(%.1f %.1f)">' %
                   (cls, sn["id"], " ".join(str(m) for m in sn["members"]), " ".join(str(r) for r in sn["recs"]), px, py))
        svg.append('<rect class="box" width="%d" height="%d" rx="6"/><rect class="bar" width="5" height="%d" rx="2" style="fill:var(--g-%s)"/>' % (pw, ph, ph, sn["g"]))
        svg.append('<text class="k" x="12" y="%d">%s</text>' % (ROW_H, esc(ln[0])))
        svg.append('<text x="12" y="%d">%s</text>' % (2 * ROW_H, esc(ln[1])))
        svg.append('<text class="m" x="12" y="%d">%s</text>' % (3 * ROW_H, esc(ln[2])))
        svg.append('<text class="m" x="12" y="%d">%s</text>' % (4 * ROW_H, esc(ln[3])))
        svg.append('<title>%s</title></g>' % esc("%s: %s" % (ln[0], gm.rows[sn["recs"][0]]["name"] if sn["recs"] else gm.nodes[sn["head"]]["name"])))
    for k, c in enumerate(cpu_boxes):
        px, py, pw, ph = pos[("c", k)]
        svg.append('<g class="gn cpu g-head" data-cpu="%d" transform="translate(%.1f %.1f)"><rect class="box" width="%d" height="%d" rx="6"/>'
                   '<rect class="bar" width="5" height="%d" rx="2" style="fill:var(--crit)"/><text class="k" x="12" y="%d">CPU fallback · %s</text><text x="12" y="%d">%s</text>'
                   '<text class="m" x="12" y="%d">%s</text></g>' %
                   (k, px, py, pw, ph, ph, ROW_H, esc(c["op"]), 2 * ROW_H, esc(trunc(c["name"], 38)), 3 * ROW_H, esc(trunc(c["why"], 40))))
    svg.append("</svg>")
    return "".join(svg), {"n_boxes": len(sns), "n_stubs": len(stubs), "n_cpu": len(cpu_boxes), "w": W, "h": H}


def render_overview(gm, model):
    """The layers as boxes in rows: a bar of the group cycles, the cycles and the share of ideal."""
    layers = model["layers"]
    per_row = 8
    bw, bh, gx, gy = 128, 74, 14, 16
    rows_n = (len(layers) + per_row - 1) // per_row
    W = per_row * (bw + gx) + 10
    H = rows_n * (bh + gy) + 10
    tot_max = max((sum(g["cycles"] for g in L["groups"].values()) for L in layers if L["layer"] not in ("in", "out")), default=1)
    out = ['<svg width="%d" height="%d" viewBox="0 0 %d %d" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Layers of the graph">' % (W, H, W, H)]
    prev = None
    for k, L in enumerate(layers):
        x = 5 + (k % per_row) * (bw + gx)
        y = 5 + (k // per_row) * (bh + gy)
        tot = sum(g["cycles"] for g in L["groups"].values())
        ide = sum(g["ideal"] for g in L["groups"].values())
        if prev is not None:
            px, py = prev
            if py == y:
                out.append('<path class="ge" d="M%.1f %.1f L%.1f %.1f"/>' % (px + bw, py + bh / 2, x, y + bh / 2))
            else:
                out.append('<path class="ge" d="M%.1f %.1f C%.1f %.1f %.1f %.1f %.1f %.1f"/>' % (px + bw / 2, py + bh, px + bw / 2, py + bh + 10, x + bw / 2, y - 10, x + bw / 2, y))
        prev = (x, y)
        out.append('<g class="stage" data-layer="%s" transform="translate(%d %d)"><rect class="box" width="%d" height="%d" rx="6"/>' % (esc(L["layer"]), x, y, bw, bh))
        out.append('<text x="8" y="16" style="font-weight:600">%s</text>' % esc("Layer %s" % L["layer"] if L["layer"] not in ("in", "out") else {"in": "Embedding", "out": "Head"}[L["layer"]]))
        out.append('<text class="lbl" x="%d" y="16" text-anchor="end">%s cyc</text>' % (bw - 8, fmt_cyc(tot)))
        cx = 8.0
        for g in M.GROUPS:
            w = (bw - 16) * L["groups"][g]["cycles"] / tot if tot else 0
            if w > 0:
                out.append('<rect class="seg g-%s" x="%.1f" y="24" width="%.1f" height="10" style="fill:var(--g-%s)"/>' % (g, cx, max(w - 1, 0.5), g))
                cx += w
        out.append('<text class="lbl" x="8" y="52">%d records</text>' % sum(g["n"] for g in L["groups"].values()))
        if ide and tot:
            out.append('<text class="lbl" x="8" y="66">%d %% of ideal</text>' % round(100 * ide / tot))
        out.append('<title>%s</title></g>' % esc("%s: %s cycles, %s" % (L["layer"], "{:,}".format(tot), ", ".join("%s %s" % (g, fmt_cyc(L["groups"][g]["cycles"])) for g in M.GROUPS if L["groups"][g]["cycles"]))))
    out.append("</svg>")
    return "".join(out)
