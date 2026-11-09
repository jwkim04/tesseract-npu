"""prof_model.py: the analysis of a profile dump (docs/prof-dbg.md "The profile report"): layers and kind groups of the ggml
nodes, the record table with measured and ideal cycles, the per-kind and per-layer tables, the utilisation split and the
sanity checks. Standard library only."""
import collections
import os
import re
import struct

import prof_lib as P

GROUPS = ["attention", "router", "experts", "dense", "residual", "head"]
GROUP_LABEL = {"attention": "attention", "router": "router", "experts": "experts", "dense": "dense branch",
               "residual": "residual and norms", "head": "embedding and head"}
B_M0 = 72.05e9                  # four channels, read bursts of 64 beats at 300 MHz (docs/performance.md "Memory bandwidth (M0)")
STREAM_B_PER_CYCLE = 256.0      # four channels x 64 B per core cycle
HANDLE_NONE = 0xFFFFFFFF
REGIONS = [(0x0, 0x4000000, "OPL"), (0x4000000, 0x14000000, "IO"), (0x14000000, 0x20000000, "AUX"), (0x20000000, 1 << 62, "WEIGHT/KV heap")]

_SUFFIX = re.compile(r"\s*\((?:reshaped|view|permuted|transposed|copy|[^)]*)\)\s*$")
_LAYER = re.compile(r"-(\d+)$")


def clean(name):
    return _SUFFIX.sub("", name)


def region_of(off):
    for lo, hi, n in REGIONS:
        if lo <= off < hi:
            return n
    return "?"


# ---------------------------------------------------------------------------------------------
# layers and groups of the ggml nodes
# ---------------------------------------------------------------------------------------------
_ATTN = re.compile(r"^(Qcur|Kcur|Vcur|attn_|q_|k_|v_|kq|kqv|cache_|inp_attn|attn$)")
_ROUTER = re.compile(r"^ffn_moe_(logits|probs|argsort|topk|weights)")
_EXPERTS = re.compile(r"^(ffn_moe_(gate_up|gate|up|geglu|swiglu|silu|down|weighted|out)|ffn_norm_2|ffn_moe-|ffn_moe_down_scaled|ffn_moe_gate_up_exps)")
_DENSE = re.compile(r"^(ffn_norm_1|ffn_gate|ffn_up|ffn_down|ffn_geglu|ffn_swiglu|ffn_mlp|ffn_act|ffn_norm$|ffn_inp)")
_RESID = re.compile(r"^(ffn_moe_combined|ffn_post_norm|l_out|ffn_out|sa_out|post_ffn)")
_HEAD = re.compile(r"^(result_|output|inp_scaled|inp_embd|inp_out_ids)")


def layer_of(nodes):
    """Layer of each node: 'in' (before the first layer), an int, or 'out' (after the last l_out)."""
    n = len(nodes)
    names = [clean(x["name"]) for x in nodes]
    last = -1
    for i, nm in enumerate(names):
        if re.match(r"^l_out-\d+$", nm):
            last = i
    first_layered = next((i for i, nm in enumerate(names) if _LAYER.search(nm) and not nm.startswith(("inp", "result"))), n)
    out = []
    cur = "in"
    for i, nm in enumerate(names):
        m = _LAYER.search(nm)
        if i < first_layered:
            cur = "in"
        elif m and not nm.startswith(("result", "inp")):
            cur = int(m.group(1))
        if last >= 0 and i > last:
            cur = "out"
        if nm.startswith("result_"):
            cur = "out"
        out.append(cur)
    return out


def group_of(nodes, layers):
    """Kind group of each node by weights, op and name, then filled from the previous node of the layer."""
    n = len(nodes)
    g = [None] * n
    names = [clean(x["name"]) for x in nodes]
    for i, x in enumerate(nodes):
        nm = names[i]
        wn = " ".join(s["name"] for s in x["src"] if s["role"] == "weight")
        if layers[i] in ("in", "out"):
            g[i] = "head"
            continue
        if x["op"] == "MUL_MAT_ID":
            g[i] = "experts"
        elif x["op"] == "FLASH_ATTN" or "cache_" in " ".join(s["name"] for s in x["src"]):
            g[i] = "attention"
        elif "ffn_gate_inp" in wn:
            g[i] = "router"
        elif ".attn_" in wn:
            g[i] = "attention"
        elif "_exps" in wn:
            g[i] = "experts"
        elif re.search(r"\.ffn_(gate|up|down)\.weight", wn):
            g[i] = "dense"
        elif _ROUTER.match(nm):
            g[i] = "router"
        elif _EXPERTS.match(nm):
            g[i] = "experts"
        elif _DENSE.match(nm):
            g[i] = "dense"
        elif _RESID.match(nm):
            g[i] = "residual"
        elif _ATTN.match(nm):
            g[i] = "attention"
    # a fusion group takes the class of its named member
    by_group = collections.defaultdict(list)
    for i, x in enumerate(nodes):
        if x.get("group", -1) >= 0:
            by_group[x["group"]].append(i)
    for members in by_group.values():
        cls = [g[i] for i in members if g[i] is not None]
        if cls:
            c = collections.Counter(cls).most_common(1)[0][0]
            for i in members:
                if g[i] is None:
                    g[i] = c
    # the nodes between attn_out and ffn_moe_logits prepare the router input (Gemma 4: RMS_NORM, SCALE, MUL)
    has_router = {l for i, l in enumerate(layers) if g[i] == "router"}
    for i in range(n):
        if g[i] is None and layers[i] in has_router:
            # previous classified node of the layer
            j = i - 1
            while j >= 0 and (g[j] is None or layers[j] != layers[i]):
                j -= 1
            if j >= 0 and names[j].startswith("attn_out"):
                g[i] = "router"
    last = None
    for i in range(n):
        if g[i] is None:
            g[i] = last if last is not None and layers[i] == layers[i - 1] else "residual"
        last = g[i]
    return g


# ---------------------------------------------------------------------------------------------
# records: decoded arguments, shapes
# ---------------------------------------------------------------------------------------------
def f32_of(bits):
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


def fmt_handle(v):
    if v == HANDLE_NONE:
        return "none"
    ch, beat = v >> 28, v & 0x0FFFFFFF
    return "ch%d+0x%x" % (ch, beat * 64)


def decode_args(spec, args):
    out = []
    for k, s in enumerate(spec):
        name, _, t = s.partition(":")
        v = args[k] if k < len(args) else 0
        if t == "h":
            disp = fmt_handle(v)
        elif t == "f":
            disp = "%.6g" % f32_of(v)
        elif t == "x":
            disp = "0x%x" % v
        elif t == "b":
            disp = "%d beats (0x%x B)" % (v, v * 64)
        elif t == "p":
            disp = "preload +%d B" % v
        elif t == "s":
            disp = "%d lines" % v
        else:
            disp = "%d" % v
        out.append([name, disp])
    return out


def shape_of(kind_name, a):
    """A one-line shape of a record from its argument words (index = the namespaces of tsr/opl.hpp)."""
    try:
        if kind_name in ("MUL_MAT", "MUL_MAT_SCALE", "MUL_MAT_SOFTCAP"):
            return "N=%d K=%d T=%d %s" % (a[6], a[7], a[8], "Q8_0" if a[5] == 0 else "Q4_K")
        if kind_name == "MUL_MAT_ID":
            return "N=%d K=%d T=%d used=%d of %d %s" % (a[10], a[11], a[12], a[13], a[14], "Q8_0" if a[9] == 0 else "Q4_K")
        if kind_name in ("ADD", "MUL"):
            return "%d x %d" % (a[6], a[7])
        if kind_name in ("SCALE", "UNARY", "SOFTPLUS"):
            return "%d x %d" % (a[4], a[5])
        if kind_name == "RMS_NORM":
            return "%d x %d" % (a[4], a[5])
        if kind_name == "GLU":
            return "%d x %d" % (a[6], a[7])
        if kind_name == "ROPE":
            return "head %d heads %d tokens %d" % (a[9], a[11], a[12])
        if kind_name == "GET_ROWS":
            return "%d rows of %d lines" % (a[5], a[6])
        if kind_name == "SET_ROWS_KV":
            return "%d rows x %d" % (a[5], a[6])
        if kind_name == "FLASH_ATTN":
            return "kv %d d %d heads %d/%d tokens %d" % (a[7], a[8], a[9], a[10], a[11])
        if kind_name == "CPY":
            return "%d rows of %d lines" % (a[5], a[4])
        if kind_name == "MOE_ROUTE":
            return "%d experts, top %d, T=%d" % (a[5], a[6], a[7])
        if kind_name == "SSM_CONV":
            return "%d ch, d_conv %d, T=%d" % (a[7], a[8], a[9])
        if kind_name == "GATED_DELTA_NET":
            return "S_v %d heads %d tokens %d" % (a[16], a[18], a[19])
    except IndexError:
        pass
    return ""


def range_str(r):
    ch, lo, hi, row, stride = r
    n = hi - lo if not (row and stride and row < stride) else ((hi - lo - row) // stride + 1) * row
    return {"ch": ch, "off": lo, "bytes": n, "region": region_of(lo), "rows": [row, stride] if row else None}


# ---------------------------------------------------------------------------------------------
# one launch of one run
# ---------------------------------------------------------------------------------------------
class LaunchView:
    """A launch joined to its plan: records with ops and cycles."""

    def __init__(self, run, k, tag):
        self.run, self.k, self.tag = run, k, tag
        self.entry, self.tl = run.launch(k)
        self.plan = run.plans[self.entry["plan"]]
        self.list = self.plan.lists[self.entry["list"]]
        self.recs = self.list["recs"]
        self.ok = self.tl is not None
        self.perf = P.perf_dict(self.entry["perf"], run.meta["num_ch"]) if self.entry.get("perf") else {}
        self.f = run.core_hz or 276.3e6
        if not self.ok:
            return
        tl = self.tl
        self.t0 = tl.t_begin
        by_idx, pre = P.join_launch(self.plan, self.entry["list"], tl)
        self.by_idx, self.pre = by_idx, pre
        order = list(range(len(self.recs)))
        self.cost, self.setup, self.tail = P.frontier_costs(order, by_idx, pre, tl.t_begin, tl.t_end)
        self.span = tl.t_end - tl.t_begin
        self.markers = len(tl.nodes)
        # tag ops: record index, ordinal within (record, unit), ideal cycles of an MVM op (the record's, split evenly)
        for i in order:
            t_mark, ops = by_idx.get(i, (None, []))
            cnt = collections.Counter()
            mvm = [o for o in ops if o.unit == 2 and o.opcode == 0x20]
            ideal = self.recs[i].get("ideal", 0.0)
            for o in ops:
                o.rec = i
                o.ordinal = cnt[o.unit]
                cnt[o.unit] += 1
                if o in mvm and mvm:
                    o.ideal = ideal / len(mvm)
        for o in pre:
            o.rec = -1
        self.unassigned = sum(1 for o in tl.ops if o.rec < 0)

    def rec_stats(self, i):
        t_mark, ops = self.by_idx.get(i, (None, []))
        if not ops:
            return {"cost": 0, "n": 0, "first": None, "last": None, "busy": {}}
        busy = collections.Counter()
        for o in ops:
            busy[P.UNITS[o.unit] if o.unit < len(P.UNITS) else str(o.unit)] += o.run if o.ran else 0
        return {"cost": self.cost.get(i, 0), "n": len(ops), "first": min(o.t_start for o in ops) - self.t0,
                "last": max(o.tend for o in ops) - self.t0, "busy": dict(busy), "mark": (t_mark - self.t0) if t_mark else None}


# ---------------------------------------------------------------------------------------------
# the report model
# ---------------------------------------------------------------------------------------------
def kind_name_of(meta, kind):
    k = meta["kinds"].get(str(kind))
    return k["name"] if k else "kind%d" % kind


def build_model(card, sim=None, launch=None, expect_cycles=None, ref_toks=None, title=None, model_name=None, sched=None, ref_run=None):
    """The data of the report: dict of plain values."""
    run = card
    dec = run.decode_launches()
    dec = [k for k in dec if run.launch(k)[1] is not None]
    if not dec:
        raise SystemExit("prof_report: no decode launch with a trace in %s" % run.path)
    k = launch if launch is not None else dec[len(dec) // 2] if len(dec) > 1 else dec[0]
    cv = LaunchView(run, k, "card")
    sv = None
    sim_note = None
    if sim is not None:
        ks = k
        if ks - sim.trw_offset >= len(sim.trw) or ks - sim.trw_offset < 0 or sim.launch_log[ks]["plan"] is None:
            sim_note = "the simulator's trace has no launch %d" % ks
        else:
            sv = LaunchView(sim, ks, "sim")
            if [r["kind"] for r in sv.recs] != [r["kind"] for r in cv.recs]:
                sim_note = "the simulator's op list differs from the card's (record kinds): not compared"
                sv = None
    meta = run.meta
    nodes = cv.plan.nodes
    layers = layer_of(nodes)
    groups = group_of(nodes, layers)
    kspec = {int(i): v for i, v in meta["kinds"].items()}
    recs = cv.recs
    # group of each record: its node's; JOINs follow the record before them
    rgroup, rlayer = [], []
    for i, r in enumerate(recs):
        nd = r["node"]
        if nd >= 0:
            rgroup.append(groups[nd])
            rlayer.append(layers[nd])
        else:
            rgroup.append(rgroup[-1] if rgroup else "head")
            rlayer.append(rlayer[-1] if rlayer else "in")
    C = meta["num_ch"]
    f = cv.f
    out = {"title": title or "Decode token profile", "model": model_name or "", "launch": k}
    # ---- records
    rows = []
    for i, r in enumerate(recs):
        kn = kind_name_of(meta, r["kind"])
        spec = kspec.get(r["kind"], {}).get("args", [])
        st = cv.rec_stats(i)
        rd_b, wr_b = r["rd_bytes"], r["wr_bytes"]
        if kn == "SET_ROWS_KV" and len(r["a"]) >= 7:           # K or V rows into the Q8_0 plane: the plane extents are not the traffic
            rd_b, wr_b = r["a"][5] * r["a"][6] * 4, r["a"][5] * r["a"][6] * 34 // 32
        row = {"i": i, "kind": r["kind"], "kn": kn, "node": r["node"], "name": r["name"], "layer": rlayer[i], "group": rgroup[i],
               "shape": shape_of(kn, r["a"]), "dep": r["dep"], "tiles": r["tiles"], "tile_rows": r["tile_rows"],
               "scratch": r["scratch_lines"], "rd": rd_b, "wr": wr_b, "w": r["w_bytes"], "ideal": r["ideal"],
               "cost": st["cost"], "n_ops": st["n"], "first": st["first"], "last": st["last"], "busy": st["busy"],
               "args": decode_args(spec, r["a"]),
               "rdr": [range_str(x) for x in r["rd"][:6]], "wrr": [range_str(x) for x in r["wr"][:6]]}
        ops = cv.by_idx.get(i, (None, []))[1]
        if ops:
            lo = max(ops, key=lambda o: o.run if o.ran else 0)
            row["slice"] = "card:r%d.%s%d" % (i, P.UNITS[lo.unit], lo.ordinal)
        if sv is not None:
            ss = sv.rec_stats(i)
            row["sim"] = {"cost": ss["cost"], "n_ops": ss["n"]}
        rows.append(row)
    out["records"] = rows
    # ---- the launch: cycles and PERF
    pf = cv.perf
    cyc = pf.get("CYCLES", 0)
    mvm = pf.get("MVM_CYCLES", 0)
    ideal_total = sum(r["ideal"] for r in rows)
    w_bytes = sum(r["w"] for r in rows)
    rd_model = sum(r["rd"] for r in rows)
    rd_meas = sum(pf.get("CH_RD_BEATS[%d]" % c, 0) for c in range(C)) * 64
    # tokens per second from the host timestamps of consecutive decode launches
    log = run.launch_log
    periods = []
    for a, b in zip(dec, dec[1:]):
        if b == a + 1:
            periods.append((log[b]["t0_ns"] - log[a]["t0_ns"]) * 1e-9)
    periods.sort()
    period = periods[len(periods) // 2] if periods else None
    t_dev = cyc / f if f else 0.0
    walls = sorted(log[x]["wall_ns"] * 1e-9 for x in dec)
    s = {"cycles": cyc, "span": cv.span, "core_hz": f, "t_dev_ms": t_dev * 1e3, "tok_s_dev": (1.0 / t_dev) if t_dev else 0.0,
         "launch_wall_ms": walls[len(walls) // 2] * 1e3 if walls else None, "token_period_ms": period * 1e3 if period else None,
         "tok_s": (1.0 / period) if period else None, "n_decode": len(dec), "ref_toks": ref_toks,
         "mvm_cycles": mvm, "ideal_cycles": ideal_total, "w_bytes": w_bytes, "rd_model": rd_model, "rd_meas": rd_meas,
         "perf": pf, "setup": cv.setup, "tail": cv.tail}
    s["llama_on"] = P.parse_llama_log(os.path.join(run.path, "run.log"))
    if ref_run:
        s["llama_off"] = P.parse_llama_log(os.path.join(ref_run, "run.log"))
        if s["llama_off"]["toks"]:
            s["ref_toks"] = s["llama_off"]["toks"]
    if cyc and mvm and ideal_total and period:
        H = 1.0 - t_dev / period
        G = 1.0 - mvm / cyc
        F = 1.0 - ideal_total / mvm
        U = rd_meas / (B_M0 * period)
        E = U / ((1 - H) * (1 - G) * (1 - F)) if (1 - H) * (1 - G) * (1 - F) > 0 else 0.0
        s.update({"H": H, "G": G, "F": F, "E": E, "U": U, "U_w": w_bytes / (B_M0 * period), "E_clock": STREAM_B_PER_CYCLE * f / B_M0,
                  "empty": pf.get("MVM_EMPTY", 0) / cyc, "wait": pf.get("MVM_WAIT", 0) / cyc})
    out["summary"] = s
    # ---- per kind
    kinds = collections.OrderedDict()
    total_cost = sum(r["cost"] for r in rows)
    for r in rows:
        d = kinds.setdefault(r["kn"], {"kind": r["kn"], "count": 0, "cycles": 0, "ideal": 0.0, "bytes": 0, "sim": 0, "best": None})
        d["count"] += 1
        d["cycles"] += r["cost"]
        d["ideal"] += r["ideal"]
        d["bytes"] += r["rd"] + r["wr"]
        if sv is not None:
            d["sim"] += r["sim"]["cost"]
        if d["best"] is None or r["cost"] > rows[d["best"]]["cost"]:
            d["best"] = r["i"]
    kt = []
    for d in kinds.values():
        d["share"] = d["cycles"] / cv.span if cv.span else 0.0
        d["eff"] = d["ideal"] / d["cycles"] if d["ideal"] and d["cycles"] else None
        d["gbs"] = d["bytes"] / (d["cycles"] / f) / 1e9 if d["cycles"] else None
        d["ratio"] = (d["cycles"] / d["sim"]) if sv is not None and d["sim"] else None
        kt.append(d)
    kt.sort(key=lambda d: -d["cycles"])
    out["kinds"] = kt
    out["other_cycles"] = {"setup": cv.setup, "tail": cv.tail}
    # ---- per layer and group
    lay = collections.OrderedDict()
    for r in rows:
        key = r["layer"]
        ent = lay.setdefault(key, {g: {"cycles": 0, "ideal": 0.0, "sim": 0, "n": 0} for g in GROUPS})
        e = ent[r["group"]]
        e["cycles"] += r["cost"]
        e["ideal"] += r["ideal"]
        e["n"] += 1
        if sv is not None:
            e["sim"] += r["sim"]["cost"]
    def lk(x):
        return (-1 if x == "in" else 10 ** 6 if x == "out" else x)
    out["layers"] = [{"layer": l, "groups": lay[l]} for l in sorted(lay, key=lk)]
    gt = {g: {"cycles": 0, "ideal": 0.0, "sim": 0} for g in GROUPS}
    for r in rows:
        gt[r["group"]]["cycles"] += r["cost"]
        gt[r["group"]]["ideal"] += r["ideal"]
        if sv is not None:
            gt[r["group"]]["sim"] += r["sim"]["cost"]
    out["group_totals"] = gt
    out["groups"] = [{"id": g, "label": GROUP_LABEL[g]} for g in GROUPS]
    # ---- the graph
    out["graph"] = graph_data(cv.plan, nodes, layers, groups, rows, run, sched or (sim.sched if sim is not None else None))
    # ---- the timeline of the token
    out["timeline"] = {"card": timeline_ops(cv), "sim": timeline_ops(sv) if sv is not None else None,
                       "span": cv.span, "sim_span": sv.span if sv is not None else None, "f": f}
    # ---- the simulator against the card
    if sv is not None:
        sp = sv.perf
        out["sim"] = {"cycles": sp.get("CYCLES", 0), "span": sv.span, "ratio": (sv.span / cv.span) if cv.span else None,
                      "cycles_ratio": (sp.get("CYCLES", 0) / cyc) if cyc else None, "perf": sp,
                      "config": sim.meta.get("description", "")}
    out["sim_note"] = sim_note
    # ---- checks
    out["checks"] = make_checks(cv, sv, out, expect_cycles, run, dec)
    out["run"] = {"card": run_info(run, cv, "card"), "sim": run_info(sim, sv, "simulator") if sv is not None else None}
    out["capture"] = run.capture
    out["spm_lines"] = run.meta.get("spm_lines")
    # cycles of every decode launch, card against simulator
    series = []
    for kk in dec:
        e = run.launch_log[kk]
        cyc_k = (e.get("perf") or [0])[0]
        sc_k = None
        if sim is not None and kk < len(sim.launch_log) and sim.launch_log[kk].get("perf"):
            sc_k = sim.launch_log[kk]["perf"][0]
        series.append({"k": kk, "card": cyc_k, "sim": sc_k})
    out["per_token"] = series
    lg = collections.OrderedDict()
    for kk, e in enumerate(log):
        pl = run.plans[e["plan"]]
        g = lg.setdefault((e["plan"], e["list"]), {"plan": e["plan"], "list": e["list"], "launches": [], "cycles": [], "hits": 0, "n_ops": e["n_ops"],
                                                    "end": pl.lists[e["list"]]["end"], "tokens": pl.nodes[0]["ne"][1] if pl.nodes else 0,
                                                    "nodes": len(pl.nodes), "decode": kk in dec})
        g["launches"].append(kk)
        g["cycles"].append((e.get("perf") or [0])[0])
        g["hits"] += 1 if e["cache_hit"] else 0
    out["launch_groups"] = list(lg.values())
    out["launches"] = [{"k": i, "plan": e["plan"], "list": e["list"], "n_ops": e["n_ops"], "cycles": (e.get("perf") or [0])[0],
                        "cache_hit": e["cache_hit"], "wall_ms": e["wall_ns"] * 1e-6, "decode": i in dec} for i, e in enumerate(log)]
    return out


def run_info(run, v, tag):
    m = run.meta
    return {"tag": tag, "dir": run.path, "description": m.get("description", ""), "image_id": m.get("image_id", ""),
            "simulated": m.get("simulated", False), "core_hz": run.core_hz, "launch": v.k if v is not None else None,
            "trw_records": m.get("trw_records"), "trw_lost": m.get("trw_lost"), "n_launches": len(run.launch_log)}


def timeline_ops(v):
    """Compact rows of the ops of a launch, relative to its begin: [rec, unit, start, dur, qwait, hwait, opcode, sub, ideal,
    ordinal, pidx]."""
    rows = []
    for o in v.tl.ops:
        rows.append([o.rec, o.unit, o.t_start - v.t0, (o.run if o.ran else 0), o.qwait, o.hwait, o.opcode, o.sub,
                     round(o.ideal, 2), o.ordinal, o.pidx])
    marks = [[i, t - v.t0] for i, np_, t in v.tl.nodes]
    return {"ops": rows, "marks": marks}


def graph_data(plan, nodes, layers, groups, rows, run, sched):
    nd = []
    for i, x in enumerate(nodes):
        recs = x["recs"]
        cost = sum(rows[r[1]]["cost"] for r in recs if r[0] == 0)
        nd.append({"i": i, "name": x["name"], "op": x["op"], "type": x["type"], "ne": x["ne"], "place": x["place"], "host": x["host"],
                   "grp": x["group"], "layer": layers[i], "g": groups[i], "recs": [r[1] for r in recs if r[0] == 0],
                   "cost": cost, "src": [[s["name"], s["node"], s["role"]] for s in x["src"]]})
    return {"nodes": nd, "groups": plan.groups, "inputs": [[s["name"], s["type"], s["ne"]] for s in plan.inputs],
            "lists": [{"idx": l["idx"], "n_ops": l["n_ops"], "joins": l["n_joins"], "end": l["end"], "blob": l["blob_bytes"],
                       "pre": [l["pre_entries"], l["pre_lines"]]} for l in plan.lists], "dynamic": plan.d["dynamic"],
            "cpu": run.cpu, "sched": sched or run.sched}


def make_checks(cv, sv, out, expect_cycles, run, dec):
    """The sanity checks written into the report: [{ok: True/False/None, text}]."""
    ck = []
    pf = cv.perf
    cyc = pf.get("CYCLES", 0)
    C = run.meta["num_ch"]
    def add(ok, text):
        ck.append({"ok": ok, "text": text})
    span = cv.span
    sc = out["other_cycles"]
    tot = sc["setup"] + sum(r["cost"] for r in out["records"]) + sc["tail"]
    add(tot == span, "Record cycles add up: setup %d + %d record cycles + tail %d = %d, the launch span (LAUNCH_END - LAUNCH_BEGIN) is %d." %
        (sc["setup"], sum(r["cost"] for r in out["records"]), sc["tail"], tot, span))
    if cyc:
        d = cyc - span
        add(abs(d) <= max(0.005 * cyc, 20000), "Launch span against PERF_CYCLES of the launch: %d against %d cycles (the difference of %d cycles is the firmware's "
            "work before LAUNCH_BEGIN and after LAUNCH_END, %.2f %%)." % (span, cyc, d, 100.0 * d / cyc))
    if expect_cycles:
        d = (cyc - expect_cycles) / expect_cycles
        add(abs(d) <= 0.01, "Card total %.3f M cycles against the earlier measurement of %.3f M (%+.2f %%)." % (cyc / 1e6, expect_cycles / 1e6, 100 * d))
    # DB 8.7: the per-unit busy cycles against PERF
    busy = collections.Counter()
    for o in cv.tl.ops:
        if o.ran:
            busy[o.unit] += o.run
    for unit, key in ((0, "DMA_CYCLES"), (1, "QACT_CYCLES"), (3, "VU_CYCLES"), (4, "ATT_CYCLES")):
        if key in pf and (pf[key] or busy[unit]):
            a, b = busy[unit], pf[key]
            pct = 100.0 * (a - b) / b if b else 0
            if unit == 3 and not run.meta.get("simulated") and a >= b:
                add(True, "VU: the sum of RUN is %d, PERF %s is %d (%+.1f %%): on the card a VU op's run time includes its wait for the op before it "
                    "(docs/performance.md, \"Card against simulator\"), so RUN is the larger." % (a, key, b, pct))
            else:
                add(abs(a - b) <= max(0.01 * b, 2000), "%s: the sum of RUN is %d, PERF %s is %d (%+.2f %%)." % (P.UNITS[unit], a, key, b, pct))
    if "MVM_CYCLES" in pf:
        add(busy[2] >= pf["MVM_CYCLES"] * 0.999, "MVM: the sum of RUN is %d, at least PERF MVM_CYCLES %d (ops overlap in the unit)." % (busy[2], pf["MVM_CYCLES"]))
    if "RETIRED" in pf:
        add(len(cv.tl.ops) == pf["RETIRED"], "Ops in the trace: %d, PERF RETIRED: %d." % (len(cv.tl.ops), pf["RETIRED"]))
    add(cv.markers == len(cv.recs),
        "Join: %d NODE_BEGIN markers for %d records; %d ops before the first record (the firmware's launch setup) belong to no record." %
        (cv.markers, len(cv.recs), len(cv.pre)))
    m = run.meta
    add((m.get("trw_lost") or 0) == 0, "Trace writer: %s records, %s dropped or overwritten." % (m.get("trw_records"), m.get("trw_lost")))
    s = out["summary"]
    if s.get("rd_meas"):
        add(None, "DDR reads: PERF counted %.2f GB over the token (CH_RD_BEATS x 64 B), the records' own formulas give %.2f GB (weights %.2f GB)." %
            (s["rd_meas"] / 1e9, s["rd_model"] / 1e9, s["w_bytes"] / 1e9))
    if sv is not None:
        ratio = sv.span / span if span else 0
        add(abs(ratio - 1) <= 0.01, "Simulator against the card: %d against %d cycles for the same launch (%+.2f %%)." % (sv.span, span, 100 * (ratio - 1)))
        nc = collections.Counter(o.unit for o in cv.tl.ops)
        ns = collections.Counter(o.unit for o in sv.tl.ops)
        add(nc == ns, "Op streams: %s on the card, %s in the simulator." % (
            ", ".join("%s %d" % (P.UNITS[u], n) for u, n in sorted(nc.items())), ", ".join("%s %d" % (P.UNITS[u], n) for u, n in sorted(ns.items()))))
    return ck
