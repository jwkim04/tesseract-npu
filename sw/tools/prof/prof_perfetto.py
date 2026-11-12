"""prof_perfetto.py: the trace-event JSON of one launch (Perfetto, chrome://tracing). One process per run (card, simulator), a
track per unit, a RISC-V issue track, a record track, counter tracks from PERF and from the trace, flow arrows record -> op.
Standard library only."""
import json

import prof_lib as P

TID = {"DMA": 1, "QACT": 2, "MVM": 3, "VU": 4, "ATT": 5, "SYNC": 6}
TID_ISSUE, TID_REC = 10, 11


def slice_id(tag, rec, unit, ordinal):
    return "%s:%s.%s%d" % (tag, ("r%d" % rec) if rec >= 0 else "setup", P.UNITS[unit], ordinal)


def _us(cycles, f):
    """Microseconds at 1 ns resolution; a slice's end is _us(end) too, so neighbors that touch in cycles touch in the file."""
    return round(cycles * 1e6 / f, 3)


def _dur(a, b, f):
    return max(round(_us(b, f) - _us(a, f), 3), 0.001)


def _lanes(ops):
    """Greedy lane per op (a unit's ops that overlap in time go to further lanes). ops sorted by start."""
    ends = []
    out = []
    for o in ops:
        for l, e in enumerate(ends):
            if o.t_start >= e:
                ends[l] = max(o.tend, o.t_start)
                out.append(l)
                break
        else:
            ends.append(max(o.tend, o.t_start))
            out.append(len(ends) - 1)
    return out


def view_events(view, rows, pid, name, f, flows=True, flow_base=0):
    """Trace events of one LaunchView. rows: the record rows of the model (name, node, kind, ...)."""
    ev = []
    t0 = view.t0
    meta = view.run.meta
    C = meta["num_ch"]
    ev.append({"ph": "M", "pid": pid, "name": "process_name", "args": {"name": name}})
    ev.append({"ph": "M", "pid": pid, "name": "process_sort_index", "args": {"sort_index": pid}})
    # unit tracks with lanes
    by_unit = {}
    for o in view.tl.ops:
        by_unit.setdefault(o.unit, []).append(o)
    tids = {}
    for u in sorted(by_unit):
        ops = sorted(by_unit[u], key=lambda o: (o.t_start, o.tend))
        lanes = _lanes(ops)
        for o, l in zip(ops, lanes):
            tids[id(o)] = (TID[P.UNITS[u]] * 100 + l) if l else TID[P.UNITS[u]] * 100
        nl = max(lanes) + 1 if lanes else 1
        for l in range(nl):
            tid = TID[P.UNITS[u]] * 100 + l
            ev.append({"ph": "M", "pid": pid, "tid": tid, "name": "thread_name", "args": {"name": P.UNITS[u] + ("" if l == 0 else " (overlap %d)" % l)}})
            ev.append({"ph": "M", "pid": pid, "tid": tid, "name": "thread_sort_index", "args": {"sort_index": TID[P.UNITS[u]] * 10 + l}})
    ev.append({"ph": "M", "pid": pid, "tid": TID_ISSUE, "name": "thread_name", "args": {"name": "RISC-V issue"}})
    ev.append({"ph": "M", "pid": pid, "tid": TID_ISSUE, "name": "thread_sort_index", "args": {"sort_index": 0}})
    # a record's span (first start to last end) overlaps its neighbors': greedy lanes keep the slices properly nested
    spans = []
    for i in range(len(rows)):
        st = view.rec_stats(i)
        if st["n"]:
            spans.append((st["first"], st["last"], i, st))
    spans.sort(key=lambda x: (x[0], -x[1]))
    rec_lane = {}
    lane_end = []
    for a, b, i, st in spans:
        for l, e in enumerate(lane_end):
            if a >= e:
                lane_end[l] = b
                rec_lane[i] = l
                break
        else:
            lane_end.append(b)
            rec_lane[i] = len(lane_end) - 1
    for l in range(max(len(lane_end), 1)):
        tid = TID_REC + l * 10 if l else TID_REC
        ev.append({"ph": "M", "pid": pid, "tid": tid, "name": "thread_name", "args": {"name": "records" if l == 0 else "records (overlap %d)" % l}})
        ev.append({"ph": "M", "pid": pid, "tid": tid, "name": "thread_sort_index", "args": {"sort_index": 1 + l * 0.01}})
    # records and issue slices
    marks = sorted(view.tl.nodes, key=lambda x: x[1])
    mark_t = {i: t for i, np_, t in marks}
    mark_next = {}
    for k, (i, np_, t) in enumerate(marks):
        mark_next[i] = marks[k + 1][2] if k + 1 < len(marks) else view.tl.t_end
    fid = flow_base
    for i, r in enumerate(rows):
        st = view.rec_stats(i)
        label = "%s %s" % (r["kn"], r["name"])
        if i in mark_t:
            a, b = mark_t[i], mark_next[i]
            ev.append({"ph": "X", "pid": pid, "tid": TID_ISSUE, "name": label, "ts": _us(a - t0, f), "dur": _dur(a - t0, b - t0, f),
                       "args": {"record": i, "node": r["node"], "kind": r["kn"], "layer": str(r["layer"]), "group": r["group"]}})
        if st["n"]:
            args = {"id": "%s:r%d" % (name_tag(pid), i), "record": i, "node": r["node"], "node_name": r["name"], "kind": r["kn"], "layer": str(r["layer"]),
                    "group": r["group"], "shape": r["shape"], "bytes_read": r["rd"], "bytes_written": r["wr"], "weight_bytes": r["w"],
                    "measured_cycles": st["cost"], "ideal_cycles": r["ideal"], "ops": st["n"]}
            if r["ideal"] and st["cost"]:
                args["efficiency"] = round(r["ideal"] / st["cost"], 4)
            ev.append({"ph": "X", "pid": pid, "tid": TID_REC + rec_lane[i] * 10 if rec_lane[i] else TID_REC, "name": label, "ts": _us(st["first"], f),
                       "dur": _dur(st["first"], st["last"], f), "args": args})
    # op slices and flows
    wcum = 0.0
    qd = {}
    for o in sorted(view.tl.ops, key=lambda o: o.pidx):
        u = P.UNITS[o.unit] if o.unit < len(P.UNITS) else "u%d" % o.unit
        tid = tids.get(id(o), TID.get(u, 9) * 100)
        r = rows[o.rec] if o.rec >= 0 else None
        opn = P.OPNAME.get(o.opcode, "op%02x" % o.opcode)
        sid = slice_id(name_tag(pid), o.rec, o.unit, o.ordinal)
        run = o.run if o.ran else 0
        args = {"id": sid, "unit": u, "op": opn, "pidx": o.pidx, "cycles": run, "queue_wait": o.qwait, "head_wait": o.hwait, "end": o.tend - t0}
        if r is not None:
            args.update({"record": o.rec, "node": r["node"], "node_name": r["name"], "kind": r["kn"], "layer": str(r["layer"]), "group": r["group"],
                         "shape": r["shape"], "record_bytes_read": r["rd"]})
            if o.ideal:
                args["ideal_cycles"] = round(o.ideal, 2)
                args["bytes_read"] = round(o.wbytes)
                if run:
                    args["efficiency"] = round(o.ideal / run, 4)
        else:
            args["record"] = "launch setup"
        ts = _us(o.t_start - t0, f)
        if run:
            ev.append({"ph": "X", "pid": pid, "tid": tid, "name": "%s %s" % (opn, r["name"] if r is not None else "setup"), "ts": ts, "dur": _dur(o.t_start - t0, o.t_start - t0 + run, f),
                       "args": args})
        else:                                        # no run time (a JOIN, an op that did not run): an instant, so it cannot overlap a slice
            ev.append({"ph": "i", "s": "t", "pid": pid, "tid": tid, "name": "%s %s" % (opn, r["name"] if r is not None else "setup"), "ts": ts, "args": args})
        if run and flows and o.rec >= 0 and o.rec in mark_t:
            fid += 1
            ev.append({"ph": "s", "pid": pid, "tid": TID_ISSUE, "id": fid, "name": "issue", "cat": "record", "ts": _us(mark_t[o.rec] - t0, f) })
            ev.append({"ph": "f", "pid": pid, "tid": tid, "id": fid, "name": "issue", "cat": "record", "bp": "e", "ts": round(ts + 0.0005, 4)})
        if o.unit == 2 and o.ideal:
            wcum += o.wbytes
            ev.append({"ph": "C", "pid": pid, "name": "weights streamed (MB, model)", "ts": _us(o.tend - t0, f), "args": {"MB": round(wcum / 1e6, 3)}})
    # queue depth per unit: ops dispatched and not completed
    for u in (0, 1, 2, 3):
        edges = {}
        for o in view.tl.ops:
            if o.unit != u:
                continue
            edges[o.t_disp] = edges.get(o.t_disp, 0) + 1
            edges[o.tend] = edges.get(o.tend, 0) - 1
        d = 0
        for t in sorted(edges):
            d += edges[t]
            ev.append({"ph": "C", "pid": pid, "name": "%s queue depth" % P.UNITS[u], "ts": _us(t - t0, f), "args": {"ops": d}})
    # PERF counters of the launch (the hardware gives them per launch): a level over the launch
    pf = view.perf
    cyc = pf.get("CYCLES", 0)
    if cyc:
        lv = {}
        for c in range(C):
            lv["DDR ch%d read (GB/s)" % c] = ("GB/s", pf.get("CH_RD_BEATS[%d]" % c, 0) * 64 / (cyc / f) / 1e9)
            lv["DDR ch%d wait (%% of cycles)" % c] = ("%", 100.0 * pf.get("CH_WAIT[%d]" % c, 0) / cyc)
        for key, nm in (("MVM_CYCLES", "MVM busy"), ("MVM_EMPTY", "MVM_EMPTY"), ("MVM_WAIT", "MVM_WAIT"), ("VU_CYCLES", "VU busy"),
                        ("VU_HAZARD", "VU_HAZARD"), ("VU_SPM_WAIT", "VU_SPM_WAIT"), ("DMA_CYCLES", "DMA busy"), ("QACT_CYCLES", "QACT busy"),
                        ("ATT_CYCLES", "ATT busy"), ("HOLD_CYCLES", "HOLD_CYCLES"), ("RV_WFI", "RV_WFI"), ("SPM_RD_WAIT", "SPM_RD_WAIT"),
                        ("SPM_WR_WAIT", "SPM_WR_WAIT")):
            if key in pf:
                lv["%s (%% of cycles)" % nm] = ("%", 100.0 * pf[key] / cyc)
        span = _us(view.span, f)
        for nm, (unit, v) in lv.items():
            ev.append({"ph": "C", "pid": pid, "name": "PERF " + nm, "ts": 0, "args": {unit: round(v, 4)}})
            ev.append({"ph": "C", "pid": pid, "name": "PERF " + nm, "ts": span, "args": {unit: 0}})
    ev.append({"ph": "i", "s": "p", "pid": pid, "tid": TID_ISSUE, "name": "LAUNCH_BEGIN", "ts": 0})
    ev.append({"ph": "i", "s": "p", "pid": pid, "tid": TID_ISSUE, "name": "LAUNCH_END", "ts": _us(view.span, f)})
    return ev, fid


def name_tag(pid):
    return {1: "card", 2: "sim"}.get(pid, "p%d" % pid)


def write_perfetto(path, views, rows, f, note=""):
    """views: [(LaunchView, pid, process name)]. Writes the JSON; returns the number of events."""
    ev = []
    fid = 0
    for v, pid, name in views:
        e, fid = view_events(v, rows, pid, name, f, True, fid)
        ev += e
    doc = {"displayTimeUnit": "ns", "otherData": {"source": "tsr prof report (docs/prof-dbg.md)", "note": note}, "traceEvents": ev}
    with open(path, "w") as fh:
        json.dump(doc, fh, separators=(",", ":"))
    return len(ev)
