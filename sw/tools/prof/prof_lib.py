"""prof_lib.py: load a profile dump (TSR_DUMP_PLAN) and join the trace to the plan (docs/prof-dbg.md "The profile report").

A dump directory holds plan_<id>.json (the ggml nodes, fusion groups and op lists of a built plan), launches.jsonl (one line per
launch with PERF deltas), trw.raw (the trace writer's 32-byte records of the whole run), meta.json, cpu_fallback.json and,
optionally, sched.txt (the scheduler's splits). The join: a launch's NODE_BEGIN markers carry the op-list index of each record
and the next push index, so the ops pushed between two markers are the ops of that record.

Standard library only.
"""
import collections
import json
import os
import re
import struct

UNITS = ["DMA", "QACT", "MVM", "VU", "ATT", "SYNC"]
OPNAME = {0x00: "NOP", 0x10: "LD", 0x11: "ST", 0x12: "DQROW", 0x14: "KVST", 0x18: "QACT", 0x20: "MVM", 0x21: "ATTQK",
          0x22: "ATTPV", 0x30: "VOP", 0x31: "RED", 0x32: "ROPE", 0x33: "TOPK", 0x34: "ASMAX", 0x35: "AMASK", 0x36: "AMERGE",
          0x37: "ANORM", 0x38: "VMASK", 0x39: "GDN", 0x40: "SOP", 0x7F: "FENCE"}
KIND_JOIN = 1

# PERF slots of EX §6 for C channels: name by index
def perf_names(num_ch=4):
    c = num_ch
    n = {0: "CYCLES", 1: "RETIRED", 2: "MVM_CYCLES", 3: "VU_CYCLES"}
    for i in range(c):
        n[5 + i] = "CH_RD_BEATS[%d]" % i
        n[5 + c + i] = "CH_WAIT[%d]" % i
    x = 5 + 2 * c
    for off, name in enumerate(["DMA_CYCLES", "QACT_CYCLES", "ATT_CYCLES", "MVM_WAIT", "MVM_EMPTY", "HOLD_CYCLES", "PUSHES",
                                "PUSH_STALL", "BRIDGE_ACC", "BRIDGE_WAIT", "REC_DROPPED", "LAUNCHES", "RV_WFI", "VU_SPM_WAIT",
                                "VU_HAZARD", "VU_LINES", "ATT_QK_CYCLES", "ATT_PV_CYCLES", "ATT_PATH_STALL", "KVST_RMW",
                                "SPM_RD_WAIT", "SPM_WR_WAIT"]):
        n[x + off] = name
    n[61], n[62], n[63] = "TRW_RECORDS", "TRW_DROPS", "DBG_STOP_CYCLES"
    return n


def perf_dict(perf, num_ch=4):
    names = perf_names(num_ch)
    return {names[i]: perf[i] for i in names if i < len(perf)}


class Op:
    """One trace-writer OP record: a unit's run of one tile op."""
    __slots__ = ("unit", "opcode", "sub", "pidx", "ticket", "tend", "run", "qwait", "hwait", "ran", "code", "rec", "ordinal",
                 "ideal", "wbytes")

    @property
    def t_start(self):
        return self.tend - self.run if self.ran else self.tend

    @property
    def t_head(self):
        return self.t_start - self.hwait if self.ran else self.tend - self.hwait

    @property
    def t_disp(self):
        return self.t_head - self.qwait


class Launch:
    """The ops and markers of one launch, in push order."""

    def __init__(self):
        self.seq = 0
        self.kernel = 0
        self.t_begin = self.t_end = 0
        self.pidx_begin = self.pidx_end = 0
        self.ops = []                  # Op, sorted by pidx
        self.nodes = []                # (op-list index, next push index, time)
        self.other_marks = []


def load_trw(path):
    """Launches of a raw trace file: [Launch]. Ops are assigned by push index between a LAUNCH_BEGIN and its LAUNCH_END."""
    data = open(path, "rb").read()
    n = len(data) // 32
    launches, begins, ops_all = [], [], []
    cur = None
    for i in range(n):
        w = struct.unpack_from("<8I", data, 32 * i)
        kind = w[0] & 15
        if kind == 15:
            continue
        if kind == 1:                                # MARK
            cls = w[2] >> 28
            t = w[3] | (w[4] << 32)
            if cls == 1:
                cur = Launch()
                cur.seq = w[2] & 0xFFFF
                cur.kernel = (w[2] >> 16) & 0xFFF
                cur.t_begin = t
                cur.pidx_begin = w[1]
                launches.append(cur)
            elif cls == 2 and cur is not None:
                cur.t_end = t
                cur.pidx_end = w[1]
                cur = None
            elif cls == 3 and cur is not None:
                cur.nodes.append((w[2] & 0xFFFFFFF, w[1], t))
            elif cur is not None:
                cur.other_marks.append((cls, w[2] & 0xFFFFFFF, t))
        elif kind == 0:                              # OP
            o = Op()
            o.unit = (w[0] >> 4) & 15
            o.code = (w[0] >> 8) & 15
            o.ran = (w[0] >> 12) & 1
            o.opcode = (w[0] >> 16) & 255
            o.sub = (w[0] >> 24) & 255
            o.pidx = w[1]
            o.ticket = w[2]
            o.tend = w[3] | (w[4] << 32)
            o.run = w[5]
            o.qwait = w[6]
            o.hwait = w[7]
            o.rec = -1
            o.ordinal = 0
            o.ideal = 0.0
            o.wbytes = 0.0
            ops_all.append(o)
    ops_all.sort(key=lambda o: o.pidx)
    done = [l for l in launches if l.t_end]          # a launch the ring cut off at either end is dropped
    j = 0
    prev_end = 0
    for l in done:
        # the firmware pushes one control entry before it writes LAUNCH_BEGIN (its push index is begin - 1): it belongs here
        lo = max(prev_end, l.pidx_begin - 1)
        while j < len(ops_all) and ops_all[j].pidx < lo:
            j += 1
        k = j
        while k < len(ops_all) and ops_all[k].pidx < l.pidx_end:
            k += 1
        l.ops = ops_all[j:k]
        j = k
        prev_end = l.pidx_end
    return done


def parse_llama_log(path):
    """llama.cpp's own speed line and the backend's PERF total out of a run.log: {toks, ms, perf_cycles, launches} (missing: None)."""
    out = {"toks": None, "ms": None, "perf_cycles": None, "launches": None}
    if not os.path.exists(path):
        return out
    for line in open(path, errors="replace"):
        m = re.search(r"eval time =\s+([\d.]+) ms /\s+(\d+) runs.*?([\d.]+) ms per token,\s+([\d.]+) tokens per second", line)
        if m and "prompt eval" not in line:
            out["ms"], out["toks"] = float(m.group(3)), float(m.group(4))
        m = re.search(r"stats: PERF_CYCLES (\d+)", line)
        if m:
            out["perf_cycles"] = int(m.group(1))
        m = re.search(r"launches (\d+) \(", line)
        if m:
            out["launches"] = int(m.group(1))
    return out


class Plan:
    def __init__(self, d):
        self.d = d
        self.id = d["plan"]
        self.nodes = d["nodes"]
        self.lists = d["lists"]
        self.groups = d["groups"]
        self.inputs = d["inputs"]


class Run:
    """A dump directory."""

    def __init__(self, path):
        self.path = path
        self.meta = json.load(open(os.path.join(path, "meta.json")))
        self.launch_log = []
        with open(os.path.join(path, "launches.jsonl")) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.launch_log.append(json.loads(line))
        self.plans = {}
        for e in sorted(os.listdir(path)):
            m = re.match(r"plan_(\d+)\.json$", e)
            if m and int(m.group(1)) in {l["plan"] for l in self.launch_log}:
                self.plans[int(m.group(1))] = Plan(json.load(open(os.path.join(path, e))))
        p = os.path.join(path, "cpu_fallback.json")
        self.cpu = json.load(open(p))["nodes"] if os.path.exists(p) else []
        p = os.path.join(path, "capture.json")
        self.capture = json.load(open(p)) if os.path.exists(p) else {}
        p = os.path.join(path, "sched.json")
        self.sched = json.load(open(p)) if os.path.exists(p) else None
        self.trw = []
        p = os.path.join(path, "trw.raw")
        if os.path.exists(p) and os.path.getsize(p):
            self.trw = load_trw(p)
        # the ring keeps the newest records: its launches are the last ones of the log
        n = len(self.trw)
        self.trw_offset = len(self.launch_log) - n
        self.core_hz = None
        hz = sorted(l["core_hz"] for l in self.launch_log if l.get("core_hz"))
        if hz:
            self.core_hz = hz[len(hz) // 2]

    def launch(self, k):
        """(log entry, trace Launch or None) of the k-th launch of the run."""
        e = self.launch_log[k]
        i = k - self.trw_offset
        return e, (self.trw[i] if 0 <= i < len(self.trw) else None)

    def decode_launches(self):
        """Launch numbers whose list is the most common plan with a cache hit or the same plan as a later launch: decode tokens."""
        by_plan = collections.defaultdict(list)
        for k, e in enumerate(self.launch_log):
            by_plan[(e["plan"], e["list"])].append(k)
        best = max(by_plan.values(), key=len) if by_plan else []
        return best


def join_launch(plan, lst, tl):
    """Attach each op of the trace launch to its record. Returns per-record dicts (index = op-list index) and the unassigned
    ops. The record cycles are the increase of the completion frontier over the record's ops, so they add to the launch."""
    recs = plan.lists[lst]["recs"]
    nodes = sorted(tl.nodes, key=lambda x: x[1])      # by next push index = issue order
    bounds = [n[1] for n in nodes] + [tl.pidx_end]
    by_idx = {}
    ops = tl.ops
    j = 0
    for k, (rec_i, np_, t_mark) in enumerate(nodes):
        lo, hi = np_, bounds[k + 1]
        while j < len(ops) and ops[j].pidx < lo:
            j += 1
        first = j
        while j < len(ops) and ops[j].pidx < hi:
            j += 1
        by_idx[rec_i] = (t_mark, ops[first:j])
    pre = []
    if nodes:
        for o in ops:
            if o.pidx < nodes[0][1]:
                pre.append(o)
    else:
        pre = list(ops)
    return by_idx, pre


def frontier_costs(order, by_idx, pre, t_begin, t_end):
    """cost per record index and the setup / tail cycles: the increase of the completion frontier (the latest end of any op
    so far), starting at LAUNCH_BEGIN; the launch's span = setup + sum of record costs + tail."""
    frontier = t_begin
    cost = {}
    setup = 0
    if pre:
        mx = max(o.tend for o in pre)
        setup = max(0, mx - frontier)
        frontier = max(frontier, mx)
    for i in order:
        t_mark, ops = by_idx.get(i, (None, []))
        if not ops:
            cost[i] = 0
            continue
        mx = max(o.tend for o in ops)
        cost[i] = max(0, mx - frontier)
        frontier = max(frontier, mx)
    tail = max(0, t_end - frontier)
    return cost, setup, tail
