#!/usr/bin/env python3
"""trw_vs_sim.py: where does a device run spend more cycles than the simulator's cycle model?

Compares the per-op times of one launch on the card (the trace writer's records, DB §8.3) with the same launch on the
simulator (the event log), op by op, and adds them up by record kind or by op class. Standard library only.

Inputs (docs/prof-dbg.md "Comparing a card run with the simulator"):
  --trw FILE    raw trace records: `tsr-dbg --device xrt --attach -ex "trw dump out.json out.raw"` after a run that had
                the trace writer on (`trw start 16` before it; DBG_CFG survives the process)
  --sim FILE    the simulator's log of the same run: TSR_SIM_LOG=w,FILE (push, start and done lines)
  --kinds FILE  the record kinds of each launch, from TSR_OPL_KINDS=FILE (the backend writes one line per launch)
Run both with TSR_OPL_PROF=1 (the descriptor's PROF bit: a NODE_BEGIN marker before every record), and the simulator with
the release image's configuration (TSR_SIM_CARD=k4dual TSR_SIM_GDN=1), so that the two op streams are the same.

  trw_vs_sim.py kinds --trw T --sim S --kinds K --launch N       per record kind: cycles the record adds to the launch
  trw_vs_sim.py units --trw T --sim S --launch N                 per unit and op class: run time of each op, sim and card
  trw_vs_sim.py gaps  --trw T --sim S --launch N [--unit 3]      per op class: the time from the done of the op before
  trw_vs_sim.py record --trw T --sim S --kinds K --launch N --index I     the ops of one record, side by side

A launch is the stretch between a LAUNCH_BEGIN and its LAUNCH_END marker; N counts from 0. The cycles of a record are the
increase of the launch's completion frontier (the latest end of any op so far) over its ops, so they add up to the launch's
time and charge a record only for the time it adds. The ops of a unit are paired in push order.
"""
import argparse
import collections
import re
import struct
import sys

KIND = {0x01: "JOIN", 0x10: "MUL_MAT", 0x11: "ADD", 0x12: "MUL", 0x13: "SCALE", 0x14: "RMS_NORM", 0x15: "GLU", 0x16: "ROPE",
        0x17: "GET_ROWS", 0x18: "SET_ROWS_KV", 0x19: "FLASH_ATTN", 0x1A: "CPY", 0x1B: "UNARY", 0x1E: "MOE_ROUTE",
        0x1F: "MUL_MAT_ID", 0x2D: "SOFTPLUS", 0x2E: "SSM_CONV", 0x2F: "GATED_DELTA_NET", 0x00: "NOP"}
UNIT = {0: "DMA", 1: "QACT", 2: "MVM", 3: "VU", 4: "ATT", 5: "SYNC"}
VOP = {0x00: "ADD", 0x01: "SUB", 0x02: "MUL", 0x03: "MAX", 0x04: "MULC", 0x05: "ADDC", 0x06: "SILU", 0x07: "SWIGLU",
       0x08: "EXP", 0x09: "COPY", 0x0A: "MIN", 0x0B: "SUBC", 0x0C: "MULCB", 0x0D: "MULCADD", 0x0E: "ADDMULC", 0x0F: "EXPSUB",
       0x10: "EXPSUBC", 0x11: "SIGMOID", 0x12: "GELU", 0x13: "GEGLU", 0x14: "TANH", 0x15: "RECIP", 0x16: "SETC",
       0x17: "GELU16", 0x18: "GEGLU16"}
SOP = {0: "MOVI", 1: "ADD", 2: "MUL", 3: "ADDI", 4: "MULI", 5: "SQRT", 6: "RECIP", 7: "LDS", 8: "STS", 9: "SUB", 10: "DIV",
       11: "RSQRT", 12: "MAX", 13: "MIN", 14: "EXP"}


def op_class(w0):
    op = w0 & 0xFF
    if op == 0x30:
        return "VOP." + VOP.get((w0 >> 16) & 0xFF, "?")
    if op == 0x31:
        return "RED.%d%s%s" % ((w0 >> 8) & 15, ".S" if (w0 >> 12) & 1 else "", ".rms" if ((w0 >> 20) & 3) == 2 else "")
    if op == 0x40:
        return "SOP." + SOP.get((w0 >> 8) & 15, "?")
    names = {0x10: "LD", 0x11: "ST", 0x14: "KVST", 0x18: "QACT", 0x20: "MVM", 0x21: "ATTQK", 0x22: "ATTPV", 0x32: "ROPE",
             0x34: "ASMAX", 0x35: "AMASK", 0x36: "AMERGE", 0x37: "ANORM"}
    return names.get(op, "op%02x" % op)


def load_trw(path):
    """Launches of the trace records: [(ops, marks)], ops = dicts (unit, opcode, pidx, tend, run), marks = [(op index, pidx)]."""
    data = open(path, "rb").read()
    out, cur = [], None
    for i in range(0, len(data) // 32 * 32, 32):
        w = struct.unpack("<8I", data[i:i + 32])
        kind = w[0] & 15
        if kind == 15:
            continue
        if kind == 1:
            cls = w[2] >> 28
            if cls == 1:
                cur = ([], [])
                out.append(cur)
            elif cls == 2:
                cur = None
            elif cls == 3 and cur is not None:
                cur[1].append((w[2] & 0xFFFFFFF, w[1]))
        elif kind == 0 and cur is not None:
            cur[0].append(dict(unit=(w[0] >> 4) & 15, opcode=(w[0] >> 16) & 255, pidx=w[1], tend=w[3] | (w[4] << 32), run=w[5]))
    return out


def load_sim(path):
    """Launches of the simulator log: [(ops by pidx, marks)]; an op: unit, w0, start, done, ran."""
    rx = re.compile(r"^\s*(\d+) (push|start|done|mark)\s+(.*)$")
    out = []
    with open(path) as f:
        for line in f:
            m = rx.match(line)
            if not m:
                continue
            t, kind, rest = int(m.group(1)), m.group(2), m.group(3)
            if kind == "mark":
                v = int(re.search(r"value=([0-9a-f]+)", rest).group(1), 16)
                if v >> 28 == 1:
                    out.append(({}, []))
                elif v >> 28 == 3 and out:
                    out[-1][1].append((v & 0xFFFFFFF, int(re.search(r"next_pidx=(\d+)", rest).group(1))))
                continue
            if not out:
                continue
            pidx = int(re.search(r"pidx=(\d+)", rest).group(1))
            ops = out[-1][0]
            if kind == "push":
                ops[pidx] = dict(unit=int(re.search(r"u(\d+)", rest).group(1)), w0=int(re.search(r"w0=([0-9a-f]+)", rest).group(1), 16),
                                 start=None, done=None, ran=False)
            elif pidx in ops:
                if kind == "start":
                    ops[pidx]["start"] = t
                else:
                    ops[pidx]["done"] = t
                    ops[pidx]["ran"] = rest.rstrip().endswith("ran")
    return out


def load_kinds(path):
    out = []
    for line in open(path):
        m = re.match(r"launch (\d+):(.*)", line)
        if m:
            out.append([int(x, 16) for x in m.group(2).split()])
    return out


def frontier_costs(ends, marks, nkinds):
    """Per record: the increase of the completion frontier. ends: pidx -> end time."""
    pidxs = sorted(ends)
    bounds = [m[1] for m in marks] + [1 << 62]
    frontier, j, costs = None, 0, []
    for r in range(len(marks)):
        mx = None
        while j < len(pidxs) and pidxs[j] < bounds[r + 1]:
            if pidxs[j] >= bounds[r]:
                mx = ends[pidxs[j]] if mx is None else max(mx, ends[pidxs[j]])
            j += 1
        if mx is None:
            costs.append(0)
            continue
        costs.append(0 if frontier is None else max(0, mx - frontier))
        frontier = mx if frontier is None else max(frontier, mx)
    return costs


def pair_units(sim_ops, card_ops, units=None):
    """Pairs of (sim op, card op) of each unit in push order; units whose op counts differ are skipped (with a note)."""
    res = {}
    for u in sorted({o["unit"] for o in sim_ops.values()}):
        if units is not None and u not in units:
            continue
        s = [sim_ops[p] for p in sorted(sim_ops) if sim_ops[p]["unit"] == u]
        c = sorted([o for o in card_ops if o["unit"] == u], key=lambda o: o["pidx"])
        if len(s) != len(c):
            print("note: unit %s has %d ops in the simulator and %d on the card (not compared)" % (UNIT.get(u, u), len(s), len(c)),
                  file=sys.stderr)
            continue
        res[u] = list(zip(s, c))
    return res


def cmd_kinds(a):
    sim, card, kinds = load_sim(a.sim)[a.launch], load_trw(a.trw)[a.launch], load_kinds(a.kinds)[a.launch]
    sends = {p: o["done"] for p, o in sim[0].items() if o["done"] is not None}
    cends = {o["pidx"]: o["tend"] for o in card[0]}
    sc = frontier_costs(sends, sim[1], len(kinds))
    cc = frontier_costs(cends, card[1], len(kinds))
    agg = collections.defaultdict(lambda: [0, 0, 0])
    for i, k in enumerate(kinds[:len(cc)]):
        agg[k][0] += 1
        agg[k][1] += sc[i] if i < len(sc) else 0
        agg[k][2] += cc[i]
    print("%-16s %6s %12s %12s %7s %10s %10s" % ("kind", "n", "simulator", "card", "ratio", "sim / rec", "card / rec"))
    ts = tc = 0
    for k, (n, s, c) in sorted(agg.items(), key=lambda kv: -kv[1][2]):
        print("%-16s %6d %12d %12d %7.2f %10.0f %10.0f" % (KIND.get(k, hex(k)), n, s, c, c / max(s, 1), s / n, c / n))
        ts, tc = ts + s, tc + c
    print("%-16s %6s %12d %12d %7.3f" % ("total", "", ts, tc, tc / max(ts, 1)))


def cmd_units(a):
    sim, card = load_sim(a.sim)[a.launch], load_trw(a.trw)[a.launch]
    print("%-5s %-12s %6s %12s %12s %7s" % ("unit", "op class", "n", "sim run", "card run", "ratio"))
    for u, pairs in pair_units(sim[0], card[0]).items():
        agg = collections.defaultdict(lambda: [0, 0, 0])
        for s, c in pairs:
            g = agg[op_class(s["w0"])]
            g[0] += 1
            g[1] += (s["done"] - s["start"]) if s["ran"] else 0
            g[2] += c["run"]
        for k, (n, s, c) in sorted(agg.items(), key=lambda kv: -kv[1][2]):
            print("%-5s %-12s %6d %12d %12d %7.2f" % (UNIT[u], k, n, s, c, c / max(s, 1)))
    print("(a VU op's run time on the card includes its wait for the op before it; use `gaps` for the VU)")


def cmd_gaps(a):
    sim, card = load_sim(a.sim)[a.launch], load_trw(a.trw)[a.launch]
    pairs = pair_units(sim[0], card[0], {a.unit}).get(a.unit, [])
    agg = collections.defaultdict(lambda: [0, 0, 0])
    ps, pc = pairs[0][0]["done"], pairs[0][1]["tend"]
    for s, c in pairs:
        g = agg[op_class(s["w0"])]
        g[0] += 1
        g[1] += s["done"] - ps
        g[2] += c["tend"] - pc
        ps, pc = s["done"], c["tend"]
    print("%-12s %6s %12s %12s %7s %9s %9s" % ("op class", "n", "sim gap", "card gap", "ratio", "sim / op", "card / op"))
    for k, (n, s, c) in sorted(agg.items(), key=lambda kv: -kv[1][2]):
        print("%-12s %6d %12d %12d %7.2f %9.1f %9.1f" % (k, n, s, c, c / max(s, 1), s / n, c / n))


def cmd_record(a):
    sim, card, kinds = load_sim(a.sim)[a.launch], load_trw(a.trw)[a.launch], load_kinds(a.kinds)[a.launch]
    i = a.index
    f, l = sim[1][i][1], sim[1][i + 1][1] if i + 1 < len(sim[1]) else 1 << 62
    cf, cl = card[1][i][1], card[1][i + 1][1] if i + 1 < len(card[1]) else 1 << 62
    s = [sim[0][p] for p in sorted(sim[0]) if f <= p < l]
    c = sorted([o for o in card[0] if cf <= o["pidx"] < cl], key=lambda o: o["pidx"])
    print("record %d: %s, %d ops in the simulator, %d on the card (times from the first op of the record)" %
          (i, KIND.get(kinds[i], hex(kinds[i])), len(s), len(c)))
    t0s = min((o["start"] if o["start"] is not None else o["done"]) for o in s)
    t0c = min(o["tend"] - o["run"] for o in c)
    for x, y in zip(s, c):
        print("%-5s %-12s sim %7d .. %7d   card %7d .. %7d" % (UNIT[x["unit"]], op_class(x["w0"]), (x["start"] or x["done"]) - t0s,
                                                              x["done"] - t0s, y["tend"] - y["run"] - t0c, y["tend"] - t0c))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["kinds", "units", "gaps", "record"])
    ap.add_argument("--trw", required=True)
    ap.add_argument("--sim", required=True)
    ap.add_argument("--kinds")
    ap.add_argument("--launch", type=int, required=True)
    ap.add_argument("--unit", type=int, default=3)
    ap.add_argument("--index", type=int, default=0)
    a = ap.parse_args()
    if a.cmd in ("kinds", "record") and not a.kinds:
        ap.error("%s needs --kinds" % a.cmd)
    {"kinds": cmd_kinds, "units": cmd_units, "gaps": cmd_gaps, "record": cmd_record}[a.cmd](a)


if __name__ == "__main__":
    main()
