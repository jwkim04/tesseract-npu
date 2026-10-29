#!/usr/bin/env python3
"""trace_stats.py — MVM shape coverage and weight traffic of v1 trace directories (the cmd.bin
format of tag v1-m4's sw/sim/include/tsr_trace.h).

    python3 traces/m3/trace_stats.py [--per-trace] [--classes] [--json FILE] [--big BEATS] PATH...

PATH is a trace directory or a directory searched recursively for trace directories (they hold
meta.json). Traces are grouped into sets by their parent directory. Only executed instructions
count: the prefix before ERR_PC when the trace ends in an error, else everything before the END
that stopped it (v1 programs have no jumps). Per set it prints the number of traces, size on disk,
error endings, the executed MVMs by format, K, N, T, ACC and dst_stride, and the MVM weight
traffic: channel c reads rows_c x ROW_BEATS beats per MVM (spec 구현 명확화, PERF), so channel 0
(R = ceil(N/C) rows) is the busiest; "weight beats" below are channel-0 beats summed over a
trace's MVMs, i.e. the cycles a cluster needs at one weight beat per cycle.

Self-check: for every channel c >= 1 the simulator's recorded perf ch_rd_beats[c] must equal the
MVM weight beats plus the LD beats decoded here (channel 0 also counts instruction fetch, so only
>= is checked there). A mismatch is reported and makes the exit code 1.

The decoder is independent of libtsr_ref (ROW_BEATS from the spec formula), standard library only.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import statistics
import struct
import sys

OP_END, OP_LD, OP_MVM = 0x01, 0x10, 0x20
STATUS_ERR = 4


def ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def row_beats(fmt: int, k: int) -> int:
    """ROW_BEATS(fmt, K) (spec 디바이스 가중치 레이아웃, 행 크기)."""
    bs, s, qb = (32, 32, 32) if fmt == 0 else (256, 4, 128)
    nb = k // bs
    full, rem = divmod(nb, s)
    return full * (1 + ceil_div(s * qb, 64)) + ((1 + ceil_div(rem * qb, 64)) if rem else 0)


def find_traces(path: str) -> list[str]:
    out = []
    if os.path.isfile(os.path.join(path, "meta.json")):
        return [path]
    for root, dirs, files in os.walk(path):
        dirs.sort()
        if "meta.json" in files:
            out.append(root)
            dirs[:] = []
    return sorted(out)


def dir_bytes(path: str) -> int:
    n = 0
    for root, _, files in os.walk(path):
        for f in files:
            n += os.path.getsize(os.path.join(root, f))
    return n


class Mvm:
    __slots__ = ("fmt", "acc", "n", "k", "t", "w6", "beats")

    def __init__(self, fmt, acc, n, k, t, w6, beats):
        self.fmt, self.acc, self.n, self.k, self.t, self.w6, self.beats = fmt, acc, n, k, t, w6, beats


def analyse(tdir: str) -> dict:
    meta = json.load(open(os.path.join(tdir, "meta.json")))
    cfg = meta["config"]
    C = cfg["num_ch"]
    cmd = open(os.path.join(tdir, "cmd.bin"), "rb").read()
    count = meta["cmd_count"]
    err = bool(meta["status"] & STATUS_ERR)
    stop = meta["err_pc"] if err else count
    mvms, ld = [], [0] * C
    for pc in range(min(stop, count)):
        w = struct.unpack_from("<8I", cmd, 32 * pc)
        op = w[0] & 0xFF
        if not err and op == OP_END:
            break
        if op == OP_LD:
            ld[(w[0] >> 8) & 0xF] += w[3]
        elif op == OP_MVM:
            fmt, n, k, t = (w[0] >> 8) & 0xF, w[4], w[5] & 0xFFFF, (w[5] >> 16) & 0xF
            rb, r = row_beats(fmt, k), ceil_div(n, C)
            beats = [max(0, min((c + 1) * r, n) - c * r) * rb for c in range(C)]
            mvms.append(Mvm(fmt, (w[0] >> 12) & 1, n, k, t, w[6], beats))
    issues = []
    perf = meta.get("perf")
    if perf and "ch_rd_beats" in perf:
        for c in range(C):
            want = sum(m.beats[c] for m in mvms) + ld[c]
            got = perf["ch_rd_beats"][c]
            if (c > 0 and got != want) or (c == 0 and got < want):
                issues.append(f"ch{c}: perf ch_rd_beats {got}, decoded MVM + LD beats {want}")
    return {
        "dir": tdir, "cfg": cfg, "err": meta["err_code"] if err else 0, "mvms": mvms,
        "wbeats": sum(m.beats[0] for m in mvms), "wall": sum(sum(m.beats) for m in mvms),
        "bytes": dir_bytes(tdir), "issues": issues,
    }


def n_bucket(n: int, c: int) -> str:
    if n == 0:
        return "0"
    if n == 1:
        return "1"
    if n < c:
        return "2..C-1"
    for hi, name in ((256, "C..255"), (1024, "256..1023"), (4096, "1024..4095"), (32768, "4096..32767")):
        if n < hi:
            return name
    return ">=32768"


def fmt_hist(h: collections.Counter, order=None) -> str:
    keys = order if order is not None else sorted(h)
    return ", ".join(f"{k}: {h[k]}" for k in keys if h[k])


def set_report(name: str, rows: list[dict], big: int) -> dict:
    cfg = rows[0]["cfg"]
    C, kmax, tmax = cfg["num_ch"], cfg["k_max"], cfg["t_max"]
    mv = [m for r in rows for m in r["mvms"]]
    fmt = collections.Counter("Q8_0" if m.fmt == 0 else "Q4_K" for m in mv)
    th = collections.Counter(m.t for m in mv)
    nh = collections.Counter(n_bucket(m.n, C) for m in mv)
    kh = collections.Counter(m.k for m in mv)
    partial = sum(1 for m in mv if (m.k // (32 if m.fmt == 0 else 256)) % (32 if m.fmt == 0 else 4))
    yl = lambda m: ceil_div(4 * m.n, 64)
    wb = [r["wbeats"] for r in rows]
    errs = collections.Counter(r["err"] for r in rows if r["err"])
    rep = {
        "set": name, "traces": len(rows), "bytes": sum(r["bytes"] for r in rows),
        "config": {k: cfg[k] for k in ("num_ch", "t_max", "k_max", "spm_bytes", "ch_bytes", "addr_beat_w")},
        "error_endings": dict(sorted(errs.items())), "mvms": len(mv), "fmt": dict(fmt),
        "acc": sum(m.acc for m in mv), "T": dict(sorted(th.items())), "N": dict(nh),
        "K_distinct": len(kh), "K_top": kh.most_common(12), "K_partial_chunk": partial,
        "K_eq_K_MAX": kh[kmax], "N_lt_C": sum(1 for m in mv if 0 < m.n < C),
        "N_gt_1000": sum(1 for m in mv if m.n > 1000),
        "shared_y_line": sum(1 for m in mv if C > 1 and m.n > ceil_div(m.n, C) and m.n % 16 and ceil_div(m.n, C) % 16),
        "dst_stride_explicit": sum(1 for m in mv if m.w6), "dst_stride_gap": sum(1 for m in mv if m.t > 1 and m.w6 > yl(m)),
        "T_eq_T_MAX": th[tmax],
        "wbeats_sum": sum(wb), "wbeats_all_ch_sum": sum(r["wall"] for r in rows),
        "wbeats_max": max(wb) if wb else 0, "wbeats_median": int(statistics.median(wb)) if wb else 0,
        "traces_above_big": sum(1 for b in wb if b > big),
        "issues": [f"{r['dir']}: {i}" for r in rows for i in r["issues"]],
    }
    return rep


def print_set(rep: dict, big: int) -> None:
    c = rep["config"]
    print(f"== {rep['set']}: {rep['traces']} traces, {rep['bytes'] / 2**20:.1f} MiB  "
          f"(C {c['num_ch']}, T_MAX {c['t_max']}, K_MAX {c['k_max']}, SPM {c['spm_bytes'] // 1024} KiB)")
    if rep["error_endings"]:
        print(f"   error endings (ERR_CODE: traces): {rep['error_endings']}")
    print(f"   MVMs executed {rep['mvms']}: {fmt_hist(collections.Counter(rep['fmt']))}; ACC {rep['acc']}; "
          f"dst_stride explicit {rep['dst_stride_explicit']} (with gaps at T>1: {rep['dst_stride_gap']})")
    print(f"   T: {fmt_hist(collections.Counter(rep['T']))}  (T = T_MAX: {rep['T_eq_T_MAX']})")
    order = ["0", "1", "2..C-1", "C..255", "256..1023", "1024..4095", "4096..32767", ">=32768"]
    print(f"   N: {fmt_hist(collections.Counter(rep['N']), order)}  (N < C: {rep['N_lt_C']}, N > 1000: "
          f"{rep['N_gt_1000']}, y line shared by 2 channels: {rep['shared_y_line']})")
    print(f"   K: {rep['K_distinct']} distinct, partial last chunk {rep['K_partial_chunk']}, K = K_MAX "
          f"{rep['K_eq_K_MAX']}; most common: {', '.join(f'{k}x{n}' for k, n in rep['K_top'])}")
    print(f"   weight beats (channel 0, per trace): sum {rep['wbeats_sum']}, median {rep['wbeats_median']}, "
          f"max {rep['wbeats_max']}, traces > {big}: {rep['traces_above_big']}; all channels sum "
          f"{rep['wbeats_all_ch_sum']}")
    for i in rep["issues"]:
        print(f"   SELF-CHECK MISMATCH {i}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--per-trace", action="store_true", help="also list every trace's MVM shapes and weight beats")
    ap.add_argument("--json", help="write the per-set reports (and per-trace rows with --per-trace) as JSON")
    ap.add_argument("--classes", action="store_true",
                    help="group each set's traces by their executed MVM shapes (count, beats and size per trace)")
    ap.add_argument("--big", type=int, default=20000, help="report traces above this many channel-0 weight beats")
    a = ap.parse_args()
    sets: dict[str, list[dict]] = collections.OrderedDict()
    for p in a.paths:
        tr = find_traces(p)
        if not tr:
            print(f"trace_stats: no trace under {p}", file=sys.stderr)
            return 1
        for t in tr:
            sets.setdefault(os.path.dirname(os.path.normpath(t)), []).append(analyse(t))
    out, bad = [], 0
    for name, rows in sets.items():
        rep = set_report(name, rows, a.big)
        print_set(rep, a.big)
        bad += len(rep["issues"])
        if a.per_trace:
            rep["per_trace"] = []
            for r in rows:
                shapes = [f"{'Q8_0' if m.fmt == 0 else 'Q4_K'} K{m.k} N{m.n} T{m.t}{' ACC' if m.acc else ''}"
                          for m in r["mvms"]]
                print(f"   {os.path.basename(r['dir'])}: {r['wbeats']} beats, {r['bytes'] / 2**20:.2f} MiB"
                      f"{', ERR ' + str(r['err']) if r['err'] else ''}: {'; '.join(shapes)}")
                rep["per_trace"].append({"trace": os.path.basename(r["dir"]), "wbeats": r["wbeats"],
                                         "bytes": r["bytes"], "err": r["err"], "mvms": shapes})
        if a.classes:
            cls: dict[str, list[dict]] = collections.OrderedDict()
            for r in rows:
                key = "; ".join(f"{'Q8_0' if m.fmt == 0 else 'Q4_K'} K{m.k} N{m.n} T{m.t}{' ACC' if m.acc else ''}"
                                for m in r["mvms"]) or "(no MVM)"
                cls.setdefault(key, []).append(r)
            rep["classes"] = []
            print(f"   {'traces':>6} {'beats/trace':>11} {'MiB/trace':>9}  MVMs (first trace)")
            for key, rs in sorted(cls.items(), key=lambda kv: -kv[1][0]["wbeats"]):
                b = sorted({x["wbeats"] for x in rs})
                print(f"   {len(rs):6d} {'/'.join(map(str, b)):>11} {rs[0]['bytes'] / 2**20 / 1:9.2f}  {key} "
                      f"({os.path.basename(rs[0]['dir'])})")
                rep["classes"].append({"mvms": key, "traces": [os.path.basename(x["dir"]) for x in rs],
                                       "wbeats": b, "bytes": rs[0]["bytes"]})
        out.append(rep)
    tot_t = sum(r["traces"] for r in out)
    print(f"== total: {tot_t} traces, {sum(r['bytes'] for r in out) / 2**20:.1f} MiB, {sum(r['mvms'] for r in out)} MVMs, "
          f"{sum(r['wbeats_sum'] for r in out)} channel-0 weight beats")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)
    if bad:
        print(f"trace_stats: {bad} self-check mismatches", file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
