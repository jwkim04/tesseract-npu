#!/usr/bin/env python3
"""Per-instance table of the fp_area.sh runs (docs/design/02-vector-unit.md §7.10).

usage: fp_area_table.py <rundir> [name ...]
  <rundir> holds one directory per run (<rundir>/<name>/vivado.log with FPAREA lines). Prints a Markdown table:
  per instance CLB tiles, LUT, FF, CARRY8, DSP, WNS (after route), and the same CLB tiles and LUTs after place.
  Without names, every run found is listed in name order.
"""
import sys
from pathlib import Path


def parse(log: Path):
    """{'place' | 'route': per-instance values (CLB, LUT, FF, CARRY8, DSP) and WNS}."""
    d = {}
    for line in log.read_text(errors="replace").splitlines():
        f = line.split()
        if len(f) < 3 or f[1] not in ("place", "route"):
            continue
        if f[0] == "FPAREA" and "WNS" in f:
            d.setdefault(f[1], {})["WNS"] = float(f[f.index("WNS") + 1])
        elif f[0] == "FPAREA_PER":
            kv = dict(zip(f[4::2], f[5::2]))          # after "FPAREA_PER <stage> n <n>"
            d.setdefault(f[1], {}).update({k: float(v) for k, v in kv.items()})
    return d


def main():
    root = Path(sys.argv[1])
    names = sys.argv[2:] or sorted(p.name for p in root.iterdir() if (p / "vivado.log").exists())
    print("| run | CLB | LUT | FF | CARRY8 | DSP | WNS ns | CLB (place) | LUT/8 | FF/16 |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for nm in names:
        d = parse(root / nm / "vivado.log")
        st = d.get("route") or d.get("place")
        if not st:
            print("| %s | no result |" % nm)
            continue
        pl = d.get("place", st)
        print("| %s | %.1f | %.0f | %.0f | %.1f | %.1f | %+.2f | %.1f | %.1f | %.1f |" % (
            nm, st["CLB"], st["LUT"], st["FF"], st["CARRY8"], st["DSP"], st["WNS"],
            pl["CLB"], st["LUT"] / 8, st["FF"] / 16))


if __name__ == "__main__":
    main()
