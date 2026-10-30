#!/usr/bin/env python3
# fw_prof_lines.py — per-source-line breakdown of a firmware profile (sw/ctrl/tests/fw_prof.h).
#
#   TSR_FW_PROF_DUMP=/tmp/p test_fw_m6_e2e IMAGE SCRATCH --perf
#       (writes /tmp/p.mulmat, .mulmat_deep, .mulmat_prod: the 0.5B decode lists with U200 / deep
#       queues / deep queues without MARK; /tmp/p.join: the JOIN list)
#   TSR_FW_PROF=1 TSR_FW_PROF_DUMP=/tmp/q test_fw_exec_rv IMAGE --perf   (writes /tmp/q.case0 ...)
#   fw_prof_lines.py IMAGE /tmp/p.mulmat [--top 40] [--addr2line PATH]
#
# Reads the per-PC histogram ("pc cycles count", hex pc; first line "# title; per = n"), maps every
# PC to file:line (with inlining: the innermost source line) through riscv-none-elf-addr2line, and
# prints the lines with the most cycles, divided by `per` (the harness's unit, e.g. op records).
import os
import subprocess
import sys


def main():
    args = sys.argv[1:]
    if len(args) < 2:
        print(__doc__ if __doc__ else "usage: fw_prof_lines.py IMAGE HIST [--top N] [--addr2line PATH]")
        return 2
    image, hist = args[0], args[1]
    top = 40
    a2l = None
    i = 2
    while i < len(args):
        if args[i] == "--top":
            top = int(args[i + 1]); i += 2
        elif args[i] == "--addr2line":
            a2l = args[i + 1]; i += 2
        else:
            print("unknown option", args[i]); return 2
    if a2l is None:
        here = os.path.dirname(os.path.abspath(__file__))
        root = os.path.abspath(os.path.join(here, "..", "..", ".."))
        cand = os.path.join(root, "build", "tools", "riscv", "xpack-riscv-none-elf-gcc-14.2.0-3", "bin",
                            "riscv-none-elf-addr2line")
        a2l = cand if os.path.exists(cand) else "riscv-none-elf-addr2line"
    per = 1.0
    rows = []
    with open(hist) as f:
        for ln in f:
            if ln.startswith("#"):
                if "per =" in ln:
                    per = float(ln.rsplit("per =", 1)[1].strip()) or 1.0
                continue
            p = ln.split()
            if len(p) == 3:
                rows.append((int(p[0], 16), float(p[1]), float(p[2])))
    pcs = [r[0] for r in rows]
    out = subprocess.run([a2l, "-e", image, "-f", "-C"] + ["%x" % pc for pc in pcs], capture_output=True, text=True,
                         check=True).stdout.splitlines()
    by_line = {}
    total = 0.0
    for k, (pc, cyc, n) in enumerate(rows):
        fn = out[2 * k] if 2 * k < len(out) else "?"
        loc = out[2 * k + 1] if 2 * k + 1 < len(out) else "?"
        loc = loc.split(" (discriminator")[0]
        if "/sw/" in loc:
            loc = "sw/" + loc.split("/sw/", 1)[1]           # repository-relative
        key = (loc, fn)
        c, i_ = by_line.get(key, (0.0, 0.0))
        by_line[key] = (c + cyc, i_ + n)
        total += cyc
    items = sorted(by_line.items(), key=lambda kv: -kv[1][0])
    print("%-56s %-40s %8s %7s %6s" % ("source line", "function", "cycles", "insns", "share"))
    for (loc, fn), (c, n) in items[:top]:
        print("%-56s %-40s %8.1f %7.1f %5.1f%%" % (loc[-56:], fn[:40], c / per, n / per, 100.0 * c / total if total else 0))
    print("total %.1f cycles per unit (%d lines)" % (total / per, len(items)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
