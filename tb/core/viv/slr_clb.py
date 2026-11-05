#!/usr/bin/env python3
"""tb/core/viv/slr_clb.py — placed CLB sites and LUT cells per SLR, split by hierarchy.

    slr_clb.py CELLS.TSV [label=regex ...]

CELLS.TSV is the file written by slr_clb.tcl (name, LOC, REF_NAME). Every primitive on a SLICE site belongs to the
first label whose regular expression matches its hierarchical name; the rest is "other". For each label and SLR
(the part's SLICE rows Y 0..299, 300..599, 600..899 are SLR0, SLR1, SLR2; 49,260 CLBs each) it prints:

    clb  distinct SLICE sites holding at least one primitive of the label (a SLICE is one CLB)
    lut  LUT1..LUT6 cells (a LUT5 + LUT6 pair in one site counts 2)
    ff   flip-flops and latches
    mem  LUT-as-memory cells (RAM*, SRL*)

A CLB shared by two labels counts for both, so the rows can add up to more than "total".
"""
import re
import sys
from collections import defaultdict


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    labels = []
    for a in argv[1:]:
        label, _, rx = a.partition("=")
        if not rx:
            raise SystemExit("slr_clb.py: expected label=regex, got %r" % a)
        labels.append((label, re.compile(rx)))
    clb = defaultdict(set)
    cnt = defaultdict(int)
    lut_re = re.compile(r"^LUT[1-6]$")
    ff_re = re.compile(r"^(FD|LD)")
    mem_re = re.compile(r"^(RAM|SRL)")
    site_re = re.compile(r"^SLICE_X\d+Y(\d+)$")
    with open(argv[0]) as f:
        for line in f:
            name, loc, ref = line.rstrip("\n").split("\t")
            m = site_re.match(loc)
            if not m:
                continue
            slr = int(m.group(1)) // 300
            lab = "other"
            for label, rx in labels:
                if rx.search(name):
                    lab = label
                    break
            kind = "lut" if lut_re.match(ref) else "ff" if ff_re.match(ref) else "mem" if mem_re.match(ref) else None
            for key in (lab, "total"):
                clb[(key, slr)].add(loc)
                if kind:
                    cnt[(key, kind, slr)] += 1
    for key in [l for l, _ in labels] + ["other", "total"]:
        for kind in ("clb", "lut", "ff", "mem"):
            vals = [len(clb[(key, s)]) if kind == "clb" else cnt[(key, kind, s)] for s in range(3)]
            print("SLRCLB %-12s %-3s SLR0 %7d SLR1 %7d SLR2 %7d" % (key, kind, *vals))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
