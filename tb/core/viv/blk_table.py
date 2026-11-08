#!/usr/bin/env python3
"""Tables of block_ooc runs: blk_table.py <rundir> ... (each <block>_<period>/vivado.log of blk_ooc.sh).

Prints one row per run: block, period, routed WNS / TNS / failing endpoints, the register-to-register and
I/O worst slack, LUT / FF / CLB, and the placed (estimate) WNS. Markdown, for docs/platform-u200.md §8.9.
"""
import re
import sys
from pathlib import Path


def parse(log: Path):
    d = {}
    for line in log.read_text(errors="replace").splitlines():
        if not line.startswith(("BOOC", "COOC util")):
            continue
        m = re.match(r"BOOC (placed|routed|synth) wns (\S+) whs (\S+) tns (\S+) fail_ep (\S+)(.*)", line)
        if m:
            st = m.group(1)
            d.setdefault(st, {}).update(wns=m.group(2), whs=m.group(3), tns=m.group(4), fail=m.group(5))
            for k, v in re.findall(r"(\w+)_wns (\S+)", m.group(6)):
                d[st][k] = v
            continue
        m = re.match(r"BOOC dev\((\w+)\) CLBs (\S+) CLB_LUTs (\S+) CLB_regs (\S+)", line)
        if m:
            d.setdefault(m.group(1), {}).update(clb=m.group(2), clut=m.group(3), creg=m.group(4))
            continue
        m = re.match(r"COOC util\((\w+)\) LUT (\d+) FF (\d+) DSP (\d+) RAMB36 (\d+) RAMB18 (\d+) URAM (\d+)", line)
        if m:
            d.setdefault(m.group(1), {}).update(lut=m.group(2), ff=m.group(3), dsp=m.group(4), b36=m.group(5), uram=m.group(7))
    return d


def main(argv):
    print("| Block | Period | Routed WNS | TNS | Failing | r2r | in | out | Placed WNS | LUT | FF | CLB | DSP |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for a in argv:
        p = Path(a)
        log = p / "vivado.log"
        if not log.exists():
            continue
        m = re.match(r"(\w+?)_([\d.]+)", p.name)
        blk, per = (m.group(1), m.group(2)) if m else (p.name, "?")
        d = parse(log)
        r = d.get("routed") or d.get("placed") or d.get("synth") or {}
        pl = d.get("placed", {})
        print("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            blk, per, r.get("wns", "-"), r.get("tns", "-"), r.get("fail", "-"), r.get("r2r", "-"), r.get("io_in", "-"),
            r.get("io_out", "-"), pl.get("wns", "-"), r.get("lut", d.get("synth", {}).get("lut", "-")),
            r.get("ff", d.get("synth", {}).get("ff", "-")), r.get("clb", "-"), r.get("dsp", "-")))


if __name__ == "__main__":
    main(sys.argv[1:])
