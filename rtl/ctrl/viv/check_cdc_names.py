#!/usr/bin/env python3
"""check_cdc_names.py — the instance-name conventions that rtl/ctrl/viv/tsr_ctrl_cdc.xdc selects cells by.

The constraints of the clock-domain crossings (ASYNC_REG on the synchronizer stages, the report_cdc waivers
of the handshake registers) find their cells by instance name: a tsr_sync instance must be called u_sync*,
a tsr_rst_sync u_rst*, a tsr_cdc_hs u_hs*. (tsr_cdc_fifo and tsr_cdc_gray are found by their register names.)
An instance outside the convention would silently lose its ASYNC_REG and its waivers, so rtl/ctrl/lint.sh
runs this over every RTL source outside rtl/ctrl/cdc/ (the primitives themselves).

usage: check_cdc_names.py [rtl dir]      exit status 0 = every instance follows the convention
"""
import re
import sys
from pathlib import Path

RULES = {"tsr_sync": "u_sync", "tsr_rst_sync": "u_rst", "tsr_cdc_hs": "u_hs"}


def instances(text):
    """(module, instance name, line) of every instantiation of a rule module in text (comments removed)."""
    text = re.sub(r"//[^\n]*", lambda m: " " * len(m.group(0)), text)
    text = re.sub(r"/\*.*?\*/", lambda m: re.sub(r"[^\n]", " ", m.group(0)), text, flags=re.S)
    for m in re.finditer(r"\b(%s)\b" % "|".join(RULES), text):
        i = m.end()
        while i < len(text) and text[i].isspace():
            i += 1
        if text[i:i + 1] == "#":                      # parameter list: skip the balanced parentheses
            i += 1
            while text[i].isspace():
                i += 1
            if text[i] != "(":
                continue
            depth = 0
            while i < len(text):
                depth += text[i] == "("
                depth -= text[i] == ")"
                i += 1
                if depth == 0:
                    break
        while i < len(text) and text[i].isspace():
            i += 1
        n = re.match(r"[A-Za-z_][A-Za-z0-9_$]*", text[i:])
        if n:
            yield m.group(1), n.group(0), text.count("\n", 0, m.start()) + 1


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2]
    bad = 0
    seen = 0
    for f in sorted(root.rglob("*.sv")):
        rel = f.relative_to(root)
        if rel.parts[:2] == ("ctrl", "cdc"):
            continue
        for mod, name, line in instances(f.read_text()):
            seen += 1
            if not name.startswith(RULES[mod]):
                print("cdc names: %s:%d: %s instance '%s' must be named %s*" % (rel, line, mod, name, RULES[mod]))
                bad += 1
    print("cdc names: %d instances checked, %d outside the convention" % (seen, bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
