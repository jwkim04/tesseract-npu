#!/usr/bin/env python3
"""hw/viz/fp_render.py — draw a placement dump of hw/viz/fp_dump.tcl like Vivado's Device view.

usage: fp_render.py <dump_dir> <out.png> --title TITLE [--mode core|link]

  --mode core   the dump is an out-of-context tsr_core (top = tsr_core)
  --mode link   the dump is a full device after the v++ link (XRT shell + kernel)

Every used CLB / DSP / RAMB / URAM site is drawn at its real RPM position in the colour of the
module that owns most of its cells; unused sites are tinted by type. Thick lines are SLR
boundaries (5 clock-region rows per SLR on the VU9P). The legend gives per-module totals: LUT and
FF count cells (a LUT6_2 counts as two LUTs, so LUT runs a little above report_utilization), DSP,
RAMB36 (+ RAMB18 / 2) and URAM count used sites and match report_utilization.
Interpreter: build/tools/svgvenv/bin/python (Pillow).
"""
import argparse
import collections
import re

from PIL import Image, ImageDraw, ImageFont

CLUSTER = ["#2E86C1", "#17A589", "#D4AC0D", "#CB4335"]

# (regex on the hierarchical name below tsr_core, label, colour); first match wins
RULES_CORE = [
    (r"g_cl\[\d\]\.u_(rp|bcp|wrp|wqp)/|g_asm\[\d\]\.u_(jp|sp)/|bc_d_q", "SLR-crossing pipes", "#1C2833"),
    (r"u_gather/g_asm\[(\d)\]", "cluster {0}", None),
    (r"g_cl\[(\d)\]", "cluster {0}", None),
    (r"u_gather", "gather (central)", "#8E44AD"),
    (r"^u_spm", "SPM (URAM)", "#E8735A"),
    (r"^u_dma", "DMA", "#A569BD"),
    (r"^u_qact", "QACT", "#5DADE2"),
    (r"^g_ch\[\d\]\.u_arb", "memory arbiters", "#6C3483"),
    # u_cmdproc / u_fetch are cell names in older checkpoints only
    (r"^u_cmdproc|^u_fetch|^u_csr", "control", "#48C9B0"),
    (r"^g_sh|^g_rst", "reset copies", "#1C2833"),
]
# (regex on the full hierarchical name of a linked design, label, colour)
RULES_LINK = [
    (r"/ulp/tsr_k\d/inst/u_kernel/(g_ch|u_ctrl|u_rst|g_st)", "platform adapter (bridges, control)", "#F5B041"),
    (r"/ulp/tsr_k\d/", "kernel (other)", "#48C9B0"),
    (r"/ulp/tsr_bw/", "M0 bandwidth kernel", "#2E86C1"),
    (r"/ulp/memory_subsystem/inst/memory/ddr4", "DDR4 controllers (memory subsystem)", "#C0392B"),
    (r"/ulp/memory_subsystem/", "memory interconnect", "#F1948A"),
    (r"^level0_i/blp/|^level0_i/ii_level0_pipe", "XRT static shell (BLP)", "#7F8C8D"),
    (r"^level0_i/ulp/", "shell dynamic glue (ULP)", "#BFC9CA"),
]
CORE_IN_KERNEL = "/u_kernel/u_core/"


def match(rules, name, prefix=""):
    for rx, lab, col in rules:
        m = re.search(rx, name)
        if m:
            if "{0}" in lab:
                c = int(m.group(1))
                return prefix + lab.format(c), CLUSTER[c % 4]
            return prefix + lab, col
    return None


def group(name, mode):
    if mode == "core":
        return match(RULES_CORE, name) or ("other", "#B3B6B7")
    if CORE_IN_KERNEL in name:
        sub = name.split(CORE_IN_KERNEL, 1)[1]
        return match(RULES_CORE, sub, "core: ") or ("core: other", "#48C9B0")
    return match(RULES_LINK, name) or ("other", "#B3B6B7")


def tint(t):
    if t.startswith("SLICE"):
        return "#E4E1D9"
    if t.startswith("DSP"):
        return "#EBDCCB"
    if t.startswith(("RAMB36", "RAMBFIFO36")):
        return "#DDD7EB"
    if t.startswith("URAM"):
        return "#D2E7E2"
    return None


def font(bold, size):
    name = "NotoSansMono-Bold.ttf" if bold else "NotoSansMono-Regular.ttf"
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/noto/" + name, size)
    except OSError:
        return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("out")
    ap.add_argument("--title", required=True)
    ap.add_argument("--mode", choices=["core", "link"], default="core")
    a = ap.parse_args()

    sites = {}
    cr_y = collections.defaultdict(lambda: [1 << 30, -1])
    for line in open(f"{a.dump}/sites.tsv"):
        n, t, x, y, cr = line.rstrip("\n").split("\t")
        if not x or not y:
            continue
        x, y = int(x), int(y)
        sites[n] = (x, y, t)
        m = re.match(r"X(\d+)Y(\d+)", cr)
        if m:
            r = cr_y[int(m.group(2))]
            r[0], r[1] = min(r[0], y), max(r[1], y)
    minx = min(v[0] for v in sites.values())
    maxx = max(v[0] for v in sites.values())
    miny = min(v[1] for v in sites.values())
    maxy = max(v[1] for v in sites.values())

    W = 1100
    H = int(W * 1.45)  # close to the VU9P's aspect in Vivado's device view
    sx = (W - 1) / (maxx - minx + 1)
    sy = (H - 1) / (maxy - miny + 1)
    PL, PT, PR, PB = 70, 90, 420, 40
    img = Image.new("RGB", (W + PL + PR, H + PT + PB), "#F4F1EA")
    dr = ImageDraw.Draw(img)
    fb, fs = font(True, 18), font(False, 15)

    def px(x, y):
        return PL + (x - minx) * sx, PT + (maxy - y) * sy

    # a site spans to the next site column (x) and to the next site of its kind in its column (y)
    ux = sorted({v[0] for v in sites.values()})
    nextx = {p: min(q - p, 4) for p, q in zip(ux, ux[1:])}
    cols = collections.defaultdict(list)
    for x, y, t in sites.values():
        if tint(t):
            cols[(x, t[:4])].append(y)
    span = {}
    for k, ys in cols.items():
        ys.sort()
        for p, q in zip(ys, ys[1:]):
            span[(k, p)] = min(q - p, 40)
        span[(k, ys[-1])] = span.get((k, ys[-2]), 2) if len(ys) > 1 else 2
    geo = {}
    for n, (x, y, t) in sites.items():
        if tint(t):
            w = nextx.get(x, 2) * sx
            h = span[((x, t[:4]), y)] * sy
            X, Y = px(x, y)
            geo[n] = (X, Y - h + sy, X + max(w - 0.6, 1), Y + sy)
            dr.rectangle(geo[n], fill=tint(t))

    per_site = collections.defaultdict(collections.Counter)
    colour, res = {}, collections.defaultdict(collections.Counter)
    used = collections.defaultdict(set)
    for line in open(f"{a.dump}/cells.tsv"):
        n, loc, ref = line.rstrip("\n").split("\t")
        s = sites.get(loc)
        if not s:
            continue
        g, c = group(n, a.mode)
        colour[g] = c
        per_site[loc][g] += 1
        t = s[2]
        kind = ("DSP" if t.startswith("DSP") else "URAM" if t.startswith("URAM") else
                "RAMB36" if t.startswith(("RAMB36", "RAMBFIFO36")) else
                "RAMB18" if t.startswith(("RAMB18", "RAMBFIFO18")) else None)
        if kind:
            used[(g, kind)].add(loc)
        if ref.startswith("LUT"):
            res[g]["LUT"] += 1
        elif ref[:3] in ("FDR", "FDC", "FDS", "FDP", "FDE"):
            res[g]["FF"] += 1
        res[g]["cells"] += 1
    for (g, k), v in used.items():
        res[g][k] = len(v)

    cent = collections.defaultdict(lambda: [0.0, 0.0, 0])
    for loc, c in per_site.items():
        g = c.most_common(1)[0][0]
        box = geo.get(loc)
        if box is None:
            X, Y = px(*sites[loc][:2])
            box = (X, Y, X + 2, Y + 2)
        dr.rectangle(box, fill=colour[g])
        cc = cent[g]
        cc[0] += box[0]
        cc[1] += box[1]
        cc[2] += 1

    for r in sorted(cr_y):
        if r % 5 == 0 and r - 1 in cr_y:
            _, Y = px(minx, (cr_y[r][0] + cr_y[r - 1][1]) / 2)
            dr.line([PL, Y, PL + W, Y], fill="#222222", width=3)
    for s in range(len(cr_y) // 5):
        _, Y = px(minx, (cr_y[s * 5][0] + cr_y[s * 5 + 4][1]) / 2)
        dr.text((8, Y - 10), f"SLR{s}", fill="#222222", font=fb)
    dr.rectangle([PL, PT, PL + W, PT + H], outline="#222222", width=2)
    dr.text((PL, 18), a.title, fill="#111111", font=fb)
    dr.text((PL, 46), "each coloured block is a used site (CLB / DSP / RAMB / URAM) in its module "
            "colour; thick lines = SLR boundaries", fill="#555555", font=fs)

    order = sorted(res, key=lambda g: -res[g]["cells"])
    lx, ly = PL + W + 24, PT
    for g in order:
        r = res[g]
        dr.rectangle([lx, ly + 3, lx + 18, ly + 21], fill=colour[g], outline="#222222")
        dr.text((lx + 28, ly), g, fill="#111111", font=fs)
        dr.text((lx + 28, ly + 18), f"LUT {r['LUT']:,}  FF {r['FF']:,}", fill="#444444", font=fs)
        dr.text((lx + 28, ly + 36), f"DSP {r['DSP']:,}  RAMB36 {r['RAMB36'] + r['RAMB18'] / 2:g}"
                f"  URAM {r['URAM']:,}", fill="#444444", font=fs)
        ly += 64
    for g, (x, y, n) in cent.items():
        if "cluster" in g and n:
            lab = g.replace("core: ", "").upper()
            cx, cy, tw = x / n, y / n, dr.textlength(lab, font=fb)
            dr.rounded_rectangle([cx - tw / 2 - 8, cy - 14, cx + tw / 2 + 8, cy + 14], radius=6,
                                 fill="#1C2833")
            dr.text((cx - tw / 2, cy - 11), lab, fill="#FFFFFF", font=fb)
    img.save(a.out)
    print("wrote", a.out, img.size)
    for g in order:
        print(f"{g:44s} {dict(res[g])}")


if __name__ == "__main__":
    main()
