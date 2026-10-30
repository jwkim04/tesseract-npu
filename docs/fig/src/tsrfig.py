"""tsrfig: the shared drawing kit for the Tesseract architecture figures.

Every figure script in docs/fig/src builds one `Fig`, draws on it in z-order (later calls paint on
top) and calls `save()`. The look follows docs/fig/src/STYLE.md: a warm card, lighter panels,
chamfered pastel blocks with a thin ink outline, hollow block arrows, thin connectors, leader-line
callouts and a monospace font. Text is converted to outlines (no <text>, no font on the viewer):
each glyph is defined once in <defs> in font units and placed with <use>. Output is deterministic:
the same calls give the same bytes.

Coordinates are SVG user units, y down; the card is 1600 units wide by convention.

    from tsrfig import Fig, split
    f = Fig("demo", 1600, 900, title="Demo", desc="One block.")
    f.panel(80, 80, 1440, 740, label="HOST PC")
    f.box(160, 200, 280, 140, "compute", "CLUSTER\\n0", sub="128 MACs")
    f.arrow([(300, 340), (300, 480)])
    f.callout("WEIGHT STREAM", (600, 410), target=(300, 410))
    f.save()           # docs/fig/demo.svg + scratchpad preview PNG

Dark variant: `TSR_FIG_THEME=dark build/tools/svgvenv/bin/python docs/fig/src/<name>.py` writes
docs/fig/<name>-dark.svg (and the previews <name>-dark.png / <name>-dark_gh.png).
"""
from __future__ import annotations

import hashlib
import os
import re
import sys
from pathlib import Path

from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont

# --------------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parent
FIG_DIR = SRC_DIR.parent                      # docs/fig
REPO = FIG_DIR.parent.parent
PNG_DIR = Path(os.environ.get("TSR_FIG_PNG_DIR", REPO / "build" / "fig" / "png"))  # git-ignored
FONT_FILES = {                                 # Noto Sans Mono 2.006, SIL OFL 1.1 (FONTS.md)
    "regular": os.environ.get("TSR_FIG_FONT_REGULAR",
                              "/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf"),
    "bold": os.environ.get("TSR_FIG_FONT_BOLD",
                           "/usr/share/fonts/truetype/noto/NotoSansMono-Bold.ttf"),
}
FONT_SHA256 = {          # the files the committed figures were made with (Ubuntu fonts-noto-mono)
    "regular": "6b692c4b6d15ccf59f1c1fe8d11cb8a92f51960f3e9f1f523781755a3af7e29f",
    "bold": "b22563d5f03b345a62115e78d8789c9ed99c2cfa8059fad6e7a5c4ced288e21e",
}

# --------------------------------------------------------------------------------------------
# Palette (STYLE.md lists the same values; change both together)
# --------------------------------------------------------------------------------------------
# Two themes with the same roles. "light" is the default and what every committed <name>.svg uses.
# TSR_FIG_THEME=dark writes <name>-dark.svg with the dark palette, for the <picture> embeds of the
# README figures (STYLE.md "Dark variants"). The theme is read once, at import, so a script's
# `from tsrfig import INK, ...` sees the palette of the run.
THEME = os.environ.get("TSR_FIG_THEME", "light")
if THEME not in ("light", "dark"):
    raise SystemExit(f"tsrfig: TSR_FIG_THEME must be 'light' or 'dark', not {THEME!r}")

_PALETTE = {
    "light": {
        "INK": "#2B2B2B",        # outlines, text, dots, dark tags
        "INK_SOFT": "#5C5B55",   # secondary text (notes, panel labels on tinted bands)
        "CARD": "#ECEBE4",       # outer card
        "PANEL": "#F8F8F5",      # inner panel
        "BAND": "#E7E5DC",       # tinted band inside a panel (SLRs, lanes); a step darker than CARD
        "ARROW": "#EAF4F1",      # hollow-arrow fill
        "WHITE": "#FFFFFF",      # plain block
        "control": "#A8E0D5",    # mint: sequencing, control bridges
        "compute": "#9EDCF2",    # sky: MAC clusters, datapaths, functional units
        "memory": "#F8AFA3",     # salmon: SPM, DRAM, buffers
        "io": "#DCD5F7",         # lavender: interconnect, buses, memory/AXI bridges, DMA
        "host": "#F4DE9C",       # butter: host software (llama.cpp, backend, transport)
        "adapter": "#F9C89B",    # peach: per-target platform adapter (control / AXI4 bridges)
        "vendor": "#DAD8CF",     # stone: third-party / vendor IP (XRT shell, PCIe)
    },
    "dark": {
        "INK": "#ECEBE4",
        "INK_SOFT": "#B4B3A9",
        "CARD": "#1D1E1B",
        "PANEL": "#292A26",
        "BAND": "#33342F",
        "ARROW": "#33403C",
        "WHITE": "#3C3D38",
        "control": "#2D6A5F",
        "compute": "#276683",
        "memory": "#8E4539",
        "io": "#574D8B",
        "host": "#76651F",
        "adapter": "#8F5528",
        "vendor": "#55544C",
    },
}[THEME]

INK = _PALETTE["INK"]
INK_SOFT = _PALETTE["INK_SOFT"]
CARD = _PALETTE["CARD"]
PANEL = _PALETTE["PANEL"]
BAND = _PALETTE["BAND"]
ARROW = _PALETTE["ARROW"]
WHITE = _PALETTE["WHITE"]

ROLES = {                # semantic role -> block fill
    "control": _PALETTE["control"],
    "compute": _PALETTE["compute"],
    "memory": _PALETTE["memory"],
    "io": _PALETTE["io"],
    "host": _PALETTE["host"],
    "adapter": _PALETTE["adapter"],
    "vendor": _PALETTE["vendor"],
    "neutral": WHITE,       # plain block
}


def tint(hex_color: str, t: float, base: str = PANEL) -> str:
    """Mix `hex_color` toward `base` by t (0 = colour, 1 = base). Used for planned blocks."""
    a = [int(hex_color[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(base[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02X}" for x, y in zip(a, b))


# --------------------------------------------------------------------------------------------
# Strokes, geometry and type scale (units at 1600 wide)
# --------------------------------------------------------------------------------------------
GRID = 8                 # all positions and sizes on multiples of 8 (4 for fine nudges)
SW = 2                   # outline of blocks, panels, card, hollow arrows
SW_LINE = 2              # thin connectors
SW_LEADER = 1.5          # callout leader lines
DOT_R = 5                # callout dot radius
DASH = "9 6"             # planned outlines and optional/dotted paths
CUT = 14                 # chamfer of a block
CUT_PANEL = 20           # chamfer of a panel
CARD_R = 36              # corner radius of the card
ARROW_W, ARROW_HW, ARROW_HL = 24, 54, 26       # hollow arrow: shaft, head width, head length
FAT_W, FAT_HW, FAT_HL = 40, 84, 36             # hollow "bulk data" arrow (weight streams)
HEAD_L, HEAD_W = 12, 11                        # solid arrowhead of thin connectors

TYPE = {                 # role -> (size, weight, tracking in em, line-height factor)
    "title": (28, "bold", 0.02, 1.25),     # optional card heading
    "panel": (20, "bold", 0.10, 1.25),     # band label (SLR0), uppercase, tracked
    "label": (24, "bold", 0.02, 1.25),     # block title, uppercase, 1-2 lines
    "label_s": (21, "bold", 0.02, 1.25),   # block title in a small block
    "sub": (19, "regular", 0.0, 1.35),     # block detail lines (identifiers keep their case)
    "callout": (19, "medium", 0.02, 1.4),  # leader-line callouts and arrow names, uppercase
    "note": (17, "regular", 0.0, 1.35),    # footnotes, legend entries
    "tag": (15, "bold", 0.08, 1.2),        # badges: panel names, PLANNED · M6, x4
    # Large type for the product-style stack figure (STYLE.md "Large type"): the same families
    # scaled up, so the figure stays readable at README width and on a phone.
    "label_l": (28, "bold", 0.02, 1.25),   # tile title in `stack`
    "sub_l": (21, "regular", 0.0, 1.35),   # tile detail lines in `stack`
    "panel_l": (21, "bold", 0.10, 1.25),   # layer names in `stack`
    "note_l": (19, "regular", 0.0, 1.35),  # footnote in `stack`
    "tag_l": (17, "bold", 0.08, 1.2),      # status badges and layer names in `stack`
}
MEDIUM_STROKE = 34       # font units of same-colour stroke that turns Regular into "medium"


def snap(v: float, g: int = GRID) -> int:
    """Round to the grid."""
    return int(round(v / g) * g)


def split(x0: float, total: float, n: int, gap: float):
    """n equal cells with `gap` between them across [x0, x0+total]: list of (x, w)."""
    w = (total - gap * (n - 1)) / n
    return [(x0 + i * (w + gap), w) for i in range(n)]


def centers(x0: float, total: float, n: int):
    """Centres of n equal columns across [x0, x0+total]."""
    w = total / n
    return [x0 + w * (i + 0.5) for i in range(n)]


def fmt(v: float) -> str:
    """Deterministic number format: at most 2 decimals, no trailing zeros, no -0."""
    s = f"{round(float(v), 2):.2f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def _pts(points) -> str:
    return " ".join(f"{fmt(x)},{fmt(y)}" for x, y in points)


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


# --------------------------------------------------------------------------------------------
# Fonts: glyph outlines in font units, y flipped
# --------------------------------------------------------------------------------------------
class _Font:
    def __init__(self, key: str, path: str):
        self.key = key
        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        if digest != FONT_SHA256[key]:
            print(f"tsrfig: warning: {path} differs from the font in FONTS.md; "
                  "glyph outlines (and the SVG bytes) will change", file=sys.stderr)
        self.tt = TTFont(path)
        self.gs = self.tt.getGlyphSet()
        self.cmap = self.tt.getBestCmap()
        self.upm = self.tt["head"].unitsPerEm
        self.adv = self.tt["hmtx"]["zero"][0]
        self.cap = self.tt["OS/2"].sCapHeight
        self.paths: dict[str, str] = {}

    def glyph(self, ch: str, slashed_zero: bool) -> str:
        if ch == "0" and slashed_zero and "zero.slash" in self.gs:
            return "zero.slash"
        name = self.cmap.get(ord(ch))
        if name is None:
            raise ValueError(f"glyph for {ch!r} (U+{ord(ch):04X}) missing in {self.key} font")
        return name

    def path(self, gname: str) -> str:
        if gname not in self.paths:
            pen = SVGPathPen(self.gs)
            self.gs[gname].draw(TransformPen(pen, (1, 0, 0, -1, 0, 0)))
            self.paths[gname] = pen.getCommands()
        return self.paths[gname]


_FONTS: dict[str, _Font] = {}


def font(weight: str) -> _Font:
    key = "bold" if weight == "bold" else "regular"
    if key not in _FONTS:
        _FONTS[key] = _Font(key, FONT_FILES[key])
    return _FONTS[key]


def _style(role_or_size, weight=None, track=None, lh=None):
    """Resolve a TYPE role name or an explicit size into (size, weight, track_em, lh)."""
    if isinstance(role_or_size, str):
        s, w, t, l = TYPE[role_or_size]
    else:
        s, w, t, l = float(role_or_size), "regular", 0.0, 1.3
    return s, weight or w, (t if track is None else track), (l if lh is None else lh)


def text_width(s: str, size: float = 22, track: float = 0.0, weight: str = "regular") -> float:
    """Advance width of one line (monospace: chars x advance + tracking between chars)."""
    f = font(weight)
    n = len(s)
    return n * f.adv * size / f.upm + max(n - 1, 0) * track * size


def wrap(text: str, maxw: float, role="note") -> str:
    """Greedy word wrap of `text` to lines of at most `maxw` units in the TYPE role `role`, joined
    by \\n (so a footnote never leaves the card)."""
    size, weight, track, _ = _style(role)
    lines, cur = [], ""
    for word in text.split(" "):
        trial = (cur + " " + word).strip()
        if cur and text_width(trial, size, track, weight) > maxw:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    lines.append(cur)
    return "\n".join(lines)


def measure(text: str, role="label", **kw):
    """(width, height) of a text block; height runs from the first cap top to the last baseline."""
    size, weight, track, lh = _style(role, kw.get("weight"), kw.get("track"), kw.get("lh"))
    lines = text.split("\n")
    w = max(text_width(l, size, track, weight) for l in lines)
    f = font(weight)
    return w, f.cap * size / f.upm + (len(lines) - 1) * size * lh


# --------------------------------------------------------------------------------------------
# The figure
# --------------------------------------------------------------------------------------------
class Fig:
    """One figure: a card of w x h units. Draw calls append in z-order; save() writes SVG + PNG."""

    def __init__(self, name: str, w: int = 1600, h: int = 1000, title: str = "", desc: str = "",
                 card: bool = True, slashed_zero: bool = True):
        self.name, self.w, self.h = name, w, h
        self.title, self.desc = title, desc
        self.slashed_zero = slashed_zero
        self.body: list[str] = []
        self.used: dict[str, set[str]] = {"regular": set(), "bold": set()}
        if card:
            self.card()

    # -- raw ----------------------------------------------------------------------------------
    def raw(self, svg: str):
        """Append raw SVG markup (escape hatch; keep it deterministic)."""
        self.body.append(svg)

    # -- backgrounds ----------------------------------------------------------------------------
    def card(self, fill: str = CARD):
        """Rounded outer card covering the whole figure, thin ink outline."""
        h = SW / 2
        self.body.append(
            f'<rect x="{fmt(h)}" y="{fmt(h)}" width="{fmt(self.w - SW)}" height="{fmt(self.h - SW)}" '
            f'rx="{CARD_R}" fill="{fill}" stroke="{INK}" stroke-width="{SW}"/>')

    def panel(self, x, y, w, h, label: str = "", fill: str = PANEL, cut: float = CUT_PANEL,
              dashed: bool = False, label_at: str = "start", stroke: str = INK,
              label_role: str = "tag"):
        """Inner panel (chamfered, lighter). `label` is its name, drawn as a dark tag sitting on
        the top edge (label_at: start = left, end = right, middle). fill=None: outline only, for
        a region nested in a panel (draw bands under it first)."""
        self.poly(chamfer(x, y, w, h, cut), fill or "none", stroke=stroke, dashed=dashed)
        if label:
            off = cut + 16
            tx = {"start": x + off, "end": x + w - off, "middle": x + w / 2}[label_at]
            self.tag(tx, y, label, anchor=label_at, role=label_role)

    def band(self, x, y, w, h, fill: str = BAND, label: str = "", label_xy=None,
             anchor: str = "start", color: str = INK_SOFT):
        """Flat tinted rectangle without outline (SLR bands, lanes, regions) plus optional label."""
        self.body.append(f'<rect x="{fmt(x)}" y="{fmt(y)}" width="{fmt(w)}" height="{fmt(h)}" '
                         f'fill="{fill}"/>')
        if label:
            lx, ly = label_xy if label_xy else (x + 20, y + 36)
            self.text(lx, ly, label, "panel", anchor=anchor, color=color)

    # -- blocks ---------------------------------------------------------------------------------
    def poly(self, points, fill, stroke: str = INK, sw: float = SW, dashed: bool = False,
             join: str = "miter"):
        dash = f' stroke-dasharray="{DASH}"' if dashed else ""
        self.body.append(f'<polygon points="{_pts(points)}" fill="{fill}" stroke="{stroke}" '
                         f'stroke-width="{fmt(sw)}" stroke-linejoin="{join}"{dash}/>')

    def box(self, x, y, w, h, role: str = "compute", label: str = "", sub: str = "",
            cut: float = CUT, planned: bool = False, tag: str = "", fill: str = "",
            label_role: str = "label", sub_role: str = "sub", stack: int = 0,
            align: str = "middle", pad: float = 20, tag_style: str = "", tag_role: str = "tag",
            plan_tag: bool = True):
        """Chamfered block filled by `role` (ROLES key). `label` is the uppercase title (\\n for
        lines), `sub` the detail lines below it. planned=True: dashed outline, paler fill, and a
        PLANNED tag unless `tag` is given. stack=n draws n offset outlines behind the block for a
        repeated unit. `tag_style` ("dark", "outline", "planned") picks the badge look of a given
        `tag` on a solid block; a planned block has the planned badge unless plan_tag=False (use
        that inside a lane or panel that is already marked planned). Returns the block's centre."""
        col = fill or ROLES[role]
        if planned:
            col = tint(col, 0.55)
        for i in range(stack, 0, -1):
            d = 8 * i
            self.poly(chamfer(x + d, y - d, w, h, cut), tint(col, 0.35 * i), dashed=planned)
        self.poly(chamfer(x, y, w, h, cut), col, dashed=planned)
        self.label_block(x, y, w, h, label, sub, label_role, sub_role, align, pad)
        if planned:
            if plan_tag:
                self.tag(x + w - cut - 8, y, tag or "PLANNED", anchor="end", planned=True,
                         role=tag_role)
        elif tag:
            self.tag(x + w - cut - 8, y, tag, anchor="end", style=tag_style, role=tag_role)
        return x + w / 2, y + h / 2

    def label_block(self, x, y, w, h, label="", sub="", label_role="label", sub_role="sub",
                    align="middle", pad=20):
        """Title + detail lines, vertically centred in the rect, aligned middle/start."""
        ls, lw, lt, llh = _style(label_role)
        ss, sw_, st, slh = _style(sub_role)
        L = label.split("\n") if label else []
        S = sub.split("\n") if sub else []
        fL, fS = font(lw), font(sw_)
        capL, capS = fL.cap * ls / fL.upm, fS.cap * ss / fS.upm
        base = []                                   # (baseline offset, line, role)
        yy = 0.0
        if L:
            yy = capL
            for i, s in enumerate(L):
                base.append((yy + i * ls * llh, s, label_role))
            yy = base[-1][0]
        if S:
            first = yy + (ss * 1.55 if L else capS)
            for i, s in enumerate(S):
                base.append((first + i * ss * slh, s, sub_role))
            yy = base[-1][0]
        if not base:
            return
        top = y + (h - yy) / 2
        tx = x + w / 2 if align == "middle" else x + pad
        for off, s, r in base:
            self.text(tx, top + off, s, r, anchor=align)

    def tag(self, x, y, text: str, anchor: str = "middle", planned: bool = False,
            style: str = "", role: str = "tag"):
        """Small badge centred vertically on y, so it sits on a panel's or block's top edge.
        anchor is the side of x: start / middle / end. Names (panels, x4 counts) are dark with
        light text; planned=True gives the PLANNED style: light fill, dashed ink outline.
        `style` names the look: "dark" (default), "planned" (as planned=True) or "outline"
        (light fill, solid ink outline: the SIMULATOR status badge of `stack`). `role` is the
        TYPE role of the text ("tag", or "tag_l" in the large `stack` figure)."""
        style = style or ("planned" if planned else "dark")
        size, weight, track, _ = _style(role)
        tw = text_width(text, size, track, weight)
        f = font(weight)
        cap = f.cap * size / f.upm
        bw, bh = tw + 24, cap + 16
        bx = x - bw / 2 if anchor == "middle" else (x - bw if anchor == "end" else x)
        if style == "planned":
            self.poly(chamfer(bx, y - bh / 2, bw, bh, 5), PANEL, dashed=True, sw=1.5)
            color = INK
        elif style == "outline":
            self.poly(chamfer(bx, y - bh / 2, bw, bh, 5), PANEL, sw=1.5)
            color = INK
        else:
            self.poly(chamfer(bx, y - bh / 2, bw, bh, 5), INK, stroke=INK, sw=1)
            color = CARD
        self.text(bx + bw / 2, y + cap / 2, text, role, anchor="middle", color=color)
        return bx, bw

    def legend(self, x, y, items, gap: float = 40, anchor: str = "middle"):
        """One row of swatches: items = [(role or hex, "LABEL", planned?), ...]. y is the row's
        centre; anchor positions the whole row on x. Returns the row width."""
        sw_w, sw_h, sp = 30, 20, 12
        size, weight, track, _ = _style("note")
        widths = [sw_w + sp + text_width(lab, size, track, weight) for _, lab, *_ in items]
        total = sum(widths) + gap * (len(items) - 1)
        cx = x - total / 2 if anchor == "middle" else (x - total if anchor == "end" else x)
        for (role, lab, *rest), wdt in zip(items, widths):
            planned = bool(rest and rest[0])
            col = ROLES.get(role, role)
            if planned:
                col = tint(col, 0.55)
            self.poly(chamfer(cx, y - sw_h / 2, sw_w, sw_h, 5), col, dashed=planned, sw=1.5)
            self.text(cx + sw_w + sp, y, lab, "note", valign="middle")
            cx += wdt + gap
        return total

    # -- arrows and lines -----------------------------------------------------------------------
    def arrow(self, pts, w: float = ARROW_W, hw: float = ARROW_HW, hl: float = ARROW_HL,
              head: bool = True, tail: bool = False, fill: str = ARROW, dashed: bool = False,
              fat: bool = False):
        """Hollow block arrow along an axis-aligned polyline `pts` (2+ points). head: arrowhead at
        the last point; tail: also at the first (two-way). fat=True uses the bulk-data size."""
        if fat:
            w, hw, hl = FAT_W, FAT_HW, FAT_HL
        self.poly(hollow_arrow(pts, w, hw, hl, head, tail), fill, dashed=dashed)

    def line(self, pts, end: bool = True, start: bool = False, dashed: bool = False,
             color: str = INK, sw: float = SW_LINE):
        """Thin connector along `pts` with solid arrowheads at the end and/or start."""
        pts = [tuple(p) for p in pts]
        a, b = list(pts), list(pts)
        if end:
            b[-1] = _back(pts[-1], pts[-2], HEAD_L * 0.8)
        if start:
            b[0] = _back(pts[0], pts[1], HEAD_L * 0.8)
        dash = f' stroke-dasharray="{DASH}"' if dashed else ""
        self.body.append(f'<polyline points="{_pts(b)}" fill="none" stroke="{color}" '
                         f'stroke-width="{fmt(sw)}" stroke-linejoin="miter"{dash}/>')
        if end:
            self.body.append(_head(a[-1], a[-2], color))
        if start:
            self.body.append(_head(a[0], a[1], color))

    def dot(self, x, y, r: float = DOT_R, color: str = INK):
        self.body.append(f'<circle cx="{fmt(x)}" cy="{fmt(y)}" r="{fmt(r)}" fill="{color}"/>')

    def callout(self, text: str, xy, target, via=(), anchor: str = "middle", side: str = "",
                gap: float = 14, role: str = "callout", dot: bool = True, color: str = INK):
        """Callout text block at `xy` (anchor = horizontal alignment of x; y = vertical centre),
        with a thin leader from the block's `side` (left/right/top/bottom; inferred from the
        target when empty) through `via` points to `target`, ending in a dot."""
        tw, th = measure(text, role)
        x, y = xy
        x0 = x - tw / 2 if anchor == "middle" else (x - tw if anchor == "end" else x)
        y0 = y - th / 2
        self.text(x, y0, text, role, anchor=anchor, valign="top", color=color)
        if target is None:
            return
        first = via[0] if via else target
        if not side:
            dx, dy = first[0] - x, first[1] - y
            if x0 <= first[0] <= x0 + tw:
                side = "bottom" if dy > 0 else "top"
            else:
                side = "right" if dx > 0 else "left"
        sx, sy = {"left": (x0 - gap, y), "right": (x0 + tw + gap, y),
                  "top": (x0 + tw / 2, y0 - gap),
                  "bottom": (x0 + tw / 2, y0 + th + gap + 4)}[side]
        pts = [(sx, sy), *via, target]
        self.body.append(f'<polyline points="{_pts(pts)}" fill="none" stroke="{color}" '
                         f'stroke-width="{fmt(SW_LEADER)}" stroke-linejoin="miter"/>')
        if dot:
            self.dot(*target, color=color)

    def bracket(self, x1, x2, y, label: str = "", up: bool = False, tick: float = 10,
                role: str = "panel", color: str = INK):
        """Horizontal dimension bracket from x1 to x2 at y with end ticks (pointing up when
        up=True) and a centred label on the far side. For spans: SLRs, channel groups, x4."""
        t = -tick if up else tick
        self.body.append(f'<polyline points="{_pts([(x1, y - t), (x1, y), (x2, y), (x2, y - t)])}" '
                         f'fill="none" stroke="{color}" stroke-width="{fmt(SW_LEADER)}"/>')
        if label:
            size = TYPE[role][0] if role in TYPE else 18
            ly = y + (size * 0.72 + 12 if not up else -12)
            self.text((x1 + x2) / 2, ly, label, role, anchor="middle", color=color)

    def vline(self, x, y1, y2, dashed: bool = True, color: str = INK_SOFT, sw: float = 1.5):
        """Plain vertical rule (band boundaries)."""
        dash = f' stroke-dasharray="{DASH}"' if dashed else ""
        self.body.append(f'<line x1="{fmt(x)}" y1="{fmt(y1)}" x2="{fmt(x)}" y2="{fmt(y2)}" '
                         f'stroke="{color}" stroke-width="{fmt(sw)}"{dash}/>')

    def hline(self, x1, x2, y, dashed: bool = True, color: str = INK_SOFT, sw: float = 1.5):
        dash = f' stroke-dasharray="{DASH}"' if dashed else ""
        self.body.append(f'<line x1="{fmt(x1)}" y1="{fmt(y)}" x2="{fmt(x2)}" y2="{fmt(y)}" '
                         f'stroke="{color}" stroke-width="{fmt(sw)}"{dash}/>')

    # -- text -----------------------------------------------------------------------------------
    def text(self, x, y, text: str, role="label", anchor: str = "start", valign: str = "baseline",
             color: str = INK, weight: str = "", track=None, lh=None):
        """Text as outlines. role: a TYPE key or a size. anchor: start/middle/end. valign:
        baseline (y = first baseline), top (y = cap top), middle (block centred on y).
        Multi-line with \\n; each line is aligned on its own."""
        size, weight, track, lh = _style(role, weight or None, track, lh)
        f = font(weight)
        k = size / f.upm
        lines = text.split("\n")
        cap = f.cap * k
        if valign == "top":
            y = y + cap
        elif valign == "middle":
            y = y - ((len(lines) - 1) * size * lh + cap) / 2 + cap
        medium = _style(role)[1] == "medium" if isinstance(role, str) else False
        stroke = (f' stroke="{color}" stroke-width="{MEDIUM_STROKE}" stroke-linejoin="round"'
                  if medium and weight != "bold" else "")
        track_fu = track * f.upm
        for i, line in enumerate(lines):
            tw = text_width(line, size, track, weight)
            lx = x - tw / 2 if anchor == "middle" else (x - tw if anchor == "end" else x)
            ly = y + i * size * lh
            uses = []
            for j, ch in enumerate(line):
                if ch == " ":
                    continue
                g = f.glyph(ch, self.slashed_zero)
                self.used[f.key].add(g)
                gx = round(j * (f.adv + track_fu))
                xa = f' x="{gx}"' if gx else ""
                uses.append(f'<use xlink:href="#{_gid(f.key, g)}"{xa}/>')
            if uses:
                self.body.append(f'<g transform="translate({fmt(lx)} {fmt(ly)}) scale({k:.6g})" '
                                 f'fill="{color}"{stroke}>{"".join(uses)}</g>')

    # -- output ---------------------------------------------------------------------------------
    def svg(self) -> str:
        defs = []
        for key in ("bold", "regular"):
            f = _FONTS.get(key)
            for g in sorted(self.used[key]):
                defs.append(f'<path id="{_gid(key, g)}" d="{f.path(g)}"/>')
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
                f'width="{self.w}" height="{self.h}" viewBox="0 0 {self.w} {self.h}" role="img" '
                f'aria-labelledby="{self.name}-t {self.name}-d">\n'
                f'<title id="{self.name}-t">{_esc(self.title or self.name)}</title>\n'
                f'<desc id="{self.name}-d">{_esc(self.desc)}</desc>\n')
        return (head + "<defs>\n" + "\n".join(defs) + "\n</defs>\n" + "\n".join(self.body)
                + "\n</svg>\n")

    def save(self, svg_path=None, png_path=None, scale: float = 1.0, gh_preview: bool = True):
        """Write docs/fig/<name>.svg and the PNG preview(s) in PNG_DIR (not in the repo):
        <name>.png at `scale` and, with gh_preview, <name>_gh.png at GitHub's ~880 px width."""
        suffix = "-dark" if THEME == "dark" else ""      # TSR_FIG_THEME=dark: <name>-dark.svg
        svg_path = Path(svg_path or FIG_DIR / f"{self.name}{suffix}.svg")
        data = self.svg()
        svg_path.write_text(data, encoding="utf-8")
        png_path = Path(png_path or PNG_DIR / f"{self.name}{suffix}.png")
        png_path.parent.mkdir(parents=True, exist_ok=True)
        render_png(data, png_path, int(self.w * scale))
        msg = [str(svg_path), str(png_path)]
        if gh_preview:
            gh = png_path.with_name(png_path.stem + "_gh.png")
            render_png(data, gh, 880)
            msg.append(str(gh))
        print("wrote", *msg)
        return svg_path, png_path


# --------------------------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------------------------
def chamfer(x, y, w, h, c=CUT):
    """Octagon points of a w x h rect at (x, y) with corners cut by c."""
    c = min(c, w / 2, h / 2)
    return [(x + c, y), (x + w - c, y), (x + w, y + c), (x + w, y + h - c),
            (x + w - c, y + h), (x + c, y + h), (x, y + h - c), (x, y + c)]


def _unit(a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    n = (dx * dx + dy * dy) ** 0.5
    return dx / n, dy / n


def _back(tip, prev, d):
    ux, uy = _unit(prev, tip)
    return tip[0] - ux * d, tip[1] - uy * d


def _head(tip, prev, color):
    ux, uy = _unit(prev, tip)
    bx, by = tip[0] - ux * HEAD_L, tip[1] - uy * HEAD_L
    nx, ny = -uy * HEAD_W / 2, ux * HEAD_W / 2
    return (f'<polygon points="{_pts([tip, (bx + nx, by + ny), (bx - nx, by - ny)])}" '
            f'fill="{color}"/>')


def hollow_arrow(pts, w=ARROW_W, hw=ARROW_HW, hl=ARROW_HL, head=True, tail=False):
    """Outline polygon of a constant-width arrow along an axis-aligned polyline (miter corners)."""
    P = [tuple(map(float, p)) for p in pts]
    h = w / 2
    d = [_unit(P[i], P[i + 1]) for i in range(len(P) - 1)]
    n = [(-dy, dx) for dx, dy in d]                      # left normal (y down)
    s0 = (P[0][0] + d[0][0] * hl, P[0][1] + d[0][1] * hl) if tail else P[0]
    e = (P[-1][0] - d[-1][0] * hl, P[-1][1] - d[-1][1] * hl) if head else P[-1]
    spine = [s0, *P[1:-1], e]

    def side(sgn):
        out = []
        for i, p in enumerate(spine):
            if i == 0:
                nx, ny = n[0]
            elif i == len(spine) - 1:
                nx, ny = n[-1]
            else:
                nx, ny = n[i - 1][0] + n[i][0], n[i - 1][1] + n[i][1]
            out.append((p[0] + sgn * nx * h, p[1] + sgn * ny * h))
        return out

    left, right = side(1), side(-1)
    poly = list(left)
    if head:
        nx, ny = n[-1]
        poly += [(e[0] + nx * hw / 2, e[1] + ny * hw / 2), P[-1],
                 (e[0] - nx * hw / 2, e[1] - ny * hw / 2)]
    poly += right[::-1]
    if tail:
        nx, ny = n[0]
        poly += [(s0[0] - nx * hw / 2, s0[1] - ny * hw / 2), P[0],
                 (s0[0] + nx * hw / 2, s0[1] + ny * hw / 2)]
    return poly


def _gid(key: str, gname: str) -> str:
    return ("b-" if key == "bold" else "r-") + re.sub(r"[^A-Za-z0-9_-]", "_", gname)


# --------------------------------------------------------------------------------------------
# PNG preview
# --------------------------------------------------------------------------------------------
def render_png(svg_data: str, png_path, width: int):
    """Rasterise SVG text to a PNG of the given pixel width (cairosvg)."""
    import cairosvg
    cairosvg.svg2png(bytestring=svg_data.encode("utf-8"), write_to=str(png_path),
                     output_width=width)
