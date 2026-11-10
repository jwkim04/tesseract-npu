"""Pixel-isometric (2:1 dimetric) rasterizer for the 16 px favicon.

Same tesseract construction as tess.py, but the 3D->2D step is the 2:1
dimetric projection used in pixel art (edge slopes exactly 1/2), so every
slanted edge and cut lands on a clean 2-px staircase.  Pixels are point
sampled at their centers; the result is written as one-pixel-high runs.
"""
import math
from tess import CUBE, HEX, w_scale, mark_shapes, ORDER


class PixFrame:
    def __init__(self, cx, cy, W, d, vert=None):
        # outer hexagon half-width W; 2:1 slopes; vertical side 2*S
        self.cx, self.cy, self.d = cx, cy, d
        self.W = W
        self.S = vert if vert is not None else W / 2.0
        self.R = W

    def _pt(self, key, k):
        x, y, z = CUBE[key]
        u = (x - y) / 2.0 * self.W            # +-W at UR/LR, 0 at T/B/C
        v = (x + y) / 2.0 * (self.W / 2.0) - z * self.S   # true 2:1 if S = W/2
        if key == "C":
            u, v = 0.0, 0.0
        return (self.cx + u * k, self.cy + v * k)

    def named(self):
        P = {}
        r = w_scale(-1, self.d)
        for pre, k in (("o", 1.0), ("i", r)):
            for key in CUBE:
                P[pre + key] = self._pt(key, k)
        return P


def inside(poly, x, y):
    c = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]; x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xi = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < xi:
                c = not c
    return c


def raster(fr, px, G, eps=(1e-3, 2e-3), keys=ORDER, **kw):
    sh = mark_shapes(fr, G=G, g=0, floor_n=0, overlap=0.0, **kw)
    grid = [[None] * px for _ in range(px)]
    for yy in range(px):
        for xx in range(px):
            x, y = xx + 0.5 + eps[0], yy + 0.5 + eps[1]
            for k in keys:
                if any(inside(p, x, y) for p in sh[k]):
                    grid[yy][xx] = k
    return grid


def grid_svg_body(grid, colours, indent="  "):
    runs = {}
    for y, row in enumerate(grid):
        x = 0
        while x < len(row):
            k = row[x]
            if k is None:
                x += 1; continue
            x0 = x
            while x < len(row) and row[x] == k:
                x += 1
            runs.setdefault(k, []).append(f"M{x0} {y}h{x - x0}v1h{x0 - x}Z")
    return "\n".join(f'{indent}<path fill="{colours[k]}" d="{"".join(runs[k])}"/>'
                     for k in ORDER if k in runs)


def ascii(grid):
    ch = {None: ".", "top": "T", "left": "L", "right": "R", "floor": "f", "rwall": "r", "lwall": "l"}
    return "\n".join("".join(ch[c] for c in row) for row in grid)
