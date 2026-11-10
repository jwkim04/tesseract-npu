"""Build every Tesseract logo asset into ../ (docs/logo/).

Run with the svg venv python (cairosvg + Pillow):
  python docs/logo/src/build.py && python docs/logo/src/render_final.py   (needs cairosvg + Pillow)

Geometry : tess.py  (4-cube (+-1)^4, perspective from w = D, isometric)
Favicon16: pix.py   (same construction in 2:1 pixel isometric)
Wordmark : word.py  (filled geometric outlines, no fonts)
Lockup   : lock.py
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tess import Frame, verify, mark_body, mono_body, mark_shapes, svg_doc, C30, inset, HEX
from pix import PixFrame, raster, grid_svg_body
from lock import lockup

DST = os.path.dirname(HERE)

# ------------------------------------------------------------------ palette
COL = {
    # outer (near) cell: the six connecting faces, lit from the upper left
    "top": "#8B7CF6", "left": "#6D5AE6", "right": "#5848DC",
    # inner (far) cell seen from inside: floor = weight tile, two walls
    "floor": "#FFC845", "rwall": "#F59E0B", "lwall": "#D9730B",
}
WORD_LIGHT = "#5848DC"
WORD_DARK = "#A497FF"
MONO = {"violet": "#6D5AE6", "black": "#000000", "white": "#FFFFFF"}
BG_DARK = "#0d1117"

# --------------------------------------------------------------- parameters
D_EYE = 4.0                          # eye at w = 4 -> inner cell = 3/5 of outer
R512 = 192 / C30                     # vertical hex sides at x = 64 / 448 (whole px at 16..512)
G512 = 20.0                          # wide cuts: tesseract edges
g512 = 5.0                           # narrow cuts: tensor cell boundaries (4x4 tile)

DESC = ("Tesseract logo mark: a 4-cube projected cube-in-cube (perspective along w, "
        "isometric view). Violet: the six faces joining the outer and inner cell; "
        "amber: the inner cell seen from inside, its floor a 4x4 weight tile.")


def write(name, text):
    p = os.path.join(DST, name)
    with open(p, "w") as fh:
        fh.write(text)
    return p


def gap_widths(fr, G, g):
    """Measure every wide cut perpendicular to its direction (fix: diagonal
    cuts must be exactly as wide as vertical ones)."""
    sh = mark_shapes(fr, G=G, g=g, floor_n=1)
    widths = []
    P = fr.named()
    c = (fr.cx, fr.cy)
    polys = [p for k in ("top", "left", "right") for p in sh[k]]
    for h in HEX:
        v = P["o" + h]
        dx, dy = v[0] - c[0], v[1] - c[1]
        L = math.hypot(dx, dy)
        nx, ny = -dy / L, dx / L
        mid = (c[0] + dx * 0.8, c[1] + dy * 0.8)   # a point on the spoke inside the ring
        # distance from the spoke line to the nearest polygon edge on each side
        ds = []
        for poly in polys:
            for i in range(len(poly)):
                p, q = poly[i], poly[(i + 1) % len(poly)]
                ex, ey = q[0] - p[0], q[1] - p[1]
                el = math.hypot(ex, ey)
                if abs(ex * dy - ey * dx) / (el * L) > 1e-6:
                    continue                            # not parallel to the spoke
                dist = ((p[0] - mid[0]) * nx + (p[1] - mid[1]) * ny)
                ds.append(dist)
        pos = min(d for d in ds if d > 1e-9); neg = max(d for d in ds if d < -1e-9)
        widths.append(pos - neg)
    return widths


def main():
    fr = Frame(256, 256, R512, D_EYE)
    kinds = verify(fr)
    widths = gap_widths(fr, G512, g512)
    assert all(abs(w - G512) < 1e-9 for w in widths), widths

    # ---------------------------------------------------------- the mark
    write("tesseract-mark.svg", svg_doc("0 0 512 512", 512, 512,
          mark_body(fr, G=G512, g=g512, colours=COL, floor_n=4), desc=DESC))

    # one-color marks: every edge is negative space (incl. the ring/core rim)
    for name, colour in MONO.items():
        write(f"tesseract-mark-mono-{name}.svg", svg_doc("0 0 512 512", 512, 512,
              mono_body(fr, G=G512, g=g512, colour=colour, floor_n=4),
              desc="Tesseract mark, single color; edges are negative space."))

    # ----------------------------------------------------- favicon (32 px)
    # pixel-hinted: hex 27 px wide centered on column 15 (x 2..29), inner cell
    # 17 px (x 7..24, eye at w = 4.4 so the inner sides land on whole pixels),
    # 1 px cuts with the vertical w-edges exactly on pixel column 15.
    f32 = Frame(15.5, 16, 13.5 / C30, 4.4)
    assert abs(f32.named()["iUR"][0] - 24.0) < 1e-9 and abs(f32.named()["oUR"][0] - 29.0) < 1e-9
    write("tesseract-favicon.svg", svg_doc("0 0 32 32", 32, 32,
          mark_body(f32, G=1.0, g=0, colours=COL, floor_n=0, overlap=0.06),
          desc="Tesseract favicon, hinted for a 32 px grid."))

    # 48 px frame for favicon.ico (same hinting rule: hex x 3..44 centered on
    # column 23, inner sides on x = 11 / 36 -> eye at w = (1+r)/(1-r), r = 12.5/20.5)
    r48 = 12.5 / 20.5
    f48 = Frame(23.5, 24, 20.5 / C30, (1 + r48) / (1 - r48))
    assert abs(f48.named()["iUR"][0] - 36.0) < 1e-9
    with open(os.path.join(HERE, "favicon-48.svg"), "w") as fh:
        fh.write(svg_doc("0 0 48 48", 48, 48,
                 mark_body(f48, G=1.0, g=0, colours=COL, floor_n=0, overlap=0.06)))

    # ----------------------------------------------------- favicon (16 px)
    # 2:1 pixel isometric, point sampled: keeps the hexagon, the three amber
    # faces and the vertical w-edges (the top one runs on to the far corner).
    f16 = PixFrame(8.5, 8.0, 6.5, D_EYE, vert=4.5)
    grid16 = raster(f16, 16, 2 / math.sqrt(5), spokes=("T", "B"), core_cut="vert")
    write("tesseract-favicon-16.svg", svg_doc("0 0 16 16", 16, 16,
          grid_svg_body(grid16, COL), desc="Tesseract favicon, pixel-drawn for 16 px."))

    # -------------------------------------------------------------- lockups
    lk = dict(R=200.0, ratio=2.15, S=21.0, track=7.0, gap_k=0.56, d=D_EYE)
    sl, W, H = lockup(COL, WORD_LIGHT, **lk)
    sd, _, _ = lockup(COL, WORD_DARK, **lk)
    write("tesseract-lockup.svg", sl)
    write("tesseract-lockup-dark.svg", sd)

    print("edges on hexagons / radii / collapsed:", kinds)
    print("wide-cut widths (perpendicular):", [round(w, 6) for w in widths])
    print("lockup viewBox:", W, H)


if __name__ == "__main__":
    main()
