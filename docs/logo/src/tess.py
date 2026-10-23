"""Tesseract mark geometry (shared by all generators).

4-cube vertices (+-1)^4 -> 3D by perspective along w (eye on the w axis at
w = D) -> 2D by true isometric projection (view along (1,1,1), z up, screen
y down).  With D = 3 the near cell (w=+1) has scale 1/2 and the far cell
(w=-1) scale 1/4: the classic cube-in-cube with inner/outer ratio exactly 1/2.
In isometric view both cells are concentric regular hexagons and the 8
w-edges become 6 radial spokes (+2 that collapse onto the centre).

Every number in the published SVGs comes from here; nothing is placed by eye.
"""
import itertools
import math

D = 3.0
C30 = math.cos(math.pi / 6)

VERTS4 = list(itertools.product((-1, 1), repeat=4))
EDGES4 = [(a, b) for a, b in itertools.combinations(VERTS4, 2)
          if sum(abs(p - q) for p, q in zip(a, b)) == 2]


def proj4(v, d=D):
    x, y, z, w = v
    k = 1.0 / (d - w)
    return (x * k, y * k, z * k)


def iso(p):
    x, y, z = p
    return ((x - y) * C30, (x + y) * 0.5 - z)


def w_scale(w, d=D):
    """Projected size of the w-slice relative to the outer (w=+1) cell."""
    return (d - 1.0) / (d - w)


HEX = ["T", "UR", "LR", "B", "LL", "UL"]          # clockwise from the top
CUBE = {"C": (1, 1, 1), "T": (-1, -1, 1), "UR": (1, -1, 1), "UL": (-1, 1, 1),
        "B": (1, 1, -1), "LR": (1, -1, -1), "LL": (-1, 1, -1)}


class Frame:
    """iso units -> drawing units; R = outer hexagon circumradius."""

    def __init__(self, cx, cy, R, d=D):
        self.cx, self.cy, self.R, self.d = cx, cy, R, d
        self.s = R / (2.0 / (d - 1.0))   # outer cell half-size 1/(d-1) -> radius 2/(d-1)

    def pt(self, v4):
        u, v = iso(proj4(v4, self.d))
        return (self.cx + u * self.s, self.cy + v * self.s)

    def named(self):
        P = {}
        for pre, w in (("o", 1), ("i", -1)):
            for k, (x, y, z) in CUBE.items():
                P[pre + k] = self.pt((x, y, z, w))
        return P

    def ring(self, w):
        """Hexagon of the w-slice (used for the w-perspective bands)."""
        k = w_scale(w, self.d)
        return [(self.cx + (self.pt((*CUBE[h], 1))[0] - self.cx) * k,
                 self.cy + (self.pt((*CUBE[h], 1))[1] - self.cy) * k) for h in HEX]


def verify(fr):
    """Numerical checks that the drawing really is the tesseract projection."""
    assert len(VERTS4) == 16 and len(EDGES4) == 32
    P = fr.named()
    R = fr.R
    oR = math.dist(P["oC"], P["oT"]); iR = math.dist(P["iC"], P["iT"])
    ratio = (fr.d - 1) / (fr.d + 1)
    assert abs(oR - R) < 1e-9 and abs(iR / oR - ratio) < 1e-12
    for pre in "oi":
        for k in range(6):
            a, b = P[pre + HEX[k]], P[pre + HEX[(k + 1) % 6]]
            assert abs(math.dist(a, b) - (R if pre == "o" else R * ratio)) < 1e-9
    # every one of the 32 edges lies on a hexagon side or on one of the 6 radii
    c = (fr.cx, fr.cy)
    segs = []
    for pre in "oi":
        for k in range(6):
            segs.append((P[pre + HEX[k]], P[pre + HEX[(k + 1) % 6]]))
    for h in HEX:
        segs.append((c, P["o" + h]))

    def on_seg(p, s):
        (x1, y1), (x2, y2) = s
        cr = (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1)
        L = math.dist(s[0], s[1])
        t = ((p[0] - x1) * (x2 - x1) + (p[1] - y1) * (y2 - y1)) / L ** 2
        return abs(cr) / L < 1e-6 and -1e-9 <= t <= 1 + 1e-9

    kinds = {"hex": 0, "radial": 0, "point": 0}
    for a, b in EDGES4:
        pa, pb = fr.pt(a), fr.pt(b)
        if math.dist(pa, pb) < 1e-9:
            kinds["point"] += 1
            continue
        hit = [i for i, s in enumerate(segs) if on_seg(pa, s) and on_seg(pb, s)]
        assert hit, (a, b)
        kinds["hex" if hit[0] < 12 else "radial"] += 1
    assert sum(kinds.values()) == 32
    return kinds


# ------------------------------------------------------------------ polygons
def area(poly):
    return sum(poly[i][0] * poly[(i + 1) % len(poly)][1] - poly[(i + 1) % len(poly)][0] * poly[i][1]
               for i in range(len(poly))) / 2


def _isect(p1, p2, p3, p4):
    x1, y1 = p1; x2, y2 = p2; x3, y3 = p3; x4, y4 = p4
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
    return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))


def inset(poly, d):
    """Move edge i (poly[i] -> poly[i+1]) of a convex polygon inward by d[i]
    (negative = outward) and return the new polygon.  Distances are
    perpendicular, so a cut built from two insets of g/2 is exactly g wide."""
    n = len(poly)
    sgn = 1 if area(poly) > 0 else -1
    lines = []
    for i in range(n):
        p, q = poly[i], poly[(i + 1) % n]
        dx, dy = q[0] - p[0], q[1] - p[1]
        L = math.hypot(dx, dy)
        nx, ny = -dy / L * sgn, dx / L * sgn          # inward normal
        lines.append(((p[0] + nx * d[i], p[1] + ny * d[i]), (q[0] + nx * d[i], q[1] + ny * d[i])))
    return [_isect(*lines[i - 1], *lines[i]) for i in range(n)]


def lerp(a, b, t):
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def fmt(v, nd=2):
    s = f"{v:.{nd}f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def path(poly, nd=2):
    return "M" + " ".join(f"{fmt(x, nd)} {fmt(y, nd)}" for x, y in poly) + "Z"


def grid_cells(rh, n, cut, edge_inset):
    """Split parallelogram rh=[C, A, X, B] into n x n cells separated by
    `cut`-wide gaps.  edge_inset[i] insets side i of rh first (C->A, A->X,
    X->B, B->C) so the visible face is divided into equal cells."""
    face = inset(rh, edge_inset)
    C, A, X, B = face
    ea = (A[0] - C[0], A[1] - C[1]); eb = (B[0] - C[0], B[1] - C[1])
    p = lambda s, t: (C[0] + ea[0] * s + eb[0] * t, C[1] + ea[1] * s + eb[1] * t)
    out = []
    for i in range(n):
        for j in range(n):
            s0, s1, t0, t1 = i / n, (i + 1) / n, j / n, (j + 1) / n
            q = [p(s0, t0), p(s1, t0), p(s1, t1), p(s0, t1)]
            h = cut / 2
            out.append(inset(q, [0 if j == 0 else h, 0 if i == n - 1 else h,
                                 0 if j == n - 1 else h, 0 if i == 0 else h]))
    return out


# --------------------------------------------------------------------- mark
VIO = {"top": "#8B7CF6", "left": "#6D5AE6", "right": "#5848DC"}
AMB = {"floor": "#FFD166", "rwall": "#F7A928", "lwall": "#E07B12"}
# which outer-cell face each ring trapezoid (outer edge HEX[k]->HEX[k+1]) belongs to
FACE_OF = {("T", "UR"): "top", ("UR", "LR"): "right", ("LR", "B"): "right",
           ("B", "LL"): "left", ("LL", "UL"): "left", ("UL", "T"): "top"}


def mark_shapes(fr, G, g, floor_n=4, bands=(), rim_gap=False, overlap=1.0, spokes=None, core_cut=True):
    """Return [(colour_key, [polys])] for the colour mark.

    G : width of the wide cuts (tesseract edges: 6 w-spokes + the far cell's
        back corner, the inverted Y)
    g : width of the narrow cuts (tensor cell boundaries, w-bands)
    floor_n : n x n weight tile on the cavity floor (0/1 = solid)
    bands : w values at which each ring trapezoid is cut (w-perspective)
    rim_gap : True -> wide gap between ring and core (mono construction)
    """
    P = fr.named()
    out = {k: [] for k in ("top", "left", "right", "floor", "rwall", "lwall")}
    rim = G / 2 if rim_gap else -overlap
    sp = set(HEX if spokes is None else spokes)
    for k in range(6):
        a, b = HEX[k], HEX[(k + 1) % 6]
        face = FACE_OF[(a, b)]
        oa, ob, ia, ib = P["o" + a], P["o" + b], P["i" + a], P["i" + b]
        ga = G / 2 if a in sp else 0.0
        gb = G / 2 if b in sp else 0.0
        if not bands:
            out[face].append(inset([oa, ob, ib, ia], [0, gb, rim, ga]))
            continue
        # w-bands: slice the trapezoid at the projected w-slices
        ws = sorted(bands, reverse=True)
        cuts = [1.0] + list(ws) + [-1.0]
        for j in range(len(cuts) - 1):
            k0, k1 = w_scale(cuts[j], fr.d), w_scale(cuts[j + 1], fr.d)
            c = (fr.cx, fr.cy)
            pa0 = lerp(c, oa, k0); pb0 = lerp(c, ob, k0)
            pa1 = lerp(c, oa, k1); pb1 = lerp(c, ob, k1)
            d_out = 0 if j == 0 else g / 2
            d_in = rim if j == len(cuts) - 2 else g / 2
            out[face].append(inset([pa0, pb0, pb1, pa1], [d_out, gb, d_in, ga]))
    iC, iT, iUR, iLR, iB, iLL, iUL = (P["i" + k] for k in ("C", "T", "UR", "LR", "B", "LL", "UL"))
    rimd = G / 2 if rim_gap else 0
    floor = [iC, iLR, iB, iLL]
    lwall = [iC, iLL, iUL, iT]
    rwall = [iC, iT, iUR, iLR]
    gc = G / 2 if core_cut else 0.0
    gv = G / 2 if core_cut in (True, "vert") else 0.0   # vertical arm of the inverted Y
    if core_cut == "vert":
        gc = 0.0
    ins = [gc, rimd, rimd, gc]            # floor: both arms are diagonals
    ins_l = [gc, rimd, rimd, gv]          # left wall: C->LL diagonal, T->C vertical
    ins_r = [gv, rimd, rimd, gc]          # right wall: C->T vertical, LR->C diagonal
    if floor_n and floor_n > 1:
        out["floor"] += grid_cells(floor, floor_n, g, ins)
    else:
        out["floor"].append(inset(floor, ins))
    out["lwall"].append(inset(lwall, ins_l))
    out["rwall"].append(inset(rwall, ins_r))
    return out


COLOURS = dict(VIO, **AMB)
ORDER = ("top", "left", "right", "floor", "rwall", "lwall")


def mark_body(fr, G, g, colours=None, indent="  ", **kw):
    colours = colours or COLOURS
    sh = mark_shapes(fr, G, g, **kw)
    lines = []
    for key in ORDER:
        if sh[key]:
            lines.append(f'{indent}<path fill="{colours[key]}" d="{" ".join(path(p) for p in sh[key])}"/>')
    return "\n".join(lines)


def mono_body(fr, G, g, colour, indent="  ", **kw):
    kw.setdefault("rim_gap", True)
    sh = mark_shapes(fr, G, g, **kw)
    polys = [p for key in ORDER for p in sh[key]]
    return f'{indent}<path fill="{colour}" d="{" ".join(path(p) for p in polys)}"/>'


def svg_doc(viewbox, w, h, body, title="Tesseract", desc=None):
    d = f'\n  <desc id="d">{desc}</desc>' if desc else ""
    lab = "t d" if desc else "t"
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{viewbox}" width="{w}" height="{h}" '
            f'role="img" aria-labelledby="{lab}">\n  <title id="t">{title}</title>{d}\n{body}\n</svg>\n')
