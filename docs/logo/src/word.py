"""Geometric monoline wordmark 'Tesseract' as filled outlines (no fonts).

Units: x-height X = 100, baseline y = 0, y down.  Every glyph is a union of
exact primitives (rectangles, circular ring sectors, and for the 's' the
offset of an ellipse-arc / common-tangent / ellipse-arc centreline), filled
with the nonzero rule.  All contours are emitted clockwise on screen so the
union is correct; counters are emitted counter-clockwise.
"""
import math

X = 100.0


def f(v):
    s = f"{v:.2f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


class Glyphs:
    def __init__(self, S=20.0, cap=140.0, asc_t=124.0, over=1.6, hbar=0.92,
                 term_e=38.0, term_c=44.0, s_wide=0.80, s_b=0.24, s_term=34.0):
        self.S = S               # stem / bowl stroke
        self.cap = cap           # cap height (T)
        self.asc_t = asc_t       # height of the t stem
        self.o = over            # overshoot of round letters
        self.hb = hbar * S       # horizontal bars slightly thinner (optical)
        self.term_e, self.term_c = term_e, term_c
        self.s_wide = s_wide     # s width relative to the o-bowl
        self.s_b, self.s_term = s_b, s_term
        self.ro = X / 2 + over   # bowl outer radius
        self.ri = self.ro - S    # bowl inner radius

    # ---------------------------------------------------------- primitives
    @staticmethod
    def rect(x0, y0, x1, y1):
        return f"M{f(x0)} {f(y0)}H{f(x1)}V{f(y1)}H{f(x0)}Z"

    @staticmethod
    def _p(cx, cy, r, a):
        t = math.radians(a)
        return cx + r * math.cos(t), cy + r * math.sin(t)

    def sector(self, cx, cy, ro, ri, a0, a1):
        """Ring sector, screen angles (clockwise from +x), a0 < a1, drawn
        clockwise: outer arc a0->a1, radial cut, inner arc back, radial cut."""
        span = a1 - a0
        large = 1 if span > 180 else 0
        x0, y0 = self._p(cx, cy, ro, a0); x1, y1 = self._p(cx, cy, ro, a1)
        x2, y2 = self._p(cx, cy, ri, a1); x3, y3 = self._p(cx, cy, ri, a0)
        return (f"M{f(x0)} {f(y0)}A{f(ro)} {f(ro)} 0 {large} 1 {f(x1)} {f(y1)}"
                f"L{f(x2)} {f(y2)}A{f(ri)} {f(ri)} 0 {large} 0 {f(x3)} {f(y3)}Z")

    def annulus(self, cx, cy, ro, ri):
        return (f"M{f(cx + ro)} {f(cy)}A{f(ro)} {f(ro)} 0 1 1 {f(cx - ro)} {f(cy)}"
                f"A{f(ro)} {f(ro)} 0 1 1 {f(cx + ro)} {f(cy)}Z"
                f"M{f(cx + ri)} {f(cy)}A{f(ri)} {f(ri)} 0 1 0 {f(cx - ri)} {f(cy)}"
                f"A{f(ri)} {f(ri)} 0 1 0 {f(cx + ri)} {f(cy)}Z")

    # -------------------------------------------------------------- glyphs
    # each returns (path_d, advance, lsb_kind, rsb_kind) with the ink box
    # starting at x = 0
    def T(self):
        W = 0.70 * self.cap
        S = self.S
        d = self.rect(0, -self.cap, W, -self.cap + self.hb) + self.rect(W / 2 - S / 2, -self.cap, W / 2 + S / 2, 0)
        return d, W

    def e(self):
        ro, ri, S, hb = self.ro, self.ri, self.S, self.hb
        cx, cy = ro, -X / 2
        ao = math.degrees(math.asin((hb / 2) / ro))      # bar's lower edge on the outer circle
        ai = math.degrees(math.asin((hb / 2) / ri))      # ... and on the inner circle
        t = self.term_e
        # bowl: everything except the aperture wedge between the bar and the terminal
        x0, y0 = self._p(cx, cy, ro, ao)
        x1, y1 = self._p(cx, cy, ro, t)
        x2, y2 = self._p(cx, cy, ri, t)
        x3, y3 = self._p(cx, cy, ri, ai)
        bowl = (f"M{f(x1)} {f(y1)}A{f(ro)} {f(ro)} 0 1 1 {f(x0)} {f(y0)}"     # outer, clockwise t -> ao
                f"L{f(x3)} {f(y3)}A{f(ri)} {f(ri)} 0 1 0 {f(x2)} {f(y2)}Z")  # bar edge, inner back
        bar = self.rect(cx - ri - S / 2, cy - hb / 2, cx + ri + S / 4, cy + hb / 2)
        return bowl + bar, 2 * ro

    def c(self):
        ro, ri = self.ro, self.ri
        cx, cy = ro, -X / 2
        t = self.term_c
        d = self.sector(cx, cy, ro, ri, t, 360 - t)
        right = cx + ro * math.cos(math.radians(t))
        return d, right

    def a(self):
        ro, ri, S = self.ro, self.ri, self.S
        cx, cy = ro, -X / 2
        d = self.annulus(cx, cy, ro, ri) + self.rect(cx + ri, -X, cx + ro, 0)
        return d, 2 * ro

    def r(self):
        ro, ri, S = self.ro, self.ri, self.S
        cx, cy = ro, -X / 2
        t = -58.0                                       # shoulder terminal angle
        d = self.rect(0, -X, S, 0) + self.sector(cx, cy, ro, ri, 180, 360 + t)
        right = cx + ro * math.cos(math.radians(t))
        return d, right

    def t(self):
        S, hb = self.S, self.hb
        arm_l, arm_r = 0.20 * X, 0.24 * X
        x0 = arm_l
        d = self.rect(x0, -self.asc_t, x0 + S, 0) + self.rect(0, -X, x0 + S + arm_r, -X + hb)
        return d, x0 + S + arm_r

    def B(self):
        """Capital B: stem and two right bowls (upper smaller), one shared middle bar."""
        S, hb, cap = self.S, self.hb, self.cap
        mid = -0.52 * cap
        hu = (mid + hb / 2) + cap        # upper bowl outer height
        hl = -(mid - hb / 2)             # lower bowl outer height
        ru, rl = hu / 2, hl / 2
        W = 0.72 * cap
        cxl = W - rl
        cxu = cxl - 3.0
        d = self.rect(0, -cap, S, 0)
        d += self.rect(S - 1, -cap, cxu + 0.5, -cap + hb) + self.rect(S - 1, mid - hb / 2, cxl + 0.5, mid + hb / 2)
        d += self.rect(S - 1, -hb, cxl + 0.5, 0)
        d += self.sector(cxu, -cap + ru, ru, ru - hb, -90, 90)
        d += self.sector(cxl, -rl, rl, rl - hb, -90, 90)
        return d, W

    def i(self):
        S = self.S
        return self.rect(0, -X, S, 0) + self.rect(0, -X - 14 - S, S, -X - 14), S

    def C(self):
        ro = self.cap / 2 + self.o
        ri = ro - self.S
        cx, cy = ro, -self.cap / 2
        t = 38.0
        d = self.sector(cx, cy, ro, ri, t, 360 - t)
        return d, cx + ro * math.cos(math.radians(t))

    def o_(self):
        ro, ri = self.ro, self.ri
        return self.annulus(ro, -X / 2, ro, ri), 2 * ro

    def s(self):
        """Centre-line: upper elliptical bowl (terminal at upper right, running
        anticlockwise over the top), the internal common tangent of the two
        ellipses as the spine, lower bowl (clockwise to the lower-left
        terminal).  The outline is the exact offset of that centre-line by
        +-S/2, sampled at 1 degree on the ellipse arcs."""
        S, o = self.S, self.o
        H = X + 2 * o - S                 # centre-line height (top bowl top .. bottom bowl bottom)
        a2 = (self.ro - S / 2) * self.s_wide * 1.0   # lower bowl semi-axes
        a1 = a2 * 0.94                    # upper bowl slightly smaller (optical)
        b2 = self.s_b * H
        b1 = b2 * 0.94
        cx = a2 + S / 2
        top = -X - o + S / 2
        bot = o - S / 2
        c1 = (cx + (a2 - a1) * 0.5, top + b1)
        c2 = (cx, bot - b2)

        def E(c, a, b, t):
            return (c[0] + a * math.cos(t), c[1] + b * math.sin(t))

        def N(a, b, t):                    # outward unit normal of ellipse at param t
            nx, ny = b * math.cos(t), a * math.sin(t)
            L = math.hypot(nx, ny)
            return nx / L, ny / L

        # internal common tangent: point t1 on the upper ellipse (left-lower
        # quadrant) and t2 on the lower one (right-upper quadrant) with equal
        # tangent direction and the connecting segment along that direction.
        def resid(t1):
            p1 = E(c1, a1, b1, t1)
            d1 = (-a1 * math.sin(t1), b1 * math.cos(t1))     # tangent (increasing t)
            # lower-ellipse point with antiparallel tangent (t2 = t1 + pi)
            t2 = t1 + math.pi
            p2 = E(c2, a2, b2, t2)
            v = (p2[0] - p1[0], p2[1] - p1[1])
            return d1[0] * v[1] - d1[1] * v[0], t2

        lo, hi = math.radians(95), math.radians(175)
        for _ in range(80):
            mid = (lo + hi) / 2
            if (resid(lo)[0] > 0) == (resid(mid)[0] > 0):
                lo = mid
            else:
                hi = mid
        t1 = (lo + hi) / 2
        t2 = t1 + math.pi
        # same tangent direction requires equal slopes on equal-shape ellipses;
        # with a1/b1 = a2/b2 the parametrisation keeps them parallel exactly.
        assert abs(a1 / b1 - a2 / b2) < 1e-9
        tu = math.radians(-self.s_term)            # upper terminal (upper right)
        tl = math.radians(180 - self.s_term)       # lower terminal (lower left)
        pts = []                                   # (point, normal) along the centre-line
        n = 140
        # upper bowl: from tu decreasing (anticlockwise on screen) to t1 - 2pi
        ta, tb = tu, t1 - 2 * math.pi
        for i in range(n + 1):
            t = ta + (tb - ta) * i / n
            pts.append((E(c1, a1, b1, t), N(a1, b1, t), +1))
        # lower bowl: from t2 increasing... (clockwise on screen) to tl + 2pi
        ta2, tb2 = t2 - 2 * math.pi, tl
        while tb2 < ta2:
            tb2 += 2 * math.pi
        for i in range(n + 1):
            t = ta2 + (tb2 - ta2) * i / n
            pts.append((E(c2, a2, b2, t), N(a2, b2, t), -1))
        left, right = [], []
        for (x, y), (nx, ny), sgn in pts:
            # offset to both sides; outward normal of the upper bowl is
            # on the 'outer' side, for the lower bowl as well
            left.append((x + nx * S / 2, y + ny * S / 2))
            right.append((x - nx * S / 2, y - ny * S / 2))
        # orient: outer side of the upper bowl continues smoothly into the inner
        # side of the lower bowl across the spine (tangent continuity)
        upper_out = left[:n + 1]; upper_in = right[:n + 1]
        lower_out = left[n + 1:]; lower_in = right[n + 1:]
        side1 = upper_out + lower_in
        side2 = upper_in + lower_out
        poly = side1 + side2[::-1]
        if _area(poly) < 0:
            poly = poly[::-1]
        d = "M" + "L".join(f"{f(x)} {f(y)}" for x, y in poly) + "Z"
        xs = [p[0] for p in poly]
        shift = -min(xs)
        d = "M" + "L".join(f"{f(x + shift)} {f(y)}" for x, y in poly) + "Z"
        return d, max(xs) - min(xs)


def _area(poly):
    return sum(poly[i][0] * poly[(i + 1) % len(poly)][1] - poly[(i + 1) % len(poly)][0] * poly[i][1]
               for i in range(len(poly))) / 2


# side kinds for spacing: 'v' straight vertical, 'o' round, 'x' open/terminal, 'b' bar
SIDES = {"T": ("b", "b"), "e": ("o", "x"), "s": ("x", "x"), "r": ("v", "x"),
         "a": ("o", "v"), "c": ("o", "x"), "t": ("b", "b"),
         "B": ("v", "o"), "i": ("v", "v"), "C": ("o", "x"), "o": ("o", "o")}
SB = {"v": 9.0, "o": 5.5, "x": 4.0, "b": 1.0}
KERN = {("T", "e"): -17.0, ("r", "a"): -6.0, ("c", "t"): -2.0, ("e", "r"): 1.0}


def wordmark(text="Tesseract", track=6.0, **kw):
    """Return (path_d in x-height units with baseline at y=0, total width)."""
    G = Glyphs(**kw)
    parts = []
    x = 0.0
    prev = None
    for ch in text:
        if ch == " ":                       # word space
            x += 62.0
            prev = None
            continue
        d, w = getattr(G, "o_" if ch == "o" else ch)()
        l, r = SIDES[ch]
        if prev is not None:
            x += SB[SIDES[prev][1]] + SB[l] + track + KERN.get((prev, ch), 0.0)
        parts.append((x, d))
        x += w
        prev = ch
    return parts, x


def translate_d(d, dx, dy=0.0, k=1.0):
    """Apply x' = k*(x+dx), y' = k*(y+dy) to an absolute path made of
    M/L/H/V/A/Z commands (as produced above)."""
    import re
    toks = re.findall(r"[MLHVAZ]|-?\d+(?:\.\d+)?", d)
    out = []
    i = 0
    cmd = None
    while i < len(toks):
        t = toks[i]
        if t in "MLHVAZ":
            cmd = t; out.append(t); i += 1
            continue
        if cmd in "ML":
            out.append(f"{f(k * (float(toks[i]) + dx))} {f(k * (float(toks[i + 1]) + dy))}"); i += 2
        elif cmd == "H":
            out.append(f(k * (float(t) + dx))); i += 1
        elif cmd == "V":
            out.append(f(k * (float(t) + dy))); i += 1
        elif cmd == "A":
            rx, ry, rot, la, sw, x, y = toks[i:i + 7]
            out.append(f"{f(k * float(rx))} {f(k * float(ry))} {rot} {la} {sw} "
                       f"{f(k * (float(x) + dx))} {f(k * (float(y) + dy))}"); i += 7
        if i < len(toks) and toks[i] not in "MLHVAZ":
            out.append(" ")
    s = "".join(out)
    return s.replace(" M", "M").replace(" L", "L").replace(" Z", "Z")
