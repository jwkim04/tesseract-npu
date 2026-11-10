"""Render PNG deliverables + preview sheet + verification images for final/."""
import io, math, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from PIL import Image, ImageDraw, ImageFont
from rend import render, on, zoom, LIGHT, DARK
from build import COL, WORD_LIGHT, WORD_DARK, MONO, BG_DARK

D = os.path.dirname(HERE)
P = lambda n: os.path.join(D, n)
W_ = os.path.join(HERE, "check")
os.makedirs(W_, exist_ok=True)
F = lambda s: ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", s)
FB = lambda s: ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", s)

def vb(svg):
    m = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', open(svg).read())
    return float(m.group(1)), float(m.group(2))

MARK, FAV, FAV16 = P("tesseract-mark.svg"), P("tesseract-favicon.svg"), P("tesseract-favicon-16.svg")
LK, LKD = P("tesseract-lockup.svg"), P("tesseract-lockup-dark.svg")

# ------------------------------------------------------------ deliverables
render(MARK, 512).save(P("mark-512.png"))
fav32 = render(FAV, 32); fav32.save(P("favicon-32.png"))
fav16 = render(FAV16, 16); fav16.save(P("favicon-16.png"))
m48 = render(os.path.join(HERE, "favicon-48.svg"), 48)
m48.save(P("favicon.ico"), sizes=[(16, 16), (32, 32), (48, 48)], append_images=[fav16, fav32])
# apple-touch-icon: solid background (iOS ignores transparency)
ati = Image.new("RGBA", (180, 180), BG_DARK)
ati.alpha_composite(render(MARK, 144), (18, 18)); ati.convert("RGB").save(P("apple-touch-icon.png"))
# social preview 1280x640, lockup centered on a solid background
lw, lh = vb(LKD)
sw = 920; sh = round(lh * sw / lw)
soc = Image.new("RGBA", (1280, 640), BG_DARK)
soc.alpha_composite(render(LKD, sw, sh), ((1280 - sw) // 2, (640 - sh) // 2))
soc.convert("RGB").save(P("social-preview.png"))

# ----------------------------------------------------------- verification
ico = Image.open(P("favicon.ico"))
print("ico sizes:", sorted(ico.info.get("sizes", [])))
ico.size = (16, 16); ico16 = ico.convert("RGBA")
assert ico16.tobytes() == fav16.tobytes(), "ico 16 frame is not the hinted one"
ico = Image.open(P("favicon.ico")); ico.size = (32, 32)
assert ico.convert("RGBA").tobytes() == fav32.tobytes(), "ico 32 frame is not the hinted one"

# webpack side-by-side at 32 / 128 (reference drawn by the engineer judge)
REF = os.path.join(os.path.dirname(D), "judge-engineer", "ref_wp.svg")
cells = []
if os.path.exists(REF):
    row = Image.new("RGB", (4 * 140 + 5 * 12, 2 * 140 + 3 * 12), "#c8c8c8")
    for j, bg in enumerate((LIGHT, DARK)):
        for i, (svg, px) in enumerate(((MARK, 128), (REF, 128), (FAV, 32), (REF, 32))):
            im = on(render(svg, px), bg)
            if px == 32:
                im = zoom(im, 4)
            row.paste(im, (12 + i * 152, 12 + j * 152))
    row.save(os.path.join(W_, "vs_webpack.png"))

# ------------------------------------------------------------------- sheet
SW = 1600
sheet = Image.new("RGB", (SW, 2140), "#d6d6dc")
dr = ImageDraw.Draw(sheet)
def label(x, y, s, size=15, fill="#1d1d24"):
    dr.text((x, y), s, font=F(size), fill=fill)
def panel(x, y, w, h, bg):
    dr.rectangle([x, y, x + w - 1, y + h - 1], fill=bg)

dr.text((40, 26), "Tesseract — logo system", font=FB(26), fill="#1d1d24")
label(40, 62, "4-cube (±1)^4 · perspective along w (eye at w=4, inner cell = 3/5) · isometric · "
      "violet = the 6 faces joining the cells · amber = inner cell from inside, floor = 4x4 weight tile")
# row 1: mark 512 light / dark (shown at 440)
y0 = 100
for i, bg in enumerate((LIGHT, DARK)):
    x = 40 + i * 500
    panel(x, y0, 480, 480, bg)
    sheet.paste(on(render(MARK, 440), bg), (x + 20, y0 + 20))
label(40, y0 + 486, "tesseract-mark.svg  (one file for both themes: every edge is a transparent cut)")
# right column: size ladder light / dark (all 1:1)
x0 = 1040
for j, bg in enumerate((LIGHT, DARK)):
    yy = y0 + j * 245
    panel(x0, yy, 520, 230, bg)
    xx = x0 + 30
    for px in (128, 64, 48, 32, 16):
        src = FAV if px == 32 else FAV16 if px == 16 else MARK
        im = on(render(src, px), bg)
        sheet.paste(im, (xx, yy + 115 - px // 2)); xx += px + 46
label(x0, y0 + 486, "1:1 — 128 · 64 · 48 (mark) · 32 (favicon) · 16 (favicon-16)", 14)

# row 2: lockups light/dark large
y1 = 620
lw, lh = vb(LK)
for i, (svg, bg) in enumerate(((LK, LIGHT), (LKD, DARK))):
    x = 40 + i * 770
    panel(x, y1, 750, 240, bg)
    w = 700; h = round(lh * w / lw)
    sheet.paste(on(render(svg, w, h), bg), (x + 25, y1 + 120 - h // 2))
label(40, y1 + 246, "tesseract-lockup.svg (light)  ·  tesseract-lockup-dark.svg (dark) — letters are geometric outlines, no fonts")
# row 3: small lockups at real README sizes + mono
y2 = 910
for i, (svg, bg) in enumerate(((LK, LIGHT), (LKD, DARK))):
    for k, hpx in enumerate((48, 32)):
        x = 40 + i * 380; yy = y2 + k * 70
        w = round(lw * hpx / lh)
        panel(x, yy, 360, 62, bg)
        sheet.paste(on(render(svg, w, hpx), bg), (x + 10, yy + 31 - hpx // 2))
label(40, y2 + 140, "lockup at 48 px and 32 px tall")
xm = 820
for i, (name, colour) in enumerate(MONO.items()):
    for j, bg in enumerate((LIGHT, DARK)):
        if (name == "white" and bg == LIGHT) or (name == "black" and bg == DARK):
            bgc = "#8a8a96"
        else:
            bgc = bg
        x = xm + i * 250 + j * 120; yy = y2
        panel(x, yy, 112, 112, bgc)
        sheet.paste(on(render(P(f"tesseract-mark-mono-{name}.svg"), 96), bgc), (x + 8, yy + 8))
label(xm, y2 + 118, "tesseract-mark-mono-{violet,black,white}.svg — one color (etch, silkscreen, stickers)")
# row 4: social preview + apple touch
y3 = 1080
soc_t = Image.open(P("social-preview.png")).resize((800, 400), Image.LANCZOS)
sheet.paste(soc_t, (40, y3 + 10))
label(40, y3 + 416, "social-preview.png (1280×640)")
sheet.paste(Image.open(P("apple-touch-icon.png")), (870, y3 + 10))
label(870, y3 + 196, "apple-touch-icon.png (180)")
# palette with contrast
def lum(h):
    h = h.lstrip('#'); r, g, b = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    f = lambda c: c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)
def cr(a, b):
    la, lb = sorted([lum(a), lum(b)], reverse=True); return (la + 0.05) / (lb + 0.05)
pal0 = [("violet top", COL["top"]), ("violet left", COL["left"]), ("violet right", COL["right"]),
       ("amber floor", COL["floor"]), ("amber r-wall", COL["rwall"]), ("amber l-wall", COL["lwall"]),
       ("word (light)", WORD_LIGHT), ("word (dark)", WORD_DARK)]
yp = y3 + 46
dr.text((1075, yp - 36), "palette — contrast vs #ffffff / #0d1117", font=FB(15), fill="#1d1d24")
rows = []
for k, (n, c) in enumerate(pal0):
    yy = yp + k * 34
    dr.rectangle([1075, yy, 1105, yy + 28], fill=c)
    s = f"{n:<13} {c}  {cr(c, '#ffffff'):.2f} / {cr(c, '#0d1117'):.2f}"
    dr.text((1115, yy + 5), s, font=F(14), fill="#1d1d24")
    rows.append((n, c, cr(c, '#ffffff'), cr(c, '#0d1117')))
# row 5: x8 zooms of the favicons on both backgrounds
y4 = 1530
k = 0
for src, px, z in ((FAV, 32, 8), (FAV16, 16, 16)):
    for bg in (LIGHT, DARK):
        im = zoom(on(render(src, px), bg), z)
        sheet.paste(im, (40 + k * 275, y4)); k += 1
label(40, y4 + 262, "favicon-32.png (×8) · favicon-16.png (×16) — vertical edges and w-edges on whole pixels")
sheet = sheet.crop((0, 0, SW, y4 + 300))
sheet.save(P("sheet.png"))
for r in rows:
    print(f"{r[0]:<13} {r[1]}  white {r[2]:.2f}  dark {r[3]:.2f}")
