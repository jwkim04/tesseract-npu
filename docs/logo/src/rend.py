"""Render helpers (run with the svgvenv python: cairosvg + Pillow)."""
import io
import cairosvg
from PIL import Image, ImageDraw, ImageFont

LIGHT, DARK = "#ffffff", "#0d1117"
try:
    FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
except Exception:  # pragma: no cover
    FONT = ImageFont.load_default()


def render(svg, w, h=None, bg=None):
    """svg: path or markup string -> RGBA PIL image (transparent unless bg)."""
    h = h or w
    kw = dict(output_width=w, output_height=h)
    if bg:
        kw["background_color"] = bg
    if svg.lstrip().startswith("<"):
        png = cairosvg.svg2png(bytestring=svg.encode(), **kw)
    else:
        png = cairosvg.svg2png(url=svg, **kw)
    return Image.open(io.BytesIO(png)).convert("RGBA")


def on(im, bg):
    base = Image.new("RGBA", im.size, bg)
    base.alpha_composite(im)
    return base.convert("RGB")


def zoom(im, k):
    return im.resize((im.size[0] * k, im.size[1] * k), Image.NEAREST)


def grid(cells, cols, pad=12, bg="#c8c8c8", label_h=18):
    """cells: list of (label, PIL image). Packs into rows of `cols`."""
    rows = [cells[i:i + cols] for i in range(0, len(cells), cols)]
    colw = [max(r[c][1].size[0] for r in rows if c < len(r)) for c in range(cols)]
    rowh = [max(im.size[1] for _, im in r) for r in rows]
    W = sum(colw) + pad * (cols + 1)
    H = sum(rowh) + (pad + label_h) * len(rows) + pad
    sheet = Image.new("RGB", (W, H), bg)
    dr = ImageDraw.Draw(sheet)
    y = pad
    for r, rh in zip(rows, rowh):
        x = pad
        for c, (lab, im) in enumerate(r):
            dr.text((x, y), lab, fill="#222", font=FONT)
            sheet.paste(im.convert("RGB") if im.mode != "RGB" else im, (x, y + label_h))
            x += colw[c] + pad
        y += rh + label_h + pad
    return sheet
