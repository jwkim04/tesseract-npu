"""Lockups "Tesseract Basis" and "Tesseract Core": the Tesseract lockup (same mark, same
wordmark construction and colors, parameters of build.py) with one more word.

  build/tools/svgvenv/bin/python docs/logo/src/sub.py

Writes tesseract-{basis,core}-lockup{,-dark}.svg and -lockup.png into docs/logo/, and the
inline-safe logo files docs/fig/logos/tesseract-{basis,core}.svg (title and desc dropped) that
the figure library embeds.

It also writes tesseract-{basis,core}-lockup-universal.png: the small inline variant for the README
list (shown at height 20), 64 px tall, tight margins, 256-color palette PNG. The wordmark is
#7765E8 as in tesseract-lockup-universal.png, so it reads on white and on #0d1117. Only these two
files, without touching the others:

  build/tools/svgvenv/bin/python docs/logo/src/sub.py universal
"""
import collections, io, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lock import lockup
from build import COL, WORD_LIGHT, WORD_DARK, D_EYE
import cairosvg
from PIL import Image

DST = os.path.dirname(HERE)
FIGLOGOS = os.path.join(os.path.dirname(os.path.dirname(DST)), "docs", "fig", "logos")
LK = dict(R=200.0, ratio=2.15, S=21.0, track=7.0, gap_k=0.56, d=D_EYE)

WORD_UNIVERSAL = "#7765E8"      # contrast 4.36 on #ffffff, 4.34 on #0d1117
UNI_H = 64                      # px; 3.2x the 20 px the README shows it at
UNI_MARGIN_K = 0.015            # margin = 6 units around the 412-unit-high lockup (default 0.10)


def palette(im, n=256):
    """RGBA -> palette PNG with per-entry alpha. Median cut on the exact (r,g,b,a) values weighted
    by pixel count (alpha is a fourth axis), so the antialiased edges keep their coverage; Pillow's
    own RGBA quantizer (octree) loses it, and its error is 4x larger."""
    px = list(zip(*[iter(im.tobytes())] * 4))
    cnt = collections.Counter(px)
    boxes = [list(cnt.items())]
    while len(boxes) < n:
        best, bi, bc = 0, None, 0
        for i, b in enumerate(boxes):
            for c in range(4):
                v = [p[c] for p, _ in b]
                if max(v) - min(v) > best:
                    best, bi, bc = max(v) - min(v), i, c
        if bi is None:
            break
        b = sorted(boxes.pop(bi), key=lambda t: t[0][bc])
        tot, acc, k = sum(w for _, w in b), 0, 0
        for k, (_, w) in enumerate(b):
            acc += w
            if acc >= tot / 2:
                break
        k = min(k + 1, len(b) - 1)
        boxes += [b[:k], b[k:]]
    pal = [tuple(round(sum(p[c] * w for p, w in b) / sum(w for _, w in b)) for c in range(4)) for b in boxes]
    near = {p: min(range(len(pal)), key=lambda j: sum((p[c] - pal[j][c]) ** 2 for c in range(4))) for p in cnt}
    out = Image.frombytes("P", im.size, bytes(near[p] for p in px))
    out.putpalette([v for p in pal for v in p[:3]])
    out.info["transparency"] = bytes(p[3] for p in pal)
    return out


def universal(key, word):
    """tesseract-<key>-lockup-universal.png: same mark and wordmark as the lockup SVGs (lock.py), only
    the margin and the wordmark color differ; rendered straight from the SVG text, no SVG written."""
    svg, W, H = lockup(COL, WORD_UNIVERSAL, text=word, margin_k=UNI_MARGIN_K, **LK)
    w = round(W * UNI_H / H)
    im = Image.open(io.BytesIO(cairosvg.svg2png(bytestring=svg.encode(), output_width=w,
                                                output_height=UNI_H))).convert("RGBA")
    palette(im).save(os.path.join(DST, f"tesseract-{key}-lockup-universal.png"), optimize=True)


def main(only_universal=False):
    for key, word in (("basis", "Tesseract Basis"), ("core", "Tesseract Core")):
        universal(key, word)
        if only_universal:
            continue
        for suffix, colour in (("", WORD_LIGHT), ("-dark", WORD_DARK)):
            svg, W, H = lockup(COL, colour, text=word, **LK)
            p = os.path.join(DST, f"tesseract-{key}-lockup{suffix}.svg")
            open(p, "w").write(svg)
            if not suffix:
                cairosvg.svg2png(url=p, write_to=os.path.join(DST, f"tesseract-{key}-lockup.png"),
                                 output_width=2 * W, output_height=2 * H)
                bare = re.sub(r'\s*<(title|desc)[^>]*>.*?</\1>', '', svg)
                bare = re.sub(r'\s*role="img" aria-labelledby="[^"]*"', '', bare)
                open(os.path.join(FIGLOGOS, f"tesseract-{key}.svg"), "w").write(bare)


if __name__ == "__main__":
    main(only_universal=sys.argv[1:] == ["universal"])
