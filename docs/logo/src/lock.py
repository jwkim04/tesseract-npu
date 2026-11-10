"""Lockup composition: mark (computed in place) + wordmark."""
import math, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from tess import Frame, mark_body, svg_doc, C30
from word import wordmark, translate_d, f

def lockup(colours, word_colour, R=200.0, ratio=2.2, S=20.0, track=7.0, gap_k=0.62,
           margin_k=0.10, d=4.0, G=None, g=None, lift=0.0, wkw=None, text="Tesseract"):
    """Return (svg, W, H). Mark hex circumradius R; cap height = 2R/ratio."""
    wkw = dict(wkw or {})
    G = G if G is not None else 20.0 * R / (192 / C30)    # same proportions as the 512 mark
    g = g if g is not None else 5.0 * R / (192 / C30)
    cap = 2 * R / ratio
    k = cap / 140.0                                       # glyph cap height is 140 x-units
    parts, Wtxt = wordmark(text=text, S=S, track=track, **wkw)
    m = margin_k * 2 * R
    H = 2 * R + 2 * m
    cx = m + R * C30
    cy = H / 2
    fr = Frame(cx, cy, R, d)
    x_text = cx + R * C30 + gap_k * cap
    # vertical: center the band between baseline and the mean of x-height and cap height
    X = 100 * k
    base = cy + (X + cap) / 4 + lift * cap
    word = "".join(translate_d(p, x + x_text / k, base / k, k) for x, p in parts)
    W = x_text + Wtxt * k + m
    body = (f'  <g id="mark">\n{mark_body(fr, G=G, g=g, colours=colours, indent="    ", floor_n=4)}\n  </g>\n'
            f'  <path id="wordmark" fill="{word_colour}" d="{word}"/>')
    W = math.ceil(W); H = math.ceil(H)
    return svg_doc(f"0 0 {W} {H}", W, H, body, desc=("Tesseract logo: the cube-in-cube tesseract mark and the Tesseract wordmark."
                                          if text == "Tesseract" else f"{text} logo: the Tesseract mark and the wordmark {text}.")), W, H
