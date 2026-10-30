"""Lockups "Tesseract Basis" and "Tesseract Core": the Tesseract lockup (same mark, same
wordmark construction and colours, parameters of build.py) with one more word.

  build/tools/svgvenv/bin/python docs/logo/src/sub.py

Writes tesseract-{basis,core}-lockup{,-dark}.svg and -lockup.png into docs/logo/, and the
inline-safe logo files docs/fig/logos/tesseract-{basis,core}.svg (title and desc dropped) that
the figure library embeds.
"""
import os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lock import lockup
from build import COL, WORD_LIGHT, WORD_DARK, D_EYE
import cairosvg

DST = os.path.dirname(HERE)
FIGLOGOS = os.path.join(os.path.dirname(os.path.dirname(DST)), "docs", "fig", "logos")
LK = dict(R=200.0, ratio=2.15, S=21.0, track=7.0, gap_k=0.56, d=D_EYE)


def main():
    for key, word in (("basis", "Tesseract Basis"), ("core", "Tesseract Core")):
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
    main()
