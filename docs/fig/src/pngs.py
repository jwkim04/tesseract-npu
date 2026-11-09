#!/usr/bin/env python
"""Render the README figures to PNG: docs/fig/<name>.png from docs/fig/<name>.svg.

Why. The README embeds each figure as one plain <img> of a PNG. The <picture> element with SVG
sources showed a broken image in the GitHub mobile app (the logo had the same problem).
A plain <img> of a PNG shows everywhere. The light SVG is the source: every figure card
carries its own background, so one PNG reads on both GitHub themes. The SVGs stay the sources, and
the technical documents under docs/ keep embedding them.

What. 1760 px wide (twice the 880 px of GitHub's README column), RGBA with the card's rounded
corners transparent, then reduced to a 256-colour palette (octree, no dithering). Figures are flat
colour with anti-aliased edges, so the palette costs at most about 30 of 255 in a few pixels and
the file is about a quarter of the size of the full-colour PNG. The result has no text chunks,
timestamps or other metadata, so the same SVG always gives the same bytes (for one Pillow and zlib
version). The script fails if a PNG is larger than BUDGET.

Run (the figure venv has cairosvg and Pillow; `build.sh` calls this after the figure scripts):

    build/tools/svgvenv/bin/python docs/fig/src/pngs.py [name ...]     (default: README_FIGS)
"""
import hashlib
import io
import sys
from pathlib import Path

sys.dont_write_bytecode = True

FIG_DIR = Path(__file__).resolve().parent.parent       # docs/fig
README_FIGS = ("hero", "stack", "dataflow")     # the figures README.md embeds
WIDTH = 1760                                           # px: 2 x 880, GitHub's README column
BUDGET = 500_000                                       # bytes per PNG


def render(svg_path: Path) -> bytes:
    """PNG bytes of the SVG at WIDTH px, palette mode, no metadata."""
    import cairosvg
    from PIL import Image

    raw = cairosvg.svg2png(bytestring=svg_path.read_bytes(), output_width=WIDTH)
    im = Image.open(io.BytesIO(raw)).convert("RGBA")
    pal = im.quantize(colors=256, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE)
    out = io.BytesIO()
    pal.save(out, "PNG", optimize=True)                # no pnginfo, no dpi: no extra chunks
    return out.getvalue()


def main(names) -> int:
    status = 0
    for n in names or README_FIGS:
        svg, png = FIG_DIR / f"{n}.svg", FIG_DIR / f"{n}.png"
        if not svg.is_file():
            print(f"missing {svg}", file=sys.stderr)
            status = 1
            continue
        data = render(svg)
        png.write_bytes(data)
        print(f"wrote {png} {len(data)} bytes sha256 {hashlib.sha256(data).hexdigest()[:16]}")
        if len(data) > BUDGET:
            print(f"{png} is over the {BUDGET} byte budget", file=sys.stderr)
            status = 1
    return status


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
