#!/bin/sh
# Regenerates every figure in docs/fig (light), the dark variants of the README figures, and the
# PNG renderings of the README figures (docs/fig/<name>.png, what README.md embeds so that the
# GitHub mobile app shows them; see pngs.py).
# Previews go to $TSR_FIG_PNG_DIR (default build/fig/png, git-ignored). The interpreter is the
# figure venv (cairosvg, fontTools, Pillow); set TSR_FIG_PYTHON to use another one.
set -e
here=$(cd "$(dirname "$0")" && pwd)
py=${TSR_FIG_PYTHON:-$here/../../../build/tools/svgvenv/bin/python}
for n in hero stack system dataflow launch core cluster layers verify issue debug alloc tree clocks; do
    "$py" "$here/$n.py" > /dev/null
done
for n in hero stack system; do
    TSR_FIG_THEME=dark "$py" "$here/$n.py" > /dev/null
done
"$py" "$here/pngs.py" > /dev/null
