#!/bin/sh
# Regenerates every figure in docs/fig (light) and the dark variants of the README figures.
# Previews go to $TSR_FIG_PNG_DIR (default build/fig/png, git-ignored). The interpreter is the
# figure venv (cairosvg, fontTools); set TSR_FIG_PYTHON to use another one.
set -e
here=$(cd "$(dirname "$0")" && pwd)
py=${TSR_FIG_PYTHON:-$here/../../../build/tools/svgvenv/bin/python}
for n in hero stack system launch core cluster layers verify milestones issue debug alloc tree; do
    "$py" "$here/$n.py" > /dev/null
done
for n in hero stack system; do
    TSR_FIG_THEME=dark "$py" "$here/$n.py" > /dev/null
done
