#!/usr/bin/env bash
# tb/vu/viv/vu_ooc.sh — out-of-context run of tsr_vu (tb/vu/viv/vu_ooc.tcl). Opt-in, Vivado 2024.2 (batch).
#
# usage: tb/vu/viv/vu_ooc.sh <outdir> [synth_only] [GEN=value ...]
#   e.g. tb/vu/viv/vu_ooc.sh build/vu_ooc_u200 SPM_RD_LAT=4
#        tb/vu/viv/vu_ooc.sh build/vu_ooc_knobs SPM_RD_LAT=4 EXP_LANES=8 BACK_LANES=4
# Environment: VIVADO_SETTINGS (default /tool/Xilinx/Vivado/2024.2/settings64.sh), TSR_VIVADO_LOCK (a flock
# file serialising Vivado runs on the shared host; default /tmp/claude-1000/-home-limagin-dev-tesseract/vivado.lock).
# Log: <outdir>/vivado.log; summary: grep VOOC <outdir>/vivado.log.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../../.." && pwd)"
if [ $# -lt 1 ]; then
  sed -n '2,11p' "$0"; exit 2
fi
out="$(mkdir -p "$1" && cd "$1" && pwd)"; shift
set +u
# shellcheck disable=SC1090
source "${VIVADO_SETTINGS:-/tool/Xilinx/Vivado/2024.2/settings64.sh}"
set -u
lock="${TSR_VIVADO_LOCK:-/tmp/claude-1000/-home-limagin-dev-tesseract/vivado.lock}"
cd "$out"
flock "$lock" vivado -mode batch -nojournal -nolog -source "$here/vu_ooc.tcl" -tclargs "$repo" "$out" "$@" \
  > "$out/vivado.log" 2>&1
grep VOOC "$out/vivado.log" || true
