#!/usr/bin/env bash
# tb/vu/viv/vu_ooc.sh — out-of-context run of tsr_vu (tb/vu/viv/vu_ooc.tcl). Opt-in, Vivado 2024.2 (batch).
#
# usage: tb/vu/viv/vu_ooc.sh <outdir> [synth_only] [place_only] [clk=<ns>] [iodly=<ns>] [define=<NAME>] [GEN=value ...]
#   e.g. tb/vu/viv/vu_ooc.sh build/vu_ooc_u200 SPM_RD_LAT=4
#        tb/vu/viv/vu_ooc.sh build/vu_ooc_knobs SPM_RD_LAT=4 EXP_LANES=8 BACK_LANES=4
#        hw/u200/vivado_slot.sh tb/vu/viv/vu_ooc.sh build/vu_k4 place_only clk=5.0 iodly=2.4 \
#          FRONT_LANES=8 EXP_LANES=8 BACK_LANES=4 LANE_PIPE=2 SPM_RD_LAT=4 NUM_CH=4     (the K4 configuration)
# Environment: VIVADO_SETTINGS (default /tool/Xilinx/Vivado/2024.2/settings64.sh); TSR_VIVADO_LOCK, when set, is a
# flock file serializing Vivado runs (leave it unset under hw/u200/vivado_slot.sh, which holds the slot).
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
cd "$out"
cmd=(vivado -mode batch -nojournal -nolog -source "$here/vu_ooc.tcl" -tclargs "$repo" "$out" "$@")
if [ -n "${TSR_VIVADO_LOCK:-}" ]; then
  flock "$TSR_VIVADO_LOCK" "${cmd[@]}" > "$out/vivado.log" 2>&1 || true
else
  "${cmd[@]}" > "$out/vivado.log" 2>&1 || true
fi
grep "^VOOC" "$out/vivado.log" || { echo "no VOOC lines; see $out/vivado.log"; tail -n 30 "$out/vivado.log"; exit 1; }
