#!/usr/bin/env bash
# tb/gdn/viv/gdn_ooc.sh — out-of-context area and timing run of a GDN module (tb/gdn/viv/gdn_ooc.tcl). Opt-in,
# Vivado 2024.2 (batch). Run it through the slot wrapper: hw/u200/vivado_slot.sh tb/gdn/viv/gdn_ooc.sh ...
#
# usage: tb/gdn/viv/gdn_ooc.sh <outdir> top=<module> [srcs=<preset|file,file,...>] [clk=<ns>] [iodly=<ns>]
#          [flow=synth|place|route] [define=<NAME>] [GEN=value ...]
#   presets for srcs: fx2f, dot, unit (default: unit)
#   e.g. hw/u200/vivado_slot.sh tb/gdn/viv/gdn_ooc.sh build/gdn_ooc/dot top=tsr_vu_gdn_dot srcs=dot flow=route
# Environment: VIVADO_SETTINGS (default /tool/Xilinx/Vivado/2024.2/settings64.sh).
# Log: <outdir>/vivado.log; summary: grep GOOC <outdir>/vivado.log.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../../.." && pwd)"
if [ $# -lt 2 ]; then
  sed -n '2,10p' "$0"; exit 2
fi
out="$(mkdir -p "$1" && cd "$1" && pwd)"; shift
set +u
# shellcheck disable=SC1090
source "${VIVADO_SETTINGS:-/tool/Xilinx/Vivado/2024.2/settings64.sh}"
set -u
cd "$out"
vivado -mode batch -nojournal -nolog -source "$here/gdn_ooc.tcl" -tclargs "$repo" "$out" "$@" > "$out/vivado.log" 2>&1 || true
grep "^GOOC" "$out/vivado.log" || { echo "no GOOC lines; see $out/vivado.log"; tail -n 30 "$out/vivado.log"; exit 1; }
