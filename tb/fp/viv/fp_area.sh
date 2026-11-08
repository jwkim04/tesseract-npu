#!/usr/bin/env bash
# tb/fp/viv/fp_area.sh — out-of-context area run of an f32 operator array (tb/fp/viv/fp_area.tcl). Opt-in,
# Vivado 2024.2 (batch). Run it through the slot wrapper: hw/u200/vivado_slot.sh tb/fp/viv/fp_area.sh ...
#
# usage: tb/fp/viv/fp_area.sh <outdir> [key=value ...]     (keys: see fp_area.tcl)
#   e.g. tb/fp/viv/fp_area.sh build/fp_area/add kind=add
#        tb/fp/viv/fp_area.sh build/fp_area/add_usedsp kind=add usedsp=1
# Environment: VIVADO_SETTINGS (default /tool/Xilinx/Vivado/2024.2/settings64.sh).
# Log: <outdir>/vivado.log; summary: grep FPAREA <outdir>/vivado.log.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../../.." && pwd)"
if [ $# -lt 1 ]; then
  sed -n '2,8p' "$0"; exit 2
fi
out="$(mkdir -p "$1" && cd "$1" && pwd)"; shift
set +u
# shellcheck disable=SC1090
source "${VIVADO_SETTINGS:-/tool/Xilinx/Vivado/2024.2/settings64.sh}"
set -u
cd "$out"
vivado -mode batch -nojournal -nolog -source "$here/fp_area.tcl" -tclargs "$repo" "$out" "$@" > "$out/vivado.log" 2>&1 || true
grep "^FPAREA" "$out/vivado.log" || { echo "no FPAREA lines; see $out/vivado.log"; tail -n 30 "$out/vivado.log"; exit 1; }
