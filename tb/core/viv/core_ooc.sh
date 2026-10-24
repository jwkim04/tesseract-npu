#!/usr/bin/env bash
# tb/core/viv/core_ooc.sh — whole-core out-of-context Vivado runs of the U200 configurations
# (tb/core/viv/core_ooc.tcl; results: docs/rtl-m3.md §14). Opt-in, needs Vivado 2024.2.
#
# usage: tb/core/viv/core_ooc.sh <config> <outdir> [core_ooc.tcl args ...]
#   config  k1    NUM_CH 1, T_MAX 1   (bring-up kernel tsr_u200_k1)
#           k4t1  NUM_CH 4, T_MAX 1
#           k4    NUM_CH 4, T_MAX 8   (target kernel tsr_u200_k4)
#   Every config: rtl/filelist_xilinx.f, SPM 4 MiB with SPM_RD_LAT 4, MVM_BC_PIPE 2,
#   MVM_RES_PIPE 2, MVM_MAC_LAT 9 (a register after every adder level of the DSP-packed MAC
#   lane), RST_PIPE 3 (reset copies per block), CH_BYTES 16 GiB, the other parameters at the
#   tsr_pkg defaults: the U200 bitstream parameters of docs/rtl-m3.md §14.
#   Extra arguments go to core_ooc.tcl (synth_only, dcp, from=, floorplan=, NAME=VALUE overrides).
# Environment: VIVADO_SETTINGS (default /tool/Xilinx/Vivado/2024.2/settings64.sh); TSR_VIVADO_LOCK
# = a flock file that serialises Vivado runs on a shared host (none by default).
# Log: <outdir>/vivado.log; summary lines: grep COOC <outdir>/vivado.log.
# Examples:
#   tb/core/viv/core_ooc.sh k4 /tmp/k4s synth_only dcp
#   tb/core/viv/core_ooc.sh k4 /tmp/k4i from=/tmp/k4s/post_synth.dcp dcp
#   tb/core/viv/core_ooc.sh k4 /tmp/k4f from=/tmp/k4s/post_synth.dcp floorplan=tb/core/viv/core_slr.tcl
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../../.." && pwd)"
if [ $# -lt 2 ]; then
  sed -n '2,20p' "$0"; exit 2
fi
cfg="$1"; out="$2"; shift 2
case "$cfg" in
  k1)   p=(NUM_CH=1 T_MAX=1) ;;
  k4t1) p=(NUM_CH=4 T_MAX=1) ;;
  k4)   p=(NUM_CH=4 T_MAX=8) ;;
  *) echo "core_ooc.sh: unknown config '$cfg' (k1, k4t1, k4)" >&2; exit 2 ;;
esac
p+=(SPM_RD_LAT=4 MVM_BC_PIPE=2 MVM_RES_PIPE=2 MVM_MAC_LAT=9 RST_PIPE=3)
mkdir -p "$out"
set +u   # the Xilinx settings scripts read unset variables
# shellcheck disable=SC1090
source "${VIVADO_SETTINGS:-/tool/Xilinx/Vivado/2024.2/settings64.sh}"
set -u
cmd=(vivado -mode batch -nojournal -nolog -source "$here/core_ooc.tcl" -tclargs "$repo" "$out" "${p[@]}" "$@")
cd "$out"
if [ -n "${TSR_VIVADO_LOCK:-}" ]; then
  flock "$TSR_VIVADO_LOCK" "${cmd[@]}" > "$out/vivado.log" 2>&1
else
  "${cmd[@]}" > "$out/vivado.log" 2>&1
fi
grep '^COOC' "$out/vivado.log" | grep -v 'COOC hier\|COOC slr\|COOC sll' || true
