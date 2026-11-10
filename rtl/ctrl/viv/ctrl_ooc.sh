#!/usr/bin/env bash
# rtl/ctrl/viv/ctrl_ooc.sh — gate T1 runs (CC §14.3) of the control subsystem out of context
# (rtl/ctrl/viv/ctrl_ooc.tcl). Opt-in, needs Vivado 2024.2 (batch only).
#
# usage: rtl/ctrl/viv/ctrl_ooc.sh <config> <outdir> [ctrl_ooc.tcl args ...]
#   config  maxperf   Ibex WritebackStage 1, BranchTargetALU 1, RV32MSingleCycle (CC-34)
#           mfast     maxperf with RV32MFast (CC §3.2 fallback 1; the RTL default after T1)
#           small     WritebackStage 0, BranchTargetALU 0, RV32MFast (CC §3.2 fallback 2)
#   Extra arguments go to ctrl_ooc.tcl (rv=4.0 for 250 MHz, top=tsr_ctrl, synth_only).
# Environment: VIVADO_SETTINGS (default /tool/Xilinx/Vivado/2024.2/settings64.sh), TSR_VIVADO_LOCK
# (a flock file serializing Vivado runs on a shared host). Without a lock the script waits until
# no other Vivado process runs (one Vivado at a time on the 14 GB build host).
# Log: <outdir>/vivado.log; summary: grep CTRLOOC <outdir>/vivado.log.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../../.." && pwd)"
if [ $# -lt 2 ]; then
  sed -n '2,15p' "$0"; exit 2
fi
cfg="$1"; out="$2"; shift 2
case "$cfg" in
  maxperf) p=(RV32M=3 WB_STAGE=1 BT_ALU=1) ;;
  mfast)   p=(RV32M=2 WB_STAGE=1 BT_ALU=1) ;;
  small)   p=(RV32M=2 WB_STAGE=0 BT_ALU=0) ;;
  *) echo "ctrl_ooc.sh: unknown config '$cfg' (maxperf, mfast, small)" >&2; exit 2 ;;
esac
mkdir -p "$out"
set +u
# shellcheck disable=SC1090
source "${VIVADO_SETTINGS:-/tool/Xilinx/Vivado/2024.2/settings64.sh}"
set -u
run() {
  (cd "$out" && vivado -mode batch -nojournal -nolog -source "$here/ctrl_ooc.tcl" \
     -tclargs "$repo" "$out" "${p[@]}" "$@" > "$out/vivado.log" 2>&1)
}
if [ -n "${TSR_VIVADO_LOCK:-}" ]; then
  exec 9>"$TSR_VIVADO_LOCK"; flock 9
else
  while pgrep -f 'unwrapped/lnx64.o/vivado' > /dev/null; do sleep 30; done
fi
run "$@"
grep CTRLOOC "$out/vivado.log" || true
