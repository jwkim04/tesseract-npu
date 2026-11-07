#!/usr/bin/env bash
# tb/at/viv/lsel_net.sh rtl|legacy OUTDIR — post-synthesis netlist simulation of the ATTPV drain's lane select
# (rtl/core/tsr_mvm_pv.sv tsr_mvm_pv_lsel; opt-in: needs Vivado, tb/at/test_at_pv_netlist.py).
#   rtl     tsr_lsel_flat: the module the core uses
#   legacy  tsr_lsel_legacy_flat: the integer form the module replaced (a control: Vivado 2024.2 gets it wrong)
# Steps: tb/at/viv/lsel_ooc.tcl (synth_design out of context, rtl/filelist_xilinx.f, xcu200; OUTDIR/net.v, write_verilog
# -mode funcsim, modules renamed <name>_net) -> xvlog (the netlist, tb/at/viv/tsr_tb_lsel_net.sv) -> xelab
# -L unisims_ver -> xsim. Prints the testbench's NETSIM line; exit status 0 iff it reports 0 mismatches.
# Environment: TSR_VIVADO_WRAP  command that wraps every Vivado tool call (hw/u200/vivado_slot.sh: at most two Vivado
#              processes on the host); XILINX_VIVADO (else /tool/Xilinx/Vivado/2024.2/settings64.sh is sourced).
set -eu
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
form=$1
out=$2
case $form in
  rtl)    top=tsr_lsel_flat; def="" ;;
  legacy) top=tsr_lsel_legacy_flat; def="-d LEGACY" ;;
  *)      echo "lsel_net.sh: form must be rtl or legacy" >&2; exit 2 ;;
esac
if [ -z "${XILINX_VIVADO:-}" ]; then
  set +u
  # shellcheck disable=SC1091
  source /tool/Xilinx/Vivado/2024.2/settings64.sh
  set -u
fi
wrap() { if [ -n "${TSR_VIVADO_WRAP:-}" ]; then "$TSR_VIVADO_WRAP" "$@"; else "$@"; fi; }
mkdir -p "$out"
out=$(cd "$out" && pwd)
cd "$out"
wrap vivado -mode batch -nojournal -nolog -source "$ROOT/tb/at/viv/lsel_ooc.tcl" -tclargs "$ROOT" "$top" "$out" > synth.log 2>&1
grep '^LSEL' synth.log || true
mods=$(sed -nE 's/^module ([A-Za-z_][A-Za-z0-9_$]*).*/\1/p' net.v | grep -vx glbl)
expr=()
for m in $mods; do expr+=(-e "s/\\b$m\\b/${m}_net/g"); done
sed -E "${expr[@]}" net.v > net_r.v
grep -q "^module ${top}_net" net_r.v || { echo "lsel_net.sh: top module not found in net.v" >&2; exit 1; }
# shellcheck disable=SC2086
wrap xvlog -sv $def net_r.v "$ROOT/tb/at/viv/tsr_tb_lsel_net.sv" > xvlog.log 2>&1
wrap xelab -L unisims_ver -relax tsr_tb_lsel_net glbl -s netsim > xelab.log 2>&1
wrap xsim netsim -R > xsim.log 2>&1
grep 'NETSIM' xsim.log
grep -q 'NETSIM lsel .* mismatches 0$' xsim.log
