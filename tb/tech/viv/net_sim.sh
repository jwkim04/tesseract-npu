#!/usr/bin/env bash
# tb/tech/viv/net_sim.sh — post-synthesis netlist simulation of one rtl/tech/xilinx module against
# the generic RTL of the same module (Vivado synthesis + xsim with the unisim models; opt-in, needs
# Vivado: tb/tech/test_tech.py test_tech_netlist).
#
#   tb/tech/viv/net_sim.sh mac|sram OUTDIR [NAME=VALUE ...]
#     mac   tsr_mac_lane   (NAME=VALUE: N, T, LAT)            testbench tsr_tb_tech_net_mac
#     sram  tsr_sram_1r1w  (NAME=VALUE: DEPTH, WIDTH, RD_LAT) testbench tsr_tb_tech_net_sram
#   Steps: tb/tech/viv/tech_ooc.tcl synth_only netlist (rtl/filelist_xilinx.f, xcu200, the module
#   out of context with these generics) -> OUTDIR/net.v (write_verilog -mode funcsim), top renamed
#   <top>_net (mac: the flat-port wrapper tb/tech/viv/tsr_tech_flat_mac.sv around the lane) ->
#   xvlog (generic RTL, netlist incl. glbl, tb/tech/viv/tsr_tb_tech_net.sv) -> xelab
#   -L unisims_ver -> xsim. Prints the testbench's NETSIM line; exit status 0 iff it reports
#   0 mismatches.
# Environment: TSR_VIVADO_LOCK  flock file wrapped around every Vivado tool call (one Vivado at a
#              time across concurrent jobs); NETSIM_CYCLES random cycles of the testbench;
#              XILINX_VIVADO (else /tool/Xilinx/Vivado/2024.2/settings64.sh is sourced).
set -eu
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
kind=$1
out=$2
shift 2
case $kind in
  mac)  mod=tsr_mac_lane;  top=tsr_tech_flat_mac; tb=tsr_tb_tech_net_mac ;;
  sram) mod=tsr_sram_1r1w; top=tsr_sram_1r1w;     tb=tsr_tb_tech_net_sram ;;
  *)    echo "net_sim.sh: kind must be mac or sram" >&2; exit 2 ;;
esac
if [ -z "${XILINX_VIVADO:-}" ]; then
  set +u   # settings64.sh reads unset variables
  # shellcheck disable=SC1091
  source /tool/Xilinx/Vivado/2024.2/settings64.sh
  set -u
fi
lock() {
  if [ -n "${TSR_VIVADO_LOCK:-}" ]; then flock "$TSR_VIVADO_LOCK" "$@"; else "$@"; fi
}
mkdir -p "$out"
out=$(cd "$out" && pwd)
cd "$out"   # Vivado / xsim leave .Xil and xsim.dir in the working directory

lock vivado -mode batch -nojournal -nolog -source "$ROOT/tb/tech/viv/tech_ooc.tcl" \
  -tclargs "$ROOT" "$top" "$out" synth_only netlist "extra=$ROOT/tb/tech/viv/tsr_tech_flat_mac.sv" "$@" \
  > "$out/synth.log" 2>&1
grep '^TOOC' "$out/synth.log" | grep -v ' path \|r2r_wns' || true
# Rename every module the netlist defines (the top and kept sub-hierarchies such as the lane inside
# the flat wrapper; not glbl, which it also defines) to <name>_net, so that none of them replaces
# the generic RTL reference (unisim primitives come from unisims_ver and are not defined here).
mods=$(sed -nE 's/^module ([A-Za-z_][A-Za-z0-9_$]*).*/\1/p' "$out/net.v" | grep -vx glbl)
expr=()
for m in $mods; do expr+=(-e "s/\\b$m\\b/${m}_net/g"); done
sed -E "${expr[@]}" "$out/net.v" > "$out/net_r.v"
grep -q "^module ${top}_net" "$out/net_r.v" || { echo "net_sim.sh: top module not found in net.v" >&2; exit 1; }

gens=()
for a in "$@"; do gens+=(-generic_top "$a"); done
if [ -n "${NETSIM_CYCLES:-}" ]; then gens+=(-generic_top "CYCLES=$NETSIM_CYCLES"); fi
lock xvlog -sv "$ROOT/rtl/tech/generic/$mod.sv" net_r.v "$ROOT/tb/tech/viv/tsr_tb_tech_net.sv" > xvlog.log 2>&1
if grep -q 'overwriting previous definition' xvlog.log; then
  echo "net_sim.sh: a netlist module redefines an RTL module (see $out/xvlog.log)" >&2
  exit 1
fi
lock xelab -L unisims_ver -relax "$tb" glbl "${gens[@]}" -s netsim > xelab.log 2>&1
lock xsim netsim -R > xsim.log 2>&1
grep 'NETSIM' xsim.log
grep -q 'NETSIM .* mismatches 0$' xsim.log
