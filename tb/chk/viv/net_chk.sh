#!/usr/bin/env bash
# tb/chk/viv/net_chk.sh OUTDIR — the post-synthesis netlist of tsr_chk (Vivado, U200 part, T_MAX 8, the
# tsr_pkg defaults otherwise) answers the directed ops of tb/chk/test_chk.py directed_ops() like the
# simulator does (opt-in, needs Vivado 2024.2: tb/chk/test_chk.py test_chk_netlist). Found by the W4-A
# review: the checker's operand extents were written by two always_ff blocks; Verilator and XSim
# simulated that correctly, Vivado kept one driver and the synthesised range checks were wrong (10 of
# 35 directed ops differed, six with X).
#   Steps: tb/tech/viv/tech_ooc.tcl tsr_chk synth_only netlist T_MAX=8 -> OUTDIR/net.v (write_verilog
#   -mode funcsim), every module renamed <name>_net -> gen_ops.py (the simulator's answers, ops.memh) ->
#   xvlog (netlist, tsr_tb_chk_net.sv) -> xelab -L unisims_ver (with glbl) -> xsim. The synthesis log must
#   have no multi-driven net (Synth 8-6858 / 8-6859 / 8-3352). Prints the NETCHK lines; exit status 0 iff
#   there is none and the testbench reports 0 mismatches.
# Environment: TSR_VIVADO_LOCK  flock file around every Vivado tool call (one Vivado at a time);
#              TSR_PYTHON       Python with numpy and cocotb (default python3); XILINX_VIVADO (else
#                               /tool/Xilinx/Vivado/2024.2/settings64.sh is sourced);
#              TSR_CHK_FILELIST another filelist than rtl/filelist_xilinx.f (e.g. one with a mutated
#                               tsr_chk.sv, to see that this regression catches it).
set -eu
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
out=$1
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
  -tclargs "$ROOT" tsr_chk "$out" synth_only netlist T_MAX=8 ${TSR_CHK_FILELIST:+"filelist=$TSR_CHK_FILELIST"} \
  > "$out/synth.log" 2>&1
if grep -qE '(CRITICAL WARNING|ERROR): .*(\[Synth 8-(6858|6859|3352)\]|multi-driven)' "$out/synth.log"; then
  echo "net_chk.sh: multi-driven nets in the synthesis of tsr_chk (see $out/synth.log)" >&2
  grep -E '(CRITICAL WARNING|ERROR): .*(\[Synth 8-(6858|6859|3352)\]|multi-driven)' "$out/synth.log" | head -5 >&2
  exit 1
fi
"${TSR_PYTHON:-python3}" "$ROOT/tb/chk/viv/gen_ops.py" "$out/ops.memh"
mods=$(sed -nE 's/^module ([A-Za-z_][A-Za-z0-9_$]*).*/\1/p' "$out/net.v" | grep -vx glbl)
expr=()
for m in $mods; do expr+=(-e "s/\\b$m\\b/${m}_net/g"); done
sed -E "${expr[@]}" "$out/net.v" > "$out/net_r.v"
grep -q '^module tsr_chk_net' "$out/net_r.v" || { echo "net_chk.sh: top module not found in net.v" >&2; exit 1; }
lock xvlog -sv net_r.v "$ROOT/tb/chk/viv/tsr_tb_chk_net.sv" > xvlog.log 2>&1
lock xelab -L unisims_ver -relax tsr_tb_chk_net glbl -s netchk > xelab.log 2>&1
lock xsim netchk -R > xsim.log 2>&1
grep 'NETCHK' xsim.log
grep -q 'NETCHK [0-9]* ops, 0 mismatches' xsim.log
