#!/usr/bin/env bash
# rtl/ctrl/lint.sh — Verilator lint (-Wall, warnings are errors) of the RISC-V control subsystem
# (rtl/ctrl, W4 item 3) with its vendored IP (third_party/ibex, riscv-dbg, common_cells: their own
# waivers, CC §3.1) and of the tb/ctrl bench HDL.
#
#   rtl/ctrl/lint.sh   every rtl/ctrl module as its own top, tsr_ctrl with and without RVFI, in the
#                      Ibex "small" and RV32MSingleCycle configurations, with the flop register file,
#                      CTRL_SYNC = 1, small TCMs and SPM, and with the Xilinx SRAM wrapper; then
#                      the bench tops tb/ctrl/sv/tb_ctrl.sv and tb_ctrl_core.sv; then the instance-name
#                      conventions of the CDC constraints (rtl/ctrl/viv/check_cdc_names.py)
# Waivers: rtl/ctrl/waivers/vendor.vlt only (file-scoped to third_party/); Ibex's own waiver file is
# not read, because its global PINCONNECTEMPTY waiver would apply to the whole core (W4-A
# integrate: tsr_ctrl_rv carries a targeted inline waiver for the ibex_top instance instead).
# Exit status: number of failing lint runs (0 = clean). rtl/lint.sh calls this script.
set -u
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
VERILATOR=${VERILATOR:-verilator}
fails=0
PRE=("$ROOT/rtl/ctrl/waivers/vendor.vlt"
     "$ROOT/rtl/core/tsr_tables_pkg.sv")
SRAM="$ROOT/rtl/tech/generic/tsr_sram_1r1w.sv"

lint() {   # lint TOP SRAM [extra verilator args...]
    local top=$1 sram=$2; shift 2
    local out
    if out=$("$VERILATOR" --lint-only -Wall --top-module "$top" "${PRE[@]}" "$sram" \
             -F "$ROOT/rtl/ctrl/filelist.f" "$@" 2>&1); then
        echo "lint ok   $top $*"
    else
        echo "lint FAIL $top $*"
        echo "$out" | grep -v '^- V e r i l a t i o n\|^- Verilator:' | sed 's/^/    /'
        fails=$((fails + 1))
    fi
}

for m in tsr_sync tsr_rst_sync tsr_cdc_gray tsr_cdc_fifo tsr_cdc_hs tsr_rv_bus tsr_ctl_win tsr_ctrl_rv \
         tsr_ctrl_cdc tsr_ctrl; do
    lint "$m" "$SRAM"
done
lint tsr_ctrl "$SRAM" -DRVFI
lint tsr_ctrl "$SRAM" -DTSR_CDC_JITTER
lint tsr_ctrl "$SRAM" -GWB_STAGE=0 -GBT_ALU=0 -GRV32M=2
lint tsr_ctrl "$SRAM" -GRV32M=3   # RV32MSingleCycle (maxperf proper)
lint tsr_ctrl "$SRAM" -GREGFILE=0
lint tsr_ctrl "$SRAM" -GCTRL_SYNC=1
lint tsr_ctrl "$SRAM" -GITCM_BYTES=16384 -GDTCM_BYTES=8192 -GSPM_BYTES=65536
lint tsr_ctrl "$SRAM" "-GQ_DEPTH=30'h08001088" -GCAPS=0   # VU and AT absent
lint tsr_ctrl "$ROOT/rtl/tech/xilinx/tsr_sram_1r1w.sv"
lint tsr_cdc_fifo "$SRAM" -GDEPTH=2 -GW=1
lint tsr_cdc_hs "$SRAM" -GTIMEOUT=0
lint tsr_sync "$SRAM" -GSTAGES=1 -DTSR_CDC_JITTER

# bench HDL (tb/ctrl/sv: not under tb/*/hdl, whose rtl/lint.sh loop has no rtl/ctrl sources)
for f in tb_ctrl_core tb_ctrl; do
    lint "$f" "$SRAM" -DRVFI "$ROOT/rtl/core/tsr_rv_local_transport.sv" \
        "$ROOT/tb/ctrl/sv/tb_ctrl_core.sv" "$ROOT/tb/ctrl/sv/tb_ctrl.sv"
done
# the instance names tsr_ctrl_cdc.xdc selects its cells by (u_sync*, u_rst*, u_hs*)
if out=$(python3 "$ROOT/rtl/ctrl/viv/check_cdc_names.py" "$ROOT/rtl" 2>&1); then
    echo "cdc ok    rtl/ctrl/viv/check_cdc_names.py ($(echo "$out" | tail -1))"
else
    echo "cdc FAIL  rtl/ctrl/viv/check_cdc_names.py"
    echo "$out" | sed 's/^/    /'
    fails=$((fails + 1))
fi
exit $fails
