#!/usr/bin/env bash
# tb/core/viv/blk_ooc.sh — block out-of-context runs of the K4 image at any core clock period
# (tb/core/viv/blk_ooc.tcl; docs/platform-u200.md §8.10). Opt-in, Vivado 2024.2 (batch).
#
# usage: tb/core/viv/blk_ooc.sh <block> <period ns> <outdir> [blk_ooc.tcl args ...] [NAME=VALUE ...]
#   block  cl      tsr_mvm_cluster: one MVM cluster with attention (CH 0, NUM_CH 4, T_MAX 4, MAC_LAT 9)
#          gather  tsr_mvm_gather: four assemblers, paired two-flit pair links, XSLR_REG 1, HAS_ATT
#          vu      tsr_vu: NUM_CH 4, lanes 8 / 8 / 4, LANE_PIPE 2 (add HAS_TOPK=1 for TOPK)
#          dma     tsr_dma with KVST, NUM_CH 4
#          spm     tsr_spm, 4 MiB, RD_LAT 4
#          qact    tsr_qact
#          ctl     tsr_core_dp without MVM, VU and attention: the issue path with its checkers, the CSR
#                  block, the DB block (core side, trace writer, history), DMA, QACT, SPM, the arbiters
#                  and the XSLR_REG / reset glue, with the K4 parameters
#          dbgrv   tsr_dbg_rv: the clk_rv part of the DB block (HAS_DB 1: g_db.u_dbg_rv of tsr_core), with the Ibex taps.
#                  ctl with HAS_DB=0 and without it give the core-domain part of the DB block (tsr_dbg, trace writer mux)
#   The cl, gather, dma and ctl presets have FAST_CLK 1 (the 250 MHz stages; FAST_CLK=0 gives the 200 MHz netlist). vu: add
#   HAS_GDN=1 for the fused GDN update (branch gdn-vu), HAS_TOPK=1 for TOPK.
#   Extra NAME=VALUE arguments after the block's own (they override) go to synth_design -generic; other
#   arguments (synth_only, place_only, dcp, from=<dcp>, io=, thr=) go to blk_ooc.tcl.
# Environment: VIVADO_SETTINGS (default /tool/Xilinx/Vivado/2024.2/settings64.sh). Run it through
# hw/u200/vivado_slot.sh. Log: <outdir>/vivado.log; summary: grep BOOC <outdir>/vivado.log.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../../.." && pwd)"
if [ $# -lt 3 ]; then sed -n '2,23p' "$0"; exit 2; fi
blk="$1"; per="$2"; out="$3"; shift 3
case "$blk" in
  cl)     top=tsr_mvm_cluster; p=(CH=0 NUM_CH=4 HAS_ATT=1 T_MAX=4 MAC_LAT=9 RST_COPY=1 FAST_CLK=1) ;;
  gather) top=tsr_mvm_gather;  p=(HAS_ATT=1 NUM_CH=4 T_MAX=4 SPM_RD_LAT=4 JOB_PIPE=2 GATHER_PAIR=1 GATHER_FLITS=2 XSLR_REG=1 FAST_CLK=1) ;;
  vu)     top=tsr_vu;          p=(NUM_CH=4 SPM_RD_LAT=4 FRONT_LANES=8 EXP_LANES=8 BACK_LANES=4 LANE_PIPE=2) ;;
  dma)    top=tsr_dma;         p=(NUM_CH=4 SPM_RD_LAT=4 HAS_KVST=1 FAST_CLK=1) ;;
  spm)    top=tsr_spm;         p=(RD_LAT=4) ;;
  qact)   top=tsr_qact;        p=(T_MAX=4 SPM_RD_LAT=4) ;;
  ctl)    top=tsr_core_dp;     p=(NUM_CH=4 T_MAX=4 HAS_MVM=0 HAS_VU=0 HAS_ATT=0 SPM_RD_LAT=4 DDR_LOCAL_PIPE=1 RV_LOCAL_TRANSPORT=1 XSLR_REG=1 RST_PIPE=3 FAST_CLK=1) ;;
  dbgrv)  top=tsr_dbg_rv;      p=(HAS_RV_TAPS=1) ;;
  *) echo "blk_ooc.sh: unknown block '$blk' (cl, gather, vu, dma, spm, qact, ctl, dbgrv)" >&2; exit 2 ;;
esac
mkdir -p "$out"
out="$(cd "$out" && pwd)"
set +u   # the Xilinx settings scripts read unset variables
# shellcheck disable=SC1090
source "${VIVADO_SETTINGS:-/tool/Xilinx/Vivado/2024.2/settings64.sh}"
set -u
cd "$out"
vivado -mode batch -nojournal -nolog -source "$here/blk_ooc.tcl" -tclargs "$repo" "$out" "$top" "$per" "${p[@]}" "$@" \
  > "$out/vivado.log" 2>&1
grep '^BOOC\|^COOC util' "$out/vivado.log" || true
