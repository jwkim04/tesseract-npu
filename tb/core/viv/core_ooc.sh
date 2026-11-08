#!/usr/bin/env bash
# tb/core/viv/core_ooc.sh — whole-core out-of-context Vivado runs of the U200 configurations
# (tb/core/viv/core_ooc.tcl; results: docs/rtl-m3.md §14). Opt-in, needs Vivado 2024.2.
# tsr_core has its command source since W4-A (the control core and the issue path), and the VU, the
# attention modes and the DB block since W4-B: synthesis keeps every unit. The VU's lane knobs are
# NAME=VALUE arguments (k1: VU_EXP_LANES=8 VU_BACK_LANES=4, the shrink knobs of its bitstream; k4 / k4t1: physical front / EXP / back 8 / 8 / 4 with
# VU_LANE_PIPE=2, four balanced lanes per outer SLR behind two register stages on each side.
# K4 presets also enable DDR0/3 local arbitration with central DMA boundaries and paired gather
# (DDR_LOCAL_PIPE=1, MVM_GATHER_PAIR=1; floorplan C'', core_slr.tcl); the k4 preset adds the two-flit gather and the
# RV transport, the packaged tsr_u200_k4.
# A synthesis of the whole core also checks what the block-level runs cannot: a register written from two
# blocks anywhere in the integration (Synth 8-6858 / 8-6859, found in tsr_dma_kvst this way, W4-B).
#
# usage: tb/core/viv/core_ooc.sh <config> <outdir> [core_ooc.tcl args ...]
#   config  k1    NUM_CH 1, T_MAX 1, VU 8 / 4 lanes, no lane pipe   (bring-up kernel tsr_u200_k1)
#           k4t1  NUM_CH 4, T_MAX 1, VU 8 / 8 / 4 lanes, VU_LANE_PIPE 2
#           k4    NUM_CH 4, T_MAX 4, VU 8 / 8 / 4 lanes, VU_LANE_PIPE 2, paired two-flit gather, DDR0 / DDR3 boundaries,
#                 RV_LOCAL_TRANSPORT 1, XSLR_REG 1, FAST_CLK 1, attention, trace writer and history: the kernel tsr_u200_k4 of
#                 rtl/platform/alveo_xrt/gen_tops.py, the full-feature card image (docs/platform-u200.md §8.8, §8.9)
#           k4base the same without FAST_CLK: the 200 MHz images of the fourth link and before
#           k4min NUM_CH 4, T_MAX 8, no vector unit, no attention (HAS_VU 0, HAS_ATT 0: no KVST, ATTQK, ATTPV),
#                 no DB trace writer or history (HAS_TRW 0, HIST_DEPTH 0), RV_LOCAL_TRANSPORT 1: the reduced first card
#                 image (branch k4m200, docs/platform-u200.md §8.7)
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
  sed -n '2,26p' "$0"; exit 2
fi
cfg="$1"; out="$2"; shift 2
case "$cfg" in
  k1)   p=(NUM_CH=1 T_MAX=1 VU_EXP_LANES=8 VU_BACK_LANES=4 VU_LANE_PIPE=0) ;;
  k4t1) p=(NUM_CH=4 T_MAX=1 VU_FRONT_LANES=8 VU_EXP_LANES=8 VU_BACK_LANES=4 VU_LANE_PIPE=2 DDR_LOCAL_PIPE=1 MVM_GATHER_PAIR=1) ;;
  k4)   p=(NUM_CH=4 T_MAX=4 VU_FRONT_LANES=8 VU_EXP_LANES=8 VU_BACK_LANES=4 VU_LANE_PIPE=2 DDR_LOCAL_PIPE=1 MVM_GATHER_PAIR=1 MVM_GATHER_FLITS=2 RV_LOCAL_TRANSPORT=1 XSLR_REG=1 FAST_CLK=1) ;;
  k4base) p=(NUM_CH=4 T_MAX=4 VU_FRONT_LANES=8 VU_EXP_LANES=8 VU_BACK_LANES=4 VU_LANE_PIPE=2 DDR_LOCAL_PIPE=1 MVM_GATHER_PAIR=1 MVM_GATHER_FLITS=2 RV_LOCAL_TRANSPORT=1 XSLR_REG=1) ;;
  k4min) p=(NUM_CH=4 T_MAX=8 RV_LOCAL_TRANSPORT=1 HAS_VU=0 HAS_ATT=0 HAS_TRW=0 HIST_DEPTH=0) ;;
  *) echo "core_ooc.sh: unknown config '$cfg' (k1, k4t1, k4, k4base, k4min)" >&2; exit 2 ;;
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
