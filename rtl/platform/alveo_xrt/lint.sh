#!/usr/bin/env bash
# rtl/platform/alveo_xrt/lint.sh — Verilator lint (-Wall, warnings are errors) of the Alveo XRT
# platform adapter and the U200 kernel tops (docs/platform-u200.md §10), plus the generated-file
# check of gen_tops.py. Meant to be called from rtl/lint.sh.
#
#   rtl/platform/alveo_xrt/lint.sh      every adapter module as its own top with default and edge
#                                       parameters, tsr_xrt_kernel in reduced / edge
#                                       configurations, the packaged tops tsr_u200_k1 / k4 with
#                                       rtl/filelist.f and rtl/filelist_xilinx.f (when present), the
#                                       testbench wrappers tb/platform/sv/kwrap_*.sv
# Sources: the core filelist (rtl/filelist.f) + rtl/platform/alveo_xrt/filelist.f.
# Exit status: number of failing runs (0 = clean).
set -u
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
CORE_FL="$ROOT/rtl/filelist.f"
XIL_FL="$ROOT/rtl/filelist_xilinx.f"
PF="$ROOT/rtl/platform/alveo_xrt/filelist.f"
VERILATOR=${VERILATOR:-verilator}
fails=0

lint() {   # lint CORE_FILELIST TOP [extra verilator args...]
    local fl=$1 top=$2; shift 2
    local out
    if out=$("$VERILATOR" --lint-only -Wall --top-module "$top" -F "$fl" -F "$PF" "$@" 2>&1); then
        echo "lint ok   $top $* ($(basename "$fl"))"
    else
        echo "lint FAIL $top $* ($(basename "$fl"))"
        echo "$out" | grep -v '^- V e r i l a t i o n\|^- Verilator:' | sed 's/^/    /'
        fails=$((fails + 1))
    fi
}

if python3 "$ROOT/rtl/platform/alveo_xrt/gen_tops.py" --check; then
    echo "gen ok    gen_tops.py --check (tsr_u200_k1.v / k4.v, kernel_k1.xml / k4.xml, kwrap_*.sv)"
else
    fails=$((fails + 1))
fi

F=$CORE_FL
lint "$F" tsr_xrt_reset
lint "$F" tsr_xrt_reset -GN_CH=1 -GCORE_STAGES=2
lint "$F" tsr_xrt_reset -GN_CH=16 -GCORE_STAGES=6
lint "$F" tsr_xrt_reset -GN_CH=1 -GCH_COPIES=4 -GCORE_STAGES=3
lint "$F" tsr_xrt_reset -GN_CH=16 -GCH_COPIES=5 -GCORE_STAGES=2
lint "$F" tsr_xrt_slices
lint "$F" tsr_xrt_slices -GW=1 -GN=1
lint "$F" tsr_xrt_slices -GW=600 -GN=4
lint "$F" tsr_xrt_axi_bridge
lint "$F" tsr_xrt_axi_bridge -GSLICES=1 -GADDR_BEAT_W=5
lint "$F" tsr_xrt_axi_bridge -GSLICES=4 -GADDR_BEAT_W=40 -GRD_CNT_W=5 -GWR_CNT_W=12 -GCNT_RST=16
lint "$F" tsr_xrt_axi_bridge -GCNT_RST=1
lint "$F" tsr_xrt_ctrl
lint "$F" tsr_xrt_ctrl -GNUM_CH=1
lint "$F" tsr_xrt_ctrl -GNUM_CH=16
lint "$F" tsr_xrt_kernel
lint "$F" tsr_xrt_kernel -GNUM_CH=1 -GT_MAX=1 -GSPM_RD_LAT=4 -GMVM_BC_PIPE=2 -GMVM_RES_PIPE=2 -GMVM_MAC_LAT=9 -GRST_PIPE=3
lint "$F" tsr_xrt_kernel -GNUM_CH=4 -GT_MAX=4 -GSPM_RD_LAT=4 -GMVM_BC_PIPE=2 -GMVM_RES_PIPE=2 -GMVM_MAC_LAT=9 -GRST_PIPE=3 -GVU_FRONT_LANES=8 -GVU_EXP_LANES=8 -GVU_BACK_LANES=4 -GVU_LANE_PIPE=2
lint "$F" tsr_xrt_kernel -GNUM_CH=4 -GT_MAX=8 -GSPM_RD_LAT=4 -GMVM_BC_PIPE=2 -GMVM_RES_PIPE=2 -GMVM_MAC_LAT=9 -GRST_PIPE=3 -GRV_LOCAL_TRANSPORT=1 -GHAS_VU=0 -GHAS_ATT=0 -GHAS_TRW=0 -GHIST_DEPTH=0
lint "$F" tsr_xrt_kernel -GNUM_CH=4 -GT_MAX=4 -GSPM_RD_LAT=4 -GMVM_BC_PIPE=2 -GMVM_RES_PIPE=2 -GMVM_MAC_LAT=9 -GRST_PIPE=3 -GMVM_GATHER_PAIR=1 -GMVM_GATHER_FLITS=2 -GDDR_LOCAL_PIPE=1 -GRV_LOCAL_TRANSPORT=1 -GXSLR_REG=1 -GVU_FRONT_LANES=8 -GVU_EXP_LANES=8 -GVU_BACK_LANES=4 -GVU_LANE_PIPE=2
lint "$F" tsr_xrt_kernel -GNUM_CH=3 -GT_MAX=3 -GSPM_BYTES=262144 -GAXI_SLICES=1 -GRST_STAGES=2 -GRST_PIPE=1
lint "$F" tsr_xrt_kernel -GNUM_CH=16 -GT_MAX=2 -GK_MAX=8192 -GSPM_BYTES=262144 -GCH_BYTES="64'd1073741824" -GAXI_SLICES=4 -GRST_STAGES=6 -GRST_PIPE=8
lint "$F" tsr_xrt_kernel -GHAS_MVM=0 -GNUM_CH=1 -GSPM_BYTES=65536 -GCH_BYTES="64'd1073741824" -GK_MAX=4096 -GADDR_BEAT_W=22
lint "$F" tsr_u200_k1
lint "$F" tsr_u200_k4
if [ -f "$XIL_FL" ]; then
    lint "$XIL_FL" tsr_u200_k1
    lint "$XIL_FL" tsr_u200_k4
else
    echo "lint skip tsr_u200_k1 / k4 with rtl/filelist_xilinx.f (not present)"
fi
for f in "$ROOT"/tb/platform/sv/kwrap_*.sv; do
    [ -e "$f" ] || continue
    top=$(basename "$f" .sv)
    lint "$F" "$top" "$f"
done

[ $fails -eq 0 ] && echo "rtl/platform/alveo_xrt/lint.sh: all clean" || echo "rtl/platform/alveo_xrt/lint.sh: $fails failing run(s)"
exit $fails
