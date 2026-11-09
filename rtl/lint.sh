#!/usr/bin/env bash
# rtl/lint.sh — Verilator lint (-Wall, warnings are errors) of the RTL in rtl/filelist.f.
#
#   rtl/lint.sh            lint every module of rtl/filelist.f's own sources as its own top with
#                          default parameters (tsr_core: the U200 configuration with the MVM unit
#                          and the control subsystem, rtl/core/tsr_core.sv header; tsr_core_dp: its
#                          core-clock datapath; the nested rtl/ctrl/filelist.f is linted by
#                          rtl/ctrl/lint.sh), then tsr_core
#                          with HAS_MVM = 0 and HAS_MVM = 1 in reduced configurations
#                          (1 / 2 / 3 / 16 channels, small SPM, T_MAX 8, small K_MAX, narrow DRAM
#                          address) and edge implementation parameters (SPM_RD_LAT 1 / 4,
#                          RD_OUTSTANDING 1 / 2, MVM pipes 0 / 3, WS FIFO 64 / 1024, MVM
#                          latencies 1 / 8, reset register stages RST_PIPE 1 / 3 / 8; W4-A: queue
#                          depths 0 / 1 / 31, CTRL_SYNC 1, small TCMs; W4-B: the VU, attention
#                          and DB parameters of tsr_core / tsr_core_dp each off, the lane knobs, the M8
#                          options, the DB sizes, the k1 / k4 / bench shapes), then the
#                          M3 blocks with edge parameters (tsr_mvm also RST_LOCAL 1 / 2), the W4-B blocks
#                          (tsr_vu and its checker, KVST on the DMA, the attention modes of the MVM unit and
#                          the MAC lane, tsr_chk_at, the DB modules) with edge parameters, then the
#                          testbench-only HDL under tb/*/hdl (each file as top, with the other
#                          files of its directory), then the generated tables (EX-30, CC
#                          §13.4: sw/tables/gen_tables.py --check and a lint of
#                          rtl/core/tsr_tables_pkg.sv on its own, when the generator exists), then
#                          the target, control-subsystem and platform lint scripts (M3 part C,
#                          W4-A): rtl/ctrl/lint.sh (the RISC-V control subsystem and the tb/ctrl
#                          bench HDL), rtl/tech/xilinx/lint.sh (Xilinx technology wrappers, the core
#                          on rtl/filelist_xilinx.f), rtl/platform/alveo_xrt/lint.sh (XRT adapter,
#                          kernel tops, gen_tops.py --check) and rtl/platform/alveo_xrt/bw/lint.sh
#                          (M0 bandwidth kernel, gen_bw.py --check)
#   rtl/lint.sh MODULE...  lint only these modules (default parameters, rtl/filelist.f)
#
# No blanket waivers: a module needing a lint_off must carry it inline, targeted, with a reason.
# Exit status: number of failing lint runs, those of the sub-scripts included (0 = clean).
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
FL="$ROOT/rtl/filelist.f"
VERILATOR=${VERILATOR:-verilator}
fails=0

lint() {   # lint TOP [extra verilator args...]
    local top=$1; shift
    local out
    if out=$("$VERILATOR" --lint-only -Wall --top-module "$top" -F "$FL" "$@" 2>&1); then
        echo "lint ok   $top $*"
    else
        echo "lint FAIL $top $*"
        echo "$out" | grep -v '^- V e r i l a t i o n\|^- Verilator:' | sed 's/^/    /'
        fails=$((fails + 1))
    fi
}

modules() {   # module names declared in the filelist's own sources (packages excluded; nested
              # -F lists, +incdir / +define options and .vlt configuration files skipped)
    grep -v '^\s*//' "$FL" | grep -v '^\s*$' | grep -v '^\s*[-+]' | grep -v '\.vlt\s*$' | while read -r f; do
        sed -n 's/^\s*module\s\+\([A-Za-z_][A-Za-z0-9_]*\).*/\1/p' "$ROOT/rtl/$f"
    done
}

if [ $# -gt 0 ]; then
    for m in "$@"; do lint "$m"; done
    exit $fails
fi

for m in $(modules); do lint "$m"; done

# Reduced / edge configurations of the core datapath. The default tsr_core (above) is the U200
# configuration with the MVM unit (HAS_MVM = 1); HAS_MVM = 0 builds first.
lint tsr_core -GHAS_MVM=0
lint tsr_core -GHAS_MVM=0 -GNUM_CH=1 -GSPM_BYTES=65536 -GCH_BYTES="64'd1073741824" -GK_MAX=4096
lint tsr_core -GHAS_MVM=0 -GNUM_CH=3 -GT_MAX=8 -GSPM_BYTES=262144 -GADDR_BEAT_W=22 -GCH_BYTES="64'd4294967296"
lint tsr_core -GHAS_MVM=0 -GNUM_CH=16 -GT_MAX=8 -GK_MAX=65280 -GADDR_BEAT_W=36 -GCH_BYTES="64'd17179869184"
# Implementation parameters.
lint tsr_core -GHAS_MVM=0 -GSPM_RD_LAT=4
lint tsr_core -GHAS_MVM=0 -GNUM_CH=1 -GSPM_BYTES=65536 -GSPM_RD_LAT=1 -GRD_OUTSTANDING=1
# HAS_MVM = 1 (docs/rtl-mvm.md §1, integration step): U200 with T_MAX 8, 1 / 3 / 16 channels,
# small K_MAX, pipe / FIFO / latency extremes.
lint tsr_core -GHAS_MVM=1 -GT_MAX=8
lint tsr_core -GHAS_MVM=1 -GNUM_CH=1 -GT_MAX=2 -GSPM_BYTES=65536 -GCH_BYTES="64'd1073741824" -GK_MAX=4096 -GADDR_BEAT_W=22
lint tsr_core -GHAS_MVM=1 -GNUM_CH=3 -GT_MAX=3 -GSPM_BYTES=262144 -GADDR_BEAT_W=22 -GCH_BYTES="64'd4294967296"
lint tsr_core -GHAS_MVM=1 -GNUM_CH=16 -GT_MAX=2 -GK_MAX=8192 -GSPM_BYTES=262144 -GCH_BYTES="64'd1073741824"
lint tsr_core -GHAS_MVM=1 -GNUM_CH=16 -GT_MAX=8 -GK_MAX=65280 -GADDR_BEAT_W=36 -GCH_BYTES="64'd17179869184"
lint tsr_core -GHAS_MVM=1 -GNUM_CH=2 -GT_MAX=4 -GK_MAX=256 -GSPM_BYTES=4096 -GCH_BYTES="64'd4096" -GADDR_BEAT_W=5
lint tsr_core -GHAS_MVM=1 -GMVM_BC_PIPE=0 -GMVM_RES_PIPE=0 -GMVM_WS_FIFO=64 -GSPM_RD_LAT=1 -GRD_OUTSTANDING=1
lint tsr_core -GHAS_MVM=1 -GMVM_BC_PIPE=3 -GMVM_RES_PIPE=3 -GMVM_WS_FIFO=1024 -GSPM_RD_LAT=4
# (attention needs MVM_MAC_LAT >= 3: the exact PV accumulation of tsr_mac_lane)
lint tsr_core -GHAS_MVM=1 -GHAS_ATT=0 -GT_MAX=3 -GMVM_ACT_RD_LAT=1 -GMVM_MAC_LAT=1 -GMVM_MUL_LAT=1 -GMVM_ADD_LAT=1 -GMVM_WS_RD_LAT=1
lint tsr_core -GHAS_MVM=1 -GNUM_CH=3 -GMVM_ACT_RD_LAT=4 -GMVM_MAC_LAT=8 -GMVM_MUL_LAT=8 -GMVM_ADD_LAT=8 -GMVM_WS_RD_LAT=4 -GRD_OUTSTANDING=2
# Reset register stages (RST_PIPE >= 1, docs/rtl-mvm.md §14.4): HAS_MVM = 0, the U200 bitstream
# parameters of §14.2 (RST_PIPE 3), and the maximum depth on a reduced configuration.
lint tsr_core -GHAS_MVM=0 -GRST_PIPE=1
lint tsr_core -GHAS_MVM=1 -GRST_PIPE=3 -GSPM_RD_LAT=4 -GMVM_BC_PIPE=2 -GMVM_RES_PIPE=2 -GMVM_MAC_LAT=9
lint tsr_core -GHAS_MVM=1 -GNUM_CH=3 -GT_MAX=3 -GSPM_BYTES=262144 -GADDR_BEAT_W=22 -GCH_BYTES="64'd4294967296" -GRST_PIPE=8
# Issue path and control subsystem (W4-A, CC §7.4, §13.1): queue depths at their limits (0 = unit
# absent, 1, 31; SYNC >= 1), CTRL_SYNC = 1, small TCMs with a small SPM.
lint tsr_core -GQD_DMA=1 -GQD_QACT=1 -GQD_MVM=1 -GQD_SYNC=1
lint tsr_core -GQD_DMA=31 -GQD_QACT=31 -GQD_MVM=31 -GQD_SYNC=31 -GRST_PIPE=3
lint tsr_core -GHAS_MVM=1 -GQD_MVM=0 -GQD_QACT=0
lint tsr_core -GCTRL_SYNC=1 -GITCM_BYTES=16384 -GDTCM_BYTES=8192 -GNUM_CH=2 -GSPM_BYTES=65536 -GCH_BYTES="64'd1073741824" -GADDR_BEAT_W=24 -GK_MAX=4096 -GT_MAX=2
lint tsr_core_dp -GHAS_MVM=0 -GNUM_CH=1 -GSPM_BYTES=4096 -GRST_PIPE=2
# VU, attention and the DB block (W4-B, tsr_core_dp parameters): each off, the knobs, the M8 options, the DB
# sizes at their limits, the k1 / k4 bitstream shapes, the queue depth of the VU
lint tsr_core -GHAS_DB=0
lint tsr_core -GHAS_VU=0 -GHAS_ATT=0
lint tsr_core -GHAS_ATT=0
lint tsr_core -GHAS_VU=0 -GHAS_ATT=0 -GHAS_DB=0
lint tsr_core -GHAS_TOPK=1 -GHAS_DQROW=1
lint tsr_core -GVU_EXP_LANES=8 -GVU_BACK_LANES=4
lint tsr_core -GHAS_TRW=0 -GHAS_REC=0 -GNTBP=0 -GNRVWP=0 -GHIST_DEPTH=2
lint tsr_core -GHAS_TRW=0 -GNTBP=2 -GNRVWP=1 -GHIST_DEPTH=64 -GREC_DEPTH=4 -GCTRL_SYNC=1
lint tsr_core -GQD_VU=1
lint tsr_core -GQD_VU=31 -GQD_DMA=31 -GQD_QACT=31 -GQD_MVM=31 -GQD_SYNC=31 -GRST_PIPE=3
lint tsr_core -GNUM_CH=1 -GT_MAX=1 -GRST_PIPE=3 -GSPM_RD_LAT=4 -GMVM_BC_PIPE=2 -GMVM_RES_PIPE=2 -GMVM_MAC_LAT=9 -GVU_EXP_LANES=8 -GVU_BACK_LANES=4
lint tsr_core -GNUM_CH=4 -GT_MAX=8 -GRST_PIPE=3 -GSPM_RD_LAT=4 -GMVM_BC_PIPE=2 -GMVM_RES_PIPE=2 -GMVM_MAC_LAT=9 -GVU_LANE_PIPE=2 -GVU_FRONT_LANES=8 -GVU_EXP_LANES=8 -GVU_BACK_LANES=4
# Optional K4 local consumers and the physical two-flit result link. Defaults
# intentionally retain the legacy path, so lint these elaborations explicitly.
lint tsr_core -GNUM_CH=4 -GT_MAX=4 -GSPM_BYTES=4194304 -GSPM_RD_LAT=4 -GMVM_RES_PIPE=2 -GMVM_GATHER_PAIR=1 -GMVM_GATHER_FLITS=2 -GLOCAL_INTERMEDIATES=1 -GLOCAL_SCALE=1 -GLOCAL_SOFTCAP=1 -GVU_FRONT_LANES=8 -GVU_EXP_LANES=8 -GVU_BACK_LANES=4 -GVU_LANE_PIPE=2
lint tsr_core -GNUM_CH=4 -GT_MAX=4 -GMVM_RES_PIPE=2 -GMVM_GATHER_PAIR=1 -GMVM_GATHER_FLITS=2 -GVU_FRONT_LANES=8 -GVU_EXP_LANES=8 -GVU_BACK_LANES=4
# The reduced first card image (tsr_u200_k4, gen_tops.py): four clusters, T_MAX 8, the RV transport, no vector unit,
# no attention (KVST, ATTQK, ATTPV), no trace writer or history.
lint tsr_core -GNUM_CH=4 -GT_MAX=8 -GRST_PIPE=3 -GSPM_RD_LAT=4 -GMVM_BC_PIPE=2 -GMVM_RES_PIPE=2 -GMVM_MAC_LAT=9 -GRV_LOCAL_TRANSPORT=1 -GHAS_VU=0 -GHAS_ATT=0 -GHAS_TRW=0 -GHIST_DEPTH=0
lint tsr_core -GNUM_CH=4 -GT_MAX=8 -GRST_PIPE=3 -GSPM_RD_LAT=4 -GMVM_BC_PIPE=2 -GMVM_RES_PIPE=2 -GMVM_MAC_LAT=9 -GRV_LOCAL_TRANSPORT=1 -GHAS_VU=0 -GHAS_ATT=0 -GHAS_TRW=0 -GHIST_DEPTH=0 -GDDR_LOCAL_PIPE=1 -GMVM_GATHER_PAIR=1 -GMVM_GATHER_FLITS=2
# The full-feature K4 image (tsr_u200_k4, gen_tops.py): four clusters, T_MAX 4, the vector unit 8 / 8 / 4 with two lane pipe
# stages, attention with masked KVST, the DB block with trace writer and history, paired two-flit gather, the DDR0 / DDR3
# boundaries and the RV transport. The local-intermediate banks stay off.
lint tsr_core -GNUM_CH=4 -GT_MAX=4 -GRST_PIPE=3 -GSPM_RD_LAT=4 -GMVM_BC_PIPE=2 -GMVM_RES_PIPE=2 -GMVM_MAC_LAT=9 -GMVM_GATHER_PAIR=1 -GMVM_GATHER_FLITS=2 -GDDR_LOCAL_PIPE=1 -GRV_LOCAL_TRANSPORT=1 -GXSLR_REG=1 -GVU_FRONT_LANES=8 -GVU_EXP_LANES=8 -GVU_BACK_LANES=4 -GVU_LANE_PIPE=2
# The same with FAST_CLK 1 (docs/platform-u200.md §8.10): the 250 MHz stages of the card image.
lint tsr_core -GNUM_CH=4 -GT_MAX=4 -GRST_PIPE=3 -GSPM_RD_LAT=4 -GMVM_BC_PIPE=2 -GMVM_RES_PIPE=2 -GMVM_MAC_LAT=9 -GMVM_GATHER_PAIR=1 -GMVM_GATHER_FLITS=2 -GDDR_LOCAL_PIPE=1 -GRV_LOCAL_TRANSPORT=1 -GXSLR_REG=1 -GFAST_CLK=1 -GVU_FRONT_LANES=8 -GVU_EXP_LANES=8 -GVU_BACK_LANES=4 -GVU_LANE_PIPE=2
lint tsr_core -GFAST_CLK=1
lint tsr_core -GHAS_MVM=0 -GFAST_CLK=1
lint tsr_core_dp -GHAS_TRW=0 -GNUM_CH=3 -GT_MAX=3 -GRST_PIPE=1 -GFAST_CLK=1
lint tsr_mvm_pair_tx -GW=563
lint tsr_mvm_pair_rx -GW=563
lint tsr_core -GVU_LANE_PIPE=1 -GVU_EXP_LANES=8 -GVU_BACK_LANES=4
lint tsr_core -GHAS_GDN=1 -GNUM_CH=4 -GT_MAX=4 -GRST_PIPE=3 -GSPM_RD_LAT=4 -GMVM_BC_PIPE=2 -GMVM_RES_PIPE=2 -GMVM_MAC_LAT=9 -GMVM_GATHER_PAIR=1 -GMVM_GATHER_FLITS=2 -GDDR_LOCAL_PIPE=1 -GRV_LOCAL_TRANSPORT=1 -GXSLR_REG=1 -GVU_FRONT_LANES=8 -GVU_EXP_LANES=8 -GVU_BACK_LANES=4 -GVU_LANE_PIPE=2
lint tsr_core -GHAS_GDN=1 -GHAS_TOPK=1 -GHAS_DQROW=1
lint tsr_core -GNUM_CH=16 -GT_MAX=8 -GK_MAX=65280 -GADDR_BEAT_W=36 -GCH_BYTES="64'd17179869184" -GHAS_TOPK=1 -GHAS_DQROW=1
lint tsr_core -GNUM_CH=2 -GT_MAX=2 -GK_MAX=4096 -GSPM_BYTES=1048576 -GCH_BYTES="64'd536870912" -GADDR_BEAT_W=24
lint tsr_core_dp -GHAS_DB=0 -GHAS_VU=0 -GHAS_ATT=0 -GNUM_CH=1 -GSPM_BYTES=4096 -GRST_PIPE=2
lint tsr_core_dp -GHAS_TRW=0 -GNUM_CH=3 -GT_MAX=3 -GRST_PIPE=1

# VU (docs/design/02-vector-unit.md "VU" §7): lanes, channels, SPM sizes and latencies, the M8 options, the
# checker class and the xc routing
lint tsr_vu -GNUM_CH=1
lint tsr_vu -GNUM_CH=16 -GHAS_TOPK=1 -GHAS_DQROW=1
lint tsr_vu -GEXP_LANES=8 -GBACK_LANES=4
lint tsr_vu -GEXP_LANES=8 -GBACK_LANES=4 -GHAS_TOPK=1 -GHAS_DQROW=1
lint tsr_vu -GHAS_GDN=1
lint tsr_vu -GHAS_GDN=1 -GFRONT_LANES=8 -GEXP_LANES=8 -GBACK_LANES=4 -GLANE_PIPE=2
lint tsr_vu -GHAS_GDN=1 -GFRONT_LANES=8 -GEXP_LANES=8 -GBACK_LANES=4 -GLANE_PIPE=2 -GHAS_TOPK=1 -GHAS_DQROW=1 -GGDN_NSLOT=16 -GGDN_STYLE=\"block\"
lint tsr_vu -GSPM_BYTES=65536 -GSPM_RD_LAT=1
lint tsr_vu -GLANE_PIPE=1
lint tsr_vu -GLANE_PIPE=2
lint tsr_vu -GLANE_PIPE=3 -GEXP_LANES=8 -GBACK_LANES=4 -GHAS_TOPK=1 -GHAS_DQROW=1
lint tsr_vu_lanes -GPIPE=1
lint tsr_vu_lanes -GPIPE=2
lint tsr_vu_lanes -GPIPE=3 -GEXP_LANES=8 -GBACK_LANES=4
lint tsr_vu -GNUM_CH=3 -GSPM_RD_LAT=4
lint tsr_vu_chk -GNUM_CH=1 -GSPM_BYTES=65536 -GK_MAX=4096
lint tsr_vu_chk -GNUM_CH=16 -GK_MAX=65280
lint tsr_vu_chk -GHAS_GDN=1
lint tsr_vu_chk -GHAS_GDN=1 -GNUM_CH=1 -GSPM_BYTES=65536 -GK_MAX=4096
lint tsr_vu_xc -GHAS_GDN=1
lint tsr_vu_xc -GNUM_CH=3 -GSPM_BYTES=262144 -GK_MAX=4096

# AT (docs/design/03-attention-kv.md "AT" §8): KVST on the DMA, the attention cluster / gather / unit modes, the
# PV mode of the MAC lane, the checker class
lint tsr_dma -GHAS_KVST=1
lint tsr_dma -GHAS_KVST=1 -GNUM_CH=1 -GSPM_BYTES=65536 -GSPM_RD_LAT=1
lint tsr_dma -GHAS_KVST=1 -GNUM_CH=16 -GADDR_BEAT_W=36 -GSPM_RD_LAT=4
lint tsr_mac_lane -GN=32 -GT=2 -GLAT=3 -GPV=1
lint tsr_mac_lane -GN=32 -GT=8 -GLAT=9 -GPV=1
lint tsr_mac_lane -GN=5 -GT=1 -GLAT=4 -GPV=1
lint tsr_mvm_back -GHAS_ATT=1 -GK_MAX=256 -GT_MAX=8 -GACT_RD_LAT=1 -GMAC_LAT=3 -GMUL_LAT=1 -GADD_LAT=1
lint tsr_mvm_back -GHAS_ATT=1 -GK_MAX=65280 -GT_MAX=2 -GACT_RD_LAT=3 -GMAC_LAT=9 -GMUL_LAT=5 -GADD_LAT=8
lint tsr_mvm_cluster -GHAS_ATT=1 -GCH=0 -GNUM_CH=1 -GK_MAX=256 -GT_MAX=1 -GADDR_BEAT_W=5 -GWS_FIFO_BEATS=64 -GMAC_LAT=3
lint tsr_mvm_cluster -GHAS_ATT=1 -GCH=3 -GNUM_CH=4 -GK_MAX=4096 -GT_MAX=8 -GMAC_LAT=9
lint tsr_mvm_cluster -GHAS_ATT=1 -GCH=15 -GNUM_CH=16 -GK_MAX=4096 -GT_MAX=3 -GADDR_BEAT_W=40 -GADD_LAT=8 -GMAC_LAT=4
lint tsr_mvm_gather -GHAS_ATT=1 -GNUM_CH=1 -GT_MAX=8 -GSPM_BYTES=4096 -GSPM_RD_LAT=1 -GADD_LAT=1
lint tsr_mvm_gather -GHAS_ATT=1 -GNUM_CH=3 -GT_MAX=3 -GSPM_BYTES=262144 -GSPM_RD_LAT=4 -GADD_LAT=8
lint tsr_mvm_gather -GHAS_ATT=1 -GNUM_CH=16 -GT_MAX=8 -GSPM_BYTES=4194304
lint tsr_mvm_gather -GHAS_ATT=1 -GNUM_CH=4 -GT_MAX=4 -GSPM_BYTES=4194304 -GJOB_PIPE=2 -GGATHER_PAIR=1 -GGATHER_FLITS=2 -GXSLR_REG=1
lint tsr_mvm_gather -GHAS_ATT=1 -GNUM_CH=4 -GT_MAX=4 -GSPM_BYTES=4194304 -GJOB_PIPE=2 -GGATHER_PAIR=1 -GXSLR_REG=1
lint tsr_mvm -GHAS_ATT=1 -GNUM_CH=1 -GT_MAX=1 -GK_MAX=256 -GSPM_BYTES=4096 -GADDR_BEAT_W=5 -GBC_PIPE=0 -GRES_PIPE=0 -GMAC_LAT=3
lint tsr_mvm -GHAS_ATT=1 -GNUM_CH=3 -GT_MAX=3 -GK_MAX=4096 -GSPM_BYTES=262144 -GADDR_BEAT_W=22 -GBC_PIPE=3 -GRES_PIPE=3 -GSPM_RD_LAT=4 -GMAC_LAT=9
lint tsr_mvm -GHAS_ATT=1 -GNUM_CH=16 -GT_MAX=8 -GK_MAX=65280 -GADDR_BEAT_W=36 -GWS_FIFO_BEATS=1024 -GMAC_LAT=5
lint tsr_chk_at -GNUM_CH=3 -GT_MAX=3 -GK_MAX=4096 -GSPM_BYTES=262144 -GADDR_BEAT_W=22 -GCH_BYTES="64'd4294967296"
lint tsr_chk_at -GNUM_CH=16 -GT_MAX=8 -GK_MAX=65280 -GADDR_BEAT_W=36 -GCH_BYTES="64'd17179869184"
lint tsr_chk_at -GNUM_CH=1 -GT_MAX=1 -GK_MAX=1024 -GSPM_BYTES=65536 -GCH_BYTES="64'd1048576" -GADDR_BEAT_W=40

# DB (docs/design/05-debug.md "DB" §10): the sizes at their limits (also tb/dbg's lint test)
lint tsr_dbg -GNUM_CH=1 -GHIST_DEPTH=0 -GHAS_TRW=0 -GNTBP=0 -GNRVWP=0 -GHAS_REC=0
lint tsr_dbg -GNUM_CH=16 -GADDR_BEAT_W=36 -GQD_ATT=4 -GQD_VU=31 -GNTBP=2 -GNRVWP=1 -GHIST_DEPTH=64
lint tsr_dbg_stop -GNTBP=0 -GNUM_CH=1
lint tsr_dbg_shq -GQD_DMA=31 -GQD_VU=0 -GQD_MVM=1
lint tsr_dbg_hist -GDEPTH=2
lint tsr_dbg_trw -GADDR_BEAT_W=22 -GCH_BYTES="64'd268435456"
lint tsr_dbg_rv -GHAS_RV_TAPS=1
lint tsr_dbg_rv -GHAS_RV_TAPS=1 -GCTRL_SYNC=1 -GREC_DEPTH=4
lint tsr_dbg_wrmux -GADDR_BEAT_W=5

# M3 MVM unit (docs/rtl-mvm.md §1): every new module as its own top with edge parameters
# (NUM_CH 1 / 3 / 16, T_MAX 1 / 8, small and maximal K_MAX, latency / pipe depth extremes).
lint tsr_fp_add -GLAT=1
lint tsr_fp_add -GLAT=8
lint tsr_fp_add_dsp -GLAT=8
lint tsr_fp_add_dsp -GNORM_DSP=1
# GDN unit (VU §7.12): the 16-lane build, small and large shapes, RAM styles, read latency 2
lint tsr_vu_gdn -GLANES=16 -GNB_MAX=16
lint tsr_vu_gdn -GLANES=16 -GNB_MAX=8 -GNSLOT=4
lint tsr_vu_gdn -GNB_MAX=16 -GNSLOT=4 -GR_MAX=64
lint tsr_vu_gdn_upd -GLANES=16 -GNB_MAX=16
lint tsr_vu_gdn_dlane -GPAIR=2
lint tsr_vu_gdn -GRD_LAT=2
lint tsr_vu_gdn -GSTYLE=\"block\"
lint tsr_vu_gdn_ram -GSTYLE=\"distributed\" -GDEPTH=64 -GRD_LAT=1
lint tsr_vu_gdn_ram -GNWE=2 -GWIDTH=64 -GRD_LAT=4
lint tsr_mac_lane -GT=8 -GLAT=1
lint tsr_mac_lane -GN=32 -GT=2 -GLAT=6
lint tsr_vpipe -GDEPTH=0 -GVW=2 -GW=527
lint tsr_vpipe -GDEPTH=3 -GVW=2 -GW=664
lint tsr_rpipe -GDEPTH=0 -GW=289
lint tsr_rpipe -GDEPTH=3 -GW=289
lint tsr_mvm_ws -GFIFO_BEATS=64 -GADDR_BEAT_W=22
lint tsr_mvm_ws -GFIFO_BEATS=1024 -GADDR_BEAT_W=40 -GRD_LAT=4
lint tsr_mvm_ws -GFIFO_BEATS=64 -GADDR_BEAT_W=5 -GRD_LAT=1
lint tsr_mvm_front -GK_MAX=256 -GADDR_BEAT_W=5 -GWS_FIFO_BEATS=64
lint tsr_mvm_front -GK_MAX=65280 -GADDR_BEAT_W=40 -GWS_FIFO_BEATS=1024 -GWS_RD_LAT=1
lint tsr_mvm_front -GK_MAX=4096 -GADDR_BEAT_W=36 -GWS_RD_LAT=4
lint tsr_mvm_actbuf -GK_MAX=256 -GT_MAX=8 -GRD_LAT=1
lint tsr_mvm_actbuf -GK_MAX=65280 -GT_MAX=3 -GRD_LAT=3
lint tsr_mvm_back -GK_MAX=256 -GT_MAX=8 -GACT_RD_LAT=1 -GMAC_LAT=1 -GMUL_LAT=1 -GADD_LAT=1
lint tsr_mvm_back -GK_MAX=65280 -GT_MAX=2 -GACT_RD_LAT=3 -GMAC_LAT=6 -GMUL_LAT=5 -GADD_LAT=8
lint tsr_mvm_cluster -GCH=0 -GK_MAX=256 -GT_MAX=8 -GADDR_BEAT_W=5 -GWS_FIFO_BEATS=64
lint tsr_mvm_cluster -GCH=15 -GK_MAX=4096 -GT_MAX=3 -GADDR_BEAT_W=40 -GADD_LAT=8
lint tsr_mvm_gather -GNUM_CH=1 -GT_MAX=8 -GSPM_BYTES=4096 -GSPM_RD_LAT=1 -GADD_LAT=1
lint tsr_mvm_gather -GNUM_CH=3 -GT_MAX=3 -GSPM_BYTES=262144 -GSPM_RD_LAT=4 -GADD_LAT=8
lint tsr_mvm_gather -GNUM_CH=16 -GT_MAX=8 -GSPM_BYTES=4194304
lint tsr_mvm_gather -GNUM_CH=3 -GT_MAX=3 -GSPM_BYTES=262144 -GSPM_RD_LAT=4 -GADD_LAT=8 -GFAST_CLK=1
lint tsr_mvm -GNUM_CH=1 -GT_MAX=8 -GK_MAX=256 -GSPM_BYTES=4096 -GADDR_BEAT_W=5 -GBC_PIPE=0 -GRES_PIPE=0
lint tsr_mvm -GNUM_CH=3 -GT_MAX=3 -GK_MAX=4096 -GSPM_BYTES=262144 -GADDR_BEAT_W=22 -GBC_PIPE=3 -GRES_PIPE=3 -GSPM_RD_LAT=4
lint tsr_mvm -GNUM_CH=16 -GT_MAX=8 -GK_MAX=65280 -GADDR_BEAT_W=36 -GWS_FIFO_BEATS=1024
lint tsr_mvm -GNUM_CH=3 -GT_MAX=3 -GRST_LOCAL=1
lint tsr_mvm -GNUM_CH=1 -GT_MAX=8 -GK_MAX=256 -GSPM_BYTES=4096 -GADDR_BEAT_W=5 -GBC_PIPE=0 -GRES_PIPE=0 -GRST_LOCAL=2
lint tsr_mvm -GNUM_CH=4 -GT_MAX=4 -GRST_LOCAL=2 -GXSLR_REG=1
lint tsr_mvm -GNUM_CH=4 -GT_MAX=4 -GRST_LOCAL=2 -GXSLR_REG=1 -GFAST_CLK=1
lint tsr_mvm_rvec -GPIPE=1
lint tsr_mvm_rvec -GPIPE=1 -GSPM_AW=11
lint tsr_rslice -GFORM=1 -GRPC=8 -GW=577
lint tsr_rslice -GFORM=1 -GRPC=3 -GW=130
lint tsr_mvm_pair_rx -GRPC=8
lint tsr_csr -GFAST_CLK=1
lint tsr_dma -GHAS_KVST=1 -GFAST_CLK=1
lint tsr_mvm_cluster -GCH=0 -GK_MAX=256 -GT_MAX=4 -GHAS_ATT=1 -GMAC_LAT=9 -GFAST_CLK=1
lint tsr_mvm_pv -GFAST_CLK=1
lint tsr_rv_spm_tx -GXSLR_REG=1

# Testbench-only HDL (block testbench toplevels); linted with the same rules. Each file is a top,
# together with the other files of its directory (wrappers may instantiate each other).
for d in "$ROOT"/tb/*/hdl; do
    [ -d "$d" ] || continue
    tb_extra=()
    # The attention bench checks the same production broadcast with the MVM
    # bench's independent cycle-history observer.
    if [ "$d" = "$ROOT/tb/at/hdl" ]; then
        tb_extra+=("$ROOT/tb/mvm/hdl/tsr_mvm_bc_observer.sv")
    fi
    for f in "$d"/*.sv; do
        [ -e "$f" ] || continue
        top=$(sed -n 's/^\s*module\s\+\([A-Za-z_][A-Za-z0-9_]*\).*/\1/p' "$f" | head -n 1)
        # a file that only binds into an RTL module (tb/core/hdl/tsr_tb_local_tcm.sv reaches the TCMs through
        # $root.tsr_core) is linted with that module as the top: its hierarchical references need it
        bind_top=$(sed -n 's/^\s*bind\s\+\([A-Za-z_][A-Za-z0-9_]*\)\s.*/\1/p' "$f" | head -n 1)
        lint_top=${bind_top:-$top}
        if out=$("$VERILATOR" --lint-only -Wall --top-module "$lint_top" -F "$FL" "$d"/*.sv "${tb_extra[@]}" 2>&1); then
            echo "lint ok   $top (tb)"
        else
            echo "lint FAIL $top (tb)"
            echo "$out" | sed 's/^/    /'
            fails=$((fails + 1))
        fi
    done
done

# Generated tables (EX-30, CC §13.4): the generated files are current, and the SV package
# lints on its own.
if [ -f "$ROOT/sw/tables/gen_tables.py" ]; then
    if out=$(python3 "$ROOT/sw/tables/gen_tables.py" --check 2>&1); then
        echo "gen ok    sw/tables/gen_tables.py --check"
    else
        echo "gen FAIL  sw/tables/gen_tables.py --check"
        echo "$out" | sed 's/^/    /'
        fails=$((fails + 1))
    fi
    if out=$("$VERILATOR" --lint-only -Wall "$ROOT/rtl/core/tsr_tables_pkg.sv" 2>&1); then
        echo "lint ok   tsr_tables_pkg (package)"
    else
        echo "lint FAIL tsr_tables_pkg (package)"
        echo "$out" | grep -v '^- V e r i l a t i o n\|^- Verilator:' | sed 's/^/    /'
        fails=$((fails + 1))
    fi
fi

# Target and platform lint (M3 part C). Each script prints its own runs and exits with its
# number of failing runs; a missing script counts as a failure.
for sub in rtl/ctrl/lint.sh rtl/tech/xilinx/lint.sh rtl/platform/alveo_xrt/lint.sh rtl/platform/alveo_xrt/bw/lint.sh; do
    if [ -x "$ROOT/$sub" ]; then
        "$ROOT/$sub"
        fails=$((fails + $?))
    else
        echo "lint FAIL $sub (missing or not executable)"
        fails=$((fails + 1))
    fi
done

[ $fails -eq 0 ] && echo "rtl/lint.sh: all clean" || echo "rtl/lint.sh: $fails failing run(s)"
exit $fails
