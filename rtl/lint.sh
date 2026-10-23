#!/usr/bin/env bash
# rtl/lint.sh — Verilator lint (-Wall, warnings are errors) of the RTL in rtl/filelist.f.
#
#   rtl/lint.sh            lint every module as its own top with default parameters (tsr_core:
#                          U200 v1 with the MVM unit), then tsr_core with HAS_MVM = 0 (M2) and
#                          HAS_MVM = 1 in reduced configurations (1 / 2 / 3 / 16 channels, small
#                          SPM, T_MAX 8, small K_MAX, narrow DRAM address) and edge
#                          implementation parameters (SPM_RD_LAT 1 / 4, FETCH_BEATS 2 / 64,
#                          RD_OUTSTANDING 1 / 2, MVM pipes 0 / 3, WS FIFO 64 / 1024, MVM
#                          latencies 1 / 8), then the M3 blocks with edge parameters, then the
#                          testbench-only HDL under tb/*/hdl (each file as top, with the other
#                          files of its directory)
#   rtl/lint.sh MODULE...  lint only these modules (default parameters)
#
# No blanket waivers: a module needing a lint_off must carry it inline, targeted, with a reason.
# Exit status: number of failing lint runs (0 = clean).
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

modules() {   # module names declared in the filelist's sources (packages excluded)
    grep -v '^\s*//' "$FL" | grep -v '^\s*$' | while read -r f; do
        sed -n 's/^\s*module\s\+\([A-Za-z_][A-Za-z0-9_]*\).*/\1/p' "$ROOT/rtl/$f"
    done
}

if [ $# -gt 0 ]; then
    for m in "$@"; do lint "$m"; done
    exit $fails
fi

for m in $(modules); do lint "$m"; done

# Reduced / edge configurations of the whole core. The default tsr_core (above) is U200 v1 with
# the MVM unit (HAS_MVM = 1, the bitstream configuration); M2 builds (HAS_MVM = 0) first.
lint tsr_core -GHAS_MVM=0
lint tsr_core -GHAS_MVM=0 -GNUM_CH=1 -GSPM_BYTES=65536 -GCH_BYTES="64'd1073741824" -GK_MAX=4096
lint tsr_core -GHAS_MVM=0 -GNUM_CH=3 -GT_MAX=8 -GSPM_BYTES=262144 -GADDR_BEAT_W=22 -GCH_BYTES="64'd4294967296"
lint tsr_core -GHAS_MVM=0 -GNUM_CH=16 -GT_MAX=8 -GK_MAX=65280 -GADDR_BEAT_W=36 -GCH_BYTES="64'd17179869184"
# Implementation parameters (tb/core test_core_impl runs these too).
lint tsr_core -GHAS_MVM=0 -GSPM_RD_LAT=4 -GFETCH_BEATS=64
lint tsr_core -GHAS_MVM=0 -GNUM_CH=1 -GSPM_BYTES=65536 -GSPM_RD_LAT=1 -GFETCH_BEATS=2 -GRD_OUTSTANDING=1
# HAS_MVM = 1 (docs/rtl-m3.md §1, integration step): U200 v1 with T_MAX 8, 1 / 3 / 16 channels,
# small K_MAX, pipe / FIFO / latency extremes (tb/core and tb/fastreplay run these configurations).
lint tsr_core -GHAS_MVM=1 -GT_MAX=8
lint tsr_core -GHAS_MVM=1 -GNUM_CH=1 -GT_MAX=2 -GSPM_BYTES=65536 -GCH_BYTES="64'd1073741824" -GK_MAX=4096 -GADDR_BEAT_W=22
lint tsr_core -GHAS_MVM=1 -GNUM_CH=3 -GT_MAX=3 -GSPM_BYTES=262144 -GADDR_BEAT_W=22 -GCH_BYTES="64'd4294967296"
lint tsr_core -GHAS_MVM=1 -GNUM_CH=16 -GT_MAX=2 -GK_MAX=8192 -GSPM_BYTES=262144 -GCH_BYTES="64'd1073741824"
lint tsr_core -GHAS_MVM=1 -GNUM_CH=16 -GT_MAX=8 -GK_MAX=65280 -GADDR_BEAT_W=36 -GCH_BYTES="64'd17179869184"
lint tsr_core -GHAS_MVM=1 -GNUM_CH=2 -GT_MAX=4 -GK_MAX=256 -GSPM_BYTES=4096 -GCH_BYTES="64'd4096" -GADDR_BEAT_W=5
lint tsr_core -GHAS_MVM=1 -GMVM_BC_PIPE=0 -GMVM_RES_PIPE=0 -GMVM_WS_FIFO=64 -GSPM_RD_LAT=1 -GRD_OUTSTANDING=1
lint tsr_core -GHAS_MVM=1 -GMVM_BC_PIPE=3 -GMVM_RES_PIPE=3 -GMVM_WS_FIFO=1024 -GSPM_RD_LAT=4 -GFETCH_BEATS=64
lint tsr_core -GHAS_MVM=1 -GT_MAX=3 -GMVM_ACT_RD_LAT=1 -GMVM_MAC_LAT=1 -GMVM_MUL_LAT=1 -GMVM_ADD_LAT=1 -GMVM_WS_RD_LAT=1
lint tsr_core -GHAS_MVM=1 -GNUM_CH=3 -GMVM_ACT_RD_LAT=4 -GMVM_MAC_LAT=8 -GMVM_MUL_LAT=8 -GMVM_ADD_LAT=8 -GMVM_WS_RD_LAT=4 -GRD_OUTSTANDING=2

# M3 MVM unit (docs/rtl-m3.md §1): every new module as its own top with edge parameters
# (NUM_CH 1 / 3 / 16, T_MAX 1 / 8, small and maximal K_MAX, latency / pipe depth extremes).
lint tsr_fp_add -GLAT=1
lint tsr_fp_add -GLAT=8
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
lint tsr_mvm -GNUM_CH=1 -GT_MAX=8 -GK_MAX=256 -GSPM_BYTES=4096 -GADDR_BEAT_W=5 -GBC_PIPE=0 -GRES_PIPE=0
lint tsr_mvm -GNUM_CH=3 -GT_MAX=3 -GK_MAX=4096 -GSPM_BYTES=262144 -GADDR_BEAT_W=22 -GBC_PIPE=3 -GRES_PIPE=3 -GSPM_RD_LAT=4
lint tsr_mvm -GNUM_CH=16 -GT_MAX=8 -GK_MAX=65280 -GADDR_BEAT_W=36 -GWS_FIFO_BEATS=1024

# Testbench-only HDL (block testbench toplevels); linted with the same rules. Each file is a top,
# together with the other files of its directory (wrappers may instantiate each other).
for d in "$ROOT"/tb/*/hdl; do
    [ -d "$d" ] || continue
    for f in "$d"/*.sv; do
        [ -e "$f" ] || continue
        top=$(sed -n 's/^\s*module\s\+\([A-Za-z_][A-Za-z0-9_]*\).*/\1/p' "$f" | head -n 1)
        if out=$("$VERILATOR" --lint-only -Wall --top-module "$top" -F "$FL" "$d"/*.sv 2>&1); then
            echo "lint ok   $top (tb)"
        else
            echo "lint FAIL $top (tb)"
            echo "$out" | sed 's/^/    /'
            fails=$((fails + 1))
        fi
    done
done

[ $fails -eq 0 ] && echo "rtl/lint.sh: all clean" || echo "rtl/lint.sh: $fails failing run(s)"
exit $fails
