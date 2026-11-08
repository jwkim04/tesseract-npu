#!/usr/bin/env bash
# rtl/tech/xilinx/lint.sh — Verilator lint (-Wall, warnings are errors) of the Xilinx target
# (rtl/filelist_xilinx.f: rtl/filelist.f with rtl/tech/generic replaced by rtl/tech/xilinx).
#
#   rtl/tech/xilinx/lint.sh            check that the filelist's core list equals rtl/filelist.f's,
#                                      lint the two wrappers with edge parameters (tsr_sram_1r1w:
#                                      URAM / BRAM / LUTRAM sizes, RD_LAT 1..4; tsr_mac_lane: N 32 /
#                                      5 / 2 / 1, T 1..8, every register allocation LAT 1..10 and
#                                      an extra output register), every module that instantiates
#                                      them, tsr_core with the xilinx filelist (U200, T_MAX 8,
#                                      the U200 bitstream implementation parameters, reduced
#                                      configurations), an XSim elaboration of the core (static
#                                      elaboration of several configurations) and the MVM unit
#                                      (xvlog / xelab of Vivado, skipped without it; a
#                                      variable with more than one procedural driver fails it), a
#                                      Vivado synthesis of tsr_issue, tsr_csr and the W4-B blocks
#                                      (synth_check.sh: fails on a multi-driven net), and the tb/tech
#                                      HDL with both targets
#   rtl/tech/xilinx/lint.sh MODULE...  lint only these modules (default parameters)
#
# No blanket waivers (as rtl/lint.sh). Exit status: number of failing lint runs (0 = clean).
set -u
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
FL="$ROOT/rtl/filelist_xilinx.f"
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

if [ $# -gt 0 ]; then
    for m in "$@"; do lint "$m"; done
    exit $fails
fi

# The xilinx filelist is rtl/filelist.f with the technology directory swapped (same core, order).
if diff <(grep -v '^\s*//' "$ROOT/rtl/filelist.f" | grep -v '^\s*$' | sed 's#tech/generic/#tech/xilinx/#') \
        <(grep -v '^\s*//' "$FL" | grep -v '^\s*$') > /dev/null; then
    echo "filelist ok   rtl/filelist_xilinx.f == rtl/filelist.f with tech/xilinx"
else
    echo "filelist FAIL rtl/filelist_xilinx.f differs from rtl/filelist.f beyond tech/generic -> tech/xilinx"
    fails=$((fails + 1))
fi

# ---- the wrappers ----
lint tsr_sram_1r1w
lint tsr_sram_1r1w -GDEPTH=16384 -GWIDTH=512 -GRD_LAT=2
lint tsr_sram_1r1w -GDEPTH=16384 -GWIDTH=512 -GRD_LAT=1
lint tsr_sram_1r1w -GDEPTH=4096 -GWIDTH=8 -GRD_LAT=3
lint tsr_sram_1r1w -GDEPTH=512 -GWIDTH=512 -GRD_LAT=4
lint tsr_sram_1r1w -GDEPTH=64 -GWIDTH=512 -GRD_LAT=1
lint tsr_sram_1r1w -GDEPTH=72 -GWIDTH=512 -GRD_LAT=2
lint tsr_sram_1r1w -GDEPTH=32 -GWIDTH=512 -GRD_LAT=3
lint tsr_sram_1r1w -GDEPTH=2 -GWIDTH=512 -GRD_LAT=1
lint tsr_sram_1r1w -GDEPTH=1 -GWIDTH=72 -GRD_LAT=2
for t in 1 2 3 8; do
    for l in 1 2 3 4 5 6 7 8 9 10 11; do lint tsr_mac_lane -GT=$t -GLAT=$l; done
done
# PV mode (ATTPV, AT §8.2; needs LAT >= 3)
for t in 1 2 8; do
    for l in 3 4 5 6 7 8 9 10 11; do lint tsr_mac_lane -GT=$t -GLAT=$l -GPV=1; done
done
lint tsr_mac_lane -GN=5 -GT=3 -GLAT=4 -GPV=1
lint tsr_mac_lane -GN=1 -GT=2 -GLAT=5 -GPV=1
lint tsr_mac_lane -GN=5 -GT=3 -GLAT=4
lint tsr_mac_lane -GN=2 -GT=2 -GLAT=5
lint tsr_mac_lane -GN=1 -GT=1 -GLAT=2
lint tsr_mac_lane -GN=1 -GT=5 -GLAT=7

# ---- every user of the wrappers (rtl/lint.sh edge parameters) ----
lint tsr_spm
lint tsr_spm -GRD_LAT=4
lint tsr_spm -GSPM_BYTES=262144 -GRD_LAT=1
lint tsr_spm -GSPM_BYTES=4096 -GRD_LAT=3
lint tsr_qact
lint tsr_qact -GSPM_RD_LAT=4
lint tsr_mvm_ws -GFIFO_BEATS=64 -GADDR_BEAT_W=22
lint tsr_mvm_ws -GFIFO_BEATS=1024 -GADDR_BEAT_W=40 -GRD_LAT=4
lint tsr_mvm_actbuf -GK_MAX=256 -GT_MAX=8 -GRD_LAT=1
lint tsr_mvm_actbuf -GK_MAX=65280 -GT_MAX=3 -GRD_LAT=3
lint tsr_mvm_back -GK_MAX=256 -GT_MAX=8 -GACT_RD_LAT=1 -GMAC_LAT=1 -GMUL_LAT=1 -GADD_LAT=1
lint tsr_mvm_back -GK_MAX=65280 -GT_MAX=2 -GACT_RD_LAT=3 -GMAC_LAT=6 -GMUL_LAT=5 -GADD_LAT=8
lint tsr_mvm_back -GT_MAX=7 -GMAC_LAT=5
lint tsr_mvm_cluster -GCH=0 -GT_MAX=1 -GMAC_LAT=5
lint tsr_mvm_cluster -GCH=3 -GT_MAX=8 -GMAC_LAT=5
lint tsr_mvm_cluster -GCH=15 -GK_MAX=4096 -GT_MAX=3 -GADDR_BEAT_W=40 -GADD_LAT=8

# ---- whole core on the xilinx target ----
lint tsr_core
lint tsr_core -GHAS_MVM=0
lint tsr_core -GHAS_DB=0
lint tsr_core -GHAS_ATT=0
lint tsr_core -GHAS_VU=0 -GHAS_ATT=0 -GHAS_DB=0
lint tsr_core -GNUM_CH=1 -GT_MAX=1 -GRST_PIPE=3 -GSPM_RD_LAT=4 -GMVM_BC_PIPE=2 -GMVM_RES_PIPE=2 -GMVM_MAC_LAT=9 -GVU_EXP_LANES=8 -GVU_BACK_LANES=4
lint tsr_core -GNUM_CH=4 -GT_MAX=8 -GRST_PIPE=3 -GSPM_RD_LAT=4 -GMVM_BC_PIPE=2 -GMVM_RES_PIPE=2 -GMVM_MAC_LAT=9 -GVU_LANE_PIPE=2 -GVU_FRONT_LANES=8 -GVU_EXP_LANES=8 -GVU_BACK_LANES=4
lint tsr_core -GNUM_CH=4 -GT_MAX=8 -GRST_PIPE=3 -GSPM_RD_LAT=4 -GMVM_BC_PIPE=2 -GMVM_RES_PIPE=2 -GMVM_MAC_LAT=9 -GRV_LOCAL_TRANSPORT=1 -GHAS_VU=0 -GHAS_ATT=0 -GHAS_TRW=0 -GHIST_DEPTH=0
lint tsr_core -GNUM_CH=4 -GT_MAX=4 -GRST_PIPE=3 -GSPM_RD_LAT=4 -GMVM_BC_PIPE=2 -GMVM_RES_PIPE=2 -GMVM_MAC_LAT=9 -GMVM_GATHER_PAIR=1 -GMVM_GATHER_FLITS=2 -GDDR_LOCAL_PIPE=1 -GRV_LOCAL_TRANSPORT=1 -GXSLR_REG=1 -GVU_LANE_PIPE=2 -GVU_FRONT_LANES=8 -GVU_EXP_LANES=8 -GVU_BACK_LANES=4
lint tsr_core -GHAS_TOPK=1 -GHAS_DQROW=1
lint tsr_core -GNUM_CH=4 -GT_MAX=4 -GRST_PIPE=3 -GSPM_RD_LAT=4 -GMVM_BC_PIPE=2 -GMVM_RES_PIPE=2 -GMVM_MAC_LAT=9 -GMVM_GATHER_PAIR=1 -GMVM_GATHER_FLITS=2 -GDDR_LOCAL_PIPE=1 -GRV_LOCAL_TRANSPORT=1 -GXSLR_REG=1 -GVU_LANE_PIPE=2 -GVU_FRONT_LANES=8 -GVU_EXP_LANES=8 -GVU_BACK_LANES=4 -GHAS_GDN=1
lint tsr_core -GHAS_MVM=1 -GT_MAX=8
lint tsr_core -GHAS_MVM=1 -GSPM_RD_LAT=4 -GMVM_BC_PIPE=2 -GMVM_RES_PIPE=2 -GMVM_MAC_LAT=5
lint tsr_core -GHAS_MVM=1 -GT_MAX=8 -GSPM_RD_LAT=4 -GMVM_BC_PIPE=2 -GMVM_RES_PIPE=2 -GMVM_MAC_LAT=5
lint tsr_core -GHAS_MVM=1 -GNUM_CH=1 -GT_MAX=2 -GSPM_BYTES=65536 -GCH_BYTES="64'd1073741824" -GK_MAX=4096 -GADDR_BEAT_W=22
lint tsr_core -GHAS_MVM=1 -GNUM_CH=2 -GT_MAX=4 -GK_MAX=256 -GSPM_BYTES=4096 -GCH_BYTES="64'd4096" -GADDR_BEAT_W=5
lint tsr_core -GHAS_MVM=1 -GMVM_BC_PIPE=0 -GMVM_RES_PIPE=0 -GMVM_WS_FIFO=64 -GSPM_RD_LAT=1 -GRD_OUTSTANDING=1
lint tsr_core -GHAS_MVM=1 -GHAS_ATT=0 -GT_MAX=3 -GMVM_ACT_RD_LAT=1 -GMVM_MAC_LAT=1 -GMVM_MUL_LAT=1 -GMVM_ADD_LAT=1 -GMVM_WS_RD_LAT=1

# ---- XSim (the Vivado simulator, used by hw_emu): elaboration of the core at the U200 bitstream
# parameters. Verilator and Vivado synthesis accept constructs XSim rejects (e.g. an initial block
# writing an always_ff variable, IEEE 1800 9.2.2.4). Needs Vivado (TSR_VIVADO_SETTINGS, default
# /tool/Xilinx/Vivado/2024.2/settings64.sh); skipped without it. Every Vivado call runs under TSR_VIVADO_WRAP (a command prefix, e.g. a slot
# script that limits the Vivado processes of a shared host) or, when that is unset, flock $TSR_VIVADO_LOCK.
VSET=${TSR_VIVADO_SETTINGS:-/tool/Xilinx/Vivado/2024.2/settings64.sh}
if [ -r "$VSET" ]; then
    xdir=$(mktemp -d)
    vlock=()
    if [ -n "${TSR_VIVADO_WRAP:-}" ]; then read -r -a vlock <<< "$TSR_VIVADO_WRAP"
    elif [ -n "${TSR_VIVADO_LOCK:-}" ]; then vlock=(flock "$TSR_VIVADO_LOCK"); fi
    # xvlog -f list: the filelist with its nested lists expanded, +incdir+ / +define+ as -i / -d,
    # Verilator configuration files (.vlt) left out (W4-A: rtl/ctrl/filelist.f)
    fl_expand() {   # fl_expand LIST
        local fl=$1 dir line
        dir=$(cd "$(dirname "$fl")" && pwd)
        while IFS= read -r line || [ -n "$line" ]; do
            line=${line%%//*}
            line="${line#"${line%%[![:space:]]*}"}"
            line="${line%"${line##*[![:space:]]}"}"
            [ -z "$line" ] && continue
            case "$line" in
                "-F "*|"-f "*) fl_expand "$dir/${line:3}" ;;
                +incdir+*) echo "-i $(cd "$dir/${line#+incdir+}" && pwd)" ;;
                +define+*) echo "-d ${line#+define+}" ;;
                *.vlt) ;;
                *) echo "$dir/$line" ;;
            esac
        done < "$fl"
    }
    # Two xvlog passes into one library: the vendored IP (third_party/) with SYNTHESIS defined
    # (XSim 2024.2 cannot compile Ibex's simulation-only DPI exports inside a generate block,
    # ibex_if_stage.sv, nor does it need the vendor assertions), then Tesseract's own sources.
    fl_expand "$FL" > "$xdir/all.txt"
    { grep '^-' "$xdir/all.txt"; echo "-d SYNTHESIS"; grep -v '^-' "$xdir/all.txt" | grep '/third_party/'; } \
        > "$xdir/vendor.txt"
    { grep '^-' "$xdir/all.txt"; grep -v '^-' "$xdir/all.txt" | grep -v '/third_party/'; } > "$xdir/files.txt"
    # Warnings that are functional bugs in the synthesised netlist, not style: a variable written by
    # more than one procedural block (VRFC 10-9892 / 10-9941 / 10-10006). Verilator accepts it, but
    # Vivado keeps one driver and ignores the other (Synth 8-6858): W4-A review, tsr_chk e_q.
    xsim_bad='WARNING: \[VRFC 10-(9892|9941|10006)\]'
    xsim_elab() {   # xsim_elab TOP NAME=VALUE...
        local top=$1; shift
        local g=() kv out
        for kv in "$@"; do g+=(-generic_top "$kv"); done
        if out=$(cd "$xdir" && set +u && source "$VSET" > /dev/null 2>&1 && \
                 "${vlock[@]}" xelab "$top" -s "x$RANDOM" "${g[@]}" 2>&1); then
            if echo "$out" | grep -qE "$xsim_bad"; then
                echo "xsim FAIL $top $* (a variable with more than one procedural driver)"
                echo "$out" | grep -E "$xsim_bad" | sed 's/^/    /'
                fails=$((fails + 1))
            else
                echo "xsim ok   $top $*"
            fi
        else
            echo "xsim FAIL $top $*"
            echo "$out" | grep -E '^(ERROR|CRITICAL)' | sed 's/^/    /'
            fails=$((fails + 1))
        fi
    }
    # The whole core: static elaboration only (xelab prints "Completed static elaboration" within seconds with --O0;
    # the code generation behind it takes over 20 minutes for the core with the VU, attention and the DB block
    # and checks nothing more here), then xelab is stopped.
    xsim_static() {   # xsim_static TOP NAME=VALUE...
        local top=$1; shift
        local g=() kv log pid t=0 st=1
        for kv in "$@"; do g+=(-generic_top "$kv"); done
        log="$xdir/static_$RANDOM.log"
        setsid bash -c 'cd "$1" && shift && set +u && source "$1" > /dev/null 2>&1 && shift && exec "$@"' _ "$xdir" "$VSET" \
            "${vlock[@]}" xelab "$top" -s "x$RANDOM" --O0 "${g[@]}" > "$log" 2>&1 &
        pid=$!
        while kill -0 "$pid" 2>/dev/null && [ $t -lt 3600 ]; do
            if grep -q 'Completed static elaboration' "$log"; then st=0; break; fi
            grep -qE '^(ERROR|CRITICAL)' "$log" && break
            sleep 1; t=$((t + 1))
        done
        kill -- "-$pid" 2>/dev/null; wait "$pid" 2>/dev/null
        if [ $st -eq 0 ] && ! grep -qE "$xsim_bad" "$log"; then
            echo "xsim ok   $top $* (static elaboration)"
        else
            echo "xsim FAIL $top $*"
            grep -E "^(ERROR|CRITICAL)|$xsim_bad" "$log" | head -10 | sed 's/^/    /'
            [ $st -ne 0 ] && [ $t -ge 3600 ] && echo "    static elaboration did not finish in an hour"
            fails=$((fails + 1))
        fi
    }
    if out=$(cd "$xdir" && set +u && source "$VSET" > /dev/null 2>&1 && \
             "${vlock[@]}" xvlog -sv -f vendor.txt 2>&1 && "${vlock[@]}" xvlog -sv -f files.txt 2>&1); then
        u200=(HAS_MVM=1 SPM_RD_LAT=4 MVM_BC_PIPE=2 MVM_RES_PIPE=2 MVM_MAC_LAT=9 RST_PIPE=3)
        xsim_static tsr_core "${u200[@]}" NUM_CH=1 T_MAX=1 VU_EXP_LANES=8 VU_BACK_LANES=4
        xsim_static tsr_core "${u200[@]}" NUM_CH=4 T_MAX=8 VU_LANE_PIPE=2
        xsim_static tsr_core "${u200[@]}" NUM_CH=4 T_MAX=8 RV_LOCAL_TRANSPORT=1 HAS_VU=0 HAS_ATT=0 HAS_TRW=0 HIST_DEPTH=0
        xsim_static tsr_core "${u200[@]}" NUM_CH=4 T_MAX=4 MVM_GATHER_PAIR=1 MVM_GATHER_FLITS=2 DDR_LOCAL_PIPE=1 RV_LOCAL_TRANSPORT=1 XSLR_REG=1 VU_FRONT_LANES=8 VU_EXP_LANES=8 VU_BACK_LANES=4 VU_LANE_PIPE=2
        xsim_static tsr_core "${u200[@]}" NUM_CH=2 T_MAX=2 K_MAX=4096 SPM_BYTES=65536 ADDR_BEAT_W=22 CH_BYTES=1073741824
        xsim_static tsr_core HAS_MVM=0 RST_PIPE=1
        xsim_static tsr_core "${u200[@]}" NUM_CH=1 T_MAX=1 HAS_DB=0
        xsim_static tsr_core "${u200[@]}" NUM_CH=1 T_MAX=1 HAS_ATT=0
        xsim_static tsr_core "${u200[@]}" NUM_CH=1 T_MAX=1 HAS_VU=0 HAS_ATT=0 HAS_DB=0
        xsim_static tsr_core "${u200[@]}" NUM_CH=2 T_MAX=2 HAS_TOPK=1 HAS_DQROW=1
        xsim_elab tsr_mvm NUM_CH=2 T_MAX=2 RST_LOCAL=2
        xsim_elab tsr_mvm NUM_CH=2 T_MAX=2 RST_LOCAL=2 XSLR_REG=1
        xsim_elab tsr_rv_spm_tx XSLR_REG=1
    else
        echo "xsim FAIL xvlog rtl/filelist_xilinx.f"
        echo "$out" | grep -E '^(ERROR|CRITICAL)' | sed 's/^/    /'
        fails=$((fails + 1))
    fi
    rm -rf "$xdir"
else
    echo "xsim skip (no Vivado at $VSET)"
fi

# ---- Vivado synthesis of the issue path, the CSR block and the W4-B blocks: no multi-driven net (synth_check.sh; skipped
# without Vivado). Verilator and XSim accept a variable written by two blocks, Vivado keeps one driver.
# W4-B: the DB block, the VU / AT checker classes and KVST as well
# (found by the first whole-core synthesis, tb/core/viv/core_ooc.sh k1: tsr_dma_kvst wrote five registers from two
# blocks); the GDN unit (VU §7.12) as well
"$ROOT/rtl/tech/xilinx/synth_check.sh" tsr_issue tsr_csr tsr_dbg tsr_chk_at tsr_vu_chk "tsr_dma:HAS_KVST=1" \
    "tsr_vu:HAS_TOPK=1,HAS_DQROW=1" "tsr_mvm:HAS_ATT=1,NUM_CH=2,T_MAX=2" tsr_vu_gdn "tsr_vu:HAS_GDN=1,FRONT_LANES=8,EXP_LANES=8,BACK_LANES=4,LANE_PIPE=2"
fails=$((fails + $?))

# ---- tb/tech HDL with both targets (each toplevel with the tech files of one target only) ----
for t in generic xilinx; do
    for f in "$ROOT"/tb/tech/hdl/*.sv; do
        [ -e "$f" ] || continue
        top=$(sed -n 's/^\s*module\s\+\([A-Za-z_][A-Za-z0-9_]*\).*/\1/p' "$f" | head -n 1)
        if out=$("$VERILATOR" --lint-only -Wall --top-module "$top" "$ROOT"/rtl/tech/$t/*.sv "$f" 2>&1); then
            echo "lint ok   $top (tb, $t)"
        else
            echo "lint FAIL $top (tb, $t)"
            echo "$out" | sed 's/^/    /'
            fails=$((fails + 1))
        fi
    done
done

[ $fails -eq 0 ] && echo "rtl/tech/xilinx/lint.sh: all clean" || echo "rtl/tech/xilinx/lint.sh: $fails failing run(s)"
exit $fails
