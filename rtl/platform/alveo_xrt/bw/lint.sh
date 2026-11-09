#!/usr/bin/env bash
# rtl/platform/alveo_xrt/bw/lint.sh — Verilator lint (-Wall, warnings are errors) of the bandwidth
# kernel (docs/bandwidth-kernel.md §5) and the generated-file check of gen_bw.py. Meant to be called from
# rtl/lint.sh (proposal) or on its own.
#
#   rtl/platform/alveo_xrt/bw/lint.sh   gen_bw.py --check; tsr_bw_engine / tsr_bw_ctrl / tsr_bw_kernel
#                                       with default and edge parameters; the packaged top tsr_bw_k4;
#                                       the testbench wrapper tb/platform/bw/sv/kwrap_tsr_bw_k4.sv;
#                                       the `core` m_axi hints of package_bw_xo.tcl equal the
#                                       Tesseract kernels' (../package_xo.tcl)
# Sources: rtl/platform/alveo_xrt/bw/filelist.f (complete: tsr_pkg, tsr_rslice, the adapter's
# reset / slices / bridge, the bandwidth kernel). Exit status: number of failing runs (0 = clean).
set -u
ROOT=$(cd "$(dirname "$0")/../../../.." && pwd)
FL="$ROOT/rtl/platform/alveo_xrt/bw/filelist.f"
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

if python3 "$ROOT/rtl/platform/alveo_xrt/bw/gen_bw.py" --check; then
    echo "gen ok    gen_bw.py --check (tsr_bw_k4.v, kernel_bw_k4.xml, kwrap_tsr_bw_k4.sv)"
else
    fails=$((fails + 1))
fi

# The bandwidth kernel measures the memory path the core gets: same m_axi interconnect hints as the Tesseract kernels
core_hints=$(grep -o '{MAX_BURST_LENGTH [0-9]* NUM_READ_OUTSTANDING [0-9]* NUM_WRITE_OUTSTANDING [0-9]*}' \
             "$ROOT/rtl/platform/alveo_xrt/package_xo.tcl")
bw_hints=$(grep -o '^ *core {MAX_BURST_LENGTH [0-9]* NUM_READ_OUTSTANDING [0-9]* NUM_WRITE_OUTSTANDING [0-9]*}' \
           "$ROOT/rtl/platform/alveo_xrt/bw/package_bw_xo.tcl" | sed 's/^ *core //')
if [ -n "$core_hints" ] && [ "$core_hints" = "$bw_hints" ]; then
    echo "hints ok  package_bw_xo.tcl core = package_xo.tcl $core_hints"
else
    echo "hints FAIL package_bw_xo.tcl core '$bw_hints' != package_xo.tcl '$core_hints'"
    fails=$((fails + 1))
fi

lint tsr_bw_engine
lint tsr_bw_engine -GCH=3 -GADDR_BEAT_W=7 -GCNT_W=33
lint tsr_bw_engine -GCH=1 -GADDR_BEAT_W=52 -GCNT_W=64
lint tsr_bw_ctrl
lint tsr_bw_ctrl -GNUM_CH=1
lint tsr_bw_ctrl -GNUM_CH=3 -GADDR_BEAT_W=20 -GAXI_SLICES=1
lint tsr_bw_kernel
lint tsr_bw_kernel -GNUM_CH=1 -GAXI_SLICES=1 -GCTRL_PIPE=1
lint tsr_bw_kernel -GNUM_CH=3 -GADDR_BEAT_W=20 -GAXI_SLICES=3 -GCTRL_PIPE=4
lint tsr_bw_kernel -GNUM_CH=2 -GADDR_BEAT_W=7 -GAXI_SLICES=4 -GCTRL_PIPE=1
lint tsr_bw_k4
lint kwrap_tsr_bw_k4 "$ROOT/tb/platform/bw/sv/kwrap_tsr_bw_k4.sv"

[ $fails -eq 0 ] && echo "rtl/platform/alveo_xrt/bw/lint.sh: all clean" || echo "rtl/platform/alveo_xrt/bw/lint.sh: $fails failing run(s)"
exit $fails
