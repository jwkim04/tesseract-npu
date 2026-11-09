#!/usr/bin/env bash
# rtl/tech/xilinx/synth_check.sh — Vivado out-of-context synthesis of RTL blocks as a gate for
# netlist bugs that Verilator and XSim simulate correctly (opt-in unless run by rtl/tech/xilinx/lint.sh;
# needs Vivado 2024.2, batch mode only).
#
#   rtl/tech/xilinx/synth_check.sh [-o OUTDIR] [TOP[:NAME=VALUE,...]...]
#     TOP         module to synthesise (default: tsr_issue tsr_csr, i.e. the issue path with the
#                 checker, and the host CSR block, at the parameter defaults = U200 values)
#     NAME=VALUE  synth_design -generic of that TOP (comma separated)
#     -o OUTDIR   keep the logs / reports there (default: a temporary directory, removed)
#
# Runs tb/tech/viv/tech_ooc.tcl <repo> TOP OUT synth_only on rtl/filelist_xilinx.f for each TOP and
# fails (exit status = number of failing tops) when Vivado exits with an error or its log has a
# CRITICAL WARNING / ERROR about a multi-driven net ([Synth 8-6858], [Synth 8-6859], [Synth 8-3352]: a
# variable written by two procedural blocks, where Vivado keeps one driver and ignores the other;
# found in tsr_chk's e_q, which made the synthesised range checks wrong). The
# netlist-versus-simulator regression for the checker is tb/chk/viv/net_chk.sh.
# Environment: TSR_VIVADO_SETTINGS (default /tool/Xilinx/Vivado/2024.2/settings64.sh);
# TSR_VIVADO_WRAP = a command prefix for every Vivado call (e.g. a slot script limiting the Vivado
# processes of a shared host), else TSR_VIVADO_LOCK = a flock file that serialises Vivado runs; TSR_SYNTH_FILELIST =
# another filelist than rtl/filelist_xilinx.f (to see that a mutated source fails this gate).
# Larger runs, opt-in: tsr_core (the whole core, U200 parameters; 8 min, 12 GB) through
# tb/core/viv/core_ooc.sh k4 OUT synth_only, whose log is checked the same way.
set -u
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
VSET=${TSR_VIVADO_SETTINGS:-/tool/Xilinx/Vivado/2024.2/settings64.sh}
outdir=""
if [ "${1:-}" = "-o" ]; then outdir=$2; shift 2; fi
[ $# -eq 0 ] && set -- tsr_issue tsr_csr
if [ ! -r "$VSET" ]; then
    echo "synth skip (no Vivado at $VSET)"
    exit 0
fi
tmp=""
if [ -z "$outdir" ]; then tmp=$(mktemp -d); outdir=$tmp; fi
mkdir -p "$outdir"
set +u
# shellcheck disable=SC1090
source "$VSET" > /dev/null 2>&1
set -u
lock=()
if [ -n "${TSR_VIVADO_WRAP:-}" ]; then read -r -a lock <<< "$TSR_VIVADO_WRAP"
elif [ -n "${TSR_VIVADO_LOCK:-}" ]; then lock=(flock "$TSR_VIVADO_LOCK"); fi

bad='(CRITICAL WARNING|ERROR): .*(\[Synth 8-(6858|6859|3352)\]|multi-driven)'
fails=0
for spec in "$@"; do
    top=${spec%%:*}
    gens=()
    if [ "$top" != "$spec" ]; then IFS=, read -r -a gens <<< "${spec#*:}"; fi
    d="$outdir/$top"
    mkdir -p "$d"
    if (cd "$d" && "${lock[@]}" vivado -mode batch -nojournal -nolog -source "$ROOT/tb/tech/viv/tech_ooc.tcl" \
            -tclargs "$ROOT" "$top" "$d/out" synth_only ${TSR_SYNTH_FILELIST:+"filelist=$TSR_SYNTH_FILELIST"} \
            "${gens[@]}" > "$d/vivado.log" 2>&1) \
       && ! grep -qE '^ERROR' "$d/vivado.log"; then
        if grep -qE "$bad" "$d/vivado.log"; then
            echo "synth FAIL $spec: multi-driven nets ($(grep -cE "$bad" "$d/vivado.log") messages)"
            grep -E "$bad" "$d/vivado.log" | sort | uniq -c | sort -rn | head -5 | sed 's/^/    /'
            fails=$((fails + 1))
        else
            echo "synth ok   $spec $(grep -m1 'TOOC util(synth)' "$d/vivado.log" | sed 's/TOOC util(synth) //')"
        fi
    else
        echo "synth FAIL $spec: Vivado error"
        grep -E '^(ERROR|CRITICAL WARNING)' "$d/vivado.log" | head -5 | sed 's/^/    /'
        fails=$((fails + 1))
    fi
done
[ -n "$tmp" ] && rm -rf "$tmp"
[ $fails -eq 0 ] && echo "rtl/tech/xilinx/synth_check.sh: all clean" || echo "rtl/tech/xilinx/synth_check.sh: $fails failing top(s)"
exit $fails
