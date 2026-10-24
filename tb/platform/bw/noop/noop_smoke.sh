#!/usr/bin/env bash
# tb/platform/bw/noop/noop_smoke.sh — sw/tools/tsr_bw's XRT path on XRT's own userspace with its no-op
# shim (XCL_EMULATION_MODE=noop, libxrt_noop: no card, no driver; docs/m0-bw.md §6): metadata-only
# xclbins of tsr_bw_k4 (mk_bw_xclbin.cpp) are opened with `tsr_bw --dry-run` — xclbin load, kernel
# lookup, argument lookup by name with the offset / size check against tsr_bw_regs.h, the DATA
# clock from the clock topology, one device-only buffer per memory group, every xrt::run argument
# set with its size, start / wait — and broken xclbins must be refused with a reason. The no-op
# shim has no register access and no DMA, so nothing past the dry run can run here.
#
#   tb/platform/bw/noop/noop_smoke.sh <tsr_bw binary> <XRT root> <scratch dir>
# Exit status: number of failed checks.
set -u
TSR_BW=$1
XRT=$(cd "$2" && pwd)
DIR=$3
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$DIR"
if [ ! -f "$XRT/lib/libxrt_noop.so" ] && ! ls "$XRT"/lib/libxrt_noop.so* >/dev/null 2>&1; then
    echo "noop_smoke: no libxrt_noop in $XRT/lib"
    exit 1
fi
# xclbin.h includes <uuid/uuid.h> (uuid-dev, an XRT dependency); without it, the transport's stand-in
# declarations (sw/transport/xrt-compat, docs/xrt-host.md §7) are enough for the struct definitions.
COMPAT=()
[ -f /usr/include/uuid/uuid.h ] || COMPAT=(-I"$HERE/../../../../sw/transport/xrt-compat")
g++ -std=c++17 -O1 -Wall -Wextra -I"$XRT/include" "${COMPAT[@]}" "$HERE/mk_bw_xclbin.cpp" -o "$DIR/mk_bw_xclbin" || exit 1
export XCL_EMULATION_MODE=noop XILINX_XRT="$XRT"
unset TSR_XCLBIN TSR_XRT_DEVICE
fails=0

expect() {   # expect NAME VARIANT EXIT_CODE PATTERN [tsr_bw args...]
    local name=$1 variant=$2 want=$3 pat=$4; shift 4
    "$DIR/mk_bw_xclbin" "$DIR/$variant.xclbin" "$variant" || { echo "FAIL $name: mk_bw_xclbin"; fails=$((fails + 1)); return; }
    local out rc
    out=$("$TSR_BW" --xclbin "$DIR/$variant.xclbin" --dry-run --sizes 16M "$@" 2>&1)
    rc=$?
    if [ "$rc" = "$want" ] && echo "$out" | grep -q -- "$pat"; then
        echo "ok   $name (exit $rc: $(echo "$out" | grep -m1 -- "$pat" | cut -c1-150))"
    else
        echo "FAIL $name: exit $rc (expected $want), pattern '$pat'"
        echo "$out" | sed 's/^/     /'
        fails=$((fails + 1))
    fi
}

expect "open + empty run"            good       0 "dry run: kernel opened, 4 buffers of 64M"  --ch-mib 64
expect "DATA clock from the xclbin"  good       0 "kernel clock 287.0 MHz (DATA_CLK)"         --ch-mib 64
expect "buffers per bank"            good       0 "ch3: DDR\[3\], 0.06 GiB device-only buffer" --ch-mib 64
expect "clock override"              good       0 "kernel clock 250.0 MHz (--clock-mhz)"      --ch-mib 64 --clock-mhz 250
expect "buffer larger than the bank" good       2 "smaller than the buffer"                   --ch-mib 512
expect "argument at a wrong offset"  bad_offset 2 "argument rd_off1 at 0x54"                  --ch-mib 64
expect "argument of a wrong size"    bad_size   2 "argument bytes2 at 0x68 (4 bytes)"         --ch-mib 64
expect "argument missing"            no_arg     2 "has no argument wr_off3"                   --ch-mib 64
expect "control range too small"     range      2 "s_axi_control range 0x800"                 --ch-mib 64
expect "two compute units"           two_cu     2 "has 2 compute units"                       --ch-mib 64
expect "unknown kernel name"         good       2 "has no kernel tsr_bw_k1"                   --ch-mib 64 --kernel tsr_bw_k1

# the no-op xclbin's argument table is the generated kernel_bw_k4.xml's (name, offset, size, port, type)
if python3 - "$HERE/../../../../rtl/platform/alveo_xrt/bw/kernel_bw_k4.xml" "$DIR/good.xclbin" <<'PY'
import re, sys
pat = re.compile(rb'<arg name="([^"]+)" addressQualifier="(\d)" id="(\d+)" port="([^"]+)" size="(0x[0-9a-fA-F]+)" offset="(0x[0-9a-fA-F]+)"[^>]*type="([^"]+)"')
def args(b):
    return sorted((m[0], int(m[1]), int(m[2]), m[3], int(m[4], 16), int(m[5], 16), m[6]) for m in pat.findall(b))
a, b = args(open(sys.argv[1], 'rb').read()), args(open(sys.argv[2], 'rb').read())
sys.exit(0 if a == b and len(a) == 21 else 1)
PY
then
    echo "ok   no-op xclbin arguments == kernel_bw_k4.xml (21 args)"
else
    echo "FAIL no-op xclbin arguments differ from kernel_bw_k4.xml"
    fails=$((fails + 1))
fi

[ $fails -eq 0 ] && echo "noop_smoke: all passed" || echo "noop_smoke: $fails failed"
exit $fails
