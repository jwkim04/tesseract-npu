#!/usr/bin/env bash
# hw/u200/bw/build_bw_xclbin.sh — build the M0 bandwidth xclbin (docs/m0-bw.md §4), the pattern of
# hw/u200/build_xclbin.sh for the kernel tsr_bw_k4.
#
#   hw/u200/bw/build_bw_xclbin.sh [hw|hw_emu]        (default target: hw)
#
# 1. the kernel object build/u200/xo/tsr_bw_k4.xo (build/u200/xo/wide/ for TSR_BW_HINTS=wide):
#    packaged now by rtl/platform/alveo_xrt/bw/package_bw_xo.tcl (Vivado batch) unless it exists and
#    its source manifest tsr_bw_k4.sources.sha256 next to it (written by package_bw_xo.tcl) still
#    matches the working tree (TSR_REPACKAGE=1 forces it)
# 2. v++ --link -t <target> --platform $TSR_U200_PLATFORM --config hw/u200/bw/bw.cfg
#    -> build/u200/bw[_wide][_hw_emu]/tsr_bw_k4.xclbin (+ logs, reports)
# Then: TSR_XCLBIN=build/u200/bw/tsr_bw_k4.xclbin build/<xrt build>/bin/tsr_bw (docs/m0-bw.md §6).
#
# Environment:
#   TSR_U200_PLATFORM   platform name or .xpfm path (default xilinx_u200_gen3x16_xdma_2_202110_1),
#                       searched in $PLATFORM_REPO_PATHS (colon list), /opt/xilinx/platforms.
#                       PLATFORM_REPO_PATHS unset: build/platforms/root/opt/xilinx/platforms (the
#                       development platform extracted on the dev server) when that directory exists
#   TSR_XILINX          Xilinx install root (default /tool/Xilinx); version 2024.2
#   TSR_VIVADO_LOCK     if set: every Vivado / v++ call runs under `flock $TSR_VIVADO_LOCK`
#   TSR_BW_FLOORPLAN    1 = apply the draft per-SLR floorplan hw/u200/bw/bw_floorplan.tcl
#   TSR_BW_HINTS        m_axi interconnect hints of the .xo (package_bw_xo.tcl): core (default; the
#                       Tesseract kernels' 64-beat bursts, 16 reads / 32 writes outstanding: the M0
#                       baseline) or wide (64 / 64: the engine's maximum, for comparison only)
# Stops with exit status 3 and a message when the platform (or the Xilinx tools) are missing.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
TARGET=${1:-hw}
case "$TARGET" in
  hw|hw_emu) ;;
  *) echo "usage: hw/u200/bw/build_bw_xclbin.sh [hw|hw_emu]" >&2; exit 2 ;;
esac
KERNEL=tsr_bw_k4
PLATFORM=${TSR_U200_PLATFORM:-xilinx_u200_gen3x16_xdma_2_202110_1}
XILINX=${TSR_XILINX:-/tool/Xilinx}
VIVADO_SETTINGS=$XILINX/Vivado/2024.2/settings64.sh
VITIS_SETTINGS=$XILINX/Vitis/2024.2/settings64.sh
BWDIR=$ROOT/rtl/platform/alveo_xrt/bw
HINTS=${TSR_BW_HINTS:-core}
case "$HINTS" in
  core) XO_DIR=$ROOT/build/u200/xo; VARIANT= ;;
  wide) XO_DIR=$ROOT/build/u200/xo/wide; VARIANT=_wide ;;
  *) echo "build_bw_xclbin.sh: TSR_BW_HINTS must be core or wide" >&2; exit 2 ;;
esac
XO=$XO_DIR/$KERNEL.xo
OUT=$ROOT/build/u200/bw$VARIANT$([ "$TARGET" = hw ] || echo "_$TARGET")
CFG=$ROOT/hw/u200/bw/bw.cfg

locked() {   # locked CMD...: run under the shared Vivado lock when TSR_VIVADO_LOCK is set
  if [ -n "${TSR_VIVADO_LOCK:-}" ]; then flock "$TSR_VIVADO_LOCK" "$@"; else "$@"; fi
}

# ---- platform: name -> .xpfm ----
DEV_PLATFORMS=$ROOT/build/platforms/root/opt/xilinx/platforms
if [ -z "${PLATFORM_REPO_PATHS+x}" ] && [ -d "$DEV_PLATFORMS" ]; then
  export PLATFORM_REPO_PATHS=$DEV_PLATFORMS
  echo "== PLATFORM_REPO_PATHS unset: using $DEV_PLATFORMS"
fi
find_platform() {
  local p=$1
  if [ -f "$p" ]; then echo "$p"; return 0; fi
  local dirs=${PLATFORM_REPO_PATHS:-}
  dirs="$dirs:/opt/xilinx/platforms"
  local IFS=:
  for d in $dirs; do
    [ -n "$d" ] || continue
    for cand in "$d/$p/$p.xpfm" "$d/$p.xpfm"; do
      if [ -f "$cand" ]; then echo "$cand"; return 0; fi
    done
  done
  return 1
}
if ! XPFM=$(find_platform "$PLATFORM"); then
  cat >&2 <<EOM
build_bw_xclbin.sh: the Vitis platform $PLATFORM is not installed.
  Looked for $PLATFORM.xpfm in \$PLATFORM_REPO_PATHS (${PLATFORM_REPO_PATHS:-unset}) and /opt/xilinx/platforms.
  Install the U200 platform packages (xilinx-u200-gen3x16-xdma-2-202110 deployment and development
  packages, with XRT) or point TSR_U200_PLATFORM at the .xpfm, then run this script again.
  Nothing was built ($KERNEL.xo is packaged on its own by rtl/platform/alveo_xrt/bw/package_bw_xo.tcl).
EOM
  exit 3
fi
for f in "$VIVADO_SETTINGS" "$VITIS_SETTINGS"; do
  if [ ! -f "$f" ]; then
    echo "build_bw_xclbin.sh: $f not found (set TSR_XILINX to the Xilinx 2024.2 install root)" >&2
    exit 3
  fi
done
mem_kb=$(awk '/MemTotal/ {print $2}' /proc/meminfo)
if [ "$mem_kb" -lt $((30 * 1024 * 1024)) ]; then
  echo "build_bw_xclbin.sh: warning: $((mem_kb / 1024 / 1024)) GiB RAM; a U200 implementation may need 32 GiB or more" >&2
fi

# ---- 1. kernel object ----
need_pkg=${TSR_REPACKAGE:-0}
MANIFEST=$XO_DIR/$KERNEL.sources.sha256
if [ ! -f "$XO" ] || [ ! -f "$MANIFEST" ] || ! grep -q "^PKG $KERNEL hints $HINTS " "$XO_DIR/$KERNEL.package.log" 2>/dev/null; then
  need_pkg=1
elif ! (cd "$ROOT" && sha256sum --quiet --strict -c "$MANIFEST" > /dev/null 2>&1); then
  echo "== $KERNEL.xo is stale (sources changed since packaging: cd $ROOT && sha256sum -c $MANIFEST)"
  need_pkg=1
fi
if [ "$need_pkg" = 1 ]; then
  echo "== packaging $KERNEL ($HINTS m_axi hints) -> $XO"
  mkdir -p "$XO_DIR"
  locked bash -c "source '$VIVADO_SETTINGS' >/dev/null && cd '$XO_DIR' && vivado -mode batch -nojournal -nolog \
      -source '$BWDIR/package_bw_xo.tcl' -tclargs '$ROOT' '$XO_DIR' '$HINTS'" \
      > "$XO_DIR/$KERNEL.package.log" 2>&1 || { tail -30 "$XO_DIR/$KERNEL.package.log"; exit 1; }
  grep '^PKG' "$XO_DIR/$KERNEL.package.log"
fi

# ---- 2. link ----
mkdir -p "$OUT"
extra=()
if [ "${TSR_BW_FLOORPLAN:-0}" = 1 ]; then
  extra+=(--vivado.prop "run.impl_1.STEPS.OPT_DESIGN.TCL.PRE=$ROOT/hw/u200/bw/bw_floorplan.tcl")
fi
echo "== v++ link $KERNEL ($TARGET, $XPFM) -> $OUT"
locked bash -c "source '$VITIS_SETTINGS' >/dev/null && cd '$OUT' && v++ --link -t '$TARGET' --platform '$XPFM' \
    --config '$CFG' --save-temps --temp_dir '$OUT/tmp' --log_dir '$OUT/logs' --report_dir '$OUT/reports' \
    ${extra[*]:+${extra[*]}} -o '$OUT/$KERNEL.xclbin' '$XO'" 2>&1 | tee "$OUT/v++_link.log"
test -f "$OUT/$KERNEL.xclbin"
echo "== $OUT/$KERNEL.xclbin"
