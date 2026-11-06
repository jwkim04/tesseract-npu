#!/usr/bin/env bash
# hw/u200/build_xclbin.sh — build the Tesseract U200 xclbin (docs/platform-u200.md §8).
#
#   hw/u200/build_xclbin.sh k1|k4 [hw|hw_emu]        (default target: hw)
#
# The M0 bandwidth kernel builds separately (hw/u200/bw/build_bw_xclbin.sh).
#
# 1. the kernel object build/u200/xo/tsr_u200_<k>.xo: packaged now by
#    rtl/platform/alveo_xrt/package_xo.tcl (Vivado batch) unless it exists and its source manifest
#    build/u200/xo/tsr_u200_<k>.sources.sha256 (written by package_xo.tcl: filelists, every RTL
#    source, kernel_<k>.xml, the CDC and TCM constraints, package_xo.tcl) still matches the working tree
#    (TSR_REPACKAGE=1 forces it). The IP carries rtl/ctrl/viv/tsr_ctrl_cdc.xdc (the clock-domain crossings of the
#    control core) and rtl/ctrl/viv/tsr_ctrl_tcm.xdc (its TCMs in RAMB36, never URAM)
# 2. v++ --link -t <target> --platform <xpfm> --config <OUT>/<k>.link.cfg
#    -> <OUT>/tsr_u200_<k>.xclbin (+ .link_summary, logs, reports); <k>.link.cfg is hw/u200/<k>.cfg
#    with its hook paths made absolute (the configs give them relative to a directory three levels below
#    the repository, where v++ resolves them: against its working directory; TSR_U200_OUT may point to
#    another one at that depth, e.g. build/link/<k>). The configs check clock/reset wiring after BD
#    creation and before opt_design (with the TCM power-optimisation exclusion), apply the k4 floorplan
#    before placement, and run the physical CDC gate after routing. A final pre-bitstream hook checks
#    Vitis's scaled frequencies in _new_clk_freq against the kernel's window (k4: core 200 MHz and clk_rv
#    150 MHz; k1: core 250 to 300 MHz). Clock identity/reset/frequency failures are fatal;
#    TSR_CDC_GATE=warn only relaxes the final CDC report gate.
#
# Environment:
#   TSR_U200_PLATFORM   platform name or .xpfm path (default xilinx_u200_gen3x16_xdma_2_202110_1),
#                       searched in $PLATFORM_REPO_PATHS (colon list), /opt/xilinx/platforms.
#                       PLATFORM_REPO_PATHS unset: build/platforms/root/opt/xilinx/platforms (the
#                       development platform extracted on the dev server) when that directory exists
#   TSR_XILINX          Xilinx install root (default /tool/Xilinx); version 2024.2
#   TSR_VIVADO_LOCK     if set: every Vivado / v++ call runs under `flock $TSR_VIVADO_LOCK`
#   TSR_U200_FLOORPLAN  k4: 1 (default) = whole-block SLR pblocks from
#                       hw/u200/k4_major_blocks.tcl; 0 = no placement hook (CDC hooks remain)
#   TSR_U200_OUT        output directory (default build/u200/<k>, build/u200/<k>_hw_emu);
#                       config hook paths are made absolute before v++ reads them
#   TSR_VPP_TO_STEP     stop v++ after this link step (v++ --to_step, e.g. vpl.config_hw_runs: the
#                       implementation runs and their hooks are set up, nothing is synthesized)
#   TSR_VPP_FROM_STEP   resume a link in TSR_U200_OUT from this step (v++ --from_step, e.g. vpl.impl after a
#                       failed implementation run: the synthesis is kept)
#   TSR_U200_PLATFORMINFO  1 = resolve the platform, print its platforminfo and stop (exit 0)
#   TSR_BUILD_JOBS      if set: the link's synth / impl jobs and Vivado's general.maxThreads (the configs: 8)
#   TSR_U200_SYNTH_JOBS optional independent positive integer count of concurrent IP synthesis jobs;
#                       overrides synth.jobs only. TSR_BUILD_JOBS still controls each tool's threads
#                       and implementation jobs.
# Stops with exit status 3 and a message when the platform (or the Xilinx tools) are missing.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
K=${1:-}
TARGET=${2:-hw}
case "$K" in
  k1|k4) ;;
  *) echo "usage: hw/u200/build_xclbin.sh k1|k4 [hw|hw_emu]" >&2; exit 2 ;;
esac
case "$TARGET" in
  hw|hw_emu) ;;
  *) echo "build_xclbin.sh: target must be hw or hw_emu" >&2; exit 2 ;;
esac
if [ -n "${TSR_U200_SYNTH_JOBS:-}" ]; then
  if ! [[ "$TSR_U200_SYNTH_JOBS" =~ ^[1-9][0-9]*$ ]]; then
    echo "build_xclbin.sh: TSR_U200_SYNTH_JOBS must be a positive integer" >&2
    exit 2
  fi
fi
KERNEL=tsr_u200_$K
PLATFORM=${TSR_U200_PLATFORM:-xilinx_u200_gen3x16_xdma_2_202110_1}
XILINX=${TSR_XILINX:-/tool/Xilinx}
VIVADO_SETTINGS=$XILINX/Vivado/2024.2/settings64.sh
VITIS_SETTINGS=$XILINX/Vitis/2024.2/settings64.sh
XO_DIR=$ROOT/build/u200/xo
XO=$XO_DIR/$KERNEL.xo
OUT=${TSR_U200_OUT:-$ROOT/build/u200/$K$([ "$TARGET" = hw ] || echo "_$TARGET")}
CFG_SRC=$ROOT/hw/u200/$K.cfg
CFG=$OUT/$K.link.cfg

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
build_xclbin.sh: the Vitis platform $PLATFORM is not installed.
  Looked for $PLATFORM.xpfm in \$PLATFORM_REPO_PATHS (${PLATFORM_REPO_PATHS:-unset}) and /opt/xilinx/platforms.
  Install the U200 platform packages (xilinx-u200-gen3x16-xdma-2-202110 deployment and development
  packages, with XRT) or point TSR_U200_PLATFORM at the .xpfm, then run this script again.
  Nothing was built ($KERNEL.xo is packaged on its own by rtl/platform/alveo_xrt/package_xo.tcl).
EOM
  exit 3
fi
for f in "$VIVADO_SETTINGS" "$VITIS_SETTINGS"; do
  if [ ! -f "$f" ]; then
    echo "build_xclbin.sh: $f not found (set TSR_XILINX to the Xilinx 2024.2 install root)" >&2
    exit 3
  fi
done
echo "== platform $XPFM"
if [ "${TSR_U200_PLATFORMINFO:-0}" = 1 ]; then
  locked bash -c "set +u; source '$VITIS_SETTINGS' >/dev/null && platforminfo -p '$XPFM'"
  exit 0
fi
mem_kb=$(awk '/MemTotal/ {print $2}' /proc/meminfo)
if [ "$mem_kb" -lt $((30 * 1024 * 1024)) ]; then
  echo "build_xclbin.sh: warning: $((mem_kb / 1024 / 1024)) GiB RAM; a U200 implementation may need 32 GiB or more" >&2
fi

# ---- 1. kernel object ----
need_pkg=${TSR_REPACKAGE:-0}
MANIFEST=$XO_DIR/$KERNEL.sources.sha256
if [ ! -f "$XO" ] || [ ! -f "$MANIFEST" ]; then
  need_pkg=1
elif ! (cd "$ROOT" && sha256sum --quiet --strict -c "$MANIFEST" > /dev/null 2>&1); then
  echo "== $KERNEL.xo is stale (sources changed since packaging: cd $ROOT && sha256sum -c $MANIFEST)"
  need_pkg=1
fi
if [ "$need_pkg" = 1 ]; then
  echo "== packaging $KERNEL -> $XO"
  mkdir -p "$XO_DIR"
  locked bash -c "set +u; source '$VIVADO_SETTINGS' >/dev/null && cd '$XO_DIR' && vivado -mode batch -nojournal -nolog \
      -source '$ROOT/rtl/platform/alveo_xrt/package_xo.tcl' -tclargs '$ROOT' '$KERNEL' '$XO_DIR'" \
      > "$XO_DIR/$KERNEL.package.log" 2>&1 || { tail -30 "$XO_DIR/$KERNEL.package.log"; exit 1; }
  grep '^PKG' "$XO_DIR/$KERNEL.package.log"
fi

# ---- 2. link ----
mkdir -p "$OUT"
# the hook paths of the config are relative to build/u200/<k> (three levels below the repository)
sed -e "s|=\.\./\.\./\.\./|=$ROOT/|" "$CFG_SRC" > "$CFG"
if [ -n "${TSR_BUILD_JOBS:-}" ]; then
  sed -i -e "s|^synth\.jobs=.*|synth.jobs=$TSR_BUILD_JOBS|" \
    -e "s|^impl\.jobs=.*|impl.jobs=$TSR_BUILD_JOBS|" \
    -e "s|^param=general\.maxThreads=.*|param=general.maxThreads=$TSR_BUILD_JOBS|" "$CFG"
fi
if [ -n "${TSR_U200_SYNTH_JOBS:-}" ]; then
  sed -i "s|^synth.jobs=.*|synth.jobs=$TSR_U200_SYNTH_JOBS|" "$CFG"
fi
if [ "$K" = k4 ] && [ "${TSR_U200_FLOORPLAN:-1}" = 0 ]; then
  sed -i -e '/k4_major_blocks\.tcl/d' "$CFG"
  echo "== TSR_U200_FLOORPLAN=0: linking without hw/u200/k4_major_blocks.tcl"
fi
extra=()
if [ -n "${TSR_VPP_TO_STEP:-}" ]; then
  extra+=(--to_step "$TSR_VPP_TO_STEP")
fi
if [ -n "${TSR_VPP_FROM_STEP:-}" ]; then
  extra+=(--from_step "$TSR_VPP_FROM_STEP")
fi
echo "== v++ link $KERNEL ($TARGET, $XPFM, $CFG) -> $OUT"
locked bash -c "set +u; source '$VITIS_SETTINGS' >/dev/null && cd '$OUT' && v++ --link -t '$TARGET' --platform '$XPFM' \
    --config '$CFG' --save-temps --temp_dir '$OUT/tmp' --log_dir '$OUT/logs' --report_dir '$OUT/reports' \
    ${extra[*]:+${extra[*]}} -o '$OUT/$KERNEL.xclbin' '$XO'" 2>&1 | tee "$OUT/v++_link.log"
if [ -n "${TSR_VPP_TO_STEP:-}" ]; then
  grep -q "exits after completing the '$TSR_VPP_TO_STEP' step" "$OUT/v++_link.log"
  echo "== stopped after $TSR_VPP_TO_STEP (no xclbin); continue with v++ --from_step in $OUT"
  exit 0
fi
test -f "$OUT/$KERNEL.xclbin"
echo "== $OUT/$KERNEL.xclbin"
