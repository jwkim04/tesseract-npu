#!/usr/bin/env bash
# hw/u200/bw/measure_bw2.sh — read-only bandwidth matrix of the two-compute-unit xclbin built with
# hw/u200/bw/bw2x200.cfg (two AXI read ports per DDR4 bank, both at 200 MHz). It asks whether two ports
# together deliver the bank's bandwidth (about 18 GB/s) where one port at 200 MHz gives 64 B x 200 MHz = 12.8 GB/s.
# Runs sw/tools/tsr_bw (docs/bandwidth-kernel.md §6) and writes one text table, build/u200/bw2x200/measure.txt
# (the complete tool output goes to measure_raw.log next to it):
#   1. one engine alone on bank 0 (tsr_bw_a), burst 64, outstanding 16: the single-port baseline
#   2. both engines on bank 0: burst 16, 32, 64 x outstanding 2, 4, 8, 16
#   3. both engines on all four banks: burst/outstanding 64/16, 16/8, 32/8, 64/8 and the best setting of 2.
#   4. bank 0, burst 64, outstanding 16, one engine and both, with one AXI ID (the other runs use TSR_BW2_IDS,
#      16 like the core)
# Only reads: the tool writes the read pattern once per run of the tool (its own buffers, 2 GiB per engine and bank)
# and verifies it. It waits until nobody has the card open (the check of sw/tools/prof/capture_card.sh), before
# the preflight and before every run of the tool, and never kills or signals another process.
#
#   hw/u200/bw/measure_bw2.sh [XCLBIN]       (default build/u200/bw2x200/tsr_bw_k4.xclbin)
#
#   TSR_BW_BIN     tsr_bw binary (default build/bw2/tsr_bw)
#   TSR_BW2_OUT    output directory (default build/u200/bw2x200)
#   TSR_BW2_SIZE   bytes per engine and bank (default 2G; the buffers of both engines of a bank must stay under
#                  the 8 GiB behind a bank of this card)
#   TSR_BW2_REPS   repetitions per point, the median is reported (default 3)
#   TSR_BW2_IDS    AXI IDs per channel (1, 2, 4, 8, 16; default 16, the core's)
#   TSR_XRT_DEVICE card index or BDF (default 0)
# Exit status: 0 = every run of the tool passed its checks, 1 = a run did not, 2 = the preflight failed.
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
XCLBIN=${1:-$ROOT/build/u200/bw2x200/tsr_bw_k4.xclbin}
BIN=${TSR_BW_BIN:-$ROOT/build/bw2/tsr_bw}
OUT=${TSR_BW2_OUT:-$ROOT/build/u200/bw2x200}
SIZE=${TSR_BW2_SIZE:-2G}
REPS=${TSR_BW2_REPS:-3}
IDS=${TSR_BW2_IDS:-16}
A=tsr_bw_a
B=tsr_bw_b
TXT=$OUT/measure.txt
RAW=$OUT/measure_raw.log
source /opt/xilinx/xrt/setup.sh > /dev/null 2>&1
for f in "$XCLBIN" "$BIN"; do
  [ -f "$f" ] || { echo "measure_bw2.sh: $f not found" >&2; exit 2; }
done
mkdir -p "$OUT"

card_idle() { ! lsof -w /dev/dri/by-path/pci-0000:01:00.1-render 2>/dev/null | awk 'NR>1' | grep -q .; }
wait_idle() {
  local n=0
  until card_idle; do n=$((n + 1)); [ $((n % 15)) -eq 1 ] && echo "$(date +%T) card busy, waiting" >&2; sleep 4; done
}

wait_idle
: > "$RAW"
echo "### preflight: $BIN --xclbin $XCLBIN --dry-run --cu $A,$B --sizes $SIZE" >> "$RAW"
if ! PRE=$("$BIN" --xclbin "$XCLBIN" --dry-run --cu $A,$B --sizes "$SIZE" 2>&1); then
  printf '%s\n' "$PRE" | tee -a "$RAW" >&2
  echo "measure_bw2.sh: preflight failed" >&2
  exit 2
fi
printf '%s\n' "$PRE" >> "$RAW"
{
  echo "# measure_bw2.sh $(date -Is) on $(hostname), $SIZE per engine and bank, $REPS repetitions (median), AXI IDs $IDS"
  echo "# xclbin $XCLBIN"
  echo "# sha256 $(sha256sum "$XCLBIN" | cut -d' ' -f1)"
  printf '%s\n' "$PRE" | grep -E '^(device|kernel clock|compute units)' | sed 's/^/# /'
  echo "# GB/s = bytes / RD_CYCLES at the kernel clock, per engine; sumGB/s = a + b; bank allN = bank N of the run on all banks, all = their total"
} > "$TXT"

# ---- one run of the tool: tool LABEL ARGS...; sets LAST (its output) and RC ----
runs=0
bad=0
tool() {
  local label=$1
  shift
  wait_idle
  runs=$((runs + 1))
  echo "$(date +%T) $label: tsr_bw $*" >&2
  echo "### $label: $BIN $*" >> "$RAW"
  LAST=$("$BIN" --xclbin "$XCLBIN" --sizes "$SIZE" --reps "$REPS" --ids "$IDS" --modes r "$@" 2>&1)
  RC=$?
  printf '%s\nrc=%d\n' "$LAST" "$RC" >> "$RAW"
  [ "$RC" -eq 0 ] || bad=$((bad + 1))
}
# ---- the rows of the last run into the table; the header lines once per section ----
section() { printf '\n== %s\n' "$1" >> "$TXT"; HDR=0; }
emit() {
  if [ "$HDR" = 0 ]; then
    printf '%s\n' "$LAST" | grep -E -m1 '^\(a = ' >> "$TXT"
    printf '%s\n' "$LAST" | grep -E -m1 '^mod ' >> "$TXT"
    HDR=1
  fi
  printf '%s\n' "$LAST" | grep -E '^(r  |    )' >> "$TXT"
  [ "$RC" -eq 0 ] || echo "   tool exit status $RC: $(printf '%s\n' "$LAST" | tail -3 | tr '\n' ' ')" >> "$TXT"
}

# 1. one engine alone
section "one engine alone ($A), bank 0, burst 64, outstanding 16"
tool single --cu $A --channels 0 --no-all --bursts 64 --outstanding 16
emit
SINGLE=$(printf '%s\n' "$LAST" | awk '$1 == "r" && $2 == "ch0" && $NF == "ok" { print $6 }')

# 2. both engines on bank 0
section "both engines, bank 0 (a = $A, b = $B)"
ALL0=
for o in 2 4 8 16; do
  tool "bank0 out $o" --cu $A,$B --channels 0 --no-all --bursts 16,32,64 --outstanding $o
  emit
  ALL0+=$LAST$'\n'
done
# columns: mod bank size bur out a b sum a_cyc b_cyc a_lat b_lat skew end overlap % check
BEST=$(printf '%s\n' "$ALL0" | awk '$1 == "r" && $2 == "ch0" && $NF == "ok" && $8 + 0 > m { m = $8 + 0; b = $4 "/" $5 } END { print b }')
BESTSUM=$(printf '%s\n' "$ALL0" | awk '$1 == "r" && $2 == "ch0" && $NF == "ok" && $8 + 0 > m { m = $8 + 0 } END { print m }')

# 3. both engines on all four banks
section "both engines, all four banks at once (8 engines)"
SETS="64/16 16/8 32/8 64/8"
if [ -n "$BEST" ] && ! printf ' %s ' $SETS | grep -q " $BEST "; then SETS="$SETS $BEST"; fi
ALL4=
for s in $SETS; do
  tool "all banks $s" --cu $A,$B --channels none --bursts "${s%/*}" --outstanding "${s#*/}"
  emit
  ALL4+=$LAST$'\n'
done
BEST4=$(printf '%s\n' "$ALL4" | awk '$1 == "r" && $2 == "all" && $NF == "ok" && $8 + 0 > m { m = $8 + 0; b = $4 "/" $5 } END { if (b != "") print m " GB/s at burst/outstanding " b }')

# 4. one AXI ID, bank 0
section "bank 0, burst 64, outstanding 16, one AXI ID (single engine, then both)"
IDS=1
tool "ids1 single" --cu $A --channels 0 --no-all --bursts 64 --outstanding 16
emit
tool "ids1 both" --cu $A,$B --channels 0 --no-all --bursts 64 --outstanding 16
emit
IDS=${TSR_BW2_IDS:-16}

{
  echo
  echo "== summary"
  echo "one engine on bank 0, burst 64, outstanding 16: ${SINGLE:-n/a} GB/s"
  echo "both engines on bank 0, best of the matrix: ${BESTSUM:-n/a} GB/s sum at burst/outstanding ${BEST:-n/a}"
  echo "both engines on all four banks, best total over the four banks: ${BEST4:-n/a}"
  echo "# done $(date -Is): $runs runs of the tool, $bad with a failed check or a setup error"
} >> "$TXT"
echo "measure_bw2.sh: $TXT ($runs runs of the tool, $bad failed)" >&2
[ "$bad" -eq 0 ]
