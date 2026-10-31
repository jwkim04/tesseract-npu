#!/usr/bin/env bash
# gdb_smoke.sh — tsr-dbg's RSP server under a real GDB (docs/design/05-debug.md "DB" §4; P3.7; the P6.3
# GDB smoke on sim): the toolchain's riscv-none-elf-gdb (SW-7) or gdb-multiarch drives
# `target remote | tsr-dbg gdb --stdio --device sim --elf fw_rt.elf --halt-on-boot` in batch mode:
# pc at boot, a breakpoint on a firmware symbol, continue, the hit, memory, stepi, CSRs, monitor
# commands, a hardware breakpoint on a kernel launched by `monitor launch`, a backtrace, detach.
# Skipped (77) without a RISC-V GDB.
#   gdb_smoke.sh <gdb> <tsr-dbg> <fw_rt.elf> <scratch>
set -u
GDB=$1 DBG=$2 ELF=$3 DIR=$4
if [ -z "$GDB" ] || [ ! -x "$GDB" ]; then
    echo "gdb_smoke: no RISC-V GDB (riscv-none-elf-gdb / gdb-multiarch): skipped"
    exit 77
fi
mkdir -p "$DIR"
LOG=$DIR/gdb.log
"$GDB" -batch -nx \
    -ex "set confirm off" -ex "set pagination off" -ex "set remotetimeout 60" \
    -ex "target remote | $DBG gdb --stdio --device sim --elf $ELF --halt-on-boot" \
    -ex "printf \"PC0=%#x\\n\", \$pc" \
    -ex "monitor why" \
    -ex "break tsr_rt_main" \
    -ex "continue" \
    -ex "printf \"PC1=%#x\\n\", \$pc" \
    -ex "x/4xw &__tsr_dbg" \
    -ex "stepi" \
    -ex "printf \"PC2=%#x\\n\", \$pc" \
    -ex "printf \"MISA=%#x\\n\", \$misa" \
    -ex "info registers sp ra" \
    -ex "delete" \
    -ex "hbreak rt_copy" \
    -ex "monitor launch rt_copy 0x100000 0x100100 2 0" \
    -ex "continue" \
    -ex "printf \"PC3=%#x\\n\", \$pc" \
    -ex "bt" \
    -ex "finish" \
    -ex "monitor hist 4" \
    -ex "monitor status" \
    -ex "detach" \
    "$ELF" > "$LOG" 2>&1
rc=$?
cat "$LOG"
fail=0
need() {
    if grep -q -- "$1" "$LOG"; then echo "ok   $2"; else echo "FAIL $2 (no \"$1\")"; fail=1; fi
}
need "PC0=0x80" "pc at boot is RV_BOOT + 0x80"
need "halted at boot" "monitor why: halted at boot"
need "Breakpoint 1, .*tsr_rt_main" "the breakpoint on tsr_rt_main is hit"
MAIN=$(grep -o "Breakpoint 1 at 0x[0-9a-f]*" "$LOG" | head -1 | awk '{print $4}')
need "PC1=$MAIN" "pc = tsr_rt_main ($MAIN)"
need "0x47424454" "x/4xw &__tsr_dbg shows the TDBG magic"
need "PC2=0x" "stepi"
need "MISA=0x40101104" "\$misa (RV32IMC)"
need "Hardware assisted breakpoint 2" "hbreak"
need "pushed as seq\|queued" "monitor launch"
need "rt_copy" "stopped in rt_copy"
need "Value returned is .* = 0" "finish: rt_copy returns 0 (a temporary breakpoint at the return address)"
need "LD ch0.*rt_copy" "monitor hist shows the kernel's LD with its push location"
need "ST ch0.*rt_copy" "monitor hist shows the kernel's ST"
need "QCLEAR" "monitor hist decodes the runtime's QCLEAR"
need "verdict (DB" "monitor status"
need "Detaching from program\|\[Inferior 1 (Remote target) detached\]" "detach"
[ $rc -eq 0 ] || { echo "FAIL gdb exit status $rc"; fail=1; }
[ $fail -eq 0 ] && echo "PASS" || echo "FAIL"
exit $fail
