#!/bin/bash
# hw/u200/vivado_slot.sh CMD ARGS... - run CMD while holding one of two Vivado slots, so at most two Vivado
# processes run on the host. The slots are vivado.lock and vivado2.lock in TSR_VIVADO_LOCK_DIR
# (default /tmp/tsr-vivado). Every checkout on the host must use the same directory. Blocks until a
# slot is free.
L=${TSR_VIVADO_LOCK_DIR:-/tmp/tsr-vivado}
mkdir -p "$L"
while :; do
  for s in vivado.lock vivado2.lock; do
    exec 9>>"$L/$s"
    if flock -n 9; then
      "$@"; rc=$?
      flock -u 9; exec 9>&-
      exit $rc
    fi
    exec 9>&-
  done
  sleep 15
done
