#!/bin/bash
# hw/u200/vivado_slot.sh CMD ARGS... - run CMD while holding one of two Vivado slots, so at most two Vivado
# processes run on the host. Slot 1 is vivado.lock (the lock the builders use), slot 2
# vivado2.lock. Blocks until a slot is free.
L=${TSR_VIVADO_LOCK_DIR:-/tmp/claude-1000/-home-limagin-dev-tesseract}
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
