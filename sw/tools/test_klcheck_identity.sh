#!/usr/bin/env bash
# test_klcheck_identity.sh KLCHECK MODEL OUT — ctest tools_klcheck_identity_qwen05b (docs/klcheck.md;
# docs/v2/06-software-verification.md §10.4 M6 gate item 2 on the simulator kinds, §10.1 "cross-transport"
# row): generate on sim (M6 settings), replay the same tokens on mock (the adapter register model; step
# replay, the default of a device replay), and require every logits row bit-identical.
set -eu
KL=$1
M=$2
OUT=$3
rm -rf "$OUT"
mkdir -p "$OUT"
"$KL" run -m "$M" --device sim --phase m6 --n-prompts 2 -n 8 --out "$OUT/sim.klc"
"$KL" run -m "$M" --device mock --phase m6 --tokens "$OUT/sim.klc" --out "$OUT/mock.klc"
"$KL" compare --ref "$OUT/sim.klc" --dev "$OUT/mock.klc" --expect-identical
