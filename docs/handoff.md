# Handoff: where the work stands (2026-11-09)

Work package numbers (W0-W6, P*.*) are those of [design/README.md](design/README.md) §10.

## Resume here for the software: the branches merged into `sw-moegdn` (November 9)

Four finished branches were merged into `sw-moegdn` (real merges, in this order), on top of SW-76: `moe-qwen3` (SW-77, Qwen3-30B-A3B), `vu-perf` (SW-80, the
simulator's cycle models), `torch-basis` (SW-81, the PyTorch backend and `QuantLinear`) and `gdn-fw` (SW-78, SW-79, SW-82: the fused GDN op of the VU in the firmware, numerics
revision 2 of GATED_DELTA_NET, the state on its cache cell). The first three change no RTL (`git diff 2d887ec -- rtl hw` was empty after them); `gdn-fw` brings the VU's GDN op
(opcode 0x39, `CAPS` bit 8) from `gdn-vu` with its `rtl/` and `tb/` files, which the card image of 2026-11-09 does not have. The sections below describe each branch as it was before the merge.

- Decisions in [design/06-software-verification.md](design/06-software-verification.md): SW-75 to SW-82, in order, with no gap. The next free number is SW-83.
- Qwen3-30B-A3B prefill runs at the default `-ub 512` (SW-76 fixed the several-tile route): 26.72 tok/s best, 26.60 median on the card, 513 of 513 rows identical to the
  simulator. Gemma 4 26B-A4B: 26.58 / 26.56 tok/s, 515 of 515 rows identical. The Qwen3 layer test in `test-tsr-moe` now runs T = 65, 70, 130 and 256 as well
  (70 aborted before SW-76).
- The simulator in the card configuration (`TSR_SIM_CARD=k4full`) is within 0.2 % (Qwen3.8 2B, 12.65 M against 12.63 M cycles) and 0.4 % (Gemma 4 26B-A4B) of the card.
  The Qwen2.5-0.5B forward of the PyTorch backend: 0.794 M against the card's 0.808 M for 8 tokens, 0.639 M against 0.659 M for 32.
  Older simulator cycle figures in the docs say that they come from the model before SW-80.
- Checked after the merges (build `build/dev`, with `TSR_WITH_TORCH=ON`): `ctest -LE slow` 318 of 318 (the tokenizer vocabulary test is disabled), including the six
  `ggml_tesseract_moe*` / `qwen3moe` cases and `torch_pytest*` at `T_MAX` 1, 4 and 8; `gen_tables.py --check` and `gen_tops.py --check`; `tb/vu/test_vu_timing.py` and
  `tb/qact` (15 passed, 4 opt-in skipped). klcheck M7 on Qwen3.8 2B (1 prompt, 16 tokens, simulator): 16 of 16 rows bit-identical to the same run of the build before the
  merges (510e282), with the default lanes and with `TSR_SIM_CARD=k4full`; the logits of the PyTorch 8-token forward keep their sha256 at `T_MAX` 1, 4 and 8.
- Checked after the `gdn-fw` merge (build `build/dev`, `TSR_WITH_TORCH=ON`): `ctest -LE slow` 323 of 323 (the tokenizer vocabulary test is disabled);
  `gen_tables.py --check` and `gen_tops.py --check`; `rtl/lint.sh` with the Vivado wrapper (all clean); `tb/run_all.sh --no-lint gdn vu` (37 passed, 62 opt-in skipped) and
  `core -k k4gdn` (1 passed). The production firmware: ITCM 85,504 B of the 98,304 B budget (SW-75), DTCM 50,616 B, 21 kinds, kind table 0x2381 (GATED_DELTA_NET revision 2).
  The merge had four conflicts: this file, the decision table of `06-software-verification.md`, `tsr-build.h` and `ggml-tesseract.cpp` (the scratch window of SW-76 and the
  `read_i32` and `gdn_cell` members of SW-82 side by side); the generated files needed no change.
  klcheck M7, 1 prompt x 16 tokens, `compare --expect-identical`: Qwen3.8 2B with `TSR_SIM_CARD=k4full` (composed path) 16 of 16 against the same run at `afcdae1`;
  Qwen3.8 2B with `TSR_SIM_CARD=k4full TSR_SIM_GDN=1` (fused path) 16 of 16 against the `gdn-fw` build (`8cc4c33`); Gemma 4 26B-A4B (default lanes) 16 of 16 against `afcdae1`.
  The plain simulator now has `CAPS` bit 8, so it runs the fused path for Qwen3.8 (S_v 128): its bits are not those of the card runs of 2026-11-09. Replay those with `TSR_SIM_CARD=k4full` or `TSR_SIM_GDN=0`.
- Open: the card run of the fused path (plan in the next section; it needs the `k4f250` image); grouping of tokens by expert (SW-66); the KV store, MOE_ROUTE and the DMA's half rate while the VU runs are not in the cycle model (SW-80).

## Profile report of one decode token (branch `prof-viz`, SW-84, November 9)

A capture and report path for one model token, run on the card and on the simulator ([prof-dbg.md](prof-dbg.md) §3.1). No RTL, firmware or llama.cpp change.

- Backend: `TSR_DUMP_PLAN=<dir>` (`sw/ggml-tesseract/tsr-prof.{h,cpp}`, hooks in `ggml-tesseract.cpp`) writes the plan of each built graph (nodes with placement, fusion groups,
  records with arguments, DRAM ranges, weight bytes, ideal MVM cycles, SPM tile plan, the reason each list ended), the launches with PERF deltas, the `supports_op` refusals and
  the trace writer's records (the process starts and stops the ring itself). It sets the PROF bit. Bits do not move: `klcheck compare --expect-identical`, Gemma 4 26B-A4B,
  1 x 16, simulator, dump on against off, and the generated text of the tiny-model tests.
- Tools: `sw/tools/prof/capture_card.sh`, `capture_sim.sh`, `make_report.sh`, `prof_report.py` (HTML page, Perfetto JSON, summary JSON; standard library only),
  `test_prof_report.py` (`tools_prof_report_qwen3moe_tiny`, `tools_prof_report_gemma_tiny`).
- Gemma 4 26B-A4B on the card (`k4full`, 33 tokens, one session, 2026-11-09): decode launch 16.303 M to 16.318 M cycles (launch 18: 16.311 M against the earlier 16.316 M),
  11.86 tok/s with the profiler off and 11.68 with it on (llama.cpp eval time, same session), U 55.9 % (H 2.9 %, G 18.6 %, F 1.5 %, E 71.7 %, the clock bound at 200 MHz being 71.1 %); the simulator in the card configuration is
  0.995 to 0.996 of the card for each of the 32 decode tokens. Numbers: [performance.md](performance.md) "Profile report of one Gemma 4 26B-A4B decode token".
- The page was checked with a stub run of its script (gjs), the checks of the ctest (title, colour tokens, no external script, size) and a render of its SVG charts and graphs
  with cairosvg; no browser was available in the session, so its layout in a real browser is not looked at.
- Open: PERF counters are per launch only (the hardware gives no finer sample); the op slices of a record share the record's ideal cycles evenly (the card trace has no op words);
  the layer and group names come from node names (Gemma 4 and Qwen3 MoE); the scheduler's CPU split is read from `GGML_SCHED_DEBUG=2` output (`-lv 5`), not from the backend.

## The fused GDN op in the firmware and the card plan for the `k4f250` image (branch `gdn-fw`, merged, November 9)

Simulator only; the RTL (the VU's GDN op) comes from `gdn-vu`. The branch was `sw-moegdn` (2d887ec) with `gdn-vu` (bdcf7b9, the GDN op in the VU) and `vu-perf` (31b3d71, SW-80, the cycle models
that follow the card) merged in. The `gdn-vu` merge had two conflicts, both in docs; the `vu-perf` merge four (`tsr_vu.c`'s constants, `test_sim_vu.c`'s test list, this file and
the decision table). The generated files came out of the generators unchanged (`gen_tables.py --check`, `gen_tops.py --check`). The card image with the op is the `k4f250` link, still
being built; no card was used. The branch is now merged into `sw-moegdn` (the first section says what was checked after the merge).

- GATED_DELTA_NET runs the VU's GDN op per head and token when `CAPS` has bit 8 (with the VU) and S_v is 64 or more, and keeps the 13-op composition otherwise. The choice is made per
  launch from the device's `CAPS` (`plan::gdn::use_fused`); the record, the planner and the ring block are unchanged. Per step: three `SOP`, `GDN` (state tile in place),
  `VOP.MULC` for the scale (SW §7.10, SW-78).
- Numerics revision 2 (`opl::kind_rev`). The bits change only on a device with `CAPS` bit 8 and for S_v ≥ 64 (block fixed-point dot products, α folded into k, the output from the old
  state and κ = k · q, NaN rows: SW §7.10 lists them). A device without the bit, and S_v 16 and 32 everywhere, give revision 1's bits. `kind_revs.txt` has a second row for
  GATED_DELTA_NET marked `composed`; its digest is revision 1's (`b040cc2f4da8c0ca`), the default row is the fused corpus (`f768379b55c9df2c`). The kind table signature is
  now 0x2381. The oracle `gdn_ref` takes the `CAPS` of the device and models both paths (`tsr_ref_gdn` for the fused one).
- Simulator knobs: `TSR_SIM_GDN=0|1` clears or sets `CAPS` bit 8. The plain simulator reports bit 8 (it implements the op); `TSR_SIM_CARD=k4full` is the card image of 2026-11-09 and has no
  bit 8 (CAPS without it, as on the card), so **a replay of a card run of 2026-11-09 uses `TSR_SIM_CARD=k4full` or `TSR_SIM_GDN=0`**; the image with the op is `TSR_SIM_CARD=k4full
  TSR_SIM_GDN=1` (add `TSR_SIM_RV_RATIO=0.6667` for the 250 MHz link). The simulator's cycle model of the GDN op is fitted to the RTL (`tb/vu test_vu_timing.py vu_gdn_op`, 2,914 cycles for 128 × 128 at 8 lanes).
- Tests: `fw_gdn_rv` (fused path, with Inf / NaN / subnormal records), `fw_gdn_rv_composed` (`--gdn 0`), `fw_gdn_ggml` (both paths against ggml-cpu and float64 up to 512 tokens: rms error 1.0e-7
  on both, the CPU's own is 0.95e-7), `fw_kind_revs` (a row per path). ITCM 85,504 B of 98,304 B. Timing moves no bit: `fw_kind_revs`, `fw_gdn_rv` and a 16-row klcheck (16 of 16 identical) are the same with `TSR_SIM_VU_TIMING=k4`.
- Simulator, Qwen3.8 2B, klcheck M7 8 × 64, `T_MAX` 4: fused ratio 1.103, top-1 97.27 %; composed 1.061, 98.05 %, equal to the card run to every digit.
- The state on its cache cell (SW-82, host lowering only): when `rs_s_copy` says every sequence keeps its cell, the GATED_DELTA_NET record reads and writes the cell (`STATE_IN` = `STATE_OUT`)
  and the `GET_ROWS` gather and the state `CPY` are absorbed: one pass over the state instead of three, for both paths, bits unchanged (512 of 512 klcheck rows identical to runs made
  before it). A plan that depends on the ids is checked at every reuse (`tsr_plan_state_holds`). `TSR_GDN_CELL=0` turns it off; `TSR_STATS` shows "GATED_DELTA_NET records on their state cache
  cell". Cases that keep the three passes: a new sequence whose zero cell is another cell, `seq_cp`, several sequences that swap cells, extra states.
- Cycles per decode token in the card-configuration model (`TSR_SIM_CARD=k4full`, which reproduces the card: 12.65 M against 12.63 M) and the 27B estimate by the 4 and 8 layer method
  ([performance.md](performance.md), SW §7.10):

  | | 2B composed (the card's run) | 2B fused | 2B fused + state on its cell | 27B composed | 27B fused | 27B fused + cell |
  |---|---|---|---|---|---|---|
  | 200 MHz image, RISC-V ratio 0.5 | 12.65 M, 63.3 ms, 15.8 tok/s | 9.72 M, 48.6 ms, 20.6 tok/s | 8.53 M, 42.6 ms, 23.5 tok/s | 123.7 M, 618 ms | 100.6 M, 503 ms | 91.1 M, 456 ms |
  | 250 MHz link, ratio 0.667 | 12.51 M, 50.0 ms | 9.23 M, 36.9 ms, 27.1 tok/s | 8.04 M, 32.2 ms, 31.1 tok/s | 123.3 M, 493 ms | 97.2 M, 389 ms | 87.7 M, 351 ms, 2.85 tok/s |

  The fused step is limited by the RISC-V issue rate (4.7 K core cycles per head and token at ratio 0.5, 3.5 K at 0.667; the VU needs about 3.1 K), so the fused path gains 23 % of a 2B token,
  and the state on its cell another 12 %.
- Left: the card run (plan below); a decode step that shares the g / beta / k / q / v loads across the heads of a token (6 pushes per head instead of 11, a gain of about 1 K cycles per
  head and token at ratio 0.5, 2 to 3 % of a 27B token); the RTL's overlap of independent heads is not in the model (VU §7.12.8).

### Card plan for the `k4f250` image (not run)

Build the host with `-DTSR_WITH_XRT=ON` from branch `sw-moegdn` (`build/xrt`, as for the earlier card runs) and use the `tsr-fw.elf` of the same build
(21 kinds, kind table 0x2381, ITCM 85,504 B). The image has `HAS_GDN` 1: `CAPS` should read 0x121 with the VU and attention bits (bit 8 is the op).
`TSR_XCLBIN` is the `k4f250` xclbin; core clock 250 MHz, `clk_rv` 166.7 MHz. If the image still has the first-ATTPV-after-load defect, replay the
trace `traces/at/errors/at_s53_0009` before each run as for `k4full`.

```sh
# 1. identity and the GDN self-test (CAPS bit 8 must show; id 11 is skipped, RET CAPS, on a device without it)
TSR_XCLBIN=$X build/xrt/bin/tsr-selftest --device xrt                       # identity: CAPS 0x...121, kinds (21, signature 0x2381)
TSR_XCLBIN=$X build/xrt/bin/tsr-selftest --device xrt --no-mem --ids 11   # H heads of the op against the simulator shadow
# 2. klcheck M7, Qwen3.8 2B, 8 prompts x 64 tokens: ratio <= 3, top-1 >= 80 % (simulator: ratio 1.103, top-1 97.27 %)
TSR_XCLBIN=$X build/xrt/bin/klcheck all -m models/qwen3.8-2b-empero/Qwen3.8-2B-Q4_K_M.gguf --device xrt --phase m7 -n 64 --n-prompts 8 -o build/card/gdnfw/q2b-m7
# 3. 1 x 16 replay on the simulator, which has CAPS bit 8 by default (do not set TSR_SIM_GDN=0): every row must be bit-identical
TSR_XCLBIN=$X build/xrt/bin/klcheck run -m models/qwen3.8-2b-empero/Qwen3.8-2B-Q4_K_M.gguf --device xrt --phase m7 -n 16 --n-prompts 1 --out build/card/gdnfw/dev16.klc
TSR_SIM_TMAX=4 build/xrt/bin/klcheck run -m models/qwen3.8-2b-empero/Qwen3.8-2B-Q4_K_M.gguf --device sim --phase m7 --tokens build/card/gdnfw/dev16.klc --out build/card/gdnfw/sim16.klc
build/xrt/bin/klcheck compare --ref build/card/gdnfw/sim16.klc --dev build/card/gdnfw/dev16.klc --expect-identical
#    the same for the 8 x 64 run: --tokens build/card/gdnfw/q2b-m7/dev.klc (512 rows)
# 4. speed: decode and per-unit busy cycles (TSR_STATS=1), as for the 2026-11-09 runs
TSR_XCLBIN=$X TSR_STATS=1 build/xrt/bin/llama-completion -m models/qwen3.8-2b-empero/Qwen3.8-2B-Q4_K_M.gguf -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -c 4096 --temp 0 -no-cnv -n 65 -p "The history of the printing press begins"
```

What to expect (card-configuration model, `TSR_SIM_CARD=k4full TSR_SIM_GDN=1 TSR_SIM_RV_RATIO=0.6667`, per decode token of the 2B): 8.04 M cycles with the state on its cell, 9.23 M with
`TSR_GDN_CELL=0`, against the card's 12.63 M for the composed three-pass run at 200 MHz. VU busy 1.26 M against the card's 4.76 M, DMA busy 1.25 M against 2.43 M. At 250 MHz that is 32 ms and
about 31 tok/s (36.9 ms and 27 tok/s without the cell), against 15.66 tok/s of the composed run at 200 MHz. Check the VU busy first (it should fall to about a quarter) and the DMA busy second (about half). If the
bits are not identical to the simulator, the first suspects are the op itself (id 11 shows it alone), then the in-place state tile (the odd heads of id 11
update in place) and the s-register ordering of the `SOP` pair before each `GDN`. For a record-level difference use `TSR_DUMP_RE` on the nodes
`attn_output-<layer>` against a `TSR_SIM_GDN=1` simulator run.

## Qwen3-30B-A3B on the simulator and the card (SW-77, November 9)

Done on a branch from `sw-moegdn` (2d887ec), now merged. No RTL, firmware, backend or llama.cpp change: llama.cpp's `qwen3moe` graph (the same `build_moe_ffn`
with softmax gating and normalised weights) lowers with the code of the Gemma 4 work. What differs from Gemma 4 is in
[design/06-software-verification.md](design/06-software-verification.md) §7.9, SW-77 (no scale chain, no router input norm, gate and up as two tensors
with SWIGLU, Q of 4,096 for a hidden size of 2,048 with per-head Q and K norms). The new things are tests and records:

- Tests: `test-tsr-moe` has the Qwen3 layer next to the Gemma 4 one (T = 1 to 256 since SW-76, four kind configurations); `ggml_tesseract_graph_qwen3moe_tiny` and
  `_q4k` run a 2-layer model of the same graph (`gen_tiny_models.py`, 64 experts) through klcheck with the M7 settings. `ctest -LE slow`: 314 of 314.
- Model: `Qwen/Qwen3-30B-A3B-GGUF`, `Qwen3-30B-A3B-Q4_K_M.gguf`, 18,556,685,824 bytes, sha256
  `0d003f6662faee786ed5da3e31b29c978de5ae5d275c8794c606a7f3c01aa8f5` (`models/qwen3-30b-a3b/`, not in git). The Instruct-2507 version has no Q4_K_M at
  `Qwen` or `ggml-org`, so this is the original model.
- klcheck M7, 8 prompts x 64 tokens ([klcheck.md](klcheck.md)): simulator ratio 0.813, top-1 96.68 %, PASS; card the same to every digit. The card's 512 rows are bit-identical to the
  simulator's; a 1 x 16 card run replayed on the simulator: 16 of 16.
- Speed on the card, M7 ([performance.md](performance.md) "Qwen3-30B-A3B on the card"): decode 15.82 tok/s best, 13.06 median (11.338 M device cycles per token,
  17.6 tok/s from device time alone); a 512-token prefill at `-ub 64` 26.30 tok/s best, 26.04 median. The host load average was 23 to 58 during the runs.
- Prefill at `-ub` 128 and above used to stop at layer 0 (`MOE_ROUTE: ... its outputs overlap the logits of a later tile`), earlier than Gemma 4 (which ran at 128).
  SW-76 fixed it (merged): the 512-token prefill at the default `-ub 512` gives 26.72 tok/s best and 26.60 median on the card, with 513 of 513 rows
  bit-identical to the simulator, and the Qwen3 layer test runs T = 65, 70, 130 and 256 (70 aborted before SW-76).
  Open: grouping of tokens by expert (SW-66) is still not built. The card's VU and DMA counters were not compared with the simulator's for this model (the corrected cycle model of SW-80 was checked on Qwen3.8 2B and Gemma 4 only).

## Why the card spent more cycles than the simulator (SW-80, November 9)

Software only; no RTL change. The card took 12.63 M cycles per Qwen3.8 2B decode token and 16.32 M per Gemma 4 26B-A4B token; the simulator said 10.5 M
and 15.1 M. Op-by-op comparison of the card's trace-writer records with the simulator's event log ([prof-dbg.md](prof-dbg.md) §3,
`sw/tools/trw_vs_sim.py`) found two errors of the simulator's cycle model, both fixed (SW-80), and no stall in the hardware:

- The VU of the card has 8 / 8 / 4 lanes behind two register stages; the model had 16 / 16 / 8, and latencies of 20 and 64 cycles where the RTL takes 59 and 95
  (at 16 lanes too) and a reduction ring it did not know. 89 % of the Qwen3.8 gap is the 18 GATED_DELTA_NET records. The RTL unit alone takes 4.737 M cycles for
  the 4,518 VU ops of the token and `VU_CYCLES` reads 4.758 M on the card: the card's VU waits for nothing the unit does not wait for in the RTL bench.
- QACT A32 takes 16 cycles per 32-block (divider bound); the model had one cycle per line. That is 43 % of the Gemma gap (`MVM_WAIT`).

Corrected, the simulator in the card configuration (`TSR_SIM_CARD=k4full`) gives 12.65 M and 16.24 M (0.2 % above, 0.4 % below the card), and every
launch of both runs, prompt and warm-up included, agrees to 0.7 %. Outputs are bit-identical before and after (`klcheck compare --expect-identical`, 2 x 16
tokens each). The default simulator configuration is still the U200 image's 16 lanes. Numbers: [performance.md](performance.md) "Card against simulator",
[design/02-vector-unit.md](design/02-vector-unit.md) §7.13 (it also lists three RTL changes that would make the VU faster: a ring of 32 partial entries
removes 0.6 M of the 12.63 M cycles of the Qwen3.8 token).
Open: the KV store (the V plane's masked writes), MOE_ROUTE and the DMA's half rate while the VU runs are not modelled (together 0.4 % of a Gemma token);
independent VU ops are charged as dependent ones (0.9 % high on the Qwen3.8 token); ROPE with T > 2 and rotation as long as the angles is up to 7 % low.

## Card runs of the merged firmware (branch `sw-moegdn`, November 9)

The K4 full-feature image of the fourth link (bitstream unchanged) ran the `tsr-fw.elf` of `sw-moegdn` before `gdn-fw` (21 kinds, kind table 0x804f) with a
host build of the branch (`build/xrt`). Results are in [klcheck.md](klcheck.md) (last section) and [performance.md](performance.md) ("MoE and
Gated DeltaNet on the card"):

- Gemma 4 26B-A4B: M6 ratio 1.294, top-1 92.77 %; M7 ratio 0.742, top-1 94.53 % (8 x 64, PASS). Both give 512 of 512 rows bit-identical to the
  simulator run of the `moe` branch. M7 1 x 16 replay on the simulator: 16 of 16 rows. The `JOIN` inside the op stream works on the card.
- Qwen3.8 2B M7: ratio 1.061, top-1 98.05 %; 512 of 512 rows bit-identical to a simulator replay, and 16 of 16 for 1 x 16.
- Speed (M7): Gemma decode 11.99 tok/s (16.32 M cycles per token), Qwen3.8 15.66 tok/s (12.63 M). A 512-token prefill runs at 26.56 tok/s (Gemma,
  the default `-ub 512`; 26.2 at `-ub 64`) and 33.0 tok/s (Qwen3.8).
- Gemma prefill at `-ub 256` and above used to stop in the host backend (MOE_ROUTE with several tiles whose outputs overlapped a later tile's logits,
  SW-64). SW-76 fixes it: a route of more than 64 tokens writes its ids and weights to a device scratch and two CPY records move them
  ([design/06-software-verification.md](design/06-software-verification.md) §7.9; host only, no firmware or kind change). The default `-ub 512` and
  `-ub 2048 -b 2048` run on the card. Qwen3-30B-A3B (section above, which hit the same refusal at `-ub 128`) runs its 512-token prefill at the default `-ub 512` with
  this fix: 26.60 tok/s median on the card, 513 of 513 rows bit-identical to the simulator, and the simulator's `-ub 512` and `-ub 64` logits are bit-identical.
  The Qwen3 layer test in `test-tsr-moe` covers 65 to 256 tokens.
  The card's VU busy cycles for Gated DeltaNet were 1.7 times the simulator's. The cause was the simulator's cycle model, not the card (next section, SW-80): in the
  card configuration the simulator now agrees with the card to 0.2 % (Qwen3.8 2B) and 0.4 % (Gemma 4 26B-A4B).

## Resume here for the software: MoE and Gated DeltaNet merged (branch `sw-moegdn`, November 9)

Software only; no RTL change. The card ran both models on 2026-11-09 (the section above). The branch merges `moe` (7c4088e) and then `gdn-sw` (079b44c) onto `k4full`
(c1f3aaf). Both merges are real merges; neither source branch was rewritten. The two sections below describe the branches as
they were before the merge.

- Decision numbers: MoE keeps SW-60 to SW-66 and section 7.9 of [design/06-software-verification.md](design/06-software-verification.md).
  The Gated DeltaNet decisions SW-60 to SW-67 became SW-67 to SW-74 (same order) and its section 7.9 became 7.10; the citations in
  docs, code comments and test comments were changed with them. SW-75 is new: the ITCM budget. SW-76 (the outputs of a MOE_ROUTE with several tiles, a host
  change in section 7.9) and SW-77 (the Qwen3 MoE graph, section 7.9) follow it.
- Kinds: MOE_ROUTE 0x1E, MUL_MAT_ID 0x1F, SOFTPLUS 0x2D, SSM_CONV 0x2E, GATED_DELTA_NET 0x2F, all at numerics revision 1, with
  UNARY func 5 (SIGMOID). 21 kinds, kind-table signature 0x804f (0x2381 since `gdn-fw`). The `fw_kind_revs` digests of the five kinds are the digests of their
  branches (the two corpora start from the same PRNG state, so neither moves the other).
- Firmware image: ITCM 84,056 B, DTCM 50,088 B (85,504 B and 50,616 B with the fused GDN path of `gdn-fw`). The old 80 KiB ITCM budget (SW-38) is exceeded by 2,136 B, so SW-75 raises it to
  96 KiB (98,304 B; the upper 48 KiB that SW-38 kept for M8 MoE / M9 are now 32 KiB). Each branch alone was below 80 KiB (78,520 B and
  78,576 B). No feature was removed. The `fw_elf` BIG_TEXT image grew to 98 KiB of code so that it is still over the budget.
- Backend: the MoE chains (router group, scale group, MUL_MAT_ID) and the Gated DeltaNet groups (SSM_CONV with its CONCAT / state CPY / SILU,
  GATED_DELTA_NET, the state caches as device windows) recognise different nodes. Their fusion passes run one after the other and
  neither takes a node the other owns. A model that has both (a Qwen3.5 MoE) is untested.
- Checks: `ctest -LE slow` passes (the two `tools_selftest_mock_traces` tests need the git-ignored trace data of `traces/`; with it they
  pass), `gen_tables.py --check` and `gen_tops.py --check` pass. `rtl/lint.sh` and `tb/run_all.sh` were not run: the branch changes no
  file under `rtl/`, `tb/` or `hw/` against `k4full`.
- Bit identity on the simulator, M7 settings, 2 prompts x 16 tokens, `klcheck compare --expect-identical`: Gemma 4 26B-A4B against the
  `moe` branch build, 32 of 32 rows; Qwen3.8 2B against the `gdn-sw` branch build, 32 of 32 rows.
- Card: see the first section.

## Resume here for MoE: Gemma 4 26B-A4B runs on the simulator (branch `moe`, November 8, night)

Software-only work on the simulator; no RTL change, no card. MoE selection is software on the control core and the TOPK RTL
stays off. The design is in [design/06-software-verification.md](design/06-software-verification.md) §7.9 (SW-60 to SW-66)
and [spec.md](spec.md) §13.

What exists (all in `sw/`, all checked against the C simulator or the CPU):

- Reference: `tsr_ref_moe_route` in `sw/sim/tsr_ref_vu.c` (softmax by the VU references, top-k in the VU TOPK order: larger value
  first, equal values by the smaller index, NaN first, then the renormalised weights by the tree sum and RC). Test `ref_moe`;
  it also measures ggml-cpu's `std::sort` on ties and NaN (SW-63).
- Firmware kinds: MOE_ROUTE **0x1E** and MUL_MAT_ID **0x1F** (`sw/ctrl/kernels/k_moe.cpp`, planners `tsr/plan_moe.hpp`, record words
  `opl::moer` / `opl::mmid`). On its own branch: 18 kinds, kind-table signature 0xC341, ITCM 78,520 B of the then 81,920 B budget (the merged image is in the section "MoE and Gated DeltaNet merged" above). Tests
  `fw_kernels_moe_rv`, `fw_kernels_moe_rv_errors`, `fw_kernels_moe_host` (7,000 rows against the reference), `fw_kind_revs`, `fw_m6_e2e`.
- Backend: 3D expert tensors placed with `EXP_STRIDE` (SW-60), `MUL_MAT_ID` lowering, fusion of the router chain into one MOE_ROUTE
  record and of the scale chain into the down MUL_MAT_ID (SW-64). Tests `ggml_tesseract_moe*` (the graph of `build_moe_ffn` in
  four kind configurations against the CPU, including the scale table in host memory as Gemma 4 has it) and
  `ggml_tesseract_backend_ops_mul_mat_id` (928 of 945 cases of the pin's `test-backend-ops`; 17 refused).
- `klcheck` / `tsr-numcheck`: `--phase m6` now sets `TSR_KINDS=m6,+MUL_MAT_ID`.

Kind numbers: the MoE kinds took the lowest free numbers, 0x1E and 0x1F. The Gated DeltaNet kinds took 0x2D to 0x2F, so the two
sets do not meet; 0x20 to 0x2C are free.

Model: `bartowski/google_gemma-4-26B-A4B-it-GGUF`, `google_gemma-4-26B-A4B-it-Q4_K_M.gguf`, 17,035,039,872 bytes, sha256
`a07f72221e8e3f77455ab0d7f7652d01a9f63c262b954aa6932a53275a0e895a` (in `models/gemma4-26b-a4b/`, not in git). The pinned llama.cpp
loads it on the CPU; no patch of `third_party/` was needed.

Results on the simulator (`klcheck all`, 8 prompts x 64 tokens, three floors; [klcheck.md](klcheck.md)):

- M6 (CPU selects, device runs the matrix products): ratio **1.294**, top-1 **92.77 %**.
- M7 (router, top-k and experts in the one launch per token): ratio **0.742**, top-1 **94.53 %**.
- The floor of this model is large (mean KL 0.17 to 0.19 for a one-ulp change); the medians are 5e-4. Use 8 x 64: a run of
  2 x 16 fails the top-1 limit (78 %) with a floor of 83 % on the same rows.
- Device cycles per decode token: 14.34 M (M6) and 15.14 M (M7), 13.08 M beats per channel; 13.2 tok/s at 200 MHz, 18.2 at 276.3 MHz
  (device time only, an estimate for the card, [performance.md](performance.md)). These are the simulator's figures before SW-80; the card takes 16.32 M for M7
  and the corrected simulator 16.24 M in the card configuration.
- Generate and replay on the simulator are bit-identical: the generated tokens of 3 prompts x 16 tokens replayed on the device
  (`klcheck run --tokens`, one token per step) give 48 of 48 rows with `compare --expect-identical`, in M6 and in M7.

What a card run needs:

1. The new `tsr-fw.elf` (a firmware change; the K4 bitstream is unchanged: MOE_ROUTE needs `CAPS_VU`; MUL_MAT_ID needs it
   only with a scale table) and the host stack of this branch. The expert weights are 17.18 GB of the 64 GB.
2. The first M6 step with `TSR_KINDS=m6,+MUL_MAT_ID`, then M7. Both kinds wait with `JOIN` inside the op stream and read the SPM
   window with the RISC-V inside the handler (SW §7.9, "Joins"): no earlier kind does that. If the card differs from the simulator,
   look at the joins first (the ids chunk of MUL_MAT_ID, the two joins per tile of MOE_ROUTE).
3. The card's top-1 and KL are compared with the simulator's by `klcheck compare --expect-identical` on a short run (2 x 16 is
   enough for identity; the KL limits need 8 x 64).

Left open:

- Prefill runs one MVM of T = 1 per (token, slot), with no expert reuse across tokens (SW-66). Grouping by expert is designed, not
  built; it pays from about 64 tokens.
- Only the graph of Gemma 4 is fused: softmax gating, top-k, weights divided by their sum. Sigmoid gating, a router bias, group
  selection and softmax-after-top-k graphs run the router on the CPU; the experts still run on the device.
- The clamp of the weight sum is not computed (it cannot act: SW-61). A NaN in the logits gives NaN weights, as in ggml.
- Ties at the 8th place of the router select the smaller index; ggml's choice is unspecified (SW-63).

## Gated DeltaNet in software (branch `gdn-sw`, November 9)

The Qwen3.5 / 3.8 layer graph (3 Gated DeltaNet layers to 1 attention layer) runs on the device through the simulator, with no
RTL change and no llama.cpp patch. Design, decisions SW-67 to SW-74 and the numbers: [design/06-software-verification.md](design/06-software-verification.md)
§7.10, [klcheck.md](klcheck.md), [performance.md](performance.md) "Gated DeltaNet on the simulator".

What is in:

- Kinds SOFTPLUS 0x2D, SSM_CONV 0x2E, GATED_DELTA_NET 0x2F (numerics revision 1, `sw/ctrl/kernels/gdn_kinds.cpp`, planners in
  `tsr/plan_gdn.hpp`, oracle in `sw/ctrl/ref/gdn_ref.cpp`). UNARY takes SIGMOID. On its own branch: ITCM 78,576 B of the then 81,920 B budget (the merged image is in the section "MoE and Gated DeltaNet merged" above).
- Backend lowering in `sw/ggml-tesseract` (`tsr-build.cpp`: `tsr_lower_gdn`, `tsr_lower_conv_group`, `tsr_lower_conv_window`,
  `tsr_lower_unary`, the per-token `tsr_lower_rms`). The recurrent caches are device windows; with `-nkvo` the ops that touch
  them stay on the CPU.
- Tests: `fw_gdn_rv`, `fw_gdn_ggml`, `fw_kind_revs`, `ggml_tesseract_units` (lowering cases), `ggml_tesseract_graph_qwen35_tiny` /
  `_s128`, `ggml_tesseract_gdn_logits_*`. The tiny models come from `gen_tiny_models.py`.
- Models in `models/` (git-ignored): `qwen3.8-2b-empero/Qwen3.8-2B-Q4_K_M.gguf` (empero-ai/Qwen3.8-2B-Distill-GGUF, sha256
  `4aa0fb13c431514262f259d420ecc95a8714df58ac2a2384514e20b93983f0ff`), `qwen3.8-27b-ggml-org/Qwen3.8-27B-Q4_K_M.gguf`
  (ggml-org/Qwen3.8-27B-GGUF, 19,403,684,480 B, sha256 `d15d4b6de55f662ac40e867702d31afe738d032fa06b829b2b1a21a49d4a48ee`) and
  truncated copies of the 27B made for the cost extrapolation (`sw/tools/gguf_truncate.py`, N = 4 and 8; deleted again for disk).

Results (simulator): the 2B at klcheck ratio 1.06 (512 positions, top-1 98.05 %); a 4,127-token prompt at ratio 0.99; the 27B's
first 4 layers at ratio 1.06. A 27B decode token is about 108 M core cycles (542 ms at 200 MHz), the composed recurrence 19.7 M
of them (98 ms), about 4.1 times the fused op's 23.8 ms estimate.

Two bugs found on the way, both worth remembering:

- ggml-alloc frees the inputs of a `CONCAT` after it. A record that reads them later reads reused memory. The SSM_CONV group now
  builds its window at the CONCAT (SW-71). The tiny model's klcheck did not see it (mean KL ratio 1.2 with a convolution that
  returned zeros), so `test-tsr-gdn-logits` checks the logits of every step.
- klcheck's embd / floor path gave an M-RoPE model one position per token where llama.cpp reads four. The floor of Qwen3.8 was
  inflated by that (ratio 0.35 instead of 1.06). Fixed in `sw/tools/klcheck.cpp`.

Open:

- The whole 27B was not simulated (the file-backed DRAM needs about 22 GB and the free disk was near the 60 GB rule). Its row
  extrapolates the first 4 and 8 layers. Run `klcheck all` on it with `TSR_SIM_DRAM_DIR` when the disk allows.
- More than one sequence per ubatch, `K` > 1 snapshots (MTP), vision positions and the KDA gate stay on the CPU or are untested end
  to end. A card run was not made.
- Cost ideas, not done: a record that updates the cache cell in place (three DRAM passes per layer become one), and a reduction
  that does not wait on the 16-cycle `RED.SUM`. `TSR_DUMP_RE` / `TSR_DUMP_DIR` dump device node values for the next debugging
  session.
- `docs/design/02-vector-unit.md` §7.12 (the fused op, 23.8 ms) is on branch `gdn-vu`; this branch cites it before the merge.

## Branch `gdn-vu`: the GDN op is in the vector unit (November 9)

The fused Gated DeltaNet state update (opcode 0x39, `HAS_GDN`, `CAPS` bit 8) is integrated in `tsr_vu` and in the `tsr_u200_k4` top
(`HAS_GDN` 1; `CAPS` reads 0x121). Design and numbers: [design/02-vector-unit.md](design/02-vector-unit.md) §7.12, decisions VU-28 to
VU-37. The element update runs on the front lanes (micro-op U_ROT); the engine keeps the DSP dot array, the converter, the setup
engine, the scalar chain and the URAM buffers, and is inside `pblock_tsr_vu`. The Pblocks are unchanged. No link was run.

- Oracle and tables: `sw/sim/tsr_ref_gdn.c`; opcode, `CAPS` bit 8, encoding and limits (m 64 to 256 in steps of 16, rows 1 to 1,024) in
  `sw/tables/tsr_tables.py` with the generated files; the simulator's VU unit executes the op and the checker class gives the
  same codes, `ERR_INFO` and extents (d W, a R, k R, o W).
- With `HAS_GDN` 0 nothing of the op is built; every existing VU test passes unchanged with `HAS_GDN` 1 as well (`tb/vu`).
- Out of context, K4 VU, after route, 4.0 ns: +5.0 K LUT, +40 DSP, +34 URAM, no RAMB; WNS +0.262 ns (a path of the unchanged
  unit), every engine path at least +0.46 ns. Throughput: heads run back to back at (rows + 1) x m / 8 cycles, 19.0 ms per token for the
  planned 48 x 48 heads of 128 x 128 at 250 MHz.
- Card test hook: `selftest` kernel id 11 (`tsr-selftest --ids 11`), only with `CAPS` bit 8.
- For the 250 MHz link branch (`k4f250`): merge `gdn-vu`; the k4 top gets `HAS_GDN` 1, `CAPS` reads 0x121, `tsr-selftest` has a new id 11
  (counts change on a device with the bit), the simulator's implemented `CAPS` includes bit 8, and the VU grows by 5.0 K LUT. A
  link has to show that the VU region still routes at 4.0 ns. `HAS_GDN` 0 in `gen_tops.py` gives back the earlier top.
- Next: the `GDN_STATE` firmware kind (`LD`, op, `ST`, double-buffered over the heads), the lowering of `GGML_OP_GATED_DELTA_NET`, the
  surrounding ops (conv1d, norms, gates), then a long-sequence comparison with llama.cpp (VU §7.12.11).

## Resume here: the full-feature image on the card, everything passes (branch `k4full`, November 8, evening)

The image of the fourth link (`build/link/k4full/tsr_u200_k4.xclbin`, source `d33f816`, 76,107,483 bytes, sha256
`6fdf31fdee80778771644fdc7708041c66070fd2ed8f0b1f753c7f66d3745284`, UUID `c7f70219-f070-c4fd-cfdc-1da14a7eb251`, 200 / 100 MHz, WNS
+0.017 ns, WHS +0.009 ns, SLR CLBs 80.1 / 86.3 / 85.0 %) has the ATTPV fix and the link-3 stages. Everything checked on the card passes,
and attention on the device is bit-identical to the simulator. Numbers and method: [platform-u200.md](platform-u200.md) §8.8 "Card
bring-up of the fourth link", [performance.md](performance.md) "The full-feature image after the ATTPV fix",
[klcheck.md](klcheck.md). Logs and the run scripts: `build/card/link4` (not in git).

What passed (all bit-identical to the C simulator at `T_MAX` 4 and `CAPS` 0x21 unless a KL is given):

- `tsr-selftest`: id 5 passes (184 of 184 results) from the second run after a load on; the first run after a load fails id 5 (below; the
  cause is an RTL reset bug, fixed in the source).
- Trace replays through the `replay` kernel: `m2` 123, `gen` 26, `m3` 171, `vu` 196, `at` 98 (was 75 and 23 failed), `ctrl` 131; 745
  in all, 0 failed.
- M6: 0.5B 8 x 64 ratio 1.077, 512 of 512 rows; 32B 2 x 16 ratio 0.750, 32 of 32 (the figures of the second link, every digit).
- Vector-unit kinds on the device (`--offload-kqv off`): 0.5B ratio 1.127, 512 of 512; 32B ratio 0.777, 32 of 32.
- M7, attention and KV cache on the device, one launch per token: 0.5B KL 2.2216e-3, ratio 0.953 (0.914 against seed 1, the
  simulator's own figure), top-1 98.05 %, 512 of 512 rows; 32B KL 1.3104e-3, ratio 0.735, top-1 100 %, 32 of 32.
- Decode, best run: M7 0.5B 75.88 tok/s (13.18 ms, 97 % device time), 32B 2.35 tok/s (424.9 ms); vector-unit kinds 68.45 and 2.32;
  `m6` 47.83 and 2.25. Prefill of 158 tokens: 0.5B 407 (`m6`), 384 (vector-unit kinds), 338 (M7) tok/s; 32B 9.5, 9.5, 9.3. The
  device counters per decode token equal the second link's to four digits: the extra `XSLR_REG` stages cost nothing visible.
- Trace writer (`tsr-selftest --trw 16`): 647 replays 16,247 records, 745 replays 18,207 records, 0 dropped, no hang. GDB smoke
  passes (halt, registers, breakpoints, step, memory, hardware breakpoint in a kernel, watchpoint, `finish`, history, trace writer).

Open:

- The first `selftest` after an image load fails id 5 in two lines (area lines 531 and 532, the causal ATTQK score block, read
  back as zero). **Cause found and fixed in the source (branch `k4full`, not in the image).** The ATTPV output stage of every
  cluster (`tsr_mvm_pv`, `ob_zero_q`) was reset to `ng == 0`, and `ng` is a setup flop that is 0 after configuration. After a load
  every PV unit held a zero-line job with no op running. The first ATTPV let the gather take about five of these jobs per cluster
  before its own start reached the unit; they were written to lines of the setup flops of the ATTQK before (here SPM lines 0, 530,
  531 and lines of `parts`). The start of the unit and the end of an op clear the stage, so only the first ATTPV after a load is hit.
  Verilator starts flops at random values, so no bench saw it. With every flop at zero (`TSR_RAND_RESET=0`) the packaged core
  reproduces the card exactly (area lines 531 and 532, first launch only). A card run of a copy of the kernel that stores SPM
  lines 0 to 3 back showed the extra zero line at SPM line 0 on the first run of 3 of 3 fresh loads, as the simulation predicts.
  Details: [platform-u200.md](platform-u200.md) §8.8 and §14.5. Tests: `tb/at` `test_at_mvm_zero_state`, `tb/core`
  `test_k4_selftest_first_run`. **Needs a relink to reach the card** (the user decides). Until then a first ATTPV after a load can
  write about 20 zero lines to SPM lines picked by stale setup flops; the M7 runs after a load were bit-exact so far.
  Not the cause (ruled out): the missing dependency of the causal ATTQK in `t_at` (the C simulator with 120 random schedules and
  the race detector fatal passes; the ops are in order on the MVM unit and the final `ST` waits for the ATTPV through the AMERGE
  chain). Reproduce on the card: load another image, then `TSR_XCLBIN=<k4full> tsr-selftest --device xrt --no-mem --ids 5`.
- The DB §8.7 self-check of the trace writer still reports differences from the PERF counters (RETIRED 14,432 against 14,737 op records
  in the 647 replays; the same numbers on both images; DMA, QACT, VU `RUN` sums 0.2 to 5.8 % above the counters). No result is wrong.
- Prefill is slower than on the host at `T_MAX` 4 for the vector-unit kinds and for attention on the device (0.5B 407 to 338 tok/s).
- `tsr-selftest` counts 184 results in a passing run. The log of the second link's run said 184 passed and 1 failed because that
  run used `--log-out FILE`, which adds one check (the log words were written). With `--log-out` a passing run counts 185 (checked on
  the card). No test went missing.
- Branch `k4full` is not merged into the main checkout's branch (`codex/k4-route-fixes`, which has the older read-modify-write KVST).

## Earlier: the full-feature image on the card, ATTPV fault found (second link, branch `k4full`, November 8, morning)

The image of the second link (sha256 `20f25925…22e7`, 200 / 100 MHz, timing met:
WNS +0.003 ns, WHS +0.006 ns) ran on the card. Everything checked passes except attention on the device. Numbers and method:
[platform-u200.md](platform-u200.md) §8.8 "Card bring-up", [performance.md](performance.md) "The full-feature image on the card".

What passed (all bit-identical to the C simulator at `T_MAX` 4 and `CAPS` 0x21 unless a KL is given):

- `tsr-selftest`: identity (`CAPS` 0x21, `T_MAX` 4, `RV_FREQ` ratio 0.5), memory, boot (8.4 ms cold, 2.5 ms warm), `PING` 6.53 µs,
  `selftest` kernel ids 1 to 4 and 6 to 10 and the misuse cases (184 checks). Id 5 (attention) failed, below.
- Trace replays through the `replay` kernel: `m2` 123, `gen` 26, `m3` 171, `vu` 196, `ctrl` 131, 0 failed. `at`: 75 passed, 23 failed.
- Trace writer: `tsr-selftest --trw 16` (new) ran the kernel tests and 647 replays with the writer on: 16,247 records, 0 dropped, no
  hang on channel 0 (the `eea134e` fix). The DB §8.7 PERF self-check differs by 0.2 to 5 % (one RETIRED strobe lost in some `m2`
  replays; open, nothing wrong in a result).
- M6: 0.5B 8 x 64 KL 2.5947e-3, ratio 1.077, 512 of 512 rows bit-identical; 32B 2 x 16 ratio 0.750, 32 of 32 bit-identical.
- Vector-unit kinds on the device (`--phase m7 --offload-kqv off`): 0.5B ratio 1.127, 512 of 512 rows bit-identical, decode 69.6
  tok/s (25 launches per token, was 145); 32B ratio 0.777, 32 of 32 bit-identical, 2.33 tok/s.
- GDB smoke over `tsr-dbg` on the card: halt, registers, breakpoints, step, memory write, a hardware breakpoint in a kernel,
  watchpoint, `finish`, history (512 entries), trace writer commands, detach.
- Decode, M6 settings: 0.5B 49.5 tok/s, 32B 2.27 tok/s (the cycles per token equal `k4m200`'s); prefill 410.8 and 9.6 tok/s
  (`T_MAX` 4 takes four tokens per weight pass, `k4m200` eight). Launch round trip 16 µs beyond the device time.

The ATTPV fault and its cause (details: §8.8 "The ATTPV fault"):

- On the card every ATTPV that executes returns, in each group of four output lines (a 64-dim slice), lines 2 and 3 equal to lines 0
  and 1. M7 with attention gives KL ratio 525 (0.5B), the first differing node of a node dump is the first `FLASH_ATTN_EXT`, dims 32 to
  63 of every head equal the simulator's dims 0 to 31.
- The RTL is right: Verilator replays of the failing traces on the packaged K4 core (Xilinx wrappers, `XSLR_REG` 1, two-flit gather,
  production firmware on the RTL Ibex), `test_at_mvm_u200_flits2` and `test_at_mvm_u200_xilinx` pass. Vivado 2024.2 synthesised
  `automatic int L = 8 * int'(dr_l_q) + k` / `cb_a[t][6'(L)]` in `tsr_mvm_pv.sv` without the top bit of `dr_l_q` (the kernel netlist
  had 4,864 `cb_a` registers instead of 9,728). A netlist simulation of the select gets 1,977 of 4,000 steps wrong.
- The Route 35-3360 RAMs (30 `RAMB36`, `CLOCK_DOMAINS` `COMMON`) are not the cause: M6 and the 647 non-attention replays are exact,
  the PV FIFO RAMs are flagged in cluster 0 only and all clusters fail alike.
- Fix (`f06fefc`): `tsr_mvm_pv_lsel` (lane `{l, k}` as a concatenation) in `tsr_mvm_pv.sv`; the same form in `tsr_dbg.sv`. `tb/at/test_at_pv_netlist.py`
  (opt-in, Vivado) is the regression test: the module 0 of 4,000 wrong, the old form (control) 1,977. Out-of-context synthesis of
  `tsr_mvm_pv`: 106 loads on `dr_l_q[2]`, 9,728 `cb_a` registers.

## Earlier: the full-feature image, source ready to link (branch `k4full`, November 7)

Branch `k4full` (worktree `/home/limagin-dev/tesseract-k4full`, from `07fbf33` of `k4m200`) makes `tsr_u200_k4` the full-feature card
image: `NUM_CH` 4, `T_MAX` 4, VU 8 / 8 / 4 with `VU_LANE_PIPE` 2, attention with masked KVST, paired two-flit gather, DDR0 / DDR3
boundaries, RV transport, trace writer and history, no local-intermediate banks (`CAPS` 0x21), core 200 MHz, `clk_rv` 100 MHz. No
link was run. [platform-u200.md](platform-u200.md) §8.8 has the parameters, the Pblocks with the map
(`docs/fig/floorplan/k4full-pblock-plan.png`), the reasons, the checks and the launch command. Short form:

- Pblocks: `pblock_tsr_ctrl_plane` as in `k4m200` plus the two checker classes; the whole VU in a new child of SLR2,
  `pblock_tsr_vu` (`CLOCKREGION_X0Y10:X2Y14`, off the DDR3 controller); clusters, assemblers, pair sources, bridges and arbiters of
  channels 0 / 3 in SLR0 / SLR2; the hub in SLR1. The legality check on the platform passed (`tb/platform/viv/k4_pblock_check.tcl`).
- Masked KVST was already in the base (from `aadf92c`); the main checkout has the older read-modify-write form, so nothing was
  ported from it. `LOCAL_SCALE` and `LOCAL_SOFTCAP` are off with the local banks.
- New: `XSLR_REG` (two register stages on every link between the control plane in SLR0 and the VU in SLR2, the CSR strobes, the
  trace writer's request, data and response). The first out-of-context placement with hard Pblocks had WNS -8.2 ns at 3.333 ns,
  from the VU's combinational `cmd_ready` into the issue decision and the other VU links. With `XSLR_REG`: WNS -0.686 ns, TNS
  -1,392, 9,370 failing endpoints, no path group across two SLRs (the reduced image's run: -0.786 ns, 448 endpoints). The
  full table is in §8.8. Out-of-context Pblocks are soft by default; `TSR_SLR_HARD=1` makes them hard as in the link.
- Read first in the link: the placed SLR1 occupancy (the estimate with the shell is 91.8 to 93.4 %, the target 93 %), the SLR1
  congestion windows (32x32 or smaller) and the router's column peaks. The kernel adds about 4.4 K SLLs at SLR0-SLR1 and 3.9 K at
  SLR1-SLR2 over the reduced image. If a placement gate misses, do not spend the route.
- Cost of `XSLR_REG`: the VU starts a command 3 cycles later and completes 4 later (it was 1); s registers 3 cycles later (2, 3 and 2
  before the third link); trace bursts a little slower; +1.3 K LUTs, +3.2 K flops, about +1 K CLBs in SLR1 (the stages sit in SLR1;
  measured with two stages).
- First link (200 / 100 MHz): routed legally, `clk_kernel_00` missed by 0.064 ns on 160 endpoints (list:
  `build/k4full/postroute_exp/fail_before_clk_kernel_00_unbuffered_net.tsv`). 82 of them cross an SLR boundary, 78 do not. Two
  crossings got `XSLR_REG` stages: the MVM attention-stall strobe (two per cluster, `tsr_mvm`; only `ATT_PATH_STALL` shifts, by 2
  cycles) and the RISC-V SPM request (one, in `u_spm_tx`; an SPM access of the RISC-V starts a cycle later). Left alone: the reset
  copies into `u_issue` and the gather (phys_opt moves loads between equal `keep` copies; `DONT_TOUCH` is untried, platform-u200.md §12),
  the DDR0 / DDR3 slice outputs into the arbiters, the pair-link flit mux into the receiver, the result-pipe slices that
  straddle SLR0 / SLR1, the VU write queue into the SPM input registers, and the 78 same-SLR paths (DMA write slices, SPM
  grant into the DMA, cluster-internal paths).
- Source also fixed: the twelve testbench-HDL lint failures and the C16 `WIDTHCONCAT` of the checkpoint. New tests:
  `test_core_k4full`, `test_core_m7_rv_local`, `test_k4_full_graph` (needs `TSR_K4_TRACE`; graph traces are in
  `build/analysis/local_intermediates/` of `tesseract-local-intermediates-20261105`).
- Setup of this worktree: `third_party` holds local clones at the pins (llama.cpp patched by `apply-patches.sh`), `build/tools`
  links riscv and svgvenv from the main tree, and `traces/` has the main tree's trace files as symlinks (untracked, so the two
  `tools_selftest_mock_traces` cases have data).

## Earlier: the 200 / 100 MHz image on the card, M6 (branch `k4m200`, November 6, evening)

The `k4m200` image (`build/link/k4m200/tsr_u200_k4.xclbin`, sha256
`7a662c678fb7a370059659205fce96741b0bd959690995a67e0001ebdb21e556`, core 200 MHz, `clk_rv` 100 MHz) is the first
image with the control core that ran on the card. It loaded through XRT as an ordinary user (no root), the first
`tsr-selftest` passed, and the M6 checks passed. Numbers and method: [performance.md](performance.md) "The first card
image with the control core"; the image: [platform-u200.md](platform-u200.md) §8.7 "Card bring-up".

What passed:

- Host tools: `build/xrt` of this worktree (`-DTSR_WITH_XRT=ON`, `-j8`). Before the card: `test-tsr-xrt-noop` 32 checks,
  `test-tsr-kdev` 10,302 checks, `test-tsr-protocol` 2,398 checks, 0 failed.
- `tsr-selftest --device xrt`: 205 checks, 0 failed (6.6 s with the xclbin load). Identity: `CAPS` 0, `T_MAX` 8,
  `NUM_CH` 4, SPM 4,096 KiB, `CTL_ID` 0x5d01, `Q_DEPTH` DMA 8 / QACT 4 / MVM 4 / VU 0 / ATT 0 / SYNC 4, `RV_FREQ` 32768
  (F_RV / F_CORE 0.5000). Cold boot to HELLO 8.4 ms, warm restart 2.5 ms, `PING` 6.57 µs median, 303,578 pipelined
  launches per second. The `selftest` kernel ids 1 to 10 and the misuse cases equal the shadow simulator (ids 4 to 6 answer
  the `KERNEL_ERR` that a device without VU and AT must give). No log record dropped, `STATUS` clean.
- 0.5B M6, the baseline of `k4min` (`dev.klc`, 2 prompts x 16 tokens): the card's generating run and its replay are both
  32 of 32 rows bit-identical to the simulator; the sha256 over tokens and logits is
  `58fa2a9b2fb84fc9ea3a4e2c91da81ee73649cd616a94a54d72252c7a857ab27` for the baseline, the generating run, the replay and
  the simulator's own replay. A full `klcheck all --device xrt -n 64` (8 x 64): KL ratio 1.026, top-1 97.07 % (the
  simulator's figures); its tokens replayed on `sim` with `TSR_SIM_VU=0 TSR_SIM_TMAX=8` are 512 of 512 rows
  bit-identical. No numeric mismatch was found, so there is no first differing op to report.
- 32B Q4_K_M, `klcheck all --device xrt --phase m6 --n-prompts 2 -n 16`: mean KL 1.3362e-3 against 3 x floor
  5.3450e-3 (floor 1.7817e-3, ratio 0.750), top-1 100 %, PASS. The same 32 rows replayed on `sim`: bit-identical.
- Throughput (host loaded by other jobs; best run): 0.5B 47.3 tok/s decode (21.16 ms per token; 43.6 to 47.3 over five
  runs) and 592.6 tok/s prefill (157 tokens); 32B 2.24 tok/s decode (447.2 ms per token) and 17.4 tok/s prefill.
  Device counters per decode token: 0.5B 2.893 M cycles, 22.5 % `MVM_EMPTY` (issue-bound gaps) and 5.0 % `MVM_WAIT`;
  32B 84.06 M cycles, 96.5 % `MVM_CYCLES`, 2.3 % `MVM_EMPTY` and 1.3 % `MVM_WAIT`. Each launch costs 17 µs beyond the
  device time. `RV_WFI` and `PUSH_STALL` read 0. The 32B slowdown against 2026-10-27 (2.93 tok/s) is the core clock.
  The cost of `clk_rv` at 100 MHz: at most about 8 % of a 0.5B token and 1.4 % of a 32B token (model arithmetic;
  performance.md).
- `test-backend-ops -o MUL_MAT -b Tesseract0` on the card: 1,402 of 1,403; the one failure is the known one-output case
  `f32,f32,m=1,n=1,k=2048` (NMSE 8.2e-4, getting-started.md "Known flake").
- Debugging, `tsr-selftest --debug` (DMI smoke) and a real `riscv-none-elf-gdb` session through
  `tsr-dbg gdb --stdio --device xrt --elf build/xrt/fw/tsr-fw.elf --halt-on-boot`: halt at `_start` (0x80),
  a software breakpoint at `tsr_rt_main` (hit), all registers and CSRs, `x/4xw &__tsr_dbg` (TDBG magic), `stepi`, a
  memory write and read back through the system bus, a hardware watchpoint on `tsr_rt_launch` (fired, old 0, new
  65537), `monitor launch selftest 11` with `hbreak selftest` (hit, backtrace to `_start`), `finish` (returned
  0x80000201, the `KERNEL_ERR` that the launch then answered), `monitor status|units|qstatus|sreg|perf|tile|trw|log|
  rv regs|rv sample|rv step`, and `detach`; a selftest after the session passed. Every step worked on the integrated RTL.

Findings and what did not work:

- `monitor hist` is empty: the image has `HIST_DEPTH` 0 (the simulator still models the history). `REC_DROPPED` counts two
  per launch, which fits that.
- `monitor rv sample` while the hart is halted returns pcs in the debug ROM (0x10000850 to 0x10000860, the park loop at
  `DmHaltAddr` 0x10000800), not the program's `dpc`. A stale sample (the known early `RV_SAMPLE` acknowledgement) was not
  seen in three sessions, but nothing here proves the handshake: the sample is always taken in the park loop. The
  `SBA_REFUSED` setter and a refused system-bus access were not exercised.
- `TSR_STATS` prints 13 PERF slots in the backend and the transport summary did not print at the exit of
  `llama-completion` or `klcheck`. The whole set is readable after the run with `tsr-dbg --attach -ex perf` (the counters
  stay until the next open).
- My first GDB script ended with `continue` and no stop: batch GDB waits forever while the core runs idle. It was
  killed by `timeout`; the next open reset the device and everything passed. Use `detach`, not `continue`, at the end.
- Setup of this worktree: `third_party/llama.cpp` was an empty directory. It is a local clone of the main tree's copy at the
  pin `e3797b18`, patched by `third_party/apply-patches.sh`; `build/tools/riscv` links to the main tree's toolchain. Neither
  is committed. No source or RTL file was changed for the bring-up.
- Host timing on this machine is noisy while other jobs run (a Vivado report run, a training job): 0.5B decode varied from
  43.6 to 47.3 tok/s, and a 12-thread run gave 3.3 tok/s. Repeat tok/s runs on a quiet host before quoting them.

Not done, next:

- Trace replays through the card (`tsr-selftest --traces`, M6 gate item 1): the trace data is only in the main tree
  (`traces/*/` is not committed). The vector-unit and attention sets do not apply to this image.
- The 32B KL used 32 positions. The M6 gate lists 8 x 64; a run of that size on the card takes about 30 to 45 min
  (the CPU reference and floor runs dominate; the device phase is about 4 min).
- Timing sign-off of the 200 / 100 MHz image is still arithmetic on the 200 / 150 MHz route. The card showed no timing
  fault in any run (selftest, 576 bit-exact rows, decodes of 65 tokens), but that is not a timing run.
  A link of its own at 200 / 100 MHz (about 4.5 h) is the check.
- The clock of the control core is the cost to watch in the next image: in the 0.5B decode `MVM_EMPTY` is 22.5 % of the device
  time; the simulator model says 200 MHz would remove about 42 % of that.
- The M7 gate needs the full image with the vector unit and attention.

## Earlier: the 200 / 100 MHz image with a control Pblock (branch `k4m200`, November 6)

The branch `k4m200` (worktree `/home/limagin-dev/tesseract-k4m200`) starts from `e7488b0` of `k4-min`, the reduced first card
image with the three register changes made after its first link. The user decided to run this image at core 200 MHz and
first tried `clk_rv` at 150 MHz; that link missed `clk_rv` and the image now runs `clk_rv` at 100 MHz (below). Two things
changed; [platform-u200.md](platform-u200.md) §8.7 "Second image" has the numbers and the comparison:

- Clocks: `hw/u200/k4.cfg` requests `0:200|1:100`; `hw/u200/clock_final_check.tcl` takes its window from the kernel it finds
  in the design (k4: exactly 200 and 100 MHz; k1 unchanged). The 3.333 ns CDC cap stays.
- Placement: the control plane (control core, issue, CSR, DB core side) has its own Pblock `pblock_tsr_ctrl_plane`, a child of
  `pblock_dynamic_SLR0`: clock-region rows Y3 and Y4 of SLR0 without X2Y3 (16,300 CLBs, 252 RAMB36, 840 DSP). It leaves the
  DDR0 controller's clock regions X2Y1 to X2Y3, which were 99.5 % full and failed their clock by 0.33 ns. The map is
  `docs/fig/floorplan/k4m200-pblock-plan.png`. The legality check (`tb/platform/viv/k4_pblock_check.tcl`) passed on the routed
  checkpoint of the first link; the 200 / 150 MHz link placed and routed with this Pblock.

The 200 / 150 MHz link (`build/link/k4m200-rv150`) routed with only `clk_rv` failing: WNS -0.048 ns after route, -0.008 to
-0.015 ns during the post-route `phys_opt_design`, 71 endpoints, Ibex `if_stage` to `u_bus` `igrant_q`, 15 logic levels. The
core and the shell clocks met (the DDR4 clocks too). The user stopped it on 2026-11-06 and decided to lower `clk_rv` to 100 MHz,
with the core at 200 MHz. The Pblocks are unchanged. The host tools read `RV_FREQ` at run time, so firmware and host need no
change.

The 100 MHz image was not placed and routed again. It reuses the routed DCP of the 200 / 150 MHz link
(`build/link/k4m200-rv150/tmp/link/vivado/vpl/prj/prj.runs/impl_1/level0_wrapper_routed.dcp`) with `kernel_frequency=0:200|1:100`,
through `TSR_VPP_REUSE_IMPL` (`v++ --reuse_impl`): Vitis ran only the frequency scaling, `write_bitstream` and the xclbin
packaging, in `build/link/k4m200` (23 min). The first attempt stopped in `rtdgen` because that flow does not write
`address_map.xml`; `build_xclbin.sh` now copies it and `kernel_service.*` from the earlier link (they do not depend on the
clocks). Details and numbers: [platform-u200.md](platform-u200.md) §8.7 "What the reuse run did".

Result: `build/link/k4m200/tsr_u200_k4.xclbin`, 76,777,347 bytes, sha256
`7a662c678fb7a370059659205fce96741b0bd959690995a67e0001ebdb21e556`. `_new_clk_freq` reads core 200.0 and `clk_rv` 100.0 (Vitis
could reach 200.4 and 148.9 MHz); `xclbinutil --info` shows `DATA_CLK` 200 and `KERNEL_CLK` 100 MHz. The route is
fully routed (`ROUTED_FULLY` check passed). Setup and hold at the shipped clocks are by arithmetic on the routed 200 / 150 MHz
summary: only `clk_rv` failed, by -0.048 ns at 6.666 ns, which is +3.286 ns at 10.0 ns; WHS +0.003 ns; every shell clock and the
core met. `CLKBD`, `CLKNET` and `CDCCHK` are the lines of the 200 / 150 MHz run (same netlist and routing); `CLKFINAL` does not run
in the reuse path and was applied by hand (200.0 and 100.0 are inside the k4 window).

The card bring-up of this xclbin is the entry above. Load it with `TSR_XCLBIN=build/link/k4m200/tsr_u200_k4.xclbin`.
The setup and hold result is not a new timing run; a link of its own at 200 / 100 MHz (about 4.5 h) is the check if the
card shows a timing fault (it showed none in the M6 runs). If a later image needs another placement, the
next lever for a DDR0 clock failure is a Pblock for cluster 0 that leaves out X2Y1 to X2Y3: the control Pblock cannot keep
cluster 0 out of them.

## Earlier: resume point, reduced first card image (branch `k4-min`, November 6)

The branch `k4-min` (worktree `/home/limagin-dev/tesseract-k4min`, from `codex/local-intermediates` at `aadf92c`)
makes `tsr_u200_k4` the reduced first card image: four MVM clusters at T_MAX 8, the RISC-V control core, issue path, CSR
block, 2R2W SPM, DMA without KVST, QACT and the DB block without trace writer and history. The elaboration has no vector
unit and no attention; `CAPS` reads 0, so the host runs MUL_MAT on the device and the rest on the CPU (M6). Read
[platform-u200.md](platform-u200.md) §8.7 first: configuration, tests, the placement gate with numbers, risks and the
launch command. Short form:

- Placement gate: out-of-context placement with the major-block map. Issue, CSR and DB beside the SPM in SLR1 give an
  estimated SLR1 occupancy of 94.7 % to 100.2 % with the shell (fail); the same blocks in SLR0 beside the RV give 88 %
  to 92 % (target 93 %; the routed October image had 92.1 %). `hw/u200/k4_major_blocks.tcl` has the passing map.
- Open risk: placed `clk_rv` slack -0.208 ns at 6.0 ns inside Ibex with that map (+0.276 ns with the other); placement
  numbers are before routing, and the DDR controllers of the platform are not in the out-of-context run.
- First link (`build/link/k4min`, November 6): routed legally, timing not met: core -2.557 ns, `clk_rv` -0.795 ns, the
  shell DDR4 clocks -0.05 to -0.33 ns (October's met by 0.02 to 0.07 ns). platform-u200.md §8.7 "First link" has the
  triage of all failing endpoints and the three register changes made after it (cluster reset copies, `DONE_REG` on the
  unit done strobes, `MT_PIPE` on the DB memory taps). They address the groups that had a crossing to register, about 76 % of the
  core clock's TNS and its three worst path families. Not changed, and each still fails the core clock or `clk_rv` by itself:
  the CSR enable fan-out (LQ ring write enable, PERF enable: 2,657 endpoints, -1.28 ns); the DB stop logic in
  `tsr_dbg_stop` (15 levels, -1.57 ns; changing it changes the DB §5.1 timing contract, or build the first image with
  `HAS_DB` 0); the Ibex decode-to-TCM-address and bus paths (17 to 20 levels, `clk_rv` -0.8 ns, inside vendored Ibex and its
  bus); the shell clocks (kernel logic inside the clock regions of MIG 0 to 2, 76 to 100 % occupied: a sub-SLR Pblock for the
  control plane against the clusters, or a placer spread directive, neither allowed so far).
- The simulator baseline for the card is `build/k4min/klcheck/qwen05b-q8-m6/dev.klc`
  (`TSR_SIM_VU=0 TSR_SIM_TMAX=8`, Qwen2.5-0.5B Q8_0, M6, 2 prompts x 16 tokens).
- The full elaboration is `core_ooc.sh k4` and `hw/u200/k4_major_blocks_full.tcl`; the simulator still models the DB trace
  writer and history, which the card image does not have.

The earlier checkpoint follows.

## Earlier: Claude checkpoint

The user requested a cumulative source and documentation checkpoint on November 6.
Start with [checkpoint-2026-11-06.md](checkpoint-2026-11-06.md): it identifies the
active checkout, retained previous source, implementation history, evidence and
next decisions. The checkpoint belongs to `codex/local-intermediates` in
`/home/limagin-dev/tesseract-local-intermediates-20261105`. The original checkout
keeps its previous uncommitted implementation for comparison.

Keep the existing native build running. No rollback, new goal or automatic retry
is requested. Use T4 and VU 8/8/4, core >=250 MHz, existing swap and major-block
Pblocks only. No Luna/LLM polling. MoE is the first model extension after the build;
GDN and speculation remain later work. Read newer entries before historical ones.

### WIP checkpoint validation, November 6, 03:19 KST

The user explicitly selected an immediate WIP commit with failed/incomplete
checks recorded and the remaining checkpoint checks stopped. This is an
exception to the normal complete pre-commit gate for this handoff only. It is
not release validation or a claim that the full gates passed.

Fresh `cmake --build build/dev -j8`, generated tables and generated U200 top
checks passed. Non-slow CTest: 295 passed, one fixture skip (`trace_conv_model`),
one disabled (`test-tokenizers-ggml-vocabs`), zero failures.

Full lint was stopped by request after 102 Verilator cases passed and three
C16 variants failed. All three report WIDTHCONCAT at `rtl/core/tsr_vu.sv:302`,
`assign local_score_req = '0`: the packed request array is 8,672 bits, above
Verilator's 8,192-bit warning threshold. `-Wall` makes it fatal. XSim and Vivado
multi-driver stages were not reached. This is outside the active K4 C4 shape,
but remains a supported-configuration lint regression. A per-owner generated
zero assignment is a candidate fix; preserve the native inputs before changing
it. The 650-item RTL suite was also stopped before completion; its partial log
is not a suite pass. Receipts are under `build/analysis/checkpoint_20261106/`.

The candidate's ignored `build/dev` symlink points to its existing `local_sw`
build. The original pinned RISC-V toolchain was reused read-only. No runtime
source was changed for these gates; all 142 frozen native input hashes still
match. FPGA routing and the Telegram notifier remain active. At 03:18 the
router was in phase 2.5.1 Update Timing, not a terminal result.

## Active local-intermediate / masked-KVST build (November 6, 00:07 KST)

The user authorized replacing the previous RV-SLR0/VU-SLR2 attempt with the
improved candidate, including DDR partial writes. At 00:04:39 the old attempt
`k4_20261105_195907_t4_rvlocal_vuslr2_core250_rv166` and its notifier/event logger
were stopped. All owned CAD processes exited. The last native log reached
route phase 4.2 Update Timing; this is an interrupted attempt, not a completed
routing failure. Logs and checkpoints remain in the main checkout.

The new full native attempt runs from the isolated source checkout
`/home/limagin-dev/tesseract-local-intermediates-20261105`, branch
`codex/local-intermediates`. Its output is
`build/link/k4_20261106_000734_t4_local_masked_core250_rv166/` under that checkout.
Start time is 00:07:34 KST. Supervisor PID is 1111688, native wrapper 1111699,
event logger 1111691 and Telegram notifier 1111692. No automatic retry or
LLM polling is scheduled. This supersedes the older active-build entries below.

The packaged design retains T4, four 512-bit DDR channels, VU 8/8/4 and
core 250 MHz / RV 166.666667 MHz. All three local-intermediate features and
2-flit gather are enabled. Masked KVST is included through AXI WSTRB; it is
not merely a status flag. VU bank 0, its local attention/collector and clusters
0/2 use SLR0; bank 1 and owners 1/3 use SLR2. RV stays in SLR0. SPM, DMA and
the SPM-facing frontend remain central. The regenerated whole-block Pblock map
was shown before launch. The whole collector keeps its hierarchy for the
Pblock; no individual cell, stage, site or BEL placement is selected.

At 00:07:57 fresh XO packaging and payload verification passed: 141 manifest
entries, 142 verified HDL/XDC files and the expected kernel XML. The private
cache contains only 100 reusable Xilinx platform entries; the previous custom
kernel entry was excluded. At 00:08:30 VPL entered generate_target after BD
creation/update. Native source and hook inputs are frozen. Use 16 concurrent
IP synthesis jobs, 8 implementation/build threads, one global Vivado slot and
the existing swap. Required clock/reset, CDC and final timing gates remain.

The model-free Telegram watcher checks every 60 seconds. After both placement
and complete congestion/utilization reports exist, it sends one comparison of
SLR1 CLB/LUT, per-boundary SLL, same-stage placement WNS and SLR0/1/2 directional
congestion against the stopped 19:59 build. The reference placed WNS is
−1.793 ns; do not compare it with that build's later −1.344 ns physopt WNS.
Errors, interruption and terminal state retain existing notifications. Unchanged
polls send nothing. No credential was copied into the repository or logs.

Launch, fixed source hashes, frozen comparison baseline, 44 passing offline
runner/notifier tests and delivery state are under the candidate's
`build/analysis/k4_local_build_20261106/`. The original stop receipt is under
that directory name in the main checkout. Astra's launch review passed.
Native exit zero still requires final route, setup/hold, >=250 MHz and CDC
report review before declaring build success. No commit or push was made.

### Placement result, November 6, 02:55 KST

Placement completed at 02:37:23 and physical optimization at 02:51:23.
The route step has started; its pre-route clock/reset and CDC checks passed
(unsafe 0, unconstrained 0). No router congestion/demand or final route result
is available yet. The native process remains alive.

| Same-stage metric | Stopped 19:59 attempt | Local/masked candidate |
|---|---:|---:|
| SLR1 occupied CLBs | 49,168 / 99.81% | 48,942 / 99.35% |
| SLR1 LUTs | 282,026 | 284,877 |
| Placed SLLs | 14,240 | 12,947 |
| Placed estimated WNS | -1.793 ns | -2.790 ns |
| Post-physopt estimated WNS | -1.344 ns | -2.124 ns |
| SLR1 north/south global congestion region | 64x64 / 64x64 | 64x64 / 64x64 |
| SLR1 west global congestion region | 16x16 | 32x32 |

Placed crossings fell 9.1%, but central CLB occupancy remains near full.
The largest central congestion regions did not shrink, westward congestion
grew, and estimated timing worsened. This is not evidence that central routing
is solved, nor a completed route failure. Full-platform reports include shell
logic. Router per-column SLL demand must be compared at the same routing stage.

The Telegram notifier records successful placement-comparison and route-step
notifications, with zero delivery failures. The route-step notification marks
the implementation step, including its pre-route hooks, not completed routing.
Reports are in the active attempt's `tmp/link/vivado/vpl/prj/prj.runs/impl_1/`
(`runme.log` and `slr_util_placed.rpt`). No build inputs were changed.


## Candidate whole-model decode checks (November 6)

The active native build was left running. The same candidate firmware image
`f7bb1fc039d7ee0c` ran real GGUF models through llama.cpp's M7 backend on the
C simulator, with T4 and all three local capabilities enabled. Flash attention
and Q8_0 KV cache stayed on the device. CPU matched-token replays, three-seed
1-ulp floors and CPU token/embedding identity checks passed.

| Model | Sample | KL / floor | CPU top-1 match | Local-on/off exact logit rows |
|---|---|---:|---:|---:|
| Qwen2.5-0.5B Q8_0 | 4 prompts × 32 generated tokens | 1.105141 | 124/128 | 128/128 |
| Qwen2.5-0.5B Q4_K_M requant | 4 prompts × 32 generated tokens | 1.376901 | 124/128 | 128/128 |
| Gemma 4 12B Q4_K_M | 1 prompt × 8 generated tokens, prompt-tail 2 | 0.355755 | 9/9 | 9/9 |

The sampled acceptance criterion is KL/floor <=3 and CPU top-1 match >=80%.
Local-off replays keep the same candidate gather and masked-KVST implementation;
they isolate local-capability behavior, not every change from the previous RTL.
Queue logs confirm local QK/PV in both model families and nine completed local
MVM softcap instructions in Gemma. Some attention records still use the shared
path. No erroneous queue completion was seen; code-zero SYNC NOPs are normal.

Each Qwen run had 4 prefill and 124 decode graph calls, with 128 device launches.
Gemma had 8 graph calls and 8 launches. M7 already removed host intervention
between supported operations within a token. The host still submits each token
graph, receives final logits and selects the next token. Local intermediates
reduce traffic inside the FPGA; they do not implement autonomous multi-token
generation. Graph preparation and model loading also remain host work.

These are functional full-model runs on the C simulator, not full-model RTL or
card throughput measurements. Primary Qwen runs used the simulator's default
RV/core ratio; separate eight-position queue samples and Gemma used
166.666667/250. The cycle model does not qualify the native VU 8/8/4 configuration.
No long-context, SWA-wrap, MoE or broad language-quality claim follows from these
samples. Astra independently reviewed the results and passed this bounded scope.

The candidate's `build/analysis/model_decode_local_20261106/model_decode_report.json`
contains model/firmware hashes, generated tokens, numerical results, queue counts
and replay hashes. Raw commands and logs are in the same directory. Review is in
`build/analysis/local_intermediates/review.md`. No runtime source was changed for
these experiments and the native build's frozen inputs were preserved.

## Milestones and dated experiment history

The sections below preserve prior checkpoints and decisions. Their statements
about an active build, paused goal or not-yet-launched candidate apply to the
date in that section. They do not override the November 6 status above.

| Package | State | Commit |
|---|---|---|
| W0-W3: tables, simulator, firmware SDK (Basis), M6 and M7 backend on the simulator | done; one launch per decode token; KL within the criterion for 0.5B, Gemma 4 12B, 32B | up to `667153f` |
| Versions: Tesseract Core capability 1.0, Basis 0.1.0, ABI 1, kind revisions | done, tag `basis-v0.1.0` | `13dbd2e` |
| llama-server on the backend, TSR_KINDS lists, klcheck multi-seed floor | done | `b58fc03`, `7175f61` |
| W4-A RTL: issue path, 2R2W SPM, control subsystem (Ibex, riscv-dbg, TCMs, CDC) | done, gate T1 met at 166.7 MHz (RV32MFast) | `4149614` |
| W4-B RTL: VU, attention modes, DB block, two-clock XRT adapter; tsr-dbg (GDB), tsr-prof | done; ctest 297/297, lint clean, all benches green | `6435e8a` |
| W5 preparation: VU lane pipes (`tsr_vu_lanes.sv`, `VU_LANE_PIPE`), floorplan C'' (`hw/u200/k4_floorplan.tcl`), TCM RAMB36 constraint in every flow (`tsr_ctrl_tcm.xdc`, `hw/u200/tcm_opt.tcl`), `build_xclbin.sh` enabled for k1/k4 | checkpoint; first K4 route failed | `58bf76d` |

## Historical local candidate qualification before launch (November 5)

This isolated `codex/local-intermediates` checkout implements the authorized
local FAST attention, MVM SCALE/softcap epilogues and required gather narrowing.
It includes masked KVST. The active native build belongs to
`/home/limagin-dev/tesseract`; none of its runtime source or hooks were changed.
No new native build, commit or push was made.

See [intermediate-locality-plan.md](intermediate-locality-plan.md) §0 for the
implemented scope, Pblock map, exact compatibility rules and test results.
Full software build and non-slow CTest pass (288 passed, 8 skipped, 1 disabled).
Integrated real-core graph replays, abort/drain/restart, focused local queues
and legacy RTL regression pass. Astra's functional review passed. Measured
small RTL graphs range from +1.98% to −10.92% cycles, with exact DDR outputs.
This does not establish whole-model performance or native routing/timing.

Candidate K4 enables all three local capabilities and two-flit gather; K1 and
module defaults retain legacy. The Pblock plan moves one VU bank back to SLR0
beside its local producers. Show the saved map before any future native build.
Use ordinary reports to qualify RAM inference, timing and actual SLR crossings.
The full RTL/technology commit gate remains required before a commit.

MoE is the first model feature after the active build. Original weight ownership,
expert bases/strides and token origins are represented in the shared window
helper; expert packing, MUL_MAT_ID, token grouping and combine remain future.
Keep llama.cpp and PyTorch compatibility, and measure the user's permitted
single-digit throughput tradeoff rather than infer it from lane count. Current
changes retain numerical bits despite permission for validated roundoff changes.

## Historical RV SLR0 / VU SLR2 native build (November 5, 19:59 KST)

Next-run requirement from the user (November 5): integrate the validated KVST
masked-write candidate from `/home/limagin-dev/tesseract-kvst-masked-20261105`
before the next verification/build run. Test it together with the RV-local/VU-SLR2
configuration and include it in the next native build. This requires merging the
candidate source changes, not merely toggling the status metadata. Preserve the
current running attempt. Check the combined regressions and record actual masked
write inclusion in the next build manifest. Do not launch a new build solely
because of this reminder.

The user requested stopping the compact-VU build and launching the validated
RV relocation candidate. The previous attempt
`build/link/k4_20261105_164800_t4_vu_compact_core250_rv166/` was stopped by request
during route phase 5.2, Global Iteration 1. Its last reported overlap count was
215,571. This is an interrupted run, not a completed native routing failure.
All of its owned CAD processes and its event logger are gone. Logs and
checkpoints remain. Stop receipts are in `build/analysis/rv_local_20261105/`.

The candidate source patch is now applied to the main checkout. The K4 wrapper
was regenerated, and all 18 changed source/test/build files match the validated
candidate byte for byte. CC §11.4 is now present in the main design document.
T4, VU 8/8/4, the four 512-bit DDR channels, DDR locality, paired gather and compact
VU inputs remain. Whole RV/TCM/CDC and RV debug use SLR0; both VU lane/coefficient
banks use SLR2. The SPM-facing VU frontend, SPM, RV-SPM bridge and DMA use SLR1.
SPM/issue/LCQ segmented control transport is enabled. The isolated masked-KVST
candidate and 500 MHz TDM are not included. The Pblock preview was shown before
launch; only whole functional blocks are assigned.

The new full native attempt started at 19:59:07 KST in
`build/link/k4_20261105_195907_t4_rvlocal_vuslr2_core250_rv166/`.
Supervisor PID is 689148; the local event logger is 689151. Target clocks remain
core 250 MHz and RV 166.666667 MHz. It holds one top-level Vivado slot, permits
16 concurrent IP synthesis jobs and uses 8 implementation threads and the
existing swap. No LLM polling monitor or automatic retry is scheduled. The user
cleared the long-running goal; this is the separately authorized single build.

Fresh XO packaging passed at 19:59:28: 139 source-manifest entries, 140 verified
HDL/XDC payload files and the expected kernel XML. The old custom-kernel cache
entry is excluded; 100 platform IP cache entries were copied privately.
Vitis entered native VPL at 19:59:36. Clock/reset, CDC and final frequency gates
remain enabled. No placement, routing or final timing result exists yet.

Launch, runner tests and events are in
`build/analysis/k4_rvlocal_vuslr2_build_20261105/`; live state is the attempt's
`status.json`. Astra's source/runner review passed, as did 25 runner checks.
Do not mutate the running attempt's RTL or hook inputs. Native exit zero still
requires report review before declaring a 250 MHz routing success.

### Telegram notifications

The user confirmed receiving the test message from `@codex6974_bot` on November 5.
A separate notifier, PID 695363, now watches this attempt every 60 seconds.
It reads the status, event log and ordinary implementation reports. It sends
placement and congestion results, routing milestones, errors, interruptions and
the final state. Unchanged polls send nothing and use no model tokens. Native
completion is reported as awaiting final report review, not as timing signoff.

The notifier and its launch receipt are under
`build/analysis/telegram_notify_20261105/`. Eleven offline checks pass and Astra's
review passed. Configuration and the token are stored outside the repository in
`~/.config/tesseract/telegram.json` and `telegram.token`, both mode 600. Do not print
their contents. Delivery state and a process lock prevent routine duplicate
alerts; an ambiguous network response can still cause a repeated message.

This process does not modify build inputs, restart CAD or wake an LLM agent.
It exits after delivering the terminal state. A later build needs its own
notifier invocation with that attempt's paths. At attachment, the native build
was still running with no recorded native error or placement result.

## Distributed SPM and local output tiles assessed (November 5)

The requested three-priority implementation plan is now in
[intermediate-locality-plan.md](intermediate-locality-plan.md): local attention
scores/P, bounded MVM output epilogues, then explicit sharing/import/export.
The common ownership, lifetime and transport contract comes first. The plan
covers llama.cpp M7 and a separate bounded PyTorch FX/runtime extension, with
legacy compatibility and physical-path removal gates. This is planning only;
no implementation, native input change or new build was made for this request.

Improvement direction agreed with the user: reduce central round trips of
intermediate data between the added engines. First assess the active build;
this direction does not authorize implementation or another native run.
Prioritize attention score/P producer-consumer locality, then local vector
epilogues on completed MVM output tiles. Preserve arithmetic capacity and
precision, explicit overflow to DDR, and required reductions/redistribution.
Evaluate removed or narrowed physical interfaces as well as transferred bytes,
whole-workload cycles, and placed congestion. Moving buffers while retaining
all existing wide remote paths is not sufficient. Full SPM distribution remains
a later option rather than the immediate scope. The next verification/build
must still include the separately validated masked-KVST change.

The user asked to preserve the active build and assess cluster-local SPM and
GPU-style local output-tile accumulation. The assessment is in
`docs/platform-u200.md` §8.6.20, with analytical payload calculations under
`build/analysis/spm_locality_20261105/`. No implementation or native run was
started. Current dense MVM already partitions output rows, replicates inputs
in local activation buffers and accumulates K locally. The missing locality
is resident output/intermediate tiles and their consumers. Current firmware
also stores MUL_MAT output to DDR; cross-record residency needs software work.

The candidate is two outer-SLR memory/compute groups with four logical owners,
retaining eight VU front/EXP lanes in total and four weight channels. Local
attention score/P storage with local ASMAX/PV consumers is the first case to
model. Global maxima and final partial reductions remain. Merely moving RAM
while preserving all wide remote paths does not establish wire savings.
Capacity is plausible from prior URAM reports; timing, net area, routability
and end-to-end performance are unmeasured. Keep the separately recorded
masked-KVST integration requirement for the next verification/build.

The follow-up software/overflow audit is in platform §8.6.21. Firmware MUL_MAT
tiles T only, and llama.cpp rejects shapes whose full-N one-token scratch or K
cannot fit. PyTorch's separate `sw/rt` already splits N/K into independently
repacked parts and assembles output through DDR/host. Do not conflate those
paths. Current PyTorch only offloads weight matmuls; its full-model card evidence
predates the control-core build. New local tile residency is not implemented.
The actual T4 planner probe gives FFN/head tile sizes 4/3/1 for the recorded
Qwen32 FFN, Qwen32 head and Gemma12 head shapes. Output tiles can spill to DDR,
but preserving original weight row ownership and defining tile completion are
required. Merely reducing N or the global SPM size is not a valid implementation.
Source/probe evidence is under `build/analysis/spm_stack_20261105/`.

## RV transport and placement candidate completed (November 5)

The user authorized narrowing RV communication and moving the control subsystem,
then asked to put RV in SLR0 and the VU in SLR2. The isolated candidate is
`/home/limagin-dev/tesseract-rv-local-20261105`, branch
`codex/rv-local-transport`, based on the complete dirty compact-VU checkout.
Sol implemented and tested; Astra reviewed. The main checkout's runtime sources
and the running compact-VU native build are unchanged. The broader goal remains
paused. No new native K4 link, card operation, commit or push was performed.

K4 enables same-core-clock packet links: SPM returns four 128-bit segments,
issue sends three 129-bit segments, and LCQ_POST sends two 128-bit segments.
The complete 64-byte RV line buffer and original CDC remain local to `u_ctrl`.
Selected interface width falls from 1,205 to 404 signals relative to moving
those interfaces without segmentation. This is not a measured whole-K4 SLL
saving; sequence, error and other debug/control links remain wide.

The Pblock plan places whole RV/TCM/CDC, RV debug and their endpoints in SLR0.
Both VU lane/coefficient banks go to SLR2. The SPM-facing frontend remains in
SLR1, avoiding new 512-bit SPM read/write crossings. SPM, its RV bridge, the
central endpoints and DMA remain in SLR1. Cluster and DDR0/3 bindings are
unchanged. The updated `k4-rv-local-pblock-plan.png` shows allowed regions,
not placed cells. Only whole functional blocks are constrained.

Validation passes the randomized packet/reset and actual CSR drain regressions,
eight transport-enabled controller cases, 25 stock control pytest entries and
four integrated cases. Matched tiny M7 decode takes 52,968 to 53,181 core cycles
(+0.402%); prefill 68,703 to 68,993 (+0.422%), with all oracle results unchanged.
These are small RTL workloads, not whole-model/card throughput measurements.
The T4 four-channel smoke/debug tests pass. Separate endpoint synthesis adds
422 LUTs and 1,620 FFs gross, no BRAM/DSP; multi-driver checks, XSim static
elaboration, focused Verilator lint and generated-file checks pass.

The review found a premature idle/drain indication during partial issue
transfer; the final implementation gates it with retained FIFO occupancy.
A regression confirms QCLEAR is refused and SOFT_RESET waits until it drains.
No functional/source-Pblock blocker remains. Full integration area, SLR2 fit,
250 MHz timing and congestion relief are still unmeasured. Concentrating both
VU banks increases upper-boundary traffic, so it is a candidate placement,
not a proven routing improvement. No further build is scheduled by this work.

Evidence is under the candidate's `build/analysis/rv_local/`; the main checkout's
`build/analysis/rv_local_20261105/` holds the baseline, candidate location and
baseline-relative source patch. See the candidate's CC §11.4 and
`docs/platform-u200.md` §8.6.18. The isolated masked-KVST change remains separate.

## Standalone SLR TDM validation completed (November 5)

The user authorized a separate measurement of SLR wire reduction and 500 MHz
timing, while preserving the running compact-VU K4 build. Sol implements and
measures; Astra reviews. Work stays under `build/analysis/slr_tdm_20261105/`.
Production RTL and native build inputs are snapshotted and remain unchanged.
This authorization does not resume the paused full-K4 retry goal.

The experiment compares native AXI Register Slice W modes 10 (ordinary SLR)
and 11 (TDM) on `xcu200-fsgd2104-2-e`. Both use two simultaneous W links from
SLR1 to SLR0 and SLR2, with 512 data bits, word-derived byte strobes and LAST.
The W-only scope is not full AXI or NoC qualification. One MMCM supplies related
250/500 MHz clocks in the TDM experiment. The current platform's two existing
kernel clocks do not already provide this aligned pair.

The same functional source and destination endpoint Pblocks are used in both
variants. The preview `pblock-plan.png` was shown before launch. No individual
cell, stage, site or BEL placement is assigned. Functional stalls/reset and
full-rate transfer checks, routed boundary counts, setup/hold timing and added
resources are required. A sparse standalone result cannot establish timing or
routing closure inside the congested K4 design.

The isolated experiments routed legally and demonstrated 529 to 273 live payload
nets per boundary: 512 fewer across the two links (48.39%). Whole-probe SLLs fell
from 1,093 to 594, or 590 with a common-output divider clock pair. The latter
passed generated-IP simulation with 964 transfers per boundary, stalls, in-flight
reset and orderly MMCM restart/edge-alignment checks. No card was programmed.

Full 500 MHz transport timing did not pass. The original clock pair passed both
fast payload crossings, but failed 500-to-250 conversion (-0.098 ns) and a READY
return path (-0.293 ns). One normal post-route optimization did not close it.
The common-output BUFGCE_DIV variant fixed conversion (+0.166 ns), but its lower
500 MHz payload crossing failed (-0.100 ns) and READY still failed (-0.225 ns).
Hold checks pass. Source, reset and synthetic external I/O setup failures are
reported separately. No false paths, weaker timing margins or manual placements
were added to make these numbers pass.

The wire saving has a resource cost: probe SLR1 CLB sites rose from 146 to 311
in the divider variant, despite total LUTs falling from 2,589 to 1,239. Total FF
primitives rose from 4,349 to 5,920; baseline CLB FFs alone exclude 2,124 dedicated
SLR registers. This is a native-slice comparison, not a measured K4 integration
delta. Clock throttling/DRP recovery and all other NoC/protocol work remain open.
Further experiments have stopped. See `docs/platform-u200.md` §8.6.17 and the
experiment's `sandbox/`, `transport/` and `review/` receipts. The broader goal
remains paused. At the last check the original K4 build continued global routing.

## Native-feature audit and trace defect (November 5)

The user requested a search for other custom logic duplicating native platform
features. Sol audited memory/transport implementations and the current shell;
Astra reviewed. No implementation source, native build or card state was changed.

A standalone Verilator test reproduced trace corruption: fill a 32-record ring,
consume one record (RPTR = 1), then flush one new record with STOP_FULL enabled.
The full-mask `{PAD, new_record}` write destroys the unread neighboring record.
WPTR advances and DROPS remains zero. The C simulator's 32-byte writes preserve
that neighbor. This behavior predates the KVST candidate and is not fixed yet.
The reproduction is in `build/analysis/k4_native_capabilities_20261105/`.

Recommended next work is trace half-beat masks (also removing the 256-bit carry),
then a measured BRAM mapping of the central gather write queue and, separately,
the KVST output queue. SPM already uses byte enables; arithmetic RMW, ownership
arbitration and SLR timing stages cannot be delegated blindly to the controller.
See `docs/platform-u200.md` §8.6.15 for evidence, resource tradeoffs and lower
priority candidates. This audit starts no further implementation or native link.

The build-impact assessment is in §8.6.16. Compact-VU native synthesis reduced
kernel LUTs by 5,165 (1.36%), but its placed congestion is still pending. Earlier
T4 floorplans demonstrate that fewer central LUTs can accompany more route
overlaps. Root routing pressure is not proven resolved. Shared debug formatting,
selected BRAM queues, narrower DDR0/3 write crossings and producer-consumer-local
attention storage remain candidates; none is implemented by this audit.

## KVST masked-write candidate, isolated from the running build (November 5)

The user authorized U200 partial-write verification, then removal and validation
of KVST's central RMW. The ongoing compact-VU native build remains untouched.
The candidate lives in `/home/limagin-dev/tesseract-kvst-masked-20261105`, branch
`codex/kvst-masked-writes`, based on the complete dirty compact-VU checkout.
No new native link, commit or push is part of this task. The broader goal remains
paused. Sol owns implementation and regression; Astra reviews.

U200 support is established enough to implement: the native platform connects
64-bit WSTRB on all four channels, and 1,088 partial/overlapping XRT BO updates
passed on the installed card with all surrounding bytes preserved. That probe
used the host DMA path without loading an image, resetting the card or launching
a CU. The modified kernel itself has not run on the card. ECC can retain RMW
inside the DDR controller. Support evidence is in the main checkout's
`build/analysis/k4_masked_write_20261105/support/`.

KVST now queues a 16-bit word mask with each write beat and expands it to AXI
byte strobes at the bridge. Its 16×512-bit metadata preimage FIFO, read-ahead
walker/credits and 512-bit merge are removed. LD remains unchanged. Full-beat
ST, quant data and trace writes use all-one masks. Completion still waits for
write responses. `KVST_RMW` and KVST's explicit channel reads are zero; the C
simulator and cycle estimates match. Capture retains preimages for preserved-byte
replay checks, not as physical traffic. Numerical conversion is unchanged; the
user waived mandatory numerical bit-for-bit equality for this work.

Matched DMA OOC synthesis reduced CLB LUTs from 5,694 to 4,781 (−913, 16.0%) and
registers from 2,654 to 2,560 (−94); RAMB36 remains 8 and DSP remains 4. This
excludes new masks in other blocks, and is not a full-K4 area or routing result.
Production review found no RTL blocker. The reviewer found that simulator
capture hooks also informed the race detector; KVST now uses a capture-only
preimage helper. A regression checks both capture modes and rejects a false
metadata read. The firmware's numerical revision is unchanged.

Validation passed 18 pytest entries / 49 selected cocotb simulations: KVST at
4/3/1 channels, DMA, memory arbitration, competing trace writes, the actual AXI
bridge, K4 locality, and tiny M7 decode. The firmware test committed two partial
writes through the core's DDR locality pipes and observed zero KVST read requests.
The AXI test covers sparse mask 0x8421, independent AW/W handshakes, stalled
payloads and reversed B response IDs. The trace test covers 40 competing masked
DMA bursts and full masks for trace. Active-KVST reset was not separately tested;
existing DMA/arbiter reset tests pass. The broad debug sweep was stopped in favor
of the targeted trace regression. No whole-model performance run was started.

Unit/platform/BW Verilator lint and changed-module Vivado multi-driver checks
pass. XSim compilation and static elaboration pass for DMA, packaged K1/K4 and
BW K4. Generated tables and tops are fresh. The isolated CMake build uses
`TSR_WITH_LLAMA=OFF`; its final non-slow CTest run has 165 passes, one missing-model
conversion skip and zero failures. The first replay checks lacked ignored trace
fixtures in the new worktree; private fixture copies resolved that setup issue.
The final capture/race/AT/replay subset also passes 8/8. The candidate's validation
receipts are `build/analysis/kvst_masked/` and
`build/analysis/masked_validation/completion.json`. It has no new native route,
250 MHz timing signoff, full-K4 resource measurement or modified-kernel card run.
The original compact-VU build continues; none of this task's RTL is in that build.
The main checkout's `build/analysis/k4_masked_write_20261105/` holds
`completion.json`, `candidate_changes.json` and baseline-relative patches.
`source.patch` excludes documentation, build outputs and the candidate's
read-only vendor symlinks (worktree setup only, do not stage them). Its dry-run
application to the unchanged main sources passes. Keep the source patch unapplied
while the current native build uses that checkout; no later link is scheduled.
See `docs/platform-u200.md` §8.6.14 and AT §5.1, §8.3, §10, §13.4.

## Authorized VU compact-transport attempt (November 5)

After the paused failure and investigation, the user authorized only VU input
compression and bank-local coefficient storage, followed by one new native build.
Sol owns the implementation and regression, with Astra review. DMA, debug and
other proposed optimizations are outside this attempt. The broader goal remains
paused; this authorization does not schedule automatic retries.

The implementation and Astra review are complete. Each FRONT8 bank receives a
366-bit operand/control packet, 74-bit angle-write channel and reset: 441 input
bits, down from 545 (19.1%). C1/C2 and angle stores reside beside their consuming
lanes. Return width stays 192 bits. This is a logical interface reduction, not a
measured SLL saving. Numerical formats and arithmetic parallelism are unchanged.

The pre-change RTL, VU/core/attention benches
and U200 build files are frozen under
`build/analysis/k4_vu_compact_20261105/baseline/`, with a SHA256 manifest and the
pre-existing dirty-worktree diff preserved. The fresh-packaging supervisor is in
`build/analysis/k4_vu_compact_build_20261105/`. Its fifteen existing runner checks
pass. The native build started at 16:48 KST in
`build/link/k4_20261105_164800_t4_vu_compact_core250_rv166/`.
Supervisor PID is 314905; the local event logger is PID 315127. Fresh kernel
packaging passed source/payload checks, and native Vitis entered IP generation.
Platform IP cache entries were copied
privately; the old custom-kernel cache entry was excluded. Launch evidence is
`build/analysis/k4_vu_compact_build_20261105/launch.json`; live native state is
the attempt's `status.json`. No automatic retry is scheduled.

Standalone validation passed 52 existing C-oracle case executions and the ordered
local-reduction regression. At input pipe depths 0/2/3, focused baseline/candidate
pairs matched all 764 SPM/SREG writes in data, order and cycle, and all nine batch
cycle counts. Cases include both coefficient contexts, back-four subslots, both
RoPE modes, half reuse, reset, both DQROW formats and SPM contention. Receipts are
`vu_validation_completion.json` and `comparison.json` in the analysis directory.
The direct registered angle-write path needs native timing signoff. Delayed
angle-half release protects the final bank read; cycle equivalence is established
for the tested cases, not every possible workload. See VU §7.10 and
`docs/platform-u200.md` §8.6.12. The three integrated pytest targets also passed:
the T4 four-channel locality smoke and M7 half-VU decode/prefill. Generated tops
and tables pass their freshness checks. Full `rtl/lint.sh` has also passed,
including Verilator, XSim and the common-module synthesis checks. Its result is recorded in
`build/analysis/k4_vu_compact_20261105/full_lint.log`.

The layout keeps SPM-facing VU selection in SLR1 and the two lane banks,
including coefficient storage and local operand expansion, in SLR0 and SLR2.
Other major-block assignments remain those of the final T4 attempt. The regenerated
Pblock map was shown before launch. The build target remains T_MAX4, VU8/8/4,
core 250 MHz and RV 166.666667 MHz, with independent MVM/VU and four 512-bit DDR
channels. Only local Python processes supervise the build and log milestone/error
events; no Luna or periodic LLM monitor is running. Placement congestion, legal
routing and final timing are not yet known.

While this build runs, the user requested central-memory architecture and paper
research. `docs/platform-u200.md` §8.6.13 records the findings. SPM arbitration is
already bank-local. The strongest new bounded candidate is propagating AXI write
masks to remove KVST's core-side partial-META RMW FIFO and merge; actual shell
support was then verified and the masked candidate was separately authorized, as
recorded above. PASTA-style buffer ownership remains an unimplemented alternative.
The research itself did not start an additional native build.

## Paused for Claude handoff after the November 5 route failure

At the user's request this goal stops after the last link failure. The final attempt
`build/link/k4_20261105_113412_t4_dma_center_core250_rv166/` exited at 15:20:02 KST,
code 1. There are no remaining native build or monitor processes. No further retry
or optimization is scheduled. The workspace contains uncommitted implementation and
documentation changes.

The final router reported 31,192 conflicted signals and 35,335 physical node overlaps.
Unrouted and partly routed nets were both zero. The route is illegal. The last
intermediate timing was WNS −6.823 ns and TNS −40,520.634 ns. Neither clock nor final
setup/hold is signed off. Pre-route clock/reset and CDC checks passed. The 547 MiB
error checkpoint is `tmp/link/vivado/vpl/prj/prj.runs/impl_1/level0_wrapper_routed_error.dcp`
under the attempt directory. The original log, source snapshot, kernel XO and manifests
are preserved. Source/package integrity checks passed.

### What changed and what the comparisons established

| T4 attempt | Kernel Pblocks | Placed SLR1 CLB / LUT | SLLs | Final outcome |
|---|---|---|---:|---|
| 02:00 `020015_t4_vu` | Original 16 major blocks | 99.86% / 296,178 | 12,213 | Route failed, 8,054 overlaps |
| 06:15 `061530_t4_six` | Four clusters + two VU banks only | Placement incomplete | — | Native platform auto-pipeline reset register left unplaced |
| 07:50 `075018_t4_front_dma` | Original 16 + VU frontend SLR0 + DMA SLR2 | 97.80% / 259,620 | 16,917 | Route failed, 60,788 overlaps |
| 11:34 `113412_t4_dma_center` | Same, DMA returned to SLR1 | 98.51% / 265,510 | 14,120 | Route failed, 35,335 overlaps |

The final failed checkout had the last row's 23 whole-block assignments. The named Pblocks
place clusters 0/2 and VU bank 0 in SLR0, clusters 1/3 and bank 1 in SLR2. VU
`u_gen/u_rd/u_ang/u_inj/u_lma/u_lmb` are in SLR0. DMA/KVST, SPM and control are in
SLR1. Cluster assemblers and DDR0/3 bridges/arbiters use their original local regions.
The first T4 placement remains the best completed T4 route among these trials.
Reverting to it means restoring the original hook from that run's source snapshot,
not reverting the T4 or VU RTL.

All four trials use the same kernel XO, SHA256
`5db5e219757a1962ce57c3d8c0f3bf7f5a80eefcb42d564448fce775fe9a52a1`.
The T4 functional checks and C-oracle evidence are in
`build/analysis/k4_t4_vu_20261105/completion.json`.
The first T4 conflict signal hierarchy counts are in
`build/analysis/k4_t4_route_failure_20261105/hierarchy_counts.json`.
A read-only `report_route_status` query on the final error DCP now lists 31,191
unique conflicted net names, one fewer than the native log. The first T4 query also
returned one fewer name than its log. The final evidence is in
`build/analysis/k4_t4_final_failure_20261105/`, including `comparison.json`,
`hierarchy_counts.json`, the exported names and the query log. This is a complete
returned net-name list, not a physical collision-location census. No such census
was run on the last two failures. Their top-ten node examples do not establish
where all physical conflicts lie.

### Final failure analysis

The user requested this analysis after pausing the goal. The goal remains paused.
The only new CAD command opened the saved error checkpoint and read route status.
No synthesis, placement, routing, RTL or constraint change was made.

The native error is `Route 35-2`: 35,335 node overlaps remain. Zero unrouted and
partly routed nets means paths were connected, not that the routes are legal.
Pre-route clock/reset and CDC checks passed. This was a native routing failure,
not an inspection hook stopping an otherwise successful route. The negative
intermediate timing is not a final frequency result from a legal route.

SLR1 remained the principal reported congestion area. It held 265,510 LUTs (67.37%)
and 48,524 occupied CLB sites (98.51%), versus 78.42% and 75.94% occupied sites in
SLR0 and SLR2. A used CLB site need not have all its LUTs or registers filled;
98.51% is not LUT utilization or a proof that no further packing is possible.
The final south-going global hotspot covered 64x64 tiles at
`INT_X0Y388` to `INT_X63Y451` in SLR1. Nine of the ten ranked physical examples
were central; one was inside cluster 3 in SLR2.

Moving DMA back from SLR2 to SLR1 helped compared with the 07:50 placement.
Overlaps fell from 60,788 to 35,335 (41.9%), and placed SLR1-2 connections fell
from 9,536 to 6,691. Peak estimated SLR1-2 column demand fell from 164% to 101%.
This supports the earlier DMA relocation as an aggravating factor, although each
Pblock change also changes the global placement.

The final placement still performed worse than the first T4 attempt. It had
30,668 fewer SLR1 LUTs but 4.39 times as many overlaps. SLR0-1 connections grew
from 5,388 to 7,429 (37.9%). Peak estimated demand in one lower-boundary column
grew from 87% to 110%. The upper-boundary peak was 104% originally and 101% finally.
These are router estimates per physical column, not measured directional SLL
occupancies. Device-wide vertical/horizontal routing utilization was only
27.88%/24.94%; the failing resource is concentrated locally.

The retained VU frontend placement is the leading explanation for this added
lower-boundary traffic. Its reader and injector sit in SLR0 while SPM remains
in SLR1 and the other lane bank is in SLR2. The source-level bank interface still
carries 544 input bits and 192 output bits per bank. Those are logical interface
widths, not measured SLL counts. Moving logic out of the middle did not remove
its communication with the middle or the opposite bank. More crossings and
central long-wire conflicts are consistent with that mechanism. The experiments
do not isolate a single net or block as the exclusive cause.

The full returned conflict-name list shows the effects across shared interfaces:

| Net-name hierarchy | First T4 | Final T4 |
|---|---:|---:|
| DMA, including KVST | 2,263 | 5,051 |
| KVST subset of DMA | 1,698 | 3,670 |
| Platform DDR1 controller and interconnect | 2,870 | 5,045 |
| Debug and trace | 196 | 4,042 |
| SPM | 836 | 2,798 |
| Instruction issue | 199 | 2,063 |
| All VU hierarchies | 875 | 1,355 |
| MVM gather | 542 | 535 |

The debug count includes 2,577 shadow-queue signals and 916 trace-writer signals.
These figures count affected signal names. They do not assign blame or physical
node counts to a block; congestion created elsewhere can obstruct these nets.
In particular, neither the top-ten examples nor the full name count supports
calling this a VU-only failure or proving that removing the RV core fixes it.

T_MAX=4 substantially reduced the cluster arithmetic, but did not proportionally
shrink the central DMA/KVST, 512-bit SPM interfaces, issue/debug transport or
platform DDR interconnect. The DMA and SPM instances in `tsr_core_dp.sv` do not
receive T_MAX. The first T4 placement still used 296,178 central LUTs, close to
299,410 in the preceding T8 major-Pblock placement. This explains why halving
compute capacity did not by itself clear the central bottleneck.

For the next worker, the original sixteen-Pblock T4 placement is the best T4
reference so far, but it also failed. The evidence favors reducing or localizing
shared central traffic before moving more whole blocks away from their data.
Use matched feature/transport comparisons to distinguish DMA/KVST, debug/issue,
SPM and VU costs. No follow-up implementation or retry has been started.

### Follow-up investigation while paused

At the user's request, Sol investigated central communication and consumer locality,
and Astra reviewed the proposals. `platform-u200.md` §8.6.11 records the current
interface rates, candidate boundaries and limitations. Priority is a narrower VU
operand packet with local coefficient storage, reduced central debug formatting
and queue read logic, then channel-local KV metadata merge. The review rejects
counting lower runtime activity as fewer wires or counting TRW relocation as an
automatic 512-to-256-bit crossing reduction. No RTL or constraints were changed,
and no tests, synthesis or implementation were run. The goal remains paused.

### Goal and implementation history

The paused goal targets four clusters with T_MAX=4 and eight physical VU front/EXP lanes.
Each outer SLR has four lanes, two with BACK support. The design retains independent
MVM/VU issue, four 512-bit weight streams, DDR0/3 locality, paired gather, full
attention and the existing numerical precision. Use only major-block Pblocks.
Signoff requires a legally routed xclbin, core at least 250 MHz, RV near 166.7 MHz,
passing setup/hold and the existing clock/reset and CDC checks.

The preceding run started at 07:50:18 KST in
`build/link/k4_20261105_075018_t4_front_dma_core250_rv166/` (supervisor 4026345).
The [shown Pblock plan](fig/floorplan/k4-pblock-plan.png) retains the original 16
major assignments and adds the six complete VU frontend submodules in SLR0 and DMA,
including KVST, in SLR2. It uses the same kernel XO. Individual cells and stages
remain under native placement and optimization. The source and launch evidence is in
`build/analysis/k4_t4_front_dma_pblocks_20261105/`.

Placement finished at 09:15 KST. SLR1 LUTs fell from 296,178 to 259,620 against
the first T4 run. Occupied CLBs fell from 99.86% to 97.80%, leaving 1,084 empty.
SLR0/2 occupied 80.55%/75.86%. SLL use grew from 12,213 to 16,917, and the final
in-placer WNS estimate changed from −0.975 to −2.205 ns (TNS −7,678.399 ns).
SLR1 north/south global congestion remained 64×64, while east/west fell from
32×32 to 16×16. At 11:27 KST route verification reported 58,757 conflicted
signals and 60,788 node overlaps. Northbound SLL demand reached 164% in one column.
Several top-ranked nodes involve DDR0 response signals crossing from SLR0 to the
relocated DMA in SLR2. Other examples include VU operands and bank returns.
The next comparison restores the whole DMA block to SLR1 and keeps the VU frontend
in SLR0, changing one hierarchy assignment. The native run exited at 11:33:26 KST and preserved its error DCP with unchanged
source and packaged inputs.

The new link started at 11:34:12 KST in
`build/link/k4_20261105_113412_t4_dma_center_core250_rv166/` (supervisor 62883).
It returns only `u_dma` to SLR1, keeping VU frontend in SLR0 and the other 22
hierarchy assignments fixed. The input XO is identical. The per-run Pblock map,
patch and launch record are in `build/analysis/k4_t4_dma_center_20261105/`.
The 07:50 run's pre-build map is preserved under
`build/analysis/k4_t4_front_dma_pblocks_20261105/pblock-plan.png`.

This run completed placement at 12:59 KST. SLL use fell from 16,917 to 14,120
against DMA in SLR2; the upper boundary fell from 9,536 to 6,691 connections.
SLR1 has 265,510 LUTs and 98.51% occupied CLBs, versus 259,620 and 97.80% in the
preceding placement. Final in-placer WNS improved from −2.205 to −1.645 ns.
SLR1 north/south congestion was still 64×64. The final route failure is recorded above.

### Earlier T4 attempts

The [initial T4 Pblock map](fig/floorplan/k4-major-pblocks-nov4.png) was shown before launch. The
normal K4 generator selects T_MAX=4. The new hardware link started at 02:00:15 KST
in `build/link/k4_20261105_020015_t4_vu_core250_rv166/` (supervisor 3590967).
It packaged a fresh kernel from the main checkout. The focused test and launch
receipts are under `build/analysis/k4_t4_vu_20261105/` and
`build/analysis/k4_t4_vu_build_20261105/` respectively.

T4 integrated core smoke/debug, MVM back-to-back and QK/PV tests passed. In three
attention fixtures (causal prefill, grouped heads, split-head sliding-window decode),
all 2,560 output words matched both the C oracle and the T8 reference bit for bit.

Synthesis finished at 02:23 KST. Against the same bank-local VU on the T8 build,
kernel LUTs fell from 526,163 to 378,867, FFs from 605,139 to 435,139, DSPs from
3,176 to 1,768 and BRAM tiles from 1,007 to 599. Both used 128 URAM. The kernel
netlist contains four clusters, four PV blocks, two VU lane banks with local reduction,
DMA/KVST, the RV subsystem and debug.

Placement finished at 03:48 KST. SLR0/1/2 occupied 76.07%, 99.86% and 76.31% of
CLB sites. SLR1 used 296,178 LUTs (75.16%) and had 71 empty CLBs. Against the completed
T8 VU-on placement with central first-level reduction, SLL use changed from 12,233
to 12,213. The maximum SLR1 global congestion regions remained 64×64 north and south.
The final in-placer WNS was −0.975 ns and TNS −621.591 ns. The native link exited
with code 1 at 06:12:47 KST. Route verification counted 9,938 conflicted signals and
8,054 node overlaps, with zero unrouted or partly routed nets. Nine of the ten listed
physical conflict nodes were SLR2 long wires used by the upper paired gather link;
the tenth was in the SLR1 platform DDR1 PHY. The final north/south congestion windows
lay in SLR1. The samples do not describe the distribution of all conflicts. The failed
run retains its 550 MiB routed error checkpoint, logs and the verified source package.

The saved error DCP lists 9,937 distinct conflicted net names, one fewer than the
router message. The DMA hierarchy owns 2,263 (including 1,698 in KVST), the DDR1
controller and its interconnect own 1,865 and 1,005 respectively, SPM owns 836,
and MVM gather owns 542. These are signals, not counts of physical nodes. See
`build/analysis/k4_t4_route_failure_20261105/hierarchy_counts.json`.

The next comparison releases the gather assemblers, DDR bridge/arbiter hierarchy, SPM
and RV/control to Vivado. Only four clusters and two VU lane banks have kernel Pblock
assignments. The six-block Pblock plan in `build/analysis/k4_t4_relaxed_pblocks_20261105/pblock-plan.png` was shown
before this retry. The prior input XO is reused unchanged.

The six-block link started at 06:15:30 KST in
`build/link/k4_20261105_061530_t4_six_core250_rv166/` (supervisor 3818127).
It uses a private copy of the 02:00 run's IP cache. The kernel XO and source manifest
match the 02:00 run byte for byte.

The link ended at 07:45:20 KST with `Place 30-433`, unplaced instances.
Vivado printed one example, a generated reset-forward register in the platform C2H
AXI auto-pipeline. Its default report gives no total unplaced count. No failed
placement DCP or route result was saved. Source/package checks passed. Available
device-wide resources did not establish a global resource overflow, and the earlier
SLR1 partition warning also occurred in the previous build that completed placement.


### Stopped VU-absent routing experiment

The T_MAX=8 diagnostic stopped at the user's request at 01:57:53 KST on November 5
in initial routing phase 4.4. Its source and package integrity checks passed, and all
owned processes exited. The comparison kept four clusters,
T_MAX=8, ATTQK/ATTPV, DMA/KVST, the RISC-V core and debug, four 512-bit DDR ports,
the major-block Pblocks and the 250/166.7 MHz clock request. The vector-backed
attention operations cannot run without the VU. Disabling both existing `HAS_VU`
and `HAS_ATT` would also remove the cluster and DMA attention hardware.
The design priority is to preserve numerical accuracy when reducing area. Retain the
q8_0 KV layout and the existing PV arithmetic.
The diagnostic uses an isolated source checkout and XO cache. Its
[pre-build Pblock plan](fig/floorplan/k4-no-vu-pblock-plan.png) omits the two VU banks.
The link started at 22:45:56 KST. Its directory is
`build/link/k4_20261104_224556_no_vu_core250_rv166/` (supervisor 3349470). Its
source is `/home/limagin-dev/tesseract-k4-novu-20261104`. A directed integrated RTL
test boots the RV, issues KVST, ATTQK and ATTPV, and checks the DMA/MVM retirement
tickets. It also verifies rejection of vector instructions and absence of the public
VU and full ATT capabilities. A second run checked 64 nonzero QK scores, 256 nonzero
PV output words and both KV planes in every DDR channel against the C oracle.
Synthesis finished at 23:09 KST. The synthesized EDIF contains four cluster instances,
four PV instances, Ibex, the debug block and DMA/KVST. The VU instance is absent.
Kernel LUTs fell from 526,163 to 455,448, FFs from 605,139 to 531,428 and DSPs from
3,176 to 2,917 against the stopped bank-local VU trial. Physical fit and both clocks
await the link result.

Placement finished at 01:06 KST on November 5. Against the last VU-on major-Pblock
placement, SLR1 CLB occupancy fell from 99.78% to 98.06% (851 fewer occupied sites).
SLR1 LUTs fell from 299,410 to 259,041. The north and south SLR1 global congestion
windows both fell from 64×64 to 32×32. SLR2 south still reached 64×64, and short-wire
congestion in SLR1 reached 64×64 in all four directions. SLL use fell from 12,233
to 10,702. The final in-placer WNS estimate improved from −1.134 to −0.790 ns.
Vivado still warned of high congestion and entered post-place physical optimization.

### Stopped bank-local VU trial

The 21:22 KST link was stopped at 22:38:34 in global placement, phase 2.1. Its
directory is `build/link/k4_20261104_212230_bank_l1_core250_rv166/`.
Its source, packaged inputs, synthesis checkpoints and logs are preserved.
It repackaged the three changed VU sources, used the same major-block
Pblocks shown before launch, and requests core 250 MHz and RV 166.7 MHz. The previous
run's custom kernel IP cache entry is excluded. The 384-bit return from each bank is now
192 bits. The three promoted RTL files and regression match the functionally tested
candidate under `build/analysis/k4_slr1_relief_20261104/`. It did not reach routing.

### Previous major-block Pblock link

At the user's request the 11:34 DMA-selector build stopped at 15:34:26 KST on November 4.
Placement had finished and `phys_opt_design` was running. Routing had not started. Its
owned processes exited and its source and package integrity checks passed. The original
directory retains the placement reports, post-opt checkpoint and `user_stop_complete.json`.
Use its level-6 placement results as the comparison baseline, not as a route failure.

The user saw the [pre-build Pblock map](fig/floorplan/k4-major-pblocks-nov4.png) before the next
native link started at 15:36:14 KST. Its directory is
`build/link/k4_20261104_153614_pblocks_core250_rv166/`, supervisor 3091196. The new link
uses the identical packaged kernel XO and a private copy of the previous Vitis IP cache.
The ordinary placer now receives `k4_major_blocks.tcl` only. Individual pipeline-stage
and helper constraints, the placement ECO, legacy post-placement ownership checks, and
the forced placer directive are absent. Clock/reset and CDC hooks remain. The watcher reads
native reports for placement and terminal events.

Placement finished at 18:05 KST. The 11:34 and 15:36 runs used the same kernel XO.
In `impl_1/slr_util_placed.rpt`, SLR1 used 299,410 LUTs, down from 300,620.
Occupied CLBs fell by 17, from 99.82% to 99.78%. SLL crossings increased from
11,850 to 12,233. In the placer's final congestion table, the SLR1 east global region
shrank from 32×32 to 16×16. North and south stayed at 64×64. Across the device,
east and west long-wire regions grew from 16×16 to 32×32. The final in-placer WNS
estimate improved from −2.077 to −1.134 ns. After physical optimization it was
−1.074 ns. Routing started at 18:35 KST. The pre-route clock/reset and CDC checks
passed (zero unsafe or unconstrained CDC paths). The global router estimated SLL
demand at 34.91% across SLR0–1, peaking at 97% in one column, and 47.89% across
SLR1–2, peaking at 88%. Initial routing reported global/short congestion level 6 and
timing congestion level 7. At 21:12 KST route verification reported 24,587 conflicted
signals and 19,799 node overlaps. The counts for unrouted and partially routed nets were
both zero. The earlier half-width build had 71 overlaps. Nine of the ten physical nodes
listed in the new log were in SLR1. The examples include platform DDR, channel 0 DMA
write data and VU/SPM paths. This list does not rank blocks by total conflicts.
The supervisor exited at 21:21:52 with code 1 and unchanged source and package hashes.
The original run retains its 692 MiB `impl_1/level0_wrapper_routed_error.dcp`.

For the next trial keep these Pblocks and move the first VU reduction level into the
two existing outer banks. The half-width bank's registered return narrows from 384 to
192 bits per stage. Isolated functional tests and ordered writes passed. The physical
benefit needs a new route. See `platform-u200.md` §8.6.10.

### Previous DMA-selector build and placement-policy change

The user limited future placement work to major-block Pblocks on November 4. The 11:34
link keeps its original configuration. Subsequent links should use hierarchical block
assignments and Vivado placement/routing within those regions. Omit the old leaf ownership
walks, stage pinning and individual-cell placement ECO. The bank-local reduction RTL was
promoted after the 15:36 build finished. The optional whole-VU audit was
stopped, and further queued probes were withdrawn. Keep functional clock/reset and CDC
constraints and final route/timing signoff.

The configuration applied at 15:36 is preserved in
`build/analysis/k4_major_blocks_20261104/major_blocks.patch`. It assigns whole clusters,
assemblers, VU banks, outer DDR bridges/arbiters, SPM and control to platform Pblocks.
It removes the old placement checker/ECO and forced placer directive. No extra CAD was
started to prepare it. The four files were applied after the prior run's source freeze ended.
The new launcher accepts `k4_major_blocks.tcl` and the CDC-only pre-physopt hook.
Show the user [the pre-build Pblock map](fig/floorplan/k4-pblock-plan.png) before each
new K4 build. It uses the prepared hook and the U200 platform's retained SLICE ranges.
`hw/viz/k4_pblock_preview.py` renders the plan offline. Update the map when assignments change.

The first half-width link failed route verification at 11:22 KST on November 4. Vivado
reported **104 conflicted signals and 71 node overlaps**, down from 16,100 and 13,267 on
the full-width run. It wrote its error DCP and exited naturally at 11:32:48, code 1.
Both source and package integrity checks passed. Its `preserved_half_route_failure/manifest.json`
verifies 1,200 copied files (1,470,206,871 bytes), including 179 checkpoints. The printed
examples included MVM broadcast and channel 3 DMA write wires. The subsequent census
covered all 104 signals and all 71 physical nodes. SLR1 has 67 nodes and SLR0 has four.
Channel 3 DMA write-data participates in 49, including 33 conflicts between its own
bits. Each of its 53 affected payload signals runs from SLR1 to SLR2. The previous
full-width KVST hotspot did not dominate this netlist. Evidence and endpoint coordinates
are in `build/analysis/k4_half_error_20261104/session/` and `platform-u200.md` §8.6.9.

The reviewed DMA/KVST selector is now in the main checkout. Its two RTL files are the only
runtime differences from the previous source snapshot. The new fresh link started at
11:34:09 KST, supervisor 2794845, in `build/link/k4_20261104_113409_central_rsp_core250_rv166/`.
`build/analysis/k4_active_attempt.txt` points to this run. It uses the same 8/8/4 VU,
four clusters, floorplan and directives, 250 MHz core and 166.7 MHz RV. This comparison
isolates the selector change at the design-input level. The v++ build forces a fresh XO
and targets a complete image. Root monitors errors and placed reports via background cell
`g8:294`. Final legal routing, timing, clock checks and the xclbin are pending.
Placement completed on November 4. At 15:17 KST its reports showed 300,620 SLR1 LUTs
(895 fewer than the preceding half-width run), 99.82% occupied SLR1 CLB sites and 11,850
SLLs (33 more). Maximum global congestion stayed at level 6, with 64x64 north/south
windows. The east/west long-wire windows shrank from 32x32 to 16x16. Placed WNS was
-2.077 ns, TNS -134.216 ns and WHS -0.299 ns. Physical optimisation followed. The matched
reports are in `build/analysis/k4_central_delta_20261104/placed_comparison.json`.

### Evidence from the preceding half-width attempt

The full-size locality attempt failed actual routing and exited naturally at 03:22:24 KST
on 2026-11-04, native code 1, with source integrity unchanged. Verification rejected 16,100
signals with routing conflicts and 13,267 node overlaps. The router's preceding summary had
zero unrouted or partially routed nets: paths existed but illegally shared resources.
The corresponding earlier failure reported 286,682 failed signals and
352,035 overlaps. The reduction is substantial, but neither run is legally routed.
This is a physical failure, not another launcher or floorplan-checker failure.

The user's conditional fallback is now active: Sol implements FRONT8/EXP8/BACK4 while keeping
four clusters, T_MAX=8, four independent 512-bit weight streams and independent MVM/VU issue.
Retain 16 logical reduction histories, the original accumulation/tree order and 512-bit
architectural SPM lines. Half-width banks have four physical lanes each and two back units
each: bank 0 physical lanes {0,1,4,5}, bank 1 {2,3,6,7}. Physical lane l retains logical
contexts l and l+8. NEOX needs separate line/half subslots. No numerical change is approved.
Astra approved the final source after implementation and focused functional validation.
Sol locality owns VU RTL/tests; Sol resume owns parameter plumbing, K4/OOC configuration,
floorplan and integration checks. Root owns documentation and the next native launch.
The goal remains active at core >=250 MHz and RV about 166.7 MHz; its older role/width wording is
superseded by the user's later instructions. Luna remains terminated.

The VU RTL and integration sources are now frozen. All 78 VU oracle case executions pass
across six full/half-width pipeline configurations. Directed context, cancellation, tail,
mixed-exit, NEOX and active-reset cases pass. Elaboration confirms actual 8/8/4 arithmetic,
balanced banks and all 16 logical histories. M7 half-width decode/prefill and four-channel
K4 smoke/DB pass. Software build, CTest, table and top checks pass. The final complete
floorplan suite passes all 157 fixtures after correcting an eager bank census on legacy
empty-helper paths; the original 63-failure log is preserved in `build/k4_half_floorplan_all.log`.
Final passing evidence is `build/analysis/k4_half_vu_20261104/floorplan_all_final.log`.
Full RTL lint passed all RTL, nine XSim elaborations and eight Vivado synthesis checks.
Four TB targets initially failed on an unconnected optional drain output. An explicit unused
TB wire fixes the warning; all four strict reruns pass without waivers or RTL changes.
Evidence is `build/analysis/k4_vu_half_20261104/tb_connection_lint_completion.json`.
The default bench suite passes: 518 passed, 115 opt-in skipped, no failures, 5,906.37 seconds.
Evidence is `build/k4_half_default_tb.log`. Background cells `g8:192` and `g8:181` are finished.
The original failed floorplan result is superseded by the explicit `floorplan_final` result.

The next fresh driver is `build/analysis/k4_half_vu_build_20261104/supervise.py NEW_OUT`.
It uses one top-level slot, 16 internal synthesis jobs and eight implementation threads,
forces a new XO, preserves source/package hashes and runs through image generation.
The production `hw/u200/k4_post_opt.tcl` hook saves `k4_post_opt.dcp` before the floorplan
gate and propagates write failures. Hook/config/supervisor checks pass (35 combined cases).
Astra approved the driver and final compatibility fix. All required pre-launch gates pass;
31 frozen source/driver hashes and the corrected TB lint evidence have been verified.
The broad bench suite completed while the native build ran.
The fresh half-width attempt started at 04:16:21 KST on 2026-11-04, supervisor PID 2457599:
`build/link/k4_20261104_041621_half_vu_core250_rv166/`.
The fresh XO was packaged. Placement completed in 1 h 47 min 39 s, and the actual placed-locality
gate passed. By 07:48 KST the placement reports were complete and physical optimisation began.
Against the full-width locality run, SLL falls 13,141 to 11,817 (-10.1%), LUT use falls 3.1%
and FF use falls 4.0%. Central CLB site occupancy remains 99.84%. Maximum global congestion
remains level 6; north/south stay 64x64, east grows 16x16 to 32x32. WNS is -1.944 ns versus
-1.874 ns previously. No physical success is established. See `platform-u200.md` §8.6.5.
At 09:03 KST the half-width build was in initial routing.
Initial SLL demand is lower overall, but the SLR1-2 peak column grows from 85% to 108%;
the SLR0-1 peak changes from 97% to 94%. A total crossing reduction does not establish
relief at each physical column. The native router has emitted its congestion warning.
The read-only census of the previous full-width error DCP is complete. Of 13,263 unique
physical conflict nodes, 12,893 (97.2%) are in SLR1; SLR0 has 102 and SLR2 has 268.
Four triple-owner nodes reconcile the router's 13,267 overlaps. All 16,100 failed signals
are covered, with no unknown tile SLR. The strongest cross-hierarchy pair is kernel DMA
versus platform DDR1 (853 nodes), including 716 involving DMA's KV block. See
`platform-u200.md` §8.6.6 and `build/analysis/k4_route_census_20261104/node_worker/session_objects_20261104/`.
At 10:28 KST the half-width route was in rip-up/reroute Phase 5.2; intermediate overlap
counts are not a terminal result. The native run remains untouched.
`platform-u200.md` §8.6.7 compares the October card image and both integrated runs at
the same placement stage. The October design used more SLLs than the failed locality
run (13,724 versus 13,141), but left 3,889 empty SLR1 CLB sites versus 60. Its host
still handled attention, normalization and RoPE. The half-width run leaves 79 sites.
The central wiring change was developed in the isolated worktree
`/home/limagin-dev/tesseract-k4-central-20261104`. Sol changed the DMA and KVST response
input to use one four-channel data selector and added a mixed LD/ST/KVST regression.
Astra approved the saved candidate. The original placed/routing run used the previous
RTL and finished before promotion to the main checkout. The candidate patch,
hashes, cycle counts and synthesis reports are in
`build/analysis/k4_central_delta_20261104/dma_selector/completion.json` and its directory.
The K4 DMA OOC comparison saves 736 LUTs and 2,048 direct response-bus loads. All 72 tested
command latencies match. See `platform-u200.md` §8.6.8 for source and hierarchy comparisons.
The candidate also passes all three core integration cases: M7 half-VU decode, prefill,
and the four-channel locality smoke. The log is `dma_selector/candidate_core.log` in the
same evidence directory. SLR1 reduction options and their costs are compared in
`platform-u200.md` §8.6.9.
Root's error watcher for that attempt used session 15336 and background cell `g8:195`.
A separate read-only placement milestone watcher used session 20519 and cell `g8:199`;
it notified root after the reports finished, following 6,742 seconds of model-free sleep.
That watcher and cell are now finished. Astra independently reviewed the placed comparison.
These loops run without periodic model invocations. Cycle evidence and its limits are
in `platform-u200.md` §8.6.4.

Root preserved 100 files (1,316,113,151 bytes), including the routed-error checkpoint
(744,673,765 bytes), opt checkpoint, reports/logs, XPR, configuration, XO and working-tree
patch, in the failed attempt's `preserved_route_failure_20261104/manifest.json`.
All 177 synthesis DCPs were reverified against their preserved copies. Retain this evidence;
changed RTL requires a fresh XO and build directory, not a resume from these checkpoints.

### Last full-size locality attempt

The latest native attempt started at 21:21:25 KST on 2026-11-03, supervisor PID 2150171:
`build/link/k4_20261103_182205_locality_core250_rv166/attempts/20261103_212125_guarded_impl_drain_proven/`.
Its `status.json` records the terminal routing failure. It reused all 177 synthesis DCPs and
ran native init/opt, placement and routing. Core was 250 MHz, RV about 166.7 MHz, and the VU
was full size. Root monitored the attempt directly.

By 00:07 KST on 2026-11-04, placement completed successfully (2:00:10 elapsed), and the
actual placed-locality gate passed (`K4FP locality PASS (placed=1)`). Compared with the previous full-system run at
the same Fully Placed stage, SLL count fell 17,205 to 13,141 (-23.6%) and the outer SLRs are
balanced. Central occupied CLB sites remain 99.88%, and the high-congestion warning remains.
WNS is -1.874 ns. The comparison is qualified in `platform-u200.md` §8.6.3; the previous run
used a 300 MHz core target and different CDC constraints. This is not timing closure.
Clock/reset and pre-implementation CDC pass.
The complete native pre-place gate passed at 21:57; `actual_pre_place_gate.txt` records it.
The current attempt also saved its own post-opt DCP (456,667,721 bytes). Pre-implementation
CDC passed with unsafe 0 and unconstrained 0. Routing later failed as recorded above.

Monitoring changed on 2026-11-04 after the user raised token cost. Luna's tool waits had
resumed its model about once a minute despite quiet output. At the user's explicit request,
root interrupted Luna and retired only its read-only watcher PID 2150434 at 00:37:39.
CAD was untouched. Do not restart Luna. Root's replacement watcher used exec session 82891
and background code cell `g8:164`, with tool polling inside the background cell and `notify`
only for an event or watcher failure. The notification woke root at 03:22 after 9,681 seconds
of uninterrupted clock sleep, with no periodic model log reads. That watcher and cell are
now finished; start a new pair only for the next native attempt. A local probe also verified
that notifications wake `clock.sleep`.
Use interruptible clock sleep while a background cell runs; do not poll `functions.wait`
or CAD logs every minute. On an event, collect the yielded cell's completion and investigate.
Python still checks errors/liveness every 10 seconds and writes samples every 15 minutes;
those loops do not invoke a model. Root also stopped duplicate 15-minute model log reads.
Keep ordinary progress quiet. Report errors, stoppage, completion and requested milestones.
The placement comparison has already been completed by Astra.

The complete production floorplan passes on the previous attempt's actual post-opt DCP.
All 184 optimized wrapper helpers are LUT3/4 reduction-drain muxes: their exact bank output
bits and delayed reduction selectors feed the matching central reduction register. The guard
proves those connections and SLR1 ownership; it does not accept arbitrary wrapper names.
All 128 floorplan fixtures pass. Astra approved. Evidence is in
`build/analysis/k4_wrapper_provenance_20261103/`; the DCP hash is unchanged.

The native scratch-project probe rejected a place-only restart because this run has no
registered opt input checkpoint. Enabling intermediate checkpoint generation afterward did
not supply one. Both scripts-only probes preserved all 402 protected live files. Use the
proven full-init/opt restart. Its driver now unwraps only verified previous recovery hooks,
sources the archive-verified native hooks directly, and protects old checkpoints/receipts
from being overwritten. All 49 driver regressions pass and Astra approved the repair.
No RTL changed for these guard and recovery fixes. Physical implementation remains unqualified.

### Third guarded resume: optimized VU wrapper ownership

The 20:29 attempt failed before placement at 21:04:31 KST; native exited naturally at
21:05:22, code 1, with source integrity unchanged. The complete post-opt floorplan gate
rejected `g_vu.u_vu/u_lanes/g_q[0][0]_i_1` as unknown banked VU ownership. The earlier bank
arithmetic-helper proof passed. Sol resolved the optimized wrapper ownership and the complete
actual-DCP floorplan check passed as described above. No placement or route result exists
from this attempt, so the half-width fallback was not triggered.

This attempt saved `post_opt.dcp` (456,670,630 bytes) before the failing hook. Root preserved
71 files (489,844,071 bytes), including the DCP, project, generated hooks and diagnostics,
with verified hashes in `preserved_pre_place_failure/manifest.json` under the attempt below.
The custom checkpoint is not automatically a native opt checkpoint. Luna reported the failure
and now monitors the newer attempt. The VPL failure summary re-echoes the old 176-run error; the fresh implementation
session identifies only the new `g_q` floorplan failure as its primary cause.

The corrected native run started at 20:29:12 KST on 2026-11-03, supervisor PID 2144510.
That historical attempt is
`build/link/k4_20261103_182205_locality_core250_rv166/attempts/20261103_202912_guarded_impl_verified_inventory/`.
Its `status.json` records its failure. At 20:29:36 Vivado
verified the existing synthesis inventory and launched `impl_1` through `write_bitstream`
with eight jobs. All 177 synthesis DCPs are reused. Placement and routing are still pending;
this launch receipt is not a physical success result. This attempt is now terminal as above.

The final recovery driver passes 36 regressions and Astra review. A read-only Vivado probe
confirms 98 completed and 79 cached synthesis runs, all at 100%. The guard checks their exact
names, directories, source sets and preserved DCP mapping. Native hook paths also match.
Only `hw/u200/k4_floorplan.tcl` changed from the original runtime source snapshot; the design
RTL and packaged inputs are unchanged. The full pre-place floorplan check still runs after
opt, and the recovery saves a post-opt checkpoint before it.

### Second guarded resume: synthesis inventory check

The 20:22 attempt exited at 20:22:56 before implementation launch: the resume guard expected
176 synthesis runs, while the actual project and archived DCP inventory both contain 177.
Cleanup was natural and source integrity unchanged. Sol replaced that assumed count with
the verified run identity set; Astra approved it. The old Complete-only test also needed to
accept verified cached synthesis results at 100%.

The repaired launcher started at 20:22:19 KST on 2026-11-03, supervisor PID 2143400.
That historical attempt is
`build/link/k4_20261103_182205_locality_core250_rv166/attempts/20261103_202219_guarded_impl_isolated_python/`.
Its `status.json` records the failure. At 20:22:35 the actual
Vivado process 2143799 had entered the guarded Tcl and was opening the existing project.
This is confirmed by `guard_entry_started.txt` and `vivado_invocations.jsonl`; a live VPL
process alone does not prove implementation progress. Placement and routing have not yet
been observed in this attempt. All 177 synthesis DCPs are reused.

The launcher repair passes 24 regressions, including the archived Xilinx Python/library
environment. Isolated Python startup retains the CAD environment for Vivado. Fresh startup
failures are recorded as the primary cause before cleanup; old log errors are excluded.
Astra approved. The full archive, package and synthesis integrity checks pass. Luna monitors
this active attempt and reports only actionable events. No RTL changed for either recovery.

### First guarded resume: Python startup failure

At 20:17:41 KST the first guarded resume was stopped after a launcher failure. The new
suffix of native `vpl/runme.log` records `Fatal Python error: Failed to import encodings
module`; the guarded Python entry exited before starting Vivado. VPL remained alive waiting
for that failed child. No placement or routing ran. Root preserved the logs and process
evidence under that attempt's `bootstrap_failure_evidence/`, then stopped its supervisor.
Owned processes exited and source integrity remained unchanged. Sol is correcting Python
startup isolation and failure propagation; Astra reviews the fix. Reuse the 177 synthesis
DCPs. This failed attempt is preserved separately from the active attempt above.

The first corrected implementation resume started at 20:09:32 KST on 2026-11-03,
supervisor PID 2142245. That attempt is
`build/link/k4_20261103_182205_locality_core250_rv166/attempts/20261103_200932_guarded_full_impl_resume/`.
Its `status.json` records the terminated attempt.
The original build-root `status.json` deliberately retains the first attempt's failure; do not
read it as the resumed run's status. Native VPL accepted `--from_step vpl.impl` for the existing
failed run. The guarded entry rejects project recreation or synthesis and retains all 177
checked synthesis DCPs. It saves a post-opt DCP and requires the complete native pre-place
floorplan gate before placement. Final routing and timing remain unqualified.

The corrected guard passes 107 fixtures and an actual synthesized-netlist probe of both VU
banks, DDR0/3 bridges/arbiters/WS and all five central DMA slices, plus both gather pairs.
All 424 reparented arithmetic helpers have known-lane connections on both sides and exactly
one lane owner. Cross-bank, central, unanchored, unsupported and misplaced cases are rejected.
Astra approved this proof and the recovery driver; its 17 regressions pass. No RTL changed
for this correction. The whole floorplan still needs its actual post-opt native check: the
pre-opt DCP has an existing BC-head shape that opt removes, so it is not that gate's input.
Luna `/root/luna_watch` reported its termination and now monitors the repaired launch above.

### First locality attempt: pre-place guard failure

The fresh native attempt below failed before placement on 2026-11-03. At 19:47:20 the
floorplan guard rejected a synthesized VU bank helper:
`u_lanes/g_bank[0].u_bank/g_back.u_rcp/p1b_q_reg_i_1` (under the kernel's `g_vu.u_vu`).
Synthesis reparented the helper into the bank. Sol corrected the guard using actual netlist
connectivity; Astra approved the proof and negative fixtures. No placement or
route ran, so this does not trigger the conditional half-width VU fallback. The goal is active.

Native returned 1. At 19:48:10 the supervisor also found remaining native control processes
and cleaned up its owned tree; its top-level error records that secondary cleanup condition.
The implementation log contains the primary floorplan failure. Runtime source and package
integrity were unchanged through the attempt. Verified copies of 259 files (222,892,315 bytes),
including all 177 synthesis DCPs, the project, generated hooks, implementation reports/logs and
status, are under `attempts/20261103_194810_pre_place_name_failure/preserved/` in this run.
This full native flow did not write an opt DCP before the failed pre-place hook. Reuse synthesis
and rerun init/opt with native hooks; there is no opt checkpoint to resume. Luna reported the
failure and ended this monitor assignment. Reassign it to the next attempt.

A fresh full native build started at 18:22:05 KST on 2026-11-03 in
`build/link/k4_20261103_182205_locality_core250_rv166/`, supervisor PID 2056316.
`build/analysis/k4_locality_latest.txt` and `k4_interconnect_latest.txt` point to it.
It requests core 250 MHz / RV 166.666667 MHz, synthesis jobs 16, implementation jobs/threads 8.
XO repackaging completed: all 138 packaged source entries match the 953-file execution
snapshot. Native synthesis started at 18:24:21 with 16 concurrent jobs; all 176 finished at
19:10:56 and top-level synthesis completed at 19:11:53. The implementation worker is confirmed
live at 19:28, when it entered `opt_design` DRC before the later pre-place failure. The synthesized clock/reset
gate passes at core 4.000 ns / RV 6.023 ns; pre-implementation CDC passes with unsafe 0,
unconstrained 0 and the 3.333 ns crossing bound. Its BD clock/reset gate also passes:
`ap_clk` uses `aclk_kernel_00`, `ap_clk_2` uses `aclk_kernel_01`, with separate reset generators.
Physical locality, routed CDC and timing checks remain pending. `status.json` in this directory is authoritative; the supervisor
snapshots execution inputs and package hashes. Luna `/root/luna_watch` monitors this attempt
only and reported its failure to root. Ordinary samples were quiet.

The current source implements DDR0/3 bridge and arbiter localization, gather consolidation for
cluster pairs {0,2}/{1,3}, and balanced VU banks across SLR0/2. Sol implements, Astra reviews,
and Luna is reserved for build monitoring only. The user corrected the original goal's role
text; do not use Luna for implementation or review. The goal is active: a legal full route and
timing-qualified xclbin at core >=250 MHz, RV about 166.7 MHz. Use 16 concurrent IP synthesis
jobs, eight implementation/build threads and the existing swap.

The user authorized a fallback on 2026-11-03: if this full-size attempt still fails, prioritize
a working build and halve VU processing width/throughput. Keep the four clusters and their
four independent 512-bit weight streams. This is conditional; the first attempt retains the
current 16 front/EXP and eight back lanes. EXP/BACK parameter reductions alone do not halve
all front arithmetic, so document the exact reduced configuration if that fallback is needed.

Source integration and verification are complete; guarded native implementation is running.
The paired gather retains eight buffered jobs per pair and one job/cycle under sustained
traffic, with one added startup cycle. Its old registered central picker can still incur a
transient bubble when the selected link drains as another arrives. VU banks contain four
back-equipped and four front-only lanes each; arithmetic and lane order are unchanged.
Five central DMA boundary slices per localized channel preserve registered crossings for
DMA, KVST and trace. Weight responses stay local. Astra's source review found no remaining
correctness blocker after generated-hierarchy naming fixes. Actual synthesis and placement
ownership must still pass the native gates. The new floorplan requires these boundary slices;
use archived hook snapshots when inspecting older checkpoints.

Pre-build evidence for this source integration: the software build and all 297 enabled
non-slow CTests pass (one disabled). Tables and generated tops match. Gather passes six
configurations, VU passes PIPE 0/2/3 oracle cases, and all 91 floorplan fixtures pass. Four-channel
`core_smoke` and `core_db` pass with the K4 pipeline depths (`RST_PIPE=3`, `SPM_RD_LAT=4`,
`MVM_MAC_LAT=9`, BC/RES/VU pipes 2), reduced K/SPM sizes and both locality knobs enabled.
They exercise all four weight channels plus DMA reads/writes on channels 0/3 and trace/stop
behavior. Two-channel M7 decode/prefill cases also pass with the new DMA boundary and banked
VU, including KVST and vector/attention operations. This is not a full-size model-on-card test.
Logs: `build/pair_core_k4_final_depths.log`, `build/core_ddr_local.log`,
`build/pair_att_deterministic.log`, `build/k4_locality_ctest.log`.
Full RTL lint passes, including nine XSim elaborations and eight Vivado synthesis checks
(`build/k4_locality_lint.log`). The complete default bench run also passes:
445 passed, 115 opt-in skips, 7,611.83 s (`build/k4_locality_tb_all.log`, exit 0).
Its elapsed time includes waiting for the native flow's slots at the duplicate technology lint.
The final-depth K4 smoke/DB case was run separately after the full suite collected its earlier
variant. The corrected 107 floorplan fixtures were likewise run after the native guard fix.
The remaining VU simulation suite also passed separately in
`build/k4_locality_vu_remaining.log`: 11 passed, 37 opt-in skips, 433.31 s.

The new full-native supervisor is
`build/analysis/k4_locality_build_20261103/supervise.py`. Sol implemented it, Astra reviewed it,
and 20 isolated failure/completion checks pass. It uses a fresh directory, snapshots runtime
sources and package hashes, preserves the old XO, forces repackaging and runs through image
generation. It lets a native failure finish writing diagnostics. Exit 0 plus a nonempty xclbin
is recorded as `native_completed_pending_signoff`; root still must review routed timing,
clock/reset, CDC and locality before declaring T3 or the goal complete.

The previous routing failure is preserved below. Its error DCP, clean native session log,
status and resume audit have verified copies and SHA-256 records under
`attempts/20261103_142123_route_from_preserved_physopt/preserved_failure/` in that run.

### Previous attempt: routing congestion

Latest status, checked at 17:25 KST on 2026-11-03: the route-only attempt below exited at
16:56:55 with code 1. No Vivado process remains and no new attempt has started. This is an
actual native routing failure: the clean current-session `native_step.log` reports 286,682
signals failing to route because of congestion (`Route 35-162`), 352,035 overlapping nodes
(`Route 35-2`) and partially conflicted nets (`Constraints 18-1000`). The router reached its
final verification phases but did not produce a legal route. Its command took about 2 h 20 min;
reported peak memory was 30,237.754 MB. The clock/reset and CDC gates had passed before routing.
There is no timing-qualified 250 MHz image. The failure checkpoint
`tmp/link/vivado/vpl/prj/prj.runs/impl_1/level0_wrapper_routed_error.dcp` exists in the current
run (736,841,992 bytes); it is an analysis input, not a successful routed checkpoint.
Root `status.json` records `state=failed`, `failure_stage=route_validation` and unchanged
source integrity. Preserve this attempt and diagnose the failed nets before another run.
The DDR/gather and VU repartition proposals in `platform-u200.md` §8.6 are now in source
integration as described above; the old failure does not measure those changes.

The user resumed work after the intentional route stop. Use the earlier four-cluster
276.3 MHz image as the baseline and reduce the added interconnect before another long route.
Two changes are implemented and pass the functional checks below:

- Broadcast shares only its central first stage (`u_bc_head`) for BC_PIPE >= 2. Every
  cluster retains its own kept local tail. Total latency, reset flush and one line per
  cycle are unchanged. BC_PIPE 0/1 retain the original path. The floorplan distinguishes
  this topology from archived checkpoints with two per-cluster stages.
- VU front/back exits share one 512-bit return after selection beside the lanes. The
  injector already orders these exits. A resettable S0+LB token delay selects the result;
  the separate 512-bit reduction drain is unchanged. BACK8 saves 256 data connections
  and adds one input token before synthesis; actual SLL savings are pending.

The four independent weight streams retain 512 bits per cycle each. The new K4 config
requests core 250 MHz / RV 166.666667 MHz. The early clock gate reads the kernel's config
and rejects a stale 300 MHz model. CDC datapath/skew bounds are capped at 3.333 ns, so the
lower core request cannot loosen the previous crossing budget. K1 retains its 300 MHz request.
Focused MVM/ATT oracle and cycle checks, VU oracle tests and 132 clock/floorplan fixtures pass.
The software build and CTest pass (297 executed, one disabled). Full RTL lint passes,
including nine XSim elaborations and eight Vivado synthesis checks. The full default RTL
run finished with 386 passes, 115 opt-in skips and four failures in the new broadcast reset
test. It initially injected reset too late in four
BC_PIPE=1/RST_LOCAL=0 configurations. The Python stimulus now observes the producer and
asserts the intended occupied stage at each reset. The complete MVM test module passes
all 14 configurations after that fix, with the original assertions and golden/cycle digests.
The four failed nodes also pass in the default build directory (`--lf`, exit 0).
All 390 executed nodes are therefore covered by passing results across the full run and
reruns. Seven new isolated build-script tests pass separately; they were added after the
full run collected its cases. The original failures remain in the first full-run log.

The default RTL suite includes block tests and integrated RTL-Ibex firmware tests.
`core_m7` covers a two-channel tiny decode/prefill layer, including the VU lane-pipe variant,
against the C simulator. It is not a full-size K4 simulation. Added checks cover the
four-channel broadcast/reset configuration, attention traffic and the VU shared-return
latency/reset behavior. Long opt-in mutation/sweep campaigns and card tests are separate.
A small actual-MVM synthesis comparison replaces only the four clusters and gather with
black boxes. Its 530-bit broadcast banks fall from eight to five (4,240 to 2,650 FF); every
source/head/tail bit and the four independent tail outputs check correctly. These black boxes
retain all 530 bits, unlike the 529 surviving bits in the integrated baseline below. This
proves the shared synthesized topology, not actual K4 placement or SLL savings. The native
U200 calculator also passes at 250 MHz (input 100, M=25, D=2, O=5, effective period 4 ns).
Both small CAD jobs exited with code 0 and returned their slots. No new
placement, routing or 250 MHz timing result is claimed. Logs are under
`build/analysis/k4_interconnect_gates_20261102/` and
`build/analysis/k4_bc_shared_20261102/`. Gather arbitration and the DDR0/3 response detour
remain candidates; they are not part of these two RTL changes.

The initial bounded native build started at 16:49:56 KST on 2026-11-02, from `acb76ca`
(the RTL changes are in `b145c5b`). Its directory is
`build/link/k4_20261102_164956_shared_core250_rv166/`; supervisor PID 1326951 recorded
state and process/memory samples there. `build/analysis/k4_interconnect_latest.txt`
points to it. Forced XO repackaging completed, and all 138 packaged source hashes match
the starting snapshot. The rendered config requests 250/166.666667 MHz with synthesis
jobs 1, implementation jobs 8 and a general thread limit of 8. Native Vitis lowers the
synthesis-run thread limit to 1; its generated kernel synthesis script confirms that setting.
The two host slots are reserved for
this native flow; internal workers are allowed under the clarified limit below.
The flow stops at `vpl.impl.phys_opt_design`. After native completion it preserves the
checkpoint/XO/manifests and runs the read-only BC/VU and utilisation/timing/congestion
census. It does not route automatically. Package identity is recorded in
`package_entry_milestone.json`. The BD clock/reset gate passes: core and RV ports use
`aclk_kernel_00` and `aclk_kernel_01`, with separate reset generators. The two platform BD
critical warnings match the prior run (excluded debug address space and narrow-burst
metadata on the shell C2H interface). Synthesis started at 16:52:06 KST with `-jobs 1`.
The synthesized clock/reset gate, placement and timing results are still pending.

The user requested concurrent synthesis jobs equal to the available logical CPUs (16;
the host has eight physical cores) at 17:45 KST. This supersedes the earlier one-job
setting. The serial native process group was intentionally stopped at 17:48:18 KST.
Its supervisor exited, all CAD processes exited, and the source snapshot was unchanged.
Completed synthesis includes the K4 kernel, which finished at 17:29:33 KST. All 44 completed
DCPs (158,299,286 bytes) and 591 files (170,139,375 bytes) are preserved with verified hashes
under `attempts/20261102_174818_serial_synth_stop/`. Only the 133 incomplete queued/running
runs were reset. Reopening the project confirms their `Not started` status and all 44
completed runs' unchanged status and checkpoint hashes.
`TSR_U200_SYNTH_JOBS=16` now sets IP concurrency independently of `TSR_BUILD_JOBS=8`.
All 14 build-script/platform checks pass, including nine isolated fixtures for 16/8/8
rendering and invalid-value rejection before tools start. No arithmetic, RTL, clock,
floorplan or CDC change is made for this scheduling correction.

A native `v++ --from_step vpl.synth` attempt at 18:02 KST was stopped during project loading.
The generated restart Tcl opens the existing project but then resets every synthesis run,
including completed runs. It had not reached that reset: the project, all 44 completed DCPs
and their archived copies remain unchanged.

The existing Vivado project resumed at 18:08:35 KST under supervisor PID 1357676.
Its attempt is `attempts/20261102_180835_preserved_parallel16/` within the same build directory.
Guarded `launch_runs my_rm_synth_1 -jobs 16` retains all generated platform hooks and the
configured clock/reset, CDC and floorplan checks. At 18:09 KST, 16 synthesis workers were
running concurrently, plus the supervising Vivado process. All 44 retained checkpoint
hashes still match. The runtime source snapshot records `d0d12ee` plus the scheduling patch.
The detached supervisor is `build/analysis/k4_interconnect_preserved.py`; root `status.json`
is authoritative. After synthesis it runs `impl_1` through `phys_opt_design`, verifies the
checkpoint and preserved inputs, and runs the read-only census after releasing the native
slots. Routing remains excluded from this attempt. The native Vitis restart attempt and
the direct Vivado project resume have separate logs; neither is a completed placement result.

At 21:06 KST on 2026-11-02, the resumed attempt is in detailed placement. Synthesis finished
at 18:21:06: all 167 checked runs are complete (98 synthesized, 69 cached), with no run-log
errors and all 44 retained checkpoints checked. Implementation started at 18:21:08.
The synthesized clock/reset gate passes at core 4.000 ns and RV 6.023 ns; the pre-implementation
CDC gate passes with unsafe 0, unconstrained 0 and the 3.333 ns bound. Global placement has
finished. The latest implementation log, updated at 21:01:52, records Phase 3.2 Commit Most
Macros & LUTRAMs. No `ERROR:` is present in that log. A five-second sample at 21:06 shows
the implementation worker using about 4.1 CPU cores with active swap I/O. Host swap use is
about 34 GiB of 72 GiB; the reported Vivado peak is 32,969 MB. Placement completion,
post-placement physical optimisation, the census and final timing remain pending.

The 2026-11-02 attempt stopped at 22:33:05. Placement completed and saved
`level0_wrapper_placed.dcp` at 22:03:46. Physical optimisation then completed its computation
with global WNS -0.809 ns and TNS -155.747 ns. Its POST locality hook rejected the new
`att_stall_q_i_3_comp_1` LUT3 name, so the native step failed before writing a physopt DCP.
This was a checker error; no route ran. The pre-physopt clock/CDC checks passed.

The new placed report has 8,447 SLR0-1 and 6,880 SLR1-2 logical connections: 15,327 total,
down from 17,205 in the earlier 300 MHz placed report. SLR1 CLB site occupancy remains
99.77% (previously 99.84%); SLR0/2 are 97.08%/91.82%. Kernel LUT/FF counts are
532,817/632,943, down by 1,927/2,108. Shared BC/VU placement and locality pass. These
results include the clock-request change and full replacement placement; they do not
isolate each interconnect change's contribution or measure routed physical SLL demand.
The placer still reports high congestion. Placed global WNS -0.860 ns is on platform AXI
control; core WNS -0.523 ns is on cluster 3 assembler reset distribution. RV setup/hold
slack is +0.032/-0.080 ns. These are pre-route estimates, not 250 MHz timing signoff.

On 2026-11-03 the floorplan guard was extended only to accept a numeric suffix after
`_comp`. Actual cone connectivity, exclusive fanout, pblock ownership and physical SLR
checks remain enforced. The exact failed name reproduces the old rejection; 60 floorplan
fixtures and all 149 related platform fixtures pass after the fix. No arithmetic RTL changed.
The recovery driver under `build/analysis/k4_resume_physopt_20261103/` archives all 167
completed synthesis DCPs, the placed/opt DCPs, native hooks, project and reports before a
physopt-only reset. It preserves native hooks and verifies checkpoint and XO/source hashes.
The first recovery at 11:42:32 was intentionally stopped at 11:45 while loading the placed
checkpoint: the resume verifier scanned old steps in Vivado's appended `runme.log`.
Its attempt `20261103_114117_physopt_name_guard_resume` retains 217 archived files
(1,298,694,645 bytes), package copies and the stop record. The supervisor returned failure
with unchanged execution sources. The log verifier now preserves the old binary prefix and
checks only the appended suffix; 10 plain Tcl regressions pass, including rejection of new
out-of-scope steps and prefix changes. On completion, `native_step.log` contains only the new
child session. The full native/runme logs include old sessions and old error messages.

The corrected recovery started at 11:52:14 KST on 2026-11-03 under supervisor PID 1890583,
attempt `20261103_115137_physopt_name_guard_resume`. Its archive holds 218 files
(1,296,106,834 bytes), plus verified package copies. All 167 completed synthesis DCPs and
the placed checkpoint are checked before and after the physopt-only reset. It runs physical
optimisation and its native clock/CDC/locality hooks, then preserves the successful DCP and
runs the read-only BC/VU/utilisation/timing/congestion census. Routing is excluded. Root
`status.json` remains the live execution record; no successful recovered physopt result is
claimed until the exact completion marker and checkpoint checks pass.

That recovery saved `level0_wrapper_physopt.dcp` at 12:27:31 on 2026-11-03. The native
clock/reset, CDC and post-physopt locality checks passed. Its wrapper then failed at
12:28:14: the bounded native run reported `Not started route_design / 66.67%`, while
the wrapper required `phys_opt_design Complete! / 100%`. This was a whole-run status
check error. The native child completed successfully and wrote its checkpoint; no route ran.

At 14:02 KST, attempt `20261103_140219_completed_physopt_recovery` resumed the pending
read-only census under supervisor PID 1929627. It reuses the 594,304,775-byte physopt DCP,
SHA256 `b506339df428e9d6d05f599ff79fdd8201327a82051a40f2e2e2f0cfb443f061`.
The recovery validates the exact appended child log, fresh typed checkpoint and completion
markers, absence of error/route markers, clock/reset/CDC/locality passes, all 167 preserved
synthesis checkpoints, the placed checkpoint, package identity and execution sources.
The old failed attempt and drivers remain unchanged. A corrected driver and 29 passing
recovery regressions are under `build/analysis/k4_completion_recovery_20261103/`.
Synthesis, placement and physopt were not repeated. Census completed at 14:21:03 with
exit code 0 and checkpoint, baseline, source and driver hashes unchanged. Shared BC has
five banks of 529 live bits (2,645 FF), 529 unique crossing nets and 1,058 logical boundary
incidences. The VU selected stage-0 scope has 3,111 directed boundary incidences; its shared
lane return and separate reduction return are each 512 bits. The 79 selector registers
remain local. Four stage-1 drain-address replicas stay in SLR0, each driven alongside its
original by the corresponding SLR1 stage-0 source. No unknown sequential cell was found.
Full-shell logical connectivity is 8,446 on SLR0-1 and 6,879 on SLR1-2. SLR1 CLB site
occupancy remains 99.77%. These counts are not routed physical SLL allocation.
Core setup WNS is -0.148 ns; RV setup is +0.032 ns. Global setup WNS -0.809 ns is on the
fixed platform DMA clock. Core/RV hold slack is -0.313/-0.080 ns. The worst core setup
sample is cluster 0 QK gate to central ATT stall aggregation. Congestion level 6 remains.
Both core/RV crossing directions retain the 3.333 ns datapath-only bounds. Timing is not
qualified; routed setup, hold and congestion still need verification.

Attempt `20261103_142123_route_from_preserved_physopt` starts the native route-only
continuation under supervisor PID 1935097. Its scripts and reviewed report hashes are
under `build/analysis/k4_route_from_physopt_20261103/`; 18 focused Tcl regressions pass.
It preserves the 167 synthesis DCPs, placed and physopt inputs, and all native platform
hooks. Only the route step is reset/launched, with eight implementation threads and the
existing swap. Completion requires the new typed routed DCP, fresh step markers and a
clean current child log. Whole-run progress alone is not a completion criterion. This
attempt stops before bitstream generation; routed timing must be reviewed afterward.
At 14:33 KST, the new child session has passed the pre-route clock/reset and CDC gates
(unsafe 0, unconstrained 0, bound 3.333 ns) and entered `Phase 1 Build RT Design`.
The child opened the preserved physopt checkpoint; no synthesis, placement or physopt
was repeated. There is no routed checkpoint or routed timing result yet.

The user clarified that monitoring must accompany active build work. Continue with routing
after reviewing the census, and repair/restart failures without another permission request.
Core acceptance remains at least 250 MHz; do not chase 300 MHz. The Sol monitor only reports.
It checks new state/failure changes within ten seconds and detailed progress every fifteen
minutes. Root owns diagnosis, fixes and continuation. Its records are under
`build/analysis/k4_build_watch_20261103/`; monitoring alone is not build progress.

The DDR0/3 localization review is in `platform-u200.md` §8.6.1. It uses the existing recovered
physopt census; no new CAD run or runtime-input change was made. Moving the central bridge,
arbiter and weight pipe ends needs roughly 1.7 K LUT / 4.0 K FF in SLR0 and 1.4 K LUT /
3.9 K FF in SLR2. Whole-SLR CLB occupancy is 97.08% / 91.82%, so SLR2 has more margin but
neither pblock fit is proven. Keep full-width weight streams and explicitly check the DMA
handshake timing problem that originally motivated central arbiters (`rtl-m3.md` §14.4).

The combined DDR/gather sufficiency review is in `platform-u200.md` §8.6.2. The nominal
2,148-connection saving is not a physical SLL or routing-success forecast. Current congestion
also includes central issue/CSR/debug queues and an outer PV/MAC window. If further relief
is needed, compare a balanced VU eight/eight lane split with central metadata/control cleanup
before reducing arithmetic. Moving four back-equipped and four front-only lanes represents
about 23 K LUT / 37 K FF / 184 DSP before boundary helpers. BACK8 -> BACK4 is a later area
tradeoff with workload-dependent latency. Keep EXP16, T_MAX8, four weight512 streams and
independent MVM/VU execution first. Analysis artifacts are under
`build/analysis/k4_followup_capacity_20261103/`; no RTL, floorplan or CAD change was made.

### Preserved attempt and baseline measurements

The corrected attempt completed placement and physical optimisation, with actual SLR locality
checks passing. It was intentionally stopped at 05:25 KST on 2026-11-02: the new timing report
shows kernel FIFO crossings timed against a 0.040 ns clock-edge separation instead of the
required datapath-only maximum delay. Restoring those constraints and moving the five ATT
stall reduction LUTs pass real-DCP checks, with no new DRC violations. The corrected analysis
checkpoint is saved. A native restart from the preserved placement completed physical
optimisation and its locality check. The route-pre clock/reset and CDC gates passed;
`route_design` entered `Phase 1 Build RT Design` at 08:24 KST. No new routed result exists yet.
By 09:57 KST it passed SLL assignment and five initial net-routing passes and entered
`Phase 5.3 Global Iteration 2`. General routing congestion remains; this is not a timing pass.
That iteration and the Phase 5 footer were printed at 10:46 KST, with 2,572,256 overlapping
routing nodes, up from 2,311,139. At 12:51 KST the log was unchanged and the router had not
returned. Two workers each used nearly a full CPU throughout a seven-minute sample, with
little I/O and no measured memory pressure. This shows continued computation, not convergence.
The user requested a stop and investigation of the next attempt. SIGTERM was sent to the
implementation worker at 14:00:55 KST; the native supervisor finished at 14:01:01 with code 1.
The worker, parent Vivado and supervisor have all exited. This is an intentional stop, not a
new confirmed router error or tool hang. There is no routed checkpoint or 250 MHz timing result.
The 25 preserved files (1,242,547,736 bytes), including the placed and corrected physopt DCPs,
are under `attempts/20261102_140055_route_congestion_stop/`. Its `archive_manifest.json`
records hashes; both checkpoint hashes match their earlier identities. That run's `status.json`
retains the native failure code and records the user-requested stop classification. The
replacement build is recorded above. The investigation targets SLR interconnect reduction
and its performance cost before choosing another long routing attempt.
The prior four-cluster image ran at 276.3 MHz before RV/VU/attention integration. Preserve
that weight-stream throughput while investigating the added paths. Historical whole-core
synthesis records show 275,657 CLB LUTs before integration and 533,573 in the W4-B OOC run;
this is not an isolated measurement of each added feature's cost.

A read-only query of the preserved corrected physopt DCP gives the following same-checkpoint
counts. Kernel/core totals are 534,801/529,186 LUTs. These are physical-optimisation counts,
not a new synthesis, route or card result. Cross-hierarchy LUT combining limits exact sums.

| Scope | LUTs | Note |
|---|---:|---|
| RV subsystem | 6,206 | 4,234 FFs, 48 RAMB36; subset of control |
| VU | 83,398 | 433 DSP; includes lanes |
| VU lane array and pipes | 49,128 | 368 DSP; subset of VU, reported LUT difference 34,270 |
| Four ATT PV modules | 86,829 | excludes shared MAC-mode and other attention logic |

The scoped crossing census finds 529 surviving broadcast nets per cluster: 1,058 SLR1->0
and 1,058 SLR1->2. The candidate shared broadcast could remove about 1,058 duplicated
logical boundary-net incidences in total, rather than the raw-width estimate of 1,060.
VU input is 2,084 surviving nets (four 512-bit operand vectors plus 36 control nets); return
is 1,280 (512 drain + 512 front + 256 back), with one reset crossing. This totals 3,365
stage-to-stage VU boundary connections. These are logical-net incidences; they do not count
physical SLLs or prove a column reduction. No sampled source net was excluded from these
selected BC/VU scopes. Other weight/gather/AXI scopes were not enumerated.
The reports, clock metadata and input-hash record are under
`build/analysis/k4_current_interconnect_20261102/`. Core/RV clocks remain 3.333/6.023 ns.
Source-level options and conditional performance models are
`build/analysis/repository_map_20261102/slr_pipe_bandwidth_options.txt` and
`slr_width_performance_tradeoff.txt`. These measurements predate the two changes above.
The query session exited normally with code 0, returned both tool slots and left the input
hash unchanged. No CAD process from that route or read-only query remains.

The user clarified the process limit on 2026-11-02: at most two top-level executions,
internal Vivado workers allowed and the existing swap. The later request sets concurrent
IP synthesis jobs to 16. The native run reserves both host slots and keeps `TSR_BUILD_JOBS=8`.
`vivado_slot.sh` limits top-level invocations, while Vivado also launches internal synthesis
workers. A small production-DBG probe with
`general.maxThreads=8` and `synth.maxThreads=1` still launched a second actual Vivado process.
The valid `synth.enableParallelHelperSpawn=none` setting removed the early helper but still
launched a direct worker. Adding Vitis's parent would exceed a literal two-process limit.
Both probes were stopped immediately within their own process groups; they do not provide
completed synthesis results. No such tool parameters are applied to the production config.
An earlier launcher-classification error and an invalid `false` enum trial are preserved
separately and excluded from that conclusion. Logs and `/proc` samples are under
`build/analysis/k4_synth_process_probe_20261102/`. No probe process remains.
No further process-limit clarification is needed. The 250 MHz run stops after physical
optimisation, then preserves the checkpoint and runs
a read-only connectivity/utilisation/timing census. It never starts routing automatically.

The VU alternatives need more than a narrower wire declaration. Front/back result sharing
is a candidate for 256 fewer surviving return nets; the independent reduction drain must
remain separate unless buffering/admission is proved. Moving the boundary to the SPM client
retains two read lines and one write line per cycle, but also carries 512 SREG bits, a
256-bit command and control/address/byte-enable signals: roughly 2.5 K declared boundary
bits, not 1.5 K. It also moves logic into already busy outer SLRs. Moving only angle/coefficient
selection is insufficient: AMERGE's AM_L mode needs two lane-varying coefficient vectors.
Keeping a full-width fallback preserves the original wires. Front/back sharing is now
implemented as described above. The other options remain source-level proposals, without
equivalence proofs, timing passes or measured SLL reductions.

`hw/u200/build_xclbin.sh k4` (4 channels, T_MAX 8, the final image) synthesised and placed, then
`route_design` failed: "The router is unable to resolve localized SLL routing demand"
(Route 35-3339), 5.4 M unrouted pins.

- Utilisation after synthesis (kernel plus shell): CLB LUT 835,805 (70.7 %), FF 43 %, BRAM tiles
  66.8 %, URAM 15.4 %, DSP 49.4 %. Kernel alone: 544,920 LUT, 3,360 DSP, 1,013 BRAM tiles, 128 URAM.
- SLL demand per boundary (17,280 SLLs each, 12 columns of 1,440):

| Boundary | Demand | Columns over 100 % |
|---|---|---|
| SLR1-SLR2 | 17,479 | 7 of 12 (up to 180 %) |
| SLR0-SLR1 | 14,261 | 6 of 12 (up to 158 %) |

The corrected router estimates 8,872 SLLs at SLR1-SLR2 (51.34 % of total capacity) and
10,052 at SLR0-SLR1 (58.17 %). One column per boundary still exceeds capacity: 121 % and
175 %, respectively. `Route 35-3311` warns of localized demand, but assignment completes;
the previous `Route 35-3339` failure was at `Phase 3.1.2`. Passing that stage does not prove
all SLL or general wire congestion is resolved. Initial routing reports 1,317,850 unrouted
nets and global/short congestion level 7 (128x128); it continues with rip-up and reroute.
The first global iteration's WNS is -8.246 ns with hold unavailable, an intermediate
incomplete-route result. Keep the original net/pin units separate when comparing failures.

The congestion windows overlap c2 PV/MAC and VU lane 0 in the lower region, c1/c3 ATT and
assembler 3 in the upper region, and central gather/QACT/SPM/issue plus platform DDR logic.
These are hierarchy correlations, not identified drivers of the remaining failed nets.
The README hero is a logical diagram: c0/c2 and all VU lanes are in SLR0, c1/c3 and their
assemblers in SLR2, and the VU reader/writer/reduction and central gather in SLR1. The
shell's fixed-DMA timing path must not be attributed to the hero's LD/ST DMA block.
The local evidence is `build/analysis/repository_map_20261102/`:
`current_router_congestion_triage.txt`, `readme_bottleneck_mapping.txt` and
`route_silence_assessment.txt`. AMD's [Router Messaging, UG904 2024.2](https://docs.amd.com/r/2024.2-English/ug904-vivado-implementation/Router-Messaging)
describes iterations lasting hours under difficult congestion/hold fixing. That profile
matches the symptoms; it does not establish progress or exclude a tool defect. The later
user-requested stop is recorded above. Any eventual result still needs fully-routed/no-route-error checks.

- For comparison, the core before the control core (floorplan C') used 7,010 SLLs in total
  (`docs/rtl-m3.md` §14). The failed checkpoint confirms extra ATT job round trips, misplaced
  broadcast decode and VU consumer LUTs. Shell/interconnect nets also contribute substantially
  to the upper boundary; attention alone does not explain the column pressure.
- Artefacts: `build/link/k4/build.log`, `build/link/k4/v++_link.log`,
  `build/link/k4/tmp/link/vivado/vpl/prj/prj.runs/impl_1/runme.log` (SLL table near "Estimated SLL
  Demand Per Column"), placed and failed-route checkpoint
  `.../impl_1/level0_wrapper_routed_error.dcp`. Opening it needs about 30 GB of memory.

### Source corrections and overnight retry

The existing failed checkpoint was opened for queries and reports only. Both kernel clock
ports use clock ID 0 at 300 MHz; all 4,227 RV FF clock pins use it too. The requested RV clock
was 166.7 MHz. The generated `ulp.v` confirms the alias. The deprecated frequency option was
present, but its being the sole cause is not established: frequency requests reached the BD
script while the secondary reference-clock selection was absent.

The corrections after `58bf76d` are on branch `codex/k4-route-fixes`:

- LCQ payload/classification reads propagate I/O errors. A complete batch is acknowledged
  before host answers/events/credits change. The batch survives ambiguous ack failure;
  uncertain reset blocks consumption until counters are resynchronised.
- RV watchpoint FIFO depth is 16, covering CDC credit-return latency for continuous access
  at the K4 clock ratio. Dense 4,096-access tests preserve both counters and a single HALT
  event at several phases, with CDC jitter too. A sustained RV source faster than the core
  still needs an aggregation/interface change for general losslessness; the simulation
  assertion reports overflow. No wider clock-ratio guarantee is claimed by this fix.
- Hart reset release replaces old pending halt state with HOB or a simultaneous new request.
  The RTL regression follows the existing simulator stale-request test.
- Whole-core and packaged-IP CDC failures now terminate the flow. Link configs use
  `kernel_frequency=0:300|1:166.666667` for U200 scalable clocks. BD and synthesized clock/reset checks run before
  implementation; the final pre-bitstream check reads Vitis's scaled `_new_clk_freq`.
- K4 attention job pipes and selectors stay with their clusters. Residual broadcast decode
  goes to its destination cluster and VU writer/reduction helpers to SLR1. Crossing stage
  counts and reset connections stay the same. Locality is rechecked after placement and
  physical optimisation; missing scopes, unowned replicas and wrong SLRs fail the run.

Validation (2026-11-02): `cmake --build build/dev -j8`, CTest (297 executed, one disabled),
both generated-file checks, and the RTL lint checks pass. The full RTL pytest run passed
290 tests and skipped 115 opt-in tests. Its technology-lint node was stopped to correct a
missing wrapper environment and passed separately. The latest clock/floorplan fixtures
pass 48 tests. The first standalone lint invocation also used a relative wrapper path;
its nine Xilinx-tool checks passed on the absolute-path rerun. `AGENTS.md` now gives the
absolute wrapper and explicit Python for both commands. Logs are in
`build/analysis/implementation_fix/`.

The user authorised an overnight placement/route attempt on 2026-11-02. The first retry
(`build/link/k4_20261102_003426_core300_rv166`) stopped at CFGEN: `[clock] id=0/1` is invalid
for this platform's scalable clocks; only fixed reference IDs 2/3/4 are accepted. The earlier
static review and fixtures missed that distinction. The configs and their regression now
use `kernel_frequency` for both scalable IDs.

The replacement run is `build/link/k4_20261102_003908_core300_rv166`, started at 00:39 KST.
It repackaged the corrected XO from 137 sources. The BD clock/reset check passed on the
generated design: core uses `ss_ucs/aclk_kernel_00`, RV uses `ss_ucs/aclk_kernel_01`, and each
uses its corresponding reset generator. Vitis truncates the RV request to 166 MHz in
`ipirun.tcl` and the generated `ulp_ooc_copy.xdc` (6.024096 ns). The UCS IP's default 500 MHz
parameter is not the implementation request; scalable clocks use the generated constraints
and later runtime programming. All 176 block-level synthesis jobs completed and the corrected
kernel OOC checkpoint was written. At 01:25 KST, top-level synthesis completed and implementation
started. At 01:39 KST, the pre-opt clock gate stopped the run: RV period was 9.999 ns, outside
5.99–6.32 ns. Vitis's native clock-calculator helper passed `Project` where the installed Vivado
API expects a part name, reported `Invalid part string Project`, then emitted a 1:1 generated
clock. Clock/reset BD wiring had passed; this failure is in implementation clock constraints.
No corrected placed/routed result or xclbin exists yet. The old checkpoint remains evidence of
the baseline routing failure.

The repository `clock_vitis_compat.tcl` hook adapts the calculator API for Vivado 2024.2 and the
pinned U200 part. It supplies PART and the input period, checks the solve result/effective ratio,
and cleans up on errors. The native generated-XDC flow is preserved. The focused fixtures now
pass 66 tests, and the real calculator/shim probes pass at 166 MHz. These are local workaround
checks, not a routed result or an AMD-published fix. The failed logs/XDC/status are preserved in
`attempts/20261102_015229_pre_compat/` under the run directory. Configuration-only resume
completed at 01:53 KST with exit 0; `.user_init_pre.dat` and `_full_init_pre.tcl` both contain
the compatibility hook. The implementation resume started at 01:56 KST from `abc4afe`,
supervisor PID 804838. It reused the 176 block-level synthesis results and regenerated only
the top-level synthesis. The actual pre-opt `CLKNET` check passed: core 3.333 ns, RV 6.023 ns,
all 4,225 RV FFs on the RV clock, and separate reset sources in their respective domains.
The calculator selected M=83, D=10, O=5 from 100 MHz, giving 166 MHz. At 02:29 KST,
`opt_design` and its checkpoint write completed. The revised floorplan passed its initial
membership check (`K4FP locality PASS (placed=0)`); placement started after two pre-place
DRC passes with zero errors. By 04:45 KST, `place_design` completed with zero errors and the
actual SLR check passed (`K4FP locality PASS (placed=1)`). The placed reports show:

| `report_utilization -slr`, placed | Previous failed build | Corrected build |
|---|---:|---:|
| SLR0-SLR1 SLLs used | 13,325 | 9,830 |
| SLR1-SLR2 SLLs used | 15,181 | 7,375 |
| Total SLLs used | 28,506 | 17,205 |
| SLR1 occupied CLB sites | 99.58 % | 99.84 % |
| SLR1 LUT utilisation | 76.46 % | 77.09 % |

These are matching placed-report counts, not the router's per-column estimated SLL demand
above. The combined changes reduce placed crossings, but central packing pressure remains:
the placer reports high congestion. Physical optimisation also completed with zero errors
and another `placed=1` locality PASS. Its timing report has global WNS -2.960 ns. The worst
core path is the ATT stall aggregation, routed in the placement estimate through
SLR0 -> SLR2 -> SLR1; 97.1 % of its data delay is wiring. RV setup is -0.367 ns, fixed DMA
clock setup is -0.821 ns, and core/RV/platform hold violations remain in this pre-route
estimate. A lower core clock alone is not yet evidence of timing closure.
The reports are `slr_util_placed.rpt` and `kernel_util_placed.rpt` in the active
implementation run directory. The current attempt, PID and source commit are in `status.json`.
After locality passes, the hook now writes `k4_<step>_timing_summary.rpt` and
`k4_<step>_congestion.rpt` in that directory. These only query the open design. The first
place hook had already finished when diagnostics were added. The first reports were produced
after `phys_opt_design`, and the placed and physopt DCPs are complete. Its runtime hook
snapshot/hash is preserved beside `placement_milestone.json` under the attempt directory.

The reports expose a confirmed constraint gap: sampled FIFO paths in both core/RV directions
have a 0.040 ns natural-edge requirement and no max-delay exception; clock interaction calls
both directions `Clean Timed`, not `Max Delay Datapath Only`. The clocks themselves remain
3.333/6.023 ns. A later real-DCP experiment restored the exceptions, then reread the native
RV clock XDC: the bounds and full CDC gate still passed. Clock redefinition did not reproduce
the loss. The stage that loses the packaged constraints during integration remains unproven.
At 05:24:56 KST, the implementation worker
was stopped after its physopt DCP was complete. The supervisor exited at 05:25:10 with code 1.
Logs, reports, generated scripts and all three implementation checkpoints are archived under
`attempts/20261102_052510_cdc_unbound_stop/`, with hashes. The intended correction restores
the original CDC XDC on the actual clocks and checks its effect before routing. A separate
cycle-identical floorplan correction keeps the exclusive ATT stall aggregation beside its
central consumer. Neither changes RTL behaviour or pipeline depth.

The archived physopt DCP now passes the restored constraint gate: both directions have a
3.333 ns datapath-only requirement and zero unsafe/unconstrained CDC paths. A kernel Gray-bus
skew constraint is present, with a reported bound of 3.333 ns and skew of 0.073 ns. The
`cdc_restore.tcl` hook generates a scoped run-local copy of the original CDC XDC and validates
the linked clocks, timing requirement, CDC structure and kernel skew coverage. `impl_pre.tcl`
uses it before optimisation; `cdc_pre_route.tcl` rechecks before routing and fails on loss.
No false paths, clock groups or new CDC waivers are added.

The remaining floorplan correction selects all exclusive ATT stall reduction LUTs from the
actual endpoint cone. It keeps their source FFs in the original cluster SLRs. A saved-placement
hook moves only the reduction LUTs, preserves their REF/INIT and every other primitive's
LOC/BEL, and checks locality. At 07:02 KST the real-DCP ECO passed: five LUTs moved, all REF/INIT
values matched, and every other primitive's LOC/BEL was unchanged (2,131,692 cells). Actual
SLR locality and the following clock/reset and fresh CDC checks passed. Post-ECO DRC has zero
errors; all 4,368 violation detail records match the baseline, including the three existing
unrouted-state critical warnings. ATT stall slack improves from -2.960 to -1.048 ns. Global
WNS is -2.557 ns, WHS is -0.324 ns, and RV setup is still -0.367 ns. These are pre-route
estimates, not a 250 MHz pass. The remaining worst core path is c0's local ATT QK gate
(10 levels, 84.1 % wire delay); the next two are DB queue/control paths within SLR1.
The corrected DCP, the CDC-only DCP and the immutable source have recorded hashes in
`build/analysis/cdc_restore_20261102/checkpoint_hashes.json`. Vivado exited with code 0.

The native restart began at 07:34:26 KST under supervisor PID 1151282, using
`resume_impl.tcl` on the existing project from `phys_opt_design` through `route_design`.
It preserves the native route-post hook and original placement and does not repeat synthesis.
The GateLvl fileset has no TOP; the driver validates the native `gen_run.xml` checkpoint
records, run/part and matching filenames instead. The actual project API, hook settings,
step reset and launch passed. Child PID 1151638 opened `level0_wrapper_placed.dcp`.
The attempt is `attempts/20261102_073426_cdc_eco_native_resume/`; its source snapshot includes
the validated working-tree hooks. `status.json` points to its `native_resume.log`.
The 25 implementation source snapshots match pushed commit `c2055f6`; the mapping is in
`committed_hook_source.json` in that attempt. On the native placed input, the five-LUT ECO
preserved all 2,131,648 other primitive placements. Physical optimisation and its P1 check
then passed. Setup estimates are core -1.606 ns, RV -0.230 ns and fixed DMA -0.821 ns;
hold remains -0.324/-0.127/-0.201 ns. The remaining core worst path is the local `qk_gate`
feedback (8 levels, 87 % estimated wire delay). These are still pre-route results.
The new physopt DCP was written at 08:19 KST. Route-pre checks passed with core/RV periods
3.333/6.023 ns, all 4,234 hart FFs on RV and both CDC directions bounded at 3.333 ns.
The router started at 08:24 KST. Its entry evidence and checkpoint hash are in
`route_entry_milestone.json`; the completed physopt DCP is 595,717,524 bytes.
The fresh commit gates pass the software build, CTest (297 executed), both generated checks,
full RTL lint (including nine XSim and eight Vivado checks), and the full RTL pytest run
(355 passed, 115 opt-in skips, exit 0). Focused link checks pass 73 tests; floorplan checks pass 32, including
two repeated-application regressions added after the full pytest run collected its cases.
Actual BEL probes cover occupied, free and physical-only resources. Actual floorplan source
applied twice preserves all four pblock ownership sets. No 250 MHz timing result is claimed.
Fresh gate logs and their hashes are in `build/analysis/implementation_fix/cdc_floorplan_gates/`.

The original detached supervisor was PID 741389 and exited with status 1. `status.json`, `build.log`, `memory.jsonl`, source
snapshot/hash manifest and the eventual `exit_code` are in that run directory. The pointer
`build/analysis/overnight_route_latest.txt` identifies it. The run uses eight jobs, the Vivado
slot wrapper and the existing 72 GiB swap. It stops at `vpl.impl.route_design`, preserving
the clock/reset, floorplan/locality and post-route CDC gates. Bitstream generation and the
final frequency gate are deferred. The user revised K4 acceptance on 2026-11-02 to a core
clock of at least 250 MHz. The stopped attempt retained its 300 MHz request. Do not repeat
implementation just to reach 300 MHz once 250 MHz or more is valid. A route
with negative setup slack at 300 MHz is useful evidence; it is successful timing only after
checking an accepted frequency of at least 250 MHz, with RV, hold and CDC checks intact.
The old 276.3 MHz floor is superseded for K4; historical card measurements are unchanged.
The final-frequency hook accepts 250.0 MHz and rejects 249.99 MHz. At that point the early netlist
check rejected a 4 ns core period for the then-active 300 MHz request; the new 250 MHz request
and matching gate above supersede it. The clock/floorplan
fixtures pass 71 tests after this change; the build and both generated-file checks pass too.

Known current debug-interface gaps remain: RV_SAMPLE/RVWP_APPLY writes acknowledge before
their CDC work completes, and DBG_STATUS.SBA_REFUSED has no RTL setter. These are separate
from the deferred profiler/report work and were outside this correction.

GDB on the C simulator is implemented and verified. The full CTest run's `tools_dbg_gdb`
used the pinned `riscv-none-elf-gdb` through `tsr-dbg gdb --stdio --device sim` and passed
boot/symbol stops, step, memory/CSR reads, a launched kernel hardware breakpoint, backtrace,
`finish`, tile history and detach. The RSP, embedded-service and dropped-connection tests
also passed. This is separate from the RTL module tests; integrated RTL/card GDB remains
unverified. `prof-dbg.md` §1 and the user-facing environment-variable docs now reflect this
status instead of the old P3.7-planned wording.

The broader repository map is [design/README.md](design/README.md) §12: code/contract boundaries,
normal decode and failure/recovery paths, simulator/RTL/card evidence and W0–W6/deferred work.
It reuses the earlier audit and covers all 759 tracked entries by role. Review depth is explicit:
63 body/call-path reviews, 144 specific prior reviews reused, 468 interface inspections,
18 generated files, 55 assets, four vendor boundaries and seven test fixtures without body review.
This is not a claim of exhaustive line/input coverage. The detailed local reports and file-level
records are in `build/analysis/repository_map_20261102/`, referenced from that implementation map.
No runtime/RTL changes or extra build/test/Vivado/DCP/link runs were made for this mapping.

### Next steps

1. Check the new build's `status.json` and logs under the current run directory above.
   Its zero exit status would mean bounded physical optimisation and census, not routing.
   Keep the stopped run and archive hashes as the comparison reference. A future routed
   checkpoint still needs full-route/no-route-error validation before calling routing successful.
2. Functional gates for the shared broadcast head and VU return pass as recorded above.
   Apply the clarified top-level execution limit above. Check the new K4 netlist, then placement/physical optimisation at
   core 250 MHz before deciding whether to route. Actual SLL savings, local fanout and new
   ownership on the full placed design remain unverified. Keep all four 512-bit weight streams.
   The shared head could remove about 1,058 duplicated logical boundary-net incidences from
   the measured BC scope; VU sharing removes 256 data connections and adds one token before
   synthesis. Subsequent candidates are per-SLR gather arbitration and the DDR0/3 central
   weight-response detour (`docs/platform-u200.md` §8.6).
   Reducing register depth alone does not reduce a bus's crossing width. Narrowing a weight
   link from 512 to 256 bits at 250 MHz lowers its ideal payload rate from 16 to 8 GB/s;
   an active slow channel can limit the whole equal-row MVM. The earlier combined floorplan changes reduced
   placed SLL counts by 11,301, but did not establish routing success or isolate their causes.
3. An implementation-only resume of the old failed run cannot repair its clock wiring. The corrected
   run rebuilt from the corrected RTL and clock selection. Its preserved placed/physopt DCPs
   have the correct clocks. Native `vpl.impl.phys_opt_design` restart is supported, but needs
   the new PRE hook to restore constraints, K4FP Tcl state and legal LUT placement. A
   changed RTL topology needs a new kernel XO/netlist; the old synthesis cannot be reused
   as if it contained that change. The existing DCPs remain analysis references. A
   `config_hw_runs` reset can invalidate those saved run results; back them up first. A full link takes 7-10 h on
   this host because of swap. A machine with
   64 GB RAM or more (Linux, or WSL2 with Linux Vivado/Vitis 2024.2: the Alveo `v++` flow is Linux
   only) runs it much faster and allows two variants at once.
4. Gate T3: K4 kernel clock at least 250 MHz, per the user's 2026-11-02 decision. A route-only
   stop precedes Vitis's final frequency selection. Check routed timing at the accepted core
   frequency and retain the RV, hold and CDC gates. Do not retry merely for 300 MHz if this
   passes. Known tight paths from the out-of-context placement (WNS -0.614 ns
   placed): DB logic 9-16 levels, `att_stall` crossing an SLR unregistered, an MVM reset copy into
   SLR2.

The k1 link (one channel, all in SLR1) was stopped: 170 K LUT in a 199,680-site pblock, placed WNS
-3.5 ns. If a quick bring-up image is wanted, k1 needs the k4 treatment (lanes out of SLR1).

## After the image (W6 and later)

- W6 card bring-up: identity, image load with VERIFY, HELLO, launch round trip, `tsr-selftest`, the
  GDB smoke over PCIe; M6 gate (0.5B logits bit-identical to the simulator, 32B KL), M7 gate (one
  launch per token, KL), DDR utilisation. Then README performance numbers on the new image and the
  Basis 1.0.0 release.
- Roadmap after the card gate: speculative decoding (n-gram, draft model, MTP) verified in one T_MAX 8
  pass; M8 MoE with Gemma 4 26B-A4B as the target; Gated DeltaNet kinds for Qwen3.8-27B; multi-card
  QSFP mesh; DeepSeek-V4-Flash and Qwen3.8-Flash-Next (multi-card).
- MoE/GDN planning decisions retained on 2026-11-02: see [design/README.md](design/README.md)
  §12.6 for the firmware/RTL boundary, TOPK enablement, independent MVM/VU paths, head-sized
  state tiles, numerical and performance checks, and llama.cpp state/MTP rollback integration.
- Optimisation levers kept for after the card gate: SPM banks as true dual-port URAM; about 812 free
  URAM (per-cluster attention buffers, larger preload area, T_MAX 16 buffers); the 256-line preload
  area (FLASH_ATTN above 680 tokens runs on the CPU today).
- Smaller open items: the `tsr_rv` cycle table still models the single-cycle multiplier (the RTL uses
  RV32MFast: MUL 3, MULH 4 cycles); 2K/16K long-prompt runs on the simulator; `tsr-prof report` and
  firmware PC sampling are designed but not built; an nvtop-like monitor (`tsr-top`) is designed but
  deferred.
