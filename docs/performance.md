# Performance and measurements

This document records what Tesseract has been measured to do and what the design expects it to do,
with the provenance of every number: decode throughput on the Alveo U200, where the decode time
goes, memory bandwidth, model load time, accuracy against the CPU, timing and resources, RTL
simulation cycle counts, the results on the simulator and the estimates. It ends with the
measurement history and the commands that reproduce the numbers.

> **Scope.** Measurement record, not a design. Card measurements were taken before the control
> core was added, with the host issuing the ops; the Vivado and RTL simulation results are of the
> design that ran on the card. The other numbers are simulator measurements or estimates, and none
> of them is a card measurement. Every number carries one of the provenance labels below. The
> normative targets and estimates of the control core are in [`docs/design`](design/README.md):
> [EX](design/01-execution-model.md) §4.11,
> [CC](design/04-control-core.md) §14, [AT](design/03-attention-kv.md) §9 and
> [SW](design/06-software-verification.md) §7.8, §10.4.

## How to read these numbers

Provenance labels:

| Label | Meaning |
| --- | --- |
| `measured on the card (k4, 276.3 MHz)` | Alveo U200, bitstream `tsr_u200_k4` at its achieved kernel clock. Other bitstreams are named: `k1`, `tsr_bw_k4`, or the milestone (M0, M3, M4) |
| `measured on the simulator (<commit>)` | Software time of this tree on the development host: host work, simulator speed, accuracy results of the tools |
| `simulator cycle model (<commit>)` | Device or RISC-V cycles from the simulator's informational cycle model. Not a card measurement |
| `measured in RTL simulation (ideal memory)` | Verilator simulation of the RTL with an ideal memory |
| `measured in Vivado 2024.2 (out of context)`, `(linked)` | Timing and utilisation of Vivado runs, without or with the shell |
| `estimate (<document> section N)`, `target (...)` | A number of the design that nothing has measured yet |
| `not re-measured (<document>)` | Copied from an earlier document of this repository |

**Decode tok/s** is the generation rate of llama.cpp's `eval time` line: batch 1, greedy
(`--temp 0`), `llama-completion -ngl 99 -nkvo -no-cnv`, a prompt of 6 to 7 tokens, 31 to 127
generated tokens. With `-nkvo` the KV cache stays on the CPU. The context was 4096 for the
Gemma 4 runs and for the `TSR_STATS` runs, and the model default for the Qwen2.5 runs (32768 for
0.5B, 131072 for 32B). Prompt processing is not part of the figure.

**Clocks.** The design target is 300 MHz (`F_CORE`). The 4-channel bitstream `tsr_u200_k4` closed
at 276.3 MHz, so every k4 number below is at 276.3 MHz; `tsr_u200_k1` and the M0 kernel
`tsr_bw_k4` ran at 300 MHz. The control core's clock `clk_rv` (planning value 166.7 MHz) exists
only in the design and the simulator.

**Host.** All card runs were made on the development host that holds the card: AMD Ryzen 7 9700X
(8 cores, 16 threads), 14 GB RAM, Ubuntu 24.04, kernel 7.0.0-38, XRT 2.18.179 with the 19 driver
patches ([`sw/xrt-driver`](../sw/xrt-driver/README.md)), shell `xilinx_u200_gen3x16_xdma_base_2`,
SC firmware 4.6.21. The U200 has a VU9P with three SLRs and 4 x 16 GB DDR4 (DDR4-2400, 19.2 GB/s
peak per channel). Simulator timings are from the same host.

## Decode throughput on the U200

Measured on the card (k4, 276.3 MHz) unless the row says otherwise. The runs offloaded only
`MUL_MAT`: attention, norms, RoPE and the other operations ran on the CPU, and llama.cpp disabled
flash attention for this reason (log line `Flash Attention not supported, set to disabled`).

| Model | Weights | Decode (tok/s) | ms per token | Generated tokens | Log |
| --- | --- | --- | --- | --- | --- |
| Qwen2.5-0.5B-Instruct | Q8_0 | **54.7** | 18.29 | 63 | `build/card/k4/llama.log` |
| Qwen2.5-0.5B-Instruct, `tsr_u200_k1` (1 channel, 300 MHz) | Q8_0 | 24.8 | 40.37 | 63 | `build/card/llama_k1.log` |
| Gemma 4 12B | Q8_0 | **4.50** | 222.03 | 127 | `build/card/m4/gemma-4-12b-it-Q8_0.log` |
| Gemma 4 12B | Q4_K_M | **6.91** | 144.67 | 64 | `build/card/m4/decode_dma_direct.log` |
| Qwen2.5-32B-Instruct | Q4_K_M | **2.93** | 341.72 | 31 | `build/card/m4/llama32b.log` |
| Gemma 4 E4B (single run) | Q8_0 | 9.65 | 103.62 | 127 | `build/card/m4/gemma-4-E4B-it-Q8_0.log` |

- The 64-token runs behind the next section give 4.51 tok/s (221.7 to 221.9 ms) for 12B Q8_0 and
  2.93 to 2.94 tok/s (340.6 to 340.9 ms) for 32B Q4_K_M.
- 12B Q4_K_M with the `staged` host DMA mode instead of the default `direct`: 6.78 tok/s
  (147.48 ms per token). Decode moves only small IO transfers, so the mode barely matters here;
  it matters for model load (see below).
- The Gemma 4 E4B row is one log; treat it as a single run.
- Logs are under `build/card/` and are not committed.

Decode is bound by the DDR weight stream. Every token reads every weight once, and the core
consumes one 64-byte beat per cycle per channel, so the next section's counters (97 to 98 % of the
cycles with a beat) and the M0 read bandwidth set the ceiling.

## Where the time goes

Measured on the card (k4, 276.3 MHz), 2026-10-27, `TSR_STATS=1`. Each figure is the
difference between a `-n 65` and a `-n 1` run of `llama-completion -ngl 99 -nkvo -c 4096`,
divided by the 64 decode tokens.

| Per decode token | Gemma 4 12B Q8_0 | Qwen2.5-32B Q4_K_M |
| --- | --- | --- |
| Wall (llama.cpp `eval time`) | 221.9 ms | 340.6 ms |
| Device busy (`PERF_CYCLES` / clock) | 184.5 ms (83.2 %) | 298.8 ms (87.7 %) |
| Exec start and completion wait (exec wall minus busy) | 10.1 ms (4.6 %), 36 µs per exec | 13.6 ms (4.0 %), 35 µs per exec |
| Backend host work (lowering, IO) | 6.3 ms (2.8 %), 22 µs per graph | 8.5 ms (2.5 %), 22 µs per graph |
| Outside the backend (CPU attention, norms, RoPE; scheduling, sampling) | 20.9 ms (9.4 %) | 19.7 ms (5.8 %) |
| Host total | 37.3 ms (16.8 %) | 41.8 ms (12.3 %) |
| Device round trips (graphs = execs) | 281 | 385 |

Device counters over the same tokens (per token):

| Per decode token | Gemma 4 12B Q8_0 | Qwen2.5-32B Q4_K_M |
| --- | --- | --- |
| `PERF_CYCLES` | 50.98 M | 82.56 M |
| `PERF_MVM_CYCLES` | 49.64 M (97.4 %) | 81.11 M (98.2 %) |
| Read beats per channel (channels 1 to 3) | 49.55 M (97.2 % of the cycles) | 80.97 M (98.1 %) |
| Bytes read (4 channels) | 12.69 GB | 20.74 GB |
| Effective read rate | 68.8 GB/s (17.2 per channel) | 69.4 GB/s (17.4 per channel) |
| `PERF_CH_WAIT` per channel | 37 K cycles (0.07 %) | 57 K cycles (0.07 %) |

- Every channel delivers a beat on 97 to 98 % of the busy cycles and DRAM almost never stalls the
  core. At 276.3 MHz the limit is the kernel's 64 bytes per cycle (17.7 GB/s per channel),
  slightly below the 18.0 GB/s the DRAM gives at 300 MHz (M0, below). Closing 300 MHz would add
  about 4 to 5 %, not the clock ratio.
- Beyond that, decode gets faster only by reading fewer bytes per token (lower-bit formats; native
  Q6_K instead of the Q8_0 conversion, which is about 1.3 GB of the 20.74 GB for 32B Q4_K_M) or by
  using several tokens per weight pass (T > 1).
- The largest part of the host share is the CPU work between the `MUL_MAT`s. The control core
  removes it by running norms, RoPE, activations and attention on the device and by launching once
  per token (designed; see [Estimates and targets](#estimates-and-targets)).
- 0.5B Q8_0 for comparison: 18.29 ms wall per token against about 8.25 ms of device time. That
  device time is derived: 2.28 M PERF cycles per decoded token from RTL simulation (see
  [RTL simulation cycle counts](#rtl-simulation-cycle-counts)) at 276.3 MHz. The remaining
  10 ms is host time.

## Memory bandwidth (M0)

Measured on the card (`tsr_bw_k4.xclbin`, 300 MHz), 2026-10-25. Kernel and tool:
[`docs/m0-bw.md`](m0-bw.md). All 1,633 runs passed their data checks.

| Measurement | Result |
| --- | --- |
| Capacity | 16 GiB per channel, no aliasing (64 GB in total); `CH_BYTES` stays 16 GiB |
| Read, bursts of 64 beats, at least 4 outstanding | 18.01 GB/s per channel (93.8 % of 19.2), 72.05 GB/s on four channels |
| Write, bursts of 64 | 15.92 GB/s per channel (82.9 %), 63.68 GB/s on four |
| Read + write | 35.6 + 35.6 = 71.2 GB/s on four channels |
| Outstanding sweep (read, bursts of 64) | 1: 8.93 GB/s (46.5 %); 2: 17.46 (90.9 %); 4 and more: 18.01 |
| Burst length (read, 32 outstanding) | 1 beat: 4.86 GB/s (25.3 %); 4: 17.07 (88.9 %); 16 and 64: 18.01 |
| First-beat latency | read 67 cycles (223 ns); write about 170 cycles |
| PCIe, `tsr_bw --pcie` (pageable buffer, 256 MiB) | host to card 1.94 to 2.01 GB/s; card to host 10.4 to 11.1 GB/s |

Reference measurements of the shell, measured on the card (`xrt-smi validate`, 8 tests
passed): memory 16.8 GB/s per bank and 67.2 GB/s in total; DMA host to card 7.3 to 7.9 GB/s, card
to host 13.3 to 13.5 GB/s. The weight upload of the backend uses either mode: `staged` (pinned
bounce buffer) reaches 8 to 9 GB/s, `direct` (pageable memory, the default) 2.2 to 3.9 GB/s.

The weight stream has 18.0 GB/s per channel, 93.8 % of the 19.2 GB/s the specification assumes.
Two outstanding bursts already give 91 %.

## Model load time

Measured on the card (k4, 276.3 MHz), 2026-10-30. `llama-completion -n 1` with
`TSR_STATS=1`, which splits `set_tensor` into transport writes, weight stores and the conversion of
non-native rows to Q8_0.

Before the parallel conversion:

| | 12B Q4_K_M, file cached, `direct` | same, `staged` | 32B Q4_K_M, partly cold, `staged` |
| --- | --- | --- | --- |
| llama.cpp load time | 12.2 s | 6.3 s | 34.9 s |
| Transport writes | 3.42 s for 7.5 GB (2.20 GB/s) | 0.88 s (8.61 GB/s) | 2.58 s for 20.8 GB (8.04 GB/s) |
| Q6_K to Q8_0 conversion | 4.6 s for 1.29 GB of Q6_K | 4.6 s | 17.1 s for 4.49 GB of Q6_K |
| Repack, page faults | about 0.3 s | about 0.3 s | about 6 s |
| Outside `set_tensor` (llama.cpp loader, warm-up) | about 0.5 s | about 0.5 s | about 9 s |

With the conversion on worker threads (`TSR_LOAD_THREADS`, default the online CPUs up to 16) and
the calling thread writing finished batches while the workers convert the next:

| `llama-completion -n 1` load time | Before | After |
| --- | --- | --- |
| 12B Q4_K_M (file cached) | 12.2 s | **4.0 s** (`set_tensor` 3.5 s, of which DMA 2.3 s) |
| 12B Q8_0 (partly cold) | 18.7 s | 11.7 s |
| 32B Q4_K_M (partly cold) | 34.9 s | **18.2 s** (`set_tensor` 9.2 s; conversion 30 s of CPU time over 16 threads) |

- The Q6_K to Q8_0 conversion (`to_float` plus `quantize_row_q8_0_ref`) ran at about 0.27 GB/s of
  Q6_K on one thread. Cold sequential disk reads ran at 2.65 GB/s.
- The device bytes are unchanged by the parallel conversion: the 12B Q4_K_M logits before and after
  are 14 of 14 rows bit-identical (`klcheck compare --expect-identical`).
- The first 32B load on the card took 70 s; the table's 34.9 s is a later run.

## Accuracy against the CPU

The criterion of `klcheck` ([`docs/klcheck.md`](klcheck.md)): mean KL(CPU, device) at most 3 times
the floor, and top-1 agreement at least 80 %. The floor is the mean KL between the CPU and the CPU
with a 1-ulp change of every input element. `ratio` is the mean KL over the floor.

**Qwen2.5-0.5B Q8_0, 8 prompts x 64 tokens = 512 positions**

| Run | Settings | Mean KL / floor | Ratio | Top-1 | Provenance |
| --- | --- | --- | --- | --- | --- |
| Simulator | FA off, f16 KV | 1.095e-3 / 1.136e-3 | 0.964 | 98.05 % (502/512) | not re-measured (`docs/klcheck.md`) |
| Card, k1 | FA off, f16 KV | 1.0950e-3 / 1.1357e-3 | 0.964 | 98.05 % | measured on the card (k1) |
| Simulator | M6: FA on, q8_0 KV, rotation off | 2.595e-3 / 2.530e-3 | 1.026 | 97.07 % | measured on the simulator (`83ac601`) |

On the card, `klcheck compare --expect-identical` between `sim` and `xrt` gives 8 of 8 rows
bit-identical (k1 and k4). The M6 run and its replay on `sim` and on `mock` give 512 of 512 logit
rows bit-identical with the same generated tokens. The ctest `tools_klcheck_qwen05b` (4 prompts x
32 tokens) gives ratio 0.899 against the mean of floor seeds 1 to 3 (0.717 against seed 1) and top-1 96.88 %.

**Qwen2.5-32B Q4_K_M.** The three ratios use different settings or prompt counts and are not
comparable.

| Run | Settings | Positions | Mean KL / floor | Ratio | Top-1 | Provenance |
| --- | --- | --- | --- | --- | --- | --- |
| M1, simulator | FA off, f16 KV | 4 x 64 = 256 | 2.550e-3 / 1.876e-3 | 1.36 | 98.83 % | measured on the simulator, `build/klcheck-32b/report.json` |
| M4, card | FA off, f16 KV | 8 x 64 = 512 | 2.4113e-3 / 1.8487e-3 | 1.304 | 99.02 % | measured on the card (M4) |
| M6, simulator | FA on, q8_0 KV, rotation off, `offload_kqv` false | 8 x 64 = 512 | 2.886e-3 / 2.246e-3 | 1.285 | 98.44 % | measured on the simulator (`83ac601`), `build/klcheck-32b-sim-m6/report.json` |

The M6 run is `klcheck all -m models/qwen2.5-32b-instruct-q4_k_m-00001-of-00005.gguf -o
build/klcheck-32b-sim-m6 -n 64` with `TSR_SIM_DRAM_DIR` set; its device phase took 4,434 s. All
three runs pass.

**numcheck** ([`docs/numcheck.md`](numcheck.md)): one prompt per model; every weight `MUL_MAT`
element is compared with `libtsr_ref` (B1), with an independent numpy model of the specification
(B1b) and against a proven fp32 error bound (B2). Baseline with FA off and f16 KV, measured on the
simulator, not re-measured:

| Model | B1 mismatches | B1b | B2 violations | KL ratio |
| --- | --- | --- | --- | --- |
| Qwen2.5-0.5B Q8_0 | 0 / 14,594,048 | 0 | 0 | 1.38 |
| Qwen2.5-0.5B Q4_K_M (requantized) | 0 / 14,594,048 | 0 | 0 | 1.46 |
| Qwen2.5-32B Q4_K_M | 0 / 76,881,920 | 0 | 0 | 2.12 (one 16-token prompt: coarse; per-position ratios 0.2 to 4.8) |

M6 check, measured on the simulator (`83ac601`): Qwen2.5-0.5B Q8_0, 32 tokens, all 169 weight
`MUL_MAT`s on the device, B1 0 mismatches, B1b 0, B2 0 violations, KL ratio 1.20, top-1 32 of 32.

**Op level.** `test-backend-ops -o MUL_MAT` ([`docs/llama-pin.md`](llama-pin.md)): 1403 cases are
supported (28 source types, K from 32 to 8192, N from 1 to 17408, up to 4096 tokens). Every case
passes except `type_a=f32,type_b=f32,m=1,n=1,k=2048`: it has a single output, so its NMSE is one
squared relative error, which exceeds 5e-4 in about 15 % of the random runs on the 8-bit path (the
exact dot product of the same 8-bit operands fails on the same runs). `ctest` leaves that case out
and checks it bit-exactly. Measured on the card (k4): 1402 of 1402 cases, including the f16
case with 2,105,344 tokens, and `test-ggml-tesseract` 28 of 28. On the simulator, 1403 of 1403
passed in one run (measured on the simulator, working tree after `83ac601`; 69 s with 4 simulator
threads).

**PyTorch backend** ([`docs/torch-backend.md`](torch-backend.md)): Qwen2.5-0.5B-Instruct (float32
safetensors), all 169 linears on the device including the tied `lm_head`; the logits are
bit-identical to eager PyTorch with the reference model in place of each linear, and 16 greedy
tokens equal the CPU's (simulator). An 8-token forward on the Verilated k4t1 kernel gives the
simulator's logits bit for bit (RTL simulation). On the card (k1): `hf_check.py
--forward-only --tokens 8 --expect` 8 of 8 rows, test suite 85 passed and 2 skipped.

## Timing, clock and resources

Measured in Vivado 2024.2 (device `xcu200-fsgd2104-2-e`). The Xilinx technology wrappers
and the U200 bitstream parameters (`SPM_RD_LAT` 4, `MVM_BC_PIPE` 2, `MVM_RES_PIPE` 2,
`MVM_MAC_LAT` 9, `RST_PIPE` 3) are used. Details and the history of the fixes:
[`docs/rtl-m3.md`](rtl-m3.md) §14, [`docs/platform-u200.md`](platform-u200.md) §8 and §14.

**Out of context** (`tsr_core` alone, routed, 3.333 ns, no shell, AXI bridges or DDR4
controllers; measured in Vivado 2024.2, out of context):

| Configuration | Floorplan | WNS / WHS (ns) | CLB LUT | FF | DSP | RAMB36 | URAM |
| --- | --- | --- | --- | --- | --- | --- | --- |
| k1 (1 channel, T_MAX 1) | none | +0.100 / +0.009 | 30,902 | 44,516 | 182 | 40 | 128 |
| k4t1 (4 channels, T_MAX 1) | C' | +0.109 / +0.012 | 67,345 | 113,748 | 632 | 136 | 128 |
| k4 (4 channels, T_MAX 8) | C' | +0.022 / +0.019 | 272,025 | 349,199 | 2,728 | 808 | 128 |

The first k4 run (no floorplan) failed at WNS -0.877 ns (Fmax 237 MHz); with the first fix set and
floorplan C (clusters 0 and 2 in SLR0, 1 and 3 in SLR2) it reached 286 MHz. The final fixes (kept
per-cluster pipes, a gather split per cluster, `MVM_MAC_LAT` 9, `RST_PIPE`) change only latencies,
and every trace still replays bit-exactly.

**Linked** (with the shell, routed, after post-route `phys_opt`; measured in Vivado 2024.2,
linked):

| Bitstream | Kernel clock | WNS (ns) | Notes |
| --- | --- | --- | --- |
| `tsr_u200_k1` | 300 MHz | +0.017 (shell XDMA logic; kernel clock +0.060), WHS +0.009 | 33,502 LUT, 51,212 FF, 40 RAMB36, 128 URAM, 182 DSP, all in SLR1; link 57 min |
| `tsr_bw_k4` (M0) | 300 MHz | +0.017 (shell; kernel clock +0.055) | 11,907 LUT, 20,740 FF, no BRAM, URAM or DSP |
| `tsr_u200_k4` | **276.3 MHz** | -0.285 at 300 MHz (809 failing endpoints, TNS -56.5 ns), so `v++` scaled the clock (`AUTO-FREQ-SCALING-04`) | link 3 h 51 min on the 14 GB host with up to 28 GB of swap; the worst paths are in the gather's job pipes |

**Floorplan C'** (`hw/u200/k4_floorplan.tcl`). Two T_MAX 8 clusters need 1,348 DSP (1,380 with QACT)
and the dynamic region of SLR1 has 1,320, so the clusters cannot all sit next to their DDR4
controller. Clusters 0 and 2 are in SLR0, clusters 1 and 3 in SLR2, and the rest of the core and
the XRT adapter in SLR1. Only register pipes cross SLRs: the activation broadcast, the weight
request and response pipes, the result-gather job and start pipes, the reset copies and the AXI
register slices of channels 0 and 3. The weight-response pipe moves one beat per cycle without
backpressure, so a crossing costs latency, not bandwidth. The DDR banks sit in SLR0 (bank 0), SLR1
(banks 1 and 2) and SLR2 (bank 3). The dynamic region of the three SLRs has 388,160 / 205,440 /
385,440 LUT, 2,280 / 1,320 / 2,280 DSP, 720 / 420 / 720 RAMB36 and 320 / 160 / 320 URAM.

<p align="center"><img src="fig/floorplan/k4-core-ooc.png" alt="Placement of the routed 4-channel T_MAX 8 tsr_core on the VU9P: clusters 0 and 2 in SLR0, clusters 1 and 3 in SLR2, gather, QACT, control and SPM in SLR1" width="100%"></p>

Routed, out of context, floorplan C'. Full-device views after the `v++` link:
[`tsr_u200_k1` with the XRT shell](fig/floorplan/k1-linked.png) and
[the M0 kernel with the four DDR4 controllers](fig/floorplan/bw-linked.png). All three images are
drawn by [`hw/viz`](../hw/viz/README.md).

**Resources.** The MAC-lane DSP48E2 count per cluster is 128 at T_MAX 1 (one product per DSP) and
512 at T_MAX 8 (two tokens share one DSP), which gives 2,048 MAC DSPs and 2,728 DSPs in all for k4.
RAMB36 per cluster is 24 x T_MAX for the activation buffer plus 8 for the weight FIFO (32 at
T_MAX 1, 200 at T_MAX 8). The 4 MiB SPM takes 128 URAM288.

## RTL simulation cycle counts

Measured in RTL simulation (ideal memory): the U200 configuration at the default
implementation parameters (M3 part C RTL), ideal memory (always ready, latency 0 to 1, on every
channel at once), cycles converted at the 300 MHz target. Not card measurements. The M3 gate
shapes, Qwen2.5-0.5B Q8_0, one token, 4 channels:

| MVM | Weight beats per channel | Cycles | Cycles per beat | At 300 MHz |
| --- | --- | --- | --- | --- |
| K 896 x N 896 (q, o) | 3,360 | 3,448 | 1.026 | 11.5 µs |
| K 896 x N 4864 (gate, up) | 18,240 | 18,328 | 1.005 | 61.1 µs |
| K 4864 x N 896 (down) | 18,144 | 18,298 | 1.008 | 61.0 µs |
| K 896 x N 151936 (output head) | 569,760 | 569,848 | 1.0002 | 1.90 ms |

- At the U200 bitstream parameters the four shapes take 11 to 12 more cycles each. The steady state
  is 1.0000 cycles per weight beat for every tested shape at T = 1 and T = 8. Example: the 32B
  `ffn_down` (Q4_K, N 5120, K 27648) is 1280 rows x 243 beats = 311,040 cycles per channel, 1.04 ms
  at 300 MHz.
- With the part B RTL the four shapes took 3,445, 18,325, 18,297 and 569,845 cycles; the part C
  pipes add 1 to 3 cycles per MVM. The whole 1-token trace set of Qwen2.5-0.5B Q8_0 (145 traces,
  the offloaded `MUL_MAT`s only) took 2.27 M cycles (7.6 ms at 300 MHz) under the `fast` memory
  regime (latency 0 to 2), with the part B RTL.
- Through the whole host stack on the Verilated U200 kernel (`vkernel`, 145 execs per token), one
  decoded token takes 2,281,377 PERF cycles on k4t1 (7.60 ms at 300 MHz) and 8,476,150 on k1
  (28.25 ms). The kernel adds 99 cycles per exec. On k4 (T_MAX 8) a 6-token prompt runs as one
  T = 6 pass of 3.14 M cycles, against 13.67 M at T_MAX 1.
- The RTL replay of the trace sets returns with the RTL work packages W4 and W5 (see
  [Verify](getting-started.md#verify)).
- Mutation campaigns (every test must fail on at least one deliberate mutant): the M3 core 31
  mutants, 28 killed and 3 equivalent (marked in advance); the M2 core all killed except 2
  equivalent; the U200 adapter 33 mutants, 29 killed, 4 equivalent; the M0 kernel 57 mutants, 54
  killed, 3 equivalent. In numcheck a sequential-sum mutant fails B1 and B1b and passes B2.

## Results on the simulator

Everything here is a simulator result: software time of this tree on the development host, or the
simulator's cycle model. No RTL or bitstream of the control core exists yet. Commands: see
[Reproducing the numbers](#reproducing-the-numbers).

**M6 host stack** (measured on the simulator (`83ac601`), Qwen2.5-0.5B Q8_0, M6 settings):

| Quantity | Value |
| --- | --- |
| Decode time on the simulator | 90 to 95 ms per token |
| Launches per token | 145.00: one per graph split |
| `MUL_MAT` records per launch | 1.17 |
| Op-list cache | 73,385 hits, 855 misses (98.85 %; the commit message of `83ac601` rounds to 98.9 %) |
| Qwen2.5-32B Q4_K_M, file-backed simulator DRAM (1 prompt x 2 tokens) | load 27.9 s, prefill 19.9 s for 7 tokens, decode 7.72 s per token, 385 launches per token |
| Qwen2.5-32B Q4_K_M, full `klcheck all` (8 x 64) | about 8 s per decoded token; device phase 4,434 s |

**Firmware** (simulator cycle model, Ibex "maxperf" pipeline, `clk_rv` 166.7 MHz, core clock
276.3 MHz, fixed schedule; the figures move by a few cycles with every firmware commit, and
`ctest --test-dir build/dev -R '_perf' -V` prints the current ones):

| Quantity | Value | Commit |
| --- | --- | --- |
| Descriptor push to kernel entry | 2.19 µs (605 core cycles) | `92f0b17` |
| Empty launch, push to DONE (device part) | 5.57 µs (1,538 core cycles) | `92f0b17` |
| Same, after the executor optimisation and the Ibex maxperf default | 1.68 µs and 4.22 µs; 8 outstanding launches 4.33 µs each (231 K launches/s). Test `fw_m6_perf` on the production image `tsr-fw.elf` (an empty launch is `selftest` with an unknown id); the test `fw_runtime` on its own test image (`fw_rt.elf`, 19 kernels) reads 1.55 µs and 3.21 µs | `41eb0da`; re-run on the tree of `a775222` |
| Executor, RISC-V cycles per op record | 1,976, then 387 with deep queues (401 with the U200 queue depths; 437 by the method of `92f0b17`) | `92f0b17`, `41eb0da` |
| One 0.5B decode layer (9 records: 7 `MUL_MAT`, 2 joins) | RISC-V 110 µs (W2a firmware, maxperf table; 141 µs on the small table), then 39.5 µs, against 197 µs of MVM work | `92f0b17`, `41eb0da` |
| MVM idle on issue, 0.5B layer | 8.1 %, then 0 % | `92f0b17`, `41eb0da` |

M7 kinds (norms, element-wise operations, GELU16, RoPE, `GET_ROWS`, `SET_ROWS_KV`,
`FLASH_ATTN`), verified at layer level on the simulator (`a775222`); the backend does not emit them
yet. Simulator cycle model, [SW](design/06-software-verification.md) §7.8; re-run on the tree of `a775222`
with `ctest --test-dir build/dev -R 'fw_exec_perf_m7|fw_m7_layer_perf|fw_kvfa_perf' -V`:

- **Short records at the Qwen2.5-0.5B decode shapes** (`fw_exec_perf_m7 --strict`, n = 896, spin-free
  RISC-V cycles of work per record, deep queues): 223 (`CPY`) to 336 (`MUL`), among them `RMS_NORM`
  324 and `ROPE` 325; every kind meets the 350-cycle requirement. Decode records of other shapes
  still exceed it: Gemma 4's per-head `RMS_NORM` (405 to 455), `ROPE` with a frequency-factor table
  or `freq_scale` other than 1 (439 to 475), and every record whose scalar-register values change
  (SW §7.8 lists them; they are a few per layer of models whose layers take milliseconds).
- **One real Qwen2.5-0.5B M7 decode layer** (`fw_m7_layer_perf`, 15 records and a join, three layers
  against one): the control core works 50.7 µs per layer against 270.6 µs of device time. Against a
  near-ideal issuer the RISC-V's speed costs 5.1 % of the launch with the U200 queue depths, 2.1 %
  with deep queues (also with only the DMA queue 31 deep) and 2.5 % at a 250 MHz control core. The
  loss comes from the 8-entry DMA queue and the planning time of `FLASH_ATTN`, not from the short
  records.
- **`FLASH_ATTN`**: a Qwen2.5-32B decode record at 4,096 cells takes 3,378 RISC-V cycles (20.3 µs)
  of planning against 140.0 µs on the units.

**Firmware image** (`tsr-elf size build/dev/fw/tsr-fw.elf`, measured on the production image;
ITCM is 131,072 B, DTCM 65,536 B):

| Image | ITCM | DTCM |
| --- | --- | --- |
| W2a (`92f0b17`: `graph`, `selftest`, `replay`) | 25.8 KiB | 46.3 KiB |
| With the M7 kinds (`a775222`) | 71,588 B (54.6 %; budget 81,920 B) | 49,448 B (75.5 %; budget 57,344 B) |

**Verification facts** (measured on the simulator, as recorded in the commit messages):

- The RV32 interpreter passes the 64 riscv-tests (`a54a6e6`); with Spike installed, `rv_isa` 65,
  `rv_arch` 92, `rv_lockstep` 62 and `rv_random` 50 pass (`1ed7460`).
- The trace sets m2, gen, vu, at, ctrl, m3 and model-v1conv regenerate byte-identically and replay
  under the fixed and two random schedules: 3,688 replays, 0 failed (`1ed7460`). The race
  detector adds 1.0x to 1.3x to the run time.
- The flash-attention reference against ggml-cpu `flash_attn_ext`: NMSE at most 1.9e-11 (9.4e-9 on
  one CPU quantization tie), KL at most 8e-12; RoPE angles are bit-identical to ggml-cpu
  (`a54a6e6`).
- GELU16 differs from the pinned ggml fp16 table in 272 of 65,536 entries (7 of them by 2 fp16
  ulp); GELU16 is the correctly rounded value. The `ref_math` tests cover all 2^32 inputs when
  `TSR_REF_MATH_FULL=1` is set (`452e27b`).
- The layer tests `fw_m7_layer_ggml` compare whole decoder layers of Qwen2.5-0.5B and a Gemma 4 12B
  sliding-window layer with ggml-cpu: layer-output NMSE between 3e-14 and 5e-6.

## Estimates and targets

None of these is measured. Each row names its source in `docs/design`; the estimates are checked on
the card in W6 and in synthesis in W4/W5.

| Quantity | Value | Label |
| --- | --- | --- |
| Round trip of an empty launch | 4 to 8 µs, target at most 10 µs | estimate (EX §4.11) |
| Pipelined launch rate | at least 300 K launches/s; target at least 100 K/s | estimate (EX §4.11) |
| Launch overhead saved per decode token in M6 | 7 to 12 ms of the 10.1 / 13.6 ms exec start and completion wait measured on the card (12B / 32B) | estimate (EX §4.11) |
| Firmware load | at most 10 ms per cold load | estimate (EX §3.3, EX-8) |
| PCIe MMIO read latency | assumed 1 to 2 µs per read; not measured (open item) | estimate (EX §4.11) |
| SLR1 LUT use | 125 to 143 K (62 to 72 %) against the 65 % gate (CC-24) | estimate (CC §14.1) |
| Control core, issue path, CSR block | 14.5 to 25 K LUT, 49 RAMB36, 4 to 6 DSP | estimate (CC §14.1, §14.5) |
| Vector unit | about 48 K LUT, 15 RAMB36, 366 DSP (plus or minus 30 %) | estimate (CC §14.1) |
| Debug block | 4.2 to 6.3 K LUT, 7 RAMB36 | estimate (CC §14.1) |
| Attention cluster modes | 19 to 27 K LUT per cluster (SLR0 and SLR2) | estimate (CC §14.1) |
| Attention share of a decode token at 4K context | 0.5B Q8_0 about 9 %, 12B Q4_K_M about 3.5 %, 32B Q4_K_M about 3.0 % (device time) | estimate (AT §9.3) |
| 32B decode layer at 4K context | about 39,900 cycles (144 µs), DDR busy about 89 % | estimate (AT §9.2) |
| Attention of D = 64 models | gather-bound, 61 % efficiency | estimate (AT §9.1) |
| Timing gates | T1 control core out of context at 166.7 MHz; T2 k4 core out of context at 3.333 ns; T3 k4 link at least 276.3 MHz (target 300 MHz); T4 k1 link 300 MHz | target (CC §14.3) |
| DDR utilisation | at least 85 % at M9 for 32B Q4_K_M decode | target ([`docs/spec.md`](spec.md) §14) |

## Measurement history

| Date | Step | Result | Provenance |
| --- | --- | --- | --- |
| 2026-10-22 | M1: ISA simulator and ggml backend | 32B klcheck on the simulator: ratio 1.36, top-1 98.83 % (4 prompts x 64) | measured on the simulator |
| 2026-10-23 | M2: RTL core skeleton | Verilator trace replay bit-exact, 172 traces | not re-measured (README of the time) |
| 2026-10-24 | M3 part C: Xilinx wrappers, U200 adapter, XRT transport, M0 kernel, Verilated kernel | the first full `v++` links on the development host: `tsr_bw_k4` and `tsr_u200_k1` at 300 MHz (k1 in 57 min); `vkernel` and hw_emu runs; PyTorch backend | measured in Vivado 2024.2 (linked); RTL simulation |
| 2026-10-25 | M0 on the card | 72.05 GB/s read on four channels, 18.01 GB/s per channel; 16 GiB per channel | measured on the card (`tsr_bw_k4`) |
| 2026-10-25 | M3 on the card, k1 | logits bit-identical to the simulator (8/8 rows), KL ratio 0.964, 24.8 tok/s | measured on the card (k1, 300 MHz) |
| 2026-10-26 | M3 on the card, k4 (link 3 h 51 min, 276.3 MHz) | same checks, 54.7 tok/s | measured on the card (k4, 276.3 MHz) |
| 2026-10-27 | Host overhead and device counters, 12B and 32B | decode is bound by the 64 B/cycle read path | measured on the card (k4) |
| 2026-10-30 | Model load: conversion on worker threads | 12B Q4_K_M 12.2 s to 4.0 s; 32B 34.9 s to 18.2 s | measured on the card (k4) |
| 2026-10-30 | M4: Qwen2.5-32B Q4_K_M on the card | 2.93 tok/s, KL ratio 1.304, top-1 99.02 % (`de07d92`) | measured on the card (M4) |
| 2026-10-30 | Design set `0dca742` and `85b7713` | six area documents, 167 numbered decisions (175 after W2b), work packages W0 to W6 | design only |
| 2026-10-30 | W0 `452e27b`, `43dcf74` | table source; VU reference math (exhaustive) | measured on the simulator |
| 2026-10-30 | W1a `a54a6e6`, W1b `1ed7460` | simulator, RV32 interpreter with Spike lockstep, VU and AT units, scheduler, race detector, trace format, debug models | measured on the simulator |
| 2026-10-30 | W2a `92f0b17`, executor optimisation `41eb0da` | firmware SDK, `graph`, `selftest`, `replay`; 1,976 to 387 RISC-V cycles per op record | simulator cycle model |
| 2026-10-30 | W3a `83ac601` | transport, M6 backend and tools; M6 on the simulator: 0.5B ratio 1.026 | measured on the simulator (`83ac601`) |
| 2026-10-31 | M6 on the simulator, Qwen2.5-32B Q4_K_M, full klcheck | ratio 1.285, top-1 98.44 % | measured on the simulator (`83ac601`) |
| 2026-10-31 | W2b `a775222`, decisions `802311e`: M7 kernels including flash attention | layer-level tests against the composed references and ggml-cpu; every short kind at the 0.5B decode shapes at or under 350 RISC-V cycles; 5.1 % of a 0.5B decode layer lost to issue with the U200 queues | simulator cycle model; `docs/design/README.md` §10 |

Next: W3b (M7 backend, one launch per token, M7 on the simulator), W4 (RTL), W5 (Verilator and
Vivado integration), W6 (card), [`docs/design/README.md`](design/README.md) §10.

## Reproducing the numbers

The logs of the card runs are under `build/card/` (not in git). The method of the decode, time
split and accuracy runs is described with their sections above; they can be repeated once the card
runs again (work package W6). The M0 bandwidth run uses the `tsr_bw` tool, which builds with
`-DTSR_WITH_XRT=ON` ([`docs/m0-bw.md`](m0-bw.md)):

```sh
# M0 bandwidth (card)
tsr_bw --dry-run
tsr_bw --pcie --sweep-outstanding --ids 1 --csv build/card/m0_ids1.csv
tsr_bw --ids 16 --csv build/card/m0_ids16.csv
```

The simulator results use the current tree (`build/dev`):

```sh
# M6 accuracy on the simulator: 8 prompts x 64 tokens
build/dev/bin/klcheck all -m models/qwen2.5-0.5b-instruct-q8_0.gguf -o build/klcheck-05b -n 64
# the 32B run needs file-backed simulator DRAM (about 20 GiB free in the directory)
TSR_SIM_DRAM_DIR=/path/with/space build/dev/bin/klcheck all \
    -m models/qwen2.5-32b-instruct-q4_k_m-00001-of-00005.gguf -o build/klcheck-32b-sim-m6 -n 64
# numeric check of every weight MUL_MAT
build/dev/bin/tsr-numcheck all -m models/qwen2.5-0.5b-instruct-q8_0.gguf -o build/numcheck-05b -T 32 --checks full
# firmware cycle counts, launch costs and the M7 layer costs
ctest --test-dir build/dev -R '_perf' -V
# op level
build/dev/bin/test-backend-ops -o MUL_MAT -b Tesseract0
# firmware image size
build/dev/sw/ctrl/host/tsr-elf size build/dev/fw/tsr-fw.elf
```

Vivado runs are described in [`docs/rtl-m3.md`](rtl-m3.md) §14 and
[`docs/platform-u200.md`](platform-u200.md) §8. The simulator runs are slow: 0.5B decodes at about
92 ms per token, 32B at about 8 s per token. The steps of
[Verify](getting-started.md#verify) cover the checks that need no model.
