# Performance

This page says how to measure Tesseract and what to report with a result. The last section holds
the results of the current release.

## What is measured

- Decode rate: tokens per second of the generation phase. `llama-bench` reports it as `tg` and
  llama.cpp prints it on the `eval time` line. Batch 1.
- Prefill rate: tokens per second of prompt processing (`pp` of `llama-bench`, `prompt eval time`
  of llama.cpp) at a stated ubatch size (`-ub`).
- Decode power: the board power of the card while it decodes, and the decode rate per watt.
- Device cycles per token: the change of the `PERF_CYCLES` counter over the launch of one decode token.
- Device memory: the DRAM in use on the card.
- Unit busy cycles per token: `MVM_CYCLES`, `VU_CYCLES`, `DMA_CYCLES` and `QACT_CYCLES`, and the
  stall counters `MVM_WAIT` (idle while the queue holds an entry that cannot start) and `MVM_EMPTY`
  (idle with an empty queue, which means issue bound).
- DDR traffic: `CH_RD_BEATS` per channel (64 B per beat) and `CH_WAIT`.

Accuracy is a separate question. Check a configuration with [klcheck](klcheck.md) and
[numcheck](numcheck.md) before quoting its speed.

## Tools

| Tool | Use |
| --- | --- |
| `llama-bench` | prefill and decode rates, with the mean and the standard deviation of the repetitions |
| `llama-cli` | one chat turn with the model's template; with `TSR_STATS=1` it also prints the backend report and the PERF totals at exit |
| `llama-completion` | decode and prefill rates of a raw prompt; with `TSR_STATS=1` it also prints the backend report and the PERF counter deltas |
| `tsr-mtp` | speculative decoding with a model's MTP head; device cycles per phase for a list of draft lengths |
| `sw/torch/tools/hf_check.py` | one forward of a Hugging Face model through the PyTorch backend: regions, launches, device cycles, host traffic and KL against eager PyTorch |
| `klcheck`, `tsr-numcheck` | accuracy against the CPU and bit identity against the simulator |
| `tsr-selftest` | identity of the device: `CAPS`, `T_MAX`, `NUM_CH`, kind table, firmware image id |
| `tsr-top` | live utilization of the card during a run, with the execution mode of each process |
| `tsr-prof`, `sw/tools/prof/prof_report.py` | trace-writer timeline and the profile report of one decode token ([prof-dbg.md](prof-dbg.md) §3.1) |
| `tsr_bw` | DDR4 read and write bandwidth of the card ([bandwidth-kernel.md](bandwidth-kernel.md)) |
| `ctest -R _perf` | RISC-V cycles per record and per layer, from the simulator's cycle model |

## Environment

Complete lists are in [getting-started.md](getting-started.md) and [prof-dbg.md](prof-dbg.md) §1.

| Variable | Effect on a measurement |
| --- | --- |
| `TSR_DEVICE` | `sim`, `mock` or `xrt` |
| `TSR_XCLBIN` | the image the `xrt` transport loads |
| `TSR_IMAGE` | the firmware ELF to boot (default: the build's `tsr-fw.elf`) |
| `TSR_KINDS` | the op kinds the device takes. Unset runs every supported kind on the device (full mode). `m6` with `-nkvo` runs only the matrix products on the device (matmul mode). |
| `TSR_STATS=1` | at exit: transport statistics, the backend report (launches, op records per launch, plan cache, time in launches) and the PERF counter deltas |
| `LLAMA_ATTN_ROT_DISABLE=1` | required with the device: the attention rotation is not supported |
| `TSR_SIM_TMAX` | `T_MAX` of the simulated device, 1 to 8 (default 1) |
| `TSR_SIM_CARD`, `TSR_SIM_RV_RATIO`, `TSR_SIM_VU_TIMING`, `TSR_SIM_GDN` | give the simulator's cycle model the configuration of a card image: VU lanes, the RISC-V to core clock ratio, `CAPS` bit 8 |
| `TSR_SIM_DRAM_DIR` | back the simulated DRAM with files, for models that do not fit in host RAM |
| `TSR_DUMP_PLAN=<dir>` | write the plan, the launches with their PERF deltas and the trace records that `prof_report.py` reads |
| `TSR_MON=0` | turn off the record that `tsr-top` reads (on by default) |
| `TSR_MON_PERF_MS=0` | read the PERF counters after every launch instead of every 100 ms, so the record holds the cycles of each launch ([prof-dbg.md](prof-dbg.md) §7.2) |
| `TSR_GDN_CELL=0`, `TSR_HEAD_SPLIT=0` | switch off a host lowering choice, for an A/B comparison |
| `TSR_MVM_DUAL=0` | one weight stream per cluster on a core with the dual weight stream, for an A/B comparison ([rtl-mvm.md](rtl-mvm.md) §15) |
| `TSR_PIVOT=0` | Tesseract Pivot off, the nodes keep llama.cpp's order, for an A/B comparison ([design/07-graph-scheduler.md](design/07-graph-scheduler.md)) |

Full mode and matmul mode are defined in [design/README.md](design/README.md).

## Procedure

Rates. Run `llama-bench -m MODEL -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -ub 512 -p 512 -n 64 -r 3 -t 8
-o jsonl` with `LLAMA_ATTN_ROT_DISABLE=1`. It loads the model once, runs a warmup, then times three
repetitions of a 512-token prompt (`pp512`) and three of 64 generated tokens (`tg64`). Each result
line gives the mean and the standard deviation of the three rates. The default `-ub` is 512.
Attention for a ubatch of more than 680 tokens runs on the CPU.

Power. While `llama-bench` runs, read the board power of the card every 0.2 s, with a timestamp. The
reading is `power1_input` (microwatts) of the hwmon node of the card's user function, which is the
XRT shell's board sensor and not the host's power. Take the idle power as the mean of ten readings
before the run, with no process on the card. Take the decode power as the mean of the readings inside
the `tg64` repetitions. A result line is timestamped when `llama-bench` prints it, and its
`samples_ns` give the length of the timed repetitions. The decode rate per watt is the `tg64` mean
over the decode power.

Cycles, memory and a text check. Run one chat turn with the model's template, so that the model
sees the prompt it was trained on: `llama-cli -m MODEL -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -c 4096
--temp 0 -n 65 -t 8 -sys "You are a helpful assistant." -p "PROMPT" -st -rea off --simple-io`, with
`TSR_STATS=1` and `TSR_MON_PERF_MS=0`, under `script -qec` so that it has a terminal. It is a single
turn with greedy sampling and thinking off. Read the text and check that it is coherent. Do not judge
a model from a raw prompt without its template: an instruct model can degenerate on one, even on the
CPU. During the run, poll the monitor record of the process (`/dev/shm/tsr-mon-<pid>`,
[prof-dbg.md](prof-dbg.md) §7.2). It holds the PERF totals after every launch and `mem_used`. The 64
launches after the prompt each decode one token. The cycles per token are the mean `PERF_CYCLES`
change of those launches. The device memory is the largest `mem_used` of the run.

Speculative decoding. Run `tsr-mtp` with `--mtp-mode all` and a `--mtp-k-list`, and compare the
cycles per token with plain decoding of the same prompts.

Host. Wall time depends on host load and the device counters do not. Measure on a quiet host, set
`-t` to the free physical cores, and note the load average at the start of each run. On the card
the PERF counters stay readable after the process exits, until the next open: `tsr-dbg --attach
-ex perf`.

Simulator. Cycle counts of the simulator are the output of a model. With the card configuration
set (`TSR_SIM_CARD` and the variables above) the model follows the RTL and the card. Say which
configuration a figure comes from.

## Device cycles and wall time

`PERF_CYCLES` counts core clock cycles while a launch is active. The device time of a token is
`PERF_CYCLES / f_core`. The core clock is that of the loaded image, which `tsr-top` and the
`clock_freqs` file in sysfs show. The RISC-V runs on its own clock, and the `RV_FREQ` register
reads the ratio to the core clock.

The wall time of a token adds three terms to the device time:

- t_launch: the launch round trip, which is the start and the completion wait over PCIe, once per
  launch;
- the backend's host work: lowering and IO;
- the work outside the backend: scheduling, sampling and any op that runs on the CPU.

The last two make t_host.

```
tokens per second = 1 / (cycles per token / f_core + t_launch * launches per token + t_host)
```

In full mode a decode token is one launch, so the device share of the wall time is the figure to check.
In matmul mode a token has one launch per graph split, and the CPU ops add to `t_host`.

The weight traffic sets a floor. A decode token reads every weight once. With the dual weight
stream (`CAPS` bit 16, [rtl-mvm.md](rtl-mvm.md) §15) a cluster can take two 64-byte beats per core
clock for an MVM with T ≤ T_MAX / 2, which covers every weight MVM of a one-token decode launch. At
200 MHz that requests 25.6 GB/s per channel, more than a DDR4-2400 channel gives (19.2 GB/s peak,
[spec](spec.md) §1, Design principles). So the memory channel, not the core clock, sets the floor:
the cycles per token cannot fall below the largest per-channel weight volume divided by the bytes
per core clock that the channel delivers. The DDR utilization U, which is the bytes read per token
times tokens per second over the measured bandwidth of the channels, compares a run with that floor
([spec](spec.md) §14, [prof-dbg.md](prof-dbg.md) §3).

Cycle counts are the portable figure. They do not depend on host load, and they carry to another
core clock as long as the DDR channels keep up. Two things move them with the clocks. The gaps where
the MVM waits for the RISC-V to issue (`MVM_EMPTY`) depend on the RISC-V to core clock ratio, and the
DDR latency in core cycles depends on the core clock. Compare cycles per token between images only
at the same ratio.

## What to report

- Host: CPU, RAM, kernel, XRT version, card and shell.
- Image: the sha256 and UUID of the xclbin, the core and RISC-V clocks, and the identity line of
  `tsr-selftest` (`CAPS`, `T_MAX`, `NUM_CH`, kind table, firmware image id).
- Software: the commit of this repository, the build type and the pinned llama.cpp commit.
- Model: file name, source, weight type and sha256.
- Run: the command line with every `TSR_*` variable, the mode (full mode or matmul mode), `-t`, `-c`, `-ub`, the KV
  types, and the prompt and generated token counts.
- Numbers: the number of runs, the mean and the standard deviation (or the median and the range),
  the idle and the decode power, the device cycles per token, the unit busy cycles, the device share
  of the wall time and the load average at the start.
- Accuracy: the klcheck ratio and top-1 for the same configuration, and the bit identity against the
  simulator where it was checked.
- For a simulator figure: the cycle model configuration.

## Results

Seven models on one Alveo U200 card with the Tesseract Core 1.1.0 image, in full mode, measured as
described under Procedure. Prefill and decode rates are the mean ± the standard deviation of three
repetitions.

| Model | Quantization | File size | Device memory | Prefill tok/s (pp512, -ub 512) | Decode tok/s (tg64) | Decode power (W) | tok/s/W | Device cycles per token |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.8 27B | Q8_0 | 26.6 GiB | 25.8 GiB | 5.589 ± 0.001 | 2.295 ± 0.003 | 60.5 | 0.038 | 86.21 M |
| Qwen3.5 9B | Q8_0 | 8.9 GiB | 8.1 GiB | 18.435 ± 0.005 | 7.138 ± 0.008 | 59.4 | 0.120 | 27.64 M |
| Gemma 4 31B | Q4_K_M | 17.1 GiB | 19.1 GiB | 9.298 ± 0.004 | 3.397 ± 0.001 | 62.7 | 0.054 | 58.55 M |
| Gemma 4 26B-A4B | Q8_0 | 25.0 GiB | 25.6 GiB | 39.310 ± 0.021 | 14.483 ± 0.023 | 58.9 | 0.246 | 13.54 M |
| Gemma 4 12B | Q8_0 | 11.8 GiB | 12.2 GiB | 15.538 ± 0.004 | 5.220 ± 0.001 | 60.8 | 0.086 | 38.11 M |
| Gemma 4 E4B | Q8_0 | 7.6 GiB | 4.9 GiB | 39.394 ± 0.021 | 12.493 ± 0.003 | 60.2 | 0.208 | 15.82 M |
| Gemma 4 E2B | Q8_0 | 4.7 GiB | 2.5 GiB | 76.002 ± 0.029 | 23.545 ± 0.029 | 58.2 | 0.404 | 8.35 M |

- Device memory is the largest `mem_used` of the chat run at `-c 4096`. It holds the weights, the KV cache
  and the work buffers. A weight type other than Q8_0 and Q4_K is converted to Q8_0 when the model
  loads. The per-layer embedding tables of Gemma 4 E4B and E2B stay in host memory.
- Decode power is the mean card board power over the timed `tg64` repetitions. The tok/s/W column is
  the decode rate over it.
- Device cycles per token are the mean over the 64 decode launches of the chat run, in millions of
  core cycles. At the 200 MHz core clock the device time of a token is 98.1 to 99.5 percent of the time
  per token of `tg64`.
- Each chat run was repeated with `TSR_PIVOT=0`, Tesseract Pivot off. The text was identical in every
  row, the bytes read per token were the same, and the device cycles per token were 1.002 to 1.078
  times as many: Gemma 4 26B-A4B 1.078, Qwen3.8 27B 1.037, Qwen3.5 9B 1.031, Gemma 4 31B 1.012,
  Gemma 4 12B 1.012, Gemma 4 E4B 1.009 and Gemma 4 E2B 1.002. The mixture-of-experts model and the two
  Qwen models gain most, because their layers have matrix products that depend on nothing nearby (the
  dense FFN beside the router, the products beside the Gated DeltaNet recurrence). The dense Gemma 4
  models gain 1.2 percent or less.
- A third chat run of the five Gemma 4 models set `TSR_HEAD_SPLIT=0`, which lowers the logit soft cap of
  the output head over whole rows and not per row block. The text was identical in every row, and the
  device cycles per token were 1.002 to 1.015 times as many: Gemma 4 E2B 1.015, Gemma 4 26B-A4B 1.009,
  Gemma 4 E4B 1.008, Gemma 4 12B 1.003 and Gemma 4 31B 1.002. The output head is a larger share of a
  token in the models with fewer device cycles per token. The Qwen models have no soft cap.
- In the decode launches of the chat runs with Tesseract Pivot on, a channel read 17.3 to
  18.2 GB/s while the MVM unit was busy, and the four channels together 59.3 to 66.8 GB/s over the
  whole token. The dual weight stream is on in every run. The rest of the token is time in which the
  MVM unit reads no weights: vector unit work, waits on results and an empty queue. The figures come
  from `PERF_CH_RD_BEATS`, which counts reads of every kind.
- Each chat run produced 65 tokens of coherent text. The runs do not measure accuracy; see
  [klcheck](klcheck.md) and [numcheck](numcheck.md).

### MTP

Multi-token prediction on the same card and image, in full mode. The target decodes a window of the
draft head's tokens in one launch and rolls back on a rejection: the recurrent state for the Qwen
models, a plain KV removal for Gemma 4, which has no recurrent layers
([software verification](design/06-software-verification.md) §7.11).

| Model | Quantization (target / MTP) | Best K | Plain tok/s | MTP tok/s | Speedup | Accepted per step | tok/s/W |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.8 27B | Q4_K_M / Q8_0 (separate file) | 1 | 3.20 | 4.48 | 1.40x | 0.87 | 0.078 |
| Qwen3.8 2B | Q4_K_M / Q4_K_M (in the file) | 1 | 31.50 | 33.97 | 1.08x | 0.81 | 0.63 |
| Gemma 4 26B-A4B | Q8_0 / Q8_0 (separate file) | 1 | 14.42 | 17.67 | 1.23x | 0.81 | 0.312 |

Device cycles per generated token, in millions, and the tok/s of each K:

| Model | Plain | K = 1 | K = 2 | K = 3 | K = 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.8 27B | 61.87 (3.20) | 43.40 (4.48) | 45.82 (4.23) | 44.20 (4.37) | 63.53 (3.07) |
| Qwen3.8 2B | 6.19 (31.50) | 5.47 (33.97) | 6.43 (28.79) | 6.94 (26.87) | 9.05 (21.00) |
| Gemma 4 26B-A4B | 13.55 (14.42) | 11.02 (17.67) | 11.99 (16.23) | 11.87 (16.36) | 14.96 (13.04) |

Command line, with `TSR_DEVICE=xrt`, the release image, `LLAMA_ATTN_ROT_DISABLE=1` and the default
`tsr-mtp` settings (flash attention, q8_0 KV for target and draft, every layer on the device):

```
tsr-mtp -m TARGET.gguf [-md MTP.gguf] --spec-type draft-mtp --mtp-mode all --mtp-k-list 1,2,3,4 \
  --mtp-device xrt --mtp-prompts PROMPTS.txt -n 128 -c 2048 -t 8
```

- The prompts are three single-turn chat prompts (a hash table, a palindrome function, documentation
  tips) in the model's template with thinking off, 22 to 30 tokens each. Decoding is greedy and every
  run generates 128 tokens per prompt, 384 in all. The Gemma 4 prompts use its template (thinking off).
- The tok/s are wall rates of the decode loop (draft, verify and catch-up) on this host, from one run of
  each K. The plain rate is the same loop with one token per decode, in the same process. The cycles
  are the device counters of the same phases. The accepted figure is drafts accepted per step at the
  best K, so a step yields that plus one tokens.
- Power is the mean card board power (`power1_input`, every 0.2 s) of a second run at the best K, over
  the samples in which the card works: those above the midpoint between the idle power and the run's
  95th percentile. The short prefills between the decode loops are in it. The idle power was 37.6 to
  39.1 W. The tok/s/W column is the rate of that second run over its power: 4.48 tok/s over 57.2 W for
  the 27B, 33.98 tok/s over 53.9 W for the 2B and 17.63 tok/s over 56.5 W for Gemma 4.
- The MTP token stream equals plain greedy decoding in every run: 3 of 3 prompts identical for each K
  of the three models.
- The best K is 1 for all three models. A verify launch of K = 1 has two tokens, so its MVMs read two
  weight streams (T ≤ T_MAX / 2, [rtl-mvm.md](rtl-mvm.md) §15); from K = 2 on they read one. Past the
  best K the verify launch gets longer than the drafts repay: K = 2 to 4 are slower than plain decoding
  on the 2B, and K = 4 on Gemma 4 and on the 27B.
- Gemma 4 has no recurrent layers, so its target keeps no rollback snapshots and the process phase
  costs no launches. The draft head shares the target's KV cache. The target is about 25.6 GiB on the
  card.

### PyTorch backend

One forward of Qwen2.5-0.5B-Instruct over 32 prompt tokens through
`torch.compile(model, backend="tesseract")`, on the same card and image, measured with
`sw/torch/tools/hf_check.py` ([torch-backend.md](torch-backend.md) §6). The model is the Hugging Face
checkpoint (bfloat16, 494 M parameters) run in float32, and its weights are uploaded as Q8_0.

| Mode | Device work of a forward | Launches | Host to device | Device to host | Device cycles | Forward time | KL against eager float32 | Top-1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Matmul mode | 169 matrix products | 169 | 31.6 MB | 58.4 MB | 20.95 M | 0.135 s | 3.6e-3 | 30 of 32 |
| Full mode, `attention="cpu"` | 25 regions and 23 matrix products | 48 | 5.6 MB | 28.3 MB | 23.17 M | 0.147 s | 3.8e-3 | 31 of 32 |
| Full mode | 1 region of 314 ops | 1 | 0.1 MB | 19.4 MB | 26.15 M | 0.148 s | 4.4e-2 | 28 of 32 |

- The launches count every launch of a forward (the counter `rt_execs`): with `attention="cpu"` these
  are the 25 region launches that `hf_check.py` reports and the 23 matrix products between the
  regions. The launches, the host traffic and the forward time are those of the second forward of a
  process, which compiles nothing and uploads no weights or constants. Each mode ran in a second process
  as well, with forward times of 0.135 s, 0.146 s and 0.148 s. The device cycles are the `PERF_CYCLES`
  of one forward.
- In full mode with device attention the activations stay on the device: the host sends the embedded
  rows and reads the logits. The forward is not faster at this size. The norms, the activations, RoPE
  and attention add device cycles, and that is more than the host round trips of matmul mode cost here.
- KL is the mean over the 32 positions of KL(eager float32 || device), and top-1 counts the positions
  whose most likely token is the same. The larger KL of full mode comes from the q8_0 attention. It
  quantizes Q, K and V as q8_0, as the llama.cpp path does with `-ctk q8_0 -ctv q8_0`. Eager PyTorch
  with q8_0 Q, K and V is 4.1e-2 from eager float32 (28 of 32), and full mode is 5.8e-3 from that run
  (30 of 32). With `attention="cpu"` attention stays in float32 on the CPU.
- The logits of the card equal the simulator's bit for bit in all three modes (32 of 32 rows): each
  command ran on the simulator with `--save-logits FILE.npy` and on the card with `--expect FILE.npy`
  ([torch-backend.md](torch-backend.md) §2.3). In full mode every region output of the first forward
  equals the reference executor's bit for bit (`TSR_TORCH_VERIFY`).
- Conditions: torch 2.9.1+cu130, transformers 5.20.2, 8 threads, the model at revision
  `7ae557604adf67be50417f59c2c2f167def9a775` of Qwen/Qwen2.5-0.5B-Instruct, the extension of a build
  with `TSR_WITH_XRT=ON` named by `TSR_TORCH_EXT_DIR`, `TSR_DEVICE=xrt`, the image of Conditions below,
  and a firmware file of the same commit with the image id of Conditions (`d4a0944a97fac583`).

```
hf_check.py MODEL_DIR --forward-only --tokens 32                 # matmul mode
hf_check.py MODEL_DIR --full-mode --attention cpu --tokens 32
hf_check.py MODEL_DIR --full-mode --tokens 32
```

### Conditions

- Host: AMD Ryzen 7 9700X (8 cores, 16 threads), 14.7 GiB RAM, Linux 7.0.0-38-generic (Ubuntu 24.04),
  XRT 2.18.179 (2024.2). `-t 8`. The 1-minute load average at the start of the `llama-bench` runs was
  1.9 to 4.1.
- Card: Alveo U200 with the shell `xilinx_u200_gen3x16_xdma_base_2` (platform 202110.1), PCIe Gen3 x16,
  32 GiB of DDR4 in four channels of 8 GiB (measured at open).
- Image: `tsr_u200_k4.xclbin`, sha256
  `628559646594efde3dc04e40ef79db01f4bc2659d6c5242d3b80ac1b819eb9cc`, UUID
  `c2920184-1184-3844-f252-4c1789b48b2a`. Core clock 200 MHz and RISC-V clock 100 MHz (the monitor's
  fit of the core clock against host time: 199.992 to 199.995 MHz). Tesseract Core 1.1.0 (capability
  1.0), `CAPS` 0x10121 (VU, ATT, GDN, MVM_DUAL), `NUM_CH` 4, `T_MAX` 4, 128 lanes, `K_MAX` 32768.
- Firmware: `tsr-fw.elf`, sha256 `fb553f05a75b189edb122dec5fcc38feabf9a67775b60a5d6c147fdebd9d60e9`,
  Tesseract Basis 1.4.0 (ABI 1), image id `d4a0944a97fac583`, 23 kinds, kind table signature 0x249c.
- Software: this repository at commit bfefda0, Release build with `TSR_WITH_XRT=ON`, llama.cpp
  pinned at e3797b184 (b11810) with `third_party/patches/0001-ggml-tesseract.patch`.
- Mode: full mode (`TSR_KINDS` unset), the dual weight stream on (`TSR_MVM_DUAL` unset), Tesseract Pivot on
  (`TSR_PIVOT` unset), Q8_0 KV cache, flash attention on. The chat run uses greedy sampling.
- Power: the card board power from the U200 sensors (`power1_input` of the hwmon node of the card's
  user function, 0.2 s interval), not the host's power. Idle power with no process on the card,
  as the mean of ten readings before each run: 37.6 W (36.8 to 38.3 W over the seven runs).

Environment of the runs: `TSR_DEVICE=xrt`, `TSR_XCLBIN=<tsr_u200_k4.xclbin>`,
`TSR_IMAGE=build/xrt/fw/tsr-fw.elf`, `LLAMA_ATTN_ROT_DISABLE=1`. The chat run adds `TSR_STATS=1` and
`TSR_MON_PERF_MS=0`. The second chat run of each model adds `TSR_PIVOT=0`, and the third chat run of
the Gemma 4 models adds `TSR_HEAD_SPLIT=0`.

```
build/xrt/bin/llama-bench -m MODEL -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -ub 512 -p 512 -n 64 -r 3 -t 8 -o jsonl

build/xrt/bin/llama-cli -m MODEL -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -c 4096 --temp 0 -n 65 -t 8 \
    -sys "You are a helpful assistant." -p "Write a short story about a robot who learns to paint a sunset." \
    -st -rea off --simple-io --no-display-prompt
```

### Model files

The files are GGUF files as published. The sha256 sums are those of the files that were run.

- Qwen3.8 27B: `Qwen3.8-27B-Q8_0.gguf`, ggml-org/Qwen3.8-27B-GGUF, sha256
  `6bc9e7220d8400574fd62df25f04a961cb5cbb0d766cf0118c21c36c0059c35a`.
- Qwen3.5 9B: `Qwen3.5-9B-Q8_0.gguf`, unsloth/Qwen3.5-9B-GGUF, sha256
  `809626574d0cb43d4becfa56169980da2bb448f2299270f7be443cb89d0a6ae4`.
- Gemma 4 31B: `gemma-4-31B-it-Q4_K_M.gguf`, unsloth/gemma-4-31B-it-GGUF, sha256
  `38bd64c852c4b460434cc7162fa9bdcf242faf86502581a754cb72956bb17f84`.
- Gemma 4 26B-A4B: `google_gemma-4-26B-A4B-it-Q8_0.gguf`, bartowski/google_gemma-4-26B-A4B-it-GGUF, sha256
  `1bb7fdb37fbd542ba972b3eb83ae87752bb9063aff560402e0e2f64235c85226`.
- Gemma 4 12B: `gemma-4-12b-it-Q8_0.gguf`, quantized by Unsloth (GGUF metadata), sha256
  `f20e7ff1be28c283eeeb18fc895733791c56a5851d5cd3fe9691b7f7d12afa72`.
- Gemma 4 E4B: `gemma-4-E4B-it-Q8_0.gguf`, quantized by Unsloth (GGUF metadata), sha256
  `f8854aa4480df62585a279e7ca0a881554fc18a41c59c4f62642d16a2ae47012`.
- Gemma 4 E2B: `gemma-4-E2B-it-Q8_0.gguf`, unsloth/gemma-4-E2B-it-GGUF, sha256
  `605d3c2647d7c58c1e4b5375ccb5702acf94c2611b4c8d4877812f8fdd32d053`.
