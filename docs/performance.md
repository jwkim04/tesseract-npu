# Performance

This page says how to measure Tesseract and what to report with a result. Results are published
with each release.

## What is measured

- Decode rate: tokens per second of the generation phase, from the `eval time` line of llama.cpp.
  Batch 1, greedy sampling (`--temp 0`).
- Prefill rate: tokens per second of prompt processing (`prompt eval time`) at a stated ubatch
  size (`-ub`).
- Device cycles per token: the `PERF_CYCLES` counter over a run, divided by the number of tokens.
- Unit busy cycles per token: `MVM_CYCLES`, `VU_CYCLES`, `DMA_CYCLES` and `QACT_CYCLES`, and the
  stall counters `MVM_WAIT` (idle while the queue holds an entry that cannot start) and `MVM_EMPTY`
  (idle with an empty queue, which means issue bound).
- DDR traffic: `CH_RD_BEATS` per channel (64 B per beat) and `CH_WAIT`.

Accuracy is a separate question. Check a configuration with [klcheck](klcheck.md) and
[numcheck](numcheck.md) before quoting its speed.

## Tools

| Tool | Use |
| --- | --- |
| `llama-completion` | decode and prefill rates; with `TSR_STATS=1` it also prints the backend report and the PERF counter deltas |
| `tsr-mtp` | speculative decoding with a model's MTP head; device cycles per phase for a list of draft lengths |
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
| `TSR_GDN_CELL=0`, `TSR_HEAD_SPLIT=0` | switch off a host lowering choice, for an A/B comparison |

Full mode and matmul mode are defined in [design/README.md](design/README.md).

## Procedure

Decode. Run `llama-completion -m MODEL -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -c 4096 --temp 0 -no-cnv
-n 65` with a prompt of a few tokens, `LLAMA_ATTN_ROT_DISABLE=1` and `TSR_STATS=1`. Run it again
with `-n 1`. The per-token device figures are the difference of the two counter totals divided by
64. The rate is the `eval time` rate of the `-n 65` run. Repeat at least three times.

Prefill. Process one prompt of a stated length with `-n 1` at a stated `-ub`. The default `-ub` is
512. Attention for a ubatch of more than 680 tokens runs on the CPU.

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

The weight stream sets a floor. A decode token reads every weight once. A cluster takes at most one
64-byte beat per core clock from its channel, so the cycles per token cannot fall below the largest
per-channel weight volume divided by 64 bytes. At 200 MHz that is 12.8 GB/s per channel. The DDR
utilization U, which is the bytes read per token times tokens per second over the measured
bandwidth of the channels, compares a run with that floor ([spec](spec.md) §14, [prof-dbg.md](prof-dbg.md) §3).

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
- Numbers: the number of runs, the median and the range, the device cycles per token, the unit busy
  cycles, the device share of the wall time and the load average at the start.
- Accuracy: the klcheck ratio and top-1 for the same configuration, and the bit identity against the
  simulator where it was checked.
- For a simulator figure: the cycle model configuration.
