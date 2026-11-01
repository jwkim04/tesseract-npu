# Getting started

How to build Tesseract, run a model on the simulator and on PyTorch, serve it over HTTP, verify the
result, and find your way around the repository. Every command names a target, tool or script of
the current tree.

> **Scope.** User and developer documentation. Status words: *on the card* means it ran on the
> Alveo U200; *on the simulator* means this tree runs it against the simulator
> (`TSR_DEVICE=sim`); *designed* and *planned* are as in
> [`docs/design/README.md`](design/README.md) §10. The control-core RTL and its module/integration
> tests exist. The corrected full K4 image has not reached card validation; older card results
> predate the control core. Current status is in [handoff.md](handoff.md). Normative for the host stack:
> [SW](design/06-software-verification.md) §6, §9, §10 and [EX](design/01-execution-model.md) §10.

## Requirements

- **Linux**, developed on Ubuntu 24.04. **CMake 3.20 or newer**; use 3.28 or newer for a clean
  `ctest`: with 3.20 to 3.27, llama.cpp's own model tests (`test-llama-context`,
  `test-llama-context-model`, `test-thread-safety`) do not get `TSR_DEVICE=none` and run against
  the Tesseract device. A **C11 and C++17 compiler** (GCC 13 on the development host).
- **The RISC-V toolchain is required.** The firmware of the control core is built with a pinned
  bare-metal GCC, and `cmake` stops at configure time without it. Install it in user space, without
  sudo:

  ```sh
  sw/ctrl/tools/install-toolchain.sh     # downloads and verifies the pins; installs under build/tools/riscv
  ```

  The pins are in [`docs/design/toolchain.lock`](design/toolchain.lock): xPack `riscv-none-elf-gcc`
  14.2.0-3 (compiler, binutils, newlib, GDB), dtc 1.8.1, Spike (`5a76066`), riscv-tests
  (`3df703c`) and riscv-arch-test 3.10.0. `--steps xpack,check` installs the compiler alone; the
  Spike-based `rv_*` tests then report "Skipped". For another location pass
  `-DTSR_RISCV_TOOLS=<prefix of install-toolchain.sh>` or
  `-DTSR_RISCV_PREFIX=<dir>/bin/riscv-none-elf-` at configure time (and `-DTSR_SPIKE=<spike>`).
  Background: [SW](design/06-software-verification.md) §6.1, §9.2, §9.6.
- **Python 3 with numpy**, used by `ctest` (the numpy models of the VU math and of numcheck), by
  `numcheck.py` and by the cocotb tests. The interpreter for the `ref_math_np_*` and `ref_vu_np`
  comparisons is `-DTSR_REF_MATH_PYTHON=<interpreter>` (default: `build/tools/torchvenv/bin/python`
  if it exists, else the `python3` CMake finds; configure warns if it has no numpy).
- **For the RTL tests** (optional): Verilator 5, cocotb and pytest. The development host runs
  Verilator 5.046, cocotb 2.0.1 and pytest 9.0.2.
- **For bitstream builds** (optional; k1/k4 and the separate M0 bandwidth build are enabled):
  Vivado and Vitis 2024.2 with the U200
  development platform `xilinx_u200_gen3x16_xdma_2_202110_1` for `v++` links.
- **For the card** (optional): XRT 2.18.179 and, on kernels newer than 6.8, the patched drivers of
  [`sw/xrt-driver`](../sw/xrt-driver/README.md).
- **For the PyTorch backend** (optional): a venv with torch, transformers, safetensors and pytest
  (`build/tools/torchvenv`, set up as in [`docs/torch-backend.md`](torch-backend.md) §2.1).

## Build and test

llama.cpp is a git submodule pinned to commit `e3797b18` (see [`docs/llama-pin.md`](llama-pin.md)).
The backend hooks into it through the patch in `third_party/patches/`; the patch script is
idempotent and also takes `--status` and `--reverse`.

```sh
git submodule update --init
third_party/apply-patches.sh              # ggml-tesseract hooks into the pinned llama.cpp
sw/ctrl/tools/install-toolchain.sh        # once, see Requirements
cmake -S . -B build/dev -DCMAKE_BUILD_TYPE=Release
cmake --build build/dev -j4
ctest --test-dir build/dev -LE slow
```

- `-j4` is the setting of the development host (14 GB RAM, 16 cores). Use `-j4` or fewer on a 14 GB
  host.
- The build compiles the firmware project `sw/ctrl` as the CMake external project `tsr_fw` (RISC-V
  compiler, its own CMake project) and puts the production image at `build/dev/fw/tsr-fw.elf`. The
  host libraries find it through a path compiled in at configure time; `TSR_IMAGE` overrides it.
- `ctest` labels: `fw` (firmware and executor tests), `rv` (the RV32 interpreter against Spike;
  skipped until Spike is installed), `model` (tests that need a model file: ours are registered
  only when the GGUF file is in `models/`, llama.cpp's download a small one and need network access
  the first time), `slow` (opt-in, left out by `-LE slow`; run them with `-L slow`) and `main` and
  `python` (llama.cpp's own tests).
- A narrower build for the backend, the tools and their tests. The firmware image is not a
  dependency of those targets, so name `tsr_fw`:

  ```sh
  cmake --build build/dev -j4 --target tsr_fw llama-completion klcheck tsr-selftest tsr-numcheck \
        test-backend-ops test-ggml-tesseract test-tsr-backend-units
  build/dev/bin/test-tsr-backend-units                  # allocator, lowering, op-list builder
  build/dev/bin/test-ggml-tesseract                     # bit-exact against tsr_ref, NMSE against the CPU
  TSR_SIM_TMAX=8 build/dev/bin/test-ggml-tesseract      # token tiling with T > 1
  ctest --test-dir build/dev -R '^ref_' --output-on-failure
  TSR_DEVICE=sim build/dev/bin/test-backend-ops -o MUL_MAT -b Tesseract0
  ```

- **Known flake.** One `MUL_MAT` case of `test-backend-ops`, `type_a=f32,type_b=f32,m=1,n=1,k=2048`,
  fails the NMSE limit in about 15 % of random runs, so an unfiltered `test-backend-ops` run
  reports 1402 of 1403 once in about seven runs. The case has a single output, so its NMSE is one
  squared relative error, and the exact dot product of the same 8-bit operands fails on the same
  runs. `ctest` leaves it out and checks it bit-exactly against the reference.

## Run a model on the simulator

Model files are not part of the repository: put GGUF files in `models/` (ignored by git). The
first command runs the M7 settings of the software path: the whole decode token is one launch of the
firmware's `graph` kernel on the Tesseract device, with the KV cache and attention on the device.

```sh
LLAMA_ATTN_ROT_DISABLE=1 build/dev/bin/llama-completion -m models/qwen2.5-0.5b-instruct-q8_0.gguf \
    -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -no-cnv -p "The capital of France is" -n 16
build/dev/bin/klcheck all -m models/qwen2.5-0.5b-instruct-q8_0.gguf -o build/klcheck-05b --phase m7 -n 64
```

The M6 settings run the weight `MUL_MAT`s only on the device, with attention and the KV cache on the
CPU:

```sh
LLAMA_ATTN_ROT_DISABLE=1 TSR_KINDS=m6 build/dev/bin/llama-completion -m models/qwen2.5-0.5b-instruct-q8_0.gguf \
    -ngl 99 -nkvo -fa on -ctk q8_0 -ctv q8_0 -no-cnv -p "The capital of France is" -n 16
build/dev/bin/klcheck all -m models/qwen2.5-0.5b-instruct-q8_0.gguf -o build/klcheck-05b -n 64
```

- `-ngl 99` offloads every layer to `Tesseract0`. In M7 that is the whole decode graph, one launch
  per token; in M6 only the weight matmuls.
- M7 is the default of the backend (`TSR_KINDS` unset: the device takes every op kind it supports) and
  needs no `-nkvo`. The KV cache lives in Tesseract buffers and the device writes and reads it.
  M6 needs `TSR_KINDS=m6` and `-nkvo`: the KV cache and attention stay on the CPU.
  A kind list such as `TSR_KINDS=all,-RMS_NORM` moves single kinds back to the CPU; how a list is
  read is in [`sw/ggml-tesseract/README.md`](../sw/ggml-tesseract/README.md).
- `-fa on -ctk q8_0 -ctv q8_0` are required with the KV cache on the device: an f16 cache cannot live
  in device memory. With `-nkvo` they are the baseline settings. `LLAMA_ATTN_ROT_DISABLE=1` (no
  Hadamard KV rotation) keeps a decode token to one launch; without it the run works with more
  launches. `klcheck` and `tsr-numcheck` set it themselves. Results with other settings are not
  comparable with the baselines.
- `-no-cnv` turns off the chat mode that `llama-completion` starts by itself for models with a chat
  template.
- `klcheck all` runs the device, the CPU, the 1-ulp floors and the embedding-path check, and prints
  the verdict (`docs/klcheck.md`). `--phase m6` (the default) and `--phase m7` select the settings
  above; it uses 8 built-in prompts, `-n` generated tokens each and three floor seeds.
- The simulator is slow. On the 16-core development host, 0.5B decodes at about 90 ms per token with
  M7 (one launch per token) and at about 92 ms with M6 (145 launches per token; measured on the
  simulator, `83ac601`), and 32B at 7 to 8 s per token. `TSR_SIM_THREADS` sets the number of MVM
  threads (default min(CPUs, 8)). A 32B model needs file-backed simulator DRAM,
  `TSR_SIM_DRAM_DIR=<dir>` with about 20 GiB free.
- Models checked on the simulator with klcheck: Qwen2.5-0.5B-Instruct Q8_0, Qwen2.5-32B-Instruct
  Q4_K_M and Gemma 4 12B Q4_K_M ([`docs/klcheck.md`](klcheck.md); Qwen2.5 also with
  [`docs/numcheck.md`](numcheck.md)). The Gemma 4 runs of [`docs/performance.md`](performance.md)
  were made on the card.
- Without a GGUF file, `build/dev/bin/tsr-selftest --device sim` still boots the firmware on the
  simulator and runs its self-checks (see [Verify](#verify)).

## Serving

`llama-server` runs on the Tesseract backend from the same build: one device launch per decode
step, the KV cache on the device. It runs on the simulator today; no card has this design yet.

```sh
LLAMA_ATTN_ROT_DISABLE=1 TSR_DEVICE=sim build/dev/bin/llama-server \
    -m models/qwen2.5-0.5b-instruct-q8_0.gguf -ngl 99 -fa on -ctk q8_0 -ctv q8_0 \
    -np 4 -kvu -c 4096 -t 4 --host 127.0.0.1 --port 8080
```

- `-ngl 99 -fa on -ctk q8_0 -ctv q8_0` are required. An f16 KV cache cannot live in device memory:
  the server stops at startup and the backend's message names these flags (or `-nkvo`).
- `LLAMA_ATTN_ROT_DISABLE=1` gives one launch per token. Without it the server works, with more
  launches per token (9 on the tiny test model).
- `-np N` with N above 1 needs `-kvu` (`--kv-unified`): the device holds one KV stream. With `-np`
  left out the server picks 4 slots with a unified cache.
- Do not pass `--context-shift`: llama.cpp crashes in `build_rope_shift` with a quantized K cache
  and no rotation, on the CPU backend too. Without it a full context ends the request with
  `finish_reason` `length` and a too-long prompt gets HTTP 400.
- `TSR_DEVICE` picks the transport (default `sim`). `TSR_STATS=1` prints the launches and device
  cycles at exit.

The endpoints are `/health`, `/v1/models`, `/v1/chat/completions` (also with `"stream": true`) and
`/v1/completions`. One request:

```sh
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"messages": [{"role": "user", "content": "Name the capital of France."}], "max_tokens": 16, "temperature": 0}'
```

On the simulator, Qwen2.5-0.5B Q8_0 decodes at 90 to 130 ms per token depending on host load; the
device needs 2.5 M cycles (9 ms at 276.3 MHz). Four concurrent requests at `-np 4` get about
1.7 tokens/s each: at the default `TSR_SIM_TMAX=1` the device gains nothing from batching. The text
is not bit-identical to the CPU backend (the model-level check is `klcheck`,
[`docs/klcheck.md`](klcheck.md)): greedy output leaves the CPU's at high-entropy tokens, and with
several slots it can also differ from a single-slot run at near-ties, because the rounding in flash
attention depends on how the slots' cells interleave in the shared cache.

## Run a PyTorch model

`torch.compile(model, backend="tesseract")` runs the weight matmuls of a PyTorch model on the
device and everything else eagerly on the CPU (`import tesseract_torch` registers the backend).
It uses the same transport (`TSR_DEVICE`) and the same numerics as the llama.cpp path.

```sh
cmake -S . -B build/torch -DCMAKE_BUILD_TYPE=Release -DTSR_WITH_TORCH=ON
cmake --build build/torch -j4 --target tesseract_torch_ext
export PYTHONPATH=$PWD/sw/torch:$PWD/build/torch/python
build/tools/torchvenv/bin/python -m pytest -q sw/torch/tests
build/tools/torchvenv/bin/python -I sw/torch/tools/hf_check.py models/hf/Qwen2.5-0.5B-Instruct
```

`-DTSR_TORCH_PYTHON=<interpreter>` selects the Python with torch (default
`build/tools/torchvenv/bin/python`). `hf_check.py` compares a Hugging Face model through the
backend with eager PyTorch; `--download Qwen/Qwen2.5-0.5B-Instruct` fills the model directory from
the Hub. Architecture, tests and results: [`docs/torch-backend.md`](torch-backend.md),
[`sw/torch/README.md`](../sw/torch/README.md), [`sw/rt/README.md`](../sw/rt/README.md).

In this tree only the C++ runtime tests have run (`ctest --test-dir build/dev -R '^rt_'`). The
Python suite and `hf_check.py` are expected to work with the commands above, and their re-run is
pending (last results: `docs/torch-backend.md`).

## Run a model on the card

The card runs again once the bitstream with the control core is built (work package W6,
[`docs/design/README.md`](design/README.md) §10); the `xrt` transport (`-DTSR_WITH_XRT=ON`) is written and
unit-tested against the XRT no-op shim and has not run on a card. Card measurements were taken
before the control core was added, with the host issuing the ops, and are in
[`docs/performance.md`](performance.md).

## Verify

What each check proves and how to run it. The oracle chain behind them is in
[`docs/spec.md`](spec.md) §17 (Verification strategy); the simulator and its tests are described in
[`sw/sim/README.md`](../sw/sim/README.md), the firmware tests in
[`sw/ctrl/README.md`](../sw/ctrl/README.md).

| Check | What it proves | Command |
| --- | --- | --- |
| Software tests | The reference library against ggml (exhaustive fp16/f32, QACT, dot products, layouts), the simulator, the RV32 interpreter, the firmware and its host build, the transport, the backend allocator and op-list builder, an end-to-end backend test bit-exact against `tsr_ref` and within NMSE 5e-4 of the CPU | `ctest --test-dir build/dev -LE slow` |
| Op level | Every supported `MUL_MAT` shape within NMSE 5e-4 of the ggml CPU backend | `build/dev/bin/test-backend-ops -o MUL_MAT -b Tesseract0` |
| Model level | Mean KL against the CPU at most 3 times the 1-ulp floor, top-1 agreement at least 80 % ([`docs/klcheck.md`](klcheck.md)) | `build/dev/bin/klcheck all -m MODEL -o DIR -n 64` |
| Numerics | Every output element of every weight `MUL_MAT` of one prompt is bit-exact against `libtsr_ref` and an independent numpy model, and within a proven fp32 error bound ([`docs/numcheck.md`](numcheck.md)) | `build/dev/bin/tsr-numcheck all -m MODEL -o DIR -T 32 --checks full`, then `python3 sw/tools/numcheck.py --build build/dev compare --cpu DIR/cpu --floor DIR/floor --dev DIR/dev --dev-noobs DIR/dev-noobs -o DIR/report` |
| Firmware and transport | The firmware image boots with receive CRC and VERIFY, `PING`, the `selftest` kernel (ids 1 to 10 and five misuses) agrees bit for bit with a shadow simulator, trace replays through the device, the log decodes | `build/dev/bin/tsr-selftest --device sim` (also `--device mock`, `--debug`, `--traces DIR`) |
| Determinism | A trace replays bit-identically under the fixed and random schedules with the race detector fatal | `build/dev/bin/tsr-trace-replay -q --sched fixed --sched random:1 traces/m3` |
| Race detector | The RISC-V firmware tests run under 8 schedules with the detector fatal, and mutants without a dependency must be caught | `ctest --test-dir build/dev -R '^fw_kernels_rv'`, `TSR_SIM_RACE=fatal` in any simulator run |
| RV32 interpreter | The interpreter runs the riscv-tests, riscv-arch-test and random programs retire by retire against Spike | `ctest --test-dir build/dev -L rv --output-on-failure` (needs Spike) |
| Tables | The generated CSR, opcode, error and PERF tables match their source `sw/tables/tsr_tables.py` | `python3 sw/tables/gen_tables.py --check` |
| RTL lint | Verilator `-Wall` clean on every RTL source and the Xilinx and platform variants | `rtl/lint.sh` |
| RTL unit tests | The cocotb benches of the datapath units against the shared reference functions | `tb/run_all.sh` (lint plus every bench), `tb/run_all.sh --no-lint -k dma`, `tb/run_all.sh --no-lint common` |

Notes:

- `numcheck.py compare` needs the native libraries of the build: `--build build/dev` points at
  `libtsr_ref.so` and `libnumcheck_helper.so` (target `numcheck_helper` and `tsr_ref_shared`).
- `tb/run_all.sh` runs `python3 -m pytest`: put a `python3` with cocotb, pytest and numpy first on
  `PATH`. It builds with `make -j${TSR_BUILD_JOBS:-4}` under `build/cocotb`; the full run takes
  about 25 minutes with cold Verilator builds.
- The opt-in slow switches: `TSR_REF_MATH_FULL=1` (all 2^32 inputs of the VU math,
  `ctest -L slow -R '^ref_math_'`), `TSR_FP_SLOW=1` (the fp sweeps of `tb/fp`),
  `TSR_PLATFORM_SLOW=1` (the mutant test of the M0 kernel), `TSR_RV_RANDOM_LONG=1` (2000 random
  RV32 programs). `ctest -L slow` runs the slow tests that need no switch.
- **Trace sets.** The trace data is not in git: only the generators (`traces/<set>/regen.sh`),
  the helper `traces/trace_tools.sh` and `traces/MANIFEST.sha256` are. Regenerate a set and compare its
  digest with the manifest:

  ```sh
  traces/gen/regen.sh build/dev      # regenerate one set, then replay it under 3 schedules
  cd traces && find gen -type f ! -name regen.sh -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum
  ```

  The sets are `m2`, `gen`, `m3`, `vu`, `at`, `ctrl` and `model-v1conv`, each with its own
  `regen.sh` (it replaces the set, then replays every trace under the fixed schedule and two random
  ones with the race detector fatal, tile-only and through the replay kernel). The second command
  prints the digest to compare with the line of the set in `traces/MANIFEST.sha256`. Generation is
  deterministic (fixed seeds), but the digests are those of the development host:
  the generator draws some random inputs through libm. Sets, formats and sizes:
  [`traces/README.md`](../traces/README.md).
- **Not runnable in this tree** (they return with the RTL work packages W4 and W5): the
  Verilated U200 kernel (`TSR_DEVICE=vkernel` is "not available until P5.3"), `tb/fastreplay/run.py`
  and the cocotb core replays, and hardware emulation of k1. Their results are in
  [`docs/performance.md`](performance.md).

## Environment variables

Variables are read when the device or the tool opens; a malformed value makes the open fail, and
is never silently ignored. Component documents hold the complete lists for their own variables.

| Variable | Effect | Defined in |
| --- | --- | --- |
| `TSR_DEVICE` | Transport kind: `sim` (default), `mock` (the `ap_ctrl_none` adapter register model around the simulator), `vkernel` (the Verilated kernel; not available until P5.3), `xrt` (the card; needs `-DTSR_WITH_XRT=ON`). `none` registers no ggml device | `sw/transport/tsr_transport.c`, `sw/ggml-tesseract/ggml-tesseract.cpp` |
| `TSR_IMAGE` | Firmware ELF to boot. Default: the build's `tsr-fw.elf` | `sw/transport/tsr_image.cpp` |
| `TSR_LAUNCH_TIMEOUT_MS`, `TSR_ABORT_MS`, `TSR_BOOT_MS`, `TSR_RESET_MS` | Transport timeouts in ms: launch wait (60000; 0 = none), cancel (10000), boot and VERIFY (2000), SOFT_RESET (15000) | `sw/transport/tsr_kdev.cpp` |
| `TSR_ATTACH=1` | Open without SOFT_RESET or bind, for diagnosis | `sw/transport/tsr_kdev.cpp` |
| `TSR_XCLBIN`, `TSR_XRT_DEVICE`, `TSR_XRT_KERNEL`, `TSR_XRT_CH_GIB`, `TSR_XRT_DMA` (`direct`, `staged`, `m2m`), `TSR_XRT_DMA_CHUNK_MIB`, `TSR_XRT_DMA_ALIGN` | Options of the `xrt` transport | `sw/transport/tsr_transport_xrt.cpp`, [`sw/transport/README.md`](../sw/transport/README.md) |
| `TSR_MOCK_NUM_CH`, `TSR_MOCK_TMAX`, `TSR_MOCK_CH_GIB`, `TSR_MOCK_DMA_*`, `TSR_MOCK_INJECT`, `TSR_MOCK_TCM_FAULT`, `TSR_MOCK_HANG` | Configuration and fault injection of the `mock` transport | `sw/transport/tsr_kdev_mock.cpp` |
| `TSR_VKERNEL_SERVER`, `TSR_VKERNEL_SOCKET` | Server command line or socket of the `vkernel` transport | `sw/transport/tsr_kdev_vkernel.cpp` |
| `TSR_LOG` | `stderr` or a file: decode the device log there (appended) | `sw/transport/tsr_kdev.cpp` |
| `TSR_LOG_LEVEL` | Device log level 0 (FATAL) to 5 (TRACE); default 3 | `sw/transport/tsr_kdev.cpp` |
| `TSR_SIM_TMAX` | Simulated `T_MAX`, 1 to 8 (default 1) | `sw/transport/tsr_kdev_sim.cpp` |
| `TSR_SIM_CH_GIB` | Simulated GiB per channel, 1 to 16 (default 16) | `sw/transport/tsr_kdev_sim.cpp` |
| `TSR_SIM_THREADS` | Simulator MVM threads, 1 to 64 (default min(CPUs, 8)); results do not depend on it | `sw/transport/tsr_kdev_sim.cpp` |
| `TSR_SIM_DRAM_DIR` | Back the simulator's DRAM with sparse temporary files in this directory (needed for 32B models) | `sw/transport/tsr_kdev_sim.cpp` |
| `TSR_SIM_SCHED` | `fixed` (default) or `random:<seed>[,<min>,<max>]`: the scheduler's time model | `sw/sim/tsr_sim.c` |
| `TSR_SIM_RACE` | Race detector: `off` (default), `report` or `fatal` | `sw/sim/tsr_sim.c` |
| `TSR_SIM_LOG` | `stderr` or a file; the prefix `rv,` adds one line per retired RISC-V instruction | `sw/sim/tsr_sim.c` |
| `TSR_SIM_RV_CORE`, `TSR_SIM_RV_RATIO` | Ibex cycle table `small` or `maxperf` (default); `clk_rv` over core clock, 0.05 to 4 (default 166.7 / 276.3) | `sw/sim/tsr_sim.c` |
| `TSR_TRACE` | Directory: dump every launch as a trace directory; several processes never overwrite each other's launches | `sw/sim/tsr_sim.c` |
| `TSR_DUMP_DIR` | Directory for hang dumps (recovery ladder level 0) | `sw/transport/tsr_kdev.cpp` |
| `TSR_GDB` | `tcp:<port>` or `unix:<path>`: start the embedded GDB service in an executable linked with `tsr_dbg_embed` | `sw/tools/dbg/embed.cpp` |
| `TSR_DEBUGGER` | `sock:<path>`: start the embedded monitor-command socket | `sw/tools/dbg/embed.cpp` |
| `TSR_GDB_WAIT` | `boot` or `launch`: with the embedded service, halt at boot or kernel entry; retain a trapped device instead of restarting it | `sw/transport/tsr_kdev.cpp`, `sw/tools/dbg/embed.cpp` |
| `TSR_TILE_SERIAL`, `TSR_TILE_HIST` | `1` sets and `0` clears tile-level serial mode and the dispatch history in `DBG_CFG` at open | `sw/transport/tsr_kdev.cpp` |
| `TSR_STATS=1` | Backend timing and device counters (`PERF_*`) on stderr; the transport's statistics at close | `sw/ggml-tesseract/ggml-tesseract.cpp`, `sw/transport/tsr_kdev.cpp` |
| `TSR_DEBUG` | Backend log: `1` placement, op lists and launches; `2` also buffer events and every launched record | `sw/ggml-tesseract/ggml-tesseract.cpp` |
| `TSR_FORCE_HOST=1` | Test knob: every non-weight buffer in host memory, so every `MUL_MAT` takes the staged path | `sw/ggml-tesseract/ggml-tesseract.cpp` |
| `TSR_STAGE_KIB` | Test knob: size of the IO stage area in KiB | `sw/ggml-tesseract/ggml-tesseract.cpp` |
| `TSR_LOAD_THREADS` | Worker threads for weight conversion and repack at load (default the online CPUs, at most 16) | `sw/ggml-tesseract/ggml-tesseract.cpp` |
| `TSR_OPL_CRC=1` | Ask the firmware to verify the CRC of every op list it executes | `sw/ggml-tesseract/ggml-tesseract.cpp` |
| `TSR_RT_DEBUG` | `tsr_rt`: `1` uploads and op lists, `2` also every record | `sw/rt/tsr_rt.cpp` |
| `TSR_TORCH_DEBUG` | PyTorch backend: `1` partition and counters, `2` also every device call, `3` also the rewritten aten graphs | `sw/torch/tesseract_torch/runtime.py` |
| `TSR_TORCH_VERIFY=1` | PyTorch backend: check every device matmul bit for bit against `libtsr_ref` | `sw/torch/tesseract_torch/runtime.py` |
| `TSR_TORCH_EXT_DIR` | Where to find the extension `_tesseract_rt` if it is not on `sys.path` | `sw/torch/tesseract_torch/_ext.py` |
| `LLAMA_ATTN_ROT_DISABLE=1` | llama.cpp: no Hadamard rotation of the KV cache (a baseline setting) | llama.cpp; read by `klcheck` and `tsr-numcheck` |
| `GGML_SCHED_DEBUG=1` or `2` | llama.cpp's scheduler log (use with `-v`) | llama.cpp |

GDB on the C simulator is implemented and verified with the pinned RISC-V GDB; see
[`prof-dbg.md`](prof-dbg.md) §1. `TSR_GDB` / `TSR_DEBUGGER` require the owning executable's
embedded-service registration. Automatic transport profiling through `TSR_PROF*` remains
inactive; use the implemented `tsr-prof trw` / `stats` commands explicitly. `TSR_DEBUG` and `TSR_FORCE_HOST` are read as
integers, so a non-numeric value counts as 0. Test-only variables (`TSR_FW_PROF`,
`TSR_FW_E2E_KEEP`, `TSR_M7L_MUTATE`, `TSR_SELFTEST_MUTATE_SHADOW`, `TSR_BW_MOCK_FAULT`) and the
build knobs of the testbenches (`TSR_BUILD_JOBS`, `TSR_SEED`, `TSR_SIM_BUILD_ROOT`, `TSR_VIVADO`)
are described in the files that read them.

## Repository layout

```
tesseract/
├── docs/                    documents (index below); design/ is the normative design set,
│                            fig/ the figures and their scripts, logo/ the logo
├── sw/
│   ├── sim/                 simulator and references: libtsr_ref (bit-exact numerics, VU and AT
│   │                        references), libtsr_sim (CSR and queue model, RV32 interpreter,
│   │                        units, scheduler, race detector, trace format, debug models), tests
│   ├── ctrl/                firmware SDK for the control core: runtime, header library, graph
│   │                        executor, kernels, tsr-elf, host build of the kernels, tests
│   ├── transport/           tsr_transport: the protocol layer and the kinds sim, mock,
│   │                        vkernel and xrt
│   ├── ggml-tesseract/      ggml backend: placement, repacking, op-list builder and cache, tests
│   ├── rt/                  tsr_rt: ggml-free runtime (weight upload, linear, op lists)
│   ├── torch/               PyTorch backend: torch.compile backend "tesseract", extension, tests
│   ├── tools/               klcheck, tsr-numcheck and numcheck.py, tsr-selftest, tsr-trace-gen,
│   │                        tsr-trace-replay, tsr-trace-conv, tsr-log, tsr_bw (M0 tool)
│   ├── tables/              single source of the CSR map, opcodes, error codes, PERF slots and
│   │                        formats; generator for the SystemVerilog package, C headers and benches
│   └── xrt-driver/          patches to build the XRT 2.18.179 PCIe drivers on kernels 6.8 to 7.0
├── rtl/
│   ├── core/                21 vendor-independent SystemVerilog files: the datapath units (DMA,
│   │                        QACT, MVM clusters, SPM, memory arbiter, FP units) and the CSR block
│   ├── tech/generic/        behavioural tsr_sram_1r1w and tsr_mac_lane (simulation, lint)
│   ├── tech/xilinx/         URAM, BRAM and DSP48E2 wrappers (two int8 tokens per DSP)
│   ├── platform/alveo_xrt/  XRT adapter, AXI4 memory bridges, kernel tops, kernel XML, .xo
│   │                        packaging; bw/ holds the M0 bandwidth kernel
│   └── filelist.f, filelist_xilinx.f, lint.sh
├── tb/                      cocotb and Verilator benches per block, run_all.sh; tech/ (Xilinx
│                            wrappers against the generic ones), platform/ (adapter, M0 bench)
├── traces/                  trace-set generators and the manifest (the data is not in git)
├── hw/
│   ├── u200/                v++ configurations, floorplan C', build_xclbin.sh, the M0 kernel build
│   └── viz/                 floorplan images and terminal-video tools
└── third_party/             llama.cpp submodule (pinned), patches/, apply-patches.sh
```

The control core, the issue path, the vector unit and the debug block are designed
([`docs/design`](design/README.md)) and exist in the simulator and the firmware, not in `rtl/core` yet. The
device runner of milestone M9 (`sw/runner`) is planned and not in the tree.

### Documentation index

**Start.** This file; [`performance.md`](performance.md) (every measurement with its provenance, the
estimates, the measurement history).

**Design.** [`spec.md`](spec.md) (the master specification: architecture, layouts, numerics, ISA,
interfaces, verification strategy, milestones, open items); [`design/README.md`](design/README.md) (the
design set, M6 to M9: execution model, vector unit, attention and KV cache, control core, debug,
software and verification); [`fig/`](fig) (the figures and the scripts that draw them);
[`versioning.md`](versioning.md) (Tesseract Core capability, Basis version, ABI and kind revisions) and
[`../CHANGELOG.md`](../CHANGELOG.md).

**Hardware and platform.** [`rtl-m2.md`](rtl-m2.md) and [`rtl-m3.md`](rtl-m3.md) (RTL contract of the
core infrastructure; the MVM unit with its numerics, throughput and Vivado results);
[`platform-u200.md`](platform-u200.md) (the U200 adapter, kernels, xclbin build, floorplan, bring-up
runbook and card results); [`m0-bw.md`](m0-bw.md) (the M0 bandwidth kernel and tool);
[`../sw/xrt-driver/README.md`](../sw/xrt-driver/README.md) (XRT 2.18.179 driver patches for kernels
6.8 to 7.0); [`../hw/viz/README.md`](../hw/viz/README.md) (floorplan images and terminal videos).

**Software and tools.** [`../sw/ctrl/README.md`](../sw/ctrl/README.md) (the firmware SDK);
[`../sw/sim/README.md`](../sw/sim/README.md) (the simulator and the references);
[`../sw/transport/README.md`](../sw/transport/README.md) and [`xrt-host.md`](xrt-host.md) (the
transport); [`../sw/ggml-tesseract/README.md`](../sw/ggml-tesseract/README.md) and
[`llama-pin.md`](llama-pin.md) (the llama.cpp backend and the pinned llama.cpp);
[`torch-backend.md`](torch-backend.md), [`../sw/torch/README.md`](../sw/torch/README.md) and
[`../sw/rt/README.md`](../sw/rt/README.md) (the PyTorch backend and its runtime);
[`klcheck.md`](klcheck.md), [`numcheck.md`](numcheck.md) and
[`../traces/README.md`](../traces/README.md) (model-level and numeric verification; trace sets).

**Records.** [`ctrl-riscv.md`](ctrl-riscv.md) (design record of the control core);
[`prof-dbg.md`](prof-dbg.md) (profiling and debugging).

## Working rules

1. **The specification first.** [`docs/spec.md`](spec.md) is the master specification and
   [`docs/design`](design/README.md) its normative design set. A change to the ISA, the memory layout, the
   numerics or a table goes into the same commit as the matching edit of the specification (and of
   `sw/tables/tsr_tables.py` for tables); if code and specification disagree, fix the specification
   first. The order of authority is stated in `docs/spec.md` §1.
2. **Bit-exact replay.** The simulator is the oracle. An RTL change merges only when the relevant
   traces replay bit-exactly, and the firmware's host build gives the same tile-op stream as its
   RISC-V build.
3. **One reference.** Repacking, QA quantization, dot products and the approximate functions
   (VU, AT) exist once, in `sw/sim`, shared by the backend, the simulator, the firmware tests and
   the cocotb tests.
4. **The pinned llama.cpp.** Use only the pinned commit. Changes to llama.cpp go into
   `third_party/patches/`; verified API facts are recorded in [`docs/llama-pin.md`](llama-pin.md).
5. **Reserved fields are zero.** The simulator and the hardware refuse an instruction or a
   descriptor whose reserved bits are set.
6. **RTL rules.** SystemVerilog, `rtl/lint.sh` clean (targeted waivers only), a cocotb test per
   module, vendor primitives only under `rtl/tech/`.
7. **State the measurement conditions.** Every performance number says how it was measured (model,
   prompt length, generated tokens, build, clock) and carries a provenance label, as in
   [`docs/performance.md`](performance.md).
