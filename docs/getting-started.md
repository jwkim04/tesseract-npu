# Getting started

How to build Tesseract, run a model on the simulator, on PyTorch and on an Alveo U200, serve it over HTTP,
verify the result and find your way around the repository.

## Requirements

- Linux (developed on Ubuntu 24.04), CMake 3.20 or newer and a C11 and C++17 compiler. Use CMake 3.28 or
  newer for a clean `ctest`: with 3.20 to 3.27 llama.cpp's own model tests do not get `TSR_DEVICE=none` and
  run against the Tesseract device.
- The RISC-V toolchain. The firmware of the control core is built with a pinned bare-metal GCC, and `cmake`
  stops at configure time without it. Install it in user space:

  ```sh
  sw/ctrl/tools/install-toolchain.sh     # downloads and verifies the pins in docs/design/toolchain.lock; installs under build/tools/riscv
  ```

  `--steps xpack,check` installs the compiler alone, and the Spike-based `rv_*` tests then report
  "Skipped". For another location pass `-DTSR_RISCV_TOOLS=<prefix>` or
  `-DTSR_RISCV_PREFIX=<dir>/bin/riscv-none-elf-` (and `-DTSR_SPIKE=<spike>`) at configure time.
- Python 3 with numpy, for `ctest`, `numcheck.py` and the cocotb tests (`-DTSR_REF_MATH_PYTHON=<interpreter>`
  selects the one `ctest` uses). For the RTL tests: Verilator 5, cocotb and pytest.
- For bitstream builds: Vivado and Vitis 2024.2 and the U200 development platform
  `xilinx_u200_gen3x16_xdma_2_202110_1`. For the card: XRT 2.18.179 and, on kernels newer than 6.8, the
  patched drivers of [`sw/xrt-driver`](../sw/xrt-driver/README.md).
- For the PyTorch backend: a venv with torch, transformers, safetensors and pytest
  ([`docs/torch-backend.md`](torch-backend.md) §2.1).

## Build and test

llama.cpp is a git submodule pinned to one commit ([`docs/llama-pin.md`](llama-pin.md)). The backend hooks
into it through the patch in `third_party/patches/`.

```sh
git submodule update --init
third_party/apply-patches.sh
sw/ctrl/tools/install-toolchain.sh        # once
cmake -S . -B build/dev -DCMAKE_BUILD_TYPE=Release
cmake --build build/dev -j8
ctest --test-dir build/dev -LE slow
```

The build compiles the firmware project `sw/ctrl` as the external project `tsr_fw` and puts the production
image at `build/dev/fw/tsr-fw.elf`. The host libraries find it through a path compiled in at configure time,
and `TSR_IMAGE` overrides it. `ctest` labels: `fw` (firmware and executor), `rv` (the RV32 interpreter against
Spike), `model` (needs a GGUF file in `models/`), `slow` (opt-in: `-L slow`), and `main` and `python`
(llama.cpp's own tests). A narrower build for the backend and the tools names `tsr_fw` explicitly, since the
image is not a dependency of those targets:

```sh
cmake --build build/dev -j8 --target tsr_fw llama-completion klcheck tsr-selftest tsr-numcheck \
      test-backend-ops test-ggml-tesseract test-tsr-backend-units
build/dev/bin/test-ggml-tesseract                     # bit-exact against tsr_ref, NMSE against the CPU
TSR_DEVICE=sim build/dev/bin/test-backend-ops -o MUL_MAT -b Tesseract0
```

One `MUL_MAT` case of `test-backend-ops` (`type_a=f32,type_b=f32,m=1,n=1,k=2048`) fails its NMSE limit on
some random runs of a correct device. `ctest` leaves it out and checks it bit-exactly
([`docs/llama-pin.md`](llama-pin.md)).

## Run a model on the simulator

Put GGUF files in `models/` (ignored by git).

```sh
LLAMA_ATTN_ROT_DISABLE=1 build/dev/bin/llama-completion -m models/qwen2.5-0.5b-instruct-q8_0.gguf \
    -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -no-cnv -p "The capital of France is" -n 16
build/dev/bin/klcheck all -m models/qwen2.5-0.5b-instruct-q8_0.gguf -o build/klcheck-05b --phase m7 -n 64
```

The backend has two modes, M7 (the default) and M6, defined in [`docs/design/README.md`](design/README.md).
The commands above run M7. The commands below run M6:

```sh
LLAMA_ATTN_ROT_DISABLE=1 TSR_KINDS=m6 build/dev/bin/llama-completion -m models/qwen2.5-0.5b-instruct-q8_0.gguf \
    -ngl 99 -nkvo -fa on -ctk q8_0 -ctv q8_0 -no-cnv -p "The capital of France is" -n 16
build/dev/bin/klcheck all -m models/qwen2.5-0.5b-instruct-q8_0.gguf -o build/klcheck-05b -n 64
```

- `-ngl 99` offloads every layer to `Tesseract0`.
- M7 needs no `-nkvo`: the KV cache lives in Tesseract buffers. M6 needs `TSR_KINDS=m6` and `-nkvo`. A kind
  list such as `TSR_KINDS=all,-RMS_NORM` moves single kinds back to the CPU
  ([`sw/ggml-tesseract/README.md`](../sw/ggml-tesseract/README.md)).
- `-fa on -ctk q8_0 -ctv q8_0` are required with the KV cache on the device, because an f16 cache cannot live
  in device memory. `LLAMA_ATTN_ROT_DISABLE=1` (no Hadamard KV rotation) keeps a decode token to one launch.
  `klcheck` and `tsr-numcheck` set it themselves.
- `-no-cnv` turns off the chat mode that `llama-completion` starts for models with a chat template.
- `klcheck all` runs the device, the CPU, the 1-ulp floors and the embedding-path check and prints the
  verdict ([`docs/klcheck.md`](klcheck.md)).
- `TSR_SIM_THREADS` sets the MVM threads of the simulator (default min(CPUs, 8)). A model whose weights do not
  fit in RAM needs file-backed simulator DRAM: `TSR_SIM_DRAM_DIR=<dir>`.
- `TSR_SIM_CARD=k4full TSR_SIM_GDN=1` configures the simulated core like the release image: `T_MAX` 4, the
  vector unit's 8 / 8 / 4 lanes, the RISC-V clock at half the core clock, the Gated DeltaNet op and none of
  the local-intermediate features. Use it to replay a card run on the simulator.
- Without a GGUF file, `build/dev/bin/tsr-selftest --device sim` boots the firmware on the simulator and runs
  its self-checks.

## Serving

`llama-server` runs on the Tesseract backend from the same build, with one device launch per decode step.

```sh
LLAMA_ATTN_ROT_DISABLE=1 TSR_DEVICE=sim build/dev/bin/llama-server \
    -m models/qwen2.5-0.5b-instruct-q8_0.gguf -ngl 99 -fa on -ctk q8_0 -ctv q8_0 \
    -np 4 -kvu -c 4096 -t 4 --host 127.0.0.1 --port 8080
```

- `-ngl 99 -fa on -ctk q8_0 -ctv q8_0` are required. With an f16 KV cache the server stops at startup and the
  backend's message names these flags (or `-nkvo`).
- `LLAMA_ATTN_ROT_DISABLE=1` gives one launch per token. Without it the server works with more launches.
- `-np N` with N above 1 needs `-kvu` (`--kv-unified`): the device holds one KV stream. With `-np` left out
  the server picks 4 slots with a unified cache.
- Do not pass `--context-shift`. llama.cpp crashes in `build_rope_shift` with a quantized K cache and no
  rotation, on the CPU backend too. Without it a full context ends the request with `finish_reason` `length`
  and a too-long prompt gets HTTP 400.

The endpoints are `/health`, `/v1/models`, `/v1/chat/completions` (also with `"stream": true`) and
`/v1/completions`:

```sh
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"messages": [{"role": "user", "content": "Name the capital of France."}], "max_tokens": 16, "temperature": 0}'
```

The text is not bit-identical to the CPU backend. Greedy output leaves the CPU's at high-entropy tokens, and
with several slots it can differ from a single-slot run at near-ties, because the rounding in flash
attention depends on how the slots' cells interleave in the shared cache. The model-level check is `klcheck`.

## Run a PyTorch model

`torch.compile(model, backend="tesseract")` runs the weight matmuls of a PyTorch model on the device and
everything else eagerly on the CPU (`import tesseract_torch` registers the backend). It uses the same
transport and numerics as the llama.cpp path.

```sh
cmake -S . -B build/dev -DCMAKE_BUILD_TYPE=Release -DTSR_WITH_TORCH=ON
cmake --build build/dev -j8 --target tesseract_torch_ext
export PYTHONPATH=$PWD/sw/torch:$PWD/build/dev/python
build/tools/torchvenv/bin/python -m pytest -q sw/torch/tests
build/tools/torchvenv/bin/python -I sw/torch/tools/hf_check.py models/hf/Qwen2.5-0.5B-Instruct
```

`-DTSR_TORCH_PYTHON=<interpreter>` selects the Python with torch. `hf_check.py` compares a Hugging Face model
through the backend with eager PyTorch (`--download Qwen/Qwen2.5-0.5B-Instruct` fills the model directory).
See [`docs/torch-backend.md`](torch-backend.md), [`sw/torch/README.md`](../sw/torch/README.md) and
[`sw/rt/README.md`](../sw/rt/README.md).

## Run on the card

The release image is `tsr_u200_k4.xclbin`. Its identity, clocks and `CAPS` are in
[`docs/platform-u200.md`](platform-u200.md) §9. `hw/u200/build_xclbin.sh k4` builds it from `rtl/` and `hw/u200/`
into `build/u200/k4`. With XRT installed:

```sh
cmake -S . -B build/xrt -DCMAKE_BUILD_TYPE=Release -DTSR_WITH_XRT=ON     # TSR_XRT_ROOT defaults to /opt/xilinx/xrt
cmake --build build/xrt -j8
source /opt/xilinx/xrt/setup.sh
export TSR_DEVICE=xrt TSR_XCLBIN=$PWD/build/u200/k4/tsr_u200_k4.xclbin
build/xrt/bin/tsr-selftest
LLAMA_ATTN_ROT_DISABLE=1 build/xrt/bin/llama-completion -m MODEL -ngl 99 -fa on -ctk q8_0 -ctv q8_0 \
    -no-cnv -p "The capital of France is" -n 64
build/xrt/bin/klcheck all -m MODEL --device xrt --phase m7 -n 64 -o build/klcheck-card
```

The firmware is not in the xclbin. The host loads `tsr-fw.elf` (or `$TSR_IMAGE`) when the device opens. The
card runs one process at a time, since XRT opens the compute unit exclusively. For the PyTorch backend add
`-DTSR_WITH_TORCH=ON` ([`docs/torch-backend.md`](torch-backend.md) §2.4). The host driver path, the card
selftest and the bring-up steps are in [`docs/xrt-host.md`](xrt-host.md). `build/xrt/bin/tsr-top` in a second terminal
shows the card live while a run is in progress, and the profile report shows what one decode token does
([`docs/prof-dbg.md`](prof-dbg.md)).

## Verify

What each check proves and how to run it. The oracle chain behind them is in [`docs/spec.md`](spec.md) §17.

| Check | What it proves | Command |
| --- | --- | --- |
| Software tests | the reference library against ggml, the simulator, the RV32 interpreter, the firmware and its host build, the transport, the backend, an end-to-end backend test bit-exact against `tsr_ref` and within NMSE 5e-4 of the CPU | `ctest --test-dir build/dev -LE slow` |
| Op level | every supported `MUL_MAT` shape within NMSE 5e-4 of the ggml CPU backend | `build/dev/bin/test-backend-ops -o MUL_MAT -b Tesseract0` |
| Model level | mean KL against the CPU at most 3 times the 1-ulp floor, top-1 agreement at least 80 % ([`klcheck.md`](klcheck.md)) | `build/dev/bin/klcheck all -m MODEL -o DIR -n 64` |
| Numerics | every output element of every weight `MUL_MAT` of one prompt is bit-exact against `libtsr_ref` and an independent numpy model and within a proven fp32 error bound ([`numcheck.md`](numcheck.md)) | `build/dev/bin/tsr-numcheck all -m MODEL -o DIR -T 32 --checks full`, then `python3 sw/tools/numcheck.py --build build/dev compare ...` |
| Firmware and transport | the image boots with receive CRC and VERIFY, the `selftest` kernel agrees bit for bit with a shadow simulator, trace replays, the log decodes | `build/dev/bin/tsr-selftest --device sim` (also `--device mock`, `--debug`, `--traces DIR`) |
| Determinism and races | a trace replays bit-identically under the fixed and random schedules with the race detector fatal | `build/dev/bin/tsr-trace-replay -q --sched fixed --sched random:1 traces/m3`, `TSR_SIM_RACE=fatal` |
| RV32 interpreter | riscv-tests, riscv-arch-test and random programs retire by retire against Spike | `ctest --test-dir build/dev -L rv` (needs Spike) |
| Tables | the generated CSR, opcode, error and PERF tables match `sw/tables/tsr_tables.py` | `python3 sw/tables/gen_tables.py --check` |
| RTL | Verilator `-Wall` clean on every RTL source, and the cocotb benches of the units against the shared reference functions | `rtl/lint.sh`, `tb/run_all.sh` |

`numcheck.py compare` needs the native libraries of the build (`--build build/dev`: `libtsr_ref.so` and
`libnumcheck_helper.so`). `tb/run_all.sh` runs `python3 -m pytest`, so put a `python3` with cocotb, pytest and
numpy first on `PATH` (`-k dma` selects benches, `--no-lint` skips the lint). Opt-in slow switches:
`TSR_REF_MATH_FULL=1` (all 2^32 inputs of the VU math), `TSR_FP_SLOW=1`, `TSR_PLATFORM_SLOW=1` and
`TSR_RV_RANDOM_LONG=1`, run with `ctest -L slow`. The trace data is not in git: regenerate a set with
`traces/m3/regen.sh build/dev` and compare its digest with the manifest ([`traces/README.md`](../traces/README.md)).

## Environment variables

Variables are read when the device or the tool opens. A malformed value makes the open fail and is never silently
ignored. Each component document holds the complete list for its own variables.

| Variable | Effect |
| --- | --- |
| `TSR_DEVICE` | transport kind: `sim` (default), `mock`, `vkernel`, `xrt`. `none` registers no ggml device. |
| `TSR_IMAGE` | firmware ELF to boot. Default: the build's `tsr-fw.elf`. |
| `TSR_XCLBIN` | `xrt`: the xclbin to load ([`xrt-host.md`](xrt-host.md) §4.3) |
| `TSR_KINDS` | the op kinds the device takes. Unset: every kind of the booted image. `m6`: the matrix products. |
| `TSR_SIM_*`, `TSR_TRACE` | the simulated core, its resources and its schedule ([`sw/transport/README.md`](../sw/transport/README.md), [`sw/sim/README.md`](../sw/sim/README.md)) |
| `TSR_LOG`, `TSR_LOG_LEVEL`, `TSR_STATS`, `TSR_DEBUG` | the device log, device counters at exit, the backend's placement and launch log |
| `TSR_GDB`, `TSR_DUMP_DIR`, `TSR_MON`, `TSR_DUMP_PLAN` | debugging and profiling ([`prof-dbg.md`](prof-dbg.md) §1) |
| `LLAMA_ATTN_ROT_DISABLE=1`, `GGML_SCHED_DEBUG=1` or `2` | llama.cpp: no Hadamard rotation of the KV cache, and its scheduler log (use with `-v`) |

## Repository layout

`docs/` has the specification, the design set and the component documents ([`docs/README.md`](README.md) is the index).
`sw/` has `sim/` (the bit-exact simulator and references), `ctrl/` (Basis, the firmware SDK), `transport/`,
`ggml-tesseract/`, `rt/` and `torch/` (the host stack), `tools/` (`klcheck`, `tsr-numcheck`, `tsr-selftest`,
`tsr-dbg`, `tsr-prof`, `tsr-top`, `tsr-log`, the trace tools, `tsr-mtp`), `tables/` (the single source of the CSR
map, opcodes, error codes, PERF slots and formats) and `xrt-driver/`. `rtl/` has `core/` (vendor-independent
SystemVerilog), `tech/` (generic and Xilinx wrappers) and `platform/alveo_xrt/` (the XRT adapter and the kernel tops).
`tb/` has the cocotb and Verilator benches, `traces/` the trace-set generators, `hw/u200/` the `v++` configurations
and `build_xclbin.sh`, and `third_party/` the pinned llama.cpp with its patches. The versions are in
[`versioning.md`](versioning.md) and [`../CHANGELOG.md`](../CHANGELOG.md).
