# PyTorch backend (`torch.compile(model, backend="tesseract")`)

The PyTorch backend runs the weight matmuls of a `torch.compile`d model on a Tesseract device and
leaves everything else to the CPU. The C++ runtime `tsr_rt` below it
runs each device matmul as a one-op op list through the firmware's `graph` kernel. Lowering whole graphs onto
the M7 kernels (attention, norms, MoE, Gated DeltaNet) is not built; §8 says what it would take.

> **Scope.** User documentation, as-built record and measurement record of `sw/torch` and `sw/rt`. The
> backend, `tsr_rt` and their tests run on the firmware of the control core (Tesseract Basis 0.1.0, 21 kinds, kind-table
> signature 0x2381; 0x804f when the results below were measured, before the fused GDN path of SW-78) on the simulator, the mock transport and the U200 card (`tsr_u200_k4`, T_MAX 4). The C++ runtime tests
> (`rt_*`), the Python suite (`torch_pytest`, `torch_pytest_tmax4`, `torch_pytest_tmax8`) and `hf_check.py` were run on all
> of them; §6 has the results, each labelled by the device that produced it. Order of authority: `docs/spec.md` (§5 Device
> weight layout, §6 Quantized activation layout, §7 Numeric rules) > the simulator (`sw/sim`) > this document. Normative for the
> software split: [SW](design/06-software-verification.md) §9.4. The package reference (every API, every
> counter) is `sw/torch/README.md`, the runtime's is `sw/rt/README.md`.

## 1. What it is

`import tesseract_torch` registers the torch.compile backend `"tesseract"`. A model compiled with it
runs **inference** with its **weight matmuls** on Tesseract and everything else on the CPU:

```python
import torch, tesseract_torch as tt

model = torch.compile(model, backend="tesseract")   # or tt.compile(model)
with torch.no_grad():                                # or torch.inference_mode()
    y = model(x)
```

- **Device**: `TSR_DEVICE` picks the transport (`sim` by default; `mock`, `vkernel`, `xrt` through
  the same `tsr_transport` as the llama.cpp path). `sim` and `mock` run on the simulator; `vkernel`
  is not available until P5.3; `xrt` runs the U200 image `tsr_u200_k4` (§6.8). The runtime boots the firmware
  image (`$TSR_IMAGE`, else the build's `tsr-fw.elf`) when it opens the device.
- **What goes to the device**: a matmul (`mm`, `addmm`, `linear`, `matmul`, `mv`) with one operand
  that is a parameter, buffer or graph constant. That operand must be a float32 / float16 / bfloat16
  CPU tensor with K % 32 == 0 and a shape the device accepts. Attention, norms, activations,
  embeddings, bias adds and every other op run on the CPU, as in the llama.cpp path's M6
  (`docs/spec.md` §12, ggml backend integration, `supports_op`).
- **Weights as ggml blocks**: a model that provides Q8_0 or Q4_K blocks (a GGUF tensor) uses `tt.QuantLinear`
  instead of `nn.Linear`. The device holds the blocks as they are (§3.9).
- **Numerics**: the llama.cpp path's. A float weight becomes Q8_0 when it is uploaded, with ggml's
  `quantize_row_q8_0_ref`, so its blocks are byte-identical to llama.cpp's load-time conversion
  (`docs/spec.md` §5, *Load-time conversion*). Activations are quantized on the device by QACT A32 (A256 for Q4_K), and the products are f32 (§3.7).
- **Equivalence**: with the CPU ops unchanged, a compiled model gives exactly eager PyTorch's
  results with every supported `nn.Linear` replaced by the device's reference model
  (`tt.reference_linears()`). The tests check this bit for bit.
- **Not**: training (graphs that need autograd run eagerly on the CPU), a torch device type (tensors
  stay CPU tensors; there is no `.to("tesseract")`), Inductor (the CPU part runs as an eager aten
  graph).

## 2. Build and run

### 2.1 Python environment

The base interpreter (here miniconda's Python 3.13 with torch 2.9.1, a CUDA 13 wheel) is not
modified. The extension and the tests use a venv on top of it that reuses its torch:

```
<base python3> -m venv --system-site-packages build/tools/torchvenv
build/tools/torchvenv/bin/pip install transformers safetensors huggingface_hub pytest
```

On the dev host the venv adds transformers 5.20.2, safetensors 0.8.0, huggingface_hub 2.3.0 and
tokenizers 0.23.3 (pytest 9.0.2 comes from the base). Model files go under `models/` (ignored by
git). Only safetensors are used, never pickle `.bin` files, and `trust_remote_code` is always off.

### 2.2 Build

```
third_party/apply-patches.sh                          # once per checkout, as for every tree with llama.cpp
cmake -S . -B build/dev -DCMAKE_BUILD_TYPE=Release -DTSR_WITH_TORCH=ON
cmake --build build/dev -j8                           # everything; --target tesseract_torch_ext for the extension alone
export PYTHONPATH=$PWD/sw/torch:$PWD/build/dev/python
build/tools/torchvenv/bin/python -c "import tesseract_torch as tt; print(tt.device().description)"
```

A worktree without `build/tools/torchvenv` passes `-DTSR_TORCH_PYTHON=<the venv's python>` (and
`-DTSR_RISCV_TOOLS=<tools dir>` for the firmware). The card needs a second tree with `-DTSR_WITH_XRT=ON -DTSR_XRT_ROOT=/opt/xilinx/xrt`
(§2.4).

- `TSR_WITH_TORCH` (default OFF) adds `sw/torch`. It needs `TSR_WITH_LLAMA=ON`, because `tsr_rt`
  links `ggml-base` of the pinned llama.cpp (§3.2): `-DTSR_WITH_TORCH=ON` with `TSR_WITH_LLAMA=OFF`
  stops with a message. `sw/rt` is added whenever llama.cpp is.
- Torch comes from `TSR_TORCH_PYTHON` (default `build/tools/torchvenv/bin/python` if it exists, else
  `python3`). CMake asks that interpreter's `torch.utils.cpp_extension` for the include and library
  directories and the C++11 ABI flag, and builds a CPU extension against `c10`, `torch`,
  `torch_cpu` and `torch_python`. `find_package(Torch)` is not used: the CUDA wheel's CMake config
  requires a matching CUDA toolkit.
- Output: `<build>/python/_tesseract_rt.cpython-313-x86_64-linux-gnu.so`, for that
  interpreter's Python only. If `<build>/python` is not on `sys.path`, the package also looks in
  `$TSR_TORCH_EXT_DIR`, `<repo>/build/torch/python` and `<repo>/build/dev/python`.
- The extension alone takes about 20 s and 1.5 GB (one translation unit that includes `torch/extension.h`).
  The tests of the whole tree need `cmake --build build/dev -j8` (the firmware image, the simulator and the pinned llama.cpp).

### 2.3 Run and test

```
# pytest suite (TSR_DEVICE=sim unless set)
build/tools/torchvenv/bin/python -m pytest -q sw/torch/tests
ctest --test-dir build/dev -R '^(torch_pytest|rt_)'     # sim T_MAX 1 / 4 / 8, rt on sim and mock
# opt-in Hugging Face test
TSR_TORCH_HF_MODEL=models/hf/Qwen2.5-0.5B-Instruct build/tools/torchvenv/bin/python -m pytest -q sw/torch/tests/test_hf.py
# end-to-end check of a Hugging Face causal LM (§6.5); -I because the model directory is data
build/tools/torchvenv/bin/python -I sw/torch/tools/hf_check.py models/hf/Qwen2.5-0.5B-Instruct \
    [--download Qwen/Qwen2.5-0.5B-Instruct] [--tokens 32] [--gen 16] [--gguf GGUF] [--json OUT]
# compiled vs eager on a Hugging Face model, f32 or bf16 (§6.5)
build/tools/torchvenv/bin/python -I sw/torch/examples/hf_compare.py models/hf/Qwen2.5-0.5B-Instruct --dtype bfloat16
```

`TSR_DEVICE=mock` runs the same suite on the adapter model of the transport. `TSR_SIM_TMAX=4` gives the simulated
core the `T_MAX` of the K4 image, `TSR_SIM_TMAX=8` the `T_MAX` of the 2026-10-25 image. `-I` ignores `PYTHONPATH`: the
tools find the extension through `TSR_TORCH_EXT_DIR` or the build trees named in §2.2.

`hf_check.py --forward-only --tokens 8 --expect sim.npy` compares the forward's logits bit for bit with
a `TSR_DEVICE=sim` run saved with `--save-logits`. That is the model-level check of `docs/spec.md` §17
(same model run, bit-identical logits), done from PyTorch. The same `--expect` check runs against any
device kind; `TSR_DEVICE=vkernel` is not available until the two-clock server exists (P5.3). `--speed 1,8,32` times
steady-state forwards (§6.8).

### 2.4 On the card

```
cmake -S . -B build/xrt -DCMAKE_BUILD_TYPE=Release -DTSR_WITH_TORCH=ON -DTSR_WITH_XRT=ON -DTSR_XRT_ROOT=/opt/xilinx/xrt
cmake --build build/xrt -j8 --target tesseract_torch_ext tsr-selftest tsr_fw numcheck_helper
source /opt/xilinx/xrt/setup.sh
export TSR_DEVICE=xrt TSR_XCLBIN=<build/link/k4full/tsr_u200_k4.xclbin> TSR_IMAGE=$PWD/build/xrt/fw/tsr-fw.elf
export TSR_TORCH_EXT_DIR=$PWD/build/xrt/python PYTHONPATH=$PWD/sw/torch:$PWD/build/xrt/python
build/xrt/bin/tsr-selftest --device xrt --no-mem --no-kernel --rate-n 10 --traces traces/at/errors/at_s53_0009   # before each session
TSR_TORCH_VERIFY=1 build/tools/torchvenv/bin/python -m pytest -q sw/torch/tests
```

The card runs one process at a time. After a load, the first attention launch of the image `tsr_u200_k4` needs a replay of
the trace `at_s53_0009` (`docs/platform-u200.md` §8.8, "Card bring-up of the fourth link"); that is the `tsr-selftest`
line above. The firmware (`tsr-fw.elf`) is not in the xclbin: the host loads the image of `TSR_IMAGE` when the device opens.

## 3. Architecture

### 3.1 Layers

```
torch.compile(model, backend="tesseract")
  └─ tesseract_torch (Python, sw/torch/tesseract_torch)
       ├─ backend.py   Dynamo graph → AOTAutograd → aten inference graph → partition → device_mm nodes;
       │               the rewritten graph runs eagerly (FX GraphModule, no Inductor)
       └─ runtime.py   device singleton, WeightCache, eager linear / linear_ref, counters, debug, verify
            └─ _tesseract_rt (C++ extension, sw/torch/csrc/tsr_torch.cpp; torch tensors in and out, GIL released)
                 └─ tsr_rt (sw/rt): weight upload, linear, N / K parts, op list per call, OPL cache, counters, reference path
                      ├─ tsr_lower (sw/ggml-tesseract/lower.cmake): shape checks, OPL builder and cache, `graph` launcher, allocator, weight layout  ← shared
                      ├─ tsr-weights.cpp + ggml-base (pinned llama.cpp): quantize_row_q8_0_ref, f16 / bf16 to_float ← shared
                      ├─ tsr_ref (libtsr_ref): the reference model behind linear_ref and TSR_TORCH_VERIFY
                      └─ tsr_transport: sim | mock | vkernel | xrt  →  firmware `graph` kernel (MUL_MAT records)
```

The two lines marked "shared" are the same code the ggml backend uses: both paths build a weight
matmul as an op list with the same builder and launcher and store the same weight bytes. The
firmware tiles the matmul with `plan.hpp`, the planner the host's `supports_op` uses.

### 3.2 Shared code: `tsr_lower` and the load-time conversion

`tsr_lower` is a static PIC library with no ggml dependency. It holds `tsr-lower` (MUL_MAT shape
checks and the SPM tiling used to place and split weights), `tsr-opl` and `tsr-opl-dev` (the op list
builder with its hazard analysis and JOINs, the OPL cache, and the launcher of the `graph` kernel on a
transport), `tsr-alloc` (first-fit range allocator) and `tsr-wlayout` (device-format row bytes, row
repack, symmetric scatter / gather over the channels). The files stay in `sw/ggml-tesseract`.
`lower.cmake` defines the target,
and the top-level CMakeLists includes it before llama.cpp. `ggml-tesseract` links it with a deferred
`target_link_libraries` in `sources.cmake`, so the llama.cpp patch (0001) is unchanged. The ggml
backend's weight store and read-back call the shared scatter / gather.

The f32 / f16 / bf16 → Q8_0 conversion is the backend's own `tsr-weights.cpp`, compiled into
`tsr_rt`, linked with `ggml-base`. libtsr_ref's `tsr_qact` (A32) was measured as a replacement and
rejected. It equals `quantize_row_q8_0_ref` on 200000 / 200000 ordinary blocks but differs on 9 of
160 blocks with subnormals or a tiny maximum (DAZ / FTZ, `docs/spec.md` §21), and it writes a QA image
with an f32 scale, not `block_q8_0`. Only ggml's function gives "the same blocks as llama.cpp".

### 3.3 The runtime `tsr_rt` (sw/rt)

`tsr::rt::device` (`sw/rt/tsr_rt.h`) opens `TSR_DEVICE`, checks the configuration, and offers
`upload`, `free_weight`, `info`, `read_blocks`, `linear`, `linear_ref`, counters and `memory`. Every
member function takes one device mutex. Targets: `tsr_rt` (static, PIC; linked into the extension)
and `tsr_rt_shared` (`libtsr_rt.so`, which needs only `libggml-base.so`).

- **Device memory.** OPL (ch0, 64 MiB) holds the cached op lists. IO (ch0, 256 MiB) holds the x
  rows and the y rows of one launch. WEIGHT (every channel, from 512 MiB) holds the weights, with
  symmetric allocation (same W_OFF on every channel, 4 KiB aligned). AUX is unused.
- **Upload.** f32 / f16 / bf16 → Q8_0 (§3.2), or native Q8_0 / Q4_K ggml blocks
  (`upload_blocks`). Any row stride is accepted, and so are misaligned rows. Rows are repacked into
  the device layout (`docs/spec.md` §5) and scattered over the channels.
- **`linear(h, x [T, K] f32) → y [T, N] f32`.** x is staged in IO. The call builds one op list with a
  MUL_MAT record per N part and K part (the ggml backend's builder) and launches the `graph` kernel
  once, per IO-sized chunk of T. The firmware tiles T by min(T_MAX, its scratch ring) and issues the
  `LD / QACT / MVM / ST` tile ops. The runtime does not tile by T_MAX itself. On the K4 image (T_MAX 4) a
  32-token call is eight tiles in one launch; the weights are read once per tile (§6.5 has the cycles per token at
  T_MAX 1, 4 and 8).
- **N parts.** A weight is split into row ranges when x + QA + y of T_MAX tokens would not fit the
  SPM. Each part is its own allocation and MVM. Rows are independent, so the result is bit-identical
  to an unsplit weight. Example: N = 130000, K = 896 at T_MAX 8 gives 2 parts.
- **K parts.** When K > K_MAX (32768), the columns are cut into equal block-multiple parts. The first
  record of an N part writes y; the others carry `ACC_DST`, so the kernel loads the old y rows and runs
  the MVM with `ACC`: y = ((y₀ + y₁) + y₂)…, one f32 add each (`docs/spec.md` §21, *ACC addition*).
  This differs from the unsplit dot product, and `linear_ref` models it exactly.
- **Op-list cache.** The key is the bytes of the list (formats, shapes, T, the x and y handles, every
  part's W_OFF). An LRU in the OPL region keeps the lists resident, so a repeated call writes x,
  launches, and reads y. Measured on the simulator for N = 64, K = 896, T = 3 at
  T_MAX 1 on a repeated call: 1 exec, 1 hit, 0 misses; h2d 10752 B (exactly x), d2h 768 B (exactly y),
  no weight bytes.
- **Counters**: `execs` (launches), `prog_hits` / `prog_misses` (op-list cache hits and misses), PERF
  cycle deltas, device read bytes (Σ PERF_CH_RD_BEATS × 64), h2d (of which weights) and d2h bytes.

### 3.4 The extension `_tesseract_rt`

`_tesseract_rt.Device` binds `tsr::rt::device` for CPU torch tensors: `upload(w [N, K])`,
`upload_blocks("q8_0" | "q4_k", uint8 blocks, N, K)`, `linear(h, x, out=None)`, `linear_ref`,
`read_blocks`, `info`, `free`, `counters`, `memory`, `unsupported_reason(dtype, N, K)`, `config` (a dict: `num_ch`, `t_max`,
`k_max`, `caps`, `local_scale`, …), `_try_linear_scale(h, x, scale)` (None when the device cannot run it), `close()`, and
`serial` (1, 2, … per device opened in the process). Device calls release the GIL. Each method holds
a lease that keeps the device alive until it returns, so `close()` from another thread waits for the
calls in flight, and a later call raises `TesseractError: ... the device is closed`. Module
functions `open` / `get` / `is_open` / `close` manage the process-wide device. `tsr::rt::error` maps
to `_tesseract_rt.TesseractError` (a `RuntimeError`), `std::invalid_argument` to `ValueError`.

### 3.5 The partitioner (backend.py)

1. **Dynamo → AOTAutograd.** The backend wraps `aot_autograd` with one compiler for the forward and
   inference graphs (`keep_inference_input_mutations=True`). A graph that needs autograd (grad mode on
   and an input that requires grad) is not partitioned. It runs eagerly with one `RuntimeWarning`,
   and `strict=True` raises instead. The device is opened at the first compilation, so an
   unavailable device fails `torch.compile` instead of running on the CPU.
2. **Weights** are found from AOT's input descriptors (`ParamAOTInput`, `BufferAOTInput`, or a
   `PlainAOTInput` of a Dynamo input that is an `nn.Parameter` or whose source is a module's
   `_parameters` / `_buffers`) and from graph constants (`get_attr`). So parameters that arrive as
   graph inputs under `inline_inbuilt_nn_modules` (the default) and parameters lifted by AOT
   (`inline_inbuilt_nn_modules=False`) are both found. A weight may be seen through view ops (`t`,
   `transpose`, `permute`, `view`, `reshape`, `slice`, `select`, `expand`, `as_strided`, …). A
   tensor computed in the graph (e.g. an autocast `_to_copy`) is not a weight.
3. **Eligible nodes**: `aten.mm`, `addmm`, `linear`, `matmul`, `mv` with one weight operand and one
   non-weight operand. After orientation the weight must be a 2D [N, K] CPU tensor of a supported
   dtype, with K % 32 == 0, and the device must accept the shape (`unsupported_reason`; tsr_rt
   splits large N and K > K_MAX itself). Both orders work: `x @ Wᵀ` (nn.Linear), `x @ W` with a
   [K, N] weight (GPT-2 Conv1D style; the uploaded copy is Wᵀ), and `W @ x` / `mv(W, v)`, computed as
   (xᵀ Wᵀ)ᵀ. Batch and sequence dims are flattened into T.
4. **Rewrite.** Each eligible node becomes a call of `device_mm(spec_id, a, b, bias, root)`. `root`
   is the parameter or buffer the weight operand is a view of. The rest of the graph is unchanged
   and runs eagerly on the CPU.
   - *Scaled result.* A bias-free `mm` / `matmul` / `linear` with the weight on the right, float32 output and one user
     that is a multiply by a positive power of two in [1, 2^127] becomes one `device_mm` with a `scale`. When the device
     has `MVM_LOCAL_SCALE` and the weight has one K part, the kernel applies the scale (MUL_MAT_SCALE, `tsr_rt`
     `try_linear_scale`); otherwise, and under `TSR_TORCH_VERIFY`, the scale is the CPU multiply after the ordinary
     `linear`. Both give the same bits (counter `local_scale_calls`). The card image has no such capability.
5. **Higher-order ops.** Matmuls in the bodies of `invoke_subgraph`
   (`torch.compiler.nested_compile_region`), `cond`, `while_loop` and `map` are partitioned
   recursively (up to 8 levels). A body placeholder is a weight if the op passes it a weight at
   every call site, so a body shared by all layers is one device node that sees every layer's
   weight. Matmuls in other HOP bodies stay on the CPU and are reported.
6. **Per-weight state.** One graph can serve many weights: regional compilation (`layer.compile()`
   per block), a compiled helper called per layer, or a shared HOP body. The run-time state of a
   device node (uploads in a row, CPU fallback) is therefore kept per (node, root weight).

`TSR_TORCH_DEBUG=1` prints the partition of every graph: which matmuls go to the device, and for
each one that stays on the CPU, why (e.g. `cpu addmm_1 addmm head.weight: K % 32 != 0 (K = 48)`).

### 3.6 Weight cache

`WeightCache` (runtime.py) keys a device copy by tensor identity: the storage (its StorageImpl,
protected from address reuse by a `StorageWeakRef`), the data pointer, storage offset, shape, stride
and dtype.

- **Hits and re-uploads.** A hit also needs the same `_version` and the device the copy was uploaded
  to (`Device.serial`). An in-place update (`mul_`, `copy_`, `load_state_dict`) bumps `_version`, so
  the next call uploads the weight again. A weight is uploaded on first use and then shared by every
  graph that uses it: recompiles for new lengths, graph breaks, tied weights (Qwen2.5's `lm_head`
  shares `embed_tokens`' upload).
- **Inference tensors.** They have no version counter. For them a hit also needs the same raw bits
  in a sample of up to 64 × 64 elements, taken at upload.
- **Release.** Copies whose storage died are released on the next miss, by `sweep()` or by
  `tt.release_weights()`.
- **Not detected.** Writes through `.data` do not bump `_version`, and an inference-tensor change
  can miss every sampled element. Call `release_weights()` after either.

### 3.7 Numerics

| Step | Where | Rule |
| --- | --- | --- |
| Weight f32 / f16 / bf16 → Q8_0 | host, at upload (tsr_rt) | f16 / bf16 widened exactly, then `quantize_row_q8_0_ref` (llama.cpp's load-time conversion), repacked into the device layout |
| Native Q8_0 / Q4_K blocks | host, at upload | no conversion, only the repack (`upload_blocks`) |
| Activation x | host | widened exactly to f32 if it is f16 / bf16 |
| QACT A32 | device | `docs/spec.md` §6 (*Quantization rules*, = `quantize_row_q8_0_ref`); A256 for Q4_K weights |
| MVM | device | `docs/spec.md` §7 (Numeric rules): exact integer block sums, 16 partials, fixed tree, no FMA; y in f32 |
| `alpha`, `beta · input`, `linear` bias | host | f32 (`out.mul_(alpha)`, `out.add_(bias, alpha=beta)`); `beta = 0` ignores the input, as in torch |
| Output | host | one cast from f32 to the op's output dtype (= the input dtype) |

The result therefore equals the llama.cpp path for the same weight and activation rows. §6.5 checks
this on three Qwen2.5 layers against libtsr_ref, the oracle of the llama.cpp path. Against torch's
float32 matmul the device has the error of 8-bit weights and 8-bit activations, as llama.cpp's Q8_0
path does. `TSR_TORCH_VERIFY=1` recomputes every device matmul with libtsr_ref on the weight rows
read back from the device and raises on the first differing bit (for vkernel / xrt bring-up).

### 3.8 Errors, fallbacks, threads

- **CPU fallback, only for placement.** A weight that cannot be placed (WEIGHT heap full, after
  releasing dead weights and retrying) runs on the CPU at that node from then on (one warning,
  counter `cpu_fallback_calls`). So does a weight uploaded on 4 consecutive calls of a node, because
  it was changed in place before each call or is seen through an op that copies it. Both rules
  are per weight.
- **Everything else is an error** (`TesseractError`): a device that cannot be opened (a
  `TSR_DEVICE` typo, xrt without a card, vkernel without a server), a transport error during an
  upload, a device execution error.
- **Threads.** Device calls release the GIL, and tsr_rt serializes them on its mutex. 4 threads
  running compiled calls concurrently gave 0 mismatches (review run).

### 3.9 Weights as ggml blocks (`QuantLinear`)

A model that provides Q8_0 or Q4_K blocks keeps them: `tt.QuantLinear(blocks, in_features, fmt, bias=None)` holds a uint8
buffer [N, row_bytes] and calls the custom op `tesseract::linear_blocks(x, blocks, fmt, K)` (`tesseract_torch/quant.py`).
`QuantLinear.from_gguf(t)` builds one from a `gguf.ReaderTensor` of type Q8_0 or Q4_K.

- **Upload.** `WeightCache.acquire_blocks` uploads the blocks with `Device.upload_blocks` (stored as they are, no
  conversion; `read_blocks` returns the same bytes). The key is the cache key of §3.6 plus the format and K, so the
  version, re-upload and release rules are the same.
- **Compute.** QACT A32 (Q8_0) or A256 (Q4_K) on the device, f32 result `[..., N]`; the module adds the bias in float32
  and casts once to the input's dtype, as `tt.linear` does. `TSR_TORCH_VERIFY` checks the call against the reference model.
- **Under `torch.compile`.** The op is a custom op with a fake implementation, so Dynamo and AOTAutograd keep it as one
  node and the partitioner does not touch it. It also runs eagerly. It is inference only.
- **No CPU fallback.** A weight that cannot be placed raises `TesseractError` (a float weight would run on the CPU). A Q4_K weight is
  read at 0.5625 bytes per element and a Q8_0 weight at 1.0625, and a float weight is stored as Q8_0.
- **Tests.** `test_quant.py` (§6.2): bits equal the numpy spec model of the format, eager and compiled, T = 1, 5, 9,
  bias and batch dimensions, a mixed model with `TSR_TORCH_VERIFY`, in-place change, GGUF tensors written and read by gguf-py.

What is missing for a model run from a GGUF file is the loader: the map from GGUF tensor names to modules, and the norms,
embeddings and RoPE parameters from the same file (§8). SW-81 records the decision.

## 4. Supported ops and fallbacks

| Case | Runs on | Note |
| --- | --- | --- |
| `nn.Linear`, `F.linear`, `x @ W.T`, `addmm` (bias / alpha / beta), `x @ W` ([K, N] weight), `W @ x`, `mv(W, v)` | **Tesseract** | weight f32 / f16 / bf16, K % 32 == 0, 2D after orientation, CPU tensor |
| the same inside `nested_compile_region`, `cond`, `while_loop`, `map` bodies | **Tesseract** | weight passed at every call site |
| weight with K % 32 != 0, float64 weight | CPU | reported by `TSR_TORCH_DEBUG=1`; bit-identical to eager |
| weight on another device (e.g. CUDA) | not offloaded | the op runs as in eager |
| `bmm`, `baddbmm`, `addmv`, `dot`, `einsum`, `_scaled_mm`; batched weights (MoE expert stacks) | CPU | not mapped |
| matmul of two activations (attention scores, SDPA) | CPU | neither operand is a weight; the fused SDPA op stays on the CPU |
| a weight computed in the graph (autocast `_to_copy`, dequantization) | CPU | not a parameter / buffer / constant |
| matmuls in other HOP bodies | CPU | reported as CPU nodes |
| norms, activations, RoPE, softmax, embeddings, residuals, bias adds | CPU | the M7 kernels exist (21 kinds), but this backend only emits MUL_MAT records (§8) |
| graphs that need autograd | CPU, eager | one warning; `strict=True` raises |
| WEIGHT heap full, or 4 uploads in a row | CPU (that weight, that node) | one warning, `cpu_fallback_calls` |
| native Q8_0 / Q4_K ggml blocks | Tesseract | `tt.QuantLinear` / `tt.linear_blocks` (§3.9), or the extension API `Device.upload_blocks` + `linear` |
| bias-free matmul followed by a positive power-of-two scalar multiply | Tesseract (the scale in the kernel when the device has `MVM_LOCAL_SCALE`, else on the CPU) | §3.5 |

## 5. Environment variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `TSR_DEVICE` | `sim` | transport: `sim`, `mock`, `vkernel`, `xrt`; with their own variables (`TSR_SIM_*`, `TSR_MOCK_*`, `TSR_VKERNEL_*`, `TSR_XRT_*`, `TSR_IMAGE`, `TSR_TRACE`, `TSR_LOG`: [`sw/transport/README.md`](../sw/transport/README.md), [`docs/getting-started.md`](getting-started.md#environment-variables)) |
| `TSR_TORCH_DEBUG` | 0 | `1`: partition of every compiled graph, every upload, counters at exit; `2`: also every device call; `3`: also the rewritten aten graphs |
| `TSR_TORCH_VERIFY` | 0 | `1`: every device matmul is checked bit for bit against libtsr_ref on the read-back weight rows; a mismatch raises, naming the weight, the element and both values |
| `TSR_TORCH_EXT_DIR` | — | where to find `_tesseract_rt` if it is not on `sys.path` (then `build/torch/python`, `build/dev/python`) |
| `TSR_TORCH_PYTHON` | venv, else `python3` | CMake cache variable: interpreter whose torch the extension is built against |
| `TSR_TORCH_HF_MODEL` | — | model directory for the opt-in `test_hf.py` |
| `TSR_RT_DEBUG` | 0 | tsr_rt's log: `1` uploads, calls and launches |

`tt.compile(model, strict=False, debug=None, verify=None, offload=True, **kw)` passes the same
switches as `options=` of `torch.compile`. `offload=False` prints the partition but keeps every
matmul on the CPU.

## 6. Tests and measured results

§6.1 to §6.5 were measured on the simulator (build tree `build/dev`, Release, torch 2.9.1, Python 3.13,
`TSR_SIM_THREADS=4`, firmware image id `99698a44030574f4`: Basis 0.1.0, 21 kinds, signature 0x804f; branch
`torch-basis`, 2026-11-09). The 16-core / 14 GB dev host ran other jobs at the same time (load average 30 to 60), so
wall times in §6.2 and §6.5 say nothing about speed; cycle counts are not affected. "Program" and "program cache"
in those sections mean the op list and the op-list cache. §6.8 has the card.

### 6.1 tsr_rt (`test-tsr-rt`)

Every case checks four things. `read_blocks` must equal ggml's blocks byte for byte. `linear` and
`linear_ref` must equal an independent model (ggml conversion + libtsr_ref) bit for bit. The y
padding must be untouched. The NMSE against an f64 product of the dequantized weight must stay
< 1e-3 (measured max 7.89e-5). Shapes: f32 / f16 / bf16 → Q8_0 and native Q8_0 / Q4_K; K 32 … 4864
and 33024 / 33280 (> K_MAX, 2 K parts); N 1 … 4864; T 1 … 20; forced N parts (2 … 7), K parts
(4 … 5) and both (4 × 4); strided x / y; padded and misaligned weight rows.

| Run | Result |
| --- | --- |
| `rt_sim` (T_MAX 1) | 43 / 43 cases |
| `rt_sim_tmax4` (`--big`; T_MAX 4, the K4 image's) | 44 / 44; includes Q8_0 N = 130000, K = 896, T = 9 in one N part, bit-exact |
| `rt_sim_tmax8` (`--big`) | 44 / 44; the same head, split automatically into 2 N parts, bit-exact |
| `rt_mock` (DMA align 64, chunk 64 KiB), `rt_mock_tmax8` (3 ch, T_MAX 8, align 4096, chunk 12 KiB) | 43 / 43 each |
| `rt_sim_gdb_embed`, `rt_sim_basis_old`, `rt_sim_no_mulmat` | the embedded debug service starts (4 / 4 cases); an image of an older Basis and one without MUL_MAT are refused with the reason |
| op-list cache, traffic | see §3.3 |
| IO split, free / re-upload, refusals, threads | a 64 KiB IO area with T = 40 gives 3 programs, bit-exact; memory fully returned after free, double free rejected; K = 48, Q4_K with K = 288, N = 0 and `max_part_k = 48` rejected; 2 threads × 12 calls bit-exact |
| mutation check (scratch builds of tsr_rt.cpp) | detected: wrong QACT column offset in K parts (12 failed checks), dropped ACC (12), shifted N-part y gather (10), K part converted from column 0 (24), wrong x stride (6); an equivalent mutant (N part + 1 row) still passes, as it should |
| build | fresh configure with `TSR_WITH_LLAMA=ON` and `OFF` both succeed (`tsr_lower` builds without llama.cpp); `libtsr_rt.so` linked alone runs a linear on sim with y == `linear_ref`; no compiler warnings |

All eight `rt_*` ctests pass (`ctest --test-dir build/dev -R '^rt_'`, 20 s). The mutation row above is the record of the
runtime's first test run (scratch builds of `tsr_rt.cpp`; the test cases have only grown since). `rt_vkernel_k4t1` (opt-in,
`-DTSR_RT_VKERNEL_TEST=ON`) reports "skipped" until P5.3.

### 6.2 pytest suite (`sw/torch/tests`)

| File | Content |
| --- | --- |
| `test_ext.py` | extension, 11 shapes (N 1 … 4864, K 32 … 4864, T 1 … 40) × 3 dtypes: stored blocks == an independent numpy `quantize_row_q8_0_ref` byte for byte (`spec_model.py`), `linear` == `linear_ref` == the numpy spec model of numcheck.py (B1b) bit for bit; directed rounding rows; the N = 151936 head (2048 sampled rows); K > K_MAX; strided / transposed / 2-byte-misaligned operands; T = 1 … 300 with exact host traffic; a T_MAX 8 sim device; native Q8_0 / Q4_K; errors |
| `test_numerics.py` | NMSE against torch float32, and the single-output exception (§6.3) |
| `test_backend.py` | compiled MLP (f32 / f16 / bf16) and pre-norm transformer block (q / k / v / o, SDPA, SwiGLU; f32 / bf16, with verify) vs eager; fallbacks; re-upload; program cache; B·S > T_MAX; autograd refusal; operand forms; dynamic lengths; tied weights, graph breaks; debug output; close / reopen; heap-full fallback; the scaled result (MUL_MAT_SCALE when the device has it, the CPU multiply otherwise, same bits) |
| `test_quant.py` | `QuantLinear` (§3.9): Q8_0 and Q4_K blocks equal the numpy spec model of the format, eager and compiled (T = 1, 5, 9), the stored blocks equal the given ones, bias and bf16 batch dimensions, a mixed model under `TSR_TORCH_VERIFY`, in-place change (re-upload), GGUF tensors written and read by gguf-py, refusals |
| `test_robustness.py` | one regression test (or group) per review finding (§6.7) |
| `test_hf.py` | opt-in, Qwen2.5-0.5B-Instruct, 12 + 4 tokens |

| Run (simulator, `build/dev`) | Result | Wall | Max RSS |
| --- | --- | --- | --- |
| `TSR_DEVICE=sim` (T_MAX 1) | 115 passed, 1 skipped (the opt-in HF test) | 192 s | 1.31 GB |
| `TSR_DEVICE=sim TSR_SIM_TMAX=4` (the K4 image's) | 115 passed, 1 skipped | 266 s | 1.13 GB |
| `TSR_DEVICE=sim TSR_SIM_TMAX=8` | 115 passed, 1 skipped | 160 s | 1.27 GB |
| `TSR_DEVICE=mock`, DMA align 64, chunk 64 KiB | 114 passed, 2 skipped (+ the sim-only T_MAX 8 device test) | 109 s | 1.18 GB |
| `TSR_DEVICE=mock`, T_MAX 8, 3 ch, align 4096, chunk 12 KiB | 114 passed, 2 skipped | 77 s | 1.46 GB |
| opt-in `test_hf.py` (sim) | 1 passed | 126 s | 4.8 GB |
| `ctest --test-dir build/dev -LE slow -j8` | 316 of 316 passed (1 disabled), among them `torch_pytest`, `torch_pytest_tmax4`, `torch_pytest_tmax8` (about 164 s each at `-j8`), the eight `rt_*` tests and the `ggml_tesseract_*` tests | 745 s | |

The wall times are those of the loaded host and are not a speed record; the card's run is in §6.8.

### 6.3 Bit-exactness and NMSE

- **Device == reference == spec model.** 11 shapes × f32 / f16 / bf16: `linear` == `linear_ref`
  == the numcheck.py numpy spec model, bit for bit. The stored blocks equal an independent numpy
  `quantize_row_q8_0_ref`, byte for byte. The same holds for the N = 151936 head (2048 sampled
  rows), K > K_MAX (2 K parts), a T_MAX 8 sim device and native Q8_0 / Q4_K (§3.9). The numbers
  below (NMSE, single-output trials) are the ones the first run on the simulator gave; the run on the Basis firmware
  (T_MAX 1, 4 and 8) gives them again to the last digit.
- **Negative controls** (scratch run, random 256 × 4096 weight, 1048576 values). A half-even
  weight quantizer differs from the stored blocks in 2 values. The x86 SIMD one (`id = 127 / amax`)
  differs in 3. Sequential accumulation differs from the device in 882 of 1024 outputs.
- **NMSE against torch float32** (uniform [−1, 1]; 12 shapes × 3 weight dtypes × 5 seeds = 180
  cases, all with N·T ≥ 16): min 7.95e-6, median 2.83e-5, p90 3.92e-5, p99 6.77e-5, max 8.90e-5.
  The max is 17.8 % of the 5e-4 limit of test-backend-ops.
- **Single-output exception** (`docs/spec.md` §17, *Exception to op-level NMSE*). For N = T = 1, K = 2048,
  f32, the NMSE is one squared relative error. In 1000 seeded trials it was ≥ 5e-4 in 149 (14.9 %;
  `docs/llama-pin.md`: 15.4 % of 10000). The exact f64 product of the same 8-bit operands fails on
  exactly the same 149, and the device is bit-exact with the reference on all 1000. Like the ggml
  ctest, this shape is checked bit for bit, not by NMSE.
- **Compiled vs eager.** The MLP (f32 / f16 / bf16) and the transformer block (f32 / bf16, verify
  on) are bit-identical to eager with reference linears. Their NMSE against float64 eager is
  < 5e-4. In the block, 7 matmuls run on the device, and attention stays inside the fused SDPA op
  on the CPU. Fallback nodes (K % 32 != 0, float64) are bit-identical to eager.

### 6.4 The llama.cpp path did not regress

- This work changes no file under `sw/ggml-tesseract` and no `third_party` file; the CMake changes add
  ctests (`rt_sim_tmax4`, `torch_pytest_tmax4`) and are behind `TSR_WITH_TORCH` (default OFF) for the torch ones. The shared code
  (`tsr_lower`, §3.2) is the code the ggml backend runs.
- `ctest --test-dir build/dev -LE slow -j8` (the tree with `TSR_WITH_TORCH=ON`) passes 316 of 316 tests (one more is
  disabled), among them `ggml_tesseract_units`, `_e2e`, `_e2e_tmax8`, `_e2e_hosted`, `_e2e_hosted_tmax8`, `_backend_ops*`,
  `fw_*`, the `tools_selftest_mock*` trace replays and the three `torch_pytest*` tests.

### 6.5 Qwen2.5-0.5B-Instruct (Hugging Face safetensors)

Model: `models/hf/Qwen2.5-0.5B-Instruct`, Hub revision 7ae55760. `model.safetensors` (988 MB, sha256
`fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`) matches the Hub's LFS etag. There are no pickle files, and
`trust_remote_code` is off. `hf_check.py` (`sw/torch/tools`) does the following:

- **Prompt.** numcheck's fixed sentence, cut to 32 tokens, teacher-forced, scored at all 32
  positions. Float32 model, 8 torch threads, `TSR_DEVICE=sim` (T_MAX 1).
- **KL.** Per position with klcheck's statistics in float64.
- **Floor.** The 1-ulp floor exactly as klcheck makes it, on the input embeddings. The Python port
  of the nudge gives the same rows as `klcheck embd --perturb-ulp` in 4 seed / sequence / position
  cases. Without the nudge, the embedding path gives the token path's bits.
- **cpu-q8.** The CPU computing each Linear from the dequantized device blocks and the A32-quantized
  activations with torch's f32 GEMM. This is the comparison the spec's ratio is defined for: the
  same quantized model with a different accumulation order.
- **Generation.** 16 greedy tokens (full recompute per step).

The run below is on the simulator, T_MAX 1, branch `torch-basis` (2026-11-09): PASS, 3.2 GB peak RSS. Its per-position KL
values and generated tokens are identical to those of the first run of this check on the simulator (October 2026). The PERF cycles are not (below). The same check on the card gives the same values (§6.8).

| Comparison | Mean KL | Median | Max | Top-1 | Max \|Δlogit\| |
| --- | --- | --- | --- | --- | --- |
| CPU f32 vs device | 3.565e-3 | 2.054e-3 | 3.74e-2 | 30 / 32 | 1.77 |
| CPU f32 vs CPU f32 + 1 ulp (floor) | 3.56e-12 | 3.33e-12 | 9.5e-12 | 32 / 32 | 2.7e-5 |
| CPU f32 vs cpu-q8 | 3.759e-3 | 2.192e-3 | 4.11e-2 | 30 / 32 | 1.77 |
| CPU f32 vs weights-only Q8_0 | 2.027e-3 | 8.59e-4 | 3.75e-2 | 31 / 32 | 1.50 |
| cpu-q8 vs device | 1.730e-3 | 1.345e-3 | 5.38e-3 | 32 / 32 | 0.75 |
| cpu-q8 vs cpu-q8 + 1 ulp (floor) | 5.93e-4 | 1.70e-4 | 2.42e-3 | 32 / 32 | 0.49 |

- **KL ratio.** Device KL / floor KL is 1.0e9 against the f32 CPU. That only says the gap is the
  8-bit quantization: an unquantized f32 model's floor is ~1e-12. Against cpu-q8 the ratio is
  **2.92**. cpu-q8's floor is exactly 0 at positions 0–15, because the 8-bit activations absorb the
  nudge (the effect `docs/spec.md` §17 describes for llama.cpp). Over the 16 positions where it moved,
  the ratio is **1.72**. hf_check reports the ratios but does not gate on them.
- **Top-1 misses.** The device and cpu-q8 miss top-1 at the same two positions (20, 23).
- **Generation.** The 16 greedy tokens of CPU and device are identical (" that the printing of books
  could be done more quickly and cheaply. The printing"). There is no divergence.
- **Offload.** All 169 `nn.Linear` weights run on the device, each exactly once per forward:
  q / k / v / o / gate / up / down × 24 and `lm_head`, which is tied to `embed_tokens` and shares
  its upload. Graph 0 has 169 matmul nodes on the device and 0 on the CPU. There are 2 graphs (the
  second is the dynamic-length recompile), 0 uploads in compiled calls and 0 CPU fallbacks.
- **Bit-exact.** The device is repeatable. The logits of the 32-token forward and of the last
  generation step (47 tokens) equal eager with `reference_linears()`.
- **llama.cpp-path oracle** on layer 0 `q_proj` (896 × 896, with bias), layer 12 `down_proj`
  (896 × 4864) and `lm_head` (151936 × 896), T = 32, 4,919,296 outputs. The device blocks equal
  ggml's `quantize_row_q8_0_ref`. y has 0 mismatches against libtsr_ref (numcheck B1), the numpy
  spec model (B1b) and `linear_ref`.
- **All weights.** 169 / 169 (15,436,288 blocks) are byte-identical to ggml's conversion.
- **GGUF head.** The head's blocks equal `output.weight` of `models/qwen2.5-0.5b-instruct-q8_0.gguf`
  (the llama.cpp path's GGUF), and libtsr_ref on those GGUF blocks gives the same y.
- **Upload.** 169 weights, 1.98 GB of f32 in, 528.7 MB (504.2 MiB) of Q8_0 written.
- **Device (sim cycle model, T_MAX 1).** 2,155,269 PERF cycles per token over the 32-token forward (MVM 2,093,776), with the
  cycle model of SW-80 (2,142,863 and 2,080,256 before it; the first run of this check (October 2026) had 2,166,289 with the
  MVM cycles of the older model). 169 execs per forward, all program-cache
  hits. 529.6 MB read from device DRAM per token.
- **T_MAX.** The firmware tiles the tokens of a record by T_MAX (§3.3), and a tile reads the weights once. Forward of
  the first 8 prompt tokens (`--forward-only --tokens 8`, the first forward), PERF cycles per token (MVM cycles):

  | `TSR_SIM_TMAX` | 1 | 4 (K4 image) | 8 |
  | --- | --- | --- | --- |
  | cycles per token | 2,233,687 (2,093,776) | 776,816 (597,970) | 585,588 (336,848) |
  | logits | sha256 `4a5e7b24…7f66`, 8 of 8 rows bit-identical | the same | the same |

  These cycles are from the model after SW-80 (default lanes, merged build, 2026-11-09); before it they were 2,213,080 (2,080,256),
  716,870 (594,580) and 478,237 (335,128). The difference is the store and MVM-fill latencies of the card and the QACT divider time that SW-80
  added to the model. The logits did not change. With `TSR_SIM_CARD=k4full` (the K4 image's
  configuration) the 8-token forward takes 793,988 cycles per token, the logits are the same, and the card takes 808 K.

- **Wall time on the simulator.** On the loaded host the 32-token forward took 21.8 s (the CPU ops 1.9 s), the first
  call 59 s (Dynamo, AOT, partition, run), the 16-token generation 536 s (600 token passes; 150 s on the CPU) and the whole
  script 1053 s. These times say nothing about the card.

**Other dtypes** (`examples/hf_compare.py`, simulator T_MAX 1, 16 prompt tokens + 8
generated, scored by eager CPU on the device's sequence; branch `torch-basis`, 2026-11-09; every value equals that of the
first run, October 2026):

| | float32 (bf16 weights widened) | bfloat16 |
| --- | --- | --- |
| matmuls per graph on the device / on the CPU | 169 / 0 | 169 / 0 |
| steps bit-identical to eager with reference linears | 8 / 8 | 8 / 8 |
| KL(eager ‖ device) mean / median / max | 6.5e-3 / 4.1e-3 / 3.7e-2 | 1.3e-2 / 5.2e-3 / 6.8e-2 |
| top-1 (23 positions / 8 generated) | 91.3 % / 100 % | 95.7 % / 100 % |
| weights-only Q8_0: KL mean, top-1 | 4.3e-3, 95.7 % | 8.0e-3, 95.7 % |
| 1-ulp floor KL mean | 3.4e-11 | 0 |
| whole script: wall (loaded host), peak RSS | 252 s, 4.9 GB | 238 s, 3.9 GB |

**GGUF layer tensors** (`--gguf`, informational). Of the 169 weights, only `output.weight`
(= `token_embd`, 4,254,208 blocks) is byte-identical to the device copy. The 168 layer tensors all
differ, because that GGUF was not made from this checkpoint's tensors. For the 4 tensors examined:

- row i of the checkpoint matches GGUF row i best (cosine 0.95 to 1.0 on 64 sampled rows), with
  column-norm ratios of 0.26 to 15;
- the dequantized GGUF tensors are 0.03 to 6.1 (relative L2) from the safetensors weights, against
  0.0053 to 0.0082 for the device's Q8_0 copy.

So the GGUF's layer tensors are per-channel rescaled versions of these weights. Only the
untransformed embedding can be compared byte for byte.

### 6.6 RTL: Qwen2.5-0.5B forward on the Verilated kernel

The Verilated kernel is reachable as `TSR_DEVICE=vkernel` once the two-clock server exists (P5.3).
The model-level check is the `--expect` run of §2. On `TSR_DEVICE=sim`, `hf_check.py --forward-only
--tokens 8` gives logits with sha256 `4a5e7b24…7f66` (8 rows, PASS), the reference for any other
device kind.

### 6.7 Review

A review of `sw/rt`, the `tsr_lower` refactor, `sw/torch` and `hf_check.py` confirmed 8 findings,
each with a run. All 8 are fixed. Their 14 regression tests are in `test_robustness.py`, and all 14
failed on the old code.

1. The upload-churn guard counted per graph node instead of per weight. It is now per (node, root
   weight). Regional compilation of 8 layers, 3 forwards, one shared graph: 24 device calls,
   0 CPU fallbacks, 8 uploads, against 4 / 20 / 4 before.
2. Cached handles are now tied to the device that issued them (`Device.serial`).
3. `open_device(...)` options now stay usable by `tt.linear`, `tt.device()` and `torch.compile`.
4. An unavailable device and a transport error during an upload now raise. Only heap full falls
   back.
5. Closing the device while another thread was inside `linear` crashed with SIGSEGV. Now `close()`
   waits for the call.
6. Concurrent `get` / `open` / `is_open` after a close deadlocked. Fixed.
7. `load_state_dict` / `mul_` on an inference-tensor weight now reach the device.
8. Matmuls in HOP bodies (`nested_compile_region`, `cond`, `while_loop`, `map`) now go to the
   device, bit-exact against the reference.

Also clean in the review:

- tsr_rt lowering: key completeness, N / K parts, IO chunking, error paths.
- Weight identity: no false cache hit after free (0 wrong results in 50 cycles, 10 of them with a
  reused data pointer); `_version` invalidation for [K, N], sliced and in-place-updated weights.
- Activations: empty, 1D, 4D and non-contiguous x in f32 / bf16 / f16.
- Concurrency and lifetime: 4 concurrent threads with 0 mismatches; no device leak after deleting
  a model; exec errors propagate and the device recovers (`TSR_MOCK_INJECT`).
- Build: the torch C++11 ABI matches, and the RUNPATH has torch/lib and build/torch/bin.

### 6.8 Card

Measured on the card, 2026-11-09. Image `tsr_u200_k4`, the full-feature image of the fourth link (xclbin sha256 `6fdf31fd…`,
four channels, `T_MAX` 4, core 200 MHz and RISC-V 100 MHz; `docs/platform-u200.md` §8.8, "Card bring-up of the fourth
link"). Firmware `tsr-fw.elf` of branch `torch-basis` (image id `99698a44030574f4`, Basis 0.1.0, 21 kinds). Host build
`build/xrt` (§2.4). `tsr-selftest --traces traces/at/errors/at_s53_0009` ran before every session (40 checks, all passed).
The host ran other jobs (load average 30 to 60), so wall times include waiting for the CPU; PERF cycles do not.

- `TSR_DEVICE=xrt TSR_TORCH_VERIFY=1`, the pytest suite: **114 passed, 2 skipped** (the sim-only T_MAX 8 device test and the
  opt-in HF test), 176 s. Every device matmul was recomputed with libtsr_ref on the weight rows read back from the card and
  compared bit for bit. The card has no `MVM_LOCAL_SCALE`, so the scaled-result tests take the CPU multiply.
- The opt-in `test_hf.py` (Qwen2.5-0.5B-Instruct, 12 + 4 tokens) passed: 4 of 4 steps bit-identical to eager with reference
  linears, top-1 above 80 %, 332 s.
- `hf_check.py --forward-only --tokens 8 --expect <sim logits>`: **8 of 8 rows bit-identical** to the simulator (T_MAX 1),
  sha256 `4a5e7b24…7f66`, the same as on `sim` (§6.6) and as the first card run of this check. 169 device calls, 169
  matmul nodes on the device, 0 on the CPU, 0 CPU fallbacks.
- `hf_check.py` (the full check of §6.5; 32 tokens, 16 generated): **PASS**. All 32 per-position KL values of the device
  against the CPU, the top-1 flags and the 16 generated tokens are identical to the simulator run of §6.5 (the table there
  is the card's too). 169 of 169 weights are byte-identical to ggml's conversion, and the three sampled linears have 0
  mismatches in 4,919,296 outputs against libtsr_ref, the numpy spec model and `linear_ref`. 169 execs per forward, all
  program-cache hits, 146.8 MB read from device DRAM per token (529.6 MB at T_MAX 1 on the simulator); 2 graphs, 0 uploads in compiled calls, 0 CPU fallbacks. The 16-token generation (600 token passes, a new
  length each step) made 2535 op-list cache misses. 21 min wall; the card was held for it (this run was CPU-bound on the
  loaded host: the CPU generation alone took 534 s).

Device cost against the first card run of this backend (October 2026, the earlier single-channel image: 1 channel, `T_MAX` 1,
300 MHz), both for the forward of 8 tokens with 169 device calls:

| | first card run | now |
| --- | --- | --- |
| channels, `T_MAX`, device clock | 1, 1, 300 MHz | 4, 4, 200 MHz |
| PERF cycles per token (MVM) | 9.06 M (8.83 M) | 0.808 M (0.604 M) |
| device time per token (cycles / clock) | 30.2 ms | 4.04 ms |
| steady 8-token forward, wall | not measured | 0.41 s (0.95 s in the speed run; device calls 0.13 to 0.31 s) |
| first 8-token forward with compilation, wall | 2.8 s | 50 s (Dynamo and AOT on the loaded host; the same step took 31 to 53 s on the simulator) |

The PERF cycles are 11.2 times fewer, and the device time per token 7.5 times shorter (the clock is lower). The cycle factor is
the four channels (4 times) and `T_MAX` 4 (2.9 times: on the simulator the same 8-token forward takes 2.23 M cycles at `T_MAX` 1
and 0.78 M at `T_MAX` 4, §6.5). The simulator's cycle model in the card configuration (`TSR_SIM_CARD=k4full`, SW-80) gives 0.794 M for this
forward and 0.639 M for the 32-token one; the card gives 0.808 M and 0.659 M (1.7 % and 3.0 % more). Before SW-80 the model gave 0.717 M for the 8-token forward.

Steady-state forwards, `hf_check.py --speed 1,8,32` (best of 3, the card, `T_MAX` 4; the values in parentheses are from a second
pass with the PERF counters off, which saves their 12 CSR reads per launch):

| tokens | wall | device calls | ms per token | PERF cycles per token (device ms per token at 200 MHz) |
| --- | --- | --- | --- | --- |
| 1 | 0.37 s (0.49 s) | 0.18 s (0.20 s) | 372 | 3.00 M (15.0) |
| 8 | 0.95 s (1.08 s) | 0.31 s (0.33 s) | 119 | 0.808 M (4.04) |
| 32 | 1.15 s (0.98 s) | 0.47 s (0.41 s) | 36 | 0.659 M (3.29) |

Each forward is 169 launches; the launches of a 1-token forward cost 1.0 ms each in the host (0.18 s of device-call time)
against 89 µs of device cycles, so at T = 1 the host's side of a launch (x in, push, wait, y out; not broken down further),
not the device, sets the time. Switching the PERF counters off made no difference that stands out of the noise of the loaded host.

## 7. Limitations

- **Inference only.** Graphs that need autograd run eagerly on the CPU.
- **Weight matmuls only.** Bias, activations, norms, RoPE, attention and embeddings run on the
  CPU, as in the llama.cpp path's M6. Every device matmul is its own launch of the `graph` kernel
  (169 per Qwen2.5-0.5B forward), with x and y crossing the host link each time. On the card a launch costs about 1 ms in the
  host against 89 µs of device cycles at T = 1 (§6.8).
- **No KV cache.** Generation in `hf_check` and `hf_compare` recomputes the whole sequence per step.
  Every new length is a new program: the 15 generation forwards (lengths 33 to 47) made 2535 device
  calls, all program-cache misses, against 169 hits for the repeated 32-token forward.
- **The CPU part runs as an eager aten graph**, not through Inductor.
- **Host memory.** The torch parameters stay in host memory next to the device copy (for Qwen2.5-0.5B
  in f32, 1.98 GB). On `sim` the device DRAM is host memory too: peak RSS 3.2 GB for hf_check.
- **Formats.** Float weights always become Q8_0. Native Q4_K / Q8_0 blocks need `QuantLinear` (§3.9); no model loader
  makes them from a GGUF file yet.
- **Not mapped.** `bmm` and batched weights (MoE expert stacks) stay on the CPU, and so do
  matmuls in HOP bodies other than `invoke_subgraph`, `cond`, `while_loop` and `map`.
- **Weight identity** is by storage. Writes through `.data`, and inference-tensor changes that miss
  the 64 × 64 sample, are not seen (§3.6).
- **Not run**: the full pytest suite on the Verilated kernels (`vkernel` does not exist yet); the bf16 model
  (`hf_compare.py --dtype bfloat16`) on the card; a whole model made of `QuantLinear` layers (its unit tests ran on the
  simulator and in the card's pytest run); a model with a KV cache.

## 8. Next steps

1. **PyTorch on the M7 kernels (phase 2, not built).** The firmware has the kinds (21; RMS_NORM, ROPE, GLU, the
   element-wise ones, FLASH_ATTN with SET_ROWS_KV, MOE_ROUTE, MUL_MAT_ID, SSM_CONV, GATED_DELTA_NET), but the kind revisions
   are still moving (`fw_kind_revs`, [`versioning.md`](versioning.md)), so the work waits. It would need:
   - an extension API above `tsr_rt::linear`: a graph-level op-list builder (the ggml backend's `tsr_opl_builder`, with its
     hazard analysis and JOINs) and device tensors in IO / AUX, so that activations stay on the device between the ops of a
     layer and one launch runs a layer or a decode step instead of one matmul;
   - pattern matching on the aten graph for each kind (RMSNorm with its weight, RoPE in the HF `rotate_half` form, SiLU × mul,
     residual adds, SDPA with grouped KV heads, the router chain of a MoE layer), the equivalent of `tsr_build_fusion`;
   - PyTorch numerics for the kinds the ggml path takes in another form: the exact tanh GELU (GLU variant `GEGLU`, `UNARY` func
     4, SW-31), `eps` and `inv_n` of the norms, softmax, f32 activations for bf16 models;
   - a device-resident KV cache (`cache_<tag>k_l<n>` planes of AT §4.6, Q8_0) behind a static-cache module, which changes
     the numerics of attention from PyTorch's f32 cache, so its oracle is the simulator's, not eager's;
   - references for the new nodes: `tt.reference_*` equivalents of `reference_linears()` and `TSR_TORCH_VERIFY` for each kind
     (the C references are in `sw/ctrl/ref`).
2. **A loader from GGUF.** `QuantLinear.from_gguf` takes one tensor. A model run from a GGUF file also needs the map from
   GGUF tensor names to modules and the norms, embeddings and RoPE parameters from the same file. The Qwen2.5 GGUF in
   use has layer tensors that are rescaled versions of the checkpoint's (§6.5), so the whole model has to come from
   the GGUF.
3. **Fewer launches in the current design.** Matmuls that read the same activation (q, k, v; gate, up) could share one op list
   and one launch: 97 launches instead of 169 per Qwen2.5-0.5B forward. It needs a multi-weight `linear` in `tsr_rt` and a
   partitioner rule for sibling `mm` nodes. At T = 1 on the card the host path of a launch is 1 ms (§6.8), so this would
   save about 40 % of the device-call time of a decode-like forward.
4. **A PrivateUse1 device.** Registering a `tesseract` device type
   (`torch.utils.rename_privateuse1_backend`, allocator over the WEIGHT heap) would make the device
   copy the parameter itself (`model.to("tesseract")`). That removes the host copy and the
   identity / `_version` cache with its `.data` caveat, and placement errors would surface at
   `.to()`. Eager `F.linear` would then dispatch to the device without torch.compile.
5. **Smaller items.** A KV-cache path (static cache, so that decode is T = 1 with cached op lists), the bf16 model and
   `QuantLinear` on the card, and the full suite on the Verilated kernel once the two-clock server exists.
