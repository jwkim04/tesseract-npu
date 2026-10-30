# PyTorch backend (`torch.compile(model, backend="tesseract")`)

The PyTorch backend runs the weight matmuls of a `torch.compile`d model on a Tesseract device and
leaves everything else to the CPU. The C++ runtime `tsr_rt` below it
runs each device matmul as a one-op op list through the firmware's `graph` kernel (W3a). The
PyTorch move onto the M7 kernels is M8.

> **Scope.** User documentation, as-built record and measurement record of `sw/torch` and `sw/rt`. The
> backend, `tsr_rt` and their tests are implemented. The C++ runtime tests (`rt_sim`,
> `rt_sim_tmax8`, `rt_mock`, `rt_mock_tmax8`) pass on the simulator kinds; the Python suite
> (`torch_pytest`) and `hf_check.py` were last run on the simulator and on the card (`tsr_u200_k1`),
> before the control core was added, with the host issuing the ops. Numbers in §6 are labelled by the
> device that produced them. Order of authority: `docs/spec.md` (§5 Device weight layout, §6 Quantized
> activation layout, §7 Numeric rules) > the simulator (`sw/sim`) > this document. Normative for the
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
  is not available until P5.3; `xrt` ran on the card (§6.8) and
  follows the bitstream with the control core.
- **What goes to the device**: a matmul (`mm`, `addmm`, `linear`, `matmul`, `mv`) with one operand
  that is a parameter, buffer or graph constant. That operand must be a float32 / float16 / bfloat16
  CPU tensor with K % 32 == 0 and a shape the device accepts. Attention, norms, activations,
  embeddings, bias adds and every other op run on the CPU, as in the llama.cpp path's M6
  (`docs/spec.md` §12, ggml backend integration, `supports_op`).
- **Numerics**: the llama.cpp path's. A weight becomes Q8_0 when it is uploaded, with ggml's
  `quantize_row_q8_0_ref`, so its blocks are byte-identical to llama.cpp's load-time conversion
  (`docs/spec.md` §5, *Load-time conversion*). Activations are quantized on the device by QACT A32, and the products are f32 (§3.7).
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
cmake -S . -B build/torch -DCMAKE_BUILD_TYPE=Release -DTSR_WITH_TORCH=ON
cmake --build build/torch -j4 --target tesseract_torch_ext
export PYTHONPATH=$PWD/sw/torch:$PWD/build/torch/python
build/tools/torchvenv/bin/python -c "import tesseract_torch as tt; print(tt.device().description)"
```

- `TSR_WITH_TORCH` (default OFF) adds `sw/torch`. It needs `TSR_WITH_LLAMA=ON`, because `tsr_rt`
  links `ggml-base` of the pinned llama.cpp (§3.2): `-DTSR_WITH_TORCH=ON` with `TSR_WITH_LLAMA=OFF`
  stops with a message. `sw/rt` is added whenever llama.cpp is.
- Torch comes from `TSR_TORCH_PYTHON` (default `build/tools/torchvenv/bin/python` if it exists, else
  `python3`). CMake asks that interpreter's `torch.utils.cpp_extension` for the include and library
  directories and the C++11 ABI flag, and builds a CPU extension against `c10`, `torch`,
  `torch_cpu` and `torch_python`. `find_package(Torch)` is not used: the CUDA wheel's CMake config
  requires a matching CUDA toolkit.
- Output: `build/torch/python/_tesseract_rt.cpython-313-x86_64-linux-gnu.so`, for that
  interpreter's Python only. If `build/torch/python` is not on `sys.path`, the package also looks in
  `$TSR_TORCH_EXT_DIR` and `<repo>/build/torch/python`.
- Measured: 20 s, 1.5 GB peak (one translation unit that includes `torch/extension.h`).

### 2.3 Run and test

```
# pytest suite (TSR_DEVICE=sim unless set)
build/tools/torchvenv/bin/python -m pytest -q sw/torch/tests
cd build/torch && ctest -R '^(torch_pytest|rt_)'        # sim T_MAX 1 / 8, rt on sim and mock
# opt-in Hugging Face test
TSR_TORCH_HF_MODEL=models/hf/Qwen2.5-0.5B-Instruct build/tools/torchvenv/bin/python -m pytest -q sw/torch/tests/test_hf.py
# end-to-end check of a Hugging Face causal LM (§6.5); -I because the model directory is data
build/tools/torchvenv/bin/python -I sw/torch/tools/hf_check.py models/hf/Qwen2.5-0.5B-Instruct \
    [--download Qwen/Qwen2.5-0.5B-Instruct] [--tokens 32] [--gen 16] [--gguf GGUF] [--json OUT]
# compiled vs eager on a Hugging Face model, f32 or bf16 (§6.5)
build/tools/torchvenv/bin/python -I sw/torch/examples/hf_compare.py models/hf/Qwen2.5-0.5B-Instruct --dtype bfloat16
```

`TSR_DEVICE=mock` runs the same suite on the adapter model of the transport, and `TSR_SIM_TMAX=8`
gives the simulated core `T_MAX` 8.

`hf_check.py --forward-only --tokens 8 --expect sim.npy` compares the forward's logits bit for bit with
a `TSR_DEVICE=sim` run saved with `--save-logits`. That is the model-level check of `docs/spec.md` §17
(same model run, bit-identical logits), done from PyTorch. The same `--expect` check runs against any
device kind; `TSR_DEVICE=vkernel` is not available until the two-clock server exists (P5.3).

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
  `LD / QACT / MVM / ST` tile ops.
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
`read_blocks`, `info`, `free`, `counters`, `memory`, `unsupported_reason(dtype, N, K)`, `close()`, and
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
| norms, activations, RoPE, softmax, embeddings, residuals, bias adds | CPU | the vector unit and the M7 kernels exist on the simulator, but this backend only emits MUL_MAT records until M8 |
| graphs that need autograd | CPU, eager | one warning; `strict=True` raises |
| WEIGHT heap full, or 4 uploads in a row | CPU (that weight, that node) | one warning, `cpu_fallback_calls` |
| native Q8_0 / Q4_K ggml blocks | Tesseract | extension API `Device.upload_blocks` + `linear`; the backend itself only sees float weights |

## 5. Environment variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `TSR_DEVICE` | `sim` | transport: `sim`, `mock`, `vkernel`, `xrt`; with their own variables (`TSR_SIM_*`, `TSR_MOCK_*`, `TSR_VKERNEL_*`, `TSR_XRT_*`, `TSR_IMAGE`, `TSR_TRACE`, `TSR_LOG`: [`sw/transport/README.md`](../sw/transport/README.md), [`docs/getting-started.md`](getting-started.md#environment-variables)) |
| `TSR_TORCH_DEBUG` | 0 | `1`: partition of every compiled graph, every upload, counters at exit; `2`: also every device call; `3`: also the rewritten aten graphs |
| `TSR_TORCH_VERIFY` | 0 | `1`: every device matmul is checked bit for bit against libtsr_ref on the read-back weight rows; a mismatch raises, naming the weight, the element and both values |
| `TSR_TORCH_EXT_DIR` | — | where to find `_tesseract_rt` if it is not on `sys.path` |
| `TSR_TORCH_PYTHON` | venv, else `python3` | CMake cache variable: interpreter whose torch the extension is built against |
| `TSR_TORCH_HF_MODEL` | — | model directory for the opt-in `test_hf.py` |
| `TSR_RT_DEBUG` | 0 | tsr_rt's log: `1` uploads, calls and launches |

`tt.compile(model, strict=False, debug=None, verify=None, offload=True, **kw)` passes the same
switches as `options=` of `torch.compile`. `offload=False` prints the partition but keeps every
matmul on the CPU.

## 6. Tests and measured results

§6.1 to §6.7 were measured on the simulator (build trees `build/rt` and `build/torch`, Release,
torch 2.9.1, Python 3.13, `TSR_SIM_THREADS=4` in the ctests, on the 16-core / 14 GB dev host).
"Program" and "program cache" in those sections mean the op list and the op-list cache. §6.1 ends
with the re-run of the runtime tests; §6.8 has the card.

### 6.1 tsr_rt (`test-tsr-rt`)

Every case checks four things. `read_blocks` must equal ggml's blocks byte for byte. `linear` and
`linear_ref` must equal an independent model (ggml conversion + libtsr_ref) bit for bit. The y
padding must be untouched. The NMSE against an f64 product of the dequantized weight must stay
< 1e-3 (measured max 7.89e-5). Shapes: f32 / f16 / bf16 → Q8_0 and native Q8_0 / Q4_K; K 32 … 4864
and 33024 / 33280 (> K_MAX, 2 K parts); N 1 … 4864; T 1 … 20; forced N parts (2 … 7), K parts
(4 … 5) and both (4 × 4); strided x / y; padded and misaligned weight rows.

| Run | Result |
| --- | --- |
| `rt_sim` (T_MAX 1) | 43 / 43 cases, 0.42 s |
| `rt_sim_tmax8` (`--big`) | 44 / 44, 2.11 s; includes Q8_0 N = 130000, K = 896, T = 9, split automatically into 2 N parts, bit-exact |
| `rt_mock` (DMA align 64, chunk 64 KiB), `rt_mock_tmax8` (3 ch, T_MAX 8, align 4096, chunk 12 KiB) | 43 / 43 each |
| op-list cache, traffic | see §3.3; PERF_CYCLES of that call 1753 on sim |
| IO split, free / re-upload, refusals, threads | a 64 KiB IO area with T = 40 gives 3 programs, bit-exact; memory fully returned after free, double free rejected; K = 48, Q4_K with K = 288, N = 0 and `max_part_k = 48` rejected; 2 threads × 12 calls bit-exact |
| mutation check (scratch builds of tsr_rt.cpp) | detected: wrong QACT column offset in K parts (12 failed checks), dropped ACC (12), shifted N-part y gather (10), K part converted from column 0 (24), wrong x stride (6); an equivalent mutant (N part + 1 row) still passes, as it should |
| build | fresh configure with `TSR_WITH_LLAMA=ON` and `OFF` both succeed (`tsr_lower` builds without llama.cpp); `libtsr_rt.so` linked alone runs a linear on sim with y == `linear_ref`; no compiler warnings |

Re-run with the op-list runtime (`tsr_rt` launching the `graph` kernel):
`ctest --test-dir build/dev -R '^rt_'` passes on the simulator kinds: `rt_sim` 43 / 43 cases,
`rt_sim_tmax8` 44 / 44 (with the N = 130000 head), `rt_mock` 43 / 43, `rt_mock_tmax8` 43 / 43, the same
assertions (measured on the simulator, `a775222`; 3 s for the four).
`rt_vkernel_k4t1` (opt-in, `-DTSR_RT_VKERNEL_TEST=ON`) reports "skipped" until P5.3.

### 6.2 pytest suite (`sw/torch/tests`)

| File | Content |
| --- | --- |
| `test_ext.py` | extension, 11 shapes (N 1 … 4864, K 32 … 4864, T 1 … 40) × 3 dtypes: stored blocks == an independent numpy `quantize_row_q8_0_ref` byte for byte (`spec_model.py`), `linear` == `linear_ref` == the numpy spec model of numcheck.py (B1b) bit for bit; directed rounding rows; the N = 151936 head (2048 sampled rows); K > K_MAX; strided / transposed / 2-byte-misaligned operands; T = 1 … 300 with exact host traffic; a T_MAX 8 sim device; native Q8_0 / Q4_K; errors |
| `test_numerics.py` | NMSE against torch float32, and the single-output exception (§6.3) |
| `test_backend.py` | compiled MLP (f32 / f16 / bf16) and pre-norm transformer block (q / k / v / o, SDPA, SwiGLU; f32 / bf16, with verify) vs eager; fallbacks; re-upload; program cache; B·S > T_MAX; autograd refusal; operand forms; dynamic lengths; tied weights, graph breaks; debug output; close / reopen; heap-full fallback |
| `test_robustness.py` | one regression test (or group) per review finding (§6.7) |
| `test_hf.py` | opt-in, Qwen2.5-0.5B-Instruct, 12 + 4 tokens |

| Run (final code) | Result | Wall | Max RSS |
| --- | --- | --- | --- |
| `TSR_DEVICE=sim` | 86 passed, 1 skipped (the opt-in HF test) | 12.5 s | 1.80 GB |
| `TSR_DEVICE=sim TSR_SIM_TMAX=8` | 86 passed, 1 skipped | 11.3 s | 1.80 GB |
| `TSR_DEVICE=mock`, DMA align 64, chunk 64 KiB | 85 passed, 2 skipped (+ the sim-only T_MAX 8 device test) | 11.1 s | 1.80 GB |
| `TSR_DEVICE=mock`, T_MAX 8, 3 ch, align 4096, chunk 12 KiB | 85 passed, 2 skipped | 10.8 s | 1.80 GB |
| opt-in `test_hf.py` (sim, after the review fixes) | 1 passed | 33.9 s | |
| ctest in `build/torch` | the 14 Tesseract tests pass: `torch_pytest` (13.7 s), `torch_pytest_tmax8` (13.8 s), `rt_sim`, `rt_sim_tmax8`, `rt_mock`, `rt_mock_tmax8`, `ggml_tesseract_units`, `_e2e`, `_e2e_tmax8`, `_e2e_hosted`, `_e2e_hosted_tmax8`, `_scalar_nmse`, `transport_ggml_e2e_mock`, `_mock_tmax8` (`test-backend-ops` is not built in that tree) | | |

### 6.3 Bit-exactness and NMSE

- **Device == reference == spec model.** 11 shapes × f32 / f16 / bf16: `linear` == `linear_ref`
  == the numcheck.py numpy spec model, bit for bit. The stored blocks equal an independent numpy
  `quantize_row_q8_0_ref`, byte for byte. The same holds for the N = 151936 head (2048 sampled
  rows), K > K_MAX (2 K parts), a T_MAX 8 sim device and native Q4_K.
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

- No `sw/ggml-tesseract` source or CMake file changed after the `tsr_lower` refactor (§3.2), so the
  trace diff below still applies. The top-level CMake change for torch is behind `TSR_WITH_TORCH`,
  which defaults to OFF.
- **Trace diff.** `TSR_TRACE` dumps from the unmodified and the modified build were compared, for
  `test-ggml-tesseract` on sim, tmax8, hosted and hosted_tmax8: 98 + 98 + 5564 + 5564 traces, every
  file byte-identical (program, init DRAM including the weights, expect). For
  `test-backend-ops -o MUL_MAT -b Tesseract0`: 1418 traces, 0 program differences, 1402 / 1402
  passed in both builds. (Measured on the simulator, with traces of the earlier op format, see `traces/README.md`.)
- **After the review fixes**, ctest in `build/rt` passed 11 / 11 in 38.9 s: `rt_*` (4) and
  `ggml_tesseract_*` (7, including `_backend_ops` with 1402 / 1402). `test-ggml-tesseract`:
  "all 28 cases OK".

### 6.5 Qwen2.5-0.5B-Instruct (Hugging Face safetensors)

Model: `models/hf/Qwen2.5-0.5B-Instruct`, Hub revision 7ae55760. `model.safetensors` (988 MB,
sha256 `fdf756fa…`) matches the Hub's LFS etag. There are no pickle files, and
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

The run below is on the final code: PASS, 88 s wall, 5.0 GiB peak RSS. Its per-position KL values,
generated tokens and PERF cycles are identical to two runs made before the review fixes.

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
- **Upload.** 169 weights, 1.98 GB of f32 in, 528.7 MB (504.2 MiB) of Q8_0 written, 1.18 s.
- **Device (sim cycle model).** 2,166,289 PERF cycles per token (MVM 2,080,256). 169 execs per
  forward, all program-cache hits. 529.7 MB read from device DRAM per token.
- **Wall time on the simulator.** 2.26 s per 32-token forward, of which the CPU ops take 0.02 s. CPU
  eager takes 0.25 s. The first call (Dynamo + AOT + partition + run; the weights were uploaded
  before it) takes 4.64 s. Generation takes 48.6 s on the device (600 token passes) and 3.0 s on
  the CPU. These are simulator times and say nothing about the card.

**Other dtypes** (`examples/hf_compare.py`, before the review fixes, 16 prompt tokens + 8
generated, scored by eager CPU on the device's sequence):

| | float32 (bf16 weights widened) | bfloat16 |
| --- | --- | --- |
| matmuls per graph on the device / on the CPU | 169 / 0 | 169 / 0 |
| steps bit-identical to eager with reference linears | 8 / 8 | 8 / 8 |
| KL(eager ‖ device) mean / median / max | 6.5e-3 / 4.1e-3 / 3.7e-2 | 1.3e-2 / 5.2e-3 / 6.8e-2 |
| top-1 (23 positions / 8 generated) | 91.3 % / 100 % | 95.7 % / 100 % |
| weights-only Q8_0: KL mean, top-1 | 4.3e-3, 95.7 % | 8.0e-3, 95.7 % |
| 1-ulp floor KL mean | 3.4e-11 | 0 |
| whole script: wall, peak RSS | 92 s, 4.7 GB | 80 s, 3.8 GB |

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

Measured on the card (k1, 300 MHz; `tsr_u200_k1`, one channel, `T_MAX` 1; `docs/platform-u200.md`
§14.1):

- `TSR_DEVICE=xrt TSR_TORCH_VERIFY=1`, the pytest suite: **85 passed, 2 skipped** (the sim-only and
  model-path tests), 12.1 s. `TSR_TORCH_VERIFY=1` recomputes every device matmul with libtsr_ref on
  the weight rows read back from the card.
- The opt-in `test_hf.py` (Qwen2.5-0.5B-Instruct) passed.
- `hf_check.py --forward-only --tokens 8 --expect <sim logits>`: **8 / 8 rows bit-identical** to the
  simulator, sha256 `4a5e7b24…7f66`, the same as on `sim` (§6.6). The
  forward made 169 device calls (169 matmul nodes on the device, 0 on the CPU, 0 CPU fallbacks); the
  PERF counters read 9.06 M cycles per token, 8.83 M of them in the MVM (one channel); the 8-token
  forward took 2.8 s including the compilation, the whole script 6.3 s.

PyTorch has not run on a card with the control core: the card path is work package W6.

## 7. Limitations

- **Inference only.** Graphs that need autograd run eagerly on the CPU.
- **Weight matmuls only.** Bias, activations, norms, RoPE, attention and embeddings run on the
  CPU, as in the llama.cpp path's M6. Every device matmul is its own launch of the `graph` kernel
  (169 per Qwen2.5-0.5B forward), with x and y crossing the host link each time.
- **No KV cache.** Generation in `hf_check` and `hf_compare` recomputes the whole sequence per step.
  Every new length is a new program: the 15 generation forwards (lengths 33 to 47) made 2535 device
  calls, all program-cache misses, against 169 hits for the repeated 32-token forward.
- **The CPU part runs as an eager aten graph**, not through Inductor.
- **Host memory.** The torch parameters stay in host memory next to the device copy (for Qwen2.5-0.5B
  in f32, 1.98 GB). On `sim` the device DRAM is host memory too: peak RSS 5.0 GiB for hf_check.
- **Formats.** Float weights always become Q8_0. Native Q4_K / Q8_0 blocks are reachable only
  through the extension (`upload_blocks`), not from a model.
- **Not mapped.** `bmm` and batched weights (MoE expert stacks) stay on the CPU, and so do
  matmuls in HOP bodies other than `invoke_subgraph`, `cond`, `while_loop` and `map`.
- **Weight identity** is by storage. Writes through `.data`, and inference-tensor changes that miss
  the 64 × 64 sample, are not seen (§3.6).
- **Not run**: the Python suite and `hf_check.py` since the control core was added (only the C++
  runtime tests ran, §6.1); the full pytest suite on the Verilated kernels;
  a model forward at T_MAX 8 (the k4 kernel or `TSR_SIM_TMAX=8`); the bf16 model on the RTL or the
  card; PyTorch on the card with the k4 kernel (the card run of §6.8 used k1).

## 8. Next steps

1. **Card with the control core.** On the U200, run the pytest suite and `hf_check --forward-only --expect`
   against saved sim logits with `TSR_DEVICE=xrt`, as the M6 gate does for llama.cpp (work package
   W6, [`docs/xrt-host.md`](xrt-host.md)). The card run so far is §6.8.
2. **PyTorch on the kernels (M8).** With the M7 kernels (RMS_NORM, ROPE, GLU, element-wise kinds,
   flash attention, KV store) on the simulator, the partitioner can take a matmul's bias add, SiLU ×
   mul, residual and norm into the same op list and keep activations in IO between the ops of a layer.
   That replaces one launch per matmul with one per layer or more. The aten-graph pattern matching is
   the place for this, and `TSR_TORCH_VERIFY` would check the new ops against the reference model.
   PyTorch uses the exact tanh GELU (the GLU variant `GEGLU`, `UNARY` `GELU`), not the fp16-table form
   that the llama.cpp graphs use (SW-31). The plan for M8 is in [`docs/design/README.md`](design/README.md)
   §2 and §10.
3. **Q4_K (and Q8_0) from GGUF tensors.** Load a GGUF's own blocks with `upload_blocks`, so that a
   Q4_K_M file runs from PyTorch with the bandwidth of the llama.cpp path. `upload_blocks` is
   already bit-exact on sim and mock. This needs a loader that maps GGUF tensor names to
   modules and a weight representation the partitioner recognises (a module swap or a tensor
   subclass). Because the Qwen2.5 GGUF's layer tensors are rescaled (§6.5), the whole model,
   including the norms, has to come from the GGUF, not just the matmul weights.
4. **A PrivateUse1 device.** Registering a `tesseract` device type
   (`torch.utils.rename_privateuse1_backend`, allocator over the WEIGHT heap) would make the device
   copy the parameter itself (`model.to("tesseract")`). That removes the host copy and the
   identity / `_version` cache with its `.data` caveat, and placement errors would surface at
   `.to()`. Eager `F.linear` would then dispatch to the device without torch.compile.
5. **Kernel selection by pattern.** Device calls are launches of firmware kernels since W3a. A
   torch.compile backend could select precompiled kernels from the image's registry by pattern
   (linear with GGUF weights, RMSNorm, SiLU × mul, SDPA) and leave the rest eager on the CPU; this
   backend's partitioner, weight cache and verification are the pieces it would reuse
   ([`docs/ctrl-riscv.md`](ctrl-riscv.md) §8.6 names it as a JIT target). It is an idea, not
   scheduled.
6. **Smaller items.** A KV-cache path (static cache, so that decode is T = 1 with cached op lists),
   the k4 kernel (T_MAX 8) for model prefill, the Python suite in the ctest list
   (`torch_pytest`, `torch_pytest_tmax8` with `-DTSR_WITH_TORCH=ON`), and the full suite on the
   Verilated kernel once the two-clock server exists.
