# PyTorch backend (`torch.compile(model, backend="tesseract")`)

The PyTorch backend runs the weight matmuls of a `torch.compile`d model on a Tesseract device and leaves
everything else to the CPU. The C++ runtime `tsr_rt` below it runs each device matmul as a one-op op list
through the firmware's `graph` kernel. This document covers usage, architecture and tests of `sw/torch` and
`sw/rt`. The package reference is [`sw/torch/README.md`](../sw/torch/README.md) and the runtime's is
[`sw/rt/README.md`](../sw/rt/README.md). Order of authority: `docs/spec.md` (§5 Device weight layout, §6 Quantized
activation layout, §7 Numeric rules), then the simulator (`sw/sim`), then this document. Normative for the
software split: [SW](design/06-software-verification.md) §9.4.

## 1. What it is

`import tesseract_torch` registers the torch.compile backend `"tesseract"`. A model compiled with it runs
inference with its weight matmuls on Tesseract and everything else on the CPU:

```python
import torch, tesseract_torch as tt

model = torch.compile(model, backend="tesseract")   # or tt.compile(model)
with torch.no_grad():                                # or torch.inference_mode()
    y = model(x)
```

- Device. `TSR_DEVICE` picks the transport (`sim` by default; `mock`, `vkernel`, `xrt` through the same
  `tsr_transport` as the llama.cpp path). `xrt` runs the release image `tsr_u200_k4`. The runtime boots the
  firmware image (`$TSR_IMAGE`, else the build's `tsr-fw.elf`) when it opens the device.
- What goes to the device. A matmul (`mm`, `addmm`, `linear`, `matmul`, `mv`) with one operand that is a
  parameter, buffer or graph constant, a float32, float16 or bfloat16 CPU tensor with K % 32 == 0 and a shape the
  device accepts. Attention, norms, activations, embeddings, bias adds and every other op run on the CPU, as in
  the llama.cpp path's matmul mode. A model that provides Q8_0 or Q4_K blocks uses `tt.QuantLinear` instead of
  `nn.Linear` (§3.9).
- Numerics. The llama.cpp path's. A float weight becomes Q8_0 at upload with ggml's `quantize_row_q8_0_ref`, so
  its blocks are byte-identical to llama.cpp's load-time conversion (`docs/spec.md` §5). Activations are
  quantized on the device by QACT A32 (A256 for Q4_K), and the products are f32 (§3.7).
- Equivalence. With the CPU ops unchanged, a compiled model gives exactly eager PyTorch's results with every
  supported `nn.Linear` replaced by the device's reference model (`tt.reference_linears()`). The tests check this
  bit for bit.
- Not supported: training (graphs that need autograd run eagerly on the CPU), a torch device type (tensors stay
  CPU tensors; there is no `.to("tesseract")`), and Inductor (the CPU part runs as an eager aten graph).

## 2. Build and run

### 2.1 Python environment

The extension and the tests use a venv on top of a base interpreter that has torch:

```
<base python3> -m venv --system-site-packages build/tools/torchvenv
build/tools/torchvenv/bin/pip install transformers safetensors huggingface_hub pytest
```

Model files go under `models/`. Only safetensors are used, never pickle `.bin` files, and `trust_remote_code` is
always off.

### 2.2 Build

```
third_party/apply-patches.sh                          # once per checkout
cmake -S . -B build/dev -DCMAKE_BUILD_TYPE=Release -DTSR_WITH_TORCH=ON
cmake --build build/dev -j8                           # everything; --target tesseract_torch_ext for the extension alone
export PYTHONPATH=$PWD/sw/torch:$PWD/build/dev/python
build/tools/torchvenv/bin/python -c "import tesseract_torch as tt; print(tt.device().description)"
```

`TSR_WITH_TORCH` (default OFF) adds `sw/torch` and needs `TSR_WITH_LLAMA=ON`, because `tsr_rt` links `ggml-base` of
the pinned llama.cpp (§3.2). Torch comes from `TSR_TORCH_PYTHON` (default `build/tools/torchvenv/bin/python`, else
`python3`). CMake asks that interpreter's `torch.utils.cpp_extension` for the include and library directories and
builds a CPU extension against `c10`, `torch`, `torch_cpu` and `torch_python`. `find_package(Torch)` is not used,
because a CUDA wheel's CMake config requires a matching CUDA toolkit. The module
`<build>/python/_tesseract_rt.cpython-<ver>-x86_64-linux-gnu.so` is for that interpreter's Python only. If
`<build>/python` is not on `sys.path`, the package also looks in `$TSR_TORCH_EXT_DIR`, `<repo>/build/torch/python`
and `<repo>/build/dev/python`.

### 2.3 Run and test

```
build/tools/torchvenv/bin/python -m pytest -q sw/torch/tests          # TSR_DEVICE=sim unless set
ctest --test-dir build/dev -R '^(torch_pytest|rt_)'                   # sim T_MAX 1 / 4 / 8, rt on sim and mock
build/tools/torchvenv/bin/python -I sw/torch/tools/hf_check.py models/hf/Qwen2.5-0.5B-Instruct   # §6
```

`TSR_DEVICE=mock` runs the same suite on the adapter model of the transport, and `TSR_SIM_TMAX=4` gives the simulated
core the `T_MAX` of the release image. `-I` ignores `PYTHONPATH`, so the tools find the extension through
`TSR_TORCH_EXT_DIR` or the build trees of §2.2. `hf_check.py --forward-only --tokens 8 --expect sim.npy` compares the
forward's logits bit for bit with a `TSR_DEVICE=sim` run saved with `--save-logits`: the model-level check of
`docs/spec.md` §17 (same model run, bit-identical logits), done from PyTorch, against any device kind.

### 2.4 On the card

```
cmake -S . -B build/xrt -DCMAKE_BUILD_TYPE=Release -DTSR_WITH_TORCH=ON -DTSR_WITH_XRT=ON
cmake --build build/xrt -j8 --target tesseract_torch_ext tsr-selftest tsr_fw numcheck_helper
source /opt/xilinx/xrt/setup.sh
export TSR_DEVICE=xrt TSR_XCLBIN=<build/u200/k4/tsr_u200_k4.xclbin> TSR_IMAGE=$PWD/build/xrt/fw/tsr-fw.elf
export TSR_TORCH_EXT_DIR=$PWD/build/xrt/python PYTHONPATH=$PWD/sw/torch:$PWD/build/xrt/python
TSR_TORCH_VERIFY=1 build/tools/torchvenv/bin/python -m pytest -q sw/torch/tests
```

The card runs one process at a time. `TSR_TORCH_VERIFY=1` recomputes every device matmul with libtsr_ref on the
weight rows read back from the card and compares it bit for bit. The release image has no `MVM_LOCAL_SCALE`, so
the scaled-result tests take the CPU multiply (§3.5).

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

### 3.2 Shared code: `tsr_lower` and the load-time conversion

The two lines marked "shared" in §3.1 are the code the ggml backend uses: both paths build a weight matmul as an op
list with the same builder and launcher and store the same weight bytes. `tsr_lower` is a static PIC library with
no ggml dependency (shape checks and SPM tiling, the op list builder with its hazard analysis and JOINs, the OPL
cache, the `graph` launcher, the allocator, the row repack and the scatter and gather over the channels). Its files
stay in `sw/ggml-tesseract`, and `lower.cmake` defines the target. The f32, f16 and bf16 to Q8_0 conversion is the
backend's `tsr-weights.cpp`, compiled into `tsr_rt`. libtsr_ref's `tsr_qact` is not used for it: it differs from
`quantize_row_q8_0_ref` on blocks with subnormals or a tiny maximum (DAZ and FTZ, `docs/spec.md` §21) and writes a
QA image with an f32 scale, not `block_q8_0`. Only ggml's function gives "the same blocks as llama.cpp".

### 3.3 The runtime `tsr_rt` (sw/rt)

`tsr::rt::device` (`sw/rt/tsr_rt.h`) opens `TSR_DEVICE`, checks the configuration, and offers `upload`, `free_weight`,
`info`, `read_blocks`, `linear`, `linear_ref`, counters and `memory`. Every member function takes one device mutex.
Device memory: OPL (ch0, 64 MiB) holds the cached op lists, IO (ch0, 256 MiB) the x and y rows of one launch, and
WEIGHT (every channel, from 512 MiB) the weights with symmetric allocation (same W_OFF on every channel, 4 KiB
aligned). Uploads convert f32, f16 and bf16 to Q8_0 (§3.2) or take native Q8_0 and Q4_K blocks (`upload_blocks`),
repack the rows into the device layout (`docs/spec.md` §5) and scatter them over the channels.

- `linear(h, x [T, K] f32) → y [T, N] f32`. x is staged in IO. The call builds one op list with a MUL_MAT record per
  N part and K part and launches the `graph` kernel once per IO-sized chunk of T. The firmware tiles T by
  min(T_MAX, its scratch ring), so the runtime does not tile by T_MAX itself.
- N parts. A weight is split into row ranges when x + QA + y of T_MAX tokens would not fit the SPM. Rows are
  independent, so the result is bit-identical to an unsplit weight.
- K parts. When K > K_MAX (32768) the columns are cut into equal block-multiple parts. The first record of an N part
  writes y, and the others carry `ACC_DST`: y = ((y0 + y1) + y2)..., one f32 add each (`docs/spec.md` §21, *ACC
  addition*). This differs from the unsplit dot product, and `linear_ref` models it exactly.
- Op-list cache. The key is the bytes of the list (formats, shapes, T, the x and y handles, every part's W_OFF). An
  LRU in the OPL region keeps the lists resident, so a repeated call writes x, launches and reads y.

### 3.4 The extension `_tesseract_rt`

`_tesseract_rt.Device` binds `tsr::rt::device` for CPU torch tensors and releases the GIL during device calls. Each
method holds a lease that keeps the device alive until it returns, so `close()` from another thread waits for the
calls in flight, and a later call raises `TesseractError: ... the device is closed`. The API is in
[`sw/torch/README.md`](../sw/torch/README.md).

### 3.5 The partitioner (backend.py)

1. Dynamo to AOTAutograd. The backend wraps `aot_autograd` with one compiler for the forward and inference graphs. A
   graph that needs autograd (grad mode on and an input that requires grad) is not partitioned. It runs eagerly with
   one `RuntimeWarning`, and `strict=True` raises instead. The device is opened at the first compilation, so an
   unavailable device fails `torch.compile`.
2. Weights are found from AOT's input descriptors (`ParamAOTInput`, `BufferAOTInput`, or a `PlainAOTInput` of a
   Dynamo input that is an `nn.Parameter` or comes from a module's `_parameters` or `_buffers`) and from graph
   constants (`get_attr`), possibly seen through view ops. A tensor computed in the graph is not a weight.
3. Eligible nodes are `aten.mm`, `addmm`, `linear`, `matmul` and `mv` with one weight operand and one non-weight
   operand. After orientation the weight must be a 2D [N, K] CPU tensor of a supported dtype with K % 32 == 0 that
   the device accepts (`unsupported_reason`). Both orders work: `x @ Wᵀ` (nn.Linear), `x @ W` with a [K, N] weight
   (the uploaded copy is Wᵀ), and `W @ x` or `mv(W, v)`, computed as (xᵀ Wᵀ)ᵀ. Batch and sequence dims are
   flattened into T.
4. Each eligible node becomes a call of `device_mm(spec_id, a, b, bias, root)`, where `root` is the parameter or
   buffer the weight operand is a view of. A bias-free `mm`, `matmul` or `linear` with the weight on the right,
   float32 output and one user that multiplies by a positive power of two in [1, 2^127] becomes one `device_mm`
   with a `scale`. With `MVM_LOCAL_SCALE` and a weight of one K part the kernel applies the scale (MUL_MAT_SCALE).
   Otherwise, and under `TSR_TORCH_VERIFY`, the scale is the CPU multiply after the ordinary `linear`. Both give
   the same bits.
5. Matmuls in the bodies of `invoke_subgraph`, `cond`, `while_loop` and `map` are partitioned recursively (up to 8
   levels). A body placeholder is a weight if the op passes it a weight at every call site, so a body shared by all
   layers is one device node that sees every layer's weight. Other HOP bodies stay on the CPU and are reported.
6. The run-time state of a device node (uploads in a row, CPU fallback) is kept per (node, root weight), because one
   graph can serve many weights (regional compilation, a compiled helper called per layer, a shared HOP body).

`TSR_TORCH_DEBUG=1` prints the partition of every graph and, for each matmul that stays on the CPU, why.

### 3.6 Weight cache

`WeightCache` (runtime.py) keys a device copy by tensor identity: the storage (protected from address reuse by a
`StorageWeakRef`), the data pointer, storage offset, shape, stride and dtype. A hit also needs the same `_version`
and the device the copy was uploaded to (`Device.serial`). An in-place update (`mul_`, `copy_`,
`load_state_dict`) bumps `_version`, so the next call uploads the weight again. A weight is uploaded on first use
and shared by every graph that uses it (recompiles, graph breaks, tied weights). Inference tensors have no
version counter, so for them a hit also needs the same raw bits in a sample of up to 64 x 64 elements taken at
upload. Copies whose storage died are released on the next miss, by `sweep()` or by `tt.release_weights()`.
Writes through `.data` do not bump `_version`, and an inference-tensor change can miss every sampled element.
Call `release_weights()` after either.

### 3.7 Numerics

| Step | Where | Rule |
| --- | --- | --- |
| Weight f32, f16, bf16 to Q8_0 | host, at upload | f16 and bf16 widened exactly, then `quantize_row_q8_0_ref`, repacked into the device layout |
| Native Q8_0 and Q4_K blocks | host, at upload | no conversion, only the repack (`upload_blocks`) |
| Activation x | host | widened exactly to f32 if it is f16 or bf16 |
| QACT A32 | device | `docs/spec.md` §6 (equal to `quantize_row_q8_0_ref`); A256 for Q4_K weights |
| MVM | device | `docs/spec.md` §7: exact integer block sums, 16 partials, fixed tree, no FMA; y in f32 |
| `alpha`, `beta * input`, `linear` bias | host | f32 (`out.mul_(alpha)`, `out.add_(bias, alpha=beta)`); `beta = 0` ignores the input, as in torch |
| Output | host | one cast from f32 to the op's output dtype (the input dtype) |

The result equals the llama.cpp path for the same weight and activation rows. Against torch's float32 matmul the
device has the error of 8-bit weights and 8-bit activations, as llama.cpp's Q8_0 path does.

### 3.8 Errors, fallbacks, threads

Only placement falls back to the CPU: a weight that cannot be placed (WEIGHT heap full, after releasing dead
weights and retrying) runs on the CPU at that node from then on (one warning, counter `cpu_fallback_calls`). So
does a weight uploaded on 4 consecutive calls of a node, because it was changed in place before each call or is
seen through an op that copies it. Everything else is a `TesseractError`: a device that cannot be opened, a
transport error during an upload, a device execution error. Device calls release the GIL, and tsr_rt serializes
them on its mutex.

### 3.9 Weights as ggml blocks (`QuantLinear`)

`tt.QuantLinear(blocks, in_features, fmt, bias=None)` holds a uint8 buffer [N, row_bytes] of Q8_0 or Q4_K blocks and
calls the custom op `tesseract::linear_blocks(x, blocks, fmt, K)` (`tesseract_torch/quant.py`).
`QuantLinear.from_gguf(t)` builds one from a `gguf.ReaderTensor`. The blocks are uploaded as they are, with the
cache key of §3.6 plus the format and K. QACT A32 (Q8_0) or A256 (Q4_K) runs on the device and the result is f32.
The module adds the bias in float32 and casts once to the input's dtype. Under `torch.compile` the op has a fake
implementation, so the partitioner does not touch it. A weight that cannot be placed raises `TesseractError`.
There is no loader that maps GGUF tensor names to modules and takes the norms, embeddings and RoPE parameters from
the same file (SW-81).

## 4. Supported ops and fallbacks

| Case | Runs on | Note |
| --- | --- | --- |
| `nn.Linear`, `F.linear`, `x @ W.T`, `addmm` (bias, alpha, beta), `x @ W` ([K, N] weight), `W @ x`, `mv(W, v)`, also inside `nested_compile_region`, `cond`, `while_loop` and `map` bodies | Tesseract | weight f32, f16 or bf16, K % 32 == 0, 2D after orientation, CPU tensor |
| native Q8_0 and Q4_K ggml blocks | Tesseract | `tt.QuantLinear` or `tt.linear_blocks` (§3.9), or `Device.upload_blocks` and `linear` |
| bias-free matmul followed by a positive power-of-two scalar multiply | Tesseract (the scale in the kernel when the device has `MVM_LOCAL_SCALE`, else on the CPU) | §3.5 |
| weight with K % 32 != 0, float64 weight | CPU | reported by `TSR_TORCH_DEBUG=1`; bit-identical to eager |
| weight on another device (CUDA, say) | not offloaded | the op runs as in eager |
| `bmm`, `baddbmm`, `addmv`, `dot`, `einsum`, `_scaled_mm`; batched weights (MoE expert stacks) | CPU | not mapped |
| matmul of two activations (attention scores, SDPA) | CPU | neither operand is a weight |
| a weight computed in the graph (autocast `_to_copy`, dequantization) | CPU | not a parameter, buffer or constant |
| matmuls in other HOP bodies | CPU | reported as CPU nodes |
| norms, activations, RoPE, softmax, embeddings, residuals, bias adds | CPU | the firmware has kinds for them, but this backend only emits MUL_MAT records (§8) |
| graphs that need autograd | CPU, eager | one warning; `strict=True` raises |
| WEIGHT heap full, or 4 uploads in a row | CPU (that weight, that node) | one warning, `cpu_fallback_calls` |

## 5. Environment variables

`TSR_DEVICE` and the transport variables: [`sw/transport/README.md`](../sw/transport/README.md).
`TSR_TORCH_DEBUG` (`1` partition, uploads and counters at exit, `2` also every device call, `3` also the
rewritten aten graphs), `TSR_TORCH_VERIFY` (`1`: check every device matmul bit for bit against libtsr_ref; a
mismatch raises, naming the weight, the element and both values), `TSR_TORCH_EXT_DIR`, `TSR_TORCH_HF_MODEL`
(model directory of the opt-in `test_hf.py`) and `TSR_RT_DEBUG` (tsr_rt's log) are described in
[`sw/torch/README.md`](../sw/torch/README.md). `TSR_TORCH_PYTHON` is a CMake cache variable.
`tt.compile(model, strict=False, debug=None, verify=None, offload=True, **kw)` passes the same switches as
`options=` of `torch.compile`. `offload=False` prints the partition but keeps every matmul on the CPU.

## 6. Tests

`ctest --test-dir build/dev -R '^(torch_pytest|rt_)'` runs the suites below on the simulator and on the mock transport
with `T_MAX` 1, 4 and 8. The same suites run on the card (§2.4).

- `test-tsr-rt` (`rt_*`, [`sw/rt/README.md`](../sw/rt/README.md)): every case checks that `read_blocks` equals ggml's
  blocks byte for byte, that `linear` and `linear_ref` equal an independent model (ggml conversion plus libtsr_ref)
  bit for bit, that the y padding is untouched, and that the NMSE against an f64 product of the dequantized weight
  stays below 1e-3, over f32, f16, bf16, native Q8_0 and native Q4_K weights, K from 32 to 4864 and above K_MAX, N
  from 1 to 4864 (the N = 130000 head with `--big`), T from 1 to 20, forced N and K parts, strided x and y, and
  misaligned rows. Also the op-list cache with exact host traffic, free and re-upload, refusals and two threads.
- `test_ext.py`: stored blocks equal an independent numpy `quantize_row_q8_0_ref` byte for byte (`spec_model.py`),
  and `linear` equals `linear_ref` equals the numpy spec model of `numcheck.py` (B1b) bit for bit, for 11 shapes times
  3 dtypes, the N = 151936 head, K > K_MAX, strided and misaligned operands, T up to 300 and native blocks.
- `test_numerics.py`: NMSE against torch float32 on uniform [-1, 1] data, well inside the 5e-4 limit of
  test-backend-ops. The single-output shape (N = T = 1, K = 2048; `docs/spec.md` §17, *Exception to op-level NMSE*)
  has an NMSE that is one squared relative error and exceeds the limit in a sizeable share of random draws
  ([`llama-pin.md`](llama-pin.md)), so it is checked bit for bit. Negative controls must fail: a half-even weight
  quantizer, the x86 SIMD quantizer (`id = 127 / amax`) and sequential accumulation all differ from the device.
- `test_backend.py`: a compiled MLP and a pre-norm transformer block against eager (bit-identical with reference
  linears, within 5e-4 NMSE of float64 eager), fallbacks, re-upload, autograd refusal, operand forms, dynamic
  lengths, tied weights, graph breaks, the scaled result. `test_quant.py`: `QuantLinear` against the numpy spec
  model, eager and compiled. `test_robustness.py`: regression tests for §3.5, §3.6 and §3.8 (the per-(node, weight)
  upload guard, device-bound handles, an unavailable device raising, `close()` waiting for calls in flight, no
  deadlock between `get`, `open` and `is_open`, `load_state_dict` on an inference-tensor weight, matmuls in HOP
  bodies). `test_hf.py` is opt-in (Qwen2.5-0.5B-Instruct, 12 plus 4 tokens).

`sw/torch/tools/hf_check.py` checks a Hugging Face causal LM from its safetensors directory. It refuses a directory
with pickle files, checks the safetensors sha256 against the Hub's LFS etag, and needs the native libraries of
numcheck (`libtsr_ref.so`, `libnumcheck_helper.so`) in `--numcheck-build` (default `build/numcheck`, else
`build/dev`). It runs one prompt (numcheck's fixed sentence, cut to `--tokens` tokens, teacher-forced) on a float32
model and reports KL per position with klcheck's statistics: CPU eager against the compiled model, against the 1-ulp
floor, against `cpu-q8` (eager with each Linear computed by torch's f32 GEMM from the dequantized device blocks and
the A32-quantized activations: the same quantized model with another accumulation order) and against weights-only
Q8_0. The floor of an unquantized f32 model is about 1e-12, so the ratio against it only says that the device's
distance is the 8-bit quantization. The ratios are reported, not gated. It checks that every `nn.Linear` weight is
called on the device exactly once per forward with no CPU fallback, that the device is repeatable, and that the
logits of the forward and of the last generation step equal eager with `reference_linears()`. For the `--sample`
linears (default layer 0 `q_proj`, layer 12 `down_proj`, `lm_head`) the device blocks must equal ggml's
`quantize_row_q8_0_ref`, and y must equal libtsr_ref's `nc_model_mm` (numcheck B1), the numpy spec model (B1b) and
`linear_ref`. `--blocks all` compares every weight's blocks with ggml, and `--gguf` the head with that GGUF's Q8_0
`output.weight`. `--gen` compares greedy tokens of CPU and device. Exit 0 if all of these and top-1 of at least 80 %
hold. `sw/torch/examples/hf_compare.py` runs the same model compiled against eager in float32 or bfloat16.

## 7. Limitations

- Inference only. Weight matmuls only: bias, activations, norms, RoPE, attention and embeddings run on the CPU,
  and every device matmul is its own launch of the `graph` kernel with x and y crossing the host link each time.
- No KV cache. Generation in `hf_check` and `hf_compare` recomputes the whole sequence per step, and every new
  length is a new op list.
- The CPU part runs as an eager aten graph, not through Inductor.
- The torch parameters stay in host memory next to the device copy. On `sim` the device DRAM is host memory too.
- Float weights always become Q8_0. Native Q4_K and Q8_0 blocks need `QuantLinear`.
- `bmm` and batched weights (MoE expert stacks) stay on the CPU, and so do matmuls in HOP bodies other than
  `invoke_subgraph`, `cond`, `while_loop` and `map`.
- Weight identity is by storage. Writes through `.data`, and inference-tensor changes that miss the 64 x 64
  sample, are not seen (§3.6).

## 8. Whole-graph lowering

The firmware has op list kinds for the rest of a decoder layer ([`versioning.md`](versioning.md)), but this backend
emits MUL_MAT records only. Running a whole graph from PyTorch on those kinds needs a graph-level op-list builder
above `tsr_rt::linear` (the ggml backend's `tsr_opl_builder`) and device tensors in IO and AUX, so that activations
stay on the device between the ops of a layer. It needs pattern matching on the aten graph for each kind (RMSNorm
with its weight, RoPE in the Hugging Face `rotate_half` form, SiLU times mul, residual adds, SDPA with grouped KV
heads, the router chain of a MoE layer) and PyTorch numerics for the kinds the ggml path takes in another form (the
exact tanh GELU, SW-31, the norms' `eps` and `inv_n`, softmax). A device-resident KV cache (AT §4.6) behind a
static-cache module changes the numerics of attention from PyTorch's f32 cache, so its oracle is the simulator's,
not eager's. Each new node needs a reference (`tt.reference_*`, the C references are in `sw/ctrl/ref`).
