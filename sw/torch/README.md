# PyTorch package (`sw/torch`, `torch.compile(model, backend="tesseract")`)

Inference with the weight matmuls on Tesseract and everything else on the CPU. The device is
`TSR_DEVICE` (`sim` by default; `mock` through the same transport on the simulator; `vkernel` is not
available until P5.3; `xrt` ran on the card).

The runtime under this package (`sw/rt`) runs each device matmul as a one-op op list through the
firmware's `graph` kernel. "Programs" below are those op lists (`rt_execs` = launches,
`rt_prog_*` = op-list cache hits and misses). The move onto the M7 kernels is M8 ([SW](../../docs/design/06-software-verification.md) §9.4).

> **Scope.** Package reference. The design record, the measurements and the next steps are in
> [`docs/torch-backend.md`](../../docs/torch-backend.md); the runtime is
> [`sw/rt/README.md`](../rt/README.md). Results below are labelled with where they were measured.
> They were taken before the control core was added, with the host issuing the ops; the Python suite
> has not been re-run since, the C++ runtime tests have.

Numerics are the llama.cpp path's (`docs/spec.md` §7 Numeric rules, §5 *Load-time conversion*):

- float32 / float16 / bfloat16 weights become Q8_0 when they are uploaded, with ggml's
  `quantize_row_q8_0_ref` (f16 / bf16 widened exactly first). This is the function the ggml backend uses
  (`sw/ggml-tesseract/tsr-weights.cpp`, compiled into tsr_rt), so a weight gets the same blocks as in
  llama.cpp's conversion.
- activations are quantized on the device by QACT (A32); the products are float32;
- bias / `beta * input` / `alpha` are applied on the CPU in float32, and the result is cast once to
  the op's output dtype (= the input dtype).

| Path | Content |
| --- | --- |
| `csrc/tsr_torch.cpp` | extension module `_tesseract_rt`: binds `tsr::rt::device` (sw/rt/tsr_rt.h) for torch tensors, device singleton |
| `CMakeLists.txt` | target `tesseract_torch_ext` (added by the top-level CMakeLists with `-DTSR_WITH_TORCH=ON`), ctests `torch_pytest`, `torch_pytest_tmax8` |
| `tesseract_torch/__init__.py` | public API, registers the backend `"tesseract"`, `compile()` |
| `tesseract_torch/backend.py` | Dynamo backend → AOTAutograd → partition of the aten graph → `device_mm` |
| `tesseract_torch/runtime.py` | device singleton, weight cache, eager `linear` / `linear_ref`, `reference_linears`, counters, debug / verify |
| `tesseract_torch/_ext.py` | finds and imports `_tesseract_rt` |
| `tests/` | pytest suite (`spec_model.py`: independent numpy models) |
| `examples/hf_compare.py` | a Hugging Face causal LM (local safetensors) compiled vs eager CPU |
| `tools/hf_check.py` | end-to-end numeric check of a Hugging Face causal LM: KL / 1-ulp floor, generation, bit-exact against the reference and the llama.cpp path's oracle (below) |

## Build

```
cmake -S . -B build/torch -DCMAKE_BUILD_TYPE=Release -DTSR_WITH_TORCH=ON && cmake --build build/torch -j4 --target tesseract_torch_ext
export PYTHONPATH=$PWD/sw/torch:$PWD/build/torch/python
build/tools/torchvenv/bin/python -c "import tesseract_torch as tt; print(tt.device().description)"
```

(`third_party/apply-patches.sh` first on a fresh checkout, as for every build tree that includes llama.cpp.)

Torch comes from `TSR_TORCH_PYTHON`. The default is `build/tools/torchvenv/bin/python` (a venv made
with `--system-site-packages` from the base interpreter, so it reuses the base torch 2.9.1, plus
transformers / safetensors / huggingface_hub / pytest) if it exists, else `python3`. CMake asks that
interpreter's `torch.utils.cpp_extension` for include and library directories and builds a CPU
extension that links c10 / torch / torch_cpu / torch_python and `tsr_rt` (static: tsr_lower,
tsr_transport, tsr_sim, tsr_ref; `libggml-base.so` of the pinned llama.cpp through the RPATH).
`find_package(Torch)` is not used: this torch is a CUDA 13 wheel, and its CMake config wants a
matching CUDA toolkit. The module (`build/torch/python/_tesseract_rt.cpython-313-x86_64-linux-gnu.so`)
is for that interpreter's Python. If `build/torch/python` is not on `sys.path`, the package also looks
in `$TSR_TORCH_EXT_DIR` and in `<repo>/build/torch/python`. Build: 20 s, 1.5 GB peak (one translation
unit with torch/extension.h).

## Use

```python
import torch, tesseract_torch as tt

model = tt.compile(model)              # = torch.compile(model, backend="tesseract"); import registers it
with torch.no_grad():                  # or torch.inference_mode(); inference only
    y = model(x)
```

`tt.compile(model, strict=False, debug=None, verify=None, offload=True, **torch_compile_kwargs)`
passes `options={"strict", "debug", "verify", "offload"}` to the backend.

What goes to the device: an aten matmul node of the AOT inference graph that

- is `mm`, `addmm`, `linear`, `matmul` or `mv`. `bmm`, `baddbmm`, `addmv`, `dot`, `einsum` and batched
  weights stay on the CPU;
- has one operand that is a **weight** and one that is not. A weight is a parameter, a registered
  buffer or a graph constant (`get_attr`), possibly seen through view ops (`t`, `transpose`,
  `permute`, `view`, `reshape`, `slice`, `select`, `expand`, ...). Weights are identified from AOT's
  input descriptors (`ParamAOTInput`, `BufferAOTInput`, or `PlainAOTInput` of a Dynamo input that is
  an `nn.Parameter` or whose source is a module's `_parameters` / `_buffers`), so parameters that
  arrive as graph inputs under `inline_inbuilt_nn_modules` (the default) are found, and so are
  parameters lifted by AOT (`inline_inbuilt_nn_modules=False`). A weight computed in the graph (e.g. a
  `_to_copy` under autocast) is not a weight;
- after orientation is a 2D [N, K] float32 / float16 / bfloat16 CPU weight with K % 32 == 0, and
  the device accepts it (`unsupported_reason`: the runtime splits large N and K > K_MAX itself).

Both operand orders work: `x @ W.T` (nn.Linear), `x @ W` (a [K, N] weight, GPT-2 Conv1D style: the
uploaded copy is W.T), and `W @ x` / `mv(W, v)` (computed as (xᵀ Wᵀ)ᵀ). The batch and sequence dims
of the activation are flattened into T. Everything else runs on the CPU. The rewritten aten graph
runs eagerly (an FX GraphModule, no Inductor). With CPU ops matching eager, a compiled model gives
exactly eager's results with every supported `nn.Linear` replaced by the device reference
(`tt.reference_linears()`), which the tests check bit for bit.

**Higher-order ops**: matmuls in the bodies of `invoke_subgraph` (`torch.compiler.nested_compile_region`),
`cond`, `while_loop` and `map` are partitioned too (recursively, up to 8 levels). A body placeholder
is a weight if the op passes it a weight at every call site, so a body shared by all layers is one
device node that sees every layer's weight; it is reported once (e.g.
`repeated_subgraph0.mm  layers.0.fc.weight .. layers.5.fc.weight (6 weights)`). Loop-carried values
and `map` slices are never weights. Matmuls in the bodies of other higher-order ops stay on the CPU
and are reported as CPU nodes (`inside a <op> body (not offloaded)`).

**Shared graphs**: one compiled graph can serve many weights (regional compilation with
`layer.compile()` per block, a compiled helper called per layer, a shared HOP body). The run-time
state of a device node is therefore kept per weight (the parameter / buffer the operand is a view of):
the upload-churn guard (below) and the heap-full fallback move only that weight to the CPU.

**Training / autograd**: a graph that needs autograd (grad mode on and an input that requires grad)
is not partitioned. It runs eagerly on the CPU, with one `RuntimeWarning`. `strict=True` raises
instead. Under `no_grad` / `inference_mode`, parameters with `requires_grad=True` are fine.

**Weight cache**: keyed by tensor identity, i.e. the storage (its StorageImpl, kept from address reuse by a
`StorageWeakRef`), data pointer, storage offset, shape, stride and dtype. A hit also needs the same
`_version`. A weight is uploaded on its first use and then kept, shared by every graph that uses it
(recompiles, graph breaks, tied weights). An in-place update (`w.mul_()`, `copy_`, `load_state_dict`)
bumps `_version`, and the next call uploads it again. Replacing `.data` gives a new storage and so a
new upload. Writes through `w.data` do not bump `_version` and are not seen; call
`tt.release_weights()` after them. Device copies of weights whose storage died are released on the
next miss, by `weight_cache().sweep()` or by `release_weights()`. **Inference tensors** (e.g. a model
built under `torch.inference_mode()`) have no version counter. For them a hit also needs the same
raw bits in a sample of up to 64 × 64 elements (an even grid, first and last row and column
included), taken at upload and compared on each use, and the first upload of one warns. So
`load_state_dict` or `mul_` on such a weight is seen, but an in-place change that misses every sampled
element is not; call `tt.release_weights()` after such changes. Handles belong to the device that
issued them (tsr_rt numbers them from 1 on every device). An entry is tied to its device
(`Device.serial`), so after that device is closed (`Device.close()`, `_C.close()` or
`tt.close_device()`) the entry is a miss on the reopened device, and its handle is never freed there.

**Runtime fallback**: if a weight cannot be placed (WEIGHT heap full, after releasing dead weights
and retrying), it runs on the CPU at that node from then on (one warning, counter
`cpu_fallback_calls`). The same happens to a weight uploaded on 4 consecutive calls of a node
(changed in place before every call, or seen through an op that copies it). Both are per weight (a
parameter replaced by a new tensor is a new weight; the copies of dead weights are released). Everything else
raises `TesseractError`: a device that cannot be opened (a `TSR_DEVICE` typo, xrt without a card,
vkernel without a server) fails `torch.compile` at the first compilation instead of computing on the
CPU, and a transport error during an upload or a device execution error fails the call.

**Threads**: the device calls release the GIL and the runtime serializes them. `Device.close()` and
`tt.close_device()` / `_C.close()` wait for the calls in flight on other threads to return; a call
that starts later raises `TesseractError: ... the device is closed` (an eager `tt.linear` that races
with `close_device()` can raise it too). `tt.device()` afterwards opens a new device.

## Environment

| Variable | Effect |
| --- | --- |
| `TSR_DEVICE` | transport: `sim` (default), `mock`, `vkernel`, `xrt`; with its own variables (`TSR_SIM_*`, `TSR_MOCK_*`, `TSR_VKERNEL_*`, `TSR_XRT_*`; `TSR_IMAGE`, `TSR_TRACE` (one trace per launch), `TSR_LOG`; [`sw/transport/README.md`](../transport/README.md)) |
| `TSR_TORCH_DEBUG` | `1`: the partition of every compiled graph (which matmuls go to the device, why the others stay on the CPU), every upload, the counters at exit; `2`: also every device call; `3`: also the rewritten aten graphs |
| `TSR_TORCH_VERIFY` | `1`: every device matmul is also computed by the reference model (libtsr_ref on the weight rows read back from the device) and compared bit for bit; a mismatch raises `TesseractError` naming the weight, the element and both values (for vkernel / xrt bring-up) |
| `TSR_TORCH_EXT_DIR` | where to find `_tesseract_rt` if it is not on `sys.path` |
| `TSR_RT_DEBUG` | the runtime's own log (`1`: uploads, calls and launches) |

`TSR_TORCH_DEBUG=1` on a 3-layer net whose head has K = 48:

```
[tesseract_torch] opened Tesseract (sim, C=4, LANES=128, T_MAX=1, K_MAX=32768, SPM=4096 KiB, 16 GiB/ch)
[tesseract_torch] graph 0: 2 matmul(s) on Tesseract, 1 on the CPU (aten graph of 20 nodes)
[tesseract_torch]   tesseract  addmm            addmm   up.weight  [N=512 K=256]  (+ input on the CPU)
[tesseract_torch]   tesseract  mm               mm      down.weight  [N=48 K=512]
[tesseract_torch]   cpu        addmm_1          addmm   head.weight: K % 32 != 0 (K = 48)
[tesseract_torch] upload up.weight [512 x 256] float32 -> q8_0 (1 x 1 parts, 0.16 MiB on the device)
[tesseract_torch] upload down.weight [48 x 512] float32 -> q8_0 (1 x 1 parts, 0.03 MiB on the device)
[tesseract_torch] counters at exit:
  graphs 1, nodes_device 2, nodes_cpu 1, device_calls 4, uploads 2, cache_hits 2, rt_prog_hits 2, ...
```

## Python API

| Name | Meaning |
| --- | --- |
| `compile(model, *, strict, debug, verify, offload, **kw)` | `torch.compile(model, backend="tesseract", options=..., **kw)` |
| `tesseract_backend(gm, example_inputs, options=None, mode=None)` | the backend function (registered as `"tesseract"`) |
| `Options(strict, debug, verify, offload)` | backend options (`options=` of torch.compile) |
| `linear(x, weight, bias=None, *, ref=False, name="")` | eager `F.linear` with the weight matmul on the device (cached weight; output in x's dtype); inference only |
| `linear_ref(x, weight, bias=None)` | the same on the host reference model; bit-identical to `linear` on a correct device |
| `reference_linears()` | context manager: eager `nn.Linear` with supported weights run through `linear_ref` |
| `supported_weight(w)` | `None` or the reason a weight cannot go to the device |
| `device()` | the process-wide `_tesseract_rt.Device` as it is open (with whatever `open_device` options), else opened with the defaults |
| `open_device(kind=None, io_bytes=None, read_perf=None)` | open it with options (`None`: `TSR_DEVICE`, 0 = the whole IO region, `True`); if it is open already, returns it when every option given matches, else `TesseractError` |
| `close_device()` | drop every cached weight and close the device (after the calls in flight) |
| `weight_cache()` → `WeightCache` | `.acquire(w, name)` → `(device, handle, uploaded)`, `.handle(w, name)` → handle, `.find(w)`, `.sweep()`, `.clear()`, `.entries()`, `len()` |
| `release_weights()` | free every device weight (re-uploaded on next use) |
| `counters()`, `reset_counters()`, `format_counters()` | backend counters and the runtime's (`rt_*`) |
| `set_debug(level)`, `debug_level()`, `set_verify(on)`, `verify_enabled()` | override `TSR_TORCH_DEBUG` / `TSR_TORCH_VERIFY` |
| `specs()` | every node partitioned to the device so far (`MMSpec`: op, side, N, K, weight, calls, tokens, cpu_calls; `.disabled()`: why each weight that runs on the CPU at that node does) |
| `TesseractError` | device / transport errors (a `RuntimeError`) |
| `_C` | the extension module `_tesseract_rt` |

`_tesseract_rt` (C++, GIL released during device work): `Device(kind=None, io_bytes=0,
read_perf=True)` with `.upload(w [N, K] f32/f16/bf16, max_part_rows=0, max_part_k=0) → handle`
(any row stride and alignment), `.upload_blocks("q8_0" | "q4_k", uint8 blocks, N, K)`, `.free(h)`,
`.info(h)`, `.read_blocks(h, r0=0, r1=-1)` (ggml blocks of the device format, uint8 [rows, row_bytes]),
`.linear(h, x [T, K] f32, out=None) → y [T, N] f32` (any row stride of x / out), `.linear_ref(...)`,
`.counters()`, `.reset_counters()`, `.memory()`, `.config`, `.transport`, `.description`, `.serial`
(1, 2, ... per device opened in the process), `.io_bytes`, `.read_perf`, `.closed`,
`.unsupported_reason(dtype, N, K)`, `.close()` (waits for the calls in flight). Every method uses
the device through a lease that keeps it alive until the call returns, so closing from another
thread is safe. Module functions (the singleton; they release the GIL before taking its lock):
`open(kind=None, io_bytes=None, read_perf=None)` (the open device if every option given matches,
else a new one), `get()` (the open device as is, else a new one with the defaults), `is_open()`,
`close()`; and `row_bytes(type, K)`.

Counters (`counters()`): `graphs`, `graphs_eager` (autograd graphs run eagerly), `nodes_device`,
`nodes_cpu`, `device_calls`, `device_tokens`, `cpu_fallback_calls`, `uploads`, `reuploads`,
`upload_bytes`, `cache_hits`, `weights_freed`, `verified_calls`, `cached_weights`, and the runtime's
`rt_linears`, `rt_tokens`, `rt_execs`, `rt_prog_hits`, `rt_prog_misses`, `rt_perf_cycles`,
`rt_perf_mvm_cycles`, `rt_dev_read_bytes`, `rt_h2d_bytes`, `rt_h2d_weight_bytes`, `rt_d2h_bytes`
(sw/rt/README.md).

## Tests

```
PYTHONPATH=sw/torch:build/torch/python build/tools/torchvenv/bin/python -m pytest -q -s sw/torch/tests
cd build/torch && ctest -R torch_pytest            # TSR_DEVICE=sim, T_MAX 1 and T_MAX 8
TSR_TORCH_HF_MODEL=models/hf/Qwen2.5-0.5B-Instruct python -m pytest -q -s sw/torch/tests/test_hf.py   # opt-in
```

- `test_ext.py` (extension): for 11 shapes (N 1 … 4864, K 32 … 4864, T 1 … 40) × f32 / f16 / bf16,
  the stored blocks equal an independent numpy `quantize_row_q8_0_ref` byte for byte (`spec_model.py`),
  and `linear` == `linear_ref` == the numpy spec model of sw/tools/numcheck.py (B1b, written from the
  spec text) bit for bit. Also: directed rounding rows (x·id on .5: {127, 2.5, −0.5, 0.5} → {127, 3,
  −1, 1}; d an fp16 / f32 subnormal); the N = 151936 head (2048 sampled rows against the spec model);
  K > K_MAX (2 K parts, against the spec model per part + one f32 add); strided, transposed and
  2-byte-misaligned weights and x, output views; T = 1 … 300 in one program with exact host traffic
  and program-cache hits; a T_MAX 8 sim device of its own; native Q8_0 / Q4_K blocks; errors.
- `test_numerics.py`: NMSE against torch's float32 `x @ W.T` on uniform [−1, 1] data, and the
  single-output exception (below).
- `test_backend.py`: an MLP (f32 / f16 / bf16) and a pre-norm transformer block (q / k / v / o,
  SDPA, SwiGLU; f32 / bf16; with `verify`) compiled vs eager: bit-identical to eager with reference
  linears, NMSE vs float64 eager < 5e-4. Also: fallback for K % 32 != 0 and float64; an all-CPU model
  identical to eager; in-place update and `load_state_dict` re-upload; repeated calls hit the
  program cache with exact host traffic; B·S > T_MAX; autograd graphs run eagerly (and `strict`
  raises); `inference_mode`; weight operand forms (left, [K, N], `matmul`, 1D input, `mv`, `addmm`
  with alpha / beta, buffer, sliced weight); `beta = 0` ignores a NaN input; dynamic sequence lengths
  with one upload per weight; tied weights and graph breaks share uploads; dead weights released;
  debug output; aten `linear` / `matmul` nodes; `inline_inbuilt_nn_modules=False`; close / reopen;
  a weight that cannot be placed runs on the CPU.
- `test_robustness.py` (one regression test per finding of the review; each failed before
  its fix): regional compilation and a compiled helper over 8 layers share one graph and still upload
  each weight once and run every call on the device; a heap-full weight in a shared graph moves only
  that weight to the CPU; after `Device.close()` / `_C.close()` a cached handle is neither a hit nor
  freed on the reopened device; `open_device(read_perf=False)` / `open_device(io_bytes=...)` stay
  usable by `tt.linear`, `tt.device()` and `torch.compile`, and a conflicting explicit open still
  raises; `TSR_DEVICE=vkernal` fails `torch.compile`; a transport error during an upload raises, while
  heap full falls back; closing a device while another thread is inside `linear` (subprocess; it used to
  crash with SIGSEGV); 25 rounds of 4 threads calling `get` / `open` / `is_open` together after a
  close (subprocess; it used to deadlock); `load_state_dict` and `mul_` on an inference-tensor weight
  reach the device; `nested_compile_region` (6 layers, one shared body), `cond`, `while_loop` and `map`
  bodies on the device, bit-exact against the reference; a HOP body the backend does not map is
  reported as CPU nodes.
- Negative controls (scratch run): a half-even weight quantizer and the x86 SIMD one
  (`id = 127 / amax`) differ from the stored blocks (2 resp. 3 of 1048576 values on a random
  256 × 4096 weight), and sequential accumulation differs from the device in 882 of 1024 outputs.

Results, measured on the simulator (build/torch, Release, torch 2.9.1, Python 3.13; the skipped
tests are the opt-in HF test and, off the sim, the sim-only T_MAX 8 device; 86 + 1 tests):

| Run | Result | Wall | Max RSS |
| --- | --- | --- | --- |
| `TSR_DEVICE=sim` | 86 passed, 1 skipped | 11.7 s | 1.8 GB |
| `TSR_DEVICE=sim TSR_SIM_TMAX=8` | 86 passed, 1 skipped | 11.7 s | 1.8 GB |
| `TSR_DEVICE=mock` (DMA align 64, chunk 64 KiB) | 85 passed, 2 skipped | 13.1 s | 1.8 GB |
| `TSR_DEVICE=mock` (T_MAX 8, 3 ch, align 4096, chunk 12 KiB) | 85 passed, 2 skipped | 12.8 s | 1.8 GB |
| opt-in `test_hf.py` (Qwen2.5-0.5B-Instruct, sim) | 1 passed (4 / 4 steps bit-exact, top-1 93.3 %) | 36 s | 4.8 GB |
| ctest in build/torch (`-j2`): `torch_pytest`, `torch_pytest_tmax8`, `rt_sim`, `rt_sim_tmax8`, `rt_mock`, `rt_mock_tmax8`, `ggml_tesseract_units`, `_e2e`, `_e2e_tmax8`, `_e2e_hosted`, `_e2e_hosted_tmax8`, `_scalar_nmse` | 12 / 12 passed | 23 s | |
| ctest in build/rt (`-j2`): `rt_*` (4) and `ggml_tesseract_*` (7, with `_backend_ops`: MUL_MAT on Tesseract0, 1402 / 1402) | 11 / 11 passed | 39 s | |

NMSE against torch float32 (12 shapes × 3 weight dtypes × 5 seeds = 180 cases, every shape with
N·T ≥ 16): min 7.95e-6, median 2.83e-5, p90 3.92e-5, p99 6.77e-5, max 8.90e-5, i.e. 17.8 % of the
5e-4 limit. **Single-output exception** (`docs/spec.md` §17, *Exception to op-level NMSE*): for N = T = 1,
K = 2048, f32, the NMSE is one squared relative error. In 1000 seeded trials it is ≥ 5e-4 in 149
(14.9 %; docs/llama-pin.md: 15.4 % of 10000). The float64 product of the same 8-bit operands fails on
exactly the same 149 trials, and the device is bit-exact with the reference model on all 1000. So
that shape is checked bit for bit, not by NMSE. Shapes with very few outputs (N·T < 16) behave
the same way and are not in the NMSE table.

## Hugging Face models

```
build/tools/torchvenv/bin/python -I sw/torch/examples/hf_compare.py models/hf/Qwen2.5-0.5B-Instruct \
    [--dtype float32|bfloat16] [--tokens 16] [--gen 8] [--gguf models/qwen2.5-0.5b-instruct-q8_0.gguf]
```

The script loads a local model directory (safetensors, `trust_remote_code=False`). It is run with
`-I` because the model directory is data; the script sets `sys.path` itself. Then:

1. The compiled model greedy-generates `--gen` tokens (full recompute per step). Every step's logits
   must be bit-identical to eager execution with `reference_linears()`.
2. Eager CPU scores the device's sequence (klcheck's method), giving KL and top-1. For scale, the
   same is done for eager with dequantized Q8_0 weights (the weight conversion alone) and for the
   1-ulp floor. The floor is ~0 against an unquantized reference, so the spec's KL ≤ 3 × floor rule
   (CPU backend vs device on the same quantized weights) does not apply.
3. `--gguf` compares every weight's device blocks with the GGUF's Q8_0 tensors.

Measured on the simulator, Qwen2.5-0.5B-Instruct (HF revision 7ae55760, `model.safetensors`
sha256 fdf756fa…), TSR_DEVICE=sim (T_MAX 1), 16 prompt tokens + 8 generated:

| | float32 (bf16 weights widened) | bfloat16 |
| --- | --- | --- |
| weight matmuls per graph on the device / on the CPU | 169 / 0 | 169 / 0 |
| uploads (Q8_0 bytes written) | 169 (528.7 MB) | 169 |
| steps bit-identical to eager with reference linears | **8 / 8** | **8 / 8** |
| KL(eager ‖ device) mean / median / max | 6.5e-3 / 4.1e-3 / 3.7e-2 | 1.3e-2 / 5.2e-3 / 6.8e-2 |
| top-1 agreement (all 23 positions / 8 generated) | 91.3 % / 100 % | 95.7 % / 100 % |
| weights-only Q8_0 (eager, dequantized weights): KL mean, top-1 | 4.3e-3, 95.7 % | 8.0e-3, 95.7 % |
| 1-ulp floor KL mean | 3.4e-11 | 0 |
| device forwards (with 2 compilations) / reference forwards | 36.4 s / 29.6 s | 45.0 s / 27.7 s |
| whole script: wall, peak RSS | 92 s, 4.7 GB | 80 s, 3.8 GB |

The 169 matmuls are q / k / v / o / gate / up / down × 24 plus the tied 151936 × 896 head; the second
graph is the dynamic-shape recompile. In both dtypes the device's continuation ", high-performance,
multi-platform, multi" is eager's argmax at every step. The opt-in pytest (12 + 4 tokens, float32)
takes 45 s and 4.8 GB.

`--gguf models/qwen2.5-0.5b-instruct-q8_0.gguf` (the GGUF used by the llama.cpp path):
`output.weight` (= `token_embd`, 4,254,208 blocks) is byte-identical to the device copy. The 168
layer tensors all differ, because they are not this checkpoint's tensors. They are the same rows
rescaled per channel: for the 4 tensors examined, row i of the checkpoint weight best matches GGUF row
i (cosine 0.95 to 1.0 on 64 sampled rows), with column-norm ratios 0.26 to 15. Dequantized, the GGUF
copy is 0.03 to 6.1 (relative L2) away from the safetensors weights, against 0.0053 to 0.0082 for the
device's Q8_0 copy. So that GGUF was made from a transformed checkpoint (per-channel scales folded into
the weights), and only the untransformed embedding can be compared byte for byte.

### End-to-end check: `tools/hf_check.py`

```
build/tools/torchvenv/bin/python -I sw/torch/tools/hf_check.py models/hf/Qwen2.5-0.5B-Instruct \
    [--download Qwen/Qwen2.5-0.5B-Instruct] [--tokens 32] [--gen 16] [--gguf GGUF] [--json OUT]
build/tools/torchvenv/bin/python -I sw/torch/tools/hf_check.py MODEL_DIR --forward-only --tokens 8 \
    --save-logits sim.npy        # then with TSR_DEVICE=vkernel ...: --expect sim.npy (bit for bit)
```

`--download REPO` fills the directory with `huggingface_hub` (`*.safetensors`, config and tokenizer
files only; a repository without safetensors is refused). The script refuses a directory with pickle
files and checks the safetensors sha256 against the Hub's LFS etag in the download metadata. It needs
the native libraries of numcheck (`libtsr_ref.so`, `libnumcheck_helper.so`) in `--numcheck-build`
(default `build/numcheck`; `--numcheck-build build/dev` uses the main build). One prompt
(numcheck's fixed sentence, the first `--tokens` tokens), float32:

- **KL** per position (klcheck's statistics, float64) of the teacher-forced prompt: CPU eager vs the
  compiled model, and vs the **1-ulp floor exactly as klcheck makes it** (the input embeddings,
  `nextafterf` ±1 ulp, direction from klcheck's splitmix64 of (seed, seq 0, position, element); the
  Python port gives the same rows as `klcheck embd --perturb-ulp S --seq Q --pos P`). The embedding
  path must give the CPU's bits without the nudge. The floor of an unquantized f32 model is ~1e-12, so
  the ratio against it only says that the device's distance is the 8-bit quantization. For the spec's
  comparison (same quantized model, different accumulation order), the script also runs **cpu-q8**:
  eager with each Linear computed by torch's f32 GEMM from the dequantized device blocks and the
  A32-quantized activations, with its own 1-ulp floor. It also runs weights-only Q8_0. The ratio is
  also given over the positions where the floor moved, because 8-bit activations can absorb the
  nudge.
- **Offload**: every `nn.Linear` weight is called on the device exactly once per forward, the tied
  head included (it shares the embedding's upload), with no CPU fallback. There are PERF cycles per
  token, per category, and device time vs CPU-op time.
- **Bit-exact**: the device is repeatable, and the logits of the forward and of the last generation step
  equal eager with `reference_linears()`. For `--sample` linears (default layer 0 `q_proj`, layer 12
  `down_proj`, `lm_head`), the x and y of the actual device call are captured. The device blocks must
  equal ggml's `quantize_row_q8_0_ref` (llama.cpp's load-time conversion), and y must equal libtsr_ref's
  `nc_model_mm` (numcheck B1, the oracle of the llama.cpp path), the numpy spec model (B1b) and the
  extension's `linear_ref`. `--blocks all` compares every weight's blocks with ggml.
  `--gguf`: the head against that GGUF's Q8_0 `output.weight`.
- **Generation**: `--gen` greedy tokens by CPU and device (full recompute), first divergence.

Exit 0 if offload, repeatability, both bit-exact checks, the oracle checks and top-1 ≥ 80 % hold.
The KL ratios are reported, not gated.

Measured on the simulator, Qwen2.5-0.5B-Instruct (revision 7ae55760, sha256 = Hub etag),
`TSR_DEVICE=sim` (T_MAX 1), 32 prompt tokens, 16 generated, seed 1, 8 torch threads. Two runs gave
the same bits:

| | mean KL | median | max | top-1 | max \|Δlogit\| |
| --- | --- | --- | --- | --- | --- |
| CPU f32 vs device | 3.565e-3 | 2.054e-3 | 3.74e-2 | 30 / 32 | 1.77 |
| CPU f32 vs CPU f32 + 1 ulp (floor) | 3.56e-12 | 3.33e-12 | 9.5e-12 | 32 / 32 | 2.7e-5 |
| CPU f32 vs cpu-q8 (same 8-bit operands, torch GEMM) | 3.759e-3 | 2.192e-3 | 4.11e-2 | 30 / 32 | 1.77 |
| CPU f32 vs weights-only Q8_0 | 2.027e-3 | 8.59e-4 | 3.75e-2 | 31 / 32 | 1.50 |
| cpu-q8 vs device | 1.730e-3 | 1.345e-3 | 5.38e-3 | 32 / 32 | 0.75 |
| cpu-q8 vs cpu-q8 + 1 ulp (floor) | 5.93e-4 | 1.70e-4 | 2.42e-3 | 32 / 32 | 0.49 |

- Ratio device / floor: 1.0e9 against the f32 CPU. Against cpu-q8 it is **2.92**, and **1.72** over
  the 16 positions where its floor moved: positions 0–15 are bit-identical with and without the nudge.
- The device and cpu-q8 miss top-1 at the same two positions (20, 23). The device's distance from the
  f32 model is the 8-bit quantization.
- 16 greedy tokens: CPU and device are identical (" that the printing of books could be done more
  quickly and cheaply. The printing"), so there is no divergence.
- Offload: 169 / 169 Linear weights on the device (q / k / v / o / gate / up / down × 24, and `lm_head`,
  tied to `embed_tokens`, shares its upload). The first graph has 169 matmul nodes on the device and 0
  on the CPU. 2 graphs (the dynamic-length recompile), 0 uploads in compiled calls, 0 CPU fallbacks.
- Bit-exact: repeatable; the forward and the last generation step (47 tokens) equal the reference
  linears. On the 3 sampled linears (896 × 896 + bias, 896 × 4864, 151936 × 896; T = 32; 4,919,296
  outputs), the device blocks equal ggml's `quantize_row_q8_0_ref`, and there are 0 mismatches against
  libtsr_ref, the numpy spec model and `linear_ref`. 169 / 169 weights (15,436,288 blocks) are
  byte-identical to ggml. The head's blocks equal `output.weight` of
  `models/qwen2.5-0.5b-instruct-q8_0.gguf`, and libtsr_ref on those GGUF blocks gives the same y.
- Device: PERF 2,166,289 cycles / token (MVM 2,080,256; sim cycle model), 169 execs per forward (all
  program-cache hits), 529.7 MB read from device DRAM per token. Wall time 2.48 s per 32-token forward
  (77.5 ms / token: device calls 2.46 s, CPU ops 0.02 s), vs 0.29 s for CPU eager. First call (Dynamo + AOT
  + run) 5.75 s. Upload of 169 weights (504.2 MiB Q8_0) 1.21 s. Generation 58.7 s on the device (600 token
  passes), 3.2 s on the CPU. Whole script 106 s wall, peak RSS 5.0 GiB.

The `--forward-only --tokens 8` run on `TSR_DEVICE=sim` gives logits with sha256 `4a5e7b24…7f66`
(8 rows): 169 / 169 matmul nodes and calls on the device, 0 CPU fallbacks, PASS, 10.5 s wall, 3.6 GB.

**Card** (k1, 300 MHz; `docs/platform-u200.md` §14.1). Measured on the card (k1):
`TSR_DEVICE=xrt TSR_TORCH_VERIFY=1` the suite 85 passed, 2 skipped (12.1 s), `test_hf.py` passed, and
`hf_check.py --forward-only --tokens 8 --expect <sim logits>` 8 / 8 rows bit-identical (sha256
`4a5e7b24…7f66`, as on `sim`), 169 device calls, 9.06 M PERF cycles per token.
PyTorch has not run on a card with the control core (work package W6).

## Limits

- Inference only; no KV-cache ops, attention or other ops on the device (as the llama.cpp path's M6).
- Plain tensor attributes and globals that are not parameters or buffers count as activations.
- `bmm` / batched weights (MoE expert stacks, `W @ x` with batched x) stay on the CPU.
- Matmuls in the bodies of higher-order ops other than `invoke_subgraph`, `cond`, `while_loop` and
  `map` stay on the CPU.
- Writes through `.data` are not detected, nor are in-place changes of an inference-tensor weight
  that miss its 64 × 64 sample (see the weight cache).
- The CPU part runs as an eager aten graph, not through Inductor.
