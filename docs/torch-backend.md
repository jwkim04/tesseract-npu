# PyTorch backend (`torch.compile(model, backend="tesseract")`)

The PyTorch backend runs a `torch.compile`d model on a Tesseract device in one of two execution modes
([design/README.md](design/README.md) §2). In full mode, the default, regions of the model's aten graph run as one
op list each. In matmul mode only the weight matmuls run on the device. The C++ runtime `tsr_rt` below the
backend runs op lists through the firmware's `graph` kernel. This document covers usage, architecture and tests
of `sw/torch` and `sw/rt`. The package reference is [`sw/torch/README.md`](../sw/torch/README.md) and the runtime's is
[`sw/rt/README.md`](../sw/rt/README.md). Order of authority: `docs/spec.md` (§5 Device weight layout, §6 Quantized
activation layout, §7 Numeric rules), then the simulator (`sw/sim`), then this document. Normative for the
software split: [SW](design/06-software-verification.md) §9.4.

## 1. What it is

`import tesseract_torch` registers the torch.compile backend `"tesseract"`. A model compiled with it runs
inference on Tesseract:

```python
import torch, tesseract_torch as tt

model = torch.compile(model, backend="tesseract")   # or tt.compile(model); full mode
model = tt.compile(model, mode="matmul")             # matmul mode
with torch.no_grad():                                # or torch.inference_mode()
    y = model(x)
```

- Modes. `tt.compile(model, mode="full" | "matmul", kv_size=4096, ...)`, or `options={"mode": ..., "kv_size":
  ...}` of `torch.compile`, or `TSR_TORCH_MODE`. An explicit `mode` wins over the variable, and the default is
  full mode. An unknown mode is a `ValueError`.
- Full mode. The aten graph is partitioned into regions of ops that lower to op list kinds of the firmware
  (§4). A region runs as one op list, and its intermediate values stay in device memory. Only the inputs and
  outputs of a region cross to the host. An op that matches no pattern runs on the CPU and ends the region
  there. The stateless forward of a Hugging Face Qwen2.5 decoder is one region and one launch (§8).
- Attention. `options={"attention": "device" | "cpu"}`, `tt.compile(model, attention=...)` or
  `TSR_TORCH_ATTENTION` choose where full mode computes attention. The default is `"device"`. On the device,
  attention quantizes Q, K and V as q8_0, the numerics of the llama.cpp path, so a region differs from eager
  float32 by more than the Q8_0 weights make it (§3.7). With `"cpu"`, attention runs in float32 on the CPU and
  a region ends at each attention. An explicit option wins over the variable. An unknown value of the option is a
  `ValueError`. The variable is read in full mode only: there an unknown value is a `ValueError`, and matmul mode
  ignores it.
- Matmul mode. Only weight matmuls (`mm`, `addmm`, `linear`, `matmul`, `mv` with one operand that is a
  parameter, buffer or graph constant, a float32, float16 or bfloat16 CPU tensor with K % 32 == 0 and a shape
  the device accepts) run on the device, each as its own launch. Everything else runs on the CPU, as in the
  llama.cpp path's matmul mode. In full mode a weight matmul outside every region is such a node too.
  A model that provides Q8_0 or Q4_K blocks uses `tt.QuantLinear` instead of `nn.Linear` (§3.9).

- Device. `TSR_DEVICE` picks the transport (`sim` by default; `mock`, `vkernel`, `xrt` through the same
  `tsr_transport` as the llama.cpp path). `xrt` runs the release image `tsr_u200_k4`. The runtime boots the
  firmware image (`$TSR_IMAGE`, else the build's `tsr-fw.elf`) when it opens the device.
- Numerics. The llama.cpp path's. A float weight becomes Q8_0 at upload with ggml's `quantize_row_q8_0_ref`, so
  its blocks are byte-identical to llama.cpp's load-time conversion (`docs/spec.md` §5). Activations are
  quantized on the device by QACT A32 (A256 for Q4_K), and the products are f32 (§3.7).
- Equivalence. In matmul mode the CPU ops are eager PyTorch's, so a compiled model gives exactly eager results
  with every supported `nn.Linear` replaced by the device's reference model (`tt.reference_linears()`). The
  tests check this bit for bit. In full mode the norms, activations, RoPE and attention of a region have the
  device's numerics. The oracle of a region is the reference executor, and eager PyTorch is a statistical target
  (§3.7).
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
ctest --test-dir build/dev -R '^(torch_pytest|rt_|ggml_tesseract_kinds)'   # sim T_MAX 1 / 4 / 8, rt on sim and mock
build/tools/torchvenv/bin/python -I sw/torch/tools/hf_check.py models/hf/Qwen2.5-0.5B-Instruct   # §6, matmul mode
build/tools/torchvenv/bin/python -I sw/torch/tools/hf_check.py models/hf/Qwen2.5-0.5B-Instruct --full-mode --tokens 16
build/tools/torchvenv/bin/python -I sw/torch/tools/hf_check.py models/hf/Qwen2.5-0.5B-Instruct --full-mode --tokens 16 --attention cpu
build/tools/torchvenv/bin/python -I sw/torch/tools/fuzz_full.py --gen elem --start 0 --count 50   # §6, the fuzzer
```

The older suites run in matmul mode (`tests/conftest.py` sets `TSR_TORCH_MODE=matmul`), and `test_graph.py`
and the full-mode test of `test_hf.py` ask for full mode themselves. `hf_check.py --full-mode` is the full-mode
check on a Hugging Face model: one compiled forward must be one region and one launch, checked against the
reference executor (§3.7), then a second forward in the steady state. It compares the logits with eager float32 and
with an eager run whose attention is the device's q8_0 path (§6). `--attention cpu` checks the same forward with
attention on the CPU.

`TSR_DEVICE=mock` runs the same suite on the adapter model of the transport, and `TSR_SIM_TMAX=4` gives the simulated
core the `T_MAX` of the release image. `-I` ignores `PYTHONPATH`, so the tools find the extension through
`TSR_TORCH_EXT_DIR` or the build trees of §2.2. `hf_check.py --forward-only --tokens 8 --expect sim.npy` compares the
forward's logits bit for bit with a `TSR_DEVICE=sim` run saved with `--save-logits`: the model-level check of
`docs/spec.md` §17 (same model run, bit-identical logits), done from PyTorch, against any device kind. With
`--full-mode` in place of `--forward-only`, `--save-logits FILE.npy` and `--expect FILE.npy` do the same check on
the full-mode logits, for example the card against the simulator.

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
weight rows read back from the card and compares it bit for bit. In full mode it also runs every region call
through the reference executor on the constants and weights read back from the card and compares the outputs
bit for bit (§3.7). Add `--full-mode` to the `hf_check.py` command of §2.3 for the model-level check. The release
image has no `MVM_LOCAL_SCALE`, so the scaled-result tests take the CPU multiply (§3.5).

## 3. Architecture

### 3.1 Layers

```
torch.compile(model, backend="tesseract")
  └─ tesseract_torch (Python, sw/torch/tesseract_torch)
       ├─ backend.py   Dynamo graph → AOTAutograd → aten inference graph → regions (full mode) → device_region nodes,
       │               then the remaining weight matmuls → device_mm nodes; the rewritten graph runs eagerly
       │               (FX GraphModule, no Inductor)
       ├─ lower.py     full mode: patterns, layout rules and region formation on the aten graph (no device, no extension)
       └─ runtime.py   device singleton, WeightCache, ConstCache, RegionRT, eager linear / linear_ref, counters, debug, verify
            └─ _tesseract_rt (C++ extension, sw/torch/csrc/tsr_torch.cpp; torch tensors in and out, GIL released)
                 └─ tsr_rt (sw/rt): weight upload, linear, N / K parts, op list per call, OPL cache, counters, reference path;
                    │              graph API: constants, planner of a region program (plan per T), run, reference run
                      ├─ tsr_lower (sw/ggml-tesseract/lower.cmake): shape checks, OPL builder and cache, `graph` launcher, allocator, weight layout,
                      │            tsr-kinds (one encoder per kind), tsr-rec (the record and its encoding)  ← shared
                      ├─ tsr-weights.cpp + ggml-base (pinned llama.cpp): quantize_row_q8_0_ref, f16 / bf16 to_float ← shared
                      ├─ tsr_ref (libtsr_ref): the reference model behind linear_ref and TSR_TORCH_VERIFY
                      ├─ tsr_opl_ref (sw/ctrl/ref/opl_ref): the reference executor of an op list (graph_run_ref, region verify)
                      └─ tsr_transport: sim | mock | vkernel | xrt  →  firmware `graph` kernel (op list records)
```

### 3.2 Shared code: `tsr_lower` and the load-time conversion

The two lines marked "shared" in §3.1 are the code the ggml backend uses: both paths build an op list with the
same builder and launcher and store the same weight bytes. `tsr_lower` is a static PIC library with
no ggml dependency (shape checks and SPM tiling, the op list builder with its hazard analysis, JOINs and deferred
MUL_MAT, the OPL cache, the `graph` launcher, the allocator, the row repack and the scatter and gather over the
channels). Its files stay in `sw/ggml-tesseract`, and `lower.cmake` defines the target. Two of them serve the
full-mode records: `tsr-rec.{h,cpp}` holds the record type and its encoding (preload entries, DRAM ranges,
dependencies and JOINs), and `tsr-kinds.{h,cpp}` one encoder per kind (MUL_MAT, ADD, MUL, SCALE, RMS_NORM, GLU,
UNARY, ROPE, SET_ROWS_KV, FLASH_ATTN, CPY) over plain descriptors: handles, row counts and strides. The encoders
write the argument words that the ggml backend's lowerers write for the same operation (SW-102), and they
refuse what the booted image or the planners do not take. The ggml lowerers keep their own code; both front ends
are held to the same per-kind references. The f32, f16 and bf16 to Q8_0 conversion is the
backend's `tsr-weights.cpp`, compiled into `tsr_rt`. libtsr_ref's `tsr_qact` is not used for it: it differs from
`quantize_row_q8_0_ref` on blocks with subnormals or a tiny maximum (DAZ and FTZ, `docs/spec.md` §21) and writes a
QA image with an f32 scale, not `block_q8_0`. Only ggml's function gives "the same blocks as llama.cpp".

### 3.3 The runtime `tsr_rt` (sw/rt)

`tsr::rt::device` (`sw/rt/tsr_rt.h`) opens `TSR_DEVICE`, checks the configuration, and offers `upload`, `free_weight`,
`info`, `read_blocks`, `linear`, `linear_ref`, counters and `memory`. Every member function takes one device mutex.
Device memory: OPL (ch0, 64 MiB) holds the cached op lists, IO (ch0, 256 MiB) the x and y rows of one launch, and
WEIGHT (every channel, from 512 MiB) the weights with symmetric allocation (same W_OFF on every channel, 4 KiB
aligned). The graph API also uses AUX (constants) and the top of WEIGHT (KV planes). Uploads convert f32, f16
and bf16 to Q8_0 (§3.2) or take native Q8_0 and Q4_K blocks (`upload_blocks`), repack the rows into the device layout (`docs/spec.md` §5) and scatter them over the
channels.

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
- Graph API. `graph_create` takes a straight-line program over float32 `[T, n]` values (`graph_desc` in
  `tsr_rt.h`) and `graph_run(h, T, inputs, outputs)` runs it. The ops are `mul_mat` (with an addend), `add`, `mul`,
  `scale`, `rms_norm`, `glu`, `unary`, `rope`, `attn` and `cpy`. `const_upload` puts float32 rows (a norm weight,
  a bias row, a RoPE angle table) in AUX and `const_free` releases them. `graph_unsupported` says why the device
  cannot run a program without allocating anything, `graph_shape_of` what a run takes and gives, `graph_info` the
  plan of a T (lists, records, records per kind, IO bytes), and `graph_run_ref` runs the same op lists on the host
  (§3.7). A program is planned for each T on first use, and the plan is cached (16 plans per graph, least recently
  used out):
  - Values. The inputs, the positions, the KV cell indices and the mask summaries sit at the bottom of the IO
    region (they are preload sources that no record writes). Every other value is placed above them by liveness,
    so a value that is dead gives its bytes to a later one.
  - KV planes. Each `attn` op has its own two planes (K and V, `kv_size` cells), zero-filled at `graph_create`
    and placed at the top of the WEIGHT heap. The op writes the T new rows with SET_ROWS_KV and reads them with
    FLASH_ATTN (`n_kv = T`).
  - Records. Each op becomes the records of the `tsr-kinds` encoders, added to the same `tsr_opl_builder` as the
    ggml backend's, so hazards, JOINs and the deferred `MUL_MAT_BEGIN` / `MUL_MAT_END` pairs are the builder's. A
    list that reaches 4096 records (SW-29) is cut and the next one started, so a run is one launch per list.
  - Run. The run writes the inputs and the static part, launches the lists in order and reads the outputs.

### 3.4 The extension `_tesseract_rt`

`_tesseract_rt.Device` binds `tsr::rt::device` for CPU torch tensors and releases the GIL during device calls. Each
method holds a lease that keeps the device alive until it returns, so `close()` from another thread waits for the
calls in flight, and a later call raises `TesseractError: ... the device is closed`. It binds the graph API too
(`const_upload`, `graph_create`, `graph_run`, `graph_run_ref`, `graph_info` and the others of §3.3), with programs
as Python dicts and values as torch tensors. The API is in
[`sw/torch/README.md`](../sw/torch/README.md).

### 3.5 The partitioner (backend.py, lower.py)

In full mode the backend forms regions first (`partition_regions`). `lower.form_regions` looks at the aten graph
and builds plain Python data. It imports no extension, and what needs the device is injected as a callable.
Then the matmul partitioner below runs on what is left.

- Candidates. Each pattern of §4 that matches a node and its neighbours gives one device op (a candidate), with
  the aten nodes it replaces. The layout rules say when an aten node is a view of a device value, which is a
  float32 `[T, n]` tensor of token-major rows: `view`, `reshape`, `unsqueeze`, `squeeze`, a float32 copy and a
  full slice are identities, and `transpose(1, 2)` turns `[1, T, H, D]` into the head-major `[1, H, T, D]`. A `[1, T, 1, D]` view of a `[1, T, D]` value followed by
  `transpose(1, 2)` is a head-major value of one head, so an attention with one KV head or with one head takes its
  Q, K and V projections into its region. The
  grouped KV expansion of attention is seen through. A node whose layout does not fit stays on the CPU, with a
  reason that `TSR_TORCH_DEBUG=1` prints. A scalar (a scale factor, an RMSNorm `eps`, a RoPE factor) is a runtime
  scalar only through float32 or float64 identity and convert nodes. A scalar or constant seen through a
  narrowing or integer cast, or through a float16, bfloat16 or float8 cast, stays a CPU op. With
  `attention="cpu"` no attention is lowered, so a region ends at each one and no KV plane is used.
- Probe. The backend asks the device about every candidate as a one-op program (`graph_unsupported`, no
  handles, answers memoized per op shape). A candidate the device refuses stays on the CPU with the device's
  reason.
- Regions. Candidates are grouped into regions that are convex: no path leads from a value of the region
  through a CPU node back into the region, so a CPU node splits a region. A group of fewer than two ops is not a
  region, and its matmul keeps its `device_mm` node. A graph with a bf16 or f16 value gets matmul nodes only. One
  graph can hold two or more regions. An output of an earlier region, or the root of one of its ops, can be an input
  of a later region, and the root is rebuilt from the rows. Two regions never read each other: when a region would read, through nodes that
  stay on the CPU, a value of another region that reads it back the same way, the op that reads moves to a later
  region, and the graph keeps full mode. The regions are rewritten in an order in which a region comes after every
  region whose value it reads, directly or through CPU nodes, so the order is not always the graph order.
- Outputs. A region output has exactly the strides the node has in eager. The rebuilt form is `reshape`, plus
  `transpose(1, 2)` for a head-major value. When eager's value is contiguous, the output is a contiguous copy.
  Otherwise the op stays on the CPU, with the reason "output shape or layout cannot be rebuilt from the region
  inputs". The same holds with a symbolic T. A value and any number of its views can leave a region together. The
  device returns one `[T, n]` tensor per distinct value. The outputs that share storage in eager (no `clone`,
  `contiguous` or `_to_copy` between them along the view chain; `_unsafe_view` counts as a view) share it after the
  rewrite, so an in-place write to one is seen by the others, and outputs that a copy separates do not share. This
  holds on the device path and on the CPU fallback, static and dynamic. A group of two or more sharing outputs with
  a member that needs the contiguous copy is refused in the same way, and its op stays on the CPU.
- Expansion of grouped KV. The expanded K or V (`expand` and `reshape`, or `repeat_interleave`) that the region's
  attention reads is seen through. When the expansion is also returned or read by a CPU op, the region still keeps
  the attention. The expansion node stays in the graph for its other users, and the saved CPU subgraph of the
  region computes it itself.
- Order. A region call, and its CPU fallback, runs before the first in-place update (a `copy_` epilogue or any op
  with a mutable schema) of anything it reads, through views. It reads the values from before the update, as eager
  does.
- Rewrite. The nodes of a region are replaced by one `device_region` call, with `getitem` and the aten
  `reshape` and `transpose` nodes that give the outputs their model shape. The call takes the flattened inputs,
  the constants, the weights, the scalars (`eps` and the RoPE factor, when Dynamo lifted them to graph inputs)
  and what the saved subgraph needs. The nodes that only the lowering read (the cos and sin tables, the mask
  arithmetic) are removed with the region when nothing else uses them. The region keeps a copy of its aten
  subgraph for the CPU fallback (§3.8). If forming the regions of a graph raises, that graph is partitioned in
  matmul mode with a warning, and `strict=True` raises instead.

Matmul nodes (matmul mode, and the weight matmuls outside every region in full mode):

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

`TSR_TORCH_DEBUG=1` prints the partition of every graph: each region with its inputs, outputs, weights and
constants, the CPU nodes beside it with the reason, and the matmuls that go to the device or stay on the CPU. It
also prints the records of the first plan of each region. `offload=False` forms no region.

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

Full mode adds these rules for the ops of a region:

- Device numerics. RMS_NORM, GLU, UNARY (the exact tanh GELU of SW-31), ROPE and FLASH_ATTN are the firmware
  kinds with their references (`sw/ctrl/ref`), not PyTorch's ops, so a region is not bit-identical to eager.
  Device attention is llama.cpp's q8_0 path (AT §2 and §6), the numerics that need `-ctk q8_0 -ctv q8_0` there:
  Q and the K and V rows are quantized as q8_0. This adds to the difference from eager float32 that the Q8_0
  weights make. Models with large attention scores (small models, the first positions of a prompt) are the most
  sensitive. `attention="cpu"` (or matmul mode) keeps attention in float32.
- Bias and scalars. A bias is the addend of the MUL_MAT record (matmul mode adds it on the CPU). A division by a
  scalar is a multiplication by `f32(1 / s)`, and every float of a program is exactly a float32.
- Oracle. The reference executor (`tsr_opl_ref`, `sw/ctrl/ref/opl_ref.{hpp,cpp}`) runs the op lists of a plan on
  the host with the per-kind references, and `graph_run_ref` runs it on an image read back from the device. A
  correct device gives its result bit for bit on every transport. `TSR_TORCH_VERIFY=1` compares every region
  output with it and raises on a difference (SW-99).
- What verify covers. The executor runs the planner's own op lists on the constants and weights as they are on
  the device. It checks that the device executes the plan correctly. It does not check the stored weights (the
  `read_blocks` tests do), and not the placement of values (the test that runs a program as one graph and as
  its ops one by one does).
- Eager. Outputs are compared with eager PyTorch statistically (NMSE, KL), not bit for bit (§6). The reference
  executor checks the device bit for bit. Eager is a target, not an oracle.

### 3.8 Errors, fallbacks, threads

A matmul node falls back only for placement: a weight that cannot be placed (WEIGHT heap full, after releasing dead
weights and retrying) runs on the CPU at that node from then on (one warning, counter `cpu_fallback_calls`). So
does a weight uploaded on 4 consecutive calls of a node, because it was changed in place before each call or is
seen through an op that copies it. Everything else is a `TesseractError`: a device that cannot be opened, a
transport error during an upload, a device execution error. Device calls release the GIL, and tsr_rt serializes
them on its mutex.

A region falls back as a whole (SW-101). A call that does not fit the device (T above `kv_size` or above 680
with an attention op, the IO region or a heap full, a mask value that is not 0, -inf or at most -1e9) runs the
region's saved aten subgraph on the CPU. Each region gives one `RuntimeWarning` and counts the call in
`region_cpu_fallback_calls`. A program that the device refuses at `graph_create` makes the region run its saved
subgraph on the CPU for that key: one `RuntimeWarning` per region carries the device's text, the calls count in
`region_cpu_fallback_calls` and in the region's `cpu_calls`, and the device is not asked again for the same key. A
malformed program is a `ValueError`. Every other error is raised as a `TesseractError`, among them a mismatch of
`TSR_TORCH_VERIFY`, and the planner's "value N is read after its slot was released".
Additive float32 masks may use `finfo.min` or any value at most -1e9 for a masked position: the backend turns such
a value into -inf. A mask row whose keys are all masked by finite values at most -1e9 does not fit the device,
because eager gives a uniform softmax there and the device does not. The call runs on the CPU with the warning.
Rows with every entry -inf and bool masks stay on the device. A mask with another non-zero value (-5.0, say) runs
on the CPU with the warning. If forming the regions of a graph raises, the warning "forming regions failed (...);
this graph uses matmul mode" is shown once per compiled graph. Region calls are serialized by one lock.

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

In matmul mode only the weight matmuls of the first table go to the device. In full mode the patterns of the second
table form regions, and the first table applies to the weight matmuls outside a region.

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
| norms, activations, RoPE, softmax, embeddings, residuals, bias adds | CPU in matmul mode | in full mode the patterns below lower them |
| graphs that need autograd | CPU, eager | one warning; `strict=True` raises |
| WEIGHT heap full, or 4 uploads in a row | CPU (that weight, that node) | one warning, `cpu_fallback_calls` |

Full mode: the aten forms of a region and the kind each one lowers to. The exact forms are in `lower.py` and its
tests (`test_lower.py` has the near misses that do not match).

| Aten form | Kind | Note |
| --- | --- | --- |
| `mm`, `addmm`, `linear`, `matmul` with a weight (`x @ Wᵀ`), also `tt.linear_blocks` | MUL_MAT | float32, 2D weight, K % 32 == 0; the bias (`addmm` with alpha = beta = 1, a 1D parameter bias) is the addend; Q8_0 and Q4_K blocks are taken as they are |
| `add` of a bias-free product and another value of the same layout | MUL_MAT | the other value is the addend: a residual add is folded into the product |
| RMSNorm: `[w *] x * rsqrt(mean(x², -1) + eps)`, also Gemma's `1 + w` | RMS_NORM | the weight is a constant row (`1 + w` computed on the host at upload) or absent |
| `silu(g) * u`, `gelu(g, "tanh") * u` | GLU | SWIGLU and GEGLU |
| `silu`, `gelu(approximate="tanh")`, `sigmoid`, `tanh`, `exp` | UNARY | erf GELU is not lowered |
| `add`, `mul` of two values, or of a value and a one-row parameter | ADD, MUL | |
| `mul`, `div`, `add`, `sub` by a scalar | SCALE | a division is a multiplication by `f32(1 / s)` |
| Hugging Face `rotate_half` RoPE, with its cos and sin chain traced to `inv_freq` and the positions | ROPE | the angle table is the constant `inv_freq`; positions are an input |
| `scaled_dot_product_attention` (also its CPU flash form), grouped KV heads, causal, bool or additive mask | SET_ROWS_KV (K, V), FLASH_ATTN | batch 1, head dimension 64, 128, 256 or 512; q, k and v of T rows each; q8_0 numerics (§3.7); not lowered with `attention="cpu"` |

Stays on the CPU in full mode, besides what the first table lists: the embedding lookup, the integer mask and
position arithmetic, the cos and sin tables, LayerNorm, erf GELU and softmax; attention with batch above 1, a
head dimension outside the four, or a mask that is not `[T, T]` (a key-value cache with more key rows than
query rows is such a case); a scalar or constant that passes through a dtype-narrowing copy, or through a float16, bfloat16 or float8 cast; an `add` of the `eps` with
`alpha != 1`; a copy of a head-major value; `scaled_dot_product_attention` with `enable_gqa=True` (grouped KV is lowered when the
model writes the expansion itself); a graph with a bf16 or f16 value; a lone matmul; and every op that
the per-op probe of the device refuses.

## 5. Environment variables

`TSR_DEVICE` and the transport variables: [`sw/transport/README.md`](../sw/transport/README.md).
`TSR_TORCH_MODE` (`full` or `matmul`, default `full`; an explicit `mode` of `tt.compile` or of `options=` wins; an
unknown value is a `ValueError`). `TSR_TORCH_ATTENTION` (`device` or `cpu`, default `device`; full mode only; an
explicit `attention` wins; in full mode an unknown value is a `ValueError` that names `device, cpu`, and matmul mode
ignores it). `TSR_TORCH_DEBUG` (`1` partition, uploads and counters at exit, `2` also every
device call, `3` also the rewritten aten graphs), `TSR_TORCH_VERIFY` (`1`: check every device matmul bit for bit
against libtsr_ref and every region output against the reference executor; a mismatch raises, naming the
weight or region, the element and both values), `TSR_TORCH_EXT_DIR`, `TSR_TORCH_HF_MODEL`
(model directory of the opt-in `test_hf.py`) and `TSR_RT_DEBUG` (tsr_rt's log) are described in
[`sw/torch/README.md`](../sw/torch/README.md). `TSR_TORCH_PYTHON` is a CMake cache variable.
`tt.compile(model, mode=None, kv_size=None, strict=False, debug=None, verify=None, offload=True, attention=None, **kw)` passes the
same switches as `options=` of `torch.compile`. `kv_size` (default 4096) is the number of cells of the KV planes of
each attention op, so the longest sequence of a region call. `offload=False` prints the partition but keeps every
matmul on the CPU and forms no region.

## 6. Tests

`ctest --test-dir build/dev -R '^(torch_pytest|rt_|ggml_tesseract_kinds)'` runs the suites below on the simulator
and on the mock transport with `T_MAX` 1, 4 and 8. The same suites run on the card (§2.4).

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
  bodies). `test_hf.py` is opt-in (Qwen2.5-0.5B-Instruct, 12 plus 4 tokens, and a full-mode forward of 16 tokens).
  These suites run in matmul mode.
- `test-tsr-rt-graph` (`rt_graph_sim`, `rt_graph_sim_tmax4`, `rt_graph_sim_tmax8`, `rt_graph_mock`): the graph API
  against three independent oracles. `graph_run_ref` (the op lists on the host), the per-kind references called
  directly with argument words written in the test, and `linear_ref` for MUL_MAT. `graph_run` must equal
  `graph_run_ref` bit for bit, and the direct references must equal both. The cases are each op alone at several T,
  N and K parts, an addend with K parts, Q8_0 and Q4_K, strided arguments, the mask forms of attention, a decoder
  layer and three layers as one program, the OPL and plan caches, the limits and their error texts, malformed and
  refused programs, `const_free` and `graph_free`, two threads, and negative controls (the comparison must fail
  for a wrong `inv_n`, the other RoPE pairing and a missing FA_NEW).
- `test-tsr-kinds` (`ggml_tesseract_kinds`, `_tmax4`, `_tmax8`, no ggml): every `tsr-kinds` encoder builds lists
  through `tsr_opl_builder`, runs them on the transport, and compares the DRAM they write with the reference
  executor's image bit for bit, and the outputs of MUL_MAT, RMS_NORM, ROPE and SET_ROWS_KV + FLASH_ATTN with the
  per-kind references called with argument words written in the test. It also runs a decoder layer of
  Qwen2.5-0.5B's shape as one list, and has negative controls and the refusals of a kind the image lacks.
- `test_lower.py` (no device): real aten graphs captured as the backend does. It checks which patterns match and
  which near misses do not, the regions that form and their convexity, the program of a region against its
  schema, and the flattening recipes of the inputs and outputs.
- `test_graph.py` (full mode, `TSR_TORCH_VERIFY` on, so every region call is also run by the reference executor):
  every pattern alone between two matmuls, quantized blocks in a region, the decoder as one region and one
  launch, the mask forms, dynamic and changing sequence lengths, weights and constants changed in place and
  released, the fallbacks (T above `kv_size`, a full heap, other errors raised), the modes and their options,
  bf16 graphs, and negative controls (a wrong program must not pass). A placement-free oracle runs a program as
  one graph and as its ops one by one as single-op graphs, so a placement error of the planner shows. The same
  file covers the storage sharing of outputs, the GQA expansion that is also read outside the region, attention with one KV head and with one head, the option
  `attention` (including its environment default and the error of an unknown value), the order of a region call
  before in-place updates, the mask and scalar rules, two regions in one graph, regions that read each other through
  CPU ops, the order of the regions, a view of a CPU value that a region's operand chain and a CPU op both read (it
  does not split the region), and the layout of outputs.
- `test_fuzz.py`: the fixed seeds of both generators of `fuzz_full.py` (the seeds that found defects) and a run of
  fresh seeds of each. It fails when more than 5 % of the fresh cases have an unstable output or an output judged by
  absolute error only (a judge that has gone blind shows there). The rules of the judge are pinned on hand-made
  records: NaN and non-finite mismatches, unstable per output, low energy with the float64 reference, strides,
  sharing and the one-ulp probe. The test takes about a minute and a half.
- `test_hf.py` (opt-in) also runs `hf_check.py --full-mode` with device attention and with `--attention cpu`.

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
hold. `sw/torch/examples/hf_compare.py` runs the same model compiled against eager in float32 or bfloat16, in
matmul mode for its bit-exact check.

`hf_check.py --full-mode [--attention device|cpu]` runs a full-mode forward of the first `--tokens` prompt tokens.
With device attention (the default) the checks are: one region and one launch, no matmul node left, no CPU
fallback, every region call verified bit for bit against the reference executor, a steady second forward, and
top-1 of at least 80 % against eager float32. The steady forward must give the same bits with one more launch and no
upload. With `--attention cpu` the checks "one region", "one launch" and "no matmul node left" become "every region
ran on the device": the region calls, the launches and the verified regions are equal to the number of regions, and
no call fell back to the CPU. The matmuls between the regions stay device matmuls. The steady forward launches once
per region. The tool reports KL(eager f32 || device), KL(eager with q8_0 attention || device) and KL(eager f32 ||
eager with q8_0 attention), each with mean, median, max and top-1, so the quantization of the attention can be told
from the device's other differences. Only the first is gated. The JSON `forward` records `attention`, `regions`,
`region_calls` and `launches`. `--save-logits FILE.npy` stores the logits, and `--expect FILE.npy` checks that the
full-mode logits are bit-identical to such a file (the JSON key `expect` holds `rows`, `bit_identical_rows` and
`sha256_expected`): it compares the simulator with the card.

`sw/torch/tools/fuzz_full.py` is a differential fuzzer of full mode. `fuzz_gen.py` draws a small model from a seed
(linear layers, residuals, norms, activations, GLU, scalars, views, RoPE, attention with masks, CPU-only ops,
in-place updates, a block called twice, outputs that share storage, a padding mask with a fully masked row of `-inf`,
`finfo.min` or `-1e9`, a grouped KV expansion that is also returned or read by a CPU op, a buffer or input changed in
place after its last read and returned, a contiguous head-major copy of a region value that is returned). Each model runs in eager float32, in matmul mode
and in full mode with `TSR_TORCH_VERIFY` on, for four calls (two inputs, a new sequence length, and one after an
in-place change of the weights). `--gen random` takes models with matmuls and `--gen elem` matmul-free models, where
full mode must equal eager to float32 accuracy. Every output of both generators gets one record: shape, dtype,
strides, the count of non-finite mismatches against eager, and the relative NMSE over the elements that are finite in
eager. A case is one of:
  - flagged, with one of these kinds:
    - `nmse`: for `random`, the NMSE against eager is above max(20 x matmul mode's NMSE, floor), the floor being 2e-4
      and 5e-3 when a region has attention; for `elem`, above 1e-8, and 1e-6 for a model with a bf16 or half island.
      An NMSE that is NaN is a flag.
    - `nonfinite`: a value is finite where eager's is not, or the reverse (for `random`: unless matmul mode has a
      mismatch on that output too).
    - `strides`: the strides of an output, in its dimensions of size above one, differ from eager's and from matmul
      mode's. (`elem` compiles the model in matmul mode when a stride differs from eager's: a layout that PyTorch's
      compiled graph gives to every backend is not a defect of full mode.)
    - `sharing`: the storage-sharing pattern of the outputs differs from matmul mode's (`random`) or from eager's
      (`elem`).
    - `shape_dtype`, `outputs` (another number of outputs), `value` (an output that is not float differs),
      `crash_full` (full mode raises where matmul mode does not; for `elem`: where eager does not), `lost_full_mode`
      (a warning that the graph went to matmul mode) or `harness`.
  - low energy: an output that fails its relative bound still passes when its absolute error is small and eager's own
    float32 result is no more accurate than that. Both must hold. The rms error is at most sqrt(bound) x 3e-4 x
    scale, where scale is the largest finite absolute value of that eager output and of the float activation inputs
    of the call (a mask is not an activation; there is no minimum). The relative NMSE is at most 1e-4 and at most
    100 x the NMSE of eager float32 against eager float64 (`elem` runs the model in float64 for this; `random` has no
    float64 run: there the relative NMSE is at most 10 x its limit, and the absolute test is applied to both sides of
    the ratio against matmul mode). An eager output that is exactly zero has no relative error: the absolute error
    alone decides. So an output that is small but exact in float32 is judged by its relative error, and only an
    output that is rounding noise in eager itself (a sum that cancels, a saturated tanh) is judged by absolute error.
  - unstable, per output and not per case: one ulp on every finite value of the float inputs moves that eager output
    by more than the bound, or changes which of its elements are finite. The values of that output are not judged.
    Its non-finite mismatches still are (unless the probe changed the finite pattern itself), and so are its shape,
    dtype and strides, every other output of the case, errors, lost full mode and sharing.
  - `eager_error`: the generator made a model that eager cannot run. It is not flagged.
The summary line gives the number of cases, of flagged cases, of cases with an unstable output, and of cases with an
output judged by absolute error only.
`--replay SEED [--gen elem] [-v]` runs one seed again and prints the model source with `-v`. Every flagged seed
prints its replay command.

## 7. Limitations

- Inference only.
- Matmul mode runs weight matmuls only: bias, activations, norms, RoPE, attention and embeddings run on the CPU,
  and every device matmul is its own launch of the `graph` kernel with x and y crossing the host link each time.
- Full mode: batch 1 through attention. T is at most `kv_size` (default 4096) and at most 680 per attention op. A
  region keeps 16 plans (one per T and mask shape). Region calls are serialized by one lock. The device graphs
  of a region are freed when the compiled function is collected, on `release_weights()` and on `close_device()`.
- No device-resident KV cache across calls. A region's attention reads the T rows of its own call, so generation
  in `hf_check` and `hf_compare` recomputes the whole sequence per step, and every new length is a new plan.
  A Hugging Face step with `past_key_values` has more key rows than query rows (the mask is not `[T, T]`): its
  attention stays on the CPU and ends the regions there, so such a graph is several regions.
- `dynamic=True` on a graph whose row length is symbolic forms no region for that op: a row length must be an
  integer.
- With device attention, Q, K and V are q8_0 (§3.7). A user who needs float32 attention sets `attention="cpu"` or
  uses matmul mode.
- A group of two or more outputs that share storage, with a member that needs the contiguous copy, keeps its op on
  the CPU (§3.5).
- A mask row with every key masked by a finite value at most -1e9, and a mask with other non-zero values, run the
  region on the CPU with a warning (§3.8).
- The CPU part runs as an eager aten graph, not through Inductor.
- The torch parameters stay in host memory next to the device copy. On `sim` the device DRAM is host memory too.
- Float weights always become Q8_0. Native Q4_K and Q8_0 blocks need `QuantLinear`.
- `bmm` and batched weights (MoE expert stacks) stay on the CPU, and so do matmuls in HOP bodies other than
  `invoke_subgraph`, `cond`, `while_loop` and `map`.
- Weight identity is by storage. Writes through `.data`, and inference-tensor changes that miss the 64 x 64
  sample, are not seen (§3.6). This holds for the norm weights, biases and `inv_freq` of a region too.

## 8. Whole-graph lowering

Full mode lowers the aten graph of a model to op lists of the firmware's kinds ([`versioning.md`](versioning.md)).
§3.5 says how regions form and §4 which aten forms lower to which kind. This section says what a region is to the
rest of the system.

- Region. A convex set of at least two lowered ops, float32 only. It is a straight-line program over float32
  `[T, n]` values, the `graph_desc` of `tsr_rt.h`. Its values are inputs given at every call, temporaries on the
  device (some of them outputs), constants, the positions and the attention mask. The stateless forward of a
  Hugging Face Qwen2.5 decoder (Qwen2.5-0.5B-Instruct, say) is one region from the first norm to the logits, so one
  launch. With `attention="cpu"` it is one region per layer and one for the head, and one launch each.
  The embedding lookup, the mask and position arithmetic and the cos and sin tables stay on the CPU in front of it.
- Contract. A region is polymorphic in one number, T, the number of rows (tokens times batch). Every row length
  is a compile-time integer. The op list is planned per T at the first call with that T and cached (§3.3). The
  values that cross the boundary are 2D float32 `[T, n]` tensors of token-major rows, and the aten `view`,
  `reshape` and `transpose` nodes that give them their model shape stay in the graph around the region call, so
  no shape arithmetic runs inside it (SW-100). Weights are given by handle through the weight cache, norm
  weights, biases and the `inv_freq` table by handle through the constant cache (`ConstCache`: the same identity
  rules as §3.6, uploaded once). A weight or constant changed in place is uploaded again and the device graph is
  made again. So is the graph when a scalar (`eps`, the RoPE factor) changes.
- Oracle. The reference executor of §3.7 is the oracle of a region, and eager PyTorch is a statistical target
  (SW-99). Eager attention is float32 and the device's is the q8_0 path, so a model-level comparison names both
  differences (§6). `attention="cpu"` removes the second.
- Fallback. A call that does not fit runs the saved aten subgraph on the CPU (§3.8, SW-101).

Not built:

- A device-resident KV cache with one launch per decode token. It needs a static-cache module whose attention
  reads the cache of earlier calls (AT §4.6), and its oracle would be the simulator's, not eager's, because the
  numerics of attention change from PyTorch's float32 cache.
- Tesseract Pivot for regions, the embedding as GET_ROWS, MoE layers (`MOE_ROUTE`, `MUL_MAT_ID`), Gated DeltaNet
  and SSM_CONV, and batch above 1 through attention.
