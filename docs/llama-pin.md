# llama.cpp pin and verified backend API

Tesseract uses only the pinned llama.cpp commit (`docs/spec.md` §22, Working rules). This file records
the pin and every item that `docs/spec.md` and `docs/design` mark "check at the pinned commit", with
file:line references into `third_party/llama.cpp` **at the pin, before our patch** unless a line says
otherwise; the references stay valid as long as the pin does. Our changes to llama.cpp live in
`third_party/patches/` until a fork exists.

> **Scope.** Verification record for the pin, and the rules of the M6 backend (`sw/ggml-tesseract`,
> whose README is its developer reference) that follow from it. The M6 backend *runs on the
> simulator*; M7's device KV cache and one launch per token are *planned* (W3b). Simulator
> measurements that name no commit were made before the control core was added and are not repeated
> unless stated.
> See [AT](design/03-attention-kv.md) §2 (what llama.cpp hands the device) and [SW](design/06-software-verification.md)
> §8 (one launch per token), §9.3 (the backend).

## Pin

| | |
| --- | --- |
| Repository | https://github.com/ggml-org/llama.cpp (submodule `third_party/llama.cpp`) |
| Commit | `e3797b184d428c7a019362612b76971f60af6670` |
| Tag | `git describe --tags`: `b11810` (the commit carries that tag); one commit after `b11809` (`git describe --tags --match b11809`: `b11809-1-ge3797b184`), 141 commits after `v0.7.0` (`git describe`: `v0.7.0-141-ge3797b184`) |
| Commit date | 2026-10-22 08:30:21 +0300 ("sycl: merge the fp16 and fp32 dequant switches (#30740)") |
| ggml version | 0.27.0 (llama.cpp 0.7.0-dev) |
| `GGML_BACKEND_API_VERSION` | **3** (`ggml/src/ggml-backend-impl.h:11`) |

Patches (apply with `third_party/apply-patches.sh`, idempotent; `--status`, `--reverse`; in the tree
`--status` reports `applied`, which is why `git status` shows the submodule as modified):

- `0001-ggml-tesseract.patch` touches three existing files and adds two:
  - `ggml/CMakeLists.txt`: `option(GGML_TESSERACT)` and `include/ggml-tesseract.h` in
    `GGML_PUBLIC_HEADERS`;
  - `ggml/src/CMakeLists.txt`: `ggml_add_backend(Tesseract)`;
  - `ggml/src/ggml-backend-reg.cpp`: registration under `#ifdef GGML_USE_TESSERACT`, plus
    `ggml_backend_load_best("tesseract")` in `ggml_backend_load_all_from_path` for `GGML_BACKEND_DL`
    builds;
  - new `ggml/include/ggml-tesseract.h`, the public header (`ggml_backend_tesseract_reg`, `_init`,
    `ggml_backend_is_tesseract`, `ggml_backend_tesseract_buffer_type`);
  - new `ggml/src/ggml-tesseract/CMakeLists.txt`, a forwarding stub: it requires `TESSERACT_ROOT` and
    the targets `tsr_ref` and `tsr_transport`, includes `${TESSERACT_ROOT}/sw/ggml-tesseract/sources.cmake`
    (the source list and include directories), builds the library with `ggml_add_backend_library` and
    links `tsr_ref` and `tsr_transport`. `sources.cmake` links `tsr_lower` (the ggml-free part of the
    backend: allocator, tiling, weight layout, the op-list builder and cache) with a deferred
    `target_link_libraries`, so the patch does not change when the backend's file list does.

Build notes: `ggml_add_backend(Tesseract)` maps to `GGML_TESSERACT` / dir `ggml-tesseract` /
define `GGML_USE_TESSERACT` (`ggml/src/CMakeLists.txt:428-439`). As a CMake subproject
llama.cpp defaults `LLAMA_BUILD_COMMON/TESTS/TOOLS/SERVER` to OFF (`CMakeLists.txt:129-135`,
`LLAMA_STANDALONE` at :44-48) and adds `tests/` only with COMMON+TESTS (:244-247); the
Tesseract top-level CMakeLists turns those on. `llama-cli` is built from the server sources
(`tools/CMakeLists.txt`: `cli` only under `LLAMA_BUILD_SERVER`); `llama-completion` is the plain
CLI.

## Backend interface (ggml/src/ggml-backend-impl.h)

All five interfaces are plain structs of function pointers; positional initialisation must
follow this order.

- `ggml_backend_buffer_type_i` (:17-33): `get_name`, `alloc_buffer(buft, size)`,
  `alloc_buffer_n(buft, tensors, n)` (optional), `get_alignment`, `get_max_size` (optional,
  default `SIZE_MAX`), `get_alloc_size(buft, tensor)` (optional, default `ggml_nbytes`),
  `get_alloc_size_n` (optional), `is_host`.
- `ggml_backend_buffer_i` (:50-71): `free_buffer`, `get_base`, `init_tensor` (returns
  `enum ggml_status`), `memset_tensor(buf, t, value, offset, size)`,
  `set_tensor(buf, t, data, offset, size)`, `get_tensor(...)`, `set_tensor_2d` /
  `get_tensor_2d` (optional; `ggml_backend_tensor_set_2d` falls back to a loop of set_tensor),
  `cpy_tensor` (optional), `clear(buf, value)`, `reset` (optional).
- `ggml_backend_buffer` (:73-79) carries `usage` (`GGML_BACKEND_BUFFER_USAGE_ANY` /
  `WEIGHTS` / `COMPUTE`, `ggml/include/ggml-backend.h:51`); new buffers start as ANY
  (`ggml_backend_buffer_init`, `ggml/src/ggml-backend.cpp:236`).
- `ggml_backend_i` (:122-157): `get_name`, `free`, `set_tensor_async`, `get_tensor_async`,
  `set_tensor_2d_async`, `get_tensor_2d_async`, `cpy_tensor_async`, `synchronize`,
  `graph_plan_create/free/update/compute`, `graph_compute(backend, cgraph)`, `event_record`,
  `event_wait`, `graph_optimize(backend, cgraph, params)`. Everything except `get_name`,
  `free`, `graph_compute` may be NULL.
- `ggml_backend_device_i` (:177-219): `get_name`, `get_description`, `get_memory(dev, &free,
  &total)`, `get_type`, `get_props`, `init_backend(dev, params)`, `get_buffer_type`,
  `get_host_buffer_type` (opt), `buffer_from_host_ptr` (opt), `supports_op`, `supports_buft`,
  `offload_op` (opt), `event_new/free/synchronize` (opt).
- `ggml_backend_reg_i` (:231-241): `get_name`, `get_device_count`, `get_device`,
  `get_proc_address`; `ggml_backend_reg` (:243-247) starts with `api_version`.
- Device props (`ggml-backend.h:150-181`): `caps = {async, host_buffer,
  buffer_from_host_ptr, events, mmap_support}`.

The Tesseract backend implements every required entry (`sw/ggml-tesseract/ggml-tesseract.cpp`); all
optional async/event/2d/plan entries are NULL; caps are `{false, false, false, false, true}`.

## Device type

`enum ggml_backend_dev_type` (`ggml/include/ggml-backend.h:136-147`): CPU, GPU, IGPU, ACCEL,
META. llama.cpp collects offload targets in `src/llama.cpp:222-273`: GPU devices go to the
`gpus` list (de-duplicated by `props.device_id` when both are non-NULL), IGPU devices to
`igpus`, CPU and ACCEL are skipped (ACCEL only contributes extra CPU buffer types,
`src/llama-model.cpp:1147`). **Tesseract registers `GGML_BACKEND_DEVICE_TYPE_GPU`** so
`-ngl` offloads layers to it; `device_id` is NULL (no PCI id yet).

## How llama.cpp places weights

- Buffer type lists: for each GPU device `make_gpu_buft_list` (`src/llama-model.cpp:1203`)
  = [split buft if `-sm row`, the device default buft, device extra bufts], followed by the CPU
  list as fallback (:1628-1634). Layer → device: `get_layer_buft_list` (:1676-1686) with
  `i_gpu_start = max(n_layer + 1 - n_gpu_layers, 0)` and `act_gpu_layers = min(n_gpu_layers,
  n_layer + 1)` (:1674-1675), so the device layers are the **last** `n_gpu_layers` of
  `0..n_layer`, where `il = n_layer` is the output layer (:1699). The output layer is therefore
  offloaded first, for **any** `n_gpu_layers >= 1` (`-ngl 1`: only the output layer; `-ngl
  n_layer`: layers 1..n_layer incl. the output; `-ngl n_layer+1` or more: everything). Checked
  on Qwen2.5-0.5B (n_layer 24): `-ngl 1 -v` logs "layer 24 assigned to device Tesseract0",
  "offloading output layer to GPU", "offloading 0 repeating layers". The input layer
  (token_embd) is always on the CPU (:1690). Qwen2 creates `output.weight` if the GGUF has it,
  otherwise a `TENSOR_DUPLICATED` copy of `token_embd.weight` for tied embeddings
  (`src/models/qwen2.cpp:26-30`); the head is `build_lora_mm(model.output, ...)` (:145), run
  only on the rows that are output (`ggml_get_rows(..., inp_out_ids)` in the last layer, :107).
  Both shipped Qwen2.5 GGUFs have a separate `output.weight` (0.5B: Q8_0 [896 x 151936];
  32B: Q6_K [5120 x 152064], in split 5).
- Per tensor `create_tensor` takes the op that consumes the weight (`info.op`, e.g. MUL_MAT
  for projections, MUL for norms, GET_ROWS for token_embd) and calls `select_weight_buft`
  (`src/llama-model-loader.cpp:1075`) → `weight_buft_supported` (:928-1073), which builds the
  op with dummy tensors and asks `ggml_backend_dev_supports_op`.
- **The probe**: for MUL_MAT, `src1 = F32 [ne0, 512, ne2, ne3]`, unallocated (:961-965). The
  weight `w` is the GGUF meta tensor (its GGUF type and shape) with
  `w->buffer = ggml_backend_buft_alloc_buffer(buft, 0)` (:1066), i.e. the size-0 dummy
  `ggml_backend_buffer_init(buft, {}, NULL, 0)` (`ggml/src/ggml-backend.cpp:39-45`): **our
  buft, usage ANY, context NULL, empty iface, data NULL**. supports_op must therefore not touch
  the buffer context and must accept usage ANY. Ops on norms/biases (MUL, ADD) get `false`
  from Tesseract in M6, so those weights land in CPU buffers (M7, planned, supports these ops and places the
  norm weights in AUX).
- **The probe's 512 tokens are not the run's token count** (it ignores `-ub`). A supports_op
  condition on the size of src1/dst therefore decides *placement* for a fixed T = 512. The
  first backend rejected `src1 + dst > IO region (256 MiB)`, i.e. N + K > 131072 at T = 512:
  the output head of Qwen2.5 (151936 / 152064) and Llama-3 (128256) silently stayed on the
  CPU (`CPU_Mapped model buffer size = 275.89 MiB` = token_embd + output.weight for 0.5B)
  while llama.cpp still printed "offloading output layer to GPU". Tesseract's supports_op has
  no token-count condition since (`docs/spec.md` §12, supports_op); activations that do not fit IO are
  staged (see "Compute buffers" below). Now (*measured on the simulator, M1*; the placement is
  unchanged in M6): `Tesseract0 model buffer size = 500.52 MiB`
  (was 362.58), `CPU_Mapped model buffer size = 137.95 MiB` (token_embd only), and
  `GGML_SCHED_DEBUG=2` shows `result_output [Tesseract0] ... output.weight (137M) [Tesseract0]`.
  Qwen2.5-32B Q4_K_M, metadata only (`TSR_DEVICE=sim llama-fit-params -m
  models/qwen2.5-32b-instruct-q4_k_m-00001-of-00005.gguf -ngl 99 -nkvo -c 4096 --fit-print on`,
  which loads with `no_alloc` and reads no weight data, 71 MB RSS): Host model 421 MiB
  (token_embd Q4_K 418 MiB + norms and biases), Tesseract0 model 18504 MiB including the
  609 MiB Q6_K `output.weight` (on the device 4 × 38016 rows × 85 beats × 64 B = 789 MiB of
  Q8_0); Tesseract0 compute 415 MiB at `-ub 512` (the 297 MiB head dst is the chunk kept in host
  memory) and 1700 MiB at `-ub 2048`. The same command on 0.5B gives Host 138 MiB, Tesseract0
  500 MiB model / 317 MiB compute, matching the real run above.
- Weight buffers: `ggml_backend_alloc_ctx_tensors_from_buft` (`src/llama-model.cpp:1966`)
  allocates (calling `init_tensor` for every tensor while usage is still ANY), then
  `ggml_backend_buffer_set_usage(buf, WEIGHTS)` (:1987) **before** any data is loaded. With
  `get_max_size` smaller than the total, the allocation becomes a ggml multi-buffer whose
  `set_usage` propagates to the parts. Tesseract decides each buffer's storage at its first data
  access (usage WEIGHTS → per-tensor placement; otherwise one IO window, or host memory when
  no window is free or the buffer holds only leaf tensors, see "Compute buffers" and "LoRA
  adapters" below).
- `mmap_support = false` in device props would force `--load-mode none` for all devices
  (:1597-1605); Tesseract reports `true`, so tensors are read from the mmap'd GGUF.

## Do `set_tensor` uploads arrive split?

`llama_model_loader::load_all_data` (`src/llama-model-loader.cpp:1560-1840`):

- mmap (default): non-host tensors get **one** `ggml_backend_tensor_set(cur, data, 0, n_size)`
  straight from the mapping (:1742).
- no mmap, no async: one `ggml_backend_tensor_set(cur, read_buf, 0, n_size)` (:1814).
- no mmap with async uploads: chunks of `buffer_size` (1 MiB, or the direct-I/O size) via
  `ggml_backend_tensor_set_async(..., offset, size)` (:1756-1808). Only used if the device
  reports `caps.async && caps.host_buffer && caps.events` (:1620). Tesseract reports all three
  false, so llama.cpp never splits weight uploads at this pin.

The backend nevertheless handles arbitrary `(offset, size)` pieces (staging until every byte
has arrived, then convert + repack), as the spec requires. Only `test-ggml-tesseract`
(`sw/ggml-tesseract/tests`, cases "ANY split upload", "WEIGHTS split upload", "q6_K WEIGHTS
converted, split") exercises it: test-backend-ops sets every MUL_MAT input with one whole-tensor
`ggml_backend_tensor_set` (`init_tensor_uniform`, `tests/test-backend-ops.cpp:90/136/139`; the
two-piece set at :143-144 is for I64 tensors only) into an ANY buffer, which never reaches the
staging path.

## Scheduler: splits, split inputs, debugging

`ggml/src/ggml-backend.cpp`:

- `GGML_SCHED_MAX_BACKENDS 16` (:903-905), `GGML_SCHED_MAX_SPLIT_INPUTS 30` (:907-909),
  `GGML_SCHED_MAX_COPIES 4` (:911-913; `GGML_SCHED_MAX_COPIES` is also a CMake cache var,
  `ggml/CMakeLists.txt:188`).
- **No hard limit on the number of splits**: `splits` starts with capacity 16 (:1994) and
  doubles on demand (:1495-1503). `GGML_SCHED_MAX_SPLIT_INPUTS` is only the initial
  capacity of a split's input list, which also grows (`ggml_backend_sched_split_inputs_grow`,
  :996-1009; graph inputs :1011-1024). The scheduler's tensor context is sized for
  `graph_size * 30 * 2` input copies (:1991) and the split graph copy is sized per graph
  (:1639-1650), so hundreds of splits per token are fine. Each split costs one
  `graph_compute` call plus the input copies; a copied input is copied once per backend and
  reused by later splits (`tensor_id_copy`, pass 5, :1445-1549).
- Assignment rules relevant to Tesseract: a node whose pre-allocated buffer cannot run the op aborts
  ("pre-allocated tensor ... cannot run the operation", :1085-1090), which is why Tesseract
  reports the no-op ops (NONE/VIEW/RESHAPE/PERMUTE/TRANSPOSE) as supported, like every GPU
  backend. Ops with a WEIGHTS-usage source run on the backend that holds the weight
  (:1099-1129); `offload_op` (false for Tesseract) is only consulted for weights in host
  buffers.
- Inputs from other backends are copied with `ggml_backend_tensor_copy` → `set_tensor` /
  `get_tensor` (:1823-1863, :628-649) because Tesseract has no async/cpy entries.
- **`GGML_SCHED_DEBUG=1`** prints every split and its inputs, `=2` also every node with its
  backend and cause (`ggml_backend_sched_new` :1962-1963, printing :1145-1183 called at
  :1551-1553). Output goes through `GGML_LOG_DEBUG`, so llama.cpp tools need `-v`.
  `GGML_SCHED_DEBUG_REALLOC` reports graph reallocations (:1969-1970).
- Measured split structure (on the simulator, M1; Qwen2.5 dense; `GGML_SCHED_DEBUG=1 llama-completion -m
  qwen2.5-0.5b-instruct-q8_0.gguf -ngl 99 -nkvo -c 512 -v`): every graph has **290 splits, 145
  on Tesseract0**, i.e. **6 Tesseract splits per layer** plus one for the output head. q, k, v
  and o are each followed by a CPU op (bias ADD, RoPE / attention, residual), but `ffn_gate` and
  `ffn_up` read the same input and share one split (the next CPU split takes both as inputs).
  For 64 layers that is ≈385 Tesseract splits (≈770 in all) per token. M6 has the same splits, one launch
  each: 145.00 launches per token on the 0.5B decode and 385 on the 32B (*measured on the simulator,
  83ac601*); M7 aims at one.

## KV cache placement (M6 constraint, M7 plan)

`llama_kv_cache` puts layer `il`'s K/V in `ggml_backend_dev_buffer_type(model.dev_layer(il))`
whenever `offload` is set (`src/llama-kv-cache.cpp:212-221`), without asking supports_op, and
clears the buffer right away (:293). **M6** (the backend in the tree) runs no attention ops, so
**llama.cpp must run with `--no-kv-offload` (`-nkvo`, `common/arg.cpp:2419`)**, i.e. `offload_kqv` =
false: attention, with the KV cache and its updates, stays on the CPU. Without it the KV buffer (an
ANY buffer of leaf tensors) is kept in host memory, and the scheduler aborts as soon as it splits a
graph that writes the cache, already at the context's graph reservation:
`ggml-backend.cpp:1089: pre-allocated tensor (cache_k_l0 (view)) in a buffer (Tesseract0) that
cannot run the operation (SET_ROWS)` (checked on the simulator, `-ngl 99 -c 512`; M6's supports_op
is the same for these ops).

**M7** (planned, W3b; decision D17) moves the KV cache to the device: run without `-nkvo`
(`offload_kqv` = true) with `-ctk q8_0 -ctv q8_0 -fa on`, because the pinned llama.cpp refuses a
quantized V cache without flash attention (`src/llama-context.cpp:566-569`). The backend then supports
SET_ROWS into KV tensors (and only those) and `FLASH_ATTN_EXT`, the device KV planes hold exactly
llama.cpp's q8_0 values ([AT](design/03-attention-kv.md) §2, §4), and the scheduler no longer aborts
because the cache is a Tesseract buffer that can run its ops (SW §8 R1, R3). The M6 settings, which
`klcheck` and `numcheck` use as baselines, are `-ngl 99 -nkvo -fa on -ctk q8_0 -ctv q8_0` with
`LLAMA_ATTN_ROT_DISABLE=1` (SW §10.4, SW-22).

With a **quantized KV cache** (`-ctk q8_0` / `-ctv q8_0`) llama.cpp rotates Q and K/V with a Hadamard
matrix (`src/llama-kv-cache.cpp:325-351`) unless `LLAMA_ATTN_ROT_DISABLE` is set
(`src/llama-kv-cache.cpp:317`): `llama_mul_mat_hadamard` (`src/llama-impl.h:57`) emits
`ggml_mul_mat(rot, x)` whose src0 `attn_inp_k_rot` / `attn_inp_v_rot` is an F32 [64, 64] graph input
(`ggml_set_input`, `src/llama-kv-cache.cpp:1472-1492`). It is not a weight, so Tesseract's supports_op
rejects it (unallocated src0 that is a graph input) and the rotation stays on the CPU with the rest of
attention. Before this rule the scheduler put these MUL_MATs on Tesseract and the backend converted
the matrix to Q8_0 again for every layer of every token (120 lazy conversions for 4 tokens); with the
rule, 0, same text. In M7 the rotation would split attention into CPU islands, so device runs and all
baselines set `LLAMA_ATTN_ROT_DISABLE=1` (SW §8 R4, SW-16); the backend warns once if it sees
`attn_inp_k_rot` / `attn_inp_v_rot`.

## test-backend-ops

- `-b <name>` compares against `ggml_backend_dev_name(dev)` (`tests/test-backend-ops.cpp:13494`),
  so the selector is the device name: **`test-backend-ops test -o MUL_MAT -b Tesseract0`**
  (`TSR_DEVICE=sim` is the default transport). The comparison backend is the CPU backend with
  its reference implementation enabled (`set_use_ref`, :13139).
- `test_mul_mat` (:5324) tolerance: **NMSE `max_nmse_err() = 5e-4`** (:5344; default for other
  ops `1e-7`, :1215).
- Support is checked for **every** tensor of the test context, including leaves and views,
  **before** allocation (src buffers are NULL) (:1415-1441).
- All tensors of a case (src0, src1, dst and the sentinel tensors that catch out-of-bounds
  writes, :1299) live in **one buffer of usage ANY** from `ggml_backend_alloc_ctx_tensors`
  (:1457). Only cases with `use_weight_context()` get a separate WEIGHTS buffer (:1442-1454):
  `test_bin_bcast` with `weights = true` (override :3473; instances e.g. :10761, :10763) and
  `test_mul_mat_vec_fusion` with `use_id && with_lane_scale` (:7778). Neither is a MUL_MAT case.
- The CPU side gets its inputs by **reading the backend's tensors back**
  (`ggml_backend_compare_graph_backend` → `ggml_backend_graph_copy`,
  `ggml/src/ggml-backend.cpp:2412`, :2324), so `get_tensor` of src0 must return the exact
  bytes; Tesseract keeps ANY-buffer tensors raw in IO and converts MUL_MAT src0 lazily.
- `-j N` runs workers with one backend each on the same device (the backend serialises on a
  device mutex).
- The "very large N" case (`F16 x F32, m=16, n=2105344, k=32`, :11083) has a 257 MiB src1. It
  gets its own buffer (larger than `get_max_size`), which is kept in host memory, so the MUL_MAT
  is staged through the IO stage in 17 tiles of 131072 tokens (≈7 s on the sim). It was
  "not supported" while supports_op still limited src1 + dst to the IO region. The case runs in its own
  ctest, `ggml_tesseract_backend_ops_bigt` (label `slow`): 8.4 M single-token tile ops through the
  interpreted firmware at the simulator's T_MAX 1, about 50 s according to the comment in
  `sw/ggml-tesseract/CMakeLists.txt` (not re-measured).

Result on the real `sim` transport (*measured on the simulator, M1 step 6*): 1403 MUL_MAT cases are
supported, 1462 not supported (882 batched 3D/4D src0, 452 F16/BF16 src1, 67 strided views
`k_v`, 14 permuted non-contiguous, 47 K % 32 != 0). The same 1403 also pass with
`TSR_FORCE_HOST=1 TSR_STAGE_KIB=256` (every buffer in host memory, every MUL_MAT staged). The supported set
covers 28 src0 types (F32, F16, BF16, Q8_0, Q4_K native; every other quantized type converted
to Q8_0 at load), K from 32 to 8192, N from 1 to 17408, up to 4096 tokens. Every supported
case passes in every run, with `TSR_SIM_TMAX=1` and `8` and with `-j 4`, **except one**:

- `MUL_MAT(type_a=f32,type_b=f32,m=1,n=1,k=2048)` has a single output element, so its NMSE
  is the squared relative error of one dot product, and test-backend-ops draws fresh random
  data every run (`std::random_device`, `init_tensor_uniform` :54-66). With the spec's 8-bit
  weights (F32 -> Q8_0 at load) and 8-bit activations (QACT A32), the absolute error of the
  dot product is ~0.08 while its value is ~N(0, 15^2), so the 5e-4 limit is exceeded whenever
  |y| is small: measured 15.4% of 10000 seeded trials (`test-tsr-scalar-nmse 10000`), and
  the *exact* (double) dot product of the same 8-bit operands fails on exactly the same trials,
  i.e. accumulation order contributes nothing. Observed in test-backend-ops: 2 failures in 11
  full runs. The device output is bit-exact with the tsr_ref spec model on every trial.
- Margins of all other cases (instrumented copy of test-backend-ops, 5 runs): the largest NMSE
  is 1.3e-4 (26% of the limit; q2_K / q6_K / IQ types converted to Q8_0, 16 outputs), the median
  per-case maximum 2.9e-5; native Q4_K ~4e-12 (accumulation order only: the x86 CPU
  backend's `quantize_row_q8_K` calls `quantize_row_q8_K_ref`, so its activations equal QACT
  A256) and native Q8_0 ~2e-8. The Q8_0 figure is **not** accumulation order: the x86 CPU
  backend quantizes Q8_0 activations with the AVX `quantize_row_q8_0`
  (`ggml-cpu/arch/x86/quants.c:302-346`: `id = 127.f/maxScalar` in one rounding,
  `_mm256_round_ps(_MM_ROUND_NEAREST)` = half-even), while QACT A32 is
  `quantize_row_q8_0_ref` (`d = amax/127`, `id = 1/d`, `roundf` = half away). They differ in
  about 1 of 3-4e5 activation values on random data (e.g. x = {127, 2.5, -0.5, 0, ...}: CPU
  q = {127, 2, 0}, ref/QACT {127, 3, -1}). In a standalone model of these shapes (M=16,
  K=256, uniform [-1,1]) accumulation order alone gives NMSE <= ~5e-14 and the activation
  quantizer difference up to ~1e-7 (N=16) / ~3e-6 (N=1), still far below 5e-4. So CPU-backend
  Q8_0 activations/outputs are not a bit-exact oracle for QACT A32 (`docs/spec.md` §6, Quantized activation (QA) layout).

The ctest `ggml_tesseract_backend_ops` therefore runs the full MUL_MAT suite with tolerances untouched
but leaves out that one shape and the very-large-N case (`-p
"^(?!type_a=f32,type_b=f32,m=1,n=1,k=2048,|type_a=f16,type_b=f32,m=16,n=2105344,k=32,)"`); the
ctest `ggml_tesseract_scalar_nmse` covers it with a stricter oracle (bit-exact vs the spec model
on 2000 seeded trials, failures identical to the exact 8-bit dot product) and `ggml_tesseract_backend_ops_bigt`
runs the large case.

`TSR_TRACE=<dir>` dumps one trace per launch (SW §5.5, §9.2); `traces/README.md` has the replay tools.

llama.cpp's own model tests (`test-llama-context`, `test-llama-context-model`,
`test-thread-safety`) load models with default offload, which puts the KV cache in a
Tesseract buffer and aborts in the scheduler (SET_ROWS on `cache_k`, the M6 `-nkvo`
constraint above). In this build they run with `TSR_DEVICE=none`
(`sw/ggml-tesseract/CMakeLists.txt`).

## For M7: KV padding and graph reuse

- `n_kv` of a ubatch is padded: `n_pad_cur = max(n_pad, 256)`,
  `n_kv = min(kv_size, max(256, GGML_PAD(used_max_p1, 256)))` (`src/llama-kv-cache.cpp:1285-1298`),
  "so that the graph remains constant across batches and can be reused". KV bucket = 256 cells.
  `kv_size % n_pad == 0` is asserted (:100).
- Graph reuse: `llama_context::process_ubatch` reuses the previous graph if
  `res->can_reuse(gparams)` (`src/llama-context.cpp:1530-1574`; `llm_graph_result::can_reuse`,
  `src/llama-graph.cpp:1478`, per-input `can_reuse` e.g. the KQ mask :49, :563). On reuse there
  is no `ggml_backend_sched_reset` / `alloc_graph`: the same split graphs with the same tensor
  addresses are computed again, only the inputs are re-set, which is what the Tesseract
  op-list cache keys on. `LLAMA_GRAPH_REUSE_DISABLE=1` turns reuse off
  (`src/llama-context.cpp:378-379`).

  M7 status: planned (W3b). The op-list cache of the backend keys on the graph identity and the node
  addresses and relies on this reuse (SW §8 R7): per token the host writes the graph inputs
  (`inp_pos`, `kq_mask`, `k_idxs`, `v_idxs`, `inp_out_ids`), launches once and reads the logits, and
  a new KV bucket builds one new op list. In M6 the same reuse gives the op-list cache its hits: 73,385
  hits and 855 misses (98.9 %) on the Qwen2.5-0.5B decode (*measured on the simulator, 83ac601*).

## For M8: sampling

The pin has an **experimental backend-sampling API**: samplers can append ggml ops to the
decode graph (`llama_sampler_i::backend_init/backend_apply/...`, `include/llama.h:1394-1425`;
context config `llama_context_params.samplers`, :419-423; results via
`llama_get_sampled_token_ith` etc., :1172-1200; CLI `-bs/--backend-sampling`,
`common/arg.cpp:2310`). Greedy uses `ggml_argmax` (`src/llama-sampler.cpp:1076-1090`), top-k
`ggml_top_k` (:1483-1495), top-p `ggml_argsort` (:1633-1655). So sampling can run on the device
inside `llama_decode` once Tesseract supports `ARGMAX` (TOPK k=1) / `TOP_K` (the vector unit's TOPK:
simulator in M6, RTL in M8); otherwise it stays on the host and the resident decode loop `sw/runner`
(M9, planned; the directory does not exist) remains the plan (`docs/spec.md` §15, Host independence
roadmap, and §20, Open items). M7 uses no backend sampling (`-bs` off, SW §8 R9): sampling stays on the
host until M8.

## Other notes for running llama.cpp on Tesseract

- `n_gpu_layers` defaults to -1 = auto/fit (`common/common.h:480`); pass `-ngl 99` (any value
  > n_layer puts every layer on the device; the output layer is on it for any value >= 1, see
  "How llama.cpp places weights") explicitly.
- `get_memory` reports the WEIGHT heap of all channels (≈ 4 × 15.5 GiB with the U200 map).
- **Compute buffers** (src1 copies and MUL_MAT outputs). `get_max_size` is the IO window
  capacity (256 MiB IO − 16 MiB stage = 240 MiB), so ggml-alloc cuts the Tesseract compute
  buffer into chunks of at most 240 MiB, and a single larger tensor gets a chunk of its own
  (`ggml/src/ggml-alloc.c:160-178`). A chunk that fits a free IO window at its first data access
  is an IO window; any other chunk is kept in host memory and the MUL_MATs that use it are
  staged through the IO stage in T tiles (`docs/spec.md` §12). There is no `-ub` limit
  from IO. llama.cpp reserves the compute buffer for the worst case,
  `n_outputs = min(n_ubatch, n_outputs_max)` logits (`src/llama-context.cpp:786-797`;
  `n_outputs_max` defaults to `n_batch`, :346), although decode and prefill only compute the
  output rows: for Qwen2.5-0.5B at `-ub 512`, `Tesseract0 compute buffer size = 317.50 MiB` =
  20.75 MiB in an IO window + the 296.75 MiB head dst (151936 × 512 × 4 B) in host memory
  (`TSR_DEBUG=1`: "buffer #5 (296.75 MiB, COMPUTE): host memory (no free IO window, staged)").
  Pages of that chunk are only touched for the rows actually computed. klcheck sets
  `n_outputs_max` to the rows it needs, so there everything is in IO.
  M7 (planned) replaces these windows by a per-channel compute heap between the top of the weights and
  the bottom of the KV planes ([EX](design/01-execution-model.md) §11), sized from the reserve graph, so
  that the reserved logits chunk stays on the device (Qwen 152,064 × 512 × 4 B = 297 MiB at `-ub 512`;
  Gemma 4 262,144 × 512 × 4 B = 512 MiB) and no launch is staged in decode (SW §8 R6, SW-26); a chunk
  that still does not fit is staged and costs one extra launch per token, reported by `TSR_STATS` as
  "staged chunks".
- **LoRA adapters** (`--lora`): `llama_adapter_lora_init` allocates the adapter tensors on the
  buffer type of the weight they modify (`src/llama-adapter.cpp:343`, alloc :393, upload :414),
  i.e. on Tesseract for offloaded layers, as ANY buffers, before the context exists
  (`common/common.cpp:1321` vs :1376). A buffer of leaf tensors only is kept in host memory (it
  takes no IO), and each `lora_a` / `lora_b` is converted to Q8_0 into the WEIGHT heap the first
  time a MUL_MAT uses it (`build_lora_mm`, `src/llama-graph.cpp:1629-1660`). Checked with a
  synthetic rank-128 F32 adapter on all seven projections of Qwen2.5-0.5B (268.5 MiB, which
  used to abort with an IO-region message): loads, 336 conversions, greedy text identical to
  the CPU with the adapter.
