# llama.cpp pin and verified backend API

Tesseract uses only the pinned llama.cpp commit. This file records the pin, the patch policy and the facts of
llama.cpp that the backend (`sw/ggml-tesseract`, whose README is its developer reference) depends on, with file
and line references into `third_party/llama.cpp` at the pin, before our patch. The references stay valid as long
as the pin does. See also [AT](design/03-attention-kv.md) §2 (what llama.cpp hands the device) and
[SW](design/06-software-verification.md) §8 (one launch per token) and §9.3 (the backend).

## Pin

| | |
| --- | --- |
| Repository | https://github.com/ggml-org/llama.cpp (submodule `third_party/llama.cpp`) |
| Commit | `e3797b184d428c7a019362612b76971f60af6670` |
| Tag | `b11810` (`git describe --tags`) |
| ggml version | 0.27.0 (llama.cpp 0.7.0-dev) |
| `GGML_BACKEND_API_VERSION` | 3 (`ggml/src/ggml-backend-impl.h:11`) |

## Patches

Our changes to llama.cpp live in `third_party/patches/`, one numbered file per change. Nothing under
`third_party/` is edited by hand. `third_party/apply-patches.sh` applies the patches in name order and is
idempotent (`--status` reports `applied` or not, `--reverse` un-applies them). With the patch applied `git
status` shows the submodule as modified. A new llama.cpp change is a new numbered patch file. The script's header
says how to regenerate a patch after editing the hook inside the submodule.

`0001-ggml-tesseract.patch` touches three files and adds two:

- `ggml/CMakeLists.txt`: `option(GGML_TESSERACT)` and `include/ggml-tesseract.h` in `GGML_PUBLIC_HEADERS`.
- `ggml/src/CMakeLists.txt`: `ggml_add_backend(Tesseract)`, which maps to `GGML_TESSERACT`, the directory
  `ggml-tesseract` and the define `GGML_USE_TESSERACT` (:428-439).
- `ggml/src/ggml-backend-reg.cpp`: registration under `#ifdef GGML_USE_TESSERACT`, plus
  `ggml_backend_load_best("tesseract")` for `GGML_BACKEND_DL` builds.
- New `ggml/include/ggml-tesseract.h`, the public header.
- New `ggml/src/ggml-tesseract/CMakeLists.txt`, a forwarding stub. It requires `TESSERACT_ROOT` and the targets
  `tsr_ref` and `tsr_transport`, includes `${TESSERACT_ROOT}/sw/ggml-tesseract/sources.cmake` (the source list)
  and builds the library with `ggml_add_backend_library`. `sources.cmake` links `tsr_lower` with a deferred
  `target_link_libraries`, so the patch does not change when the backend's file list does.

As a CMake subproject llama.cpp defaults `LLAMA_BUILD_COMMON`, `_TESTS`, `_TOOLS` and `_SERVER` to OFF
(`CMakeLists.txt:129-135`). The Tesseract top-level CMakeLists turns them on. `llama-cli` is built from the
server sources, and `llama-completion` is the plain CLI.

## Device type

llama.cpp collects offload targets in `src/llama.cpp:222-273`: GPU devices go to the `gpus` list, IGPU devices to
`igpus`, and CPU and ACCEL are skipped. Tesseract registers `GGML_BACKEND_DEVICE_TYPE_GPU` (`ggml-backend.h:136-147`),
so `-ngl` offloads layers to it, with `device_id` NULL. The backend implements every required entry of the backend,
device and buffer interfaces (`ggml/src/ggml-backend-impl.h`). Every optional async, event, 2d and plan entry is NULL,
and the device caps (`ggml-backend.h:150-181`) are `{async, host_buffer, buffer_from_host_ptr, events, mmap_support}
= {false, false, false, false, true}`. New buffers start with usage ANY (`ggml/src/ggml-backend.cpp:236`).

## How llama.cpp places weights

- Layer to device. `make_gpu_buft_list` (`src/llama-model.cpp:1203`) gives the device buffer type followed by the
  CPU list as fallback. `get_layer_buft_list` (:1676-1686) uses `i_gpu_start = max(n_layer + 1 - n_gpu_layers, 0)`,
  so the device layers are the last `n_gpu_layers` of `0..n_layer`, where `il = n_layer` is the output layer
  (:1699). The output layer is therefore offloaded first, for any `n_gpu_layers >= 1`. The input layer
  (token_embd) is always on the CPU (:1690). A model with tied embeddings gets a `TENSOR_DUPLICATED` copy of
  `token_embd.weight` as its output weight, and the head runs only on the rows that are output.
- The probe. `create_tensor` takes the op that consumes the weight (MUL_MAT for projections, MUL for norms,
  GET_ROWS for token_embd) and calls `weight_buft_supported` (`src/llama-model-loader.cpp:928-1073`), which builds
  the op with dummy tensors and asks `ggml_backend_dev_supports_op`. For MUL_MAT, `src1 = F32 [ne0, 512, ne2,
  ne3]`, unallocated (:961-965). The weight `w` is the GGUF meta tensor with `w->buffer =
  ggml_backend_buft_alloc_buffer(buft, 0)` (:1066), the size-0 dummy buffer: our buft, usage ANY, context NULL,
  empty iface, data NULL (`ggml/src/ggml-backend.cpp:39-45`). `supports_op` must therefore not touch the buffer
  context and must accept usage ANY.
- The probe's 512 tokens are not the run's token count (it ignores `-ub`). A `supports_op` condition on the size
  of src1 or dst would decide placement for a fixed T = 512 and keep every large-vocabulary output head on the
  CPU, while llama.cpp still printed "offloading output layer to GPU". The backend's `supports_op` therefore has
  no token-count condition, and activations that do not fit the IO region are staged.
- Weight buffers. `ggml_backend_alloc_ctx_tensors_from_buft` (`src/llama-model.cpp:1966`) allocates, calling
  `init_tensor` for every tensor while usage is still ANY, then `ggml_backend_buffer_set_usage(buf, WEIGHTS)`
  (:1987) before any data is loaded. The backend decides each buffer's storage at its first data access.
  `mmap_support = false` would force `--load-mode none` for all devices (:1597-1605). Tesseract reports `true`,
  so tensors are read from the mmap'd GGUF.

## MoE (MUL_MAT_ID)

Design and decisions: SW §7.9. The probe: for `GGML_OP_MUL_MAT_ID`, `weight_buft_supported` builds `w`, `b = F32 [w->ne[0],
n_ids_used, 512]` and `ids = I32 [n_ids_used, 512]` (`src/llama-model-loader.cpp:966-974`). `w` is the real 3D expert
tensor with the size-0 dummy buffer, and `b` and `ids` are unallocated leaves, so `supports_op` must accept an
unallocated src1 and src2 and look only at shapes and types.

The graph (`build_moe_ffn`, `src/llama-graph.cpp`) is: router logits (MUL_MAT), SOFT_MAX, ARGSORT descending with a
view of the first k numbers, GET_ROWS of the probabilities, SUM_ROWS, CLAMP at 6.1035e-5, DIV; MUL_MAT_ID for the
gate-up (src1 `[n_embd, 1, n_tokens]`, broadcast over the used slots), GLU, MUL_MAT_ID for the down, the per-expert
scale (GET_ROWS of the scale by the ids, MUL), the MUL by the weights, then k views and k - 1 ADDs. The ids are the
strided ARGSORT view. Gemma 4 26B-A4B (`src/models/gemma4.cpp:111-125`) has `ffn_gate_inp` (the router, whose
presence marks a MoE layer), `ffn_gate_inp.scale`, `ffn_gate_up_exps` (gate and up in one Q4_K tensor; the loader
also accepts separate `ffn_gate_exps` and `ffn_up_exps`), `ffn_down_exps` (Q5_0 is converted to Q8_0 at load) and
`ffn_down_exps.scale`, a per-expert factor after the down projection.

- Sorting is `std::sort` with `>` (`ggml/src/ggml-cpu/ops.cpp`, `ggml_compute_forward_argsort`). Equal values leave
  in an unspecified order and NaN violates the strict weak ordering. The device's rule is SW-63.
- `ggml_node_has_n_uses` is false for view nodes and for nodes with the OUTPUT flag, so a fusion pass counts uses
  with `ggml_node_get_use_count`.
- The allocator may put a MUL's result over one of the MUL_MAT_ID's inputs (in-place reuse), so a record that writes
  the MUL's tensor directly from the MUL_MAT_ID's loads can overwrite what it still reads. SW-64 has the device write
  the MUL_MAT_ID's own tensor and copy.
- `test-backend-ops test -o MUL_MAT_ID -b Tesseract0` (ctest `ggml_tesseract_backend_ops_mul_mat_id`) runs the
  supported cases. Cases with m not a multiple of 16, k not a multiple of the block, or iq2_xxs are refused.

## Do `set_tensor` uploads arrive split?

`llama_model_loader::load_all_data` (`src/llama-model-loader.cpp:1560-1840`) gives non-host tensors one
`ggml_backend_tensor_set(cur, data, 0, n_size)`, straight from the mapping with mmap and from a read buffer without
it. Only a no-mmap load with async uploads sends chunks, and only if the device reports `caps.async &&
caps.host_buffer && caps.events` (:1620). Tesseract reports all three false, so llama.cpp never splits weight uploads
at this pin. The backend nevertheless handles arbitrary `(offset, size)` pieces (staging until every byte has
arrived, then convert and repack). Only `test-ggml-tesseract` exercises it ("ANY split upload", "WEIGHTS split
upload", "q6_K WEIGHTS converted, split"), because test-backend-ops sets every input with one whole-tensor
`ggml_backend_tensor_set` into an ANY buffer.

## Scheduler: splits, split inputs, debugging

`ggml/src/ggml-backend.cpp`: there is no hard limit on the number of splits (`splits` starts with capacity 16 and
doubles on demand, :1495-1503). Each split costs one `graph_compute` call plus the input copies, and a copied input
is copied once per backend and reused by later splits (:1445-1549). A node whose pre-allocated buffer cannot run the
op aborts ("pre-allocated tensor ... cannot run the operation", :1085-1090), which is why Tesseract reports the
no-op ops (NONE, VIEW, RESHAPE, PERMUTE, TRANSPOSE) as supported, like every GPU backend. Ops with a WEIGHTS-usage
source run on the backend that holds the weight (:1099-1129), and `offload_op` (false for Tesseract) is only
consulted for weights in host buffers. Inputs from other backends are copied with `ggml_backend_tensor_copy`
(`set_tensor` and `get_tensor`, :1823-1863), because Tesseract has no async or cpy entries. `GGML_SCHED_DEBUG=1`
prints every split and its inputs, `=2` also every node with its backend and cause (output through `GGML_LOG_DEBUG`,
so llama.cpp tools need `-v`).

With only the matrix products on the device (M6 mode), q, k, v and o are each followed by a CPU op (bias ADD, RoPE or
attention, residual), but `ffn_gate` and `ffn_up` read the same input and share one split. A dense layer is 6
Tesseract splits plus one for the output head, each split one launch. In M7 mode the whole decode graph is one
Tesseract split.

## KV cache placement

`llama_kv_cache` puts layer `il`'s K and V in `ggml_backend_dev_buffer_type(model.dev_layer(il))` whenever
`offload` is set (`src/llama-kv-cache.cpp:212-221`), without asking `supports_op`, and clears the buffer right
away (:293).

M6 mode runs no attention ops on the device, so llama.cpp runs with `--no-kv-offload` (`-nkvo`,
`common/arg.cpp:2419`, `offload_kqv = false`): attention, with the KV cache and its updates, stays on the CPU.
Without it the KV buffer (an ANY buffer of leaf tensors) is kept in host memory, and the scheduler aborts as soon
as it splits a graph that writes the cache, already at the context's graph reservation: `pre-allocated tensor
(cache_k_l0 (view)) in a buffer (Tesseract0) that cannot run the operation (SET_ROWS)`.

M7 mode moves the KV cache to the device: no `-nkvo` (`offload_kqv = true`), with `-ctk q8_0 -ctv q8_0 -fa on`,
because the pinned llama.cpp refuses a quantized V cache without flash attention (`src/llama-context.cpp:566-569`).
The backend then supports SET_ROWS into KV tensors (and only those) and `FLASH_ATTN_EXT`, the device KV planes
hold exactly llama.cpp's q8_0 values ([AT](design/03-attention-kv.md) §2, §4), and the scheduler does not abort,
because the cache is a Tesseract buffer that can run its ops (SW §8 R1, R3). The settings `klcheck` and
`numcheck` use are `-ngl 99 -fa on -ctk q8_0 -ctv q8_0` with `LLAMA_ATTN_ROT_DISABLE=1` (SW §10.4, SW-22), and
`-nkvo` in M6 mode.

With a quantized KV cache llama.cpp rotates Q and K/V with a Hadamard matrix (`src/llama-kv-cache.cpp:325-351`)
unless `LLAMA_ATTN_ROT_DISABLE` is set (:317). `llama_mul_mat_hadamard` (`src/llama-impl.h:57`) emits
`ggml_mul_mat(rot, x)` whose src0 `attn_inp_k_rot` or `attn_inp_v_rot` is an F32 [64, 64] graph input
(:1472-1492). It is not a weight, so Tesseract's `supports_op` rejects it and the rotation stays on the CPU with
the rest of attention. The rotation would split attention into CPU islands, so device runs and all checks set
`LLAMA_ATTN_ROT_DISABLE=1` (SW §8 R4, SW-16). The backend warns once if it sees `attn_inp_k_rot` or
`attn_inp_v_rot`.

## test-backend-ops

`-b <name>` compares against `ggml_backend_dev_name(dev)`, so the selector is the device name: `test-backend-ops test -o
MUL_MAT -b Tesseract0` (`TSR_DEVICE=sim` is the default transport). The comparison backend is the CPU backend with
its reference implementation enabled (`set_use_ref`). `test_mul_mat` has NMSE tolerance `max_nmse_err() = 5e-4`
(`tests/test-backend-ops.cpp:5344`; the default for other ops is `1e-7`). Support is checked for every tensor of the
test context, including leaves and views, before allocation (src buffers are NULL). All tensors of a case, with the
sentinel tensors that catch out-of-bounds writes, live in one buffer of usage ANY (:1457). Only cases with
`use_weight_context()` get a separate WEIGHTS buffer, and none of them is a MUL_MAT case. The CPU side gets its inputs
by reading the backend's tensors back (`ggml_backend_compare_graph_backend`), so `get_tensor` of src0 must return
the exact bytes. Tesseract keeps ANY-buffer tensors raw in IO and converts MUL_MAT src0 lazily. `-j N` runs workers
with one backend each on the same device, serialized by a device mutex. The "very large N" case (`F16 x F32, m=16,
n=2105344, k=32`) has a 257 MiB src1. It gets its own buffer, which is kept in host memory, so the MUL_MAT is staged
through the IO stage in tiles of 131072 tokens. It runs in its own ctest, `ggml_tesseract_backend_ops_bigt` (label
`slow`).

The supported set covers 28 src0 types (F32, F16, BF16, Q8_0 and Q4_K native, every other quantized type converted
to Q8_0 at load), K from 32 to 8192, N from 1 to 17408 and up to 4096 tokens. Batched 3D and 4D src0, F16 and BF16
src1, strided views, non-contiguous permutations and K % 32 != 0 are not supported. The supported cases also pass
with `TSR_FORCE_HOST=1 TSR_STAGE_KIB=256`, with `TSR_SIM_TMAX` 1 and 8, and with `-j 4`.

One case does not pass reliably. `MUL_MAT(type_a=f32,type_b=f32,m=1,n=1,k=2048)` has a single output element, so its
NMSE is the squared relative error of one dot product, and test-backend-ops draws fresh random data every run. With
the spec's 8-bit weights (F32 to Q8_0 at load) and 8-bit activations (QACT A32) the absolute error of the dot product
is about 0.08 while its value is about N(0, 15^2), so the 5e-4 limit is exceeded whenever |y| is small. That happens
in about 15 % of random draws (`test-tsr-scalar-nmse`). The exact (double) dot product of the same 8-bit operands
fails on the same draws, so accumulation order contributes nothing, and the device output is bit-exact with the
`tsr_ref` spec model on every draw. The ctest `ggml_tesseract_backend_ops` runs the full MUL_MAT suite with
tolerances untouched but leaves out that shape and the very-large-N case. `ggml_tesseract_scalar_nmse` covers the
first with a stricter oracle (bit-exact against the spec model on seeded trials, failures identical to the exact
8-bit dot product) and `_bigt` runs the second.

The other cases keep a wide margin. Native Q4_K differs from the CPU in accumulation order only (the x86 CPU
backend's `quantize_row_q8_K` calls `quantize_row_q8_K_ref`, so its activations equal QACT A256). Native Q8_0
differs for another reason: the x86 CPU backend quantizes Q8_0 activations with the AVX `quantize_row_q8_0`
(`ggml-cpu/arch/x86/quants.c:302-346`: `id = 127.f/maxScalar` in one rounding, half-even), while QACT A32 is
`quantize_row_q8_0_ref` (`d = amax/127`, `id = 1/d`, `roundf`, half away). They differ in about 1 of 3e5 activation
values on random data (for `x = {127, 2.5, -0.5, 0, ...}` the CPU gives `q = {127, 2, 0}` and QACT `{127, 3, -1}`).
CPU-backend Q8_0 activations and outputs are therefore not a bit-exact oracle for QACT A32 (`docs/spec.md` §6).

`TSR_TRACE=<dir>` dumps one trace per launch (SW §5.5, §9.2, [`traces/README.md`](../traces/README.md)). llama.cpp's
own model tests (`test-llama-context`, `test-llama-context-model`, `test-thread-safety`) load models with default
offload and an f16 KV cache, which can land the cache in a Tesseract buffer and abort in the scheduler (SET_ROWS on
`cache_k`). They run with `TSR_DEVICE=none` (`sw/ggml-tesseract/CMakeLists.txt`).

## KV padding and graph reuse

`n_kv` of a ubatch is padded to a bucket of 256 cells (`n_kv = min(kv_size, max(256, GGML_PAD(used_max_p1, 256)))`,
`src/llama-kv-cache.cpp:1285-1298`), "so that the graph remains constant across batches and can be reused".
`llama_context::process_ubatch` reuses the previous graph if `res->can_reuse(gparams)`
(`src/llama-context.cpp:1530-1574`). On reuse there is no `ggml_backend_sched_reset` or `alloc_graph`: the same split
graphs with the same tensor addresses are computed again and only the inputs are re-set. The backend's op-list cache
keys on this (SW §8 R7): per token the host writes the graph inputs (`inp_pos`, `kq_mask`, `k_idxs`, `v_idxs`,
`inp_out_ids`), launches once and reads the logits, and a new KV bucket builds one new op list.
`LLAMA_GRAPH_REUSE_DISABLE=1` turns reuse off.

## Backend sampling

The pin has an experimental backend-sampling API: samplers can append ggml ops to the decode graph
(`llama_sampler_i::backend_init`, `backend_apply`; `include/llama.h:1394-1425`; `-bs/--backend-sampling`). Sampling on
the device inside `llama_decode` would need the backend to support `ARGMAX` (TOPK with k = 1) and `TOP_K` (the vector
unit's TOPK). The backend does not use backend sampling (`-bs` off, SW §8 R9): sampling stays on the host.

## Other notes for running llama.cpp on Tesseract

- `n_gpu_layers` defaults to -1 (auto, fit; `common/common.h:480`). Pass `-ngl 99` explicitly: any value above
  `n_layer` puts every layer on the device, and the output layer is on it for any value of 1 or more.
- `get_memory` reports the WEIGHT heap of all channels.
- Compute buffers (src1 copies and MUL_MAT outputs). `get_max_size` is 512 MiB, so ggml-alloc cuts the Tesseract
  compute buffer into chunks of at most that size, and a single larger tensor gets a chunk of its own
  (`ggml/src/ggml-alloc.c:160-178`). A chunk that gets a window in device memory is addressed in place. A chunk
  that gets none is kept in host memory, and the MUL_MATs that use it are staged through the IO stage in T tiles
  (`docs/spec.md` §12). There is no `-ub` limit from IO. llama.cpp reserves the compute buffer for the worst case,
  `n_outputs = min(n_ubatch, n_outputs_max)` logits (`src/llama-context.cpp:786-797`), although decode and
  prefill only compute the output rows, so a large-vocabulary model reserves a large logits chunk (Qwen 152,064 x
  512 x 4 B, Gemma 4 262,144 x 512 x 4 B). The backend places it in a per-channel compute heap between the top of
  the weights and the bottom of the KV planes ([EX](design/01-execution-model.md) §11, SW §8 R6, SW-26), so no
  launch is staged in decode. A chunk that still does not fit is staged and costs one extra launch per token,
  reported by `TSR_STATS` as "staged chunks". `klcheck` sets `n_outputs_max` to the rows it needs.
- LoRA adapters (`--lora`). `llama_adapter_lora_init` allocates the adapter tensors on the buffer type of the
  weight they modify (`src/llama-adapter.cpp:343`), that is on Tesseract for offloaded layers, as ANY buffers,
  before the context exists. A buffer of leaf tensors only is kept in host memory (it takes no IO), and each
  `lora_a` and `lora_b` is converted to Q8_0 into the WEIGHT heap the first time a MUL_MAT uses it
  (`build_lora_mm`, `src/llama-graph.cpp:1629-1660`).
