# PyTorch package (`sw/torch`)

`torch.compile(model, backend="tesseract")` runs a PyTorch model on a Tesseract device in one of two modes. In full
mode, the default, the aten graph is cut into regions of ops that lower to op list kinds of the firmware, and each
region runs as one op list; the ops that match no pattern run on the CPU. In matmul mode only the weight matmuls
run on the device, each as a one-op op list. The device is `TSR_DEVICE` (`sim` by default, `mock`, `vkernel`,
`xrt`). The runtime under this package, [`sw/rt`](../rt/README.md), runs the op lists through the firmware's `graph`
kernel. This is the package reference. The architecture, the partitioner, the numerics, the supported ops and the
tests are in [`docs/torch-backend.md`](../../docs/torch-backend.md).

| Path | Content |
| --- | --- |
| `csrc/tsr_torch.cpp` | extension module `_tesseract_rt`: binds `tsr::rt::device` (`sw/rt/tsr_rt.h`) for torch tensors, device singleton |
| `CMakeLists.txt` | target `tesseract_torch_ext` (added by the top-level CMakeLists with `-DTSR_WITH_TORCH=ON`), ctests `torch_pytest*` |
| `tesseract_torch/` | `__init__.py` (public API, registers the backend, `compile()`), `backend.py` (Dynamo backend, AOTAutograd, modes and options, regions of the aten graph and their rewrite to `device_region`, partition of the matmuls, `device_mm`), `lower.py` (full mode: patterns, layout rules and region formation, no device), `runtime.py` (device singleton, weight and constant caches, regions, the device shadows of KV caches, eager `linear` and `linear_ref`, `reference_linears`, counters, debug and verify), `quant.py` (`QuantLinear` and the custom op `tesseract::linear_blocks`), `_ext.py` (finds and imports `_tesseract_rt`) |
| `tests/` | pytest suite (`spec_model.py`: independent numpy models; `test_lower.py`, `test_graph.py` and `test_kvcache.py` for full mode) |
| `examples/hf_compare.py`, `tools/hf_check.py` | a Hugging Face causal LM (local safetensors) compiled against eager CPU, and the end-to-end numeric check (`--full-mode` for a region forward, `--generate` for `generate` with a static cache) |
| `tools/fuzz_full.py`, `tools/fuzz_gen.py` | the differential fuzzer of full mode: random models from a seed, run in eager, matmul mode and full mode, and the judge ([`docs/torch-backend.md`](../../docs/torch-backend.md) §6) |

## Build and use

```
cmake -S . -B build/dev -DCMAKE_BUILD_TYPE=Release -DTSR_WITH_TORCH=ON && cmake --build build/dev -j8 --target tesseract_torch_ext
export PYTHONPATH=$PWD/sw/torch:$PWD/build/dev/python
build/tools/torchvenv/bin/python -c "import tesseract_torch as tt; print(tt.device().description)"
```

Run `third_party/apply-patches.sh` first on a fresh checkout. For the card, configure with `-DTSR_WITH_XRT=ON` as
well ([`docs/torch-backend.md`](../../docs/torch-backend.md) §2.4). Torch comes from `TSR_TORCH_PYTHON` (default
`build/tools/torchvenv/bin/python`, else `python3`), and the extension is built for that interpreter's Python. If
`<build>/python` is not on `sys.path`, the package also looks in `$TSR_TORCH_EXT_DIR`, `<repo>/build/torch/python` and
`<repo>/build/dev/python`.

```python
import torch, tesseract_torch as tt

model = tt.compile(model)              # = torch.compile(model, backend="tesseract"); import registers it
with torch.no_grad():                  # or torch.inference_mode(); inference only
    y = model(x)
```

`tt.compile(model, mode=None, kv_size=None, strict=False, debug=None, verify=None, offload=True, attention=None,
kv_cache=None, **torch_compile_kwargs)` passes `options={"mode", "kv_size", "strict", "debug", "verify", "offload",
"attention", "kv_cache"}` to the backend.
`mode` is `"full"` or `"matmul"`. An explicit `mode` wins over `TSR_TORCH_MODE`, the default is full, and an unknown
mode is a `ValueError`. `kv_size` (default 4096) is the cells of the KV planes of each attention op without a cache in a region, the
longest sequence of a call. `attention` is `"device"` (default) or `"cpu"`: full mode computes attention on the device
or in float32 on the CPU, where a region ends at each attention. An explicit `attention` wins over
`TSR_TORCH_ATTENTION` (read in full mode only), an unknown value of the option is a `ValueError`, and so is an unknown value of the variable in full mode; matmul mode ignores the option and the variable. `kv_cache` (default on) is the switch of the generation paragraph below. `strict=True` raises where the backend would run a graph eagerly or, in full mode, fall back to matmul mode because forming the regions of a graph failed, and
`offload=False` prints the partition but keeps every matmul on the CPU and forms no region.

In full mode a region is a convex set of at least two ops that lower to kinds (matmuls with a weight, RMSNorm, RoPE,
attention, activations, residual adds, scalar scales; the aten forms are in
[`docs/torch-backend.md`](../../docs/torch-backend.md) §4), in float32. It runs as one op list and its intermediate
values stay on the device. The stateless forward of a Hugging Face Qwen2.5 decoder is one region. Ops that match no pattern run on
the CPU. A call that does not fit the device (T above `kv_size` or above 680 with an attention op that has no KV cache, a full heap, a mask
row that every key of which is masked by a finite value at most -1e9) and a program the device refuses run the
region's aten subgraph on the CPU, with one warning per region. Inside a region the norms, activations, RoPE and
attention have the device's numerics, so a region is not bit-identical to eager PyTorch; its oracle is the reference
executor, which `TSR_TORCH_VERIFY=1` compares with every region output. Device attention quantizes Q, K and V as
q8_0, the numerics of the llama.cpp path, which adds to the difference from eager float32 that the Q8_0 weights make;
models with large attention scores (small models, the first positions of a prompt) are the most sensitive.
`attention="cpu"` or matmul mode keeps attention in float32. The outputs of a region that share storage in eager
(a value and its views) share storage after the rewrite, and a copy between outputs keeps them separate.

Generation with a Hugging Face static KV cache runs on the device in full mode. The device keeps the K and V rows of
earlier calls, quantized to q8_0, in KV planes that outlive graphs and calls, so a decode step sends one row. The host
`StaticCache` tensors (float32) stay the complete cache and the device holds a shadow of them, so every cache feature
and every CPU fallback stays exact. `generate` calls `self.forward`, so compile the forward:

```python
model.forward = tt.compile(model.forward)        # model is the Hugging Face model itself, not a compiled one
out = model.generate(**inputs, max_new_tokens=64, do_sample=False, cache_implementation="static")
```

`torch.compile(model.forward, backend="tesseract")` is the same, and `inputs` is the tokenizer's output.
`tt.compile(model)` alone compiles `model.__call__`, which `generate` does not go through. With the default dynamic
cache the attention stays on the CPU and the results are still right. For a Hugging Face Qwen2.5 decoder the prefill is
one region call and every decode step is one region call and one launch. `kv_cache=False` (or `TSR_TORCH_KV_CACHE=0`)
keeps the cache update and the attention over the cache on the CPU. `generate(..., max_cache_len=N)` keeps one cache
geometry, so the planes and the device graph are reused across generations. The cache must be float32 CPU tensors with a
version counter (call `generate` under `torch.no_grad()`, not inside `torch.inference_mode()`), batch 1, and a write that bumps no version counter
(`.data`) is not seen. `tt.kv_caches()` lists the caches. The state of a cache, its memory and the verify checks are in
[`docs/torch-backend.md`](../../docs/torch-backend.md) §3.10.

An aten matmul node goes to the device (matmul mode, and full mode outside the regions) when it is `mm`, `addmm`,
`linear`, `matmul` or `mv`, has one operand that is a weight (a parameter, a registered buffer or a graph constant,
possibly through view ops) and one that is not, and after orientation is a 2D [N, K] float32, float16 or bfloat16 CPU
weight with K % 32 == 0 that the device accepts. Everything else runs on the CPU in matmul mode. A device that cannot
be opened fails `torch.compile` at the first compilation. A matmul node falls back to the CPU only when its weight
cannot be placed, with one warning.

## Environment

| Variable | Effect |
| --- | --- |
| `TSR_DEVICE` | transport kind, with its own variables ([`sw/transport/README.md`](../transport/README.md)) |
| `TSR_TORCH_MODE` | `full` (default) or `matmul`; an explicit `mode` of `tt.compile` or `options=` wins |
| `TSR_TORCH_ATTENTION` | `device` (default) or `cpu`, full mode only (ignored in matmul mode); an explicit `attention` wins |
| `TSR_TORCH_KV_CACHE` | `0` or `1` (default `1`): full mode puts the attention over a static KV cache on the device; `0` keeps it on the CPU; an explicit `kv_cache` wins; an unknown value is a `ValueError` in full mode |
| `TSR_TORCH_DEBUG` | `1`: the partition of every compiled graph (each region with its inputs, outputs, weights and constants, the CPU nodes beside it with the reason, which matmuls go to the device and why the others stay on the CPU), the records of the first plan of each region, every upload, the counters at exit. `2`: also every device call. `3`: also the rewritten aten graphs. |
| `TSR_TORCH_VERIFY` | `1`: every device matmul is also computed by the reference model (libtsr_ref on the weight rows read back from the device) and every region output by the reference executor (the region's op lists run on the host on the device's constants and weights), and compared bit for bit. A mismatch raises `TesseractError` naming the weight or region, the element and both values. |
| `TSR_TORCH_EXT_DIR` | where to find `_tesseract_rt` if it is not on `sys.path`; `hf_check.py`, `hf_compare.py` and `fuzz_full.py` put this directory on `sys.path` (default `build/torch/python`) |
| `TSR_TORCH_HF_MODEL` | model directory of the opt-in `tests/test_hf.py` |
| `TSR_RT_DEBUG` | the runtime's own log (`1`: uploads, calls and launches) |

## Python API

| Name | Meaning |
| --- | --- |
| `compile(model, *, mode, kv_size, strict, debug, verify, offload, attention, kv_cache, **kw)`, `tesseract_backend(gm, example_inputs, options=None, mode=None)`, `Options(...)`, `MODES`, `resolve_mode(mode)`, `ATTENTIONS`, `resolve_attention(attention)` | `torch.compile(model, backend="tesseract", options=..., **kw)`, the backend function (registered as `"tesseract"`), its options, the two mode names, the mode that `mode` and `TSR_TORCH_MODE` give, the two attention names and the one that `attention` and `TSR_TORCH_ATTENTION` give (`resolve_kv_cache(kv_cache)` gives the setting of `kv_cache` and `TSR_TORCH_KV_CACHE`) |
| `linear(x, weight, bias=None, *, ref=False, name="")` | eager `F.linear` with the weight matmul on the device (cached weight; output in x's dtype); inference only |
| `QuantLinear(blocks, in_features, fmt, bias=None)`, `QuantLinear.from_gguf(tensor, bias=None)`, `linear_blocks(x, blocks, fmt, K)` | `nn.Linear` over ggml Q8_0 or Q4_K blocks (`fmt` is `"q8_0"` or `"q4_k"`) and the custom op `torch.ops.tesseract.linear_blocks` |
| `linear_ref(x, weight, bias=None)`, `reference_linears()` | the same on the host reference model, bit-identical to `linear` on a correct device, and a context manager that runs eager `nn.Linear` with supported weights through it |
| `supported_weight(w)` | `None` or the reason a weight cannot go to the device |
| `device()`, `open_device(kind=None, io_bytes=None, read_perf=None)`, `close_device()` | the process-wide `_tesseract_rt.Device`, opening it with options (if it is open already it is returned when every option given matches, else `TesseractError`), and closing it after the calls in flight |
| `weight_cache()`, `release_weights()` | the `WeightCache` (`.acquire`, `.acquire_blocks`, `.handle`, `.find`, `.sweep`, `.clear`, `.entries`) and freeing every device weight (re-uploaded on next use), the constants and every KV cache |
| `kv_caches()` | a list of dicts, one per KV cache the device holds a shadow of (`name`, `E`, `cells`, `n_dev`: cells that equal the host rows, `alive`, `handle`, `k_off`) and one per pooled set of planes of a dead cache (`name` is `(pool)`) |
| `counters()`, `reset_counters()`, `format_counters()`, `specs()`, `regions()` | backend and runtime counters, every matmul node partitioned to the device so far (`MMSpec`: op, side, N, K, weight, calls, tokens, cpu_calls; `.disabled()`: why a weight runs on the CPU at that node) and every region formed so far (`RegionRT`: name, calls, tokens, cpu_calls, graph) |
| `set_debug(level)`, `set_verify(on)`, `TesseractError`, `_C` | override `TSR_TORCH_DEBUG` and `TSR_TORCH_VERIFY`, the error type (a `RuntimeError`), the extension module |

`_tesseract_rt.Device(kind=None, io_bytes=0, read_perf=True)` has `.upload(w [N, K] f32/f16/bf16, max_part_rows=0,
max_part_k=0) → handle`, `.upload_blocks("q8_0" | "q4_k", uint8 blocks, N, K)`, `.free(h)`, `.info(h)`,
`.read_blocks(h, r0=0, r1=-1)`, `.linear(h, x [T, K] f32, out=None) → y [T, N] f32`, `.linear_ref(...)`, `.counters()`,
`.reset_counters()`, `.memory()`, `.config` (a dict: `num_ch`, `t_max`, `k_max`, `caps`, `local_scale`, ...),
`.transport`, `.description`, `.serial` (1, 2, ... per device opened in the process), `.io_bytes`, `.read_perf`,
`.closed`, `.unsupported_reason(dtype, N, K)`, `._try_linear_scale(h, x, scale)` and `.close()`. The graph API (full
mode, [`sw/rt`](../rt/README.md)) is `.const_upload(tensor) → handle`, `.const_free(h)`, `.graph_unsupported(program)`
(`None` or the reason), `.graph_create(program) → handle`, `.graph_free(h)`, `.graph_shape(h)`,
`.graph_run(h, inputs, kv=None, pos0=0) → [y [T, n] f32, ...]`, `.graph_run_ref(h, inputs, kv=None, pos0=0)` (the same
on the host) and `.graph_info(h, T)` (the plan of T; for a graph that runs in chunks the plan of the first chunk, whose
`T` is the rows of that chunk). A program is a dict of `values`, `ops` and `kv_size` (`lower.Region.to_program`
and `tsr_rt.h`), and for a cached attention `kv_slots` (the cells of each KV slot), `kv` (the slot) in an `attn` op and
`cols` (the cells) in a mask value. The KV objects, the planes of a cache that outlive graphs and calls, are
`.kv_create(E, cells) → handle` (E a multiple of 64), `.kv_free(h)`, `.kv_info(h)` (`E`, `cells`, `bytes`, `k_off`,
`v_off`), `.kv_store(h, cell0, k [n, E] f32, v [n, E] f32)` (quantized as an attention op stores its rows) and
`.kv_read(h, cell0, n) → (K, V)` (uint8 `[n, E / 32 * 34]`, ggml `block_q8_0` rows). `kv` of `graph_run` is one KV handle
per slot. A wrong argument of these functions (a bool or a float where an int is meant, a handle below 0 or above 64
bits, cells beyond the planes, also for no rows) is a `ValueError`. The module functions are `open`, `get`, `is_open`, `close` and `row_bytes(type, K)`.

Counters: `graphs`, `graphs_eager` (autograd graphs run eagerly), `nodes_device`, `nodes_cpu`, `device_calls`,
`local_scale_calls`, `device_tokens`, `cpu_fallback_calls`, `uploads`, `reuploads`, `upload_bytes`, `cache_hits`,
`weights_freed`, `verified_calls`, `cached_weights`, and in full mode `regions`, `region_calls`, `region_tokens`,
`region_cpu_fallback_calls` (calls that ran the saved aten subgraph on the CPU), `consts_uploaded`, `const_reuploads`,
`verified_regions`, `cached_consts`, for cached attention `kv_caches` (cache states made), `kv_pool_hits` (states whose
planes came from the pool of dead caches), `kv_resync_rows` (rows stored again from the host cache) and `kv_region_calls`
(device calls that used a KV slot), and the runtime's `rt_graphs`, `rt_graph_runs`, `rt_graph_launches`,
`rt_graph_records`, `rt_graph_plan_hits`, `rt_graph_plan_misses`, `rt_consts`, `rt_kv_creates`, `rt_kv_stores`,
`rt_kv_store_rows`, `rt_linears`, `rt_tokens`, `rt_execs`,
`rt_prog_hits`, `rt_prog_misses`, `rt_perf_cycles`, `rt_perf_mvm_cycles`, `rt_dev_read_bytes`, `rt_h2d_bytes`,
`rt_h2d_weight_bytes` and `rt_d2h_bytes` ([`sw/rt/README.md`](../rt/README.md)).

## Tests

```
PYTHONPATH=sw/torch:build/dev/python build/tools/torchvenv/bin/python -m pytest -q -s sw/torch/tests
ctest --test-dir build/dev -R torch_pytest         # TSR_DEVICE=sim, T_MAX 1, 4 and 8
TSR_TORCH_HF_MODEL=models/hf/Qwen2.5-0.5B-Instruct python -m pytest -q -s sw/torch/tests/test_hf.py   # opt-in
```

The suites run in matmul mode (`tests/conftest.py`); `test_graph.py`, `test_kvcache.py` and the full-mode cases of
`test_hf.py` ask for full mode. `test_lower.py` needs no device. `test_kvcache.py` runs a tiny Qwen2 with a Hugging Face
`StaticCache` through every state of the device shadow and checks each step against the stateless run of the whole
sequence and against a fresh run on a copy of the host cache.

`hf_compare.py` and `hf_check.py` take a local Hugging Face model directory (safetensors, `trust_remote_code=False`)
and run with `-I`, because the model directory is data (`docs/torch-backend.md` §6 describes the checks):

```
build/tools/torchvenv/bin/python -I sw/torch/examples/hf_compare.py models/hf/Qwen2.5-0.5B-Instruct \
    [--dtype float32|bfloat16] [--tokens 16] [--gen 8] [--gguf models/qwen2.5-0.5b-instruct-q8_0.gguf]
build/tools/torchvenv/bin/python -I sw/torch/tools/hf_check.py models/hf/Qwen2.5-0.5B-Instruct \
    [--download Qwen/Qwen2.5-0.5B-Instruct] [--tokens 32] [--gen 16] [--gguf GGUF] [--json OUT]
build/tools/torchvenv/bin/python -I sw/torch/tools/hf_check.py MODEL_DIR --forward-only --tokens 8 \
    --save-logits sim.npy        # then on another device: --expect sim.npy (bit for bit)
build/tools/torchvenv/bin/python -I sw/torch/tools/hf_check.py MODEL_DIR --full-mode --tokens 16   # one region, one launch
build/tools/torchvenv/bin/python -I sw/torch/tools/hf_check.py MODEL_DIR --full-mode --tokens 16 --attention cpu
build/tools/torchvenv/bin/python -I sw/torch/tools/hf_check.py MODEL_DIR --generate --tokens 24 --gen 8   # generate with a static cache
build/tools/torchvenv/bin/python -I sw/torch/tools/fuzz_full.py --gen elem --start 0 --count 50   # fuzzer; --replay SEED
```

The limits are in [`docs/torch-backend.md`](../../docs/torch-backend.md) §7: inference only, batch 1 through attention
in full mode, the KV cache across calls is the static cache only (the host keeps the float32 cache), weight matmuls
only in matmul mode, float weights stored as
Q8_0, `bmm` and batched weights on the CPU, and writes through `.data` not detected.
