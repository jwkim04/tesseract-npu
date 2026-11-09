# PyTorch package (`sw/torch`)

`torch.compile(model, backend="tesseract")` runs the weight matmuls of a PyTorch model on a Tesseract device and
everything else on the CPU. The device is `TSR_DEVICE` (`sim` by default, `mock`, `vkernel`, `xrt`). The runtime under
this package, [`sw/rt`](../rt/README.md), runs each device matmul as a one-op op list through the firmware's `graph`
kernel. This is the package reference. The architecture, the partitioner, the numerics, the supported ops and the
tests are in [`docs/torch-backend.md`](../../docs/torch-backend.md).

| Path | Content |
| --- | --- |
| `csrc/tsr_torch.cpp` | extension module `_tesseract_rt`: binds `tsr::rt::device` (`sw/rt/tsr_rt.h`) for torch tensors, device singleton |
| `CMakeLists.txt` | target `tesseract_torch_ext` (added by the top-level CMakeLists with `-DTSR_WITH_TORCH=ON`), ctests `torch_pytest*` |
| `tesseract_torch/` | `__init__.py` (public API, registers the backend, `compile()`), `backend.py` (Dynamo backend, AOTAutograd, partition of the aten graph, `device_mm`), `runtime.py` (device singleton, weight cache, eager `linear` and `linear_ref`, `reference_linears`, counters, debug and verify), `quant.py` (`QuantLinear` and the custom op `tesseract::linear_blocks`), `_ext.py` (finds and imports `_tesseract_rt`) |
| `tests/` | pytest suite (`spec_model.py`: independent numpy models) |
| `examples/hf_compare.py`, `tools/hf_check.py` | a Hugging Face causal LM (local safetensors) compiled against eager CPU, and the end-to-end numeric check |

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

`tt.compile(model, strict=False, debug=None, verify=None, offload=True, **torch_compile_kwargs)` passes
`options={"strict", "debug", "verify", "offload"}` to the backend. `strict=True` raises where the backend would run a
graph eagerly, and `offload=False` prints the partition but keeps every matmul on the CPU. An aten matmul node goes
to the device when it is `mm`, `addmm`, `linear`, `matmul` or `mv`, has one operand that is a weight (a parameter, a
registered buffer or a graph constant, possibly through view ops) and one that is not, and after orientation is a 2D
[N, K] float32, float16 or bfloat16 CPU weight with K % 32 == 0 that the device accepts. Everything else runs on the
CPU. A device that cannot be opened fails `torch.compile` at the first compilation. Only a weight that cannot be
placed falls back to the CPU, with one warning.

## Environment

| Variable | Effect |
| --- | --- |
| `TSR_DEVICE` | transport kind, with its own variables ([`sw/transport/README.md`](../transport/README.md)) |
| `TSR_TORCH_DEBUG` | `1`: the partition of every compiled graph (which matmuls go to the device and why the others stay on the CPU), every upload, the counters at exit. `2`: also every device call. `3`: also the rewritten aten graphs. |
| `TSR_TORCH_VERIFY` | `1`: every device matmul is also computed by the reference model (libtsr_ref on the weight rows read back from the device) and compared bit for bit. A mismatch raises `TesseractError` naming the weight, the element and both values. |
| `TSR_TORCH_EXT_DIR` | where to find `_tesseract_rt` if it is not on `sys.path` |
| `TSR_TORCH_HF_MODEL` | model directory of the opt-in `tests/test_hf.py` |
| `TSR_RT_DEBUG` | the runtime's own log (`1`: uploads, calls and launches) |

## Python API

| Name | Meaning |
| --- | --- |
| `compile(model, *, strict, debug, verify, offload, **kw)`, `tesseract_backend(gm, example_inputs, options=None, mode=None)`, `Options(...)` | `torch.compile(model, backend="tesseract", options=..., **kw)`, the backend function (registered as `"tesseract"`) and its options |
| `linear(x, weight, bias=None, *, ref=False, name="")` | eager `F.linear` with the weight matmul on the device (cached weight; output in x's dtype); inference only |
| `QuantLinear(blocks, in_features, fmt, bias=None)`, `QuantLinear.from_gguf(tensor, bias=None)`, `linear_blocks(x, blocks, fmt, K)` | `nn.Linear` over ggml Q8_0 or Q4_K blocks (`fmt` is `"q8_0"` or `"q4_k"`) and the custom op `torch.ops.tesseract.linear_blocks` |
| `linear_ref(x, weight, bias=None)`, `reference_linears()` | the same on the host reference model, bit-identical to `linear` on a correct device, and a context manager that runs eager `nn.Linear` with supported weights through it |
| `supported_weight(w)` | `None` or the reason a weight cannot go to the device |
| `device()`, `open_device(kind=None, io_bytes=None, read_perf=None)`, `close_device()` | the process-wide `_tesseract_rt.Device`, opening it with options (if it is open already it is returned when every option given matches, else `TesseractError`), and closing it after the calls in flight |
| `weight_cache()`, `release_weights()` | the `WeightCache` (`.acquire`, `.acquire_blocks`, `.handle`, `.find`, `.sweep`, `.clear`, `.entries`) and freeing every device weight (re-uploaded on next use) |
| `counters()`, `reset_counters()`, `format_counters()`, `specs()` | backend and runtime counters, and every node partitioned to the device so far (`MMSpec`: op, side, N, K, weight, calls, tokens, cpu_calls; `.disabled()`: why a weight runs on the CPU at that node) |
| `set_debug(level)`, `set_verify(on)`, `TesseractError`, `_C` | override `TSR_TORCH_DEBUG` and `TSR_TORCH_VERIFY`, the error type (a `RuntimeError`), the extension module |

`_tesseract_rt.Device(kind=None, io_bytes=0, read_perf=True)` has `.upload(w [N, K] f32/f16/bf16, max_part_rows=0,
max_part_k=0) → handle`, `.upload_blocks("q8_0" | "q4_k", uint8 blocks, N, K)`, `.free(h)`, `.info(h)`,
`.read_blocks(h, r0=0, r1=-1)`, `.linear(h, x [T, K] f32, out=None) → y [T, N] f32`, `.linear_ref(...)`, `.counters()`,
`.reset_counters()`, `.memory()`, `.config` (a dict: `num_ch`, `t_max`, `k_max`, `caps`, `local_scale`, ...),
`.transport`, `.description`, `.serial` (1, 2, ... per device opened in the process), `.io_bytes`, `.read_perf`,
`.closed`, `.unsupported_reason(dtype, N, K)`, `._try_linear_scale(h, x, scale)` and `.close()`. The module
functions are `open`, `get`, `is_open`, `close` and `row_bytes(type, K)`.

Counters: `graphs`, `graphs_eager` (autograd graphs run eagerly), `nodes_device`, `nodes_cpu`, `device_calls`,
`local_scale_calls`, `device_tokens`, `cpu_fallback_calls`, `uploads`, `reuploads`, `upload_bytes`, `cache_hits`,
`weights_freed`, `verified_calls`, `cached_weights`, and the runtime's `rt_linears`, `rt_tokens`, `rt_execs`,
`rt_prog_hits`, `rt_prog_misses`, `rt_perf_cycles`, `rt_perf_mvm_cycles`, `rt_dev_read_bytes`, `rt_h2d_bytes`,
`rt_h2d_weight_bytes` and `rt_d2h_bytes` ([`sw/rt/README.md`](../rt/README.md)).

## Tests

```
PYTHONPATH=sw/torch:build/dev/python build/tools/torchvenv/bin/python -m pytest -q -s sw/torch/tests
ctest --test-dir build/dev -R torch_pytest         # TSR_DEVICE=sim, T_MAX 1, 4 and 8
TSR_TORCH_HF_MODEL=models/hf/Qwen2.5-0.5B-Instruct python -m pytest -q -s sw/torch/tests/test_hf.py   # opt-in
```

`hf_compare.py` and `hf_check.py` take a local Hugging Face model directory (safetensors, `trust_remote_code=False`)
and run with `-I`, because the model directory is data (`docs/torch-backend.md` §6 describes the checks):

```
build/tools/torchvenv/bin/python -I sw/torch/examples/hf_compare.py models/hf/Qwen2.5-0.5B-Instruct \
    [--dtype float32|bfloat16] [--tokens 16] [--gen 8] [--gguf models/qwen2.5-0.5b-instruct-q8_0.gguf]
build/tools/torchvenv/bin/python -I sw/torch/tools/hf_check.py models/hf/Qwen2.5-0.5B-Instruct \
    [--download Qwen/Qwen2.5-0.5B-Instruct] [--tokens 32] [--gen 16] [--gguf GGUF] [--json OUT]
build/tools/torchvenv/bin/python -I sw/torch/tools/hf_check.py MODEL_DIR --forward-only --tokens 8 \
    --save-logits sim.npy        # then on another device: --expect sim.npy (bit for bit)
```

The limits are in [`docs/torch-backend.md`](../../docs/torch-backend.md) §7: inference only, weight matmuls only,
float weights stored as Q8_0, `bmm` and batched weights on the CPU, and writes through `.data` not detected.
