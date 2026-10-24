# ggml-tesseract

ggml backend for the Tesseract NPU core (spec "ggml 백엔드 통합"). One registry
`"Tesseract"` with one device `"Tesseract0"` of type `GGML_BACKEND_DEVICE_TYPE_GPU`; v1 runs
`MUL_MAT` only, everything else stays on the CPU. Pin and verified llama.cpp API:
`docs/llama-pin.md`.

## Files

| File | Content |
| --- | --- |
| `ggml-tesseract.cpp` | reg / device / buffer type / buffer / backend, placement, lazy conversion, segments, program cache, execution |
| `tsr-lower.{h,cpp}` | MUL_MAT lowering v1: SPM tiling and `LD / QACT / MVM / ST` emission (no ggml dependency) |
| `tsr-weights.{h,cpp}` | ggml type → device format (Q8_0, Q4_K native; others `to_float` + `quantize_row_q8_0_ref`) and back |
| `tsr-alloc.{h,cpp}` | first-fit range allocator for IO / AUX / WEIGHT heap |
| `tsr-wlayout.{h,cpp}` | device-format row bytes, row repack + symmetric scatter / gather over the channels (no ggml) |
| `lower.cmake` | static library `tsr_lower` = tsr-lower + tsr-alloc + tsr-wlayout (no ggml), linked by this backend and by the runtime `sw/rt` (`tsr_rt`) |
| `sources.cmake` | source list (the ggml-dependent files), included by llama.cpp's `ggml/src/ggml-tesseract/CMakeLists.txt` (patch 0001); links `tsr_lower` with a deferred call, so the patch is unchanged |
| `CMakeLists.txt`, `tests/` | unit tests and the end-to-end test (added by the top-level CMakeLists after llama.cpp) |

Public C API (`third_party/llama.cpp/ggml/include/ggml-tesseract.h`, from the patch):
`ggml_backend_tesseract_reg()`, `ggml_backend_tesseract_init()`, `ggml_backend_is_tesseract()`,
`ggml_backend_tesseract_buffer_type()`. ggml registers the backend itself under
`GGML_USE_TESSERACT`.

## Design (where to look)

- **Transport**: `tsr_transport_open(NULL)` on first use (`tsr_dev_open_locked`); config from
  `tsr_transport_config`. All device state is behind one mutex (`tsr_device::mtx`).
- **Buffer type = virtual address space**: each buffer gets a 64-aligned base in a
  non-canonical range (`0x4000...`, never dereferenced); `tensor->data = base + offset`.
  Alignment 64, `get_alloc_size` = nbytes rounded up to 64 (the line-granular ST tail of the
  last output row lands in this padding), `get_max_size` = IO window capacity (IO region minus
  the stage: 240 MiB), so ggml-alloc cuts compute buffers into chunks that can each get a window.
- **Storage mode, fixed at a buffer's first data access** (`tsr_buf_decide`; llama.cpp calls
  `init_tensor` while usage is still ANY and sets WEIGHTS afterwards, so `init_tensor` only
  records metadata):
  - usage WEIGHTS → *PLACED*: per tensor (records made in `init_tensor`). 2D loadable
    weights (`tsr_rec_is_devweight_shape`) become *DEVWEIGHT*: converted to Q8_0 unless the
    type is Q8_0/Q4_K, repacked with `tsr_repack_row`, symmetric allocation (same W_OFF on all
    channels, `tsr_tensor_ch_beats` per channel) in the WEIGHT heap. Everything else (1D norms,
    3D, odd K) is *RAW* in AUX (ch0), or the heap if AUX is full.
  - usage ANY / COMPUTE → bytes stored verbatim (activations, test-backend-ops tensors), as
    *LINEAR* (the whole buffer is one window in the ch0 IO region) if a window is free, else
    *HOSTED* (host memory, `calloc`). An ANY buffer whose tensors are all leaves (op NONE, not
    a graph input: LoRA adapters, control vectors, a KV cache without `-nkvo`) is HOSTED from the
    start: no program addresses it, so it takes no IO.
  - IO layout: windows below, the **stage** at the top (max(16 MiB, 2 × SPM), reserved at open).
- **MUL_MAT src0 resolution** (`tsr_resolve_weight`): a DEVWEIGHT is used in place; any other
  src0 (raw bytes in IO, host memory or AUX, any loadable type) is converted lazily into the
  heap at graph_compute and cached by (buffer id, offset, type, N, K) and the buffer's write
  generation (bumped by every host write and by device writes into it). test-backend-ops
  therefore reads back exact src0 bytes, and in a pre-allocated graph an F32 src0 produced by
  an earlier device op works (the pending program is executed first). In llama.cpp graphs
  supports_op keeps such src0 on the CPU (see supports_op).
- **Split uploads**: a DEVWEIGHT is staged on the host until every byte has arrived, then
  converted and scattered row by row; updates of a complete weight re-store whole rows
  (partial rows: read-modify-write for Q8_0/Q4_K, abort for converted types).
  `get_tensor` of a DEVWEIGHT un-repacks (exact for Q8_0/Q4_K; converted types are re-encoded
  from the device copy with a warning — debug only).
- **graph_compute** (`tsr_backend_graph_compute`): skips nodes without
  `GGML_TENSOR_FLAG_COMPUTE` and NONE/VIEW/RESHAPE/PERMUTE/TRANSPOSE; lowers each MUL_MAT
  (`tsr_lower_mul_mat`) into a *segment*; a segment becomes one command buffer + END, written
  to the CMD half not used last (double buffering) and run with `tsr_transport_exec`; device
  errors abort with ERR_CODE / ERR_PC / ERR_INFO and the disassembled instruction.
  T is tiled by `min(T_MAX, SPM capacity)`; src1 `[K, T, B2, B3]` contiguous is flattened.
- **Staged MUL_MATs** (`tsr_lower_staged`): src1 or dst in a HOSTED buffer, or dst rows that
  are not line-aligned (N*4 % 64 != 0 with T > 1). Everything pending runs first, then the
  MUL_MAT runs on its own in T tiles that fit the stage: src1 rows host → stage (a LINEAR src1
  is read in place), LD/QACT/MVM/ST, dst rows stage → host or → the LINEAR dst (compacted to
  N*4 bytes). Only one tile has to be resident in IO, whatever T is, so supports_op has no
  token-count limit. Typical case: llama.cpp reserves the output head for n_ubatch logits
  (Qwen2.5 at -ub 512: 296.75 MiB), a chunk that is HOSTED; decode computes one row of it.
  (Single-token outputs with N*4 % 64 != 0 ST straight into a LINEAR dst; the tail lands in the
  64-byte padding.) LLM projections have N % 16 == 0 and are staged only through HOSTED buffers.
- **Program cache** (`tsr_prog_cache`, tsr-lower): key = the resolved MUL_MAT descriptors of a
  segment (fmt, N, K, T, W_OFF, x/y beats); a program still resident in its CMD half runs
  without being rewritten. Bounded to 64 MiB and 4096 entries with LRU eviction; programs above
  4 MiB (long prefills: with T_MAX = 1 a program has 4 instructions per token per MUL_MAT) are
  run without being cached.
- **supports_op**: MUL_MAT with src1/dst F32, src0 `ne2 = ne3 = 1`, all contiguous, src0 type
  loadable, `tsr_mm_shape_ok` (K ≤ K_MAX, K % 32 == 0 (Q4_K: % 256), one token fits SPM),
  src0 (or the tensor it views) in a Tesseract buffer that is not COMPUTE (a WEIGHTS tensor only
  unsliced; usage ANY includes llama.cpp's size-0 placement-probe buffer), or unallocated and a
  leaf that is not a graph input (test-backend-ops asks before allocating; in llama.cpp an
  unallocated src0 is an input/activation such as `attn_inp_k_rot` of `-ctk q8_0` and stays on
  the CPU), src1 not in a WEIGHTS buffer and 64-byte aligned. **No condition on the token
  count**: llama.cpp places weights with a fixed 512-token probe (`weight_buft_supported`), so
  a size limit there kept every large-vocabulary output head on the CPU. No-op ops (NONE, VIEW,
  RESHAPE, PERMUTE, TRANSPOSE) are supported. `supports_buft`: own buft only; `offload_op`:
  false. `get_memory`: WEIGHT heap free/total × NUM_CH.
- **Weight read-back by the scheduler**: if an op that uses a device weight runs elsewhere, the
  scheduler copies the weight with `get_tensor` on every graph; for converted types that is a
  lossy re-encode (warned once) and types without `from_float_ref` (IQ2_XXS, IQ2_XS, IQ1_S,
  IQ1_M) abort with a message naming the cause. With the token-count rule gone, a resident
  weight's MUL_MAT is not rejected at run time.

## Environment

| Variable | Effect |
| --- | --- |
| `TSR_DEVICE` | transport kind (`sim` default, `xrt`); `none` registers no device |
| `TSR_DEBUG=1` | placement, lowering and exec log on stderr; `2` adds buffer events and a disassembly of every executed program |
| `TSR_FORCE_HOST=1` | test knob: every non-WEIGHTS buffer in host memory, so every MUL_MAT takes the staged path |
| `TSR_STAGE_KIB=<n>` | test knob: IO stage size in KiB (default max(16 MiB, 2 × SPM)); a small stage forces several T tiles |
| `TSR_TRACE`, `TSR_SIM_*` | handled by the sim transport (`sw/transport/tsr_transport.h`) |

## Build and test

```
third_party/apply-patches.sh
cmake -S . -B build/dev -DCMAKE_BUILD_TYPE=Release
cmake --build build/dev -j4 --target test-backend-ops test-ggml-tesseract test-tsr-backend-units llama-completion
build/dev/bin/test-tsr-backend-units                        # allocator + lowering
build/dev/bin/test-ggml-tesseract                            # bit-exact vs tsr_ref + NMSE vs CPU
TSR_SIM_TMAX=8 build/dev/bin/test-ggml-tesseract             # T tiling with T > 1
build/dev/bin/test-backend-ops test -o MUL_MAT -b Tesseract0 # vs CPU, NMSE 5e-4
build/dev/bin/llama-completion -m models/qwen2.5-0.5b-instruct-q8_0.gguf -ngl 99 -nkvo -p "..." -n 16
```

Configure only through the top-level CMakeLists. It defines `tsr_ref`, `tsr_transport` and
`tsr_lower` (`lower.cmake`) before llama.cpp. `sources.cmake` stops with a message if `tsr_lower`
is missing, and `test-tsr-backend-units` links it. The same tree adds the runtime `sw/rt` after
llama.cpp, so its ctest also lists `rt_sim`, `rt_sim_tmax8`, `rt_mock` and `rt_mock_tmax8`
(`sw/rt/README.md`; the PyTorch backend on top of it: `docs/torch-backend.md`).

ctest names: `ggml_tesseract_units`, `ggml_tesseract_e2e`, `ggml_tesseract_e2e_tmax8`,
`ggml_tesseract_e2e_hosted` and `ggml_tesseract_e2e_hosted_tmax8` (`TSR_FORCE_HOST=1
TSR_STAGE_KIB=64`, with and without `TSR_SIM_TMAX=8`), `ggml_tesseract_scalar_nmse`,
`ggml_tesseract_backend_ops`. Test hook for the storage decision:
`ggml_backend_reg_get_proc_address(reg, "ggml_backend_tesseract_buffer_storage")` returns
`const char * (*)(ggml_backend_buffer_t)` ("undecided", "io", "placed", "host").

`ggml_tesseract_backend_ops` leaves out the single-output case
`MUL_MAT(type_a=f32,type_b=f32,m=1,n=1,k=2048)`: with fresh random data each run its NMSE is one
squared relative error, which the spec's 8-bit path pushes over 5e-4 in ~15% of runs on a correct
device. `ggml_tesseract_scalar_nmse` (`tests/test-tsr-scalar-nmse.cpp`) covers that shape
bit-exactly against the spec model instead; numbers in `docs/llama-pin.md` (test-backend-ops).
The unfiltered `test-backend-ops test -o MUL_MAT -b Tesseract0` above therefore reports
1402/1403 in about one run in seven.

llama.cpp must run with `-nkvo` (KV cache on the CPU) and an explicit `-ngl`; see
`docs/llama-pin.md` for how compute buffers, the output head and LoRA adapters are placed.
