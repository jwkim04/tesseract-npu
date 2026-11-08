# ggml-tesseract

ggml backend for the Tesseract NPU core (spec "ggml backend integration"). One registry
`"Tesseract"` with one device `"Tesseract0"` of type `GGML_BACKEND_DEVICE_TYPE_GPU`. M7 (W3b) runs a
whole decode token of Qwen2.5 / Gemma 4 as one launch with the KV cache on the device; M6 (`TSR_KINDS=m6`)
runs the matrix products (`MUL_MAT`, and `MUL_MAT_ID` of a MoE model: `TSR_KINDS=m6,+MUL_MAT_ID`). Pin and verified
llama.cpp API: `docs/llama-pin.md`.

> **M6 (`docs/design/06-software-verification.md` §9.3, P3.2).** Each split's MUL_MATs become one
> op list (the op list format, SW §7.2) of MUL_MAT records (+ JOINs), written into an OPL cache in
> the ch0 OPL region and run by one launch of the firmware's `graph` kernel, which tiles each
> record with the shared planner `plan.hpp` and pushes the tile-op sequence of SW §7.4.
> M6 keeps attention on the CPU (`-nkvo`, `offload_kqv = false`); M7 (P3.5) adds the other kinds.

> **M7 (W3b, SW §8, §9.3 M7, P3.5).** `supports_op` takes the whole SW §9.3 table (RMS_NORM, MUL,
> ADD, SCALE, GLU, ROPE, GET_ROWS, SET_ROWS into KV buffers, FLASH_ATTN_EXT on KV-buffer views, CPY /
> CONT, UNARY TANH / GELU), so a llama.cpp decode graph of Qwen2.5 or Gemma 4 is one CPU split (the
> token embedding) and **one Tesseract split, run as one launch** (`-fa on -ctk q8_0 -ctv q8_0`,
> no `-nkvo`, `LLAMA_ATTN_ROT_DISABLE=1`). The builder (`tsr-build.{h,cpp}`) lowers each node to a
> record of its kind, fuses MUL_MAT + ADD (bias, residual) and RMS_NORM + MUL (weight), orders the
> records by DRAM byte ranges, puts positions / ids / KV indices / mask summaries into the list's
> preload table and caches each graph's plan by identity, so a decode token reuses its list and only
> its inputs change. Compute buffers live in device memory (IO, the low 512 MiB of ch1-3, the compute
> heap), so nothing is staged.

> **MoE (SW §7.9, SW-60 to SW-66, SW-76).** Gemma 4 26B-A4B (softmax gating, normalised top-8 of 128, a per-expert scale)
> runs on the simulator. 3D expert tensors (names with `_exps`) are placed per channel with the expert stride
> (`tsr_store_expert_rows`), `MUL_MAT_ID` is lowered by `tsr_lower_mul_mat_id` (the weights resolved by
> `tsr_resolve_weight` to W_OFF and EXP_STRIDE, the probe `tsr_moe_weight_ok`), and `tsr_build_fusion` turns
> llama.cpp's router nodes into one `MOE_ROUTE` record and the scale nodes into the scale table of the down
> `MUL_MAT_ID` plus a `CPY`. A router of more than 64 tokens (several tiles) writes its ids and weights to a device
> scratch window (`tsr_scratch_window`) and two `CPY` records move them into the graph's tensors (SW-76), because
> ggml-alloc may have put those over logits a later tile still reads. Without `MOE_ROUTE` in `TSR_KINDS` llama.cpp's own CPU ops select the experts and the
> device runs the matrix products (the M6 form). Tests: `ggml_tesseract_moe*` (`tests/test-tsr-moe.cpp`: the
> graph of `build_moe_ffn` in three kind configurations against the CPU, and the router group with its outputs placed by
> hand over the logits of later tiles, 65 to 11,000 tokens), `test-backend-ops -o MUL_MAT_ID -b
> Tesseract0` (ctest `ggml_tesseract_backend_ops_mul_mat_id`: 928 of 945 cases supported and passing; 17 refused: m not a
> multiple of 16, k not a multiple of the block, iq2_xxs).

## Files

| File | Content |
| --- | --- |
| `ggml-tesseract.cpp` | reg / device / buffer type / buffer / backend, placement, lazy conversion, segments (op lists), staging, execution |
| `tsr-opl.{h,cpp}`, `tsr-opl-dev.cpp` | the op-list path (no ggml): OPL builder (records of every kind, their DRAM ranges (`kind_ranges`), hazards over DRAM byte ranges, transitive reduction, JOINs, the preload table; encoding by `sw/ctrl/include/tsr/opl.hpp`'s writer), the OPL cache, the `graph` kernel on a transport (boot, launch, wait, error text) |
| `tsr-build.{h,cpp}` | W3b, the M7 builder (ggml-aware, device-free): the record of each ggml node in probe mode (supports_op) or build mode, fusion (pass 3), the encoding of one record with its preload entries and FA_NEW (passes 4-6), the graph key (R7); the device enters through callbacks (`tsr_build_env`) |
| `tsr-lower.{h,cpp}` | MUL_MAT shape checks and SPM tiling (placement and staging; the firmware tiles with `plan.hpp`) |
| `tsr-weights.{h,cpp}` | ggml type → device format (Q8_0, Q4_K native; others `to_float` + `quantize_row_q8_0_ref`) and back |
| `tsr-alloc.{h,cpp}` | first-fit range allocator for IO / AUX / WEIGHT heap |
| `tsr-wlayout.{h,cpp}` | device-format row bytes, row repack + symmetric scatter / gather over the channels (no ggml) |
| `lower.cmake` | static library `tsr_lower` = tsr-lower + tsr-opl + tsr-alloc + tsr-wlayout (no ggml; links `tsr_transport`, includes `sw/ctrl/include`), linked by this backend and by the runtime `sw/rt` (`tsr_rt`) |
| `sources.cmake` | source list (the ggml-dependent files), included by llama.cpp's `ggml/src/ggml-tesseract/CMakeLists.txt` (patch 0001); links `tsr_lower` with a deferred call, so the patch is unchanged |
| `CMakeLists.txt`, `tests/` | unit tests and the end-to-end test (added by the top-level CMakeLists after llama.cpp) |

Public C API (`third_party/llama.cpp/ggml/include/ggml-tesseract.h`, from the patch):
`ggml_backend_tesseract_reg()`, `ggml_backend_tesseract_init()`, `ggml_backend_is_tesseract()`,
`ggml_backend_tesseract_buffer_type()`. ggml registers the backend itself under
`GGML_USE_TESSERACT`.

## Design (where to look)

- **Transport and firmware**: `tsr_transport_open(NULL)` on first use (`tsr_dev_open_locked`;
  `SOFT_RESET` + bind), config from `tsr_transport_config`, then `tsr_opl_dev::open` boots
  `$TSR_IMAGE` or the build's `tsr-fw.elf` (`tsr_image_open_default`, warm if the image is already
  loaded) unless the transport runs an image already, and finds the `graph` kernel. All device
  state is behind one mutex (`tsr_device::mtx`); the transport has its own.
- **Buffer type = virtual address space**: each buffer gets a 64-aligned base in a
  non-canonical range (`0x4000...`, never dereferenced); `tensor->data = base + offset`.
  Alignment 64, `get_alloc_size` = nbytes rounded up to 64 (the line-granular ST tail of the
  last output row lands in this padding), `get_max_size` = IO window capacity (IO region minus
  the stage: 240 MiB), so ggml-alloc cuts compute buffers into chunks that can each get a window.
- **Compute windows (W3b, SW §8 R6, SW-26).** A buffer the device addresses in place (`io`)
  gets one window in one channel: the ch0 IO region, else the low 512 MiB of ch1..ch3 (the channel
  with the most free bytes), else the compute heap: symmetric reservations (≥ 64 MiB) of the WEIGHT
  heap between the weights (bottom-up) and the KV planes (top-down), shared out per channel.
  `get_max_size` is 512 MiB, so the reserved logits of Qwen2.5 (297 MiB at `-ub 512`) or Gemma 4
  (512 MiB) are one chunk on the device. `get_alloc_size` pads to 64 bytes only where ggml allows it
  (leaves, views, quantized types, MUL_MAT / FLASH_ATTN_EXT: `[TAG_ALLOC_SIZE_EXPAND]`); ggml-alloc
  pads every block to the 64-byte alignment anyway, which holds a single row's whole-line store.
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
    start: no op list addresses it, so it takes no IO.
  - IO layout: windows below, the **stage** at the top (max(16 MiB, 2 × SPM), reserved at open).
- **MUL_MAT src0 resolution** (`tsr_resolve_weight`): a DEVWEIGHT is used in place; any other
  src0 (raw bytes in IO, host memory or AUX, any loadable type) is converted lazily into the
  heap at graph_compute and cached by (buffer id, offset, type, N, K) and the buffer's write
  generation (bumped by every host write and by device writes into it). test-backend-ops
  therefore reads back exact src0 bytes, and in a pre-allocated graph an F32 src0 produced by
  an earlier device op works (the pending op list is launched first). In llama.cpp graphs
  supports_op keeps such src0 on the CPU (see supports_op).
- **Split uploads**: a DEVWEIGHT is staged on the host until every byte has arrived, then
  converted and scattered row by row; updates of a complete weight re-store whole rows
  (partial rows: read-modify-write for Q8_0/Q4_K, abort for converted types).
  `get_tensor` of a DEVWEIGHT un-repacks (exact for Q8_0/Q4_K; converted types are re-encoded
  from the device copy with a warning — debug only).
- **graph_compute** (`tsr_backend_graph_compute`, SW §9.3 steps 1-5): skips nodes without
  `GGML_TENSOR_FLAG_COMPUTE` and NONE/VIEW/RESHAPE/PERMUTE/TRANSPOSE; each MUL_MAT
  (`tsr_lower_mul_mat`) becomes one MUL_MAT record of a *segment* (`tsr_segment` = a
  `tsr_opl_builder`): x / y handles of its IO window rows (strides in lines), W_OFF, fmt, N, K and
  the whole T (src1 `[K, T, B2, B3]` contiguous is flattened); no addend, no ACC_DST in M6.
  Dependencies come from DRAM byte ranges: a record depends on an earlier one iff it reads what
  that one writes or writes what that one reads or writes (x rows, the weight on every channel,
  the line-padded y rows); conflicts implied through another kept predecessor are dropped, and
  more than two predecessors are folded by JOIN records. The segment's list (graph_id = its CRC)
  goes into the **OPL cache** (`tsr_opl_cache`: LRU over the 64 MiB OPL region, keyed by a 64-bit
  hash of the bytes and confirmed by a byte compare, so a list seen before, e.g. the same split of
  the next decode token, is not written again) and runs as one `graph` launch (a0 handle, a1
  bytes, a2 flags; `tsr_transport_call` with the transport's debug-aware timeout). A list is
  launched early when it would exceed 4096 records, before a src0 in a buffer it writes is
  converted, and before a staged MUL_MAT. Any answer but DONE OK with RET 0 aborts: on TILE_ERR
  RET − 1 is the failing op, mapped back to its node name, with the first error's code, ERR_INFO,
  ticket, w0 and push index; executor refusals (KERNEL_ERR, RET 0x8000_000x) by name.
- **Staged MUL_MATs** (`tsr_lower_staged`): src1 or dst in a HOSTED buffer, or dst rows that
  are not line-aligned (N*4 % 64 != 0 with T > 1). Everything pending runs first, then the
  MUL_MAT runs on its own in T tiles that fit the stage, one single-record launch per tile:
  src1 rows host → stage (a LINEAR src1 is read in place), the record (the firmware's LD / QACT /
  MVM / ST), dst rows stage → host or → the LINEAR dst (compacted to N*4 bytes). Only one tile
  has to be resident in IO, whatever T is, so supports_op has no token-count limit. Since W3b's
  compute windows only a device without room (or `TSR_FORCE_HOST=1` / `TSR_COMPUTE_WINDOWS=io`)
  leaves a compute chunk in host memory (only MUL_MAT is staged: an M7 op on such a chunk aborts
  graph_compute naming it, SW-53); before them the reserved logits of a large vocabulary
  (Qwen2.5 at -ub 512: 296.75 MiB) were such a chunk, one extra launch per token (TSR_STATS: "staged
  chunks"). (Single-token outputs with N*4 % 64 != 0 ST straight into a LINEAR dst; the tail lands in
  the 64-byte padding.) LLM projections have N % 16 == 0 and are staged only through HOSTED buffers.
- **M7 graph_compute and supports_op (W3b, `tsr-build.h`).** Every non-MUL_MAT node of SW §9.3's
  table is lowered by `tsr_lower_m7` to its record (SW §7.2 words): F32 row operands as a handle and a
  row stride in lines (the 64-byte rule: base and stride multiples of 64 bytes; a result that is a view
  has rows of whole lines, since the kinds store whole lines; a result of its own may be one row ending
  inside a line), ADD / MUL broadcast as a row period or per-row scalars
  (`FLAG_B_SCALAR`), ROPE on `[head_dim, n_heads, T]` views with contiguous heads (a fused-qkv view's
  token stride), SET_ROWS / FLASH_ATTN on KV planes (`tsr_kv_view_of`; views as llama.cpp's
  `cpy_k` / `get_k` build them with one stream: cell 0, cells `row_size(E)` and heads `row_size(D)`
  apart), FLASH_ATTN's mask through its
  summary (`tsr_fa_mask_ref`), index inputs (positions, ids, KV indices: graph leaves) and MSUMs as
  preload entries. `supports_op` runs the same lowering in probe mode and the kind's planner
  (`plan.hpp` `plan_kind` on the device's CFG values, SW-20). Operands in host memory (a HOSTED
  buffer, or one that will be: an ANY buffer of leaf data only, e.g. llama.cpp's control vectors and
  LoRA adapters, and every buffer under `TSR_FORCE_HOST`) are refused, since only MUL_MAT is staged
  (SW §8 R6 as built, SW-53): such ops run on the CPU. A compute buffer that gets no window is
  placed after the scheduler split the graph; graph_compute then aborts naming it. Fusion (`tsr_build_fusion`): a
  MUL_MAT whose only user (ggml use count over the whole graph) is an ADD of a bias row or the
  residual takes it as its addend; an RMS_NORM whose only user is a MUL by a weight takes the weight;
  the fused record sits at the first node's place when nothing in between writes the other operand or
  touches the result (bit-identical: `ggml_tesseract_opl_launch` F). FLASH_ATTN's FA_NEW word names
  the indices of the SET_ROWS_KV that wrote its K plane (AT §12). A list is cut early only at 4096
  records or a full 256-line preload area (prefill ubatches of a few hundred tokens: TSR_STATS "lists
  cut early"); a FLASH_ATTN's MSUM and FA_NEW indices must fit one list together, so one of more than
  680 tokens is refused and runs on the CPU (SW-43). When a cut separates a SET_ROWS_KV from its
  FLASH_ATTN, graph_compute checks that no node between them overwrote the K indices the FA's list
  preloads again (tsr-build.h). The plan of a graph (its lists) is cached by `tsr_graph_key` (addresses, shapes, op
  parameters; not values) salted with a placement epoch; a cached plan re-registers its FLASH_ATTN
  nodes as mask readers and runs. Nodes without `GGML_TENSOR_FLAG_COMPUTE` are skipped; supports_op
  does not test the flag (test-backend-ops asks before the flag is set), DUP and non-KV SET_ROWS are
  refused (R1). A refused op writing into the device KV cache (context shift, defragmentation, a
  SET_ROWS into a cache of several sequence streams or under `TSR_FORCE_HOST`) is reported once
  before the scheduler stops on it (SW §8 R10). llama.cpp's weight-placement probe of `rope_freqs` (ROPE with every parameter 0) is
  accepted, so Gemma 4's freq factors live on the device.
- **supports_op of MUL_MAT** (the rules since M6): MUL_MAT with src1/dst
  F32, src0 `ne2 = ne3 = 1`, all contiguous, src0 type loadable, `tsr_mm_shape_ok` (K ≤ K_MAX,
  K % 32 == 0 (Q4_K: % 256), one token fits SPM) and `plan.hpp`'s `mm_supported` on the device's
  CFG values (one token fits the firmware's scratch-ring budget: the host rejects exactly what the
  `graph` kernel cannot plan, SW-20),
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
| `TSR_IMAGE` | firmware image (default: the build's `tsr-fw.elf`) |
| `TSR_DEBUG=1` | placement, lowering and launch log on stderr; `2` adds buffer events, the records and preload entries of every launched op list and supports_op's reasons |
| `TSR_KINDS` | the kinds the device takes, a comma-separated list read from left to right: `m6` (NOP, JOIN, MUL_MAT; M6 runs; `m6,+MUL_MAT_ID` for a MoE model, what `klcheck` and `tsr-numcheck` `--phase m6` set), `all`, `NAME` / `+NAME` adds a kind, `-NAME` removes one (`all,-FLASH_ATTN,-RMS_NORM`). What a list starts from: a list whose first word is a removal starts from every kind; any other list starts from NOP and JOIN only, which cannot be removed. So `TSR_KINDS=+FLASH_ATTN` alone turns MUL_MAT off; write `m6,+FLASH_ATTN` or `all,...` to keep it. The removed kinds' ops run on the CPU; a SET_ROWS into a device KV cache cannot, so SET_ROWS_KV leaves the device only with `-nkvo` (which keeps FLASH_ATTN on the CPU too). Unset: every kind of the booted image |
| `TSR_COMPUTE_WINDOWS=io` | test knob: compute windows in ch0 IO only (M6 placement; larger buffers in host memory, their MUL_MATs staged) |
| `TSR_OPL_SPLIT=<n>` | test knob: cut every op list after n records (a split's results do not depend on the cuts: Qwen2.5-0.5B with n = 7, 53 launches per token, gives logits bit-identical to one launch per token) |
| `TSR_FORCE_HOST=1` | test knob: every non-WEIGHTS buffer in host memory, so every MUL_MAT takes the staged path and the other kinds stay on the CPU (with M7 settings add `-nkvo`: a KV cache on the device needs its SET_ROWS on the device) |
| `TSR_STAGE_KIB=<n>` | test knob: IO stage size in KiB (default max(16 MiB, 2 × SPM)); a small stage forces several T tiles |
| `TSR_DUMP_RE=<regex>` `TSR_DUMP_DIR=<dir>` | debug knob: when a graph is built, run the pending op list after every node whose result name matches the regex, read the result from device memory, print its range to stderr (`DUMP name ...`) and write it to `<dir>/<name>.f32`. Compare with the same nodes of a CPU run (`cb_eval` on the CPU backend) to find the first node that differs. Only the first run of a graph dumps (later runs use the cached plan) |
| `TSR_OPL_CRC=1` | debug knob: `graph` launch flag VERIFY_CRC (the executor checks each list's CRC-32 before running it) |
| `TSR_STATS=1` | at backend free: upload statistics; graph_compute calls, launches, launches per scheduler pass (≈ per token: a split seen again starts a pass) and per graph_compute (0 / 1 / 2 / ≥ 3), op records per launch, records per kind, fused ops, MUL_MAT / JOIN records, staged chunks and their launches, the plan cache (hits, builds, builds not cacheable), lists cut early, OPL cache hits / misses / evictions, launch wall time, PERF counter deltas; with the KV agent's lines on mask summaries |
| `TSR_SIM_*`, `TSR_TRACE`, `TSR_LOG`, `TSR_LAUNCH_TIMEOUT_MS`, … | handled by the transport (`sw/transport/tsr_transport.h`, SW §9.2) |

## Build and test

```
third_party/apply-patches.sh
cmake -S . -B build/dev -DCMAKE_BUILD_TYPE=Release
cmake --build build/dev -j4 --target test-backend-ops test-ggml-tesseract test-tsr-backend-units llama-completion
build/dev/bin/test-tsr-backend-units                        # allocator, tiling, OPL builder + cache
build/dev/bin/test-ggml-tesseract                            # bit-exact vs tsr_ref + NMSE vs CPU
TSR_SIM_TMAX=8 build/dev/bin/test-ggml-tesseract             # T tiling with T > 1
build/dev/bin/test-backend-ops test -o MUL_MAT -b Tesseract0 # vs CPU, NMSE 5e-4
LLAMA_ATTN_ROT_DISABLE=1 TSR_KINDS=m6 build/dev/bin/llama-completion -m models/qwen2.5-0.5b-instruct-q8_0.gguf \
    -ngl 99 -nkvo -fa on -ctk q8_0 -ctv q8_0 -p "..." -n 16          # M6 settings (SW §10.4)
LLAMA_ATTN_ROT_DISABLE=1 TSR_STATS=1 build/dev/bin/llama-completion -m models/qwen2.5-0.5b-instruct-q8_0.gguf \
    -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -p "..." -n 16                # M7: one launch per token
```

Configure only through the top-level CMakeLists. It defines `tsr_ref`, `tsr_transport` and
`tsr_lower` (`lower.cmake`) before llama.cpp. `sources.cmake` stops with a message if `tsr_lower`
is missing, and `test-tsr-backend-units` links it. The same tree adds the runtime `sw/rt` after
llama.cpp, so its ctest also lists `rt_sim`, `rt_sim_tmax8`, `rt_mock` and `rt_mock_tmax8`
(`sw/rt/README.md`; the PyTorch backend on top of it: `docs/torch-backend.md`).

ctest names: `ggml_tesseract_units` (allocator, tiling, staging decisions, OPL builder: encoding,
hazards incl. strided rows, transitive reduction, JOINs, the 4096-op limit, a random property check;
OPL cache; W3b: the DRAM ranges of every M7 kind, the preload table, and with ggml-base the M7 builder:
fusion pairs and non-pairs, records of views and KV planes with their dependencies and FA_NEW, the
graph key incl. shape and stride changes, supports_op against the planners over 2,850 shapes and the
rules the sweep does not reach: strided and partial-line views, KV views, the FLASH_ATTN preload limit), `ggml_tesseract_opl_launch` and
`_mock` (`tests/test-tsr-opl-launch.cpp`: the launcher on a device: a JOIN and an ACC_DST record
against the spec model, a cache hit, TILE_ERR and KERNEL_ERR answers mapped to their op; W3b: fusion
bit-identical to the unfused records, a chain of every short kind with preload entries whose inputs
change through the cached list, a TILE_ERR in GET_ROWS named by its tag), `ggml_tesseract_e2e`, `ggml_tesseract_e2e_tmax8`,
`ggml_tesseract_e2e_hosted` and `ggml_tesseract_e2e_hosted_tmax8` (`TSR_FORCE_HOST=1
TSR_STAGE_KIB=64`, with and without `TSR_SIM_TMAX=8`), `transport_ggml_e2e_mock` and
`transport_ggml_e2e_mock_tmax8` (the same test on the mock kdev; defined here since P3.2; the e2e
tests are M6's and run with `TSR_KINDS=m6 TSR_COMPUTE_WINDOWS=io`),
`ggml_tesseract_scalar_nmse`, `ggml_tesseract_backend_ops` and `ggml_tesseract_backend_ops_bigt`
(label `slow`: the one case `MUL_MAT(f16, m=16, n=2105344, k=32)`, 8.4 M single-token tile ops
through the interpreted firmware at the sim's T_MAX 1, ~50 s; the two together run the whole case set),
and `ggml_tesseract_server_tiny` (`tests/server_smoke.py`, no model file needed: `llama-server` on the
tiny Qwen2 model, the device and the CPU side by side with 2 slots; every device text must equal the
CPU's, both servers must exit 0, and the device runs one launch per `graph_compute`; about 2 s).
The device ctests run with the simulator's race detector fatal (`TSR_SIM_RACE=fatal`, SW §4.6),
except `_bigt` (over 15 minutes with it). `TSR_TEST_DUMP=<file>` makes
`test-ggml-tesseract` append every device output it reads, so two builds can be compared byte for
byte. Test hook for the storage decision:
`ggml_backend_reg_get_proc_address(reg, "ggml_backend_tesseract_buffer_storage")` returns
`const char * (*)(ggml_backend_buffer_t)` ("undecided", "io", "placed", "host").

`ggml_tesseract_backend_ops` leaves out the single-output case
`MUL_MAT(type_a=f32,type_b=f32,m=1,n=1,k=2048)`: with fresh random data each run its NMSE is one
squared relative error, which the spec's 8-bit path pushes over 5e-4 in ~15% of runs on a correct
device. `ggml_tesseract_scalar_nmse` (`tests/test-tsr-scalar-nmse.cpp`) covers that shape
bit-exactly against the spec model instead; numbers in `docs/llama-pin.md` (test-backend-ops).
The unfiltered `test-backend-ops test -o MUL_MAT -b Tesseract0` above therefore reports
1402/1403 in about one run in seven.

M6 runs llama.cpp with `-nkvo` (KV cache on the CPU) and `TSR_KINDS=m6`; M7 without `-nkvo` (the
KV cache in KV buffers), with `-fa on -ctk q8_0 -ctv q8_0` and `LLAMA_ATTN_ROT_DISABLE=1`, always with
an explicit `-ngl`; see `docs/llama-pin.md` for how compute buffers, the output head and LoRA adapters
are placed. `klcheck run` and `tsr-numcheck run` set `TSR_KINDS=m6` for `--phase m6`; for `--phase m7`
they keep the caller's `TSR_KINDS` (unset: every kind), so a list such as `all,-RMS_NORM` reaches the
backend (SW-49). The value is recorded in klcheck's `report.json` and in numcheck's `meta.json`.

## llama-server

`llama-server` runs on the Tesseract backend from the same build, with the M7 settings:

```
LLAMA_ATTN_ROT_DISABLE=1 TSR_DEVICE=sim build/dev/bin/llama-server -m MODEL -ngl 99 -fa on \
    -ctk q8_0 -ctv q8_0 -np 4 -kvu -c 4096
```

- `-ngl 99 -fa on -ctk q8_0 -ctv q8_0` are required: an f16 KV cache cannot live in device memory.
- `-kvu` (`--kv-unified`) is needed for `-np` above 1: the device holds one KV stream.
- Do not use `--context-shift`: llama.cpp crashes in `build_rope_shift` with a quantized K cache
  and no rotation, on the CPU backend too.
- The endpoints, the request example and the measured speed are in
  [`docs/getting-started.md`](../../docs/getting-started.md#serving). The ctest is
  `ggml_tesseract_server_tiny` (above).

## M7 on the simulator (W3b integration, 2026-10-31)

Settings: `--phase m7` (FA on, q8_0 / q8_0 KV on the device, `LLAMA_ATTN_ROT_DISABLE=1`), `TSR_DEVICE=sim`
(4 channels, T_MAX 1), race detector off unless noted; sim times are host wall time of the interpreted
firmware on this 16-core host, device times are `PERF_CYCLES` at the restated k4 clock of 276.3 MHz.

| Model | Launches | Splits per decode graph | KL vs CPU (klcheck criterion) | Sim time per token |
| --- | --- | --- | --- | --- |
| Qwen2.5-0.5B Q8_0, 8 × 64 tokens | 512 for 504 decodes + 8 prefills (1.00 per token), 0 staged | CPU + 1 Tesseract | mean 2.22·10⁻³, floor 2.43·10⁻³, ratio 0.914, top-1 98.05 %: PASS | 89 ms (11.3 tokens/s) |
| Gemma 4 12B Q4_K_M, 2 × 33 tokens | 66 for 64 decodes + 2 prefills, 0 staged at `-ub 512 -c 4096` | CPU + 1 Tesseract | mean 3.07·10⁻², floor 4.51·10⁻², ratio 0.680, top-1 96.97 %: PASS | 0.82-0.86 s |
| Gemma 4 12B Q8_0, 2 × 33 tokens | 66 for 64 decodes + 2 prefills, 0 staged at `-ub 512 -c 4096` | CPU + 1 Tesseract | mean 2.23·10⁻², floor 2.65·10⁻², ratio 0.843, top-1 96.97 %: PASS | 1.5-1.75 s |
| Qwen2.5-32B Q4_K_M, smoke 1 × 2 tokens (`TSR_SIM_DRAM_DIR`) | 2 for 1 decode + 1 prefill, 0 staged | CPU + 1 Tesseract | 2 rows: mean 4.63·10⁻³, floor 4.80·10⁻³, ratio 0.964, top-1 2 / 2 | 7.44 s; full 8 × 64 klcheck projected ≈ 72 min |

- **Identity.** A second 0.5B sim run replaying the tokens is bit-identical (512 / 512 rows); so is a
  replay with `TSR_SIM_SCHED=random:7 TSR_SIM_RACE=fatal` (no race in 512 launches of the whole graph),
  and a replay of the Gemma 4 12B Q4_K_M tokens (66 / 66 rows).
- **Qwen2.5-32B.** One decode token is 83.5 M device cycles (MVM 81.5 M) = 0.30 s at 276.3 MHz (the
  difference of a 2- and a 3-token run; card measurement before the control core: 341.7 ms per token), 965 records per launch.
- **Device cycles per 0.5B decode token** (two lengths, 32 decodes apart): M7 2.48 M cycles (MVM 2.22 M,
  incl. ATTQK / ATTPV) in 1 launch = 9.0 ms, ≈ 111 tokens/s of device time; M6 2.63 M cycles (MVM 2.08 M)
  in 145 launches, with attention, norms and the other ops on the CPU besides. Records per 0.5B decode
  launch: 365 (169 MUL_MAT, 168 of them fused with an ADD or a MUL, 49 RMS_NORM, 24 GLU, 48 ROPE,
  48 SET_ROWS_KV, 24 FLASH_ATTN, 2 GET_ROWS, 1 ADD); Gemma 4 12B: 1,152 records, 289 fused pairs.
- **Plan cache.** 0.5B 8 × 64: 512 `graph_compute` calls, 6 builds, 506 cached runs; Gemma 4 12B: 3
  builds for 66 calls.

## KV cache and mask summaries (M7, W3b P3.5)

`tsr-kvbuf.{h,cpp}` and `tsr-msum.{h,cpp}` (no ggml, part of `tsr_lower`; their headers are the
reference) implement SW §9.3's "KV buffers" (SW-17) and "Mask summaries" (SW-32):

- **KV buffers.** A buffer of usage ANY whose tensors are all `cache_<tag>k_l<n>` /
  `cache_<tag>v_l<n>` q8_0 `[E, kv_size, n_stream]` (an offloaded llama.cpp cache, `-ctk q8_0 -ctv
  q8_0`, AT §4.6) becomes storage `kv` at its first data access (llama.cpp's clear): one plane per
  (tensor, stream) in AT's layout (K token-major, V slice-major, LG = ⌈kv_size / 16C⌉), allocated
  symmetrically and top-down in the WEIGHT heap (EX §11, AT-21). `set_tensor` / `get_tensor` /
  `memset_tensor` / `clear` on any byte range convert between ggml `block_q8_0` rows and the planes
  with AT's C reference (`tsr_kv_set_bytes` / `tsr_kv_get_bytes`) in windows of whole local groups
  (read-modify-write unless the access covers the window's rows), so state save / load is lossless.
  The builder resolves the SET_ROWS / FLASH_ATTN views with `tsr_kv_view_of` (plane base beat,
  GEOM word, stream) and supports_op asks `tsr_kv_tensor_supported`.
- **Mask summaries.** `get_alloc_size` of an F16 `*kq_mask*` tensor `[n_kv, n_q, 1, 1]` reserves,
  behind the mask, its MSUM (AT §7.3) and one f32 copy (`n_q · 64 · ⌈n_kv/16⌉` bytes). On every
  `set_tensor` of such a mask in device memory (the scheduler's input copy, once per token) the
  producer writes the MSUM (`tsr_msum_build_f16`) and, when some tile reads it, the f32 copy
  (`tsr_msum_mask_f32`, fdiv(m, scale)). The builder registers every FLASH_ATTN as a reader of its
  mask (`tsr_fa_mask_ref`: scale, E, D, G) and gets the preload source (word 12) and the copy handle
  for its scale (word 13). Decisions (tsr-msum.h): readers only grow; one copy per distinct scale (the
  first in the reservation, further ones as side allocations); a copy is written when the MSUM is
  GENERAL or the shared planner puts a tile on the GENERAL path; a tile the planner would refuse
  (AT §6.6 rule 5 / 6) promotes the MSUM to GENERAL, so the executor never refuses for mask data.
- **Tests.** `ggml_tesseract_kvbuf` (`tests/test-tsr-kvbuf.cpp`, no device: placement, conversion
  against `tsr_kv_pack_row` images after random partial accesses and windows, state save / load,
  MSUM / copies / readers / promotion / side copies) and `ggml_tesseract_attn`, `_tmax8`, `_split1`
  (`TSR_OPL_SPLIT=1`) (`tests/test-tsr-attn.cpp`, on the device: KV buffers through the ggml API, the
  device's MSUMs and copies against the references, and SET_ROWS + FLASH_ATTN_EXT in llama.cpp's graph
  shape against the CPU backend (NMSE) and `tsr_ref_fa` (bit for bit), several tokens per graph
  through the cached op list, incl. a wrapped ring whose output bits depend on FA_NEW; supports_op's
  placement rules: results that are views, leaf data in host memory) and `ggml_tesseract_kvstate` (`tests/test-tsr-kvstate.cpp`, llama.cpp's state API on the tiny
  Qwen2 model: device → device bit-identical logits, the device's state file loaded by a CPU context
  byte for byte, sequence state into a cleared context). Test hooks:
  `ggml_backend_tesseract_kv_plane`, `ggml_backend_tesseract_mask_summary`,
  `ggml_backend_tesseract_att_cfg` (signatures at their definitions in `ggml-tesseract.cpp`).

## M7 on the simulator: tiny models, graph tests, test-backend-ops per op (P3.6)

Everything is defined in `tests/tiny.cmake` (included from `CMakeLists.txt`); the rationale of each
bound and the W3b measurements are in its header.

- **Tiny models (SW-21).** `tests/gen_tiny_models.py` writes, at build time (target
  `tsr_tiny_models`, part of `all`), `build/<tree>/models/tiny/`: `qwen2-tiny-q8_0.gguf` and
  `qwen2-tiny-q4_k.gguf` (2 layers, n_embd 256, GQA 4:2, head_dim 64, Q/K/V bias, SWIGLU),
  `gemma4-tiny-q8_0.gguf` (4 layers: 3 SWA with head_dim 64 / GQA 16:8 and 1 full with head_dim 128 /
  16:1, no `attn_v` on the full layer, per-head Q/K norms, `rope_freqs`, `layer_output_scale`, GEGLU,
  tied head, softcap 30, n_swa 16) and `tiny-prompts.txt`. Random weights from numpy's legacy
  `RandomState` (identical files on every host), 64-row vocabularies with valid tokenizer metadata
  (Qwen2: GPT-2 byte-level BPE `gpt2` / `qwen2`; Gemma 4: `gemma4` BPE), Q8_0 by gguf-py, Q4_K by
  the script's encoder checked against gguf-py's dequantizer. No binary is in git. Python: the first
  of `TSR_TINY_PYTHON`, `build/tools/torchvenv/bin/python`, `TSR_REF_MATH_PYTHON` and python3 that
  imports numpy and `third_party/llama.cpp/gguf-py`.
- **Graph tests** (`tests/m7_tests.py graph`): `ggml_tesseract_graph_qwen_tiny`, `_qwen_tiny_q4k`,
  `_gemma_tiny` (1-6 s each) and, with `models/qwen2.5-0.5b-instruct-q8_0.gguf` present,
  `ggml_tesseract_graph_qwen05b` (label `model`, ~23 s). klcheck generates on `sim` with the M7
  settings (`--phase m7`, the SWA ring as llama-cli) under `TSR_DEBUG=1`, `GGML_SCHED_DEBUG=2`,
  `TSR_STATS=1` and `-v` (decode markers); the test asserts one Tesseract split in every single-token
  decode graph, one launch per single-token decode (64 for the tiny models, 128 for 0.5B), one launch
  per ubatch in prefill (`-ub 16`: 3-22 ubatches per prompt; the long prompts cross the 256-cell KV
  bucket in decode and in prefill and wrap Gemma 4's SWA ring), no launch outside a decode, the launch
  count of TSR_STATS, the embd-path identity, and klcheck's KL criterion against the CPU with the mean
  of the floor seeds 1 to 3 (Gemma 4 with `--min-kl 3e-3`, see the header). The tests ignore the shell's
  `TSR_KINDS`: they check the full kind set. `ggml_tesseract_graph_qwen_tiny_ub1024` runs one 692-token prompt
  (`--join-prompts 4`) at `-ub 1024`: its prefill FLASH_ATTN (T > 680) stays on the CPU, so prefill
  launches are only reported (`--prefill any`), with `--min-prefill-tokens 681`. Files of a run: `build/<tree>/sw/ggml-tesseract/<test>/`
  (`dev.log`, `graph.json`, `report.json`). By hand:
  `python3 sw/ggml-tesseract/tests/m7_tests.py graph --klcheck build/dev/bin/klcheck --model
  build/dev/models/tiny/gemma4-tiny-q8_0.gguf --prompts build/dev/models/tiny/tiny-prompts.txt
  --out-dir /tmp/g --min-kl 3e-3`.
- **Gated DeltaNet (Qwen3.5 / 3.8, SW §7.10).** `qwen35-tiny-q8_0.gguf` and `qwen35-tiny-s128-q8_0.gguf`
  (4 layers: 3 Gated DeltaNet and 1 full attention with the joint Q and gate projection, per-head norms and IMROPE;
  S_v 32 with 4 key heads for 16 value heads, and S_v 128 with the real models' 16 key heads and D 256), the graph
  tests `ggml_tesseract_graph_qwen35_tiny` and `_s128` and `ggml_tesseract_gdn_logits_<model>`
  (`tests/test-tsr-gdn-logits.cpp`: the logits of every step of prompts of 1, 2, 5 and 17 tokens against the CPU).
  The logits test exists because klcheck's mean KL on a random tiny model did not see a convolution that returned
  zeros (SW-71, SW-74). The device holds the conv and recurrent state caches (`cache_r_l*`, `cache_s_l*`) as device
  windows; with `-nkvo` the ops that touch them stay on the CPU.
- **test-backend-ops per op** (`tests/m7_tests.py ops`): `ggml_tesseract_backend_ops_<op>` for ADD,
  MUL, RMS_NORM, SWIGLU, GEGLU, ROPE, GET_ROWS, CPY, CONT, TANH, GELU, SCALE, SOFTCAP, the fused cases
  RMS_NORM_MUL_ADD, ADD_RMS_NORM, ADD_ADD, MUL_MAT_ADD, RMS_NORM_MUL_ROPE, and SET_ROWS, ROPE_SET_ROWS,
  FLASH_ATTN_EXT (q8_0 K / V): Tesseract0 against the CPU with the race detector fatal; every supported
  case must pass, at least one must be supported (none for SCALE / SOFTCAP, whose test shapes have
  10-float rows, and the KV-cache ops, which need KV-buffer tensors: `ggml_tesseract_attn`), and the
  unsupported cases are counted, listed and written to
  `build/<tree>/sw/ggml-tesseract/backend_ops/<op>.unsupported.txt`. `ggml_tesseract_backend_ops`
  (MUL_MAT) is unchanged.
