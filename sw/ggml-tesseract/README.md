# ggml-tesseract

The ggml backend for Tesseract. It registers one backend `"Tesseract"` with one device `"Tesseract0"` of type
`GGML_BACKEND_DEVICE_TYPE_GPU`. The pinned llama.cpp and the API facts the backend depends on are in
[`docs/llama-pin.md`](../../docs/llama-pin.md). The design is [SW](../../docs/design/06-software-verification.md) §7
to §9 and [`docs/spec.md`](../../docs/spec.md) §12.

In full mode a whole decode token of a supported model runs as one launch of the firmware's `graph` kernel with the KV
cache on the device (`-fa on -ctk q8_0 -ctv q8_0`, no `-nkvo`, `LLAMA_ATTN_ROT_DISABLE=1`). In matmul mode
(`TSR_KINDS=m6`, `-nkvo`) only the matrix products run on the device. The modes are defined in
[`docs/design/README.md`](../../docs/design/README.md). The backend lowers the nodes of a ggml graph into an op list of
records (SW §7.2) with the builder in `tsr-build.{h,cpp}`: it fuses MUL_MAT + ADD and RMS_NORM + MUL, and the OPL builder in `tsr-opl.{h,cpp}` splits a MUL_MAT whose result the next record leaves alone into MUL_MAT_BEGIN and MUL_MAT_END (SW §9.3) and orders the
records by DRAM byte ranges (a record depends on an earlier one iff it reads what that one writes or writes what that
one reads or writes; more than two predecessors are folded by JOINs), puts positions, ids, KV indices and mask
summaries into the preload table, and caches each graph's plan by identity (`tsr_graph_key`: addresses, shapes, op
parameters, not values). The op list goes into an OPL cache in the ch0 OPL region and runs as one launch, so a decode
token reuses its list and only its inputs change. `supports_op` runs the same lowering in probe mode and the kind's
planner (`plan.hpp`, SW-20), so the host refuses exactly what the firmware cannot plan.

- Placement. A buffer's storage is fixed at its first data access. Weights (usage WEIGHTS) are converted to Q8_0
  unless they are Q8_0 or Q4_K, repacked and allocated symmetrically in the WEIGHT heap. 3D expert tensors of a MoE
  model are placed per channel (SW §7.9). Compute buffers get a window in device memory (the ch0 IO region, the low
  512 MiB of ch1-3, or the compute heap), so nothing is staged. A buffer of KV cache planes becomes storage `kv`
  (`tsr-kvbuf`) in the layout of AT §4.6, and an F16 `*kq_mask*` gets its mask summaries (`tsr-msum`, AT §7.3).
  Buffers in host memory (LoRA adapters, control vectors, or `TSR_FORCE_HOST=1`) are staged for MUL_MAT only, and
  other ops on them run on the CPU.
- Recurrent models and multi-token prediction. Gated DeltaNet layers (SW §7.10) keep their state caches as device
  windows, and a `GATED_DELTA_NET` sits on its cache cell when every sequence keeps its cell (SW-82). Rollback
  snapshots (`n_rs_seq` = K, SW-83 to SW-86) lower to chains of records. A router of more than 64 tokens writes its
  ids and weights to a scratch window and two `CPY` records move them (SW-76).

| File | Content |
| --- | --- |
| `ggml-tesseract.cpp` | reg, device, buffer type, buffer and backend, placement, lazy conversion, segments, staging, execution |
| `tsr-opl.*`, `tsr-opl-dev.cpp` | the op-list path (no ggml): the OPL builder, the OPL cache, the `graph` kernel on a transport |
| `tsr-build.*`, `tsr-lower.*` | the graph builder (ggml-aware, device-free) and MUL_MAT shape checks and SPM tiling |
| `tsr-pivot.*`, `tsr-pivot-graph.*` | Tesseract Pivot, the graph scheduler (`graph_optimize`; [PV](../../docs/design/07-graph-scheduler.md)): the core (units, machine model, search; no ggml) and the ggml adapter (units from nodes, order constraints, the node validator) |
| `tsr-weights.*`, `tsr-alloc.*`, `tsr-wlayout.*` | ggml type to device format and back, the first-fit allocator, row repack with symmetric scatter and gather |
| `tsr-kvbuf.*`, `tsr-msum.*`, `tsr-prof.*` | KV buffers, mask summaries, the plan dump for the profile report ([`docs/prof-dbg.md`](../../docs/prof-dbg.md) §3.1) |
| `lower.cmake`, `sources.cmake` | `tsr_lower` (the ggml-free files, shared with `sw/rt`) and the source list that llama.cpp's `ggml/src/ggml-tesseract/CMakeLists.txt` (patch 0001) includes |

The public C API is `ggml-tesseract.h` (from the patch). `ggml_backend_reg_get_proc_address` returns test hooks
(`ggml_backend_tesseract_buffer_storage`, `_kv_plane`, `_mask_summary`, `_att_cfg`, and `_counters`, the device's PERF
counters used by `tsr-mtp`).

## Environment

| Variable | Effect |
| --- | --- |
| `TSR_DEVICE`, `TSR_IMAGE` | transport kind (`sim` default, `xrt`; `none` registers no device) and firmware image |
| `TSR_KINDS` | the kinds the device takes, a comma-separated list read from left to right: `m6` (NOP, JOIN, MUL_MAT; `m6,+MUL_MAT_ID` for a MoE model), `all`, `NAME` or `+NAME` adds a kind, `-NAME` removes one (`all,-FLASH_ATTN,-RMS_NORM`). A list whose first word is a removal starts from every kind, and any other list starts from NOP and JOIN only, so `TSR_KINDS=+FLASH_ATTN` alone turns MUL_MAT off. The ops of a removed kind run on the CPU. SET_ROWS into a device KV cache cannot, so SET_ROWS_KV leaves the device only with `-nkvo`. Unset: every kind of the booted image. `klcheck` and `tsr-numcheck` set `m6` for `--phase m6` and keep the caller's value for `--phase m7` (SW-49). |
| `TSR_PIVOT=0` | no node ordering by Tesseract Pivot ([PV](../../docs/design/07-graph-scheduler.md) §8). Read once at open; on by default, and off with `TSR_MM_DEFER=0` or an image without MUL_MAT_BEGIN and MUL_MAT_END. `TSR_DEBUG=1` logs the moves and the modeled cycles. The device output is the same either way. |
| `TSR_MM_DEFER=0` | no MUL_MAT is lowered as MUL_MAT_BEGIN and MUL_MAT_END: every MUL_MAT is one record, as on an image without the two kinds. Read once; the device output is the same either way. |
| `TSR_DEBUG=1`, `2` | placement, lowering and launch log on stderr. `2` adds buffer events, every launched record and `supports_op`'s reasons. |
| `TSR_STATS=1`, `TSR_MON=0`, `TSR_DUMP_PLAN`, `TSR_PROF_TRW_MIB`, `TSR_OPL_PROF`, `TSR_OPL_KINDS` | counters at backend free, the live monitor record, the profile dump ([`docs/prof-dbg.md`](../../docs/prof-dbg.md)) |
| `TSR_DUMP_RE=<regex>`, `TSR_DUMP_DIR=<dir>` | after every node whose result name matches, run the pending op list and write the result to `<dir>/<name>.f32`, to find the first node that differs from a CPU run (`cb_eval`). Only the first run of a graph dumps. |
| `TSR_LOAD_THREADS`, `TSR_OPL_CRC=1` | worker threads for weight conversion at load (default the online CPUs, at most 16), and the `graph` launch flag VERIFY_CRC |
| `TSR_COMPUTE_WINDOWS=io`, `TSR_OPL_SPLIT=<n>`, `TSR_FORCE_HOST=1`, `TSR_STAGE_KIB=<n>`, `TSR_GDN_CELL=0`, `TSR_HEAD_SPLIT=0` | test knobs: compute windows in ch0 IO only, cut every op list after n records, every non-WEIGHTS buffer in host memory, IO stage size, no state cell placement (SW-82), whole weight matrices (SW-88). Results do not depend on the last four. |

The other variables (`TSR_SIM_*`, `TSR_TRACE`, `TSR_LOG`, `TSR_LAUNCH_TIMEOUT_MS`, ...) belong to the transport
([`sw/transport/README.md`](../transport/README.md)).

## Build and test

```
third_party/apply-patches.sh
cmake -S . -B build/dev -DCMAKE_BUILD_TYPE=Release
cmake --build build/dev -j8 --target test-backend-ops test-ggml-tesseract test-tsr-backend-units llama-completion
build/dev/bin/test-ggml-tesseract                            # bit-exact against tsr_ref, NMSE against the CPU
build/dev/bin/test-backend-ops test -o MUL_MAT -b Tesseract0 # against the CPU, NMSE 5e-4
ctest --test-dir build/dev -R 'ggml_tesseract|transport_ggml'
```

Configure only through the top-level CMakeLists, which defines `tsr_ref`, `tsr_transport` and `tsr_lower` before
llama.cpp. The device ctests run with the simulator's race detector fatal (`TSR_SIM_RACE=fatal`). They cover the
allocator, the OPL builder and cache, fusion, `supports_op` against the planners, the launcher on a device, the
end-to-end tests in matmul mode (`ggml_tesseract_e2e*`, hosted and `T_MAX` 8 variants), `test-backend-ops` per op
(`ggml_tesseract_backend_ops*`, `tests/m7_tests.py ops`; the single-output MUL_MAT case that fails its NMSE limit on
some random draws is covered bit-exactly by `_scalar_nmse`, [`llama-pin.md`](../../docs/llama-pin.md)), KV buffers and
attention (`_kvbuf`, `_attn*`, `_kvstate`), multi-token prediction (`_mtp_*`, `tsr-mtp`), mixture of experts
(`_moe*`), and `llama-server` on the tiny Qwen2 model (`_server_tiny`: every device text must equal the CPU's and the
device runs one launch per `graph_compute`). `TSR_TEST_DUMP=<file>` makes `test-ggml-tesseract` append every device
output it reads, so two builds can be compared byte for byte.

`tests/gen_tiny_models.py` writes tiny GGUF models at build time (target `tsr_tiny_models`, into
`build/<tree>/models/tiny/`): Qwen2 (Q8_0 and Q4_K), Gemma 4, and two Gated DeltaNet models, with random weights from
numpy's legacy `RandomState` and valid tokenizer metadata (SW-21). The graph tests (`ggml_tesseract_graph_*`,
`tests/m7_tests.py graph`) run klcheck on them in full mode and assert one Tesseract split in every decode graph, one
launch per decode, the launch count of `TSR_STATS`, the embd-path identity and klcheck's KL criterion
([`docs/klcheck.md`](../../docs/klcheck.md)). `ggml_tesseract_gdn_logits_*` compares the logits of every step against
the CPU, because klcheck's mean KL on a random tiny model does not see a convolution that returns zeros (SW-71).

`llama-server` runs on the backend with the full-mode settings (`-ngl 99 -fa on -ctk q8_0 -ctv q8_0 -np 4 -kvu`, no
`--context-shift`; [`docs/getting-started.md`](../../docs/getting-started.md#serving)).
