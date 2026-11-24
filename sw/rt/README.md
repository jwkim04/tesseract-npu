# Runtime (`sw/rt`)

`tsr_rt` uploads weight matrices to a Tesseract device and runs `y = x · Wᵀ` on it, and runs programs of several
ops (a region of the PyTorch backend's full mode) as op lists, without ggml graphs or llama.cpp. It is the host stack of the PyTorch backend ([`sw/torch`](../torch/README.md), whose extension
`_tesseract_rt` binds this API). It runs on `tsr_transport`, so `TSR_DEVICE=sim | mock | vkernel | xrt` select the
device as for the ggml backend. A `linear` call is one launch of the firmware's `graph` kernel per IO tile: an op
list with one MUL_MAT record per weight part (K parts as further records with `ACC_DST`), cached by content in the
OPL region. How the call is lowered, the device memory layout, the N and K parts, the op-list cache and the numerics
are described in [`docs/torch-backend.md`](../../docs/torch-backend.md) §3. Design:
[SW](../../docs/design/06-software-verification.md) §9.4.

| File | Content |
| --- | --- |
| `tsr_rt.h` | the public C++ API (`tsr::rt::device`), the one the torch extension binds |
| `tsr_rt.cpp` | device (transport and firmware boot), weight parts, the op list of one call, counters, reference path |
| `tsr_rt_graph.cpp`, `tsr_rt_internal.h` | the graph API: validation, the planner per T (placement, records, lists), constants, KV planes, run and reference run |
| `tests/test_tsr_rt.cpp` | `test-tsr-rt`: device against an independent model, bit for bit |
| `tests/test_tsr_rt_graph.cpp` | `test-tsr-rt-graph`: `graph_run` against `graph_run_ref` and the per-kind references, bit for bit, also for cached attention (KV objects, slots, chunks) |
| `CMakeLists.txt` | `tsr_rt` (static, PIC), `tsr_rt_shared` (`libtsr_rt.so`), `test-tsr-rt`, `test-tsr-rt-graph`, ctests `rt_*` |

Shared with the ggml backend: `tsr_lower` (`sw/ggml-tesseract/lower.cmake`: shape checks, the OPL builder, cache and
`graph` launcher, the allocator, the row repack and scatter and gather, and for the graph API the record type
`tsr-rec` and the per-kind encoders `tsr-kinds`, which use no ggml) and the load-time conversion
`tsr_row_to_dev_blocks` (`sw/ggml-tesseract/tsr-weights.cpp`, compiled into `tsr_rt`). ggml-base of the pinned
llama.cpp provides `quantize_row_q8_0_ref` and the f16 and bf16 `to_float`.

## API (tsr_rt.h)

```cpp
namespace tsr::rt {
enum class dtype : uint32_t { f32, f16, bf16, q8_0, q4_k };
struct open_options   { const char * kind = nullptr; uint64_t io_bytes = 0; bool read_perf = true; };
struct weight_options { uint32_t max_part_rows = 0; uint32_t max_part_k = 0; };
struct weight_info    { uint32_t N, K; dtype src; uint32_t fmt, n_parts, n_part, k_parts, k_part; uint64_t dev_bytes; };
struct counters       { linears, tokens, execs, prog_hits, prog_misses, perf_cycles, perf_mvm_cycles,
                        dev_read_bytes, h2d_bytes, h2d_weight_bytes, d2h_bytes, h2d_writes, d2h_reads; };      // uint64_t (and the graph and KV counters below)
using weight_handle = uint64_t;
using kv_handle     = uint64_t;                                   // the planes of a cache (kv_create)

class device {
    explicit device(const open_options & = {});
    const tsr_config & config() const;  const char * transport() const;  const char * description() const;
    const char * unsupported_reason(dtype, uint32_t N, uint32_t K, const weight_options & = {}) const;
    weight_handle upload(dtype, const void * data, uint32_t N, uint32_t K, size_t row_stride_bytes = 0,
                         const weight_options & = {});
    void          free_weight(weight_handle);
    weight_info   info(weight_handle) const;
    void          read_blocks(weight_handle, uint32_t r0, uint32_t r1, void * dst) const;
    void linear    (weight_handle, const float * x, uint32_t T, float * y, size_t x_stride = 0, size_t y_stride = 0);
    bool has_local_scale() const;                                   // image has MUL_MAT_SCALE and CAPS.MVM_LOCAL_SCALE, C = 4
    bool try_linear_scale(weight_handle, const float * x, uint32_t T, float * y, float scale,
                          size_t x_stride = 0, size_t y_stride = 0); // false: nothing transferred, use linear()
    void linear_ref(weight_handle, const float * x, uint32_t T, float * y, size_t x_stride = 0, size_t y_stride = 0) const;
    counters get_counters() const;  void reset_counters();  memory_info memory() const;

    // graphs (full mode of the PyTorch backend)
    const_handle const_upload(const float * data, uint32_t rows, uint32_t n, size_t row_stride_floats = 0);
    void         const_free(const_handle);
    const char * graph_unsupported(const graph_desc &);            // nullptr, or why the device cannot run the program
    graph_handle graph_create(const graph_desc &);
    void         graph_free(graph_handle);
    graph_shape  graph_shape_of(graph_handle) const;
    void         graph_run(graph_handle, uint32_t T, const graph_inputs &, const graph_outputs &, const graph_state & = {});
    void         graph_run_ref(graph_handle, uint32_t T, const graph_inputs &, const graph_outputs &, const graph_state & = {});
    graph_plan_info graph_info(graph_handle, uint32_t T);

    // KV objects (the planes of a cached attention, which outlive graphs and calls)
    kv_handle    kv_create(uint32_t E, uint32_t cells);
    void         kv_free(kv_handle);
    kv_info      kv_info_of(kv_handle) const;
    void         kv_store(kv_handle, uint32_t cell0, uint32_t n, const float * k, const float * v, size_t k_stride = 0, size_t v_stride = 0);
    void         kv_read(kv_handle, uint32_t cell0, uint32_t n, void * k_blocks, void * v_blocks) const;
};
}
```

Strides of `linear` are in floats (0 = packed), and the weight row stride is in bytes. `try_linear_scale` computes
`scale * (W x)` in the kernel (kind MUL_MAT_SCALE, 0x1C) for a weight of one K part, on an image that has the kind and
`CAPS.MVM_LOCAL_SCALE`. Otherwise it returns false and the caller multiplies on the CPU. Errors throw
`tsr::rt::error` (a `std::runtime_error`) or `std::invalid_argument`. Every member function takes one device mutex.

A graph is a straight-line program (`graph_desc`) over float32 `[T, n]` values: inputs given at every run,
temporaries on the device (some of them outputs), constants uploaded with `const_upload` (AUX), positions and an
attention mask. The ops are `mul_mat` (with an addend), `add`, `mul`, `scale`, `rms_norm`, `glu`, `unary`, `rope`,
`attn` and `cpy`. `graph_unsupported` says why the device cannot run a program without allocating anything.
`graph_create` allocates the KV planes of the attention ops that have no KV slot (two per op, zero-filled, at the top of
the WEIGHT heap); the weights and constants must stay until `graph_free`. `graph_run(h, T, in, out)` runs T rows (`1 <= T <= kv_size`). A
program is planned for each T (and mask shape) on first use, and the plan is cached (16 per graph). The
values of a run are placed in the IO region by liveness (for a graph with KV slots the out values first, as one
block, unless that plan does not fit the IO region), the records are the `tsr-kinds` encoders' through the ggml
backend's builder, and a list that reaches 4096 records is cut, so a run is one launch per list. The inputs are
written in address order, and writes whose byte ranges touch go out as one transfer (a write of 64 KiB or more on its
own). Out values whose byte ranges touch are read in one transfer. A gap is never written and never read. A T above
`kv_size` or above 680 with an attention op that has no KV slot, a run that does not fit the IO region and a mask value other than 0
and -inf throw an error whose text contains "does not fit". The planner checks its own placement: a value that
is read after its slot was released is an error ("value N is read after its slot was released"). The PyTorch
backend runs a region's saved aten subgraph on the CPU when `graph_create` refuses a program or a call does not
fit, and turns the mask values at most -1e9 into -inf before the run. `graph_run_ref` runs the same lists on the host
(`tsr_opl_ref`, `sw/ctrl/ref/opl_ref.hpp`) on an image of the constants and weights read back from the device, and
its result equals `graph_run`'s bit for bit on a correct device. How the PyTorch backend uses this is in
[`docs/torch-backend.md`](../../docs/torch-backend.md) §3.3 and §8.

Cached attention keeps the K and V of earlier calls on the device. `kv_create(E, cells)` makes a KV object: a K and a V
plane of `cells` cells of E values each (E a multiple of 64), zero-filled in every channel, from the top of the WEIGHT
heap. It lives until `kv_free`, across graphs and runs. `kv_info_of` gives its geometry, `kv_store` writes float32 rows
into cells, quantized by the device as an attention op stores its rows, and `kv_read` returns cells as ggml `block_q8_0`
rows. A full heap or a geometry that cannot be placed is an error whose text contains "does not fit". An E or a cell
count that no plane can have, and a range beyond the cells (also for no rows), is `std::invalid_argument`.

A graph declares KV slots in `graph_desc::kv_slots` (the cells of each). An `attn` op with `kv = s` reads and writes
the slot s instead of owning planes, one op per slot, with E = `n_head_kv * d` equal to the E of the bound object and
`kv_slots[s]` equal to its cells. The mask value of such an op has `cols` = the cells, so the mask is `[T, cells]`. A run
binds one KV object to each slot through `graph_state` (`kv`, one distinct handle per slot, and `pos0`). Row t of the run
is cell `pos0 + t`. Each cached op stores the K and V rows of the run into the cells `[pos0, pos0 + T)` and attends over
the planes: with the causal mask (mask -1) row t sees the cells `0 .. pos0 + t`, and with a given mask the cells it
shows. `pos0 + T` above the cells is an error whose text contains "does not fit", and so is a given mask that shows, for
a row, a cell that the call does not write. A graph without slots ignores `pos0` and takes no handles. `graph_run_ref`
also takes the state. Its image holds the cells `[0, pos0 + T)` of the planes, so it can run after `graph_run` of the same
step and does not change the device's planes.

A graph whose attention ops all have a slot runs in consecutive chunks of rows when T is above 680 or the plan of T does
not fit the IO region. A chunk is the largest that plans, and a chunk of one row that does not fit is the error. The
inputs, positions and mask rows of the chunk's range are used, and its outputs go to the same rows of the results.
`graph_info(h, T)` of such a graph answers with the plan the first chunk of a run of T rows starts with, and its `T` is the
rows of that chunk, so a smaller `T` than asked means the run is chunked. `pos0` plays no part in it.

The counters are `execs` (`graph` launches), `prog_hits` and `prog_misses` (op lists found in or written to the OPL
cache), `perf_cycles` and `perf_mvm_cycles` (sums of the PERF deltas of every launch), `dev_read_bytes`, `h2d_bytes` (of
which `h2d_weight_bytes`) and `d2h_bytes`, and `h2d_writes` and `d2h_reads` (the host to device and device to host
transfers: calls of the transport with a size above 0). The graph counters are `graphs`, `graph_runs`, `graph_launches`,
`graph_records`, `graph_plan_hits`, `graph_plan_misses` and `consts`. A run in chunks counts once in `graph_runs`, adds
the lists of every chunk to `graph_launches`, and looks up the plan once per chunk. The KV counters are `kv_creates`,
`kv_stores` (calls with at least one row) and `kv_store_rows`. `open_options.read_perf = false` skips the PERF CSR reads per launch.

## Environment, build and test

`TSR_DEVICE`, `TSR_IMAGE` and the transport's own variables ([`sw/transport/README.md`](../transport/README.md)).
`TSR_RT_DEBUG=1` logs uploads, calls and launches, and `2` also every record.

```sh
cmake -S . -B build/dev -DCMAKE_BUILD_TYPE=Release
cmake --build build/dev -j8 --target tsr_rt tsr_rt_shared test-tsr-rt
ctest --test-dir build/dev -R '^rt_'
```

The graph tests are `rt_graph_sim`, `rt_graph_sim_tmax4`, `rt_graph_sim_tmax8` and `rt_graph_mock`
(`test-tsr-rt-graph`, which also covers the KV objects, cached attention with KV slots and runs in chunks). The other ctests are `rt_sim`, `rt_sim_tmax4` and `rt_sim_tmax8` (`--big`: the N = 130000 head), `rt_mock` (DMA align 64,
chunk 64 KiB) and `rt_mock_tmax8` (3 channels, `T_MAX` 8, align 4096, chunk 12 KiB), all with `TSR_SIM_RACE=fatal`,
and `rt_sim_gdb_embed`, `rt_sim_basis_old` and `rt_sim_no_mulmat` (the embedded debug service starts, and an older
Basis and an image without MUL_MAT are refused). `rt_vkernel_k4t1` (`-DTSR_RT_VKERNEL_TEST=ON`) is meant for a
Verilated kernel and reports "skipped" (exit 77) without a server. What each run checks is listed in
[`docs/torch-backend.md`](../../docs/torch-backend.md) §6.
