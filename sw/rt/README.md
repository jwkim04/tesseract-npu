# Runtime (`sw/rt`)

`tsr_rt` uploads weight matrices to a Tesseract device and runs `y = x · Wᵀ` on it, without ggml graphs or
llama.cpp. It is the host stack of the PyTorch backend ([`sw/torch`](../torch/README.md), whose extension
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
| `tests/test_tsr_rt.cpp` | `test-tsr-rt`: device against an independent model, bit for bit |
| `CMakeLists.txt` | `tsr_rt` (static, PIC), `tsr_rt_shared` (`libtsr_rt.so`), `test-tsr-rt`, ctests `rt_*` |

Shared with the ggml backend: `tsr_lower` (`sw/ggml-tesseract/lower.cmake`: shape checks, the OPL builder, cache and
`graph` launcher, the allocator, the row repack and scatter and gather) and the load-time conversion
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
                        dev_read_bytes, h2d_bytes, h2d_weight_bytes, d2h_bytes; };      // uint64_t
using weight_handle = uint64_t;

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
};
}
```

Strides of `linear` are in floats (0 = packed), and the weight row stride is in bytes. `try_linear_scale` computes
`scale * (W x)` in the kernel (kind MUL_MAT_SCALE, 0x1C) for a weight of one K part, on an image that has the kind and
`CAPS.MVM_LOCAL_SCALE`. Otherwise it returns false and the caller multiplies on the CPU. Errors throw
`tsr::rt::error` (a `std::runtime_error`) or `std::invalid_argument`. Every member function takes one device mutex.

The counters are `execs` (`graph` launches), `prog_hits` and `prog_misses` (op lists found in or written to the OPL
cache), `perf_cycles` and `perf_mvm_cycles` (sums of the PERF deltas of every launch), `dev_read_bytes`, `h2d_bytes` (of
which `h2d_weight_bytes`) and `d2h_bytes`. `open_options.read_perf = false` skips the PERF CSR reads per launch.

## Environment, build and test

`TSR_DEVICE`, `TSR_IMAGE` and the transport's own variables ([`sw/transport/README.md`](../transport/README.md)).
`TSR_RT_DEBUG=1` logs uploads, calls and launches, and `2` also every record.

```sh
cmake -S . -B build/dev -DCMAKE_BUILD_TYPE=Release
cmake --build build/dev -j8 --target tsr_rt tsr_rt_shared test-tsr-rt
ctest --test-dir build/dev -R '^rt_'
```

The ctests are `rt_sim`, `rt_sim_tmax4` and `rt_sim_tmax8` (`--big`: the N = 130000 head), `rt_mock` (DMA align 64,
chunk 64 KiB) and `rt_mock_tmax8` (3 channels, `T_MAX` 8, align 4096, chunk 12 KiB), all with `TSR_SIM_RACE=fatal`,
and `rt_sim_gdb_embed`, `rt_sim_basis_old` and `rt_sim_no_mulmat` (the embedded debug service starts, and an older
Basis and an image without MUL_MAT are refused). `rt_vkernel_k4t1` (`-DTSR_RT_VKERNEL_TEST=ON`) is meant for a
Verilated kernel and reports "skipped" (exit 77) without a server. What each run checks is listed in
[`docs/torch-backend.md`](../../docs/torch-backend.md) §6.
