# Runtime (`sw/rt`)

`tsr_rt` uploads weight matrices to a Tesseract device and runs `y = x · Wᵀ` on it, without
ggml graphs or llama.cpp. It is the host stack of the PyTorch backend (`torch.compile(...,
backend="tesseract")`: the weight matmuls go to the device, everything else stays on the CPU;
`sw/torch`, whose extension `_tesseract_rt` binds this API). It runs on `tsr_transport`, so
`TSR_DEVICE=sim | mock | vkernel | xrt` select the device as for the ggml backend (`sim` and `mock`
run on the simulator; `vkernel` is not available until P5.3; `xrt` runs the K4 image on the U200 card with the
control core, firmware kind table 0x804f).

A `linear` call is one launch of the firmware's `graph` kernel per IO tile: an op list with one
MUL_MAT record per weight part (K parts as further records with `ACC_DST`), cached by content in the
OPL region. The kernel issues the `LD / QACT / MVM / ST` tile ops. The counters are `execs` (launches) and
`prog_*` (op-list cache hits and misses). Design: [SW](../../docs/design/06-software-verification.md) §9.4.

> **Scope.** Developer documentation of the runtime as built in this tree. Status words: *on the
> simulator* means tested against the simulator kinds (`sim`, `mock`); *on the card* means run on the U200 image
> `tsr_u200_k4` through `xrt` (results: [`docs/torch-backend.md`](../../docs/torch-backend.md) §6.8). The package built on
> top is [`sw/torch`](../torch/README.md); the design record of the PyTorch backend is
> [`docs/torch-backend.md`](../../docs/torch-backend.md).

## Files

| File | Content |
| --- | --- |
| `tsr_rt.h` | the public C++ API (`tsr::rt::device`), the one the torch extension binds |
| `tsr_rt.cpp` | device (transport + firmware boot), weight parts, the op list of one call, counters, reference path |
| `tests/test_tsr_rt.cpp` | `test-tsr-rt`: device vs an independent model, bit for bit |
| `CMakeLists.txt` | `tsr_rt` (static, PIC), `tsr_rt_shared` (`libtsr_rt.so`), `test-tsr-rt`, ctests `rt_*` |

Shared with the ggml backend: `tsr_lower` (`sw/ggml-tesseract/lower.cmake`: `tsr-lower` shape
checks; `tsr-opl` OPL builder, OPL cache and `graph` launcher; `tsr-alloc`; `tsr-wlayout` row repack
and symmetric scatter / gather) and
the load-time conversion `tsr_row_to_dev_blocks` (`sw/ggml-tesseract/tsr-weights.cpp`, compiled into
`tsr_rt`). ggml-base of the pinned llama.cpp provides `quantize_row_q8_0_ref` and the f16 / bf16
`to_float`.

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

Strides of `linear` are in floats (0 = packed); the weight row stride is in bytes. `try_linear_scale` computes
`scale * (W x)` in the kernel (kind MUL_MAT_SCALE, 0x1C) for a weight of one K part, on an image that has the kind and
`CAPS.MVM_LOCAL_SCALE`; otherwise it returns false and the caller multiplies on the CPU. Errors throw
`tsr::rt::error` (a `std::runtime_error`) or `std::invalid_argument`. Every member function takes
one device mutex.

## Numerics

The llama.cpp path's (`docs/spec.md` §7 Numeric rules, §5 *Load-time conversion*): f32 / f16 / bf16 weights become Q8_0 with
`quantize_row_q8_0_ref` (f16 / bf16 widened exactly first), Q8_0 / Q4_K blocks are stored as they
are, activations are quantized by QACT on the device (A32 / A256), results are f32.
libtsr_ref's `tsr_qact` (A32) is not used for weights: measured against `quantize_row_q8_0_ref` it
agrees on 200000 / 200000 ordinary blocks but differs on 9 of 160 blocks with subnormals or a tiny
amax (DAZ / FTZ, `docs/spec.md` §21), and it emits a QA image (f32 scale), not `block_q8_0`.

## Lowering

- **Device memory**: OPL (ch0, 64 MiB) holds the cached op lists (LRU); IO (ch0, 256 MiB) the x
  rows (K·4 B each) and, per N part, the y rows (⌈4n/64⌉ lines each) of one launch; WEIGHT (every
  channel, from 512 MiB) the weights, symmetric allocation, 4 KiB aligned. AUX is unused.
- **The firmware**: the device constructor boots `$TSR_IMAGE` or the build's `tsr-fw.elf` (warm if
  it is already loaded) and finds `graph`.
- **One launch per call** (per IO-sized chunk of T): an op list with, for every N part i and K part
  j, one MUL_MAT record (SW §7.2): x = the IO x rows from column k₀ (stride K/16 lines), y = part
  i's IO rows (stride ⌈4n/64⌉ lines), W_OFF of part (i, j), N = n_i, K = k_j and the chunk's T; the
  firmware tiles T by min(T_MAX, its scratch ring) with `plan.hpp`. Records of one N part are
  ordered by the builder's hazard analysis (each reads and writes the part's y rows); N parts are
  independent. A list above 4096 records would run in pieces.
- **T tiling is the firmware's.** The record carries the whole T of an IO tile and the firmware cuts it into tiles of
  min(T_MAX, ring) tokens. The runtime does not split T by T_MAX. On the K4 image (T_MAX 4) a 32-token call is eight
  tiles in one launch and reads the weights eight times, not 32 (Qwen2.5-0.5B, 8 tokens: 716,870 PERF cycles per
  token at T_MAX 4 against 2,213,080 at T_MAX 1, on the simulator; `docs/torch-backend.md` §6.5).
- **N parts**: a weight is cut into row ranges when x + QA + y of T_MAX tokens (or of one token, if
  T_MAX tokens can never fit) would exceed the SPM, and one token of a part must fit the
  firmware's scratch-ring budget (`plan.hpp` `ring_budget`: the SPM minus the executor's argument,
  OPL and preload areas). With T_MAX ≥ 2 the SPM / T_MAX bound is the smaller one. Each part is its own allocation and record; rows are independent, so results equal an
  unsplit weight bit for bit. Example: N = 130000, K = 896 at T_MAX 8 → 2 parts of 65000.
- **K parts**: when K > K_MAX (32768), the columns are cut into equal parts (block multiples). The
  first record of an N part writes y, the others have ACC_DST (the kernel loads the old y rows and
  runs MVM with ACC): y = ((y₀ + y₁) + y₂)… with one f32 add each (`docs/spec.md` §21, *ACC addition*),
  the in-SPM ACC chain. This is not the unsplit dot product; `linear_ref` and `weight_info.k_part` describe it
  exactly.
- **Op-list cache**: keyed by the list's bytes (formats, shapes, T, x / y handles, W_OFFs). A hit
  writes nothing to the OPL region; a decode call writes x, launches, reads y.

## Counters

`execs` (`graph` launches), `prog_hits` / `prog_misses` (op lists found in / written to the OPL
cache), `perf_cycles` / `perf_mvm_cycles` (sums of the PERF_CYCLES / PERF_MVM_CYCLES deltas of every
launch; PERF does not change while the device is idle), and `dev_read_bytes` (Σ_c PERF_CH_RD_BEATS
deltas × 64: weight stream, LD), `h2d_bytes` (of which `h2d_weight_bytes`; op lists count on a miss)
and `d2h_bytes`. `open_options.read_perf = false` skips the 2 × (2 + NUM_CH) CSR reads per launch.

## Environment

`TSR_DEVICE` (transport, default sim), `TSR_IMAGE` (firmware image) and the transport's own
variables (`TSR_SIM_*`, `TSR_MOCK_*`, `TSR_VKERNEL_*`, `TSR_XRT_*`, `TSR_TRACE`, `TSR_LOG`, the
timeouts; SW §9.2). `TSR_RT_DEBUG=1` logs uploads, calls and launches.

## Build and test

```sh
cmake -S . -B build/dev -DCMAKE_BUILD_TYPE=Release
cmake --build build/dev -j8 --target tsr_rt tsr_rt_shared test-tsr-rt
ctest --test-dir build/dev -R '^rt_'
```

ctests: `rt_sim`, `rt_sim_tmax4` (T_MAX 4, the K4 image's; `--big`) and `rt_sim_tmax8` (`--big`: the N = 130000 head), `rt_mock` (DMA align 64, chunk
64 KiB), `rt_mock_tmax8` (3 channels, T_MAX 8, align 4096, chunk 12 KiB); all run with
`TSR_SIM_RACE=fatal`. `rt_sim_gdb_embed` checks that the embedded debug service starts; `rt_sim_basis_old` and
`rt_sim_no_mulmat` check that an image of an older Basis and one without MUL_MAT are refused. Each run checks 37 shapes (f32 / f16 / bf16 → Q8_0, native Q8_0 / Q4_K;
K 32 … 4864 and 33024 / 33280 (> K_MAX); N 1 … 4864 (130000 with `--big`); T 1 … 20; forced N
and K parts; strided x / y; padded and misaligned weight rows): `read_blocks` equals ggml's blocks
byte for byte, `linear` and `linear_ref` equal the model bit for bit, y padding untouched, NMSE
against an f64 product of the dequantized weight < 1e-3 (measured 0 … 7.9e-5). Plus the op-list
cache and exact host traffic, free / re-upload, unsupported shapes, two threads, an IO area of
64 KiB (T = 40 → 3 launches).

Results, measured on the simulator (branch `torch-basis`, 2026-11-09, firmware image id `99698a44030574f4`): `rt_sim` 43 / 43 cases,
`rt_sim_tmax4` 44 / 44, `rt_sim_tmax8` 44 / 44 (with the N = 130000 head: 2 N parts at T_MAX 8, one at T_MAX 4, T = 9),
`rt_mock` 43 / 43, `rt_mock_tmax8` 43 / 43.

An opt-in `rt_vkernel_k4t1` ctest (`-DTSR_RT_VKERNEL_TEST=ON`) is meant for the Verilated k4t1 kernel
and reports "skipped" (exit 77) until the two-clock vkernel server exists (P5.3).
