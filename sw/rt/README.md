# sw/rt — tsr_rt, the ggml-free Tesseract runtime

`tsr_rt` uploads weight matrices to a Tesseract device and runs `y = x · Wᵀ` on it, without
ggml graphs or llama.cpp. It is the host stack of the PyTorch backend (`torch.compile(...,
backend="tesseract")`: the weight matmuls go to the device, everything else stays on the CPU;
`sw/torch`, whose extension `_tesseract_rt` binds this API). It runs on `tsr_transport`, so
`TSR_DEVICE=sim | mock | vkernel | xrt` select the device as for the ggml backend.

## Files

| File | Content |
| --- | --- |
| `tsr_rt.h` | the public C++ API (`tsr::rt::device`), the one the torch extension binds |
| `tsr_rt.cpp` | device, weight parts, lowering of one call, program cache, counters, reference path |
| `tests/test_tsr_rt.cpp` | `test-tsr-rt`: device vs an independent model, bit for bit |
| `CMakeLists.txt` | `tsr_rt` (static, PIC), `tsr_rt_shared` (`libtsr_rt.so`), `test-tsr-rt`, ctests `rt_*` |

Shared with the ggml backend: `tsr_lower` (`sw/ggml-tesseract/lower.cmake`: `tsr-lower` tiling,
emission, disassembly; `tsr-alloc`; `tsr-wlayout` row repack and symmetric scatter / gather) and
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
    void linear_ref(weight_handle, const float * x, uint32_t T, float * y, size_t x_stride = 0, size_t y_stride = 0) const;
    counters get_counters() const;  void reset_counters();  memory_info memory() const;
};
}
```

Strides of `linear` are in floats (0 = packed); the weight row stride is in bytes. Errors throw
`tsr::rt::error` (a `std::runtime_error`) or `std::invalid_argument`. Every member function takes
one device mutex.

## Numerics

The llama.cpp path's (spec 수치 규칙, 로드 시 변환): f32 / f16 / bf16 weights become Q8_0 with
`quantize_row_q8_0_ref` (f16 / bf16 widened exactly first), Q8_0 / Q4_K blocks are stored as they
are, activations are quantized by QACT on the device (A32 / A256), results are f32.
libtsr_ref's `tsr_qact` (A32) is not used for weights: measured against `quantize_row_q8_0_ref` it
agrees on 200000 / 200000 ordinary blocks but differs on 9 of 160 blocks with subnormals or a tiny
amax (DAZ / FTZ of spec 구현 명확화), and it emits a QA image (f32 scale), not `block_q8_0`.

## Lowering

- **Device memory**: CMD (ch0, 64 MiB) holds the cached programs, one slot each; IO (ch0, 256 MiB)
  the x rows (K·4 B each) and, per N part, the y rows (⌈4n/64⌉ lines each) of one program; WEIGHT
  (every channel, from 512 MiB) the weights, symmetric allocation, 4 KiB aligned. AUX is unused.
- **One program per call** (per IO-sized chunk of T): for every N part, `tsr_mm_emit` (the ggml
  backend's emitter) with T tiles of min(T_MAX, SPM capacity) tokens: `LD / QACT / MVM / ST`.
- **N parts**: a weight is cut into row ranges when x + QA + y of T_MAX tokens (or of one token, if
  T_MAX tokens can never fit) would exceed the SPM. Each part is its own allocation and MVM; rows
  are independent, so results equal an unsplit weight bit for bit. Example: N = 130000, K = 896 at
  T_MAX 8 → 2 parts of 65000 (unsplit, x + QA + y of 7 tokens fit the 65536-line SPM; split, 15, so a
  T = 8 tile streams each row once).
- **K parts**: when K > K_MAX (32768), the columns are cut into equal parts (block multiples). The
  program LDs the x tile once and, per part, runs QACT on its columns (`src_stride` = the full row)
  and MVM, with `ACC` after the first: y = ((y₀ + y₁) + y₂)… with one f32 add each (spec ACC). This
  is not the unsplit dot product; `linear_ref` and `weight_info.k_part` describe it exactly.
- **Program cache**: key = format, K, T, x and y beats, and every part's (n, k₀, k, W_OFF). A hit
  writes nothing to CMD; a decode call writes x, runs, reads y. Programs are at most 4 MiB
  (131071 instructions; T is chunked to stay below), at most 65536 are kept, LRU eviction when the
  CMD region is full.

## Counters

`execs` (programs run), `prog_hits` / `prog_misses`, `perf_cycles` / `perf_mvm_cycles` (sums of
the PERF_CYCLES / PERF_MVM_CYCLES deltas of every run; PERF does not change between runs), and
`dev_read_bytes` (Σ_c PERF_CH_RD_BEATS deltas × 64: weight stream, LD, instruction fetch), `h2d_bytes`
(of which `h2d_weight_bytes`) and `d2h_bytes`. `open_options.read_perf = false` skips the
2 × (2 + NUM_CH) CSR reads per run.

## Environment

`TSR_DEVICE` (transport, default sim) and the transport's own variables (`TSR_SIM_*`, `TSR_MOCK_*`,
`TSR_VKERNEL_*`, `TSR_XRT_*`, `TSR_TRACE` for sim). `TSR_RT_DEBUG=1` logs uploads and emitted
programs, `2` adds a disassembly of every new program.

## Build and test

```
cmake -S . -B build/rt -DCMAKE_BUILD_TYPE=Release
cmake --build build/rt -j4 --target tsr_rt tsr_rt_shared test-tsr-rt
cd build/rt && ctest -R '^rt_'
tb/platform/vkernel/vkernel.py run --config k4t1 -- build/rt/bin/test-tsr-rt --quick    # RTL smoke
```

ctests: `rt_sim`, `rt_sim_tmax8` (`--big`: the N = 130000 head), `rt_mock` (DMA align 64, chunk
64 KiB), `rt_mock_tmax8` (3 channels, T_MAX 8, align 4096, chunk 12 KiB); `rt_vkernel_k4t1` with
`-DTSR_RT_VKERNEL_TEST=ON`. Each run checks 37 shapes (f32 / f16 / bf16 → Q8_0, native Q8_0 / Q4_K;
K 32 … 4864 and 33024 / 33280 (> K_MAX); N 1 … 4864 (130000 with `--big`); T 1 … 20; forced N
and K parts; strided x / y; padded and misaligned weight rows): `read_blocks` equals ggml's blocks
byte for byte, `linear` and `linear_ref` equal the model bit for bit, y padding untouched, NMSE
against an f64 product of the dequantized weight < 1e-3 (measured 0 … 7.9e-5). Plus program cache
and exact host traffic, free / re-upload, unsupported shapes, two threads, an IO area of 64 KiB
(T = 40 → 3 programs).

Results 2026-10-24 (`build/rt`, Release): all 43 cases OK on `rt_sim`, `rt_mock`, `rt_mock_tmax8`,
44 on `rt_sim_tmax8` (with the N = 130000 head: 2 N parts, T = 9). On the Verilated kernel
(`TSR_DEVICE=vkernel`, servers built from the M3 RTL at 04:37 the same day, regime fast): k4t1
(4 ch, T_MAX 1) all 43 cases, 67 runs, 0 findings, 0 reads of unwritten memory, 9 s wall; k4 (4 ch,
T_MAX 8) all 44 cases with `--big`, 68 runs, 0 findings, 0 reads of unwritten memory, 112 s wall.
Mutation check (scratch builds of tsr_rt.cpp): a wrong QACT column offset in K parts, a dropped
ACC, a shifted y gather of N parts, a K part converted from column 0, and x packed with the wrong
stride each fail the test (6 to 24 failed checks).
