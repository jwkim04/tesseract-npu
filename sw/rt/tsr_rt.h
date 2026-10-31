// tsr_rt.h — Tesseract runtime without ggml: weight upload and y = x · Wᵀ on the device.
//
// The host stack for frameworks other than llama.cpp (the PyTorch backend binds this API). It
// runs on tsr_transport (TSR_DEVICE = sim | mock | vkernel | xrt, as the ggml backend), boots the
// firmware image ($TSR_IMAGE or the build's) when it opens the device, and shares the ggml backend's
// code (tsr_lower: shape checks, the range allocator, the symmetric weight scatter, the OPL builder,
// OPL cache and `graph` launcher of tsr-opl.h). A linear call is one launch of the firmware's
// `graph` kernel per IO tile (docs/design/06-software-verification.md §9.4): an op list with one
// MUL_MAT record per weight part (K parts as further records with ACC_DST), which the kernel runs
// per T tile as
//
//     LD x tile -> QACT (A32 / A256) -> MVM (Q8_0 / Q4_K) -> ST y tile
//
// Numerics are the llama.cpp path's (docs/spec.md "Numeric rules", "Load-time conversion"):
//   - weights given as f32 / f16 / bf16 become Q8_0 at upload with ggml's quantize_row_q8_0_ref
//     (f16 / bf16 widened exactly with ggml's to_float first), the function the ggml backend uses
//     (sw/ggml-tesseract/tsr-weights.cpp, compiled into this library; ggml-base provides it);
//   - Q8_0 and Q4_K blocks (GGUF / ggml layout) are stored as they are;
//   - activations are quantized on the device by QACT; results are f32.
//
// Device memory (spec "Per-channel DRAM map", EX §11): OPL (ch0) holds the cached op lists, IO (ch0) the x / y
// tiles of one linear call, WEIGHT (every channel) the weights with symmetric allocation (one W_OFF
// for all channels). AUX is not used.
//
// Shapes. K must be a multiple of the weight block (32 for Q8_0, 256 for Q4_K); N >= 1. A weight
// is stored as a grid of parts so that any N, K and T can be lowered:
//   - N parts (row ranges): automatically when x + QA + y of T_MAX tokens (or of one token, if
//     T_MAX tokens can never fit) would not fit the SPM; each part is its own symmetric
//     allocation and its own MVM. Results are bit-identical to an unsplit weight (rows are
//     independent).
//   - K parts (column ranges): automatically when K > K_MAX. The parts are accumulated on the
//     device with MVM.ACC (records with ACC_DST): y = ((y_0 + y_1) + y_2) ..., each y_j the spec's
//     row result over part j (16 partials + tree), each + one f32 add with the fp rules. This
//     differs from an unsplit dot product, so linear_ref() and weight_info.k_part describe it
//     exactly.
//   - T: the firmware tiles each record by min(T_MAX, its scratch ring) tokens (plan.hpp); a call
//     whose x and y exceed the IO area runs as several launches.
// Op lists are cached by content in the OPL region (an LRU), so a repeated call (decode) writes
// only x, launches and reads y.
// weight_options can force smaller parts (tests).
//
// Thread safety: every member function takes one device mutex; calls are serialized.
//
// Errors are thrown as tsr::rt::error (a std::runtime_error) with a message; invalid arguments as
// std::invalid_argument. A failed launch reports the DONE status, the failing record (N / K part)
// and the first error's code and ERR_INFO.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>

#include "tsr_isa.h"

namespace tsr::rt {

class error : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Host types a weight can be uploaded from. Row-major [N, K], any row stride (bytes).
enum class dtype : uint32_t {
    f32  = 0,   // float32                               -> Q8_0
    f16  = 1,   // IEEE binary16 bits (uint16)           -> f32 (exact) -> Q8_0
    bf16 = 2,   // bfloat16 bits (uint16)                -> f32 (exact) -> Q8_0
    q8_0 = 3,   // ggml block_q8_0, 34 B per 32 weights  -> Q8_0 as is
    q4_k = 4,   // ggml block_q4_K, 144 B per 256        -> Q4_K as is
};

const char * dtype_name(dtype t);
// Device weight format for a host type: TSR_FMT_Q4_K for q4_k, TSR_FMT_Q8_0 otherwise.
uint32_t     dtype_device_format(dtype t);
// Bytes of one packed row of K elements of t (blocks for q8_0 / q4_k); 0 if K is not a multiple of
// the block size or t is unknown.
size_t       dtype_row_bytes(dtype t, uint32_t K);

struct open_options {
    const char * kind = nullptr;   // transport kind; nullptr or "" -> getenv("TSR_DEVICE"), default "sim"
    uint64_t io_bytes = 0;         // use at most this many bytes of the 256 MiB IO region for x / y (0: all;
                                   // test knob that forces several launches per call)
    bool read_perf = true;         // read PERF_CYCLES / PERF_MVM_CYCLES / PERF_CH_RD_BEATS after every
                                   // launch for counters (2 x (2 + NUM_CH) CSR reads per launch)
};

struct weight_options {
    uint32_t max_part_rows = 0;    // 0: automatic N split; else at most this many rows per N part
    uint32_t max_part_k = 0;       // 0: automatic K split (K_MAX); else at most this K per K part
                                   // (a multiple of the block size, 32 or 256)
};

struct weight_info {
    uint32_t N, K;
    dtype    src;                  // host type given at upload
    uint32_t fmt;                  // TSR_FMT_Q8_0 / TSR_FMT_Q4_K
    uint32_t n_parts;              // N parts: rows [i * n_part, min((i + 1) * n_part, N))
    uint32_t n_part;
    uint32_t k_parts;              // K parts: columns [j * k_part, min((j + 1) * k_part, K)), ACC-summed in order
    uint32_t k_part;
    uint64_t dev_bytes;            // device DRAM reserved, all channels (per channel x NUM_CH)
};

struct counters {
    uint64_t linears = 0;          // linear() calls with T >= 1
    uint64_t tokens = 0;           // sum of their T
    // launches of `graph` and op-list cache hits / misses.
    uint64_t execs = 0;            // graph launches
    uint64_t prog_hits = 0;        // op lists found in the OPL cache
    uint64_t prog_misses = 0;      // op lists built and written to the OPL region
    uint64_t perf_cycles = 0;      // sum of PERF_CYCLES deltas over the launches (sim: its cycle model)
    uint64_t perf_mvm_cycles = 0;  // sum of PERF_MVM_CYCLES deltas
    uint64_t dev_read_bytes = 0;   // sum over channels of PERF_CH_RD_BEATS deltas x 64 (weights, LD)
    uint64_t h2d_bytes = 0;        // host -> device bytes, every write (weights, x, op lists)
    uint64_t h2d_weight_bytes = 0; //   of which weight uploads
    uint64_t d2h_bytes = 0;        // device -> host bytes (y, read-backs)
};

struct memory_info {
    uint64_t weight_total;         // WEIGHT heap per channel (bytes)
    uint64_t weight_free;          // free per channel
    uint64_t weight_largest_free;  // largest free range per channel
    uint32_t n_weights;
    size_t   prog_cached;          // op lists in the OPL cache
    uint64_t prog_cmd_bytes;       // OPL region bytes they occupy
};

using weight_handle = uint64_t;    // 0 is never a valid handle

class device {
public:
    // Open the transport (TSR_DEVICE selection as the ggml backend), read its configuration and
    // boot the firmware ($TSR_IMAGE or the build's image; warm if it is already loaded).
    explicit device(const open_options & o = open_options());
    ~device();                     // closes the transport; every weight is released
    device(const device &) = delete;
    device & operator=(const device &) = delete;

    const tsr_config & config() const;   // NUM_CH, T_MAX, SPM_BYTES, K_MAX, CH_BYTES, ...
    const char * transport() const;      // "sim", "mock", "vkernel", "xrt", ...
    const char * description() const;    // one line: transport and configuration

    // nullptr if a weight of this type and shape can be uploaded, else the reason (static string).
    const char * unsupported_reason(dtype t, uint32_t N, uint32_t K, const weight_options & o = weight_options()) const;

    // Upload W [N, K] (row r at data + r * row_stride_bytes; 0 = packed rows, dtype_row_bytes).
    // f32 / f16 / bf16 rows may have any alignment. Returns a handle (never 0).
    weight_handle upload(dtype t, const void * data, uint32_t N, uint32_t K, size_t row_stride_bytes = 0,
                         const weight_options & o = weight_options());
    void          free_weight(weight_handle w);  // releases its device memory; unknown handle -> invalid_argument
    weight_info   info(weight_handle w) const;

    // Rows [r0, r1) as stored on the device, in ggml blocks of the device format (Q8_0: K/32 x 34 B,
    // Q4_K: K/256 x 144 B per row, rows packed at dst). For f32 / f16 / bf16 weights these are the
    // bytes quantize_row_q8_0_ref produced; for q8_0 / q4_k the uploaded bytes.
    void read_blocks(weight_handle w, uint32_t r0, uint32_t r1, void * dst) const;

    // y[t, n] = sum_k W[n, k] x[t, k] on the device, for t < T: x row t at x + t * x_stride floats
    // (0 = K), y row t at y + t * y_stride floats (0 = N). Only y[t, 0..N) is written. T = 0: no-op.
    void linear(weight_handle w, const float * x, uint32_t T, float * y, size_t x_stride = 0, size_t y_stride = 0);

    // The same y computed on the host with libtsr_ref (the spec's oracle) from the weight rows read
    // back from the device: tsr_qact per token and K part, tsr_dot_row per row and K part, K parts
    // added in order with tsr_fadd. Bit-identical to linear() on a correct device.
    void linear_ref(weight_handle w, const float * x, uint32_t T, float * y, size_t x_stride = 0, size_t y_stride = 0) const;

    counters    get_counters() const;
    void        reset_counters();
    memory_info memory() const;

    struct impl;
private:
    std::unique_ptr<impl> p_;
};

}  // namespace tsr::rt
