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
// for all channels). linear does not use AUX; the graph API keeps its constants there (below).
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
// Graphs (whole-graph lowering, docs: the PyTorch backend's full mode): a straight-line program of ops over f32 [T, n]
// values (graph_desc below) is planned per number of rows T into op lists of the firmware's kinds (MUL_MAT, RMS_NORM, ROPE,
// SET_ROWS_KV + FLASH_ATTN, ADD, MUL, SCALE, GLU, UNARY, CPY), run as one launch per list (normally one), and read back. The
// activations stay on the device between ops. Memory: the IO region holds the values of a run (placed by liveness), the AUX
// region the constants (const_upload), the WEIGHT heap the weights and, from the top, the KV planes of each attn op (two planes
// per op, zero-filled at graph_create). The plan of a T is cached per graph. graph_run_ref runs the same lists on the host
// with the per-kind references (sw/ctrl/ref/opl_ref.hpp) on an image read from the device: its result equals graph_run's bit
// for bit on a correct device.
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
#include <vector>

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
    uint64_t h2d_writes = 0;       // host -> device transfers (calls of write with a size above 0)
    uint64_t d2h_reads = 0;        // device -> host transfers (calls of read with a size above 0)
    // graphs
    uint64_t graphs = 0;           // graph_create calls
    uint64_t graph_runs = 0;       // graph_run calls (a run in several chunks counts once)
    uint64_t graph_launches = 0;   // op lists launched by graph_run (a chunked run adds the lists of every chunk)
    uint64_t graph_records = 0;    // records of those lists (JOINs included)
    uint64_t graph_plan_hits = 0;  // graph_run / graph_run_ref found the plan of its T (mask shape, KV plane offsets) in the graph's cache; one lookup per chunk
    uint64_t graph_plan_misses = 0;// ... had to plan it
    uint64_t consts = 0;           // const_upload calls
    // KV objects (kv_create ...)
    uint64_t kv_creates = 0;       // kv_create calls
    uint64_t kv_stores = 0;        // kv_store calls with n >= 1
    uint64_t kv_store_rows = 0;    // rows given to kv_store
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
using const_handle  = uint64_t;    // f32 rows in the AUX region (const_upload); 0 is never a valid handle
using graph_handle  = uint64_t;    // 0 is never a valid handle
using kv_handle     = uint64_t;    // a K and a V plane that outlive graphs and calls (kv_create); 0 is never a valid handle

// ---- the graph program (the contract with the PyTorch backend: docs/torch-backend.md) ----
// Values are f32 [T, n] (T rows, one per token); T is the one polymorphic number of a graph. An op reads values and writes
// its `dst`. Every dst is a tmp value written by exactly one op and is never an operand of that op; a value is written before
// it is read. A value id is the index in graph_desc::values; -1 is "none".
enum class gv : uint32_t {
    in,       // given at every run: f32 [T, n]
    tmp,      // on the device; out: also returned by graph_run
    konst,    // `rows` rows of n f32 uploaded with const_upload (norm weight, bias row: rows 1; RoPE table: n_dims / 2 values)
    pos,      // I32 positions [T], given at every run (rope)
    mask,     // attention mask [T, T] (cols = 0) or [T, cols], given at every run: f32, 0 = visible, -inf = masked (attn)
};
enum class gop : uint32_t { mul_mat, add, mul, scale, rms_norm, glu, unary, rope, attn, cpy };
const char * gop_name(gop op);
const char * record_kind_name(uint32_t kind);   // name of an op-list record kind (graph_plan_info::by_kind), "?" if unknown

struct graph_value {
    gv           kind = gv::tmp;
    uint32_t     n = 0;            // in, tmp, konst
    uint32_t     rows = 1;         // konst
    bool         out = false;      // tmp
    const_handle handle = 0;       // konst (0 only for graph_unsupported, which does not touch the device's constants)
    uint32_t     cols = 0;         // mask: 0 = [T, T] (the attn op's own planes); else [T, cols], cols = the cells of the KV slot of the
                                   //       attn ops that read it (cached attention)
};

// The fields an op uses (the others stay at their defaults):
//   mul_mat  w, x, dst, addend (a [T, N] value, or a one-row konst: a bias)   y = x W^T (+ addend)
//   add, mul a, b, dst        b: a [T, n] value or a one-row konst
//   scale    x, dst, s, sb    y = x * s, + sb when sb is not +0.0
//   rms_norm x, dst, weight (a one-row konst of n values, or -1), eps
//   glu      gate, up, dst, variant (0 SWIGLU, 2 GEGLU: exact tanh form)
//   unary    x, dst, func (0 TANH, 2 SILU, 3 EXP, 4 GELU: exact tanh form, 5 SIGMOID)
//   rope     x, dst, n_heads, head_dim, n_dims, table (a one-row konst of n_dims / 2 angles A_i), pos, factor
//            NEOX pairing (i, i + n_dims / 2); the angle of position p and pair i is FM(f32(p), A_i)
//   attn     q, k, v, dst, n_head, n_head_kv, d, scale, mask (a mask value, or -1: causal); kv
//            kv = -1: the op's own planes (allocated by graph_create): the K and V rows of row t go to cell t and the op attends over
//                     the T cells of the run (n_kv = T); the mask value, if any, has cols = 0
//            kv = s >= 0 (cached attention): the op reads and writes the KV slot s: graph_desc::kv_slots[s] cells, bound to a kv_handle
//                     at every run (graph_state). The rows of row t go to cell pos0 + t and the op attends over the cells of the
//                     planes (see graph_run). The op owns no planes. One op per slot. E = n_head_kv * d must equal the bound
//                     object's E and kv_slots[s] its cells. The mask value, if any, has cols = kv_slots[s]. A graph whose attn ops
//                     all have a slot is run in chunks of rows when it must be (graph_run).
//   cpy      x, dst
struct graph_op {
    gop          op = gop::cpy;
    int32_t      tag = -1;         // the caller's number for the op (the aten node), named in errors
    weight_handle w = 0;
    int32_t      x = -1, a = -1, b = -1, dst = -1, addend = -1, weight = -1, gate = -1, up = -1, table = -1, pos = -1;
    int32_t      q = -1, k = -1, v = -1, mask = -1;
    float        s = 1.0f, sb = 0.0f, eps = 0.0f, factor = 1.0f, scale = 1.0f;
    uint32_t     variant = 0, func = 0, n_heads = 0, head_dim = 0, n_dims = 0, n_head = 0, n_head_kv = 0, d = 0;
    int32_t      kv = -1;          // attn: the KV slot, -1 none
};

struct graph_desc {
    std::vector<graph_value> values;
    std::vector<graph_op>    ops;
    uint32_t                 kv_size = 0;   // cells of every KV plane of the attn ops without a slot: the largest T of a run (0 is
                                            // allowed when every attn op has a slot)
    std::vector<uint32_t>    kv_slots;      // cells of each KV slot (cached attention); a slot is used by exactly one attn op
};

// Limits of graph_run (a T beyond them is refused with an error whose text contains "does not fit"; a graph whose attn ops all have
// a KV slot is run in chunks instead of being refused for T above graph_max_attn_tokens).
constexpr uint32_t graph_max_attn_tokens = 680;   // tokens of one attn op: its mask summary and cell indices in the 256-line preload area
constexpr uint32_t graph_plan_cache = 16;         // plans (one per T and mask shape) kept per graph, least recently used out

// One argument of graph_run, in the order of the graph's in, pos and mask values (graph_shape::arg_kind):
//   in    data: float*, T rows of n values, row r at data + r * stride (stride 0: n)
//   pos   data: int32_t*, T values (stride unused)
//   mask  data: float*, T rows of T values (cols values for a mask with cols != 0), row r at data + r * stride (stride 0: T or cols);
//         every value 0 or -inf
struct graph_arg {
    const void * data = nullptr;
    size_t       stride = 0;
};
struct graph_inputs {
    std::vector<graph_arg> args;
};
// One result per tmp value with out = true, in value order: T rows of n floats, row r at data + r * stride (stride 0: n).
struct graph_out {
    float * data = nullptr;
    size_t  stride = 0;
};
struct graph_outputs {
    std::vector<graph_out> outs;
};

// What a graph takes and gives (to size the arguments).
struct graph_shape {
    std::vector<gv>       arg_kind;   // of each argument: gv::in, gv::pos or gv::mask
    std::vector<uint32_t> arg_n;      // in: n; mask: its cols (0: [T, T]); pos: 0
    std::vector<uint32_t> out_n;      // n of each out value
    uint32_t              kv_size = 0;
    uint64_t              kv_bytes = 0;   // KV planes of the graph's own (attn ops without a slot), per channel
    std::vector<uint32_t> kv_slots;       // cells of each KV slot (empty: the graph has none)
};

// The KV object bound to each slot of a run and the first cell of its rows.
struct graph_state {
    std::vector<kv_handle> kv;            // one handle per slot of the graph (empty for a graph without slots)
    uint32_t               pos0 = 0;      // row t of the run is cell pos0 + t, for every slot
};

// A KV object (kv_info_of): E values per cell, `cells` cells in each of the K and V plane, `bytes` per channel for both planes,
// k_off / v_off the byte offsets of the planes (the same in every channel).
struct kv_info {
    uint32_t E = 0, cells = 0;
    uint64_t bytes = 0;
    uint64_t k_off = 0, v_off = 0;
};

// The plan of one T (graph_info).
struct graph_plan_info {
    uint32_t T = 0;
    uint32_t lists = 0;               // op lists = launches of a run
    uint32_t records = 0;             // records of all lists (JOINs included)
    uint32_t joins = 0;
    uint32_t mul_mat_begin = 0;       // MUL_MAT_BEGIN records (deferred MUL_MAT)
    uint32_t by_kind[64] = {};        // records per kind (tsr::opl kind numbers)
    uint64_t io_bytes = 0;            // IO region bytes the run uses
    uint32_t preload_lines = 0;       // preload lines of the fullest list
    uint32_t fa_new = 0;              // FLASH_ATTN records whose flags word has FA_NEW for all T cells their SET_ROWS_KV records wrote
};

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
    uint32_t pcap() const;               // AT PCAP of the device (CFG3), for the attention planners

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

    // Optional scaled result. False means no input or output was transferred.
    // The original linear and host scale remain available to the caller.
    bool has_local_scale() const;
    bool try_linear_scale(weight_handle w, const float * x, uint32_t T, float * y, float scale,
                          size_t x_stride = 0, size_t y_stride = 0);

    // The same y computed on the host with libtsr_ref (the spec's oracle) from the weight rows read
    // back from the device: tsr_qact per token and K part, tsr_dot_row per row and K part, K parts
    // added in order with tsr_fadd. Bit-identical to linear() on a correct device.
    void linear_ref(weight_handle w, const float * x, uint32_t T, float * y, size_t x_stride = 0, size_t y_stride = 0) const;

    // ---- graphs ----

    // f32 rows [rows, n] (row r at data + r * row_stride_floats, 0 = n) into the AUX region, each row padded to whole lines.
    const_handle const_upload(const float * data, uint32_t rows, uint32_t n, size_t row_stride_floats = 0);
    void         const_free(const_handle h);   // unknown handle -> invalid_argument

    // nullptr if the device can run the program, else the reason (it names the op's tag and kind, or the value). Static checks
    // (values and ops well formed, shapes, the kinds the booted image has, weights and KV planes) and the plans of T = 1 and
    // T = T_MAX + 1 (nothing is allocated or uploaded: a konst with handle 0 and a mul_mat with w = 0 are checked by shape, as
    // a Q8_0 weight). A malformed program throws std::invalid_argument. The string is valid until the calling thread's next graph_unsupported call.
    const char * graph_unsupported(const graph_desc & g);
    // Validates as graph_unsupported (a program it refuses: tsr::rt::error for what the device cannot run, std::invalid_argument
    // for a malformed one), allocates and zero-fills the KV planes (error "does not fit" if the heap is full). Weights and
    // consts must exist and stay until graph_free.
    graph_handle graph_create(const graph_desc & g);
    void         graph_free(graph_handle h);   // releases the planes; unknown handle -> invalid_argument
    graph_shape  graph_shape_of(graph_handle h) const;

    // Runs T rows through the graph: write the inputs, launch the plan's lists, read the outs. A T that cannot be placed (the IO
    // region; T above kv_size or above graph_max_attn_tokens with an attn op without a slot) throws tsr::rt::error whose text
    // contains "does not fit"; a mask value other than 0 and -inf the same. Any other failure names the op (tag and kind) or the
    // device's answer.
    //
    // Cached attention (the graph has KV slots; `state` binds a kv_handle to each slot): row t of the run is cell state.pos0 + t.
    // Each cached attn op stores the K and V rows of the run into the cells [pos0, pos0 + T) of its planes and attends over the
    // planes: with the causal mask (mask -1) row t sees the cells 0 .. pos0 + t, with a given mask the cells it shows. pos0 + T
    // above the slot's cells is an error whose text contains "does not fit"; so is a given mask that shows, for a row, a cell at
    // or above pos0 + T (a cell the call does not write; the text names the row and the cell), and in a chunked run a cell at or
    // above the end of the row's chunk. The handles must exist, match the slot's E and cells (std::invalid_argument otherwise) and
    // be distinct.
    // A graph whose attn ops all have a slot (and that has at least one) runs in consecutive chunks of rows when T is above
    // graph_max_attn_tokens or the plan of T does not fit the IO region: chunk c runs the rows [r0, r1) as cells pos0 + r0 ..,
    // with the input rows, positions and mask rows of that range, its outs at the same rows of the results. The chunk is the
    // largest that plans (min(T, graph_max_attn_tokens), halved on "does not fit"; a chunk of one row that does not fit is the
    // error). Every other graph runs in one piece, with the same errors as ever. A graph without slots ignores state.pos0 and
    // takes no handles.
    void graph_run(graph_handle h, uint32_t T, const graph_inputs & in, const graph_outputs & out, const graph_state & state = {});
    // The same lists executed on the host (tsr::ref::opl_exec) on an image of the device's memory (constants, weights, and for a
    // cached attention the cells [0, pos0 + T) of the planes, whole groups of 16 cells) and the inputs: bit-identical to graph_run
    // on a correct device. It does not use the kernel or the simulator. It may be called after graph_run of the same step (the
    // cells >= pos0 are written again by its own SET_ROWS_KV records, in the image; the device's planes are not changed).
    void         graph_run_ref(graph_handle h, uint32_t T, const graph_inputs & in, const graph_outputs & out, const graph_state & state = {});
    // The plan of T (made, and cached, if the graph has none; a plan the runs already made is not made again). For a graph
    // with attn ops that all have a slot (it runs in chunks): the plan the first chunk of a run of T rows starts with, found as
    // graph_run finds it; info.T is the rows of that chunk (info.T < T: the run is chunked). pos0 plays no part, the plane
    // offsets are those of the last graph_run / graph_run_ref (fake ones before any). T above a slot's cells is refused ("does
    // not fit"). Any other graph: the plan of T with the KV objects and pos0 of the last run; T above graph_max_attn_tokens with
    // an attn op that has no slot is refused ("does not fit").
    graph_plan_info graph_info(graph_handle h, uint32_t T);

    // ---- KV objects (the planes of a cached attention that outlive graphs and calls) ----
    // K and V plane of `cells` cells of E values each (E a multiple of 64, up to 511 * 64; geometry tsr_k_kv_geom(E, cells,
    // num_ch) as the planes of a graph), taken from the top of the WEIGHT heap and zero-filled in every channel (device memory
    // this open did not write is never read). tsr::rt::error whose text contains "does not fit" when the heap is full or the
    // geometry cannot be placed; std::invalid_argument for E or cells that no plane can have.
    kv_handle    kv_create(uint32_t E, uint32_t cells);
    void         kv_free(kv_handle h);                 // unknown handle -> invalid_argument
    kv_info      kv_info_of(kv_handle h) const;        // unknown handle -> invalid_argument
    // n rows of E f32 (row t of k / v at + t * stride floats, 0 = E) into the cells [cell0, cell0 + n) of the K and the V plane,
    // quantized by the device exactly as the SET_ROWS_KV records of an attn op store them (op lists of two SET_ROWS_KV records per
    // chunk of at most graph_max_attn_tokens rows). cell0 + n <= cells (std::invalid_argument otherwise, also for n = 0). n = 0: no rows are stored.
    void         kv_store(kv_handle h, uint32_t cell0, uint32_t n, const float * k, const float * v, size_t k_stride = 0, size_t v_stride = 0);
    // The stored rows as ggml block_q8_0 rows (E / 32 blocks of 34 bytes per cell, n rows packed), read from the device through
    // tsr_kv_unpack_row of libtsr_ref. cell0 + n <= cells (std::invalid_argument otherwise, also for n = 0).
    void         kv_read(kv_handle h, uint32_t cell0, uint32_t n, void * k_blocks, void * v_blocks) const;

    counters    get_counters() const;
    void        reset_counters();
    memory_info memory() const;

    struct impl;
private:
    bool linear_impl(weight_handle w, const float * x, uint32_t T, float * y,
                     size_t x_stride, size_t y_stride, const float * scale);
    std::unique_ptr<impl> p_;
};

}  // namespace tsr::rt
