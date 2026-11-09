// ggml-tesseract.cpp — ggml backend for the Tesseract NPU core (spec "ggml backend integration";
// docs/design/06-software-verification.md "SW" §9.3, M6 = P3.2, M7 = P3.5 (W3b)).
//
//   reg "Tesseract" -> one device "Tesseract0" (GGML_BACKEND_DEVICE_TYPE_GPU, so -ngl offloads to it)
//   buffer type     -> a virtual address space; each buffer's storage is fixed at its first data
//                      access, by usage (llama.cpp sets it only after init_tensor):
//       WEIGHTS-usage buffers ("PLACED"): per tensor. MUL_MAT weights (2D, loadable type) are
//           converted to the device format (Q8_0 / Q4_K, spec "Load-time conversion"), repacked
//           (tsr_repack_row) and stored with symmetric allocation in the WEIGHT heap of every
//           channel. Anything else (1D norms, odd shapes) is stored raw in AUX (ch0).
//       KV buffers (W3b kv, tsr-kvbuf.h): an ANY buffer whose tensors are all llama.cpp KV-cache
//           tensors is stored as KV planes (AT §4), converted on get / set (SW-17).
//       ANY / COMPUTE buffers: raw bytes, byte-for-byte like any GPU buffer (activations,
//           test-backend-ops tensors), either
//           "LINEAR": the whole buffer is one window that the device ops read and write in place:
//               in the ch0 IO region, the low 512 MiB of ch1..ch(C-1), or the compute heap
//               (W3b, SW §8 R6, SW-26: symmetric reservations of the WEIGHT heap between the
//               weights and the KV planes, shared out per channel); or
//           "HOSTED": host memory, when no window fits (device memory exhausted; with
//               TSR_COMPUTE_WINDOWS=io: the IO region) or the buffer holds only leaf tensors (data
//               such as LoRA adapters, which no device op addresses directly). graph_compute stages
//               the src1/dst rows of a MUL_MAT through the IO staging area (top of IO) in T tiles.
//           A MUL_MAT src0 found in such a buffer is converted lazily at graph_compute into the
//           WEIGHT heap and cached by (buffer, offset, type, shape, write generation).
//   graph_compute   -> one op list (op list format) per split, run as one launch of the firmware's `graph`
//                      kernel (SW §9.3; one launch per decode token, SW §8):
//                      M6: every MUL_MAT node becomes one MUL_MAT record (dependencies from a hazard
//                      analysis over DRAM byte ranges, JOINs above two predecessors, tsr-opl.h); the
//                      firmware tiles each record with plan.hpp and pushes LD / QACT / MVM / ST
//                      (SW §7.4).
//                      M7 (W3b, tsr-build.h): every node of the supports_op table becomes a record of
//                      its kind (SW §7.2): passes 1-6 (no-ops dropped; tensors resolved to handles and
//                      line strides; MUL_MAT + ADD and RMS_NORM + MUL fused; dependencies; the preload
//                      table of positions, ids, KV indices and mask summaries; encoding), the plan of a
//                      graph cached by its identity (R7: a decode token runs its list again; only the
//                      inputs change). A list is cut early only at 4096 records or 256 preload lines.
//                      A MUL_MAT whose src1 or dst is in host memory (or whose dst rows do not start on
//                      a line, N*4 % 64 != 0 with T > 1) is staged: it runs on its own, one
//                      one-record launch per IO stage tile. Every list goes through the OPL cache (an LRU
//                      in the ch0 OPL region keyed by content). A TILE_ERR answer is mapped back to the
//                      ggml node (RET - 1 = the failing record, of any kind) and aborts with its name,
//                      the error code and ERR_INFO.
//
// The device: the transport selected by TSR_DEVICE (default sim; "none" registers no device) is
// opened (SOFT_RESET + bind, EX §10.1) and booted with TSR_IMAGE or the build's firmware image
// (tsr_image_open_default) at the first use. Versions (SW §6.9): an image whose Tesseract Basis
// (major, minor) is older than this backend's is refused (its planners may refuse what this
// backend's accept); supports_op routes only the kinds the booted image declares (.tsr_kinds), so
// a subset image leaves the other ops on the CPU; the description names the Tesseract Core
// capability and the Tesseract Basis version (get_proc_address "ggml_backend_tesseract_versions").
//
// Environment: TSR_DEBUG=1|2 (lowering and placement log on stderr; 2 adds every launched op
// list's records and preload entries, and supports_op's reasons). TSR_KINDS narrows the booted image's
// kinds (tsr_kinds_parse): m6 keeps M6's MUL_MAT (M6 runs, SW-14: NOP, JOIN and MUL_MAT); a list such
// as all,-FLASH_ATTN,-RMS_NORM leaves those kinds' ops on the CPU (finding which kind moves a result).
// Test knobs: TSR_FORCE_HOST=1 keeps every
// non-WEIGHTS buffer in host memory (exercises the staged path), TSR_COMPUTE_WINDOWS=io limits
// compute windows to ch0 IO (M6 placement), TSR_OPL_SPLIT=<n> cuts every op list after n records,
// TSR_STAGE_KIB=<n> sets the IO staging area size,
// TSR_DUMP_RE=<regex> TSR_DUMP_DIR=<dir> dumps the device value of matching nodes (tsr_dump_node),
// TSR_OPL_CRC=1 sets the `graph` launch flag VERIFY_CRC (the executor checks each list's CRC, SW
// §7.3). TSR_LOAD_THREADS=<n> sets the workers that convert and repack weights while they are
// uploaded (default: online CPUs, at most 16). TSR_STATS=1 prints, when the backend is freed, the
// upload statistics and (SW §9.2) graph_compute calls, launches, op records per launch,
// launches per scheduler pass (≈ per token) and per graph_compute, fused ops, records per kind, the
// plan cache, lists cut early, staged chunks, OPL cache hits / misses, the wall time spent in
// graph_compute and in launches and the device's PERF counters over the same span (host overhead =
// launch wall - PERF_CYCLES / f_kernel). The transport handles TSR_SIM_*, TSR_TRACE, TSR_LOG and the
// timeouts itself.
#include "ggml-tesseract.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include "tsr_isa.h"
#include "tsr_ref.h"
#include "tsr_mon.h"
#include "tsr_transport.h"

#include "tsr-alloc.h"
#include "tsr-build.h"
#include "tsr-lower.h"
#include "tsr-opl.h"
#include "tsr-weights.h"
#include "tsr-wlayout.h"
// W3b kv: KV-cache buffers (AT §4.6, SW-17) and the mask-summary producer (AT §7.3, SW-32)
#include <memory>
#include "tsr-kvbuf.h"
#include "tsr-msum.h"

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "ggml-tesseract writes op lists and tensors in host byte order; only little-endian hosts are supported"
#endif

#define TSR_REG_NAME  "Tesseract"
#define TSR_DEV_NAME  "Tesseract0"

static constexpr uint64_t TSR_TENSOR_ALIGN   = 64;                       // beat / SPM line
static constexpr uint64_t TSR_VADDR_BASE     = 0x4000000000000000ull;    // non-canonical: never dereferenced
static constexpr uint64_t TSR_VADDR_GUARD    = 1ull << 30;
static constexpr uint64_t TSR_HEAP_ALIGN     = 4096;
static constexpr size_t   TSR_XFER_CHUNK     = 8u << 20;                 // host-side staging granularity
// IO staging area (top of the ch0 IO region, reserved at open): activations of HOSTED buffers
// and line-padded output rows pass through it (staged, SW §9.3 step 2). At least
// 2 x SPM so that one token of any lowerable MUL_MAT (whose x + QA + y fit the SPM) always fits.
static constexpr uint64_t TSR_STAGE_DEFAULT  = 16ull << 20;

static int tsr_debug_level() {
    static const int lvl = [] {
        const char * e = getenv("TSR_DEBUG");
        return e ? atoi(e) : 0;
    }();
    return lvl;
}

static bool tsr_stats_on() {
    static const bool on = [] {
        const char * e = getenv("TSR_STATS");
        return e && atoi(e) != 0;
    }();
    return on;
}

// Workers for weight conversion and repack (TSR_LOAD_THREADS, default: online CPUs, at most 16).
static unsigned tsr_load_threads() {
    static const unsigned n = [] {
        const char * e = getenv("TSR_LOAD_THREADS");
        const int v = e ? atoi(e) : (int) std::min(16u, std::max(1u, std::thread::hardware_concurrency()));
        return (unsigned) std::max(1, std::min(v, 64));
    }();
    return n;
}

static uint64_t tsr_now_ns() {
    return (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

#define TSR_DBG(lvl, ...)                                   \
    do {                                                    \
        if (tsr_debug_level() >= (lvl)) {                   \
            fprintf(stderr, "[tesseract] " __VA_ARGS__);    \
        }                                                   \
    } while (0)

static uint64_t tsr_pad(uint64_t v, uint64_t a) {
    return (v + a - 1) / a * a;
}

static double tsr_mib(uint64_t b) {
    return (double) b / (1024.0 * 1024.0);
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

// Disjoint [lo, hi) intervals, merged on insert (split-upload coverage tracking).
struct tsr_intervals {
    std::map<uint64_t, uint64_t> m;

    void add(uint64_t lo, uint64_t hi) {
        if (lo >= hi) {
            return;
        }
        auto it = m.upper_bound(lo);
        if (it != m.begin()) {
            auto p = std::prev(it);
            if (p->second >= lo) {
                lo = p->first;
                hi = std::max(hi, p->second);
                it = m.erase(p);
            }
        }
        while (it != m.end() && it->first <= hi) {
            hi = std::max(hi, it->second);
            it = m.erase(it);
        }
        m[lo] = hi;
    }

    bool covers(uint64_t lo, uint64_t hi) const {
        if (lo >= hi) {
            return true;
        }
        auto it = m.upper_bound(lo);
        if (it == m.begin()) {
            return false;
        }
        --it;
        return it->first <= lo && it->second >= hi;
    }
};

// ---------------------------------------------------------------------------
// Device, buffer and tensor state
// ---------------------------------------------------------------------------

enum class tsr_kind : uint8_t {
    UNPLACED,    // known (init_tensor) but no data written yet
    DEVWEIGHT,   // device-format weight in the WEIGHT heap (symmetric allocation)
    RAW,         // raw ggml bytes on ch0 (AUX region, or WEIGHT heap if AUX is full)
};

enum class tsr_raw_region : uint8_t { NONE, AUX, HEAP };

struct tsr_tensor_rec {
    std::string name;
    enum ggml_type type = GGML_TYPE_F32;
    int64_t  ne[GGML_MAX_DIMS] = { 0, 0, 0, 0 };
    size_t   nb[GGML_MAX_DIMS] = { 0, 0, 0, 0 };
    size_t   nbytes = 0;
    uint64_t vaddr  = 0;          // virtual address of byte 0 (tensor->data)
    bool     leaf   = false;      // op NONE and not a graph input (data, not an activation)

    tsr_kind kind = tsr_kind::UNPLACED;

    // DEVWEIGHT
    uint32_t fmt = 0;
    uint32_t N = 0;
    uint32_t K = 0;
    uint64_t w_off = 0;           // byte offset of W_OFF (identical on every channel)
    uint64_t ch_bytes = 0;        // bytes reserved per channel = n_exp * exp_ch_bytes
    // Expert tensor (MoE, spec §4 "Expert tensors", SW-60): n_exp matrices of N x K, ggml dimension 2; expert e is
    // symmetric-allocated at w_off + e * exp_ch_bytes (EXP_STRIDE = exp_ch_bytes / 64 beats). n_exp = 1: a matrix.
    uint32_t n_exp = 1;
    uint64_t exp_ch_bytes = 0;    // bytes of one expert per channel = R * ROW_BEATS * 64
    bool     complete = false;    // all bytes received and stored
    std::vector<uint8_t> staging; // split upload in progress: original ggml bytes
    tsr_intervals have;           // received byte ranges of `staging`
    bool     warned_lossy = false;

    // RAW
    tsr_raw_region raw_region = tsr_raw_region::NONE;
    uint64_t raw_off = 0;         // ch0 byte offset
    // SW §7.10: the taps of a Gated DeltaNet convolution (ggml [d_conv, C], tap fastest) are stored as d_conv rows of C
    // values, the layout of the SSM_CONV kind; taps = d_conv, 0 for every other tensor. The upload is staged whole
    // (staging / have / complete as for a device-format weight) and converted when complete.
    uint32_t taps = 0;
};

enum class tsr_bmode : uint8_t { UNDECIDED, LINEAR, PLACED, HOSTED,
                                 KV /* W3b kv: a KV-cache buffer stored as KV planes (tsr-kvbuf.h) */ };

struct tsr_device;

struct tsr_buffer_ctx {
    tsr_device * dev = nullptr;
    uint64_t id   = 0;            // unique per process, never reused
    uint64_t base = 0;            // virtual base (64-aligned, non-NULL)
    uint64_t size = 0;
    tsr_bmode mode = tsr_bmode::UNDECIDED;
    bool     has_io = false;
    uint64_t io_off = 0;          // LINEAR: byte offset of the window in channel `ch` (base <-> io_off)
    // W3b builder (SW §8 R6, SW-26, EX §11): the window's channel and region: ch0 IO, the low
    // 512 MiB of ch1..ch(C-1), or the compute heap (a symmetric reservation of the WEIGHT heap that
    // the channels share out, reservation `win_res`).
    uint32_t ch = 0;
    uint8_t  win = 0;             // tsr_win_*
    int32_t  win_res = -1;
    bool     computed_view = false;   // init_tensor saw the view result of an op (not leaf data)
    uint8_t * host  = nullptr;    // HOSTED: the buffer's bytes in host memory (base <-> host)
    uint8_t  clear_value = 0;     // PLACED: value read back for never-written tensors
    std::map<uint64_t, tsr_tensor_rec> recs;   // PLACED (and UNDECIDED non-COMPUTE): by vaddr
    uint64_t gen = 1;             // bumped on every host or device write
    std::unique_ptr<tsr_kvbuf> kv;   // W3b kv: KV mode: the planes of the buffer's cache tensors
};

struct tsr_lazy_key {
    uint64_t buf_id;
    uint64_t src_off;             // ch0 byte offset of the raw source bytes
    int      type;
    uint32_t N;
    uint32_t K;
    uint32_t n_exp;

    bool operator<(const tsr_lazy_key & o) const {
        if (buf_id != o.buf_id)   return buf_id < o.buf_id;
        if (src_off != o.src_off) return src_off < o.src_off;
        if (type != o.type)       return type < o.type;
        if (N != o.N)             return N < o.N;
        if (K != o.K)             return K < o.K;
        return n_exp < o.n_exp;
    }
};

struct tsr_lazy_entry {
    uint64_t gen = 0;             // source buffer generation the conversion was made from
    uint32_t fmt = 0;
    uint64_t w_off = 0;
    uint64_t ch_bytes = 0;        // all experts, per channel
    uint64_t exp_ch_bytes = 0;    // one expert, per channel
};

// W3b builder: a graph's cached plan (SW §8 R7, §9.3 "cached per graph identity"): the op lists one
// graph_compute of it launched, in order. A plan is kept only when its build was static (no staged
// MUL_MAT, no lazily converted weight), so running its lists again is exactly what a rebuild would
// launch; the inputs that change per token (positions, KV indices, masks) reach the device through
// the lists' preload tables.
struct tsr_buffer_ctx;
struct tsr_plan_list {
    std::vector<uint8_t>     blob;
    std::vector<std::string> names;        // node names by record tag
    std::vector<int32_t>     tags;         // record index -> tag (-1: JOIN)
    uint32_t n_ops = 0, n_joins = 0, n_fused = 0, n_gdn_cell = 0;   // n_gdn_cell: GATED_DELTA_NET records on their cache cell (SW-82)
    uint32_t n_kind[tsr::opl::N_KINDS] = {};
    std::vector<tsr_buffer_ctx *> written; // buffers the list writes (their generations)
};
struct tsr_plan {
    std::vector<tsr_plan_list> lists;
    std::vector<const ggml_tensor *> fa_nodes;   // re-registered as mask readers on every reuse
    std::vector<tsr_msum_ref> fa_refs;           // the words the lists carry
    std::vector<tsr_gdn_cell> gdn_cells;         // the state gathers of GATED_DELTA_NET nodes and what the build decided (SW-82)
    uint64_t used = 0;                           // LRU stamp
};

struct tsr_device {
    std::mutex   mtx;
    bool         tried_open = false;
    tsr_transport * tp = nullptr;
    tsr_config   cfg = {};
    std::string  description = "Tesseract Core (not opened)";
    uint64_t     kind_mask = 0;   // the op list kinds supports_op may route (SW §6.9: the booted image's,
                                  // narrowed by TSR_KINDS, tsr_kinds_parse)
    std::string  image_desc;      // "Tesseract Basis 0.1.0, image <id>" (supports_op's reasons)

    tsr_range_alloc io;           // ch0 IO region below the stage: LINEAR buffer windows
    tsr_range_alloc aux;          // ch0 AUX region: RAW tensors of weight buffers
    tsr_range_alloc heap;         // WEIGHT heap, symmetric: an offset is reserved on every channel
    // W3b builder: compute windows besides ch0 IO (SW §8 R6, SW-26): low[c] = [0, 512 MiB) of
    // channel c >= 1; the compute heap: symmetric reservations of the WEIGHT heap between the
    // weights (bottom-up) and the KV planes (top-down), each shared out per channel.
    std::vector<tsr_range_alloc> low;
    struct cheap_res {
        uint64_t off = 0, size = 0;            // the reservation (every channel)
        std::vector<tsr_range_alloc> ch;       // its windows, per channel
    };
    std::vector<cheap_res> cheap;

    uint64_t stage_off  = 0;      // ch0 byte offset of the IO staging area (top of IO)
    uint64_t stage_size = 0;
    // The scratch of the multi-tile MOE_ROUTE records (SW-76): one window (channel, byte offset, size), taken at the first
    // use from the top of IO (or of a low window) and replaced by a larger one when a route needs more. A window that was
    // replaced stays reserved: cached plans may still name it.
    uint32_t scr_ch = 0;
    uint64_t scr_off = 0, scr_size = 0;
    bool     force_host = false;  // TSR_FORCE_HOST=1: every non-WEIGHTS buffer is HOSTED

    uint64_t next_vaddr  = TSR_VADDR_BASE;
    uint64_t next_buf_id = 1;

    std::map<tsr_lazy_key, tsr_lazy_entry> lazy;

    tsr_opl_dev odev;             // the `graph` kernel: boot, OPL cache, launches (tsr-opl.h)
    uint32_t    launch_flags = 0; // `graph` a2 (TSR_OPL_CRC=1: VERIFY_CRC)

    // W3b builder: the per-graph plan cache (R7), keyed by tsr_graph_key salted with the placement
    // epoch (bumped whenever a buffer is freed or a weight / lazily converted weight moves).
    std::map<uint64_t, tsr_plan> plans;
    uint64_t plan_epoch = 1, plan_stamp = 0;
    bool     build_dynamic = false;     // the build in progress staged a MUL_MAT or converted a weight
    tsr::plan::dev_cfg m7cfg = {};      // the planner's device view with the M7 fields (CFG3 PCAP, CAPS)
    bool     m7cfg_ok = false;

    // W3b kv: the AT parameters (C, T_MAX, PCAP from CFG3, K_MAX) and the mask summaries
    tsr_att_cfg    att_cfg = {};
    tsr_msum_store msum;

    // The live monitor record (tsr_mon.h): device weight bytes by ggml type (the dominant one is the
    // quantisation shown), and the splits of the scheduler pass in progress (a split seen again starts
    // the next pass, as in the TSR_STATS pass count, which this does not depend on)
    uint64_t mon_wbytes[GGML_TYPE_COUNT] = {};
    int      mon_wtype = -1;
    std::set<const ggml_cgraph *> mon_splits;

    // TSR_STATS=1
    uint64_t st_graphs = 0;        // graph_compute calls
    uint64_t st_compute_ns = 0;    // their wall time
    uint64_t st_mm = 0;            // MUL_MAT nodes run
    uint64_t st_mm_records = 0;    // MUL_MAT records launched (staged ones included)
    uint64_t st_joins = 0;         // JOIN records launched
    uint64_t st_staged_mm = 0;     // MUL_MATs that took the staged path ("staged chunks", SW §8 R6)
    uint64_t st_staged_launches = 0;
    uint64_t st_fused = 0;         // W3b: fused records (MUL_MAT + ADD, RMS_NORM + MUL) launched
    uint64_t st_gdn_cell = 0;      // SW-82: GATED_DELTA_NET records that read and write their cache cell in place
    uint64_t st_kind[tsr::opl::N_KINDS] = {};   // W3b: records launched per kind
    uint64_t st_plan_hits = 0, st_plan_misses = 0, st_plan_dynamic = 0;   // W3b: plan cache (R7)
    uint64_t st_cut_lists = 0;     // W3b: lists cut early (4096 records or 256 preload lines)
    uint64_t st_launch_hist[4] = {};   // W3b: graph_compute calls with 0, 1, 2, >= 3 launches
    uint64_t st_passes = 0;        // scheduler passes (≈ tokens): a split seen again starts a new pass
    std::set<const ggml_cgraph *> st_pass_splits;
    std::vector<uint64_t> st_perf0;   // every PERF_* counter at the first launch
    uint64_t st_write_ns = 0;      // tsr_transport_write
    uint64_t st_write_bytes = 0;
    uint64_t st_set_ns = 0;        // set_tensor (conversion + repack + writes)
    uint64_t st_set_bytes = 0;
    uint64_t st_store_ns = 0;      // weight stores (convert + repack + writes)
    std::atomic<uint64_t> st_conv_ns { 0 };   // non-native rows converted to the device format (CPU time, all threads)

    ggml_backend_buffer_type buft;
    ggml_backend_device      gdev;
};

static tsr_device * tsr_dev_singleton() {
    static tsr_device * d = new tsr_device();   // intentionally leaked: buffers may outlive static dtors
    return d;
}

static tsr_buffer_ctx * tsr_buf(ggml_backend_buffer_t buffer) {
    return (tsr_buffer_ctx *) buffer->context;
}

// ---------------------------------------------------------------------------
// Transport access (callers hold dev->mtx)
// ---------------------------------------------------------------------------

static void tsr_mon_note_mode(tsr_device * d);

static bool tsr_dev_open_locked(tsr_device * d) {
    if (d->tp != nullptr) {
        return true;
    }
    if (d->tried_open) {
        return false;
    }
    d->tried_open = true;

    tsr_transport * tp = tsr_transport_open(nullptr);
    if (tp == nullptr) {
        GGML_LOG_ERROR("%s: failed to open the Tesseract transport (TSR_DEVICE=%s)\n", __func__,
                       getenv("TSR_DEVICE") ? getenv("TSR_DEVICE") : "(unset: sim)");
        return false;
    }
    const tsr_config * c = tsr_transport_config(tp);
    const char * bad = nullptr;
    if (c == nullptr)                                     bad = "no configuration";
    else if (c->beat_bits != 512)                         bad = "BEAT_BITS != 512 (the software assumes 64-byte beats)";
    else if (c->num_ch < 1)                               bad = "NUM_CH == 0";
    else if (c->t_max < 1)                                bad = "T_MAX == 0";
    else if (c->spm_bytes < 64 * TSR_LINE_BYTES)          bad = "SPM too small";
    else if (c->ch_bytes <= TSR_REGION_WEIGHT_OFF)        bad = "channel smaller than the DRAM map (CH_BYTES <= WEIGHT_OFF)";
    // IO staging area: default max(16 MiB, 2 x SPM); TSR_STAGE_KIB overrides (tests use a small
    // stage to force T tiling). At most half of IO, so LINEAR windows keep most of the region.
    uint64_t stage = std::max<uint64_t>(TSR_STAGE_DEFAULT, 2ull * (c != nullptr ? c->spm_bytes : 0));
    if (bad == nullptr && getenv("TSR_STAGE_KIB") != nullptr) {
        char * end = nullptr;
        const unsigned long long kib = strtoull(getenv("TSR_STAGE_KIB"), &end, 10);
        if (end == getenv("TSR_STAGE_KIB") || *end != '\0' || kib == 0 || kib * 1024 > TSR_REGION_IO_SIZE / 2) {
            bad = "TSR_STAGE_KIB must be a number of KiB in [1, IO region / 2]";
        } else {
            stage = kib * 1024;
        }
    }
    stage = tsr_pad(std::min<uint64_t>(stage, TSR_REGION_IO_SIZE / 2), TSR_TENSOR_ALIGN);
    // TSR_KINDS (SW-14: m6 for M6 runs; a list such as all,-FLASH_ATTN leaves those kinds on the CPU)
    uint64_t kinds_env = ~0ull;
    std::string kinds_why;
    if (bad == nullptr && !tsr_kinds_parse(getenv("TSR_KINDS"), &kinds_env, &kinds_why)) {
        bad = kinds_why.c_str();
    }
    if (bad != nullptr) {
        GGML_LOG_ERROR("%s: unusable Tesseract configuration: %s\n", __func__, bad);
        tsr_transport_close(tp);
        return false;
    }
    // Boot the firmware (warm if the image is already loaded, EX §3.5) and find `graph`.
    std::string err;
    if (!d->odev.open(tp, &err)) {
        GGML_LOG_ERROR("%s: Tesseract on %s: %s\n", __func__, tsr_transport_name(tp), err.c_str());
        tsr_transport_close(tp);
        return false;
    }
    // SW §6.9: the image's Tesseract Basis (major, minor) must be at least this backend's (CUDA's
    // "insufficient driver" rule; the patch is ignored, a newer image is fine): the backend's planners
    // are the image's (SW-20), so an older image could refuse what supports_op accepts.
    const tsr_versions & v = d->odev.versions();
    const uint32_t host = tsr_basis_version();
    char hb[32], ib[32];
    snprintf(hb, sizeof(hb), "%u.%u.%u", (host >> 16) & 0xFFu, (host >> 8) & 0xFFu, host & 0xFFu);
    snprintf(ib, sizeof(ib), "%u.%u.%u", (v.basis >> 16) & 0xFFu, (v.basis >> 8) & 0xFFu, v.basis & 0xFFu);
    if ((v.basis >> 8) < (host >> 8)) {
        GGML_LOG_ERROR("tesseract: the booted image %s has Tesseract Basis %s, older than this backend's %s: its planners may "
                       "refuse what this backend sends; rebuild the image or unset TSR_IMAGE\n", d->odev.image_path(), ib, hb);
        tsr_transport_close(tp);
        return false;
    }
    d->kind_mask = 0;
    for (uint32_t k = 0; k < tsr::opl::N_KINDS; ++k) {
        d->kind_mask |= v.kind_rev[k] != 0 ? 1ull << k : 0ull;
    }
    d->kind_mask &= kinds_env;            // TSR_KINDS (SW-14: m6 keeps NOP, JOIN and MUL_MAT)
    if (getenv("TSR_KINDS") != nullptr) {
        std::string names;
        for (uint32_t k = 0; k < tsr::opl::N_KINDS; ++k) {
            names += ((d->kind_mask >> k) & 1u) != 0 ? std::string(names.empty() ? "" : " ") + tsr_kind_name(k) : "";
        }
        TSR_DBG(1, "TSR_KINDS=%s: the device takes %s\n", getenv("TSR_KINDS"), names.c_str());
    }
    {
        char idb[96];
        snprintf(idb, sizeof(idb), "Tesseract Basis %s, image %016" PRIx64, ib, v.image_id);
        d->image_desc = idb;
    }
    tsr_mon_note_mode(d);
    d->launch_flags = getenv("TSR_OPL_CRC") != nullptr && atoi(getenv("TSR_OPL_CRC")) != 0 ? tsr::opl::LF_VERIFY_CRC : 0u;
    d->tp  = tp;
    d->cfg = *c;
    d->stage_size = stage;
    d->stage_off  = TSR_REGION_IO_OFF + TSR_REGION_IO_SIZE - stage;
    d->io.init  (TSR_REGION_IO_OFF,  TSR_REGION_IO_SIZE - stage, TSR_TENSOR_ALIGN);
    d->aux.init (TSR_REGION_AUX_OFF, TSR_REGION_AUX_SIZE, TSR_TENSOR_ALIGN);
    d->heap.init(TSR_REGION_WEIGHT_OFF, c->ch_bytes - TSR_REGION_WEIGHT_OFF, TSR_HEAP_ALIGN);
    // W3b builder: the low 512 MiB of ch1..ch(C-1) hold compute windows (EX §11, SW §8 R6)
    d->low.assign(c->num_ch, tsr_range_alloc());
    for (uint32_t ch = 1; ch < c->num_ch; ++ch) {
        d->low[ch].init(0, TSR_REGION_WEIGHT_OFF, TSR_TENSOR_ALIGN);
    }
    d->force_host = getenv("TSR_FORCE_HOST") != nullptr && atoi(getenv("TSR_FORCE_HOST")) != 0;

    // "Tesseract Core 1.0 (VU ATT; sim, C=4, ...), Tesseract Basis 0.1.0 (ABI 1, 14 kinds, image <id>)":
    // tsr_versions_str's two parts (SW §6.9), the configuration and the image id added inside them.
    char part[256], cfgs[160];
    snprintf(cfgs, sizeof(cfgs), "%s, C=%u, LANES=%u, T_MAX=%u, K_MAX=%u, SPM=%u KiB, %.0f GiB/ch", tsr_transport_name(tp),
             c->num_ch, c->lanes, c->t_max, c->k_max, c->spm_bytes / 1024, (double) c->ch_bytes / (1024.0 * 1024.0 * 1024.0));
    tsr_versions only = v;
    only.basis = 0;
    tsr_versions_str(&only, 0, part, sizeof(part));            // "Tesseract Core 1.0 (VU ATT)"
    std::string desc = part;
    desc = desc.back() == ')' ? desc.substr(0, desc.size() - 1) + "; " + cfgs + ")" : desc + " (" + cfgs + ")";
    only = v;
    only.core = 0;
    tsr_versions_str(&only, 0, part, sizeof(part));            // "Tesseract Basis 0.1.0 (ABI 1, 14 kinds)"
    std::string bas = part;
    char idb[32];
    snprintf(idb, sizeof(idb), ", image %016" PRIx64 ")", v.image_id);
    desc += ", " + (bas.empty() ? std::string() : bas.substr(0, bas.size() - 1) + idb);
    d->description = desc;
    TSR_DBG(1, "opened %s; image %s (graph = kernel %d); IO windows %.2f MiB, IO stage %.2f MiB at ch0+0x%09" PRIx64 "%s\n",
            desc.c_str(), d->odev.image_path(), d->odev.graph_kernel(), tsr_mib(d->io.total()), tsr_mib(stage), d->stage_off,
            d->force_host ? ", TSR_FORCE_HOST" : "");
    return true;
}

static void tsr_write(tsr_device * d, uint32_t ch, uint64_t off, const void * src, size_t size) {
    if (size == 0) {
        return;
    }
    const uint64_t t0 = tsr_stats_on() ? tsr_now_ns() : 0;
    if (tsr_transport_write(d->tp, ch, off, src, size) != 0) {
        GGML_ABORT("tesseract: transport write failed (ch %u, offset 0x%" PRIx64 ", %zu bytes)", ch, off, size);
    }
    if (tsr_stats_on()) {
        d->st_write_ns += tsr_now_ns() - t0;
        d->st_write_bytes += size;
    }
}

static void tsr_read(tsr_device * d, uint32_t ch, uint64_t off, void * dst, size_t size) {
    if (size == 0) {
        return;
    }
    if (tsr_transport_read(d->tp, ch, off, dst, size) != 0) {
        GGML_ABORT("tesseract: transport read failed (ch %u, offset 0x%" PRIx64 ", %zu bytes)", ch, off, size);
    }
}

static void tsr_fill(tsr_device * d, uint32_t ch, uint64_t off, uint8_t value, size_t size) {
    std::vector<uint8_t> buf(std::min(size, TSR_XFER_CHUNK), value);
    while (size > 0) {
        const size_t n = std::min(size, buf.size());
        tsr_write(d, ch, off, buf.data(), n);
        off  += n;
        size -= n;
    }
}

// ---------------------------------------------------------------------------
// The live monitor record (tsr_mon.h): what only the backend knows. Every call is cheap and none of
// them reaches the device.
// ---------------------------------------------------------------------------

// Weight bytes by type; the dominant type names the quantisation ("Q4_K").
static void tsr_mon_note_weight(tsr_device * d, enum ggml_type type, uint64_t bytes) {
    if ((int) type < 0 || (int) type >= (int) GGML_TYPE_COUNT) {
        return;
    }
    d->mon_wbytes[type] += bytes;
    int best = d->mon_wtype;
    for (int t = 0; t < (int) GGML_TYPE_COUNT; ++t) {
        if (best < 0 || d->mon_wbytes[t] > d->mon_wbytes[best]) {
            best = t;
        }
    }
    if (best != d->mon_wtype) {
        d->mon_wtype = best;
        tsr_mon_set_model(nullptr, ggml_type_name((enum ggml_type) best));
    }
}

// Device DRAM in use over all channels: the symmetric heap (weights, KV, compute windows), ch0 IO, AUX,
// the staging area, the low windows of the other channels and the OPL cache.
static void tsr_mon_note_mem(tsr_device * d) {
    uint64_t used = d->heap.used() * d->cfg.num_ch + d->io.used() + d->aux.used() + d->stage_size + d->odev.cache().used_bytes();
    for (uint32_t ch = 1; ch < d->low.size(); ++ch) {
        used += d->low[ch].used();
    }
    tsr_mon_set_mem(used, (uint64_t) d->cfg.num_ch * d->cfg.ch_bytes);
}

// "M6" when the device takes the matrix products only (TSR_KINDS=m6, with MUL_MAT_ID for a MoE model, as
// klcheck --phase m6 sets it), else "M7"; the kinds it takes by name.
static void tsr_mon_note_mode(tsr_device * d) {
    uint64_t m6 = 0;
    std::string why;
    tsr_kinds_parse("m6,+MUL_MAT_ID", &m6, &why);
    std::string names;
    for (uint32_t k = 0; k < tsr::opl::N_KINDS; ++k) {
        if (((d->kind_mask >> k) & 1u) != 0) {
            names += (names.empty() ? "" : " ") + std::string(tsr_kind_name(k));
        }
    }
    tsr_mon_set_mode((d->kind_mask & ~m6) == 0 ? "M6" : "M7", d->kind_mask, names.c_str());
}

// The tokens of the scheduler pass whose first split this is: the second dimension of the first MUL_MAT's
// activation (the third for MUL_MAT_ID), 0 if the split has none.
static uint32_t tsr_mon_pass_tokens(const ggml_cgraph * g) {
    const int n = std::min(g->n_nodes, 512);
    for (int i = 0; i < n; ++i) {
        const ggml_tensor * t = g->nodes[i];
        if (t->op == GGML_OP_MUL_MAT && t->src[1] != nullptr) {
            return (uint32_t) std::max<int64_t>(1, t->src[1]->ne[1]);
        }
        if (t->op == GGML_OP_MUL_MAT_ID && t->src[1] != nullptr) {
            return (uint32_t) std::max<int64_t>(1, t->src[1]->ne[2]);
        }
    }
    return 0;
}

static void tsr_mon_note_graph(tsr_device * d, const ggml_cgraph * g) {
    if (!tsr_mon_enabled()) {
        return;
    }
    if (d->mon_splits.size() > 4096) {
        d->mon_splits.clear();
    }
    if (d->mon_splits.empty() || d->mon_splits.count(g) != 0) {
        d->mon_splits.clear();
        const uint32_t n = tsr_mon_pass_tokens(g);
        tsr_mon_add_tokens(n == 1 ? 1u : 0u, n > 1 ? n : 0u, 1u);
    }
    d->mon_splits.insert(g);
}

// ---------------------------------------------------------------------------
// Weight storage: convert + repack + scatter rows (symmetric allocation)
// ---------------------------------------------------------------------------

// Store rows [r0, r1) of an N x K weight of ggml type `type` at W_OFF = w_off (bytes) in the
// device format `fmt`. `src` points at row r0 (ggml layout, rows contiguous).
static void tsr_store_weight_rows(tsr_device * d, enum ggml_type type, uint32_t fmt, uint32_t N, uint32_t K,
                                  uint64_t w_off, const uint8_t * src, uint32_t r0, uint32_t r1) {
    const size_t src_rb = ggml_row_size(type, K);
    const bool stats = tsr_stats_on();
    const uint64_t t0 = stats ? tsr_now_ns() : 0;
    // repack + symmetric scatter: tsr-wlayout (shared with the ggml-free runtime sw/rt). The
    // conversion runs on tsr_load_threads() workers (each with its own f32 row); the writes stay
    // on this thread.
    tsr_weight_store_rows(d->cfg.num_ch, fmt, N, K, w_off, r0, r1, TSR_XFER_CHUNK,
        [&](uint32_t rr, void * scratch) {
            thread_local std::vector<float> f32;
            if (f32.size() < K) {
                f32.resize(K);
            }
            const uint64_t c0 = stats && !tsr_type_is_native(type) ? tsr_now_ns() : 0;
            const void * blk = tsr_row_to_dev_blocks(type, fmt, src + (size_t) (rr - r0) * src_rb, K, scratch, f32.data());
            if (c0) {
                d->st_conv_ns += tsr_now_ns() - c0;
            }
            return blk;
        },
        [d](uint32_t ch, uint64_t off, const void * p, size_t n) { tsr_write(d, ch, off, p, n); },
        tsr_load_threads());
    if (stats) {
        d->st_store_ns += tsr_now_ns() - t0;
    }
}

// Read rows [r0, r1) of a DEVWEIGHT back in the tensor's own ggml type (debug get_tensor).
static void tsr_load_weight_rows(tsr_device * d, tsr_tensor_rec & rec, uint32_t r0, uint32_t r1, uint8_t * dst) {
    const size_t src_rb = ggml_row_size(rec.type, rec.K);

    if (!tsr_type_is_native(rec.type) && !rec.warned_lossy) {
        rec.warned_lossy = true;
        GGML_LOG_WARN("tesseract: reading back weight '%s' (%s, stored as %s on the device): the data is "
                      "re-encoded from the device copy and is NOT the original bytes\n",
                      rec.name.c_str(), ggml_type_name(rec.type), rec.fmt == TSR_FMT_Q8_0 ? "Q8_0" : "Q4_K");
    }

    std::vector<float> f32(rec.K);
    // symmetric gather + un-repack: tsr-wlayout (shared with the ggml-free runtime sw/rt)
    tsr_weight_load_rows(d->cfg.num_ch, rec.fmt, rec.N, rec.K, rec.w_off, r0, r1, TSR_XFER_CHUNK,
        [d](uint32_t ch, uint64_t off, void * p, size_t n) { tsr_read(d, ch, off, p, n); },
        [&](uint32_t rr, const void * blocks) {
            if (!tsr_dev_blocks_to_row(rec.type, rec.fmt, blocks, rec.K, dst + (size_t) (rr - r0) * src_rb, f32.data())) {
                GGML_ABORT("tesseract: cannot read back weight '%s': it is stored on the device as %s and type %s "
                           "cannot be re-encoded from f32. Something outside the device is reading this weight: "
                           "either get_tensor was called on it, or the scheduler copies it to another backend "
                           "because an op using it was not accepted by Tesseract (supports_op)",
                           rec.name.c_str(), rec.fmt == TSR_FMT_Q8_0 ? "Q8_0" : "Q4_K", ggml_type_name(rec.type));
            }
        });
}

// Expert tensors (MoE, SW-60): the n_exp x N rows of the ggml tensor in order. Row rr is row rr % N of expert
// rr / N, and expert e sits at w_off + e * exp_ch_bytes on every channel (spec §4 EXP_STRIDE). Rows [r0, r1)
// at src (row r0), as tsr_store_weight_rows per expert.
static void tsr_store_expert_rows(tsr_device * d, enum ggml_type type, uint32_t fmt, uint32_t N, uint32_t K,
                                  uint64_t exp_ch_bytes, uint64_t w_off, const uint8_t * src, uint32_t r0, uint32_t r1) {
    const size_t src_rb = ggml_row_size(type, K);
    while (r0 < r1) {
        const uint32_t e = r0 / N, lo = r0 - e * N, hi = std::min<uint32_t>(r1 - e * N, N);
        tsr_store_weight_rows(d, type, fmt, N, K, w_off + (uint64_t) e * exp_ch_bytes, src, lo, hi);
        src += (size_t) (hi - lo) * src_rb;
        r0 += hi - lo;
    }
}

// The same rows read back in the tensor's type (debug get_tensor).
static void tsr_load_expert_rows(tsr_device * d, tsr_tensor_rec & rec, uint32_t r0, uint32_t r1, uint8_t * dst) {
    const size_t src_rb = ggml_row_size(rec.type, rec.K);
    const uint64_t w_off = rec.w_off;
    while (r0 < r1) {
        const uint32_t e = r0 / rec.N, lo = r0 - e * rec.N, hi = std::min<uint32_t>(r1 - e * rec.N, rec.N);
        rec.w_off = w_off + (uint64_t) e * rec.exp_ch_bytes;
        tsr_load_weight_rows(d, rec, lo, hi, dst);
        dst += (size_t) (hi - lo) * src_rb;
        r0 += hi - lo;
    }
    rec.w_off = w_off;
}

// ---------------------------------------------------------------------------
// PLACED (weights) buffers
// ---------------------------------------------------------------------------

// W3b builder: something a cached plan may have baked into its lists moved or went away (a weight's
// W_OFF, a lazily converted weight, a buffer): forget every plan (R7).
static void tsr_plans_invalidate(tsr_device * d) {
    d->plan_epoch++;
    d->plans.clear();
}

static void tsr_rec_release(tsr_device * d, tsr_tensor_rec & r) {
    if (r.kind != tsr_kind::UNPLACED) {
        tsr_plans_invalidate(d);
    }
    if (r.kind == tsr_kind::DEVWEIGHT) {
        d->heap.release(r.w_off);
    } else if (r.kind == tsr_kind::RAW) {
        if (r.raw_region == tsr_raw_region::AUX) {
            d->aux.release(r.raw_off);
        } else if (r.raw_region == tsr_raw_region::HEAP) {
            d->heap.release(r.raw_off);
        }
    }
    r.kind = tsr_kind::UNPLACED;
    r.raw_region = tsr_raw_region::NONE;
    r.complete = false;
    r.staging.clear();
    r.staging.shrink_to_fit();
    r.have = tsr_intervals();
}

// Free every lazily converted weight (their heap space). Callers must have flushed any
// pending op list that references them.
static void tsr_lazy_evict_all(tsr_device * d) {
    for (auto & kv : d->lazy) {
        d->heap.release(kv.second.w_off);
    }
    if (!d->lazy.empty()) {
        TSR_DBG(1, "evicted %zu lazily converted weights\n", d->lazy.size());
        tsr_plans_invalidate(d);
    }
    d->lazy.clear();
}

static void tsr_lazy_evict_buffer(tsr_device * d, uint64_t buf_id) {
    for (auto it = d->lazy.begin(); it != d->lazy.end();) {
        if (it->first.buf_id == buf_id) {
            tsr_plans_invalidate(d);
            d->heap.release(it->second.w_off);
            it = d->lazy.erase(it);
        } else {
            ++it;
        }
    }
}

static bool tsr_heap_alloc(tsr_device * d, uint64_t size, uint64_t * off) {
    if (d->heap.alloc(size, off)) {
        return true;
    }
    // lazily converted weights are a cache: drop them and retry (only called with no
    // pending op list, see tsr_place / tsr_lazy_weight)
    tsr_lazy_evict_all(d);
    return d->heap.alloc(size, off);
}

// Is this tensor stored as a device weight (vs raw)? Mirrors what MUL_MAT can use directly.
// An expert tensor of a MoE model (llama.cpp: ffn_gate_exps, ffn_up_exps, ffn_down_exps, ffn_gate_up_exps): 3D, the
// experts in dimension 2, used by MUL_MAT_ID. Recognised by its name, because a placed weight must be decided when it
// is uploaded; other 3D tensors stay raw (a lazily converted MUL_MAT_ID source is the other way in).
static bool tsr_rec_is_expert_name(const tsr_tensor_rec & r) {
    return r.name.find("_exps") != std::string::npos;
}

static bool tsr_rec_is_devweight_shape(const tsr_device * d, const tsr_tensor_rec & r) {
    if (r.ne[3] != 1 || (r.ne[2] != 1 && (!tsr_rec_is_expert_name(r) || r.ne[2] > (int64_t) tsr::opl::mmid::MAX_EXPERTS ||
                                            r.nb[2] != r.nb[1] * (size_t) r.ne[1]))) {
        return false;
    }
    if (!tsr_type_loadable(r.type)) {
        return false;
    }
    if (r.ne[1] == 1 && !ggml_is_quantized(r.type)) {
        return false;   // 1D float vector (norm weight, bias): stays raw (spec: 1D -> AUX)
    }
    if (r.nb[1] != ggml_row_size(r.type, r.ne[0])) {
        return false;
    }
    return tsr_mm_shape_ok(d->cfg, tsr_dev_fmt_for_type(r.type), (uint64_t) r.ne[1], (uint64_t) r.ne[0]);
}

static bool tsr_rec_is_conv_taps(const tsr_tensor_rec & r);

static void tsr_place(tsr_device * d, tsr_tensor_rec & r) {
    GGML_ASSERT(r.kind == tsr_kind::UNPLACED);
    if (tsr_rec_is_devweight_shape(d, r)) {
        r.fmt = tsr_dev_fmt_for_type(r.type);
        r.N   = (uint32_t) r.ne[1];
        r.K   = (uint32_t) r.ne[0];
        r.n_exp = (uint32_t) r.ne[2];
        r.exp_ch_bytes = tsr_tensor_ch_beats(r.fmt, r.N, r.K, d->cfg.num_ch) * TSR_BEAT_BYTES;
        r.ch_bytes = r.exp_ch_bytes * r.n_exp;
        if (!tsr_heap_alloc(d, r.ch_bytes, &r.w_off)) {
            GGML_ABORT("tesseract: WEIGHT heap full: weight '%s' needs %.1f MiB per channel, %.1f MiB free (largest %.1f MiB)",
                       r.name.c_str(), tsr_mib(r.ch_bytes), tsr_mib(d->heap.free_bytes()), tsr_mib(d->heap.largest_free()));
        }
        r.kind = tsr_kind::DEVWEIGHT;
        r.complete = false;
        tsr_mon_note_weight(d, r.type, r.ch_bytes * d->cfg.num_ch);
        tsr_mon_note_mem(d);
        TSR_DBG(1, "weight %-28s %-6s [%6u x %6u%s] -> %s W_OFF=0x%09" PRIx64 " (%.2f MiB/ch x %u ch)%s\n",
                r.name.c_str(), ggml_type_name(r.type), r.N, r.K, r.n_exp > 1 ? (" x " + std::to_string(r.n_exp)).c_str() : "",
                r.fmt == TSR_FMT_Q8_0 ? "Q8_0" : "Q4_K",
                r.w_off, tsr_mib(r.ch_bytes), d->cfg.num_ch, tsr_type_is_native(r.type) ? "" : " (converted)");
        return;
    }
    const uint64_t size = tsr_pad(std::max<size_t>(r.nbytes, 1), TSR_TENSOR_ALIGN);
    if (d->aux.alloc(size, &r.raw_off)) {
        r.raw_region = tsr_raw_region::AUX;
    } else if (tsr_heap_alloc(d, size, &r.raw_off)) {
        r.raw_region = tsr_raw_region::HEAP;
    } else {
        GGML_ABORT("tesseract: no device memory for tensor '%s' (%.1f MiB)", r.name.c_str(), tsr_mib(size));
    }
    r.kind = tsr_kind::RAW;
    tsr_mon_note_mem(d);
    r.taps = tsr_rec_is_conv_taps(r) ? (uint32_t) r.ne[0] : 0u;
    TSR_DBG(1, "raw    %-28s %-6s [%lld,%lld,%lld,%lld] -> %s ch0+0x%09" PRIx64 " (%zu B)%s\n",
            r.name.c_str(), ggml_type_name(r.type), (long long) r.ne[0], (long long) r.ne[1], (long long) r.ne[2],
            (long long) r.ne[3], r.raw_region == tsr_raw_region::AUX ? "AUX" : "HEAP", r.raw_off, r.nbytes,
            r.taps != 0 ? " (convolution taps, stored as tap rows)" : "");
}

static void tsr_rec_store_all(tsr_device * d, tsr_tensor_rec & r, const uint8_t * src) {
    tsr_store_expert_rows(d, r.type, r.fmt, r.N, r.K, r.exp_ch_bytes, r.w_off, src, 0, r.N * r.n_exp);
    r.complete = true;
    r.staging.clear();
    r.staging.shrink_to_fit();
    r.have = tsr_intervals();
}

// A tensor that is a Gated DeltaNet convolution kernel: F32 [d_conv, C], d_conv in 2 .. 8, C a multiple of 16,
// named *ssm_conv1d* (the weight of llama.cpp's qwen35 / qwen3next graphs, read by SSM_CONV).
static bool tsr_rec_is_conv_taps(const tsr_tensor_rec & r) {
    return r.type == GGML_TYPE_F32 && r.name.find("ssm_conv1d") != std::string::npos && r.ne[0] >= 2 &&
           r.ne[0] <= (int64_t) tsr::opl::sconv::D_CONV_MAX && r.ne[1] > 0 && r.ne[1] % 16 == 0 && r.ne[2] == 1 && r.ne[3] == 1 &&
           r.nb[0] == 4 && (uint64_t) r.nb[1] == 4 * (uint64_t) r.ne[0];
}

// ggml tap-fastest rows <-> tap rows (dev[w * C + c] = raw[c * d_conv + w])
static void tsr_taps_to_dev(const tsr_tensor_rec & r, const uint8_t * raw, uint8_t * dev) {
    const size_t dc = (size_t) r.ne[0], C = (size_t) r.ne[1];
    for (size_t c = 0; c < C; ++c) {
        for (size_t w = 0; w < dc; ++w) {
            memcpy(dev + 4 * (w * C + c), raw + 4 * (c * dc + w), 4);
        }
    }
}
static void tsr_taps_from_dev(const tsr_tensor_rec & r, const uint8_t * dev, uint8_t * raw) {
    const size_t dc = (size_t) r.ne[0], C = (size_t) r.ne[1];
    for (size_t c = 0; c < C; ++c) {
        for (size_t w = 0; w < dc; ++w) {
            memcpy(raw + 4 * (c * dc + w), dev + 4 * (w * C + c), 4);
        }
    }
}
static void tsr_taps_store(tsr_device * d, tsr_tensor_rec & r, const uint8_t * raw) {
    std::vector<uint8_t> dev(r.nbytes);
    tsr_taps_to_dev(r, raw, dev.data());
    tsr_write(d, 0, r.raw_off, dev.data(), r.nbytes);
    r.complete = true;
    r.staging.clear();
    r.staging.shrink_to_fit();
    r.have = tsr_intervals();
}
static void tsr_rec_read(tsr_device * d, const tsr_buffer_ctx * b, tsr_tensor_rec & r, uint64_t rel, uint8_t * dst, size_t n);

// Write bytes [rel, rel+n) of a tensor in a PLACED buffer.
static void tsr_rec_write(tsr_device * d, tsr_tensor_rec & r, uint64_t rel, const uint8_t * data, size_t n) {
    if (n == 0) {
        return;
    }
    if (r.kind == tsr_kind::UNPLACED) {
        tsr_place(d, r);
    }
    if (r.kind == tsr_kind::RAW && r.taps == 0) {
        tsr_write(d, 0, r.raw_off + rel, data, n);
        return;
    }
    if (r.kind == tsr_kind::RAW) {             // convolution taps: converted when every byte is here
        if (!r.complete) {
            if (rel == 0 && n == r.nbytes && r.staging.empty()) {
                tsr_taps_store(d, r, data);
                return;
            }
            if (r.staging.empty()) {
                r.staging.assign(r.nbytes, 0);
            }
            memcpy(r.staging.data() + rel, data, n);
            r.have.add(rel, rel + n);
            if (r.have.covers(0, r.nbytes)) {
                std::vector<uint8_t> full;
                full.swap(r.staging);
                tsr_taps_store(d, r, full.data());
            }
            return;
        }
        std::vector<uint8_t> raw(r.nbytes);        // a complete tensor updated in part
        tsr_rec_read(d, nullptr, r, 0, raw.data(), r.nbytes);
        memcpy(raw.data() + rel, data, n);
        tsr_taps_store(d, r, raw.data());
        return;
    }

    // DEVWEIGHT
    if (!r.complete) {
        if (rel == 0 && n == r.nbytes && r.staging.empty()) {
            tsr_rec_store_all(d, r, data);   // the common case: one set_tensor for the whole tensor
            return;
        }
        // split upload (spec "Split upload"): stage until every byte has arrived
        if (r.staging.empty()) {
            r.staging.assign(r.nbytes, 0);
        }
        memcpy(r.staging.data() + rel, data, n);
        r.have.add(rel, rel + n);
        if (r.have.covers(0, r.nbytes)) {
            std::vector<uint8_t> full;
            full.swap(r.staging);
            tsr_rec_store_all(d, r, full.data());
        }
        return;
    }

    // update of a complete weight: re-store the affected rows
    const size_t   src_rb = ggml_row_size(r.type, r.K);
    const uint32_t r0 = (uint32_t) (rel / src_rb);
    const uint32_t r1 = (uint32_t) ((rel + n + src_rb - 1) / src_rb);
    if (rel % src_rb == 0 && n % src_rb == 0) {
        tsr_store_expert_rows(d, r.type, r.fmt, r.N, r.K, r.exp_ch_bytes, r.w_off, data, r0, r1);
        return;
    }
    if (!tsr_type_is_native(r.type)) {
        GGML_ABORT("tesseract: partial-row update of weight '%s' (%s, converted to Q8_0 at load) is not supported: "
                   "the original bytes are not kept", r.name.c_str(), ggml_type_name(r.type));
    }
    std::vector<uint8_t> rows((size_t) (r1 - r0) * src_rb);
    tsr_load_expert_rows(d, r, r0, r1, rows.data());
    memcpy(rows.data() + (rel - (uint64_t) r0 * src_rb), data, n);
    tsr_store_expert_rows(d, r.type, r.fmt, r.N, r.K, r.exp_ch_bytes, r.w_off, rows.data(), r0, r1);
}

static void tsr_rec_read(tsr_device * d, const tsr_buffer_ctx * b, tsr_tensor_rec & r, uint64_t rel, uint8_t * dst, size_t n) {
    if (n == 0) {
        return;
    }
    switch (r.kind) {
        case tsr_kind::UNPLACED:
            memset(dst, b != nullptr ? b->clear_value : 0, n);
            return;
        case tsr_kind::RAW:
            if (r.taps == 0) {
                tsr_read(d, 0, r.raw_off + rel, dst, n);
                return;
            }
            if (!r.complete) {
                if (r.staging.empty()) {
                    memset(dst, b != nullptr ? b->clear_value : 0, n);
                } else {
                    memcpy(dst, r.staging.data() + rel, n);
                }
                return;
            } else {
                std::vector<uint8_t> dev(r.nbytes), raw(r.nbytes);
                tsr_read(d, 0, r.raw_off, dev.data(), r.nbytes);
                tsr_taps_from_dev(r, dev.data(), raw.data());
                memcpy(dst, raw.data() + rel, n);
                return;
            }
        case tsr_kind::DEVWEIGHT:
            break;
    }
    if (!r.complete) {
        if (r.staging.empty()) {
            memset(dst, b->clear_value, n);
        } else {
            memcpy(dst, r.staging.data() + rel, n);
        }
        return;
    }
    const size_t   src_rb = ggml_row_size(r.type, r.K);
    const uint32_t r0 = (uint32_t) (rel / src_rb);
    const uint32_t r1 = (uint32_t) ((rel + n + src_rb - 1) / src_rb);
    std::vector<uint8_t> rows((size_t) (r1 - r0) * src_rb);
    tsr_load_expert_rows(d, r, r0, r1, rows.data());
    memcpy(dst, rows.data() + (rel - (uint64_t) r0 * src_rb), n);
}

static tsr_tensor_rec * tsr_find_rec(tsr_buffer_ctx * b, uint64_t addr) {
    auto it = b->recs.upper_bound(addr);
    if (it == b->recs.begin()) {
        return nullptr;
    }
    --it;
    tsr_tensor_rec & r = it->second;
    if (addr >= r.vaddr && addr < r.vaddr + r.nbytes) {
        return &r;
    }
    return nullptr;
}

static void tsr_placed_write(tsr_device * d, tsr_buffer_ctx * b, uint64_t addr, const uint8_t * data, size_t size) {
    while (size > 0) {
        tsr_tensor_rec * r = tsr_find_rec(b, addr);
        if (r == nullptr) {
            GGML_ABORT("tesseract: write to 0x%" PRIx64 " in a weight buffer does not hit an initialized tensor", addr);
        }
        const uint64_t rel = addr - r->vaddr;
        const size_t   n   = (size_t) std::min<uint64_t>(size, r->nbytes - rel);
        tsr_rec_write(d, *r, rel, data, n);
        addr += n;
        data += n;
        size -= n;
    }
}

static void tsr_placed_read(tsr_device * d, tsr_buffer_ctx * b, uint64_t addr, uint8_t * dst, size_t size) {
    while (size > 0) {
        tsr_tensor_rec * r = tsr_find_rec(b, addr);
        if (r == nullptr) {
            GGML_ABORT("tesseract: read from 0x%" PRIx64 " in a weight buffer does not hit an initialized tensor", addr);
        }
        const uint64_t rel = addr - r->vaddr;
        const size_t   n   = (size_t) std::min<uint64_t>(size, r->nbytes - rel);
        tsr_rec_read(d, b, *r, rel, dst, n);
        addr += n;
        dst  += n;
        size -= n;
    }
}

static void tsr_placed_fill(tsr_device * d, tsr_buffer_ctx * b, uint64_t addr, uint8_t value, size_t size) {
    std::vector<uint8_t> buf;
    while (size > 0) {
        tsr_tensor_rec * r = tsr_find_rec(b, addr);
        if (r == nullptr) {
            GGML_ABORT("tesseract: memset at 0x%" PRIx64 " in a weight buffer does not hit an initialized tensor", addr);
        }
        const uint64_t rel = addr - r->vaddr;
        const size_t   n   = (size_t) std::min<uint64_t>(size, r->nbytes - rel);
        if (r->kind == tsr_kind::UNPLACED) {
            tsr_place(d, *r);
        }
        if (r->kind == tsr_kind::DEVWEIGHT && !(r->complete || (rel == 0 && n == r->nbytes))) {
            // part of a weight that is not complete: goes through the staging path
            buf.assign(n, value);
            tsr_rec_write(d, *r, rel, buf.data(), n);
        } else if (r->kind == tsr_kind::DEVWEIGHT) {
            // whole rows in bounded chunks (complete weight, or the whole tensor at once)
            const size_t src_rb = ggml_row_size(r->type, r->K);
            const bool   whole  = rel == 0 && n == r->nbytes;
            if (whole && !r->complete) {
                r->complete = true;   // every byte is defined by this fill
                r->staging.clear();
                r->have = tsr_intervals();
            }
            if (rel % src_rb == 0 && n % src_rb == 0) {
                const size_t rows_per = std::max<size_t>(1, TSR_XFER_CHUNK / src_rb);
                buf.assign(std::min<size_t>(rows_per, n / src_rb) * src_rb, value);
                for (size_t done = 0; done < n;) {
                    const size_t m  = std::min(buf.size(), n - done);
                    const uint32_t r0 = (uint32_t) ((rel + done) / src_rb);
                    tsr_store_weight_rows(d, r->type, r->fmt, r->N, r->K, r->w_off, buf.data(), r0, r0 + (uint32_t) (m / src_rb));
                    done += m;
                }
            } else {
                buf.assign(n, value);
                tsr_rec_write(d, *r, rel, buf.data(), n);
            }
        } else if (r->taps != 0) {
            buf.assign(n, value);
            tsr_rec_write(d, *r, rel, buf.data(), n);
        } else {
            tsr_fill(d, 0, r->raw_off + rel, value, n);
        }
        addr += n;
        size -= n;
    }
}

// ---------------------------------------------------------------------------
// W3b kv: KV-cache buffers and the mask-summary store (SW §9.3 M7 "KV buffers" / "Mask summaries",
// SW-17, SW-32; AT §4.6, §7.3; tsr-kvbuf.h, tsr-msum.h). Callers hold dev->mtx, device open.
// ---------------------------------------------------------------------------

// The mask-summary store, set up at its first use: the chunk planner's AT parameters (C, T_MAX,
// PCAP = CFG3[31:16] x 16, K_MAX: the values the firmware reads per launch, SW §7.3) and side
// allocations (the f32 copy of a further FA scale, all-visible MSUMs) from the WEIGHT heap, without
// evicting lazily converted weights (a pending op list may reference them).
static tsr_msum_store & tsr_mask_store(tsr_device * d) {
    if (!d->msum.ready()) {
        const uint32_t cfg3 = tsr_transport_csr_read(d->tp, TSR_CSR_CFG3);
        d->att_cfg = { d->cfg.num_ch, d->cfg.t_max, ((cfg3 & TSR_CFG3_PCAP16) >> TSR_CFG3_PCAP16_LSB) * 16u, d->cfg.k_max };
        d->msum.init(d->att_cfg,
            [d](uint32_t ch, uint64_t off, const void * p, size_t n) { tsr_write(d, ch, off, p, n); },
            [d](uint64_t bytes, uint32_t * ch, uint64_t * off) { *ch = 0; return d->heap.alloc(bytes, off); },
            [d](uint32_t ch, uint64_t off) { GGML_UNUSED(ch); d->heap.release(off); });
        TSR_DBG(1, "mask summaries: AT C=%u T_MAX=%u PCAP=%u K_MAX=%u\n", d->att_cfg.C, d->att_cfg.t_max, d->att_cfg.pcap,
                d->att_cfg.k_max);
    }
    return d->msum;
}

// A buffer whose tensors (init_tensor records) are all KV-cache tensors the device can hold (AT
// §4.6: cache_<tag>k_l<n> / cache_<tag>v_l<n>, q8_0, [E, kv_size, n_stream]; tsr_kv_identify).
static bool tsr_kv_candidate(const tsr_device * d, const tsr_buffer_ctx * b) {
    if (b->recs.empty()) {
        return false;
    }
    for (const auto & kv : b->recs) {
        const tsr_tensor_rec & r = kv.second;
        if (!tsr_kv_identify(r.name.c_str(), r.type == GGML_TYPE_Q8_0, r.ne, r.nb, d->cfg.num_ch, d->cfg.k_max, nullptr, nullptr)) {
            return false;
        }
    }
    return true;
}

// Make b a KV buffer: one plane per (tensor, stream), top-down in the WEIGHT heap (AT §4.5, EX §11).
// Called at the buffer's first data access (llama.cpp clears a KV buffer right after allocating it).
static void tsr_kv_place(tsr_device * d, tsr_buffer_ctx * b) {
    b->kv.reset(new tsr_kvbuf(d->cfg.num_ch, &d->heap,
        [d](uint32_t ch, uint64_t off, void * p, size_t n) { tsr_read(d, ch, off, p, n); },
        [d](uint32_t ch, uint64_t off, const void * p, size_t n) { tsr_write(d, ch, off, p, n); }));
    for (const auto & kv : b->recs) {
        const tsr_tensor_rec & r = kv.second;
        tsr_kv_shape s;
        tsr_kv_identify(r.name.c_str(), true, r.ne, r.nb, d->cfg.num_ch, d->cfg.k_max, &s, nullptr);
        b->kv->add(r.name, r.vaddr, r.nbytes, s);
    }
    std::string err;
    if (!b->kv->place(&err)) {
        tsr_lazy_evict_all(d);             // a cache of converted weights: drop it and retry
        if (!b->kv->place(&err)) {
            GGML_ABORT("tesseract: the KV cache does not fit the device: %s (reduce the context, or keep the cache on the "
                       "CPU with -nkvo)", err.c_str());
        }
    }
    b->mode = tsr_bmode::KV;
    b->recs.clear();
    TSR_DBG(1, "buffer #%" PRIu64 " (%.2f MiB): KV cache, %zu tensors -> KV planes, %.2f MiB per channel x %u ch\n",
            b->id, tsr_mib(b->size), b->kv->tensors().size(), tsr_mib(b->kv->device_bytes_per_ch()), d->cfg.num_ch);
    for (const tsr_kv_tensor & t : b->kv->tensors()) {
        for (const tsr_kv_plane & p : t.planes) {
            TSR_DBG(2, "  plane %-24s stream %u: %c E=%u LG=%u (%u cells) B=0x%07x (%.2f MiB/ch)\n", t.name.c_str(), p.stream,
                    p.which == TSR_KV_K ? 'K' : 'V', p.geom.E, p.geom.LG, (unsigned) tsr_kv_plane_cells(&p.geom),
                    p.base_beat(), tsr_mib(p.bytes));
        }
    }
}

// ---------------------------------------------------------------------------
// LINEAR buffers (IO window)
// ---------------------------------------------------------------------------

static const char * tsr_usage_name(enum ggml_backend_buffer_usage u) {
    switch (u) {
        case GGML_BACKEND_BUFFER_USAGE_ANY:     return "ANY";
        case GGML_BACKEND_BUFFER_USAGE_WEIGHTS: return "WEIGHTS";
        case GGML_BACKEND_BUFFER_USAGE_COMPUTE: return "COMPUTE";
    }
    return "?";
}

// W3b builder: compute windows (SW §8 R6, SW-26, EX §11). A buffer the device addresses in place
// (LINEAR) gets one contiguous window in one channel, tried in this order: the ch0 IO region (below
// the stage), the low 512 MiB of ch1..ch(C-1) (the channel with the most free bytes first, which
// spreads activation traffic over the channels), the compute heap. The compute heap takes symmetric
// reservations of the WEIGHT heap (bottom-up, above the weights llama.cpp loads first; the KV planes
// grow top-down, AT §4.5) of at least 64 MiB and shares each one out per channel; a reservation that
// is empty on every channel goes back. Only when nothing fits is a buffer kept in host memory (its
// MUL_MATs staged, one extra launch per chunk: TSR_STATS "staged chunks").
enum : uint8_t { TSR_WIN_NONE = 0, TSR_WIN_IO = 1, TSR_WIN_LOW = 2, TSR_WIN_HEAP = 3 };
static constexpr uint64_t TSR_CHEAP_GRAIN = 64ull << 20;

// TSR_COMPUTE_WINDOWS=io: ch0 IO windows only (M6 placement: a buffer that does not fit IO is
// kept in host memory and its MUL_MATs staged; get_max_size the IO capacity). Test knob.
static bool tsr_io_windows_only() {
    static const bool on = [] {
        const char * e = getenv("TSR_COMPUTE_WINDOWS");
        return e != nullptr && strcmp(e, "io") == 0;
    }();
    return on;
}

static bool tsr_window_alloc(tsr_device * d, uint64_t size, uint32_t * ch, uint64_t * off, uint8_t * win, int32_t * res) {
    if (d->io.alloc(size, off)) {
        *ch = 0;
        *win = TSR_WIN_IO;
        return true;
    }
    if (tsr_io_windows_only()) {
        return false;
    }
    // low windows: the channel with the most free bytes that fits
    int best = -1;
    for (uint32_t c = 1; c < (uint32_t) d->low.size(); ++c) {
        if (d->low[c].largest_free() >= tsr_pad(size, TSR_TENSOR_ALIGN) &&
            (best < 0 || d->low[c].free_bytes() > d->low[(size_t) best].free_bytes())) {
            best = (int) c;
        }
    }
    if (best > 0 && d->low[(size_t) best].alloc(size, off)) {
        *ch = (uint32_t) best;
        *win = TSR_WIN_LOW;
        return true;
    }
    // the compute heap: an existing reservation on any channel, else a new one
    for (size_t r = 0; r < d->cheap.size(); ++r) {
        for (uint32_t c = 0; c < (uint32_t) d->cheap[r].ch.size(); ++c) {
            if (d->cheap[r].ch[c].alloc(size, off)) {
                *ch = c;
                *win = TSR_WIN_HEAP;
                *res = (int32_t) r;
                return true;
            }
        }
    }
    tsr_device::cheap_res nr;
    nr.size = tsr_pad(std::max<uint64_t>(size, 1), TSR_CHEAP_GRAIN);
    if (!d->heap.alloc(nr.size, &nr.off)) {
        return false;
    }
    nr.ch.assign(d->cfg.num_ch, tsr_range_alloc());
    for (tsr_range_alloc & a : nr.ch) {
        a.init(nr.off, nr.size, TSR_TENSOR_ALIGN);
    }
    if (!nr.ch[0].alloc(size, off)) {
        d->heap.release(nr.off);
        return false;
    }
    TSR_DBG(1, "compute heap: reserved %.0f MiB at +0x%09" PRIx64 " on every channel (WEIGHT heap %.0f MiB free)\n",
            tsr_mib(nr.size), nr.off, tsr_mib(d->heap.free_bytes()));
    d->cheap.push_back(std::move(nr));
    *ch = 0;
    *win = TSR_WIN_HEAP;
    *res = (int32_t) d->cheap.size() - 1;
    return true;
}

static void tsr_window_release(tsr_device * d, tsr_buffer_ctx * b) {
    if (b->win == TSR_WIN_IO) {
        d->io.release(b->io_off);
    } else if (b->win == TSR_WIN_LOW) {
        d->low[b->ch].release(b->io_off);
    } else if (b->win == TSR_WIN_HEAP && b->win_res >= 0 && (size_t) b->win_res < d->cheap.size()) {
        tsr_device::cheap_res & r = d->cheap[(size_t) b->win_res];
        r.ch[b->ch].release(b->io_off);
        bool empty = true;
        for (const tsr_range_alloc & a : r.ch) {
            empty = empty && a.used() == 0;
        }
        if (empty && (size_t) b->win_res + 1 == d->cheap.size()) {   // the last one: give it back (indices stay valid)
            d->heap.release(r.off);
            d->cheap.pop_back();
        }
    }
    b->win = TSR_WIN_NONE;
}

// Fix the buffer's storage mode at its first data access (llama.cpp sets WEIGHTS usage right
// after allocating a weight buffer and before uploading; compute buffers get COMPUTE usage
// right after allocation; init_tensor runs while usage is still ANY).
//   WEIGHTS                       -> PLACED (per tensor)
//   ANY holding only leaf tensors -> HOSTED: data that no op list reads or writes in place
//                                    (LoRA adapters, control vectors, a KV cache without -nkvo);
//                                    a MUL_MAT src0 among them is converted into the heap
//   otherwise (ANY, COMPUTE)      -> LINEAR if a window of the IO region is free, else HOSTED
//                                    (graph_compute stages its MUL_MAT rows through IO)
// An ANY buffer whose recorded tensors are all leaf data (no op result, no input): it becomes HOSTED
// (tsr_buf_decide); supports_op asks it ahead of the decision (tsr_probe_raw_ok).
// llama.cpp's recurrent memory (llama-memory-recurrent.cpp) names the per-layer state tensors cache_r_l<N> (the
// convolution state) and cache_s_l<N> (the Gated DeltaNet state). The graph reads and writes them in place
// (GET_ROWS of a cell, SCALE of a cell to zero it, CPY of the new state into a cell, the SSM_CONV and
// GATED_DELTA_NET records), so a buffer that holds them is a device window, not leaf data (SW §7.10).
static bool tsr_is_state_name(const std::string & n) {
    return (n.compare(0, 9, "cache_r_l") == 0 || n.compare(0, 9, "cache_s_l") == 0) && n.size() > 9 && n[9] >= '0' && n[9] <= '9';
}

static bool tsr_buf_leaves_only(const tsr_buffer_ctx * b, const ggml_backend_buffer * gbuf) {
    bool leaves_only = gbuf->usage == GGML_BACKEND_BUFFER_USAGE_ANY && !b->recs.empty() && !b->computed_view;
    for (const auto & kv : b->recs) {
        leaves_only = leaves_only && kv.second.leaf && !tsr_is_state_name(kv.second.name);
    }
    return leaves_only;
}

static void tsr_buf_decide(tsr_buffer_ctx * b, ggml_backend_buffer_t gbuf) {
    if (b->mode != tsr_bmode::UNDECIDED) {
        return;
    }
    tsr_device * d = b->dev;
    if (gbuf->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        b->mode = tsr_bmode::PLACED;
        TSR_DBG(1, "buffer #%" PRIu64 " (%.2f MiB): WEIGHTS -> per-tensor placement\n", b->id, tsr_mib(b->size));
        return;
    }
    // W3b kv: only KV-cache tensors (an offloaded llama.cpp KV cache, -ctk q8_0 -ctv q8_0) -> KV planes
    if (gbuf->usage == GGML_BACKEND_BUFFER_USAGE_ANY && tsr_kv_candidate(d, b)) {
        tsr_kv_place(d, b);
        return;
    }
    const bool leaves_only = tsr_buf_leaves_only(b, gbuf);
    uint64_t off = 0;
    uint32_t ch = 0;
    uint8_t win = TSR_WIN_NONE;
    int32_t res = -1;
    if (!d->force_host && !leaves_only && tsr_window_alloc(d, b->size, &ch, &off, &win, &res)) {
        b->mode    = tsr_bmode::LINEAR;
        b->has_io  = true;
        b->io_off  = off;
        b->ch      = ch;
        b->win     = win;
        b->win_res = res;
        b->recs.clear();
        TSR_DBG(1, "buffer #%" PRIu64 " (%.2f MiB, %s): %s window ch%u+0x%09" PRIx64 "\n", b->id, tsr_mib(b->size),
                tsr_usage_name(gbuf->usage), win == TSR_WIN_IO ? "IO" : win == TSR_WIN_LOW ? "low" : "compute-heap", ch, off);
        return;
    }
    b->host = (uint8_t *) calloc((size_t) std::max<uint64_t>(b->size, 1), 1);
    if (b->host == nullptr) {
        GGML_ABORT("tesseract: out of host memory for a %s buffer of %.2f MiB", tsr_usage_name(gbuf->usage), tsr_mib(b->size));
    }
    b->mode = tsr_bmode::HOSTED;
    b->recs.clear();
    TSR_DBG(1, "buffer #%" PRIu64 " (%.2f MiB, %s): host memory (%s)\n", b->id, tsr_mib(b->size),
            tsr_usage_name(gbuf->usage),
            leaves_only ? "leaf tensors only" : d->force_host ? "TSR_FORCE_HOST" : "no free IO window, staged");
    if (!leaves_only && !d->force_host) {
        // not an error for MUL_MAT: the MUL_MATs that use it are staged through IO in T tiles; the M7
        // kinds address their operands in place only (graph_compute aborts on one, naming the buffer)
        GGML_LOG_INFO("tesseract: %s buffer #%" PRIu64 " of %.2f MiB does not fit a device window (largest free IO "
                      "window %.2f MiB of %.2f MiB): kept in host memory, MUL_MAT activations in it pass through the "
                      "%.0f MiB IO stage; the other device kinds cannot use it (SW §8 R6)\n", tsr_usage_name(gbuf->usage),
                      b->id, tsr_mib(b->size), tsr_mib(d->io.largest_free()), tsr_mib(d->io.total()), tsr_mib(d->stage_size));
    }
}

// Storage of a buffer for the test hook (get_proc_address "ggml_backend_tesseract_buffer_storage").
static const char * tsr_bmode_name(tsr_bmode m) {
    switch (m) {
        case tsr_bmode::UNDECIDED: return "undecided";
        case tsr_bmode::LINEAR:    return "io";
        case tsr_bmode::PLACED:    return "placed";
        case tsr_bmode::HOSTED:    return "host";
        case tsr_bmode::KV:        return "kv";     // W3b kv
    }
    return "?";
}

static uint64_t tsr_vaddr(const void * p) {
    return (uint64_t) (uintptr_t) p;
}

static void tsr_check_range(const tsr_buffer_ctx * b, uint64_t addr, size_t size) {
    if (addr < b->base || addr + size > b->base + b->size) {
        GGML_ABORT("tesseract: access [0x%" PRIx64 ", +%zu) outside buffer #%" PRIu64 " [0x%" PRIx64 ", +%" PRIu64 ")",
                   addr, size, b->id, b->base, b->size);
    }
}

// ---------------------------------------------------------------------------
// W3b kv: the mask-summary producer hook, helpers for the M7 builder and supports_op, test hooks
// (tsr-kvbuf.h, tsr-msum.h). Callers hold dev->mtx (the test hooks take it).
// ---------------------------------------------------------------------------

// Device channel and byte offset of a tensor byte in a buffer the device addresses in place (LINEAR:
// its IO window on ch0). False for host-memory, placed and KV buffers.
static bool tsr_dev_loc(const tsr_buffer_ctx * b, uint64_t vaddr, uint32_t * ch, uint64_t * off) {
    if (b->mode != tsr_bmode::LINEAR) {
        return false;
    }
    *ch  = b->ch;                      // W3b builder: the window's channel (compute heap, SW §8 R6)
    *off = b->io_off + (vaddr - b->base);
    return true;
}

// After set_tensor / memset_tensor of a KQ mask (name *kq_mask*, F16 [n_kv, n_q, 1, 1]) in device
// memory: its MSUM and the f32 copies its registered FA readers need, written into the bytes
// get_alloc_size reserved behind it (AT §7.3, SW §9.3). data: the bytes just written at `offset`
// (nullptr, or a partial write: the mask is read back).
static void tsr_msum_on_write(tsr_device * d, tsr_buffer_ctx * b, const ggml_tensor * t, const void * data,
                              size_t offset, size_t size) {
    if (t->view_src != nullptr || !tsr_msum_is_mask_name(t->name)) {
        return;
    }
    tsr_msum_layout L;
    if (!tsr_msum_layout_of(t->name, t->type == GGML_TYPE_F16, t->ne, t->nb, ggml_nbytes(t), &L)) {
        return;
    }
    const uint64_t a = tsr_vaddr(t->data);
    uint32_t ch = 0;
    uint64_t off = 0;
    if (!tsr_dev_loc(b, a, &ch, &off)) {
        return;                            // host memory: no FLASH_ATTN can read this mask in place
    }
    if (a - b->base + L.total > b->size) {
        static std::once_flag warned;      // allocated without this buffer type's get_alloc_size
        std::call_once(warned, [t] {
            GGML_LOG_WARN("tesseract: mask '%s' has no room for its mask summary (allocated without the buffer type's "
                          "get_alloc_size); FLASH_ATTN cannot read it on the device\n", t->name);
        });
        return;
    }
    std::vector<uint16_t> tmp;
    const uint16_t * f16 = (const uint16_t *) data;
    if (data == nullptr || offset != 0 || size != ggml_nbytes(t)) {
        tmp.resize(ggml_nbytes(t) / 2);
        tsr_read(d, ch, off, tmp.data(), ggml_nbytes(t));
        f16 = tmp.data();
    }
    tsr_mask_store(d).on_set(a, ch, off, L, f16);
}

// The M7 builder (SW §9.3; SET_ROWS_KV words 0, 8-11, FLASH_ATTN words 5-6, 16-18): the KV plane a
// tensor lies in (a view of a cache tensor: cpy_k / cpy_v's SET_ROWS dst, get_k / get_v), with its
// stream and first cell. False if it is not in a placed KV buffer.
[[maybe_unused]] static bool tsr_kv_view_of(tsr_device * d, const ggml_tensor * t, tsr_kv_view * out) {
    const ggml_tensor * base = t->view_src != nullptr ? t->view_src : t;
    const ggml_backend_buffer * gb = base->buffer;
    if (gb == nullptr || gb->buft != &d->buft || gb->context == nullptr || t->data == nullptr) {
        return false;
    }
    const tsr_buffer_ctx * b = (const tsr_buffer_ctx *) gb->context;
    if (b->mode != tsr_bmode::KV || !b->kv) {
        return false;
    }
    return b->kv->resolve(tsr_vaddr(t->data), out);
}

// supports_op (SW §9.3 M7 table: SET_ROWS dst / FLASH_ATTN k, v): `base` (the tensor they view) is a
// KV-cache tensor (tsr_kv_identify) in a Tesseract buffer that is a KV buffer or becomes one at its
// first data access (every record is a KV-cache tensor; no records: llama.cpp's no_alloc probe
// buffer, whose tensors are never initialized). shape (optional): its plane geometry.
[[maybe_unused]] static bool tsr_kv_tensor_supported(tsr_device * d, const ggml_tensor * base, tsr_kv_shape * shape = nullptr) {
    if (!tsr_kv_identify(base->name, base->type == GGML_TYPE_Q8_0, base->ne, base->nb, d->cfg.num_ch, d->cfg.k_max,
                         shape, nullptr)) {
        return false;
    }
    const ggml_backend_buffer * gb = base->buffer;
    if (gb == nullptr || gb->buft != &d->buft || gb->context == nullptr) {
        return false;
    }
    const tsr_buffer_ctx * b = (const tsr_buffer_ctx *) gb->context;
    if (b->mode == tsr_bmode::KV) {
        return true;
    }
    if (b->mode != tsr_bmode::UNDECIDED || gb->usage != GGML_BACKEND_BUFFER_USAGE_ANY) {
        return false;
    }
    return b->recs.empty() || tsr_kv_candidate(d, b);
}

// supports_op: FLASH_ATTN_EXT's mask (src[3]) is one the producer summarizes (name *kq_mask*, F16
// [n_kv, n_q, 1, 1] with n_kv = K's cells and n_q = Q's tokens: the MSUM's n_kv / n_q must equal
// FLASH_ATTN words 7 / 11), or there is none.
[[maybe_unused]] static bool tsr_fa_mask_supported(const ggml_tensor * fa) {
    const ggml_tensor * m = fa->src[3];
    if (m == nullptr) {
        return true;
    }
    tsr_msum_layout L;
    return m->view_src == nullptr && tsr_msum_layout_of(m->name, m->type == GGML_TYPE_F16, m->ne, m->nb, ggml_nbytes(m), &L) &&
           m->ne[0] == fa->src[1]->ne[1] && m->ne[1] == fa->src[0]->ne[1];
}

// The M7 builder: the mask-summary words of a FLASH_ATTN_EXT node (FLASH_ATTN word 12 via
// writer.pre_msum(ref.msum_handle, ref.n_q), word 13 = ref.mask_handle; the record reads
// ref.copy_range). Registers the node as a reader of its mask (scale = op_params[0], E = D·n_head_kv,
// D, G = n_head / n_head_kv); no mask: an all-visible MSUM. valid = false: no summary on the device.
[[maybe_unused]] static tsr_msum_ref tsr_fa_mask_ref(tsr_device * d, const ggml_tensor * fa) {
    const ggml_tensor * q = fa->src[0];
    const ggml_tensor * k = fa->src[1];
    const ggml_tensor * m = fa->src[3];
    if (m == nullptr) {
        return tsr_mask_store(d).nomask((uint32_t) q->ne[1], (uint32_t) k->ne[1]);
    }
    if (m->view_src != nullptr || m->data == nullptr || k->ne[2] <= 0) {
        return tsr_msum_ref();
    }
    tsr_msum_reader r;
    memcpy(&r.scale_bits, &fa->op_params[0], sizeof(uint32_t));
    r.D = (uint32_t) k->ne[0];
    r.E = (uint32_t) (k->ne[0] * k->ne[2]);
    r.G = (uint32_t) (q->ne[2] / k->ne[2]);
    return tsr_mask_store(d).attach(tsr_vaddr(m->data), r);
}

// Test hook "ggml_backend_tesseract_kv_plane": the KV plane of t (a cache tensor or a view of one):
// info = {which, stream, cell0, base beat, GEOM word, E, LG, kv_size, C}; 0, or -1 if t is not in a
// KV buffer.
static int tsr_hook_kv_plane(const ggml_tensor * t, uint32_t * info) {
    tsr_device * d = tsr_dev_singleton();
    std::lock_guard<std::mutex> lock(d->mtx);
    tsr_kv_view v;
    if (!tsr_kv_view_of(d, t, &v)) {
        return -1;
    }
    const uint32_t w[9] = { v.plane->which, v.stream, v.cell0, v.plane->base_beat(), v.plane->geom_word(), v.plane->geom.E,
                            v.plane->geom.LG, v.plane->kv_size, v.plane->geom.C };
    memcpy(info, w, sizeof w);
    return 0;
}

// Test hook "ggml_backend_tesseract_att_cfg": the AT parameters the producer plans with (= what the
// firmware reads per launch): out = {C, T_MAX, PCAP, K_MAX}; 0, or -1 without a device.
static int tsr_hook_att_cfg(uint32_t * out) {
    tsr_device * d = tsr_dev_singleton();
    std::lock_guard<std::mutex> lock(d->mtx);
    if (!tsr_dev_open_locked(d)) {
        return -1;
    }
    tsr_mask_store(d);
    const uint32_t w[4] = { d->att_cfg.C, d->att_cfg.t_max, d->att_cfg.pcap, d->att_cfg.k_max };
    memcpy(out, w, sizeof w);
    return 0;
}

// Test hook "ggml_backend_tesseract_mask_summary": attaches the FLASH_ATTN_EXT node fa as the builder
// does (tsr_fa_mask_ref) and reads back from the device its MSUM (up to n_words words) and, when copy
// is not NULL, its scale's f32 copy (up to n_floats); returns the MSUM's word count, 0 without one.
static size_t tsr_hook_mask_summary(const ggml_tensor * fa, uint32_t * words, size_t n_words, float * copy, size_t n_floats) {
    tsr_device * d = tsr_dev_singleton();
    std::lock_guard<std::mutex> lock(d->mtx);
    if (!tsr_dev_open_locked(d)) {
        return 0;
    }
    const tsr_msum_ref r = tsr_fa_mask_ref(d, fa);
    if (!r.valid) {
        return 0;
    }
    const size_t nw = tsr_msum_words(r.n_q);
    tsr_read(d, r.msum_range.ch, r.msum_range.lo, words, std::min(nw, n_words) * 4u);
    if (copy != nullptr && r.mask_handle != tsr::opl::HANDLE_NONE) {
        const size_t nf = (size_t) ((r.copy_range.hi - r.copy_range.lo) / 4u);
        tsr_read(d, r.copy_range.ch, r.copy_range.lo, copy, std::min(nf, n_floats) * 4u);
    }
    return nw;
}

// ---------------------------------------------------------------------------
// ggml buffer interface
// ---------------------------------------------------------------------------

static void tsr_buffer_free(ggml_backend_buffer_t buffer) {
    tsr_buffer_ctx * b = tsr_buf(buffer);
    tsr_device * d = b->dev;
    {
        std::lock_guard<std::mutex> lock(d->mtx);
        if (b->has_io) {
            tsr_window_release(d, b);     // W3b builder: IO, a low window or the compute heap
        }
        tsr_plans_invalidate(d);          // W3b builder: cached plans may name this buffer (R7)
        for (auto & kv : b->recs) {
            tsr_rec_release(d, kv.second);
        }
        tsr_lazy_evict_buffer(d, b->id);
        b->kv.reset();                                    // W3b kv: the KV planes go back to the heap
        if (d->msum.ready()) {
            d->msum.forget(b->base, b->base + b->size);   // W3b kv: summaries of masks in this buffer
        }
        TSR_DBG(2, "buffer #%" PRIu64 " freed\n", b->id);
    }
    free(b->host);
    delete b;
}

static void * tsr_buffer_get_base(ggml_backend_buffer_t buffer) {
    return (void *) (uintptr_t) tsr_buf(buffer)->base;
}

static enum ggml_status tsr_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    tsr_buffer_ctx * b = tsr_buf(buffer);
    if (tensor->view_src != nullptr) {
        if (tensor->op != GGML_OP_NONE) {
            // W3b builder: an in-place op's result (a view, e.g. test-backend-ops' inplace cases) makes
            // the buffer one the device computes into, not leaf data (tsr_buf_decide)
            std::lock_guard<std::mutex> lock(b->dev->mtx);
            b->computed_view = true;
        }
        return GGML_STATUS_SUCCESS;     // views resolve by address to their base tensor
    }
    if (buffer->usage == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        return GGML_STATUS_SUCCESS;     // LINEAR storage needs no per-tensor state
    }
    std::lock_guard<std::mutex> lock(b->dev->mtx);
    if (b->mode == tsr_bmode::LINEAR || b->mode == tsr_bmode::HOSTED) {
        return GGML_STATUS_SUCCESS;     // raw storage needs no per-tensor state
    }
    if (b->mode == tsr_bmode::KV) {
        return GGML_STATUS_SUCCESS;     // W3b kv: the planes were fixed at the first data access
    }
    const uint64_t a = tsr_vaddr(tensor->data);
    const size_t nbytes = ggml_nbytes(tensor);
    tsr_check_range(b, a, nbytes);

    // drop records this tensor overlaps (re-initialization of a reused buffer)
    for (auto it = b->recs.begin(); it != b->recs.end();) {
        const tsr_tensor_rec & r = it->second;
        const bool overlap = a < r.vaddr + std::max<size_t>(r.nbytes, 1) && r.vaddr < a + std::max<size_t>(nbytes, 1);
        if (overlap) {
            const bool same = r.vaddr == a && r.type == tensor->type && r.nbytes == nbytes &&
                              memcmp(r.ne, tensor->ne, sizeof(r.ne)) == 0;
            if (same) {
                return GGML_STATUS_SUCCESS;
            }
            tsr_rec_release(b->dev, it->second);
            it = b->recs.erase(it);
        } else {
            ++it;
        }
    }
    if (nbytes == 0) {
        return GGML_STATUS_SUCCESS;
    }
    tsr_tensor_rec & r = b->recs[a];
    r.name   = tensor->name;
    r.type   = tensor->type;
    memcpy(r.ne, tensor->ne, sizeof(r.ne));
    memcpy(r.nb, tensor->nb, sizeof(r.nb));
    r.nbytes = nbytes;
    r.vaddr  = a;
    r.leaf   = tensor->op == GGML_OP_NONE && (tensor->flags & GGML_TENSOR_FLAG_INPUT) == 0;
    return GGML_STATUS_SUCCESS;
}

static void tsr_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    tsr_buffer_ctx * b = tsr_buf(buffer);
    tsr_device * d = b->dev;
    std::lock_guard<std::mutex> lock(d->mtx);
    tsr_buf_decide(b, buffer);
    const uint64_t addr = tsr_vaddr(tensor->data) + offset;
    tsr_check_range(b, addr, size);
    b->gen++;
    const uint64_t t0 = tsr_stats_on() ? tsr_now_ns() : 0;
    if (b->mode == tsr_bmode::LINEAR) {
        tsr_write(d, b->ch, b->io_off + (addr - b->base), data, size);
    } else if (b->mode == tsr_bmode::HOSTED) {
        memcpy(b->host + (addr - b->base), data, size);
    } else if (b->mode == tsr_bmode::KV) {
        b->kv->set(addr, data, size);      // W3b kv: ggml q8_0 rows -> KV planes (AT §4.6)
    } else {
        tsr_placed_write(d, b, addr, (const uint8_t *) data, size);
    }
    tsr_msum_on_write(d, b, tensor, data, offset, size);   // W3b kv: a KQ mask's summary (AT §7.3)
    if (tsr_stats_on()) {
        d->st_set_ns += tsr_now_ns() - t0;
        d->st_set_bytes += size;
    }
}

static void tsr_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    tsr_buffer_ctx * b = tsr_buf(buffer);
    tsr_device * d = b->dev;
    std::lock_guard<std::mutex> lock(d->mtx);
    tsr_buf_decide(b, buffer);
    const uint64_t addr = tsr_vaddr(tensor->data) + offset;
    tsr_check_range(b, addr, size);
    if (b->mode == tsr_bmode::LINEAR) {
        tsr_read(d, b->ch, b->io_off + (addr - b->base), data, size);
    } else if (b->mode == tsr_bmode::HOSTED) {
        memcpy(data, b->host + (addr - b->base), size);
    } else if (b->mode == tsr_bmode::KV) {
        b->kv->get(addr, data, size);      // W3b kv: KV planes -> ggml q8_0 rows (AT §4.6)
    } else {
        tsr_placed_read(d, b, addr, (uint8_t *) data, size);
    }
}

static void tsr_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    tsr_buffer_ctx * b = tsr_buf(buffer);
    tsr_device * d = b->dev;
    std::lock_guard<std::mutex> lock(d->mtx);
    tsr_buf_decide(b, buffer);
    const uint64_t addr = tsr_vaddr(tensor->data) + offset;
    tsr_check_range(b, addr, size);
    b->gen++;
    if (b->mode == tsr_bmode::LINEAR) {
        tsr_fill(d, b->ch, b->io_off + (addr - b->base), value, size);
    } else if (b->mode == tsr_bmode::HOSTED) {
        memset(b->host + (addr - b->base), value, size);
    } else if (b->mode == tsr_bmode::KV) {
        b->kv->fill(addr, value, size);    // W3b kv
    } else {
        tsr_placed_fill(d, b, addr, value, size);
    }
    tsr_msum_on_write(d, b, tensor, nullptr, offset, size);   // W3b kv: a KQ mask's summary
}

static void tsr_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    tsr_buffer_ctx * b = tsr_buf(buffer);
    tsr_device * d = b->dev;
    std::lock_guard<std::mutex> lock(d->mtx);
    tsr_buf_decide(b, buffer);
    b->gen++;
    if (b->mode == tsr_bmode::KV) {
        b->kv->clear(value);               // W3b kv: every plane (llama.cpp clears a KV buffer to 0)
        return;
    }
    if (d->msum.ready()) {
        d->msum.forget(b->base, b->base + b->size);   // W3b kv: masks in it no longer match their summaries
    }
    if (b->mode == tsr_bmode::LINEAR) {
        tsr_fill(d, b->ch, b->io_off, value, b->size);
        return;
    }
    if (b->mode == tsr_bmode::HOSTED) {
        memset(b->host, value, (size_t) b->size);
        return;
    }
    b->clear_value = value;
    for (auto & kv : b->recs) {
        tsr_tensor_rec & r = kv.second;
        if (r.kind != tsr_kind::UNPLACED) {
            tsr_placed_fill(d, b, r.vaddr, value, r.nbytes);
        }
    }
}

static void tsr_buffer_reset(ggml_backend_buffer_t buffer) {
    tsr_buffer_ctx * b = tsr_buf(buffer);
    std::lock_guard<std::mutex> lock(b->dev->mtx);
    if (b->mode != tsr_bmode::PLACED) {
        b->recs.clear();
    }
    if (b->dev->msum.ready() && b->mode != tsr_bmode::KV) {
        b->dev->msum.forget(b->base, b->base + b->size);   // W3b kv: ggml-alloc lays the tensors out anew
    }
}

static const ggml_backend_buffer_i tsr_buffer_iface = {
    /* .free_buffer     = */ tsr_buffer_free,
    /* .get_base        = */ tsr_buffer_get_base,
    /* .init_tensor     = */ tsr_buffer_init_tensor,
    /* .memset_tensor   = */ tsr_buffer_memset_tensor,
    /* .set_tensor      = */ tsr_buffer_set_tensor,
    /* .get_tensor      = */ tsr_buffer_get_tensor,
    /* .set_tensor_2d   = */ nullptr,
    /* .get_tensor_2d   = */ nullptr,
    /* .cpy_tensor      = */ nullptr,
    /* .clear           = */ tsr_buffer_clear,
    /* .reset           = */ tsr_buffer_reset,
};

// ---------------------------------------------------------------------------
// ggml buffer type interface
// ---------------------------------------------------------------------------

static const char * tsr_buft_get_name(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return TSR_DEV_NAME;
}

static ggml_backend_buffer_t tsr_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    tsr_device * d = (tsr_device *) buft->context;
    tsr_buffer_ctx * b = new tsr_buffer_ctx();
    {
        std::lock_guard<std::mutex> lock(d->mtx);
        if (!tsr_dev_open_locked(d)) {
            delete b;
            return nullptr;
        }
        b->dev  = d;
        b->id   = d->next_buf_id++;
        b->size = size;
        b->base = d->next_vaddr;
        d->next_vaddr = tsr_pad(b->base + size + TSR_VADDR_GUARD, TSR_VADDR_GUARD);
    }
    TSR_DBG(2, "buffer #%" PRIu64 " allocated: %.2f MiB at virtual 0x%" PRIx64 "\n", b->id, tsr_mib(size), b->base);
    return ggml_backend_buffer_init(buft, tsr_buffer_iface, b, size);
}

static size_t tsr_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return TSR_TENSOR_ALIGN;
}

// One buffer type serves weights and activations. W3b builder (SW §8 R6, SW-26): the limit is
// 512 MiB, the size of a low window of ch1..ch3 (at least the IO window capacity), so that ggml-alloc
// cuts compute buffers into chunks that each get a window (IO, a low window or the compute heap) and
// the reserved logits of a large vocabulary stay on the device (Qwen2.5 at -ub 512: 297 MiB in one
// chunk; Gemma 4: 512 MiB). A single tensor larger than this still gets its own buffer (the compute
// heap holds any size the WEIGHT heap has room for); weight buffers above it become ggml
// multi-buffers (placed per tensor anyway).
static size_t tsr_buft_get_max_size(ggml_backend_buffer_type_t buft) {
    tsr_device * d = (tsr_device *) buft->context;
    std::lock_guard<std::mutex> lock(d->mtx);
    if (!tsr_dev_open_locked(d)) {
        return (size_t) TSR_REGION_IO_SIZE;
    }
    return (size_t) (tsr_io_windows_only() ? d->io.total() : std::max<uint64_t>(d->io.total(), TSR_REGION_WEIGHT_OFF));
}

// Pad to whole lines: a line-granular ST of the last row lands in this padding.
static size_t tsr_buft_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    GGML_UNUSED(buft);
    // W3b kv: a KQ mask (*kq_mask*, F16) also reserves its MSUM and one f32 copy (AT §7.3, tsr-msum.h)
    tsr_msum_layout L;
    if (tensor->view_src == nullptr &&
        tsr_msum_layout_of(tensor->name, tensor->type == GGML_TYPE_F16, tensor->ne, tensor->nb, ggml_nbytes(tensor), &L)) {
        return (size_t) L.total;
    }
    // W3b builder: ggml lets get_alloc_size exceed ggml_nbytes only for empty ops (leaves, views),
    // quantized types and the ops of ggml_op_alloc_size_may_expand (MUL_MAT, FLASH_ATTN_EXT, ...;
    // ggml-backend.cpp [TAG_ALLOC_SIZE_EXPAND] asserts it). Every other result keeps nbytes: ggml-alloc
    // pads each block to the 64-byte alignment anyway (ggml-alloc.c aligned_offset), which holds an M7
    // kind's whole-line store of a single row (tsr-build.cpp tsr_out_rows refuses it into a view).
    const size_t n = ggml_nbytes(tensor);
    if (ggml_op_is_empty(tensor->op) || ggml_is_quantized(tensor->type) || ggml_op_alloc_size_may_expand(tensor->op)) {
        return (size_t) tsr_pad(n, TSR_TENSOR_ALIGN);
    }
    return n;
}

static bool tsr_buft_is_host(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return false;
}

static const ggml_backend_buffer_type_i tsr_buft_iface = {
    /* .get_name         = */ tsr_buft_get_name,
    /* .alloc_buffer     = */ tsr_buft_alloc_buffer,
    /* .alloc_buffer_n   = */ nullptr,
    /* .get_alignment    = */ tsr_buft_get_alignment,
    /* .get_max_size     = */ tsr_buft_get_max_size,
    /* .get_alloc_size   = */ tsr_buft_get_alloc_size,
    /* .get_alloc_size_n = */ nullptr,
    /* .is_host          = */ tsr_buft_is_host,
};

// ---------------------------------------------------------------------------
// Lowering and execution (SW §9.3: M6 MUL_MAT records; M7 (W3b) every kind of the supports_op table,
// one op list per split)
// ---------------------------------------------------------------------------

struct tsr_gctx;

// A segment: the records of the pending op list (tsr_opl_builder: hazards over DRAM byte ranges,
// JOINs, the preload table), built in program order and run as one `graph` launch.
struct tsr_segment {
    explicit tsr_segment(uint32_t num_ch, tsr_gctx * g = nullptr) : ops(num_ch, tsr::opl::F_STOP_ON_ERROR), gctx(g) {}

    tsr_opl_builder               ops;        // records, hazards, JOINs, preload (tsr-opl.h)
    tsr_gctx *                    gctx;       // the graph_compute recording its lists (nullptr: none)
    std::vector<std::string>      names;      // node name per record tag
    std::set<tsr_buffer_ctx *>    written;    // buffers the pending list writes
    std::vector<uint64_t>         w_offs;     // W_OFF (bytes) of the pending MUL_MATs
    uint32_t                      n_fused = 0, n_gdn_cell = 0;
    uint32_t                      n_kind[tsr::opl::N_KINDS] = {};

    void clear() {
        ops.reset();
        names.clear();
        written.clear();
        w_offs.clear();
        n_fused = 0;
        n_gdn_cell = 0;
        std::fill(std::begin(n_kind), std::end(n_kind), 0u);
    }
};

// The graph_compute in progress (W3b): the builder's environment, the fusion pairs (pass 3), the
// FA_NEW sources (the SET_ROWS_KV of each K plane), and the plan being recorded for the cache (R7).
struct tsr_gctx {
    tsr_build_env env;
    tsr_fusion    fusion;
    tsr_fanew_map fanew;
    tsr_plan      plan;
};

// Every PERF_* counter: CYCLES, RETIRED, MVM, VU, (reserved), CH_RD_BEATS[c], CH_WAIT[c].
static std::vector<uint64_t> tsr_perf_all(tsr_device * d) {
    const uint32_t n = 5 + 2 * tsr_transport_config(d->tp)->num_ch;
    std::vector<uint64_t> v(n);
    for (uint32_t i = 0; i < n; i++) {
        const uint32_t lo = tsr_transport_csr_read(d->tp, TSR_CSR_PERF_BASE + 8 * i);   // LO snapshots
        const uint32_t hi = tsr_transport_csr_read(d->tp, TSR_CSR_PERF_BASE + 8 * i + 4);
        v[i] = ((uint64_t) hi << 32) | lo;
    }
    return v;
}

// TSR_DEBUG=2: the records of a launched list, with their nodes.
static void tsr_dump_opl(const tsr_plan_list & L) {
    tsr::opl::reader rd;
    if (rd.open(L.blob.data(), (uint32_t) L.blob.size(), false) != tsr::opl::RET_OK) {
        return;
    }
    for (uint32_t i = 0; i < rd.hdr().n_pre; ++i) {
        const tsr::opl::pre_entry e = rd.pre(i);
        fprintf(stderr, "[tesseract]   pre  %2u: ch%u+0x%09" PRIx64 " %5u B -> preload byte %u\n", i, tsr_handle_ch(e.src),
                tsr_handle_byte(e.src), e.bytes, e.pre_byte);
    }
    tsr::opl::record r;
    while (rd.next(r) == tsr::opl::RET_OK) {
        const int32_t tag = r.idx < L.tags.size() ? L.tags[r.idx] : -1;
        fprintf(stderr, "[tesseract]   %4u %-11s deps %5d %5d", r.idx, tsr_kind_name(r.kind),
                r.dep0 == tsr::opl::DEP_NONE ? -1 : (int) r.dep0, r.dep1 == tsr::opl::DEP_NONE ? -1 : (int) r.dep1);
        for (uint32_t i = 0; i < r.n_args; ++i) {
            fprintf(stderr, " %08x", r.arg(i));
        }
        fprintf(stderr, "%s%s\n", tag >= 0 ? "  " : "", tag >= 0 && (size_t) tag < L.names.size() ? L.names[tag].c_str() : "");
    }
}

// Run one list: OPL cache, `graph` launch, wait; abort on any error with the failing node (SW §9.3
// step 5: RET - 1 is the failing record, mapped to its tag and node name for every kind).
static void tsr_run_list(tsr_device * d, const tsr_plan_list & L) {
    const bool stats = tsr_stats_on();
    // SW §6.9: supports_op routed only the booted image's kinds, so every record is one of them.
    for (uint32_t k = 0; k < tsr::opl::N_KINDS; ++k) {
        if ((L.n_kind[k] != 0 || (k == tsr::opl::K_JOIN && L.n_joins != 0)) && !((d->kind_mask >> k) & 1u)) {
            GGML_ABORT("tesseract: internal error: record of kind %s, which the booted image does not provide", tsr_kind_name(k));
        }
    }
    if (stats && d->st_perf0.empty()) {
        d->st_perf0 = tsr_perf_all(d);
    }
    const tsr_opl_run_result r = d->odev.run(L.blob, d->launch_flags,
        [d](uint32_t ch, uint64_t off, const void * p, size_t n) { tsr_write(d, ch, off, p, n); });
    tsr_mon_note_mem(d);
    const uint32_t n_pre = L.blob.size() >= tsr::opl::HDR_BYTES ? tsr::opl::rd32(L.blob.data() + 4u * tsr::opl::hw::N_PRE) : 0u;
    TSR_DBG(1, "launch #%" PRIu64 ": %u records (%u MUL_MAT, %u fused, %u JOIN), %u preload entries, %zu B at OPL ch0+0x%07" PRIx64
               "%s, %.3f ms\n", d->odev.launches, L.n_ops, L.n_kind[tsr::opl::K_MUL_MAT], L.n_fused, L.n_joins, n_pre,
            L.blob.size(), r.opl_off, r.cache_hit ? " (cached)" : "", r.wall_ns * 1e-6);
    if (tsr_debug_level() >= 2) {
        tsr_dump_opl(L);
    }
    const bool ok = r.rc == 0 && r.res.type == TSR_LCQ_R0_TYPE_DONE && r.res.status == TSR_LCQ_R0_STATUS_OK && r.res.ret == 0;
    if (!ok) {
        const char * name = nullptr;
        const int32_t op = tsr_opl_failed_op(r, L.n_ops);
        if (op >= 0) {                        // SW §9.3 step 5: TILE_ERR -> the node
            const int32_t tag = (size_t) op < L.tags.size() ? L.tags[(size_t) op] : -1;
            name = tag >= 0 && (size_t) tag < L.names.size() ? L.names[(size_t) tag].c_str() : "(JOIN)";
        }
        GGML_ABORT("tesseract: device execution failed%s%s: %s (op list of %u records, %zu B at OPL ch0+0x%" PRIx64 ", graph_id 0x%08x)",
                   name ? " in node " : "", name ? name : "", tsr_opl_describe(r, name).c_str(), L.n_ops, L.blob.size(), r.opl_off,
                   tsr::opl::rd32(L.blob.data() + 4u * tsr::opl::hw::GRAPH_ID));
    }
    if (stats) {
        d->st_mm_records += L.n_kind[tsr::opl::K_MUL_MAT];
        d->st_joins += L.n_joins;
        d->st_fused += L.n_fused;
        d->st_gdn_cell += L.n_gdn_cell;
        for (uint32_t k = 0; k < tsr::opl::N_KINDS; ++k) {
            d->st_kind[k] += L.n_kind[k];
        }
    }
    for (tsr_buffer_ctx * b : L.written) {
        b->gen++;
    }
}

// Run the pending list (and record it in the graph_compute's plan).
static void tsr_flush(tsr_device * d, tsr_segment & seg) {
    if (seg.ops.empty()) {
        return;
    }
    tsr_plan_list L;
    L.blob = seg.ops.finish();
    if (L.blob.empty()) {
        GGML_ABORT("tesseract: op list build failed: %s", seg.ops.error());
    }
    L.names = seg.names;
    L.n_ops = seg.ops.n_ops();
    L.n_joins = seg.ops.n_joins();
    L.n_fused = seg.n_fused;
    L.n_gdn_cell = seg.n_gdn_cell;
    L.tags.resize(L.n_ops);
    for (uint32_t i = 0; i < L.n_ops; ++i) {
        L.tags[i] = seg.ops.tag_of(i);
    }
    std::copy(std::begin(seg.n_kind), std::end(seg.n_kind), std::begin(L.n_kind));
    L.written.assign(seg.written.begin(), seg.written.end());
    TSR_DBG(1, "op list: %u records, %u deps kept, %u implied, preload %u entries / %u lines\n", L.n_ops, seg.ops.n_deps(),
            seg.ops.n_dropped(), seg.ops.n_pre(), seg.ops.pre_lines());
    tsr_run_list(d, L);
    if (seg.gctx != nullptr) {
        seg.gctx->plan.lists.push_back(std::move(L));
    }
    seg.clear();
}

// TSR_OPL_SPLIT=<n> (test knob): cut every op list after n records, as a full list is cut (a split's
// results must not depend on where its lists are cut: FA_NEW, the preload entries per list).
static uint32_t tsr_opl_split() {
    static const uint32_t n = [] {
        const char * e = getenv("TSR_OPL_SPLIT");
        return e != nullptr ? (uint32_t) std::max(0, atoi(e)) : 0u;
    }();
    return n;
}

static void tsr_seg_maybe_cut(tsr_device * d, tsr_segment & seg);

// Append one MUL_MAT record (args: tsr::opl::mm words) for node `name`; a full list is run first.
static void tsr_seg_add_mm(tsr_device * d, tsr_segment & seg, const uint32_t * args, const char * name, tsr_buffer_ctx * dst_buf,
                           bool fused, uint32_t kind = tsr::opl::K_MUL_MAT) {
    tsr_seg_maybe_cut(d, seg);
    uint32_t idx = seg.ops.mul_mat(args, (int32_t) seg.names.size(), kind);
    if (idx == tsr_opl_builder::NO_ROOM) {
        tsr_flush(d, seg);                    // 4096 records per list (SW-29): run what is pending
        d->st_cut_lists++;
        idx = seg.ops.mul_mat(args, (int32_t) seg.names.size(), kind);
    }
    if (idx == tsr_opl_builder::NO_ROOM || idx == tsr_opl_builder::FAILED) {
        GGML_ABORT("tesseract: cannot add MUL_MAT '%s' to the op list: %s", name, seg.ops.error());
    }
    seg.names.push_back(name);
    seg.w_offs.push_back((uint64_t) args[tsr::opl::mm::W_OFF] * TSR_BEAT_BYTES);
    seg.n_kind[kind]++;
    seg.n_fused += kind == tsr::opl::K_MUL_MAT_SOFTCAP ? 3u : (fused ? 1u : 0u);
    if (dst_buf != nullptr) {
        seg.written.insert(dst_buf);
    }
}

static void tsr_seg_maybe_cut(tsr_device * d, tsr_segment & seg) {
    if (tsr_opl_split() != 0 && seg.ops.n_ops() >= tsr_opl_split()) {
        tsr_flush(d, seg);
        d->st_cut_lists++;
    }
}

// Append one M7 record (W3b, SW §9.3 passes 4-6: preload entries, ranges, dependencies, JOINs); a
// list without room for it (4096 records or 256 preload lines) is run first.
static void tsr_seg_add_rec(tsr_device * d, tsr_segment & seg, tsr_rec & r, const std::string & name, tsr_buffer_ctx * dst_buf,
                            bool fused) {
    static const tsr_fanew_map no_fanew;
    const tsr_fanew_map & fanew = seg.gctx != nullptr ? seg.gctx->fanew : no_fanew;
    tsr_seg_maybe_cut(d, seg);
    uint32_t idx = tsr_encode(seg.ops, r, (int32_t) seg.names.size(), fanew);
    if (idx == tsr_opl_builder::NO_ROOM) {
        tsr_flush(d, seg);
        d->st_cut_lists++;
        idx = tsr_encode(seg.ops, r, (int32_t) seg.names.size(), fanew);
    }
    if (idx == tsr_opl_builder::NO_ROOM || idx == tsr_opl_builder::FAILED) {
        GGML_ABORT("tesseract: cannot add %s '%s' to the op list: %s", tsr_kind_name(r.kind), name.c_str(),
                   idx == tsr_opl_builder::NO_ROOM ? "its preload entries alone exceed the 256-line preload area" : seg.ops.error());
    }
    seg.names.push_back(name);
    seg.n_kind[r.kind]++;
    seg.n_fused += fused ? 1u : 0u;
    if (dst_buf != nullptr) {
        seg.written.insert(dst_buf);
    }
}

struct tsr_wref {
    uint32_t fmt;
    uint64_t w_off;
    uint64_t exp_ch_bytes = 0;    // expert tensors: bytes of one expert per channel (EXP_STRIDE x 64); matrices: the whole
    uint32_t n_exp = 1;
};

// Convert a MUL_MAT src0 stored as raw ggml bytes into the WEIGHT heap (cached). The bytes are
// at byte offset `src_off` of channel `src_ch` (LINEAR window, RAW tensor of a weight buffer), or at
// `host` for a HOSTED buffer (src_off is then the offset inside the buffer and only keys the cache).
static tsr_wref tsr_lazy_weight(tsr_device * d, tsr_segment & seg, tsr_buffer_ctx * b, uint32_t src_ch, uint64_t src_off,
                                const uint8_t * host, enum ggml_type type, uint32_t N, uint32_t K, const char * name,
                                uint32_t n_exp = 1) {
    d->build_dynamic = true;    // W3b: a plan that converts weights is rebuilt every time (R7)
    if (seg.written.count(b)) {
        tsr_flush(d, seg);      // src0 may be produced by the pending list
    }
    const tsr_lazy_key key = { b->id, src_off, (int) type, N, K, n_exp };
    const uint32_t fmt = tsr_dev_fmt_for_type(type);
    auto it = d->lazy.find(key);
    if (it != d->lazy.end() && it->second.gen == b->gen) {
        return { it->second.fmt, it->second.w_off, it->second.exp_ch_bytes, n_exp };
    }
    if (it == d->lazy.end()) {
        tsr_lazy_entry e;
        e.fmt = fmt;
        e.exp_ch_bytes = tsr_tensor_ch_beats(fmt, N, K, d->cfg.num_ch) * TSR_BEAT_BYTES;
        e.ch_bytes = e.exp_ch_bytes * n_exp;
        if (!d->heap.alloc(e.ch_bytes, &e.w_off)) {
            // evicting cached conversions is only safe with nothing pending
            tsr_flush(d, seg);
            tsr_lazy_evict_all(d);
            if (!d->heap.alloc(e.ch_bytes, &e.w_off)) {
                GGML_ABORT("tesseract: WEIGHT heap full converting '%s' (%.1f MiB per channel needed)", name, tsr_mib(e.ch_bytes));
            }
        }
        it = d->lazy.emplace(key, e).first;
    } else if (!seg.ops.empty()) {
        // re-converting in place: the pending list may still read the old contents
        for (uint64_t w_off : seg.w_offs) {
            if (w_off == it->second.w_off) {
                tsr_flush(d, seg);
                break;
            }
        }
    }
    const size_t src_rb = ggml_row_size(type, K);
    const uint64_t exp_ch_bytes = it->second.exp_ch_bytes;
    if (host != nullptr) {
        tsr_store_expert_rows(d, type, fmt, N, K, exp_ch_bytes, it->second.w_off, host, 0, N * n_exp);
    } else {
        const uint32_t batch = (uint32_t) std::max<size_t>(1, TSR_XFER_CHUNK / src_rb);
        std::vector<uint8_t> rows;
        const uint64_t total = (uint64_t) N * n_exp;
        for (uint64_t r = 0; r < total;) {
            const uint32_t e = (uint32_t) (r / N), lo = (uint32_t) (r - (uint64_t) e * N);
            const uint32_t n = std::min<uint32_t>(batch, N - lo);       // a batch stays inside one expert
            rows.resize((size_t) n * src_rb);
            tsr_read(d, src_ch, src_off + r * src_rb, rows.data(), rows.size());
            tsr_store_weight_rows(d, type, fmt, N, K, it->second.w_off + (uint64_t) e * exp_ch_bytes, rows.data(), lo, lo + n);
            r += n;
        }
    }
    it->second.gen = b->gen;
    TSR_DBG(1, "lazy   %-28s %-6s [%6u x %6u%s] -> %s W_OFF=0x%09" PRIx64 " (from %s%u+0x%09" PRIx64 ", gen %" PRIu64 ")\n",
            name, ggml_type_name(type), N, K, n_exp > 1 ? (" x " + std::to_string(n_exp)).c_str() : "",
            fmt == TSR_FMT_Q8_0 ? "Q8_0" : "Q4_K", it->second.w_off,
            host != nullptr ? "host" : "ch", src_ch, src_off, b->gen);
    return { it->second.fmt, it->second.w_off, exp_ch_bytes, n_exp };
}

static tsr_buffer_ctx * tsr_tensor_buf(tsr_device * d, const ggml_tensor * t, const char * what) {
    ggml_backend_buffer_t gb = t->view_src ? t->view_src->buffer : t->buffer;
    if (gb == nullptr || gb->buft != &d->buft || gb->context == nullptr) {
        GGML_ABORT("tesseract: %s '%s' is not in a Tesseract buffer", what, t->name);
    }
    tsr_buffer_ctx * b = tsr_buf(gb);
    tsr_buf_decide(b, gb);
    return b;
}

// Where the bytes of a MUL_MAT activation (src1 or dst) live: in a LINEAR window (IO, a low window or
// the compute heap: the device reads / writes them in place) or in a HOSTED buffer (staged).
struct tsr_act {
    tsr_buffer_ctx * b;
    uint64_t rel;                 // byte offset inside the buffer
    bool     io;                  // LINEAR
    uint64_t  io_off() const { return b->io_off + rel; }                      // byte offset in channel b->ch
    uint32_t  handle(uint64_t extra = 0) const { return tsr_dram_handle(b->ch, b->io_off + rel + extra); }
    uint8_t * host()   const { return b->host + rel; }
};

static tsr_act tsr_act_loc(tsr_device * d, const ggml_tensor * t, const char * what) {
    tsr_buffer_ctx * b = tsr_tensor_buf(d, t, what);
    if (b->mode == tsr_bmode::PLACED || b->mode == tsr_bmode::KV) {
        GGML_ABORT("tesseract: %s '%s' is in a weight / KV buffer; MUL_MAT activations must be in compute buffers", what, t->name);
    }
    const uint64_t a = tsr_vaddr(t->data);
    tsr_check_range(b, a, ggml_nbytes(t));
    return { b, a - b->base, b->mode == tsr_bmode::LINEAR };
}

static tsr_wref tsr_resolve_weight(tsr_device * d, tsr_segment & seg, const ggml_tensor * src0) {
    tsr_buffer_ctx * b = tsr_tensor_buf(d, src0, "src0");
    const uint32_t K = (uint32_t) src0->ne[0];
    const uint32_t N = (uint32_t) src0->ne[1];
    const uint32_t NE = (uint32_t) src0->ne[2];          // experts (MUL_MAT_ID); 1 for a matrix
    const uint64_t a = tsr_vaddr(src0->data);
    if (b->mode == tsr_bmode::LINEAR) {
        tsr_check_range(b, a, ggml_nbytes(src0));
        return tsr_lazy_weight(d, seg, b, b->ch, b->io_off + (a - b->base), nullptr, src0->type, N, K, src0->name, NE);
    }
    if (b->mode == tsr_bmode::HOSTED) {
        tsr_check_range(b, a, ggml_nbytes(src0));
        return tsr_lazy_weight(d, seg, b, 0, a - b->base, b->host + (a - b->base), src0->type, N, K, src0->name, NE);
    }
    if (b->mode == tsr_bmode::KV) {
        GGML_ABORT("tesseract: MUL_MAT src0 '%s' is a KV cache; attention runs as FLASH_ATTN only (AT §2.6)", src0->name);
    }
    tsr_tensor_rec * r = tsr_find_rec(b, a);
    if (r == nullptr) {
        GGML_ABORT("tesseract: weight '%s' was never initialized in its buffer", src0->name);
    }
    if (r->kind == tsr_kind::UNPLACED) {
        GGML_ABORT("tesseract: weight '%s' was never uploaded", src0->name);
    }
    if (r->kind == tsr_kind::DEVWEIGHT) {
        if (r->vaddr != a || r->type != src0->type || r->K != K || r->N != N || r->n_exp != NE) {
            GGML_ABORT("tesseract: MUL_MAT uses a view of device weight '%s'; views of placed weights are not supported", r->name.c_str());
        }
        if (!r->complete) {
            GGML_ABORT("tesseract: weight '%s' is only partially uploaded", r->name.c_str());
        }
        return { r->fmt, r->w_off, r->exp_ch_bytes, r->n_exp };
    }
    // RAW in a weight buffer (e.g. a 1D f32 tensor used as a 1 x K matrix): ch0 (AUX or the heap)
    return tsr_lazy_weight(d, seg, b, 0, r->raw_off + (a - r->vaddr), nullptr, src0->type, N, K, src0->name, NE);
}

// The MUL_MAT record (SW §7.2 kind 0x10) of T tokens: x rows x_lines apart at handle x, y rows
// y_lines apart at handle y, no addend, no ACC_DST.
static void tsr_mm_args(uint32_t * a, uint32_t x, uint32_t y, const tsr_wref & w, uint32_t N, uint32_t K, uint32_t T,
                        const tsr_mm_tiling & tl) {
    namespace mm = tsr::opl::mm;
    a[mm::X]          = x;
    a[mm::X_STRIDE]   = tl.x_lines;
    a[mm::Y]          = y;
    a[mm::Y_STRIDE]   = tl.y_lines;
    a[mm::W_OFF]      = (uint32_t) (w.w_off / TSR_BEAT_BYTES);
    a[mm::FMT]        = w.fmt;
    a[mm::N]          = N;
    a[mm::K]          = K;
    a[mm::T]          = T;
    a[mm::ADD]        = tsr::opl::HANDLE_NONE;
    a[mm::ADD_STRIDE] = 0;
    a[mm::FLAGS]      = 0;
}

// A MUL_MAT whose operands the device cannot address in place: src1 or dst in host memory
// (HOSTED buffer), or dst rows that do not start on a line (N*4 % 64 != 0 with T > 1). It runs
// on its own (after everything pending), one single-record launch per T tile that fits the IO
// stage (SW §9.3 step 2):
//   src1 rows host -> stage (unless src1 is in a LINEAR window: LD reads it in place),
//   LD / QACT / MVM / ST (dst rows to the stage, line-padded, or straight into a LINEAR dst),
//   stage -> dst rows (compacted to N*4 bytes each; host memory or the LINEAR window).
// So only one tile of the activations has to be resident in IO at a time, whatever T is.
static void tsr_lower_staged(tsr_device * d, tsr_segment & seg, const ggml_tensor * node, const tsr_act & ax,
                             const tsr_act & ay, const tsr_stage_plan & sp, uint32_t fmt, uint32_t N, uint32_t K,
                             uint32_t T, const tsr_mm_tiling & tl) {
    d->build_dynamic = true;    // W3b: host transfers between launches: not a cached plan (R7)
    tsr_flush(d, seg);          // program order: what is pending may produce src1 / src0
    const ggml_tensor * src0 = node->src[0];
    const tsr_wref w = tsr_resolve_weight(d, seg, src0);
    GGML_ASSERT(w.fmt == fmt && w.w_off % TSR_BEAT_BYTES == 0 && seg.ops.empty());

    const bool     stage_x   = sp.stage_x, stage_y = sp.stage_y;
    const uint64_t x_row     = (uint64_t) tl.x_lines * TSR_LINE_BYTES;   // K*4 (K % 32 == 0)
    const uint64_t row_bytes = (uint64_t) N * 4;
    const uint64_t y_stride  = (uint64_t) tl.y_lines * TSR_LINE_BYTES;   // line-padded row
    const uint64_t tc_max    = sp.tc_max;
    if (tc_max == 0) {
        GGML_ABORT("tesseract: MUL_MAT '%s' (N=%u K=%u): one token needs %.2f MiB of IO stage, the stage has %.2f MiB",
                   node->name, N, K, tsr_mib(sp.per_token), tsr_mib(d->stage_size));
    }
    const uint64_t xs = sp.x_off;
    const uint64_t ys = sp.y_off;
    std::vector<uint8_t> yb;
    std::vector<uint8_t> packed;
    uint32_t n_tiles = 0;
    for (uint64_t t0 = 0; t0 < T; t0 += tc_max, ++n_tiles) {
        const uint32_t tc = (uint32_t) std::min<uint64_t>(tc_max, T - t0);
        if (stage_x) {
            tsr_write(d, 0, xs, ax.host() + t0 * x_row, (size_t) (tc * x_row));
        }
        uint32_t args[tsr::opl::mm::NARGS];
        tsr_mm_args(args, stage_x ? tsr_dram_handle(0, xs) : ax.handle(t0 * x_row),
                    stage_y ? tsr_dram_handle(0, ys) : ay.handle(t0 * y_stride), w, N, K, tc, tl);
        tsr_segment one(d->cfg.num_ch);
        tsr_seg_add_mm(d, one, args, node->name, stage_y ? nullptr : ay.b, false);
        tsr_flush(d, one);
        if (stage_y) {
            yb.resize((size_t) (tc * y_stride));
            tsr_read(d, 0, ys, yb.data(), yb.size());
            if (ay.io) {
                packed.resize((size_t) (tc * row_bytes));
                for (uint32_t t = 0; t < tc; ++t) {
                    memcpy(packed.data() + t * row_bytes, yb.data() + t * y_stride, (size_t) row_bytes);
                }
                tsr_write(d, ay.b->ch, ay.io_off() + t0 * row_bytes, packed.data(), packed.size());
            } else {
                for (uint32_t t = 0; t < tc; ++t) {
                    memcpy(ay.host() + (t0 + t) * row_bytes, yb.data() + t * y_stride, (size_t) row_bytes);
                }
            }
        }
    }
    ay.b->gen++;
    if (tsr_stats_on()) {
        d->st_staged_mm++;
        d->st_staged_launches += n_tiles;
    }

    TSR_DBG(1, "MUL_MAT %-24s %-6s N=%u K=%u T=%u W_OFF=0x%09" PRIx64 " staged: src1 %s, dst %s, "
               "%u stage tile(s) of <= %" PRIu64 " tokens, one launch each\n",
            node->name, ggml_type_name(src0->type), N, K, T, w.w_off,
            stage_x ? "host->stage" : "in place", stage_y ? (ay.io ? "stage->window (compacted)" : "stage->host") : "in place",
            n_tiles, tc_max);
}

// One MUL_MAT and its candidate fusion. Bias and residual ADD use the existing addend.
// A pointwise epilogue exports only the final result. A staged MUL_MAT or an
// ineligible epilogue leaves the graph's original consumers to be lowered separately.
static bool tsr_lower_mul_mat(tsr_device * d, tsr_segment & seg, const ggml_tensor * node, const ggml_tensor * after,
                              const ggml_tensor * softcap_tail) {
    const ggml_tensor * src0 = node->src[0];
    const ggml_tensor * src1 = node->src[1];
    if (src1->type != GGML_TYPE_F32 || node->type != GGML_TYPE_F32 || src0->ne[2] != 1 || src0->ne[3] != 1 ||
        !ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(node)) {
        GGML_ABORT("tesseract: MUL_MAT '%s' reached graph_compute in an unsupported form", node->name);
    }
    const uint32_t K = (uint32_t) src0->ne[0];
    const uint32_t N = (uint32_t) src0->ne[1];
    const int64_t  T64 = ggml_nrows(src1);
    if (N == 0 || T64 == 0) {
        return false;
    }
    GGML_ASSERT(node->ne[0] == (int64_t) N && ggml_nrows(node) == T64);
    if (T64 > (int64_t) UINT32_MAX) {
        GGML_ABORT("tesseract: MUL_MAT '%s' has too many tokens", node->name);
    }
    const uint32_t T = (uint32_t) T64;
    const uint32_t fmt = tsr_dev_fmt_for_type(src0->type);
    tsr_mm_tiling tl;
    if (!tsr_mm_plan_tiling(d->cfg, fmt, N, K, T, &tl) ||
        !tsr::plan::mm_supported(tsr_plan_cfg(d->cfg), fmt, N, K)) {
        GGML_ABORT("tesseract: MUL_MAT '%s' (N=%u K=%u) cannot be lowered", node->name, N, K);
    }

    const tsr_act ax = tsr_act_loc(d, src1, "src1");
    const tsr_act ay = tsr_act_loc(d, node, "dst");
    const uint64_t y_stride  = (uint64_t) tl.y_lines * TSR_LINE_BYTES;
    tsr_stage_plan sp;
    tsr_plan_stage(d->stage_off, d->stage_size, N, T, (uint64_t) tl.x_lines * TSR_LINE_BYTES, y_stride, ax.io, ay.io, &sp);
    const bool stage_x = sp.stage_x;
    const bool stage_y = sp.stage_y;
    if ((ax.io && ax.io_off() % TSR_BEAT_BYTES != 0) || (ay.io && ay.io_off() % TSR_BEAT_BYTES != 0)) {
        GGML_ABORT("tesseract: MUL_MAT '%s': src1/dst not 64-byte aligned in device memory", node->name);
    }
    if (!stage_y) {
        // the ST writes whole lines: T*N*4 bytes exactly (N % 16 == 0), or the single row plus
        // its tail, which lands in the 64-byte padding of get_alloc_size
        const uint64_t st_end = ay.io_off() + (uint64_t) T * y_stride;
        if (st_end > ay.b->io_off + tsr_pad(ay.b->size, TSR_TENSOR_ALIGN) ||
            st_end > ay.io_off() + tsr_pad(ggml_nbytes(node), TSR_TENSOR_ALIGN)) {
            GGML_ABORT("tesseract: MUL_MAT '%s': output store would exceed the tensor allocation", node->name);
        }
    }
    if (stage_x || stage_y) {
        tsr_lower_staged(d, seg, node, ax, ay, sp, fmt, N, K, T, tl);
        return false;
    }

    const tsr_wref w = tsr_resolve_weight(d, seg, src0);
    GGML_ASSERT(w.fmt == fmt && w.w_off % TSR_BEAT_BYTES == 0);

    uint32_t args[tsr::opl::mm_scale::SOFTCAP_NARGS];
    tsr_mm_args(args, ax.handle(), ay.handle(), w, N, K, T, tl);
    bool fused = false;
    uint32_t kind = tsr::opl::K_MUL_MAT;
    std::string name = node->name;
    tsr_buffer_ctx * dst_buf = ay.b;
    const ggml_tensor * fused_out = softcap_tail ? softcap_tail : after;
    if (after && seg.gctx) {
        std::string why;
        if (softcap_tail) {
            fused = tsr_fuse_mm_softcap(seg.gctx->env, node, after, softcap_tail, args);
            if (fused) kind = tsr::opl::K_MUL_MAT_SOFTCAP;
        } else if (after->op == GGML_OP_SCALE) {
            fused = tsr_fuse_mm_scale(seg.gctx->env, node, after, args);
            if (fused) kind = tsr::opl::K_MUL_MAT_SCALE;
        } else {
            fused = tsr_fuse_mm_add(seg.gctx->env, node, after, args, &why);
        }
        if (fused) {
            name = std::string(node->name) + " + " + fused_out->name;
            dst_buf = tsr_tensor_buf(d, fused_out, "fused result");
        } else {
            TSR_DBG(1, "MUL_MAT %s + %s not fused: %s\n", node->name, fused_out->name, why.c_str());
        }
    }
    tsr_seg_add_mm(d, seg, args, name.c_str(), dst_buf, fused, kind);

    TSR_DBG(1, "%s %-24s %-6s N=%u K=%u T=%u W_OFF=0x%09" PRIx64 " x=ch%u+0x%09" PRIx64 " y=ch%u+0x%09" PRIx64 "%s%s\n",
            tsr_kind_name(kind), node->name, ggml_type_name(src0->type), N, K, T, w.w_off, ax.b->ch, ax.io_off(), tsr_handle_ch(args[tsr::opl::mm::Y]),
            tsr_handle_byte(args[tsr::opl::mm::Y]), fused ? " + " : "", fused ? fused_out->name : "");
    return fused;
}

// ---------------------------------------------------------------------------
// W3b builder: the M7 environment (tsr-build.h) on this device
// ---------------------------------------------------------------------------

// Device location of a tensor's first byte that the M7 kinds address in place: a LINEAR window, or
// a RAW tensor of a weight buffer (AUX or the heap, ch0). Not: host memory, device-format weights,
// KV planes (tsr_kv_view_of), never-written weights.
static bool tsr_tensor_loc(tsr_device * d, const ggml_tensor * t, tsr_tloc * out) {
    const ggml_tensor * base = t->view_src != nullptr ? t->view_src : t;
    ggml_backend_buffer_t gb = base->buffer;
    if (gb == nullptr || gb->buft != &d->buft || gb->context == nullptr || t->data == nullptr) {
        return false;
    }
    tsr_buffer_ctx * b = tsr_buf(gb);
    tsr_buf_decide(b, gb);
    const uint64_t a = tsr_vaddr(t->data);
    if (a < b->base || a + ggml_nbytes(t) > b->base + b->size) {
        return false;
    }
    if (b->mode == tsr_bmode::LINEAR) {
        out->ch = b->ch;
        out->off = b->io_off + (a - b->base);
        return true;
    }
    if (b->mode == tsr_bmode::PLACED) {
        const tsr_tensor_rec * r = tsr_find_rec(b, a);
        if (r == nullptr || r->kind != tsr_kind::RAW || a + ggml_nbytes(t) > r->vaddr + r->nbytes) {
            return false;
        }
        out->ch = 0;
        out->off = r->raw_off + (a - r->vaddr);
        return true;
    }
    return false;
}

// supports_op's probe of an operand (tsr-build.h raw_ok): its bytes will be raw rows in device memory
// at a 64-byte-aligned address: not a device-format weight, not a KV cache, not host memory; a view
// 64-byte aligned within what it views; a tensor of another backend's buffer is copied in by the
// scheduler (aligned) into a compute buffer.
//
// Host memory (W3b fix; SW §8 R6 as built): the M7 kinds address their operands in place and have no
// staging (only MUL_MAT does, tsr_lower_staged), so an operand in a HOSTED buffer, or in one that will
// become HOSTED (TSR_FORCE_HOST, or an ANY buffer holding only leaf data: llama.cpp's control vectors
// and LoRA adapters), keeps its op on the CPU. A compute buffer that gets no window is decided only at
// its first data access, after the scheduler has split the graph: graph_compute then aborts naming it.
static bool tsr_probe_raw_ok(tsr_device * d, const ggml_tensor * t) {
    const ggml_tensor * base = t->view_src != nullptr ? t->view_src : t;
    const ggml_backend_buffer * gb = base->buffer;
    if (t->view_src != nullptr && t->view_offs % TSR_TENSOR_ALIGN != 0) {
        return false;
    }
    if (d->force_host) {
        return false;     // every compute buffer, hence every result and copied-in operand, is HOSTED
    }
    if (gb == nullptr || gb->buft != &d->buft || gb->context == nullptr) {
        return true;
    }
    tsr_buffer_ctx * b = (tsr_buffer_ctx *) gb->context;
    if (t->data != nullptr && tsr_vaddr(t->data) % TSR_TENSOR_ALIGN != 0) {
        return false;
    }
    if (b->mode == tsr_bmode::KV || (b->mode == tsr_bmode::UNDECIDED && gb->usage == GGML_BACKEND_BUFFER_USAGE_ANY &&
                                     tsr_kv_candidate(d, b))) {
        return false;
    }
    if (b->mode == tsr_bmode::HOSTED || (b->mode == tsr_bmode::UNDECIDED && tsr_buf_leaves_only(b, gb))) {
        return false;
    }
    if (b->mode == tsr_bmode::PLACED || gb->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        const tsr_tensor_rec * r = base->data != nullptr ? tsr_find_rec(b, tsr_vaddr(base->data)) : nullptr;
        if (r != nullptr) {
            return r->kind == tsr_kind::RAW || (r->kind == tsr_kind::UNPLACED && !tsr_rec_is_devweight_shape(d, *r));
        }
    }
    return true;
}

// supports_op: base is a KV-cache tensor (tsr_kv_identify; its shape in *s) in a Tesseract buffer that is
// or becomes a KV buffer. llama.cpp's no_alloc passes (memory fitting, src/llama-kv-cache.cpp:280) put
// the cache in a dummy size-0 buffer of this type, which has no context: it never holds data, so the
// graph they reserve is planned as with a KV buffer.
static bool tsr_kv_probe_tensor(tsr_device * d, const ggml_tensor * base, tsr_kv_shape * s, bool dbg) {
    std::string why;
    if (!tsr_kv_identify(base->name, base->type == GGML_TYPE_Q8_0, base->ne, base->nb, d->cfg.num_ch, d->cfg.k_max, s, &why)) {
        if (dbg) {
            TSR_DBG(2, "supports_op: '%s' is not a KV-cache tensor the device holds: %s\n", base->name, why.c_str());
        }
        return false;
    }
    const ggml_backend_buffer * gb = base->buffer;
    const bool dummy = gb != nullptr && gb->buft == &d->buft && gb->context == nullptr && gb->size == 0;
    if (!dummy && !tsr_kv_tensor_supported(d, base)) {
        if (dbg) {
            TSR_DBG(2, "supports_op: '%s' is not in a Tesseract buffer that is (or becomes) a KV buffer\n", base->name);
        }
        return false;
    }
    return true;
}

// The planner's view of the device with the M7 fields (SW §7.2: C, PCAP, VL, CAPS; 0 = unknown).
static tsr::plan::dev_cfg tsr_plan_cfg_m7(tsr_device * d) {
    tsr::plan::dev_cfg c = tsr_plan_cfg(d->cfg);
    const uint32_t cfg3 = tsr_transport_csr_read(d->tp, TSR_CSR_CFG3);
    c.num_ch = d->cfg.num_ch;
    c.pcap = ((cfg3 & TSR_CFG3_PCAP16) >> TSR_CFG3_PCAP16_LSB) * 16u;
    c.vl = d->cfg.vl;
    c.caps = d->cfg.caps;
    return c;
}

// MUL_MAT_ID's src0 (supports_op, SW-60): the rules of MUL_MAT's weights for a 3D expert tensor. A device-format weight
// (Q8_0 / Q4_K, or anything the device converts at load) in a weights buffer, raw bytes in any other Tesseract buffer
// that the backend converts at graph_compute, or an unallocated leaf (llama.cpp's weight-placement probe).
static bool tsr_is_our_buffer(const tsr_device * d, const ggml_backend_buffer * buf);
static bool tsr_moe_weight_ok(tsr_device * d, const ggml_tensor * w) {
    if (!tsr_type_loadable(w->type) || w->ne[3] != 1 || w->ne[2] < 1 || !ggml_is_contiguous(w)) {
        return false;
    }
    const uint32_t fmt = tsr_dev_fmt_for_type(w->type);
    if (w->ne[1] > (int64_t) UINT32_MAX || !tsr_mm_shape_ok(d->cfg, fmt, (uint64_t) w->ne[1], (uint64_t) w->ne[0]) ||
        !tsr::plan::mm_supported(tsr_plan_cfg(d->cfg), fmt, (uint32_t) w->ne[1], (uint32_t) w->ne[0])) {
        return false;
    }
    const ggml_tensor * base0 = w->view_src ? w->view_src : w;
    const ggml_backend_buffer * buf0 = base0->buffer;
    if (buf0 == nullptr) {
        return base0->op == GGML_OP_NONE && (base0->flags & GGML_TENSOR_FLAG_INPUT) == 0;
    }
    if (!tsr_is_our_buffer(d, buf0) || buf0->usage == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        return false;
    }
    return !(buf0->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS && w->view_src != nullptr);
}

// SW-76: the scratch window of at least `bytes` bytes (env.scratch). Callers hold dev->mtx.
static bool tsr_scratch_window(tsr_device * d, uint64_t bytes, tsr_tloc * out) {
    if (bytes > d->scr_size) {
        const uint64_t want = tsr_pad(std::max<uint64_t>({ bytes, 2 * d->scr_size, 1ull << 20 }), 1ull << 16);
        uint64_t off = 0;
        uint32_t ch = 0;
        bool ok = d->io.alloc_top(want, &off);
        if (!ok && !tsr_io_windows_only()) {
            for (uint32_t c = 1; c < (uint32_t) d->low.size() && !ok; ++c) {
                if (d->low[c].alloc_top(want, &off)) {
                    ok = true;
                    ch = c;
                }
            }
        }
        if (!ok) {
            return false;
        }
        d->scr_ch = ch;
        d->scr_off = off;
        d->scr_size = want;
        TSR_DBG(1, "MOE_ROUTE scratch: %.2f MiB at ch%u+0x%09" PRIx64 "\n", tsr_mib(want), ch, off);
    }
    out->ch = d->scr_ch;
    out->off = d->scr_off;
    return true;
}

// The builder's environment (tsr-build.h) for this device. Callers hold dev->mtx (device open).
static bool tsr_is_our_buffer(const tsr_device * d, const ggml_backend_buffer * buf);

// The values of an I32 input as the next launch sees them: in a device window (read through the transport), in a hosted
// buffer, or in another backend's host buffer. False when t is not I32, has no data or lives somewhere the backend cannot read
// (graph_compute then makes no plan that depends on the values). Callers hold dev->mtx.
static bool tsr_is_our_buffer(const tsr_device * d, const ggml_backend_buffer * buf);
static bool tsr_read_i32(tsr_device * d, const ggml_tensor * t, std::vector<int32_t> * out) {
    ggml_backend_buffer_t gb = t->buffer;
    if (t->type != GGML_TYPE_I32 || gb == nullptr || t->data == nullptr || t->view_src != nullptr) {
        return false;
    }
    const size_t nb = ggml_nbytes(t);
    out->assign(nb / 4, 0);
    if (!tsr_is_our_buffer(d, gb)) {
        if (!ggml_backend_buffer_is_host(gb)) {
            return false;
        }
        memcpy(out->data(), t->data, nb);
        return true;
    }
    tsr_buffer_ctx * b = tsr_buf(gb);
    const uint64_t addr = tsr_vaddr(t->data);
    if (b->mode == tsr_bmode::LINEAR) {
        tsr_read(d, b->ch, b->io_off + (addr - b->base), out->data(), nb);
    } else if (b->mode == tsr_bmode::HOSTED) {
        memcpy(out->data(), b->host + (addr - b->base), nb);
    } else {
        return false;
    }
    return true;
}

static tsr_build_env tsr_env_of(tsr_device * d) {
    if (!d->m7cfg_ok) {
        d->m7cfg = tsr_plan_cfg_m7(d);   // one CFG3 read per device
        d->m7cfg_ok = true;
    }
    tsr_build_env e;
    e.cfg = d->m7cfg;
    e.pl = tsr::plan::plan_limits_of(e.cfg);
    e.num_ch = d->cfg.num_ch;
    e.kinds = d->kind_mask;              // SW §6.9: the booted image's kinds, narrowed by TSR_KINDS
    e.image = d->image_desc;
    e.loc = [d](const ggml_tensor * t, tsr_tloc * l) { return tsr_tensor_loc(d, t, l); };
    e.kv_view = [d](const ggml_tensor * t, tsr_kv_view * v) { return tsr_kv_view_of(d, t, v); };
    e.fa_mask = [d](const ggml_tensor * fa) { return tsr_fa_mask_ref(d, fa); };
    e.raw_ok = [d](const ggml_tensor * t) { return tsr_probe_raw_ok(d, t); };
    e.on_device = [d](const ggml_tensor * t) {
        const ggml_backend_buffer * gb = t->view_src != nullptr ? t->view_src->buffer : t->buffer;
        return tsr_is_our_buffer(d, gb);
    };
    e.kv_probe = [d](const ggml_tensor * t, tsr_kv_shape * s) {
        return tsr_kv_probe_tensor(d, t->view_src != nullptr ? t->view_src : t, s, true);
    };
    e.fa_mask_probe = [](const ggml_tensor * fa) { return tsr_fa_mask_supported(fa); };
    e.moe_w_probe = [d](const ggml_tensor * w) { return tsr_moe_weight_ok(d, w); };
    e.scratch = [d](uint64_t bytes, tsr_tloc * l) { return tsr_scratch_window(d, bytes, l); };
    e.read_i32 = [d](const ggml_tensor * t, std::vector<int32_t> * v) { return tsr_read_i32(d, t, v); };
    static const char * cell_env = getenv("TSR_GDN_CELL");
    e.gdn_cell = cell_env == nullptr || strcmp(cell_env, "0") != 0;
    return e;
}

// The buffer a record writes (its generation is bumped after the launch; nullptr for KV planes).
static tsr_buffer_ctx * tsr_written_buf(tsr_device * d, const ggml_tensor * t) {
    const ggml_tensor * base = t->view_src != nullptr ? t->view_src : t;
    ggml_backend_buffer_t gb = base->buffer;
    if (gb == nullptr || gb->buft != &d->buft || gb->context == nullptr) {
        return nullptr;
    }
    tsr_buffer_ctx * b = tsr_buf(gb);
    return b->mode == tsr_bmode::KV ? nullptr : b;
}

// A cached plan is reused only while its FLASH_ATTN records' mask words are what the producer hands
// out now (re-registering every reader, tsr-msum.h: the producer then keeps their copies current).
static bool tsr_plan_masks_hold(tsr_device * d, const tsr_plan & p) {
    for (size_t i = 0; i < p.fa_nodes.size(); ++i) {
        const tsr_msum_ref r = tsr_fa_mask_ref(d, p.fa_nodes[i]);
        if (!r.valid || r.msum_handle != p.fa_refs[i].msum_handle || r.mask_handle != p.fa_refs[i].mask_handle) {
            return false;
        }
    }
    return true;
}

// The plan of a graph with a GATED_DELTA_NET state gather (SW-82) holds while the ids of the gather say what the build
// decided: the identity (every sequence keeps its cache cell, the record works on the cell in place) or not (the gather and
// the copy run). The ids change with the sequences, not with the shape of the graph, so the graph key does not see them.
static bool tsr_plan_state_holds(tsr_device * d, const tsr_plan & p) {
    std::vector<int32_t> ids;
    for (const tsr_gdn_cell & c : p.gdn_cells) {
        if (!tsr_read_i32(d, c.ids, &ids) || ids.size() != (size_t) c.n) {
            return false;
        }
        bool identity = true;
        for (uint32_t j = 0; j < c.n; ++j) {
            identity = identity && ids[j] == c.cell0 + (int32_t) j;
        }
        if (identity != c.in_place) {
            return false;
        }
    }
    return true;
}

// An operand of an M7 node (or of the MUL folded into it) that lives in host memory (a HOSTED buffer:
// a compute buffer the device had no window for); nullptr if none.
static const ggml_tensor * tsr_hosted_operand(tsr_device * d, const ggml_tensor * node, const ggml_tensor * mul) {
    std::vector<const ggml_tensor *> ts = { node, mul };
    for (const ggml_tensor * n : { node, mul }) {
        for (int s = 0; n != nullptr && s < GGML_MAX_SRC; ++s) {
            ts.push_back(n->src[s]);
        }
    }
    for (const ggml_tensor * t : ts) {
        const ggml_tensor * base = t != nullptr && t->view_src != nullptr ? t->view_src : t;
        const ggml_backend_buffer * gb = base != nullptr ? base->buffer : nullptr;
        if (gb != nullptr && gb->buft == &d->buft && gb->context != nullptr &&
            ((const tsr_buffer_ctx *) gb->context)->mode == tsr_bmode::HOSTED) {
            return t;
        }
    }
    return nullptr;
}

// The SET_ROWS_KV of a K plane in this graph_compute (FA_NEW's source): its index input, its node and
// the list it went into (tsr_gctx plan.lists index).
struct tsr_kv_write {
    const ggml_tensor * idx;
    int                 node;
    size_t              list;
};

// FA_NEW across a cut (tsr-build.h, the assumption): the FLASH_ATTN's list preloads the K indices from
// DRAM again, so no node between the SET_ROWS_KV and the FLASH_ATTN may have written their bytes (ggml-
// alloc may reuse them: the FLASH_ATTN is not their ggml consumer). Never the case in llama.cpp's graphs
// (only the V SET_ROWS_KV lies between); checked whenever a cut separates the two.
static void tsr_check_fanew_source(const ggml_cgraph * cg, const tsr_kv_write & w, int fa_node) {
    const uint64_t lo = (uint64_t) (uintptr_t) w.idx->data, hi = lo + ggml_nbytes(w.idx);
    for (int j = w.node + 1; j < fa_node; ++j) {
        const ggml_tensor * n = cg->nodes[j];
        if ((n->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || tsr_node_is_noop(n) || n->data == nullptr) {
            continue;
        }
        const uint64_t a = (uint64_t) (uintptr_t) n->data, b = a + ggml_nbytes(n);
        if (a < hi && lo < b) {
            GGML_ABORT("tesseract: '%s' overwrites the K indices '%s' between their SET_ROWS_KV and the FLASH_ATTN '%s', "
                       "and an op-list cut separates the two, so the FLASH_ATTN's FA_NEW entry would read the "
                       "overwritten bytes (tsr-build.h FA_NEW assumption; SW §9.3)", n->name, w.idx->name,
                       cg->nodes[fa_node]->name);
        }
    }
}

// MUL_MAT_ID (SW-62): the weights resolved to W_OFF and EXP_STRIDE (placed, or converted at graph_compute like MUL_MAT's), then
// the record from tsr-build; with a scale group (SW-64) the record carries the table and a CPY moves its result into the MUL's
// tensor (the allocator may have put that tensor over the MUL_MAT_ID's inputs), unless both tensors are the same bytes.
static void tsr_lower_mul_mat_id_node(tsr_device * d, tsr_segment & seg, tsr_gctx & g, ggml_cgraph * cgraph, int i,
                                      std::vector<char> & absorbed) {
    ggml_tensor * node = cgraph->nodes[i];
    const tsr_wref w = tsr_resolve_weight(d, seg, node->src[0]);
    const tsr_moe_w mw = { w.fmt, (uint32_t) (w.w_off / TSR_BEAT_BYTES), (uint32_t) (w.exp_ch_bytes / TSR_BEAT_BYTES), w.n_exp };
    const int si = g.fusion.scale_of[i];
    const tsr_moe_scale * sc = si >= 0 ? &g.fusion.scales[(size_t) si] : nullptr;
    tsr_rec r;
    std::string why;
    if (!tsr_lower_mul_mat_id(g.env, false, node, sc, &mw, &r, &why)) {
        const ggml_tensor * h = tsr_hosted_operand(d, node, nullptr);
        GGML_ABORT("tesseract: MUL_MAT_ID '%s' reached graph_compute but cannot run on the device: %s%s%s", node->name, why.c_str(),
                   h != nullptr ? "; in host memory: " : "", h != nullptr ? h->name : "");
    }
    seg.w_offs.push_back(w.w_off);
    tsr_seg_add_rec(d, seg, r, node->name, tsr_written_buf(d, node), false);
    TSR_DBG(1, "MUL_MAT_ID  %-24s %-6s N=%lld K=%lld experts=%u used=%lld T=%lld W_OFF=0x%09" PRIx64 " stride=%u beats%s\n", node->name,
            ggml_type_name(node->src[0]->type), (long long) node->src[0]->ne[1], (long long) node->src[0]->ne[0], w.n_exp,
            (long long) node->src[2]->ne[0], (long long) node->src[2]->ne[1], w.w_off, mw.exp_stride, sc != nullptr ? " + scale" : "");
    if (sc == nullptr) {
        return;
    }
    ggml_tensor * mul = cgraph->nodes[sc->mul];
    tsr_rec c;
    if (!tsr_lower_moe_scale_copy(g.env, node, mul, &c, &why)) {
        GGML_ABORT("tesseract: the scaled MUL_MAT_ID result of '%s' cannot be moved into '%s': %s", node->name, mul->name, why.c_str());
    }
    if (c.kind != tsr::opl::K_NOP) {
        tsr_seg_add_rec(d, seg, c, std::string(node->name) + " -> " + mul->name, tsr_written_buf(d, mul), false);
    }
    for (int m : { sc->repeat, sc->get_rows, sc->mul }) {
        absorbed[(size_t) m] = 1;
    }
}

// SW §9.3 M7 passes 1-6 over one graph, launching each list as it fills (a split is one list unless a
// list fills, a MUL_MAT is staged or a src0 must be converted after the pending list ran).
// Debug knob: TSR_DUMP_RE=<regex> TSR_DUMP_DIR=<dir> runs the pending list after each node whose result name matches
// (an unfused-style cut: results of the other nodes are unchanged), reads the result tensor from device memory,
// prints its range to stderr and writes it to <dir>/<name>.f32. Only the first run of a graph builds records, so
// only that run dumps. Compare with a CPU run of the same model (a cb_eval on the CPU backend) to find the first
// node whose device value is off.
#include <regex>
static void tsr_dump_node(tsr_device * d, tsr_segment & seg, const ggml_tensor * res) {
    static const char * re_s = getenv("TSR_DUMP_RE");
    static const char * dir = getenv("TSR_DUMP_DIR");
    if (re_s == nullptr || dir == nullptr || res == nullptr || res->type != GGML_TYPE_F32) {
        return;
    }
    static const std::regex re(re_s);
    if (!std::regex_match(res->name, re)) {
        return;
    }
    tsr_flush(d, seg);
    tsr_buffer_ctx * b = tsr_written_buf(d, res);
    if (b == nullptr || b->mode != tsr_bmode::LINEAR) {
        return;
    }
    const size_t nb = ggml_nbytes(res);
    std::vector<float> v(nb / 4);
    tsr_read(d, b->ch, b->io_off + (tsr_vaddr(res->data) - b->base), v.data(), nb);
    double mn = 1e300, mx = -1e300, s2 = 0;
    size_t bad = 0;
    for (float x : v) {
        if (!std::isfinite(x)) { ++bad; continue; }
        mn = std::min<double>(mn, x); mx = std::max<double>(mx, x); s2 += (double) x * x;
    }
    fprintf(stderr, "DUMP %-28s @ch%u+line 0x%llx ne=[%lld,%lld,%lld,%lld] min %.4g max %.4g rms %.4g nonfinite %zu\n", res->name,
            b->ch, (unsigned long long) ((b->io_off + (tsr_vaddr(res->data) - b->base)) / 64),
            (long long) res->ne[0], (long long) res->ne[1], (long long) res->ne[2], (long long) res->ne[3], mn, mx,
            std::sqrt(s2 / std::max<size_t>(v.size(), 1)), bad);
    const std::string path = std::string(dir) + "/" + res->name + ".f32";
    if (FILE * f = fopen(path.c_str(), "wb")) {
        fwrite(v.data(), 4, v.size(), f);
        fclose(f);
    }
}

static void tsr_build_and_run(tsr_device * d, ggml_cgraph * cgraph, uint64_t key) {
    tsr_gctx g;
    g.env = tsr_env_of(d);
    g.fusion = tsr_build_fusion(g.env, cgraph);
    d->build_dynamic = false;
    std::vector<char> absorbed(cgraph->n_nodes, 0);
    std::vector<int32_t> window_of(cgraph->n_nodes, -1);   // a CONCAT of an SSM_CONV group: that SSM_CONV
    for (int i = 0; i < cgraph->n_nodes; i++) {          // the state gather and the state CPY of a GATED_DELTA_NET on its cache cell
        const tsr_gdn_cell & c = g.fusion.gdn_cell[(size_t) i];
        if (c.ids != nullptr) {
            g.plan.gdn_cells.push_back(c);
        }
        if (c.in_place) {
            absorbed[(size_t) c.gather] = 1;
            absorbed[(size_t) c.cpy] = 1;
        }
    }
    for (int i = 0; i < cgraph->n_nodes; i++) {          // the state CPY and the SILU of an SSM_CONV group; its CONCAT fills the window
        if (g.fusion.conv_concat[i] >= 0) {
            window_of[g.fusion.conv_concat[i]] = i;
            absorbed[g.fusion.conv_cpy[i]] = 1;
            if (g.fusion.conv_silu[i] >= 0) {
                absorbed[g.fusion.conv_silu[i]] = 1;
            }
        }
    }
    tsr_segment seg(d->cfg.num_ch, &g);
    std::map<uint32_t, tsr_kv_write> kv_writes;      // by plane base beat, as tsr_fanew_map
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        if (window_of[i] >= 0 && (node->flags & GGML_TENSOR_FLAG_COMPUTE) != 0) {
            tsr_rec recs[2];
            uint32_t nr = 0;
            std::string why;
            if (!tsr_lower_conv_window(g.env, false, node, recs, &nr, &why)) {
                GGML_ABORT("tesseract: CONCAT '%s' of an SSM_CONV group reached graph_compute but cannot run on the device: %s",
                           node->name, why.c_str());
            }
            for (uint32_t q = 0; q < nr; ++q) {
                tsr_seg_add_rec(d, seg, recs[q], node->name, tsr_written_buf(d, node), false);
            }
            TSR_DBG(1, "%-11s %-32s (window)\n", tsr_kind_name(recs[0].kind), node->name);
            continue;
        }
        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || tsr_node_is_noop(node) || absorbed[i]) {
            continue;                        // pass 1 (and the second node of a fused pair)
        }
        if (ggml_is_empty(node)) {
            continue;
        }
        if (node->op == GGML_OP_SSM_CONV) {
            // SW §7.10: the SSM_CONV record carries the group's CONCAT, state CPY and SILU
            tsr_rec r;
            std::string why;
            const ggml_tensor * silu = g.fusion.conv_silu[i] >= 0 ? cgraph->nodes[g.fusion.conv_silu[i]] : nullptr;
            const ggml_tensor * cpy = g.fusion.conv_cpy[i] >= 0 ? cgraph->nodes[g.fusion.conv_cpy[i]] : nullptr;
            if (cpy == nullptr || !tsr_lower_conv_group(g.env, false, node, silu, cpy, &r, &why)) {
                GGML_ABORT("tesseract: SSM_CONV '%s' reached graph_compute but cannot run on the device: %s", node->name,
                           cpy == nullptr ? "its CONCAT / state CPY group was not found in the graph" : why.c_str());
            }
            const std::string name = std::string(node->name) + (silu != nullptr ? " + SILU" : "");
            tsr_seg_add_rec(d, seg, r, name, tsr_written_buf(d, silu != nullptr ? silu : node), true);
            if (cpy->view_src != nullptr) {
                if (tsr_buffer_ctx * sb = tsr_written_buf(d, cpy)) {
                    seg.written.insert(sb);
                }
            }
            TSR_DBG(1, "%-11s %-32s\n", tsr_kind_name(r.kind), name.c_str());
            tsr_dump_node(d, seg, silu != nullptr ? silu : node);
            continue;
        }
        if (node->op == GGML_OP_GATED_DELTA_NET && g.fusion.gdn_cell[(size_t) i].in_place) {
            // SW-82: the record reads the cache cell the state CPY writes and writes the new state back into it
            const ggml_tensor * cpy = cgraph->nodes[g.fusion.gdn_cell[(size_t) i].cpy];
            tsr_rec r;
            std::string why;
            if (!tsr_lower_m7(g.env, false, node, cpy, &r, &why)) {
                GGML_ABORT("tesseract: GATED_DELTA_NET '%s' on its state cache cell cannot run on the device: %s", node->name,
                           why.c_str());
            }
            tsr_seg_add_rec(d, seg, r, std::string(node->name) + " (state in place)", tsr_written_buf(d, node), true);
            seg.n_gdn_cell++;
            if (tsr_buffer_ctx * sb = tsr_written_buf(d, cpy)) {
                seg.written.insert(sb);
            }
            TSR_DBG(1, "%-11s %-32s (state in place)\n", tsr_kind_name(r.kind), node->name);
            tsr_dump_node(d, seg, node);
            continue;
        }
        const int32_t k = g.fusion.with[i];
        if (g.fusion.route_of[i] >= 0) {                   // the MoE routing group (SW-64): one MOE_ROUTE record
            const tsr_moe_route & grp = g.fusion.routes[(size_t) g.fusion.route_of[i]];
            tsr_rec rs[TSR_MOE_ROUTE_RECS];
            uint32_t nr = 0;
            std::string why;
            if (!tsr_lower_moe_route(g.env, cgraph, grp, rs, &nr, &why)) {
                GGML_ABORT("tesseract: MoE routing at '%s' cannot run on the device: %s", node->name, why.c_str());
            }
            for (int m : { grp.argsort, grp.get_rows, grp.sum_rows, grp.clamp, grp.div }) {
                absorbed[(size_t) m] = 1;
            }
            // one record, or (several tiles, SW-76) the record into the scratch and the CPYs of the ids and the weights
            const std::string rname = std::string(node->name) + " + " + cgraph->nodes[grp.div]->name;
            tsr_buffer_ctx * ib = tsr_written_buf(d, cgraph->nodes[grp.argsort]);
            tsr_buffer_ctx * wb = tsr_written_buf(d, cgraph->nodes[grp.div]);
            tsr_seg_add_rec(d, seg, rs[0], rname, nr == 1 ? wb : nullptr, true);
            if (nr == 3) {
                tsr_seg_add_rec(d, seg, rs[1], rname + " (ids)", ib, false);
                tsr_seg_add_rec(d, seg, rs[2], rname + " (weights)", wb, false);
            }
            if (ib != nullptr) {
                seg.written.insert(ib);
            }
            TSR_DBG(1, "MOE_ROUTE   %-32s%s\n", node->name, nr == 3 ? " (scratch, 2 CPY)" : "");
            continue;
        }
        if (node->op == GGML_OP_MUL_MAT_ID) {              // MUL_MAT_ID (+ the scale group) (SW-62)
            tsr_lower_mul_mat_id_node(d, seg, g, cgraph, i, absorbed);
            continue;
        }
        if (node->op == GGML_OP_MUL_MAT) {
            const int32_t tail = g.fusion.softcap_tail[i];
            if (tsr_lower_mul_mat(d, seg, node, k >= 0 ? cgraph->nodes[k] : nullptr,
                                  tail >= 0 ? cgraph->nodes[tail] : nullptr) && k >= 0) {
                absorbed[k] = 1;
                if (tail >= 0) {
                    absorbed[g.fusion.softcap_mid[i]] = 1;
                    absorbed[tail] = 1;
                }
            }
            if (tsr_stats_on()) {
                d->st_mm++;
            }
            tsr_dump_node(d, seg, absorbed[k >= 0 ? k : i] && k >= 0 ? cgraph->nodes[k] : node);
            continue;
        }
        const ggml_tensor * mul = node->op == GGML_OP_RMS_NORM && k >= 0 ? cgraph->nodes[k] : nullptr;
        tsr_rec r;
        std::string why;
        bool ok = tsr_lower_m7(g.env, false, node, mul, &r, &why);
        if (!ok && mul != nullptr) {
            TSR_DBG(1, "RMS_NORM %s + MUL %s not fused: %s\n", node->name, mul->name, why.c_str());
            mul = nullptr;
            ok = tsr_lower_m7(g.env, false, node, nullptr, &r, &why);
        }
        if (!ok) {
            const ggml_tensor * h = tsr_hosted_operand(d, node, mul);
            if (h != nullptr) {
                GGML_ABORT("tesseract: op %s ('%s') reached graph_compute but cannot run on the device: '%s' is in host "
                           "memory (a compute buffer the device had no window for); the M7 kinds address their operands "
                           "in place and only MUL_MAT is staged (docs/design/06-software-verification.md §8 R6): use a "
                           "smaller -ub or -c, or keep the device to MUL_MAT (TSR_KINDS=m6)", ggml_op_desc(node), node->name,
                           h->name);
            }
            GGML_ABORT("tesseract: op %s ('%s') reached graph_compute but cannot run on the device: %s", ggml_op_desc(node),
                       node->name, why.c_str());
        }
        if (mul != nullptr) {
            absorbed[k] = 1;
        }
        if (r.kind == tsr::opl::K_FLASH_ATTN) {
            g.plan.fa_nodes.push_back(node);
            g.plan.fa_refs.push_back(r.mask);
        }
        tsr_note_kv_write(r, &g.fanew);
        const std::string name = mul != nullptr ? std::string(node->name) + " + " + mul->name : std::string(node->name);
        tsr_seg_add_rec(d, seg, r, name, tsr_written_buf(d, mul != nullptr ? mul : node), mul != nullptr);
        for (tsr_rec & more : r.more) {
            tsr_seg_add_rec(d, seg, more, name, tsr_written_buf(d, mul != nullptr ? mul : node), mul != nullptr);
        }
        if (r.kind == tsr::opl::K_SET_ROWS_KV) {
            kv_writes[r.kv_plane] = { node->src[1], i, g.plan.lists.size() };
        } else if (r.kind == tsr::opl::K_FLASH_ATTN && (r.args[tsr::opl::fa::FLAGS] & 1u)) {
            const auto w = kv_writes.find(r.kv_plane);
            if (w != kv_writes.end() && w->second.list != g.plan.lists.size()) {
                tsr_check_fanew_source(cgraph, w->second, i);     // a cut lies between them
            }
        }
        TSR_DBG(1, "%-11s %-32s%s\n", tsr_kind_name(r.kind), name.c_str(),
                r.kind == tsr::opl::K_FLASH_ATTN && (r.args[tsr::opl::fa::FLAGS] & 1u) ? " (FA_NEW)" : "");
        tsr_dump_node(d, seg, mul != nullptr ? mul : node);
    }
    tsr_flush(d, seg);
    if (d->build_dynamic) {
        d->st_plan_dynamic++;
        return;
    }
    // R7: keep the plan for this graph identity (at most 16 graphs, least recently used out)
    if (d->plans.size() >= 16) {
        auto lru = d->plans.begin();
        for (auto it = d->plans.begin(); it != d->plans.end(); ++it) {
            if (it->second.used < lru->second.used) {
                lru = it;
            }
        }
        d->plans.erase(lru);
    }
    g.plan.used = ++d->plan_stamp;
    d->plans[key] = std::move(g.plan);
}

// ---------------------------------------------------------------------------
// ggml backend interface
// ---------------------------------------------------------------------------

struct tsr_backend_ctx {
    tsr_device * dev;
};

static const char * tsr_backend_get_name(ggml_backend_t backend) {
    GGML_UNUSED(backend);
    return TSR_DEV_NAME;
}

static void tsr_backend_free(ggml_backend_t backend) {
    tsr_device * d = ((tsr_backend_ctx *) backend->context)->dev;
    if (tsr_stats_on() && !d->st_perf0.empty()) {
        std::lock_guard<std::mutex> lock(d->mtx);
        std::vector<uint64_t> p = tsr_perf_all(d);
        for (size_t i = 0; i < p.size(); i++) {
            p[i] -= d->st_perf0[i];
        }
        const uint64_t passes = d->st_passes + (d->st_pass_splits.empty() ? 0 : 1);
        const uint64_t launches = d->odev.launches;
        const tsr_opl_cache::stats & cs = d->odev.cache().get_stats();
        fprintf(stderr, "[tesseract] stats: graph_compute %" PRIu64 " calls, %.3f s; launches %" PRIu64 " (%.3f s wall), "
                "%.2f per scheduler pass (%" PRIu64 " passes, ~tokens), %.2f op records per launch\n",
                d->st_graphs, d->st_compute_ns * 1e-9, launches, d->odev.launch_ns * 1e-9,
                passes ? (double) launches / passes : 0.0, passes, launches ? (double) d->odev.ops / launches : 0.0);
        fprintf(stderr, "[tesseract] stats: MUL_MAT nodes lowered %" PRIu64 ", MUL_MAT records %" PRIu64 ", JOIN records %" PRIu64
                ", staged chunks %" PRIu64 " (%" PRIu64 " launches); OPL cache %" PRIu64 " hits, %" PRIu64 " misses, %" PRIu64
                " evictions, %zu lists / %.2f MiB resident\n",
                d->st_mm, d->st_mm_records, d->st_joins, d->st_staged_mm, d->st_staged_launches, cs.hits, cs.misses,
                cs.evictions, d->odev.cache().entries(), tsr_mib(d->odev.cache().used_bytes()));
        // W3b (TSR_STATS, SW §9.2): launches per graph_compute, fused ops, records per kind, the
        // plan cache (R7) and lists cut early
        fprintf(stderr, "[tesseract] stats: graph_compute calls with 0 / 1 / 2 / >= 3 launches: %" PRIu64 " / %" PRIu64 " / %" PRIu64
                " / %" PRIu64 "; fused ops %" PRIu64 "; plan cache %" PRIu64 " hits, %" PRIu64 " builds (%" PRIu64
                " not cacheable); lists cut early %" PRIu64 "\n", d->st_launch_hist[0], d->st_launch_hist[1], d->st_launch_hist[2],
                d->st_launch_hist[3], d->st_fused, d->st_plan_hits, d->st_plan_misses, d->st_plan_dynamic, d->st_cut_lists);
        std::string kinds;
        for (uint32_t k = 0; k < tsr::opl::N_KINDS; ++k) {
            if (d->st_kind[k] != 0) {
                char b[64];
                snprintf(b, sizeof(b), "%s%s %" PRIu64, kinds.empty() ? "" : ", ", tsr_kind_name(k), d->st_kind[k]);
                kinds += b;
            }
        }
        fprintf(stderr, "[tesseract] stats: records per kind: %s\n", kinds.empty() ? "none" : kinds.c_str());
        fprintf(stderr, "[tesseract] stats: GATED_DELTA_NET records on their state cache cell (no gather, no copy): %" PRIu64 " of %" PRIu64 "\n",
                d->st_gdn_cell, d->st_kind[tsr::opl::K_GATED_DELTA_NET]);
        fprintf(stderr, "[tesseract] stats: PERF_CYCLES %" PRIu64 ", MVM %" PRIu64 ", RETIRED %" PRIu64 "\n",
                p[TSR_PERF_CYCLES], p[TSR_PERF_MVM_CYCLES], p[TSR_PERF_RETIRED]);
        const size_t nch = (p.size() - 5) / 2;
        {   // cycles in which a unit holds an op (EX §6; since the device was opened): the VU, DMA, QACT and AT units
            auto rd64 = [d](uint32_t idx) {
                const uint32_t lo = tsr_transport_csr_read(d->tp, TSR_CSR_PERF_BASE + 8 * idx);
                const uint32_t hi = tsr_transport_csr_read(d->tp, TSR_CSR_PERF_BASE + 8 * idx + 4);
                return ((uint64_t) hi << 32) | lo;
            };
            fprintf(stderr, "[tesseract] stats: unit busy cycles VU %" PRIu64 ", DMA %" PRIu64 ", QACT %" PRIu64 ", ATT %" PRIu64 "\n",
                    rd64(TSR_PERF_VU_CYCLES), rd64(TSR_PERF_EXT((uint32_t) nch, TSR_PERF_XOFF_DMA_CYCLES)),
                    rd64(TSR_PERF_EXT((uint32_t) nch, TSR_PERF_XOFF_QACT_CYCLES)),
                    rd64(TSR_PERF_EXT((uint32_t) nch, TSR_PERF_XOFF_ATT_CYCLES)));
        }
        for (size_t c = 0; c < nch; c++) {
            fprintf(stderr, "[tesseract] stats: ch%zu rd_beats %" PRIu64 ", wait %" PRIu64 "\n", c, p[5 + c], p[5 + nch + c]);
        }
    }
    if (tsr_stats_on()) {
        std::lock_guard<std::mutex> lock(d->mtx);
        fprintf(stderr, "[tesseract] stats: set_tensor %.3f s for %.3f GB, of which transport writes %.3f s for %.3f GB (%.2f GB/s)\n",
                d->st_set_ns * 1e-9, d->st_set_bytes * 1e-9, d->st_write_ns * 1e-9, d->st_write_bytes * 1e-9,
                d->st_write_ns ? d->st_write_bytes / (double) d->st_write_ns : 0.0);
        fprintf(stderr, "[tesseract] stats: weight stores %.3f s (%u load threads), conversion to Q8_0 %.3f s of CPU time\n",
                d->st_store_ns * 1e-9, tsr_load_threads(), d->st_conv_ns.load() * 1e-9);
        if (d->msum.ready()) {   // W3b kv
            fprintf(stderr, "[tesseract] stats: mask summaries %" PRIu64 " (GENERAL %" PRIu64 ", promoted %" PRIu64 "), f32 mask "
                    "copies %" PRIu64 ", re-produced for new readers %" PRIu64 "\n", d->msum.st_produced, d->msum.st_general,
                    d->msum.st_promoted, d->msum.st_copies, d->msum.st_reattach);
        }
    }
    delete (tsr_backend_ctx *) backend->context;
    delete backend;
}

// SW §9.3: the split as one op list (M6: its MUL_MATs; M7: every node, fused, with the preload table),
// one launch; staged MUL_MATs on their own. The plan of a graph is cached by its identity (R7): a
// decode token's graph runs its lists again without a rebuild while only the inputs change.
static int32_t tsr_param_i32_of(const ggml_tensor * t, int i) {
    int32_t v;
    memcpy(&v, (const char *) t->op_params + 4 * i, 4);
    return v;
}

// IMROPE ROPEs run as NEOX on the first n_tokens positions (the temporal stream, tsr_lower_rope). That is the
// rotation of text tokens, whose [t, h, w] positions are equal; a batch with image or video positions would
// rotate differently, so the position input is checked before the graph runs (SW §7.10). Holds d->mtx.
static void tsr_check_imrope_pos(tsr_device * d, const ggml_cgraph * cg) {
    const ggml_tensor * checked = nullptr;
    for (int i = 0; i < cg->n_nodes; ++i) {
        const ggml_tensor * n = cg->nodes[i];
        if (n->op != GGML_OP_ROPE || tsr_param_i32_of(n, 2) != GGML_ROPE_TYPE_IMROPE || n->src[1] == checked ||
            (n->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }
        const ggml_tensor * pos = n->src[1];
        checked = pos;
        ggml_backend_buffer_t gb = pos->buffer;
        if (pos->type != GGML_TYPE_I32 || gb == nullptr || gb->buft != &d->buft || gb->context == nullptr) {
            continue;
        }
        tsr_buffer_ctx * b = tsr_buf(gb);
        const size_t nb = ggml_nbytes(pos);
        std::vector<int32_t> v(nb / 4);
        const uint64_t addr = tsr_vaddr(pos->data);
        if (b->mode == tsr_bmode::LINEAR) {
            tsr_read(d, b->ch, b->io_off + (addr - b->base), v.data(), nb);
        } else if (b->mode == tsr_bmode::HOSTED) {
            memcpy(v.data(), b->host + (addr - b->base), nb);
        } else {
            continue;
        }
        const size_t T = v.size() / 4;
        for (size_t t = 0; t < T; ++t) {
            if (v[t] != v[T + t] || v[t] != v[2 * T + t]) {
                GGML_ABORT("tesseract: the M-RoPE positions of token %zu differ between sections (t %d, h %d, w %d): the device "
                           "rotates with the temporal position only (image / video input); keep ROPE on the CPU with "
                           "TSR_KINDS=all,-ROPE (docs/design/06-software-verification.md SW §7.10)", t, v[t], v[T + t], v[2 * T + t]);
            }
        }
    }
}

static enum ggml_status tsr_backend_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    tsr_device * d = ((tsr_backend_ctx *) backend->context)->dev;
    std::lock_guard<std::mutex> lock(d->mtx);
    tsr_check_imrope_pos(d, cgraph);
    tsr_mon_note_graph(d, cgraph);
    const uint64_t t_start = tsr_stats_on() ? tsr_now_ns() : 0;
    const uint64_t launches0 = d->odev.launches;
    const uint64_t key = tsr_graph_key(cgraph, d->plan_epoch);
    auto it = d->plans.find(key);
    if (it != d->plans.end() && tsr_plan_masks_hold(d, it->second) && tsr_plan_state_holds(d, it->second)) {
        it->second.used = ++d->plan_stamp;
        TSR_DBG(1, "graph %p (%d nodes): cached plan, %zu list(s)\n", (void *) cgraph, cgraph->n_nodes, it->second.lists.size());
        for (const tsr_plan_list & L : it->second.lists) {
            tsr_run_list(d, L);
        }
        d->st_plan_hits++;
    } else {
        if (it != d->plans.end()) {
            d->plans.erase(it);              // a mask summary moved or a state gather changed its mind: build again
        }
        d->st_plan_misses++;
        tsr_build_and_run(d, cgraph, key);
    }
    if (tsr_stats_on()) {
        d->st_launch_hist[std::min<uint64_t>(d->odev.launches - launches0, 3)]++;
        d->st_graphs++;
        d->st_compute_ns += tsr_now_ns() - t_start;
        // a scheduler pass (one llama_decode, ~ one token) calls each split once: a split seen
        // again starts the next pass
        if (d->st_pass_splits.count(cgraph) != 0) {
            d->st_passes++;
            d->st_pass_splits.clear();
        }
        d->st_pass_splits.insert(cgraph);
    }
    return GGML_STATUS_SUCCESS;
}

static ggml_guid_t tsr_backend_guid() {
    static ggml_guid guid = { 0x54, 0x53, 0x52, 0x31, 0x9e, 0x2b, 0x4c, 0x61, 0xa3, 0x0f, 0x77, 0x1d, 0x5e, 0xc8, 0x42, 0x10 };
    return &guid;
}

static const ggml_backend_i tsr_backend_iface = {
    /* .get_name            = */ tsr_backend_get_name,
    /* .free                = */ tsr_backend_free,
    /* .set_tensor_async    = */ nullptr,
    /* .get_tensor_async    = */ nullptr,
    /* .set_tensor_2d_async = */ nullptr,
    /* .get_tensor_2d_async = */ nullptr,
    /* .cpy_tensor_async    = */ nullptr,
    /* .synchronize         = */ nullptr,
    /* .graph_plan_create   = */ nullptr,
    /* .graph_plan_free     = */ nullptr,
    /* .graph_plan_update   = */ nullptr,
    /* .graph_plan_compute  = */ nullptr,
    /* .graph_compute       = */ tsr_backend_graph_compute,
    /* .event_record        = */ nullptr,
    /* .event_wait          = */ nullptr,
    /* .graph_optimize      = */ nullptr,
};

// ---------------------------------------------------------------------------
// ggml device interface
// ---------------------------------------------------------------------------

static const char * tsr_dev_get_name(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return TSR_DEV_NAME;
}

static const char * tsr_dev_get_description(ggml_backend_dev_t dev) {
    tsr_device * d = (tsr_device *) dev->context;
    std::lock_guard<std::mutex> lock(d->mtx);
    tsr_dev_open_locked(d);
    return d->description.c_str();
}

static void tsr_dev_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    tsr_device * d = (tsr_device *) dev->context;
    std::lock_guard<std::mutex> lock(d->mtx);
    if (!tsr_dev_open_locked(d)) {
        *free = 0;
        *total = 0;
        return;
    }
    // weights dominate device memory: report the symmetric WEIGHT heap over all channels
    *free  = (size_t) (d->heap.free_bytes() * d->cfg.num_ch);
    *total = (size_t) (d->heap.total()      * d->cfg.num_ch);
}

static enum ggml_backend_dev_type tsr_dev_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void tsr_dev_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = tsr_dev_get_name(dev);
    props->description = tsr_dev_get_description(dev);
    props->type        = tsr_dev_get_type(dev);
    props->device_id   = nullptr;
    tsr_dev_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                = */ false,
        /* .host_buffer          = */ false,
        /* .buffer_from_host_ptr = */ false,
        /* .events               = */ false,
        /* .mmap_support         = */ true,
    };
}

static ggml_backend_t tsr_dev_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    tsr_device * d = (tsr_device *) dev->context;
    {
        std::lock_guard<std::mutex> lock(d->mtx);
        if (!tsr_dev_open_locked(d)) {
            return nullptr;
        }
    }
    return new ggml_backend {
        /* .guid    = */ tsr_backend_guid(),
        /* .iface   = */ tsr_backend_iface,
        /* .device  = */ dev,
        /* .context = */ new tsr_backend_ctx { d },
    };
}

static ggml_backend_buffer_type_t tsr_dev_get_buffer_type(ggml_backend_dev_t dev) {
    return &((tsr_device *) dev->context)->buft;
}

static bool tsr_is_our_buffer(const tsr_device * d, const ggml_backend_buffer * buf) {
    return buf != nullptr && buf->buft == &d->buft;
}

// spec "supports_op", unchanged in M6 (SW §9.3): MUL_MAT only, with a
// 2D weight src0 that the device can hold (Q8_0 / Q4_K natively, anything else with to_float
// converted to Q8_0) and F32 activations; the shape must also be one the firmware can plan
// (plan.hpp mm_supported on the device's CFG values, SW-20: the host rejects exactly what the
// `graph` kernel would refuse). The no-op ops (leaves and views) are "supported" as on every GPU
// backend: they cost nothing in graph_compute, test-backend-ops asks for every tensor of a test
// graph, and the scheduler aborts on a view of a Tesseract-resident tensor that no backend claims.
//
// M7 (W3b, SW §9.3 supports_op (M7), AT §2.6): every other op of the table through tsr-build's
// tsr_build_supports: the node is lowered to its OPL record in probe mode (the same function the
// builder encodes with) and the kind's planner (plan.hpp plan_kind on the device's CFG values) must
// accept it, so the host refuses exactly what the firmware cannot plan or the builder cannot encode.
// SW §6.9: only the kinds the booted image declares are routed (the device's kind mask, which
// TSR_KINDS narrows, e.g. m6 to M6's MUL_MAT, SW-14): a subset image leaves the others on the CPU.
static bool tsr_dev_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    tsr_device * d = (tsr_device *) dev->context;
    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;
        case GGML_OP_MUL_MAT:
            break;
        default: {
            std::lock_guard<std::mutex> lock(d->mtx);
            if (!tsr_dev_open_locked(d)) {
                return false;
            }
            std::string why;
            const bool ok = tsr_build_supports(tsr_env_of(d), op, &why);
            if (!ok) {
                TSR_DBG(2, "supports_op: %s '%s' not on the device: %s\n", ggml_op_desc(op), op->name, why.c_str());
                // SW §8 R10: an op writing into a device KV cache other than SET_ROWS (llama.cpp's context
                // shift ROPEs the cache, defragmentation copies cells) makes the scheduler abort on the
                // pre-allocated tensor; say why first
                const ggml_tensor * base = op->view_src != nullptr ? op->view_src : op;
                tsr_kv_shape kvs;
                if (op->op != GGML_OP_SET_ROWS && base != op && tsr_kv_tensor_supported(d, base)) {
                    static std::once_flag warned;
                    std::call_once(warned, [op] {
                        GGML_LOG_ERROR("tesseract: %s '%s' writes into the device KV cache: context shift and KV "
                                       "defragmentation are not supported in M7 (docs/design/06-software-verification.md §8 "
                                       "R10); keep the cache on the CPU (-nkvo) or disable context shift\n",
                                       ggml_op_desc(op), op->name);
                    });
                } else if (op->op == GGML_OP_SET_ROWS && base != op && tsr_kv_probe_tensor(d, base, &kvs, false)) {
                    // W3b fix: a SET_ROWS into a device KV cache must run here (the scheduler stops on the
                    // pre-allocated tensor otherwise): name the cause first, e.g. several sequence streams
                    // (llama.cpp -np N without --kv-unified) or TSR_FORCE_HOST's host-memory compute buffers
                    static std::once_flag warned_kv;
                    const bool force_host = d->force_host;
                    std::call_once(warned_kv, [op, &kvs, &why, force_host] {
                        if (kvs.n_stream > 1) {
                            GGML_LOG_ERROR("tesseract: SET_ROWS '%s' writes a device KV cache of %u sequence streams: "
                                           "multi-sequence KV streams are not supported in M7 "
                                           "(docs/design/06-software-verification.md §8 R10); use --kv-unified or keep the "
                                           "cache on the CPU (-nkvo)\n", op->name, kvs.n_stream);
                        } else {
                            // the cell indices are one preload entry: at most 256 lines of I64 per ubatch
                            const int64_t max_rows = (int64_t) tsr::opl::PRE_MAX_LINES * tsr::opl::PRE_ALIGN /
                                                     tsr::opl::PRE_KVIDX_BYTES;
                            const bool many = op->src[0] != nullptr && ggml_nrows(op->src[0]) > max_rows;
                            GGML_LOG_ERROR("tesseract: SET_ROWS '%s' into the device KV cache cannot run on the device "
                                           "(%s)%s; %skeep the cache on the CPU (-nkvo)\n", op->name, why.c_str(),
                                           force_host ? ": TSR_FORCE_HOST keeps the compute buffers in host memory" : "",
                                           many ? "use -ub <= 2048 (the device writes at most 2048 cells per ubatch) or " : "");
                        }
                    });
                } else if (op->op == GGML_OP_SET_ROWS && base != op && base->type != GGML_TYPE_Q8_0 &&
                           tsr_is_our_buffer(d, base->buffer) && tsr_kv_parse_name(base->name, nullptr, nullptr)) {
                    // llama.cpp's default KV cache type is f16 (llama-server, llama-cli without
                    // -ctk / -ctv): a cache of another type in device memory makes the scheduler stop on the
                    // pre-allocated tensor with no hint; name the flags first
                    static std::once_flag warned_kv_type;
                    std::call_once(warned_kv_type, [op, base] {
                        GGML_LOG_ERROR("tesseract: SET_ROWS '%s' writes a %s KV cache in device memory: the device "
                                       "holds q8_0 K and V only (AT §2.6); use -ctk q8_0 -ctv q8_0 (with -fa on) or keep "
                                       "the cache on the CPU (-nkvo)\n", op->name, ggml_type_name(base->type));
                    });
                }
            }
            return ok;
        }
    }
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    if (src0 == nullptr || src1 == nullptr) {
        return false;
    }
    if (op->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32) {
        return false;
    }
    if (src0->ne[2] != 1 || src0->ne[3] != 1) {
        return false;
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(op)) {
        return false;
    }
    if (!tsr_type_loadable(src0->type)) {
        return false;
    }
    tsr_config cfg;
    {
        std::lock_guard<std::mutex> lock(d->mtx);
        if (!tsr_dev_open_locked(d)) {
            return false;
        }
        if (!tsr_kind_in_image(d->kind_mask, tsr::opl::K_MUL_MAT)) {   // SW §6.9: a subset image without MUL_MAT
            TSR_DBG(2, "supports_op: %s '%s' not on the device: kind MUL_MAT is not in the booted image (%s)\n",
                    ggml_op_desc(op), op->name, d->image_desc.c_str());
            return false;
        }
        cfg = d->cfg;
    }
    const uint32_t fmt = tsr_dev_fmt_for_type(src0->type);
    if (!tsr_mm_shape_ok(cfg, fmt, (uint64_t) src0->ne[1], (uint64_t) src0->ne[0])) {
        return false;                         // K > K_MAX, K % 32 != 0, or one token exceeds SPM
    }
    if (src0->ne[1] > (int64_t) UINT32_MAX ||
        !tsr::plan::mm_supported(tsr_plan_cfg(cfg), fmt, (uint32_t) src0->ne[1], (uint32_t) src0->ne[0])) {
        return false;                         // the firmware cannot plan one token in its ring budget
    }
    if (ggml_nrows(src1) > (int64_t) UINT32_MAX) {
        return false;
    }
    // No limit on the token count: the size of src1 / dst grows with T, but the activations are
    // never all resident at once. Compute buffers that do not fit the IO region live in host
    // memory and graph_compute stages them through IO in T tiles (spec "lowering constraints"). A
    // T-dependent rejection here would also misplace weights: llama.cpp decides where a weight
    // lives with a fixed 512-token probe (weight_buft_supported), so the output head of any
    // large-vocabulary model would silently stay on the CPU, and a resident weight rejected at
    // run time is copied back from the device for the CPU on every graph.
    //
    // src0 must be a weight. It is judged by the tensor it views (if a view):
    //   - in a Tesseract buffer that is not a compute buffer: WEIGHTS (device format) or ANY
    //     (raw bytes, converted lazily; includes llama.cpp's size-0 placement-probe buffer);
    //   - unallocated: only a leaf that is not a graph input. test-backend-ops asks before it
    //     allocates; in llama.cpp scheduling every weight is allocated, so an unallocated src0
    //     there is a graph input or an activation (e.g. attn_inp_k_rot, the Hadamard rotation
    //     that a quantized KV cache applies to Q) and stays on the CPU.
    const ggml_tensor * base0 = src0->view_src ? src0->view_src : src0;
    if (strncmp(base0->name, "attn_inp_k_rot", 14) == 0 || strncmp(base0->name, "attn_inp_v_rot", 14) == 0) {
        // SW §8 R4: a graph input, not a weight (rejected either way); device runs and their
        // baselines turn llama.cpp's Hadamard KV rotation off. (In M6 the scheduler assigns these
        // MUL_MATs to the CPU by their input's buffer without asking, so this fires only when a
        // caller offers one.)
        static std::once_flag warned;
        std::call_once(warned, [] {
            GGML_LOG_WARN("tesseract: the graph rotates Q/K/V with a Hadamard matrix (attn_inp_k_rot / attn_inp_v_rot, "
                          "quantized KV cache): those MUL_MATs stay on the CPU; set LLAMA_ATTN_ROT_DISABLE=1 for "
                          "device runs and their baselines (docs/design/06-software-verification.md §8 R4)\n");
        });
        return false;
    }
    const ggml_backend_buffer * buf0 = base0->buffer;
    if (buf0 == nullptr) {
        if (base0->op != GGML_OP_NONE || (base0->flags & GGML_TENSOR_FLAG_INPUT) != 0) {
            return false;
        }
    } else {
        if (!tsr_is_our_buffer(d, buf0) || buf0->usage == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
            return false;
        }
        if (buf0->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS && src0->view_src != nullptr) {
            const ggml_tensor * v = src0->view_src;
            if (src0->view_offs != 0 || v->type != src0->type || v->ne[0] != src0->ne[0] || v->ne[1] != src0->ne[1] ||
                v->ne[2] != 1 || v->ne[3] != 1) {
                return false;                 // a slice of a repacked weight cannot be addressed
            }
        }
    }
    // src1: activations; must be 64-byte aligned on the device (LD is line/beat granular)
    const ggml_backend_buffer * buf1 = src1->view_src ? src1->view_src->buffer : src1->buffer;
    if (buf1 != nullptr && tsr_is_our_buffer(d, buf1)) {
        if (buf1->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            return false;
        }
        if (src1->data != nullptr && tsr_vaddr(src1->data) % TSR_TENSOR_ALIGN != 0) {
            return false;
        }
    }
    if (src1->view_src != nullptr && src1->view_offs % TSR_TENSOR_ALIGN != 0) {
        return false;
    }
    return true;
}

static bool tsr_dev_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return buft == &((tsr_device *) dev->context)->buft;
}

static bool tsr_dev_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);
    GGML_UNUSED(op);
    return false;
}

static const ggml_backend_device_i tsr_device_iface = {
    /* .get_name             = */ tsr_dev_get_name,
    /* .get_description      = */ tsr_dev_get_description,
    /* .get_memory           = */ tsr_dev_get_memory,
    /* .get_type             = */ tsr_dev_get_type,
    /* .get_props            = */ tsr_dev_get_props,
    /* .init_backend         = */ tsr_dev_init_backend,
    /* .get_buffer_type      = */ tsr_dev_get_buffer_type,
    /* .get_host_buffer_type = */ nullptr,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ tsr_dev_supports_op,
    /* .supports_buft        = */ tsr_dev_supports_buft,
    /* .offload_op           = */ tsr_dev_offload_op,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

// ---------------------------------------------------------------------------
// ggml backend registry
// ---------------------------------------------------------------------------

static bool tsr_disabled() {
    const char * e = getenv("TSR_DEVICE");
    return e != nullptr && strcmp(e, "none") == 0;
}

static const char * tsr_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return TSR_REG_NAME;
}

static size_t tsr_reg_get_device_count(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return tsr_disabled() ? 0 : 1;
}

static ggml_backend_dev_t tsr_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_UNUSED(reg);
    GGML_ASSERT(index == 0 && !tsr_disabled());
    return &tsr_dev_singleton()->gdev;
}

// Test hook: storage chosen for a Tesseract buffer ("undecided", "io", "placed", "host"), or
// NULL for a buffer of another type. Reached through
// ggml_backend_reg_get_proc_address(reg, "ggml_backend_tesseract_buffer_storage").
static const char * tsr_buffer_storage(ggml_backend_buffer_t buffer) {
    tsr_device * d = tsr_dev_singleton();
    if (buffer == nullptr || buffer->buft != &d->buft || buffer->context == nullptr) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(d->mtx);
    return tsr_bmode_name(tsr_buf(buffer)->mode);
}

// Hook "ggml_backend_tesseract_versions" (SW §6.9; klcheck, numcheck): the versions of the opened
// device and its booted image (tsr_transport_versions); false if the device is not open.
static bool tsr_hook_versions(ggml_backend_dev_t dev, tsr_versions * v) {
    if (dev == nullptr || v == nullptr) {
        return false;
    }
    tsr_device * d = (tsr_device *) dev->context;
    std::lock_guard<std::mutex> lock(d->mtx);
    if (d->tp == nullptr) {
        return false;
    }
    *v = d->odev.versions();
    return true;
}

static void * tsr_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    if (name != nullptr && strcmp(name, "ggml_backend_tesseract_versions") == 0) {
        return (void *) tsr_hook_versions;
    }
    if (name != nullptr && strcmp(name, "ggml_backend_tesseract_buffer_storage") == 0) {
        return (void *) tsr_buffer_storage;
    }
    // W3b kv: test hooks (tests/test-tsr-attn.cpp)
    if (name != nullptr && strcmp(name, "ggml_backend_tesseract_kv_plane") == 0) {
        return (void *) tsr_hook_kv_plane;
    }
    if (name != nullptr && strcmp(name, "ggml_backend_tesseract_mask_summary") == 0) {
        return (void *) tsr_hook_mask_summary;
    }
    if (name != nullptr && strcmp(name, "ggml_backend_tesseract_att_cfg") == 0) {
        return (void *) tsr_hook_att_cfg;
    }
    return nullptr;
}

static const ggml_backend_reg_i tsr_reg_iface = {
    /* .get_name         = */ tsr_reg_get_name,
    /* .get_device_count = */ tsr_reg_get_device_count,
    /* .get_device       = */ tsr_reg_get_device,
    /* .get_proc_address = */ tsr_reg_get_proc_address,
};

ggml_backend_reg_t ggml_backend_tesseract_reg(void) {
    static ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ tsr_reg_iface,
        /* .context     = */ nullptr,
    };
    static std::once_flag once;
    std::call_once(once, [] {
        tsr_device * d = tsr_dev_singleton();
        d->gdev = {
            /* .iface   = */ tsr_device_iface,
            /* .reg     = */ &reg,
            /* .context = */ d,
        };
        d->buft = {
            /* .iface   = */ tsr_buft_iface,
            /* .device  = */ &d->gdev,
            /* .context = */ d,
        };
    });
    return &reg;
}

ggml_backend_t ggml_backend_tesseract_init(void) {
    ggml_backend_reg_t reg = ggml_backend_tesseract_reg();
    if (ggml_backend_reg_dev_count(reg) == 0) {
        return nullptr;
    }
    return ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, 0), nullptr);
}

bool ggml_backend_is_tesseract(ggml_backend_t backend) {
    return backend != nullptr && ggml_guid_matches(backend->guid, tsr_backend_guid());
}

ggml_backend_buffer_type_t ggml_backend_tesseract_buffer_type(void) {
    ggml_backend_tesseract_reg();
    return &tsr_dev_singleton()->buft;
}

GGML_BACKEND_DL_IMPL(ggml_backend_tesseract_reg)
