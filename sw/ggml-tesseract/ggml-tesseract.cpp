// ggml-tesseract.cpp — ggml backend for the Tesseract overlay core (spec "ggml 백엔드 통합").
//
//   reg "Tesseract" -> one device "Tesseract0" (GGML_BACKEND_DEVICE_TYPE_GPU, so -ngl offloads to it)
//   buffer type     -> a virtual address space; each buffer's storage is fixed at its first data
//                      access, by usage (llama.cpp sets it only after init_tensor):
//       WEIGHTS-usage buffers ("PLACED"): per tensor. MUL_MAT weights (2D, loadable type) are
//           converted to the device format (Q8_0 / Q4_K, spec 로드 시 변환), repacked
//           (tsr_repack_row) and stored with symmetric allocation in the WEIGHT heap of every
//           channel. Anything else (1D norms, odd shapes) is stored raw in AUX (ch0).
//       ANY / COMPUTE buffers: raw bytes, byte-for-byte like any GPU buffer (activations,
//           test-backend-ops tensors), either
//           "LINEAR": the whole buffer is one window in the ch0 IO region, read and written by
//               the device in place; or
//           "HOSTED": host memory, when the buffer does not fit the free IO windows (e.g. the
//               worst-case logits of the output head that llama.cpp reserves for n_ubatch
//               outputs) or holds only leaf tensors (data such as LoRA adapters, which no
//               device op addresses directly). Device ops on such buffers are staged through
//               the IO staging area (top of IO), which open reserves.
//   graph_compute   -> nothing to execute: see "Between X0 and the op-list path" below.
//
// Between X0 and the op-list path (docs/v2/06-software-verification.md §9.3, P3.2): v1's MUL_MAT
// execution (host-emitted LD/QACT/MVM/ST command buffers run from the CMD region, the program
// cache, the staged and lazily converted MUL_MATs) is deleted with the command-buffer path (X0,
// D14). The device and its buffer type stay registered, so weights still upload, convert and read
// back exactly as before; supports_op accepts only the no-op ops (leaves and views, which the
// scheduler needs a backend to claim for Tesseract-resident tensors) and graph_compute skips
// them, so every compute op runs on another backend and no deleted path can be reached. P3.2
// brings MUL_MAT back as OPL records for the `graph` kernel.
//
// Environment: TSR_DEVICE (transport kind; "none" registers no device), TSR_DEBUG=1|2 (placement
// log on stderr). Test knobs: TSR_FORCE_HOST=1 keeps every non-WEIGHTS buffer in host memory,
// TSR_STAGE_KIB=<n> sets the IO staging area size. TSR_LOAD_THREADS=<n> sets the workers that
// convert and repack weights while they are uploaded (default: online CPUs, at most 16).
// TSR_STATS=1 prints the upload statistics when the backend is freed. The sim transport handles
// TSR_SIM_* itself.
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
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include "tsr_isa.h"
#include "tsr_ref.h"
#include "tsr_transport.h"

#include "tsr-alloc.h"
#include "tsr-lower.h"
#include "tsr-weights.h"
#include "tsr-wlayout.h"

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "ggml-tesseract writes tensors in host byte order; only little-endian hosts are supported"
#endif

#define TSR_REG_NAME  "Tesseract"
#define TSR_DEV_NAME  "Tesseract0"

static constexpr uint64_t TSR_TENSOR_ALIGN   = 64;                       // beat / SPM line
static constexpr uint64_t TSR_VADDR_BASE     = 0x4000000000000000ull;    // non-canonical: never dereferenced
static constexpr uint64_t TSR_VADDR_GUARD    = 1ull << 30;
static constexpr uint64_t TSR_HEAP_ALIGN     = 4096;
static constexpr size_t   TSR_XFER_CHUNK     = 8u << 20;                 // host-side staging granularity
// IO staging area (top of the ch0 IO region, reserved at open): activations of HOSTED buffers
// and line-padded output rows pass through it (P3.2 stages chunks "as v1 does", SW §9.3). At
// least 2 x SPM so that one token of any lowerable MUL_MAT (whose x + QA + y fit the SPM) always
// fits. Reserving it keeps the IO window capacity, and with it get_max_size, as it was.
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
    uint64_t ch_bytes = 0;        // bytes reserved per channel = R * ROW_BEATS * 64
    bool     complete = false;    // all bytes received and stored
    std::vector<uint8_t> staging; // split upload in progress: original ggml bytes
    tsr_intervals have;           // received byte ranges of `staging`
    bool     warned_lossy = false;

    // RAW
    tsr_raw_region raw_region = tsr_raw_region::NONE;
    uint64_t raw_off = 0;         // ch0 byte offset
};

enum class tsr_bmode : uint8_t { UNDECIDED, LINEAR, PLACED, HOSTED };

struct tsr_device;

struct tsr_buffer_ctx {
    tsr_device * dev = nullptr;
    uint64_t id   = 0;            // unique per process, never reused
    uint64_t base = 0;            // virtual base (64-aligned, non-NULL)
    uint64_t size = 0;
    tsr_bmode mode = tsr_bmode::UNDECIDED;
    bool     has_io = false;
    uint64_t io_off = 0;          // LINEAR: ch0 byte offset of the window (base <-> io_off)
    uint8_t * host  = nullptr;    // HOSTED: the buffer's bytes in host memory (base <-> host)
    uint8_t  clear_value = 0;     // PLACED: value read back for never-written tensors
    std::map<uint64_t, tsr_tensor_rec> recs;   // PLACED (and UNDECIDED non-COMPUTE): by vaddr
};

struct tsr_device {
    std::mutex   mtx;
    bool         tried_open = false;
    tsr_transport * tp = nullptr;
    tsr_config   cfg = {};
    std::string  description = "Tesseract overlay core (not opened)";

    tsr_range_alloc io;           // ch0 IO region below the stage: LINEAR buffer windows
    tsr_range_alloc aux;          // ch0 AUX region: RAW tensors of weight buffers
    tsr_range_alloc heap;         // WEIGHT heap, symmetric: an offset is reserved on every channel

    uint64_t stage_off  = 0;      // ch0 byte offset of the IO staging area (top of IO)
    uint64_t stage_size = 0;
    bool     force_host = false;  // TSR_FORCE_HOST=1: every non-WEIGHTS buffer is HOSTED

    uint64_t next_vaddr  = TSR_VADDR_BASE;
    uint64_t next_buf_id = 1;

    // TSR_STATS=1
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
    else if (c->beat_bits != 512)                         bad = "BEAT_BITS != 512 (v1 software assumes 64-byte beats)";
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
    if (bad != nullptr) {
        GGML_LOG_ERROR("%s: unusable Tesseract configuration: %s\n", __func__, bad);
        tsr_transport_close(tp);
        return false;
    }
    d->tp  = tp;
    d->cfg = *c;
    d->stage_size = stage;
    d->stage_off  = TSR_REGION_IO_OFF + TSR_REGION_IO_SIZE - stage;
    d->io.init  (TSR_REGION_IO_OFF,  TSR_REGION_IO_SIZE - stage, TSR_TENSOR_ALIGN);
    d->aux.init (TSR_REGION_AUX_OFF, TSR_REGION_AUX_SIZE, TSR_TENSOR_ALIGN);
    d->heap.init(TSR_REGION_WEIGHT_OFF, c->ch_bytes - TSR_REGION_WEIGHT_OFF, TSR_HEAP_ALIGN);
    d->force_host = getenv("TSR_FORCE_HOST") != nullptr && atoi(getenv("TSR_FORCE_HOST")) != 0;

    char desc[256];
    snprintf(desc, sizeof(desc), "Tesseract overlay core (%s, C=%u, LANES=%u, T_MAX=%u, K_MAX=%u, SPM=%u KiB, %.0f GiB/ch)",
             tsr_transport_name(tp), c->num_ch, c->lanes, c->t_max, c->k_max, c->spm_bytes / 1024,
             (double) c->ch_bytes / (1024.0 * 1024.0 * 1024.0));
    d->description = desc;
    TSR_DBG(1, "opened %s; IO windows %.2f MiB, IO stage %.2f MiB at ch0+0x%09" PRIx64 "%s\n", desc,
            tsr_mib(d->io.total()), tsr_mib(stage), d->stage_off, d->force_host ? ", TSR_FORCE_HOST" : "");
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

// ---------------------------------------------------------------------------
// PLACED (weights) buffers
// ---------------------------------------------------------------------------

static void tsr_rec_release(tsr_device * d, tsr_tensor_rec & r) {
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

// Is this tensor stored as a device weight (vs raw)? Mirrors what MUL_MAT can use directly.
static bool tsr_rec_is_devweight_shape(const tsr_device * d, const tsr_tensor_rec & r) {
    if (r.ne[2] != 1 || r.ne[3] != 1) {
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

static void tsr_place(tsr_device * d, tsr_tensor_rec & r) {
    GGML_ASSERT(r.kind == tsr_kind::UNPLACED);
    if (tsr_rec_is_devweight_shape(d, r)) {
        r.fmt = tsr_dev_fmt_for_type(r.type);
        r.N   = (uint32_t) r.ne[1];
        r.K   = (uint32_t) r.ne[0];
        r.ch_bytes = tsr_tensor_ch_beats(r.fmt, r.N, r.K, d->cfg.num_ch) * TSR_BEAT_BYTES;
        if (!d->heap.alloc(r.ch_bytes, &r.w_off)) {
            GGML_ABORT("tesseract: WEIGHT heap full: weight '%s' needs %.1f MiB per channel, %.1f MiB free (largest %.1f MiB)",
                       r.name.c_str(), tsr_mib(r.ch_bytes), tsr_mib(d->heap.free_bytes()), tsr_mib(d->heap.largest_free()));
        }
        r.kind = tsr_kind::DEVWEIGHT;
        r.complete = false;
        TSR_DBG(1, "weight %-28s %-6s [%6u x %6u] -> %s W_OFF=0x%09" PRIx64 " (%.2f MiB/ch x %u ch)%s\n",
                r.name.c_str(), ggml_type_name(r.type), r.N, r.K, r.fmt == TSR_FMT_Q8_0 ? "Q8_0" : "Q4_K",
                r.w_off, tsr_mib(r.ch_bytes), d->cfg.num_ch, tsr_type_is_native(r.type) ? "" : " (converted)");
        return;
    }
    const uint64_t size = tsr_pad(std::max<size_t>(r.nbytes, 1), TSR_TENSOR_ALIGN);
    if (d->aux.alloc(size, &r.raw_off)) {
        r.raw_region = tsr_raw_region::AUX;
    } else if (d->heap.alloc(size, &r.raw_off)) {
        r.raw_region = tsr_raw_region::HEAP;
    } else {
        GGML_ABORT("tesseract: no device memory for tensor '%s' (%.1f MiB)", r.name.c_str(), tsr_mib(size));
    }
    r.kind = tsr_kind::RAW;
    TSR_DBG(1, "raw    %-28s %-6s [%lld,%lld,%lld,%lld] -> %s ch0+0x%09" PRIx64 " (%zu B)\n",
            r.name.c_str(), ggml_type_name(r.type), (long long) r.ne[0], (long long) r.ne[1], (long long) r.ne[2],
            (long long) r.ne[3], r.raw_region == tsr_raw_region::AUX ? "AUX" : "HEAP", r.raw_off, r.nbytes);
}

static void tsr_rec_store_all(tsr_device * d, tsr_tensor_rec & r, const uint8_t * src) {
    tsr_store_weight_rows(d, r.type, r.fmt, r.N, r.K, r.w_off, src, 0, r.N);
    r.complete = true;
    r.staging.clear();
    r.staging.shrink_to_fit();
    r.have = tsr_intervals();
}

// Write bytes [rel, rel+n) of a tensor in a PLACED buffer.
static void tsr_rec_write(tsr_device * d, tsr_tensor_rec & r, uint64_t rel, const uint8_t * data, size_t n) {
    if (n == 0) {
        return;
    }
    if (r.kind == tsr_kind::UNPLACED) {
        tsr_place(d, r);
    }
    if (r.kind == tsr_kind::RAW) {
        tsr_write(d, 0, r.raw_off + rel, data, n);
        return;
    }

    // DEVWEIGHT
    if (!r.complete) {
        if (rel == 0 && n == r.nbytes && r.staging.empty()) {
            tsr_rec_store_all(d, r, data);   // the common case: one set_tensor for the whole tensor
            return;
        }
        // split upload (spec 주의할 점: 분할 업로드): stage until every byte has arrived
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
        tsr_store_weight_rows(d, r.type, r.fmt, r.N, r.K, r.w_off, data, r0, r1);
        return;
    }
    if (!tsr_type_is_native(r.type)) {
        GGML_ABORT("tesseract: partial-row update of weight '%s' (%s, converted to Q8_0 at load) is not supported: "
                   "the original bytes are not kept", r.name.c_str(), ggml_type_name(r.type));
    }
    std::vector<uint8_t> rows((size_t) (r1 - r0) * src_rb);
    tsr_load_weight_rows(d, r, r0, r1, rows.data());
    memcpy(rows.data() + (rel - (uint64_t) r0 * src_rb), data, n);
    tsr_store_weight_rows(d, r.type, r.fmt, r.N, r.K, r.w_off, rows.data(), r0, r1);
}

static void tsr_rec_read(tsr_device * d, const tsr_buffer_ctx * b, tsr_tensor_rec & r, uint64_t rel, uint8_t * dst, size_t n) {
    if (n == 0) {
        return;
    }
    switch (r.kind) {
        case tsr_kind::UNPLACED:
            memset(dst, b->clear_value, n);
            return;
        case tsr_kind::RAW:
            tsr_read(d, 0, r.raw_off + rel, dst, n);
            return;
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
    tsr_load_weight_rows(d, r, r0, r1, rows.data());
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
        } else {
            tsr_fill(d, 0, r->raw_off + rel, value, n);
        }
        addr += n;
        size -= n;
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

// Fix the buffer's storage mode at its first data access (llama.cpp sets WEIGHTS usage right
// after allocating a weight buffer and before uploading; compute buffers get COMPUTE usage
// right after allocation; init_tensor runs while usage is still ANY).
//   WEIGHTS                       -> PLACED (per tensor)
//   ANY holding only leaf tensors -> HOSTED: data that no device op reads or writes in place
//                                    (LoRA adapters, control vectors, a KV cache without -nkvo)
//   otherwise (ANY, COMPUTE)      -> LINEAR if a window of the IO region is free, else HOSTED
//                                    (device ops stage its rows through IO)
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
    bool leaves_only = gbuf->usage == GGML_BACKEND_BUFFER_USAGE_ANY && !b->recs.empty();
    for (const auto & kv : b->recs) {
        leaves_only = leaves_only && kv.second.leaf;
    }
    uint64_t off = 0;
    if (!d->force_host && !leaves_only && d->io.alloc(b->size, &off)) {
        b->mode   = tsr_bmode::LINEAR;
        b->has_io = true;
        b->io_off = off;
        b->recs.clear();
        TSR_DBG(1, "buffer #%" PRIu64 " (%.2f MiB, %s): IO window ch0+0x%09" PRIx64 "\n",
                b->id, tsr_mib(b->size), tsr_usage_name(gbuf->usage), off);
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
        // not an error: device ops on it are staged through IO
        GGML_LOG_INFO("tesseract: %s buffer #%" PRIu64 " of %.2f MiB does not fit the IO region (largest free window "
                      "%.2f MiB of %.2f MiB): kept in host memory, device activations in it pass through the "
                      "%.0f MiB IO stage\n", tsr_usage_name(gbuf->usage), b->id, tsr_mib(b->size),
                      tsr_mib(d->io.largest_free()), tsr_mib(d->io.total()), tsr_mib(d->stage_size));
    }
}

// Storage of a buffer for the test hook (get_proc_address "ggml_backend_tesseract_buffer_storage").
static const char * tsr_bmode_name(tsr_bmode m) {
    switch (m) {
        case tsr_bmode::UNDECIDED: return "undecided";
        case tsr_bmode::LINEAR:    return "io";
        case tsr_bmode::PLACED:    return "placed";
        case tsr_bmode::HOSTED:    return "host";
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
// ggml buffer interface
// ---------------------------------------------------------------------------

static void tsr_buffer_free(ggml_backend_buffer_t buffer) {
    tsr_buffer_ctx * b = tsr_buf(buffer);
    tsr_device * d = b->dev;
    {
        std::lock_guard<std::mutex> lock(d->mtx);
        if (b->has_io) {
            d->io.release(b->io_off);
        }
        for (auto & kv : b->recs) {
            tsr_rec_release(d, kv.second);
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
        return GGML_STATUS_SUCCESS;     // views resolve by address to their base tensor
    }
    if (buffer->usage == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        return GGML_STATUS_SUCCESS;     // LINEAR storage needs no per-tensor state
    }
    std::lock_guard<std::mutex> lock(b->dev->mtx);
    if (b->mode == tsr_bmode::LINEAR || b->mode == tsr_bmode::HOSTED) {
        return GGML_STATUS_SUCCESS;     // raw storage needs no per-tensor state
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
    const uint64_t t0 = tsr_stats_on() ? tsr_now_ns() : 0;
    if (b->mode == tsr_bmode::LINEAR) {
        tsr_write(d, 0, b->io_off + (addr - b->base), data, size);
    } else if (b->mode == tsr_bmode::HOSTED) {
        memcpy(b->host + (addr - b->base), data, size);
    } else {
        tsr_placed_write(d, b, addr, (const uint8_t *) data, size);
    }
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
        tsr_read(d, 0, b->io_off + (addr - b->base), data, size);
    } else if (b->mode == tsr_bmode::HOSTED) {
        memcpy(data, b->host + (addr - b->base), size);
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
    if (b->mode == tsr_bmode::LINEAR) {
        tsr_fill(d, 0, b->io_off + (addr - b->base), value, size);
    } else if (b->mode == tsr_bmode::HOSTED) {
        memset(b->host + (addr - b->base), value, size);
    } else {
        tsr_placed_fill(d, b, addr, value, size);
    }
}

static void tsr_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    tsr_buffer_ctx * b = tsr_buf(buffer);
    tsr_device * d = b->dev;
    std::lock_guard<std::mutex> lock(d->mtx);
    tsr_buf_decide(b, buffer);
    if (b->mode == tsr_bmode::LINEAR) {
        tsr_fill(d, 0, b->io_off, value, b->size);
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

// One buffer type serves weights and activations. The limit is the IO window capacity (IO
// region minus the stage), so that ggml-alloc cuts compute buffers into chunks of which the
// first can live in IO; weight buffers above it become ggml multi-buffers. A single tensor
// larger than this still gets its own buffer (HOSTED unless it fits a free window).
static size_t tsr_buft_get_max_size(ggml_backend_buffer_type_t buft) {
    tsr_device * d = (tsr_device *) buft->context;
    std::lock_guard<std::mutex> lock(d->mtx);
    if (!tsr_dev_open_locked(d)) {
        return (size_t) TSR_REGION_IO_SIZE;
    }
    return (size_t) d->io.total();
}

// Pad to whole lines: a line-granular ST of the last row lands in this padding (device layout
// of compute tensors, kept for the op-list path).
static size_t tsr_buft_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    GGML_UNUSED(buft);
    return (size_t) tsr_pad(ggml_nbytes(tensor), TSR_TENSOR_ALIGN);
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
    if (tsr_stats_on()) {
        std::lock_guard<std::mutex> lock(d->mtx);
        fprintf(stderr, "[tesseract] stats: set_tensor %.3f s for %.3f GB, of which transport writes %.3f s for %.3f GB (%.2f GB/s)\n",
                d->st_set_ns * 1e-9, d->st_set_bytes * 1e-9, d->st_write_ns * 1e-9, d->st_write_bytes * 1e-9,
                d->st_write_ns ? d->st_write_bytes / (double) d->st_write_ns : 0.0);
        fprintf(stderr, "[tesseract] stats: weight stores %.3f s (%u load threads), conversion to Q8_0 %.3f s of CPU time\n",
                d->st_store_ns * 1e-9, tsr_load_threads(), d->st_conv_ns.load() * 1e-9);
    }
    delete (tsr_backend_ctx *) backend->context;
    delete backend;
}

// Only no-op nodes are ever assigned to this backend (supports_op), so there is nothing to run.
static enum ggml_status tsr_backend_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    GGML_UNUSED(backend);
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }
        switch (node->op) {
            case GGML_OP_NONE:
            case GGML_OP_VIEW:
            case GGML_OP_RESHAPE:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                continue;
            default:
                GGML_ABORT("tesseract: op %s ('%s') reached graph_compute, but this build executes no op on the device "
                           "(the v1 command-buffer path is deleted; MUL_MAT returns with the op-list path)",
                           ggml_op_desc(node), node->name);
        }
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
    static std::once_flag warned;
    std::call_once(warned, [] {
        GGML_LOG_WARN("tesseract: this build runs no op on the device (v1 MUL_MAT execution deleted, the op-list path "
                      "is not implemented yet): every compute op runs on another backend; weights still upload\n");
    });
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

// No compute op is supported until the op-list path (P3.2): MUL_MAT's v1 execution is deleted (X0).
// The no-op ops (leaves and views) stay "supported" as on every GPU backend: they cost nothing in
// graph_compute, test-backend-ops asks for every tensor of a test graph, and the scheduler aborts
// on a leaf or view of a Tesseract-resident tensor that no backend claims. Under llama.cpp this
// keeps the weights on the CPU (its placement probe asks for the weight's MUL_MAT).
static bool tsr_dev_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);
    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;
        default:
            return false;
    }
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

static void * tsr_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    if (name != nullptr && strcmp(name, "ggml_backend_tesseract_buffer_storage") == 0) {
        return (void *) tsr_buffer_storage;
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
