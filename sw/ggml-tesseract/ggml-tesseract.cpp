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
//               the programs in place; or
//           "HOSTED": host memory, when the buffer does not fit the free IO windows (e.g. the
//               worst-case logits of the output head that llama.cpp reserves for n_ubatch
//               outputs) or holds only leaf tensors (data such as LoRA adapters, which no
//               program addresses directly). graph_compute stages the src1/dst rows of a
//               MUL_MAT through the IO staging area (top of IO) in T tiles.
//           A MUL_MAT src0 found in such a buffer is converted lazily at graph_compute into the
//           WEIGHT heap and cached by (buffer, offset, type, shape, write generation).
//   graph_compute   -> every MUL_MAT is lowered (tsr-lower.h) into LD/QACT/MVM/ST, the split is
//                      executed as one command buffer (+END) from a CMD-region half (staged
//                      MUL_MATs run on their own, one program per T tile of the stage).
//
// Environment: TSR_DEVICE (transport kind; "none" registers no device), TSR_DEBUG=1|2 (lowering
// and placement log on stderr; 2 adds a disassembly of every executed program). Test knobs:
// TSR_FORCE_HOST=1 keeps every non-WEIGHTS buffer in host memory (exercises the staged path),
// TSR_STAGE_KIB=<n> sets the IO staging area size. The sim transport handles TSR_TRACE /
// TSR_SIM_* itself.
#include "ggml-tesseract.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <map>
#include <mutex>
#include <set>
#include <string>
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
#error "ggml-tesseract writes instructions and tensors in host byte order; only little-endian hosts are supported"
#endif
static_assert(sizeof(tsr_insn) == TSR_INSN_BYTES, "tsr_insn must be 32 bytes");

#define TSR_REG_NAME  "Tesseract"
#define TSR_DEV_NAME  "Tesseract0"

static constexpr uint64_t TSR_TENSOR_ALIGN   = 64;                       // beat / SPM line
static constexpr uint64_t TSR_VADDR_BASE     = 0x4000000000000000ull;    // non-canonical: never dereferenced
static constexpr uint64_t TSR_VADDR_GUARD    = 1ull << 30;
static constexpr uint64_t TSR_HEAP_ALIGN     = 4096;
static constexpr size_t   TSR_XFER_CHUNK     = 8u << 20;                 // host-side staging granularity
// IO staging area (top of the ch0 IO region, reserved at open): activations of HOSTED buffers
// and line-padded output rows pass through it. At least 2 x SPM so that one token of any
// lowerable MUL_MAT (whose x + QA + y fit the SPM) always fits.
static constexpr uint64_t TSR_STAGE_DEFAULT  = 16ull << 20;
// Program cache bounds (backend review: bounded by bytes, not only by entries).
static constexpr size_t   TSR_PROG_CACHE_BYTES       = 64u << 20;
static constexpr size_t   TSR_PROG_CACHE_ENTRY_BYTES = 4u << 20;
static constexpr size_t   TSR_PROG_CACHE_ENTRIES     = 4096;

static int tsr_debug_level() {
    static const int lvl = [] {
        const char * e = getenv("TSR_DEBUG");
        return e ? atoi(e) : 0;
    }();
    return lvl;
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
    uint64_t gen = 1;             // bumped on every host or device write
    uint8_t  clear_value = 0;     // PLACED: value read back for never-written tensors
    std::map<uint64_t, tsr_tensor_rec> recs;   // PLACED (and UNDECIDED non-COMPUTE): by vaddr
};

struct tsr_lazy_key {
    uint64_t buf_id;
    uint64_t src_off;             // ch0 byte offset of the raw source bytes
    int      type;
    uint32_t N;
    uint32_t K;

    bool operator<(const tsr_lazy_key & o) const {
        if (buf_id != o.buf_id)   return buf_id < o.buf_id;
        if (src_off != o.src_off) return src_off < o.src_off;
        if (type != o.type)       return type < o.type;
        if (N != o.N)             return N < o.N;
        return K < o.K;
    }
};

struct tsr_lazy_entry {
    uint64_t gen = 0;             // source buffer generation the conversion was made from
    uint32_t fmt = 0;
    uint64_t w_off = 0;
    uint64_t ch_bytes = 0;
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

    std::map<tsr_lazy_key, tsr_lazy_entry> lazy;

    tsr_prog_cache progs { TSR_PROG_CACHE_BYTES, TSR_PROG_CACHE_ENTRY_BYTES, TSR_PROG_CACHE_ENTRIES };
    uint64_t cmd_seq = 0;
    uint64_t half_seq[2] = { 0, 0 };
    int      last_half = 1;
    uint64_t n_exec = 0;

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
    if (tsr_transport_write(d->tp, ch, off, src, size) != 0) {
        GGML_ABORT("tesseract: transport write failed (ch %u, offset 0x%" PRIx64 ", %zu bytes)", ch, off, size);
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
    std::vector<float> f32(K);
    // repack + symmetric scatter: tsr-wlayout (shared with the ggml-free runtime sw/rt)
    tsr_weight_store_rows(d->cfg.num_ch, fmt, N, K, w_off, r0, r1, TSR_XFER_CHUNK,
        [&](uint32_t rr, void * scratch) {
            return tsr_row_to_dev_blocks(type, fmt, src + (size_t) (rr - r0) * src_rb, K, scratch, f32.data());
        },
        [d](uint32_t ch, uint64_t off, const void * p, size_t n) { tsr_write(d, ch, off, p, n); });
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

// Free every lazily converted weight (their heap space). Callers must have flushed any
// pending program that references them.
static void tsr_lazy_evict_all(tsr_device * d) {
    for (auto & kv : d->lazy) {
        d->heap.release(kv.second.w_off);
    }
    if (!d->lazy.empty()) {
        TSR_DBG(1, "evicted %zu lazily converted weights\n", d->lazy.size());
    }
    d->lazy.clear();
}

static void tsr_lazy_evict_buffer(tsr_device * d, uint64_t buf_id) {
    for (auto it = d->lazy.begin(); it != d->lazy.end();) {
        if (it->first.buf_id == buf_id) {
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
    // pending program, see tsr_place / tsr_lazy_weight)
    tsr_lazy_evict_all(d);
    return d->heap.alloc(size, off);
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
        if (!tsr_heap_alloc(d, r.ch_bytes, &r.w_off)) {
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
    } else if (tsr_heap_alloc(d, size, &r.raw_off)) {
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
//   ANY holding only leaf tensors -> HOSTED: data that no program reads or writes in place
//                                    (LoRA adapters, control vectors, a KV cache without -nkvo);
//                                    a MUL_MAT src0 among them is converted into the heap
//   otherwise (ANY, COMPUTE)      -> LINEAR if a window of the IO region is free, else HOSTED
//                                    (graph_compute stages its MUL_MAT rows through IO)
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
        // not an error: the MUL_MATs that use it are staged through IO in T tiles
        GGML_LOG_INFO("tesseract: %s buffer #%" PRIu64 " of %.2f MiB does not fit the IO region (largest free window "
                      "%.2f MiB of %.2f MiB): kept in host memory, MUL_MAT activations in it pass through the "
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
        tsr_lazy_evict_buffer(d, b->id);
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
    b->gen++;
    if (b->mode == tsr_bmode::LINEAR) {
        tsr_write(d, 0, b->io_off + (addr - b->base), data, size);
    } else if (b->mode == tsr_bmode::HOSTED) {
        memcpy(b->host + (addr - b->base), data, size);
    } else {
        tsr_placed_write(d, b, addr, (const uint8_t *) data, size);
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
    b->gen++;
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
    b->gen++;
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

// Pad to whole lines: a line-granular ST of the last row lands in this padding.
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
// Lowering and execution
// ---------------------------------------------------------------------------

// A segment: MUL_MATs whose operands the programs address in place (src1 and dst in LINEAR IO
// windows), lowered into one command buffer and run together.
struct tsr_segment {
    std::vector<tsr_mm_desc>      mms;
    std::vector<std::string>      names;
    std::set<tsr_buffer_ctx *>    written;    // buffers the pending program writes
};

static std::string tsr_prog_key(const std::vector<tsr_mm_desc> & mms) {
    std::string k;
    k.reserve(mms.size() * 56);
    auto put = [&k](uint64_t v) { k.append((const char *) &v, sizeof(v)); };
    for (const tsr_mm_desc & m : mms) {
        put(m.fmt);
        put(m.N);
        put(m.K);
        put(m.T);
        put(m.w_off_beat);
        put(m.x_beat);
        put(m.y_beat);
    }
    return k;
}


static void tsr_exec(tsr_device * d, const std::vector<tsr_insn> & insns, size_t first, size_t count, uint64_t cmd_beat) {
    tsr_exec_result res = {};
    const int rc = tsr_transport_exec(d->tp, cmd_beat, (uint32_t) count, nullptr, &res);
    d->n_exec++;
    if (rc != 0 || (res.status & TSR_STATUS_ERR) || !(res.status & TSR_STATUS_DONE)) {
        std::string at = "-";
        if (res.err_pc < count) {
            at = res.err_pc + 1 == count ? std::string("END") : tsr_insn_str(insns[first + res.err_pc]);
        }
        GGML_ABORT("tesseract: device execution failed: rc=%d STATUS=0x%x ERR_CODE=%u ERR_PC=%u ERR_INFO=0x%x "
                   "(program of %zu instructions at CMD beat 0x%" PRIx64 "; failing instruction: %s)",
                   rc, res.status, res.err_code, res.err_pc, res.err_info, count, cmd_beat, at.c_str());
    }
}

// Write [first, first+n) + END into CMD half h and run it.
static void tsr_run_range(tsr_device * d, const std::vector<tsr_insn> & insns, size_t first, size_t n, int h) {
    const uint64_t half_bytes = TSR_REGION_CMD_SIZE / 2;
    const uint64_t off = TSR_REGION_CMD_OFF + (uint64_t) h * half_bytes;
    tsr_write(d, 0, off, insns.data() + first, n * sizeof(tsr_insn));
    const tsr_insn end = tsr_enc_end();
    tsr_write(d, 0, off + n * sizeof(tsr_insn), &end, sizeof(end));
    d->half_seq[h] = ++d->cmd_seq;
    d->last_half = h;
    if (tsr_debug_level() >= 2) {
        for (size_t i = 0; i < n; ++i) {
            fprintf(stderr, "[tesseract]   %5zu  %s\n", i, tsr_insn_str(insns[first + i]).c_str());
        }
        fprintf(stderr, "[tesseract]   %5zu  END\n", n);
    }
    tsr_exec(d, insns, first, n + 1, off / TSR_BEAT_BYTES);
}

static void tsr_run_prog(tsr_device * d, tsr_prog & p) {
    const uint64_t half_bytes = TSR_REGION_CMD_SIZE / 2;
    const size_t   cap = (size_t) (half_bytes / sizeof(tsr_insn)) - 1;   // room for END
    p.uses++;
    if (p.insns.size() <= cap) {
        if (p.half >= 0 && d->half_seq[p.half] == p.seq) {
            // still resident in its CMD half: run it in place
            TSR_DBG(2, "exec resident program (%zu insns) in CMD half %d\n", p.insns.size(), p.half);
            tsr_exec(d, p.insns, 0, p.insns.size() + 1,
                     (TSR_REGION_CMD_OFF + (uint64_t) p.half * half_bytes) / TSR_BEAT_BYTES);
            return;
        }
        const int h = 1 - d->last_half;   // double buffering: the other half
        tsr_run_range(d, p.insns, 0, p.insns.size(), h);
        p.half = h;
        p.seq  = d->half_seq[h];
        return;
    }
    // larger than a CMD half: run in pieces split at tile boundaries
    p.half = -1;
    size_t start = 0;
    size_t ci = 0;
    while (start < p.insns.size()) {
        size_t end = start;
        while (ci < p.cuts.size() && p.cuts[ci] - start <= cap) {
            end = p.cuts[ci++];
        }
        GGML_ASSERT(end > start && "program piece does not fit a CMD half");
        tsr_run_range(d, p.insns, start, end - start, 1 - d->last_half);
        start = end;
    }
}

static void tsr_flush(tsr_device * d, tsr_segment & seg) {
    if (seg.mms.empty()) {
        return;
    }
    const std::string key = tsr_prog_key(seg.mms);
    tsr_prog * prog = d->progs.find(key);
    const bool hit = prog != nullptr;
    tsr_prog uncached;
    if (!hit) {
        tsr_prog p;
        for (const tsr_mm_desc & m : seg.mms) {
            tsr_mm_emit(d->cfg, m, p.insns, p.cuts);
        }
        if (d->progs.cacheable(key, p)) {
            prog = d->progs.insert(key, std::move(p));
        } else {
            uncached = std::move(p);          // too large to keep (long prefill): run once
            prog = &uncached;
        }
    }
    TSR_DBG(1, "exec #%" PRIu64 ": %zu MUL_MAT, %zu insns%s\n", d->n_exec, seg.mms.size(), prog->insns.size(),
            hit ? " (cached program)" : (prog == &uncached ? " (not cached: too large)" : ""));
    tsr_run_prog(d, *prog);

    for (tsr_buffer_ctx * b : seg.written) {
        b->gen++;
    }
    seg.mms.clear();
    seg.names.clear();
    seg.written.clear();
}

struct tsr_wref {
    uint32_t fmt;
    uint64_t w_off;
};

// Convert a MUL_MAT src0 stored as raw ggml bytes into the WEIGHT heap (cached). The bytes are
// at ch0 byte offset `src_off` (LINEAR window, RAW tensor of a weight buffer), or at `host` for
// a HOSTED buffer (src_off is then the offset inside the buffer and only keys the cache).
static tsr_wref tsr_lazy_weight(tsr_device * d, tsr_segment & seg, tsr_buffer_ctx * b, uint64_t src_off,
                                const uint8_t * host, enum ggml_type type, uint32_t N, uint32_t K, const char * name) {
    if (seg.written.count(b)) {
        tsr_flush(d, seg);      // src0 may be produced by the pending program
    }
    const tsr_lazy_key key = { b->id, src_off, (int) type, N, K };
    const uint32_t fmt = tsr_dev_fmt_for_type(type);
    auto it = d->lazy.find(key);
    if (it != d->lazy.end() && it->second.gen == b->gen) {
        return { it->second.fmt, it->second.w_off };
    }
    if (it == d->lazy.end()) {
        tsr_lazy_entry e;
        e.fmt = fmt;
        e.ch_bytes = tsr_tensor_ch_beats(fmt, N, K, d->cfg.num_ch) * TSR_BEAT_BYTES;
        if (!d->heap.alloc(e.ch_bytes, &e.w_off)) {
            // evicting cached conversions is only safe with nothing pending
            tsr_flush(d, seg);
            tsr_lazy_evict_all(d);
            if (!d->heap.alloc(e.ch_bytes, &e.w_off)) {
                GGML_ABORT("tesseract: WEIGHT heap full converting '%s' (%.1f MiB per channel needed)", name, tsr_mib(e.ch_bytes));
            }
        }
        it = d->lazy.emplace(key, e).first;
    } else if (!seg.mms.empty()) {
        // re-converting in place: the pending program may still read the old contents
        for (const tsr_mm_desc & m : seg.mms) {
            if (m.w_off_beat * TSR_BEAT_BYTES == it->second.w_off) {
                tsr_flush(d, seg);
                break;
            }
        }
    }
    const size_t src_rb = ggml_row_size(type, K);
    if (host != nullptr) {
        tsr_store_weight_rows(d, type, fmt, N, K, it->second.w_off, host, 0, N);
    } else {
        const uint32_t batch = (uint32_t) std::max<size_t>(1, TSR_XFER_CHUNK / src_rb);
        std::vector<uint8_t> rows;
        for (uint32_t r = 0; r < N; r += batch) {
            const uint32_t n = std::min(batch, N - r);
            rows.resize((size_t) n * src_rb);
            tsr_read(d, 0, src_off + (uint64_t) r * src_rb, rows.data(), rows.size());
            tsr_store_weight_rows(d, type, fmt, N, K, it->second.w_off, rows.data(), r, r + n);
        }
    }
    it->second.gen = b->gen;
    TSR_DBG(1, "lazy   %-28s %-6s [%6u x %6u] -> %s W_OFF=0x%09" PRIx64 " (from %s+0x%09" PRIx64 ", gen %" PRIu64 ")\n",
            name, ggml_type_name(type), N, K, fmt == TSR_FMT_Q8_0 ? "Q8_0" : "Q4_K", it->second.w_off,
            host != nullptr ? "host" : "ch0", src_off, b->gen);
    return { it->second.fmt, it->second.w_off };
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

// Where the bytes of a MUL_MAT activation (src1 or dst) live: in a LINEAR window of the IO
// region (the program reads / writes them in place) or in a HOSTED buffer (staged).
struct tsr_act {
    tsr_buffer_ctx * b;
    uint64_t rel;                 // byte offset inside the buffer
    bool     io;                  // LINEAR
    uint64_t  io_off() const { return b->io_off + rel; }
    uint8_t * host()   const { return b->host + rel; }
};

static tsr_act tsr_act_loc(tsr_device * d, const ggml_tensor * t, const char * what) {
    tsr_buffer_ctx * b = tsr_tensor_buf(d, t, what);
    if (b->mode == tsr_bmode::PLACED) {
        GGML_ABORT("tesseract: %s '%s' is in a weight buffer; MUL_MAT activations must be in compute buffers", what, t->name);
    }
    const uint64_t a = tsr_vaddr(t->data);
    tsr_check_range(b, a, ggml_nbytes(t));
    return { b, a - b->base, b->mode == tsr_bmode::LINEAR };
}

static tsr_wref tsr_resolve_weight(tsr_device * d, tsr_segment & seg, const ggml_tensor * src0) {
    tsr_buffer_ctx * b = tsr_tensor_buf(d, src0, "src0");
    const uint32_t K = (uint32_t) src0->ne[0];
    const uint32_t N = (uint32_t) src0->ne[1];
    const uint64_t a = tsr_vaddr(src0->data);
    if (b->mode == tsr_bmode::LINEAR) {
        tsr_check_range(b, a, ggml_nbytes(src0));
        return tsr_lazy_weight(d, seg, b, b->io_off + (a - b->base), nullptr, src0->type, N, K, src0->name);
    }
    if (b->mode == tsr_bmode::HOSTED) {
        tsr_check_range(b, a, ggml_nbytes(src0));
        return tsr_lazy_weight(d, seg, b, a - b->base, b->host + (a - b->base), src0->type, N, K, src0->name);
    }
    tsr_tensor_rec * r = tsr_find_rec(b, a);
    if (r == nullptr) {
        GGML_ABORT("tesseract: weight '%s' was never initialized in its buffer", src0->name);
    }
    if (r->kind == tsr_kind::UNPLACED) {
        GGML_ABORT("tesseract: weight '%s' was never uploaded", src0->name);
    }
    if (r->kind == tsr_kind::DEVWEIGHT) {
        if (r->vaddr != a || r->type != src0->type || r->K != K || r->N != N) {
            GGML_ABORT("tesseract: MUL_MAT uses a view of device weight '%s'; views of placed weights are not supported", r->name.c_str());
        }
        if (!r->complete) {
            GGML_ABORT("tesseract: weight '%s' is only partially uploaded", r->name.c_str());
        }
        return { r->fmt, r->w_off };
    }
    // RAW in a weight buffer (e.g. a 1D f32 tensor used as a 1 x K matrix)
    return tsr_lazy_weight(d, seg, b, r->raw_off + (a - r->vaddr), nullptr, src0->type, N, K, src0->name);
}

// A MUL_MAT whose operands the program cannot address in place: src1 or dst in host memory
// (HOSTED buffer), or dst rows that do not start on a line (N*4 % 64 != 0 with T > 1). It runs
// on its own (after everything pending), one program per T tile that fits the IO stage:
//   src1 rows host -> stage (unless src1 is in a LINEAR window: LD reads it in place),
//   LD / QACT / MVM / ST (dst rows to the stage, line-padded, or straight into a LINEAR dst),
//   stage -> dst rows (compacted to N*4 bytes each; host memory or the LINEAR window).
// So only one tile of the activations has to be resident in IO at a time, whatever T is.
static void tsr_lower_staged(tsr_device * d, tsr_segment & seg, const ggml_tensor * node, const tsr_act & ax,
                             const tsr_act & ay, bool stage_x, bool stage_y, uint32_t fmt, uint32_t N, uint32_t K,
                             uint32_t T, const tsr_mm_tiling & tl) {
    tsr_flush(d, seg);          // program order: what is pending may produce src1 / src0
    const ggml_tensor * src0 = node->src[0];
    const tsr_wref w = tsr_resolve_weight(d, seg, src0);
    GGML_ASSERT(w.fmt == fmt && w.w_off % TSR_BEAT_BYTES == 0 && seg.mms.empty());

    const uint64_t x_row     = (uint64_t) tl.x_lines * TSR_LINE_BYTES;   // K*4 (K % 32 == 0)
    const uint64_t row_bytes = (uint64_t) N * 4;
    const uint64_t y_stride  = (uint64_t) tl.y_lines * TSR_LINE_BYTES;   // line-padded row
    const uint64_t per_token = (stage_x ? x_row : 0) + (stage_y ? y_stride : 0);
    const uint64_t tc_max    = std::min<uint64_t>(T, d->stage_size / per_token);
    if (tc_max == 0) {
        GGML_ABORT("tesseract: MUL_MAT '%s' (N=%u K=%u): one token needs %.2f MiB of IO stage, the stage has %.2f MiB",
                   node->name, N, K, tsr_mib(per_token), tsr_mib(d->stage_size));
    }
    const uint64_t xs = d->stage_off;
    const uint64_t ys = d->stage_off + (stage_x ? tc_max * x_row : 0);
    std::vector<uint8_t> yb;
    std::vector<uint8_t> packed;
    uint32_t n_tiles = 0;
    for (uint64_t t0 = 0; t0 < T; t0 += tc_max, ++n_tiles) {
        const uint32_t tc = (uint32_t) std::min<uint64_t>(tc_max, T - t0);
        if (stage_x) {
            tsr_write(d, 0, xs, ax.host() + t0 * x_row, (size_t) (tc * x_row));
        }
        tsr_mm_desc m = {};
        m.fmt = fmt;
        m.N = N;
        m.K = K;
        m.T = tc;
        m.w_off_beat = w.w_off / TSR_BEAT_BYTES;
        m.x_beat = (stage_x ? xs : ax.io_off() + t0 * x_row) / TSR_BEAT_BYTES;
        m.y_beat = (stage_y ? ys : ay.io_off() + t0 * y_stride) / TSR_BEAT_BYTES;
        tsr_segment one;
        one.mms.push_back(m);
        one.names.push_back(node->name);
        if (!stage_y) {
            one.written.insert(ay.b);
        }
        tsr_flush(d, one);
        if (stage_y) {
            yb.resize((size_t) (tc * y_stride));
            tsr_read(d, 0, ys, yb.data(), yb.size());
            if (ay.io) {
                packed.resize((size_t) (tc * row_bytes));
                for (uint32_t t = 0; t < tc; ++t) {
                    memcpy(packed.data() + t * row_bytes, yb.data() + t * y_stride, (size_t) row_bytes);
                }
                tsr_write(d, 0, ay.io_off() + t0 * row_bytes, packed.data(), packed.size());
            } else {
                for (uint32_t t = 0; t < tc; ++t) {
                    memcpy(ay.host() + (t0 + t) * row_bytes, yb.data() + t * y_stride, (size_t) row_bytes);
                }
            }
        }
    }
    ay.b->gen++;

    TSR_DBG(1, "MUL_MAT %-24s %-6s N=%u K=%u T=%u (t_tile %u) W_OFF=0x%09" PRIx64 " staged: src1 %s, dst %s, "
               "%u stage tile(s) of <= %" PRIu64 " tokens\n",
            node->name, ggml_type_name(src0->type), N, K, T, tl.t_tile, w.w_off,
            stage_x ? "host->stage" : "in IO", stage_y ? (ay.io ? "stage->IO (compacted)" : "stage->host") : "in IO",
            n_tiles, tc_max);
}

static void tsr_lower_mul_mat(tsr_device * d, tsr_segment & seg, const ggml_tensor * node) {
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
        return;
    }
    GGML_ASSERT(node->ne[0] == (int64_t) N && ggml_nrows(node) == T64);
    if (T64 > (int64_t) UINT32_MAX) {
        GGML_ABORT("tesseract: MUL_MAT '%s' has too many tokens", node->name);
    }
    const uint32_t T = (uint32_t) T64;
    const uint32_t fmt = tsr_dev_fmt_for_type(src0->type);
    tsr_mm_tiling tl;
    if (!tsr_mm_plan_tiling(d->cfg, fmt, N, K, T, &tl)) {
        GGML_ABORT("tesseract: MUL_MAT '%s' (N=%u K=%u) cannot be lowered", node->name, N, K);
    }

    const tsr_act ax = tsr_act_loc(d, src1, "src1");
    const tsr_act ay = tsr_act_loc(d, node, "dst");
    const uint64_t row_bytes = (uint64_t) N * 4;
    const uint64_t y_stride  = (uint64_t) tl.y_lines * TSR_LINE_BYTES;
    const bool stage_x = !ax.io;
    const bool stage_y = !ay.io || (row_bytes % TSR_LINE_BYTES != 0 && T > 1);
    if ((ax.io && ax.io_off() % TSR_BEAT_BYTES != 0) || (ay.io && ay.io_off() % TSR_BEAT_BYTES != 0)) {
        GGML_ABORT("tesseract: MUL_MAT '%s': src1/dst not 64-byte aligned in IO", node->name);
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
        tsr_lower_staged(d, seg, node, ax, ay, stage_x, stage_y, fmt, N, K, T, tl);
        return;
    }

    const tsr_wref w = tsr_resolve_weight(d, seg, src0);
    GGML_ASSERT(w.fmt == fmt && w.w_off % TSR_BEAT_BYTES == 0);

    tsr_mm_desc m = {};
    m.fmt = fmt;
    m.N = N;
    m.K = K;
    m.T = T;
    m.w_off_beat = w.w_off / TSR_BEAT_BYTES;
    m.x_beat = ax.io_off() / TSR_BEAT_BYTES;
    m.y_beat = ay.io_off() / TSR_BEAT_BYTES;
    seg.mms.push_back(m);
    seg.names.push_back(node->name);
    seg.written.insert(ay.b);

    TSR_DBG(1, "MUL_MAT %-24s %-6s N=%u K=%u T=%u (t_tile %u, %u tiles) W_OFF=0x%09" PRIx64 " x=ch0+0x%09" PRIx64 " y=ch0+0x%09" PRIx64 "\n",
            node->name, ggml_type_name(src0->type), N, K, T, tl.t_tile, (T + tl.t_tile - 1) / tl.t_tile,
            w.w_off, ax.io_off(), ay.io_off());
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
    delete (tsr_backend_ctx *) backend->context;
    delete backend;
}

static enum ggml_status tsr_backend_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    tsr_device * d = ((tsr_backend_ctx *) backend->context)->dev;
    std::lock_guard<std::mutex> lock(d->mtx);
    tsr_segment seg;
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
            case GGML_OP_MUL_MAT:
                if (!ggml_is_empty(node)) {
                    tsr_lower_mul_mat(d, seg, node);
                }
                continue;
            default:
                GGML_ABORT("tesseract: op %s ('%s') reached graph_compute but is not supported", ggml_op_desc(node), node->name);
        }
    }
    tsr_flush(d, seg);
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

// spec "supports_op (v1)": MUL_MAT only, with a 2D weight src0 that the device can hold
// (Q8_0 / Q4_K natively, anything else with to_float converted to Q8_0) and F32 activations.
// The no-op ops (leaves and views) are "supported" as on every GPU backend: they cost nothing
// in graph_compute, test-backend-ops asks for every tensor of a test graph, and the scheduler
// aborts on a view of a Tesseract-resident tensor that no backend claims.
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
        default:
            return false;
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
        cfg = d->cfg;
    }
    const uint32_t fmt = tsr_dev_fmt_for_type(src0->type);
    if (!tsr_mm_shape_ok(cfg, fmt, (uint64_t) src0->ne[1], (uint64_t) src0->ne[0])) {
        return false;                         // K > K_MAX, K % 32 != 0, or one token exceeds SPM
    }
    if (ggml_nrows(src1) > (int64_t) UINT32_MAX) {
        return false;
    }
    // No limit on the token count: the size of src1 / dst grows with T, but the activations are
    // never all resident at once. Compute buffers that do not fit the IO region live in host
    // memory and graph_compute stages them through IO in T tiles (spec v1 lowering 제약). A
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
