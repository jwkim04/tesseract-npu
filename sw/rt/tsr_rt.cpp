// tsr_rt.cpp — see tsr_rt.h
#include "tsr_rt.h"

#include <algorithm>
#include <cinttypes>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "tsr_isa.h"
#include "tsr_ref.h"
#include "tsr_transport.h"

#include "tsr-alloc.h"     // tsr_lower
#include "tsr-lower.h"
#include "tsr-wlayout.h"
#include "tsr-weights.h"   // tsr_row_to_dev_blocks: the ggml backend's load-time conversion (ggml-base)

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "tsr_rt writes weights in host byte order; only little-endian hosts are supported"
#endif

namespace tsr::rt {

namespace {

constexpr uint64_t HEAP_ALIGN      = 4096;              // as the ggml backend
constexpr size_t   XFER_CHUNK      = 8u << 20;          // weight scatter batch (tsr_weight_store_rows)

int debug_level() {
    static const int lvl = [] {
        const char * e = getenv("TSR_RT_DEBUG");
        return e ? atoi(e) : 0;
    }();
    return lvl;
}

#define TSR_RT_DBG(lvl, ...)                                \
    do {                                                    \
        if (debug_level() >= (lvl)) {                       \
            fprintf(stderr, "[tsr_rt] " __VA_ARGS__);       \
        }                                                   \
    } while (0)

std::string strf(const char * fmt, ...) __attribute__((format(printf, 1, 2)));
std::string strf(const char * fmt, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    return buf;
}

double mib(uint64_t b) { return (double) b / (1024.0 * 1024.0); }

enum ggml_type ggml_type_of(dtype t) {
    switch (t) {
        case dtype::f32:  return GGML_TYPE_F32;
        case dtype::f16:  return GGML_TYPE_F16;
        case dtype::bf16: return GGML_TYPE_BF16;
        case dtype::q8_0: return GGML_TYPE_Q8_0;
        case dtype::q4_k: return GGML_TYPE_Q4_K;
    }
    return GGML_TYPE_COUNT;
}

bool dtype_valid(dtype t) { return (uint32_t) t <= (uint32_t) dtype::q4_k; }

// Elements per block and bytes per block of the host type (1 element "blocks" for f32/f16/bf16).
uint32_t dtype_block(dtype t) { return t == dtype::q8_0 ? 32u : (t == dtype::q4_k ? 256u : 1u); }
size_t   dtype_block_bytes(dtype t) {
    switch (t) {
        case dtype::f32:  return 4;
        case dtype::f16:
        case dtype::bf16: return 2;
        case dtype::q8_0: return sizeof(tsr_block_q8_0);
        case dtype::q4_k: return sizeof(tsr_block_q4_K);
    }
    return 0;
}
// Alignment the converters read the host data with (f32: float, f16/bf16 and blocks: uint16).
size_t dtype_align(dtype t) { return t == dtype::f32 ? 4 : 2; }

// One N x K part of a weight: rows [r0, r0 + n), columns [k0, k0 + k); its own symmetric allocation.
struct part {
    uint32_t r0 = 0, n = 0, k0 = 0, k = 0;
    uint64_t w_off = 0;        // byte offset of W_OFF (every channel)
    uint64_t ch_bytes = 0;     // reserved per channel
};

struct plan {
    uint32_t fmt = 0, kind = 0;
    uint32_t n_parts = 0, n_part = 0;
    uint32_t k_parts = 0, k_part = 0;
};

struct weight {
    dtype    src = dtype::f32;
    uint32_t N = 0, K = 0;
    plan     pl;
    std::vector<part> parts;   // [i * k_parts + j]
    uint64_t dev_bytes = 0;
    const part & at(uint32_t i, uint32_t j) const { return parts[(size_t) i * pl.k_parts + j]; }
};

// The N / K split of a weight (see tsr_rt.h). nullptr on success, else the reason.
const char * make_plan(const tsr_config & cfg, dtype t, uint32_t N, uint32_t K, const weight_options & o, plan * p) {
    if (!dtype_valid(t)) return "unknown dtype";
    const uint32_t fmt = dtype_device_format(t);
    const uint32_t bs = tsr_fmt_block_size(fmt);
    if (N == 0) return "N must be >= 1";
    if (K == 0 || K % bs != 0) return fmt == TSR_FMT_Q4_K ? "K must be a positive multiple of 256 (Q4_K)"
                                                          : "K must be a positive multiple of 32 (Q8_0)";
    uint32_t k_lim = std::min<uint32_t>(cfg.k_max, 0xFFFFu) / bs * bs;   // K_MAX and the 16-bit MVM K field
    if (o.max_part_k != 0) {
        if (o.max_part_k % bs != 0) return "max_part_k must be a multiple of the block size (32 Q8_0, 256 Q4_K)";
        k_lim = std::min(k_lim, o.max_part_k);
    }
    if (k_lim < bs) return "K_MAX is smaller than one block";
    const uint32_t B  = K / bs;
    const uint32_t L  = k_lim / bs;
    const uint32_t kp = (B + L - 1) / L;
    const uint32_t pb = (B + kp - 1) / kp;                 // blocks per K part (last one may be smaller)
    p->fmt     = fmt;
    p->kind    = tsr_qa_kind_for_fmt(fmt);
    p->k_part  = pb * bs;
    p->k_parts = (B + pb - 1) / pb;

    // SPM per token: the x row (all of K, K * 4 / 64 lines), the QA of the largest K part, y.
    const uint64_t spm_lines = cfg.spm_bytes / TSR_LINE_BYTES;
    const uint64_t base = (uint64_t) K / 16 + tsr_qa_lines(p->kind, p->k_part);
    if (base + 1 > spm_lines) return "one token of x and its QA does not fit the SPM";
    const uint64_t per_tok = spm_lines / tsr_mm_t_max(cfg);
    const uint64_t yl_cap = per_tok > base ? per_tok - base : spm_lines - base;   // T_MAX tokens if possible
    uint64_t rows_cap = std::min<uint64_t>(yl_cap * (TSR_LINE_BYTES / 4), UINT32_MAX);
    if (o.max_part_rows != 0) rows_cap = std::min<uint64_t>(rows_cap, o.max_part_rows);
    const uint64_t np = (N + rows_cap - 1) / rows_cap;
    p->n_part  = (uint32_t) ((N + np - 1) / np);
    p->n_parts = (uint32_t) ((N + p->n_part - 1) / p->n_part);
    if (p->k_parts == 1 && !tsr_mm_shape_ok(cfg, fmt, p->n_part, K)) return "shape not lowerable (tsr_mm_shape_ok)";
    return nullptr;
}

}  // namespace

const char * dtype_name(dtype t) {
    switch (t) {
        case dtype::f32:  return "f32";
        case dtype::f16:  return "f16";
        case dtype::bf16: return "bf16";
        case dtype::q8_0: return "q8_0";
        case dtype::q4_k: return "q4_k";
    }
    return "?";
}

uint32_t dtype_device_format(dtype t) { return t == dtype::q4_k ? TSR_FMT_Q4_K : TSR_FMT_Q8_0; }

size_t dtype_row_bytes(dtype t, uint32_t K) {
    if (!dtype_valid(t) || K % dtype_block(t) != 0) return 0;
    return (size_t) (K / dtype_block(t)) * dtype_block_bytes(t);
}

// ---------------------------------------------------------------------------
// Device state
// ---------------------------------------------------------------------------

struct device::impl {
    mutable std::mutex mtx;
    tsr_transport * tp = nullptr;
    tsr_config  cfg = {};
    std::string kind;
    std::string desc;

    uint64_t io_off = 0;              // ch0 byte offset of the activation area (IO region)
    uint64_t io_size = 0;
    tsr_range_alloc heap;             // WEIGHT heap, symmetric (one offset on every channel)

    std::unordered_map<weight_handle, weight> weights;
    weight_handle next_handle = 1;

    counters cnt;

    ~impl() {
        if (tp != nullptr) tsr_transport_close(tp);
    }

    // -- transport -----------------------------------------------------------
    void write(uint32_t ch, uint64_t off, const void * src, size_t size, bool is_weight = false) {
        if (size == 0) return;
        if (tsr_transport_write(tp, ch, off, src, size) != 0) {
            throw error(strf("tsr_rt: transport write failed (ch %u, offset 0x%" PRIx64 ", %zu bytes)", ch, off, size));
        }
        cnt.h2d_bytes += size;
        if (is_weight) cnt.h2d_weight_bytes += size;
    }
    void read(uint32_t ch, uint64_t off, void * dst, size_t size) {
        if (size == 0) return;
        if (tsr_transport_read(tp, ch, off, dst, size) != 0) {
            throw error(strf("tsr_rt: transport read failed (ch %u, offset 0x%" PRIx64 ", %zu bytes)", ch, off, size));
        }
        cnt.d2h_bytes += size;
    }

    const weight & get(weight_handle h) const {
        auto it = weights.find(h);
        if (it == weights.end()) throw std::invalid_argument(strf("tsr_rt: unknown weight handle %" PRIu64, h));
        return it->second;
    }

    void release_parts(weight & w) {
        for (part & p : w.parts) {
            if (p.ch_bytes != 0) heap.release(p.w_off);
            p.ch_bytes = 0;
        }
    }
};

// ---------------------------------------------------------------------------
// device
// ---------------------------------------------------------------------------

device::device(const open_options & o) : p_(new impl) {
    impl & d = *p_;
    d.tp = tsr_transport_open(o.kind);
    if (d.tp == nullptr) {
        const char * env = getenv("TSR_DEVICE");
        throw error(strf("tsr_rt: failed to open the Tesseract transport (%s%s)",
                         o.kind && o.kind[0] ? "kind " : "TSR_DEVICE=",
                         o.kind && o.kind[0] ? o.kind : (env ? env : "(unset: sim)")));
    }
    const tsr_config * c = tsr_transport_config(d.tp);
    const char * bad = nullptr;
    if (c == nullptr)                                 bad = "no configuration";
    else if (c->beat_bits != 512)                     bad = "BEAT_BITS != 512 (v1 software assumes 64-byte beats)";
    else if (c->num_ch < 1 || c->num_ch > TSR_NUM_CH_MAX) bad = "NUM_CH outside 1..16";
    else if (c->t_max < 1)                            bad = "T_MAX == 0";
    else if (c->spm_bytes < 64 * TSR_LINE_BYTES)      bad = "SPM too small";
    else if (c->ch_bytes <= TSR_REGION_WEIGHT_OFF)    bad = "channel smaller than the DRAM map (CH_BYTES <= WEIGHT_OFF)";
    if (bad != nullptr) throw error(strf("tsr_rt: unusable Tesseract configuration: %s", bad));   // ~impl closes
    d.cfg  = *c;
    d.kind = tsr_transport_name(d.tp);
    d.io_off  = TSR_REGION_IO_OFF;
    d.io_size = TSR_REGION_IO_SIZE;
    if (o.io_bytes != 0) {
        d.io_size = std::min<uint64_t>(d.io_size, o.io_bytes / TSR_LINE_BYTES * TSR_LINE_BYTES);
        if (d.io_size == 0) throw std::invalid_argument("tsr_rt: open_options.io_bytes must be >= 64");
    }
    d.heap.init(TSR_REGION_WEIGHT_OFF, d.cfg.ch_bytes - TSR_REGION_WEIGHT_OFF, HEAP_ALIGN);
    d.desc = strf("Tesseract (%s, C=%u, LANES=%u, T_MAX=%u, K_MAX=%u, SPM=%u KiB, %.0f GiB/ch)", d.kind.c_str(), c->num_ch,
                  c->lanes, c->t_max, c->k_max, c->spm_bytes / 1024, (double) c->ch_bytes / (1024.0 * 1024.0 * 1024.0));
    TSR_RT_DBG(1, "opened %s; IO %.2f MiB\n", d.desc.c_str(), mib(d.io_size));
}

device::~device() = default;

const tsr_config & device::config() const { return p_->cfg; }
const char * device::transport() const { return p_->kind.c_str(); }
const char * device::description() const { return p_->desc.c_str(); }

const char * device::unsupported_reason(dtype t, uint32_t N, uint32_t K, const weight_options & o) const {
    std::lock_guard<std::mutex> lk(p_->mtx);
    plan pl;
    return make_plan(p_->cfg, t, N, K, o, &pl);
}

weight_handle device::upload(dtype t, const void * data, uint32_t N, uint32_t K, size_t row_stride_bytes,
                             const weight_options & o) {
    std::lock_guard<std::mutex> lk(p_->mtx);
    impl & d = *p_;
    weight w;
    if (const char * why = make_plan(d.cfg, t, N, K, o, &w.pl)) {
        throw std::invalid_argument(strf("tsr_rt: cannot upload a %s weight [%u x %u]: %s", dtype_name(t), N, K, why));
    }
    const size_t row_bytes = dtype_row_bytes(t, K);
    const size_t rs = row_stride_bytes != 0 ? row_stride_bytes : row_bytes;
    if (rs < row_bytes) throw std::invalid_argument(strf("tsr_rt: row stride %zu < row bytes %zu", rs, row_bytes));
    if (data == nullptr) throw std::invalid_argument("tsr_rt: upload of a null pointer");
    w.src = t;
    w.N = N;
    w.K = K;
    const uint32_t C = d.cfg.num_ch;
    for (uint32_t i = 0; i < w.pl.n_parts; ++i) {
        for (uint32_t j = 0; j < w.pl.k_parts; ++j) {
            part p;
            p.r0 = i * w.pl.n_part;
            p.n  = std::min(w.pl.n_part, N - p.r0);
            p.k0 = j * w.pl.k_part;
            p.k  = std::min(w.pl.k_part, K - p.k0);
            w.parts.push_back(p);
        }
    }
    try {
        for (part & p : w.parts) {
            const uint64_t need = tsr_tensor_ch_beats(w.pl.fmt, p.n, p.k, C) * TSR_BEAT_BYTES;
            if (!d.heap.alloc(need, &p.w_off)) {
                throw error(strf("tsr_rt: WEIGHT heap full: a %s weight [%u x %u] part needs %.1f MiB per channel, %.1f MiB "
                                 "free (largest %.1f MiB)", dtype_name(t), N, K, mib(need), mib(d.heap.free_bytes()),
                                 mib(d.heap.largest_free())));
            }
            p.ch_bytes = need;
            w.dev_bytes += need * C;
        }
        // Convert (the ggml backend's tsr_row_to_dev_blocks: f32 / f16 / bf16 -> quantize_row_q8_0_ref,
        // native blocks as they are), repack and scatter each part. Conversion is per 32-block, so a
        // K part converted on its own equals the same columns of the converted whole row.
        const enum ggml_type gt = ggml_type_of(t);
        const size_t align = dtype_align(t);
        std::vector<float> f32tmp(w.pl.k_part);
        std::vector<uint8_t> aligned;
        const uint8_t * base = (const uint8_t *) data;
        for (const part & p : w.parts) {
            const size_t slice_off = (size_t) (p.k0 / dtype_block(t)) * dtype_block_bytes(t);
            const size_t slice_bytes = dtype_row_bytes(t, p.k);
            aligned.resize(slice_bytes + 8);
            tsr_weight_store_rows(C, w.pl.fmt, p.n, p.k, p.w_off, 0, p.n, XFER_CHUNK,
                [&](uint32_t r, void * scratch) -> const void * {
                    const uint8_t * src = base + (size_t) (p.r0 + r) * rs + slice_off;
                    if ((uintptr_t) src % align != 0) {
                        memcpy(aligned.data(), src, slice_bytes);
                        src = aligned.data();
                    }
                    return tsr_row_to_dev_blocks(gt, w.pl.fmt, src, p.k, scratch, f32tmp.data());
                },
                [&d](uint32_t ch, uint64_t off, const void * src, size_t size) { d.write(ch, off, src, size, true); });
        }
    } catch (...) {
        d.release_parts(w);
        throw;
    }
    const weight_handle h = d.next_handle++;
    TSR_RT_DBG(1, "weight #%" PRIu64 " %s [%u x %u] -> %s, %u x %u parts (n_part %u, k_part %u), %.2f MiB on the device\n", h,
               dtype_name(t), N, K, w.pl.fmt == TSR_FMT_Q8_0 ? "Q8_0" : "Q4_K", w.pl.n_parts, w.pl.k_parts, w.pl.n_part,
               w.pl.k_part, mib(w.dev_bytes));
    d.weights.emplace(h, std::move(w));
    return h;
}

void device::free_weight(weight_handle h) {
    std::lock_guard<std::mutex> lk(p_->mtx);
    auto it = p_->weights.find(h);
    if (it == p_->weights.end()) throw std::invalid_argument(strf("tsr_rt: unknown weight handle %" PRIu64, h));
    p_->release_parts(it->second);
    p_->weights.erase(it);
}

weight_info device::info(weight_handle h) const {
    std::lock_guard<std::mutex> lk(p_->mtx);
    const weight & w = p_->get(h);
    weight_info r = {};
    r.N = w.N;
    r.K = w.K;
    r.src = w.src;
    r.fmt = w.pl.fmt;
    r.n_parts = w.pl.n_parts;
    r.n_part = w.pl.n_part;
    r.k_parts = w.pl.k_parts;
    r.k_part = w.pl.k_part;
    r.dev_bytes = w.dev_bytes;
    return r;
}

void device::read_blocks(weight_handle h, uint32_t r0, uint32_t r1, void * dst) const {
    std::lock_guard<std::mutex> lk(p_->mtx);
    impl & d = *p_;
    const weight & w = d.get(h);
    if (r0 > r1 || r1 > w.N) throw std::invalid_argument("tsr_rt: read_blocks row range outside the weight");
    const size_t full_rb = tsr_dev_blocks_row_bytes(w.pl.fmt, w.K);
    const uint32_t bs = tsr_fmt_block_size(w.pl.fmt);
    const size_t bb = tsr_fmt_ggml_block_bytes(w.pl.fmt);
    uint8_t * out = (uint8_t *) dst;
    for (uint32_t i = 0; i < w.pl.n_parts; ++i) {
        const part & pi = w.at(i, 0);
        const uint32_t a = std::max(r0, pi.r0), b = std::min(r1, pi.r0 + pi.n);
        if (a >= b) continue;
        for (uint32_t j = 0; j < w.pl.k_parts; ++j) {
            const part & p = w.at(i, j);
            const size_t prb = tsr_dev_blocks_row_bytes(w.pl.fmt, p.k);
            tsr_weight_load_rows(d.cfg.num_ch, w.pl.fmt, p.n, p.k, p.w_off, a - p.r0, b - p.r0, XFER_CHUNK,
                [&d](uint32_t ch, uint64_t off, void * q, size_t n) { d.read(ch, off, q, n); },
                [&](uint32_t rr, const void * blocks) {
                    memcpy(out + (size_t) (p.r0 + rr - r0) * full_rb + (size_t) (p.k0 / bs) * bb, blocks, prb);
                });
        }
    }
}

void device::linear(weight_handle h, const float * x, uint32_t T, float * y, size_t x_stride, size_t y_stride) {
    std::lock_guard<std::mutex> lk(p_->mtx);
    impl & d = *p_;
    const weight & w = d.get(h);
    if (T == 0) return;
    const size_t xs = x_stride != 0 ? x_stride : w.K;
    const size_t ys = y_stride != 0 ? y_stride : w.N;
    if (xs < w.K || ys < w.N) throw std::invalid_argument("tsr_rt: linear: x_stride < K or y_stride < N");
    if (x == nullptr || y == nullptr) throw std::invalid_argument("tsr_rt: linear: null x or y");
    // v1 ran a cached LD / QACT / MVM / ST program per call through the transport's run
    // interface; that path is deleted with the host command buffer (X0). The one-op OPL launch of the `graph` kernel
    // replaces it (docs/v2/06-software-verification.md §9.4, P3.2).
    throw error(strf("tsr_rt: linear on %s: device execution is not available in this build (the v1 command-buffer "
                     "path is deleted; linear returns with the op-list launch path). linear_ref computes the same "
                     "result on the host.", d.kind.c_str()));
}

void device::linear_ref(weight_handle h, const float * x, uint32_t T, float * y, size_t x_stride, size_t y_stride) const {
    std::lock_guard<std::mutex> lk(p_->mtx);
    impl & d = *p_;
    const weight & w = d.get(h);
    if (T == 0) return;
    const size_t xs = x_stride != 0 ? x_stride : w.K;
    const size_t ys = y_stride != 0 ? y_stride : w.N;
    if (xs < w.K || ys < w.N) throw std::invalid_argument("tsr_rt: linear_ref: x_stride < K or y_stride < N");

    // the repacked rows as stored on the device, per part
    const uint32_t C = d.cfg.num_ch;
    std::vector<std::vector<uint8_t>> rows(w.parts.size());
    for (size_t pi = 0; pi < w.parts.size(); ++pi) {
        const part & p = w.parts[pi];
        const uint64_t rb = (uint64_t) tsr_row_beats(w.pl.fmt, p.k) * TSR_BEAT_BYTES;
        const uint32_t R = tsr_rows_per_ch(p.n, C);
        rows[pi].resize((size_t) (p.n * rb));
        for (uint32_t c = 0; c < C; ++c) {
            const uint64_t lo = (uint64_t) c * R, hi = std::min<uint64_t>((uint64_t) (c + 1) * R, p.n);
            if (hi > lo) d.read(c, p.w_off, rows[pi].data() + lo * rb, (size_t) ((hi - lo) * rb));
        }
    }
    // QA of every token and K part (tsr_qact = QACT)
    const uint32_t kp = w.pl.k_parts;
    std::vector<size_t> qa_off(kp + 1, 0);
    for (uint32_t j = 0; j < kp; ++j) qa_off[j + 1] = qa_off[j] + (size_t) tsr_qa_lines(w.pl.kind, w.at(0, j).k) * TSR_LINE_BYTES;
    std::vector<uint8_t> qa((size_t) T * qa_off[kp]);
    for (uint32_t t = 0; t < T; ++t) {
        for (uint32_t j = 0; j < kp; ++j) {
            const part & p = w.at(0, j);
            tsr_qact(w.pl.kind, x + (size_t) t * xs + p.k0, p.k, qa.data() + (size_t) t * qa_off[kp] + qa_off[j]);
        }
    }
    // rows in parallel
    auto work = [&](uint32_t n_lo, uint32_t n_hi) {
        for (uint32_t n = n_lo; n < n_hi; ++n) {
            const uint32_t i = n / w.pl.n_part;
            const uint32_t r = n - w.at(i, 0).r0;
            for (uint32_t t = 0; t < T; ++t) {
                float acc = 0.0f;
                for (uint32_t j = 0; j < kp; ++j) {
                    const size_t pi = (size_t) i * kp + j;
                    const part & p = w.parts[pi];
                    const uint64_t rb = (uint64_t) tsr_row_beats(w.pl.fmt, p.k) * TSR_BEAT_BYTES;
                    const float v = tsr_dot_row(w.pl.fmt, rows[pi].data() + r * rb, p.k,
                                                qa.data() + (size_t) t * qa_off[kp] + qa_off[j]);
                    acc = j == 0 ? v : tsr_fadd(acc, v);
                }
                y[(size_t) t * ys + n] = acc;
            }
        }
    };
    const uint64_t work_units = (uint64_t) w.N * T * w.K;
    unsigned nth = std::max(1u, std::min(4u, std::thread::hardware_concurrency()));
    if (work_units < (1u << 22)) nth = 1;
    nth = std::min<unsigned>(nth, w.N);
    if (nth <= 1) {
        work(0, w.N);
    } else {
        std::vector<std::thread> th;
        const uint32_t step = (w.N + nth - 1) / nth;
        for (unsigned k = 0; k < nth; ++k) {
            const uint32_t lo = std::min<uint32_t>(w.N, k * step), hi = std::min<uint32_t>(w.N, lo + step);
            if (lo < hi) th.emplace_back(work, lo, hi);
        }
        for (auto & t : th) t.join();
    }
}

counters device::get_counters() const {
    std::lock_guard<std::mutex> lk(p_->mtx);
    return p_->cnt;
}

void device::reset_counters() {
    std::lock_guard<std::mutex> lk(p_->mtx);
    p_->cnt = counters();
}

memory_info device::memory() const {
    std::lock_guard<std::mutex> lk(p_->mtx);
    const impl & d = *p_;
    memory_info m = {};
    m.weight_total = d.heap.total();
    m.weight_free = d.heap.free_bytes();
    m.weight_largest_free = d.heap.largest_free();
    m.n_weights = (uint32_t) d.weights.size();
    return m;
}

}  // namespace tsr::rt
