// tsr_rt.cpp — see tsr_rt.h
#include "tsr_rt.h"

#include <algorithm>
#include <cinttypes>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <list>
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
#error "tsr_rt writes instructions and weights in host byte order; only little-endian hosts are supported"
#endif
static_assert(sizeof(tsr_insn) == TSR_INSN_BYTES, "tsr_insn must be 32 bytes");

namespace tsr::rt {

namespace {

constexpr uint64_t HEAP_ALIGN      = 4096;              // as the ggml backend
constexpr size_t   XFER_CHUNK      = 8u << 20;          // weight scatter batch (tsr_weight_store_rows)
constexpr uint32_t PROG_MAX_INSNS  = (4u << 20) / TSR_INSN_BYTES - 1;   // 4 MiB per program incl. END
constexpr size_t   PROG_MAX_COUNT  = 65536;
constexpr uint64_t CMD_ALIGN       = TSR_BEAT_BYTES;

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

struct prog_slot {
    uint64_t cmd_off = 0;      // ch0 byte offset of the program (CMD region)
    uint64_t bytes = 0;
    uint32_t count = 0;        // instructions including END
    std::list<std::string>::iterator lru;
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
    bool        read_perf = true;

    uint64_t io_off = 0;              // ch0 byte offset of the activation area (IO region)
    uint64_t io_size = 0;
    tsr_range_alloc heap;             // WEIGHT heap, symmetric (one offset on every channel)
    tsr_range_alloc cmd;              // CMD region: one slot per cached program

    std::unordered_map<weight_handle, weight> weights;
    weight_handle next_handle = 1;

    std::unordered_map<std::string, prog_slot> progs;
    std::list<std::string> lru;       // front = most recently used

    std::vector<uint64_t> perf_last;  // PERF_CYCLES, PERF_MVM_CYCLES, PERF_CH_RD_BEATS[c]
    counters cnt;
    std::vector<uint8_t> hbuf;        // host staging (x packing, y gathering)

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
    uint64_t perf(uint32_t idx) {
        const uint32_t lo = tsr_transport_csr_read(tp, TSR_CSR_PERF_BASE + 8 * idx);       // LO first: snapshot
        const uint32_t hi = tsr_transport_csr_read(tp, TSR_CSR_PERF_BASE + 8 * idx + 4);
        return ((uint64_t) hi << 32) | lo;
    }
    void perf_snapshot(std::vector<uint64_t> & v) {
        v.resize(2 + cfg.num_ch);
        v[0] = perf(TSR_PERF_CYCLES);
        v[1] = perf(TSR_PERF_MVM_CYCLES);
        for (uint32_t c = 0; c < cfg.num_ch; ++c) v[2 + c] = perf(tsr_perf_ch_rd_beats(c));
    }

    // -- programs ------------------------------------------------------------
    void evict_lru() {
        auto it = progs.find(lru.back());
        cmd.release(it->second.cmd_off);
        progs.erase(it);
        lru.pop_back();
    }

    // The resident program for `key`, emitting and writing it on a miss.
    template <class Emit>
    const prog_slot & program(const std::string & key, Emit && emit) {
        auto it = progs.find(key);
        if (it != progs.end()) {
            lru.splice(lru.begin(), lru, it->second.lru);
            cnt.prog_hits++;
            return it->second;
        }
        std::vector<tsr_insn> insns;
        emit(insns);
        if (insns.size() > PROG_MAX_INSNS) {
            throw error(strf("tsr_rt: internal error: program of %zu instructions exceeds the bound", insns.size()));
        }
        insns.push_back(tsr_enc_end());
        const uint64_t bytes = insns.size() * sizeof(tsr_insn);
        while (progs.size() >= PROG_MAX_COUNT) evict_lru();
        uint64_t off = 0;
        while (!cmd.alloc(bytes, &off)) {
            if (lru.empty()) throw error(strf("tsr_rt: program of %" PRIu64 " bytes does not fit the CMD region", bytes));
            evict_lru();
        }
        try {
            write(0, off, insns.data(), bytes);
        } catch (...) {
            cmd.release(off);
            throw;
        }
        lru.push_front(key);
        prog_slot & s = progs[key];
        s.cmd_off = off;
        s.bytes = bytes;
        s.count = (uint32_t) insns.size();
        s.lru = lru.begin();
        cnt.prog_misses++;
        if (debug_level() >= 2) {
            for (size_t i = 0; i < insns.size(); ++i) {
                fprintf(stderr, "[tsr_rt]   %5zu  %s\n", i, i + 1 == insns.size() ? "END" : tsr_insn_str(insns[i]).c_str());
            }
        }
        return s;
    }

    void exec(const prog_slot & s) {
        tsr_exec_result res = {};
        const int rc = tsr_transport_exec(tp, s.cmd_off / TSR_BEAT_BYTES, s.count, nullptr, &res);
        cnt.execs++;
        if (read_perf) {
            std::vector<uint64_t> now;
            perf_snapshot(now);
            // PERF_* do not change between runs (spec 구현 명확화), so now - last is this run's;
            // a counter that went down was reset (SOFT_RESET): count from zero.
            auto delta = [&](size_t i) { return now[i] >= perf_last[i] ? now[i] - perf_last[i] : now[i]; };
            cnt.perf_cycles += delta(0);
            cnt.perf_mvm_cycles += delta(1);
            for (uint32_t c = 0; c < cfg.num_ch; ++c) cnt.dev_read_bytes += delta(2 + c) * TSR_BEAT_BYTES;
            perf_last.swap(now);
        }
        if (rc != 0 || (res.status & TSR_STATUS_ERR) || !(res.status & TSR_STATUS_DONE)) {
            std::string at = "-";
            if (res.err_pc + 1 == s.count) {
                at = "END";
            } else if (res.err_pc < s.count) {
                tsr_insn in = {};
                if (tsr_transport_read(tp, 0, s.cmd_off + (uint64_t) res.err_pc * sizeof(tsr_insn), &in, sizeof(in)) == 0) {
                    at = tsr_insn_str(in);
                }
            }
            throw error(strf("tsr_rt: device execution failed on %s: rc=%d STATUS=0x%x ERR_CODE=%u ERR_PC=%u ERR_INFO=0x%x "
                             "(program of %u instructions at CMD beat 0x%" PRIx64 "; failing instruction: %s)",
                             kind.c_str(), rc, res.status, res.err_code, res.err_pc, res.err_info, s.count,
                             s.cmd_off / TSR_BEAT_BYTES, at.c_str()));
        }
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

    // -- lowering of one linear call of tc tokens ---------------------------
    // IO layout: x rows (K * 4 bytes each) at io_off, then per N part i its y rows
    // (y_lines(n_i) * 64 bytes each).
    static uint32_t y_lines(uint32_t n) { return (uint32_t) tsr_ceil_div_u64((uint64_t) n * 4, TSR_LINE_BYTES); }

    uint64_t y_off(const weight & w, uint32_t tc, uint32_t i) const {
        uint64_t off = io_off + (uint64_t) tc * w.K * 4;
        for (uint32_t ii = 0; ii < i; ++ii) off += (uint64_t) tc * y_lines(w.at(ii, 0).n) * TSR_LINE_BYTES;
        return off;
    }

    // K parts of N part i: LD the x tile once, then per K part QACT on its columns and MVM (ACC
    // after the first), then ST (spec MVM ACC: y_out = y_old + y, one f32 add).
    void emit_ksplit(const weight & w, uint32_t i, uint32_t tc, uint64_t x_beat, uint64_t y_beat,
                     std::vector<tsr_insn> & insns) const {
        const uint32_t xl  = w.K / 16;
        const uint32_t qal = tsr_qa_lines(w.pl.kind, w.pl.k_part);
        const uint32_t n   = w.at(i, 0).n;
        const uint32_t yl  = y_lines(n);
        const uint64_t spm_lines = cfg.spm_bytes / TSR_LINE_BYTES;
        const uint64_t lpt = (uint64_t) xl + qal + yl;
        const uint32_t tt_max = (uint32_t) std::min<uint64_t>({ (uint64_t) tc, (uint64_t) tsr_mm_t_max(cfg), spm_lines / lpt });
        if (tt_max < 1) throw error("tsr_rt: internal error: K-split tile does not fit the SPM");
        const uint32_t spm_x  = 0;
        const uint32_t spm_qa = spm_x + tt_max * xl;
        const uint32_t spm_y  = spm_qa + tt_max * qal;
        for (uint32_t t0 = 0; t0 < tc; t0 += tt_max) {
            const uint32_t tt = std::min(tt_max, tc - t0);
            insns.push_back(tsr_enc_ld(0, spm_x, (uint32_t) (x_beat + (uint64_t) t0 * xl), tt * xl));
            for (uint32_t j = 0; j < w.pl.k_parts; ++j) {
                const part & p = w.at(i, j);
                insns.push_back(tsr_enc_qact(w.pl.kind, spm_qa, spm_x + p.k0 / 16, p.k, tt, xl));
                insns.push_back(tsr_enc_mvm(w.pl.fmt, j == 0 ? 0 : TSR_MVM_FLAG_ACC, spm_y, spm_qa,
                                            (uint32_t) (p.w_off / TSR_BEAT_BYTES), n, p.k, tt, yl));
            }
            insns.push_back(tsr_enc_st(0, spm_y, (uint32_t) (y_beat + (uint64_t) t0 * yl), tt * yl));
        }
    }

    const prog_slot & linear_program(const weight & w, uint32_t tc) {
        const uint64_t x_beat = io_off / TSR_BEAT_BYTES;
        std::vector<uint64_t> k;
        k.reserve(8 + w.parts.size() * 5);
        k.push_back(1);                          // key layout version
        k.push_back(w.pl.fmt);
        k.push_back(w.K);
        k.push_back(tc);
        k.push_back(x_beat);
        k.push_back(w.pl.n_parts);
        k.push_back(w.pl.k_parts);
        for (uint32_t i = 0; i < w.pl.n_parts; ++i) {
            k.push_back(y_off(w, tc, i) / TSR_BEAT_BYTES);
            for (uint32_t j = 0; j < w.pl.k_parts; ++j) {
                const part & p = w.at(i, j);
                k.push_back(p.n);
                k.push_back(p.k0);
                k.push_back(p.k);
                k.push_back(p.w_off / TSR_BEAT_BYTES);
            }
        }
        const std::string key((const char *) k.data(), k.size() * sizeof(uint64_t));
        return program(key, [&](std::vector<tsr_insn> & insns) {
            for (uint32_t i = 0; i < w.pl.n_parts; ++i) {
                const uint64_t y_beat = y_off(w, tc, i) / TSR_BEAT_BYTES;
                if (w.pl.k_parts == 1) {
                    const part & p = w.at(i, 0);
                    tsr_mm_desc d = {};
                    d.fmt = w.pl.fmt;
                    d.N = p.n;
                    d.K = w.K;
                    d.T = tc;
                    d.w_off_beat = p.w_off / TSR_BEAT_BYTES;
                    d.x_beat = x_beat;
                    d.y_beat = y_beat;
                    std::vector<size_t> cuts;
                    tsr_mm_emit(cfg, d, insns, cuts);
                } else {
                    emit_ksplit(w, i, tc, x_beat, y_beat, insns);
                }
            }
            TSR_RT_DBG(1, "program: %s [%u x %u] T=%u, %u x %u parts, %zu insns\n", w.pl.fmt == TSR_FMT_Q8_0 ? "Q8_0" : "Q4_K",
                       w.N, w.K, tc, w.pl.n_parts, w.pl.k_parts, insns.size());
        });
    }
};

// ---------------------------------------------------------------------------
// device
// ---------------------------------------------------------------------------

device::device(const open_options & o) : p_(new impl) {
    impl & d = *p_;
    d.read_perf = o.read_perf;
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
    d.cmd.init(TSR_REGION_CMD_OFF, TSR_REGION_CMD_SIZE, CMD_ALIGN);
    d.perf_snapshot(d.perf_last);
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

    const uint64_t x_row = (uint64_t) w.K * 4;
    uint64_t y_tok = 0;
    for (uint32_t i = 0; i < w.pl.n_parts; ++i) y_tok += (uint64_t) impl::y_lines(w.at(i, 0).n) * TSR_LINE_BYTES;
    const uint64_t t_io = d.io_size / (x_row + y_tok);
    if (t_io == 0) {
        throw error(strf("tsr_rt: one token of x and y needs %" PRIu64 " bytes of IO, %" PRIu64 " available",
                         x_row + y_tok, d.io_size));
    }
    const uint64_t t_prog = std::max<uint64_t>(1, PROG_MAX_INSNS / ((uint64_t) w.pl.n_parts * (2 + 2 * w.pl.k_parts)));
    const uint32_t tc_max = (uint32_t) std::min<uint64_t>({ (uint64_t) T, t_io, t_prog });
    const bool direct_y = w.pl.n_parts == 1 && w.N % 16 == 0 && ys == w.N;

    for (uint32_t t0 = 0; t0 < T; t0 += tc_max) {
        const uint32_t tc = std::min(tc_max, T - t0);
        // x tile -> IO (rows of K floats; K % 32 == 0 so every row starts on a line)
        if (xs == w.K) {
            d.write(0, d.io_off, x + (size_t) t0 * w.K, (size_t) tc * x_row);
        } else {
            d.hbuf.resize((size_t) tc * x_row);
            for (uint32_t t = 0; t < tc; ++t) {
                memcpy(d.hbuf.data() + (size_t) t * x_row, x + (size_t) (t0 + t) * xs, (size_t) x_row);
            }
            d.write(0, d.io_off, d.hbuf.data(), d.hbuf.size());
        }
        const prog_slot & s = d.linear_program(w, tc);
        d.exec(s);
        if (direct_y) {
            d.read(0, d.y_off(w, tc, 0), y + (size_t) t0 * w.N, (size_t) tc * w.N * 4);
        } else {
            for (uint32_t i = 0; i < w.pl.n_parts; ++i) {
                const part & p = w.at(i, 0);
                const uint64_t ystr = (uint64_t) impl::y_lines(p.n) * TSR_LINE_BYTES;
                d.hbuf.resize((size_t) (tc * ystr));
                d.read(0, d.y_off(w, tc, i), d.hbuf.data(), d.hbuf.size());
                for (uint32_t t = 0; t < tc; ++t) {
                    memcpy(y + (size_t) (t0 + t) * ys + p.r0, d.hbuf.data() + (size_t) t * ystr, (size_t) p.n * 4);
                }
            }
        }
    }
    d.cnt.linears++;
    d.cnt.tokens += T;
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
    m.prog_cached = d.progs.size();
    m.prog_cmd_bytes = d.cmd.used();
    return m;
}

}  // namespace tsr::rt
