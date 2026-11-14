// tsr_rt_internal.h — the state of tsr::rt::device shared by tsr_rt.cpp (weights, linear) and tsr_rt_graph.cpp (graphs).
// Not part of the API: tsr_rt.h is.
#pragma once

#include <cinttypes>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "tsr_rt.h"

#include "tsr-alloc.h"     // tsr_lower
#include "tsr-kinds.h"
#include "tsr-lower.h"
#include "tsr-opl.h"
#include "tsr_isa.h"
#include "tsr_transport.h"

namespace tsr::rt {

inline int debug_level() {
    static const int lvl = [] {
        const char * e = getenv("TSR_RT_DEBUG");
        return e ? atoi(e) : 0;
    }();
    return lvl;
}

#define TSR_RT_DBG(lvl, ...)                                \
    do {                                                    \
        if (::tsr::rt::debug_level() >= (lvl)) {            \
            fprintf(stderr, "[tsr_rt] " __VA_ARGS__);       \
        }                                                   \
    } while (0)

std::string strf(const char * fmt, ...) __attribute__((format(printf, 1, 2)));
inline double mib(uint64_t b) { return (double) b / (1024.0 * 1024.0); }

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
const char * make_plan(const tsr_config & cfg, dtype t, uint32_t N, uint32_t K, const weight_options & o, plan * p);

// f32 rows in the AUX region (const_upload): rows of n values, each padded to whole lines.
struct konst {
    uint32_t rows = 0, n = 0;
    uint64_t off = 0;          // ch0 byte offset
    uint64_t bytes = 0;
};

struct graph_obj;              // tsr_rt_graph.cpp

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
    tsr_range_alloc aux;              // AUX region (ch0): constants of the graphs
    tsr_opl_dev odev;                 // `graph` kernel + OPL cache over the OPL region (tsr-opl.h)
    tsr_kenv kenv;                    // what the record encoders check against (the device and the image's kinds)
    uint32_t pcap = 0;                // AT PCAP of the device (CFG3)

    std::unordered_map<weight_handle, weight> weights;
    weight_handle next_handle = 1;
    std::unordered_map<const_handle, konst> consts;
    const_handle next_const = 1;
    std::unordered_map<graph_handle, std::shared_ptr<graph_obj>> graphs;
    graph_handle next_graph = 1;

    std::vector<uint64_t> perf_last;  // PERF_CYCLES, PERF_MVM_CYCLES, PERF_CH_RD_BEATS[c]
    counters cnt;
    std::vector<uint8_t> hbuf;        // host staging (x packing, y gathering)

    ~impl();

    // -- transport -----------------------------------------------------------
    void write(uint32_t ch, uint64_t off, const void * src, size_t size, bool is_weight = false);
    void read(uint32_t ch, uint64_t off, void * dst, size_t size);
    uint64_t perf(uint32_t idx);
    void perf_snapshot(std::vector<uint64_t> & v);

    // -- op lists -------------------------------------------------------------
    // Run one op list blob of n_ops records through `graph` (OPL cache: a list seen before is not written again).
    // name_op(i): the text that names the failed record i in an error (the op that the record came from).
    void run_blob(const std::vector<uint8_t> & blob, uint32_t n_ops, uint32_t n_joins,
                  const std::function<std::string(int32_t)> & name_op);
    void run(const tsr_opl_builder & b);

    // IO layout of one linear call of tc tokens: x rows (K * 4 bytes each) at io_off, then per N
    // part i its y rows (y_lines(n_i) * 64 bytes each).
    static uint32_t y_lines(uint32_t n) { return (uint32_t) tsr_ceil_div_u64((uint64_t) n * 4, TSR_LINE_BYTES); }
    uint64_t y_off(const weight & w, uint32_t tc, uint32_t i) const;
    void run_linear(const weight & w, uint32_t tc, const float * scale);

    const weight & get(weight_handle h) const;
    void release_parts(weight & w);
};

}  // namespace tsr::rt
