// tsr_kdev_mock.cpp — the "mock" kdev (tsr_kdev_mock.h): platform adapter register model
// around the ISA simulator. TSR_DEVICE=mock opens it with these environment variables (a
// malformed value makes the open fail):
//   TSR_MOCK_NUM_CH         channels, 1..16 (default 4)
//   TSR_MOCK_TMAX           T_MAX of the core, 1..8 (default 1)
//   TSR_MOCK_CH_GIB         channel buffer size in GiB, 1..16 (default 16, lazily mapped)
//   TSR_MOCK_DMA_ALIGN      alignment the raw DMA demands, power of two 1..4096 (default 1)
//   TSR_MOCK_DMA_CHUNK_KIB  largest raw DMA transfer in KiB (default 65536)
//   TSR_MOCK_DMA_BOUNCE     1: every transfer goes through the bounce buffer (default 0)
//   TSR_MOCK_INJECT         fault injection for tool tests: comma list of N:V, OR V (a non-zero
//                           ADAPTER_STATUS value, decimal or 0x hex) into ADAPTER_STATUS. Only
//                           N = 0 exists until the v2 launch path (P3.1): an error left by an
//                           earlier process, present at open
//   TSR_SIM_THREADS         simulator worker threads (default min(online CPUs, 8))

#include "tsr_kdev_mock.h"

#include <cerrno>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

namespace tsr {

namespace {

// Device addresses of the channel buffers: above 4 GiB, 4 KiB aligned but not bank aligned,
// so nothing can silently depend on a zero or bank-aligned base.
uint64_t mock_base(uint32_t ch) { return 0x4000000000ull + (uint64_t) ch * 0x1000000000ull + 0x1000ull * (ch + 1); }

int env_u64(const char * name, uint64_t lo, uint64_t hi, uint64_t * out) {
    const char * v = getenv(name);
    if (v == nullptr || v[0] == '\0') return 0;
    char * end = nullptr;
    errno = 0;
    const unsigned long long r = strtoull(v, &end, 10);
    if (errno != 0 || end == v || *end != '\0' || v[0] == '-' || r < lo || r > hi) {
        fprintf(stderr, "tsr_transport(mock): %s=\"%s\" is not an integer in [%" PRIu64 ", %" PRIu64 "]\n", name, v, lo, hi);
        return -1;
    }
    *out = (uint64_t) r;
    return 1;
}

} // namespace

mock_kdev::mock_kdev(const mock_options & o) : o_(o), dma_(*this, o.dma) {
    tsr_config cfg = TSR_CONFIG_U200_V1;
    cfg.num_ch = o.num_ch;
    cfg.t_max = o.t_max;
    cfg.ch_bytes = o.ch_bytes;
    sim_ = tsr_sim_create(&cfg, nullptr);
    if (sim_ == nullptr) return;
    const long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
    tsr_sim_set_threads(sim_, ncpu < 1 ? 1u : (ncpu > 8 ? 8u : (unsigned) ncpu));
    addr_.resize(o.num_ch);
    for (uint32_t c = 0; c < o.num_ch; ++c) addr_[c] = mock_base(c);
    base_.assign(2 * (size_t) o.num_ch, 0);
}

mock_kdev::~mock_kdev() { tsr_sim_destroy(sim_); }

int mock_kdev::reg_read(uint32_t off, uint32_t * val) {
    if (off >= KREG_SPACE || (off & 3u) != 0) {
        fprintf(stderr, "tsr_transport(mock): register read at 0x%x outside the s_axi_control space\n", off);
        return -1;
    }
    uint32_t v = 0;
    if (off >= KREG_CSR_WINDOW) {
        const uint32_t c = off - KREG_CSR_WINDOW;
        window_reads++;
        v = (c == TSR_CSR_ID && id_override != 0) ? id_override : tsr_sim_csr_read(sim_, c);
    } else if (off >= KREG_CH_BASE0 && off < KREG_CH_BASE0 + 8 * o_.num_ch) {
        v = base_[(off - KREG_CH_BASE0) / 4];
    } else if (off == KREG_ADAPTER_STATUS) {
        v = ast_;
    } else if (off == KREG_ADAPTER_VERSION) {
        v = o_.adapter_version != 0 ? o_.adapter_version : (0x00010000u | (o_.num_ch & 0xFFu));
    } else {
        unmapped_accesses++;               // reads 0 (v1's AP_CTRL .. arg7 included)
    }
    *val = v;
    return 0;
}

int mock_kdev::reg_write(uint32_t off, uint32_t val) {
    if (off >= KREG_SPACE || (off & 3u) != 0) {
        fprintf(stderr, "tsr_transport(mock): register write at 0x%x outside the s_axi_control space\n", off);
        return -1;
    }
    if (off >= KREG_CSR_WINDOW) {
        const uint32_t c = off - KREG_CSR_WINDOW;
        window_writes++;
        window_write_log.push_back(c);
        tsr_sim_csr_write(sim_, c, val);
        return 0;
    }
    if (off >= KREG_CH_BASE0 && off < KREG_CH_BASE0 + 8 * o_.num_ch) {
        base_[(off - KREG_CH_BASE0) / 4] = val;
    } else if (off == KREG_ADAPTER_STATUS) {   // W1C: bits [3:0] one by one, the count as a whole
        ast_ &= ~(val & 0xFu);
        if (val & 0xFFFF0000u) ast_ &= 0xFFFFu;
    } else {
        unmapped_accesses++;               // ignored
    }
    return 0;
}

void mock_kdev::inject_status(uint32_t v) {
    const uint32_t cnt = (ast_ >> 16) + (v >> 16);
    ast_ = (ast_ & 0xFFFFu) | (v & 0xFFFFu) | ((cnt > 0xFFFFu ? 0xFFFFu : cnt) << 16);
}

int mock_kdev::mem_write(uint32_t ch, uint64_t off, const void * src, size_t n) { return dma_.write(ch, off, src, n); }
int mock_kdev::mem_read(uint32_t ch, uint64_t off, void * dst, size_t n) { return dma_.read(ch, off, dst, n); }

int mock_kdev::raw_check(const char * what, uint32_t ch, uint64_t off, const void * p, size_t n) const {
    const uint64_t A = o_.dma.align;
    if (ch >= o_.num_ch || off > o_.ch_bytes || n > o_.ch_bytes - off || n == 0 || n > o_.dma.max_xfer || off % A != 0 ||
        n % A != 0 || (uintptr_t) p % A != 0) {
        fprintf(stderr, "tsr_transport(mock): raw DMA %s violates the rules: ch %u, offset 0x%" PRIx64 ", %zu bytes, host %p "
                        "(align %zu, max %zu, ch_bytes %" PRIu64 ")\n",
                what, ch, off, n, p, o_.dma.align, o_.dma.max_xfer, o_.ch_bytes);
        return -1;
    }
    return 0;
}

int mock_kdev::raw_write(uint32_t ch, uint64_t off, const void * src, size_t n) {
    if (raw_check("write", ch, off, src, n)) return -1;
    memcpy(tsr_sim_dram(sim_, ch) + off, src, n);
    return 0;
}

int mock_kdev::raw_read(uint32_t ch, uint64_t off, void * dst, size_t n) {
    if (raw_check("read", ch, off, dst, n)) return -1;
    memcpy(dst, tsr_sim_dram(sim_, ch) + off, n);
    return 0;
}

std::unique_ptr<kdev> open_mock_kdev() {
    mock_options o;
    uint64_t v = 0;
    int r;
    if ((r = env_u64("TSR_MOCK_NUM_CH", 1, TSR_NUM_CH_MAX, &v)) < 0) return nullptr;
    if (r) o.num_ch = (uint32_t) v;
    if ((r = env_u64("TSR_MOCK_TMAX", 1, TSR_T_MAX_ARCH, &v)) < 0) return nullptr;
    if (r) o.t_max = (uint32_t) v;
    if ((r = env_u64("TSR_MOCK_CH_GIB", 1, 16, &v)) < 0) return nullptr;
    if (r) o.ch_bytes = v << 30;
    if ((r = env_u64("TSR_MOCK_DMA_ALIGN", 1, 4096, &v)) < 0) return nullptr;
    if (r) o.dma.align = (size_t) v;
    if ((r = env_u64("TSR_MOCK_DMA_CHUNK_KIB", 1, 1u << 20, &v)) < 0) return nullptr;
    if (r) o.dma.max_xfer = (size_t) v << 10;
    if ((r = env_u64("TSR_MOCK_DMA_BOUNCE", 0, 1, &v)) < 0) return nullptr;
    if (r) o.dma.always_bounce = v != 0;
    const std::string bad = dma_rules_check(o.dma);
    if (!bad.empty()) {
        fprintf(stderr, "tsr_transport(mock): TSR_MOCK_DMA_*: %s\n", bad.c_str());
        return nullptr;
    }
    std::vector<std::pair<uint64_t, uint32_t>> inj;   // TSR_MOCK_INJECT=N:V[,N:V...]
    if (const char * e = getenv("TSR_MOCK_INJECT"); e != nullptr && e[0] != '\0') {
        for (const char * p = e;;) {
            char * end = nullptr;
            errno = 0;
            const unsigned long long n = strtoull(p, &end, 10);
            bool good = errno == 0 && end != p && *end == ':' && p[0] != '-';
            unsigned long long val = 0;
            if (good) {
                p = end + 1;
                errno = 0;
                val = strtoull(p, &end, 0);
                good = errno == 0 && end != p && p[0] != '-' && val != 0 && val <= 0xFFFFFFFFull && (*end == ',' || *end == '\0');
            }
            if (!good) {
                fprintf(stderr, "tsr_transport(mock): TSR_MOCK_INJECT=\"%s\": expected N:V[,N:V...] (V a non-zero 32-bit value)\n", e);
                return nullptr;
            }
            if (n != 0) {
                fprintf(stderr, "tsr_transport(mock): TSR_MOCK_INJECT=\"%s\": N = %llu: there are no runs to inject into until "
                                "the v2 launch path; only N = 0 (left by an earlier process) exists\n", e, n);
                return nullptr;
            }
            inj.emplace_back((uint64_t) n, (uint32_t) val);
            if (*end == '\0') break;
            p = end + 1;
        }
    }
    std::unique_ptr<mock_kdev> m(new mock_kdev(o));
    if (!m->ok()) {
        fprintf(stderr, "tsr_transport(mock): simulator creation failed\n");
        return nullptr;
    }
    for (const auto & e : inj) m->inject_status(e.second);   // N = 0: left by an earlier process
    return m;
}

} // namespace tsr
