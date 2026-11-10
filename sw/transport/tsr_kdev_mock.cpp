// tsr_kdev_mock.cpp — the "mock" kdev (tsr_kdev_mock.h): the adapter register model around the
// sim kdev. TSR_DEVICE=mock opens it with these environment variables (a malformed value makes
// the open fail):
//   TSR_MOCK_NUM_CH         channels, 1..16 (default 4)
//   TSR_MOCK_TMAX           T_MAX of the core, 1..8 (default 1)
//   TSR_MOCK_CH_GIB         channel buffer size in GiB, 1..16 (default 16, lazily mapped)
//   TSR_MOCK_DMA_ALIGN      alignment the raw DMA demands, power of two 1..4096 (default 1)
//   TSR_MOCK_DMA_CHUNK_KIB  largest raw DMA transfer in KiB (default 65536)
//   TSR_MOCK_DMA_BOUNCE     1: every transfer goes through the bounce buffer (default 0)
//   TSR_MOCK_INJECT         adapter errors: comma list of N:V, OR V (a non-zero ADAPTER_STATUS
//                           value, decimal or 0x hex) into ADAPTER_STATUS: N = 0 at open (left by
//                           an earlier process), N >= 1 at the N-th LQ_PUSH (an AXI error during
//                           that launch: plat_err_i rises, EX §9.3)
//   TSR_MOCK_TCM_FAULT      ADDR:XOR (0x hex allowed): after every LOAD_DATA write to the TCM word
//                           at ADDR the word is XORed with XOR (a TCM write fault; the receive CRC
//                           still matches, VERIFY does not, EX §3.3)
//   TSR_MOCK_HANG           N: the MVM unit hangs at the N-th LQ_PUSH (STATUS.UNIT_HUNG,
//                           Q_STATE[MVM].HUNG; the model stops; SOFT_RESET never completes)
//   TSR_SIM_THREADS         simulator worker threads (default min(online CPUs, 8)); the other
//                           TSR_SIM_* variables the simulator reads itself (tsr_kdev_sim.h)

#include "tsr_kdev_mock.h"

#include "tsr_sim_int.h"   // tsr_sim_itcm / tsr_sim_dtcm (the TCM fault)

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

constexpr uint32_t AST_W1C_BITS = 0x3Fu;   // bits 0-5 (EX §9.2)

} // namespace

mock_kdev::mock_kdev(const mock_options & o) : o_(o), dma_(*this, o.dma) {
    sim_options so;
    so.cfg = TSR_CONFIG_U200;
    so.cfg.caps = o.no_vu ? 0u : tsr_sim_caps_implemented();
    so.no_vu = o.no_vu;
    so.cfg.num_ch = o.num_ch;
    so.cfg.t_max = o.t_max;
    so.cfg.ch_bytes = o.ch_bytes;
    so.threads = o.threads;
    core_.reset(new sim_kdev(so));
    if (!core_->ok()) return;
    core_->may_advance = [this] { return may_advance(); };
    addr_.resize(o.num_ch);
    for (uint32_t c = 0; c < o.num_ch; ++c) addr_[c] = mock_base(c);
    base_.assign(2 * (size_t) o.num_ch, 0);
    latched_.assign(o.num_ch, 0);
}

mock_kdev::~mock_kdev() = default;

std::string mock_kdev::describe() const {
    char b[160];
    snprintf(b, sizeof(b), "mock adapter (ADAPTER_VERSION 0x%08x) around the ", o_.adapter_version ? o_.adapter_version :
             (TSR_PUT(TSR_ADAPTER_VERSION_MAJOR, TSR_ADAPTER_MAJOR) | (o_.num_ch & 0xFFu)));
    return b + core_->describe();
}

bool mock_kdev::may_advance() {
    if (hung_ || bind_errors) return false;
    if (!mem_en_ && !(tsr_sim_csr_read(core_->sim(), TSR_CSR_STATUS) & TSR_STATUS_QIDLE)) {
        ast_ |= AST_REQ_HELD;   // an op would issue memory requests: held (EX §9.2)
        return false;
    }
    return true;
}

void mock_kdev::plat_err_update() { tsr_sim_set_plat_err(core_->sim(), adapter_plat_err(ast_) ? 1 : 0); }

void mock_kdev::inject_status(uint32_t v) {
    const uint32_t cnt = (ast_ >> 16) + (v >> 16);
    ast_ = (ast_ & 0xFFFFu) | (v & 0xFFFFu) | ((cnt > 0xFFFFu ? 0xFFFFu : cnt) << 16);
    plat_err_update();
}

uint32_t mock_kdev::window_read(uint32_t c) {
    if (c == TSR_CSR_ID && id_override != 0) return id_override;
    if (c == TSR_CSR_VERSION && version_override != 0) return version_override;
    if (c == TSR_CSR_CTL_ID && ctl_id_override != UINT32_MAX) return ctl_id_override;
    if (c == TSR_CSR_RV_FREQ && rv_freq_override != UINT32_MAX) return rv_freq_override;
    uint32_t v = core_->csr_read(c);
    if (c == TSR_CSR_STATUS) {
        if (adapter_plat_err(ast_)) v |= TSR_STATUS_PLAT_ERR;
        if (hung_) v |= TSR_STATUS_UNIT_HUNG;
        if ((hung_ || sr_stuck) && sr_busy_) v |= TSR_STATUS_SR_BUSY;
    } else if (c == TSR_CSR_Q_STATE(TSR_UNIT_MVM) && hung_) {
        v |= TSR_Q_STATE_HUNG | TSR_Q_STATE_RUNNING;
    }
    return v;
}

int mock_kdev::reg_read(uint32_t off, uint32_t * val) {
    if (off >= KREG_SPACE || (off & 3u) != 0) {
        fprintf(stderr, "tsr_transport(mock): register read at 0x%x outside the s_axi_control space\n", off);
        return -1;
    }
    uint32_t v = 0;
    if (off >= KREG_CSR_WINDOW) {
        window_reads++;
        v = window_read(off - KREG_CSR_WINDOW);
    } else if (off >= KREG_CH_BASE0 && off < KREG_CH_BASE0 + 8 * o_.num_ch) {
        v = base_[(off - KREG_CH_BASE0) / 4];
    } else if (off == KREG_ADAPTER_CTRL) {
        v = (mem_en_ ? TSR_ADAPTER_CTRL_MEM_EN : 0u) | (dual_dis_ ? TSR_ADAPTER_CTRL_DUAL_DIS : 0u);
    } else if (off == KREG_ADAPTER_STATUS) {
        v = ast_;
    } else if (off == KREG_ADAPTER_VERSION) {
        v = o_.adapter_version != 0 ? o_.adapter_version : (TSR_PUT(TSR_ADAPTER_VERSION_MAJOR, TSR_ADAPTER_MAJOR) | (o_.num_ch & 0xFFu));
    } else {
        unmapped_accesses++;               // reads 0
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
        if (c == TSR_CSR_CTRL && (val & TSR_CTRL_SOFT_RESET) && (hung_ || sr_stuck)) {
            sr_busy_ = true;               // EX §8.6: SOFT_RESET step 2 never ends with a hung unit
            return 0;
        }
        core_->csr_write(c, val);
        if (c == TSR_CSR_LOAD_ADDR) {
            load_addr_ = val;
        } else if (c == TSR_CSR_LOAD_DATA) {
            if (o_.tcm_fault_xor != 0 && load_addr_ == o_.tcm_fault_addr) {
                uint32_t ib = 0, db = 0;
                uint8_t * itcm = tsr_sim_itcm(core_->sim(), &ib);
                uint8_t * dtcm = tsr_sim_dtcm(core_->sim(), &db);
                uint8_t * w = nullptr;
                if (load_addr_ + 4 <= ib) w = itcm + load_addr_;
                else if (load_addr_ >= TSR_RV_DTCM_BASE && load_addr_ - TSR_RV_DTCM_BASE + 4 <= db) w = dtcm + (load_addr_ - TSR_RV_DTCM_BASE);
                if (w != nullptr) {
                    for (int k = 0; k < 4; ++k) w[k] ^= (uint8_t) (o_.tcm_fault_xor >> (8 * k));
                }
            }
            load_addr_ += 4;
        } else if (c == TSR_CSR_LQ_PUSH) {
            pushes++;
            for (const auto & e : push_inj_) {
                if (e.first == pushes) inject_status(e.second);
            }
            if (o_.hang_at_push != 0 && pushes == o_.hang_at_push) hung_ = true;
        }
        return 0;
    }
    if (off >= KREG_CH_BASE0 && off < KREG_CH_BASE0 + 8 * o_.num_ch) {
        if (mem_en_) ast_ |= AST_BASE_LOCKED;                    // ignored while MEM_EN = 1
        else base_[(off - KREG_CH_BASE0) / 4] = val;
    } else if (off == KREG_ADAPTER_CTRL) {
        const bool en = (val & TSR_ADAPTER_CTRL_MEM_EN) != 0;
        dual_dis_ = (val & TSR_ADAPTER_CTRL_DUAL_DIS) != 0;       // the run-time enable of the dual weight stream
        if (en && !mem_en_) {                                      // rising edge: check and latch
            bool aligned = true;
            for (uint32_t c = 0; c < o_.num_ch; ++c) aligned = aligned && (base_[2 * c] & 0xFFFu) == 0;
            if (!aligned) {
                ast_ |= AST_BASE_ALIGN;
            } else {
                mem_en_ = true;
                for (uint32_t c = 0; c < o_.num_ch; ++c) {
                    latched_[c] = (uint64_t) base_[2 * c + 1] << 32 | base_[2 * c];
                    if (latched_[c] != addr_[c]) {
                        bind_errors++;
                        fprintf(stderr, "tsr_transport(mock): ch%u latched base 0x%" PRIx64 " but its buffer is at 0x%" PRIx64
                                        ": the core would address the wrong memory (the model stops)\n", c, latched_[c], addr_[c]);
                    }
                }
            }
        } else if (!en) {
            mem_en_ = false;
        }
    } else if (off == KREG_ADAPTER_STATUS) {                       // W1C: bits [5:0] one by one, the count as a whole
        ast_ &= ~(val & AST_W1C_BITS);
        if (val & 0xFFFF0000u) ast_ &= 0xFFFFu;
        plat_err_update();
    } else {
        unmapped_accesses++;               // ignored
    }
    return 0;
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
    return core_->mem_write(ch, off, src, n);
}

int mock_kdev::raw_read(uint32_t ch, uint64_t off, void * dst, size_t n) {
    if (raw_check("read", ch, off, dst, n)) return -1;
    return core_->mem_read(ch, off, dst, n);
}

namespace {

// "N:V[,N:V...]" or "A:X" pairs of unsigned values (base 10 for N when dec_first, else base 0).
bool parse_pairs(const char * name, const char * e, bool dec_first, std::vector<std::pair<uint64_t, uint64_t>> * out) {
    for (const char * p = e;;) {
        char * end = nullptr;
        errno = 0;
        const unsigned long long a = strtoull(p, &end, dec_first ? 10 : 0);
        bool good = errno == 0 && end != p && *end == ':' && p[0] != '-';
        unsigned long long b = 0;
        if (good) {
            p = end + 1;
            errno = 0;
            b = strtoull(p, &end, 0);
            good = errno == 0 && end != p && p[0] != '-' && b != 0 && b <= 0xFFFFFFFFull && (*end == ',' || *end == '\0');
        }
        if (!good) {
            fprintf(stderr, "tsr_transport(mock): %s=\"%s\": expected A:B[,A:B...] (B a non-zero 32-bit value)\n", name, e);
            return false;
        }
        out->emplace_back((uint64_t) a, (uint64_t) b);
        if (*end == '\0') return true;
        p = end + 1;
    }
}

} // namespace

std::unique_ptr<kdev> open_mock_kdev() {
    mock_options o;
    uint64_t v = 0;
    int r;
    if ((r = env_u64("mock", "TSR_MOCK_NUM_CH", 1, TSR_NUM_CH_MAX, &v)) < 0) return nullptr;
    if (r) o.num_ch = (uint32_t) v;
    if ((r = env_u64("mock", "TSR_MOCK_TMAX", 1, TSR_T_MAX_ARCH, &v)) < 0) return nullptr;
    if (r) o.t_max = (uint32_t) v;
    if ((r = env_u64("mock", "TSR_MOCK_CH_GIB", 1, 16, &v)) < 0) return nullptr;
    if (r) o.ch_bytes = v << 30;
    if ((r = env_u64("mock", "TSR_MOCK_DMA_ALIGN", 1, 4096, &v)) < 0) return nullptr;
    if (r) o.dma.align = (size_t) v;
    if ((r = env_u64("mock", "TSR_MOCK_DMA_CHUNK_KIB", 1, 1u << 20, &v)) < 0) return nullptr;
    if (r) o.dma.max_xfer = (size_t) v << 10;
    if ((r = env_u64("mock", "TSR_MOCK_DMA_BOUNCE", 0, 1, &v)) < 0) return nullptr;
    if (r) o.dma.always_bounce = v != 0;
    const long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
    o.threads = ncpu < 1 ? 1u : (ncpu > 8 ? 8u : (unsigned) ncpu);
    if ((r = env_u64("mock", "TSR_SIM_THREADS", 1, 64, &v)) < 0) return nullptr;
    if (r) o.threads = (unsigned) v;
    if ((r = env_u64("mock", "TSR_SIM_VU", 0, 1, &v)) < 0) return nullptr;
    if (r) o.no_vu = v == 0;
    if ((r = env_u64("mock", "TSR_MOCK_HANG", 1, UINT32_MAX, &v)) < 0) return nullptr;
    if (r) o.hang_at_push = (uint32_t) v;
    const std::string bad = dma_rules_check(o.dma);
    if (!bad.empty()) {
        fprintf(stderr, "tsr_transport(mock): TSR_MOCK_DMA_*: %s\n", bad.c_str());
        return nullptr;
    }
    std::vector<std::pair<uint64_t, uint64_t>> inj, tcm;
    if (const char * e = getenv("TSR_MOCK_INJECT"); e != nullptr && e[0] != '\0') {
        if (!parse_pairs("TSR_MOCK_INJECT", e, true, &inj)) return nullptr;
    }
    if (const char * e = getenv("TSR_MOCK_TCM_FAULT"); e != nullptr && e[0] != '\0') {
        if (!parse_pairs("TSR_MOCK_TCM_FAULT", e, false, &tcm)) return nullptr;
        if (tcm.size() != 1 || (tcm[0].first & 3u) != 0 || tcm[0].first > 0xFFFFFFFFull) {
            fprintf(stderr, "tsr_transport(mock): TSR_MOCK_TCM_FAULT=\"%s\": expected one word-aligned ADDR:XOR\n", e);
            return nullptr;
        }
        o.tcm_fault_addr = (uint32_t) tcm[0].first;
        o.tcm_fault_xor = (uint32_t) tcm[0].second;
    }
    std::unique_ptr<mock_kdev> m(new mock_kdev(o));
    if (!m->ok()) {
        fprintf(stderr, "tsr_transport(mock): simulator creation failed\n");
        return nullptr;
    }
    for (const auto & e : inj) {
        if (e.first == 0) m->inject_status((uint32_t) e.second);   // left by an earlier process
        else m->inject_at_push((uint32_t) e.first, (uint32_t) e.second);
    }
    return m;
}

} // namespace tsr
