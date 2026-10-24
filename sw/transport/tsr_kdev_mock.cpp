// tsr_kdev_mock.cpp — the "mock" kdev (tsr_kdev_mock.h): platform adapter register model
// around the ISA simulator. TSR_DEVICE=mock opens it with these environment variables (a
// malformed value makes the open fail):
//   TSR_MOCK_NUM_CH         channels, 1..16 (default 4)
//   TSR_MOCK_TMAX           T_MAX of the core, 1..8 (default 1)
//   TSR_MOCK_CH_GIB         channel buffer size in GiB, 1..16 (default 16, lazily mapped)
//   TSR_MOCK_DMA_ALIGN      alignment the raw DMA demands, power of two 1..4096 (default 1)
//   TSR_MOCK_DMA_CHUNK_KIB  largest raw DMA transfer in KiB (default 65536)
//   TSR_MOCK_DMA_BOUNCE     1: every transfer goes through the bounce buffer (default 0)
//   TSR_MOCK_CTRL           regs (default: regs_start / regs_wait) or kds (the XRT scheduler's
//                           sequence, kds_start / kds_wait: the path of xrt::run)
//   TSR_MOCK_INJECT         fault injection for tool tests: comma list of N:V, OR V (a non-zero
//                           ADAPTER_STATUS value, decimal or 0x hex) into ADAPTER_STATUS when the
//                           N-th run ends; N = 0: a run before the open (left by an earlier process)
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
    last_bases.assign(o.num_ch, 0);
}

mock_kdev::~mock_kdev() { tsr_sim_destroy(sim_); }

bool mock_kdev::is_ctl_reg(uint32_t off) const {
    return off < KREG_CH_BASE0 + 8 * o_.num_ch && off < KREG_ADAPTER_STATUS;
}

int mock_kdev::reg_read(uint32_t off, uint32_t * val) {
    if (off >= KREG_SPACE || (off & 3u) != 0) {
        fprintf(stderr, "tsr_transport(mock): register read at 0x%x outside the s_axi_control space\n", off);
        return -1;
    }
    if (is_ctl_reg(off) && in_ctl_ == 0) ctl_access_outside++;
    if (running_ && run_hang_ == mock_hang::reads && reads_left_ > 0 && --reads_left_ == 0) {
        tsr_sim_run(sim_);                   // the long run ends by itself (before this read)
        core_finished();
    }
    uint32_t v = 0;
    if (off >= KREG_CSR_WINDOW) {
        const uint32_t c = off - KREG_CSR_WINDOW;
        window_reads++;
        v = (c == TSR_CSR_ID && id_override != 0) ? id_override : tsr_sim_csr_read(sim_, c);
    } else if (off == KREG_AP_CTRL) {
        // {ap_ready, ap_idle, ap_done, ap_start} as tsr_xrt_ctrl; ap_done / ap_ready clear on read
        // (accepting a start leaves them alone). done_after_polls: the run looks unfinished for
        // that many reads after it ended.
        if (done_ && polls_left_ > 0) {
            polls_left_--;
            v = 0;
        } else {
            v = (done_ ? AP_DONE : 0u) | (running_ ? 0u : AP_IDLE) | (ready_ ? AP_READY : 0u);
            done_ = ready_ = false;
        }
    } else if (off == KREG_GIE) {
        v = gie_;
    } else if (off == KREG_IER) {
        v = ier_;
    } else if (off == KREG_ISR) {
        v = isr_;
    } else if (off == KREG_CMD_ADDR_LO) {
        v = cmd_lo_;
    } else if (off == KREG_CMD_ADDR_HI) {
        v = cmd_hi_;
    } else if (off == KREG_CMD_COUNT) {
        v = count_;
    } else if (off >= KREG_ARG0 && off < KREG_ARG0 + 32) {
        v = args_[(off - KREG_ARG0) / 4];
    } else if (off >= KREG_CH_BASE0 && off < KREG_CH_BASE0 + 8 * o_.num_ch) {
        v = base_[(off - KREG_CH_BASE0) / 4];
    } else if (off == KREG_ADAPTER_STATUS) {
        v = ast_;
    } else if (off == KREG_ADAPTER_VERSION) {
        v = o_.adapter_version != 0 ? o_.adapter_version : (0x00010000u | (o_.num_ch & 0xFFu));
    }
    *val = v;
    return 0;
}

int mock_kdev::reg_write(uint32_t off, uint32_t val) {
    if (off >= KREG_SPACE || (off & 3u) != 0) {
        fprintf(stderr, "tsr_transport(mock): register write at 0x%x outside the s_axi_control space\n", off);
        return -1;
    }
    if (is_ctl_reg(off) && in_ctl_ == 0) ctl_access_outside++;
    if (off >= KREG_CSR_WINDOW) {
        const uint32_t c = off - KREG_CSR_WINDOW;
        window_writes++;
        window_write_log.push_back(c);
        tsr_sim_csr_write(sim_, c, val);
        if (c == TSR_CSR_CTRL) {
            if (running_ && (run_hang_ == mock_hang::until_abort || run_hang_ == mock_hang::reads) &&
                (val & (TSR_CTRL_ABORT | TSR_CTRL_SOFT_RESET))) {
                tsr_sim_run(sim_);
                core_finished();
            } else if (!running_ && (val & TSR_CTRL_START)) {
                tsr_sim_run(sim_);           // a debug start through the window: no ap_done
            }
        }
        return 0;
    }
    switch (off) {
    case KREG_AP_CTRL:
        if ((val & AP_START) && !running_) accept_start();
        return 0;                            // auto_restart (bit 7) and the read-only bits: ignored
    case KREG_GIE:          gie_ = val & 1u; return 0;
    case KREG_IER:          ier_ = val & 1u; return 0;    // bit1 (ap_ready interrupt) not supported
    case KREG_ISR:          isr_ ^= val & 1u; return 0;
    case KREG_CMD_ADDR_LO:  cmd_lo_ = val; return 0;
    case KREG_CMD_ADDR_HI:  cmd_hi_ = val; return 0;
    case KREG_CMD_COUNT:    count_ = val; return 0;
    case KREG_ADAPTER_STATUS:                        // W1C: bits [3:0] one by one, the count as a whole
        ast_ &= ~(val & 0xFu);
        if (val & 0xFFFF0000u) ast_ &= 0xFFFFu;
        return 0;
    default: break;
    }
    if (off >= KREG_ARG0 && off < KREG_ARG0 + 32) {
        args_[(off - KREG_ARG0) / 4] = val;
    } else if (off >= KREG_CH_BASE0 && off < KREG_CH_BASE0 + 8 * o_.num_ch) {
        base_[(off - KREG_CH_BASE0) / 4] = val;
    }
    return 0;
}

void mock_kdev::accept_start() {
    starts++;
    last_cmd_lo = cmd_lo_;
    last_cmd_hi = cmd_hi_;
    last_count = count_;
    memcpy(last_args, args_, sizeof(args_));
    bool misaligned = false;
    for (uint32_t c = 0; c < o_.num_ch; ++c) {
        const uint64_t b = ((uint64_t) base_[2 * c + 1] << 32) | base_[2 * c];
        last_bases[c] = b;
        misaligned |= (b & 0xFFFu) != 0;
        if (b != addr_[c] && (b & 0xFFFu) == 0) {   // the core would reach some other memory: DECERR
            ast_ |= AST_RRESP;
            if ((ast_ >> 16) != 0xFFFFu) ast_ += 1u << 16;
        }
    }
    if (misaligned) {                        // like tsr_xrt_ctrl: flag it, complete at once, core untouched
        ast_ |= AST_BASE_ALIGN;
        running_ = false;
        done_ = ready_ = true;
        polls_left_ = done_after_polls;
        if (ier_ & 1u) isr_ |= 1u;
        run_ended();
        return;
    }
    tsr_sim_csr_write(sim_, TSR_CSR_CMD_ADDR_LO, cmd_lo_);
    tsr_sim_csr_write(sim_, TSR_CSR_CMD_ADDR_HI, cmd_hi_);
    tsr_sim_csr_write(sim_, TSR_CSR_CMD_COUNT, count_);
    for (uint32_t i = 0; i < 8; ++i) tsr_sim_csr_write(sim_, TSR_CSR_ARG0 + 4 * i, args_[i]);
    tsr_sim_csr_write(sim_, TSR_CSR_STATUS, TSR_STATUS_DONE | TSR_STATUS_ERR);
    tsr_sim_csr_write(sim_, TSR_CSR_CTRL, TSR_CTRL_START);
    running_ = true;                         // ap_done / ap_ready unchanged: only a read clears them
    run_hang_ = hang;
    reads_left_ = hang_reads;
    if (run_hang_ == mock_hang::none || (run_hang_ == mock_hang::reads && reads_left_ == 0)) {
        tsr_sim_run(sim_);
        core_finished();
    }
}

void mock_kdev::core_finished() {
    running_ = false;
    done_ = ready_ = true;
    polls_left_ = done_after_polls;
    if (ier_ & 1u) isr_ |= 1u;
    run_ended();
}

void mock_kdev::run_ended() {
    runs_ended++;
    auto add = [this](uint32_t inj) {
        const uint32_t cnt = (ast_ >> 16) + (inj >> 16);
        ast_ = (ast_ & 0xFFFFu) | (inj & 0xFFFFu) | ((cnt > 0xFFFFu ? 0xFFFFu : cnt) << 16);
    };
    if (inject_status != 0) {
        add(inject_status);
        inject_status = 0;
    }
    for (const auto & e : inject_at) {
        if (e.first == runs_ended) add(e.second);
    }
}

int mock_kdev::start(const kargs & a) {
    in_ctl_++;
    const int r = o_.ctrl_kds ? kds_start(*this, a) : regs_start(*this, a);
    in_ctl_--;
    return r;
}

kwait mock_kdev::wait(uint32_t timeout_ms) {
    in_ctl_++;
    const kwait w = o_.ctrl_kds ? kds_wait(*this, timeout_ms) : regs_wait(*this, timeout_ms);
    in_ctl_--;
    return w;
}

int mock_kdev::settle(uint32_t * ap) {
    in_ctl_++;
    settles++;
    const int r = reg_read(KREG_AP_CTRL, ap);
    in_ctl_--;
    return r;
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
    if (const char * c = getenv("TSR_MOCK_CTRL"); c != nullptr && c[0] != '\0') {
        if (strcmp(c, "kds") == 0) {
            o.ctrl_kds = true;
        } else if (strcmp(c, "regs") != 0) {
            fprintf(stderr, "tsr_transport(mock): TSR_MOCK_CTRL=\"%s\": must be regs or kds\n", c);
            return nullptr;
        }
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
    for (const auto & e : inj) {
        if (e.first != 0) {
            m->inject_at.push_back(e);
            continue;
        }
        // N = 0: an earlier process's run left the error (its ap_done is consumed, like that
        // process would have)
        m->inject_status = e.second;
        kargs a;
        if (m->start(a) != 0 || m->wait(1000) != kwait::done) {
            fprintf(stderr, "tsr_transport(mock): TSR_MOCK_INJECT: the run before the open failed\n");
            return nullptr;
        }
        m->runs_ended = 0;
        m->starts = 0;
    }
    return m;
}

} // namespace tsr
