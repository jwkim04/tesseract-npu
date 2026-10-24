// tsr_kdev.cpp — the kernel-protocol transport on top of a kdev, the register-level
// ap_ctrl_hs helpers, the chunked / aligned transfer engine and the kdev registry
// (tsr_kdev.h, docs/xrt-host.md).

#include "tsr_kdev.h"
#include "tsr_transport_impl.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <thread>

namespace tsr {

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------
std::string adapter_status_str(uint32_t v) {
    if (v == 0) return "ok";
    std::string s;
    auto add = [&s](const char * w) {
        if (!s.empty()) s += ", ";
        s += w;
    };
    if (v & AST_RRESP)      add("RRESP error (read response not OKAY)");
    if (v & AST_BRESP)      add("BRESP error (write response not OKAY)");
    if (v & AST_ORPHAN)     add("response with no outstanding request for its id");
    if (v & AST_BASE_ALIGN) add("ch<c>_base not 4 KiB aligned at ap_start");
    if (v & 0xFFF0u)        add("unknown bits");
    char buf[64];
    snprintf(buf, sizeof(buf), "%u error response%s%s", adapter_status_count(v),
             adapter_status_count(v) == 1 ? "" : "s", adapter_status_count(v) == 0xFFFFu ? " (saturated)" : "");
    add(buf);
    return s;
}

// Returns 0 if unset, 1 if set and valid (*out), -1 if malformed (message printed).
static int env_u64(const char * name, uint64_t lo, uint64_t hi, uint64_t * out) {
    const char * v = getenv(name);
    if (v == nullptr || v[0] == '\0') return 0;
    char * end = nullptr;
    errno = 0;
    const unsigned long long r = strtoull(v, &end, 10);
    if (errno != 0 || end == v || *end != '\0' || v[0] == '-' || r < lo || r > hi) {
        fprintf(stderr, "tsr_transport: %s=\"%s\" is not an integer in [%" PRIu64 ", %" PRIu64 "]\n", name, v, lo, hi);
        return -1;
    }
    *out = (uint64_t) r;
    return 1;
}

using clk = std::chrono::steady_clock;

static uint64_t ms_since(clk::time_point t0) {
    return (uint64_t) std::chrono::duration_cast<std::chrono::milliseconds>(clk::now() - t0).count();
}

// ---------------------------------------------------------------------------
// Register-level ap_ctrl_hs
// ---------------------------------------------------------------------------
int regs_start(kdev & d, const kargs & a) {
    uint32_t ap = 0;
    if (d.reg_read(KREG_AP_CTRL, &ap) != 0) return -1;   // also clears a stale ap_done / ap_ready
    if (!(ap & AP_IDLE) || (ap & AP_START)) {
        fprintf(stderr, "tsr_transport(%s): kernel not idle at start (AP_CTRL = 0x%08x)\n", d.name(), ap);
        return -1;
    }
    int rc = 0;
    rc |= d.reg_write(KREG_CMD_ADDR_LO, (uint32_t) a.cmd_addr);
    rc |= d.reg_write(KREG_CMD_ADDR_HI, (uint32_t) (a.cmd_addr >> 32));
    rc |= d.reg_write(KREG_CMD_COUNT, a.cmd_count);
    for (uint32_t i = 0; i < 8; ++i) rc |= d.reg_write(KREG_ARG0 + 4 * i, a.arg[i]);
    for (uint32_t c = 0; c < d.num_ch(); ++c) {
        const uint64_t base = d.ch_addr(c);
        rc |= d.reg_write(KREG_CH_BASE0 + 8 * c, (uint32_t) base);
        rc |= d.reg_write(KREG_CH_BASE0 + 8 * c + 4, (uint32_t) (base >> 32));
    }
    if (rc != 0) return -1;
    return d.reg_write(KREG_AP_CTRL, AP_START);
}

kwait regs_wait(kdev & d, uint32_t timeout_ms) {
    const clk::time_point t0 = clk::now();
    uint32_t sleep_us = 0;
    for (uint64_t polls = 0;; ++polls) {
        uint32_t ap = 0;
        if (d.reg_read(KREG_AP_CTRL, &ap) != 0) return kwait::error;
        if (ap & AP_DONE) return kwait::done;
        if (timeout_ms != 0 && ms_since(t0) >= timeout_ms) return kwait::timeout;
        // Spin briefly (short programs finish within a few register reads), then back off.
        if (polls >= 64) {
            sleep_us = sleep_us == 0 ? 1 : std::min<uint32_t>(sleep_us * 2, 100);
            std::this_thread::sleep_for(std::chrono::microseconds(sleep_us));
        }
    }
}

// The XRT scheduler's ap_ctrl_hs sequence (xocl cu_hls_configure / cu_hls_start /
// cu_hls_ctrl_hs_check; ERT start_cu / check_cu): the register map from 0x010 to the last
// argument, AP_CTRL = ap_start, no AP_CTRL read before the start; complete on the first read with
// ap_done (ap_idle and ap_start are not looked at).
int kds_start(kdev & d, const kargs & a) {
    int rc = 0;
    rc |= d.reg_write(KREG_CMD_ADDR_LO, (uint32_t) a.cmd_addr);
    rc |= d.reg_write(KREG_CMD_ADDR_HI, (uint32_t) (a.cmd_addr >> 32));
    rc |= d.reg_write(KREG_CMD_COUNT, a.cmd_count);
    rc |= d.reg_write(KREG_CMD_COUNT + 4, 0);
    for (uint32_t i = 0; i < 8; ++i) rc |= d.reg_write(KREG_ARG0 + 4 * i, a.arg[i]);
    for (uint32_t c = 0; c < d.num_ch(); ++c) {
        const uint64_t base = d.ch_addr(c);
        rc |= d.reg_write(KREG_CH_BASE0 + 8 * c, (uint32_t) base);
        rc |= d.reg_write(KREG_CH_BASE0 + 8 * c + 4, (uint32_t) (base >> 32));
    }
    if (rc != 0) return -1;
    return d.reg_write(KREG_AP_CTRL, AP_START);
}

kwait kds_wait(kdev & d, uint32_t timeout_ms) { return regs_wait(d, timeout_ms); }

// ---------------------------------------------------------------------------
// Chunked / aligned transfers
// ---------------------------------------------------------------------------
std::string dma_rules_check(const dma_rules & r) {
    if (r.align == 0 || r.align > 4096 || (r.align & (r.align - 1)) != 0) return "align must be a power of two in [1, 4096]";
    if (r.max_xfer == 0 || r.max_xfer % r.align != 0) return "max_xfer must be a positive multiple of align";
    return "";
}

dma_engine::dma_engine(dma_raw & raw, const dma_rules & rules) : raw_(raw), r_(rules) {}

dma_engine::~dma_engine() { free(buf_); }

uint8_t * dma_engine::bounce() {
    if (buf_ == nullptr) {
        void * p = nullptr;
        if (posix_memalign(&p, std::max<size_t>(r_.align, 64), r_.max_xfer) != 0) {
            fprintf(stderr, "tsr_transport: cannot allocate a %zu-byte DMA bounce buffer\n", r_.max_xfer);
            return nullptr;
        }
        buf_ = (uint8_t *) p;
    }
    return buf_;
}

int dma_engine::write(uint32_t ch, uint64_t off, const void * src, size_t n) {
    if (n == 0) return 0;
    const uint8_t * s = (const uint8_t *) src;
    const uint64_t A = r_.align;
    if (!r_.always_bounce && off % A == 0 && n % A == 0 && (uintptr_t) s % A == 0) {
        for (size_t done = 0; done < n;) {
            const size_t k = std::min(r_.max_xfer, n - done);
            raw_calls++;
            if (raw_.raw_write(ch, off + done, s + done, k) != 0) return -1;
            done += k;
        }
        return 0;
    }
    uint8_t * b = bounce();
    if (b == nullptr) return -1;
    const uint64_t a0 = off & ~(A - 1), a1 = (off + n + A - 1) & ~(A - 1), end = off + n;
    for (uint64_t p = a0; p < a1;) {
        const size_t k = (size_t) std::min<uint64_t>(r_.max_xfer, a1 - p);
        const uint64_t lo = std::max(p, off), hi = std::min(p + k, end);
        if (lo > p) {                       // head unit only partly written: read it first
            raw_calls++;
            rmw_reads++;
            if (raw_.raw_read(ch, p, b, A) != 0) return -1;
        }
        if (hi < p + k && !(lo > p && k == A)) {   // tail unit (unless the head read covered it)
            raw_calls++;
            rmw_reads++;
            if (raw_.raw_read(ch, p + k - A, b + k - A, A) != 0) return -1;
        }
        memcpy(b + (lo - p), s + (lo - off), (size_t) (hi - lo));
        bounce_copies++;
        raw_calls++;
        if (raw_.raw_write(ch, p, b, k) != 0) return -1;
        p += k;
    }
    return 0;
}

int dma_engine::read(uint32_t ch, uint64_t off, void * dst, size_t n) {
    if (n == 0) return 0;
    uint8_t * d = (uint8_t *) dst;
    const uint64_t A = r_.align;
    if (!r_.always_bounce && off % A == 0 && n % A == 0 && (uintptr_t) d % A == 0) {
        for (size_t done = 0; done < n;) {
            const size_t k = std::min(r_.max_xfer, n - done);
            raw_calls++;
            if (raw_.raw_read(ch, off + done, d + done, k) != 0) return -1;
            done += k;
        }
        return 0;
    }
    uint8_t * b = bounce();
    if (b == nullptr) return -1;
    const uint64_t a0 = off & ~(A - 1), a1 = (off + n + A - 1) & ~(A - 1), end = off + n;
    for (uint64_t p = a0; p < a1;) {
        const size_t k = (size_t) std::min<uint64_t>(r_.max_xfer, a1 - p);
        const uint64_t lo = std::max(p, off), hi = std::min(p + k, end);
        raw_calls++;
        if (raw_.raw_read(ch, p, b, k) != 0) return -1;
        memcpy(d + (lo - off), b + (lo - p), (size_t) (hi - lo));
        bounce_copies++;
        p += k;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// Protocol layer
// ---------------------------------------------------------------------------
namespace {

constexpr uint32_t ABORT_MS_DEFAULT   = 10000;
constexpr uint32_t TIMEOUT_MS_DEFAULT = 60000;

struct kproto {
    std::unique_ptr<kdev> dev;
    tsr_config cfg{};
    uint32_t   timeout_ms = TIMEOUT_MS_DEFAULT;
    uint32_t   abort_ms   = ABORT_MS_DEFAULT;
    bool       broken     = false;   // the core did not stop after ABORT: every exec fails
    adapter_log alog;

    const char * nm() const { return dev->name(); }

    // Core CSR through the window. On a failed access: message printed, *ok = false, 0xFFFFFFFF.
    uint32_t csr(uint32_t off, bool * ok = nullptr) {
        uint32_t v = 0xFFFFFFFFu;
        if (dev->reg_read(KREG_CSR_WINDOW + off, &v) != 0) {
            if (ok) *ok = false;
            return 0xFFFFFFFFu;
        }
        return v;
    }
    int csr_write(uint32_t off, uint32_t v) { return dev->reg_write(KREG_CSR_WINDOW + off, v); }

    // Reads ADAPTER_STATUS; if non-zero: reports it (prefix), clears it (W1C), returns it.
    // Returns 0xFFFFFFFF if the register cannot be read.
    uint32_t check_adapter(const char * when) {
        uint32_t st = 0;
        if (dev->reg_read(KREG_ADAPTER_STATUS, &st) != 0) return 0xFFFFFFFFu;
        if (st != 0) {
            fprintf(stderr, "tsr_transport(%s): ADAPTER_STATUS = 0x%08x %s: %s\n", nm(), st, when,
                    adapter_status_str(st).c_str());
            uint32_t after = 0;
            if (dev->reg_write(KREG_ADAPTER_STATUS, 0xFFFFFFFFu) != 0 || dev->reg_read(KREG_ADAPTER_STATUS, &after) != 0) {
                return st;
            }
            if (after != 0) {
                fprintf(stderr, "tsr_transport(%s): ADAPTER_STATUS still 0x%08x after the W1C write\n", nm(), after);
            }
        }
        return st;
    }

    int range(uint32_t ch, uint64_t off, size_t size, const char * what) const {
        if (ch >= cfg.num_ch || off > cfg.ch_bytes || (uint64_t) size > cfg.ch_bytes - off) {
            fprintf(stderr, "tsr_transport(%s): %s out of range: ch %u, offset 0x%" PRIx64 ", %zu bytes (num_ch %u, ch_bytes %" PRIu64 ")\n",
                    nm(), what, ch, off, size, cfg.num_ch, cfg.ch_bytes);
            return -1;
        }
        return 0;
    }

    // Read AP_CTRL through kdev::settle (no run outstanding) until the kernel is idle with no
    // start pending, for at most ms (0: no limit). Every read consumes ap_done / ap_ready, so a
    // completion that ends meanwhile is consumed too. false: not idle in time, or a read failed
    // (*ap = the last value read).
    bool settle_idle(uint32_t ms, uint32_t * ap) {
        const clk::time_point t0 = clk::now();
        for (;;) {
            if (dev->settle(ap) != 0) return false;
            if ((*ap & AP_IDLE) && !(*ap & AP_START)) return true;
            if (ms != 0 && ms_since(t0) >= ms) return false;
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }

    // Wait until the core is not BUSY, polling STATUS through the window (open-time recovery).
    bool wait_not_busy(uint32_t ms) {
        const clk::time_point t0 = clk::now();
        for (;;) {
            bool ok = true;
            const uint32_t st = csr(TSR_CSR_STATUS, &ok);
            if (!ok) return false;
            if (!(st & TSR_STATUS_BUSY)) return true;
            if (ms_since(t0) >= ms) return false;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    int exec(uint64_t cmd_addr_beat, uint32_t cmd_count, const uint32_t args[8], tsr_exec_result * res);
};

int kproto::exec(uint64_t cmd_addr_beat, uint32_t cmd_count, const uint32_t args[8], tsr_exec_result * res) {
    *res = tsr_exec_result{};
    alog.last_run = 0;
    if (broken) {
        fprintf(stderr, "tsr_transport(%s): exec refused: an earlier run did not stop after CTRL.ABORT "
                        "(reset the card, e.g. xbutil reset, and reopen)\n", nm());
        return -1;
    }
    kargs a;
    a.cmd_addr = cmd_addr_beat;
    a.cmd_count = cmd_count;
    for (int i = 0; i < 8; ++i) a.arg[i] = args ? args[i] : 0u;
    if (dev->start(a) != 0) {
        fprintf(stderr, "tsr_transport(%s): kernel start failed\n", nm());
        return -1;
    }
    int rc = 0;
    const clk::time_point t0 = clk::now();
    kwait w = dev->wait(timeout_ms);
    uint32_t ap = 0;
    bool stopped = false;   // the run timed out and was stopped with CTRL.ABORT
    auto stop = [&](const char * why) {   // timeout: report PC / STATUS, write CTRL.ABORT
        const uint32_t pc = csr(TSR_CSR_PC), st = csr(TSR_CSR_STATUS);
        fprintf(stderr, "tsr_transport(%s): exec timed out after %u ms (TSR_XRT_TIMEOUT_MS)%s: PC = %u, STATUS = 0x%x "
                        "(program of %u instructions at CMD beat 0x%" PRIx64 "); writing CTRL.ABORT\n",
                nm(), timeout_ms, why, pc, st, cmd_count, cmd_addr_beat);
        csr_write(TSR_CSR_CTRL, TSR_CTRL_ABORT);
        stopped = true;
    };
    auto stuck = [&]() {   // the core does not stop: refuse every further exec
        broken = true;
        res->status = csr(TSR_CSR_STATUS);
        res->err_pc = csr(TSR_CSR_PC);
        fprintf(stderr, "tsr_transport(%s): the core did not stop within %u ms after CTRL.ABORT (PC = %u, STATUS = 0x%x, "
                        "ADAPTER_STATUS: %s); a memory transaction is probably stuck: reset the card (xbutil reset)\n",
                nm(), abort_ms, res->err_pc, res->status, [&] {
                    uint32_t v = 0;
                    return dev->reg_read(KREG_ADAPTER_STATUS, &v) == 0 ? adapter_status_str(v) : std::string("unreadable");
                }().c_str());
        return -1;
    };
    if (w == kwait::timeout) {
        stop("");
        w = dev->wait(abort_ms);
        if (w != kwait::done) return stuck();
    } else if (w == kwait::error) {
        fprintf(stderr, "tsr_transport(%s): waiting for the kernel failed\n", nm());
        return -1;
    }
    // The completion must be this run's: with nothing outstanding, AP_CTRL shows the kernel idle.
    // A kernel still running means the completion was a stale ap_done (one nobody consumed, see
    // kdev::settle): wait for this run to end within the rest of the timeout (consuming its
    // ap_done, which would otherwise complete the next run early), else stop it like a timeout.
    if (dev->settle(&ap) != 0) return -1;
    if (!(ap & AP_IDLE) || (ap & AP_START)) {
        fprintf(stderr, "tsr_transport(%s): completion reported while the kernel is still running (AP_CTRL = 0x%x): a "
                        "stale ap_done of an earlier, untracked run; waiting for this run to end\n", nm(), ap);
        const uint64_t spent = ms_since(t0);
        const uint32_t left = timeout_ms == 0 ? 0 : (spent >= timeout_ms ? 1 : (uint32_t) (timeout_ms - spent));
        if (!settle_idle(left, &ap)) {
            if (stopped) return stuck();
            stop(" (after a stale completion)");
            if (!settle_idle(abort_ms, &ap)) return stuck();
        }
    }
    if (stopped) rc = -1;   // stopped by ABORT: ERR_CODE = ABORTED below
    bool ok = true;
    res->status = csr(TSR_CSR_STATUS, &ok);
    res->err_code = csr(TSR_CSR_ERR_CODE, &ok);
    res->err_pc = csr(TSR_CSR_ERR_PC, &ok);
    res->err_info = csr(TSR_CSR_ERR_INFO, &ok);
    if (!ok) return -1;
    const uint32_t ast = check_adapter("after a run");
    if (ast != 0 && ast != 0xFFFFFFFFu) {
        alog.last_run = ast;
        alog.sticky |= ast & 0xFFFFu;
        alog.runs++;
        alog.count += adapter_status_count(ast);
    }
    if (ast != 0) {
        if (ast != 0xFFFFFFFFu) {
            fprintf(stderr, "tsr_transport(%s): run failed in the platform adapter (STATUS = 0x%x, ERR_CODE = %u, ERR_PC = %u, "
                            "program of %u instructions at CMD beat 0x%" PRIx64 ")\n",
                    nm(), res->status, res->err_code, res->err_pc, cmd_count, cmd_addr_beat);
        }
        return -1;
    }
    if (!(res->status & TSR_STATUS_DONE)) {
        fprintf(stderr, "tsr_transport(%s): ap_done without STATUS.DONE (STATUS = 0x%x)\n", nm(), res->status);
        return -1;
    }
    return (rc == 0 && !(res->status & TSR_STATUS_ERR)) ? 0 : -1;
}

void kp_close(void * impl) { delete (kproto *) impl; }

const tsr_config * kp_config(void * impl) { return &((kproto *) impl)->cfg; }

int kp_write(void * impl, uint32_t ch, uint64_t off, const void * src, size_t size) {
    kproto * k = (kproto *) impl;
    if (k->range(ch, off, size, "write")) return -1;
    if (size == 0) return 0;
    return k->dev->mem_write(ch, off, src, size);
}

int kp_read(void * impl, uint32_t ch, uint64_t off, void * dst, size_t size) {
    kproto * k = (kproto *) impl;
    if (k->range(ch, off, size, "read")) return -1;
    if (size == 0) return 0;
    return k->dev->mem_read(ch, off, dst, size);
}

int kp_exec(void * impl, uint64_t cmd_addr_beat, uint32_t cmd_count, const uint32_t args[8], tsr_exec_result * res) {
    return ((kproto *) impl)->exec(cmd_addr_beat, cmd_count, args, res);
}

uint32_t kp_csr_read(void * impl, uint32_t off) {
    kproto * k = (kproto *) impl;
    if (off >= KREG_SPACE - KREG_CSR_WINDOW || (off & 3u) != 0) {
        fprintf(stderr, "tsr_transport(%s): CSR offset 0x%x outside the window\n", k->nm(), off);
        return 0xFFFFFFFFu;
    }
    return k->csr(off);
}

// One ops table per kdev name: tsr_transport_name() reports the kdev kind.
struct named_ops {
    std::string       name;
    tsr_transport_ops ops;
};

const tsr_transport_ops * ops_for(const char * name) {
    static std::mutex mtx;
    static std::map<std::string, std::unique_ptr<named_ops>> tables;
    std::lock_guard<std::mutex> lk(mtx);
    auto & e = tables[name];
    if (!e) {
        e.reset(new named_ops);
        e->name = name;
        e->ops = tsr_transport_ops{ e->name.c_str(), kp_close, kp_config, kp_write, kp_read, kp_exec, kp_csr_read };
    }
    return &e->ops;
}

bool is_kproto_ops(const tsr_transport_ops * ops) { return ops != nullptr && ops->close == kp_close; }

// Opens the protocol layer on dev; returns the impl handle (kproto) or nullptr.
kproto * kproto_open(std::unique_ptr<kdev> dev) {
    if (!dev) return nullptr;
    std::unique_ptr<kproto> k(new kproto);
    k->dev = std::move(dev);
    const char * nm = k->dev->name();
    uint64_t v = 0;
    int r;
    if ((r = env_u64("TSR_XRT_TIMEOUT_MS", 0, 0xFFFFFFFFu, &v)) < 0) return nullptr;
    if (r) k->timeout_ms = (uint32_t) v;
    if ((r = env_u64("TSR_XRT_ABORT_MS", 1, 0xFFFFFFFFu, &v)) < 0) return nullptr;
    if (r) k->abort_ms = (uint32_t) v;

    uint32_t aver = 0;
    if (k->dev->reg_read(KREG_ADAPTER_VERSION, &aver) != 0) {
        fprintf(stderr, "tsr_transport(%s): cannot read ADAPTER_VERSION\n", nm);
        return nullptr;
    }
    bool ok = true;
    const uint32_t id = k->csr(TSR_CSR_ID, &ok);
    if (!ok) {
        fprintf(stderr, "tsr_transport(%s): cannot read the core ID through the CSR window at s_axi_control + 0x%x "
                        "(the kernel must declare an s_axi_control range of 0x%x)\n", nm, KREG_CSR_WINDOW, KREG_SPACE);
        return nullptr;
    }
    if (id != TSR_ID_MAGIC) {
        fprintf(stderr, "tsr_transport(%s): core ID reads 0x%08x, expected 0x%08x (\"TSR1\"): not a Tesseract kernel, "
                        "or the CSR window at s_axi_control + 0x%x does not reach the core (ADAPTER_VERSION = 0x%08x)\n",
                nm, id, TSR_ID_MAGIC, KREG_CSR_WINDOW, aver);
        return nullptr;
    }
    const uint32_t ver = k->csr(TSR_CSR_VERSION, &ok);
    const uint32_t cfg0 = k->csr(TSR_CSR_CFG0, &ok), cfg1 = k->csr(TSR_CSR_CFG1, &ok), cfg2 = k->csr(TSR_CSR_CFG2, &ok);
    const uint32_t caps = k->csr(TSR_CSR_CAPS, &ok);
    if (!ok) return nullptr;
    if ((ver >> 8) != TSR_ISA_MAJOR) {
        fprintf(stderr, "tsr_transport(%s): core ISA version %u.%u, this software implements major %u\n", nm, (ver >> 8) & 0xFFu,
                ver & 0xFFu, TSR_ISA_MAJOR);
        return nullptr;
    }
    // CSR-visible parameters from CFG0..2 / CAPS; BEAT_BITS, NIREG, LOOP_DEPTH, RD_OUTSTANDING and
    // ADDR_BEAT_W are fixed by the U200 kernel contract (512-bit m_axi ports, the v1 core).
    tsr_config d = TSR_CONFIG_U200_V1;
    d.num_ch = cfg0 & 0xFFu;
    d.lanes = (cfg0 >> 8) & 0xFFFu;
    d.t_max = (cfg0 >> 20) & 0xFu;
    d.spm_bytes = cfg1 * TSR_LINE_BYTES;
    d.vl = cfg2 & 0xFFu;
    d.nsreg = (cfg2 >> 8) & 0xFFu;
    d.k_max = ((cfg2 >> 16) & 0xFFFFu) * 256u;
    d.caps = caps;
    if (d.num_ch < 1 || d.num_ch > TSR_NUM_CH_MAX || d.num_ch != k->dev->num_ch()) {
        fprintf(stderr, "tsr_transport(%s): CFG0.NUM_CH = %u, but the kernel has %u channel buffer%s bound\n", nm, d.num_ch,
                k->dev->num_ch(), k->dev->num_ch() == 1 ? "" : "s");
        return nullptr;
    }
    if (d.t_max < 1 || d.t_max > TSR_T_MAX_ARCH || d.lanes == 0 || d.spm_bytes == 0 || d.k_max == 0) {
        fprintf(stderr, "tsr_transport(%s): implausible configuration CFG0 0x%08x CFG1 0x%08x CFG2 0x%08x\n", nm, cfg0, cfg1, cfg2);
        return nullptr;
    }
    // Usable bytes per channel: the smallest bound buffer (symmetric allocation), in beats, and
    // never more than the core addresses (CH_BYTES, 2^ADDR_BEAT_W beats).
    uint64_t chb = std::min<uint64_t>(KERNEL_CH_BYTES, (1ull << d.addr_beat_w) * TSR_BEAT_BYTES);
    for (uint32_t c = 0; c < d.num_ch; ++c) {
        chb = std::min(chb, k->dev->ch_bytes(c));
        if (k->dev->ch_addr(c) % 4096 != 0) {
            fprintf(stderr, "tsr_transport(%s): channel %u buffer at device address 0x%" PRIx64 " is not 4 KiB aligned\n", nm, c,
                    k->dev->ch_addr(c));
            return nullptr;
        }
    }
    d.ch_bytes = chb & ~(uint64_t) (TSR_BEAT_BYTES - 1);
    if (d.ch_bytes == 0) {
        fprintf(stderr, "tsr_transport(%s): channel buffers are empty\n", nm);
        return nullptr;
    }
    k->cfg = d;

    // Leftovers of an earlier process: a core that is still running, sticky adapter errors.
    const uint32_t st = k->csr(TSR_CSR_STATUS, &ok);
    if (!ok) return nullptr;
    if (st & TSR_STATUS_BUSY) {
        fprintf(stderr, "tsr_transport(%s): core BUSY at open (PC = %u): an earlier run did not finish; writing CTRL.ABORT\n", nm,
                k->csr(TSR_CSR_PC));
        k->csr_write(TSR_CSR_CTRL, TSR_CTRL_ABORT);
        if (!k->wait_not_busy(k->abort_ms)) {
            fprintf(stderr, "tsr_transport(%s): the core is still BUSY %u ms after CTRL.ABORT: reset the card (xbutil reset)\n", nm,
                    k->abort_ms);
            return nullptr;
        }
    }
    // A completion nobody consumed (the run just stopped, or one an earlier process started and
    // did not wait for): consume it now, or the next xrt::run would complete at once on it.
    uint32_t ap = 0;
    if (k->dev->settle(&ap) != 0) return nullptr;
    if ((ap & AP_DONE) && !(st & TSR_STATUS_BUSY)) {
        fprintf(stderr, "tsr_transport(%s): consumed a completion left by an earlier run (AP_CTRL = 0x%x)\n", nm, ap);
    }
    if ((!(ap & AP_IDLE) || (ap & AP_START)) && !k->settle_idle(k->abort_ms, &ap)) {
        fprintf(stderr, "tsr_transport(%s): the kernel is not idle at open (AP_CTRL = 0x%x, core STATUS = 0x%x): reset the "
                        "card (xbutil reset)\n", nm, ap, k->csr(TSR_CSR_STATUS));
        return nullptr;
    }
    const uint32_t ast = k->check_adapter("at open (left by an earlier run; cleared)");
    if (ast == 0xFFFFFFFFu) return nullptr;
    k->alog.at_open = ast;
    return k.release();
}

std::mutex & registry_mutex() {
    static std::mutex m;
    return m;
}
std::map<std::string, kdev_factory> & registry() {
    static std::map<std::string, kdev_factory> r;
    return r;
}

} // namespace

tsr_transport * transport_open_kdev(std::unique_ptr<kdev> dev) {
    if (!dev) return nullptr;
    const tsr_transport_ops * ops = ops_for(dev->name());
    kproto * k = kproto_open(std::move(dev));
    if (k == nullptr) return nullptr;
    tsr_transport * t = tsr_transport_wrap(ops, k);
    if (t == nullptr) delete k;
    return t;
}

kdev * transport_kdev(tsr_transport * t) {
    if (t == nullptr || !is_kproto_ops(tsr_transport_get_ops(t))) return nullptr;
    return ((kproto *) tsr_transport_get_impl(t))->dev.get();
}

adapter_log transport_adapter_log(tsr_transport * t) {
    if (t == nullptr || !is_kproto_ops(tsr_transport_get_ops(t))) return adapter_log();
    return ((kproto *) tsr_transport_get_impl(t))->alog;
}

uint64_t transport_perf(tsr_transport * t, uint32_t idx) {
    const uint32_t lo = tsr_transport_csr_read(t, TSR_CSR_PERF_BASE + 8 * idx);
    const uint32_t hi = tsr_transport_csr_read(t, TSR_CSR_PERF_BASE + 8 * idx + 4);
    return ((uint64_t) hi << 32) | lo;
}

bool register_kdev(const std::string & kind, kdev_factory f) {
    if (kind.empty() || kind == "sim" || kind == "xrt" || kind == "mock" || kind == "vkernel" || !f) return false;
    std::lock_guard<std::mutex> lk(registry_mutex());
    registry()[kind] = std::move(f);
    return true;
}

std::vector<env_default> xrt_env_defaults(const std::function<const char *(const char *)> & env) {
    auto get = [&env](const char * n) -> const char * {
        const char * v = env ? env(n) : getenv(n);
        return (v != nullptr && v[0] != '\0') ? v : nullptr;
    };
    // docs/xrt-host.md §9: defect 1 (32-bit mmapStorageEngine size), defect 2 (unaligned DMA).
    static const env_default HW_EMU[] = { { "TSR_XRT_CH_GIB", "3" }, { "TSR_XRT_DMA", "staged" }, { "TSR_XRT_DMA_ALIGN", "4096" } };
    std::vector<env_default> r;
    const char * em = get("XCL_EMULATION_MODE");
    if (em == nullptr || strcmp(em, "hw_emu") != 0) return r;
    for (const env_default & d : HW_EMU) {
        if (get(d.name) == nullptr) r.push_back(d);
    }
    return r;
}

#ifndef TSR_HAVE_XRT
std::unique_ptr<kdev> open_xrt_kdev() {
    fprintf(stderr, "tsr_transport: the \"xrt\" transport is not built in (configure with -DTSR_WITH_XRT=ON)\n");
    return nullptr;
}
#endif

} // namespace tsr

// Dispatcher entry (tsr_transport.c): every kind except "sim".
extern "C" void * tsr_transport_kdev_open(const char * kind, const tsr_transport_ops ** ops) {
    using namespace tsr;
    std::unique_ptr<kdev> dev;
    try {
        if (strcmp(kind, "xrt") == 0) {
            dev = open_xrt_kdev();
        } else if (strcmp(kind, "mock") == 0) {
            dev = open_mock_kdev();
        } else if (strcmp(kind, "vkernel") == 0) {
            dev = open_vkernel_kdev();
        } else {
            kdev_factory f;
            {
                std::lock_guard<std::mutex> lk(registry_mutex());
                auto it = registry().find(kind);
                if (it != registry().end()) f = it->second;
            }
            if (!f) {
                fprintf(stderr, "tsr_transport: unknown transport \"%s\" (TSR_DEVICE must be sim, xrt, mock, vkernel%s)\n", kind, [] {
                    std::string s;
                    std::lock_guard<std::mutex> lk(registry_mutex());
                    for (auto & e : registry()) s += " or " + e.first;
                    return s;
                }().c_str());
                return nullptr;
            }
            dev = f();
        }
    } catch (const std::exception & e) {
        fprintf(stderr, "tsr_transport(%s): %s\n", kind, e.what());
        return nullptr;
    }
    if (!dev) return nullptr;
    *ops = ops_for(dev->name());
    return kproto_open(std::move(dev));
}
