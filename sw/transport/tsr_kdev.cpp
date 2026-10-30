// tsr_kdev.cpp — the protocol layer: tsr_transport over any kdev (tsr_kdev.h), one implementation
// for every kind (V2-SW-18). It implements docs/v2/01-execution-model.md "EX" §2.1, §3, §4.8-§4.9,
// §8.4-§8.6 and §10, as refined by docs/v2/06-software-verification.md "SW" §9.1-§9.2 and
// docs/v2/05-debug.md "DB" §4.1, §6.1, §6.3, §7.3-§7.4, §9.2:
//
//   open    identity (ADAPTER_VERSION, ID, VERSION 2, CFG*, CAPS, CFG3, CTL_ID, RV_CFG, Q_DEPTH,
//           RV_FREQ != 0, LQ_CFG, PERF_INFO, DBG_ID), ADAPTER_STATUS read and cleared; the host
//           counters (LQ_SEQ, LCQ_RD, LOG_RD) and the LQ_W shadow read from the device (the rings
//           keep counting across processes: only a card reset zeroes them); unless TSR_ATTACH=1:
//           SOFT_RESET (stops a control core left running by an earlier process and discards the
//           records and log words it left unread), bind (MEM_EN 0, ch<c>_base, MEM_EN 1, checked),
//           LOG_CFG and the DBG_CFG settings; a debug configuration left armed (SERIAL, stop
//           sources, TBPs: DB keeps them across SOFT_RESET) makes the waits debug-aware.
//   boot    .note.tsr requirements against the device; VERIFY of every ITCM segment, a cold load
//           (LOAD_ADDR, LOAD_DATA words, receive CRC, VERIFY CRC) of each that does not match and
//           of every DTCM segment; RV_BOOT = 0, RV_RUN = 1; HELLO first, with the image id, the SDK
//           ABI and the kernel count (EX §3.3-§3.5). Warm = no ITCM segment written.
//   launch  host-side credit (at most LQ_DEPTH outstanding, no device read), the LQ_W words that
//           differ from the shadow, LQ_PUSH (EX §4.8).
//   wait    LCQ consumption (an OK DONE / PONG of the oldest launch: r0, plus r2 unless the kernel
//           is graph or replay; every word otherwise), answers in descriptor order, events to the
//           callback; the poll policy (busy-poll while the elapsed time is below the last service
//           time of the same kernel, measured by the host from push or previous answer to the
//           consumption, then every max(10 µs, 1 % of max(that time, the time waited so far)) with
//           idle() and the mutex released); the log
//           drained every poll with TSR_LOG, else every 10 ms; the debug-aware timeout (time counts
//           only while STATUS.RV_HALTED = 0 and DBG_STATUS.STOP_ACTIVE = 0, read once per backoff
//           poll; off with DBG_CFG.SERIAL = 1); on a timeout DB's recovery ladder: level 0 hang
//           dump, UNIT_HUNG -> "card reset needed" (FAILED), level 2 LQ_CANCEL covering every
//           pushed launch and TSR_ABORT_MS for their answers, level 3 SOFT_RESET + warm restart.
//   TRAP    the log drained and decoded, the trap reported (cause, mepc -> function and line),
//           SOFT_RESET + warm restart unless TSR_GDB_WAIT (EX §8.6, DB §6.3).
//   cancel / stop  LQ_CANCEL / LQ_STOP watermarks (EX §4.9), moved up behind the answered launches
//           every 2^30 launches so that they never wrap onto new ones; soft reset (EX §8.5); log
//           drain.
//
// Time base: kdev::now_ns() (device time on sim, mock and vkernel, whose server reports it; host
// time on xrt).

#include "tsr_kdev.h"

#include "tsr_elf.h"
#include "tsr_image_int.h"
#include "tsr_logdec.h"
#include "tsr_trace.h"
#include "tsr_transport_impl.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cinttypes>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

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
    if (v & AST_RRESP)       add("RRESP error (read response not OKAY)");
    if (v & AST_BRESP)       add("BRESP error (write response not OKAY)");
    if (v & AST_ORPHAN)      add("response with no outstanding request for its id");
    if (v & AST_BASE_ALIGN)  add("ch<c>_base not 4 KiB aligned at the MEM_EN edge");
    if (v & AST_REQ_HELD)    add("core memory request held while MEM_EN = 0");
    if (v & AST_BASE_LOCKED) add("ch<c>_base written while MEM_EN = 1 (ignored)");
    if (v & 0xFFC0u)         add("unknown bits");
    char buf[64];
    snprintf(buf, sizeof(buf), "%u error response%s%s", adapter_status_count(v),
             adapter_status_count(v) == 1 ? "" : "s", adapter_status_count(v) == 0xFFFFu ? " (saturated)" : "");
    add(buf);
    return s;
}

int env_u64(const char * who, const char * name, uint64_t lo, uint64_t hi, uint64_t * out) {
    const char * v = getenv(name);
    if (v == nullptr || v[0] == '\0') return 0;
    char * end = nullptr;
    errno = 0;
    const unsigned long long r = strtoull(v, &end, 10);
    if (errno != 0 || end == v || *end != '\0' || v[0] == '-' || v[0] == '+' || v[0] == ' ' || r < lo || r > hi) {
        fprintf(stderr, "tsr_transport(%s): %s=\"%s\" is not an integer in [%" PRIu64 ", %" PRIu64 "]\n", who, name, v, lo, hi);
        return -1;
    }
    *out = (uint64_t) r;
    return 1;
}

uint64_t kdev::now_ns() {
    return (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

void kdev::idle(uint32_t, uint64_t interval_ns) { std::this_thread::sleep_for(std::chrono::nanoseconds(interval_ns)); }

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

// ===========================================================================
// The protocol layer
// ===========================================================================
using namespace tsr;

namespace {

uint64_t wall_ns() {
    return (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// mkdir -p with mkdir(2), component by component (no shell inside a host process; an existing
// directory is fine). 0, or the errno of the failure.
int make_dirs(const std::string & path) {
    for (size_t i = 0; i <= path.size();) {
        const size_t j = path.find('/', i);
        const std::string p = path.substr(0, j == std::string::npos ? path.size() : j);
        if (!p.empty() && mkdir(p.c_str(), 0777) != 0 && errno != EEXIST) return errno;
        if (j == std::string::npos) break;
        i = j + 1;
    }
    struct stat sb;
    if (stat(path.c_str(), &sb) != 0) return errno;
    return S_ISDIR(sb.st_mode) ? 0 : ENOTDIR;
}

constexpr uint64_t MS = 1000000ull;
constexpr uint64_t POLL_MIN_NS = 10000;        // EX §4.8: poll at most every 10 µs
constexpr uint64_t POLL_MAX_NS = 50 * MS;      // ... and at least every 50 ms (responsiveness of long waits)
constexpr uint64_t LOG_EVERY_NS = 10 * MS;     // SW §9.1: drain the log every 10 ms without TSR_LOG
constexpr size_t   LOG_BUF_WORDS = 64 * 1024;  // raw words kept for tsr_transport_log_drain
constexpr size_t   LOG_TAIL_LINES = 256;       // decoded lines kept for failure reports and dumps
constexpr size_t   KEEP_RECORDS = 1u << 16;    // answered-but-unwaited / lost launches remembered

enum class dstate { reset, ready, trapped, failed, attached };

const char * state_name(dstate s) {
    switch (s) {
    case dstate::reset:    return "not booted";
    case dstate::ready:    return "ready";
    case dstate::trapped:  return "TRAPPED (TSR_GDB_WAIT)";
    case dstate::failed:   return "FAILED";
    case dstate::attached: return "attached (TSR_ATTACH): boot an image first";
    }
    return "?";
}

struct launch_rec {
    uint32_t seq;
    uint32_t key;       // type << 16 | kernel: the poll policy's "same kernel"
    uint64_t t_wall;    // push time (stats)
    uint64_t t_push;    // push time in the transport's time base (the poll policy's duration)
    bool     ret0;      // a base kernel whose RET is 0 on success (graph, replay: SW §6.4)
};

// A recursive mutex that the wait loop can release completely between polls (DB §4.1: an
// in-process debug service uses the device while a launch waits).
struct rmutex {
    std::mutex m;
    std::atomic<std::thread::id> owner{};
    int depth = 0;
    std::atomic<int> waiters{ 0 };
    void lock() {
        if (owner.load() == std::this_thread::get_id()) {
            depth++;
            return;
        }
        waiters++;
        m.lock();
        waiters--;
        owner = std::this_thread::get_id();
        depth = 1;
    }
    void unlock() {
        if (--depth == 0) {
            owner = std::thread::id();
            m.unlock();
        }
    }
    int release_all() {
        const int d = depth;
        depth = 0;
        owner = std::thread::id();
        m.unlock();
        return d;
    }
    void reacquire(int d) {
        m.lock();
        owner = std::this_thread::get_id();
        depth = d;
    }
};

} // namespace

struct tsr_transport {
    std::unique_ptr<kdev> dev;
    std::string kind;
    tsr_config cfg{};

    // identity (open)
    uint32_t r_version = 0, r_caps = 0, r_cfg3 = 0, r_ctl_id = 0, r_rv_cfg = 0, r_q_depth = 0, r_rv_freq = 0;
    uint32_t r_lq_cfg = 0, r_perf_info = 0, r_dbg_id = 0, r_adapter_version = 0;
    uint32_t lq_depth = TSR_LQ_DEPTH, lcq_depth = TSR_LCQ_DEPTH, log_depth = TSR_LOG_DEPTH;
    uint32_t itcm_bytes = 0, dtcm_bytes = 0;
    bool     has_dbg = false;
    // A debugger may be in play: TSR_TILE_SERIAL, or a csr_write into the DBG block or the DMI
    // window (DB's in-process service writes through the transport, DB §4.1). Only then does a
    // backoff poll also read DBG_STATUS and DBG_CFG (STATUS.RV_HALTED is read always).
    bool     dbg_active = false;

    // settings (SW §9.2, DB §9.5)
    uint64_t launch_ms = 60000, abort_ms = 10000, boot_ms = 2000, reset_ms = 15000;
    bool     attach = false, stats_on = false, gdb_wait = false, full_records = false;
    int      log_level = 3, tile_serial = -1, tile_hist = -1;
    FILE *   log_out = nullptr;
    bool     log_close = false;
    std::string dump_dir;

    // protocol state
    dstate state = dstate::reset;
    std::string fail_reason;
    const tsr_image * img = nullptr;      // booted image (a reference is held)
    uint32_t pushed = 0, lcq_rd = 0, log_rd = 0;
    uint32_t lq_shadow[8] = {};
    std::deque<launch_rec> outstanding;   // pushed, not answered, in push order
    std::map<uint32_t, tsr_record> answered;
    std::deque<uint32_t> answered_order;
    std::set<uint32_t> lost;
    std::deque<uint32_t> lost_order;
    std::map<uint32_t, uint64_t> last_dur;   // launch_rec::key -> the last service time (time base, ns)
    uint64_t t_last_answer = 0;               // time base at the last answer consumed
    void (*event_cb)(void *, const tsr_record *) = nullptr;
    void * event_user = nullptr;
    std::deque<tsr_record> ev_pending;
    bool trap_pending = false, trap_is_answer = false;
    tsr_record trap_rec{};
    bool plat_err_reported = false;   // this PLAT_ERR episode was reported (and counted)
    bool plat_err_clear = false;      // ADAPTER_STATUS to clear once the episode's launches answered
    uint32_t plat_err_upto = 0;       // the episode covers the launches pushed before this seq
    uint32_t plat_err_status = 0;     // ADAPTER_STATUS when the episode was seen (adapter kinds)
    // The LQ_CANCEL / LQ_STOP watermarks written since the last SOFT_RESET (EX §4.9: valid until
    // then), kept below the answered launches (refresh_watermarks).
    struct watermark {
        bool     valid = false;
        uint32_t s = 0;
    } wm_cancel, wm_stop;
    uint32_t wm_refresh = 1u << 30;   // rewrite a watermark this many launches after it was written

    // log
    std::deque<uint32_t> log_buf;
    log_decoder dec;
    std::deque<std::pair<uint32_t, std::string>> log_tail;   // (LSEQ, decoded line)
    uint64_t log_last = 0;

    // statistics and records
    tsr_transport_stats st{};
    adapter_log alog;
    boot_info binfo;
    recovery_log rlog;
    uint32_t dumps = 0;

    rmutex mu;

    ~tsr_transport() {
        image_unref(img);
        if (log_close && log_out) fclose(log_out);
    }

    // ---- messages ----
    void msg(const char * f, ...) __attribute__((format(printf, 2, 3))) {
        char b[2048];
        va_list ap;
        va_start(ap, f);
        vsnprintf(b, sizeof(b), f, ap);
        va_end(ap);
        fprintf(stderr, "tsr_transport(%s): %s\n", kind.c_str(), b);
    }

    void fail(const std::string & why) {
        if (state != dstate::failed) msg("device FAILED: %s", why.c_str());
        state = dstate::failed;
        fail_reason = why;
        rlog.failed = true;
        lose_outstanding("device failed");
    }

    // ---- register access (stats; an access the kdev fails is reported there) ----
    bool rd(uint32_t off, uint32_t * v) {
        st.csr_reads++;
        if (dev->reg_read(KREG_CSR_WINDOW + off, v) != 0) {
            *v = 0xFFFFFFFFu;
            return false;
        }
        return true;
    }
    uint32_t rdv(uint32_t off) {
        uint32_t v = 0;
        rd(off, &v);
        return v;
    }
    bool wr(uint32_t off, uint32_t v) {
        st.csr_writes++;
        return dev->reg_write(KREG_CSR_WINDOW + off, v) == 0;
    }
    bool ard(uint32_t off, uint32_t * v) {
        st.csr_reads++;
        return dev->reg_read(off, v) == 0;
    }
    bool awr(uint32_t off, uint32_t v) {
        st.csr_writes++;
        return dev->reg_write(off, v) == 0;
    }
    uint64_t now() { return dev->now_ns(); }
    const char * tbase() { return dev->simulated() ? "device" : "host"; }
    static uint64_t cyc_ns(uint64_t cycles) { return (uint64_t) ((double) cycles * 1e9 / CORE_HZ_NOMINAL); }

    // STATUS, with the "card lost" check of DB §7.2 (reads return 0xFFFFFFFF; bits 31:28 are 0).
    bool status(uint32_t * s) {
        if (!rd(TSR_CSR_STATUS, s)) return false;
        if (*s == 0xFFFFFFFFu) {
            fail("STATUS reads 0xFFFFFFFF: the AXI firewall tripped or the card is lost (card reset)");
            return false;
        }
        return true;
    }

    // ---- bookkeeping ----
    void remember_lost(uint32_t seq) {
        lost.insert(seq);
        lost_order.push_back(seq);
        rlog.lost++;
        while (lost_order.size() > KEEP_RECORDS) {
            lost.erase(lost_order.front());
            lost_order.pop_front();
        }
    }
    void lose_outstanding(const char * why) {
        if (!outstanding.empty())
            msg("%zu launch%s lost without an answer (%s): seq %u..%u", outstanding.size(), outstanding.size() == 1 ? "" : "es", why,
                outstanding.front().seq, outstanding.back().seq);
        for (const launch_rec & l : outstanding) remember_lost(l.seq);
        outstanding.clear();
    }
    bool is_outstanding(uint32_t seq) const {
        for (const launch_rec & l : outstanding)
            if (l.seq == seq) return true;
        return false;
    }

    // ---- the log (EX §4.10, DB §6.1, SW §9.1) ----
    void log_line(uint32_t lseq, const std::string & l) {
        if (log_out) {
            fprintf(log_out, "%s\n", l.c_str());
            fflush(log_out);
        }
        log_tail.emplace_back(lseq, l);
        while (log_tail.size() > LOG_TAIL_LINES) log_tail.pop_front();
    }
    // Moves [LOG_RD, LOG_WR) into the host buffer and the decoder, then writes LOG_RD.
    bool drain_log() {
        log_last = now();
        uint32_t wr_ = 0;
        if (!rd(TSR_CSR_LOG_WR, &wr_)) return false;
        uint32_t n = wr_ - log_rd;
        if (n == 0) return true;
        if (n > log_depth) {
            msg("LOG_WR - LOG_RD = %u exceeds LOG_DEPTH %u: resynchronising at LOG_WR", n, log_depth);
            log_rd = wr_;
            dec.reset();
            return wr(TSR_CSR_LOG_RD, wr_);
        }
        std::vector<uint32_t> w(n);
        for (uint32_t k = 0; k < n; ++k) {
            if (!rd(TSR_CSR_LOG_RING((log_rd + k) % log_depth), &w[k])) return false;
        }
        if (!wr(TSR_CSR_LOG_RD, wr_)) return false;
        log_rd = wr_;
        st.log_words += n;
        log_buf.insert(log_buf.end(), w.begin(), w.end());
        while (log_buf.size() > LOG_BUF_WORDS) log_buf.pop_front();   // the newest words are kept
        dec.feed(w.data(), w.size(), [this](const std::string & l) { log_line(dec.last_lseq, l); });
        return true;
    }
    void maybe_drain_log() {
        if (log_out != nullptr || now() - log_last >= LOG_EVERY_NS) drain_log();
    }
    // The last n decoded records of launch seq (its LSEQ = SEQ[7:0], DB §6.1), on stderr.
    void print_log_tail(size_t n, const char * why, uint32_t seq) {
        if (log_out != nullptr) return;   // already printed with TSR_LOG
        std::vector<const std::string *> sel;
        for (auto it = log_tail.rbegin(); it != log_tail.rend() && sel.size() < n; ++it)
            if (it->first == (seq & 0xFFu)) sel.push_back(&it->second);
        if (sel.empty()) return;
        fprintf(stderr, "tsr_transport(%s): device log of launch seq %u %s (last %zu record%s):\n", kind.c_str(), seq, why, sel.size(),
                sel.size() == 1 ? "" : "s");
        for (auto it = sel.rbegin(); it != sel.rend(); ++it) fprintf(stderr, "  %s\n", (*it)->c_str());
    }

    // ---- PLAT_ERR (EX §8.6: read, report and clear ADAPTER_STATUS; outputs invalid) ----
    // One report per episode (STATUS.PLAT_ERR seen while waiting and the DONE PLAT_ERR answers
    // that follow). The episode covers the launches pushed before it was seen: they ran or queued
    // while the error was raised. ADAPTER_STATUS (W1C) is cleared once all of them have answered,
    // not earlier: plat_err_i follows it, and the runtime samples LQ_FLAGS.PLAT_ERR when it answers
    // (R7), so clearing it while one of them runs would let it answer OK with invalid outputs. Not
    // later either: with launches pipelined one is always outstanding, and every later launch would
    // answer PLAT_ERR after a single transient error.
    void plat_err_seen(const char * where) {
        if (!plat_err_reported) {
            plat_err_reported = true;
            plat_err_upto = pushed;
            uint32_t a = 0;
            if (dev->has_adapter() && ard(KREG_ADAPTER_STATUS, &a)) {
                msg("platform error (%s): ADAPTER_STATUS 0x%08x: %s; the outputs of the launches in flight are invalid", where, a,
                    adapter_status_str(a).c_str());
            } else {
                msg("platform error (%s): STATUS.PLAT_ERR; the outputs of the launches in flight are invalid", where);
            }
            plat_err_status = a;
            alog.plat_errs++;
            alog.last = a;
        }
        plat_err_clear = true;
        plat_err_try_clear();
    }
    // Ends the episode once no launch pushed before it was seen is outstanding. An adapter error
    // raised since (ADAPTER_STATUS changed) may have hit the launches now in flight: it extends the
    // episode to every launch pushed so far (one more event). No latch without an adapter: the sim
    // kind's plat_err_i is the simulator's input.
    void plat_err_try_clear() {
        if (!plat_err_clear) return;
        if (!outstanding.empty() && (int32_t) (outstanding.front().seq - plat_err_upto) < 0) return;
        if (dev->has_adapter()) {
            uint32_t a = 0;
            if (ard(KREG_ADAPTER_STATUS, &a) && a != 0 && a != plat_err_status && !outstanding.empty()) {
                msg("platform error during the episode: ADAPTER_STATUS 0x%08x: %s; the outputs of seq %u..%u are invalid too", a,
                    adapter_status_str(a).c_str(), outstanding.front().seq, pushed - 1);
                alog.plat_errs++;
                alog.last = a;
                plat_err_status = a;
                plat_err_upto = pushed;
                return;
            }
            awr(KREG_ADAPTER_STATUS, 0xFFFFFFFFu);
        }
        plat_err_clear = false;
        plat_err_reported = false;
    }

    // ---- watermarks (EX §4.9) ----
    // Launch q is covered by s iff (s - q) mod 2^32 < 2^31, and a watermark stays valid until
    // SOFT_RESET: 2^31 launches after a cancel or stop at s it would cover new launches again
    // (about 6 days at 145 launches per token and 30 tokens/s). 2^30 launches after a write, it
    // is rewritten just below the oldest unanswered launch, where it covers only answered ones.
    void wm_written(uint32_t off, uint32_t s) {
        watermark & w = off == TSR_CSR_LQ_CANCEL ? wm_cancel : wm_stop;
        w.valid = true;
        w.s = s;
    }
    bool refresh_watermarks() {
        const uint32_t below = (outstanding.empty() ? pushed : outstanding.front().seq) - 1;
        for (uint32_t off : { (uint32_t) TSR_CSR_LQ_CANCEL, (uint32_t) TSR_CSR_LQ_STOP }) {
            watermark & w = off == TSR_CSR_LQ_CANCEL ? wm_cancel : wm_stop;
            const uint32_t age = pushed - w.s;   // >= 2^31: s is ahead of PUSHED (covers launches to come): kept
            if (!w.valid || age <= wm_refresh || age >= 0x80000000u) continue;
            if (!wr(off, below)) return false;
            w.s = below;
        }
        return true;
    }

    // ---- LCQ consumption (EX §4.8) ----
    void read_record(uint32_t k, tsr_record * r) {
        const uint32_t slot = k % lcq_depth;
        memset(r, 0, sizeof(*r));
        rd(TSR_CSR_LCQ_REC(slot, 0), &r->w[0]);
        const uint32_t type = TSR_GET(r->w[0], TSR_LCQ_R0_TYPE), stt = TSR_GET(r->w[0], TSR_LCQ_R0_STATUS),
                       err = TSR_GET(r->w[0], TSR_LCQ_R0_ERR_CODE), s8 = TSR_GET(r->w[0], TSR_LCQ_R0_SEQ);
        const bool minimal = !full_records && (type == TSR_LCQ_R0_TYPE_DONE || type == TSR_LCQ_R0_TYPE_PONG) &&
                             stt == TSR_LCQ_R0_STATUS_OK && err == 0 && !outstanding.empty() && s8 == (outstanding.front().seq & 0xFFu);
        if (minimal) {
            // EX §4.8: an OK answer to the oldest launch is r0 alone; r1 = its SEQ, DONE r3 =
            // TICKET_NONE and r4-r6 = 0 by EX §4.4; r2 (RET, the PONG's ARG0 echo) is read unless
            // the kernel's RET is 0 on success (graph, replay: SW §6.4); r7 (DURATION) is not read
            // (0 here: the poll policy measures the service time itself).
            r->w[1] = outstanding.front().seq;
            if (type == TSR_LCQ_R0_TYPE_PONG || !outstanding.front().ret0) rd(TSR_CSR_LCQ_REC(slot, 2), &r->w[2]);
            r->w[3] = type == TSR_LCQ_R0_TYPE_DONE ? TSR_TICKET_NONE : 0u;
        } else {
            for (uint32_t j = 1; j < 8; ++j) rd(TSR_CSR_LCQ_REC(slot, j), &r->w[j]);
        }
    }

    void answer(const tsr_record & rec) {
        const uint32_t s8 = TSR_GET(rec.w[0], TSR_LCQ_R0_SEQ);
        if (outstanding.empty()) {
            msg("answer record (r0 0x%08x, seq %u) with no launch outstanding: left by an earlier process or pushed behind the "
                "transport; passed to the event callback", rec.w[0], rec.w[1]);
            ev_pending.push_back(rec);
            return;
        }
        launch_rec l = outstanding.front();
        if (s8 != (l.seq & 0xFFu) || rec.w[1] != l.seq) {
            msg("answer for seq %u (r0 0x%08x) while seq %u is the oldest outstanding launch: answers out of order (a runtime "
                "or device fault)", rec.w[1], rec.w[0], l.seq);
            auto it = std::find_if(outstanding.begin(), outstanding.end(), [&](const launch_rec & x) { return x.seq == rec.w[1]; });
            if (it == outstanding.end()) {
                ev_pending.push_back(rec);
                return;
            }
            l = *it;
            outstanding.erase(it);
        } else {
            outstanding.pop_front();
        }
        answered[l.seq] = rec;
        answered_order.push_back(l.seq);
        while (answered_order.size() > KEEP_RECORDS) {
            answered.erase(answered_order.front());
            answered_order.pop_front();
        }
        // The service time: from the push, or from the previous answer if that came later (the
        // launch waited in the queue), to this consumption.
        const uint64_t tn = now();
        last_dur[l.key] = tn - std::max(l.t_push, t_last_answer);
        t_last_answer = tn;
        st.launch_wall_ns += wall_ns() - l.t_wall;
        if (TSR_GET(rec.w[0], TSR_LCQ_R0_TYPE) == TSR_LCQ_R0_TYPE_DONE &&
            TSR_GET(rec.w[0], TSR_LCQ_R0_STATUS) == TSR_LCQ_R0_STATUS_PLAT_ERR) {
            char w[48];
            snprintf(w, sizeof(w), "DONE PLAT_ERR of seq %u", l.seq);
            plat_err_seen(w);
        }
    }

    // Consumes [LCQ_RD, LCQ_WR): answers, events (queued for the callback), TRAP (flagged).
    int consume() {
        uint32_t v = 0;
        if (!rd(TSR_CSR_LCQ_WR, &v)) return TSR_TE_IO;
        uint32_t n = v - lcq_rd;
        if (n == 0) return 0;
        if (n > lcq_depth) {
            uint32_t r = 0;
            rd(TSR_CSR_LCQ_RD, &r);
            msg("LCQ_WR - LCQ_RD = %u exceeds LCQ_DEPTH %u (host LCQ_RD %u, device %u): resynchronising", n, lcq_depth, lcq_rd, r);
            lcq_rd = r;
            n = v - r;
            if (n > lcq_depth) {
                fail("the LCQ counters are inconsistent (LCQ_WR " + std::to_string(v) + ", LCQ_RD " + std::to_string(r) + ")");
                return TSR_TE_DEVICE;
            }
        }
        for (uint32_t k = lcq_rd; k != v; ++k) {
            tsr_record rec;
            read_record(k, &rec);
            const uint32_t type = TSR_GET(rec.w[0], TSR_LCQ_R0_TYPE);
            if (type == TSR_LCQ_R0_TYPE_DONE || type == TSR_LCQ_R0_TYPE_PONG) {
                answer(rec);
            } else if (type == TSR_LCQ_R0_TYPE_TRAP) {
                // An answer iff the trap hit a running launch: then ANSWERED moved past the oldest
                // outstanding launch (r1 = SEQ = TAKEN either way; the runtime parks after a trap).
                bool is_answer = false;
                if (!outstanding.empty()) {
                    uint32_t ans = 0;
                    rd(TSR_CSR_LCQ_ANSWERED, &ans);
                    is_answer = (int32_t) (ans - outstanding.front().seq) > 0 && rec.w[1] == outstanding.front().seq;
                }
                if (is_answer) answer(rec);
                trap_pending = true;
                trap_is_answer = is_answer;
                trap_rec = rec;
            } else if (type >= TSR_LCQ_R0_TYPE_EVENT_FIRST) {
                ev_pending.push_back(rec);
            } else if (type == TSR_LCQ_R0_TYPE_HELLO) {
                hello_outside_boot(rec);
            } else {
                msg("unexpected record type 0x%02x (r0 0x%08x) outside boot; passed to the event callback", type, rec.w[0]);
                ev_pending.push_back(rec);
            }
        }
        if (!wr(TSR_CSR_LCQ_RD, v)) return TSR_TE_IO;
        lcq_rd = v;
        plat_err_try_clear();
        return (int) n;
    }

    // A HELLO outside a boot: the RISC-V was reset and released behind the transport (RV_RUN
    // 1 -> 0 -> 1, the DM's ndmreset). EX §8.5: the running launch got no record (ANSWERED :=
    // TAKEN), queued descriptors stay and are served after the boot. HELLO r1 = TAKEN at the post:
    // every outstanding launch below it is lost. The record goes to the event callback.
    void hello_outside_boot(const tsr_record & h) {
        const uint32_t taken = h.w[TSR_LCQ_HELLO_TAKEN_WORD];
        const uint64_t id = (uint64_t) h.w[TSR_LCQ_HELLO_ID_HI_WORD] << 32 | h.w[TSR_LCQ_HELLO_ID_LO_WORD];
        msg("HELLO outside a boot: the RISC-V restarted (a RISC-V reset behind the transport)%s",
            img != nullptr && id != tsr_image_id(img) ? ", and it runs another image than the one booted" : "");
        std::vector<uint32_t> gone;
        while (!outstanding.empty() && (int32_t) (taken - outstanding.front().seq) > 0) {
            gone.push_back(outstanding.front().seq);
            remember_lost(outstanding.front().seq);
            outstanding.pop_front();
        }
        if (!gone.empty())
            msg("%zu launch%s taken before the RISC-V reset got no answer (EX §8.5): seq %u..%u", gone.size(), gone.size() == 1 ? "" : "es",
                gone.front(), gone.back());
        ev_pending.push_back(h);
    }

    void deliver_events() {
        while (!ev_pending.empty()) {
            const tsr_record r = ev_pending.front();
            ev_pending.pop_front();
            if (event_cb) event_cb(event_user, &r);
        }
    }

    int consume_and_handle() {
        const int rc = consume();
        deliver_events();
        if (rc < 0) return rc;
        if (trap_pending) return handle_trap();
        return 0;
    }

    // ---- polling (EX §4.8, SW §9.1, DB §4.1) ----
    // Releases the mutex between polls, letting the device time pass (idle()). debug_wait: the
    // RISC-V is halted or a tile stop is active, so only a debugger (another thread) moves things on:
    // the simulated kinds then also give the host 100 µs per poll instead of spinning.
    void between_polls(uint32_t iter, uint64_t interval, bool debug_wait = false) {
        const bool simulated = dev->simulated();
        if (simulated) dev->idle(iter, interval);   // drives the model (in-process or the server): needs the lock
        const int d = mu.release_all();
        if (!simulated) dev->idle(iter, interval);   // host sleep
        else if (debug_wait) std::this_thread::sleep_for(std::chrono::microseconds(100));
        else if (mu.waiters.load() > 0) std::this_thread::sleep_for(std::chrono::microseconds(50));
        mu.reacquire(d);
    }

    // Polls until pred() holds after a consume. expect_ns: the busy-poll span (the last DURATION
    // of the same kernel). timeout_ns: UINT64_MAX = none. debug_aware: count time only while the
    // RISC-V is not halted and no tile stop is active, and not at all in serial mode (DB §4.1).
    // 0 when pred() held; TSR_TE_TIMEOUT; other errors.
    template <class P> int poll_until(P pred, uint64_t timeout_ns, uint64_t expect_ns, bool debug_aware) {
        const uint64_t t0 = now();
        uint64_t prev = t0, counted = 0;
        for (uint32_t iter = 0;; ++iter) {
            const int rc = consume_and_handle();
            if (rc < 0) return rc;
            if (pred()) return 0;
            if (state == dstate::failed) return TSR_TE_DEVICE;
            maybe_drain_log();
            const uint64_t t = now(), elapsed = t - t0;
            // Busy-poll while below the expected duration (and the timeout), unless the time base
            // did not move (the simulator kinds' reads run the model only while something can
            // happen). The debug-aware state is read, and a timeout declared, only at backoff polls.
            const bool backoff = elapsed >= std::min(expect_ns, timeout_ns) || t == prev;
            bool counting = true, serial = false;
            if (backoff && debug_aware) {
                uint32_t s = 0;
                if (!status(&s)) return state == dstate::failed ? TSR_TE_DEVICE : TSR_TE_IO;
                if (s & TSR_STATUS_RV_HALTED) counting = false;
                if (has_dbg && dbg_active) {
                    if (rdv(TSR_CSR_DBG_STATUS) & TSR_DBG_STATUS_STOP_ACTIVE) counting = false;
                    if (rdv(TSR_CSR_DBG_CFG) & TSR_DBG_CFG_SERIAL) serial = true;
                }
                if ((s & TSR_STATUS_PLAT_ERR) && !plat_err_reported) plat_err_seen("STATUS.PLAT_ERR while waiting");
            }
            if (counting) counted += t - prev;
            prev = t;
            if (backoff && timeout_ns != UINT64_MAX && !serial && counted >= timeout_ns) return TSR_TE_TIMEOUT;
            if (backoff) {
                uint64_t interval = std::max(POLL_MIN_NS, std::max(expect_ns, elapsed) / 100);
                interval = std::min(interval, POLL_MAX_NS);
                if (timeout_ns != UINT64_MAX && timeout_ns > counted) interval = std::min(interval, std::max(POLL_MIN_NS, timeout_ns - counted));
                between_polls(iter, interval, !counting || serial);
            } else if (mu.waiters.load() > 0) {
                const int d = mu.release_all();
                std::this_thread::yield();
                mu.reacquire(d);
            }
        }
    }

    uint64_t expect_for(uint32_t key) const {
        auto it = last_dur.find(key);
        return it == last_dur.end() ? 0 : it->second;
    }

    // ---- SOFT_RESET (EX §8.5) ----
    // keep_posted: consume the records and drain the log words posted before the flush (LCQ_RD :=
    // LCQ_WR, LOG_RD := LOG_WR); false at open, where they are an earlier process's.
    int soft_reset_hw(const char * why, bool keep_posted = true) {
        if (state == dstate::failed) return TSR_TE_DEVICE;
        if (keep_posted) {
            consume();
            deliver_events();
            if (trap_pending) {
                trap_pending = false;
                const std::string rep = trap_report(img, trap_rec);
                rlog.last_trap = rep;
                msg("%s (found while resetting)", rep.c_str());
            }
            drain_log();
        }
        uint32_t ctrl = 0;
        rd(TSR_CSR_CTRL, &ctrl);
        const uint32_t levels = ctrl & (TSR_CTRL_HOST_HOLD | TSR_CTRL_HOLD_ON_HALT | TSR_CTRL_IRQ_EN);
        if (!wr(TSR_CSR_CTRL, levels | TSR_CTRL_SOFT_RESET)) return TSR_TE_IO;
        uint32_t s = 0;
        const uint64_t t0 = now();
        for (uint32_t iter = 0;; ++iter) {
            if (!status(&s)) return state == dstate::failed ? TSR_TE_DEVICE : TSR_TE_IO;
            if (!(s & TSR_STATUS_SR_BUSY)) break;
            if (now() - t0 >= reset_ms * MS) {
                hang_dump("SOFT_RESET did not complete");
                fail(std::string("SOFT_RESET did not complete within TSR_RESET_MS (") + std::to_string(reset_ms) + " ms, " + tbase() +
                     " time)" + ((s & TSR_STATUS_UNIT_HUNG) ? ": a unit is hung (STATUS.UNIT_HUNG): card reset needed (xrt-smi reset)"
                                                             : ": a unit does not drain: card reset needed (xrt-smi reset)"));
                return TSR_TE_DEVICE;
            }
            between_polls(iter, std::max(POLL_MIN_NS, std::min<uint64_t>((now() - t0) / 100, POLL_MAX_NS)));
        }
        if (!(s & TSR_STATUS_RV_CLK_OK) || (s & TSR_STATUS_SR_RV_TIMEOUT)) {
            char b[96];
            snprintf(b, sizeof(b), "no clk_rv: the RISC-V reset did not complete during SOFT_RESET (STATUS 0x%08x)", s);
            fail(b);
            return TSR_TE_DEVICE;
        }
        lose_outstanding(why);
        trap_pending = false;
        plat_err_try_clear();
        plat_err_reported = false;
        wm_cancel.valid = wm_stop.valid = false;   // EX §4.9: SOFT_RESET clears both VALID bits
        // The flushed counters (EX §8.5): PUSHED is kept, LCQ_RD := LCQ_WR, LOG_RD := LOG_WR.
        if (!rd(TSR_CSR_LQ_SEQ, &pushed) || !rd(TSR_CSR_LCQ_RD, &lcq_rd) || !rd(TSR_CSR_LOG_RD, &log_rd)) return TSR_TE_IO;
        dec.reset();
        if (state != dstate::attached) state = dstate::reset;
        return 0;
    }

    // ---- boot (EX §3.3-§3.5) ----
    int check_requirements(const tsr_image * im) {
        const image_note n = image_get_note(im);
        std::string why;
        if (n.sdk_abi != TSR_ELF_SDK_ABI) why += " SDK ABI " + std::to_string(n.sdk_abi) + " (this host speaks " +
                                                        std::to_string(TSR_ELF_SDK_ABI) + ");";
        if (n.ctl_id_min > TSR_GET(r_ctl_id, TSR_CTL_ID_VERSION))
            why += " control-core version " + std::to_string(n.ctl_id_min) + " (device CTL_ID[7:0] " +
                   std::to_string(TSR_GET(r_ctl_id, TSR_CTL_ID_VERSION)) + ");";
        if (n.itcm_min > itcm_bytes) why += " ITCM " + std::to_string(n.itcm_min) + " B (device " + std::to_string(itcm_bytes) + ");";
        if (n.dtcm_min > dtcm_bytes) why += " DTCM " + std::to_string(n.dtcm_min) + " B (device " + std::to_string(dtcm_bytes) + ");";
        if (n.caps_req & ~r_caps) {
            char b[96];
            snprintf(b, sizeof(b), " CAPS 0x%08x (device 0x%08x);", n.caps_req, r_caps);
            why += b;
        }
        for (const image_seg & g : image_segments(im)) {
            const uint64_t end = (uint64_t) g.paddr + g.bytes.size();
            const bool fits = g.itcm ? end <= itcm_bytes : (g.paddr >= TSR_RV_DTCM_BASE && end <= (uint64_t) TSR_RV_DTCM_BASE + dtcm_bytes);
            if (!fits) {
                char b[96];
                snprintf(b, sizeof(b), " segment 0x%08x + %zu B outside the device TCMs;", g.paddr, g.bytes.size());
                why += b;
            }
        }
        if (!why.empty()) {
            msg("image %s refused: the device does not meet its requirements (.note.tsr, EX §3.1):%s", tsr_image_path(im), why.c_str());
            return TSR_TE_IMAGE;
        }
        return 0;
    }

    int wait_load_status(uint64_t t0) {
        for (uint32_t iter = 0;; ++iter) {
            uint32_t ls = 0;
            if (!rd(TSR_CSR_LOAD_STATUS, &ls)) return TSR_TE_IO;
            if (!(ls & TSR_LOAD_STATUS_VERIFY_BUSY)) return 0;
            if (now() - t0 >= boot_ms * MS) {
                msg("VERIFY still busy after TSR_BOOT_MS (%" PRIu64 " ms)", boot_ms);
                return TSR_TE_DEVICE;
            }
            between_polls(iter, POLL_MIN_NS);
        }
    }

    // VERIFY of one segment (EX §3.3): the CRC of what the TCM holds.
    int verify(const image_seg & g, uint32_t * crc, uint64_t t0) {
        if (!wr(TSR_CSR_LOAD_ADDR, g.paddr) || !wr(TSR_CSR_LOAD_COUNT, (uint32_t) (g.bytes.size() / 4)) ||
            !wr(TSR_CSR_LOAD_CTRL, TSR_LOAD_CTRL_VERIFY))
            return TSR_TE_IO;
        const int rc = wait_load_status(t0);
        if (rc) return rc;
        return rd(TSR_CSR_LOAD_CRC, crc) ? 0 : TSR_TE_IO;
    }

    // Cold load of one segment: the words, the receive CRC, VERIFY.
    int load(const image_seg & g, uint64_t t0) {
        if (!wr(TSR_CSR_LOAD_ADDR, g.paddr)) return TSR_TE_IO;
        const uint8_t * b = g.bytes.data();
        for (size_t i = 0; i < g.bytes.size(); i += 4) {
            const uint32_t w = (uint32_t) b[i] | (uint32_t) b[i + 1] << 8 | (uint32_t) b[i + 2] << 16 | (uint32_t) b[i + 3] << 24;
            if (!wr(TSR_CSR_LOAD_DATA, w)) return TSR_TE_IO;
        }
        binfo.words_written += (uint32_t) (g.bytes.size() / 4);
        uint32_t rx = 0;
        if (!rd(TSR_CSR_LOAD_CRC, &rx)) return TSR_TE_IO;
        if (rx != g.crc) {
            msg("segment 0x%08x (%zu B): receive CRC 0x%08x, expected 0x%08x: words were lost or changed on the way (PCIe / "
                "AXI-Lite / LOAD_TIMEOUT; STATUS 0x%08x)", g.paddr, g.bytes.size(), rx, g.crc, rdv(TSR_CSR_STATUS));
            return TSR_TE_IMAGE;
        }
        uint32_t vc = 0;
        const int rc = verify(g, &vc, t0);
        if (rc) return rc;
        if (vc != g.crc) {
            msg("segment 0x%08x (%zu B): VERIFY CRC 0x%08x, expected 0x%08x: the TCM does not hold what was written (a TCM "
                "write fault or an address decode fault, EX §3.3; STATUS 0x%08x)", g.paddr, g.bytes.size(), vc, g.crc,
                rdv(TSR_CSR_STATUS));
            return TSR_TE_IMAGE;
        }
        return 0;
    }

    int boot_img(const tsr_image * im) {
        if (state == dstate::failed) return TSR_TE_DEVICE;
        int rc = check_requirements(im);
        if (rc) return rc;
        uint32_t s = 0;
        if (!status(&s)) return state == dstate::failed ? TSR_TE_DEVICE : TSR_TE_IO;
        if (state != dstate::reset || (s & (TSR_STATUS_RV_RUNNING | TSR_STATUS_SR_BUSY))) {
            rc = soft_reset_hw("boot");
            if (rc) return rc;
            state = dstate::reset;
        }
        boot_info bi;
        binfo = bi;
        // The simulated kinds' launch capture (TSR_TRACE, SW §5.5) stores the image of each launch:
        // give it the ELF itself, so a capture carries its .tsr_id / .note.tsr and opens with
        // tsr_image_open anywhere (else the capture synthesises one from the loader's writes).
        if (tsr_sim * sm = dev->sim()) {
            size_t n = 0;
            const uint8_t * b = tsr_elf_bytes(image_elf(im), &n);
            tsr_sim_capture_set_image(sm, b, n);
        }
        const uint64_t t0 = now(), w0 = wall_ns();
        if (!wr(TSR_CSR_STATUS, TSR_STATUS_LOAD_REFUSED | TSR_STATUS_LOAD_TIMEOUT)) return TSR_TE_IO;
        // EX §3.5: VERIFY every ITCM segment; write only those that do not match (warm restart
        // when none). Every DTCM segment is reloaded and verified.
        for (const image_seg & g : image_segments(im)) {
            if (!g.itcm) continue;
            uint32_t crc = 0;
            if ((rc = verify(g, &crc, t0)) != 0) return rc;
            if (crc == g.crc) continue;
            if ((rc = load(g, t0)) != 0) return rc;
            binfo.itcm_loaded++;
        }
        for (const image_seg & g : image_segments(im)) {
            if (g.itcm) continue;
            if ((rc = load(g, t0)) != 0) return rc;
            binfo.dtcm_loaded++;
        }
        if (!status(&s)) return state == dstate::failed ? TSR_TE_DEVICE : TSR_TE_IO;
        if (s & (TSR_STATUS_LOAD_REFUSED | TSR_STATUS_LOAD_TIMEOUT)) {
            msg("the loader reported %s%s (STATUS 0x%08x)", (s & TSR_STATUS_LOAD_REFUSED) ? "LOAD_REFUSED " : "",
                (s & TSR_STATUS_LOAD_TIMEOUT) ? "LOAD_TIMEOUT" : "", s);
            return TSR_TE_IMAGE;
        }
        binfo.warm = binfo.itcm_loaded == 0;
        // EX §3.4: boot.
        uint32_t ctrl = 0;
        rd(TSR_CSR_CTRL, &ctrl);
        const uint32_t levels = ctrl & (TSR_CTRL_HOST_HOLD | TSR_CTRL_HOLD_ON_HALT | TSR_CTRL_IRQ_EN);
        if (!wr(TSR_CSR_RV_BOOT, 0) || !wr(TSR_CSR_CTRL, levels | TSR_CTRL_RV_RUN)) return TSR_TE_IO;
        if ((rc = wait_hello(im, t0)) != 0) return rc;
        binfo.time_ns = now() - t0;
        binfo.wall_ns = wall_ns() - w0;
        if (img != im) {
            image_ref(im);
            image_unref(img);
            img = im;
        }
        dec.set_image(img);
        state = dstate::ready;
        return 0;
    }

    int wait_hello(const tsr_image * im, uint64_t t0) {
        for (uint32_t iter = 0;; ++iter) {
            uint32_t v = 0;
            if (!rd(TSR_CSR_LCQ_WR, &v)) return TSR_TE_IO;
            if (v != lcq_rd) {
                tsr_record h;
                const bool save = full_records;
                full_records = true;
                read_record(lcq_rd, &h);
                full_records = save;
                if (!wr(TSR_CSR_LCQ_RD, lcq_rd + 1)) return TSR_TE_IO;
                lcq_rd++;
                const uint32_t type = TSR_GET(h.w[0], TSR_LCQ_R0_TYPE);
                if (type == TSR_LCQ_R0_TYPE_TRAP) {
                    drain_log();
                    const std::string rep = trap_report(im, h);
                    rlog.last_trap = rep;
                    rlog.traps++;
                    msg("the firmware trapped during boot: %s", rep.c_str());
                    print_log_tail(16, "before the trap", 0);
                    return TSR_TE_IMAGE;
                }
                if (type != TSR_LCQ_R0_TYPE_HELLO) {
                    msg("record type 0x%02x (r0 0x%08x) before HELLO: not the runtime of EX §4.7 R1", type, h.w[0]);
                    return TSR_TE_IMAGE;
                }
                const uint64_t id = (uint64_t) h.w[TSR_LCQ_HELLO_ID_HI_WORD] << 32 | h.w[TSR_LCQ_HELLO_ID_LO_WORD];
                const uint32_t abi = TSR_GET(h.w[TSR_LCQ_HELLO_VERSIONS_WORD], TSR_HELLO_R4_SDK_ABI);
                const uint32_t nk = h.w[TSR_LCQ_HELLO_N_KERNELS_WORD];
                std::string why;
                char b[160];
                if (id != tsr_image_id(im)) {
                    snprintf(b, sizeof(b), " image id 0x%016" PRIx64 ", expected 0x%016" PRIx64 ";", id, tsr_image_id(im));
                    why += b;
                }
                if (abi != image_get_note(im).sdk_abi) {
                    snprintf(b, sizeof(b), " SDK ABI %u, expected %u;", abi, image_get_note(im).sdk_abi);
                    why += b;
                }
                if (nk != image_n_kernels(im)) {
                    snprintf(b, sizeof(b), " %u kernels, expected %u;", nk, image_n_kernels(im));
                    why += b;
                }
                if (!why.empty()) {
                    msg("HELLO does not match the image %s:%s the device runs another image or the load failed (EX §3.4)",
                        tsr_image_path(im), why.c_str());
                    return TSR_TE_IMAGE;
                }
                binfo.hello_cycles = h.w[TSR_LCQ_HELLO_BOOT_CYCLES_WORD];
                return 0;
            }
            maybe_drain_log();
            const uint64_t el = now() - t0;
            if (el >= boot_ms * MS) {
                const uint32_t rt = rdv(TSR_CSR_RT_STATE);
                msg("no HELLO within TSR_BOOT_MS (%" PRIu64 " ms, %s time): STATUS 0x%08x, RT_STATE %s, RT_INFO 0x%08x", boot_ms,
                    tbase(), rdv(TSR_CSR_STATUS), rt_phase_name(TSR_GET(rt, TSR_RT_STATE_PHASE)), rdv(TSR_CSR_RT_INFO));
                drain_log();
                print_log_tail(16, "during boot", 0);
                return TSR_TE_TIMEOUT;
            }
            between_polls(iter, std::max(POLL_MIN_NS, std::min<uint64_t>(el / 100, POLL_MAX_NS)));
        }
    }

    // ---- TRAP (EX §8.6, DB §6.3) ----
    int handle_trap() {
        trap_pending = false;
        rlog.traps++;
        drain_log();
        const std::string rep = trap_report(img, trap_rec);
        rlog.last_trap = rep;
        if (trap_is_answer) msg("launch seq %u: %s", trap_rec.w[TSR_LCQ_TRAP_SEQ_WORD], rep.c_str());
        else msg("%s (between launches)", rep.c_str());
        print_log_tail(16, "before the trap", trap_rec.w[TSR_LCQ_TRAP_SEQ_WORD]);
        if (gdb_wait) {
            state = dstate::trapped;
            msg("TSR_GDB_WAIT is set: the device stays TRAPPED for the debugger (RT_STATE TRAPPED; the GDB service is P3.7); "
                "tsr_transport_soft_reset + tsr_transport_boot restart it");
            return 0;
        }
        return restart("trap");
    }

    int restart(const char * why) {
        const tsr_image * im = img;
        image_ref(im);
        int rc = soft_reset_hw(why);
        if (rc == 0 && im != nullptr) {
            rc = boot_img(im);
            if (rc == 0) {
                rlog.restarts++;
                msg("warm restart after %s: %s boot in %.3f ms (%s time)", why, binfo.warm ? "warm" : "cold",
                    (double) binfo.time_ns / 1e6, tbase());
            } else {
                msg("restart after %s failed (%s); the device is not booted", why, tsr_transport_strerror(rc));
            }
        }
        image_unref(im);
        return rc;
    }

    // ---- diagnosis (DB §7.2) and the hang dump (DB §7.4) ----
    std::string triage(uint32_t s) {
        std::string t;
        char b[256];
        const uint32_t rt = rdv(TSR_CSR_RT_STATE), info0 = rdv(TSR_CSR_RT_INFO);
        const uint32_t ph = TSR_GET(rt, TSR_RT_STATE_PHASE), k = TSR_GET(rt, TSR_RT_STATE_KERNEL);
        snprintf(b, sizeof(b), "STATUS 0x%08x, RT_STATE %s", s, rt_phase_name(ph));
        t = b;
        if (ph >= TSR_RT_STATE_PHASE_DISPATCH && ph <= TSR_RT_STATE_PHASE_ANSWER) {
            const char * kn = img ? image_kernel_name(img, k) : "";
            snprintf(b, sizeof(b), " kernel %u%s%s%s", k, kn[0] ? " (" : "", kn, kn[0] ? ")" : "");
            t += b;
        }
        uint32_t running = 0, hung = 0;
        for (uint32_t u = 0; u < TSR_NUM_UNITS; ++u) {
            const uint32_t q = rdv(TSR_CSR_Q_STATE(u));
            if (q & TSR_Q_STATE_RUNNING) running |= 1u << u;
            if (q & TSR_Q_STATE_HUNG) hung |= 1u << u;
        }
        between_polls(0, MS);   // 1 ms: does RT_INFO move?
        const uint32_t info1 = rdv(TSR_CSR_RT_INFO);
        snprintf(b, sizeof(b), ", RT_INFO 0x%08x%s, units running 0x%02x, hung 0x%02x", info1, info0 != info1 ? " (moving)" : " (frozen)",
                 running, hung);
        t += b;
        const char * verdict;
        if (s & TSR_STATUS_UNIT_HUNG) verdict = "a unit is hung without progress (card reset)";
        else if (s & TSR_STATUS_RV_HALTED) verdict = "the RISC-V is halted (debugger, breakpoint)";
        else if (ph == TSR_RT_STATE_PHASE_TRAPPED) verdict = "the firmware trapped";
        else if (!(s & TSR_STATUS_RV_RUNNING)) verdict = "the RISC-V is not running";
        else if (running) verdict = "a unit is running";
        else if (s & TSR_STATUS_RV_SLEEP) verdict = "the RISC-V sleeps (WFI) waiting for a unit or the host";
        else if (info0 != info1) verdict = "the RISC-V is progressing";
        else verdict = "RT_STATE / RT_INFO frozen with no unit running: the RISC-V is stuck in a loop or a join";
        return t + "; " + verdict;
    }

    void hang_dump(const char * why) {
        if (dump_dir.empty()) return;
        char sub[64];
        snprintf(sub, sizeof(sub), "/hang-%ld-%u", (long) getpid(), ++dumps);
        const std::string dir = dump_dir + sub;
        if (const int e = make_dirs(dir)) {
            msg("TSR_DUMP_DIR: cannot create %s: %s", dir.c_str(), strerror(e));
            return;
        }
        // state.json: every EX, Q_*, PERF and DBG register raw (DMI excluded: reads have side effects),
        // the adapter registers, the LCQ ring.
        if (FILE * f = fopen((dir + "/state.json").c_str(), "w")) {
            fprintf(f, "{\n  \"why\": \"%s\",\n  \"kind\": \"%s\",\n  \"time_ns\": %" PRIu64 ",\n  \"time_base\": \"%s\",\n", why, kind.c_str(),
                    now(), tbase());
            fprintf(f, "  \"image\": \"%s\",\n  \"image_id\": \"0x%016" PRIx64 "\",\n  \"state\": \"%s\",\n", img ? tsr_image_path(img) : "",
                    img ? tsr_image_id(img) : 0, state_name(state));
            fprintf(f, "  \"host\": {\"pushed\": %u, \"lcq_rd\": %u, \"log_rd\": %u, \"outstanding\": [", pushed, lcq_rd, log_rd);
            for (size_t i = 0; i < outstanding.size(); ++i) fprintf(f, "%s%u", i ? ", " : "", outstanding[i].seq);
            fprintf(f, "]},\n  \"csr\": {");
            bool first = true;
            auto one = [&](uint32_t off) {
                fprintf(f, "%s\n    \"0x%03x\": \"0x%08x\"", first ? "" : ",", off, rdv(off));
                first = false;
            };
            for (uint32_t off = 0; off < 0x100; off += 4) {
                if (off == TSR_CSR_LQ_PUSH || off == TSR_CSR_LOAD_CTRL) continue;   // write-only
                one(off);
            }
            for (uint32_t off = TSR_CSR_LCQ_REC_BASE; off < TSR_CSR_LCQ_REC_BASE + 32 * lcq_depth; off += 4) one(off);
            for (uint32_t off = 0xE00; off < 0xF20; off += 4) one(off);
            if (has_dbg)
                for (uint32_t off = 0x800; off < 0xB00; off += 4) one(off);
            fprintf(f, "\n  },\n  \"perf\": [");
            for (uint32_t i = 0; i < TSR_PERF_SLOTS; ++i) {
                const uint32_t lo = rdv(TSR_CSR_PERF_LO(i)), hi = rdv(TSR_CSR_PERF_HI(i));
                fprintf(f, "%s%" PRIu64, i ? ", " : "", (uint64_t) hi << 32 | lo);
            }
            fprintf(f, "]");
            if (dev->has_adapter()) {
                fprintf(f, ",\n  \"adapter\": {");
                for (uint32_t c = 0; c < cfg.num_ch; ++c) {
                    uint32_t lo = 0, hi = 0;
                    ard(KREG_CH_BASE0 + 8 * c, &lo);
                    ard(KREG_CH_BASE0 + 8 * c + 4, &hi);
                    fprintf(f, "\"ch%u_base\": \"0x%08x%08x\", ", c, hi, lo);
                }
                uint32_t ac = 0, as = 0, av = 0;
                ard(KREG_ADAPTER_CTRL, &ac);
                ard(KREG_ADAPTER_STATUS, &as);
                ard(KREG_ADAPTER_VERSION, &av);
                fprintf(f, "\"ADAPTER_CTRL\": \"0x%08x\", \"ADAPTER_STATUS\": \"0x%08x\", \"ADAPTER_VERSION\": \"0x%08x\"}", ac, as, av);
            }
            fprintf(f, "\n}\n");
            fclose(f);
        }
        if (FILE * f = fopen((dir + "/units.txt").c_str(), "w")) {
            static const char * const un[6] = { "DMA", "QACT", "MVM", "VU", "ATT", "SYNC" };
            for (uint32_t u = 0; u < TSR_NUM_UNITS; ++u) {
                const uint32_t q = rdv(TSR_CSR_Q_STATE(u));
                fprintf(f, "%-4s depth %2u entries %2u%s%s%s%s%s done %u issued %u err seq %u code %u info 0x%08x w0 0x%08x pidx %u\n", un[u],
                        (r_q_depth >> (5 * u)) & 31u, q & TSR_Q_STATE_ENTRIES, (q & TSR_Q_STATE_RUNNING) ? " RUNNING" : "",
                        (q & TSR_Q_STATE_DEP_WAIT) ? " DEP_WAIT" : "", (q & TSR_Q_STATE_HELD) ? " HELD" : "",
                        (q & TSR_Q_STATE_ERR) ? " ERR" : "", (q & TSR_Q_STATE_HUNG) ? " HUNG" : "", rdv(TSR_CSR_Q_DONE(u)),
                        rdv(TSR_CSR_Q_ISSUED(u)), rdv(TSR_CSR_Q_ERR_SEQ(u)), rdv(TSR_CSR_Q_ERR_CODE(u)), rdv(TSR_CSR_Q_ERR_INFO(u)),
                        rdv(TSR_CSR_Q_ERR_W0(u)), rdv(TSR_CSR_Q_ERR_PIDX(u)));
            }
            fprintf(f, "issue FIFO %u, Q_PIDX %u\n", rdv(TSR_CSR_Q_FIFO), rdv(TSR_CSR_Q_PIDX));
            fclose(f);
        }
        drain_log();
        if (FILE * f = fopen((dir + "/log.txt").c_str(), "w")) {
            for (const auto & l : log_tail) fprintf(f, "%s\n", l.second.c_str());
            fclose(f);
        }
        if (has_dbg) {   // five RV samples 1 ms apart (DB §6.2, §7.4)
            if (FILE * f = fopen((dir + "/rv_samples.txt").c_str(), "w")) {
                for (int i = 0; i < 5; ++i) {
                    wr(TSR_CSR_DBG_CTRL, TSR_DBG_CTRL_RV_SAMPLE);
                    const uint32_t stt = rdv(TSR_CSR_DBG_RV_STATE), pc = rdv(TSR_CSR_DBG_RV_PC), daddr = rdv(TSR_CSR_DBG_RV_DADDR);
                    fprintf(f, "state 0x%08x pc %s npc 0x%08x daddr 0x%08x epc 0x%08x eaddr 0x%08x\n", stt, image_symbolize(img, pc).c_str(),
                            rdv(TSR_CSR_DBG_RV_NPC), daddr, rdv(TSR_CSR_DBG_RV_EPC), rdv(TSR_CSR_DBG_RV_EADDR));
                    between_polls(0, MS);
                }
                fclose(f);
            }
        }
        rlog.dumps++;
        rlog.last_dump_dir = dir;
        msg("hang dump written to %s (DB §7.4; history, trace ring and SBA reads are tsr-dbg's, P3.7)", dir.c_str());
    }

    // ---- DB §7.3 recovery ladder, entered by a wait timeout ----
    // take: hand the timed-out launch's answer (if the cancel produced one) to *rec and forget the
    // launch (its wait returns TSR_TE_TIMEOUT); else leave it for a later wait (the credit path).
    int ladder(uint32_t seq, uint64_t waited_ns, tsr_record * rec, bool take) {
        rlog.timeouts++;
        uint32_t s = 0;
        if (!status(&s)) return state == dstate::failed ? TSR_TE_DEVICE : TSR_TE_IO;
        msg("launch seq %u timed out after %.3f ms (%s time): %s", seq, (double) waited_ns / 1e6, tbase(), triage(s).c_str());
        drain_log();
        print_log_tail(16, "at the timeout", seq);
        hang_dump("wait timeout");   // level 0
        if (s & TSR_STATUS_UNIT_HUNG) {
            fail("a unit is hung (STATUS.UNIT_HUNG; SOFT_RESET cannot complete): card reset needed (xrt-smi reset), EX §8.6");
            return TSR_TE_DEVICE;
        }
        // Level 2: LQ_CANCEL covering every launch pushed so far (EX §4.9: QABORT of the running one).
        rlog.level2++;
        const uint32_t wm = pushed - 1;
        msg("recovery level 2: LQ_CANCEL = %u, waiting up to TSR_ABORT_MS (%" PRIu64 " ms) for %zu answer%s", wm, abort_ms,
            outstanding.size(), outstanding.size() == 1 ? "" : "s");
        if (!wr(TSR_CSR_LQ_CANCEL, wm)) return TSR_TE_IO;
        wm_written(TSR_CSR_LQ_CANCEL, wm);
        int rc = poll_until([&] { return outstanding.empty() || state != dstate::ready; }, abort_ms * MS, 0, false);
        if (rc < 0 && rc != TSR_TE_TIMEOUT) return rc;
        if (state == dstate::failed) return TSR_TE_DEVICE;
        if (state == dstate::ready && outstanding.empty()) {
            rlog.level2_ok++;
            msg("recovered at level 2: every outstanding launch answered");
        } else if (state == dstate::ready) {
            // Level 3: the RISC-V does not answer.
            rlog.level3++;
            msg("recovery level 3: no answer within TSR_ABORT_MS: SOFT_RESET + warm restart");
            rc = restart("timeout recovery");
            if (rc < 0) return rc;
        }
        if (take) {
            auto it = answered.find(seq);
            if (it != answered.end()) {
                if (rec) *rec = it->second;
                answered.erase(it);
            } else {
                lost.erase(seq);
            }
        }
        return TSR_TE_TIMEOUT;
    }

    // ---- wait (EX §4.8) ----
    int wait_seq(uint32_t seq, uint64_t timeout_ns, tsr_record * rec) {
        if (rec) memset(rec, 0, sizeof(*rec));
        auto take = [&]() -> int {
            auto it = answered.find(seq);
            if (it != answered.end()) {
                if (rec) *rec = it->second;
                answered.erase(it);
                return 0;
            }
            if (lost.erase(seq)) return TSR_TE_LOST;
            return 1;
        };
        int r = take();
        if (r <= 0) return r;
        if (!is_outstanding(seq)) {
            msg("wait: seq %u is not a launch of this transport, or it was waited for already", seq);
            return TSR_TE_INVAL;
        }
        if (state == dstate::failed) return TSR_TE_DEVICE;
        uint32_t key = 0;
        for (const launch_rec & l : outstanding)
            if (l.seq == seq) key = l.key;
        if (timeout_ns == TSR_TIMEOUT_DEFAULT) timeout_ns = launch_ms ? launch_ms * MS : UINT64_MAX;
        const uint64_t t0 = now();
        int rc = poll_until([&] { return answered.count(seq) || lost.count(seq) || state != dstate::ready; }, timeout_ns,
                            expect_for(key), true);
        if (rc == 0) {
            r = take();
            if (r <= 0) return r;
            return state == dstate::failed ? TSR_TE_DEVICE : TSR_TE_STATE;
        }
        if (rc == TSR_TE_TIMEOUT) return ladder(seq, now() - t0, rec, true);
        return rc;
    }

    // ---- launch (EX §4.8) ----
    int launch(const tsr_launch * l, uint32_t * seq) {
        if (state != dstate::ready) {
            msg("launch refused: the device is %s%s%s", state_name(state), fail_reason.empty() ? "" : ": ", fail_reason.c_str());
            return state == dstate::failed ? TSR_TE_DEVICE : TSR_TE_STATE;
        }
        // Host-side credit: at most LQ_DEPTH outstanding (EX §4.2: an overflow is then impossible).
        while (outstanding.size() >= lq_depth) {
            const launch_rec oldest = outstanding.front();
            const uint64_t t0 = now();
            const uint64_t tmo = launch_ms ? launch_ms * MS : UINT64_MAX;
            int rc = poll_until([&] { return outstanding.size() < lq_depth || state != dstate::ready; }, tmo, expect_for(oldest.key), true);
            if (rc == TSR_TE_TIMEOUT) {
                rc = ladder(oldest.seq, now() - t0, nullptr, false);   // the answers stay for the waits
                if (rc != TSR_TE_TIMEOUT) return rc;
            } else if (rc < 0) {
                return rc;
            }
            if (state != dstate::ready) return state == dstate::failed ? TSR_TE_DEVICE : TSR_TE_STATE;
        }
        if ((wm_cancel.valid || wm_stop.valid) && !refresh_watermarks()) return TSR_TE_IO;
        uint32_t w[8];
        w[0] = TSR_PUT(TSR_DESC_TYPE, l->type) | (uint32_t) l->flags << 8 | TSR_PUT(TSR_DESC_KERNEL, l->kernel);
        for (int i = 0; i < 7; ++i) w[1 + i] = l->arg[i];
        for (uint32_t i = 0; i < 8; ++i) {
            if (w[i] == lq_shadow[i]) continue;
            if (!wr(TSR_CSR_LQ_W(i), w[i])) return TSR_TE_IO;
            lq_shadow[i] = w[i];
        }
        if (!wr(TSR_CSR_LQ_PUSH, 1u)) return TSR_TE_IO;
        const uint32_t s = pushed++;
        const char * kn = (img && l->type == TSR_DESC_TYPE_LAUNCH) ? image_kernel_name(img, l->kernel) : "";
        const bool ret0 = !strcmp(kn, "graph") || !strcmp(kn, "replay");
        outstanding.push_back({ s, (uint32_t) l->type << 16 | l->kernel, wall_ns(), now(), ret0 });
        st.launches++;
        if (seq) *seq = s;
        return 0;
    }

    // ---- open (SW §9.1 open steps 1-3) ----
    int identity() {
        const char * nm = kind.c_str();
        if (dev->has_adapter()) {
            if (!ard(KREG_ADAPTER_VERSION, &r_adapter_version)) {
                msg("cannot read ADAPTER_VERSION");
                return TSR_TE_IO;
            }
            const uint32_t maj = TSR_GET(r_adapter_version, TSR_ADAPTER_VERSION_MAJOR);
            if (maj != TSR_ADAPTER_MAJOR) {
                msg("ADAPTER_VERSION 0x%08x: adapter major %u, this software implements %u (a v1 bitstream? its tools are in "
                    "build/v1)", r_adapter_version, maj, TSR_ADAPTER_MAJOR);
                return TSR_TE_DEVICE;
            }
            if (TSR_GET(r_adapter_version, TSR_ADAPTER_VERSION_NUM_CH) != dev->num_ch()) {
                msg("ADAPTER_VERSION.NUM_CH = %u, but the kernel has %u channel buffer%s bound",
                    TSR_GET(r_adapter_version, TSR_ADAPTER_VERSION_NUM_CH), dev->num_ch(), dev->num_ch() == 1 ? "" : "s");
                return TSR_TE_DEVICE;
            }
        }
        uint32_t id = 0;
        if (!rd(TSR_CSR_ID, &id)) {
            msg("cannot read the core ID through the CSR window at s_axi_control + 0x%x (the kernel must declare an "
                "s_axi_control range of 0x%x)", KREG_CSR_WINDOW, KREG_SPACE);
            return TSR_TE_IO;
        }
        if (id != TSR_ID_MAGIC) {
            msg("core ID reads 0x%08x, expected 0x%08x (\"TSR1\"): not a Tesseract kernel, or the CSR window at "
                "s_axi_control + 0x%x does not reach the core", id, TSR_ID_MAGIC, KREG_CSR_WINDOW);
            return TSR_TE_DEVICE;
        }
        bool ok = rd(TSR_CSR_VERSION, &r_version);
        uint32_t cfg0 = 0, cfg1 = 0, cfg2 = 0;
        ok = ok && rd(TSR_CSR_CFG0, &cfg0) && rd(TSR_CSR_CFG1, &cfg1) && rd(TSR_CSR_CFG2, &cfg2) && rd(TSR_CSR_CAPS, &r_caps) &&
             rd(TSR_CSR_CFG3, &r_cfg3) && rd(TSR_CSR_CTL_ID, &r_ctl_id) && rd(TSR_CSR_RV_CFG, &r_rv_cfg) &&
             rd(TSR_CSR_Q_DEPTH, &r_q_depth) && rd(TSR_CSR_RV_FREQ, &r_rv_freq) && rd(TSR_CSR_LQ_CFG, &r_lq_cfg) &&
             rd(TSR_CSR_PERF_INFO, &r_perf_info) && rd(TSR_CSR_DBG_ID, &r_dbg_id);
        if (!ok) return TSR_TE_IO;
        if ((r_version >> 8 & 0xFFu) != TSR_ISA_MAJOR) {
            msg("core ISA version %u.%u, this software implements major %u (a v1 bitstream reports 1; its tools are in build/v1)",
                (r_version >> 8) & 0xFFu, r_version & 0xFFu, TSR_ISA_MAJOR);
            return TSR_TE_DEVICE;
        }
        // CSR-visible parameters from CFG0..2 / CAPS; BEAT_BITS, RD_OUTSTANDING and ADDR_BEAT_W are
        // fixed by the U200 kernel contract (512-bit m_axi ports).
        tsr_config d = TSR_CONFIG_U200_V1;
        d.num_ch = cfg0 & 0xFFu;
        d.lanes = (cfg0 >> 8) & 0xFFFu;
        d.t_max = (cfg0 >> 20) & 0xFu;
        d.spm_bytes = cfg1 * TSR_LINE_BYTES;
        d.vl = cfg2 & 0xFFu;
        d.nsreg = (cfg2 >> 8) & 0xFFu;
        d.k_max = ((cfg2 >> 16) & 0xFFFFu) * 256u;
        d.caps = r_caps;
        if (d.num_ch < 1 || d.num_ch > TSR_NUM_CH_MAX || d.num_ch != dev->num_ch()) {
            msg("CFG0.NUM_CH = %u, but the kernel has %u channel buffer%s bound", d.num_ch, dev->num_ch(), dev->num_ch() == 1 ? "" : "s");
            return TSR_TE_DEVICE;
        }
        if (d.t_max < 1 || d.t_max > TSR_T_MAX_ARCH || d.lanes == 0 || d.spm_bytes == 0 || d.k_max == 0) {
            msg("implausible configuration CFG0 0x%08x CFG1 0x%08x CFG2 0x%08x", cfg0, cfg1, cfg2);
            return TSR_TE_DEVICE;
        }
        if (r_ctl_id == 0) {
            msg("CTL_ID = 0: no control core (EX §5.1)");
            return TSR_TE_DEVICE;
        }
        itcm_bytes = TSR_GET(r_rv_cfg, TSR_RV_CFG_ITCM_KIB) * 1024u;
        dtcm_bytes = TSR_GET(r_rv_cfg, TSR_RV_CFG_DTCM_KIB) * 1024u;
        if (itcm_bytes == 0 || dtcm_bytes == 0) {
            msg("RV_CFG 0x%08x: no ITCM or DTCM", r_rv_cfg);
            return TSR_TE_DEVICE;
        }
        if (r_q_depth == 0 || TSR_GET(r_q_depth, TSR_Q_DEPTH_SYNC) == 0) {
            msg("Q_DEPTH 0x%08x: no SYNC queue", r_q_depth);
            return TSR_TE_DEVICE;
        }
        if (r_rv_freq == 0) {
            msg("RV_FREQ = 0: no clk_rv (a bitstream or clock problem, EX §8.6)");
            return TSR_TE_DEVICE;
        }
        lq_depth = TSR_GET(r_lq_cfg, TSR_LQ_CFG_LQ_DEPTH);
        lcq_depth = TSR_GET(r_lq_cfg, TSR_LQ_CFG_LCQ_DEPTH);
        log_depth = TSR_GET(r_lq_cfg, TSR_LQ_CFG_LOG_DEPTH);
        if (lq_depth == 0 || lcq_depth == 0 || lcq_depth > TSR_LCQ_DEPTH_MAX || log_depth == 0 || log_depth > TSR_LOG_DEPTH_MAX) {
            msg("LQ_CFG 0x%08x: ring depths outside the CSR map (EX §4.2)", r_lq_cfg);
            return TSR_TE_DEVICE;
        }
        has_dbg = r_dbg_id != 0;
        // Usable bytes per channel: the smallest bound buffer (symmetric allocation), in beats, and
        // never more than the core addresses (CH_BYTES, 2^ADDR_BEAT_W beats).
        uint64_t chb = std::min<uint64_t>(KERNEL_CH_BYTES, (1ull << d.addr_beat_w) * TSR_BEAT_BYTES);
        for (uint32_t c = 0; c < d.num_ch; ++c) {
            chb = std::min(chb, dev->ch_bytes(c));
            if (dev->has_adapter() && dev->ch_addr(c) % 4096 != 0) {
                msg("channel %u buffer at device address 0x%" PRIx64 " is not 4 KiB aligned", c, dev->ch_addr(c));
                return TSR_TE_DEVICE;
            }
        }
        d.ch_bytes = chb & ~(uint64_t) (TSR_BEAT_BYTES - 1);
        if (d.ch_bytes == 0) {
            msg("channel buffers are empty");
            return TSR_TE_DEVICE;
        }
        cfg = d;
        (void) nm;
        return 0;
    }

    int adapter_open() {
        if (!dev->has_adapter()) return 0;
        uint32_t a = 0;
        if (!ard(KREG_ADAPTER_STATUS, &a)) return TSR_TE_IO;
        alog.at_open = a;
        if (a != 0) {
            msg("ADAPTER_STATUS = 0x%08x at open (left by an earlier run; cleared): %s", a, adapter_status_str(a).c_str());
            uint32_t after = 0;
            if (!awr(KREG_ADAPTER_STATUS, 0xFFFFFFFFu) || !ard(KREG_ADAPTER_STATUS, &after)) return TSR_TE_IO;
            if (after != 0) msg("ADAPTER_STATUS still 0x%08x after the W1C write", after);
        }
        return 0;
    }

    // EX §9.2: MEM_EN off (the core is idle after SOFT_RESET), the channel bases, MEM_EN on.
    int bind() {
        if (!dev->has_adapter()) return 0;
        if (!awr(KREG_ADAPTER_CTRL, 0)) return TSR_TE_IO;
        for (uint32_t c = 0; c < cfg.num_ch; ++c) {
            const uint64_t a = dev->ch_addr(c);
            if (!awr(KREG_CH_BASE0 + 8 * c, (uint32_t) a) || !awr(KREG_CH_BASE0 + 8 * c + 4, (uint32_t) (a >> 32))) return TSR_TE_IO;
        }
        if (!awr(KREG_ADAPTER_CTRL, TSR_ADAPTER_CTRL_MEM_EN)) return TSR_TE_IO;
        uint32_t ctl = 0, a = 0;
        if (!ard(KREG_ADAPTER_CTRL, &ctl) || !ard(KREG_ADAPTER_STATUS, &a)) return TSR_TE_IO;
        if (!(ctl & TSR_ADAPTER_CTRL_MEM_EN) || (a & (AST_BASE_ALIGN | AST_BASE_LOCKED))) {
            msg("bind failed: ADAPTER_CTRL 0x%08x, ADAPTER_STATUS 0x%08x: %s", ctl, a, adapter_status_str(a).c_str());
            awr(KREG_ADAPTER_STATUS, 0xFFFFFFFFu);
            return TSR_TE_DEVICE;
        }
        for (uint32_t c = 0; c < cfg.num_ch; ++c) {
            uint32_t lo = 0, hi = 0;
            if (!ard(KREG_CH_BASE0 + 8 * c, &lo) || !ard(KREG_CH_BASE0 + 8 * c + 4, &hi)) return TSR_TE_IO;
            if (((uint64_t) hi << 32 | lo) != dev->ch_addr(c)) {
                msg("bind failed: ch%u_base reads 0x%08x%08x, the buffer is at 0x%" PRIx64, c, hi, lo, dev->ch_addr(c));
                return TSR_TE_DEVICE;
            }
        }
        return 0;
    }

    int settings_from_env() {
        const char * who = kind.c_str();
        uint64_t v = 0;
        int r;
        if ((r = env_u64(who, "TSR_LAUNCH_TIMEOUT_MS", 0, 1ull << 40, &v)) < 0) return -1;
        if (r) launch_ms = v;
        if ((r = env_u64(who, "TSR_ABORT_MS", 1, 1ull << 40, &v)) < 0) return -1;
        if (r) abort_ms = v;
        if ((r = env_u64(who, "TSR_BOOT_MS", 1, 1ull << 40, &v)) < 0) return -1;
        if (r) boot_ms = v;
        if ((r = env_u64(who, "TSR_RESET_MS", 1, 1ull << 40, &v)) < 0) return -1;
        if (r) reset_ms = v;
        if ((r = env_u64(who, "TSR_ATTACH", 0, 1, &v)) < 0) return -1;
        attach = r && v;
        if ((r = env_u64(who, "TSR_STATS", 0, 1, &v)) < 0) return -1;
        stats_on = r && v;
        if ((r = env_u64(who, "TSR_LOG_LEVEL", 0, 5, &v)) < 0) return -1;
        if (r) log_level = (int) v;
        if ((r = env_u64(who, "TSR_TILE_SERIAL", 0, 1, &v)) < 0) return -1;
        if (r) tile_serial = (int) v;
        if ((r = env_u64(who, "TSR_TILE_HIST", 0, 1, &v)) < 0) return -1;
        if (r) tile_hist = (int) v;
        if (const char * e = getenv("TSR_LOG"); e != nullptr && e[0] != '\0') {
            if (strcmp(e, "stderr") == 0) {
                log_out = stderr;
            } else {
                log_out = fopen(e, "a");
                if (log_out == nullptr) {
                    msg("TSR_LOG=\"%s\": cannot open: %s", e, strerror(errno));
                    return -1;
                }
                log_close = true;
            }
        }
        if (const char * e = getenv("TSR_DUMP_DIR"); e != nullptr && e[0] != '\0') dump_dir = e;
        if (const char * e = getenv("TSR_GDB_WAIT"); e != nullptr && e[0] != '\0') {
            if (strcmp(e, "boot") != 0 && strcmp(e, "launch") != 0) {
                msg("TSR_GDB_WAIT=\"%s\": expected boot or launch", e);
                return -1;
            }
            gdb_wait = true;
        }
        static std::once_flag noted;
        std::call_once(noted, [this] {
            for (const char * n : { "TSR_GDB", "TSR_DEBUGGER" })
                if (const char * e = getenv(n); e != nullptr && e[0] != '\0')
                    msg("%s is set: the embedded debug service arrives with libtsr_dbg (P3.7); ignored", n);
            for (const char * n : { "TSR_PROF", "TSR_PROF_TRW_MIB", "TSR_PROF_EVERY", "TSR_PROF_RV_SAMPLE_US" })
                if (const char * e = getenv(n); e != nullptr && e[0] != '\0')
                    msg("%s is set: the trace-writer profiler arrives with tsr-prof trw (P3.8); ignored", n);
            // EX §10.2: renamed and removed v1 variables.
            static const char * const renamed[][2] = { { "TSR_XRT_TIMEOUT_MS", "TSR_LAUNCH_TIMEOUT_MS" }, { "TSR_XRT_ABORT_MS", "TSR_ABORT_MS" } };
            for (const auto & rn : renamed)
                if (const char * e = getenv(rn[0]); e != nullptr && e[0] != '\0')
                    msg("%s is a v1 variable, renamed %s in v2 (EX §10.2); ignored", rn[0], rn[1]);
            if (const char * e = getenv("TSR_XRT_CTRL"); e != nullptr && e[0] != '\0')
                msg("TSR_XRT_CTRL is a v1 variable (the run protocol), removed in v2 (EX §10.2); ignored");
        });
        return 0;
    }

    // DB §7.3, EX §8.5: DB's configuration (SERIAL, the stop sources, the TBPs) survives
    // SOFT_RESET, so a debug session before this open may have left it armed. The waits must then
    // see its stops and serial mode, or they would count a debug stop as a hang and run the
    // recovery ladder against it: the debug-aware state is read at every backoff poll from now on.
    // The configuration is the debugger's to clear (ladder level 1, P3.7); TSR_TILE_SERIAL=0 clears
    // SERIAL at open.
    void debug_config_check() {
        if (!has_dbg) return;
        const uint32_t c = rdv(TSR_CSR_DBG_CFG), s = rdv(TSR_CSR_DBG_STATUS);
        uint32_t tbps = 0;
        for (uint32_t i = 0; i < TSR_DBG_NTBP; ++i) {
            const uint32_t ctl = rdv(TSR_CSR_DBG_TBP_CTL(i));
            if ((ctl & TSR_DBG_TBP_CTL_EN) && !(ctl & TSR_DBG_TBP_CTL_COUNT_ONLY)) tbps |= 1u << i;
        }
        uint32_t armed = c & (TSR_DBG_CFG_ERR_STOP | TSR_DBG_CFG_HUNG_STOP | TSR_DBG_CFG_HALT_ON_BOOT);
        if (tile_serial != 1) armed |= c & TSR_DBG_CFG_SERIAL;   // TSR_TILE_SERIAL=1 is this process's own
        const uint32_t stopped = s & (TSR_DBG_STATUS_STOP_ACTIVE | TSR_DBG_STATUS_S_VALID);
        if (armed == 0 && tbps == 0 && stopped == 0) return;
        dbg_active = true;
        std::string what;
        auto add = [&what](bool on, const char * w) {
            if (on) what += what.empty() ? w : std::string(", ") + w;
        };
        add(armed & TSR_DBG_CFG_SERIAL, "SERIAL");
        add(armed & TSR_DBG_CFG_ERR_STOP, "ERR_STOP");
        add(armed & TSR_DBG_CFG_HUNG_STOP, "HUNG_STOP");
        add(armed & TSR_DBG_CFG_HALT_ON_BOOT, "HALT_ON_BOOT");
        char tb[48];
        snprintf(tb, sizeof(tb), "tile breakpoints 0x%x", tbps);
        add(tbps != 0, tb);
        add(stopped & TSR_DBG_STATUS_STOP_ACTIVE, "a stop active");
        add(stopped & TSR_DBG_STATUS_S_VALID, "STOP_AT armed");
        msg("debug configuration left armed by an earlier debug session (DB keeps it across SOFT_RESET): %s (DBG_CFG 0x%08x, "
            "DBG_STATUS 0x%08x); waits do not count time while stopped or in serial mode", what.c_str(), c, s);
    }

    int open_device() {
        int rc = identity();
        if (rc) return rc;
        if ((rc = adapter_open()) != 0) return rc;
        // The host counters from the device first: the rings count on across processes (only a card
        // reset zeroes them), and the SOFT_RESET below must not take what an earlier process consumed
        // for new records.
        bool ok = rd(TSR_CSR_LQ_SEQ, &pushed) && rd(TSR_CSR_LCQ_RD, &lcq_rd) && rd(TSR_CSR_LOG_RD, &log_rd);
        for (uint32_t i = 0; i < 8 && ok; ++i) ok = rd(TSR_CSR_LQ_W(i), &lq_shadow[i]);
        if (!ok) return TSR_TE_IO;
        if (!attach) {
            // Records and log words an earlier process left unread are not this process's: the flush
            // discards them unread (no image is known to decode the log, and their sequence numbers
            // would alias this process's launches).
            uint32_t lw = 0, gw = 0;
            if (!rd(TSR_CSR_LCQ_WR, &lw) || !rd(TSR_CSR_LOG_WR, &gw)) return TSR_TE_IO;
            if (lw != lcq_rd || gw != log_rd)
                msg("SOFT_RESET at open discards %u LCQ record%s and %u log word%s left unread by an earlier process", lw - lcq_rd,
                    lw - lcq_rd == 1 ? "" : "s", gw - log_rd, gw - log_rd == 1 ? "" : "s");
            if ((rc = soft_reset_hw("open", false)) != 0) return rc;
            if ((rc = bind()) != 0) return rc;
            if (!wr(TSR_CSR_LOG_CFG, TSR_LOG_CFG_ENABLE | TSR_PUT(TSR_LOG_CFG_LEVEL, (uint32_t) log_level))) return TSR_TE_IO;
            dbg_active = has_dbg && tile_serial == 1;
            if (has_dbg && (tile_serial >= 0 || tile_hist >= 0)) {
                uint32_t c = rdv(TSR_CSR_DBG_CFG);
                if (tile_serial >= 0) c = tile_serial ? (c | TSR_DBG_CFG_SERIAL) : (c & ~TSR_DBG_CFG_SERIAL);
                if (tile_hist >= 0) c = tile_hist ? (c | TSR_DBG_CFG_HIST_EN) : (c & ~TSR_DBG_CFG_HIST_EN);
                if (!wr(TSR_CSR_DBG_CFG, c)) return TSR_TE_IO;
            }
            state = dstate::reset;
        } else {
            state = dstate::attached;
        }
        debug_config_check();
        log_last = now();
        return 0;
    }

    void print_stats() {
        const uint64_t n = st.launches;
        msg("stats: %" PRIu64 " launches, launch wall %.3f ms (%.2f us each); CSR %" PRIu64 " reads / %" PRIu64
            " writes; DRAM %.2f MB written / %.2f MB read; log %" PRIu64 " words, %" PRIu64 " records dropped; last boot %s "
            "(%u ITCM + %u DTCM segments, %u words) %.3f ms %s time; %u timeouts, %u traps, %u restarts",
            n, (double) st.launch_wall_ns / 1e6, n ? (double) st.launch_wall_ns / 1e3 / (double) n : 0.0, st.csr_reads, st.csr_writes,
            (double) st.bytes_written / 1e6, (double) st.bytes_read / 1e6, st.log_words, dec.dropped_reported,
            binfo.warm ? "warm" : "cold", binfo.itcm_loaded, binfo.dtcm_loaded, binfo.words_written, (double) binfo.time_ns / 1e6,
            tbase(), rlog.timeouts, rlog.traps, rlog.restarts);
    }
};

namespace {

struct guard {
    tsr_transport * t;
    explicit guard(tsr_transport * x) : t(x) { t->mu.lock(); }
    ~guard() { t->mu.unlock(); }
};

std::mutex & registry_mutex() {
    static std::mutex m;
    return m;
}
std::map<std::string, kdev_factory> & registry() {
    static std::map<std::string, kdev_factory> r;
    return r;
}

tsr_transport * open_on(std::unique_ptr<kdev> dev) {
    if (!dev) return nullptr;
    std::unique_ptr<tsr_transport> t(new tsr_transport);
    t->kind = dev->name();
    t->dev = std::move(dev);
    if (t->settings_from_env() != 0) return nullptr;
    guard g(t.get());
    if (t->open_device() != 0) return nullptr;
    return t.release();
}

} // namespace

namespace tsr {

tsr_transport * transport_open_kdev(std::unique_ptr<kdev> dev) { return open_on(std::move(dev)); }

std::unique_ptr<kdev> transport_detach_kdev(tsr_transport * t) {
    if (t == nullptr) return nullptr;
    std::unique_ptr<kdev> d = std::move(t->dev);
    delete t;
    return d;
}

kdev * transport_kdev(tsr_transport * t) { return t ? t->dev.get() : nullptr; }
tsr_sim * transport_sim(tsr_transport * t) { return t ? t->dev->sim() : nullptr; }

adapter_log transport_adapter_log(tsr_transport * t) {
    if (t == nullptr) return adapter_log();
    guard g(t);
    return t->alog;
}

boot_info transport_boot_info(tsr_transport * t) {
    if (t == nullptr) return boot_info();
    guard g(t);
    return t->binfo;
}

recovery_log transport_recovery_log(tsr_transport * t) {
    if (t == nullptr) return recovery_log();
    guard g(t);
    return t->rlog;
}

void transport_set_full_records(tsr_transport * t, bool on) {
    guard g(t);
    t->full_records = on;
}

void transport_set_timeouts(tsr_transport * t, uint64_t launch_ms, uint64_t abort_ms, uint64_t boot_ms, uint64_t reset_ms) {
    guard g(t);
    if (launch_ms) t->launch_ms = launch_ms;
    if (abort_ms) t->abort_ms = abort_ms;
    if (boot_ms) t->boot_ms = boot_ms;
    if (reset_ms) t->reset_ms = reset_ms;
}

void transport_set_wm_refresh(tsr_transport * t, uint32_t launches) {
    guard g(t);
    t->wm_refresh = launches != 0 && launches < 0x80000000u ? launches : 1u << 30;
}

uint64_t transport_perf(tsr_transport * t, uint32_t idx) {
    guard g(t);
    const uint32_t lo = t->rdv(TSR_CSR_PERF_LO(idx));
    const uint32_t hi = t->rdv(TSR_CSR_PERF_HI(idx));
    return ((uint64_t) hi << 32) | lo;
}

bool register_kdev(const std::string & kind, kdev_factory f) {
    if (kind.empty() || kind == "sim" || kind == "xrt" || kind == "mock" || kind == "vkernel" || !f) return false;
    std::lock_guard<std::mutex> lk(registry_mutex());
    registry()[kind] = std::move(f);
    return true;
}

} // namespace tsr

// ===========================================================================
// C API (EX §10.1, SW §9.1)
// ===========================================================================
extern "C" tsr_transport * tsr_transport_kdev_open(const char * kind) {
    std::unique_ptr<kdev> dev;
    try {
        if (strcmp(kind, "sim") == 0) {
            dev = open_sim_kdev();
        } else if (strcmp(kind, "mock") == 0) {
            dev = open_mock_kdev();
        } else if (strcmp(kind, "vkernel") == 0) {
            dev = open_vkernel_kdev();
        } else if (strcmp(kind, "xrt") == 0) {
            dev = open_xrt_kdev();
        } else {
            kdev_factory f;
            std::string known;
            {
                std::lock_guard<std::mutex> lk(registry_mutex());
                auto it = registry().find(kind);
                if (it != registry().end()) f = it->second;
                for (auto & e : registry()) known += " or " + e.first;
            }
            if (!f) {
                fprintf(stderr, "tsr_transport: unknown transport \"%s\" (TSR_DEVICE must be sim, mock, vkernel, xrt%s)\n", kind,
                        known.c_str());
                return nullptr;
            }
            dev = f();
        }
    } catch (const std::exception & e) {
        fprintf(stderr, "tsr_transport(%s): %s\n", kind, e.what());
        return nullptr;
    }
    return open_on(std::move(dev));
}

extern "C" void tsr_transport_close(tsr_transport * t) {
    if (t == nullptr) return;
    {
        guard g(t);
        if (t->state != dstate::failed) t->drain_log();
        if (t->stats_on) t->print_stats();
    }
    delete t;
}

extern "C" const char * tsr_transport_name(const tsr_transport * t) { return t ? t->kind.c_str() : ""; }

extern "C" const tsr_config * tsr_transport_config(const tsr_transport * t) { return &t->cfg; }

extern "C" int tsr_transport_write(tsr_transport * t, uint32_t ch, uint64_t off, const void * src, size_t size) {
    guard g(t);
    if (ch >= t->cfg.num_ch || off > t->cfg.ch_bytes || (uint64_t) size > t->cfg.ch_bytes - off) {
        t->msg("write out of range: ch %u, offset 0x%" PRIx64 ", %zu bytes (num_ch %u, ch_bytes %" PRIu64 ")", ch, off, size, t->cfg.num_ch,
               t->cfg.ch_bytes);
        return TSR_TE_INVAL;
    }
    if (size == 0) return 0;
    if (t->dev->mem_write(ch, off, src, size) != 0) return TSR_TE_IO;
    t->st.bytes_written += size;
    return 0;
}

extern "C" int tsr_transport_read(tsr_transport * t, uint32_t ch, uint64_t off, void * dst, size_t size) {
    guard g(t);
    if (ch >= t->cfg.num_ch || off > t->cfg.ch_bytes || (uint64_t) size > t->cfg.ch_bytes - off) {
        t->msg("read out of range: ch %u, offset 0x%" PRIx64 ", %zu bytes (num_ch %u, ch_bytes %" PRIu64 ")", ch, off, size, t->cfg.num_ch,
               t->cfg.ch_bytes);
        return TSR_TE_INVAL;
    }
    if (size == 0) return 0;
    if (t->dev->mem_read(ch, off, dst, size) != 0) return TSR_TE_IO;
    t->st.bytes_read += size;
    return 0;
}

extern "C" uint32_t tsr_transport_csr_read(tsr_transport * t, uint32_t off) {
    guard g(t);
    if (off >= TSR_SAXI_CORE_BYTES || (off & 3u) != 0) {
        t->msg("CSR offset 0x%x outside the core CSR map", off);
        return 0xFFFFFFFFu;
    }
    uint32_t v = 0;
    t->rd(off, &v);
    return v;
}

extern "C" int tsr_transport_csr_write(tsr_transport * t, uint32_t off, uint32_t val) {
    guard g(t);
    if (off >= TSR_SAXI_CORE_BYTES || (off & 3u) != 0) {
        t->msg("CSR offset 0x%x outside the core CSR map", off);
        return TSR_TE_INVAL;
    }
    if (!t->wr(off, val)) return TSR_TE_IO;
    if (off >= TSR_CSR_LQ_W(0) && off < TSR_CSR_LQ_W(8)) t->lq_shadow[(off - TSR_CSR_LQ_W(0)) / 4] = val;   // keep the shadow true
    if (off == TSR_CSR_LQ_CANCEL || off == TSR_CSR_LQ_STOP) t->wm_written(off, val);
    if (off >= TSR_CSR_DBG_BASE && off < TSR_CSR_DMI_BASE + 4 * TSR_CSR_DMI_N) t->dbg_active = t->has_dbg;   // DB §4.1
    return 0;
}

extern "C" int tsr_transport_boot(tsr_transport * t, const tsr_image * img) {
    if (t == nullptr || img == nullptr) return TSR_TE_INVAL;
    guard g(t);
    return t->boot_img(img);
}

extern "C" int tsr_transport_launch(tsr_transport * t, const tsr_launch * l, uint32_t * seq) {
    if (t == nullptr || l == nullptr) return TSR_TE_INVAL;
    guard g(t);
    return t->launch(l, seq);
}

extern "C" int tsr_transport_wait(tsr_transport * t, uint32_t seq, uint64_t timeout_ns, tsr_record * rec) {
    if (t == nullptr) return TSR_TE_INVAL;
    guard g(t);
    return t->wait_seq(seq, timeout_ns, rec);
}

extern "C" int tsr_transport_call(tsr_transport * t, const tsr_launch * l, uint64_t timeout_ns, tsr_record * rec) {
    if (t == nullptr || l == nullptr) return TSR_TE_INVAL;
    guard g(t);
    uint32_t seq = 0;
    const int rc = t->launch(l, &seq);
    if (rc) return rc;
    return t->wait_seq(seq, timeout_ns, rec);
}

extern "C" int tsr_transport_cancel(tsr_transport * t, uint32_t seq_upto) {
    if (t == nullptr) return TSR_TE_INVAL;
    guard g(t);
    if (!t->wr(TSR_CSR_LQ_CANCEL, seq_upto)) return TSR_TE_IO;
    t->wm_written(TSR_CSR_LQ_CANCEL, seq_upto);
    return 0;
}

extern "C" int tsr_transport_stop(tsr_transport * t, uint32_t seq_upto) {
    if (t == nullptr) return TSR_TE_INVAL;
    guard g(t);
    if (!t->wr(TSR_CSR_LQ_STOP, seq_upto)) return TSR_TE_IO;
    t->wm_written(TSR_CSR_LQ_STOP, seq_upto);
    return 0;
}

extern "C" int tsr_transport_set_event_cb(tsr_transport * t, void (*cb)(void *, const tsr_record *), void * user) {
    if (t == nullptr) return TSR_TE_INVAL;
    guard g(t);
    t->event_cb = cb;
    t->event_user = user;
    return 0;
}

extern "C" int tsr_transport_soft_reset(tsr_transport * t) {
    if (t == nullptr) return TSR_TE_INVAL;
    guard g(t);
    const int rc = t->soft_reset_hw("tsr_transport_soft_reset");
    if (rc == 0 && t->state == dstate::trapped) t->state = dstate::reset;
    return rc;
}

extern "C" int tsr_transport_log_drain(tsr_transport * t, uint32_t * words, size_t max, size_t * n) {
    if (t == nullptr || n == nullptr || (words == nullptr && max != 0)) return TSR_TE_INVAL;
    guard g(t);
    *n = 0;
    if (t->state != dstate::failed && !t->drain_log()) return TSR_TE_IO;
    const size_t k = std::min(max, t->log_buf.size());
    for (size_t i = 0; i < k; ++i) words[i] = t->log_buf[i];
    t->log_buf.erase(t->log_buf.begin(), t->log_buf.begin() + (long) k);
    *n = k;
    return 0;
}

extern "C" const tsr_image * tsr_transport_image(const tsr_transport * t) {
    if (t == nullptr) return nullptr;
    tsr_transport * m = const_cast<tsr_transport *>(t);   // the state is written by boots and restarts on other threads
    guard g(m);
    return m->state == dstate::ready || m->state == dstate::trapped ? m->img : nullptr;
}

extern "C" void tsr_transport_get_stats(const tsr_transport * t, tsr_transport_stats * s) {
    if (t == nullptr || s == nullptr) return;
    tsr_transport * m = const_cast<tsr_transport *>(t);
    guard g(m);
    *s = m->st;
    s->log_records_dropped = m->dec.dropped_reported;
    s->staged_chunks = 0;
}
