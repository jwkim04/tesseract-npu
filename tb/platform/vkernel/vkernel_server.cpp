// vkernel_server.cpp — the Verilated Alveo kernel as a device server for the host software
// (TSR_DEVICE=vkernel, sw/transport/tsr_kdev_vkernel.cpp; wire protocol
// sw/transport/tsr_vkernel_proto.h; docs/platform-u200.md §13). Built by
// tb/platform/vkernel/vkernel.py like the kernel bench (fr_config.h, model class Vkernel: the kernel
// body tsr_xrt_kernel or a packaged top behind tb/platform/sv/kwrap_<kernel>.sv).
//
//   vkernel_server --fd N | --socket PATH  [--regime fast|rand|slow|ordered] [--mem K=V]...
//                  [--dut K=V]... [--seed N] [--runlog FILE] [--summary FILE]
//                  [--max-run-cycles N] [--quiet]
//
// The kernel is the one tb/platform/kernel_dut.h drives (KernelDut): the RTL AXI4-Lite slave
// behind an XRT-like AXI4-Lite master (random VALID / READY delays unless --dut axil=fast), AXI4
// slave memories per channel served by fr::MemModel in the chosen regime (default fast: always
// ready, latency 0..2), random 4 KiB aligned channel bases (--dut base=rand|zero|0xHEX), and every
// check of the bench: the AXI4 checker on every channel and cycle, the AXI4-Lite slave checks, the
// memory model's rules P1..P6 and the RTL assertions. The client's register accesses are AXI4-Lite
// transactions through the RTL; its memory accesses are host DMA into the channel memories (no
// kernel cycles, as XDMA on the card). The clock runs while a run is in progress (from the client's
// AP_CTRL start write until a completion is seen) and during each AXI4-Lite transaction; it stops
// while the kernel is idle. A run ends for the server when an AP_CTRL read (the client's, or the
// server's own in VK_WAIT) shows ap_done: then nothing may be in flight in the memories, and the
// server reads PERF_* through the window and logs the run.
//
// Findings (any check failing, an access timing out, traffic in flight at completion, a run over
// --max-run-cycles) are printed, counted in every response (viol) and returned by VK_MSGS; the
// client fails the operation that saw the count grow. The run log (--runlog, one JSON object per
// run): the program (instructions up to END, MVM count, largest N, MACs), cycles from the start
// write to the completion, PERF deltas (cycles, retired, MVM cycles, read beats per channel), STATUS
// and ERR_CODE, wall time, reads of memory nobody wrote. --summary: the session totals as JSON.

#include <verilated.h>

#include "Vkernel.h"
#if VM_TRACE
#include <verilated_vcd_c.h>
#endif

#include "fr_config.h"
#include "fr_common.h"
#include "kernel_dut.h"
#include "tsr_vkernel_proto.h"

#include <poll.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <csignal>

namespace {

struct SrvOpts {
    int fd = -1;
    std::string socket_path;
    std::string regime = "fast";
    std::string runlog, summary;
    uint64_t max_run_cycles = 500000000ull;
    bool quiet = false;
};

bool sock_write(int fd, const void* p, size_t n) {
    const uint8_t* b = (const uint8_t*)p;
    while (n > 0) {
        const ssize_t k = ::send(fd, b, n, MSG_NOSIGNAL);
        if (k < 0 && errno == EINTR) continue;
        if (k <= 0) return false;
        b += k;
        n -= (size_t)k;
    }
    return true;
}

bool sock_read(int fd, void* p, size_t n) {
    uint8_t* b = (uint8_t*)p;
    while (n > 0) {
        const ssize_t k = ::recv(fd, b, n, 0);
        if (k < 0 && errno == EINTR) continue;
        if (k <= 0) return false;
        b += k;
        n -= (size_t)k;
    }
    return true;
}

class Server {
  public:
    Server(KernelDut& d, const SrvOpts& o, std::string desc) : d_(d), o_(o), desc_(std::move(desc)) {
        if (!o.runlog.empty()) {
            log_ = fopen(o.runlog.c_str(), "w");
            if (log_ == nullptr) fprintf(stderr, "vkernel: cannot create %s\n", o.runlog.c_str());
        }
        t0_ = now_s();
    }
    ~Server() {
        if (log_) fclose(log_);
    }

    // Start-up check: constants through the AXI4-Lite slave and the window, as the kernel bench.
    void startup_check() {
        const std::string e = d_.check_constants();
        if (!e.empty()) finding("start-up check: " + e);
        collect();
    }

    // Serve one connection until VK_BYE or EOF. Returns false after VK_BYE (stop serving).
    bool serve(int fd) {
        fd_ = fd;
        for (;;) {
            if (active_ && !stuck_) free_run();
            tsr_vk_req q{};
            if (!sock_read(fd_, &q, sizeof(q))) return true;            // client gone
            if (!handle(q)) return q.op != VK_BYE;
        }
    }

    void write_summary() {
        const std::string s = summary_json();
        if (!o_.summary.empty()) {
            FILE* f = fopen((o_.summary + ".tmp").c_str(), "w");
            if (f) {
                fputs(s.c_str(), f);
                fputc('\n', f);
                fclose(f);
                rename((o_.summary + ".tmp").c_str(), o_.summary.c_str());
            }
        }
        if (!o_.quiet) fprintf(stderr, "vkernel: %s\n", s.c_str());
    }

  private:
    KernelDut& d_;
    const SrvOpts& o_;
    std::string desc_;
    FILE* log_ = nullptr;
    int fd_ = -1;
    double t0_ = 0;
    // findings
    uint32_t viol_ = 0;
    std::string pending_;
    uint64_t mem_err_seen_ = 0, rtl_err_seen_ = 0;
    // current run
    bool active_ = false, stuck_ = false;
    uint64_t run_start_cyc_ = 0;
    double run_start_wall_ = 0;
    uint32_t cmd_lo_ = 0, cmd_hi_ = 0, cmd_cnt_ = 0;
    uint64_t perf_prev_[N_PERF] = {0};
    uint64_t bad_prev_[NUM_CH] = {0};
    // session totals
    uint64_t runs_ = 0, busy_cycles_ = 0, mvm_cycles_ = 0, kernel_cycles_ = 0, retired_ = 0;
    uint64_t host_wr_bytes_ = 0, host_rd_bytes_ = 0, reg_rd_ = 0, reg_wr_ = 0, waits_ = 0;
    uint64_t bad_reads_ = 0, macs_ = 0;
    double run_wall_ = 0;
    std::vector<uint8_t> buf_;

    void finding(const std::string& s) {
        ++viol_;
        fprintf(stderr, "vkernel: FINDING cycle %llu: %s\n", (unsigned long long)d_.cycles(), s.c_str());
        if (pending_.size() < 64 * 1024) pending_ += strf("cycle %llu: ", (unsigned long long)d_.cycles()) + s + "\n";
    }

    // Move every check result produced since the last call into the findings.
    void collect() {
        for (const auto& e : d_.errs) finding(e);
        d_.errs.clear();
        for (const auto& e : d_.done_errs()) finding(e);
        d_.done_errs().clear();
        const fr::MemModel& m = d_.mem();
        if (m.n_errors() > mem_err_seen_) {
            const auto& es = m.errors();
            for (uint64_t i = mem_err_seen_; i < m.n_errors(); ++i)
                finding(i < es.size() ? es[(size_t)i] : std::string("memory model rule violated (message not kept)"));
            mem_err_seen_ = m.n_errors();
        }
        if (g_rtl_error_count > rtl_err_seen_) {
            for (uint64_t i = rtl_err_seen_; i < g_rtl_error_count; ++i)
                finding(i < g_rtl_errors.size() ? g_rtl_errors[(size_t)i] : std::string("RTL $error / assertion"));
            rtl_err_seen_ = g_rtl_error_count;
        }
    }

    // Clock the kernel while a run is in progress, until the client sends a request (checked
    // every 16 cycles: a poll() costs far less than 16 simulated cycles).
    void free_run() {
        pollfd p{fd_, POLLIN, 0};
        while (::poll(&p, 1, 0) == 0) {
            for (int i = 0; i < 16; ++i) d_.step();
            if (budget_exceeded()) break;
            collect();
        }
        collect();
    }

    bool budget_exceeded() {
        if (active_ && !stuck_ && d_.cycles() - run_start_cyc_ > o_.max_run_cycles) {
            stuck_ = true;
            finding(strf("run %llu still in progress after %llu cycles (--max-run-cycles): the clock stops",
                         (unsigned long long)runs_ + 1, (unsigned long long)o_.max_run_cycles));
            return true;
        }
        return stuck_;
    }

    void begin_run() {
        active_ = true;
        stuck_ = false;
        run_start_cyc_ = d_.cycles();
        run_start_wall_ = now_s();
        d_.mem().set_active(true);
    }

    // A completion was seen (ap_done in an AP_CTRL read).
    void complete() {
        active_ = false;
        const uint64_t kcyc = d_.cycles() - run_start_cyc_;
        const double wall = now_s() - run_start_wall_;
        if (!d_.mem().idle()) finding("memory traffic in flight at completion: " + d_.mem().busy_desc());
        d_.mem().set_active(false);
        ++runs_;
        kernel_cycles_ += kcyc;
        run_wall_ += wall;
        // PERF_* through the window (LO first: the core snapshots on LO), STATUS, ERR_CODE
        bool ok = true;
        uint64_t p[N_PERF];
        for (unsigned i = 0; i < N_PERF; ++i) {
            const uint32_t a = TSR_CSR_PERF_BASE + 8 * i;
            const uint64_t lo = d_.csr(a, false, 0, &ok);
            const uint64_t hi = d_.csr(a + 4, false, 0, &ok);
            p[i] = (hi << 32) | lo;
        }
        const uint32_t st = d_.csr(TSR_CSR_STATUS, false, 0, &ok), ec = d_.csr(TSR_CSR_ERR_CODE, false, 0, &ok);
        if (!ok) finding("window read timed out after a run");
        uint64_t dp[N_PERF];
        for (unsigned i = 0; i < N_PERF; ++i) dp[i] = p[i] - perf_prev_[i], perf_prev_[i] = p[i];
        busy_cycles_ += dp[TSR_PERF_CYCLES];
        mvm_cycles_ += dp[TSR_PERF_MVM_CYCLES];
        retired_ += dp[TSR_PERF_RETIRED];
        uint64_t bad = 0;
        for (unsigned c = 0; c < NUM_CH; ++c) {
            const uint64_t b = d_.mem().stats(c).bad_reads;
            bad += b - bad_prev_[c];
            bad_prev_[c] = b;
        }
        bad_reads_ += bad;
        // the program (immediates of MVM: N in w4, K / T in w5)
        const uint64_t cmd = ((uint64_t)cmd_hi_ << 32) | cmd_lo_;
        uint64_t ninsn = 0, nmvm = 0, max_n = 0, max_t = 0, macs = 0;
        if (cmd < FR_CH_BYTES / 64) {
            const uint8_t* m = d_.mem().mem(0);
            for (uint64_t i = 0; i < cmd_cnt_; ++i) {
                const uint64_t a = cmd * 64 + (uint64_t)TSR_INSN_BYTES * i;
                if (a + TSR_INSN_BYTES > FR_CH_BYTES) break;
                uint32_t w[8];
                memcpy(w, m + a, sizeof(w));
                ++ninsn;
                const uint32_t opc = tsr_w0_opcode(w[0]);
                if (opc == TSR_OP_MVM) {
                    const uint64_t n = w[4], k = w[5] & 0xFFFFu, t = (w[5] >> 16) & 0xFu;
                    ++nmvm;
                    max_n = std::max(max_n, n);
                    max_t = std::max(max_t, t);
                    macs += n * k * t;
                }
                if (opc == TSR_OP_END) break;
            }
        }
        macs_ += macs;
        if (log_) {
            std::string j = strf("{\"run\": %llu, \"cmd_addr\": %llu, \"cmd_count\": %u, \"insns\": %llu, \"mvm\": %llu, "
                                 "\"max_n\": %llu, \"max_t\": %llu, \"macs\": %llu, \"start_cycle\": %llu, \"kcycles\": %llu, "
                                 "\"status\": %u, \"err_code\": %u, \"wall\": %.6f, \"bad_reads\": %llu, \"perf\": {"
                                 "\"cycles\": %llu, \"retired\": %llu, \"mvm_cycles\": %llu, \"ch_rd_beats\": [",
                                 (unsigned long long)runs_, (unsigned long long)cmd, cmd_cnt_, (unsigned long long)ninsn,
                                 (unsigned long long)nmvm, (unsigned long long)max_n, (unsigned long long)max_t,
                                 (unsigned long long)macs, (unsigned long long)run_start_cyc_, (unsigned long long)kcyc, st,
                                 ec, wall, (unsigned long long)bad, (unsigned long long)dp[TSR_PERF_CYCLES],
                                 (unsigned long long)dp[TSR_PERF_RETIRED], (unsigned long long)dp[TSR_PERF_MVM_CYCLES]);
            for (unsigned c = 0; c < NUM_CH; ++c)
                j += strf("%s%llu", c ? ", " : "", (unsigned long long)dp[TSR_PERF_CH_RD_BEATS + c]);
            j += "]}}\n";
            fputs(j.c_str(), log_);
            fflush(log_);
        }
        collect();
    }

    void check_completion(uint32_t ap) {
        if (active_ && (ap & AP_DONE)) complete();
    }

    std::string summary_json() const {
        const double wall = now_s() - t0_;
        return strf("{\"runs\": %llu, \"findings\": %u, \"sim_cycles\": %llu, \"kernel_cycles\": %llu, "
                    "\"perf_cycles\": %llu, \"perf_mvm_cycles\": %llu, \"perf_retired\": %llu, \"macs\": %llu, "
                    "\"bad_reads\": %llu, \"reg_reads\": %llu, \"reg_writes\": %llu, \"waits\": %llu, "
                    "\"host_write_bytes\": %llu, \"host_read_bytes\": %llu, \"run_wall\": %.3f, \"wall\": %.3f, "
                    "\"cycles_per_s\": %.1f, \"build\": %s}",
                    (unsigned long long)runs_, viol_, (unsigned long long)d_.cycles(), (unsigned long long)kernel_cycles_,
                    (unsigned long long)busy_cycles_, (unsigned long long)mvm_cycles_, (unsigned long long)retired_,
                    (unsigned long long)macs_, (unsigned long long)bad_reads_, (unsigned long long)reg_rd_,
                    (unsigned long long)reg_wr_, (unsigned long long)waits_, (unsigned long long)host_wr_bytes_,
                    (unsigned long long)host_rd_bytes_, run_wall_, wall, run_wall_ > 0 ? kernel_cycles_ / run_wall_ : 0.0,
                    json_str(desc_).c_str());
    }

    bool reply(int32_t rc, uint32_t val = 0, uint64_t x = 0, const void* pl = nullptr, size_t n = 0) {
        collect();
        tsr_vk_rsp r{rc, val, x, viol_, (uint32_t)n};
        return sock_write(fd_, &r, sizeof(r)) && (n == 0 || sock_write(fd_, pl, n));
    }

    bool range_ok(uint32_t ch, uint64_t off, uint64_t n, const char* what) {
        if (ch < NUM_CH && off <= (uint64_t)FR_CH_BYTES && n <= (uint64_t)FR_CH_BYTES - off) return true;
        finding(strf("host %s outside the channel buffers: ch %u, offset 0x%llx, %llu bytes", what, ch,
                     (unsigned long long)off, (unsigned long long)n));
        return false;
    }

    // VK_WAIT: the XRT scheduler's side of a run (polling or interrupt), at cycle granularity.
    int32_t wait(uint32_t mode, uint64_t poll, uint64_t timeout_ms) {
        ++waits_;
        const double t_end = timeout_ms ? now_s() + timeout_ms / 1000.0 : 0;
        if (poll < 1) poll = 1;
        for (;;) {
            if (mode == VK_WAIT_POLL) {
                for (uint64_t i = 0; i + 1 < poll; ++i) d_.step();
                bool ok = true;
                const uint32_t ap = d_.reg(KR_AP_CTRL, &ok);
                if (!ok) return VK_ERROR;
                if (ap & AP_DONE) {
                    check_completion(ap);
                    return VK_DONE;
                }
            } else if (mode == VK_WAIT_IRQ) {
                for (int i = 0; i < 256 && !d_.irq(); ++i) d_.step();
                if (d_.irq()) {
                    // the scheduler's interrupt handler: clear the done status (ISR toggles on a
                    // write of 1), then the CU state from AP_CTRL (clears ap_done / ap_ready)
                    bool ok = true;
                    d_.wreg(KR_ISR, 1, &ok);
                    const uint32_t ap = d_.reg(KR_AP_CTRL, &ok);
                    if (!ok) return VK_ERROR;
                    if (!(ap & AP_DONE)) {
                        finding(strf("interrupt without ap_done (AP_CTRL = 0x%x)", ap));
                        return VK_ERROR;
                    }
                    check_completion(ap);
                    return VK_DONE;
                }
            } else {
                finding(strf("VK_WAIT with unknown mode %u", mode));
                return VK_ERROR;
            }
            if (budget_exceeded()) return VK_ERROR;
            if (t_end && now_s() > t_end) return VK_TIMEOUT;
            collect();
        }
    }

    // One request. Returns false to end the connection (VK_BYE or a broken socket).
    bool handle(const tsr_vk_req& q) {
        switch (q.op) {
        case VK_HELLO: {
            if (q.a != TSR_VK_PROTO_VERSION) {
                fprintf(stderr, "vkernel: client protocol %u, server %u\n", q.a, TSR_VK_PROTO_VERSION);
                return reply(-1);
            }
            std::vector<uint8_t> pl(8 * NUM_CH);
            memcpy(pl.data(), d_.base, 8 * NUM_CH);
            pl.insert(pl.end(), desc_.begin(), desc_.end());
            pl.push_back(0);
            return reply(viol_ ? -1 : 0, NUM_CH, (uint64_t)FR_CH_BYTES, pl.data(), pl.size());
        }
        case VK_REG_RD: {
            ++reg_rd_;
            bool ok = q.a < 0x2000 && (q.a & 3) == 0;
            if (!ok) finding(strf("register read at 0x%x outside s_axi_control", q.a));
            const uint32_t v = ok ? d_.reg(q.a, &ok) : 0;
            if (ok && q.a == KR_AP_CTRL) check_completion(v);
            return reply(ok ? 0 : -1, v);
        }
        case VK_REG_WR: {
            ++reg_wr_;
            bool ok = q.a < 0x2000 && (q.a & 3) == 0;
            if (!ok) finding(strf("register write at 0x%x outside s_axi_control", q.a));
            if (ok) {
                if (q.a == KR_CMD_LO) cmd_lo_ = (uint32_t)q.b;
                if (q.a == KR_CMD_HI) cmd_hi_ = (uint32_t)q.b;
                if (q.a == KR_CMD_CNT) cmd_cnt_ = (uint32_t)q.b;
                if (q.a == KR_AP_CTRL && (q.b & AP_START) && !active_) begin_run();
                d_.wreg(q.a, (uint32_t)q.b, &ok);
            }
            return reply(ok ? 0 : -1);
        }
        case VK_MEM_WR: {
            if (q.c > TSR_VK_MAX_XFER) return false;
            buf_.resize((size_t)q.c);
            if (!sock_read(fd_, buf_.data(), buf_.size())) return false;
            if (!range_ok(q.a, q.b, q.c, "write")) return reply(-1);
            d_.mem().load(q.a, q.b, buf_.data(), q.c);
            if (q.c) d_.mem().allow_beats(q.a, q.b / 64, (q.b + q.c + 63) / 64);
            host_wr_bytes_ += q.c;
            return reply(0);
        }
        case VK_MEM_RD: {
            if (q.c > TSR_VK_MAX_XFER) return false;
            if (!range_ok(q.a, q.b, q.c, "read")) return reply(-1);
            host_rd_bytes_ += q.c;
            return reply(0, 0, 0, d_.mem().mem(q.a) + q.b, (size_t)q.c);
        }
        case VK_WAIT:
            return reply(wait(q.a, q.b, q.c));
        case VK_STATS: {
            const std::string s = summary_json();
            return reply(0, 0, 0, s.c_str(), s.size() + 1);
        }
        case VK_MSGS: {
            const std::string s = pending_;
            pending_.clear();
            return reply(0, 0, 0, s.c_str(), s.size() + 1);
        }
        case VK_BYE:
            reply(0);
            return false;
        default:
            finding(strf("unknown request %u", q.op));
            return reply(-1);
        }
    }
};

void server_usage() {
    fprintf(stderr,
            "usage: vkernel_server --fd N | --socket PATH [--regime fast|rand|slow|ordered] [--mem K=V]...\n"
            "                      [--dut K=V]... [--seed N] [--runlog FILE] [--summary FILE]\n"
            "                      [--max-run-cycles N] [--quiet]\n"
            "build: " FR_BUILD_DESC "\n");
}

}  // namespace

int main(int argc, char** argv) {
    signal(SIGPIPE, SIG_IGN);
    SrvOpts so;
    Options opt;
    opt.seed = 1;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto need = [&]() -> const char* {
            if (i + 1 >= argc) {
                server_usage();
                exit(2);
            }
            return argv[++i];
        };
        if (a == "--fd") so.fd = atoi(need());
        else if (a == "--socket") so.socket_path = need();
        else if (a == "--regime") so.regime = need();
        else if (a == "--mem") opt.mem_overrides.push_back(need());
        else if (a == "--dut") opt.dut_args.push_back(need());
        else if (a == "--seed") opt.seed = strtoull(need(), nullptr, 0);
        else if (a == "--runlog") so.runlog = need();
        else if (a == "--summary") so.summary = need();
        else if (a == "--max-run-cycles") so.max_run_cycles = strtoull(need(), nullptr, 0);
        else if (a == "--vcd") opt.vcd = need();
        else if (a == "--quiet") so.quiet = true;
        else if (a == "-h" || a == "--help") {
            server_usage();
            return 0;
        } else {
            fprintf(stderr, "vkernel: unknown option %s\n", a.c_str());
            server_usage();
            return 2;
        }
    }
    if ((so.fd < 0) == so.socket_path.empty()) {
        server_usage();
        return 2;
    }
    fr::Regime reg;
    if (!fr::Regime::by_name(so.regime, &reg)) {
        fprintf(stderr, "vkernel: unknown regime %s\n", so.regime.c_str());
        return 2;
    }
    for (const auto& kv : opt.mem_overrides) {
        if (KernelDut::mem_key(kv)) continue;
        const std::string e = reg.set(kv);
        if (!e.empty()) {
            fprintf(stderr, "vkernel: --mem %s: %s\n", kv.c_str(), e.c_str());
            return 2;
        }
    }
    KernelDut dut(opt, argc, argv);
    dut.manual = true;                 // completions are detected by the server (AP_CTRL reads)
    dut.hw_reset();
    dut.prepare(reg, opt.seed);        // channel memories, bases, AXI4-Lite master delays
    std::string desc = FR_BUILD_DESC;
    desc += strf(", regime %s, seed %llu, bases", reg.name.c_str(), (unsigned long long)opt.seed);
    for (unsigned c = 0; c < NUM_CH; ++c) desc += strf(" 0x%llx", (unsigned long long)dut.base[c]);
    for (const auto& kv : opt.mem_overrides) desc += ", " + kv;
    for (const auto& kv : opt.dut_args) desc += ", " + kv;
    Server srv(dut, so, desc);
    srv.startup_check();
    if (!so.quiet) fprintf(stderr, "vkernel: ready: %s\n", desc.c_str());
    if (so.fd >= 0) {
        srv.serve(so.fd);
        ::close(so.fd);
    } else {
        const int ls = ::socket(AF_UNIX, SOCK_STREAM, 0);
        sockaddr_un a{};
        a.sun_family = AF_UNIX;
        if (ls < 0 || so.socket_path.size() >= sizeof(a.sun_path)) {
            fprintf(stderr, "vkernel: bad socket path\n");
            return 2;
        }
        strcpy(a.sun_path, so.socket_path.c_str());
        ::unlink(so.socket_path.c_str());
        if (::bind(ls, (sockaddr*)&a, sizeof(a)) != 0 || ::listen(ls, 1) != 0) {
            fprintf(stderr, "vkernel: cannot listen on %s: %s\n", so.socket_path.c_str(), strerror(errno));
            return 2;
        }
        // clients one after another (the card stays loaded between host processes) until VK_BYE
        for (bool more = true; more;) {
            const int c = ::accept(ls, nullptr, nullptr);
            if (c < 0) {
                if (errno == EINTR) continue;
                break;
            }
            more = srv.serve(c);
            ::close(c);
        }
        ::close(ls);
        ::unlink(so.socket_path.c_str());
    }
    srv.write_summary();
    return 0;
}
