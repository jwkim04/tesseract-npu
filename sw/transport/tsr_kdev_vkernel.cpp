// tsr_kdev_vkernel.cpp — the "vkernel" kdev: the protocol layer (tsr_kdev.cpp) against the
// Verilated Alveo kernel running in a separate server process (wire protocol
// tsr_vkernel_proto.h), so a host process exercises the real host path — register map, the CSR
// window, ADAPTER_STATUS, transfers — against the RTL without linking Verilator.
// docs/platform-u200.md §13. The two-clock server comes with P5.3
// (docs/design/06-software-verification.md §11). The client follows the kdev interface (EX §10.2):
// idle() lets device time pass (VK_ADVANCE, the "advance" hint) and now_ns() is the server's device time at the nominal kernel clock (VK_TIME),
// so the protocol's timeouts count simulated time, as on the simulator kinds.
// TSR_DEVICE=vkernel opens it with:
//   TSR_VKERNEL_SERVER   server command line (words separated by blanks, no quoting): the client
//                        spawns it with one end of a socketpair (appends --fd N); it exits when
//                        the client closes the connection (also at process exit)
//   TSR_VKERNEL_SOCKET   instead: path of a Unix socket a server listens on (vkernel_server --socket)
// Every response carries the server's count of verification findings (AXI4 / AXI4-Lite checks,
// memory-model rules, RTL assertions); when it grows the client prints them and fails the
// operation (or the open).

#include "tsr_kdev.h"
#include "tsr_vkernel_proto.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cinttypes>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <mutex>
#include <spawn.h>
#include <string>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

extern char ** environ;

namespace tsr {

namespace {

bool write_all(int fd, const void * p, size_t n) {
    const uint8_t * b = (const uint8_t *) p;
    while (n > 0) {
        const ssize_t k = ::send(fd, b, n, MSG_NOSIGNAL);
        if (k < 0 && errno == EINTR) continue;
        if (k <= 0) return false;
        b += k;
        n -= (size_t) k;
    }
    return true;
}

bool read_all(int fd, void * p, size_t n) {
    uint8_t * b = (uint8_t *) p;
    while (n > 0) {
        const ssize_t k = ::recv(fd, b, n, 0);
        if (k < 0 && errno == EINTR) continue;
        if (k <= 0) return false;
        b += k;
        n -= (size_t) k;
    }
    return true;
}

std::vector<std::string> split_words(const char * s) {
    std::vector<std::string> w;
    std::string cur;
    for (const char * p = s; *p; ++p) {
        if (*p == ' ' || *p == '\t' || *p == '\n') {
            if (!cur.empty()) w.push_back(cur);
            cur.clear();
        } else {
            cur += *p;
        }
    }
    if (!cur.empty()) w.push_back(cur);
    return w;
}

class vkernel_kdev;

// Open connections, closed at process exit (the backend never closes its transport): the server
// then prints its summary before this process is gone.
std::mutex & live_mutex() {
    static std::mutex m;
    return m;
}
std::vector<vkernel_kdev *> & live() {
    static std::vector<vkernel_kdev *> v;
    return v;
}
void close_live_at_exit();

class vkernel_kdev final : public kdev {
public:
    ~vkernel_kdev() override {
        {
            std::lock_guard<std::mutex> lk(live_mutex());
            auto & v = live();
            for (size_t i = 0; i < v.size(); ++i) {
                if (v[i] == this) {
                    v.erase(v.begin() + (long) i);
                    break;
                }
            }
        }
        shutdown();
    }

    // Spawn or connect, then HELLO. nullptr-equivalent: ok() false (message printed).
    bool open() {
        const char * srv = getenv("TSR_VKERNEL_SERVER");
        const char * sock = getenv("TSR_VKERNEL_SOCKET");
        if (sock != nullptr && sock[0] != '\0') {
            fd_ = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
            sockaddr_un a{};
            a.sun_family = AF_UNIX;
            if (strlen(sock) >= sizeof(a.sun_path)) return fail("TSR_VKERNEL_SOCKET path too long");
            strcpy(a.sun_path, sock);
            if (fd_ < 0 || ::connect(fd_, (sockaddr *) &a, sizeof(a)) != 0) {
                fprintf(stderr, "tsr_transport(vkernel): cannot connect to %s: %s\n", sock, strerror(errno));
                return false;
            }
        } else if (srv != nullptr && srv[0] != '\0') {
            if (!spawn(srv)) return false;
        } else {
            return fail("not available until P5.3: set TSR_VKERNEL_SERVER (server command line) or TSR_VKERNEL_SOCKET "
                        "to a two-clock vkernel server (protocol version 3); there is no in-tree server yet");
        }
        tsr_vk_rsp r{};
        std::vector<uint8_t> pl;
        if (!call(VK_HELLO, TSR_VK_PROTO_VERSION, 0, 0, nullptr, 0, &r, &pl) || r.rc != 0) {
            return fail("the server refused the connection (protocol version 3 of the two-clock server, P5.3, or a start-up check)");
        }
        nch_ = r.val;
        chb_ = r.x;
        if (nch_ < 1 || nch_ > KERNEL_MAX_CH || pl.size() < 8ull * nch_) return fail("malformed HELLO response");
        base_.resize(nch_);
        memcpy(base_.data(), pl.data(), 8ull * nch_);
        desc_.assign((const char *) pl.data() + 8ull * nch_, pl.size() - 8ull * nch_);
        while (!desc_.empty() && desc_.back() == '\0') desc_.pop_back();
        {
            std::lock_guard<std::mutex> lk(live_mutex());
            live().push_back(this);
            static bool once = false;
            if (!once) {
                once = true;
                atexit(close_live_at_exit);
            }
        }
        return true;
    }

    // BYE, close, reap a spawned server (bounded wait, then SIGTERM).
    void shutdown() {
        if (fd_ >= 0) {
            tsr_vk_req q{VK_BYE, 0, 0, 0};
            if (write_all(fd_, &q, sizeof(q))) {
                tsr_vk_rsp r{};
                (void) read_all(fd_, &r, sizeof(r));
            }
            ::close(fd_);
            fd_ = -1;
        }
        if (pid_ > 0) {
            for (int i = 0; i < 600; ++i) {          // the server writes its summary, then exits
                int st = 0;
                const pid_t w = ::waitpid(pid_, &st, WNOHANG);
                if (w == pid_ || w < 0) {
                    pid_ = -1;
                    return;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            ::kill(pid_, SIGTERM);
            ::waitpid(pid_, nullptr, 0);
            pid_ = -1;
        }
    }

    const char * name() const override { return "vkernel"; }
    uint32_t num_ch() const override { return nch_; }
    uint64_t ch_bytes(uint32_t) const override { return chb_; }
    uint64_t ch_addr(uint32_t ch) const override { return base_[ch]; }
    std::string describe() const override { return "Verilated kernel server: " + desc_ + "\n"; }
    bool simulated() const override { return true; }

    int reg_read(uint32_t off, uint32_t * val) override {
        tsr_vk_rsp r{};
        if (!call(VK_REG_RD, off, 0, 0, nullptr, 0, &r, nullptr) || r.rc != 0) return -1;
        *val = r.val;
        return 0;
    }
    int reg_write(uint32_t off, uint32_t val) override {
        tsr_vk_rsp r{};
        return call(VK_REG_WR, off, val, 0, nullptr, 0, &r, nullptr) && r.rc == 0 ? 0 : -1;
    }

    int mem_write(uint32_t ch, uint64_t off, const void * src, size_t n) override {
        const uint8_t * s = (const uint8_t *) src;
        for (size_t done = 0; done < n;) {
            const size_t k = std::min<size_t>(TSR_VK_MAX_XFER, n - done);
            tsr_vk_rsp r{};
            if (!call(VK_MEM_WR, ch, off + done, k, s + done, k, &r, nullptr) || r.rc != 0) return -1;
            done += k;
        }
        return 0;
    }
    // EX §10.2: the server's clocks run only here (VK_ADVANCE); device time for the timeouts.
    void idle(uint32_t, uint64_t interval_ns) override {
        tsr_vk_rsp r{};
        const uint64_t cyc = (uint64_t) ((double) interval_ns * CORE_HZ_NOMINAL / 1e9) + 1;
        if (call(VK_ADVANCE, 0, cyc, 0, nullptr, 0, &r, nullptr) && r.rc == 0) dev_cycles_ = r.x;
    }
    uint64_t now_ns() override {
        tsr_vk_rsp r{};
        if (call(VK_TIME, 0, 0, 0, nullptr, 0, &r, nullptr) && r.rc == 0) dev_cycles_ = r.x;
        return (uint64_t) ((double) dev_cycles_ * 1e9 / CORE_HZ_NOMINAL);
    }

    int mem_read(uint32_t ch, uint64_t off, void * dst, size_t n) override {
        uint8_t * d = (uint8_t *) dst;
        for (size_t done = 0; done < n;) {
            const size_t k = std::min<size_t>(TSR_VK_MAX_XFER, n - done);
            tsr_vk_rsp r{};
            if (!call(VK_MEM_RD, ch, off + done, k, nullptr, 0, &r, nullptr, d + done, k) || r.rc != 0) return -1;
            done += k;
        }
        return 0;
    }

private:
    int      fd_ = -1;
    pid_t    pid_ = -1;
    uint32_t nch_ = 0;
    uint64_t chb_ = 0;
    std::vector<uint64_t> base_;
    std::string desc_;
    uint32_t viol_seen_ = 0;
    bool     dead_ = false;
    uint64_t dev_cycles_ = 0;   // last device time the server reported (core cycles)

    bool fail(const char * msg) {
        fprintf(stderr, "tsr_transport(vkernel): %s\n", msg);
        return false;
    }

    bool spawn(const char * cmdline) {
        std::vector<std::string> w = split_words(cmdline);
        if (w.empty()) return fail("TSR_VKERNEL_SERVER is empty");
        int sv[2];
        if (::socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0) return fail("socketpair failed");
        ::fcntl(sv[0], F_SETFD, FD_CLOEXEC);          // ours; sv[1] is inherited by the server
        w.push_back("--fd");
        w.push_back(std::to_string(sv[1]));
        std::vector<char *> argv;
        for (auto & s : w) argv.push_back(s.data());
        argv.push_back(nullptr);
        const int rc = ::posix_spawnp(&pid_, argv[0], nullptr, nullptr, argv.data(), environ);
        ::close(sv[1]);
        if (rc != 0) {
            ::close(sv[0]);
            pid_ = -1;
            fprintf(stderr, "tsr_transport(vkernel): cannot start %s: %s\n", argv[0], strerror(rc));
            return false;
        }
        fd_ = sv[0];
        return true;
    }

    // One request / response. out_pl receives a payload of any size; dst (dst_n bytes) receives
    // a payload of exactly that size (VK_MEM_RD). Returns false on a broken connection or when the
    // server reported new findings (printed here).
    bool call(uint32_t op, uint32_t a, uint64_t b, uint64_t c, const void * src, size_t src_n, tsr_vk_rsp * r,
              std::vector<uint8_t> * out_pl, void * dst = nullptr, size_t dst_n = 0) {
        if (dead_ || fd_ < 0) return false;
        tsr_vk_req q{op, a, b, c};
        if (!write_all(fd_, &q, sizeof(q)) || (src_n && !write_all(fd_, src, src_n)) || !read_all(fd_, r, sizeof(*r))) {
            dead_ = true;
            fprintf(stderr, "tsr_transport(vkernel): connection to the kernel server lost (op %u)\n", op);
            return false;
        }
        if (r->len) {
            if (dst != nullptr) {
                if (r->len != dst_n || !read_all(fd_, dst, dst_n)) {
                    dead_ = true;
                    fprintf(stderr, "tsr_transport(vkernel): bad payload from the kernel server (op %u)\n", op);
                    return false;
                }
            } else {
                std::vector<uint8_t> tmp(r->len);
                if (!read_all(fd_, tmp.data(), tmp.size())) {
                    dead_ = true;
                    return false;
                }
                if (out_pl) *out_pl = std::move(tmp);
            }
        }
        if (r->viol != viol_seen_ && op != VK_MSGS && op != VK_BYE) {
            report_findings(r->viol);
            return false;
        }
        return true;
    }

    void report_findings(uint32_t now) {
        const uint32_t n = now - viol_seen_;
        viol_seen_ = now;
        tsr_vk_rsp r{};
        std::vector<uint8_t> pl;
        call(VK_MSGS, 0, 0, 0, nullptr, 0, &r, &pl);
        fprintf(stderr, "tsr_transport(vkernel): the kernel server reported %u new verification finding%s:\n%s", n,
                n == 1 ? "" : "s", pl.empty() ? "(no text)\n" : std::string((const char *) pl.data(),
                                                                          strnlen((const char *) pl.data(), pl.size())).c_str());
    }

    friend void close_live_at_exit();
};

void close_live_at_exit() {
    std::vector<vkernel_kdev *> v;
    {
        std::lock_guard<std::mutex> lk(live_mutex());
        v = live();
    }
    for (vkernel_kdev * k : v) k->shutdown();
}

} // namespace

std::unique_ptr<kdev> open_vkernel_kdev() {
    auto k = std::make_unique<vkernel_kdev>();
    if (!k->open()) return nullptr;
    return k;
}

} // namespace tsr
