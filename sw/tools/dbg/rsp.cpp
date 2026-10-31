// rsp.cpp — libtsr_dbg: the GDB remote serial protocol server (docs/design/05-debug.md "DB" §4.1-§4.9):
// one connection at a time (TCP, Unix socket or stdio), all-stop, thread 1 = the hart.
//
// States (DB §4.2): DETACHED -> (connect) ATTACH -> HALTED; HALTED serves g G p P m M X Z z qRcmd and
// leaves with c / s / vCont (RUNNING / STEPPING) or D / k (a Ctrl-C while halted is ignored, as in
// gdbserver's all-stop mode); a connection that ends or fails to write (EOF, EPIPE, ECONNRESET; never
// SIGPIPE) ends the session like D, with the detach; RUNNING polls the hart (STATUS, dmstatus)
// every 100 µs of the time base backing off to 10 ms, forwards the decoded log and the answers of the
// standalone server's launches as O packets, and halts on Ctrl-C (T02); STEPPING reports a hart that
// does not halt within 5 ms with an O packet carrying an RV sample and keeps waiting. On the
// simulated kinds the polls advance the model (EX §10.2), so a session on sim is deterministic for a
// fixed schedule.

#include "tsr_dbg_lib.hpp"

#include "tsr_logdec.h"

#include <atomic>
#include <cerrno>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <csignal>
#include <poll.h>
#include <pthread.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

namespace tsr::dbg {

namespace {

const char * const k_hex = "0123456789abcdef";

std::string hex_bytes(const uint8_t * b, size_t n) {
    std::string s;
    s.reserve(2 * n);
    for (size_t i = 0; i < n; ++i) {
        s += k_hex[b[i] >> 4];
        s += k_hex[b[i] & 15];
    }
    return s;
}
std::string hex_str(const std::string & t) { return hex_bytes((const uint8_t *) t.data(), t.size()); }
std::string hex_le32(uint32_t v) {
    const uint8_t b[4] = { (uint8_t) v, (uint8_t) (v >> 8), (uint8_t) (v >> 16), (uint8_t) (v >> 24) };
    return hex_bytes(b, 4);
}
int hexval(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}
bool unhex(const std::string & s, std::vector<uint8_t> * out) {
    if (s.size() % 2) return false;
    out->clear();
    for (size_t i = 0; i < s.size(); i += 2) {
        const int a = hexval(s[i]), b = hexval(s[i + 1]);
        if (a < 0 || b < 0) return false;
        out->push_back((uint8_t) (a << 4 | b));
    }
    return true;
}
bool parse_hex(const std::string & s, uint64_t * v) {
    if (s.empty() || s.size() > 16) return false;
    uint64_t r = 0;
    for (char c : s) {
        const int h = hexval(c);
        if (h < 0) return false;
        r = r << 4 | (uint64_t) h;
    }
    *v = r;
    return true;
}
uint32_t le32_hex(const std::string & s, bool * ok) {
    std::vector<uint8_t> b;
    *ok = unhex(s, &b) && b.size() == 4;
    return *ok ? (uint32_t) (b[0] | b[1] << 8 | b[2] << 16 | (uint32_t) b[3] << 24) : 0;
}
std::string escape_bin(const std::string & s) {
    std::string o;
    for (char c : s) {
        if (c == '$' || c == '#' || c == '}' || c == '*') {
            o += '}';
            o += (char) (c ^ 0x20);
        } else {
            o += c;
        }
    }
    return o;
}

} // namespace

struct rsp_server::impl {
    int fd_in = -1, fd_out = -1, conn = -1;
    bool out_sock = false;         // fd_out is a socket (send with MSG_NOSIGNAL); else a pipe / tty (stdio)
    bool dead = false;             // a write failed or the peer vanished: the session ends (DB §4.2)
    bool noack = false;
    std::string inbuf;
    std::atomic<bool> stop{ false };
    bool verbose = false;
    stop_info last;               // the last stop (for '?')
    bool have_last = false;
    std::vector<std::string> sym_wanted;   // qSymbol requests still to make (no ELF)
    log_decoder dec;

    // Reads what is available within timeout_ms into inbuf: > 0 bytes, 0 nothing, -1 EOF / error.
    int read_some(int timeout_ms) {
        pollfd p{ fd_in, POLLIN, 0 };
        const int r = poll(&p, 1, timeout_ms);
        if (r < 0) return errno == EINTR ? 0 : -1;
        if (r == 0) return 0;
        char b[4096];
        const ssize_t n = ::read(fd_in, b, sizeof(b));
        if (n <= 0) return -1;
        inbuf.append(b, (size_t) n);
        return (int) n;
    }
    // One write that never raises SIGPIPE: the process that owns an embedded service is the application
    // (the ggml backend, tsr_rt, PyTorch), and a GDB that vanishes must end the session, not that process
    // (DB §4.1, §4.2). Sockets use MSG_NOSIGNAL; a pipe (stdio) blocks SIGPIPE on this thread around the
    // write and consumes the signal the write raised.
    ssize_t write_nosig(const char * b, size_t n) {
        if (out_sock) return ::send(fd_out, b, n, MSG_NOSIGNAL);
        sigset_t ss, old, pend;
        sigemptyset(&ss);
        sigaddset(&ss, SIGPIPE);
        pthread_sigmask(SIG_BLOCK, &ss, &old);
        sigpending(&pend);
        const bool was_pending = sigismember(&pend, SIGPIPE) == 1;
        const ssize_t r = ::write(fd_out, b, n);
        const int e = errno;
        if (r < 0 && e == EPIPE && !was_pending) {
            const timespec z{ 0, 0 };
            sigtimedwait(&ss, nullptr, &z);
        }
        pthread_sigmask(SIG_SETMASK, &old, nullptr);
        errno = e;
        return r;
    }
    // A failed write (EPIPE, ECONNRESET, ...) is the end of the session: dead stays set, every later
    // write is refused, and the loops in serve() leave and detach.
    bool write_all(const std::string & s) {
        if (dead) return false;
        size_t off = 0;
        while (off < s.size()) {
            const ssize_t n = write_nosig(s.data() + off, s.size() - off);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) {
                dead = true;
                return false;
            }
            off += (size_t) n;
        }
        return true;
    }
    // The first ack ('+' / '-') outside a packet. A complete client packet in front of it is skipped
    // (an ack can trail the client's next packet when the client answered something we did not wait for);
    // an incomplete packet ends the scan (what follows is its body).
    size_t find_ack() const {
        size_t i = 0;
        while (i < inbuf.size()) {
            const char c = inbuf[i];
            if (c == '+' || c == '-') return i;
            if (c == '$') {
                const size_t h = inbuf.find('#', i);
                if (h == std::string::npos || h + 2 >= inbuf.size()) return std::string::npos;
                i = h + 3;
                continue;
            }
            ++i;
        }
        return std::string::npos;
    }
    bool send(const std::string & payload) {
        if (dead) return false;
        uint8_t cs = 0;
        for (char c : payload) cs = (uint8_t) (cs + (uint8_t) c);
        char t[4];
        snprintf(t, sizeof(t), "#%02x", cs);
        const std::string pkt = "$" + payload + t;
        if (verbose) fprintf(stderr, "tsr-dbg rsp <- %.200s\n", payload.c_str());
        for (int attempt = 0; attempt < 5; ++attempt) {
            if (!write_all(pkt)) return false;
            if (noack) return true;
            // Wait for the ack ('+'; '-' asks for a retransmission).
            for (int i = 0; i < 50; ++i) {
                const size_t k = find_ack();
                if (k != std::string::npos) {
                    const char c = inbuf[k];
                    inbuf.erase(k, 1);
                    if (c == '+') return true;
                    break;   // retransmit
                }
                if (read_some(100) < 0) {
                    dead = true;
                    return false;
                }
            }
        }
        return false;
    }
    // One packet from inbuf (acked unless noack); interrupt set on 0x03 outside packets.
    // 1 packet, 0 none yet, -1 EOF.
    int get(std::string * pkt, int timeout_ms, bool * interrupt) {
        for (int round = 0; round < 2; ++round) {
            for (;;) {
                const size_t d = inbuf.find('$');
                const size_t ci = inbuf.find('\x03');
                if (ci != std::string::npos && (d == std::string::npos || ci < d)) {
                    inbuf.erase(ci, 1);
                    if (interrupt) *interrupt = true;
                    continue;
                }
                if (d == std::string::npos) {
                    // stray acks, noise
                    inbuf.clear();
                    break;
                }
                const size_t h = inbuf.find('#', d);
                if (h == std::string::npos || h + 2 >= inbuf.size()) {
                    if (d) inbuf.erase(0, d);
                    break;
                }
                const std::string body = inbuf.substr(d + 1, h - d - 1);
                const int c1 = hexval(inbuf[h + 1]), c2 = hexval(inbuf[h + 2]);
                inbuf.erase(0, h + 3);
                uint8_t cs = 0;
                for (char c : body) cs = (uint8_t) (cs + (uint8_t) c);
                if (!noack) {
                    if (c1 < 0 || c2 < 0 || (uint8_t) (c1 << 4 | c2) != cs) {
                        if (!write_all("-")) return -1;
                        continue;
                    }
                    if (!write_all("+")) return -1;   // the peer is gone: the packet is not executed
                }
                *pkt = body;
                if (verbose) fprintf(stderr, "tsr-dbg rsp -> %.200s\n", body.c_str());
                return 1;
            }
            if (round == 0) {
                const int r = read_some(timeout_ms);
                if (r < 0) return -1;
                if (r == 0) return 0;
            }
        }
        return 0;
    }
    void out(const std::string & text) {   // O packets (console output), ≤ 512 characters each
        for (size_t i = 0; i < text.size(); i += 512) send("O" + hex_str(text.substr(i, 512)));
    }
};

rsp_server::rsp_server(target & tg, monitor & mon, const server_options & o) : p_(new impl), tg_(tg), mon_(mon), o_(o) {
    p_->verbose = o.verbose;
    if (tg.image()) p_->dec.set_image(tg.image()->image());
}

rsp_server::~rsp_server() {
    if (p_->conn >= 0) close(p_->conn);
    if (lfd_ >= 0) close(lfd_);
    if (o_.listen.rfind("unix:", 0) == 0) unlink(o_.listen.c_str() + 5);
}

void rsp_server::request_stop() { p_->stop = true; }

int rsp_server::open_listener(std::string * why) {
    if (o_.listen == "stdio") return 0;
    if (o_.listen.rfind("tcp:", 0) == 0) {
        std::string spec = o_.listen.substr(4);
        uint32_t addr = INADDR_LOOPBACK;
        const size_t colon = spec.rfind(':');
        if (colon != std::string::npos) {
            if (spec.substr(0, colon) == "0.0.0.0" || spec.substr(0, colon) == "*") addr = INADDR_ANY;
            spec = spec.substr(colon + 1);
        }
        uint32_t port = 0;
        if (!parse_u32(spec, &port) || port > 65535) {
            if (why) *why = "bad TCP port in " + o_.listen;
            return -1;
        }
        lfd_ = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
        int one = 1;
        setsockopt(lfd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
        sockaddr_in sa{};
        sa.sin_family = AF_INET;
        sa.sin_port = htons((uint16_t) port);
        sa.sin_addr.s_addr = htonl(addr);
        if (lfd_ < 0 || bind(lfd_, (sockaddr *) &sa, sizeof(sa)) != 0 || listen(lfd_, 1) != 0) {
            if (why) *why = std::string("cannot listen on ") + o_.listen + ": " + strerror(errno);
            return -1;
        }
        socklen_t sl = sizeof(sa);
        getsockname(lfd_, (sockaddr *) &sa, &sl);
        port_ = ntohs(sa.sin_port);
        fprintf(stderr, "tsr-dbg: GDB server listening on tcp port %d (target remote :%d)\n", port_, port_);
    } else if (o_.listen.rfind("unix:", 0) == 0) {
        const std::string path = o_.listen.substr(5);
        lfd_ = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
        sockaddr_un su{};
        su.sun_family = AF_UNIX;
        if (path.size() >= sizeof(su.sun_path)) {
            if (why) *why = "unix socket path too long";
            return -1;
        }
        memcpy(su.sun_path, path.c_str(), path.size() + 1);
        unlink(path.c_str());
        if (lfd_ < 0 || bind(lfd_, (sockaddr *) &su, sizeof(su)) != 0 || listen(lfd_, 1) != 0) {
            if (why) *why = std::string("cannot listen on ") + o_.listen + ": " + strerror(errno);
            return -1;
        }
        fprintf(stderr, "tsr-dbg: GDB server listening on %s (target extended-remote %s)\n", path.c_str(), path.c_str());
    } else {
        if (why) *why = "listen must be tcp:<port>, unix:<path> or stdio";
        return -1;
    }
    if (!o_.port_file.empty()) {
        const std::string tmp = o_.port_file + ".tmp";
        if (FILE * f = fopen(tmp.c_str(), "w")) {
            fprintf(f, "%d\n", port_);
            fclose(f);
            rename(tmp.c_str(), o_.port_file.c_str());
        }
    }
    return 0;
}

int rsp_server::serve(std::string * why) {
    impl & P = *p_;
    if (o_.listen == "stdio") {
        P.fd_in = 0;
        P.fd_out = 1;
    } else {
        for (;;) {
            if (P.stop) return 0;
            pollfd pf{ lfd_, POLLIN, 0 };
            const int r = poll(&pf, 1, 100);
            if (r < 0 && errno != EINTR) {
                if (why) *why = std::string("poll: ") + strerror(errno);
                return -1;
            }
            if (r > 0) break;
        }
        P.conn = accept4(lfd_, nullptr, nullptr, SOCK_CLOEXEC);
        if (P.conn < 0) {
            if (why) *why = std::string("accept: ") + strerror(errno);
            return -1;
        }
        int one = 1;
        setsockopt(P.conn, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        P.fd_in = P.fd_out = P.conn;
        fprintf(stderr, "tsr-dbg: GDB connected\n");
    }
    {
        struct stat st;
        P.out_sock = fstat(P.fd_out, &st) == 0 && S_ISSOCK(st.st_mode);
    }
    P.dead = false;
    P.noack = false;
    P.inbuf.clear();
    P.have_last = false;
    tsr_transport * t = tg_.transport();
    const bool host_time = strcmp(tsr_transport_name(t), "xrt") == 0;

    // DB §4.2 ATTACH
    std::string awhy;
    std::unique_lock<std::mutex> alk;
    if (o_.serial_mu) alk = std::unique_lock<std::mutex>(*o_.serial_mu);
    int ae = tg_.attach(&awhy);
    if (ae == 0) {
        P.last = stop_info();
        tg_.stop(&P.last);
        P.have_last = true;
        if (!P.last.note.empty()) fprintf(stderr, "tsr-dbg: %s\n", P.last.note.c_str());
    } else {
        fprintf(stderr, "tsr-dbg: attach failed: %s (DEGRADED: monitor commands only)\n", awhy.c_str());
    }
    if (tg_.image() && !tg_.image()->loaded()) P.sym_wanted = { "__tsr_dbg", "__tsr_crash", "__tsr_launch_halt", "__tsr_trap_halt" };
    if (alk.owns_lock()) alk.unlock();

    auto stop_reply = [&](const stop_info & si) {
        char b[64];
        snprintf(b, sizeof(b), "T%02xthread:1;", si.signal);
        std::string r = b;
        if (si.reason == "swbreak" || si.reason == "hwbreak") r += si.reason + ":;";
        else if (!si.reason.empty()) {
            snprintf(b, sizeof(b), "%s:%x;", si.reason.c_str(), si.watch_addr);
            r += b;
        }
        static const uint32_t ex[4] = { 32, 2, 1, 8 };   // DB §4.7: pc, sp, ra, s0
        for (uint32_t n : ex) {
            uint32_t v = 0;
            if (tg_.read_reg(n, &v) == 0) {
                snprintf(b, sizeof(b), "%02x:%s;", n, hex_le32(v).c_str());
                r += b;
            }
        }
        return r;
    };
    auto ecode = [](int e) {
        char b[8];
        snprintf(b, sizeof(b), "E%02x", e & 0xFF);
        return std::string(b);
    };
    auto drain_log_o = [&]() {
        std::vector<uint32_t> w(4096);
        size_t n = 0;
        if (tsr_transport_log_drain(t, w.data(), w.size(), &n) == 0 && n) {
            std::string txt;
            P.dec.feed(w.data(), n, [&](const std::string & l) { txt += l + "\n"; });
            if (!txt.empty()) P.out(txt);
        }
    };

    // RUNNING / STEPPING until the hart halts (Ctrl-C halts it). false: the connection ended.
    auto run_until_stop = [&](bool stepping) -> bool {
        uint64_t interval = 100000;   // 100 µs, backing off to 10 ms
        const uint64_t t0 = tsr_transport_now_ns(t);
        bool warned = false;
        for (;;) {
            if (P.stop || P.dead) return false;
            bool intr = false;
            std::string pkt;
            int g = P.get(&pkt, 0, &intr);
            if (g < 0) return false;
            if (intr) tg_.interrupt();
            if (g > 0) fprintf(stderr, "tsr-dbg: packet while running ignored: %.60s\n", pkt.c_str());
            if (o_.tick) o_.tick();
            if (tg_.poll_halted()) {
                stop_info si;
                tg_.stop(&si);
                drain_log_o();
                const std::string ans = mon_.poll_launches();
                if (!ans.empty()) P.out(ans);
                if (!si.note.empty()) P.out(si.note + "\n");
                P.last = si;
                P.have_last = true;
                return P.send(stop_reply(si));
            }
            const std::string ans = mon_.poll_launches();
            if (!ans.empty()) P.out(ans);
            drain_log_o();
            if (stepping && !warned && tsr_transport_now_ns(t) - t0 > 5000000) {
                std::string s;
                tg_.rv_sample(&s);
                P.out("step: no halt within 5 ms (DB §4.2); " + s + "\n");
                warned = true;
            }
            const uint32_t st = tg_.csr(TSR_CSR_STATUS);
            const bool idle_dev = (st & TSR_STATUS_RV_SLEEP) && (st & TSR_STATUS_QIDLE) && mon_.pending.empty() && mon_.deferred.empty() &&
                                  !mon_.boot_running();
            if (host_time || idle_dev) {
                // Nothing moves without the host (or host time): wait on the connection.
                const int r = P.read_some((int) std::max<uint64_t>(1, interval / 1000000));
                if (r < 0) return false;
                interval = std::min<uint64_t>(interval * 2, 10000000);
                if (host_time) tsr_transport_idle(t, 1000);
            } else {
                tsr_transport_idle(t, 36000);   // ≈ 10 K core cycles of the time base (DB §9.1)
                interval = 100000;
            }
        }
    };

    int rc = 0;
    bool done = false;
    while (!done) {
        if (P.stop || P.dead) break;
        std::string pkt;
        bool intr = false;
        const int g = P.get(&pkt, 100, &intr);
        if (g < 0) break;
        // Ctrl-C (0x03) while the target is halted is ignored, as gdbserver does in all-stop mode: a
        // second stop reply would be taken as the answer to the client's next request and shift every
        // later reply of the session (a breakpoint hit racing the user's Ctrl-C).
        (void) intr;
        if (g == 0) {
            if (o_.tick) {
                std::unique_lock<std::mutex> lk;
                if (o_.serial_mu) lk = std::unique_lock<std::mutex>(*o_.serial_mu);
                o_.tick();
            }
            continue;
        }
        std::unique_lock<std::mutex> lk;
        if (o_.serial_mu) lk = std::unique_lock<std::mutex>(*o_.serial_mu);
        if (pkt.empty()) {
            P.send("");
            continue;
        }
        const char c = pkt[0];
        std::string rest = pkt.substr(1);
        if (c == '?') {
            if (!tg_.attached()) P.send("W00");
            else P.send(stop_reply(P.have_last ? P.last : stop_info()));
        } else if (pkt.rfind("qSupported", 0) == 0) {
            P.send("PacketSize=4000;qXfer:features:read+;qXfer:memory-map:read+;swbreak+;hwbreak+;vContSupported+;QStartNoAckMode+");
        } else if (pkt == "QStartNoAckMode") {
            P.send("OK");
            P.noack = true;
        } else if (pkt.rfind("qXfer:features:read:target.xml:", 0) == 0 || pkt.rfind("qXfer:memory-map:read::", 0) == 0) {
            const bool feat = pkt[6] == 'f';
            const std::string doc = feat ? tg_.target_xml() : tg_.memory_map_xml();
            const std::string ol = pkt.substr(pkt.rfind(':') + 1);
            const size_t comma = ol.find(',');
            uint64_t off = 0, len = 0;
            if (comma == std::string::npos || !parse_hex(ol.substr(0, comma), &off) || !parse_hex(ol.substr(comma + 1), &len)) {
                P.send("E06");
            } else if (off >= doc.size()) {
                P.send("l");
            } else {
                const std::string part = doc.substr(off, len);
                P.send(std::string(off + part.size() >= doc.size() ? "l" : "m") + escape_bin(part));
            }
        } else if (pkt.rfind("qXfer:", 0) == 0) {
            P.send("");
        } else if (pkt == "qC") {
            P.send("QC1");
        } else if (pkt == "qfThreadInfo") {
            P.send("m1");
        } else if (pkt == "qsThreadInfo") {
            P.send("l");
        } else if (pkt.rfind("qAttached", 0) == 0) {
            P.send("1");
        } else if (pkt.rfind("qThreadExtraInfo", 0) == 0) {
            P.send(hex_str(tg_.halted() ? "hart 0 (halted)" : "hart 0 (running)"));
        } else if (pkt.rfind("qSymbol:", 0) == 0) {
            // qSymbol:: starts; qSymbol:<value>:<hex name> answers the previous request (DB §4.3).
            const std::string body = pkt.substr(8);
            const size_t col = body.find(':');
            if (col != std::string::npos && col + 1 < body.size()) {
                std::vector<uint8_t> nm;
                uint64_t v = 0;
                if (unhex(body.substr(col + 1), &nm) && parse_hex(body.substr(0, col), &v) && tg_.image())
                    tg_.image()->given[std::string(nm.begin(), nm.end())] = (uint32_t) v;
            }
            if (!P.sym_wanted.empty()) {
                const std::string n = P.sym_wanted.front();
                P.sym_wanted.erase(P.sym_wanted.begin());
                P.send("qSymbol:" + hex_str(n));
            } else {
                P.send("OK");
                uint32_t a = 0;
                if (tg_.attached() && tg_.sym("__tsr_dbg", &a)) tg_.set_dbg_word(TSR_DBG_INFO_ATTACHED_WORD, 1);
            }
        } else if (pkt.rfind("qRcmd,", 0) == 0) {
            std::vector<uint8_t> cmd;
            if (!unhex(pkt.substr(6), &cmd)) {
                P.send("E06");
                continue;
            }
            const std::string line(cmd.begin(), cmd.end());
            std::string o;
            int e = 0;
            if (line == "why") o = P.have_last ? (P.last.note.empty() ? "stopped (no note)\n" : P.last.note + "\n") : "no stop yet\n";
            else e = mon_.run(line, &o);
            if (!o.empty()) P.out(o);
            P.send(e ? ecode(e) : "OK");
        } else if (c == 'H' || c == 'T') {
            P.send("OK");
        } else if (c == 'g') {
            uint32_t r[33];
            const int e = tg_.read_gprs(r);
            if (e) P.send(ecode(e));
            else {
                std::string s;
                for (uint32_t i = 0; i < 33; ++i) s += hex_le32(r[i]);
                P.send(s);
            }
        } else if (c == 'G') {
            int e = 0;
            for (uint32_t i = 1; i < 33 && !e && rest.size() >= 8 * (i + 1); ++i) {
                bool ok = false;
                const uint32_t v = le32_hex(rest.substr(8 * i, 8), &ok);
                if (!ok) e = E_MALFORMED;
                else e = tg_.write_reg(i, v);
            }
            P.send(e ? ecode(e) : "OK");
        } else if (c == 'p') {
            uint64_t n = 0;
            uint32_t v = 0;
            if (!parse_hex(rest, &n)) P.send("E06");
            else {
                const int e = tg_.read_reg((uint32_t) n, &v);
                P.send(e ? ecode(e) : hex_le32(v));
            }
        } else if (c == 'P') {
            const size_t eq = rest.find('=');
            uint64_t n = 0;
            bool ok = false;
            const uint32_t v = eq == std::string::npos ? 0 : le32_hex(rest.substr(eq + 1), &ok);
            if (eq == std::string::npos || !parse_hex(rest.substr(0, eq), &n) || !ok) P.send("E06");
            else {
                const int e = tg_.write_reg((uint32_t) n, v);
                P.send(e ? ecode(e) : "OK");
            }
        } else if (c == 'm') {
            const size_t comma = rest.find(',');
            uint64_t a = 0, n = 0;
            if (comma == std::string::npos || !parse_hex(rest.substr(0, comma), &a) || !parse_hex(rest.substr(comma + 1), &n) || n > 0x2000) {
                P.send("E06");
            } else {
                std::vector<uint8_t> b((size_t) n);
                const int e = tg_.read_mem((uint32_t) a, b.data(), b.size());
                P.send(e ? ecode(e) : hex_bytes(b.data(), b.size()));
            }
        } else if (c == 'M' || c == 'X') {
            const size_t comma = rest.find(','), col = rest.find(':');
            uint64_t a = 0, n = 0;
            if (comma == std::string::npos || col == std::string::npos || !parse_hex(rest.substr(0, comma), &a) ||
                !parse_hex(rest.substr(comma + 1, col - comma - 1), &n)) {
                P.send("E06");
                continue;
            }
            std::vector<uint8_t> b;
            bool ok = true;
            if (c == 'M') {
                ok = unhex(rest.substr(col + 1), &b);
            } else {
                for (size_t i = col + 1; i < rest.size(); ++i) {
                    if (rest[i] == '}' && i + 1 < rest.size()) b.push_back((uint8_t) (rest[++i] ^ 0x20));
                    else b.push_back((uint8_t) rest[i]);
                }
            }
            if (!ok || b.size() != n) {
                P.send("E06");
                continue;
            }
            const int e = n ? tg_.write_mem((uint32_t) a, b.data(), b.size()) : 0;
            P.send(e ? ecode(e) : "OK");
        } else if (c == 'Z' || c == 'z') {
            const std::vector<std::string> f = [&] {
                std::vector<std::string> v;
                size_t i = 0;
                while (i <= rest.size()) {
                    const size_t j = rest.find_first_of(",;", i);
                    v.push_back(rest.substr(i, j == std::string::npos ? std::string::npos : j - i));
                    if (j == std::string::npos) break;
                    i = j + 1;
                }
                return v;
            }();
            uint64_t ty = 0, a = 0, k = 0;
            if (f.size() < 3 || !parse_hex(f[0], &ty) || !parse_hex(f[1], &a) || !parse_hex(f[2], &k) || ty > 4) {
                P.send(f.size() >= 1 && f[0].size() == 1 && f[0][0] > '4' ? "" : "E06");
                continue;
            }
            const int e = c == 'Z' ? tg_.insert_bp((int) ty, (uint32_t) a, (uint32_t) k) : tg_.remove_bp((int) ty, (uint32_t) a, (uint32_t) k);
            P.send(e ? ecode(e) : "OK");
        } else if (c == 'c' || c == 'C' || c == 's' || c == 'S' || pkt.rfind("vCont;", 0) == 0) {
            bool step = c == 's' || c == 'S';
            std::string addr;
            if (c == 'c' || c == 's') addr = rest;
            else if (c == 'C' || c == 'S') {
                const size_t sc = rest.find(';');
                if (sc != std::string::npos) addr = rest.substr(sc + 1);
            } else {
                // vCont;<action>[:thread];...: the first action for thread 1 or every thread.
                const std::string acts = pkt.substr(6);
                const char a0 = acts.empty() ? 'c' : acts[0];
                if (a0 == 't') {   // stop
                    tg_.interrupt();
                    if (!run_until_stop(false)) break;
                    continue;
                }
                step = a0 == 's' || a0 == 'S';
            }
            if (!tg_.attached()) {
                P.send("E05");
                continue;
            }
            if (!addr.empty()) {
                uint64_t a = 0;
                if (parse_hex(addr, &a)) tg_.write_reg(32, (uint32_t) a);
            }
            const int e = step ? tg_.start_step() : tg_.resume();
            if (e) {
                P.send(ecode(e));
                continue;
            }
            if (!run_until_stop(step)) break;
        } else if (pkt == "vCont?") {
            P.send("vCont;c;C;s;S;t");
        } else if (pkt == "vMustReplyEmpty") {
            P.send("");
        } else if (c == 'D') {
            tg_.detach(o_.detach_halted);
            P.send("OK");
            done = true;
        } else if (c == 'k' || pkt.rfind("vKill", 0) == 0) {
            // DB §4.2 KILL: the standalone server restarts the RISC-V without HALT_ON_BOOT; the
            // embedded service detaches (the owner keeps running).
            if (pkt[0] == 'v') P.send("OK");
            if (!o_.sessions_forever && tg_.attached()) {
                std::string o;
                tg_.detach(true);
                mon_.run("reset run", &o);
            } else {
                tg_.detach(false);
            }
            killed_ = true;
            done = true;
        } else {
            P.send("");   // unsupported (vRun, R, qOffsets, non-stop, vFile, tracepoints: DB §4.3)
        }
    }
    {
        std::unique_lock<std::mutex> dlk;
        if (o_.serial_mu) dlk = std::unique_lock<std::mutex>(*o_.serial_mu);
        if (tg_.attached()) tg_.detach(o_.detach_halted);
    }
    if (P.conn >= 0) {
        close(P.conn);
        P.conn = -1;
    }
    fprintf(stderr, "tsr-dbg: GDB session ended%s\n", killed_ ? " (kill)" : "");
    return rc;
}

} // namespace tsr::dbg
