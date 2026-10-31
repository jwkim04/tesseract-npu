// embed.cpp — libtsr_dbg: the embedded debug service (docs/design/05-debug.md "DB" DB-17, §4.1, §9.5).
//
// One process owns the device (xrt::ip is exclusive, EX §2.4), so the debugger runs inside it: the
// transport starts this service at open when TSR_GDB=tcp:<port>|unix:<path> or TSR_DEBUGGER=
// sock:<path> is set (tsr_transport_set_debug_service, registered by embed_install). Its threads
// use the device through the transport like any caller (the transport's mutex serialises every
// access; the owner's waits release it between polls):
//   gdb   an RSP server (rsp.cpp) accepting one GDB at a time, again after each session;
//   cli   a Unix socket for tsr-dbg attach: one command line per request, the monitor's output
//         back, ending with a line "% <rc>".
// TSR_GDB_WAIT=boot: the first boot halts the hart before its first instruction (HALT_ON_BOOT,
// before the load) and the boot's HELLO wait waits for the debugger (it counts no time while the
// hart is halted). TSR_GDB_WAIT=launch: after the first boot __tsr_dbg.attached = 1 and
// launch_break = all, so the first launch halts at __tsr_launch_halt (DB §3.8); its wait is
// debug-aware. The boot hooks run on the owner's thread with the transport mutex held; device
// sequences that must not interleave with a session's (SBA) are deferred to the session's thread.

#include "tsr_dbg_lib.hpp"

#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <poll.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <thread>
#include <unistd.h>

namespace tsr::dbg {

namespace {

struct service {
    tsr_transport * t = nullptr;
    std::string gdb, cli, wait;
    image_info img;
    std::unique_ptr<target> tg;
    std::unique_ptr<monitor> mon;
    std::unique_ptr<rsp_server> srv;
    std::mutex mu;                       // the session's device work and the CLI's
    std::thread gdb_th, cli_th;
    std::atomic<bool> stop{ false };
    std::atomic<bool> want_attach_flag{ false };   // deferred: __tsr_dbg.attached (+ launch_break)
    std::atomic<uint32_t> want_launch_break{ 0 };
    int boots = 0;
    int cli_fd = -1;
    bool image_set = false;
};

// Deferred work of the boot hooks, on a thread that may take the service mutex.
void apply_deferred(service * s) {
    if (!s->want_attach_flag.exchange(false)) return;
    if (!s->image_set) return;
    if (!(tsr_transport_csr_read(s->t, TSR_CSR_DBG_CFG) & TSR_DBG_CFG_DMI_EN)) tsr_transport_dmi_enable(s->t, 1);
    s->tg->dm().activate();
    if (!s->tg->attached()) {
        // No session yet: __tsr_dbg.attached makes the firmware execute ebreak (launch halt, trap
        // halt), which needs dcsr.ebreakm = 1, or the ebreak traps and the trap handler parks the
        // hart (crt0.S step 1). Halt briefly to set it, then let the hart run on.
        dmi & d = s->tg->dm();
        if (d.halt(5000000) == 0) {
            uint32_t dcsr = 0;
            if (d.reg_read(0x7B0, &dcsr) == 0) d.reg_write(0x7B0, dcsr | (1u << 15));
            d.resume(5000000);
        } else {
            fprintf(stderr, "tsr-dbg: the hart does not halt: dcsr.ebreakm not set, __tsr_dbg.attached left 0\n");
            return;
        }
    }
    s->tg->set_dbg_word(TSR_DBG_INFO_ATTACHED_WORD, 1);
    const uint32_t lb = s->want_launch_break.exchange(0);
    if (lb) s->tg->set_dbg_word(TSR_DBG_INFO_LAUNCH_BREAK_WORD, lb);
}

void cli_loop(service * s) {
    while (!s->stop) {
        pollfd p{ s->cli_fd, POLLIN, 0 };
        if (poll(&p, 1, 100) <= 0) continue;
        const int c = accept4(s->cli_fd, nullptr, nullptr, SOCK_CLOEXEC);
        if (c < 0) continue;
        std::string buf;
        char b[1024];
        for (;;) {
            pollfd q{ c, POLLIN, 0 };
            if (s->stop) break;
            if (poll(&q, 1, 100) <= 0) continue;
            const ssize_t n = read(c, b, sizeof(b));
            if (n <= 0) break;
            buf.append(b, (size_t) n);
            size_t nl;
            while ((nl = buf.find('\n')) != std::string::npos) {
                const std::string line = buf.substr(0, nl);
                buf.erase(0, nl + 1);
                std::string out;
                int rc;
                {
                    std::lock_guard<std::mutex> lk(s->mu);
                    apply_deferred(s);
                    rc = s->mon->run(line, &out);
                }
                char tail[32];
                snprintf(tail, sizeof(tail), "%% %d\n", rc);
                out += tail;
                // MSG_NOSIGNAL: a CLI client that vanishes must not kill the owning process (DB §4.1).
                size_t off = 0;
                bool gone = false;
                while (off < out.size() && !gone) {
                    const ssize_t w = send(c, out.data() + off, out.size() - off, MSG_NOSIGNAL);
                    if (w < 0 && errno == EINTR) continue;
                    if (w <= 0) gone = true;
                    else off += (size_t) w;
                }
                if (gone) break;
            }
        }
        close(c);
    }
}

void * svc_start(tsr_transport * t) {
    const char * g = getenv("TSR_GDB");
    const char * c = getenv("TSR_DEBUGGER");
    const char * w = getenv("TSR_GDB_WAIT");
    std::unique_ptr<service> s(new service);
    s->t = t;
    s->gdb = g ? g : "";
    s->cli = c ? c : "";
    s->wait = w ? w : "";
    if (!s->gdb.empty() && s->gdb.rfind("tcp:", 0) != 0 && s->gdb.rfind("unix:", 0) != 0) {
        fprintf(stderr, "tsr-dbg: TSR_GDB=\"%s\": expected tcp:<port> or unix:<path>; the debug service is off\n", s->gdb.c_str());
        return nullptr;
    }
    if (!s->cli.empty() && s->cli.rfind("sock:", 0) != 0) {
        fprintf(stderr, "tsr-dbg: TSR_DEBUGGER=\"%s\": expected sock:<path>; the debug service is off\n", s->cli.c_str());
        return nullptr;
    }
    s->tg.reset(new target(t, &s->img));
    s->mon.reset(new monitor(*s->tg, false));
    if (!s->gdb.empty()) {
        server_options o;
        o.listen = s->gdb;
        o.sessions_forever = true;
        o.serial_mu = &s->mu;
        service * sp = s.get();
        o.tick = [sp] { apply_deferred(sp); };
        if (const char * pf = getenv("TSR_GDB_PORT_FILE")) o.port_file = pf;
        s->srv.reset(new rsp_server(*s->tg, *s->mon, o));
        std::string why;
        if (s->srv->open_listener(&why) != 0) {
            fprintf(stderr, "tsr-dbg: embedded GDB service: %s\n", why.c_str());
            return nullptr;
        }
        s->gdb_th = std::thread([sp] {
            while (!sp->stop) {
                std::string err;
                if (sp->srv->serve(&err) != 0) {
                    fprintf(stderr, "tsr-dbg: embedded GDB service: %s\n", err.c_str());
                    break;
                }
            }
        });
    }
    if (!s->cli.empty()) {
        const std::string path = s->cli.substr(5);
        s->cli_fd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
        sockaddr_un su{};
        su.sun_family = AF_UNIX;
        if (path.size() < sizeof(su.sun_path)) {
            memcpy(su.sun_path, path.c_str(), path.size() + 1);
            unlink(path.c_str());
            if (bind(s->cli_fd, (sockaddr *) &su, sizeof(su)) == 0 && listen(s->cli_fd, 1) == 0) {
                fprintf(stderr, "tsr-dbg: embedded CLI on %s (tsr-dbg attach %s)\n", path.c_str(), path.c_str());
                service * sp = s.get();
                s->cli_th = std::thread([sp] { cli_loop(sp); });
            } else {
                fprintf(stderr, "tsr-dbg: TSR_DEBUGGER: cannot listen on %s: %s\n", path.c_str(), strerror(errno));
            }
        }
    }
    return s.release();
}

void svc_boot(void * h, tsr_transport * t, int phase) {
    service * s = (service *) h;
    if (phase == 0) {
        if (s->wait == "boot" && s->boots == 0) {
            // CSR writes only (single accesses): safe beside a session.
            tsr_transport_csr_write(t, TSR_CSR_DBG_CFG, tsr_transport_csr_read(t, TSR_CSR_DBG_CFG) | TSR_DBG_CFG_DMI_EN |
                                                            TSR_DBG_CFG_RV_HALT_EN | TSR_DBG_CFG_HALT_ON_BOOT);
            tsr_transport_dmi_write(t, TSR_DMI_DMCONTROL, 1u);   // dmactive
            tsr_transport_set_debugger(t, 1);
            s->tg->note_boot_halt();
            fprintf(stderr, "tsr-dbg: TSR_GDB_WAIT=boot: the hart halts at its first instruction; connect GDB (%s) and continue\n",
                    s->gdb.c_str());
        }
        return;
    }
    s->boots++;
    if (const tsr_image * im = tsr_transport_image(t)) {
        if (!s->image_set || s->img.image() != im) s->img.load(im, nullptr);
        s->image_set = true;
    }
    if (s->wait == "launch" && s->boots == 1) {
        s->want_launch_break = 0xFFFFFFFFu;
        s->want_attach_flag = true;
        tsr_transport_set_debugger(t, 1);
        fprintf(stderr, "tsr-dbg: TSR_GDB_WAIT=launch: the first launch halts at __tsr_launch_halt; connect GDB (%s)\n", s->gdb.c_str());
    } else if (s->tg->attached()) {
        s->want_attach_flag = true;   // the boot reloaded DTCM: __tsr_dbg.attached is 0 again
    }
    // No session thread may be running yet (TSR_GDB_WAIT=launch before any connection): apply now.
    if (s->want_attach_flag && !s->tg->attached()) apply_deferred(s);
}

void svc_stop(void * h) {
    service * s = (service *) h;
    s->stop = true;
    if (s->srv) s->srv->request_stop();
    if (s->gdb_th.joinable()) s->gdb_th.join();
    if (s->cli_th.joinable()) s->cli_th.join();
    if (s->cli_fd >= 0) {
        close(s->cli_fd);
        unlink(s->cli.substr(5).c_str());
    }
    {
        std::lock_guard<std::mutex> lk(s->mu);
        s->mon.reset();
        if (s->tg && s->tg->attached()) s->tg->detach(false);
    }
    delete s;
}

} // namespace

void embed_install() {
    static const tsr_debug_service svc = { svc_start, svc_boot, svc_stop };
    tsr_transport_set_debug_service(&svc);
}

} // namespace tsr::dbg

extern "C" void tsr_dbg_embed_install(void) { tsr::dbg::embed_install(); }
