// tsr_dbg.cpp — tsr-dbg, the debugger CLI (docs/design/05-debug.md "DB" §4.1, §9.3; docs/design/
// 06-software-verification.md "SW" §11 P3.7). Built on libtsr_dbg (sw/tools/dbg/tsr_dbg_lib.hpp).
//
//   tsr-dbg gdb [--device KIND] [--elf F] [--port N | --unix P | --stdio] [--attach | --load | --halt-on-boot]
//               [--port-file F] [--detach-halted] [-v]
//       the standalone GDB RSP server (DB §4.1): it owns the device. `gdbserver` is the same command.
//       Default: --load when an image is known (--elf, TSR_IMAGE, the build's tsr-fw.elf); --attach
//       opens with TSR_ATTACH semantics (no reset, no load); --halt-on-boot loads with HALT_ON_BOOT
//       (the hart stops before its first instruction, DB §3.8). One GDB session; the server exits
//       when it ends (D, k or a closed connection).
//   tsr-dbg [--device KIND] [--elf F] [--attach | --load | --halt-on-boot] [--json] [-ex CMD]... [-x FILE] [CMD ...]
//       batch mode (CI, agents): every -ex command, the lines of -x FILE, then CMD ... as one
//       command, through the monitor set (DB §4.8, §9.3: status, units, hist, tile, rv, perf, spm,
//       mem, sreg, log, crash, dump, recover, trw, launch, ping, ...); --json prints one object per
//       command. The default open is --attach (diagnosis of a device another process left).
//   tsr-dbg attach <socket> [-ex CMD]...
//       the CLI of an embedded service (TSR_DEBUGGER=sock:<path>, DB §9.5); commands from -ex or stdin.
//   tsr-dbg diverge <trace> ...   not implemented (DB §9.4; reported)
//
// Exit status: 0 when every command succeeded, 1 otherwise, 2 for usage errors.

#include "tsr_dbg_lib.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <vector>

using namespace tsr::dbg;

namespace {

void usage() {
    fprintf(stderr,
            "usage: tsr-dbg gdb|gdbserver [--device KIND] [--elf F] [--port N | --unix P | --stdio] [--attach | --load | --halt-on-boot]\n"
            "                             [--port-file F] [--detach-halted] [-v]\n"
            "       tsr-dbg [--device KIND] [--elf F] [--attach | --load | --halt-on-boot] [--json] [-ex CMD]... [-x FILE] [CMD ...]\n"
            "       tsr-dbg attach <socket> [-ex CMD]...\n"
            "commands: tsr-dbg -ex help\n");
}

std::string json_str(const std::string & s) {
    std::string o = "\"";
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') o += '\\', o += (char) c;
        else if (c == '\n') o += "\\n";
        else if (c < 0x20) {
            char b[8];
            snprintf(b, sizeof(b), "\\u%04x", c);
            o += b;
        } else o += (char) c;
    }
    return o + "\"";
}

int attach_cli(const std::string & path, const std::vector<std::string> & cmds) {
    const int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    sockaddr_un su{};
    su.sun_family = AF_UNIX;
    if (path.size() >= sizeof(su.sun_path)) return 2;
    memcpy(su.sun_path, path.c_str(), path.size() + 1);
    if (connect(fd, (sockaddr *) &su, sizeof(su)) != 0) {
        fprintf(stderr, "tsr-dbg attach: cannot connect to %s: %s\n", path.c_str(), strerror(errno));
        return 1;
    }
    std::vector<std::string> lines = cmds;
    if (lines.empty()) {
        std::string l;
        while (std::getline(std::cin, l)) lines.push_back(l);
    }
    int worst = 0;
    std::string buf;
    for (const std::string & l : lines) {
        const std::string m = l + "\n";
        if (send(fd, m.data(), m.size(), MSG_NOSIGNAL) < 0) return 1;
        for (;;) {
            const size_t k = buf.find("\n% ");
            const size_t k0 = buf.rfind("% ", 0) == 0 ? 0 : std::string::npos;
            const size_t at = k0 == 0 ? 0 : (k == std::string::npos ? std::string::npos : k + 1);
            if (at != std::string::npos) {
                const size_t e = buf.find('\n', at);
                if (e != std::string::npos) {
                    fwrite(buf.data(), 1, at, stdout);
                    const int rc = atoi(buf.c_str() + at + 2);
                    if (rc) worst = 1;
                    buf.erase(0, e + 1);
                    break;
                }
            }
            char b[4096];
            const ssize_t n = read(fd, b, sizeof(b));
            if (n <= 0) return 1;
            buf.append(b, (size_t) n);
        }
    }
    close(fd);
    return worst;
}

} // namespace

int main(int argc, char ** argv) {
    std::vector<std::string> args(argv + 1, argv + argc);
    if (args.empty()) {
        usage();
        return 2;
    }
    if (args[0] == "attach") {
        if (args.size() < 2) {
            usage();
            return 2;
        }
        std::vector<std::string> cmds;
        for (size_t i = 2; i < args.size(); ++i)
            if (args[i] == "-ex" && i + 1 < args.size()) cmds.push_back(args[++i]);
        return attach_cli(args[1], cmds);
    }
    bool gdb = false;
    std::string device, elf, listen, port_file, mode;
    bool json = false, detach_halted = false, verbose = false;
    std::vector<std::string> cmds, words;
    size_t i = 0;
    if (args[0] == "gdb" || args[0] == "gdbserver") {
        gdb = true;
        i = 1;
    }
    if (!args.empty() && args[0] == "diverge") {
        fprintf(stderr, "tsr-dbg diverge: not implemented yet (DB §9.4: serial-mode stop points and QSTEP against the simulator)\n");
        return 2;
    }
    for (; i < args.size(); ++i) {
        const std::string & a = args[i];
        auto val = [&]() -> std::string {
            if (i + 1 >= args.size()) {
                fprintf(stderr, "tsr-dbg: %s needs a value\n", a.c_str());
                exit(2);
            }
            return args[++i];
        };
        if (a == "--device") device = val();
        else if (a == "--elf" || a == "--image") elf = val();
        else if (a == "--port") listen = "tcp:" + val();
        else if (a == "--unix") listen = "unix:" + val();
        else if (a == "--stdio") listen = "stdio";
        else if (a == "--port-file") port_file = val();
        else if (a == "--attach" || a == "--load" || a == "--halt-on-boot") mode = a.substr(2);
        else if (a == "--detach-halted") detach_halted = true;
        else if (a == "--json") json = true;
        else if (a == "-v") verbose = true;
        else if (a == "-ex") cmds.push_back(val());
        else if (a == "-x") {
            FILE * f = fopen(val().c_str(), "r");
            if (!f) {
                fprintf(stderr, "tsr-dbg: cannot read %s\n", args[i].c_str());
                return 2;
            }
            char b[1024];
            while (fgets(b, sizeof(b), f)) {
                std::string l(b);
                while (!l.empty() && (l.back() == '\n' || l.back() == '\r')) l.pop_back();
                if (!l.empty() && l[0] != '#') cmds.push_back(l);
            }
            fclose(f);
        } else if (a == "-h" || a == "--help") {
            usage();
            return 0;
        } else if (!a.empty() && a[0] == '-' && words.empty()) {
            fprintf(stderr, "tsr-dbg: unknown option %s\n", a.c_str());
            usage();
            return 2;
        } else {
            words.push_back(a);
        }
    }
    if (!words.empty()) {
        std::string c;
        for (const std::string & w : words) c += (c.empty() ? "" : " ") + w;
        cmds.push_back(c);
    }
    if (mode.empty()) mode = gdb ? "load" : "attach";
    if (mode == "attach") setenv("TSR_ATTACH", "1", 1);
    // The firmware image (symbols; the load).
    image_info img;
    std::string why;
    if (!elf.empty()) {
        if (!img.load_path(elf, &why)) {
            fprintf(stderr, "tsr-dbg: %s\n", why.c_str());
            return 2;
        }
    } else {
        tsr_image * def = nullptr;
        if (tsr_image_open_default(&def) == 0 && def) {
            img.load_path(tsr_image_path(def), &why);
            tsr_image_close(def);
        }
    }
    tsr_transport * t = tsr_transport_open(device.empty() ? nullptr : device.c_str());
    if (!t) {
        fprintf(stderr, "tsr-dbg: cannot open the device\n");
        return 1;
    }
    int status = 0;
    {
        target tg(t, &img);
        monitor mon(tg, true);
        if (mode == "load" || mode == "halt-on-boot") {
            if (!img.loaded()) {
                fprintf(stderr, "tsr-dbg: --%s needs a firmware image (--elf)\n", mode.c_str());
                tsr_transport_close(t);
                return 2;
            }
            const int e = mon.boot_async(mode == "halt-on-boot", &why);
            if (e) {
                fprintf(stderr, "tsr-dbg: %s\n", why.c_str());
                tsr_transport_close(t);
                return 1;
            }
            fprintf(stderr, "tsr-dbg: %s %s on %s%s\n", mode == "load" ? "booted" : "loaded", img.path().c_str(), tsr_transport_name(t),
                    mode == "halt-on-boot" ? ", the hart halted at its first instruction" : "");
        } else if (const tsr_image * im = tsr_transport_image(t)) {
            if (!img.loaded()) img.load(im, &why);
        }
        if (gdb && tsr_transport_csr_read(t, TSR_CSR_DBG_ID) == 0) {
            // An image built without the DB block (HAS_DB 0) has no DMI window: no GDB session can attach. Say so before listening.
            fprintf(stderr, "tsr-dbg: DBG_ID = 0: this image has no DB block (HAS_DB 0), so no DMI window and no GDB session (DB §3.2, §10.1)\n");
            status = 1;
        } else if (gdb) {
            server_options o;
            o.listen = listen.empty() ? "tcp:3333" : listen;
            o.port_file = port_file;
            o.detach_halted = detach_halted;
            o.verbose = verbose;
            rsp_server srv(tg, mon, o);
            if (srv.open_listener(&why) != 0) {
                fprintf(stderr, "tsr-dbg: %s\n", why.c_str());
                status = 1;
            } else if (srv.serve(&why) != 0) {
                fprintf(stderr, "tsr-dbg: %s\n", why.c_str());
                status = 1;
            }
        } else {
            if (cmds.empty()) cmds.push_back("status");
            for (const std::string & c : cmds) {
                std::string out;
                const int e = mon.run(c, &out);
                if (json) {
                    printf("{\"command\": %s, \"rc\": %d, \"output\": %s}\n", json_str(c).c_str(), e, json_str(out).c_str());
                } else {
                    printf("(tsr-dbg) %s\n%s", c.c_str(), out.c_str());
                }
                fflush(stdout);
                if (e) status = 1;
            }
            // Leave the target as found: a CLI that attached for rv commands detaches (resumes).
            if (tg.attached()) tg.detach(false);
        }
        std::string bw;
        if (mon.boot_running()) {
            if (tg.attached()) tg.detach(false);
        }
    }
    tsr_transport_close(t);
    return status;
}
