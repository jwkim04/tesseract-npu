// monitor.cpp — libtsr_dbg: the monitor commands (docs/design/05-debug.md "DB" §4.8; the same set in
// the tsr-dbg CLI, §9.3), the hang dump (§7.4), the recovery ladder (§7.3), the standalone server's
// launches, and the asynchronous boot (halt-on-boot, §3.8).

#include "tsr_dbg_lib.hpp"

#include "tsr_logdec.h"

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

namespace tsr::dbg {

namespace {
std::string strf(const char * f, ...) __attribute__((format(printf, 1, 2)));
std::string strf(const char * f, ...) {
    char b[2048];
    va_list ap;
    va_start(ap, f);
    vsnprintf(b, sizeof(b), f, ap);
    va_end(ap);
    return b;
}
float f32_of(uint32_t u) {
    float f;
    memcpy(&f, &u, 4);
    return f;
}
float f16_to_f32(uint16_t h) {
    const uint32_t s = (uint32_t) (h >> 15) << 31, e = (h >> 10) & 0x1F, m = h & 0x3FF;
    uint32_t u;
    if (e == 0) {
        if (m == 0) u = s;
        else {
            int k = -1;
            uint32_t mm = m;
            do { k++; mm <<= 1; } while (!(mm & 0x400));
            u = s | (uint32_t) (127 - 15 - k) << 23 | (mm & 0x3FF) << 13;
        }
    } else if (e == 31) u = s | 0x7F800000u | m << 13;
    else u = s | (e + 112) << 23 | m << 13;
    return f32_of(u);
}
int mkdir_p(const std::string & d) {
    for (size_t i = 1; i <= d.size(); ++i)
        if (i == d.size() || d[i] == '/') {
            const std::string p = d.substr(0, i);
            if (mkdir(p.c_str(), 0777) != 0 && errno != EEXIST) return -1;
        }
    return 0;
}
constexpr uint32_t REG_PC = 32;
const char * const k_abi[32] = { "zero", "ra", "sp", "gp", "tp", "t0", "t1", "t2", "s0", "s1", "a0", "a1", "a2", "a3", "a4", "a5",
                                 "a6", "a7", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "t3", "t4", "t5", "t6" };
} // namespace

std::vector<std::string> split_words(const std::string & s) {
    std::vector<std::string> w;
    size_t i = 0;
    while (i < s.size()) {
        while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) ++i;
        size_t j = i;
        while (j < s.size() && !(s[j] == ' ' || s[j] == '\t' || s[j] == '\n' || s[j] == '\r')) ++j;
        if (j > i) w.push_back(s.substr(i, j - i));
        i = j;
    }
    return w;
}

bool parse_u32(const std::string & s, uint32_t * v) {
    if (s.empty()) return false;
    char * end = nullptr;
    errno = 0;
    const unsigned long long r = strtoull(s.c_str(), &end, 0);
    if (errno || end == s.c_str() || *end != '\0' || r > 0xFFFFFFFFull || s[0] == '-') return false;
    *v = (uint32_t) r;
    return true;
}

// ---------------------------------------------------------------------------------------------
// Asynchronous boot (DB §3.8: the HELLO wait counts no time while the hart is halted)
// ---------------------------------------------------------------------------------------------
struct monitor::boot_state {
    std::thread th;
    std::atomic<bool> done{ false };
    int rc = 0;
};

monitor::~monitor() {
    if (boot_ && boot_->th.joinable()) {
        if (!boot_->done) {
            // The boot waits for HELLO; a hart left halted would keep it waiting: let it run.
            fprintf(stderr, "tsr-dbg: the boot is still waiting for HELLO; resuming the hart so that it can finish\n");
            tg_.dm().resume(5000000);
        }
        boot_->th.join();
    }
}

bool monitor::boot_running() const { return boot_ && !boot_->done; }

int monitor::boot_join(std::string * why) {
    if (!boot_) return 0;
    if (boot_->th.joinable()) boot_->th.join();
    const int rc = boot_->rc;
    if (rc && why) *why = std::string("boot failed: ") + tsr_transport_strerror(rc);
    return rc;
}

int monitor::boot_async(bool halt_on_boot, std::string * why) {
    tsr_transport * t = tg_.transport();
    image_info * img = tg_.image();
    if (!img || !img->image()) {
        if (why) *why = "no firmware image to load (--elf)";
        return E_POLICY;
    }
    if (boot_ && boot_->th.joinable()) {
        if (!boot_->done) {
            if (why) *why = "a boot is in progress";
            return E_RUNNING;
        }
        boot_->th.join();
    }
    if (halt_on_boot) {
        // DB §3.8: the request is pending when the RISC-V reset is released, so the hart halts
        // before its first instruction; DB keeps DBG_CFG across the boot's SOFT_RESET (EX §8.5).
        tg_.csr_write(TSR_CSR_DBG_CFG, tg_.csr(TSR_CSR_DBG_CFG) | TSR_DBG_CFG_DMI_EN | TSR_DBG_CFG_RV_HALT_EN | TSR_DBG_CFG_HALT_ON_BOOT);
        tg_.dm().activate();
        tg_.note_boot_halt();
        tsr_transport_set_debugger(t, 1);
    }
    boot_ = std::make_shared<boot_state>();
    std::shared_ptr<boot_state> b = boot_;
    const tsr_image * im = img->image();
    b->th = std::thread([b, t, im] {
        b->rc = tsr_transport_boot(t, im);
        b->done = true;
    });
    if (!halt_on_boot) return boot_join(why);
    // Wait for the halt at boot (or a boot that ended without one).
    const uint64_t t0 = tsr_transport_now_ns(t);
    while (!b->done) {
        uint32_t ds = 0;
        if (tg_.dm().read(TSR_DMI_DMSTATUS, &ds) == 0 && (ds & (1u << 9))) return E_OK;   // allhalted
        if (tsr_transport_now_ns(t) - t0 > 2000000000ull) break;
        tsr_transport_idle(t, 20000);
    }
    if (b->done) return boot_join(why) ? E_TRANSPORT : E_OK;
    if (why) *why = "the hart did not halt at boot within 2 s (HALT_ON_BOOT, DB §3.8)";
    return E_RUNNING;
}

// ---------------------------------------------------------------------------------------------
// Launches of the standalone server
// ---------------------------------------------------------------------------------------------
std::string monitor::poll_launches() {
    std::string o;
    if (!deferred.empty() && !boot_running()) {
        // The boot ended (HELLO or a failure): push what `launch` queued meanwhile.
        for (const tsr_launch & l : deferred) {
            uint32_t seq = 0;
            const int rc = tsr_transport_launch(tg_.transport(), &l, &seq);
            if (rc) o += strf("deferred launch: %s\n", tsr_transport_strerror(rc));
            else {
                pending.push_back(seq);
                o += strf("deferred launch pushed as seq %u\n", seq);
            }
        }
        deferred.clear();
    }
    for (size_t i = 0; i < pending.size();) {
        tsr_record rec;
        const int r = tsr_transport_poll(tg_.transport(), pending[i], &rec);
        if (r == 0) {
            ++i;
            continue;
        }
        if (r == 1) {
            tsr_result res;
            tsr_record_decode(&rec, &res);
            o += strf("launch seq %u answered: %s, RET %d (0x%08x)%s\n", pending[i], tsr_status_name(res.status), res.ret, (uint32_t) res.ret,
                      res.err_code ? strf(", error %u at ticket 0x%08x pidx %u", res.err_code, res.err_ticket, res.err_pidx).c_str() : "");
        } else {
            o += strf("launch seq %u: %s\n", pending[i], tsr_transport_strerror(r));
        }
        pending.erase(pending.begin() + (long) i);
    }
    return o;
}

int monitor::cmd_launch(const std::vector<std::string> & a, bool ping, std::string * out) {
    if (!standalone_) {
        *out += "launch / ping: only in the standalone server (the owning process launches)\n";
        return E_POLICY;
    }
    tsr_launch l;
    memset(&l, 0, sizeof(l));
    size_t k = 1;
    if (ping) {
        l.type = TSR_DESC_TYPE_PING;
        l.arg[0] = 0xDB60000Fu;
        if (a.size() > 1 && !parse_u32(a[1], &l.arg[0])) return *out += "ping [value]\n", E_MALFORMED;
    } else {
        if (a.size() < 2) return *out += "launch <kernel> [a0 .. a6] [--prof]\n", E_MALFORMED;
        l.type = TSR_DESC_TYPE_LAUNCH;
        uint32_t ki = 0;
        int idx = tg_.image() ? tg_.image()->kernel_index(a[1]) : -1;
        if (idx < 0 && parse_u32(a[1], &ki)) idx = (int) ki;
        if (idx < 0) return *out += "launch: unknown kernel " + a[1] + "\n", E_MALFORMED;
        l.kernel = (uint16_t) idx;
        uint32_t n = 0;
        for (k = 2; k < a.size(); ++k) {
            if (a[k] == "--prof") {
                l.flags |= 1u;
                continue;
            }
            if (n >= 7 || !parse_u32(a[k], &l.arg[n])) return *out += "launch: bad argument " + a[k] + "\n", E_MALFORMED;
            n++;
        }
    }
    if (boot_running()) {
        // The boot waits for HELLO (the hart is halted in crt0 or the runtime's start): the
        // descriptor goes in once the boot has finished (poll_launches, while the hart runs).
        deferred.push_back(l);
        *out += strf("%s queued: the boot waits for HELLO; it is pushed when the boot completes (continue)\n",
                     ping ? "PING" : ("launch " + a[1]).c_str());
        return E_OK;
    }
    uint32_t seq = 0;
    const int rc = tsr_transport_launch(tg_.transport(), &l, &seq);
    if (rc) return *out += strf("launch: %s\n", tsr_transport_strerror(rc)), E_TRANSPORT;
    pending.push_back(seq);
    *out += strf("%s pushed as seq %u; it runs while the hart runs (continue)\n", ping ? "PING" : ("launch " + a[1]).c_str(), seq);
    return E_OK;
}

// ---------------------------------------------------------------------------------------------
// DB §3.8 reset halt | run
// ---------------------------------------------------------------------------------------------
int monitor::cmd_reset(bool halt, std::string * out) {
    tsr_transport * t = tg_.transport();
    tg_.csr_write(TSR_CSR_DBG_CTRL, TSR_DBG_CTRL_RELEASE);   // no tile stop holds the reboot's pushes
    const uint32_t ctrl = tg_.csr(TSR_CSR_CTRL);
    const uint32_t levels = ctrl & (TSR_CTRL_HOST_HOLD | TSR_CTRL_HOLD_ON_HALT | TSR_CTRL_IRQ_EN);
    tg_.csr_write(TSR_CSR_CTRL, levels);                      // RV_RUN 1 -> 0: QABORT, ANSWERED := TAKEN (EX §8.5)
    const uint64_t t0 = tsr_transport_now_ns(t);
    while (!(tg_.csr(TSR_CSR_STATUS) & TSR_STATUS_RV_RESET_DONE)) {
        if (tsr_transport_now_ns(t) - t0 > 100000000ull) return *out += "reset: RV_RESET_DONE did not rise within 100 ms\n", E_DMI;
        tsr_transport_idle(t, 10000);
    }
    for (int i = 0; i < 1000 && !(tg_.csr(TSR_CSR_STATUS) & TSR_STATUS_QIDLE); ++i) tsr_transport_idle(t, 10000);
    if (tg_.csr(TSR_CSR_STATUS) & TSR_STATUS_QIDLE) tg_.csr_write(TSR_CSR_CTRL, levels | TSR_CTRL_QCLEAR);
    uint32_t cfg = tg_.csr(TSR_CSR_DBG_CFG);
    cfg = halt ? (cfg | TSR_DBG_CFG_HALT_ON_BOOT | TSR_DBG_CFG_RV_HALT_EN) : (cfg & ~TSR_DBG_CFG_HALT_ON_BOOT);
    tg_.csr_write(TSR_CSR_DBG_CFG, cfg);
    if (halt) tg_.note_boot_halt();
    tg_.csr_write(TSR_CSR_CTRL, levels | TSR_CTRL_RV_RUN);   // RV_RUN 0 -> 1 (the TCM image stays)
    tg_.halted_ = false;
    tg_.stepping_ = false;
    tg_.skip_ebreak_ = false;
    if (!halt) {
        *out += "reset run: the RISC-V restarts from RV_BOOT + 0x80 with the image in its TCMs\n";
        return E_OK;
    }
    const uint64_t t1 = tsr_transport_now_ns(t);
    while (!tg_.poll_halted()) {
        if (tsr_transport_now_ns(t) - t1 > 100000000ull) return *out += "reset halt: the hart did not halt at boot\n", E_RUNNING;
        tsr_transport_idle(t, 10000);
    }
    stop_info si;
    tg_.stop(&si);
    *out += strf("reset halt: %s, pc 0x%08x\n", si.note.empty() ? "halted" : si.note.c_str(), si.pc);
    return E_OK;
}

// ---------------------------------------------------------------------------------------------
// DB §7.4 hang dump
// ---------------------------------------------------------------------------------------------
int monitor::cmd_dump(const std::string & dir, std::string * out) {
    tsr_transport * t = tg_.transport();
    if (mkdir_p(dir) != 0) return *out += "dump: cannot create " + dir + "\n", E_POLICY;
    auto put = [&](const std::string & name, const std::string & text) {
        if (FILE * f = fopen((dir + "/" + name).c_str(), "w")) {
            fputs(text.c_str(), f);
            fclose(f);
        }
    };
    // state.json: every EX, Q_*, PERF and DBG register raw (the DMI window excluded: reads have side effects).
    std::string j = "{\n  \"kind\": \"" + std::string(tsr_transport_name(t)) + "\",\n";
    j += strf("  \"image\": \"%s\",\n  \"image_id\": \"0x%016" PRIx64 "\",\n  \"csr\": {", tg_.image() ? tg_.image()->path().c_str() : "",
              tg_.image() ? tg_.image()->image_id() : 0);
    bool first = true;
    auto one = [&](uint32_t off) {
        j += strf("%s\n    \"0x%03x\": \"0x%08x\"", first ? "" : ",", off, tg_.csr(off));
        first = false;
    };
    for (uint32_t off = 0; off < 0x100; off += 4)
        if (off != TSR_CSR_LQ_PUSH && off != TSR_CSR_LOAD_CTRL && off != TSR_CSR_LOAD_DATA) one(off);
    for (uint32_t off = 0x300; off < 0x400; off += 4) one(off);
    for (uint32_t off = 0x800; off < 0xB00; off += 4)
        if (off != TSR_CSR_DBG_CTRL) one(off);
    for (uint32_t off = 0xE00; off < 0xF20; off += 4) one(off);
    j += "\n  },\n  \"perf\": [";
    for (uint32_t i = 0; i < TSR_PERF_SLOTS; ++i) {
        const uint32_t lo = tg_.csr(TSR_CSR_PERF_LO(i)), hi = tg_.csr(TSR_CSR_PERF_HI(i));
        j += strf("%s%" PRIu64, i ? ", " : "", (uint64_t) hi << 32 | lo);
    }
    j += "]\n}\n";
    put("state.json", j);
    put("status.txt", status_text(t, tg_.image()));
    put("units.txt", units_text(t));
    put("perf.txt", perf_text(t));
    std::string h;
    for (const hist_entry & e : hist_last(t, 512)) h += hist_line(t, e, tg_.image()) + "\n";
    put("hist.txt", h);
    // log.txt: the words drained now, decoded with the image.
    std::vector<uint32_t> w(64 * 1024);
    size_t n = 0;
    tsr_transport_log_drain(t, w.data(), w.size(), &n);
    std::string lg;
    log_decoder dec(tg_.image() ? tg_.image()->image() : nullptr);
    dec.feed(w.data(), n, [&](const std::string & l) { lg += l + "\n"; });
    put("log.txt", lg);
    // Five RV samples 1 ms apart (DB §6.2).
    std::string rs;
    for (int i = 0; i < 5; ++i) {
        std::string s;
        tg_.rv_sample(&s);
        rs += s + "\n";
        tsr_transport_idle(t, 1000000);
    }
    put("rv_samples.txt", rs);
    // __tsr_dbg, __tsr_crash and DTCM through SBA, if clk_rv answers.
    if (tg_.csr(TSR_CSR_STATUS) & TSR_STATUS_RV_CLK_OK) {
        if (!(tg_.csr(TSR_CSR_DBG_CFG) & TSR_DBG_CFG_DMI_EN)) tsr_transport_dmi_enable(t, 1);
        tg_.dm().activate();
        std::vector<uint8_t> d(tg_.dtcm_bytes());
        if (tg_.read_mem(TSR_RV_DTCM_BASE, d.data(), d.size()) == 0) {
            if (FILE * f = fopen((dir + "/dtcm.bin").c_str(), "wb")) {
                fwrite(d.data(), 1, d.size(), f);
                fclose(f);
            }
        }
        std::string c;
        cmd_mem({ "x/16x", "__tsr_dbg" }, false, &c);
        std::string cr;
        run("crash", &cr);
        put("tsr_dbg.txt", c);
        put("tsr_crash.txt", cr);
    }
    if (trw_on) {
        std::vector<trw_rec> recs;
        uint64_t lost = 0;
        std::string why;
        if (trw_drain(t, trw, &recs, &lost, &why) == 0) trw_save_raw(recs, dir + "/trw.bin");
    }
    *out += "hang dump written to " + dir + " (DB §7.4)\n";
    return E_OK;
}

// ---------------------------------------------------------------------------------------------
// DB §7.3 recovery ladder
// ---------------------------------------------------------------------------------------------
int monitor::cmd_recover(int level, std::string * out) {
    tsr_transport * t = tg_.transport();
    const char * dd = getenv("TSR_DUMP_DIR");
    const std::string dir = std::string(dd && dd[0] ? dd : ".") + strf("/tsr-dbg-dump-%ld", (long) getpid());
    cmd_dump(dir, out);   // level 0, always first
    if (level >= 1) {
        tg_.csr_write(TSR_CSR_DBG_CTRL, TSR_DBG_CTRL_RELEASE);
        tg_.csr_write(TSR_CSR_DBG_CFG, tg_.csr(TSR_CSR_DBG_CFG) & ~TSR_DBG_CFG_SERIAL);
        const uint32_t ctrl = tg_.csr(TSR_CSR_CTRL);
        tg_.csr_write(TSR_CSR_CTRL, ctrl & (TSR_CTRL_RV_RUN | TSR_CTRL_HOST_HOLD | TSR_CTRL_IRQ_EN));
        for (uint32_t i = 0; i < TSR_DBG_NTBP; ++i) tg_.csr_write(TSR_CSR_DBG_TBP_CTL(i), 0);
        tg_.session_tbps_ = 0;
        if (tg_.attached() && tg_.halted()) tg_.resume();
        *out += "recovery level 1: tile stops released, SERIAL and HOLD_ON_HALT off, tile breakpoints cleared, the hart resumed\n";
    }
    if (level >= 2) {
        const uint32_t pushed = tg_.csr(TSR_CSR_LQ_SEQ);
        tsr_transport_cancel(t, pushed - 1);
        const uint64_t t0 = tsr_transport_now_ns(t);
        while (tg_.csr(TSR_CSR_LCQ_ANSWERED) != pushed && tsr_transport_now_ns(t) - t0 < 10000000000ull) tsr_transport_idle(t, 100000);
        *out += strf("recovery level 2: LQ_CANCEL = %u; answered %u of %u\n", pushed - 1, tg_.csr(TSR_CSR_LCQ_ANSWERED), pushed);
        *out += poll_launches();
    }
    if (level >= 3) {
        int rc = tsr_transport_soft_reset(t);
        if (rc == 0 && tg_.image() && tg_.image()->image()) rc = tsr_transport_boot(t, tg_.image()->image());
        *out += strf("recovery level 3: SOFT_RESET + warm restart: %s\n", tsr_transport_strerror(rc));
        tg_.halted_ = false;
        if (rc) return E_TRANSPORT;
    }
    if (level >= 4) *out += "recovery level 4: a card reset (xrt-smi reset) destroys everything but the level-0 dump; run it by hand\n";
    return E_OK;
}

// ---------------------------------------------------------------------------------------------
// mem / spm x/<n><f> <addr | line>
// ---------------------------------------------------------------------------------------------
int monitor::cmd_mem(const std::vector<std::string> & a, bool spm, std::string * out) {
    if (a.size() < 2 || a[0].size() < 2 || a[0][0] != 'x' || a[0][1] != '/')
        return *out += (spm ? "spm x/<n><f> <line>" : "mem x/<n><f> <addr|symbol>") + std::string(" (f: x hex words, f f32, h f16, b bytes, i i8)\n"), E_MALFORMED;
    const std::string spec = a[0].substr(2);
    size_t p = 0;
    while (p < spec.size() && isdigit((unsigned char) spec[p])) ++p;
    const uint32_t n = p ? (uint32_t) atoi(spec.substr(0, p).c_str()) : 1;
    const char f = p < spec.size() ? spec[p] : 'x';
    uint32_t addr = 0;
    if (!parse_u32(a[1], &addr)) {
        if (!(tg_.image() && tg_.image()->symbol(a[1], &addr))) return *out += "unknown address or symbol " + a[1] + "\n", E_MALFORMED;
    }
    if (spm) addr = TSR_RV_SPM_BASE + 64u * addr;
    const uint32_t esz = (f == 'b' || f == 'i') ? 1 : (f == 'h') ? 2 : 4;
    if (n == 0 || n > 65536) return *out += "count 1..65536\n", E_MALFORMED;
    std::vector<uint8_t> b((size_t) n * esz);
    const int e = tg_.read_mem(addr, b.data(), b.size());
    if (e) return *out += err_str(e) + "\n", e;
    const uint32_t per = esz == 1 ? 16 : esz == 2 ? 8 : 4;
    for (uint32_t i = 0; i < n; ++i) {
        if (i % per == 0) *out += strf("%s0x%08x:", i ? "\n" : "", addr + i * esz);
        const uint8_t * q = &b[(size_t) i * esz];
        const uint32_t v = esz == 1 ? q[0] : esz == 2 ? (uint32_t) (q[0] | q[1] << 8) : (uint32_t) (q[0] | q[1] << 8 | q[2] << 16 | (uint32_t) q[3] << 24);
        if (f == 'f') *out += strf(" %.9g", (double) f32_of(v));
        else if (f == 'h') *out += strf(" %.5g", (double) f16_to_f32((uint16_t) v));
        else if (f == 'i') *out += strf(" %4d", (int) (int8_t) v);
        else if (esz == 1) *out += strf(" %02x", v);
        else *out += strf(" 0x%08x", v);
    }
    *out += "\n";
    return E_OK;
}

// ---------------------------------------------------------------------------------------------
// tile (DB §5, §9.3)
// ---------------------------------------------------------------------------------------------
int monitor::cmd_tile(const std::vector<std::string> & a, std::string * out) {
    tsr_transport * t = tg_.transport();
    const std::string sub = a.size() > 1 ? a[1] : "show";
    std::vector<std::string> rest(a.begin() + std::min<size_t>(2, a.size()), a.end());
    if (sub == "show") {
        *out += tile_show(t);
    } else if (sub == "stop") {
        if (rest.empty() || rest[0] == "now") {
            tg_.csr_write(TSR_CSR_DBG_CTRL, TSR_DBG_CTRL_STOP_NOW);
        } else if (rest[0] == "at" && rest.size() > 1) {
            uint32_t p = 0;
            if (!parse_u32(rest[1], &p)) return *out += "tile stop at P\n", E_MALFORMED;
            tg_.csr_write(TSR_CSR_DBG_STOP_AT, p);
        } else {
            return *out += "tile stop [now | at P]\n", E_MALFORMED;
        }
        *out += tile_show(t);
    } else if (sub == "break") {
        std::string why;
        const int i = tile_break(t, rest, tg_.image(), &why);
        if (i < 0) return *out += why + "\n", E_POLICY;
        tg_.session_tbps_ |= 1u << i;
        *out += strf("tile breakpoint %d set\n", i);
    } else if (sub == "delete") {
        uint32_t i = 0;
        if (rest.empty() || !parse_u32(rest[0], &i) || i >= TSR_DBG_NTBP) return *out += "tile delete N (0..3)\n", E_MALFORMED;
        tg_.csr_write(TSR_CSR_DBG_TBP_CTL(i), 0);
        tg_.session_tbps_ &= ~(1u << i);
        *out += strf("tile breakpoint %u deleted\n", i);
    } else if (sub == "step") {
        uint32_t n = 1;
        if (!rest.empty() && !parse_u32(rest[0], &n)) return *out += "tile step [n]\n", E_MALFORMED;
        const int e = tile_step(t, n, 1000000000ull, out);
        if (e) return e;
    } else if (sub == "cont" || sub == "release") {
        tg_.csr_write(TSR_CSR_DBG_CTRL, TSR_DBG_CTRL_RELEASE);
        *out += tile_show(t);
    } else if (sub == "serial") {
        if (rest.empty() || (rest[0] != "on" && rest[0] != "off")) return *out += "tile serial on|off\n", E_MALFORMED;
        const uint32_t c = tg_.csr(TSR_CSR_DBG_CFG);
        tg_.csr_write(TSR_CSR_DBG_CFG, rest[0] == "on" ? (c | TSR_DBG_CFG_SERIAL) : (c & ~TSR_DBG_CFG_SERIAL));
        if (rest[0] == "on") tsr_transport_set_debugger(t, 1);   // the waits see serial mode (no timeout)
        *out += strf("serial mode %s (DB §5.4)\n", rest[0].c_str());
    } else {
        return *out += "tile show | stop [now|at P] | break ... | delete N | step [n] | cont | serial on|off\n", E_MALFORMED;
    }
    return E_OK;
}

// ---------------------------------------------------------------------------------------------
// rv (DB §9.3: RISC-V control without GDB)
// ---------------------------------------------------------------------------------------------
int monitor::cmd_rv(const std::vector<std::string> & a, std::string * out) {
    tsr_transport * t = tg_.transport();
    const std::string sub = a.size() > 1 ? a[1] : "regs";
    if (sub == "sample") {
        uint32_t n = 1;
        if (a.size() > 3 && a[2] == "-n" && !parse_u32(a[3], &n)) return *out += "rv sample [-n N]\n", E_MALFORMED;
        for (uint32_t i = 0; i < n; ++i) {
            std::string s;
            tg_.rv_sample(&s);
            *out += s + "\n";
            if (i + 1 < n) tsr_transport_idle(t, 1000000);
        }
        return E_OK;
    }
    if (sub == "reset") return cmd_reset(a.size() > 2 && a[2] == "halt", out);
    if (sub == "load") {
        if (a.size() > 2) {
            std::string why;
            if (!tg_.image()->load_path(a[2], &why)) return *out += why + "\n", E_POLICY;
        }
        return run("load", out);
    }
    if (!tg_.attached()) {
        std::string why;
        const int e = tg_.attach(&why);
        if (e) return *out += "attach: " + why + "\n", e;
        for (const std::string & n : tg_.attach_notes()) *out += "note: " + n + "\n";
    }
    if (sub == "halt") {
        if (!tg_.halted()) {
            tg_.interrupt();
            const uint64_t t0 = tsr_transport_now_ns(t);
            while (!tg_.poll_halted()) {
                if (tsr_transport_now_ns(t) - t0 > 5000000) return *out += "rv halt: no halt within 5 ms\n", E_RUNNING;
                tsr_transport_idle(t, 1000);
            }
        }
        stop_info si;
        tg_.stop(&si);
        *out += strf("halted at %s (dcsr.cause %u)%s%s\n", tg_.image() ? tg_.image()->symbolize(si.pc).c_str() : "?", si.dcsr_cause,
                     si.note.empty() ? "" : ": ", si.note.c_str());
    } else if (sub == "resume" || sub == "cont") {
        const int e = tg_.resume();
        if (e) return *out += err_str(e) + "\n", e;
        *out += "resumed\n";
    } else if (sub == "step") {
        uint32_t n = 1;
        if (a.size() > 2 && !parse_u32(a[2], &n)) return *out += "rv step [n]\n", E_MALFORMED;
        for (uint32_t i = 0; i < n; ++i) {
            stop_info si;
            const int e = tg_.step(&si);
            if (e) return *out += err_str(e) + "\n", e;
            if (i + 1 == n || !si.reason.empty())
                *out += strf("stepped to %s%s%s\n", tg_.image() ? tg_.image()->symbolize(si.pc).c_str() : "?", si.reason.empty() ? "" : " ",
                             si.reason.c_str());
        }
    } else if (sub == "regs") {
        uint32_t r[33];
        const int e = tg_.read_gprs(r);
        if (e) return *out += err_str(e) + "\n", e;
        for (uint32_t i = 0; i < 32; ++i) *out += strf("x%-2u %-4s 0x%08x%s", i, k_abi[i], r[i], (i % 4 == 3) ? "\n" : "  ");
        *out += strf("pc 0x%08x %s\n", r[REG_PC], tg_.image() ? tg_.image()->symbolize(r[REG_PC]).c_str() : "");
        static const struct { const char * n; uint32_t c; } cs[] = { { "mstatus", 0x300 }, { "mepc", 0x341 }, { "mcause", 0x342 },
                                                                    { "mtval", 0x343 }, { "mie", 0x304 }, { "mip", 0x344 } };
        for (const auto & c : cs) {
            uint32_t v = 0;
            if (tg_.read_reg(65 + c.c, &v) == 0) *out += strf("%s 0x%08x  ", c.n, v);
        }
        *out += "\n";
    } else {
        return *out += "rv halt | resume | step [n] | regs | sample [-n N] | reset [halt] | load F\n", E_MALFORMED;
    }
    return E_OK;
}

// ---------------------------------------------------------------------------------------------
// trw (DB §8.3)
// ---------------------------------------------------------------------------------------------
int monitor::cmd_trw(const std::vector<std::string> & a, std::string * out) {
    tsr_transport * t = tg_.transport();
    const std::string sub = a.size() > 1 ? a[1] : "status";
    if (sub == "start") {
        uint32_t mib = TSR_TRW_RING_DEFAULT_MIB;
        if (a.size() > 2 && !parse_u32(a[2], &mib)) return *out += "trw start [MiB]\n", E_MALFORMED;
        trw = trw_default_ring(mib);
        std::string why;
        const int e = trw_start(t, trw, &why);
        if (e) return *out += why + "\n", e;
        trw_on = true;
        *out += strf("trace writer on: ring ch%u offset 0x%" PRIx64 ", %u lines (%u records), overwrite\n", trw.ch, trw.off, trw.lines, 2 * trw.lines);
    } else if (sub == "stop") {
        trw_stop(t);
        trw_on = false;
        *out += "trace writer off\n";
    } else if (sub == "status") {
        *out += strf("TRW %s: WPTR %u RPTR %u drops %u, PERF TRW_RECORDS %u\n", trw_on ? "on" : "off", tg_.csr(TSR_CSR_DBG_TRW_WPTR),
                     tg_.csr(TSR_CSR_DBG_TRW_RPTR), tg_.csr(TSR_CSR_DBG_TRW_DROPS), tg_.csr(TSR_CSR_PERF_LO(TSR_PERF_TRW_RECORDS)));
    } else if (sub == "dump") {
        if (!trw_on) {
            // A ring written by an earlier process (the DBG configuration, the ring registers and the ring's DRAM survive
            // SOFT_RESET and the end of that process): rebuild the ring from DBG_TRW_BASE / DBG_TRW_LINES.
            const uint32_t base = tg_.csr(TSR_CSR_DBG_TRW_BASE), lines = tg_.csr(TSR_CSR_DBG_TRW_LINES);
            if (lines == 0) return *out += "trw dump: no ring is configured (trw start)\n", E_POLICY;
            trw.ch = TSR_GET(base, TSR_HANDLE_CH);
            trw.off = (uint64_t) TSR_GET(base, TSR_HANDLE_BEAT) * 64;
            trw.lines = lines;
        }
        if (a.size() < 3) return *out += "trw dump <file.json> [raw.bin]\n", E_MALFORMED;
        std::vector<trw_rec> recs;
        uint64_t lost = 0;
        std::string why;
        const int e = trw_drain(t, trw, &recs, &lost, &why);
        if (e) return *out += why + "\n", e;
        timeline_opts o;
        o.img = tg_.image();
        o.num_ch = tsr_transport_config(t)->num_ch;
        const std::string js = trw_chrome_json(recs, o);
        FILE * f = fopen(a[2].c_str(), "w");
        if (!f) return *out += "cannot write " + a[2] + "\n", E_POLICY;
        fputs(js.c_str(), f);
        fclose(f);
        if (a.size() > 3) trw_save_raw(recs, a[3]);
        uint32_t ops = 0;
        for (const trw_rec & r : recs) ops += r.kind == TSR_TRW_W0_KIND_OP;
        *out += strf("%zu records (%u ops), %" PRIu64 " overwritten before the drain; timeline written to %s\n", recs.size(), ops, lost, a[2].c_str());
    } else {
        return *out += "trw start [MiB] | stop | status | dump <file.json> [raw.bin]\n", E_MALFORMED;
    }
    return E_OK;
}

// ---------------------------------------------------------------------------------------------
// The command line
// ---------------------------------------------------------------------------------------------
int monitor::run(const std::string & line, std::string * out) {
    const std::vector<std::string> a = split_words(line);
    if (a.empty()) return E_OK;
    tsr_transport * t = tg_.transport();
    const std::string & c = a[0];
    if (c == "help") {
        *out += "monitor commands (DB §4.8, §9.3):\n"
                "  status | units | qstatus | perf | sreg | hist [n] [--pidx P] [--export F] | log [level]\n"
                "  crash | crash-view on|off | hold on|off | launch-break <kernel>|all|off | spm-write safe|unsafe\n"
                "  tile show | stop [now|at P] | break <pidx P[-Q]|op NAME|spm A B|dram CH A B|ticket T> [r|w|rw] [after] [halt] [count]\n"
                "       | delete N | step [n] | cont | serial on|off\n"
                "  spm x/<n><f> <line> | mem x/<n><f> <addr|symbol>   (f: x f h b i)\n"
                "  trw start [MiB] | stop | status | dump <file.json> [raw.bin]\n"
                "  reset halt|run | load | dump <dir> | recover [level]\n"
                "  rv halt|resume|step [n]|regs|sample [-n N]|reset [halt]|load F\n"
                "  launch <kernel> [a0..a6] [--prof] | ping [value]   (standalone server)\n";
        return E_OK;
    }
    if (c == "status") {
        *out += status_text(t, tg_.image());
        *out += strf("debugger: %s%s%s\n", tg_.attached() ? "attached" : "detached", tg_.attached() ? (tg_.halted() ? ", hart halted" : ", hart running") : "",
                     tg_.degraded() ? ", DEGRADED (DMI timeouts: core-domain reads only)" : "");
        return E_OK;
    }
    if (c == "units" || c == "qstatus") return *out += units_text(t), E_OK;
    if (c == "perf") return *out += perf_text(t), E_OK;
    if (c == "sreg") return *out += sreg_text(t), E_OK;
    if (c == "tile") return cmd_tile(a, out);
    if (c == "rv") return cmd_rv(a, out);
    if (c == "trw") return cmd_trw(a, out);
    if (c == "launch") return cmd_launch(a, false, out);
    if (c == "ping") return cmd_launch(a, true, out);
    if (c == "mem") return cmd_mem(std::vector<std::string>(a.begin() + 1, a.end()), false, out);
    if (c == "spm") return cmd_mem(std::vector<std::string>(a.begin() + 1, a.end()), true, out);
    if (c == "hist") {
        uint32_t n = 16, pidx = 0;
        bool one = false;
        std::string exp;
        for (size_t i = 1; i < a.size(); ++i) {
            if (a[i] == "--pidx" && i + 1 < a.size() && parse_u32(a[i + 1], &pidx)) one = true, ++i;
            else if (a[i] == "--export" && i + 1 < a.size()) exp = a[++i];
            else if ((a[i] == "-n" && i + 1 < a.size() && parse_u32(a[++i], &n)) || parse_u32(a[i], &n)) {}
            else return *out += "hist [n] [--pidx P] [--export F]\n", E_MALFORMED;
        }
        std::vector<hist_entry> es;
        if (one) {
            hist_entry e;
            hist_read(t, pidx, &e);
            es.push_back(e);
        } else {
            es = hist_last(t, n);
        }
        *out += strf("history head %u (next push index to record), %zu entries:\n", tg_.csr(TSR_CSR_DBG_HIST_HEAD), es.size());
        for (const hist_entry & e : es) *out += hist_line(t, e, tg_.image()) + "\n";
        if (!exp.empty()) {
            std::string why;
            if (!hist_export(es, exp, &why)) return *out += why + "\n", E_POLICY;
            *out += "exported as push.bin to " + exp + ": " + why + "\n";
        }
        return E_OK;
    }
    if (c == "log") {
        if (a.size() > 1) {
            uint32_t lv = 0;
            if (!parse_u32(a[1], &lv) || lv > 5) return *out += "log [level 0..5]\n", E_MALFORMED;
            const uint32_t cfg = tg_.csr(TSR_CSR_LOG_CFG);
            tg_.csr_write(TSR_CSR_LOG_CFG, (cfg & ~TSR_LOG_CFG_LEVEL) | TSR_PUT(TSR_LOG_CFG_LEVEL, lv));
            *out += strf("LOG_CFG.LEVEL = %u\n", lv);
        }
        std::vector<uint32_t> w(64 * 1024);
        size_t n = 0;
        tsr_transport_log_drain(t, w.data(), w.size(), &n);
        log_decoder dec(tg_.image() ? tg_.image()->image() : nullptr);
        dec.feed(w.data(), n, [&](const std::string & l) { *out += l + "\n"; });
        if (n == 0) *out += "(no log words)\n";
        return E_OK;
    }
    if (c == "crash") {
        uint32_t base = 0;
        if (!tg_.sym("__tsr_crash", &base)) return *out += "no __tsr_crash symbol\n", E_POLICY;
        uint32_t w[TSR_CRASH_NWORDS];
        const int e = tg_.dm().sba_read32(base, w, TSR_CRASH_NWORDS);
        if (e) return *out += "__tsr_crash: " + err_str(e) + "\n", e;
        if (w[0] != TSR_CRASH_MAGIC) return *out += strf("__tsr_crash holds no crash (magic 0x%08x)\n", w[0]), E_OK;
        const std::string kn = tg_.image() ? tg_.image()->kernel_name(w[TSR_CRASH_KERNEL_WORD]) : std::string();
        *out += strf("crash: %s (mcause %u) at %s, mtval 0x%08x, mstatus 0x%08x, MTIME %" PRIu64 ", launch seq %u kernel %u%s%s, RT_STATE 0x%08x "
                     "RT_INFO 0x%08x, nested %u\n",
                     tsr_mcause_name(w[TSR_CRASH_MCAUSE_WORD]), w[TSR_CRASH_MCAUSE_WORD],
                     tg_.image() ? tg_.image()->symbolize(w[TSR_CRASH_MEPC_WORD]).c_str() : "?", w[TSR_CRASH_MTVAL_WORD],
                     w[TSR_CRASH_MSTATUS_WORD], (uint64_t) w[TSR_CRASH_MTIME_HI_WORD] << 32 | w[TSR_CRASH_MTIME_LO_WORD], w[TSR_CRASH_SEQ_WORD],
                     w[TSR_CRASH_KERNEL_WORD], kn.empty() ? "" : " ", kn.c_str(), w[TSR_CRASH_RT_STATE_WORD], w[TSR_CRASH_RT_INFO_WORD],
                     w[TSR_CRASH_NESTED_WORD]);
        for (uint32_t i = 1; i < 32; ++i)
            *out += strf("x%-2u %-4s 0x%08x%s", i, k_abi[i], w[TSR_CRASH_X1_WORD + i - 1], (i % 4 == 3) ? "\n" : "  ");
        *out += "\n";
        return E_OK;
    }
    if (c == "crash-view") {
        if (a.size() < 2 || (a[1] != "on" && a[1] != "off")) return *out += "crash-view on|off\n", E_MALFORMED;
        tg_.crash_view = a[1] == "on";
        *out += strf("crash view %s: g / p return %s (flushregs in GDB)\n", a[1].c_str(), tg_.crash_view ? "the saved trap registers" : "the live registers");
        return E_OK;
    }
    if (c == "hold") {
        if (a.size() < 2 || (a[1] != "on" && a[1] != "off")) return *out += "hold on|off\n", E_MALFORMED;
        tg_.opts().hold_on_halt = a[1] == "on";
        const uint32_t ctrl = tg_.csr(TSR_CSR_CTRL) & (TSR_CTRL_RV_RUN | TSR_CTRL_HOST_HOLD | TSR_CTRL_HOLD_ON_HALT | TSR_CTRL_IRQ_EN);
        tg_.csr_write(TSR_CSR_CTRL, a[1] == "on" ? (ctrl | TSR_CTRL_HOLD_ON_HALT) : (ctrl & ~TSR_CTRL_HOLD_ON_HALT));
        *out += strf("CTRL.HOLD_ON_HALT %s\n", a[1].c_str());
        return E_OK;
    }
    if (c == "launch-break") {
        if (a.size() < 2) return *out += "launch-break <kernel>|all|off\n", E_MALFORMED;
        uint32_t v = 0;
        if (a[1] == "all") v = 0xFFFFFFFFu;
        else if (a[1] == "off") v = 0;
        else {
            int k = tg_.image() ? tg_.image()->kernel_index(a[1]) : -1;
            uint32_t ki = 0;
            if (k < 0 && parse_u32(a[1], &ki)) k = (int) ki;
            if (k < 0) return *out += "launch-break: unknown kernel " + a[1] + "\n", E_MALFORMED;
            v = (uint32_t) k + 1;
        }
        if (v && !tg_.attached()) {
            // The launch halt is an ebreak: it needs the session's dcsr.ebreakm (attach), else it
            // traps. Attach, and leave the hart running as it was.
            std::string why;
            const bool was_running = !(tg_.csr(TSR_CSR_STATUS) & TSR_STATUS_RV_HALTED);
            const int ae = tg_.attach(&why);
            if (ae) return *out += "launch-break: attach: " + why + "\n", ae;
            if (was_running) tg_.resume();
        }
        int e = tg_.set_dbg_word(TSR_DBG_INFO_LAUNCH_BREAK_WORD, v);
        if (!e && v) e = tg_.set_dbg_word(TSR_DBG_INFO_ATTACHED_WORD, 1);
        if (e) return *out += err_str(e) + "\n", e;
        *out += strf("__tsr_dbg.launch_break = 0x%08x (DB §3.8)\n", v);
        return E_OK;
    }
    if (c == "spm-write") {
        if (a.size() < 2 || (a[1] != "safe" && a[1] != "unsafe")) return *out += "spm-write safe|unsafe\n", E_MALFORMED;
        tg_.opts().spm_write_unsafe = a[1] == "unsafe";
        *out += "SPM writes " + std::string(tg_.opts().spm_write_unsafe ? "allowed any time (racy)" : "only in a stopped tile state") + "\n";
        return E_OK;
    }
    if (c == "reset") {
        if (a.size() < 2 || (a[1] != "halt" && a[1] != "run")) return *out += "reset halt|run\n", E_MALFORMED;
        return cmd_reset(a[1] == "halt", out);
    }
    if (c == "load") {
        if (!standalone_) return *out += "load: only in the standalone server\n", E_POLICY;
        // SOFT_RESET + boot (EX's LOAD_* with VERIFY), halting at boot so that the session stays halted.
        std::string why;
        const int e = boot_async(tg_.attached(), &why);
        if (e) return *out += why + "\n", e;
        if (tg_.attached()) {
            tg_.halted_ = false;
            while (!tg_.poll_halted()) tsr_transport_idle(t, 10000);
            stop_info si;
            tg_.stop(&si);
            *out += strf("loaded %s; %s at 0x%08x\n", tg_.image()->path().c_str(), si.note.empty() ? "halted" : si.note.c_str(), si.pc);
        } else {
            *out += "loaded " + tg_.image()->path() + "\n";
        }
        return E_OK;
    }
    if (c == "dump") {
        if (a.size() < 2) return *out += "dump <dir>\n", E_MALFORMED;
        return cmd_dump(a[1], out);
    }
    if (c == "recover") {
        uint32_t lv = 1;
        for (size_t i = 1; i < a.size(); ++i)
            if (!(a[i] == "--level" || parse_u32(a[i], &lv))) return *out += "recover [--level] n\n", E_MALFORMED;
        return cmd_recover((int) lv, out);
    }
    *out += "unknown monitor command \"" + c + "\" (monitor help)\n";
    return E_MALFORMED;
}

} // namespace tsr::dbg
