// tsr_xrt_selftest.cpp — card bring-up checklist for the Tesseract kernel (docs/xrt-host.md §5).
//
//   tsr_xrt_selftest [--device KIND] [--mem-mib N] [--no-mem] [--seed N] [-v]
//
//   --device KIND  transport (default: $TSR_DEVICE, else xrt). "mock" runs the same checks on the
//                  adapter register model (ctest); "sim" checks the tool itself.
//   --mem-mib N    size of the per-channel bandwidth transfer (default 64; 0 skips it)
//
// 1. Identity: transport / kdev description, ID, VERSION (ISA 2), CFG0..2, CAPS, ADAPTER_VERSION,
//    ADAPTER_STATUS as found at open (the protocol layer clears it there), the channel buffers.
// 2. Memory, per channel: pattern write / read-back at aligned, unaligned, 4 KiB- and
//    4 GiB-crossing and end-of-channel ranges, with guard bytes that must stay untouched;
//    channel aliasing (same offset, different data on every channel); write / read bandwidth.
// The last check requires a clean ADAPTER_STATUS at the end.
// v1's host-written programs compared with the ISA simulator and the cmd.bin trace replays are
// deleted (X0, D15: the host has no access to the issue port); the device checks return as the
// `selftest` kernel and --traces as replays through the `replay` kernel in tsr-selftest (P3.3,
// docs/v2/06-software-verification.md §9.5).
// Exit code: 0 every check passed, 1 a check failed, 2 the transport did not open / bad usage.

#include "tsr_isa.h"
#include "tsr_kdev.h"
#include "tsr_transport.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

int  g_fail = 0, g_pass = 0;
bool g_verbose = false;

void result(bool ok, const char * what, const std::string & detail = std::string()) {
    (ok ? g_pass : g_fail)++;
    if (!ok || g_verbose) printf("  %s  %s%s%s\n", ok ? "PASS" : "FAIL", what, detail.empty() ? "" : ": ", detail.c_str());
}

std::string strf(const char * f, ...) __attribute__((format(printf, 1, 2)));
std::string strf(const char * f, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, f);
    vsnprintf(buf, sizeof(buf), f, ap);
    va_end(ap);
    return buf;
}

struct rng {
    uint64_t s;
    uint64_t next() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return s;
    }
    void fill(void * p, size_t n) {
        uint8_t * b = (uint8_t *) p;
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            const uint64_t v = next();
            memcpy(b + i, &v, 8);
        }
        for (; i < n; ++i) b[i] = (uint8_t) next();
    }
};

// First differing byte, or -1.
int64_t first_diff(const uint8_t * a, const uint8_t * b, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (a[i] != b[i]) return (int64_t) i;
    }
    return -1;
}

double secs(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// ---------------------------------------------------------------------------
// 1. Identity
// ---------------------------------------------------------------------------
void identity(tsr_transport * t) {
    const tsr_config * c = tsr_transport_config(t);
    printf("== identity (%s)\n", tsr_transport_name(t));
    tsr::kdev * kd = tsr::transport_kdev(t);
    if (kd != nullptr) {
        std::string d = kd->describe();
        size_t p = 0;
        while (p < d.size()) {
            const size_t e = d.find('\n', p);
            printf("  %s\n", d.substr(p, e == std::string::npos ? std::string::npos : e - p).c_str());
            if (e == std::string::npos) break;
            p = e + 1;
        }
    }
    const uint32_t id = tsr_transport_csr_read(t, TSR_CSR_ID), ver = tsr_transport_csr_read(t, TSR_CSR_VERSION);
    const uint32_t cfg0 = tsr_transport_csr_read(t, TSR_CSR_CFG0), cfg1 = tsr_transport_csr_read(t, TSR_CSR_CFG1);
    const uint32_t cfg2 = tsr_transport_csr_read(t, TSR_CSR_CFG2), caps = tsr_transport_csr_read(t, TSR_CSR_CAPS);
    printf("  ID 0x%08x  VERSION 0x%08x (ISA %u.%u)  CFG0 0x%08x  CFG1 0x%08x  CFG2 0x%08x  CAPS 0x%08x\n", id, ver, (ver >> 8) & 0xFFu,
           ver & 0xFFu, cfg0, cfg1, cfg2, caps);
    printf("  NUM_CH %u  LANES %u  T_MAX %u  K_MAX %u  SPM %u KiB  VL %u  NSREG %u  CAPS 0x%x  ch_bytes %.3f GiB\n", c->num_ch, c->lanes,
           c->t_max, c->k_max, c->spm_bytes / 1024, c->vl, c->nsreg, c->caps, (double) c->ch_bytes / (1ull << 30));
    result(id == TSR_ID_MAGIC, "ID is \"TSR1\"", strf("0x%08x", id));
    result((ver >> 8) == TSR_ISA_MAJOR, "ISA major", strf("%u (expected %u)", ver >> 8, TSR_ISA_MAJOR));
    if (kd != nullptr) {
        uint32_t av = 0, as = 0;
        const bool ok = kd->reg_read(tsr::KREG_ADAPTER_VERSION, &av) == 0 && kd->reg_read(tsr::KREG_ADAPTER_STATUS, &as) == 0;
        const uint32_t at_open = tsr::transport_adapter_log(t).at_open;
        printf("  ADAPTER_VERSION 0x%08x  ADAPTER_STATUS 0x%08x now, 0x%08x at open (%s)\n", av, as, at_open,
               tsr::adapter_status_str(at_open).c_str());
        // A value at open was left by an earlier process (an error nobody saw); the open cleared it.
        result(ok && at_open == 0, "ADAPTER_STATUS clean at open",
               at_open == 0 ? std::string("ok") : tsr::adapter_status_str(at_open) + " (left by an earlier run; cleared at open)");
        result(ok && as == 0, "ADAPTER_STATUS clean after open", tsr::adapter_status_str(as));
        // tsr_xrt_ctrl v1 encodes {16'h0001, 8'h00, NUM_CH}.
        if ((av >> 16) == 1) result((av & 0xFFu) == c->num_ch, "ADAPTER_VERSION.NUM_CH == CFG0.NUM_CH", strf("%u", av & 0xFFu));
        result(kd->num_ch() == c->num_ch, "channel buffers == CFG0.NUM_CH", strf("%u vs %u", kd->num_ch(), c->num_ch));
        for (uint32_t ch = 0; ch < kd->num_ch(); ++ch) {
            printf("  ch%u: %.3f GiB at device address 0x%" PRIx64 "\n", ch, (double) kd->ch_bytes(ch) / (1ull << 30), kd->ch_addr(ch));
            result(kd->ch_addr(ch) % 4096 == 0, strf("ch%u base 4 KiB aligned", ch).c_str());
        }
    }
}

// ---------------------------------------------------------------------------
// 2. Memory
// ---------------------------------------------------------------------------
void memory(tsr_transport * t, uint64_t mem_mib, rng & r) {
    const tsr_config * c = tsr_transport_config(t);
    const uint64_t CB = c->ch_bytes, G4 = 4ull << 30;
    printf("== memory\n");
    struct mcase { uint64_t off, n; };
    std::vector<mcase> cases = { { 0, 64 }, { 1, 1 }, { 63, 2 }, { 4093, 4099 }, { 65536 + 17, (1u << 20) + 13 } };
    if (CB > G4 + 65536) cases.push_back({ G4 - 3, 8192 + 7 });
    if (CB > (12ull << 30) + (1u << 20)) cases.push_back({ (12ull << 30) + 4095, 300001 });
    cases.push_back({ CB - 777, 777 });
    std::vector<uint8_t> buf, back;
    int bad = 0, total = 0;
    for (uint32_t ch = 0; ch < c->num_ch; ++ch) {
        for (const mcase & m : cases) {
            const uint64_t g0 = m.off >= 64 ? m.off - 64 : 0, g1 = std::min(CB, m.off + m.n + 64);
            buf.resize(g1 - g0);
            back.resize(g1 - g0);
            r.fill(buf.data(), buf.size());
            bool ok = tsr_transport_write(t, ch, g0, buf.data(), buf.size()) == 0;            // guards (and old data)
            std::vector<uint8_t> data(m.n);
            r.fill(data.data(), data.size());
            ok = ok && tsr_transport_write(t, ch, m.off, data.data(), data.size()) == 0;
            memcpy(buf.data() + (m.off - g0), data.data(), data.size());
            ok = ok && tsr_transport_read(t, ch, g0, back.data(), back.size()) == 0;
            const int64_t d = ok ? first_diff(back.data(), buf.data(), buf.size()) : -2;
            // An unaligned read of exactly the range, into an odd host address.
            std::vector<uint8_t> exact(m.n + 3);
            const bool ok2 = tsr_transport_read(t, ch, m.off, exact.data() + 3, m.n) == 0 && memcmp(exact.data() + 3, data.data(), m.n) == 0;
            total++;
            if (d != -1 || !ok2) bad++;
            result(d == -1 && ok2, strf("ch%u [0x%" PRIx64 ", +%" PRIu64 ") with guards", ch, m.off, m.n).c_str(),
                   d == -2 ? "transfer failed" : (d >= 0 ? strf("first difference at 0x%" PRIx64, g0 + (uint64_t) d) : (ok2 ? "" : "exact read differs")));
        }
    }
    printf("  pattern ranges: %d of %d passed (%zu ranges x %u channels, incl. 4 GiB crossing: %s)\n", total - bad, total, cases.size(),
           c->num_ch, CB > G4 + 65536 ? "yes" : "no (channel <= 4 GiB)");

    // Aliasing: the same offset on every channel holds different data.
    for (uint64_t off : { (uint64_t) (1u << 20), G4 + (1u << 20) }) {
        if (off + 4096 > CB) continue;
        std::vector<std::vector<uint8_t>> pat(c->num_ch, std::vector<uint8_t>(4096));
        bool ok = true;
        for (uint32_t ch = 0; ch < c->num_ch; ++ch) {
            r.fill(pat[ch].data(), 4096);
            ok = ok && tsr_transport_write(t, ch, off, pat[ch].data(), 4096) == 0;
        }
        uint32_t wrong = 0;
        for (uint32_t ch = 0; ch < c->num_ch; ++ch) {
            std::vector<uint8_t> b(4096);
            ok = ok && tsr_transport_read(t, ch, off, b.data(), 4096) == 0;
            if (b != pat[ch]) wrong++;
        }
        result(ok && wrong == 0, strf("no channel aliasing at 0x%" PRIx64, off).c_str(), strf("%u channels differ", wrong));
    }

    // Bandwidth (host <-> card, one channel at a time, page-aligned host buffer).
    if (mem_mib == 0) return;
    const uint64_t n = mem_mib << 20;
    const uint64_t off = CB >= (1ull << 30) + n ? (1ull << 30) : 0;
    if (off + n > CB) {
        printf("  bandwidth: skipped (%" PRIu64 " MiB does not fit the channel)\n", mem_mib);
        return;
    }
    void * p = nullptr, * q = nullptr;
    if (posix_memalign(&p, 4096, n) != 0 || posix_memalign(&q, 4096, n) != 0) {
        result(false, "bandwidth buffers", "out of memory");
        free(p);
        return;
    }
    r.fill(p, n);
    for (uint32_t ch = 0; ch < c->num_ch; ++ch) {
        auto t0 = std::chrono::steady_clock::now();
        const bool okw = tsr_transport_write(t, ch, off, p, n) == 0;
        const double tw = secs(t0);
        memset(q, 0, n);
        t0 = std::chrono::steady_clock::now();
        const bool okr = tsr_transport_read(t, ch, off, q, n) == 0;
        const double tr = secs(t0);
        const bool same = okw && okr && memcmp(p, q, n) == 0;
        printf("  ch%u: %" PRIu64 " MiB write %.2f GB/s, read %.2f GB/s (host wall clock, measured)\n", ch, mem_mib, n / tw / 1e9, n / tr / 1e9);
        result(same, strf("ch%u %" PRIu64 " MiB transfer intact", ch, mem_mib).c_str());
    }
    free(p);
    free(q);
}

void usage() {
    fprintf(stderr, "usage: tsr_xrt_selftest [--device KIND] [--mem-mib N] [--no-mem] [--seed N] [-v]\n");
}

} // namespace

int main(int argc, char ** argv) {
    const char * dev = getenv("TSR_DEVICE");
    if (dev == nullptr || dev[0] == '\0') dev = "xrt";
    uint64_t mem_mib = 64, seed = 1;
    bool do_mem = true;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&](const char * what) -> const char * {
            if (i + 1 >= argc) {
                fprintf(stderr, "tsr_xrt_selftest: %s needs a value\n", what);
                exit(2);
            }
            return argv[++i];
        };
        if (a == "--device") dev = val("--device");
        else if (a == "--mem-mib") mem_mib = strtoull(val("--mem-mib"), nullptr, 10);
        else if (a == "--no-mem") do_mem = false;
        else if (a == "--seed") seed = strtoull(val("--seed"), nullptr, 10);
        else if (a == "-v") g_verbose = true;
        else {
            usage();
            return 2;
        }
    }
    tsr_transport * t = tsr_transport_open(dev);
    if (t == nullptr) {
        fprintf(stderr, "tsr_xrt_selftest: cannot open the \"%s\" transport (see the messages above)\n", dev);
        return 2;
    }
    rng r{ 0x9E3779B97F4A7C15ull ^ (seed * 0x100000001B3ull) };
    identity(t);
    if (do_mem) memory(t, mem_mib, r);
    if (tsr::kdev * kd = tsr::transport_kdev(t)) {
        // The protocol layer cleared ADAPTER_STATUS at open: the register shows what happened since.
        uint32_t as = 0;
        const bool ok = kd->reg_read(tsr::KREG_ADAPTER_STATUS, &as) == 0;
        result(ok && as == 0, "ADAPTER_STATUS clean at the end", ok ? tsr::adapter_status_str(as) : std::string("unreadable"));
    }
    tsr_transport_close(t);
    printf("tsr_xrt_selftest (%s): %d checks passed, %d failed -> %s\n", dev, g_pass, g_fail, g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
