// tsr_selftest.cpp — tsr-selftest: the card bring-up checklist over any transport (docs/design/
// 06-software-verification.md "SW" §9.5, §7.6, SW-28; docs/design/01-execution-model.md "EX" §3-§5,
// §4.11; was tsr_xrt_selftest, docs/xrt-host.md §5).
//
//   tsr-selftest [--device KIND] [--image ELF] [--mem-mib N] [--no-mem] [--no-kernel] [--ids LIST]
//                [--no-shadow] [--rate-n N] [--traces DIR]... [--traces-serial DIR]...
//                [--max-traces N] [--trw MIB] [--debug] [--log-out FILE] [--seed N] [-v]
//   tsr-selftest --version          this tool's Tesseract Basis version and Basis ABI
//
//   --device KIND   transport (default: $TSR_DEVICE, else xrt): xrt on the card; mock (the adapter
//                   register model, ctest) and sim check the tool and the protocol path
//   --image ELF     firmware image (default $TSR_IMAGE, else the build's tsr-fw.elf)
//   --mem-mib N     per-channel bandwidth transfer (default 64; 0 skips it); --no-mem skips section 2
//   --no-kernel     skip section 5 (the `selftest` kernel); --ids LIST runs only those ids (e.g. 1,3,7;
//                   11 stands for the misuses)
//   --no-shadow     run section 5 and the replays without the in-process simulator (device-side and
//                   reference checks only)
//   --rate-n N      launches of the PING latency and rate measurements (default 1000)
//   --traces DIR    replay every trace below DIR (kind ops through the `replay` kernel, kind
//                   launch as full launches of their own image); --traces-serial DIR with the replay
//                   kernel's SERIAL flag (traces/model-v1conv, SW §5.4); repeatable; --max-traces N
//                   caps the replays per directory
//   --trw MIB       run the kernel tests and the trace replays with the trace writer on (ring of MIB MiB at the top
//                   of the channel 0 IO region), then drain it and run the DB §8.7 self-checks against the PERF
//                   deltas of the whole run (the card's channel 0 then carries trace and DMA writes together)
//   --debug         DMI smoke (DB §3.4-§3.5): halt, read dpc / dcsr, single step, resume, PING after
//   --log-out FILE  every log word drained during the run, raw little-endian (decode: tsr-log)
//
// Sections (each check prints PASS / FAIL with -v, FAIL always):
//   1. Identity: kdev description; ID, VERSION (Tesseract Core capability 1.x, EX §7.5), CFG0-3,
//      CAPS, CTL_ID (a control core present), RV_FREQ (clk_rv present; F_RV / F_CORE), RV_CFG,
//      Q_DEPTH per unit, LQ_CFG ring depths, PERF_INFO, LOG_CFG, STATUS; the adapter (ADAPTER_VERSION
//      major 1 and NUM_CH, ADAPTER_STATUS at open and now, MEM_EN, channel bases = the buffers'
//      device addresses) or "no adapter" (sim).
//   2. Memory, per channel: pattern write / read-back at aligned, unaligned,
//      4 KiB- and 4 GiB-crossing and end-of-channel ranges with guard bytes; channel aliasing;
//      write / read bandwidth.
//   3. Boot (EX §3.3-§3.5): the image (id, base kernels graph / selftest / replay at 0 / 1 / 2; its
//      Tesseract Basis version, Basis ABI, the capability it needs and its kinds, SW §6.9); a boot
//      (load with the receive CRC and VERIFY, HELLO with the image id, Basis ABI, kind table
//      signature, kernel count and Basis version: the transport checks them) and a second boot that
//      must be a warm restart (no ITCM segment written); the booted image's id; RT_STATE IDLE.
//   4. PING (EX §4.11): one round trip with the ARG0 echo; N sequential round trips (median / p99
//      host wall time, the device's DURATION); N pipelined launches (LQ_DEPTH outstanding): rate.
//      Measured values, not gated (EX §4.11's targets are estimates for the card).
//   5. The `selftest` kernel (SW §7.6): ids 1-11 (id 2 twice; id 11, the GDN op, needs CAPS bit 8) and five misuses, each run on the
//      device and on the shadow simulator with the same image, inputs and arguments; compared:
//      the answer (r0-r6, tickets and push indices relative), every output byte (the data area and
//      the ids' other output regions), the per-unit completion counts, the kernel events and the
//      log records; plus RET (the device-side checks of ids 1, 7, 8, 10) and references computed
//      here for ids 1, 2, 3, 7, 8, 9 and 11 (copies, QACT, MVM, the recovery round trip, the SPM window
//      pattern, the id 9 log strings from the ELF and its event).
//   6. --traces: trace replays through the device (tsr_devrun.h): answer = the trace's DONE, every
//      expect region, and device = shadow; traces made for another configuration are skipped.
//   7. --debug: the DMI smoke (needs CTL_ID.DM).
//   8. Log and status: every drained word decodes (DB §6.1), no record dropped (LOG_DROPPED,
//      STATUS.LOG_DROP), no sticky error in STATUS, the transport's recovery counters 0, ADAPTER_STATUS
//      clean at the end and no PLAT_ERR seen.
// Exit: 0 every check passed, 1 a check failed, 2 the transport did not open / bad usage.
// Test hook (tools_selftest_mock_faults): TSR_SELFTEST_MUTATE_SHADOW=1 flips one input byte of every
// case on the shadow only, so the device = shadow comparisons must fail while the device's own and
// reference checks pass.

#include "tsr_devrun.h"
#include "tsr_elf.h"
#include "tsr_isa.h"
#include "tsr_kdev.h"
#include "tsr_logdec.h"
#include "tsr_ref.h"
#include "tsr_ref_gdn.h"
#include "tsr_trace.h"
#include "tsr_dbg_lib.hpp"   // the trace writer (sw/tools/dbg)
#include "tsr_transport.h"

#include "selftest_abi.hpp"   // sw/ctrl/ref: the selftest kernel's arguments and layouts
#include "tsr/opl.hpp"         // kind_name (SW §6.9)

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

int  g_fail = 0, g_pass = 0;
bool g_verbose = false;

void result(bool ok, const std::string & what, const std::string & detail = std::string()) {
    (ok ? g_pass : g_fail)++;
    if (!ok || g_verbose) printf("  %s  %s%s%s\n", ok ? "PASS" : "FAIL", what.c_str(), detail.empty() ? "" : ": ", detail.c_str());
}

std::string strf(const char * f, ...) __attribute__((format(printf, 1, 2)));
std::string strf(const char * f, ...) {
    char buf[2048];
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
    uint32_t below(uint32_t n) { return (uint32_t) ((next() >> 32) % n); }
    double unit() { return (double) (next() >> 11) * 0x1.0p-53; }
    double normal() {
        const double u1 = ((next() >> 11) + 1) * 0x1.0p-53, u2 = (next() >> 11) * 0x1.0p-53;
        return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
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

uint32_t f2u(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    return u;
}
float u2f(uint32_t u) {
    float f;
    memcpy(&f, &u, 4);
    return f;
}
void wr(uint8_t * p, uint32_t i, uint32_t v) {
    p[4 * i] = (uint8_t) v; p[4 * i + 1] = (uint8_t) (v >> 8); p[4 * i + 2] = (uint8_t) (v >> 16); p[4 * i + 3] = (uint8_t) (v >> 24);
}
uint32_t rd(const uint8_t * p, uint32_t i) {
    return (uint32_t) p[4 * i] | ((uint32_t) p[4 * i + 1] << 8) | ((uint32_t) p[4 * i + 2] << 16) | ((uint32_t) p[4 * i + 3] << 24);
}

const char * unit_name(uint32_t u) {
    static const char * n[6] = { "DMA", "QACT", "MVM", "VU", "ATT", "SYNC" };
    return u < 6 ? n[u] : "?";
}

// ---------------------------------------------------------------------------
// 1. Identity
// ---------------------------------------------------------------------------
void identity(tsr_transport * t) {
    const tsr_config * c = tsr_transport_config(t);
    printf("== identity (%s)\n", tsr_transport_name(t));
    tsr::kdev * kd = tsr::transport_kdev(t);
    if (kd != nullptr) {
        const std::string d = kd->describe();
        size_t p = 0;
        while (p < d.size()) {
            const size_t e = d.find('\n', p);
            printf("  %s\n", d.substr(p, e == std::string::npos ? std::string::npos : e - p).c_str());
            if (e == std::string::npos) break;
            p = e + 1;
        }
    }
    auto R = [&](uint32_t off) { return tsr_transport_csr_read(t, off); };
    const uint32_t id = R(TSR_CSR_ID), ver = R(TSR_CSR_VERSION), cfg0 = R(TSR_CSR_CFG0), cfg1 = R(TSR_CSR_CFG1), cfg2 = R(TSR_CSR_CFG2);
    const uint32_t cfg3 = R(TSR_CSR_CFG3), caps = R(TSR_CSR_CAPS), ctl = R(TSR_CSR_CTL_ID), freq = R(TSR_CSR_RV_FREQ);
    const uint32_t rvcfg = R(TSR_CSR_RV_CFG), qd = R(TSR_CSR_Q_DEPTH), lqcfg = R(TSR_CSR_LQ_CFG), pinfo = R(TSR_CSR_PERF_INFO);
    const uint32_t logcfg = R(TSR_CSR_LOG_CFG), st = R(TSR_CSR_STATUS), wdog = R(TSR_CSR_WDOG_CFG);
    tsr_versions dv;                                      // the device part of tsr_versions_str (SW §6.9):
    tsr_transport_versions(t, &dv);                       // "Tesseract Core 1.0 (VU ATT)"
    dv.basis = 0;
    char core[128];
    tsr_versions_str(&dv, 0, core, sizeof(core));
    const char * capn = strchr(core, '(');
    printf("  ID 0x%08x  Tesseract Core capability %u.%u (VERSION 0x%08x), CAPS 0x%08x %s\n", id, (ver >> 8) & 0xFFu, ver & 0xFFu, ver,
           caps, capn ? capn : "");
    printf("  CFG0 0x%08x  CFG1 0x%08x  CFG2 0x%08x  CFG3 0x%08x\n", cfg0, cfg1, cfg2, cfg3);
    printf("  NUM_CH %u  LANES %u  T_MAX %u  K_MAX %u  SPM %u KiB  VL %u  NSREG %u  ch_bytes %.3f GiB\n", c->num_ch, c->lanes, c->t_max,
           c->k_max, c->spm_bytes / 1024, c->vl, c->nsreg, (double) c->ch_bytes / (1ull << 30));
    // The low byte of CTL_ID is an internal identifier (SW §6.9), shown in the raw value only.
    printf("  CTL_ID 0x%08x (control core%s%s%s%s%s%s%s)  RV_CFG ITCM %u KiB DTCM %u KiB  RV_FREQ %u (F_RV / F_CORE %.4f)\n", ctl,
           (ctl & TSR_CTL_ID_DM) ? ", debug module" : "", (ctl & (1u << 9)) ? ", completion records" : "",
           (ctl & (1u << 10)) ? ", SPM bridge" : "", (ctl & (1u << 11)) ? ", log ring" : "", (ctl & (1u << 12)) ? ", watchdog" : "",
           (ctl & (1u << 13)) ? ", SBA" : "", (ctl & (1u << 14)) ? ", SPM alias" : "", rvcfg & 0xFFFFu, rvcfg >> 16, freq,
           freq / 65536.0);
    std::string qs;
    for (uint32_t u = 0; u < TSR_NUM_UNITS; ++u) qs += strf("%s%s %u", u ? ", " : "", unit_name(u), (qd >> (5 * u)) & 0x1Fu);
    printf("  Q_DEPTH 0x%08x (%s)  LQ_CFG LQ %u LCQ %u LOG %u  PERF_INFO %u counters, X = %u  WDOG_CFG %u  LOG_CFG 0x%08x\n", qd, qs.c_str(),
           TSR_GET(lqcfg, TSR_LQ_CFG_LQ_DEPTH), TSR_GET(lqcfg, TSR_LQ_CFG_LCQ_DEPTH), TSR_GET(lqcfg, TSR_LQ_CFG_LOG_DEPTH),
           TSR_GET(pinfo, TSR_PERF_INFO_COUNT), TSR_GET(pinfo, TSR_PERF_INFO_X), wdog, logcfg);
    printf("  STATUS 0x%08x after open\n", st);
    result(id == TSR_ID_MAGIC, "ID is \"TSR1\"", strf("0x%08x", id));
    result(TSR_GET(ver, TSR_VERSION_MAJOR) == TSR_ISA_MAJOR, strf("Tesseract Core capability %u.x", TSR_ISA_MAJOR),
           strf("%u.%u", (ver >> 8) & 0xFFu, ver & 0xFFu));
    result(TSR_GET(ctl, TSR_CTL_ID_VERSION) != 0u, "CTL_ID: control core present", strf("0x%08x", ctl));
    result(freq != 0, "RV_FREQ != 0 (clk_rv runs)", strf("%u", freq));
    result((rvcfg & 0xFFFFu) != 0 && (rvcfg >> 16) != 0, "RV_CFG: ITCM and DTCM present", strf("0x%08x", rvcfg));
    result(((qd >> 0) & 0x1F) && ((qd >> 5) & 0x1F) && ((qd >> 10) & 0x1F) && ((qd >> 25) & 0x1F), "Q_DEPTH: DMA, QACT, MVM and SYNC built",
           strf("0x%08x", qd));
    result(TSR_GET(lqcfg, TSR_LQ_CFG_LQ_DEPTH) != 0 && TSR_GET(lqcfg, TSR_LQ_CFG_LCQ_DEPTH) != 0 && TSR_GET(lqcfg, TSR_LQ_CFG_LOG_DEPTH) != 0,
           "LQ_CFG: LQ, LCQ and log ring present", strf("0x%08x", lqcfg));
    if (kd == nullptr) return;
    if (!kd->has_adapter()) {
        printf("  no adapter (the %s kind is the core CSR port only)\n", tsr_transport_name(t));
    } else {
        uint32_t av = 0, as = 0, actl = 0;
        const bool ok = kd->reg_read(tsr::KREG_ADAPTER_VERSION, &av) == 0 && kd->reg_read(tsr::KREG_ADAPTER_STATUS, &as) == 0 &&
                        kd->reg_read(tsr::KREG_ADAPTER_CTRL, &actl) == 0;
        const uint32_t at_open = tsr::transport_adapter_log(t).at_open;
        printf("  ADAPTER_VERSION 0x%08x  ADAPTER_CTRL 0x%08x  ADAPTER_STATUS 0x%08x now, 0x%08x at open (%s)\n", av, actl, as, at_open,
               tsr::adapter_status_str(at_open).c_str());
        // A value at open was left by an earlier process (an error nobody saw); the open cleared it.
        result(ok && at_open == 0, "ADAPTER_STATUS clean at open",
               at_open == 0 ? std::string("ok") : tsr::adapter_status_str(at_open) + " (left by an earlier run; cleared at open)");
        result(ok && as == 0, "ADAPTER_STATUS clean after open", tsr::adapter_status_str(as));
        result(TSR_GET(av, TSR_ADAPTER_VERSION_MAJOR) == TSR_ADAPTER_MAJOR && (av & 0xFFu) == c->num_ch,
               strf("ADAPTER_VERSION: major %u, NUM_CH = CFG0.NUM_CH", TSR_ADAPTER_MAJOR), strf("0x%08x", av));
        result((actl & TSR_ADAPTER_CTRL_MEM_EN) != 0, "ADAPTER_CTRL.MEM_EN (bound at open)", strf("0x%08x", actl));
        for (uint32_t ch = 0; ch < kd->num_ch(); ++ch) {
            uint32_t lo = 0, hi = 0;
            const bool okb = kd->reg_read(tsr::KREG_CH_BASE0 + 8 * ch, &lo) == 0 && kd->reg_read(tsr::KREG_CH_BASE0 + 8 * ch + 4, &hi) == 0;
            const uint64_t base = ((uint64_t) hi << 32) | lo;
            result(okb && base == kd->ch_addr(ch), strf("ch%u base register = buffer address", ch),
                   strf("0x%" PRIx64 " vs 0x%" PRIx64, base, kd->ch_addr(ch)));
        }
    }
    result(kd->num_ch() == c->num_ch, "channel buffers == CFG0.NUM_CH", strf("%u vs %u", kd->num_ch(), c->num_ch));
    for (uint32_t ch = 0; ch < kd->num_ch(); ++ch) {
        printf("  ch%u: %.3f GiB at device address 0x%" PRIx64 "\n", ch, (double) kd->ch_bytes(ch) / (1ull << 30), kd->ch_addr(ch));
        result(kd->ch_addr(ch) % 4096 == 0, strf("ch%u base 4 KiB aligned", ch));
    }
}

// ---------------------------------------------------------------------------
// 2. Memory
// ---------------------------------------------------------------------------
// Channel capacity. The memory behind a buffer can be smaller than the buffer, and an address above
// it then wraps: on the lab U200 the four DIMMs hold 8 GiB each under a memory topology of 16 GB per
// bank, and offset + 8 GiB reads and writes offset. A write and a read back at the same address
// cannot see that. This writes two distinct patterns at 0 and at P, for every power of two P from
// 1 MiB up to below the buffer, and reads both back: the first P at which 0 returns the second
// pattern is the capacity (memory repeats every P bytes), and none means the whole buffer is
// memory. It uses the kdev's own access to the whole buffer, not the transport's (which refuses
// offsets above its channel size), and restores the bytes it touches. The transport's channel size
// must equal the capacity found: larger, and it would place data on wrapped memory.
void capacity(tsr_transport * t, rng & r) {
    tsr::kdev * kd = tsr::transport_kdev(t);
    if (kd == nullptr) return;
    const tsr_config * c = tsr_transport_config(t);
    const double G = (double) (1ull << 30);
    printf("  channel capacity (distinct patterns at 0 and P, P = 2^20 up to below the buffer):\n");
    uint64_t smallest = UINT64_MAX;
    for (uint32_t ch = 0; ch < kd->num_ch(); ++ch) {
        const uint64_t buf = kd->buf_bytes(ch);
        uint64_t cap = buf & ~(uint64_t) 63;
        std::string bad;
        for (uint64_t P = 1ull << 20; P + 64 <= buf && bad.empty(); P <<= 1) {
            uint8_t sa[64], sb[64], pa[64], pb[64], ra[64], rb[64];
            r.fill(pa, 64);
            for (int i = 0; i < 64; ++i) pb[i] = (uint8_t) ~pa[i];
            const bool ok = kd->mem_read(ch, 0, sa, 64) == 0 && kd->mem_read(ch, P, sb, 64) == 0 && kd->mem_write(ch, 0, pa, 64) == 0 &&
                            kd->mem_write(ch, P, pb, 64) == 0 && kd->mem_read(ch, 0, ra, 64) == 0 && kd->mem_read(ch, P, rb, 64) == 0;
            const bool restored = kd->mem_write(ch, P, sb, 64) == 0 && kd->mem_write(ch, 0, sa, 64) == 0;
            if (!ok || !restored) {
                bad = strf("access failed at 0 / 0x%" PRIx64, P);
            } else if (memcmp(ra, pa, 64) == 0 && memcmp(rb, pb, 64) == 0) {
                continue;                                      // two cells: no wrap at P
            } else if (memcmp(ra, pb, 64) == 0 && memcmp(rb, pb, 64) == 0) {
                cap = P;                                       // 0 reads what was written at P: wraps every P
                break;
            } else {
                bad = strf("0 and 0x%" PRIx64 " read neither pattern", P);
            }
        }
        smallest = std::min(smallest, cap);
        printf("  ch%u: %.2f GiB of memory in a %.2f GiB buffer%s, transport channel size %.2f GiB\n", ch, (double) cap / G, (double) buf / G,
               cap < buf ? strf(" (wraps at 0x%" PRIx64 ")", cap).c_str() : "", (double) kd->ch_bytes(ch) / G);
        result(bad.empty() && kd->ch_bytes(ch) == cap, strf("ch%u capacity: the transport's channel size equals the measured memory", ch),
               bad.empty() ? strf("%.3f GiB vs %.3f GiB", (double) kd->ch_bytes(ch) / G, (double) cap / G) : bad);
    }
    result(c->ch_bytes <= smallest, "the configured channel size (CFG ch_bytes) is not above the smallest capacity",
           strf("%.3f GiB vs %.3f GiB", (double) c->ch_bytes / G, (double) smallest / G));
}

void memory(tsr_transport * t, uint64_t mem_mib, rng & r) {
    const tsr_config * c = tsr_transport_config(t);
    const uint64_t CB = c->ch_bytes, G4 = 4ull << 30;
    printf("== memory\n");
    capacity(t, r);
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
            result(d == -1 && ok2, strf("ch%u [0x%" PRIx64 ", +%" PRIu64 ") with guards", ch, m.off, m.n),
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
        result(ok && wrong == 0, strf("no channel aliasing at 0x%" PRIx64, off), strf("%u channels differ", wrong));
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
        result(same, strf("ch%u %" PRIu64 " MiB transfer intact", ch, mem_mib));
    }
    free(p);
    free(q);
}

// ---------------------------------------------------------------------------
// 3. Boot
// ---------------------------------------------------------------------------
bool boot(tsr_transport * t, tsr_image * img) {
    printf("== boot (%s, id %016" PRIx64 ")\n", tsr_image_path(img), tsr_image_id(img));
    const int kg = tsr_image_kernel(img, "graph"), ks = tsr_image_kernel(img, "selftest"), kr = tsr_image_kernel(img, "replay");
    result(kg == 0 && ks == 1 && kr == 2, "base kernels graph / selftest / replay at 0 / 1 / 2 (EX §4.6)", strf("%d / %d / %d", kg, ks, kr));
    // The image's versions (SW §6.9), as the transport read them from .note.tsr and .tsr_kinds.
    tsr_versions iv;
    tsr_image_versions(img, &iv);
    const uint32_t hb = tsr_basis_version();
    printf("  image %s: id %016" PRIx64 ", Tesseract Basis %u.%u.%u, Basis ABI %u, needs capability %u.%u; host Tesseract Basis "
           "%u.%u.%u\n", tsr_image_path(img), iv.image_id, (iv.basis >> 16) & 0xFFu, (iv.basis >> 8) & 0xFFu, iv.basis & 0xFFu, iv.abi,
           (iv.core_min >> 8) & 0xFFu, iv.core_min & 0xFFu, (hb >> 16) & 0xFFu, (hb >> 8) & 0xFFu, hb & 0xFFu);
    std::string kl;
    for (uint32_t k = 0; k < 64; ++k)
        if (iv.kind_rev[k]) kl += strf(" %s r%u", tsr::opl::kind_name(k), iv.kind_rev[k]);
    printf("  kinds (%u, signature 0x%04x):%s\n", iv.n_kinds, iv.kinds_sig, kl.c_str());
    bool all = true;
    for (int pass = 0; pass < 2; ++pass) {
        const auto t0 = std::chrono::steady_clock::now();
        const int rc = tsr_transport_boot(t, img);
        const double w = secs(t0);
        const tsr::boot_info bi = tsr::transport_boot_info(t);
        printf("  boot %d: %s, %s, %u ITCM / %u DTCM segments loaded, %u LOAD_DATA words, HELLO after %.3f ms (transport time), "
               "%.3f ms wall, crt0 to HELLO %u core cycles\n", pass + 1, rc == 0 ? "ok" : tsr_transport_strerror(rc),
               bi.warm ? "warm restart" : "cold load", bi.itcm_loaded, bi.dtcm_loaded, bi.words_written, bi.time_ns / 1e6, w * 1e3,
               bi.hello_cycles);
        result(rc == 0, strf("boot %d: load, VERIFY, HELLO with the image id, Basis ABI, kind table and kernel count", pass + 1),
               rc ? tsr_transport_strerror(rc) : "");
        if (rc) return false;
        tsr_versions bv;
        result(tsr_transport_versions(t, &bv) == 0 && bv.basis == iv.basis && bv.abi == iv.abi && bv.kinds_sig == iv.kinds_sig &&
                   bv.image_id == iv.image_id,
               strf("boot %d: HELLO: Tesseract Basis, ABI and kind table match the image", pass + 1),
               strf("%u.%u.%u, ABI %u, kinds 0x%04x", (bv.basis >> 16) & 0xFFu, (bv.basis >> 8) & 0xFFu, bv.basis & 0xFFu, bv.abi,
                    bv.kinds_sig));
        const tsr_image * cur = tsr_transport_image(t);
        result(cur != nullptr && tsr_image_id(cur) == tsr_image_id(img), strf("boot %d: the booted image's id", pass + 1),
               cur ? strf("%016" PRIx64, tsr_image_id(cur)) : "none");
        if (pass == 1) result(bi.warm && bi.itcm_loaded == 0, "boot 2: warm restart (ITCM verified, not rewritten, EX §3.5)");
        // HELLO done: the runtime is idle (R1) and nothing is outstanding.
        const uint32_t rt = tsr_transport_csr_read(t, TSR_CSR_RT_STATE), st = tsr_transport_csr_read(t, TSR_CSR_STATUS);
        result(TSR_GET(rt, TSR_RT_STATE_PHASE) == TSR_RT_STATE_PHASE_IDLE && (st & TSR_STATUS_RV_RUNNING),
               strf("boot %d: RT_STATE IDLE, STATUS.RV_RUNNING", pass + 1), strf("RT_STATE 0x%08x STATUS 0x%08x", rt, st));
        all = all && rc == 0;
    }
    return all;
}

// ---------------------------------------------------------------------------
// 4. PING round trip and pipelined rate (EX §4.11)
// ---------------------------------------------------------------------------
void ping(tsr_transport * t, uint32_t n) {
    printf("== launch latency (EX §4.11: PING)\n");
    tsr_launch l;
    memset(&l, 0, sizeof(l));
    l.type = TSR_DESC_TYPE_PING;
    l.arg[0] = 0xC0FFEE00u;
    tsr_record rec;
    int rc = tsr_transport_call(t, &l, TSR_TIMEOUT_DEFAULT, &rec);
    tsr_result res;
    tsr_record_decode(&rec, &res);
    result(rc == 0 && res.type == TSR_LCQ_R0_TYPE_PONG && res.status == 0 && (uint32_t) res.ret == l.arg[0],
           "PING -> PONG with the ARG0 echo", strf("rc %d type 0x%02x status %s r2 0x%08x", rc, res.type, tsr_status_name(res.status),
                                                  (uint32_t) res.ret));
    if (n == 0) return;
    std::vector<double> rt;
    uint32_t bad = 0;
    for (uint32_t i = 0; i < n; ++i) {
        l.arg[0] = 0xC0DE0000u + i;
        const auto t0 = std::chrono::steady_clock::now();
        rc = tsr_transport_call(t, &l, TSR_TIMEOUT_DEFAULT, &rec);
        rt.push_back(secs(t0));
        tsr_record_decode(&rec, &res);
        if (rc || res.type != TSR_LCQ_R0_TYPE_PONG || (uint32_t) res.ret != l.arg[0]) bad++;
    }
    // The round trips above use EX §4.8's minimal LCQ reads, which skip r7 (DURATION): a short
    // sample with whole records gives the PONG's DURATION.
    std::vector<uint32_t> dur;
    tsr::transport_set_full_records(t, true);
    for (uint32_t i = 0; i < std::min<uint32_t>(n, 64); ++i) {
        l.arg[0] = 0xD0DE0000u + i;
        rc = tsr_transport_call(t, &l, TSR_TIMEOUT_DEFAULT, &rec);
        tsr_record_decode(&rec, &res);
        if (rc || res.type != TSR_LCQ_R0_TYPE_PONG || (uint32_t) res.ret != l.arg[0]) bad++;
        dur.push_back(res.duration);
    }
    tsr::transport_set_full_records(t, false);
    std::sort(rt.begin(), rt.end());
    std::sort(dur.begin(), dur.end());
    const double med = rt[n / 2], p99 = rt[std::min<size_t>(n - 1, (size_t) (n * 0.99))];
    printf("  %u sequential round trips: median %.2f us, p99 %.2f us (host wall clock); PONG DURATION median %u core cycles "
           "(%.2f us at the nominal %.1f MHz; %zu PINGs read whole)\n", n, med * 1e6, p99 * 1e6, dur[dur.size() / 2],
           dur[dur.size() / 2] / tsr::CORE_HZ_NOMINAL * 1e6, tsr::CORE_HZ_NOMINAL / 1e6, dur.size());
    result(bad == 0, strf("%u sequential PINGs answered with their echo", n), strf("%u wrong", bad));
    // Pipelined: launch blocks only at LQ_DEPTH outstanding (host-side credit).
    std::vector<uint32_t> seqs(n);
    bad = 0;
    const auto t0 = std::chrono::steady_clock::now();
    uint32_t waited = 0;
    for (uint32_t i = 0; i < n; ++i) {
        l.arg[0] = 0xBEEF0000u + i;
        if (tsr_transport_launch(t, &l, &seqs[i]) != 0) {
            bad++;
            break;
        }
        // keep the answers flowing: wait for the oldest once the queue is full
        while (i + 1 - waited >= 8u && waited <= i) {
            if (tsr_transport_wait(t, seqs[waited], TSR_TIMEOUT_DEFAULT, &rec) != 0 || rec.w[2] != 0xBEEF0000u + waited) bad++;
            waited++;
        }
    }
    for (; waited < n; ++waited)
        if (tsr_transport_wait(t, seqs[waited], TSR_TIMEOUT_DEFAULT, &rec) != 0 || rec.w[2] != 0xBEEF0000u + waited) bad++;
    const double w = secs(t0);
    printf("  %u pipelined launches: %.0f launches/s (host wall clock)\n", n, n / w);
    printf("  EX §4.11 targets on the card (estimates): round trip <= 10 us, pipelined >= 100 K launches/s%s\n",
           strcmp(tsr_transport_name(t), "xrt") == 0 ? "" : " (this kind runs the simulator: the times above are its speed)");
    result(bad == 0, strf("%u pipelined PINGs answered in order with their echo", n), strf("%u wrong", bad));
}

// ---------------------------------------------------------------------------
// 5. The `selftest` kernel on the device and the shadow (SW §7.6)
// ---------------------------------------------------------------------------
constexpr uint64_t AREA0     = TSR_REGION_IO_OFF / 64u;     // ch0 beat of the first data area
constexpr uint64_t AREA_STEP = 0x10000u;                     // beats between the ids' areas (4 MiB)
constexpr uint64_t W_BASE    = TSR_REGION_WEIGHT_OFF / 64u;  // weights, every channel
constexpr uint64_t S1_BEAT   = 0x400000u;                    // id 1 scratch, every channel (256 MiB)
constexpr uint64_t KV_BASE   = 0xC00000u;                    // id 5 planes, every channel (768 MiB)
constexpr uint32_t LMAX      = 4096u;

struct region { uint32_t ch; uint64_t beat; std::vector<uint8_t> data; };
struct outreg { uint32_t ch; uint64_t beat; uint64_t lines; };

struct kcase {
    std::string name;
    uint32_t    id = 0;
    uint64_t    area = 0;         // ch0 beat of the data area
    uint32_t    lines = 0;        // a2
    uint32_t    seed = 0;         // a3
    int32_t     want_ret = 0;
    uint32_t    want_status = TSR_LCQ_R0_STATUS_OK;
    std::vector<region> in;       // written to both targets (the area itself is in[0])
    std::vector<outreg> out;      // compared device vs shadow after the launch (besides the area)
    // reference checks on the device's DRAM and run (ids 1, 2, 3, 7, 8, 9, 11)
    std::function<void(tsr_tgt *, const tsr_tgt_run &)> ref;
};

struct ctx {
    tsr_tgt *  dev = nullptr;
    tsr_tgt *  sh = nullptr;     // may be NULL (--no-shadow)
    int        kidx = -1;
    tsr_elf *  elf = nullptr;
    uint32_t   C = 0, TM = 1, caps = 0;
    std::vector<uint32_t> * log_all = nullptr;
};

std::vector<uint8_t> read_dev(tsr_tgt * g, uint32_t ch, uint64_t beat, uint64_t lines) {
    std::vector<uint8_t> b((size_t) lines * 64u);
    if (tsr_tgt_read(g, ch, beat * 64u, b.data(), b.size()) != 0) b.clear();
    return b;
}

std::vector<uint8_t> area_with_params(uint32_t id, uint32_t lines, const std::vector<uint32_t> & p) {
    std::vector<uint8_t> a((size_t) lines * 64u, 0);
    wr(a.data(), 0, TSR_ST_MAGIC);
    wr(a.data(), 1, id);
    for (size_t i = 0; i < p.size(); ++i) wr(a.data(), 2 + (uint32_t) i, p[i]);
    return a;
}

void rand_f32(rng & r, uint8_t * dst, uint32_t n, double scale) {
    for (uint32_t i = 0; i < n; ++i) wr(dst, i, f2u((float) (r.normal() * scale)));
}

uint16_t rand_fp16_scale(rng & r) {   // mostly normal fp16 magnitudes 2^-14 .. 2^-2 (sw/sim/tests/test_sim_util.h)
    const uint32_t k = r.below(64);
    const uint16_t sign = (uint16_t) ((r.next() & 1u) << 15);
    if (k == 0) return sign;
    if (k == 1) return (uint16_t) (sign | (1 + r.below(0x3FF)));
    const uint16_t e = (uint16_t) (1 + r.below(13));
    return (uint16_t) (sign | (e << 10) | r.below(1024));
}

// An N x K tensor of random ggml blocks, repacked into the device layout at w_off (beats) on every
// channel (symmetric allocation, tsr_row_location). Returns the per-channel regions; rows[n] gets
// the repacked row n (for the references).
std::vector<region> weights(rng & r, uint32_t fmt, uint32_t N, uint32_t K, uint32_t C, uint64_t w_off,
                            std::vector<std::vector<uint8_t>> * rows) {
    const size_t bsz = tsr_fmt_ggml_block_bytes(fmt), nb = K / tsr_fmt_block_size(fmt);
    const uint64_t chb = tsr_tensor_ch_beats(fmt, N, K, C);
    std::vector<region> out(C);
    for (uint32_t c = 0; c < C; ++c) out[c] = { c, w_off, std::vector<uint8_t>((size_t) chb * 64u, 0) };
    std::vector<uint8_t> blk(bsz * nb), row((size_t) tsr_row_beats(fmt, K) * 64u);
    if (rows) rows->assign(N, {});
    for (uint32_t n = 0; n < N; ++n) {
        r.fill(blk.data(), blk.size());
        for (size_t b = 0; b < nb; ++b) {
            const uint16_t d = rand_fp16_scale(r);
            blk[b * bsz] = (uint8_t) d;
            blk[b * bsz + 1] = (uint8_t) (d >> 8);
            if (fmt == TSR_FMT_Q4_K) {
                const uint16_t dm = rand_fp16_scale(r);
                blk[b * bsz + 2] = (uint8_t) dm;
                blk[b * bsz + 3] = (uint8_t) (dm >> 8);
            }
        }
        tsr_repack_row(fmt, blk.data(), K, row.data());
        uint32_t ch;
        uint64_t off;
        tsr_row_location(fmt, N, K, C, n, &ch, &off);
        memcpy(out[ch].data.data() + off * 64u, row.data(), row.size());
        if (rows) (*rows)[n] = row;
    }
    return out;
}

bool cmp_ref(const std::string & what, const uint8_t * got, const uint8_t * want, size_t n) {
    const int64_t d = first_diff(got, want, n);
    if (d < 0) {
        result(true, what, strf("%zu bytes", n));
        return true;
    }
    result(false, what, strf("differs at byte %" PRId64 " (word 0x%08x, reference 0x%08x)", d, rd(got, (uint32_t) (d / 4)),
                             rd(want, (uint32_t) (d / 4))));
    return false;
}

// Runs one case on the device and the shadow; every comparison of SW §7.6.
void run_case(ctx & x, kcase & c) {
    tsr_tgt * tg[2] = { x.dev, x.sh };
    const int ntg = x.sh ? 2 : 1;
    bool ok = true;
    char err[512];
    if (x.sh && tsr_tgt_ready(x.sh, err, sizeof(err)) != 0) {
        result(false, c.name + ": shadow restart after a failed launch", err);
        return;
    }
    for (int k = 0; k < ntg; ++k)
        for (const region & r : c.in)
            if (tsr_tgt_write(tg[k], r.ch, r.beat * 64u, r.data.data(), r.data.size()) != 0) {
                result(false, c.name + ": writing the inputs", tsr_tgt_name(tg[k]));
                return;
            }
    static const bool mutate = getenv("TSR_SELFTEST_MUTATE_SHADOW") && atoi(getenv("TSR_SELFTEST_MUTATE_SHADOW")) != 0;
    if (mutate && x.sh && !c.in.empty() && c.in[0].data.size() > 64) {   // test hook: the byte after the parameter line
        uint8_t b = (uint8_t) (c.in[0].data[64] ^ 0x01u);
        tsr_tgt_write(x.sh, c.in[0].ch, c.in[0].beat * 64u + 64u, &b, 1);
    }
    const uint32_t args[7] = { c.id, (uint32_t) c.area, c.lines, c.seed, 0, 0, 0 };
    tsr_tgt_run run[2];
    memset(run, 0, sizeof(run));
    for (int k = 0; k < ntg; ++k) tsr_tgt_launch(tg[k], TSR_DESC_TYPE_LAUNCH, 0, (uint32_t) x.kidx, args, &run[k]);
    if (x.log_all) x.log_all->insert(x.log_all->end(), run[0].log, run[0].log + run[0].n_log);
    const uint32_t st = TSR_GET(run[0].rec[0], TSR_LCQ_R0_STATUS), type = TSR_GET(run[0].rec[0], TSR_LCQ_R0_TYPE);
    const bool answer_ok = run[0].rc == 0 && type == TSR_LCQ_R0_TYPE_DONE && st == c.want_status && (int32_t) run[0].rec[2] == c.want_ret;
    result(answer_ok, c.name + ": DONE",
           strf("rc %d type 0x%02x status %s RET 0x%08x (expected %s, 0x%08x); first error ticket 0x%08x code %u info 0x%08x w0 0x%08x",
                run[0].rc, type, tsr_status_name((uint8_t) st), run[0].rec[2], tsr_status_name((uint8_t) c.want_status),
                (uint32_t) c.want_ret, run[0].rec[3], TSR_GET(run[0].rec[0], TSR_LCQ_R0_ERR_CODE), run[0].rec[4], run[0].rec[5]));
    ok = ok && answer_ok;
    std::vector<outreg> outs = c.out;
    outs.insert(outs.begin(), outreg{ 0, c.area, c.in.empty() ? (uint64_t) c.lines : (uint64_t) (c.in[0].data.size() / 64u) });
    if (x.sh) {
        char m[1024];
        const uint32_t nd = tsr_tgt_compare(&run[0], &run[1], m, sizeof(m));
        result(nd == 0, c.name + ": device = shadow (answer, per-unit completion counts, events, log)", m);
        ok = ok && nd == 0;
        uint64_t lines = 0;
        bool same = true;
        std::string where;
        for (const outreg & o : outs) {
            const std::vector<uint8_t> a = read_dev(x.dev, o.ch, o.beat, o.lines), b = read_dev(x.sh, o.ch, o.beat, o.lines);
            lines += o.lines;
            const int64_t d = a.size() == b.size() && !a.empty() ? first_diff(a.data(), b.data(), a.size()) : 0;
            if (d >= 0) {
                // Name every differing line (the first 96) so a card failure can be placed in the area layout.
                uint64_t nd_lines = 0;
                std::string list;
                for (uint64_t l = 0; l < o.lines && (l + 1) * 64u <= a.size() && (l + 1) * 64u <= b.size(); ++l) {
                    if (memcmp(a.data() + l * 64u, b.data() + l * 64u, 64u) == 0) continue;
                    if (nd_lines++ < 96) list += strf(" %" PRIu64, l);
                }
                if (const char * dd = getenv("TSR_SELFTEST_DUMP")) {   // keep both images of a failing region
                    for (int k = 0; k < 2; ++k) {
                        const std::vector<uint8_t> & v = k ? b : a;
                        const std::string fn = strf("%s/id%u_ch%u_%" PRIx64 "_%s.bin", dd, c.id, o.ch, o.beat, k ? "shadow" : "device");
                        if (FILE * f = fopen(fn.c_str(), "wb")) {
                            fwrite(v.data(), 1, v.size(), f);
                            fclose(f);
                        }
                    }
                }
                if (same) {
                    where = strf("ch%u beat 0x%" PRIx64 " byte %" PRId64, o.ch, o.beat, d);
                    // The words of the first differing line, device then shadow.
                    const uint64_t fl = (uint64_t) d / 64u;
                    for (int k = 0; k < 2; ++k) {
                        const std::vector<uint8_t> & v = k ? b : a;
                        where += k ? "; shadow:" : "; device:";
                        for (unsigned w = 0; w < 16 && (fl + 1) * 64u <= v.size(); ++w) {
                            uint32_t u;
                            memcpy(&u, v.data() + fl * 64u + w * 4u, 4);
                            where += strf(" %08x", u);
                        }
                    }
                }
                where += strf("%sch%u 0x%" PRIx64 ": %" PRIu64 " lines differ (%s%s)", same ? "; " : " | ", o.ch, o.beat, nd_lines,
                              list.c_str(), nd_lines > 96 ? " ..." : "");
                same = false;
            }
        }
        result(same, c.name + strf(": every output byte = shadow (%" PRIu64 " lines)", lines), where);
        ok = ok && same;
    }
    const int f0 = g_fail;
    if (c.ref && answer_ok) c.ref(x.dev, run[0]);
    ok = ok && g_fail == f0;
    if (!g_verbose) printf("  %s  %s\n", ok ? "ok  " : "FAIL", c.name.c_str());
    for (int k = 0; k < ntg; ++k) tsr_tgt_run_free(&run[k]);
}

std::vector<kcase> make_cases(ctx & x, const std::vector<uint32_t> & ids) {
    std::vector<kcase> cs;
    const uint32_t C = x.C, TM = x.TM;
    auto want = [&](uint32_t id) { return ids.empty() || std::find(ids.begin(), ids.end(), id) != ids.end(); };
    // 1 LDST: lengths 1, 63, 64, 65, LMAX on every channel and the cross-channel copies
    if (want(1)) {
        kcase c;
        c.name = "id 1 LD/ST round trips and cross-channel copies";
        c.id = 1;
        c.area = AREA0 + 1 * AREA_STEP;
        c.lines = 1 + LMAX;
        c.seed = 11;
        std::vector<uint8_t> a = area_with_params(1, c.lines, { LMAX, (uint32_t) S1_BEAT });
        rng r{ 0x1D1 };
        r.fill(a.data() + 64, (size_t) LMAX * 64u);
        uint32_t lens[5];
        const uint32_t nl = tsr_st1_lens(LMAX, lens);
        uint32_t sum = 0;
        for (uint32_t i = 0; i < nl; ++i) sum += lens[i];
        c.in.push_back({ 0, c.area, a });
        for (uint32_t ch = 0; ch < C; ++ch) {   // the scratch, poisoned
            c.in.push_back({ ch, S1_BEAT, std::vector<uint8_t>((size_t) 2 * sum * 64u, 0xEE) });
            c.out.push_back({ ch, S1_BEAT, 2ull * sum });
        }
        const std::vector<uint8_t> src(a.begin() + 64, a.end());
        c.ref = [=](tsr_tgt * d, const tsr_tgt_run &) {
            for (uint32_t ch = 0; ch < C; ++ch) {
                const std::vector<uint8_t> got = read_dev(d, ch, S1_BEAT, 2ull * sum);
                uint32_t off = 0;
                for (uint32_t k = 0; k < nl; ++k) {
                    cmp_ref(strf("id 1: length %u channel %u = source", lens[k], ch), got.data() + (size_t) off * 64u, src.data(),
                            (size_t) lens[k] * 64u);
                    off += lens[k];
                }
                cmp_ref(strf("id 1: cross copy into channel %u = source", ch), got.data() + (size_t) sum * 64u, src.data(), src.size());
            }
        };
        cs.push_back(std::move(c));
    }
    // 2 QACT: A32 and A256 (K 1024 T T_MAX; K 4096 T 1)
    for (int pass = 0; pass < 2 && want(2); ++pass) {
        const uint32_t K = pass ? 4096 : 1024, T = pass ? 1 : TM, slot = pass ? 12 : 2;
        const tsr_st2_layout L = tsr_st2(K, T);
        kcase c;
        c.name = strf("id 2 QACT A32 / A256, K %u T %u", K, T);
        c.id = 2;
        c.area = AREA0 + slot * AREA_STEP;
        c.lines = L.end;
        std::vector<uint8_t> a = area_with_params(2, L.end, { K, T });
        rng r{ 0x2D2 + K };
        rand_f32(r, a.data() + (size_t) L.x * 64u, T * K, 3.0);
        std::fill(a.begin() + (size_t) L.qa32 * 64u, a.end(), (uint8_t) 0x5A);
        c.in.push_back({ 0, c.area, a });
        const uint64_t area = c.area;
        c.ref = [=](tsr_tgt * d, const tsr_tgt_run &) {
            const std::vector<uint8_t> got = read_dev(d, 0, area, L.end);
            const uint32_t q32 = tsr_st_qa_lines(0, K), q256 = tsr_st_qa_lines(1, K);
            std::vector<float> xv(K);
            std::vector<uint8_t> qa((size_t) std::max(q32, q256) * 64u);
            for (uint32_t t = 0; t < T; ++t) {
                for (uint32_t k = 0; k < K; ++k) xv[k] = u2f(rd(a.data() + (size_t) (L.x + t * (K / 16u)) * 64u, k));
                tsr_qact(TSR_QA_A32, xv.data(), K, qa.data());
                cmp_ref(strf("id 2: K %u token %u A32 = tsr_qact", K, t), got.data() + (size_t) (L.qa32 + t * q32) * 64u, qa.data(), q32 * 64u);
                tsr_qact(TSR_QA_A256, xv.data(), K, qa.data());
                cmp_ref(strf("id 2: K %u token %u A256 = tsr_qact", K, t), got.data() + (size_t) (L.qa256 + t * q256) * 64u, qa.data(),
                        q256 * 64u);
            }
        };
        cs.push_back(std::move(c));
    }
    // 3 MVM: Q8_0 and Q4_K, T = 1 and T = T_MAX, ACC
    if (want(3)) {
        const uint32_t N = 200, K = 512;
        const tsr_st3_layout L = tsr_st3(N, K, TM);
        kcase c;
        c.name = strf("id 3 MVM Q8_0 / Q4_K, T 1 and %u, ACC", TM);
        c.id = 3;
        c.area = AREA0 + 3 * AREA_STEP;
        c.lines = L.end;
        rng r{ 0x3D3 };
        auto rows8 = std::make_shared<std::vector<std::vector<uint8_t>>>(), rows4 = std::make_shared<std::vector<std::vector<uint8_t>>>();
        std::vector<region> w8 = weights(r, TSR_FMT_Q8_0, N, K, C, W_BASE, rows8.get());
        std::vector<region> w4 = weights(r, TSR_FMT_Q4_K, N, K, C, W_BASE + 0x10000u, rows4.get());
        std::vector<uint8_t> a = area_with_params(3, L.end, { N, K, (uint32_t) W_BASE, (uint32_t) W_BASE + 0x10000u });
        rand_f32(r, a.data() + 64, TM * K, 1.0);
        std::fill(a.begin() + (size_t) L.y8_1 * 64u, a.end(), (uint8_t) 0x5A);
        c.in.push_back({ 0, c.area, a });
        c.in.insert(c.in.end(), w8.begin(), w8.end());
        c.in.insert(c.in.end(), w4.begin(), w4.end());
        const uint64_t area = c.area;
        c.ref = [=](tsr_tgt * d, const tsr_tgt_run &) {
            const std::vector<uint8_t> got = read_dev(d, 0, area, L.end);
            const uint32_t yl = (N + 15u) / 16u;
            std::vector<float> xv(K);
            std::vector<uint8_t> qa8((size_t) tsr_qa_lines(TSR_QA_A32, K) * 64u), qa4((size_t) tsr_qa_lines(TSR_QA_A256, K) * 64u);
            std::vector<uint8_t> y8((size_t) yl * 64u), y4((size_t) yl * 64u), ya((size_t) yl * 64u);
            bool all = true;
            for (uint32_t t = 0; t < TM && all; ++t) {
                for (uint32_t k = 0; k < K; ++k) xv[k] = u2f(rd(a.data() + (size_t) (L.x + t * (K / 16u)) * 64u, k));
                tsr_qact(TSR_QA_A32, xv.data(), K, qa8.data());
                tsr_qact(TSR_QA_A256, xv.data(), K, qa4.data());
                std::fill(y8.begin(), y8.end(), 0);
                std::fill(y4.begin(), y4.end(), 0);
                std::fill(ya.begin(), ya.end(), 0);
                for (uint32_t n = 0; n < N; ++n) {
                    const float v8 = tsr_dot_row(TSR_FMT_Q8_0, (*rows8)[n].data(), K, qa8.data());
                    const float v4 = tsr_dot_row(TSR_FMT_Q4_K, (*rows4)[n].data(), K, qa4.data());
                    wr(y8.data(), n, f2u(v8));
                    wr(y4.data(), n, f2u(v4));
                    wr(ya.data(), n, f2u(tsr_fadd(v8, v4)));
                }
                if (t == 0) {
                    all = all && cmp_ref("id 3: Q8_0 T=1 = tsr_dot_row", got.data() + (size_t) L.y8_1 * 64u, y8.data(), y8.size());
                    all = all && cmp_ref("id 3: Q4_K T=1 = tsr_dot_row", got.data() + (size_t) L.y4_1 * 64u, y4.data(), y4.size());
                }
                all = all && cmp_ref(strf("id 3: Q8_0 T=%u token %u", TM, t), got.data() + (size_t) (L.y8_t + t * yl) * 64u, y8.data(), y8.size());
                all = all && cmp_ref(strf("id 3: Q4_K T=%u token %u", TM, t), got.data() + (size_t) (L.y4_t + t * yl) * 64u, y4.data(), y4.size());
                all = all && cmp_ref(strf("id 3: ACC token %u", t), got.data() + (size_t) (L.yacc + t * yl) * 64u, ya.data(), ya.size());
            }
        };
        cs.push_back(std::move(c));
    }
    // 4 VU: every function and reduction (shadow only; fw_selftest_sim checks the references)
    if (want(4)) {
        const uint32_t M = 40, R = 5;
        const tsr_st4_layout L = tsr_st4(M, R);
        const uint32_t ts = f2u(powf(10000.0f, -2.0f / 64.0f)), mrope = f2u(1.0f), fsc = f2u(0.5f);
        const uint32_t sk = f2u(1.0f / (float) M), se = f2u(1e-6f);
        kcase c;
        c.name = "id 4 VU functions, reductions, SOP, ROPE, TOPK, DQROW";
        c.id = 4;
        c.area = AREA0 + 4 * AREA_STEP;
        c.lines = L.end;
        if (!(x.caps & TSR_CAPS_VU)) {
            c.want_ret = (int32_t) TSR_ST_RET_CAPS;
            c.want_status = TSR_LCQ_R0_STATUS_KERNEL_ERR;
        }
        std::vector<uint8_t> a = area_with_params(4, L.end, { M, R, ts, mrope, fsc, sk, se });
        uint8_t * A = a.data();
        rng r{ 0x4D4 };
        for (uint32_t row = 0; row < R; ++row) {
            rand_f32(r, A + (size_t) (L.a + row * L.L) * 64u, M, 2.0);
            rand_f32(r, A + (size_t) (L.b + row * L.L) * 64u, M, 2.0);
        }
        rand_f32(r, A + (size_t) L.c * 64u, R, 1.0);
        rand_f32(r, A + (size_t) L.sv * 64u, 16, 1.5);
        wr(A + (size_t) L.sv * 64u, 10, f2u(2.25f));
        rand_f32(r, A + (size_t) L.rope_x * 64u, 256, 1.0);
        wr(A + (size_t) L.rope_pos * 64u, 0, 7u);
        wr(A + (size_t) L.rope_pos * 64u, 1, 300u);
        for (uint32_t i = 0; i < 16u; ++i) wr(A + (size_t) L.rope_ff * 64u, i, f2u(1.0f + (float) r.unit()));
        for (uint32_t rr = 0; rr < 2u; ++rr) {
            std::vector<region> q8 = weights(r, TSR_FMT_Q8_0, 1, 256, 1, 0, nullptr), q4 = weights(r, TSR_FMT_Q4_K, 1, 256, 1, 0, nullptr);
            memcpy(A + (size_t) (L.dq8 + rr * L.dq8_beats) * 64u, q8[0].data.data(), (size_t) L.dq8_beats * 64u);
            memcpy(A + (size_t) (L.dq4 + rr * L.dq4_beats) * 64u, q4[0].data.data(), (size_t) L.dq4_beats * 64u);
        }
        c.in.push_back({ 0, c.area, a });
        cs.push_back(std::move(c));
    }
    // 5 AT: every AT op (shadow only)
    if (want(5)) {
        const uint32_t n_kv = std::min<uint32_t>(40, 16 * C), Tq = std::min<uint32_t>(4, TM);
        const tsr_st5_layout L = tsr_st5(n_kv, Tq, C);
        const uint32_t BK = (uint32_t) KV_BASE, BV = (uint32_t) KV_BASE + 0x100u;
        kcase c;
        c.name = strf("id 5 AT ops, n_kv %u T_q %u", n_kv, Tq);
        c.id = 5;
        c.area = AREA0 + 5 * AREA_STEP;
        c.lines = L.end;
        if (!(x.caps & TSR_CAPS_ATT)) {
            c.want_ret = (int32_t) TSR_ST_RET_CAPS;
            c.want_status = TSR_LCQ_R0_STATUS_KERNEL_ERR;
        }
        std::vector<uint8_t> a = area_with_params(5, L.end, { n_kv, Tq, f2u(0.125f), BK, BV });
        uint8_t * A = a.data();
        rng r{ 0x5D5 };
        rand_f32(r, A + (size_t) L.k * 64u, n_kv * 64u, 1.0);
        rand_f32(r, A + (size_t) L.v * 64u, n_kv * 64u, 1.0);
        rand_f32(r, A + (size_t) L.q * 64u, Tq * 64u, 1.0);
        for (uint32_t q = 0; q < Tq; ++q)
            for (uint32_t j = 0; j < 16u * L.n_groups; ++j) {
                const uint32_t k = r.below(4);
                const float m = k == 0 ? -INFINITY : k == 1 ? 0.0f : k == 2 ? -0.0f : (float) (r.normal() * 0.5);
                wr(A + (size_t) (L.mask + q * L.n_groups) * 64u, j, f2u(m));
            }
        c.in.push_back({ 0, c.area, a });
        for (uint32_t ch = 0; ch < C; ++ch) {   // the planes start zeroed (KVST read-modify-writes partial meta beats)
            c.in.push_back({ ch, BK, std::vector<uint8_t>(17u * 64u, 0) });
            c.in.push_back({ ch, BV, std::vector<uint8_t>(17u * 64u, 0) });
            c.out.push_back({ ch, BK, 17 });
            c.out.push_back({ ch, BV, 17 });
        }
        cs.push_back(std::move(c));
    }
    // 6 CONC: a matvec_batch pipeline on every unit (shadow only)
    if (want(6)) {
        const uint32_t N = 96, K = 512, J = 9, fmt = TSR_FMT_Q4_K;
        const tsr_st6_layout L = tsr_st6(N, K, J);
        kcase c;
        c.name = "id 6 concurrency: matvec_batch pipeline";
        c.id = 6;
        c.area = AREA0 + 6 * AREA_STEP;
        c.lines = L.end;
        if (!(x.caps & TSR_CAPS_VU)) {
            c.want_ret = (int32_t) TSR_ST_RET_CAPS;
            c.want_status = TSR_LCQ_R0_STATUS_KERNEL_ERR;
        }
        rng r{ 0x6D6 };
        std::vector<region> w = weights(r, fmt, N, K, C, W_BASE + 0x20000u, nullptr);
        std::vector<uint8_t> a = area_with_params(6, L.end, { N, K, (uint32_t) W_BASE + 0x20000u, 1u, J });
        rand_f32(r, a.data() + (size_t) L.x * 64u, J * K, 1.0);
        std::fill(a.begin() + (size_t) L.y * 64u, a.end(), (uint8_t) 0x5A);
        c.in.push_back({ 0, c.area, a });
        c.in.insert(c.in.end(), w.begin(), w.end());
        cs.push_back(std::move(c));
    }
    // 7 ERRORS: DRAM_RANGE, ILLEGAL_OP, BAD_DEP, DEP_FAILED chain, QABORT + QCLEAR (device-checked)
    if (want(7)) {
        const tsr_st7_layout L = tsr_st7();
        kcase c;
        c.name = "id 7 errors and recovery";
        c.id = 7;
        c.area = AREA0 + 7 * AREA_STEP;
        c.lines = L.end;
        rng r{ 0x7D7 };
        std::vector<region> w = weights(r, TSR_FMT_Q8_0, 16, 32, C, W_BASE + 0x30000u, nullptr);
        std::vector<uint8_t> a = area_with_params(7, L.end, { 0xFFFFFFF0u, (uint32_t) W_BASE + 0x30000u });
        r.fill(a.data() + (size_t) L.in * 64u, 128);
        std::fill(a.begin() + (size_t) L.out * 64u, a.end(), (uint8_t) 0x5A);
        c.in.push_back({ 0, c.area, a });
        c.in.insert(c.in.end(), w.begin(), w.end());
        const uint64_t area = c.area;
        c.ref = [=](tsr_tgt * d, const tsr_tgt_run &) {
            const std::vector<uint8_t> got = read_dev(d, 0, area, L.end);
            cmp_ref("id 7: the recovery round trip after QABORT + QCLEAR = its input", got.data() + (size_t) L.out * 64u,
                    a.data() + (size_t) L.in * 64u, 128);
        };
        cs.push_back(std::move(c));
    }
    // 8 SPMWIN: byte / half / word / misaligned stores into the SPM window, ordering before a push
    if (want(8)) {
        const tsr_st8_layout L = tsr_st8();
        kcase c;
        c.name = "id 8 SPM window stores and store-before-push ordering";
        c.id = 8;
        c.area = AREA0 + 8 * AREA_STEP;
        c.lines = L.end;
        c.seed = 0x8D8A11u;
        std::vector<uint8_t> a = area_with_params(8, L.end, { 1000u });
        std::fill(a.begin() + 64, a.end(), (uint8_t) 0x5A);
        c.in.push_back({ 0, c.area, a });
        const uint64_t area = c.area;
        const uint32_t seed = c.seed;
        c.ref = [=](tsr_tgt * d, const tsr_tgt_run &) {
            uint8_t want_b[128];
            tsr_st8_pattern(seed, want_b);
            const std::vector<uint8_t> got = read_dev(d, 0, area, L.end);
            cmp_ref("id 8: the stored pattern = tsr_st8_pattern", got.data() + (size_t) L.out * 64u, want_b, 128);
        };
        cs.push_back(std::move(c));
    }
    // 9 LOG: two FMT records and a kernel event
    if (want(9)) {
        kcase c;
        c.name = "id 9 log records and kernel event";
        c.id = 9;
        c.area = AREA0 + 9 * AREA_STEP;
        c.lines = 1;
        c.seed = 0x9D9C0DEu;
        c.in.push_back({ 0, c.area, area_with_params(9, 1, {}) });
        const uint32_t seed = c.seed;
        tsr_elf * elf = x.elf;
        c.ref = [=](tsr_tgt *, const tsr_tgt_run & run) {
            int ev = 0;
            for (uint32_t i = 0; i < run.n_ev && i < TSR_TGT_MAX_EVENTS; ++i) {
                const uint32_t * e = run.ev[i];
                if (TSR_GET(e[0], TSR_LCQ_R0_TYPE) != TSR_ST9_EVENT) continue;
                ev++;
                result(TSR_GET(e[0], TSR_LCQ_R0_STATUS) == (seed & 0xFFu) && TSR_GET(e[0], TSR_LCQ_R0_ERR_CODE) == ((seed >> 8) & 0xFFu) &&
                           e[1] == run.seq && e[2] == seed && e[3] == ~seed,
                       "id 9: kernel event 0xC9 (info, SEQ, seed, ~seed)", strf("0x%08x 0x%08x 0x%08x 0x%08x", e[0], e[1], e[2], e[3]));
            }
            result(ev == 1, "id 9: exactly one kernel event 0xC9 delivered to the event callback", strf("%d", ev));
            // The two FMT records of SOURCE kernel, decoded with the ELF's .tsr_logfmt.
            int found = 0;
            for (size_t i = 0; i < run.n_log;) {
                const uint32_t hd = run.log[i], n = TSR_GET(hd, TSR_LOG_HDR_N);
                if (TSR_GET(hd, TSR_LOG_HDR_SYNC) != TSR_LOG_HDR_SYNC_MAGIC || i + 1 + n > run.n_log) {
                    result(false, "id 9: log framing", strf("word %zu 0x%08x", i, hd));
                    break;
                }
                const uint32_t * pl = run.log + i + 1;
                if (TSR_GET(hd, TSR_LOG_HDR_SOURCE) == TSR_LOG_HDR_SOURCE_KERNEL && TSR_GET(hd, TSR_LOG_HDR_TYPE) == TSR_LOG_HDR_TYPE_FMT) {
                    const uint32_t ts = TSR_GET(hd, TSR_LOG_HDR_TS) ? 1u : 0u;
                    const char * f = n > ts && elf ? tsr_elf_logfmt(elf, pl[ts]) : nullptr;
                    result(TSR_GET(hd, TSR_LOG_HDR_LSEQ) == (run.seq & 0xFFu), "id 9: LSEQ = the launch's SEQ[7:0]");
                    if (found == 0)
                        result(f && !strcmp(f, "selftest 9: seed %x, id %u") && TSR_GET(hd, TSR_LOG_HDR_LEVEL) == TSR_LOG_HDR_LEVEL_INFO &&
                                   n == ts + 3u && pl[ts + 1] == seed && pl[ts + 2] == 9u,
                               "id 9: first record INFO \"selftest 9: seed %x, id %u\" (seed, 9)", f ? f : "?");
                    else if (found == 1)
                        result(f && !strcmp(f, "selftest 9: %u %x") && TSR_GET(hd, TSR_LOG_HDR_LEVEL) == TSR_LOG_HDR_LEVEL_WARN &&
                                   n == ts + 3u && pl[ts + 1] == ~seed && pl[ts + 2] == seed + 1u,
                               "id 9: second record WARN \"selftest 9: %u %x\" (~seed, seed + 1)", f ? f : "?");
                    ++found;
                }
                i += 1u + n;
            }
            result(found == 2, "id 9: two kernel FMT records drained", strf("%d", found));
        };
        cs.push_back(std::move(c));
    }
    // 10 CREDITS: every queue filled under QHOLD, each credit 0 (device-checked)
    if (want(10)) {
        const tsr_st10_layout L = tsr_st10();
        kcase c;
        c.name = "id 10 credits under QHOLD";
        c.id = 10;
        c.area = AREA0 + 10 * AREA_STEP;
        c.lines = L.end;
        rng r{ 0xAD10 };
        std::vector<region> w = weights(r, TSR_FMT_Q8_0, 16, 32, C, W_BASE + 0x40000u, nullptr);
        std::vector<uint8_t> a = area_with_params(10, L.end, { (uint32_t) W_BASE + 0x40000u });
        rand_f32(r, a.data() + 64, 32, 1.0);
        c.in.push_back({ 0, c.area, a });
        c.in.insert(c.in.end(), w.begin(), w.end());
        cs.push_back(std::move(c));
    }
    // 11 GDN: four heads of the fused Gated DeltaNet state update (needs CAPS.GDN; shadow and tsr_ref_gdn)
    if (want(11)) {
        const uint32_t m = 128, rows = 20, H = 4;
        const tsr_st11_layout L = tsr_st11(m, rows, H);
        kcase c;
        c.name = strf("id 11 GDN, %u heads of %u x %u", H, rows, m);
        c.id = 11;
        c.area = AREA0 + 11 * AREA_STEP;
        c.lines = L.end;
        if (!(x.caps & TSR_CAPS_GDN) || !(x.caps & TSR_CAPS_VU)) {
            c.want_ret = (int32_t) TSR_ST_RET_CAPS;
            c.want_status = TSR_LCQ_R0_STATUS_KERNEL_ERR;
        }
        const uint32_t al[4] = { f2u(0.95f), f2u(0.5f), f2u(0.99f), f2u(1.0f) }, be[4] = { f2u(0.5f), f2u(0.25f), f2u(0.9f), f2u(1.0f) };
        std::vector<uint32_t> p = { m, rows, H };
        for (uint32_t h = 0; h < H; ++h) {
            p.push_back(al[h]);
            p.push_back(be[h]);
        }
        std::vector<uint8_t> a = area_with_params(11, L.end, p);
        rng r{ 0x6D11 };
        for (uint32_t h = 0; h < H; ++h) {
            rand_f32(r, a.data() + (size_t) tsr_st11_k(&L, h) * 64u, m, 0.25);
            rand_f32(r, a.data() + (size_t) tsr_st11_q(&L, h) * 64u, m, 0.25);
            rand_f32(r, a.data() + (size_t) tsr_st11_v(&L, h) * 64u, rows, 1.0);
            rand_f32(r, a.data() + (size_t) tsr_st11_s(&L, h) * 64u, rows * m, 0.5);
        }
        std::fill(a.begin() + (size_t) L.in_end * 64u, a.end(), (uint8_t) 0x5A);
        c.in.push_back({ 0, c.area, a });
        const uint64_t area = c.area;
        if ((x.caps & TSR_CAPS_GDN) && (x.caps & TSR_CAPS_VU)) c.ref = [=](tsr_tgt * d, const tsr_tgt_run &) {   // else RET CAPS: no outputs
            const std::vector<uint8_t> got = read_dev(d, 0, area, L.end);
            std::vector<uint32_t> k(m), q(m), v(rows), S((size_t) rows * m), Sp((size_t) rows * m), o(16u * L.lv);
            for (uint32_t h = 0; h < H; ++h) {
                for (uint32_t j = 0; j < m; ++j) {
                    k[j] = rd(a.data() + (size_t) tsr_st11_k(&L, h) * 64u, j);
                    q[j] = rd(a.data() + (size_t) tsr_st11_q(&L, h) * 64u, j);
                }
                for (uint32_t i = 0; i < rows; ++i) v[i] = rd(a.data() + (size_t) tsr_st11_v(&L, h) * 64u, i);
                for (size_t i = 0; i < S.size(); ++i) S[i] = rd(a.data() + (size_t) tsr_st11_s(&L, h) * 64u, (uint32_t) i);
                tsr_ref_gdn(m, rows, al[h], be[h], k.data(), q.data(), v.data(), S.data(), Sp.data(), o.data());
                cmp_ref(strf("id 11: head %u S' = tsr_ref_gdn", h), got.data() + (size_t) tsr_st11_sp(&L, h) * 64u,
                        (const uint8_t *) Sp.data(), (size_t) L.tile * 64u);
                cmp_ref(strf("id 11: head %u o = tsr_ref_gdn", h), got.data() + (size_t) tsr_st11_o(&L, h) * 64u,
                        (const uint8_t *) o.data(), (size_t) L.lv * 64u);
            }
        };
        cs.push_back(std::move(c));
    }
    // Misuses: unknown id, wrong magic, an area too small (KERNEL_ERR); a device-side check failing
    // (id 7 with an in-range "bad" beat: RET 1, OK); id 8 with its SPM lines in the argument range.
    if (want(12)) {
        const uint64_t area = AREA0 + 12 * AREA_STEP;
        struct mu { const char * name; uint32_t id, lines; std::vector<uint8_t> a; int32_t ret; uint32_t st; std::vector<region> extra; };
        std::vector<mu> m;
        m.push_back({ "misuse: unknown id 12", 12, 1, area_with_params(12, 1, {}), (int32_t) TSR_ST_RET_BAD_ID,
                      TSR_LCQ_R0_STATUS_KERNEL_ERR, {} });
        m.push_back({ "misuse: parameter line of id 3 for id 2", 2, 100, area_with_params(3, 100, {}), (int32_t) TSR_ST_RET_BAD_PARAMS,
                      TSR_LCQ_R0_STATUS_KERNEL_ERR, {} });
        m.push_back({ "misuse: id 2 with too small an area", 2, 10, area_with_params(2, 10, { 1024, TM }), (int32_t) TSR_ST_RET_AREA,
                      TSR_LCQ_R0_STATUS_KERNEL_ERR, {} });
        m.push_back({ "misuse: id 7 without a DRAM_RANGE (device-side check fails, RET 1)", 7, tsr_st7().end,
                      area_with_params(7, tsr_st7().end, { (uint32_t) AREA0, (uint32_t) W_BASE + 0x30000u }), 1,
                      TSR_LCQ_R0_STATUS_OK, {} });
        {
            rng r{ 0x7D7 };
            m.back().extra = weights(r, TSR_FMT_Q8_0, 16, 32, C, W_BASE + 0x30000u, nullptr);
        }
        m.push_back({ "misuse: id 8 with its SPM lines in the argument range", 8, tsr_st8().end, area_with_params(8, tsr_st8().end, { 65530u }),
                      (int32_t) TSR_ST_RET_BAD_PARAMS, TSR_LCQ_R0_STATUS_KERNEL_ERR, {} });
        for (mu & u : m) {
            kcase c;
            c.name = u.name;
            c.id = u.id;
            c.area = area;
            c.lines = u.lines;
            c.want_ret = u.ret;
            c.want_status = u.st;
            c.in.push_back({ 0, area, u.a });
            c.in.insert(c.in.end(), u.extra.begin(), u.extra.end());
            cs.push_back(std::move(c));
        }
    }
    return cs;
}

void kernel_tests(ctx & x, const std::vector<uint32_t> & ids) {
    printf("== selftest kernel (SW §7.6)%s\n", x.sh ? ", with the in-process shadow simulator" : ", without a shadow");
    {   // PING on both: the protocol and the sequence bookkeeping of the shadow
        const uint32_t args[7] = { 0xC0FFEE09u, 0, 0, 0, 0, 0, 0 };
        tsr_tgt_run a, b;
        tsr_tgt_launch(x.dev, TSR_DESC_TYPE_PING, 0, 0, args, &a);
        if (x.sh) {
            tsr_tgt_launch(x.sh, TSR_DESC_TYPE_PING, 0, 0, args, &b);
            char m[512];
            result(tsr_tgt_compare(&a, &b, m, sizeof(m)) == 0, "PING: device = shadow", m);
            tsr_tgt_run_free(&b);
        }
        result(a.rc == 0 && TSR_GET(a.rec[0], TSR_LCQ_R0_TYPE) == TSR_LCQ_R0_TYPE_PONG && a.rec[2] == args[0], "PING -> PONG");
        tsr_tgt_run_free(&a);
    }
    std::vector<kcase> cs = make_cases(x, ids);
    for (kcase & c : cs) run_case(x, c);
    const uint32_t st = tsr_tgt_csr(x.dev, TSR_CSR_STATUS);
    result(!(st & TSR_STATUS_QERR), "no unit left in error state", strf("STATUS 0x%08x", st));
}

// ---------------------------------------------------------------------------
// 6. Trace replays
// ---------------------------------------------------------------------------
void find_traces(const fs::path & dir, std::vector<std::string> & out) {
    std::error_code ec;
    if (fs::exists(dir / "meta.json", ec)) {
        out.push_back(dir.string());
        return;
    }
    std::vector<fs::path> subs;
    for (const auto & e : fs::directory_iterator(dir, ec)) {
        if (e.is_directory(ec) && e.path().filename() != "images") subs.push_back(e.path());
    }
    std::sort(subs.begin(), subs.end());
    for (const auto & s : subs) find_traces(s, out);
}

void traces(ctx & x, const std::vector<std::pair<std::string, uint32_t>> & dirs, size_t max_traces, const char * image) {
    printf("== trace replays (%zu director%s)\n", dirs.size(), dirs.size() == 1 ? "y" : "ies");
    size_t tp = 0, tf = 0, ts = 0;
    for (const auto & d : dirs) {
        std::vector<std::string> list;
        find_traces(d.first, list);
        size_t pass = 0, failed = 0, skip = 0;
        double wall = 0;
        for (const std::string & dir : list) {
            if (pass + failed >= max_traces) break;
            tsr_trace2 * t = nullptr;
            char err[512];
            if (tsr_trace2_read(dir.c_str(), &t, err, sizeof(err)) != 0) {
                result(false, dir, strf("unreadable trace: %s", err));
                failed++;
                continue;
            }
            tsr_devrun_opts o;
            memset(&o, 0, sizeof(o));
            o.image = image;
            o.replay_flags = d.second;
            o.verbose = g_verbose;
            tsr_devrun_result r;
            const int rc = tsr_devrun_trace(x.dev, x.sh, t, &o, &r);
            tsr_trace2_free(t);
            if (rc == 3) {
                skip++;
                if (g_verbose) printf("  skip  %s (%s)\n", dir.c_str(), r.msg);
                continue;
            }
            wall += r.wall_ns / 1e9;
            if (rc == 0) {
                pass++;
                if (g_verbose) printf("  ok    %s: %s\n", dir.c_str(), r.msg);
            } else {
                failed++;
                result(false, dir, r.msg);
            }
        }
        printf("  %s%s: %zu passed, %zu failed, %zu skipped (configuration), device launches %.1f ms\n", d.first.c_str(),
               d.second ? " (SERIAL)" : "", pass, failed, skip, wall * 1e3);
        // A set made for another configuration (channels, SPM, a held deeper queue) is skipped whole:
        // reported, not a failure; a directory without any trace is one.
        result(failed == 0 && pass + skip > 0, strf("traces under %s replay through the device%s", d.first.c_str(), x.sh ? " = shadow" : ""),
               pass + skip == 0 ? "no trace there" : strf("%zu failed", failed));
        tp += pass;
        tf += failed;
        ts += skip;
    }
    // Kind launch traces boot their own image: back to the tool's.
    char err[512];
    for (tsr_tgt * g : { x.dev, x.sh })
        if (g && tsr_tgt_boot(g, image, err, sizeof(err)) != 0) result(false, "re-boot after the replays", err);
    printf("  replays: %zu passed, %zu failed, %zu skipped\n", tp, tf, ts);
    result(tp > 0, "at least one trace replayed", strf("%zu skipped", ts));
}

// ---------------------------------------------------------------------------
// 7. Debug smoke: DMI halt / step / resume (DB §3.4-§3.5)
// ---------------------------------------------------------------------------
struct dmi {
    tsr_transport * t;
    uint32_t r(uint32_t reg) { return tsr_transport_csr_read(t, TSR_CSR_DMI(reg)); }
    void w(uint32_t reg, uint32_t v) { tsr_transport_csr_write(t, TSR_CSR_DMI(reg), v); }
    // Polls dmstatus for bit; STATUS reads between polls let the simulator kinds run (EX §10.2).
    bool poll(uint32_t bit) {
        for (int i = 0; i < 20000; ++i) {
            if (r(TSR_DMI_DMSTATUS) & bit) return true;
            (void) tsr_transport_csr_read(t, TSR_CSR_STATUS);
            if (i > 100) {
                const struct timespec ts = { 0, 10000 };
                nanosleep(&ts, nullptr);
            }
        }
        return false;
    }
    bool cmd(uint32_t command, uint32_t * err) {
        w(TSR_DMI_COMMAND, command);
        for (int i = 0; i < 10000; ++i) {
            const uint32_t acs = r(TSR_DMI_ABSTRACTCS);
            if (!(acs & (1u << 12))) {
                *err = (acs >> 8) & 7u;
                if (*err) w(TSR_DMI_ABSTRACTCS, 0x700u);
                return *err == 0;
            }
            (void) tsr_transport_csr_read(t, TSR_CSR_STATUS);
        }
        *err = 0xFF;
        return false;
    }
    bool csr_read(uint32_t csr, uint32_t * v) {
        uint32_t e;
        if (!cmd(0x00220000u | csr, &e)) return false;
        *v = r(TSR_DMI_DATA0);
        return true;
    }
    bool csr_write(uint32_t csr, uint32_t v) {
        uint32_t e;
        w(TSR_DMI_DATA0, v);
        return cmd(0x00230000u | csr, &e);
    }
};

void debug_smoke(tsr_transport * t) {
    printf("== debug smoke (DB §3.4-§3.5: DMI halt, step, resume)\n");
    const uint32_t ctl = tsr_transport_csr_read(t, TSR_CSR_CTL_ID);
    if (!(ctl & TSR_CTL_ID_DM)) {
        result(false, "CTL_ID.DM: a debug module reachable through the DMI window", strf("CTL_ID 0x%08x", ctl));
        return;
    }
    constexpr uint32_t ALLHALTED = 1u << 9, ALLRUNNING = 1u << 11, ALLRESUMEACK = 1u << 17;
    constexpr uint32_t DCSR = 0x7B0u, DPC = 0x7B1u;
    const uint32_t dbg_cfg = tsr_transport_csr_read(t, TSR_CSR_DBG_CFG);
    tsr_transport_csr_write(t, TSR_CSR_DBG_CFG, dbg_cfg | TSR_DBG_CFG_DMI_EN);
    dmi d{ t };
    d.w(TSR_DMI_DMCONTROL, 0x00000001u);   // dmactive
    const uint32_t s0 = d.r(TSR_DMI_DMSTATUS);
    result((s0 & 0xFu) != 0 && (s0 & ALLRUNNING), "dmstatus: a debug module, hart running", strf("0x%08x", s0));
    d.w(TSR_DMI_DMCONTROL, 0x80000001u);   // haltreq
    const bool halted = d.poll(ALLHALTED);
    d.w(TSR_DMI_DMCONTROL, 0x00000001u);
    const uint32_t st = tsr_transport_csr_read(t, TSR_CSR_STATUS);
    result(halted && (st & TSR_STATUS_RV_HALTED), "halt: allhalted, STATUS.RV_HALTED", strf("dmstatus 0x%08x STATUS 0x%08x",
                                                                                            d.r(TSR_DMI_DMSTATUS), st));
    if (!halted) {
        tsr_transport_csr_write(t, TSR_CSR_DBG_CFG, dbg_cfg);
        return;
    }
    uint32_t dpc0 = 0, dcsr = 0, dpc1 = 0, dcsr1 = 0;
    const bool rd0 = d.csr_read(DPC, &dpc0) && d.csr_read(DCSR, &dcsr);
    result(rd0 && ((dcsr >> 6) & 7u) == 3u, "dpc / dcsr read (Access Register): cause haltreq",
           strf("dpc 0x%08x, dcsr 0x%08x", dpc0, dcsr));
    // single step: dcsr.step, resume, halted again with cause step
    const bool w1 = d.csr_write(DCSR, dcsr | 4u);
    d.w(TSR_DMI_DMCONTROL, 0x40000001u);   // resumereq
    const bool ack = d.poll(ALLRESUMEACK);
    d.w(TSR_DMI_DMCONTROL, 0x00000001u);
    const bool halted2 = d.poll(ALLHALTED);
    const bool rd1 = halted2 && d.csr_read(DPC, &dpc1) && d.csr_read(DCSR, &dcsr1);
    result(w1 && ack && rd1 && ((dcsr1 >> 6) & 7u) == 4u, "single step: resume ack, halted again with cause step",
           strf("dpc 0x%08x -> 0x%08x, dcsr 0x%08x", dpc0, dpc1, dcsr1));
    // resume for good
    const bool w2 = d.csr_write(DCSR, dcsr1 & ~4u);
    d.w(TSR_DMI_DMCONTROL, 0x40000001u);
    const bool ack2 = d.poll(ALLRESUMEACK);
    d.w(TSR_DMI_DMCONTROL, 0x00000001u);
    const uint32_t st2 = tsr_transport_csr_read(t, TSR_CSR_STATUS);
    result(w2 && ack2 && !(st2 & TSR_STATUS_RV_HALTED), "resume: allresumeack, STATUS.RV_HALTED clear", strf("STATUS 0x%08x", st2));
    printf("  halted at dpc 0x%08x (dcsr 0x%08x), stepped to 0x%08x (dcsr 0x%08x), resumed\n", dpc0, dcsr, dpc1, dcsr1);
    d.w(TSR_DMI_DMCONTROL, 0x00000000u);
    tsr_transport_csr_write(t, TSR_CSR_DBG_CFG, dbg_cfg);
    // the device still works
    tsr_launch l;
    memset(&l, 0, sizeof(l));
    l.type = TSR_DESC_TYPE_PING;
    l.arg[0] = 0xDB600001u;
    tsr_record rec;
    const int rc = tsr_transport_call(t, &l, TSR_TIMEOUT_DEFAULT, &rec);
    result(rc == 0 && rec.w[2] == l.arg[0], "PING after the debug session", strf("rc %d r2 0x%08x", rc, rec.w[2]));
}

// ---------------------------------------------------------------------------
// 8. Log and final status
// ---------------------------------------------------------------------------
void log_and_status(tsr_transport * t, const tsr_image * img, std::vector<uint32_t> & log_all, const char * log_out) {
    printf("== log and status\n");
    std::vector<uint32_t> w(65536);
    size_t n = 0;
    if (tsr_transport_log_drain(t, w.data(), w.size(), &n) == 0) log_all.insert(log_all.end(), w.begin(), w.begin() + (long) n);
    tsr::log_decoder dec(img);
    size_t shown = 0;
    std::vector<std::string> lines;
    dec.feed(log_all.data(), log_all.size(), [&](const std::string & s) { lines.push_back(s); });
    for (const std::string & s : lines) {
        const bool important = s.find("[tsr FATAL") == 0 || s.find("[tsr ERROR") == 0 || s.find("firmware log bug") != std::string::npos;
        if (g_verbose || (important && shown < 20)) {
            printf("  %s\n", s.c_str());
            shown++;
        }
    }
    printf("  %zu log words drained during the run: %" PRIu64 " records, %" PRIu64 " malformed\n", log_all.size(), dec.records, dec.bad);
    result(dec.bad == 0 && dec.pending_words() == 0, "every drained log word decodes (DB §6.1)",
           strf("%" PRIu64 " bad, %zu pending", dec.bad, dec.pending_words()));
    const uint32_t dropped = tsr_transport_csr_read(t, TSR_CSR_LOG_DROPPED), st = tsr_transport_csr_read(t, TSR_CSR_STATUS);
    result(dropped == 0 && !(st & TSR_STATUS_LOG_DROP) && dec.dropped_reported == 0, "no log record dropped",
           strf("LOG_DROPPED %u, STATUS 0x%08x, %" PRIu64 " reported dropped", dropped, st, dec.dropped_reported));
    const uint32_t bad_sticky = TSR_STATUS_LOAD_TIMEOUT | TSR_STATUS_DMI_TIMEOUT | TSR_STATUS_BRIDGE_TIMEOUT | TSR_STATUS_LQ_OVERFLOW |
                                TSR_STATUS_RING_RD_BAD | TSR_STATUS_UNIT_HUNG | TSR_STATUS_SR_RV_TIMEOUT | TSR_STATUS_PLAT_ERR;
    result(!(st & bad_sticky), "STATUS: no timeout, overflow, ring, hang or platform error bit", strf("0x%08x", st));
    const tsr::recovery_log rl = tsr::transport_recovery_log(t);
    result(rl.timeouts == 0 && rl.traps == 0 && rl.lost == 0 && !rl.failed, "transport: no timeout, trap or lost launch",
           strf("timeouts %u, traps %u, lost %u, failed %d%s%s", rl.timeouts, rl.traps, rl.lost, (int) rl.failed,
                rl.last_trap.empty() ? "" : "; last trap: ", rl.last_trap.c_str()));
    tsr_transport_stats ss;
    tsr_transport_get_stats(t, &ss);
    printf("  transport: %" PRIu64 " launches (%.3f s waiting), %" PRIu64 " CSR reads, %" PRIu64 " CSR writes, %.1f MiB written, "
           "%.1f MiB read, %" PRIu64 " log words\n", ss.launches, ss.launch_wall_ns / 1e9, ss.csr_reads, ss.csr_writes,
           ss.bytes_written / 1048576.0, ss.bytes_read / 1048576.0, ss.log_words);
    if (log_out) {
        FILE * f = fopen(log_out, "wb");
        bool ok = f != nullptr;
        for (uint32_t v : log_all) {
            const uint8_t b[4] = { (uint8_t) v, (uint8_t) (v >> 8), (uint8_t) (v >> 16), (uint8_t) (v >> 24) };
            ok = ok && fwrite(b, 1, 4, f) == 4;
        }
        if (f) ok = fclose(f) == 0 && ok;
        result(ok, strf("log words written to %s", log_out));
    }
    if (tsr::kdev * kd = tsr::transport_kdev(t); kd != nullptr && kd->has_adapter()) {
        // The protocol layer cleared ADAPTER_STATUS at open: the register shows what happened since.
        uint32_t as = 0;
        const bool ok = kd->reg_read(tsr::KREG_ADAPTER_STATUS, &as) == 0;
        const tsr::adapter_log al = tsr::transport_adapter_log(t);
        result(ok && as == 0 && al.plat_errs == 0, "ADAPTER_STATUS clean at the end, no PLAT_ERR during the run",
               ok ? strf("%s; %u PLAT_ERR event(s)%s%s", tsr::adapter_status_str(as).c_str(), al.plat_errs, al.plat_errs ? ", last " : "",
                         al.plat_errs ? tsr::adapter_status_str(al.last).c_str() : "")
                  : std::string("unreadable"));
    }
}

void usage() {
    fprintf(stderr, "usage: tsr-selftest [--device KIND] [--image ELF] [--mem-mib N] [--no-mem] [--no-kernel] [--ids LIST]\n"
                    "                    [--no-shadow] [--rate-n N] [--traces DIR]... [--traces-serial DIR]... [--max-traces N]\n"
                    "                    [--trw MIB] [--debug] [--log-out FILE] [--seed N] [-v]\n"
                    "       tsr-selftest --version\n");
}

} // namespace

int main(int argc, char ** argv) {
    if (argc == 2 && strcmp(argv[1], "--version") == 0) {
        const uint32_t v = tsr_basis_version();
        printf("tsr-selftest (Tesseract Basis %u.%u.%u, Basis ABI %u; Tesseract Core capability %u.x)\n", (v >> 16) & 0xFFu,
               (v >> 8) & 0xFFu, v & 0xFFu, TSR_SDK_ABI, TSR_ISA_MAJOR);
        return 0;
    }
    const char * dev = getenv("TSR_DEVICE");
    if (dev == nullptr || dev[0] == '\0') dev = "xrt";
    const char * image_path = nullptr, * log_out = nullptr;
    uint64_t mem_mib = 64, seed = 1;
    uint32_t rate_n = 1000, trw_mib = 0;
    size_t max_traces = SIZE_MAX;
    bool do_mem = true, do_kernel = true, do_shadow = true, do_debug = false;
    std::vector<uint32_t> ids;
    std::vector<std::pair<std::string, uint32_t>> trace_dirs;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&](const char * what) -> const char * {
            if (i + 1 >= argc) {
                fprintf(stderr, "tsr-selftest: %s needs a value\n", what);
                exit(2);
            }
            return argv[++i];
        };
        if (a == "--device") dev = val("--device");
        else if (a == "--image") image_path = val("--image");
        else if (a == "--mem-mib") mem_mib = strtoull(val("--mem-mib"), nullptr, 10);
        else if (a == "--no-mem") do_mem = false;
        else if (a == "--no-kernel") do_kernel = false;
        else if (a == "--no-shadow") do_shadow = false;
        else if (a == "--ids") {
            std::string s = val("--ids");
            for (size_t p = 0; p < s.size();) {
                const size_t e = s.find(',', p);
                ids.push_back((uint32_t) strtoul(s.substr(p, e - p).c_str(), nullptr, 10));
                p = e == std::string::npos ? s.size() : e + 1;
            }
        }
        else if (a == "--rate-n") rate_n = (uint32_t) strtoul(val("--rate-n"), nullptr, 10);
        else if (a == "--traces") trace_dirs.push_back({ val("--traces"), 0u });
        else if (a == "--traces-serial") trace_dirs.push_back({ val("--traces-serial"), (uint32_t) TSR_REPLAY_F_SERIAL });
        else if (a == "--max-traces") max_traces = strtoull(val("--max-traces"), nullptr, 10);
        else if (a == "--trw") trw_mib = (uint32_t) strtoul(val("--trw"), nullptr, 10);
        else if (a == "--debug") do_debug = true;
        else if (a == "--log-out") log_out = val("--log-out");
        else if (a == "--seed") seed = strtoull(val("--seed"), nullptr, 10);
        else if (a == "-v") g_verbose = true;
        else {
            usage();
            return 2;
        }
    }
    tsr_transport * t = tsr_transport_open(dev);
    if (t == nullptr) {
        fprintf(stderr, "tsr-selftest: cannot open the \"%s\" transport (see the messages above)\n", dev);
        return 2;
    }
    rng r{ 0x9E3779B97F4A7C15ull ^ (seed * 0x100000001B3ull) };
    identity(t);
    if (do_mem) memory(t, mem_mib, r);

    tsr_image * img = nullptr;
    const int irc = image_path ? tsr_image_open(image_path, &img) : tsr_image_open_default(&img);
    std::vector<uint32_t> log_all;
    if (irc != 0 || img == nullptr) {
        result(false, "firmware image", strf("%s: %s", image_path ? image_path : "$TSR_IMAGE / the build's tsr-fw.elf",
                                             tsr_transport_strerror(irc)));
    } else if (boot(t, img)) {
        {   // the boot's log records (HELLO time) are part of the run
            std::vector<uint32_t> w(65536);
            size_t n = 0;
            if (tsr_transport_log_drain(t, w.data(), w.size(), &n) == 0) log_all.insert(log_all.end(), w.begin(), w.begin() + (long) n);
        }
        ping(t, rate_n);
        tsr_tgt * dt = nullptr, * sh = nullptr;
        if (do_kernel || !trace_dirs.empty()) {
            dt = tsr_tgt_transport(t);
            char err[512];
            tsr_tgt_boot(dt, tsr_image_path(img), err, sizeof(err));   // the image is running: no reboot
            if (do_shadow) {
                tsr_tgt_cfg sc;
                tsr_tgt_cfg_of_transport(t, &sc);
                sh = tsr_tgt_shadow(&sc, 0);
                if (sh == nullptr || tsr_tgt_boot(sh, tsr_image_path(img), err, sizeof(err)) != 0) {
                    result(false, "shadow simulator with the device's configuration and image", sh ? err : "create failed");
                    tsr_tgt_free(sh);
                    sh = nullptr;
                } else {
                    tsr_tgt_csr_write(sh, TSR_CSR_LOG_CFG, tsr_transport_csr_read(t, TSR_CSR_LOG_CFG));   // the same records
                    const tsr_config * c = tsr_tgt_config(sh);
                    printf("  shadow: NUM_CH %u T_MAX %u CAPS 0x%x ch_bytes %.2f GiB, image %016" PRIx64 "\n", c->num_ch, c->t_max, c->caps,
                           (double) c->ch_bytes / (1ull << 30), tsr_tgt_image_id(sh));
                }
            }
        }
        tsr::dbg::trw_ring trw_r;
        tsr_perf_summary trw_p0;
        bool trw_on = false;
        if (trw_mib != 0) {   // the trace writer runs through the kernel tests and the replays below
            std::string why;
            trw_r = tsr::dbg::trw_default_ring(trw_mib);
            trw_on = tsr::dbg::trw_start(t, trw_r, &why) == 0;
            result(trw_on, strf("trace writer started (ring %u MiB at ch%u 0x%" PRIx64 ")", trw_mib, trw_r.ch, (uint64_t) trw_r.off), why);
            tsr_transport_perf_summary(t, &trw_p0);
        }
        if (do_kernel) {
            ctx x;
            x.dev = dt;
            x.sh = sh;
            x.kidx = tsr_image_kernel(img, "selftest");
            char err[256];
            if (tsr_elf_read(tsr_image_path(img), &x.elf, err, sizeof(err)) != 0) x.elf = nullptr;
            const tsr_config * c = tsr_transport_config(t);
            x.C = c->num_ch;
            x.TM = c->t_max;
            x.caps = c->caps;
            x.log_all = &log_all;
            if (x.kidx < 0) result(false, "the image has a selftest kernel");
            else kernel_tests(x, ids);
            if (x.elf) tsr_elf_free(x.elf);
        }
        if (!trace_dirs.empty()) {
            ctx x;
            x.dev = dt;
            x.sh = sh;
            traces(x, trace_dirs, max_traces, tsr_image_path(img));
        }
        if (trw_on) {
            printf("== trace writer\n");
            tsr_perf_summary p1;
            tsr_transport_perf_summary(t, &p1);
            uint64_t pd[64];
            for (int k = 0; k < 64; ++k) pd[k] = p1.perf[k] - trw_p0.perf[k];
            std::vector<tsr::dbg::trw_rec> recs;
            uint64_t lost = 0;
            std::string why;
            const int drc = tsr::dbg::trw_drain(t, trw_r, &recs, &lost, &why);
            tsr::dbg::trw_stop(t);
            result(drc == 0, "trace writer ring drained", why);
            uint32_t ops = 0, marks = 0;
            for (const tsr::dbg::trw_rec & q : recs) {
                ops += q.kind == TSR_TRW_W0_KIND_OP;
                marks += q.kind == TSR_TRW_W0_KIND_MARK;
            }
            const uint64_t drops = pd[TSR_PERF_TRW_DROPS] + lost;
            printf("  %zu records (%u ops, %u markers), %" PRIu64 " dropped or overwritten; PERF TRW_RECORDS %" PRIu64 ", TRW_DROPS %" PRIu64 "\n",
                   recs.size(), ops, marks, drops, pd[TSR_PERF_TRW_RECORDS], pd[TSR_PERF_TRW_DROPS]);
            result(!recs.empty(), "trace writer wrote records", strf("%zu records", recs.size()));
            const tsr::dbg::trw_check_result cr = tsr::dbg::trw_check(recs, pd, tsr_transport_config(t)->num_ch, drops);
            for (const std::string & l : cr.lines) printf("  check: %s\n", l.c_str());
            result(cr.ok, "DB §8.7 trace writer self-checks against the PERF deltas");
        }
        tsr_tgt_free(sh);
        tsr_tgt_free(dt);
        if (do_debug) debug_smoke(t);
        log_and_status(t, img, log_all, log_out);
    }
    if (img) tsr_image_close(img);
    tsr_transport_close(t);
    printf("tsr-selftest (%s): %d checks passed, %d failed -> %s\n", dev, g_pass, g_fail, g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
