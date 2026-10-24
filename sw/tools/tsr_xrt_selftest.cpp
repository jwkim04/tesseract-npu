// tsr_xrt_selftest.cpp — card bring-up checklist for the Tesseract kernel (docs/xrt-host.md §5).
//
//   tsr_xrt_selftest [--device KIND] [--mem-mib N] [--no-mem] [--no-programs]
//                    [--traces DIR]... [--max-traces N] [--seed N] [-v]
//
//   --device KIND  transport (default: $TSR_DEVICE, else xrt). "mock" runs the same checks on the
//                  adapter register model (ctest); "sim" checks the tool itself.
//   --mem-mib N    size of the per-channel bandwidth transfer (default 64; 0 skips it)
//   --traces DIR   also replay every trace directory under DIR over the transport (repeatable)
//
// 1. Identity: transport / kdev description, ID, VERSION, CFG0..2, CAPS, ADAPTER_VERSION,
//    ADAPTER_STATUS as found at open (the protocol layer clears it there), the channel buffers.
// 2. Memory, per channel: pattern write / read-back at aligned, unaligned, 4 KiB- and
//    4 GiB-crossing and end-of-channel ranges, with guard bytes that must stay untouched;
//    channel aliasing (same offset, different data on every channel); write / read bandwidth.
// 3. Programs, each compared bit-exactly with the ISA simulator given the same program and
//    memory: END; LD + ST through every ARG register as an address + LI / ADDI; LD -> ST round
//    trips on every channel (single line, a 4 KiB burst, unaligned bursts, cross-channel);
//    QACT A32 / A256; a Q8_0 MVM and a Q4_K MVM (T = 1, and T = T_MAX with ACC); a DRAM_RANGE
//    fault. STATUS / ERR_* and every stored byte must match; PERF_RETIRED and PERF_CH_RD_BEATS[c]
//    for c >= 1 must match the simulator (rtl-m2 §11); the other counters are printed.
// 4. --traces: each trace (spec 트레이스 재생 규칙) whose configuration the device can run is
//    loaded (cmd.bin, init regions), run, and compared: STATUS / ERR_* and every expect region,
//    PERF_RETIRED and PERF_CH_RD_BEATS[c >= 1] when the trace records perf, and the exec return
//    code (0 exactly when the trace ends in DONE without ERR).
// Every run (program or trace) also fails when the platform adapter flagged an error during it
// (ADAPTER_STATUS, tsr::transport_adapter_log), even if the core's results match; the last check
// requires no adapter error in any run and a clean register.
// Exit code: 0 every check passed, 1 a check failed, 2 the transport did not open / bad usage.

#include "tsr_isa.h"
#include "tsr_kdev.h"
#include "tsr_ref.h"
#include "tsr_sim.h"
#include "tsr_trace.h"
#include "tsr_transport.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

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
    uint32_t u32() { return (uint32_t) (next() >> 32); }
    uint32_t below(uint32_t n) { return (uint32_t) ((next() >> 32) % n); }
    void fill(void * p, size_t n) {
        uint8_t * b = (uint8_t *) p;
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            const uint64_t v = next();
            memcpy(b + i, &v, 8);
        }
        for (; i < n; ++i) b[i] = (uint8_t) next();
    }
    double normal() {
        const double u1 = ((next() >> 11) + 1) * 0x1.0p-53, u2 = (next() >> 11) * 0x1.0p-53;
        return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
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
    result((ver >> 8) == TSR_ISA_MAJOR, "ISA major", strf("%u", ver >> 8));
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

// ---------------------------------------------------------------------------
// 3. Programs vs the ISA simulator
// ---------------------------------------------------------------------------
struct region { uint32_t ch; uint64_t off; std::vector<uint8_t> data; };
struct span   { uint32_t ch; uint64_t off; uint64_t n; };

struct program {
    std::string            name;
    uint64_t               cmd_beat = 64;
    std::vector<tsr_insn>  insns;
    uint32_t               args[8] = {};
    std::vector<region>    init;
    std::vector<span>      check;
};

class runner {
public:
    runner(tsr_transport * t) : t_(t), cfg_(*tsr_transport_config(t)) {
        ref_ = tsr_sim_create(&cfg_, nullptr);
        if (ref_) tsr_sim_set_threads(ref_, 4);
    }
    ~runner() { tsr_sim_destroy(ref_); }
    bool ok() const { return ref_ != nullptr; }
    const tsr_config & cfg() const { return cfg_; }
    void run(const program & p);

private:
    std::vector<uint64_t> perf_dev() {
        std::vector<uint64_t> v(tsr_perf_count(cfg_.num_ch));
        for (uint32_t i = 0; i < v.size(); ++i) v[i] = tsr::transport_perf(t_, i);
        return v;
    }
    std::vector<uint64_t> perf_ref() {
        std::vector<uint64_t> v(tsr_perf_count(cfg_.num_ch));
        for (uint32_t i = 0; i < v.size(); ++i) v[i] = tsr_sim_perf(ref_, i);
        return v;
    }
    tsr_transport * t_;
    tsr_config      cfg_;
    tsr_sim *       ref_ = nullptr;
};

void runner::run(const program & p) {
    bool ok = true;
    for (const region & r : p.init) {
        ok = ok && tsr_transport_write(t_, r.ch, r.off, r.data.data(), r.data.size()) == 0;
        memcpy(tsr_sim_dram(ref_, r.ch) + r.off, r.data.data(), r.data.size());
    }
    const size_t cb = p.insns.size() * sizeof(tsr_insn);
    ok = ok && tsr_transport_write(t_, 0, p.cmd_beat * 64, p.insns.data(), cb) == 0;
    memcpy(tsr_sim_dram(ref_, 0) + p.cmd_beat * 64, p.insns.data(), cb);
    if (!ok) {
        result(false, p.name.c_str(), "loading the program failed");
        return;
    }
    const std::vector<uint64_t> d0 = perf_dev(), r0 = perf_ref();
    tsr_exec_result res;
    const auto t0 = std::chrono::steady_clock::now();
    const int rc = tsr_transport_exec(t_, p.cmd_beat, (uint32_t) p.insns.size(), p.args, &res);
    const double wall = secs(t0);
    const uint32_t ast = tsr::transport_adapter_log(t_).last_run;   // the run raised an adapter error
    const uint32_t st = tsr_sim_exec(ref_, p.cmd_beat, (uint32_t) p.insns.size(), p.args);
    const uint32_t ec = tsr_sim_csr_read(ref_, TSR_CSR_ERR_CODE), epc = tsr_sim_csr_read(ref_, TSR_CSR_ERR_PC);
    const uint32_t ei = tsr_sim_csr_read(ref_, TSR_CSR_ERR_INFO);
    const std::vector<uint64_t> d1 = perf_dev(), r1 = perf_ref();
    const bool want_ok = (st & TSR_STATUS_DONE) && !(st & TSR_STATUS_ERR);
    const bool state_ok = res.status == st && res.err_code == ec && res.err_pc == epc && res.err_info == ei && (rc == 0) == want_ok &&
                          ast == 0;
    std::string why;
    if (!state_ok) {
        why = strf("device rc %d STATUS 0x%x ERR %u PC %u INFO 0x%x, simulator STATUS 0x%x ERR %u PC %u INFO 0x%x", rc, res.status, res.err_code,
                   res.err_pc, res.err_info, st, ec, epc, ei);
        if (ast != 0) why += strf("; ADAPTER_STATUS 0x%08x during the run: %s", ast, tsr::adapter_status_str(ast).c_str());
    }
    bool data_ok = true;
    for (const span & s : p.check) {
        std::vector<uint8_t> b(s.n);
        if (tsr_transport_read(t_, s.ch, s.off, b.data(), s.n) != 0) {
            data_ok = false;
            why += strf(" read of ch%u 0x%" PRIx64 " failed;", s.ch, s.off);
            continue;
        }
        const uint8_t * want = tsr_sim_dram(ref_, s.ch) + s.off;
        const int64_t d = first_diff(b.data(), want, s.n);
        if (d >= 0) {
            data_ok = false;
            why += strf(" ch%u byte 0x%" PRIx64 ": device 0x%02x simulator 0x%02x;", s.ch, s.off + (uint64_t) d, b[(size_t) d], want[d]);
        }
    }
    const uint64_t ret_d = d1[TSR_PERF_RETIRED] - d0[TSR_PERF_RETIRED], ret_r = r1[TSR_PERF_RETIRED] - r0[TSR_PERF_RETIRED];
    bool perf_ok = ret_d == ret_r;
    std::string beats;
    for (uint32_t ch = 0; ch < cfg_.num_ch; ++ch) {
        const uint64_t bd = d1[tsr_perf_ch_rd_beats(ch)] - d0[tsr_perf_ch_rd_beats(ch)];
        const uint64_t br = r1[tsr_perf_ch_rd_beats(ch)] - r0[tsr_perf_ch_rd_beats(ch)];
        if (ch >= 1 && bd != br) perf_ok = false;
        beats += strf(" ch%u %" PRIu64 "/%" PRIu64, ch, bd, br);
    }
    if (!perf_ok) why += " PERF differs (retired or rd beats of ch >= 1);";
    result(state_ok && data_ok && perf_ok, p.name.c_str(), why);
    printf("  %-28s %s  retired %" PRIu64 "/%" PRIu64 "  cycles %" PRIu64 " (sim model %" PRIu64 ")  mvm %" PRIu64 "  rd beats dev/sim%s  wall %.3f ms\n",
           p.name.c_str(), state_ok && data_ok && perf_ok ? "ok  " : "FAIL", ret_d, ret_r, d1[TSR_PERF_CYCLES] - d0[TSR_PERF_CYCLES],
           r1[TSR_PERF_CYCLES] - r0[TSR_PERF_CYCLES], d1[TSR_PERF_MVM_CYCLES] - d0[TSR_PERF_MVM_CYCLES], beats.c_str(), wall * 1e3);
}

std::vector<uint8_t> f32_bytes(const std::vector<float> & x) {
    std::vector<uint8_t> b(x.size() * 4);
    memcpy(b.data(), x.data(), b.size());   // little-endian host
    return b;
}

std::vector<float> gen_x(rng & r, size_t n) {
    std::vector<float> x(n);
    const double scale = ldexp(1.0, (int) r.below(8) - 4);
    for (float & v : x) v = (float) (r.normal() * scale);
    if (n >= 64) {
        for (int i = 0; i < 32; ++i) x[32 + i] = 0.0f;                 // an all-zero block
        x[5] = 1.0e-40f;                                                // a subnormal input (DAZ)
        x[7] = -x[9] * 3.0f;
    }
    return x;
}

// ggml blocks of N rows x K for fmt, with finite fp16 scales.
std::vector<uint8_t> gen_blocks(rng & r, uint32_t fmt, uint32_t N, uint32_t K) {
    const size_t bs = tsr_fmt_ggml_block_bytes(fmt), nb = (size_t) N * (K / tsr_fmt_block_size(fmt));
    std::vector<uint8_t> b(bs * nb);
    r.fill(b.data(), b.size());
    for (size_t i = 0; i < nb; ++i) {
        uint8_t * blk = b.data() + i * bs;
        const uint16_t d = tsr_f32_to_fp16((float) ldexp(1.0 + r.below(1000) / 1000.0, -(int) r.below(8) - 6));
        memcpy(blk, &d, 2);
        if (fmt == TSR_FMT_Q4_K) {
            const uint16_t dm = tsr_f32_to_fp16((float) ldexp(1.0 + r.below(1000) / 1000.0, -(int) r.below(8) - 7));
            memcpy(blk + 2, &dm, 2);
        }
    }
    return b;
}

// Repacked weights as one init region per channel at W_OFF (beats).
void put_weights(program & p, uint32_t fmt, uint32_t N, uint32_t K, uint32_t C, uint64_t w_off, const std::vector<uint8_t> & blocks) {
    const uint64_t per_ch = tsr_tensor_ch_beats(fmt, N, K, C) * 64;
    const size_t row_bytes = (size_t) (K / tsr_fmt_block_size(fmt)) * tsr_fmt_ggml_block_bytes(fmt);
    std::vector<std::vector<uint8_t>> img(C, std::vector<uint8_t>(per_ch, 0));
    for (uint32_t n = 0; n < N; ++n) {
        uint32_t ch;
        uint64_t off;
        tsr_row_location(fmt, N, K, C, n, &ch, &off);
        tsr_repack_row(fmt, blocks.data() + n * row_bytes, K, img[ch].data() + off * 64);
    }
    for (uint32_t ch = 0; ch < C; ++ch) {
        if (per_ch) p.init.push_back({ ch, w_off * 64, std::move(img[ch]) });
    }
}

void programs(tsr_transport * t, rng & r) {
    printf("== programs vs the ISA simulator (device/simulator)\n");
    runner run(t);
    if (!run.ok()) {
        result(false, "reference simulator", "cannot create it for this configuration");
        return;
    }
    const tsr_config & c = run.cfg();
    const uint32_t C = c.num_ch, TM = c.t_max;
    const uint64_t IO = TSR_REGION_IO_OFF / 64, CHB = c.ch_bytes / 64, W = TSR_REGION_WEIGHT_OFF / 64 + 0x100;
    auto rnd = [&](size_t n) {
        std::vector<uint8_t> v(n);
        r.fill(v.data(), n);
        return v;
    };

    {   // END
        program p;
        p.name = "END";
        p.insns = { tsr_enc_end() };
        run.run(p);
    }
    {   // LD + ST through every ARG register, LI / ADDI
        program p;
        p.name = "LD/LI/ADDI/ST via ARG0..7";
        p.cmd_beat = 0x1001;
        const uint64_t data = IO + 0x100, high = CHB - 0x2000;
        p.init.push_back({ 0, data * 64, rnd(9 * 64) });
        p.insns.push_back(tsr_enc_ld(0, 0, (uint32_t) data, 9));
        for (uint32_t i = 0; i < 8; ++i) {
            p.args[i] = (uint32_t) (high - 0x100 * i - 3 * i);
            tsr_insn st = tsr_enc_st(i % C, i, 0, 1);
            tsr_insn_set_base(&st, 0, 8 + i, 0);
            p.insns.push_back(st);
            p.check.push_back({ i % C, (uint64_t) p.args[i] * 64, 64 });
        }
        const uint32_t li = (uint32_t) (high - 0x1000);
        p.insns.push_back(tsr_enc_li(1, li));
        p.insns.push_back(tsr_enc_addi(2, 1, 7));
        tsr_insn st = tsr_enc_st(C - 1, 8, 0, 1);
        tsr_insn_set_base(&st, 0, 2, 0);
        p.insns.push_back(st);
        p.check.push_back({ C - 1, ((uint64_t) li + 7) * 64, 64 });
        p.insns.push_back(tsr_enc_end());
        run.run(p);
    }
    {   // LD -> ST round trips on every channel
        program p;
        p.name = "LD->ST all channels";
        for (uint32_t ch = 0; ch < C; ++ch) {
            const uint64_t base = (ch == 0 ? IO + 0x10000 : CHB - 0x10000) + 0x400 * ch;      // 4 KiB aligned
            const uint64_t src = base + 0x41;                                                  // unaligned to 4 KiB bursts
            const uint64_t dst = (ch == 0 ? IO + 0x20000 : CHB - 0x20000) + 0x13 + 0x400 * ch;
            p.init.push_back({ ch, src * 64, rnd(600 * 64) });
            p.insns.push_back(tsr_enc_ld(ch, 0, (uint32_t) src, 200));                     // 200 lines, split at 4 KiB
            p.insns.push_back(tsr_enc_st((ch + 1) % C, 0, (uint32_t) dst, 200));             // to the next channel
            p.insns.push_back(tsr_enc_ld(ch, 200, (uint32_t) (base + 320), 64));            // one aligned 4 KiB burst
            p.insns.push_back(tsr_enc_st(ch, 200, (uint32_t) (dst + 300), 64));
            p.insns.push_back(tsr_enc_ld(ch, 300, (uint32_t) (src + 599), 1));             // a single line
            p.insns.push_back(tsr_enc_st(ch, 300, (uint32_t) (dst + 400), 1));
            p.check.push_back({ (ch + 1) % C, dst * 64, 200 * 64 });
            p.check.push_back({ ch, (dst + 300) * 64, 64 * 64 });
            p.check.push_back({ ch, (dst + 400) * 64, 64 });
        }
        p.insns.push_back(tsr_enc_end());
        run.run(p);
    }
    {   // QACT A32 and A256, T = min(2, T_MAX)
        program p;
        const uint32_t K = 1024, T = std::min<uint32_t>(2, TM), xl = K * 4 / 64;
        p.name = strf("QACT A32+A256 K=%u T=%u", K, T);
        const uint64_t xb = IO + 0x30000;
        std::vector<float> x;
        for (uint32_t tk = 0; tk < T; ++tk) {
            std::vector<float> v = gen_x(r, K);
            x.insert(x.end(), v.begin(), v.end());
        }
        p.init.push_back({ 0, xb * 64, f32_bytes(x) });
        const uint32_t q32 = T * xl, l32 = tsr_qa_lines(TSR_QA_A32, K), q256 = q32 + T * l32, l256 = tsr_qa_lines(TSR_QA_A256, K);
        p.insns = { tsr_enc_ld(0, 0, (uint32_t) xb, T * xl), tsr_enc_qact(TSR_QA_A32, q32, 0, K, T, 0),
                    tsr_enc_qact(TSR_QA_A256, q256, 0, K, T, 0), tsr_enc_st(0, q32, (uint32_t) (IO + 0x31000), T * l32),
                    tsr_enc_st(C - 1, q256, (uint32_t) (IO + 0x32000), T * l256), tsr_enc_end() };
        p.check = { { 0, (IO + 0x31000) * 64, (uint64_t) T * l32 * 64 }, { C - 1, (IO + 0x32000) * 64, (uint64_t) T * l256 * 64 } };
        run.run(p);
    }
    // MVMs: Q8_0 (A32) and Q4_K (A256); T = 1 plain, then T = T_MAX with ACC.
    for (uint32_t fmt : { (uint32_t) TSR_FMT_Q8_0, (uint32_t) TSR_FMT_Q4_K }) {
        for (int pass = 0; pass < 2; ++pass) {
            const bool acc = pass == 1;
            const uint32_t T = pass == 0 ? 1 : TM;
            const uint32_t K = 512, N = fmt == TSR_FMT_Q8_0 ? 4 * C + 3 : 2 * C + 1;
            const uint32_t kind = tsr_qa_kind_for_fmt(fmt), xl = K * 4 / 64, ql = tsr_qa_lines(kind, K), yl = (4 * N + 63) / 64;
            program p;
            p.name = strf("MVM %s N=%u K=%u T=%u%s", fmt == TSR_FMT_Q8_0 ? "Q8_0" : "Q4_K", N, K, T, acc ? " ACC" : "");
            const uint64_t w_off = W + (fmt == TSR_FMT_Q4_K ? 0x1000 : 0), xb = IO + 0x40000, yo = IO + 0x41000, yb = IO + 0x42000;
            put_weights(p, fmt, N, K, C, w_off, gen_blocks(r, fmt, N, K));
            std::vector<float> x;
            for (uint32_t tk = 0; tk < T; ++tk) {
                std::vector<float> v = gen_x(r, K);
                x.insert(x.end(), v.begin(), v.end());
            }
            p.init.push_back({ 0, xb * 64, f32_bytes(x) });
            const uint32_t sq = T * xl, sy = sq + T * ql;
            p.insns = { tsr_enc_ld(0, 0, (uint32_t) xb, T * xl), tsr_enc_qact(kind, sq, 0, K, T, 0) };
            if (acc) {
                std::vector<float> yold(T * yl * 16);
                for (float & v : yold) v = (float) r.normal();
                p.init.push_back({ 0, yo * 64, f32_bytes(yold) });
                p.insns.push_back(tsr_enc_ld(0, sy, (uint32_t) yo, T * yl));
            }
            p.insns.push_back(tsr_enc_mvm(fmt, acc ? TSR_MVM_FLAG_ACC : 0, sy, sq, (uint32_t) w_off, N, K, T, 0));
            p.insns.push_back(tsr_enc_st(0, sy, (uint32_t) yb, T * yl));
            p.insns.push_back(tsr_enc_end());
            p.check = { { 0, yb * 64, (uint64_t) T * yl * 64 } };
            run.run(p);
        }
    }
    {   // A fault: DRAM_RANGE on an LD past every channel limit (ERR_PC 1, ERR_INFO = the beat).
        program p;
        p.name = "fault DRAM_RANGE";
        p.insns = { tsr_enc_nop(), tsr_enc_ld(0, 0, 1u << 28, 1), tsr_enc_end() };
        run.run(p);
    }
}

// ---------------------------------------------------------------------------
// 4. Trace replay over the transport
// ---------------------------------------------------------------------------
void find_traces(const fs::path & dir, std::vector<std::string> & out) {
    std::error_code ec;
    if (fs::exists(dir / "meta.json", ec)) {
        out.push_back(dir.string());
        return;
    }
    std::vector<fs::path> subs;
    for (const auto & e : fs::directory_iterator(dir, ec)) {
        if (e.is_directory(ec)) subs.push_back(e.path());
    }
    std::sort(subs.begin(), subs.end());
    for (const auto & s : subs) find_traces(s, out);
}

// Empty if the device can replay tr exactly, else why not.
std::string trace_unfit(const tsr_trace & tr, const tsr_config & c) {
    const tsr_config & a = tr.cfg;
    if (a.num_ch != c.num_ch) return strf("NUM_CH %u", a.num_ch);
    if (a.lanes != c.lanes || a.k_max != c.k_max || a.spm_bytes != c.spm_bytes || a.vl != c.vl || a.nsreg != c.nsreg || a.caps != c.caps) {
        return "LANES / K_MAX / SPM / VL / NSREG / CAPS differ";
    }
    // A program legal under a smaller T_MAX runs identically under a larger one, except T_RANGE.
    if (a.t_max > c.t_max || (a.t_max != c.t_max && tr.err_code == TSR_ERR_T_RANGE)) return strf("T_MAX %u", a.t_max);
    // DRAM limit: equal, or the trace never depends on it and fits the buffers.
    const uint64_t lim_tr = std::min<uint64_t>(a.ch_bytes / 64, 1ull << a.addr_beat_w);
    const uint64_t lim_dev = std::min<uint64_t>(c.ch_bytes / 64, 1ull << c.addr_beat_w);
    if (lim_tr != lim_dev) {
        if (tr.err_code == TSR_ERR_DRAM_RANGE) return "DRAM_RANGE trace with a different channel size";
        auto fits = [&](const tsr_region * r, size_t n) {
            for (size_t i = 0; i < n; ++i) {
                if (r[i].byte_off + r[i].size > c.ch_bytes) return false;
            }
            return true;
        };
        if (!fits(tr.cap.init, tr.cap.n_init) || !fits(tr.cap.expect, tr.cap.n_expect) ||
            (tr.cmd_addr_beat + (tr.cmd_count + 1) / 2) * 64 > c.ch_bytes) {
            return "regions beyond the channel buffer";
        }
    }
    return "";
}

void traces(tsr_transport * t, const std::vector<std::string> & dirs, size_t max_traces) {
    const tsr_config & c = *tsr_transport_config(t);
    std::vector<std::string> list;
    for (const std::string & d : dirs) find_traces(d, list);
    printf("== trace replay (%zu trace directories)\n", list.size());
    size_t n_pass = 0, n_fail = 0, n_skip = 0;
    for (const std::string & dir : list) {
        if (n_pass + n_fail >= max_traces) break;
        tsr_trace tr;
        memset(&tr, 0, sizeof(tr));
        if (tsr_trace_read(dir.c_str(), &tr) != 0) {
            result(false, dir.c_str(), "unreadable trace");
            n_fail++;
            continue;
        }
        const std::string unfit = trace_unfit(tr, c);
        if (!unfit.empty()) {
            if (g_verbose) printf("  skip  %s (%s)\n", dir.c_str(), unfit.c_str());
            n_skip++;
            tsr_trace_free(&tr);
            continue;
        }
        bool ok = true;
        std::string why;
        // cmd.bin, clipped to the channel like tsr_trace_replay (a command buffer past the end faults at fetch).
        if (tr.cmd != nullptr && tr.cmd_count > 0 && tr.cmd_addr_beat < c.ch_bytes / 64) {
            const uint64_t lo = tr.cmd_addr_beat * 64;
            const uint64_t n = std::min<uint64_t>((uint64_t) tr.cmd_count * 32, c.ch_bytes - lo);
            ok = tsr_transport_write(t, 0, lo, tr.cmd, (size_t) n) == 0;
        }
        for (size_t i = 0; ok && i < tr.cap.n_init; ++i) {
            const tsr_region & r = tr.cap.init[i];
            ok = tsr_transport_write(t, r.ch, r.byte_off, r.data, r.size) == 0;
        }
        if (!ok) why = "loading failed";
        std::vector<uint64_t> p0(tsr_perf_count(c.num_ch)), p1(p0.size());
        for (uint32_t i = 0; i < p0.size(); ++i) p0[i] = tsr::transport_perf(t, i);
        tsr_exec_result res;
        if (ok) {
            const int rc = tsr_transport_exec(t, tr.cmd_addr_beat, tr.cmd_count, tr.args, &res);
            const uint32_t ast = tsr::transport_adapter_log(t).last_run;
            const bool want_ok = (tr.status & TSR_STATUS_DONE) && !(tr.status & TSR_STATUS_ERR);
            if (res.status != tr.status || res.err_code != tr.err_code || res.err_pc != tr.err_pc ||
                (tr.has_err_info && res.err_info != tr.err_info)) {
                ok = false;
                why = strf("STATUS 0x%x ERR %u PC %u INFO 0x%x, expected 0x%x %u %u 0x%x", res.status, res.err_code, res.err_pc, res.err_info,
                           tr.status, tr.err_code, tr.err_pc, tr.err_info);
            } else if ((rc == 0) != want_ok) {
                ok = false;
                why = strf("exec returned %d for a run that ends in STATUS 0x%x", rc, res.status);
            }
            if (ast != 0) {
                ok = false;
                why += strf("%sADAPTER_STATUS 0x%08x during the run: %s", why.empty() ? "" : "; ", ast, tsr::adapter_status_str(ast).c_str());
            }
        }
        for (size_t i = 0; ok && i < tr.cap.n_expect; ++i) {
            const tsr_region & r = tr.cap.expect[i];
            std::vector<uint8_t> b(r.size);
            if (tsr_transport_read(t, r.ch, r.byte_off, b.data(), r.size) != 0) {
                ok = false;
                why = "read-back failed";
                break;
            }
            const int64_t d = first_diff(b.data(), r.data, r.size);
            if (d >= 0) {
                ok = false;
                why = strf("expect ch%u byte 0x%" PRIx64 ": got 0x%02x expected 0x%02x", r.ch, r.byte_off + (uint64_t) d, b[(size_t) d], r.data[d]);
            }
        }
        if (ok && tr.perf.valid) {
            for (uint32_t i = 0; i < p1.size(); ++i) p1[i] = tsr::transport_perf(t, i);
            if (p1[TSR_PERF_RETIRED] - p0[TSR_PERF_RETIRED] != tr.perf.retired) {
                ok = false;
                why = strf("PERF_RETIRED %" PRIu64 ", trace %" PRIu64, p1[TSR_PERF_RETIRED] - p0[TSR_PERF_RETIRED], tr.perf.retired);
            }
            for (uint32_t ch = 1; ok && ch < c.num_ch; ++ch) {
                const uint64_t b = p1[tsr_perf_ch_rd_beats(ch)] - p0[tsr_perf_ch_rd_beats(ch)];
                if (b != tr.perf.ch_rd_beats[ch]) {
                    ok = false;
                    why = strf("PERF_CH_RD_BEATS[%u] %" PRIu64 ", trace %" PRIu64, ch, b, tr.perf.ch_rd_beats[ch]);
                }
            }
        }
        result(ok, dir.c_str(), why);
        (ok ? n_pass : n_fail)++;
        tsr_trace_free(&tr);
    }
    printf("  traces: %zu passed, %zu failed, %zu skipped (configuration the device cannot replay exactly)\n", n_pass, n_fail, n_skip);
}

void usage() {
    fprintf(stderr, "usage: tsr_xrt_selftest [--device KIND] [--mem-mib N] [--no-mem] [--no-programs] [--traces DIR]... "
                    "[--max-traces N] [--seed N] [-v]\n");
}

} // namespace

int main(int argc, char ** argv) {
    const char * dev = getenv("TSR_DEVICE");
    if (dev == nullptr || dev[0] == '\0') dev = "xrt";
    uint64_t mem_mib = 64, seed = 1;
    size_t max_traces = (size_t) -1;
    bool do_mem = true, do_prog = true;
    std::vector<std::string> trace_dirs;
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
        else if (a == "--no-programs") do_prog = false;
        else if (a == "--traces") trace_dirs.push_back(val("--traces"));
        else if (a == "--max-traces") max_traces = (size_t) strtoull(val("--max-traces"), nullptr, 10);
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
    if (do_prog) programs(t, r);
    if (!trace_dirs.empty()) traces(t, trace_dirs, max_traces);
    if (tsr::kdev * kd = tsr::transport_kdev(t)) {
        // The protocol layer clears ADAPTER_STATUS after every run that set it: the record of all
        // runs is the adapter log, the register shows only what happened since the last run.
        uint32_t as = 0;
        const bool ok = kd->reg_read(tsr::KREG_ADAPTER_STATUS, &as) == 0;
        const tsr::adapter_log lg = tsr::transport_adapter_log(t);
        result(ok && as == 0 && lg.runs == 0, "ADAPTER_STATUS clean in every run and at the end",
               lg.runs == 0 && as == 0 ? std::string("ok")
                                       : strf("%" PRIu64 " run(s) with adapter errors (%s); register now 0x%08x", lg.runs,
                                              tsr::adapter_status_str(lg.sticky | (uint32_t) std::min<uint64_t>(lg.count, 0xFFFFu) << 16).c_str(),
                                              as));
    }
    tsr_transport_close(t);
    printf("tsr_xrt_selftest (%s): %d checks passed, %d failed -> %s\n", dev, g_pass, g_fail, g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
