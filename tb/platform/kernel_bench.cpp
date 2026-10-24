// kernel_bench.cpp — Verilator C++ bench of the Alveo XRT kernel (tsr_xrt_kernel, or a packaged
// top through tb/platform/sv/kwrap_<kernel>.sv; model class Vkernel). Built by
// tb/fastreplay/run.py --top kernel|tsr_u200_k1|tsr_u200_k4 (fr_config.h from the trace
// configuration, as for the core replayer).
//
//   kernel_bench [fastreplay options] TRACE_DIR...      trace replay through the kernel: every check
//                                                       of tb/fastreplay (fr_common.h) plus the
//                                                       kernel checks of kernel_dut.h
//   kernel_bench --directed all|NAME,... [--seed N] [--dut K=V] [TRACE_DIR...]
//                                                       directed tests; trace directories (optional)
//                                                       serve as workloads of `reset`
// Directed tests (docs/platform-u200.md §7):
//   regmap     reset values, read-back of every RW register (byte strobes), unmapped offsets,
//              GIE / IER / ISR toggle and the interrupt pin, auto_restart ignored, ADAPTER_VERSION,
//              the core CSR window (ID / CFG, RW registers, partial strobes dropped, unaligned and
//              unmapped offsets), concurrent read and write lanes
//   apctrl     ap_ctrl_hs: completion by interrupt, ap_done / ap_ready clear on read, back-to-back
//              starts, ap_start while busy (queued run), a write of 0 starts nothing, IER / GIE
//              combinations; every AP_CTRL read during a run is checked against XRT's completion
//              rule (never ap_start = 0 with ap_idle = 1 and ap_done = 0)
//   window     window access while running: STATUS / PC / PERF polling (monotonic), PERF_CH_RD_BEATS
//              of a copy program on every channel while its ch<c>_base registers are rewritten (the
//              run keeps the bases latched at its start), ABORT through the window (ABORTED, ap_done),
//              SOFT_RESET through the window while BUSY (no DONE: the adapter completes by polling)
//   errors     SLVERR / DECERR injection on R and B: data unaffected, ADAPTER_STATUS bits and count
//              == injected, W1C per bit and of the count, count saturation at 0xFFFF; stale R / B
//              responses (no request outstanding) dropped and flagged (bit2), the core sees none
//   misaligned ch<c>_base not 4 KiB aligned: the run is refused (bit3, ap_done, core untouched,
//              no memory traffic), then an aligned run works
//   reset      ap_rst_n in the middle of traffic (4, 5 or 20 cycles low, random points of LD / ST /
//              MVM traffic): every register at its reset value, no AXI traffic, then a fresh run
//              passes every replay check (4 = RST_MIN: the minimum for RST_PIPE <= 3)
// Exit status: number of failed directed tests (or of failed runs).

#include <verilated.h>

#include "Vkernel.h"
#if VM_TRACE
#include <verilated_vcd_c.h>
#endif

#include "fr_config.h"
#include "fr_common.h"
#include "kernel_dut.h"

namespace {

// ---------------------------------------------------------------------------------------------
// Directed-test helpers
// ---------------------------------------------------------------------------------------------
struct T {
    KernelDut& d;
    const Options& opt;
    std::vector<std::string> errs;
    uint64_t checks = 0;

    void fail(const std::string& s) {
        if (errs.size() < 40) errs.push_back(s);
    }
    void eq(uint64_t got, uint64_t want, const std::string& what) {
        ++checks;
        if (got != want)
            fail(strf("%s = 0x%llx, expected 0x%llx", what.c_str(), (unsigned long long)got, (unsigned long long)want));
    }
    void ok(bool c, const std::string& what) {
        ++checks;
        if (!c) fail(what);
    }
    uint32_t reg(uint32_t a) { return d.reg(a); }
    void wreg(uint32_t a, uint32_t v, uint8_t strb = 0xF) { d.wreg(a, v, nullptr, strb); }
    uint32_t csr(uint32_t a) {
        bool o = true;
        return d.csr(a, false, 0, &o);
    }
    void wcsr(uint32_t a, uint32_t v) {
        bool o = true;
        d.csr(a, true, v, &o);
    }
    uint64_t perf(unsigned i) {
        const uint64_t lo = csr(TSR_CSR_PERF_BASE + 8 * i), hi = csr(TSR_CSR_PERF_BASE + 8 * i + 4);
        return lo | hi << 32;
    }
    void steps(unsigned n) {
        for (unsigned i = 0; i < n; ++i) d.step();
    }
    // collect bench / memory / RTL findings
    void harvest(const char* where) {
        for (const auto& e : d.errs) fail(strf("[%s] %s", where, e.c_str()));
        d.errs.clear();
        for (const auto& e : d.mem().errors()) fail(strf("[%s] %s", where, e.c_str()));
        for (const auto& e : d.done_errs()) fail(strf("[%s] %s", where, e.c_str()));
        d.done_errs().clear();
        for (const auto& e : g_rtl_errors) fail(strf("[%s] %s", where, e.c_str()));
        g_rtl_errors.clear();
        g_rtl_error_count = 0;
    }
};

std::vector<uint8_t> prog_bytes(const std::vector<tsr_insn>& p) {
    std::vector<uint8_t> b(p.size() * TSR_INSN_BYTES);
    for (size_t i = 0; i < p.size(); ++i)
        for (unsigned w = 0; w < 8; ++w)
            for (unsigned k = 0; k < 4; ++k) b[i * 32 + 4 * w + k] = (uint8_t)(p[i].w[w] >> (8 * k));
    return b;
}

std::vector<tsr_insn> prog_nops(uint32_t n) {
    std::vector<tsr_insn> p(n, tsr_enc_nop());
    p.push_back(tsr_enc_end());
    return p;
}

// Copy program: per channel c, `reps` times: LD n lines from SRC to SPM, ST them to DST + rep * n.
constexpr uint64_t CP_SRC = 0x200000, CP_DST = 0x300000;   // beats (128 MiB, 192 MiB)
std::vector<tsr_insn> prog_copy(uint32_t n, uint32_t reps) {
    std::vector<tsr_insn> p;
    const uint32_t spm_lines = FR_SPM_BYTES / 64;
    const uint32_t nn = std::min<uint32_t>(n, spm_lines / NUM_CH);
    for (uint32_t r = 0; r < reps; ++r)
        for (unsigned c = 0; c < NUM_CH; ++c) {
            p.push_back(tsr_enc_ld(c, c * nn, (uint32_t)CP_SRC + r * nn, nn));
            p.push_back(tsr_enc_st(c, c * nn, (uint32_t)CP_DST + r * nn, nn));
        }
    p.push_back(tsr_enc_end());
    return p;
}

constexpr uint64_t CMD_BEAT = 0x1000;     // 256 KiB

// Fresh memory, program at CMD_BEAT, copy sources (random) and destinations (expected).
void setup(T& t, const std::vector<tsr_insn>& prog, uint64_t seed, uint32_t cp_n = 0, uint32_t cp_reps = 0,
           const char* regime = "rand") {
    fr::Regime reg;
    fr::Regime::by_name(regime, &reg);
    t.d.prepare(reg, seed);
    const auto b = prog_bytes(prog);
    t.d.mem().load(0, CMD_BEAT * 64, b.data(), b.size());
    t.d.mem().allow_beats(0, CMD_BEAT, CMD_BEAT + (b.size() + 63) / 64);
    if (cp_n) {
        fr::Rng r;
        r.seed(seed * 77 + 1);
        const uint32_t nn = std::min<uint32_t>(cp_n, (FR_SPM_BYTES / 64) / NUM_CH);
        std::vector<uint8_t> src((size_t)nn * cp_reps * 64);
        for (unsigned c = 0; c < NUM_CH; ++c) {
            for (auto& x : src) x = (uint8_t)r.next();
            t.d.mem().load(c, CP_SRC * 64, src.data(), src.size());
            t.d.mem().allow_beats(c, CP_SRC, CP_SRC + (uint64_t)nn * cp_reps);
            t.d.mem().expect_beats(c, CP_DST, CP_DST + (uint64_t)nn * cp_reps);
        }
    }
}

void check_copy(T& t, uint32_t cp_n, uint32_t cp_reps, const char* what) {
    const uint32_t nn = std::min<uint32_t>(cp_n, (FR_SPM_BYTES / 64) / NUM_CH);
    for (unsigned c = 0; c < NUM_CH; ++c) {
        const uint8_t* m = t.d.mem().mem(c);
        t.ok(memcmp(m + CP_SRC * 64, m + CP_DST * 64, (size_t)nn * cp_reps * 64) == 0,
             strf("%s: ch%u copy destination differs from its source", what, c));
        t.ok(t.d.mem().stats(c).extra_writes == 0, strf("%s: ch%u writes outside the destination", what, c));
        t.ok(t.d.mem().stats(c).bad_reads == 0, strf("%s: ch%u reads of uninitialised beats", what, c));
    }
}

// AP_CTRL read during a run: XRT's completion rule must never fire early.
uint32_t ap_read(T& t, bool run_open) {
    const uint32_t v = t.reg(KR_AP_CTRL);
    if (run_open && !(v & AP_START) && (v & AP_IDLE) && !(v & AP_DONE))
        t.fail(strf("AP_CTRL 0x%x during a run: ap_start = 0 and ap_idle = 1 without ap_done", v));
    return v;
}

// Manual completion wait: poll AP_CTRL until ap_done. Returns the AP_CTRL value (0 on timeout).
uint32_t wait_done(T& t, uint64_t max_cycles, const char* what) {
    const uint64_t c0 = t.d.cycles();
    while (t.d.cycles() - c0 < max_cycles) {
        const uint32_t v = ap_read(t, true);
        if (v & AP_DONE) return v;
        t.steps(8);
    }
    t.fail(strf("%s: no ap_done within %llu cycles", what, (unsigned long long)max_cycles));
    return 0;
}

// start with the bench's own completion detection disabled (the test polls)
void start_manual(T& t, uint32_t count, bool irq = false) {
    bool ok = true;
    t.d.irq_mode = irq;
    t.d.manual = true;
    t.d.start_raw(CMD_BEAT, count, nullptr, &ok);
    t.ok(ok, "start: AXI4-Lite writes completed");
}
void end_manual(T& t) {
    t.d.end_run();
    t.d.manual = false;
}

// ---------------------------------------------------------------------------------------------
// regmap
// ---------------------------------------------------------------------------------------------
void t_regmap(T& t) {
    KernelDut& d = t.d;
    d.hw_reset();
    // reset values
    t.eq(t.reg(KR_AP_CTRL), AP_IDLE, "AP_CTRL after reset");
    for (uint32_t a : {KR_GIE, KR_IER, KR_ISR, KR_CMD_LO, KR_CMD_HI, KR_CMD_CNT, KR_ASTATUS}) t.eq(t.reg(a), 0, strf("reg 0x%03x after reset", a));
    for (unsigned i = 0; i < 8; ++i) t.eq(t.reg(KR_ARG0 + 4 * i), 0, strf("arg%u after reset", i));
    for (unsigned c = 0; c < NUM_CH; ++c) {
        t.eq(t.reg(KR_BASE0 + 8 * c), 0, strf("ch%u_base lo after reset", c));
        t.eq(t.reg(KR_BASE0 + 8 * c + 4), 0, strf("ch%u_base hi after reset", c));
    }
    t.eq(t.reg(KR_AVERSION), 0x00010000u | NUM_CH, "ADAPTER_VERSION");
    t.ok(!d.irq(), "interrupt low after reset");
    // RW registers: random values read back
    std::vector<uint32_t> rw = {KR_CMD_LO, KR_CMD_HI, KR_CMD_CNT};
    for (unsigned i = 0; i < 8; ++i) rw.push_back(KR_ARG0 + 4 * i);
    for (unsigned c = 0; c < NUM_CH; ++c) rw.push_back(KR_BASE0 + 8 * c), rw.push_back(KR_BASE0 + 8 * c + 4);
    fr::Rng r;
    r.seed(t.opt.seed * 31 + 7);
    std::vector<uint32_t> val(rw.size());
    for (size_t i = 0; i < rw.size(); ++i) {
        val[i] = (uint32_t)r.next();
        t.wreg(rw[i], val[i]);
    }
    for (size_t i = 0; i < rw.size(); ++i) t.eq(t.reg(rw[i]), val[i], strf("RW reg 0x%03x read-back", rw[i]));
    // byte strobes
    for (size_t i = 0; i < rw.size(); i += 3) {
        const uint8_t strb = (uint8_t)(1 + r.range(0, 13));
        const uint32_t v = (uint32_t)r.next();
        uint32_t m = 0;
        for (unsigned b = 0; b < 4; ++b) if (strb >> b & 1) m |= 0xFFu << (8 * b);
        t.wreg(rw[i], v, strb);
        val[i] = (val[i] & ~m) | (v & m);
        t.eq(t.reg(rw[i]), val[i], strf("RW reg 0x%03x after a write with WSTRB 0x%x", rw[i], strb));
    }
    // unmapped offsets: read 0, writes ignored
    std::vector<uint32_t> um = {0x01C, 0x0E0, 0x0EC, 0x0F8, 0x0FC, 0x100, 0x7FC, 0x800, 0xFFC};
    if (NUM_CH < 16) um.push_back(KR_BASE0 + 8 * NUM_CH), um.push_back(KR_BASE0 + 8 * NUM_CH + 4);
    for (uint32_t a : um) t.wreg(a, 0xFFFFFFFFu);
    for (uint32_t a : um) t.eq(t.reg(a), 0, strf("unmapped 0x%03x", a));
    for (size_t i = 0; i < rw.size(); ++i) t.eq(t.reg(rw[i]), val[i], strf("RW reg 0x%03x after unmapped writes", rw[i]));
    // AP_CTRL: auto_restart and ap_start = 0 writes start nothing
    t.wreg(KR_AP_CTRL, 0x80);
    t.wreg(KR_AP_CTRL, 0x0E);
    t.steps(20);
    t.eq(t.reg(KR_AP_CTRL), AP_IDLE, "AP_CTRL after writes of auto_restart / done / idle / ready");
    // GIE / IER: bit 0 only; ISR toggles on a write of 1; the interrupt pin
    t.wreg(KR_GIE, 0xFFFFFFFEu);
    t.wreg(KR_IER, 0xFFFFFFFEu);
    t.eq(t.reg(KR_GIE), 0, "GIE after 0xFFFFFFFE");
    t.eq(t.reg(KR_IER), 0, "IER after 0xFFFFFFFE (ap_ready interrupt not supported)");
    t.wreg(KR_GIE, 0xFFFFFFFFu);
    t.wreg(KR_IER, 0xFFFFFFFFu);
    t.eq(t.reg(KR_GIE), 1, "GIE");
    t.eq(t.reg(KR_IER), 1, "IER");
    t.wreg(KR_ISR, 0);
    t.eq(t.reg(KR_ISR), 0, "ISR after a write of 0");
    t.ok(!d.irq(), "interrupt low with ISR = 0");
    t.wreg(KR_ISR, 1);
    t.eq(t.reg(KR_ISR), 1, "ISR after toggle (write 1)");
    t.ok(d.irq(), "interrupt high with GIE & IER & ISR");
    t.wreg(KR_GIE, 0);
    t.steps(2);
    t.ok(!d.irq(), "interrupt low with GIE = 0");
    t.wreg(KR_GIE, 1);
    t.wreg(KR_IER, 0);
    t.steps(2);
    t.ok(!d.irq(), "interrupt low with IER = 0");
    t.wreg(KR_ISR, 1, 0x2);           // strobe of byte 1 only: no toggle
    t.eq(t.reg(KR_ISR), 1, "ISR after a write without WSTRB[0]");
    t.wreg(KR_ISR, 1);
    t.eq(t.reg(KR_ISR), 0, "ISR after second toggle");
    t.wreg(KR_GIE, 0);
    // ADAPTER_STATUS: W1C of nothing
    t.wreg(KR_ASTATUS, 0xFFFFFFFFu);
    t.eq(t.reg(KR_ASTATUS), 0, "ADAPTER_STATUS");
    // core CSR window
    t.eq(t.csr(TSR_CSR_ID), TSR_ID_MAGIC, "window: core ID");
    t.eq(t.csr(TSR_CSR_CFG0), FR_NUM_CH | (FR_LANES << 8) | (FR_T_MAX << 20), "window: CFG0");
    t.eq(t.csr(TSR_CSR_CFG1), FR_SPM_BYTES / 64, "window: CFG1");
    std::vector<uint32_t> cw = {TSR_CSR_CMD_ADDR_LO, TSR_CSR_CMD_ADDR_HI, TSR_CSR_CMD_COUNT};
    for (unsigned i = 0; i < 8; ++i) cw.push_back(TSR_CSR_ARG0 + 4 * i);
    std::vector<uint32_t> cv(cw.size());
    for (size_t i = 0; i < cw.size(); ++i) {
        cv[i] = (uint32_t)r.next();
        t.wcsr(cw[i], cv[i]);
    }
    for (size_t i = 0; i < cw.size(); ++i) t.eq(t.csr(cw[i]), cv[i], strf("window: core 0x%03x read-back", cw[i]));
    t.wreg(KB_AXIL_WIN + TSR_CSR_ARG0, 0x12345678, 0x3);       // partial strobe: dropped
    t.eq(t.csr(TSR_CSR_ARG0), cv[3], "window: core ARG0 after a partial-strobe write (dropped)");
    t.eq(t.reg(KB_AXIL_WIN + TSR_CSR_ARG0 + 1), 0, "window: unaligned core offset reads 0");
    t.eq(t.reg(KB_AXIL_WIN + 0xFFC), 0, "window: unmapped core offset 0xFFC");
    t.eq(t.csr(TSR_CSR_STATUS), 0, "window: core STATUS");
    // concurrent read and write lanes: reads of registers that the writes do not touch
    d.host.serial = false;
    std::vector<std::shared_ptr<LiteTx>> rd;
    for (unsigned i = 0; i < 24; ++i) {
        const uint32_t a = KR_ARG0 + 4 * (i % 8);
        d.host.push(true, a, 0xC0DE0000u + i);
        if (i % 3 == 0) rd.push_back(d.host.push(false, KR_AVERSION, 0));
        if (i % 3 == 1) rd.push_back(d.host.push(false, KB_AXIL_WIN + TSR_CSR_ID, 0));
        if (i % 3 == 2) rd.push_back(d.host.push(false, KR_CMD_CNT, 0));
    }
    for (unsigned n = 0; !d.host.idle() && n < 20000; ++n) d.step();
    d.host.serial = true;
    for (size_t i = 0; i < rd.size(); ++i) {
        const uint32_t want = i % 3 == 0 ? (0x00010000u | NUM_CH) : i % 3 == 1 ? TSR_ID_MAGIC : val[2];
        t.ok(rd[i]->done, "concurrent read completed");
        t.eq(rd[i]->rdata, want, strf("concurrent read %zu", i));
    }
    for (unsigned i = 0; i < 8; ++i) t.eq(t.reg(KR_ARG0 + 4 * i), 0xC0DE0000u + 16 + i, strf("arg%u after concurrent writes", i));
    // access latency with an ideal master (VALID at once, READY always): reported, not checked
    {
        const bool fast = d.host.fast;
        d.host.fast = true;
        uint64_t c0 = d.cycles();
        t.reg(KR_AVERSION);
        const uint64_t l_reg = d.cycles() - c0;
        c0 = d.cycles();
        t.csr(TSR_CSR_ID);
        const uint64_t l_win = d.cycles() - c0;
        c0 = d.cycles();
        t.wreg(KR_ARG0, 1);
        const uint64_t l_wr = d.cycles() - c0;
        c0 = d.cycles();
        t.wcsr(TSR_CSR_ARG0, 1);
        const uint64_t l_wwr = d.cycles() - c0;
        d.host.fast = fast;
        fprintf(stderr, "[regmap] AXI4-Lite cycles per access (ideal master, incl. its issue cycle): register read %llu, "
                        "write %llu; window read %llu, write %llu\n",
                (unsigned long long)l_reg, (unsigned long long)l_wr, (unsigned long long)l_win, (unsigned long long)l_wwr);
    }
    t.harvest("regmap");
}

// ---------------------------------------------------------------------------------------------
// apctrl
// ---------------------------------------------------------------------------------------------
void t_apctrl(T& t) {
    KernelDut& d = t.d;
    d.hw_reset();
    const uint32_t N = 200;
    setup(t, prog_nops(N), t.opt.seed + 1);
    // (a) completion by interrupt; ap_done / ap_ready clear on read; ISR toggle
    uint64_t r0 = t.perf(TSR_PERF_RETIRED);
    start_manual(t, N + 1, true);
    {
        const uint64_t c0 = d.cycles();
        while (!d.irq() && d.cycles() - c0 < 100000) d.step();
        t.ok(d.irq(), "interrupt at completion");
    }
    t.eq(t.reg(KR_AP_CTRL), AP_IDLE | AP_DONE | AP_READY, "AP_CTRL at the interrupt");
    t.eq(t.reg(KR_AP_CTRL), AP_IDLE, "AP_CTRL second read (clear on read)");
    t.eq(t.reg(KR_ISR), 1, "ISR at completion");
    t.wreg(KR_ISR, 1);
    t.steps(2);
    t.eq(t.reg(KR_ISR), 0, "ISR after toggle");
    t.ok(!d.irq(), "interrupt low after the ISR toggle");
    end_manual(t);
    t.eq(t.csr(TSR_CSR_STATUS), 2, "core STATUS (DONE)");
    t.eq(t.perf(TSR_PERF_RETIRED) - r0, N + 1, "retired instructions of one run");
    t.wreg(KR_GIE, 0);
    t.wreg(KR_IER, 0);
    // (b) back to back: ap_start right after the completion read
    r0 = t.perf(TSR_PERF_RETIRED);
    start_manual(t, N + 1);
    for (int k = 0; k < 4; ++k) {
        const uint32_t v = wait_done(t, 100000, "back-to-back");
        t.eq(v & (AP_IDLE | AP_DONE | AP_READY | AP_START), AP_IDLE | AP_DONE | AP_READY, "AP_CTRL at completion");
        if (k < 3) t.wreg(KR_AP_CTRL, AP_START);
    }
    end_manual(t);
    t.eq(t.perf(TSR_PERF_RETIRED) - r0, 4 * (N + 1), "retired instructions of 4 back-to-back runs");
    t.eq(t.reg(KR_AP_CTRL), AP_IDLE, "AP_CTRL after the back-to-back runs");
    // (c) ap_start while busy: stays pending, runs when the first run completes
    const uint32_t NB = 3000;
    setup(t, prog_nops(NB), t.opt.seed + 2);
    r0 = t.perf(TSR_PERF_RETIRED);
    start_manual(t, NB + 1);
    t.steps(150);
    t.wreg(KR_AP_CTRL, AP_START);
    {
        const uint32_t v = ap_read(t, true);
        t.eq(v & (AP_START | AP_IDLE | AP_DONE), AP_START, "AP_CTRL with a start pending during a run");
    }
    const uint32_t v1 = wait_done(t, 400000, "start while busy (first run)");
    t.ok((v1 & AP_START) == 0 || (ap_read(t, true) & AP_START) == 0, "pending start accepted at the first completion");
    const uint64_t mid = t.perf(TSR_PERF_RETIRED) - r0;
    t.ok(mid >= NB + 1 && mid < 2 * (NB + 1), strf("second run started after the first (retired %llu)", (unsigned long long)mid));
    wait_done(t, 400000, "start while busy (second run)");
    end_manual(t);
    t.eq(t.perf(TSR_PERF_RETIRED) - r0, 2 * (NB + 1), "retired instructions of the queued pair");
    t.eq(t.reg(KR_AP_CTRL), AP_IDLE, "AP_CTRL after the queued pair");
    // (d) IER / GIE combinations
    setup(t, prog_nops(N), t.opt.seed + 3);
    t.wreg(KR_GIE, 1);
    start_manual(t, N + 1);            // irq_mode false: GIE / IER written 0 by start
    t.wreg(KR_GIE, 1);
    t.wreg(KR_IER, 0);
    wait_done(t, 100000, "IER = 0 run");
    t.steps(4);
    t.eq(t.reg(KR_ISR), 0, "ISR after a run with IER = 0");
    t.ok(!d.irq(), "no interrupt with IER = 0");
    end_manual(t);
    start_manual(t, N + 1);
    t.wreg(KR_GIE, 0);
    t.wreg(KR_IER, 1);
    wait_done(t, 100000, "GIE = 0 run");
    t.steps(4);
    t.eq(t.reg(KR_ISR), 1, "ISR after a run with GIE = 0, IER = 1");
    t.ok(!d.irq(), "no interrupt with GIE = 0");
    t.wreg(KR_GIE, 1);
    t.steps(3);
    t.ok(d.irq(), "interrupt once GIE = 1");
    t.wreg(KR_ISR, 1);
    t.steps(3);
    t.ok(!d.irq(), "interrupt low after the ISR toggle");
    t.wreg(KR_GIE, 0);
    t.wreg(KR_IER, 0);
    end_manual(t);
    // (e) a write of 0 to AP_CTRL starts nothing
    r0 = t.perf(TSR_PERF_RETIRED);
    t.wreg(KR_AP_CTRL, 0);
    t.steps(300);
    t.eq(t.perf(TSR_PERF_RETIRED) - r0, 0, "no run after writing AP_CTRL = 0");
    t.eq(t.reg(KR_AP_CTRL), AP_IDLE, "AP_CTRL idle");
    t.harvest("apctrl");
}

// ---------------------------------------------------------------------------------------------
// window
// ---------------------------------------------------------------------------------------------
void t_window(T& t) {
    KernelDut& d = t.d;
    d.hw_reset();
    const uint32_t N = 20000;
    setup(t, prog_nops(N), t.opt.seed + 11);
    start_manual(t, N + 1);
    t.steps(300);
    uint64_t last_cyc = 0, last_pc = 0, last_ret = 0;
    for (int k = 0; k < 6; ++k) {
        const uint32_t st = t.csr(TSR_CSR_STATUS);
        t.eq(st & 1, 1, "window STATUS.BUSY while running");
        const uint64_t cyc = t.perf(TSR_PERF_CYCLES), pc = t.csr(TSR_CSR_PC), ret = t.perf(TSR_PERF_RETIRED);
        t.ok(cyc > last_cyc && pc >= last_pc && ret > last_ret, strf("PERF_CYCLES / PC / RETIRED advance (%llu %llu %llu)",
                                                                      (unsigned long long)cyc, (unsigned long long)pc,
                                                                      (unsigned long long)ret));
        last_cyc = cyc, last_pc = pc, last_ret = ret;
        const uint32_t ap = ap_read(t, true);
        t.eq(ap & (AP_IDLE | AP_DONE), 0, "AP_CTRL busy");
        t.steps(200);
    }
    // ABORT through the window
    t.wcsr(TSR_CSR_CTRL, 4);
    wait_done(t, 20000, "ABORT");
    end_manual(t);
    t.eq(t.csr(TSR_CSR_STATUS), 6, "STATUS after ABORT (DONE | ERR)");
    t.eq(t.csr(TSR_CSR_ERR_CODE), 8, "ERR_CODE ABORTED");
    const uint32_t epc = t.csr(TSR_CSR_ERR_PC);
    t.ok(epc > last_pc && epc < N, strf("ERR_PC %u inside the program", epc));
    // SOFT_RESET through the window while BUSY: the run ends without DONE
    start_manual(t, N + 1);
    t.steps(500);
    t.eq(t.csr(TSR_CSR_STATUS) & 1, 1, "BUSY before SOFT_RESET");
    t.wcsr(TSR_CSR_CTRL, 2);
    const uint32_t v = wait_done(t, 20000, "SOFT_RESET while BUSY");
    t.eq(v & (AP_IDLE | AP_DONE), AP_IDLE | AP_DONE, "AP_CTRL after SOFT_RESET");
    end_manual(t);
    t.eq(t.csr(TSR_CSR_STATUS), 0, "STATUS after SOFT_RESET while BUSY");
    t.eq(t.csr(TSR_CSR_PC), 0, "PC after SOFT_RESET");
    t.eq(t.perf(TSR_PERF_RETIRED), 0, "PERF_RETIRED after SOFT_RESET");
    // the next run is normal
    setup(t, prog_nops(100), t.opt.seed + 12);
    start_manual(t, 101);
    wait_done(t, 20000, "run after SOFT_RESET");
    end_manual(t);
    t.eq(t.csr(TSR_CSR_STATUS), 2, "STATUS after the next run");
    // PERF_CH_RD_BEATS polled during a copy on every channel
    const uint32_t n = 256, reps = 3;
    setup(t, prog_copy(n, reps), t.opt.seed + 13, n, reps);
    uint64_t b0[NUM_CH];
    for (unsigned c = 0; c < NUM_CH; ++c) b0[c] = t.perf(TSR_PERF_CH_RD_BEATS + c);
    const auto prog = prog_copy(n, reps);
    start_manual(t, (uint32_t)prog.size());
    uint64_t lastb[NUM_CH] = {0};
    bool moved = false;
    for (int k = 0; k < 400; ++k) {
        const uint32_t ap = ap_read(t, true);
        if (ap & AP_DONE) break;
        if (k == 2) {
            // the bases are latched at the start: rewriting them during the run changes nothing
            for (unsigned c = 0; c < NUM_CH; ++c) {
                t.wreg(KR_BASE0 + 8 * c, 0x5A5A5000u + 0x1000u * c);
                t.wreg(KR_BASE0 + 8 * c + 4, 0x00000777u);
            }
            moved = true;
        }
        for (unsigned c = 0; c < NUM_CH; ++c) {
            const uint64_t b = t.perf(TSR_PERF_CH_RD_BEATS + c) - b0[c];
            t.ok(b >= lastb[c], strf("PERF_CH_RD_BEATS[%u] monotonic", c));
            lastb[c] = b;
        }
        if (k == 399) t.fail("copy program did not finish");
    }
    end_manual(t);
    t.ok(moved, "base registers rewritten during the copy");
    t.eq(t.reg(KR_BASE0 + 4), 0x777, "ch0_base hi reads the value written during the run");
    const uint32_t nn = std::min<uint32_t>(n, (FR_SPM_BYTES / 64) / NUM_CH);
    for (unsigned c = 0; c < NUM_CH; ++c) {
        const uint64_t b = t.perf(TSR_PERF_CH_RD_BEATS + c) - b0[c];
        const uint64_t want = (uint64_t)nn * reps + (c == 0 ? (prog.size() + 1) / 2 : 0);
        t.ok(c == 0 ? (b >= (uint64_t)nn * reps && b <= want + 32) : b == want,
             strf("PERF_CH_RD_BEATS[%u] = %llu after the copy (LD %llu)", c, (unsigned long long)b,
                  (unsigned long long)nn * reps));
        t.eq(b, t.d.mem().stats(c).rd_beats, strf("PERF_CH_RD_BEATS[%u] == AXI read beats", c));
    }
    check_copy(t, n, reps, "window copy");
    t.eq(t.csr(TSR_CSR_STATUS), 2, "STATUS after the copy");
    t.harvest("window");
}

// ---------------------------------------------------------------------------------------------
// errors
// ---------------------------------------------------------------------------------------------
void t_errors(T& t) {
    KernelDut& d = t.d;
    d.hw_reset();
    // (a) SLVERR / DECERR on R and B: data still correct, ADAPTER_STATUS == injected
    const uint32_t n = 128, reps = 4;
    const auto prog = prog_copy(n, reps);
    setup(t, prog, t.opt.seed + 21, n, reps);
    d.ko.p_rresp_err = 0.05;
    d.ko.p_bresp_err = 0.3;
    start_manual(t, (uint32_t)prog.size());
    wait_done(t, 400000, "copy with error responses");
    end_manual(t);
    d.ko.p_rresp_err = d.ko.p_bresp_err = 0;
    check_copy(t, n, reps, "copy with error responses");
    t.eq(t.csr(TSR_CSR_STATUS), 2, "STATUS (the core does not see memory errors)");
    t.ok(d.inj_r > 0 && d.inj_b > 0, strf("errors injected (%llu R, %llu B)", (unsigned long long)d.inj_r, (unsigned long long)d.inj_b));
    const uint32_t want = 3u | (uint32_t)std::min<uint64_t>(d.inj_r + d.inj_b, 0xFFFF) << 16;
    t.eq(t.reg(KR_ASTATUS), want, "ADAPTER_STATUS after injected errors");
    t.wreg(KR_ASTATUS, 1);
    t.eq(t.reg(KR_ASTATUS), want & ~1u, "ADAPTER_STATUS after W1C of bit0");
    t.wreg(KR_ASTATUS, 0x00010000u);
    t.eq(t.reg(KR_ASTATUS), 2, "ADAPTER_STATUS after W1C of the count");
    t.wreg(KR_ASTATUS, 0x2);
    t.eq(t.reg(KR_ASTATUS), 0, "ADAPTER_STATUS after W1C of bit1");
    // (b) saturation: every fetch beat of a long NOP program with an error response
    const uint32_t NS = 2 * 66000;
    setup(t, prog_nops(NS), t.opt.seed + 22, 0, 0, "fast");
    d.ko.p_rresp_err = 1.0;
    start_manual(t, NS + 1);
    wait_done(t, 2000000, "NOP program with error responses");
    end_manual(t);
    d.ko.p_rresp_err = 0;
    t.ok(d.inj_r >= 66000, strf("%llu error responses injected", (unsigned long long)d.inj_r));
    t.eq(t.reg(KR_ASTATUS), 0xFFFF0001u, "ADAPTER_STATUS count saturates at 0xFFFF");
    t.wreg(KR_ASTATUS, 0xFFFFFFFFu);
    t.eq(t.reg(KR_ASTATUS), 0, "ADAPTER_STATUS cleared");
    // (c) stale responses (no request outstanding, e.g. from before a kernel reset): dropped
    d.hw_reset();
    setup(t, prog_nops(10), t.opt.seed + 23);
    uint64_t want_stale = 0;
    for (unsigned c = 0; c < NUM_CH; ++c) {
        for (unsigned id : {0u, 4u, 8u, 15u}) {
            const unsigned beats = 1 + (id + c) % 3;
            d.inject_stale(c, true, id, beats);
            want_stale += beats;
        }
        d.inject_stale(c, false, 4, 0);
        d.inject_stale(c, false, 5 + c % 2, 0);
        want_stale += 2;
    }
    for (unsigned k = 0; k < 5000 && d.stale_pending(); ++k) d.step();
    t.ok(!d.stale_pending(), "stale responses consumed by the adapter");
    t.eq(d.inj_stale, want_stale, "stale responses accepted");
    t.steps(10);
    t.eq(t.reg(KR_ASTATUS), 4, "ADAPTER_STATUS bit2 after stale responses");
    t.wreg(KR_ASTATUS, 4);
    t.harvest("errors (stale)");   // an RTL assertion here would mean the core saw a stale response
    setup(t, prog, t.opt.seed + 24, n, reps);
    start_manual(t, (uint32_t)prog.size());
    wait_done(t, 400000, "copy after stale responses");
    end_manual(t);
    check_copy(t, n, reps, "copy after stale responses");
    t.eq(t.reg(KR_ASTATUS), 0, "ADAPTER_STATUS after a clean run");
    t.harvest("errors");
}

// ---------------------------------------------------------------------------------------------
// misaligned
// ---------------------------------------------------------------------------------------------
void t_misaligned(T& t) {
    KernelDut& d = t.d;
    d.hw_reset();
    setup(t, prog_nops(50), t.opt.seed + 31);
    start_manual(t, 51);
    wait_done(t, 20000, "aligned run");
    end_manual(t);
    t.eq(t.csr(TSR_CSR_STATUS), 2, "STATUS after the aligned run");
    std::vector<unsigned> chs = {0};
    if (NUM_CH > 1) chs.push_back(NUM_CH - 1);
    for (unsigned c : chs) {
        for (uint64_t off : {0x800ull, 0x40ull, 0x1ull}) {
            const uint64_t keep = d.base[c];
            d.base[c] = keep + off;
            const uint64_t r0 = t.perf(TSR_PERF_RETIRED), rb = d.mem().stats(0).rd_bursts;
            start_manual(t, 51);
            const uint32_t v = wait_done(t, 2000, "misaligned base");
            // from the AP_CTRL write to the AP_CTRL read that saw ap_done (one poll ~ 15 cycles)
            const uint64_t dt = d.cycles() - d.start_cyc();
            end_manual(t);
            t.ok(dt < 40, strf("refused run completes at once (%llu cycles)", (unsigned long long)dt));
            t.eq(v & (AP_IDLE | AP_DONE), AP_IDLE | AP_DONE, "AP_CTRL after the refused run");
            t.eq(t.reg(KR_ASTATUS), 8, strf("ADAPTER_STATUS bit3 (ch%u base + 0x%llx)", c, (unsigned long long)off));
            t.eq(t.csr(TSR_CSR_STATUS), 2, "core STATUS untouched by the refused run");
            t.eq(t.perf(TSR_PERF_RETIRED) - r0, 0, "no instruction executed");
            t.eq(d.mem().stats(0).rd_bursts - rb, 0, "no memory traffic");
            t.wreg(KR_ASTATUS, 8);
            d.base[c] = keep;
        }
    }
    start_manual(t, 51);
    wait_done(t, 20000, "aligned run after the refused ones");
    end_manual(t);
    t.eq(t.csr(TSR_CSR_STATUS), 2, "STATUS after the aligned run");
    t.eq(t.reg(KR_ASTATUS), 0, "ADAPTER_STATUS clean");
    t.harvest("misaligned");
}

// ---------------------------------------------------------------------------------------------
// reset
// ---------------------------------------------------------------------------------------------
void t_reset(T& t) {
    KernelDut& d = t.d;
    fr::Rng r;
    r.seed(t.opt.seed * 1000 + 41);
    const unsigned iters = 12;
    for (unsigned it = 0; it < iters; ++it) {
        d.hw_reset();
        // workload: a trace (if given) or a copy program on every channel
        const bool use_trace = !t.opt.traces.empty() && (it % 2 == 0);
        uint64_t budget = 50000;
        if (use_trace) {
            const std::string& path = t.opt.traces[(it / 2) % t.opt.traces.size()];
            tsr_trace tr;
            if (tsr_trace_read(path.c_str(), &tr) != 0) {
                t.fail("cannot read " + path);
                continue;
            }
            fr::Regime reg;
            fr::Regime::by_name(it % 4 == 0 ? "rand" : "slow", &reg);
            d.prepare(reg, t.opt.seed + it);
            if (tr.cmd && tr.cmd_count) {
                d.mem().load(0, tr.cmd_addr_beat * 64, tr.cmd, (uint64_t)tr.cmd_count * TSR_INSN_BYTES);
                d.mem().allow_beats(0, tr.cmd_addr_beat, tr.cmd_addr_beat + (tr.cmd_count * 32ull + 63) / 64);
            }
            for (size_t i = 0; i < tr.cap.n_init; ++i) {
                const tsr_region& g = tr.cap.init[i];
                d.mem().load(g.ch, g.byte_off, g.data, g.size);
                d.mem().allow_beats(g.ch, g.byte_off / 64, (g.byte_off + g.size) / 64);
            }
            budget = derive(tr).est_cycles / 4 + 200;
            d.irq_mode = false;
            d.manual = true;
            bool ok = true;
            d.start_raw(tr.cmd_addr_beat, tr.cmd_count, tr.args, &ok);
            tsr_trace_free(&tr);
        } else {
            const uint32_t n = 64 + 64 * (it % 4), reps = 4;
            setup(t, prog_copy(n, reps), t.opt.seed + 100 + it, n, reps, it % 3 ? "rand" : "slow");
            start_manual(t, (uint32_t)prog_copy(n, reps).size());
            budget = 4000;
        }
        const uint64_t when = r.range(20, budget);
        for (uint64_t k = 0; k < when; ++k) d.step();
        // ap_rst_n low for RST_MIN (docs/platform-u200.md §5), RST_MIN + 1 or 20 cycles
        const unsigned rst = it % 3 == 0 ? RST_MIN : it % 3 == 1 ? RST_MIN + 1 : 20;
        const bool busy = (t.csr(TSR_CSR_STATUS) & 1) != 0;
        d.hw_reset_n(rst);
        d.manual = false;
        t.harvest(strf("reset %u (%s, %llu cycles in, %u low, core %s)", it, use_trace ? "trace" : "copy",
                       (unsigned long long)when, rst, busy ? "busy" : "idle").c_str());
        // every register at its reset value, nothing on the AXI side
        t.eq(t.reg(KR_AP_CTRL), AP_IDLE, "AP_CTRL after reset");
        for (uint32_t a : {KR_GIE, KR_IER, KR_ISR, KR_CMD_LO, KR_CMD_CNT, KR_ARG0, KR_BASE0, KR_ASTATUS})
            t.eq(t.reg(a), 0, strf("reg 0x%03x after reset", a));
        t.ok(!d.irq(), "interrupt low after reset");
        for (uint32_t a : {TSR_CSR_STATUS, TSR_CSR_CMD_ADDR_LO, TSR_CSR_CMD_COUNT, TSR_CSR_ARG0, TSR_CSR_PC})
            t.eq(t.csr(a), 0, strf("core 0x%03x after reset", a));
        t.eq(t.perf(TSR_PERF_CYCLES), 0, "PERF_CYCLES after reset");
        t.steps(100);        // memory inactive: any AXI request is a memmodel P6 error
        t.harvest("after reset");
        // a fresh run passes every replay check
        if (!t.opt.traces.empty()) {
            const std::string& path = t.opt.traces[it % t.opt.traces.size()];
            fr::Regime reg;
            fr::Regime::by_name("rand", &reg);
            const RunResult rr = run_trace(d, t.opt, path, reg, t.opt.seed * 7 + it);
            for (const auto& e : rr.errors) t.fail(strf("fresh run %s after reset %u: %s", path.c_str(), it, e.c_str()));
        } else {
            const uint32_t n = 128, reps = 2;
            setup(t, prog_copy(n, reps), t.opt.seed + 200 + it, n, reps);
            start_manual(t, (uint32_t)prog_copy(n, reps).size());
            wait_done(t, 200000, "copy after reset");
            end_manual(t);
            check_copy(t, n, reps, "copy after reset");
            t.eq(t.csr(TSR_CSR_STATUS), 2, "STATUS of the copy after reset");
        }
        t.harvest(strf("fresh run after reset %u", it).c_str());
    }
}

int directed_main(int argc, char** argv) {
    std::string which;
    std::vector<char*> rest;
    for (int i = 0; i < argc; ++i) {
        if (std::string(argv[i]) == "--directed" && i + 1 < argc) {
            which = argv[++i];
            continue;
        }
        rest.push_back(argv[i]);
    }
    Options opt;
    if (!parse_args((int)rest.size(), rest.data(), &opt)) return 2;
    // workload traces must have this build's configuration (as run_trace requires)
    for (const auto& path : opt.traces) {
        tsr_trace tr;
        if (tsr_trace_read(path.c_str(), &tr) != 0) {
            fprintf(stderr, "kernel bench: cannot read trace %s\n", path.c_str());
            return 2;
        }
        const tsr_config& k = tr.cfg;
        const bool same = k.num_ch == FR_NUM_CH && k.t_max == FR_T_MAX && k.k_max == FR_K_MAX &&
                          k.spm_bytes == FR_SPM_BYTES && k.ch_bytes == (uint64_t)FR_CH_BYTES &&
                          k.addr_beat_w == FR_ADDR_BEAT_W && k.rd_outstanding == FR_CFG_RD_OUTSTANDING;
        tsr_trace_free(&tr);
        if (!same) {
            fprintf(stderr, "kernel bench: trace %s: configuration differs from this build (" FR_BUILD_DESC ")\n",
                    path.c_str());
            return 2;
        }
    }
    struct Test {
        const char* name;
        void (*fn)(T&);
    };
    const Test tests[] = {{"regmap", t_regmap}, {"apctrl", t_apctrl}, {"window", t_window},
                          {"errors", t_errors}, {"misaligned", t_misaligned}, {"reset", t_reset}};
    KernelDut d(opt, (int)rest.size(), rest.data());
    d.hw_reset();
    {
        const std::string e = d.check_constants();
        if (!e.empty()) {
            fprintf(stderr, "kernel bench: build check failed: %s\n", e.c_str());
            return 100;
        }
    }
    FILE* out = opt.out.empty() ? nullptr : fopen(opt.out.c_str(), "w");
    int failed = 0, ran = 0;
    for (const Test& tc : tests) {
        if (which != "all" && ("," + which + ",").find(std::string(",") + tc.name + ",") == std::string::npos) continue;
        ++ran;
        T t{d, opt, {}, 0};
        const double w0 = now_s();
        const uint64_t c0 = d.cycles();
        tc.fn(t);
        t.harvest(tc.name);
        const bool ok = t.errs.empty();
        failed += !ok;
        fprintf(stderr, "%s %-10s %llu checks, %llu cycles, %.1f s%s\n", ok ? "PASS" : "FAIL", tc.name,
                (unsigned long long)t.checks, (unsigned long long)(d.cycles() - c0), now_s() - w0, ok ? "" : ":");
        for (size_t i = 0; i < t.errs.size() && i < 20; ++i) fprintf(stderr, "        %s\n", t.errs[i].c_str());
        if (out) {
            std::string j = strf("{\"directed\": %s, \"ok\": %s, \"checks\": %llu, \"cycles\": %llu, \"errors\": [",
                                 json_str(tc.name).c_str(), ok ? "true" : "false", (unsigned long long)t.checks,
                                 (unsigned long long)(d.cycles() - c0));
            for (size_t i = 0; i < t.errs.size(); ++i) j += (i ? ", " : "") + json_str(t.errs[i]);
            fputs((j + "]}\n").c_str(), out);
        }
        if (!ok) d.hw_reset();
    }
    if (out) fclose(out);
    if (!ran) {
        fprintf(stderr, "kernel bench: no directed test matches %s\n", which.c_str());
        return 2;
    }
    fprintf(stderr, "kernel bench: %d directed test(s), %d failed (" FR_BUILD_DESC ")\n", ran, failed);
    return failed;
}

}  // namespace

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--directed") return directed_main(argc, argv);
    return replay_main<KernelDut>(argc, argv);
}
