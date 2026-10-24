// fr_common.h — the DUT-independent part of the fast trace replayers: options, the program
// analysis of a trace (derive), the replay of one trace with every check of fastreplay.cpp's
// header (run_trace) and the main loop (replay_main). Included once per binary, after
// fr_config.h and the Verilated model's header:
//   tb/fastreplay/fastreplay.cpp     DUT = tsr_core (CoreDut: CSR port + memory ports)
//   tb/platform/kernel_bench.cpp     DUT = the XRT kernel (tsr_xrt_kernel or a packaged top behind
//                                    AXI4-Lite / AXI4; tb/platform/kernel_dut.h)
// A DUT class provides (see CoreDut in fastreplay.cpp):
//   Dut(const Options&, int argc, char** argv)        model, memory configuration, inputs idle
//   void hw_reset()                                    hardware reset (memory drops what is in flight)
//   std::string check_constants()                      build constants vs CSRs ("" = ok)
//   uint64_t cycles() const                            simulated cycles so far
//   void step()                                        one clock cycle
//   uint32_t csr(uint32_t addr, bool we, uint32_t wdata, bool* ok)   one core CSR transfer
//   fr::MemModel& mem()                                the channel memories
//   void prepare(const fr::Regime&, uint64_t seed)     fresh memory for a trace (mem().prepare + DUT state)
//   void start(const tsr_trace&, bool* ok)             program CMD_ADDR / CMD_COUNT / ARG* and start
//   bool running() const, bool finished() const        run in progress / run ended with done
//   uint64_t start_cyc() const, done_cyc() const       cycle of the start / of done
//   std::vector<std::string>& done_errs()              errors found at done (quiescence)
//   void perf_window(uint64_t cycles, uint64_t* lo, uint64_t* hi)   PERF_CYCLES must lie in [lo, hi]
//   void post_run(const tsr_trace&, RunResult&, bool timed_out)     DUT-specific checks of the run
//   static const char* kind()                          "core" / "kernel" (messages)
#pragma once

#include <verilated.h>

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "memmodel.h"
#include "tsr_isa.h"
#include "tsr_ref.h"
#include "tsr_trace.h"

#ifndef VL_USER_STOP
#error "build with -DVL_USER_STOP (run.py does): RTL $error/$stop must not abort the replay"
#endif

// ---------------------------------------------------------------------------------------------
// RTL $stop / $error / failed assertions: recorded, the run continues (the trace fails)
// ---------------------------------------------------------------------------------------------
static std::vector<std::string> g_rtl_errors;
static uint64_t g_rtl_error_count = 0;

void vl_stop(const char* filename, int linenum, const char* hier) VL_MT_UNSAFE {
    ++g_rtl_error_count;
    if (g_rtl_errors.size() < 8) {
        char b[600];
        snprintf(b, sizeof b, "RTL $stop/$error at %s:%d%s%s (message in the process log)", filename ? filename : "?",
                 linenum, hier && *hier ? " in " : "", hier ? hier : "");
        g_rtl_errors.push_back(b);
    }
    fflush(stdout);
}

namespace {

using fr::ChIn;
using fr::ChOut;

constexpr unsigned NUM_CH = FR_NUM_CH;
constexpr unsigned AW = FR_ADDR_BEAT_W;
constexpr unsigned LEN_W = 7;
constexpr unsigned ID_W = 4;

static_assert(NUM_CH >= 1 && NUM_CH <= 16, "NUM_CH");

// ---------------------------------------------------------------------------------------------
// Bit access on Verilator port types (CData/SData/IData/QData or VlWide<N>)
// ---------------------------------------------------------------------------------------------
inline uint64_t mask64(unsigned w) { return w >= 64 ? ~0ull : ((1ull << w) - 1); }

template <typename T>
inline uint64_t getb(const T& v, unsigned lsb, unsigned w) {
    if constexpr (VlIsVlWide<T>::value) {
        const unsigned wi = lsb >> 5, sh = lsb & 31;
        unsigned __int128 acc = 0;
        for (unsigned k = 0; k < 3 && wi + k < T::Words; ++k) acc |= (unsigned __int128)v.at(wi + k) << (32 * k);
        return (uint64_t)(acc >> sh) & mask64(w);
    } else {
        return lsb >= 64 ? 0 : ((uint64_t)v >> lsb) & mask64(w);
    }
}

template <typename T>
inline void setb(T& v, unsigned lsb, unsigned w, uint64_t val) {
    if constexpr (VlIsVlWide<T>::value) {
        unsigned i = 0;
        while (i < w) {
            const unsigned bit = lsb + i, wi = bit >> 5, sh = bit & 31;
            const unsigned n = std::min(32u - sh, w - i);
            const uint32_t m = (n == 32 ? 0xFFFFFFFFu : ((1u << n) - 1)) << sh;
            v.at(wi) = (v.at(wi) & ~m) | (((uint32_t)(val >> i) << sh) & m);
            i += n;
        }
    } else {
        const uint64_t m = mask64(w) << lsb;
        v = (T)(((uint64_t)v & ~m) | ((val << lsb) & m));
    }
}

template <typename T>
inline uint32_t* wide_words(T& v) {
    static_assert(VlIsVlWide<T>::value, "512-bit data ports are wide");
    return v.data();
}
template <typename T>
inline const uint32_t* wide_words_c(const T& v) {
    static_assert(VlIsVlWide<T>::value, "512-bit data ports are wide");
    return v.data();
}

// ---------------------------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------------------------
std::string strf(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
std::string strf(const char* fmt, ...) {
    char b[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(b, sizeof b, fmt, ap);
    va_end(ap);
    return b;
}

std::string json_str(const std::string& s) {
    std::string o = "\"";
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') o += '\\', o += (char)c;
        else if (c == '\n') o += "\\n";
        else if (c < 0x20) o += strf("\\u%04x", c);
        else o += (char)c;
    }
    return o + "\"";
}

uint64_t fnv1a(const std::string& s) {
    uint64_t h = 0xCBF29CE484222325ull;
    for (unsigned char c : s) h = (h ^ c) * 0x100000001B3ull;
    return h;
}

// Last two path components ("set/trace"): stable trace identity for seeds.
std::string trace_key(const std::string& path) {
    std::string p = path;
    while (p.size() > 1 && p.back() == '/') p.pop_back();
    const size_t a = p.rfind('/');
    if (a == std::string::npos || a == 0) return p;
    const size_t b = p.rfind('/', a - 1);
    return b == std::string::npos ? p : p.substr(b + 1);
}

double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

uint32_t insn_word(const tsr_trace& tr, uint32_t pc, unsigned i) {
    const uint8_t* p = tr.cmd + (size_t)pc * TSR_INSN_BYTES + 4 * i;
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

// ---------------------------------------------------------------------------------------------
// What the program does (derived from the trace alone; tb/common/trace.py, replay.py)
// ---------------------------------------------------------------------------------------------
struct Derived {
    uint32_t executed = 0;              // instructions that executed successfully
    uint64_t dma[16] = {0};             // LD beats per channel (source DMA)
    uint64_t ws[16] = {0};              // MVM weight beats per channel (source WS)
    uint32_t mvms = 0;                  // executed MVMs
    uint64_t est_cycles = 0;            // replay.estimate_cycles
};

Derived derive(const tsr_trace& tr) {
    Derived d;
    const unsigned C = tr.cfg.num_ch;
    if (tr.err_code) {
        d.executed = std::min(tr.err_pc, tr.cmd_count);
    } else {
        d.executed = tr.cmd_count;
        for (uint32_t pc = 0; pc < tr.cmd_count; ++pc) {
            if ((insn_word(tr, pc, 0) & 0xFF) == TSR_OP_END) {
                d.executed = pc + 1;
                break;
            }
        }
    }
    uint64_t cyc = 20000 + 200ull * tr.cmd_count;
    for (uint32_t pc = 0; pc < d.executed; ++pc) {
        const uint32_t w0 = insn_word(tr, pc, 0), op = w0 & 0xFF;
        if (op == TSR_OP_LD || op == TSR_OP_ST) {
            const uint64_t n = insn_word(tr, pc, 3);
            if (op == TSR_OP_LD) d.dma[(w0 >> 8) & 0xF] += n;
            const uint64_t m = std::min<uint64_t>(n, 1u << 20);
            cyc += 64 * m / 8 + 8 * m;
        } else if (op == TSR_OP_QACT) {
            const uint64_t K = std::min<uint64_t>(insn_word(tr, pc, 3), 65280), T = std::min<uint64_t>(insn_word(tr, pc, 4), 8);
            cyc += T * (K / 16 + (K / 32) * 120 + 400);
        } else if (op == TSR_OP_MVM) {
            ++d.mvms;
            const uint32_t w4 = insn_word(tr, pc, 4), w5 = insn_word(tr, pc, 5);
            const uint64_t N = w4, K = w5 & 0xFFFF;
            const uint32_t fmtf = (w0 >> 8) & 0xF;
            const uint64_t rb = tsr_row_beats(fmtf, (uint32_t)K);
            const uint64_t R = (N + C - 1) / C;
            for (unsigned c = 0; c < C && c < 16; ++c) {
                const uint64_t first = (uint64_t)c * R;
                const uint64_t rows = first >= N ? 0 : std::min<uint64_t>(R, N - first);
                d.ws[c] += rows * rb;
            }
            // budget (replay.estimate_cycles)
            const uint32_t fmt = (w0 >> 8) & 1;
            const uint64_t Kc = std::min<uint64_t>(K, 65280), T = std::min<uint64_t>((w5 >> 16) & 0xF, 8);
            const uint64_t bs = fmt ? 256 : 32;
            if (Kc == 0 || Kc % bs || T == 0) continue;
            const uint64_t Nc = std::min<uint64_t>(N, 1u << 22), Rc = (Nc + C - 1) / C;
            const uint64_t rbc = tsr_row_beats(fmt, (uint32_t)Kc);
            cyc += 4000 + 8 * T * tsr_qa_lines(fmt, (uint32_t)Kc) + Rc * (24 * rbc + 200) + 40 * T * (Nc / 16 + 1);
        }
    }
    d.est_cycles = cyc;
    return d;
}

// ---------------------------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------------------------
struct Options {
    std::vector<std::string> traces;
    std::vector<std::string> regimes = {"rand"};
    unsigned rotate = 0;                 // > 0: passes per trace rotating the regimes
    uint64_t seed = 1;
    bool have_run_seed = false;
    uint64_t run_seed = 0;
    std::vector<std::string> mem_overrides;
    std::string out;
    double timeout_mult = 1.0;
    uint64_t max_cycles = 0;             // > 0: cap of the per-trace cycle budget
    bool reset_each = false;
    bool check_sim = false;
    bool derive_only = false;            // no simulation: derived beats vs the trace's perf record
    std::string vcd;
    std::string tag;
    bool stop_on_fail = false;
    bool verbose = false;
    unsigned progress_s = 30;
    std::vector<std::string> dut_args;   // --dut KEY=VALUE: DUT-specific options (kernel bench)
};

void usage() {
    fprintf(stderr,
            "usage: fastreplay [options] TRACE_DIR...\n"
            "  --list FILE          more trace directories, one per line\n"
            "  --regimes a,b        memory regimes per trace (rand fast slow ordered; all = the four)\n"
            "  --rotate P           P passes per trace, regimes rotating per trace (tb/core style)\n"
            "  --seed N             base seed (run seed = f(seed, set/trace name, regime))\n"
            "  --run-seed N         use N as the memory seed of every run (reproduce one run)\n"
            "  --mem key=value      memory model override (rd_lat=lo:hi, p_rd_rsp_valid=0.5, ...)\n"
            "  --out FILE           JSON line per run\n"
            "  --timeout-mult X     scale the per-trace cycle budget\n"
            "  --max-cycles N       cap the per-trace cycle budget (START .. done)\n"
            "  --reset-each         hardware reset before every trace\n"
            "  --check-sim          also replay each trace on the ISA simulator (tsr_trace_replay)\n"
            "  --derive-only        no simulation: check the per-channel LD + MVM weight beats derived\n"
            "                       from the program against the trace's perf record\n"
            "  --vcd FILE           dump waves (build with run.py --waves)\n"
            "  --dut KEY=VALUE      DUT-specific option (kernel bench: see tb/platform/kernel_dut.h)\n"
            "  --stop-on-fail, -v, --tag STR, --progress SECONDS\n");
}

bool parse_args(int argc, char** argv, Options* o) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto need = [&](const char* what) -> const char* {
            if (i + 1 >= argc) {
                fprintf(stderr, "fastreplay: %s needs a value\n", what);
                exit(2);
            }
            return argv[++i];
        };
        if (a == "--list") {
            std::ifstream f(need("--list"));
            std::string line;
            while (std::getline(f, line))
                if (!line.empty()) o->traces.push_back(line);
        } else if (a == "--regimes") {
            o->regimes.clear();
            std::string s = need("--regimes");
            size_t p = 0;
            while (p <= s.size()) {
                const size_t q = s.find(',', p);
                const std::string r = s.substr(p, q == std::string::npos ? std::string::npos : q - p);
                if (r == "all") {
                    for (const char* const* n = fr::Regime::names(); *n; ++n) o->regimes.push_back(*n);
                } else if (!r.empty()) {
                    o->regimes.push_back(r);
                }
                if (q == std::string::npos) break;
                p = q + 1;
            }
        } else if (a == "--rotate") {
            o->rotate = (unsigned)strtoul(need("--rotate"), nullptr, 0);
        } else if (a == "--seed") {
            o->seed = strtoull(need("--seed"), nullptr, 0);
        } else if (a == "--run-seed") {
            o->have_run_seed = true;
            o->run_seed = strtoull(need("--run-seed"), nullptr, 0);
        } else if (a == "--mem") {
            o->mem_overrides.push_back(need("--mem"));
        } else if (a == "--out") {
            o->out = need("--out");
        } else if (a == "--timeout-mult") {
            o->timeout_mult = atof(need("--timeout-mult"));
        } else if (a == "--max-cycles") {
            o->max_cycles = strtoull(need("--max-cycles"), nullptr, 0);
        } else if (a == "--reset-each") {
            o->reset_each = true;
        } else if (a == "--check-sim") {
            o->check_sim = true;
        } else if (a == "--derive-only") {
            o->derive_only = true;
        } else if (a == "--vcd") {
            o->vcd = need("--vcd");
        } else if (a == "--dut") {
            o->dut_args.push_back(need("--dut"));
        } else if (a == "--tag") {
            o->tag = need("--tag");
        } else if (a == "--stop-on-fail") {
            o->stop_on_fail = true;
        } else if (a == "--progress") {
            o->progress_s = (unsigned)strtoul(need("--progress"), nullptr, 0);
        } else if (a == "-v") {
            o->verbose = true;
        } else if (a == "-h" || a == "--help") {
            usage();
            exit(0);
        } else if (!a.empty() && a[0] == '+') {
            // Verilator plusargs (handled by commandArgs)
        } else if (!a.empty() && a[0] == '-') {
            fprintf(stderr, "fastreplay: unknown option %s\n", a.c_str());
            usage();
            return false;
        } else {
            o->traces.push_back(a);
        }
    }
    for (const auto& r : o->regimes) {
        fr::Regime tmp;
        if (!fr::Regime::by_name(r, &tmp)) {
            fprintf(stderr, "fastreplay: unknown regime %s\n", r.c_str());
            return false;
        }
    }
    if (o->regimes.empty()) o->regimes = {"rand"};
    return true;
}

// ---------------------------------------------------------------------------------------------
// One run
// ---------------------------------------------------------------------------------------------
constexpr uint32_t CSR_TIMEOUT = 2000;
constexpr unsigned N_PERF = TSR_PERF_CH_RD_BEATS + 2 * NUM_CH;

struct RunResult {
    bool ok = true;
    std::vector<std::string> errors;
    uint64_t cycles = 0;          // START handshake .. first cycle of done
    uint64_t total_cycles = 0;    // every simulated cycle of this run (CSR traffic included)
    double wall = 0;              // seconds
    uint32_t status = 0, err_code = 0, err_pc = 0, err_info = 0;
    uint64_t port_beats[16] = {0}, ws_beats[16] = {0}, dma_beats[16] = {0};
    uint64_t mvm_cycles = 0, perf_cycles = 0;
    unsigned max_out = 0;
    std::string extra_json;       // DUT-specific fields (", \"k\": v" ...)
    void err(const std::string& s) {
        ok = false;
        if (errors.size() < 24) errors.push_back(s);
    }
};

// Constants of the build vs the CSRs (ID, VERSION, CFG0..2, CAPS). Returns error text or "".
template <class Dut>
std::string check_core_constants(Dut& dut) {
    std::string e;
    auto chk = [&](uint32_t addr, uint32_t want, const char* name) {
        bool ok = true;
        const uint32_t got = dut.csr(addr, false, 0, &ok);
        if (!ok || got != want) e += strf("%s = 0x%08x, expected 0x%08x; ", name, got, want);
    };
    chk(TSR_CSR_ID, TSR_ID_MAGIC, "ID");
    chk(TSR_CSR_VERSION, (TSR_ISA_MAJOR << 8) | TSR_ISA_MINOR, "VERSION");
    chk(TSR_CSR_CFG0, FR_NUM_CH | (FR_LANES << 8) | (FR_T_MAX << 20), "CFG0");
    chk(TSR_CSR_CFG1, FR_SPM_BYTES / 64, "CFG1");
    chk(TSR_CSR_CFG2, FR_VL | (FR_NSREG << 8) | ((FR_K_MAX / 256) << 16), "CFG2");
    chk(TSR_CSR_CAPS, 0, "CAPS");
    chk(TSR_CSR_STATUS, 0, "STATUS after reset");
    return e;
}

template <class Dut>
bool read_perf(Dut& dut, uint64_t* p) {
    bool ok = true;
    for (unsigned i = 0; i < N_PERF; ++i) {
        const uint32_t a = TSR_CSR_PERF_BASE + 8 * i;
        const uint64_t lo = dut.csr(a, false, 0, &ok);
        const uint64_t hi = dut.csr(a + 4, false, 0, &ok);
        p[i] = (hi << 32) | lo;
    }
    return ok;
}

template <class Dut>
RunResult run_trace(Dut& dut, const Options& opt, const std::string& dir, const fr::Regime& reg, uint64_t seed) {
    RunResult r;
    const double t0 = now_s();
    const uint64_t c0 = dut.cycles();
    tsr_trace tr;
    if (tsr_trace_read(dir.c_str(), &tr) != 0) {
        r.err("cannot read the trace directory");
        return r;
    }
    // ---- configuration of the build ----
    const tsr_config& k = tr.cfg;
    if (k.num_ch != FR_NUM_CH || k.beat_bits != FR_BEAT_BITS || k.lanes != FR_LANES || k.t_max != FR_T_MAX ||
        k.k_max != FR_K_MAX || k.spm_bytes != FR_SPM_BYTES || k.vl != FR_VL || k.nireg != FR_NIREG ||
        k.nsreg != FR_NSREG || k.loop_depth != FR_LOOP_DEPTH || k.rd_outstanding != FR_CFG_RD_OUTSTANDING ||
        k.ch_bytes != (uint64_t)FR_CH_BYTES || k.addr_beat_w != FR_ADDR_BEAT_W || k.caps != 0) {
        r.err("trace configuration differs from this build (" FR_BUILD_DESC ")");
        tsr_trace_free(&tr);
        return r;
    }
    const Derived d = derive(tr);
    if (opt.derive_only) {
        // Channel c >= 1: perf beats = LD + MVM beats exactly; channel 0 also counts fetch beats.
        if (!tr.perf.valid) r.err("trace has no perf record");
        for (unsigned c = 0; c < NUM_CH && tr.perf.valid; ++c) {
            const uint64_t want = d.dma[c] + d.ws[c], got = tr.perf.ch_rd_beats[c];
            if (c >= 1 ? got != want : got < want)
                r.err(strf("ch%u: LD + MVM beats derived %llu (LD %llu, MVM %llu), trace perf %llu", c,
                           (unsigned long long)want, (unsigned long long)d.dma[c], (unsigned long long)d.ws[c],
                           (unsigned long long)got));
            r.ws_beats[c] = d.ws[c];
            r.dma_beats[c] = d.dma[c];
        }
        tsr_trace_free(&tr);
        r.wall = now_s() - t0;
        return r;
    }
    if (opt.check_sim) {
        if (tsr_trace_replay(&tr, 0) != 0) r.err("the ISA simulator does not replay this trace (tsr_trace_replay)");
    }
    // ---- memory ----
    fr::MemModel& mem = dut.mem();
    dut.prepare(reg, seed);
    const uint64_t ch_beats = FR_CH_BYTES / 64;
    if (tr.cmd != nullptr && tr.cmd_count > 0 && tr.cmd_addr_beat < ch_beats) {
        const uint64_t lo = tr.cmd_addr_beat * 64;
        const uint64_t n = std::min<uint64_t>((uint64_t)tr.cmd_count * TSR_INSN_BYTES, FR_CH_BYTES - lo);
        mem.load(0, lo, tr.cmd, n);
        mem.allow_beats(0, tr.cmd_addr_beat, std::min<uint64_t>(tr.cmd_addr_beat + (n + 63) / 64, ch_beats));
    }
    for (size_t i = 0; i < tr.cap.n_init; ++i) {
        const tsr_region& g = tr.cap.init[i];
        mem.load(g.ch, g.byte_off, g.data, g.size);
        mem.allow_beats(g.ch, g.byte_off / 64, (g.byte_off + g.size) / 64);
    }
    for (size_t i = 0; i < tr.cap.n_expect; ++i) {
        const tsr_region& g = tr.cap.expect[i];
        mem.expect_beats(g.ch, g.byte_off / 64, (g.byte_off + g.size) / 64);
    }
    g_rtl_errors.clear();
    g_rtl_error_count = 0;
    dut.done_errs().clear();
    if (opt.reset_each) dut.hw_reset();
    // ---- run ----
    uint64_t p0[N_PERF], p1[N_PERF];
    bool csr_ok = read_perf(dut, p0);
    dut.start(tr, &csr_ok);
    uint64_t budget = (uint64_t)((double)d.est_cycles * opt.timeout_mult) + 1000;
    if (opt.max_cycles && budget > opt.max_cycles) budget = opt.max_cycles;
    while (dut.running() && dut.cycles() - dut.start_cyc() <= budget) dut.step();
    // timed_out: no START handshake, or no done within the budget; the core is then reset
    bool timed_out = false;
    if (!csr_ok || !dut.finished()) {
        timed_out = true;
        if (!csr_ok) r.err("CSR transfer timed out while starting the run");
        else r.err(strf("no done within %llu cycles (regime %s): %s", (unsigned long long)budget, reg.name.c_str(),
                        mem.idle() ? "memory idle" : mem.busy_desc().c_str()));
    }
    r.cycles = dut.finished() ? dut.done_cyc() - dut.start_cyc() : dut.cycles() - dut.start_cyc();
    for (const auto& e : dut.done_errs()) r.err(e);
    if (!timed_out) {
        bool ok = true;
        r.status = dut.csr(TSR_CSR_STATUS, false, 0, &ok);
        r.err_code = dut.csr(TSR_CSR_ERR_CODE, false, 0, &ok);
        r.err_pc = dut.csr(TSR_CSR_ERR_PC, false, 0, &ok);
        r.err_info = dut.csr(TSR_CSR_ERR_INFO, false, 0, &ok);
        ok &= read_perf(dut, p1);
        if (!ok) r.err("CSR transfer timed out after the run");
        // ---- final state ----
        if (r.status != tr.status) r.err(strf("STATUS = 0x%x, expected 0x%x", r.status, tr.status));
        if (r.err_code != tr.err_code) r.err(strf("ERR_CODE = %u, expected %u", r.err_code, tr.err_code));
        if (r.err_pc != tr.err_pc) r.err(strf("ERR_PC = %u, expected %u", r.err_pc, tr.err_pc));
        if (tr.has_err_info && r.err_info != tr.err_info)
            r.err(strf("ERR_INFO = 0x%x, expected 0x%x", r.err_info, tr.err_info));
        // ---- PERF deltas ----
        uint64_t dp[N_PERF];
        for (unsigned i = 0; i < N_PERF; ++i) dp[i] = p1[i] - p0[i];
        const uint64_t pc_cyc = dp[TSR_PERF_CYCLES];
        r.perf_cycles = pc_cyc;
        r.mvm_cycles = dp[TSR_PERF_MVM_CYCLES];
        if (tr.perf.valid && dp[TSR_PERF_RETIRED] != tr.perf.retired)
            r.err(strf("PERF_RETIRED %llu, trace %llu", (unsigned long long)dp[TSR_PERF_RETIRED],
                       (unsigned long long)tr.perf.retired));
        for (unsigned c = 0; c < NUM_CH; ++c) {
            const uint64_t got = dp[TSR_PERF_CH_RD_BEATS + c];
            const fr::MemModel::Stats& s = mem.stats(c);
            if (c >= 1 && tr.perf.valid && got != tr.perf.ch_rd_beats[c])
                r.err(strf("PERF_CH_RD_BEATS[%u] %llu, trace %llu", c, (unsigned long long)got,
                           (unsigned long long)tr.perf.ch_rd_beats[c]));
            if (got != s.rd_beats)
                r.err(strf("PERF_CH_RD_BEATS[%u] %llu, port beats %llu", c, (unsigned long long)got,
                           (unsigned long long)s.rd_beats));
            if (dp[TSR_PERF_CH_RD_BEATS + NUM_CH + c] > pc_cyc) r.err(strf("PERF_CH_WAIT[%u] > PERF_CYCLES", c));
        }
        if (dp[TSR_PERF_VU_CYCLES] || dp[TSR_PERF_WAITM_CYCLES])
            r.err(strf("PERF_VU_CYCLES %llu / PERF_WAITM_CYCLES %llu (no VU / WAITM)",
                       (unsigned long long)dp[TSR_PERF_VU_CYCLES], (unsigned long long)dp[TSR_PERF_WAITM_CYCLES]));
        if (d.mvms == 0 && r.mvm_cycles) r.err(strf("PERF_MVM_CYCLES %llu without an executed MVM", (unsigned long long)r.mvm_cycles));
        if (FR_HAS_MVM && d.mvms > 0 && r.mvm_cycles == 0) r.err("PERF_MVM_CYCLES 0 although MVMs executed");
        if (r.mvm_cycles > pc_cyc) r.err("PERF_MVM_CYCLES > PERF_CYCLES");
        uint64_t lo = 0, hi = 0;
        dut.perf_window(r.cycles, &lo, &hi);
        if (pc_cyc < lo || pc_cyc > hi)
            r.err(strf("PERF_CYCLES %llu vs %llu cycles from START to done (allowed [%llu, %llu])",
                       (unsigned long long)pc_cyc, (unsigned long long)r.cycles, (unsigned long long)lo,
                       (unsigned long long)hi));
    }
    // ---- memory: protocol, sources, reads, writes, expect data ----
    for (const auto& e : mem.errors()) r.err(e);
    if (mem.n_errors() > mem.errors().size())
        r.err(strf("%llu memory protocol violations in total", (unsigned long long)mem.n_errors()));
    for (unsigned c = 0; c < NUM_CH; ++c) {
        const fr::MemModel::Stats& s = mem.stats(c);
        r.port_beats[c] = s.rd_beats;
        r.ws_beats[c] = s.src_beats[fr::SRC_WS];
        r.dma_beats[c] = s.src_beats[fr::SRC_DMA];
        r.max_out = std::max(r.max_out, s.max_outstanding);
        if (!timed_out) {
            if (s.src_beats[fr::SRC_DMA] != d.dma[c])
                r.err(strf("ch%u: %llu read beats for source DMA, expected %llu (LD)", c,
                           (unsigned long long)s.src_beats[fr::SRC_DMA], (unsigned long long)d.dma[c]));
            if (s.src_beats[fr::SRC_WS] != d.ws[c])
                r.err(strf("ch%u: %llu read beats for source WS, expected %llu (MVM rows_c x ROW_BEATS)", c,
                           (unsigned long long)s.src_beats[fr::SRC_WS], (unsigned long long)d.ws[c]));
            if (c >= 1 && tr.perf.valid && d.dma[c] + d.ws[c] != tr.perf.ch_rd_beats[c])
                r.err(strf("harness: ch%u LD + MVM beats derived from the program %llu != trace perf %llu", c,
                           (unsigned long long)(d.dma[c] + d.ws[c]), (unsigned long long)tr.perf.ch_rd_beats[c]));
        }
        if (s.bad_reads)
            r.err(strf("ch%u: read %llu beat(s) the trace does not initialise, first 0x%llx", c,
                       (unsigned long long)s.bad_reads, (unsigned long long)s.first_bad_read));
        if (s.extra_writes)
            r.err(strf("ch%u: %llu beat(s) written outside the expect regions, first beat 0x%llx", c,
                       (unsigned long long)s.extra_writes, (unsigned long long)s.first_extra_write));
    }
    if (!timed_out) {
        for (size_t i = 0; i < tr.cap.n_expect; ++i) {
            const tsr_region& g = tr.cap.expect[i];
            uint64_t missing = 0, first_missing = 0;
            for (uint64_t b = g.byte_off / 64; b < (g.byte_off + g.size) / 64; ++b) {
                if (!mem.written(g.ch, b) && missing++ == 0) first_missing = b;
            }
            if (missing)
                r.err(strf("ch%u: %llu expected beat(s) not written, first beat 0x%llx", g.ch, (unsigned long long)missing,
                           (unsigned long long)first_missing));
            const uint8_t* m = mem.mem(g.ch) + g.byte_off;
            if (memcmp(m, g.data, g.size) != 0) {
                uint64_t i0 = 0;
                while (m[i0] == g.data[i0]) ++i0;
                uint64_t nd = 0;
                for (uint64_t j = i0; j < g.size; ++j) nd += m[j] != g.data[j];
                r.err(strf("expect mismatch ch%u byte 0x%llx: got 0x%02x expected 0x%02x (%llu bytes differ)", g.ch,
                           (unsigned long long)(g.byte_off + i0), m[i0], g.data[i0], (unsigned long long)nd));
            }
        }
    }
    dut.post_run(tr, r, timed_out);
    for (const auto& e : g_rtl_errors) r.err(e);
    if (g_rtl_error_count > g_rtl_errors.size())
        r.err(strf("%llu RTL errors in total", (unsigned long long)g_rtl_error_count));
    // A failing run may leave the core in any state (busy, RTL assertion fired, units out of
    // step): reset it so that every later trace starts clean and failures do not cascade.
    if (!r.ok) dut.hw_reset();
    mem.release();
    tsr_trace_free(&tr);
    r.total_cycles = dut.cycles() - c0;
    r.wall = now_s() - t0;
    return r;
}

// ---------------------------------------------------------------------------------------------
// Main loop: every trace x regime (or rotation), JSON line per run, summary
// ---------------------------------------------------------------------------------------------
template <class Dut>
int replay_main(int argc, char** argv) {
    Options opt;
    if (!parse_args(argc, argv, &opt)) return 2;
    if (opt.traces.empty()) {
        fprintf(stderr, "fastreplay: no trace directory given\n");
        usage();
        return 2;
    }
    std::vector<fr::Regime> regimes;
    for (const auto& n : opt.regimes) {
        fr::Regime r;
        fr::Regime::by_name(n, &r);
        for (const auto& kv : opt.mem_overrides) {
            if (Dut::mem_key(kv)) continue;          // a DUT-side memory option (kernel bench)
            const std::string e = r.set(kv);
            if (!e.empty()) {
                fprintf(stderr, "fastreplay: --mem %s: %s\n", kv.c_str(), e.c_str());
                return 2;
            }
        }
        regimes.push_back(r);
    }
    FILE* out = nullptr;
    if (!opt.out.empty()) {
        out = fopen(opt.out.c_str(), "w");
        if (out == nullptr) {
            fprintf(stderr, "fastreplay: cannot create %s\n", opt.out.c_str());
            return 2;
        }
    }
    const std::string tag = opt.tag.empty() ? "" : "[" + opt.tag + "] ";
    const double t_start = now_s();
    Dut bench(opt, argc, argv);
    bench.hw_reset();
    {
        const std::string e = bench.check_constants();
        if (!e.empty() || !g_rtl_errors.empty()) {
            fprintf(stderr, "%sfastreplay: build check failed: %s%s\n", tag.c_str(), e.c_str(),
                    g_rtl_errors.empty() ? "" : g_rtl_errors[0].c_str());
            if (out) {
                fprintf(out, "{\"fatal\": %s}\n", json_str("build check failed: " + e).c_str());
                fclose(out);
            }
            return 3;
        }
    }
    uint64_t n_runs = 0, n_fail = 0, sim_cycles = 0, run_cycles = 0;
    double sim_wall = 0, t_progress = now_s();
    const size_t n_reg = regimes.size();
    bool stop = false;
    for (size_t k = 0; k < opt.traces.size() && !stop; ++k) {
        const std::string& path = opt.traces[k];
        const unsigned passes = opt.rotate ? opt.rotate : (unsigned)n_reg;
        for (unsigned ps = 0; ps < passes && !stop; ++ps) {
            // tb/core rotation: regime (k + ps) % n when fewer passes than regimes, else ps % n
            const size_t ri = opt.rotate ? (opt.rotate < n_reg ? (k + ps) % n_reg : ps % n_reg) : ps;
            const fr::Regime& reg = regimes[ri];
            uint64_t s = opt.have_run_seed ? opt.run_seed
                                           : (opt.seed * 1000003ull) ^ fnv1a(trace_key(path)) ^ (0x9E37ull * (ri + 1));
            if (!opt.have_run_seed) s = fr::splitmix64(s) & 0x7FFFFFFFull;
            const RunResult r = run_trace(bench, opt, path, reg, s);
            ++n_runs;
            sim_cycles += r.total_cycles;
            run_cycles += r.cycles;
            sim_wall += r.wall;
            if (!r.ok) {
                ++n_fail;
                fprintf(stderr, "%sFAIL %s (regime %s, run seed %llu): %s\n", tag.c_str(), path.c_str(), reg.name.c_str(),
                        (unsigned long long)s, r.errors.empty() ? "?" : r.errors[0].c_str());
                for (size_t i = 1; i < r.errors.size() && i < 6; ++i) fprintf(stderr, "        %s\n", r.errors[i].c_str());
                if (opt.stop_on_fail) stop = true;
            } else if (opt.verbose) {
                fprintf(stderr, "%sok   %s (%s) %llu cycles, status %u err %u\n", tag.c_str(), path.c_str(),
                        reg.name.c_str(), (unsigned long long)r.cycles, r.status, r.err_code);
            }
            if (out) {
                std::string j = "{\"trace\": " + json_str(path) + ", \"regime\": " + json_str(reg.name) +
                                strf(", \"seed\": %llu, \"ok\": %s, \"cycles\": %llu, \"total_cycles\": %llu, \"wall\": %.6f",
                                     (unsigned long long)s, r.ok ? "true" : "false", (unsigned long long)r.cycles,
                                     (unsigned long long)r.total_cycles, r.wall) +
                                strf(", \"status\": %u, \"err_code\": %u, \"perf_cycles\": %llu, \"mvm_cycles\": %llu, "
                                     "\"max_outstanding\": %u",
                                     r.status, r.err_code, (unsigned long long)r.perf_cycles,
                                     (unsigned long long)r.mvm_cycles, r.max_out);
                j += ", \"port_beats\": [";
                for (unsigned c = 0; c < NUM_CH; ++c) j += strf("%s%llu", c ? ", " : "", (unsigned long long)r.port_beats[c]);
                j += "], \"ws_beats\": [";
                for (unsigned c = 0; c < NUM_CH; ++c) j += strf("%s%llu", c ? ", " : "", (unsigned long long)r.ws_beats[c]);
                j += "]" + r.extra_json + ", \"errors\": [";
                for (size_t i = 0; i < r.errors.size(); ++i) j += (i ? ", " : "") + json_str(r.errors[i]);
                j += "]}\n";
                fputs(j.c_str(), out);
                fflush(out);
            }
            if (opt.progress_s && now_s() - t_progress > opt.progress_s) {
                t_progress = now_s();
                fprintf(stderr, "%s%llu runs (%llu failed), %zu/%zu traces, %.3g cycles/s\n", tag.c_str(),
                        (unsigned long long)n_runs, (unsigned long long)n_fail, k + 1, opt.traces.size(),
                        sim_wall > 0 ? sim_cycles / sim_wall : 0.0);
            }
        }
    }
    const double wall = now_s() - t_start;
    const std::string summ = strf(
        "{\"summary\": {\"runs\": %llu, \"failed\": %llu, \"sim_cycles\": %llu, \"run_cycles\": %llu, \"sim_wall\": %.3f, "
        "\"wall\": %.3f, \"cycles_per_s\": %.1f, \"build\": %s}}\n",
        (unsigned long long)n_runs, (unsigned long long)n_fail, (unsigned long long)sim_cycles,
        (unsigned long long)run_cycles, sim_wall, wall, sim_wall > 0 ? sim_cycles / sim_wall : 0.0,
        json_str(FR_BUILD_DESC).c_str());
    if (out) {
        fputs(summ.c_str(), out);
        fclose(out);
    }
    fprintf(stderr, "%sfastreplay: %llu runs, %llu failed, %llu cycles in %.2f s (%.3g cycles/s)\n", tag.c_str(),
            (unsigned long long)n_runs, (unsigned long long)n_fail, (unsigned long long)sim_cycles, sim_wall,
            sim_wall > 0 ? sim_cycles / sim_wall : 0.0);
    return n_fail ? 1 : 0;
}

}  // namespace
