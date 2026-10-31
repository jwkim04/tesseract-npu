// vu_sweep.cpp — Verilator C++ harness: bit-exact sweeps of the VU arithmetic blocks (rtl/core/tsr_vu_exp.sv,
// tsr_vu_rcp.sv, tsr_vu_sqrt.sv) against libtsr_ref (sw/sim/tsr_ref_math.c tsr_m_expk, sw/sim/tsr_ref.c
// tsr_frecip / tsr_fsqrt). Toplevel: tb/vu/hdl/tsr_vu_arith_tb.sv (N lanes of each block). Built and run by
// tb/vu/test_vu_arith.py; manual use:
//   vu_sweep exp  <kappa 0..3 | 4 = all> <start> <end> <stride>   every pattern u = start + i·stride < end
//   vu_sweep exp_rand <n> <seed>                                    n random patterns per κ (boundary biased)
//   vu_sweep rcp  <exp> [<exp> ...]                                 every mantissa, both signs, at these
//                                                                   exponent fields (0..255)
//   vu_sweep sqrt <exp> [<exp> ...]                                 likewise for SQ
//   vu_sweep rand <n> <seed>                                        random patterns through RC and SQ
//   vu_sweep specials                                               zeros, subnormals, infinities, NaNs,
//                                                                   the FTZ / saturation edges, all blocks
// Exit status 0 iff no mismatch. Every f32 crosses into libtsr_ref as a bit pattern.

#include <verilated.h>

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "Vtsr_vu_arith_tb.h"

extern "C" {
#include "tsr_ref.h"
#include "tsr_ref_math.h"
}

#ifndef VU_N
#define VU_N 16
#endif

static inline float u2f(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static inline uint32_t f2u(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

template <std::size_t W>
static inline void set32(VlWide<W>& w, unsigned i, uint32_t v) { w[i] = v; }
template <std::size_t W>
static inline uint32_t get32(const VlWide<W>& w, unsigned i) { return w[i]; }

static std::unique_ptr<VerilatedContext> ctx;
static std::unique_ptr<Vtsr_vu_arith_tb> top;
static uint64_t edges = 0;

static void tick() {
    top->clk = 0;
    top->eval();
    top->clk = 1;
    top->eval();
    ++edges;
}

struct Unit {
    const char * name;
    unsigned lat;
    std::function<void(unsigned, uint32_t, uint32_t)> set;   // lane, x, aux (κ)
    std::function<uint64_t(unsigned)> get;
    std::function<uint64_t(uint32_t, uint32_t)> ref;
};

struct Exp {
    uint64_t edge;
    unsigned n;
    uint32_t x[VU_N], aux[VU_N];
    uint64_t y[VU_N];
};

static uint64_t g_checked = 0, g_bad = 0;

// Streams the inputs gen() produces (false = done) through unit u, N per cycle.
static void run(const Unit & u, const std::function<bool(uint32_t &, uint32_t &)> & gen) {
    std::deque<Exp> q;
    bool more = true;
    while (more || !q.empty()) {
        Exp e;
        e.n = 0;
        while (more && e.n < VU_N) {
            uint32_t x, aux;
            if (!gen(x, aux)) {
                more = false;
                break;
            }
            e.x[e.n] = x;
            e.aux[e.n] = aux;
            e.y[e.n] = u.ref(x, aux);
            u.set(e.n, x, aux);
            ++e.n;
        }
        for (unsigned i = e.n; i < VU_N; ++i) u.set(i, 0, 0);
        e.edge = edges + 1 + u.lat - 1;      // inputs sampled at the next edge; result LAT - 1 edges later
        if (e.n) q.push_back(e);
        tick();
        while (!q.empty() && q.front().edge == edges) {
            const Exp & f = q.front();
            for (unsigned i = 0; i < f.n; ++i) {
                const uint64_t got = u.get(i);
                ++g_checked;
                if (got != f.y[i]) {
                    if (g_bad < 20)
                        fprintf(stderr, "%s mismatch: x=0x%08" PRIx32 " aux=%u rtl=0x%016" PRIx64 " ref=0x%016" PRIx64 "\n",
                                u.name, f.x[i], f.aux[i], got, f.y[i]);
                    ++g_bad;
                }
            }
            q.pop_front();
        }
    }
}

static Unit unit_exp() {
    return Unit{"expk", top->lat_exp,
                [](unsigned i, uint32_t x, uint32_t k) {
                    set32(top->e_v, i, x);
                    top->e_k = (top->e_k & ~(3u << (2 * i))) | ((k & 3u) << (2 * i));
                },
                [](unsigned i) { return get32(top->e_y, i); },
                [](uint32_t x, uint32_t k) { return f2u(tsr_m_expk(u2f(x), k)); }};
}
static Unit unit_sc() {
    return Unit{"sincos", top->lat_sc, [](unsigned i, uint32_t x, uint32_t) { set32(top->c_t, i, x); },
                [](unsigned i) { return (uint64_t)get32(top->c_s, i) << 32 | get32(top->c_c, i); },
                [](uint32_t x, uint32_t) {
                    float sn, cs;
                    tsr_m_sincos(u2f(x), &sn, &cs);
                    return (uint64_t)f2u(sn) << 32 | f2u(cs);
                }};
}
static Unit unit_rcp() {
    return Unit{"rcp", top->lat_rcp, [](unsigned i, uint32_t x, uint32_t) { set32(top->r_x, i, x); },
                [](unsigned i) { return get32(top->r_y, i); },
                [](uint32_t x, uint32_t) { return f2u(tsr_frecip(u2f(x))); }};
}
static Unit unit_sqrt() {
    return Unit{"sqrt", top->lat_sqrt, [](unsigned i, uint32_t x, uint32_t) { set32(top->s_x, i, x); },
                [](unsigned i) { return get32(top->s_y, i); },
                [](uint32_t x, uint32_t) { return f2u(tsr_fsqrt(u2f(x))); }};
}

static std::vector<uint32_t> special_patterns() {
    std::vector<uint32_t> v = {0x00000000u, 0x80000000u, 0x00000001u, 0x807FFFFFu, 0x007FFFFFu, 0x00800000u,
                               0x80800000u, 0x7F800000u, 0xFF800000u, 0x7FC00000u, 0xFFC00000u, 0x7F800001u,
                               0xFFFFFFFFu, 0x7FBFFFFFu, 0x3F800000u, 0xBF800000u, 0x7F7FFFFFu, 0xFF7FFFFFu,
                               0x3C800000u, 0x3C7FFFFFu, 0x41200000u, 0xC1200000u};
    for (uint32_t e = 0; e < 256; ++e) {          // every exponent with mantissas 0, 1, max
        for (uint32_t s = 0; s < 2; ++s) {
            v.push_back(s << 31 | e << 23);
            v.push_back(s << 31 | e << 23 | 1u);
            v.push_back(s << 31 | e << 23 | 0x7FFFFFu);
            v.push_back(s << 31 | e << 23 | 0x400000u);
        }
    }
    return v;
}

int main(int argc, char ** argv) {
    ctx = std::make_unique<VerilatedContext>();
    ctx->commandArgs(argc, argv);
    top = std::make_unique<Vtsr_vu_arith_tb>(ctx.get());
    top->clk = 0;
    top->rst_n = 0;
    for (int i = 0; i < 4; ++i) tick();
    top->rst_n = 1;
    tick();
    if (argc < 2) {
        fprintf(stderr, "usage: vu_sweep exp|exp_rand|rcp|sqrt|rand|specials ...\n");
        return 2;
    }
    const std::string mode = argv[1];
    if (mode == "exp" && argc >= 6) {
        const unsigned kap = (unsigned)strtoul(argv[2], nullptr, 0);
        const uint64_t a = strtoull(argv[3], nullptr, 0), b = strtoull(argv[4], nullptr, 0);
        const uint64_t st = strtoull(argv[5], nullptr, 0);
        for (unsigned k = (kap == 4 ? 0 : kap); k <= (kap == 4 ? 3 : kap); ++k) {
            uint64_t u = a;
            run(unit_exp(), [&](uint32_t & x, uint32_t & aux) {
                if (u >= b) return false;
                x = (uint32_t)u;
                aux = k;
                u += st;
                return true;
            });
        }
    } else if (mode == "exp_rand" && argc >= 4) {
        const uint64_t n = strtoull(argv[2], nullptr, 0);
        std::mt19937_64 rng(strtoull(argv[3], nullptr, 0));
        for (unsigned k = 0; k < 4; ++k) {
            uint64_t i = 0;
            run(unit_exp(), [&](uint32_t & x, uint32_t & aux) {
                if (i++ >= n) return false;
                const uint64_t r = rng();
                switch (r % 4) {
                case 0: x = (uint32_t)(r >> 8); break;                                         // anywhere
                case 1: x = (uint32_t)((r >> 32) & 0x807FFFFFu) | (uint32_t)(120 + (r >> 20) % 40) << 23; break;  // |v| in [2^-7, 2^32)
                case 2: x = 0x42AE0000u + (uint32_t)((r >> 16) & 0x3FFFFu) - 0x20000u | (uint32_t)(r & 1) << 31; break; // ±87 region
                default: x = (uint32_t)((r >> 30) & 0x807FFFFFu) | 156u << 23 | (uint32_t)(r & 3); break; // Ev = 156
                }
                aux = k;
                return true;
            });
        }
    } else if ((mode == "rcp" || mode == "sqrt") && argc >= 3) {
        const Unit u = (mode == "rcp") ? unit_rcp() : unit_sqrt();
        for (int ai = 2; ai < argc; ++ai) {
            const uint32_t e = (uint32_t)strtoul(argv[ai], nullptr, 0) & 0xFFu;
            for (uint32_t s = 0; s < 2; ++s) {
                uint32_t m = 0;
                bool done = false;
                run(u, [&](uint32_t & x, uint32_t & aux) {
                    if (done) return false;
                    x = s << 31 | e << 23 | m;
                    aux = 0;
                    if (++m == (1u << 23)) done = true;
                    return true;
                });
            }
        }
    } else if (mode == "rand" && argc >= 4) {
        const uint64_t n = strtoull(argv[2], nullptr, 0);
        std::mt19937_64 rng(strtoull(argv[3], nullptr, 0));
        for (const Unit & u : {unit_rcp(), unit_sqrt()}) {
            uint64_t i = 0;
            run(u, [&](uint32_t & x, uint32_t & aux) {
                if (i++ >= n) return false;
                x = (uint32_t)rng();
                aux = 0;
                return true;
            });
        }
    } else if (mode == "sincos" && argc >= 5) {
        // every pattern with |θ| <= 2^hi_exp ... in [start, end) of the magnitude with a stride, both signs
        const uint64_t a = strtoull(argv[2], nullptr, 0), b = strtoull(argv[3], nullptr, 0);
        const uint64_t st = strtoull(argv[4], nullptr, 0);
        for (uint32_t sg = 0; sg < 2; ++sg) {
            uint64_t u = a;
            run(unit_sc(), [&](uint32_t & x, uint32_t & aux) {
                if (u >= b) return false;
                x = (uint32_t)u | sg << 31;
                aux = 0;
                u += st;
                return true;
            });
        }
    } else if (mode == "probe") {
        for (unsigned i = 0; i < VU_N; ++i) set32(top->c_t, i, 0);
        for (int cyc = 0; cyc < 80; ++cyc) {
            set32(top->c_t, 0, cyc == 0 ? 0x3F800000u : 0x7F800000u);
            tick();
            printf("after edge %d: s=%08x c=%08x\n", cyc + 1, get32(top->c_s, 0), get32(top->c_c, 0));
        }
        return 0;
    } else if (mode == "sincos_rand" && argc >= 4) {
        const uint64_t n = strtoull(argv[2], nullptr, 0);
        std::mt19937_64 rng(strtoull(argv[3], nullptr, 0));
        uint64_t i = 0;
        run(unit_sc(), [&](uint32_t & x, uint32_t & aux) {
            if (i++ >= n) return false;
            const uint64_t r = rng();
            switch (r % 3) {
            case 0: x = (uint32_t)(r >> 8); break;
            case 1: x = (uint32_t)((r >> 32) & 0x807FFFFFu) | (uint32_t)(110 + (r >> 20) % 50) << 23; break;  // 2^-17..2^32
            default: { const float f = (float)((double)(int64_t)(r >> 34) * 1e-3); memcpy(&x, &f, 4); break; } // ±8e6
            }
            aux = 0;
            return true;
        });
    } else if (mode == "specials") {
        const std::vector<uint32_t> v = special_patterns();
        for (unsigned k = 0; k < 4; ++k) {
            size_t i = 0;
            run(unit_exp(), [&](uint32_t & x, uint32_t & aux) {
                if (i >= v.size()) return false;
                x = v[i++];
                aux = k;
                return true;
            });
        }
        for (const Unit & u : {unit_rcp(), unit_sqrt(), unit_sc()}) {
            size_t i = 0;
            run(u, [&](uint32_t & x, uint32_t & aux) {
                if (i >= v.size()) return false;
                x = v[i++];
                aux = 0;
                return true;
            });
        }
    } else {
        fprintf(stderr, "vu_sweep: bad arguments\n");
        return 2;
    }
    printf("vu_sweep %s: %" PRIu64 " checked, %" PRIu64 " mismatches\n", mode.c_str(), g_checked, g_bad);
    top->final();
    top.reset();
    ctx.reset();
    return g_bad ? 1 : 0;
}
