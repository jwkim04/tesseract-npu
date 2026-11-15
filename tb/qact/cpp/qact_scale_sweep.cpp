// qact_scale_sweep.cpp — Verilator C++ harness: bit-exact sweeps of the A32 scale pipeline (rtl/core/tsr_qact_scale.sv)
// against libtsr_ref (sw/sim/tsr_ref.c): d = tsr_fdiv(amax, 127), scale = (d != +-0) ? tsr_frecip(d) : +0,
// h16 = tsr_f32_to_fp16(d), zero = (amax == 0). Toplevel: tb/qact/hdl/tsr_qact_scale_tb.sv (N lanes). Built and run
// by tb/qact/test_qact_scale.py; manual use:
//   qact_scale_sweep exps <exp> [<exp> ...]     every mantissa at these exponent fields (0..255)
//   qact_scale_sweep range <start> <end>        every amax pattern start <= u < end (amax <= 0x7F800000)
//   qact_scale_sweep rand <n> <seed>            n random amax patterns, valid gaps and garbage on idle lanes
//   qact_scale_sweep specials                   0, Inf and, at every exponent, the mantissas 0, 1, 0x400000, max
// An input is an amax as tsr_qact produces it: |DAZ(x)| bits, never NaN (0 .. 0x7F800000). Exit status 0 iff no
// mismatch. The sweeps of rand and specials leave lanes and cycles idle (in_valid low, random garbage on amag) and
// check that out_valid follows exactly and that idle lanes produce no result.

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

#include "Vtsr_qact_scale_tb.h"

extern "C" {
#include "tsr_ref.h"
}

#ifndef QS_N
#define QS_N 64       // lanes (in_valid is one 64-bit port)
#endif
static_assert(QS_N <= 64, "in_valid is set as one 64-bit word");

static inline float u2f(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static inline uint32_t f2u(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

static std::unique_ptr<VerilatedContext> ctx;
static std::unique_ptr<Vtsr_qact_scale_tb> top;
static uint64_t edges = 0;
static uint64_t g_checked = 0, g_bad = 0;
static std::mt19937_64 g_rng(1);

static void tick() {
    top->clk = 0;
    top->eval();
    top->clk = 1;
    top->eval();
    ++edges;
}

struct Exp {
    uint64_t edge;
    uint32_t amag[QS_N];
    bool valid[QS_N];
    uint32_t scale[QS_N], aux[QS_N];     // aux = {zero, out_valid, h16} as the toplevel packs it
};

static void reference(uint32_t amag, uint32_t & scale, uint32_t & aux) {
    const float d = tsr_fdiv(u2f(amag), 127.0f);
    scale = (f2u(d) & 0x7FFFFFFFu) ? f2u(tsr_frecip(d)) : 0u;
    aux = (uint32_t)(amag == 0) << 17 | 1u << 16 | tsr_f32_to_fp16(d);
}

// Streams the amax patterns gen() produces (false = done) through the unit, N per cycle. gaps: idle lanes / cycles.
static void run(const std::function<bool(uint32_t &)> & gen, bool gaps) {
    const unsigned lat = top->lat;
    std::deque<Exp> q;
    bool more = true;
    while (more || !q.empty()) {
        Exp e;
        const bool feeding = more;       // idle cycles are checked while inputs are still coming
        const bool idle_cycle = gaps && more && (g_rng() % 8 == 0);
        for (unsigned i = 0; i < QS_N; ++i) {
            bool v = false;
            uint32_t a = (uint32_t)g_rng() & 0x7FFFFFFFu;      // garbage unless a pattern is taken
            if (more && !idle_cycle && (!gaps || g_rng() % 4 != 0)) {
                uint32_t x;
                if (gen(x)) {
                    a = x;
                    v = true;
                } else {
                    more = false;
                }
            }
            e.amag[i] = a;
            e.valid[i] = v;
            e.scale[i] = e.aux[i] = 0;
            if (v) reference(a, e.scale[i], e.aux[i]);
            top->amag[i] = a;
        }
        uint64_t m = 0;
        for (unsigned i = 0; i < QS_N; ++i) m |= (uint64_t)e.valid[i] << i;
        top->in_valid = m;
        e.edge = edges + 1 + lat - 1;      // inputs sampled at the next edge; result LAT - 1 edges later
        if (feeding) q.push_back(e);
        tick();
        while (!q.empty() && q.front().edge == edges) {
            const Exp & f = q.front();
            for (unsigned i = 0; i < QS_N; ++i) {
                const uint32_t aux = top->aux[i], scale = top->scale[i];
                const bool ov = (aux >> 16) & 1;
                if (ov != f.valid[i]) {
                    if (g_bad < 20) fprintf(stderr, "out_valid %d for a lane with in_valid %d (amax 0x%08" PRIx32 ")\n", ov, f.valid[i], f.amag[i]);
                    ++g_bad;
                    continue;
                }
                if (!f.valid[i]) continue;
                ++g_checked;
                if (aux != f.aux[i] || scale != f.scale[i]) {
                    if (g_bad < 20)
                        fprintf(stderr, "mismatch: amax=0x%08" PRIx32 " rtl scale 0x%08" PRIx32 " {zero,v,h16} 0x%05" PRIx32 " ref 0x%08" PRIx32 " 0x%05" PRIx32 "\n",
                                f.amag[i], scale, aux, f.scale[i], f.aux[i]);
                    ++g_bad;
                }
            }
            q.pop_front();
        }
    }
}

int main(int argc, char ** argv) {
    ctx = std::make_unique<VerilatedContext>();
    ctx->commandArgs(argc, argv);
    top = std::make_unique<Vtsr_qact_scale_tb>(ctx.get());
    top->clk = 0;
    top->rst_n = 0;
    for (int i = 0; i < 4; ++i) tick();
    top->rst_n = 1;
    tick();
    if (argc < 2) {
        fprintf(stderr, "usage: qact_scale_sweep exps|range|rand|specials ...\n");
        return 2;
    }
    const std::string mode = argv[1];
    if (mode == "exps" && argc >= 3) {
        for (int ai = 2; ai < argc; ++ai) {
            const uint32_t e = (uint32_t)strtoul(argv[ai], nullptr, 0) & 0xFFu;
            uint32_t m = 0;
            run([&](uint32_t & x) {
                while (m < (1u << 23)) {
                    x = e << 23 | m++;
                    if (x <= 0x7F800000u) return true;
                }
                return false;
            }, false);
        }
    } else if (mode == "range" && argc >= 4) {
        const uint64_t a = strtoull(argv[2], nullptr, 0), b = strtoull(argv[3], nullptr, 0);
        uint64_t u = a;
        run([&](uint32_t & x) {
            if (u >= b || u > 0x7F800000ull) return false;
            x = (uint32_t)u++;
            return true;
        }, false);
    } else if (mode == "rand" && argc >= 4) {
        const uint64_t n = strtoull(argv[2], nullptr, 0);
        g_rng.seed(strtoull(argv[3], nullptr, 0));
        uint64_t i = 0;
        run([&](uint32_t & x) {
            if (i++ >= n) return false;
            x = (uint32_t)(g_rng() % 0x7F800001ull);
            return true;
        }, true);
    } else if (mode == "specials") {
        std::vector<uint32_t> v = {0x00000000u, 0x7F800000u, 0x7F7FFFFFu, 0x00800000u, 0x00800001u, 0x3F800000u,
                                   0x42FE0000u, 0x42FDFFFFu, 0x42FE0001u};   // 127.0 and its neighbors
        for (uint32_t e = 0; e < 256; ++e) {
            for (uint32_t m : {0u, 1u, 0x400000u, 0x7FFFFFu}) {
                const uint32_t x = e << 23 | m;
                if (x <= 0x7F800000u) v.push_back(x);
            }
        }
        for (int pass = 0; pass < 2; ++pass) {      // without and with idle lanes
            size_t i = 0;
            run([&](uint32_t & x) {
                if (i >= v.size()) return false;
                x = v[i++];
                return true;
            }, pass == 1);
        }
    } else {
        fprintf(stderr, "qact_scale_sweep: bad arguments\n");
        return 2;
    }
    printf("qact_scale_sweep %s: %" PRIu64 " checked, %" PRIu64 " mismatches\n", mode.c_str(), g_checked, g_bad);
    top->final();
    top.reset();
    ctx.reset();
    return g_bad ? 1 : 0;
}
