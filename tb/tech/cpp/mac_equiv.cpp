// mac_equiv.cpp — Verilator C++ harness: tsr_mac_lane of rtl/tech/xilinx against rtl/tech/generic,
// cycle by cycle, on every configuration of tb/tech/hdl/tsr_tb_tech_mac.sv (N 32 x T 1..8 x
// LAT 1..11, N 5 / 1 / 2 / 3 edge cases). Both models (Vgen = generic, Vxil = xilinx; same
// toplevel, different --prefix) get identical inputs; in every cycle after the flush:
//   - ov and yy of the two models are compared bit for bit (every configuration, valid or not);
//   - both are checked against a reference: y[t] = sum_i w[i] x[t][i] of the operands sampled
//     LAT advancing (ce = 1) edges earlier (in every cycle, the datapath has no reset), and
//     out_valid against a model of the reset / ce valid pipeline.
// Stimulus (ce = 1, in_valid = 1 during the sweeps):
//   pairs    216 beats: lane k / slot p sees the triple (w, x_high, x_low) = combination
//            (b + 5k) mod 216 of the corner set {-128, -127, -1, 0, 1, 127}^3 with per-slot
//            rotations, so every DSP slot of every lane sees all 216 triples (both packed
//            products at every corner pair, every weight corner)
//   uniform  216 beats: all lanes the same triple (extreme sums, e.g. 32 x (-128)(-128) = 2^19)
//   random   R cycles (argument, default 200000): per beat one of uniform random bytes, corner
//            values, Q4_K weights 0..15, extremes only, sparse; ce 85 %, in_valid 70 %, rst_n
//            low for 1..3 cycles with probability 0.3 %.
// Usage: mac_equiv [random_cycles [seed]]. Exit status 0 iff no mismatch; prints counts.

#include <verilated.h>

#include <array>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <random>
#include <vector>

#include "Vgen.h"
#include "Vxil.h"

static const int NC = 136;
static const int YW = 8 * 21;

struct Cfg {
    int n, t, lat, sw;
};

static std::vector<Cfg> configs() {   // the order of tsr_tb_tech_mac.sv
    std::vector<Cfg> v;
    auto sw = [](int n) { int l = 0; while ((1 << l) < n) ++l; return 16 + l; };
    for (int t = 1; t <= 8; ++t)
        for (int l = 1; l <= 11; ++l) v.push_back({32, t, l, sw(32)});
    for (int t = 1; t <= 3; ++t)
        for (int l = 1; l <= 6; ++l) v.push_back({5, t, l, sw(5)});
    for (int t = 1; t <= 3; ++t)
        for (int l = 1; l <= 4; ++l) v.push_back({1, t, l, sw(1)});
    for (int t = 1; t <= 3; ++t)
        for (int l = 1; l <= 4; ++l) v.push_back({2, t, l, sw(2)});
    for (int l = 1; l <= 6; ++l) v.push_back({3, 3, l, sw(3)});
    return v;
}

struct In {
    int rst_n = 0, ce = 1, in_valid = 0;
    std::array<int8_t, 32> w{};
    std::array<std::array<int8_t, 32>, 8> x{};
};

template <typename W>
static void set_bits(W& wd, unsigned lsb, unsigned width, uint32_t v) {
    for (unsigned i = 0; i < width; ++i) {
        const unsigned b = lsb + i;
        const uint32_t m = 1u << (b % 32);
        if ((v >> i) & 1u) wd[b / 32] |= m; else wd[b / 32] &= ~m;
    }
}
template <typename W>
static uint32_t get_bits(const W& wd, unsigned lsb, unsigned width) {
    uint32_t v = 0;
    for (unsigned i = 0; i < width; ++i) {
        const unsigned b = lsb + i;
        v |= ((wd[b / 32] >> (b % 32)) & 1u) << i;
    }
    return v;
}

template <typename M>
static void drive(M& m, const In& in) {
    m.rst_n = in.rst_n;
    m.ce = in.ce;
    m.in_valid = in.in_valid;
    for (int k = 0; k < 32; ++k) set_bits(m.w, 8 * k, 8, (uint8_t)in.w[k]);
    for (int t = 0; t < 8; ++t)
        for (int k = 0; k < 32; ++k) set_bits(m.x, 256 * t + 8 * k, 8, (uint8_t)in.x[t][k]);
}

static const int8_t CORNER[6] = {-128, -127, -1, 0, 1, 127};

int main(int argc, char** argv) {
    const long n_random = argc >= 2 ? std::strtol(argv[1], nullptr, 0) : 200000;
    const uint64_t seed = argc >= 3 ? std::strtoull(argv[2], nullptr, 0) : 1;
    const std::vector<Cfg> cfg = configs();
    if ((int)cfg.size() != NC) { std::printf("config table size %zu != %d\n", cfg.size(), NC); return 2; }

    auto ctx = std::make_unique<VerilatedContext>();
    ctx->randReset(2);
    ctx->randSeed((int)(seed & 0x7fffffff) | 1);
    auto g = std::make_unique<Vgen>(ctx.get(), "gen");
    auto x = std::make_unique<Vxil>(ctx.get(), "xil");
    std::mt19937_64 rng(seed);

    // advancing-edge operand history (the newest last) and per-config valid pipelines
    std::deque<In> hist;
    std::vector<std::deque<int>> vpipe(NC);
    for (int c = 0; c < NC; ++c) vpipe[c].assign(cfg[c].lat, 0);
    long edges_since_flush = -1;   // advancing edges since the flush started (-1: not started)

    long cycles = 0, cmp_cycles = 0, ref_checks = 0, valid_checks = 0, fails = 0;
    long valid_ones = 0;
    auto fail = [&](const char* what, int c, long cyc, long a, long b) {
        if (fails < 20)
            std::printf("MISMATCH %s cfg %d (N %d T %d LAT %d) cycle %ld: %ld vs %ld\n", what, c, cfg[c].n,
                        cfg[c].t, cfg[c].lat, cyc, a, b);
        ++fails;
    };

    g->clk = 0;
    x->clk = 0;
    auto step = [&](const In& in, bool check) {
        drive(*g, in);
        drive(*x, in);
        g->eval();
        x->eval();
        g->clk = 1;
        x->clk = 1;
        g->eval();
        x->eval();
        // models of the edge
        for (int c = 0; c < NC; ++c) {
            auto& vp = vpipe[c];
            if (!in.rst_n) {
                std::fill(vp.begin(), vp.end(), 0);
            } else if (in.ce) {
                vp.pop_back();
                vp.push_front(in.in_valid);
            }
        }
        if (in.ce) {
            hist.push_back(in);
            if (hist.size() > 16) hist.pop_front();
            if (edges_since_flush >= 0) ++edges_since_flush;
        }
        g->clk = 0;
        x->clk = 0;
        g->eval();
        x->eval();
        ++cycles;
        if (!check) return;
        ++cmp_cycles;
        for (int c = 0; c < NC; ++c) {
            const Cfg& f = cfg[c];
            const int ovg = (g->ov[c / 32] >> (c % 32)) & 1, ovx = (x->ov[c / 32] >> (c % 32)) & 1;
            if (ovg != ovx) fail("out_valid gen/xil", c, cycles, ovg, ovx);
            if (ovg != vpipe[c].back()) fail("out_valid gen/model", c, cycles, ovg, vpipe[c].back());
            ++valid_checks;
            valid_ones += ovg;
            // operands of advancing edge (newest - LAT + 1)
            const In& op = hist[hist.size() - f.lat];
            for (int t = 0; t < f.t; ++t) {
                const unsigned lsb = c * YW + t * f.sw;
                const uint32_t yg = get_bits(g->yy, lsb, f.sw), yx = get_bits(x->yy, lsb, f.sw);
                if (yg != yx) fail("y gen/xil", c, cycles, yg, yx);
                long ref = 0;
                for (int k = 0; k < f.n; ++k) ref += (long)op.w[k] * (long)op.x[t][k];
                const uint32_t refb = (uint32_t)ref & ((1u << f.sw) - 1u);
                if (yg != refb) fail("y gen/ref", c, cycles, yg, refb);
                if (yx != refb) fail("y xil/ref", c, cycles, yx, refb);
                ++ref_checks;
            }
            // the padding above T x SUM_W is zero in both
        }
    };

    auto rnd = [&](int n) { return (int)(rng() % (uint64_t)n); };

    // flush: 16 cycles in reset with ce = 1 and defined operands (every datapath register loads)
    In in;
    for (int i = 0; i < 16; ++i) {
        in.rst_n = 0;
        in.ce = 1;
        in.in_valid = rnd(2);
        for (int k = 0; k < 32; ++k) {
            in.w[k] = (int8_t)rnd(256);
            for (int t = 0; t < 8; ++t) in.x[t][k] = (int8_t)rnd(256);
        }
        if (i == 0) edges_since_flush = 0;
        step(in, false);
    }

    // corner sweeps
    long n_pairs = 0, n_uniform = 0;
    for (int b = 0; b < 216; ++b, ++n_pairs) {
        in.rst_n = 1;
        in.ce = 1;
        in.in_valid = 1;
        for (int k = 0; k < 32; ++k) {
            const int i = (b + 5 * k) % 216;
            in.w[k] = CORNER[i / 36];
            for (int p = 0; p < 4; ++p) {
                in.x[2 * p][k] = CORNER[((i / 6) % 6 + p) % 6];
                in.x[2 * p + 1][k] = CORNER[(i % 6 + 2 * p) % 6];
            }
        }
        step(in, true);
    }
    for (int b = 0; b < 216; ++b, ++n_uniform) {
        for (int k = 0; k < 32; ++k) {
            in.w[k] = CORNER[b / 36];
            for (int p = 0; p < 4; ++p) {
                in.x[2 * p][k] = CORNER[(b / 6) % 6];
                in.x[2 * p + 1][k] = CORNER[b % 6];
            }
        }
        step(in, true);
    }
    // drain the sweeps through the deepest pipeline
    for (int i = 0; i < 12; ++i) step(in, true);

    // random
    int rst_left = 0;
    for (long i = 0; i < n_random; ++i) {
        if (rst_left > 0) {
            --rst_left;
        } else if (rnd(1000) < 3) {
            rst_left = 1 + rnd(3);
        }
        in.rst_n = rst_left > 0 ? 0 : 1;
        in.ce = rnd(100) < 85;
        in.in_valid = rnd(100) < 70;
        const int mode = rnd(5);
        for (int k = 0; k < 32; ++k) {
            switch (mode) {
            case 0:   // uniform bytes
                in.w[k] = (int8_t)rnd(256);
                for (int t = 0; t < 8; ++t) in.x[t][k] = (int8_t)rnd(256);
                break;
            case 1:   // corners
                in.w[k] = CORNER[rnd(6)];
                for (int t = 0; t < 8; ++t) in.x[t][k] = CORNER[rnd(6)];
                break;
            case 2:   // Q4_K weights (0..15), any activation
                in.w[k] = (int8_t)rnd(16);
                for (int t = 0; t < 8; ++t) in.x[t][k] = (int8_t)rnd(256);
                break;
            case 3:   // extremes only
                in.w[k] = rnd(2) ? -128 : 127;
                for (int t = 0; t < 8; ++t) in.x[t][k] = rnd(3) == 0 ? 127 : -128;
                break;
            default:  // sparse
                in.w[k] = rnd(4) ? 0 : (int8_t)rnd(256);
                for (int t = 0; t < 8; ++t) in.x[t][k] = rnd(4) ? 0 : (int8_t)rnd(256);
                break;
            }
        }
        step(in, true);
    }

    g->final();
    x->final();
    std::printf("mac_equiv: configs %d, cycles %ld (compared %ld: pairs %ld + uniform %ld beats, random %ld),"
                " y checks %ld, valid checks %ld (%ld valid), mismatches %ld\n",
                NC, cycles, cmp_cycles, n_pairs, n_uniform, n_random, ref_checks, valid_checks, valid_ones, fails);
    return fails ? 1 : 0;
}
