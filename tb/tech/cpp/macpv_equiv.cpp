// macpv_equiv.cpp — Verilator C++ harness (W4-B at): tsr_mac_lane with PV = 1 (ATTPV's accumulate
// mode, AT §8.2) of rtl/tech/xilinx against rtl/tech/generic, cycle by cycle, on every
// configuration of tb/tech/hdl/tsr_tb_tech_macpv.sv (N 32 x T {1, 2, 3, 8} x LAT {3..7, 9, 11}, N 5).
// Both models (Vgen, Vxil) get identical inputs. In every cycle:
//   - PV phases (pv_en = 1, in_valid = 0): pv_acc of both models is compared with a reference model
//     of the contract (operands of the advancing edges, PV_LAT = IN + AD + M + P registers of the
//     lane's allocation, acc = (first ? 0 : acc) + a[s] x w[i] per valid, hold otherwise; the
//     valid pipeline cleared by reset, the accumulator not) once the slot has been started by a
//     `first`; the two models must agree there bit for bit;
//   - MVM phases (pv_en = 0): ov / yy of the two models are compared once LAT + 2 advancing edges of
//     the phase have flushed every register of the configuration (the same lanes' MVM mode after PV
//     use; mac_equiv.cpp checks the MVM contract).
// Stimulus: alternating phases; mode switches only after 16 idle cycles (the contract: pv_en
// changes while the lane is empty). PV: segments of 1..70 products starting with pv_first, pv_valid
// 80 %, ce 90 %, a in {+-(2^24 - 1), +-2^23, 0, +-1, random 25-bit}, w random / {-128, 127, 0, -1};
// rst_n low for 1..3 cycles with probability 0.1 %.
// Usage: macpv_equiv [cycles [seed]]. Exit status 0 iff no mismatch.

#include <verilated.h>

#include <array>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <memory>
#include <random>
#include <vector>

#include "Vgen.h"
#include "Vxil.h"

static const int NC = 30;
static const int YW = 8 * 21;
static const int PW = 4 * 32 * 38;

struct Cfg {
    int n, t, lat, ns, pvlat, sw;
};

static int pv_lat(int lat, int n) {   // tsr_att_pkg::tsr_mac_pv_lat
    int lv = 0;
    while ((1 << lv) < n) ++lv;
    const int tcmx = lv > 1 ? lv - 1 : 0, nreg = lat - 1, ra0 = nreg > 2 ? nreg - 2 : 0;
    const int tc1 = (ra0 >= 1 && tcmx >= 1) ? 1 : 0, ra1 = ra0 - tc1, rad = ra1 >= 1 ? 1 : 0;
    const int ra2 = ra1 - rad, rin = ra2 >= 1 ? 1 : 0;
    return rin + rad + 2;
}

static std::vector<Cfg> configs() {
    std::vector<Cfg> v;
    const int ts[4] = {1, 2, 3, 8}, ls[7] = {3, 4, 5, 6, 7, 9, 11};
    auto sw = [](int n) { int l = 0; while ((1 << l) < n) ++l; return 16 + l; };
    for (int a = 0; a < 4; ++a)
        for (int b = 0; b < 7; ++b) v.push_back({32, ts[a], ls[b], (ts[a] + 1) / 2, pv_lat(ls[b], 32), sw(32)});
    v.push_back({5, 3, 3, 2, pv_lat(3, 5), sw(5)});
    v.push_back({5, 3, 6, 2, pv_lat(6, 5), sw(5)});
    return v;
}

struct In {
    int rst_n = 0, ce = 1, in_valid = 0, pv_en = 0, pv_valid = 0, pv_first = 0;
    std::array<int8_t, 32> w{};
    std::array<std::array<int8_t, 32>, 8> x{};
    std::array<int32_t, 4> a{};
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
static uint64_t get_bits(const W& wd, unsigned lsb, unsigned width) {
    uint64_t v = 0;
    for (unsigned i = 0; i < width; ++i) {
        const unsigned b = lsb + i;
        v |= (uint64_t)((wd[b / 32] >> (b % 32)) & 1u) << i;
    }
    return v;
}

template <typename M>
static void drive(M& m, const In& in) {
    m.rst_n = in.rst_n;
    m.ce = in.ce;
    m.in_valid = in.in_valid;
    m.pv_en = in.pv_en;
    m.pv_valid = in.pv_valid;
    m.pv_first = in.pv_first;
    for (int k = 0; k < 32; ++k) set_bits(m.w, 8 * k, 8, (uint8_t)in.w[k]);
    for (int t = 0; t < 8; ++t)
        for (int k = 0; k < 32; ++k) set_bits(m.x, 256 * t + 8 * k, 8, (uint8_t)in.x[t][k]);
    for (int s = 0; s < 4; ++s) set_bits(m.pv_a, 25 * s, 25, (uint32_t)in.a[s] & 0x1FFFFFFu);
}

struct Stage {
    int v = 0, f = 0;
    std::array<int32_t, 4> a{};
    std::array<int8_t, 32> w{};
};

int main(int argc, char** argv) {
    const long n_cycles = argc >= 2 ? std::strtol(argv[1], nullptr, 0) : 200000;
    const uint64_t seed = argc >= 3 ? std::strtoull(argv[2], nullptr, 0) : 1;
    const std::vector<Cfg> cfg = configs();
    if ((int)cfg.size() != NC) return 2;

    auto ctx = std::make_unique<VerilatedContext>();
    ctx->randReset(2);
    ctx->randSeed((int)(seed & 0x7fffffff) | 1);
    auto g = std::make_unique<Vgen>(ctx.get(), "gen");
    auto x = std::make_unique<Vxil>(ctx.get(), "xil");
    std::mt19937_64 rng(seed);
    auto U = [&](int n) { return (int)(rng() % (uint64_t)n); };

    // reference per config: pipeline stages (pvlat - 1 of them) and the accumulators
    std::vector<std::deque<Stage>> pipe(NC);
    std::vector<std::vector<int64_t>> acc(NC);
    std::vector<std::vector<int>> started(NC);
    for (int c = 0; c < NC; ++c) {
        pipe[c].assign(cfg[c].pvlat - 1, Stage{});
        acc[c].assign(cfg[c].ns * cfg[c].n, 0);
        started[c].assign(cfg[c].ns, 0);
    }

    long fails = 0, pv_checks = 0, mvm_checks = 0, pv_products = 0;
    auto fail = [&](const char* what, int c, long cyc, int64_t a, int64_t b) {
        if (fails < 20)
            std::printf("MISMATCH %s cfg %d (N %d T %d LAT %d) cycle %ld: %" PRId64 " vs %" PRId64 "\n", what, c,
                        cfg[c].n, cfg[c].t, cfg[c].lat, cyc, a, b);
        ++fails;
    };

    g->clk = 0;
    x->clk = 0;
    In in;
    int mode = 0, idle = 0, phase_left = 0, mvm_age = 0, seg_left = 0;
    for (long cyc = 0; cyc < n_cycles; ++cyc) {
        // ---- stimulus ----
        in.rst_n = (cyc < 4) ? 0 : (U(1000) == 0 ? 0 : 1);
        in.ce = U(10) < 9;
        if (phase_left <= 0 && idle >= 16) {          // switch modes only while empty
            mode ^= 1;
            phase_left = 300 + U(3000);
            mvm_age = 0;
            seg_left = 0;
        }
        const bool draining = phase_left <= 0;
        in.pv_en = mode;
        in.in_valid = 0;
        in.pv_valid = 0;
        in.pv_first = 0;
        if (mode == 0 && !draining) {
            in.in_valid = U(10) < 7;
            for (int k = 0; k < 32; ++k) in.w[k] = (int8_t)rng();
            for (int t = 0; t < 8; ++t)
                for (int k = 0; k < 32; ++k) in.x[t][k] = (int8_t)rng();
        }
        if (mode == 1 && !draining) {
            in.pv_valid = U(10) < 8;
            if (in.pv_valid) {
                if (seg_left <= 0) {
                    in.pv_first = 1;
                    seg_left = 1 + U(70);
                }
                --seg_left;
            }
            for (int k = 0; k < 32; ++k) {
                const int r = U(10);
                in.w[k] = r == 0 ? -128 : r == 1 ? 127 : r == 2 ? 0 : r == 3 ? -1 : (int8_t)rng();
            }
            for (int s = 0; s < 4; ++s) {
                const int r = U(9);
                const int32_t A = (1 << 24) - 1;
                in.a[s] = r == 0 ? A : r == 1 ? -A : r == 2 ? (1 << 23) : r == 3 ? -(1 << 23) : r == 4 ? 0
                        : r == 5 ? 1 : r == 6 ? -1 : ((int32_t)(rng() << 7) >> 7);
            }
        }
        if (draining) {
            in.pv_valid = 0;
            in.in_valid = 0;
        }
        const bool any_valid = in.pv_valid || in.in_valid;
        idle = (!any_valid && in.ce) ? idle + 1 : (any_valid ? 0 : idle);
        --phase_left;

        // ---- edge ----
        drive(*g, in);
        drive(*x, in);
        g->eval();
        x->eval();
        g->clk = 1;
        x->clk = 1;
        g->eval();
        x->eval();
        // reference: P update from the last stage, then the stages
        for (int c = 0; c < NC; ++c) {
            auto& p = pipe[c];
            const Cfg& f = cfg[c];
            if (in.ce) {
                const Stage& last = p.back();
                if (last.v) {
                    for (int s = 0; s < f.ns; ++s) {
                        if (last.f) started[c][s] = 1;
                        for (int k = 0; k < f.n; ++k) {
                            const int64_t prod = (int64_t)last.a[s] * (int64_t)last.w[k];
                            int64_t& a = acc[c][s * f.n + k];
                            a = (last.f ? 0 : a) + prod;
                            a = (a << 26) >> 26;         // 38-bit two's complement
                        }
                    }
                    ++pv_products;
                }
                p.pop_back();
                Stage n;
                n.v = in.pv_valid && in.pv_en;
                n.f = in.pv_first;
                n.a = in.a;
                n.w = in.w;
                p.push_front(n);
            }
            if (!in.rst_n)
                for (auto& st : p) st.v = 0;
        }
        g->clk = 0;
        x->clk = 0;
        g->eval();
        x->eval();

        // ---- compare ----
        if (in.pv_en) {
            for (int c = 0; c < NC; ++c) {
                const Cfg& f = cfg[c];
                for (int s = 0; s < f.ns; ++s) {
                    if (!started[c][s]) continue;
                    for (int k = 0; k < f.n; ++k) {
                        const unsigned lsb = 38u * (unsigned)(s * f.n + k);
                        const unsigned base = (unsigned)c * PW + lsb;
                        const int64_t gv = ((int64_t)get_bits(g->pva, base, 38) << 26) >> 26;
                        const int64_t xv = ((int64_t)get_bits(x->pva, base, 38) << 26) >> 26;
                        const int64_t rv = acc[c][s * f.n + k];
                        if (gv != rv) fail("gen pv_acc vs reference", c, cyc, gv, rv);
                        if (xv != rv) fail("xil pv_acc vs reference", c, cyc, xv, rv);
                        ++pv_checks;
                    }
                }
            }
        } else {
            for (auto& v : started) std::fill(v.begin(), v.end(), 0);
            if (in.ce) ++mvm_age;
            {
                for (int c = 0; c < NC; ++c) {
                    if (mvm_age <= cfg[c].lat + 2) continue;
                    const int og = (g->ov >> c) & 1, ox = (x->ov >> c) & 1;
                    if (og != ox) fail("ov", c, cyc, og, ox);
                    for (int b = 0; b < cfg[c].t * cfg[c].sw; b += 21) {
                        const int w = std::min(21, cfg[c].t * cfg[c].sw - b);
                        const unsigned base = (unsigned)c * YW + (unsigned)b;
                        const uint64_t a = get_bits(g->yy, base, w), bb = get_bits(x->yy, base, w);
                        if (a != bb) fail("yy", c, cyc, (int64_t)a, (int64_t)bb);
                    }
                    ++mvm_checks;
                }
            }
        }
    }
    std::printf("cycles %ld pv products %ld pv checks %ld mvm checks %ld mismatches %ld\n", n_cycles, pv_products,
                pv_checks, mvm_checks, fails);
    return fails ? 1 : 0;
}
