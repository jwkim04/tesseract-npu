// sram_equiv.cpp — Verilator C++ harness: tsr_sram_1r1w of rtl/tech/xilinx against rtl/tech/generic,
// cycle by cycle, on the configurations of tb/tech/hdl/tsr_tb_tech_sram.sv (URAM / BRAM / LUTRAM
// sizes of the core, RD_LAT 1..4). Both models (Vgen = generic, Vxil = xilinx) get identical
// inputs. After an initialisation that writes every address of every configuration (all bytes)
// and one read, rd of the two models is compared bit for bit, and both against a byte-enable
// model of the contract (read-first collision): in every cycle for the BRAM / LUTRAM sizes (the
// pipelines are then fully defined and identical: RAM read register holding while re = 0,
// free-running output stages), in every cycle with a read result for the URAM sizes (the xilinx
// URAM output stages load only behind a read, rtl/tech/xilinx/tsr_sram_1r1w.sv; rdata is
// undefined in the other cycles by the contract).
// Stimulus phases of 1000 cycles each, chosen at random: uniform addresses, an 8-address window
// (dense read-after-write), forced same-cycle collisions (raddr = waddr), read of the previous
// cycle's write address, sparse enables; byte enables all / random / one byte / none.
// Usage: sram_equiv [random_cycles [seed]]. Exit status 0 iff no mismatch; prints counts.

#include <verilated.h>

#include <array>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

#include "Vgen.h"
#include "Vxil.h"

static const int NC = 21;
static const int CD[NC] = {16384, 16384, 16384, 4096, 4096, 1024, 512, 256, 72, 64, 64, 100, 37, 32, 2, 1,
                           16384, 16384, 16384, 4096, 4096};
static const int CW[NC] = {512, 512, 512, 512, 8, 512, 512, 512, 512, 512, 512, 512, 8, 512, 512, 72,
                           64, 64, 64, 64, 64};
static const int CL[NC] = {1, 2, 3, 2, 3, 3, 2, 2, 2, 1, 4, 1, 4, 2, 1, 2,
                           1, 2, 3, 1, 3};

static bool is_ultra(int c) {   // rtl/tech/xilinx/tsr_sram_1r1w.sv IS_ULTRA
    return CD[c] >= 4096 && (long)CD[c] * CW[c] >= 4096L * 64;
}

typedef std::array<uint32_t, 16> Line;   // 512 bits

struct In {
    int we = 0, re = 0;
    uint32_t waddr = 0, raddr = 0;
    Line wdata{};
    uint64_t wbe = 0;
};

struct Model {   // the contract, per configuration
    int depth, width, lat;
    std::vector<Line> mem;
    Line rd0{};                 // RAM read register
    std::vector<Line> pipe;     // stages 2..RD_LAT
    uint32_t rv = 0;            // bit i: a read was issued i + 1 edges ago
    Model(int d, int w, int l) : depth(d), width(w), lat(l), mem(d), pipe(l > 1 ? l - 1 : 0) {}
    bool result() const { return (rv >> (lat - 1)) & 1u; }   // rd holds a read result now
    Line mask(Line v) const {
        for (int i = 0; i < 16; ++i) {
            const int lo = 32 * i;
            if (lo >= width) v[i] = 0;
            else if (lo + 32 > width) v[i] &= (1u << (width - lo)) - 1u;
        }
        return v;
    }
    void edge(const In& in) {
        // output stages (free-running), then the read register (old data), then the write
        for (int s = (int)pipe.size() - 1; s > 0; --s) pipe[s] = pipe[s - 1];
        if (!pipe.empty()) pipe[0] = rd0;
        if (in.re) rd0 = mem[in.raddr % depth];
        rv = (rv << 1) | (uint32_t)in.re;
        if (in.we) {
            Line& m = mem[in.waddr % depth];
            for (int b = 0; b < width / 8; ++b) {
                if ((in.wbe >> b) & 1u) {
                    const uint32_t sh = 8 * (b % 4), msk = 0xffu << sh;
                    m[b / 4] = (m[b / 4] & ~msk) | (in.wdata[b / 4] & msk);
                }
            }
        }
    }
    Line out() const { return mask(pipe.empty() ? rd0 : pipe.back()); }
};

template <typename M>
static void drive(M& m, const In& in) {
    m.we = in.we;
    m.re = in.re;
    m.waddr = in.waddr;
    m.raddr = in.raddr;
    for (int i = 0; i < 16; ++i) m.wdata[i] = in.wdata[i];
    m.wbe = in.wbe;
}

int main(int argc, char** argv) {
    const long n_random = argc >= 2 ? std::strtol(argv[1], nullptr, 0) : 300000;
    const uint64_t seed = argc >= 3 ? std::strtoull(argv[2], nullptr, 0) : 1;
    auto ctx = std::make_unique<VerilatedContext>();
    ctx->randReset(2);
    ctx->randSeed((int)(seed & 0x7fffffff) | 1);
    auto g = std::make_unique<Vgen>(ctx.get(), "gen");
    auto x = std::make_unique<Vxil>(ctx.get(), "xil");
    std::mt19937_64 rng(seed);
    auto rnd = [&](uint64_t n) { return rng() % n; };
    std::vector<Model> mdl;
    for (int c = 0; c < NC; ++c) mdl.emplace_back(CD[c], CW[c], CL[c]);

    long cycles = 0, cmp_cycles = 0, reads = 0, fails = 0;
    std::vector<long> cmp(NC, 0), cmp_res(NC, 0);
    std::vector<long> coll(NC, 0), raw1(NC, 0);
    uint32_t prev_waddr = 0;
    int prev_we = 0;
    auto fail = [&](const char* what, int c, long cyc, int word, uint32_t a, uint32_t b) {
        if (fails < 20)
            std::printf("MISMATCH %s cfg %d (%d x %d, RD_LAT %d) cycle %ld word %d: %08x vs %08x\n", what, c, CD[c],
                        CW[c], CL[c], cyc, word, a, b);
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
        for (int c = 0; c < NC; ++c) {
            if (in.we && in.re && in.waddr % CD[c] == in.raddr % CD[c]) ++coll[c];
            if (prev_we && in.re && prev_waddr % CD[c] == in.raddr % CD[c]) ++raw1[c];
            mdl[c].edge(in);
        }
        prev_we = in.we;
        prev_waddr = in.waddr;
        reads += in.re;
        g->clk = 0;
        x->clk = 0;
        g->eval();
        x->eval();
        ++cycles;
        if (!check) return;
        ++cmp_cycles;
        for (int c = 0; c < NC; ++c) {
            const bool res = mdl[c].result();
            if (is_ultra(c) && !res) continue;
            ++cmp[c];
            cmp_res[c] += res;
            const Line ref = mdl[c].out();
            for (int i = 0; i < 16; ++i) {
                const uint32_t vg = g->rd[16 * c + i], vx = x->rd[16 * c + i];
                if (vg != vx) fail("gen/xil", c, cycles, i, vg, vx);
                if (vg != ref[i]) fail("gen/model", c, cycles, i, vg, ref[i]);
                if (vx != ref[i]) fail("xil/model", c, cycles, i, vx, ref[i]);
            }
        }
    };
    auto rand_line = [&](Line& l) { for (auto& w : l) w = (uint32_t)rng(); };

    // initialisation: every address of every configuration, all bytes; then one read everywhere
    In in;
    for (uint32_t a = 0; a < 16384; ++a) {
        in.we = 1;
        in.re = 0;
        in.waddr = a;
        in.raddr = 0;
        in.wbe = ~0ull;
        rand_line(in.wdata);
        step(in, false);
    }
    in.we = 0;
    in.re = 1;
    step(in, false);
    in.re = 0;
    for (int i = 0; i < 4; ++i) step(in, false);

    // random phases
    int mode = 0, bmode = 0;
    for (long i = 0; i < n_random; ++i) {
        if (i % 1000 == 0) {
            mode = (int)rnd(5);
            bmode = (int)rnd(4);
        }
        const int dense = mode == 4 ? 15 : 70;
        in.we = rnd(100) < (uint64_t)dense;
        in.re = rnd(100) < (uint64_t)dense;
        switch (mode) {
        case 0:   // uniform
            in.waddr = (uint32_t)rnd(16384);
            in.raddr = (uint32_t)rnd(16384);
            break;
        case 1:   // window of 8 lines
            in.waddr = (uint32_t)rnd(8);
            in.raddr = (uint32_t)rnd(8);
            break;
        case 2:   // same-cycle collision
            in.waddr = (uint32_t)rnd(16384);
            in.raddr = in.waddr;
            break;
        case 3:   // read the previous cycle's write address
            in.raddr = prev_waddr;
            in.waddr = (uint32_t)rnd(16384);
            break;
        default:  // sparse, uniform
            in.waddr = (uint32_t)rnd(16384);
            in.raddr = (uint32_t)rnd(16384);
            break;
        }
        switch (bmode) {
        case 0: in.wbe = ~0ull; break;
        case 1: in.wbe = rng(); break;
        case 2: in.wbe = 1ull << rnd(64); break;
        default: in.wbe = rnd(8) ? rng() : 0; break;
        }
        rand_line(in.wdata);
        step(in, true);
    }

    g->final();
    x->final();
    std::printf("sram_equiv: configs %d, cycles %ld (compared %ld), reads %ld, mismatches %ld\n", NC, cycles,
                cmp_cycles, reads, fails);
    std::printf("sram_equiv: compared cycles per config (with a read result):");
    for (int c = 0; c < NC; ++c) std::printf(" %ld (%ld)%s", cmp[c], cmp_res[c], is_ultra(c) ? "u" : "");
    std::printf("\nsram_equiv: same-cycle collisions per config:");
    for (int c = 0; c < NC; ++c) std::printf(" %ld", coll[c]);
    std::printf("\nsram_equiv: read of the previous cycle's write address per config:");
    for (int c = 0; c < NC; ++c) std::printf(" %ld", raw1[c]);
    std::printf("\n");
    return fails ? 1 : 0;
}
