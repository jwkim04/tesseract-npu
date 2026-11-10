// vu_lane_sweep.cpp — Verilator C++ harness: the VU lane (rtl/core/tsr_vu_lane.sv) against libtsr_ref, bit for
// bit, element by element. Toplevel tb/vu/hdl/tsr_vu_lane_tb.sv (16 lanes with EXPK and back lane). Built and
// run by tb/vu/test_vu_arith.py; manual use:
//   vu_lane_sweep vop <n> <seed>     every VOP function (VU §5.2), n elements each: (a, b, c) from an edge-heavy
//                                    distribution, against tsr_ref_vop_elem (front or back exit by the rate)
//   vu_lane_sweep gelu16 <seed>      GELU16 and GEGLU16 over all 65,536 fp16 inputs (and their f32 neighbors
//                                    ±1 ulp, the ±10 clamp edges, fp16 subnormals), GEGLU16 with random b
//   vu_lane_sweep at <n> <seed>      U_ASMAX, U_ESCB, U_AMASK, U_RCPMAX, U_ROT (both signs)
//                                    against the compositions of AT §5.4-§5.7 / VU §6.7-§6.8 in libtsr_ref
//   vu_lane_sweep red <rows> <seed>  rows of m <= 512 elements through U_RSUM / U_RSUMSQ / U_RMAX / U_AMASK into
//                                    the lanes' partials (16 lanes = one line), then every partial read back
//                                    against tsr_ref_red_partials (SUM, SUMSQ) or the per-partial MAXT
// Exit status 0 iff no mismatch.

#include <verilated.h>

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "Vtsr_vu_lane_tb.h"

extern "C" {
#include "tsr_ref.h"
#include "tsr_ref_math.h"
#include "tsr_ref_vu.h"
}

static const unsigned N = 16;
// tsr_vu_pkg::tsr_vu_uop_e
enum { U_NOP = 0, U_RSUM = 32, U_RSUMSQ = 33, U_RMAX = 34, U_ROT = 35, U_ASMAX = 36, U_ESCB = 37, U_AMASK = 38,
       U_RCPMAX = 39 };

static inline float u2f(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static inline uint32_t f2u(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static uint32_t FA(uint32_t a, uint32_t b) { return f2u(tsr_fadd(u2f(a), u2f(b))); }
static uint32_t FS(uint32_t a, uint32_t b) { return f2u(tsr_fsub(u2f(a), u2f(b))); }
static uint32_t FM(uint32_t a, uint32_t b) { return f2u(tsr_fmul(u2f(a), u2f(b))); }

template <std::size_t W>
static void setw(VlWide<W>& w, unsigned lsb, unsigned width, uint64_t v) {
    for (unsigned i = 0; i < width; ++i) {
        const unsigned b = lsb + i;
        if ((v >> i) & 1u) w[b / 32] |= 1u << (b % 32);
        else w[b / 32] &= ~(1u << (b % 32));
    }
}

static std::unique_ptr<VerilatedContext> ctx;
static std::unique_ptr<Vtsr_vu_lane_tb> top;
static uint64_t edges = 0, g_checked = 0, g_bad = 0;

static void tick() {
    top->clk = 0;
    top->eval();
    top->clk = 1;
    top->eval();
    ++edges;
}

struct El {
    uint32_t uop, a, b, c1, c2, neg;
};
struct Exp {
    uint64_t edge;
    bool back[N];
    bool v[N];
    El e[N];
    uint32_t y[N];
};

static void drive(unsigned i, const El & e, bool v) {
    top->in_v = (top->in_v & ~(1u << i)) | ((uint32_t)v << i);
    setw(top->in_uop, 6 * i, 6, e.uop);
    top->in_a[i] = e.a;
    top->in_b[i] = e.b;
    top->in_c1[i] = e.c1;
    top->in_c2[i] = e.c2;
    top->in_neg = (top->in_neg & ~(1u << i)) | ((e.neg & 1u) << i);
}

// Streams elements (16 per cycle) and checks the exit selected by back(uop).
template <class Gen, class Ref, class IsBack>
static void stream(Gen gen, Ref ref, IsBack is_back, const char * name) {
    std::deque<Exp> qf, qb;
    bool more = true;
    const unsigned lf = top->lat_f, lb = top->lat_b;
    while (more || !qf.empty() || !qb.empty()) {
        Exp ef{}, eb{};
        bool anyf = false, anyb = false;
        for (unsigned i = 0; i < N; ++i) {
            El e{};
            bool v = more && gen(e);
            if (!v) {
                more = false;
                e = El{};
            }
            drive(i, e, v);
            if (v) {
                const bool bk = is_back(e.uop);
                Exp & x = bk ? eb : ef;
                x.v[i] = true;
                x.e[i] = e;
                x.y[i] = ref(e);
                (bk ? anyb : anyf) = true;
            }
        }
        ef.edge = edges + 1 + lf;   // S0 samples at the next edge (cycle 0); the exit is valid LAT edges later
        eb.edge = edges + 1 + lb;
        if (anyf) qf.push_back(ef);
        if (anyb) qb.push_back(eb);
        tick();
        for (int pass = 0; pass < 2; ++pass) {
            std::deque<Exp> & q = pass ? qb : qf;
            while (!q.empty() && q.front().edge == edges) {
                const Exp & x = q.front();
                for (unsigned i = 0; i < N; ++i) {
                    if (!x.v[i]) continue;
                    const uint32_t got = pass ? top->b_y[i] : top->f_y[i];
                    ++g_checked;
                    if (got != x.y[i]) {
                        if (g_bad < 30)
                            fprintf(stderr, "%s: uop %u a=%08x b=%08x c1=%08x c2=%08x neg=%u rtl=%08x ref=%08x\n", name,
                                    x.e[i].uop, x.e[i].a, x.e[i].b, x.e[i].c1, x.e[i].c2, x.e[i].neg, got, x.y[i]);
                        ++g_bad;
                    }
                }
                q.pop_front();
            }
        }
    }
}

// Edge-heavy f32 patterns
static uint32_t rnd_f32(std::mt19937_64 & r) {
    static const uint32_t sp[] = {0x00000000u, 0x80000000u, 0x00000001u, 0x807FFFFFu, 0x00800000u, 0x80800000u,
                                  0x7F800000u, 0xFF800000u, 0x7FC00000u, 0xFFC00001u, 0x7F7FFFFFu, 0xFF7FFFFFu,
                                  0x3F800000u, 0xBF800000u, 0x41200000u, 0xC1200000u, 0x3C800000u, 0x3C7FFFFFu,
                                  0x42B17218u, 0xC2CE8ED0u, 0x42AEAC50u, 0xC2AEAC50u};
    const uint64_t x = r();
    switch (x % 8) {
    case 0: return sp[(x >> 8) % (sizeof sp / sizeof sp[0])];
    case 1: case 2: return (uint32_t)(x >> 32);                                        // anything
    case 3: return (uint32_t)((x >> 32) & 0x807FFFFFu) | (uint32_t)(100 + (x >> 20) % 60) << 23;   // |x| 2^-27..2^32
    case 4: return (uint32_t)((x >> 32) & 0x807FFFFFu) | (uint32_t)(124 + (x >> 20) % 8) << 23;    // 2^-3..2^4
    case 5: return f2u(tsr_fp16_to_f32((uint16_t)(x >> 40)));                        // fp16-exact
    case 6: return f2u((float)((int64_t)(x >> 44) - (1 << 19)) / 4096.0f);         // [-128, 128)
    default: return (uint32_t)((x >> 32) & 0x80FFFFFFu) | (uint32_t)(130 + (x >> 20) % 4) << 23;   // ±[8, 64)
    }
}

static bool half_func(uint32_t f) {
    switch (f) {
    case 0x06: case 0x07: case 0x11: case 0x12: case 0x13: case 0x14: case 0x15: case 0x17: case 0x18: return true;
    default: return false;
    }
}

int main(int argc, char ** argv) {
    ctx = std::make_unique<VerilatedContext>();
    ctx->commandArgs(argc, argv);
    top = std::make_unique<Vtsr_vu_lane_tb>(ctx.get());
    top->rst_n = 0;
    for (int i = 0; i < 4; ++i) tick();
    top->rst_n = 1;
    tick();
    if (argc < 2) return 2;
    const std::string mode = argv[1];
    auto back_uop = [](uint32_t u) {
        return (u >= 1 && u <= 25 && half_func(u - 1)) || u == U_RCPMAX;
    };
    if (mode == "vop" && argc >= 4) {
        const uint64_t n = strtoull(argv[2], nullptr, 0);
        std::mt19937_64 rng(strtoull(argv[3], nullptr, 0));
        for (uint32_t f = 0; f < 25; ++f) {
            uint64_t i = 0;
            stream([&](El & e) {
                       if (i++ >= n) return false;
                       e = El{f + 1, rnd_f32(rng), rnd_f32(rng), rnd_f32(rng), 0, 0};
                       return true;
                   },
                   [&](const El & e) { return tsr_ref_vop_elem(e.uop - 1, e.a, e.b, e.c1); }, back_uop, "vop");
        }
    } else if (mode == "gelu16" && argc >= 3) {
        std::mt19937_64 rng(strtoull(argv[2], nullptr, 0));
        std::vector<uint32_t> xs;
        for (uint32_t h = 0; h < 65536; ++h) {
            const uint32_t x = f2u(tsr_fp16_to_f32((uint16_t)h));
            xs.push_back(x);
            xs.push_back(x + 1);
            xs.push_back(x - 1);
        }
        for (uint32_t d = 0; d < 4096; ++d) {          // the ±10 clamp edges and fp16 rounding midpoints
            xs.push_back(0x41200000u - 2048 + d);
            xs.push_back(0xC1200000u - 2048 + d);
            xs.push_back(0x33000000u + d * 0x1000u);   // around the fp16 subnormal range
        }
        for (uint32_t f : {0x17u, 0x18u}) {
            size_t i = 0;
            stream([&](El & e) {
                       if (i >= xs.size()) return false;
                       e = El{f + 1, xs[i++], rnd_f32(rng), 0, 0, 0};
                       return true;
                   },
                   [&](const El & e) { return tsr_ref_vop_elem(e.uop - 1, e.a, e.b, 0); }, back_uop, "gelu16");
        }
    } else if (mode == "at" && argc >= 4) {
        const uint64_t n = strtoull(argv[2], nullptr, 0);
        std::mt19937_64 rng(strtoull(argv[3], nullptr, 0));
        const uint32_t uops[] = {U_ASMAX, U_ESCB, U_AMASK, U_RCPMAX, U_ROT};
        for (uint32_t u : uops) {
            uint64_t i = 0;
            stream([&](El & e) {
                       if (i++ >= n) return false;
                       e = El{u, rnd_f32(rng), rnd_f32(rng), rnd_f32(rng), rnd_f32(rng), (uint32_t)(rng() & 1)};
                       if (u == U_AMASK && (rng() & 3) == 0) e.b = 0xFF800000u;
                       if (u == U_ASMAX || u == U_ESCB) e.c2 = (e.c2 & 0x007FFFFFu) | (uint32_t)(100 + rng() % 50) << 23;
                       return true;
                   },
                   [&](const El & e) -> uint32_t {
                       switch (e.uop) {
                       case U_ASMAX: return f2u(tsr_m_exp(u2f(FM(FS(e.a, e.c1), e.c2))));
                       case U_ESCB: return f2u(tsr_m_exp(u2f(FM(FS(e.a, e.b), e.c2))));
                       case U_AMASK: return e.b == 0xFF800000u ? 0xFF800000u : FA(e.a, e.b);
                       case U_RCPMAX: return f2u(tsr_frecip(u2f(tsr_ref_vu_maxt(e.a, 0x00800000u))));
                       default: return e.neg ? FS(FM(e.a, e.c1), FM(e.b, e.c2)) : FA(FM(e.a, e.c1), FM(e.b, e.c2));
                       }
                   },
                   back_uop, "at");
        }
    } else if (mode == "red" && argc >= 4) {
        const uint64_t rows = strtoull(argv[2], nullptr, 0);
        std::mt19937_64 rng(strtoull(argv[3], nullptr, 0));
        const uint32_t kinds[] = {U_RSUM, U_RSUMSQ, U_RMAX, U_AMASK};
        for (uint64_t r = 0; r < rows; ++r) {
            const uint32_t k = kinds[rng() % 4];
            const uint32_t m = (rng() % 4 == 0) ? (uint32_t)(rng() % 17) : (uint32_t)(rng() % 513);
            const uint32_t buf = (uint32_t)(r & 1);
            std::vector<uint32_t> x(m), mk(m), v(m);
            for (uint32_t j = 0; j < m; ++j) {
                x[j] = rnd_f32(rng);
                if (rng() % 5 == 0) x[j] = (uint32_t)(rng() % 3 == 0 ? 0x80000000u : rng() % 2 ? 0x80C00000u : 0x00800000u);
                mk[j] = (rng() % 4 == 0) ? 0xFF800000u : rnd_f32(rng);
                v[j] = (k == U_AMASK) ? (mk[j] == 0xFF800000u ? 0xFF800000u : FA(x[j], mk[j])) : x[j];
            }
            const uint32_t lines = (m + 15) / 16;
            for (uint32_t li = 0; li < lines; ++li) {
                for (unsigned i = 0; i < N; ++i) {
                    const uint32_t j = 16 * li + i;
                    El e{k, j < m ? x[j] : 0u, j < m ? mk[j] : 0u, 0, 0, 0};
                    drive(i, e, true);
                    top->acc_lv = (top->acc_lv & ~(1u << i)) | ((uint32_t)(j < m) << i);
                    top->acc_fresh = (top->acc_fresh & ~(1u << i)) | ((uint32_t)(li < 8) << i);
                    top->acc_pa = (top->acc_pa & ~(15ull << (4 * i))) | ((uint64_t)(buf << 3 | (li & 7)) << (4 * i));
                }
                tick();
                for (unsigned i = 0; i < N; ++i) drive(i, El{}, false);
            }
            for (int w = 0; w < 16; ++w) tick();      // the last partial write ends lane cycle 13
            // expected partials (unwritten = untouched: compare only partials written in this row, i.e. phases
            // with a line, which every phase < lines has)
            uint32_t p[128];
            if (k == U_RSUM || k == U_RSUMSQ) {
                tsr_ref_red_partials(k == U_RSUM ? 0 : 1, x.data(), m, p);
            } else {
                const uint32_t init = (k == U_AMASK) ? 0xF1800000u : 0xFF800000u;
                for (int i = 0; i < 128; ++i) p[i] = init;
                for (uint32_t j = 0; j < m; ++j) p[j % 128] = tsr_ref_vu_maxt(p[j % 128], v[j]);
            }
            for (uint32_t ph = 0; ph < 8 && ph < lines; ++ph) {
                top->dr_addr = (uint8_t)(buf << 3 | ph);
                top->eval();
                for (unsigned i = 0; i < N; ++i) {
                    ++g_checked;
                    const uint32_t got = top->dr_data[i];
                    if (got != p[16 * ph + i]) {
                        if (g_bad < 30)
                            fprintf(stderr, "red kind %u m %u partial %u: rtl=%08x ref=%08x\n", k, m, 16 * ph + i, got,
                                    p[16 * ph + i]);
                        ++g_bad;
                    }
                }
            }
        }
    } else {
        fprintf(stderr, "vu_lane_sweep: bad arguments\n");
        return 2;
    }
    printf("vu_lane_sweep %s: %" PRIu64 " checked, %" PRIu64 " mismatches\n", mode.c_str(), g_checked, g_bad);
    top->final();
    top.reset();
    ctx.reset();
    return g_bad ? 1 : 0;
}
