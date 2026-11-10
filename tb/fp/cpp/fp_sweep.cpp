// fp_sweep.cpp — Verilator C++ harness: multi-million-vector bit-exact sweeps of the fp library
// (rtl/core/tsr_fp_pkg.sv, tsr_fp_mul.sv, tsr_fp_div.sv, tsr_fp_add.sv) against libtsr_ref
// (sw/sim/tsr_ref.c).
//
// Toplevel: tb/fp/hdl/tsr_fp_sweep_tb.sv (LANES function lanes, MLANES multipliers, DLANES
// dividers, ALANES adders). Built and run by tb/fp/test_fp.py (test_fp_sweep_*); manual use:
//   fp_sweep funcs  <start> <end>      every f32 / int32 pattern u in [start, end) (hex ok) through
//                                      f32_to_fp16, f32_to_q8 (away / even), i32_to_f32, daz,
//                                      is_nan, amag, amax_cand; fp16 h = u & 0xFFFF through
//                                      fp16_to_f32 (so any range of >= 65536 covers all halves)
//   fp_sweep funcs_rand <n> <seed>     n random patterns (biased to rounding boundaries)
//   fp_sweep amax   <blocks> <seed>    random LANES-element blocks: tree and chain reductions vs
//                                      the reference scan
//   fp_sweep mul    <n> <seed>         n products per generator class (random ce)
//   fp_sweep div    <n> <seed>         n quotients per generator class (random out_ready stalls,
//                                      fixed-latency check)
//   fp_sweep add    <n> <seed>         n sums / differences per add class (random sub per lane,
//                                      random ce and in_valid; checks out_valid timing against a
//                                      LAT-deep model and that out_valid / y hold while ce = 0)
// Exit status 0 iff no mismatch. Every f32 crosses into libtsr_ref as a bit pattern (memcpy), so
// NaN payloads / sNaNs reach the reference unchanged.

#include <verilated.h>

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <string>
#include <vector>

#include "Vtsr_fp_sweep_tb.h"

extern "C" {
#include "tsr_ref.h"
}

// ---------------------------------------------------------------------------------------------
// Bit helpers for Verilator ports (VlWide for > 64 bits, plain integers otherwise).
// ---------------------------------------------------------------------------------------------
template <std::size_t N>
static inline uint64_t getb(const VlWide<N>& w, unsigned lsb, unsigned width) {
    uint64_t v = 0;
    for (unsigned i = 0; i < width; ++i) {
        const unsigned b = lsb + i;
        v |= (uint64_t)((w[b / 32] >> (b % 32)) & 1u) << i;
    }
    return v;
}
static inline uint64_t getb(uint64_t w, unsigned lsb, unsigned width) {
    return (w >> lsb) & (width >= 64 ? ~0ull : ((1ull << width) - 1));
}
template <std::size_t N>
static inline void setb(VlWide<N>& w, unsigned lsb, unsigned width, uint64_t v) {
    for (unsigned i = 0; i < width; ++i) {
        const unsigned b = lsb + i;
        const uint32_t m = 1u << (b % 32);
        if ((v >> i) & 1u) w[b / 32] |= m;
        else w[b / 32] &= ~m;
    }
}
template <typename T>
static inline void setb(T& w, unsigned lsb, unsigned width, uint64_t v) {
    const uint64_t m = (width >= 64 ? ~0ull : ((1ull << width) - 1)) << lsb;
    w = (T)(((uint64_t)w & ~m) | ((v << lsb) & m));
}
// 32-bit aligned lanes of a wide port are plain words.
template <std::size_t N>
static inline uint32_t get32(const VlWide<N>& w, unsigned lane) { return w[lane]; }
template <std::size_t N>
static inline void set32(VlWide<N>& w, unsigned lane, uint32_t v) { w[lane] = v; }

// ---------------------------------------------------------------------------------------------
// Reference wrappers (bits in, bits out).
// ---------------------------------------------------------------------------------------------
static inline float u2f(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
static inline uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static inline uint32_t ref_fmul(uint32_t a, uint32_t b) { return f2u(tsr_fmul(u2f(a), u2f(b))); }
static inline uint32_t ref_fdiv(uint32_t a, uint32_t b) { return f2u(tsr_fdiv(u2f(a), u2f(b))); }
static inline uint32_t ref_daz(uint32_t a) { return f2u(tsr_f32_daz(u2f(a))); }
static inline uint32_t ref_fadd(uint32_t a, uint32_t b) { return f2u(tsr_fadd(u2f(a), u2f(b))); }
static inline uint32_t ref_fsub(uint32_t a, uint32_t b) { return f2u(tsr_fsub(u2f(a), u2f(b))); }

// amax_candidate() + "a > amax" scan of tsr_ref.c tsr_qact (the function is static there):
// transcribed for the reduction check; tb/fp/test_fp.py also checks the result through
// tsr_qact itself.
static void ref_amax_scan(const uint32_t* x, unsigned n, uint32_t* amax, uint32_t* max) {
    uint32_t am = 0, mx = 0;
    for (unsigned i = 0; i < n; ++i) {
        const uint32_t d = ref_daz(x[i]);
        const uint32_t a = d & 0x7FFFFFFFu;
        if (a <= 0x7F800000u && a > am) {
            am = a;
            mx = d;
        }
    }
    *amax = am;
    *max = mx;
}

// ---------------------------------------------------------------------------------------------
// RNG (splitmix64) and vector generators.
// ---------------------------------------------------------------------------------------------
struct Rng {
    uint64_t s;
    static uint64_t mix(uint64_t z) {
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    // The seed is scrambled so that seed k's stream is not seed 1's stream shifted by k steps.
    explicit Rng(uint64_t seed) : s(mix(seed ^ 0x6A09E667F3BCC909ull)) {}
    uint64_t u64() { return mix(s += 0x9E3779B97F4A7C15ull); }
    uint32_t u32() { return (uint32_t)(u64() >> 32); }
    uint32_t below(uint32_t n) { return (uint32_t)(((u64() >> 32) * (uint64_t)n) >> 32); }
    bool coin() { return (u64() >> 63) != 0; }
};

static uint32_t special_mant(Rng& r) {
    switch (r.below(8)) {
        case 0: return 0;
        case 1: return 0x7FFFFF;
        case 2: return 0x7FFFFF - r.below(4);
        case 3: return r.below(8);
        case 4: return 1u << r.below(23);
        case 5: return 0x7FFFFF ^ (1u << r.below(23));
        default: return r.u32() & 0x7FFFFF;
    }
}
static uint32_t mk(uint32_t s, uint32_t e, uint32_t m) { return (s << 31) | ((e & 0xFF) << 23) | (m & 0x7FFFFF); }
static uint32_t rnd_any(Rng& r) {           // biased toward special exponents and mantissas
    switch (r.below(4)) {
        case 0: return r.u32();
        case 1: {
            static const uint32_t ex[] = {0, 1, 2, 126, 127, 128, 253, 254, 255};
            return mk(r.coin(), ex[r.below(9)], special_mant(r));
        }
        default: return mk(r.coin(), r.below(256), special_mant(r));
    }
}
static uint32_t rnd_normal(Rng& r, uint32_t emin, uint32_t emax) {
    return mk(r.coin(), emin + r.below(emax - emin + 1), r.coin() ? special_mant(r) : (r.u32() & 0x7FFFFF));
}
static uint32_t bump(uint32_t u, int d) { return (uint32_t)((int64_t)u + d); }

// Classes of operand pairs (a, b) for mul (div = 0) and div (div = 1).
// G_EDGE1/2 walk all 2^23 mantissas (k-th vector: a bijective scramble of k, so a shorter run
// samples the whole range):
//   mul EDGE1: {s, 126, m} * +-2^-126  -> exact in [2^-127, 2^-126): FTZ / min-normal decision
//   mul EDGE2: {s, 127, m} * FLT_MAX   -> FLT_MAX or Inf (overflow rounding)
//   div EDGE1: {s, 1, m} / +-2.0       -> exact in [2^-127, 2^-126)
//   div EDGE2: 1.0 / {s, 127, m} (frecip over all mantissas)
// The first 2^23 vectors of a class use positive operands; operand signs and the -127.0
// numerator (A256 iscale) of EDGE2 start at k >= 2^23 (signs are XOR-ed separately in the RTL and
// are covered by every random class).
enum { G_ANY, G_NORMAL, G_FTZ_WINDOW, G_UNDERFLOW, G_OVERFLOW, G_TIES, G_RECIP, G_QACT,
       G_EDGE1, G_EDGE2, G_NUM };
static const char* gname[G_NUM] = {"any", "normal", "ftz_window", "underflow", "overflow",
                                   "ties/exact", "recip", "qact", "edge1", "edge2"};

static const double LO = 0x1p-126 - 0x1p-150;   // flush threshold |exact| < LO

static void gen_pair(Rng& r, int g, bool div, uint64_t k, uint32_t* pa, uint32_t* pb) {
    uint32_t a = 0, b = 0;
    const uint32_t em = (uint32_t)(k * 2654435761ull) & 0x7FFFFFu;   // bijective on 23 bits
    const uint32_t es = (uint32_t)(k >> 23) & 1u, es2 = (uint32_t)(k >> 24) & 1u;
    switch (g) {
        case G_EDGE1:
            a = div ? mk(es, 1, em) : mk(es, 126, em);
            b = div ? mk(es2, 128, 0) : mk(es2, 1, 0);
            break;
        case G_EDGE2:
            if (div) {
                a = (k >> 23) % 3 == 2 ? 0xC2FE0000u : mk(es, 127, 0);
                b = mk(es2, 127, em);
            } else {
                a = mk(es, 127, em);
                b = mk(es2, 254, 0x7FFFFF);
            }
            break;
        case G_ANY:
            a = rnd_any(r);
            b = rnd_any(r);
            break;
        case G_NORMAL:
            a = rnd_normal(r, 1, 254);
            b = rnd_normal(r, 1, 254);
            break;
        case G_FTZ_WINDOW: {
            // exact result within a few ulps of 2^-126 - 2^-150 (the FTZ decision point); b is
            // drawn so that the matching a is normal (mul: b <= 1, div: b >= 1)
            b = div ? rnd_normal(r, 127, 250) : rnd_normal(r, 1, 126);
            const double bb = std::fabs((double)u2f(b));
            const double t = LO * (1.0 + (double)((int)r.below(9) - 4) * 0x1p-27);
            const double av = div ? t * bb : t / bb;
            if (!(av >= 0x1p-126 && av < 0x1p127)) { a = rnd_normal(r, 1, 254); break; }
            a = bump(f2u((float)av), (int)r.below(3) - 1) | (r.coin() << 31);
            break;
        }
        case G_UNDERFLOW: {
            // result exponent around 2^-128 .. 2^-124 (e <= 0 .. e == 2), random significands
            b = rnd_normal(r, 1, 254);
            const double bb = std::fabs((double)u2f(b));
            const double t = std::ldexp(1.0 + (double)(r.u32() & 0xFFFFFF) * 0x1p-24, -128 + (int)r.below(5));
            const double av = div ? t * bb : t / bb;
            if (!(av >= 0x1p-126 && av < 0x1p127)) { a = rnd_normal(r, 1, 254); break; }
            a = bump(f2u((float)av), (int)r.below(3) - 1) | (r.coin() << 31);
            break;
        }
        case G_OVERFLOW: {
            // result near FLT_MAX / 2^128 (rounding into Inf)
            b = rnd_normal(r, 1, 254);
            const double bb = std::fabs((double)u2f(b));
            const double t = 0x1p128 * (1.0 + (double)((int)r.below(9) - 6) * 0x1p-25);
            const double av = div ? t * bb : t / bb;
            if (!(av >= 0x1p-126 && av < 0x1p127)) { a = rnd_normal(r, 1, 254); break; }
            a = bump(f2u((float)av), (int)r.below(5) - 2) | (r.coin() << 31);
            break;
        }
        case G_TIES:
            if (!div) {
                // short significands: products are exact or exact ties at the 24-bit boundary
                const uint32_t ma = (r.u32() & 0xFFF) << 11 | (r.coin() ? (r.u32() & 0x7FF) : 0);
                const uint32_t mb = (r.u32() & 0x1FFF) << 10;
                a = mk(r.coin(), 64 + r.below(128), ma);
                b = mk(r.coin(), 64 + r.below(128), mb);
            } else {
                // exact quotients: a = b * c with a short c
                b = rnd_normal(r, 64, 190);
                const uint32_t c = mk(0, 120 + r.below(16), (r.u32() & 0x7) << 20);
                a = ref_fmul(b, c) | (r.coin() << 31);
            }
            break;
        case G_RECIP:
            a = r.coin() ? 0x3F800000u : 0xC2FE0000u;           // 1.0 or -127.0
            b = r.below(8) == 0 ? rnd_any(r) : rnd_normal(r, 1, 254);
            break;
        default: {  // G_QACT: d = amax / 127, id = 1 / d, q = x * id, iscale * x
            const uint32_t x = rnd_any(r) & (r.coin() ? 0x7FFFFFFFu : 0xFFFFFFFFu);
            if (div) {
                a = r.coin() ? x : 0x3F800000u;
                b = r.coin() ? 0x42FE0000u : x;
            } else {
                a = x;
                b = ref_fdiv(0x3F800000u, ref_fdiv(rnd_normal(r, 1, 254) & 0x7FFFFFFFu, 0x42FE0000u));
            }
            break;
        }
    }
    *pa = a;
    *pb = b;
}

// Classes of add / sub operand triples (a, b, sub); sub is random unless a class needs a given
// effective operation (eff_sub = sign(a) ^ sign(b) ^ sub).
//   uniform   uniform 32-bit patterns (mostly huge exponent differences)
//   any       rnd_any: special exponents / mantissas, zeros, subnormals, Inf, NaN
//   diff      every exponent difference 0..40 (>= 26: the small operand lands in the sticky
//             bit only), random / special mantissas, both operand orders
//   near      near-equal magnitudes (b = a +- 2^j ulps, across binade boundaries), effective
//             subtraction 7/8: cancellation of every depth (near path, LZC, exact zero)
//   ties      RNE ties at every alignment distance d = 1..24 (the bits of b below a's ulp are
//             exactly 10..0) and their +-1 ulp neighbors, effective add and sub
//   carry     significands near all ones plus a small addend: rounding carry into the next
//             binade, and exact carries (d = 0 additions)
//   overflow  exponents 254 / 225..254, effective addition 7/8: rounding to Inf, FLT_MAX ties
//   tiny      exponents 1..4: exact subnormal differences (FTZ to a signed zero) and results
//             at 2^-126 .. 2^-124
//   walk      d = 0 / 1 differences N = 2^(1-d) delta, delta = a bijective scramble of the walk
//             index k (k-th vector; k starts at seed * 2^21), exponents 1..30: near-path difference
//             patterns, every lzc and FTZ decision; 2^23 consecutive k take every delta once
//   special   operands from a table of specials (signed zeros, subnormals, min normal, FLT_MAX,
//             Inf, NaN payloads, 1.0), mixed with rnd_any
enum { A_UNIFORM, A_ANY, A_DIFF, A_NEAR, A_TIES, A_CARRY, A_OVERFLOW, A_TINY, A_WALK, A_SPECIAL,
       A_NUM };
static const char* aname[A_NUM] = {"uniform", "any", "diff", "near", "ties", "carry", "overflow",
                                   "tiny", "walk", "special"};

static uint32_t rnd_mant(Rng& r) { return r.coin() ? special_mant(r) : (r.u32() & 0x7FFFFF); }

// Sets *ps so that the effective operation is a subtraction with probability p_sub / 8.
static void eff_op(Rng& r, uint32_t a, uint32_t b, unsigned p_sub, uint32_t* ps) {
    const uint32_t want_sub = r.below(8) < p_sub;
    *ps = ((a >> 31) ^ (b >> 31) ^ want_sub) & 1u;
}

static void gen_add(Rng& r, int g, uint64_t k, uint32_t* pa, uint32_t* pb, uint32_t* ps) {
    uint32_t a = 0, b = 0, sub = r.coin();
    switch (g) {
        case A_UNIFORM:
            a = r.u32();
            b = r.u32();
            break;
        case A_ANY:
            a = rnd_any(r);
            b = rnd_any(r);
            break;
        case A_DIFF: {
            const uint32_t d = r.below(41);
            const uint32_t ea = d + 1 + r.below(254 - d);           // e_b = ea - d >= 1
            a = mk(r.coin(), ea, rnd_mant(r));
            b = mk(r.coin(), ea - d, rnd_mant(r));
            if (r.coin()) std::swap(a, b);
            break;
        }
        case A_NEAR: {
            for (;;) {
                a = mk(r.coin(), 1 + r.below(254), r.u32() & 0x7FFFFF);
                const int64_t delta = (int64_t)(1u + (r.u32() >> r.below(32))) >> r.below(24);
                const int64_t m = (int64_t)(a & 0x7FFFFFFFu) + (r.coin() ? delta : -delta);
                if (m < 0x00800000 || m > 0x7F7FFFFF) continue;
                b = (uint32_t)m | (r.u32() & 0x80000000u);
                break;
            }
            eff_op(r, a, b, 7, &sub);
            if (r.coin()) std::swap(a, b);
            break;
        }
        case A_TIES: {
            const uint32_t d = 1 + r.below(24);
            const uint32_t ea = d + 1 + r.below(254 - d);
            const uint32_t mb = (((r.u32() << d) | (1u << (d - 1))) & 0xFFFFFFu) | 0x800000u;
            a = mk(r.coin(), ea, r.u32() & 0x7FFFFF);
            b = mk(r.coin(), ea - d, mb & 0x7FFFFF);
            if (r.below(4) == 0) b = bump(b, r.coin() ? 1 : -1);
            if (r.coin()) std::swap(a, b);
            break;
        }
        case A_CARRY: {
            const uint32_t d = r.below(27);
            const uint32_t ea = 27 + r.below(228);
            a = mk(r.coin(), ea, 0x7FFFFF - r.below(r.coin() ? 4 : 64));
            b = mk(r.coin(), ea - d, rnd_mant(r));
            eff_op(r, a, b, 1, &sub);
            if (r.coin()) std::swap(a, b);
            break;
        }
        case A_OVERFLOW: {
            a = mk(r.coin(), 254, r.coin() ? 0x7FFFFF - r.below(8) : rnd_mant(r));
            b = mk(r.coin(), 254 - r.below(30), rnd_mant(r));
            eff_op(r, a, b, 1, &sub);
            if (r.coin()) std::swap(a, b);
            break;
        }
        case A_TINY: {
            a = mk(r.coin(), 1 + r.below(4), rnd_mant(r));
            b = mk(r.coin(), 1 + r.below(4), rnd_mant(r));
            eff_op(r, a, b, 6, &sub);
            break;
        }
        case A_WALK: {
            // k-th vector: delta = a bijective scramble of k (23 bits; 0 -> 1); d and e random
            const uint32_t delta = ((uint32_t)(k * 2654435761ull) & 0x7FFFFFu) | ((k & 0x7FFFFFu) ? 0u : 1u);
            const uint32_t e = 1 + r.below(30);                  // exponent of the smaller operand
            const uint32_t d = r.coin();
            uint32_t ma, mbv;                                     // N = 2 m_a - (m_b << (1 - d)) = 2^(1-d) delta
            if (d) {   // m_b = 2 m_a - delta in [2^24 - delta, 2^24)
                ma = 0x800000u + r.below((delta + 1) / 2);
                mbv = 2u * ma - delta;
            } else {   // m_b = m_a + delta < 2^24
                ma = 0x800000u + r.below(0x800000u - delta);
                mbv = ma + delta;
            }
            const uint32_t sa = r.coin();
            a = mk(sa, e + d, ma & 0x7FFFFF);
            b = mk(sa ^ r.coin(), e, mbv & 0x7FFFFF);
            eff_op(r, a, b, 8, &sub);
            if (r.coin()) std::swap(a, b);
            break;
        }
        default: {   // A_SPECIAL
            static const uint32_t sp[] = {
                0x00000000u, 0x80000000u, 0x00000001u, 0x80000001u, 0x007FFFFFu, 0x807FFFFFu,
                0x00400000u, 0x00800000u, 0x80800000u, 0x00800001u, 0x00FFFFFFu, 0x01000000u,
                0x7F7FFFFFu, 0xFF7FFFFFu, 0x7F000000u, 0x7F800000u, 0xFF800000u, 0x7FC00000u,
                0xFFC00000u, 0x7F800001u, 0xFFFFFFFFu, 0x7FBFFFFFu, 0x7FC12345u, 0x3F800000u,
                0xBF800000u, 0x33800000u, 0x34000000u};
            const unsigned ns = sizeof sp / sizeof sp[0];
            a = r.coin() ? sp[r.below(ns)] : rnd_any(r);
            b = r.coin() ? sp[r.below(ns)] : rnd_any(r);
            break;
        }
    }
    *pa = a;
    *pb = b;
    *ps = sub;
}

// ---------------------------------------------------------------------------------------------
// Harness.
// ---------------------------------------------------------------------------------------------
struct H {
    VerilatedContext ctx;
    std::unique_ptr<Vtsr_fp_sweep_tb> m;
    unsigned lanes, mlanes, dlanes, alanes, mul_lat, add_lat;
    uint64_t fails = 0, checks = 0;

    H(int argc, char** argv) {
        ctx.commandArgs(argc, argv);
        ctx.randReset(2);
        m.reset(new Vtsr_fp_sweep_tb{&ctx});
        lanes = sizeof(m->x) / 4;
        mlanes = sizeof(m->mul_a) / 4;
        dlanes = sizeof(m->div_a) / 4;
        alanes = sizeof(m->add_a) / 4;
        mul_lat = MUL_LAT_DEF;
        add_lat = ADD_LAT_DEF;
        m->clk = 0;
        m->rst_n = 0;
        m->mul_ce = 1;
        m->mul_in_valid = 0;
        m->div_in_valid = 0;
        m->div_out_ready = 1;
        m->add_ce = 1;
        m->add_in_valid = 0;
        for (int i = 0; i < 4; ++i) tick();
        m->rst_n = 1;
    }
    ~H() { m->final(); }
    void tick() {
        m->clk = 1;
        m->eval();
        m->clk = 0;
        m->eval();
    }
    void fail(const char* what, uint32_t a, uint32_t b, uint64_t got, uint64_t want) {
        if (fails < 20) {
            std::fprintf(stderr, "MISMATCH %s: in 0x%08x 0x%08x got 0x%08" PRIx64 " want 0x%08" PRIx64 "\n",
                         what, a, b, got, want);
        }
        ++fails;
    }
    void chk(const char* what, uint32_t a, uint32_t b, uint64_t got, uint64_t want) {
        ++checks;
        if (got != want) fail(what, a, b, got, want);
    }

    // ---- package functions on one batch of x values ----
    void funcs_batch(const uint32_t* xs) {
        for (unsigned i = 0; i < lanes; ++i) {
            set32(m->x, i, xs[i]);
            setb(m->h, 16 * i, 16, xs[i] & 0xFFFF);
        }
        m->eval();
        for (unsigned i = 0; i < lanes; ++i) {
            const uint32_t u = xs[i];
            const uint16_t hh = (uint16_t)(u & 0xFFFF);
            const float f = u2f(u);
            chk("f32_to_fp16", u, 0, getb(m->f2h, 16 * i, 16), tsr_f32_to_fp16(f));
            chk("q8_round_away", u, 0, getb(m->q8a, 8 * i, 8), (uint8_t)tsr_q8_round_away(f));
            chk("q8_round_even", u, 0, getb(m->q8e, 8 * i, 8), (uint8_t)tsr_q8_round_even(f));
            chk("i32_to_f32", u, 0, get32(m->i2f, i), f2u(tsr_i32_to_f32((int32_t)u)));
            chk("fp16_to_f32", hh, 0, get32(m->h2f, i), f2u(tsr_fp16_to_f32(hh)));
            const uint32_t d = ref_daz(u);
            chk("daz", u, 0, get32(m->daz, i), d);
            const bool nan = (u & 0x7FFFFFFFu) > 0x7F800000u;
            chk("is_nan", u, 0, getb(m->is_nan, i, 1), nan);
            chk("amag", u, 0, getb(m->amag, 31 * i, 31), d & 0x7FFFFFFFu);
            uint32_t am, mx;
            ref_amax_scan(&u, 1, &am, &mx);
            chk("amax_cand", u, 0, get32(m->cand, i), ((uint64_t)am << 1) | (mx >> 31));
        }
    }

    int run_funcs(uint64_t start, uint64_t end) {
        std::vector<uint32_t> xs(lanes);
        for (uint64_t u = start; u < end; u += lanes) {
            for (unsigned i = 0; i < lanes; ++i) xs[i] = (uint32_t)(u + i < end ? u + i : start);
            funcs_batch(xs.data());
        }
        return report("funcs");
    }

    int run_funcs_rand(uint64_t n, uint64_t seed) {
        Rng r(seed);
        std::vector<uint32_t> xs(lanes);
        for (uint64_t k = 0; k < n; k += lanes) {
            for (unsigned i = 0; i < lanes; ++i) {
                switch (r.below(4)) {
                    case 0: xs[i] = rnd_any(r); break;
                    case 1: xs[i] = mk(r.coin(), 100 + r.below(45), special_mant(r)); break;   // fp16 range
                    case 2: xs[i] = mk(r.coin(), 124 + r.below(13), special_mant(r)); break;   // q8 range
                    default: xs[i] = (uint32_t)(r.u32() >> r.below(32)) ^ (r.coin() ? 0xFFFFFFFFu : 0); break;
                }
            }
            funcs_batch(xs.data());
        }
        return report("funcs_rand");
    }

    int run_amax(uint64_t blocks, uint64_t seed) {
        Rng r(seed);
        std::vector<uint32_t> xs(lanes);
        static const uint32_t sp[] = {0x00000000u, 0x80000000u, 0x00000001u, 0x807FFFFFu, 0x7F800000u,
                                      0xFF800000u, 0x7FC00000u, 0xFFFFFFFFu, 0x7F800001u, 0x00800000u,
                                      0x80800000u, 0x7F7FFFFFu};
        for (uint64_t k = 0; k < blocks; ++k) {
            const unsigned mode = r.below(4);
            const uint32_t pool_n = 1 + r.below(4);
            uint32_t pool[4];
            for (unsigned j = 0; j < 4; ++j) pool[j] = rnd_normal(r, 100, 140) & 0x7FFFFFFFu;
            for (unsigned i = 0; i < lanes; ++i) {
                uint32_t v;
                if (mode == 0) v = rnd_any(r);
                else if (mode == 1) v = sp[r.below(sizeof sp / sizeof sp[0])];
                else if (mode == 2) v = pool[r.below(pool_n)] | (r.coin() << 31);    // magnitude ties
                else v = r.below(3) ? sp[r.below(sizeof sp / sizeof sp[0])] : (pool[r.below(pool_n)] | (r.coin() << 31));
                xs[i] = v;
            }
            for (unsigned i = 0; i < lanes; ++i) set32(m->x, i, xs[i]);
            m->eval();
            uint32_t am, mx;
            ref_amax_scan(xs.data(), lanes, &am, &mx);
            chk("amax tree", xs[0], xs[1], m->tree_amax, am);
            chk("max tree", xs[0], xs[1], m->tree_max, mx);
            chk("amax chain", xs[0], xs[1], m->chain_amax, am);
            chk("max chain", xs[0], xs[1], m->chain_max, mx);
        }
        return report("amax");
    }

    // ---- multipliers: random ce, every lane every advancing cycle ----
    int run_mul(uint64_t n, uint64_t seed) {
        Rng r(seed);
        struct Slot { bool v; int g; std::vector<uint32_t> a, b; };
        std::deque<Slot> pipe;   // model: inputs taken at the last mul_lat advancing edges
        uint64_t stats[G_NUM] = {0};
        uint64_t window = 0, flushed = 0, inf = 0, nanc = 0;
        auto cycle = [&](int g, bool ce, bool iv, uint64_t* k) {
            Slot s{iv, g, std::vector<uint32_t>(mlanes), std::vector<uint32_t>(mlanes)};
            uint64_t kk = *k;
            for (unsigned i = 0; i < mlanes; ++i) {
                gen_pair(r, g, false, kk++, &s.a[i], &s.b[i]);
                set32(m->mul_a, i, s.a[i]);
                set32(m->mul_b, i, s.b[i]);
            }
            m->mul_ce = ce;
            m->mul_in_valid = iv;
            tick();
            if (ce && iv) *k = kk;                // vectors actually taken
            if (ce) {
                pipe.push_back(std::move(s));
                if (pipe.size() > mul_lat) pipe.pop_front();
            }
            const bool mv = pipe.size() == mul_lat && pipe.front().v;
            for (unsigned i = 0; i < mlanes; ++i) {
                chk("mul out_valid", 0, 0, getb(m->mul_out_valid, i, 1), mv);
            }
            if (ce && mv) {                       // a fresh result (presented after this edge)
                const Slot& o = pipe.front();
                for (unsigned i = 0; i < mlanes; ++i) {
                    const uint32_t want = ref_fmul(o.a[i], o.b[i]);
                    chk("fmul", o.a[i], o.b[i], get32(m->mul_y, i), want);
                    const double ex = std::fabs((double)u2f(ref_daz(o.a[i])) * (double)u2f(ref_daz(o.b[i])));
                    window += ex >= LO && ex < 0x1p-126 - 0x1p-151;
                    flushed += ex > 0 && ex < LO;
                    inf += (want & 0x7FFFFFFFu) == 0x7F800000u;
                    nanc += want == 0x7FC00000u;
                }
                stats[o.g] += mlanes;
            }
        };
        for (int g = 0; g < G_NUM; ++g) {
            uint64_t k = 0;
            while (k < n) cycle(g, r.below(8) != 0, r.below(16) != 0, &k);
        }
        uint64_t k = 0;
        for (unsigned i = 0; i < mul_lat + 2; ++i) cycle(G_ANY, true, false, &k);   // drain
        std::printf("mul LAT=%u lanes=%u:", mul_lat, mlanes);
        for (int g = 0; g < G_NUM; ++g) std::printf(" %s=%" PRIu64, gname[g], stats[g]);
        std::printf("\n  exact in [2^-126-2^-150, 2^-126-2^-151): %" PRIu64 ", flushed: %" PRIu64
                    ", Inf: %" PRIu64 ", NaN: %" PRIu64 "\n", window, flushed, inf, nanc);
        if (window < 100 || flushed < 100) {
            std::fprintf(stderr, "FTZ window not exercised enough (%" PRIu64 ", %" PRIu64 ")\n", window, flushed);
            ++fails;
        }
        return report("mul");
    }

    // ---- dividers in lockstep, random out_ready stalls, latency check ----
    int run_div(uint64_t n, uint64_t seed) {
        Rng r(seed);
        uint64_t stats[G_NUM] = {0};
        uint64_t window = 0, flushed = 0, inf = 0, nanc = 0;
        std::vector<uint32_t> a(dlanes), b(dlanes);
        m->div_out_ready = 1;
        for (int g = 0; g < G_NUM; ++g) {
            uint64_t k = 0;
            for (uint64_t done = 0; done < n; done += dlanes) {
                // wait for in_ready (registered; all lanes identical)
                int guard = 0;
                while (!getb(m->div_in_ready, 0, 1)) {
                    tick();
                    if (++guard > 100) { std::fprintf(stderr, "div: in_ready stuck low\n"); return 1; }
                }
                for (unsigned i = 0; i < dlanes; ++i) {
                    gen_pair(r, g, true, k++, &a[i], &b[i]);
                    set32(m->div_a, i, a[i]);
                    set32(m->div_b, i, b[i]);
                }
                m->div_in_valid = 1;
                const bool stall = r.below(4) == 0;
                m->div_out_ready = !stall;
                tick();                                   // accepting edge
                m->div_in_valid = 0;
                for (unsigned i = 0; i < dlanes; ++i) set32(m->div_a, i, r.u32());   // inputs latched
                unsigned lat = 0;                         // edges after the accepting one
                while (!getb(m->div_out_valid, 0, 1)) {
                    tick();
                    if (++lat > 64) { std::fprintf(stderr, "div: no out_valid\n"); return 1; }
                }
                chk("div latency", a[0], b[0], lat, 14);
                const uint32_t y0 = get32(m->div_y, 0);
                if (stall) {                              // hold under backpressure (R2)
                    const unsigned w = 1 + r.below(5);
                    for (unsigned k = 0; k < w; ++k) {
                        tick();
                        chk("div out_valid hold", a[0], b[0], getb(m->div_out_valid, 0, 1), 1);
                        chk("div y hold", a[0], b[0], get32(m->div_y, 0), y0);
                    }
                }
                for (unsigned i = 0; i < dlanes; ++i) {
                    chk("div out_valid lanes", a[i], b[i], getb(m->div_out_valid, i, 1), 1);
                    const uint32_t want = ref_fdiv(a[i], b[i]);
                    chk("fdiv", a[i], b[i], get32(m->div_y, i), want);
                    const double da = (double)u2f(ref_daz(a[i])), db = (double)u2f(ref_daz(b[i]));
                    if (db != 0 && std::isfinite(da) && std::isfinite(db)) {
                        // classify by the exact quotient's position relative to LO (long double
                        // is only used for statistics, never for the check)
                        const long double q = std::fabs((long double)da / (long double)db);
                        window += q >= LO && q < 0x1p-126L - 0x1p-151L;
                        flushed += q > 0 && q < LO;
                    }
                    inf += (want & 0x7FFFFFFFu) == 0x7F800000u;
                    nanc += want == 0x7FC00000u;
                }
                m->div_out_ready = 1;
                tick();                                   // consume
                chk("div out_valid drop", 0, 0, getb(m->div_out_valid, 0, 1), 0);
                stats[g] += dlanes;
            }
        }
        std::printf("div lanes=%u:", dlanes);
        for (int g = 0; g < G_NUM; ++g) std::printf(" %s=%" PRIu64, gname[g], stats[g]);
        std::printf("\n  quotient in [2^-126-2^-150, 2^-126-2^-151): %" PRIu64 ", flushed: %" PRIu64
                    ", Inf: %" PRIu64 ", NaN: %" PRIu64 "\n", window, flushed, inf, nanc);
        if (window < 100 || flushed < 100) {
            std::fprintf(stderr, "FTZ window not exercised enough (%" PRIu64 ", %" PRIu64 ")\n", window, flushed);
            ++fails;
        }
        return report("div");
    }

    // ---- adders: random ce / in_valid, every lane its own sub ----
    int run_add(uint64_t n, uint64_t seed) {
        Rng r(seed);
        struct Slot { bool v; int g; std::vector<uint32_t> a, b, s; };
        std::deque<Slot> pipe;   // model: inputs taken at the last add_lat advancing edges
        uint64_t stats[A_NUM] = {0};
        uint64_t c_ftz = 0, c_mzero = 0, c_pzero = 0, c_round = 0, c_tie = 0, c_tie_up = 0;
        uint64_t c_ovf = 0, c_nan = 0, c_inf = 0, c_cancel = 0, c_bound = 0, c_dazop = 0;
        std::vector<uint32_t> prev_y(alanes);
        std::vector<uint8_t> prev_v(alanes);
        auto is_sub = [](uint32_t u) { return (u & 0x7F800000u) == 0 && (u & 0x7FFFFFu) != 0; };
        // walk index base: run s walks k in [s * 2^21, s * 2^21 + n), so runs with consecutive
        // seeds and n >= 2^21 vectors per class together walk a contiguous range (>= 2^23 for four
        // runs of 2^21: every delta of the walk class)
        const uint64_t kbase = seed << 21;
        auto cycle = [&](int g, bool ce, bool iv, uint64_t* k) {
            Slot sl{iv, g, std::vector<uint32_t>(alanes), std::vector<uint32_t>(alanes),
                    std::vector<uint32_t>(alanes)};
            uint64_t kk = *k;
            for (unsigned i = 0; i < alanes; ++i) {
                gen_add(r, g, kbase + kk++, &sl.a[i], &sl.b[i], &sl.s[i]);
                set32(m->add_a, i, sl.a[i]);
                set32(m->add_b, i, sl.b[i]);
                setb(m->add_sub, i, 1, sl.s[i]);
                prev_y[i] = get32(m->add_y, i);
                prev_v[i] = (uint8_t)getb(m->add_out_valid, i, 1);
            }
            m->add_ce = ce;
            m->add_in_valid = iv;
            tick();
            if (ce && iv) *k = kk;                // vectors actually taken
            if (ce) {
                pipe.push_back(std::move(sl));
                if (pipe.size() > add_lat) pipe.pop_front();
            } else {
                for (unsigned i = 0; i < alanes; ++i) {   // ce = 0: nothing may change
                    chk("add y hold", prev_y[i], 0, get32(m->add_y, i), prev_y[i]);
                    chk("add out_valid hold", 0, 0, getb(m->add_out_valid, i, 1), prev_v[i]);
                }
            }
            const bool mv = pipe.size() == add_lat && pipe.front().v;
            for (unsigned i = 0; i < alanes; ++i) chk("add out_valid", 0, 0, getb(m->add_out_valid, i, 1), mv);
            if (!(ce && mv)) return;              // a fresh result is presented after this edge
            const Slot& o = pipe.front();
            for (unsigned i = 0; i < alanes; ++i) {
                const uint32_t want = o.s[i] ? ref_fsub(o.a[i], o.b[i]) : ref_fadd(o.a[i], o.b[i]);
                const uint32_t got = get32(m->add_y, i);
                if (got != want && fails < 20) std::fprintf(stderr, "  (class %s)\n", aname[o.g]);
                chk(o.s[i] ? "fsub" : "fadd", o.a[i], o.b[i], got, want);
                // statistics on the exact result (long double: exact for exponent differences <= 39)
                const uint32_t da = ref_daz(o.a[i]), db = ref_daz(o.b[i]) ^ (o.s[i] << 31);
                c_dazop += is_sub(o.a[i]) || is_sub(o.b[i]);
                c_nan += want == 0x7FC00000u;
                const bool xa = (da & 0x7F800000u) == 0x7F800000u, xb = (db & 0x7F800000u) == 0x7F800000u;
                if ((want & 0x7FFFFFFFu) == 0x7F800000u) {
                    ++c_inf;
                    c_ovf += !xa && !xb;
                }
                if (xa || xb) continue;           // Inf / NaN operand: no finite statistics
                const long double fa = (long double)u2f(da), fb = (long double)u2f(db);
                const long double ex = fa + fb, ax = std::fabs(ex);
                c_mzero += want == 0x80000000u;
                c_pzero += want == 0x00000000u;
                c_ftz += ax != 0 && ax < 0x1p-126L;
                c_bound += ax >= 0x1p-126L && ax < 0x1p-124L;
                c_cancel += ax != 0 && ax < std::fmax(std::fabs(fa), std::fabs(fb)) * 0x1p-8L;
                const long double wf = (long double)u2f(want);
                if ((want & 0x7F800000u) != 0x7F800000u && ax >= 0x1p-126L && wf != ex) {
                    ++c_round;
                    const long double ulp = std::ldexp(1.0L, (int)((want >> 23) & 0xFF) - 150);
                    const long double dd = std::fabs(wf - ex);
                    const bool up = std::fabs(wf) > ax;
                    // a tie is half an ulp of the result, or a quarter when it rounded up into
                    // the next binade (the result's ulp is then twice the operand binade's)
                    if (dd == ulp / 2 || (up && (want & 0x7FFFFFu) == 0 && dd == ulp / 4)) {
                        ++c_tie;
                        c_tie_up += up;
                    }
                }
            }
            stats[o.g] += alanes;
        };
        for (int g = 0; g < A_NUM; ++g) {
            uint64_t k = 0;
            while (k < n) cycle(g, r.below(8) != 0, r.below(16) != 0, &k);
        }
        uint64_t k = 0;
        for (unsigned i = 0; i < add_lat + 2; ++i) cycle(A_ANY, true, false, &k);   // drain
        uint64_t total = 0;
        std::printf("add LAT=%u lanes=%u:", add_lat, alanes);
        for (int g = 0; g < A_NUM; ++g) {
            std::printf(" %s=%" PRIu64, aname[g], stats[g]);
            total += stats[g];
        }
        std::printf("\n  total %" PRIu64 " results; FTZ (exact subnormal) %" PRIu64 ", [2^-126, 2^-124) %" PRIu64
                    ", +0 %" PRIu64 ", -0 %" PRIu64 ", cancellation >= 8 bits %" PRIu64 "\n  rounded %" PRIu64
                    ", ties %" PRIu64 " (rounded up %" PRIu64 "), overflow %" PRIu64 ", Inf %" PRIu64
                    ", NaN %" PRIu64 ", with a DAZ'd operand %" PRIu64 "\n",
                    total, c_ftz, c_bound, c_pzero, c_mzero, c_cancel, c_round, c_tie, c_tie_up, c_ovf, c_inf,
                    c_nan, c_dazop);
        // coverage floor (for runs of >= 2^14 vectors per class): every class of the contract
        const uint64_t need = n >= (1u << 14) ? 100 : 0;
        if (c_ftz < need || c_mzero < need || c_pzero < need || c_tie < need || c_tie_up < need ||
            c_tie - c_tie_up < need || c_ovf < need || c_nan < need || c_cancel < need || c_bound < need ||
            c_dazop < need) {
            std::fprintf(stderr, "add coverage too low\n");
            ++fails;
        }
        return report("add");
    }

    int report(const char* what) {
        std::printf("%s: %" PRIu64 " checks, %" PRIu64 " mismatches\n", what, checks, fails);
        return fails ? 1 : 0;
    }
};

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s funcs|funcs_rand|amax|mul|div|add <a> <b>\n", argv[0]);
        return 2;
    }
    const std::string mode = argv[1];
    const uint64_t p1 = std::strtoull(argv[2], nullptr, 0);
    const uint64_t p2 = std::strtoull(argv[3], nullptr, 0);
    H h(argc, argv);
    // Mandatory vectors (spec Implementation clarifications, "FTZ decision point") also go through the RTL multiplier /
    // divider in every mul / div run (they are part of the G_FTZ_WINDOW class as well).
    if (mode == "funcs") return h.run_funcs(p1, p2);
    if (mode == "funcs_rand") return h.run_funcs_rand(p1, p2);
    if (mode == "amax") return h.run_amax(p1, p2);
    if (mode == "mul") return h.run_mul(p1, p2);
    if (mode == "div") return h.run_div(p1, p2);
    if (mode == "add") return h.run_add(p1, p2);
    std::fprintf(stderr, "unknown mode %s\n", mode.c_str());
    return 2;
}
