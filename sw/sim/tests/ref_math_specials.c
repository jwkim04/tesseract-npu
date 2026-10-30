// ref_math_specials.c — directed cases for tsr_ref_math (VU §6.3–§6.6): special values and
// the specials table of §6.5, the EXPSUB and GEGLU16 rules, symmetries that hold by
// construction, the binary16 helpers, and a second transcription of the integer steps of EXPK
// (int64 arithmetic shifts) and SIN/COS (unsigned __int128) compared bit for bit with the
// portable code on a strided sweep. Accuracy is ref_math_acc's job; the independent numpy
// model is the bit-level oracle (VU §10 item 1).

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "test_util.h"
#include "tsr_ref.h"
#include "tsr_ref_math.h"

#define CANON 0x7FC00000u
#define PINF  0x7F800000u
#define NINF  0xFF800000u
#define ONE   0x3F800000u
#define NONE_ 0xBF800000u
#define HALF  0x3F000000u

static void eqbits(const char * what, uint32_t x, float got, uint32_t want, int line) {
    ++g_checks;
    if (f2u(got) != want) {
        fprintf(stderr, "FAIL line %d: %s(0x%08x) = 0x%08x, want 0x%08x\n", line, what, x, f2u(got), want);
        if (++g_fail >= 40) exit(1);
    }
}
#define EQ(what, x, got, want) eqbits(what, x, got, want, __LINE__)

static const uint32_t k_nans[] = { 0x7FC00000u, 0xFFC00000u, 0x7F800001u, 0xFFFFFFFFu, 0x7FBFFFFFu };

static void test_expk(void) {
    for (unsigned k = 0; k <= TSR_M_NG; ++k) {
        EQ("expk(+0)", 0, tsr_m_expk(u2f(0x00000000u), k), ONE);
        EQ("expk(-0)", 0, tsr_m_expk(u2f(0x80000000u), k), ONE);
        EQ("expk(+sub)", 1, tsr_m_expk(u2f(0x007FFFFFu), k), ONE);   // DAZ
        EQ("expk(-sub)", 2, tsr_m_expk(u2f(0x80000001u), k), ONE);
        for (size_t i = 0; i < sizeof k_nans / sizeof k_nans[0]; ++i) {
            EQ("expk(NaN)", k_nans[i], tsr_m_expk(u2f(k_nans[i]), k), CANON);
        }
        const int neg = k != TSR_M_P1;   // sK
        EQ("expk(+inf)", PINF, tsr_m_expk(u2f(PINF), k), neg ? 0u : PINF);
        EQ("expk(-inf)", NINF, tsr_m_expk(u2f(NINF), k), neg ? PINF : 0u);
        // Step 5: Ev > 156 saturates; Ev = 156 goes through step 6 (T >= 2^40 saturates too).
        EQ("expk(2^30)", 0x4E800000u, tsr_m_expk(u2f(0x4E800000u), k), neg ? 0u : PINF);
        EQ("expk(-2^30)", 0xCE800000u, tsr_m_expk(u2f(0xCE800000u), k), neg ? PINF : 0u);
        EQ("expk(2^29)", 0x4E000000u, tsr_m_expk(u2f(0x4E000000u), k), neg ? 0u : PINF);
        EQ("expk(FLT_MAX)", 0x7F7FFFFFu, tsr_m_expk(u2f(0x7F7FFFFFu), k), neg ? 0u : PINF);
        // Tiny |v|: T = 0 -> exactly 1; the smallest normal and 2^-60 round to 1 as well.
        EQ("expk(2^-126)", 0x00800000u, tsr_m_expk(u2f(0x00800000u), k), ONE);
        EQ("expk(-2^-60)", 0xA1800000u, tsr_m_expk(u2f(0xA1800000u), k), ONE);
    }
    EQ("expk(kappa 4)", 0, tsr_m_expk(1.0f, 4u), CANON);
    EQ("exp == expk P1", ONE, tsr_m_exp(1.0f), f2u(tsr_m_expk(1.0f, TSR_M_P1)));
    // Results that are exact powers of two: v = n·ln2 is not an f32, but EXPK(v) with T an exact
    // multiple of 2^32 (F = 0) is exactly 2^n. v = 2^k with k >= 0 never gives F = 0, so use the
    // structure instead: EXPK(0) = 1 above; ln2·LK-based checks are in the exhaustive tests.
    // Underflow: exp(-104) < 2^-149 -> +0 (never -0, never subnormal).
    EQ("exp(-104)", 0xC2D00000u, tsr_m_exp(-104.0f), 0u);
    EQ("exp(-88)", 0xC2B00000u, tsr_m_exp(-88.0f), 0u);          // 6.05e-39 < 2^-126 -> +0
    EQ("exp(89)", 0x42B20000u, tsr_m_exp(89.0f), PINF);
}

static void test_composites(void) {
    // VU §6.5 specials table.
    EQ("exp(-inf)", NINF, tsr_m_exp(u2f(NINF)), 0u);
    EQ("exp(+inf)", PINF, tsr_m_exp(u2f(PINF)), PINF);
    EQ("sig(-inf)", NINF, tsr_m_sigmoid(u2f(NINF)), 0u);
    EQ("sig(+inf)", PINF, tsr_m_sigmoid(u2f(PINF)), ONE);
    EQ("silu(-inf)", NINF, tsr_m_silu(u2f(NINF)), CANON);
    EQ("silu(+inf)", PINF, tsr_m_silu(u2f(PINF)), PINF);
    EQ("gelu(-inf)", NINF, tsr_m_gelu_tanh(u2f(NINF)), CANON);
    EQ("gelu(+inf)", PINF, tsr_m_gelu_tanh(u2f(PINF)), PINF);
    EQ("gelu16(-inf)", NINF, tsr_m_gelu16(u2f(NINF)), 0u);
    EQ("gelu16(+inf)", PINF, tsr_m_gelu16(u2f(PINF)), PINF);
    EQ("tanh(-inf)", NINF, tsr_m_tanh(u2f(NINF)), NONE_);
    EQ("tanh(+inf)", PINF, tsr_m_tanh(u2f(PINF)), ONE);
    // Zeros and subnormals (DAZ).
    EQ("sig(+0)", 0, tsr_m_sigmoid(0.0f), HALF);
    EQ("sig(-0)", 0x80000000u, tsr_m_sigmoid(u2f(0x80000000u)), HALF);
    EQ("silu(+0)", 0, tsr_m_silu(0.0f), 0u);
    EQ("silu(-0)", 0x80000000u, tsr_m_silu(u2f(0x80000000u)), 0x80000000u);
    EQ("silu(+sub)", 1, tsr_m_silu(u2f(1u)), 0u);
    EQ("gelu(+0)", 0, tsr_m_gelu_tanh(0.0f), 0u);
    EQ("gelu(-0)", 0x80000000u, tsr_m_gelu_tanh(u2f(0x80000000u)), 0x80000000u);
    EQ("gelu(-sub)", 0x80000001u, tsr_m_gelu_tanh(u2f(0x80000001u)), 0x80000000u);
    EQ("tanh(+0)", 0, tsr_m_tanh(0.0f), 0u);
    EQ("tanh(-0)", 0x80000000u, tsr_m_tanh(u2f(0x80000000u)), 0x80000000u);
    EQ("tanh(-sub)", 0x807FFFFFu, tsr_m_tanh(u2f(0x807FFFFFu)), 0x80000000u);
    EQ("tanh(FLT_MAX)", 0x7F7FFFFFu, tsr_m_tanh(u2f(0x7F7FFFFFu)), ONE);
    EQ("tanh(-20)", 0xC1A00000u, tsr_m_tanh(-20.0f), NONE_);
    // GELU of a large negative finite x is -0 (VU §6.5).
    EQ("gelu(-100)", 0xC2C80000u, tsr_m_gelu_tanh(-100.0f), 0x80000000u);
    EQ("gelu(-1e19)", f2u(-1e19f), tsr_m_gelu_tanh(-1e19f), 0x80000000u);
    EQ("gelu(-FLT_MAX)", 0xFF7FFFFFu, tsr_m_gelu_tanh(u2f(0xFF7FFFFFu)), 0x80000000u);
    EQ("gelu(FLT_MAX)", 0x7F7FFFFFu, tsr_m_gelu_tanh(u2f(0x7F7FFFFFu)), 0x7F7FFFFFu);
    // NaN inputs give the canonical NaN everywhere.
    for (size_t i = 0; i < sizeof k_nans / sizeof k_nans[0]; ++i) {
        const float n = u2f(k_nans[i]);
        EQ("sig(NaN)", k_nans[i], tsr_m_sigmoid(n), CANON);
        EQ("silu(NaN)", k_nans[i], tsr_m_silu(n), CANON);
        EQ("gelu(NaN)", k_nans[i], tsr_m_gelu_tanh(n), CANON);
        EQ("gelu16(NaN)", k_nans[i], tsr_m_gelu16(n), CANON);
        EQ("tanh(NaN)", k_nans[i], tsr_m_tanh(n), CANON);
        EQ("sin(NaN)", k_nans[i], tsr_m_sin(n), CANON);
        EQ("cos(NaN)", k_nans[i], tsr_m_cos(n), CANON);
        EQ("geglu16(NaN, 1)", k_nans[i], tsr_m_geglu16(n, 1.0f), CANON);
        EQ("expsub(NaN, 0)", k_nans[i], tsr_m_expsub(n, 0.0f), CANON);
    }
    // TANH paths: 2^-6 - ulp takes the polynomial, 2^-6 the exp path; both are odd.
    rng_t rng = { 0x7A11ull };
    for (int i = 0; i < 1000000; ++i) {
        const uint32_t x = rng_u32(&rng) & 0x7FFFFFFFu;
        if (x > 0x7F800000u) continue;
        CHECK(f2u(tsr_m_tanh(u2f(x | 0x80000000u))) == (f2u(tsr_m_tanh(u2f(x))) ^ 0x80000000u),
              "tanh not odd at 0x%08x", x);
        float s1, c1, s2, c2;
        tsr_m_sincos(u2f(x), &s1, &c1);
        tsr_m_sincos(u2f(x | 0x80000000u), &s2, &c2);
        CHECK(f2u(s2) == (f2u(s1) ^ 0x80000000u) || (f2u(s1) == CANON && f2u(s2) == CANON),
              "sin not odd at 0x%08x", x);
        CHECK(f2u(c2) == f2u(c1), "cos not even at 0x%08x", x);
        CHECK(f2u(tsr_m_sin(u2f(x))) == f2u(s1) && f2u(tsr_m_cos(u2f(x))) == f2u(c1), "sin/cos != sincos at 0x%08x", x);
    }
    {
        const float lo = u2f(0x3C7FFFFFu), hi = u2f(0x3C800000u);
        const float q = tsr_fmul(lo, lo);
        const float want = tsr_fmul(tsr_fmul(tsr_fsub(3.0f, q), lo), u2f(0x3EAAAAABu));
        EQ("tanh small path", 0x3C7FFFFFu, tsr_m_tanh(lo), f2u(want));
        const float e = tsr_m_expk(hi, TSR_M_N2);
        const float want2 = tsr_fmul(tsr_fsub(1.0f, e), tsr_frecip(tsr_fadd(1.0f, e)));
        EQ("tanh exp path", 0x3C800000u, tsr_m_tanh(hi), f2u(want2));
    }
    // EXPSUB: a = -inf gives +0 whatever x is (VU-21); otherwise EXPK(FS(a, x), P1).
    const uint32_t xs[] = { NINF, PINF, CANON, 0u, 0x80000000u, 0x42C80000u, 0xC2C80000u };
    for (size_t i = 0; i < sizeof xs / sizeof xs[0]; ++i) {
        EQ("expsub(-inf, x)", xs[i], tsr_m_expsub(u2f(NINF), u2f(xs[i])), 0u);
    }
    EQ("expsub(+inf, +inf)", PINF, tsr_m_expsub(u2f(PINF), u2f(PINF)), CANON);
    EQ("expsub(0, -inf)", NINF, tsr_m_expsub(0.0f, u2f(NINF)), PINF);
    EQ("expsub(3, 3)", 0x40400000u, tsr_m_expsub(3.0f, 3.0f), ONE);
    EQ("expsub(-max, max)", 0xFF7FFFFFu, tsr_m_expsub(u2f(0xFF7FFFFFu), u2f(0x7F7FFFFFu)), 0u);
    EQ("expsub(1, 2)", ONE, tsr_m_expsub(1.0f, 2.0f), f2u(tsr_m_exp(-1.0f)));
    // GEGLU16: DAZ(a) <= -10 gives +0 whatever b is; else FM(GELU16(a), b).
    EQ("geglu16(-10, inf)", 0xC1200000u, tsr_m_geglu16(-10.0f, u2f(PINF)), 0u);
    EQ("geglu16(-inf, NaN)", NINF, tsr_m_geglu16(u2f(NINF), u2f(CANON)), 0u);
    EQ("geglu16(10, 2)", 0x41200000u, tsr_m_geglu16(10.0f, 2.0f), 0x41A00000u);
    EQ("geglu16(1, -3)", ONE, tsr_m_geglu16(1.0f, -3.0f), f2u(tsr_fmul(tsr_m_gelu16(1.0f), -3.0f)));
    EQ("geglu16(+inf, 0)", PINF, tsr_m_geglu16(u2f(PINF), 0.0f), CANON);
    // GELU16 clamps (x' = DAZ(x)).
    EQ("gelu16(-10)", 0xC1200000u, tsr_m_gelu16(-10.0f), 0u);
    EQ("gelu16(10)", 0x41200000u, tsr_m_gelu16(10.0f), 0x41200000u);
    EQ("gelu16(12.5)", 0x41480000u, tsr_m_gelu16(12.5f), 0x41480000u);
    EQ("gelu16(-12.5)", 0xC1480000u, tsr_m_gelu16(-12.5f), 0u);
    EQ("gelu16(+sub)", 0x00000005u, tsr_m_gelu16(u2f(5u)), 0u);
    EQ("gelu16(-sub)", 0x80000005u, tsr_m_gelu16(u2f(0x80000005u)), 0x80000000u);
    // GELU16 rounds its input to fp16 first: x and F32(F16(x)) give the same result inside (-10, 10).
    for (int i = 0; i < 100000; ++i) {
        const float x = (float)((rng_unit(&rng) * 2.0 - 1.0) * 9.99);
        const float h = tsr_f16_to_f32(tsr_f32_to_f16_rne(x));
        if (!(h > -10.0f && h < 10.0f)) continue;
        CHECK(f2u(tsr_m_gelu16(x)) == f2u(tsr_m_gelu16(h)), "gelu16 input rounding at 0x%08x", f2u(x));
        const uint32_t y = f2u(tsr_m_gelu16(x));
        CHECK(f2u(tsr_f16_to_f32(tsr_f32_to_f16_rne(u2f(y)))) == y, "gelu16 result not fp16-exact at 0x%08x", f2u(x));
    }
}

static void test_f16(void) {
    struct { uint32_t f; uint16_t h; } c[] = {
        { 0x00000001u, 0x0000u }, { 0x80000001u, 0x8000u },       // f32 subnormal -> DAZ -> 0
        { 0x33800000u, 0x0001u }, { 0xB3800000u, 0x8001u },       // 2^-24: smallest fp16 subnormal
        { 0x33000000u, 0x0000u },                                 // 2^-25: tie -> even (0)
        { 0x33400000u, 0x0001u },                                 // 0.75·2^-24 -> 1
        { 0x33C00000u, 0x0002u },                                 // 1.5·2^-24: tie -> even (2)
        { 0x387FC000u, 0x03FFu },                                 // largest fp16 subnormal
        { 0x38800000u, 0x0400u },                                 // 2^-14: smallest fp16 normal
        { 0x477FE000u, 0x7BFFu }, { 0x477FF000u, 0x7C00u },       // 65504; 65520 -> inf
        { 0x7F800000u, 0x7C00u }, { 0xFF800000u, 0xFC00u },
        { 0x7FC00000u, 0x7E00u }, { 0x3F800000u, 0x3C00u }, { 0x41200000u, 0x4900u },
    };
    for (size_t i = 0; i < sizeof c / sizeof c[0]; ++i) {
        const uint16_t got = tsr_f32_to_f16_rne(u2f(c[i].f));
        CHECK(got == c[i].h, "f32_to_f16_rne(0x%08x) = 0x%04x, want 0x%04x", c[i].f, got, c[i].h);
    }
    CHECK(f2u(tsr_f16_to_f32(0x0001u)) == 0x33800000u, "f16_to_f32(1)");
    CHECK(f2u(tsr_f16_to_f32(0x03FFu)) == 0x387FC000u, "f16_to_f32(0x3ff)");
    CHECK(f2u(tsr_f16_to_f32(0xFC00u)) == 0xFF800000u, "f16_to_f32(-inf)");
    for (uint32_t h = 0; h < 0x10000u; ++h) {
        if ((h & 0x7C00u) == 0x7C00u && (h & 0x3FFu)) continue;
        CHECK(tsr_f32_to_f16_rne(tsr_f16_to_f32((uint16_t)h)) == h, "f16 round trip 0x%04x", h);
    }
}

static void test_sincos_special(void) {
    const uint32_t bad[] = { PINF, NINF, CANON, 0xFFC00001u };
    for (size_t i = 0; i < sizeof bad / sizeof bad[0]; ++i) {
        float s, c;
        tsr_m_sincos(u2f(bad[i]), &s, &c);
        EQ("sin(nan/inf)", bad[i], s, CANON);
        EQ("cos(nan/inf)", bad[i], c, CANON);
    }
    // E = 0 and E < 115: SIN = θ (DAZ'd), COS = ONE.
    const uint32_t small[] = { 0u, 0x80000000u, 0x00000001u, 0x807FFFFFu, 0x00800000u, 0x397FFFFFu, 0xB97FFFFFu };
    for (size_t i = 0; i < sizeof small / sizeof small[0]; ++i) {
        const uint32_t x = small[i];
        const uint32_t xd = ((x & 0x7F800000u) == 0) ? (x & 0x80000000u) : x;
        EQ("sin(small)", x, tsr_m_sin(u2f(x)), xd);
        EQ("cos(small)", x, tsr_m_cos(u2f(x)), ONE);
    }
    // sh < 0 (E > 214): F = 0, so r = +0, sn = +0, cs = 1 and q = k mod 4 = 0 for E >= 216.
    EQ("sin(2^100)", 0x71800000u, tsr_m_sin(u2f(0x71800000u)), 0u);
    EQ("cos(2^100)", 0x71800000u, tsr_m_cos(u2f(0x71800000u)), ONE);
    EQ("sin(-2^100)", 0xF1800000u, tsr_m_sin(u2f(0xF1800000u)), 0x80000000u);
    EQ("sin(FLT_MAX)", 0x7F7FFFFFu, tsr_m_sin(u2f(0x7F7FFFFFu)), 0u);
}

#ifdef __SIZEOF_INT128__
typedef unsigned __int128 u128;

// EXPK steps 6-7 with a signed 64-bit t and an arithmetic shift, as the document writes them.
static float expk_alt(float v, unsigned kappa) {
    static const uint64_t LK[4] = { 0x5C551D94AEull, 0x5C551D94AEull, 0xB8AA3B295Cull, 0x6969F3DD6ull };
    uint32_t u = f2u(v);
    if ((u & 0x7F800000u) == 0) u &= 0x80000000u;
    if ((u & 0x7FFFFFFFu) > 0x7F800000u) return u2f(CANON);
    const uint32_t sgn = (u >> 31) ^ (kappa != 0);
    const uint32_t a = u & 0x7FFFFFFFu;
    if (a == 0x7F800000u) return u2f(sgn ? 0u : PINF);
    if (a == 0) return u2f(ONE);
    const uint32_t Ev = a >> 23;
    if (Ev > 156) return u2f(sgn ? 0u : PINF);
    const u128 P = (u128)(0x800000u | (a & 0x7FFFFFu)) * LK[kappa];
    const uint64_t T = (156 - Ev >= 64) ? 0 : (uint64_t)(P >> (156 - Ev));   // P < 2^64
    if (T >= (1ull << 40)) return u2f(sgn ? 0u : PINF);
    const int64_t t = sgn ? -(int64_t)T : (int64_t)T;
    int64_t n = t >> 32;                      // arithmetic on GCC/Clang
    const uint64_t F = (uint64_t)t & 0xFFFFFFFFull;
    static const uint32_t T64[64] = {
        0x40000000u, 0x40B268FAu, 0x4166C34Cu, 0x421D1462u, 0x42D561B4u, 0x438FB0CBu, 0x444C0740u, 0x450A6ABBu,
        0x45CAE0F2u, 0x468D6FAEu, 0x47521CC6u, 0x4818EE22u, 0x48E1E9BAu, 0x49AD1598u, 0x4A7A77D4u, 0x4B4A169Cu,
        0x4C1BF829u, 0x4CF022CAu, 0x4DC69CDDu, 0x4E9F6CD4u, 0x4F7A9930u, 0x50582888u, 0x51382182u, 0x521A8AD7u,
        0x52FF6B55u, 0x53E6C9DAu, 0x54D0AD5Au, 0x55BD1CDBu, 0x56AC1F75u, 0x579DBC57u, 0x5891FAC1u, 0x5988E209u,
        0x5A82799Au, 0x5B7EC8F2u, 0x5C7DD7A4u, 0x5D7FAD59u, 0x5E8451D0u, 0x5F8BCCDBu, 0x60962665u, 0x61A3666Du,
        0x62B39509u, 0x63C6BA64u, 0x64DCDEC3u, 0x65F60A7Fu, 0x6712460Bu, 0x683199EDu, 0x69540EC9u, 0x6A79AD56u,
        0x6BA27E65u, 0x6CCE8AE1u, 0x6DFDDBCCu, 0x6F307A41u, 0x70666F76u, 0x719FC4B9u, 0x72DC8374u, 0x741CB528u,
        0x75606374u, 0x76A7980Fu, 0x77F25CCEu, 0x7940BB9Eu, 0x7A92BE8Bu, 0x7BE86FBAu, 0x7D41D96Eu, 0x7E9F0606u,
    };
    const uint64_t j = F >> 26, G = F & 0x3FFFFFFu;
    const uint64_t PF = (G * (0xB17217F8ull + ((G * (0x3D7F7BFFull + ((G * 0x0E35846Cull) >> 32))) >> 32))) >> 32;
    const uint64_t M = T64[j] + (((uint64_t)T64[j] * PF) >> 32);
    const int drop = (M >> 31) ? 8 : 7;
    n += (drop == 8);
    uint64_t sig = M >> drop;
    const uint64_t rem = M & ((1ull << drop) - 1), half = 1ull << (drop - 1);
    if (rem > half || (rem == half && (sig & 1))) ++sig;
    if (sig == (1ull << 24)) { sig >>= 1; ++n; }
    const int64_t be = n + 127;
    if (be >= 255) return u2f(PINF);
    if (be <= 0) return u2f(0u);
    return u2f((uint32_t)(be << 23) | (uint32_t)(sig & 0x7FFFFFu));
}

// SIN/COS steps 3-5 with unsigned __int128; the polynomial and quadrant steps as in the C file.
static void sincos_alt(float t, float * so, float * co) {
    uint32_t u = f2u(t);
    if ((u & 0x7F800000u) == 0) u &= 0x80000000u;
    const uint32_t a = u & 0x7FFFFFFFu;
    if (a >= 0x7F800000u) { *so = *co = u2f(CANON); return; }
    const uint32_t E = a >> 23;
    if (E < 115) { *so = u2f(u); *co = u2f(ONE); return; }
    const u128 C64 = 0xA2F9836E4E44152Aull, PIO2 = 0x6487ED5110B4611Aull;
    const u128 P = (u128)(0x800000u | (a & 0x7FFFFFu)) * C64;
    const int sh = 214 - (int)E;
    u128 k;
    uint64_t F;
    const u128 m62 = ((u128)1 << 62) - 1;
    if (sh >= 62)     { k = sh >= 128 ? 0 : P >> sh; F = (uint64_t)((P >> (sh - 62)) & m62); }
    else if (sh >= 0) { k = P >> sh; F = (uint64_t)((P << (62 - sh)) & m62); }
    else              { k = P << -sh; F = 0; }
    int64_t Fs = (int64_t)F;
    if (F >= (1ull << 61)) { k += 1; Fs = (int64_t)F - (int64_t)(1ull << 62); }
    const unsigned q = (unsigned)(k & 3);
    float r = 0.0f;
    if (Fs != 0) {
        const int neg = Fs < 0;
        const u128 R = (u128)(uint64_t)(neg ? -Fs : Fs) * PIO2;   // < 2^124
        int b = 127;
        while (!((R >> b) & 1)) --b;
        const int s = b - 23;
        uint64_t sig = (uint64_t)(R >> s);
        const u128 rem = R & (((u128)1 << s) - 1), halfv = (u128)1 << (s - 1);
        uint32_t be = (uint32_t)(b + 3);
        if (rem > halfv || (rem == halfv && (sig & 1))) {
            if (++sig == (1ull << 24)) { sig >>= 1; ++be; }
        }
        r = u2f((neg ? 0x80000000u : 0u) | (be << 23) | ((uint32_t)sig & 0x7FFFFFu));
    }
    const float z = tsr_fmul(r, r);
    float p = tsr_fadd(tsr_fmul(u2f(0xB94CA1F9u), z), u2f(0x3C08839Eu));
    p = tsr_fadd(tsr_fmul(p, z), u2f(0xBE2AAAA3u));
    const float sn = tsr_fadd(tsr_fmul(tsr_fmul(p, z), r), r);
    p = tsr_fadd(tsr_fmul(u2f(0x37CCF5CEu), z), u2f(0xBAB6061Au));
    p = tsr_fadd(tsr_fmul(p, z), u2f(0x3D2AAAA5u));
    p = tsr_fmul(tsr_fmul(p, z), z);
    const float cs = tsr_fadd(tsr_fsub(p, tsr_fmul(u2f(HALF), z)), u2f(ONE));
    const uint32_t S[4] = { f2u(sn), f2u(cs), f2u(sn) ^ 0x80000000u, f2u(cs) ^ 0x80000000u };
    const uint32_t C[4] = { f2u(cs), f2u(sn) ^ 0x80000000u, f2u(cs) ^ 0x80000000u, f2u(sn) };
    *so = u2f(S[q] ^ (u & 0x80000000u));
    *co = u2f(C[q]);
}

static void test_alt(uint64_t start, uint64_t count, uint64_t stride) {
    long n = 0;
    // A strided sweep over the bit space plus every exponent with random mantissas.
    for (uint64_t i = 0; i < count; ++i) {
        const uint32_t x = (uint32_t)(start + i * stride);
        for (unsigned k = 0; k <= TSR_M_NG; ++k) {
            const uint32_t a = f2u(tsr_m_expk(u2f(x), k)), b = f2u(expk_alt(u2f(x), k));
            CHECK(a == b, "expk(0x%08x, %u): 0x%08x vs alt 0x%08x", x, k, a, b);
        }
        float s1, c1, s2, c2;
        tsr_m_sincos(u2f(x), &s1, &c1);
        sincos_alt(u2f(x), &s2, &c2);
        CHECK(f2u(s1) == f2u(s2) && f2u(c1) == f2u(c2), "sincos(0x%08x): 0x%08x 0x%08x vs alt 0x%08x 0x%08x",
              x, f2u(s1), f2u(c1), f2u(s2), f2u(c2));
        ++n;
    }
    rng_t rng = { 0xA17ull };
    for (uint32_t E = 0; E < 255; ++E) {
        for (int i = 0; i < 4096; ++i) {
            const uint32_t x = (rng_u32(&rng) & 0x807FFFFFu) | (E << 23);
            float s1, c1, s2, c2;
            tsr_m_sincos(u2f(x), &s1, &c1);
            sincos_alt(u2f(x), &s2, &c2);
            CHECK(f2u(s1) == f2u(s2) && f2u(c1) == f2u(c2), "sincos(0x%08x) vs alt", x);
            for (unsigned k = 0; k <= TSR_M_NG; ++k) {
                CHECK(f2u(tsr_m_expk(u2f(x), k)) == f2u(expk_alt(u2f(x), k)), "expk(0x%08x, %u) vs alt", x, k);
            }
            ++n;
        }
    }
    printf("ref_math_specials: %ld inputs against the int64 / __int128 transcriptions\n", n);
}
#endif

// Usage: ref_math_specials [<start> <count>]: the transcription comparison over the inputs
// start .. start + count - 1 (stride 1; e.g. 8 processes over 2^29 inputs each cover all 2^32)
// instead of the default 2^22 inputs with stride 1021.
int main(int argc, char ** argv) {
    uint64_t start = 0, count = 1ull << 22, stride = 1021;
    if (argc == 3) {
        start = strtoull(argv[1], NULL, 0);
        count = strtoull(argv[2], NULL, 0);
        stride = 1;
    } else if (argc != 1) {
        fprintf(stderr, "usage: ref_math_specials [<start> <count>]\n");
        return 2;
    }
    test_expk();
    test_composites();
    test_f16();
    test_sincos_special();
#ifdef __SIZEOF_INT128__
    test_alt(start, count, stride);
#else
    (void)start; (void)count; (void)stride;
    printf("ref_math_specials: no __int128, transcription comparison skipped\n");
#endif
    return test_finish("ref_math_specials");
}
