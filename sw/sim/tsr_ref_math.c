// tsr_ref_math.c — the VU's approximate and composite math, bit for bit.
//
// Normative source: docs/design/02-vector-unit.md §6.3 (EXPK), §6.5 (composites), §6.6 (SIN/COS)
// and Appendix A (constants). Step numbers in the comments are the document's.
//
// Implementation rules (docs/design/06-software-verification.md §4.5)
//   - Every f32 operation is a call to tsr_fadd/fsub/fmul/frecip (tsr_ref.c: DAZ, RNE, FTZ,
//     canonical NaN). Everything else works on integer bit patterns; no libm, no float
//     operators. The file is plain C11 and builds for RV32: integers are at most uint64_t, and
//     the two exact SIN/COS products use the two-word helper below instead of __int128.
//   - DAZ(x) maps exponent-0 encodings to signed zero; "-x" flips the sign bit (VU §6.1).
//   - Shifts follow VU §6.1: '>>' is a floor shift and a shift by >= the width gives 0.

#include "tsr_ref_math.h"

#include <string.h>

#include "tsr_ref.h"

#if defined(__FAST_MATH__)
#error "tsr_ref_math.c must not be compiled with -ffast-math"
#endif

#define F32_SIGN 0x80000000u
#define F32_EXP  0x7F800000u
#define F32_MANT 0x007FFFFFu
#define F32_ABS  0x7FFFFFFFu

#define B_ZERO   0x00000000u
#define B_ONE    0x3F800000u   // ONE
#define B_HALF   0x3F000000u   // HALF
#define B_THREE  0x40400000u   // THREE
#define B_THIRD  0x3EAAAAABu   // THIRD = f32(1/3)
#define B_KG     0x41B2E92Fu   // KG = f32(1/0.044715)
#define B_TEN    0x41200000u   // 10.0, the GELU16 / GEGLU16 clamp
#define B_INF    0x7F800000u
#define B_NINF   0xFF800000u
#define B_NAN    TSR_CANON_NAN_BITS

static inline uint32_t f2u(float f) { uint32_t u; memcpy(&u, &f, sizeof u); return u; }
static inline float u2f(uint32_t u) { float f; memcpy(&f, &u, sizeof f); return f; }

static inline uint32_t daz_bits(uint32_t u) {
    return ((u & F32_EXP) == 0) ? (u & F32_SIGN) : u;
}
static inline int is_nan_bits(uint32_t u) { return (u & F32_ABS) > F32_EXP; }
static inline float fneg(float x) { return u2f(f2u(x) ^ F32_SIGN); }

// ---------------------------------------------------------------------------
// EXPK (VU §6.3)
// ---------------------------------------------------------------------------
// LK = round(|K|·log2(e)·2^38) and sK (1 if K < 0), indexed by κ.
static const uint64_t k_expk_lk[4] = {
    0x5C551D94AEull,   // P1
    0x5C551D94AEull,   // N1
    0xB8AA3B295Cull,   // N2 = 2·LK(P1)
    0x6969F3DD6ull,    // NG
};
static const uint32_t k_expk_sk[4] = { 0u, 1u, 1u, 1u };

// round(ln2·2^32), round(ln²2/2·2^32), round(ln³2/6·2^32)
#define EXPK_P1F 0xB17217F8ull
#define EXPK_P2F 0x3D7F7BFFull
#define EXPK_P3F 0x0E35846Cull

// T64[j] = round(2^(j/64)·2^30)
static const uint32_t k_expk_t64[64] = {
    0x40000000u, 0x40B268FAu, 0x4166C34Cu, 0x421D1462u, 0x42D561B4u, 0x438FB0CBu, 0x444C0740u, 0x450A6ABBu,
    0x45CAE0F2u, 0x468D6FAEu, 0x47521CC6u, 0x4818EE22u, 0x48E1E9BAu, 0x49AD1598u, 0x4A7A77D4u, 0x4B4A169Cu,
    0x4C1BF829u, 0x4CF022CAu, 0x4DC69CDDu, 0x4E9F6CD4u, 0x4F7A9930u, 0x50582888u, 0x51382182u, 0x521A8AD7u,
    0x52FF6B55u, 0x53E6C9DAu, 0x54D0AD5Au, 0x55BD1CDBu, 0x56AC1F75u, 0x579DBC57u, 0x5891FAC1u, 0x5988E209u,
    0x5A82799Au, 0x5B7EC8F2u, 0x5C7DD7A4u, 0x5D7FAD59u, 0x5E8451D0u, 0x5F8BCCDBu, 0x60962665u, 0x61A3666Du,
    0x62B39509u, 0x63C6BA64u, 0x64DCDEC3u, 0x65F60A7Fu, 0x6712460Bu, 0x683199EDu, 0x69540EC9u, 0x6A79AD56u,
    0x6BA27E65u, 0x6CCE8AE1u, 0x6DFDDBCCu, 0x6F307A41u, 0x70666F76u, 0x719FC4B9u, 0x72DC8374u, 0x741CB528u,
    0x75606374u, 0x76A7980Fu, 0x77F25CCEu, 0x7940BB9Eu, 0x7A92BE8Bu, 0x7BE86FBAu, 0x7D41D96Eu, 0x7E9F0606u,
};

float tsr_m_expk(float v, unsigned kappa) {
    if (kappa > TSR_M_NG) {
        return u2f(B_NAN);
    }
    // 1. DAZ; NaN -> canonical NaN.
    const uint32_t u = daz_bits(f2u(v));
    if (is_nan_bits(u)) {
        return u2f(B_NAN);
    }
    // 2. Sign of K·v.
    const uint32_t sgn = (u >> 31) ^ k_expk_sk[kappa];
    const uint32_t a = u & F32_ABS;
    // 3. Infinities and zeros.
    if (a == F32_EXP) {
        return u2f(sgn ? B_ZERO : B_INF);
    }
    if (a == 0) {
        return u2f(B_ONE);
    }
    // 4. Ev in 1 .. 254, Mv = 2^23 | fraction.
    const uint32_t Ev = a >> 23;
    const uint64_t Mv = 0x00800000u | (a & F32_MANT);
    // 5. Saturation by exponent.
    if (Ev > 156u) {
        return u2f(sgn ? B_ZERO : B_INF);
    }
    // 6. T = |K·v·log2(e)| in units of 2^-32.
    const uint64_t P = Mv * k_expk_lk[kappa];          // < 2^24 · 2^40
    const uint32_t sh = 156u - Ev;                      // 0 .. 155
    const uint64_t T = (sh >= 64u) ? 0u : (P >> sh);
    if (T >= (1ull << 40)) {
        return u2f(sgn ? B_ZERO : B_INF);
    }
    // 7. t = ±T; n = floor(t / 2^32); F = t mod 2^32.
    int32_t n;
    uint32_t F;
    if (sgn) {
        n = -(int32_t)((T + 0xFFFFFFFFull) >> 32);     // floor(-T / 2^32) = -ceil(T / 2^32)
        F = 0u - (uint32_t)T;
    } else {
        n = (int32_t)(T >> 32);
        F = (uint32_t)T;
    }
    // 8. Table index and remainder.
    const uint32_t j = F >> 26;
    const uint64_t G = F & 0x03FFFFFFu;
    // 9. 2^(G·2^-32) - 1 in units of 2^-32 (Horner, truncating products).
    const uint64_t X3 = (G * EXPK_P3F) >> 32;
    const uint64_t S2 = EXPK_P2F + X3;
    const uint64_t X2 = (G * S2) >> 32;
    const uint64_t S1 = EXPK_P1F + X2;
    const uint64_t PF = (G * S1) >> 32;
    // 10. M = 2^frac in Q1.30, in [2^30, 2^31].
    const uint64_t t64 = k_expk_t64[j];
    const uint64_t M = t64 + ((t64 * PF) >> 32);
    // 11. Normalize.
    uint32_t drop = 7u;
    if (M >= (1ull << 31)) {
        drop = 8u;
        n += 1;
    }
    // 12. Round to 24 bits, nearest-even.
    uint64_t sig = M >> drop;
    const uint64_t rem = M & ((1ull << drop) - 1u);
    const uint64_t half = 1ull << (drop - 1u);
    if (rem > half || (rem == half && (sig & 1u))) {
        sig += 1u;
    }
    if (sig == (1ull << 24)) {
        sig = 1ull << 23;
        n += 1;
    }
    // 13. Pack; overflow -> +inf, underflow -> +0 (never subnormal, never -0).
    const int32_t be = n + 127;
    if (be >= 255) {
        return u2f(B_INF);
    }
    if (be <= 0) {
        return u2f(B_ZERO);
    }
    return u2f(((uint32_t)be << 23) | (uint32_t)(sig - (1ull << 23)));
}

float tsr_m_exp(float x) { return tsr_m_expk(x, TSR_M_P1); }

// ---------------------------------------------------------------------------
// Composite functions (VU §6.5)
// ---------------------------------------------------------------------------
float tsr_m_sigmoid(float x) {
    return tsr_frecip(tsr_fadd(u2f(B_ONE), tsr_m_expk(x, TSR_M_N1)));
}

float tsr_m_silu(float x) {
    return tsr_fmul(x, tsr_m_sigmoid(x));
}

// GELU(x) = x·σ(2u), 2u = C2G·x·(x² + KG); C2G is folded into LK(NG).
float tsr_m_gelu_tanh(float x) {
    float w = tsr_fmul(x, x);
    w = tsr_fadd(w, u2f(B_KG));
    const float v = tsr_fmul(w, x);
    return tsr_fmul(x, tsr_frecip(tsr_fadd(u2f(B_ONE), tsr_m_expk(v, TSR_M_NG))));
}

float tsr_m_tanh(float x) {
    const uint32_t u = f2u(x);
    if (is_nan_bits(u)) {
        return u2f(B_NAN);
    }
    const uint32_t axb = daz_bits(u) & F32_ABS;
    const float ax = u2f(axb);
    float t;
    if (axb < 0x3C800000u) {                    // ax < 2^-6: t = ax·(3 - ax²)·THIRD
        const float q = tsr_fmul(ax, ax);
        const float w = tsr_fsub(u2f(B_THREE), q);
        t = tsr_fmul(tsr_fmul(w, ax), u2f(B_THIRD));
    } else {                                    // t = (1 - e) / (1 + e), e = exp(-2·ax)
        const float e = tsr_m_expk(ax, TSR_M_N2);
        t = tsr_fmul(tsr_fsub(u2f(B_ONE), e), tsr_frecip(tsr_fadd(u2f(B_ONE), e)));
    }
    return (u & F32_SIGN) ? fneg(t) : t;
}

// binary16 helpers: the conversions of tsr_ref.c (integer-only, ggml-identical). F16 there
// already maps f32 subnormals to signed zero; the explicit DAZ states the rule of VU §6.5.
uint16_t tsr_f32_to_f16_rne(float x) { return tsr_f32_to_fp16(u2f(daz_bits(f2u(x)))); }
float    tsr_f16_to_f32(uint16_t h)  { return tsr_fp16_to_f32(h); }

static inline float h16(float v) { return tsr_f16_to_f32(tsr_f32_to_f16_rne(v)); }

// DAZ'd, non-NaN bits x' with x' <= -10 (the ggml CPU's 'x <= -10.0f').
static inline int le_minus_ten(uint32_t xd) {
    return (xd & F32_SIGN) && (xd & F32_ABS) >= B_TEN;
}

float tsr_m_gelu16(float x) {
    const uint32_t xd = daz_bits(f2u(x));
    if (is_nan_bits(xd)) {
        return u2f(B_NAN);
    }
    if (le_minus_ten(xd)) {
        return u2f(B_ZERO);
    }
    if (!(xd & F32_SIGN) && xd >= B_TEN) {       // x' >= 10 (incl. +inf): x'
        return u2f(xd);
    }
    return h16(tsr_m_gelu_tanh(h16(u2f(xd))));
}

float tsr_m_geglu16(float a, float b) {
    const uint32_t ad = daz_bits(f2u(a));
    if (!is_nan_bits(ad) && le_minus_ten(ad)) {
        return u2f(B_ZERO);
    }
    return tsr_fmul(tsr_m_gelu16(a), b);
}

float tsr_m_expsub(float a, float x) {
    if (daz_bits(f2u(a)) == B_NINF) {
        return u2f(B_ZERO);
    }
    return tsr_m_expk(tsr_fsub(a, x), TSR_M_P1);
}

// ---------------------------------------------------------------------------
// SIN and COS (VU §6.6)
// ---------------------------------------------------------------------------
// Unsigned 128-bit value as two 64-bit words (portable; no __int128 on RV32).
typedef struct { uint64_t hi, lo; } u128w;

static u128w mul_64x64(uint64_t a, uint64_t b) {
    const uint64_t a0 = a & 0xFFFFFFFFu, a1 = a >> 32;
    const uint64_t b0 = b & 0xFFFFFFFFu, b1 = b >> 32;
    const uint64_t p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
    const uint64_t mid = (p00 >> 32) + (p01 & 0xFFFFFFFFu) + (p10 & 0xFFFFFFFFu);   // < 2^34
    u128w r;
    r.lo = (mid << 32) | (p00 & 0xFFFFFFFFu);
    r.hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
    return r;
}

static u128w shr_128(u128w x, unsigned n) {
    u128w r;
    if (n >= 128u) {
        r.hi = 0; r.lo = 0;
    } else if (n >= 64u) {
        r.hi = 0; r.lo = x.hi >> (n - 64u);
    } else if (n == 0u) {
        r = x;
    } else {
        r.hi = x.hi >> n; r.lo = (x.lo >> n) | (x.hi << (64u - n));
    }
    return r;
}

static u128w shl_128(u128w x, unsigned n) {
    u128w r;
    if (n >= 128u) {
        r.hi = 0; r.lo = 0;
    } else if (n >= 64u) {
        r.lo = 0; r.hi = x.lo << (n - 64u);
    } else if (n == 0u) {
        r = x;
    } else {
        r.lo = x.lo << n; r.hi = (x.hi << n) | (x.lo >> (64u - n));
    }
    return r;
}

static int msb_index_u64(uint64_t v) {   // v != 0
#if defined(__GNUC__) || defined(__clang__)
    return 63 - __builtin_clzll(v);
#else
    int i = 0;
    while (v >>= 1) ++i;
    return i;
#endif
}

// Bits [0, n) of x contain a 1 (n <= 127).
static int low_bits_nonzero(u128w x, unsigned n) {
    if (n == 0u) {
        return 0;
    }
    if (n >= 64u) {
        return x.lo != 0 || (n > 64u && (x.hi & ((1ull << (n - 64u)) - 1u)) != 0);
    }
    return (x.lo & ((1ull << n) - 1u)) != 0;
}

// The f32 nearest-even to R·2^-124, for 0 < R < 2^124, with sign 'neg' (always normal here).
static float r_to_f32(u128w R, int neg) {
    const int b = R.hi ? 64 + msb_index_u64(R.hi) : msb_index_u64(R.lo);   // R in [2^b, 2^(b+1))
    const unsigned shift = (unsigned)(b - 23);                              // b >= 62
    uint64_t sig = shr_128(R, shift).lo;                                    // 24 bits
    const int guard = (int)(shr_128(R, shift - 1u).lo & 1u);
    const int sticky = low_bits_nonzero(R, shift - 1u);
    uint32_t be = (uint32_t)(b + 3);                                        // b - 124 + 127
    if (guard && (sticky || (sig & 1u))) {
        sig += 1u;
        if (sig == (1ull << 24)) {
            sig >>= 1;
            be += 1u;
        }
    }
    return u2f((neg ? F32_SIGN : 0u) | (be << 23) | ((uint32_t)sig & F32_MANT));
}

#define SC_C64  0xA2F9836E4E44152Aull   // round(2/pi·2^64)
#define SC_PIO2 0x6487ED5110B4611Aull   // round(pi/2·2^62)
#define SC_SN1  0xBE2AAAA3u
#define SC_SN2  0x3C08839Eu
#define SC_SN3  0xB94CA1F9u
#define SC_CS1  0x3D2AAAA5u
#define SC_CS2  0xBAB6061Au
#define SC_CS3  0x37CCF5CEu
#define MASK62  ((1ull << 62) - 1u)

void tsr_m_sincos(float t, float * s_out, float * c_out) {
    const uint32_t u = daz_bits(f2u(t));
    const uint32_t a = u & F32_ABS;
    if (a >= F32_EXP) {                          // NaN or ±inf
        *s_out = u2f(B_NAN);
        *c_out = u2f(B_NAN);
        return;
    }
    const uint32_t E = a >> 23;
    // 1.-2. Zero and |θ| < 2^-12: SIN = θ, COS = ONE.
    if (E < 115u) {
        *s_out = u2f(u);
        *c_out = u2f(B_ONE);
        return;
    }
    const uint64_t M = 0x00800000u | (a & F32_MANT);

    // 3. Quadrant reduction: θ·2/π ≈ P·2^-sh, P = M·C64 (88 bits).
    const u128w P = mul_64x64(M, SC_C64);
    const int sh = 214 - (int)E;                 // -40 .. 99
    uint32_t k4;                                 // k mod 4
    uint64_t F;                                  // fraction in units of 2^-62, [0, 2^62)
    if (sh >= 62) {
        k4 = (uint32_t)(shr_128(P, (unsigned)sh).lo & 3u);
        F = shr_128(P, (unsigned)(sh - 62)).lo & MASK62;
    } else if (sh >= 0) {
        k4 = (uint32_t)(shr_128(P, (unsigned)sh).lo & 3u);
        F = shl_128(P, (unsigned)(62 - sh)).lo & MASK62;
    } else {
        k4 = (uint32_t)(shl_128(P, (unsigned)(-sh)).lo & 3u);
        F = 0;
    }
    // 4. Round to the nearest quadrant: F in [-2^61, 2^61) (sign and magnitude here).
    int rneg = 0;
    uint64_t Fmag = F;
    if (F >= (1ull << 61)) {
        k4 = (k4 + 1u) & 3u;
        rneg = 1;
        Fmag = (1ull << 62) - F;                 // |F - 2^62|, in (0, 2^61]
    }
    const uint32_t q = k4;
    // 5. r = f32 nearest-even to F·PIO2·2^-124 (+0 if F = 0).
    float r;
    if (Fmag == 0) {
        r = u2f(B_ZERO);
    } else {
        r = r_to_f32(mul_64x64(Fmag, SC_PIO2), rneg);
    }

    // 6. Polynomials (Cephes sinf / cosf).
    const float z = tsr_fmul(r, r);
    float p = tsr_fmul(u2f(SC_SN3), z);
    p = tsr_fadd(p, u2f(SC_SN2));
    p = tsr_fmul(p, z);
    p = tsr_fadd(p, u2f(SC_SN1));
    p = tsr_fmul(p, z);
    p = tsr_fmul(p, r);
    const float sn = tsr_fadd(p, r);

    p = tsr_fmul(u2f(SC_CS3), z);
    p = tsr_fadd(p, u2f(SC_CS2));
    p = tsr_fmul(p, z);
    p = tsr_fadd(p, u2f(SC_CS1));
    p = tsr_fmul(p, z);
    p = tsr_fmul(p, z);
    const float h = tsr_fmul(u2f(B_HALF), z);
    const float uu = tsr_fsub(p, h);
    const float cs = tsr_fadd(uu, u2f(B_ONE));

    // 7. Quadrant.
    float S, C;
    switch (q) {
    case 0:  S = sn;       C = cs;       break;
    case 1:  S = cs;       C = fneg(sn); break;
    case 2:  S = fneg(sn); C = fneg(cs); break;
    default: S = fneg(cs); C = sn;       break;
    }
    // 8. Sign of θ.
    *s_out = (u & F32_SIGN) ? fneg(S) : S;
    *c_out = C;
}

float tsr_m_sin(float t) {
    float s, c;
    tsr_m_sincos(t, &s, &c);
    return s;
}

float tsr_m_cos(float t) {
    float s, c;
    tsr_m_sincos(t, &s, &c);
    return c;
}
