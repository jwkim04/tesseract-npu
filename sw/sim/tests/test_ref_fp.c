// test_ref_fp.c — f32 helper edge cases (spec "Common fp rules" + §21 "FTZ decision point").
// DAZ/FTZ/signed zero/Inf/NaN canonicalisation, int32 -> f32 rounding (exhaustive against
// the double path), fp16 <-> f32 corner values, QACT rounding helpers.

#include <float.h>

#include "test_util.h"
#include "tsr_ref.h"

#define CANON 0x7FC00000u

static void eqbits(const char * what, float got, uint32_t want, int line) {
    ++g_checks;
    if (f2u(got) != want) {
        fprintf(stderr, "FAIL line %d: %s: got 0x%08x want 0x%08x\n", line, what, f2u(got), want);
        ++g_fail;
    }
}
#define EQ(what, got, want) eqbits(what, got, want, __LINE__)

static void test_daz(void) {
    EQ("daz +min sub", tsr_f32_daz(u2f(0x00000001)), 0x00000000);
    EQ("daz +max sub", tsr_f32_daz(u2f(0x007FFFFF)), 0x00000000);
    EQ("daz -min sub", tsr_f32_daz(u2f(0x80000001)), 0x80000000);
    EQ("daz -max sub", tsr_f32_daz(u2f(0x807FFFFF)), 0x80000000);
    EQ("daz +min normal", tsr_f32_daz(u2f(0x00800000)), 0x00800000);
    EQ("daz -0", tsr_f32_daz(u2f(0x80000000)), 0x80000000);
    EQ("daz inf", tsr_f32_daz(u2f(0x7F800000)), 0x7F800000);
    EQ("daz nan untouched", tsr_f32_daz(u2f(0x7F800123)), 0x7F800123);
}

static void test_arith(void) {
    const float one = 1.0f, two = 2.0f;
    const float sub = u2f(0x00000001), nsub = u2f(0x80000001), minn = u2f(0x00800000);
    const float inf = u2f(0x7F800000), ninf = u2f(0xFF800000);

    // DAZ on inputs: subnormals act as signed zero.
    EQ("sub + 0", tsr_fadd(sub, 0.0f), 0x00000000);
    EQ("sub + sub", tsr_fadd(sub, sub), 0x00000000);
    EQ("-sub + -0", tsr_fadd(nsub, u2f(0x80000000)), 0x80000000);   // -0 + -0 = -0
    EQ("-sub + +0", tsr_fadd(nsub, 0.0f), 0x00000000);              // -0 + +0 = +0
    EQ("1 + sub", tsr_fadd(one, sub), 0x3F800000);
    EQ("sub * huge", tsr_fmul(sub, u2f(0x7F000000)), 0x00000000);    // would be 2^-22 without DAZ
    EQ("-sub * huge", tsr_fmul(nsub, u2f(0x7F000000)), 0x80000000);
    EQ("1 / sub", tsr_frecip(sub), 0x7F800000);                      // 1/+0
    EQ("1 / -sub", tsr_frecip(nsub), 0xFF800000);                    // 1/-0
    EQ("sqrt(sub)", tsr_fsqrt(sub), 0x00000000);
    EQ("sqrt(-sub)", tsr_fsqrt(nsub), 0x80000000);                   // sqrt(-0) = -0

    // FTZ on results: flush iff the IEEE RNE result *with the binary32 exponent range*
    // (gradual underflow) is subnormal, i.e. iff |exact| < 2^-126 - 2^-150. This is neither
    // IEEE 754 "tininess after rounding" (unbounded exponent; x86 MXCSR.FTZ) nor "before
    // rounding": both of those flush part of [2^-126 - 2^-150, 2^-126), which is kept here.
    EQ("1.5m - m", tsr_fsub(u2f(0x00C00000), minn), 0x00000000);    // 0.5*2^-126 -> +0
    EQ("-1.5m + m", tsr_fadd(u2f(0x80C00000), minn), 0x80000000);   // -> -0
    EQ("2^-64*2^-63", tsr_fmul(u2f(0x1F800000), u2f(0x20000000)), 0x00000000);
    EQ("-2^-64*2^-63", tsr_fmul(u2f(0x9F800000), u2f(0x20000000)), 0x80000000);
    EQ("2^-63*2^-63", tsr_fmul(u2f(0x20000000), u2f(0x20000000)), 0x00800000);  // exactly 2^-126
    // (1 - 2^-24) * 2^-126 = 2^-126 - 2^-150: the bounded-exponent RNE result is the tie
    // between 0x007FFFFF and 0x00800000 -> even = 2^-126 (normal) -> kept. Both a
    // tininess-before-rounding FTZ and an IEEE tininess-after-rounding FTZ (unbounded
    // exponent: the 24-bit rounding is exact and < 2^-126; x86 SSE with MXCSR.FTZ) give 0.
    EQ("edge round up to min normal", tsr_fmul(u2f(0x3F7FFFFF), minn), 0x00800000);
    EQ("edge -", tsr_fmul(u2f(0xBF7FFFFF), minn), 0x80800000);
    // Interior of the window: exact = 2^-126 - 2^-150 + 0.02*2^-150 -> 2^-126 (x86 FTZ: 0).
    EQ("window interior", tsr_fmul(u2f(0x3F1FA0BA), u2f(0x00CD4708)), 0x00800000);
    // Division: (2^-125 - 2^-149) / 2 = 2^-126 - 2^-150 -> 2^-126; one ulp less -> flushed.
    EQ("div window edge", tsr_fdiv(u2f(0x00FFFFFF), two), 0x00800000);
    EQ("div below window", tsr_fdiv(u2f(0x00FFFFFE), two), 0x00000000);
    // (1 - 2^-23) * 2^-126 = 2^-126 - 2^-149 is exactly subnormal -> flushed.
    EQ("edge subnormal", tsr_fmul(u2f(0x3F7FFFFE), minn), 0x00000000);
    EQ("edge subnormal -", tsr_fmul(u2f(0xBF7FFFFE), minn), 0x80000000);
    EQ("recip 2^127", tsr_frecip(u2f(0x7F000000)), 0x00000000);      // 2^-127 -> +0
    EQ("recip -2^127", tsr_frecip(u2f(0xFF000000)), 0x80000000);
    EQ("recip 2^126", tsr_frecip(u2f(0x7E800000)), 0x00800000);
    EQ("div tiny", tsr_fdiv(minn, two), 0x00000000);

    // Signed zero (IEEE RNE rules).
    EQ("1 - 1", tsr_fsub(one, one), 0x00000000);
    EQ("-0 + -0", tsr_fadd(u2f(0x80000000), u2f(0x80000000)), 0x80000000);
    EQ("+0 * -1", tsr_fmul(0.0f, -1.0f), 0x80000000);
    EQ("-0 - +0", tsr_fsub(u2f(0x80000000), 0.0f), 0x80000000);

    // Correct rounding.
    EQ("1/3", tsr_fdiv(1.0f, 3.0f), 0x3EAAAAAB);
    EQ("recip 3", tsr_frecip(3.0f), 0x3EAAAAAB);
    EQ("sqrt 2", tsr_fsqrt(2.0f), 0x3FB504F3);
    EQ("1 + 2^-24 (tie, even)", tsr_fadd(one, u2f(0x33800000)), 0x3F800000);
    EQ("1+2^-23 + 2^-24 (tie, up)", tsr_fadd(u2f(0x3F800001), u2f(0x33800000)), 0x3F800002);

    // Inf propagates.
    EQ("inf + 1", tsr_fadd(inf, one), 0x7F800000);
    EQ("-inf * 2", tsr_fmul(ninf, two), 0xFF800000);
    EQ("max + max", tsr_fadd(FLT_MAX, FLT_MAX), 0x7F800000);
    EQ("max * 2", tsr_fmul(FLT_MAX, two), 0x7F800000);
    EQ("-max * 2", tsr_fmul(-FLT_MAX, two), 0xFF800000);
    EQ("1/inf", tsr_frecip(inf), 0x00000000);
    EQ("1/-inf", tsr_frecip(ninf), 0x80000000);
    EQ("sqrt inf", tsr_fsqrt(inf), 0x7F800000);
    EQ("1/0", tsr_fdiv(one, 0.0f), 0x7F800000);
    EQ("-1/0", tsr_fdiv(-1.0f, 0.0f), 0xFF800000);

    // NaN -> canonical 0x7FC00000 (x86 would produce 0xFFC00000 or keep payloads).
    EQ("inf - inf", tsr_fadd(inf, ninf), CANON);
    EQ("inf - inf sub", tsr_fsub(inf, inf), CANON);
    EQ("0 * inf", tsr_fmul(0.0f, inf), CANON);
    EQ("sub * inf", tsr_fmul(sub, inf), CANON);       // DAZ makes it 0 * inf
    EQ("0/0", tsr_fdiv(0.0f, 0.0f), CANON);
    EQ("inf/inf", tsr_fdiv(inf, ninf), CANON);
    EQ("sqrt -1", tsr_fsqrt(-1.0f), CANON);
    EQ("sqrt -inf", tsr_fsqrt(ninf), CANON);
    EQ("snan + 1", tsr_fadd(u2f(0x7F800001), one), CANON);
    EQ("-qnan payload * 2", tsr_fmul(u2f(0xFFC12345), two), CANON);
    EQ("1 - nan", tsr_fsub(one, u2f(0x7FC00001)), CANON);
    EQ("recip nan", tsr_frecip(u2f(0xFFFFFFFF)), CANON);
    EQ("sqrt nan", tsr_fsqrt(u2f(0x7FBFFFFF)), CANON);

    // Tree: explicit parenthesisation and special values.
    float p[16];
    for (int i = 0; i < 16; ++i) p[i] = 0.0f;
    p[0] = 1.0f; p[1] = u2f(0x33800000); p[2] = u2f(0x33800000);   // (1 + 2^-24) + (2^-24 + 0)
    // ((p0+p1)+(p2+p3)): p0+p1 = 1 (tie even), p2+p3 = 2^-24, 1 + 2^-24 = 1 -> 0x3F800000
    EQ("tree order", tsr_tree16(p), 0x3F800000);
    p[0] = 1.0f; p[1] = 0.0f; p[2] = u2f(0x33800000); p[3] = u2f(0x33800000);
    // p0+p1 = 1, p2+p3 = 2^-23, 1 + 2^-23 exact -> 0x3F800001 (a left-to-right sum gives 1.0)
    EQ("tree order 2", tsr_tree16(p), 0x3F800001);
    for (int i = 0; i < 16; ++i) p[i] = u2f(0x80000000);
    EQ("tree all -0", tsr_tree16(p), 0x80000000);
    p[7] = 0.0f;
    EQ("tree one +0", tsr_tree16(p), 0x00000000);
    for (int i = 0; i < 16; ++i) p[i] = 1.0f;
    p[3] = inf; p[12] = ninf;
    EQ("tree inf-inf", tsr_tree16(p), CANON);
    p[12] = 1.0f;
    EQ("tree inf", tsr_tree16(p), 0x7F800000);
}

// FTZ rule (spec §21 (FTZ decision point)), checked against its documented equivalent form
// on products (exact in double): |exact| < 2^-126 - 2^-150 -> signed zero, else RNE(exact).
// The operand pairs are chosen around lo/b so that many exact products fall into
// [2^-126 - 2^-150, 2^-126 - 2^-151), where an IEEE tininess-after-rounding FTZ (x86
// MXCSR.FTZ) flushes but the spec keeps 2^-126; the test requires that window to be hit.
static void test_ftz_window(void) {
    const double lo = ldexp(1.0, -126) - ldexp(1.0, -150);
    const double hi_after = ldexp(1.0, -126) - ldexp(1.0, -151);
    rng_t r = { 0xF72Bull };
    uint64_t window = 0, flushed = 0, bad = 0;
    for (int it = 0; it < (1 << 20); ++it) {
        const uint32_t ub = (1u + rng_below(&r, 2)) << 23 | (rng_u32(&r) & 0x007FFFFFu);  // [2^-126, 2^-124)
        const uint32_t ua0 = f2u((float)(lo / (double)u2f(ub)));
        for (int d = -2; d <= 2; ++d) {
            const uint32_t sa = rng_u32(&r) & 0x80000000u, sb = rng_u32(&r) & 0x80000000u;
            const float a = u2f((ua0 + (uint32_t)d) | sa), b = u2f(ub | sb);
            const double ex = (double)a * (double)b;
            const uint32_t want = fabs(ex) < lo ? ((sa ^ sb) & 0x80000000u) : f2u((float)ex);
            const uint32_t got = f2u(tsr_fmul(a, b));
            if (got != want) {
                if (bad < 5) fprintf(stderr, "ftz window: 0x%08x * 0x%08x = %a: got 0x%08x want 0x%08x\n",
                                     f2u(a), f2u(b), ex, got, want);
                ++bad;
            }
            window += fabs(ex) >= lo && fabs(ex) < hi_after;
            flushed += fabs(ex) < lo;
        }
    }
    ++g_checks;
    if (bad) { fprintf(stderr, "FAIL: %llu FTZ-rule mismatches\n", (unsigned long long)bad); ++g_fail; }
    CHECK(window > 1000 && flushed > 1000, "FTZ window not exercised: %llu in window, %llu flushed",
          (unsigned long long)window, (unsigned long long)flushed);
}

static void test_i32_to_f32(void) {
    EQ("0", tsr_i32_to_f32(0), 0x00000000);
    EQ("1", tsr_i32_to_f32(1), 0x3F800000);
    EQ("-1", tsr_i32_to_f32(-1), 0xBF800000);
    EQ("2^24", tsr_i32_to_f32(16777216), 0x4B800000);
    EQ("2^24+1 tie->even down", tsr_i32_to_f32(16777217), 0x4B800000);
    EQ("2^24+2", tsr_i32_to_f32(16777218), 0x4B800001);
    EQ("2^24+3 tie->even up", tsr_i32_to_f32(16777219), 0x4B800002);
    EQ("-(2^24+1)", tsr_i32_to_f32(-16777217), 0xCB800000);
    EQ("2^25+3 (rem 3/4 up)", tsr_i32_to_f32(33554435), 0x4C000001);
    EQ("2^25+2 tie even down", tsr_i32_to_f32(33554434), 0x4C000000);
    EQ("2^25+6 tie even up", tsr_i32_to_f32(33554438), 0x4C000002);
    EQ("Q4_K isum max-ish 31M", tsr_i32_to_f32(30965760), 0x4BEC4000);
    EQ("INT32_MAX", tsr_i32_to_f32(INT32_MAX), 0x4F000000);
    EQ("INT32_MIN", tsr_i32_to_f32(INT32_MIN), 0xCF000000);
    EQ("2^31-65", tsr_i32_to_f32(2147483583), 0x4EFFFFFF);  // below the midpoint
    EQ("2^31-64 tie", tsr_i32_to_f32(2147483584), 0x4F000000);

    // Exhaustive over all 2^32 int32 values against the double path
    // (int32 -> double is exact, double -> float is one RNE rounding).
    uint64_t bad = 0;
    for (uint64_t k = 0; k <= 0xFFFFFFFFull; ++k) {
        const int32_t v = (int32_t)(uint32_t)k;
        const float want = (float)(double)v;
        if (f2u(tsr_i32_to_f32(v)) != f2u(want)) {
            if (bad < 5) fprintf(stderr, "i32->f32 mismatch at %d\n", v);
            ++bad;
        }
    }
    ++g_checks;
    if (bad) { fprintf(stderr, "FAIL: %llu i32->f32 mismatches\n", (unsigned long long)bad); ++g_fail; }
}

static void test_fp16(void) {
    EQ("h 0x0001", tsr_fp16_to_f32(0x0001), 0x33800000);            // 2^-24
    EQ("h 0x8001", tsr_fp16_to_f32(0x8001), 0xB3800000);
    EQ("h 0x03FF", tsr_fp16_to_f32(0x03FF), 0x387FC000);            // 1023 * 2^-24
    EQ("h 0x0200", tsr_fp16_to_f32(0x0200), 0x38000000);            // 2^-15
    EQ("h 0x0400", tsr_fp16_to_f32(0x0400), 0x38800000);            // 2^-14
    EQ("h 0x3C00", tsr_fp16_to_f32(0x3C00), 0x3F800000);
    EQ("h 0x7BFF", tsr_fp16_to_f32(0x7BFF), 0x477FE000);            // 65504
    EQ("h +0", tsr_fp16_to_f32(0x0000), 0x00000000);
    EQ("h -0", tsr_fp16_to_f32(0x8000), 0x80000000);
    EQ("h inf", tsr_fp16_to_f32(0x7C00), 0x7F800000);
    EQ("h -inf", tsr_fp16_to_f32(0xFC00), 0xFF800000);
    EQ("h qnan", tsr_fp16_to_f32(0x7E00), 0x7FC00000);
    EQ("h snan quieted", tsr_fp16_to_f32(0x7C01), 0x7FC02000);
    EQ("h -nan payload", tsr_fp16_to_f32(0xFE01), 0xFFC02000);

    struct { uint32_t f; uint16_t h; const char * what; } c[] = {
        { 0x3F800000, 0x3C00, "1" },
        { 0x477FE000, 0x7BFF, "65504" },
        { 0x477FEFFF, 0x7BFF, "65519.99" },
        { 0x477FF000, 0x7C00, "65520 tie -> even = inf" },
        { 0x47800000, 0x7C00, "65536" },
        { 0x7F7FFFFF, 0x7C00, "FLT_MAX" },
        { 0xFF7FFFFF, 0xFC00, "-FLT_MAX" },
        { 0x33800000, 0x0001, "2^-24" },
        { 0x33000000, 0x0000, "2^-25 tie -> 0" },
        { 0x33000001, 0x0001, "2^-25(1+2^-23) -> 2^-24" },
        { 0xB3000001, 0x8001, "-2^-25(1+2^-23)" },
        { 0x33C00000, 0x0002, "1.5*2^-24 tie -> 2" },
        { 0x34200000, 0x0002, "2.5*2^-24 tie -> 2" },
        { 0x387FE000, 0x0400, "0x3FF.8 * 2^-24 tie -> min normal" },
        { 0x00000001, 0x0000, "f32 subnormal" },
        { 0x80400000, 0x8000, "-f32 subnormal" },
        { 0x80000000, 0x8000, "-0" },
        { 0x3F801000, 0x3C00, "1+2^-11 tie -> even" },
        { 0x3F803000, 0x3C02, "1+3*2^-11 tie -> even up" },
        { 0x3F801001, 0x3C01, "just above tie" },
        { 0x7F800000, 0x7C00, "inf" },
        { 0x7FC00000, 0x7E00, "nan" },
        { 0x7F800001, 0x7E00, "snan" },
        { 0xFFC12345, 0xFE00, "-nan" },
    };
    for (size_t i = 0; i < sizeof c / sizeof c[0]; ++i) {
        const uint16_t h = tsr_f32_to_fp16(u2f(c[i].f));
        CHECK(h == c[i].h, "f32->fp16 %s: got 0x%04x want 0x%04x", c[i].what, h, c[i].h);
    }
    // fp16 -> f32 -> fp16 is the identity on all non-NaN halves.
    for (uint32_t h = 0; h < 65536; ++h) {
        if ((h & 0x7C00) == 0x7C00 && (h & 0x3FF)) continue;
        CHECK(tsr_f32_to_fp16(tsr_fp16_to_f32((uint16_t)h)) == h, "h round trip 0x%04x", h);
    }
}

static void test_round8(void) {
    struct { float x; int away, even; } c[] = {
        { 0.0f, 0, 0 }, { -0.0f, 0, 0 }, { 0.49999997f, 0, 0 }, { 0.5f, 1, 0 }, { -0.5f, -1, 0 },
        { 1.5f, 2, 2 }, { 2.5f, 3, 2 }, { -2.5f, -3, -2 }, { -1.5f, -2, -2 }, { 3.5f, 4, 4 },
        { 126.5f, 127, 126 }, { -126.5f, -127, -126 }, { 127.0f, 127, 127 }, { 127.5f, 127, 127 },
        { 127.49998f, 127, 127 }, { -127.5f, -128, -128 }, { -128.0f, -128, -128 },
        { -128.5f, -128, -128 }, { 128.0f, 127, 127 }, { 1e10f, 127, 127 }, { -1e10f, -128, -128 },
        { 0.50000006f, 1, 1 }, { 2.4999998f, 2, 2 }, { 255.5f, 127, 127 }, { 300.0f, 127, 127 },
    };
    for (size_t i = 0; i < sizeof c / sizeof c[0]; ++i) {
        CHECK(tsr_q8_round_away(c[i].x) == c[i].away, "away(%.9g) = %d want %d", (double)c[i].x, tsr_q8_round_away(c[i].x), c[i].away);
        CHECK(tsr_q8_round_even(c[i].x) == c[i].even, "even(%.9g) = %d want %d", (double)c[i].x, tsr_q8_round_even(c[i].x), c[i].even);
    }
    CHECK(tsr_q8_round_away(u2f(0x7F800000)) == 127 && tsr_q8_round_even(u2f(0xFF800000)) == -128, "inf sat");
    CHECK(tsr_q8_round_away(u2f(0x7FC00000)) == 0 && tsr_q8_round_even(u2f(0xFFC00001)) == 0, "nan -> 0");
    CHECK(tsr_q8_round_away(u2f(0x00000001)) == 0 && tsr_q8_round_even(u2f(0x80000001)) == 0, "subnormal");
    // Against libm on a dense sweep of [-130, 130] (roundf / nearbyintf in default RNE mode).
    for (int32_t k = -130 * 4096; k <= 130 * 4096; ++k) {
        const float x = (float)k / 4096.0f;
        long ra = lroundf(x), re = lrintf(x);
        ra = ra < -128 ? -128 : ra > 127 ? 127 : ra;
        re = re < -128 ? -128 : re > 127 ? 127 : re;
        if (tsr_q8_round_away(x) != ra || tsr_q8_round_even(x) != re) {
            CHECK(0, "round sweep x=%.9g", (double)x);
        }
    }
}

int main(void) {
    test_daz();
    test_arith();
    test_ftz_window();
    test_fp16();
    test_round8();
    test_i32_to_f32();
    return test_finish("test_ref_fp");
}
