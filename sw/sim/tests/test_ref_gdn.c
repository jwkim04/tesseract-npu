// test_ref_gdn.c — directed and cross checks of the GDN reference (sw/sim/tsr_ref_gdn.c; VU §7.12,
// VU-28 … VU-33). The independent numpy model (tests/ref_gdn_np.py, ctest ref_gdn_np) is the
// bit-level oracle on generated heads; this test holds the directed cases and the checks against
// exact and float64 arithmetic.
//
// Checks (one function each):
//   fx2f      the fixed-point to f32 conversion against exact long-double arithmetic rounded by the
//             hardware conversion (RNE, gradual underflow) and flushed as FM flushes: random
//             magnitudes of every length and exponents over the whole range, the ties, the carry
//             into the next binade, overflow, the 2^-126 - 2^-150 edge, mag = 0 (+0)
//   table     the head table on directed vectors (frame, round half up, shifts of 24 / 25 and more,
//             zero and subnormal elements, signs)
//   rowdot    integer dot products below 2^24 are exact; a single element; zero rows and zero
//             tables give +0; rows of mixed exponents against long double (error bound relative to
//             sum |s_j t_j|); Inf / NaN anywhere gives the canonical NaN
//   scalars   e, o, upd_elem against the primitive formulas
//   head      random heads against the float64 recurrence (S' and o); alpha = 0; beta = 0; zero
//             state; in place equals out of place; the o tail; rows = 0; k with a NaN poisons the
//             head, a state row with an Inf poisons only that row's e / S' / o

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "test_util.h"
#include "tsr_ref.h"
#include "tsr_ref_gdn.h"

#define CANON 0x7FC00000u
#define PINF  0x7F800000u
#define NINF  0xFF800000u
#define ONE   0x3F800000u

static uint64_t g_rng = 0x9E3779B97F4A7C15ull;
static uint64_t rnd64(void) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    return g_rng;
}
static uint32_t rnd32(void) { return (uint32_t)(rnd64() >> 16); }
static double rndu(void) { return (double)(rnd64() >> 11) / 9007199254740992.0; }

static void eqw(const char * what, uint32_t got, uint32_t want, int line) {
    ++g_checks;
    if (got != want) {
        fprintf(stderr, "FAIL line %d: %s: got %08x want %08x\n", line, what, got, want);
        if (++g_fail >= 40) {
            exit(1);
        }
    }
}
#define EQW(what, got, want) eqw(what, got, want, __LINE__)

// ---------------------------------------------------------------------------
// fx2f
// ---------------------------------------------------------------------------
// The exact rational mag * 2^x as a long double (exact for mag < 2^64 while the long double range
// holds), converted with one IEEE rounding, then flushed like FM.
static uint32_t fx2f_model(uint32_t neg, uint64_t mag, int32_t x) {
    if (mag == 0) {
        return 0;
    }
    long double v = ldexpl((long double)mag, x);
    uint32_t r;
    if (v >= ldexpl(1.0L, 128)) {
        r = PINF;
    } else {
        volatile float f = (float)v;                    // one rounding of an exact value: RNE
        r = f2u(f);
        r = f2u(tsr_fmul(u2f(r), 1.0f));                // FTZ of a subnormal result
    }
    return r | (neg ? 0x80000000u : 0u);
}

static void test_fx2f(void) {
    // directed
    EQW("one", tsr_ref_gdn_fx2f(0, 1, 0), ONE);
    EQW("-one", tsr_ref_gdn_fx2f(1, 1, 0), 0xBF800000u);
    EQW("zero is +0", tsr_ref_gdn_fx2f(1, 0, 5), 0u);
    EQW("two^24+1 ties to even", tsr_ref_gdn_fx2f(0, (1ull << 24) + 1, 0), f2u(16777216.0f));
    EQW("two^24+3 rounds up", tsr_ref_gdn_fx2f(0, (1ull << 24) + 3, 0), f2u(16777220.0f));
    EQW("carry into the next binade", tsr_ref_gdn_fx2f(0, (1ull << 25) - 1, 0), f2u(33554432.0f));
    EQW("overflow", tsr_ref_gdn_fx2f(0, 1, 128), PINF);
    EQW("overflow neg", tsr_ref_gdn_fx2f(1, 3, 127), NINF);
    EQW("max finite", tsr_ref_gdn_fx2f(0, 0xFFFFFFull, 104), 0x7F7FFFFFu);
    EQW("rounds to Inf", tsr_ref_gdn_fx2f(0, 0x1FFFFFFull, 103), PINF);   // 2^128 - 2^103: tie, even -> Inf
    EQW("min normal", tsr_ref_gdn_fx2f(0, 1, -126), 0x00800000u);
    EQW("below min normal flushes", tsr_ref_gdn_fx2f(0, 1, -127), 0u);
    EQW("below min normal flushes neg", tsr_ref_gdn_fx2f(1, 1, -127), 0x80000000u);
    // 2^-126 - 2^-150 = 0xFFFFFF * 2^-150: the FTZ edge, a tie between 0x7FFFFF * 2^-149 and 2^-126
    // that rounds to even, 2^-126
    EQW("ftz edge", tsr_ref_gdn_fx2f(0, 0xFFFFFFull, -150), 0x00800000u);
    EQW("below the ftz edge", tsr_ref_gdn_fx2f(0, 0xFFFFFEull, -150), 0u);
    EQW("above the ftz edge", tsr_ref_gdn_fx2f(0, 0x1FFFFFFull, -151), 0x00800000u);
    EQW("below, by a tie of the other kind", tsr_ref_gdn_fx2f(0, 0xFFFFFDull, -150), 0u);
    EQW("ftz edge, neg", tsr_ref_gdn_fx2f(1, 0xFFFFFFull, -150), 0x80800000u);
    // random against the model
    for (int it = 0; it < 4000000; ++it) {
        const uint32_t len = 1u + (rnd32() % 59u);
        uint64_t mag = rnd64();
        mag = len == 64 ? mag : (mag & ((1ull << len) - 1ull));
        mag |= 1ull << (len - 1u);
        if ((it & 3) == 0) {
            mag = (mag & ~0xFFFFull) | (rnd32() & 3u ? 0x8000u : 0u);    // tie-heavy
        }
        int32_t x;
        switch (it % 5) {
        case 0:  x = (int32_t)(rnd32() % 460u) - 330;  break;           // all exponents
        case 1:  x = -126 - (int32_t)len + 1 + (int32_t)(rnd32() % 5u) - 2; break;   // around min normal
        case 2:  x = -150 - (int32_t)len + (int32_t)(rnd32() % 8u);    break;        // around the FTZ edge
        case 3:  x = 128 - (int32_t)len + (int32_t)(rnd32() % 5u) - 2; break;        // around overflow
        default: x = (int32_t)(rnd32() % 100u) - 70; break;
        }
        const uint32_t neg = rnd32() & 1u;
        const uint32_t got = tsr_ref_gdn_fx2f(neg, mag, x);
        const uint32_t want = fx2f_model(neg, mag, x);
        if (got != want) {
            ++g_checks;
            fprintf(stderr, "FAIL fx2f mag=%llx x=%d neg=%u: got %08x want %08x\n", (unsigned long long)mag, x,
                    neg, got, want);
            if (++g_fail >= 40) {
                exit(1);
            }
        } else {
            ++g_checks;
        }
    }
}

// ---------------------------------------------------------------------------
// table and rowdot
// ---------------------------------------------------------------------------
static void test_table(void) {
    uint32_t x[16] = {0};
    tsr_gdn_table t;
    // 1.0, 0.5, -0.25, 2^-23 (sh 23), 2^-24 (sh 24: rounds to 1), 2^-25 (sh 25: 0), subnormal, 0
    x[0] = ONE; x[1] = 0x3F000000u; x[2] = 0xBE800000u; x[3] = f2u(ldexpf(1.0f, -23));
    x[4] = f2u(ldexpf(1.0f, -24)); x[5] = f2u(ldexpf(1.0f, -25)); x[6] = 0x00400000u; x[7] = 0x80000000u;
    tsr_ref_gdn_table(&t, x, 16);
    CHECK(t.exp == 127 && t.bad == 0, "exp %u bad %u", t.exp, t.bad);
    CHECK(t.v[0] == 0x800000, "v0 %x", t.v[0]);
    CHECK(t.v[1] == 0x400000, "v1 %x", t.v[1]);
    CHECK(t.v[2] == -0x200000, "v2 %d", t.v[2]);
    CHECK(t.v[3] == 1, "v3 %d", t.v[3]);
    CHECK(t.v[4] == 1, "v4 %d (2^-24 rounds half up to 1)", t.v[4]);
    CHECK(t.v[5] == 0, "v5 %d", t.v[5]);
    CHECK(t.v[6] == 0 && t.v[7] == 0 && t.v[15] == 0, "zero and subnormal elements");
    // round half up: 1.0 and 1 + 2^-23 scaled by 1/2 (sh 1): M = 0x800001 -> (0x800001 + 1) >> 1 = 0x400001
    x[0] = f2u(2.0f); x[1] = f2u(1.0f + ldexpf(1.0f, -23)); x[2] = f2u(1.0f);
    tsr_ref_gdn_table(&t, x, 16);
    CHECK(t.exp == 128 && t.v[0] == 0x800000 && t.v[1] == 0x400001 && t.v[2] == 0x400000, "half up: %x %x %x",
          t.v[0], t.v[1], t.v[2]);
    // all zero
    memset(x, 0, sizeof x);
    tsr_ref_gdn_table(&t, x, 16);
    CHECK(t.exp == 0 && t.bad == 0, "zero table");
    // NaN and Inf set bad
    x[3] = PINF;
    tsr_ref_gdn_table(&t, x, 16);
    CHECK(t.bad == 1, "inf is bad");
    x[3] = CANON;
    tsr_ref_gdn_table(&t, x, 16);
    CHECK(t.bad == 1, "nan is bad");
    // entries stay below 2^24 for random vectors
    for (int it = 0; it < 2000; ++it) {
        uint32_t y[256];
        for (int j = 0; j < 256; ++j) {
            y[j] = (rnd32() & 0x807FFFFFu) | ((uint32_t)(100 + rnd32() % 40u) << 23);
        }
        tsr_gdn_table tt;
        tsr_ref_gdn_table(&tt, y, 256);
        for (int j = 0; j < 256; ++j) {
            CHECK(tt.v[j] > -(1 << 24) && tt.v[j] < (1 << 24), "entry range");
        }
    }
}

static void test_rowdot(void) {
    tsr_gdn_table t;
    uint32_t row[256], tab[256];
    // integer rows and tables: exact while the result is below 2^24
    for (int it = 0; it < 20000; ++it) {
        const uint32_t m = 16u * (1u + rnd32() % 16u);
        int64_t want = 0;
        for (uint32_t j = 0; j < m; ++j) {
            const int a = (int)(rnd32() % 61u) - 30;
            const int b = (int)(rnd32() % 61u) - 30;
            row[j] = f2u((float)a);
            tab[j] = f2u((float)b);
            want += (int64_t)a * b;
        }
        tsr_ref_gdn_table(&t, tab, m);
        EQW("integer dot", tsr_ref_gdn_rowdot(row, &t), f2u((float)want));
    }
    // single element: row = [x, 0, ...], table = [c, ...]: the product rounded once, exactly when the
    // table entry keeps all of c (c is the table maximum)
    for (int it = 0; it < 20000; ++it) {
        const uint32_t m = 16;
        memset(row, 0, sizeof row);
        memset(tab, 0, sizeof tab);
        row[3] = (rnd32() & 0x807FFFFFu) | ((uint32_t)(60 + rnd32() % 120u) << 23);
        tab[3] = (rnd32() & 0x807FFFFFu) | ((uint32_t)(60 + rnd32() % 120u) << 23);
        tsr_ref_gdn_table(&t, tab, m);
        // A = M_s * 4 (26 bits, exact), entry = M_t: the product M_s * M_t * 4 is exact; the model is
        // FM(row, tab) up to its own flush (both are RNE of the exact product)
        EQW("single element", tsr_ref_gdn_rowdot(row, &t), f2u(tsr_fmul(u2f(row[3]), u2f(tab[3]))));
    }
    // zero rows and tables
    memset(row, 0, sizeof row);
    tsr_ref_gdn_table(&t, tab, 16);
    EQW("zero row", tsr_ref_gdn_rowdot(row, &t), 0u);
    row[0] = ONE;
    memset(tab, 0, sizeof tab);
    tsr_ref_gdn_table(&t, tab, 16);
    EQW("zero table", tsr_ref_gdn_rowdot(row, &t), 0u);
    // the window edge: elements 2^-24, 2^-25 and 2^-26 below the row maximum keep 2, 1 and 0 units
    {
        const uint32_t tiny[3] = {f2u(ldexpf(1.0f, -24)), f2u(ldexpf(1.0f, -25)), f2u(ldexpf(1.0f, -26))};
        const float want[3] = {1.0f + 15.0f * ldexpf(1.0f, -24), 1.0f + 15.0f * ldexpf(1.0f, -25), 1.0f};
        for (int c = 0; c < 3; ++c) {
            memset(row, 0, sizeof row);
            memset(tab, 0, sizeof tab);
            row[0] = ONE;
            tab[0] = ONE;
            for (int j = 1; j < 16; ++j) {
                row[j] = tiny[c];
                tab[j] = ONE;
            }
            tsr_ref_gdn_table(&t, tab, 16);
            EQW("window edge", tsr_ref_gdn_rowdot(row, &t), f2u(want[c]));
        }
    }
    // cancellation to an exact zero gives +0
    memset(row, 0, sizeof row);
    memset(tab, 0, sizeof tab);
    row[0] = ONE; row[1] = ONE; tab[0] = ONE; tab[1] = 0xBF800000u;
    tsr_ref_gdn_table(&t, tab, 16);
    EQW("1 - 1", tsr_ref_gdn_rowdot(row, &t), 0u);
    // mixed exponents against long double: |err| <= 2^-22 * sum |s t| + the rounding of the result
    for (int it = 0; it < 4000; ++it) {
        const uint32_t m = 16u * (1u + rnd32() % 16u);
        long double exact = 0.0L;
        double sabs = 0.0, tabs = 0.0, smax = 0.0, tmax = 0.0;
        const int spread = 1 + (int)(rnd32() % 30u);
        for (uint32_t j = 0; j < m; ++j) {
            const double sv = (rndu() * 2.0 - 1.0) * ldexp(1.0, (int)(rnd32() % (uint32_t)spread) - spread / 2);
            const double tv = (rndu() * 2.0 - 1.0) * ldexp(1.0, (int)(rnd32() % 6u) - 3);
            row[j] = f2u((float)sv);
            tab[j] = f2u((float)tv);
        }
        tsr_ref_gdn_table(&t, tab, m);
        for (uint32_t j = 0; j < m; ++j) {
            const long double a = (long double)u2f(row[j]), b = (long double)u2f(tab[j]);
            exact += a * b;
            sabs += fabs((double)a);
            tabs += fabs((double)b);
            smax = fmax(smax, fabs((double)a));
            tmax = fmax(tmax, fabs((double)b));
        }
        const double got = (double)u2f(tsr_ref_gdn_rowdot(row, &t));
        const double err = fabs(got - (double)exact);
        // table entries are within 2^-24 tmax of the elements, aligned row elements within 2^-25 smax
        // (VU §7.12.3); the result is rounded once
        const double bound = 1.5 * ldexp(sabs * tmax + smax * tabs, -23) + ldexp(fabs((double)exact), -23);
        CHECK(err <= bound, "mixed dot err %g > %g (m %u exact %Lg)", err, bound, m, exact);
    }
    // Inf / NaN anywhere
    memset(row, 0, sizeof row);
    memset(tab, 0, sizeof tab);
    row[0] = ONE; tab[0] = ONE;
    tsr_ref_gdn_table(&t, tab, 16);
    EQW("clean", tsr_ref_gdn_rowdot(row, &t), ONE);
    row[5] = PINF;
    EQW("inf in the row", tsr_ref_gdn_rowdot(row, &t), CANON);
    row[5] = 0xFFC00001u;
    EQW("nan in the row", tsr_ref_gdn_rowdot(row, &t), CANON);
    row[5] = 0;
    tab[7] = NINF;
    tsr_ref_gdn_table(&t, tab, 16);
    EQW("inf in the table", tsr_ref_gdn_rowdot(row, &t), CANON);
    // result overflow and underflow
    memset(row, 0, sizeof row);
    memset(tab, 0, sizeof tab);
    row[0] = 0x7F000000u; tab[0] = 0x7F000000u;      // 2^127 * 2^127
    tsr_ref_gdn_table(&t, tab, 16);
    EQW("overflow", tsr_ref_gdn_rowdot(row, &t), PINF);
    row[0] = 0x80800000u; tab[0] = 0x00800000u;      // -2^-126 * 2^-126
    tsr_ref_gdn_table(&t, tab, 16);
    EQW("underflow keeps the sign", tsr_ref_gdn_rowdot(row, &t), 0x80000000u);
}

// ---------------------------------------------------------------------------
// scalars
// ---------------------------------------------------------------------------
static void test_scalars(void) {
    for (int it = 0; it < 200000; ++it) {
        const uint32_t a = rnd32(), b = rnd32(), c = rnd32(), d = rnd32();
        EQW("e", tsr_ref_gdn_e(a, b, c), f2u(tsr_fsub(tsr_fmul(u2f(a), u2f(b)), u2f(c))));
        EQW("o", tsr_ref_gdn_o(a, b, c, d),
            f2u(tsr_fadd(tsr_fmul(u2f(a), u2f(b)), tsr_fmul(u2f(c), u2f(d)))));
        EQW("upd", tsr_ref_gdn_upd_elem(a, b, c, d),
            f2u(tsr_fadd(tsr_fmul(u2f(a), u2f(b)), tsr_fmul(u2f(c), u2f(d)))));
    }
}

// ---------------------------------------------------------------------------
// head
// ---------------------------------------------------------------------------
static void gauss_vec(float * x, uint32_t n, double scale, int normalize) {
    double ss = 0.0;
    for (uint32_t j = 0; j < n; ++j) {
        double s = 0.0;
        for (int i = 0; i < 12; ++i) {
            s += rndu();
        }
        x[j] = (float)((s - 6.0) * scale);
        ss += (double)x[j] * (double)x[j];
    }
    if (normalize) {
        const double inv = 1.0 / sqrt(ss);
        for (uint32_t j = 0; j < n; ++j) {
            x[j] = (float)((double)x[j] * inv);
        }
    }
}

static void test_head_vs_double(void) {
    enum { MM = 256, RR = 48 };
    static float s[RR * MM], k[MM], q[MM], v[RR];
    static uint32_t sw[RR * MM], kw[MM], qw[MM], vw[RR], ow[64], sout[RR * MM];
    static double sr[RR * MM];
    double worst_s = 0.0, worst_o = 0.0;
    for (int it = 0; it < 300; ++it) {
        const uint32_t m = 16u * (1u + rnd32() % 16u);
        const uint32_t rows = 1u + rnd32() % RR;
        gauss_vec(s, rows * m, 1.0, 0);
        for (uint32_t i = 0; i < rows; ++i) {   // rows of different scales
            const float sc = ldexpf(1.0f, (int)(rnd32() % 9u) - 4);
            for (uint32_t j = 0; j < m; ++j) {
                s[i * m + j] *= sc;
            }
        }
        gauss_vec(k, m, 1.0, 1);
        gauss_vec(q, m, 1.0, 1);
        gauss_vec(v, rows, 1.0, 0);
        const float alpha = (float)exp(-rndu() * 0.5), beta = (float)(1.0 / (1.0 + exp(-(rndu() * 6.0 - 3.0))));
        for (uint32_t j = 0; j < rows * m; ++j) { sw[j] = f2u(s[j]); sr[j] = (double)s[j]; }
        for (uint32_t j = 0; j < m; ++j) { kw[j] = f2u(k[j]); qw[j] = f2u(q[j]); }
        for (uint32_t i = 0; i < rows; ++i) { vw[i] = f2u(v[i]); }
        tsr_ref_gdn(m, rows, f2u(alpha), f2u(beta), kw, qw, vw, sw, sout, ow);
        for (uint32_t i = 0; i < rows; ++i) {
            double u = 0.0;
            for (uint32_t j = 0; j < m; ++j) {
                u += sr[i * m + j] * (double)k[j];
            }
            const double e = (double)beta * ((double)v[i] - (double)alpha * u);
            double rowmax = 0.0, oref = 0.0, oabs = 0.0;
            for (uint32_t j = 0; j < m; ++j) {
                sr[i * m + j] = (double)alpha * sr[i * m + j] + e * (double)k[j];
                oref += sr[i * m + j] * (double)q[j];
                oabs += fabs(sr[i * m + j] * (double)q[j]);
                if (fabs(sr[i * m + j]) > rowmax) {
                    rowmax = fabs(sr[i * m + j]);
                }
            }
            for (uint32_t j = 0; j < m; ++j) {
                const double err = fabs((double)u2f(sout[i * m + j]) - sr[i * m + j]) / (rowmax + 1e-30);
                if (err > worst_s) {
                    worst_s = err;
                }
            }
            const double oerr = fabs((double)u2f(ow[i]) - oref) / (oabs + 1e-30);   // relative to sum |S'_ij q_j|
            if (oerr > worst_o) {
                worst_o = oerr;
            }
        }
        // the tail of o is zero
        for (uint32_t i = rows; i < 16u * ((rows + 15u) / 16u); ++i) {
            CHECK(ow[i] == 0u, "o tail word %u", i);
        }
    }
    printf("ref_gdn head vs float64: worst S' error %.3g of the row maximum, worst o error %.3g of sum |S' q|\n", worst_s, worst_o);
    CHECK(worst_s < 2e-6, "S' error %g", worst_s);
    CHECK(worst_o < 2e-6, "o error %g", worst_o);
}

static void test_head_cases(void) {
    enum { MM = 64, RR = 20 };
    static uint32_t s[RR * MM], k[MM], q[MM], v[RR], o[32], out[RR * MM], o2[32], out2[RR * MM];
    for (uint32_t j = 0; j < RR * MM; ++j) { s[j] = f2u((float)((int)(rnd32() % 17u) - 8) * 0.125f); }
    for (uint32_t j = 0; j < MM; ++j) { k[j] = f2u((float)((int)(rnd32() % 9u) - 4) * 0.25f); q[j] = f2u((float)((int)(rnd32() % 9u) - 4) * 0.5f); }
    for (uint32_t i = 0; i < RR; ++i) { v[i] = f2u((float)((int)(rnd32() % 9u) - 4)); }
    const uint32_t alpha = f2u(0.75f), beta = f2u(0.5f);

    // in place equals out of place
    memset(o, 0xAB, sizeof o);
    tsr_ref_gdn(MM, RR, alpha, beta, k, q, v, s, out, o);
    uint32_t cp[RR * MM];
    memcpy(cp, s, sizeof cp);
    tsr_ref_gdn(MM, RR, alpha, beta, k, q, v, cp, cp, o2);
    CHECK(memcmp(cp, out, sizeof cp) == 0, "in place differs");
    CHECK(memcmp(o, o2, RR * 4) == 0, "o differs in place");
    for (uint32_t i = RR; i < 32; ++i) {
        CHECK(o[i] == 0u && o2[i] == 0u, "tail");
    }

    // rows = 0 touches nothing
    memset(out2, 0x5A, sizeof out2);
    memset(o2, 0x5A, sizeof o2);
    tsr_ref_gdn(MM, 0, alpha, beta, k, q, v, s, out2, o2);
    CHECK(((uint32_t *)out2)[0] == 0x5A5A5A5Au && o2[0] == 0x5A5A5A5Au, "rows = 0 wrote");

    // alpha = 0: S' = FA(0, FM(e, k)), u' = +0 so e = FM(beta, v); o = FA(FM(0, w), FM(e, kappa))
    tsr_ref_gdn(MM, RR, 0u, beta, k, q, v, s, out, o);
    {
        tsr_gdn_table tq;
        tsr_ref_gdn_table(&tq, q, MM);
        const uint32_t kap = tsr_ref_gdn_kappa(k, &tq);
        for (uint32_t i = 0; i < RR; ++i) {
            const uint32_t e = f2u(tsr_fmul(u2f(beta), u2f(v[i])));
            for (uint32_t j = 0; j < MM; ++j) {
                EQW("alpha 0 S'", out[i * MM + j],
                    f2u(tsr_fadd(tsr_fmul(0.0f, u2f(s[i * MM + j])), tsr_fmul(u2f(e), u2f(k[j])))));
            }
            uint32_t wq = tsr_ref_gdn_rowdot(s + i * MM, &tq);
            EQW("alpha 0 o", o[i], f2u(tsr_fadd(tsr_fmul(0.0f, u2f(wq)), tsr_fmul(u2f(e), u2f(kap)))));
        }
        // the exact value of kappa for these dyadic inputs
        double kd = 0.0;
        for (uint32_t j = 0; j < MM; ++j) {
            kd += (double)u2f(k[j]) * (double)u2f(q[j]);
        }
        EQW("kappa exact", kap, f2u((float)kd));
    }
    // beta = 0: e = FS(+-0, u') = -u' ... with u' = 0: e = 0 -> S' = FA(FM(alpha, S), 0) = alpha * S
    tsr_ref_gdn(MM, RR, alpha, 0u, k, q, v, s, out, o);
    for (uint32_t j = 0; j < RR * MM; ++j) {
        EQW("beta 0 S'", out[j], f2u(tsr_fmul(u2f(alpha), u2f(s[j]))));
    }
    // zero state: u' = 0, e = beta v
    memset(cp, 0, sizeof cp);
    tsr_ref_gdn(MM, RR, alpha, beta, k, q, v, cp, out, o);
    for (uint32_t i = 0; i < RR; ++i) {
        const uint32_t e = f2u(tsr_fmul(u2f(beta), u2f(v[i])));
        for (uint32_t j = 0; j < MM; ++j) {
            EQW("zero state S'", out[i * MM + j],
                f2u(tsr_fadd(tsr_fmul(u2f(alpha), 0.0f), tsr_fmul(u2f(e), u2f(k[j])))));
        }
    }
    // dyadic exact head: the dots are exact, so the whole head equals the plain f32 formulas on
    // exactly evaluated dots
    tsr_ref_gdn(MM, RR, alpha, beta, k, q, v, s, out, o);
    for (uint32_t i = 0; i < RR; ++i) {
        double ud = 0.0;
        for (uint32_t j = 0; j < MM; ++j) {
            ud += (double)u2f(s[i * MM + j]) * (double)u2f(k[j]);
        }
        // u' = alpha*beta*u is dyadic (0.75 * 0.5 * multiples of 2^-5 * ...): exact in f32 here
        const uint32_t ab = f2u(tsr_fmul(u2f(alpha), u2f(beta)));
        const uint32_t up = f2u((float)((double)u2f(ab) * ud));
        const uint32_t e = f2u(tsr_fsub(tsr_fmul(u2f(beta), u2f(v[i])), u2f(up)));
        for (uint32_t j = 0; j < MM; ++j) {
            EQW("dyadic S'", out[i * MM + j], tsr_ref_gdn_upd_elem(alpha, s[i * MM + j], e, k[j]));
        }
    }
    // poison: NaN in k -> every e is NaN, every S' word of every row is NaN, o is NaN
    memcpy(cp, s, sizeof cp);
    {
        uint32_t k2[MM];
        memcpy(k2, k, sizeof k2);
        k2[9] = CANON;
        tsr_ref_gdn(MM, RR, alpha, beta, k2, q, v, cp, out, o);
        for (uint32_t i = 0; i < RR; ++i) {
            EQW("nan k: o", o[i], CANON);
            EQW("nan k: S' word", out[i * MM + 9], CANON);
        }
    }
    // Inf in one state row poisons that row only
    memcpy(cp, s, sizeof cp);
    cp[3 * MM + 7] = PINF;
    tsr_ref_gdn(MM, RR, alpha, beta, k, q, v, cp, out, o);
    tsr_ref_gdn(MM, RR, alpha, beta, k, q, v, s, out2, o2);
    for (uint32_t i = 0; i < RR; ++i) {
        if (i == 3) {
            EQW("poisoned row: o", o[i], CANON);
            for (uint32_t j = 0; j < MM; ++j) {
                EQW("poisoned row: S'", out[i * MM + j], CANON);
            }
        } else {
            EQW("clean row: o", o[i], o2[i]);
            CHECK(memcmp(out + i * MM, out2 + i * MM, MM * 4) == 0, "clean row %u differs", i);
        }
    }
}

int main(void) {
    test_fx2f();
    test_table();
    test_rowdot();
    test_scalars();
    test_head_vs_double();
    test_head_cases();
    return test_finish("ref_gdn");
}
