// ref_math_acc.c — accuracy of the tsr_ref_math functions against double-precision libm, and
// their structural properties (VU §10 items 1-2, SW §10.1 (b), SW-23).
//
//   ref_math_acc [--full | --count N --stride S --start X] [--threads T] [--slow] [func ...]
//
// Sweep: the f32 inputs with bits start + i·stride (uint32 wraparound), i < count, plus a fixed
// list of special values. Default: 2^24 inputs with stride 257 (odd, so the points are distinct
// and spread over the whole bit space). --full: all 2^32 inputs. --slow: the full sweep, run only
// if the environment has TSR_REF_MATH_FULL=1, else exit 77 (ctest SKIP_RETURN_CODE).
// Threads: --threads, else TSR_REF_MATH_THREADS, else min(8, online CPUs).
//
// Error measures (y = our f32 result, r = the exact value in double):
//   - ulp: |y - r| / ulp(r), ulp(r) = 2^(max(floor(log2|r|), -126) - 23); +-inf results count as
//     +-2^128 and |r| is clamped to 2^128. Results with |r| < 2^-126 lie in the FTZ region: y = 0
//     or +-2^-126 is correct there and is not measured. y = 0 for r within 4 ulp above 2^-126 is
//     an FTZ edge (an approximation just below 2^-126, then flushed), counted but not measured.
//   - abs: |y - r| for finite y and r.
// Gate (VU §10 item 2): every measured maximum with a *model* figure from VU §6.9 is <= 2x the
// figure, no NaN mismatch (NaN exactly where r is NaN), and no property violation:
//   every NaN result is 0x7FC00000; no result is subnormal; finite inputs give no NaN;
//   EXPK: never -0, monotonic in v (non-decreasing for P1, non-increasing for N1/N2/NG),
//   EXPK(v, N2) = EXPK(2v, N1) for normal v with 2v finite; SIN, COS in [-1, 1].
// The domains of the gated figures are this file's choice where §6.9 states none (see the
// metric labels): SILU is measured where σ(x) >= 2^-126 (below that SIG is flushed to 0 by the
// spec's FTZ and SILU returns -0), SIN/COS on |θ| <= 2^18 (the model's sample range and the
// RoPE range of VU-6); |θ| < 2^24 is reported as well, larger θ only property-checked.

#define _POSIX_C_SOURCE 200809L   // clock_gettime, sysconf

#include <errno.h>
#include <float.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "tsr_ref.h"
#include "tsr_ref_math.h"

#define CANON 0x7FC00000u

static inline uint32_t f2u(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static inline float u2f(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static inline int nan_bits(uint32_t u) { return (u & 0x7FFFFFFFu) > 0x7F800000u; }
static inline int inf_bits(uint32_t u) { return (u & 0x7FFFFFFFu) == 0x7F800000u; }

// ---------------------------------------------------------------------------
// Functions under test
// ---------------------------------------------------------------------------
enum { MET_ULP, MET_ABS };
enum {
    P_NO_NEGZERO = 1u << 0,   // never returns -0
    P_MONO_INC   = 1u << 1,   // non-decreasing in x
    P_MONO_DEC   = 1u << 2,   // non-increasing in x
    P_N2N1       = 1u << 3,   // EXPK(v, N2) = EXPK(2v, N1)
    P_UNIT       = 1u << 4,   // |y| <= 1
};

typedef struct {
    const char * label;
    int kind;                 // MET_ULP / MET_ABS
    double model;             // VU §6.9 *model* figure (ulp or absolute); 0 = reported only
    int (*dom)(double x, double r);
} metric_t;

#define MAX_MET 5
typedef struct {
    const char * name;
    uint32_t (*eval)(uint32_t x);
    double (*ref)(double x);
    double ref_max_abs_x;     // > 0: no reference for |x| >= this (property checks only)
    unsigned props;
    metric_t met[MAX_MET];
} func_t;

static const double C2G = 2.0 * 0.79788456080286535587989211986876 * 0.044715;   // 2·sqrt(2/π)·0.044715

static uint32_t e_expk_p1(uint32_t x) { return f2u(tsr_m_expk(u2f(x), TSR_M_P1)); }
static uint32_t e_expk_n1(uint32_t x) { return f2u(tsr_m_expk(u2f(x), TSR_M_N1)); }
static uint32_t e_expk_n2(uint32_t x) { return f2u(tsr_m_expk(u2f(x), TSR_M_N2)); }
static uint32_t e_expk_ng(uint32_t x) { return f2u(tsr_m_expk(u2f(x), TSR_M_NG)); }
static uint32_t e_sigmoid(uint32_t x) { return f2u(tsr_m_sigmoid(u2f(x))); }
static uint32_t e_silu(uint32_t x)    { return f2u(tsr_m_silu(u2f(x))); }
static uint32_t e_gelu(uint32_t x)    { return f2u(tsr_m_gelu_tanh(u2f(x))); }
static uint32_t e_tanh(uint32_t x)    { return f2u(tsr_m_tanh(u2f(x))); }
static uint32_t e_sin(uint32_t x)     { return f2u(tsr_m_sin(u2f(x))); }
static uint32_t e_cos(uint32_t x)     { return f2u(tsr_m_cos(u2f(x))); }

static double r_expk_p1(double x) { return exp(x); }
static double r_expk_n1(double x) { return exp(-x); }
static double r_expk_n2(double x) { return exp(-2.0 * x); }
static double r_expk_ng(double x) { return exp(-C2G * x); }
static double r_sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
static double r_silu(double x)    { return x / (1.0 + exp(-x)); }
// x·σ(2u) has no cancellation, unlike 0.5·x·(1 + tanh(u)) for x << 0.
static double r_gelu(double x)    { return x / (1.0 + exp(-2.0 * 0.79788456080286535587989211986876 * (x + 0.044715 * x * x * x))); }
static double r_tanh(double x)    { return tanh(x); }
static double r_sin(double x)     { return sin(x); }
static double r_cos(double x)     { return cos(x); }

static int d_all(double x, double r)       { (void)x; (void)r; return 1; }
static int d_sig_normal(double x, double r) { (void)r; return 1.0 / (1.0 + exp(-x)) >= 0x1p-126; }
static int d_gelu_sig_normal(double x, double r) {   // σ(2u) of GELU normal
    (void)r;
    return 1.0 / (1.0 + exp(-2.0 * 0.79788456080286535587989211986876 * (x + 0.044715 * x * x * x))) >= 0x1p-126;
}
static int d_ge_m3(double x, double r)     { (void)r; return x >= -3.0; }
static int d_ge_m5(double x, double r)     { (void)r; return x >= -5.0; }
static int d_pm8(double x, double r)       { (void)r; return fabs(x) <= 8.0; }
static int d_t18(double x, double r)       { (void)r; return fabs(x) <= 0x1p18; }
static int d_t18_big(double x, double r)   { return fabs(x) <= 0x1p18 && fabs(r) > 1e-3; }
static int d_t24(double x, double r)       { (void)r; return fabs(x) < 0x1p24; }
static int d_t24_big(double x, double r)   { return fabs(x) < 0x1p24 && fabs(r) > 1e-3; }

static const func_t k_funcs[] = {
    { "expk_p1", e_expk_p1, r_expk_p1, 0, P_NO_NEGZERO | P_MONO_INC,
      { { "ulp, all x", MET_ULP, 0.53, d_all } } },
    { "expk_n1", e_expk_n1, r_expk_n1, 0, P_NO_NEGZERO | P_MONO_DEC,
      { { "ulp, all x", MET_ULP, 0.53, d_all } } },
    { "expk_n2", e_expk_n2, r_expk_n2, 0, P_NO_NEGZERO | P_MONO_DEC | P_N2N1,
      { { "ulp, all x", MET_ULP, 0.53, d_all } } },
    { "expk_ng", e_expk_ng, r_expk_ng, 0, P_NO_NEGZERO | P_MONO_DEC,
      { { "ulp, all x", MET_ULP, 0.53, d_all } } },
    { "sigmoid", e_sigmoid, r_sigmoid, 0, 0,
      { { "ulp, all x", MET_ULP, 2.0, d_all } } },
    { "silu", e_silu, r_silu, 0, 0,
      { { "ulp, sigma(x) >= 2^-126", MET_ULP, 2.3, d_sig_normal },
        { "ulp, all x (info)", MET_ULP, 0, d_all } } },
    { "gelu_tanh", e_gelu, r_gelu, 0, 0,
      { { "ulp, x >= -3", MET_ULP, 11.0, d_ge_m3 },
        { "ulp, x >= -5", MET_ULP, 29.0, d_ge_m5 },
        { "abs, |x| <= 8", MET_ABS, 5.2e-7, d_pm8 },
        { "ulp, sigma(2u) >= 2^-126 (info)", MET_ULP, 0, d_gelu_sig_normal },
        { "ulp, all x (info)", MET_ULP, 0, d_all } } },
    { "tanh", e_tanh, r_tanh, 0, 0,
      { { "ulp, all x", MET_ULP, 9.8, d_all },
        { "abs, all x", MET_ABS, 1.2e-7, d_all } } },
    { "sin", e_sin, r_sin, 0x1p24, P_UNIT,
      { { "abs, |t| <= 2^18", MET_ABS, 7.545e-8, d_t18 },   // 2^-23.66
        { "ulp, |t| <= 2^18, |y| > 1e-3", MET_ULP, 1.5, d_t18_big },
        { "abs, |t| < 2^24 (info)", MET_ABS, 0, d_t24 },
        { "ulp, |t| < 2^24, |y| > 1e-3 (info)", MET_ULP, 0, d_t24_big } } },
    { "cos", e_cos, r_cos, 0x1p24, P_UNIT,
      { { "abs, |t| <= 2^18", MET_ABS, 7.545e-8, d_t18 },   // 2^-23.66
        { "ulp, |t| <= 2^18, |y| > 1e-3", MET_ULP, 1.5, d_t18_big },
        { "abs, |t| < 2^24 (info)", MET_ABS, 0, d_t24 },
        { "ulp, |t| < 2^24, |y| > 1e-3 (info)", MET_ULP, 0, d_t24_big } } },
};
#define N_FUNCS (sizeof k_funcs / sizeof k_funcs[0])

static const uint32_t k_specials[] = {
    0x00000000u, 0x80000000u, 0x00000001u, 0x80000001u, 0x007FFFFFu, 0x807FFFFFu,
    0x00800000u, 0x80800000u, 0x3F800000u, 0xBF800000u, 0x3F000000u, 0xBF000000u,
    0x40000000u, 0xC0000000u, 0x7F7FFFFFu, 0xFF7FFFFFu, 0x7F800000u, 0xFF800000u,
    0x7FC00000u, 0xFFC00000u, 0x7F800001u, 0xFFFFFFFFu, 0x41200000u, 0xC1200000u,
    0x42B17217u, 0x42B17218u, 0xC2AEAC4Fu, 0xC2AEAC50u, 0xC2B17217u, 0xC2B17218u,   // ln FLT_MAX, ln 2^-126
    0x4E000000u, 0x4E7FFFFFu, 0x4E800000u, 0xCE800000u,                             // EXPK Ev = 156 / 157
    0x39800000u, 0x397FFFFFu, 0xB9800000u, 0x3C800000u, 0x3C7FFFFFu, 0xBC800000u,   // 2^-12, 2^-6 edges
    0x3F490FDBu, 0x3FC90FDBu, 0x40490FDBu, 0x40C90FDBu, 0x48800000u, 0x4B800000u,   // π/4 .. 2π, 2^18, 2^24
    0x4B7FFFFFu, 0x5A000000u, 0x7E800000u, 0xC0400000u, 0xC0A00000u, 0xC1000000u,
};

// ---------------------------------------------------------------------------
// Accumulation
// ---------------------------------------------------------------------------
typedef struct {
    double max;               // -1: no point yet
    uint32_t x, y;
    double r;
    uint64_t n;
} mstat_t;

#define MAX_EX 4
typedef struct {
    uint64_t inputs, nan_ok, nan_bad, ftz, ftz_edge, ftz_bad, prop_bad, no_ref;
    mstat_t m[MAX_MET];
    uint32_t ex_x[MAX_EX], ex_y[MAX_EX];
    const char * ex_what[MAX_EX];
    int n_ex;
    // monotonicity: previous evaluated non-NaN input
    int have_prev;
    float prev_x, prev_y;
} acc_t;

static void acc_init(acc_t * a) {
    memset(a, 0, sizeof *a);
    for (int i = 0; i < MAX_MET; ++i) {
        a->m[i].max = -1.0;
    }
}

static void note_bad(acc_t * a, const char * what, uint32_t x, uint32_t y) {
    ++a->prop_bad;
    if (a->n_ex < MAX_EX) {
        a->ex_what[a->n_ex] = what;
        a->ex_x[a->n_ex] = x;
        a->ex_y[a->n_ex] = y;
        ++a->n_ex;
    }
}

static void mupdate(mstat_t * m, double err, uint32_t x, uint32_t y, double r) {
    ++m->n;
    if (err > m->max) {
        m->max = err;
        m->x = x;
        m->y = y;
        m->r = r;
    }
}

static double ulp_of(double ar) {   // ar = |r| finite, clamped to <= 2^128
    int e = -126;
    if (ar >= 0x1p-126) {
        (void)frexp(ar, &e);
        e -= 1;                     // ar in [2^e, 2^(e+1))
    }
    if (e > 127) {
        e = 127;
    }
    return ldexp(1.0, e - 23);
}

static void eval_one(const func_t * f, acc_t * a, uint32_t x) {
    const uint32_t y = f->eval(x);
    const float xf = u2f(x), yf = u2f(y);
    ++a->inputs;

    // Properties.
    if (nan_bits(y) && y != CANON) {
        note_bad(a, "non-canonical NaN", x, y);
    }
    if ((y & 0x7F800000u) == 0 && (y & 0x007FFFFFu) != 0) {
        note_bad(a, "subnormal result", x, y);
    }
    if (!nan_bits(x) && !inf_bits(x) && nan_bits(y)) {
        note_bad(a, "NaN for a finite input", x, y);
    }
    if ((f->props & P_NO_NEGZERO) && y == 0x80000000u) {
        note_bad(a, "-0 result", x, y);
    }
    if ((f->props & P_UNIT) && !nan_bits(y) && (y & 0x7FFFFFFFu) > 0x3F800000u) {
        note_bad(a, "|y| > 1", x, y);
    }
    if ((f->props & P_N2N1) && (x & 0x7F800000u) != 0 && (x & 0x7F800000u) < 0x7F000000u) {
        const uint32_t x2 = x + 0x00800000u;   // 2x, exact for normal x with E <= 253
        if (f2u(tsr_m_expk(u2f(x2), TSR_M_N1)) != y) {
            note_bad(a, "EXPK(v,N2) != EXPK(2v,N1)", x, y);
        }
    }
    if ((f->props & (P_MONO_INC | P_MONO_DEC)) && !nan_bits(x)) {
        if (a->have_prev && !nan_bits(f2u(a->prev_y)) && !nan_bits(y)) {
            const double px = a->prev_x, cx = xf, py = a->prev_y, cy = yf;
            const int inc = (f->props & P_MONO_INC) != 0;
            if ((px < cx && (inc ? py > cy : py < cy)) || (px > cx && (inc ? py < cy : py > cy))) {
                note_bad(a, "not monotonic", x, y);
            }
        }
        a->have_prev = 1;
        a->prev_x = xf;
        a->prev_y = yf;
    }

    // Accuracy.
    if (nan_bits(x) || (f->ref_max_abs_x > 0 && fabs((double)xf) >= f->ref_max_abs_x)) {
        if (nan_bits(x)) {
            if (nan_bits(y)) ++a->nan_ok; else ++a->nan_bad;
        } else {
            ++a->no_ref;
        }
        return;
    }
    const double xd = (double)xf;
    // DAZ: a subnormal input is a signed zero for the function.
    const double xin = ((x & 0x7F800000u) == 0) ? copysign(0.0, xd) : xd;
    const double r = f->ref(xin);
    if (isnan(r) || nan_bits(y)) {
        if (isnan(r) && nan_bits(y)) {
            ++a->nan_ok;
        } else {
            ++a->nan_bad;
            note_bad(a, isnan(r) ? "no NaN where exact is NaN" : "NaN where exact is not", x, y);
        }
        return;
    }
    const double yd = inf_bits(y) ? copysign(0x1p128, (double)yf) : (double)yf;
    const double rc = (fabs(r) > 0x1p128) ? copysign(0x1p128, r) : r;
    const double ar = fabs(rc);
    double ulp_err = -1.0;   // < 0: not measured
    if (ar < 0x1p-126) {
        if (y == 0u || y == 0x80000000u || (y & 0x7FFFFFFFu) == 0x00800000u) {
            ++a->ftz;
        } else {
            ++a->ftz_bad;
            ulp_err = fabs(yd - rc) / 0x1p-149;
        }
    } else if ((y & 0x7FFFFFFFu) == 0 && ar < 0x1p-126 * (1.0 + 0x1p-21)) {
        ++a->ftz_edge;
    } else {
        ulp_err = fabs(yd - rc) / ulp_of(ar);
    }
    const int finite = !inf_bits(y) && isfinite(r);
    for (int i = 0; i < MAX_MET && f->met[i].label; ++i) {
        const metric_t * m = &f->met[i];
        if (!m->dom(xin, r)) {
            continue;
        }
        if (m->kind == MET_ULP && ulp_err >= 0) {
            mupdate(&a->m[i], ulp_err, x, y, r);
        } else if (m->kind == MET_ABS && finite) {
            mupdate(&a->m[i], fabs((double)yf - r), x, y, r);
        }
    }
}

static void acc_merge(acc_t * d, const acc_t * s) {
    d->inputs += s->inputs; d->nan_ok += s->nan_ok; d->nan_bad += s->nan_bad; d->ftz += s->ftz;
    d->ftz_edge += s->ftz_edge; d->ftz_bad += s->ftz_bad; d->prop_bad += s->prop_bad;
    d->no_ref += s->no_ref;
    for (int i = 0; i < MAX_MET; ++i) {
        d->m[i].n += s->m[i].n;
        if (s->m[i].max > d->m[i].max) {
            d->m[i].max = s->m[i].max; d->m[i].x = s->m[i].x; d->m[i].y = s->m[i].y; d->m[i].r = s->m[i].r;
        }
    }
    for (int i = 0; i < s->n_ex && d->n_ex < MAX_EX; ++i) {
        d->ex_what[d->n_ex] = s->ex_what[i]; d->ex_x[d->n_ex] = s->ex_x[i]; d->ex_y[d->n_ex] = s->ex_y[i];
        ++d->n_ex;
    }
}

typedef struct {
    const func_t * f;
    uint32_t start, stride;
    uint64_t i0, i1;
    acc_t acc;
} job_t;

static void * worker(void * p) {
    job_t * j = (job_t *)p;
    acc_init(&j->acc);
    if (j->i0 > 0 && (j->f->props & (P_MONO_INC | P_MONO_DEC))) {
        // Seed the monotonicity check with the previous input of the sweep.
        const uint32_t xp = j->start + (uint32_t)((j->i0 - 1) * j->stride);
        if (!nan_bits(xp)) {
            j->acc.have_prev = 1;
            j->acc.prev_x = u2f(xp);
            j->acc.prev_y = u2f(j->f->eval(xp));
        }
    }
    uint32_t x = j->start + (uint32_t)(j->i0 * j->stride);
    for (uint64_t i = j->i0; i < j->i1; ++i, x += j->stride) {
        eval_one(j->f, &j->acc, x);
    }
    return NULL;
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------
static int parse_u64(const char * s, uint64_t * out) {
    char * end = NULL;
    errno = 0;
    const unsigned long long v = strtoull(s, &end, (s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) ? 16 : 10);
    if (errno || end == s || *end != '\0') {
        return 0;
    }
    *out = (uint64_t)v;
    return 1;
}

static void usage(void) {
    fprintf(stderr, "usage: ref_math_acc [--full | --count N --stride S --start X] [--threads T] [--slow] [func ...]\n"
                    "functions:");
    for (size_t i = 0; i < N_FUNCS; ++i) {
        fprintf(stderr, " %s", k_funcs[i].name);
    }
    fprintf(stderr, "\n");
    exit(2);
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static int run_func(const func_t * f, uint32_t start, uint64_t count, uint32_t stride, int nthreads) {
    const double t0 = now_s();
    job_t * jobs = calloc((size_t)nthreads, sizeof *jobs);
    pthread_t * th = calloc((size_t)nthreads, sizeof *th);
    if (!jobs || !th) {
        fprintf(stderr, "out of memory\n");
        exit(1);
    }
    for (int t = 0; t < nthreads; ++t) {
        jobs[t].f = f;
        jobs[t].start = start;
        jobs[t].stride = stride;
        jobs[t].i0 = count * (uint64_t)t / (uint64_t)nthreads;
        jobs[t].i1 = count * (uint64_t)(t + 1) / (uint64_t)nthreads;
        if (pthread_create(&th[t], NULL, worker, &jobs[t]) != 0) {
            fprintf(stderr, "pthread_create failed\n");
            exit(1);
        }
    }
    acc_t total;
    acc_init(&total);
    for (int t = 0; t < nthreads; ++t) {
        pthread_join(th[t], NULL);
        acc_merge(&total, &jobs[t].acc);
    }
    acc_t sp;
    acc_init(&sp);
    for (size_t i = 0; i < sizeof k_specials / sizeof k_specials[0]; ++i) {
        sp.have_prev = 0;   // specials are not in order
        eval_one(f, &sp, k_specials[i]);
    }
    acc_merge(&total, &sp);
    free(jobs);
    free(th);

    int fail = 0;
    printf("%s: %llu inputs (%llu swept + %zu specials), %.1f s\n", f->name,
           (unsigned long long)total.inputs, (unsigned long long)count,
           sizeof k_specials / sizeof k_specials[0], now_s() - t0);
    printf("  NaN ok %llu, NaN mismatches %llu; FTZ region %llu, FTZ edge %llu, FTZ violations %llu; "
           "no reference (|x| too large) %llu\n",
           (unsigned long long)total.nan_ok, (unsigned long long)total.nan_bad,
           (unsigned long long)total.ftz, (unsigned long long)total.ftz_edge,
           (unsigned long long)total.ftz_bad, (unsigned long long)total.no_ref);
    printf("  %-36s %14s %11s %11s  %-6s %s\n", "metric", "measured", "model", "gate 2x", "result",
           "worst: input -> result (exact)");
    for (int i = 0; i < MAX_MET && f->met[i].label; ++i) {
        const metric_t * m = &f->met[i];
        const mstat_t * s = &total.m[i];
        const char * res = "info";
        if (m->model > 0) {
            if (s->n == 0) {
                res = "EMPTY";
                fail = 1;
            } else if (s->max <= 2.0 * m->model) {
                res = "PASS";
            } else {
                res = "FAIL";
                fail = 1;
            }
        }
        char meas[32], model[32], gate[32];
        if (m->kind == MET_ULP) {
            snprintf(meas, sizeof meas, "%.4f ulp", s->max < 0 ? 0.0 : s->max);
            snprintf(model, sizeof model, "%.3g", m->model);
            snprintf(gate, sizeof gate, "%.3g", 2.0 * m->model);
        } else {
            snprintf(meas, sizeof meas, "%.4g (2^%.2f)", s->max < 0 ? 0.0 : s->max,
                     s->max > 0 ? log2(s->max) : -(double)INFINITY);
            snprintf(model, sizeof model, "%.3g", m->model);
            snprintf(gate, sizeof gate, "%.3g", 2.0 * m->model);
        }
        if (m->model <= 0) {
            snprintf(model, sizeof model, "-");
            snprintf(gate, sizeof gate, "-");
        }
        printf("  %-36s %14s %11s %11s  %-6s 0x%08x (%.9g) -> 0x%08x (%.9g) (%.12g)  [%llu pts]\n", m->label,
               meas, model, gate, res, s->x, (double)u2f(s->x), s->y, (double)u2f(s->y), s->r,
               (unsigned long long)s->n);
    }
    if (total.nan_bad || total.ftz_bad) {
        fail = 1;
    }
    if (total.prop_bad) {
        fail = 1;
        printf("  property violations: %llu\n", (unsigned long long)total.prop_bad);
        for (int i = 0; i < total.n_ex; ++i) {
            printf("    %s: x 0x%08x -> 0x%08x\n", total.ex_what[i], total.ex_x[i], total.ex_y[i]);
        }
    } else {
        printf("  properties: ok\n");
    }
    printf("  => %s\n", fail ? "FAIL" : "PASS");
    fflush(stdout);
    return fail;
}

int main(int argc, char ** argv) {
    uint64_t count = 1ull << 24, stride = 257, start = 0, threads = 0;
    int full = 0, slow = 0;
    const func_t * sel[N_FUNCS];
    size_t nsel = 0;
    for (int i = 1; i < argc; ++i) {
        const char * a = argv[i];
        if (strcmp(a, "--full") == 0) {
            full = 1;
        } else if (strcmp(a, "--slow") == 0) {
            slow = 1;
        } else if (strcmp(a, "--count") == 0 && i + 1 < argc) {
            if (!parse_u64(argv[++i], &count) || count > (1ull << 32)) usage();
        } else if (strcmp(a, "--stride") == 0 && i + 1 < argc) {
            if (!parse_u64(argv[++i], &stride) || stride > 0xFFFFFFFFull) usage();
        } else if (strcmp(a, "--start") == 0 && i + 1 < argc) {
            if (!parse_u64(argv[++i], &start) || start > 0xFFFFFFFFull) usage();
        } else if (strcmp(a, "--threads") == 0 && i + 1 < argc) {
            if (!parse_u64(argv[++i], &threads) || threads == 0 || threads > 256) usage();
        } else if (a[0] == '-') {
            usage();
        } else {
            size_t k = 0;
            while (k < N_FUNCS && strcmp(k_funcs[k].name, a) != 0) ++k;
            if (k == N_FUNCS) {
                fprintf(stderr, "unknown function '%s'\n", a);
                usage();
            }
            if (nsel < N_FUNCS) sel[nsel++] = &k_funcs[k];
        }
    }
    if (slow) {
        const char * e = getenv("TSR_REF_MATH_FULL");
        if (!e || strcmp(e, "1") != 0) {
            printf("ref_math_acc: full 2^32 sweep skipped (set TSR_REF_MATH_FULL=1 to run it)\n");
            return 77;
        }
        full = 1;
    }
    if (full) {
        count = 1ull << 32;
        stride = 1;
        start = 0;
    }
    if (threads == 0) {
        const char * e = getenv("TSR_REF_MATH_THREADS");
        uint64_t t = 0;
        if (e && parse_u64(e, &t) && t > 0 && t <= 256) {
            threads = t;
        } else {
            const long n = sysconf(_SC_NPROCESSORS_ONLN);
            threads = (n > 8) ? 8 : (n < 1 ? 1 : (uint64_t)n);
        }
    }
    if (nsel == 0) {
        for (size_t k = 0; k < N_FUNCS; ++k) sel[nsel++] = &k_funcs[k];
    }
    printf("ref_math_acc: sweep start 0x%08llx, count %llu, stride %llu, %llu threads\n",
           (unsigned long long)start, (unsigned long long)count, (unsigned long long)stride,
           (unsigned long long)threads);
    int fail = 0;
    for (size_t k = 0; k < nsel; ++k) {
        fail |= run_func(sel[k], (uint32_t)start, count, (uint32_t)stride, (int)threads);
    }
    printf("ref_math_acc: %s\n", fail ? "FAILED" : "all gates passed");
    return fail ? 1 : 0;
}
