// ref_math_gelu16.c — tsr_m_gelu16 / tsr_m_geglu16 against the pinned ggml CPU (VU §6.5,
// §10 item 1; SW §10.1; risk 7 of docs/design/README.md).
//
// The CPU (GGML_GELU_FP16, ggml-cpu/vec.h:46) evaluates GELU as
//     x <= -10 ? 0 : x >= 10 ? x : FP16_TO_FP32(ggml_table_gelu_f16[FP32_TO_FP16(x)])
// and GEGLU with the same clamps times g (vec.h:999-1013, :1529-1543). The table is filled by
// ggml_cpu_init() with FP32_TO_FP16(ggml_gelu_f32(f)), libm tanhf (ggml-cpu.c:3894). This test
// links the pinned ggml-cpu library, calls ggml_cpu_init() and reads that table, so it measures
// the table the backend really uses (its build flags and this host's libm included).
//
// Checks:
//   1. All 65,536 fp16 inputs: GELU16 against the CPU form. Report the mismatches, their
//      fp16-ulp distance, and which side equals the exact GELU(x) rounded to fp16 (double
//      reference). NaN inputs give NaN on both sides (encodings differ: canonical here).
//      Target and gate (VU §6.5, SW §10.1, risk 7 of docs/design/README.md): < 1 % of the entries,
//      no sign-of-zero mismatch, and every mismatch of more than 1 fp16 ulp is one where GELU16
//      is the correctly rounded value and the CPU table entry is not (the CPU's
//      0.5·x·(1 + tanhf(u)) cancels in f32 for x << 0). Whether every mismatch is <= 1 fp16 ulp
//      (the first statement of the target, not reachable for that reason) is reported as
//      information.
//   2. Informative: the pinned table against ggml_gelu_f32's expression recompiled here without
//      and with the fused 1 + a·x·x (the CPU build is -O3 -march=native, where GCC may contract).
//   3. Random f32 inputs (not fp16-exact) for GELU16 and GEGLU16 against the CPU form: every
//      mismatch must be explained by a table-entry mismatch of check 1.
//   4. Directed cases: the ±10 clamps, ±inf, NaN, f32 and fp16 subnormal inputs, GEGLU16's
//      a <= -10 rule with b = inf / NaN.
// Exit status 0 iff the gate of check 1 holds and checks 3-4 pass.

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ggml-cpu.h"
#include "test_util.h"
#include "tsr_ref.h"
#include "tsr_ref_math.h"

extern uint16_t ggml_table_gelu_f16[1 << 16];   // ggml-cpu/vec.cpp (ggml_fp16_t = uint16_t)

#define CANON 0x7FC00000u

static uint16_t g_table[1 << 16];

static int f16_is_nan(uint16_t h) { return (h & 0x7C00u) == 0x7C00u && (h & 0x03FFu) != 0; }
static int f32_is_nan(uint32_t u) { return (u & 0x7FFFFFFFu) > 0x7F800000u; }
static int32_t f16_ord(uint16_t h) { return (h & 0x8000u) ? -(int32_t)(h & 0x7FFFu) : (int32_t)(h & 0x7FFFu); }

// The CPU's GELU and GEGLU (GGML_GELU_FP16 forms), native f32, on the pinned table.
static float cpu_gelu(float x) {
    if (x <= -10.0f) {
        return 0.0f;
    }
    if (x >= 10.0f) {
        return x;
    }
    return tsr_fp16_to_f32(g_table[tsr_f32_to_fp16(x)]);
}
static float cpu_geglu(float x, float g) {
    if (x <= -10.0f) {
        return 0.0f;
    }
    if (x >= 10.0f) {
        return x * g;
    }
    return tsr_fp16_to_f32(g_table[tsr_f32_to_fp16(x)]) * g;
}

// ggml_gelu_f32 (ggml-cpu/vec.h:962-969), recompiled here; fused selects 1 + (a·x)·x as one FMA.
static float local_gelu(float x, int fused) {
    const float GELU_COEF_A = 0.044715f;
    const float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;
    const float inner = fused ? fmaf(GELU_COEF_A * x, x, 1.0f) : 1.0f + GELU_COEF_A * x * x;
    return 0.5f * x * (1.0f + tanhf(SQRT_2_OVER_PI * x * inner));
}

// fp16 code of the exact GELU(x), rounded to nearest-even (x / (1 + exp(-2u)) in double: no
// cancellation for x << 0, unlike 0.5·x·(1 + tanh(u))).
static uint16_t exact_gelu_f16(double x) {
    const double y = x / (1.0 + exp(-2.0 * 0.79788456080286535587989211986876 * (x + 0.044715 * x * x * x)));
    const double ay = fabs(y);
    int e = -14;
    if (ay >= 0x1p-14) {
        (void)frexp(ay, &e);
        e -= 1;
    }
    const double q = ldexp(1.0, e - 10);         // fp16 ulp at y
    return tsr_f32_to_fp16((float)(nearbyint(y / q) * q));
}

int main(void) {
    ggml_cpu_init();
    memcpy(g_table, ggml_table_gelu_f16, sizeof g_table);

    // ---- 1. all fp16 inputs ----------------------------------------------------------------
    static uint8_t entry_bad[1 << 16];   // fp16 code whose table entry disagrees with GELU16
    long n_domain = 0, n_clamp = 0, n_nan = 0, n_match = 0, n_mis = 0, n_zero_sign = 0;
    long n_dist1 = 0, n_dist2 = 0, n_sub_mis = 0;
    long n_ours_cr = 0, n_cpu_cr = 0;            // over the domain: entries equal to the exact RNE
    long mis_ours_only = 0, mis_cpu_only = 0, mis_neither = 0, far_unexcused = 0;
    int32_t max_dist = 0;
    int shown = 0;
    for (uint32_t h = 0; h < (1u << 16); ++h) {
        const float x = tsr_fp16_to_f32((uint16_t)h);
        const uint32_t ours = f2u(tsr_m_gelu16(x));
        const uint32_t cpu = f2u(cpu_gelu(x));
        if (f16_is_nan((uint16_t)h)) {
            ++n_nan;
            CHECK(ours == CANON && f32_is_nan(cpu), "NaN input 0x%04x: ours 0x%08x cpu 0x%08x", h, ours, cpu);
            continue;
        }
        if (!(x > -10.0f && x < 10.0f)) {
            ++n_clamp;
            CHECK(ours == cpu, "clamp input 0x%04x: ours 0x%08x cpu 0x%08x", h, ours, cpu);
            continue;
        }
        ++n_domain;
        const uint16_t ex16 = exact_gelu_f16((double)x);
        const int ours_cr = tsr_f32_to_fp16(u2f(ours)) == ex16;
        const int cpu_cr = g_table[h] == ex16;
        n_ours_cr += ours_cr;
        n_cpu_cr += cpu_cr;
        if (ours == cpu) {
            ++n_match;
            continue;
        }
        ++n_mis;
        entry_bad[h] = 1;
        const uint16_t o16 = tsr_f32_to_fp16(u2f(ours)), c16 = g_table[h];
        CHECK(tsr_f32_bits(tsr_fp16_to_f32(o16)) == ours, "GELU16(0x%04x) = 0x%08x is not fp16-exact", h, ours);
        int32_t d = f16_ord(o16) - f16_ord(c16);
        if (d < 0) d = -d;
        if (d == 0) ++n_zero_sign;
        else if (d == 1) ++n_dist1;
        else ++n_dist2;
        if (d > max_dist) max_dist = d;
        if ((h & 0x7C00u) == 0) ++n_sub_mis;
        if (ours_cr && !cpu_cr) ++mis_ours_only;
        else if (cpu_cr && !ours_cr) ++mis_cpu_only;
        else ++mis_neither;
        if (d > 1 && !(ours_cr && !cpu_cr)) ++far_unexcused;
        if (shown < 12 || d > 1) {
            printf("  mismatch x = 0x%04x (%.9g): GELU16 0x%04x (%.9g), CPU table 0x%04x (%.9g), %d ulp; "
                   "f32 GELU before rounding %.9g; exact RNE 0x%04x\n", h, (double)x, o16,
                   (double)tsr_fp16_to_f32(o16), c16, (double)tsr_fp16_to_f32(c16), (int)d,
                   (double)tsr_m_gelu_tanh(x), ex16);
            ++shown;
        }
    }
    const double pct_all = 100.0 * (double)n_mis / 65536.0;
    const double pct_dom = 100.0 * (double)n_mis / (double)(n_domain ? n_domain : 1);
    printf("ref_math_gelu16: 65536 fp16 inputs: %ld in (-10, 10), %ld clamped (|x| >= 10 incl. inf), %ld NaN\n",
           n_domain, n_clamp, n_nan);
    printf("  table domain: %ld match, %ld mismatch = %.3f %% of 65536 entries (%.3f %% of the domain)\n",
           n_match, n_mis, pct_all, pct_dom);
    printf("  mismatch distance: %ld at 1 fp16 ulp, %ld at >= 2 ulp, %ld sign of zero only; max %d ulp; "
           "%ld of them at fp16-subnormal inputs\n", n_dist1, n_dist2, n_zero_sign, (int)max_dist, n_sub_mis);
    printf("  mismatches by exact fp16 rounding of GELU(x): GELU16 correct and CPU table not %ld, "
           "CPU table correct and GELU16 not %ld, neither %ld\n", mis_ours_only, mis_cpu_only, mis_neither);
    printf("  correctly rounded entries over the domain: GELU16 %ld, CPU table %ld (of %ld)\n",
           n_ours_cr, n_cpu_cr, n_domain);
    printf("  every mismatch <= 1 fp16 ulp (information): %s\n", max_dist <= 1 ? "yes" : "no");
    // Target and exit gate (VU 6.5): a mismatch of more than 1 fp16 ulp is accepted only when
    // GELU16 is the correctly rounded value and the CPU table entry is not (the CPU's own error).
    const int gate_ok = (pct_all < 1.0) && n_zero_sign == 0 && far_unexcused == 0;
    printf("  target (VU 6.5: < 1 %% of entries, no sign-of-zero mismatch, > 1 ulp only where the CPU "
           "table entry is the inexact one): %s\n", gate_ok ? "MET" : "MISSED");

    // ---- 2. informative: the pinned table vs ggml_gelu_f32 recompiled here --------------------
    for (int fused = 0; fused < 2; ++fused) {
        long diff = 0;
        for (uint32_t h = 0; h < (1u << 16); ++h) {
            if (f16_is_nan((uint16_t)h)) continue;
            const uint16_t l16 = tsr_f32_to_fp16(local_gelu(tsr_fp16_to_f32((uint16_t)h), fused));
            if (l16 != g_table[h]) ++diff;
        }
        printf("  info: pinned table vs ggml_gelu_f32 recompiled %s: %ld of 63488 non-NaN entries differ\n",
               fused ? "with fmaf(a*x, x, 1)" : "without contraction", diff);
    }

    // ---- 3. random f32 inputs: every mismatch explained by a table entry ----------------------
    rng_t rng = { 0x5EED6E11ull };
    long unexplained = 0, rnd_mis = 0;
    for (int i = 0; i < 2000000; ++i) {
        // x spread over (-12, 12) plus a share of tiny magnitudes; b a moderate random f32.
        float x = (float)((rng_unit(&rng) * 2.0 - 1.0) * 12.0);
        if ((i & 7) == 0) x = (float)ldexp(rng_unit(&rng) - 0.5, -(int)rng_below(&rng, 40));
        const float b = (float)(rng_normal(&rng) * 4.0);
        const uint32_t o1 = f2u(tsr_m_gelu16(x)), c1 = f2u(cpu_gelu(x));
        const uint32_t o2 = f2u(tsr_m_geglu16(x, b)), c2 = f2u(cpu_geglu(x, b));
        // GEGLU: the CPU's native product keeps subnormals; ours flushes them (FTZ).
        const int c2_sub = (c2 & 0x7F800000u) == 0 && (c2 & 0x007FFFFFu) != 0;
        const int bad1 = o1 != c1, bad2 = o2 != c2 && !(c2_sub && (o2 & 0x7FFFFFFFu) == 0);
        if (bad1 || bad2) {
            ++rnd_mis;
            const uint16_t code = tsr_f32_to_fp16(x);
            const int in_dom = x > -10.0f && x < 10.0f;
            if (!in_dom || !entry_bad[code]) {
                if (++unexplained <= 10) {
                    fprintf(stderr, "unexplained: x 0x%08x b 0x%08x: gelu16 0x%08x cpu 0x%08x; geglu16 0x%08x cpu 0x%08x\n",
                            f2u(x), f2u(b), o1, c1, o2, c2);
                }
            }
        }
    }
    CHECK(unexplained == 0, "%ld random GELU16/GEGLU16 mismatches not explained by a table entry", unexplained);
    printf("  random f32 inputs: 2000000 (GELU16 and GEGLU16), %ld mismatches, all explained by table entries: %s\n",
           rnd_mis, unexplained ? "NO" : "yes");

    // ---- 4. directed cases -------------------------------------------------------------------
    struct { uint32_t x, want; } g16[] = {
        { 0x41200000u, 0x41200000u },   // 10 -> 10 (clamp)
        { 0xC1200000u, 0x00000000u },   // -10 -> +0 (clamp)
        { 0x41200001u, 0x41200001u },   // > 10 -> x
        { 0xC1200001u, 0x00000000u },   // < -10 -> +0
        { 0x7F800000u, 0x7F800000u },   // +inf
        { 0xFF800000u, 0x00000000u },   // -inf -> +0
        { 0x7FC00000u, CANON }, { 0xFFC00001u, CANON }, { 0x7F800001u, CANON },
        { 0x00000001u, 0x00000000u },   // f32 subnormal -> DAZ -> GELU16(+0) = +0
        { 0x80000001u, 0x80000000u },   // -> -0
        { 0x00000000u, 0x00000000u }, { 0x80000000u, 0x80000000u },
        { 0xC11FFFFFu, 0x80000000u },   // -9.9999995: H(x) = -10, GELU(-10) ~ -1.2e-37 -> fp16 -0
    };
    for (size_t i = 0; i < sizeof g16 / sizeof g16[0]; ++i) {
        const uint32_t got = f2u(tsr_m_gelu16(u2f(g16[i].x)));
        CHECK(got == g16[i].want, "GELU16(0x%08x) = 0x%08x, want 0x%08x", g16[i].x, got, g16[i].want);
    }
    // 9.9999995 rounds to fp16 10.0 and is computed (not clamped): H(GELU(10)) = 10.
    CHECK(f2u(tsr_m_gelu16(u2f(0x411FFFFFu))) == 0x41200000u, "GELU16(0x411FFFFF) = 0x%08x",
          f2u(tsr_m_gelu16(u2f(0x411FFFFFu))));
    // fp16 subnormal inputs are kept (gradual underflow), not flushed: GELU16(2^-14 · 1023/1024)
    // is an fp16 subnormal ~ x/2 and nonzero.
    {
        const uint32_t got = f2u(tsr_m_gelu16(tsr_fp16_to_f32(0x03FFu)));
        const uint16_t g = tsr_f32_to_fp16(u2f(got));
        CHECK(g != 0 && (g & 0x7C00u) == 0, "GELU16(fp16 0x03ff) = 0x%08x (fp16 0x%04x)", got, g);
    }
    struct { uint32_t a, b, want; } gg[] = {
        { 0xC1200000u, 0x7F800000u, 0x00000000u },   // a = -10, b = inf -> +0 (not NaN)
        { 0xFF800000u, 0x7FC00000u, 0x00000000u },   // a = -inf, b = NaN -> +0
        { 0xC1200001u, 0xFF800000u, 0x00000000u },
        { 0x7FC00000u, 0x3F800000u, CANON },         // a = NaN -> NaN
        { 0x41200000u, 0x40000000u, 0x41A00000u },   // 10 · 2 = 20
        { 0x7F800000u, 0x00000000u, CANON },         // inf · 0 -> NaN (as the CPU's x·g)
        { 0x7F800000u, 0xBF800000u, 0xFF800000u },
        { 0xC11FFFFFu, 0x7FC00000u, CANON },         // a > -10: FM(GELU16(a), NaN) = NaN
    };
    for (size_t i = 0; i < sizeof gg / sizeof gg[0]; ++i) {
        const uint32_t got = f2u(tsr_m_geglu16(u2f(gg[i].a), u2f(gg[i].b)));
        CHECK(got == gg[i].want, "GEGLU16(0x%08x, 0x%08x) = 0x%08x, want 0x%08x", gg[i].a, gg[i].b, got, gg[i].want);
    }

    CHECK(gate_ok, "GELU16 gate: %ld mismatches (%.3f %%), max %d fp16 ulp, %ld sign-of-zero, "
          "%ld mismatches > 1 ulp not explained by the CPU table", n_mis, pct_all, (int)max_dist, n_zero_sign,
          far_unexcused);
    return test_finish("ref_math_gelu16");
}
