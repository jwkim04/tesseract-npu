// test_ref_vu.c — directed and cross checks of the VU references (sw/sim/tsr_ref_vu.c; VU §4–§6,
// SW §10.1 "VU/AT references", VU §10 items 1 and 3). The independent numpy model
// (tests/ref_vu_np.py, ctest ref_vu_np) is the bit-level oracle on generated vectors; this test
// holds the directed cases and the cross-checks against ggml.
//
// Checks (one function each):
//   order      total order, MAXT / MINT (VU §6.8, VU-22): NaN, ±0, DAZ, ±inf
//   vop_elem   every function's specials (VU §6.5 table, VU-21, VU-23 raw ops) and each
//              function against its VU §5.2 formula on tsr_ref_math / tsr_ref primitives
//   vop_shape  rows x m with line strides, stride-0 broadcast, cmode SREG / VEC, the tail rule,
//              words outside the extent untouched, in place, m = 0 / rows = 0 (VU §4.2–§4.5)
//   red        the -0 subtree cases of VU §6.2 at m = 8k + r (complete tree against both skip
//              models; the naive skip without z() is shown to differ), empty rows, MAX order,
//              SUMSQ rounding, the post-ops (RMS = the spec's SOP recipe, VU-3), TO_S, the
//              vector destination and its tail
//   rope       identity at position 0, NORM / NEOX pairs, the raw tail, in place, strides; the
//              angles against a C transcription of ggml_rope_cache_init (ext_factor = 0) for
//              Qwen2.5 and Gemma 4 parameters at positions 0 .. 2^17 - 1 (bit-identical,
//              VU-19, VU §10 item 1); the rotation's accuracy against double
//   topk       ties, NaN, ±0, DAZ, padding, tail, strides; random rows against a sort
//   sop        every sub incl. the two-rounding RSQRT and the raw moves
//   dqrow      against tsr_dequant_row with strides and tails
//   softmax    EXPSUB sequences: an online softmax composed of RED / VOP / SOP steps, a fully
//              masked first block stays finite (VU-21), accuracy against double
// With TSR_REF_VU_GGML (TSR_WITH_LLAMA builds; links the pinned ggml-cpu):
//   ggml_rope  ggml_rope_ext on ggml-cpu, Qwen2.5 / Gemma 4 parameters (NEOX; bases 1e6 / 1e4;
//              head_dim 64 / 128 / 256 and Gemma 4's global 512 with rope_freqs), positions
//              0 .. 32767: on unit input pairs ggml's output is cosf / sinf of its angles, and
//              libm cosf / sinf of our angles reproduce it bit for bit. Full rotations on random
//              input are compared too (informative): the VU against the CPU, and the CPU against
//              libm on our angles with the rotation evaluated plainly and as the contracted
//              fma(x0, c, -(x1·s)) / fma(x0, s, x1·c) (the pinned -march=native build contracts)
//   ggml_rms   RMSNorm as RED.SUMSQ (post RMS, sk = f32(1/n), se = eps) + VOP.MULC against
//              ggml_rms_norm: rows bit-identical and the ulp difference (gate <= 8 ulp; reported)

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "test_util.h"
#include "tsr_ref.h"
#include "tsr_ref_math.h"
#include "tsr_ref_vu.h"

#ifdef TSR_REF_VU_GGML
#include "ggml-cpu.h"
#include "ggml.h"
#endif

#define CANON 0x7FC00000u
#define PINF  0x7F800000u
#define NINF  0xFF800000u
#define NZERO 0x80000000u
#define ONE   0x3F800000u

static void eqw(const char * what, uint32_t got, uint32_t want, int line) {
    ++g_checks;
    if (got != want) {
        fprintf(stderr, "FAIL line %d: %s = 0x%08x, want 0x%08x\n", line, what, got, want);
        if (++g_fail >= 40) exit(1);
    }
}
#define EQ(what, got, want) eqw(what, (got), (want), __LINE__)

static uint32_t fb(float f) { return f2u(f); }
static float bf(uint32_t u) { return u2f(u); }
static double bd(uint32_t u) { return (double)u2f(u); }

static const uint32_t k_nans[] = { 0x7FC00000u, 0xFFC00000u, 0x7F800001u, 0xFFFFFFFFu, 0x7FBFFFFFu };
#define N_NANS (sizeof k_nans / sizeof k_nans[0])

// Interesting f32 bit patterns for random operands.
static const uint32_t k_specials[] = {
    0x00000000u, 0x80000000u, 0x7F800000u, 0xFF800000u, 0x7FC00000u, 0xFFC00001u, 0x7F800001u,
    0x00000001u, 0x807FFFFFu, 0x00800000u, 0x80800000u, 0x80C00000u, 0x7F7FFFFFu, 0xFF7FFFFFu,
    0x41200000u, 0xC1200000u, 0x3F800000u, 0xBF800000u, 0x42B17218u, 0xC2CE0000u, 0x3C7FFFFFu,
};
#define N_SPECIALS (sizeof k_specials / sizeof k_specials[0])

static uint32_t rand_val(rng_t * r) {
    const uint32_t k = rng_below(r, 16);
    if (k == 0) return k_specials[rng_below(r, N_SPECIALS)];
    if (k == 1) return rng_u32(r);
    return fb((float)(rng_normal(r) * 3.0));
}

// ---------------------------------------------------------------------------
static void test_order(void) {
    const uint32_t seq[] = { NINF, 0xC0000000u, 0xBF800000u, 0x80800000u, NZERO, 0u, 0x00800000u, ONE, PINF };
    for (size_t i = 0; i + 1 < sizeof seq / sizeof seq[0]; ++i) {
        CHECK(tsr_ref_vu_key(seq[i]) < tsr_ref_vu_key(seq[i + 1]), "key order at %zu", i);
        EQ("maxt ordered", tsr_ref_vu_maxt(seq[i], seq[i + 1]), seq[i + 1]);
        EQ("maxt swapped", tsr_ref_vu_maxt(seq[i + 1], seq[i]), seq[i + 1]);
        EQ("mint ordered", tsr_ref_vu_mint(seq[i], seq[i + 1]), seq[i]);
        EQ("mint swapped", tsr_ref_vu_mint(seq[i + 1], seq[i]), seq[i]);
    }
    for (size_t i = 0; i < N_NANS; ++i) {
        EQ("maxt(NaN, 1)", tsr_ref_vu_maxt(k_nans[i], ONE), CANON);
        EQ("maxt(-inf, NaN)", tsr_ref_vu_maxt(NINF, k_nans[i]), CANON);
        EQ("mint(NaN, 1)", tsr_ref_vu_mint(k_nans[i], ONE), ONE);
        EQ("mint(+inf, NaN)", tsr_ref_vu_mint(PINF, k_nans[i]), PINF);
        EQ("mint(NaN, NaN)", tsr_ref_vu_mint(k_nans[i], k_nans[(i + 1) % N_NANS]), CANON);
        EQ("maxt(NaN, NaN)", tsr_ref_vu_maxt(k_nans[i], k_nans[i]), CANON);
    }
    // DAZ: subnormals compare as signed zeros and come out as signed zeros.
    EQ("maxt(+sub, -1)", tsr_ref_vu_maxt(0x00000005u, 0xBF800000u), 0u);
    EQ("maxt(-sub, +0)", tsr_ref_vu_maxt(0x80000005u, 0u), 0u);
    EQ("maxt(-sub, -1)", tsr_ref_vu_maxt(0x80000005u, 0xBF800000u), NZERO);
    EQ("mint(+sub, -sub)", tsr_ref_vu_mint(0x00000005u, 0x80000007u), NZERO);
    EQ("mint(NaN, -sub)", tsr_ref_vu_mint(0x7FC00000u, 0x807FFFFFu), NZERO);
    EQ("maxt(-0, -0)", tsr_ref_vu_maxt(NZERO, NZERO), NZERO);
    EQ("maxt(+0, -0)", tsr_ref_vu_maxt(0u, NZERO), 0u);
    EQ("mint(+0, -0)", tsr_ref_vu_mint(0u, NZERO), NZERO);
}

// ---------------------------------------------------------------------------
static uint32_t ve(uint32_t func, uint32_t a, uint32_t b, uint32_t c) { return tsr_ref_vop_elem(func, a, b, c); }

static void test_vop_elem(void) {
    // Raw functions keep every bit (VU-23).
    const uint32_t raws[] = { 0x7F800001u, 0xFFFFFFFFu, 0x00000001u, 0x807FFFFFu, NZERO, 0x12345678u };
    for (size_t i = 0; i < sizeof raws / sizeof raws[0]; ++i) {
        EQ("COPY raw", ve(TSR_VOP_FUNC_COPY, raws[i], 0, 0), raws[i]);
        EQ("SETC raw", ve(TSR_VOP_FUNC_SETC, 0, 0, raws[i]), raws[i]);
    }
    // fp rules: canonical NaN, DAZ, FTZ.
    EQ("ADD inf-inf", ve(TSR_VOP_FUNC_ADD, PINF, NINF, 0), CANON);
    EQ("ADD NaN payload", ve(TSR_VOP_FUNC_ADD, 0xFFC12345u, ONE, 0), CANON);
    EQ("ADD subnormals", ve(TSR_VOP_FUNC_ADD, 0x00400000u, 0x00400000u, 0), 0u);
    EQ("ADD FTZ -0", ve(TSR_VOP_FUNC_ADD, 0x80C00000u, 0x00800000u, 0), NZERO);
    EQ("SUB x-x", ve(TSR_VOP_FUNC_SUB, 0x40490FDBu, 0x40490FDBu, 0), 0u);
    EQ("MUL 0*inf", ve(TSR_VOP_FUNC_MUL, 0u, PINF, 0), CANON);
    EQ("MUL FTZ", ve(TSR_VOP_FUNC_MUL, 0x00800000u, 0x3F000000u, 0), 0u);
    EQ("MULC", ve(TSR_VOP_FUNC_MULC, fb(3.0f), 0, fb(0.5f)), fb(1.5f));
    EQ("ADDC", ve(TSR_VOP_FUNC_ADDC, fb(3.0f), 0, fb(0.5f)), fb(3.5f));
    EQ("SUBC", ve(TSR_VOP_FUNC_SUBC, fb(3.0f), 0, fb(0.5f)), fb(2.5f));
    EQ("MULCB", ve(TSR_VOP_FUNC_MULCB, fb(3.0f), fb(4.0f), fb(0.5f)), fb(6.0f));
    EQ("MULCADD", ve(TSR_VOP_FUNC_MULCADD, fb(3.0f), fb(4.0f), fb(0.5f)), fb(5.5f));
    EQ("ADDMULC", ve(TSR_VOP_FUNC_ADDMULC, fb(3.0f), fb(4.0f), fb(0.5f)), fb(3.5f));
    // MULCB rounds (a·c) first: (1 + 2^-23)·(1 + 2^-23) rounds, then ·b.
    {
        const uint32_t a = 0x3F800001u, c = 0x3F800001u, b = 0x3F7FFFFFu;
        EQ("MULCB order", ve(TSR_VOP_FUNC_MULCB, a, b, c), fb(tsr_fmul(tsr_fmul(bf(a), bf(c)), bf(b))));
        EQ("MULCADD order", ve(TSR_VOP_FUNC_MULCADD, a, b, c), fb(tsr_fadd(tsr_fmul(bf(a), bf(c)), bf(b))));
        EQ("ADDMULC order", ve(TSR_VOP_FUNC_ADDMULC, a, b, c), fb(tsr_fmul(tsr_fadd(bf(a), bf(b)), bf(c))));
    }
    // RECIP: correctly rounded, DAZ in, FTZ out.
    EQ("RECIP +0", ve(TSR_VOP_FUNC_RECIP, 0u, 0, 0), PINF);
    EQ("RECIP -0", ve(TSR_VOP_FUNC_RECIP, NZERO, 0, 0), NINF);
    EQ("RECIP +sub", ve(TSR_VOP_FUNC_RECIP, 0x00000001u, 0, 0), PINF);
    EQ("RECIP -inf", ve(TSR_VOP_FUNC_RECIP, NINF, 0, 0), NZERO);
    EQ("RECIP 2^127", ve(TSR_VOP_FUNC_RECIP, 0x7F000000u, 0, 0), 0u);
    EQ("RECIP 2^126", ve(TSR_VOP_FUNC_RECIP, 0x7E800000u, 0, 0), 0x00800000u);
    EQ("RECIP 3", ve(TSR_VOP_FUNC_RECIP, fb(3.0f), 0, 0), 0x3EAAAAABu);
    // Specials table of VU §6.5.
    EQ("EXP -inf", ve(TSR_VOP_FUNC_EXP, NINF, 0, 0), 0u);
    EQ("EXP +inf", ve(TSR_VOP_FUNC_EXP, PINF, 0, 0), PINF);
    EQ("EXP -0", ve(TSR_VOP_FUNC_EXP, NZERO, 0, 0), ONE);
    EQ("SIGMOID -inf", ve(TSR_VOP_FUNC_SIGMOID, NINF, 0, 0), 0u);
    EQ("SIGMOID +inf", ve(TSR_VOP_FUNC_SIGMOID, PINF, 0, 0), ONE);
    EQ("SILU -inf", ve(TSR_VOP_FUNC_SILU, NINF, 0, 0), CANON);
    EQ("SILU +inf", ve(TSR_VOP_FUNC_SILU, PINF, 0, 0), PINF);
    EQ("GELU -inf", ve(TSR_VOP_FUNC_GELU, NINF, 0, 0), CANON);
    EQ("GELU +inf", ve(TSR_VOP_FUNC_GELU, PINF, 0, 0), PINF);
    EQ("GELU16 -inf", ve(TSR_VOP_FUNC_GELU16, NINF, 0, 0), 0u);
    EQ("GELU16 +inf", ve(TSR_VOP_FUNC_GELU16, PINF, 0, 0), PINF);
    EQ("TANH -inf", ve(TSR_VOP_FUNC_TANH, NINF, 0, 0), 0xBF800000u);
    EQ("TANH +inf", ve(TSR_VOP_FUNC_TANH, PINF, 0, 0), ONE);
    EQ("GELU16 -10", ve(TSR_VOP_FUNC_GELU16, 0xC1200000u, 0, 0), 0u);
    EQ("GELU16 10", ve(TSR_VOP_FUNC_GELU16, 0x41200000u, 0, 0), 0x41200000u);
    EQ("GEGLU16 -10*NaN", ve(TSR_VOP_FUNC_GEGLU16, 0xC1200000u, CANON, 0), 0u);
    EQ("GEGLU16 -inf*inf", ve(TSR_VOP_FUNC_GEGLU16, NINF, PINF, 0), 0u);
    // EXPSUB's -inf rule (VU-21): +0 whatever the subtrahend, also -inf and NaN.
    const uint32_t xs[] = { NINF, PINF, CANON, 0xFFC00001u, 0u, NZERO, fb(-3.0f), 0x7F7FFFFFu };
    for (size_t i = 0; i < sizeof xs / sizeof xs[0]; ++i) {
        EQ("EXPSUB(-inf, x)", ve(TSR_VOP_FUNC_EXPSUB, NINF, xs[i], 0), 0u);
        EQ("EXPSUBC(-inf, x)", ve(TSR_VOP_FUNC_EXPSUBC, NINF, 0, xs[i]), 0u);
    }
    EQ("EXPSUB(1, -inf)", ve(TSR_VOP_FUNC_EXPSUB, ONE, NINF, 0), PINF);
    EQ("EXPSUB(NaN, 1)", ve(TSR_VOP_FUNC_EXPSUB, CANON, ONE, 0), CANON);
    EQ("EXPSUB(-sub, 0)", ve(TSR_VOP_FUNC_EXPSUB, 0x80000001u, 0u, 0), ONE);   // DAZ(a) = -0, not -inf
    EQ("EXPSUB(x, x)", ve(TSR_VOP_FUNC_EXPSUB, fb(-7.5f), fb(-7.5f), 0), ONE);
    // Every function on random operands against its VU §5.2 formula.
    rng_t r = { 0xF00Dull };
    for (int it = 0; it < 20000; ++it) {
        const uint32_t a = rand_val(&r), b = rand_val(&r), c = rand_val(&r);
        const float A = bf(a), B = bf(b), C = bf(c);
        EQ("ADD", ve(0x00, a, b, c), fb(tsr_fadd(A, B)));
        EQ("SUB", ve(0x01, a, b, c), fb(tsr_fsub(A, B)));
        EQ("MUL", ve(0x02, a, b, c), fb(tsr_fmul(A, B)));
        EQ("MAX", ve(0x03, a, b, c), tsr_ref_vu_maxt(a, b));
        EQ("MULC", ve(0x04, a, b, c), fb(tsr_fmul(A, C)));
        EQ("ADDC", ve(0x05, a, b, c), fb(tsr_fadd(A, C)));
        EQ("SILU", ve(0x06, a, b, c), fb(tsr_m_silu(A)));
        EQ("SWIGLU", ve(0x07, a, b, c), fb(tsr_fmul(tsr_m_silu(A), B)));
        EQ("EXP", ve(0x08, a, b, c), fb(tsr_m_expk(A, TSR_M_P1)));
        EQ("COPY", ve(0x09, a, b, c), a);
        EQ("MIN", ve(0x0A, a, b, c), tsr_ref_vu_mint(a, b));
        EQ("SUBC", ve(0x0B, a, b, c), fb(tsr_fsub(A, C)));
        EQ("EXPSUB", ve(0x0F, a, b, c), fb(tsr_m_expsub(A, B)));
        EQ("EXPSUBC", ve(0x10, a, b, c), fb(tsr_m_expsub(A, C)));
        EQ("SIGMOID", ve(0x11, a, b, c), fb(tsr_m_sigmoid(A)));
        EQ("GELU", ve(0x12, a, b, c), fb(tsr_m_gelu_tanh(A)));
        EQ("GEGLU", ve(0x13, a, b, c), fb(tsr_fmul(tsr_m_gelu_tanh(A), B)));
        EQ("TANH", ve(0x14, a, b, c), fb(tsr_m_tanh(A)));
        EQ("RECIP", ve(0x15, a, b, c), fb(tsr_frecip(A)));
        EQ("SETC", ve(0x16, a, b, c), c);
        EQ("GELU16", ve(0x17, a, b, c), fb(tsr_m_gelu16(A)));
        EQ("GEGLU16", ve(0x18, a, b, c), fb(tsr_m_geglu16(A, B)));
        // No fp result is subnormal or a non-canonical NaN (VU §4.1).
        for (uint32_t f = 0; f < TSR_VOP_FUNC_N; ++f) {
            if (f == TSR_VOP_FUNC_COPY || f == TSR_VOP_FUNC_SETC) continue;
            const uint32_t y = ve(f, a, b, c);
            CHECK((y & 0x7F800000u) != 0 || (y & 0x007FFFFFu) == 0, "func %u subnormal result 0x%08x", f, y);
            CHECK((y & 0x7FFFFFFFu) <= 0x7F800000u || y == CANON, "func %u NaN 0x%08x", f, y);
        }
    }
}

// ---------------------------------------------------------------------------
#define CANARY 0xA5A5A5A5u

static void test_vop_shape(void) {
    rng_t r = { 0x5EEDull };
    enum { NL = 1024 };
    uint32_t * spm = malloc(NL * 16 * sizeof *spm);
    uint32_t * ref = malloc(NL * 16 * sizeof *ref);
    const uint32_t ms[] = { 1, 5, 15, 16, 17, 33, 100, 255 };
    for (int it = 0; it < 600; ++it) {
        const uint32_t func = (uint32_t)it % TSR_VOP_FUNC_N;
        const uint32_t reads = tsr_vop_func_reads(func);
        const uint32_t m = ms[rng_below(&r, 8)];
        const uint32_t rows = 1 + rng_below(&r, 6);
        const uint32_t Lm = tsr_ref_vu_lines(m);
        const uint32_t ds = (rows > 1) ? Lm + rng_below(&r, 3) : rng_below(&r, 7);
        const uint32_t as = (reads & TSR_VOP_READS_A) ? (rng_below(&r, 3) == 0 ? 0 : Lm + rng_below(&r, 2)) : 0;
        const uint32_t bs = (reads & TSR_VOP_READS_B) ? (rng_below(&r, 3) == 0 ? 0 : Lm + rng_below(&r, 2)) : 0;
        const uint32_t cmode = (reads & TSR_VOP_READS_C) ? 1 + rng_below(&r, 2) : 0;
        const int inplace = (reads & TSR_VOP_READS_A) && rng_below(&r, 4) == 0;
        // Placement: d, then a, b, c, each after the previous extent.
        const uint32_t ed = (rows - 1) * ds + Lm;
        const uint32_t dl = 1 + rng_below(&r, 4);
        uint32_t al = dl, asx = inplace ? ds : as;
        uint32_t nxt = dl + ed + 1;
        if ((reads & TSR_VOP_READS_A) && !inplace) { al = nxt; nxt += (rows - 1) * as + Lm + 1; }
        const uint32_t bl = nxt; nxt += (rows - 1) * bs + Lm + 1;
        const uint32_t cl = nxt; nxt += tsr_ref_vu_lines(rows) + 1;
        for (uint32_t i = 0; i < NL * 16; ++i) spm[i] = rand_val(&r);
        for (uint32_t i = 0; i < 16 * (dl + ed); ++i) if (i >= 16 * dl && !inplace) spm[i] = CANARY;
        memcpy(ref, spm, NL * 16 * sizeof *ref);
        const uint32_t sreg = rand_val(&r);
        // Expected: VU §4.2 directly (element j of row r at word 16·(X + r·s) + j).
        for (uint32_t row = 0; row < rows; ++row) {
            const uint32_t c = (cmode == 1) ? sreg : (cmode == 2) ? spm[16 * cl + row] : 0;
            for (uint32_t j = 0; j < 16 * Lm; ++j) {
                const uint32_t a = spm[16 * (al + row * asx) + j];
                const uint32_t b = spm[16 * (bl + row * bs) + j];
                ref[16 * (dl + row * ds) + j] = (j < m) ? ve(func, a, b, c) : 0u;
            }
        }
        tsr_ref_vop(func, cmode, rows, m, spm + 16 * dl, ds,
                    (reads & TSR_VOP_READS_A) ? spm + 16 * al : NULL, asx,
                    (reads & TSR_VOP_READS_B) ? spm + 16 * bl : NULL, bs,
                    cmode == 1 ? &sreg : cmode == 2 ? spm + 16 * cl : NULL);
        size_t bad = 0, first = 0;
        for (size_t i = 0; i < NL * 16; ++i) {
            if (spm[i] != ref[i] && bad++ == 0) first = i;
        }
        CHECK(bad == 0, "vop func %u rows %u m %u strides %u/%u/%u cmode %u inplace %d: %zu words differ (first %zu)",
              func, rows, m, ds, asx, bs, cmode, inplace, bad, first);
    }
    // m = 0 and rows = 0 write nothing; rows = 1 ignores strides.
    for (uint32_t i = 0; i < 64; ++i) spm[i] = CANARY;
    const uint32_t s1 = ONE;
    tsr_ref_vop(TSR_VOP_FUNC_SETC, TSR_VOP_CMODE_SREG, 3, 0, spm, 1, NULL, 0, NULL, 0, &s1);
    tsr_ref_vop(TSR_VOP_FUNC_SETC, TSR_VOP_CMODE_SREG, 0, 40, spm, 3, NULL, 0, NULL, 0, &s1);
    for (uint32_t i = 0; i < 64; ++i) EQ("n = 0 writes nothing", spm[i], CANARY);
    tsr_ref_vop(TSR_VOP_FUNC_SETC, TSR_VOP_CMODE_SREG, 1, 3, spm, 0xFFFFu, NULL, 0, NULL, 0, &s1);
    EQ("rows 1 word 2", spm[2], ONE);
    EQ("rows 1 tail", spm[3], 0u);
    EQ("rows 1 next line", spm[16], CANARY);
    // Broadcast: b with stride 0 (a bias over heads), c per row from a vector.
    for (uint32_t i = 0; i < 16 * 16; ++i) spm[i] = 0;
    for (uint32_t j = 0; j < 20; ++j) { spm[16 * 0 + j] = fb((float)j); spm[16 * 2 + j] = fb((float)j); }
    for (uint32_t j = 0; j < 20; ++j) spm[16 * 4 + j] = fb(100.0f);
    const uint32_t cv[3] = { fb(1.0f), fb(2.0f), fb(3.0f) };
    memcpy(spm + 16 * 6, cv, sizeof cv);
    // rows 2: a rows at lines 0 and 2 (stride 2), b = line 4 stride 0, c vector line 6: MULCADD.
    tsr_ref_vop(TSR_VOP_FUNC_MULCADD, TSR_VOP_CMODE_VEC, 2, 20, spm + 16 * 8, 2, spm, 2, spm + 16 * 4, 0, spm + 16 * 6);
    for (uint32_t j = 0; j < 20; ++j) {
        EQ("broadcast row 0", spm[16 * 8 + j], fb((float)j * 1.0f + 100.0f));
        EQ("broadcast row 1", spm[16 * 10 + j], fb((float)j * 2.0f + 100.0f));
    }
    for (uint32_t j = 20; j < 32; ++j) EQ("tail rows", spm[16 * 8 + j] | spm[16 * 10 + j], 0u);
    free(spm);
    free(ref);
}

// ---------------------------------------------------------------------------
// The skip implementation without the z() rule (what VU §6.2 forbids), to show the directed cases
// are sensitive: a skipped pairing passes q unchanged.
static uint32_t tree_skip_naive(const uint32_t p[128], uint32_t n) {
    uint32_t v[128];
    uint8_t pr[128];
    for (uint32_t i = 0; i < 128; ++i) { v[i] = p[i]; pr[i] = (uint8_t)(i < n); }
    for (uint32_t w = 128; w > 1; w >>= 1) {
        for (uint32_t k = 0; k < w / 2; ++k) {
            const uint8_t pl = pr[2 * k], pq = pr[2 * k + 1];
            v[k] = (pl && pq) ? fb(tsr_fadd(bf(v[2 * k]), bf(v[2 * k + 1]))) : pl ? v[2 * k] : pq ? v[2 * k + 1] : 0u;
            pr[k] = pl | pq;
        }
    }
    return pr[0] ? v[0] : 0u;
}

static void test_red(void) {
    uint32_t x[1024], p[128];
    // VU §6.2's example: FA(0x80C00000, 0x00800000) = -0 and FA(-0, +0) = +0.
    EQ("FA -0 pair", fb(tsr_fadd(bf(0x80C00000u), bf(0x00800000u))), NZERO);
    EQ("FA(-0,+0)", fb(tsr_fadd(bf(NZERO), 0.0f)), 0u);
    // -0 subtrees at m = 8k + r: every level-1 pair (or a random subset) is (0x80C00000, 0x00800000).
    rng_t r = { 0xBEEFull };
    int naive_differs = 0;
    for (uint32_t k = 0; k <= 16; ++k) {
        for (uint32_t rr = 0; rr < 8; ++rr) {
            const uint32_t m = 8 * k + rr;
            if (m > 128) continue;
            for (int variant = 0; variant < 4; ++variant) {
                for (uint32_t i = 0; i < m; ++i) x[i] = 0u;
                for (uint32_t i = 0; i + 1 < m; i += 2) {
                    if (variant == 0 || rng_below(&r, 2)) { x[i] = 0x80C00000u; x[i + 1] = 0x00800000u; }
                }
                if (variant == 3 && m) x[m - 1] = 0x80000000u;      // -0 element: partial FA(+0, -0) = +0
                tsr_ref_red_partials(TSR_RED_KIND_SUM, x, m, p);
                const uint32_t full = tsr_ref_red_tree128(p);
                EQ("row = tree", tsr_ref_red_row(TSR_RED_KIND_SUM, TSR_RED_POST_NONE, x, m, 0, 0), full);
                EQ("skip g1", tsr_ref_red_tree_skip(p, m, 1), full);
                EQ("skip g8", tsr_ref_red_tree_skip(p, m, 8), full);
                if (m < 128 && variant < 3) {
                    // Only -0 pairs and +0: the complete tree is +0 (a -0 meets an unwritten +0).
                    EQ("short -0 rows sum to +0", full, 0u);
                }
                if (variant == 0 && m >= 2 && m < 128 && tree_skip_naive(p, m) == NZERO) naive_differs = 1;
            }
        }
    }
    CHECK(naive_differs, "the naive skip (no z()) never differed: directed cases are not sensitive");
    // A partial starts as FA(+0, x), never a raw load (VU §6.2): rows of -0 and negative
    // subnormals give +0 for every m (a raw first load would leave -0 partials and, for m >= 128,
    // a -0 result).
    {
        const uint32_t negs[] = { NZERO, 0x80000001u, 0x807FFFFFu };
        const uint32_t lens[] = { 1, 2, 127, 128, 129, 256, 300 };
        for (size_t li = 0; li < sizeof lens / sizeof lens[0]; ++li) {
            for (uint32_t i = 0; i < lens[li]; ++i) x[i] = negs[rng_below(&r, 3)];
            EQ("SUM of -0 row", tsr_ref_red_row(TSR_RED_KIND_SUM, 0, x, lens[li], 0, 0), 0u);
            EQ("SUMSQ of -0 row", tsr_ref_red_row(TSR_RED_KIND_SUMSQ, 0, x, lens[li], 0, 0), 0u);
            EQ("MAX of -0 row", tsr_ref_red_row(TSR_RED_KIND_MAX, 0, x, lens[li], 0, 0), NZERO);
            tsr_ref_red_partials(TSR_RED_KIND_SUM, x, lens[li], p);
            for (uint32_t i = 0; i < 128; ++i) EQ("-0 row partial", p[i], 0u);
        }
    }
    // m = 128, every pair -0: all 7 levels are FA(-0, -0) = -0.
    for (uint32_t i = 0; i < 128; i += 2) { x[i] = 0x80C00000u; x[i + 1] = 0x00800000u; }
    EQ("m 128 all -0 pairs", tsr_ref_red_row(TSR_RED_KIND_SUM, 0, x, 128, 0, 0), NZERO);
    tsr_ref_red_partials(TSR_RED_KIND_SUM, x, 128, p);
    EQ("m 128 skip g8", tsr_ref_red_tree_skip(p, 128, 8), NZERO);
    // Random rows of every length up to 300 with -0 pairs mixed in: skip models = tree.
    for (int it = 0; it < 3000; ++it) {
        const uint32_t m = rng_below(&r, 300);
        for (uint32_t i = 0; i < m; ++i) {
            x[i] = (rng_below(&r, 4) == 0) ? rand_val(&r) : (i & 1) ? 0x00800000u : 0x80C00000u;
        }
        const uint32_t kind = rng_below(&r, 2);
        tsr_ref_red_partials(kind, x, m, p);
        const uint32_t full = tsr_ref_red_tree128(p);
        EQ("random skip g1", tsr_ref_red_tree_skip(p, m, 1), full);
        EQ("random skip g8", tsr_ref_red_tree_skip(p, m, 8), full);
        // Partial i holds elements i, i + 128, ... in order (VU §6.2).
        uint32_t q[128] = { 0 };
        for (uint32_t i = 0; i < m; ++i) {
            const float v = kind ? tsr_fmul(bf(x[i]), bf(x[i])) : bf(x[i]);
            q[i % 128] = fb(tsr_fadd(bf(q[i % 128]), v));
        }
        CHECK(memcmp(p, q, sizeof q) == 0, "partials m %u kind %u", m, kind);
    }
    // Tree pairing: level 1 pairs adjacent partials; a value only representable by that order.
    for (uint32_t i = 0; i < 128; ++i) p[i] = 0u;
    p[0] = fb(1.0f); p[1] = fb(0x1p-24f); p[2] = fb(0x1p-24f);   // (1 + 2^-24) rounds to 1, then + 2^-24 -> 1
    EQ("tree order", tsr_ref_red_tree128(p), fb(1.0f));
    p[1] = 0; p[3] = fb(0x1p-24f);                               // (2^-24 + 2^-24) first, then 1 + 2^-23
    EQ("tree order 2", tsr_ref_red_tree128(p), fb(1.0f + 0x1p-23f));
    // Empty rows and post-ops.
    EQ("empty SUM", tsr_ref_red_row(TSR_RED_KIND_SUM, 0, x, 0, 0, 0), 0u);
    EQ("empty SUMSQ", tsr_ref_red_row(TSR_RED_KIND_SUMSQ, 0, x, 0, 0, 0), 0u);
    EQ("empty MAX", tsr_ref_red_row(TSR_RED_KIND_MAX, 0, x, 0, 0, 0), NINF);
    EQ("empty SUM RCP", tsr_ref_red_row(TSR_RED_KIND_SUM, TSR_RED_POST_RCP, x, 0, 0, 0), PINF);
    EQ("empty SUMSQ RMS eps 1", tsr_ref_red_row(TSR_RED_KIND_SUMSQ, TSR_RED_POST_RMS, x, 0, fb(0.25f), ONE), ONE);
    // MAX: DAZ'd, -0 < +0, NaN anywhere -> canonical.
    const uint32_t mx[] = { NZERO, 0x80000001u, NINF, 0x00000003u };
    EQ("MAX of zeros", tsr_ref_red_row(TSR_RED_KIND_MAX, 0, mx, 4, 0, 0), 0u);
    EQ("MAX of -0 and -inf", tsr_ref_red_row(TSR_RED_KIND_MAX, 0, mx, 3, 0, 0), NZERO);
    const uint32_t mn[] = { ONE, PINF, 0xFFC00123u, 0x7F7FFFFFu };
    EQ("MAX NaN", tsr_ref_red_row(TSR_RED_KIND_MAX, 0, mn, 4, 0, 0), CANON);
    EQ("MAX inf", tsr_ref_red_row(TSR_RED_KIND_MAX, 0, mn, 2, 0, 0), PINF);
    // SUMSQ squares round first: x = 1 + 2^-12, x² = 1 + 2^-11 + 2^-24 -> 1 + 2^-11 (tie to even).
    const uint32_t sq1[] = { fb(1.0f + 0x1p-12f) };
    EQ("SUMSQ rounding", tsr_ref_red_row(TSR_RED_KIND_SUMSQ, 0, sq1, 1, 0, 0), fb(1.0f + 0x1p-11f));
    // RMS = the spec's recipe SOP.MULI -> ADDI -> SQRT -> RECIP (VU-3), on random sums.
    for (int it = 0; it < 20000; ++it) {
        const uint32_t y = rand_val(&r), sk = rand_val(&r), se = rand_val(&r);
        uint32_t s = tsr_ref_sop(TSR_SOP_SUB_MULI, y, 0, sk, 0);
        s = tsr_ref_sop(TSR_SOP_SUB_ADDI, s, 0, se, 0);
        s = tsr_ref_sop(TSR_SOP_SUB_SQRT, s, 0, 0, 0);
        s = tsr_ref_sop(TSR_SOP_SUB_RECIP, s, 0, 0, 0);
        EQ("RMS = SOP recipe", tsr_ref_red_post(TSR_RED_POST_RMS, y, sk, se), s);
        EQ("RCP", tsr_ref_red_post(TSR_RED_POST_RCP, y, sk, se), fb(tsr_frecip(bf(y))));
    }
    // Vector destination: rows words + zero tail; strides; m = 0 writes nothing.
    {
        enum { NL = 256 };
        uint32_t * spm = malloc(NL * 16 * sizeof *spm);
        for (uint32_t i = 0; i < NL * 16; ++i) spm[i] = CANARY;
        const uint32_t rows = 19, m = 40, as = 4;
        for (uint32_t row = 0; row < rows; ++row) {
            for (uint32_t j = 0; j < m; ++j) spm[16 * (10 + row * as) + j] = fb((float)(row + j));
        }
        tsr_ref_red(TSR_RED_KIND_SUM, TSR_RED_POST_NONE, rows, m, spm, spm + 160, as, 0, 0);
        for (uint32_t row = 0; row < rows; ++row) {
            EQ("RED vector", spm[row], fb((float)(row * m + m * (m - 1) / 2)));
        }
        for (uint32_t j = rows; j < 32; ++j) EQ("RED vector tail", spm[j], 0u);
        EQ("RED vector extent", spm[32], CANARY);
        // a stride 0: every row reads row 0.
        tsr_ref_red(TSR_RED_KIND_MAX, TSR_RED_POST_NONE, 3, m, spm + 16 * 3, spm + 160, 0, 0, 0);
        for (uint32_t row = 0; row < 3; ++row) EQ("RED stride 0", spm[48 + row], fb((float)(m - 1)));
        for (uint32_t i = 0; i < 16; ++i) spm[16 * 5 + i] = CANARY;
        tsr_ref_red(TSR_RED_KIND_SUM, TSR_RED_POST_RCP, 7, 0, spm + 16 * 5, spm + 160, 4, 0, 0);
        tsr_ref_red(TSR_RED_KIND_SUM, TSR_RED_POST_RCP, 0, 9, spm + 16 * 5, spm + 160, 4, 0, 0);
        for (uint32_t i = 0; i < 16; ++i) EQ("RED n = 0 writes nothing", spm[16 * 5 + i], CANARY);
        free(spm);
    }
    // SUM accuracy against double (informative): 128 partials of n/128 terms.
    {
        double worst = 0;
        for (int it = 0; it < 200; ++it) {
            const uint32_t m = 1 + rng_below(&r, 1024);
            double ex = 0, ab = 0;
            for (uint32_t i = 0; i < m; ++i) { x[i] = fb((float)rng_normal(&r)); ex += bd(x[i]); ab += fabs(bd(x[i])); }
            const double e = fabs(bd(tsr_ref_red_row(TSR_RED_KIND_SUM, 0, x, m, 0, 0)) - ex) / ab;
            if (e > worst) worst = e;
        }
        printf("  red: SUM relative error (to sum|x|) max %.3g over m <= 1024 (bound ~ (m/128 + 7)·2^-24)\n", worst);
        CHECK(worst < 16 * 0x1p-24, "SUM error %g", worst);
    }
}

// ---------------------------------------------------------------------------
// ggml_rope_cache_init's angles (ops.cpp:6287-6301, rope_yarn :6270-6285) for ext_factor = 0,
// transcribed: theta_extrap = theta/ff, theta = freq_scale·theta_extrap, recurrence theta *= TS.
// Native f32 (this file is built with -ffp-contract=off; single operations cannot contract).
static void ggml_angles(int32_t p, float freq_scale, const float * ff, int n_dims, float theta_scale, float * out) {
    float theta = (float)(int64_t)p;
    for (int i0 = 0; i0 < n_dims; i0 += 2) {
        const float f = ff ? ff[i0 / 2] : 1.0f;
        const float theta_extrap = theta / f;
        out[i0 / 2] = freq_scale * theta_extrap;
        theta *= theta_scale;
    }
}

typedef struct {
    const char * name;
    float base;
    uint32_t head_dim, n_dims, mode, ff_kind;   // ff_kind: 0 none, 1 Gemma 4 proportional, 2 ramp
    float freq_scale;
} rope_model;

static const rope_model k_rope_models[] = {
    { "Qwen2.5-0.5B (NEOX 64, base 1e6)", 1e6f, 64, 64, TSR_ROPE_MODE_NEOX, 0, 1.0f },
    { "Qwen2.5-32B (NEOX 128, base 1e6)", 1e6f, 128, 128, TSR_ROPE_MODE_NEOX, 0, 1.0f },
    { "Gemma 4 SWA (NEOX 256, base 1e4)", 1e4f, 256, 256, TSR_ROPE_MODE_NEOX, 0, 1.0f },
    { "Gemma 4 global (NEOX 512, base 1e6, rope_freqs 1e30)", 1e6f, 512, 512, TSR_ROPE_MODE_NEOX, 1, 1.0f },
    { "linear scaling (NEOX 128, base 1e6, freq_scale 0.25)", 1e6f, 128, 128, TSR_ROPE_MODE_NEOX, 0, 0.25f },
    { "freq factors 1..8 (NORM 128, base 5e5)", 5e5f, 128, 128, TSR_ROPE_MODE_NORM, 2, 1.0f },
};
#define N_ROPE_MODELS (sizeof k_rope_models / sizeof k_rope_models[0])

static void rope_freqs(const rope_model * mdl, float * ff) {
    const uint32_t half = mdl->n_dims / 2;
    for (uint32_t i = 0; i < half; ++i) {
        if (mdl->ff_kind == 1) {
            ff[i] = (i < half / 4) ? 1.0f : 1e30f;    // conversion/gemma.py: partial_rotary_factor 0.25
        } else {
            ff[i] = 1.0f + 7.0f * (float)i / (float)half;
        }
    }
}

static tsr_ref_rope_cfg rope_cfg_of(const rope_model * mdl, uint32_t n_heads, uint32_t T) {
    tsr_ref_rope_cfg cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.mode = mdl->mode;
    cfg.ff = mdl->ff_kind != 0;
    cfg.fs = mdl->freq_scale != 1.0f;
    cfg.m = ONE;
    cfg.ts = fb(powf(mdl->base, -2.0f / (float)mdl->n_dims));   // the host's TS (VU §6.7)
    cfg.freq_scale = fb(mdl->freq_scale);
    cfg.head_dim = mdl->head_dim;
    cfg.n_dims = mdl->n_dims;
    cfg.n_heads = n_heads;
    cfg.T = T;
    return cfg;
}

static void test_rope(void) {
    rng_t r = { 0x0123ull };
    static uint32_t phi[512], c[512], s[512], tab[512];
    static float ffv[512], ang[512];
    // 1. Angles: bit-identical to the transcription of ggml_rope_cache_init, positions 0 .. 2^17-1.
    for (size_t mi = 0; mi < N_ROPE_MODELS; ++mi) {
        const rope_model * mdl = &k_rope_models[mi];
        tsr_ref_rope_cfg cfg = rope_cfg_of(mdl, 1, 1);
        if (mdl->ff_kind) rope_freqs(mdl, ffv);
        for (uint32_t i = 0; i < mdl->n_dims / 2; ++i) tab[i] = fb(ffv[i]);
        const float ts = powf(mdl->base, -2.0f / (float)mdl->n_dims);
        long diff = 0;
        for (int32_t p = 0; p < (1 << 17); ++p) {
            tsr_ref_rope_angles(&cfg, p, tab, phi);
            ggml_angles(p, mdl->freq_scale, mdl->ff_kind ? ffv : NULL, (int)mdl->n_dims, ts, ang);
            for (uint32_t i = 0; i < mdl->n_dims / 2; ++i) diff += phi[i] != fb(ang[i]);
        }
        CHECK(diff == 0, "%s: %ld angles differ from ggml_rope_cache_init", mdl->name, diff);
        printf("  rope angles: %-55s positions 0..2^17-1: %s\n", mdl->name, diff ? "DIFFER" : "bit-identical");
    }
    // A-table form: phi_i = FM(Pf, A_i).
    {
        tsr_ref_rope_cfg cfg = rope_cfg_of(&k_rope_models[1], 1, 1);
        cfg.atab = 1;
        cfg.ts = 0;
        for (uint32_t i = 0; i < 64; ++i) tab[i] = fb((float)pow(1e6, -2.0 * i / 128.0) * 0.7f);
        tsr_ref_rope_angles(&cfg, -77777, tab, phi);
        for (uint32_t i = 0; i < 64; ++i) EQ("ATAB angle", phi[i], fb(tsr_fmul(tsr_i32_to_f32(-77777), bf(tab[i]))));
    }
    // 2. Position 0: phi = 0, c = M, s = 0, the rotation is exact (M = 1) and the tail is raw.
    enum { NL = 512 };
    uint32_t * spm = malloc(NL * 16 * sizeof *spm);
    uint32_t * ref = malloc(NL * 16 * sizeof *ref);
    {
        tsr_ref_rope_cfg cfg = rope_cfg_of(&k_rope_models[0], 2, 1);
        cfg.n_dims = 32;                            // NEOX, head_dim 64: tail [32, 64)
        for (uint32_t i = 0; i < NL * 16; ++i) spm[i] = fb((float)rng_normal(&r));
        spm[16 * 100] = 0;                          // pos[0] = 0
        spm[16 * 3 + 40] = 0x7F800001u;             // NaN payload in head 0's tail
        spm[16 * 3 + 64 + 63] = 0x00000001u;        // subnormal in head 1's tail
        memcpy(ref, spm, NL * 16 * sizeof *ref);
        tsr_ref_rope(&cfg, spm + 16 * 20, 0, spm + 16 * 3, 0, spm + 16 * 100, NULL);
        for (uint32_t j = 0; j < 128; ++j) EQ("rope pos 0", spm[16 * 20 + j], ref[16 * 3 + j]);
    }
    // 3. NORM vs NEOX pairing, in place = out of place, strides, T = 0 / n_heads = 0.
    for (int it = 0; it < 200; ++it) {
        const rope_model * mdl = &k_rope_models[rng_below(&r, 3)];
        tsr_ref_rope_cfg cfg = rope_cfg_of(mdl, 1 + rng_below(&r, 2), 1 + rng_below(&r, 3));
        cfg.mode = rng_below(&r, 2);
        cfg.n_dims = (cfg.mode == TSR_ROPE_MODE_NEOX) ? 32 * (1 + rng_below(&r, mdl->head_dim / 32)) : 16 * (1 + rng_below(&r, mdl->head_dim / 16));
        cfg.m = fb(0.5f + (float)rng_unit(&r));
        const uint32_t H = cfg.n_heads * cfg.head_dim / 16;
        const uint32_t ss = rng_below(&r, 2) ? 0 : H + rng_below(&r, 3);
        const uint32_t ssx = ss ? ss : H;
        for (uint32_t i = 0; i < NL * 16; ++i) spm[i] = fb((float)rng_normal(&r));
        for (uint32_t t = 0; t < cfg.T; ++t) spm[16 * 500 + t] = rng_u32(&r) >> 14;   // positions < 2^18
        memcpy(ref, spm, NL * 16 * sizeof *ref);
        tsr_ref_rope(&cfg, ref + 16 * 250, 0, ref + 16 * 1, ss, ref + 16 * 500, NULL);   // out of place
        tsr_ref_rope(&cfg, spm + 16 * 1, ss, spm + 16 * 1, ss, spm + 16 * 500, NULL);    // in place
        for (uint32_t t = 0; t < cfg.T; ++t) {
            int32_t pos = (int32_t)spm[16 * 500 + t];
            tsr_ref_rope_cs(&cfg, pos, NULL, c, s);
            for (uint32_t h = 0; h < cfg.n_heads; ++h) {
                const uint32_t * x = ref + 16 * (1 + t * ssx) + h * cfg.head_dim;   // ref keeps the source
                const uint32_t * yo = ref + 16 * (250 + t * H) + h * cfg.head_dim;
                const uint32_t * yi = spm + 16 * (1 + t * ssx) + h * cfg.head_dim;
                for (uint32_t i = 0; i < cfg.n_dims / 2; ++i) {
                    const uint32_t j0 = cfg.mode ? i : 2 * i, j1 = cfg.mode ? i + cfg.n_dims / 2 : 2 * i + 1;
                    const float x0 = bf(x[j0]), x1 = bf(x[j1]), cc = bf(c[i]), sn = bf(s[i]);
                    EQ("rope y0", yo[j0], fb(tsr_fsub(tsr_fmul(x0, cc), tsr_fmul(x1, sn))));
                    EQ("rope y1", yo[j1], fb(tsr_fadd(tsr_fmul(x0, sn), tsr_fmul(x1, cc))));
                }
                for (uint32_t j = cfg.n_dims; j < cfg.head_dim; ++j) EQ("rope tail", yo[j], x[j]);
                for (uint32_t j = 0; j < cfg.head_dim; ++j) EQ("rope in place", yi[j], yo[j]);
            }
        }
    }
    {
        tsr_ref_rope_cfg cfg = rope_cfg_of(&k_rope_models[0], 0, 4);
        for (uint32_t i = 0; i < 64; ++i) spm[i] = CANARY;
        tsr_ref_rope(&cfg, spm, 0, spm + 64, 0, spm + 64, NULL);
        cfg.n_heads = 3; cfg.T = 0;
        tsr_ref_rope(&cfg, spm, 0, spm + 64, 0, spm + 64, NULL);
        for (uint32_t i = 0; i < 64; ++i) EQ("rope T' = 0 writes nothing", spm[i], CANARY);
    }
    // 4. Accuracy of the rotation against double, given the f32 angles (VU §6.6 + 5 roundings),
    //    and the recurrence's angle error against exact p·base^(-2i/n) (informative; the CPU's).
    for (size_t mi = 0; mi < 4; ++mi) {
        const rope_model * mdl = &k_rope_models[mi];
        tsr_ref_rope_cfg cfg = rope_cfg_of(mdl, 1, 1);
        cfg.m = fb(1.25f);
        if (mdl->ff_kind) rope_freqs(mdl, ffv);
        for (uint32_t i = 0; i < mdl->n_dims / 2; ++i) tab[i] = fb(ffv[i]);
        double worst = 0, worst_ang = 0;
        for (int it = 0; it < 400; ++it) {
            const int32_t p = (int32_t)rng_below(&r, 32768);
            for (uint32_t j = 0; j < mdl->head_dim; ++j) spm[j] = fb((float)rng_normal(&r));
            spm[16 * 100] = (uint32_t)p;
            tsr_ref_rope(&cfg, spm + 16 * 200, 0, spm, 0, spm + 16 * 100, tab);
            tsr_ref_rope_angles(&cfg, p, tab, phi);
            const uint32_t half = mdl->n_dims / 2;
            for (uint32_t i = 0; i < half; ++i) {
                const double th = bf(phi[i]), M = 1.25;
                const double x0 = bf(spm[i]), x1 = bf(spm[i + half]);
                const double y0 = x0 * M * cos(th) - x1 * M * sin(th), y1 = x0 * M * sin(th) + x1 * M * cos(th);
                const double sc = (fabs(x0) + fabs(x1)) * M;
                const double e0 = fabs(bd(spm[16 * 200 + i]) - y0) / sc, e1 = fabs(bd(spm[16 * 200 + i + half]) - y1) / sc;
                if (e0 > worst) worst = e0;
                if (e1 > worst) worst = e1;
                const double exact = (double)p * pow((double)mdl->base, -2.0 * i / mdl->n_dims) / (mdl->ff_kind ? (double)ffv[i] : 1.0);
                const double ea = fabs(th - exact);
                if (ea > worst_ang) worst_ang = ea;
            }
        }
        printf("  rope accuracy: %-55s rotation error <= %.3g·(|x0|+|x1|)·M; recurrence angle error <= %.3g rad (p < 32768)\n",
               mdl->name, worst, worst_ang);
        CHECK(worst <= 3e-7, "%s rotation error %g", mdl->name, worst);
    }
    free(spm);
    free(ref);
}

// ---------------------------------------------------------------------------
typedef struct { uint32_t v, j; } cand;

static int cand_cmp(const void * pa, const void * pb) {
    const cand * a = pa, * b = pb;
    const int na = a->v == CANON, nb = b->v == CANON;
    if (na != nb) return na ? -1 : 1;
    if (!na) {
        const uint32_t ka = tsr_ref_vu_key(a->v), kb = tsr_ref_vu_key(b->v);
        if (ka != kb) return ka > kb ? -1 : 1;
    }
    return a->j < b->j ? -1 : a->j > b->j;
}

static void test_topk(void) {
    uint32_t d[128], x[400];
    // Ties by index, NaN first (canonical), -0 below +0, DAZ, padding, tail.
    const uint32_t a[] = { ONE, 0x7FC00001u, ONE, NZERO, 0x00000001u, 0xFFFFFFFFu, 0x80000003u, PINF };
    for (uint32_t i = 0; i < 64; ++i) d[i] = CANARY;
    tsr_ref_topk(10, 1, 8, d, 0, a, 0);
    const uint32_t want[20] = { 1, CANON, 5, CANON, 7, PINF, 0, ONE, 2, ONE, 4, 0u, 3, NZERO, 6, NZERO,
                                0xFFFFFFFFu, NINF, 0xFFFFFFFFu, NINF };
    for (uint32_t i = 0; i < 20; ++i) EQ("topk directed", d[i], want[i]);
    for (uint32_t i = 20; i < 32; ++i) EQ("topk tail", d[i], 0u);
    EQ("topk extent", d[32], CANARY);
    // k = 16 fills two lines exactly; d_stride 0 = ceil(k/8) lines; m = 0 writes nothing.
    for (uint32_t i = 0; i < 64; ++i) d[i] = CANARY;
    tsr_ref_topk(3, 4, 0, d, 0, a, 0);
    for (uint32_t i = 0; i < 64; ++i) EQ("topk m = 0", d[i], CANARY);
    // Random rows against a full sort.
    rng_t r = { 0x70Bull };
    for (int it = 0; it < 2000; ++it) {
        const uint32_t k = 1 + rng_below(&r, 16), m = 1 + rng_below(&r, 400), rows = 1 + rng_below(&r, 2);
        const uint32_t nv = 1 + rng_below(&r, 6);
        for (uint32_t i = 0; i < m; ++i) x[i] = (it & 1) ? rand_val(&r) : k_specials[rng_below(&r, nv)];
        const uint32_t dsx = (k + 7) / 8 + (rows > 1 ? rng_below(&r, 2) : 0);
        for (uint32_t i = 0; i < 128; ++i) d[i] = CANARY;
        tsr_ref_topk(k, rows, m, d, dsx == (k + 7) / 8 ? 0 : dsx, x, 0);
        cand cs[400];
        for (uint32_t i = 0; i < m; ++i) {
            const uint32_t v = ((x[i] & 0x7FFFFFFFu) > 0x7F800000u) ? CANON : ((x[i] & 0x7F800000u) ? x[i] : (x[i] & NZERO));
            cs[i].v = v; cs[i].j = i;
        }
        qsort(cs, m, sizeof cs[0], cand_cmp);
        for (uint32_t row = 0; row < rows; ++row) {
            const uint32_t * y = d + 16 * row * dsx;
            for (uint32_t i = 0; i < k; ++i) {
                EQ("topk index", y[2 * i], i < m ? cs[i].j : 0xFFFFFFFFu);
                EQ("topk value", y[2 * i + 1], i < m ? cs[i].v : NINF);
            }
            for (uint32_t i = 2 * k; i < 16 * ((k + 7) / 8); ++i) EQ("topk tail", y[i], 0u);
        }
    }
}

// ---------------------------------------------------------------------------
static void test_sop(void) {
    rng_t r = { 0x50Bull };
    for (int it = 0; it < 20000; ++it) {
        const uint32_t sa = rand_val(&r), sb = rand_val(&r), imm = rand_val(&r), mem = rng_u32(&r);
        const float A = bf(sa), B = bf(sb), I = bf(imm);
        EQ("MOVI", tsr_ref_sop(TSR_SOP_SUB_MOVI, sa, sb, imm, mem), imm);
        EQ("ADD", tsr_ref_sop(TSR_SOP_SUB_ADD, sa, sb, imm, mem), fb(tsr_fadd(A, B)));
        EQ("MUL", tsr_ref_sop(TSR_SOP_SUB_MUL, sa, sb, imm, mem), fb(tsr_fmul(A, B)));
        EQ("ADDI", tsr_ref_sop(TSR_SOP_SUB_ADDI, sa, sb, imm, mem), fb(tsr_fadd(A, I)));
        EQ("MULI", tsr_ref_sop(TSR_SOP_SUB_MULI, sa, sb, imm, mem), fb(tsr_fmul(A, I)));
        EQ("SQRT", tsr_ref_sop(TSR_SOP_SUB_SQRT, sa, sb, imm, mem), fb(tsr_fsqrt(A)));
        EQ("RECIP", tsr_ref_sop(TSR_SOP_SUB_RECIP, sa, sb, imm, mem), fb(tsr_frecip(A)));
        EQ("LDS", tsr_ref_sop(TSR_SOP_SUB_LDS, sa, sb, imm, mem), mem);
        EQ("STS", tsr_ref_sop(TSR_SOP_SUB_STS, sa, sb, imm, mem), sa);
        EQ("SUB", tsr_ref_sop(TSR_SOP_SUB_SUB, sa, sb, imm, mem), fb(tsr_fsub(A, B)));
        EQ("DIV", tsr_ref_sop(TSR_SOP_SUB_DIV, sa, sb, imm, mem), fb(tsr_fdiv(A, B)));
        EQ("RSQRT", tsr_ref_sop(TSR_SOP_SUB_RSQRT, sa, sb, imm, mem), fb(tsr_frecip(tsr_fsqrt(A))));
        EQ("MAX", tsr_ref_sop(TSR_SOP_SUB_MAX, sa, sb, imm, mem), tsr_ref_vu_maxt(sa, sb));
        EQ("MIN", tsr_ref_sop(TSR_SOP_SUB_MIN, sa, sb, imm, mem), tsr_ref_vu_mint(sa, sb));
        EQ("EXP", tsr_ref_sop(TSR_SOP_SUB_EXP, sa, sb, imm, mem), fb(tsr_m_exp(A)));
    }
    // RSQRT rounds twice: 1/sqrt(x) correctly rounded differs from RC(SQ(x)) for some x.
    int two = 0;
    for (uint32_t u = 0x3F800000u; u < 0x3F800000u + 200000u && !two; ++u) {
        const double ex = 1.0 / sqrt((double)bf(u));
        two = tsr_ref_sop(TSR_SOP_SUB_RSQRT, u, 0, 0, 0) != fb((float)ex);
    }
    CHECK(two, "RSQRT never differed from a correctly rounded 1/sqrt");
    EQ("SQRT -1", tsr_ref_sop(TSR_SOP_SUB_SQRT, 0xBF800000u, 0, 0, 0), CANON);
    EQ("SQRT -0", tsr_ref_sop(TSR_SOP_SUB_SQRT, NZERO, 0, 0, 0), NZERO);
    EQ("RSQRT -0", tsr_ref_sop(TSR_SOP_SUB_RSQRT, NZERO, 0, 0, 0), NINF);
    EQ("MOVI NaN payload raw", tsr_ref_sop(TSR_SOP_SUB_MOVI, 0, 0, 0x7F800001u, 0), 0x7F800001u);
    EQ("DIV 1/0", tsr_ref_sop(TSR_SOP_SUB_DIV, ONE, 0, 0, 0), PINF);
}

// ---------------------------------------------------------------------------
static void test_dqrow(void) {
    rng_t r = { 0xD0ull };
    for (uint32_t fmt = 0; fmt < 2; ++fmt) {
        for (int it = 0; it < 30; ++it) {
            const uint32_t K = tsr_fmt_block_size(fmt) * (1 + rng_below(&r, fmt ? 6 : 40));
            const uint32_t rows = 1 + rng_below(&r, 4);
            const uint32_t rb = tsr_row_beats(fmt, K), Lk = tsr_ref_vu_lines(K);
            const uint32_t ss = rng_below(&r, 2) ? 0 : rb + rng_below(&r, 3);
            const uint32_t ds = rng_below(&r, 2) ? 0 : Lk + rng_below(&r, 3);
            const uint32_t ssx = ss ? ss : rb, dsx = ds ? ds : Lk;
            const size_t nb = K / tsr_fmt_block_size(fmt), bb = tsr_fmt_ggml_block_bytes(fmt);
            uint8_t * src = calloc((size_t)rows * ssx, 64);
            uint8_t * blocks = malloc(nb * bb);
            for (uint32_t row = 0; row < rows; ++row) {
                rng_bytes(&r, blocks, nb * bb);
                for (size_t b = 0; b < nb; ++b) {        // finite scales: fp16 d (and dmin) in [2^-10, 1)
                    uint8_t * blk = blocks + b * bb;
                    blk[1] = (uint8_t)(0x14 + rng_below(&r, 0x18));
                    if (fmt == TSR_FMT_Q4_K) blk[3] = (uint8_t)(0x14 + rng_below(&r, 0x18));
                }
                tsr_repack_row(fmt, blocks, K, src + (size_t)row * ssx * 64);
            }
            const size_t nd = (size_t)(rows - 1) * dsx * 16 + 16 * Lk + 16;
            uint32_t * d = malloc(nd * sizeof *d);
            float * want = malloc(K * sizeof *want);
            for (size_t i = 0; i < nd; ++i) d[i] = CANARY;
            tsr_ref_dqrow(fmt, K, rows, d, ds, src, ss);
            for (uint32_t row = 0; row < rows; ++row) {
                tsr_dequant_row(fmt, src + (size_t)row * ssx * 64, K, want);
                const uint32_t * y = d + (size_t)row * dsx * 16;
                long bad = 0;
                for (uint32_t j = 0; j < K; ++j) bad += y[j] != fb(want[j]);
                CHECK(bad == 0, "dqrow fmt %u K %u row %u: %ld differ", fmt, K, row, bad);
            }
            EQ("dqrow extent", d[(size_t)(rows - 1) * dsx * 16 + 16 * Lk], CANARY);
            free(src); free(blocks); free(d); free(want);
        }
    }
}

// ---------------------------------------------------------------------------
// Online softmax from generic VU ops (VU §8.4's composition): per chunk RED.MAX, SOP.MAX,
// VOP.EXPSUBC for the rescale and the exponentials, RED.SUM, SOP.MUL / ADD; then EXPSUBC with the
// final maximum and MULC by RC(l).
static void softmax_vu(const uint32_t * x, uint32_t m, uint32_t chunk, uint32_t * p) {
    uint32_t m_run = NINF, l_run = 0;
    uint32_t * e = malloc(16 * tsr_ref_vu_lines(chunk) * sizeof *e);
    for (uint32_t k0 = 0; k0 < m; k0 += chunk) {
        const uint32_t n = (m - k0 < chunk) ? m - k0 : chunk;
        const uint32_t mc = tsr_ref_red_row(TSR_RED_KIND_MAX, 0, x + k0, n, 0, 0);
        const uint32_t mn = tsr_ref_sop(TSR_SOP_SUB_MAX, m_run, mc, 0, 0);
        uint32_t alpha_line[16];                   // a one-element row still writes its line's tail
        tsr_ref_vop(TSR_VOP_FUNC_EXPSUBC, TSR_VOP_CMODE_SREG, 1, 1, alpha_line, 0, &m_run, 0, NULL, 0, &mn);
        const uint32_t alpha = alpha_line[0];
        tsr_ref_vop(TSR_VOP_FUNC_EXPSUBC, TSR_VOP_CMODE_SREG, 1, n, e, 0, x + k0, 0, NULL, 0, &mn);
        const uint32_t lc = tsr_ref_red_row(TSR_RED_KIND_SUM, 0, e, n, 0, 0);
        l_run = tsr_ref_sop(TSR_SOP_SUB_ADD, tsr_ref_sop(TSR_SOP_SUB_MUL, l_run, alpha, 0, 0), lc, 0, 0);
        m_run = mn;
    }
    const uint32_t rl = tsr_ref_sop(TSR_SOP_SUB_RECIP, l_run, 0, 0, 0);
    tsr_ref_vop(TSR_VOP_FUNC_EXPSUBC, TSR_VOP_CMODE_SREG, 1, m, p, 0, x, 0, NULL, 0, &m_run);
    tsr_ref_vop(TSR_VOP_FUNC_MULC, TSR_VOP_CMODE_SREG, 1, m, p, 0, p, 0, NULL, 0, &rl);
    free(e);
}

static void test_softmax(void) {
    rng_t r = { 0x50F7ull };
    enum { M = 600 };
    uint32_t x[M], p[16 * ((M + 15) / 16)], q[16 * ((M + 15) / 16)];
    double worst = 0;
    for (int it = 0; it < 200; ++it) {
        const uint32_t m = 1 + rng_below(&r, M);
        const uint32_t masked = (it % 3 == 0) ? (m > 128 ? 128 : m / 2) : 0;   // fully masked first chunk
        for (uint32_t i = 0; i < m; ++i) {
            x[i] = (i < masked || rng_below(&r, 10) == 0) ? NINF : fb((float)(rng_normal(&r) * 4.0));
        }
        x[m - 1] = fb(1.0f);                       // at least one unmasked element
        softmax_vu(x, m, 128, p);
        softmax_vu(x, m, 1024, q);                 // one chunk
        double mx = -INFINITY, sum = 0;
        for (uint32_t i = 0; i < m; ++i) if (bd(x[i]) > mx) mx = bd(x[i]);
        for (uint32_t i = 0; i < m; ++i) sum += exp((double)bf(x[i]) - mx);
        double tot = 0;
        for (uint32_t i = 0; i < m; ++i) {
            const double ex = exp((double)bf(x[i]) - mx) / sum;
            CHECK(!(bf(p[i]) != bf(p[i])), "softmax NaN at %u (m %u)", i, m);
            if (x[i] == NINF) {
                EQ("masked prob exactly 0", p[i], 0u);
            } else {
                const double e = fabs(bd(p[i]) - ex) / ex;
                if (e > worst) worst = e;
                const double e1 = fabs(bd(q[i]) - ex) / ex;
                if (e1 > worst) worst = e1;
            }
            tot += bd(p[i]);
        }
        CHECK(fabs(tot - 1.0) < 1e-5, "softmax sum %.9f (m %u)", tot, m);
    }
    printf("  softmax (online, EXPSUB/EXPSUBC, chunks of 128 and one chunk): max relative error %.3g\n", worst);
    CHECK(worst < 1e-5, "softmax relative error %g", worst);
}

// ---------------------------------------------------------------------------
#ifdef TSR_REF_VU_GGML
// ggml_rope_ext on ggml-cpu: [head_dim, 1, T] f32, positions 0 .. T-1.
static float * ggml_rope_run(const rope_model * mdl, const float * x, uint32_t T, const float * ff) {
    const size_t n = (size_t)mdl->head_dim * T;
    struct ggml_init_params ip = { 3 * n * sizeof(float) + 64u * 1024u * 1024u, NULL, false };
    struct ggml_context * ctx = ggml_init(ip);
    struct ggml_tensor * a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, mdl->head_dim, 1, T);
    struct ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    struct ggml_tensor * fft = NULL;
    memcpy(a->data, x, n * sizeof(float));
    for (uint32_t t = 0; t < T; ++t) ((int32_t *)pos->data)[t] = (int32_t)t;
    if (ff) {
        fft = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, mdl->n_dims / 2);
        memcpy(fft->data, ff, mdl->n_dims / 2 * sizeof(float));
    }
    const int mode = mdl->mode == TSR_ROPE_MODE_NEOX ? GGML_ROPE_TYPE_NEOX : 0;
    struct ggml_tensor * y = ggml_rope_ext(ctx, a, pos, fft, (int)mdl->n_dims, mode, 32768, mdl->base,
                                           mdl->freq_scale, 0.0f, 1.0f, 32.0f, 1.0f);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    ggml_graph_compute_with_ctx(ctx, gf, 4);
    float * out = malloc(n * sizeof(float));
    memcpy(out, y->data, n * sizeof(float));
    ggml_free(ctx);
    return out;
}

static void test_ggml_rope(void) {
    enum { T = 32768, TB = 2048 };
    rng_t r = { 0x6611ull };
    static uint32_t phi[512], tab[512];
    static float ffv[512];
    for (size_t mi = 0; mi < N_ROPE_MODELS; ++mi) {
        const rope_model * mdl = &k_rope_models[mi];
        const uint32_t hd = mdl->head_dim, nd = mdl->n_dims, half = nd / 2;
        if (mdl->ff_kind) rope_freqs(mdl, ffv);
        for (uint32_t i = 0; i < half; ++i) tab[i] = fb(ffv[i]);
        const float * ffp = mdl->ff_kind ? ffv : NULL;
        tsr_ref_rope_cfg cfg = rope_cfg_of(mdl, 1, 1);
        // (a) Unit pairs, positions 0 .. 32767: x0 = 1, x1 = 0 gives y0 = cosf(theta)·1 and
        //     y1 = sinf(theta)·1 exactly (FMA or not), so ggml's output is libm of its angles.
        size_t n = (size_t)hd * T;
        float * x = malloc(n * sizeof(float));
        for (size_t i = 0; i < n; ++i) x[i] = 0.0f;
        for (uint32_t t = 0; t < T; ++t) {
            for (uint32_t i = 0; i < half; ++i) x[(size_t)t * hd + (mdl->mode ? i : 2 * i)] = 1.0f;
        }
        float * y = ggml_rope_run(mdl, x, T, ffp);
        long bad = 0;
        for (uint32_t t = 0; t < T; ++t) {
            tsr_ref_rope_angles(&cfg, (int32_t)t, tab, phi);
            for (uint32_t i = 0; i < half; ++i) {
                const uint32_t j0 = mdl->mode ? i : 2 * i, j1 = mdl->mode ? i + half : 2 * i + 1;
                bad += fb(y[(size_t)t * hd + j0]) != fb(cosf(bf(phi[i])));
                bad += fb(y[(size_t)t * hd + j1]) != fb(sinf(bf(phi[i])));
            }
        }
        CHECK(bad == 0, "%s: %ld of %ld ggml-cpu cos/sin values differ from libm on our angles", mdl->name, bad,
              2L * T * half);
        printf("  ggml_rope: %-55s %ld angles, positions 0..%d: %s\n", mdl->name, (long)T * half, T - 1,
               bad ? "DIFFER" : "bit-identical to ggml-cpu");
        free(y);
        free(x);
        // (b) Random inputs, positions 0 .. TB-1: the full op. Informative: VU against ggml-cpu (libm
        //     cos/sin and the CPU compiler's rotation), and ggml-cpu against libm-on-our-angles with
        //     the rotation evaluated here without FMA (equal everywhere iff the CPU does not contract).
        n = (size_t)hd * TB;
        x = malloc(n * sizeof(float));
        for (size_t i = 0; i < n; ++i) x[i] = (float)rng_normal(&r);
        y = ggml_rope_run(mdl, x, TB, ffp);
        uint32_t * xs = malloc(n * sizeof *xs), * yv = malloc(n * sizeof *yv), * pos = malloc(TB * sizeof *pos);
        memcpy(xs, x, n * sizeof(float));
        for (uint32_t t = 0; t < TB; ++t) pos[t] = t;
        cfg.T = TB;
        tsr_ref_rope(&cfg, yv, hd / 16, xs, hd / 16, pos, tab);
        long same_vu = 0, same_libm = 0, same_fma = 0, tail_bad = 0;
        double maxd = 0;
        for (uint32_t t = 0; t < TB; ++t) {
            tsr_ref_rope_angles(&cfg, (int32_t)t, tab, phi);
            const size_t o = (size_t)t * hd;
            for (uint32_t i = 0; i < half; ++i) {
                const uint32_t j0 = mdl->mode ? i : 2 * i, j1 = mdl->mode ? i + half : 2 * i + 1;
                const float cc = cosf(bf(phi[i])), sn = sinf(bf(phi[i]));
                const float x0 = x[o + j0], x1 = x[o + j1];
                const float l0 = x0 * cc - x1 * sn, l1 = x0 * sn + x1 * cc;    // -ffp-contract=off here
                same_libm += (fb(l0) == fb(y[o + j0])) + (fb(l1) == fb(y[o + j1]));
                // The contracted forms GCC emits for a·b ± c·d: fma(a, b, ±(c·d)).
                const float f0 = fmaf(x0, cc, -(x1 * sn)), f1 = fmaf(x0, sn, x1 * cc);
                same_fma += (fb(f0) == fb(y[o + j0])) + (fb(f1) == fb(y[o + j1]));
                same_vu += (yv[o + j0] == fb(y[o + j0])) + (yv[o + j1] == fb(y[o + j1]));
                const double d0 = fabs(bd(yv[o + j0]) - (double)y[o + j0]), d1 = fabs(bd(yv[o + j1]) - (double)y[o + j1]);
                if (d0 > maxd) maxd = d0;
                if (d1 > maxd) maxd = d1;
            }
            for (uint32_t j = nd; j < hd; ++j) tail_bad += yv[o + j] != fb(y[o + j]);
        }
        const double tot = 2.0 * TB * half;
        printf("    full op, random x, %d positions: VU = CPU on %.2f%% of rotated values (max |diff| %.3g); "
               "CPU = libm-on-our-angles without FMA on %.2f%%, with fma(x0, c, -(x1·s)) / fma(x0, s, x1·c) on %.2f%%\n",
               TB, 100.0 * same_vu / tot, maxd, 100.0 * same_libm / tot, 100.0 * same_fma / tot);
        CHECK(tail_bad == 0, "%s: unrotated tail differs", mdl->name);
        CHECK(maxd < 8e-6, "%s: VU vs CPU rope max diff %g", mdl->name, maxd);
        free(xs); free(yv); free(pos); free(y); free(x);
    }
}

// ggml_rms_norm on ggml-cpu ([n, rows]) against RED.SUMSQ (post RMS, sk = f32(1/n), se = eps) +
// VOP.MULC (VU §8.1).
static void test_ggml_rms(void) {
    rng_t r = { 0x4A5ull };
    const uint32_t ns[] = { 64, 128, 256, 512, 896, 3840, 5120 };
    const float eps = 1e-6f;
    for (size_t ni = 0; ni < sizeof ns / sizeof ns[0]; ++ni) {
        const uint32_t n = ns[ni], rows = 256;
        struct ggml_init_params ip = { (size_t)3 * n * rows * sizeof(float) + 16u * 1024u * 1024u, NULL, false };
        struct ggml_context * ctx = ggml_init(ip);
        struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, rows);
        float * x = (float *)a->data;
        for (uint32_t row = 0; row < rows; ++row) {
            const double sc = pow(10.0, (double)rng_below(&r, 7) - 3.0);
            for (uint32_t j = 0; j < n; ++j) x[(size_t)row * n + j] = (float)(rng_normal(&r) * sc * (rng_below(&r, 100) == 0 ? 30.0 : 1.0));
        }
        struct ggml_tensor * y = ggml_rms_norm(ctx, a, eps);
        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, y);
        ggml_graph_compute_with_ctx(ctx, gf, 4);
        const float * yc = (const float *)y->data;
        const uint32_t sk = fb(tsr_frecip((float)n)), se = fb(eps);
        uint32_t * row_out = malloc(16 * tsr_ref_vu_lines(n) * sizeof *row_out);
        long rows_same = 0, scale_same = 0, mean_same = 0, mean_same_dsum = 0;
        uint32_t max_ulp = 0;
        for (uint32_t row = 0; row < rows; ++row) {
            const uint32_t * xr = (const uint32_t *)(x + (size_t)row * n);
            const uint32_t s1 = tsr_ref_red_row(TSR_RED_KIND_SUMSQ, TSR_RED_POST_RMS, xr, n, sk, se);
            tsr_ref_vop(TSR_VOP_FUNC_MULC, TSR_VOP_CMODE_SREG, 1, n, row_out, 0, xr, 0, NULL, 0, &s1);
            // The CPU's scale (ops.cpp:4224-4231): double sum of f32 squares, mean = f32(sum/n),
            // 1.0f/sqrtf(mean + eps); its output is x·scale (checked here bit for bit).
            double sum = 0;
            for (uint32_t j = 0; j < n; ++j) sum += (double)(x[(size_t)row * n + j] * x[(size_t)row * n + j]);
            const float mean = (float)(sum / n);
            const float scale = 1.0f / sqrtf(mean + eps);
            int same = 1;
            for (uint32_t j = 0; j < n; ++j) {
                const float yt = x[(size_t)row * n + j] * scale;
                CHECK(fb(yt) == fb(yc[(size_t)row * n + j]), "CPU rms transcription n %u row %u j %u", n, row, j);
                same &= row_out[j] == fb(yc[(size_t)row * n + j]);
            }
            rows_same += same;
            scale_same += s1 == fb(scale);
            // Where the difference comes from: the VU's mean FM(f32 tree sum, f32(1/n)) against the
            // CPU's f32(double sum / n), and the same multiply applied to the CPU's sum rounded to f32.
            const uint32_t sum_vu = tsr_ref_red_row(TSR_RED_KIND_SUMSQ, TSR_RED_POST_NONE, xr, n, 0, 0);
            mean_same += fb(tsr_fmul(bf(sum_vu), bf(sk))) == fb(mean);
            mean_same_dsum += fb(tsr_fmul((float)sum, bf(sk))) == fb(mean);
            const uint32_t du = s1 > fb(scale) ? s1 - fb(scale) : fb(scale) - s1;   // positive normals: ulp distance
            if (du > max_ulp) max_ulp = du;
        }
        printf("  ggml_rms: n %5u: scale bit-identical in %3ld of %u rows, max %u ulp; output rows identical %3ld; "
               "mean identical %3ld (with the CPU's double sum and FM by f32(1/n): %3ld)\n",
               n, scale_same, rows, max_ulp, rows_same, mean_same, mean_same_dsum);
        CHECK(max_ulp <= 8, "rms n %u: %u ulp", n, max_ulp);
        free(row_out);
        ggml_free(ctx);
    }
}
#endif

int main(void) {
    test_order();
    test_vop_elem();
    test_vop_shape();
    test_red();
    test_rope();
    test_topk();
    test_sop();
    test_dqrow();
    test_softmax();
#ifdef TSR_REF_VU_GGML
    test_ggml_rope();
    test_ggml_rms();
#else
    printf("  (ggml cross-checks not built: TSR_WITH_LLAMA is off)\n");
#endif
    return test_finish("ref_vu");
}
