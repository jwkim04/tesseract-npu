// test_sim_vu.c — the simulator's VU unit (sw/sim/tsr_vu.c): VOP, RED, ROPE, TOPK, SOP,
// DQROW and GDN pushed as encoded ops through the test hooks (tsr_sim_op, and tsr_sim_push on the queue
// path), each compared bit for bit with direct tsr_ref_vu calls applied to a copy of the SPM and
// of the s bank (whole SPM images: results, tails, strides, in place, and every untouched line).
// Also: every check code each op can raise with its ERR_INFO and the check order (VU §5.9, EX
// §8.1), the queue model's records of them, CAPS gating, VMASK and the reserved VU opcodes, the
// operand extents (tsr_sim_int.h §1), the cycle model, PERF (VU_CYCLES, VU_LINES, gating by
// ACTIVE), and three kernels as op sequences: RMSNorm (RED SUMSQ RMS + VOP MULCB, and the spec's
// SOP recipe, VU §6.2 / §8.1), per-head norms with a c vector, and a softmax (RED MAX, EXPSUBC,
// RED SUM RCP, VOP MULC), the latter also through the queue model.
#include "test_sim_util.h"
#include "tsr_sim_int.h"
#include "tsr_ref_vu.h"
#include "tsr_ref_math.h"
#include "tsr_ref_gdn.h"

#define SPM_BYTES (1u << 20)                  // 16,384 lines: whole-image comparisons stay cheap
#define SPM_LINES (SPM_BYTES / 64u)
#define CAPS_ALL  (TSR_CAPS_VU | TSR_CAPS_TOPK | TSR_CAPS_DQROW | TSR_CAPS_GDN)
#define ALL_WORDS 0xFF00u                     // ISSUE_PUSH: send all eight words

static const uint32_t F_ONE = 0x3F800000u;

// ---------------------------------------------------------------------------
// Encoders (field constants of the generated table)
// ---------------------------------------------------------------------------
static tsr_insn e_vop(uint32_t func, uint32_t cmode, uint32_t si, uint32_t d, uint32_t a, uint32_t b, uint32_t c,
                      uint32_t m, uint32_t rows, uint32_t ds, uint32_t as, uint32_t bs) {
    tsr_insn i = {{0}};
    i.w[0] = TSR_OP_VOP;
    TSR_INSN_SET(i.w, TSR_VOP_CMODE, cmode);
    TSR_INSN_SET(i.w, TSR_VOP_FUNC, func);
    TSR_INSN_SET(i.w, TSR_VOP_SI, si);
    TSR_INSN_SET(i.w, TSR_VOP_D, d);
    TSR_INSN_SET(i.w, TSR_VOP_A, a);
    TSR_INSN_SET(i.w, TSR_VOP_B, b);
    TSR_INSN_SET(i.w, TSR_VOP_C, c);
    TSR_INSN_SET(i.w, TSR_VOP_M, m);
    TSR_INSN_SET(i.w, TSR_VOP_ROWS, rows);
    TSR_INSN_SET(i.w, TSR_VOP_D_STRIDE, ds);
    TSR_INSN_SET(i.w, TSR_VOP_A_STRIDE, as);
    TSR_INSN_SET(i.w, TSR_VOP_B_STRIDE, bs);
    return i;
}
static tsr_insn e_red(uint32_t kind, uint32_t to_s, uint32_t sd, uint32_t post, uint32_t sk, uint32_t se,
                      uint32_t d, uint32_t a, uint32_t m, uint32_t rows, uint32_t as) {
    tsr_insn i = {{0}};
    i.w[0] = TSR_OP_RED;
    TSR_INSN_SET(i.w, TSR_RED_KIND, kind);
    TSR_INSN_SET(i.w, TSR_RED_TO_S, to_s);
    TSR_INSN_SET(i.w, TSR_RED_SD, sd);
    TSR_INSN_SET(i.w, TSR_RED_POST, post);
    TSR_INSN_SET(i.w, TSR_RED_SK, sk);
    TSR_INSN_SET(i.w, TSR_RED_SE, se);
    TSR_INSN_SET(i.w, TSR_RED_D, d);
    TSR_INSN_SET(i.w, TSR_RED_A, a);
    TSR_INSN_SET(i.w, TSR_RED_M, m);
    TSR_INSN_SET(i.w, TSR_RED_ROWS, rows);
    TSR_INSN_SET(i.w, TSR_RED_A_STRIDE, as);
    return i;
}
typedef struct rope_f {
    uint32_t mode, ff, fs, atab, sm, sts, sfs, d, src, pos, table, hd, nd, nh, T, ss, ds;
} rope_f;
static tsr_insn e_rope(const rope_f * f) {
    tsr_insn i = {{0}};
    i.w[0] = TSR_OP_ROPE;
    TSR_INSN_SET(i.w, TSR_ROPE_MODE, f->mode);
    TSR_INSN_SET(i.w, TSR_ROPE_FF, f->ff);
    TSR_INSN_SET(i.w, TSR_ROPE_FS, f->fs);
    TSR_INSN_SET(i.w, TSR_ROPE_ATAB, f->atab);
    TSR_INSN_SET(i.w, TSR_ROPE_SM, f->sm);
    TSR_INSN_SET(i.w, TSR_ROPE_STS, f->sts);
    TSR_INSN_SET(i.w, TSR_ROPE_SFS, f->sfs);
    TSR_INSN_SET(i.w, TSR_ROPE_D, f->d);
    TSR_INSN_SET(i.w, TSR_ROPE_SRC, f->src);
    TSR_INSN_SET(i.w, TSR_ROPE_POS, f->pos);
    TSR_INSN_SET(i.w, TSR_ROPE_TABLE, f->table);
    TSR_INSN_SET(i.w, TSR_ROPE_HEAD_DIM, f->hd);
    TSR_INSN_SET(i.w, TSR_ROPE_N_DIMS, f->nd);
    TSR_INSN_SET(i.w, TSR_ROPE_N_HEADS, f->nh);
    TSR_INSN_SET(i.w, TSR_ROPE_T, f->T);
    TSR_INSN_SET(i.w, TSR_ROPE_SRC_STRIDE, f->ss);
    TSR_INSN_SET(i.w, TSR_ROPE_D_STRIDE, f->ds);
    return i;
}
static tsr_insn e_topk(uint32_t k, uint32_t d, uint32_t a, uint32_t m, uint32_t rows, uint32_t ds, uint32_t as) {
    tsr_insn i = {{0}};
    i.w[0] = TSR_OP_TOPK;
    TSR_INSN_SET(i.w, TSR_TOPK_K, k);
    TSR_INSN_SET(i.w, TSR_TOPK_D, d);
    TSR_INSN_SET(i.w, TSR_TOPK_A, a);
    TSR_INSN_SET(i.w, TSR_TOPK_M, m);
    TSR_INSN_SET(i.w, TSR_TOPK_ROWS, rows);
    TSR_INSN_SET(i.w, TSR_TOPK_D_STRIDE, ds);
    TSR_INSN_SET(i.w, TSR_TOPK_A_STRIDE, as);
    return i;
}
static tsr_insn e_sop(uint32_t sub, uint32_t sd, uint32_t sa, uint32_t sb, uint32_t imm, uint32_t addr) {
    tsr_insn i = {{0}};
    i.w[0] = TSR_OP_SOP;
    TSR_INSN_SET(i.w, TSR_SOP_SUB, sub);
    TSR_INSN_SET(i.w, TSR_SOP_SD, sd);
    TSR_INSN_SET(i.w, TSR_SOP_SA, sa);
    TSR_INSN_SET(i.w, TSR_SOP_SB, sb);
    TSR_INSN_SET(i.w, TSR_SOP_IMM, imm);
    TSR_INSN_SET(i.w, TSR_SOP_ADDR, addr);
    return i;
}
static tsr_insn e_dqrow(uint32_t fmt, uint32_t d, uint32_t src, uint32_t K, uint32_t rows, uint32_t ss, uint32_t ds) {
    tsr_insn i = {{0}};
    i.w[0] = TSR_OP_DQROW;
    TSR_INSN_SET(i.w, TSR_DQROW_FMT, fmt);
    TSR_INSN_SET(i.w, TSR_DQROW_D, d);
    TSR_INSN_SET(i.w, TSR_DQROW_SRC, src);
    TSR_INSN_SET(i.w, TSR_DQROW_K, K);
    TSR_INSN_SET(i.w, TSR_DQROW_ROWS, rows);
    TSR_INSN_SET(i.w, TSR_DQROW_SRC_STRIDE, ss);
    TSR_INSN_SET(i.w, TSR_DQROW_D_STRIDE, ds);
    return i;
}

static tsr_insn e_gdn(uint32_t sa, uint32_t sb, uint32_t d, uint32_t a, uint32_t k, uint32_t q, uint32_t v,
                      uint32_t o, uint32_t m, uint32_t rows) {
    tsr_insn i = {{0}};
    i.w[0] = TSR_OP_GDN;
    TSR_INSN_SET(i.w, TSR_GDN_SA, sa);
    TSR_INSN_SET(i.w, TSR_GDN_SB, sb);
    TSR_INSN_SET(i.w, TSR_GDN_D, d);
    TSR_INSN_SET(i.w, TSR_GDN_A, a);
    TSR_INSN_SET(i.w, TSR_GDN_K, k);
    TSR_INSN_SET(i.w, TSR_GDN_Q, q);
    TSR_INSN_SET(i.w, TSR_GDN_V, v);
    TSR_INSN_SET(i.w, TSR_GDN_O, o);
    TSR_INSN_SET(i.w, TSR_GDN_M, m);
    TSR_INSN_SET(i.w, TSR_GDN_ROWS, rows);
    return i;
}

// ---------------------------------------------------------------------------
// The twin: the simulator, and the expected SPM image and s bank that direct reference calls
// produce on the state before each op.
// ---------------------------------------------------------------------------
typedef struct twin {
    tsr_sim * sim;
    uint8_t * exp;
    uint32_t  s[16];
    rng_t     r;
} twin;

static tsr_sim * mk_sim(uint32_t caps, uint32_t t_max) {
    tsr_config c = cfg_u200(t_max);
    c.spm_bytes = SPM_BYTES;
    c.ch_bytes = 64ull << 20;
    c.caps = caps;
    tsr_sim * s = tsr_sim_create(&c, NULL);
    if (s == NULL) {
        fprintf(stderr, "tsr_sim_create failed\n");
        exit(1);
    }
    return s;
}

// A word with the specials of VU §4.1 mixed in: +-0, +-inf, NaNs, subnormals (DAZ), the FTZ
// boundary, large and small normals.
static uint32_t rand_word(rng_t * r) {
    const uint32_t sgn = rng_u32(r) & 0x80000000u;
    switch (rng_below(r, 32)) {
    case 0: return sgn;
    case 1: return sgn | 0x7F800000u;
    case 2: return 0x7F800000u | (rng_u32(r) | 1u);                       // NaN, any sign / payload
    case 3: return sgn | (rng_u32(r) & 0x007FFFFFu);                      // subnormal
    case 4: return sgn | 0x00800000u | (rng_u32(r) & 3u);                 // FTZ boundary
    case 5: return sgn | ((uint32_t)(200u + rng_below(r, 54)) << 23) | (rng_u32(r) & 0x7FFFFFu);   // huge
    case 6: return sgn | ((uint32_t)(1u + rng_below(r, 40)) << 23) | (rng_u32(r) & 0x7FFFFFu);     // tiny
    default: return f2u((float)(rng_normal(r) * 4.0));
    }
}
static uint32_t rand_norm(rng_t * r, double scale) { return f2u((float)(rng_normal(r) * scale)); }

static uint32_t * W(uint8_t * img, uint32_t line) { return (uint32_t *)(void *)(img + (size_t)line * 64u); }
static uint32_t * SW(tsr_sim * s, uint32_t line) { return W(tsr_sim_spm(s), line); }

static void tw_init(twin * t, uint32_t caps, uint64_t seed) {
    t->sim = mk_sim(caps, 8);
    t->r.s = seed;
    uint32_t * w = SW(t->sim, 0);
    for (uint32_t i = 0; i < SPM_BYTES / 4u; ++i) w[i] = rand_word(&t->r);
    t->exp = (uint8_t *)malloc(SPM_BYTES);
    memcpy(t->exp, tsr_sim_spm(t->sim), SPM_BYTES);
    memset(t->s, 0, sizeof t->s);
}
static void tw_free(twin * t) {
    tsr_sim_destroy(t->sim);
    free(t->exp);
}
static int tw_same(twin * t, const char * what) {
    const int spm_ok = memcmp(t->exp, tsr_sim_spm(t->sim), SPM_BYTES) == 0;
    const int s_ok = memcmp(t->s, tsr_sim_sreg(t->sim), sizeof t->s) == 0;
    if (!spm_ok) {
        const uint32_t * a = W(t->exp, 0), * b = SW(t->sim, 0);
        uint32_t i = 0;
        while (a[i] == b[i]) ++i;
        fprintf(stderr, "  %s: first SPM difference at line %u word %u: exp %08x sim %08x\n", what, i / 16u, i % 16u, a[i], b[i]);
    }
    if (!s_ok) {
        const uint32_t * b = (const uint32_t *)(const void *)tsr_sim_sreg(t->sim);
        for (int i = 0; i < 16; ++i) {
            if (t->s[i] != b[i]) fprintf(stderr, "  %s: s%d exp %08x sim %08x\n", what, i, t->s[i], b[i]);
        }
    }
    CHECK(spm_ok && s_ok, "%s: SPM and s bank equal the direct reference", what);
    return spm_ok && s_ok;
}
// Random words into n lines at `line` of both images.
static void tw_rand(twin * t, uint32_t line, uint32_t n) {
    uint32_t * a = SW(t->sim, line), * b = W(t->exp, line);
    for (uint32_t i = 0; i < 16u * n; ++i) a[i] = b[i] = rand_word(&t->r);
}
static void tw_put(twin * t, uint32_t line, uint32_t k, uint32_t v) {
    SW(t->sim, line)[k] = v;
    W(t->exp, line)[k] = v;
}
static uint32_t tw_op(twin * t, tsr_insn i, const char * what) {
    uint32_t info = 0;
    const uint32_t e = tsr_sim_op(t->sim, i.w, &info);
    CHECK(e == TSR_ERR_NONE, "%s: op failed with %u (info 0x%x)", what, e, info);
    return e;
}
// s[i] = v through SOP MOVI (raw).
static void tw_movi(twin * t, uint32_t i, uint32_t v) {
    tw_op(t, e_sop(TSR_SOP_SUB_MOVI, i, 0, 0, v, 0), "MOVI");
    t->s[i] = v;
}

// ---------------------------------------------------------------------------
// VOP: every function, every legal cmode, shapes with tails, strides 0 / larger than L(m), in
// place, m = 0 and rows = 0.
// ---------------------------------------------------------------------------
typedef struct { uint32_t rows, m, ds, as, bs; int inplace; } vshape;

static void test_vop(void) {
    twin t;
    tw_init(&t, CAPS_ALL, 0x5EED0001u);
    static const vshape shapes[] = {
        { 1, 1, 0, 0, 0, 0 },  { 1, 16, 0, 0, 0, 0 }, { 3, 17, 2, 2, 0, 0 }, { 4, 40, 3, 5, 3, 0 },
        { 2, 33, 4, 4, 1, 1 }, { 5, 7, 1, 0, 1, 0 },  { 1, 0, 0, 0, 0, 0 },  { 0, 20, 2, 2, 2, 0 },
        { 20, 3, 1, 1, 0, 1 },
    };
    const uint32_t D = 100, A = 2100, B = 4100, CV = 6100;
    uint32_t n_ops = 0;
    for (uint32_t func = 0; func < TSR_VOP_FUNC_N; ++func) {
        const uint32_t reads = tsr_vop_func_reads(func);
        const int ra = (reads & TSR_VOP_READS_A) != 0, rb = (reads & TSR_VOP_READS_B) != 0;
        for (uint32_t cm = 0; cm <= TSR_VOP_CMODE_VEC; ++cm) {
            if (((reads & TSR_VOP_READS_C) != 0) != (cm != TSR_VOP_CMODE_NONE)) continue;
            for (size_t k = 0; k < sizeof(shapes) / sizeof(shapes[0]); ++k) {
                const vshape * sh = &shapes[k];
                if (sh->inplace && !ra) continue;
                tw_rand(&t, D, 128);
                tw_rand(&t, A, 128);
                tw_rand(&t, B, 128);
                tw_rand(&t, CV, 4);
                uint32_t si = 0, cval = 0;
                if (cm == TSR_VOP_CMODE_SREG) {
                    si = rng_below(&t.r, 16);
                    tw_movi(&t, si, rand_word(&t.r));
                    cval = t.s[si];
                }
                const uint32_t a = ra ? (sh->inplace ? D : A) : 0, as = ra ? (sh->inplace ? sh->ds : sh->as) : 0;
                const uint32_t b = rb ? B : 0, bs = rb ? sh->bs : 0;
                const uint32_t c = (cm == TSR_VOP_CMODE_VEC) ? CV : 0;
                char what[96];
                snprintf(what, sizeof what, "VOP func %u cmode %u shape %zu", func, cm, k);
                if (tw_op(&t, e_vop(func, cm, si, D, a, b, c, sh->m, sh->rows, sh->ds, as, bs), what) != 0) continue;
                tsr_ref_vop(func, cm, sh->rows, sh->m, W(t.exp, D), sh->ds, ra ? W(t.exp, a) : NULL, as,
                            rb ? W(t.exp, b) : NULL, bs,
                            cm == TSR_VOP_CMODE_SREG ? &cval : cm == TSR_VOP_CMODE_VEC ? W(t.exp, c) : NULL);
                tw_same(&t, what);
                ++n_ops;
            }
        }
    }
    CHECK(n_ops > 200, "VOP cases ran (%u)", n_ops);
    tw_free(&t);
}

// ---------------------------------------------------------------------------
// RED: kinds x posts x TO_S / vector, m across the partial / tree boundaries, strides, the -0
// subtree rows of VU §6.2.
// ---------------------------------------------------------------------------
static void test_red(void) {
    twin t;
    tw_init(&t, CAPS_ALL, 0x5EED0002u);
    static const uint32_t ms[] = { 0, 1, 7, 8, 9, 64, 127, 128, 129, 300 };
    const uint32_t D = 200, A = 1000;
    uint32_t n_ops = 0;
    for (uint32_t kind = 0; kind <= TSR_RED_KIND_MAX; ++kind) {
        for (uint32_t post = 0; post <= TSR_RED_POST_RMS; ++post) {
            if (kind == TSR_RED_KIND_MAX && post != TSR_RED_POST_NONE) continue;
            for (size_t mi = 0; mi < sizeof(ms) / sizeof(ms[0]); ++mi) {
                const uint32_t m = ms[mi], L = tsr_ref_vu_lines(m);
                // TO_S (rows = 1, any a_stride)
                {
                    tw_rand(&t, A, 32);
                    const uint32_t sd = rng_below(&t.r, 16), sk = (post == TSR_RED_POST_RMS) ? 1u + rng_below(&t.r, 15) : 0u;
                    const uint32_t se = (post == TSR_RED_POST_RMS) ? rng_below(&t.r, 16) : 0u;
                    if (post == TSR_RED_POST_RMS) {
                        tw_movi(&t, sk, f2u(1.0f / (float)(m ? m : 1u)));
                        tw_movi(&t, se, f2u(1e-6f));
                    }
                    char what[96];
                    snprintf(what, sizeof what, "RED kind %u post %u TO_S m %u", kind, post, m);
                    if (tw_op(&t, e_red(kind, 1, sd, post, sk, se, 0, A, m, 1, 7), what) == 0) {
                        t.s[sd] = tsr_ref_red_row(kind, post, W(t.exp, A), m, t.s[sk], t.s[se]);
                        tw_same(&t, what);
                        ++n_ops;
                    }
                }
                // vector destination
                static const uint32_t rowsv[] = { 1, 5, 20 };
                for (size_t ri = 0; ri < 3; ++ri) {
                    for (int sv = 0; sv < 3; ++sv) {
                        const uint32_t rows = rowsv[ri], as = sv == 0 ? L : sv == 1 ? L + 1u : 0u;
                        tw_rand(&t, A, (rows + 1u) * (L + 1u));
                        tw_rand(&t, D, 2);
                        const uint32_t sk = (post == TSR_RED_POST_RMS) ? 2u : 0u, se = (post == TSR_RED_POST_RMS) ? 3u : 0u;
                        if (post == TSR_RED_POST_RMS) {
                            tw_movi(&t, 2, rand_word(&t.r));
                            tw_movi(&t, 3, f2u(1e-5f));
                        }
                        char what[96];
                        snprintf(what, sizeof what, "RED kind %u post %u m %u rows %u as %u", kind, post, m, rows, as);
                        if (tw_op(&t, e_red(kind, 0, 0, post, sk, se, D, A, m, rows, as), what) != 0) continue;
                        tsr_ref_red(kind, post, rows, m, W(t.exp, D), W(t.exp, A), as, t.s[sk], t.s[se]);
                        tw_same(&t, what);
                        ++n_ops;
                    }
                }
            }
        }
    }
    // -0 subtrees (VU §6.2): pairs FA(0x80C00000, 0x00800000) = -0 at m = 8k + r.
    static const uint32_t mz[] = { 2, 9, 17, 23, 66, 130, 255 };
    for (size_t i = 0; i < sizeof(mz) / sizeof(mz[0]); ++i) {
        for (uint32_t j = 0; j < mz[i]; ++j) tw_put(&t, A + j / 16u, j % 16u, (j & 1u) ? 0x00800000u : 0x80C00000u);
        if (tw_op(&t, e_red(TSR_RED_KIND_SUM, 1, 4, 0, 0, 0, 0, A, mz[i], 1, 0), "RED -0 subtree") == 0) {
            t.s[4] = tsr_ref_red_row(TSR_RED_KIND_SUM, 0, W(t.exp, A), mz[i], 0, 0);
            tw_same(&t, "RED -0 subtree");
            if (mz[i] <= 128u && (mz[i] & 1u) == 0u)               // one element per partial, all paired: cancel
                CHECK(t.s[4] == 0u || t.s[4] == 0x80000000u, "RED -0 subtree m %u gives a signed zero (%08x)", mz[i], t.s[4]);
        }
    }
    CHECK(n_ops > 500, "RED cases ran (%u)", n_ops);
    tw_free(&t);
}

// ---------------------------------------------------------------------------
// ROPE: NEOX / NORM, partial rotation with the raw tail, FF, FS, ATAB, 8 tokens, explicit and
// default token strides, in place, T' = 0.
// ---------------------------------------------------------------------------
typedef struct { uint32_t mode, ff, fs, atab, hd, nd, nh, T, ss_extra, ds_extra; int inplace; float base, m, fsc; } rcase;

static void test_rope(void) {
    twin t;
    tw_init(&t, CAPS_ALL, 0x5EED0003u);
    static const rcase cs[] = {
        { TSR_ROPE_MODE_NEOX, 0, 0, 0, 64, 64, 3, 8, 0, 0, 0, 1e6f, 1.0f, 1.0f },     // Qwen2.5-0.5B-like q+k
        { TSR_ROPE_MODE_NEOX, 0, 0, 0, 128, 128, 2, 8, 0, 0, 1, 1e6f, 1.0f, 1.0f },   // in place
        { TSR_ROPE_MODE_NORM, 1, 1, 0, 128, 64, 2, 8, 3, 1, 0, 1e4f, 0.9f, 0.25f },  // FF + FS, raw tail, strides
        { TSR_ROPE_MODE_NEOX, 0, 0, 1, 256, 256, 1, 8, 0, 2, 0, 1e4f, 1.1f, 1.0f },   // A table
        { TSR_ROPE_MODE_NORM, 0, 1, 0, 16, 16, 5, 3, 1, 0, 0, 1e4f, 1.0f, 0.5f },
        { TSR_ROPE_MODE_NEOX, 1, 0, 0, 512, 512, 1, 2, 0, 0, 0, 1e6f, 1.0f, 1.0f },   // Gemma 4 global: 1e30 factors
        { TSR_ROPE_MODE_NEOX, 0, 0, 0, 64, 32, 4, 1, 0, 0, 0, 1e6f, 1.0f, 1.0f },
        { TSR_ROPE_MODE_NEOX, 0, 0, 0, 64, 64, 3, 0, 0, 0, 0, 1e6f, 1.0f, 1.0f },     // T = 0
        { TSR_ROPE_MODE_NEOX, 0, 0, 0, 64, 64, 0, 4, 0, 0, 0, 1e6f, 1.0f, 1.0f },     // n_heads = 0
    };
    const uint32_t SRC = 1000, DST = 5000, POS = 9000, TAB = 9100;
    for (size_t k = 0; k < sizeof(cs) / sizeof(cs[0]); ++k) {
        const rcase * c = &cs[k];
        const uint32_t H = c->nh * c->hd / 16u;
        rope_f f = { c->mode, c->ff, c->fs, c->atab, 1, c->atab ? 0u : 2u, c->fs ? 3u : 0u, DST, SRC, POS,
                     (c->ff || c->atab) ? TAB : 0u, c->hd, c->nd, c->nh, c->T, 0, 0 };
        f.ss = c->ss_extra ? H + c->ss_extra : 0u;
        f.ds = c->ds_extra ? H + c->ds_extra : 0u;
        if (c->inplace) { f.d = SRC; f.ds = f.ss; }
        tw_rand(&t, SRC, 8u * (H + 4u));
        tw_rand(&t, DST, 8u * (H + 4u));
        for (uint32_t i = 0; i < 16; ++i) {
            const int32_t p = i == 0 ? 0 : i == 1 ? -5 : i == 2 ? 131071 : (int32_t)rng_below(&t.r, 40000);
            tw_put(&t, POS, i, (uint32_t)p);
        }
        const float ts = powf(c->base, -2.0f / (float)c->nd);
        for (uint32_t i = 0; i < c->nd / 2u; ++i) {
            uint32_t v;
            if (c->atab) v = f2u((float)pow((double)ts, (double)i) * 0.5f);
            else if (k == 5) v = (i % 4u) ? 0x7149F2CAu : F_ONE;                // 1e30 on 3/4 of the pairs
            else v = f2u(1.0f + (float)rng_below(&t.r, 40));
            tw_put(&t, TAB + i / 16u, i % 16u, v);
        }
        tw_movi(&t, 1, f2u(c->m));
        tw_movi(&t, 2, f2u(ts));
        tw_movi(&t, 3, f2u(c->fsc));
        char what[64];
        snprintf(what, sizeof what, "ROPE case %zu", k);
        if (tw_op(&t, e_rope(&f), what) != 0) continue;
        tsr_ref_rope_cfg cfg = { c->mode, c->ff, c->fs, c->atab, t.s[1], c->atab ? 0u : t.s[2], c->fs ? t.s[3] : 0u,
                                 c->hd, c->nd, c->nh, c->T };
        tsr_ref_rope(&cfg, W(t.exp, f.d), f.ds, W(t.exp, f.src), f.ss, W(t.exp, POS), f.table ? W(t.exp, TAB) : NULL);
        tw_same(&t, what);
    }
    tw_free(&t);
}

// ---------------------------------------------------------------------------
// TOPK, SOP, DQROW
// ---------------------------------------------------------------------------
static void test_topk(void) {
    twin t;
    tw_init(&t, CAPS_ALL, 0x5EED0004u);
    static const uint32_t ks[] = { 1, 5, 8, 9, 16 }, ms[] = { 3, 16, 100 }, rowsv[] = { 1, 4 };
    const uint32_t D = 300, A = 2000;
    for (size_t ki = 0; ki < 5; ++ki) {
        for (size_t mi = 0; mi < 3; ++mi) {
            for (size_t ri = 0; ri < 2; ++ri) {
                for (int dsx = 0; dsx < 2; ++dsx) {
                    const uint32_t k = ks[ki], m = ms[mi], rows = rowsv[ri], L = tsr_ref_vu_lines(m);
                    const uint32_t ds = dsx ? (k + 7u) / 8u + 1u : 0u, as = L + (uint32_t)dsx;
                    tw_rand(&t, A, rows * (L + 1u));
                    tw_rand(&t, D, rows * 3u);
                    if (m >= 16) {                                 // ties and signed zeros
                        tw_put(&t, A, 3, 0x40400000u);
                        tw_put(&t, A, 9, 0x40400000u);
                        tw_put(&t, A, 4, 0x80000000u);
                        tw_put(&t, A, 5, 0x00000000u);
                        tw_put(&t, A, 6, 0x00000017u);             // subnormal: DAZ'd to +0
                    }
                    char what[80];
                    snprintf(what, sizeof what, "TOPK k %u m %u rows %u ds %u", k, m, rows, ds);
                    if (tw_op(&t, e_topk(k, D, A, m, rows, ds, as), what) != 0) continue;
                    tsr_ref_topk(k, rows, m, W(t.exp, D), ds, W(t.exp, A), as);
                    tw_same(&t, what);
                }
            }
        }
    }
    tw_free(&t);
}

static void test_sop(void) {
    twin t;
    tw_init(&t, CAPS_ALL, 0x5EED0005u);
    for (int round = 0; round < 40; ++round) {
        for (uint32_t i = 0; i < 16; ++i) tw_movi(&t, i, rand_word(&t.r));
        for (uint32_t sub = 0; sub < TSR_SOP_SUB_N; ++sub) {
            const uint32_t uses = tsr_sop_uses(sub);
            const uint32_t sd = (uses & TSR_SOP_USES_SD) ? rng_below(&t.r, 16) : 0u;
            const uint32_t sa = (uses & TSR_SOP_USES_SA) ? rng_below(&t.r, 16) : 0u;
            const uint32_t sb = (uses & TSR_SOP_USES_SB) ? (round & 3 ? rng_below(&t.r, 16) : sa) : 0u;
            const uint32_t imm = (uses & TSR_SOP_USES_IMM) ? rand_word(&t.r) : 0u;
            const uint32_t addr = (uses & TSR_SOP_USES_ADDR) ? (round == 0 ? SPM_BYTES / 4u - 1u : rng_below(&t.r, SPM_BYTES / 4u)) : 0u;
            char what[48];
            snprintf(what, sizeof what, "SOP sub %u round %d", sub, round);
            if (tw_op(&t, e_sop(sub, sd, sa, sb, imm, addr), what) != 0) continue;
            uint32_t * word = W(t.exp, 0) + addr;
            const uint32_t v = tsr_ref_sop(sub, t.s[sa], t.s[sb], imm, *word);
            if (sub == TSR_SOP_SUB_STS) *word = v;
            else t.s[sd] = v;
            tw_same(&t, what);
        }
    }
    tw_free(&t);
}

static void test_dqrow(void) {
    twin t;
    tw_init(&t, CAPS_ALL, 0x5EED0006u);
    typedef struct { uint32_t fmt, K, rows, ss_extra, ds_extra; } dcase;
    static const dcase cs[] = {
        { TSR_FMT_Q8_0, 64, 1, 0, 0 }, { TSR_FMT_Q8_0, 96, 3, 1, 2 }, { TSR_FMT_Q8_0, 1024, 2, 0, 1 },
        { TSR_FMT_Q4_K, 256, 1, 0, 0 }, { TSR_FMT_Q4_K, 512, 3, 2, 0 }, { TSR_FMT_Q8_0, 32, 0, 0, 0 },
    };
    const uint32_t SRC = 1000, D = 6000;
    for (size_t k = 0; k < sizeof(cs) / sizeof(cs[0]); ++k) {
        const dcase * c = &cs[k];
        const uint32_t rb = tsr_row_beats(c->fmt, c->K), L = tsr_ref_vu_lines(c->K);
        const uint32_t ss = c->ss_extra ? rb + c->ss_extra : 0u, ds = c->ds_extra ? L + c->ds_extra : 0u;
        const uint32_t ssp = ss ? ss : rb, dsp = ds ? ds : L;
        tw_rand(&t, D, 4u * (L + 3u));
        const size_t row_bytes = (size_t)(c->K / tsr_fmt_block_size(c->fmt)) * tsr_fmt_ggml_block_bytes(c->fmt);
        uint8_t * blocks = (uint8_t *)malloc(row_bytes);
        for (uint32_t r = 0; r < c->rows; ++r) {
            gen_row_blocks(&t.r, c->fmt, c->K, blocks);
            tsr_repack_row(c->fmt, blocks, c->K, tsr_sim_spm(t.sim) + (size_t)(SRC + r * ssp) * 64u);
            tsr_repack_row(c->fmt, blocks, c->K, t.exp + (size_t)(SRC + r * ssp) * 64u);
        }
        free(blocks);
        char what[64];
        snprintf(what, sizeof what, "DQROW case %zu", k);
        if (tw_op(&t, e_dqrow(c->fmt, D, SRC, c->K, c->rows, ss, ds), what) != 0) continue;
        tsr_ref_dqrow(c->fmt, c->K, c->rows, W(t.exp, D), ds, t.exp + (size_t)SRC * 64u, ss);
        tw_same(&t, what);
        // and against tsr_dequant_row directly (the ggml-exact dequantisation, VU §6.8)
        float * row = (float *)malloc(c->K * sizeof(float));
        for (uint32_t r = 0; r < c->rows; ++r) {
            tsr_dequant_row(c->fmt, t.exp + (size_t)(SRC + r * ssp) * 64u, c->K, row);
            CHECK(memcmp(row, SW(t.sim, D + r * dsp), c->K * 4u) == 0, "%s row %u = tsr_dequant_row", what, r);
            for (uint32_t j = c->K; j < 16u * L; ++j) CHECK(SW(t.sim, D + r * dsp)[j] == 0u, "%s tail word %u is 0", what, j);
        }
        free(row);
    }
    tw_free(&t);
}

// ---------------------------------------------------------------------------
// Checks: every code with ERR_INFO, the order, a failing op changes nothing
// ---------------------------------------------------------------------------
static void expect_err(twin * t, tsr_insn i, uint32_t code, uint32_t info, const char * what) {
    uint32_t inf = 0xDEADBEEFu;
    const uint32_t e = tsr_sim_op(t->sim, i.w, &inf);
    CHECK(e == code && inf == info, "%s: code %u info 0x%x (want %u 0x%x)", what, e, inf, code, info);
    tw_same(t, what);
}
// The checker alone (no execution): code and ERR_INFO.
static uint32_t check_only(tsr_sim * s, tsr_insn i, uint32_t * info, tsr_extents * ext) {
    const tsr_op_handler * h = tsr_sim_op_handler((uint8_t)(i.w[0] & 0xFFu));
    if (ext) memset(ext, 0, sizeof *ext);
    uint32_t inf = 0;
    const uint32_t e = h ? h->check(s, i.w, ext, &inf) : TSR_ERR_ILLEGAL_OP;
    if (info) *info = inf;
    return e;
}
static void expect_ok(twin * t, tsr_insn i, const char * what) {
    uint32_t info = 0;
    const uint32_t e = check_only(t->sim, i, &info, NULL);
    CHECK(e == TSR_ERR_NONE, "%s: passes the checker (got %u info 0x%x)", what, e, info);
}
static tsr_insn with(tsr_insn i, unsigned word, uint32_t or_bits) { i.w[word] |= or_bits; return i; }

static void test_checks(void) {
    twin t;
    tw_init(&t, CAPS_ALL, 0x5EED0007u);
    const uint32_t IL = TSR_ERR_ILLEGAL_OP, OP = TSR_ERR_OPERAND, SR = TSR_ERR_SPM_RANGE;
    const uint32_t N = SPM_LINES;

    // VOP
    const tsr_insn add = e_vop(TSR_VOP_FUNC_ADD, 0, 0, 10, 20, 30, 0, 40, 3, 3, 3, 3);   // L = 3
    expect_ok(&t, add, "VOP base");
    expect_err(&t, with(add, 0, 1u << 12), IL, 0, "VOP flags");
    expect_err(&t, with(add, 0, 1u << 28), IL, 0, "VOP x16[12]");
    expect_err(&t, with(add, 1, 1u << 24), IL, 0, "VOP d[24]");
    expect_err(&t, with(add, 5, 1u << 24), IL, 0, "VOP m[24]");
    expect_err(&t, e_vop(TSR_VOP_FUNC_N, 0, 0, 10, 20, 30, 0, 40, 3, 3, 3, 3), IL, 0, "VOP func 25");
    expect_err(&t, e_vop(0xFF, 0, 0, 10, 20, 30, 0, 40, 3, 3, 3, 3), IL, 0, "VOP func 0xFF");
    expect_err(&t, e_vop(TSR_VOP_FUNC_MULC, 3, 0, 10, 20, 0, 0, 40, 3, 3, 3, 0), IL, 0, "VOP cmode 3");
    expect_err(&t, e_vop(TSR_VOP_FUNC_MULC, 0, 0, 10, 20, 0, 0, 40, 3, 3, 3, 0), IL, 0, "VOP MULC without c");
    expect_err(&t, e_vop(TSR_VOP_FUNC_ADD, 1, 0, 10, 20, 30, 0, 40, 3, 3, 3, 3), IL, 0, "VOP ADD with cmode 1");
    expect_err(&t, e_vop(TSR_VOP_FUNC_MULC, 2, 1, 10, 20, 0, 40, 40, 3, 3, 3, 0), IL, 0, "VOP si with cmode 2");
    expect_err(&t, e_vop(TSR_VOP_FUNC_ADD, 0, 1, 10, 20, 30, 0, 40, 3, 3, 3, 3), IL, 0, "VOP si with cmode 0");
    expect_err(&t, e_vop(TSR_VOP_FUNC_MULC, 1, 1, 10, 20, 0, 40, 40, 3, 3, 3, 0), IL, 0, "VOP c with cmode 1");
    expect_err(&t, e_vop(TSR_VOP_FUNC_SILU, 0, 0, 10, 20, 30, 0, 40, 3, 3, 3, 0), IL, 0, "VOP SILU with b");
    expect_err(&t, e_vop(TSR_VOP_FUNC_SILU, 0, 0, 10, 20, 0, 0, 40, 3, 3, 3, 1), IL, 0, "VOP SILU with b_stride");
    expect_err(&t, e_vop(TSR_VOP_FUNC_SETC, 1, 0, 10, 20, 0, 0, 40, 3, 3, 0, 0), IL, 0, "VOP SETC with a");
    expect_err(&t, e_vop(TSR_VOP_FUNC_SETC, 1, 0, 10, 0, 0, 0, 40, 3, 3, 2, 0), IL, 0, "VOP SETC with a_stride");
    expect_err(&t, e_vop(TSR_VOP_FUNC_ADD, 0, 0, 10, 20, 30, 0, 40, 3, 2, 3, 3), OP, 2, "VOP d_stride < L(m)");
    expect_ok(&t, e_vop(TSR_VOP_FUNC_ADD, 0, 0, 10, 20, 30, 0, 40, 1, 0, 0, 0), "VOP rows 1, strides unused");
    expect_err(&t, e_vop(TSR_VOP_FUNC_ADD, 0, 0, N, 20, 30, 0, 40, 3, 2, 3, 3), OP, 2, "VOP OPERAND before SPM_RANGE");
    expect_err(&t, e_vop(TSR_VOP_FUNC_N, 0, 0, 10, 20, 30, 0, 40, 3, 2, 3, 3), IL, 0, "VOP ILLEGAL_OP before OPERAND");
    expect_ok(&t, e_vop(TSR_VOP_FUNC_ADD, 0, 0, N - 9, 20, 30, 0, 40, 3, 3, 3, 3), "VOP d ends at the limit");
    expect_err(&t, e_vop(TSR_VOP_FUNC_ADD, 0, 0, N - 8, 20, 30, 0, 40, 3, 3, 3, 3), SR, N - 8, "VOP d range");
    expect_err(&t, e_vop(TSR_VOP_FUNC_ADD, 0, 0, 10, N - 4, N, 0, 40, 3, 3, 3, 3), SR, N - 4, "VOP a range (a before b)");
    expect_err(&t, e_vop(TSR_VOP_FUNC_ADD, 0, 0, 10, 20, N - 6, 0, 40, 3, 3, 3, 3), SR, N - 6, "VOP b range");
    expect_err(&t, e_vop(TSR_VOP_FUNC_ADD, 0, 0, 10, 20, N - 1, 0, 40, 3, 3, 3, 0), SR, N - 1, "VOP b range, stride 0");
    expect_err(&t, e_vop(TSR_VOP_FUNC_MULC, 2, 0, 10, 20, 0, N, 40, 3, 3, 3, 0), SR, N, "VOP c range");
    expect_ok(&t, e_vop(TSR_VOP_FUNC_MULC, 2, 0, 10, 20, 0, N - 1, 40, 3, 3, 3, 0), "VOP c vector of 3 rows: 1 line");
    expect_err(&t, e_vop(TSR_VOP_FUNC_MULC, 2, 0, 10, 20, 0, N - 1, 40, 17, 3, 3, 0), SR, N - 1, "VOP c of 17 rows: 2 lines");
    expect_ok(&t, e_vop(TSR_VOP_FUNC_ADD, 0, 0, N, N, N, 0, 0, 3, 3, 3, 3), "VOP m = 0: extent 0 at the limit");
    expect_err(&t, e_vop(TSR_VOP_FUNC_ADD, 0, 0, N + 1, 20, 30, 0, 0, 3, 3, 3, 3), SR, N + 1, "VOP m = 0 still range-checked");

    // RED
    const tsr_insn red = e_red(TSR_RED_KIND_SUM, 0, 0, TSR_RED_POST_RCP, 0, 0, 10, 20, 40, 3, 3);
    expect_ok(&t, red, "RED base");
    expect_err(&t, e_red(3, 0, 0, 0, 0, 0, 10, 20, 40, 3, 3), IL, 0, "RED kind 3");
    expect_err(&t, e_red(TSR_RED_KIND_SUM, 0, 0, 3, 0, 0, 10, 20, 40, 3, 3), IL, 0, "RED post 3");
    expect_err(&t, e_red(TSR_RED_KIND_MAX, 0, 0, TSR_RED_POST_RCP, 0, 0, 10, 20, 40, 3, 3), IL, 0, "RED MAX with a post-op");
    expect_err(&t, e_red(TSR_RED_KIND_SUM, 0, 1, 0, 0, 0, 10, 20, 40, 3, 3), IL, 0, "RED sd without TO_S");
    expect_err(&t, e_red(TSR_RED_KIND_SUM, 1, 1, 0, 0, 0, 10, 20, 40, 1, 3), IL, 0, "RED d with TO_S");
    expect_err(&t, e_red(TSR_RED_KIND_SUM, 0, 0, TSR_RED_POST_RCP, 1, 0, 10, 20, 40, 3, 3), IL, 0, "RED sk without RMS");
    expect_err(&t, e_red(TSR_RED_KIND_SUMSQ, 0, 0, 0, 0, 2, 10, 20, 40, 3, 3), IL, 0, "RED se without RMS");
    expect_err(&t, with(red, 3, 1u), IL, 0, "RED w3");
    expect_err(&t, with(red, 4, 1u << 31), IL, 0, "RED w4");
    expect_err(&t, with(red, 6, 1u << 16), IL, 0, "RED w6[16]");
    expect_err(&t, with(red, 7, 1u << 31), IL, 0, "RED w7[31]");
    expect_err(&t, with(red, 0, 1u << 22), IL, 0, "RED x16[6]");
    expect_err(&t, with(red, 0, 1u << 13), IL, 0, "RED w0[13]");
    expect_err(&t, e_red(TSR_RED_KIND_SUM, 1, 1, 0, 0, 0, 0, 20, 40, 2, 3), OP, 2, "RED TO_S rows 2");
    expect_err(&t, e_red(TSR_RED_KIND_SUM, 1, 1, 0, 0, 0, 0, 20, 40, 0, 3), OP, 0, "RED TO_S rows 0");
    expect_ok(&t, e_red(TSR_RED_KIND_SUM, 1, 1, TSR_RED_POST_RMS, 2, 3, 0, N - 3, 40, 1, 999), "RED TO_S a ends at the limit");
    expect_err(&t, e_red(TSR_RED_KIND_SUM, 0, 0, 0, 0, 0, N - 1, 20, 40, 17, 3), SR, N - 1, "RED d range (17 rows: 2 lines)");
    expect_err(&t, e_red(TSR_RED_KIND_SUM, 0, 0, 0, 0, 0, 10, N - 8, 40, 3, 3), SR, N - 8, "RED a range");
    expect_err(&t, e_red(TSR_RED_KIND_SUM, 1, 1, 0, 0, 0, 0, N + 1, 0, 1, 3), SR, N + 1, "RED TO_S m = 0: a still checked");

    // ROPE (head_dim 64, n_dims 64, 2 heads: H = 8)
    rope_f rf = { TSR_ROPE_MODE_NEOX, 0, 0, 0, 1, 2, 0, 100, 200, 300, 0, 64, 64, 2, 4, 0, 0 };
    expect_ok(&t, e_rope(&rf), "ROPE base");
    { rope_f g = rf; g.mode = 2; expect_err(&t, e_rope(&g), IL, 0, "ROPE mode 2"); }
    expect_err(&t, with(e_rope(&rf), 0, 1u << 15), IL, 0, "ROPE w0[15]");
    expect_err(&t, with(e_rope(&rf), 0, 1u << 28), IL, 0, "ROPE x16[12]");
    expect_err(&t, with(e_rope(&rf), 3, 1u << 24), IL, 0, "ROPE pos[24]");
    { rope_f g = rf; g.atab = 1; g.ff = 1; g.sts = 0; g.table = 400; expect_err(&t, e_rope(&g), IL, 0, "ROPE ATAB with FF"); }
    { rope_f g = rf; g.atab = 1; g.fs = 1; g.sts = 0; g.sfs = 3; g.table = 400; expect_err(&t, e_rope(&g), IL, 0, "ROPE ATAB with FS"); }
    { rope_f g = rf; g.atab = 1; g.table = 400; expect_err(&t, e_rope(&g), IL, 0, "ROPE ATAB with sts"); }
    { rope_f g = rf; g.atab = 1; g.ff = 1; g.sts = 0; g.table = 400; g.hd = 24;
      expect_err(&t, e_rope(&g), IL, 0, "ROPE ATAB/FF ILLEGAL_OP before OPERAND"); }
    { rope_f g = rf; g.sfs = 3; expect_err(&t, e_rope(&g), IL, 0, "ROPE sfs without FS"); }
    { rope_f g = rf; g.table = 400; expect_err(&t, e_rope(&g), IL, 0, "ROPE table without FF / ATAB"); }
    { rope_f g = rf; g.atab = 1; g.sts = 0; g.table = 400; expect_ok(&t, e_rope(&g), "ROPE ATAB"); }
    { rope_f g = rf; g.hd = 0; g.nd = 0; expect_err(&t, e_rope(&g), OP, 0, "ROPE head_dim 0"); }
    { rope_f g = rf; g.hd = 24; expect_err(&t, e_rope(&g), OP, 24, "ROPE head_dim 24"); }
    { rope_f g = rf; g.hd = 1040; expect_err(&t, e_rope(&g), OP, 1040, "ROPE head_dim 1040"); }
    { rope_f g = rf; g.nd = 0; expect_err(&t, e_rope(&g), OP, 0, "ROPE n_dims 0"); }
    { rope_f g = rf; g.nd = 96; expect_err(&t, e_rope(&g), OP, 96, "ROPE n_dims > head_dim"); }
    { rope_f g = rf; g.nd = 48; expect_err(&t, e_rope(&g), OP, 48, "ROPE NEOX n_dims 48"); }
    { rope_f g = rf; g.mode = TSR_ROPE_MODE_NORM; g.nd = 24; expect_err(&t, e_rope(&g), OP, 24, "ROPE NORM n_dims 24"); }
    { rope_f g = rf; g.mode = TSR_ROPE_MODE_NORM; g.nd = 48; expect_ok(&t, e_rope(&g), "ROPE NORM n_dims 48"); }
    { rope_f g = rf; g.ds = 7; expect_err(&t, e_rope(&g), OP, 7, "ROPE d_stride < H"); }
    { rope_f g = rf; g.ds = 7; g.T = 2; expect_err(&t, e_rope(&g), OP, 7, "ROPE T = 2, d_stride = H - 1"); }
    { rope_f g = rf; g.ds = 8; g.T = 2; expect_ok(&t, e_rope(&g), "ROPE T = 2, d_stride = H"); }
    { rope_f g = rf; g.ds = 7; g.T = 1; expect_ok(&t, e_rope(&g), "ROPE T = 1: d_stride unused"); }
    { rope_f g = rf; g.ds = 7; g.nh = 0; expect_ok(&t, e_rope(&g), "ROPE n_heads = 0: d_stride unused"); }
    { rope_f g = rf; g.ss = 3; expect_ok(&t, e_rope(&g), "ROPE overlapping source tokens"); }
    { rope_f g = rf; g.d = N - 31; expect_err(&t, e_rope(&g), SR, N - 31, "ROPE d range"); }
    { rope_f g = rf; g.d = N - 32; expect_ok(&t, e_rope(&g), "ROPE d ends at the limit"); }
    { rope_f g = rf; g.src = N - 20; g.ss = 4; expect_ok(&t, e_rope(&g), "ROPE src ends at the limit"); }
    { rope_f g = rf; g.src = N - 19; g.ss = 4; expect_err(&t, e_rope(&g), SR, N - 19, "ROPE src range"); }
    { rope_f g = rf; g.pos = N; expect_err(&t, e_rope(&g), SR, N, "ROPE pos range"); }
    { rope_f g = rf; g.ff = 1; g.table = N - 1; g.nd = 64; expect_err(&t, e_rope(&g), SR, N - 1, "ROPE table range"); }
    { rope_f g = rf; g.ff = 1; g.table = N - 1; g.nd = 32; expect_ok(&t, e_rope(&g), "ROPE table of 16 entries: 1 line"); }
    { rope_f g = rf; g.T = 0; g.d = N; g.src = N; g.pos = N; expect_ok(&t, e_rope(&g), "ROPE T = 0: extents 0"); }

    // TOPK
    const tsr_insn tk = e_topk(9, 10, 20, 100, 3, 0, 7);
    expect_ok(&t, tk, "TOPK base");
    expect_err(&t, e_topk(0, 10, 20, 100, 3, 0, 7), OP, 0, "TOPK k 0");
    expect_err(&t, e_topk(17, 10, 20, 100, 3, 0, 7), OP, 17, "TOPK k 17");
    expect_err(&t, e_topk(31, 10, 20, 100, 3, 0, 7), OP, 31, "TOPK k 31");
    expect_err(&t, e_topk(9, 10, 20, 100, 3, 1, 7), OP, 1, "TOPK d_stride < ceil(k/8)");
    expect_ok(&t, e_topk(9, 10, 20, 100, 1, 1, 7), "TOPK rows 1: d_stride unused");
    expect_err(&t, with(tk, 0, 1u << 21), IL, 0, "TOPK x16[5]");
    expect_err(&t, with(tk, 0, 1u << 8), IL, 0, "TOPK sub");
    expect_err(&t, with(tk, 3, 1u), IL, 0, "TOPK w3");
    expect_err(&t, with(tk, 7, 1u << 16), IL, 0, "TOPK w7[16]");
    expect_err(&t, e_topk(9, N - 5, 20, 100, 3, 0, 7), SR, N - 5, "TOPK d range");
    expect_err(&t, e_topk(9, 10, N - 20, 100, 3, 0, 7), SR, N - 20, "TOPK a range");

    // SOP
    expect_err(&t, e_sop(15, 1, 0, 0, 0, 0), IL, 0, "SOP sub 15");
    expect_err(&t, e_sop(TSR_SOP_SUB_MOVI, 1, 2, 0, 5, 0), IL, 0, "SOP MOVI with sa");
    expect_err(&t, e_sop(TSR_SOP_SUB_MOVI, 1, 0, 0, 5, 4), IL, 0, "SOP MOVI with addr");
    expect_err(&t, e_sop(TSR_SOP_SUB_ADD, 1, 2, 3, 5, 0), IL, 0, "SOP ADD with imm");
    expect_err(&t, e_sop(TSR_SOP_SUB_SQRT, 1, 2, 3, 0, 0), IL, 0, "SOP SQRT with sb");
    expect_err(&t, e_sop(TSR_SOP_SUB_STS, 1, 2, 0, 0, 4), IL, 0, "SOP STS with sd");
    expect_err(&t, e_sop(TSR_SOP_SUB_LDS, 1, 2, 0, 0, 4), IL, 0, "SOP LDS with sa");
    expect_err(&t, with(e_sop(TSR_SOP_SUB_ADD, 1, 2, 3, 0, 0), 1, 1u << 12), IL, 0, "SOP w1[12]");
    expect_err(&t, with(e_sop(TSR_SOP_SUB_LDS, 1, 0, 0, 0, 4), 3, 1u << 28), IL, 0, "SOP w3[28]");
    expect_err(&t, with(e_sop(TSR_SOP_SUB_ADD, 1, 2, 3, 0, 0), 4, 1u), IL, 0, "SOP w4");
    expect_err(&t, with(e_sop(TSR_SOP_SUB_ADD, 1, 2, 3, 0, 0), 0, 1u << 16), IL, 0, "SOP x16");
    expect_ok(&t, e_sop(TSR_SOP_SUB_LDS, 1, 0, 0, 0, SPM_BYTES / 4u - 1u), "SOP LDS last word");
    expect_err(&t, e_sop(TSR_SOP_SUB_LDS, 1, 0, 0, 0, SPM_BYTES / 4u), SR, SPM_BYTES / 4u, "SOP LDS range");
    expect_err(&t, e_sop(TSR_SOP_SUB_STS, 0, 1, 0, 0, 0x0FFFFFFFu), SR, 0x0FFFFFFFu, "SOP STS range");

    // DQROW
    const tsr_insn dq = e_dqrow(TSR_FMT_Q8_0, 10, 100, 64, 2, 0, 0);
    expect_ok(&t, dq, "DQROW base");
    expect_err(&t, with(dq, 0, 1u << 16), IL, 0, "DQROW x16");
    expect_err(&t, with(e_dqrow(2, 10, 100, 0, 2, 0, 0), 0, 1u << 12), IL, 0, "DQROW flags: ILLEGAL_OP before BAD_FMT");
    expect_err(&t, with(dq, 3, 1u), IL, 0, "DQROW w3");
    expect_err(&t, with(dq, 6, 1u << 16), IL, 0, "DQROW w6[16]");
    expect_err(&t, e_dqrow(2, 10, 100, 64, 2, 0, 0), TSR_ERR_BAD_FMT, 0, "DQROW fmt 2");
    expect_err(&t, e_dqrow(15, 10, 100, 0, 2, 0, 0), TSR_ERR_BAD_FMT, 0, "DQROW BAD_FMT before K_RANGE");
    expect_err(&t, e_dqrow(TSR_FMT_Q8_0, 10, 100, 0, 2, 0, 0), TSR_ERR_K_RANGE, 0, "DQROW K 0");
    expect_err(&t, e_dqrow(TSR_FMT_Q8_0, 10, 100, 48, 2, 0, 0), TSR_ERR_K_RANGE, 0, "DQROW K 48");
    expect_err(&t, e_dqrow(TSR_FMT_Q4_K, 10, 100, 128, 2, 0, 0), TSR_ERR_K_RANGE, 0, "DQROW Q4_K K 128");
    expect_err(&t, e_dqrow(TSR_FMT_Q8_0, 10, 100, 32768u + 32u, 1, 0, 0), TSR_ERR_K_RANGE, 0, "DQROW K > K_MAX");
    expect_err(&t, e_dqrow(TSR_FMT_Q8_0, 10, 100, 0, 2, 0, 1), TSR_ERR_K_RANGE, 0, "DQROW K_RANGE before OPERAND");
    expect_err(&t, e_dqrow(TSR_FMT_Q8_0, 10, 100, 64, 2, 0, 3), OP, 3, "DQROW d_stride < L(K)");
    expect_err(&t, e_dqrow(TSR_FMT_Q8_0, N - 7, 100, 64, 2, 0, 0), SR, N - 7, "DQROW d range");
    const uint32_t rb64 = tsr_row_beats(TSR_FMT_Q8_0, 64);
    expect_err(&t, e_dqrow(TSR_FMT_Q8_0, 10, N - 2u * rb64 + 1u, 64, 2, 0, 0), SR, N - 2u * rb64 + 1u, "DQROW src range");
    expect_ok(&t, e_dqrow(TSR_FMT_Q8_0, 10, N - 2u * rb64, 64, 2, 0, 0), "DQROW src ends at the limit");

    // VMASK and the reserved VU opcodes: refused by the issue path (ILLEGAL_OP), with any words.
    tsr_insn vm = {{ TSR_OP_VMASK, 10, 0, 0, 0, 16, 1u | (1u << 16), 1 }};
    expect_err(&t, vm, IL, 0, "VMASK");
    static const uint8_t rsvd[] = { 0x13, 0x3A, 0x3B, 0x3C, 0x3D, 0x3E, 0x3F, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47 };
    for (size_t k = 0; k < sizeof rsvd; ++k) {
        tsr_insn r0 = {{ rsvd[k], 0, 0, 0, 0, 0, 0, 0 }};
        char what[40];
        snprintf(what, sizeof what, "reserved VU opcode 0x%02X", rsvd[k]);
        expect_err(&t, r0, IL, 0, what);
    }
    tw_free(&t);

    // CAPS gating (EX §7.5, tsr_sim_accept_op): an opcode whose CAPS bits are missing is ILLEGAL_OP.
    const uint32_t impl = tsr_sim_caps_implemented();
    CHECK((impl & CAPS_ALL) == CAPS_ALL, "CAPS implemented include VU, TOPK, DQROW (0x%x)", impl);
    CHECK((impl & TSR_CAPS_VMASK) == 0u, "VMASK is not implemented");
    {
        tsr_config c = cfg_u200(8);
        c.caps = CAPS_ALL | TSR_CAPS_VMASK;
        tsr_sim * s = tsr_sim_create(&c, NULL);
        CHECK(s == NULL, "a config asking for VMASK is refused");
        tsr_sim_destroy(s);
    }
    {
        tsr_sim * s = mk_sim(TSR_CAPS_VU, 8);
        uint32_t info;
        CHECK(tsr_sim_op(s, e_topk(1, 10, 20, 16, 1, 0, 0).w, &info) == IL, "TOPK without CAPS.TOPK");
        CHECK(tsr_sim_op(s, e_dqrow(0, 10, 100, 64, 1, 0, 0).w, &info) == IL, "DQROW without CAPS.DQROW");
        CHECK(tsr_sim_op(s, e_sop(TSR_SOP_SUB_MOVI, 1, 0, 0, F_ONE, 0).w, &info) == 0, "SOP with CAPS.VU");
        tsr_sim_destroy(s);
        s = mk_sim(0, 8);
        CHECK(tsr_sim_op(s, e_sop(TSR_SOP_SUB_MOVI, 1, 0, 0, F_ONE, 0).w, &info) == IL, "SOP without CAPS.VU");
        CHECK(tsr_sim_op(s, e_vop(TSR_VOP_FUNC_COPY, 0, 0, 10, 20, 0, 0, 16, 1, 0, 0, 0).w, &info) == IL, "VOP without CAPS.VU");
        CHECK(tsr_sim_csr_read(s, TSR_CSR_CFG3) == 0u, "CFG3 VU fields 0 without the VU");
        tsr_sim_destroy(s);
        s = mk_sim(CAPS_ALL, 8);
        CHECK(TSR_GET(tsr_sim_csr_read(s, TSR_CSR_CFG3), TSR_CFG3_VU_FRONT) == TSR_VU_FRONT_LANES &&
              TSR_GET(tsr_sim_csr_read(s, TSR_CSR_CFG3), TSR_CFG3_VU_BACK) == TSR_VU_BACK_LANES, "CFG3 lanes");
        CHECK(tsr_sim_csr_read(s, TSR_CSR_CAPS) == CAPS_ALL, "CAPS register = the config");
        tsr_sim_destroy(s);
    }
}


// ---------------------------------------------------------------------------
// GDN (VU §7.12): a head of rows x m against tsr_ref_gdn applied to a copy of the SPM, out of place and in
// place, with the specials of rand_word in the tile (Inf, NaN, subnormals, huge and tiny exponents) and with
// finite model-like data, the tail words of o written 0, every untouched line unchanged. d or o on the
// vectors changes nothing: the vectors are read before the first write.
// ---------------------------------------------------------------------------
static void test_gdn(void) {
    twin t;
    tw_init(&t, CAPS_ALL, 0x5EED00A1u);
    static const uint32_t ms[] = { 64, 128, 256 }, rowsv[] = { 1, 7, 16, 17, 100 };
    const uint32_t A = 1000, D = 5000, K = 200, Q = 300, V = 400, O = 600;
    for (size_t mi = 0; mi < 3; ++mi) {
        for (size_t ri = 0; ri < 5; ++ri) {
            for (int mode = 0; mode < 4; ++mode) {
                const uint32_t m = ms[mi], rows = rowsv[ri], nl = m / 16u, lines = rows * nl;
                const int inplace = (mode & 1) != 0, tame = (mode & 2) != 0;
                tw_rand(&t, A, lines);
                tw_rand(&t, D, lines);
                tw_rand(&t, K, nl);
                tw_rand(&t, Q, nl);
                tw_rand(&t, V, tsr_ref_vu_lines(rows));
                tw_rand(&t, O, tsr_ref_vu_lines(rows));
                if (tame) {                                       // finite, comparable magnitudes
                    for (uint32_t i = 0; i < 16u * lines; ++i) tw_put(&t, A + i / 16u, i % 16u, rand_norm(&t.r, 1.0));
                    for (uint32_t i = 0; i < m; ++i) {
                        tw_put(&t, K + i / 16u, i % 16u, rand_norm(&t.r, 0.2));
                        tw_put(&t, Q + i / 16u, i % 16u, rand_norm(&t.r, 0.2));
                    }
                    for (uint32_t i = 0; i < rows; ++i) tw_put(&t, V + i / 16u, i % 16u, rand_norm(&t.r, 1.0));
                }
                const uint32_t alpha = tame ? f2u(0.97f) : rand_word(&t.r), beta = tame ? f2u(0.4f) : rand_word(&t.r);
                tw_movi(&t, 3, alpha);
                tw_movi(&t, 9, beta);
                const uint32_t d = inplace ? A : D;
                char what[96];
                snprintf(what, sizeof what, "GDN m %u rows %u %s%s", m, rows, inplace ? "in place" : "to D", tame ? " tame" : "");
                if (tw_op(&t, e_gdn(3, 9, d, A, K, Q, V, O, m, rows), what) != 0) continue;
                uint32_t o[1040];
                tsr_ref_gdn(m, rows, alpha, beta, W(t.exp, K), W(t.exp, Q), W(t.exp, V), W(t.exp, A), W(t.exp, d), o);
                memcpy(W(t.exp, O), o, 64u * tsr_ref_vu_lines(rows));
                tw_same(&t, what);
            }
        }
    }
    {
        const uint32_t m = 64, rows = 20;                         // d on k, o on v: the vectors are read first
        tw_rand(&t, A, rows * 4u);
        tw_rand(&t, K, 4);
        tw_rand(&t, Q, 4);
        tw_rand(&t, V, 2);
        const uint32_t alpha = f2u(0.9f), beta = f2u(0.5f);
        tw_movi(&t, 1, alpha);
        tw_movi(&t, 2, beta);
        uint32_t k[64], q[64], v[32], o[32], dd[1280];
        memcpy(k, W(t.exp, K), sizeof k);
        memcpy(q, W(t.exp, Q), sizeof q);
        memcpy(v, W(t.exp, V), 4u * rows);
        tw_op(&t, e_gdn(1, 2, 1100, A, K, Q, V, V, m, rows), "GDN o on v");
        tsr_ref_gdn(m, rows, alpha, beta, k, q, v, W(t.exp, A), dd, o);
        memcpy(W(t.exp, 1100), dd, 4u * rows * m);
        memcpy(W(t.exp, V), o, 64u * 2u);
        tw_same(&t, "GDN o on v");
    }
    tw_free(&t);
}

static void test_gdn_checks(void) {
    twin t;
    tw_init(&t, CAPS_ALL, 0x5EED00A2u);
    const uint32_t IL = TSR_ERR_ILLEGAL_OP, OP = TSR_ERR_OPERAND, SR = TSR_ERR_SPM_RANGE, N = SPM_LINES;
    // m 128 (8 lines per row), 20 rows: the tile is 160 lines, v and o two lines
    const tsr_insn g = e_gdn(1, 2, 3000, 1000, 200, 300, 400, 500, 128, 20);
    expect_ok(&t, g, "GDN base");
    expect_ok(&t, e_gdn(1, 2, 1000, 1000, 200, 300, 400, 500, 128, 20), "GDN in place");
    expect_ok(&t, e_gdn(1, 1, 3000, 1000, 200, 300, 400, 500, 128, 20), "GDN alpha and beta in one register");
    expect_ok(&t, e_gdn(1, 2, 3000, 1000, 200, 200, 400, 500, 64, 1), "GDN k and q equal");
    expect_ok(&t, e_gdn(1, 2, 9000, 1000, 200, 300, 400, 500, 64, 1024), "GDN m 64, rows 1,024");
    expect_ok(&t, e_gdn(1, 2, 3000, 1000, 200, 300, 400, 500, 256, 64), "GDN m 256, rows 64");
    expect_ok(&t, e_gdn(1, 2, 3000, 1000, 200, 300, 400, 500, 64, 1), "GDN m 64");
    expect_err(&t, with(g, 0, 1u << 12), IL, 0, "GDN flags");
    expect_err(&t, with(g, 0, 1u << 8), IL, 0, "GDN sub");
    expect_err(&t, with(g, 0, 1u << 24), IL, 0, "GDN x16[8]");
    expect_err(&t, with(g, 0, 1u << 31), IL, 0, "GDN x16[15]");
    for (unsigned w = 1; w <= 6; ++w) {
        char what[32];
        snprintf(what, sizeof what, "GDN w%u[24]", w);
        expect_err(&t, with(g, w, 1u << 24), IL, 0, what);
    }
    expect_err(&t, with(e_gdn(1, 2, 3000, 1000, 200, 300, 400, 500, 48, 20), 1, 1u << 24), IL, 0, "GDN ILLEGAL_OP before OPERAND");
    expect_err(&t, e_gdn(1, 2, 3000, 1000, 200, 300, 400, 500, 0, 20), OP, 0, "GDN m 0");
    expect_err(&t, e_gdn(1, 2, 3000, 1000, 200, 300, 400, 500, 32, 20), OP, 32, "GDN m 32 (below the 64 of every build)");
    expect_err(&t, e_gdn(1, 2, 3000, 1000, 200, 300, 400, 500, 48, 20), OP, 48, "GDN m 48");
    expect_err(&t, e_gdn(1, 2, 3000, 1000, 200, 300, 400, 500, 72, 20), OP, 72, "GDN m 72 (not a multiple of 16)");
    expect_err(&t, e_gdn(1, 2, 3000, 1000, 200, 300, 400, 500, 272, 20), OP, 272, "GDN m 272");
    expect_err(&t, e_gdn(1, 2, 3000, 1000, 200, 300, 400, 500, 0xFFF0, 20), OP, 0xFFF0, "GDN m 65,520");
    expect_err(&t, e_gdn(1, 2, 3000, 1000, 200, 300, 400, 500, 72, 0), OP, 72, "GDN m before rows");
    expect_err(&t, e_gdn(1, 2, 3000, 1000, 200, 300, 400, 500, 128, 0), OP, 0, "GDN rows 0");
    expect_err(&t, e_gdn(1, 2, 3000, 1000, 200, 300, 400, 500, 128, 1025), OP, 1025, "GDN rows 1,025");
    expect_err(&t, e_gdn(1, 2, 3000, 1000, 200, 300, 400, 500, 128, 0xFFFF), OP, 0xFFFF, "GDN rows 65,535");
    expect_err(&t, e_gdn(1, 2, 1040, 1000, 200, 300, 400, 500, 128, 20), OP, 1040, "GDN d overlaps a (shifted)");
    expect_err(&t, e_gdn(1, 2, 900, 1000, 200, 300, 400, 500, 128, 20), OP, 900, "GDN d overlaps a (below)");
    expect_err(&t, e_gdn(1, 2, 1159, 1000, 200, 300, 400, 500, 128, 20), OP, 1159, "GDN d overlaps a by one line");
    expect_ok(&t, e_gdn(1, 2, 1160, 1000, 200, 300, 400, 500, 128, 20), "GDN d right after a");
    expect_ok(&t, e_gdn(1, 2, 840, 1000, 200, 300, 400, 500, 128, 20), "GDN d right before a");
    expect_err(&t, e_gdn(1, 2, 3000, 1000, 200, 300, 400, 3000, 128, 20), OP, 3000, "GDN o on d");
    expect_err(&t, e_gdn(1, 2, 3000, 1000, 200, 300, 400, 1159, 128, 20), OP, 1159, "GDN o in a");
    expect_err(&t, e_gdn(1, 2, 3000, 1000, 200, 300, 400, 3159, 128, 20), OP, 3159, "GDN o in d");
    expect_ok(&t, e_gdn(1, 2, 3000, 1000, 200, 300, 400, 3160, 128, 20), "GDN o right after d");
    expect_ok(&t, e_gdn(1, 2, 3000, 1000, 200, 300, 200, 200, 128, 20), "GDN o and v on k");
    expect_err(&t, e_gdn(1, 2, N - 159, 1000, 200, 300, 400, 500, 128, 20), SR, N - 159, "GDN d range");
    expect_ok(&t, e_gdn(1, 2, N - 160, 1000, 200, 300, 400, 500, 128, 20), "GDN d ends at the limit");
    expect_err(&t, e_gdn(1, 2, 3000, N - 159, 200, 300, 400, 500, 128, 20), SR, N - 159, "GDN a range (d before a)");
    expect_err(&t, e_gdn(1, 2, N - 159, N - 159, 200, 300, 400, 500, 128, 20), SR, N - 159, "GDN in place, range");
    expect_err(&t, e_gdn(1, 2, 3000, 1000, N - 7, 300, 400, 500, 128, 20), SR, N - 7, "GDN k range");
    expect_ok(&t, e_gdn(1, 2, 3000, 1000, N - 8, 300, 400, 500, 128, 20), "GDN k ends at the limit");
    expect_err(&t, e_gdn(1, 2, 3000, 1000, 200, N - 7, 400, 500, 128, 20), SR, N - 7, "GDN q range");
    expect_err(&t, e_gdn(1, 2, 3000, 1000, 200, 300, N - 1, 500, 128, 20), SR, N - 1, "GDN v range (20 rows: 2 lines)");
    expect_ok(&t, e_gdn(1, 2, 3000, 1000, 200, 300, N - 1, 500, 128, 16), "GDN v of 16 rows: one line");
    expect_err(&t, e_gdn(1, 2, 3000, 1000, 200, 300, 400, N - 1, 128, 20), SR, N - 1, "GDN o range");
    expect_err(&t, e_gdn(1, 2, N - 159, 1000, 200, 300, N, 500, 128, 20), SR, N - 159, "GDN d before v in field order");
    expect_err(&t, e_gdn(1, 2, N, 1000, 200, 300, 400, 500, 128, 20), SR, N, "GDN d at N");
    tw_free(&t);
}

// The queue path records the checker's codes (EX §8.2): ERR_CODE / ERR_INFO / ERR_W0 of unit 3,
// BAD_DEP's place in the order, a failing op poisons only its own entry.
static uint32_t q_push(tsr_sim * s, tsr_insn i, uint32_t d0, uint32_t d1) { return tsr_sim_push(s, i.w, d0, d1, ALL_WORDS); }

// Completion codes by ticket (queue listener, tsr_sim_int.h §5).
typedef struct { uint32_t ticket[64], code[64], n; } done_log;
static void done_cb(void * ctx, tsr_sim * s, const tsr_q_event * e) {
    (void)s;
    done_log * L = (done_log *)ctx;
    if (e->kind == TSR_QEV_DONE && L->n < 64u) {
        L->ticket[L->n] = e->ticket;
        L->code[L->n++] = e->code;
    }
}
static uint32_t done_code(const done_log * L, uint32_t ticket) {
    for (uint32_t i = 0; i < L->n; ++i) {
        if (L->ticket[i] == ticket) return L->code[i];
    }
    return 0xFFFFFFFFu;
}

static void test_queue_records(void) {
    const tsr_insn bad = e_vop(TSR_VOP_FUNC_ADD, 0, 0, 10, 20, 30, 0, 40, 3, 2, 3, 3);   // OPERAND, info 2
    {
        tsr_sim * s = mk_sim(CAPS_ALL, 8);
        q_push(s, bad, TSR_TICKET_NONE, TSR_TICKET_NONE);
        CHECK(tsr_sim_run_until_idle(s, 1000000), "idle");
        CHECK(tsr_sim_csr_read(s, TSR_CSR_Q_ERR_CODE(TSR_UNIT_VU)) == TSR_ERR_OPERAND, "queue: ERR_CODE OPERAND");
        CHECK(tsr_sim_csr_read(s, TSR_CSR_Q_ERR_INFO(TSR_UNIT_VU)) == 2u, "queue: ERR_INFO = d_stride");
        CHECK(tsr_sim_csr_read(s, TSR_CSR_Q_ERR_W0(TSR_UNIT_VU)) == bad.w[0], "queue: ERR_W0");
        tsr_sim_destroy(s);
    }
    {
        tsr_sim * s = mk_sim(CAPS_ALL, 8);
        q_push(s, bad, TSR_TICKET_NONE, (uint32_t)TSR_UNIT_ATT << 28);          // dep on the absent unit 4
        CHECK(tsr_sim_run_until_idle(s, 1000000), "idle");
        CHECK(tsr_sim_csr_read(s, TSR_CSR_Q_ERR_CODE(TSR_UNIT_VU)) == TSR_ERR_BAD_DEP, "BAD_DEP before OPERAND");
        CHECK(tsr_sim_csr_read(s, TSR_CSR_Q_ERR_INFO(TSR_UNIT_VU)) == 0u, "BAD_DEP info 0");
        tsr_sim_destroy(s);
    }
    {
        tsr_sim * s = mk_sim(CAPS_ALL, 8);
        q_push(s, e_vop(TSR_VOP_FUNC_N, 0, 0, 10, 20, 30, 0, 40, 3, 2, 3, 3), TSR_TICKET_NONE, (uint32_t)TSR_UNIT_ATT << 28);
        CHECK(tsr_sim_run_until_idle(s, 1000000), "idle");
        CHECK(tsr_sim_csr_read(s, TSR_CSR_Q_ERR_CODE(TSR_UNIT_VU)) == TSR_ERR_ILLEGAL_OP, "ILLEGAL_OP before BAD_DEP");
        tsr_sim_destroy(s);
    }
    // DEP_FAILED (a dependency failed on another unit) and ABORTED (the VU is then in error state)
    {
        tsr_sim * s = mk_sim(CAPS_ALL, 8);
        done_log L;
        memset(&L, 0, sizeof L);
        tsr_sim_set_listener(s, done_cb, &L);
        const uint32_t t_ld = q_push(s, tsr_enc_ld(7, 10, 0, 1), TSR_TICKET_NONE, TSR_TICKET_NONE);   // channel 7: ILLEGAL_OP
        const tsr_insn ok = e_vop(TSR_VOP_FUNC_COPY, 0, 0, 10, 20, 0, 0, 16, 1, 0, 0, 0);
        const uint32_t t1 = q_push(s, ok, t_ld, TSR_TICKET_NONE);
        const uint32_t t2 = q_push(s, ok, TSR_TICKET_NONE, TSR_TICKET_NONE);
        CHECK(tsr_sim_run_until_idle(s, 1000000), "idle");
        CHECK(done_code(&L, t1) == TSR_ERR_DEP_FAILED, "VOP after a failed LD: DEP_FAILED (%u)", done_code(&L, t1));
        CHECK(done_code(&L, t2) == TSR_ERR_ABORTED, "next VOP: ABORTED, the VU in error state (%u)", done_code(&L, t2));
        CHECK(tsr_sim_csr_read(s, TSR_CSR_Q_ERR_CODE(TSR_UNIT_VU)) == TSR_ERR_DEP_FAILED, "VU record: DEP_FAILED");
        tsr_sim_destroy(s);
    }
}

// ---------------------------------------------------------------------------
// Extents (tsr_sim_int.h §1)
// ---------------------------------------------------------------------------
static int ext_is(const tsr_extents * x, uint32_t k, uint8_t kind, uint8_t wr, uint64_t lo, uint64_t hi) {
    return k < x->n && x->e[k].kind == kind && x->e[k].write == wr && x->e[k].lo == lo && x->e[k].hi == hi &&
           x->e[k].ch == 0u;
}

static void test_extents(void) {
    tsr_sim * s = mk_sim(CAPS_ALL, 8);
    tsr_extents x;
    uint32_t info;
    const uint8_t SP = TSR_EXT_SPM, SG = TSR_EXT_SREG, R = TSR_EXT_READ, Wr = TSR_EXT_WRITE;

    check_only(s, e_vop(TSR_VOP_FUNC_MULCB, 1, 5, 10, 100, 200, 0, 40, 4, 3, 5, 0), &info, &x);
    CHECK(x.n == 4 && ext_is(&x, 0, SP, Wr, 10, 22) && ext_is(&x, 1, SP, R, 100, 118) && ext_is(&x, 2, SP, R, 200, 203) &&
          ext_is(&x, 3, SG, R, 5, 6), "VOP MULCB cmode 1: d, a, b (stride 0), s5");
    check_only(s, e_vop(TSR_VOP_FUNC_MULCADD, 2, 0, 10, 100, 200, 300, 16, 20, 1, 1, 1), &info, &x);
    CHECK(x.n == 4 && ext_is(&x, 0, SP, Wr, 10, 30) && ext_is(&x, 1, SP, R, 100, 120) && ext_is(&x, 2, SP, R, 200, 220) &&
          ext_is(&x, 3, SP, R, 300, 302), "VOP MULCADD cmode 2: four SPM extents");
    check_only(s, e_vop(TSR_VOP_FUNC_EXP, 0, 0, 10, 10, 0, 0, 17, 2, 2, 2, 0), &info, &x);
    CHECK(x.n == 2 && ext_is(&x, 0, SP, Wr, 10, 14) && ext_is(&x, 1, SP, R, 10, 14), "VOP in place: d and a both listed");
    check_only(s, e_vop(TSR_VOP_FUNC_MULC, 1, 7, 10, 100, 0, 0, 0, 4, 3, 5, 0), &info, &x);
    CHECK(x.n == 0, "VOP m = 0: no extent, no s read");
    check_only(s, e_red(TSR_RED_KIND_SUMSQ, 1, 1, TSR_RED_POST_RMS, 2, 2, 0, 50, 896, 1, 0), &info, &x);
    CHECK(x.n == 3 && ext_is(&x, 0, SP, R, 50, 106) && ext_is(&x, 1, SG, Wr, 1, 2) && ext_is(&x, 2, SG, R, 2, 3),
          "RED TO_S RMS: a, s1 written, s2 read once (sk = se)");
    check_only(s, e_red(TSR_RED_KIND_SUM, 1, 3, 0, 0, 0, 0, 50, 0, 1, 0), &info, &x);
    CHECK(x.n == 1 && ext_is(&x, 0, SG, Wr, 3, 4), "RED TO_S m = 0: only the s write");
    check_only(s, e_red(TSR_RED_KIND_MAX, 0, 0, 0, 0, 0, 7, 50, 20, 17, 2), &info, &x);
    CHECK(x.n == 2 && ext_is(&x, 0, SP, Wr, 7, 9) && ext_is(&x, 1, SP, R, 50, 84), "RED vector: d ceil(rows/16), a");
    rope_f rf = { TSR_ROPE_MODE_NORM, 1, 1, 0, 1, 2, 3, 100, 200, 300, 400, 64, 64, 2, 4, 9, 10 };
    check_only(s, e_rope(&rf), &info, &x);
    CHECK(x.n == 7 && ext_is(&x, 0, SP, Wr, 100, 138) && ext_is(&x, 1, SP, R, 200, 235) && ext_is(&x, 2, SP, R, 300, 301) &&
          ext_is(&x, 3, SP, R, 400, 402) && ext_is(&x, 4, SG, R, 1, 2) && ext_is(&x, 5, SG, R, 2, 3) && ext_is(&x, 6, SG, R, 3, 4),
          "ROPE FF + FS: d, src, pos, table, sm, sts, sfs");
    rope_f ra = { TSR_ROPE_MODE_NEOX, 0, 0, 1, 4, 0, 0, 100, 100, 300, 400, 64, 64, 2, 1, 0, 0 };
    check_only(s, e_rope(&ra), &info, &x);
    CHECK(x.n == 5 && ext_is(&x, 0, SP, Wr, 100, 108) && ext_is(&x, 1, SP, R, 100, 108) && ext_is(&x, 3, SP, R, 400, 402) &&
          ext_is(&x, 4, SG, R, 4, 5), "ROPE ATAB in place: no sts");
    check_only(s, e_topk(9, 10, 20, 100, 3, 0, 7), &info, &x);
    CHECK(x.n == 2 && ext_is(&x, 0, SP, Wr, 10, 16) && ext_is(&x, 1, SP, R, 20, 41), "TOPK: d, a");
    check_only(s, e_sop(TSR_SOP_SUB_ADD, 3, 3, 4, 0, 0), &info, &x);
    CHECK(x.n == 3 && ext_is(&x, 0, SG, Wr, 3, 4) && ext_is(&x, 1, SG, R, 3, 4) && ext_is(&x, 2, SG, R, 4, 5), "SOP ADD s3 = s3 + s4");
    check_only(s, e_sop(TSR_SOP_SUB_LDS, 6, 0, 0, 0, 1000), &info, &x);
    CHECK(x.n == 2 && ext_is(&x, 0, SP, R, 62, 63) && ext_is(&x, 1, SG, Wr, 6, 7), "SOP LDS: the word's line, s6");
    check_only(s, e_sop(TSR_SOP_SUB_STS, 0, 9, 0, 0, 33), &info, &x);
    CHECK(x.n == 2 && ext_is(&x, 0, SP, Wr, 2, 3) && ext_is(&x, 1, SG, R, 9, 10), "SOP STS");
    check_only(s, e_dqrow(TSR_FMT_Q4_K, 10, 100, 512, 3, 0, 40), &info, &x);
    CHECK(x.n == 2 && ext_is(&x, 0, SP, Wr, 10, 122) && ext_is(&x, 1, SP, R, 100, 100 + 3u * tsr_row_beats(TSR_FMT_Q4_K, 512)),
          "DQROW: d, src");
    tsr_sim_destroy(s);
}

// ---------------------------------------------------------------------------
// Cycle model and PERF
// ---------------------------------------------------------------------------
static uint64_t cyc(tsr_sim * s, tsr_insn i) {
    const tsr_op_handler * h = tsr_sim_op_handler((uint8_t)(i.w[0] & 0xFFu));
    return h->cycles(s, i.w);
}

// A simulator with the VU's lanes of a configuration ("u200" 16 / 16 / 8 lanes, no lane pipe; "k4" 8 / 8 / 4 behind two
// stages: the card image of the fourth link).
static tsr_sim * mk_sim_vu(const char * vu_timing) {
    tsr_config c = cfg_u200(8);
    c.spm_bytes = SPM_BYTES;
    c.ch_bytes = 64ull << 20;
    c.caps = CAPS_ALL;
    tsr_sim_opts o;
    memset(&o, 0, sizeof o);
    o.vu_timing = vu_timing;
    tsr_sim * s = tsr_sim_create_opts(&c, &o);
    if (s == NULL) {
        fprintf(stderr, "tsr_sim_create_opts(vu_timing %s) failed\n", vu_timing);
        exit(1);
    }
    return s;
}

// The cycle model against what the RTL unit takes (SW-80). The numbers are tb/vu test_vu_timing.py's marginal cycles of an
// op that waits for the one before it (RTL on a 4 MiB SPM, nothing else running), at the two lane configurations
// that were built: first the U200 lanes, then the K4 image's.
static void test_cycles_rtl(void) {
    static const char * const cfg[2] = { "u200", "k4" };
    for (int k = 0; k < 2; ++k) {
        tsr_sim * s = mk_sim_vu(cfg[k]);
        const int K = k;
#define EXPECT(what, u200, k4, ...) CHECK(cyc(s, __VA_ARGS__) == (K ? (k4) : (u200)), "%s: %s: %llu, RTL %u", cfg[K], what, \
                                          (unsigned long long)cyc(s, __VA_ARGS__), K ? (k4) : (u200))
        EXPECT("VOP ADD rows 4 L 3", 72, 88, e_vop(TSR_VOP_FUNC_ADD, 0, 0, 10, 100, 200, 0, 48, 4, 3, 3, 3));
        EXPECT("VOP ADD one row L 8", 68, 80, e_vop(TSR_VOP_FUNC_ADD, 0, 0, 10, 100, 200, 0, 128, 1, 0, 0, 0));
        EXPECT("VOP ADD 128 rows L 9", 1212, 2368, e_vop(TSR_VOP_FUNC_ADD, 0, 0, 10, 2000, 4000, 0, 144, 128, 9, 9, 9));
        EXPECT("VOP COPY 16 rows L 8", 187, 319, e_vop(TSR_VOP_FUNC_COPY, 0, 0, 10, 100, 0, 0, 128, 16, 8, 8, 0));
        EXPECT("VOP EXP L 8", 67, 79, e_vop(TSR_VOP_FUNC_EXP, 0, 0, 10, 100, 0, 0, 128, 1, 0, 0, 0));
        EXPECT("VOP EXPSUB L 8", 68, 80, e_vop(TSR_VOP_FUNC_EXPSUB, 0, 0, 10, 100, 200, 0, 128, 1, 0, 0, 0));
        EXPECT("VOP SILU 16 rows L 8", 351, 611, e_vop(TSR_VOP_FUNC_SILU, 0, 0, 10, 100, 0, 0, 128, 16, 8, 8, 0));
        EXPECT("VOP SWIGLU L 64", 224, 356, e_vop(TSR_VOP_FUNC_SWIGLU, 0, 0, 10, 100, 2000, 0, 1024, 1, 0, 0, 0));
        EXPECT("VOP MULC tile 128 x 8", 1083, 2111, e_vop(TSR_VOP_FUNC_MULC, 1, 1, 10, 10, 0, 0, 128, 128, 8, 8, 0));
        EXPECT("VOP MULCADD tile 128 x 8, c vector", 1092, 2120, e_vop(TSR_VOP_FUNC_MULCADD, 2, 0, 10, 10, 2000, 4000, 128, 128, 8, 8, 8));
        EXPECT("VOP m = 0", 1, 1, e_vop(TSR_VOP_FUNC_ADD, 0, 0, 10, 100, 200, 0, 0, 4, 3, 3, 3));
        EXPECT("RED SUM 1 row m 128", 149, 161, e_red(TSR_RED_KIND_SUM, 0, 0, 0, 0, 0, 7, 50, 128, 1, 8));
        EXPECT("RED SUM 1 row m 2048", 269, 401, e_red(TSR_RED_KIND_SUM, 0, 0, 0, 0, 0, 7, 50, 2048, 1, 128));
        EXPECT("RED SUMSQ RMS 1 row m 4096", 397, 657, e_red(TSR_RED_KIND_SUMSQ, 0, 0, TSR_RED_POST_RMS, 2, 3, 7, 50, 4096, 1, 256));
        EXPECT("RED SUM 16 rows m 128", 452, 520, e_red(TSR_RED_KIND_SUM, 0, 0, 0, 0, 0, 7, 50, 128, 16, 8));
        EXPECT("RED SUM 16 rows m 256", 516, 664, e_red(TSR_RED_KIND_SUM, 0, 0, 0, 0, 0, 7, 50, 256, 16, 16));
        EXPECT("RED SUM 128 rows m 128 (the Gated DeltaNet tile)", 2748, 3264, e_red(TSR_RED_KIND_SUM, 0, 0, 0, 0, 0, 7, 50, 128, 128, 8));
        EXPECT("RED SUM 128 rows m 64", 1153, 1192, e_red(TSR_RED_KIND_SUM, 0, 0, 0, 0, 0, 7, 50, 64, 128, 4));
        EXPECT("RED SUM to s m 256", 155, 175, e_red(TSR_RED_KIND_SUM, 1, 4, 0, 0, 0, 0, 50, 256, 1, 0));
        EXPECT("SOP MOVI", 53, 58, e_sop(TSR_SOP_SUB_MOVI, 1, 0, 0, F_ONE, 0));
        EXPECT("SOP ADD", 53, 58, e_sop(TSR_SOP_SUB_ADD, 1, 2, 3, 0, 0));
        EXPECT("SOP LDS", 58, 63, e_sop(TSR_SOP_SUB_LDS, 6, 0, 0, 0, 1000));
        EXPECT("SOP RSQRT", 68, 68, e_sop(TSR_SOP_SUB_RSQRT, 1, 2, 0, 0, 0));
        EXPECT("SOP DIV", 25, 25, e_sop(TSR_SOP_SUB_DIV, 1, 2, 3, 0, 0));
        rope_f rf = { TSR_ROPE_MODE_NEOX, 0, 0, 0, 1, 2, 0, 100, 3000, 5000, 0, 128, 64, 16, 2, 0, 0 };
        EXPECT("ROPE recurrence hd 128 nd 64 nh 16 T 2", 575, 803, e_rope(&rf));
        rf.hd = 256; rf.nd = 256; rf.nh = 4; rf.T = 1; rf.ff = 1; rf.table = 7000;
        EXPECT("ROPE freq factors hd 256 nd 256 nh 4", 1168, 1236, e_rope(&rf));
        rf.ff = 0; rf.atab = 1; rf.sts = 0;
        EXPECT("ROPE A table hd 256 nd 256 nh 4", 334, 402, e_rope(&rf));
        rf.atab = 0; rf.table = 0; rf.sts = 2; rf.T = 0;
        EXPECT("ROPE T = 0", 1, 1, e_rope(&rf));
#undef EXPECT
        tsr_sim_destroy(s);
    }
    // Not measured on the RTL of the cards (no TOPK, no DQROW in the K4 image): the estimates of VU §7.9 on the front lanes.
    tsr_sim * s = mk_sim(CAPS_ALL, 8);
    CHECK(cyc(s, e_red(TSR_RED_KIND_SUM, 0, 0, 0, 0, 0, 7, 50, 0, 3, 32)) == 1u, "RED vector m = 0");
    CHECK(cyc(s, e_topk(5, 10, 20, 100, 2, 0, 7)) == 2u * (7u + 5u) + 20u, "TOPK");
    CHECK(cyc(s, e_dqrow(TSR_FMT_Q8_0, 10, 100, 512, 3, 0, 0)) == 3u * 32u + 30u, "DQROW");
    tsr_sim_destroy(s);
    // The configuration is validated.
    tsr_config c = cfg_u200(8);
    c.spm_bytes = SPM_BYTES;
    c.ch_bytes = 64ull << 20;
    c.caps = CAPS_ALL;
    tsr_sim_opts o;
    memset(&o, 0, sizeof o);
    const char * good[] = { "u200", "k4", "16,16,8,0", "8,8,4,3" }, * bad[] = { "k5", "12,16,8,0", "8,8,4", "8,8,4,4", "8,8,3,0", "" };
    for (size_t k = 0; k < sizeof good / sizeof good[0]; ++k) {
        o.vu_timing = good[k];
        tsr_sim * g = tsr_sim_create_opts(&c, &o);
        CHECK(g != NULL, "vu_timing %s is accepted", good[k]);
        if (g) tsr_sim_destroy(g);
    }
    for (size_t k = 0; k < sizeof bad / sizeof bad[0]; ++k) {
        o.vu_timing = bad[k];
        tsr_sim * g = tsr_sim_create_opts(&c, &o);
        CHECK(g == NULL || bad[k][0] == '\0', "vu_timing %s is refused", bad[k]);
        if (g) tsr_sim_destroy(g);
    }
    o.vu_timing = "k4";
    s = tsr_sim_create_opts(&c, &o);
    CHECK(TSR_GET(tsr_sim_csr_read(s, TSR_CSR_CFG3), TSR_CFG3_VU_FRONT) == 8 && TSR_GET(tsr_sim_csr_read(s, TSR_CSR_CFG3), TSR_CFG3_VU_BACK) == 4,
          "CFG3 reports the EXP and back lanes of the card image (0x02480088 with the attention field)");
    tsr_sim_destroy(s);
}

static void test_gdn_misc(void) {
    tsr_sim * s = mk_sim(CAPS_ALL, 8);
    tsr_extents x;
    uint32_t info;
    const uint8_t SP = TSR_EXT_SPM, SG = TSR_EXT_SREG, R = TSR_EXT_READ, Wr = TSR_EXT_WRITE;
    // extents: d W, a R, k R, o W, then the s registers; q and v are not listed (four SPM slots)
    check_only(s, e_gdn(1, 2, 3000, 1000, 200, 300, 400, 500, 128, 20), &info, &x);
    CHECK(x.n == 6 && ext_is(&x, 0, SP, Wr, 3000, 3160) && ext_is(&x, 1, SP, R, 1000, 1160) && ext_is(&x, 2, SP, R, 200, 208) &&
          ext_is(&x, 3, SP, Wr, 500, 502) && ext_is(&x, 4, SG, R, 1, 2) && ext_is(&x, 5, SG, R, 2, 3), "GDN extents");
    check_only(s, e_gdn(4, 4, 1000, 1000, 200, 300, 400, 500, 64, 1), &info, &x);
    CHECK(x.n == 5 && ext_is(&x, 0, SP, Wr, 1000, 1004) && ext_is(&x, 1, SP, R, 1000, 1004) && ext_is(&x, 4, SG, R, 4, 5) &&
          x.overflow == 0, "GDN in place, one s register");
    // The RTL unit's cycles of a dependent op (tb/vu test_vu_timing.py vu_gdn_op, SW-78): 2,913.7 for 128 x 128 at the K4 lanes,
    // 1,917 at the U200 lanes (the model: 1,898, within 1.5 %), 5,554 for m 256 rows 128 and 989 for m 128 rows 16 at the K4 lanes.
    CHECK(cyc(s, e_gdn(1, 2, 3000, 1000, 200, 300, 400, 500, 128, 128)) == 1032u + 168u + 576u + 122u, "GDN cycles, U200 lanes");
    tsr_sim_destroy(s);
    s = mk_sim_vu("k4");
    CHECK(cyc(s, e_gdn(1, 2, 3000, 1000, 200, 300, 400, 500, 128, 128)) == 2914u, "GDN cycles 128 x 128, K4 lanes");
    CHECK(cyc(s, e_gdn(1, 2, 8000, 1000, 200, 300, 400, 500, 256, 128)) == 5554u, "GDN cycles m 256 rows 128, K4 lanes");
    CHECK(cyc(s, e_gdn(1, 2, 8000, 1000, 200, 300, 400, 500, 128, 16)) == 989u, "GDN cycles m 128 rows 16, K4 lanes");
    CHECK(cyc(s, e_gdn(1, 2, 8000, 1000, 200, 300, 400, 500, 128, 256)) == 5114u, "GDN cycles m 128 rows 256, K4 lanes");
    tsr_sim_destroy(s);
    s = mk_sim(TSR_CAPS_VU | TSR_CAPS_TOPK | TSR_CAPS_DQROW, 8);
    CHECK(tsr_sim_op(s, e_gdn(1, 2, 3000, 1000, 200, 300, 400, 500, 128, 20).w, &info) == TSR_ERR_ILLEGAL_OP, "GDN without CAPS.GDN");
    tsr_sim_destroy(s);
}

static void make_active(tsr_sim * s) {
    tsr_sim_csr_write(s, TSR_CSR_LQ_PUSH, 0);
    uint32_t v = 0;
    CHECK(tsr_sim_cw_access(s, 1, TSR_CW_LQ_POP, &v) == TSR_CW_OK, "LQ_POP");
    CHECK(tsr_sim_csr_read(s, TSR_CSR_STATUS) & TSR_STATUS_ACTIVE, "ACTIVE");
}

static void test_perf(void) {
    const uint32_t C = 4, LINES = TSR_PERF_VU_LINES(C), VCYC = TSR_PERF_VU_CYCLES;
    const tsr_insn ops[] = {
        e_vop(TSR_VOP_FUNC_ADD, 0, 0, 10, 100, 200, 0, 40, 3, 3, 3, 3),          // 9 lines
        e_red(TSR_RED_KIND_SUM, 0, 0, 0, 0, 0, 500, 100, 40, 20, 3),               // 2 lines
        e_red(TSR_RED_KIND_SUM, 1, 4, 0, 0, 0, 0, 100, 40, 1, 0),                  // 0 lines (s4)
        e_sop(TSR_SOP_SUB_STS, 0, 4, 0, 0, 600 * 16),                              // 1 line
        e_sop(TSR_SOP_SUB_MOVI, 5, 0, 0, F_ONE, 0),                                // 0 lines
        e_topk(9, 700, 100, 40, 3, 0, 3),                                          // 6 lines
        e_dqrow(TSR_FMT_Q8_0, 800, 100, 64, 2, 0, 0),                              // 8 lines
        e_vop(TSR_VOP_FUNC_COPY, 0, 0, 10, 100, 0, 0, 0, 3, 3, 3, 0),              // m = 0: 0 lines
    };
    const uint64_t lines_exp = 9 + 2 + 0 + 1 + 0 + 6 + 8 + 0;
    const size_t n = sizeof(ops) / sizeof(ops[0]);
    // The op-level hook: ungated, CYCLES += 1 + cycles, VU_CYCLES += cycles.
    {
        tsr_sim * s = mk_sim(CAPS_ALL, 8);
        uint64_t sum = 0;
        for (size_t k = 0; k < n; ++k) {
            sum += cyc(s, ops[k]);
            CHECK(tsr_sim_op(s, ops[k].w, NULL) == 0, "perf op %zu", k);
        }
        uint32_t info;
        tsr_sim_op(s, e_topk(0, 700, 100, 40, 3, 0, 3).w, &info);     // fails: RETIRED only
        CHECK(tsr_sim_perf(s, LINES) == lines_exp, "tsr_sim_op: VU_LINES %llu", (unsigned long long)tsr_sim_perf(s, LINES));
        CHECK(tsr_sim_perf(s, VCYC) == sum, "tsr_sim_op: VU_CYCLES = sum of cycles()");
        CHECK(tsr_sim_perf(s, TSR_PERF_CYCLES) == sum + n, "tsr_sim_op: CYCLES = sum of (1 + cycles())");
        CHECK(tsr_sim_perf(s, TSR_PERF_RETIRED) == n + 1u, "tsr_sim_op: RETIRED");
        CHECK(tsr_sim_perf(s, TSR_PERF_VU_SPM_WAIT(C)) == 0u && tsr_sim_perf(s, TSR_PERF_VU_HAZARD(C)) == 0u, "no SPM wait / hazard model");
        tsr_sim_destroy(s);
    }
    // The queue path: counted only while ACTIVE (EX §6); VU_CYCLES is the unit's busy time.
    for (int active = 0; active < 2; ++active) {
        tsr_sim * s = mk_sim(CAPS_ALL, 8);
        if (active) make_active(s);
        uint64_t sum = 0;
        for (size_t k = 0; k < n; ++k) {
            sum += cyc(s, ops[k]);
            q_push(s, ops[k], TSR_TICKET_NONE, TSR_TICKET_NONE);
        }
        CHECK(tsr_sim_run_until_idle(s, 10000000), "idle");
        CHECK(tsr_sim_csr_read(s, TSR_CSR_Q_DONE(TSR_UNIT_VU)) == n, "all VU ops done");
        CHECK(tsr_sim_csr_read(s, TSR_CSR_Q_ERR_CODE(TSR_UNIT_VU)) == 0u, "no VU error");
        CHECK(tsr_sim_perf(s, LINES) == (active ? lines_exp : 0u), "queue: VU_LINES (active %d)", active);
        CHECK(tsr_sim_perf(s, VCYC) == (active ? sum : 0u), "queue: VU_CYCLES (active %d) %llu vs %llu", active,
              (unsigned long long)tsr_sim_perf(s, VCYC), (unsigned long long)sum);
        CHECK(tsr_sim_perf(s, TSR_PERF_RETIRED) == (active ? n : 0u), "queue: RETIRED");
        CHECK(tsr_sim_now(s) >= sum, "queue: the VU ops run one after another (%llu >= %llu)",
              (unsigned long long)tsr_sim_now(s), (unsigned long long)sum);
        tsr_sim_destroy(s);
    }
}

// ---------------------------------------------------------------------------
// Kernels as op sequences: RMSNorm, per-head norms, softmax
// ---------------------------------------------------------------------------
static void test_rmsnorm(void) {
    twin t;
    tw_init(&t, CAPS_ALL, 0x5EED0008u);
    const uint32_t n = 896, X = 100, WT = 200, Y = 300;              // Qwen2.5-0.5B n_embd
    for (uint32_t j = 0; j < n; ++j) {
        tw_put(&t, X + j / 16u, j % 16u, rand_norm(&t.r, 3.0));
        tw_put(&t, WT + j / 16u, j % 16u, f2u(1.0f + (float)(rng_normal(&t.r) * 0.1)));
    }
    const uint32_t inv_n = f2u(1.0f / (float)n), eps = f2u(1e-6f);
    tw_movi(&t, 2, inv_n);
    tw_movi(&t, 3, eps);
    // VU §8.1: RED.SUMSQ TO_S sd=s1 post=RMS sk=s2 se=s3, then VOP.MULCB d = (x * s1) * w.
    tw_op(&t, e_red(TSR_RED_KIND_SUMSQ, 1, 1, TSR_RED_POST_RMS, 2, 3, 0, X, n, 1, 0), "RMSNorm RED");
    t.s[1] = tsr_ref_red_row(TSR_RED_KIND_SUMSQ, TSR_RED_POST_RMS, W(t.exp, X), n, inv_n, eps);
    tw_op(&t, e_vop(TSR_VOP_FUNC_MULCB, 1, 1, Y, X, WT, 0, n, 1, 0, 0, 0), "RMSNorm MULCB");
    tsr_ref_vop(TSR_VOP_FUNC_MULCB, TSR_VOP_CMODE_SREG, 1, n, W(t.exp, Y), 0, W(t.exp, X), 0, W(t.exp, WT), 0, &t.s[1]);
    tw_same(&t, "RMSNorm");
    // The spec recipe (RED SUMSQ, SOP MULI, ADDI, SQRT, RECIP) gives the same bits (VU §6.2).
    tw_op(&t, e_red(TSR_RED_KIND_SUMSQ, 1, 4, 0, 0, 0, 0, X, n, 1, 0), "recipe RED");
    tw_op(&t, e_sop(TSR_SOP_SUB_MULI, 4, 4, 0, inv_n, 0), "recipe MULI");
    tw_op(&t, e_sop(TSR_SOP_SUB_ADDI, 4, 4, 0, eps, 0), "recipe ADDI");
    tw_op(&t, e_sop(TSR_SOP_SUB_SQRT, 4, 4, 0, 0, 0), "recipe SQRT");
    tw_op(&t, e_sop(TSR_SOP_SUB_RECIP, 4, 4, 0, 0, 0), "recipe RECIP");
    const uint32_t * sb = (const uint32_t *)(const void *)tsr_sim_sreg(t.sim);
    CHECK(sb[4] == sb[1], "RED post RMS = the SOP recipe (%08x vs %08x)", sb[1], sb[4]);
    t.s[4] = t.s[1];
    // Sanity against double: y = x·w / sqrt(mean(x²) + eps).
    double ss = 0;
    for (uint32_t j = 0; j < n; ++j) ss += (double)u2f(W(t.exp, X)[j]) * (double)u2f(W(t.exp, X)[j]);
    const double sc = 1.0 / sqrt(ss / n + 1e-6);
    double worst = 0;
    for (uint32_t j = 0; j < n; ++j) {
        const double ref = (double)u2f(W(t.exp, X)[j]) * sc * (double)u2f(W(t.exp, WT)[j]);
        const double got = (double)u2f(SW(t.sim, Y)[j]);
        const double e = fabs(got - ref) / (fabs(ref) + 1e-30);
        if (e > worst) worst = e;
    }
    CHECK(worst < 4e-6, "RMSNorm within 4e-6 relative of double (%.3g)", worst);
    // Gemma-style per-head norms: one RED over the head rows (post RMS) -> c vector, MULCB with the
    // weight broadcast (b stride 0) on the q rows, MULC on the v rows.
    const uint32_t heads = 20, hd = 256, L = hd / 16u, Q = 2000, CV = 1900, QW = 1800, QO = 4000;
    tw_rand(&t, Q, heads * L);
    for (uint32_t j = 0; j < hd; ++j) tw_put(&t, QW + j / 16u, j % 16u, f2u(1.0f + (float)(rng_normal(&t.r) * 0.1)));
    tw_movi(&t, 2, f2u(1.0f / (float)hd));
    tw_op(&t, e_red(TSR_RED_KIND_SUMSQ, 0, 0, TSR_RED_POST_RMS, 2, 3, CV, Q, hd, heads, L), "per-head RED");
    tsr_ref_red(TSR_RED_KIND_SUMSQ, TSR_RED_POST_RMS, heads, hd, W(t.exp, CV), W(t.exp, Q), L, t.s[2], t.s[3]);
    tw_op(&t, e_vop(TSR_VOP_FUNC_MULCB, 2, 0, QO, Q, QW, CV, hd, 16, L, L, 0), "per-head MULCB");
    tsr_ref_vop(TSR_VOP_FUNC_MULCB, TSR_VOP_CMODE_VEC, 16, hd, W(t.exp, QO), L, W(t.exp, Q), L, W(t.exp, QW), 0, W(t.exp, CV));
    // rows 16..19 (no weight): their scales are words 16..19 of the vector, i.e. words 0..3 at CV + 1
    tw_op(&t, e_vop(TSR_VOP_FUNC_MULC, 2, 0, QO + 16u * L, Q + 16u * L, 0, CV + 1u, hd, 4, L, L, 0), "per-head MULC (v rows)");
    tsr_ref_vop(TSR_VOP_FUNC_MULC, TSR_VOP_CMODE_VEC, 4, hd, W(t.exp, QO + 16u * L), L, W(t.exp, Q + 16u * L), L, NULL, 0,
                W(t.exp, CV + 1u));
    tw_same(&t, "per-head norms");
    tw_free(&t);
}

// Softmax over rows (VU §8.4's generic composition): m = RED MAX, e = EXPSUBC(x, m), r = RED SUM
// post RCP over e, y = MULC(e, r). Fully masked rows give exp -> +0 everywhere (the -inf rule).
static void softmax_ops(tsr_insn * ops, uint32_t rows, uint32_t m, uint32_t X, uint32_t MV, uint32_t E, uint32_t RV, uint32_t Y) {
    const uint32_t L = tsr_ref_vu_lines(m);
    ops[0] = e_red(TSR_RED_KIND_MAX, 0, 0, 0, 0, 0, MV, X, m, rows, L);
    ops[1] = e_vop(TSR_VOP_FUNC_EXPSUBC, 2, 0, E, X, 0, MV, m, rows, L, L, 0);
    ops[2] = e_red(TSR_RED_KIND_SUM, 0, 0, TSR_RED_POST_RCP, 0, 0, RV, E, m, rows, L);
    ops[3] = e_vop(TSR_VOP_FUNC_MULC, 2, 0, Y, E, 0, RV, m, rows, L, L, 0);
}

static void test_softmax(void) {
    twin t;
    tw_init(&t, CAPS_ALL, 0x5EED0009u);
    const uint32_t rows = 6, m = 100, L = tsr_ref_vu_lines(m), X = 100, MV = 300, E = 400, RV = 500, Y = 600;
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t j = 0; j < 16u * L; ++j) {
            uint32_t v = rand_norm(&t.r, 6.0);
            if (r == 2 && j % 3u == 0u) v = 0xFF800000u;            // masked cells
            if (r == 4) v = 0xFF800000u;                            // fully masked row
            tw_put(&t, X + r * L + j / 16u, j % 16u, v);
        }
    }
    tsr_insn ops[4];
    softmax_ops(ops, rows, m, X, MV, E, RV, Y);
    uint8_t * init = (uint8_t *)malloc(SPM_BYTES);
    memcpy(init, tsr_sim_spm(t.sim), SPM_BYTES);
    for (int k = 0; k < 4; ++k) tw_op(&t, ops[k], "softmax op");
    tsr_ref_red(TSR_RED_KIND_MAX, 0, rows, m, W(t.exp, MV), W(t.exp, X), L, 0, 0);
    tsr_ref_vop(TSR_VOP_FUNC_EXPSUBC, TSR_VOP_CMODE_VEC, rows, m, W(t.exp, E), L, W(t.exp, X), L, NULL, 0, W(t.exp, MV));
    tsr_ref_red(TSR_RED_KIND_SUM, TSR_RED_POST_RCP, rows, m, W(t.exp, RV), W(t.exp, E), L, 0, 0);
    tsr_ref_vop(TSR_VOP_FUNC_MULC, TSR_VOP_CMODE_VEC, rows, m, W(t.exp, Y), L, W(t.exp, E), L, NULL, 0, W(t.exp, RV));
    tw_same(&t, "softmax");
    for (uint32_t r = 0; r < rows; ++r) {
        double sum = 0;
        for (uint32_t j = 0; j < m; ++j) sum += (double)u2f(SW(t.sim, Y + r * L)[j]);
        if (r == 4) {
            for (uint32_t j = 0; j < m; ++j) CHECK(SW(t.sim, E + r * L)[j] == 0u, "fully masked row: EXPSUB gives +0");
        } else {
            CHECK(fabs(sum - 1.0) < 1e-5, "softmax row %u sums to 1 (%.9f)", r, sum);
        }
    }
    // The same sequence through the queue model (VU queue order, no deps needed) gives the same SPM.
    tsr_sim * q = mk_sim(CAPS_ALL, 8);
    memcpy(tsr_sim_spm(q), init, SPM_BYTES);
    free(init);
    for (int k = 0; k < 4; ++k) q_push(q, ops[k], TSR_TICKET_NONE, TSR_TICKET_NONE);
    CHECK(tsr_sim_run_until_idle(q, 10000000), "queue softmax idle");
    CHECK(memcmp(tsr_sim_spm(q), tsr_sim_spm(t.sim), SPM_BYTES) == 0, "queue path = op hook, bit for bit");
    tsr_sim_destroy(q);
    tw_free(&t);
}

int main(void) {
    test_gdn();
    test_gdn_checks();
    test_vop();
    test_red();
    test_rope();
    test_topk();
    test_sop();
    test_dqrow();
    test_checks();
    test_queue_records();
    test_extents();
    test_cycles_rtl();
    test_gdn_misc();
    test_perf();
    test_rmsnorm();
    test_softmax();
    return test_finish("sim_vu");
}
