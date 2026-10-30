// tsr_ref_vu.c — reference semantics of the VU ops VOP, RED, ROPE, TOPK, SOP and DQROW, bit for bit.
//
// Normative source: docs/v2/02-vector-unit.md (VU) §4 (fp rules, operand model), §5.2–§5.8 (field
// meaning), §6.2 (RED), §6.7 (ROPE), §6.8 (total order, TOPK, DQROW). The API and its conventions
// (word arrays, line strides, the tail and n = 0 rules, aliasing) are in tsr_ref_vu.h.
//
// Implementation rules (as tsr_ref_math.c, docs/v2/06-software-verification.md §4.5)
//   - Every f32 operation is a call to the tsr_ref.c primitives (FA tsr_fadd, FS tsr_fsub, FM
//     tsr_fmul, FD tsr_fdiv, SQ tsr_fsqrt, RC tsr_frecip: DAZ, RNE, FTZ, canonical NaN) or to the
//     tsr_ref_math.c functions; everything else is integer bit manipulation. No float operators.
//   - Element by element: every function reads the sources of an element (of a ROPE pair) before it
//     writes the element, so a destination that equals a source exactly works in place (VU §4.5).

#include "tsr_ref_vu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tsr_ref.h"
#include "tsr_ref_math.h"

#if defined(__FAST_MATH__)
#error "tsr_ref_vu.c must not be compiled with -ffast-math"
#endif

#define F32_SIGN 0x80000000u
#define F32_EXP  0x7F800000u
#define F32_ABS  0x7FFFFFFFu
#define B_NAN    TSR_CANON_NAN_BITS
#define B_NINF   0xFF800000u

static void vu_fatal(const char * fn, const char * msg, uint32_t a, uint32_t b) {
    fprintf(stderr, "tsr_ref_vu: %s: %s (%u, %u)\n", fn, msg, a, b);
    abort();
}

static inline uint32_t f2u(float f) { uint32_t u; memcpy(&u, &f, sizeof u); return u; }
static inline float u2f(uint32_t u) { float f; memcpy(&f, &u, sizeof f); return f; }

static inline uint32_t daz_bits(uint32_t u) { return ((u & F32_EXP) == 0) ? (u & F32_SIGN) : u; }
static inline int is_nan_bits(uint32_t u) { return (u & F32_ABS) > F32_EXP; }

// The f32 primitives on bit patterns.
static inline uint32_t FA(uint32_t a, uint32_t b) { return f2u(tsr_fadd(u2f(a), u2f(b))); }
static inline uint32_t FS(uint32_t a, uint32_t b) { return f2u(tsr_fsub(u2f(a), u2f(b))); }
static inline uint32_t FM(uint32_t a, uint32_t b) { return f2u(tsr_fmul(u2f(a), u2f(b))); }
static inline uint32_t FD(uint32_t a, uint32_t b) { return f2u(tsr_fdiv(u2f(a), u2f(b))); }
static inline uint32_t SQ(uint32_t a) { return f2u(tsr_fsqrt(u2f(a))); }
static inline uint32_t RC(uint32_t a) { return f2u(tsr_frecip(u2f(a))); }

// Word offset of row r of an operand with a line stride s (VU §4.2).
static inline size_t row_off(uint32_t r, uint32_t s) { return (size_t)r * s * 16u; }

// Tail rule: words [n, 16·L(n)) of a destination row are written 0 (VU §4.2).
static void zero_tail(uint32_t * row, uint32_t n) {
    const size_t end = (size_t)16u * tsr_ref_vu_lines(n);
    for (size_t j = n; j < end; ++j) {
        row[j] = 0u;
    }
}

// ---------------------------------------------------------------------------
// Total order, MAXT, MINT (VU §6.8)
// ---------------------------------------------------------------------------
uint32_t tsr_ref_vu_key(uint32_t x) {
    const uint32_t d = daz_bits(x);
    return (d & F32_SIGN) ? ~d : (d | F32_SIGN);
}

uint32_t tsr_ref_vu_maxt(uint32_t a, uint32_t b) {
    if (is_nan_bits(a) || is_nan_bits(b)) {
        return B_NAN;
    }
    const uint32_t ad = daz_bits(a), bd = daz_bits(b);
    return (tsr_ref_vu_key(ad) >= tsr_ref_vu_key(bd)) ? ad : bd;
}

uint32_t tsr_ref_vu_mint(uint32_t a, uint32_t b) {
    const int na = is_nan_bits(a), nb = is_nan_bits(b);
    if (na && nb) {
        return B_NAN;
    }
    const uint32_t ad = daz_bits(a), bd = daz_bits(b);
    if (na) {
        return bd;
    }
    if (nb) {
        return ad;
    }
    return (tsr_ref_vu_key(ad) <= tsr_ref_vu_key(bd)) ? ad : bd;
}

// ---------------------------------------------------------------------------
// VOP (VU §5.2)
// ---------------------------------------------------------------------------
static inline uint32_t expk_p1(uint32_t v) { return f2u(tsr_m_expk(u2f(v), TSR_M_P1)); }
static inline uint32_t silu(uint32_t x) { return f2u(tsr_m_silu(u2f(x))); }
static inline uint32_t gelu(uint32_t x) { return f2u(tsr_m_gelu_tanh(u2f(x))); }
static inline uint32_t expsub(uint32_t a, uint32_t x) { return f2u(tsr_m_expsub(u2f(a), u2f(x))); }

uint32_t tsr_ref_vop_elem(uint32_t func, uint32_t a, uint32_t b, uint32_t c) {
    switch (func) {
    case TSR_VOP_FUNC_ADD:     return FA(a, b);
    case TSR_VOP_FUNC_SUB:     return FS(a, b);
    case TSR_VOP_FUNC_MUL:     return FM(a, b);
    case TSR_VOP_FUNC_MAX:     return tsr_ref_vu_maxt(a, b);
    case TSR_VOP_FUNC_MULC:    return FM(a, c);
    case TSR_VOP_FUNC_ADDC:    return FA(a, c);
    case TSR_VOP_FUNC_SILU:    return silu(a);
    case TSR_VOP_FUNC_SWIGLU:  return FM(silu(a), b);
    case TSR_VOP_FUNC_EXP:     return expk_p1(a);
    case TSR_VOP_FUNC_COPY:    return a;                                  // raw
    case TSR_VOP_FUNC_MIN:     return tsr_ref_vu_mint(a, b);
    case TSR_VOP_FUNC_SUBC:    return FS(a, c);
    case TSR_VOP_FUNC_MULCB:   return FM(FM(a, c), b);
    case TSR_VOP_FUNC_MULCADD: return FA(FM(a, c), b);
    case TSR_VOP_FUNC_ADDMULC: return FM(FA(a, b), c);
    case TSR_VOP_FUNC_EXPSUB:  return expsub(a, b);
    case TSR_VOP_FUNC_EXPSUBC: return expsub(a, c);
    case TSR_VOP_FUNC_SIGMOID: return f2u(tsr_m_sigmoid(u2f(a)));
    case TSR_VOP_FUNC_GELU:    return gelu(a);
    case TSR_VOP_FUNC_GEGLU:   return FM(gelu(a), b);
    case TSR_VOP_FUNC_TANH:    return f2u(tsr_m_tanh(u2f(a)));
    case TSR_VOP_FUNC_RECIP:   return RC(a);
    case TSR_VOP_FUNC_SETC:    return c;                                  // raw
    case TSR_VOP_FUNC_GELU16:  return f2u(tsr_m_gelu16(u2f(a)));
    case TSR_VOP_FUNC_GEGLU16: return f2u(tsr_m_geglu16(u2f(a), u2f(b)));
    default:
        vu_fatal("tsr_ref_vop_elem", "undefined VOP function", func, 0);
    }
    return 0;
}

void tsr_ref_vop(uint32_t func, uint32_t cmode, uint32_t rows, uint32_t m,
                 uint32_t * d, uint32_t d_stride,
                 const uint32_t * a, uint32_t a_stride,
                 const uint32_t * b, uint32_t b_stride,
                 const uint32_t * c) {
    const uint32_t reads = tsr_vop_func_reads(func);
    if (reads == 0u) {
        vu_fatal("tsr_ref_vop", "undefined VOP function", func, 0);
    }
    const int uses_c = (reads & TSR_VOP_READS_C) != 0;
    if (cmode > TSR_VOP_CMODE_VEC || uses_c != (cmode != TSR_VOP_CMODE_NONE)) {
        vu_fatal("tsr_ref_vop", "cmode does not match the function", cmode, func);
    }
    if (rows == 0u || m == 0u) {                       // n = 0 rule (VU §4.2)
        return;
    }
    if (!d || ((reads & TSR_VOP_READS_A) && !a) || ((reads & TSR_VOP_READS_B) && !b) || (uses_c && !c)) {
        vu_fatal("tsr_ref_vop", "missing operand", func, reads);
    }
    for (uint32_t r = 0; r < rows; ++r) {
        uint32_t * dr = d + row_off(r, d_stride);
        const uint32_t * ar = (reads & TSR_VOP_READS_A) ? a + row_off(r, a_stride) : NULL;
        const uint32_t * br = (reads & TSR_VOP_READS_B) ? b + row_off(r, b_stride) : NULL;
        const uint32_t cr = !uses_c ? 0u : (cmode == TSR_VOP_CMODE_SREG) ? c[0] : c[r];
        for (uint32_t j = 0; j < m; ++j) {
            const uint32_t av = ar ? ar[j] : 0u;       // read before the write (in place)
            const uint32_t bv = br ? br[j] : 0u;
            dr[j] = tsr_ref_vop_elem(func, av, bv, cr);
        }
        zero_tail(dr, m);
    }
}

// ---------------------------------------------------------------------------
// RED (VU §6.2)
// ---------------------------------------------------------------------------
void tsr_ref_red_partials(uint32_t kind, const uint32_t * x, uint32_t m, uint32_t p[TSR_RED_PARTIALS]) {
    if (kind != TSR_RED_KIND_SUM && kind != TSR_RED_KIND_SUMSQ) {
        vu_fatal("tsr_ref_red_partials", "kind is not SUM or SUMSQ", kind, 0);
    }
    for (uint32_t i = 0; i < TSR_RED_PARTIALS; ++i) {
        p[i] = 0u;                                     // +0.0
    }
    for (uint32_t i = 0; i < m; ++i) {
        const uint32_t v = (kind == TSR_RED_KIND_SUMSQ) ? FM(x[i], x[i]) : x[i];
        p[i % TSR_RED_PARTIALS] = FA(p[i % TSR_RED_PARTIALS], v);   // the first update is FA(+0, v)
    }
}

uint32_t tsr_ref_red_tree128(const uint32_t p[TSR_RED_PARTIALS]) {
    uint32_t v[TSR_RED_PARTIALS];
    memcpy(v, p, sizeof v);
    for (uint32_t n = TSR_RED_PARTIALS; n > 1u; n >>= 1) {         // 7 levels
        for (uint32_t k = 0; k < n / 2u; ++k) {
            v[k] = FA(v[2u * k], v[2u * k + 1u]);
        }
    }
    return v[0];
}

static inline uint32_t zfn(uint32_t q) { return (q == F32_SIGN) ? 0u : q; }   // z(): -0 -> +0

uint32_t tsr_ref_red_tree_skip(const uint32_t p[TSR_RED_PARTIALS], uint32_t n_written, uint32_t group) {
    if (group == 0u || group > TSR_RED_PARTIALS) {
        vu_fatal("tsr_ref_red_tree_skip", "group must be in 1..128", group, 0);
    }
    if (n_written > TSR_RED_PARTIALS) {
        n_written = TSR_RED_PARTIALS;
    }
    // Present leaves: the groups that hold a written partial (partials of a partly written group
    // enter with their value, +0 when unwritten).
    uint32_t n_present = ((n_written + group - 1u) / group) * group;
    if (n_present > TSR_RED_PARTIALS) {
        n_present = TSR_RED_PARTIALS;
    }
    uint32_t v[TSR_RED_PARTIALS];
    uint8_t pr[TSR_RED_PARTIALS];
    for (uint32_t i = 0; i < TSR_RED_PARTIALS; ++i) {
        v[i] = p[i];
        pr[i] = (uint8_t)(i < n_present);
    }
    for (uint32_t n = TSR_RED_PARTIALS; n > 1u; n >>= 1) {
        for (uint32_t k = 0; k < n / 2u; ++k) {
            const uint32_t l = v[2u * k], r = v[2u * k + 1u];
            const uint8_t pl = pr[2u * k], prr = pr[2u * k + 1u];
            if (pl && prr) {
                v[k] = FA(l, r);
            } else if (pl) {
                v[k] = zfn(l);                         // FA(l, +0) = z(l): tree values are never subnormal
            } else if (prr) {
                v[k] = zfn(r);
            } else {
                v[k] = 0u;                             // skipped
            }
            pr[k] = (uint8_t)(pl | prr);
        }
    }
    return pr[0] ? v[0] : 0u;
}

uint32_t tsr_ref_red_post(uint32_t post, uint32_t y, uint32_t sk, uint32_t se) {
    switch (post) {
    case TSR_RED_POST_NONE: return y;
    case TSR_RED_POST_RCP:  return RC(y);
    case TSR_RED_POST_RMS:  return RC(SQ(FA(FM(y, sk), se)));
    default:
        vu_fatal("tsr_ref_red_post", "undefined post-op", post, 0);
    }
    return 0;
}

uint32_t tsr_ref_red_row(uint32_t kind, uint32_t post, const uint32_t * x, uint32_t m,
                         uint32_t sk, uint32_t se) {
    uint32_t y;
    if (kind == TSR_RED_KIND_MAX) {
        y = B_NINF;                                    // empty row: -inf; MAXT(-inf, v) = DAZ(v)
        for (uint32_t i = 0; i < m; ++i) {
            y = tsr_ref_vu_maxt(y, x[i]);              // a NaN makes y canonical NaN, which stays
        }
    } else if (kind == TSR_RED_KIND_SUM || kind == TSR_RED_KIND_SUMSQ) {
        uint32_t p[TSR_RED_PARTIALS];
        tsr_ref_red_partials(kind, x, m, p);           // m = 0: all +0, the tree gives +0
        y = tsr_ref_red_tree128(p);
    } else {
        vu_fatal("tsr_ref_red_row", "undefined RED kind", kind, 0);
        return 0;
    }
    return tsr_ref_red_post(post, y, sk, se);
}

void tsr_ref_red(uint32_t kind, uint32_t post, uint32_t rows, uint32_t m, uint32_t * d,
                 const uint32_t * a, uint32_t a_stride, uint32_t sk, uint32_t se) {
    if (kind > TSR_RED_KIND_MAX || post > TSR_RED_POST_RMS) {
        vu_fatal("tsr_ref_red", "undefined RED kind or post-op", kind, post);
    }
    if (rows == 0u || m == 0u) {                       // n = 0 rule (VU §4.2)
        return;
    }
    if (!d || !a) {
        vu_fatal("tsr_ref_red", "missing operand", rows, m);
    }
    // All row results first: the vector d may not overlap a (VU §4.5), but keep reads before writes.
    uint32_t * y = malloc((size_t)rows * sizeof *y);
    if (!y) {
        vu_fatal("tsr_ref_red", "out of memory", rows, 0);
    }
    for (uint32_t r = 0; r < rows; ++r) {
        y[r] = tsr_ref_red_row(kind, post, a + row_off(r, a_stride), m, sk, se);
    }
    memcpy(d, y, (size_t)rows * sizeof *y);
    zero_tail(d, rows);
    free(y);
}

// ---------------------------------------------------------------------------
// ROPE (VU §6.7)
// ---------------------------------------------------------------------------
static void rope_check(const char * fn, const tsr_ref_rope_cfg * cfg) {
    if (cfg->mode > TSR_ROPE_MODE_NEOX) {
        vu_fatal(fn, "undefined ROPE mode", cfg->mode, 0);
    }
    if (cfg->atab && (cfg->ff || cfg->fs)) {
        vu_fatal(fn, "ATAB with FF or FS", cfg->ff, cfg->fs);
    }
    if (cfg->head_dim == 0u || (cfg->head_dim & 15u) || cfg->head_dim > TSR_ROPE_MAX_HEAD_DIM) {
        vu_fatal(fn, "illegal head_dim", cfg->head_dim, 0);
    }
    const uint32_t q = (cfg->mode == TSR_ROPE_MODE_NEOX) ? 32u : 16u;
    if (cfg->n_dims == 0u || cfg->n_dims > cfg->head_dim || cfg->n_dims % q) {
        vu_fatal(fn, "illegal n_dims", cfg->n_dims, cfg->head_dim);
    }
}

void tsr_ref_rope_angles(const tsr_ref_rope_cfg * cfg, int32_t pos, const uint32_t * table, uint32_t * phi) {
    rope_check("tsr_ref_rope_angles", cfg);
    const uint32_t half = cfg->n_dims / 2u;
    if ((cfg->ff || cfg->atab) && !table) {
        vu_fatal("tsr_ref_rope_angles", "missing table", cfg->ff, cfg->atab);
    }
    const uint32_t pf = f2u(tsr_i32_to_f32(pos));
    if (cfg->atab) {                                   // A-table form: phi_i = FM(Pf, A_i)
        for (uint32_t i = 0; i < half; ++i) {
            phi[i] = FM(pf, table[i]);
        }
        return;
    }
    uint32_t theta = pf;                               // recurrence form (ggml_rope_cache_init)
    for (uint32_t i = 0; i < half; ++i) {
        uint32_t f = cfg->ff ? FD(theta, table[i]) : theta;
        if (cfg->fs) {
            f = FM(cfg->freq_scale, f);
        }
        phi[i] = f;
        theta = FM(theta, cfg->ts);                    // theta_(i+1) = FM(theta_i, TS)
    }
}

void tsr_ref_rope_cs(const tsr_ref_rope_cfg * cfg, int32_t pos, const uint32_t * table,
                     uint32_t * c, uint32_t * s) {
    uint32_t phi[TSR_ROPE_MAX_HEAD_DIM / 2u];
    tsr_ref_rope_angles(cfg, pos, table, phi);
    for (uint32_t i = 0; i < cfg->n_dims / 2u; ++i) {
        float sn, cs;
        tsr_m_sincos(u2f(phi[i]), &sn, &cs);
        c[i] = FM(cfg->m, f2u(cs));
        s[i] = FM(cfg->m, f2u(sn));
    }
}

void tsr_ref_rope(const tsr_ref_rope_cfg * cfg, uint32_t * d, uint32_t d_stride,
                  const uint32_t * src, uint32_t src_stride,
                  const uint32_t * pos, const uint32_t * table) {
    rope_check("tsr_ref_rope", cfg);
    if (cfg->T == 0u || cfg->n_heads == 0u) {          // T' = 0: nothing read or written
        return;
    }
    if (!d || !src || !pos) {
        vu_fatal("tsr_ref_rope", "missing operand", cfg->T, cfg->n_heads);
    }
    const uint32_t hd = cfg->head_dim, nd = cfg->n_dims, half = nd / 2u;
    const uint32_t H = (uint32_t)(((uint64_t)cfg->n_heads * hd) / 16u);    // lines per token
    const size_t ss = (size_t)(src_stride ? src_stride : H) * 16u;
    const size_t ds = (size_t)(d_stride ? d_stride : H) * 16u;
    uint32_t c[TSR_ROPE_MAX_HEAD_DIM / 2u], s[TSR_ROPE_MAX_HEAD_DIM / 2u];
    for (uint32_t t = 0; t < cfg->T; ++t) {
        tsr_ref_rope_cs(cfg, (int32_t)pos[t], table, c, s);
        for (uint32_t h = 0; h < cfg->n_heads; ++h) {
            const uint32_t * x = src + t * ss + (size_t)h * hd;
            uint32_t * y = d + t * ds + (size_t)h * hd;
            for (uint32_t i = 0; i < half; ++i) {
                const uint32_t j0 = (cfg->mode == TSR_ROPE_MODE_NEOX) ? i : 2u * i;
                const uint32_t j1 = (cfg->mode == TSR_ROPE_MODE_NEOX) ? i + half : 2u * i + 1u;
                const uint32_t x0 = x[j0], x1 = x[j1];          // both read before either write
                y[j0] = FS(FM(x0, c[i]), FM(x1, s[i]));
                y[j1] = FA(FM(x0, s[i]), FM(x1, c[i]));
            }
            for (uint32_t j = nd; j < hd; ++j) {
                y[j] = x[j];                           // raw tail copy (V2-VU-23)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// TOPK (VU §6.8)
// ---------------------------------------------------------------------------
// Candidate a ranks before b: larger value (NaN above all), then the smaller index.
static inline int topk_before(uint32_t va, uint32_t ja, uint32_t vb, uint32_t jb) {
    const int na = (va == B_NAN), nb = (vb == B_NAN);
    if (na != nb) {
        return na;
    }
    if (!na) {
        const uint32_t ka = tsr_ref_vu_key(va), kb = tsr_ref_vu_key(vb);
        if (ka != kb) {
            return ka > kb;
        }
    }
    return ja < jb;
}

void tsr_ref_topk(uint32_t k, uint32_t rows, uint32_t m, uint32_t * d, uint32_t d_stride,
                  const uint32_t * a, uint32_t a_stride) {
    if (k == 0u || k > TSR_TOPK_MAX_K) {
        vu_fatal("tsr_ref_topk", "k must be in 1..16", k, 0);
    }
    if (rows == 0u || m == 0u) {                       // n = 0 rule (VU §4.2)
        return;
    }
    if (!d || !a) {
        vu_fatal("tsr_ref_topk", "missing operand", rows, m);
    }
    const uint32_t dsl = d_stride ? d_stride : (k + 7u) / 8u;
    for (uint32_t r = 0; r < rows; ++r) {
        const uint32_t * x = a + row_off(r, a_stride);
        uint32_t vv[TSR_TOPK_MAX_K], jj[TSR_TOPK_MAX_K];
        uint32_t n = 0;                                // sorted list of the best n <= k candidates
        for (uint32_t j = 0; j < m; ++j) {
            const uint32_t v = is_nan_bits(x[j]) ? B_NAN : daz_bits(x[j]);
            if (n == k && !topk_before(v, j, vv[k - 1u], jj[k - 1u])) {
                continue;
            }
            uint32_t pos = (n < k) ? n++ : k - 1u;     // insertion point from the end
            while (pos > 0u && topk_before(v, j, vv[pos - 1u], jj[pos - 1u])) {
                vv[pos] = vv[pos - 1u];
                jj[pos] = jj[pos - 1u];
                --pos;
            }
            vv[pos] = v;
            jj[pos] = j;
        }
        uint32_t * y = d + row_off(r, dsl);
        for (uint32_t i = 0; i < k; ++i) {
            y[2u * i] = (i < n) ? jj[i] : 0xFFFFFFFFu;
            y[2u * i + 1u] = (i < n) ? vv[i] : B_NINF;
        }
        zero_tail(y, 2u * k);
    }
}

// ---------------------------------------------------------------------------
// SOP (VU §5.8)
// ---------------------------------------------------------------------------
uint32_t tsr_ref_sop(uint32_t sub, uint32_t sa, uint32_t sb, uint32_t imm, uint32_t mem) {
    switch (sub) {
    case TSR_SOP_SUB_MOVI:  return imm;                // raw
    case TSR_SOP_SUB_ADD:   return FA(sa, sb);
    case TSR_SOP_SUB_MUL:   return FM(sa, sb);
    case TSR_SOP_SUB_ADDI:  return FA(sa, imm);
    case TSR_SOP_SUB_MULI:  return FM(sa, imm);
    case TSR_SOP_SUB_SQRT:  return SQ(sa);
    case TSR_SOP_SUB_RECIP: return RC(sa);
    case TSR_SOP_SUB_LDS:   return mem;                // raw
    case TSR_SOP_SUB_STS:   return sa;                 // raw
    case TSR_SOP_SUB_SUB:   return FS(sa, sb);
    case TSR_SOP_SUB_DIV:   return FD(sa, sb);
    case TSR_SOP_SUB_RSQRT: return RC(SQ(sa));         // two roundings
    case TSR_SOP_SUB_MAX:   return tsr_ref_vu_maxt(sa, sb);
    case TSR_SOP_SUB_MIN:   return tsr_ref_vu_mint(sa, sb);
    case TSR_SOP_SUB_EXP:   return expk_p1(sa);
    default:
        vu_fatal("tsr_ref_sop", "undefined SOP sub", sub, 0);
    }
    return 0;
}

// ---------------------------------------------------------------------------
// DQROW (VU §5.6, §6.8)
// ---------------------------------------------------------------------------
void tsr_ref_dqrow(uint32_t fmt, uint32_t K, uint32_t rows, uint32_t * d, uint32_t d_stride,
                   const uint8_t * src, uint32_t src_stride) {
    if (!tsr_fmt_valid(fmt) || K == 0u || K % tsr_fmt_block_size(fmt)) {
        vu_fatal("tsr_ref_dqrow", "invalid fmt or K", fmt, K);
    }
    if (rows == 0u) {
        return;
    }
    if (!d || !src) {
        vu_fatal("tsr_ref_dqrow", "missing operand", rows, K);
    }
    const size_t ssb = (size_t)(src_stride ? src_stride : tsr_row_beats(fmt, K)) * TSR_LINE_BYTES;
    const uint32_t dsl = d_stride ? d_stride : tsr_ref_vu_lines(K);
    float * row = malloc((size_t)K * sizeof *row);
    if (!row) {
        vu_fatal("tsr_ref_dqrow", "out of memory", K, 0);
    }
    for (uint32_t r = 0; r < rows; ++r) {
        tsr_dequant_row(fmt, src + r * ssb, K, row);
        uint32_t * y = d + row_off(r, dsl);
        for (uint32_t j = 0; j < K; ++j) {
            y[j] = f2u(row[j]);
        }
        zero_tail(y, K);
    }
    free(row);
}
