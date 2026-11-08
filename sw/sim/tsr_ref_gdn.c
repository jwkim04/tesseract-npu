// tsr_ref_gdn.c — reference semantics of the fused Gated DeltaNet state update (GDN), bit for bit.
//
// Normative source: docs/design/02-vector-unit.md §7.12 (numerics: §7.12.3) and VU-28 … VU-37.
// The API and the definitions are in tsr_ref_gdn.h.
//
// Implementation rules (as tsr_ref_vu.c, docs/design/06-software-verification.md §4.5)
//   - Every f32 operation is a call to the tsr_ref.c primitives (FA tsr_fadd, FS tsr_fsub, FM
//     tsr_fmul); everything else is integer arithmetic. No float operators.
//   - A row is read completely (both dot products) before its words are written, so s_out may
//     equal s_in.

#include "tsr_ref_gdn.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tsr_ref.h"

#if defined(__FAST_MATH__)
#error "tsr_ref_gdn.c must not be compiled with -ffast-math"
#endif

#define F32_SIGN 0x80000000u
#define F32_INF  0x7F800000u
#define B_NAN    TSR_CANON_NAN_BITS

static inline uint32_t f2u(float f) { uint32_t u; memcpy(&u, &f, sizeof u); return u; }
static inline float u2f(uint32_t u) { float f; memcpy(&f, &u, sizeof f); return f; }

static inline uint32_t FA(uint32_t a, uint32_t b) { return f2u(tsr_fadd(u2f(a), u2f(b))); }
static inline uint32_t FS(uint32_t a, uint32_t b) { return f2u(tsr_fsub(u2f(a), u2f(b))); }
static inline uint32_t FM(uint32_t a, uint32_t b) { return f2u(tsr_fmul(u2f(a), u2f(b))); }

static void gdn_fatal(const char * fn, const char * msg, uint32_t a) {
    fprintf(stderr, "tsr_ref_gdn: %s: %s (%u)\n", fn, msg, a);
    abort();
}

// ---------------------------------------------------------------------------
// Fixed point to f32 (VU §7.12.3, the conversion rule)
// ---------------------------------------------------------------------------
static inline uint32_t bitlen64(uint64_t v) {   // v != 0
    return 64u - (uint32_t)__builtin_clzll(v);
}

uint32_t tsr_ref_gdn_fx2f(uint32_t neg, uint64_t mag, int32_t x) {
    if (mag == 0) {
        return 0u;
    }
    const uint32_t sign = neg ? F32_SIGN : 0u;
    const int32_t  len  = (int32_t)bitlen64(mag);          // 1 .. 64
    const int32_t  e0   = len - 1 + x + 127;               // biased exponent of the leading bit
    if (e0 >= 255) {
        return sign | F32_INF;
    }
    if (e0 >= 1) {
        uint64_t q;
        int32_t  e = e0;
        if (len <= 24) {
            q = mag << (24 - len);
        } else {
            const uint32_t sh   = (uint32_t)(len - 24);
            const uint64_t rem  = mag & ((1ull << sh) - 1ull);
            const uint64_t half = 1ull << (sh - 1u);
            q = mag >> sh;
            if (rem > half || (rem == half && (q & 1ull))) {
                q += 1ull;
            }
            if (q == (1ull << 24)) {
                q >>= 1;
                e += 1;
            }
        }
        if (e >= 255) {
            return sign | F32_INF;
        }
        return sign | ((uint32_t)e << 23) | (uint32_t)(q & 0x7FFFFFu);
    }
    // Below 2^-126: round to a multiple of 2^-149 (RNE, gradual underflow), then flush as FM does:
    // only a result of exactly 2^-126 survives.
    const int32_t t = x + 149;                             // value = mag * 2^t units of 2^-149
    uint64_t qq;
    if (t >= 0) {
        qq = mag << t;                                     // e0 < 1 keeps this below 2^23
    } else {
        const int32_t sh = -t;
        if (sh >= 64) {
            qq = 0;
        } else {
            const uint64_t rem  = mag & ((1ull << sh) - 1ull);
            const uint64_t half = 1ull << (sh - 1);
            qq = mag >> sh;
            if (rem > half || (rem == half && (qq & 1ull))) {
                qq += 1ull;
            }
        }
    }
    return (qq >= (1ull << 23)) ? (sign | (1u << 23)) : sign;
}

// ---------------------------------------------------------------------------
// Head table and row dot product
// ---------------------------------------------------------------------------
void tsr_ref_gdn_table(tsr_gdn_table * t, const uint32_t * x, uint32_t m) {
    if (m > TSR_GDN_M_MAX) {
        gdn_fatal("tsr_ref_gdn_table", "m too large", m);
    }
    uint32_t emax = 0, bad = 0;
    for (uint32_t j = 0; j < m; ++j) {
        const uint32_t e = (x[j] >> 23) & 0xFFu;
        if (e == 0xFFu) {
            bad = 1;
        } else if (e > emax) {
            emax = e;
        }
    }
    t->m = m;
    t->exp = emax;
    t->bad = bad;
    for (uint32_t j = 0; j < m; ++j) {
        const uint32_t e = (x[j] >> 23) & 0xFFu;
        int32_t v = 0;
        if (e != 0u && e != 0xFFu) {
            const uint32_t mant = (x[j] & 0x7FFFFFu) | 0x800000u;
            const uint32_t sh = emax - e;
            uint32_t mag;
            if (sh == 0u) {
                mag = mant;
            } else if (sh >= 25u) {
                mag = 0u;
            } else {
                mag = (mant + (1u << (sh - 1u))) >> sh;
            }
            v = (x[j] & F32_SIGN) ? -(int32_t)mag : (int32_t)mag;
        }
        t->v[j] = v;
    }
    for (uint32_t j = m; j < TSR_GDN_M_MAX; ++j) {
        t->v[j] = 0;
    }
}

uint32_t tsr_ref_gdn_rowdot(const uint32_t * row, const tsr_gdn_table * t) {
    const uint32_t m = t->m;
    uint32_t es = 0, bad = t->bad;
    for (uint32_t j = 0; j < m; ++j) {
        const uint32_t e = (row[j] >> 23) & 0xFFu;
        if (e == 0xFFu) {
            bad = 1;
        } else if (e > es) {
            es = e;
        }
    }
    if (bad) {
        return B_NAN;
    }
    int64_t acc = 0;
    for (uint32_t j = 0; j < m; ++j) {
        const uint32_t e = (row[j] >> 23) & 0xFFu;
        if (e == 0u) {
            continue;
        }
        const uint64_t mant = (row[j] & 0x7FFFFFu) | 0x800000u;
        const uint32_t sh = es - e;
        const int64_t a = (sh >= 26u) ? 0 : (int64_t)((mant << 2) >> sh);
        const int64_t term = a * (int64_t)t->v[j];
        acc += (row[j] & F32_SIGN) ? -term : term;
    }
    const uint32_t neg = acc < 0;
    const uint64_t mag = neg ? (uint64_t)0 - (uint64_t)acc : (uint64_t)acc;
    return tsr_ref_gdn_fx2f(neg, mag, (int32_t)es + (int32_t)t->exp - 302);
}

// ---------------------------------------------------------------------------
// Scalars and the element update
// ---------------------------------------------------------------------------
uint32_t tsr_ref_gdn_upd_elem(uint32_t alpha, uint32_t s, uint32_t e, uint32_t k) {
    return FA(FM(alpha, s), FM(e, k));
}

uint32_t tsr_ref_gdn_kappa(const uint32_t * k, const tsr_gdn_table * tq) {
    return tsr_ref_gdn_rowdot(k, tq);
}

uint32_t tsr_ref_gdn_e(uint32_t beta, uint32_t v, uint32_t u) {
    return FS(FM(beta, v), u);
}

uint32_t tsr_ref_gdn_o(uint32_t alpha, uint32_t w, uint32_t e, uint32_t kappa) {
    return FA(FM(alpha, w), FM(e, kappa));
}

// ---------------------------------------------------------------------------
// One head
// ---------------------------------------------------------------------------
void tsr_ref_gdn(uint32_t m, uint32_t rows, uint32_t alpha, uint32_t beta,
                 const uint32_t * k, const uint32_t * q, const uint32_t * v,
                 const uint32_t * s_in, uint32_t * s_out, uint32_t * o) {
    if (m < 16u || m > TSR_GDN_M_MAX || (m & 15u) != 0u) {
        gdn_fatal("tsr_ref_gdn", "m must be a multiple of 16 in [16, TSR_GDN_M_MAX]", m);
    }
    if (rows == 0u) {
        return;
    }
    uint32_t kp[TSR_GDN_M_MAX];
    tsr_gdn_table tk, tq;
    const uint32_t ab = FM(alpha, beta);
    for (uint32_t j = 0; j < m; ++j) {
        kp[j] = FM(ab, k[j]);
    }
    tsr_ref_gdn_table(&tk, kp, m);
    tsr_ref_gdn_table(&tq, q, m);
    const uint32_t kappa = tsr_ref_gdn_kappa(k, &tq);
    for (uint32_t i = 0; i < rows; ++i) {
        const uint32_t * si = s_in + (size_t)i * m;
        uint32_t * so = s_out + (size_t)i * m;
        const uint32_t u = tsr_ref_gdn_rowdot(si, &tk);
        const uint32_t w = tsr_ref_gdn_rowdot(si, &tq);
        const uint32_t e = tsr_ref_gdn_e(beta, v[i], u);
        for (uint32_t j = 0; j < m; ++j) {
            so[j] = tsr_ref_gdn_upd_elem(alpha, si[j], e, k[j]);
        }
        o[i] = tsr_ref_gdn_o(alpha, w, e, kappa);
    }
    const uint32_t end = 16u * ((rows + 15u) / 16u);
    for (uint32_t i = rows; i < end; ++i) {
        o[i] = 0u;
    }
}
