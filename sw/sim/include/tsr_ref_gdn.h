// tsr_ref_gdn.h — reference semantics of the fused Gated DeltaNet state update (GDN, VU opcode 0x39),
// bit for bit.
//
// Normative source: docs/design/02-vector-unit.md (VU) §7.12 (the op, its operands and encoding,
// the numerics in §7.12.3) and decisions VU-28 … VU-33. The f32 primitives are tsr_ref.c's
// (FA tsr_fadd, FS tsr_fsub, FM tsr_fmul: DAZ, RNE, FTZ, canonical NaN); the dot products are not
// f32 sums. They are exact sums of integer products in a block-floating-point frame, converted to
// f32 once (VU-29).
//
// The op, per head (S is `rows` x m f32, row-major, row stride m words; m is a multiple of 16):
//     u_i = S_i . k            e_i = beta * (v_i - alpha * u_i)
//     S'_i = alpha * S_i + e_i * k          o_i = S'_i . q
// The reference evaluates it as (all FP32 operations are the VU primitives):
//     ab   = FM(alpha, beta)        kp_j = FM(ab, k_j)        bv_i = FM(beta, v_i)
//     TK   = table(kp)              TQ   = table(q)           (head frames, 24-bit entries)
//     kap  = rowdot(k, TQ)          u'_i = rowdot(S_i, TK)    w_i = rowdot(S_i, TQ)
//     e_i  = FS(bv_i, u'_i)
//     S'_ij = FA(FM(alpha, S_ij), FM(e_i, k_j))
//     o_i  = FA(FM(alpha, w_i), FM(e_i, kap))
// u' is alpha*beta*(S_i . k) and w_i is S_i . q, so o_i = alpha*(S_i . q) + e_i*(k . q) = S'_i . q.
//
// table(x): DAZ; Emax = largest biased exponent of the nonzero elements (0 if none); element j is
//   the signed integer sign * round_half_up(M_j / 2^(Emax - E_j)) with M_j the 24-bit significand
//   (0 for a zero element and for a shift of 25 or more). The value of entry j is entry * 2^(Emax-150).
// rowdot(row, T): Es = largest biased exponent of the row's nonzero elements (0 if none); element j
//   is A_j = sign * ((M_j << 2) >> (Es - E_j)) (26-bit window, truncation; 0 for a shift of 26 or
//   more). I = sum_j A_j * T_j exactly (|I| < 2^58 for m <= 256), result = f32(I * 2^(Es + Emax_T
//   - 302)) rounded to nearest even, overflow to +-Inf, a result below 2^-126 - 2^-150 flushed to a
//   signed zero as in FM (the spec's FTZ rule), I = 0 gives +0.
// Non-finite values: any Inf or NaN among the row's elements or the table's elements makes the
//   rowdot result the canonical NaN 0x7FC00000 (the row, or the head, is poisoned).
//
// These are references on plain word arrays: no simulator state, no encoding decode. All functions
// are thread-safe; none keeps state. A source may be the destination exactly (in place).
#pragma once

#include <stddef.h>
#include <stdint.h>

#include "tsr_isa.h"

#ifdef __cplusplus
extern "C" {
#endif

#define TSR_GDN_M_MAX     256u   // d_k limit of the unit (the 48-bit DSP accumulator bound, VU §7.12.3)
#define TSR_GDN_S_WINDOW  26     // bits of a row element after alignment
#define TSR_GDN_T_WINDOW  24     // bits of a table entry

typedef struct tsr_gdn_table {
    uint32_t m;                         // elements
    uint32_t exp;                       // Emax, biased (0 if every element is zero)
    uint32_t bad;                       // 1 if any element is Inf or NaN
    int32_t  v[TSR_GDN_M_MAX];          // signed entries, |v| < 2^24 (0 where the element is zero or bad)
} tsr_gdn_table;

// f32 bits of (-1)^neg * mag * 2^x: RNE, overflow to Inf, FTZ below 2^-126 - 2^-150, mag = 0 gives +0.
uint32_t tsr_ref_gdn_fx2f(uint32_t neg, uint64_t mag, int32_t x);

// Head table of x[0..m): m <= TSR_GDN_M_MAX.
void     tsr_ref_gdn_table(tsr_gdn_table * t, const uint32_t * x, uint32_t m);

// f32 bits of the dot product of one row (t->m words) with a table.
uint32_t tsr_ref_gdn_rowdot(const uint32_t * row, const tsr_gdn_table * t);

// S'_ij of one element: FA(FM(alpha, s), FM(e, k)).
uint32_t tsr_ref_gdn_upd_elem(uint32_t alpha, uint32_t s, uint32_t e, uint32_t k);

// The scalars of the head (VU §7.12.3): kappa = rowdot(k, TQ), e_i, o_i from u'_i and w_i.
uint32_t tsr_ref_gdn_kappa(const uint32_t * k, const tsr_gdn_table * tq);
uint32_t tsr_ref_gdn_e(uint32_t beta, uint32_t v, uint32_t u);                       // FS(FM(beta, v), u)
uint32_t tsr_ref_gdn_o(uint32_t alpha, uint32_t w, uint32_t e, uint32_t kappa);      // FA(FM(alpha, w), FM(e, kappa))

// One head: `rows` rows of m words at s_in (row stride m words) become S' at s_out (s_out may equal
// s_in), o[0..rows) the outputs; words [rows, 16 * ceil(rows / 16)) of o are written 0 (the tail
// rule of VU §4.2). m must be a multiple of 16 in [16, TSR_GDN_M_MAX], else abort (callers check
// at dispatch). rows = 0 reads and writes nothing.
void tsr_ref_gdn(uint32_t m, uint32_t rows, uint32_t alpha, uint32_t beta,
                 const uint32_t * k, const uint32_t * q, const uint32_t * v,
                 const uint32_t * s_in, uint32_t * s_out, uint32_t * o);

#ifdef __cplusplus
}
#endif
