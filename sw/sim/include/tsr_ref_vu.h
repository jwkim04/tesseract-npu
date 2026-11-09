// tsr_ref_vu.h — reference semantics of the VU ops VOP, RED, ROPE, TOPK, SOP and DQROW, bit for bit.
//
// Normative source: docs/design/02-vector-unit.md (VU) §4 (fp rules, operand model, c, aliasing),
// §5.2–§5.8 (functions, kinds, modes and what each field means), §6.2 (reductions), §6.7 (ROPE)
// and §6.8 (total order, raw ops, TOPK, DQROW). The element math is tsr_ref_math.c (EXPK and the
// composites, SIN/COS: VU §6.3–§6.6) and the f32 primitives FA, FS, FM, FD, SQ, RC are tsr_ref.c's.
// AT's vector ops (VU §5.11) are AT's references; VMASK (0x38) is not implemented and not here.
//
// These are references on plain word arrays: no simulator state and no encoding decode. The
// simulator's VU unit decodes and checks an op (VU §5.9) and calls these functions with
// pointers into the SPM; the cocotb benches call them through libtsr_ref.so (ctypes).
//
// Conventions
//   - An element is a 32-bit word (uint32_t, host order). The SPM is little-endian, so on a
//     little-endian host (every supported one) SPM line X is the word array (uint32_t *)(spm + 64·X).
//   - Operand pointers point at word 0 of row 0. Strides are in lines of 16 words, as in the
//     encodings: row r of an operand with stride s starts at word 16·r·s (VU §4.2). A source
//     stride of 0 repeats row 0 over all rows. With rows <= 1 strides are not used.
//   - s registers are passed as their 32-bit values (the caller reads s[si], s[sk], ... when the op
//     starts, VU §4.4).
//   - Each function writes exactly the destination extent of VU §5.9 and applies the tail rule
//     (VU §4.2, docs/spec.md "Tail bytes"): a destination row of n words occupies L(n) = ceil(n/16)
//     lines and words [n, 16·L(n)) are written 0. With m = 0 or rows = 0 nothing is read or written
//     (VU §4.2); the one exception, RED to an s register, is tsr_ref_red_row (it never writes SPM).
//   - In place: a source may be the destination exactly (same pointer and stride; for ROPE the same
//     start and token stride). Every function reads an element's sources (a ROPE pair's two
//     elements) before it writes that element. Any other overlap of a source with the destination
//     is undefined (VU §4.5) and not detected here.
//   - Codes are the encodings' (tsr_tables.h): TSR_VOP_FUNC_*, TSR_VOP_CMODE_*, TSR_RED_KIND_*,
//     TSR_RED_POST_*, TSR_ROPE_MODE_*, TSR_SOP_SUB_*, TSR_FMT_*. An undefined code, a missing
//     operand pointer, k outside 1..16 or an illegal ROPE shape abort() with a message, as the data
//     functions of tsr_ref.c do: callers check first (ILLEGAL_OP / BAD_FMT / K_RANGE / OPERAND at
//     dispatch, VU §5.9).
//   - Every fp result follows VU §4.1: DAZ inputs, RNE, FTZ, canonical NaN 0x7FC00000; raw
//     functions (COPY, SETC, SOP MOVI / LDS / STS, the ROPE tail) copy bits.
// All functions are thread-safe; none keeps state.
#pragma once

#include <stddef.h>
#include <stdint.h>

#include "tsr_isa.h"

#ifdef __cplusplus
extern "C" {
#endif

// L(n) = ceil(n/16): lines of a row of n words (VU §4.2).
static inline uint32_t tsr_ref_vu_lines(uint32_t n) { return (n >> 4) + ((n & 15u) != 0u); }

// ---------------------------------------------------------------------------
// Total order, MAXT, MINT (VU §6.8, VU-22)
// ---------------------------------------------------------------------------
// Order key of a non-NaN value after DAZ: sign ? ~bits : bits | 2^31, so that
// -inf < ... < -0 < +0 < ... < +inf as unsigned keys. (NaN is above every key; callers test it.)
uint32_t tsr_ref_vu_key(uint32_t x);
uint32_t tsr_ref_vu_maxt(uint32_t a, uint32_t b);   // NaN in either -> 0x7FC00000, else the larger DAZ'd value
uint32_t tsr_ref_vu_mint(uint32_t a, uint32_t b);   // both NaN -> 0x7FC00000, one NaN -> the other (DAZ'd), else the smaller

// ---------------------------------------------------------------------------
// VOP 0x30 (VU §5.2, §4.2–§4.3)
// ---------------------------------------------------------------------------
// One element y = func(a, b, c) of the VU §5.2 table, on bit patterns. Operands the function does
// not read are ignored.
uint32_t tsr_ref_vop_elem(uint32_t func, uint32_t a, uint32_t b, uint32_t c);

// rows x m elements: d[r][j] = func(a[r][j], b[r][j], c_r), then the tail rule per row.
//   a, b: the sources the function reads (tsr_vop_func_reads), NULL otherwise.
//   c:    cmode TSR_VOP_CMODE_NONE: not read (NULL allowed); TSR_VOP_CMODE_SREG: c[0] is s[si] for
//         every row; TSR_VOP_CMODE_VEC: c[r] for row r (the vector at SPM line C). A function that
//         reads c needs cmode != NONE, one that does not needs NONE (VU §5.2).
void tsr_ref_vop(uint32_t func, uint32_t cmode, uint32_t rows, uint32_t m,
                 uint32_t * d, uint32_t d_stride,
                 const uint32_t * a, uint32_t a_stride,
                 const uint32_t * b, uint32_t b_stride,
                 const uint32_t * c);

// ---------------------------------------------------------------------------
// RED 0x31 (VU §5.3, §6.2)
// ---------------------------------------------------------------------------
// The 128 partials of one SUM / SUMSQ row: p_i = +0, then for i = 0 .. m-1 in order
// p_(i mod 128) <- FA(p_(i mod 128), x_i) (SUMSQ: FM(x_i, x_i) in place of x_i).
void     tsr_ref_red_partials(uint32_t kind, const uint32_t * x, uint32_t m, uint32_t p[TSR_RED_PARTIALS]);
// The complete binary tree over p_0 .. p_127: level 1 FA(p_2k, p_2k+1), adjacent pairs left to
// right, 7 levels (docs/spec.md §9). This is the definition.
uint32_t tsr_ref_red_tree128(const uint32_t p[TSR_RED_PARTIALS]);
// The skip implementation that VU §6.2 allows (VU-2), as a model for the RTL: only partials
// below n_present = min(128, group·ceil(n_written/group)) exist; a node whose subtree holds none is
// skipped, a node with one existing child passes z(child) upward (z: -0 -> +0, else unchanged),
// and an all-skipped tree gives +0. group 1 skips at leaf granularity; group 8 is the hardware's
// 8-input subtree adder (VU §7.5: ceil(m/8) subtree sums for m < 128, the partials of a partly
// written group enter as +0). Bit-identical to tsr_ref_red_tree128 when the partials at and above
// n_written are +0 (the claim of VU §6.2, checked by test_ref_vu and ref_vu_np).
uint32_t tsr_ref_red_tree_skip(const uint32_t p[TSR_RED_PARTIALS], uint32_t n_written, uint32_t group);
// The row post-op on a row result y: NONE y; RCP RC(y); RMS RC(SQ(FA(FM(y, sk), se))).
uint32_t tsr_ref_red_post(uint32_t post, uint32_t y, uint32_t sk, uint32_t se);
// One row's result after the post-op. SUM / SUMSQ: the tree over the partials; MAX: the maximum
// of the DAZ'd elements under the total order, any NaN -> 0x7FC00000. Empty row (m = 0): SUM and
// SUMSQ +0, MAX -inf (0xFF800000), then the post-op. This is also the TO_S result (rows = 1,
// written to s[sd], also when m = 0).
uint32_t tsr_ref_red_row(uint32_t kind, uint32_t post, const uint32_t * x, uint32_t m,
                         uint32_t sk, uint32_t se);
// Per-row vector destination (TO_S = 0): d[r] = tsr_ref_red_row(row r) for r < rows, words
// [rows, 16·L(rows)) written 0. Writes nothing when m = 0 or rows = 0 (VU §4.2).
void     tsr_ref_red(uint32_t kind, uint32_t post, uint32_t rows, uint32_t m, uint32_t * d,
                     const uint32_t * a, uint32_t a_stride, uint32_t sk, uint32_t se);

// ---------------------------------------------------------------------------
// ROPE 0x32 (VU §5.4, §6.7, VU-19)
// ---------------------------------------------------------------------------
typedef struct tsr_ref_rope_cfg {
    uint32_t mode;        // TSR_ROPE_MODE_NORM (pairs 2i, 2i+1) or TSR_ROPE_MODE_NEOX (i, i + n_dims/2)
    uint32_t ff;          // FF: recurrence form, the table holds freq factors: phi_i = FD(theta_i, ff_i)
    uint32_t fs;          // FS: recurrence form, phi_i <- FM(freq_scale, phi_i)
    uint32_t atab;        // ATAB: A-table form (YaRN), phi_i = FM(Pf, A_i); ff and fs must be 0
    uint32_t m;           // s[sm]: M (attn_factor, or the folded YaRN mscale)
    uint32_t ts;          // s[sts]: TS = powf(freq_base, -2.0f/n_dims) (recurrence form)
    uint32_t freq_scale;  // s[sfs] (FS)
    uint32_t head_dim;    // multiple of 16 in [16, 1024]
    uint32_t n_dims;      // [1, head_dim]; multiple of 32 (NEOX) or 16 (NORM)
    uint32_t n_heads;     // heads per token (one token's heads are contiguous: H = n_heads·head_dim/16 lines)
    uint32_t T;           // tokens
} tsr_ref_rope_cfg;

// The angles phi_0 .. phi_(n_dims/2 - 1) of one position (VU §6.7). Recurrence form: theta_0 =
// tsr_i32_to_f32(pos), theta_(i+1) = FM(theta_i, TS); phi_i = FF ? FD(theta_i, table[i]) : theta_i;
// FS: phi_i <- FM(freq_scale, phi_i). A-table form: phi_i = FM(Pf, table[i]). table may be NULL
// when neither FF nor ATAB is set. Bit-identical to ggml-cpu's ggml_rope_cache_init angles for
// ext_factor = 0 (VU-19; test_ref_vu checks it against the pinned ggml-cpu).
void tsr_ref_rope_angles(const tsr_ref_rope_cfg * cfg, int32_t pos, const uint32_t * table, uint32_t * phi);
// c_i = FM(M, COS(phi_i)), s_i = FM(M, SIN(phi_i)) for i < n_dims/2.
void tsr_ref_rope_cs(const tsr_ref_rope_cfg * cfg, int32_t pos, const uint32_t * table,
                     uint32_t * c, uint32_t * s);
// T tokens x n_heads heads. Token t's heads start at src + 16·t·src_stride and d + 16·t·d_stride
// (stride 0 = H lines, VU §5.4); head h is at word h·head_dim; pos[t] is token t's int32 position.
// Pair (j0, j1): y[j0] = FS(FM(x0, c_i), FM(x1, s_i)), y[j1] = FA(FM(x0, s_i), FM(x1, c_i));
// words [n_dims, head_dim) of each head are copied raw. Nothing is read or written when T = 0 or
// n_heads = 0 (VU §5.9: T' = 0).
void tsr_ref_rope(const tsr_ref_rope_cfg * cfg, uint32_t * d, uint32_t d_stride,
                  const uint32_t * src, uint32_t src_stride,
                  const uint32_t * pos, const uint32_t * table);

// ---------------------------------------------------------------------------
// TOPK 0x33 (VU §5.5, §6.8, VU-14)
// ---------------------------------------------------------------------------
// Per row: candidates (j, x^_j), x^ = DAZ(x) with every NaN as 0x7FC00000, sorted by x^ descending
// under the total order (NaN above everything), ties by smaller j first. Pair i < k: word 2i = j
// (u32), word 2i+1 = x^; with m < k the missing pairs are (0xFFFFFFFF, 0xFF800000). Row r's
// output is at d + 16·r·d_stride' (d_stride' = d_stride, or ceil(k/8) if 0), 2k words, tail rule.
void tsr_ref_topk(uint32_t k, uint32_t rows, uint32_t m, uint32_t * d, uint32_t d_stride,
                  const uint32_t * a, uint32_t a_stride);

// ---------------------------------------------------------------------------
// MoE routing (docs/spec.md §13; SW §7.5 MOE_ROUTE, SW-61)
// ---------------------------------------------------------------------------
// One token's expert selection as the MOE_ROUTE kind computes it: a composition of the VU
// references above, with the selection itself done by integer compares on the RISC-V (the TOPK RTL
// stays disabled). n router logits (f32 bits), 1 <= k <= min(n, 16), n <= TSR_MOE_MAX_EXPERTS.
//   m  = RED.MAX(logits)                          (any NaN: canonical NaN)
//   e  = VOP.EXPSUBC(logits, c = m)               EXPSUB(l, m) = EXPK(FS(l, m))
//   r  = RED.SUM post RCP (e)                     the 128-partial tree, then RC
//   p  = VOP.MULC(e, c = r)                       the softmax probabilities
//   (id_j, q_j) = TOPK(p, k)                      value descending under the total order (NaN first),
//                                                 ties by the smaller index (VU §6.8); q_j is the DAZ'd value
//   r2 = RED.SUM post RCP (q_0 .. q_(k-1))       the tree over the k selected probabilities
//   w_j = VOP.MULC(q_j, c = r2)                   the renormalised weights, in selection order
// ggml's graph computes the same quantities as soft_max, argsort(DESC) + view, get_rows, sum_rows,
// clamp(6.1035e-5, inf) and div; the clamp never acts because the k largest of n probabilities sum
// to at least k / n (a caller that fuses the graph checks k / n >= the clamp), and a NaN passes it.
// ids[0..k), w[0..k) are written; probs (n words) receives p when it is not NULL.
#define TSR_MOE_MAX_EXPERTS 512u
void tsr_ref_moe_route(uint32_t n, uint32_t k, const uint32_t * logits, uint32_t * ids, uint32_t * w,
                       uint32_t * probs);

// ---------------------------------------------------------------------------
// SOP 0x40 (VU §5.8)
// ---------------------------------------------------------------------------
// The value an SOP writes: to s[sd] for every sub except STS, to the SPM word for STS. sa, sb are
// s[sa], s[sb]; imm is w2; mem is the SPM word at the w3 word address (read by LDS only).
//   MOVI imm, ADD FA(sa,sb), MUL FM, ADDI FA(sa,imm), MULI FM(sa,imm), SQRT SQ(sa), RECIP RC(sa),
//   LDS mem, STS sa, SUB FS, DIV FD, RSQRT RC(SQ(sa)), MAX MAXT, MIN MINT, EXP EXPK(sa, P1).
uint32_t tsr_ref_sop(uint32_t sub, uint32_t sa, uint32_t sb, uint32_t imm, uint32_t mem);

// ---------------------------------------------------------------------------
// DQROW 0x12 (VU §5.6, §6.8, VU-15)
// ---------------------------------------------------------------------------
// rows rows of K elements: row r = tsr_dequant_row(fmt, src + 64·r·src_stride', K) as f32 bits at
// d + 16·r·d_stride', tail rule. src points at row 0's repacked bytes in the SPM (as LD loaded them
// from DRAM); src_stride' = src_stride, or ROW_BEATS(fmt, K) if 0; d_stride' = d_stride, or L(K)
// if 0. fmt and K must be valid (tsr_fmt_k_valid; BAD_FMT / K_RANGE at dispatch).
void tsr_ref_dqrow(uint32_t fmt, uint32_t K, uint32_t rows, uint32_t * d, uint32_t d_stride,
                   const uint8_t * src, uint32_t src_stride);

#ifdef __cplusplus
}
#endif
