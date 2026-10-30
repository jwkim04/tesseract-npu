// tsr_ref_at.h — references of area AT (attention and the KV cache on the device), bit for bit.
//
// Normative source: docs/design/03-attention-kv.md (AT): the KV plane layout §4, the ops KVST §5.1,
// ATTQK §5.2, ATTPV §5.3, ASMAX §5.4, AMASK §5.5, AMERGE §5.6, ANORM §5.7, their numerics §6.1–§6.5,
// the composite flash attention §6.6 (normative kernel algorithm), the mask summary MSUM §7.3 and
// the reference list §13.1 (which names tsr_ref.c; the AT references live in their own file of the
// same library, tsr_ref / libtsr_ref.so). The f32 primitives FA / FS / FM / RC, the fp16
// conversions, QACT and the Q8_0 row dot product are tsr_ref.c's; EXP is tsr_m_exp
// (tsr_ref_math.c, VU §6.3). C callers holding uint8_t * channel pointers cast them to
// (const uint8_t * const *) for the read-only plane parameters.
//
// These are references on plain arrays: no simulator state and no encoding decode. The
// simulator's AT unit semantics (P1.5) decode and check an op (AT §5.8) and call these functions
// with pointers into the SPM and the channel DRAM; the firmware's host reference (sw/ctrl/ref)
// composes them in kernel order; the backend uses the layout and MSUM functions for KV tensors
// (get/set_tensor, SW-17) and masks; the cocotb benches call them through libtsr_ref.so.
//
// Conventions
//   - KV planes (AT §4) are passed as one pointer per channel: plane[c] points at byte 64·B of
//     channel c (B = the plane base beat, the same in every channel, AT §4.2), so a DRAM beat
//     B + k of channel c is plane[c] + 64 k. A plane holds 16·C·LG cells; each channel image is
//     tsr_kv_plane_beats() beats.
//   - SPM operands are f32 word arrays (float *, host order; the SPM is little-endian and every
//     supported host is, as for tsr_qa_d). Line X is (float *)(spm + 64 X); strides are in lines
//     of 16 words, as in the encodings, and a stride field of 0 takes the op's default (AT §5).
//     QA operands are byte images exactly as tsr_qact writes them.
//   - Each op function writes exactly the words the op defines and nothing else: rows of other
//     passes (another hg0, AT §5.2) and the unused words of the last line of a row vector are left
//     unchanged (decision recorded in tsr_ref_at.c's header; see AT §5.2, §5.3).
//   - Operands must not overlap, except dst == src (same pointer and stride) for ASMAX and AMASK,
//     which work in place element by element.
//   - Every fp result follows the spec's rules (DAZ inputs, RNE, FTZ, canonical NaN 0x7FC00000).
//   - Preconditions (shapes, ranges) are the simulator's dispatch checks (AT §5.8). A violated
//     precondition abort()s with a message, as the data functions of tsr_ref.c do. tsr_ref_fa
//     and tsr_fa_plan return error codes instead (they compose ops for a whole ggml op).
// All functions are thread-safe; none keeps state.
#pragma once

#include <stddef.h>
#include <stdint.h>

#include "tsr_isa.h"
#include "tsr_ref.h"

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// KV plane layout (AT §4)
// ---------------------------------------------------------------------------
#define TSR_KV_K 0u                 // K plane: token-major Q8_0 chunks (AT §4.3); = KVST sub 0
#define TSR_KV_V 1u                 // V plane: slice-major (AT §4.4); = KVST sub 1

typedef struct tsr_kv_geom {
    uint32_t E;     // values per cell row (n_head_kv·D): E % 64 == 0, E > 0
    uint32_t C;     // NUM_CH, 1..TSR_NUM_CH_MAX
    uint32_t LG;    // local groups per channel (>= 1); the plane holds 16·C·LG cells
} tsr_kv_geom;

// LG >= ceil(kv_size / (16 C)) (AT §4.1): the smallest LG for kv_size cells.
uint32_t tsr_kv_lg_min(uint32_t kv_size, uint32_t C);
// GB = 17·E/64: beats per group per plane (AT §4.1).
uint32_t tsr_kv_group_beats(uint32_t E);
// Beats one plane occupies in each channel: LG·GB (K and V alike, AT §4.4).
uint64_t tsr_kv_plane_beats(const tsr_kv_geom * g);
// Cells a plane holds: 16·C·LG.
uint64_t tsr_kv_plane_cells(const tsr_kv_geom * g);

// Where block blk (0 <= blk < E/32) of cell `cell` lives (AT §4.3 K, §4.4 V): channel, and byte
// offsets relative to plane[ch] of its 2-byte fp16 scale (meta) and of its 32 quant bytes.
typedef struct tsr_kv_loc {
    uint32_t ch;
    uint64_t meta;    // byte offset of the fp16 d (2 bytes)
    uint64_t quant;   // byte offset of q[32]
} tsr_kv_loc;
void tsr_kv_addr_k(const tsr_kv_geom * g, uint32_t cell, uint32_t blk, tsr_kv_loc * loc);
void tsr_kv_addr_v(const tsr_kv_geom * g, uint32_t cell, uint32_t blk, tsr_kv_loc * loc);

// One ggml q8_0 row (E/32 blocks of {fp16 d, int8 qs[32]}, i.e. llama.cpp's cache row of cell
// `cell`) into / out of a plane (kv = TSR_KV_K or TSR_KV_V). A pure byte permutation (AT §4.6).
void tsr_kv_pack_row(uint32_t kv, const tsr_kv_geom * g, uint8_t * const plane[], uint32_t cell,
                     const tsr_block_q8_0 * row);
void tsr_kv_unpack_row(uint32_t kv, const tsr_kv_geom * g, const uint8_t * const plane[], uint32_t cell,
                       tsr_block_q8_0 * row);
// Byte ranges of one stream of a ggml KV tensor [E, kv_size] q8_0 (row j = cell j, row size
// E/32·34 bytes): bytes [off, off + size) of the ggml image go to / come from the plane
// (set_tensor / get_tensor, SW-17). Any byte range is allowed, partial blocks and rows
// included; off + size must not exceed 16·C·LG rows. The caller splits a multi-stream tensor at
// stream boundaries (one plane per stream, AT §4.5).
void tsr_kv_set_bytes(uint32_t kv, const tsr_kv_geom * g, uint8_t * const plane[],
                      const void * src, size_t off, size_t size);
void tsr_kv_get_bytes(uint32_t kv, const tsr_kv_geom * g, const uint8_t * const plane[],
                      void * dst, size_t off, size_t size);

// ---------------------------------------------------------------------------
// KVST (AT §5.1): store n_rows QA(A32, E) vectors into cells c0 .. c0 + n_rows - 1
// ---------------------------------------------------------------------------
// Beats a KVST writes and reads, per channel (cycle model "40 + beats + RMW reads", AT §13.4;
// PERF KVST_RMW, AT §10). A meta beat the op fully covers is written without a read; a partially
// covered one (K with E % 1024 != 0 or a row spanning chunks, V unless the op writes all 16 cells
// of the group) is read-modify-written (AT §5.1).
typedef struct tsr_kvst_stats {
    uint32_t meta_beats[TSR_NUM_CH_MAX];    // meta beats written
    uint32_t meta_rmw[TSR_NUM_CH_MAX];      // of those, read-modify-written (one beat read each)
    uint32_t quant_beats[TSR_NUM_CH_MAX];   // quant beats written (always full)
} tsr_kvst_stats;
void tsr_kvst_count(uint32_t kv, const tsr_kv_geom * g, uint32_t c0, uint32_t n_rows, tsr_kvst_stats * st);

// src: QA vector i at src + 64·i·src_stride (src_stride 0 -> QA_LINES(A32, E)); its q bytes are
// QS[0..E), its scales fp16(D[b]) (tsr_f32_to_fp16, exact for QACT output). Bytes are placed per
// AT §4.3 / §4.4; every other byte of the plane is unchanged (the RMW merge). st may be NULL.
// Preconditions: kv <= 1, 1 <= n_rows <= 64, c0 + n_rows <= 16·C·LG.
void tsr_ref_kvst(uint32_t kv, const tsr_kv_geom * g, uint8_t * const plane[], uint32_t c0, uint32_t n_rows,
                  const uint8_t * src, uint32_t src_stride, tsr_kvst_stats * st);

// ---------------------------------------------------------------------------
// Shape shared by ATTQK and ATTPV (AT §5)
// ---------------------------------------------------------------------------
typedef struct tsr_att_shape {
    uint32_t E;     // row values per cell (w4[8:0]·64)
    uint32_t D;     // head dimension 64, 128, 256 or 512 (w4[10:9])
    uint32_t G;     // heads per KV head, 1..64 (x16[5:0] + 1)
    uint32_t hg0;   // first head of the pass within each KV-head group (x16[11:6])
    uint32_t GS;    // heads of the pass per KV head (w4[16:14] + 1)
    uint32_t T_q;   // query tokens of the pass (w4[13:11] + 1)
} tsr_att_shape;

static inline uint32_t tsr_att_n_head(const tsr_att_shape * s) { return s->G * (s->E / s->D); }
static inline uint32_t tsr_att_R(const tsr_att_shape * s) { return s->T_q * tsr_att_n_head(s); }
// Row r(q, g, i) = q·n_head + g·G + hg0 + i (q < T_q, g < E/D, i < GS) and slot t(q, i) = q·GS + i.
static inline uint32_t tsr_att_row(const tsr_att_shape * s, uint32_t q, uint32_t g, uint32_t i) {
    return q * tsr_att_n_head(s) + g * s->G + s->hg0 + i;
}
static inline uint32_t tsr_att_slot(const tsr_att_shape * s, uint32_t q, uint32_t i) { return q * s->GS + i; }
// Groups of [u0, u0 + n_groups) that channel c streams (u = c mod C): an ATTQK or ATTPV reads
// that many times GB beats of channel c (cycle model, AT §13.4; CH_RD_BEATS).
static inline uint32_t tsr_att_ch_groups(uint32_t C, uint32_t u0, uint32_t n_groups, uint32_t c) {
    const uint32_t first = u0 + (c + C - u0 % C) % C;   // first group of channel c at or after u0
    return first >= u0 + n_groups ? 0u : (u0 + n_groups - 1u - first) / C + 1u;
}

// Flag bits of the reference calls (= the w0 flag bits of the ops, shifted down by 12).
#define TSR_ATT_CAUSAL 0x1u   // ATTQK w0[12]
#define TSR_ATT_SLIDE  0x2u   // ATTQK w0[13]
#define TSR_ATT_MAXOUT 0x4u   // ATTQK w0[14]
#define TSR_ATT_LSUM   0x1u   // ATTPV w0[12]
#define TSR_ATT_INIT   0x1u   // AMERGE w0[12]

// ---------------------------------------------------------------------------
// ATTQK (AT §5.2, §6.2, §6.3)
// ---------------------------------------------------------------------------
// SEG: the spec's Q8_0 row rule on one D-element segment (AT §6.2): t_k = (dK_k (*) Dx_k) (*)
// f32(sum_i qK_k[i]·qx[32k + i]), p_k = +0 (+) t_k for k < D/32, p_k = +0 otherwise, result
// tsr_tree16(p). kseg = the segment's D/32 ggml blocks, qx = the head's D quants, dx = its D/32
// QA scales. Equals tsr_dot_row(Q8_0, segment repacked as a row, D, head QA) (tested).
float tsr_att_seg(uint32_t D, const tsr_block_q8_0 * kseg, const int8_t * qx, const float * dx);

// Scores of the pass rows over groups [u0, u0 + n_groups) of the K plane (kplane: C channel
// pointers; the K layout does not depend on LG). For pass row r(q, g, i) and cell j_rel:
//   lo_q = max(0, c_lo - SLIDE·(T_q-1-q)), hi_q = c_hi - CAUSAL·(T_q-1-q)   (signed)
//   s = (lo_q <= j_rel < hi_q) ? SEG(K row of cell 16 u0 + j_rel, segment g; x_q, head g·G+hg0+i) : -inf
// at word j_rel of row r, rows at dst + r·rstride lines (rstride 0 -> n_groups). x: T_q QA(A32,
// n_head·D) vectors at stride QA_LINES(A32, n_head·D). MAXOUT: word r of mdst =
// MAXT(M_FLOOR, MAXT over the row's 16·n_groups scores) (mdst unused without MAXOUT).
// Preconditions: E % D == 0, D in {64, 128, 256, 512}, 1 <= GS <= 8, hg0 + GS <= G <= 64,
// 1 <= T_q <= 8, 1 <= n_groups, c_lo <= c_hi <= 16·n_groups. Cells outside every row's bounds are
// not read from the plane.
void tsr_ref_attqk(const tsr_att_shape * sh, uint32_t C, const uint8_t * const kplane[],
                   uint32_t u0, uint32_t n_groups, uint32_t c_lo, uint32_t c_hi, uint32_t flags,
                   const uint8_t * x, float * dst, uint32_t rstride, float * mdst);

// ---------------------------------------------------------------------------
// ATTPV (AT §5.3, §6.4, §6.5): block floating point per 64-cell channel segment
// ---------------------------------------------------------------------------
// eps(w) = floor(log2 |w|) of a normal f32 (DAZ: callers pass only finite nonzero normals).
int32_t tsr_att_ilog2(float w);
// RNE_int(w·2^(23 - e_s)) for a finite w with eps(w) <= e_s: an exact power-of-two scaling and
// one round-half-even to an integer; |result| <= 2^24. A subnormal or zero w gives 0.
int32_t tsr_att_bf_int(float w, int32_t e_s);
// F(A, k) = RNE24(A)·2^k: A rounded to 24 significant bits (RNE), then scaled exactly; a result
// below 2^-126 in magnitude is a signed zero (sign of A; A = 0 gives +0), one >= 2^128 is +-inf.
float tsr_att_bf_f(int64_t A, int32_t k);

// For every channel c < C and pass row r: O_c[r][d] at dst + c·cstride + r·D/16 lines, word d
// (cstride 0 -> R·D/16); with LSUM also l_c[r] at dst + C·cstride + c·ceil(R/16) lines, word r.
// p: row r at p + r·pstride lines (pstride 0 -> n_groups), cell j_rel at word j_rel. V plane of
// LG local groups. Channel c accumulates its groups u = c (mod C) of [u0, u0 + n_groups) in
// segments of 4 groups (AT §6.4); a cell whose DAZ(p) is +-0 is skipped (it adds nothing).
// Preconditions as tsr_ref_attqk, and floor((u0 + n_groups - 1)/C) < LG.
void tsr_ref_attpv(const tsr_att_shape * sh, uint32_t C, uint32_t LG, const uint8_t * const vplane[],
                   uint32_t u0, uint32_t n_groups, uint32_t flags,
                   const float * p, uint32_t pstride, float * dst, uint32_t cstride);

// ---------------------------------------------------------------------------
// AT's vector ops on the VU (AT §5.4–§5.7, VU §5.11)
// ---------------------------------------------------------------------------
// MAXT (VU §6.8): NaN in either -> 0x7FC00000, else the larger DAZ'd value (-0 < +0).
float tsr_att_maxt(float a, float b);

// ASMAX: dst[r][k] = EXP(FM(FS(src[r][k], mvec[r]), scale)) for r < R, k < 16·n_lines; rows at
// stride lines (0 -> n_lines). dst may equal src.
void tsr_ref_asmax(float * dst, const float * src, const float * mvec, uint32_t n_lines, uint32_t stride,
                   uint32_t R, float scale);
// AMASK: m = mask[floor(r/rpq)][k] (mask rows at mstride lines); dst[r][k] = (bits(m) == 0xFF800000)
// ? -inf : FA(src[r][k], m); mdst[r] = MAXT(M_FLOOR, MAXT over dst[r]). dst may equal src.
void tsr_ref_amask(float * dst, const float * src, const float * mask, uint32_t n_lines, uint32_t stride,
                   uint32_t R, uint32_t rpq, uint32_t mstride, float * mdst);
// CTREE over x[0 .. n-1] (AT §5.6): pairwise by levels, left to right; at each level element 2k+1
// is added to element 2k and an unpaired last element passes unchanged.
float tsr_att_ctree(const float * x, uint32_t n);
// AMERGE: Ok[r][d] = CTREE_c(O_c[r][d]), lk[r] = CTREE_c(l_c[r]) over the C partials of an ATTPV
// dst with LSUM (`parts`, cstride as the producing ATTPV: 0 -> R·D/16); m_run at st (word r),
// l_run at st + ceil(R/16) lines. INIT: Orun = Ok, l_run = lk, m_run = mk; else the online merge
// of AT §5.6 with EXP and `scale`. Orun: row r at Orun + r·D/16 lines.
void tsr_ref_amerge(float * orun, const float * parts, const float * mk, float * st, uint32_t R, uint32_t D,
                    float scale, uint32_t C, uint32_t cstride, uint32_t flags);
// ANORM: out[r][d] = FM(Orun[r][d], RC(MAXT(l_run[r], 2^-126))); out and Orun rows at r·D/16 lines.
void tsr_ref_anorm(float * out, const float * orun, const float * lrun, uint32_t R, uint32_t D);

// ---------------------------------------------------------------------------
// Mask summary MSUM (AT §7.3)
// ---------------------------------------------------------------------------
// Word image: header {'MSUM' (0x4D55534D), n_q, n_kv, flags (bit0 GENERAL), f32-mask handle
// (valid iff GENERAL), 0, 0, 0}, then per query row {lo0, hi0, lo1, hi1}: at most two runs
// [lo, hi) of cells whose mask is +0 or -0, ascending and disjoint; an unused run is lo = hi = 0.
// Every other cell of the row is -inf. GENERAL is set when some row has a value that is neither
// +-0 nor -inf (finite nonzero, +inf, NaN) or more than two runs; its rows then hold the hull of
// the row's non-(-inf) cells as run 0 (informative: the kernel uses all n_kv cells).
#define TSR_MSUM_MAGIC     0x4D55534Du
#define TSR_MSUM_GENERAL   0x1u
#define TSR_MSUM_HDR_WORDS 8u
#define TSR_MSUM_ROW_WORDS 4u

typedef struct tsr_msum_row { uint32_t lo0, hi0, lo1, hi1; } tsr_msum_row;
typedef struct tsr_msum {
    uint32_t n_q, n_kv, flags;
    const tsr_msum_row * rows;    // n_q rows
} tsr_msum;

static inline size_t tsr_msum_words(uint32_t n_q) { return TSR_MSUM_HDR_WORDS + (size_t)TSR_MSUM_ROW_WORDS * n_q; }
// The backend's producer: MSUM words of an F16 KQ mask (row q at mask + q·row_stride elements,
// n_kv cells). Returns the flags word. words: tsr_msum_words(n_q) words.
uint32_t tsr_msum_build_f16(const uint16_t * mask, size_t row_stride, uint32_t n_kv, uint32_t n_q,
                            uint32_t f32_handle, uint32_t * words);
// View of a word image (rows point into words). Returns 0, or -1 for a bad magic / size.
int tsr_msum_view(const uint32_t * words, size_t n_words, tsr_msum * out);
// The f32 copy AMASK reads (GENERAL): row q at dst + q·16·ceil(n_kv/16) words; word j =
// fdiv(fp16 -> f32 of the mask, scale), words [n_kv, 16·ceil(n_kv/16)) of each row = -inf. scale is
// the reading FA op's (ggml op_params[0]; finite, normal, > 0, else abort). AMASK adds the copy to
// the raw score and ASMAX / AMERGE scale the sum (AT §5.5, AT-9), so the pre-division makes a
// finite mask value m enter as scale·(s + m/scale) = scale·s + m, the CPU's form, up to f32
// rounding; +-0 and -inf are unchanged by it (every mask of the supported models), and scale = 1
// gives the exact fp16 -> f32 copy. The backend keeps one copy per distinct scale of the FA ops
// that read the mask (W1a decision on AT §5.5).
void tsr_msum_mask_f32(const uint16_t * mask, size_t row_stride, uint32_t n_kv, uint32_t n_q, float scale,
                       float * dst);

// ---------------------------------------------------------------------------
// Composite flash attention (AT §6.6, normative; AT-15 pass rule, AT-19 chunk policy)
// ---------------------------------------------------------------------------
typedef struct tsr_att_cfg {
    uint32_t C;       // NUM_CH
    uint32_t t_max;   // T_MAX (1..8)
    uint32_t pcap;    // PCAP (CFG3[31:16]·16; U200 9,344)
    uint32_t k_max;   // K_MAX for the shape check G·E <= K_MAX (0: not checked)
} tsr_att_cfg;

// Pass rule (AT §6.6 step 2): GS = min(G, T_MAX), T_q = floor(T_MAX / GS); the last head subset
// of a group has GS' = G - hg0 heads, the last query tile T - q0 tokens.
void tsr_fa_pass_rule(const tsr_att_cfg * cfg, uint32_t G, uint32_t * GS, uint32_t * T_q);
// NCG = min(128, C·floor(PCAP/(16·E/D)), max(4C, roundup_4C(ceil(n_total/2)))) (AT §6.6 step 3).
uint32_t tsr_fa_ncg(const tsr_att_cfg * cfg, uint32_t E, uint32_t D, uint32_t n_total);

typedef struct tsr_fa_chunk {
    uint32_t u0, n_groups;   // groups [u0, u0 + n_groups)
    uint32_t c_lo, c_hi;     // ATTQK cell bounds relative to 16·u0 (GENERAL: 0, 16·n_groups)
    uint32_t flags;          // TSR_ATT_CAUSAL | TSR_ATT_SLIDE (FAST only)
    uint32_t general;        // 1: GENERAL path (ATTQK without bounds, then AMASK)
} tsr_fa_chunk;

// Chunk list of the query tile [q0, q0 + T_q) (rows of `ms`), in execution order (AT §6.6 step 3,
// §7.3; the rules are in tsr_ref_at.c, "Chunk planner"). new_runs: n_new runs [lo, hi) of cells
// written in this launch (their chunks go last). Writes min(count, cap) chunks and returns the
// count; returns -1 if the shape admits no chunk (NCG = 0), -2 for bad arguments or a malformed
// MSUM row. A tile with no visible cell gets one empty chunk (group 0, c_lo = c_hi = 0).
long tsr_fa_plan(const tsr_att_cfg * cfg, uint32_t E, uint32_t D, const tsr_msum * ms, uint32_t q0, uint32_t T_q,
                 const uint32_t (*new_runs)[2], uint32_t n_new, tsr_fa_chunk * out, size_t cap);

typedef struct tsr_fa_stats {
    uint32_t tiles, passes, chunks, general_chunks;   // summed over tiles (passes: per tile)
    uint32_t ops_attqk, ops_attpv, ops_asmax, ops_amask, ops_amerge, ops_anorm, ops_qact;
} tsr_fa_stats;

typedef struct tsr_fa_args {
    uint32_t T;                          // query tokens (n_batch)
    uint32_t D, G;                       // head dimension; heads per KV head
    float scale;                         // ggml op_params[0], unchanged
    const float * q;                     // [T][n_head][D] f32 (n_head = G·E/D)
    uint32_t E, LG;                      // plane geometry (C = cfg->C)
    const uint8_t * const * kplane;      // C channel pointers each (see Conventions)
    const uint8_t * const * vplane;
    const tsr_msum * msum;               // n_q = T, n_kv = the op's n_kv (<= 16·C·LG)
    const float * mask_f32;              // f32 mask copy (tsr_msum_mask_f32 with this scale); needed by GENERAL tiles
    const uint32_t (*new_runs)[2];       // cells written in this launch (may be NULL)
    uint32_t n_new;
    float * out;                         // [T][n_head][D] f32 (ggml's permuted FA result)
    tsr_fa_stats * stats;                // optional
} tsr_fa_args;

#define TSR_FA_OK      0
#define TSR_FA_E_SHAPE 1   // unsupported shape (D, E % D, G, rows > 128, R > 1024, G·E > K_MAX, NCG = 0)
#define TSR_FA_E_MASK  2   // a tile needs the GENERAL path and mask_f32 is NULL
#define TSR_FA_E_NOMEM 3
#define TSR_FA_E_MSUM  4   // malformed MSUM rows (a run beyond n_kv, lo > hi, overlapping runs)

// The FA kernel's tile-op sequence (AT §6.6) composed from the references above, per query tile:
// QACT A32 of each token's Q row; per chunk: per head subset ATTQK (FAST: bounds, MAXOUT -> mk;
// GENERAL: no bounds), GENERAL: AMASK -> mk; ASMAX in place; per head subset ATTPV (LSUM); AMERGE
// (INIT on the first chunk); then ANORM into out. Every op is row-independent, so this order (head
// subsets inside each chunk) gives each row the result of AT §6.6's per-pass sequence.
int tsr_ref_fa(const tsr_att_cfg * cfg, const tsr_fa_args * a);

#ifdef __cplusplus
}
#endif
