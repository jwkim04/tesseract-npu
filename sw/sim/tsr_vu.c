// tsr_vu.c — the simulator's VU unit (P1.4): decode, dispatch checks, operand extents, execution and
// cycle model of VOP 0x30, RED 0x31, ROPE 0x32, TOPK 0x33, SOP 0x40 and DQROW 0x12 on unit 3.
//
// Normative sources: docs/design/02-vector-unit.md (VU) §4 (operand model, n = 0 and tail rules, the s
// bank, aliasing), §5.1–§5.9 (encodings, dispatch checks, extents), §5.10 (CAPS, PERF) and §7 (the
// datapath the cycle model follows); docs/design/01-execution-model.md (EX) §6 (PERF), §7.1 (x16),
// §8.1 (error codes and their order); docs/design/06-software-verification.md (SW) §4.5. Every numeric
// result comes from tsr_ref_vu.c (which builds on tsr_ref_math.c and tsr_ref.c): this file decodes
// with the generated field constants (tsr_tables.h), checks, and points the references at the
// simulator's SPM and s bank. AT's vector ops 0x34–0x37 also run on unit 3 but are AT's ops
// (tsr_at.c). VMASK 0x38 (M9) has no handler, so the issue path completes it ILLEGAL_OP (CAPS
// bit7 = 0), as it does every reserved VU opcode (0x13, 0x39–0x3F, 0x41–0x47, EX §7.2).
//
// Checks (VU §5.1, §5.9; EX §8.1 order ILLEGAL_OP -> BAD_FMT -> K_RANGE -> T_RANGE -> OPERAND ->
// SPM_RANGE; BAD_DEP is the queue model's; no VU op has a DRAM operand):
//   ILLEGAL_OP  every reserved bit of the encoding tables (TSR_*_RSV_W*: x16 bits, flags, w[31:24]
//               of line addresses and m), an undefined enumeration value (VOP function >= 25 or
//               cmode >= 3, RED kind >= 3 or post 3, ROPE mode >= 2, SOP sub 15), and every field
//               the op or function does not use (VU §5.1): VOP a / a_stride without a read of a,
//               b / b_stride likewise, c unless cmode 2, si unless cmode 1, a function reading c
//               with cmode 0 or one not reading c with cmode != 0; RED MAX with a post-op, sd
//               without TO_S, d with TO_S, sk / se without post RMS; ROPE ATAB with FF, FS or
//               sts != 0, sfs without FS, a table line without FF or ATAB; SOP fields its sub does
//               not use (tsr_sop_uses).
//   BAD_FMT / K_RANGE  DQROW only: fmt >= 2; K = 0, K > K_MAX or K not a multiple of the block.
//   OPERAND     VU §5.9's conditions in its order, ERR_INFO = the value shown there (the raw field).
//   SPM_RANGE   operands in field order (w1, w2, w3, w4), extents of VU §5.9, ERR_INFO = the start
//               line (SOP LDS / STS: the word address, against SPM_BYTES/4).
// The issue path (tsr_sim_accept_op) already refuses an opcode whose CAPS bits are missing (TOPK
// bit3, DQROW bit6) with ILLEGAL_OP on SYNC.
//
// Extents (tsr_sim_int.h §1): the range-check extents in field order, then the s registers the op
// really accesses (a VOP with m = 0 reads no s[si]; a TO_S RED writes s[sd] even for m = 0), one
// entry per (index, direction). The aliasing rule of VU §4.5 (a source may equal the destination
// only exactly) is the race detector's (P1.9, VU §10 item 3), not a dispatch check: EX has no code
// for it.
//
// Execution: at completion (SW §4.1), sources read through the references after every earlier op's
// effect. s operands are read from the bank at exec time, which is the op's start for every
// observer because the queue model serves one VU op at a time (VU §4.4).
//
// Cycle model (unit busy time, deterministic, from the words and the configuration only; the
// figures are VU §7.9's *estimates*, not part of the bit-exact contract). Lane rates from
// TSR_VU_FRONT_LANES / TSR_VU_BACK_LANES (CFG3): a full-rate function streams 16/front lines per
// cycle-line (1 cycle per line on the U200), a half-rate (back-lane) one 16/back (2).
//   VOP    rows·L(m)·rate + rows if a and b both stream and L(m) is odd (pair reads end a row with
//          a single-line read, VU §7.7) + ceil(rows/16) for a c vector + latency: 64 for the long
//          class (back lanes, or EXPK: EXP, EXPSUB, EXPSUBC), else 20 (VU §7.4)
//   RED    rows·max(L(m), min(ceil(m/8), 16)) + 45, + 30 with post RMS (VU §7.5)
//   ROPE   T·(4·n_dims/2 + n_heads·head_dim/16) + 40; 8 per angle with FF (two iterative dividers,
//          VU §7.6); angle reuse (§7.6, optional and invisible) is not modelled
//   TOPK   rows·(L(m) + k) + 20 (the typical running-threshold case of VU §7.9; the data-dependent
//          worst case m per row is not modelled because cycles() sees no data)
//   DQROW  rows·L(K) + 30
//   SOP    30, DIV 40
//   An op that reads and writes nothing (n = 0 rule) takes 1 cycle.
// PERF (EX §6, VU §5.10): VU_CYCLES is the queue model's (busy time). VU_LINES (X + 15, event) is
// counted here: destination lines written (VOP rows·L(m), RED ceil(rows/16) for a vector, ROPE
// T·H, TOPK rows·ceil(k/8), DQROW rows·L(K), SOP STS 1 line, 0 for writes of s registers only).
// VU_SPM_WAIT and VU_HAZARD stay 0: the model has no SPM arbitration and no accept-ahead (the queue
// model runs one VU op at a time).

#include "tsr_sim.h"
#include "tsr_sim_int.h"
#include "tsr_ref.h"
#include "tsr_ref_vu.h"

#include <string.h>

#define VU_LAT_SHORT   20u   // VU §7.4: FM_A .. FM_C, cmp, incl. SPM read and write (*estimate*)
#define VU_LAT_LONG    64u   // VU §7.4: the full chain (EXPK, back lanes)
#define VU_RED_LAT     45u   // VU §7.5: result latency after the last line
#define VU_RED_RMS_LAT 30u   // VU §7.5: + RMS post (SQ, RC)
#define VU_ROPE_LAT    40u   // VU §7.9
#define VU_ANGLE_CYC   4u    // VU §7.6: cycles per angle (recurrence form, A table)
#define VU_ANGLE_FF    8u    // VU §7.6: with freq factors
#define VU_TOPK_LAT    20u   // VU §7.9 (sorted insertion per row; short class)
#define VU_DQROW_LAT   30u   // VU §7.9
#define VU_SOP_CYC     30u   // VU §7.9
#define VU_SOP_DIV_CYC 40u

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static inline uint32_t lines_of(uint32_t n) { return tsr_ref_vu_lines(n); }    // L(n) = ceil(n/16)

// VU §5.9: E(n, s, len) = 0 if n = 0 or len = 0, else (n - 1)·s + len.
static inline uint64_t ext_rows(uint32_t n, uint64_t s, uint64_t len) {
    return (n == 0u || len == 0u) ? 0u : (uint64_t)(n - 1u) * s + len;
}

static int rsv_bad(const uint32_t w[8], const uint32_t rsv[8]) {
    for (unsigned i = 0; i < TSR_INSN_WORDS; ++i) {
        if (w[i] & rsv[i]) return 1;
    }
    return 0;
}

// Spec range-check rule (docs/spec.md:1053): [a, a + e) must end at or below the SPM's line count.
static int spm_bad(const tsr_sim * sim, uint32_t a, uint64_t e, uint32_t * info) {
    if ((uint64_t)a + e <= tsr_sim_spm_lines(sim)) return 0;
    *info = a;
    return 1;
}

static inline void ext_spm(tsr_extents * x, uint8_t write, uint32_t a, uint64_t e) {
    tsr_ext_add(x, TSR_EXT_SPM, write, 0, a, (uint64_t)a + e);
}
// One SREG entry per (index, direction).
static void ext_sreg(tsr_extents * x, uint8_t write, uint32_t i) {
    if (x == NULL) return;
    for (uint32_t k = 0; k < x->n; ++k) {
        if (x->e[k].kind == TSR_EXT_SREG && x->e[k].write == write && x->e[k].lo == i) return;
    }
    tsr_ext_add(x, TSR_EXT_SREG, write, 0, i, (uint64_t)i + 1u);
}

static inline uint32_t * spm_words(tsr_sim * sim, uint32_t line) {
    return (uint32_t *)(void *)(tsr_sim_spm(sim) + (size_t)line * TSR_LINE_BYTES);
}
static inline uint32_t sreg_get(tsr_sim * sim, uint32_t i) {
    uint32_t u;
    memcpy(&u, &tsr_sim_sreg(sim)[i], sizeof u);
    return u;
}
static inline void sreg_set(tsr_sim * sim, uint32_t i, uint32_t v) { memcpy(&tsr_sim_sreg(sim)[i], &v, sizeof v); }

static void vu_lines(tsr_sim * sim, uint64_t n) {
    if (n != 0u) tsr_sim_perf_add(sim, TSR_PERF_VU_LINES(tsr_sim_num_ch(sim)), n);
}

// Cycles per line of a full-rate / half-rate function (VU §7.2-§7.4, CFG3 lanes).
static inline uint64_t rate_full(void) { return (TSR_VL + TSR_VU_FRONT_LANES - 1u) / TSR_VU_FRONT_LANES; }
static inline uint64_t rate_half(void) { return (TSR_VL + TSR_VU_BACK_LANES - 1u) / TSR_VU_BACK_LANES; }

// ---------------------------------------------------------------------------
// VOP 0x30 (VU §5.2)
// ---------------------------------------------------------------------------
static const uint32_t k_vop_rsv[8] = { TSR_VOP_RSV_W0, TSR_VOP_RSV_W1, TSR_VOP_RSV_W2, TSR_VOP_RSV_W3,
                                       TSR_VOP_RSV_W4, TSR_VOP_RSV_W5, TSR_VOP_RSV_W6, TSR_VOP_RSV_W7 };

typedef struct vop_dec {
    uint32_t cmode, func, si, d, a, b, c, m, rows, ds, as, bs, reads;
} vop_dec;

static void vop_decode(const uint32_t w[8], vop_dec * v) {
    v->cmode = TSR_INSN_GET(w, TSR_VOP_CMODE);
    v->func  = TSR_INSN_GET(w, TSR_VOP_FUNC);
    v->si    = TSR_INSN_GET(w, TSR_VOP_SI);
    v->d     = TSR_INSN_GET(w, TSR_VOP_D);
    v->a     = TSR_INSN_GET(w, TSR_VOP_A);
    v->b     = TSR_INSN_GET(w, TSR_VOP_B);
    v->c     = TSR_INSN_GET(w, TSR_VOP_C);
    v->m     = TSR_INSN_GET(w, TSR_VOP_M);
    v->rows  = TSR_INSN_GET(w, TSR_VOP_ROWS);
    v->ds    = TSR_INSN_GET(w, TSR_VOP_D_STRIDE);
    v->as    = TSR_INSN_GET(w, TSR_VOP_A_STRIDE);
    v->bs    = TSR_INSN_GET(w, TSR_VOP_B_STRIDE);
    v->reads = tsr_vop_func_reads(v->func);              // 0: undefined function
}

static uint32_t chk_vop(tsr_sim * sim, const uint32_t w[8], tsr_extents * ext, uint32_t * info) {
    if (rsv_bad(w, k_vop_rsv)) return TSR_ERR_ILLEGAL_OP;
    vop_dec v;
    vop_decode(w, &v);
    if (v.reads == 0u || v.cmode > TSR_VOP_CMODE_VEC) return TSR_ERR_ILLEGAL_OP;
    const int ra = (v.reads & TSR_VOP_READS_A) != 0, rb = (v.reads & TSR_VOP_READS_B) != 0;
    if (((v.reads & TSR_VOP_READS_C) != 0) != (v.cmode != TSR_VOP_CMODE_NONE)) return TSR_ERR_ILLEGAL_OP;
    if (v.cmode != TSR_VOP_CMODE_SREG && v.si != 0u) return TSR_ERR_ILLEGAL_OP;
    if (v.cmode != TSR_VOP_CMODE_VEC && v.c != 0u) return TSR_ERR_ILLEGAL_OP;
    if (!ra && (v.a | v.as) != 0u) return TSR_ERR_ILLEGAL_OP;
    if (!rb && (v.b | v.bs) != 0u) return TSR_ERR_ILLEGAL_OP;
    const uint32_t L = lines_of(v.m);
    if (v.rows > 1u && v.ds < L) { *info = v.ds; return TSR_ERR_OPERAND; }
    const uint64_t ed = ext_rows(v.rows, v.ds, L), ea = ext_rows(v.rows, v.as, L), eb = ext_rows(v.rows, v.bs, L);
    const uint64_t ec = (v.cmode == TSR_VOP_CMODE_VEC) ? lines_of(v.rows) : 0u;
    if (spm_bad(sim, v.d, ed, info)) return TSR_ERR_SPM_RANGE;
    if (ra && spm_bad(sim, v.a, ea, info)) return TSR_ERR_SPM_RANGE;
    if (rb && spm_bad(sim, v.b, eb, info)) return TSR_ERR_SPM_RANGE;
    if (v.cmode == TSR_VOP_CMODE_VEC && spm_bad(sim, v.c, ec, info)) return TSR_ERR_SPM_RANGE;
    ext_spm(ext, TSR_EXT_WRITE, v.d, ed);
    if (ra) ext_spm(ext, TSR_EXT_READ, v.a, ea);
    if (rb) ext_spm(ext, TSR_EXT_READ, v.b, eb);
    if (v.cmode == TSR_VOP_CMODE_VEC) ext_spm(ext, TSR_EXT_READ, v.c, ec);
    if (v.cmode == TSR_VOP_CMODE_SREG && ed != 0u) ext_sreg(ext, TSR_EXT_READ, v.si);
    return TSR_ERR_NONE;
}

static void ex_vop(tsr_sim * sim, const uint32_t w[8]) {
    vop_dec v;
    vop_decode(w, &v);
    if (v.rows == 0u || v.m == 0u) return;                   // n = 0 rule (VU §4.2)
    uint32_t cval = 0;
    const uint32_t * c = NULL;
    if (v.cmode == TSR_VOP_CMODE_SREG) {
        cval = sreg_get(sim, v.si);
        c = &cval;
    } else if (v.cmode == TSR_VOP_CMODE_VEC) {
        c = spm_words(sim, v.c);
    }
    tsr_ref_vop(v.func, v.cmode, v.rows, v.m, spm_words(sim, v.d), v.ds,
                (v.reads & TSR_VOP_READS_A) ? spm_words(sim, v.a) : NULL, v.as,
                (v.reads & TSR_VOP_READS_B) ? spm_words(sim, v.b) : NULL, v.bs, c);
    vu_lines(sim, (uint64_t)v.rows * lines_of(v.m));
}

static int vop_long(uint32_t func) {
    return tsr_vop_func_half(func) || func == TSR_VOP_FUNC_EXP || func == TSR_VOP_FUNC_EXPSUB ||
           func == TSR_VOP_FUNC_EXPSUBC;
}

static uint64_t cyc_vop(const tsr_sim * sim, const uint32_t w[8]) {
    (void)sim;
    vop_dec v;
    vop_decode(w, &v);
    if (v.rows == 0u || v.m == 0u) return 1u;
    const uint64_t L = lines_of(v.m);
    uint64_t cyc = (uint64_t)v.rows * L * (tsr_vop_func_half(v.func) ? rate_half() : rate_full());
    if ((v.reads & TSR_VOP_READS_A) && (v.reads & TSR_VOP_READS_B) && (L & 1u)) cyc += v.rows;
    if (v.cmode == TSR_VOP_CMODE_VEC) cyc += lines_of(v.rows);
    return cyc + (vop_long(v.func) ? VU_LAT_LONG : VU_LAT_SHORT);
}

// ---------------------------------------------------------------------------
// RED 0x31 (VU §5.3, §6.2)
// ---------------------------------------------------------------------------
static const uint32_t k_red_rsv[8] = { TSR_RED_RSV_W0, TSR_RED_RSV_W1, TSR_RED_RSV_W2, TSR_RED_RSV_W3,
                                       TSR_RED_RSV_W4, TSR_RED_RSV_W5, TSR_RED_RSV_W6, TSR_RED_RSV_W7 };

typedef struct red_dec {
    uint32_t kind, to_s, sd, post, sk, se, d, a, m, rows, as;
} red_dec;

static void red_decode(const uint32_t w[8], red_dec * r) {
    r->kind = TSR_INSN_GET(w, TSR_RED_KIND);
    r->to_s = TSR_INSN_GET(w, TSR_RED_TO_S);
    r->sd   = TSR_INSN_GET(w, TSR_RED_SD);
    r->post = TSR_INSN_GET(w, TSR_RED_POST);
    r->sk   = TSR_INSN_GET(w, TSR_RED_SK);
    r->se   = TSR_INSN_GET(w, TSR_RED_SE);
    r->d    = TSR_INSN_GET(w, TSR_RED_D);
    r->a    = TSR_INSN_GET(w, TSR_RED_A);
    r->m    = TSR_INSN_GET(w, TSR_RED_M);
    r->rows = TSR_INSN_GET(w, TSR_RED_ROWS);
    r->as   = TSR_INSN_GET(w, TSR_RED_A_STRIDE);
}

static uint32_t chk_red(tsr_sim * sim, const uint32_t w[8], tsr_extents * ext, uint32_t * info) {
    if (rsv_bad(w, k_red_rsv)) return TSR_ERR_ILLEGAL_OP;
    red_dec r;
    red_decode(w, &r);
    if (r.kind > TSR_RED_KIND_MAX || r.post > TSR_RED_POST_RMS) return TSR_ERR_ILLEGAL_OP;
    if (r.kind == TSR_RED_KIND_MAX && r.post != TSR_RED_POST_NONE) return TSR_ERR_ILLEGAL_OP;
    if (r.to_s ? r.d != 0u : r.sd != 0u) return TSR_ERR_ILLEGAL_OP;
    if (r.post != TSR_RED_POST_RMS && (r.sk | r.se) != 0u) return TSR_ERR_ILLEGAL_OP;
    if (r.to_s && r.rows != 1u) { *info = r.rows; return TSR_ERR_OPERAND; }
    const uint64_t ed = r.to_s ? 0u : lines_of(r.rows), ea = ext_rows(r.rows, r.as, lines_of(r.m));
    if (!r.to_s && spm_bad(sim, r.d, ed, info)) return TSR_ERR_SPM_RANGE;
    if (spm_bad(sim, r.a, ea, info)) return TSR_ERR_SPM_RANGE;
    ext_spm(ext, TSR_EXT_WRITE, r.d, ed);
    ext_spm(ext, TSR_EXT_READ, r.a, ea);
    const int runs = r.to_s || (r.rows != 0u && r.m != 0u);   // TO_S writes s[sd] even for m = 0
    if (r.to_s) ext_sreg(ext, TSR_EXT_WRITE, r.sd);
    if (runs && r.post == TSR_RED_POST_RMS) {
        ext_sreg(ext, TSR_EXT_READ, r.sk);
        ext_sreg(ext, TSR_EXT_READ, r.se);
    }
    return TSR_ERR_NONE;
}

static void ex_red(tsr_sim * sim, const uint32_t w[8]) {
    red_dec r;
    red_decode(w, &r);
    const uint32_t sk = sreg_get(sim, r.sk), se = sreg_get(sim, r.se);
    const uint32_t * a = spm_words(sim, r.a);                // not read when m = 0
    if (r.to_s) {
        sreg_set(sim, r.sd, tsr_ref_red_row(r.kind, r.post, a, r.m, sk, se));
        return;
    }
    if (r.rows == 0u || r.m == 0u) return;                   // n = 0 rule (VU §4.2)
    tsr_ref_red(r.kind, r.post, r.rows, r.m, spm_words(sim, r.d), a, r.as, sk, se);
    vu_lines(sim, lines_of(r.rows));
}

static uint64_t cyc_red(const tsr_sim * sim, const uint32_t w[8]) {
    (void)sim;
    red_dec r;
    red_decode(w, &r);
    if (!r.to_s && (r.rows == 0u || r.m == 0u)) return 1u;
    const uint32_t tree = (r.m + 7u) / 8u < 16u ? (r.m + 7u) / 8u : 16u;    // VU §7.5
    const uint32_t per_row = lines_of(r.m) > tree ? lines_of(r.m) : tree;
    return (uint64_t)r.rows * per_row + VU_RED_LAT + (r.post == TSR_RED_POST_RMS ? VU_RED_RMS_LAT : 0u);
}

// ---------------------------------------------------------------------------
// ROPE 0x32 (VU §5.4, §6.7)
// ---------------------------------------------------------------------------
static const uint32_t k_rope_rsv[8] = { TSR_ROPE_RSV_W0, TSR_ROPE_RSV_W1, TSR_ROPE_RSV_W2, TSR_ROPE_RSV_W3,
                                        TSR_ROPE_RSV_W4, TSR_ROPE_RSV_W5, TSR_ROPE_RSV_W6, TSR_ROPE_RSV_W7 };

typedef struct rope_dec {
    uint32_t mode, ff, fs, atab, sm, sts, sfs, d, src, pos, table, hd, nd, nh, T, ss, ds;
    uint64_t H;     // lines per token: n_heads·head_dim/16 (VU §5.4)
    uint32_t Tp;    // T' = T if n_heads > 0 else 0 (VU §5.9)
} rope_dec;

static void rope_decode(const uint32_t w[8], rope_dec * r) {
    r->mode  = TSR_INSN_GET(w, TSR_ROPE_MODE);
    r->ff    = TSR_INSN_GET(w, TSR_ROPE_FF);
    r->fs    = TSR_INSN_GET(w, TSR_ROPE_FS);
    r->atab  = TSR_INSN_GET(w, TSR_ROPE_ATAB);
    r->sm    = TSR_INSN_GET(w, TSR_ROPE_SM);
    r->sts   = TSR_INSN_GET(w, TSR_ROPE_STS);
    r->sfs   = TSR_INSN_GET(w, TSR_ROPE_SFS);
    r->d     = TSR_INSN_GET(w, TSR_ROPE_D);
    r->src   = TSR_INSN_GET(w, TSR_ROPE_SRC);
    r->pos   = TSR_INSN_GET(w, TSR_ROPE_POS);
    r->table = TSR_INSN_GET(w, TSR_ROPE_TABLE);
    r->hd    = TSR_INSN_GET(w, TSR_ROPE_HEAD_DIM);
    r->nd    = TSR_INSN_GET(w, TSR_ROPE_N_DIMS);
    r->nh    = TSR_INSN_GET(w, TSR_ROPE_N_HEADS);
    r->T     = TSR_INSN_GET(w, TSR_ROPE_T);
    r->ss    = TSR_INSN_GET(w, TSR_ROPE_SRC_STRIDE);
    r->ds    = TSR_INSN_GET(w, TSR_ROPE_D_STRIDE);
    r->H     = (uint64_t)r->nh * r->hd / 16u;
    r->Tp    = r->nh > 0u ? r->T : 0u;
}

static uint32_t chk_rope(tsr_sim * sim, const uint32_t w[8], tsr_extents * ext, uint32_t * info) {
    if (rsv_bad(w, k_rope_rsv)) return TSR_ERR_ILLEGAL_OP;
    rope_dec r;
    rope_decode(w, &r);
    if (r.mode > TSR_ROPE_MODE_NEOX) return TSR_ERR_ILLEGAL_OP;
    if (r.atab && (r.ff | r.fs | r.sts) != 0u) return TSR_ERR_ILLEGAL_OP;    // VU §5.9: before OPERAND
    if (!r.fs && r.sfs != 0u) return TSR_ERR_ILLEGAL_OP;
    const int tab = r.ff || r.atab;
    if (!tab && r.table != 0u) return TSR_ERR_ILLEGAL_OP;
    if (r.hd == 0u || (r.hd & 15u) != 0u || r.hd > TSR_ROPE_MAX_HEAD_DIM) { *info = r.hd; return TSR_ERR_OPERAND; }
    const uint32_t q = (r.mode == TSR_ROPE_MODE_NEOX) ? 32u : 16u;
    if (r.nd == 0u || r.nd > r.hd || r.nd % q != 0u) { *info = r.nd; return TSR_ERR_OPERAND; }
    const uint64_t dsp = r.ds ? r.ds : r.H, ssp = r.ss ? r.ss : r.H;
    if (r.T > 1u && r.nh > 0u && dsp < r.H) { *info = r.ds; return TSR_ERR_OPERAND; }
    const uint64_t ed = ext_rows(r.Tp, dsp, r.H), es = ext_rows(r.Tp, ssp, r.H), ep = lines_of(r.Tp);
    const uint64_t et = (tab && r.Tp > 0u) ? (r.nd + 31u) / 32u : 0u;
    if (spm_bad(sim, r.d, ed, info)) return TSR_ERR_SPM_RANGE;
    if (spm_bad(sim, r.src, es, info)) return TSR_ERR_SPM_RANGE;
    if (spm_bad(sim, r.pos, ep, info)) return TSR_ERR_SPM_RANGE;
    if (tab && spm_bad(sim, r.table, et, info)) return TSR_ERR_SPM_RANGE;
    ext_spm(ext, TSR_EXT_WRITE, r.d, ed);
    ext_spm(ext, TSR_EXT_READ, r.src, es);
    ext_spm(ext, TSR_EXT_READ, r.pos, ep);
    if (tab) ext_spm(ext, TSR_EXT_READ, r.table, et);
    if (r.Tp > 0u) {
        ext_sreg(ext, TSR_EXT_READ, r.sm);
        if (!r.atab) ext_sreg(ext, TSR_EXT_READ, r.sts);
        if (r.fs) ext_sreg(ext, TSR_EXT_READ, r.sfs);
    }
    return TSR_ERR_NONE;
}

static void ex_rope(tsr_sim * sim, const uint32_t w[8]) {
    rope_dec r;
    rope_decode(w, &r);
    if (r.Tp == 0u) return;                                  // VU §5.9: T' = 0, nothing read or written
    tsr_ref_rope_cfg cfg;
    cfg.mode = r.mode;
    cfg.ff = r.ff;
    cfg.fs = r.fs;
    cfg.atab = r.atab;
    cfg.m = sreg_get(sim, r.sm);
    cfg.ts = r.atab ? 0u : sreg_get(sim, r.sts);
    cfg.freq_scale = r.fs ? sreg_get(sim, r.sfs) : 0u;
    cfg.head_dim = r.hd;
    cfg.n_dims = r.nd;
    cfg.n_heads = r.nh;
    cfg.T = r.T;
    tsr_ref_rope(&cfg, spm_words(sim, r.d), r.ds, spm_words(sim, r.src), r.ss, spm_words(sim, r.pos),
                 (r.ff || r.atab) ? spm_words(sim, r.table) : NULL);
    vu_lines(sim, (uint64_t)r.Tp * r.H);
}

static uint64_t cyc_rope(const tsr_sim * sim, const uint32_t w[8]) {
    (void)sim;
    rope_dec r;
    rope_decode(w, &r);
    if (r.Tp == 0u) return 1u;
    const uint64_t angles = (uint64_t)(r.ff ? VU_ANGLE_FF : VU_ANGLE_CYC) * (r.nd / 2u);
    return (uint64_t)r.T * (angles + r.H * rate_full()) + VU_ROPE_LAT;
}

// ---------------------------------------------------------------------------
// TOPK 0x33 (VU §5.5, §6.8; simulator from M6, RTL M8)
// ---------------------------------------------------------------------------
static const uint32_t k_topk_rsv[8] = { TSR_TOPK_RSV_W0, TSR_TOPK_RSV_W1, TSR_TOPK_RSV_W2, TSR_TOPK_RSV_W3,
                                        TSR_TOPK_RSV_W4, TSR_TOPK_RSV_W5, TSR_TOPK_RSV_W6, TSR_TOPK_RSV_W7 };

typedef struct topk_dec {
    uint32_t k, d, a, m, rows, ds, as, kl;   // kl = ceil(k/8): output lines per row
} topk_dec;

static void topk_decode(const uint32_t w[8], topk_dec * t) {
    t->k    = TSR_INSN_GET(w, TSR_TOPK_K);
    t->d    = TSR_INSN_GET(w, TSR_TOPK_D);
    t->a    = TSR_INSN_GET(w, TSR_TOPK_A);
    t->m    = TSR_INSN_GET(w, TSR_TOPK_M);
    t->rows = TSR_INSN_GET(w, TSR_TOPK_ROWS);
    t->ds   = TSR_INSN_GET(w, TSR_TOPK_D_STRIDE);
    t->as   = TSR_INSN_GET(w, TSR_TOPK_A_STRIDE);
    t->kl   = (t->k + 7u) / 8u;
}

static uint32_t chk_topk(tsr_sim * sim, const uint32_t w[8], tsr_extents * ext, uint32_t * info) {
    if (rsv_bad(w, k_topk_rsv)) return TSR_ERR_ILLEGAL_OP;
    topk_dec t;
    topk_decode(w, &t);
    if (t.k == 0u || t.k > TSR_TOPK_MAX_K) { *info = t.k; return TSR_ERR_OPERAND; }
    const uint32_t dsp = t.ds ? t.ds : t.kl;
    if (t.rows > 1u && dsp < t.kl) { *info = t.ds; return TSR_ERR_OPERAND; }
    const uint64_t ed = ext_rows(t.rows, dsp, t.kl), ea = ext_rows(t.rows, t.as, lines_of(t.m));
    if (spm_bad(sim, t.d, ed, info)) return TSR_ERR_SPM_RANGE;
    if (spm_bad(sim, t.a, ea, info)) return TSR_ERR_SPM_RANGE;
    ext_spm(ext, TSR_EXT_WRITE, t.d, ed);
    ext_spm(ext, TSR_EXT_READ, t.a, ea);
    return TSR_ERR_NONE;
}

static void ex_topk(tsr_sim * sim, const uint32_t w[8]) {
    topk_dec t;
    topk_decode(w, &t);
    if (t.rows == 0u || t.m == 0u) return;                   // n = 0 rule (VU §4.2)
    tsr_ref_topk(t.k, t.rows, t.m, spm_words(sim, t.d), t.ds, spm_words(sim, t.a), t.as);
    vu_lines(sim, (uint64_t)t.rows * t.kl);
}

static uint64_t cyc_topk(const tsr_sim * sim, const uint32_t w[8]) {
    (void)sim;
    topk_dec t;
    topk_decode(w, &t);
    if (t.rows == 0u || t.m == 0u) return 1u;
    return (uint64_t)t.rows * ((uint64_t)lines_of(t.m) * rate_full() + t.k) + VU_TOPK_LAT;
}

// ---------------------------------------------------------------------------
// SOP 0x40 (VU §5.8)
// ---------------------------------------------------------------------------
static const uint32_t k_sop_rsv[8] = { TSR_SOP_RSV_W0, TSR_SOP_RSV_W1, TSR_SOP_RSV_W2, TSR_SOP_RSV_W3,
                                       TSR_SOP_RSV_W4, TSR_SOP_RSV_W5, TSR_SOP_RSV_W6, TSR_SOP_RSV_W7 };

typedef struct sop_dec {
    uint32_t sub, sd, sa, sb, imm, addr, uses;
} sop_dec;

static void sop_decode(const uint32_t w[8], sop_dec * s) {
    s->sub  = TSR_INSN_GET(w, TSR_SOP_SUB);
    s->sd   = TSR_INSN_GET(w, TSR_SOP_SD);
    s->sa   = TSR_INSN_GET(w, TSR_SOP_SA);
    s->sb   = TSR_INSN_GET(w, TSR_SOP_SB);
    s->imm  = TSR_INSN_GET(w, TSR_SOP_IMM);
    s->addr = TSR_INSN_GET(w, TSR_SOP_ADDR);
    s->uses = tsr_sop_uses(s->sub);                          // 0: undefined sub
}

static uint32_t chk_sop(tsr_sim * sim, const uint32_t w[8], tsr_extents * ext, uint32_t * info) {
    if (rsv_bad(w, k_sop_rsv)) return TSR_ERR_ILLEGAL_OP;
    sop_dec s;
    sop_decode(w, &s);
    if (s.uses == 0u) return TSR_ERR_ILLEGAL_OP;
    if ((!(s.uses & TSR_SOP_USES_SD) && s.sd) || (!(s.uses & TSR_SOP_USES_SA) && s.sa) ||
        (!(s.uses & TSR_SOP_USES_SB) && s.sb) || (!(s.uses & TSR_SOP_USES_IMM) && s.imm) ||
        (!(s.uses & TSR_SOP_USES_ADDR) && s.addr)) {
        return TSR_ERR_ILLEGAL_OP;
    }
    if (s.uses & TSR_SOP_USES_ADDR) {                        // LDS / STS: one word (VU §5.9)
        if ((uint64_t)s.addr + 1u > tsr_sim_config(sim)->spm_bytes / 4u) { *info = s.addr; return TSR_ERR_SPM_RANGE; }
        ext_spm(ext, s.sub == TSR_SOP_SUB_STS ? TSR_EXT_WRITE : TSR_EXT_READ, s.addr / 16u, 1u);
    }
    if (s.uses & TSR_SOP_USES_SD) ext_sreg(ext, TSR_EXT_WRITE, s.sd);
    if (s.uses & TSR_SOP_USES_SA) ext_sreg(ext, TSR_EXT_READ, s.sa);
    if (s.uses & TSR_SOP_USES_SB) ext_sreg(ext, TSR_EXT_READ, s.sb);
    return TSR_ERR_NONE;
}

static void ex_sop(tsr_sim * sim, const uint32_t w[8]) {
    sop_dec s;
    sop_decode(w, &s);
    uint32_t * word = (s.uses & TSR_SOP_USES_ADDR) ? spm_words(sim, 0) + s.addr : NULL;
    const uint32_t mem = (s.sub == TSR_SOP_SUB_LDS) ? *word : 0u;
    const uint32_t v = tsr_ref_sop(s.sub, sreg_get(sim, s.sa), sreg_get(sim, s.sb), s.imm, mem);
    if (s.sub == TSR_SOP_SUB_STS) {
        *word = v;                                           // exactly 4 bytes, no tail rule
        vu_lines(sim, 1u);
    } else {
        sreg_set(sim, s.sd, v);
    }
}

static uint64_t cyc_sop(const tsr_sim * sim, const uint32_t w[8]) {
    (void)sim;
    return TSR_INSN_GET(w, TSR_SOP_SUB) == TSR_SOP_SUB_DIV ? VU_SOP_DIV_CYC : VU_SOP_CYC;
}

// ---------------------------------------------------------------------------
// DQROW 0x12 (VU §5.6, §6.8; simulator from M6, RTL M8)
// ---------------------------------------------------------------------------
static const uint32_t k_dqrow_rsv[8] = { TSR_DQROW_RSV_W0, TSR_DQROW_RSV_W1, TSR_DQROW_RSV_W2, TSR_DQROW_RSV_W3,
                                         TSR_DQROW_RSV_W4, TSR_DQROW_RSV_W5, TSR_DQROW_RSV_W6, TSR_DQROW_RSV_W7 };

typedef struct dqrow_dec {
    uint32_t fmt, d, src, K, rows, ss, ds;
} dqrow_dec;

static void dqrow_decode(const uint32_t w[8], dqrow_dec * q) {
    q->fmt  = TSR_INSN_GET(w, TSR_DQROW_FMT);
    q->d    = TSR_INSN_GET(w, TSR_DQROW_D);
    q->src  = TSR_INSN_GET(w, TSR_DQROW_SRC);
    q->K    = TSR_INSN_GET(w, TSR_DQROW_K);
    q->rows = TSR_INSN_GET(w, TSR_DQROW_ROWS);
    q->ss   = TSR_INSN_GET(w, TSR_DQROW_SRC_STRIDE);
    q->ds   = TSR_INSN_GET(w, TSR_DQROW_D_STRIDE);
}

static uint32_t chk_dqrow(tsr_sim * sim, const uint32_t w[8], tsr_extents * ext, uint32_t * info) {
    if (rsv_bad(w, k_dqrow_rsv)) return TSR_ERR_ILLEGAL_OP;
    dqrow_dec q;
    dqrow_decode(w, &q);
    if (!tsr_fmt_valid(q.fmt)) return TSR_ERR_BAD_FMT;
    if (!tsr_fmt_k_valid(q.fmt, q.K, tsr_sim_config(sim)->k_max)) return TSR_ERR_K_RANGE;
    const uint32_t L = lines_of(q.K), rb = tsr_row_beats(q.fmt, q.K);
    const uint32_t dsp = q.ds ? q.ds : L, ssp = q.ss ? q.ss : rb;
    if (q.rows > 1u && dsp < L) { *info = q.ds; return TSR_ERR_OPERAND; }
    const uint64_t ed = ext_rows(q.rows, dsp, L), es = ext_rows(q.rows, ssp, rb);
    if (spm_bad(sim, q.d, ed, info)) return TSR_ERR_SPM_RANGE;
    if (spm_bad(sim, q.src, es, info)) return TSR_ERR_SPM_RANGE;
    ext_spm(ext, TSR_EXT_WRITE, q.d, ed);
    ext_spm(ext, TSR_EXT_READ, q.src, es);
    return TSR_ERR_NONE;
}

static void ex_dqrow(tsr_sim * sim, const uint32_t w[8]) {
    dqrow_dec q;
    dqrow_decode(w, &q);
    if (q.rows == 0u) return;
    tsr_ref_dqrow(q.fmt, q.K, q.rows, spm_words(sim, q.d), q.ds,
                  tsr_sim_spm(sim) + (size_t)q.src * TSR_LINE_BYTES, q.ss);
    vu_lines(sim, (uint64_t)q.rows * lines_of(q.K));
}

static uint64_t cyc_dqrow(const tsr_sim * sim, const uint32_t w[8]) {
    (void)sim;
    dqrow_dec q;
    dqrow_decode(w, &q);
    if (q.rows == 0u) return 1u;
    return (uint64_t)q.rows * lines_of(q.K) * rate_full() + VU_DQROW_LAT;
}

// ---------------------------------------------------------------------------
// Registration (tsr_sim_int.h §2)
// ---------------------------------------------------------------------------
void tsr_vu_register(void) {
    const tsr_op_handler vop   = { TSR_UNIT_VU, chk_vop, ex_vop, cyc_vop };
    const tsr_op_handler red   = { TSR_UNIT_VU, chk_red, ex_red, cyc_red };
    const tsr_op_handler rope  = { TSR_UNIT_VU, chk_rope, ex_rope, cyc_rope };
    const tsr_op_handler topk  = { TSR_UNIT_VU, chk_topk, ex_topk, cyc_topk };
    const tsr_op_handler sop   = { TSR_UNIT_VU, chk_sop, ex_sop, cyc_sop };
    const tsr_op_handler dqrow = { TSR_UNIT_VU, chk_dqrow, ex_dqrow, cyc_dqrow };
    tsr_sim_register_op(TSR_OP_VOP, &vop);
    tsr_sim_register_op(TSR_OP_RED, &red);
    tsr_sim_register_op(TSR_OP_ROPE, &rope);
    tsr_sim_register_op(TSR_OP_TOPK, &topk);
    tsr_sim_register_op(TSR_OP_SOP, &sop);
    tsr_sim_register_op(TSR_OP_DQROW, &dqrow);
}
