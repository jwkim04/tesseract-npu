// tsr_vu.c — the simulator's VU unit: decode, dispatch checks, operand extents, execution and
// cycle model of VOP 0x30, RED 0x31, ROPE 0x32, TOPK 0x33, SOP 0x40, DQROW 0x12 and GDN 0x39 on unit 3.
//
// Normative sources: docs/design/02-vector-unit.md (VU) §4 (operand model, n = 0 and tail rules, the s
// bank, aliasing), §5.1–§5.9 (encodings, dispatch checks, extents), §5.10 (CAPS, PERF) and §7 (the
// datapath the cycle model follows); docs/design/01-execution-model.md (EX) §6 (PERF), §7.1 (x16),
// §8.1 (error codes and their order); docs/design/06-software-verification.md (SW) §4.5. Every numeric
// result comes from tsr_ref_vu.c (which builds on tsr_ref_math.c and tsr_ref.c): this file decodes
// with the generated field constants (tsr_tables.h), checks, and points the references at the
// simulator's SPM and s bank. AT's vector ops 0x34–0x37 also run on unit 3 but are AT's ops
// (tsr_at.c). VMASK 0x38 has no handler, so the issue path completes it ILLEGAL_OP (CAPS
// bit7 = 0), as it does every reserved VU opcode (0x13, 0x3A–0x3F, 0x41–0x47, EX §7.2).
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
// only exactly) is the race detector's (VU §10 item 3), not a dispatch check: EX has no code
// for it.
//
// Execution: at completion (SW §4.1), sources read through the references after every earlier op's
// effect. s operands are read from the bank at exec time, which is the op's start for every
// observer because the queue model serves one VU op at a time (VU §4.4).
//
// Cycle model (unit busy time, deterministic, from the words and the configuration only; SW-80). It is the cycles
// of an op that waits for its predecessor: the unit's time from the done of the op before to its own done, the
// quantity the queue model serves one op at a time (VU §4.4). The RTL overlaps the fill and drain of independent
// ops (accept-ahead, VU §7.8); that gain is not modelled (VU §7.9). The constants are the unit's measured
// behaviour: tb/vu test_vu_timing.py (the RTL unit alone on a 4 MiB SPM, SPM_RD_LAT 4) at both configurations that
// were built, and on the card (the trace writer's per-op times of Qwen3.8 2B and Gemma 4 26B-A4B decode equal the
// RTL's to the cycle where the VU waits for nothing else). The lanes come from tsr_sim_opts.vu_timing: F front, E EXP
// and B back lanes (a line of 16 takes 16/F, 16/E and 16/B cycles per class) and P lane-pipe stages (X = 2P cycles
// at the exit of the lanes and of a partial drain read; U200 image 16 / 16 / 8, P = 0; K4 image 8 / 8 / 4, P = 2).
//   VOP    rows·L(m)·rate + ceil(rows/16) for a c vector + latency D: 59 + X for the front class (16/F per line)
//          and for EXP, EXPSUB and EXPSUBC (16/E per line), 95 + X for the half-rate class (16/B per line); one more
//          when the function reads both a and b. Odd line counts and rows cost nothing more: the RTL's pair reads
//          alternate the sources.
//   RED    a row streams L(m)·(16/F) cycles; its partials (P = min(L, 8) of the 16 ring entries) drain through the
//          subtree one group of 8 per cycle, G = min(ceil(m/8), 16) cycles, starting 17 cycles after the row's last
//          line; the entries are free again when the drain ends, so floor(16 / P) rows are in flight. The op ends
//          108 + X cycles after the last drain (106 + X to an s register), for every kind and post-op (the post pipeline
//          is fixed). The recurrence below is the RTL's ring; it reproduces every (m, rows) measured.
//   ROPE   T tokens through two stages, angles (A per token) then rotation (R = H·16/F, H = n_heads·head_dim/16):
//          (T - 1)·max(A, R) + A + R + D. A = 5 per angle of n_dims/2 (the recurrence form), 121/16 with freq factors (two
//          iterative dividers) and 17/16 with the A table; D = 127 + X, 136 + X with freq factors, 134 + X with the table.
//   TOPK   rows·(L(m)·16/F + k) + 20, DQROW rows·L(K)·16/F + 30 (not in the K4 image; the VU §7.9 estimates)
//   SOP    53 + 5X/4 (LDS 58 + 5X/4, SQRT and RSQRT 68, DIV 25)
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
#include "tsr_ref_gdn.h"

#include <string.h>

// SW-80: latencies at P = 0 (the U200 image); every stage of the lane pipe adds 2 (VU_X).
#define VU_LAT_F        59u   // VOP, front and EXP classes: accept of the dependent op to done (one more with a and b)
#define VU_LAT_H        95u   // VOP, half-rate class (one more with a and b)
#define VU_RED_T0       17u   // last line of a row to its first drain read
#define VU_RED_RING     16u   // partial ring entries
#define VU_RED_TAIL     108u  // last drain to the result in the SPM (106 for an s register)
#define VU_ROPE_ANG_N   80u   // 16ths of a cycle per angle: the recurrence form (5)
#define VU_ROPE_ANG_FF  121u  // with freq factors (two dividers: 121/16)
#define VU_ROPE_ANG_TAB 17u   // with the A table
#define VU_ROPE_LAT_N   127u
#define VU_ROPE_LAT_FF  136u
#define VU_ROPE_LAT_TAB 134u
#define VU_TOPK_LAT     20u   // VU §7.9 (sorted insertion per row; short class)
#define VU_DQROW_LAT    30u   // VU §7.9
#define VU_SOP_CYC      53u
#define VU_SOP_LDS      58u
#define VU_SOP_SQRT     68u
#define VU_SOP_DIV_CYC  25u
#define VU_GDN_BASE     122u  // GDN op: setup tail, dot pipeline, scalar chain and lane exit (fitted to the RTL, cyc_gdn)

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

// Spec range-check rule (docs/spec.md §21, Range check): [a, a + e) must end at or below the SPM's line count.
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

// The lanes of the cycle model (SW-80): cycles per line of the front, EXP and back lane classes and the lane pipe's
// 2P (tsr_sim_opts.vu_timing).
typedef struct vu_rates { uint64_t front, exp, back, x; } vu_rates;
static inline vu_rates vu_rates_of(const tsr_sim * sim) {
    const tsr_vu_timing t = tsr_sim_vu_timing(sim);
    vu_rates r = { (TSR_VL + t.front - 1u) / t.front, (TSR_VL + t.exp - 1u) / t.exp, (TSR_VL + t.back - 1u) / t.back, 2u * t.pipe };
    return r;
}

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

static int vop_exp_class(uint32_t func) {
    return func == TSR_VOP_FUNC_EXP || func == TSR_VOP_FUNC_EXPSUB || func == TSR_VOP_FUNC_EXPSUBC;
}

static uint64_t cyc_vop(const tsr_sim * sim, const uint32_t w[8]) {
    vop_dec v;
    vop_decode(w, &v);
    if (v.rows == 0u || v.m == 0u) return 1u;
    const vu_rates r = vu_rates_of(sim);
    const uint64_t L = lines_of(v.m);
    const int half = tsr_vop_func_half(v.func);
    uint64_t cyc = (uint64_t)v.rows * L * (half ? r.back : (vop_exp_class(v.func) ? r.exp : r.front));
    if (v.cmode == TSR_VOP_CMODE_VEC) cyc += lines_of(v.rows);
    const int two = (v.reads & TSR_VOP_READS_A) && (v.reads & TSR_VOP_READS_B);
    return cyc + (half ? VU_LAT_H : VU_LAT_F) + r.x + (two ? 1u : 0u);
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

// The RTL's partial ring (tsr_vu_inj.sv, tsr_vu_red.sv): a row of L lines takes P = min(L, 8) of 16 entries, streams
// L(m)·(16/F) cycles, is drained 17 cycles after its last line in G = min(ceil(m/8), 16) cycles (one group of 8
// partials per cycle, one row at a time) and frees its entries when the drain ends.
static uint64_t cyc_red(const tsr_sim * sim, const uint32_t w[8]) {
    red_dec r;
    red_decode(w, &r);
    if (!r.to_s && (r.rows == 0u || r.m == 0u)) return 1u;
    const vu_rates x = vu_rates_of(sim);
    const uint64_t tail = VU_RED_TAIL + x.x - (r.to_s ? 2u : 0u);
    if (r.m == 0u) return VU_RED_T0 + 1u + tail;                     // TO_S of an empty row: one group of init values
    const uint64_t L = lines_of(r.m), s = L * x.front;
    const uint64_t P = L < 8u ? L : 8u, in_flight = VU_RED_RING / P;
    const uint64_t G = (r.m + 7u) / 8u < 16u ? (r.m + 7u) / 8u : 16u;
    // The drain ends of the last `in_flight` rows are all the recurrence needs (the entries of row i come back at the
    // drain end of row i - in_flight); rows are at most 1024 (a RED has no more rows than the s bank's vector).
    uint64_t dend[VU_RED_RING] = { 0 };
    uint64_t t_end = 0, last = 0;
    for (uint32_t i = 0; i < r.rows; ++i) {
        uint64_t a = t_end;
        if (i >= in_flight && dend[i % in_flight] > a) a = dend[i % in_flight];
        t_end = a + s;
        const uint64_t ds = t_end + VU_RED_T0 > last ? t_end + VU_RED_T0 : last;
        last = ds + G;
        dend[i % in_flight] = last;
    }
    return last + tail;
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
    rope_dec r;
    rope_decode(w, &r);
    if (r.Tp == 0u) return 1u;
    const vu_rates x = vu_rates_of(sim);
    const uint64_t n_ang = r.nd / 2u;
    const uint64_t ang16 = r.atab ? VU_ROPE_ANG_TAB : (r.ff ? VU_ROPE_ANG_FF : VU_ROPE_ANG_N);
    const uint64_t A = (ang16 * n_ang + 15u) / 16u, R = (uint64_t)r.H * x.front;
    const uint64_t lat = (r.atab ? VU_ROPE_LAT_TAB : (r.ff ? VU_ROPE_LAT_FF : VU_ROPE_LAT_N)) + x.x;
    return (uint64_t)(r.T - 1u) * (A > R ? A : R) + A + R + lat;
}

// ---------------------------------------------------------------------------
// TOPK 0x33 (VU §5.5, §6.8; simulator and RTL option)
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
    topk_dec t;
    topk_decode(w, &t);
    if (t.rows == 0u || t.m == 0u) return 1u;
    return (uint64_t)t.rows * ((uint64_t)lines_of(t.m) * vu_rates_of(sim).front + t.k) + VU_TOPK_LAT;
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
    const uint64_t x = 5u * vu_rates_of(sim).x / 4u;          // 5 more cycles with the K4 lane pipe (X = 4)
    switch (TSR_INSN_GET(w, TSR_SOP_SUB)) {
    case TSR_SOP_SUB_DIV:   return VU_SOP_DIV_CYC;
    case TSR_SOP_SUB_SQRT:
    case TSR_SOP_SUB_RSQRT: return VU_SOP_SQRT;
    case TSR_SOP_SUB_LDS:   return VU_SOP_LDS + x;
    default:                return VU_SOP_CYC + x;
    }
}

// ---------------------------------------------------------------------------
// DQROW 0x12 (VU §5.6, §6.8; simulator and RTL option)
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
    dqrow_dec q;
    dqrow_decode(w, &q);
    if (q.rows == 0u) return 1u;
    return (uint64_t)q.rows * lines_of(q.K) * vu_rates_of(sim).front + VU_DQROW_LAT;
}

// ---------------------------------------------------------------------------
// GDN 0x39 (VU §7.12; CAPS bit 8)
// ---------------------------------------------------------------------------
static const uint32_t k_gdn_rsv[8] = { TSR_GDN_RSV_W0, TSR_GDN_RSV_W1, TSR_GDN_RSV_W2, TSR_GDN_RSV_W3,
                                       TSR_GDN_RSV_W4, TSR_GDN_RSV_W5, TSR_GDN_RSV_W6, TSR_GDN_RSV_W7 };

typedef struct gdn_dec {
    uint32_t sa, sb, d, a, k, q, v, o, m, rows;
} gdn_dec;

static void gdn_decode(const uint32_t w[8], gdn_dec * g) {
    g->sa   = TSR_INSN_GET(w, TSR_GDN_SA);
    g->sb   = TSR_INSN_GET(w, TSR_GDN_SB);
    g->d    = TSR_INSN_GET(w, TSR_GDN_D);
    g->a    = TSR_INSN_GET(w, TSR_GDN_A);
    g->k    = TSR_INSN_GET(w, TSR_GDN_K);
    g->q    = TSR_INSN_GET(w, TSR_GDN_Q);
    g->v    = TSR_INSN_GET(w, TSR_GDN_V);
    g->o    = TSR_INSN_GET(w, TSR_GDN_O);
    g->m    = TSR_INSN_GET(w, TSR_GDN_M);
    g->rows = TSR_INSN_GET(w, TSR_GDN_ROWS);
}

static inline int ranges_overlap(uint32_t alo, uint64_t alen, uint32_t blo, uint64_t blen) {
    return alen != 0u && blen != 0u && (uint64_t)alo < (uint64_t)blo + blen && (uint64_t)blo < (uint64_t)alo + alen;
}

// OPERAND (VU §7.12.4, in this order): m not a multiple of 16 or outside [64, 256]; rows 0 or above 1,024; d != a with
// overlapping extents; o overlapping d or a. SPM_RANGE in field order d, a, k, q, v, o. The extent list holds four SPM
// operands (d W, a R, k R, o W: every write is listed, the race detector needs them) and the s registers; q and v are not
// listed (CC §8.2: four slots).
static uint32_t chk_gdn(tsr_sim * sim, const uint32_t w[8], tsr_extents * ext, uint32_t * info) {
    if (rsv_bad(w, k_gdn_rsv)) return TSR_ERR_ILLEGAL_OP;
    gdn_dec g;
    gdn_decode(w, &g);
    if ((g.m & 15u) != 0u || g.m < TSR_GDN_MIN_M || g.m > TSR_GDN_MAX_M) { *info = g.m; return TSR_ERR_OPERAND; }
    if (g.rows == 0u || g.rows > TSR_GDN_MAX_ROWS) { *info = g.rows; return TSR_ERR_OPERAND; }
    const uint32_t nl = g.m / 16u;
    const uint64_t et = (uint64_t)g.rows * nl, ev = lines_of(g.rows);
    if (g.d != g.a && ranges_overlap(g.d, et, g.a, et)) { *info = g.d; return TSR_ERR_OPERAND; }
    if (ranges_overlap(g.o, ev, g.d, et) || ranges_overlap(g.o, ev, g.a, et)) { *info = g.o; return TSR_ERR_OPERAND; }
    if (spm_bad(sim, g.d, et, info) || spm_bad(sim, g.a, et, info) || spm_bad(sim, g.k, nl, info) ||
        spm_bad(sim, g.q, nl, info) || spm_bad(sim, g.v, ev, info) || spm_bad(sim, g.o, ev, info))
        return TSR_ERR_SPM_RANGE;
    ext_spm(ext, TSR_EXT_WRITE, g.d, et);
    ext_spm(ext, TSR_EXT_READ, g.a, et);
    ext_spm(ext, TSR_EXT_READ, g.k, nl);
    ext_spm(ext, TSR_EXT_WRITE, g.o, ev);
    ext_sreg(ext, TSR_EXT_READ, g.sa);
    ext_sreg(ext, TSR_EXT_READ, g.sb);
    return TSR_ERR_NONE;
}

// The vectors k, q and v are read before any word is written (the unit reads them in its first phase), so a d or o
// that overlaps them changes nothing; the tile may be updated in place.
static void ex_gdn(tsr_sim * sim, const uint32_t w[8]) {
    gdn_dec g;
    gdn_decode(w, &g);
    uint32_t k[TSR_GDN_MAX_M], q[TSR_GDN_MAX_M], v[TSR_GDN_MAX_ROWS];
    memcpy(k, spm_words(sim, g.k), (size_t)g.m * 4u);
    memcpy(q, spm_words(sim, g.q), (size_t)g.m * 4u);
    memcpy(v, spm_words(sim, g.v), (size_t)g.rows * 4u);
    uint32_t o[TSR_GDN_MAX_ROWS + 16u];
    tsr_ref_gdn(g.m, g.rows, sreg_get(sim, g.sa), sreg_get(sim, g.sb), k, q, v, spm_words(sim, g.a),
                spm_words(sim, g.d), o);
    const uint32_t nov = 16u * lines_of(g.rows);
    memcpy(spm_words(sim, g.o), o, (size_t)nov * 4u);
    vu_lines(sim, (uint64_t)g.rows * (g.m / 16u) + lines_of(g.rows));
}

// One head alone, the op waiting for its predecessor (a dependent repetition on the same tile): fitted to the RTL unit
// (tb/vu test_vu_timing.py vu_gdn_op: ten shapes, m 64 .. 256 and rows 16 .. 256, at the K4 and the U200 lanes; SW-78):
//   (rows + 1) L(m) R + ceil(rows (23 - 2 R) / 16) + 9 m / 2 + 122,   R = cycles per line of the front lanes (16 / F)
// The first term is the (rows + 1) m / F beats of VU §7.12.8, the others the setup (4 m + rows + 45 in the estimate of the
// VU §7.12.8 table, measured: a row costs a little more than its beats, the setup 4.5 cycles per element and 122 cycles).
// Equal to the RTL within 2 cycles at the K4 lanes (2,914 for 128 x 128, 8 lanes), within 1.5 % at the U200 lanes
// (1,898 against 1,917). Heads that follow each other with independent tiles overlap the next setup under the rows of
// the previous head in the RTL (VU §7.12.8); that gain is not modelled, as for the other ops (VU §7.9).
static uint64_t cyc_gdn(const tsr_sim * sim, const uint32_t w[8]) {
    gdn_dec g;
    gdn_decode(w, &g);
    const uint64_t R = vu_rates_of(sim).front;
    return (uint64_t)(g.rows + 1u) * lines_of(g.m) * R + ((uint64_t)g.rows * (23u - 2u * R) + 15u) / 16u +
           (9u * (uint64_t)g.m) / 2u + VU_GDN_BASE;
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
    const tsr_op_handler gdn   = { TSR_UNIT_VU, chk_gdn, ex_gdn, cyc_gdn };
    tsr_sim_register_op(TSR_OP_VOP, &vop);
    tsr_sim_register_op(TSR_OP_RED, &red);
    tsr_sim_register_op(TSR_OP_ROPE, &rope);
    tsr_sim_register_op(TSR_OP_TOPK, &topk);
    tsr_sim_register_op(TSR_OP_SOP, &sop);
    tsr_sim_register_op(TSR_OP_DQROW, &dqrow);
    tsr_sim_register_op(TSR_OP_GDN, &gdn);
}
