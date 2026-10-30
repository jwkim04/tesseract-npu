// tsr_fa_plan.h — the flash-attention chunk planner (docs/design/03-attention-kv.md "AT" §6.6 steps 2-3,
// §7.3; AT-15 pass rule, AT-19 chunk policy): ONE source for the reference and the firmware.
//
// tsr_ref_at.c (tsr_fa_pass_rule, tsr_fa_ncg, tsr_fa_plan, hence tsr_ref_fa) and the firmware's
// FLASH_ATTN kernel (sw/ctrl/kernels/k_fattn.cpp, through sw/ctrl/include/tsr/plan_attn.hpp) both
// compile this file, so their chunk lists, hence their tile-op sequences and numerics, are the same
// by construction (SW §7.5: the tile-op sequence is the numerical contract). W2b (P2.7) moved the
// planner of W1a's tsr_ref_at.c here; sw/ctrl/tests/test_fw_kvfa_plan.cpp keeps W1a's code as an
// oracle and compares the two chunk lists over random and adversarial mask summaries.
//
// Freestanding: C11 and C++17 / C++20, <stdint.h> only; no heap, no libc, no float, no 64-bit
// divide (a libgcc call on the RV32 control core; 64-bit values are only added, compared and
// divided by 16 or 2). Every function is static inline (no symbols: libtsr_ref keeps exporting its
// own tsr_fa_* wrappers, tsr_ref_at.h).
//
// Rules (AT §6.6 step 3, §7.3; the parts AT leaves open are W1a decisions), per query tile
// [q0, q0 + T) of the mask summary MSUM (rows {lo0, hi0, lo1, hi1}, tsr_ref_at.h):
//   1. MSUM GENERAL: one GENERAL sub-range over all n_kv cells, groups [0, ceil(n_kv/16)).
//   2. T = 1 (one query row: decode, or a prefill tile of one token): each of its 0..2 runs is a
//      FAST sub-range over the groups it intersects, with c_lo / c_hi from the run and no flags
//      (AT §7.3 "two FAST sub-ranges"). Two runs may share a group; it is then streamed twice,
//      with disjoint bounds.
//   3. T > 1: if every row has exactly one run and the runs are [max(0, lo - SLIDE·d_q),
//      hi - CAUSAL·d_q) with d_q = T - 1 - q for some lo, hi, CAUSAL, SLIDE (each flag set only
//      when the bounds actually differ between rows), one FAST sub-range over the groups of the
//      union [min L_q, max H_q). Otherwise one GENERAL sub-range over the groups of the hull of
//      all runs (it needs the f32 mask copy).
//   4. NCG = ncg(n_total) with n_total = the sum of the sub-ranges' groups. A sub-range
//      [Ulo, Uhi) is cut at Ulo + k·NCG.
//   5. FAST T > 1 only: a cut strictly inside the spread (min, max) of the rows' upper bounds
//      (CAUSAL) or lower bounds (SLIDE) cannot be encoded (c_lo / c_hi would exceed 16·n_groups),
//      so it is replaced by the cuts at the group boundaries just below and above the spread
//      (repeated until no cut is inside a spread, at most 8 rounds). Chunks only shrink, so NCG
//      and PCAP hold.
//   6. Per chunk [A, A') in cells: lower bound c_lo = 0 and no SLIDE if lo <= A, else lo - A and
//      SLIDE as the tile; upper bound c_hi = 16·n_groups and no CAUSAL if every row's upper bound
//      is >= A', else hi - A and CAUSAL as the tile. The planner verifies that the encoded bounds
//      give every row exactly its run inside the chunk (failure: -3, the whole tile is refused).
//   7. Order: chunks by (u0, c_lo) (generation order); then every chunk whose groups contain a
//      cell written in this launch moves to the end, keeping its order (AT §6.6 "newest last").
//   8. No visible cell for any row: one chunk, group 0, c_lo = c_hi = 0 (all -inf: m = M_FLOOR,
//      p = 0, l = 0), so ANORM writes 0 as the CPU does (AT §7.3).
//
// Use: tsr_fap_tile_init (rules 1-5: the sub-ranges, NCG, rule 5's cuts), then tsr_fap_next until
// it returns 0 (rules 6-8: one chunk per call, in execution order; the reference's tsr_fa_plan), or
// tsr_fap_list (the same list at once, every chunk generated once; the firmware, with a bounded
// array: a longer list falls back to tsr_fap_next). The caller supplies "has this chunk a cell
// written in this launch" (rule 7) as a callback.
//
// Rule 5 without a cut array (W2b): W1a's code held every cut of a sub-range in a heap array; here
// the regular cuts Ulo + k·NCG are generated on the fly and only the cuts rule 5 touches are kept
// (`near`). A regular cut that is not strictly inside a spread is never changed by rule 5 and adds
// no cut, and a spread (an open interval of at most T - 1 <= 7 cells) holds at most one group
// boundary, so rule 5 starts from at most two regular cuts; the rounds of W1a's code applied to
// that sorted sub-list (same order of visits, same appends, same sort / dedup / range drop, same
// round limit) give the same cuts, and the chunk walk merges them with the untouched regular cuts
// (dropping duplicates). A round appends at most 4 cuts and the list holds at most 2 + 4 distinct
// cuts after a round, so W1a's array never reached its capacity and TSR_FAP_NEAR_MAX = 16 always
// suffices (an overflow, which cannot happen, would be reported as -3).
#ifndef TSR_FA_PLAN_H
#define TSR_FA_PLAN_H

#include <stdint.h>

#define TSR_FAP_T_MAX    8u     // query rows per tile (= TSR_T_MAX_ARCH)
#define TSR_FAP_CAUSAL   0x1u   // chunk flags = ATTQK w0[13:12] >> 12 (= TSR_ATT_CAUSAL)
#define TSR_FAP_SLIDE    0x2u   //                                     (= TSR_ATT_SLIDE)
#define TSR_FAP_GENERAL  0x1u   // MSUM flags bit0 (= TSR_MSUM_GENERAL)
#define TSR_FAP_NEAR_MAX 16u    // rule 5's cut list (see the header comment)
#define TSR_FAP_START    0xFFFFFFFFu

// The pass rule and NCG are constexpr in C++ (tsr/plan_attn.hpp's planners use them in constant
// expressions); the rest is plain static inline.
#if defined(__cplusplus)
#define TSR_FAP_CONSTEXPR constexpr
#else
#define TSR_FAP_CONSTEXPR
#endif

typedef struct tsr_fap_chunk {
    uint32_t u0, n_groups;   // groups [u0, u0 + n_groups)
    uint32_t c_lo, c_hi;     // ATTQK cell bounds relative to 16·u0 (GENERAL: 0, 16·n_groups)
    uint32_t flags;          // TSR_FAP_CAUSAL | TSR_FAP_SLIDE (FAST only)
    uint32_t general;        // 1: GENERAL path (ATTQK without bounds, then AMASK)
} tsr_fap_chunk;

typedef struct tsr_fap_sub {
    uint32_t ulo, uhi;       // groups [ulo, uhi)
    uint32_t general;        // 1: GENERAL
    uint32_t multi;          // FAST with T > 1 (bounds from lo / hi / flags)
    int64_t  lo, hi;         // FAST: T = 1 the run; T > 1 lo_abs, hi_abs
    uint32_t flags;          // FAST T > 1: CAUSAL | SLIDE of the tile
    uint32_t n_rm;           // rule 5: regular cuts it replaced (skipped by the walk)
    uint32_t rm[2];
    uint32_t n_near;         // rule 5's cuts: sorted, distinct, inside (ulo, uhi)
    uint32_t near[TSR_FAP_NEAR_MAX];
} tsr_fap_sub;

typedef struct tsr_fap_tile {
    uint32_t T;              // query rows of the tile (1..8)
    uint32_t ncg;            // NCG (rule 4); 0 when nsub = 0
    uint32_t nsub;           // sub-ranges; 0: no visible cell (rule 8)
    int64_t  L[TSR_FAP_T_MAX], H[TSR_FAP_T_MAX];   // FAST T > 1: each row's run
    tsr_fap_sub sub[2];
} tsr_fap_tile;

typedef struct tsr_fap_iter {
    uint32_t pass;           // 0: chunks without new cells, 1: with them, 2: done
    uint32_t k;              // sub-range
    uint32_t a;              // the next chunk's first group (TSR_FAP_START: the sub-range starts)
    uint32_t fin;            // the sub-range's last chunk was generated
    uint64_t rnext;          // the next regular cut
    uint32_t ni;             // the next of near[]
} tsr_fap_iter;

// Rule 7's question: does the chunk [u0, u0 + n_groups) hold a cell written in this launch (0 / 1).
typedef int (*tsr_fap_has_new_fn)(void * ctx, uint32_t u0, uint32_t n_groups);

static inline int64_t tsr_fap_min64(int64_t a, int64_t b) { return a < b ? a : b; }
static inline int64_t tsr_fap_max64(int64_t a, int64_t b) { return a > b ? a : b; }

// Pass rule (AT §6.6 step 2): GS = min(G, T_MAX), T_q = floor(T_MAX / GS); the last head subset
// of a group has GS' = G - hg0 heads, the last query tile T - q0 tokens.
static inline TSR_FAP_CONSTEXPR void tsr_fap_pass_rule(uint32_t t_max, uint32_t G, uint32_t * GS, uint32_t * T_q) {
    const uint32_t gs = G < t_max ? G : t_max;
    *GS = gs;
    *T_q = gs ? t_max / gs : 0u;
}

// NCG = min(128, C·floor(PCAP/(16·E/D)), max(4C, roundup_4C(ceil(n_total/2)))) (AT §6.6 step 3);
// 0 when the shape admits no chunk. tsr_fap_pc is the op's term C·floor(PCAP/(16·E/D)) (0 for a
// shape without chunks: D = 0, E % D != 0, E < D, C = 0), tsr_fap_ncg_pc the rest (the firmware
// computes pc once per op, its divides out of the per-tile planner). No 64-bit divide (the
// firmware's libgcc call): with half = ceil(n_total/2) <= 2^31, max(4C, roundup_4C(half)) is 4C
// when 4C >= half (half = 0 included), else roundup_4C(half) with 4C < half, which 32 bits hold.
static inline TSR_FAP_CONSTEXPR uint64_t tsr_fap_pc(uint32_t C, uint32_t pcap, uint32_t E, uint32_t D) {
    if (D == 0u || E % D != 0u || C == 0u || E / D == 0u) return 0u;
    return (uint64_t)C * (pcap / (16u * (E / D)));
}
static inline TSR_FAP_CONSTEXPR uint32_t tsr_fap_ncg_pc(uint32_t C, uint64_t pc, uint32_t n_total) {
    if (C == 0u || pc == 0u) return 0u;
    const uint32_t half = (uint32_t)(((uint64_t)n_total + 1u) / 2u);
    const uint64_t c4 = 4u * (uint64_t)C;
    uint64_t m = c4;
    if (c4 < half) {
        const uint32_t c = (uint32_t)c4;
        m = (uint64_t)((half + c - 1u) / c * c);
    }
    uint64_t ncg = 128u;
    if (pc < ncg) ncg = pc;
    if (m < ncg) ncg = m;
    return (uint32_t)ncg;
}
static inline TSR_FAP_CONSTEXPR uint32_t tsr_fap_ncg(uint32_t C, uint32_t pcap, uint32_t E, uint32_t D, uint32_t n_total) {
    return tsr_fap_ncg_pc(C, tsr_fap_pc(C, pcap, E, D), n_total);
}

// An MSUM row is well formed: runs inside [0, n_kv], lo <= hi, two nonempty runs disjoint.
static inline int tsr_fap_run_ok(const uint32_t * w, uint32_t n_kv) {
    if (w[0] > w[1] || w[2] > w[3] || w[1] > n_kv || w[3] > n_kv) return 0;
    if (w[1] > w[0] && w[3] > w[2] && !(w[1] <= w[2] || w[3] <= w[0])) return 0;   // overlapping runs
    return 1;
}

// Nonempty runs of an MSUM row, ascending.
static inline uint32_t tsr_fap_row_runs(const uint32_t * w, int64_t lo[2], int64_t hi[2]) {
    uint32_t n = 0;
    if (w[1] > w[0]) { lo[n] = w[0]; hi[n] = w[1]; ++n; }
    if (w[3] > w[2]) { lo[n] = w[2]; hi[n] = w[3]; ++n; }
    if (n == 2 && lo[1] < lo[0]) {
        const int64_t a = lo[0], b = hi[0];
        lo[0] = lo[1]; hi[0] = hi[1]; lo[1] = a; hi[1] = b;
    }
    return n;
}

static inline void tsr_fap_sub_set(tsr_fap_sub * s, uint32_t ulo, uint32_t uhi, uint32_t general, uint32_t multi,
                                   int64_t lo, int64_t hi, uint32_t flags) {
    s->ulo = ulo;
    s->uhi = uhi;
    s->general = general;
    s->multi = multi;
    s->lo = lo;
    s->hi = hi;
    s->flags = flags;
    s->n_rm = 0;
    s->n_near = 0;
}

// Rule 5 for one sub-range (see the header comment): the regular cuts strictly inside a spread,
// then W1a's rounds on them. 0, or -3 (capacity; cannot happen).
static inline int tsr_fap_sub_cuts(tsr_fap_sub * s, uint32_t ncg, uint32_t T) {
    s->n_rm = 0;
    s->n_near = 0;
    if (!s->multi || s->flags == 0u) return 0;
    int64_t sp[2][2];
    uint32_t nsp = 0;
    if (s->flags & TSR_FAP_CAUSAL) { sp[nsp][0] = s->hi - (int64_t)(T - 1u); sp[nsp][1] = s->hi; ++nsp; }
    if (s->flags & TSR_FAP_SLIDE)  { sp[nsp][0] = tsr_fap_max64(0, s->lo - (int64_t)(T - 1u)); sp[nsp][1] = s->lo; ++nsp; }
    uint32_t * cuts = s->near;
    uint32_t n = 0;
    // The regular cuts inside a spread, ascending and distinct: the multiple of 16 just above sp0
    // is the only candidate (sp1 - sp0 <= 7).
    for (uint32_t t = 0; t < nsp; ++t) {
        const int64_t c = sp[t][0] >= 0 ? sp[t][0] / 16 + 1 : 0;
        if (!(16 * c < sp[t][1]) || c <= (int64_t)s->ulo || c >= (int64_t)s->uhi) continue;
        if (((uint32_t)c - s->ulo) % ncg != 0u) continue;   // ulo < c < uhi: 32 bits
        const uint32_t v = (uint32_t)c;
        if (n == 1u && cuts[0] == v) continue;
        if (n == 1u && cuts[0] > v) { cuts[1] = cuts[0]; cuts[0] = v; }
        else cuts[n] = v;
        ++n;
    }
    s->n_rm = n;
    for (uint32_t i = 0; i < n; ++i) s->rm[i] = cuts[i];
    if (n == 0u) return 0;
    // W1a's rounds (tsr_ref_at.c plan_cuts as of a54a6e6, unchanged up to 7dc1453), on the sub-list.
    for (int iter = 0; iter < 8; ++iter) {
        int changed = 0;
        for (uint32_t k = 0; k < n; ++k) {
            const int64_t X = 16 * (int64_t)cuts[k];
            for (uint32_t t = 0; t < nsp; ++t) {
                if (sp[t][0] < X && X < sp[t][1]) {
                    // replace X by the group boundaries just below and above the spread
                    const uint32_t lo = (uint32_t)(sp[t][0] / 16), hi = (uint32_t)((sp[t][1] + 15) / 16);
                    cuts[k] = lo;
                    if (n >= TSR_FAP_NEAR_MAX) return -3;
                    cuts[n++] = hi;
                    changed = 1;
                    break;
                }
            }
        }
        // sort, drop duplicates and cuts outside (ulo, uhi)
        for (uint32_t i = 1; i < n; ++i) {
            const uint32_t v = cuts[i];
            uint32_t j = i;
            while (j > 0 && cuts[j - 1] > v) { cuts[j] = cuts[j - 1]; --j; }
            cuts[j] = v;
        }
        uint32_t m = 0;
        for (uint32_t i = 0; i < n; ++i) {
            if (cuts[i] <= s->ulo || cuts[i] >= s->uhi) continue;
            if (m > 0 && cuts[m - 1] == cuts[i]) continue;
            cuts[m++] = cuts[i];
        }
        n = m;
        if (!changed) break;
    }
    s->n_near = n;
    return 0;
}

// Rules 1-5 for the tile of T_q query rows whose MSUM rows (4 words each, row 0 = the tile's first
// row) are `rows`; msum_flags and n_kv are the MSUM header's; pc = tsr_fap_pc(C, PCAP, E, D), the
// op's NCG term (tsr_fap_tile_init computes it). Returns 0, -1 (the shape admits no chunk: NCG = 0),
// -2 (bad T_q or a malformed MSUM row) or -3 (rule 5's capacity: cannot happen). rows is not read
// when the MSUM is GENERAL (rule 1).
static inline int tsr_fap_tile_init_pc(tsr_fap_tile * t, uint32_t C, uint64_t pc, uint32_t n_kv, uint32_t msum_flags,
                                       const uint32_t * rows, uint32_t T_q) {
    t->T = T_q;
    t->ncg = 0;
    t->nsub = 0;
    for (uint32_t q = 0; q < TSR_FAP_T_MAX; ++q) { t->L[q] = 0; t->H[q] = 0; }
    if (T_q == 0u || T_q > TSR_FAP_T_MAX) return -2;
    uint32_t nsub = 0;
    if (msum_flags & TSR_FAP_GENERAL) {
        if (n_kv > 0u) {
            tsr_fap_sub_set(&t->sub[0], 0u, (n_kv + 15u) >> 4, 1u, 0u, 0, 0, 0u);
            nsub = 1;
        }
    } else {
        int64_t rlo[TSR_FAP_T_MAX][2], rhi[TSR_FAP_T_MAX][2];
        uint32_t nr[TSR_FAP_T_MAX];
        int all_one = 1;
        int64_t hlo = INT64_MAX, hhi = -1;
        for (uint32_t q = 0; q < T_q; ++q) {
            const uint32_t * w = rows + 4u * q;
            if (!tsr_fap_run_ok(w, n_kv)) return -2;
            nr[q] = tsr_fap_row_runs(w, rlo[q], rhi[q]);
            if (nr[q] != 1u) all_one = 0;
            for (uint32_t k = 0; k < nr[q]; ++k) { hlo = tsr_fap_min64(hlo, rlo[q][k]); hhi = tsr_fap_max64(hhi, rhi[q][k]); }
        }
        if (T_q == 1u) {
            for (uint32_t k = 0; k < nr[0]; ++k) {
                tsr_fap_sub_set(&t->sub[nsub++], (uint32_t)(rlo[0][k] / 16), (uint32_t)((rhi[0][k] + 15) / 16), 0u, 0u,
                                rlo[0][k], rhi[0][k], 0u);
            }
        } else if (hhi >= 0) {
            int fit = all_one;
            uint32_t fl = 0;
            int64_t lo_abs = 0, hi_abs = 0;
            if (fit) {
                for (uint32_t q = 0; q < T_q; ++q) { t->L[q] = rlo[q][0]; t->H[q] = rhi[q][0]; }
                hi_abs = t->H[T_q - 1];
                lo_abs = t->L[T_q - 1];
                int heq = 1, leq = 1, hca = 1, lsl = lo_abs > 0;
                for (uint32_t q = 0; q < T_q; ++q) {
                    const int64_t dq = (int64_t)T_q - 1 - q;
                    heq &= t->H[q] == hi_abs;
                    leq &= t->L[q] == lo_abs;
                    hca &= t->H[q] == hi_abs - dq;
                    lsl &= t->L[q] == tsr_fap_max64(0, lo_abs - dq);
                }
                if (!heq) { if (hca) fl |= TSR_FAP_CAUSAL; else fit = 0; }
                if (!leq) { if (lsl) fl |= TSR_FAP_SLIDE; else fit = 0; }
            }
            if (fit) {
                int64_t mnL = INT64_MAX, mxH = -1;
                for (uint32_t q = 0; q < T_q; ++q) { mnL = tsr_fap_min64(mnL, t->L[q]); mxH = tsr_fap_max64(mxH, t->H[q]); }
                tsr_fap_sub_set(&t->sub[nsub++], (uint32_t)(mnL / 16), (uint32_t)((mxH + 15) / 16), 0u, 1u, lo_abs, hi_abs, fl);
            } else {
                tsr_fap_sub_set(&t->sub[nsub++], (uint32_t)(hlo / 16), (uint32_t)((hhi + 15) / 16), 1u, 0u, 0, 0, 0u);
            }
        }
    }
    t->nsub = nsub;
    if (nsub == 0u) return 0;          // rule 8: tsr_fap_next gives the empty chunk
    uint64_t n_total = 0;
    for (uint32_t k = 0; k < nsub; ++k) n_total += t->sub[k].uhi - t->sub[k].ulo;
    t->ncg = tsr_fap_ncg_pc(C, pc, (uint32_t)n_total);
    if (t->ncg == 0u) return -1;
    for (uint32_t k = 0; k < nsub; ++k) {
        if (tsr_fap_sub_cuts(&t->sub[k], t->ncg, T_q) != 0) return -3;
    }
    return 0;
}

static inline int tsr_fap_tile_init(tsr_fap_tile * t, uint32_t C, uint32_t pcap, uint32_t E, uint32_t D,
                                    uint32_t n_kv, uint32_t msum_flags, const uint32_t * rows, uint32_t T_q) {
    return tsr_fap_tile_init_pc(t, C, tsr_fap_pc(C, pcap, E, D), n_kv, msum_flags, rows, T_q);
}

static inline void tsr_fap_iter_init(tsr_fap_iter * it) {
    it->pass = 0;
    it->k = 0;
    it->a = TSR_FAP_START;
    it->fin = 0;
    it->rnext = 0;
    it->ni = 0;
}

// Rule 6 for the chunk [a, b) of sub-range s: its encoding, verified for FAST T > 1. 0 or -3.
static inline int tsr_fap_encode(const tsr_fap_tile * t, const tsr_fap_sub * s, uint32_t a, uint32_t b,
                                 tsr_fap_chunk * c) {
    const uint32_t T = t->T;
    c->u0 = a;
    c->n_groups = b - a;
    c->flags = 0;
    c->general = 0;
    const int64_t A = 16 * (int64_t)a, A2 = 16 * (int64_t)b, n16 = A2 - A;
    if (s->general) {
        c->general = 1;
        c->c_lo = 0;
        c->c_hi = (uint32_t)n16;
    } else if (!s->multi) {
        c->c_lo = (uint32_t)tsr_fap_max64(s->lo - A, 0);
        c->c_hi = (uint32_t)tsr_fap_min64(s->hi - A, n16);
    } else {
        const int causal = (s->flags & TSR_FAP_CAUSAL) != 0, slide = (s->flags & TSR_FAP_SLIDE) != 0;
        int64_t clo, chi;
        uint32_t fl = 0;
        if (s->lo <= A) clo = 0;
        else { clo = s->lo - A; if (slide) fl |= TSR_FAP_SLIDE; }
        if (s->hi - (causal ? (int64_t)(T - 1u) : 0) >= A2) chi = n16;
        else { chi = s->hi - A; if (causal) fl |= TSR_FAP_CAUSAL; }
        if (clo < 0 || clo > chi || chi > n16) return -3;
        for (uint32_t q = 0; q < T; ++q) {   // the encoding must give each row its run in the chunk
            const int64_t dq = (int64_t)T - 1 - q;
            const int64_t lo_q = tsr_fap_max64(0, clo - ((fl & TSR_FAP_SLIDE) ? dq : 0));
            const int64_t hi_q = tsr_fap_min64(chi - ((fl & TSR_FAP_CAUSAL) ? dq : 0), n16);
            const int64_t elo = tsr_fap_max64(t->L[q] - A, 0), ehi = tsr_fap_min64(t->H[q] - A, n16);
            const int empty_a = hi_q <= lo_q, empty_b = ehi <= elo;
            if (empty_a != empty_b || (!empty_a && (lo_q != elo || hi_q != ehi))) return -3;
        }
        c->c_lo = (uint32_t)clo;
        c->c_hi = (uint32_t)chi;
        c->flags = fl;
    }
    return 0;
}

// The next chunk in generation order (rule 6: sub-range by sub-range, cut by cut), the pass state
// of `it` aside. 1 (*out written), 0 (no more) or -3 (its encoding does not reproduce its rows'
// runs). For nsub = 0 the caller handles rule 8.
static inline int tsr_fap_gen(const tsr_fap_tile * t, tsr_fap_iter * it, tsr_fap_chunk * out) {
    for (;;) {
        if (it->k >= t->nsub) return 0;
        const tsr_fap_sub * s = &t->sub[it->k];
        if (it->a == TSR_FAP_START) {
            it->a = s->ulo;
            it->fin = 0;
            it->rnext = (uint64_t)s->ulo + t->ncg;
            it->ni = 0;
        }
        if (it->fin) {
            it->k += 1u;
            it->a = TSR_FAP_START;
            continue;
        }
        // The next cut: the smaller of the next regular cut not replaced by rule 5 and the next of
        // rule 5's cuts (equal ones once); uhi after the last.
        while (it->rnext < s->uhi && ((s->n_rm > 0u && it->rnext == s->rm[0]) || (s->n_rm > 1u && it->rnext == s->rm[1])))
            it->rnext += t->ncg;
        uint64_t b = s->uhi;
        if (it->rnext < b) b = it->rnext;
        if (it->ni < s->n_near && s->near[it->ni] < b) b = s->near[it->ni];
        if (it->rnext == b) it->rnext += t->ncg;
        if (it->ni < s->n_near && s->near[it->ni] == b) it->ni += 1u;
        if (b >= s->uhi) { b = s->uhi; it->fin = 1; }
        const uint32_t a = it->a;
        it->a = (uint32_t)b;
        return tsr_fap_encode(t, s, a, (uint32_t)b, out) != 0 ? -3 : 1;
    }
}

static inline void tsr_fap_empty_chunk(tsr_fap_chunk * out) {   // rule 8
    out->u0 = 0; out->n_groups = 1; out->c_lo = 0; out->c_hi = 0; out->flags = 0; out->general = 0;
}

// The next chunk in execution order (rules 6-8). Returns 1 (*out written), 0 (no more chunks) or -3
// (a chunk's encoding does not reproduce its rows' runs: the tile is refused; every chunk of both
// passes is verified, so a dry run to the end finds it before anything is used). has_new may be
// NULL (no cell written in this launch).
static inline int tsr_fap_next(const tsr_fap_tile * t, tsr_fap_iter * it, tsr_fap_has_new_fn has_new, void * ctx,
                               tsr_fap_chunk * out) {
    if (t->nsub == 0u) {               // rule 8
        if (it->pass != 0u) return 0;
        it->pass = 2;
        tsr_fap_empty_chunk(out);
        return 1;
    }
    for (;;) {
        if (it->pass >= 2u) return 0;
        tsr_fap_chunk c;
        const int r = tsr_fap_gen(t, it, &c);
        if (r < 0) return -3;
        if (r == 0) {                  // the pass ends: the next one starts at sub-range 0
            it->pass += 1u;
            it->k = 0;
            it->a = TSR_FAP_START;
            continue;
        }
        const int nw = has_new ? (has_new(ctx, c.u0, c.n_groups) != 0) : 0;
        if ((uint32_t)nw != it->pass) continue;
        *out = c;
        return 1;
    }
}

// The tile's whole chunk list in execution order into out[0 .. count), generating every chunk once
// (tsr_fap_next generates each chunk once per pass): the chunks without a new cell in generation
// order, then those with one in generation order (rule 7), as tsr_fap_next lists them; the same
// verification (rule 6). Returns the count, -3 (a chunk's encoding: the tile is refused), or
// count > cap with out[] unspecified (the caller then walks with tsr_fap_next).
static inline long tsr_fap_list(const tsr_fap_tile * t, tsr_fap_has_new_fn has_new, void * ctx, tsr_fap_chunk * out,
                                uint32_t cap) {
    if (t->nsub == 0u) {
        if (cap > 0u) tsr_fap_empty_chunk(out);
        return 1;
    }
    tsr_fap_iter it;
    tsr_fap_iter_init(&it);
    uint32_t n0 = 0, n1 = 0;           // pass-0 chunks from the front, pass-1 chunks from the back
    for (;;) {
        tsr_fap_chunk c;
        const int r = tsr_fap_gen(t, &it, &c);
        if (r < 0) return -3;
        if (r == 0) break;
        const int nw = has_new ? (has_new(ctx, c.u0, c.n_groups) != 0) : 0;
        if (n0 + n1 < cap) {
            if (nw) out[cap - 1u - n1] = c;
            else out[n0] = c;
        }
        if (nw) ++n1;
        else ++n0;
    }
    const uint32_t n = n0 + n1;
    if (n > cap) return (long)n;
    for (uint32_t i = 0; i < n1; ++i) out[n0 + i] = out[cap - 1u - i];   // the back, reversed, after the front
    return (long)n;
}

#endif  // TSR_FA_PLAN_H
