// tsr_ref_at.c — references of area AT (docs/design/03-attention-kv.md), bit for bit.
//
// Normative sources: AT §4 (KV plane layout), §5.1–§5.7 (the seven ops), §6 (numerics), §6.6 (the
// composite FA, normative), §7.3 (mask summary). The simulator's AT units, the firmware's host
// reference and the backend's KV tensor conversion call these functions; the RTL must match them
// bit for bit. Rules as tsr_ref.c / tsr_ref_math.c (docs/design/06-software-verification.md §4.5):
// every f32 operation is a tsr_fadd / tsr_fsub / tsr_fmul / tsr_fdiv / tsr_frecip call (DAZ, RNE,
// FTZ, canonical NaN), EXP is tsr_m_exp (VU §6.3), everything else is integer bit manipulation;
// device images are accessed byte-wise little-endian. Built with -ffp-contract=off -fno-fast-math.
//
// Choices this file makes where AT leaves them open (each is part of the bit-exact contract):
//   - Words an op does not define are unchanged: rows of other passes (ATTQK scores and MAXOUT
//     words, ATTPV partial and LSUM words), and the words past R of the last line of a row vector
//     (MAXOUT, LSUM, AMASK mdst, AMERGE m_run / l_run). Multi-pass FA (G > T_MAX) shares those
//     lines between passes, so they must keep the other pass's words (AT §5.2, §8.1 speak of
//     full-line gather writes; the RTL needs byte enables or a merge there).
//   - ATTPV skips a cell when DAZ(p) is +-0 (a subnormal p counts as zero, as every fp input
//     does), so 0·inf of a skipped cell is not a NaN. LSUM uses the same test.
//   - ATTPV E_S counts the finite nonzero w of both blocks of the slice even when one block is
//     non-finite (AT §6.4 step 4 as written); that block's lanes are NaN regardless.
//   - The FA composite runs the head subsets of a query tile inside each chunk (all ATTQK of the
//     chunk, AMASK, one ASMAX, all ATTPV, one AMERGE), so every VU op covers all R rows with
//     valid data; AT §6.6 lists the passes outermost, which AMERGE / ANORM (no row selection) cannot
//     express for G > T_MAX. Every op is row-independent, so each row gets the per-pass result.
//   - Chunk planner (AT §6.6 step 3, §7.3): include/tsr_fa_plan.h (shared with the firmware;
//     the "Chunk planner" section below wraps it).
//   - The f32 mask copy (AT §7.3) holds fdiv(m, scale): AMASK adds the mask before ASMAX applies
//     the scale, so a finite m must be pre-divided to give the CPU's scale·s + m (AT §5.5 as written
//     would give scale·(s + m)); see tsr_msum_mask_f32.

#include "tsr_ref_at.h"

#include "tsr_ref_math.h"
#include "tsr_fa_plan.h"   // the chunk planner shared with the firmware

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__FAST_MATH__)
#error "tsr_ref_at.c must not be compiled with -ffast-math"
#endif

_Static_assert(sizeof(tsr_msum_row) == 4u * sizeof(uint32_t), "tsr_msum_row is the 4-word MSUM row");

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------
static void at_fatal(const char * fn, const char * msg, uint64_t a, uint64_t b) {
    fprintf(stderr, "tsr_ref_at: %s: %s (%llu, %llu)\n", fn, msg, (unsigned long long)a, (unsigned long long)b);
    abort();
}

#define F32_SIGN     0x80000000u
#define F32_EXP      0x7F800000u
#define F32_MANT     0x007FFFFFu
#define F32_NEG_INF  0xFF800000u

static inline uint32_t fb(float f) { return tsr_f32_bits(f); }
static inline float    bf(uint32_t u) { return tsr_bits_f32(u); }
static inline uint32_t daz_bits(uint32_t u) { return ((u & F32_EXP) == 0) ? (u & F32_SIGN) : u; }
static inline int      is_nan_bits(uint32_t u) { return (u & ~F32_SIGN) > F32_EXP; }
static inline int      nonfinite_bits(uint32_t u) { return (u & F32_EXP) == F32_EXP; }

static inline uint16_t rd_u16le(const uint8_t * p) { return (uint16_t)(p[0] | (p[1] << 8)); }
static inline uint32_t rd_u32le(const uint8_t * p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static inline void wr_u16le(uint8_t * p, uint16_t v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); }

static inline uint32_t ceil16(uint32_t n) { return (n + 15u) >> 4; }
static inline int64_t  i64max(int64_t a, int64_t b) { return a > b ? a : b; }
static inline int64_t  i64min(int64_t a, int64_t b) { return a < b ? a : b; }

static void * xmalloc(size_t n) {
    void * p = malloc(n ? n : 1u);
    if (p == NULL) at_fatal(__func__, "out of memory", n, 0);
    return p;
}

// ---------------------------------------------------------------------------
// KV plane layout (AT §4)
// ---------------------------------------------------------------------------
static void geom_check(const char * fn, const tsr_kv_geom * g) {
    if (g == NULL) at_fatal(fn, "geometry is NULL", 0, 0);
    if (g->E == 0 || (g->E & 63u) != 0) at_fatal(fn, "E must be a positive multiple of 64", g->E, 0);
    if (g->C == 0 || g->C > TSR_NUM_CH_MAX) at_fatal(fn, "C must be in 1..16", g->C, 0);
    if (g->LG == 0) at_fatal(fn, "LG must be >= 1", g->LG, 0);
}

uint32_t tsr_kv_lg_min(uint32_t kv_size, uint32_t C) {
    if (C == 0) at_fatal(__func__, "C must be > 0", kv_size, C);
    return (uint32_t)(((uint64_t)kv_size + 16u * (uint64_t)C - 1u) / (16u * (uint64_t)C));
}
uint32_t tsr_kv_group_beats(uint32_t E) { return 17u * (E / 64u); }
uint64_t tsr_kv_plane_beats(const tsr_kv_geom * g) {
    geom_check(__func__, g);
    return (uint64_t)g->LG * tsr_kv_group_beats(g->E);
}
uint64_t tsr_kv_plane_cells(const tsr_kv_geom * g) {
    geom_check(__func__, g);
    return 16u * (uint64_t)g->C * g->LG;
}

// K (AT §4.3): beta = r·(E/32) + b, kappa = beta >> 5, o = beta & 31; meta beat l·GB + 17 kappa,
// bytes [2o, 2o+2); quant beat l·GB + 17 kappa + 1 + (o >> 1), bytes [32 (o & 1), +32). LG unused.
static inline void addr_k_raw(uint32_t E, uint32_t C, uint32_t cell, uint32_t blk, tsr_kv_loc * loc) {
    const uint32_t u = cell >> 4, r = cell & 15u, l = u / C;
    const uint64_t beta  = (uint64_t)r * (E / 32u) + blk;
    const uint64_t chunk = (uint64_t)l * tsr_kv_group_beats(E) + 17u * (beta >> 5);
    const uint32_t o = (uint32_t)(beta & 31u);
    loc->ch    = u % C;
    loc->meta  = chunk * TSR_BEAT_BYTES + 2u * o;
    loc->quant = (chunk + 1u + (o >> 1)) * TSR_BEAT_BYTES + 32u * (o & 1u);
}
// V (AT §4.4): slice s = b >> 1, chunk beat (s·LG + l)·17; meta bytes [4r + 2 (b & 1), +2); quant
// beat chunk + 1 + r, bytes [32 (b & 1), +32).
static inline void addr_v_raw(uint32_t C, uint32_t LG, uint32_t cell, uint32_t blk, tsr_kv_loc * loc) {
    const uint32_t u = cell >> 4, r = cell & 15u, l = u / C;
    const uint64_t chunk = ((uint64_t)(blk >> 1) * LG + l) * 17u;
    loc->ch    = u % C;
    loc->meta  = chunk * TSR_BEAT_BYTES + 4u * r + 2u * (blk & 1u);
    loc->quant = (chunk + 1u + r) * TSR_BEAT_BYTES + 32u * (blk & 1u);
}

static void addr_check(const char * fn, const tsr_kv_geom * g, uint32_t cell, uint32_t blk) {
    geom_check(fn, g);
    if ((uint64_t)cell >= tsr_kv_plane_cells(g)) at_fatal(fn, "cell beyond the plane", cell, tsr_kv_plane_cells(g));
    if (blk >= g->E / 32u) at_fatal(fn, "block beyond the row", blk, g->E / 32u);
}

void tsr_kv_addr_k(const tsr_kv_geom * g, uint32_t cell, uint32_t blk, tsr_kv_loc * loc) {
    addr_check(__func__, g, cell, blk);
    addr_k_raw(g->E, g->C, cell, blk, loc);
}
void tsr_kv_addr_v(const tsr_kv_geom * g, uint32_t cell, uint32_t blk, tsr_kv_loc * loc) {
    addr_check(__func__, g, cell, blk);
    addr_v_raw(g->C, g->LG, cell, blk, loc);
}

static inline void kv_addr(uint32_t kv, const tsr_kv_geom * g, uint32_t cell, uint32_t blk, tsr_kv_loc * loc) {
    if (kv == TSR_KV_K) addr_k_raw(g->E, g->C, cell, blk, loc);
    else                addr_v_raw(g->C, g->LG, cell, blk, loc);
}

static void kv_check(const char * fn, uint32_t kv) {
    if (kv > TSR_KV_V) at_fatal(fn, "kv must be TSR_KV_K or TSR_KV_V", kv, 0);
}

// ggml block bytes: [0, 2) = fp16 d as stored (little-endian), [2, 34) = qs.
void tsr_kv_pack_row(uint32_t kv, const tsr_kv_geom * g, uint8_t * const plane[], uint32_t cell,
                     const tsr_block_q8_0 * row) {
    kv_check(__func__, kv);
    addr_check(__func__, g, cell, 0);
    const uint8_t * src = (const uint8_t *)row;
    for (uint32_t b = 0; b < g->E / 32u; ++b) {
        tsr_kv_loc loc;
        kv_addr(kv, g, cell, b, &loc);
        memcpy(plane[loc.ch] + loc.meta, src + 34u * b, 2);
        memcpy(plane[loc.ch] + loc.quant, src + 34u * b + 2u, 32);
    }
}

void tsr_kv_unpack_row(uint32_t kv, const tsr_kv_geom * g, const uint8_t * const plane[], uint32_t cell,
                       tsr_block_q8_0 * row) {
    kv_check(__func__, kv);
    addr_check(__func__, g, cell, 0);
    uint8_t * dst = (uint8_t *)row;
    for (uint32_t b = 0; b < g->E / 32u; ++b) {
        tsr_kv_loc loc;
        kv_addr(kv, g, cell, b, &loc);
        memcpy(dst + 34u * b, plane[loc.ch] + loc.meta, 2);
        memcpy(dst + 34u * b + 2u, plane[loc.ch] + loc.quant, 32);
    }
}

// Byte range [off, off + size) of the ggml image, block piece by block piece.
static void kv_bytes(const char * fn, uint32_t kv, const tsr_kv_geom * g, uint8_t * const plane[],
                     uint8_t * ggml, size_t off, size_t size, int to_plane) {
    kv_check(fn, kv);
    geom_check(fn, g);
    const uint64_t row_bytes = (uint64_t)(g->E / 32u) * 34u;
    const uint64_t total = row_bytes * tsr_kv_plane_cells(g);
    if ((uint64_t)off > total || (uint64_t)size > total - off) at_fatal(fn, "byte range beyond the plane", off, size);
    uint64_t pos = off;
    const uint64_t end = (uint64_t)off + size;
    while (pos < end) {
        const uint32_t cell = (uint32_t)(pos / row_bytes);
        const uint64_t inrow = pos % row_bytes;
        const uint32_t blk = (uint32_t)(inrow / 34u);
        const uint32_t k0 = (uint32_t)(inrow % 34u);
        const uint32_t kend = (uint32_t)i64min(34, (int64_t)(k0 + (end - pos)));
        tsr_kv_loc loc;
        kv_addr(kv, g, cell, blk, &loc);
        uint8_t * gb = ggml + (pos - (uint64_t)off);       // ggml byte k0 of this block
        for (uint32_t k = k0; k < kend; ++k) {
            uint8_t * dev = plane[loc.ch] + (k < 2u ? loc.meta + k : loc.quant + (k - 2u));
            if (to_plane) *dev = gb[k - k0];
            else          gb[k - k0] = *dev;
        }
        pos += kend - k0;
    }
}

void tsr_kv_set_bytes(uint32_t kv, const tsr_kv_geom * g, uint8_t * const plane[],
                      const void * src, size_t off, size_t size) {
    kv_bytes(__func__, kv, g, plane, (uint8_t *)(uintptr_t)src, off, size, 1);
}
void tsr_kv_get_bytes(uint32_t kv, const tsr_kv_geom * g, const uint8_t * const plane[],
                      void * dst, size_t off, size_t size) {
    kv_bytes(__func__, kv, g, (uint8_t * const *)(uintptr_t)plane, (uint8_t *)dst, off, size, 0);
}

// ---------------------------------------------------------------------------
// KVST (AT §5.1)
// ---------------------------------------------------------------------------
void tsr_kvst_count(uint32_t kv, const tsr_kv_geom * g, uint32_t c0, uint32_t n_rows, tsr_kvst_stats * st) {
    kv_check(__func__, kv);
    geom_check(__func__, g);
    if ((uint64_t)c0 + n_rows > tsr_kv_plane_cells(g)) at_fatal(__func__, "cells beyond the plane", c0, n_rows);
    memset(st, 0, sizeof *st);
    const uint32_t nb = g->E / 32u, slices = g->E / 64u;
    const uint32_t end = c0 + n_rows;
    for (uint32_t cell = c0; cell < end;) {
        const uint32_t u = cell >> 4, ch = u % g->C;
        const uint32_t r_lo = cell & 15u;
        const uint32_t r_hi = (end - 16u * u) < 16u ? end - 16u * u : 16u;   // group cells [r_lo, r_hi)
        const uint32_t k = r_hi - r_lo;
        st->quant_beats[ch] += k * slices;          // E/64 quant beats per row, never shared
        if (kv == TSR_KV_V) {
            st->meta_beats[ch] += slices;           // one meta beat per slice holds the group's 16 cells
            if (k < 16u) st->meta_rmw[ch] += slices;
        } else {
            const uint64_t b_lo = (uint64_t)r_lo * nb, b_hi = (uint64_t)r_hi * nb;   // beta range
            for (uint64_t kap = b_lo >> 5; kap <= (b_hi - 1u) >> 5; ++kap) {
                st->meta_beats[ch] += 1u;
                if (32u * kap < b_lo || 32u * kap + 32u > b_hi) st->meta_rmw[ch] += 1u;
            }
        }
        cell = 16u * u + r_hi;
    }
}

void tsr_ref_kvst(uint32_t kv, const tsr_kv_geom * g, uint8_t * const plane[], uint32_t c0, uint32_t n_rows,
                  const uint8_t * src, uint32_t src_stride, tsr_kvst_stats * st) {
    kv_check(__func__, kv);
    geom_check(__func__, g);
    if (n_rows == 0 || n_rows > TSR_KVST_MAX_ROWS) at_fatal(__func__, "n_rows must be in 1..64", n_rows, 0);
    if ((uint64_t)c0 + n_rows > tsr_kv_plane_cells(g)) at_fatal(__func__, "cells beyond the plane", c0, n_rows);
    const uint32_t E = g->E, nb = E / 32u;
    const size_t stride = (size_t)(src_stride ? src_stride : tsr_qa_lines(TSR_QA_A32, E)) * TSR_LINE_BYTES;
    const size_t d_off = (size_t)tsr_qa_qs_lines(TSR_QA_A32, E) * TSR_LINE_BYTES;
    for (uint32_t i = 0; i < n_rows; ++i) {
        const uint8_t * qa = src + (size_t)i * stride;
        for (uint32_t b = 0; b < nb; ++b) {
            tsr_kv_loc loc;
            kv_addr(kv, g, c0 + i, b, &loc);
            const uint16_t h = tsr_f32_to_fp16(bf(rd_u32le(qa + d_off + 4u * b)));
            wr_u16le(plane[loc.ch] + loc.meta, h);
            memcpy(plane[loc.ch] + loc.quant, qa + 32u * b, 32);
        }
    }
    if (st != NULL) tsr_kvst_count(kv, g, c0, n_rows, st);
}

// ---------------------------------------------------------------------------
// Shape checks shared by ATTQK and ATTPV (AT §5, §5.8)
// ---------------------------------------------------------------------------
static int d_valid(uint32_t D) { return D == 64u || D == 128u || D == 256u || D == 512u; }

static void shape_check(const char * fn, const tsr_att_shape * s, uint32_t C, uint32_t n_groups) {
    if (s == NULL) at_fatal(fn, "shape is NULL", 0, 0);
    if (!d_valid(s->D)) at_fatal(fn, "D must be 64, 128, 256 or 512", s->D, 0);
    if (s->E == 0 || s->E % s->D != 0) at_fatal(fn, "E must be a positive multiple of D", s->E, s->D);
    if (s->G == 0 || s->G > TSR_ATT_MAX_G) at_fatal(fn, "G must be in 1..64", s->G, 0);
    if (s->GS == 0 || s->hg0 + s->GS > s->G) at_fatal(fn, "need 1 <= GS and hg0 + GS <= G", s->hg0, s->GS);
    if (s->T_q == 0 || s->T_q > TSR_T_MAX_ARCH) at_fatal(fn, "T_q must be in 1..8", s->T_q, 0);
    if (s->GS > TSR_T_MAX_ARCH) at_fatal(fn, "GS must be in 1..8", s->GS, 0);
    if (C == 0 || C > TSR_NUM_CH_MAX) at_fatal(fn, "C must be in 1..16", C, 0);
    if (n_groups == 0) at_fatal(fn, "n_groups must be >= 1", n_groups, 0);
}

// ---------------------------------------------------------------------------
// ATTQK (AT §5.2, §6.2, §6.3)
// ---------------------------------------------------------------------------
float tsr_att_maxt(float a, float b) {
    const uint32_t ua = daz_bits(fb(a)), ub = daz_bits(fb(b));
    if (is_nan_bits(ua) || is_nan_bits(ub)) return bf(TSR_CANON_NAN_BITS);
    const uint32_t ka = (ua & F32_SIGN) ? ~ua : (ua | F32_SIGN);
    const uint32_t kb = (ub & F32_SIGN) ? ~ub : (ub | F32_SIGN);
    return bf(ka >= kb ? ua : ub);
}

float tsr_att_seg(uint32_t D, const tsr_block_q8_0 * kseg, const int8_t * qx, const float * dx) {
    if (D == 0 || D % 32u != 0 || D / 32u > TSR_ROW_PARTIALS) at_fatal(__func__, "D/32 must be in 1..16", D, 0);
    float p[TSR_ROW_PARTIALS];
    for (uint32_t k = 0; k < TSR_ROW_PARTIALS; ++k) p[k] = 0.0f;
    for (uint32_t k = 0; k < D / 32u; ++k) {
        int32_t isum = 0;
        for (uint32_t l = 0; l < 32u; ++l) isum += (int32_t)kseg[k].qs[l] * (int32_t)qx[32u * k + l];
        const float t = tsr_fmul(tsr_fmul(tsr_fp16_to_f32(kseg[k].d), dx[k]), tsr_i32_to_f32(isum));
        p[k] = tsr_fadd(0.0f, t);
    }
    return tsr_tree16(p);
}

void tsr_ref_attqk(const tsr_att_shape * sh, uint32_t C, const uint8_t * const kplane[],
                   uint32_t u0, uint32_t n_groups, uint32_t c_lo, uint32_t c_hi, uint32_t flags,
                   const uint8_t * x, float * dst, uint32_t rstride, float * mdst) {
    shape_check(__func__, sh, C, n_groups);
    const uint32_t ncell = 16u * n_groups;
    if (c_lo > c_hi || c_hi > ncell) at_fatal(__func__, "need c_lo <= c_hi <= 16·n_groups", c_lo, c_hi);
    if (flags & ~(TSR_ATT_CAUSAL | TSR_ATT_SLIDE | TSR_ATT_MAXOUT)) at_fatal(__func__, "undefined flag", flags, 0);
    const uint32_t E = sh->E, D = sh->D, nseg = E / D, nbs = D / 32u, nb = E / 32u;
    const uint32_t n_head = tsr_att_n_head(sh), K = n_head * D;
    const uint64_t rs = rstride ? rstride : n_groups;
    const size_t xl = (size_t)tsr_qa_lines(TSR_QA_A32, K) * TSR_LINE_BYTES;
    const size_t x_doff = (size_t)tsr_qa_qs_lines(TSR_QA_A32, K) * TSR_LINE_BYTES;
    const int causal = (flags & TSR_ATT_CAUSAL) != 0, slide = (flags & TSR_ATT_SLIDE) != 0;

    // Query scales (D region of each QA vector) as host floats.
    float * dx = (float *)xmalloc((size_t)sh->T_q * (K / 32u) * sizeof(float));
    for (uint32_t q = 0; q < sh->T_q; ++q) {
        for (uint32_t b = 0; b < K / 32u; ++b) dx[(size_t)q * (K / 32u) + b] = bf(rd_u32le(x + q * xl + x_doff + 4u * b));
    }
    // Row bounds lo_q, hi_q (signed; AT §5.2).
    int64_t lo[TSR_T_MAX_ARCH], hi[TSR_T_MAX_ARCH];
    for (uint32_t q = 0; q < sh->T_q; ++q) {
        const int64_t dq = (int64_t)sh->T_q - 1 - q;
        lo[q] = i64max(0, (int64_t)c_lo - (slide ? dq : 0));
        hi[q] = (int64_t)c_hi - (causal ? dq : 0);
    }
    tsr_block_q8_0 * krow = (tsr_block_q8_0 *)xmalloc((size_t)nb * sizeof(tsr_block_q8_0));
    for (uint32_t j = 0; j < ncell; ++j) {
        int loaded = 0;
        for (uint32_t q = 0; q < sh->T_q; ++q) {
            const int vis = (int64_t)j >= lo[q] && (int64_t)j < hi[q];
            if (vis && !loaded) {        // gather the cell's K row (E/32 blocks) from the plane
                const uint32_t cell = 16u * u0 + j;
                for (uint32_t b = 0; b < nb; ++b) {
                    tsr_kv_loc loc;
                    addr_k_raw(E, C, cell, b, &loc);
                    krow[b].d = rd_u16le(kplane[loc.ch] + loc.meta);
                    memcpy(krow[b].qs, kplane[loc.ch] + loc.quant, 32);
                }
                loaded = 1;
            }
            const int8_t * qs = (const int8_t *)(x + q * xl);
            for (uint32_t g = 0; g < nseg; ++g) {
                for (uint32_t i = 0; i < sh->GS; ++i) {
                    const uint32_t r = tsr_att_row(sh, q, g, i);
                    const uint32_t h = g * sh->G + sh->hg0 + i;
                    dst[(size_t)(r * rs) * 16u + j] =
                        vis ? tsr_att_seg(D, krow + (size_t)g * nbs, qs + (size_t)h * D, dx + (size_t)q * (K / 32u) + (size_t)h * nbs)
                            : bf(F32_NEG_INF);
                }
            }
        }
    }
    if (flags & TSR_ATT_MAXOUT) {
        for (uint32_t q = 0; q < sh->T_q; ++q) {
            for (uint32_t g = 0; g < nseg; ++g) {
                for (uint32_t i = 0; i < sh->GS; ++i) {
                    const uint32_t r = tsr_att_row(sh, q, g, i);
                    const float * row = dst + (size_t)(r * rs) * 16u;
                    float m = bf(TSR_M_FLOOR);
                    for (uint32_t j = 0; j < ncell; ++j) m = tsr_att_maxt(m, row[j]);
                    mdst[r] = m;
                }
            }
        }
    }
    free(krow);
    free(dx);
}

// ---------------------------------------------------------------------------
// ATTPV (AT §5.3, §6.4, §6.5)
// ---------------------------------------------------------------------------
int32_t tsr_att_ilog2(float w) {
    const uint32_t e = (fb(w) >> 23) & 0xFFu;
    if (e == 0u || e == 0xFFu) at_fatal(__func__, "w must be a finite normal", fb(w), 0);
    return (int32_t)e - 127;
}

int32_t tsr_att_bf_int(float w, int32_t e_s) {
    const uint32_t u = fb(w), a = u & ~F32_SIGN, e = a >> 23;
    if (e == 0u) return 0;                                   // zero or subnormal (DAZ)
    if (e == 0xFFu) at_fatal(__func__, "w must be finite", u, 0);
    const int32_t eps = (int32_t)e - 127;
    if (eps > e_s) at_fatal(__func__, "eps(w) > E_S", (uint64_t)(int64_t)eps, (uint64_t)(int64_t)e_s);
    const uint32_t m = 0x00800000u | (a & F32_MANT);         // |w| = m·2^(eps-23); |w|·2^(23-E_S) = m·2^(eps-E_S)
    const uint32_t sh = (uint32_t)(e_s - eps);
    uint32_t v;
    if (sh == 0u) {
        v = m;
    } else if (sh >= 26u) {
        v = 0u;                                              // m < 2^24 < half
    } else {
        const uint32_t rem = m & ((1u << sh) - 1u), half = 1u << (sh - 1u);
        v = m >> sh;
        if (rem > half || (rem == half && (v & 1u))) v += 1u;
    }
    return (u & F32_SIGN) ? -(int32_t)v : (int32_t)v;
}

float tsr_att_bf_f(int64_t A, int32_t k) {
    if (A == 0) return 0.0f;
    const uint32_t sign = A < 0 ? F32_SIGN : 0u;
    const uint64_t mag = A < 0 ? 0u - (uint64_t)A : (uint64_t)A;
    const int msb = 63 - __builtin_clzll(mag);
    uint64_t m;
    int64_t e;                                               // value = m·2^(e-23), m in [2^23, 2^24)
    if (msb > 23) {
        uint32_t sh = (uint32_t)(msb - 23);
        const uint64_t rem = mag & ((1ull << sh) - 1u), half = 1ull << (sh - 1u);
        m = mag >> sh;
        if (rem > half || (rem == half && (m & 1u))) m += 1u;
        if (m == (1ull << 24)) { m >>= 1; sh += 1u; }
        e = 23 + (int64_t)sh + k;
    } else {
        m = mag << (23 - msb);
        e = (int64_t)msb + k;
    }
    if (e < -126) return bf(sign);                           // below 2^-126: signed zero
    if (e > 127) return bf(sign | F32_EXP);                  // >= 2^128: signed inf
    return bf(sign | ((uint32_t)(e + 127) << 23) | ((uint32_t)m & F32_MANT));
}

typedef struct pv_cell {
    float w[2];                // p (*) dV of blocks 2s, 2s+1
    const int8_t * q;          // the slice's 64 quants
} pv_cell;

// One 64-cell segment of slice s for one row (AT §6.4 steps 2-7): c[64] = the segment values.
static void pv_segment(const pv_cell * cells, uint32_t n, float c[64]) {
    int nonfin[2] = { 0, 0 }, any = 0;
    int32_t es = -1000;
    for (uint32_t k = 0; k < n; ++k) {
        for (int e = 0; e < 2; ++e) {
            const uint32_t u = fb(cells[k].w[e]);
            if (nonfinite_bits(u)) nonfin[e] = 1;
            else if ((u & ~F32_SIGN) != 0u) {                // w is normal here (FTZ results)
                const int32_t le = tsr_att_ilog2(cells[k].w[e]);
                if (le > es) es = le;
                any = 1;
            }
        }
    }
    int64_t A[64];
    for (int L = 0; L < 64; ++L) A[L] = 0;
    if (any) {
        for (uint32_t k = 0; k < n; ++k) {
            for (int e = 0; e < 2; ++e) {
                if (nonfin[e]) continue;                     // NaN lanes regardless of the sum
                const int64_t W = tsr_att_bf_int(cells[k].w[e], es);
                if (W == 0) continue;
                const int8_t * q = cells[k].q + 32 * e;
                for (int L = 0; L < 32; ++L) A[32 * e + L] += W * (int64_t)q[L];
            }
        }
    }
    for (int L = 0; L < 64; ++L) {
        if (nonfin[L >> 5]) c[L] = bf(TSR_CANON_NAN_BITS);
        else if (!any)      c[L] = 0.0f;
        else                c[L] = tsr_att_bf_f(A[L], es - 23);
    }
}

// Row-sum value of one segment (AT §6.5) over the n p values of its cells.
static float lsum_segment(const float * p, uint32_t n) {
    int nonfin = 0, any = 0;
    int32_t ep = -1000;
    for (uint32_t k = 0; k < n; ++k) {
        const uint32_t u = daz_bits(fb(p[k]));
        if (nonfinite_bits(u)) nonfin = 1;
        else if ((u & ~F32_SIGN) != 0u) {
            const int32_t le = (int32_t)((u >> 23) & 0xFFu) - 127;
            if (le > ep) ep = le;
            any = 1;
        }
    }
    if (nonfin) return bf(TSR_CANON_NAN_BITS);
    if (!any) return 0.0f;
    int64_t S = 0;
    for (uint32_t k = 0; k < n; ++k) S += tsr_att_bf_int(bf(daz_bits(fb(p[k]))), ep);
    return tsr_att_bf_f(S, ep - 23);
}

void tsr_ref_attpv(const tsr_att_shape * sh, uint32_t C, uint32_t LG, const uint8_t * const vplane[],
                   uint32_t u0, uint32_t n_groups, uint32_t flags,
                   const float * p, uint32_t pstride, float * dst, uint32_t cstride) {
    shape_check(__func__, sh, C, n_groups);
    if (flags & ~TSR_ATT_LSUM) at_fatal(__func__, "undefined flag", flags, 0);
    if (LG == 0 || ((uint64_t)u0 + n_groups - 1u) / C >= LG) at_fatal(__func__, "groups beyond LG", u0 + (uint64_t)n_groups, LG);
    const uint32_t D = sh->D, nseg = sh->E / D, nsl = D / 64u;
    const uint32_t R = tsr_att_R(sh), Rl = ceil16(R);
    const uint64_t ps = pstride ? pstride : n_groups;
    const uint64_t cs = cstride ? cstride : (uint64_t)R * D / 16u;
    const uint32_t ug_end = u0 + n_groups;

    uint32_t * U = (uint32_t *)xmalloc(((size_t)n_groups / C + 1u) * sizeof(uint32_t));
    pv_cell cells[64];
    float c[64], O[64], pseg[64];
    for (uint32_t ch = 0; ch < C; ++ch) {
        // U_c: the op's groups u = ch (mod C), ascending; segments are runs of 4 of them.
        uint32_t nu = 0;
        for (uint32_t u = u0 + (ch + C - u0 % C) % C; u < ug_end; u += C) U[nu++] = u;
        for (uint32_t q = 0; q < sh->T_q; ++q) {
            for (uint32_t g = 0; g < nseg; ++g) {
                for (uint32_t i = 0; i < sh->GS; ++i) {
                    const uint32_t r = tsr_att_row(sh, q, g, i);
                    const float * prow = p + (size_t)(r * ps) * 16u;
                    for (uint32_t sl = 0; sl < nsl; ++sl) {
                        const uint32_t s = g * nsl + sl;     // slice of the cell row
                        for (int L = 0; L < 64; ++L) O[L] = 0.0f;
                        for (uint32_t k0 = 0; k0 < nu; k0 += 4u) {
                            uint32_t n = 0;
                            for (uint32_t k = k0; k < nu && k < k0 + 4u; ++k) {
                                const uint32_t u = U[k], l = u / C;
                                const uint8_t * chunk = vplane[ch] + ((uint64_t)s * LG + l) * 17u * TSR_BEAT_BYTES;
                                for (uint32_t rr = 0; rr < 16u; ++rr) {
                                    const float pj = prow[16u * (u - u0) + rr];
                                    if ((daz_bits(fb(pj)) & ~F32_SIGN) == 0u) continue;   // p = +-0: skipped
                                    cells[n].w[0] = tsr_fmul(pj, tsr_fp16_to_f32(rd_u16le(chunk + 4u * rr)));
                                    cells[n].w[1] = tsr_fmul(pj, tsr_fp16_to_f32(rd_u16le(chunk + 4u * rr + 2u)));
                                    cells[n].q = (const int8_t *)(chunk + (1u + rr) * TSR_BEAT_BYTES);
                                    ++n;
                                }
                            }
                            pv_segment(cells, n, c);
                            for (int L = 0; L < 64; ++L) O[L] = tsr_fadd(O[L], c[L]);
                        }
                        float * orow = dst + (size_t)(ch * cs + (uint64_t)r * D / 16u) * 16u + 64u * sl;
                        for (int L = 0; L < 64; ++L) orow[L] = O[L];
                    }
                    if (flags & TSR_ATT_LSUM) {
                        float l = 0.0f;
                        for (uint32_t k0 = 0; k0 < nu; k0 += 4u) {
                            uint32_t n = 0;
                            for (uint32_t k = k0; k < nu && k < k0 + 4u; ++k) {
                                for (uint32_t rr = 0; rr < 16u; ++rr) pseg[n++] = prow[16u * (U[k] - u0) + rr];
                            }
                            l = tsr_fadd(l, lsum_segment(pseg, n));
                        }
                        dst[(size_t)(C * cs + (uint64_t)ch * Rl) * 16u + r] = l;
                    }
                }
            }
        }
    }
    free(U);
}

// ---------------------------------------------------------------------------
// AT's vector ops (AT §5.4–§5.7)
// ---------------------------------------------------------------------------
void tsr_ref_asmax(float * dst, const float * src, const float * mvec, uint32_t n_lines, uint32_t stride,
                   uint32_t R, float scale) {
    const uint64_t st = stride ? stride : n_lines;
    for (uint32_t r = 0; r < R; ++r) {
        const float m = mvec[r];
        for (uint64_t k = 0; k < 16u * (uint64_t)n_lines; ++k) {
            const size_t ix = (size_t)(r * st * 16u + k);
            dst[ix] = tsr_m_exp(tsr_fmul(tsr_fsub(src[ix], m), scale));
        }
    }
}

void tsr_ref_amask(float * dst, const float * src, const float * mask, uint32_t n_lines, uint32_t stride,
                   uint32_t R, uint32_t rpq, uint32_t mstride, float * mdst) {
    if (rpq == 0) at_fatal(__func__, "rpq must be >= 1", rpq, 0);
    const uint64_t st = stride ? stride : n_lines;
    for (uint32_t r = 0; r < R; ++r) {
        const float * mrow = mask + (size_t)((uint64_t)(r / rpq) * mstride * 16u);
        float mx = bf(TSR_M_FLOOR);
        for (uint64_t k = 0; k < 16u * (uint64_t)n_lines; ++k) {
            const size_t ix = (size_t)(r * st * 16u + k);
            const float m = mrow[k];
            const float v = fb(m) == F32_NEG_INF ? bf(F32_NEG_INF) : tsr_fadd(src[ix], m);
            dst[ix] = v;
            mx = tsr_att_maxt(mx, v);
        }
        mdst[r] = mx;
    }
}

float tsr_att_ctree(const float * x, uint32_t n) {
    if (n == 0 || n > TSR_NUM_CH_MAX) at_fatal(__func__, "n must be in 1..16", n, 0);
    float v[TSR_NUM_CH_MAX];
    for (uint32_t i = 0; i < n; ++i) v[i] = x[i];
    while (n > 1u) {
        for (uint32_t k = 0; 2u * k + 1u < n; ++k) v[k] = tsr_fadd(v[2u * k], v[2u * k + 1u]);
        if (n & 1u) v[n / 2u] = v[n - 1u];
        n = (n + 1u) / 2u;
    }
    return v[0];
}

void tsr_ref_amerge(float * orun, const float * parts, const float * mk, float * st, uint32_t R, uint32_t D,
                    float scale, uint32_t C, uint32_t cstride, uint32_t flags) {
    if (C == 0 || C > TSR_NUM_CH_MAX) at_fatal(__func__, "C must be in 1..16", C, 0);
    if (!d_valid(D)) at_fatal(__func__, "D must be 64, 128, 256 or 512", D, 0);
    if (flags & ~TSR_ATT_INIT) at_fatal(__func__, "undefined flag", flags, 0);
    const uint32_t Rl = ceil16(R);
    const uint64_t cs = cstride ? cstride : (uint64_t)R * D / 16u;
    float * mrun = st;
    float * lrun = st + (size_t)Rl * 16u;
    const float * lparts = parts + (size_t)(C * cs) * 16u;
    float xv[TSR_NUM_CH_MAX];
    for (uint32_t r = 0; r < R; ++r) {
        for (uint32_t c = 0; c < C; ++c) xv[c] = lparts[(size_t)c * Rl * 16u + r];
        const float lk = tsr_att_ctree(xv, C);
        float a = 0.0f, b = 0.0f;
        const int init = (flags & TSR_ATT_INIT) != 0;
        if (!init) {
            const float mn = tsr_att_maxt(mrun[r], mk[r]);
            a = tsr_m_exp(tsr_fmul(tsr_fsub(mrun[r], mn), scale));
            b = tsr_m_exp(tsr_fmul(tsr_fsub(mk[r], mn), scale));
            lrun[r] = tsr_fadd(tsr_fmul(lrun[r], a), tsr_fmul(lk, b));
            mrun[r] = mn;
        } else {
            lrun[r] = lk;
            mrun[r] = mk[r];
        }
        for (uint32_t d = 0; d < D; ++d) {
            for (uint32_t c = 0; c < C; ++c) xv[c] = parts[(size_t)(c * cs) * 16u + (size_t)r * D + d];
            const float ok = tsr_att_ctree(xv, C);
            float * o = orun + (size_t)r * D + d;
            *o = init ? ok : tsr_fadd(tsr_fmul(*o, a), tsr_fmul(ok, b));
        }
    }
}

void tsr_ref_anorm(float * out, const float * orun, const float * lrun, uint32_t R, uint32_t D) {
    if (!d_valid(D)) at_fatal(__func__, "D must be 64, 128, 256 or 512", D, 0);
    for (uint32_t r = 0; r < R; ++r) {
        const float rinv = tsr_frecip(tsr_att_maxt(lrun[r], bf(TSR_ANORM_L_FLOOR)));
        for (uint32_t d = 0; d < D; ++d) out[(size_t)r * D + d] = tsr_fmul(orun[(size_t)r * D + d], rinv);
    }
}

// ---------------------------------------------------------------------------
// Mask summary (AT §7.3)
// ---------------------------------------------------------------------------
uint32_t tsr_msum_build_f16(const uint16_t * mask, size_t row_stride, uint32_t n_kv, uint32_t n_q,
                            uint32_t f32_handle, uint32_t * words) {
    uint32_t general = 0;
    uint32_t * rows = words + TSR_MSUM_HDR_WORDS;
    for (uint32_t q = 0; q < n_q; ++q) {
        const uint16_t * m = mask + (size_t)q * row_stride;
        uint32_t nrun = 0, run[2][2] = { { 0, 0 }, { 0, 0 } };
        for (uint32_t j = 0; j < n_kv;) {
            const uint16_t h = m[j];
            if (h == 0xFC00u) { ++j; continue; }                  // -inf
            if ((h & 0x7FFFu) != 0u) {                             // finite nonzero, +inf or NaN
                general = 1;
                ++j;
                continue;
            }
            uint32_t e = j;                                        // a run of +-0
            while (e < n_kv && (m[e] & 0x7FFFu) == 0u) ++e;
            if (nrun < 2u) { run[nrun][0] = j; run[nrun][1] = e; }
            ++nrun;
            j = e;
        }
        if (nrun > 2u) general = 1;
        uint32_t * w = rows + (size_t)TSR_MSUM_ROW_WORDS * q;
        w[0] = run[0][0]; w[1] = run[0][1]; w[2] = run[1][0]; w[3] = run[1][1];
    }
    if (general) {   // rows hold the hull of the row's non-(-inf) cells as run 0 (informative)
        for (uint32_t q = 0; q < n_q; ++q) {
            const uint16_t * m = mask + (size_t)q * row_stride;
            uint32_t lo = 0, hi = 0;
            int found = 0;
            for (uint32_t j = 0; j < n_kv; ++j) {
                if (m[j] == 0xFC00u) continue;
                if (!found) { lo = j; found = 1; }
                hi = j + 1u;
            }
            uint32_t * w = rows + (size_t)TSR_MSUM_ROW_WORDS * q;
            w[0] = lo; w[1] = hi; w[2] = 0; w[3] = 0;
        }
    }
    words[0] = TSR_MSUM_MAGIC;
    words[1] = n_q;
    words[2] = n_kv;
    words[3] = general ? TSR_MSUM_GENERAL : 0u;
    words[4] = general ? f32_handle : 0u;
    words[5] = words[6] = words[7] = 0u;
    return words[3];
}

int tsr_msum_view(const uint32_t * words, size_t n_words, tsr_msum * out) {
    if (n_words < TSR_MSUM_HDR_WORDS || words[0] != TSR_MSUM_MAGIC) return -1;
    if (n_words < tsr_msum_words(words[1])) return -1;
    out->n_q = words[1];
    out->n_kv = words[2];
    out->flags = words[3];
    out->rows = (const tsr_msum_row *)(const void *)(words + TSR_MSUM_HDR_WORDS);
    return 0;
}

void tsr_msum_mask_f32(const uint16_t * mask, size_t row_stride, uint32_t n_kv, uint32_t n_q, float scale,
                       float * dst) {
    const uint32_t sb = fb(scale), e = (sb >> 23) & 0xFFu;
    if ((sb & F32_SIGN) || e == 0u || e == 0xFFu) at_fatal(__func__, "scale must be finite, normal and > 0", sb, 0);
    const size_t w = 16u * (size_t)ceil16(n_kv);
    for (uint32_t q = 0; q < n_q; ++q) {
        const uint16_t * m = mask + (size_t)q * row_stride;
        float * d = dst + (size_t)q * w;
        for (uint32_t j = 0; j < n_kv; ++j) d[j] = tsr_fdiv(tsr_fp16_to_f32(m[j]), scale);
        for (size_t j = n_kv; j < w; ++j) d[j] = bf(F32_NEG_INF);
    }
}

// ---------------------------------------------------------------------------
// Chunk planner (AT §6.6 step 3, §7.3, AT-19)
//
// the planner's rules and code live in include/tsr_fa_plan.h, shared with the
// firmware's FLASH_ATTN kernel (sw/ctrl/kernels/k_fattn.cpp), so the two chunk lists are the same
// by construction. Rules 1-8 are stated there; these are the library's wrappers
// with its signatures and return codes. sw/ctrl/tests/test_fw_kvfa_plan.cpp compares the shared
// planner with the original code (kept there as the oracle).
// ---------------------------------------------------------------------------
void tsr_fa_pass_rule(const tsr_att_cfg * cfg, uint32_t G, uint32_t * GS, uint32_t * T_q) {
    tsr_fap_pass_rule(cfg->t_max, G, GS, T_q);
}

uint32_t tsr_fa_ncg(const tsr_att_cfg * cfg, uint32_t E, uint32_t D, uint32_t n_total) {
    return tsr_fap_ncg(cfg->C, cfg->pcap, E, D, n_total);
}

typedef struct plan_new {
    const uint32_t (*new_runs)[2];
    uint32_t n_new;
} plan_new;

// Rule 7: the chunk's groups hold a cell of one of the runs written in this launch.
static int chunk_has_new(void * ctx, uint32_t u0, uint32_t n_groups) {
    const plan_new * pn = (const plan_new *)ctx;
    const uint64_t a = 16u * (uint64_t)u0, b = 16u * ((uint64_t)u0 + n_groups);
    for (uint32_t i = 0; i < pn->n_new; ++i) {
        if ((uint64_t)pn->new_runs[i][0] < b && (uint64_t)pn->new_runs[i][1] > a && pn->new_runs[i][1] > pn->new_runs[i][0]) return 1;
    }
    return 0;
}

long tsr_fa_plan(const tsr_att_cfg * cfg, uint32_t E, uint32_t D, const tsr_msum * ms, uint32_t q0, uint32_t T_q,
                 const uint32_t (*new_runs)[2], uint32_t n_new, tsr_fa_chunk * out, size_t cap) {
    if (T_q == 0 || T_q > TSR_T_MAX_ARCH || (uint64_t)q0 + T_q > ms->n_q) return -2;
    tsr_fap_tile t;
    const int rc = tsr_fap_tile_init(&t, cfg->C, cfg->pcap, E, D, ms->n_kv, ms->flags,
                                     (ms->flags & TSR_MSUM_GENERAL) ? NULL : &ms->rows[q0].lo0, T_q);
    if (rc != 0) return rc;
    plan_new pn = { new_runs, new_runs ? n_new : 0u };
    tsr_fap_iter it;
    tsr_fap_iter_init(&it);
    size_t count = 0;
    for (;;) {
        tsr_fap_chunk c;
        const int r = tsr_fap_next(&t, &it, chunk_has_new, &pn, &c);
        if (r < 0) return -3;
        if (r == 0) break;
        if (count < cap) out[count] = (tsr_fa_chunk){ c.u0, c.n_groups, c.c_lo, c.c_hi, c.flags, c.general };
        count += 1u;
    }
    return (long)count;
}

// ---------------------------------------------------------------------------
// Composite flash attention (AT §6.6)
// ---------------------------------------------------------------------------
int tsr_ref_fa(const tsr_att_cfg * cfg, const tsr_fa_args * a) {
    const uint32_t C = cfg->C, D = a->D, E = a->E, G = a->G;
    if (C == 0 || C > TSR_NUM_CH_MAX || cfg->t_max == 0 || cfg->t_max > TSR_T_MAX_ARCH) return TSR_FA_E_SHAPE;
    if (!d_valid(D) || E == 0 || E % D != 0 || E / 64u > 511u || G == 0 || G > TSR_ATT_MAX_G || a->T == 0) return TSR_FA_E_SHAPE;
    if (a->LG == 0 || a->msum == NULL || a->msum->n_q != a->T) return TSR_FA_E_SHAPE;
    if ((uint64_t)a->msum->n_kv > 16u * (uint64_t)C * a->LG) return TSR_FA_E_SHAPE;
    const uint32_t sb = fb(a->scale);
    if ((sb & F32_SIGN) || (sb & F32_EXP) == 0u || (sb & F32_EXP) == F32_EXP) return TSR_FA_E_SHAPE;   // ASMAX: finite, normal, > 0
    const uint32_t nkv_h = E / D, n_head = G * nkv_h, K = n_head * D;
    if (cfg->k_max != 0u && (uint64_t)G * E > cfg->k_max) return TSR_FA_E_SHAPE;
    uint32_t GS0, Tq0;
    tsr_fa_pass_rule(cfg, G, &GS0, &Tq0);
    if (nkv_h * GS0 * Tq0 > TSR_ATT_MAX_PASS_ROWS || Tq0 * n_head > TSR_ATT_MAX_R) return TSR_FA_E_SHAPE;
    if (tsr_fa_ncg(cfg, E, D, 1u) == 0u) return TSR_FA_E_SHAPE;

    const uint32_t R0 = Tq0 * n_head, Rl0 = ceil16(R0);
    const size_t xl = (size_t)tsr_qa_lines(TSR_QA_A32, K) * TSR_LINE_BYTES;
    const uint32_t mstride = ceil16(a->msum->n_kv);
    const uint64_t cs0 = (uint64_t)R0 * D / 16u;
    uint8_t * X = (uint8_t *)calloc((size_t)Tq0 * xl, 1);
    float * S = (float *)calloc((size_t)R0 * 128u * 16u, sizeof(float));
    float * MK = (float *)calloc((size_t)Rl0 * 16u, sizeof(float));
    float * P = (float *)calloc((size_t)(C * cs0 + (uint64_t)C * Rl0) * 16u, sizeof(float));
    float * ORUN = (float *)calloc((size_t)R0 * D, sizeof(float));
    float * ST = (float *)calloc((size_t)2u * Rl0 * 16u, sizeof(float));
    tsr_fa_chunk * ch = NULL;
    int rc = TSR_FA_OK;
    if (!X || !S || !MK || !P || !ORUN || !ST) { rc = TSR_FA_E_NOMEM; goto done; }
    if (a->stats) memset(a->stats, 0, sizeof *a->stats);

    for (uint32_t q0 = 0; q0 < a->T; q0 += Tq0) {
        const uint32_t Tq = a->T - q0 < Tq0 ? a->T - q0 : Tq0;
        const uint32_t R = Tq * n_head, Rl = ceil16(R);
        const uint64_t cs = (uint64_t)R * D / 16u;
        // 1. Q: QACT A32 of each token's Q row (K = n_head·D); the scale is not applied to Q.
        for (uint32_t q = 0; q < Tq; ++q) tsr_qact(TSR_QA_A32, a->q + (size_t)(q0 + q) * K, K, X + q * xl);
        // 3. Chunks.
        const long nch = tsr_fa_plan(cfg, E, D, a->msum, q0, Tq, a->new_runs, a->n_new, NULL, 0);
        if (nch < 0) { rc = nch == -2 ? TSR_FA_E_MSUM : TSR_FA_E_SHAPE; goto done; }
        free(ch);
        ch = (tsr_fa_chunk *)malloc((size_t)nch * sizeof *ch);
        if (ch == NULL) { rc = TSR_FA_E_NOMEM; goto done; }
        tsr_fa_plan(cfg, E, D, a->msum, q0, Tq, a->new_runs, a->n_new, ch, (size_t)nch);
        for (long k = 0; k < nch; ++k) {
            if (ch[k].general && a->mask_f32 == NULL) { rc = TSR_FA_E_MASK; goto done; }
        }
        if (a->stats) { a->stats->tiles += 1u; a->stats->ops_qact += Tq; }
        // 4. Per chunk.
        for (long k = 0; k < nch; ++k) {
            const tsr_fa_chunk * c = &ch[k];
            const uint32_t n = c->n_groups;
            for (uint32_t hg0 = 0; hg0 < G; hg0 += GS0) {
                const tsr_att_shape sh = { E, D, G, hg0, G - hg0 < GS0 ? G - hg0 : GS0, Tq };
                if (c->general) tsr_ref_attqk(&sh, C, a->kplane, c->u0, n, 0, 16u * n, 0, X, S, 0, NULL);
                else tsr_ref_attqk(&sh, C, a->kplane, c->u0, n, c->c_lo, c->c_hi, c->flags | TSR_ATT_MAXOUT, X, S, 0, MK);
                if (a->stats) a->stats->ops_attqk += 1u;
            }
            if (c->general) {
                tsr_ref_amask(S, S, a->mask_f32 + ((size_t)q0 * mstride + c->u0) * 16u, n, 0, R, n_head, mstride, MK);
                if (a->stats) a->stats->ops_amask += 1u;
            }
            tsr_ref_asmax(S, S, MK, n, 0, R, a->scale);
            for (uint32_t hg0 = 0; hg0 < G; hg0 += GS0) {
                const tsr_att_shape sh = { E, D, G, hg0, G - hg0 < GS0 ? G - hg0 : GS0, Tq };
                tsr_ref_attpv(&sh, C, a->LG, a->vplane, c->u0, n, TSR_ATT_LSUM, S, 0, P, (uint32_t)cs);
                if (a->stats) a->stats->ops_attpv += 1u;
            }
            tsr_ref_amerge(ORUN, P, MK, ST, R, D, a->scale, C, (uint32_t)cs, k == 0 ? TSR_ATT_INIT : 0u);
            if (a->stats) {
                a->stats->ops_asmax += 1u;
                a->stats->ops_amerge += 1u;
                a->stats->chunks += 1u;
                a->stats->general_chunks += c->general ? 1u : 0u;
            }
        }
        // 5. ANORM writes the tile's FA output rows [token][head][D].
        tsr_ref_anorm(a->out + (size_t)q0 * n_head * D, ORUN, ST + (size_t)Rl * 16u, R, D);
        if (a->stats) {
            a->stats->ops_anorm += 1u;
            a->stats->passes += (G + GS0 - 1u) / GS0;
        }
    }
done:
    free(ch);
    free(X); free(S); free(MK); free(P); free(ORUN); free(ST);
    return rc;
}
