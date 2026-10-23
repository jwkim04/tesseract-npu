// tsr-trace-gen — random but valid M1 programs (LD / QACT / MVM / ST with LI / ADDI) with
// random weights, executed on the ISA simulator and written as trace directories for RTL
// regression (spec 검증 전략: "RTL 코어 전체 — 트레이스 재생 비트 일치").
//
//   tsr-trace-gen -o DIR [-n COUNT] [-s SEED] [--prefix NAME] [--ops m1|m2|m3] [--t-max T]
//                 [--num-ch C] [--spm-kib S] [--ch-gib G] [--addr-beat-w W] [--k-max K]
//                 [--max-k K] [--max-n N] [--max-groups G] [--errors PCT] [--no-special] [-q]
//                 [--max-wbeats B] [--big-pct PCT]
//
// --ops m2 generates programs for the M2 RTL core instead: only NOP END LI ADDI LD ST QACT
// FENCE (see "M2 programs" below); --ops m3 generates MVM-centred programs for the M3 RTL core
// (see "M3 programs" below; --max-wbeats / --big-pct bound its weight traffic); --ops m1
// (default) is the M1 generator described here. The m1 and m2 outputs do not depend on the m3
// additions (same seed -> same files as before them).
//
// Trace i is written to DIR/<prefix>_s<SEED>_<i, 4 digits>/ and depends only on (SEED, i) and
// the options. Coverage per program: both weight formats, K with partial chunks, N < C, odd N,
// N not a multiple of C, T in [1, T_MAX], ACC (from loaded old values and chained MVMs sharing
// one QA), dst_stride / src_stride 0 and explicit (with gaps), register-relative addressing
// for every address field (bases set with LI / ADDI and the ARG registers R8..R15), per-token
// and whole-span LD/ST on random channels, NOP / FENCE, and (unless --no-special) activation
// blocks with zeros, ties, subnormals, Inf and NaN, plus fp16 subnormal / zero weight scales.
// Programs never read SPM they did not write (spec: SPM is undefined at start).
// --errors PCT ends PCT% of the programs with an instruction that faults (each M1-reachable
// error code except ABORTED), instead of END.
//
// Before writing, every stored output is compared with an independent host computation from
// the generated data via tsr_ref (tsr_qact, tsr_dot_row, tsr_fadd); a mismatch aborts with
// exit code 1 (simulator or generator bug).

#define _GNU_SOURCE

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "tsr_isa.h"
#include "tsr_ref.h"
#include "tsr_sim.h"
#include "tsr_trace.h"

// ---------------------------------------------------------------------------
// RNG (splitmix64)
// ---------------------------------------------------------------------------
typedef struct { uint64_t s; } rng_t;
static uint64_t rng_u64(rng_t * r) {
    uint64_t z = (r->s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
static uint32_t rng_u32(rng_t * r) { return (uint32_t)(rng_u64(r) >> 32); }
static uint32_t rng_below(rng_t * r, uint32_t n) { return n ? (uint32_t)(((rng_u64(r) >> 32) * (uint64_t)n) >> 32) : 0; }
static double rng_unit(rng_t * r) { return (double)(rng_u64(r) >> 11) * 0x1.0p-53; }
static double rng_normal(rng_t * r) {
    double u1 = rng_unit(r), u2 = rng_unit(r);
    if (u1 < 1e-300) u1 = 1e-300;
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}
static void rng_bytes(rng_t * r, void * dst, size_t n) {
    uint8_t * p = (uint8_t *)dst;
    for (size_t i = 0; i < n; ++i) p[i] = (uint8_t)rng_u32(r);
}

static float u2f(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static uint32_t f2u(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static void put_f32(uint8_t * d, float f) {
    const uint32_t u = f2u(f);
    d[0] = (uint8_t)u; d[1] = (uint8_t)(u >> 8); d[2] = (uint8_t)(u >> 16); d[3] = (uint8_t)(u >> 24);
}
static float get_f32(const uint8_t * p) {
    return u2f((uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24));
}
static void * xmalloc(size_t n) {
    void * p = malloc(n ? n : 1);
    if (!p) { fprintf(stderr, "tsr-trace-gen: out of memory\n"); exit(2); }
    return p;
}

// ---------------------------------------------------------------------------
// Options and state
// ---------------------------------------------------------------------------
typedef struct {
    tsr_config cfg;
    uint32_t   max_k, max_n, max_groups;
    int        m2;            // --ops m2
    int        m3;            // --ops m3
    uint32_t   max_wbeats;    // --max-wbeats (m3): MVM weight beats per trace, busiest channel
    unsigned   big_pct;       // --big-pct (m3): every (100/PCT)-th trace gets a 4x weight budget
    int        special;
    unsigned   err_pct;
    int        quiet;
} opts_t;

typedef struct {
    uint32_t  ch;
    uint64_t  beat;
    uint32_t  lines;
    uint8_t * want;
} check_t;

enum { COV_Q8, COV_Q4, COV_PARTIAL, COV_NLTC, COV_ODDN, COV_NMODC, COV_TGT1, COV_ACC, COV_CHAIN, COV_XSTRIDE,
       COV_YSTRIDE, COV_ZSTRIDE, COV_REGREL, COV_SPECIAL, COV_ERR, COV_N };
#define COV_MAX 96     // coverage slots (M1 uses COV_N, M2 uses C2_N, M3 uses C2_N + C3_N, + error-code counts)
#define COV_ERRC 80    // cov[COV_ERRC + code]: traces ending in error `code`
static const char * cov_names[COV_N] = { "Q8_0 MVMs", "Q4_K MVMs", "partial-chunk K", "N < C", "odd N",
                                         "N % C != 0", "T > 1", "ACC (loaded)", "ACC chains", "src_stride gaps",
                                         "dst_stride gaps", "stride-0 encodings", "reg-relative fields",
                                         "special activations", "error endings" };

// M2 coverage slots (--ops m2).
enum { C2_A32, C2_A256, C2_TGT1, C2_XSTRIDE, C2_ZSTRIDE, C2_KMIN, C2_KMAX, C2_QPAIR, C2_COPY, C2_ZERO,
       C2_X4K, C2_DLIM, C2_SLIM, C2_REGREL, C2_R0W, C2_SPECIAL, C2_ERR, C2_N };
static const char * cov2_names[C2_N] = { "QACT A32", "QACT A256", "T > 1", "src_stride gaps",
                                         "stride-0 encodings", "K = block size", "K = K_MAX", "QACT pairs",
                                         "LD/ST copies", "zero-length LD/ST", "4 KiB-crossing LD/ST",
                                         "operands ending at the DRAM limit", "operands ending at the SPM end",
                                         "reg-relative fields", "R0 writes", "special activations",
                                         "error endings" };

typedef struct {
    const opts_t * o;
    tsr_sim *  sim;
    rng_t      r;
    uint32_t   R[16];
    tsr_insn * p;
    uint32_t   n, cap;
    uint32_t   spm_next, spm_lines;
    uint64_t   io_next[TSR_TRACE_MAX_CH];
    uint64_t   heap_next;
    uint64_t   lim;
    check_t *  chk;
    size_t     n_chk, cap_chk;
    unsigned long cov[COV_MAX];
    // M2 only
    uint32_t   spm_hi;                       // SPM lines [spm_hi, spm_lines) used from the top
    uint64_t   top_next[TSR_TRACE_MAX_CH];   // DRAM beats [top_next, lim) used from the channel end
    int        reserve_ch0_top;              // keep the end of ch0 free (fetch-range tail)
    int        last_base[3];                 // base registers emit() chose for the address fields
    // M3 only
    uint64_t   io_lim;                       // IO allocations stay below (= w_lo)
    uint64_t   w_lo;                         // weight heap start (WEIGHT region, or mid-channel if beyond the limit)
    int        wlim_pending;                 // next non-empty tensor ends exactly at the DRAM limit
    uint64_t   wbudget;                      // weight beats allowed (busiest channel, sum over MVMs)
    uint64_t   wch0, wall;                   // weight beats used: busiest channel / all channels
    uint32_t   n_mvm;                        // MVMs executed (before any error tail)
} gen_t;

static void push(gen_t * g, tsr_insn i) {
    if (g->n == g->cap) {
        g->cap = g->cap ? 2 * g->cap : 64;
        g->p = (tsr_insn *)realloc(g->p, g->cap * sizeof(tsr_insn));
        if (!g->p) { fprintf(stderr, "tsr-trace-gen: out of memory\n"); exit(2); }
    }
    g->p[g->n++] = i;
}

// Emits insn whose first n_addr words hold absolute addresses; each is rewritten as a random
// base register + immediate with probability 1/2. Sometimes follows it with NOP / FENCE.
static void emit(gen_t * g, tsr_insn i, int n_addr) {
    uint32_t b[3] = { 0, 0, 0 };
    for (int f = 0; f < n_addr; ++f) {
        if (rng_below(&g->r, 2)) {
            b[f] = rng_below(&g->r, 16);
            i.w[1 + f] -= g->R[b[f]];
            if (b[f]) g->cov[(g->o->m2 || g->o->m3) ? C2_REGREL : COV_REGREL]++;
        }
    }
    tsr_insn_set_base(&i, b[0], b[1], b[2]);
    for (int f = 0; f < 3; ++f) g->last_base[f] = (int)b[f];
    push(g, i);
    const uint32_t k = rng_below(&g->r, 24);
    if (k == 0) push(g, tsr_enc_nop());
    if (k == 1) push(g, tsr_enc_fence());
}

// Random register updates (LI / ADDI), tracked in the model.
static void mutate_regs(gen_t * g) {
    const uint32_t n = rng_below(&g->r, 3);
    for (uint32_t k = 0; k < n; ++k) {
        const uint32_t rd = 1 + rng_below(&g->r, 7);
        if (rng_below(&g->r, 2)) {
            const uint32_t v = rng_below(&g->r, 2) ? rng_u32(&g->r) : rng_below(&g->r, 1u << 24);
            push(g, tsr_enc_li(rd, v));
            g->R[rd] = v;
        } else {
            const uint32_t ra = rng_below(&g->r, 16), imm = rng_u32(&g->r);
            push(g, tsr_enc_addi(rd, ra, imm));
            g->R[rd] = g->R[ra] + imm;
        }
    }
}

static void add_check(gen_t * g, uint32_t ch, uint64_t beat, uint32_t lines, uint8_t * want) {
    if (g->n_chk == g->cap_chk) {
        g->cap_chk = g->cap_chk ? 2 * g->cap_chk : 16;
        g->chk = (check_t *)realloc(g->chk, g->cap_chk * sizeof(check_t));
        if (!g->chk) { fprintf(stderr, "tsr-trace-gen: out of memory\n"); exit(2); }
    }
    g->chk[g->n_chk].ch = ch;
    g->chk[g->n_chk].beat = beat;
    g->chk[g->n_chk].lines = lines;
    g->chk[g->n_chk].want = want;
    g->n_chk++;
}

static uint16_t rand_fp16_scale(rng_t * r, int special) {
    const uint32_t k = rng_below(r, 64);
    const uint16_t sign = (uint16_t)((rng_u32(r) & 1u) << 15);
    if (special && k == 0) return sign;                                           // +-0
    if (special && k == 1) return (uint16_t)(sign | (1 + rng_below(r, 0x3FF)));   // subnormal
    const uint16_t e = (uint16_t)(1 + rng_below(r, 13));
    return (uint16_t)(sign | (e << 10) | rng_below(r, 1024));
}

static void gen_row_blocks(rng_t * r, uint32_t fmt, uint32_t K, uint8_t * out, int special) {
    const uint32_t nb = K / tsr_fmt_block_size(fmt);
    const size_t bsz = tsr_fmt_ggml_block_bytes(fmt);
    for (uint32_t b = 0; b < nb; ++b) {
        uint8_t * blk = out + (size_t)b * bsz;
        rng_bytes(r, blk, bsz);
        const uint16_t d = rand_fp16_scale(r, special);
        blk[0] = (uint8_t)d; blk[1] = (uint8_t)(d >> 8);
        if (fmt == TSR_FMT_Q4_K) {
            const uint16_t dm = rand_fp16_scale(r, special);
            blk[2] = (uint8_t)dm; blk[3] = (uint8_t)(dm >> 8);
        }
    }
}

static int gen_x(rng_t * r, float * x, uint32_t K, int special) {
    const double scale = ldexp(1.0, (int)rng_below(r, 16) - 8);
    for (uint32_t i = 0; i < K; ++i) x[i] = (float)(rng_normal(r) * scale);
    if (!special || rng_below(r, 3) != 0) return 0;
    for (uint32_t b = 0; b < K / 32; ++b) {
        float * xb = x + 32 * b;
        switch (rng_below(r, 16)) {
        case 0: memset(xb, 0, 32 * sizeof(float)); break;
        case 1: xb[rng_below(r, 32)] = u2f(0x00012345u | (rng_u32(r) & 0x80000000u)); break;
        case 2: { const float m = xb[0]; xb[5] = -m; xb[9] = m; } break;
        case 3: if (rng_below(r, 4) == 0) xb[rng_below(r, 32)] = (rng_u32(r) & 1) ? INFINITY : -INFINITY; break;
        case 4: if (rng_below(r, 4) == 0) xb[rng_below(r, 32)] = u2f(0x7FC00000u | rng_below(r, 1u << 22)); break;
        case 5: for (int i = 0; i < 32; ++i) xb[i] = u2f(0x00000100u * (uint32_t)(i + 1)); break;
        case 6: for (int i = 0; i < 32; ++i) xb[i] = (float)((int)rng_below(r, 255) - 127) * 0.5f; break;  // .5 ties
        default: break;
        }
    }
    return 1;
}

static uint32_t rand_K(rng_t * r, uint32_t fmt, uint32_t kmax) {
    const uint32_t bs = tsr_fmt_block_size(fmt);
    const uint32_t nmax = kmax / bs;
    switch (rng_below(r, 3)) {
    case 0: if (kmax >= 1024) return 1024 * (1 + rng_below(r, kmax / 1024)); /* fallthrough */
    case 1: return bs * (1 + rng_below(r, nmax < 8 ? nmax : 8));
    default: return bs * (1 + rng_below(r, nmax));
    }
}

// One LD -> QACT -> MVM (-> MVM ACC) -> ST group. Returns 0 if nothing fits any more.
static int gen_group(gen_t * g) {
    const tsr_config * c = &g->o->cfg;
    rng_t * r = &g->r;
    const uint32_t C = c->num_ch;
    const uint32_t kmax = g->o->max_k < c->k_max ? g->o->max_k : c->k_max;
    uint32_t fmt = 0, K = 0, N = 0, T = 0, xs = 0, ys = 0, xl = 0, ql = 0, yl = 0, rb = 0, R = 0;
    int acc = 0, chain = 0, fits = 0;
    for (int attempt = 0; attempt < 32 && !fits; ++attempt) {
        fmt = (kmax >= 256) ? rng_below(r, 2) : TSR_FMT_Q8_0;
        K = rand_K(r, fmt, kmax);
        switch (rng_below(r, 8)) {
        case 0: N = C > 1 ? 1 + rng_below(r, C - 1) : 1; break;    // N < C: some channels empty
        case 1: N = C + rng_below(r, C + 1); break;                 // C <= N <= 2C
        default: N = 1 + rng_below(r, g->o->max_n); break;
        }
        T = 1 + rng_below(r, c->t_max);
        acc = rng_below(r, 3) == 0;
        chain = rng_below(r, 4) == 0;
        xl = K / 16;
        ql = tsr_qa_lines(tsr_qa_kind_for_fmt(fmt), K);
        yl = tsr_ceil_div_u32(4 * N, 64);
        xs = xl + (rng_below(r, 3) == 0 ? 1 + rng_below(r, 3) : 0);
        ys = yl + (rng_below(r, 3) == 0 ? 1 + rng_below(r, 3) : 0);
        rb = tsr_row_beats(fmt, K);
        R = tsr_ceil_div_u32(N, C);
        const uint64_t spm_need = (uint64_t)(T - 1) * xs + xl + (uint64_t)T * ql + (uint64_t)(T - 1) * ys + yl + 8;
        const uint64_t w_need = (uint64_t)R * rb * 2 + 64;
        fits = g->spm_next + spm_need <= g->spm_lines && g->heap_next + w_need <= g->lim;
    }
    if (!fits) return 0;
    const uint32_t kind = tsr_qa_kind_for_fmt(fmt);
    g->cov[fmt ? COV_Q4 : COV_Q8] += 1 + chain;
    g->cov[COV_PARTIAL] += (K % 1024) != 0;
    g->cov[COV_NLTC] += N < C;
    g->cov[COV_ODDN] += N & 1;
    g->cov[COV_NMODC] += (N % C) != 0;
    g->cov[COV_TGT1] += T > 1;
    g->cov[COV_ACC] += acc;
    g->cov[COV_CHAIN] += chain;
    g->cov[COV_XSTRIDE] += xs != xl;
    g->cov[COV_YSTRIDE] += ys != yl;

    // Weights (one or two tensors), symmetric allocation in the WEIGHT region.
    const size_t rbytes = (size_t)(K / tsr_fmt_block_size(fmt)) * tsr_fmt_ggml_block_bytes(fmt);
    uint8_t * blocks[2] = { NULL, NULL };
    uint64_t w_off[2] = { 0, 0 };
    uint8_t * row = (uint8_t *)xmalloc((size_t)rb * 64);
    for (int w = 0; w < 1 + chain; ++w) {
        w_off[w] = g->heap_next + rng_below(r, 16);
        g->heap_next = w_off[w] + (uint64_t)R * rb + rng_below(r, 16);
        blocks[w] = (uint8_t *)xmalloc(rbytes * N);
        for (uint32_t n = 0; n < N; ++n) {
            gen_row_blocks(r, fmt, K, blocks[w] + rbytes * n, g->o->special);
            uint32_t ch;
            uint64_t off;
            tsr_row_location(fmt, N, K, C, n, &ch, &off);
            tsr_repack_row(fmt, blocks[w] + rbytes * n, K, tsr_sim_dram(g->sim, ch) + (w_off[w] + off) * 64);
        }
    }

    // Activations in some channel's IO region.
    float * x = (float *)xmalloc(sizeof(float) * K * T);
    int sp = 0;
    for (uint32_t t = 0; t < T; ++t) sp |= gen_x(r, x + (size_t)t * K, K, g->o->special);
    g->cov[COV_SPECIAL] += sp;
    const uint32_t cx = rng_below(r, C);
    const uint64_t x_beat = g->io_next[cx];
    g->io_next[cx] += (uint64_t)(T - 1) * xs + xl + rng_below(r, 4);
    for (uint32_t t = 0; t < T; ++t) {
        uint8_t * d = tsr_sim_dram(g->sim, cx) + (x_beat + (uint64_t)t * xs) * 64;
        for (uint32_t k = 0; k < K; ++k) put_f32(d + 4 * k, x[(size_t)t * K + k]);
    }

    // SPM layout (never reused within a program).
    const uint32_t spm_x = g->spm_next + rng_below(r, 2);
    const uint32_t spm_qa = spm_x + (T - 1) * xs + xl + rng_below(r, 2);
    const uint32_t spm_y = spm_qa + T * ql + rng_below(r, 2);
    const uint32_t y_span = (T - 1) * ys + yl;
    g->spm_next = spm_y + y_span + rng_below(r, 2);

    mutate_regs(g);
    if (rng_below(r, 2)) {
        emit(g, tsr_enc_ld(cx, spm_x, (uint32_t)x_beat, (T - 1) * xs + xl), 2);
    } else {
        for (uint32_t t = 0; t < T; ++t) emit(g, tsr_enc_ld(cx, spm_x + t * xs, (uint32_t)(x_beat + (uint64_t)t * xs), xl), 2);
    }
    const int zx = xs == xl && rng_below(r, 2);
    const int zy = ys == yl && rng_below(r, 2);
    g->cov[COV_ZSTRIDE] += zx + zy;
    emit(g, tsr_enc_qact(kind, spm_qa, spm_x, K, T, zx ? 0 : xs), 2);

    // Old values for ACC: loaded over the whole dst span (so every dst line is written).
    uint8_t * yspan = (uint8_t *)xmalloc((size_t)y_span * 64);
    if (acc) {
        const uint32_t cy = rng_below(r, C);
        const uint64_t yo = g->io_next[cy];
        g->io_next[cy] += y_span + rng_below(r, 4);
        for (uint32_t i = 0; i < y_span * 16; ++i) {
            float v = (float)rng_normal(r);
            if (rng_below(r, 40) == 0) v = (rng_u32(r) & 1) ? 0.0f : -0.0f;
            put_f32(yspan + 4 * i, v);
        }
        memcpy(tsr_sim_dram(g->sim, cy) + yo * 64, yspan, (size_t)y_span * 64);
        mutate_regs(g);
        emit(g, tsr_enc_ld(cy, spm_y, (uint32_t)yo, y_span), 2);
    }
    for (int w = 0; w < 1 + chain; ++w) {
        if (w) mutate_regs(g);
        const uint32_t fl = (acc || w) ? TSR_MVM_FLAG_ACC : 0;
        emit(g, tsr_enc_mvm(fmt, fl, spm_y, spm_qa, (uint32_t)w_off[w], N, K, T, zy ? 0 : ys), 3);
    }

    // Expected dst span (tokens' lines; gap lines = loaded old values when acc).
    uint8_t * qa = (uint8_t *)xmalloc((size_t)ql * 64);
    for (uint32_t t = 0; t < T; ++t) {
        tsr_qact(kind, x + (size_t)t * K, K, qa);
        uint8_t * yt = yspan + (size_t)t * ys * 64;
        for (uint32_t n = 0; n < N; ++n) {
            float y = 0.0f;
            for (int w = 0; w < 1 + chain; ++w) {
                tsr_repack_row(fmt, blocks[w] + rbytes * n, K, row);
                const float d = tsr_dot_row(fmt, row, K, qa);
                y = (w == 0 && !acc) ? d : tsr_fadd(w == 0 ? get_f32(yt + 4 * n) : y, d);
            }
            put_f32(yt + 4 * n, y);
        }
        memset(yt + 4 * N, 0, (size_t)yl * 64 - 4 * N);
    }

    // Store: whole span when every line of it was written (acc or no gaps), else per token.
    mutate_regs(g);
    if (acc || ys == yl) {
        const uint32_t cs = rng_below(r, C);
        const uint64_t ob = g->io_next[cs];
        g->io_next[cs] += y_span + rng_below(r, 4);
        emit(g, tsr_enc_st(cs, spm_y, (uint32_t)ob, y_span), 2);
        add_check(g, cs, ob, y_span, yspan);
    } else {
        for (uint32_t t = 0; t < T; ++t) {
            const uint32_t cs = rng_below(r, C);
            const uint64_t ob = g->io_next[cs];
            g->io_next[cs] += yl + rng_below(r, 4);
            emit(g, tsr_enc_st(cs, spm_y + t * ys, (uint32_t)ob, yl), 2);
            uint8_t * w = (uint8_t *)xmalloc((size_t)yl * 64);
            memcpy(w, yspan + (size_t)t * ys * 64, (size_t)yl * 64);
            add_check(g, cs, ob, yl, w);
        }
        free(yspan);
    }
    free(qa);
    free(x);
    free(row);
    free(blocks[0]);
    free(blocks[1]);
    return 1;
}

// Appends a faulting instruction instead of END. Returns the expected error code; *err_pc
// receives the expected ERR_PC (the faulting instruction, or CMD_COUNT when END is omitted).
static uint32_t gen_error_tail(gen_t * g, uint32_t * err_pc) {
    rng_t * r = &g->r;
    const tsr_config * c = &g->o->cfg;
    for (;;) {
        *err_pc = g->n;
        switch (rng_below(r, 9)) {
        case 0: {   // reserved bit in a NOP
            tsr_insn i = tsr_enc_nop();
            const uint32_t bit = 8 + rng_below(r, 256 - 8);
            i.w[bit / 32] |= 1u << (bit % 32);
            push(g, i);
            return TSR_ERR_ILLEGAL_OP;
        }
        case 1: {   // opcode outside M1
            static const uint32_t ops[] = { 0x04, 0x05, 0x06, 0x07, 0x08, 0x12, 0x30, 0x31, 0x32, 0x33, 0x40, 0x09, 0xFF };
            tsr_insn i = tsr_enc_nop();
            i.w[0] = ops[rng_below(r, sizeof(ops) / sizeof(ops[0]))];
            push(g, i);
            return TSR_ERR_ILLEGAL_OP;
        }
        case 2:
            if (c->num_ch >= 16) break;
            push(g, tsr_enc_ld(c->num_ch + rng_below(r, 16 - c->num_ch), 0, 0, 1));
            return TSR_ERR_ILLEGAL_OP;
        case 3:
            push(g, tsr_enc_mvm(2 + rng_below(r, 14), 0, 0, 0, 0, 1, 32, 1, 0));
            return TSR_ERR_BAD_FMT;
        case 4:
            push(g, tsr_enc_qact(rng_below(r, 2), 0, 0, 48, 1, 0));
            return TSR_ERR_K_RANGE;
        case 5:
            if (c->t_max >= 15) break;
            push(g, tsr_enc_mvm(0, 0, 0, 0, 0, 1, 32, c->t_max + 1 + rng_below(r, 15 - c->t_max), 0));
            return TSR_ERR_T_RANGE;
        case 6:
            push(g, tsr_enc_ld(0, g->spm_lines - 1, 0, 2));
            return TSR_ERR_SPM_RANGE;
        case 7:
            push(g, tsr_enc_st(rng_below(r, c->num_ch), 0, (uint32_t)(g->lim - 1), 2));
            return TSR_ERR_DRAM_RANGE;
        default:
            return TSR_ERR_ILLEGAL_OP;   // no END: runs past CMD_COUNT, ERR_PC = CMD_COUNT = g->n
        }
    }
}

// ---------------------------------------------------------------------------
// M2 programs (--ops m2): only NOP END LI ADDI LD ST QACT FENCE (the M2 RTL core has no MVM).
// Groups: LD -> QACT (-> QACT of the other kind on the same source) -> ST of the QA images, and
// LD -> ST copies (unaligned, 4 KiB-crossing, up to 320 lines, partial re-stores, operands ending
// exactly at the last SPM line / last addressable DRAM beat, zero-length LD/ST exactly at the
// limits). Registers: LI / ADDI on R0..R15 (R0 writes are discarded), every address field
// register-relative with probability 1/2. Error endings cover every error code an M2 program can
// reach (BAD_FMT needs MVM / DQROW, ABORTED needs the CSR port): reserved fields of every M2
// instruction, opcodes outside M2 (MVM excluded: the simulator executes it), LD/ST channel >=
// NUM_CH, QACT kind >= 2, K_RANGE, T_RANGE, SPM_RANGE (each operand, 64-bit extents, zero length
// beyond the end), DRAM_RANGE, command fetch beyond the DRAM limit (mid-program and CMD_ADDR
// itself, including CMD_ADDR_HI != 0) and a missing END; several of them with lower-priority
// violations in the same instruction (check order ILLEGAL_OP > K > T > SPM > DRAM, A1 before A2).
// ---------------------------------------------------------------------------
enum { T2_RSV, T2_OPC, T2_CH, T2_KIND, T2_K, T2_T, T2_SPM, T2_DRAM, T2_FETCH_END, T2_FETCH_START,
       T2_NOEND, T2_N };

static void mutate_regs_m2(gen_t * g) {
    const uint32_t n = rng_below(&g->r, 3);
    for (uint32_t k = 0; k < n; ++k) {
        uint32_t rd = 1 + rng_below(&g->r, 15);
        if (rng_below(&g->r, 16) == 0) {
            rd = 0;                                   // discarded: R0 stays 0
            g->cov[C2_R0W]++;
        }
        if (rng_below(&g->r, 2)) {
            const uint32_t v = rng_below(&g->r, 2) ? rng_u32(&g->r) : rng_below(&g->r, 1u << 24);
            push(g, tsr_enc_li(rd, v));
            if (rd) g->R[rd] = v;
        } else {
            const uint32_t ra = rng_below(&g->r, 16), imm = rng_u32(&g->r);
            push(g, tsr_enc_addi(rd, ra, imm));
            if (rd) g->R[rd] = g->R[ra] + imm;
        }
    }
}

// Like emit() (register-relative address fields) but never followed by NOP / FENCE.
static void emit_tail(gen_t * g, tsr_insn i, int n_addr) {
    uint32_t b[3] = { 0, 0, 0 };
    for (int f = 0; f < n_addr; ++f) {
        if (rng_below(&g->r, 2)) {
            b[f] = rng_below(&g->r, 16);
            i.w[1 + f] -= g->R[b[f]];
        }
    }
    tsr_insn_set_base(&i, b[0], b[1], b[2]);
    push(g, i);
}

// IO allocations grow upward from io_next; "top" allocations grow downward from the channel's
// addressable end (the first one ends exactly at the DRAM limit). The end of ch0 is kept free when
// a fetch-range tail will place the command buffer there.
static uint64_t io_room(const gen_t * g, uint32_t c) {
    uint64_t hi = g->top_next[c] - ((c == 0 && g->reserve_ch0_top) ? 8192 : 0);
    if (g->o->m3 && hi > g->io_lim) hi = g->io_lim;           // M3: the weight heap sits above IO
    return hi > g->io_next[c] + 128 ? hi - g->io_next[c] - 128 : 0;
}
static uint64_t heap_ceiling(const gen_t * g);
static uint64_t io_alloc(gen_t * g, uint32_t c, uint64_t n) {        // caller checked io_room >= n
    const uint64_t a = g->io_next[c] + rng_below(&g->r, 64);
    g->io_next[c] = a + n;
    return a;
}
static int top_alloc(gen_t * g, uint32_t c, uint64_t n, uint64_t * beat) {
    if (c == 0 && g->reserve_ch0_top) return 0;
    if (g->o->m3) {                       // M3: above everything the weight heap may still take
        if (g->wlim_pending || g->top_next[c] < heap_ceiling(g) + n + 256) return 0;
    } else if (g->top_next[c] < g->io_next[c] + n + 256) {
        return 0;
    }
    g->top_next[c] -= n;
    *beat = g->top_next[c];
    return 1;
}
static uint32_t spm_room(const gen_t * g) { return g->spm_hi > g->spm_next + 8 ? g->spm_hi - g->spm_next - 8 : 0; }

static uint32_t pick_k_m2(gen_t * g, uint32_t kind, uint32_t kcap, int * edge) {
    rng_t * r = &g->r;
    const uint32_t bs = tsr_qa_block_size(kind);
    const uint32_t nmax = kcap / bs;
    const uint32_t sel = rng_below(r, 16);
    *edge = 0;
    if (sel == 0 || nmax == 0) { *edge = 1; return bs; }
    if (sel == 1) { *edge = 2; return g->o->cfg.k_max; }
    switch (rng_below(r, 3)) {
    case 0: return bs * (1 + rng_below(r, nmax < 8 ? nmax : 8));
    case 1: if (kcap >= 1024) return 1024 * (1 + rng_below(r, kcap / 1024)); /* fallthrough */
    default: return bs * (1 + rng_below(r, nmax));
    }
}

// LD (whole span or per token) of T vectors placed xs lines apart.
static void emit_loads(gen_t * g, uint32_t ch, uint32_t spm, uint64_t beat, uint32_t T, uint32_t xs, uint32_t xl) {
    mutate_regs_m2(g);
    if (rng_below(&g->r, 2)) {
        emit(g, tsr_enc_ld(ch, spm, (uint32_t)beat, (T - 1) * xs + xl), 2);
    } else {
        for (uint32_t t = 0; t < T; ++t) emit(g, tsr_enc_ld(ch, spm + t * xs, (uint32_t)(beat + (uint64_t)t * xs), xl), 2);
    }
}

// ST of T images of `l` lines each (consecutive in the SPM), whole span or per token, to channel cs.
static void emit_stores(gen_t * g, uint32_t cs, uint32_t spm, const uint8_t * img, uint32_t T, uint32_t l) {
    mutate_regs_m2(g);
    if (T == 1 || rng_below(&g->r, 2)) {
        const uint64_t ob = io_alloc(g, cs, (uint64_t)T * l);
        g->cov[C2_X4K] += (ob % 64) + (uint64_t)T * l > 64;
        emit(g, tsr_enc_st(cs, spm, (uint32_t)ob, T * l), 2);
        uint8_t * w = (uint8_t *)xmalloc((size_t)T * l * 64);
        memcpy(w, img, (size_t)T * l * 64);
        add_check(g, cs, ob, T * l, w);
    } else {
        for (uint32_t t = 0; t < T; ++t) {
            const uint64_t ob = io_alloc(g, cs, l);
            emit(g, tsr_enc_st(cs, spm + t * l, (uint32_t)ob, l), 2);
            uint8_t * w = (uint8_t *)xmalloc((size_t)l * 64);
            memcpy(w, img + (size_t)t * l * 64, (size_t)l * 64);
            add_check(g, cs, ob, l, w);
        }
    }
}

// LD -> QACT (-> second QACT of the other kind) -> ST. Returns 0 if nothing fits any more.
static int gen_qact_group_m2(gen_t * g) {
    const tsr_config * c = &g->o->cfg;
    rng_t * r = &g->r;
    const uint32_t kcap = g->o->max_k < c->k_max ? g->o->max_k : c->k_max;
    uint32_t kind = 0, K = 0, T = 0, xl = 0, ql[2] = { 0, 0 }, xs = 0, cx = 0, cs = 0;
    int pair = 0, edge = 0, fits = 0;
    for (int attempt = 0; attempt < 32 && !fits; ++attempt) {
        kind = rng_below(r, 2);
        K = pick_k_m2(g, kind, kcap, &edge);
        T = 1 + rng_below(r, c->t_max);
        if (K > 8192 && T > 2) T = 2;                         // keeps traces small
        xl = K / 16;
        ql[0] = tsr_qa_lines(kind, K);
        xs = xl + (rng_below(r, 3) == 0 ? 1 + rng_below(r, 3) : 0);
        pair = rng_below(r, 4) == 0 && K % 256 == 0;
        ql[1] = pair ? tsr_qa_lines(!kind, K) : 0;
        cx = rng_below(r, c->num_ch);
        cs = rng_below(r, c->num_ch);
        const uint64_t span = (uint64_t)(T - 1) * xs + xl;
        const uint64_t out = (uint64_t)T * (ql[0] + ql[1] + 128);
        fits = span + (uint64_t)T * (ql[0] + ql[1]) <= spm_room(g) &&
               io_room(g, cx) >= span + 64 && io_room(g, cs) >= out + (cs == cx ? span + 64 : 0);
    }
    if (!fits) return 0;
    g->cov[kind ? C2_A256 : C2_A32]++;
    if (pair) g->cov[kind ? C2_A32 : C2_A256]++;
    g->cov[C2_QPAIR] += pair;
    g->cov[C2_TGT1] += T > 1;
    g->cov[C2_XSTRIDE] += xs != xl;
    g->cov[C2_KMIN] += edge == 1;
    g->cov[C2_KMAX] += edge == 2;

    float * x = (float *)xmalloc(sizeof(float) * K * T);
    int sp = 0;
    for (uint32_t t = 0; t < T; ++t) sp |= gen_x(r, x + (size_t)t * K, K, g->o->special);
    g->cov[C2_SPECIAL] += sp;
    const uint64_t span = (uint64_t)(T - 1) * xs + xl;
    const uint64_t x_beat = io_alloc(g, cx, span);
    for (uint32_t t = 0; t < T; ++t) {
        uint8_t * d = tsr_sim_dram(g->sim, cx) + (x_beat + (uint64_t)t * xs) * 64;
        for (uint32_t k = 0; k < K; ++k) put_f32(d + 4 * k, x[(size_t)t * K + k]);
    }
    g->cov[C2_X4K] += (x_beat % 64) + span > 64;

    const uint32_t spm_x = g->spm_next + rng_below(r, 2);
    uint32_t spm_qa[2];
    spm_qa[0] = spm_x + (uint32_t)span + rng_below(r, 2);
    spm_qa[1] = spm_qa[0] + T * ql[0] + rng_below(r, 2);
    g->spm_next = spm_qa[1] + T * ql[1] + rng_below(r, 2);

    emit_loads(g, cx, spm_x, x_beat, T, xs, xl);
    for (int v = 0; v < 1 + pair; ++v) {
        const uint32_t kv = v ? !kind : kind;
        const int z = xs == xl && rng_below(r, 2);
        g->cov[C2_ZSTRIDE] += z;
        mutate_regs_m2(g);
        emit(g, tsr_enc_qact(kv, spm_qa[v], spm_x, K, T, z ? 0 : xs), 2);
    }
    for (int v = 0; v < 1 + pair; ++v) {
        const uint32_t kv = v ? !kind : kind;
        uint8_t * img = (uint8_t *)xmalloc((size_t)T * ql[v] * 64);
        for (uint32_t t = 0; t < T; ++t) tsr_qact(kv, x + (size_t)t * K, K, img + (size_t)t * ql[v] * 64);
        emit_stores(g, cs, spm_qa[v], img, T, ql[v]);
        free(img);
    }
    free(x);
    return 1;
}

// LD -> ST copies, and zero-length LD / ST exactly at the SPM / DRAM limits.
static int gen_copy_group_m2(gen_t * g) {
    const tsr_config * c = &g->o->cfg;
    rng_t * r = &g->r;
    const uint32_t C = c->num_ch;
    if (rng_below(r, 8) == 0) {
        const uint32_t ch = rng_below(r, C);
        const uint32_t dlim = g->lim > 0xFFFFFFFFull ? 0xFFFFFFFFu : (uint32_t)g->lim;
        const uint32_t spm = rng_below(r, 2) ? g->spm_lines : rng_below(r, g->spm_lines + 1);
        const uint32_t dram = (spm != g->spm_lines || rng_below(r, 2)) ? dlim : rng_below(r, dlim);
        mutate_regs_m2(g);
        emit(g, rng_below(r, 2) ? tsr_enc_ld(ch, spm, dram, 0) : tsr_enc_st(ch, spm, dram, 0), 2);
        g->cov[C2_ZERO]++;
        return 1;
    }
    uint32_t n = 0, ca = 0, cb = 0, spm = 0, k = 0;
    int spm_top = 0, src_top = 0, dst_top = 0, fits = 0;
    for (int attempt = 0; attempt < 32 && !fits; ++attempt) {
        n = rng_below(r, 4) == 0 ? 1 + rng_below(r, 64) : 1 + rng_below(r, 320);
        ca = rng_below(r, C);
        cb = rng_below(r, C);
        spm_top = rng_below(r, 6) == 0;
        src_top = rng_below(r, 6) == 0;
        dst_top = rng_below(r, 6) == 0;
        k = rng_below(r, 4) == 0 ? rng_below(r, n) : 0;
        fits = (spm_top ? g->spm_hi >= g->spm_next + n + 16 : spm_room(g) >= n + 2) &&
               io_room(g, ca) >= (uint64_t)n + 64 && io_room(g, cb) >= (uint64_t)n + 128 + (ca == cb ? n + 64 : 0);
    }
    if (!fits) return 0;
    g->cov[C2_COPY]++;
    if (spm_top) {
        g->spm_hi -= n;
        spm = g->spm_hi;
        g->cov[C2_SLIM] += spm + n == g->spm_lines;
    } else {
        spm = g->spm_next + rng_below(r, 2);
        g->spm_next = spm + n + rng_below(r, 2);
    }
    uint64_t src = 0, dst = 0;
    if (!src_top || !top_alloc(g, ca, n, &src)) src = io_alloc(g, ca, n);
    uint8_t * data = (uint8_t *)xmalloc((size_t)n * 64);
    rng_bytes(r, data, (size_t)n * 64);
    memcpy(tsr_sim_dram(g->sim, ca) + src * 64, data, (size_t)n * 64);
    if (!dst_top || !top_alloc(g, cb, n - k, &dst)) dst = io_alloc(g, cb, n - k);
    g->cov[C2_DLIM] += (src + n == g->lim) + (dst + (n - k) == g->lim);
    g->cov[C2_X4K] += ((src % 64) + n > 64) + ((dst % 64) + (n - k) > 64);
    mutate_regs_m2(g);
    emit(g, tsr_enc_ld(ca, spm, (uint32_t)src, n), 2);
    mutate_regs_m2(g);
    emit(g, tsr_enc_st(cb, spm + k, (uint32_t)dst, n - k), 2);
    uint8_t * want = (uint8_t *)xmalloc((size_t)(n - k) * 64);
    memcpy(want, data + (size_t)k * 64, (size_t)(n - k) * 64);
    add_check(g, cb, dst, n - k, want);
    free(data);
    return 1;
}

static int gen_group_m2(gen_t * g) {
    if (rng_below(&g->r, 4) == 0) return gen_copy_group_m2(g) || gen_qact_group_m2(g);
    return gen_qact_group_m2(g) || gen_copy_group_m2(g);
}

// An instruction of opcode op with every used field random (in range or not).
static tsr_insn rand_shaped(gen_t * g, uint32_t op) {
    rng_t * r = &g->r;
    tsr_insn i = {{0}};
    const uint32_t ra = rng_below(r, 16), rb = rng_below(r, 16), rd = rng_below(r, 16);
    switch (op) {
    case TSR_OP_LI:   i.w[0] = tsr_w0(op, 0, 0, 0, 0, 0, rd); i.w[1] = rng_u32(r); break;
    case TSR_OP_ADDI: i.w[0] = tsr_w0(op, 0, 0, ra, 0, 0, rd); i.w[1] = rng_u32(r); break;
    case TSR_OP_LD:
    case TSR_OP_ST:
        i.w[0] = tsr_w0(op, rng_below(r, g->o->cfg.num_ch), 0, ra, rb, 0, 0);
        for (int k = 1; k <= 3; ++k) i.w[k] = rng_below(r, 2) ? rng_u32(r) : rng_below(r, 64);
        break;
    case TSR_OP_QACT:
        i.w[0] = tsr_w0(op, rng_below(r, 2), 0, ra, rb, 0, 0);
        for (int k = 1; k <= 5; ++k) i.w[k] = rng_below(r, 2) ? rng_u32(r) : 32u * rng_below(r, 64);
        break;
    default:          i.w[0] = op; break;           // NOP, END, FENCE
    }
    return i;
}

// QACT with a legal K (multiple of the block size, <= K_MAX, small) and T.
static void legal_kt(gen_t * g, uint32_t kind, uint32_t * K, uint32_t * T) {
    const uint32_t bs = tsr_qa_block_size(kind);
    const uint32_t kcap = g->o->max_k < g->o->cfg.k_max ? g->o->max_k : g->o->cfg.k_max;
    const uint32_t nmax = kcap / bs ? kcap / bs : 1;
    *K = bs * (1 + rng_below(&g->r, nmax < 8 ? nmax : 8));
    *T = 1 + rng_below(&g->r, g->o->cfg.t_max);
}

// Appends the faulting instruction of error tail `tail` (not the fetch / END variants).
// Returns the expected error code; *err_pc = its index; *err_info = expected ERR_INFO.
static uint32_t gen_error_tail_m2(gen_t * g, int tail, uint32_t * err_pc, int64_t * err_info) {
    rng_t * r = &g->r;
    const tsr_config * c = &g->o->cfg;
    const uint32_t S = g->spm_lines;
    const uint64_t L = g->lim;
    *err_pc = g->n;
    *err_info = 0;
    switch (tail) {
    case T2_CH:
        if (c->num_ch < 16) {
            tsr_insn i = rand_shaped(g, rng_below(r, 2) ? TSR_OP_LD : TSR_OP_ST);
            i.w[0] = (i.w[0] & ~0xF00u) | ((c->num_ch + rng_below(r, 16 - c->num_ch)) << 8);
            push(g, i);
            return TSR_ERR_ILLEGAL_OP;
        }
        /* fallthrough */
    case T2_RSV: {
        static const uint32_t ops[] = { TSR_OP_NOP, TSR_OP_END, TSR_OP_FENCE, TSR_OP_LI, TSR_OP_ADDI,
                                        TSR_OP_LD, TSR_OP_ST, TSR_OP_QACT };
        const uint32_t op = ops[rng_below(r, 8)];
        uint32_t mask, wfree;
        switch (op) {
        case TSR_OP_LI:   mask = 0x0FFFFF00u; wfree = 2; break;
        case TSR_OP_ADDI: mask = 0x0FF0FF00u; wfree = 2; break;
        case TSR_OP_LD:
        case TSR_OP_ST:   mask = 0xFF00F000u; wfree = 4; break;
        case TSR_OP_QACT: mask = 0xFF00F000u; wfree = 6; break;
        default:          mask = 0xFFFFFF00u; wfree = 1; break;
        }
        tsr_insn i = rand_shaped(g, op);
        const uint32_t nm = (uint32_t)__builtin_popcount(mask);
        uint32_t k = rng_below(r, nm + 32u * (8u - wfree));
        if (k < nm) {
            for (uint32_t b = 0; b < 32; ++b) {
                if (((mask >> b) & 1u) && k-- == 0) { i.w[0] |= 1u << b; break; }
            }
        } else {
            k -= nm;
            i.w[wfree + k / 32] |= 1u << (k % 32);
        }
        push(g, i);
        return TSR_ERR_ILLEGAL_OP;
    }
    case T2_OPC: {
        static const uint8_t ops[] = { 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0F, 0x12, 0x13, 0x17, 0x19,
                                       0x1F, 0x21, 0x2F, 0x30, 0x31, 0x32, 0x33, 0x34, 0x40, 0x41, 0x7E, 0x80,
                                       0xA5, 0xFE, 0xFF };
        tsr_insn i = {{0}};
        i.w[0] = ops[rng_below(r, sizeof(ops))] | (rng_below(r, 2) ? (rng_u32(r) & ~0xFFu) : 0u);
        if (rng_below(r, 2)) for (int k = 1; k < 8; ++k) i.w[k] = rng_u32(r);
        push(g, i);
        return TSR_ERR_ILLEGAL_OP;
    }
    case T2_KIND: {
        tsr_insn i = rand_shaped(g, TSR_OP_QACT);
        i.w[0] = (i.w[0] & ~0xF00u) | ((2 + rng_below(r, 14)) << 8);
        push(g, i);
        return TSR_ERR_ILLEGAL_OP;
    }
    case T2_K: {
        const uint32_t kind = rng_below(r, 2), bs = tsr_qa_block_size(kind);
        static const uint32_t big[] = { 0xFFFFFFFFu, 0x80000000u, 0x00010000u };
        uint32_t K, T;
        legal_kt(g, kind, &K, &T);
        switch (rng_below(r, 5)) {
        case 0:  K = 0; break;
        case 1:  K = bs / 2 + bs * rng_below(r, 4); break;                    // not a multiple
        case 2:  K = kind ? 256 * rng_below(r, 4) + 32 * (1 + rng_below(r, 7)) : 32 * rng_below(r, 4) + 16; break;
        case 3:  K = c->k_max + bs * (1 + rng_below(r, 4)); break;           // multiple, > K_MAX
        default: K = big[rng_below(r, 3)]; break;
        }
        if (rng_below(r, 2)) T = rng_below(r, 2) ? 0 : c->t_max + 1 + rng_below(r, 8);   // T also bad
        const uint32_t dst = rng_below(r, 2) ? 0 : S + rng_below(r, 100);              // SPM maybe bad
        push(g, tsr_enc_qact(kind, dst, rng_u32(r), K, T, rng_u32(r)));
        return TSR_ERR_K_RANGE;
    }
    case T2_T: {
        const uint32_t kind = rng_below(r, 2);
        uint32_t K, T;
        legal_kt(g, kind, &K, &T);
        static const uint32_t bad[] = { 0, 9, 16, 0xFFFFFFFFu, 0x80000001u };
        T = rng_below(r, 3) ? bad[rng_below(r, 5)] : c->t_max + 1 + rng_below(r, 8);
        if (T >= 1 && T <= c->t_max) T = 0;
        const uint32_t dst = rng_below(r, 2) ? 0 : S + rng_below(r, 100);
        push(g, tsr_enc_qact(kind, dst, rng_u32(r), K, T, rng_below(r, 4)));
        return TSR_ERR_T_RANGE;
    }
    case T2_SPM: {
        const uint32_t v = rng_below(r, 5);
        if (v <= 1) {                          // LD / ST SPM operand beyond the SPM
            const uint32_t n = v ? 0 : 1 + rng_below(r, 64);
            const uint32_t spm = v ? S + 1 + rng_below(r, 1000) : S - (n < S ? n : S) + 1 + rng_below(r, n);
            uint32_t dram = rng_below(r, 1u << 20);
            if (rng_below(r, 2)) dram = (uint32_t)(L > 0xFFFFFFFFull ? 0xFFFFFFFFu : L);   // DRAM also bad
            emit_tail(g, rng_below(r, 2) ? tsr_enc_ld(rng_below(r, c->num_ch), spm, dram, n)
                                         : tsr_enc_st(rng_below(r, c->num_ch), spm, dram, n), 2);
            *err_info = spm;
            return TSR_ERR_SPM_RANGE;
        }
        const uint32_t kind = rng_below(r, 2);
        uint32_t K, T;
        legal_kt(g, kind, &K, &T);
        const uint32_t xl = K / 16, ql = tsr_qa_lines(kind, K);
        if (v == 2) {                          // QACT dst beyond the SPM (src maybe too)
            const uint64_t ext = (uint64_t)T * ql;
            const uint32_t dst = (uint32_t)(ext <= S ? S - ext + 1 + rng_below(r, (uint32_t)ext) : rng_below(r, S));
            const uint32_t src = rng_below(r, 2) ? S + rng_below(r, 50) : 0;
            emit_tail(g, tsr_enc_qact(kind, dst, src, K, T, 0), 2);
            *err_info = dst;
        } else if (v == 3 || c->t_max < 2) {  // QACT src beyond the SPM, dst fine
            const uint32_t st = rng_below(r, 2) ? 0 : xl + rng_below(r, 8);
            const uint64_t ext = (uint64_t)(T - 1) * (st ? st : xl) + xl;
            const uint32_t src = (uint32_t)(ext <= S ? S - ext + 1 + rng_below(r, 16) : rng_below(r, S));
            emit_tail(g, tsr_enc_qact(kind, 0, src, K, T, st), 2);
            *err_info = src;
        } else {                               // (T-1) * stride beyond 32 bits
            T = 2 + rng_below(r, c->t_max - 1);
            const uint32_t src = rng_below(r, S / 2 + 1);
            emit_tail(g, tsr_enc_qact(kind, 0, src, K, T, 0x80000000u | rng_u32(r)), 2);
            *err_info = src;
        }
        return TSR_ERR_SPM_RANGE;
    }
    case T2_DRAM:
    default: {
        if (L >= 0xFFFFFF00ull) {              // limit not reachable with 32-bit addresses
            push(g, tsr_enc_nop());
            g->p[g->n - 1].w[7] = 1;
            return TSR_ERR_ILLEGAL_OP;
        }
        const uint32_t n = rng_below(r, 4) == 0 ? 0 : 1 + rng_below(r, 64);
        const uint32_t spm = rng_below(r, S - (n < S ? n : S) + 1);
        uint32_t dram;
        if (n == 0) dram = (uint32_t)L + 1 + rng_below(r, 100);
        else dram = (uint32_t)(L - n) + 1 + rng_below(r, n + 50);
        if (rng_below(r, 8) == 0) dram = 0xFFFFFFFFu - rng_below(r, 16);
        emit_tail(g, rng_below(r, 2) ? tsr_enc_ld(rng_below(r, c->num_ch), spm, dram, n)
                                     : tsr_enc_st(rng_below(r, c->num_ch), spm, dram, n), 2);
        *err_info = dram;
        return TSR_ERR_DRAM_RANGE;
    }
    }
}

// ---------------------------------------------------------------------------
// M3 programs (--ops m3): MVM-centred programs for the M3 RTL core (MVM clusters), mixed with the
// M2 QACT and LD -> ST copy groups (and their coverage slots). An MVM group is
//
//   LD x -> QACT (or LD of a raw QA image) -> [LD old values / guard lines] -> MVM x n -> ST
//
// with up to two outputs sharing the QA (siblings like ffn_gate / ffn_up), ACC chains of up to
// three MVMs per output, an optional N = 0 MVM in between (must write nothing) and an optional
// second stage QACT(y) -> MVM (an MVM result read back as activations). Shapes cover both formats,
// K = block size / partial last chunk / whole chunks / K_MAX (the activation buffer at full depth,
// also at T = T_MAX through raw QA), N = 0, 1, < C (empty channels), random (a y line holding
// rows of two channels), > 1000; T = 1..T_MAX with dst_stride 0 or explicit with gap lines.
// Values: special activations (gen_x), near-constant "extreme" activations, raw QA images (LD'd:
// full int8 range incl. -128, f32 Dx incl. +-0 / subnormal / Inf / NaN / max finite, garbage in
// the unused tail bytes, BS either the sums of QS or arbitrary int16), weight scales d / dmin that
// are +-0, subnormal, Inf, NaN or +-65504, Q8_0 q = -128, Q4_K 6-bit scales / mins = 63 and
// isum > 2^24 (int -> f32 rounds), ACC old values incl. +-0, subnormals, Inf, NaN payloads.
// Placement: register-relative A1 / A2 / A3 (LI / ADDI on R0..R15), weight streams crossing 4 KiB
// blocks at random offsets with unread gaps between tensors (an over-reading cluster reads
// outside the init regions), one tensor per trace (1/4) ending exactly at the DRAM limit, dst /
// QA regions ending exactly at the SPM end, guard lines around dst spans (an MVM that writes
// outside its token lines changes a stored line). Every MVM reads its weights once per channel:
// channel c reads rows_c x ROW_BEATS beats (the trace's init regions hold exactly those).
// Error endings: 3/4 MVM faults (reserved fields, BAD_FMT, K_RANGE, T_RANGE, SPM_RANGE on A1 /
// A2, DRAM_RANGE on A3, each with lower-priority violations in the same instruction, extents
// that wrap 32 bits), 1/4 the M2 endings.
// ---------------------------------------------------------------------------
enum { C3_Q8 = C2_N, C3_Q4, C3_K32, C3_K256, C3_KPART, C3_KMAX, C3_KMAXT, C3_N0, C3_N1, C3_NLTC, C3_ODDN,
       C3_NMIX, C3_NBIG, C3_TGT1, C3_TMAX, C3_YZERO, C3_YGAP, C3_GUARD, C3_ACC, C3_ACCSP, C3_CHAIN,
       C3_SHARE, C3_STAGE2, C3_RAWQA, C3_RAWBS, C3_XSP, C3_XEXT, C3_DZ, C3_DSUB, C3_DINF, C3_DNAN,
       C3_DMAX, C3_Q128, C3_SC63, C3_ISUM, C3_A1REL, C3_A2REL, C3_A3REL, C3_W4K, C3_WLIM, C3_N0LIM,
       C3_YEND, C3_QAEND, C3_ERR, C3_WRAP, C3_END };
#define C3_N (C3_END - C2_N)
_Static_assert(C3_END <= COV_ERRC, "coverage slots overlap the error-code counts");
static const char * cov3_names[C3_N] = {
    "Q8_0 MVMs", "Q4_K MVMs", "K = 32", "K = 256", "partial last chunk", "K = K_MAX", "K = K_MAX at T = T_MAX",
    "N = 0", "N = 1", "N < C (empty channels)", "odd N", "y lines shared by 2 channels", "N > 1000", "T > 1",
    "T = T_MAX", "dst_stride 0", "dst_stride gaps", "guarded dst spans", "ACC (loaded old values)",
    "ACC special old values", "ACC chains", "sibling MVMs sharing a QA", "MVM -> QACT -> MVM", "raw (LD) QA",
    "raw QA with arbitrary BS", "special activations", "extreme activations", "d/dmin +-0", "d/dmin subnormal",
    "d/dmin Inf", "d/dmin NaN", "d/dmin +-65504", "Q8_0 q = -128", "Q4_K sc/m = 63",
    "Q4_K int->f32 rounded isum terms", "reg-relative A1", "reg-relative A2", "reg-relative A3",
    "4 KiB-crossing weight streams", "tensors ending at the DRAM limit", "N = 0 with W_OFF = limit",
    "dst ending at the SPM end", "QA ending at the SPM end", "MVM error endings",
    "faulting operands whose 32-bit extent wraps" };
_Static_assert(sizeof(cov3_names) / sizeof(cov3_names[0]) == C3_N, "cov3_names");

enum { T3_RSV, T3_FMT, T3_K, T3_T, T3_SPM_DST, T3_SPM_X, T3_DRAM, T3_N };

static uint64_t min_top(const gen_t * g) {
    uint64_t m = g->lim;
    for (uint32_t c = 0; c < g->o->cfg.num_ch; ++c) m = g->top_next[c] < m ? g->top_next[c] : m;
    return m;
}
// Highest beat the weight heap can still reach (remaining budget + per-tensor gaps).
static uint64_t heap_ceiling(const gen_t * g) {
    const uint64_t left = g->wbudget > g->wch0 ? g->wbudget - g->wch0 : 0;
    return g->heap_next + left + 4096;
}

// Symmetric weight allocation (same W_OFF in every channel). Returns W_OFF; *at_lim when the
// tensor (channel 0: R x ROW_BEATS) ends exactly at the DRAM limit.
static uint64_t heap_take(gen_t * g, uint64_t n, int * at_lim) {
    *at_lim = 0;
    if (g->wlim_pending && n > 0) {
        g->wlim_pending = 0;
        const uint64_t w = g->lim - n;
        for (uint32_t c = 0; c < g->o->cfg.num_ch; ++c) if (g->top_next[c] > w) g->top_next[c] = w;
        *at_lim = 1;
        return w;
    }
    const uint64_t w = g->heap_next + (rng_below(&g->r, 4) ? rng_below(&g->r, 16) : 0);
    g->heap_next = w + n + (rng_below(&g->r, 4) ? 1 + rng_below(&g->r, 16) : 0);
    return w;
}

// SPM region of n lines from the bottom (spm_next, 0-1 line gap) or the top (spm_hi).
static uint32_t spm_take(gen_t * g, uint32_t n, int top) {
    if (top) {
        g->spm_hi -= n;
        return g->spm_hi;
    }
    const uint32_t a = g->spm_next + rng_below(&g->r, 2);
    g->spm_next = a + n;
    return a;
}

// fp16 weight scale with a special value: +-0, subnormal, Inf, NaN, +-65504, min normal, large.
static uint16_t fp16_special(rng_t * r) {
    const uint16_t sign = (uint16_t)((rng_u32(r) & 1u) << 15);
    switch (rng_below(r, 7)) {
    case 0:  return sign;
    case 1:  return (uint16_t)(sign | (1 + rng_below(r, 0x3FF)));
    case 2:  return (uint16_t)(sign | 0x7C00);
    case 3:  return (uint16_t)(sign | 0x7C00 | (1 + rng_below(r, 0x3FF)));
    case 4:  return (uint16_t)(sign | 0x7BFF);
    case 5:  return (uint16_t)(sign | 0x0400);
    default: return (uint16_t)(sign | ((25 + rng_below(r, 6)) << 10) | rng_below(r, 1024));
    }
}

enum { WF_DZ = 1, WF_DSUB = 2, WF_DINF = 4, WF_DNAN = 8, WF_DMAX = 16, WF_Q128 = 32, WF_SC63 = 64 };
static uint32_t fp16_class(uint16_t h) {
    const uint32_t e = (h >> 10) & 0x1Fu, m = h & 0x3FFu;
    if (e == 0) return m ? WF_DSUB : WF_DZ;
    if (e == 31) return m ? WF_DNAN : WF_DINF;
    return (h & 0x7FFF) == 0x7BFF ? WF_DMAX : 0;
}

// N rows of ggml blocks. mode 0: random bytes, fp16 scales as M1 (rare +-0 / subnormal);
// 1: a quarter of the rows get special scales in 1/8 of their blocks; 2: extreme rows (Q8_0
// quants -128 / -127 / 127, Q4_K nibbles 14 / 15 with every 6-bit scale and min = 63).
// Returns WF_* flags of what the tensor contains.
static uint32_t gen_tensor_m3(gen_t * g, uint32_t fmt, uint32_t N, uint32_t K, int mode, uint8_t * blocks) {
    rng_t * r = &g->r;
    const uint32_t nb = K / tsr_fmt_block_size(fmt);
    const size_t bsz = tsr_fmt_ggml_block_bytes(fmt);
    uint32_t fl = 0;
    for (uint32_t n = 0; n < N; ++n) {
        const int srow = mode == 1 && rng_below(r, 4) == 0;
        for (uint32_t b = 0; b < nb; ++b) {
            uint8_t * blk = blocks + ((size_t)n * nb + b) * bsz;
            rng_bytes(r, blk, bsz);
            if (mode == 2 && fmt == TSR_FMT_Q8_0) {
                for (int l = 0; l < 32; ++l) blk[2 + l] = rng_below(r, 4) ? 0x80 : (uint8_t)(0x7F + rng_below(r, 3));
            } else if (mode == 2) {
                static const uint8_t nib[4] = { 0xFF, 0xFF, 0xEF, 0xFE };
                memset(blk + 4, 0xFF, 12);
                for (int l = 0; l < 128; ++l) blk[16 + l] = nib[rng_below(r, 4)];
            }
            const int nsc = fmt == TSR_FMT_Q4_K ? 2 : 1;
            for (int k = 0; k < nsc; ++k) {
                uint16_t d = (srow && rng_below(r, 8) == 0) ? fp16_special(r)
                                                            : rand_fp16_scale(r, g->o->special && mode != 2);
                if (rng_below(r, 4) == 0 && (d & 0x7C00) != 0 && (d & 0x7C00) != 0x7C00)   // normal: exponent 1..30
                    d = (uint16_t)((d & 0x83FF) | ((1 + rng_below(r, 30)) << 10));
                blk[2 * k] = (uint8_t)d;
                blk[2 * k + 1] = (uint8_t)(d >> 8);
                fl |= fp16_class(d);
            }
            if (fmt == TSR_FMT_Q8_0) {
                for (int l = 0; l < 32; ++l) if (blk[2 + l] == 0x80) fl |= WF_Q128;
            } else {
                for (int j = 0; j < 8; ++j) {
                    uint8_t sc, m;
                    tsr_get_scale_min_k4(j, blk + 4, &sc, &m);
                    if (sc == 63 || m == 63) fl |= WF_SC63;
                }
            }
        }
    }
    return fl;
}

// Q4_K isum of ggml block blk against 256 activation int8 (ggml nibble order, written from
// dequantize_row_q4_K / get_scale_min_k4, not via tsr_ref): coverage of int -> f32 rounding.
static int32_t q4k_isum(const uint8_t * blk, const int8_t * qx) {
    const uint8_t * sc = blk + 4;
    const uint8_t * qs = blk + 16;
    int32_t isum = 0;
    for (int j = 0; j < 8; ++j) {
        const int32_t s = j < 4 ? (sc[j] & 63) : ((sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4));
        int32_t S = 0;
        for (int l = 0; l < 32; ++l) {
            const uint8_t b = qs[32 * (j / 2) + l];
            S += (int32_t)((j & 1) ? (b >> 4) : (b & 0xF)) * qx[32 * j + l];
        }
        isum += s * S;
    }
    return isum;
}

// Near-constant activations: |x| within 6% of the block maximum, one sign per 256 elements, so
// QACT gives q of one sign near +-127 (Q4_K isum up to ~3e7 > 2^24 with extreme weights).
static void gen_x_extreme(rng_t * r, float * x, uint32_t K) {
    const double scale = ldexp(1.0, (int)rng_below(r, 16) - 8);
    double s = scale;
    for (uint32_t i = 0; i < K; ++i) {
        if (i % 256 == 0) s = (rng_u32(r) & 1) ? scale : -scale;
        x[i] = (float)(s * (1.0 - 0.06 * rng_unit(r)));
    }
}

static uint32_t rand_dx_bits(rng_t * r, int * sp) {
    const uint32_t sign = rng_u32(r) & 0x80000000u;
    const uint32_t k = rng_below(r, 20);
    if (k <= 6) *sp = 1;
    switch (k) {
    case 0:  return sign;                                              // +-0
    case 1:
    case 2:  return sign | (1 + rng_below(r, 0x7FFFFF));               // subnormal (DAZ)
    case 3:  return sign | 0x7F800000u;                                // +-Inf
    case 4:  return sign | (0x7F800001u + rng_below(r, 0x7FFFFE));     // NaN, quiet or signalling
    case 5:  return sign | 0x7F7FFFFFu;                                // max finite (products overflow)
    case 6:  return sign | 0x00800000u;                                // min normal (products flush)
    default: return sign | ((107u + rng_below(r, 21)) << 23) | rng_below(r, 1u << 23);   // 2^-20 .. 2^0
    }
}

// Raw QA image (loaded with LD instead of produced by QACT): every byte random first (unused tail
// bytes stay garbage), then QS blocks of all -128 / all 127 / small / random values (extreme:
// -128..-120), Dx incl. special values, BS = sums of QS (natural) or arbitrary int16 (arb_bs).
// Returns 1 if any Dx is special.
static int gen_raw_qa(rng_t * r, uint32_t kind, uint32_t K, uint8_t * qa, int arb_bs, int extreme) {
    const uint32_t ql = tsr_qa_lines(kind, K);
    rng_bytes(r, qa, (size_t)ql * 64);
    int8_t * qs = (int8_t *)qa;
    for (uint32_t b = 0; b < K / 32; ++b) {
        int8_t * q = qs + 32 * b;
        const uint32_t m = extreme ? 99 : rng_below(r, 12);
        for (int l = 0; l < 32; ++l) {
            if (m == 99) q[l] = (int8_t)(-128 + (int)rng_below(r, 9));
            else if (m == 0) q[l] = -128;
            else if (m == 1) q[l] = 127;
            else if (m == 2) q[l] = (int8_t)((int)rng_below(r, 5) - 2);
        }
    }
    int sp = 0;
    uint8_t * dreg = qa + (size_t)tsr_qa_qs_lines(kind, K) * 64;
    const uint32_t bsz = tsr_qa_block_size(kind);
    for (uint32_t b = 0; b < K / bsz; ++b) {
        const uint32_t v = extreme ? f2u((float)ldexp(1.0, -(int)rng_below(r, 12))) : rand_dx_bits(r, &sp);
        put_f32(dreg + 4 * b, u2f(v));
    }
    if (kind == TSR_QA_A256) {
        uint8_t * bs = dreg + (size_t)tsr_qa_d_lines(kind, K) * 64;
        for (uint32_t j = 0; j < K / 16; ++j) {
            int32_t v;
            if (arb_bs) {
                const uint32_t k = rng_below(r, 8);
                v = k == 0 ? 32767 : k == 1 ? -32768 : (int32_t)(int16_t)rng_u32(r);
            } else {
                v = 0;
                for (int l = 0; l < 16; ++l) v += qs[16 * j + l];
            }
            bs[2 * j] = (uint8_t)v;
            bs[2 * j + 1] = (uint8_t)((uint32_t)v >> 8);
        }
    }
    return sp;
}

// Old value for ACC / guard lines: mostly N(0,1), 1/20 special.
static uint32_t old_bits(rng_t * r, int * sp) {
    if (rng_below(r, 20) != 0) return f2u((float)rng_normal(r));
    *sp = 1;
    const uint32_t sign = rng_u32(r) & 0x80000000u;
    switch (rng_below(r, 6)) {
    case 0:  return sign;                                              // +-0
    case 1:  return sign | (1 + rng_below(r, 0x7FFFFF));               // subnormal
    case 2:  return sign | 0x7F800000u;                                // +-Inf
    case 3:  return sign | 0x7FC00000u | rng_below(r, 1u << 22);       // quiet NaN with payload
    case 4:  return sign | (0x7F800001u + rng_below(r, 0x3FFFFE));     // signalling NaN
    default: return sign | 0x7F7FFFFFu;                                // max finite
    }
}

typedef struct {
    uint64_t  w_off;
    uint8_t * blocks;              // N rows of ggml blocks
} m3_tensor;

typedef struct {                   // one MVM output (dst span) and the MVMs that write it
    uint32_t  fmt, K, N, R, rb, yl, ys, w6, n_w, acc, pre, g0, g1, top;
    uint32_t  spm;                 // first line of the region; dst = spm + g0
    uint32_t  lines;               // g0 + span + g1
    uint32_t  span;                // (T - 1) * ys + yl
    m3_tensor w[3];
    uint8_t * img;                 // expected region content (lines * 64 bytes)
} m3_out;

static uint32_t pick_n_m3(gen_t * g, int small) {
    rng_t * r = &g->r;
    const uint32_t C = g->o->cfg.num_ch;
    if (small) return rng_below(r, 2 * C + 2);
    switch (rng_below(r, 16)) {
    case 0:  return 0;
    case 1:  return 1;
    case 2:  return C > 1 ? 1 + rng_below(r, C - 1) : 1 + rng_below(r, 3);
    case 3:  return 1001 + rng_below(r, 3000);
    default: return 1 + rng_below(r, g->o->max_n);
    }
}

static void out_shape(gen_t * g, m3_out * o, uint32_t fmt, uint32_t K, uint32_t N, uint32_t T, int may_chain) {
    rng_t * r = &g->r;
    memset(o, 0, sizeof(*o));
    o->fmt = fmt;
    o->K = K;
    o->N = N;
    o->R = tsr_ceil_div_u32(N, g->o->cfg.num_ch);
    o->rb = tsr_row_beats(fmt, K);
    o->yl = tsr_ceil_div_u32(4 * N, 64);
    o->ys = o->yl + (rng_below(r, 3) == 0 ? 1 + rng_below(r, 3) : 0);
    o->w6 = (o->ys == o->yl && rng_below(r, 2)) ? 0 : o->ys;
    o->n_w = (may_chain && rng_below(r, 4) == 0) ? 2 + rng_below(r, 2) : 1;
    o->acc = rng_below(r, 3) == 0;
    o->pre = o->acc || N == 0 || rng_below(r, 2) == 0;
    o->g0 = o->pre ? rng_below(r, 3) : 0;
    o->g1 = o->pre ? rng_below(r, 3) + (N == 0) : 0;
    o->top = rng_below(r, 8) == 0;
    o->span = (T - 1) * o->ys + o->yl;
    o->lines = o->g0 + o->span + o->g1;
}

// Allocates and fills the tensors of output o (weights written to every channel's DRAM).
static uint32_t out_tensors(gen_t * g, m3_out * o, int mode) {
    const tsr_config * c = &g->o->cfg;
    const size_t rbytes = (size_t)(o->K / tsr_fmt_block_size(o->fmt)) * tsr_fmt_ggml_block_bytes(o->fmt);
    uint8_t * row = (uint8_t *)xmalloc((size_t)o->rb * 64);
    uint32_t fl = 0;
    for (uint32_t w = 0; w < o->n_w; ++w) {
        const uint64_t n = (uint64_t)o->R * o->rb;
        int at_lim;
        o->w[w].w_off = heap_take(g, n, &at_lim);
        g->cov[C3_WLIM] += at_lim;
        g->cov[C3_W4K] += n > 0 && (o->w[w].w_off % 64) + n > 64;
        g->wch0 += n;
        g->wall += (uint64_t)o->N * o->rb;
        o->w[w].blocks = (uint8_t *)xmalloc(rbytes * o->N);
        fl |= gen_tensor_m3(g, o->fmt, o->N, o->K, mode, o->w[w].blocks);
        for (uint32_t r = 0; r < o->N; ++r) {
            uint32_t ch;
            uint64_t off;
            tsr_row_location(o->fmt, o->N, o->K, c->num_ch, r, &ch, &off);
            tsr_repack_row(o->fmt, o->w[w].blocks + rbytes * r, o->K, row);
            memcpy(tsr_sim_dram(g->sim, ch) + (o->w[w].w_off + off) * 64, row, (size_t)o->rb * 64);
        }
    }
    free(row);
    return fl;
}

// Expected region content of output o after its MVMs, from the T QA images (stride ql bytes*64).
static void out_expect(gen_t * g, m3_out * o, const uint8_t * qa, uint32_t ql, uint32_t T) {
    const size_t rbytes = (size_t)(o->K / tsr_fmt_block_size(o->fmt)) * tsr_fmt_ggml_block_bytes(o->fmt);
    uint8_t * row = (uint8_t *)xmalloc((size_t)o->rb * 64 + 64);
    for (uint32_t t = 0; t < T; ++t) {
        const uint8_t * qt = qa + (size_t)t * ql * 64;
        uint8_t * yt = o->img + ((size_t)o->g0 + (size_t)t * o->ys) * 64;
        for (uint32_t n = 0; n < o->N; ++n) {
            float y = 0.0f;
            for (uint32_t w = 0; w < o->n_w; ++w) {
                const uint8_t * blk = o->w[w].blocks + rbytes * n;
                tsr_repack_row(o->fmt, blk, o->K, row);
                const float d = tsr_dot_row(o->fmt, row, o->K, qt);
                y = (w == 0 && !o->acc) ? d : tsr_fadd(w == 0 ? get_f32(yt + 4 * n) : y, d);
                if (o->fmt == TSR_FMT_Q4_K) {
                    for (uint32_t s = 0; s < o->K / 256; ++s) {
                        const int32_t is = q4k_isum(blk + (size_t)s * 144, (const int8_t *)qt + 256 * s);
                        if ((int64_t)(float)is != (int64_t)is) g->cov[C3_ISUM]++;
                    }
                }
            }
            put_f32(yt + 4 * n, y);
        }
        memset(yt + 4 * (size_t)o->N, 0, (size_t)o->yl * 64 - 4 * (size_t)o->N);
    }
    free(row);
}

// Region preload (guards, gap lines, ACC old values) from channel cy; image = loaded content.
static void out_preload(gen_t * g, m3_out * o, uint32_t cy) {
    int sp = 0;
    for (uint32_t i = 0; i < o->lines * 16; ++i) put_f32(o->img + 4 * i, u2f(old_bits(&g->r, &sp)));
    g->cov[C3_ACCSP] += o->acc && sp;
    g->cov[C3_GUARD] += o->g0 + o->g1 > 0;
    const uint64_t b = io_alloc(g, cy, o->lines);
    memcpy(tsr_sim_dram(g->sim, cy) + b * 64, o->img, (size_t)o->lines * 64);
    mutate_regs_m2(g);
    emit(g, tsr_enc_ld(cy, o->spm, (uint32_t)b, o->lines), 2);
}

static void out_store(gen_t * g, m3_out * o, uint32_t T) {
    rng_t * r = &g->r;
    const uint32_t C = g->o->cfg.num_ch;
    mutate_regs_m2(g);
    if (o->pre || (o->ys == o->yl && (T == 1 || rng_below(r, 2)))) {
        if (o->lines == 0) return;
        const uint32_t cs = rng_below(r, C);
        const uint64_t ob = io_alloc(g, cs, o->lines);
        emit(g, tsr_enc_st(cs, o->spm, (uint32_t)ob, o->lines), 2);
        add_check(g, cs, ob, o->lines, o->img);
        o->img = NULL;
        return;
    }
    for (uint32_t t = 0; t < T; ++t) {
        if (o->yl == 0) break;
        const uint32_t cs = rng_below(r, C);
        const uint64_t ob = io_alloc(g, cs, o->yl);
        emit(g, tsr_enc_st(cs, o->spm + o->g0 + t * o->ys, (uint32_t)ob, o->yl), 2);
        uint8_t * w = (uint8_t *)xmalloc((size_t)o->yl * 64);
        memcpy(w, o->img + ((size_t)o->g0 + (size_t)t * o->ys) * 64, (size_t)o->yl * 64);
        add_check(g, cs, ob, o->yl, w);
    }
}

static void emit_mvm_m3(gen_t * g, const m3_out * o, uint32_t w, uint32_t qa, uint32_t T) {
    const uint32_t fl = (o->acc || w > 0) ? TSR_MVM_FLAG_ACC : 0;
    if (rng_below(&g->r, 2)) mutate_regs_m2(g);
    emit(g, tsr_enc_mvm(o->fmt, fl, o->spm + o->g0, qa, (uint32_t)o->w[w].w_off, o->N, o->K, T, o->w6), 3);
    g->cov[C3_A1REL] += g->last_base[0] != 0;
    g->cov[C3_A2REL] += g->last_base[1] != 0;
    g->cov[C3_A3REL] += g->last_base[2] != 0;
    g->n_mvm++;
}

static void out_cov(gen_t * g, const m3_out * o, uint32_t T, uint32_t fl) {
    const uint32_t C = g->o->cfg.num_ch;
    g->cov[o->fmt ? C3_Q4 : C3_Q8] += o->n_w;
    g->cov[C3_N0] += o->N == 0;
    g->cov[C3_N1] += o->N == 1;
    g->cov[C3_NLTC] += o->N > 0 && o->N < C;
    g->cov[C3_ODDN] += o->N & 1;
    g->cov[C3_NMIX] += C > 1 && o->N > o->R && o->N % 16 && o->R % 16;
    g->cov[C3_NBIG] += o->N > 1000;
    g->cov[C3_TGT1] += T > 1;
    g->cov[C3_TMAX] += T == g->o->cfg.t_max;
    g->cov[C3_YZERO] += o->w6 == 0;
    g->cov[C3_YGAP] += o->ys != o->yl && T > 1;
    g->cov[C3_ACC] += o->acc;
    g->cov[C3_CHAIN] += o->n_w > 1;
    g->cov[C3_YEND] += o->spm + o->g0 + o->span == g->spm_lines;
    g->cov[C3_DZ] += (fl & WF_DZ) != 0;
    g->cov[C3_DSUB] += (fl & WF_DSUB) != 0;
    g->cov[C3_DINF] += (fl & WF_DINF) != 0;
    g->cov[C3_DNAN] += (fl & WF_DNAN) != 0;
    g->cov[C3_DMAX] += (fl & WF_DMAX) != 0;
    g->cov[C3_Q128] += (fl & WF_Q128) != 0;
    g->cov[C3_SC63] += (fl & WF_SC63) != 0;
}

// One MVM group. Returns 0 if nothing fits any more.
static int gen_mvm_group_m3(gen_t * g) {
    const tsr_config * c = &g->o->cfg;
    rng_t * r = &g->r;
    const uint32_t C = c->num_ch;
    const uint32_t kcap = g->o->max_k < c->k_max ? g->o->max_k : c->k_max;
    const uint64_t wleft = g->wbudget > g->wch0 ? g->wbudget - g->wch0 : 0;
    const int big = g->wbudget > g->o->max_wbeats && g->wch0 == 0;   // first group of a big trace: one long stream
    uint32_t fmt = 0, K = 0, T = 0, kind = 0, ql = 0, n_out = 0, xl = 0, xs = 0, kclass = 0;
    uint32_t kind2 = 0, ql2 = 0;
    int raw = 0, st2 = 0, qa_top = 0, fits = 0;
    m3_out out[3];                      // out[2] = stage 2
    for (int attempt = 0; attempt < 64 && !fits; ++attempt) {
        fmt = (kcap >= 256 && rng_below(r, 2)) ? TSR_FMT_Q4_K : TSR_FMT_Q8_0;
        const uint32_t bs = tsr_fmt_block_size(fmt), S = tsr_fmt_chunk_blocks(fmt);
        const uint32_t nmax = kcap / bs ? kcap / bs : 1;
        T = 1 + rng_below(r, c->t_max);
        uint32_t N = pick_n_m3(g, 0);
        const uint32_t ksel = rng_below(r, 16);
        kclass = 0;
        if (N > 1000) {
            K = bs * (1 + rng_below(r, nmax < 2 ? nmax : 2));
        } else if (ksel <= 1 || nmax == 1) {
            K = bs;
            kclass = 1;
        } else if (ksel == 2) {
            K = c->k_max;                                       // activation buffer at full depth
            kclass = 2;
            N = pick_n_m3(g, 1);
            if (rng_below(r, 2)) T = c->t_max;                   // every token slot of the buffer
        } else if (ksel <= 5) {                                  // partial last chunk
            uint32_t nb = 1 + rng_below(r, nmax);
            if (nb % S == 0) nb -= 1;
            if (nb == 0) nb = 1;
            K = bs * nb;
            kclass = 3;
        } else if (ksel <= 7 && kcap >= 1024) {
            K = 1024 * (1 + rng_below(r, kcap / 1024));
        } else {
            K = bs * (1 + rng_below(r, (ksel & 1) && nmax > 8 ? 8 : nmax));
        }
        if (big) {                      // 50-90 % of the budget in one MVM, K up to 5120
            K = bs * (1 + rng_below(r, nmax < 5120 / bs ? nmax : 5120 / bs));
            const uint64_t n = (uint64_t)C * (wleft * (5 + rng_below(r, 5)) / 10 / tsr_row_beats(fmt, K));
            const uint64_t nspm = 16ull * (spm_room(g) / 4) / T;        // y span stays below a quarter of the SPM
            N = (uint32_t)(n < nspm ? n : nspm);
            kclass = 0;
        }
        kind = tsr_qa_kind_for_fmt(fmt);
        ql = tsr_qa_lines(kind, K);
        st2 = !big && kclass != 2 && N <= 1000 && c->k_max >= 512 && rng_below(r, 6) == 0;
        if (st2) N = (fmt == TSR_FMT_Q4_K || rng_below(r, 2)) ? 256 * (1 + rng_below(r, 2)) : 32 * (1 + rng_below(r, 8));
        n_out = big ? 1 : 1 + (rng_below(r, 4) == 0);
        out_shape(g, &out[0], fmt, K, N, T, !big);
        if (n_out > 1) out_shape(g, &out[1], fmt, K, pick_n_m3(g, kclass == 2), T, 1);
        if (st2) {
            kind2 = (N % 256 == 0 && rng_below(r, 2)) ? TSR_QA_A256 : TSR_QA_A32;
            ql2 = tsr_qa_lines(kind2, N);
            out_shape(g, &out[2], kind2, N, 1 + rng_below(r, g->o->max_n), T, 0);
        }
        raw = rng_below(r, 5) == 0 || (kclass == 2 && T > 2);
        qa_top = rng_below(r, 8) == 0;
        xl = K / 16;
        xs = xl + (rng_below(r, 3) == 0 ? 1 + rng_below(r, 3) : 0);
        const uint64_t xspan = (uint64_t)(T - 1) * xs + xl;
        uint64_t wneed = 0, heap = 0, spm = (raw ? 0 : xspan + 2) + (uint64_t)T * ql + 2 + 16, io = (raw ? (uint64_t)T * ql : xspan) + 1024;
        for (uint32_t k = 0; k < 3; ++k) {
            if (k == 1 && n_out < 2) continue;
            if (k == 2 && !st2) continue;
            const m3_out * o = &out[k];
            wneed += (uint64_t)o->n_w * o->R * o->rb;
            heap += (uint64_t)o->n_w * (o->R * (uint64_t)o->rb + 40);
            spm += o->lines + 2 + (k == 2 ? (uint64_t)T * ql2 + 2 : 0);
            io += 2ull * o->lines + (uint64_t)T * o->yl + 512;
        }
        uint64_t io_min = ~0ull;
        for (uint32_t ch = 0; ch < C; ++ch) io_min = io_room(g, ch) < io_min ? io_room(g, ch) : io_min;
        fits = wneed <= wleft && g->heap_next + heap + 64 <= min_top(g) && spm <= spm_room(g) && io <= io_min;
    }
    if (!fits) return 0;
    const int extreme = rng_below(r, 8) == 0;
    const int mode = extreme ? 2 : (rng_below(r, 4) == 0 ? 1 : 0);
    g->cov[C3_K32] += K == 32;
    g->cov[C3_K256] += K == 256;
    g->cov[C3_KPART] += kclass == 3 || (K / tsr_fmt_block_size(fmt)) % tsr_fmt_chunk_blocks(fmt) != 0;
    g->cov[C3_KMAX] += K == c->k_max;
    g->cov[C3_KMAXT] += K == c->k_max && T == c->t_max;
    g->cov[C3_SHARE] += n_out > 1;
    g->cov[C3_STAGE2] += st2;
    g->cov[C3_RAWQA] += raw;

    // Weights.
    uint32_t wfl[3] = { 0, 0, 0 };
    for (uint32_t k = 0; k < 3; ++k) {
        if ((k == 1 && n_out < 2) || (k == 2 && !st2)) continue;
        wfl[k] = out_tensors(g, &out[k], mode);
    }

    // Activations: LD x -> QACT, or LD of raw QA images.
    uint8_t * qa = (uint8_t *)xmalloc((size_t)T * ql * 64);
    uint32_t spm_qa;
    if (raw) {
        const int arb = kind == TSR_QA_A256 && rng_below(r, 2) == 0;
        int sp = 0;
        for (uint32_t t = 0; t < T; ++t) sp |= gen_raw_qa(r, kind, K, qa + (size_t)t * ql * 64, arb, extreme);
        g->cov[C3_RAWBS] += arb;
        g->cov[C3_XSP] += sp;
        g->cov[C3_XEXT] += extreme;
        const uint32_t cq = rng_below(r, C);
        const uint64_t qb = io_alloc(g, cq, (uint64_t)T * ql);
        memcpy(tsr_sim_dram(g->sim, cq) + qb * 64, qa, (size_t)T * ql * 64);
        spm_qa = spm_take(g, T * ql, qa_top);
        emit_loads(g, cq, spm_qa, qb, T, ql, ql);
    } else {
        float * x = (float *)xmalloc(sizeof(float) * K * T);
        int sp = 0;
        for (uint32_t t = 0; t < T; ++t) {
            if (extreme) gen_x_extreme(r, x + (size_t)t * K, K);
            else sp |= gen_x(r, x + (size_t)t * K, K, g->o->special);
        }
        g->cov[C3_XSP] += sp;
        g->cov[C3_XEXT] += extreme;
        for (uint32_t t = 0; t < T; ++t) tsr_qact(kind, x + (size_t)t * K, K, qa + (size_t)t * ql * 64);
        const uint32_t cx = rng_below(r, C);
        const uint64_t span = (uint64_t)(T - 1) * xs + xl;
        const uint64_t xb = io_alloc(g, cx, span);
        for (uint32_t t = 0; t < T; ++t) {
            uint8_t * d = tsr_sim_dram(g->sim, cx) + (xb + (uint64_t)t * xs) * 64;
            for (uint32_t k = 0; k < K; ++k) put_f32(d + 4 * k, x[(size_t)t * K + k]);
        }
        const uint32_t spm_x = spm_take(g, (uint32_t)span, 0);
        spm_qa = spm_take(g, T * ql, qa_top);
        emit_loads(g, cx, spm_x, xb, T, xs, xl);
        const int zx = xs == xl && rng_below(r, 2);
        g->cov[C2_ZSTRIDE] += zx;
        mutate_regs_m2(g);
        emit(g, tsr_enc_qact(kind, spm_qa, spm_x, K, T, zx ? 0 : xs), 2);
        g->cov[C2_A32 + kind]++;
        free(x);
    }
    g->cov[C3_QAEND] += spm_qa + T * ql == g->spm_lines;

    // Output regions (preloaded with old values / guard lines where o->pre).
    for (uint32_t k = 0; k < 3; ++k) {
        if ((k == 1 && n_out < 2) || (k == 2 && !st2)) continue;
        m3_out * o = &out[k];
        o->spm = spm_take(g, o->lines, o->top);
        o->img = (uint8_t *)xmalloc((size_t)o->lines * 64 + 64);
        memset(o->img, 0, (size_t)o->lines * 64);
        if (o->pre) out_preload(g, o, rng_below(r, C));
        out_cov(g, o, T, wfl[k]);
    }

    // MVMs of out[0] / out[1], interleaved (each output's ACC chain stays in order).
    uint32_t done[2] = { 0, 0 };
    while (done[0] < out[0].n_w || (n_out > 1 && done[1] < out[1].n_w)) {
        uint32_t k = (n_out > 1 && done[1] < out[1].n_w && (done[0] == out[0].n_w || rng_below(r, 2))) ? 1 : 0;
        emit_mvm_m3(g, &out[k], done[k]++, spm_qa, T);
    }
    for (uint32_t k = 0; k < n_out; ++k) out_expect(g, &out[k], qa, ql, T);

    // N = 0 MVM in between: writes nothing (dst over out[0]'s span, or at the SPM end).
    if (rng_below(r, 10) == 0) {
        const int at_end = rng_below(r, 3) == 0;
        const uint32_t w6 = at_end ? 0 : out[0].w6 ? out[0].ys : 0;
        const uint32_t dst = at_end ? g->spm_lines : out[0].spm + out[0].g0;
        const int wl = rng_below(r, 2);
        const uint32_t woff = wl ? (uint32_t)g->lim : (uint32_t)(g->w_lo + rng_below(r, 1u << 20));
        if (rng_below(r, 2)) mutate_regs_m2(g);
        emit(g, tsr_enc_mvm(fmt, rng_below(r, 2) ? TSR_MVM_FLAG_ACC : 0, dst, spm_qa, woff, 0, K, T, w6), 3);
        g->cov[C3_N0]++;
        g->cov[C3_N0LIM] += wl;
        g->cov[fmt ? C3_Q4 : C3_Q8]++;
        g->n_mvm++;
    }

    // Stage 2: QACT of out[0]'s y (K2 = N0, src_stride = its dst stride) -> MVM -> out[2].
    if (st2) {
        const m3_out * o0 = &out[0];
        m3_out * o2 = &out[2];
        const uint32_t K2 = o0->N;
        float * x2 = (float *)xmalloc(sizeof(float) * K2 * T);
        for (uint32_t t = 0; t < T; ++t)
            for (uint32_t k = 0; k < K2; ++k) x2[(size_t)t * K2 + k] = get_f32(o0->img + ((size_t)o0->g0 + (size_t)t * o0->ys) * 64 + 4 * k);
        uint8_t * qa2 = (uint8_t *)xmalloc((size_t)T * ql2 * 64);
        for (uint32_t t = 0; t < T; ++t) tsr_qact(kind2, x2 + (size_t)t * K2, K2, qa2 + (size_t)t * ql2 * 64);
        const uint32_t spm_qa2 = spm_take(g, T * ql2, 0);
        mutate_regs_m2(g);
        emit(g, tsr_enc_qact(kind2, spm_qa2, o0->spm + o0->g0, K2, T, o0->w6 ? o0->ys : 0), 2);
        g->cov[C2_A32 + kind2]++;
        emit_mvm_m3(g, o2, 0, spm_qa2, T);
        out_expect(g, o2, qa2, ql2, T);
        free(qa2);
        free(x2);
    }

    for (uint32_t k = 0; k < 3; ++k) {
        if ((k == 1 && n_out < 2) || (k == 2 && !st2)) continue;
        out_store(g, &out[k], T);
        free(out[k].img);
        for (uint32_t w = 0; w < out[k].n_w; ++w) free(out[k].w[w].blocks);
    }
    free(qa);
    return 1;
}

static int gen_group_m3(gen_t * g, uint32_t k) {
    if (k == 0) return gen_mvm_group_m3(g);
    switch (rng_below(&g->r, 8)) {
    case 0:  return gen_copy_group_m2(g) || gen_mvm_group_m3(g);
    case 1:  return gen_qact_group_m2(g) || gen_mvm_group_m3(g);
    default: return gen_mvm_group_m3(g) || gen_copy_group_m2(g);
    }
}

// A legal MVM (absolute operands) to be broken by an error tail.
typedef struct { uint32_t fmt, fl, dst, x, woff, N, K, T, w6; } mvm_ops;

static uint64_t mvm_dst_ext(const mvm_ops * m) {
    const uint64_t yl = tsr_ceil_div_u64(4ull * m->N, 64);
    return (uint64_t)(m->T - 1) * (m->w6 ? m->w6 : yl) + yl;
}

static void legal_mvm(gen_t * g, mvm_ops * m) {
    rng_t * r = &g->r;
    const tsr_config * c = &g->o->cfg;
    const uint32_t S = g->spm_lines;
    m->fmt = rng_below(r, 2);
    const uint32_t bs = tsr_fmt_block_size(m->fmt);
    const uint32_t nmax = c->k_max / bs;
    m->K = bs * (1 + rng_below(r, nmax < 8 ? nmax : 8));
    m->T = 1 + rng_below(r, c->t_max);
    m->N = rng_below(r, 6) == 0 ? 0 : 1 + rng_below(r, 64);
    m->fl = rng_below(r, 2) ? TSR_MVM_FLAG_ACC : 0;
    const uint32_t yl = tsr_ceil_div_u32(4 * m->N, 64);
    m->w6 = rng_below(r, 2) ? 0 : yl + rng_below(r, 4);
    const uint64_t de = mvm_dst_ext(m);
    const uint64_t xe = (uint64_t)m->T * tsr_qa_lines(tsr_qa_kind_for_fmt(m->fmt), m->K);
    m->dst = rng_below(r, (uint32_t)(S - de + 1));
    m->x = rng_below(r, (uint32_t)(S - xe + 1));
    const uint64_t we = (uint64_t)tsr_ceil_div_u32(m->N, c->num_ch) * tsr_row_beats(m->fmt, m->K);
    m->woff = rng_below(r, (uint32_t)(g->lim - we + 1));
}

// Operand values beyond the limit for SPM (A1 = dst, A2 = x; extent ext) / DRAM (A3). Variant v:
// 0 just past the limit, 1 far beyond it, 2 so close to 2^32 that a + e wraps 32 bits; -1 random.
static uint32_t past_spm(gen_t * g, uint64_t ext, int v) {
    rng_t * r = &g->r;
    const uint64_t S = g->spm_lines;
    if (v < 0) v = (int)rng_below(r, 3);
    switch (v) {
    case 0:  return ext == 0 ? (uint32_t)(S + 1 + rng_below(r, 64))
                             : (uint32_t)(S - (ext < S ? ext : S) + 1 + rng_below(r, (uint32_t)(ext < 16 ? ext : 16)));
    case 1:  return (uint32_t)(S + 1 + rng_below(r, 1u << 20));
    default:
        g->cov[C3_WRAP] += ext > 0;
        return ext == 0 ? 0xFFFFFFFFu - rng_below(r, 16)
                        : (uint32_t)(0x100000000ull - 1 - rng_below(r, (uint32_t)(ext < 64 ? ext : 64)));
    }
}
static uint32_t past_dram(gen_t * g, uint64_t ext, int v) {
    rng_t * r = &g->r;
    if (v < 0) v = (int)rng_below(r, 3);
    switch (v) {
    case 0:  return (uint32_t)(g->lim - (ext < g->lim ? ext : g->lim) + 1 + rng_below(r, 16));
    case 1:  return (uint32_t)(g->lim + 1 + rng_below(r, 1u << 16));
    default:
        g->cov[C3_WRAP] += ext > 0;
        return ext == 0 ? 0xFFFFFFFFu - rng_below(r, 16)
                        : (uint32_t)(0x100000000ull - 1 - rng_below(r, (uint32_t)(ext < 64 ? ext : 64)));
    }
}

// MVM error tails with their variants (--errors 100 cycles through this list, then the M2 tails).
static const struct { int tail, var; } m3_tail_cycle[] = {
    { T3_RSV, 0 }, { T3_FMT, 0 }, { T3_K, 0 }, { T3_T, 0 }, { T3_SPM_DST, 0 }, { T3_SPM_X, 0 }, { T3_DRAM, 0 },
    { T3_RSV, 1 }, { T3_FMT, 1 }, { T3_K, 1 }, { T3_T, 1 }, { T3_SPM_DST, 1 }, { T3_SPM_X, 1 }, { T3_DRAM, 1 },
    { T3_RSV, 2 }, { T3_K, 2 }, { T3_SPM_DST, 2 }, { T3_SPM_X, 2 }, { T3_DRAM, 2 },
    { T3_RSV, 3 }, { T3_K, 3 }, { T3_SPM_DST, 3 }, { T3_DRAM, 3 },
};
#define M3_TAIL_CYCLE ((unsigned)(sizeof(m3_tail_cycle) / sizeof(m3_tail_cycle[0])))

// Appends a faulting MVM (error tail `tail`, T3_*; variant `var`, -1 = random). Returns the
// expected error code; *err_pc = its index; *err_info = expected ERR_INFO.
//   T3_RSV     var: reserved bit in 0 flags[3:1], 1 rd, 2 w5[31:20], 3 w7
//   T3_K       var: 0 K = 0, 1 not a block multiple, 2 a multiple > K_MAX, 3 16-bit field extremes
//   T3_T       var: 0 T = 0, 1 T > T_MAX (when T_MAX < 15)
//   T3_SPM_DST var: 0 (T-1) x dst_stride = 2^32 + small (T >= 3), 1 N = k x 2^30 + small (4N wraps
//                   32 bits to a small value),
//                   2 N > 16 x SPM lines, 3 dst past the end (just / far / 32-bit wrap)
//   T3_SPM_X   var: x 0 just past the end, 1 far beyond, 2 x + T x QA_LINES wraps 32 bits
//   T3_DRAM    var: W_OFF 0 just past (W_OFF + R x ROW_BEATS = limit + 1..16), 1 far beyond,
//                   2 wraps 32 bits, 3 N = 0 with W_OFF > limit
// Lower-priority violations (later checks) are added at random.
static uint32_t gen_error_tail_m3(gen_t * g, int tail, int var, uint32_t * err_pc, int64_t * err_info) {
    rng_t * r = &g->r;
    const tsr_config * c = &g->o->cfg;
    mvm_ops m;
    legal_mvm(g, &m);
    *err_pc = g->n;
    *err_info = 0;
    uint32_t code = TSR_ERR_ILLEGAL_OP, rsv_w0 = 0, rsv_w5 = 0, rsv_w7 = 0;
    int lo_k = 0, lo_t = 0, lo_dst = 0, lo_x = 0, lo_w = 0;     // violations (lower priority ones random)
    int vk = -1, vt = -1, vd = -1, vx = -1, vw = -1;            // their variants
    switch (tail) {
    case T3_RSV: {
        const int k = var >= 0 ? var : (int)rng_below(r, 4);
        if (k == 0) rsv_w0 = 1u << (13 + rng_below(r, 3));
        else if (k == 1) rsv_w0 = 1u << (28 + rng_below(r, 4));
        else if (k == 2) rsv_w5 = 1u << (20 + rng_below(r, 12));
        else rsv_w7 = 1u << rng_below(r, 32);
        if (rng_below(r, 2)) m.fmt = 2 + rng_below(r, 14);
        lo_k = rng_below(r, 3) == 0;
        lo_t = rng_below(r, 3) == 0;
        lo_dst = rng_below(r, 3) == 0;
        lo_w = rng_below(r, 3) == 0;
        code = TSR_ERR_ILLEGAL_OP;
        break;
    }
    case T3_FMT:
        m.fmt = 2 + rng_below(r, 14);
        lo_k = rng_below(r, 3) == 0;
        lo_t = rng_below(r, 3) == 0;
        lo_dst = rng_below(r, 4) == 0;
        lo_x = rng_below(r, 4) == 0;
        lo_w = rng_below(r, 4) == 0;
        code = TSR_ERR_BAD_FMT;
        break;
    case T3_K:
        lo_k = 1;
        vk = var;
        lo_t = rng_below(r, 3) == 0;
        lo_dst = rng_below(r, 4) == 0;
        lo_w = rng_below(r, 4) == 0;
        code = TSR_ERR_K_RANGE;
        break;
    case T3_T:
        lo_t = 1;
        vt = var;
        lo_dst = rng_below(r, 3) == 0;
        lo_x = rng_below(r, 4) == 0;
        lo_w = rng_below(r, 3) == 0;
        code = TSR_ERR_T_RANGE;
        break;
    case T3_SPM_DST:
        lo_dst = 1;
        vd = var;
        lo_x = rng_below(r, 3) == 0;
        lo_w = rng_below(r, 3) == 0;
        code = TSR_ERR_SPM_RANGE;
        break;
    case T3_SPM_X:
        lo_x = 1;
        vx = var;
        lo_w = rng_below(r, 3) == 0;
        code = TSR_ERR_SPM_RANGE;
        break;
    default:
        lo_w = 1;
        vw = var;
        code = TSR_ERR_DRAM_RANGE;
        break;
    }
    const uint32_t bs = tsr_fmt_valid(m.fmt) ? tsr_fmt_block_size(m.fmt) : 32;
    if (lo_k) {
        switch (vk >= 0 ? vk : (int)rng_below(r, 4)) {
        case 0:  m.K = 0; break;
        case 1:  m.K = m.fmt == TSR_FMT_Q4_K ? 256 * rng_below(r, 4) + 32 * (1 + rng_below(r, 7))
                                             : 32 * rng_below(r, 4) + 1 + rng_below(r, 31); break;
        case 2:  m.K = c->k_max + bs * (1 + rng_below(r, 4)); break;
        default: { static const uint32_t big[] = { 0xFFFF, 0xFFE0, 0xFF00, 0xFFFE }; m.K = big[rng_below(r, 4)]; } break;
        }
        if (m.K > 0xFFFF || (m.K <= c->k_max && m.K % bs == 0 && m.K != 0)) m.K = 0;
    }
    if (lo_t) {
        const int v = vt >= 0 ? vt : (int)rng_below(r, 2);
        m.T = (c->t_max < 15 && v == 1) ? c->t_max + 1 + rng_below(r, 15 - c->t_max) : 0;
    }
    if (lo_dst) {
        int v = vd >= 0 ? vd : (int)rng_below(r, 4);
        if (v == 0 && tail == T3_SPM_DST) {
            if (c->t_max >= 3) m.T = 3 + rng_below(r, c->t_max - 2);
            else v = 3;
        }
        if (v == 0 && m.T >= 3 && m.T <= 15) {
            // (T - 1) * dst_stride >= 2^32: a 32-bit product would wrap to a small extent
            m.w6 = (uint32_t)((0x100000000ull + m.T - 2) / (m.T - 1)) + rng_below(r, 4);
            m.dst = rng_below(r, 64);
            g->cov[C3_WRAP]++;
        } else if (v == 1 || v == 2) {
            // y_lines > SPM lines; for N >= 2^30, 4N does not fit 32 bits
            m.N = v == 1 ? 0x40000000u * (1 + rng_below(r, 3)) + rng_below(r, 1024)    // 4N mod 2^32 < 4096
                         : 16 * g->spm_lines + 1 + rng_below(r, 4096);
            g->cov[C3_WRAP] += v == 1;
            m.w6 = 0;
            m.dst = rng_below(r, 64);
        } else {
            m.dst = past_spm(g, (m.T >= 1 && m.T <= 15) ? mvm_dst_ext(&m) : 0, -1);
        }
    }
    if (lo_x) {
        const uint64_t xe = (tsr_fmt_valid(m.fmt) && m.K && m.K % bs == 0 && m.T >= 1 && m.T <= 15)
                                ? (uint64_t)m.T * tsr_qa_lines(tsr_qa_kind_for_fmt(m.fmt), m.K) : 0;
        m.x = past_spm(g, xe, vx);
    }
    if (lo_w) {
        const uint64_t we = (tsr_fmt_valid(m.fmt) && m.K && m.K % bs == 0)
                                ? tsr_ceil_div_u64(m.N, c->num_ch) * tsr_row_beats(m.fmt, m.K) : 0;
        const int v = vw >= 0 ? vw : (code == TSR_ERR_DRAM_RANGE ? (int)rng_below(r, 4) : (int)rng_below(r, 3));
        if (v == 3) {
            m.N = 0;                                     // e = 0: only a > limit fails
            m.woff = (uint32_t)(g->lim + 1 + rng_below(r, 64));
        } else {
            m.woff = past_dram(g, we, v);
        }
    }
    tsr_insn i = tsr_enc_mvm(m.fmt, m.fl, m.dst, m.x, m.woff, m.N, m.K, m.T, m.w6);
    i.w[0] |= rsv_w0;
    i.w[5] |= rsv_w5;
    i.w[7] |= rsv_w7;
    if (code == TSR_ERR_SPM_RANGE) *err_info = lo_dst ? m.dst : m.x;
    if (code == TSR_ERR_DRAM_RANGE) *err_info = m.woff;
    emit_tail(g, i, 3);
    g->cov[C3_ERR]++;
    return code;
}

static int mkdir_p(const char * path) {
    char * p = strdup(path);
    if (!p) return -1;
    const size_t n = strlen(p);
    for (size_t i = 1; i <= n; ++i) {
        if (p[i] == '/' || p[i] == '\0') {
            const char ch = p[i];
            p[i] = '\0';
            if (mkdir(p, 0755) != 0 && errno != EEXIST) {
                fprintf(stderr, "tsr-trace-gen: mkdir %s: %s\n", p, strerror(errno));
                free(p);
                return -1;
            }
            p[i] = ch;
        }
    }
    free(p);
    return 0;
}

// --ops m3 weight-traffic totals over the traces written (MVM weight beats; "busiest channel" is
// channel 0, which holds R = ceil(N/C) rows of every tensor).
static struct {
    unsigned long long ch0, all, max, over, mvms;
} m3_stats;
#define M3_BIG_BEATS 20000ull   // traces above this many weight beats on the busiest channel are reported

// Generates, runs, verifies and writes trace `idx`. Returns 0 on success.
static int gen_trace(const opts_t * o, uint64_t seed, unsigned idx, const char * dir, unsigned long * cov_total) {
    gen_t g;
    memset(&g, 0, sizeof(g));
    g.o = o;
    g.r.s = seed ^ (0xD1B54A32D192ED03ull * (idx + 1));
    rng_u64(&g.r);
    g.sim = tsr_sim_create(&o->cfg, NULL);
    if (!g.sim) return 1;
    g.spm_lines = o->cfg.spm_bytes / 64;
    g.lim = tsr_sim_dram_limit_beats(g.sim);
    const uint64_t io_base = TSR_REGION_IO_OFF / 64, w_base = TSR_REGION_WEIGHT_OFF / 64;
    if (!o->m2 && !o->m3 && w_base + 1024 >= g.lim) {
        fprintf(stderr, "tsr-trace-gen: channel too small for the DRAM map (WEIGHT region at 512 MiB)\n");
        tsr_sim_destroy(g.sim);
        return 1;
    }
    if ((o->m2 || o->m3) && io_base + 65536 >= g.lim) {
        fprintf(stderr, "tsr-trace-gen: addressable channel too small for the IO region (64 MiB + 4 MiB)\n");
        tsr_sim_destroy(g.sim);
        return 1;
    }
    g.spm_hi = g.spm_lines;
    for (uint32_t c = 0; c < o->cfg.num_ch; ++c) g.top_next[c] = g.lim;
    for (uint32_t c = 0; c < o->cfg.num_ch; ++c) g.io_next[c] = io_base + rng_below(&g.r, 4096);
    g.heap_next = w_base + rng_below(&g.r, 4096);
    g.spm_next = rng_below(&g.r, 64);
    if (o->m3) {   // weights in the WEIGHT region, or in the upper half of [IO, limit) if that is beyond the limit
        g.w_lo = (w_base + 65536 < g.lim) ? w_base : io_base + (g.lim - io_base) / 2;
        g.io_lim = g.w_lo;
        g.heap_next = g.w_lo + rng_below(&g.r, 4096);
        g.wbudget = o->max_wbeats;
        if ((idx * o->big_pct) % 100 < o->big_pct) g.wbudget *= 4;      // every (100 / PCT)-th trace
    }

    // Arguments and initial base registers.
    uint32_t args[8];
    for (int i = 0; i < 8; ++i) {
        args[i] = rng_below(&g.r, 2) ? rng_u32(&g.r) : (uint32_t)(io_base + rng_below(&g.r, 1u << 16));
        g.R[8 + i] = args[i];
    }
    push(&g, tsr_enc_li(1, (uint32_t)io_base));
    g.R[1] = (uint32_t)io_base;
    push(&g, tsr_enc_li(2, (uint32_t)w_base));
    g.R[2] = (uint32_t)w_base;
    push(&g, tsr_enc_addi(3, 0, rng_below(&g.r, 1024)));
    g.R[3] = g.R[0] + (g.p[g.n - 1].w[1]);
    // M2: the error tail is chosen before the groups (a fetch-range tail reserves the end of ch0).
    int tail = -1, tail_var = -1;
    if (o->m2 && o->err_pct && rng_below(&g.r, 100) < o->err_pct) {
        tail = (int)rng_below(&g.r, T2_N);
        if (tail == T2_FETCH_END) g.reserve_ch0_top = 1;
    }
    if (o->m3) {   // tails >= T2_N are MVM faults (T3_*); --errors 100 cycles through every kind
        if (o->err_pct == 100) {
            const unsigned j = idx % (M3_TAIL_CYCLE + T2_N);
            if (j < M3_TAIL_CYCLE) {
                tail = T2_N + m3_tail_cycle[j].tail;
                tail_var = m3_tail_cycle[j].var;
            } else {
                tail = (int)(j - M3_TAIL_CYCLE);
            }
        } else if (o->err_pct && rng_below(&g.r, 100) < o->err_pct) {
            tail = rng_below(&g.r, 4) ? T2_N + (int)rng_below(&g.r, T3_N) : (int)rng_below(&g.r, T2_N);
        }
        if (tail == T2_FETCH_END) g.reserve_ch0_top = 1;
        g.wlim_pending = !g.reserve_ch0_top && rng_below(&g.r, 4) == 0;
    }
    const uint32_t n_groups = 1 + rng_below(&g.r, o->max_groups);
    uint32_t made = 0;
    for (uint32_t k = 0; k < n_groups; ++k)
        made += (uint32_t)(o->m3 ? gen_group_m3(&g, k) : o->m2 ? gen_group_m2(&g) : gen_group(&g));
    if (made == 0) {
        fprintf(stderr, "tsr-trace-gen: trace %u: no %s group fits the configuration\n", idx, o->m2 ? "M2" : "MVM");
        tsr_sim_destroy(g.sim);
        free(g.p);
        return 1;
    }
    uint32_t want_err = 0, want_pc = 0;
    int64_t want_info = -1;             // expected ERR_INFO (-1: not predicted)
    uint64_t cmd = 0;
    int cmd_fixed = 0;                  // M2 fetch tails place the command buffer themselves
    if (!o->m2 && !o->m3) {
        if (o->err_pct && rng_below(&g.r, 100) < o->err_pct) {
            want_err = gen_error_tail(&g, &want_pc);
            g.cov[COV_ERR]++;
        } else {
            push(&g, tsr_enc_end());
        }
    } else if (tail < 0) {
        push(&g, tsr_enc_end());
    } else {
        g.cov[C2_ERR]++;
        if (tail == T2_NOEND) {                       // runs past CMD_COUNT
            want_err = TSR_ERR_ILLEGAL_OP;
            want_pc = g.n;
            want_info = 0;
        } else if (tail == T2_FETCH_END) {            // instruction want_pc is the first one beyond the limit
            if (g.n & 1) push(&g, tsr_enc_nop());
            want_pc = g.n;
            const uint32_t extra = 1 + rng_below(&g.r, 5);
            for (uint32_t k = 0; k < extra; ++k) push(&g, k + 1 == extra ? tsr_enc_end() : tsr_enc_nop());
            want_err = TSR_ERR_DRAM_RANGE;
            cmd = g.lim - want_pc / 2;
            want_info = (uint32_t)g.lim;
            cmd_fixed = 1;
        } else if (tail == T2_FETCH_START) {          // CMD_ADDR itself beyond the limit: nothing executes
            push(&g, tsr_enc_end());
            switch (rng_below(&g.r, 4)) {
            case 0:  cmd = g.lim; break;
            case 1:  cmd = g.lim + 1 + rng_below(&g.r, 1u << 20); break;
            case 2:  cmd = (1ull << 32) | rng_below(&g.r, 4096); break;     // CMD_ADDR_HI != 0, LO small
            default: cmd = ((uint64_t)rng_u32(&g.r) << 32 | rng_u32(&g.r)) | (1ull << 63); break;
            }
            want_err = TSR_ERR_DRAM_RANGE;
            want_pc = 0;
            want_info = (uint32_t)cmd;
            cmd_fixed = 2;
            for (size_t i = 0; i < g.n_chk; ++i) free(g.chk[i].want);
            g.n_chk = 0;
        } else {
            want_err = tail >= T2_N ? gen_error_tail_m3(&g, tail - T2_N, tail_var, &want_pc, &want_info)
                                    : gen_error_tail_m2(&g, tail, &want_pc, &want_info);
            if (rng_below(&g.r, 2)) push(&g, tsr_enc_end());
        }
    }

    // Program into the CMD region of ch0 (M2 fetch tails: at the end of the addressable range, or
    // not at all when CMD_ADDR lies beyond it). Bytes beyond the channel are not stored.
    if (!cmd_fixed) {
        const uint64_t cmd_beats = tsr_ceil_div_u64(g.n, 2);
        cmd = rng_below(&g.r, (uint32_t)(TSR_REGION_CMD_SIZE / 64 - cmd_beats));
    }
    if (cmd_fixed != 2) {
        for (uint32_t k = 0; k < g.n; ++k) {
            if ((cmd * 64 + (uint64_t)k * 32 + 32) > o->cfg.ch_bytes) break;
            uint8_t * cb = tsr_sim_dram(g.sim, 0) + cmd * 64 + (size_t)k * 32;
            for (uint32_t j = 0; j < 8; ++j) {
                const uint32_t w = g.p[k].w[j];
                uint8_t * b = cb + 4 * j;
                b[0] = (uint8_t)w; b[1] = (uint8_t)(w >> 8); b[2] = (uint8_t)(w >> 16); b[3] = (uint8_t)(w >> 24);
            }
        }
    }

    tsr_sim_capture_begin(g.sim);
    const uint32_t status = tsr_sim_exec(g.sim, cmd, g.n, args);
    tsr_trace tr;
    memset(&tr, 0, sizeof(tr));
    tsr_sim_capture_end(g.sim, &tr.cap);
    const uint32_t ecode = tsr_sim_csr_read(g.sim, TSR_CSR_ERR_CODE);
    int rc = 0;
    if (want_err == 0 && status != TSR_STATUS_DONE) {
        fprintf(stderr, "tsr-trace-gen: trace %u: valid program failed: STATUS %u ERR_CODE %u ERR_PC %u ERR_INFO 0x%x\n", idx,
                status, ecode, tsr_sim_csr_read(g.sim, TSR_CSR_ERR_PC), tsr_sim_csr_read(g.sim, TSR_CSR_ERR_INFO));
        rc = 1;
    }
    if (want_err != 0 && (status != (TSR_STATUS_DONE | TSR_STATUS_ERR) || ecode != want_err ||
                          tsr_sim_csr_read(g.sim, TSR_CSR_ERR_PC) != want_pc ||
                          (want_info >= 0 && tsr_sim_csr_read(g.sim, TSR_CSR_ERR_INFO) != (uint32_t)want_info))) {
        fprintf(stderr, "tsr-trace-gen: trace %u: expected error %u at %u (info %lld), got STATUS %u ERR_CODE %u "
                        "ERR_PC %u ERR_INFO 0x%x (count %u)\n", idx, want_err, want_pc, (long long)want_info, status,
                ecode, tsr_sim_csr_read(g.sim, TSR_CSR_ERR_PC), tsr_sim_csr_read(g.sim, TSR_CSR_ERR_INFO), g.n);
        rc = 1;
    }
    // Independent check of every stored output.
    for (size_t i = 0; i < g.n_chk && rc == 0; ++i) {
        const check_t * k = &g.chk[i];
        const uint8_t * got = tsr_sim_dram(g.sim, k->ch) + k->beat * 64;
        if (memcmp(got, k->want, (size_t)k->lines * 64) != 0) {
            size_t b = 0;
            while (got[b] == k->want[b]) ++b;
            fprintf(stderr, "tsr-trace-gen: trace %u: output %zu (ch %u beat %llu) differs from the host reference at byte %zu\n",
                    idx, i, k->ch, (unsigned long long)k->beat, b);
            rc = 1;
        }
    }
    if (rc == 0) {
        tr.cfg = o->cfg;
        tr.cmd_addr_beat = cmd;
        tr.cmd_count = g.n;
        memcpy(tr.args, args, sizeof(args));
        tr.status = status;
        tr.err_code = ecode;
        tr.err_pc = tsr_sim_csr_read(g.sim, TSR_CSR_ERR_PC);
        tr.err_info = tsr_sim_csr_read(g.sim, TSR_CSR_ERR_INFO);
        tr.has_err_info = 1;
        tr.perf.valid = 1;
        tr.perf.cycles = tsr_sim_perf(g.sim, TSR_PERF_CYCLES);
        tr.perf.retired = tsr_sim_perf(g.sim, TSR_PERF_RETIRED);
        tr.perf.mvm_cycles = tsr_sim_perf(g.sim, TSR_PERF_MVM_CYCLES);
        for (uint32_t c = 0; c < o->cfg.num_ch; ++c) tr.perf.ch_rd_beats[c] = tsr_sim_perf(g.sim, tsr_perf_ch_rd_beats(c));
        if (tsr_trace_write(dir, &tr) != 0) rc = 1;
    }
    if (o->m3 && cmd_fixed == 2) g.wch0 = g.wall = g.n_mvm = 0;   // CMD_ADDR beyond the limit: nothing ran
    if (rc == 0 && o->m3) {
        m3_stats.ch0 += g.wch0;
        m3_stats.all += g.wall;
        m3_stats.mvms += g.n_mvm;
        if (g.wch0 > m3_stats.max) m3_stats.max = g.wch0;
        m3_stats.over += g.wch0 > M3_BIG_BEATS;
    }
    if (rc == 0 && !o->quiet) {
        size_t bytes = 0;
        for (size_t i = 0; i < tr.cap.n_init; ++i) bytes += (size_t)tr.cap.init[i].size;
        printf("%s: %u insns, %zu outputs checked, %zu init regions (%.1f KiB)%s", dir, g.n, g.n_chk, tr.cap.n_init,
               (double)bytes / 1024.0, want_err ? ", ends in an error" : "");
        if (o->m3) printf(", %u MVMs, %llu weight beats (ch0)", g.n_mvm, (unsigned long long)g.wch0);
        printf("\n");
    }
    if (rc == 0 && want_err < 16) g.cov[COV_ERRC + want_err]++;
    for (int i = 0; i < COV_MAX; ++i) cov_total[i] += g.cov[i];
    tsr_capture_free(&tr.cap);
    for (size_t i = 0; i < g.n_chk; ++i) free(g.chk[i].want);
    free(g.chk);
    free(g.p);
    tsr_sim_destroy(g.sim);
    return rc;
}

static void usage(void) {
    fprintf(stderr,
            "usage: tsr-trace-gen -o DIR [-n COUNT] [-s SEED] [--prefix NAME] [--ops m1|m2|m3] [--t-max T]\n"
            "                     [--num-ch C] [--spm-kib S] [--ch-gib G] [--addr-beat-w W] [--k-max K]\n"
            "                     [--max-k K] [--max-n N] [--max-groups G] [--errors PCT] [--no-special] [-q]\n"
            "                     [--max-wbeats B] [--big-pct PCT]\n"
            "defaults: -n 20 -s 1 --prefix gen --ops m1, U200 v1 config (T_MAX 1, 4 channels, 4 MiB SPM,\n"
            "          16 GiB/ch, ADDR_BEAT_W 28), --max-k 4096 --max-n 256 --max-groups 3 --errors 0;\n"
            "          --num-ch is 1..16, --addr-beat-w 1..40\n"
            "--ops m2: programs of NOP END LI ADDI LD ST QACT FENCE only (M2 RTL core; --max-n unused)\n"
            "--ops m3: MVM-centred programs for the M3 RTL core (with M2 groups mixed in); --max-wbeats B\n"
            "          (default 16384) bounds the MVM weight beats per trace on the busiest channel,\n"
            "          --big-pct PCT (default 0) gives every (100/PCT)-th trace (index 0 first) a 4x budget\n");
}

static int parse_u(const char * s, unsigned long lo, unsigned long hi, unsigned long * out) {
    char * end;
    errno = 0;
    const unsigned long v = strtoul(s, &end, 0);
    if (errno || *end || end == s || v < lo || v > hi) return -1;
    *out = v;
    return 0;
}

int main(int argc, char ** argv) {
    opts_t o;
    memset(&o, 0, sizeof(o));
    const tsr_config u200 = TSR_CONFIG_U200_V1;
    o.cfg = u200;
    o.max_k = 4096;
    o.max_n = 256;
    o.max_groups = 3;
    o.max_wbeats = 16384;
    o.special = 1;
    const char * out = NULL;
    const char * prefix = "gen";
    unsigned long count = 20, seed = 1, v;
    for (int i = 1; i < argc; ++i) {
        const char * a = argv[i];
        const char * val = (i + 1 < argc) ? argv[i + 1] : NULL;
#define OPT(name) (strcmp(a, name) == 0 && val && ++i)
        if (OPT("-o")) out = val;
        else if (OPT("-n")) { if (parse_u(val, 1, 1000000, &count)) goto bad; }
        else if (OPT("-s")) { if (parse_u(val, 0, ~0ul, &seed)) goto bad; }
        else if (OPT("--prefix")) prefix = val;
        else if (OPT("--t-max")) { if (parse_u(val, 1, 8, &v)) goto bad; o.cfg.t_max = (uint32_t)v; }
        else if (OPT("--num-ch")) { if (parse_u(val, 1, TSR_NUM_CH_MAX, &v)) goto bad; o.cfg.num_ch = (uint32_t)v; }
        else if (OPT("--spm-kib")) { if (parse_u(val, 64, 65536, &v)) goto bad; o.cfg.spm_bytes = (uint32_t)v * 1024; }
        else if (OPT("--ch-gib")) { if (parse_u(val, 1, 16, &v)) goto bad; o.cfg.ch_bytes = (uint64_t)v << 30; }
        else if (OPT("--addr-beat-w")) { if (parse_u(val, 1, 40, &v)) goto bad; o.cfg.addr_beat_w = (uint32_t)v; }
        else if (OPT("--ops")) {
            if (strcmp(val, "m1") == 0) o.m2 = o.m3 = 0;
            else if (strcmp(val, "m2") == 0) { o.m2 = 1; o.m3 = 0; }
            else if (strcmp(val, "m3") == 0) { o.m2 = 0; o.m3 = 1; }
            else goto bad;
        }
        else if (OPT("--k-max")) { if (parse_u(val, 256, 65280, &v) || v % 256) goto bad; o.cfg.k_max = (uint32_t)v; }
        else if (OPT("--max-k")) { if (parse_u(val, 32, 65280, &v)) goto bad; o.max_k = (uint32_t)v; }
        else if (OPT("--max-n")) { if (parse_u(val, 1, 1u << 20, &v)) goto bad; o.max_n = (uint32_t)v; }
        else if (OPT("--max-groups")) { if (parse_u(val, 1, 64, &v)) goto bad; o.max_groups = (uint32_t)v; }
        else if (OPT("--errors")) { if (parse_u(val, 0, 100, &v)) goto bad; o.err_pct = (unsigned)v; }
        else if (OPT("--max-wbeats")) { if (parse_u(val, 1, 1u << 24, &v)) goto bad; o.max_wbeats = (uint32_t)v; }
        else if (OPT("--big-pct")) { if (parse_u(val, 0, 100, &v)) goto bad; o.big_pct = (unsigned)v; }
        else if (strcmp(a, "--no-special") == 0) o.special = 0;
        else if (strcmp(a, "-q") == 0) o.quiet = 1;
        else if (strcmp(a, "-h") == 0 || strcmp(a, "--help") == 0) { usage(); return 0; }
        else goto bad;
#undef OPT
    }
    if (!out) goto bad;
    if (mkdir_p(out)) return 2;
    {
        unsigned long cov[COV_MAX] = {0};
        unsigned long fails = 0;
        for (unsigned long i = 0; i < count; ++i) {
            const size_t n = strlen(out) + strlen(prefix) + 64;
            char * dir = (char *)xmalloc(n);
            snprintf(dir, n, "%s/%s_s%lu_%04lu", out, prefix, seed, i);
            if (gen_trace(&o, (uint64_t)seed, (unsigned)i, dir, cov) != 0) {
                fprintf(stderr, "tsr-trace-gen: FAILED %s\n", dir);
                ++fails;
            }
            free(dir);
        }
        printf("tsr-trace-gen: %lu traces in %s (%lu failed); coverage:", count - fails, out, fails);
        if (o.m3) {
            for (int i = 0; i < C3_N; ++i) printf("%s %s %lu", i ? "," : "", cov3_names[i], cov[C2_N + i]);
            printf("; M2 groups and QACTs:");
            for (int i = 0; i < C2_N; ++i) printf("%s %s %lu", i ? "," : "", cov2_names[i], cov[i]);
            printf("; final ERR_CODE:");
            for (int e = 0; e < 16; ++e) {
                if (cov[COV_ERRC + e]) printf(" %d:%lu", e, cov[COV_ERRC + e]);
            }
            printf("\ntsr-trace-gen: %llu MVMs; MVM weight beats: %llu on the busiest channel (sum over traces), "
                   "%llu on all channels, max %llu per trace, %llu traces above %llu", m3_stats.mvms, m3_stats.ch0,
                   m3_stats.all, m3_stats.max, m3_stats.over, M3_BIG_BEATS);
        } else if (!o.m2) {
            for (int i = 0; i < COV_N; ++i) printf("%s %s %lu", i ? "," : "", cov_names[i], cov[i]);
        } else {
            for (int i = 0; i < C2_N; ++i) printf("%s %s %lu", i ? "," : "", cov2_names[i], cov[i]);
            printf("; final ERR_CODE:");
            for (int e = 0; e < 16; ++e) {
                if (cov[COV_ERRC + e]) printf(" %d:%lu", e, cov[COV_ERRC + e]);
            }
        }
        printf("\n");
        return fails ? 1 : 0;
    }
bad:
    usage();
    return 2;
}
