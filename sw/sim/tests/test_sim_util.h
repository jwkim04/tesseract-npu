// test_sim_util.h — helpers shared by the simulator / capture / transport tests (prefix sim_).
#pragma once

#ifndef _GNU_SOURCE   // C++ compilers predefine it (g++); redefining it warns
#define _GNU_SOURCE
#endif
#include <ftw.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "test_util.h"
#include "tsr_isa.h"
#include "tsr_ref.h"
#include "tsr_sim.h"

// ---------------------------------------------------------------------------
// Configs
// ---------------------------------------------------------------------------
static inline tsr_config cfg_u200(uint32_t t_max) {
    tsr_config c = TSR_CONFIG_U200;
    c.t_max = t_max;
    return c;
}

// ---------------------------------------------------------------------------
// Op sequences (executed one at a time, in order, through tsr_sim_op)
// ---------------------------------------------------------------------------
typedef struct { tsr_insn * v; uint32_t n, cap; } ops_t;

static inline void ops_reset(ops_t * p) { p->n = 0; }
static inline void ops_free(ops_t * p) { free(p->v); p->v = NULL; p->n = p->cap = 0; }
static inline uint32_t ops_add(ops_t * p, tsr_insn i) {
    if (p->n == p->cap) {
        p->cap = p->cap ? 2 * p->cap : 64;
        p->v = (tsr_insn *)realloc(p->v, p->cap * sizeof(tsr_insn));
        if (!p->v) abort();
    }
    p->v[p->n] = i;
    return p->n++;
}
static inline tsr_insn insn_raw(uint32_t w0, uint32_t w1, uint32_t w2, uint32_t w3, uint32_t w4, uint32_t w5,
                                uint32_t w6, uint32_t w7) {
    tsr_insn i = {{ w0, w1, w2, w3, w4, w5, w6, w7 }};
    return i;
}
// Runs the ops in order and stops at the first one that fails. Returns 0, or that op's error code
// with *err_idx = its index and *err_info = its ERR_INFO (either pointer may be NULL).
static inline uint32_t ops_run(tsr_sim * sim, const ops_t * p, uint32_t * err_idx, uint32_t * err_info) {
    for (uint32_t k = 0; k < p->n; ++k) {
        uint32_t info = 0;
        const uint32_t err = tsr_sim_op(sim, p->v[k].w, &info);
        if (err != TSR_ERR_NONE) {
            if (err_idx) *err_idx = k;
            if (err_info) *err_info = info;
            return err;
        }
    }
    if (err_idx) *err_idx = p->n;
    if (err_info) *err_info = 0;
    return TSR_ERR_NONE;
}

// ---------------------------------------------------------------------------
// Data
// ---------------------------------------------------------------------------
static inline uint16_t rand_fp16_scale(rng_t * r) {
    // mostly normal fp16 magnitudes 2^-14 .. 2^-2, random sign; rarely subnormal or zero
    const uint32_t k = rng_below(r, 64);
    uint16_t sign = (uint16_t)((rng_u32(r) & 1u) << 15);
    if (k == 0) return sign;                                           // +-0
    if (k == 1) return (uint16_t)(sign | (1 + rng_below(r, 0x3FF)));    // subnormal
    const uint16_t e = (uint16_t)(1 + rng_below(r, 13));                 // biased exponent 1..13
    return (uint16_t)(sign | (e << 10) | rng_below(r, 1024));
}

// Random ggml blocks for one row of K weights (Q8_0: 34 B/block, Q4_K: 144 B/sb).
static inline void gen_row_blocks(rng_t * r, uint32_t fmt, uint32_t K, uint8_t * out) {
    const uint32_t nb = K / tsr_fmt_block_size(fmt);
    const size_t bsz = tsr_fmt_ggml_block_bytes(fmt);
    for (uint32_t b = 0; b < nb; ++b) {
        uint8_t * blk = out + (size_t)b * bsz;
        rng_bytes(r, blk, bsz);                // qs / scales: any bit pattern is valid
        const uint16_t d = rand_fp16_scale(r);
        blk[0] = (uint8_t)d; blk[1] = (uint8_t)(d >> 8);
        if (fmt == TSR_FMT_Q4_K) {
            const uint16_t dm = rand_fp16_scale(r);
            blk[2] = (uint8_t)dm; blk[3] = (uint8_t)(dm >> 8);
        }
    }
}

// Writes an N x K tensor (ggml blocks, row-major) repacked at W_OFF (beats) per the
// symmetric allocation, using tsr_row_location.
static inline void put_weights(tsr_sim * sim, uint32_t fmt, uint32_t N, uint32_t K, uint64_t w_off,
                               const uint8_t * blocks) {
    const size_t row_bytes = (size_t)(K / tsr_fmt_block_size(fmt)) * tsr_fmt_ggml_block_bytes(fmt);
    const uint32_t C = tsr_sim_config(sim)->num_ch;
    for (uint32_t n = 0; n < N; ++n) {
        uint32_t ch;
        uint64_t off;
        tsr_row_location(fmt, N, K, C, n, &ch, &off);
        tsr_repack_row(fmt, blocks + (size_t)n * row_bytes, K, tsr_sim_dram(sim, ch) + (w_off + off) * TSR_BEAT_BYTES);
    }
}

static inline void gen_f32(rng_t * r, float * x, size_t n) {
    const double scale = ldexp(1.0, (int)rng_below(r, 16) - 8);
    for (size_t i = 0; i < n; ++i) x[i] = (float)(rng_normal(r) * scale);
}

static inline void put_f32_le(uint8_t * dst, const float * x, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        const uint32_t u = f2u(x[i]);
        dst[4 * i] = (uint8_t)u; dst[4 * i + 1] = (uint8_t)(u >> 8);
        dst[4 * i + 2] = (uint8_t)(u >> 16); dst[4 * i + 3] = (uint8_t)(u >> 24);
    }
}
static inline uint32_t get_u32_le(const uint8_t * p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

// FNV-1a over a byte range (state comparisons).
static inline uint64_t fnv64(const void * p, size_t n) {
    const uint8_t * b = (const uint8_t *)p;
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

// ---------------------------------------------------------------------------
// Scratch directories
// ---------------------------------------------------------------------------
static int rm_cb(const char * path, const struct stat * sb, int flag, struct FTW * ftw) {
    (void)sb; (void)flag; (void)ftw;
    return remove(path);
}
static inline void rm_rf(const char * path) { nftw(path, rm_cb, 16, FTW_DEPTH | FTW_PHYS); }

// Creates a fresh directory <base>/<name>.XXXXXX (base created if needed). Returns malloc'd path.
static inline char * make_scratch(const char * base, const char * name) {
    mkdir(base, 0755);
    const size_t n = strlen(base) + strlen(name) + 16;
    char * p = (char *)malloc(n);
    snprintf(p, n, "%s/%s.XXXXXX", base, name);
    if (mkdtemp(p) == NULL) {
        fprintf(stderr, "mkdtemp(%s) failed\n", p);
        exit(1);
    }
    return p;
}
