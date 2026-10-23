// tsr_sim.c — Tesseract ISA simulator, M1 instruction set (NOP END LI ADDI LD ST QACT MVM FENCE).
//
// Normative sources: docs/spec.md "실행 모델과 제어 인터페이스", "ISA", "MVM 의미",
// "구현 명확화 (v0.2.1)". All numerics come from tsr_ref.c (spec rule 3); this file only
// decodes, checks, moves bytes and sequences.
//
// Decisions pinned in spec 구현 명확화 (v0.2.1, simulator section) and implemented here:
//   - Reserved fields: every word, register field, flag bit and sub value an instruction does
//     not use must be zero, else ILLEGAL_OP. LD/ST sub = channel >= NUM_CH -> ILLEGAL_OP.
//     QACT sub (QA kind) >= 2 -> ILLEGAL_OP. MVM sub (fmt) >= 2 -> BAD_FMT.
//   - Check order ILLEGAL_OP -> BAD_FMT -> K_RANGE -> T_RANGE -> SPM_RANGE -> DRAM_RANGE;
//     inside SPM_RANGE / DRAM_RANGE the operands are checked in field order A1, A2, A3.
//   - Range check: an operand with start address a (32-bit, R[x] + imm wrapped) and extent e
//     (lines/beats, 64-bit, for T strided segments e = (T-1)*stride + len) is in range iff
//     a + e <= limit, computed without wrap. ERR_INFO = a (the operand's start address).
//     SPM limit = SPM_BYTES/64 lines; DRAM limit = min(CH_BYTES/64, 2^ADDR_BEAT_W) beats.
//   - Command fetch: instruction pc lives in ch0 beat CMD_ADDR + pc/2, bytes 32*(pc%2)..;
//     a beat is read once when the first instruction in it is fetched (perf + capture).
//     Fetch beyond the DRAM limit -> DRAM_RANGE, ERR_PC = pc, ERR_INFO = low 32 bits of the
//     beat address. pc == CMD_COUNT without END -> ILLEGAL_OP, ERR_PC = CMD_COUNT.
//   - ABORT is sampled between instructions: ABORTED, ERR_PC = first instruction not executed.
//   - Any error stops the run with STATUS = DONE | ERR (so `done` fires on errors too).
//
// Cycle model (PERF_*; deterministic, informational — not part of the bit-exact contract):
//   every instruction       1 issue cycle (command fetch is prefetched: no extra cycles)
//   LD / ST                 + 64 (DMA latency) + n_lines
//   QACT                    + T * (ceil(K / VL) + QA_LINES(kind, K))
//   MVM                     + 64 (pipeline fill/drain) + T * QA_LINES (activation broadcast)
//                           + max over channels of rows_c * ROW_BEATS (1 weight beat/cycle)
//   PERF_MVM_CYCLES         the MVM part above (without the issue cycle)
//   PERF_CH_RD_BEATS[c]     command fetch beats (ch0), LD beats, MVM weight beats
//   PERF_CH_WAIT, VU, WAITM 0 (no DRAM stalls modelled; no VU/WAITM in M1)
// Counters are cleared by tsr_sim_create and CTRL.SOFT_RESET only; they accumulate over runs.

#define _GNU_SOURCE

#include "tsr_sim.h"
#include "tsr_ref.h"

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#define CYC_ISSUE    1u
#define CYC_DMA_LAT  64u
#define CYC_MVM_FILL 64u

// MVMs with fewer weight beats x tokens than this run on the calling thread only.
#define MT_MIN_WORK_BEATS 4096u

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------
static void * xrealloc(void * p, size_t n) {
    void * q = realloc(p, n ? n : 1);
    if (q == NULL) {
        fprintf(stderr, "tsr_sim: out of memory (%zu bytes)\n", n);
        abort();
    }
    return q;
}
static void * xcalloc(size_t n, size_t sz) {
    void * p = calloc(n ? n : 1, sz ? sz : 1);
    if (p == NULL) {
        fprintf(stderr, "tsr_sim: out of memory (%zu x %zu bytes)\n", n, sz);
        abort();
    }
    return p;
}

static inline uint32_t rd32le(const uint8_t * p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static inline void wr32le(uint8_t * p, uint32_t v) {
    p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24);
}

// ---------------------------------------------------------------------------
// Interval sets of beats [lo, hi): sorted, disjoint, never adjacent (adjacent ones merge).
// ---------------------------------------------------------------------------
typedef struct { uint64_t lo, hi; } ival;
typedef struct { ival * v; size_t n, cap; } iset;

static void iset_free(iset * s) { free(s->v); s->v = NULL; s->n = s->cap = 0; }

// First index i with v[i].hi >= x.
static size_t iset_lb(const iset * s, uint64_t x) {
    size_t a = 0, b = s->n;
    while (a < b) {
        const size_t m = a + (b - a) / 2;
        if (s->v[m].hi < x) a = m + 1; else b = m;
    }
    return a;
}

static void iset_add(iset * s, uint64_t lo, uint64_t hi) {
    if (lo >= hi) return;
    const size_t i = iset_lb(s, lo);          // first interval touching or after lo
    size_t j = i;
    while (j < s->n && s->v[j].lo <= hi) {    // overlapping or adjacent
        if (s->v[j].lo < lo) lo = s->v[j].lo;
        if (s->v[j].hi > hi) hi = s->v[j].hi;
        ++j;
    }
    if (j == i) {
        if (s->n == s->cap) {
            s->cap = s->cap ? 2 * s->cap : 16;
            s->v = (ival *)xrealloc(s->v, s->cap * sizeof(ival));
        }
        memmove(s->v + i + 1, s->v + i, (s->n - i) * sizeof(ival));
        s->n++;
    } else if (j > i + 1) {
        memmove(s->v + i + 1, s->v + j, (s->n - j) * sizeof(ival));
        s->n -= j - i - 1;
    }
    s->v[i].lo = lo;
    s->v[i].hi = hi;
}

// ---------------------------------------------------------------------------
// Access capture state
// ---------------------------------------------------------------------------
typedef struct { uint64_t lo, hi; uint8_t * data; } cap_chunk;
typedef struct {
    iset seen;         // init chunks + written beats
    iset written;
    cap_chunk * c; size_t n, cap;
} cap_ch;

// ---------------------------------------------------------------------------
// Worker pool for MVM rows
// ---------------------------------------------------------------------------
typedef void (*row_fn)(void * ctx, uint32_t lo, uint32_t hi);

typedef struct pool {
    unsigned        n;           // worker threads (the caller works too)
    pthread_t *     th;
    pthread_mutex_t mu;
    pthread_cond_t  cv_go, cv_done;
    uint64_t        gen;
    unsigned        busy;
    int             quit;
    row_fn          fn;
    void *          ctx;
    uint32_t        total, chunk;
    atomic_uint     next;
} pool;

static void pool_work(pool * p) {
    for (;;) {
        const uint32_t lo = atomic_fetch_add_explicit(&p->next, p->chunk, memory_order_relaxed);
        if (lo >= p->total) break;
        const uint32_t hi = (p->total - lo < p->chunk) ? p->total : lo + p->chunk;
        p->fn(p->ctx, lo, hi);
    }
}

static void * pool_thread(void * arg) {
    pool * p = (pool *)arg;
    uint64_t seen = 0;
    pthread_mutex_lock(&p->mu);
    for (;;) {
        while (!p->quit && p->gen == seen) pthread_cond_wait(&p->cv_go, &p->mu);
        if (p->quit) break;
        seen = p->gen;
        pthread_mutex_unlock(&p->mu);
        pool_work(p);
        pthread_mutex_lock(&p->mu);
        if (--p->busy == 0) pthread_cond_signal(&p->cv_done);
    }
    pthread_mutex_unlock(&p->mu);
    return NULL;
}

static pool * pool_create(unsigned n_workers) {
    pool * p = (pool *)xcalloc(1, sizeof(pool));
    pthread_mutex_init(&p->mu, NULL);
    pthread_cond_init(&p->cv_go, NULL);
    pthread_cond_init(&p->cv_done, NULL);
    atomic_init(&p->next, 0);
    p->th = (pthread_t *)xcalloc(n_workers, sizeof(pthread_t));
    for (unsigned i = 0; i < n_workers; ++i) {
        if (pthread_create(&p->th[i], NULL, pool_thread, p) != 0) {
            fprintf(stderr, "tsr_sim: pthread_create failed; using %u worker threads\n", i);
            break;
        }
        p->n++;
    }
    return p;
}

static void pool_destroy(pool * p) {
    if (p == NULL) return;
    pthread_mutex_lock(&p->mu);
    p->quit = 1;
    pthread_cond_broadcast(&p->cv_go);
    pthread_mutex_unlock(&p->mu);
    for (unsigned i = 0; i < p->n; ++i) pthread_join(p->th[i], NULL);
    pthread_cond_destroy(&p->cv_go);
    pthread_cond_destroy(&p->cv_done);
    pthread_mutex_destroy(&p->mu);
    free(p->th);
    free(p);
}

static void pool_run(pool * p, row_fn fn, void * ctx, uint32_t total, uint32_t chunk) {
    pthread_mutex_lock(&p->mu);
    p->fn = fn;
    p->ctx = ctx;
    p->total = total;
    p->chunk = chunk ? chunk : 1;
    atomic_store_explicit(&p->next, 0, memory_order_relaxed);
    p->busy = p->n;
    p->gen++;
    pthread_cond_broadcast(&p->cv_go);
    pthread_mutex_unlock(&p->mu);
    pool_work(p);
    pthread_mutex_lock(&p->mu);
    while (p->busy) pthread_cond_wait(&p->cv_done, &p->mu);
    pthread_mutex_unlock(&p->mu);
}

// ---------------------------------------------------------------------------
// Simulator state
// ---------------------------------------------------------------------------
struct tsr_sim {
    tsr_config cfg;
    uint32_t   num_ch;
    uint64_t   ch_bytes;
    uint64_t   limit_beats;     // min(ch_bytes/64, 2^addr_beat_w)
    uint32_t   spm_lines;
    uint8_t ** dram;
    int *      dram_fd;
    uint8_t *  spm;

    // CSRs
    uint32_t   status;
    uint32_t   cmd_addr_lo, cmd_addr_hi, cmd_count;
    uint32_t   err_pc, err_code, err_info, pc;
    uint32_t   args[8];
    atomic_int abort_req;

    // run state latched at START
    uint64_t   run_cmd_addr;
    uint32_t   run_cmd_count;
    uint32_t   R[16];
    float      S[16];

    uint64_t * perf;
    uint32_t   n_perf;

    FILE *     log;

    // capture
    int        capturing;
    cap_ch *   cap;
    ival *     gaps;
    size_t     gaps_cap;

    // scratch
    float *    xbuf;            // k_max floats (QACT source vector)
    float *    ybuf;            // T*N MVM results
    size_t     ybuf_cap;

    unsigned   threads;
    pool *     pool;
};

// ---------------------------------------------------------------------------
// Capture
// ---------------------------------------------------------------------------
static void cap_read(tsr_sim * sim, uint32_t ch, uint64_t lo, uint64_t n) {
    if (!sim->capturing || n == 0) return;
    cap_ch * cc = &sim->cap[ch];
    const uint64_t hi = lo + n;
    // Collect the parts of [lo, hi) not yet seen (neither captured nor written in this run).
    size_t ng = 0;
    uint64_t cur = lo;
    for (size_t k = iset_lb(&cc->seen, lo + 1); k < cc->seen.n && cc->seen.v[k].lo < hi; ++k) {
        if (cc->seen.v[k].lo > cur) {
            if (ng == sim->gaps_cap) {
                sim->gaps_cap = sim->gaps_cap ? 2 * sim->gaps_cap : 16;
                sim->gaps = (ival *)xrealloc(sim->gaps, sim->gaps_cap * sizeof(ival));
            }
            sim->gaps[ng].lo = cur;
            sim->gaps[ng].hi = cc->seen.v[k].lo;
            ++ng;
        }
        if (cc->seen.v[k].hi > cur) cur = cc->seen.v[k].hi;
    }
    if (cur < hi) {
        if (ng == sim->gaps_cap) {
            sim->gaps_cap = sim->gaps_cap ? 2 * sim->gaps_cap : 16;
            sim->gaps = (ival *)xrealloc(sim->gaps, sim->gaps_cap * sizeof(ival));
        }
        sim->gaps[ng].lo = cur;
        sim->gaps[ng].hi = hi;
        ++ng;
    }
    for (size_t g = 0; g < ng; ++g) {
        const uint64_t glo = sim->gaps[g].lo, ghi = sim->gaps[g].hi;
        const size_t bytes = (size_t)(ghi - glo) * TSR_BEAT_BYTES;
        uint8_t * data = (uint8_t *)xrealloc(NULL, bytes);
        memcpy(data, sim->dram[ch] + glo * TSR_BEAT_BYTES, bytes);
        if (cc->n == cc->cap) {
            cc->cap = cc->cap ? 2 * cc->cap : 16;
            cc->c = (cap_chunk *)xrealloc(cc->c, cc->cap * sizeof(cap_chunk));
        }
        cc->c[cc->n].lo = glo;
        cc->c[cc->n].hi = ghi;
        cc->c[cc->n].data = data;
        cc->n++;
        iset_add(&cc->seen, glo, ghi);
    }
}

static void cap_write(tsr_sim * sim, uint32_t ch, uint64_t lo, uint64_t n) {
    if (!sim->capturing || n == 0) return;
    iset_add(&sim->cap[ch].written, lo, lo + n);
    iset_add(&sim->cap[ch].seen, lo, lo + n);
}

static void cap_state_free(tsr_sim * sim) {
    if (sim->cap == NULL) return;
    for (uint32_t c = 0; c < sim->num_ch; ++c) {
        cap_ch * cc = &sim->cap[c];
        for (size_t i = 0; i < cc->n; ++i) free(cc->c[i].data);
        free(cc->c);
        iset_free(&cc->seen);
        iset_free(&cc->written);
    }
    free(sim->cap);
    sim->cap = NULL;
    sim->capturing = 0;
}

void tsr_sim_capture_begin(tsr_sim * sim) {
    cap_state_free(sim);
    sim->cap = (cap_ch *)xcalloc(sim->num_ch, sizeof(cap_ch));
    sim->capturing = 1;
}

static int chunk_cmp(const void * a, const void * b) {
    const cap_chunk * x = (const cap_chunk *)a;
    const cap_chunk * y = (const cap_chunk *)b;
    return (x->lo > y->lo) - (x->lo < y->lo);
}

void tsr_sim_capture_end(tsr_sim * sim, tsr_capture * out) {
    memset(out, 0, sizeof(*out));
    if (!sim->capturing) return;
    size_t cap_init = 0, cap_exp = 0;
    for (uint32_t c = 0; c < sim->num_ch; ++c) {
        cap_ch * cc = &sim->cap[c];
        // init: chunks never overlap (each was a gap of `seen`); merge adjacent runs.
        // (cc->c is NULL for a channel without captured reads; qsort(NULL, 0) is UB.)
        if (cc->n > 1) qsort(cc->c, cc->n, sizeof(cap_chunk), chunk_cmp);
        size_t i = 0;
        while (i < cc->n) {
            size_t j = i + 1;
            uint64_t hi = cc->c[i].hi;
            while (j < cc->n && cc->c[j].lo == hi) { hi = cc->c[j].hi; ++j; }
            uint8_t * data = cc->c[i].data;
            if (j > i + 1) {
                data = (uint8_t *)xrealloc(data, (size_t)(hi - cc->c[i].lo) * TSR_BEAT_BYTES);
                for (size_t k = i + 1; k < j; ++k) {
                    memcpy(data + (size_t)(cc->c[k].lo - cc->c[i].lo) * TSR_BEAT_BYTES, cc->c[k].data,
                           (size_t)(cc->c[k].hi - cc->c[k].lo) * TSR_BEAT_BYTES);
                    free(cc->c[k].data);
                    cc->c[k].data = NULL;
                }
            }
            cc->c[i].data = NULL;
            if (out->n_init == cap_init) {
                cap_init = cap_init ? 2 * cap_init : 16;
                out->init = (tsr_region *)xrealloc(out->init, cap_init * sizeof(tsr_region));
            }
            tsr_region * r = &out->init[out->n_init++];
            r->ch = c;
            r->byte_off = cc->c[i].lo * TSR_BEAT_BYTES;
            r->size = (hi - cc->c[i].lo) * TSR_BEAT_BYTES;
            r->data = data;
            i = j;
        }
        cc->n = 0;
        // expect: written ranges (already merged), content at the end of the run.
        for (size_t k = 0; k < cc->written.n; ++k) {
            if (out->n_expect == cap_exp) {
                cap_exp = cap_exp ? 2 * cap_exp : 16;
                out->expect = (tsr_region *)xrealloc(out->expect, cap_exp * sizeof(tsr_region));
            }
            tsr_region * r = &out->expect[out->n_expect++];
            r->ch = c;
            r->byte_off = cc->written.v[k].lo * TSR_BEAT_BYTES;
            r->size = (cc->written.v[k].hi - cc->written.v[k].lo) * TSR_BEAT_BYTES;
            r->data = (uint8_t *)xrealloc(NULL, (size_t)r->size);
            memcpy(r->data, sim->dram[c] + r->byte_off, (size_t)r->size);
        }
    }
    cap_state_free(sim);
}

void tsr_capture_free(tsr_capture * cap) {
    if (cap == NULL) return;
    for (size_t i = 0; i < cap->n_init; ++i) free(cap->init[i].data);
    for (size_t i = 0; i < cap->n_expect; ++i) free(cap->expect[i].data);
    free(cap->init);
    free(cap->expect);
    memset(cap, 0, sizeof(*cap));
}

// ---------------------------------------------------------------------------
// Creation / destruction
// ---------------------------------------------------------------------------
static const char * cfg_problem(const tsr_config * c) {
    if (c->beat_bits != 512) return "beat_bits must be 512 (64-byte beats and lines)";
    if (c->num_ch < 1 || c->num_ch > TSR_NUM_CH_MAX)
        return "num_ch must be in [1, 16] (LD/ST address the channel with the 4-bit w0 sub field)";
    if (c->lanes < 1 || c->lanes > 4095) return "lanes must be in [1, 4095] (CFG0 LANES[19:8])";
    if (c->t_max < 1 || c->t_max > TSR_T_MAX_ARCH) return "t_max must be in [1, 8]";
    if (c->k_max < 256 || c->k_max > 65280 || c->k_max % 256 != 0)
        return "k_max must be a multiple of 256 in [256, 65280] (MVM K field is 16 bits)";
    if (c->spm_bytes < TSR_LINE_BYTES || c->spm_bytes % TSR_LINE_BYTES != 0)
        return "spm_bytes must be a positive multiple of 64";
    if (c->vl < 1 || c->vl > 255) return "vl must be in [1, 255] (CFG2 VL[7:0])";
    if (c->nireg != 16) return "nireg must be 16 (4-bit register fields, R8..R15 = ARG0..7)";
    if (c->nsreg < 1 || c->nsreg > 16) return "nsreg must be in [1, 16]";
    if (c->ch_bytes < TSR_BEAT_BYTES || c->ch_bytes % TSR_BEAT_BYTES != 0)
        return "ch_bytes must be a positive multiple of 64";
    if ((uint64_t)(size_t)c->ch_bytes != c->ch_bytes) return "ch_bytes does not fit in size_t";
    if (c->addr_beat_w < 1 || c->addr_beat_w > 40) return "addr_beat_w must be in [1, 40]";
    if (c->caps & ~TSR_SIM_CAPS_IMPLEMENTED) return "caps requests features this simulator does not implement";
    return NULL;
}

static uint8_t * map_channel(uint64_t bytes, const char * dir, uint32_t ch, int * fd_out) {
    *fd_out = -1;
    if (dir == NULL) {
        void * p = mmap(NULL, (size_t)bytes, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (p == MAP_FAILED) {
            fprintf(stderr, "tsr_sim: mmap of %llu bytes for ch%u failed: %s\n",
                    (unsigned long long)bytes, ch, strerror(errno));
            return NULL;
        }
        return (uint8_t *)p;
    }
    const size_t len = strlen(dir) + 64;
    char * path = (char *)xrealloc(NULL, len);
    snprintf(path, len, "%s/tsr-sim-%ld-ch%u-XXXXXX", dir, (long)getpid(), ch);
    const int fd = mkstemp(path);
    if (fd < 0) {
        fprintf(stderr, "tsr_sim: cannot create DRAM file %s: %s\n", path, strerror(errno));
        free(path);
        return NULL;
    }
    unlink(path);   // the mapping keeps it alive; nothing is left behind
    if (ftruncate(fd, (off_t)bytes) != 0) {
        fprintf(stderr, "tsr_sim: ftruncate(%s, %llu) failed: %s\n", path, (unsigned long long)bytes,
                strerror(errno));
        close(fd);
        free(path);
        return NULL;
    }
    void * p = mmap(NULL, (size_t)bytes, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_NORESERVE, fd, 0);
    if (p == MAP_FAILED) {
        fprintf(stderr, "tsr_sim: mmap of DRAM file %s failed: %s\n", path, strerror(errno));
        close(fd);
        free(path);
        return NULL;
    }
    free(path);
    *fd_out = fd;
    return (uint8_t *)p;
}

tsr_sim * tsr_sim_create(const tsr_config * cfg, const char * dram_dir) {
    if (cfg == NULL) {
        fprintf(stderr, "tsr_sim_create: NULL config\n");
        return NULL;
    }
    const char * why = cfg_problem(cfg);
    if (why != NULL) {
        fprintf(stderr, "tsr_sim_create: unsupported config: %s\n", why);
        return NULL;
    }
    tsr_sim * sim = (tsr_sim *)xcalloc(1, sizeof(tsr_sim));
    sim->cfg = *cfg;
    sim->num_ch = cfg->num_ch;
    sim->ch_bytes = cfg->ch_bytes;
    sim->limit_beats = cfg->ch_bytes / TSR_BEAT_BYTES;
    if (sim->limit_beats > (1ull << cfg->addr_beat_w)) sim->limit_beats = 1ull << cfg->addr_beat_w;
    sim->spm_lines = cfg->spm_bytes / TSR_LINE_BYTES;
    atomic_init(&sim->abort_req, 0);
    sim->dram = (uint8_t **)xcalloc(sim->num_ch, sizeof(uint8_t *));
    sim->dram_fd = (int *)xcalloc(sim->num_ch, sizeof(int));
    for (uint32_t c = 0; c < sim->num_ch; ++c) sim->dram_fd[c] = -1;
    for (uint32_t c = 0; c < sim->num_ch; ++c) {
        sim->dram[c] = map_channel(sim->ch_bytes, dram_dir, c, &sim->dram_fd[c]);
        if (sim->dram[c] == NULL) {
            tsr_sim_destroy(sim);
            return NULL;
        }
    }
    sim->spm = (uint8_t *)xrealloc(NULL, cfg->spm_bytes);
    memset(sim->spm, 0xA5, cfg->spm_bytes);
    sim->n_perf = tsr_perf_count(sim->num_ch);
    sim->perf = (uint64_t *)xcalloc(sim->n_perf, sizeof(uint64_t));
    sim->xbuf = (float *)xcalloc(cfg->k_max, sizeof(float));
    sim->threads = 1;
    return sim;
}

void tsr_sim_destroy(tsr_sim * sim) {
    if (sim == NULL) return;
    pool_destroy(sim->pool);
    cap_state_free(sim);
    if (sim->dram != NULL) {
        for (uint32_t c = 0; c < sim->num_ch; ++c) {
            if (sim->dram[c] != NULL) munmap(sim->dram[c], (size_t)sim->ch_bytes);
            if (sim->dram_fd[c] >= 0) close(sim->dram_fd[c]);
        }
    }
    free(sim->dram);
    free(sim->dram_fd);
    free(sim->spm);
    free(sim->perf);
    free(sim->xbuf);
    free(sim->ybuf);
    free(sim->gaps);
    free(sim);
}

const tsr_config * tsr_sim_config(const tsr_sim * sim) { return &sim->cfg; }

uint8_t * tsr_sim_dram(tsr_sim * sim, uint32_t ch) { return ch < sim->num_ch ? sim->dram[ch] : NULL; }
uint8_t * tsr_sim_spm(tsr_sim * sim) { return sim->spm; }
uint64_t  tsr_sim_dram_limit_beats(const tsr_sim * sim) { return sim->limit_beats; }

void tsr_sim_set_log(tsr_sim * sim, void * file) { sim->log = (FILE *)file; }

void tsr_sim_set_threads(tsr_sim * sim, unsigned n) {
    if (n < 1) n = 1;
    if (n > 64) n = 64;
    if (n == sim->threads) return;
    pool_destroy(sim->pool);
    sim->pool = NULL;
    sim->threads = n;
    if (n > 1) sim->pool = pool_create(n - 1);
}

int32_t tsr_sim_ireg(const tsr_sim * sim, uint32_t i) { return i < 16 ? (int32_t)sim->R[i] : 0; }
float   tsr_sim_sreg(const tsr_sim * sim, uint32_t i) { return i < 16 ? sim->S[i] : 0.0f; }
uint64_t tsr_sim_perf(const tsr_sim * sim, uint32_t idx) { return idx < sim->n_perf ? sim->perf[idx] : 0; }

// ---------------------------------------------------------------------------
// CSR port
// ---------------------------------------------------------------------------
static void soft_reset(tsr_sim * sim) {
    // Execution state and counters; host-programmed CMD_ADDR/CMD_COUNT/ARG* are kept,
    // SPM and DRAM contents are untouched.
    sim->status = 0;
    sim->err_pc = sim->err_code = sim->err_info = sim->pc = 0;
    memset(sim->R, 0, sizeof(sim->R));
    memset(sim->S, 0, sizeof(sim->S));
    memset(sim->perf, 0, sim->n_perf * sizeof(uint64_t));
    atomic_store(&sim->abort_req, 0);
}

static void start(tsr_sim * sim) {
    if (sim->status & TSR_STATUS_BUSY) return;   // START while busy is ignored
    sim->run_cmd_addr = ((uint64_t)sim->cmd_addr_hi << 32) | sim->cmd_addr_lo;
    sim->run_cmd_count = sim->cmd_count;
    for (int i = 0; i < 8; ++i) sim->R[i] = 0;
    for (int i = 0; i < 8; ++i) sim->R[8 + i] = sim->args[i];
    for (int i = 0; i < 16; ++i) sim->S[i] = 0.0f;
    sim->status = TSR_STATUS_BUSY;
    sim->err_pc = sim->err_code = sim->err_info = sim->pc = 0;
    atomic_store(&sim->abort_req, 0);
}

uint32_t tsr_sim_csr_read(tsr_sim * sim, uint32_t off) {
    const tsr_config * c = &sim->cfg;
    switch (off) {
    case TSR_CSR_ID:          return TSR_ID_MAGIC;
    case TSR_CSR_VERSION:     return (TSR_ISA_MAJOR << 8) | TSR_ISA_MINOR;
    case TSR_CSR_CFG0:        return (c->num_ch & 0xFFu) | ((c->lanes & 0xFFFu) << 8) | ((c->t_max & 0xFu) << 20);
    case TSR_CSR_CFG1:        return c->spm_bytes / TSR_LINE_BYTES;
    case TSR_CSR_CFG2:        return (c->vl & 0xFFu) | ((c->nsreg & 0xFFu) << 8) | (((c->k_max / 256u) & 0xFFFFu) << 16);
    case TSR_CSR_CAPS:        return c->caps;
    case TSR_CSR_CTRL:        return 0;
    case TSR_CSR_STATUS:      return sim->status;
    case TSR_CSR_CMD_ADDR_LO: return sim->cmd_addr_lo;
    case TSR_CSR_CMD_ADDR_HI: return sim->cmd_addr_hi;
    case TSR_CSR_CMD_COUNT:   return sim->cmd_count;
    case TSR_CSR_ERR_PC:      return sim->err_pc;
    case TSR_CSR_ERR_CODE:    return sim->err_code;
    case TSR_CSR_ERR_INFO:    return sim->err_info;
    case TSR_CSR_PC:          return sim->pc;
    default: break;
    }
    if (off % 4 != 0) return 0;
    if (off >= TSR_CSR_ARG0 && off < TSR_CSR_ARG0 + 32) return sim->args[(off - TSR_CSR_ARG0) / 4];
    if (off >= TSR_CSR_PERF_BASE) {
        const uint32_t idx = (off - TSR_CSR_PERF_BASE) / 8;
        if (idx < sim->n_perf) {
            const uint64_t v = sim->perf[idx];
            return ((off - TSR_CSR_PERF_BASE) % 8) ? (uint32_t)(v >> 32) : (uint32_t)v;
        }
    }
    return 0;
}

void tsr_sim_csr_write(tsr_sim * sim, uint32_t off, uint32_t val) {
    switch (off) {
    case TSR_CSR_CTRL:
        if (val & TSR_CTRL_SOFT_RESET) soft_reset(sim);
        if (val & TSR_CTRL_START) start(sim);
        if (val & TSR_CTRL_ABORT) atomic_store(&sim->abort_req, 1);   // cleared by START
        return;
    case TSR_CSR_STATUS:      sim->status &= ~(val & (TSR_STATUS_DONE | TSR_STATUS_ERR)); return;
    case TSR_CSR_CMD_ADDR_LO: sim->cmd_addr_lo = val; return;
    case TSR_CSR_CMD_ADDR_HI: sim->cmd_addr_hi = val; return;
    case TSR_CSR_CMD_COUNT:   sim->cmd_count = val; return;
    default: break;
    }
    if (off % 4 == 0 && off >= TSR_CSR_ARG0 && off < TSR_CSR_ARG0 + 32) sim->args[(off - TSR_CSR_ARG0) / 4] = val;
    // everything else is read-only or unmapped: ignored
}

// ---------------------------------------------------------------------------
// Instructions
// ---------------------------------------------------------------------------
static inline int words_zero_from(const uint32_t * w, unsigned from) {
    for (unsigned i = from; i < TSR_INSN_WORDS; ++i) {
        if (w[i] != 0) return 0;
    }
    return 1;
}

static inline void set_reg(tsr_sim * sim, uint32_t rd, uint32_t v) {
    if (rd != 0) sim->R[rd] = v;   // R0 writes are discarded
}

static inline int spm_ok(const tsr_sim * sim, uint32_t a, uint64_t extent) {
    return (uint64_t)a + extent <= sim->spm_lines;
}
static inline int dram_ok(const tsr_sim * sim, uint32_t a, uint64_t extent) {
    return (uint64_t)a + extent <= sim->limit_beats;
}

static uint32_t ex_ldst(tsr_sim * sim, const uint32_t * w, int is_st, uint32_t * info, uint64_t * cyc) {
    // used: sub = ch, ra (A1 SPM), rb (A2 DRAM), w1..w3
    if ((w[0] & 0xFF00F000u) != 0 || !words_zero_from(w, 4)) return TSR_ERR_ILLEGAL_OP;
    const uint32_t ch = tsr_w0_sub(w[0]);
    if (ch >= sim->num_ch) return TSR_ERR_ILLEGAL_OP;
    const uint32_t spm  = sim->R[tsr_w0_ra(w[0])] + w[1];
    const uint32_t dram = sim->R[tsr_w0_rb(w[0])] + w[2];
    const uint32_t n    = w[3];
    if (!spm_ok(sim, spm, n))   { *info = spm;  return TSR_ERR_SPM_RANGE; }
    if (!dram_ok(sim, dram, n)) { *info = dram; return TSR_ERR_DRAM_RANGE; }
    uint8_t * s = sim->spm + (size_t)spm * TSR_LINE_BYTES;
    uint8_t * d = sim->dram[ch] + (uint64_t)dram * TSR_BEAT_BYTES;
    const size_t bytes = (size_t)n * TSR_LINE_BYTES;
    if (!is_st) {
        cap_read(sim, ch, dram, n);
        sim->perf[tsr_perf_ch_rd_beats(ch)] += n;
        memcpy(s, d, bytes);
    } else {
        memcpy(d, s, bytes);
        cap_write(sim, ch, dram, n);
    }
    *cyc = CYC_ISSUE + CYC_DMA_LAT + n;
    return TSR_ERR_NONE;
}

static uint32_t ex_qact(tsr_sim * sim, const uint32_t * w, uint32_t * info, uint64_t * cyc) {
    // used: sub = kind, ra (A1 QA dst), rb (A2 f32 src), w1..w5
    if ((w[0] & 0xFF00F000u) != 0 || !words_zero_from(w, 6)) return TSR_ERR_ILLEGAL_OP;
    const uint32_t kind = tsr_w0_sub(w[0]);
    if (kind != TSR_QA_A32 && kind != TSR_QA_A256) return TSR_ERR_ILLEGAL_OP;
    const uint32_t K = w[3], T = w[4];
    const uint32_t bs = tsr_qa_block_size(kind);
    if (K == 0 || K % bs != 0 || K > sim->cfg.k_max) return TSR_ERR_K_RANGE;
    if (T == 0 || T > sim->cfg.t_max) return TSR_ERR_T_RANGE;
    const uint32_t x_lines  = K * 4u / TSR_LINE_BYTES;          // exact: K % 32 == 0
    const uint64_t sstride  = w[5] ? w[5] : x_lines;             // 구현 명확화: stride 0
    const uint32_t qa_lines = tsr_qa_lines(kind, K);
    const uint32_t dst = sim->R[tsr_w0_ra(w[0])] + w[1];
    const uint32_t src = sim->R[tsr_w0_rb(w[0])] + w[2];
    if (!spm_ok(sim, dst, (uint64_t)T * qa_lines))                 { *info = dst; return TSR_ERR_SPM_RANGE; }
    if (!spm_ok(sim, src, (uint64_t)(T - 1) * sstride + x_lines)) { *info = src; return TSR_ERR_SPM_RANGE; }
    for (uint32_t t = 0; t < T; ++t) {
        const uint8_t * s = sim->spm + ((uint64_t)src + (uint64_t)t * sstride) * TSR_LINE_BYTES;
        for (uint32_t k = 0; k < K; ++k) {
            const uint32_t u = rd32le(s + 4u * k);
            memcpy(&sim->xbuf[k], &u, 4);
        }
        tsr_qact(kind, sim->xbuf, K, sim->spm + ((uint64_t)dst + (uint64_t)t * qa_lines) * TSR_LINE_BYTES);
    }
    *cyc = CYC_ISSUE + (uint64_t)T * (tsr_ceil_div_u32(K, sim->cfg.vl) + qa_lines);
    return TSR_ERR_NONE;
}

typedef struct {
    tsr_sim *  sim;
    uint32_t   fmt, N, K, T;
    uint32_t   woff;
    uint64_t   x_line;
    uint32_t   qa_lines;
} mvm_job;

static void mvm_rows(void * vctx, uint32_t lo, uint32_t hi) {
    const mvm_job * j = (const mvm_job *)vctx;
    tsr_sim * sim = j->sim;
    for (uint32_t n = lo; n < hi; ++n) {
        uint32_t ch;
        uint64_t off;
        tsr_row_location(j->fmt, j->N, j->K, sim->num_ch, n, &ch, &off);
        const uint8_t * row = sim->dram[ch] + ((uint64_t)j->woff + off) * TSR_BEAT_BYTES;
        for (uint32_t t = 0; t < j->T; ++t) {
            const uint8_t * qa = sim->spm + (j->x_line + (uint64_t)t * j->qa_lines) * TSR_LINE_BYTES;
            sim->ybuf[(size_t)t * j->N + n] = tsr_dot_row(j->fmt, row, j->K, qa);
        }
    }
}

static uint32_t ex_mvm(tsr_sim * sim, const uint32_t * w, uint32_t * info, uint64_t * cyc) {
    // used: sub = fmt, flags bit0 ACC, ra (A1 dst), rb (A2 x), rc (A3 W_OFF), w1..w6 (w5[19:0])
    if ((w[0] & 0xF000E000u) != 0 || (w[5] & 0xFFF00000u) != 0 || w[7] != 0) return TSR_ERR_ILLEGAL_OP;
    const uint32_t fmt = tsr_w0_sub(w[0]);
    if (!tsr_fmt_valid(fmt)) return TSR_ERR_BAD_FMT;
    const int acc = (tsr_w0_flags(w[0]) & TSR_MVM_FLAG_ACC) != 0;
    const uint32_t N = w[4];
    const uint32_t K = w[5] & 0xFFFFu;
    const uint32_t T = (w[5] >> 16) & 0xFu;
    if (!tsr_fmt_k_valid(fmt, K, sim->cfg.k_max)) return TSR_ERR_K_RANGE;
    if (T == 0 || T > sim->cfg.t_max) return TSR_ERR_T_RANGE;
    const uint32_t kind     = tsr_qa_kind_for_fmt(fmt);
    const uint32_t qa_lines = tsr_qa_lines(kind, K);
    const uint64_t y_lines  = tsr_ceil_div_u64(4ull * N, TSR_LINE_BYTES);
    const uint64_t dstride  = w[6] ? w[6] : y_lines;
    const uint32_t dst  = sim->R[tsr_w0_ra(w[0])] + w[1];
    const uint32_t x    = sim->R[tsr_w0_rb(w[0])] + w[2];
    const uint32_t woff = sim->R[tsr_w0_rc(w[0])] + w[3];
    if (!spm_ok(sim, dst, (uint64_t)(T - 1) * dstride + y_lines)) { *info = dst; return TSR_ERR_SPM_RANGE; }
    if (!spm_ok(sim, x, (uint64_t)T * qa_lines))                  { *info = x;   return TSR_ERR_SPM_RANGE; }
    // N <= 16 * spm_lines < 2^31 here (the dst check bounds it), so 32-bit row math is safe.
    const uint64_t R = tsr_ceil_div_u64(N, sim->num_ch);
    const uint32_t row_beats = tsr_row_beats(fmt, K);
    if (!dram_ok(sim, woff, R * row_beats)) { *info = woff; return TSR_ERR_DRAM_RANGE; }

    // Weight stream: channel c reads its rows [cR, min((c+1)R, N)) = rows_c * ROW_BEATS beats.
    uint64_t max_beats = 0;
    for (uint32_t c = 0; c < sim->num_ch; ++c) {
        const uint64_t first = (uint64_t)c * R;
        const uint64_t rows = (N > first) ? ((N - first < R) ? N - first : R) : 0;
        const uint64_t beats = rows * row_beats;
        if (beats == 0) continue;
        cap_read(sim, c, woff, beats);
        sim->perf[tsr_perf_ch_rd_beats(c)] += beats;
        if (beats > max_beats) max_beats = beats;
    }

    // Compute every y first (x is read-only here; dst/x overlap is undefined per spec).
    const size_t ny = (size_t)T * N;
    if (ny > sim->ybuf_cap) {
        sim->ybuf = (float *)xrealloc(sim->ybuf, ny * sizeof(float));
        sim->ybuf_cap = ny;
    }
    mvm_job job = { sim, fmt, N, K, T, woff, x, qa_lines };
    const uint64_t work = (uint64_t)N * row_beats * T;
    if (sim->pool != NULL && N > 1 && work >= MT_MIN_WORK_BEATS) {
        uint32_t chunk = N / (8u * sim->threads);
        if (chunk < 1) chunk = 1;
        if (chunk > 64) chunk = 64;
        pool_run(sim->pool, mvm_rows, &job, N, chunk);
    } else {
        mvm_rows(&job, 0, N);
    }

    // Write: token t -> dst + t*stride, element n; ACC adds the old value; tail bytes zero.
    for (uint32_t t = 0; t < T; ++t) {
        uint8_t * base = sim->spm + ((uint64_t)dst + (uint64_t)t * dstride) * TSR_LINE_BYTES;
        const float * y = sim->ybuf + (size_t)t * N;
        for (uint32_t n = 0; n < N; ++n) {
            float v = y[n];
            if (acc) {
                const uint32_t ou = rd32le(base + 4u * n);
                float old;
                memcpy(&old, &ou, 4);
                v = tsr_fadd(old, v);
            }
            wr32le(base + 4u * n, tsr_f32_bits(v));
        }
        memset(base + 4ull * N, 0, (size_t)(y_lines * TSR_LINE_BYTES - 4ull * N));
    }
    const uint64_t mvm_cyc = CYC_MVM_FILL + (uint64_t)T * qa_lines + max_beats;
    sim->perf[TSR_PERF_MVM_CYCLES] += mvm_cyc;
    *cyc = CYC_ISSUE + mvm_cyc;
    return TSR_ERR_NONE;
}

// Executes one instruction. Returns 0 or an error code (then *info is ERR_INFO and no state
// has changed). *end is set by END.
static uint32_t exec_insn(tsr_sim * sim, const uint32_t * w, uint32_t * info, uint64_t * cyc, int * end) {
    const uint32_t w0 = w[0];
    switch (tsr_w0_opcode(w0)) {
    case TSR_OP_NOP:
    case TSR_OP_FENCE:   // no execution overlap in this model: same as NOP
        if ((w0 & ~0xFFu) != 0 || !words_zero_from(w, 1)) return TSR_ERR_ILLEGAL_OP;
        *cyc = CYC_ISSUE;
        return TSR_ERR_NONE;
    case TSR_OP_END:
        if ((w0 & ~0xFFu) != 0 || !words_zero_from(w, 1)) return TSR_ERR_ILLEGAL_OP;
        *cyc = CYC_ISSUE;
        *end = 1;
        return TSR_ERR_NONE;
    case TSR_OP_LI:      // used: rd, w1
        if ((w0 & 0x0FFFFF00u) != 0 || !words_zero_from(w, 2)) return TSR_ERR_ILLEGAL_OP;
        set_reg(sim, tsr_w0_rd(w0), w[1]);
        *cyc = CYC_ISSUE;
        return TSR_ERR_NONE;
    case TSR_OP_ADDI:    // used: rd, ra, w1
        if ((w0 & 0x0FF0FF00u) != 0 || !words_zero_from(w, 2)) return TSR_ERR_ILLEGAL_OP;
        set_reg(sim, tsr_w0_rd(w0), sim->R[tsr_w0_ra(w0)] + w[1]);
        *cyc = CYC_ISSUE;
        return TSR_ERR_NONE;
    case TSR_OP_LD:   return ex_ldst(sim, w, 0, info, cyc);
    case TSR_OP_ST:   return ex_ldst(sim, w, 1, info, cyc);
    case TSR_OP_QACT: return ex_qact(sim, w, info, cyc);
    case TSR_OP_MVM:  return ex_mvm(sim, w, info, cyc);
    default:
        // MULI/RLD/LOOP (CAPS bit2), WAITM/JUMP (bit4), VOP/RED/ROPE/SOP (bit0), TOPK (bit3),
        // DQROW (M7) and every undefined opcode: not implemented by this simulator.
        return TSR_ERR_ILLEGAL_OP;
    }
}

static const char * op_name(uint32_t op) {
    switch (op) {
    case TSR_OP_NOP: return "NOP";     case TSR_OP_END: return "END";     case TSR_OP_LI: return "LI";
    case TSR_OP_ADDI: return "ADDI";   case TSR_OP_MULI: return "MULI";   case TSR_OP_RLD: return "RLD";
    case TSR_OP_LOOP: return "LOOP";   case TSR_OP_WAITM: return "WAITM"; case TSR_OP_JUMP: return "JUMP";
    case TSR_OP_LD: return "LD";       case TSR_OP_ST: return "ST";       case TSR_OP_DQROW: return "DQROW";
    case TSR_OP_QACT: return "QACT";   case TSR_OP_MVM: return "MVM";     case TSR_OP_VOP: return "VOP";
    case TSR_OP_RED: return "RED";     case TSR_OP_ROPE: return "ROPE";   case TSR_OP_TOPK: return "TOPK";
    case TSR_OP_SOP: return "SOP";     case TSR_OP_FENCE: return "FENCE";
    default: return "?";
    }
}

static void log_insn(tsr_sim * sim, uint32_t pc, const uint32_t * w, uint32_t err, uint32_t info) {
    const uint32_t w0 = w[0];
    fprintf(sim->log, "%8u %-5s sub=%u fl=%u ra=%u rb=%u rc=%u rd=%u w1-7=%08x %08x %08x %08x %08x %08x %08x",
            pc, op_name(tsr_w0_opcode(w0)), tsr_w0_sub(w0), tsr_w0_flags(w0), tsr_w0_ra(w0), tsr_w0_rb(w0),
            tsr_w0_rc(w0), tsr_w0_rd(w0), w[1], w[2], w[3], w[4], w[5], w[6], w[7]);
    if (err) fprintf(sim->log, "  -> ERR %u info=0x%08x", err, info);
    fputc('\n', sim->log);
}

static void stop_err(tsr_sim * sim, uint32_t code, uint32_t pc, uint32_t info) {
    sim->status = TSR_STATUS_DONE | TSR_STATUS_ERR;
    sim->err_code = code;
    sim->err_pc = pc;
    sim->err_info = info;
}

uint32_t tsr_sim_run(tsr_sim * sim) {
    if (!(sim->status & TSR_STATUS_BUSY)) return sim->status;
    const uint64_t limit = sim->limit_beats;
    uint8_t beat[TSR_BEAT_BYTES];
    uint64_t cur_beat = 0;
    int have_beat = 0;
    uint32_t pc = 0;
    for (;;) {
        if (atomic_load(&sim->abort_req)) {
            stop_err(sim, TSR_ERR_ABORTED, pc, 0);
            if (sim->log) fprintf(sim->log, "%8u ABORTED\n", pc);
            break;
        }
        if (pc >= sim->run_cmd_count) {          // ran past the buffer without END
            sim->pc = pc;
            stop_err(sim, TSR_ERR_ILLEGAL_OP, sim->run_cmd_count, 0);
            if (sim->log) fprintf(sim->log, "%8u past CMD_COUNT -> ERR %u\n", pc, TSR_ERR_ILLEGAL_OP);
            break;
        }
        sim->pc = pc;
        const uint64_t boff = pc / 2u;
        if (sim->run_cmd_addr >= limit || boff >= limit - sim->run_cmd_addr) {
            const uint32_t info = (uint32_t)(sim->run_cmd_addr + boff);
            stop_err(sim, TSR_ERR_DRAM_RANGE, pc, info);
            if (sim->log) fprintf(sim->log, "%8u fetch beyond DRAM -> ERR %u info=0x%08x\n", pc, TSR_ERR_DRAM_RANGE, info);
            break;
        }
        const uint64_t b = sim->run_cmd_addr + boff;
        if (!have_beat || b != cur_beat) {
            cap_read(sim, 0, b, 1);
            sim->perf[tsr_perf_ch_rd_beats(0)] += 1;
            memcpy(beat, sim->dram[0] + b * TSR_BEAT_BYTES, TSR_BEAT_BYTES);
            cur_beat = b;
            have_beat = 1;
        }
        uint32_t w[TSR_INSN_WORDS];
        for (unsigned i = 0; i < TSR_INSN_WORDS; ++i) w[i] = rd32le(beat + TSR_INSN_BYTES * (pc % 2u) + 4u * i);
        uint32_t info = 0;
        uint64_t cyc = 0;
        int end = 0;
        const uint32_t err = exec_insn(sim, w, &info, &cyc, &end);
        if (sim->log) log_insn(sim, pc, w, err, info);
        if (err != TSR_ERR_NONE) {
            stop_err(sim, err, pc, info);
            break;
        }
        sim->perf[TSR_PERF_RETIRED] += 1;
        sim->perf[TSR_PERF_CYCLES] += cyc;
        if (end) {
            sim->status = TSR_STATUS_DONE;
            break;
        }
        ++pc;
    }
    sim->status &= ~TSR_STATUS_BUSY;
    if (sim->log) fflush(sim->log);
    return sim->status;
}

uint32_t tsr_sim_exec(tsr_sim * sim, uint64_t cmd_addr_beat, uint32_t cmd_count, const uint32_t args[8]) {
    tsr_sim_csr_write(sim, TSR_CSR_CMD_ADDR_LO, (uint32_t)cmd_addr_beat);
    tsr_sim_csr_write(sim, TSR_CSR_CMD_ADDR_HI, (uint32_t)(cmd_addr_beat >> 32));
    tsr_sim_csr_write(sim, TSR_CSR_CMD_COUNT, cmd_count);
    for (uint32_t i = 0; i < 8; ++i) tsr_sim_csr_write(sim, TSR_CSR_ARG0 + 4u * i, args ? args[i] : 0u);
    tsr_sim_csr_write(sim, TSR_CSR_CTRL, TSR_CTRL_START);
    return tsr_sim_run(sim);
}
