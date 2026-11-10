// tsr_sim.c — Tesseract simulator, core: memories, the host CSR port (EX §5), the firmware loader
// (EX §3), the launch queue / completion ring / log ring (EX §4), SOFT_RESET (EX §8.5), PERF
// (EX §6), the op handler table and the tile units NOP LD ST QACT MVM FENCE (EX §7.2).
//
// Normative sources: docs/spec.md "ISA", "MVM semantics", "Implementation clarifications", docs/design/01-execution-
// model.md (EX), 04-control-core.md (CC) and 06-software-verification.md (SW) §4. All numerics
// come from tsr_ref.c (spec rule 3); this file only decodes, checks and moves bytes. The queue
// model (dispatcher, checker order, unit queues, SYNC, errors, the RISC-V side control window and
// the time base) is tsr_q.c; the interface between the two and toward the other sim files is
// tsr_sim_int.h.
//
// Decisions pinned in spec §21 (Implementation clarifications, simulator section) and EX §7.1, implemented here:
//   - Reserved fields: every word, flag bit and sub value an op does not use, and x16
//     (w0[31:16], the former register fields), must be zero, else ILLEGAL_OP. LD/ST sub = channel
//     >= NUM_CH -> ILLEGAL_OP. QACT sub (QA kind) >= 2 -> ILLEGAL_OP. MVM sub (fmt) >= 2 -> BAD_FMT.
//   - Check order ILLEGAL_OP -> BAD_FMT -> K_RANGE -> T_RANGE -> SPM_RANGE -> DRAM_RANGE (BAD_DEP
//     and OPERAND are the queue model's / the VU and AT ops'); inside SPM_RANGE / DRAM_RANGE the
//     operands are checked in field order A1, A2, A3.
//   - Range check: an operand with start address a (the 32-bit immediate) and extent e
//     (lines/beats, 64-bit, for T strided segments e = (T-1)*stride + len) is in range iff
//     a + e <= limit, computed without wrap. ERR_INFO = a (the operand's start address). The
//     checker reports [a, a + e) as the operand's tagged extent (tsr_sim_int.h).
//     SPM limit = SPM_BYTES/64 lines; DRAM limit = min(CH_BYTES/64, 2^ADDR_BEAT_W) beats.
//
// Cycle model (unit busy time, the handlers' cycles(); deterministic, informational — not part of
// the bit-exact contract):
//   LD / ST                 64 (DMA latency) + n_lines
//   QACT                    T * (ceil(K / VL) + QA_LINES(kind, K))
//   MVM                     CYC_MVM_FILL (pipeline fill/drain) + T * QA_LINES (activation broadcast)
//                           + max over channels of rows_c * ROW_BEATS (1 weight beat/cycle); with CAPS.MVM_DUAL, the
//                           run-time enable and T <= T_MAX / 2 the dual weight stream: ceil(rows_c / 2) * ROW_BEATS;
//                           the optional memory rate bound (tsr_sim_opts.mem_rate) raises a channel to at least
//                           rows_c * ROW_BEATS / mem_rate
//   NOP, FENCE              none (SYNC entries complete without running: 1 cycle in the queue model)
// The queue model (tsr_q.c) runs ops for these times and integrates PERF over time (EX §6:
// time counters gated by ACTIVE and not QHALT, event counters by ACTIVE). The tsr_sim_op test
// hook keeps the op-level counting instead: 1 issue cycle + cycles() per passing op in
// PERF_CYCLES, cycles() in MVM_CYCLES / VU_CYCLES (and ATT_QK / ATT_PV_CYCLES for ATTQK / ATTPV),
// RETIRED for every op, nothing gated.
//   PERF_CH_RD_BEATS[c]     LD beats and MVM weight beats (no command fetch: ch0 is exact)
//   PERF_CH_WAIT            0 (no DRAM stalls modeled)

#define _GNU_SOURCE
#define TSR_SIM_CORE_INTERNAL

#include "tsr_sim.h"
#include "tsr_local_tile.h"
#include "tsr_ref_math.h"
#include "tsr_sim_int.h"
#include "tsr_ref.h"
#include "tsr_rv.h"       // TSR_RV_CORE_* (header only: the option is parsed with or without tsr_rv.c)
#if defined(TSR_SIM_HAVE_RV) && TSR_SIM_HAVE_RV
#include "tsr_bus.h"      // the built-in RISC-V agent (tsr_bus.c, tsr_rv.c)
#endif

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#define CYC_ISSUE    1u      // tsr_sim_op only: the issue cycle
// SW-80: the platform's DDR latencies as the card shows them (the trace writer's run times of the DMA and MVM ops of Qwen3.8 2B
// and Gemma 4 26B-A4B decode; the RTL benches run on ideal memory): a load takes its lines plus 64 cycles, a store its lines plus
// 110 (54 up to 8 lines: the write response), an MVM 144 cycles before its first weight beat is used.
#define CYC_DMA_LAT  64u
#define CYC_ST_LAT   110u
#define CYC_ST_SMALL 54u
#define CYC_MVM_FILL 144u

#define RV_RATIO_DEFAULT (166.7 / 276.3)   // k4 clocks (SW §4.2)

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
struct tsr_cap_ch {
    iset seen;         // init chunks + written beats
    iset written;
    cap_chunk * c; size_t n, cap;
};
typedef struct tsr_cap_ch cap_ch;

// ---------------------------------------------------------------------------
// Worker pool for MVM rows
// ---------------------------------------------------------------------------
struct tsr_pool {
    unsigned        n;           // worker threads (the caller works too)
    pthread_t *     th;
    pthread_mutex_t mu;
    pthread_cond_t  cv_go, cv_done;
    uint64_t        gen;
    unsigned        busy;
    int             quit;
    tsr_row_fn      fn;
    void *          ctx;
    uint32_t        total, chunk;
    atomic_uint     next;
};
typedef struct tsr_pool pool;

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

static void pool_run(pool * p, tsr_row_fn fn, void * ctx, uint32_t total, uint32_t chunk) {
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
// Op handler table (tsr_sim_int.h §2). Per process; the defaults register once, before the first
// simulator or the first explicit registration, so a test's own registration always wins.
// ---------------------------------------------------------------------------
static tsr_op_handler  g_ops[256];
static uint8_t         g_have[256];
static pthread_mutex_t g_ops_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_once_t  g_ops_once = PTHREAD_ONCE_INIT;
static _Thread_local int g_in_defaults;

static void register_defaults(void) {
    g_in_defaults = 1;
    tsr_core_register();
#if defined(TSR_SIM_HAVE_VU) && TSR_SIM_HAVE_VU
    tsr_vu_register();
#endif
#if defined(TSR_SIM_HAVE_AT) && TSR_SIM_HAVE_AT
    tsr_at_register();
#endif
    g_in_defaults = 0;
}
static void ensure_defaults(void) {
    if (!g_in_defaults) pthread_once(&g_ops_once, register_defaults);
}

void tsr_sim_register_op(uint8_t opcode, const tsr_op_handler * h) {
    ensure_defaults();
    pthread_mutex_lock(&g_ops_mu);
    if (h != NULL) {
        g_ops[opcode] = *h;
        g_have[opcode] = 1;
    } else {
        memset(&g_ops[opcode], 0, sizeof(g_ops[opcode]));
        g_have[opcode] = 0;
    }
    pthread_mutex_unlock(&g_ops_mu);
}

const tsr_op_handler * tsr_sim_op_handler(uint8_t opcode) {
    ensure_defaults();
    return g_have[opcode] ? &g_ops[opcode] : NULL;
}

uint32_t tsr_sim_caps_implemented(void) {
    ensure_defaults();
    static const uint8_t vu[] = { TSR_OP_VOP, TSR_OP_RED, TSR_OP_ROPE, TSR_OP_SOP };
    static const uint8_t at[] = { TSR_OP_KVST, TSR_OP_ATTQK, TSR_OP_ATTPV, TSR_OP_ASMAX, TSR_OP_AMASK,
                                  TSR_OP_AMERGE, TSR_OP_ANORM };
    uint32_t caps = 0, all = 1;
    for (size_t i = 0; i < sizeof(vu); ++i) all &= g_have[vu[i]];
    if (all) caps |= TSR_CAPS_VU | TSR_CAPS_MVM_LOCAL_SCALE | TSR_CAPS_MVM_LOCAL_SOFTCAP;
    all = 1;
    for (size_t i = 0; i < sizeof(at); ++i) all &= g_have[at[i]];
    if (all) caps |= TSR_CAPS_ATT | TSR_CAPS_ATT_LOCAL;
    if (g_have[TSR_OP_TOPK])  caps |= TSR_CAPS_TOPK;
    if (g_have[TSR_OP_DQROW]) caps |= TSR_CAPS_DQROW;
    if (g_have[TSR_OP_GDN])   caps |= TSR_CAPS_GDN;
    if (g_have[TSR_OP_VMASK]) caps |= TSR_CAPS_VMASK;
    return caps;
}

const tsr_op_handler * tsr_sim_accept_op(const tsr_sim * sim, uint32_t opcode) {
    opcode &= 0xFFu;
    if (!tsr_op_legal(opcode)) return NULL;
    const uint32_t need = tsr_op_caps(opcode);
    if ((sim->cfg.caps & need) != need) return NULL;
    const uint32_t u = tsr_op_unit(opcode);
    if (u >= TSR_NUM_UNITS || sim->u[u].depth == 0) return NULL;
    const tsr_op_handler * h = tsr_sim_op_handler((uint8_t)opcode);
    if (h == NULL || h->check == NULL || h->unit != u) return NULL;
    return h;
}

// ---------------------------------------------------------------------------
// Capture
// ---------------------------------------------------------------------------
void tsr_sim_cap_read(tsr_sim * sim, uint32_t ch, uint64_t lo, uint64_t n) {
#if defined(TSR_SIM_HAVE_RACE) && TSR_SIM_HAVE_RACE
    if (sim->models[TSR_MODEL_RACE] != NULL) tsr_race_cap_note(sim, ch, lo, n, 0);   // exact DRAM access sets
#endif
    tsr_sim_cap_preserve(sim, ch, lo, n);
}

void tsr_sim_cap_preserve(tsr_sim * sim, uint32_t ch, uint64_t lo, uint64_t n) {
    if (!sim->capturing || n == 0 || ch >= sim->num_ch) return;
    cap_ch * cc = &sim->cap[ch];
    const uint64_t hi = lo + n;
    ival * gaps = (ival *)sim->gaps;
    // Collect the parts of [lo, hi) not yet seen (neither captured nor written in this run).
    size_t ng = 0;
    uint64_t cur = lo;
    for (size_t k = iset_lb(&cc->seen, lo + 1); k < cc->seen.n && cc->seen.v[k].lo < hi; ++k) {
        if (cc->seen.v[k].lo > cur) {
            if (ng == sim->gaps_cap) {
                sim->gaps_cap = sim->gaps_cap ? 2 * sim->gaps_cap : 16;
                sim->gaps = gaps = (ival *)xrealloc(gaps, sim->gaps_cap * sizeof(ival));
            }
            gaps[ng].lo = cur;
            gaps[ng].hi = cc->seen.v[k].lo;
            ++ng;
        }
        if (cc->seen.v[k].hi > cur) cur = cc->seen.v[k].hi;
    }
    if (cur < hi) {
        if (ng == sim->gaps_cap) {
            sim->gaps_cap = sim->gaps_cap ? 2 * sim->gaps_cap : 16;
            sim->gaps = gaps = (ival *)xrealloc(gaps, sim->gaps_cap * sizeof(ival));
        }
        gaps[ng].lo = cur;
        gaps[ng].hi = hi;
        ++ng;
    }
    for (size_t g = 0; g < ng; ++g) {
        const uint64_t glo = gaps[g].lo, ghi = gaps[g].hi;
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

void tsr_sim_cap_write(tsr_sim * sim, uint32_t ch, uint64_t lo, uint64_t n) {
#if defined(TSR_SIM_HAVE_RACE) && TSR_SIM_HAVE_RACE
    if (sim->models[TSR_MODEL_RACE] != NULL) tsr_race_cap_note(sim, ch, lo, n, 1);   // exact DRAM access sets
#endif
    if (!sim->capturing || n == 0 || ch >= sim->num_ch) return;
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
    if (c->nsreg < 1 || c->nsreg > 16) return "nsreg must be in [1, 16]";
    if (c->ch_bytes < TSR_BEAT_BYTES || c->ch_bytes % TSR_BEAT_BYTES != 0)
        return "ch_bytes must be a positive multiple of 64";
    if ((uint64_t)(size_t)c->ch_bytes != c->ch_bytes) return "ch_bytes does not fit in size_t";
    if (c->addr_beat_w < 1 || c->addr_beat_w > 40) return "addr_beat_w must be in [1, 40]";
    if ((c->caps & TSR_CAPS_ATT_LOCAL) && !(c->caps & TSR_CAPS_ATT))
        return "ATT_LOCAL requires CAPS.ATT";
    if ((c->caps & (TSR_CAPS_MVM_LOCAL_SCALE | TSR_CAPS_MVM_LOCAL_SOFTCAP)) && !(c->caps & TSR_CAPS_VU))
        return "local MVM epilogues require CAPS.VU";
    if (c->caps & ~(tsr_sim_caps_implemented() | TSR_CAPS_MVM_DUAL)) return "caps requests features this simulator does not implement";
    return NULL;
}

#if defined(TSR_SIM_HAVE_RACE) && TSR_SIM_HAVE_RACE
#define HAVE_RACE_MODEL 1
#else
#define HAVE_RACE_MODEL 0
#endif

static const char * opts_problem(const tsr_sim_opts * o) {
    if (o->race > 2u) return "race must be 0 (off), 1 (report) or 2 (fatal)";
    if (o->race != 0 && !HAVE_RACE_MODEL) return "race: this simulator library was built without the race detector (tsr_race.c)";
    if (o->rv_ratio < 0.0 || (o->rv_ratio != 0.0 && (o->rv_ratio < 0.05 || o->rv_ratio > 4.0)))
        return "rv_ratio must be in [0.05, 4] (SW §4.9)";
    if (o->log_flags & ~(TSR_SIM_LOG_RV | TSR_SIM_LOG_WORDS)) return "log_flags: unknown bits";
    if (o->q_depth != NULL) {
        for (uint32_t u = 0; u < TSR_NUM_UNITS; ++u)
            if (o->q_depth[u] > TSR_QDEPTH_MAX) return "q_depth: a depth is above 31 (5-bit Q_DEPTH field)";
        if (o->q_depth[TSR_UNIT_SYNC] == 0) return "q_depth: the SYNC unit must be built";
    }
    const uint32_t tcm[2] = { o->itcm_bytes, o->dtcm_bytes };
    for (int i = 0; i < 2; ++i)
        if (tcm[i] % 1024u != 0 || tcm[i] > (1u << 20)) return "itcm_bytes / dtcm_bytes: a multiple of 1 KiB up to 1 MiB";
    return NULL;
}

// The Ibex configuration names of the RISC-V cycle model (tsr_rv.h; parsed here too, so that a
// simulator library without tsr_rv.c validates the option the same way). 0, or -1 if unknown.
// SW-76: "u200" | "k4" | "F,E,B,P". out[0..3] = front, EXP, back lanes and lane-pipe stages.
static int vu_timing_parse(const char * s, uint32_t * out) {
    uint32_t v[4];
    if (strcmp(s, "u200") == 0) {
        v[0] = 16; v[1] = 16; v[2] = 8; v[3] = 0;
    } else if (strcmp(s, "k4") == 0) {
        v[0] = 8; v[1] = 8; v[2] = 4; v[3] = 2;
    } else {
        char * end = NULL;
        for (int i = 0; i < 4; ++i) {
            const unsigned long x = strtoul(s, &end, 10);
            if (end == s || (i < 3 && *end != ',') || (i == 3 && *end != '\0')) return -1;
            v[i] = (uint32_t)x;
            s = end + 1;
        }
    }
    if ((v[0] != 8 && v[0] != 16) || (v[1] != 8 && v[1] != 16) || (v[2] != 4 && v[2] != 8) || v[3] > 3) return -1;
    if (out) memcpy(out, v, sizeof v);
    return 0;
}

static int rv_core_parse(const char * name, uint32_t * core) {
    uint32_t c;
    if (strcmp(name, "small") == 0) c = TSR_RV_CORE_SMALL;
    else if (strcmp(name, "maxperf") == 0) c = TSR_RV_CORE_MAXPERF;
    else return -1;
    if (core) *core = c;
    return 0;
}

// SW §4.9: the environment fills the options left unset (tsr_sim.h). *log_open receives a FILE
// the simulator opened itself (TSR_SIM_LOG=<file>, appended to), else NULL. Returns NULL or a
// problem (written to buf).
static const char * env_fill(tsr_sim_opts * o, FILE ** log_open, char * buf, size_t n) {
    *log_open = NULL;
    const char * e;
    if (o->sched == NULL) o->sched = getenv("TSR_SIM_SCHED");
    if (o->rv_ratio == 0.0 && (e = getenv("TSR_SIM_RV_RATIO")) != NULL && e[0] != '\0') {
        char * end = NULL;
        const double r = strtod(e, &end);
        if (end == e || *end != '\0' || !(r >= 0.05 && r <= 4.0)) {
            snprintf(buf, n, "TSR_SIM_RV_RATIO=\"%s\": expected a number in [0.05, 4]", e);
            return buf;
        }
        o->rv_ratio = r;
    }
    if (o->race == 0 && (e = getenv("TSR_SIM_RACE")) != NULL && e[0] != '\0') {
        uint32_t r;
        if (strcmp(e, "off") == 0) r = 0;
        else if (strcmp(e, "report") == 0) r = 1;
        else if (strcmp(e, "fatal") == 0) r = 2;
        else {
            snprintf(buf, n, "TSR_SIM_RACE=\"%s\": expected off, report or fatal", e);
            return buf;
        }
        // Without the race detector compiled in the variable is ignored (one note per process),
        // so that a global TSR_SIM_RACE does not break libraries built without tsr_race.c.
        static atomic_int noted;
        if (r != 0 && !HAVE_RACE_MODEL) {
            if (atomic_exchange(&noted, 1) == 0)
                fprintf(stderr, "tsr_sim: TSR_SIM_RACE=%s ignored: built without the race detector\n", e);
            r = 0;
        }
        o->race = r;
    }
    if (o->log == NULL && (e = getenv("TSR_SIM_LOG")) != NULL && e[0] != '\0') {
        for (;;) {   // prefixes: "rv," (retire lines), "w," (the eight words of every push)
            if (strncmp(e, "rv,", 3) == 0) {
                o->log_flags |= TSR_SIM_LOG_RV;
                e += 3;
            } else if (strncmp(e, "w,", 2) == 0) {
                o->log_flags |= TSR_SIM_LOG_WORDS;
                e += 2;
            } else {
                break;
            }
        }
        if (strcmp(e, "stderr") == 0) {
            o->log = stderr;
        } else {
            FILE * f = fopen(e, "a");
            if (f == NULL) {
                snprintf(buf, n, "TSR_SIM_LOG=\"%s\": cannot open: %s", e, strerror(errno));
                return buf;
            }
            *log_open = f;
            o->log = f;
        }
    }
    if (o->trace_dir == NULL && (e = getenv("TSR_TRACE")) != NULL && e[0] != '\0') o->trace_dir = e;
    if (o->rv_core == NULL && (e = getenv("TSR_SIM_RV_CORE")) != NULL && e[0] != '\0') o->rv_core = e;
    if (o->vu_timing == NULL && (e = getenv("TSR_SIM_VU_TIMING")) != NULL && e[0] != '\0') o->vu_timing = e;
    if (o->mvm_dual == 0 && (e = getenv("TSR_MVM_DUAL")) != NULL && e[0] != '\0') {
        if (strcmp(e, "1") == 0) o->mvm_dual = 1;
        else if (strcmp(e, "0") == 0) o->mvm_dual = 2;
        else {
            snprintf(buf, n, "TSR_MVM_DUAL=\"%s\": expected 0 or 1", e);
            return buf;
        }
    }
    if (o->mvm_dual > 2) {
        snprintf(buf, n, "mvm_dual=%u: expected 0 (default), 1 (enabled) or 2 (disabled)", o->mvm_dual);
        return buf;
    }
    if (o->mem_rate == 0.0 && (e = getenv("TSR_SIM_MEM_RATE")) != NULL && e[0] != '\0') {
        char * end = NULL;
        const double r = strtod(e, &end);
        if (end == e || *end != '\0' || !(r > 0.0 && r <= 64.0)) {
            snprintf(buf, n, "TSR_SIM_MEM_RATE=\"%s\": expected a number in (0, 64] (weight beats per cycle and channel)", e);
            return buf;
        }
        o->mem_rate = r;
    }
    if (o->mem_rate < 0.0 || o->mem_rate > 64.0 || o->mem_rate != o->mem_rate) {
        snprintf(buf, n, "mem_rate=%g: expected a number in [0, 64] (0: unbounded)", o->mem_rate);
        return buf;
    }
    if (o->vu_timing != NULL && vu_timing_parse(o->vu_timing, NULL) != 0) {
        snprintf(buf, n, "TSR_SIM_VU_TIMING / vu_timing=\"%s\": expected u200, k4 or F,E,B,P (front 8|16, EXP 8|16, back 4|8, pipe 0..3)",
                 o->vu_timing);
        return buf;
    }
    if (o->rv_core != NULL && rv_core_parse(o->rv_core, NULL) != 0) {
        snprintf(buf, n, "TSR_SIM_RV_CORE / rv_core=\"%s\": expected small or maxperf", o->rv_core);
        return buf;
    }
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

tsr_sim * tsr_sim_create_opts(const tsr_config * cfg, const tsr_sim_opts * opts) {
    if (cfg == NULL) {
        fprintf(stderr, "tsr_sim_create: NULL config\n");
        return NULL;
    }
    ensure_defaults();
    const char * why = cfg_problem(cfg);
    if (why != NULL) {
        fprintf(stderr, "tsr_sim_create: unsupported config: %s\n", why);
        return NULL;
    }
    tsr_sim_opts o;
    memset(&o, 0, sizeof(o));
    if (opts != NULL) o = *opts;
    why = opts_problem(&o);
    if (why != NULL) {
        fprintf(stderr, "tsr_sim_create: unsupported options: %s\n", why);
        return NULL;
    }
    char ebuf[256];
    FILE * log_open = NULL;
    why = env_fill(&o, &log_open, ebuf, sizeof(ebuf));
    tsr_sched_spec sched;
    if (why == NULL && tsr_sched_parse(o.sched, &sched, ebuf, sizeof(ebuf)) != 0) why = ebuf;
    if (why != NULL) {
        fprintf(stderr, "tsr_sim_create: unsupported options: %s\n", why);
        if (log_open) fclose(log_open);
        return NULL;
    }
    tsr_sim * sim = (tsr_sim *)xcalloc(1, sizeof(tsr_sim));
    sim->log_owned = log_open;
    sim->log_flags = o.log_flags;
    sim->race = o.race;
    if (o.trace_dir != NULL) {
        const size_t n = strlen(o.trace_dir) + 1;
        sim->trace_dir = (char *)xrealloc(NULL, n);
        memcpy(sim->trace_dir, o.trace_dir, n);
    }
    tsr_sched_init(sim, &sched);
    sim->cfg = *cfg;
    // Existing callers use all implemented capabilities for different channel
    // counts. Report the optional four-owner paths only on their geometry.
    if (cfg->num_ch != TSR_LOCAL_AT_OWNERS)
        sim->cfg.caps &= ~(TSR_CAPS_ATT_LOCAL | TSR_CAPS_MVM_LOCAL_SCALE | TSR_CAPS_MVM_LOCAL_SOFTCAP);
    sim->num_ch = cfg->num_ch;
    sim->ch_bytes = cfg->ch_bytes;
    sim->limit_beats = cfg->ch_bytes / TSR_BEAT_BYTES;
    if (sim->limit_beats > (1ull << cfg->addr_beat_w)) sim->limit_beats = 1ull << cfg->addr_beat_w;
    sim->spm_lines = cfg->spm_bytes / TSR_LINE_BYTES;
    sim->dram = (uint8_t **)xcalloc(sim->num_ch, sizeof(uint8_t *));
    sim->dram_fd = (int *)xcalloc(sim->num_ch, sizeof(int));
    for (uint32_t c = 0; c < sim->num_ch; ++c) sim->dram_fd[c] = -1;
    for (uint32_t c = 0; c < sim->num_ch; ++c) {
        sim->dram[c] = map_channel(sim->ch_bytes, o.dram_dir, c, &sim->dram_fd[c]);
        if (sim->dram[c] == NULL) {
            tsr_sim_destroy(sim);
            return NULL;
        }
    }
    sim->spm = (uint8_t *)xrealloc(NULL, cfg->spm_bytes);
    memset(sim->spm, 0xA5, cfg->spm_bytes);
    if (sim->cfg.caps & TSR_CAPS_ATT_LOCAL)
        sim->att_local = (float *)xcalloc(TSR_LOCAL_AT_SLOTS * TSR_LOCAL_AT_SLOT_LINES, TSR_LINE_BYTES);
    sim->itcm_bytes = o.itcm_bytes ? o.itcm_bytes : TSR_ITCM_BYTES;
    sim->dtcm_bytes = o.dtcm_bytes ? o.dtcm_bytes : TSR_DTCM_BYTES;
    sim->itcm = (uint8_t *)xcalloc(sim->itcm_bytes, 1);
    sim->dtcm = (uint8_t *)xcalloc(sim->dtcm_bytes, 1);
    sim->rv_ratio = o.rv_ratio != 0.0 ? o.rv_ratio : RV_RATIO_DEFAULT;
    sim->rv_core = TSR_RV_CORE_DEFAULT;
    if (o.rv_core != NULL) (void)rv_core_parse(o.rv_core, &sim->rv_core);
    {
        uint32_t v[4] = { TSR_VU_FRONT_LANES, TSR_VU_FRONT_LANES, TSR_VU_BACK_LANES, 0 };
        if (o.vu_timing != NULL) (void)vu_timing_parse(o.vu_timing, v);
        sim->vu_front = v[0]; sim->vu_exp = v[1]; sim->vu_back = v[2]; sim->vu_pipe = v[3];
    }
    sim->mvm_dual_en = o.mvm_dual != 2;
    sim->mem_rate = o.mem_rate;
    sim->xbuf = (float *)xcalloc(cfg->k_max, sizeof(float));
    sim->threads = 1;
    sim->log = (FILE *)o.log;

    // Hardware-reset values (EX §5.1).
    sim->wdog_cfg = TSR_CSR_WDOG_CFG_RESET;
    sim->log_cfg = TSR_CSR_LOG_CFG_RESET;
    sim->load_crc_state = 0xFFFFFFFFu;
    tsr_q_init(sim, o.q_depth);
    if (o.threads > 1) tsr_sim_set_threads(sim, o.threads);
#if defined(TSR_SIM_HAVE_RV) && TSR_SIM_HAVE_RV
    // The control core (SW §4.1): the hart and its firmware-profile bus, registered as the time
    // base's agent. It stays in reset until the host sets CTRL.RV_RUN (EX §3.4).
    sim->rv_agent = tsr_rv_agent_create(sim);
    if (sim->rv_agent == NULL) {
        fprintf(stderr, "tsr_sim_create: RISC-V agent creation failed\n");
        tsr_sim_destroy(sim);
        return NULL;
    }
#endif
    // Optional models (tsr_sim_int.h §5), in this order; tsr_sim_destroy detaches in reverse.
#if defined(TSR_SIM_HAVE_DM) && TSR_SIM_HAVE_DM
    tsr_dm_attach(sim);
#endif
#if defined(TSR_SIM_HAVE_DBG) && TSR_SIM_HAVE_DBG
    tsr_dbg_attach(sim);
#endif
#if defined(TSR_SIM_HAVE_RACE) && TSR_SIM_HAVE_RACE
    tsr_race_attach(sim);
#endif
#if defined(TSR_SIM_HAVE_TRACE) && TSR_SIM_HAVE_TRACE
    tsr_trace_attach(sim);
#endif
    return sim;
}

tsr_sim * tsr_sim_create(const tsr_config * cfg, const char * dram_dir) {
    tsr_sim_opts o;
    memset(&o, 0, sizeof(o));
    o.dram_dir = dram_dir;
    return tsr_sim_create_opts(cfg, &o);
}

void tsr_sim_destroy(tsr_sim * sim) {
    if (sim == NULL) return;
    // The optional models first, in the reverse attach order (each frees its slot). A model whose
    // attach did not run (create failed earlier) sees a NULL slot.
#if defined(TSR_SIM_HAVE_TRACE) && TSR_SIM_HAVE_TRACE
    tsr_trace_detach(sim);
#endif
#if defined(TSR_SIM_HAVE_RACE) && TSR_SIM_HAVE_RACE
    tsr_race_detach(sim);
#endif
#if defined(TSR_SIM_HAVE_DBG) && TSR_SIM_HAVE_DBG
    tsr_dbg_detach(sim);
#endif
#if defined(TSR_SIM_HAVE_DM) && TSR_SIM_HAVE_DM
    tsr_dm_detach(sim);
#endif
#if defined(TSR_SIM_HAVE_RV) && TSR_SIM_HAVE_RV
    tsr_rv_agent_destroy(sim->rv_agent);
    sim->rv_agent = NULL;
#endif
    pool_destroy(sim->pool);
    cap_state_free(sim);
    if (sim->dram != NULL) {
        for (uint32_t c = 0; c < sim->num_ch; ++c) {
            if (sim->dram[c] != NULL) munmap(sim->dram[c], (size_t)sim->ch_bytes);
            if (sim->dram_fd[c] >= 0) close(sim->dram_fd[c]);
        }
    }
    tsr_q_free(sim);
    free(sim->dram);
    free(sim->dram_fd);
    free(sim->spm);
    free(sim->att_local);
    free(sim->itcm);
    free(sim->dtcm);
    free(sim->xbuf);
    free(sim->ybuf);
    free(sim->gaps);
    free(sim->scratch);
    free(sim->trace_dir);
    if (sim->log_owned) fclose(sim->log_owned);
    free(sim);
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------
const tsr_config * tsr_sim_config(const tsr_sim * sim) { return &sim->cfg; }

uint8_t * tsr_sim_dram(tsr_sim * sim, uint32_t ch) { return ch < sim->num_ch ? sim->dram[ch] : NULL; }
uint8_t * tsr_sim_spm(tsr_sim * sim) { return sim->spm; }
float * tsr_sim_att_local(tsr_sim * sim) { return sim->att_local; }
uint64_t  tsr_sim_dram_limit_beats(const tsr_sim * sim) { return sim->limit_beats; }
uint32_t  tsr_sim_spm_lines(const tsr_sim * sim) { return sim->spm_lines; }
uint32_t  tsr_sim_num_ch(const tsr_sim * sim) { return sim->num_ch; }
float *   tsr_sim_sreg(tsr_sim * sim) { return sim->sreg; }

// The built-in RISC-V agent and the RISC-V debug view (SW §4.8).
struct tsr_rv_agent * tsr_sim_rv_agent(tsr_sim * sim) { return sim->rv_agent; }
#if defined(TSR_SIM_HAVE_RV) && TSR_SIM_HAVE_RV
uint32_t tsr_sim_rv_reg(const tsr_sim * sim, unsigned i) {
    return sim->rv_agent != NULL && i < 32u ? tsr_rv_reg(tsr_rv_agent_rv(sim->rv_agent), i) : 0u;
}
uint32_t tsr_sim_rv_pc(const tsr_sim * sim) {
    return sim->rv_agent != NULL ? tsr_rv_pc(tsr_rv_agent_rv(sim->rv_agent)) : 0u;
}
uint64_t tsr_sim_rv_instret(const tsr_sim * sim) {
    return sim->rv_agent != NULL ? tsr_rv_retired(tsr_rv_agent_rv(sim->rv_agent)) : 0u;
}
#else
uint32_t tsr_sim_rv_reg(const tsr_sim * sim, unsigned i) { (void)sim; (void)i; return 0u; }
uint32_t tsr_sim_rv_pc(const tsr_sim * sim) { (void)sim; return 0u; }
uint64_t tsr_sim_rv_instret(const tsr_sim * sim) { (void)sim; return 0u; }
#endif

void tsr_sim_set_log(tsr_sim * sim, void * file) { sim->log = (FILE *)file; }
int  tsr_sim_log_on(const tsr_sim * sim) { return sim->log != NULL; }
int  tsr_sim_log_rv(const tsr_sim * sim) { return sim->log != NULL && (sim->log_flags & TSR_SIM_LOG_RV) != 0; }
int  tsr_sim_log_words(const tsr_sim * sim) { return sim->log != NULL && (sim->log_flags & TSR_SIM_LOG_WORDS) != 0; }

void ** tsr_sim_model_slot(tsr_sim * sim, int id) {
    return (id >= 0 && id < TSR_MODEL_COUNT) ? &sim->models[id] : NULL;
}
uint32_t     tsr_sim_opt_race(const tsr_sim * sim) { return sim->race; }
const char * tsr_sim_opt_trace_dir(const tsr_sim * sim) { return sim->trace_dir; }

// Host DRAM through the API (visible to the access listeners, tsr_sim_int.h §5).
static int dram_range_ok(const tsr_sim * sim, uint32_t ch, uint64_t off, uint64_t len) {
    return ch < sim->num_ch && off <= sim->ch_bytes && len <= sim->ch_bytes - off;
}
static void dram_acc(tsr_sim * sim, uint32_t kind, uint32_t ch, uint64_t off, uint64_t len) {
    if (sim->n_acc == 0) return;
    tsr_sim_acc a;
    memset(&a, 0, sizeof(a));
    a.kind = kind;
    a.ch = ch;
    a.addr = off;
    a.len = len;
    tsr_sim_acc_call(sim, &a);
}
int tsr_sim_dram_write(tsr_sim * sim, uint32_t ch, uint64_t off, const void * src, uint64_t len) {
    if (!dram_range_ok(sim, ch, off, len)) return -1;
    if (len) memcpy(sim->dram[ch] + off, src, (size_t)len);
    dram_acc(sim, TSR_ACC_HOST_DRAM_WR, ch, off, len);
    return 0;
}
int tsr_sim_dram_read(tsr_sim * sim, uint32_t ch, uint64_t off, void * dst, uint64_t len) {
    if (!dram_range_ok(sim, ch, off, len)) return -1;
    dram_acc(sim, TSR_ACC_HOST_DRAM_RD, ch, off, len);
    if (len) memcpy(dst, sim->dram[ch] + off, (size_t)len);
    return 0;
}
void tsr_sim_dram_note_write(tsr_sim * sim, uint32_t ch, uint64_t off, uint64_t len) {
    if (dram_range_ok(sim, ch, off, len)) dram_acc(sim, TSR_ACC_HOST_DRAM_WR, ch, off, len);
}

void tsr_sim_log_line(tsr_sim * sim, const char * fmt, ...) {
    if (sim->log == NULL) return;
    va_list ap;
    va_start(ap, fmt);
    vfprintf(sim->log, fmt, ap);
    va_end(ap);
    fputc('\n', sim->log);
}

void tsr_sim_set_threads(tsr_sim * sim, unsigned n) {
    if (n < 1) n = 1;
    if (n > 64) n = 64;
    if (n == sim->threads) return;
    pool_destroy(sim->pool);
    sim->pool = NULL;
    sim->threads = n;
    if (n > 1) sim->pool = pool_create(n - 1);
}

uint64_t tsr_sim_perf(const tsr_sim * sim, uint32_t idx) { return idx < TSR_PERF_SLOTS ? sim->perf[idx] : 0; }

int tsr_sim_active(const tsr_sim * sim) { return sim->taken != sim->answered; }

void tsr_sim_perf_add(tsr_sim * sim, uint32_t idx, uint64_t n) {
    if (idx >= TSR_PERF_SLOTS) return;
    if (sim->perf_force || tsr_sim_active(sim)) sim->perf[idx] += n;
}

void * tsr_sim_scratch(tsr_sim * sim, size_t n) {
    if (n > sim->scratch_cap) {
        free(sim->scratch);
        const size_t cap = (n + 63u) & ~(size_t)63u;
        sim->scratch = aligned_alloc(64, cap);
        if (sim->scratch == NULL) {
            fprintf(stderr, "tsr_sim: out of memory (%zu bytes of scratch)\n", cap);
            abort();
        }
        sim->scratch_cap = cap;
    }
    return sim->scratch;
}

void tsr_sim_parallel_rows(tsr_sim * sim, void (*fn)(void * ctx, uint32_t lo, uint32_t hi),
                           void * ctx, uint32_t total, uint32_t chunk) {
    if (total == 0) return;
    if (sim->pool != NULL && total > 1) pool_run(sim->pool, fn, ctx, total, chunk);
    else fn(ctx, 0, total);
}

uint32_t  tsr_sim_rv_run(const tsr_sim * sim) { return (sim->ctrl & TSR_CTRL_RV_RUN) ? 1u : 0u; }
uint32_t  tsr_sim_rv_boot(const tsr_sim * sim) { return sim->rv_boot; }
uint64_t  tsr_sim_rv_boot_gen(const tsr_sim * sim) { return sim->rv_boot_gen; }
double    tsr_sim_rv_ratio(const tsr_sim * sim) { return sim->rv_ratio; }
tsr_vu_timing tsr_sim_vu_timing(const tsr_sim * sim) {
    tsr_vu_timing t = { sim->vu_front, sim->vu_exp, sim->vu_back, sim->vu_pipe };
    return t;
}
uint32_t  tsr_sim_rv_core(const tsr_sim * sim) { return sim->rv_core; }
void      tsr_sim_rv_set_state(tsr_sim * sim, uint32_t bits) {
    const uint32_t old = sim->rv_state;
    sim->rv_state = bits & (TSR_STATUS_RV_HALTED | TSR_STATUS_RV_SLEEP);
    tsr_q_view_note(sim);                                 // random: the debug-halt hold source
    if (sim->rv_state != old && sim->have_dbg && sim->dbg.rv_state) sim->dbg.rv_state(sim->dbg.ctx);
}
uint8_t * tsr_sim_itcm(tsr_sim * sim, uint32_t * bytes) { if (bytes) *bytes = sim->itcm_bytes; return sim->itcm; }
uint8_t * tsr_sim_dtcm(tsr_sim * sim, uint32_t * bytes) { if (bytes) *bytes = sim->dtcm_bytes; return sim->dtcm; }

// ---------------------------------------------------------------------------
// Firmware loader (EX §3.3): LOAD_* CSRs, receive CRC, VERIFY
// ---------------------------------------------------------------------------
// The TCM word at RISC-V byte address a (word aligned), or NULL if no TCM is there (writes are
// lost, reads return 0: VERIFY then shows the address-decode fault the receive CRC cannot see).
static uint8_t * tcm_word(tsr_sim * sim, uint32_t a) {
    if (a - TSR_RV_ITCM_BASE < sim->itcm_bytes) return sim->itcm + (a - TSR_RV_ITCM_BASE);   // ITCM base 0
    if (a >= TSR_RV_DTCM_BASE && a - TSR_RV_DTCM_BASE < sim->dtcm_bytes) return sim->dtcm + (a - TSR_RV_DTCM_BASE);
    return NULL;
}

// zlib crc32 (reflected 0xEDB88320, init / final XOR 0xFFFFFFFF) over the 4 bytes of a word in
// address order; st is the running pre-final state.
static uint32_t crc32_word(uint32_t st, uint32_t w) {
    for (int b = 0; b < 4; ++b) {
        st ^= (w >> (8 * b)) & 0xFFu;
        for (int k = 0; k < 8; ++k) st = (st >> 1) ^ (TSR_CRC32_POLY & (0u - (st & 1u)));
    }
    return st;
}

static int load_refused(const tsr_sim * sim) { return (sim->ctrl & TSR_CTRL_RV_RUN) || sim->sr_busy; }

static void load_verify(tsr_sim * sim) {
    if (load_refused(sim)) {
        sim->sticky |= TSR_STATUS_LOAD_REFUSED;
        return;
    }
    // The sequencer reads LOAD_COUNT words from LOAD_ADDR (completes at once in this model;
    // LOAD_ADDR itself is not moved).
    uint32_t st = 0xFFFFFFFFu;
    for (uint32_t i = 0; i < sim->load_count; ++i) {
        const uint8_t * p = tcm_word(sim, sim->load_addr + 4u * i);
        st = crc32_word(st, p ? rd32le(p) : 0u);
    }
    sim->load_crc_state = st;
}

// ---------------------------------------------------------------------------
// Resets (EX §8.5, CC §12.2)
// ---------------------------------------------------------------------------
// RISC-V reset sequence (RV_RUN 1 -> 0, SOFT_RESET step 1): the core applies QABORT and sets
// ANSWERED := TAKEN (the running launch gets no record); the clk_rv side resets staging, deps,
// WAIT_*, the RISC-V hold source, LCQ_W*, RT_*. Completes at once (RV_RESET_DONE = !RV_RUN).
static void rv_reset_seq(tsr_sim * sim) {
    sim->ctrl &= ~TSR_CTRL_RV_RUN;
    tsr_q_abort(sim);
    sim->answered = sim->taken;
    tsr_q_rv_reset(sim);
    memset(sim->lcq_w, 0, sizeof(sim->lcq_w));
    sim->rt_state = 0;
    sim->rt_info = 0;
    sim->rv_state = 0;
}

void tsr_sim_rv_ndmreset(tsr_sim * sim, int level) {
    level = level != 0;
    if (level == sim->ndmreset) return;
    if (level) {
        const uint32_t run = sim->ctrl & TSR_CTRL_RV_RUN;
        rv_reset_seq(sim);                                // QABORT, ANSWERED := TAKEN, clk_rv resets
        sim->ctrl |= run;                                 // CTRL.RV_RUN keeps the host's value
        sim->ndmreset = 1;
    } else {
        sim->ndmreset = 0;
        if (sim->ctrl & TSR_CTRL_RV_RUN) {                // the hart restarts at RV_BOOT + 0x80
            sim->rv_boot_gen++;
            sim->rv_state = 0;
            sim->agent_next = sim->now;
            tsr_q_view_flush(sim);
        }
    }
    sim->agent_kick = 1;
    tsr_q_view_note(sim);
}

// the DM model's kick (tsr_sim_int.h §5b).
void tsr_sim_rv_kick(tsr_sim * sim) { sim->agent_kick = 1; }

static void soft_reset_start(tsr_sim * sim) {
    if (sim->sr_busy) return;
    sim->sr_busy = 1;
    rv_reset_seq(sim);                                    // step 1 (QABORT even if RV_RUN was 0)
    if (sim->have_dbg && sim->dbg.soft_reset) sim->dbg.soft_reset(sim->dbg.ctx);
    tsr_sim_sr_poll(sim);                                 // steps 2-4 at once if already drained
}

void tsr_sim_sr_poll(tsr_sim * sim) {
    if (!sim->sr_busy || !tsr_q_idle(sim)) return;       // step 2: QIDLE, no unit holds an op
    // step 3
    sim->taken = sim->pushed;
    sim->answered = sim->taken;
    sim->lcq_rd = sim->lcq_wr;
    sim->log_rd = sim->log_wr;
    sim->cancel_valid = 0;
    sim->stop_valid = 0;
    tsr_q_clear_all(sim, NULL);                           // host QCLEAR effects
    memset(sim->perf, 0, sizeof(sim->perf));
    sim->snap_valid = 0;
    sim->sticky = 0;
    sim->log_dropped = 0;
    sim->sr_busy = 0;                                     // step 4
}

// ---------------------------------------------------------------------------
// Host CSR port (EX §5)
// ---------------------------------------------------------------------------
static int covered(uint32_t wm, uint32_t q) { return (uint32_t)(wm - q) < TSR_SEQ_HALF; }

static uint32_t status_read(const tsr_sim * sim) {
    uint32_t st = sim->sticky | TSR_STATUS_RV_CLK_OK | tsr_q_status(sim);
    if ((sim->ctrl & TSR_CTRL_RV_RUN) && !sim->ndmreset) st |= TSR_STATUS_RV_RUNNING | sim->rv_state;
    else st |= TSR_STATUS_RV_RESET_DONE;
    if (tsr_sim_active(sim)) st |= TSR_STATUS_ACTIVE;
    if (sim->sr_busy) st |= TSR_STATUS_SR_BUSY;
    if (sim->lcq_wr != sim->lcq_rd) st |= TSR_STATUS_LCQ_PENDING;
    if (sim->log_wr != sim->log_rd) st |= TSR_STATUS_LOG_PENDING;
    if (sim->plat_err) st |= TSR_STATUS_PLAT_ERR;          // plat_err_i
    return st;
}

// plat_err_i (EX §9.3), see tsr_sim.h.
void tsr_sim_set_plat_err(tsr_sim * sim, int level) { sim->plat_err = level != 0; }

static uint32_t cfg3_read(const tsr_sim * sim) {
    uint32_t v = 0;
    if (sim->cfg.caps & TSR_CAPS_VU)
        v |= TSR_PUT(TSR_CFG3_VU_FRONT, sim->vu_exp) | TSR_PUT(TSR_CFG3_VU_BACK, sim->vu_back);   // the RTL reports EXP_LANES
    if (sim->cfg.caps & TSR_CAPS_ATT) v |= TSR_PUT(TSR_CFG3_PCAP16, tsr_qa_lines(TSR_QA_A256, sim->cfg.k_max));
    return v;
}

// host = 1: the host port (PERF LO snapshot, LOAD_DATA, DMI); 0: the core window (CC §5.3).
static uint32_t csr_read(tsr_sim * sim, uint32_t off, int host) {
    const tsr_config * c = &sim->cfg;
    if (off % 4 != 0 || off >= 0x1000u) return 0;
    switch (off) {
    case TSR_CSR_ID:           return TSR_ID_MAGIC;
    case TSR_CSR_VERSION:      return TSR_VERSION_VALUE;
    case TSR_CSR_CFG0:         return (c->num_ch & 0xFFu) | ((c->lanes & 0xFFFu) << 8) | ((c->t_max & 0xFu) << 20);
    case TSR_CSR_CFG1:         return c->spm_bytes / TSR_LINE_BYTES;
    case TSR_CSR_CFG2:         return (c->vl & 0xFFu) | ((c->nsreg & 0xFFu) << 8) | (((c->k_max / 256u) & 0xFFFFu) << 16);
    case TSR_CSR_CAPS:         return c->caps;
    case TSR_CSR_CFG3:         return cfg3_read(sim);
    case TSR_CSR_CTL_ID:       // DM only with the DM model and DBG's DMI window (tsr_sim_int.h §3)
                               return TSR_SIM_CTL_ID_BASE |
                                      ((sim->models[TSR_MODEL_DM] != NULL && sim->have_dbg && sim->dbg.csr_read) ? TSR_CTL_ID_DM : 0u);
    case TSR_CSR_CTRL:         return sim->ctrl & (TSR_CTRL_RV_RUN | TSR_CTRL_HOST_HOLD | TSR_CTRL_HOLD_ON_HALT | TSR_CTRL_IRQ_EN);
    case TSR_CSR_STATUS:       return status_read(sim);
    case TSR_CSR_RV_BOOT:      return sim->rv_boot;
    case TSR_CSR_RV_FREQ:      return (uint32_t)(sim->rv_ratio * 65536.0 + 0.5);
    case TSR_CSR_RV_CFG:       return TSR_PUT(TSR_RV_CFG_ITCM_KIB, sim->itcm_bytes / 1024u) |
                                      TSR_PUT(TSR_RV_CFG_DTCM_KIB, sim->dtcm_bytes / 1024u);
    case TSR_CSR_WDOG_CFG:     return sim->wdog_cfg;
    case TSR_CSR_RT_STATE:     return sim->rt_state;
    case TSR_CSR_RT_INFO:      return sim->rt_info;
    case TSR_CSR_LQ_SEQ:       return sim->pushed;
    case TSR_CSR_LQ_TAKEN:     return sim->taken;
    case TSR_CSR_LQ_CANCEL:    return sim->cancel_wm;
    case TSR_CSR_LQ_STOP:      return sim->stop_wm;
    case TSR_CSR_LCQ_WR:       return sim->lcq_wr;
    case TSR_CSR_LCQ_RD:       return sim->lcq_rd;
    case TSR_CSR_LQ_CFG:       return TSR_PUT(TSR_LQ_CFG_LQ_DEPTH, TSR_LQ_DEPTH) | TSR_PUT(TSR_LQ_CFG_LCQ_DEPTH, TSR_LCQ_DEPTH) |
                                      TSR_PUT(TSR_LQ_CFG_LOG_DEPTH, TSR_LOG_DEPTH) |
                                      (sim->stop_valid ? TSR_LQ_CFG_STOP_VALID : 0u) |
                                      (sim->cancel_valid ? TSR_LQ_CFG_CANCEL_VALID : 0u);
    case TSR_CSR_PERF_INFO:    return TSR_PUT(TSR_PERF_INFO_COUNT, TSR_PERF_SLOTS) | TSR_PUT(TSR_PERF_INFO_X, TSR_PERF_X(sim->num_ch));
    case TSR_CSR_LCQ_ANSWERED: return sim->answered;
    case TSR_CSR_Q_DEPTH:      return tsr_q_depth_reg(sim);
    case TSR_CSR_LOG_WR:       return sim->log_wr;
    case TSR_CSR_LOG_RD:       return sim->log_rd;
    case TSR_CSR_LOG_CFG:      return sim->log_cfg;
    case TSR_CSR_LOG_DROPPED:  return sim->log_dropped;
    case TSR_CSR_LOAD_ADDR:    return sim->load_addr;
    case TSR_CSR_LOAD_DATA: {
        if (!host || load_refused(sim)) return 0;
        const uint8_t * p = tcm_word(sim, sim->load_addr);
        return p ? rd32le(p) : 0u;
    }
    case TSR_CSR_LOAD_CRC:     return sim->load_crc_state ^ 0xFFFFFFFFu;
    case TSR_CSR_LOAD_COUNT:   return sim->load_count;
    case TSR_CSR_LOAD_STATUS:  return 0;                  // VERIFY completes at once
    default: break;
    }
    if (off >= TSR_CSR_LQ_W(0) && off < TSR_CSR_LQ_W(8)) return sim->lq_w[(off - TSR_CSR_LQ_W(0)) / 4];
    if (off >= TSR_CSR_PERF_LO_BASE && off < TSR_CSR_PERF_LO_BASE + 8u * TSR_PERF_SLOTS) {
        const uint32_t idx = (off - TSR_CSR_PERF_LO_BASE) / 8;
        const uint64_t v = sim->perf[idx];
        if ((off - TSR_CSR_PERF_LO_BASE) % 8 == 0) {
            if (host) {                                   // LO: snapshot for the next HI read
                sim->snap_valid = 1;
                sim->snap_idx = idx;
                sim->snap_val = v;
            }
            return (uint32_t)v;
        }
        if (host && sim->snap_valid && sim->snap_idx == idx) return (uint32_t)(sim->snap_val >> 32);
        return (uint32_t)(v >> 32);
    }
    if (off >= TSR_CSR_LCQ_REC_BASE && off < TSR_CSR_LCQ_REC_BASE + TSR_LCQ_REC_BYTES * TSR_LCQ_DEPTH) {
        const uint32_t k = (off - TSR_CSR_LCQ_REC_BASE) / TSR_LCQ_REC_BYTES;
        return sim->lcq_ring[k][(off - TSR_CSR_LCQ_REC_BASE) % TSR_LCQ_REC_BYTES / 4];
    }
    if (off >= TSR_CSR_LOG_RING_BASE && off < TSR_CSR_LOG_RING_BASE + 4u * TSR_LOG_DEPTH)
        return sim->log_ring[(off - TSR_CSR_LOG_RING_BASE) / 4];
    if (off >= TSR_CSR_SREG(0) && off < TSR_CSR_SREG(16)) {
        uint32_t u;
        memcpy(&u, &sim->sreg[(off - TSR_CSR_SREG(0)) / 4], 4);
        return u;
    }
    const int dbg_range = off >= TSR_CSR_DBG_BASE && off < TSR_CSR_DBG_BASE + TSR_CSR_DBG_BYTES;
    const int dmi_range = off >= TSR_CSR_DMI_BASE && off < TSR_CSR_DMI(TSR_CSR_DMI_N);
    if (dbg_range || dmi_range) {
        if (dmi_range && !host) return 0;                 // core window: DMI reads 0 (CC §5.3)
        uint32_t v = 0;
        if (sim->have_dbg && sim->dbg.csr_read && sim->dbg.csr_read(sim->dbg.ctx, off, &v, host)) return v;
        return 0;                                         // DBG_ID = 0: block absent
    }
    if (off >= TSR_CSR_Q_DONE(0) && off < TSR_CSR_Q_ERR_PIDX(TSR_UNIT_SLOTS)) return tsr_q_csr_read(sim, off);
    return 0;
}

static void csr_acc(tsr_sim * sim, uint32_t kind, uint32_t off, uint32_t val) {
    tsr_sim_acc a;
    memset(&a, 0, sizeof(a));
    a.kind = kind;
    a.size = 4;
    a.off = off;
    a.val = val;
    tsr_sim_acc_call(sim, &a);
}

uint32_t tsr_sim_csr_read(tsr_sim * sim, uint32_t off) {
    const uint32_t v = csr_read(sim, off, 1);
    if (sim->n_acc) csr_acc(sim, TSR_ACC_HOST_CSR_RD, off, v);
    return v;
}
uint32_t tsr_sim_corewin_read(tsr_sim * sim, uint32_t csr_off) { return csr_read(sim, csr_off, 0); }

static void ctrl_write(tsr_sim * sim, uint32_t val) {
    // Level bits first (EX §5.2), then the pulses in bit order.
    const uint32_t old = sim->ctrl;
    // "The RISC-V is quiet" for a QCLEAR in this write (EX §5.2 bit 9) is judged before the write's own
    // RV_RUN change: a reset started by this write has not completed, so RV_RUN 1 -> 0 plus QCLEAR is
    // refused unless the hart was halted (the core samples its inputs in the cycle of the write).
    const int rv_quiet = !(old & TSR_CTRL_RV_RUN) || (sim->rv_state & TSR_STATUS_RV_HALTED);
    uint32_t lv = val & (TSR_CTRL_HOST_HOLD | TSR_CTRL_HOLD_ON_HALT | TSR_CTRL_IRQ_EN);
    uint32_t run = old & TSR_CTRL_RV_RUN;
    if ((val & TSR_CTRL_RV_RUN) != run) {
        if (sim->sr_busy) sim->sticky |= TSR_STATUS_RV_RUN_REFUSED;   // a change while SR_BUSY
        else run = val & TSR_CTRL_RV_RUN;
    }
    sim->ctrl = lv | run;
    if (!(old & TSR_CTRL_RV_RUN) && run) {                // boot: the hart leaves reset at RV_BOOT + 0x80
        sim->rv_boot_gen++;
        sim->rv_state = 0;
        sim->agent_next = sim->now;
        tsr_q_view_flush(sim);                            // the clk_rv side reloads (CC §12.2)
    } else if ((old & TSR_CTRL_RV_RUN) && !run) {
        sim->ctrl |= TSR_CTRL_RV_RUN;                     // rv_reset_seq clears it
        rv_reset_seq(sim);
    }
    if (val & TSR_CTRL_QABORT) tsr_q_abort(sim);
    if (val & TSR_CTRL_QCLEAR) {
        if (tsr_q_host_clear(sim, rv_quiet) != 0) sim->sticky |= TSR_STATUS_QCLEAR_REFUSED;
    }
    if (val & TSR_CTRL_SOFT_RESET) soft_reset_start(sim);
}

// Ring read pointer write rule (EX §4.2): v accepted iff (v - RD) <= (WR - RD), mod 2^32.
static void ring_rd_write(tsr_sim * sim, uint32_t * rd, uint32_t wr, uint32_t v) {
    if ((uint32_t)(v - *rd) <= (uint32_t)(wr - *rd)) *rd = v;
    else sim->sticky |= TSR_STATUS_RING_RD_BAD;
}

static void csr_write(tsr_sim * sim, uint32_t off, uint32_t val);

// Every host CSR write: the core-state change is noted for the RISC-V-side view (random
// schedule), then the write is reported to the access listeners.
void tsr_sim_csr_write(tsr_sim * sim, uint32_t off, uint32_t val) {
    csr_write(sim, off, val);
    tsr_q_view_note(sim);
    if (sim->n_acc) csr_acc(sim, TSR_ACC_HOST_CSR_WR, off, val);
}

static void csr_write(tsr_sim * sim, uint32_t off, uint32_t val) {
    if (off % 4 != 0 || off >= 0x1000u) return;
    sim->agent_kick = 1;
    switch (off) {
    case TSR_CSR_CTRL:      ctrl_write(sim, val); return;
    case TSR_CSR_STATUS:    sim->sticky &= ~(val & 0x0FFF0000u); return;   // W1C, sticky bits 16..27
    case TSR_CSR_RV_BOOT:                                  // taken when RV_RUN rises (EX §5.1): ignored while RV_RUN = 1
        if (!(sim->ctrl & TSR_CTRL_RV_RUN)) sim->rv_boot = val & TSR_RV_BOOT_ADDR;
        return;
    case TSR_CSR_WDOG_CFG:  sim->wdog_cfg = val & TSR_WDOG_CFG_L; return;
    case TSR_CSR_LQ_PUSH:
        if ((uint32_t)(sim->pushed - sim->taken) >= TSR_LQ_DEPTH) {
            sim->sticky |= TSR_STATUS_LQ_OVERFLOW;        // dropped
        } else {
            memcpy(sim->lq_ring[sim->pushed % TSR_LQ_DEPTH], sim->lq_w, sizeof(sim->lq_w));
            sim->pushed++;
        }
        return;
    case TSR_CSR_LQ_CANCEL:
        sim->cancel_wm = val;
        sim->cancel_valid = 1;
        // Hard cancel (EX §4.9): the running launch (sequence ANSWERED) covered -> host QABORT now.
        if (tsr_sim_active(sim) && covered(val, sim->answered)) tsr_q_abort(sim);
        return;
    case TSR_CSR_LQ_STOP:
        sim->stop_wm = val;
        sim->stop_valid = 1;
        return;
    case TSR_CSR_LCQ_RD:    ring_rd_write(sim, &sim->lcq_rd, sim->lcq_wr, val); return;
    case TSR_CSR_LOG_RD:    ring_rd_write(sim, &sim->log_rd, sim->log_wr, val); return;
    case TSR_CSR_LOG_CFG:   sim->log_cfg = val & (TSR_LOG_CFG_ENABLE | TSR_LOG_CFG_BLOCK | TSR_LOG_CFG_LEVEL); return;
    case TSR_CSR_LOAD_ADDR:
        sim->load_addr = val & ~3u;                       // word aligned; restarts LOAD_CRC
        sim->load_crc_state = 0xFFFFFFFFu;
        return;
    case TSR_CSR_LOAD_DATA:
        if (load_refused(sim)) {
            sim->sticky |= TSR_STATUS_LOAD_REFUSED;
        } else {
            uint8_t * p = tcm_word(sim, sim->load_addr);
            if (p != NULL) wr32le(p, val);
            sim->load_crc_state = crc32_word(sim->load_crc_state, val);
            sim->load_addr += 4;
        }
        return;
    case TSR_CSR_LOAD_COUNT: sim->load_count = val; return;
    case TSR_CSR_LOAD_CTRL:
        if (val & TSR_LOAD_CTRL_VERIFY) load_verify(sim);
        return;
    default: break;
    }
    if (off >= TSR_CSR_LQ_W(0) && off < TSR_CSR_LQ_W(8)) {
        sim->lq_w[(off - TSR_CSR_LQ_W(0)) / 4] = val;
        return;
    }
    const int dbg_range = off >= TSR_CSR_DBG_BASE && off < TSR_CSR_DBG_BASE + TSR_CSR_DBG_BYTES &&
                          !(off >= TSR_CSR_SREG(0) && off < TSR_CSR_SREG(16));
    const int dmi_range = off >= TSR_CSR_DMI_BASE && off < TSR_CSR_DMI(TSR_CSR_DMI_N);
    if ((dbg_range || dmi_range) && sim->have_dbg && sim->dbg.csr_write) sim->dbg.csr_write(sim->dbg.ctx, off, val);
}

// ---------------------------------------------------------------------------
// RISC-V side of the rings (EX §4.5, control window 0x200-0x2FC)
// ---------------------------------------------------------------------------
int tsr_sim_lq_cw_access(tsr_sim * sim, int is_write, uint32_t off, uint32_t * val) {
    const uint32_t avail = sim->pushed - sim->taken;
    const uint32_t lcq_free = TSR_LCQ_DEPTH - (sim->lcq_wr - sim->lcq_rd);
    if (!is_write) {
        uint32_t v = 0;
        if (off >= TSR_CW_LQ_HEAD_W(0) && off < TSR_CW_LQ_HEAD_W(8)) {
            if (avail != 0) v = sim->lq_ring[sim->taken % TSR_LQ_DEPTH][(off - TSR_CW_LQ_HEAD_W(0)) / 4];
        } else if (off >= TSR_CW_LCQ_W(0) && off < TSR_CW_LCQ_W(8)) {
            v = sim->lcq_w[(off - TSR_CW_LCQ_W(0)) / 4];
        } else {
            switch (off) {
            case TSR_CW_LQ_AVAIL:    v = avail; break;
            case TSR_CW_LQ_HEAD_SEQ: v = sim->taken; break;
            case TSR_CW_LQ_CANCEL:   v = sim->cancel_wm; break;
            case TSR_CW_LQ_STOP:     v = sim->stop_wm; break;
            case TSR_CW_LQ_FLAGS:
                v = (sim->cancel_valid ? TSR_LQ_FLAGS_CANCEL_VALID : 0u) | (sim->stop_valid ? TSR_LQ_FLAGS_STOP_VALID : 0u) |
                    (tsr_sim_active(sim) ? TSR_LQ_FLAGS_ACTIVE : 0u) |
                    (sim->plat_err ? TSR_LQ_FLAGS_PLAT_ERR : 0u);   // plat_err_i
                break;
            case TSR_CW_LCQ_FREE:    v = lcq_free; break;
            case TSR_CW_LOG_FREE:    v = TSR_LOG_DEPTH - (sim->log_wr - sim->log_rd); break;
            case TSR_CW_LOG_CFG:     v = sim->log_cfg; break;
            case TSR_CW_RT_STATE:    v = sim->rt_state; break;
            case TSR_CW_RT_INFO:     v = sim->rt_info; break;
            default: break;
            }
        }
        *val = v;
        return TSR_CW_OK;
    }
    const uint32_t v = *val;
    if (off >= TSR_CW_LCQ_W(0) && off < TSR_CW_LCQ_W(8)) {
        sim->lcq_w[(off - TSR_CW_LCQ_W(0)) / 4] = v;
        return TSR_CW_OK;
    }
    switch (off) {
    case TSR_CW_LQ_POP:
        if (avail == 0 || tsr_sim_active(sim)) return TSR_CW_FAULT;
        sim->taken++;                                     // ACTIVE := 1
        return TSR_CW_OK;
    case TSR_CW_LCQ_POST: {
        const int answer = (v & TSR_LCQ_POST_ANSWER) != 0;
        if (lcq_free == 0 || (answer && !tsr_sim_active(sim))) return TSR_CW_FAULT;
        memcpy(sim->lcq_ring[sim->lcq_wr % TSR_LCQ_DEPTH], sim->lcq_w, sizeof(sim->lcq_w));
        sim->lcq_wr++;
        if (answer) {
            tsr_sim_perf_add(sim, TSR_PERF_LAUNCHES(sim->num_ch), 1);   // counted while still ACTIVE
            sim->answered++;
        }
        return TSR_CW_OK;
    }
    case TSR_CW_LOG_PUSH:
        if ((uint32_t)(sim->log_wr - sim->log_rd) < TSR_LOG_DEPTH) {
            sim->log_ring[sim->log_wr % TSR_LOG_DEPTH] = v;
            sim->log_wr++;
        } else {                                          // full: dropped and counted, never faults
            if (sim->log_dropped != 0xFFFFFFFFu) sim->log_dropped++;
            sim->sticky |= TSR_STATUS_LOG_DROP;
        }
        return TSR_CW_OK;
    case TSR_CW_RT_STATE: sim->rt_state = v; return TSR_CW_OK;
    case TSR_CW_RT_INFO:  sim->rt_info = v; return TSR_CW_OK;
    default: return TSR_CW_OK;                            // read-only / reserved: ignored
    }
}

// ---------------------------------------------------------------------------
// Tile units: NOP, FENCE (SYNC), LD, ST (DMA), QACT, MVM. Checks, extents, execution, cycles.
// ---------------------------------------------------------------------------
static inline int words_zero_from(const uint32_t * w, unsigned from) {
    for (unsigned i = from; i < TSR_INSN_WORDS; ++i) {
        if (w[i] != 0) return 0;
    }
    return 1;
}

static inline int spm_ok(const tsr_sim * sim, uint32_t a, uint64_t extent) {
    return (uint64_t)a + extent <= sim->spm_lines;
}
static inline int dram_ok(const tsr_sim * sim, uint32_t a, uint64_t extent) {
    return (uint64_t)a + extent <= sim->limit_beats;
}

// NOP / FENCE use no words: w1-w7 and every bit of w0 above the opcode must be 0 (EX §7.1).
static uint32_t chk_sync(tsr_sim * sim, const uint32_t w[8], tsr_extents * ext, uint32_t * info) {
    (void)sim; (void)ext; (void)info;
    if ((w[0] & ~0xFFu) != 0 || !words_zero_from(w, 1)) return TSR_ERR_ILLEGAL_OP;
    return TSR_ERR_NONE;
}

// LD / ST: sub = ch, w1 (A1 SPM line), w2 (A2 DRAM beat), w3 (lines).
static uint32_t chk_ldst(tsr_sim * sim, const uint32_t * w, tsr_extents * ext, uint32_t * info, int is_st) {
    if ((w[0] & 0xFFFFF000u) != 0 || !words_zero_from(w, 4)) return TSR_ERR_ILLEGAL_OP;
    const uint32_t ch = tsr_w0_sub(w[0]);
    if (ch >= sim->num_ch) return TSR_ERR_ILLEGAL_OP;
    const uint32_t spm  = w[1];
    const uint32_t dram = w[2];
    const uint32_t n    = w[3];
    if (!spm_ok(sim, spm, n))   { *info = spm;  return TSR_ERR_SPM_RANGE; }
    if (!dram_ok(sim, dram, n)) { *info = dram; return TSR_ERR_DRAM_RANGE; }
    tsr_ext_add(ext, TSR_EXT_SPM, is_st ? TSR_EXT_READ : TSR_EXT_WRITE, 0, spm, (uint64_t)spm + n);
    tsr_ext_add(ext, TSR_EXT_DRAM, is_st ? TSR_EXT_WRITE : TSR_EXT_READ, (uint8_t)ch, dram, (uint64_t)dram + n);
    return TSR_ERR_NONE;
}
static uint32_t chk_ld(tsr_sim * sim, const uint32_t w[8], tsr_extents * ext, uint32_t * info) { return chk_ldst(sim, w, ext, info, 0); }
static uint32_t chk_st(tsr_sim * sim, const uint32_t w[8], tsr_extents * ext, uint32_t * info) { return chk_ldst(sim, w, ext, info, 1); }

static void ex_ld(tsr_sim * sim, const uint32_t w[8]) {
    const uint32_t ch = tsr_w0_sub(w[0]), spm = w[1], dram = w[2], n = w[3];
    tsr_sim_cap_read(sim, ch, dram, n);
    tsr_sim_perf_add(sim, tsr_perf_ch_rd_beats(ch), n);
    memcpy(sim->spm + (size_t)spm * TSR_LINE_BYTES, sim->dram[ch] + (uint64_t)dram * TSR_BEAT_BYTES,
           (size_t)n * TSR_LINE_BYTES);
}
static void ex_st(tsr_sim * sim, const uint32_t w[8]) {
    const uint32_t ch = tsr_w0_sub(w[0]), spm = w[1], dram = w[2], n = w[3];
    memcpy(sim->dram[ch] + (uint64_t)dram * TSR_BEAT_BYTES, sim->spm + (size_t)spm * TSR_LINE_BYTES,
           (size_t)n * TSR_LINE_BYTES);
    tsr_sim_cap_write(sim, ch, dram, n);
}
static uint64_t cyc_ldst(const tsr_sim * sim, const uint32_t w[8]) {
    (void)sim;
    return CYC_DMA_LAT + (uint64_t)w[3];
}
static uint64_t cyc_st(const tsr_sim * sim, const uint32_t w[8]) {
    (void)sim;
    return (w[3] <= 8u ? CYC_ST_SMALL : CYC_ST_LAT) + (uint64_t)w[3];
}

// QACT: sub = kind, w1 (A1 QA dst), w2 (A2 f32 src), w3 K, w4 T, w5 src stride (0 = dense).
static uint32_t chk_qact(tsr_sim * sim, const uint32_t w[8], tsr_extents * ext, uint32_t * info) {
    if ((w[0] & 0xFFFFF000u) != 0 || !words_zero_from(w, 6)) return TSR_ERR_ILLEGAL_OP;
    const uint32_t kind = tsr_w0_sub(w[0]);
    if (kind != TSR_QA_A32 && kind != TSR_QA_A256) return TSR_ERR_ILLEGAL_OP;
    const uint32_t K = w[3], T = w[4];
    const uint32_t bs = tsr_qa_block_size(kind);
    if (K == 0 || K % bs != 0 || K > sim->cfg.k_max) return TSR_ERR_K_RANGE;
    if (T == 0 || T > sim->cfg.t_max) return TSR_ERR_T_RANGE;
    const uint32_t x_lines  = K * 4u / TSR_LINE_BYTES;          // exact: K % 32 == 0
    const uint64_t sstride  = w[5] ? w[5] : x_lines;             // spec §21 (Stride 0)
    const uint32_t qa_lines = tsr_qa_lines(kind, K);
    const uint32_t dst = w[1];
    const uint32_t src = w[2];
    const uint64_t e_dst = (uint64_t)T * qa_lines, e_src = (uint64_t)(T - 1) * sstride + x_lines;
    if (!spm_ok(sim, dst, e_dst)) { *info = dst; return TSR_ERR_SPM_RANGE; }
    if (!spm_ok(sim, src, e_src)) { *info = src; return TSR_ERR_SPM_RANGE; }
    tsr_ext_add(ext, TSR_EXT_SPM, TSR_EXT_WRITE, 0, dst, dst + e_dst);
    tsr_ext_add(ext, TSR_EXT_SPM, TSR_EXT_READ, 0, src, src + e_src);
    return TSR_ERR_NONE;
}
static void ex_qact(tsr_sim * sim, const uint32_t w[8]) {
    const uint32_t kind = tsr_w0_sub(w[0]);
    const uint32_t K = w[3], T = w[4];
    const uint32_t x_lines  = K * 4u / TSR_LINE_BYTES;
    const uint64_t sstride  = w[5] ? w[5] : x_lines;
    const uint32_t qa_lines = tsr_qa_lines(kind, K);
    const uint32_t dst = w[1], src = w[2];
    for (uint32_t t = 0; t < T; ++t) {
        const uint8_t * s = sim->spm + ((uint64_t)src + (uint64_t)t * sstride) * TSR_LINE_BYTES;
        for (uint32_t k = 0; k < K; ++k) {
            const uint32_t u = rd32le(s + 4u * k);
            memcpy(&sim->xbuf[k], &u, 4);
        }
        tsr_qact(kind, sim->xbuf, K, sim->spm + ((uint64_t)dst + (uint64_t)t * qa_lines) * TSR_LINE_BYTES);
    }
}
// SW-80: tsr_qact.sv is divider bound for A32 (one 32-block per 16 cycles, 8 per source line) and nearly line bound
// for A256 (one 256-block per 17 cycles, tb/qact qact_timing); accept to done adds the pipeline fill (44 and 72 cycles
// in the unit's run time as the trace writer reports it on the card). The tokens of one op run back to back.
static uint64_t cyc_qact(const tsr_sim * sim, const uint32_t w[8]) {
    (void)sim;
    const uint32_t kind = tsr_w0_sub(w[0]), K = w[3], T = w[4];
    const uint64_t blocks = (uint64_t)T * (K / tsr_qa_block_size(kind));
    return kind == TSR_QA_A32 ? 16u * blocks + 44u : 17u * blocks + 72u;
}

// MVM: sub = fmt, flags bit0 ACC, w1 (A1 dst), w2 (A2 x), w3 (A3 W_OFF), w4 N, w5 K | T << 16,
// w6 dst stride (0 = dense).
typedef struct {
    uint32_t fmt, acc, N, K, T, qa_lines, row_beats;
    uint64_t y_lines, dstride, R;
    uint32_t dst, x, woff, channels;
} mvm_dec;

static void mvm_decode(const tsr_sim * sim, const uint32_t * w, mvm_dec * d) {
    d->fmt = tsr_w0_sub(w[0]);
    d->acc = (tsr_w0_flags(w[0]) & TSR_MVM_FLAG_ACC) != 0;
    d->N = w[4];
    d->K = w[5] & 0xFFFFu;
    d->T = (w[5] >> 16) & 0xFu;
    d->qa_lines = tsr_qa_lines(tsr_qa_kind_for_fmt(d->fmt), d->K);
    d->y_lines = tsr_ceil_div_u64(4ull * d->N, TSR_LINE_BYTES);
    d->dstride = w[6] && !(w[0] & TSR_MVM_LOCAL_SOFTCAP) ? w[6] : d->y_lines;
    d->dst = w[1];
    d->x = w[2];
    d->woff = w[3];
    d->channels = sim->num_ch;
    d->R = tsr_ceil_div_u64(d->N, sim->num_ch);
    d->row_beats = tsr_row_beats(d->fmt, d->K);
}

static uint32_t chk_mvm(tsr_sim * sim, const uint32_t w[8], tsr_extents * ext, uint32_t * info) {
    const uint32_t scaled = w[0] & TSR_MVM_LOCAL_SCALE, softcap = w[0] & TSR_MVM_LOCAL_SOFTCAP;
    if ((w[0] & TSR_MVM_RSV_W0) || (w[5] & TSR_MVM_RSV_W5) || (!scaled && !softcap && w[7]) ||
        ((scaled || softcap) && (w[0] & TSR_MVM_ACC)) || (scaled && softcap) ||
        (scaled && !(sim->cfg.caps & TSR_CAPS_MVM_LOCAL_SCALE)) ||
        (softcap && !(sim->cfg.caps & TSR_CAPS_MVM_LOCAL_SOFTCAP)))
        return TSR_ERR_ILLEGAL_OP;
    const uint32_t fmt = tsr_w0_sub(w[0]);
    if (!tsr_fmt_valid(fmt)) return TSR_ERR_BAD_FMT;
    const uint32_t K = w[5] & 0xFFFFu;
    const uint32_t T = (w[5] >> 16) & 0xFu;
    if (!tsr_fmt_k_valid(fmt, K, sim->cfg.k_max)) return TSR_ERR_K_RANGE;
    if (T == 0 || T > sim->cfg.t_max) return TSR_ERR_T_RANGE;
    mvm_dec d;
    mvm_decode(sim, w, &d);
    const uint64_t e_dst = (uint64_t)(T - 1) * d.dstride + d.y_lines, e_x = (uint64_t)T * d.qa_lines;
    if (!spm_ok(sim, d.dst, e_dst)) { *info = d.dst; return TSR_ERR_SPM_RANGE; }
    if (!spm_ok(sim, d.x, e_x))     { *info = d.x;   return TSR_ERR_SPM_RANGE; }
    // N <= 16 * spm_lines < 2^31 here (the dst check bounds it), so 32-bit row math is safe.
    const uint64_t e_w = d.R * d.row_beats;
    if (!dram_ok(sim, d.woff, e_w)) { *info = d.woff; return TSR_ERR_DRAM_RANGE; }
    tsr_ext_add(ext, TSR_EXT_SPM, TSR_EXT_WRITE, 0, d.dst, d.dst + e_dst);
    tsr_ext_add(ext, TSR_EXT_SPM, TSR_EXT_READ, 0, d.x, d.x + e_x);
    tsr_ext_add(ext, TSR_EXT_DRAM, TSR_EXT_READ, TSR_EXT_CH_ALL, d.woff, d.woff + e_w);
    if (d.acc) tsr_ext_add(ext, TSR_EXT_SPM, TSR_EXT_READ, 0, d.dst, d.dst + e_dst);
    return TSR_ERR_NONE;
}

// Weight beats channel c streams: its rows [cR, min((c+1)R, N)) times ROW_BEATS.
static uint64_t mvm_ch_beats(const mvm_dec * d, uint32_t c) {
    const tsr_dense_tile tile = { .weight_base = d->woff, .channels = d->channels, .rows = d->N,
                                  .row_beats = d->row_beats, .tokens = d->T };
    tsr_dense_window window;
    return tsr_local_dense_window(&tile, c, 0u, UINT32_MAX, &window) ? window.weight_beats : 0u;
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

static void ex_mvm(tsr_sim * sim, const uint32_t w[8]) {
    mvm_dec d;
    mvm_decode(sim, w, &d);
    for (uint32_t c = 0; c < sim->num_ch; ++c) {
        const uint64_t beats = mvm_ch_beats(&d, c);
        if (beats == 0) continue;
        tsr_sim_cap_read(sim, c, d.woff, beats);
        tsr_sim_perf_add(sim, tsr_perf_ch_rd_beats(c), beats);
    }
    // Compute every y first (x is read-only here; dst/x overlap is undefined per spec).
    const size_t ny = (size_t)d.T * d.N;
    if (ny > sim->ybuf_cap) {
        sim->ybuf = (float *)xrealloc(sim->ybuf, ny * sizeof(float));
        sim->ybuf_cap = ny;
    }
    mvm_job job = { sim, d.fmt, d.N, d.K, d.T, d.woff, d.x, d.qa_lines };
    const uint64_t work = (uint64_t)d.N * d.row_beats * d.T;
    if (sim->pool != NULL && d.N > 1 && work >= MT_MIN_WORK_BEATS) {
        uint32_t chunk = d.N / (8u * sim->threads);
        if (chunk < 1) chunk = 1;
        if (chunk > 64) chunk = 64;
        pool_run(sim->pool, mvm_rows, &job, d.N, chunk);
    } else {
        mvm_rows(&job, 0, d.N);
    }
    // Scale the completed FP32 result before the final SPM write, as the local
    // VU does. ACC is a separate legacy path.
    const int scaled = (w[0] & TSR_MVM_LOCAL_SCALE) != 0u;
    const int softcap = (w[0] & TSR_MVM_LOCAL_SOFTCAP) != 0u;
    const float first_scale = tsr_bits_f32(w[6]), scale = tsr_bits_f32(w[7]);
    // Write: token t -> dst + t*stride, element n; tail bytes zero.
    for (uint32_t t = 0; t < d.T; ++t) {
        uint8_t * base = sim->spm + ((uint64_t)d.dst + (uint64_t)t * d.dstride) * TSR_LINE_BYTES;
        const float * y = sim->ybuf + (size_t)t * d.N;
        for (uint32_t n = 0; n < d.N; ++n) {
            float v = y[n];
            if (d.acc) {
                const uint32_t ou = rd32le(base + 4u * n);
                float old;
                memcpy(&old, &ou, 4);
                v = tsr_fadd(old, v);
            }
            if (softcap) v = tsr_fmul(tsr_m_tanh(tsr_fmul(v, first_scale)), scale);
            else if (scaled) v = tsr_fmul(v, scale);
            wr32le(base + 4u * n, tsr_f32_bits(v));
        }
        memset(base + 4ull * d.N, 0, (size_t)(d.y_lines * TSR_LINE_BYTES - 4ull * d.N));
    }
}

// Cycles of the weight phase: the slowest channel. A cluster consumes one weight beat per cycle per stream: rows_c ROW_BEATS
// cycles in one stream, ceil(rows_c / 2) ROW_BEATS in the two streams of the dual configuration (CAPS.MVM_DUAL, the run-time
// enable on, T <= T_MAX / 2: rows [first, first + rA) and [first + rA, first + rows_c) of the cluster are read in lockstep,
// rA = ceil(rows_c / 2); docs/rtl-mvm.md §15). A memory that delivers at most mem_rate beats per cycle and channel bounds
// it from below by rows_c ROW_BEATS / mem_rate. Results do not depend on any of this.
static uint64_t mvm_weight_cycles(const tsr_sim * sim, const mvm_dec * d) {
    const int dual = (sim->cfg.caps & TSR_CAPS_MVM_DUAL) != 0 && sim->mvm_dual_en && d->T <= sim->cfg.t_max / 2u;
    uint64_t max_cyc = 0;
    for (uint32_t c = 0; c < sim->num_ch; ++c) {
        const uint64_t b = mvm_ch_beats(d, c);
        const uint64_t rows = d->row_beats ? b / d->row_beats : 0;
        uint64_t cyc = (dual ? (rows + 1) / 2 : rows) * d->row_beats;
        if (sim->mem_rate > 0.0) {
            const uint64_t mem = (uint64_t)((double)b / sim->mem_rate + 0.999999999);
            if (mem > cyc) cyc = mem;
        }
        if (cyc > max_cyc) max_cyc = cyc;
    }
    return max_cyc;
}

static uint64_t cyc_mvm(const tsr_sim * sim, const uint32_t w[8]) {
    mvm_dec d;
    mvm_decode(sim, w, &d);
    return CYC_MVM_FILL + (uint64_t)d.T * d.qa_lines + mvm_weight_cycles(sim, &d);
}

void tsr_core_register(void) {
    const tsr_op_handler nop   = { TSR_UNIT_SYNC, chk_sync, NULL, NULL };
    const tsr_op_handler ld    = { TSR_UNIT_DMA, chk_ld, ex_ld, cyc_ldst };
    const tsr_op_handler st    = { TSR_UNIT_DMA, chk_st, ex_st, cyc_st };
    const tsr_op_handler qact  = { TSR_UNIT_QACT, chk_qact, ex_qact, cyc_qact };
    const tsr_op_handler mvm   = { TSR_UNIT_MVM, chk_mvm, ex_mvm, cyc_mvm };
    tsr_sim_register_op(TSR_OP_NOP, &nop);
    tsr_sim_register_op(TSR_OP_FENCE, &nop);
    tsr_sim_register_op(TSR_OP_LD, &ld);
    tsr_sim_register_op(TSR_OP_ST, &st);
    tsr_sim_register_op(TSR_OP_QACT, &qact);
    tsr_sim_register_op(TSR_OP_MVM, &mvm);
}

// ---------------------------------------------------------------------------
// tsr_sim_op: the op-level test hook (outside the queues)
// ---------------------------------------------------------------------------
static const char * op_name(uint32_t op) {
    switch (op) {
    case TSR_OP_NOP: return "NOP";     case TSR_OP_LD: return "LD";       case TSR_OP_ST: return "ST";
    case TSR_OP_DQROW: return "DQROW"; case TSR_OP_KVST: return "KVST";   case TSR_OP_QACT: return "QACT";
    case TSR_OP_MVM: return "MVM";     case TSR_OP_ATTQK: return "ATTQK"; case TSR_OP_ATTPV: return "ATTPV";
    case TSR_OP_VOP: return "VOP";     case TSR_OP_RED: return "RED";     case TSR_OP_ROPE: return "ROPE";
    case TSR_OP_TOPK: return "TOPK";   case TSR_OP_ASMAX: return "ASMAX"; case TSR_OP_AMASK: return "AMASK";
    case TSR_OP_AMERGE: return "AMERGE"; case TSR_OP_ANORM: return "ANORM"; case TSR_OP_VMASK: return "VMASK";
    case TSR_OP_GDN: return "GDN";
    case TSR_OP_SOP: return "SOP";     case TSR_OP_FENCE: return "FENCE";
    default: return "?";
    }
}

static void log_op(tsr_sim * sim, const uint32_t * w, uint32_t err, uint32_t info) {
    const uint32_t w0 = w[0];
    fprintf(sim->log, "%8llu %-5s sub=%u fl=%u x16=%04x w1-7=%08x %08x %08x %08x %08x %08x %08x",
            (unsigned long long)sim->n_ops, op_name(tsr_w0_opcode(w0)), tsr_w0_sub(w0), tsr_w0_flags(w0),
            tsr_w0_x16(w0), w[1], w[2], w[3], w[4], w[5], w[6], w[7]);
    if (err) fprintf(sim->log, "  -> ERR %u info=0x%08x", err, info);
    fputc('\n', sim->log);
}

uint32_t tsr_sim_op(tsr_sim * sim, const uint32_t w[8], uint32_t * err_info) {
    uint32_t info = 0;
    uint32_t err = TSR_ERR_ILLEGAL_OP;
    const tsr_op_handler * h = tsr_sim_accept_op(sim, tsr_w0_opcode(w[0]));
    if (h != NULL) err = h->check(sim, w, NULL, &info);
    if (err == TSR_ERR_NONE) {
        sim->perf_force = 1;
        if (h->exec != NULL) h->exec(sim, w);
        sim->perf_force = 0;
        const uint64_t cyc = h->cycles != NULL ? h->cycles(sim, w) : 0;
        sim->perf[TSR_PERF_CYCLES] += CYC_ISSUE + cyc;
        if (h->unit == TSR_UNIT_MVM) sim->perf[TSR_PERF_MVM_CYCLES] += cyc;
        if (h->unit == TSR_UNIT_VU) sim->perf[TSR_PERF_VU_CYCLES] += cyc;
        const uint32_t op = tsr_w0_opcode(w[0]);         // as the queue path (tsr_q.c q_integrate)
        if (op == TSR_OP_ATTQK) sim->perf[TSR_PERF_ATT_QK_CYCLES(sim->num_ch)] += cyc;
        if (op == TSR_OP_ATTPV) sim->perf[TSR_PERF_ATT_PV_CYCLES(sim->num_ch)] += cyc;
    }
    if (sim->log) log_op(sim, w, err, info);
    sim->n_ops++;
    sim->perf[TSR_PERF_RETIRED] += 1;
    if (err_info) *err_info = err != TSR_ERR_NONE ? info : 0;
    return err;
}
