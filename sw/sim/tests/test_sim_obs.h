// test_sim_obs.h — SW §4.7's determinism observables of a simulator run (docs/design/06-software-
// verification.md §4.7), collected through the listeners of tsr_sim_int.h §5: the per-unit op
// streams (PUSH / QCTRL events projected per unit, deps as relative tickets), the RISC-V's SPM
// store stream (RV_SPM_WR access events) and the final written DRAM beats and SPM lines (write
// extents of every op that ran, the RISC-V's SPM stores; content at the end). The LCQ records are
// the caller's (test_sim_rvhost.h collects them). Factored out of test_sim_sched.c for
// sim_w1b_e2e; used by both.
//   obs_attach (at create, before the hart runs) -> run -> obs_finish -> obs_equal / obs_free
#pragma once

#include "test_sim_util.h"

#include "tsr_bus.h"
#include "tsr_sim_int.h"

// ---------------------------------------------------------------------------
// Observables (SW §4.7)
// ---------------------------------------------------------------------------
typedef struct ival_t { uint32_t ch; uint64_t lo, hi; } ival_t;

typedef struct obs {
    uint32_t base[TSR_NUM_UNITS];               // ISSUED_SEQ at attach (relative tickets)
    uint64_t stream[TSR_NUM_UNITS];             // per-unit op stream hashes
    uint32_t stream_n[TSR_NUM_UNITS];
    uint64_t spm_st;                            // RISC-V SPM store stream hash
    uint32_t spm_st_n;
    uint8_t * spm_wr;                           // written SPM lines
    uint32_t spm_lines;
    ival_t * dram_wr;                           // written DRAM beat ranges
    uint32_t n_dram, cap_dram;
    uint64_t final_hash;                        // content of the written lines / beats at the end
    uint32_t n_written_lines;
    uint64_t n_written_beats;
    uint64_t mtime, instret, ff_jumps;
} obs;

static uint64_t h64(uint64_t h, uint64_t v) {
    for (int i = 0; i < 8; ++i) {
        h ^= (v >> (8 * i)) & 0xFFu;
        h *= 1099511628211ull;
    }
    return h;
}

static uint32_t rel_ticket(const obs * o, uint32_t t) {
    const uint32_t u = t >> 28;
    if (u >= TSR_NUM_UNITS) return t;
    return (u << 28) | ((t - o->base[u]) & 0x0FFFFFFFu);
}

static void obs_q(void * ctx, tsr_sim * sim, const tsr_q_event * ev) {
    (void)sim;
    obs * o = (obs *)ctx;
    if (ev->kind == TSR_QEV_PUSH || ev->kind == TSR_QEV_QCTRL) {
        const uint32_t u = ev->unit < TSR_NUM_UNITS ? ev->unit : TSR_UNIT_SYNC;   // QCTRL without QCLEAR
        uint64_t h = o->stream[u] ? o->stream[u] : 1469598103934665603ull;
        h = h64(h, ev->kind);
        for (int i = 0; i < 8; ++i) h = h64(h, ev->w[i]);
        h = h64(h, ev->kind == TSR_QEV_PUSH ? rel_ticket(o, ev->dep[0]) : 0u);
        h = h64(h, ev->kind == TSR_QEV_PUSH ? rel_ticket(o, ev->dep[1]) : 0u);
        h = h64(h, ev->push);
        o->stream[u] = h;
        o->stream_n[u]++;
        return;
    }
    if (ev->kind != TSR_QEV_DONE || !ev->ran || ev->ext == NULL) return;
    for (uint32_t i = 0; i < ev->ext->n; ++i) {
        const tsr_extent * e = &ev->ext->e[i];
        if (e->write != TSR_EXT_WRITE) continue;
        if (e->kind == TSR_EXT_SPM) {
            for (uint64_t l = e->lo; l < e->hi && l < o->spm_lines; ++l) o->spm_wr[l] = 1;
        } else if (e->kind == TSR_EXT_DRAM && e->ch == TSR_EXT_CH_ALL) {
            // a write extent on every channel (KVST's plane, AT §5.8) is one range per channel.
            const uint32_t nch = tsr_sim_config(sim)->num_ch;
            for (uint32_t c = 0; c < nch; ++c) {
                if (o->n_dram == o->cap_dram) {
                    o->cap_dram = o->cap_dram ? 2 * o->cap_dram : 64;
                    o->dram_wr = (ival_t *)realloc(o->dram_wr, o->cap_dram * sizeof(ival_t));
                }
                ival_t * r = &o->dram_wr[o->n_dram++];
                r->ch = c;
                r->lo = e->lo;
                r->hi = e->hi;
            }
        } else if (e->kind == TSR_EXT_DRAM) {
            if (o->n_dram == o->cap_dram) {
                o->cap_dram = o->cap_dram ? 2 * o->cap_dram : 64;
                o->dram_wr = (ival_t *)realloc(o->dram_wr, o->cap_dram * sizeof(ival_t));
            }
            ival_t * r = &o->dram_wr[o->n_dram++];
            r->ch = e->ch;
            r->lo = e->lo;
            r->hi = e->hi;
        }
    }
}

static void obs_acc(void * ctx, tsr_sim * sim, const tsr_sim_acc * a) {
    (void)sim;
    obs * o = (obs *)ctx;
    if (a->kind != TSR_ACC_RV_SPM_WR) return;
    uint64_t h = o->spm_st ? o->spm_st : 1469598103934665603ull;
    h = h64(h, a->addr);
    h = h64(h, a->size);
    h = h64(h, a->val);
    o->spm_st = h;
    o->spm_st_n++;
    for (uint64_t l = a->addr / 64u; l <= (a->addr + a->size - 1u) / 64u && l < o->spm_lines; ++l) o->spm_wr[l] = 1;
}

static void obs_attach(obs * o, tsr_sim * sim) {
    memset(o, 0, sizeof(*o));
    o->spm_lines = tsr_sim_config(sim)->spm_bytes / 64u;
    o->spm_wr = (uint8_t *)calloc(o->spm_lines, 1);
    for (uint32_t u = 0; u < TSR_NUM_UNITS; ++u) o->base[u] = tsr_sim_cw_read(sim, TSR_CW_ISSUED_SEQ(u));
    tsr_sim_add_listener(sim, obs_q, o);
    tsr_sim_add_acc_listener(sim, obs_acc, o);
}

static int ival_cmp(const void * a, const void * b) {
    const ival_t * x = (const ival_t *)a, * y = (const ival_t *)b;
    if (x->ch != y->ch) return x->ch < y->ch ? -1 : 1;
    return (x->lo > y->lo) - (x->lo < y->lo);
}

// The final state of everything written: SPM lines in order, then the merged DRAM ranges.
static void obs_finish(obs * o, tsr_sim * sim) {
    uint64_t h = 1469598103934665603ull;
    const uint8_t * spm = tsr_sim_spm(sim);
    for (uint32_t l = 0; l < o->spm_lines; ++l) {
        if (!o->spm_wr[l]) continue;
        o->n_written_lines++;
        h = h64(h, l);
        for (uint32_t k = 0; k < 64u; ++k) h = (h ^ spm[64u * l + k]) * 1099511628211ull;
    }
    if (o->n_dram > 1) qsort(o->dram_wr, o->n_dram, sizeof(ival_t), ival_cmp);
    for (uint32_t i = 0; i < o->n_dram;) {
        const uint32_t ch = o->dram_wr[i].ch;
        uint64_t lo = o->dram_wr[i].lo, hi = o->dram_wr[i].hi;
        uint32_t j = i + 1;
        while (j < o->n_dram && o->dram_wr[j].ch == ch && o->dram_wr[j].lo <= hi) {
            if (o->dram_wr[j].hi > hi) hi = o->dram_wr[j].hi;
            ++j;
        }
        h = h64(h64(h64(h, ch), lo), hi);
        o->n_written_beats += hi - lo;
        const uint8_t * d = tsr_sim_dram(sim, ch) + lo * 64u;
        for (uint64_t k = 0; k < (hi - lo) * 64u; ++k) h = (h ^ d[k]) * 1099511628211ull;
        i = j;
    }
    o->final_hash = h;
    o->mtime = tsr_sim_now(sim);
    o->instret = tsr_sim_rv_instret(sim);
    o->ff_jumps = tsr_sim_rv_agent(sim) ? tsr_rv_agent_ff_jumps(tsr_sim_rv_agent(sim)) : 0;
}

static void obs_free(obs * o) {
    free(o->spm_wr);
    free(o->dram_wr);
    o->spm_wr = NULL;
    o->dram_wr = NULL;
}

// Compares the four observables of b against a (records separately).
static int obs_equal(const obs * a, const obs * b, const char * what) {
    int ok = 1;
    for (uint32_t u = 0; u < TSR_NUM_UNITS; ++u) {
        if (a->stream[u] != b->stream[u] || a->stream_n[u] != b->stream_n[u]) {
            CHECK(0, "%s: unit %u op stream differs (%u vs %u entries)", what, u, a->stream_n[u], b->stream_n[u]);
            ok = 0;
        }
    }
    if (a->spm_st != b->spm_st || a->spm_st_n != b->spm_st_n) {
        CHECK(0, "%s: RISC-V SPM store stream differs (%u vs %u stores)", what, a->spm_st_n, b->spm_st_n);
        ok = 0;
    }
    if (a->final_hash != b->final_hash || a->n_written_lines != b->n_written_lines || a->n_written_beats != b->n_written_beats) {
        CHECK(0, "%s: final written state differs (%u / %u SPM lines, %llu / %llu DRAM beats)", what, a->n_written_lines,
              b->n_written_lines, (unsigned long long)a->n_written_beats, (unsigned long long)b->n_written_beats);
        ok = 0;
    }
    return ok;
}
