// test_fw_kvfa_plan.cpp — fw_kvfa_plan: the shared FA chunk planner (sw/sim/include/tsr_fa_plan.h)
// against the original planner, kept below verbatim as the oracle (the original tsr_ref_at.c code:
// tsr_fa_pass_rule, tsr_fa_ncg, plan_cuts, plan_sub_emit, tsr_fa_plan;
// only xmalloc / free became std::vector and the names got the w1a_ prefix).
//
// The chunk list is part of FLASH_ATTN's numerical contract (AT §6.6, SW §7.5): the firmware's
// kernel (k_fattn.cpp) and the reference (tsr_ref_fa) both take it from tsr_fa_plan.h, so this
// test pins that the shared planner gives the original lists. It compares, for random and adversarial mask summaries
// (causal / sliding-window tiles whose bound spreads straddle chunk cuts, tiny windows with both
// spreads next to each other, small NCG, two runs, holes, GENERAL, empty and malformed rows, up to
// 3 runs of new cells), configurations (C 1..16, T_MAX 1..8, PCAP down to one segment) and query
// tiles (T_q 1..8):
//   - libtsr_ref's tsr_fa_plan (the wrapper over the shared header) with the oracle: return code
//     and every chunk, also with a truncating capacity;
//   - tsr_fa_pass_rule / tsr_fa_ncg with the oracle's;
//   - the shared header's walk itself (tsr_fap_tile_init + tsr_fap_next) and its one-generation
//     list (tsr_fap_list, as k_fattn.cpp uses it; also past its capacity: the count) with the
//     oracle, including the -3 cases (the tile is refused).
//   test_fw_kvfa_plan [N]        N random cases per family (default 200000)
#undef _GNU_SOURCE
#include "test_util.h"

#include <vector>

extern "C" {
#include "tsr_fa_plan.h"
#include "tsr_ref_at.h"
}

namespace {

// =================================================================================================
// The original planner (the oracle), verbatim but for the names and the cut array.
// =================================================================================================
void w1a_tsr_fa_pass_rule(const tsr_att_cfg * cfg, uint32_t G, uint32_t * GS, uint32_t * T_q) {
    const uint32_t gs = G < cfg->t_max ? G : cfg->t_max;
    *GS = gs;
    *T_q = gs ? cfg->t_max / gs : 0u;
}

uint32_t w1a_tsr_fa_ncg(const tsr_att_cfg * cfg, uint32_t E, uint32_t D, uint32_t n_total) {
    if (D == 0 || E % D != 0 || cfg->C == 0) return 0;
    const uint64_t C4 = 4u * (uint64_t)cfg->C;
    const uint64_t pc = (uint64_t)cfg->C * (cfg->pcap / (16u * (E / D)));
    const uint64_t half = ((uint64_t)n_total + 1u) / 2u;
    const uint64_t ru = (half + C4 - 1u) / C4 * C4;
    const uint64_t m = ru > C4 ? ru : C4;
    uint64_t ncg = 128u;
    if (pc < ncg) ncg = pc;
    if (m < ncg) ncg = m;
    return (uint32_t)ncg;
}

inline int64_t i64max(int64_t a, int64_t b) { return a > b ? a : b; }
inline int64_t i64min(int64_t a, int64_t b) { return a < b ? a : b; }
inline uint32_t ceil16(uint32_t n) { return (n + 15u) >> 4; }

typedef struct plan_sub {
    uint32_t ulo, uhi;          // groups [ulo, uhi)
    int general;
    int multi;                  // FAST with T > 1 (bounds from lo_abs / hi_abs / flags)
    int64_t lo, hi;             // FAST: T = 1 the run; T > 1 lo_abs, hi_abs
    uint32_t flags;             // FAST T > 1: CAUSAL | SLIDE of the tile
} plan_sub;

typedef struct plan_emit {
    tsr_fa_chunk * out;
    size_t cap, count;
    int pass;                   // 0: chunks without new cells, 1: with
    const uint32_t (*new_runs)[2];
    uint32_t n_new;
} plan_emit;

int chunk_has_new(const plan_emit * pe, const tsr_fa_chunk * c) {
    const uint64_t a = 16u * (uint64_t)c->u0, b = 16u * ((uint64_t)c->u0 + c->n_groups);
    for (uint32_t i = 0; i < pe->n_new; ++i) {
        if ((uint64_t)pe->new_runs[i][0] < b && (uint64_t)pe->new_runs[i][1] > a && pe->new_runs[i][1] > pe->new_runs[i][0]) return 1;
    }
    return 0;
}

void plan_put(plan_emit * pe, const tsr_fa_chunk * c) {
    if (chunk_has_new(pe, c) != pe->pass) return;
    if (pe->count < pe->cap) pe->out[pe->count] = *c;
    pe->count += 1u;
}

uint32_t plan_cuts(const plan_sub * s, uint32_t ncg, uint32_t T, uint32_t * cuts, uint32_t cap) {
    uint32_t n = 0;
    for (uint64_t x = (uint64_t)s->ulo + ncg; x < s->uhi && n < cap; x += ncg) cuts[n++] = (uint32_t)x;
    if (!s->multi || s->flags == 0u) return n;
    int64_t sp[2][2];
    uint32_t nsp = 0;
    if (s->flags & TSR_ATT_CAUSAL) { sp[nsp][0] = s->hi - (int64_t)(T - 1u); sp[nsp][1] = s->hi; ++nsp; }
    if (s->flags & TSR_ATT_SLIDE)  { sp[nsp][0] = i64max(0, s->lo - (int64_t)(T - 1u)); sp[nsp][1] = s->lo; ++nsp; }
    for (int iter = 0; iter < 8; ++iter) {
        int changed = 0;
        for (uint32_t k = 0; k < n; ++k) {
            const int64_t X = 16 * (int64_t)cuts[k];
            for (uint32_t t = 0; t < nsp; ++t) {
                if (sp[t][0] < X && X < sp[t][1]) {
                    const uint32_t lo = (uint32_t)(sp[t][0] / 16), hi = (uint32_t)((sp[t][1] + 15) / 16);
                    cuts[k] = lo;
                    if (n < cap) cuts[n++] = hi;
                    changed = 1;
                    break;
                }
            }
        }
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
        if (!changed) return n;
    }
    return n;
}

int plan_sub_emit(plan_emit * pe, const plan_sub * s, uint32_t ncg, uint32_t T, const int64_t * L, const int64_t * H) {
    const uint32_t cap = (s->uhi - s->ulo) / ncg + 8u;
    std::vector<uint32_t> cutv(cap);
    uint32_t * cuts = cutv.data();
    const uint32_t ncut = plan_cuts(s, ncg, T, cuts, cap);
    int rc = 0;
    uint32_t a = s->ulo;
    for (uint32_t k = 0; k <= ncut; ++k) {
        const uint32_t b = k < ncut ? cuts[k] : s->uhi;
        tsr_fa_chunk c;
        c.u0 = a;
        c.n_groups = b - a;
        c.flags = 0;
        c.general = 0;
        const int64_t A = 16 * (int64_t)a, A2 = 16 * (int64_t)b, n16 = A2 - A;
        if (s->general) {
            c.general = 1;
            c.c_lo = 0;
            c.c_hi = (uint32_t)n16;
        } else if (!s->multi) {
            c.c_lo = (uint32_t)i64max(s->lo - A, 0);
            c.c_hi = (uint32_t)i64min(s->hi - A, n16);
        } else {
            const int causal = (s->flags & TSR_ATT_CAUSAL) != 0, slide = (s->flags & TSR_ATT_SLIDE) != 0;
            int64_t clo, chi;
            uint32_t fl = 0;
            if (s->lo <= A) clo = 0;
            else { clo = s->lo - A; if (slide) fl |= TSR_ATT_SLIDE; }
            if (s->hi - (causal ? (int64_t)(T - 1u) : 0) >= A2) chi = n16;
            else { chi = s->hi - A; if (causal) fl |= TSR_ATT_CAUSAL; }
            if (clo < 0 || clo > chi || chi > n16) { rc = -3; break; }
            for (uint32_t q = 0; q < T; ++q) {
                const int64_t dq = (int64_t)T - 1 - q;
                const int64_t lo_q = i64max(0, clo - ((fl & TSR_ATT_SLIDE) ? dq : 0));
                const int64_t hi_q = i64min(chi - ((fl & TSR_ATT_CAUSAL) ? dq : 0), n16);
                const int64_t elo = i64max(L[q] - A, 0), ehi = i64min(H[q] - A, n16);
                const int empty_a = hi_q <= lo_q, empty_b = ehi <= elo;
                if (empty_a != empty_b || (!empty_a && (lo_q != elo || hi_q != ehi))) rc = -3;
            }
            if (rc != 0) break;
            c.c_lo = (uint32_t)clo;
            c.c_hi = (uint32_t)chi;
            c.flags = fl;
        }
        plan_put(pe, &c);
        a = b;
    }
    return rc;
}

int run_ok(const tsr_msum_row * w, uint32_t n_kv) {
    if (w->lo0 > w->hi0 || w->lo1 > w->hi1 || w->hi0 > n_kv || w->hi1 > n_kv) return 0;
    if (w->hi0 > w->lo0 && w->hi1 > w->lo1 && !(w->hi0 <= w->lo1 || w->hi1 <= w->lo0)) return 0;
    return 1;
}

uint32_t row_runs(const tsr_msum_row * w, int64_t lo[2], int64_t hi[2]) {
    uint32_t n = 0;
    if (w->hi0 > w->lo0) { lo[n] = w->lo0; hi[n] = w->hi0; ++n; }
    if (w->hi1 > w->lo1) { lo[n] = w->lo1; hi[n] = w->hi1; ++n; }
    if (n == 2 && lo[1] < lo[0]) {
        const int64_t a = lo[0], b = hi[0];
        lo[0] = lo[1]; hi[0] = hi[1]; lo[1] = a; hi[1] = b;
    }
    return n;
}

long w1a_tsr_fa_plan(const tsr_att_cfg * cfg, uint32_t E, uint32_t D, const tsr_msum * ms, uint32_t q0, uint32_t T_q,
                     const uint32_t (*new_runs)[2], uint32_t n_new, tsr_fa_chunk * out, size_t cap) {
    if (T_q == 0 || T_q > TSR_T_MAX_ARCH || (uint64_t)q0 + T_q > ms->n_q) return -2;
    const uint32_t n_kv = ms->n_kv;
    plan_sub sub[2];
    uint32_t nsub = 0;
    int64_t L[TSR_T_MAX_ARCH] = { 0 }, H[TSR_T_MAX_ARCH] = { 0 };
    if (ms->flags & TSR_MSUM_GENERAL) {
        if (n_kv > 0) {
            sub[0] = plan_sub{ 0, ceil16(n_kv), 1, 0, 0, 0, 0 };
            nsub = 1;
        }
    } else {
        int64_t rlo[TSR_T_MAX_ARCH][2], rhi[TSR_T_MAX_ARCH][2];
        uint32_t nr[TSR_T_MAX_ARCH];
        int all_one = 1;
        int64_t hlo = INT64_MAX, hhi = -1;
        for (uint32_t q = 0; q < T_q; ++q) {
            const tsr_msum_row * w = &ms->rows[q0 + q];
            if (!run_ok(w, n_kv)) return -2;
            nr[q] = row_runs(w, rlo[q], rhi[q]);
            if (nr[q] != 1u) all_one = 0;
            for (uint32_t k = 0; k < nr[q]; ++k) { hlo = i64min(hlo, rlo[q][k]); hhi = i64max(hhi, rhi[q][k]); }
        }
        if (T_q == 1u) {
            for (uint32_t k = 0; k < nr[0]; ++k) {
                sub[nsub++] = plan_sub{ (uint32_t)(rlo[0][k] / 16), (uint32_t)((rhi[0][k] + 15) / 16), 0, 0,
                                          rlo[0][k], rhi[0][k], 0 };
            }
        } else if (hhi >= 0) {
            int fit = all_one;
            uint32_t fl = 0;
            int64_t lo_abs = 0, hi_abs = 0;
            if (fit) {
                for (uint32_t q = 0; q < T_q; ++q) { L[q] = rlo[q][0]; H[q] = rhi[q][0]; }
                hi_abs = H[T_q - 1];
                lo_abs = L[T_q - 1];
                int heq = 1, leq = 1, hca = 1, lsl = lo_abs > 0;
                for (uint32_t q = 0; q < T_q; ++q) {
                    const int64_t dq = (int64_t)T_q - 1 - q;
                    heq &= H[q] == hi_abs;
                    leq &= L[q] == lo_abs;
                    hca &= H[q] == hi_abs - dq;
                    lsl &= L[q] == i64max(0, lo_abs - dq);
                }
                if (!heq) { if (hca) fl |= TSR_ATT_CAUSAL; else fit = 0; }
                if (!leq) { if (lsl) fl |= TSR_ATT_SLIDE; else fit = 0; }
            }
            if (fit) {
                int64_t mnL = INT64_MAX, mxH = -1;
                for (uint32_t q = 0; q < T_q; ++q) { mnL = i64min(mnL, L[q]); mxH = i64max(mxH, H[q]); }
                sub[nsub++] = plan_sub{ (uint32_t)(mnL / 16), (uint32_t)((mxH + 15) / 16), 0, 1, lo_abs, hi_abs, fl };
            } else {
                sub[nsub++] = plan_sub{ (uint32_t)(hlo / 16), (uint32_t)((hhi + 15) / 16), 1, 0, 0, 0, 0 };
            }
        }
    }
    if (nsub == 0) {
        if (cap > 0) out[0] = tsr_fa_chunk{ 0, 1, 0, 0, 0, 0 };
        return 1;
    }
    uint64_t n_total = 0;
    for (uint32_t k = 0; k < nsub; ++k) n_total += sub[k].uhi - sub[k].ulo;
    const uint32_t ncg = w1a_tsr_fa_ncg(cfg, E, D, (uint32_t)n_total);
    if (ncg == 0) return -1;
    plan_emit pe = { out, cap, 0, 0, new_runs, new_runs ? n_new : 0u };
    for (pe.pass = 0; pe.pass < 2; ++pe.pass) {
        for (uint32_t k = 0; k < nsub; ++k) {
            if (plan_sub_emit(&pe, &sub[k], ncg, T_q, L, H) != 0) return -3;
        }
    }
    return (long)pe.count;
}

// =================================================================================================
// Generators
// =================================================================================================
struct pcase {
    tsr_att_cfg cfg;
    uint32_t E, D, n_kv, T, flags;
    std::vector<tsr_msum_row> rows;
    uint32_t new_runs[3][2];
    uint32_t n_new;
};

uint32_t pick(rng_t * r, std::initializer_list<uint32_t> v) { return *(v.begin() + rng_below(r, uint32_t(v.size()))); }

// A query tile: family 0 random runs, 1 causal prefill tile, 2 sliding window (small windows
// included), 3 causal / SWA bounds placed on chunk cuts, 4 GENERAL / malformed / empty.
void gen(rng_t * r, int fam, pcase & c) {
    c.cfg.C = pick(r, {1, 2, 3, 4, 4, 4, 16});
    c.cfg.t_max = 1u + rng_below(r, 8);
    c.cfg.pcap = rng_below(r, 4) == 0 ? 16u * (1u + rng_below(r, 64)) : 9344u;
    c.cfg.k_max = 0;
    c.D = pick(r, {64, 128, 256, 512});
    c.E = c.D * pick(r, {1, 2, 8});
    c.T = 1u + rng_below(r, 8);
    c.n_kv = fam == 3 ? 16u * (1u + rng_below(r, 300)) : rng_below(r, 3) == 0 ? rng_below(r, 64) : rng_below(r, 6000);
    c.flags = 0;
    c.rows.assign(c.T, tsr_msum_row{0, 0, 0, 0});
    const uint32_t n = c.n_kv;
    auto clamp = [&](int64_t v) { return uint32_t(v < 0 ? 0 : v > int64_t(n) ? int64_t(n) : v); };
    switch (fam) {
    case 0:   // random 0..2 runs per row
        for (auto & w : c.rows) {
            const uint32_t k = rng_below(r, 3);
            uint32_t a = rng_below(r, n + 1), b = rng_below(r, n + 1);
            if (a > b) std::swap(a, b);
            if (k >= 1) { w.lo0 = a; w.hi0 = b; }
            if (k == 2) {
                uint32_t x = rng_below(r, n + 1), y = rng_below(r, n + 1);
                if (x > y) std::swap(x, y);
                if (y <= a || x >= b) { w.lo1 = x; w.hi1 = y; }
            }
        }
        break;
    case 1:   // causal prefill tile (llama.cpp: row q sees [0, pos0 + q + 1))
    case 2: { // sliding window (standard SWA: row q sees (p_q - n_swa, p_q])
        const uint32_t w = fam == 2 ? (rng_below(r, 2) ? 1u + rng_below(r, 40) : 1u + rng_below(r, 1500)) : 0u;
        const uint32_t p0 = rng_below(r, n + 1);
        for (uint32_t q = 0; q < c.T; ++q) {
            const int64_t hi = int64_t(p0) + q + 1 - int64_t(c.T) + 1;   // the tile's last row at p0
            const int64_t lo = fam == 2 ? hi - int64_t(w) : 0;
            c.rows[q].lo0 = clamp(lo);
            c.rows[q].hi0 = clamp(hi);
            if (c.rows[q].hi0 < c.rows[q].lo0) c.rows[q].hi0 = c.rows[q].lo0;
        }
        break;
    }
    case 3: { // bounds whose spreads straddle a multiple of 16 (chunk cuts at small NCG)
        const uint32_t g = 1u + rng_below(r, n / 16u);
        const int64_t hi = 16 * int64_t(g) + int64_t(rng_below(r, c.T + 1));
        const int64_t w = rng_below(r, 3) == 0 ? int64_t(c.T) + int64_t(rng_below(r, 20)) : int64_t(16 * (1 + rng_below(r, 40)) + rng_below(r, 9));
        const uint32_t mode = rng_below(r, 3);   // 0 causal, 1 causal + slide, 2 slide only
        for (uint32_t q = 0; q < c.T; ++q) {
            const int64_t dq = int64_t(c.T) - 1 - q;
            const int64_t h = mode == 2 ? hi : hi - dq;
            const int64_t l = mode == 0 ? 0 : (mode == 2 ? hi - w - dq : h - w);
            c.rows[q].lo0 = clamp(l);
            c.rows[q].hi0 = clamp(h);
            if (c.rows[q].hi0 < c.rows[q].lo0) c.rows[q].hi0 = c.rows[q].lo0;
        }
        break;
    }
    default: {
        const uint32_t k = rng_below(r, 4);
        if (k == 0) c.flags = TSR_MSUM_GENERAL;
        for (auto & w : c.rows) {
            if (k == 1) { w.lo0 = rng_below(r, n + 3); w.hi0 = rng_below(r, n + 3); w.lo1 = rng_below(r, n + 3); w.hi1 = rng_below(r, n + 3); }
            if (k == 0) { w.lo0 = rng_below(r, n + 1); w.hi0 = n; }
        }
        break;
    }
    }
    c.n_new = rng_below(r, 4);
    for (uint32_t i = 0; i < c.n_new; ++i) {
        const uint32_t a = rng_below(r, n + 8), len = rng_below(r, 3) ? 1u + rng_below(r, c.T + 1) : rng_below(r, 3);
        c.new_runs[i][0] = a;
        c.new_runs[i][1] = a + len;
    }
}

struct new_ctx { const uint32_t (*runs)[2]; uint32_t n; };
int has_new_cb(void * ctx, uint32_t u0, uint32_t ng) {
    const new_ctx * p = static_cast<const new_ctx *>(ctx);
    const uint64_t a = 16u * uint64_t(u0), b = 16u * (uint64_t(u0) + ng);
    for (uint32_t i = 0; i < p->n; ++i)
        if (uint64_t(p->runs[i][0]) < b && uint64_t(p->runs[i][1]) > a && p->runs[i][1] > p->runs[i][0]) return 1;
    return 0;
}

bool same_chunks(const std::vector<tsr_fa_chunk> & a, const std::vector<tsr_fa_chunk> & b, size_t n) {
    for (size_t i = 0; i < n; ++i)
        if (memcmp(&a[i], &b[i], sizeof(tsr_fa_chunk)) != 0) return false;
    return true;
}

long g_err[5], g_chunks, g_spread_hits;

void one(const pcase & c, const char * fam) {
    tsr_msum ms{c.T, c.n_kv, c.flags, c.rows.data()};
    const uint32_t (*nr)[2] = c.n_new ? c.new_runs : nullptr;
    std::vector<tsr_fa_chunk> a(4096), b(4096);
    const long ra = w1a_tsr_fa_plan(&c.cfg, c.E, c.D, &ms, 0, c.T, nr, c.n_new, a.data(), a.size());
    const long rb = tsr_fa_plan(&c.cfg, c.E, c.D, &ms, 0, c.T, nr, c.n_new, b.data(), b.size());
    CHECK(ra == rb, "%s: return %ld vs %ld (shared): C %u t_max %u pcap %u E %u D %u n_kv %u T %u", fam, ra, rb,
          c.cfg.C, c.cfg.t_max, c.cfg.pcap, c.E, c.D, c.n_kv, c.T);
    if (ra < 0) g_err[-ra]++;
    if (ra == rb && ra > 0) {
        const size_t n = size_t(ra) < a.size() ? size_t(ra) : a.size();
        g_chunks += long(n);
        CHECK(same_chunks(a, b, n), "%s: chunk lists differ (%ld chunks)", fam, ra);
        // a truncating capacity
        if (ra > 1) {
            std::vector<tsr_fa_chunk> b2(size_t(ra) / 2u, tsr_fa_chunk{7, 7, 7, 7, 7, 7});
            const long r2 = tsr_fa_plan(&c.cfg, c.E, c.D, &ms, 0, c.T, nr, c.n_new, b2.data(), b2.size());
            CHECK(r2 == ra && same_chunks(a, b2, b2.size()), "%s: truncated list", fam);
        }
    }
    // The shared walk as the firmware uses it: init, then a dry run to the end, then the list.
    tsr_fap_tile t;
    const int ri = tsr_fap_tile_init(&t, c.cfg.C, c.cfg.pcap, c.E, c.D, c.n_kv, c.flags, &c.rows[0].lo0, c.T);
    for (uint32_t k = 0; k < t.nsub; ++k) g_spread_hits += t.sub[k].n_rm != 0u;
    long rw = ri;
    std::vector<tsr_fa_chunk> w;
    if (ri == 0) {
        new_ctx nc{nr, c.n_new};
        tsr_fap_iter it;
        tsr_fap_iter_init(&it);
        for (;;) {
            tsr_fap_chunk x;
            const int k = tsr_fap_next(&t, &it, has_new_cb, &nc, &x);
            if (k < 0) { rw = -3; break; }
            if (k == 0) break;
            w.push_back(tsr_fa_chunk{x.u0, x.n_groups, x.c_lo, x.c_hi, x.flags, x.general});
            if (w.size() > 100000u) { rw = -99; break; }
        }
        if (rw == 0) rw = long(w.size());
    }
    CHECK(rw == ra, "%s: shared walk %ld vs original %ld", fam, rw, ra);
    if (rw == ra && ra > 0) CHECK(same_chunks(a, w, size_t(ra) < a.size() ? size_t(ra) : a.size()), "%s: walk differs", fam);
    // tsr_fap_list (the firmware's one-generation list): the same list, and the count past its cap.
    if (ri == 0) {
        new_ctx nc{nr, c.n_new};
        std::vector<tsr_fap_chunk> l(4096);
        const long rl = tsr_fap_list(&t, has_new_cb, &nc, l.data(), uint32_t(l.size()));
        CHECK(rl == ra, "%s: tsr_fap_list %ld vs original %ld", fam, rl, ra);
        if (rl == ra && ra > 0) {
            std::vector<tsr_fa_chunk> lc(static_cast<size_t>(ra));
            for (long i = 0; i < ra; ++i)
                lc[size_t(i)] = tsr_fa_chunk{l[size_t(i)].u0, l[size_t(i)].n_groups, l[size_t(i)].c_lo, l[size_t(i)].c_hi,
                                             l[size_t(i)].flags, l[size_t(i)].general};
            CHECK(same_chunks(a, lc, size_t(ra)), "%s: tsr_fap_list differs", fam);
            if (ra > 1) {
                std::vector<tsr_fap_chunk> l2(size_t(ra) - 1u);
                CHECK(tsr_fap_list(&t, has_new_cb, &nc, l2.data(), uint32_t(l2.size())) == ra, "%s: tsr_fap_list over its cap", fam);
            }
        }
    }
}

}  // namespace

int main(int argc, char ** argv) {
    const long N = argc > 1 ? atol(argv[1]) : 200000;
    rng_t r = { 0xFA7A11u };
    // pass rule and NCG over a grid
    for (uint32_t C : {1u, 2u, 3u, 4u, 16u, 0x00FFFFFFu, 0x40000000u, 0xFFFFFFFFu})
        for (uint32_t tm = 1; tm <= 8u; ++tm)
            for (uint32_t pcap : {0u, 16u, 512u, 9344u, 0xFFFFFFFFu})
                for (uint32_t D : {64u, 128u, 512u})
                    for (uint32_t ed : {1u, 2u, 8u}) {
                        const tsr_att_cfg cfg{C, tm, pcap, 0};
                        for (uint32_t G : {1u, 2u, 5u, 7u, 8u, 16u, 64u}) {
                            uint32_t a1, a2, b1, b2;
                            w1a_tsr_fa_pass_rule(&cfg, G, &a1, &a2);
                            tsr_fa_pass_rule(&cfg, G, &b1, &b2);
                            CHECK(a1 == b1 && a2 == b2, "pass rule G %u t_max %u", G, tm);
                        }
                        for (uint32_t nt : {0u, 1u, 7u, 8u, 33u, 255u, 256u, 1024u, 65535u, 0x7FFFFFFFu, 0xFFFFFFFEu, 0xFFFFFFFFu}) {
                            const uint32_t want = w1a_tsr_fa_ncg(&cfg, D * ed, D, nt);
                            CHECK(want == tsr_fa_ncg(&cfg, D * ed, D, nt) && want == tsr_fap_ncg(C, pcap, D * ed, D, nt),
                                  "ncg C %u pcap %u E %u D %u n %u", C, pcap, D * ed, D, nt);
                        }
                    }
    static const char * const fams[] = {"random", "causal", "swa", "spread-on-cut", "general/malformed"};
    for (int fam = 0; fam < 5; ++fam) {
        for (long i = 0; i < N; ++i) {
            pcase c;
            gen(&r, fam, c);
            one(c, fams[fam]);
        }
        printf("  %-18s %ld cases\n", fams[fam], N);
    }
    printf("fw_kvfa_plan: %ld chunks compared; refusals -1 %ld, -2 %ld, -3 %ld; tiles with a cut inside a spread %ld\n",
           g_chunks, g_err[1], g_err[2], g_err[3], g_spread_hits);
    CHECK(g_spread_hits > 1000, "the adversarial family reached rule 5 only %ld times", g_spread_hits);
    return test_finish("fw_kvfa_plan");
}
