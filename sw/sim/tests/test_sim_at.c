// test_sim_at.c — the simulator's AT ops (P1.5, sw/sim/tsr_at.c): KVST, ATTQK, ATTPV, ASMAX, AMASK,
// AMERGE and ANORM pushed as encoded ops through the test hooks (tsr_sim_op, tsr_sim_push) and
// compared bit for bit with direct tsr_ref_at calls on copies of the SPM and of the KV planes;
// every check code with ERR_INFO and the check order (AT §5.8, EX §8.1), CAPS gating, the operand
// extents, capture of the plane beats (KVST read-modify-write reads, streamed groups and slices),
// the cycle model (AT §13.4, VU §7.9), PERF (ATT_QK / ATT_PV / MVM / VU / DMA cycles, CH_RD_BEATS,
// KVST_RMW, ATT_PATH_STALL, VU_LINES) on the queue path under ACTIVE, and complete AT §6.6 flash-
// attention decode layers: QACT and KVST of the new token's K and V, QACT of Q, then per chunk of
// tsr_fa_plan ATTQK (per head subset, MAXOUT), ASMAX, ATTPV (LSUM), AMERGE, and ANORM, pushed
// through the queue model with dependencies and checked against tsr_ref_fa bit for bit (one pass;
// two head subsets with a wrapped two-run SWA mask; a Qwen2.5-32B-like shape).
#include "test_sim_util.h"
#include "tsr_sim_int.h"
#include "tsr_ref_at.h"

#define SPM_BYTES (4u << 20)                  // the U200 SPM: 65,536 lines
#define SPM_LINES (SPM_BYTES / 64u)
#define CH_BYTES  (64ull << 20)               // 1 Mi beats per channel
#define LIMIT     (CH_BYTES / 64u)
#define NCH       4u
#define CAPS_AT   (TSR_CAPS_VU | TSR_CAPS_ATT)
#define ALL_WORDS 0xFF00u

// ---------------------------------------------------------------------------
// Encoders
// ---------------------------------------------------------------------------
static tsr_insn e_kvst(uint32_t kv, uint32_t n_rows, uint32_t src, uint32_t B, uint32_t E64, uint32_t LG, uint32_t c0,
                       uint32_t ss) {
    tsr_insn i = {{0}};
    i.w[0] = TSR_OP_KVST;
    TSR_INSN_SET(i.w, TSR_KVST_KV, kv);
    TSR_INSN_SET(i.w, TSR_KVST_N_ROWS_M1, n_rows - 1u);
    TSR_INSN_SET(i.w, TSR_KVST_SRC, src);
    TSR_INSN_SET(i.w, TSR_KVST_BASE, B);
    TSR_INSN_SET(i.w, TSR_KVST_E64, E64);
    TSR_INSN_SET(i.w, TSR_KVST_LG, LG);
    TSR_INSN_SET(i.w, TSR_KVST_C0, c0);
    TSR_INSN_SET(i.w, TSR_KVST_SRC_STRIDE, ss);
    return i;
}
// ATTQK / ATTPV fields. w6 = c_lo | c_hi << 16 (ATTQK) or LG (ATTPV); w7 = mdst (ATTQK) or cstride.
typedef struct att_f {
    uint32_t fmt, flags, G, hg0, E64, dcode, Tq, GS, rs, dst, x, B, u0, n, w6, w7;
} att_f;
static tsr_insn e_att(uint32_t op, const att_f * f) {
    tsr_insn i = {{0}};
    i.w[0] = op | (f->fmt << 8) | (f->flags << 12);
    TSR_INSN_SET(i.w, TSR_ATTQK_G_M1, f->G - 1u);
    TSR_INSN_SET(i.w, TSR_ATTQK_HG0, f->hg0);
    i.w[1] = f->dst;
    i.w[2] = f->x;
    i.w[3] = f->B;
    TSR_INSN_SET(i.w, TSR_ATTQK_E64, f->E64);
    TSR_INSN_SET(i.w, TSR_ATTQK_DCODE, f->dcode);
    TSR_INSN_SET(i.w, TSR_ATTQK_TQ_M1, f->Tq - 1u);
    TSR_INSN_SET(i.w, TSR_ATTQK_GS_M1, f->GS - 1u);
    TSR_INSN_SET(i.w, TSR_ATTQK_RSTRIDE, f->rs);
    TSR_INSN_SET(i.w, TSR_ATTQK_U0, f->u0);
    TSR_INSN_SET(i.w, TSR_ATTQK_N_GROUPS, f->n);
    i.w[6] = f->w6;
    i.w[7] = f->w7;
    return i;
}
static uint32_t lohi(uint32_t lo, uint32_t hi) { return lo | (hi << 16); }
static tsr_insn e_asmax(uint32_t dst, uint32_t src, uint32_t mvec, uint32_t nl, uint32_t st, uint32_t R, uint32_t scale) {
    tsr_insn i = {{ TSR_OP_ASMAX, dst, src, mvec, nl | (st << 16), R, scale, 0 }};
    return i;
}
static tsr_insn e_amask(uint32_t dst, uint32_t src, uint32_t mask, uint32_t nl, uint32_t st, uint32_t R, uint32_t rpq,
                        uint32_t ms, uint32_t mdst) {
    tsr_insn i = {{ TSR_OP_AMASK, dst, src, mask, nl | (st << 16), R | (rpq << 16), ms, mdst }};
    return i;
}
static tsr_insn e_amerge(uint32_t init, uint32_t orun, uint32_t parts, uint32_t mk, uint32_t st, uint32_t R, uint32_t dcode,
                         uint32_t scale, uint32_t cs) {
    tsr_insn i = {{ TSR_OP_AMERGE | (init << 12), orun, parts, mk, st, R | (dcode << 16), scale, cs }};
    return i;
}
static tsr_insn e_anorm(uint32_t out, uint32_t orun, uint32_t lrun, uint32_t R, uint32_t dcode) {
    tsr_insn i = {{ TSR_OP_ANORM, out, orun, lrun, R | (dcode << 16), 0, 0, 0 }};
    return i;
}
static uint32_t dcode_of(uint32_t D) { return D == 64u ? 0u : D == 128u ? 1u : D == 256u ? 2u : 3u; }

// ---------------------------------------------------------------------------
// Simulator, images, data
// ---------------------------------------------------------------------------
static tsr_sim * mk_sim(uint32_t caps, uint32_t t_max) {
    tsr_config c = cfg_u200(t_max);
    c.spm_bytes = SPM_BYTES;
    c.ch_bytes = CH_BYTES;
    c.caps = caps;
    tsr_sim * s = tsr_sim_create(&c, NULL);
    if (s == NULL) {
        fprintf(stderr, "tsr_sim_create failed\n");
        exit(1);
    }
    return s;
}

static uint8_t * spm_copy(tsr_sim * s) {
    uint8_t * p = (uint8_t *)malloc(SPM_BYTES);
    memcpy(p, tsr_sim_spm(s), SPM_BYTES);
    return p;
}
static float * F(uint8_t * img, uint32_t line) { return (float *)(void *)(img + (size_t)line * 64u); }
static uint32_t * U(uint8_t * img, uint32_t line) { return (uint32_t *)(void *)(img + (size_t)line * 64u); }
static int spm_same(tsr_sim * s, const uint8_t * exp, const char * what) {
    const uint8_t * a = tsr_sim_spm(s);
    if (memcmp(a, exp, SPM_BYTES) == 0) {
        CHECK(1, "%s", what);
        return 1;
    }
    size_t i = 0;
    while (a[i] == exp[i]) ++i;
    CHECK(0, "%s: SPM differs from the direct reference first at line %zu byte %zu (sim %02x exp %02x)", what, i / 64u,
          i % 64u, a[i], exp[i]);
    return 0;
}

static uint32_t rand_word(rng_t * r) {
    const uint32_t sgn = rng_u32(r) & 0x80000000u;
    switch (rng_below(r, 40)) {
    case 0: return sgn;
    case 1: return sgn | 0x7F800000u;
    case 2: return 0x7FC00000u | (rng_u32(r) & 0x803FFFFFu);
    case 3: return sgn | (rng_u32(r) & 0x007FFFFFu);
    default: return f2u((float)(rng_normal(r) * 3.0));
    }
}
static void rand_lines(rng_t * r, tsr_sim * s, uint32_t line, uint32_t n, int specials) {
    uint32_t * w = U(tsr_sim_spm(s), line);
    for (uint32_t i = 0; i < 16u * n; ++i) w[i] = specials ? rand_word(r) : f2u((float)(rng_normal(r) * 3.0));
}

// One cell row as llama.cpp stores it: QACT A32 of random f32 values (fp16 scales, int8 quants).
static void rand_kv_row(rng_t * r, uint32_t E, tsr_block_q8_0 * row) {
    float * x = (float *)malloc(E * sizeof(float));
    uint8_t * qa = (uint8_t *)malloc((size_t)tsr_qa_lines(TSR_QA_A32, E) * 64u);
    const double sc = ldexp(1.0, (int)rng_below(r, 6) - 2);
    for (uint32_t j = 0; j < E; ++j) x[j] = (float)(rng_normal(r) * sc);
    if (rng_below(r, 16) == 0) memset(x, 0, 32 * sizeof(float));          // a zero block (d = 0)
    tsr_qact(TSR_QA_A32, x, E, qa);
    const int8_t * qs = tsr_qa_qs(TSR_QA_A32, E, qa);
    const float * d = tsr_qa_d(TSR_QA_A32, E, qa);
    for (uint32_t b = 0; b < E / 32u; ++b) {
        row[b].d = tsr_f32_to_fp16(d[b]);
        memcpy(row[b].qs, qs + 32u * b, 32);
    }
    free(qa);
    free(x);
}
static void planes_of(tsr_sim * s, uint32_t B, uint8_t * pl[NCH]) {
    for (uint32_t c = 0; c < NCH; ++c) pl[c] = tsr_sim_dram(s, c) + (uint64_t)B * 64u;
}
// Fills cells [0, n_cells) of a K or V plane at beat B with random rows.
static void fill_plane(rng_t * r, tsr_sim * s, uint32_t kv, uint32_t B, uint32_t E, uint32_t LG, uint32_t n_cells) {
    const tsr_kv_geom g = { E, NCH, LG };
    uint8_t * pl[NCH];
    planes_of(s, B, pl);
    tsr_block_q8_0 * row = (tsr_block_q8_0 *)malloc((E / 32u) * sizeof(tsr_block_q8_0));
    for (uint32_t cell = 0; cell < n_cells; ++cell) {
        rand_kv_row(r, E, row);
        tsr_kv_pack_row(kv, &g, pl, cell, row);
    }
    free(row);
}
// QA(A32, K) of random f32 values written to SPM line `line`.
static void put_qa(rng_t * r, tsr_sim * s, uint32_t line, uint32_t K) {
    float * x = (float *)malloc(K * sizeof(float));
    for (uint32_t j = 0; j < K; ++j) x[j] = (float)(rng_normal(r) * 2.0);
    tsr_qact(TSR_QA_A32, x, K, tsr_sim_spm(s) + (size_t)line * 64u);
    free(x);
}

static uint32_t op_err(tsr_sim * s, tsr_insn i, uint32_t * info) {
    uint32_t inf = 0;
    const uint32_t e = tsr_sim_op(s, i.w, &inf);
    if (info) *info = inf;
    return e;
}
static uint32_t check_only(tsr_sim * s, tsr_insn i, uint32_t * info, tsr_extents * ext) {
    const tsr_op_handler * h = tsr_sim_op_handler((uint8_t)(i.w[0] & 0xFFu));
    if (ext) memset(ext, 0, sizeof *ext);
    uint32_t inf = 0;
    const uint32_t e = h ? h->check(s, i.w, ext, &inf) : TSR_ERR_ILLEGAL_OP;
    if (info) *info = inf;
    return e;
}
static uint64_t cyc(tsr_sim * s, tsr_insn i) {
    return tsr_sim_op_handler((uint8_t)(i.w[0] & 0xFFu))->cycles(s, i.w);
}
static tsr_att_shape shape_of(const att_f * f) {
    tsr_att_shape sh = { 64u * f->E64, 64u << f->dcode, f->G, f->hg0, f->GS, f->Tq };
    return sh;
}

// ---------------------------------------------------------------------------
// KVST: K and V planes, partial and full meta beats, E across 1024, explicit source stride,
// KVST_RMW / CH_RD_BEATS, the stored rows equal the QA vectors.
// ---------------------------------------------------------------------------
static void test_kvst(void) {
    rng_t r = { 0xA7001u };
    typedef struct { uint32_t kv, E, n_rows, c0, ss_extra, LG; } kcase;
    static const kcase cs[] = {
        { TSR_KV_K, 64, 1, 0, 0, 4 },      { TSR_KV_K, 128, 5, 7, 0, 4 },   { TSR_KV_K, 512, 3, 5, 2, 4 },
        { TSR_KV_K, 1024, 16, 16, 0, 2 },  { TSR_KV_K, 1536, 33, 250, 1, 8 }, { TSR_KV_K, 512, 64, 0, 0, 1 },
        { TSR_KV_V, 64, 1, 0, 0, 4 },      { TSR_KV_V, 128, 16, 32, 0, 4 },  { TSR_KV_V, 256, 20, 9, 3, 4 },
        { TSR_KV_V, 1024, 64, 100, 0, 3 }, { TSR_KV_V, 512, 7, 505, 0, 8 },
    };
    const uint32_t B = 5000, SRC = 1000;
    for (size_t k = 0; k < sizeof(cs) / sizeof(cs[0]); ++k) {
        const kcase * c = &cs[k];
        tsr_sim * s = mk_sim(CAPS_AT, 8);
        const tsr_kv_geom g = { c->E, NCH, c->LG };
        fill_plane(&r, s, c->kv, B, c->E, c->LG, 16u * NCH * c->LG);
        const uint32_t ql = tsr_qa_lines(TSR_QA_A32, c->E), ss = c->ss_extra ? ql + c->ss_extra : 0u, ssp = ss ? ss : ql;
        for (uint32_t i = 0; i < c->n_rows; ++i) put_qa(&r, s, SRC + i * ssp, c->E);
        // expected planes: copies of the channel images, the reference applied to them
        const size_t pbytes = (size_t)tsr_kv_plane_beats(&g) * 64u;
        uint8_t * exp[NCH], * cur[NCH];
        planes_of(s, B, cur);
        for (uint32_t ch = 0; ch < NCH; ++ch) {
            exp[ch] = (uint8_t *)malloc(pbytes);
            memcpy(exp[ch], cur[ch], pbytes);
        }
        uint8_t * spm0 = spm_copy(s);
        tsr_kvst_stats st;
        tsr_ref_kvst(c->kv, &g, exp, c->c0, c->n_rows, spm0 + (size_t)SRC * 64u, ss, &st);
        char what[64];
        snprintf(what, sizeof what, "KVST case %zu", k);
        uint32_t info;
        CHECK(op_err(s, e_kvst(c->kv, c->n_rows, SRC, B, c->E / 64u, c->LG, c->c0, ss), &info) == 0, "%s runs", what);
        for (uint32_t ch = 0; ch < NCH; ++ch) CHECK(memcmp(exp[ch], cur[ch], pbytes) == 0, "%s: channel %u plane", what, ch);
        CHECK(memcmp(spm0, tsr_sim_spm(s), SPM_BYTES) == 0, "%s: SPM unchanged", what);
        // the rows read back from the plane are the QA vectors (fp16 of D, the quants)
        tsr_block_q8_0 * row = (tsr_block_q8_0 *)malloc((c->E / 32u) * sizeof(tsr_block_q8_0));
        for (uint32_t i = 0; i < c->n_rows; ++i) {
            tsr_kv_unpack_row(c->kv, &g, (const uint8_t * const *)cur, c->c0 + i, row);
            const uint8_t * qa = spm0 + (size_t)(SRC + i * ssp) * 64u;
            const float * d = tsr_qa_d(TSR_QA_A32, c->E, qa);
            int ok = 1;
            for (uint32_t b = 0; b < c->E / 32u; ++b) {
                ok &= row[b].d == tsr_f32_to_fp16(d[b]) && memcmp(row[b].qs, tsr_qa_qs(TSR_QA_A32, c->E, qa) + 32u * b, 32) == 0;
            }
            CHECK(ok, "%s: cell %u holds QA row %u", what, c->c0 + i, i);
        }
        free(row);
        uint64_t rmw = 0;
        for (uint32_t ch = 0; ch < NCH; ++ch) {
            rmw += st.meta_rmw[ch];
            CHECK(tsr_sim_perf(s, TSR_PERF_CH_RD_BEATS(ch)) == st.meta_rmw[ch], "%s: CH_RD_BEATS[%u] = RMW reads", what, ch);
        }
        CHECK(tsr_sim_perf(s, TSR_PERF_KVST_RMW(NCH)) == rmw, "%s: KVST_RMW %llu", what, (unsigned long long)rmw);
        if (c->kv == TSR_KV_K && c->E == 1024u) CHECK(rmw == 0u, "%s: E = 1024 rows fill whole chunks: no RMW", what);
        if (c->kv == TSR_KV_K && c->E == 512u)              // two cells per K chunk: RMW iff a run end is odd
            CHECK((rmw > 0u) == (((c->c0 | (c->c0 + c->n_rows)) & 1u) != 0u), "%s: E = 512 K chunks: RMW %llu", what,
                  (unsigned long long)rmw);
        for (uint32_t ch = 0; ch < NCH; ++ch) free(exp[ch]);
        free(spm0);
        tsr_sim_destroy(s);
    }
}

// Capture (SW §5.5): KVST reports the partial meta beats it reads (init) and every beat it writes
// (expect); ATTQK / ATTPV report the streamed beats (init). The sets are derived here from the
// address functions of AT §4.3 / §4.4.
typedef struct { uint32_t ch; uint64_t beat; } cbeat;
static int cbeat_cmp(const void * a, const void * b) {
    const cbeat * x = (const cbeat *)a, * y = (const cbeat *)b;
    if (x->ch != y->ch) return x->ch < y->ch ? -1 : 1;
    return x->beat < y->beat ? -1 : x->beat > y->beat;
}
static size_t uniq(cbeat * v, size_t n) {
    qsort(v, n, sizeof *v, cbeat_cmp);
    size_t m = 0;
    for (size_t i = 0; i < n; ++i) {
        if (m == 0 || cbeat_cmp(&v[m - 1], &v[i]) != 0) v[m++] = v[i];
    }
    return m;
}
static size_t region_beats(const tsr_region * rg, size_t n, cbeat * out) {
    size_t k = 0;
    for (size_t i = 0; i < n; ++i) {
        for (uint64_t b = 0; b < rg[i].size / 64u; ++b) out[k++] = (cbeat){ rg[i].ch, rg[i].byte_off / 64u + b };
    }
    return k;
}
static int same_set(cbeat * a, size_t na, cbeat * b, size_t nb) {
    na = uniq(a, na);
    nb = uniq(b, nb);
    if (na != nb) return 0;
    for (size_t i = 0; i < na; ++i) {
        if (cbeat_cmp(&a[i], &b[i]) != 0) return 0;
    }
    return 1;
}

static void test_capture(void) {
    rng_t r = { 0xA7002u };
    const uint32_t B = 2000, SRC = 1000;
    for (uint32_t kv = 0; kv < 2; ++kv) {
        tsr_sim * s = mk_sim(CAPS_AT, 8);
        const uint32_t E = 512, LG = 4, c0 = 5, n_rows = 30;    // groups 0 and 2 partial, group 1 full
        const tsr_kv_geom g = { E, NCH, LG };
        fill_plane(&r, s, kv, B, E, LG, 16u * NCH * LG);
        const uint32_t ql = tsr_qa_lines(TSR_QA_A32, E);
        for (uint32_t i = 0; i < n_rows; ++i) put_qa(&r, s, SRC + i * ql, E);
        tsr_sim_capture_begin(s);
        CHECK(op_err(s, e_kvst(kv, n_rows, SRC, B, E / 64u, LG, c0, 0), NULL) == 0, "capture KVST runs");
        tsr_capture cap;
        tsr_sim_capture_end(s, &cap);
        // written beats and per-meta-beat slot counts from the address functions
        const size_t nmax = (size_t)n_rows * (E / 32u) * 2u;
        cbeat * wr = (cbeat *)malloc(nmax * sizeof(cbeat)), * meta = (cbeat *)malloc(nmax * sizeof(cbeat));
        size_t nw = 0, nm = 0;
        for (uint32_t i = 0; i < n_rows; ++i) {
            for (uint32_t b = 0; b < E / 32u; ++b) {
                tsr_kv_loc loc;
                if (kv == TSR_KV_K) tsr_kv_addr_k(&g, c0 + i, b, &loc);
                else tsr_kv_addr_v(&g, c0 + i, b, &loc);
                wr[nw++] = (cbeat){ loc.ch, B + loc.meta / 64u };
                wr[nw++] = (cbeat){ loc.ch, B + loc.quant / 64u };
                meta[nm++] = (cbeat){ loc.ch, B + loc.meta / 64u };
            }
        }
        qsort(meta, nm, sizeof *meta, cbeat_cmp);
        cbeat * part = (cbeat *)malloc(nm * sizeof(cbeat));
        size_t np = 0;
        for (size_t i = 0; i < nm;) {                       // a meta beat holds 32 fp16 slots
            size_t j = i;
            while (j < nm && cbeat_cmp(&meta[j], &meta[i]) == 0) ++j;
            if (j - i < 32u) part[np++] = meta[i];
            i = j;
        }
        cbeat * got = (cbeat *)malloc((nmax + 4096u) * sizeof(cbeat));
        size_t ng = region_beats(cap.expect, cap.n_expect, got);
        CHECK(same_set(got, ng, wr, nw), "capture KVST %s: expect = the written beats", kv ? "V" : "K");
        ng = region_beats(cap.init, cap.n_init, got);
        CHECK(np > 0 && same_set(got, ng, part, np), "capture KVST %s: init = the partial meta beats (%zu)", kv ? "V" : "K", np);
        tsr_capture_free(&cap);
        free(wr); free(meta); free(part); free(got);
        tsr_sim_destroy(s);
    }
    // ATTQK / ATTPV: whole local groups of K, slice regions of V
    {
        tsr_sim * s = mk_sim(CAPS_AT, 8);
        const uint32_t E = 128, LG = 8, u0 = 3, n = 13, GB = 17u * E / 64u;
        att_f q = { 0, TSR_ATT_MAXOUT, 4, 0, E / 64u, 0, 2, 4, 0, 100, 1000, B, u0, n, lohi(0, 16u * n), 3000 };
        put_qa(&r, s, 1000, 4u * E);
        tsr_sim_capture_begin(s);
        CHECK(op_err(s, e_att(TSR_OP_ATTQK, &q), NULL) == 0, "capture ATTQK runs");
        att_f p = { 0, TSR_ATT_LSUM, 4, 0, E / 64u, 0, 2, 4, 0, 4000, 100, 9000, u0, n, LG, 0 };
        CHECK(op_err(s, e_att(TSR_OP_ATTPV, &p), NULL) == 0, "capture ATTPV runs");
        tsr_capture cap;
        tsr_sim_capture_end(s, &cap);
        cbeat * exp = (cbeat *)malloc(8192 * sizeof(cbeat)), * got = (cbeat *)malloc(8192 * sizeof(cbeat));
        size_t ne = 0;
        for (uint32_t u = u0; u < u0 + n; ++u) {
            for (uint32_t b = 0; b < GB; ++b) exp[ne++] = (cbeat){ u % NCH, B + (u / NCH) * GB + b };
            for (uint32_t sl = 0; sl < E / 64u; ++sl)
                for (uint32_t b = 0; b < 17u; ++b) exp[ne++] = (cbeat){ u % NCH, 9000u + (sl * LG + u / NCH) * 17u + b };
        }
        const size_t ng = region_beats(cap.init, cap.n_init, got);
        CHECK(same_set(got, ng, exp, ne) && cap.n_expect == 0, "capture ATTQK + ATTPV: init = the streamed beats");
        tsr_capture_free(&cap);
        free(exp); free(got);
        tsr_sim_destroy(s);
    }
}

// ---------------------------------------------------------------------------
// ATTQK and ATTPV against the references
// ---------------------------------------------------------------------------
typedef struct { uint32_t E, D, G, hg0, GS, Tq, u0, n, flags, c_lo, c_hi, rs_extra, cs_extra; int specials; } acase;

static void test_attqk_attpv(void) {
    rng_t r = { 0xA7003u };
    static const acase cs[] = {
        { 128, 64, 4, 0, 4, 2, 3, 13, TSR_ATT_CAUSAL | TSR_ATT_SLIDE | TSR_ATT_MAXOUT, 5, 200, 0, 0, 0 },
        { 256, 128, 5, 0, 5, 1, 0, 40, 0, 0, 640, 3, 5, 0 },
        { 512, 512, 16, 8, 8, 1, 1, 10, TSR_ATT_MAXOUT, 0, 160, 0, 0, 0 },          // Gemma 4 global, 2nd head subset
        { 64, 64, 1, 0, 1, 8, 0, 8, TSR_ATT_CAUSAL | TSR_ATT_MAXOUT, 0, 128, 0, 0, 1 },   // prefill tile, specials
        { 1024, 128, 5, 2, 1, 1, 6, 6, TSR_ATT_MAXOUT, 17, 90, 0, 0, 0 },          // Qwen2.5-32B shape, subset
        { 128, 64, 2, 1, 1, 4, 2, 5, TSR_ATT_SLIDE | TSR_ATT_MAXOUT, 40, 80, 1, 2, 1 },
    };
    const uint32_t BK = 1000, BV = 300000, X = 100, S = 4000, MK = 3900, P = 20000;
    for (size_t k = 0; k < sizeof(cs) / sizeof(cs[0]); ++k) {
        const acase * c = &cs[k];
        tsr_sim * s = mk_sim(CAPS_AT, 8);
        const uint32_t LG = (c->u0 + c->n - 1u) / NCH + 2u, ncell = 16u * NCH * LG;
        fill_plane(&r, s, TSR_KV_K, BK, c->E, LG, ncell);
        fill_plane(&r, s, TSR_KV_V, BV, c->E, LG, ncell);
        const uint32_t n_head = c->G * (c->E / c->D), R = c->Tq * n_head;
        for (uint32_t q = 0; q < c->Tq; ++q) put_qa(&r, s, X + q * tsr_qa_lines(TSR_QA_A32, n_head * c->D), n_head * c->D);
        const uint32_t rs = c->rs_extra ? c->n + c->rs_extra : 0u, rsp = rs ? rs : c->n;
        rand_lines(&r, s, S, R * rsp, 1);                  // rows of other passes must stay as they are
        rand_lines(&r, s, MK, (R + 15u) / 16u, 1);
        uint8_t * kp[NCH], * vp[NCH];
        planes_of(s, BK, kp);
        planes_of(s, BV, vp);
        att_f q = { 0, c->flags, c->G, c->hg0, c->E / 64u, dcode_of(c->D), c->Tq, c->GS, rs, S, X, BK, c->u0, c->n,
                    lohi(c->c_lo, c->c_hi), (c->flags & TSR_ATT_MAXOUT) ? MK : 0u };
        const tsr_att_shape sh = shape_of(&q);
        uint8_t * exp = spm_copy(s);
        tsr_ref_attqk(&sh, NCH, (const uint8_t * const *)kp, c->u0, c->n, c->c_lo, c->c_hi, c->flags, exp + (size_t)X * 64u,
                      F(exp, S), rs, (c->flags & TSR_ATT_MAXOUT) ? F(exp, MK) : NULL);
        char what[48];
        snprintf(what, sizeof what, "ATTQK case %zu", k);
        uint32_t info;
        CHECK(op_err(s, e_att(TSR_OP_ATTQK, &q), &info) == 0, "%s runs (info %u)", what, info);
        spm_same(s, exp, what);
        // ATTPV over p = the scores turned into weights (EXP of the shifted score), plus specials
        const uint32_t ps = rs;
        for (uint32_t row = 0; row < R; ++row) {
            float * pr = F(tsr_sim_spm(s), S + row * rsp);
            for (uint32_t j = 0; j < 16u * c->n; ++j) {
                float v = (rng_below(&r, 8) == 0) ? 0.0f : (float)rng_unit(&r);
                if (c->specials && rng_below(&r, 400) == 0) v = u2f(rng_below(&r, 2) ? 0x7F800000u : 0x00000005u);
                pr[j] = v;
            }
        }
        const uint32_t cs0 = R * c->D / 16u, cstr = c->cs_extra ? cs0 + c->cs_extra : 0u;
        rand_lines(&r, s, P, NCH * (cstr ? cstr : cs0) + NCH * ((R + 15u) / 16u), 1);
        const uint32_t pv_flags = (k & 1u) ? 0u : TSR_ATT_LSUM;
        att_f p = { 0, pv_flags, c->G, c->hg0, c->E / 64u, dcode_of(c->D), c->Tq, c->GS, ps, P, S, BV, c->u0, c->n, LG, cstr };
        free(exp);
        exp = spm_copy(s);
        tsr_ref_attpv(&sh, NCH, LG, (const uint8_t * const *)vp, c->u0, c->n, pv_flags, F(exp, S), ps, F(exp, P), cstr);
        snprintf(what, sizeof what, "ATTPV case %zu", k);
        CHECK(op_err(s, e_att(TSR_OP_ATTPV, &p), &info) == 0, "%s runs (info %u)", what, info);
        spm_same(s, exp, what);
        free(exp);
        tsr_sim_destroy(s);
    }
}

// ---------------------------------------------------------------------------
// ASMAX, AMASK, AMERGE, ANORM against the references
// ---------------------------------------------------------------------------
static void test_vector_ops(void) {
    rng_t r = { 0xA7004u };
    tsr_sim * s = mk_sim(CAPS_AT, 8);
    rand_lines(&r, s, 0, SPM_LINES, 1);
    const uint32_t scales[] = { 0x3E000000u, 0x3F800000u, 0x3DB504F3u };     // 1/8, 1, 1/sqrt(128)
    for (int k = 0; k < 6; ++k) {
        const uint32_t R = k == 0 ? 1u : k == 1 ? 16u : k == 2 ? 40u : k == 3 ? 7u : k == 4 ? 33u : 1024u;
        const uint32_t nl = k == 5 ? 1u : 1u + rng_below(&r, 12), st = (k & 1) ? nl + 2u : 0u, stp = st ? st : nl;
        const uint32_t sc = scales[k % 3];
        const uint32_t SRC = 100, DST = (k & 2) ? SRC : 20000, MV = 60000, MASK = 40000, MD = 61000;
        char what[48];
        // ASMAX (in place for k & 2)
        rand_lines(&r, s, SRC, R * stp, 1);
        rand_lines(&r, s, MV, (R + 15u) / 16u, 0);
        uint8_t * exp = spm_copy(s);
        tsr_ref_asmax(F(exp, DST), F(exp, SRC), F(exp, MV), nl, st, R, u2f(sc));
        snprintf(what, sizeof what, "ASMAX case %d", k);
        CHECK(op_err(s, e_asmax(DST, SRC, MV, nl, st, R, sc), NULL) == 0, "%s runs", what);
        spm_same(s, exp, what);
        free(exp);
        // AMASK: mask rows per query (rpq heads), -inf cells, finite values
        const uint32_t rpq = 1u + rng_below(&r, 8), ms = nl + (uint32_t)(k & 1);
        rand_lines(&r, s, SRC, R * stp, 0);
        for (uint32_t q = 0; q < (R + rpq - 1u) / rpq; ++q) {
            float * mr = F(tsr_sim_spm(s), MASK + q * ms);
            for (uint32_t j = 0; j < 16u * nl; ++j) {
                const uint32_t t = rng_below(&r, 4);
                mr[j] = t == 0 ? u2f(0xFF800000u) : t == 1 ? -0.0f : t == 2 ? 0.0f : (float)(rng_normal(&r) * 2.0);
            }
        }
        if (k == 3) for (uint32_t j = 0; j < 16u * nl; ++j) F(tsr_sim_spm(s), MASK)[j] = u2f(0xFF800000u);   // a fully masked row
        exp = spm_copy(s);
        tsr_ref_amask(F(exp, DST), F(exp, SRC), F(exp, MASK), nl, st, R, rpq, ms, F(exp, MD));
        snprintf(what, sizeof what, "AMASK case %d", k);
        CHECK(op_err(s, e_amask(DST, SRC, MASK, nl, st, R, rpq, ms, MD), NULL) == 0, "%s runs", what);
        spm_same(s, exp, what);
        free(exp);
        if (k == 3) CHECK(F(tsr_sim_spm(s), MD)[0] == u2f(TSR_M_FLOOR), "AMASK fully masked row: M_FLOOR");
        // AMERGE: INIT then two merges; ANORM (D cycles through 64 .. 512, R <= 128 for the buffers)
        const uint32_t D = 64u << (k & 3), Rm = R > 64u ? 64u : R, rl = Rm * D / 16u;
        const uint32_t cs = (k & 1) ? rl + 3u : 0u, csp = cs ? cs : rl;
        const uint32_t ORUN = 2000, PARTS = 10000, MKV = 1900, ST = 1950, OUT = 30000;
        for (int step = 0; step < 3; ++step) {
            rand_lines(&r, s, PARTS, NCH * csp + NCH * ((Rm + 15u) / 16u), 0);
            rand_lines(&r, s, MKV, (Rm + 15u) / 16u, 0);
            if (step == 0) {
                rand_lines(&r, s, ORUN, rl, 1);
                rand_lines(&r, s, ST, 2u * ((Rm + 15u) / 16u), 1);
            }
            if (step == 2) F(tsr_sim_spm(s), MKV)[0] = u2f(TSR_M_FLOOR);   // a chunk without visible cells
            exp = spm_copy(s);
            tsr_ref_amerge(F(exp, ORUN), F(exp, PARTS), F(exp, MKV), F(exp, ST), Rm, D, u2f(sc), NCH, cs, step == 0 ? TSR_ATT_INIT : 0u);
            snprintf(what, sizeof what, "AMERGE case %d step %d", k, step);
            CHECK(op_err(s, e_amerge(step == 0, ORUN, PARTS, MKV, ST, Rm, dcode_of(D), sc, cs), NULL) == 0, "%s runs", what);
            spm_same(s, exp, what);
            free(exp);
        }
        if (k == 4) F(tsr_sim_spm(s), ST + 3u)[1] = 0.0f;               // an empty row: l = 0
        exp = spm_copy(s);
        tsr_ref_anorm(F(exp, OUT), F(exp, ORUN), F(exp, ST + (Rm + 15u) / 16u), Rm, D);
        snprintf(what, sizeof what, "ANORM case %d", k);
        CHECK(op_err(s, e_anorm(OUT, ORUN, ST + (Rm + 15u) / 16u, Rm, dcode_of(D)), NULL) == 0, "%s runs", what);
        spm_same(s, exp, what);
        free(exp);
    }
    tsr_sim_destroy(s);
}

// ---------------------------------------------------------------------------
// Checks
// ---------------------------------------------------------------------------
static void expect_err(tsr_sim * s, tsr_insn i, uint32_t code, uint32_t info, const char * what) {
    uint8_t * before = spm_copy(s);
    uint32_t inf = 0xDEADBEEFu;
    const uint32_t e = tsr_sim_op(s, i.w, &inf);
    CHECK(e == code && inf == info, "%s: code %u info 0x%x (want %u 0x%x)", what, e, inf, code, info);
    CHECK(memcmp(before, tsr_sim_spm(s), SPM_BYTES) == 0, "%s: a failing op changes nothing", what);
    free(before);
}
static void expect_ok(tsr_sim * s, tsr_insn i, const char * what) {
    uint32_t info = 0;
    const uint32_t e = check_only(s, i, &info, NULL);
    CHECK(e == TSR_ERR_NONE, "%s: passes the checker (got %u info 0x%x)", what, e, info);
}
static tsr_insn with(tsr_insn i, unsigned word, uint32_t or_bits) { i.w[word] |= or_bits; return i; }

static void test_checks(void) {
    tsr_sim * s = mk_sim(CAPS_AT, 8);
    const uint32_t IL = TSR_ERR_ILLEGAL_OP, OP = TSR_ERR_OPERAND, SR = TSR_ERR_SPM_RANGE, DR = TSR_ERR_DRAM_RANGE;
    const uint32_t KR = TSR_ERR_K_RANGE, TR = TSR_ERR_T_RANGE, BF = TSR_ERR_BAD_FMT, N = SPM_LINES;

    // KVST (E = 128: QA_LINES 3, GB 34; C = 4, LG 8: 512 cells)
    const uint32_t ql = tsr_qa_lines(TSR_QA_A32, 128);
    const tsr_insn kv = e_kvst(TSR_KV_K, 3, 100, 1000, 2, 8, 5, 0);
    expect_ok(s, kv, "KVST base");
    expect_err(s, with(kv, 0, 1u << 22), IL, 0, "KVST x16[6]");
    expect_err(s, with(kv, 0, 1u << 12), IL, 0, "KVST flags");
    expect_err(s, e_kvst(2, 3, 100, 1000, 2, 8, 5, 0), IL, 0, "KVST sub 2");
    expect_err(s, with(kv, 3, 1u << 9), IL, 0, "KVST w3[9]");
    expect_err(s, with(kv, 6, 1u), IL, 0, "KVST w6");
    expect_err(s, with(kv, 7, 1u << 31), IL, 0, "KVST w7");
    expect_err(s, e_kvst(TSR_KV_K, 3, 100, 1000, 0, 8, 5, 0), KR, 0, "KVST E/64 = 0");
    expect_err(s, e_kvst(TSR_KV_K, 3, 100, 1000, 0, 0, 5, 0), KR, 0, "KVST K_RANGE before OPERAND");
    expect_err(s, with(e_kvst(TSR_KV_K, 3, 100, 1000, 0, 0, 5, 0), 0, 1u << 13), IL, 0, "KVST ILLEGAL_OP before K_RANGE");
    expect_err(s, e_kvst(TSR_KV_K, 3, 100, 1000, 2, 0, 5, 0), OP, 0, "KVST LG = 0");
    expect_err(s, e_kvst(TSR_KV_K, 3, 100, 1000, 2, 8, 510, 0), OP, 510, "KVST c0 + n_rows > 16·C·LG");
    expect_ok(s, e_kvst(TSR_KV_V, 3, 100, 1000, 2, 8, 509, 0), "KVST last cells");
    expect_ok(s, e_kvst(TSR_KV_K, 64, 100, 1000, 2, 8, 0, 0), "KVST 64 rows");
    expect_ok(s, e_kvst(TSR_KV_K, 3, N - 3u * ql, 1000, 2, 8, 5, 0), "KVST src ends at the limit");
    expect_err(s, e_kvst(TSR_KV_K, 3, N - 3u * ql + 1u, 1000, 2, 8, 5, 0), SR, N - 3u * ql + 1u, "KVST src range");
    expect_ok(s, e_kvst(TSR_KV_K, 3, N - 19, 1000, 2, 8, 5, 8), "KVST src with stride 8 ends at the limit");
    expect_err(s, e_kvst(TSR_KV_K, 3, N - 18, 1000, 2, 8, 5, 8), SR, N - 18, "KVST src range, stride 8");
    expect_ok(s, e_kvst(TSR_KV_V, 3, 100, LIMIT - 8u * 34u, 2, 8, 5, 0), "KVST plane ends at the limit");
    expect_err(s, e_kvst(TSR_KV_V, 3, 100, LIMIT - 8u * 34u + 1u, 2, 8, 5, 0), DR, LIMIT - 8u * 34u + 1u, "KVST plane range");
    expect_err(s, e_kvst(TSR_KV_V, 3, N, LIMIT, 2, 8, 5, 0), SR, N, "KVST SPM_RANGE before DRAM_RANGE");

    // ATTQK (E 128, D 64, G 4, GS 4, T_q 2: n_head 8, R 16, 16 pass rows, last row 15)
    const uint32_t xl = 2u * tsr_qa_lines(TSR_QA_A32, 4u * 128u);
    const att_f q0 = { 0, TSR_ATT_MAXOUT, 4, 0, 2, 0, 2, 4, 0, 100, 1000, 1000, 3, 13, lohi(0, 208), 2000 };
    att_f q;
    expect_ok(s, e_att(TSR_OP_ATTQK, &q0), "ATTQK base");
    expect_err(s, with(e_att(TSR_OP_ATTQK, &q0), 0, 1u << 15), IL, 0, "ATTQK w0[15]");
    expect_err(s, with(e_att(TSR_OP_ATTQK, &q0), 0, 1u << 28), IL, 0, "ATTQK x16[12]");
    q = q0; q.flags = 0; expect_err(s, e_att(TSR_OP_ATTQK, &q), IL, 0, "ATTQK mdst without MAXOUT");
    q = q0; q.flags = 0; q.w7 = 0; expect_ok(s, e_att(TSR_OP_ATTQK, &q), "ATTQK without MAXOUT");
    q = q0; q.fmt = 1; expect_err(s, e_att(TSR_OP_ATTQK, &q), BF, 0, "ATTQK fmt 1");
    q = q0; q.fmt = 1; q.E64 = 0; expect_err(s, e_att(TSR_OP_ATTQK, &q), BF, 0, "ATTQK BAD_FMT before K_RANGE");
    q = q0; q.E64 = 0; expect_err(s, e_att(TSR_OP_ATTQK, &q), KR, 0, "ATTQK E/64 = 0");
    q = q0; q.G = 64; q.E64 = 16; q.dcode = 1; q.GS = 1; q.Tq = 1; expect_err(s, e_att(TSR_OP_ATTQK, &q), KR, 0, "ATTQK G·E > K_MAX");
    q = q0; q.G = 8; q.GS = 8; expect_err(s, e_att(TSR_OP_ATTQK, &q), TR, 0, "ATTQK T_q·GS > T_MAX");
    q = q0; q.G = 8; q.GS = 8; q.E64 = 0; expect_err(s, e_att(TSR_OP_ATTQK, &q), KR, 0, "ATTQK K_RANGE before T_RANGE");
    q = q0; q.E64 = 3; q.dcode = 1; expect_err(s, e_att(TSR_OP_ATTQK, &q), OP, 192, "ATTQK E % D");
    q = q0; q.hg0 = 2; expect_err(s, e_att(TSR_OP_ATTQK, &q), OP, 6, "ATTQK hg0 + GS > G");
    q = q0; q.n = 0; q.w6 = 0; expect_err(s, e_att(TSR_OP_ATTQK, &q), OP, 0, "ATTQK n_groups 0");
    q = q0; q.n = 4096; expect_err(s, e_att(TSR_OP_ATTQK, &q), OP, 4096, "ATTQK n_groups 4096");
    q = q0; q.w6 = lohi(0, 209); expect_err(s, e_att(TSR_OP_ATTQK, &q), OP, 209, "ATTQK c_hi > 16·n_groups");
    q = q0; q.w6 = lohi(10, 5); expect_err(s, e_att(TSR_OP_ATTQK, &q), OP, 5, "ATTQK c_lo > c_hi");
    q = q0; q.E64 = 32; q.dcode = 0; q.G = 4; q.GS = 4; q.Tq = 2; q.w6 = lohi(0, 208);
    expect_err(s, e_att(TSR_OP_ATTQK, &q), OP, 256, "ATTQK pass rows > 128");
    q = q0; q.rs = 5; expect_err(s, e_att(TSR_OP_ATTQK, &q), OP, 5, "ATTQK 0 < rstride < n_groups");
    q = q0; q.hg0 = 2; q.rs = 5; expect_err(s, e_att(TSR_OP_ATTQK, &q), OP, 6, "ATTQK OPERAND order");
    q = q0; q.rs = 13; expect_ok(s, e_att(TSR_OP_ATTQK, &q), "ATTQK rstride = n_groups");
    q = q0; q.dst = N - 208; expect_ok(s, e_att(TSR_OP_ATTQK, &q), "ATTQK dst ends at the limit");
    q = q0; q.dst = N - 207; expect_err(s, e_att(TSR_OP_ATTQK, &q), SR, N - 207, "ATTQK dst range");
    q = q0; q.x = N - xl + 1u; expect_err(s, e_att(TSR_OP_ATTQK, &q), SR, N - xl + 1u, "ATTQK x range");
    q = q0; q.w7 = N; expect_err(s, e_att(TSR_OP_ATTQK, &q), SR, N, "ATTQK mdst range");
    q = q0; q.B = LIMIT - 135u; expect_err(s, e_att(TSR_OP_ATTQK, &q), DR, LIMIT - 135u, "ATTQK plane range");
    q = q0; q.B = LIMIT - 136u; expect_ok(s, e_att(TSR_OP_ATTQK, &q), "ATTQK plane ends at the limit");
    q = q0; q.B = LIMIT; q.w7 = N; expect_err(s, e_att(TSR_OP_ATTQK, &q), SR, N, "ATTQK SPM_RANGE before DRAM_RANGE");

    // ATTPV (same shape, LSUM, LG 8; cs 64; dst 4·64 + 4 = 260; p 15·13 + 13 = 208; plane (8 + 4)·17)
    const att_f p0 = { 0, TSR_ATT_LSUM, 4, 0, 2, 0, 2, 4, 0, 3000, 100, 5000, 3, 13, 8, 0 };
    att_f p;
    expect_ok(s, e_att(TSR_OP_ATTPV, &p0), "ATTPV base");
    expect_err(s, with(e_att(TSR_OP_ATTPV, &p0), 0, 1u << 13), IL, 0, "ATTPV w0[13]");
    expect_err(s, with(e_att(TSR_OP_ATTPV, &p0), 6, 1u << 16), IL, 0, "ATTPV w6[16]");
    p = p0; p.fmt = 3; expect_err(s, e_att(TSR_OP_ATTPV, &p), BF, 0, "ATTPV fmt 3");
    p = p0; p.E64 = 0; expect_err(s, e_att(TSR_OP_ATTPV, &p), KR, 0, "ATTPV E/64 = 0");
    p = p0; p.G = 8; p.GS = 8; expect_err(s, e_att(TSR_OP_ATTPV, &p), TR, 0, "ATTPV T_RANGE");
    p = p0; p.E64 = 3; p.dcode = 1; expect_err(s, e_att(TSR_OP_ATTPV, &p), OP, 192, "ATTPV E % D");
    p = p0; p.hg0 = 1; expect_err(s, e_att(TSR_OP_ATTPV, &p), OP, 5, "ATTPV hg0 + GS > G");
    p = p0; p.n = 0; expect_err(s, e_att(TSR_OP_ATTPV, &p), OP, 0, "ATTPV n_groups 0");
    p = p0; p.E64 = 32; p.w6 = 200; expect_err(s, e_att(TSR_OP_ATTPV, &p), OP, 256, "ATTPV pass rows > 128");
    p = p0; p.w6 = 3; expect_err(s, e_att(TSR_OP_ATTPV, &p), OP, 15, "ATTPV ℓ_max >= LG");
    p = p0; p.w6 = 0; expect_err(s, e_att(TSR_OP_ATTPV, &p), OP, 15, "ATTPV LG = 0");
    p = p0; p.n = 1169; p.w6 = 400; expect_err(s, e_att(TSR_OP_ATTPV, &p), OP, 1169, "ATTPV p above PCAP");
    p = p0; p.n = 1168; p.w6 = 400; p.x = 20000; expect_ok(s, e_att(TSR_OP_ATTPV, &p), "ATTPV p at PCAP");
    p = p0; p.rs = 12; expect_err(s, e_att(TSR_OP_ATTPV, &p), OP, 12, "ATTPV 0 < pstride < n_groups");
    p = p0; p.w6 = 3; p.rs = 12; expect_err(s, e_att(TSR_OP_ATTPV, &p), OP, 15, "ATTPV OPERAND order (ℓ_max before pstride)");
    p = p0; p.dst = N - 260; expect_ok(s, e_att(TSR_OP_ATTPV, &p), "ATTPV dst with LSUM ends at the limit");
    p = p0; p.dst = N - 259; expect_err(s, e_att(TSR_OP_ATTPV, &p), SR, N - 259, "ATTPV dst range (LSUM)");
    p = p0; p.flags = 0; p.dst = N - 256; expect_ok(s, e_att(TSR_OP_ATTPV, &p), "ATTPV dst without LSUM");
    p = p0; p.w7 = 70; p.dst = N - 4u * 70u - 4u; expect_ok(s, e_att(TSR_OP_ATTPV, &p), "ATTPV explicit cstride");
    p = p0; p.w7 = 70; p.dst = N - 4u * 70u - 3u; expect_err(s, e_att(TSR_OP_ATTPV, &p), SR, N - 283u, "ATTPV dst range, cstride");
    p = p0; p.x = N - 207; expect_err(s, e_att(TSR_OP_ATTPV, &p), SR, N - 207, "ATTPV p range");
    p = p0; p.B = LIMIT - 204u; expect_ok(s, e_att(TSR_OP_ATTPV, &p), "ATTPV plane ends at the limit");
    p = p0; p.B = LIMIT - 203u; expect_err(s, e_att(TSR_OP_ATTPV, &p), DR, LIMIT - 203u, "ATTPV plane range");

    // ASMAX, AMASK, AMERGE, ANORM (R 16, n_lines 13: extents 208)
    const uint32_t sc = 0x3E000000u;
    const tsr_insn as = e_asmax(100, 100, 50, 13, 0, 16, sc);
    expect_ok(s, as, "ASMAX base");
    expect_err(s, with(as, 0, 1u << 8), IL, 0, "ASMAX sub");
    expect_err(s, with(as, 0, 1u << 16), IL, 0, "ASMAX x16");
    expect_err(s, with(as, 7, 1u), IL, 0, "ASMAX w7");
    expect_err(s, e_asmax(100, 100, 50, 0, 0, 16, sc), OP, 0, "ASMAX n_lines 0");
    expect_err(s, e_asmax(100, 100, 50, 4096, 0, 16, sc), OP, 4096, "ASMAX n_lines 4096");
    expect_err(s, e_asmax(100, 100, 50, 13, 0, 0, sc), OP, 0, "ASMAX R 0");
    expect_err(s, e_asmax(100, 100, 50, 13, 0, 1025, sc), OP, 1025, "ASMAX R 1025");
    static const uint32_t bad_sc[] = { 0x00000000u, 0x80000000u, 0xBF800000u, 0x7F800000u, 0x7FC00000u, 0x00000001u, 0xFF800000u };
    for (size_t k = 0; k < sizeof(bad_sc) / sizeof(bad_sc[0]); ++k) {
        char what[40];
        snprintf(what, sizeof what, "ASMAX scale %08x", bad_sc[k]);
        expect_err(s, e_asmax(100, 100, 50, 13, 0, 16, bad_sc[k]), OP, bad_sc[k], what);
    }
    expect_ok(s, e_asmax(100, 100, 50, 13, 0, 16, 0x00800000u), "ASMAX scale 2^-126 (normal)");
    expect_err(s, e_asmax(100, 100, 50, 0, 0, 1025, 0xBF800000u), OP, 0, "ASMAX n_lines before R before scale");
    expect_err(s, e_asmax(100, 100, 50, 13, 0, 1025, 0xBF800000u), OP, 1025, "ASMAX R before scale");
    expect_err(s, e_asmax(N - 207, 100, 50, 13, 0, 16, sc), SR, N - 207, "ASMAX dst range");
    expect_err(s, e_asmax(100, N - 200, 50, 13, 15, 16, sc), SR, N - 200, "ASMAX src range, stride");
    expect_err(s, e_asmax(100, 100, N, 13, 0, 16, sc), SR, N, "ASMAX mvec range");
    expect_ok(s, e_asmax(N - 208, N - 208, N - 1, 13, 0, 16, sc), "ASMAX ends at the limit");

    const tsr_insn am = e_amask(100, 100, 400, 13, 0, 16, 8, 13, 50);   // mask: (2 - 1)·13 + 13 = 26
    expect_ok(s, am, "AMASK base");
    expect_err(s, with(am, 0, 1u << 12), IL, 0, "AMASK flags");
    expect_err(s, e_amask(100, 100, 400, 0, 0, 16, 8, 13, 50), OP, 0, "AMASK n_lines 0");
    expect_err(s, e_amask(100, 100, 400, 13, 0, 1025, 8, 13, 50), OP, 1025, "AMASK R 1025");
    expect_err(s, e_amask(100, 100, 400, 13, 0, 16, 0, 13, 50), OP, 0, "AMASK rpq 0");
    expect_err(s, e_amask(100, 100, N - 25, 13, 0, 16, 8, 13, 50), SR, N - 25, "AMASK mask range");
    expect_ok(s, e_amask(100, 100, N - 26, 13, 0, 16, 8, 13, 50), "AMASK mask ends at the limit");
    expect_err(s, e_amask(100, 100, 400, 13, 0, 17, 8, 13, N - 1), SR, N - 1, "AMASK mdst range (17 rows)");
    expect_err(s, e_amask(N - 1, 100, 400, 13, 0, 16, 8, 13, N), SR, N - 1, "AMASK operands in field order");

    // AMERGE (R 16, D 64: Orun 64, parts 4·64 + 4 = 260, mk 1, st 2)
    const tsr_insn mg = e_amerge(0, 100, 200, 50, 60, 16, 0, sc, 0);
    expect_ok(s, mg, "AMERGE base");
    expect_err(s, with(mg, 0, 1u << 13), IL, 0, "AMERGE flags[13]");
    expect_err(s, with(mg, 0, 1u << 8), IL, 0, "AMERGE sub");
    expect_err(s, with(mg, 5, 1u << 18), IL, 0, "AMERGE w5[18]");
    expect_err(s, e_amerge(0, 100, 200, 50, 60, 0, 0, sc, 0), OP, 0, "AMERGE R 0");
    expect_err(s, e_amerge(1, 100, 200, 50, 60, 16, 0, 0xBF800000u, 0), OP, 0xBF800000u, "AMERGE scale");
    expect_err(s, e_amerge(0, N - 63, 200, 50, 60, 16, 0, sc, 0), SR, N - 63, "AMERGE Orun range");
    expect_err(s, e_amerge(0, 100, N - 259, 50, 60, 16, 0, sc, 0), SR, N - 259, "AMERGE parts range");
    expect_ok(s, e_amerge(0, 100, N - 260, 50, 60, 16, 0, sc, 0), "AMERGE parts end at the limit");
    expect_err(s, e_amerge(0, 100, 200, N, 60, 16, 0, sc, 0), SR, N, "AMERGE mk range");
    expect_err(s, e_amerge(0, 100, 200, 50, N - 1, 16, 0, sc, 0), SR, N - 1, "AMERGE st range");
    expect_err(s, e_amerge(0, 100, N - 200, 50, 60, 16, 0, sc, 70), SR, N - 200, "AMERGE parts range, cstride");

    // ANORM
    const tsr_insn an = e_anorm(100, 200, 50, 16, 1);                  // D 128: 128 lines
    expect_ok(s, an, "ANORM base");
    expect_err(s, with(an, 4, 1u << 18), IL, 0, "ANORM w4[18]");
    expect_err(s, with(an, 5, 1u), IL, 0, "ANORM w5");
    expect_err(s, e_anorm(100, 200, 50, 1025, 1), OP, 1025, "ANORM R 1025");
    expect_err(s, e_anorm(N - 127, 200, 50, 16, 1), SR, N - 127, "ANORM out range");
    expect_err(s, e_anorm(100, N - 127, 50, 16, 1), SR, N - 127, "ANORM Orun range");
    expect_err(s, e_anorm(100, 200, N, 16, 1), SR, N, "ANORM l_run range");
    expect_ok(s, e_anorm(N - 128, N - 128, N - 1, 16, 1), "ANORM ends at the limit");
    tsr_sim_destroy(s);

    // CAPS.ATT gates all seven ops (EX §7.5); AT's vector ops need only ATT.
    CHECK(tsr_sim_caps_implemented() & TSR_CAPS_ATT, "CAPS.ATT implemented");
    s = mk_sim(TSR_CAPS_VU, 8);
    const tsr_insn all7[] = { kv, e_att(TSR_OP_ATTQK, &q0), e_att(TSR_OP_ATTPV, &p0), as, am, mg, an };
    for (size_t k = 0; k < 7; ++k) CHECK(op_err(s, all7[k], NULL) == IL, "op %zu without CAPS.ATT: ILLEGAL_OP", k);
    tsr_sim_destroy(s);
    s = mk_sim(TSR_CAPS_ATT, 8);
    CHECK(check_only(s, as, NULL, NULL) == 0 && op_err(s, an, NULL) == 0, "ATT without VU: the vector ops run");
    CHECK(TSR_GET(tsr_sim_csr_read(s, TSR_CSR_CFG3), TSR_CFG3_PCAP16) * 16u == TSR_PCAP_U200, "CFG3 PCAP/16");
    tsr_sim_destroy(s);
}

// ---------------------------------------------------------------------------
// Extents, cycles, PERF
// ---------------------------------------------------------------------------
static int ext_is(const tsr_extents * x, uint32_t k, uint8_t kind, uint8_t wr, uint8_t ch, uint64_t lo, uint64_t hi) {
    return k < x->n && x->e[k].kind == kind && x->e[k].write == wr && x->e[k].ch == ch && x->e[k].lo == lo && x->e[k].hi == hi;
}

static void test_extents(void) {
    tsr_sim * s = mk_sim(CAPS_AT, 8);
    tsr_extents x;
    const uint8_t SP = TSR_EXT_SPM, DM = TSR_EXT_DRAM, R = TSR_EXT_READ, W = TSR_EXT_WRITE, ALL = TSR_EXT_CH_ALL;
    check_only(s, e_kvst(TSR_KV_V, 3, 100, 1000, 2, 8, 5, 4), NULL, &x);
    CHECK(x.n == 2 && ext_is(&x, 0, SP, R, 0, 100, 111) && ext_is(&x, 1, DM, W, ALL, 1000, 1272), "KVST: src, plane");
    const att_f q0 = { 0, TSR_ATT_MAXOUT, 4, 0, 2, 0, 2, 4, 0, 100, 1000, 7000, 3, 13, lohi(0, 208), 2000 };
    check_only(s, e_att(TSR_OP_ATTQK, &q0), NULL, &x);
    CHECK(x.n == 4 && ext_is(&x, 0, SP, W, 0, 100, 308) && ext_is(&x, 1, SP, R, 0, 1000, 1000 + 2u * tsr_qa_lines(TSR_QA_A32, 512)) &&
          ext_is(&x, 2, DM, R, ALL, 7000, 7136) && ext_is(&x, 3, SP, W, 0, 2000, 2001), "ATTQK: dst, x, plane, mdst");
    const att_f p0 = { 0, TSR_ATT_LSUM, 4, 0, 2, 0, 2, 4, 0, 3000, 100, 5000, 3, 13, 8, 0 };
    check_only(s, e_att(TSR_OP_ATTPV, &p0), NULL, &x);
    CHECK(x.n == 3 && ext_is(&x, 0, SP, W, 0, 3000, 3260) && ext_is(&x, 1, SP, R, 0, 100, 308) && ext_is(&x, 2, DM, R, ALL, 5000, 5204),
          "ATTPV: dst, p, plane");
    check_only(s, e_asmax(100, 100, 50, 13, 0, 16, 0x3F800000u), NULL, &x);
    CHECK(x.n == 3 && ext_is(&x, 0, SP, W, 0, 100, 308) && ext_is(&x, 1, SP, R, 0, 100, 308) && ext_is(&x, 2, SP, R, 0, 50, 51),
          "ASMAX in place: dst, src, mvec");
    check_only(s, e_amask(100, 1000, 400, 13, 0, 16, 8, 13, 50), NULL, &x);
    CHECK(x.n == 4 && ext_is(&x, 1, SP, R, 0, 1000, 1208) && ext_is(&x, 2, SP, R, 0, 400, 426) && ext_is(&x, 3, SP, W, 0, 50, 51),
          "AMASK: dst, src, mask, mdst");
    check_only(s, e_amerge(1, 100, 200, 50, 60, 16, 0, 0x3F800000u, 0), NULL, &x);
    CHECK(x.n == 4 && ext_is(&x, 0, SP, W, 0, 100, 164) && ext_is(&x, 1, SP, R, 0, 200, 460) && ext_is(&x, 2, SP, R, 0, 50, 51) &&
          ext_is(&x, 3, SP, W, 0, 60, 62), "AMERGE INIT: Orun and st written only");
    check_only(s, e_amerge(0, 100, 200, 50, 60, 16, 0, 0x3F800000u, 0), NULL, &x);
    CHECK(x.n == 6 && ext_is(&x, 0, SP, W, 0, 100, 164) && ext_is(&x, 1, SP, R, 0, 100, 164) && ext_is(&x, 4, SP, W, 0, 60, 62) &&
          ext_is(&x, 5, SP, R, 0, 60, 62), "AMERGE merge: Orun and st read-modify-write");
    check_only(s, e_anorm(100, 200, 50, 16, 1), NULL, &x);
    CHECK(x.n == 3 && ext_is(&x, 0, SP, W, 0, 100, 228) && ext_is(&x, 1, SP, R, 0, 200, 328) && ext_is(&x, 2, SP, R, 0, 50, 51),
          "ANORM: out, Orun, l_run");
    tsr_sim_destroy(s);
}

static void test_cycles(void) {
    tsr_sim * s = mk_sim(CAPS_AT, 8);
    CHECK(cyc(s, e_kvst(TSR_KV_K, 1, 100, 1000, 16, 4, 0, 0)) == 40u + 1u + 16u, "KVST K E = 1024, one row: no RMW");
    CHECK(cyc(s, e_kvst(TSR_KV_V, 1, 100, 1000, 16, 4, 0, 0)) == 40u + 16u + 16u + 16u, "KVST V one row: 16 RMW metas");
    CHECK(cyc(s, e_kvst(TSR_KV_V, 16, 100, 1000, 16, 4, 16, 0)) == 40u + 16u + 256u, "KVST V full group: no RMW");
    att_f q = { 0, TSR_ATT_MAXOUT, 4, 0, 2, 0, 2, 4, 0, 100, 1000, 7000, 0, 16, lohi(0, 256), 2000 };
    CHECK(cyc(s, e_att(TSR_OP_ATTQK, &q)) == 350u + 256u, "ATTQK D 64, 16 pass rows: gather-bound");
    att_f q2 = { 0, 0, 5, 0, 16, 1, 1, 5, 0, 100, 1000, 7000, 0, 128, lohi(0, 2048), 0 };
    CHECK(cyc(s, e_att(TSR_OP_ATTQK, &q2)) == 350u + 32u * 272u, "ATTQK Qwen2.5-32B shape: stream-bound");
    att_f q3 = q2; q3.u0 = 1; q3.n = 6;                                 // channels get 2, 2, 1, 1 groups
    CHECK(cyc(s, e_att(TSR_OP_ATTQK, &q3)) == 350u + 2u * 272u, "ATTQK the busiest channel");
    att_f p = { 0, TSR_ATT_LSUM, 5, 0, 16, 1, 1, 5, 0, 100, 1000, 7000, 0, 128, 40, 0 };
    CHECK(cyc(s, e_att(TSR_OP_ATTPV, &p)) == 350u + 68u + 32u * 272u, "ATTPV");
    CHECK(cyc(s, e_asmax(100, 100, 50, 13, 0, 16, 0x3F800000u)) == 16u * 13u + 64u, "ASMAX");
    CHECK(cyc(s, e_amask(100, 100, 400, 13, 0, 16, 8, 13, 50)) == 16u * 13u + 16u + 45u, "AMASK odd n_lines");
    CHECK(cyc(s, e_amask(100, 100, 400, 12, 0, 16, 8, 12, 50)) == 16u * 12u + 45u, "AMASK even n_lines");
    CHECK(cyc(s, e_amerge(0, 100, 200, 50, 60, 40, 1, 0x3F800000u, 0)) == 6u * 3u + 64u + 320u * 5u, "AMERGE 32B (AT §9.2: 1,600 + prepass)");
    CHECK(cyc(s, e_amerge(1, 100, 200, 50, 60, 40, 1, 0x3F800000u, 0)) == 6u * 3u + 64u + 320u * 3u, "AMERGE INIT");
    CHECK(cyc(s, e_anorm(100, 200, 50, 40, 1)) == 4u * 3u + 64u + 320u, "ANORM");
    tsr_sim_destroy(s);
}

static uint32_t q_push(tsr_sim * s, tsr_insn i, uint32_t d0, uint32_t d1) {
    const uint32_t t = tsr_sim_push(s, i.w, d0, d1, ALL_WORDS);
    CHECK(t != TSR_TICKET_NONE, "push of op 0x%02x", i.w[0] & 0xFFu);
    return t;
}
static void make_active(tsr_sim * s) {
    tsr_sim_csr_write(s, TSR_CSR_LQ_PUSH, 0);
    uint32_t v = 0;
    CHECK(tsr_sim_cw_access(s, 1, TSR_CW_LQ_POP, &v) == TSR_CW_OK, "LQ_POP");
}

static void test_perf(void) {
    rng_t r = { 0xA7005u };
    tsr_sim * s = mk_sim(CAPS_AT, 8);
    rand_lines(&r, s, 0, SPM_LINES, 1);
    const uint32_t E = 128, LG = 8, BK = 1000, BV = 9000;
    fill_plane(&r, s, TSR_KV_K, BK, E, LG, 16u * NCH * LG);
    fill_plane(&r, s, TSR_KV_V, BV, E, LG, 16u * NCH * LG);
    put_qa(&r, s, 1000, 4u * E);
    put_qa(&r, s, 1100, 4u * E);
    put_qa(&r, s, 1200, E);
    put_qa(&r, s, 1210, E);
    put_qa(&r, s, 1220, E);
    make_active(s);
    const tsr_insn kvst = e_kvst(TSR_KV_V, 3, 1200, BV, 2, LG, 5, 0);
    const att_f qf = { 0, TSR_ATT_MAXOUT, 4, 0, 2, 0, 2, 4, 0, 2000, 1000, BK, 3, 13, lohi(0, 208), 1990 };
    const att_f pf = { 0, TSR_ATT_LSUM, 4, 0, 2, 0, 2, 4, 0, 3000, 5000, BV, 3, 13, LG, 0 };
    const tsr_insn qk = e_att(TSR_OP_ATTQK, &qf), pv = e_att(TSR_OP_ATTPV, &pf);
    const tsr_insn vops[] = { e_asmax(6000, 6000, 6900, 13, 0, 16, 0x3F800000u), e_amask(7000, 7000, 7300, 13, 0, 16, 8, 13, 7400),
                              e_amerge(1, 8000, 3000, 1990, 8100, 16, 0, 0x3F800000u, 0), e_anorm(8200, 8000, 8101, 16, 0) };
    const uint64_t vlines = 16u * 13u + (16u * 13u + 1u) + (64u + 2u) + 64u;
    q_push(s, kvst, TSR_TICKET_NONE, TSR_TICKET_NONE);
    q_push(s, qk, TSR_TICKET_NONE, TSR_TICKET_NONE);
    const uint32_t tpv = q_push(s, pv, TSR_TICKET_NONE, TSR_TICKET_NONE);
    q_push(s, vops[0], TSR_TICKET_NONE, TSR_TICKET_NONE);
    q_push(s, vops[1], TSR_TICKET_NONE, TSR_TICKET_NONE);
    q_push(s, vops[2], tpv, TSR_TICKET_NONE);
    q_push(s, vops[3], TSR_TICKET_NONE, TSR_TICKET_NONE);
    CHECK(tsr_sim_run_until_idle(s, 100000000ull), "idle");
    for (uint32_t u = 0; u < TSR_NUM_UNITS; ++u) CHECK(tsr_sim_csr_read(s, TSR_CSR_Q_ERR_CODE(u)) == 0u, "unit %u no error", u);
    const tsr_kv_geom g = { E, NCH, LG };
    tsr_kvst_stats st;
    tsr_kvst_count(TSR_KV_V, &g, 5, 3, &st);
    uint64_t rmw = 0, vcyc = 0;
    for (uint32_t c = 0; c < NCH; ++c) rmw += st.meta_rmw[c];
    for (int k = 0; k < 4; ++k) vcyc += cyc(s, vops[k]);
    const uint32_t C = NCH;
    CHECK(tsr_sim_perf(s, TSR_PERF_ATT_QK_CYCLES(C)) == cyc(s, qk), "ATT_QK_CYCLES = the ATTQK's busy time");
    CHECK(tsr_sim_perf(s, TSR_PERF_ATT_PV_CYCLES(C)) == cyc(s, pv), "ATT_PV_CYCLES");
    CHECK(tsr_sim_perf(s, TSR_PERF_MVM_CYCLES) == cyc(s, qk) + cyc(s, pv), "MVM_CYCLES = ATTQK + ATTPV");
    CHECK(tsr_sim_perf(s, TSR_PERF_VU_CYCLES) == vcyc, "VU_CYCLES = the vector ops");
    CHECK(tsr_sim_perf(s, TSR_PERF_DMA_CYCLES(C)) == cyc(s, kvst), "DMA_CYCLES = the KVST");
    CHECK(rmw > 0 && tsr_sim_perf(s, TSR_PERF_KVST_RMW(C)) == rmw, "KVST_RMW");
    CHECK(tsr_sim_perf(s, TSR_PERF_VU_LINES(C)) == vlines, "VU_LINES of the vector ops (%llu)",
          (unsigned long long)tsr_sim_perf(s, TSR_PERF_VU_LINES(C)));
    // groups 3..15: channel c streams its groups twice (K and V), 34 beats each, plus KVST's RMW reads
    for (uint32_t c = 0; c < NCH; ++c) {
        const uint64_t g_c = tsr_att_ch_groups(NCH, 3, 13, c);
        CHECK(tsr_sim_perf(s, TSR_PERF_CH_RD_BEATS(c)) == 2u * g_c * 34u + st.meta_rmw[c], "CH_RD_BEATS[%u]", c);
    }
    // 16 pass rows x 13 groups = 208 lines against 4 groups x 34 = 136 beats: 72 stalled cycles per op
    CHECK(tsr_sim_perf(s, TSR_PERF_ATT_PATH_STALL(C)) == 2u * (208u - 136u), "ATT_PATH_STALL");
    CHECK(tsr_sim_perf(s, TSR_PERF_RETIRED) == 7u, "RETIRED");
    tsr_sim_destroy(s);
}

// ---------------------------------------------------------------------------
// The AT §6.6 flash-attention decode layer, pushed through the queue model, against tsr_ref_fa
// ---------------------------------------------------------------------------
typedef struct {
    const char * name;
    uint32_t t_max, E, D, G, n_kv, c_new;
    uint32_t runs[4];          // MSUM row: lo0, hi0, lo1, hi1
    uint32_t scale;
} facase;

static void fa_decode(const facase * fc, uint64_t seed) {
    rng_t r = { seed };
    tsr_sim * s = mk_sim(CAPS_AT, fc->t_max);
    rand_lines(&r, s, 0, SPM_LINES, 1);                    // garbage everywhere the kernel does not write
    const uint32_t E = fc->E, D = fc->D, G = fc->G, n_head = G * (E / D), K = n_head * D, R = n_head;   // T = 1
    const uint32_t LG = tsr_kv_lg_min(fc->n_kv, NCH), BK = 4096, BV = 200000;
    fill_plane(&r, s, TSR_KV_K, BK, E, LG, 16u * NCH * LG);
    fill_plane(&r, s, TSR_KV_V, BV, E, LG, 16u * NCH * LG);
    // the token's f32 Q, K and V rows in the SPM
    const uint32_t QF = 64, KF = QF + K / 16u, VF = KF + E / 16u, QA = VF + E / 16u;
    const uint32_t KQA = QA + tsr_qa_lines(TSR_QA_A32, K), VQA = KQA + tsr_qa_lines(TSR_QA_A32, E);
    uint32_t next = VQA + tsr_qa_lines(TSR_QA_A32, E);
    float * qf = F(tsr_sim_spm(s), QF);
    for (uint32_t j = 0; j < K; ++j) qf[j] = (float)(rng_normal(&r) * 1.5);
    for (uint32_t j = 0; j < E; ++j) {
        F(tsr_sim_spm(s), KF)[j] = (float)(rng_normal(&r) * 0.7);
        F(tsr_sim_spm(s), VF)[j] = (float)(rng_normal(&r) * 0.7);
    }
    float * q_host = (float *)malloc(K * sizeof(float));
    memcpy(q_host, qf, K * sizeof(float));
    // the mask summary and the chunk plan (AT §6.6 step 3, §7.3)
    uint32_t mwords[TSR_MSUM_HDR_WORDS + TSR_MSUM_ROW_WORDS] = { TSR_MSUM_MAGIC, 1, fc->n_kv, 0, 0, 0, 0, 0,
                                                                 fc->runs[0], fc->runs[1], fc->runs[2], fc->runs[3] };
    tsr_msum ms;
    CHECK(tsr_msum_view(mwords, sizeof mwords / 4u, &ms) == 0, "%s: MSUM", fc->name);
    const tsr_att_cfg cfg = { NCH, fc->t_max, tsr_qa_lines(TSR_QA_A256, 32768) * 16u, 32768 };
    const uint32_t new_run[1][2] = { { fc->c_new, fc->c_new + 1u } };
    tsr_fa_chunk ch[64];
    const long nch = tsr_fa_plan(&cfg, E, D, &ms, 0, 1, new_run, 1, ch, 64);
    CHECK(nch >= 2 && nch <= 64, "%s: %ld chunks", fc->name, nch);
    if (nch < 1 || nch > 64) { tsr_sim_destroy(s); free(q_host); return; }
    uint32_t GS, Tq0;
    tsr_fa_pass_rule(&cfg, G, &GS, &Tq0);
    const uint32_t Rl = (R + 15u) / 16u, cs = R * D / 16u, dc = dcode_of(D);
    const uint32_t ORUN = next, ST = ORUN + cs, OUT = ST + 2u * Rl;
    next = OUT + cs;
    // the kernel's op stream (AT §6.6, §7.1): new K / V rows, Q, then the chunks, then ANORM
    const tsr_insn qk_new = tsr_enc_qact(TSR_QA_A32, KQA, KF, E, 1, 0), qv_new = tsr_enc_qact(TSR_QA_A32, VQA, VF, E, 1, 0);
    const uint32_t t_qk = q_push(s, qk_new, TSR_TICKET_NONE, TSR_TICKET_NONE);
    const uint32_t t_qv = q_push(s, qv_new, TSR_TICKET_NONE, TSR_TICKET_NONE);
    const uint32_t t_ks = q_push(s, e_kvst(TSR_KV_K, 1, KQA, BK, E / 64u, LG, fc->c_new, 0), t_qk, TSR_TICKET_NONE);
    const uint32_t t_vs = q_push(s, e_kvst(TSR_KV_V, 1, VQA, BV, E / 64u, LG, fc->c_new, 0), t_qv, TSR_TICKET_NONE);
    const uint32_t t_qa = q_push(s, tsr_enc_qact(TSR_QA_A32, QA, QF, K, 1, 0), TSR_TICKET_NONE, TSR_TICKET_NONE);
    uint32_t t_last = TSR_TICKET_NONE, n_ops = 5;
    int new_last = 0;
    for (long k = 0; k < nch; ++k) {
        const tsr_fa_chunk * c = &ch[k];
        CHECK(!c->general, "%s: FAST chunks only", fc->name);
        const uint32_t n = c->n_groups, S = next, MK = S + R * n, P = MK + Rl;
        next = P + NCH * cs + NCH * Rl;
        const int has_new = 16u * c->u0 <= fc->c_new && fc->c_new < 16u * (c->u0 + n);
        if (has_new) new_last = (k == nch - 1);
        uint32_t t_qk_last = TSR_TICKET_NONE, t_pv_last = TSR_TICKET_NONE;
        for (uint32_t hg0 = 0; hg0 < G; hg0 += GS) {
            const att_f a = { 0, c->flags | TSR_ATT_MAXOUT, G, hg0, E / 64u, dc, 1, G - hg0 < GS ? G - hg0 : GS, 0, S, QA, BK,
                              c->u0, n, lohi(c->c_lo, c->c_hi), MK };
            t_qk_last = q_push(s, e_att(TSR_OP_ATTQK, &a), t_qa, has_new ? t_ks : TSR_TICKET_NONE);
            ++n_ops;
        }
        const uint32_t t_sm = q_push(s, e_asmax(S, S, MK, n, 0, R, fc->scale), t_qk_last, TSR_TICKET_NONE);
        for (uint32_t hg0 = 0; hg0 < G; hg0 += GS) {
            const att_f a = { 0, TSR_ATT_LSUM, G, hg0, E / 64u, dc, 1, G - hg0 < GS ? G - hg0 : GS, 0, P, S, BV, c->u0, n, LG, cs };
            t_pv_last = q_push(s, e_att(TSR_OP_ATTPV, &a), t_sm, has_new ? t_vs : TSR_TICKET_NONE);
            ++n_ops;
        }
        t_last = q_push(s, e_amerge(k == 0, ORUN, P, MK, ST, R, dc, fc->scale, cs), t_pv_last, TSR_TICKET_NONE);
        n_ops += 2;
    }
    CHECK(new_last, "%s: the chunk with the new cell is last (AT §6.6)", fc->name);
    q_push(s, e_anorm(OUT, ORUN, ST + Rl, R, dc), t_last, TSR_TICKET_NONE);
    ++n_ops;
    CHECK(next < SPM_LINES, "%s: SPM layout fits (%u lines)", fc->name, next);
    CHECK(tsr_sim_run_until_idle(s, 1000000000ull), "%s: idle", fc->name);
    uint32_t done = 0;
    for (uint32_t u = 0; u < TSR_NUM_UNITS; ++u) {
        CHECK(tsr_sim_csr_read(s, TSR_CSR_Q_ERR_CODE(u)) == 0u, "%s: unit %u no error (code %u info 0x%x)", fc->name, u,
              tsr_sim_csr_read(s, TSR_CSR_Q_ERR_CODE(u)), tsr_sim_csr_read(s, TSR_CSR_Q_ERR_INFO(u)));
        done += tsr_sim_csr_read(s, TSR_CSR_Q_DONE(u));
    }
    CHECK(done == n_ops, "%s: %u ops done (%u pushed)", fc->name, done, n_ops);
    // tsr_ref_fa on the same planes (now holding the new token) and the same Q
    uint8_t * kp[NCH], * vp[NCH];
    planes_of(s, BK, kp);
    planes_of(s, BV, vp);
    float * out = (float *)malloc((size_t)R * D * sizeof(float));
    tsr_fa_stats fs;
    const tsr_fa_args fa = { 1, D, G, u2f(fc->scale), q_host, E, LG, (const uint8_t * const *)kp, (const uint8_t * const *)vp,
                             &ms, NULL, new_run, 1, out, &fs };
    const int rc = tsr_ref_fa(&cfg, &fa);
    CHECK(rc == TSR_FA_OK, "%s: tsr_ref_fa %d", fc->name, rc);
    CHECK(fs.chunks == (uint32_t)nch && fs.ops_attqk == fs.ops_attpv, "%s: same chunk count (%u)", fc->name, fs.chunks);
    const float * got = F(tsr_sim_spm(s), OUT);
    uint32_t bad = 0, finite = 1;
    for (uint32_t i = 0; i < R * D; ++i) {
        bad += f2u(got[i]) != f2u(out[i]);
        finite &= isfinite(got[i]) != 0;
    }
    CHECK(bad == 0, "%s: FA output equals tsr_ref_fa bit for bit (%u of %u words differ)", fc->name, bad, R * D);
    CHECK(finite, "%s: FA output finite", fc->name);
    printf("  %s: %ld chunks, %u ops, %llu core cycles\n", fc->name, nch, n_ops, (unsigned long long)tsr_sim_now(s));
    free(out);
    free(q_host);
    tsr_sim_destroy(s);
}

static void test_fa(void) {
    static const facase cs[] = {
        // one pass (GS = G = 4): 38 groups, the cell written in this launch in the first sub-range
        { "decode G4 D64", 8, 128, 64, 4, 600, 100, { 0, 600, 0, 0 }, 0x3E000000u },
        // G = 8 > T_MAX = 4: two head subsets; wrapped SWA ring: two runs, two FAST chunks with bounds
        { "decode 2 passes, SWA ring", 4, 128, 128, 8, 400, 149, { 0, 150, 300, 400 }, 0x3DB504F3u },
        // Qwen2.5-32B: 8 KV heads of D 128, G 5, R 40
        { "decode Qwen2.5-32B shape", 8, 1024, 128, 5, 300, 299, { 0, 300, 0, 0 }, 0x3DB504F3u },
        // Gemma 4 global layer: one KV head of D 512, G 16 > T_MAX: two passes of 8, scale 1
        { "decode Gemma 4 global", 8, 512, 512, 16, 520, 519, { 0, 520, 0, 0 }, 0x3F800000u },
    };
    for (size_t k = 0; k < sizeof(cs) / sizeof(cs[0]); ++k) fa_decode(&cs[k], 0xFA000u + k);
}

int main(void) {
    test_kvst();
    test_capture();
    test_attqk_attpv();
    test_vector_ops();
    test_checks();
    test_extents();
    test_cycles();
    test_perf();
    test_fa();
    return test_finish("sim_at");
}
