// test_fw_replay_sim.c — fw_replay_sim: trace replay through the `replay` kernel of the production
// image (docs/v2/06-software-verification.md "SW" §5.4, §7.7, §10.2; EX §4.6), the route P2.5 adds to
// tsr_trace2_replay_sim (sw/sim/tsr_trace.c "kernel_replay").
//
//   test_fw_replay_sim IMAGE TRACES_DIR [--all]
//     Sets traces/m2, gen, vu, at, ctrl (and m3 with --all) and traces/model-v1conv (with the SERIAL
//     flag, SW §5.4). Default: a fixed sample (every set's traces at a fixed stride, 6 per set and 3
//     per v1conv model), each replayed tile-only (the default route, under the fixed schedule) and
//     through the kernel under the fixed schedule and 7 random ones; --all: every trace, the kernel
//     route under the fixed and one random schedule. Every replay must pass every comparison of SW §5.4
//     (the DONE fields incl. the kernel's RET rule, expect regions, written / read DRAM, per-unit counts,
//     error records and push-stream projections), i.e. the kernel route gives the tile-only route's
//     results. The race detector runs in report mode and no report other than SPM_STALE is allowed;
//     SPM_STALE (a replayed op reads SPM that nothing wrote since the launch's LQ_POP, V2-SW-4) is a
//     property of some generated traces (traces/m2, gen, m3 read the simulator's initial SPM fill: an
//     open issue of the generator, not of the kernel) and is counted and allowed only for those sets.
//   Also: a synthetic stream of 3,100 records (three chunks: the kernel's own chunk LDs interleave
//     with the replayed DMA ops, whose deps reach across chunk boundaries) generated with
//     tsr_trace2_run_ops and replayed through the kernel under 8 schedules; and the SERIAL flag on a
//     stream whose recorded deps are all "none" while its ops chain through SPM across units (LD ->
//     QACT -> ST on the same lines; 120 records in chunks of 8): through the kernel with SERIAL under
//     the fixed and 2 random schedules, every replayed op's pushed dep (from the captured launch)
//     must be the previous op record's ticket, the race detector must stay quiet and the outputs
//     must equal the chained stream's; without SERIAL the same stream must race (the check can tell
//     SERIAL from the recorded deps, which the traces/model-v1conv chains cannot: there the recorded
//     deps already are the chain).
// Exit 0 when everything passed, 1 on a failure, 77 (skipped) when the image is missing.
#define _GNU_SOURCE
#include <dirent.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "test_util.h"
#include "tsr_isa.h"
#include "tsr_race.h"
#include "tsr_trace.h"

#include "fw_sim_host.hpp"       // fwh_open / fwh_kernel / fwh_launch (plain C)
#include "selftest_abi.hpp"      // tsr_st_qa_lines

static const char * g_image;

typedef struct { char ** v; size_t n, cap; } strlist;

static void sl_add(strlist * l, const char * s) {
    if (l->n == l->cap) {
        l->cap = l->cap ? 2 * l->cap : 64;
        l->v = (char **)realloc(l->v, l->cap * sizeof(char *));
    }
    l->v[l->n++] = strdup(s);
}
static void sl_free(strlist * l) {
    for (size_t i = 0; i < l->n; ++i) free(l->v[i]);
    free(l->v);
    memset(l, 0, sizeof(*l));
}
static int str_cmp(const void * a, const void * b) { return strcmp(*(char * const *)a, *(char * const *)b); }

static void collect(const char * dir, strlist * out, int depth) {
    char p[4096];
    struct stat st;
    if (snprintf(p, sizeof(p), "%s/meta.json", dir) >= (int)sizeof(p)) return;   // W2a: path too long (and -Wformat-truncation)
    if (stat(p, &st) == 0) {
        sl_add(out, dir);
        return;
    }
    if (depth > 6) return;
    DIR * d = opendir(dir);
    if (d == NULL) return;
    strlist subs = { 0 };
    struct dirent * e;
    while ((e = readdir(d)) != NULL) {
        if (e->d_name[0] == '.' || strcmp(e->d_name, "images") == 0) continue;
        if (snprintf(p, sizeof(p), "%s/%s", dir, e->d_name) >= (int)sizeof(p)) continue;
        if (stat(p, &st) == 0 && S_ISDIR(st.st_mode)) sl_add(&subs, p);
    }
    closedir(d);
    if (subs.n > 1) qsort(subs.v, subs.n, sizeof(char *), str_cmp);
    for (size_t i = 0; i < subs.n; ++i) collect(subs.v[i], out, depth + 1);
    sl_free(&subs);
}

typedef struct stats { uint32_t traces, replays, stale_traces; } stats;

// One trace: tile-only once, then the kernel route under n_sched schedules.
static void one(const tsr_trace2 * t, const char * name, uint32_t flags, uint32_t n_sched, int stale_ok, stats * st) {
    tsr_trace2_replay_opts o;
    memset(&o, 0, sizeof(o));
    o.sched = "fixed";
    o.race = 1;
    o.threads = 2;
    tsr_trace2_result r;
    int rc = tsr_trace2_replay_sim_ex(t, &o, &r);
    CHECK(rc == 0, "%s: tile-only route: %s", name, r.msg);
    CHECK(r.race_reports == r.race_stale && r.race_stale == 0, "%s: tile-only route: %u race reports", name, r.race_reports);
    int stale = 0;
    for (uint32_t k = 0; k < n_sched; ++k) {
        char sched[48];
        if (k == 0) snprintf(sched, sizeof(sched), "fixed");
        else snprintf(sched, sizeof(sched), "random:%u", 77u + k);
        memset(&o, 0, sizeof(o));
        o.sched = sched;
        o.race = 1;
        o.threads = 2;
        o.replay_image = g_image;
        o.replay_flags = flags;
        rc = tsr_trace2_replay_sim_ex(t, &o, &r);
        CHECK(rc == 0, "%s [%s]: replay kernel route: %s", name, sched, r.msg);
        CHECK(r.race_reports == r.race_stale, "%s [%s]: %u race reports besides SPM_STALE", name, sched,
              r.race_reports - r.race_stale);
        CHECK(r.race_stale == 0 || stale_ok, "%s [%s]: %u SPM_STALE reports", name, sched, r.race_stale);
        stale |= r.race_stale != 0;
        st->replays++;
    }
    st->traces++;
    st->stale_traces += stale ? 1u : 0u;
}

static void set_run(const char * root, const char * set, uint32_t flags, size_t sample, uint32_t n_sched, int stale_ok) {
    char dir[4096];
    snprintf(dir, sizeof(dir), "%s/%s", root, set);
    strlist l = { 0 };
    collect(dir, &l, 0);
    if (l.n == 0) {
        printf("fw_replay_sim: %s: no traces on disk (regen.sh): not run\n", dir);
        return;
    }
    const size_t step = sample && l.n > sample ? l.n / sample : 1;
    stats st = { 0, 0, 0 };
    for (size_t i = 0; i < l.n; i += step) {
        tsr_trace2 * t = NULL;
        char err[1024];
        if (tsr_trace2_read(l.v[i], &t, err, sizeof(err)) != 0) {
            CHECK(0, "%s: %s", l.v[i], err);
            continue;
        }
        if (t->kind == TSR_TRACE2_KIND_OPS) one(t, l.v[i], flags, n_sched, stale_ok, &st);
        tsr_trace2_free(t);
    }
    printf("fw_replay_sim: %s: %u traces, %u kernel-route replays, %u with SPM_STALE reports\n", set, st.traces, st.replays,
           st.stale_traces);
    sl_free(&l);
}

// ---------------------------------------------------------------------------
// A synthetic multi-chunk stream
// ---------------------------------------------------------------------------
static void synthetic(uint32_t n_sched) {
    tsr_trace2 * t = tsr_trace2_new();
    tsr_config core = TSR_CONFIG_U200_V1;
    core.ch_bytes = 1ull << 28;
    tsr_trace2_config_from_core(&t->cfg, &core);
    t->cfg.ctl_id = tsr_trace2_ctl_id_full();
    // Input rows: 256 lines on ch0 at the IO region; outputs: one line per record pair.
    rng_t r = { 0x5A11 };
    uint8_t * in = (uint8_t *)malloc(256u * 64u);
    rng_bytes(&r, in, 256u * 64u);
    const uint64_t in_beat = TSR_REGION_IO_OFF / 64u, out_beat = in_beat + 1024u;
    tsr_trace2_add_init(t, 0, in_beat * 64u, in, 256u * 64u);
    free(in);
    const uint32_t pairs = 1550;                              // 3,100 records: chunks of 1,364
    for (uint32_t i = 0; i < pairs; ++i) {
        const uint32_t line = i % 512u, src = in_beat + (i * 7u) % 256u;
        tsr_insn ld = tsr_enc_ld(0, line, (uint32_t)src, 1);
        // The LD depends on a DMA op ~1,400 records back (across a chunk boundary) every 5th pair.
        uint32_t dep = TSR_TICKET_NONE;
        if (i >= 700u && i % 5u == 0u) dep = tsr_trace2_ticket_of(t, t->n_push - 1400u);
        tsr_trace2_add_op(t, ld.w, dep, TSR_TICKET_NONE, 0x0F00u);
        const uint32_t ld_t = tsr_trace2_ticket_of(t, t->n_push - 1u);
        tsr_insn st = tsr_enc_st((i & 1u) ? 1u : 0u, line, (uint32_t)(out_beat + i), 1);
        tsr_trace2_add_op(t, st.w, ld_t, TSR_TICKET_NONE, 0x0F00u);
    }
    tsr_trace2_replay_opts o;
    memset(&o, 0, sizeof(o));
    o.init_from_reads = 1;
    o.race = 1;
    tsr_trace2_result res;
    int rc = tsr_trace2_run_ops(t, &o, &res);
    CHECK(rc == 0, "synthetic: run_ops: %s", res.msg);
    if (rc == 0) {
        stats st = { 0, 0, 0 };
        one(t, "synthetic 3100 records", 0, n_sched, 0, &st);
        printf("fw_replay_sim: synthetic stream of %zu records (3 chunks): %u kernel-route replays\n", t->n_push, st.replays);
    }
    tsr_trace2_free(t);
}

// ---------------------------------------------------------------------------
// SERIAL (SW §7.7 step 3, EX §4.6; fix W2a)
// ---------------------------------------------------------------------------
#define SER_GROUPS   40u                  // LD x, QACT, ST q: 120 records
#define SER_CHUNK    8u                   // records per chunk: 15 chunks, 13 own chunk LDs after base
#define SER_BUF      60000u               // SPM buffer line (2 x 6 lines), outside the ops' lines
#define SER_K        64u                  // f32 per row: 4 lines; QA(A32, 64): 2 lines

// The stream: deps = the previous record (chain = 1) or none (chain = 0); same ops otherwise.
static tsr_trace2 * serial_stream(int chain) {
    tsr_trace2 * t = tsr_trace2_new();
    tsr_config core = TSR_CONFIG_U200_V1;
    core.ch_bytes = 1ull << 28;
    tsr_trace2_config_from_core(&t->cfg, &core);
    t->cfg.ctl_id = tsr_trace2_ctl_id_full();
    rng_t r = { 0x5E81A1u };
    const uint32_t xl = SER_K / 16u, ql = tsr_st_qa_lines(0, SER_K);
    uint8_t * in = (uint8_t *)malloc(16u * xl * 64u);
    for (uint32_t i = 0; i < 16u * xl * 16u; ++i) {
        const uint32_t u = f2u((float)rng_normal(&r));
        memcpy(in + 4u * i, &u, 4);              // little-endian host
    }
    const uint64_t in_beat = TSR_REGION_IO_OFF / 64u + 0x2000u, out_beat = in_beat + 1024u;
    tsr_trace2_add_init(t, 0, in_beat * 64u, in, 16u * xl * 64u);
    free(in);
    for (uint32_t i = 0; i < SER_GROUPS; ++i) {
        const uint32_t X = 100u + 8u * (i % 4u), Q = 200u + 4u * (i % 4u);
        const tsr_insn op[3] = { tsr_enc_ld(0, X, (uint32_t)(in_beat + xl * (i % 16u)), xl),
                                 tsr_enc_qact(TSR_QA_A32, Q, X, SER_K, 1, 0),
                                 tsr_enc_st(i & 1u, Q, (uint32_t)(out_beat + ql * i), ql) };
        for (int k = 0; k < 3; ++k) {
            const uint32_t dep = chain && t->n_push > 0 ? tsr_trace2_ticket_of(t, t->n_push - 1u) : TSR_TICKET_NONE;
            tsr_trace2_add_op(t, op[k].w, dep, TSR_TICKET_NONE, 0xFF00u);
        }
    }
    return t;
}

// One launch of `t` through the replay kernel of the image on a fresh simulator. Returns the race
// report count; *done_ok: DONE OK / RET 0; outputs compared with want's expect regions; with
// check_deps, the pushed deps of the replayed ops against the SERIAL chain.
static size_t serial_launch(const tsr_trace2 * t, const tsr_trace2 * want, const char * sched, uint32_t flags,
                            int check_deps) {
    tsr_sim_opts so;
    memset(&so, 0, sizeof(so));
    so.sched = sched;
    so.race = 1;                                             // report: count, do not abort
    so.threads = 2;
    tsr_sim * sim = tsr_sim_create_opts(&t->cfg.core, &so);
    CHECK(sim != NULL, "serial [%s]: simulator", sched);
    if (sim == NULL) return 0;
    for (size_t i = 0; i < want->mem.n_init; ++i)
        tsr_sim_dram_write(sim, want->mem.init[i].ch, want->mem.init[i].byte_off, want->mem.init[i].data,
                           want->mem.init[i].size);
    const uint64_t load_beat = (48ull << 20) / 64u, lines = (t->n_push * TSR_TRACE_PUSH_BYTES + 63u) / 64u;
    uint8_t * b = (uint8_t *)calloc((size_t)lines * 64u, 1);
    for (size_t i = 0; i < t->n_push; ++i) {
        uint32_t w[12];
        memcpy(w, t->push[i].w, 32);
        w[8] = t->push[i].dep[0]; w[9] = t->push[i].dep[1]; w[10] = t->push[i].push; w[11] = t->push[i].kind;
        memcpy(b + i * TSR_TRACE_PUSH_BYTES, w, sizeof(w));   // little-endian host
    }
    tsr_sim_dram_write(sim, 0, load_beat * 64u, b, lines * 64u);
    free(b);
    fwh h;
    size_t n_race = 0;
    if (fwh_open(&h, sim, g_image) != 0) {
        CHECK(0, "serial [%s]: boot", sched);
        tsr_sim_destroy(sim);
        return 0;
    }
    const int idx = fwh_kernel(&h, "replay");
    tsr_sim_capture_launch(sim, 1);
    const uint32_t a[7] = { (uint32_t)load_beat, (uint32_t)t->n_push, flags | TSR_REPLAY_F_BUF, SER_BUF, SER_CHUNK, 0, 0 };
    const int ok = idx >= 0 && fwh_launch(&h, (uint32_t)idx, a, 0) == 0;
    CHECK(ok, "serial [%s]: replay launch answered", sched);
    if (ok && check_deps) {
        CHECK(fwh_status(&h) == TSR_LCQ_R0_STATUS_OK && fwh_ret(&h) == 0, "serial [%s]: DONE status %u RET %d", sched,
              fwh_status(&h), fwh_ret(&h));
        for (size_t i = 0; i < want->mem.n_expect; ++i) {
            const tsr_region * x = &want->mem.expect[i];
            CHECK(memcmp(tsr_sim_dram(sim, x->ch) + x->byte_off, x->data, (size_t)x->size) == 0,
                  "serial [%s]: expect region %zu (ch %u 0x%llx) differs from the chained stream's", sched, i, x->ch,
                  (unsigned long long)x->byte_off);
        }
        tsr_trace2 c;
        tsr_trace2_init(&c);
        CHECK(tsr_sim_capture_take(sim, &c) == 0, "serial [%s]: the launch was captured", sched);
        // The replayed ops: the kind-0 records other than the kernel's chunk LDs (from push.bin) and FENCEs.
        size_t n_rep = 0, prev = 0, bad = 0;
        for (size_t i = 0; i < c.n_push; ++i) {
            const tsr_trace_push * p = &c.push[i];
            if (p->kind != TSR_TRACE_PUSH_OP || (p->w[0] & 0xFFu) == TSR_OP_FENCE) continue;
            if ((p->w[0] & 0xFFu) == TSR_OP_LD && p->w[2] >= load_beat && p->w[2] < load_beat + lines) continue;
            const uint32_t want0 = n_rep == 0 ? TSR_TICKET_NONE : tsr_trace2_ticket_of(&c, prev);
            if ((p->dep[0] != want0 || p->dep[1] != TSR_TICKET_NONE) && bad++ < 3)
                CHECK(0, "serial [%s]: replayed op %zu pushed deps 0x%08x 0x%08x, expected 0x%08x (the previous op record) "
                      "and none", sched, n_rep, p->dep[0], p->dep[1], want0);
            prev = i;
            ++n_rep;
        }
        CHECK(n_rep == t->n_push && bad == 0, "serial [%s]: %zu replayed ops (expected %zu), %zu with other deps", sched,
              n_rep, t->n_push, bad);
        tsr_trace2_clear(&c);
    }
    const tsr_race_report * rr = NULL;
    n_race = tsr_sim_race_reports(sim, &rr);
    tsr_sim_race_clear(sim);                                 // the expected races of the non-SERIAL run: not printed
    fwh_close(&h);
    tsr_sim_destroy(sim);
    return n_race;
}

static void serial_case(void) {
    tsr_trace2 * chain = serial_stream(1);
    tsr_trace2_replay_opts o;
    memset(&o, 0, sizeof(o));
    o.init_from_reads = 1;
    o.race = 1;
    tsr_trace2_result res;
    const int rc = tsr_trace2_run_ops(chain, &o, &res);
    CHECK(rc == 0 && res.race_reports == 0 && chain->done.status == TSR_LCQ_R0_STATUS_OK && chain->mem.n_expect > 0,
          "serial: the chained stream: run_ops: %s (%u race reports)", res.msg, res.race_reports);
    if (rc == 0) {
        stats st = { 0, 0, 0 };
        one(chain, "serial: chained stream (SERIAL = its own deps)", TSR_REPLAY_F_SERIAL, 2, 0, &st);
        tsr_trace2 * none = serial_stream(0);
        static const char * const sch[3] = { "fixed", "random:81", "random:82" };
        for (int k = 0; k < 3; ++k) {
            const size_t n = serial_launch(none, chain, sch[k], TSR_REPLAY_F_SERIAL, 1);
            CHECK(n == 0, "serial [%s]: SERIAL replay of the dep-free stream: %zu race reports", sch[k], n);
        }
        const size_t n = serial_launch(none, chain, "fixed", 0, 0);
        CHECK(n > 0, "serial: the dep-free stream without SERIAL must race (%zu reports): the test cannot see SERIAL", n);
        printf("fw_replay_sim: SERIAL: %zu-record dep-free stream, chained under SERIAL (3 schedules), %zu race reports "
               "without it\n", none->n_push, n);
        tsr_trace2_free(none);
    }
    tsr_trace2_free(chain);
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s IMAGE TRACES_DIR [--all]\n", argv[0]);
        return 2;
    }
    g_image = argv[1];
    const char * root = argv[2];
    const int all = argc > 3 && strcmp(argv[3], "--all") == 0;
    struct stat st;
    if (stat(g_image, &st) != 0) {
        printf("%s: no image %s: skipped\n", argv[0], g_image);
        return 77;
    }
    const uint32_t n_sched = all ? 2u : 8u;
    const size_t sample = all ? 0 : 6;
    set_run(root, "m2", 0, sample, n_sched, 1);
    set_run(root, "gen", 0, sample, n_sched, 1);
    set_run(root, "vu", 0, sample, n_sched, 0);
    set_run(root, "at", 0, sample, n_sched, 0);
    set_run(root, "ctrl", 0, sample, n_sched, 0);
    if (all) set_run(root, "m3", 0, 0, n_sched, 1);
    set_run(root, "model-v1conv/qwen05b-q8_0", TSR_REPLAY_F_SERIAL, all ? 0 : 3, n_sched, 0);
    set_run(root, "model-v1conv/qwen05b-q4_k_m-requant", TSR_REPLAY_F_SERIAL, all ? 0 : 3, n_sched, 0);
    synthetic(all ? 2u : 8u);
    serial_case();
    return test_finish("fw_replay_sim");
}
