// test_trace_roundtrip.c — ctest trace_v2_roundtrip (docs/design/06-software-verification.md "SW" §5.5,
// §10.1 "launch traces: fresh-boot replay", §10.2): launch capture on the simulator, the trace set
// on disk, and replay on freshly booted simulators.
//
// The firmware is sim_rv_issue's (tests/rv_issue/fw_issue.c: a minimal EX §4.7 runtime with one
// kernel "mvm_act" that pushes LD, QACT, MVM, VOP, RED and ST with dependencies, then joins with a
// FENCE). The host side loads it through the CSR port, boots it and pushes 8 descriptors (two
// before HELLO): PING, three good kernel launches, one whose LD is out of range (TILE_ERR with
// DEP_FAILED / ABORTED behind it), an unknown TYPE, a kernel index beyond the table (BAD_DESC)
// and a last PING. Scenario A captures with tsr_sim_capture_launch and, at the same time, with
// TSR_TRACE (tsr_sim_opts.trace_dir); the image is synthesised from the loader's writes.
// Scenario B gives the real ELF with tsr_sim_capture_set_image.
// Checked:
//   - which launches become traces (5: the LAUNCH descriptors answered by DONE; PINGs and the
//     unknown TYPE are skipped), and every field of each: DONE, relative tickets of the deps
//     (base = ISSUED_SEQ at kernel entry, the runtime's ISSUE_PIDX read), per-unit counts and
//     error records, expect = the kernel's output region, the descriptor, the image id;
//   - the set on disk: one image file for all traces, every trace read back equal to the captured
//     one, the TSR_TRACE dump equal as well;
//   - every trace replays on a fresh simulator (boot, HELLO, the descriptor, its DONE) under the
//     fixed schedule and two random ones with every SW §5.4 comparison passing;
//   - a corrupted expect byte, push record and tickets field are each detected by the replay.
// Usage: trace_v2_roundtrip <fw_issue.elf> <scratch base>. Exit 77 without the ELF.
#include "test_sim_util.h"

#include <dirent.h>

#include "tsr_bus.h"
#include "tsr_ref_vu.h"
#include "tsr_sim_int.h"
#include "tsr_trace.h"

#define NCH      2u
#define N_LAUNCH 8u

static char * pj(const char * a, const char * b) {
    const size_t n = strlen(a) + strlen(b) + 2;
    char * p = (char *)malloc(n);
    snprintf(p, n, "%s/%s", a, b);
    return p;
}

// ---------------------------------------------------------------------------
// Host side of EX §3.3 and §4.8 (as sim_rv_issue)
// ---------------------------------------------------------------------------
static uint32_t crc32_zlib(const uint8_t * p, size_t n) {
    uint32_t c = 0xFFFFFFFFu;
    for (size_t i = 0; i < n; ++i) {
        c ^= p[i];
        for (int k = 0; k < 8; ++k) c = (c >> 1) ^ (TSR_CRC32_POLY & (0u - (c & 1u)));
    }
    return ~c;
}

static void load_image(tsr_sim * sim, const tsr_rv_elf * elf) {
    for (uint32_t s = 0; s < elf->nseg; ++s) {
        const tsr_rv_elf_seg * g = &elf->seg[s];
        if (g->filesz == 0) continue;
        const uint32_t bytes = (g->filesz + 3u) & ~3u;
        uint8_t * buf = (uint8_t *)calloc(bytes, 1);
        memcpy(buf, g->data, g->filesz);
        tsr_sim_csr_write(sim, TSR_CSR_LOAD_ADDR, g->paddr);
        for (uint32_t i = 0; i < bytes; i += 4) tsr_sim_csr_write(sim, TSR_CSR_LOAD_DATA, get_u32_le(buf + i));
        CHECK(tsr_sim_csr_read(sim, TSR_CSR_LOAD_CRC) == crc32_zlib(buf, bytes), "segment 0x%08x: receive CRC", g->paddr);
        free(buf);
    }
}

typedef struct host {
    tsr_sim * sim;
    uint32_t  lcq_rd;
    uint32_t  rec[64][8];
    uint32_t  n_rec;
} host;

static void push_launch(host * h, const uint32_t d[8]) {
    for (uint32_t k = 0; k < 8u; ++k) tsr_sim_csr_write(h->sim, TSR_CSR_LQ_W(k), d[k]);
    tsr_sim_csr_write(h->sim, TSR_CSR_LQ_PUSH, 1u);
}

static void consume(host * h) {
    const uint32_t wr = tsr_sim_csr_read(h->sim, TSR_CSR_LCQ_WR);
    for (uint32_t k = h->lcq_rd; k != wr; ++k) {
        if (h->n_rec < 64u) {
            for (uint32_t j = 0; j < 8u; ++j) h->rec[h->n_rec][j] = tsr_sim_csr_read(h->sim, TSR_CSR_LCQ_REC(k % TSR_LCQ_DEPTH, j));
            ++h->n_rec;
        }
    }
    h->lcq_rd = wr;
    tsr_sim_csr_write(h->sim, TSR_CSR_LCQ_RD, wr);
    tsr_sim_csr_write(h->sim, TSR_CSR_LOG_RD, tsr_sim_csr_read(h->sim, TSR_CSR_LOG_WR));
}

static int run_until(host * h, uint32_t n_want) {
    for (int it = 0; it < 4000 && h->n_rec < n_want; ++it) {
        tsr_sim_advance(h->sim, 200000);
        consume(h);
        if (TSR_GET(tsr_sim_csr_read(h->sim, TSR_CSR_RT_STATE), TSR_RT_STATE_PHASE) == TSR_RT_STATE_PHASE_TRAPPED) break;
    }
    CHECK(h->n_rec >= n_want, "%u records, expected %u", h->n_rec, n_want);
    return h->n_rec >= n_want;
}

// ---------------------------------------------------------------------------
// Kernel cases (sim_rv_issue's "mvm_act": y = act(W x), stored to DRAM)
// ---------------------------------------------------------------------------
typedef struct kcase {
    uint32_t fmt, N, K, func;
    uint64_t x_beat, w_off, y_beat;
    int      bad_ld;
    uint32_t qa_lines, y_lines;
    uint8_t  y[64 * 64];
} kcase;

static void set_case(tsr_sim * sim, rng_t * r, kcase * kc) {
    const uint32_t fmt = kc->fmt, N = kc->N, K = kc->K, kind = tsr_qa_kind_for_fmt(fmt);
    kc->qa_lines = tsr_qa_lines(kind, K);
    kc->y_lines = tsr_ceil_div_u32(N, 16);
    const size_t row_bytes = (size_t)(K / tsr_fmt_block_size(fmt)) * tsr_fmt_ggml_block_bytes(fmt);
    uint8_t * blocks = (uint8_t *)malloc(row_bytes * N);
    for (uint32_t n = 0; n < N; ++n) gen_row_blocks(r, fmt, K, blocks + row_bytes * n);
    put_weights(sim, fmt, N, K, kc->w_off, blocks);
    float x[1024];
    gen_f32(r, x, K);
    put_f32_le(tsr_sim_dram(sim, 0) + kc->x_beat * 64u, x, K);
    uint8_t * qa = (uint8_t *)malloc((size_t)kc->qa_lines * 64u);
    uint8_t * row = (uint8_t *)malloc((size_t)tsr_row_beats(fmt, K) * 64u);
    tsr_qact(kind, x, K, qa);
    uint32_t yw[1024] = { 0 }, zw[1024] = { 0 };
    for (uint32_t n = 0; n < N; ++n) {
        tsr_repack_row(fmt, blocks + row_bytes * n, K, row);
        yw[n] = f2u(tsr_dot_row(fmt, row, K, qa));
    }
    tsr_ref_vop(kc->func, TSR_VOP_CMODE_NONE, 1, N, zw, kc->y_lines, yw, kc->y_lines, NULL, 0, NULL);
    for (uint32_t i = 0; i < 16u * kc->y_lines; ++i) {
        kc->y[4 * i] = (uint8_t)zw[i]; kc->y[4 * i + 1] = (uint8_t)(zw[i] >> 8);
        kc->y[4 * i + 2] = (uint8_t)(zw[i] >> 16); kc->y[4 * i + 3] = (uint8_t)(zw[i] >> 24);
    }
    memset(tsr_sim_dram(sim, 1) + kc->y_beat * 64u, 0xEE, (size_t)kc->y_lines * 64u);
    free(blocks); free(qa); free(row);
}

static void kernel_desc(const kcase * kc, uint32_t d[8]) {
    memset(d, 0, 8 * sizeof(uint32_t));
    d[0] = TSR_DESC_TYPE_LAUNCH | TSR_PUT(TSR_DESC_KERNEL, 0u);
    d[1] = TSR_PUT(TSR_TICKET_UNIT, 0u) | (kc->bad_ld ? 0x0FFFFFF0u : (uint32_t)kc->x_beat);
    d[2] = (uint32_t)kc->w_off;
    d[3] = TSR_PUT(TSR_TICKET_UNIT, 1u) | (uint32_t)kc->y_beat;
    d[4] = kc->N;
    d[5] = kc->K;
    d[6] = kc->fmt | kc->qa_lines << 16;
    d[7] = kc->func;
}

// ---------------------------------------------------------------------------
// Trace comparison
// ---------------------------------------------------------------------------
static int regions_eq(const tsr_region * a, size_t na, const tsr_region * b, size_t nb) {
    if (na != nb) return 0;
    for (size_t i = 0; i < na; ++i)
        if (a[i].ch != b[i].ch || a[i].byte_off != b[i].byte_off || a[i].size != b[i].size ||
            memcmp(a[i].data, b[i].data, (size_t)a[i].size) != 0)
            return 0;
    return 1;
}

static void check_eq(const tsr_trace2 * a, const tsr_trace2 * b, const char * tag) {
    CHECK(a->kind == b->kind && a->cfg.core.num_ch == b->cfg.core.num_ch && a->cfg.core.ch_bytes == b->cfg.core.ch_bytes &&
          a->cfg.core.caps == b->cfg.core.caps && a->cfg.ctl_id == b->cfg.ctl_id && a->cfg.cfg3 == b->cfg.cfg3 &&
          memcmp(a->cfg.q_depth, b->cfg.q_depth, sizeof(a->cfg.q_depth)) == 0, "%s: config", tag);
    CHECK(a->image_id == b->image_id && a->image_size == b->image_size && memcmp(a->image, b->image, a->image_size) == 0,
          "%s: image", tag);
    CHECK(strcmp(a->kernel, b->kernel) == 0 && a->kernel_index == b->kernel_index && a->prof == b->prof &&
          memcmp(a->args, b->args, sizeof(a->args)) == 0, "%s: launch", tag);
    CHECK(a->n_push == b->n_push && (a->n_push == 0 || memcmp(a->push, b->push, a->n_push * sizeof(tsr_trace_push)) == 0),
          "%s: push records", tag);
    CHECK(memcmp(&a->done, &b->done, sizeof(a->done)) == 0, "%s: done", tag);
    CHECK(regions_eq(a->mem.init, a->mem.n_init, b->mem.init, b->mem.n_init), "%s: init", tag);
    CHECK(regions_eq(a->mem.expect, a->mem.n_expect, b->mem.expect, b->mem.n_expect), "%s: expect", tag);
    CHECK(a->n_units == b->n_units && memcmp(a->unit, b->unit, sizeof(a->unit)) == 0, "%s: tickets", tag);
    CHECK(strcmp(a->sched, b->sched) == 0 && a->rv_ratio == b->rv_ratio, "%s: sched", tag);
}

static void expect_fail(const tsr_trace2 * t, uint32_t cls, const char * what) {
    tsr_trace2_result r;
    const int rc = tsr_trace2_replay_sim(t, NULL, NULL, &r);
    CHECK(rc == 1 && (r.fail_mask & cls), "%s: detected (rc %d, mask 0x%x: %s)", what, rc, r.fail_mask, r.msg);
}

// ---------------------------------------------------------------------------
// Scenarios
// ---------------------------------------------------------------------------
static tsr_sim * make_sim(const char * dump_dir) {
    tsr_config cfg = TSR_CONFIG_U200_V1;
    cfg.num_ch = NCH;
    cfg.ch_bytes = 1ull << 30;
    cfg.caps = TSR_CAPS_VU;
    tsr_sim_opts o;
    memset(&o, 0, sizeof(o));
    o.trace_dir = dump_dir;
    tsr_sim * sim = tsr_sim_create_opts(&cfg, &o);
    CHECK(sim != NULL, "create");
    return sim;
}

static void setup_cases(tsr_sim * sim, kcase * kc) {
    rng_t r = { 0x5EED1234u };
    const uint32_t fmts[4] = { TSR_FMT_Q8_0, TSR_FMT_Q4_K, TSR_FMT_Q8_0, TSR_FMT_Q8_0 };
    const uint32_t Ns[4] = { 40, 24, 16, 17 }, Ks[4] = { 128, 256, 64, 96 };
    const uint32_t funcs[4] = { TSR_VOP_FUNC_SILU, TSR_VOP_FUNC_TANH, TSR_VOP_FUNC_SILU, TSR_VOP_FUNC_GELU };
    for (uint32_t i = 0; i < 4; ++i) {
        memset(&kc[i], 0, sizeof(kc[i]));
        kc[i].fmt = fmts[i]; kc[i].N = Ns[i]; kc[i].K = Ks[i]; kc[i].func = funcs[i];
        kc[i].x_beat = TSR_REGION_IO_OFF / 64u + 1000u * i;
        kc[i].w_off = TSR_REGION_WEIGHT_OFF / 64u + 100000u * i;
        kc[i].y_beat = TSR_REGION_IO_OFF / 64u + 50000u + 1000u * i;
        kc[i].bad_ld = i == 2;
        set_case(sim, &r, &kc[i]);
    }
}

// The deps the kernel pushes, as relative tickets: LD -, QACT <- LD, MVM <- QACT, VOP <- MVM, RED
// <- VOP, ST <- VOP; then the runtime's FENCE.
static void check_kernel_pushes(const tsr_trace2 * t, const char * tag) {
    static const uint32_t ops[7] = { TSR_OP_LD, TSR_OP_QACT, TSR_OP_MVM, TSR_OP_VOP, TSR_OP_RED, TSR_OP_ST, TSR_OP_FENCE };
    static const uint32_t dep0[7] = { TSR_TICKET_NONE, 0x00000000u, 0x10000000u, 0x20000000u, 0x30000000u, 0x30000000u,
                                      TSR_TICKET_NONE };
    CHECK(t->n_push == 7, "%s: %zu push records (6 ops + the join FENCE)", tag, t->n_push);
    for (size_t i = 0; i < 7 && i < t->n_push; ++i)
        CHECK(t->push[i].kind == TSR_TRACE_PUSH_OP && (t->push[i].w[0] & 0xFFu) == ops[i] && t->push[i].dep[0] == dep0[i] &&
              t->push[i].dep[1] == TSR_TICKET_NONE && t->push[i].push == TSR_ISSUE_PUSH_WMASK,
              "%s: push %zu: op 0x%02x deps 0x%08x,0x%08x", tag, i, t->push[i].w[0] & 0xFFu, t->push[i].dep[0], t->push[i].dep[1]);
}

static void scenario_a(const char * elf_path, const char * scratch) {
    char * dump = pj(scratch, "dump");
    tsr_sim * sim = make_sim(dump);
    if (sim == NULL) { free(dump); return; }
    tsr_sim_capture_launch(sim, 1);
    tsr_rv_elf elf;
    if (tsr_rv_elf_read(elf_path, &elf) != 0) {
        CHECK(0, "read %s", elf_path);
        tsr_sim_destroy(sim);
        free(dump);
        return;
    }
    load_image(sim, &elf);
    tsr_rv_elf_free(&elf);
    static kcase kc[4];
    setup_cases(sim, kc);
    uint32_t desc[N_LAUNCH][8];
    memset(desc, 0, sizeof(desc));
    desc[0][0] = TSR_DESC_TYPE_PING; desc[0][1] = 0xC0FFEE01u;
    kernel_desc(&kc[0], desc[1]);
    kernel_desc(&kc[1], desc[2]);
    kernel_desc(&kc[2], desc[3]);
    desc[4][0] = 0x07u;
    kernel_desc(&kc[3], desc[5]);
    kernel_desc(&kc[0], desc[6]); desc[6][0] = TSR_DESC_TYPE_LAUNCH | TSR_PUT(TSR_DESC_KERNEL, 3u);
    desc[7][0] = TSR_DESC_TYPE_PING; desc[7][1] = 0xC0FFEE07u;
    host h;
    memset(&h, 0, sizeof(h));
    h.sim = sim;
    push_launch(&h, desc[0]);
    push_launch(&h, desc[1]);
    tsr_sim_csr_write(sim, TSR_CSR_RV_BOOT, 0u);
    tsr_sim_csr_write(sim, TSR_CSR_CTRL, TSR_CTRL_RV_RUN);
    if (!run_until(&h, 1)) goto out;
    for (uint32_t i = 2; i < N_LAUNCH; ++i) push_launch(&h, desc[i]);
    if (!run_until(&h, 13)) goto out;
    tsr_sim_advance(sim, 100000);
    consume(&h);
    {
        tsr_trace_capture_stats st;
        tsr_sim_capture_stats(sim, &st);
        CHECK(st.captured == 5 && st.written == 5 && st.skipped == 3 && st.failed == 0 && st.queued == 5,
              "capture stats: captured %u written %u skipped %u failed %u queued %u", st.captured, st.written, st.skipped,
              st.failed, st.queued);
        // Launch seq -> what the trace holds.
        static const uint32_t seqs[5] = { 1, 2, 3, 5, 6 };
        const kcase * kcs[5] = { &kc[0], &kc[1], &kc[2], &kc[3], &kc[0] };
        tsr_trace2 tr[5];
        char * set = pj(scratch, "set");
        char * images = pj(set, "images");
        for (int i = 0; i < 5; ++i) {
            tsr_trace2_init(&tr[i]);
            CHECK(tsr_sim_capture_take(sim, &tr[i]) == 0, "take %d", i);
            char tag[32];
            snprintf(tag, sizeof(tag), "launch %u", seqs[i]);
            const tsr_trace2 * t = &tr[i];
            uint32_t d[8];
            memcpy(d, i == 4 ? desc[6] : desc[seqs[i]], sizeof(d));
            CHECK(t->kind == TSR_TRACE2_KIND_LAUNCH && t->kernel_index == TSR_GET(d[0], TSR_DESC_KERNEL) && t->prof == 0 &&
                  memcmp(t->args, d + 1, sizeof(t->args)) == 0, "%s: descriptor", tag);
            uint64_t id = 0;
            char err[256];
            CHECK(tsr_trace2_image_id(t->image, t->image_size, &id, err, sizeof(err)) == 0 && id == t->image_id,
                  "%s: image id is EX §3.2's", tag);
            CHECK(strcmp(t->kernel, "") == 0, "%s: a synthesised image has no kernel table", tag);
            if (i == 4) {                                                  // kernel 3: BAD_DESC, nothing pushed
                CHECK(t->done.status == TSR_LCQ_R0_STATUS_BAD_DESC && t->done.err_ticket_rel == TSR_TICKET_NONE && t->n_push == 0 &&
                      t->mem.n_init == 0 && t->mem.n_expect == 0, "%s: BAD_DESC launch", tag);
                for (uint32_t u = 0; u < TSR_NUM_UNITS; ++u) CHECK(t->unit[u].count == 0 && t->unit[u].err_code == 0, "%s: unit %u", tag, u);
            } else if (kcs[i]->bad_ld) {
                const uint32_t ld_w0 = tsr_w0(TSR_OP_LD, 0, 0, 0);
                CHECK(t->done.status == TSR_LCQ_R0_STATUS_TILE_ERR && t->done.err_code == TSR_ERR_DRAM_RANGE && t->done.ret == 6 &&
                      t->done.err_ticket_rel == 0x00000000u && t->done.err_pidx_rel == 0 && t->done.err_info == 0x0FFFFFF0u &&
                      t->done.err_w0 == ld_w0, "%s: DONE TILE_ERR relative (status %u ticket 0x%08x pidx %u)", tag,
                      t->done.status, t->done.err_ticket_rel, t->done.err_pidx_rel);
                check_kernel_pushes(t, tag);
                CHECK(t->unit[TSR_UNIT_DMA].err_code == TSR_ERR_DRAM_RANGE && t->unit[TSR_UNIT_DMA].err_rel_seq == 0 &&
                      t->unit[TSR_UNIT_DMA].err_rel_pidx == 0 && t->unit[TSR_UNIT_DMA].err_info == 0x0FFFFFF0u,
                      "%s: DMA error record", tag);
                CHECK(t->unit[TSR_UNIT_QACT].err_code == TSR_ERR_DEP_FAILED && t->unit[TSR_UNIT_MVM].err_code == TSR_ERR_DEP_FAILED &&
                      t->unit[TSR_UNIT_VU].err_code == TSR_ERR_DEP_FAILED && t->unit[TSR_UNIT_VU].err_rel_pidx == 3,
                      "%s: DEP_FAILED records", tag);
                CHECK(t->unit[TSR_UNIT_DMA].count == 2 && t->unit[TSR_UNIT_VU].count == 2 && t->unit[TSR_UNIT_SYNC].count == 1,
                      "%s: counts", tag);
                CHECK(t->mem.n_expect == 0 && t->mem.n_init == 0, "%s: nothing read or written (%zu / %zu)", tag,
                      t->mem.n_init, t->mem.n_expect);
            } else {
                CHECK(t->done.status == TSR_LCQ_R0_STATUS_OK && t->done.err_code == 0 && t->done.ret == 6 &&
                      t->done.err_ticket_rel == TSR_TICKET_NONE && t->done.err_pidx_rel == 0, "%s: DONE OK", tag);
                check_kernel_pushes(t, tag);
                static const uint32_t cnt[6] = { 2, 1, 1, 2, 0, 1 };
                for (uint32_t u = 0; u < TSR_NUM_UNITS; ++u)
                    CHECK(t->unit[u].count == cnt[u] && t->unit[u].err_code == 0, "%s: unit %u count %u", tag, u, t->unit[u].count);
                CHECK(t->mem.n_expect == 1 && t->mem.expect[0].ch == 1 && t->mem.expect[0].byte_off == kcs[i]->y_beat * 64u &&
                      t->mem.expect[0].size == (uint64_t)kcs[i]->y_lines * 64u &&
                      memcmp(t->mem.expect[0].data, kcs[i]->y, (size_t)kcs[i]->y_lines * 64u) == 0,
                      "%s: expect = the kernel's output = the reference", tag);
                CHECK(t->mem.n_init >= 2, "%s: init holds x and the weights (%zu regions)", tag, t->mem.n_init);
            }
            char name[16];
            snprintf(name, sizeof(name), "%06u", seqs[i]);
            char * dir = pj(set, name);
            CHECK(tsr_trace2_write(t, dir, images) == 0, "%s: write", tag);
            free(dir);
        }
        CHECK(tsr_sim_capture_take(sim, &tr[0]) == 1, "the queue is empty after 5 takes");
        // One image file for the whole set.
        {
            int n_img = 0;
            DIR * dd = opendir(images);
            struct dirent * e;
            while (dd && (e = readdir(dd)) != NULL) n_img += strstr(e->d_name, ".elf") != NULL;
            if (dd) closedir(dd);
            CHECK(n_img == 1, "one image in %s (%d)", images, n_img);
        }
        // Read back, compare with the captured trace and with the TSR_TRACE dump, replay.
        const int have_random = 1;
        for (int i = 0; i < 5; ++i) {
            char name[16], tag[32];
            snprintf(name, sizeof(name), "%06u", seqs[i]);
            snprintf(tag, sizeof(tag), "launch %u", seqs[i]);
            char * dir = pj(set, name), * ddir = pj(dump, name);
            tsr_trace2 * r = NULL, * dr = NULL;
            char err[600];
            CHECK(tsr_trace2_read(dir, &r, err, sizeof(err)) == 0, "%s: read back: %s", tag, err);
            CHECK(tsr_trace2_read(ddir, &dr, err, sizeof(err)) == 0, "%s: read the TSR_TRACE dump: %s", tag, err);
            if (r) check_eq(&tr[i], r, tag);
            if (dr) check_eq(&tr[i], dr, tag);
            if (r) {
                static const char * const scheds[3] = { NULL, "random:11", "random:12,0.5,3" };
                for (int s = 0; s < (have_random ? 3 : 1); ++s) {
                    tsr_trace2_result res;
                    const int rc = tsr_trace2_replay_sim(r, NULL, scheds[s], &res);
                    CHECK(rc == 0 && res.ok, "%s: fresh-boot replay (%s): %s", tag, scheds[s] ? scheds[s] : "fixed", res.msg);
                }
            }
            if (r && i == 0) {                                             // corruptions, each detected
                tsr_trace2 x;
                tsr_trace2_init(&x);
                tsr_trace2_copy(&x, r);
                x.mem.expect[0].data[7] ^= 0x40;
                expect_fail(&x, TSR_TRACE2_CMP_EXPECT, "one expect byte");
                tsr_trace2_copy(&x, r);
                x.push[3].w[1] ^= 1u;                                      // the VOP's destination line
                expect_fail(&x, TSR_TRACE2_CMP_PUSH, "one push record");
                tsr_trace2_copy(&x, r);
                x.push[5].dep[0] = 0x00000000u;                            // the ST's dependency
                expect_fail(&x, TSR_TRACE2_CMP_PUSH, "one push dependency");
                tsr_trace2_copy(&x, r);
                x.unit[TSR_UNIT_VU].count = 3;
                expect_fail(&x, TSR_TRACE2_CMP_COUNTS, "one tickets count");
                tsr_trace2_copy(&x, r);
                x.unit[TSR_UNIT_MVM].err_code = TSR_ERR_ABORTED;
                expect_fail(&x, TSR_TRACE2_CMP_ERRREC, "one tickets error record");
                tsr_trace2_copy(&x, r);
                x.args[4] ^= 1u;                                           // a different K: another launch
                tsr_trace2_result res;
                CHECK(tsr_trace2_replay_sim(&x, NULL, NULL, &res) != 0, "a changed descriptor argument is detected (%s)", res.msg);
                tsr_trace2_clear(&x);
            }
            tsr_trace2_free(r);
            tsr_trace2_free(dr);
            free(dir);
            free(ddir);
        }
        for (int i = 0; i < 5; ++i) tsr_trace2_clear(&tr[i]);
        free(images);
        free(set);
    }
out:
    tsr_sim_destroy(sim);
    free(dump);
}

// The real ELF given to the capture: the trace's image is the file, its id EX §3.2's.
static void scenario_b(const char * elf_path) {
    tsr_sim * sim = make_sim(NULL);
    if (sim == NULL) return;
    FILE * f = fopen(elf_path, "rb");
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t * bytes = (uint8_t *)malloc((size_t)n);
    const size_t got = fread(bytes, 1, (size_t)n, f);
    fclose(f);
    CHECK(got == (size_t)n, "read the ELF");
    tsr_sim_capture_set_image(sim, bytes, (size_t)n);
    tsr_sim_capture_launch(sim, 1);
    tsr_rv_elf elf;
    if (tsr_rv_elf_read(elf_path, &elf) != 0) {
        free(bytes);
        tsr_sim_destroy(sim);
        return;
    }
    load_image(sim, &elf);
    tsr_rv_elf_free(&elf);
    static kcase kc[4];
    setup_cases(sim, kc);
    uint32_t d[8];
    kernel_desc(&kc[1], d);
    host h;
    memset(&h, 0, sizeof(h));
    h.sim = sim;
    tsr_sim_csr_write(sim, TSR_CSR_RV_BOOT, 0u);
    tsr_sim_csr_write(sim, TSR_CSR_CTRL, TSR_CTRL_RV_RUN);
    if (run_until(&h, 1)) {
        push_launch(&h, d);
        if (run_until(&h, 3)) {
            tsr_trace2 t;
            tsr_trace2_init(&t);
            CHECK(tsr_sim_capture_take(sim, &t) == 0, "B: one launch captured");
            uint64_t id = 0;
            char err[256];
            CHECK(t.image_size == (size_t)n && memcmp(t.image, bytes, (size_t)n) == 0, "B: the image is the ELF file");
            CHECK(tsr_trace2_image_id(bytes, (size_t)n, &id, err, sizeof(err)) == 0 && id == t.image_id, "B: its id");
            tsr_trace2_result res;
            CHECK(tsr_trace2_replay_sim(&t, NULL, NULL, &res) == 0, "B: replay booting the real ELF: %s", res.msg);
            tsr_trace2_clear(&t);
        }
    }
    free(bytes);
    tsr_sim_destroy(sim);
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: trace_v2_roundtrip <fw_issue.elf> <scratch base>\n");
        return 2;
    }
    FILE * f = fopen(argv[1], "rb");
    if (f == NULL) {
        printf("trace_v2_roundtrip: %s not built (no RISC-V compiler at configure time); skipped\n", argv[1]);
        return 77;
    }
    fclose(f);
    char * scratch = make_scratch(argv[2], "trace_v2_roundtrip");
    scenario_a(argv[1], scratch);
    scenario_b(argv[1]);
    if (!g_fail) rm_rf(scratch);
    else fprintf(stderr, "trace_v2_roundtrip: scratch kept in %s\n", scratch);
    free(scratch);
    return test_finish("trace_v2_roundtrip");
}
