// test_sim_rv_issue.c — W1a integration (ctest sim_rv_issue): a RISC-V program on tsr_rv drives the
// core model through the control window, launched by the host through the CSR port.
//
// The firmware (tests/rv_issue/: crt0.S, link.ld, fw_issue.c; built by the pinned xPack compiler,
// sw/sim/rv.cmake) is a minimal EX §4.7 runtime with one kernel (fw_issue.c has the details). The
// host side here does what the transport will (EX §3.3, §4.8):
//   - cold load through LOAD_ADDR / LOAD_DATA with the receive CRC and VERIFY, RV_BOOT, RV_RUN;
//   - two launches pushed before boot (served after HELLO, EX §4.7 R1), the rest after it;
//   - consume(): LCQ_WR, the records, LCQ_RD; the log ring likewise. The ring holds 8 records and
//     the program posts 13, so the runtime has to wait for LCQ_FREE (R8) while the host lags.
// Launches: PING; three "mvm_act" kernels (Q8_0 + SILU, Q4_K + TANH, Q8_0 + GELU) each pushing
// LD, QACT, MVM, VOP, RED (to s0) and ST with dependencies; one whose LD is out of range (TILE_ERR
// with the first error's ticket, ERR_INFO, ERR_W0 and ERR_PIDX, DEP_FAILED / ABORTED behind it,
// then the runtime's QCLEAR, R9); two BAD_DESC; a last PING. Every value is checked: DRAM outputs
// and s0 against direct references (tsr_qact, tsr_repack_row + tsr_dot_row, tsr_ref_vop,
// tsr_ref_red_row), the push indices and tickets the firmware reports, the log records, the
// RT_STATE / RT_INFO mirrors and the SW §4.8 debug view.
//
// Co-stepping (the placeholder for P1.8's scheduler, SW §4.4): the core's fixed time base
// (tsr_q.c) and the built-in RISC-V agent (tsr_bus.c) advance together, deterministically. The
// agent maps core time to RV cycles by F_RV / F_CORE (tsr_sim_opts.rv_ratio) and runs the hart in
// slices that end before the next core event; a control-window, core-window or SPM access ends
// its slice, so each such access sees the core state of its own instant. The scenario runs at the
// k4 clocks (166.7 / 276.3 MHz, the default) and at 1:1; SW §4.7's schedule-independent
// observables (records r0-r6, DRAM outputs, s0, log words) must be identical, only r7 (DURATION)
// and MTIME may differ.
//
// Usage: sim_rv_issue <fw_issue.elf>. Exit 77 (ctest "Skipped") when the ELF does not exist (the
// RISC-V compiler was not found at configure time; sw/ctrl/tools/install-toolchain.sh).
#include "test_sim_util.h"

#include "tsr_bus.h"
#include "tsr_ref_vu.h"
#include "tsr_rv.h"
#include "tsr_sim_int.h"

#define NCH       2u
#define N_LAUNCH  8u

// ---------------------------------------------------------------------------
// Host side of EX §3.3 (cold load) and EX §4.8 (launch / consume)
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
        if (g->filesz == 0) continue;                       // .bss only: crt0 zeroes it
        const uint32_t bytes = (g->filesz + 3u) & ~3u;      // the host pads to a whole word
        uint8_t * buf = (uint8_t *)calloc(bytes, 1);
        memcpy(buf, g->data, g->filesz);
        const uint32_t crc = crc32_zlib(buf, bytes);
        tsr_sim_csr_write(sim, TSR_CSR_LOAD_ADDR, g->paddr);
        CHECK(tsr_sim_csr_read(sim, TSR_CSR_LOAD_CRC) == 0u, "LOAD_CRC restarts");
        for (uint32_t i = 0; i < bytes; i += 4) tsr_sim_csr_write(sim, TSR_CSR_LOAD_DATA, get_u32_le(buf + i));
        CHECK(tsr_sim_csr_read(sim, TSR_CSR_LOAD_CRC) == crc, "segment 0x%08x: receive CRC", g->paddr);
        tsr_sim_csr_write(sim, TSR_CSR_LOAD_ADDR, g->paddr);
        tsr_sim_csr_write(sim, TSR_CSR_LOAD_COUNT, bytes / 4u);
        tsr_sim_csr_write(sim, TSR_CSR_LOAD_CTRL, TSR_LOAD_CTRL_VERIFY);
        for (int i = 0; i < 1000 && (tsr_sim_csr_read(sim, TSR_CSR_LOAD_STATUS) & TSR_LOAD_STATUS_VERIFY_BUSY); ++i)
            tsr_sim_advance(sim, 1000);
        CHECK(tsr_sim_csr_read(sim, TSR_CSR_LOAD_CRC) == crc, "segment 0x%08x: VERIFY CRC", g->paddr);
        free(buf);
    }
    const uint32_t st = tsr_sim_csr_read(sim, TSR_CSR_STATUS);
    CHECK(!(st & (TSR_STATUS_LOAD_REFUSED | TSR_STATUS_LOAD_TIMEOUT)), "loader status 0x%08x", st);
}

typedef struct { uint32_t r[8]; } rec_t;

typedef struct host {
    tsr_sim * sim;
    uint32_t  lcq_rd, log_rd;
    rec_t     rec[64];
    uint32_t  n_rec;
    uint32_t  log[256];
    uint32_t  n_log;
    uint32_t  max_lag;           // records the host left in the ring at most (back-pressure seen)
} host;

static void push_launch(host * h, const uint32_t d[8]) {
    for (uint32_t k = 0; k < 8u; ++k) tsr_sim_csr_write(h->sim, TSR_CSR_LQ_W(k), d[k]);
    tsr_sim_csr_write(h->sim, TSR_CSR_LQ_PUSH, 1u);
}

// EX §4.8 consume(): every record in [LCQ_RD, LCQ_WR), then the log words.
static void consume(host * h) {
    const uint32_t wr = tsr_sim_csr_read(h->sim, TSR_CSR_LCQ_WR);
    if (wr - h->lcq_rd > h->max_lag) h->max_lag = wr - h->lcq_rd;
    for (uint32_t k = h->lcq_rd; k != wr; ++k) {
        if (h->n_rec < 64u) {
            for (uint32_t j = 0; j < 8u; ++j)
                h->rec[h->n_rec].r[j] = tsr_sim_csr_read(h->sim, TSR_CSR_LCQ_REC(k % TSR_LCQ_DEPTH, j));
            ++h->n_rec;
        }
    }
    h->lcq_rd = wr;
    tsr_sim_csr_write(h->sim, TSR_CSR_LCQ_RD, wr);
    const uint32_t lwr = tsr_sim_csr_read(h->sim, TSR_CSR_LOG_WR);
    for (uint32_t k = h->log_rd; k != lwr; ++k)
        if (h->n_log < 256u) h->log[h->n_log++] = tsr_sim_csr_read(h->sim, TSR_CSR_LOG_RING(k % TSR_LOG_DEPTH));
    h->log_rd = lwr;
    tsr_sim_csr_write(h->sim, TSR_CSR_LOG_RD, lwr);
}

// Advances until n_want records were consumed. The host consumes at every `lag`-th LCQ event, when
// the ring is full and when the budget runs out: with a large lag the ring fills and the runtime
// waits for LCQ_FREE (EX §4.7 R8) until the host catches up.
static int run_until(host * h, uint32_t n_want, uint32_t lag) {
    uint32_t events = 0;
    for (int it = 0; it < 4000 && h->n_rec < n_want; ++it) {
        const uint32_t ev = tsr_sim_advance(h->sim, 200000);
        const uint32_t rt = tsr_sim_csr_read(h->sim, TSR_CSR_RT_STATE);
        if (TSR_GET(rt, TSR_RT_STATE_PHASE) == TSR_RT_STATE_PHASE_TRAPPED) {
            CHECK(0, "the firmware trapped: mcause %u (RT_INFO), pc 0x%08x", tsr_sim_csr_read(h->sim, TSR_CSR_RT_INFO),
                  tsr_sim_rv_pc(h->sim));
            return 0;
        }
        if (ev & TSR_SIM_EV_LCQ) ++events;
        // Consume at every lag-th record event, and whenever the ring is full (else nothing moves).
        const uint32_t pending = tsr_sim_csr_read(h->sim, TSR_CSR_LCQ_WR) - h->lcq_rd;
        if (((ev & TSR_SIM_EV_LCQ) && events % lag == 0) || pending >= TSR_LCQ_DEPTH || (ev & TSR_SIM_EV_BUDGET))
            consume(h);
    }
    consume(h);
    CHECK(h->n_rec >= n_want, "%u records, expected %u", h->n_rec, n_want);
    return h->n_rec >= n_want;
}

// ---------------------------------------------------------------------------
// Launch data and direct references
// ---------------------------------------------------------------------------
typedef struct kcase {
    uint32_t fmt, N, K, func;
    uint64_t x_beat, w_off, y_beat;     // x on ch0, weights on every channel, y on ch1
    int      bad_ld;                    // x handle out of range: LD fails with DRAM_RANGE
    // expected
    uint32_t qa_lines, y_lines;
    uint8_t  y[64 * 64];                // ceil(N/16) lines, tail words 0
    uint32_t s0;
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
    // Expected: QA image, MVM rows, VOP, RED SUM (EX §7, VU §5.2-§5.3).
    uint8_t * qa = (uint8_t *)malloc((size_t)kc->qa_lines * 64u);
    uint8_t * row = (uint8_t *)malloc((size_t)tsr_row_beats(fmt, K) * 64u);
    tsr_qact(kind, x, K, qa);
    uint32_t yw[1024] = {0}, zw[1024] = {0};
    for (uint32_t n = 0; n < N; ++n) {
        tsr_repack_row(fmt, blocks + row_bytes * n, K, row);
        yw[n] = f2u(tsr_dot_row(fmt, row, K, qa));
    }
    tsr_ref_vop(kc->func, TSR_VOP_CMODE_NONE, 1, N, zw, kc->y_lines, yw, kc->y_lines, NULL, 0, NULL);
    kc->s0 = tsr_ref_red_row(TSR_RED_KIND_SUM, TSR_RED_POST_NONE, zw, N, 0, 0);
    for (uint32_t i = 0; i < 16u * kc->y_lines; ++i) {
        kc->y[4 * i] = (uint8_t)zw[i]; kc->y[4 * i + 1] = (uint8_t)(zw[i] >> 8);
        kc->y[4 * i + 2] = (uint8_t)(zw[i] >> 16); kc->y[4 * i + 3] = (uint8_t)(zw[i] >> 24);
    }
    // Old content of the output region, so a missing ST shows.
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

// What SW §4.7 makes schedule-independent: records r0-r6, log words, outputs, s0.
typedef struct observed {
    uint32_t n_rec, rec[64][7];
    uint32_t n_log, log[256];
    uint64_t out_hash;
    uint32_t s0;
} observed;

// ---------------------------------------------------------------------------
// One complete scenario at a given RV / core clock ratio
// ---------------------------------------------------------------------------
static void scenario(const char * elf_path, double ratio, observed * ob) {
    char tag[64];
    tsr_config cfg = TSR_CONFIG_U200_V1;
    cfg.num_ch = NCH;
    cfg.ch_bytes = 1ull << 30;
    cfg.caps = TSR_CAPS_VU;
    tsr_sim_opts o;
    memset(&o, 0, sizeof(o));
    o.rv_ratio = ratio;
    tsr_sim * sim = tsr_sim_create_opts(&cfg, &o);
    CHECK(sim != NULL, "create (rv_ratio %.3f)", ratio);
    if (sim == NULL) return;
    snprintf(tag, sizeof(tag), "F_RV/F_CORE %.4f", tsr_sim_rv_ratio(sim));
    CHECK(tsr_sim_rv_agent(sim) != NULL, "%s: the simulator has its RISC-V agent", tag);

    tsr_rv_elf elf;
    if (tsr_rv_elf_read(elf_path, &elf) != 0) {
        CHECK(0, "read %s", elf_path);
        tsr_sim_destroy(sim);
        return;
    }
    load_image(sim, &elf);
    tsr_rv_elf_free(&elf);

    // Launch data. Kernel cases: seq 1, 2, 3 (bad LD), 5.
    rng_t r = { 0x5EED1234u };
    static kcase kc[4];
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
    uint32_t desc[N_LAUNCH][8];
    memset(desc, 0, sizeof(desc));
    desc[0][0] = TSR_DESC_TYPE_PING; desc[0][1] = 0xC0FFEE01u;
    kernel_desc(&kc[0], desc[1]);
    kernel_desc(&kc[1], desc[2]);
    kernel_desc(&kc[2], desc[3]);
    desc[4][0] = 0x07u;                                                          // unknown TYPE
    kernel_desc(&kc[3], desc[5]);
    kernel_desc(&kc[0], desc[6]); desc[6][0] = TSR_DESC_TYPE_LAUNCH | TSR_PUT(TSR_DESC_KERNEL, 3u);   // no kernel 3
    desc[7][0] = TSR_DESC_TYPE_PING; desc[7][1] = 0xC0FFEE07u;

    host h;
    memset(&h, 0, sizeof(h));
    h.sim = sim;
    const uint32_t seq0 = tsr_sim_csr_read(sim, TSR_CSR_LQ_SEQ);
    CHECK(seq0 == 0u, "%s: LQ_SEQ 0 after create", tag);
    push_launch(&h, desc[0]);                                // queued before boot (R1)
    push_launch(&h, desc[1]);
    tsr_sim_csr_write(sim, TSR_CSR_LOG_CFG, TSR_LOG_CFG_ENABLE | TSR_PUT(TSR_LOG_CFG_LEVEL, TSR_LOG_HDR_LEVEL_INFO));
    tsr_sim_csr_write(sim, TSR_CSR_RV_BOOT, 0u);
    tsr_sim_csr_write(sim, TSR_CSR_CTRL, TSR_CTRL_RV_RUN);
    if (!run_until(&h, 1, 1)) goto out;
    CHECK(TSR_GET(h.rec[0].r[0], TSR_LCQ_R0_TYPE) == TSR_LCQ_R0_TYPE_HELLO, "%s: HELLO first (0x%08x)", tag, h.rec[0].r[0]);
    CHECK(h.rec[0].r[1] == 0u && h.rec[0].r[2] == 0x5EED0001u && h.rec[0].r[3] == 0x0000F00Du &&
          h.rec[0].r[4] == 0x00010001u && h.rec[0].r[5] == 1u, "%s: HELLO r1-r5", tag);
    for (uint32_t i = 2; i < N_LAUNCH; ++i) push_launch(&h, desc[i]);
    // 13 records in all; the host consumes only when the ring is full (or nothing happens).
    if (!run_until(&h, 13, 1000)) goto out;
    CHECK(h.max_lag == TSR_LCQ_DEPTH, "%s: the ring was full at least once (max %u)", tag, h.max_lag);
    // Let the runtime finish its last R9 / R10 and sleep.
    tsr_sim_advance(sim, 100000);
    consume(&h);

    {
        // Record stream: HELLO, PONG 0, EV 1, DONE 1, EV 2, DONE 2, EV 3, DONE 3, DONE 4, EV 5, DONE 5, DONE 6, PONG 7.
        static const uint32_t want_type[13] = { 0x80, 2, 0xC0, 1, 0xC0, 1, 0xC0, 1, 1, 0xC0, 1, 1, 2 };
        static const uint32_t want_seq[13] = { 0, 0, 1, 1, 2, 2, 3, 3, 4, 5, 5, 6, 7 };
        CHECK(h.n_rec == 13u, "%s: %u records", tag, h.n_rec);
        for (uint32_t i = 0; i < 13u && i < h.n_rec; ++i) {
            const uint32_t * q = h.rec[i].r;
            CHECK(TSR_GET(q[0], TSR_LCQ_R0_TYPE) == want_type[i], "%s: record %u type 0x%02x, expected 0x%02x", tag, i,
                  TSR_GET(q[0], TSR_LCQ_R0_TYPE), want_type[i]);
            if (i > 0) CHECK(TSR_GET(q[0], TSR_LCQ_R0_SEQ) == want_seq[i] && q[1] == want_seq[i], "%s: record %u seq", tag, i);
        }
        const rec_t * R = h.rec;
        // PONGs echo ARG0.
        CHECK(R[1].r[2] == 0xC0FFEE01u && R[12].r[2] == 0xC0FFEE07u, "%s: PONG ARG0 echo", tag);
        CHECK(TSR_GET(R[1].r[0], TSR_LCQ_R0_STATUS) == 0u && TSR_GET(R[12].r[0], TSR_LCQ_R0_STATUS) == 0u, "%s: PONG status", tag);
        // BAD_DESC answers.
        for (uint32_t i = 8; i <= 11; i += 3)
            CHECK(TSR_GET(R[i].r[0], TSR_LCQ_R0_STATUS) == TSR_LCQ_R0_STATUS_BAD_DESC && R[i].r[3] == TSR_TICKET_NONE,
                  "%s: record %u BAD_DESC", tag, i);
        // Kernel launches: record index of the event, its DONE, the kase, first push index, DMA seq of the ST.
        const uint32_t ev_i[4] = { 2, 4, 6, 9 }, pidx0[4] = { 1, 8, 15, 23 }, st_seq[4] = { 1, 3, 5, 7 };
        uint32_t s0_prev = 0;
        for (uint32_t k = 0; k < 4; ++k) {
            const uint32_t * ev = R[ev_i[k]].r, * dn = R[ev_i[k] + 1].r;
            const kcase * c = &kc[k];
            const uint32_t s0_want = c->bad_ld ? s0_prev : c->s0;
            CHECK(ev[2] == s0_want, "%s: kernel %u: s0 0x%08x (core window), reference 0x%08x", tag, k, ev[2], s0_want);
            CHECK(ev[3] == pidx0[k] && ev[4] == 6u && ev[5] == (TSR_PUT(TSR_TICKET_UNIT, TSR_UNIT_DMA) | st_seq[k]),
                  "%s: kernel %u: first push index %u, ops %u, ST ticket 0x%08x", tag, k, ev[3], ev[4], ev[5]);
            CHECK(dn[2] == 6u, "%s: kernel %u: RET %u", tag, k, dn[2]);
            CHECK(dn[7] > 0u, "%s: kernel %u: DURATION", tag, k);
            const uint8_t * got = tsr_sim_dram(sim, 1) + c->y_beat * 64u;
            if (!c->bad_ld) {
                CHECK(TSR_GET(dn[0], TSR_LCQ_R0_STATUS) == TSR_LCQ_R0_STATUS_OK && TSR_GET(dn[0], TSR_LCQ_R0_ERR_CODE) == 0u &&
                      dn[3] == TSR_TICKET_NONE && dn[4] == 0u && dn[5] == 0u && dn[6] == 0u, "%s: kernel %u: DONE OK (0x%08x)", tag, k, dn[0]);
                CHECK(memcmp(got, c->y, (size_t)c->y_lines * 64u) == 0, "%s: kernel %u: DRAM output = act(W x) reference", tag, k);
                s0_prev = c->s0;
            } else {
                // First error = the LD (smallest push index): DRAM_RANGE, ERR_INFO = its beat, ERR_W0 = its w0.
                CHECK(TSR_GET(dn[0], TSR_LCQ_R0_STATUS) == TSR_LCQ_R0_STATUS_TILE_ERR &&
                      TSR_GET(dn[0], TSR_LCQ_R0_ERR_CODE) == TSR_ERR_DRAM_RANGE, "%s: bad LD: TILE_ERR / DRAM_RANGE (0x%08x)", tag, dn[0]);
                CHECK(dn[3] == (TSR_PUT(TSR_TICKET_UNIT, TSR_UNIT_DMA) | 4u) && dn[4] == 0x0FFFFFF0u &&
                      dn[5] == tsr_w0(TSR_OP_LD, 0, 0, 0) && dn[6] == pidx0[k],
                      "%s: bad LD: ticket 0x%08x info 0x%08x w0 0x%08x pidx %u", tag, dn[3], dn[4], dn[5], dn[6]);
                uint8_t ee[64 * 64];
                memset(ee, 0xEE, sizeof(ee));
                CHECK(memcmp(got, ee, (size_t)c->y_lines * 64u) == 0, "%s: bad LD: the aborted ST wrote nothing", tag);
            }
        }
        // Log: one INFO / EVENT record per kernel run, payload N, K, s0.
        CHECK(h.n_log == 16u, "%s: %u log words", tag, h.n_log);
        for (uint32_t k = 0; k < 4 && 4 * k + 3 < h.n_log; ++k) {
            const uint32_t hd = h.log[4 * k];
            const uint32_t seq = ev_i[k] == 9 ? 5u : 1u + k;
            CHECK(TSR_GET(hd, TSR_LOG_HDR_N) == 3u && TSR_GET(hd, TSR_LOG_HDR_LEVEL) == TSR_LOG_HDR_LEVEL_INFO &&
                  TSR_GET(hd, TSR_LOG_HDR_SOURCE) == TSR_LOG_HDR_SOURCE_KERNEL &&
                  TSR_GET(hd, TSR_LOG_HDR_TYPE) == TSR_LOG_HDR_TYPE_EVENT && TSR_GET(hd, TSR_LOG_HDR_LSEQ) == seq &&
                  TSR_GET(hd, TSR_LOG_HDR_SYNC) == TSR_LOG_HDR_SYNC_MAGIC, "%s: log header %u 0x%08x", tag, k, hd);
            CHECK(h.log[4 * k + 1] == kc[k].N && h.log[4 * k + 2] == kc[k].K && h.log[4 * k + 3] == R[ev_i[k]].r[2],
                  "%s: log payload %u", tag, k);
        }
        // Device state at the end: idle runtime asleep in WFI, queues idle and clean, counters.
        CHECK(TSR_GET(tsr_sim_csr_read(sim, TSR_CSR_RT_STATE), TSR_RT_STATE_PHASE) == TSR_RT_STATE_PHASE_IDLE, "%s: RT_STATE IDLE", tag);
        CHECK(tsr_sim_csr_read(sim, TSR_CSR_RT_INFO) == 6u, "%s: RT_INFO mirror", tag);
        const uint32_t st = tsr_sim_csr_read(sim, TSR_CSR_STATUS);
        CHECK((st & TSR_STATUS_QIDLE) && (st & TSR_STATUS_RV_SLEEP) && !(st & TSR_STATUS_QERR), "%s: STATUS 0x%08x", tag, st);
        CHECK(!(st & (TSR_STATUS_LQ_OVERFLOW | TSR_STATUS_RING_RD_BAD | TSR_STATUS_LOG_DROP)), "%s: no sticky errors (0x%08x)", tag, st);
        CHECK(tsr_sim_csr_read(sim, TSR_CSR_LQ_TAKEN) == N_LAUNCH && tsr_sim_csr_read(sim, TSR_CSR_LCQ_ANSWERED) == N_LAUNCH,
              "%s: TAKEN = ANSWERED = %u", tag, N_LAUNCH);
        // 1 boot QCLEAR, 4 x (6 ops + FENCE), 1 QCLEAR after the error.
        CHECK(tsr_sim_cw_read(sim, TSR_CW_ISSUE_PIDX) == 30u, "%s: ISSUE_PIDX %u", tag, tsr_sim_cw_read(sim, TSR_CW_ISSUE_PIDX));
        CHECK(tsr_sim_rv_instret(sim) > 0u && tsr_sim_rv_instret(sim) == tsr_rv_retired(tsr_rv_agent_rv(tsr_sim_rv_agent(sim))),
              "%s: debug view instret", tag);
        CHECK(f2u(tsr_sim_sreg(sim)[0]) == kc[3].s0, "%s: s0 of the last kernel", tag);

        ob->n_rec = h.n_rec;
        for (uint32_t i = 0; i < h.n_rec; ++i) memcpy(ob->rec[i], h.rec[i].r, 7 * sizeof(uint32_t));
        ob->n_log = h.n_log;
        memcpy(ob->log, h.log, h.n_log * sizeof(uint32_t));
        uint64_t hs = 0;
        for (uint32_t k = 0; k < 4; ++k) hs = hs * 31u + fnv64(tsr_sim_dram(sim, 1) + kc[k].y_beat * 64u, (size_t)kc[k].y_lines * 64u);
        ob->out_hash = hs;
        ob->s0 = f2u(tsr_sim_sreg(sim)[0]);
        printf("sim_rv_issue %s: %llu RV instructions, MTIME %llu core cycles, %u records, %u log words\n", tag,
               (unsigned long long)tsr_sim_rv_instret(sim), (unsigned long long)tsr_sim_now(sim), h.n_rec, h.n_log);
    }
out:
    tsr_sim_destroy(sim);
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: sim_rv_issue <fw_issue.elf>\n");
        return 2;
    }
    FILE * f = fopen(argv[1], "rb");
    if (f == NULL) {
        printf("sim_rv_issue: %s not built (no RISC-V compiler at configure time); skipped. "
               "Install it with sw/ctrl/tools/install-toolchain.sh and reconfigure.\n", argv[1]);
        return 77;
    }
    fclose(f);
    static observed a, b;
    memset(&a, 0, sizeof(a));
    memset(&b, 0, sizeof(b));
    scenario(argv[1], 0.0, &a);           // default ratio: 166.7 / 276.3 (SW §4.9)
    scenario(argv[1], 1.0, &b);
    CHECK(a.n_rec == b.n_rec && memcmp(a.rec, b.rec, sizeof(a.rec)) == 0, "records r0-r6 equal at both clock ratios");
    CHECK(a.n_log == b.n_log && memcmp(a.log, b.log, sizeof(a.log)) == 0, "log words equal at both clock ratios");
    CHECK(a.out_hash == b.out_hash && a.s0 == b.s0, "DRAM outputs and s0 equal at both clock ratios");
    return test_finish("sim_rv_issue");
}
