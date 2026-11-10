// tsr_devrun.c — launch targets (a device behind a transport, or the in-process shadow simulator)
// and trace replays through them, for tsr-selftest and tsr-trace-replay --device. The
// interface and the comparison rules are in tsr_devrun.h; the protocol references are EX §3-§5
// (docs/design/01-execution-model.md) and SW §5.4, §7.6, §7.7 (docs/design/06-software-verification.md).
// Decisions:
//   - The shadow is a fresh tsr_sim with the device's configuration (tsr_tgt_cfg: CFG*/CAPS, the
//     inner simulator's tsr_config for sim / mock, Q_DEPTH, RV_CFG), booted by the host loader of
//     fw_sim_host.hpp (receive CRC, VERIFY, HELLO with the image id); it takes the device's LOG_CFG
//     after the boot, so both write the same log records. A boot with another image re-creates it.
//   - Queue counters are read at quiescence (RT_STATE IDLE, QIDLE, RV_SLEEP, TAKEN = ANSWERED), so
//     the runtime's R9 QCLEAR after an erroring launch falls inside the launch on both targets.
//   - A trace's SPM buffer for the replay kernel is chosen as the simulator's kernel route chooses
//     it (sw/sim/tsr_trace.c kr_spm_buffer: the highest free run below the runtime argument range
//     outside every SPM extent of the trace's accepted ops), so a device replay pushes exactly what
//     `tsr-trace-replay --replay-image` pushes on the simulator.
#define _GNU_SOURCE

#include "tsr_devrun.h"

#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "tsr_sim_int.h"   // the checker classes (tsr_sim_op_handler, extents): fit test, SPM buffer

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#include "fw_sim_host.hpp"   // the shadow's host loader and LQ / LCQ handling (plain C)
#pragma GCC diagnostic pop

enum { TGT_DEVICE = 0, TGT_SHADOW = 1 };

struct tsr_tgt {
    int          kind;
    char         name[48];
    tsr_tgt_cfg  cfg;
    // device
    tsr_transport * t;
    tsr_image *  img;               // the image this target booted (device)
    // shadow
    tsr_sim *    sim;
    fwh          h;
    int          h_open;
    int          dirty;             // a launch did not finish: the next boot re-creates the simulator
    char *       image_path;        // the last image booted (tsr_tgt_ready)
    uint32_t     threads;
    // both
    uint64_t     image_id;
    tsr_tgt_run * cur;              // the launch collecting events (device callback)
    tsr_sim *    check_sim;         // a simulator of cfg for the checker (lazily created)
};

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static void seterr(char * err, size_t n, const char * fmt, ...) __attribute__((format(printf, 3, 4)));
static void seterr(char * err, size_t n, const char * fmt, ...) {
    if (err == NULL || n == 0) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, n, fmt, ap);
    va_end(ap);
}

static void appendf(char * msg, size_t n, const char * fmt, ...) __attribute__((format(printf, 3, 4)));
static void appendf(char * msg, size_t n, const char * fmt, ...) {
    if (msg == NULL || n == 0) return;
    const size_t used = strnlen(msg, n);
    if (used + 1 >= n) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg + used, n - used, fmt, ap);
    va_end(ap);
}

static inline void wr32le(uint8_t * p, uint32_t v) {
    p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24);
}

// ---------------------------------------------------------------------------
// Targets
// ---------------------------------------------------------------------------
static void dev_event(void * user, const tsr_record * rec) {
    tsr_tgt * g = (tsr_tgt *)user;
    tsr_tgt_run * r = g->cur;
    const uint32_t type = TSR_GET(rec->w[0], TSR_LCQ_R0_TYPE);
    if (r == NULL || type < TSR_LCQ_R0_TYPE_EVENT_FIRST) return;
    if (r->n_ev < TSR_TGT_MAX_EVENTS) memcpy(r->ev[r->n_ev], rec->w, sizeof(rec->w));
    r->n_ev++;
}

tsr_tgt * tsr_tgt_transport(tsr_transport * t) {
    tsr_tgt * g = (tsr_tgt *)calloc(1, sizeof(*g));
    if (g == NULL) return NULL;
    g->kind = TGT_DEVICE;
    g->t = t;
    snprintf(g->name, sizeof(g->name), "device (%s)", tsr_transport_name(t));
    tsr_tgt_cfg_of_transport(t, &g->cfg);
    tsr_transport_set_event_cb(t, dev_event, g);
    return g;
}

static tsr_sim * make_sim(const tsr_tgt_cfg * c, uint32_t threads) {
    tsr_sim_opts o;
    memset(&o, 0, sizeof(o));
    o.threads = threads;
    o.sched = "fixed";
    o.rv_ratio = c->rv_ratio >= 0.05 && c->rv_ratio <= 4.0 ? c->rv_ratio : 0.0;
    o.q_depth = c->q_depth;
    o.itcm_bytes = c->itcm_bytes;
    o.dtcm_bytes = c->dtcm_bytes;
    return tsr_sim_create_opts(&c->core, &o);
}

tsr_tgt * tsr_tgt_shadow(const tsr_tgt_cfg * c, uint32_t threads) {
    if (threads == 0) {
        const char * e = getenv("TSR_SIM_THREADS");
        threads = e && *e ? (uint32_t)strtoul(e, NULL, 0) : 1u;
        if (threads == 0 || threads > 64) threads = 1;
    }
    tsr_sim * sim = make_sim(c, threads);
    if (sim == NULL) return NULL;
    tsr_tgt * g = (tsr_tgt *)calloc(1, sizeof(*g));
    if (g == NULL) {
        tsr_sim_destroy(sim);
        return NULL;
    }
    g->kind = TGT_SHADOW;
    g->cfg = *c;
    g->sim = sim;
    g->threads = threads;
    snprintf(g->name, sizeof(g->name), "shadow");
    return g;
}

void tsr_tgt_free(tsr_tgt * g) {
    if (g == NULL) return;
    if (g->kind == TGT_DEVICE) {
        tsr_transport_set_event_cb(g->t, NULL, NULL);
        if (g->img) tsr_image_close(g->img);
    }
    if (g->h_open) fwh_close(&g->h);
    if (g->sim) tsr_sim_destroy(g->sim);
    free(g->image_path);
    if (g->check_sim) tsr_sim_destroy(g->check_sim);
    free(g);
}

const char * tsr_tgt_name(const tsr_tgt * g) { return g->name; }
const tsr_config * tsr_tgt_config(const tsr_tgt * g) { return &g->cfg.core; }
tsr_sim * tsr_tgt_sim(tsr_tgt * g) { return g->kind == TGT_SHADOW ? g->sim : NULL; }
tsr_transport * tsr_tgt_transport_of(tsr_tgt * g) { return g->kind == TGT_DEVICE ? g->t : NULL; }
uint64_t tsr_tgt_image_id(const tsr_tgt * g) { return g->image_id; }

// A simulator of the target's configuration for the checker (never booted).
static tsr_sim * check_sim_of(tsr_tgt * g) {
    if (g->kind == TGT_SHADOW) return g->sim;
    if (g->check_sim == NULL) g->check_sim = make_sim(&g->cfg, 1);
    return g->check_sim;
}

int tsr_tgt_write(tsr_tgt * g, uint32_t ch, uint64_t off, const void * src, size_t n) {
    if (g->kind == TGT_DEVICE) return tsr_transport_write(g->t, ch, off, src, n);
    return tsr_sim_dram_write(g->sim, ch, off, src, n);
}

int tsr_tgt_read(tsr_tgt * g, uint32_t ch, uint64_t off, void * dst, size_t n) {
    if (g->kind == TGT_DEVICE) return tsr_transport_read(g->t, ch, off, dst, n);
    return tsr_sim_dram_read(g->sim, ch, off, dst, n);
}

uint32_t tsr_tgt_csr(tsr_tgt * g, uint32_t off) {
    return g->kind == TGT_DEVICE ? tsr_transport_csr_read(g->t, off) : tsr_sim_csr_read(g->sim, off);
}

void tsr_tgt_csr_write(tsr_tgt * g, uint32_t off, uint32_t val) {
    if (g->kind == TGT_DEVICE) tsr_transport_csr_write(g->t, off, val);
    else tsr_sim_csr_write(g->sim, off, val);
}

int tsr_tgt_boot(tsr_tgt * g, const char * elf_path, char * err, size_t errlen) {
    if (g->kind == TGT_DEVICE) {
        tsr_image * img = NULL;
        const int rc = tsr_image_open(elf_path, &img);
        if (rc != 0 || img == NULL) {
            seterr(err, errlen, "%s: cannot open the image %s (%s)", g->name, elf_path, tsr_transport_strerror(rc));
            return -1;
        }
        const tsr_image * cur = tsr_transport_image(g->t);
        if (cur != NULL && tsr_image_id(cur) == tsr_image_id(img)) {   // running already
            if (g->img) tsr_image_close(g->img);
            g->img = img;
            g->image_id = tsr_image_id(img);
            return 0;
        }
        const int br = tsr_transport_boot(g->t, img);
        if (br != 0) {
            seterr(err, errlen, "%s: boot of %s failed: %s", g->name, elf_path, tsr_transport_strerror(br));
            tsr_image_close(img);
            return -1;
        }
        if (g->img) tsr_image_close(g->img);
        g->img = img;
        g->image_id = tsr_image_id(img);
        return 0;
    }
    // shadow: same image already running -> nothing; else a fresh simulator and the host loader.
    size_t n = 0;
    uint8_t * bytes = fwh_read_file(elf_path, &n);
    uint64_t id = 0;
    char why[256];
    if (bytes == NULL || tsr_trace2_image_id(bytes, n, &id, why, sizeof(why)) != 0) {
        seterr(err, errlen, "shadow: %s is not a readable image", elf_path);
        free(bytes);
        return -1;
    }
    free(bytes);
    if (g->h_open && g->image_id == id && !g->dirty) return 0;
    if (g->h_open || g->dirty) {
        fwh_close(&g->h);
        g->h_open = 0;
        tsr_sim_destroy(g->sim);
        g->sim = make_sim(&g->cfg, g->threads);
        if (g->sim == NULL) {
            seterr(err, errlen, "shadow: cannot re-create the simulator");
            return -1;
        }
    }
    g->dirty = 0;
    if (fwh_open(&g->h, g->sim, elf_path) != 0) {
        seterr(err, errlen, "shadow: boot of %s failed (messages above)", elf_path);
        fwh_close(&g->h);
        return -1;
    }
    g->h_open = 1;
    g->image_id = g->h.image_id;
    if (g->image_path != elf_path) {
        free(g->image_path);
        g->image_path = strdup(elf_path);
    }
    return 0;
}

int tsr_tgt_ready(tsr_tgt * g, char * err, size_t errlen) {
    if (g->kind != TGT_SHADOW || !g->dirty) return 0;
    if (g->image_path == NULL) {
        seterr(err, errlen, "shadow: no image to restart with");
        return -1;
    }
    char * p = strdup(g->image_path);
    const int rc = p ? tsr_tgt_boot(g, p, err, errlen) : -1;
    free(p);
    return rc;
}

int tsr_tgt_kernel(const tsr_tgt * g, const char * name) {
    if (g->kind == TGT_DEVICE) return g->img ? tsr_image_kernel(g->img, name) : -1;
    return g->h_open ? fwh_kernel(&g->h, name) : -1;
}

static int quiet_now(tsr_tgt * g) {
    const uint32_t st = tsr_tgt_csr(g, TSR_CSR_STATUS);
    const uint32_t rt = tsr_tgt_csr(g, TSR_CSR_RT_STATE);
    return TSR_GET(rt, TSR_RT_STATE_PHASE) == TSR_RT_STATE_PHASE_IDLE && (st & TSR_STATUS_QIDLE) && (st & TSR_STATUS_RV_SLEEP) &&
           tsr_tgt_csr(g, TSR_CSR_LQ_TAKEN) == tsr_tgt_csr(g, TSR_CSR_LCQ_ANSWERED) &&
           tsr_tgt_csr(g, TSR_CSR_LQ_SEQ) == tsr_tgt_csr(g, TSR_CSR_LQ_TAKEN);
}

int tsr_tgt_quiesce(tsr_tgt * g) {
    if (g->kind == TGT_SHADOW) {
        for (int i = 0; i < 4096; ++i) {
            if (quiet_now(g)) return 0;
            tsr_sim_advance(g->sim, 1u << 16);
        }
        return -1;
    }
    // Device: on the simulator kinds a STATUS read runs the model to the next host-visible event or
    // to the RISC-V's WFI (EX §10.2), so this settles at once; elsewhere poll for up to 2 s.
    const uint64_t t0 = now_ns();
    for (uint32_t i = 0;; ++i) {
        if (quiet_now(g)) return 0;
        if (now_ns() - t0 > 2000000000ull && i > 64) return -1;
        if (i > 16) {
            const struct timespec ts = { 0, 20000 };
            nanosleep(&ts, NULL);
        }
    }
}

static void read_q(tsr_tgt * g, uint32_t * done, uint32_t * issued, uint32_t * pidx) {
    for (uint32_t u = 0; u < TSR_NUM_UNITS; ++u) {
        if (done) done[u] = tsr_tgt_csr(g, TSR_CSR_Q_DONE(u));
        if (issued) issued[u] = tsr_tgt_csr(g, TSR_CSR_Q_ISSUED(u));
    }
    if (pidx) *pidx = tsr_tgt_csr(g, TSR_CSR_Q_PIDX);
}

int tsr_tgt_log_drain(tsr_tgt * g, uint32_t ** words, size_t * n) {
    *words = NULL;
    *n = 0;
    if (g->kind == TGT_SHADOW) {
        if (!g->h_open) return 0;
        fwh_consume(&g->h, 0, 0);
        if (g->h.n_log) {
            *words = (uint32_t *)malloc(g->h.n_log * sizeof(uint32_t));
            if (*words == NULL) return -1;
            memcpy(*words, g->h.log, g->h.n_log * sizeof(uint32_t));
            *n = g->h.n_log;
        }
        g->h.n_log = 0;
        return 0;
    }
    size_t cap = 1024, got = 0;
    uint32_t * w = NULL;
    for (;;) {
        uint32_t * nw = (uint32_t *)realloc(w, cap * sizeof(uint32_t));
        if (nw == NULL) {
            free(w);
            return -1;
        }
        w = nw;
        size_t k = 0;
        if (tsr_transport_log_drain(g->t, w + got, cap - got, &k) != 0) {
            free(w);
            return -1;
        }
        got += k;
        if (got < cap) break;
        cap *= 2;
    }
    if (got == 0) {
        free(w);
        w = NULL;
    }
    *words = w;
    *n = got;
    return 0;
}

void tsr_tgt_run_free(tsr_tgt_run * r) {
    free(r->log);
    r->log = NULL;
    r->n_log = 0;
}

int tsr_tgt_launch(tsr_tgt * g, uint32_t type, uint32_t flags, uint32_t kernel, const uint32_t args[7], tsr_tgt_run * r) {
    memset(r, 0, sizeof(*r));
    if (tsr_tgt_quiesce(g) != 0) {
        r->rc = -1;
        return r->rc;
    }
    read_q(g, r->q_done0, r->q_issued0, &r->q_pidx0);
    {   // the log words before this launch belong to no launch of the caller's
        uint32_t * w = NULL;
        size_t n = 0;
        tsr_tgt_log_drain(g, &w, &n);
        free(w);
    }
    const uint64_t t0 = now_ns();
    if (g->kind == TGT_DEVICE) {
        tsr_launch l;
        memset(&l, 0, sizeof(l));
        l.type = (uint8_t)type;
        l.flags = (uint8_t)flags;
        l.kernel = (uint16_t)kernel;
        for (int i = 0; i < 7; ++i) l.arg[i] = args ? args[i] : 0u;
        tsr_record rec;
        memset(&rec, 0, sizeof(rec));
        g->cur = r;
        const int rc = tsr_transport_call(g->t, &l, TSR_TIMEOUT_DEFAULT, &rec);
        g->cur = NULL;
        r->rc = rc;
        memcpy(r->rec, rec.w, sizeof(r->rec));
        r->seq = rec.w[1];
    } else {
        fwh * h = &g->h;
        h->n_ev = 0;
        h->n_log = 0;
        memset(h->done, 0, sizeof(h->done));
        const uint32_t seq = tsr_sim_csr_read(g->sim, TSR_CSR_LQ_SEQ);
        const uint32_t w0 = TSR_PUT(TSR_DESC_TYPE, type) | ((flags & 0xFFu) << 8) | TSR_PUT(TSR_DESC_KERNEL, kernel);
        tsr_sim_csr_write(g->sim, TSR_CSR_LQ_W(0), w0);
        for (uint32_t i = 0; i < 7u; ++i) tsr_sim_csr_write(g->sim, TSR_CSR_LQ_W(1 + i), args ? args[i] : 0u);
        tsr_sim_csr_write(g->sim, TSR_CSR_LQ_PUSH, 1u);
        r->rc = fwh_run(h, 1, seq, 1ull << 36, "shadow launch") == 0 ? 0 : -1;
        if (r->rc) g->dirty = 1;
        memcpy(r->rec, h->done, sizeof(r->rec));
        r->seq = seq;
        for (uint32_t i = 0; i < h->n_ev; ++i) {
            if (TSR_GET(h->ev[i][0], TSR_LCQ_R0_TYPE) < TSR_LCQ_R0_TYPE_EVENT_FIRST) continue;
            if (r->n_ev < TSR_TGT_MAX_EVENTS) memcpy(r->ev[r->n_ev], h->ev[i], 8 * sizeof(uint32_t));
            r->n_ev++;
        }
        h->n_ev = 0;
    }
    r->wall_ns = now_ns() - t0;
    if (tsr_tgt_quiesce(g) != 0 && r->rc == 0) r->rc = -1;
    read_q(g, r->q_done1, NULL, NULL);
    tsr_tgt_log_drain(g, &r->log, &r->n_log);
    return r->rc;
}

// ---------------------------------------------------------------------------
// Comparison of two runs (SW §7.6)
// ---------------------------------------------------------------------------
static uint32_t rel_ticket(uint32_t abs, const uint32_t * issued0) {
    const uint32_t u = abs >> 28;
    return u < TSR_NUM_UNITS ? (u << 28) | ((abs - issued0[u]) & 0x0FFFFFFFu) : abs;
}

uint32_t tsr_tgt_compare(const tsr_tgt_run * a, const tsr_tgt_run * b, char * msg, size_t msglen) {
    uint32_t nd = 0;
    if (msg && msglen) msg[0] = '\0';
    if (a->rc != b->rc) {
        appendf(msg, msglen, "launch result %d vs %d; ", a->rc, b->rc);
        nd++;
    }
    const uint32_t ra0 = a->rec[0] & 0x00FFFFFFu, rb0 = b->rec[0] & 0x00FFFFFFu;
    if (ra0 != rb0 || a->rec[2] != b->rec[2] || a->rec[4] != b->rec[4] || a->rec[5] != b->rec[5]) {
        appendf(msg, msglen, "answer r0[23:0] 0x%06x vs 0x%06x, RET 0x%08x vs 0x%08x, ERR_INFO 0x%08x vs 0x%08x, ERR_W0 0x%08x vs 0x%08x; ",
                ra0, rb0, a->rec[2], b->rec[2], a->rec[4], b->rec[4], a->rec[5], b->rec[5]);
        nd++;
    }
    const uint32_t type = TSR_GET(a->rec[0], TSR_LCQ_R0_TYPE);
    if (type == TSR_LCQ_R0_TYPE_DONE) {
        const uint32_t ta = rel_ticket(a->rec[3], a->q_issued0), tb = rel_ticket(b->rec[3], b->q_issued0);
        const uint32_t pa = a->rec[3] == TSR_TICKET_NONE ? a->rec[6] : a->rec[6] - a->q_pidx0;
        const uint32_t pb = b->rec[3] == TSR_TICKET_NONE ? b->rec[6] : b->rec[6] - b->q_pidx0;
        if (ta != tb || pa != pb) {
            appendf(msg, msglen, "first error ticket (relative) 0x%08x vs 0x%08x, push index (relative) %u vs %u; ", ta, tb, pa, pb);
            nd++;
        }
    }
    for (uint32_t u = 0; u < TSR_NUM_UNITS; ++u) {
        const uint32_t ca = (a->q_done1[u] - a->q_done0[u]) & 0x0FFFFFFFu, cb = (b->q_done1[u] - b->q_done0[u]) & 0x0FFFFFFFu;
        if (ca != cb) {
            appendf(msg, msglen, "unit %u completed %u vs %u entries; ", u, ca, cb);
            nd++;
        }
    }
    if (a->n_ev != b->n_ev) {
        appendf(msg, msglen, "%u vs %u kernel events; ", a->n_ev, b->n_ev);
        nd++;
    } else {
        for (uint32_t i = 0; i < a->n_ev && i < TSR_TGT_MAX_EVENTS; ++i) {
            const uint32_t * x = a->ev[i], * y = b->ev[i];
            int same = (x[0] & 0x00FFFFFFu) == (y[0] & 0x00FFFFFFu) && x[1] == a->seq && y[1] == b->seq;
            for (int k = 2; k < 8; ++k) same = same && x[k] == y[k];
            if (!same) {
                appendf(msg, msglen, "event %u: 0x%08x 0x%08x 0x%08x vs 0x%08x 0x%08x 0x%08x; ", i, x[0], x[2], x[3], y[0], y[2], y[3]);
                nd++;
                break;
            }
        }
    }
    // Log words, record by record: the header without LSEQ, the payload without the MTIME word.
    size_t i = 0, j = 0;
    uint32_t rec_no = 0;
    while (i < a->n_log && j < b->n_log) {
        const uint32_t ha = a->log[i], hb = b->log[j];
        const uint32_t mask = ~TSR_LOG_HDR_LSEQ;
        const uint32_t n = TSR_GET(ha, TSR_LOG_HDR_N);
        if ((ha & mask) != (hb & mask)) {
            appendf(msg, msglen, "log record %u: header 0x%08x vs 0x%08x; ", rec_no, ha, hb);
            nd++;
            break;
        }
        if (i + 1 + n > a->n_log || j + 1 + n > b->n_log) {
            appendf(msg, msglen, "log record %u: truncated; ", rec_no);
            nd++;
            break;
        }
        const uint32_t ts = (ha & TSR_LOG_HDR_TS) ? 1u : 0u;
        if (n > ts && memcmp(a->log + i + 1 + ts, b->log + j + 1 + ts, (n - ts) * sizeof(uint32_t)) != 0) {
            appendf(msg, msglen, "log record %u: payload differs; ", rec_no);
            nd++;
            break;
        }
        i += 1 + n;
        j += 1 + n;
        rec_no++;
    }
    if (nd == 0 && (i != a->n_log || j != b->n_log)) {
        appendf(msg, msglen, "log: %zu vs %zu words; ", a->n_log, b->n_log);
        nd++;
    }
    return nd;
}

// ---------------------------------------------------------------------------
// Trace replays
// ---------------------------------------------------------------------------
// A small per-process cache of checker simulators for trace configurations (tools are single-threaded).
typedef struct { tsr_trace2_config c; tsr_sim * sim; } cfg_sim;
static cfg_sim g_cfg_sims[8];
static uint32_t g_n_cfg_sims;

static tsr_sim * trace_check_sim(const tsr_trace2_config * c) {
    for (uint32_t i = 0; i < g_n_cfg_sims; ++i)
        if (memcmp(&g_cfg_sims[i].c, c, sizeof(*c)) == 0) return g_cfg_sims[i].sim;
    tsr_tgt_cfg tc;
    memset(&tc, 0, sizeof(tc));
    tc.core = c->core;
    memcpy(tc.q_depth, c->q_depth, sizeof(tc.q_depth));
    tc.itcm_bytes = c->itcm_bytes;
    tc.dtcm_bytes = c->dtcm_bytes;
    tsr_sim * sim = make_sim(&tc, 1);
    if (sim == NULL) return NULL;
    const uint32_t k = g_n_cfg_sims < 8u ? g_n_cfg_sims++ : 7u;
    if (g_cfg_sims[k].sim) tsr_sim_destroy(g_cfg_sims[k].sim);
    g_cfg_sims[k].c = *c;
    g_cfg_sims[k].sim = sim;
    return sim;
}

static uint32_t check_code(tsr_sim * sim, const uint32_t w[8]) {
    const tsr_op_handler * h = tsr_sim_op_handler((uint8_t)(w[0] & 0xFFu));
    if (h == NULL || h->check == NULL) return 0;
    uint32_t info = 0;
    return h->check(sim, w, NULL, &info);
}

static int same_core(const tsr_config * a, const tsr_config * b) {
    return a->num_ch == b->num_ch && a->beat_bits == b->beat_bits && a->lanes == b->lanes && a->t_max == b->t_max &&
           a->k_max == b->k_max && a->spm_bytes == b->spm_bytes && a->vl == b->vl && a->nsreg == b->nsreg &&
           a->ch_bytes == b->ch_bytes && a->addr_beat_w == b->addr_beat_w && a->caps == b->caps;
}

const char * tsr_devrun_trace_fits(const tsr_trace2 * t, const tsr_config * dev, const uint8_t * dev_q_depth,
                                   tsr_sim * check_sim, char * why, size_t whylen) {
    if (why && whylen) why[0] = '\0';
    const tsr_config * a = &t->cfg.core;
    if (a->num_ch != dev->num_ch || a->lanes != dev->lanes || a->k_max != dev->k_max || a->spm_bytes != dev->spm_bytes ||
        a->vl != dev->vl || a->nsreg != dev->nsreg || a->beat_bits != dev->beat_bits) {
        seterr(why, whylen, "NUM_CH %u, LANES %u, K_MAX %u, SPM %u KiB, VL %u, NSREG %u (the device: %u, %u, %u, %u KiB, %u, %u)",
               a->num_ch, a->lanes, a->k_max, a->spm_bytes / 1024u, a->vl, a->nsreg, dev->num_ch, dev->lanes, dev->k_max,
               dev->spm_bytes / 1024u, dev->vl, dev->nsreg);
        return why;
    }
    for (int pass = 0; pass < 2; ++pass) {
        const tsr_region * r = pass ? t->mem.expect : t->mem.init;
        const size_t n = pass ? t->mem.n_expect : t->mem.n_init;
        for (size_t i = 0; i < n; ++i)
            if (r[i].ch >= dev->num_ch || r[i].byte_off + r[i].size > dev->ch_bytes) {
                seterr(why, whylen, "%s region ch %u [0x%llx, +%llu) beyond the device's channel", pass ? "expect" : "init", r[i].ch,
                       (unsigned long long)r[i].byte_off, (unsigned long long)r[i].size);
                return why;
            }
    }
    if (t->kind == TSR_TRACE2_KIND_OPS && t->load_ch != TSR_TRACE2_LOAD_AUTO &&
        t->load_offset + (uint64_t)t->n_push * TSR_TRACE_PUSH_BYTES > dev->ch_bytes) {
        seterr(why, whylen, "push.bin beyond the device's channel");
        return why;
    }
    // A trace that holds the queues (QCTRL QHOLD set) may fill a held queue to the depth it was made
    // with: on a shallower queue the replay kernel would wait for a credit that never comes.
    int holds = 0;
    for (size_t i = 0; i < t->n_push && !holds; ++i)
        holds = t->push[i].kind == TSR_TRACE_PUSH_QCTRL && (t->push[i].w[0] & TSR_QCTRL_QHOLD_SET);
    for (uint32_t u = 0; holds && u < TSR_NUM_UNITS; ++u)
        if (dev_q_depth[u] < t->cfg.q_depth[u]) {
            seterr(why, whylen, "the trace holds the queues and unit %u is %u deep on the device, %u in the trace", u, dev_q_depth[u],
                   t->cfg.q_depth[u]);
            return why;
        }
    tsr_trace2_config dc;
    memset(&dc, 0, sizeof(dc));
    dc.core = *dev;
    memcpy(dc.q_depth, dev_q_depth, TSR_NUM_UNITS);
    if (same_core(a, dev) && memcmp(t->cfg.q_depth, dev_q_depth, TSR_NUM_UNITS) == 0) return why;   // identical: fits
    if (t->kind == TSR_TRACE2_KIND_LAUNCH) {
        seterr(why, whylen, "kind launch trace made on another configuration (T_MAX %u, CAPS 0x%x, Q_DEPTH or DRAM limit)", a->t_max,
               a->caps);
        return why;
    }
    tsr_sim * ts = check_sim ? trace_check_sim(&t->cfg) : NULL;
    if (ts == NULL) {
        seterr(why, whylen, "T_MAX %u, CAPS 0x%x, Q_DEPTH or the DRAM limit differ (no checker to compare the ops)", a->t_max, a->caps);
        return why;
    }
    for (size_t i = 0; i < t->n_push; ++i) {
        const tsr_trace_push * p = &t->push[i];
        const uint32_t ut = tsr_trace2_push_unit(&t->cfg, p), ud = tsr_trace2_push_unit(&dc, p);
        if (ut != ud) {
            seterr(why, whylen, "record %zu (opcode 0x%02x) goes to unit %u on the device, %u in the trace", i, p->w[0] & 0xFFu, ud, ut);
            return why;
        }
        if (p->kind != TSR_TRACE_PUSH_OP || ut >= TSR_UNIT_SYNC) continue;
        const uint32_t ct = check_code(ts, p->w), cd = check_code(check_sim, p->w);
        if (ct != cd) {
            seterr(why, whylen, "record %zu (opcode 0x%02x) checks %u on the device, %u in the trace", i, p->w[0] & 0xFFu, cd, ct);
            return why;
        }
    }
    return why;
}

// Lines [*line, *line + need) below spm_lines - 16 that no SPM extent of t's accepted op records
// touches, the highest such run (tsr_trace.c kr_spm_buffer, the same rule). 0, or -1.
static int spm_buffer(tsr_sim * sim, const tsr_trace2 * t, uint32_t need, uint32_t * line) {
    const uint32_t L = tsr_sim_spm_lines(sim);
    if (L < 16u + need) return -1;
    uint8_t * used = (uint8_t *)calloc(L, 1);
    if (used == NULL) return -1;
    for (size_t i = 0; i < t->n_push; ++i) {
        const tsr_trace_push * p = &t->push[i];
        if (p->kind != TSR_TRACE_PUSH_OP) continue;
        const uint32_t u = tsr_trace2_push_unit(&t->cfg, p);
        const tsr_op_handler * h = tsr_sim_op_handler((uint8_t)(p->w[0] & 0xFFu));
        if (u == TSR_UNIT_SYNC || h == NULL || h->check == NULL) continue;
        tsr_extents ext;
        memset(&ext, 0, sizeof(ext));
        uint32_t info = 0;
        if (h->check(sim, p->w, &ext, &info) != 0) continue;
        for (uint32_t k = 0; k < ext.n; ++k) {
            if (ext.e[k].kind != TSR_EXT_SPM) continue;
            for (uint64_t l = ext.e[k].lo; l < ext.e[k].hi && l < L; ++l) used[l] = 1;
        }
    }
    int rc = -1;
    uint32_t run = 0;
    for (uint32_t l = L - 16u; l-- > 0;) {
        run = used[l] ? 0u : run + 1u;
        if (run == need) {
            *line = l;
            rc = 0;
            break;
        }
    }
    free(used);
    return rc;
}

// push.bin's place when the trace has none: ch 0, the lowest 4 KiB-aligned offset at or above the
// OPL region where it overlaps no region (tsr_trace.c choose_load).
static int choose_load(const tsr_trace2 * t, const tsr_config * c, uint32_t * ch, uint64_t * off) {
    if (t->load_ch != TSR_TRACE2_LOAD_AUTO) {
        *ch = t->load_ch;
        *off = t->load_offset;
        return 0;
    }
    const uint64_t bytes = ((uint64_t)t->n_push * TSR_TRACE_PUSH_BYTES + 63u) & ~63ull;
    for (uint64_t o = TSR_REGION_OPL_OFF; o + bytes <= c->ch_bytes; o += 4096u) {
        int clash = 0;
        for (int pass = 0; pass < 2 && !clash; ++pass) {
            const tsr_region * r = pass ? t->mem.expect : t->mem.init;
            const size_t n = pass ? t->mem.n_expect : t->mem.n_init;
            for (size_t i = 0; i < n && !clash; ++i)
                clash = r[i].ch == 0 && r[i].byte_off < o + bytes && o < r[i].byte_off + r[i].size;
        }
        if (!clash) {
            *ch = 0;
            *off = o;
            return 0;
        }
    }
    return -1;
}

// 1 when two ELF files load the same words: the same PT_LOAD segments (address, padded size, bytes).
// A capture's image synthesized from the loader's words (SW §5.5, no section headers) has another
// image id than the stamped ELF it came from (the id zeroes .tsr_id, which only the ELF names, EX
// §3.2), but loads exactly the same words.
static int same_loaded_image(const char * a, const char * b) {
    tsr_rv_elf x, y;
    memset(&x, 0, sizeof(x));
    memset(&y, 0, sizeof(y));
    if (tsr_rv_elf_read(a, &x) != 0) return 0;
    if (tsr_rv_elf_read(b, &y) != 0) {
        tsr_rv_elf_free(&x);
        return 0;
    }
    uint32_t i = 0, j = 0;
    int same = 1;
    while (same) {   // segments in file order may differ: compare the non-empty ones by address
        while (i < x.nseg && x.seg[i].filesz == 0) ++i;
        while (j < y.nseg && y.seg[j].filesz == 0) ++j;
        if (i == x.nseg || j == y.nseg) {
            same = i == x.nseg && j == y.nseg;
            break;
        }
        const tsr_rv_elf_seg * p = &x.seg[i], * q = &y.seg[j];
        same = p->paddr == q->paddr && ((p->filesz + 3u) & ~3u) == ((q->filesz + 3u) & ~3u) &&
               memcmp(p->data, q->data, p->filesz < q->filesz ? p->filesz : q->filesz) == 0;
        for (uint32_t k = p->filesz < q->filesz ? p->filesz : q->filesz; same && k < ((p->filesz + 3u) & ~3u); ++k)
            same = (k < p->filesz ? p->data[k] : 0) == (k < q->filesz ? q->data[k] : 0);
        ++i;
        ++j;
    }
    tsr_rv_elf_free(&x);
    tsr_rv_elf_free(&y);
    return same;
}

static void fail(tsr_devrun_result * r, int verbose, uint32_t cls, const char * fmt, ...) __attribute__((format(printf, 4, 5)));
static void fail(tsr_devrun_result * r, int verbose, uint32_t cls, const char * fmt, ...) {
    char b[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(b, sizeof(b), fmt, ap);
    va_end(ap);
    if (r->ok) snprintf(r->msg, sizeof(r->msg), "%s", b);
    r->ok = 0;
    r->fail_mask |= cls;
    r->n_fail++;
    if (verbose) fprintf(stderr, "devrun: %s\n", b);
}

// Writes the trace's DRAM: expect regions first filled with the complement of their bytes, then the
// init regions (an init region may overlap an expect region: read, then written). 0 / -1.
static int load_regions(tsr_tgt * g, const tsr_trace2 * t) {
    for (size_t i = 0; i < t->mem.n_expect; ++i) {
        const tsr_region * e = &t->mem.expect[i];
        uint8_t * b = (uint8_t *)malloc((size_t)e->size);
        if (b == NULL) return -1;
        for (uint64_t k = 0; k < e->size; ++k) b[k] = (uint8_t)~e->data[k];
        const int rc = tsr_tgt_write(g, e->ch, e->byte_off, b, (size_t)e->size);
        free(b);
        if (rc) return -1;
    }
    for (size_t i = 0; i < t->mem.n_init; ++i) {
        const tsr_region * e = &t->mem.init[i];
        if (tsr_tgt_write(g, e->ch, e->byte_off, e->data, (size_t)e->size)) return -1;
    }
    return 0;
}

// Compares a target's expect regions with the trace; the first difference into r.
static void check_expect(tsr_tgt * g, const tsr_trace2 * t, tsr_devrun_result * r, int verbose, uint32_t cls, const char * who) {
    for (size_t i = 0; i < t->mem.n_expect; ++i) {
        const tsr_region * e = &t->mem.expect[i];
        uint8_t * b = (uint8_t *)malloc((size_t)e->size);
        if (b == NULL || tsr_tgt_read(g, e->ch, e->byte_off, b, (size_t)e->size)) {
            fail(r, verbose, TSR_DEVRUN_SETUP, "%s: reading expect region ch %u 0x%llx failed", who, e->ch, (unsigned long long)e->byte_off);
            free(b);
            return;
        }
        if (memcmp(b, e->data, (size_t)e->size) != 0) {
            uint64_t k = 0;
            while (b[k] == e->data[k]) ++k;
            // How far the region differs, in 64-byte beats of the region (a card failure is placed from this).
            uint64_t nb = 0;
            for (uint64_t o = 0; o < e->size; o += 64) {
                const uint64_t n = e->size - o < 64 ? e->size - o : 64;
                if (memcmp(b + o, e->data + o, (size_t)n) != 0) ++nb;
            }
            const char * dd = getenv("TSR_DEVRUN_DUMP");
            if (dd != NULL) {   // keep the device's bytes and the expected ones of the first bad region
                char fn[1024];
                for (int q = 0; q < 2; ++q) {
                    snprintf(fn, sizeof(fn), "%s/ch%u_%llx_%s.bin", dd, e->ch, (unsigned long long)e->byte_off, q ? "expect" : "device");
                    FILE * f = fopen(fn, "wb");
                    if (f) {
                        fwrite(q ? e->data : b, 1, (size_t)e->size, f);
                        fclose(f);
                    }
                }
            }
            fail(r, verbose, cls, "%s: expect ch %u byte 0x%llx: 0x%02x, expected 0x%02x (region ch %u 0x%llx +%llu: %llu of %llu beats differ)",
                 who, e->ch, (unsigned long long)(e->byte_off + k), b[k], e->data[k], e->ch, (unsigned long long)e->byte_off,
                 (unsigned long long)e->size, (unsigned long long)nb, (unsigned long long)((e->size + 63) / 64));
            free(b);
            return;
        }
        free(b);
    }
}

// The device's answer against the trace's DONE (SW §5.4; kind ops: what the replay kernel answers).
static void check_done(const tsr_trace2 * t, const uint32_t rec[8], tsr_devrun_result * r, int verbose, const char * who) {
    const tsr_trace_done * d = &t->done;
    const uint32_t type = TSR_GET(rec[0], TSR_LCQ_R0_TYPE), st = TSR_GET(rec[0], TSR_LCQ_R0_STATUS);
    const uint32_t code = TSR_GET(rec[0], TSR_LCQ_R0_ERR_CODE);
    int ok = type == TSR_LCQ_R0_TYPE_DONE && st == d->status && code == d->err_code && (int32_t)rec[2] == d->ret;
    if (ok && d->err_code != 0)
        ok = rec[4] == d->err_info && rec[5] == d->err_w0 && (rec[3] >> 28) == (d->err_ticket_rel >> 28);
    if (!ok)
        fail(r, verbose, TSR_DEVRUN_DONE, "%s: answer type 0x%02x status %u code %u RET %d INFO 0x%08x W0 0x%08x ticket 0x%08x; the trace "
             "has status %u code %u RET %d INFO 0x%08x W0 0x%08x ticket unit %u", who, type, st, code, (int32_t)rec[2], rec[4], rec[5],
             rec[3], d->status, d->err_code, d->ret, d->err_info, d->err_w0, d->err_ticket_rel >> 28);
}

int tsr_devrun_trace(tsr_tgt * dev, tsr_tgt * shadow, const tsr_trace2 * t, const tsr_devrun_opts * o, tsr_devrun_result * r) {
    static const tsr_devrun_opts none;
    if (o == NULL) o = &none;
    memset(r, 0, sizeof(*r));
    r->ok = 1;
    char why[512];
    if (tsr_trace2_validate(t, why, sizeof(why))) {
        fail(r, o->verbose, TSR_DEVRUN_SETUP, "invalid trace: %s", why);
        return 2;
    }
    tsr_sim * cs = check_sim_of(dev);
    tsr_devrun_trace_fits(t, &dev->cfg.core, dev->cfg.q_depth, cs, why, sizeof(why));
    if (why[0]) {
        r->ok = 0;
        r->skipped = 1;
        snprintf(r->msg, sizeof(r->msg), "skipped: %s", why);
        return 3;
    }
    tsr_tgt * tg[2] = { dev, shadow };
    const int ntg = shadow ? 2 : 1;
    char tmp_image[512] = "";
    uint32_t type = TSR_DESC_TYPE_LAUNCH, flags = 0, kernel = 0, args[7] = { 0 };
    int rc = 0;
    if (t->kind == TSR_TRACE2_KIND_LAUNCH) {
        // The trace's own image. A target that runs an image with the same id (EX §3.2: the same loaded
        // bytes) needs nothing; o->image is tried next (a capture's image may be synthesized from the
        // loader's words, without the sections tsr_image_open checks, SW §5.5); else the trace's ELF is
        // written to a private file and booted.
        for (int k = 0; k < ntg && rc == 0; ++k) {
            if (tsr_tgt_image_id(tg[k]) == t->image_id) continue;
            if (o->image && tsr_tgt_boot(tg[k], o->image, why, sizeof(why)) == 0 && tsr_tgt_image_id(tg[k]) == t->image_id) continue;
            if (tmp_image[0] == '\0') {
                const char * td = getenv("TMPDIR");
                snprintf(tmp_image, sizeof(tmp_image), "%s/tsr-devrun-XXXXXX", td && *td ? td : "/tmp");
                const int fd = mkstemp(tmp_image);
                if (fd < 0 || write(fd, t->image, t->image_size) != (ssize_t)t->image_size) {
                    if (fd >= 0) close(fd);
                    fail(r, o->verbose, TSR_DEVRUN_SETUP, "cannot write the trace's image to %s: %s", tmp_image, strerror(errno));
                    if (fd >= 0) unlink(tmp_image);
                    return 2;
                }
                close(fd);
            }
            // the same words as o->image (a synthesized capture image): run with o->image
            if (o->image && same_loaded_image(tmp_image, o->image) && tsr_tgt_boot(tg[k], o->image, why, sizeof(why)) == 0) continue;
            if (tsr_tgt_boot(tg[k], tmp_image, why, sizeof(why))) {
                fail(r, o->verbose, TSR_DEVRUN_SETUP, "%s", why);
                rc = 2;
            }
        }
        if (rc == 0 && shadow) tsr_tgt_csr_write(shadow, TSR_CSR_LOG_CFG, tsr_tgt_csr(dev, TSR_CSR_LOG_CFG));   // same log records
        kernel = t->kernel_index;
        flags = t->prof ? 1u : 0u;
        memcpy(args, t->args, sizeof(args));
    } else {
        for (int k = 0; k < ntg && rc == 0; ++k) {
            if (o->image && tsr_tgt_boot(tg[k], o->image, why, sizeof(why))) {
                fail(r, o->verbose, TSR_DEVRUN_SETUP, "%s", why);
                rc = 2;
            } else if (tsr_tgt_image_id(tg[k]) == 0) {
                fail(r, o->verbose, TSR_DEVRUN_SETUP, "%s: no image booted (the replay kernel is needed)", tsr_tgt_name(tg[k]));
                rc = 2;
            }
        }
        if (rc == 0 && shadow) tsr_tgt_csr_write(shadow, TSR_CSR_LOG_CFG, tsr_tgt_csr(dev, TSR_CSR_LOG_CFG));
        int ki = rc == 0 ? tsr_tgt_kernel(dev, "replay") : -1;
        if (rc == 0 && (ki < 0 || (shadow && tsr_tgt_kernel(shadow, "replay") != ki))) {
            fail(r, o->verbose, TSR_DEVRUN_SETUP, "the booted image has no `replay` kernel");
            rc = 2;
        }
        uint32_t lch = 0;
        uint64_t loff = 0;
        if (rc == 0 && choose_load(t, &dev->cfg.core, &lch, &loff)) {
            fail(r, o->verbose, TSR_DEVRUN_SETUP, "no DRAM place for push.bin outside the trace's regions");
            rc = 2;
        }
        const uint32_t chunk = t->n_push <= TSR_REPLAY_CHUNK_MAX ? (uint32_t)((t->n_push + 3u) & ~(size_t)3u) : TSR_REPLAY_CHUNK_MAX;
        const uint32_t chunk_lines = chunk * TSR_TRACE_PUSH_BYTES / 64u;
        const uint32_t nbuf = t->n_push > chunk ? 2u : 1u;
        uint32_t buf_line = 0;
        if (rc == 0 && chunk > 0 && spm_buffer(cs, t, nbuf * chunk_lines, &buf_line)) {
            fail(r, o->verbose, TSR_DEVRUN_SETUP, "no %u free SPM lines for the replay buffer outside the trace's SPM operands",
                 nbuf * chunk_lines);
            rc = 2;
        }
        if (rc == 0 && t->n_push > 0) {
            const uint64_t lines = ((uint64_t)t->n_push * TSR_TRACE_PUSH_BYTES + 63u) / 64u;
            uint8_t * buf = (uint8_t *)calloc((size_t)lines, 64);
            for (size_t i = 0; buf && i < t->n_push; ++i) {
                const tsr_trace_push * p = &t->push[i];
                uint8_t * q = buf + i * TSR_TRACE_PUSH_BYTES;
                for (int k = 0; k < 8; ++k) wr32le(q + 4 * k, p->w[k]);
                wr32le(q + 32, p->dep[0]);
                wr32le(q + 36, p->dep[1]);
                wr32le(q + 40, p->push);
                wr32le(q + 44, p->kind);
            }
            for (int k = 0; k < ntg && buf; ++k)
                if (tsr_tgt_write(tg[k], lch, loff, buf, (size_t)lines * 64u)) {
                    fail(r, o->verbose, TSR_DEVRUN_SETUP, "%s: writing push.bin failed", tsr_tgt_name(tg[k]));
                    rc = 2;
                }
            if (buf == NULL) rc = 2;
            free(buf);
        }
        kernel = (uint32_t)(ki < 0 ? 0 : ki);
        args[0] = (lch << 28) | (uint32_t)(loff / 64u);
        args[1] = (uint32_t)t->n_push;
        args[2] = (o->replay_flags & TSR_REPLAY_F_SERIAL) | TSR_REPLAY_F_BUF;
        args[3] = buf_line;
        args[4] = chunk ? chunk : 4u;
    }
    if (t->kind == TSR_TRACE2_KIND_OPS && getenv("TSR_DEVRUN_ARGS") != NULL) {   // the replay kernel's launch, for an RTL replay of the trace
        const tsr_region * e0 = t->mem.n_expect ? &t->mem.expect[0] : NULL;
        fprintf(stderr, "REPLAY_ARGS n_push %zu first_expect %u:%llx args %08x %u %u %u %u\n", t->n_push, e0 ? e0->ch : 0u,
                (unsigned long long)(e0 ? e0->byte_off : 0u), args[0], args[1], args[2], args[3], args[4]);
    }
    for (int k = 0; k < ntg && rc == 0; ++k)
        if (load_regions(tg[k], t)) {
            fail(r, o->verbose, TSR_DEVRUN_SETUP, "%s: writing the trace's regions failed", tsr_tgt_name(tg[k]));
            rc = 2;
        }
    tsr_tgt_run run[2];
    memset(run, 0, sizeof(run));
    if (rc == 0) {
        for (int k = 0; k < ntg; ++k) {
            if (tsr_tgt_launch(tg[k], type, flags, kernel, args, &run[k]) != 0) {
                fail(r, o->verbose, TSR_DEVRUN_SETUP, "%s: the launch failed (%d: %s)", tsr_tgt_name(tg[k]), run[k].rc,
                     k == 0 ? tsr_transport_strerror(run[k].rc) : "shadow");
                rc = 2;
            }
        }
        r->wall_ns = run[0].wall_ns;
        memcpy(r->rec, run[0].rec, sizeof(r->rec));
    }
    if (rc == 0) {
        check_done(t, run[0].rec, r, o->verbose, "device");
        check_expect(dev, t, r, o->verbose, TSR_DEVRUN_EXPECT, "device");
        if (shadow) {
            char m[512];
            if (tsr_tgt_compare(&run[0], &run[1], m, sizeof(m)) != 0)
                fail(r, o->verbose, TSR_DEVRUN_SHADOW, "device vs shadow: %s", m);
            check_done(t, run[1].rec, r, o->verbose, "shadow");
            check_expect(shadow, t, r, o->verbose, TSR_DEVRUN_SHADOW, "shadow");
        }
    }
    for (int k = 0; k < ntg; ++k) tsr_tgt_run_free(&run[k]);
    if (tmp_image[0]) unlink(tmp_image);
    if (rc) return rc;
    if (r->ok)
        snprintf(r->msg, sizeof(r->msg), "ok: %s, %zu pushes, %zu init / %zu expect regions, status %u RET %d%s",
                 t->kind == TSR_TRACE2_KIND_LAUNCH ? "launch" : "ops via replay", t->n_push, t->mem.n_init, t->mem.n_expect,
                 TSR_GET(r->rec[0], TSR_LCQ_R0_STATUS), (int32_t)r->rec[2], shadow ? ", = shadow" : "");
    return r->ok ? 0 : 1;
}
