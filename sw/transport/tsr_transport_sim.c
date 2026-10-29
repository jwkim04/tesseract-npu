// tsr_transport_sim.c — the "sim" transport: the ISA simulator in-process.
//
// Environment (read once, at open):
//   TSR_SIM_DRAM_DIR  back channel DRAM with sparse unlinked temp files in this directory
//                     (needed when the model does not fit in RAM; default: anonymous memory)
//   TSR_SIM_TMAX      T_MAX override, 1..8 (default 1, the U200 v1 synthesis value)
//   TSR_SIM_CH_GIB    per-channel DRAM size in GiB, 1..16 (default 16; ADDR_BEAT_W = 28 caps
//                     the addressable size at 16 GiB)
//   TSR_SIM_THREADS   simulator worker threads for large MVMs (default min(online CPUs, 8))
//   TSR_SIM_LOG       the simulator's per-op log: "stderr" or a file path (default off)
// Any malformed value makes tsr_transport_open fail with a message (never silently ignored).
// The per-exec TSR_TRACE dump of v1 is deleted with the run interface (X0); trace v2 dumps one
// trace per launch (docs/v2/06-software-verification.md §5.5, P1.10 / P3.1). This file is
// replaced by the "sim" kdev of the v2 protocol layer (SW V2-SW-18).

#define _GNU_SOURCE

#include "tsr_sim.h"
#include "tsr_transport_impl.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef struct {
    tsr_sim *  sim;
    tsr_config cfg;          // as read back through the CFG*/CAPS CSRs (+ ch_bytes etc.)
    FILE *     log;
    int        log_close;
} sim_transport;

// Returns 0 if unset, 1 if set and valid (*out), -1 if set but invalid (message printed).
static int env_uint(const char * name, unsigned long lo, unsigned long hi, unsigned long * out) {
    const char * v = getenv(name);
    if (v == NULL || v[0] == '\0') return 0;
    unsigned long r = 0;
    for (const char * p = v; *p; ++p) {
        if (*p < '0' || *p > '9' || r > hi) break;    // r > hi: stop before it can overflow
        r = r * 10 + (unsigned long)(*p - '0');
        if (p[1] == '\0' && r >= lo && r <= hi) {
            *out = r;
            return 1;
        }
    }
    fprintf(stderr, "tsr_transport(sim): %s=\"%s\" is not an integer in [%lu, %lu]\n", name, v, lo, hi);
    return -1;
}

static void st_close(void * impl) {
    sim_transport * s = (sim_transport *)impl;
    if (s == NULL) return;
    tsr_sim_destroy(s->sim);
    if (s->log_close && s->log) fclose(s->log);
    free(s);
}

static const tsr_config * st_config(void * impl) { return &((sim_transport *)impl)->cfg; }

static int st_range(const sim_transport * s, uint32_t ch, uint64_t off, size_t size, const char * what) {
    if (ch >= s->cfg.num_ch || off > s->cfg.ch_bytes || (uint64_t)size > s->cfg.ch_bytes - off) {
        fprintf(stderr, "tsr_transport(sim): %s out of range: ch %u, offset 0x%llx, %zu bytes (num_ch %u, ch_bytes %llu)\n",
                what, ch, (unsigned long long)off, size, s->cfg.num_ch, (unsigned long long)s->cfg.ch_bytes);
        return -1;
    }
    return 0;
}

static int st_write(void * impl, uint32_t ch, uint64_t off, const void * src, size_t size) {
    sim_transport * s = (sim_transport *)impl;
    if (st_range(s, ch, off, size, "write")) return -1;
    if (size) memcpy(tsr_sim_dram(s->sim, ch) + off, src, size);
    return 0;
}

static int st_read(void * impl, uint32_t ch, uint64_t off, void * dst, size_t size) {
    sim_transport * s = (sim_transport *)impl;
    if (st_range(s, ch, off, size, "read")) return -1;
    if (size) memcpy(dst, tsr_sim_dram(s->sim, ch) + off, size);
    return 0;
}

static uint32_t st_csr_read(void * impl, uint32_t off) { return tsr_sim_csr_read(((sim_transport *)impl)->sim, off); }

static const tsr_transport_ops sim_ops = {
    "sim", st_close, st_config, st_write, st_read, st_csr_read,
};

void * tsr_transport_sim_open(const tsr_transport_ops ** ops) {
    tsr_config cfg = TSR_CONFIG_U200_V1;
    unsigned long v = 0;
    int r;
    if ((r = env_uint("TSR_SIM_TMAX", 1, TSR_T_MAX_ARCH, &v)) < 0) return NULL;
    if (r) cfg.t_max = (uint32_t)v;
    if ((r = env_uint("TSR_SIM_CH_GIB", 1, 16, &v)) < 0) return NULL;
    if (r) cfg.ch_bytes = (uint64_t)v << 30;
    long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
    unsigned long threads = (ncpu < 1) ? 1ul : (ncpu > 8 ? 8ul : (unsigned long)ncpu);
    if ((r = env_uint("TSR_SIM_THREADS", 1, 64, &v)) < 0) return NULL;
    if (r) threads = v;

    const char * dram_dir = getenv("TSR_SIM_DRAM_DIR");
    if (dram_dir != NULL && dram_dir[0] == '\0') dram_dir = NULL;

    sim_transport * s = (sim_transport *)calloc(1, sizeof(sim_transport));
    if (s == NULL) return NULL;
    s->sim = tsr_sim_create(&cfg, dram_dir);
    if (s->sim == NULL) {
        fprintf(stderr, "tsr_transport(sim): simulator creation failed%s%s%s\n",
                dram_dir ? " (TSR_SIM_DRAM_DIR=" : "", dram_dir ? dram_dir : "", dram_dir ? ")" : "");
        free(s);
        return NULL;
    }
    tsr_sim_set_threads(s->sim, (unsigned)threads);

    const char * logv = getenv("TSR_SIM_LOG");
    if (logv != NULL && logv[0] != '\0') {
        if (strcmp(logv, "stderr") == 0) {
            s->log = stderr;
        } else {
            s->log = fopen(logv, "w");
            if (s->log == NULL) {
                fprintf(stderr, "tsr_transport(sim): TSR_SIM_LOG: cannot open %s: %s\n", logv, strerror(errno));
                st_close(s);
                return NULL;
            }
            s->log_close = 1;
        }
        tsr_sim_set_log(s->sim, s->log);
    }

    // Configuration as software sees it on hardware: decoded from the ID/VERSION/CFG*/CAPS CSRs.
    // BEAT_BITS, RD_OUTSTANDING, ADDR_BEAT_W and CH_BYTES are not carried by the CSRs; they come
    // from the platform description (here: the simulator's config).
    const tsr_config * sc = tsr_sim_config(s->sim);
    const uint32_t id = tsr_sim_csr_read(s->sim, TSR_CSR_ID);
    const uint32_t ver = tsr_sim_csr_read(s->sim, TSR_CSR_VERSION);
    const uint32_t cfg0 = tsr_sim_csr_read(s->sim, TSR_CSR_CFG0);
    const uint32_t cfg1 = tsr_sim_csr_read(s->sim, TSR_CSR_CFG1);
    const uint32_t cfg2 = tsr_sim_csr_read(s->sim, TSR_CSR_CFG2);
    if (id != TSR_ID_MAGIC || (ver >> 8) != TSR_ISA_MAJOR) {
        fprintf(stderr, "tsr_transport(sim): bad ID 0x%08x / VERSION 0x%08x\n", id, ver);
        st_close(s);
        return NULL;
    }
    tsr_config d = *sc;
    d.num_ch = cfg0 & 0xFFu;
    d.lanes = (cfg0 >> 8) & 0xFFFu;
    d.t_max = (cfg0 >> 20) & 0xFu;
    d.spm_bytes = cfg1 * TSR_LINE_BYTES;
    d.vl = cfg2 & 0xFFu;
    d.nsreg = (cfg2 >> 8) & 0xFFu;
    d.k_max = ((cfg2 >> 16) & 0xFFFFu) * 256u;
    d.caps = tsr_sim_csr_read(s->sim, TSR_CSR_CAPS);
    if (d.num_ch != sc->num_ch || d.lanes != sc->lanes || d.t_max != sc->t_max || d.spm_bytes != sc->spm_bytes ||
        d.vl != sc->vl || d.nsreg != sc->nsreg || d.k_max != sc->k_max || d.caps != sc->caps) {
        fprintf(stderr, "tsr_transport(sim): internal error: CSR-decoded config differs from the simulator config\n");
        st_close(s);
        return NULL;
    }
    s->cfg = d;
    *ops = &sim_ops;
    return s;
}
