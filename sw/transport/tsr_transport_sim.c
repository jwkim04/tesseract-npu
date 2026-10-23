// tsr_transport_sim.c — the "sim" transport: the ISA simulator in-process.
//
// Environment (read once, at open):
//   TSR_SIM_DRAM_DIR  back channel DRAM with sparse unlinked temp files in this directory
//                     (needed when the model does not fit in RAM; default: anonymous memory)
//   TSR_SIM_TMAX      T_MAX override, 1..8 (default 1, the U200 v1 synthesis value)
//   TSR_SIM_CH_GIB    per-channel DRAM size in GiB, 1..16 (default 16; ADDR_BEAT_W = 28 caps
//                     the addressable size at 16 GiB)
//   TSR_SIM_THREADS   simulator worker threads for large MVMs (default min(online CPUs, 8))
//   TSR_SIM_LOG       per-instruction log: "stderr" or a file path (default off)
//   TSR_TRACE         dump every exec as a trace directory <TSR_TRACE>/<seq, 6 digits>/;
//                     the sequence number is the first free one, so runs never overwrite
//                     earlier dumps. The directory is logged to stderr once.
// Any malformed value makes tsr_transport_open fail with a message (never silently ignored).

#define _GNU_SOURCE

#include "tsr_sim.h"
#include "tsr_trace.h"
#include "tsr_transport_impl.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

typedef struct {
    tsr_sim *  sim;
    tsr_config cfg;          // as read back through the CFG*/CAPS CSRs (+ ch_bytes etc.)
    char *     trace_dir;
    uint32_t   trace_seq;
    int        trace_logged;
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

static int mkdir_p(const char * path) {
    const size_t n = strlen(path);
    char * p = (char *)malloc(n + 1);
    if (p == NULL) return -1;
    memcpy(p, path, n + 1);
    for (size_t i = 1; i <= n; ++i) {
        if (p[i] == '/' || p[i] == '\0') {
            const char c = p[i];
            p[i] = '\0';
            if (mkdir(p, 0755) != 0 && errno != EEXIST) {
                fprintf(stderr, "tsr_transport(sim): mkdir %s: %s\n", p, strerror(errno));
                free(p);
                return -1;
            }
            p[i] = c;
        }
    }
    free(p);
    struct stat st;
    if (stat(path, &st) != 0 || !S_ISDIR(st.st_mode)) {
        fprintf(stderr, "tsr_transport(sim): %s is not a directory\n", path);
        return -1;
    }
    return 0;
}

static void st_close(void * impl) {
    sim_transport * s = (sim_transport *)impl;
    if (s == NULL) return;
    tsr_sim_destroy(s->sim);
    if (s->log_close && s->log) fclose(s->log);
    free(s->trace_dir);
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

// Creates the next free <trace_dir>/<seq> directory; returns its path (malloc) or NULL.
static char * next_trace_dir(sim_transport * s) {
    const size_t n = strlen(s->trace_dir) + 32;
    char * p = (char *)malloc(n);
    if (p == NULL) return NULL;
    for (;;) {
        snprintf(p, n, "%s/%06u", s->trace_dir, s->trace_seq);
        if (mkdir(p, 0755) == 0) break;
        if (errno != EEXIST) {
            fprintf(stderr, "tsr_transport(sim): TSR_TRACE: cannot create %s: %s\n", p, strerror(errno));
            free(p);
            return NULL;
        }
        s->trace_seq++;
    }
    s->trace_seq++;
    return p;
}

static int st_exec(void * impl, uint64_t cmd_addr_beat, uint32_t cmd_count, const uint32_t args[8],
                   tsr_exec_result * res) {
    sim_transport * s = (sim_transport *)impl;
    const uint32_t nch = s->cfg.num_ch;
    uint64_t perf0[3 + TSR_TRACE_MAX_CH] = { 0 };
    if (s->trace_dir) {
        perf0[0] = tsr_sim_perf(s->sim, TSR_PERF_CYCLES);
        perf0[1] = tsr_sim_perf(s->sim, TSR_PERF_RETIRED);
        perf0[2] = tsr_sim_perf(s->sim, TSR_PERF_MVM_CYCLES);
        for (uint32_t c = 0; c < nch; ++c) perf0[3 + c] = tsr_sim_perf(s->sim, tsr_perf_ch_rd_beats(c));
        tsr_sim_capture_begin(s->sim);
    }
    const uint32_t status = tsr_sim_exec(s->sim, cmd_addr_beat, cmd_count, args);
    res->status = status;
    res->err_code = tsr_sim_csr_read(s->sim, TSR_CSR_ERR_CODE);
    res->err_pc = tsr_sim_csr_read(s->sim, TSR_CSR_ERR_PC);
    res->err_info = tsr_sim_csr_read(s->sim, TSR_CSR_ERR_INFO);

    if (s->trace_dir) {
        tsr_trace tr;
        memset(&tr, 0, sizeof(tr));
        tr.cfg = *tsr_sim_config(s->sim);
        tr.cmd_addr_beat = cmd_addr_beat;
        tr.cmd_count = cmd_count;
        for (int i = 0; i < 8; ++i) tr.args[i] = args ? args[i] : 0u;
        tr.status = status;
        tr.err_code = res->err_code;
        tr.err_pc = res->err_pc;
        tr.err_info = res->err_info;
        tr.has_err_info = 1;
        tr.perf.valid = 1;
        tr.perf.cycles = tsr_sim_perf(s->sim, TSR_PERF_CYCLES) - perf0[0];
        tr.perf.retired = tsr_sim_perf(s->sim, TSR_PERF_RETIRED) - perf0[1];
        tr.perf.mvm_cycles = tsr_sim_perf(s->sim, TSR_PERF_MVM_CYCLES) - perf0[2];
        for (uint32_t c = 0; c < nch; ++c) {
            tr.perf.ch_rd_beats[c] = tsr_sim_perf(s->sim, tsr_perf_ch_rd_beats(c)) - perf0[3 + c];
        }
        tsr_sim_capture_end(s->sim, &tr.cap);
        char * dir = next_trace_dir(s);
        if (dir != NULL) {
            if (!s->trace_logged) {
                fprintf(stderr, "tsr_transport(sim): TSR_TRACE: dumping every exec as a trace directory under %s/ "
                                "(first: %s)\n", s->trace_dir, dir);
                s->trace_logged = 1;
            }
            if (tsr_trace_write(dir, &tr) != 0) {
                fprintf(stderr, "tsr_transport(sim): TSR_TRACE: failed to write trace %s\n", dir);
            }
            free(dir);
        }
        tsr_capture_free(&tr.cap);
    }
    return ((status & TSR_STATUS_DONE) && !(status & TSR_STATUS_ERR)) ? 0 : -1;
}

static const tsr_transport_ops sim_ops = {
    "sim", st_close, st_config, st_write, st_read, st_exec, st_csr_read,
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

    const char * tdir = getenv("TSR_TRACE");
    if (tdir != NULL && tdir[0] != '\0') {
        if (mkdir_p(tdir) != 0) {
            st_close(s);
            return NULL;
        }
        s->trace_dir = (char *)malloc(strlen(tdir) + 1);
        if (s->trace_dir == NULL) {
            st_close(s);
            return NULL;
        }
        strcpy(s->trace_dir, tdir);
    }

    // Configuration as software sees it on hardware: decoded from the ID/VERSION/CFG*/CAPS CSRs.
    // BEAT_BITS, NIREG, LOOP_DEPTH, RD_OUTSTANDING, ADDR_BEAT_W and CH_BYTES are not carried by
    // the CSRs; they come from the platform description (here: the simulator's config).
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
