// tsr_trace_replay.c — tsr-trace-replay v2 (docs/v2/06-software-verification.md "SW" §5.4, §9.5):
// replays trace v2 directories on fresh simulators and compares everything SW §5.4 lists (the
// DONE fields, expect regions bit-exactly, written beats = the union of the expect regions, reads
// inside init or written earlier, per-unit completion counts and error records, the per-unit
// projections of push.bin). Kind `launch` boots the trace's image and pushes its descriptor; kind
// `ops` is a tile-only replay through the issue port (tsr_trace.h). The library does the work
// (tsr_trace2_read, tsr_trace2_replay_sim_ex); this is the command line around it.
//
//   tsr-trace-replay [-q | -v] [--sched S]... [--rv-ratio R] [--threads N] [--max-cycles N] [--tmax N] PATH...
//
//   PATH          a trace directory (it has meta.json) or a set: every directory below it that has
//                 a meta.json, in sorted order (an `images` directory is not searched)
//   --sched S     replay under schedule S ("fixed" or "random:<seed>[,<min>,<max>]", SW §4.4);
//                 repeatable: every trace under every schedule. Default: the simulator's default
//                 (TSR_SIM_SCHED, else fixed)
//   --rv-ratio R  F_RV / F_CORE of the replays (kind launch; default: the trace's "rv_ratio")
//   --threads N   MVM worker threads of each simulator (results do not depend on it; default
//                 TSR_SIM_THREADS, else 1)
//   --max-cycles  simulated-time budget per replay (core cycles; default 2^40)
//   --tmax N      replay on a core with T_MAX = N instead of the trace's (a trace must pass on any
//                 T_MAX at least its own)
//   -q            print only failures and the summary; -v also every failed comparison
// Output: one line per trace and schedule ("ok" / "FAIL" with the first difference), then
//   tsr-trace-replay: <n> traces x <m> schedules: <p> passed, <f> failed, <e> errors
// Exit status: 0 all passed, 1 a mismatch, 2 a trace that could not be read or set up (or usage).
#define _GNU_SOURCE

#include "tsr_trace.h"

#include <dirent.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

typedef struct { char ** v; size_t n, cap; } strlist;

static void sl_add(strlist * l, const char * s) {
    if (l->n == l->cap) {
        l->cap = l->cap ? 2 * l->cap : 64;
        l->v = (char **)realloc(l->v, l->cap * sizeof(char *));
        if (l->v == NULL) {
            fprintf(stderr, "tsr-trace-replay: out of memory\n");
            exit(2);
        }
    }
    l->v[l->n++] = strdup(s);
}

static int str_cmp(const void * a, const void * b) { return strcmp(*(char * const *)a, *(char * const *)b); }

static int has_meta(const char * dir) {
    char p[4096];
    snprintf(p, sizeof(p), "%s/meta.json", dir);
    struct stat st;
    return stat(p, &st) == 0 && S_ISREG(st.st_mode);
}

// Trace directories at or below dir (a trace directory is not searched further).
static void collect(const char * dir, strlist * out, int depth) {
    if (has_meta(dir)) {
        sl_add(out, dir);
        return;
    }
    if (depth > 16) return;
    DIR * d = opendir(dir);
    if (d == NULL) return;
    strlist subs = { 0 };
    struct dirent * e;
    while ((e = readdir(d)) != NULL) {
        if (e->d_name[0] == '.' || strcmp(e->d_name, "images") == 0) continue;
        char p[4096];
        snprintf(p, sizeof(p), "%s/%s", dir, e->d_name);
        struct stat st;
        if (stat(p, &st) == 0 && S_ISDIR(st.st_mode)) sl_add(&subs, p);
    }
    closedir(d);
    if (subs.n > 1) qsort(subs.v, subs.n, sizeof(char *), str_cmp);
    for (size_t i = 0; i < subs.n; ++i) {
        collect(subs.v[i], out, depth + 1);
        free(subs.v[i]);
    }
    free(subs.v);
}

static void usage(void) {
    fprintf(stderr, "usage: tsr-trace-replay [-q | -v] [--sched S]... [--rv-ratio R] [--threads N] [--max-cycles N] [--tmax N] PATH...\n"
                    "  PATH: a trace v2 directory or a set of them (searched recursively for meta.json)\n");
}

int main(int argc, char ** argv) {
    int quiet = 0, verbose = 0;
    unsigned threads = 0;
    uint64_t max_cycles = 0;
    uint32_t tmax = 0;
    double rv_ratio = 0.0;
    strlist scheds = { 0 }, paths = { 0 };
    for (int i = 1; i < argc; ++i) {
        const char * a = argv[i];
        if (strcmp(a, "-q") == 0) quiet = 1;
        else if (strcmp(a, "-v") == 0) verbose = 1;
        else if (strcmp(a, "--sched") == 0 && i + 1 < argc) sl_add(&scheds, argv[++i]);
        else if (strcmp(a, "--rv-ratio") == 0 && i + 1 < argc) rv_ratio = strtod(argv[++i], NULL);
        else if (strcmp(a, "--threads") == 0 && i + 1 < argc) threads = (unsigned)strtoul(argv[++i], NULL, 0);
        else if (strcmp(a, "--max-cycles") == 0 && i + 1 < argc) max_cycles = strtoull(argv[++i], NULL, 0);
        else if (strcmp(a, "--tmax") == 0 && i + 1 < argc) tmax = (uint32_t)strtoul(argv[++i], NULL, 0);
        else if (strcmp(a, "-h") == 0 || strcmp(a, "--help") == 0) { usage(); return 0; }
        else if (a[0] == '-') { fprintf(stderr, "tsr-trace-replay: unknown option %s\n", a); usage(); return 2; }
        else sl_add(&paths, a);
    }
    if (paths.n == 0) {
        usage();
        return 2;
    }
    if (threads == 0) {
        const char * e = getenv("TSR_SIM_THREADS");
        threads = e ? (unsigned)strtoul(e, NULL, 0) : 1u;
    }
    strlist traces = { 0 };
    int bad_path = 0;
    for (size_t i = 0; i < paths.n; ++i) {
        struct stat st;
        if (stat(paths.v[i], &st) != 0 || !S_ISDIR(st.st_mode)) {
            fprintf(stderr, "tsr-trace-replay: %s: not a directory\n", paths.v[i]);
            bad_path = 1;
            continue;
        }
        const size_t before = traces.n;
        collect(paths.v[i], &traces, 0);
        if (traces.n == before) fprintf(stderr, "tsr-trace-replay: %s: no trace (no meta.json below it)\n", paths.v[i]);
    }
    const size_t n_sched = scheds.n ? scheds.n : 1;
    size_t passed = 0, failed = 0, errors = 0;
    for (size_t i = 0; i < traces.n && !bad_path; ++i) {
        tsr_trace2 * t = NULL;
        char err[1024];
        if (tsr_trace2_read(traces.v[i], &t, err, sizeof(err)) != 0) {
            printf("ERROR %s: %s\n", traces.v[i], err);
            errors += n_sched;
            continue;
        }
        tsr_config cfg = t->cfg.core;
        if (tmax) cfg.t_max = tmax;
        for (size_t s = 0; s < n_sched; ++s) {
            tsr_trace2_replay_opts o;
            memset(&o, 0, sizeof(o));
            o.cfg = tmax ? &cfg : NULL;
            o.sched = scheds.n ? scheds.v[s] : NULL;
            o.threads = threads;
            o.rv_ratio = rv_ratio;
            o.max_cycles = max_cycles;
            o.verbose = verbose;
            tsr_trace2_result r;
            const int rc = tsr_trace2_replay_sim_ex(t, &o, &r);
            const char * sname = o.sched ? o.sched : "default";
            if (rc == 0) {
                ++passed;
                if (!quiet) printf("ok    %s [%s]: %s\n", traces.v[i], sname, r.msg);
            } else if (rc == 1) {
                ++failed;
                printf("FAIL  %s [%s]: %s (%u comparison(s) failed, classes 0x%x)\n", traces.v[i], sname, r.msg, r.n_fail,
                       r.fail_mask);
            } else {
                ++errors;
                printf("ERROR %s [%s]: %s\n", traces.v[i], sname, r.msg);
            }
            fflush(stdout);
        }
        tsr_trace2_free(t);
    }
    if (!bad_path)
        printf("tsr-trace-replay: %zu traces x %zu schedule%s: %zu passed, %zu failed, %zu errors\n", traces.n, n_sched,
               n_sched == 1 ? "" : "s", passed, failed, errors);
    for (size_t i = 0; i < traces.n; ++i) free(traces.v[i]);
    for (size_t i = 0; i < paths.n; ++i) free(paths.v[i]);
    for (size_t i = 0; i < scheds.n; ++i) free(scheds.v[i]);
    free(traces.v);
    free(paths.v);
    free(scheds.v);
    if (traces.n == 0 || bad_path) return 2;
    return errors ? 2 : failed ? 1 : 0;
}
