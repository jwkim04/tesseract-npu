// tsr-trace-replay — replay trace directories on the ISA simulator and compare bit-exactly.
//
//   tsr-trace-replay [-v] [-q] [-j THREADS] PATH...
//
// PATH is a trace directory (it contains meta.json) or any directory, which is searched
// recursively for trace directories (visited in sorted order). Each trace is read, replayed
// on a fresh simulator built from its own config, and compared (tsr_trace_replay).
// Exit code: the number of failed traces (unreadable traces and PATHs without any trace count
// as failures), capped at 255. 0 means every trace passed.

#define _GNU_SOURCE

#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "tsr_trace.h"

typedef struct { char ** v; size_t n, cap; } strvec;

static void sv_push(strvec * s, const char * p) {
    if (s->n == s->cap) {
        s->cap = s->cap ? 2 * s->cap : 64;
        s->v = (char **)realloc(s->v, s->cap * sizeof(char *));
        if (!s->v) { fprintf(stderr, "out of memory\n"); exit(255); }
    }
    s->v[s->n] = strdup(p);
    if (!s->v[s->n]) { fprintf(stderr, "out of memory\n"); exit(255); }
    s->n++;
}

static int cmp_str(const void * a, const void * b) { return strcmp(*(char * const *)a, *(char * const *)b); }

static int is_dir(const char * p) {
    struct stat st;
    return stat(p, &st) == 0 && S_ISDIR(st.st_mode);
}
static int has_meta(const char * dir) {
    char * m = (char *)malloc(strlen(dir) + 16);
    sprintf(m, "%s/meta.json", dir);
    struct stat st;
    const int ok = stat(m, &st) == 0 && S_ISREG(st.st_mode);
    free(m);
    return ok;
}

// Collects trace directories under dir (dir itself if it has meta.json), sorted.
static void collect(const char * dir, strvec * out, int depth) {
    if (has_meta(dir)) {
        sv_push(out, dir);
        return;
    }
    if (depth > 32) return;
    DIR * d = opendir(dir);
    if (!d) return;
    strvec subs = {0};
    struct dirent * e;
    while ((e = readdir(d)) != NULL) {
        if (e->d_name[0] == '.') continue;
        char * p = (char *)malloc(strlen(dir) + strlen(e->d_name) + 2);
        sprintf(p, "%s/%s", dir, e->d_name);
        if (is_dir(p)) sv_push(&subs, p);
        free(p);
    }
    closedir(d);
    qsort(subs.v, subs.n, sizeof(char *), cmp_str);
    for (size_t i = 0; i < subs.n; ++i) {
        collect(subs.v[i], out, depth + 1);
        free(subs.v[i]);
    }
    free(subs.v);
}

static void usage(void) {
    fprintf(stderr,
            "usage: tsr-trace-replay [-v] [-q] [-j THREADS] PATH...\n"
            "  PATH: a trace directory or a directory searched recursively for trace directories\n"
            "  -v  verbose (per-trace details)   -q  only print failures and the summary\n"
            "  -j  simulator worker threads (default 1)\n"
            "exit code = number of failed traces (max 255)\n");
}

int main(int argc, char ** argv) {
    int verbose = 0, quiet = 0;
    unsigned threads = 1;
    strvec paths = {0};
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-v") == 0) verbose = 1;
        else if (strcmp(argv[i], "-q") == 0) quiet = 1;
        else if (strcmp(argv[i], "-j") == 0 && i + 1 < argc) {
            const int j = atoi(argv[++i]);
            if (j < 1 || j > 64) { usage(); return 255; }
            threads = (unsigned)j;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) { usage(); return 0; }
        else if (argv[i][0] == '-') { usage(); return 255; }
        else sv_push(&paths, argv[i]);
    }
    if (paths.n == 0) { usage(); return 255; }

    unsigned long n_pass = 0, n_fail = 0;
    for (size_t pi = 0; pi < paths.n; ++pi) {
        strvec traces = {0};
        if (is_dir(paths.v[pi])) collect(paths.v[pi], &traces, 0);
        if (traces.n == 0) {
            fprintf(stderr, "FAIL %s: no trace directories found\n", paths.v[pi]);
            ++n_fail;
        }
        for (size_t t = 0; t < traces.n; ++t) {
            tsr_trace tr;
            int rc;
            if (tsr_trace_read(traces.v[t], &tr) != 0) {
                fprintf(stderr, "FAIL %s: unreadable trace\n", traces.v[t]);
                rc = 1;
            } else {
                rc = tsr_trace_replay_threads(&tr, verbose, threads);
                if (rc != 0) fprintf(stderr, "FAIL %s\n", traces.v[t]);
                else if (!quiet) printf("PASS %s (%u insns, %zu init / %zu expect regions)\n", traces.v[t], tr.cmd_count,
                                        tr.cap.n_init, tr.cap.n_expect);
                tsr_trace_free(&tr);
            }
            if (rc) ++n_fail; else ++n_pass;
            free(traces.v[t]);
        }
        free(traces.v);
        free(paths.v[pi]);
    }
    free(paths.v);
    printf("tsr-trace-replay: %lu passed, %lu failed\n", n_pass, n_fail);
    fflush(stdout);
    return n_fail > 255 ? 255 : (int)n_fail;
}
