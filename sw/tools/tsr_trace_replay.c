// tsr_trace_replay.c — tsr-trace-replay (docs/design/06-software-verification.md "SW" §5.4, §9.5):
// replays trace directories on fresh simulators and compares everything SW §5.4 lists (the
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
//   --replay-image ELF   (P2.5) replay kind-ops traces through the `replay` kernel of this firmware
//                 image (SW §5.4, §7.7) instead of the tile-only route through the issue port; kind
//                 launch traces are unaffected
//   --replay-kernel NAME the kernel's name in .tsr_kernels (default "replay")
//   --serial      (with --replay-image or --device) the replay kernel's SERIAL flag: every op record
//                 depends on the previous op record (traces/model-v1conv, SW §5.4)
//   --device KIND (P3.3, SW §9.5) replay through a transport instead of fresh simulators (sim, mock,
//                 vkernel, xrt; tsr_devrun.h): the device boots --replay-image (default $TSR_IMAGE, else
//                 the build's tsr-fw.elf); kind ops go through its `replay` kernel, kind launch boots
//                 the trace's own image and pushes its descriptor; compared: the answer with the
//                 trace's DONE and every expect region (filled with their complement first). A trace
//                 made for another configuration than the device's is skipped. --sched, --rv-ratio,
//                 --threads, --max-cycles and --tmax do not apply (the device has its own)
//   --shadow      (with --device) also run every launch on an in-process simulator with the device's
//                 configuration and image, and compare it with the device (answer, per-unit completion
//                 counts, events, log; SW §7.6)
// Output: one line per trace and schedule ("ok" / "FAIL" with the first difference), then
//   tsr-trace-replay: <n> traces x <m> schedules: <p> passed, <f> failed, <e> errors
// or with --device: tsr-trace-replay: <n> traces on <kind>: <p> passed, <f> failed, <s> skipped, <e> errors
// Exit status: 0 all passed, 1 a mismatch, 2 a trace that could not be read or set up (or usage; with
// --device also when no trace fits the device).
#define _GNU_SOURCE

#include "tsr_devrun.h"
#include "tsr_trace.h"
#include "tsr_transport.h"

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

// P3.3: every trace through a device (tsr_devrun.h). Returns the exit status.
static int replay_device(const char * kind, int shadow, const char * image, uint32_t flags, const strlist * traces, int quiet,
                         int verbose) {
    tsr_transport * t = tsr_transport_open(kind);
    if (t == NULL) {
        fprintf(stderr, "tsr-trace-replay: cannot open the \"%s\" transport\n", kind);
        return 2;
    }
    tsr_image * img = NULL;
    const int irc = image ? tsr_image_open(image, &img) : tsr_image_open_default(&img);
    if (irc != 0 || img == NULL) {
        fprintf(stderr, "tsr-trace-replay: no firmware image (%s): %s\n", image ? image : "$TSR_IMAGE / the build's tsr-fw.elf",
                tsr_transport_strerror(irc));
        tsr_transport_close(t);
        return 2;
    }
    char err[512];
    const char * path = tsr_image_path(img);
    tsr_tgt * dev = tsr_tgt_transport(t), * sh = NULL;
    int rc = tsr_tgt_boot(dev, path, err, sizeof(err)) ? 2 : 0;
    if (rc == 0 && shadow) {
        tsr_tgt_cfg c;
        tsr_tgt_cfg_of_transport(t, &c);
        sh = tsr_tgt_shadow(&c, 0);
        if (sh == NULL || tsr_tgt_boot(sh, path, err, sizeof(err))) rc = 2;
    }
    if (rc) fprintf(stderr, "tsr-trace-replay: %s\n", err);
    size_t passed = 0, failed = 0, skipped = 0, errors = 0;
    for (size_t i = 0; i < traces->n && rc == 0; ++i) {
        tsr_trace2 * tr = NULL;
        if (tsr_trace2_read(traces->v[i], &tr, err, sizeof(err)) != 0) {
            printf("ERROR %s: %s\n", traces->v[i], err);
            errors++;
            continue;
        }
        tsr_devrun_opts o;
        memset(&o, 0, sizeof(o));
        o.image = path;
        o.replay_flags = flags;
        o.verbose = verbose;
        tsr_devrun_result r;
        const int k = tsr_devrun_trace(dev, sh, tr, &o, &r);
        tsr_trace2_free(tr);
        if (k == 0) {
            passed++;
            if (!quiet) printf("ok    %s [%s]: %s\n", traces->v[i], kind, r.msg);
        } else if (k == 3) {
            skipped++;
            if (!quiet) printf("skip  %s [%s]: %s\n", traces->v[i], kind, r.msg);
        } else if (k == 1) {
            failed++;
            printf("FAIL  %s [%s]: %s (%u comparison(s) failed, classes 0x%x)\n", traces->v[i], kind, r.msg, r.n_fail, r.fail_mask);
        } else {
            errors++;
            printf("ERROR %s [%s]: %s\n", traces->v[i], kind, r.msg);
        }
        fflush(stdout);
    }
    if (rc == 0)
        printf("tsr-trace-replay: %zu traces on %s%s: %zu passed, %zu failed, %zu skipped, %zu errors\n", traces->n, kind,
               sh ? " (+ shadow)" : "", passed, failed, skipped, errors);
    tsr_tgt_free(sh);
    tsr_tgt_free(dev);
    tsr_image_close(img);
    tsr_transport_close(t);
    if (rc) return rc;
    if (errors || passed + failed == 0) return 2;
    return failed ? 1 : 0;
}

static void usage(void) {
    fprintf(stderr, "usage: tsr-trace-replay [-q | -v] [--sched S]... [--rv-ratio R] [--threads N] [--max-cycles N] [--tmax N]\n"
                    "                        [--replay-image ELF [--replay-kernel NAME] [--serial]] PATH...\n"
                    "       tsr-trace-replay [-q | -v] --device KIND [--shadow] [--replay-image ELF] [--serial] PATH...\n"
                    "  PATH: a trace directory or a set of them (searched recursively for meta.json)\n");
}

int main(int argc, char ** argv) {
    int quiet = 0, verbose = 0;
    unsigned threads = 0;
    uint64_t max_cycles = 0;
    uint32_t tmax = 0;
    double rv_ratio = 0.0;
    const char * replay_image = NULL, * replay_kernel = NULL;   // P2.5: the replay-kernel route
    const char * device = NULL;                                  // P3.3: through a transport
    int shadow = 0;
    uint32_t replay_flags = 0;
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
        else if (strcmp(a, "--replay-image") == 0 && i + 1 < argc) replay_image = argv[++i];        // P2.5
        else if (strcmp(a, "--replay-kernel") == 0 && i + 1 < argc) replay_kernel = argv[++i];
        else if (strcmp(a, "--serial") == 0) replay_flags |= TSR_REPLAY_F_SERIAL;
        else if (strcmp(a, "--device") == 0 && i + 1 < argc) device = argv[++i];                   // P3.3
        else if (strcmp(a, "--shadow") == 0) shadow = 1;
        else if (strcmp(a, "-h") == 0 || strcmp(a, "--help") == 0) { usage(); return 0; }
        else if (a[0] == '-') { fprintf(stderr, "tsr-trace-replay: unknown option %s\n", a); usage(); return 2; }
        else sl_add(&paths, a);
    }
    if (paths.n == 0 || (device == NULL && replay_image == NULL && (replay_kernel != NULL || replay_flags != 0)) ||
        (device != NULL && (replay_kernel != NULL || scheds.n || tmax || rv_ratio != 0.0 || max_cycles)) || (shadow && device == NULL)) {
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
    if (device != NULL && !bad_path) {
        const int rc = replay_device(device, shadow, replay_image, replay_flags, &traces, quiet, verbose);
        for (size_t i = 0; i < traces.n; ++i) free(traces.v[i]);
        for (size_t i = 0; i < paths.n; ++i) free(paths.v[i]);
        free(traces.v);
        free(paths.v);
        return rc;
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
            o.replay_image = replay_image;
            o.replay_kernel = replay_kernel;
            o.replay_flags = replay_flags;
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
