// tsr_trace.c — trace format v2 (docs/v2/06-software-verification.md "SW" §5; API and file format
// in include/tsr_trace.h): the trace object, the meta.json writer and its strict reader, push.bin /
// tickets.bin / region files, the EX §3.2 image id (SHA-256) and the image checks of EX §3.1, the
// TRACE model of the simulator (launch capture, SW §5.5, attached by tsr_sim_create_opts), and
// replay on a fresh simulator with the comparisons of SW §5.4.
//
// Decisions (P1.10, where SW §5 leaves a choice):
//   - Kernel entry, the base of relative tickets (SW §5.2, EX §4.7 R5): the first RISC-V read of
//     ISSUE_PIDX after the launch's LQ_POP. The SDK runtime reads ISSUE_PIDX, then ISSUED_SEQ[u],
//     right before it calls the kernel (SW §6.3 R5), so this is a firmware-visible marker that
//     needs no new register. A firmware that never reads it gets the counters at its LQ_POP.
//     Pushes between LQ_POP and kernel entry (the argument-block LD) get relative tickets below the
//     base (mod 2^28), the same in capture and replay.
//   - DONE r6 (ERR_PIDX) is stored relative only when r3 names a unit (a failing ticket); without
//     one, r3..r6 are stored as posted. tickets.bin error fields are all 0 without an error record.
//   - Captured are LAUNCH descriptors without reserved bits answered by a DONE record. PING, other
//     descriptor types, TRAP answers and launches ended by a reset are skipped (and counted).
//   - Image: the ELF given to tsr_sim_capture_set_image, else one synthesised from the loader's
//     writes (one PT_LOAD per maximal run of loaded words, contents at the RV_RUN rise, entry
//     RV_BOOT + 0x80, no section headers); its id is EX §3.2's hash of those segments. The HELLO
//     id is not compared (test firmware posts constants).
//   - Kind ops is replayed tile-only by default: each record goes in through the control window
//     from outside the time base, waiting for its credit or FIFO space by advancing to the next
//     event (what the replay kernel does with WAIT_CREDIT), then the run continues until QIDLE. Its
//     DONE is the replay kernel's answer (SW §7.7 step 4). P2.5 adds the route through the `replay`
//     kernel of a firmware image (tsr_trace2_replay_opts.replay_image; "kernel_replay" below).
//   - CTL_ID / CFG3 in "config" are informational on replay (CTL_ID.DM depends on the models
//     linked into the library): a difference is a warning, not a failure.
//   - meta.json: integers in decimal (ret signed), rv_ratio with 17 significant digits, the
//     image id as 16 lower-case hex digits; unknown keys are refused on read.
//   - W2a: the launch's own LAUNCH_BEGIN / LAUNCH_END markers (DB §8.4: class 1 / 2, [27:16] the
//     descriptor's KERNEL, [15:0] the launch SEQ low bits; the SDK runtime pushes them around every
//     kernel) are stored launch-relative: SEQ field 0, like the relative tickets and push indices.
//     Otherwise a launch captured with SEQ s != 0 could never replay on a freshly booted simulator,
//     where it runs as SEQ 0 (SW §10.1 "launch traces"). Only a marker naming exactly this launch's
//     kernel and SEQ is rewritten, so the random markers of the generated sets pass unchanged (and
//     on a fresh simulator, SEQ 0, the rewrite is the identity).

#define _GNU_SOURCE

#include "tsr_trace.h"
#include "tsr_sim_int.h"
#if defined(TSR_SIM_HAVE_RACE) && TSR_SIM_HAVE_RACE
#include "tsr_race.h"     // P2.5: race report counts of a replay
#endif

#include <dirent.h>
#include <errno.h>
#include <limits.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define SEQ_MASK      0x0FFFFFFFu
#define IMG_SEG_MAX   64u                    // PT_LOAD segments of an image
#define IMG_MAX_BYTES (256u << 20)           // an image file larger than this is refused
#define PUSH_MAX      (1ull << 32)           // records of one push.bin
#define TCM_LIMIT     (1u << 20)             // ITCM / DTCM bytes at most (tsr_sim_opts)

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------
static void * xmalloc(size_t n) {
    void * p = malloc(n ? n : 1);
    if (p == NULL) {
        fprintf(stderr, "tsr_trace: out of memory (%zu bytes)\n", n);
        abort();
    }
    return p;
}
static void * xcalloc(size_t n, size_t s) {
    void * p = calloc(n ? n : 1, s ? s : 1);
    if (p == NULL) {
        fprintf(stderr, "tsr_trace: out of memory (%zu x %zu bytes)\n", n, s);
        abort();
    }
    return p;
}
static void * xrealloc(void * p, size_t n) {
    void * q = realloc(p, n ? n : 1);
    if (q == NULL) {
        fprintf(stderr, "tsr_trace: out of memory (%zu bytes)\n", n);
        abort();
    }
    return q;
}
static char * xstrdup(const char * s) {
    const size_t n = strlen(s) + 1;
    char * d = (char *)xmalloc(n);
    memcpy(d, s, n);
    return d;
}
static char * path_join(const char * a, const char * b) {
    const size_t n = strlen(a) + strlen(b) + 2;
    char * p = (char *)xmalloc(n);
    snprintf(p, n, "%s/%s", a, b);
    return p;
}

static void seterr(char * err, size_t n, const char * fmt, ...) __attribute__((format(printf, 3, 4)));
static void seterr(char * err, size_t n, const char * fmt, ...) {
    if (err == NULL || n == 0) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, n, fmt, ap);
    va_end(ap);
}

static inline uint32_t rd32(const uint8_t * p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static inline uint16_t rd16(const uint8_t * p) { return (uint16_t)(p[0] | (p[1] << 8)); }
static inline void wr32(uint8_t * p, uint32_t v) {
    p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24);
}
static inline void wr16(uint8_t * p, uint32_t v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); }

// ---------------------------------------------------------------------------
// Files
// ---------------------------------------------------------------------------
static int mkdir_p(const char * path, char * err, size_t en) {
    char * p = xstrdup(path);
    const size_t n = strlen(p);
    for (size_t i = 1; i <= n; ++i) {
        if (p[i] == '/' || p[i] == '\0') {
            const char c = p[i];
            p[i] = '\0';
            if (mkdir(p, 0755) != 0 && errno != EEXIST) {
                seterr(err, en, "mkdir %s: %s", p, strerror(errno));
                free(p);
                return -1;
            }
            p[i] = c;
        }
    }
    free(p);
    struct stat st;
    if (stat(path, &st) != 0 || !S_ISDIR(st.st_mode)) {
        seterr(err, en, "%s is not a directory", path);
        return -1;
    }
    return 0;
}

// Writes a whole file through <path>.tmp.<pid> and a rename (readers never see a partial file).
static int write_file(const char * path, const void * data, size_t size, char * err, size_t en) {
    const size_t L = strlen(path) + 32;
    char * tmp = (char *)xmalloc(L);
    snprintf(tmp, L, "%s.tmp.%ld", path, (long)getpid());
    FILE * f = fopen(tmp, "wb");
    if (f == NULL) {
        seterr(err, en, "cannot create %s: %s", tmp, strerror(errno));
        free(tmp);
        return -1;
    }
    const size_t w = size ? fwrite(data, 1, size, f) : 0;
    const int rc_close = fclose(f);
    if (w != size || rc_close != 0) {
        seterr(err, en, "write error on %s", tmp);
        unlink(tmp);
        free(tmp);
        return -1;
    }
    if (rename(tmp, path) != 0) {
        seterr(err, en, "rename %s -> %s: %s", tmp, path, strerror(errno));
        unlink(tmp);
        free(tmp);
        return -1;
    }
    free(tmp);
    return 0;
}

// Reads a whole regular file (NUL appended). NULL with err set on failure.
static uint8_t * read_file(const char * path, size_t * size, size_t max, char * err, size_t en) {
    FILE * f = fopen(path, "rb");
    if (f == NULL) {
        seterr(err, en, "cannot open %s: %s", path, strerror(errno));
        return NULL;
    }
    struct stat st;
    if (fstat(fileno(f), &st) != 0 || !S_ISREG(st.st_mode)) {
        seterr(err, en, "%s is not a regular file", path);
        fclose(f);
        return NULL;
    }
    if ((uint64_t)st.st_size > (uint64_t)max) {
        seterr(err, en, "%s: %llu bytes, more than the %zu allowed", path, (unsigned long long)st.st_size, max);
        fclose(f);
        return NULL;
    }
    const size_t n = (size_t)st.st_size;
    uint8_t * buf = (uint8_t *)xmalloc(n + 1);
    if (n && fread(buf, 1, n, f) != n) {
        seterr(err, en, "read error on %s", path);
        free(buf);
        fclose(f);
        return NULL;
    }
    fclose(f);
    buf[n] = 0;
    *size = n;
    return buf;
}

static void region_name(char * buf, size_t n, const char * sub, uint32_t ch, uint64_t off) {
    snprintf(buf, n, "%s/ch%u_%010llx.bin", sub, ch, (unsigned long long)off);
}

// Removes the files of an earlier trace in dir (meta.json first, so a half-removed directory is
// never a trace).
static void remove_stale(const char * dir) {
    static const char * const files[] = { "meta.json", "ops/push.bin", "expect/tickets.bin" };
    for (size_t i = 0; i < sizeof(files) / sizeof(files[0]); ++i) {
        char * p = path_join(dir, files[i]);
        unlink(p);
        free(p);
    }
    static const char * const subs[2] = { "init", "expect" };
    for (int s = 0; s < 2; ++s) {
        char * sd = path_join(dir, subs[s]);
        DIR * d = opendir(sd);
        if (d != NULL) {
            struct dirent * e;
            while ((e = readdir(d)) != NULL) {
                const size_t L = strlen(e->d_name);
                if (L > 6 && strncmp(e->d_name, "ch", 2) == 0 && strchr(e->d_name, '_') != NULL &&
                    strcmp(e->d_name + L - 4, ".bin") == 0) {
                    char * f = path_join(sd, e->d_name);
                    unlink(f);
                    free(f);
                }
            }
            closedir(d);
        }
        free(sd);
    }
}

// Relative path from directory `from` to file `to` (both must exist). NULL on failure.
static char * relpath(const char * from, const char * to) {
    char * a = realpath(from, NULL);
    char * b = realpath(to, NULL);
    if (a == NULL || b == NULL) {
        free(a);
        free(b);
        return NULL;
    }
    // Common prefix by whole components.
    size_t i = 0, last = 0;
    while (a[i] && b[i] && a[i] == b[i]) {
        if (a[i] == '/') last = i;
        ++i;
    }
    if (a[i] == '\0' && b[i] == '/') last = i;   // a is a prefix directory of b
    size_t ups = 0;
    if (a[last] != '\0') {
        ups = 1;
        for (size_t k = last + 1; a[k]; ++k) ups += a[k] == '/';
    }
    const char * rest = b + last + 1;
    char * r = (char *)xmalloc(3 * ups + strlen(rest) + 1);
    r[0] = '\0';
    for (size_t k = 0; k < ups; ++k) strcat(r, "../");
    strcat(r, rest);
    free(a);
    free(b);
    return r;
}

// A relative path without "." / ".." components (region and push files live inside the trace).
static int inside_rel_path(const char * p) {
    if (p[0] == '\0' || p[0] == '/') return 0;
    const char * s = p;
    for (;;) {
        const char * e = strchr(s, '/');
        const size_t L = e ? (size_t)(e - s) : strlen(s);
        if (L == 0 || (L == 1 && s[0] == '.') || (L == 2 && s[0] == '.' && s[1] == '.')) return 0;
        if (!e) break;
        s = e + 1;
    }
    return 1;
}

// ---------------------------------------------------------------------------
// Minimal JSON DOM (objects, arrays, strings with escapes, numbers, true / false / null)
// ---------------------------------------------------------------------------
typedef enum { J_NULL, J_BOOL, J_NUM, J_STR, J_ARR, J_OBJ } jtype;

typedef struct jval {
    jtype         t;
    int           b;          // J_BOOL
    int           is_uint;    // J_NUM: a non-negative integer that fits in uint64
    int           is_int;     // J_NUM: an integer that fits in int64
    uint64_t      u;
    int64_t       i;
    double        d;
    char *        s;          // J_STR
    struct jval * items;      // J_ARR / J_OBJ
    char **       keys;       // J_OBJ
    size_t        n;
} jval;

typedef struct { const char * p; const char * end; const char * begin; const char * err; } jparser;

static void jfree(jval * v) {
    if (v == NULL) return;
    free(v->s);
    for (size_t i = 0; i < v->n; ++i) {
        jfree(&v->items[i]);
        if (v->keys) free(v->keys[i]);
    }
    free(v->items);
    free(v->keys);
    memset(v, 0, sizeof(*v));
}

static void jws(jparser * ps) {
    while (ps->p < ps->end && (*ps->p == ' ' || *ps->p == '\t' || *ps->p == '\n' || *ps->p == '\r')) ps->p++;
}

static int jfail(jparser * ps, const char * msg) {
    if (ps->err == NULL) ps->err = msg;
    return -1;
}

static void utf8_put(char ** out, size_t * n, size_t * cap, uint32_t cp) {
    char tmp[4];
    size_t k;
    if (cp < 0x80) { tmp[0] = (char)cp; k = 1; }
    else if (cp < 0x800) { tmp[0] = (char)(0xC0 | (cp >> 6)); tmp[1] = (char)(0x80 | (cp & 0x3F)); k = 2; }
    else if (cp < 0x10000) {
        tmp[0] = (char)(0xE0 | (cp >> 12)); tmp[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        tmp[2] = (char)(0x80 | (cp & 0x3F)); k = 3;
    } else {
        tmp[0] = (char)(0xF0 | (cp >> 18)); tmp[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
        tmp[2] = (char)(0x80 | ((cp >> 6) & 0x3F)); tmp[3] = (char)(0x80 | (cp & 0x3F)); k = 4;
    }
    if (*n + k + 1 > *cap) {
        *cap = (*n + k + 1) * 2;
        *out = (char *)xrealloc(*out, *cap);
    }
    memcpy(*out + *n, tmp, k);
    *n += k;
}

static int jhex4(jparser * ps, uint32_t * v) {
    if (ps->end - ps->p < 4) return jfail(ps, "truncated \\u escape");
    uint32_t r = 0;
    for (int i = 0; i < 4; ++i) {
        const char c = ps->p[i];
        r <<= 4;
        if (c >= '0' && c <= '9') r |= (uint32_t)(c - '0');
        else if (c >= 'a' && c <= 'f') r |= (uint32_t)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') r |= (uint32_t)(c - 'A' + 10);
        else return jfail(ps, "bad \\u escape");
    }
    ps->p += 4;
    *v = r;
    return 0;
}

static int jstring(jparser * ps, char ** out_s) {
    ps->p++;                                          // the opening quote
    size_t n = 0, cap = 32;
    char * s = (char *)xmalloc(cap);
    for (;;) {
        if (ps->p >= ps->end) { free(s); return jfail(ps, "unterminated string"); }
        const unsigned char c = (unsigned char)*ps->p++;
        if (c == '"') break;
        if (c < 0x20) { free(s); return jfail(ps, "control character in string"); }
        if (c == '\\') {
            if (ps->p >= ps->end) { free(s); return jfail(ps, "unterminated escape"); }
            const char e = *ps->p++;
            uint32_t cp = 0;
            switch (e) {
            case '"': cp = '"'; break;
            case '\\': cp = '\\'; break;
            case '/': cp = '/'; break;
            case 'b': cp = '\b'; break;
            case 'f': cp = '\f'; break;
            case 'n': cp = '\n'; break;
            case 'r': cp = '\r'; break;
            case 't': cp = '\t'; break;
            case 'u':
                if (jhex4(ps, &cp)) { free(s); return -1; }
                if (cp >= 0xD800 && cp < 0xDC00 && ps->end - ps->p >= 6 && ps->p[0] == '\\' && ps->p[1] == 'u') {
                    ps->p += 2;
                    uint32_t lo;
                    if (jhex4(ps, &lo)) { free(s); return -1; }
                    if (lo >= 0xDC00 && lo < 0xE000) cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    else { free(s); return jfail(ps, "bad surrogate pair"); }
                }
                if (cp == 0) { free(s); return jfail(ps, "NUL character in string"); }
                break;
            default: free(s); return jfail(ps, "bad escape");
            }
            utf8_put(&s, &n, &cap, cp);
        } else {
            if (n + 2 > cap) { cap *= 2; s = (char *)xrealloc(s, cap); }
            s[n++] = (char)c;
        }
    }
    s[n] = '\0';
    *out_s = s;
    return 0;
}

static int jnumber(jparser * ps, jval * v) {
    const char * st = ps->p;
    int neg = 0, frac = 0, overflow = 0;
    if (ps->p < ps->end && *ps->p == '-') { neg = 1; ps->p++; }
    if (ps->p >= ps->end || *ps->p < '0' || *ps->p > '9') return jfail(ps, "bad number");
    uint64_t u = 0;
    if (*ps->p == '0') ps->p++;
    else {
        while (ps->p < ps->end && *ps->p >= '0' && *ps->p <= '9') {
            const uint64_t d = (uint64_t)(*ps->p - '0');
            if (u > (UINT64_MAX - d) / 10u) overflow = 1;
            u = u * 10u + d;
            ps->p++;
        }
    }
    if (ps->p < ps->end && *ps->p == '.') {
        frac = 1;
        ps->p++;
        if (ps->p >= ps->end || *ps->p < '0' || *ps->p > '9') return jfail(ps, "bad number");
        while (ps->p < ps->end && *ps->p >= '0' && *ps->p <= '9') ps->p++;
    }
    if (ps->p < ps->end && (*ps->p == 'e' || *ps->p == 'E')) {
        frac = 1;
        ps->p++;
        if (ps->p < ps->end && (*ps->p == '+' || *ps->p == '-')) ps->p++;
        if (ps->p >= ps->end || *ps->p < '0' || *ps->p > '9') return jfail(ps, "bad number");
        while (ps->p < ps->end && *ps->p >= '0' && *ps->p <= '9') ps->p++;
    }
    v->t = J_NUM;
    v->is_uint = !neg && !frac && !overflow;
    v->u = v->is_uint ? u : 0;
    v->is_int = !frac && !overflow && (neg ? u <= (1ull << 63) : u <= (uint64_t)INT64_MAX);
    if (v->is_int) v->i = neg ? (u == (1ull << 63) ? INT64_MIN : -(int64_t)u) : (int64_t)u;
    v->d = strtod(st, NULL);                          // the text is NUL-terminated after the document
    return 0;
}

static int jvalue(jparser * ps, jval * v, int depth) {
    memset(v, 0, sizeof(*v));
    if (depth > 32) return jfail(ps, "nesting too deep");
    jws(ps);
    if (ps->p >= ps->end) return jfail(ps, "unexpected end of input");
    const char c = *ps->p;
    if (c == '{' || c == '[') {
        const int obj = c == '{';
        v->t = obj ? J_OBJ : J_ARR;
        ps->p++;
        jws(ps);
        if (ps->p < ps->end && *ps->p == (obj ? '}' : ']')) { ps->p++; return 0; }
        size_t cap = 0;
        for (;;) {
            jws(ps);
            char * key = NULL;
            if (obj) {
                if (ps->p >= ps->end || *ps->p != '"') return jfail(ps, "expected object key");
                if (jstring(ps, &key)) return -1;
                jws(ps);
                if (ps->p >= ps->end || *ps->p != ':') { free(key); return jfail(ps, "expected ':'"); }
                ps->p++;
            }
            if (v->n == cap) {
                cap = cap ? 2 * cap : 8;
                v->items = (jval *)xrealloc(v->items, cap * sizeof(jval));
                if (obj) v->keys = (char **)xrealloc(v->keys, cap * sizeof(char *));
            }
            if (obj) v->keys[v->n] = key;
            const int rc = jvalue(ps, &v->items[v->n], depth + 1);
            v->n++;
            if (rc) return -1;
            jws(ps);
            if (ps->p < ps->end && *ps->p == ',') { ps->p++; continue; }
            if (ps->p < ps->end && *ps->p == (obj ? '}' : ']')) { ps->p++; return 0; }
            return jfail(ps, obj ? "expected ',' or '}'" : "expected ',' or ']'");
        }
    }
    if (c == '"') {
        v->t = J_STR;
        return jstring(ps, &v->s);
    }
    if (c == '-' || (c >= '0' && c <= '9')) return jnumber(ps, v);
    if (ps->end - ps->p >= 4 && strncmp(ps->p, "true", 4) == 0)  { v->t = J_BOOL; v->b = 1; ps->p += 4; return 0; }
    if (ps->end - ps->p >= 5 && strncmp(ps->p, "false", 5) == 0) { v->t = J_BOOL; v->b = 0; ps->p += 5; return 0; }
    if (ps->end - ps->p >= 4 && strncmp(ps->p, "null", 4) == 0)  { v->t = J_NULL; ps->p += 4; return 0; }
    return jfail(ps, "unexpected character");
}

static int jparse(const char * text, size_t len, jval * root, const char * what, char * err, size_t en) {
    jparser ps = { text, text + len, text, NULL };
    if (jvalue(&ps, root, 0) == 0) {
        jws(&ps);
        if (ps.p == ps.end) return 0;
        jfail(&ps, "trailing characters after the JSON value");
    }
    seterr(err, en, "%s: JSON error at byte %ld: %s", what, (long)(ps.p - ps.begin), ps.err ? ps.err : "?");
    jfree(root);
    return -1;
}

static const jval * jget(const jval * obj, const char * key) {
    if (obj == NULL || obj->t != J_OBJ) return NULL;
    for (size_t i = 0; i < obj->n; ++i)
        if (strcmp(obj->keys[i], key) == 0) return &obj->items[i];
    return NULL;
}

// Reader context: the file being read and the error sink.
typedef struct rctx { const char * file; char * err; size_t en; } rctx;

static int rfail(rctx * c, const char * fmt, ...) __attribute__((format(printf, 2, 3)));
static int rfail(rctx * c, const char * fmt, ...) {
    char msg[400];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof(msg), fmt, ap);
    va_end(ap);
    seterr(c->err, c->en, "%s: %s", c->file, msg);
    return -1;
}

// obj must be an object whose keys are all in allowed (NULL-terminated), each at most once.
static int jkeys(rctx * c, const jval * obj, const char * what, const char * const * allowed) {
    if (obj == NULL || obj->t != J_OBJ) return rfail(c, "%s must be an object", what);
    for (size_t i = 0; i < obj->n; ++i) {
        int ok = 0;
        for (size_t k = 0; allowed[k]; ++k) ok |= strcmp(obj->keys[i], allowed[k]) == 0;
        if (!ok) return rfail(c, "%s: unknown key \"%s\"", what, obj->keys[i]);
        for (size_t j = 0; j < i; ++j)
            if (strcmp(obj->keys[i], obj->keys[j]) == 0) return rfail(c, "%s: duplicate key \"%s\"", what, obj->keys[i]);
    }
    return 0;
}

static int jreq(rctx * c, const jval * obj, const char * key, jtype t, const char * what, const jval ** out) {
    static const char * const tn[] = { "null", "a boolean", "a number", "a string", "an array", "an object" };
    const jval * v = jget(obj, key);
    if (v == NULL) return rfail(c, "%s: missing \"%s\"", what, key);
    if (v->t != t) return rfail(c, "%s: \"%s\" must be %s", what, key, tn[t]);
    *out = v;
    return 0;
}

static int jnum_u64(rctx * c, const jval * v, uint64_t max, uint64_t * out, const char * what, const char * key) {
    if (v->t != J_NUM || !v->is_uint || v->u > max)
        return rfail(c, "%s: \"%s\" must be an unsigned integer <= %llu", what, key, (unsigned long long)max);
    *out = v->u;
    return 0;
}

static int jreq_u64(rctx * c, const jval * obj, const char * key, uint64_t max, uint64_t * out, const char * what) {
    const jval * v = jget(obj, key);
    if (v == NULL) return rfail(c, "%s: missing \"%s\"", what, key);
    return jnum_u64(c, v, max, out, what, key);
}

static int jreq_u32(rctx * c, const jval * obj, const char * key, uint32_t max, uint32_t * out, const char * what) {
    uint64_t v = 0;
    if (jreq_u64(c, obj, key, max, &v, what)) return -1;
    *out = (uint32_t)v;
    return 0;
}

// ---------------------------------------------------------------------------
// SHA-256 (FIPS 180-4), for the EX §3.2 image id
// ---------------------------------------------------------------------------
typedef struct { uint32_t h[8]; uint8_t buf[64]; uint64_t len; size_t fill; } sha256;

static const uint32_t k_sha[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
};

static inline uint32_t ror32(uint32_t x, unsigned n) { return (x >> n) | (x << (32u - n)); }

static void sha_block(sha256 * s, const uint8_t * p) {
    uint32_t w[64];
    for (int i = 0; i < 16; ++i)
        w[i] = ((uint32_t)p[4 * i] << 24) | ((uint32_t)p[4 * i + 1] << 16) | ((uint32_t)p[4 * i + 2] << 8) | p[4 * i + 3];
    for (int i = 16; i < 64; ++i) {
        const uint32_t s0 = ror32(w[i - 15], 7) ^ ror32(w[i - 15], 18) ^ (w[i - 15] >> 3);
        const uint32_t s1 = ror32(w[i - 2], 17) ^ ror32(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = s->h[0], b = s->h[1], c = s->h[2], d = s->h[3], e = s->h[4], f = s->h[5], g = s->h[6], h = s->h[7];
    for (int i = 0; i < 64; ++i) {
        const uint32_t t1 = h + (ror32(e, 6) ^ ror32(e, 11) ^ ror32(e, 25)) + ((e & f) ^ (~e & g)) + k_sha[i] + w[i];
        const uint32_t t2 = (ror32(a, 2) ^ ror32(a, 13) ^ ror32(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
        h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    s->h[0] += a; s->h[1] += b; s->h[2] += c; s->h[3] += d; s->h[4] += e; s->h[5] += f; s->h[6] += g; s->h[7] += h;
}

static void sha_init(sha256 * s) {
    static const uint32_t h0[8] = { 0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                                    0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u };
    memcpy(s->h, h0, sizeof(h0));
    s->len = 0;
    s->fill = 0;
}

static void sha_update(sha256 * s, const void * data, size_t n) {
    const uint8_t * p = (const uint8_t *)data;
    s->len += n;
    while (n > 0) {
        size_t k = 64 - s->fill;
        if (k > n) k = n;
        memcpy(s->buf + s->fill, p, k);
        s->fill += k;
        p += k;
        n -= k;
        if (s->fill == 64) {
            sha_block(s, s->buf);
            s->fill = 0;
        }
    }
}

static void sha_final(sha256 * s, uint8_t out[32]) {
    const uint64_t bits = s->len * 8u;
    const uint8_t one = 0x80, zero = 0;
    sha_update(s, &one, 1);
    while (s->fill != 56) sha_update(s, &zero, 1);
    uint8_t L[8];
    for (int i = 0; i < 8; ++i) L[i] = (uint8_t)(bits >> (56 - 8 * i));
    sha_update(s, L, 8);
    for (int i = 0; i < 8; ++i) {
        out[4 * i] = (uint8_t)(s->h[i] >> 24); out[4 * i + 1] = (uint8_t)(s->h[i] >> 16);
        out[4 * i + 2] = (uint8_t)(s->h[i] >> 8); out[4 * i + 3] = (uint8_t)s->h[i];
    }
}

// ---------------------------------------------------------------------------
// Firmware images: ELF32 RISC-V executables (EX §3.1-§3.2, SW §6.2)
// ---------------------------------------------------------------------------
#define EM_RISCV_  243u
#define PT_LOAD_   1u
#define SHT_NOBITS_ 8u

typedef struct img_seg { uint32_t paddr, filesz, memsz; uint64_t off; } img_seg;
typedef struct img_info {
    uint32_t entry;
    uint32_t nseg;
    img_seg  seg[IMG_SEG_MAX];      // PT_LOAD segments, ascending paddr
    int      has_id;                // ".tsr_id" (8 bytes at file offset id_off)
    uint64_t id_off;
    int      has_kernels;           // ".tsr_kernels" bytes [k_off, k_off + k_size)
    uint64_t k_off, k_size;
} img_info;

static int seg_cmp(const void * a, const void * b) {
    const img_seg * x = (const img_seg *)a, * y = (const img_seg *)b;
    return (x->paddr > y->paddr) - (x->paddr < y->paddr);
}

// Parses and checks an image against EX §3.1 with the given TCM sizes.
static int img_parse(const uint8_t * b, size_t n, uint32_t itcm, uint32_t dtcm, img_info * e, char * err, size_t en) {
    memset(e, 0, sizeof(*e));
    if (n < 52 || memcmp(b, "\177ELF", 4) != 0) { seterr(err, en, "not an ELF file"); return -1; }
    if (b[4] != 1 || b[5] != 1) { seterr(err, en, "not a 32-bit little-endian ELF"); return -1; }
    if (rd16(b + 18) != EM_RISCV_) { seterr(err, en, "not a RISC-V ELF (e_machine %u)", rd16(b + 18)); return -1; }
    if (rd16(b + 16) != 2) { seterr(err, en, "not an executable (e_type %u, EX §3.1: static, no relocations)", rd16(b + 16)); return -1; }
    e->entry = rd32(b + 24);
    const uint64_t phoff = rd32(b + 28), shoff = rd32(b + 32);
    const uint32_t phentsize = rd16(b + 42), phnum = rd16(b + 44), shentsize = rd16(b + 46), shnum = rd16(b + 48),
                   shstrndx = rd16(b + 50);
    if (phnum > 0 && (phentsize != 32 || phoff + 32ull * phnum > n)) { seterr(err, en, "bad program header table"); return -1; }
    for (uint32_t i = 0; i < phnum; ++i) {
        const uint8_t * ph = b + phoff + 32ull * i;
        if (rd32(ph) != PT_LOAD_) continue;
        img_seg s = { rd32(ph + 12), rd32(ph + 16), rd32(ph + 20), rd32(ph + 4) };
        if (e->nseg == IMG_SEG_MAX) { seterr(err, en, "more than %u PT_LOAD segments", IMG_SEG_MAX); return -1; }
        if (s.off + s.filesz > n) { seterr(err, en, "PT_LOAD %u: file data beyond the end of the file", i); return -1; }
        if (s.filesz > s.memsz) { seterr(err, en, "PT_LOAD %u: p_filesz > p_memsz", i); return -1; }
        if (s.paddr % 4 != 0) { seterr(err, en, "PT_LOAD %u: physical address 0x%08x is not word aligned (EX §3.1)", i, s.paddr); return -1; }
        const uint64_t lo = s.paddr, hi = (uint64_t)s.paddr + s.memsz;
        const int in_itcm = hi <= itcm;
        const int in_dtcm = lo >= TSR_RV_DTCM_BASE && hi <= (uint64_t)TSR_RV_DTCM_BASE + dtcm;
        if (!in_itcm && !in_dtcm) {
            seterr(err, en, "PT_LOAD %u: [0x%08llx, 0x%08llx) is not inside ITCM [0, 0x%x) or DTCM [0x%x, 0x%x) (EX §3.1)", i,
                   (unsigned long long)lo, (unsigned long long)hi, itcm, TSR_RV_DTCM_BASE, TSR_RV_DTCM_BASE + dtcm);
            return -1;
        }
        e->seg[e->nseg++] = s;
    }
    if (e->nseg == 0) { seterr(err, en, "no PT_LOAD segment"); return -1; }
    qsort(e->seg, e->nseg, sizeof(img_seg), seg_cmp);
    for (uint32_t i = 1; i < e->nseg; ++i) {
        const uint64_t prev_hi = (uint64_t)e->seg[i - 1].paddr + ((e->seg[i - 1].memsz + 3u) & ~3ull);
        if (e->seg[i].paddr < prev_hi) { seterr(err, en, "PT_LOAD segments overlap at 0x%08x (EX §3.1)", e->seg[i].paddr); return -1; }
    }
    // Section headers are optional (a synthesised image has none): .tsr_id and .tsr_kernels.
    if (shoff != 0 && shnum != 0) {
        if (shentsize != 40 || shoff + 40ull * shnum > n || shstrndx >= shnum) { seterr(err, en, "bad section header table"); return -1; }
        const uint8_t * ss = b + shoff + 40ull * shstrndx;
        const uint64_t str_off = rd32(ss + 16), str_size = rd32(ss + 20);
        if (str_off + str_size > n) { seterr(err, en, "bad section name table"); return -1; }
        for (uint32_t i = 0; i < shnum; ++i) {
            const uint8_t * sh = b + shoff + 40ull * i;
            const uint64_t name = rd32(sh), type = rd32(sh + 4), off = rd32(sh + 16), size = rd32(sh + 20);
            if (name >= str_size) continue;
            const char * nm = (const char *)b + str_off + name;
            const size_t maxlen = (size_t)(str_size - name);
            if (strnlen(nm, maxlen) == maxlen) continue;
            if (strcmp(nm, ".tsr_id") == 0) {
                if (type == SHT_NOBITS_ || size != 8 || off + 8 > n) { seterr(err, en, ".tsr_id must hold 8 bytes of file data"); return -1; }
                e->has_id = 1;
                e->id_off = off;
            } else if (strcmp(nm, ".tsr_kernels") == 0 && type != SHT_NOBITS_ && off + size <= n) {
                e->has_kernels = 1;
                e->k_off = off;
                e->k_size = size;
            }
        }
    }
    return 0;
}

// EX §3.2 over parsed segments.
static uint64_t img_id(const uint8_t * b, const img_info * e) {
    sha256 s;
    sha_init(&s);
    for (uint32_t i = 0; i < e->nseg; ++i) {
        const img_seg * g = &e->seg[i];
        const uint32_t padded = (g->filesz + 3u) & ~3u;
        uint8_t hdr[8];
        wr32(hdr, g->paddr);
        wr32(hdr + 4, padded);
        sha_update(&s, hdr, 8);
        if (padded == 0) continue;
        uint8_t * tmp = (uint8_t *)xcalloc(padded, 1);
        memcpy(tmp, b + g->off, g->filesz);
        if (e->has_id && e->id_off + 8 > g->off && e->id_off < g->off + g->filesz) {   // .tsr_id taken as zero
            for (uint64_t k = e->id_off; k < e->id_off + 8; ++k)
                if (k >= g->off && k < g->off + g->filesz) tmp[k - g->off] = 0;
        }
        sha_update(&s, tmp, padded);
        free(tmp);
    }
    uint8_t d[32];
    sha_final(&s, d);
    uint64_t id = 0;
    for (int i = 7; i >= 0; --i) id = (id << 8) | d[i];
    return id;
}

// Kernel name at index from .tsr_kernels (SW §6.2 entry: name_hash, entry, nargs, flags,
// argblk_lines, name[24]: a 36-byte stride as declared), "" if absent.
static void img_kernel_name(const uint8_t * b, const img_info * e, uint32_t index, char out[32]) {
    out[0] = '\0';
    const uint64_t stride = 36;
    if (!e->has_kernels || e->k_size % stride != 0 || (uint64_t)index >= e->k_size / stride) return;
    const uint8_t * nm = b + e->k_off + stride * index + 12;
    size_t L = 0;
    while (L < 24 && nm[L] >= 0x20 && nm[L] < 0x7F && nm[L] != '"' && nm[L] != '\\') ++L;
    if (L < 24 && nm[L] != 0) L = 0;                 // not a NUL-padded printable name
    memcpy(out, nm, L);
    out[L] = '\0';
}

int tsr_trace2_image_id(const uint8_t * elf, size_t n, uint64_t * id, char * err, size_t errlen) {
    img_info e;
    if (elf == NULL || img_parse(elf, n, TCM_LIMIT, TCM_LIMIT, &e, err, errlen)) return -1;
    *id = img_id(elf, &e);
    return 0;
}

// A minimal ELF32 RISC-V executable: one RWX PT_LOAD per segment, no sections.
static uint8_t * img_synth(const img_seg * seg, const uint8_t * const * data, uint32_t nseg, uint32_t entry,
                           size_t * n_out) {
    size_t n = 52 + 32u * nseg;
    for (uint32_t i = 0; i < nseg; ++i) n += seg[i].filesz;
    uint8_t * b = (uint8_t *)xcalloc(n, 1);
    memcpy(b, "\177ELF", 4);
    b[4] = 1; b[5] = 1; b[6] = 1;                    // ELFCLASS32, ELFDATA2LSB, EV_CURRENT
    wr16(b + 16, 2);                                 // ET_EXEC
    wr16(b + 18, EM_RISCV_);
    wr32(b + 20, 1);
    wr32(b + 24, entry);
    wr32(b + 28, 52);                                // e_phoff
    wr16(b + 40, 52);                                // e_ehsize
    wr16(b + 42, 32);
    wr16(b + 44, nseg);
    wr16(b + 46, 40);
    size_t off = 52 + 32u * nseg;
    for (uint32_t i = 0; i < nseg; ++i) {
        uint8_t * ph = b + 52 + 32u * i;
        wr32(ph, PT_LOAD_);
        wr32(ph + 4, (uint32_t)off);
        wr32(ph + 8, seg[i].paddr);
        wr32(ph + 12, seg[i].paddr);
        wr32(ph + 16, seg[i].filesz);
        wr32(ph + 20, seg[i].filesz);
        wr32(ph + 24, 7);                            // R W X
        wr32(ph + 28, 4);
        memcpy(b + off, data[i], seg[i].filesz);
        off += seg[i].filesz;
    }
    *n_out = n;
    return b;
}

// ---------------------------------------------------------------------------
// Trace object
// ---------------------------------------------------------------------------
static const uint8_t k_depth_u200[TSR_NUM_UNITS] = {
    TSR_QDEPTH_U200_DMA, TSR_QDEPTH_U200_QACT, TSR_QDEPTH_U200_MVM,
    TSR_QDEPTH_U200_VU,  TSR_QDEPTH_U200_ATT,  TSR_QDEPTH_U200_SYNC,
};

void tsr_trace2_config_from_core(tsr_trace2_config * c, const tsr_config * core) {
    memset(c, 0, sizeof(*c));
    c->core = *core;
    c->itcm_bytes = TSR_ITCM_BYTES;
    c->dtcm_bytes = TSR_DTCM_BYTES;
    memcpy(c->q_depth, k_depth_u200, sizeof(c->q_depth));
}

uint32_t tsr_trace2_ctl_id_full(void) { return TSR_SIM_CTL_ID_FULL; }   // W1b fix

void tsr_trace2_config_from_sim(tsr_trace2_config * c, tsr_sim * sim) {
    memset(c, 0, sizeof(*c));
    c->core = *tsr_sim_config(sim);
    c->ctl_id = tsr_sim_corewin_read(sim, TSR_CSR_CTL_ID);
    c->cfg3 = tsr_sim_corewin_read(sim, TSR_CSR_CFG3);
    (void)tsr_sim_itcm(sim, &c->itcm_bytes);
    (void)tsr_sim_dtcm(sim, &c->dtcm_bytes);
    const uint32_t qd = tsr_sim_corewin_read(sim, TSR_CSR_Q_DEPTH);
    for (uint32_t u = 0; u < TSR_NUM_UNITS; ++u) c->q_depth[u] = (uint8_t)((qd >> (TSR_Q_DEPTH_FIELD_W * u)) & 0x1Fu);
}

void tsr_trace2_init(tsr_trace2 * t) {
    memset(t, 0, sizeof(*t));
    t->kind = TSR_TRACE2_KIND_OPS;
    const tsr_config u200 = TSR_CONFIG_U200_V1;
    tsr_trace2_config_from_core(&t->cfg, &u200);
    t->load_ch = TSR_TRACE2_LOAD_AUTO;
    t->done.err_ticket_rel = TSR_TICKET_NONE;
    t->n_units = TSR_NUM_UNITS;
    snprintf(t->sched, sizeof(t->sched), "fixed");
}

void tsr_trace2_clear(tsr_trace2 * t) {
    if (t == NULL) return;
    free(t->image);
    free(t->push);
    tsr_capture_free(&t->mem);
    tsr_trace2_init(t);
}

tsr_trace2 * tsr_trace2_new(void) {
    tsr_trace2 * t = (tsr_trace2 *)xmalloc(sizeof(tsr_trace2));
    tsr_trace2_init(t);
    return t;
}

void tsr_trace2_free(tsr_trace2 * t) {
    if (t == NULL) return;
    tsr_trace2_clear(t);
    free(t);
}

static void regions_copy(tsr_region ** dst, size_t * nd, const tsr_region * src, size_t n) {
    *dst = (tsr_region *)xcalloc(n ? n : 1, sizeof(tsr_region));
    for (size_t i = 0; i < n; ++i) {
        (*dst)[i] = src[i];
        (*dst)[i].data = (uint8_t *)xmalloc((size_t)src[i].size);
        memcpy((*dst)[i].data, src[i].data, (size_t)src[i].size);
    }
    *nd = n;
}

int tsr_trace2_copy(tsr_trace2 * dst, const tsr_trace2 * src) {
    tsr_trace2_clear(dst);
    *dst = *src;
    dst->image = NULL;
    dst->push = NULL;
    memset(&dst->mem, 0, sizeof(dst->mem));
    if (src->image_size) {
        dst->image = (uint8_t *)xmalloc(src->image_size);
        memcpy(dst->image, src->image, src->image_size);
    }
    dst->cap_push = src->n_push;
    dst->push = (tsr_trace_push *)xmalloc((src->n_push ? src->n_push : 1) * sizeof(tsr_trace_push));
    if (src->n_push) memcpy(dst->push, src->push, src->n_push * sizeof(tsr_trace_push));
    regions_copy(&dst->mem.init, &dst->mem.n_init, src->mem.init, src->mem.n_init);
    regions_copy(&dst->mem.expect, &dst->mem.n_expect, src->mem.expect, src->mem.n_expect);
    return 0;
}

int tsr_trace2_add_push(tsr_trace2 * t, const tsr_trace_push * p) {
    if (t->n_push == t->cap_push) {
        t->cap_push = t->cap_push ? 2 * t->cap_push : 64;
        t->push = (tsr_trace_push *)xrealloc(t->push, t->cap_push * sizeof(tsr_trace_push));
    }
    t->push[t->n_push++] = *p;
    return 0;
}

int tsr_trace2_add_op(tsr_trace2 * t, const uint32_t w[8], uint32_t dep0_rel, uint32_t dep1_rel, uint32_t push_val) {
    tsr_trace_push p;
    memset(&p, 0, sizeof(p));
    const uint32_t wmask = TSR_GET(push_val, TSR_ISSUE_PUSH_WMASK);
    for (uint32_t i = 0; i < 8; ++i) p.w[i] = (i == 0 || ((wmask >> i) & 1u)) ? w[i] : 0u;
    p.dep[0] = dep0_rel;
    p.dep[1] = dep1_rel;
    p.push = push_val;
    p.kind = TSR_TRACE_PUSH_OP;
    return tsr_trace2_add_push(t, &p);
}

int tsr_trace2_add_qctrl(tsr_trace2 * t, uint32_t bits) {
    tsr_trace_push p;
    memset(&p, 0, sizeof(p));
    p.w[0] = bits;
    p.kind = TSR_TRACE_PUSH_QCTRL;
    return tsr_trace2_add_push(t, &p);
}

int tsr_trace2_add_mark(tsr_trace2 * t, uint32_t value) {
    tsr_trace_push p;
    memset(&p, 0, sizeof(p));
    p.w[0] = value;
    p.kind = TSR_TRACE_PUSH_MARK;
    return tsr_trace2_add_push(t, &p);
}

static void region_add(tsr_region ** r, size_t * n, uint32_t ch, uint64_t off, const void * data, uint64_t size) {
    *r = (tsr_region *)xrealloc(*r, (*n + 1) * sizeof(tsr_region));
    tsr_region * x = &(*r)[(*n)++];
    x->ch = ch;
    x->byte_off = off;
    x->size = size;
    x->data = (uint8_t *)xmalloc((size_t)size);
    if (size) memcpy(x->data, data, (size_t)size);
}

int tsr_trace2_add_init(tsr_trace2 * t, uint32_t ch, uint64_t byte_off, const void * data, uint64_t size) {
    region_add(&t->mem.init, &t->mem.n_init, ch, byte_off, data, size);
    return 0;
}

int tsr_trace2_add_expect(tsr_trace2 * t, uint32_t ch, uint64_t byte_off, const void * data, uint64_t size) {
    region_add(&t->mem.expect, &t->mem.n_expect, ch, byte_off, data, size);
    return 0;
}

uint32_t tsr_trace2_push_unit(const tsr_trace2_config * c, const tsr_trace_push * p) {
    if (p->kind == TSR_TRACE_PUSH_QCTRL) return (p->w[0] & TSR_QCTRL_QCLEAR) ? TSR_UNIT_SYNC : TSR_UNIT_NONE;
    if (p->kind != TSR_TRACE_PUSH_OP) return TSR_UNIT_NONE;
    // The issue path's acceptance (tsr_sim_accept_op, CC §7.4) evaluated on c instead of a simulator.
    const uint32_t op = p->w[0] & 0xFFu;
    if (!tsr_op_legal(op)) return TSR_UNIT_SYNC;
    const uint32_t need = tsr_op_caps(op);
    if ((c->core.caps & need) != need) return TSR_UNIT_SYNC;
    const uint32_t u = tsr_op_unit(op);
    if (u >= TSR_NUM_UNITS || c->q_depth[u] == 0) return TSR_UNIT_SYNC;
    const tsr_op_handler * h = tsr_sim_op_handler((uint8_t)op);
    if (h == NULL || h->check == NULL || h->unit != u) return TSR_UNIT_SYNC;
    return u;
}

uint32_t tsr_trace2_ticket_of(const tsr_trace2 * t, size_t i) {
    if (i >= t->n_push) return TSR_TICKET_NONE;
    uint32_t cnt[TSR_NUM_UNITS] = { 0 };
    for (size_t k = 0; k < i; ++k) {
        const uint32_t u = tsr_trace2_push_unit(&t->cfg, &t->push[k]);
        if (u < TSR_NUM_UNITS) cnt[u]++;
    }
    const uint32_t u = tsr_trace2_push_unit(&t->cfg, &t->push[i]);
    return u < TSR_NUM_UNITS ? (u << 28) | (cnt[u] & SEQ_MASK) : TSR_TICKET_NONE;
}

// ---------------------------------------------------------------------------
// Spans of DRAM bytes and region checks
// ---------------------------------------------------------------------------
typedef struct { uint32_t ch; uint64_t lo, hi; } span;   // bytes [lo, hi)

static int span_cmp(const void * a, const void * b) {
    const span * x = (const span *)a, * y = (const span *)b;
    if (x->ch != y->ch) return x->ch < y->ch ? -1 : 1;
    return (x->lo > y->lo) - (x->lo < y->lo);
}

// Sorts and merges overlapping / adjacent spans; returns the new count.
static size_t span_normalize(span * s, size_t n) {
    if (n == 0) return 0;
    qsort(s, n, sizeof(span), span_cmp);
    size_t k = 0;
    for (size_t i = 1; i < n; ++i) {
        if (s[i].ch == s[k].ch && s[i].lo <= s[k].hi) {
            if (s[i].hi > s[k].hi) s[k].hi = s[i].hi;
        } else {
            s[++k] = s[i];
        }
    }
    return k + 1;
}

static span * regions_spans(const tsr_region * r, size_t n) {
    span * s = (span *)xmalloc((n ? n : 1) * sizeof(span));
    for (size_t i = 0; i < n; ++i) {
        s[i].ch = r[i].ch;
        s[i].lo = r[i].byte_off;
        s[i].hi = r[i].byte_off + r[i].size;
    }
    return s;
}

// Is [lo, hi) on ch inside one span of a normalized list?
static int span_covered(const span * s, size_t n, uint32_t ch, uint64_t lo, uint64_t hi) {
    size_t a = 0, b = n;
    while (a < b) {
        const size_t m = a + (b - a) / 2;
        if (s[m].ch < ch || (s[m].ch == ch && s[m].hi <= lo)) a = m + 1; else b = m;
    }
    return a < n && s[a].ch == ch && s[a].lo <= lo && hi <= s[a].hi;
}


static int check_regions(const tsr_config * cfg, const tsr_region * r, size_t n, const char * what, char * err, size_t en) {
    for (size_t i = 0; i < n; ++i) {
        if (r[i].ch >= cfg->num_ch || r[i].size == 0 || r[i].byte_off % 64 != 0 || r[i].size % 64 != 0 ||
            r[i].byte_off > cfg->ch_bytes || r[i].size > cfg->ch_bytes - r[i].byte_off) {
            seterr(err, en, "%s region %zu (ch %u, offset 0x%llx, size %llu) must be non-empty, 64-byte aligned and inside a "
                            "channel (num_ch %u, ch_bytes %llu)", what, i, r[i].ch, (unsigned long long)r[i].byte_off,
                   (unsigned long long)r[i].size, cfg->num_ch, (unsigned long long)cfg->ch_bytes);
            return -1;
        }
        if (r[i].data == NULL) {
            seterr(err, en, "%s region %zu has no data", what, i);
            return -1;
        }
    }
    span * s = regions_spans(r, n);
    if (n > 1) qsort(s, n, sizeof(span), span_cmp);
    for (size_t i = 1; i < n; ++i) {
        if (s[i].ch == s[i - 1].ch && s[i].lo < s[i - 1].hi) {
            seterr(err, en, "%s regions overlap (ch %u, offset 0x%llx)", what, s[i].ch, (unsigned long long)s[i].lo);
            free(s);
            return -1;
        }
    }
    free(s);
    return 0;
}

// The ops.load placement (kind ops, SW §5.1): ch 0, the lowest 4 KiB-aligned offset at or above
// the OPL region where push.bin overlaps no region.
static int choose_load(const tsr_trace2 * t, uint32_t * ch, uint64_t * off) {
    const uint64_t bytes = ((uint64_t)t->n_push * TSR_TRACE_PUSH_BYTES + 63u) & ~63ull;
    const size_t nr = t->mem.n_init + t->mem.n_expect;
    span * s = (span *)xmalloc((nr ? nr : 1) * sizeof(span));
    for (size_t i = 0; i < t->mem.n_init; ++i)
        s[i] = (span){ t->mem.init[i].ch, t->mem.init[i].byte_off, t->mem.init[i].byte_off + t->mem.init[i].size };
    for (size_t i = 0; i < t->mem.n_expect; ++i)
        s[t->mem.n_init + i] = (span){ t->mem.expect[i].ch, t->mem.expect[i].byte_off,
                                       t->mem.expect[i].byte_off + t->mem.expect[i].size };
    const size_t ns = span_normalize(s, nr);
    uint64_t o = TSR_REGION_OPL_OFF;
    for (;;) {
        if (o + bytes > t->cfg.core.ch_bytes) {
            free(s);
            return -1;
        }
        int moved = 0;
        for (size_t i = 0; i < ns; ++i) {
            if (s[i].ch == 0 && s[i].lo < o + bytes && o < s[i].hi && bytes > 0) {
                o = (s[i].hi + 4095u) & ~4095ull;
                moved = 1;
            }
        }
        if (!moved) break;
    }
    free(s);
    *ch = 0;
    *off = o;
    return 0;
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------
static int config_problem(const tsr_trace2_config * c, char * err, size_t en) {
    const tsr_config * k = &c->core;
    if (k->num_ch < 1 || k->num_ch > TSR_NUM_CH_MAX) { seterr(err, en, "config: num_ch %u not in [1, %u]", k->num_ch, TSR_NUM_CH_MAX); return -1; }
    if (k->beat_bits != 512) { seterr(err, en, "config: beat_bits %u (must be 512)", k->beat_bits); return -1; }
    if (k->ch_bytes < 64 || k->ch_bytes % 64 != 0) { seterr(err, en, "config: ch_bytes must be a positive multiple of 64"); return -1; }
    if (k->spm_bytes < 64 || k->spm_bytes % 64 != 0) { seterr(err, en, "config: spm_bytes must be a positive multiple of 64"); return -1; }
    if (k->addr_beat_w < 1 || k->addr_beat_w > 40) { seterr(err, en, "config: addr_beat_w %u not in [1, 40]", k->addr_beat_w); return -1; }
    for (uint32_t u = 0; u < TSR_NUM_UNITS; ++u)
        if (c->q_depth[u] > 31) { seterr(err, en, "config: q_depth[%u] = %u above 31", u, c->q_depth[u]); return -1; }
    if (c->q_depth[TSR_UNIT_SYNC] == 0) { seterr(err, en, "config: the SYNC unit must be built (q_depth[5] >= 1)"); return -1; }
    const uint32_t tcm[2] = { c->itcm_bytes, c->dtcm_bytes };
    for (int i = 0; i < 2; ++i)
        if (tcm[i] == 0 || tcm[i] % 1024u != 0 || tcm[i] > TCM_LIMIT) {
            seterr(err, en, "config: %s %u must be a positive multiple of 1 KiB up to 1 MiB", i ? "dtcm_bytes" : "itcm_bytes", tcm[i]);
            return -1;
        }
    return 0;
}

static int push_problem(const tsr_trace_push * p, size_t i, char * err, size_t en) {
    if (p->kind > TSR_TRACE_PUSH_MARK) { seterr(err, en, "push record %zu: kind %u (0 op, 1 QCTRL, 2 trace mark)", i, p->kind); return -1; }
    if (p->kind == TSR_TRACE_PUSH_OP) {
        const uint32_t wmask = TSR_GET(p->push, TSR_ISSUE_PUSH_WMASK);
        for (uint32_t k = 1; k < 8; ++k)
            if (!((wmask >> k) & 1u) && p->w[k] != 0) {
                seterr(err, en, "push record %zu: word %u is masked by the push value 0x%08x but not 0", i, k, p->push);
                return -1;
            }
        return 0;
    }
    int bad = p->dep[0] != 0 || p->dep[1] != 0 || p->push != 0;
    for (uint32_t k = 1; k < 8; ++k) bad |= p->w[k] != 0;
    if (bad) {
        seterr(err, en, "push record %zu: a %s record must have w[1..7], dep and push 0", i,
               p->kind == TSR_TRACE_PUSH_QCTRL ? "QCTRL" : "trace-mark");
        return -1;
    }
    return 0;
}

int tsr_trace2_validate(const tsr_trace2 * t, char * err, size_t en) {
    if (t->kind != TSR_TRACE2_KIND_LAUNCH && t->kind != TSR_TRACE2_KIND_OPS) { seterr(err, en, "kind %u", t->kind); return -1; }
    if (config_problem(&t->cfg, err, en)) return -1;
    if (t->kind == TSR_TRACE2_KIND_LAUNCH) {
        if (t->image == NULL || t->image_size == 0) { seterr(err, en, "kind launch without an image"); return -1; }
        img_info e;
        char why[256];
        if (img_parse(t->image, t->image_size, t->cfg.itcm_bytes, t->cfg.dtcm_bytes, &e, why, sizeof(why))) {
            seterr(err, en, "image: %s", why);
            return -1;
        }
        const uint64_t id = img_id(t->image, &e);
        if (id != t->image_id) {
            seterr(err, en, "image id %016llx differs from the image's EX §3.2 id %016llx", (unsigned long long)t->image_id,
                   (unsigned long long)id);
            return -1;
        }
        if (strnlen(t->kernel, sizeof(t->kernel)) >= sizeof(t->kernel)) { seterr(err, en, "kernel name not terminated"); return -1; }
        for (const char * p = t->kernel; *p; ++p)
            if ((unsigned char)*p < 0x20 || (unsigned char)*p >= 0x7F) { seterr(err, en, "kernel name: non-printable character"); return -1; }
        if (t->kernel_index > 0xFFFFu || t->prof > 1) { seterr(err, en, "launch: index %u / prof %u out of range", t->kernel_index, t->prof); return -1; }
    }
    if ((uint64_t)t->n_push >= PUSH_MAX) { seterr(err, en, "%zu push records", t->n_push); return -1; }
    for (size_t i = 0; i < t->n_push; ++i)
        if (push_problem(&t->push[i], i, err, en)) return -1;
    if (t->n_units < 1 || t->n_units > TSR_TRACE_UNITS_MAX) { seterr(err, en, "tickets: n_units %u not in [1, 8]", t->n_units); return -1; }
    for (uint32_t u = 0; u < t->n_units; ++u) {
        const tsr_trace_unit * x = &t->unit[u];
        const int absent = u >= TSR_NUM_UNITS || t->cfg.q_depth[u] == 0;
        if (x->reserved[0] || x->reserved[1]) { seterr(err, en, "tickets: unit %u: reserved words not 0", u); return -1; }
        if (x->err_code == 0 && (x->err_rel_seq || x->err_info || x->err_w0 || x->err_rel_pidx)) {
            seterr(err, en, "tickets: unit %u: error fields without an error code", u);
            return -1;
        }
        if (x->err_code > 0xFFu || x->err_rel_seq > SEQ_MASK) { seterr(err, en, "tickets: unit %u: err_code / err_rel_seq out of range", u); return -1; }
        if (absent && (x->count || x->err_code)) { seterr(err, en, "tickets: unit %u is not built but has entries", u); return -1; }
    }
    if (t->done.status > 0xFFu || t->done.err_code > 0xFFu) { seterr(err, en, "done: status / err_code above 255"); return -1; }
    if (check_regions(&t->cfg.core, t->mem.init, t->mem.n_init, "init", err, en) ||
        check_regions(&t->cfg.core, t->mem.expect, t->mem.n_expect, "expect", err, en))
        return -1;
    if (t->kind == TSR_TRACE2_KIND_OPS && t->load_ch != TSR_TRACE2_LOAD_AUTO) {
        const uint64_t bytes = (uint64_t)t->n_push * TSR_TRACE_PUSH_BYTES;
        if (t->load_ch >= t->cfg.core.num_ch || t->load_offset % 64 != 0 || t->load_offset > t->cfg.core.ch_bytes ||
            bytes > t->cfg.core.ch_bytes - t->load_offset) {
            seterr(err, en, "ops.load (ch %u, offset 0x%llx) must be 64-byte aligned with push.bin inside the channel", t->load_ch,
                   (unsigned long long)t->load_offset);
            return -1;
        }
        for (int pass = 0; pass < 2 && bytes > 0; ++pass) {
            const tsr_region * r = pass ? t->mem.expect : t->mem.init;
            const size_t n = pass ? t->mem.n_expect : t->mem.n_init;
            for (size_t i = 0; i < n; ++i)
                if (r[i].ch == t->load_ch && r[i].byte_off < t->load_offset + bytes && t->load_offset < r[i].byte_off + r[i].size) {
                    seterr(err, en, "ops.load (ch %u, offset 0x%llx) overlaps %s region %zu", t->load_ch,
                           (unsigned long long)t->load_offset, pass ? "expect" : "init", i);
                    return -1;
                }
        }
    }
    const size_t sl = strnlen(t->sched, sizeof(t->sched));
    if (sl == 0 || sl >= sizeof(t->sched)) { seterr(err, en, "sched must be a non-empty string below 64 characters"); return -1; }
    for (size_t i = 0; i < sl; ++i)
        if ((unsigned char)t->sched[i] < 0x20 || (unsigned char)t->sched[i] >= 0x7F || t->sched[i] == '"' || t->sched[i] == '\\') {
            seterr(err, en, "sched: invalid character");
            return -1;
        }
    if (!(t->rv_ratio == 0.0 || (t->rv_ratio >= 0.05 && t->rv_ratio <= 4.0))) { seterr(err, en, "rv_ratio %g not 0 or in [0.05, 4]", t->rv_ratio); return -1; }
    return 0;
}

// ---------------------------------------------------------------------------
// Write
// ---------------------------------------------------------------------------
static void write_region_list(FILE * f, const char * key, const char * sub, const tsr_region * r, size_t n) {
    fprintf(f, "  \"%s\": [", key);
    for (size_t i = 0; i < n; ++i) {
        char name[96];
        region_name(name, sizeof(name), sub, r[i].ch, r[i].byte_off);
        fprintf(f, "%s\n    {\"ch\": %u, \"offset\": %llu, \"size\": %llu, \"file\": \"%s\"}", i ? "," : "", r[i].ch,
                (unsigned long long)r[i].byte_off, (unsigned long long)r[i].size, name);
    }
    fprintf(f, "%s]", n ? "\n  " : "");
}

static int write_image(const tsr_trace2 * t, const char * dir, const char * images_dir, char ** rel, char * err, size_t en) {
    char * idir = images_dir ? xstrdup(images_dir) : path_join(dir, "../images");
    char name[32];
    snprintf(name, sizeof(name), "%016llx.elf", (unsigned long long)t->image_id);
    char * path = path_join(idir, name);
    int rc = -1;
    if (mkdir_p(idir, err, en) == 0) {
        size_t n = 0;
        char why[256];
        uint8_t * old = read_file(path, &n, IMG_MAX_BYTES, why, sizeof(why));
        if (old != NULL) {                           // written once per id (SW §5.5)
            if (n != t->image_size || memcmp(old, t->image, n) != 0)
                seterr(err, en, "%s exists with different bytes (two images with one id?)", path);
            else
                rc = 0;
            free(old);
        } else {
            rc = write_file(path, t->image, t->image_size, err, en);
        }
    }
    if (rc == 0) {
        *rel = relpath(dir, path);
        if (*rel == NULL) {
            seterr(err, en, "cannot express %s relative to %s", path, dir);
            rc = -1;
        }
    }
    free(path);
    free(idir);
    return rc;
}

static void json_str(FILE * f, const char * s) {
    fputc('"', f);
    for (; *s; ++s) {
        const unsigned char c = (unsigned char)*s;
        if (c == '"' || c == '\\') fprintf(f, "\\%c", c);
        else if (c < 0x20) fprintf(f, "\\u%04x", c);
        else fputc(c, f);
    }
    fputc('"', f);
}

int tsr_trace2_write(const tsr_trace2 * t, const char * dir, const char * images_dir) {
    char err[600];
    if (tsr_trace2_validate(t, err, sizeof(err))) {
        fprintf(stderr, "tsr_trace2_write %s: %s\n", dir, err);
        return -1;
    }
    uint32_t load_ch = t->load_ch;
    uint64_t load_off = t->load_offset;
    if (t->kind == TSR_TRACE2_KIND_OPS && load_ch == TSR_TRACE2_LOAD_AUTO && choose_load(t, &load_ch, &load_off)) {
        fprintf(stderr, "tsr_trace2_write %s: no room for push.bin (%zu records) on ch0 outside the regions\n", dir, t->n_push);
        return -1;
    }
    char * sub[3] = { path_join(dir, "init"), path_join(dir, "expect"), path_join(dir, "ops") };
    char * rel_img = NULL;
    uint8_t * buf = NULL;
    FILE * f = NULL;
    char * meta_tmp = NULL, * meta = NULL;
    int rc = -1;
    err[0] = '\0';
    if (mkdir_p(dir, err, sizeof(err)) || mkdir_p(sub[0], err, sizeof(err)) || mkdir_p(sub[1], err, sizeof(err)) ||
        mkdir_p(sub[2], err, sizeof(err)))
        goto done;
    remove_stale(dir);
    for (int pass = 0; pass < 2; ++pass) {
        const tsr_region * r = pass ? t->mem.expect : t->mem.init;
        const size_t n = pass ? t->mem.n_expect : t->mem.n_init;
        for (size_t i = 0; i < n; ++i) {
            char name[96];
            region_name(name, sizeof(name), pass ? "expect" : "init", r[i].ch, r[i].byte_off);
            char * p = path_join(dir, name);
            const int w = write_file(p, r[i].data, (size_t)r[i].size, err, sizeof(err));
            free(p);
            if (w) goto done;
        }
    }
    {   // ops/push.bin
        const size_t bytes = t->n_push * TSR_TRACE_PUSH_BYTES;
        buf = (uint8_t *)xmalloc(bytes);
        for (size_t i = 0; i < t->n_push; ++i) {
            uint8_t * q = buf + i * TSR_TRACE_PUSH_BYTES;
            const tsr_trace_push * p = &t->push[i];
            for (int k = 0; k < 8; ++k) wr32(q + 4 * k, p->w[k]);
            wr32(q + 32, p->dep[0]);
            wr32(q + 36, p->dep[1]);
            wr32(q + 40, p->push);
            wr32(q + 44, p->kind);
        }
        char * p = path_join(dir, "ops/push.bin");
        const int w = write_file(p, buf, bytes, err, sizeof(err));
        free(p);
        free(buf);
        buf = NULL;
        if (w) goto done;
    }
    {   // expect/tickets.bin
        const size_t bytes = TSR_TRACE_TICKETS_HDR_BYTES + (size_t)t->n_units * TSR_TRACE_UNIT_BYTES;
        buf = (uint8_t *)xcalloc(bytes, 1);
        wr32(buf, t->n_units);
        for (uint32_t u = 0; u < t->n_units; ++u) {
            uint8_t * q = buf + TSR_TRACE_TICKETS_HDR_BYTES + (size_t)u * TSR_TRACE_UNIT_BYTES;
            const tsr_trace_unit * x = &t->unit[u];
            wr32(q, x->count);
            wr32(q + 4, x->err_code);
            wr32(q + 8, x->err_rel_seq);
            wr32(q + 12, x->err_info);
            wr32(q + 16, x->err_w0);
            wr32(q + 20, x->err_rel_pidx);
        }
        char * p = path_join(dir, "expect/tickets.bin");
        const int w = write_file(p, buf, bytes, err, sizeof(err));
        free(p);
        free(buf);
        buf = NULL;
        if (w) goto done;
    }
    if (t->kind == TSR_TRACE2_KIND_LAUNCH && write_image(t, dir, images_dir, &rel_img, err, sizeof(err))) goto done;

    meta_tmp = path_join(dir, "meta.json.tmp");
    meta = path_join(dir, "meta.json");
    f = fopen(meta_tmp, "w");
    if (f == NULL) {
        snprintf(err, sizeof(err), "cannot create %s: %s", meta_tmp, strerror(errno));
        goto done;
    }
    {
        const tsr_trace2_config * c = &t->cfg;
        const tsr_config * k = &c->core;
        fprintf(f, "{\n  \"format\": \"%s\",\n  \"version\": %u,\n  \"kind\": \"%s\",\n", TSR_TRACE2_FORMAT, TSR_TRACE2_VERSION,
                t->kind == TSR_TRACE2_KIND_LAUNCH ? "launch" : "ops");
        fprintf(f, "  \"config\": {\"num_ch\": %u, \"beat_bits\": %u, \"lanes\": %u, \"t_max\": %u, \"k_max\": %u, "
                   "\"spm_bytes\": %u, \"vl\": %u, \"nsreg\": %u, \"rd_outstanding\": %u, \"ch_bytes\": %llu, "
                   "\"addr_beat_w\": %u, \"caps\": %u, \"ctl_id\": %u, \"cfg3\": %u, \"itcm_bytes\": %u, "
                   "\"dtcm_bytes\": %u, \"q_depth\": [%u, %u, %u, %u, %u, %u]},\n",
                k->num_ch, k->beat_bits, k->lanes, k->t_max, k->k_max, k->spm_bytes, k->vl, k->nsreg, k->rd_outstanding,
                (unsigned long long)k->ch_bytes, k->addr_beat_w, k->caps, c->ctl_id, c->cfg3, c->itcm_bytes, c->dtcm_bytes,
                c->q_depth[0], c->q_depth[1], c->q_depth[2], c->q_depth[3], c->q_depth[4], c->q_depth[5]);
        if (t->kind == TSR_TRACE2_KIND_LAUNCH) {
            fprintf(f, "  \"image\": {\"id\": \"%016llx\", \"file\": ", (unsigned long long)t->image_id);
            json_str(f, rel_img);
            fprintf(f, "},\n  \"launch\": {\"kernel\": ");
            json_str(f, t->kernel);
            fprintf(f, ", \"index\": %u, \"prof\": %u, \"args\": [%u, %u, %u, %u, %u, %u, %u]},\n", t->kernel_index, t->prof,
                    t->args[0], t->args[1], t->args[2], t->args[3], t->args[4], t->args[5], t->args[6]);
            fprintf(f, "  \"ops\": {\"file\": \"ops/push.bin\", \"count\": %zu},\n", t->n_push);
        } else {
            fprintf(f, "  \"ops\": {\"file\": \"ops/push.bin\", \"count\": %zu, \"load\": {\"ch\": %u, \"offset\": %llu}},\n",
                    t->n_push, load_ch, (unsigned long long)load_off);
        }
        const tsr_trace_done * d = &t->done;
        fprintf(f, "  \"done\": {\"status\": %u, \"ret\": %d, \"err_code\": %u, \"err_ticket_rel\": %u, \"err_pidx_rel\": %u, "
                   "\"err_info\": %u, \"err_w0\": %u},\n",
                d->status, d->ret, d->err_code, d->err_ticket_rel, d->err_pidx_rel, d->err_info, d->err_w0);
        write_region_list(f, "init", "init", t->mem.init, t->mem.n_init);
        fputs(",\n", f);
        write_region_list(f, "expect", "expect", t->mem.expect, t->mem.n_expect);
        fputs(",\n  \"sched\": ", f);
        json_str(f, t->sched);
        fprintf(f, ",\n  \"rv_ratio\": %.17g", t->rv_ratio);
        if (t->perf.valid) {
            const tsr_trace2_perf * p = &t->perf;
            fprintf(f, ",\n  \"perf\": {\"core_cycles\": %llu, \"rv_instret\": %llu, \"unit_busy\": [",
                    (unsigned long long)p->core_cycles, (unsigned long long)p->rv_instret);
            for (uint32_t u = 0; u < TSR_NUM_UNITS; ++u) fprintf(f, "%s%llu", u ? ", " : "", (unsigned long long)p->unit_busy[u]);
            fputs("], \"ch_rd_beats\": [", f);
            for (uint32_t ch = 0; ch < k->num_ch; ++ch) fprintf(f, "%s%llu", ch ? ", " : "", (unsigned long long)p->ch_rd_beats[ch]);
            fputs("]}", f);
        }
        fputs("\n}\n", f);
    }
    {
        const int bad = ferror(f);
        const int rc_close = fclose(f);
        f = NULL;
        if (bad || rc_close != 0) {
            snprintf(err, sizeof(err), "write error on %s", meta_tmp);
            goto done;
        }
    }
    if (rename(meta_tmp, meta) != 0) {
        snprintf(err, sizeof(err), "rename %s -> %s: %s", meta_tmp, meta, strerror(errno));
        goto done;
    }
    rc = 0;
done:
    if (f) fclose(f);
    if (rc) fprintf(stderr, "tsr_trace2_write %s: %s\n", dir, err);
    free(buf);
    free(meta_tmp);
    free(meta);
    free(rel_img);
    for (int i = 0; i < 3; ++i) free(sub[i]);
    return rc;
}

// ---------------------------------------------------------------------------
// Read
// ---------------------------------------------------------------------------
static int read_regions(rctx * c, const char * dir, const jval * root, const char * key, const tsr_config * cfg,
                        tsr_region ** out, size_t * n_out) {
    const jval * arr = NULL;
    if (jreq(c, root, key, J_ARR, "meta.json", &arr)) return -1;
    *out = (tsr_region *)xcalloc(arr->n ? arr->n : 1, sizeof(tsr_region));
    *n_out = 0;
    static const char * const keys[] = { "ch", "offset", "size", "file", NULL };
    for (size_t i = 0; i < arr->n; ++i) {
        const jval * e = &arr->items[i];
        char what[64];
        snprintf(what, sizeof(what), "%s[%zu]", key, i);
        const jval * fv = NULL;
        tsr_region * r = &(*out)[i];
        if (jkeys(c, e, what, keys) || jreq_u32(c, e, "ch", UINT32_MAX, &r->ch, what) ||
            jreq_u64(c, e, "offset", UINT64_MAX, &r->byte_off, what) || jreq_u64(c, e, "size", UINT64_MAX, &r->size, what) ||
            jreq(c, e, "file", J_STR, what, &fv))
            return -1;
        *n_out = i + 1;
        if (!inside_rel_path(fv->s)) return rfail(c, "%s: \"file\" must be a relative path inside the trace", what);
        if (r->ch >= cfg->num_ch || r->size == 0 || r->byte_off % 64 != 0 || r->size % 64 != 0 ||
            r->byte_off > cfg->ch_bytes || r->size > cfg->ch_bytes - r->byte_off)
            return rfail(c, "%s (ch %u, offset 0x%llx, size %llu) must be non-empty, 64-byte aligned and inside a channel "
                            "(num_ch %u, ch_bytes %llu)", what, r->ch, (unsigned long long)r->byte_off,
                         (unsigned long long)r->size, cfg->num_ch, (unsigned long long)cfg->ch_bytes);
        char why[256];
        char * p = path_join(dir, fv->s);
        size_t got = 0;
        r->data = read_file(p, &got, (size_t)r->size + 1, why, sizeof(why));
        if (r->data == NULL) {
            free(p);
            return rfail(c, "%s: %s", what, why);
        }
        if ((uint64_t)got != r->size) {
            free(p);
            return rfail(c, "%s: %s has %zu bytes, \"size\" says %llu", what, fv->s, got, (unsigned long long)r->size);
        }
        free(p);
    }
    char why[256];
    if (check_regions(cfg, *out, *n_out, key, why, sizeof(why))) return rfail(c, "%s", why);
    return 0;
}

static int read_meta(rctx * c, const char * dir, const jval * root, tsr_trace2 * t) {
    static const char * const top[] = { "format", "version", "kind", "config", "image", "launch", "ops", "done",
                                        "init", "expect", "sched", "rv_ratio", "perf", NULL };
    const jval * v = NULL;
    if (root->t != J_OBJ) return rfail(c, "not a JSON object");
    v = jget(root, "format");
    if (v == NULL || v->t != J_STR || strcmp(v->s, TSR_TRACE2_FORMAT) != 0)
        return rfail(c, "not a %s (missing or wrong \"format\")", TSR_TRACE2_FORMAT);
    uint32_t version = 0;
    if (jreq_u32(c, root, "version", UINT32_MAX, &version, "meta.json")) return -1;
    if (version == 1) return rfail(c, "a version 1 trace (cmd.bin); convert it with tsr-trace-conv (SW §5.6)");
    if (version != TSR_TRACE2_VERSION) return rfail(c, "unsupported version %u (expected %u)", version, TSR_TRACE2_VERSION);
    if (jkeys(c, root, "meta.json", top)) return -1;
    if (jreq(c, root, "kind", J_STR, "meta.json", &v)) return -1;
    if (strcmp(v->s, "launch") == 0) t->kind = TSR_TRACE2_KIND_LAUNCH;
    else if (strcmp(v->s, "ops") == 0) t->kind = TSR_TRACE2_KIND_OPS;
    else return rfail(c, "\"kind\" must be \"launch\" or \"ops\", not \"%s\"", v->s);
    const int launch = t->kind == TSR_TRACE2_KIND_LAUNCH;

    {   // config
        static const char * const keys[] = { "num_ch", "beat_bits", "lanes", "t_max", "k_max", "spm_bytes", "vl", "nsreg",
                                             "rd_outstanding", "ch_bytes", "addr_beat_w", "caps", "ctl_id", "cfg3",
                                             "itcm_bytes", "dtcm_bytes", "q_depth", NULL };
        const jval * o = NULL;
        if (jreq(c, root, "config", J_OBJ, "meta.json", &o) || jkeys(c, o, "config", keys)) return -1;
        tsr_trace2_config * g = &t->cfg;
        tsr_config * k = &g->core;
        memset(g, 0, sizeof(*g));
        if (jreq_u32(c, o, "num_ch", UINT32_MAX, &k->num_ch, "config") || jreq_u32(c, o, "beat_bits", UINT32_MAX, &k->beat_bits, "config") ||
            jreq_u32(c, o, "lanes", UINT32_MAX, &k->lanes, "config") || jreq_u32(c, o, "t_max", UINT32_MAX, &k->t_max, "config") ||
            jreq_u32(c, o, "k_max", UINT32_MAX, &k->k_max, "config") || jreq_u32(c, o, "spm_bytes", UINT32_MAX, &k->spm_bytes, "config") ||
            jreq_u32(c, o, "vl", UINT32_MAX, &k->vl, "config") || jreq_u32(c, o, "nsreg", UINT32_MAX, &k->nsreg, "config") ||
            jreq_u32(c, o, "rd_outstanding", UINT32_MAX, &k->rd_outstanding, "config") ||
            jreq_u64(c, o, "ch_bytes", UINT64_MAX, &k->ch_bytes, "config") ||
            jreq_u32(c, o, "addr_beat_w", UINT32_MAX, &k->addr_beat_w, "config") || jreq_u32(c, o, "caps", UINT32_MAX, &k->caps, "config") ||
            jreq_u32(c, o, "ctl_id", UINT32_MAX, &g->ctl_id, "config") || jreq_u32(c, o, "cfg3", UINT32_MAX, &g->cfg3, "config") ||
            jreq_u32(c, o, "itcm_bytes", UINT32_MAX, &g->itcm_bytes, "config") ||
            jreq_u32(c, o, "dtcm_bytes", UINT32_MAX, &g->dtcm_bytes, "config"))
            return -1;
        const jval * qd = NULL;
        if (jreq(c, o, "q_depth", J_ARR, "config", &qd)) return -1;
        if (qd->n != TSR_NUM_UNITS) return rfail(c, "config: \"q_depth\" must have %u entries", TSR_NUM_UNITS);
        for (uint32_t u = 0; u < TSR_NUM_UNITS; ++u) {
            uint64_t d = 0;
            if (jnum_u64(c, &qd->items[u], 31, &d, "config", "q_depth")) return -1;
            g->q_depth[u] = (uint8_t)d;
        }
        char why[256];
        if (config_problem(g, why, sizeof(why))) return rfail(c, "%s", why);
    }
    if (launch) {
        static const char * const ikeys[] = { "id", "file", NULL };
        static const char * const lkeys[] = { "kernel", "index", "prof", "args", NULL };
        const jval * o = NULL, * idv = NULL, * fv = NULL, * kv = NULL, * av = NULL;
        if (jreq(c, root, "image", J_OBJ, "meta.json", &o) || jkeys(c, o, "image", ikeys) ||
            jreq(c, o, "id", J_STR, "image", &idv) || jreq(c, o, "file", J_STR, "image", &fv))
            return -1;
        if (strlen(idv->s) != 16 || strspn(idv->s, "0123456789abcdef") != 16)
            return rfail(c, "image: \"id\" must be 16 lower-case hex digits");
        t->image_id = strtoull(idv->s, NULL, 16);
        if (fv->s[0] == '\0' || fv->s[0] == '/') return rfail(c, "image: \"file\" must be a relative path");
        char * p = path_join(dir, fv->s);
        char why[256];
        t->image = read_file(p, &t->image_size, IMG_MAX_BYTES, why, sizeof(why));
        free(p);
        if (t->image == NULL) return rfail(c, "image: %s", why);
        img_info e;
        if (img_parse(t->image, t->image_size, t->cfg.itcm_bytes, t->cfg.dtcm_bytes, &e, why, sizeof(why)))
            return rfail(c, "image %s: %s", fv->s, why);
        const uint64_t id = img_id(t->image, &e);
        if (id != t->image_id)
            return rfail(c, "image %s: EX §3.2 id %016llx differs from \"id\" %016llx", fv->s, (unsigned long long)id,
                         (unsigned long long)t->image_id);
        if (jreq(c, root, "launch", J_OBJ, "meta.json", &o) || jkeys(c, o, "launch", lkeys) ||
            jreq(c, o, "kernel", J_STR, "launch", &kv) || jreq_u32(c, o, "index", 0xFFFFu, &t->kernel_index, "launch") ||
            jreq_u32(c, o, "prof", 1, &t->prof, "launch") || jreq(c, o, "args", J_ARR, "launch", &av))
            return -1;
        if (strlen(kv->s) >= sizeof(t->kernel)) return rfail(c, "launch: \"kernel\" longer than %zu characters", sizeof(t->kernel) - 1);
        snprintf(t->kernel, sizeof(t->kernel), "%s", kv->s);
        if (av->n != 7) return rfail(c, "launch: \"args\" must have 7 entries (ARG0..ARG6)");
        for (int i = 0; i < 7; ++i) {
            uint64_t a = 0;
            if (jnum_u64(c, &av->items[i], UINT32_MAX, &a, "launch", "args")) return -1;
            t->args[i] = (uint32_t)a;
        }
    } else if (jget(root, "image") != NULL || jget(root, "launch") != NULL) {
        return rfail(c, "kind ops has no \"image\" / \"launch\"");
    }
    uint64_t count = 0;
    {   // ops + push.bin
        static const char * const okeys[] = { "file", "count", "load", NULL };
        static const char * const ldkeys[] = { "ch", "offset", NULL };
        const jval * o = NULL, * fv = NULL, * ld = NULL;
        if (jreq(c, root, "ops", J_OBJ, "meta.json", &o) || jkeys(c, o, "ops", okeys) || jreq(c, o, "file", J_STR, "ops", &fv) ||
            jreq_u64(c, o, "count", PUSH_MAX - 1, &count, "ops"))
            return -1;
        if (!inside_rel_path(fv->s)) return rfail(c, "ops: \"file\" must be a relative path inside the trace");
        ld = jget(o, "load");
        if (launch && ld != NULL) return rfail(c, "ops: \"load\" is for kind ops only");
        if (!launch) {
            if (ld == NULL) return rfail(c, "ops: missing \"load\" (kind ops)");
            if (jkeys(c, ld, "ops.load", ldkeys) || jreq_u32(c, ld, "ch", UINT32_MAX, &t->load_ch, "ops.load") ||
                jreq_u64(c, ld, "offset", UINT64_MAX, &t->load_offset, "ops.load"))
                return -1;
            if (t->load_ch == TSR_TRACE2_LOAD_AUTO) return rfail(c, "ops.load: ch %u", t->load_ch);
        }
        char * p = path_join(dir, fv->s);
        char why[256];
        size_t got = 0;
        uint8_t * b = read_file(p, &got, (size_t)(count * TSR_TRACE_PUSH_BYTES) + 1, why, sizeof(why));
        free(p);
        if (b == NULL) return rfail(c, "ops: %s", why);
        if ((uint64_t)got != count * TSR_TRACE_PUSH_BYTES) {
            free(b);
            return rfail(c, "%s: %zu bytes, \"count\" %llu needs %llu (48 bytes per record)", fv->s, got,
                         (unsigned long long)count, (unsigned long long)(count * TSR_TRACE_PUSH_BYTES));
        }
        t->cap_push = (size_t)count;
        t->push = (tsr_trace_push *)xmalloc((count ? (size_t)count : 1) * sizeof(tsr_trace_push));
        t->n_push = (size_t)count;
        for (size_t i = 0; i < t->n_push; ++i) {
            const uint8_t * q = b + i * TSR_TRACE_PUSH_BYTES;
            tsr_trace_push * x = &t->push[i];
            for (int k = 0; k < 8; ++k) x->w[k] = rd32(q + 4 * k);
            x->dep[0] = rd32(q + 32);
            x->dep[1] = rd32(q + 36);
            x->push = rd32(q + 40);
            x->kind = rd32(q + 44);
            if (push_problem(x, i, why, sizeof(why))) {
                free(b);
                return rfail(c, "%s: %s", fv->s, why);
            }
        }
        free(b);
    }
    {   // done
        static const char * const keys[] = { "status", "ret", "err_code", "err_ticket_rel", "err_pidx_rel", "err_info", "err_w0", NULL };
        const jval * o = NULL, * rv = NULL;
        tsr_trace_done * d = &t->done;
        if (jreq(c, root, "done", J_OBJ, "meta.json", &o) || jkeys(c, o, "done", keys) ||
            jreq_u32(c, o, "status", 0xFFu, &d->status, "done") || jreq_u32(c, o, "err_code", 0xFFu, &d->err_code, "done") ||
            jreq_u32(c, o, "err_ticket_rel", UINT32_MAX, &d->err_ticket_rel, "done") ||
            jreq_u32(c, o, "err_pidx_rel", UINT32_MAX, &d->err_pidx_rel, "done") ||
            jreq_u32(c, o, "err_info", UINT32_MAX, &d->err_info, "done") || jreq_u32(c, o, "err_w0", UINT32_MAX, &d->err_w0, "done") ||
            jreq(c, o, "ret", J_NUM, "done", &rv))
            return -1;
        if (!rv->is_int || rv->i < INT32_MIN || rv->i > (int64_t)UINT32_MAX)
            return rfail(c, "done: \"ret\" must be a 32-bit integer");
        d->ret = (int32_t)(uint32_t)(uint64_t)rv->i;
    }
    if (read_regions(c, dir, root, "init", &t->cfg.core, &t->mem.init, &t->mem.n_init) ||
        read_regions(c, dir, root, "expect", &t->cfg.core, &t->mem.expect, &t->mem.n_expect))
        return -1;
    {
        const jval * s = NULL;
        if (jreq(c, root, "sched", J_STR, "meta.json", &s)) return -1;
        if (strlen(s->s) == 0 || strlen(s->s) >= sizeof(t->sched)) return rfail(c, "\"sched\" must be 1..63 characters");
        snprintf(t->sched, sizeof(t->sched), "%s", s->s);
        if (jreq(c, root, "rv_ratio", J_NUM, "meta.json", &s)) return -1;
        t->rv_ratio = s->d;
    }
    {   // perf (optional, informational)
        static const char * const keys[] = { "core_cycles", "rv_instret", "unit_busy", "ch_rd_beats", NULL };
        const jval * o = jget(root, "perf"), * ub = NULL, * rb = NULL;
        if (o != NULL) {
            tsr_trace2_perf * p = &t->perf;
            if (jkeys(c, o, "perf", keys) || jreq_u64(c, o, "core_cycles", UINT64_MAX, &p->core_cycles, "perf") ||
                jreq_u64(c, o, "rv_instret", UINT64_MAX, &p->rv_instret, "perf") || jreq(c, o, "unit_busy", J_ARR, "perf", &ub) ||
                jreq(c, o, "ch_rd_beats", J_ARR, "perf", &rb))
                return -1;
            if (ub->n != TSR_NUM_UNITS || rb->n != t->cfg.core.num_ch)
                return rfail(c, "perf: \"unit_busy\" needs %u entries and \"ch_rd_beats\" num_ch", TSR_NUM_UNITS);
            for (size_t i = 0; i < ub->n; ++i)
                if (jnum_u64(c, &ub->items[i], UINT64_MAX, &p->unit_busy[i], "perf", "unit_busy")) return -1;
            for (size_t i = 0; i < rb->n; ++i)
                if (jnum_u64(c, &rb->items[i], UINT64_MAX, &p->ch_rd_beats[i], "perf", "ch_rd_beats")) return -1;
            p->valid = 1;
        }
    }
    {   // expect/tickets.bin
        char * p = path_join(dir, "expect/tickets.bin");
        char why[256];
        size_t got = 0;
        uint8_t * b = read_file(p, &got, TSR_TRACE_TICKETS_HDR_BYTES + TSR_TRACE_UNITS_MAX * TSR_TRACE_UNIT_BYTES, why, sizeof(why));
        free(p);
        if (b == NULL) return rfail(c, "tickets: %s", why);
        const uint32_t nu = got >= TSR_TRACE_TICKETS_HDR_BYTES ? rd32(b) : 0;
        if (got < TSR_TRACE_TICKETS_HDR_BYTES || nu < 1 || nu > TSR_TRACE_UNITS_MAX ||
            got != TSR_TRACE_TICKETS_HDR_BYTES + (size_t)nu * TSR_TRACE_UNIT_BYTES || rd32(b + 4) || rd32(b + 8) || rd32(b + 12)) {
            free(b);
            return rfail(c, "expect/tickets.bin: %zu bytes; needs a 16-byte header {n_units 1..8, 0, 0, 0} and n_units x 32 bytes", got);
        }
        t->n_units = nu;
        for (uint32_t u = 0; u < nu; ++u) {
            const uint8_t * q = b + TSR_TRACE_TICKETS_HDR_BYTES + (size_t)u * TSR_TRACE_UNIT_BYTES;
            tsr_trace_unit * x = &t->unit[u];
            x->count = rd32(q);
            x->err_code = rd32(q + 4);
            x->err_rel_seq = rd32(q + 8);
            x->err_info = rd32(q + 12);
            x->err_w0 = rd32(q + 16);
            x->err_rel_pidx = rd32(q + 20);
            x->reserved[0] = rd32(q + 24);
            x->reserved[1] = rd32(q + 28);
        }
        free(b);
    }
    char why[600];
    if (tsr_trace2_validate(t, why, sizeof(why))) return rfail(c, "%s", why);
    return 0;
}

int tsr_trace2_read(const char * dir, tsr_trace2 ** out, char * err, size_t errlen) {
    *out = NULL;
    char * mpath = path_join(dir, "meta.json");
    rctx c = { mpath, err, errlen };
    size_t mlen = 0;
    uint8_t * text = read_file(mpath, &mlen, 64u << 20, err, errlen);
    if (text == NULL) {
        free(mpath);
        return -1;
    }
    jval root;
    if (jparse((const char *)text, mlen, &root, mpath, err, errlen)) {
        free(text);
        free(mpath);
        return -1;
    }
    free(text);
    tsr_trace2 * t = tsr_trace2_new();
    const int rc = read_meta(&c, dir, &root, t);
    jfree(&root);
    free(mpath);
    if (rc) {
        tsr_trace2_free(t);
        return -1;
    }
    *out = t;
    return 0;
}

// ---------------------------------------------------------------------------
// The TRACE model (SW §5.5): launch capture on the simulator's queue and access events
//
// State machine, re-evaluated at every relevant event (sync): LQ_SEQ steps -> the staged
// descriptor is remembered under its sequence number; TAKEN steps after an LQ_POP store -> the
// launch TAKEN - 1 starts (DRAM capture on, counters noted); ANSWERED steps after an LCQ_POST
// with ANSWER -> the launch ends with the staged record (TAKEN / ANSWERED moved by a reset end
// it without a trace); RV_BOOT generation steps -> the image is taken from the loader's words.
// Comparing counters instead of trusting one event per transition keeps it exact whether an
// access event comes before or after its effect.
// ---------------------------------------------------------------------------
#define LQ_SHADOW 64u

typedef struct cap_cur {
    int              active;
    int              run;                       // a replay's tile-only run (no LQ / LCQ)
    uint32_t         seq;
    uint32_t         desc[8];
    uint32_t         base[TSR_NUM_UNITS];       // ISSUED_SEQ at kernel entry
    uint32_t         pidx_base;                 // ISSUE_PIDX at kernel entry
    int              entry_seen;
    int              pidx_known;                // pidx_base set (entry, or the first push's index)
    uint32_t         d0[TSR_NUM_UNITS];         // DONE_SEQ at LQ_POP
    uint64_t         t0, instret0;
    uint64_t         busy0[TSR_NUM_UNITS], rd0[TSR_NUM_CH_MAX];
    tsr_trace_push * push;                      // deps absolute until the end
    size_t           n_push, cap_push;
    uint32_t         host_writes;
} cap_cur;

typedef struct tmodel {
    tsr_sim *   sim;
    int         lis_id, acc_id;
    int         on;                             // tsr_sim_capture_launch
    char *      dir;                            // TSR_TRACE
    int         dir_index;                      // this simulator's number among TSR_TRACE writers, -1 before
    int         busy;                           // inside our own handler (our reads raise events)
    struct { uint32_t seq; int valid; uint32_t w[8]; } lq[LQ_SHADOW];
    uint32_t    seen_pushed, seen_taken, seen_answered;
    int         pop_seen, post_seen;
    uint32_t    post_w[8];
    uint32_t    load_cur;                       // the loader's LOAD_ADDR
    uint8_t *   tcm[2];                         // words the loader wrote (ITCM, DTCM)
    uint8_t *   mask[2];                        // one byte per word: written
    uint32_t    tcm_bytes[2];
    uint64_t    boot_gen;
    uint8_t *   img;                            // image of the current boot (synthesised)
    size_t      img_n;
    uint8_t *   user_img;                       // tsr_sim_capture_set_image
    size_t      user_n;
    cap_cur     cur;
    tsr_trace2 ** q;                            // captured, waiting for tsr_sim_capture_take
    size_t      qn, qcap;
    tsr_trace_capture_stats st;
} tmodel;

static tmodel * model_of(tsr_sim * sim) {
    void ** s = tsr_sim_model_slot(sim, TSR_MODEL_TRACE);
    return s ? (tmodel *)*s : NULL;
}

// Register reads of the model go through the core window's side-effect-free CSR view: a
// control-window read would itself be reported to the access listeners (the race detector takes
// a TSR_SIM_PC_HOOK control-window access for a test-harness access).
static inline uint32_t csr(tsr_sim * sim, uint32_t off) { return tsr_sim_corewin_read(sim, off); }

static const uint32_t * busy_slots(uint32_t C, uint32_t out[TSR_NUM_UNITS]) {
    out[TSR_UNIT_DMA] = TSR_PERF_DMA_CYCLES(C);
    out[TSR_UNIT_QACT] = TSR_PERF_QACT_CYCLES(C);
    out[TSR_UNIT_MVM] = TSR_PERF_MVM_CYCLES;
    out[TSR_UNIT_VU] = TSR_PERF_VU_CYCLES;
    out[TSR_UNIT_ATT] = TSR_PERF_ATT_CYCLES(C);
    out[TSR_UNIT_SYNC] = UINT32_MAX;            // SYNC has no busy counter
    return out;
}

static void cur_reset(cap_cur * c) {
    free(c->push);
    memset(c, 0, sizeof(*c));
}

// Counters at the start of a launch / run (the base is refined at kernel entry).
static void cur_start(tmodel * m, int run) {
    tsr_sim * sim = m->sim;
    cap_cur * c = &m->cur;
    cur_reset(c);
    c->active = 1;
    c->run = run;
    for (uint32_t u = 0; u < TSR_NUM_UNITS; ++u) {
        c->base[u] = csr(sim, TSR_CSR_Q_ISSUED(u)) & SEQ_MASK;
        c->d0[u] = csr(sim, TSR_CSR_Q_DONE(u)) & SEQ_MASK;
    }
    // ISSUE_PIDX now = the index the next push gets: taken from the first push event (only the
    // RISC-V moves it) unless kernel entry comes first.
    c->pidx_base = 0;
    c->pidx_known = 0;
    c->t0 = tsr_sim_now(sim);
    c->instret0 = tsr_sim_rv_instret(sim);
    uint32_t bs[TSR_NUM_UNITS];
    busy_slots(tsr_sim_num_ch(sim), bs);
    for (uint32_t u = 0; u < TSR_NUM_UNITS; ++u) c->busy0[u] = bs[u] == UINT32_MAX ? 0 : tsr_sim_perf(sim, bs[u]);
    for (uint32_t ch = 0; ch < tsr_sim_num_ch(sim); ++ch) c->rd0[ch] = tsr_sim_perf(sim, TSR_PERF_CH_RD_BEATS(ch));
    tsr_sim_capture_begin(sim);
}

// W2a: the launch's own LAUNCH_BEGIN / LAUNCH_END marker with its SEQ field made launch-relative (0);
// every other marker value unchanged (decisions above).
static uint32_t launch_mark_rel(const cap_cur * c, uint32_t v) {
    const uint32_t cls = TSR_GET(v, TSR_MARK_CLASS);
    if ((cls == TSR_MARK_CLASS_LAUNCH_BEGIN || cls == TSR_MARK_CLASS_LAUNCH_END) &&
        TSR_GET(v, TSR_MARK_LAUNCH_KERNEL) == (TSR_GET(c->desc[0], TSR_DESC_KERNEL) & (TSR_MARK_LAUNCH_KERNEL >> TSR_MARK_LAUNCH_KERNEL_LSB)) &&
        TSR_GET(v, TSR_MARK_LAUNCH_SEQ) == (c->seq & TSR_MARK_LAUNCH_SEQ))
        return v & ~TSR_MARK_LAUNCH_SEQ;
    return v;
}

static void cur_push(cap_cur * c, const tsr_trace_push * p) {
    if (c->n_push == c->cap_push) {
        c->cap_push = c->cap_push ? 2 * c->cap_push : 64;
        c->push = (tsr_trace_push *)xrealloc(c->push, c->cap_push * sizeof(tsr_trace_push));
    }
    c->push[c->n_push++] = *p;
}

// Everything but the DONE record and the kind-specific fields, at the end of a launch / run.
static void cur_finish(tmodel * m, tsr_trace2 * t) {
    tsr_sim * sim = m->sim;
    cap_cur * c = &m->cur;
    tsr_capture_free(&t->mem);
    tsr_sim_capture_end(sim, &t->mem);
    tsr_trace2_config_from_sim(&t->cfg, sim);
    t->n_units = TSR_NUM_UNITS;
    memset(t->unit, 0, sizeof(t->unit));
    for (uint32_t u = 0; u < TSR_NUM_UNITS; ++u) {
        if (t->cfg.q_depth[u] == 0) continue;
        tsr_trace_unit * x = &t->unit[u];
        x->count = (csr(sim, TSR_CSR_Q_DONE(u)) - c->d0[u]) & SEQ_MASK;
        x->err_code = csr(sim, TSR_CSR_Q_ERR_CODE(u)) & 0xFFu;
        if (x->err_code != 0) {
            x->err_rel_seq = (csr(sim, TSR_CSR_Q_ERR_SEQ(u)) - c->base[u]) & SEQ_MASK;
            x->err_info = csr(sim, TSR_CSR_Q_ERR_INFO(u));
            x->err_w0 = csr(sim, TSR_CSR_Q_ERR_W0(u));
            x->err_rel_pidx = csr(sim, TSR_CSR_Q_ERR_PIDX(u)) - c->pidx_base;
        }
    }
    free(t->push);
    t->push = c->push;
    t->n_push = c->n_push;
    t->cap_push = c->cap_push;
    c->push = NULL;
    c->n_push = c->cap_push = 0;
    for (size_t i = 0; i < t->n_push; ++i) {
        tsr_trace_push * p = &t->push[i];
        if (p->kind != TSR_TRACE_PUSH_OP) continue;
        p->dep[0] = tsr_trace_rel_ticket(p->dep[0], c->base);
        p->dep[1] = tsr_trace_rel_ticket(p->dep[1], c->base);
    }
    tsr_trace2_perf * pf = &t->perf;
    memset(pf, 0, sizeof(*pf));
    pf->valid = 1;
    pf->core_cycles = tsr_sim_now(sim) - c->t0;
    pf->rv_instret = tsr_sim_rv_instret(sim) - c->instret0;
    uint32_t bs[TSR_NUM_UNITS];
    busy_slots(tsr_sim_num_ch(sim), bs);
    for (uint32_t u = 0; u < TSR_NUM_UNITS; ++u)
        if (bs[u] != UINT32_MAX) pf->unit_busy[u] = tsr_sim_perf(sim, bs[u]) - c->busy0[u];
    for (uint32_t ch = 0; ch < tsr_sim_num_ch(sim); ++ch) pf->ch_rd_beats[ch] = tsr_sim_perf(sim, TSR_PERF_CH_RD_BEATS(ch)) - c->rd0[ch];
    for (uint32_t u = 0; u < TSR_NUM_UNITS; ++u) pf->valid &= pf->unit_busy[u] < (1ull << 62);   // a reset cleared PERF
    const char * s = tsr_sim_opt_sched(sim);
    snprintf(t->sched, sizeof(t->sched), "%s", (s && *s) ? s : "fixed");
    t->rv_ratio = tsr_sim_rv_ratio(sim);
    if (c->host_writes)
        fprintf(stderr, "tsr_trace: warning: the host wrote DRAM %u time(s) during a captured %s (EX §2.3 H3); the trace "
                        "holds what the ops saw\n", c->host_writes, c->run ? "run" : "launch");
    m->st.host_dram_writes += c->host_writes;
}

static void cur_abort(tmodel * m) {
    tsr_capture tmp;
    tsr_sim_capture_end(m->sim, &tmp);
    tsr_capture_free(&tmp);
    cur_reset(&m->cur);
}

// The replay kernel's answer for a run (SW §7.7 step 4, EX §4.4 precedence): the first error is
// the smallest relative ERR_PIDX over the units in error state.
static void done_of_units(tsr_trace2 * t) {
    tsr_trace_done * d = &t->done;
    memset(d, 0, sizeof(*d));
    d->err_ticket_rel = TSR_TICKET_NONE;
    int found = 0;
    uint32_t best = 0, bu = 0;
    for (uint32_t u = 0; u < TSR_NUM_UNITS && u < t->n_units; ++u) {
        if (t->unit[u].err_code == 0) continue;
        if (!found || t->unit[u].err_rel_pidx < best) {
            found = 1;
            best = t->unit[u].err_rel_pidx;
            bu = u;
        }
    }
    if (!found) return;
    const tsr_trace_unit * x = &t->unit[bu];
    d->status = x->err_code == TSR_ERR_ABORTED ? TSR_LCQ_R0_STATUS_ABORTED : TSR_LCQ_R0_STATUS_TILE_ERR;
    d->err_code = x->err_code;
    d->ret = (int32_t)(1u + x->err_rel_pidx);
    d->err_ticket_rel = (bu << 28) | x->err_rel_seq;
    d->err_pidx_rel = x->err_rel_pidx;
    d->err_info = x->err_info;
    d->err_w0 = x->err_w0;
}

// The image of the current boot from the loader's words: one PT_LOAD per maximal run.
static void image_snapshot(tmodel * m) {
    free(m->img);
    m->img = NULL;
    m->img_n = 0;
    img_seg seg[IMG_SEG_MAX];
    const uint8_t * data[IMG_SEG_MAX];
    uint32_t n = 0;
    static const uint32_t base[2] = { TSR_RV_ITCM_BASE, TSR_RV_DTCM_BASE };
    for (int k = 0; k < 2; ++k) {
        const uint32_t words = m->tcm_bytes[k] / 4u;
        uint32_t i = 0;
        while (i < words) {
            if (!m->mask[k][i]) { ++i; continue; }
            uint32_t j = i;
            while (j < words && m->mask[k][j]) ++j;
            if (n == IMG_SEG_MAX) {
                fprintf(stderr, "tsr_trace: warning: the loaded image has more than %u runs of words; no image for captures "
                                "(use tsr_sim_capture_set_image)\n", IMG_SEG_MAX);
                return;
            }
            seg[n].paddr = base[k] + 4u * i;
            seg[n].filesz = seg[n].memsz = 4u * (j - i);
            seg[n].off = 0;
            data[n] = m->tcm[k] + 4u * i;
            ++n;
            i = j;
        }
    }
    if (n == 0) return;
    const uint32_t entry = (csr(m->sim, TSR_CSR_RV_BOOT) & TSR_RV_BOOT_ADDR) + TSR_BOOT_ENTRY_OFF;
    m->img = img_synth(seg, data, n, entry, &m->img_n);
    // An ELF given with tsr_sim_capture_set_image must be what the loader wrote: every word of its
    // PT_LOAD file data loaded with that value. Otherwise it is dropped (a later load replaced it).
    if (m->user_img != NULL) {
        img_info e;
        int same = img_parse(m->user_img, m->user_n, m->tcm_bytes[0], m->tcm_bytes[1], &e, NULL, 0) == 0;
        for (uint32_t i = 0; same && i < e.nseg; ++i) {
            const img_seg * g = &e.seg[i];
            const int k = g->paddr >= TSR_RV_DTCM_BASE;
            const uint32_t off = g->paddr - base[k];
            for (uint32_t b = 0; same && b < g->filesz; ++b) {
                const uint32_t w = (off + b) / 4u;
                same = m->mask[k][w] && m->tcm[k][off + b] == m->user_img[g->off + b];
            }
        }
        if (!same) {
            fprintf(stderr, "tsr_trace: warning: the image given to tsr_sim_capture_set_image is not what the loader wrote "
                            "before this boot; captures use the loaded words instead\n");
            free(m->user_img);
            m->user_img = NULL;
            m->user_n = 0;
        }
    }
}

static void loader_word(tmodel * m, uint32_t val) {
    tsr_sim * sim = m->sim;
    if (tsr_sim_rv_run(sim) || (csr(sim, TSR_CSR_STATUS) & TSR_STATUS_SR_BUSY)) return;   // refused (EX §3.3)
    const uint32_t a = m->load_cur;
    m->load_cur += 4;
    int k = -1;
    uint32_t off = 0;
    if (a - TSR_RV_ITCM_BASE < m->tcm_bytes[0]) { k = 0; off = a - TSR_RV_ITCM_BASE; }
    else if (a >= TSR_RV_DTCM_BASE && a - TSR_RV_DTCM_BASE < m->tcm_bytes[1]) { k = 1; off = a - TSR_RV_DTCM_BASE; }
    if (k < 0) return;                                // no TCM there: the word is lost
    wr32(m->tcm[k] + off, val);
    m->mask[k][off / 4u] = 1;
}

// TSR_TRACE simulators of this process, numbered in the order of their first written trace: the
// first writes <dir>/<seq>, the k-th (k >= 1) <dir>/sim<k>/<seq>, so that two simulators of one
// process (tests running several) never overwrite each other's launches; images stay shared.
static atomic_uint g_trace_sims;

static void deliver(tmodel * m, tsr_trace2 * t, uint32_t seq) {
    m->st.captured++;
    if (m->dir != NULL) {
        if (m->dir_index < 0) m->dir_index = (int)atomic_fetch_add(&g_trace_sims, 1u);
        char sub[48];
        if (m->dir_index == 0) snprintf(sub, sizeof(sub), "%06u", seq);
        else snprintf(sub, sizeof(sub), "sim%d/%06u", m->dir_index, seq);
        char * d = path_join(m->dir, sub);
        char * im = path_join(m->dir, "images");
        if (tsr_trace2_write(t, d, im) != 0) m->st.failed++;
        else m->st.written++;
        free(d);
        free(im);
    }
    if (m->on) {
        if (m->qn == m->qcap) {
            m->qcap = m->qcap ? 2 * m->qcap : 8;
            m->q = (tsr_trace2 **)xrealloc(m->q, m->qcap * sizeof(tsr_trace2 *));
        }
        m->q[m->qn++] = t;
    } else {
        tsr_trace2_free(t);
    }
}

static void launch_begin(tmodel * m, uint32_t seq) {
    if (m->cur.active) cur_abort(m);                  // (cannot happen: TAKEN - ANSWERED <= 1)
    if (!m->on && m->dir == NULL) return;
    const int slot = (int)(seq % LQ_SHADOW);
    if (!m->lq[slot].valid || m->lq[slot].seq != seq) { m->st.skipped++; return; }
    const uint32_t * d = m->lq[slot].w;
    if (TSR_GET(d[0], TSR_DESC_TYPE) != TSR_DESC_TYPE_LAUNCH || (d[0] & 0x0000FE00u) != 0) { m->st.skipped++; return; }
    if (m->user_img == NULL && m->img == NULL) {
        fprintf(stderr, "tsr_trace: warning: launch %u: no image known (loaded before the TRACE model?); not captured\n", seq);
        m->st.skipped++;
        return;
    }
    cur_start(m, 0);
    m->cur.seq = seq;
    memcpy(m->cur.desc, d, sizeof(m->cur.desc));
}

static void launch_end(tmodel * m, const uint32_t r[8]) {
    cap_cur * c = &m->cur;
    if (TSR_GET(r[0], TSR_LCQ_R0_TYPE) != TSR_LCQ_R0_TYPE_DONE) {   // TRAP / PONG answer
        cur_abort(m);
        m->st.skipped++;
        return;
    }
    tsr_trace2 * t = tsr_trace2_new();
    t->kind = TSR_TRACE2_KIND_LAUNCH;
    const uint32_t seq = c->seq;
    const uint8_t * img = m->user_img ? m->user_img : m->img;
    const size_t img_n = m->user_img ? m->user_n : m->img_n;
    t->image = (uint8_t *)xmalloc(img_n);
    memcpy(t->image, img, img_n);
    t->image_size = img_n;
    img_info e;
    if (img_parse(t->image, img_n, TCM_LIMIT, TCM_LIMIT, &e, NULL, 0) == 0) {
        t->image_id = img_id(t->image, &e);
        img_kernel_name(t->image, &e, TSR_GET(c->desc[0], TSR_DESC_KERNEL), t->kernel);
    }
    t->kernel_index = TSR_GET(c->desc[0], TSR_DESC_KERNEL);
    t->prof = TSR_GET(c->desc[0], TSR_DESC_PROF);
    memcpy(t->args, c->desc + 1, sizeof(t->args));
    tsr_trace_done * d = &t->done;
    d->status = TSR_GET(r[0], TSR_LCQ_R0_STATUS);
    d->err_code = TSR_GET(r[0], TSR_LCQ_R0_ERR_CODE);
    d->ret = (int32_t)r[2];
    d->err_info = r[4];
    d->err_w0 = r[5];
    if ((r[3] >> 28) < TSR_NUM_UNITS) {               // a failing ticket: relative ticket and push index
        d->err_ticket_rel = tsr_trace_rel_ticket(r[3], c->base);
        d->err_pidx_rel = r[6] - c->pidx_base;
    } else {
        d->err_ticket_rel = r[3];
        d->err_pidx_rel = r[6];
    }
    cur_finish(m, t);
    cur_reset(c);
    deliver(m, t, seq);
}

static void m_sync(tmodel * m) {
    tsr_sim * sim = m->sim;
    const uint64_t gen = tsr_sim_rv_boot_gen(sim);
    if (gen != m->boot_gen) {
        m->boot_gen = gen;
        image_snapshot(m);
    }
    const uint32_t pushed = csr(sim, TSR_CSR_LQ_SEQ);
    if (pushed != m->seen_pushed) {
        uint32_t w[8];
        for (uint32_t i = 0; i < 8; ++i) w[i] = csr(sim, TSR_CSR_LQ_W(i));
        for (uint32_t k = 0; m->seen_pushed != pushed && k < LQ_SHADOW; ++k, ++m->seen_pushed) {
            const uint32_t s = m->seen_pushed % LQ_SHADOW;
            m->lq[s].seq = m->seen_pushed;
            m->lq[s].valid = 1;
            memcpy(m->lq[s].w, w, sizeof(w));
        }
        m->seen_pushed = pushed;
    }
    const uint32_t answered = csr(sim, TSR_CSR_LCQ_ANSWERED);
    if (answered != m->seen_answered) {
        if (m->cur.active && !m->cur.run) {
            if (m->post_seen && answered == m->seen_answered + 1u) launch_end(m, m->post_w);
            else { cur_abort(m); m->st.skipped++; }                  // reset sequence / SOFT_RESET
        }
        m->seen_answered = answered;
        m->post_seen = 0;
    }
    const uint32_t taken = csr(sim, TSR_CSR_LQ_TAKEN);
    if (taken != m->seen_taken) {
        if (m->pop_seen && taken == m->seen_taken + 1u && !m->cur.run) launch_begin(m, m->seen_taken);
        m->seen_taken = taken;
        m->pop_seen = 0;
    }
}

static void m_acc(void * ctx, tsr_sim * sim, const tsr_sim_acc * a) {
    tmodel * m = (tmodel *)ctx;
    (void)sim;
    if (m->busy) return;
    m->busy = 1;
    switch (a->kind) {
    case TSR_ACC_HOST_CSR_WR:
        if (a->off == TSR_CSR_LOAD_ADDR) m->load_cur = a->val & ~3u;
        else if (a->off == TSR_CSR_LOAD_DATA) loader_word(m, a->val);
        m_sync(m);
        break;
    case TSR_ACC_RV_CW_WR:
        if (a->off == TSR_CW_LQ_POP) {
            m->pop_seen = 1;
        } else if (a->off == TSR_CW_LCQ_POST && (a->val & TSR_LCQ_POST_ANSWER)) {
            m->post_seen = 1;
            // The record, from its ring slot (the event comes after the store completed).
            const uint32_t k = (csr(m->sim, TSR_CSR_LCQ_WR) - 1u) % TSR_LCQ_DEPTH;
            for (uint32_t i = 0; i < 8; ++i) m->post_w[i] = csr(m->sim, TSR_CSR_LCQ_REC(k, i));
        }
        m_sync(m);
        break;
    case TSR_ACC_RV_CW_RD:
        m_sync(m);
        if (a->off == TSR_CW_ISSUE_PIDX && m->cur.active && !m->cur.run && !m->cur.entry_seen) {
            m->cur.entry_seen = 1;                    // kernel entry (SW §6.3 R5)
            m->cur.pidx_known = 1;
            m->cur.pidx_base = a->val;
            for (uint32_t u = 0; u < TSR_NUM_UNITS; ++u) m->cur.base[u] = csr(m->sim, TSR_CSR_Q_ISSUED(u)) & SEQ_MASK;
        }
        break;
    case TSR_ACC_HOST_DRAM_WR:
        if (m->cur.active) m->cur.host_writes++;
        break;
    default:
        break;
    }
    m->busy = 0;
}

static void m_q(void * ctx, tsr_sim * sim, const tsr_q_event * ev) {
    tmodel * m = (tmodel *)ctx;
    (void)sim;
    if (ev->kind != TSR_QEV_PUSH && ev->kind != TSR_QEV_QCTRL && ev->kind != TSR_QEV_MARK_PUSH) return;
    if (!m->busy) {
        m->busy = 1;
        m_sync(m);
        m->busy = 0;
    }
    if (!m->cur.active) return;
    if (!m->cur.pidx_known) {                         // ISSUE_PIDX before this store
        m->cur.pidx_known = 1;
        m->cur.pidx_base = ev->pidx;
    }
    tsr_trace_push p;
    memset(&p, 0, sizeof(p));
    if (ev->kind == TSR_QEV_PUSH) {
        memcpy(p.w, ev->w, sizeof(p.w));
        p.dep[0] = ev->dep[0];
        p.dep[1] = ev->dep[1];
        p.push = ev->push;
        p.kind = TSR_TRACE_PUSH_OP;
    } else {
        p.w[0] = ev->w[0];
        p.kind = ev->kind == TSR_QEV_QCTRL ? TSR_TRACE_PUSH_QCTRL : TSR_TRACE_PUSH_MARK;
        if (p.kind == TSR_TRACE_PUSH_MARK && !m->cur.run) p.w[0] = launch_mark_rel(&m->cur, p.w[0]);
    }
    cur_push(&m->cur, &p);
}

void tsr_trace_attach(tsr_sim * sim) {
    void ** slot = tsr_sim_model_slot(sim, TSR_MODEL_TRACE);
    if (slot == NULL || *slot != NULL) return;
    tmodel * m = (tmodel *)xcalloc(1, sizeof(tmodel));
    m->sim = sim;
    m->dir_index = -1;
    const char * d = tsr_sim_opt_trace_dir(sim);
    if (d != NULL && *d != '\0') m->dir = xstrdup(d);
    (void)tsr_sim_itcm(sim, &m->tcm_bytes[0]);
    (void)tsr_sim_dtcm(sim, &m->tcm_bytes[1]);
    for (int k = 0; k < 2; ++k) {
        m->tcm[k] = (uint8_t *)xcalloc(m->tcm_bytes[k] ? m->tcm_bytes[k] : 4, 1);
        m->mask[k] = (uint8_t *)xcalloc(m->tcm_bytes[k] / 4u + 1u, 1);
    }
    m->seen_pushed = csr(sim, TSR_CSR_LQ_SEQ);
    m->seen_taken = csr(sim, TSR_CSR_LQ_TAKEN);
    m->seen_answered = csr(sim, TSR_CSR_LCQ_ANSWERED);
    m->boot_gen = tsr_sim_rv_boot_gen(sim);
    m->load_cur = csr(sim, TSR_CSR_LOAD_ADDR);
    m->lis_id = tsr_sim_add_listener(sim, m_q, m);
    m->acc_id = tsr_sim_add_acc_listener(sim, m_acc, m);
    if (m->lis_id < 0 || m->acc_id < 0) fprintf(stderr, "tsr_trace: warning: no listener slot left; launch capture is off\n");
    *slot = m;
}

void tsr_trace_detach(tsr_sim * sim) {
    void ** slot = tsr_sim_model_slot(sim, TSR_MODEL_TRACE);
    if (slot == NULL || *slot == NULL) return;
    tmodel * m = (tmodel *)*slot;
    if (m->lis_id >= 0) tsr_sim_remove_listener(sim, m->lis_id);
    if (m->acc_id >= 0) tsr_sim_remove_acc_listener(sim, m->acc_id);
    if (m->cur.active) cur_abort(m);
    cur_reset(&m->cur);
    for (size_t i = 0; i < m->qn; ++i) tsr_trace2_free(m->q[i]);
    free(m->q);
    for (int k = 0; k < 2; ++k) {
        free(m->tcm[k]);
        free(m->mask[k]);
    }
    free(m->img);
    free(m->user_img);
    free(m->dir);
    free(m);
    *slot = NULL;
}

void tsr_sim_capture_launch(tsr_sim * sim, int on) {
    tmodel * m = model_of(sim);
    if (m != NULL) m->on = on != 0;
}

int tsr_sim_capture_take(tsr_sim * sim, tsr_trace2 * out) {
    tmodel * m = model_of(sim);
    if (m == NULL) return -1;
    if (m->qn == 0) return 1;
    tsr_trace2 * t = m->q[0];
    memmove(m->q, m->q + 1, (m->qn - 1) * sizeof(tsr_trace2 *));
    m->qn--;
    tsr_trace2_clear(out);
    *out = *t;                                        // move the contents
    free(t);
    return 0;
}

void tsr_sim_capture_set_image(tsr_sim * sim, const void * elf, size_t n) {
    tmodel * m = model_of(sim);
    if (m == NULL) return;
    free(m->user_img);
    m->user_img = NULL;
    m->user_n = 0;
    if (elf == NULL || n == 0) return;
    img_info e;
    char why[256];
    if (img_parse((const uint8_t *)elf, n, m->tcm_bytes[0], m->tcm_bytes[1], &e, why, sizeof(why))) {
        fprintf(stderr, "tsr_trace: warning: tsr_sim_capture_set_image: %s; ignored\n", why);
        return;
    }
    m->user_img = (uint8_t *)xmalloc(n);
    memcpy(m->user_img, elf, n);
    m->user_n = n;
}

void tsr_sim_capture_stats(tsr_sim * sim, tsr_trace_capture_stats * s) {
    memset(s, 0, sizeof(*s));
    tmodel * m = model_of(sim);
    if (m == NULL) return;
    *s = m->st;
    s->queued = (uint32_t)m->qn;
}

// ---------------------------------------------------------------------------
// Replay (SW §5.4)
// ---------------------------------------------------------------------------
static void rfailf(tsr_trace2_result * r, int verbose, uint32_t cls, const char * fmt, ...) __attribute__((format(printf, 4, 5)));
static void rfailf(tsr_trace2_result * r, int verbose, uint32_t cls, const char * fmt, ...) {
    char msg[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof(msg), fmt, ap);
    va_end(ap);
    if (r->n_fail == 0) snprintf(r->msg, sizeof(r->msg), "%s", msg);
    r->n_fail++;
    r->fail_mask |= cls;
    r->ok = 0;
    if (verbose) fprintf(stderr, "replay: %s\n", msg);
}

static uint32_t crc32_bytes(const uint8_t * p, size_t n) {
    uint32_t c = 0xFFFFFFFFu;
    for (size_t i = 0; i < n; ++i) {
        c ^= p[i];
        for (int k = 0; k < 8; ++k) c = (c >> 1) ^ (TSR_CRC32_POLY & (0u - (c & 1u)));
    }
    return ~c;
}

// Cold load through the host CSR port (EX §3.3) and boot (EX §3.4).
static int boot_image(tsr_sim * sim, const uint8_t * elf, size_t n, char * why, size_t wn) {
    img_info e;
    uint32_t itcm = 0, dtcm = 0;
    (void)tsr_sim_itcm(sim, &itcm);
    (void)tsr_sim_dtcm(sim, &dtcm);
    if (img_parse(elf, n, itcm, dtcm, &e, why, wn)) return -1;
    if (e.entry < TSR_BOOT_ENTRY_OFF || ((e.entry - TSR_BOOT_ENTRY_OFF) & ~TSR_RV_BOOT_ADDR) != 0) {
        seterr(why, wn, "image entry 0x%08x is not RV_BOOT + 0x80 with RV_BOOT 256-byte aligned (EX §3.4)", e.entry);
        return -1;
    }
    for (uint32_t s = 0; s < e.nseg; ++s) {
        const img_seg * g = &e.seg[s];
        if (g->filesz == 0) continue;                 // .bss only: crt0 zeroes it
        const uint32_t bytes = (g->filesz + 3u) & ~3u;
        uint8_t * buf = (uint8_t *)xcalloc(bytes, 1);
        memcpy(buf, elf + g->off, g->filesz);
        const uint32_t crc = crc32_bytes(buf, bytes);
        tsr_sim_csr_write(sim, TSR_CSR_LOAD_ADDR, g->paddr);
        for (uint32_t i = 0; i < bytes; i += 4) tsr_sim_csr_write(sim, TSR_CSR_LOAD_DATA, rd32(buf + i));
        const uint32_t rx = tsr_sim_csr_read(sim, TSR_CSR_LOAD_CRC);
        tsr_sim_csr_write(sim, TSR_CSR_LOAD_ADDR, g->paddr);
        tsr_sim_csr_write(sim, TSR_CSR_LOAD_COUNT, bytes / 4u);
        tsr_sim_csr_write(sim, TSR_CSR_LOAD_CTRL, TSR_LOAD_CTRL_VERIFY);
        for (int i = 0; i < 1000 && (tsr_sim_csr_read(sim, TSR_CSR_LOAD_STATUS) & TSR_LOAD_STATUS_VERIFY_BUSY); ++i)
            tsr_sim_advance(sim, 1000);
        const uint32_t vf = tsr_sim_csr_read(sim, TSR_CSR_LOAD_CRC);
        free(buf);
        if (rx != crc || vf != crc) {
            seterr(why, wn, "segment 0x%08x: receive CRC 0x%08x / VERIFY CRC 0x%08x, expected 0x%08x", g->paddr, rx, vf, crc);
            return -1;
        }
    }
    const uint32_t st = tsr_sim_csr_read(sim, TSR_CSR_STATUS);
    if (st & (TSR_STATUS_LOAD_REFUSED | TSR_STATUS_LOAD_TIMEOUT)) {
        seterr(why, wn, "loader refused or timed out (STATUS 0x%08x)", st);
        return -1;
    }
    tsr_sim_csr_write(sim, TSR_CSR_RV_BOOT, e.entry - TSR_BOOT_ENTRY_OFF);
    tsr_sim_csr_write(sim, TSR_CSR_CTRL, tsr_sim_csr_read(sim, TSR_CSR_CTRL) | TSR_CTRL_RV_RUN);
    return 0;
}

#define STEP_CYCLES (1ull << 22)

// Kind launch: boot, HELLO, push the descriptor, wait for its answer. The TRACE model of sim
// captures the launch; *got receives it.
static int launch_replay(tsr_sim * sim, tmodel * m, const tsr_trace2 * t, uint64_t max_cycles, tsr_trace2 ** got,
                         char * why, size_t wn) {
    *got = NULL;
    m->on = 1;
    if (boot_image(sim, t->image, t->image_size, why, wn)) return -1;
    uint32_t lcq_rd = tsr_sim_csr_read(sim, TSR_CSR_LCQ_RD);
    int hello = 0, pushed = 0, answered = 0, idle_steps = 0;
    uint32_t seq = 0, ans_type = 0;
    const uint64_t t_start = tsr_sim_now(sim);
    uint64_t last_progress = 0;
    while (!answered) {
        const uint32_t ev = tsr_sim_advance(sim, STEP_CYCLES);
        const uint32_t wr = tsr_sim_csr_read(sim, TSR_CSR_LCQ_WR);
        for (uint32_t k = lcq_rd; k != wr; ++k) {
            const uint32_t r0 = tsr_sim_csr_read(sim, TSR_CSR_LCQ_REC(k % TSR_LCQ_DEPTH, 0));
            const uint32_t r1 = tsr_sim_csr_read(sim, TSR_CSR_LCQ_REC(k % TSR_LCQ_DEPTH, 1));
            const uint32_t type = TSR_GET(r0, TSR_LCQ_R0_TYPE);
            if (!hello) {
                if (type != TSR_LCQ_R0_TYPE_HELLO) {
                    seterr(why, wn, "the first record after boot is type 0x%02x, not HELLO (EX §3.4)", type);
                    return -1;
                }
                hello = 1;
            } else if (pushed && r1 == seq &&
                       (type == TSR_LCQ_R0_TYPE_DONE || type == TSR_LCQ_R0_TYPE_PONG || type == TSR_LCQ_R0_TYPE_TRAP)) {
                answered = 1;
                ans_type = type;
            }
        }
        lcq_rd = wr;
        tsr_sim_csr_write(sim, TSR_CSR_LCQ_RD, wr);
        tsr_sim_csr_write(sim, TSR_CSR_LOG_RD, tsr_sim_csr_read(sim, TSR_CSR_LOG_WR));   // log content is not compared
        if (hello && !pushed) {
            seq = tsr_sim_csr_read(sim, TSR_CSR_LQ_SEQ);
            uint32_t d[8];
            d[0] = TSR_PUT(TSR_DESC_TYPE, TSR_DESC_TYPE_LAUNCH) | TSR_PUT(TSR_DESC_PROF, t->prof) |
                   TSR_PUT(TSR_DESC_KERNEL, t->kernel_index);
            memcpy(d + 1, t->args, sizeof(t->args));
            for (uint32_t i = 0; i < 8; ++i) tsr_sim_csr_write(sim, TSR_CSR_LQ_W(i), d[i]);
            tsr_sim_csr_write(sim, TSR_CSR_LQ_PUSH, 1u);
            pushed = 1;
        }
        if (answered) break;
        const uint32_t rt = tsr_sim_csr_read(sim, TSR_CSR_RT_STATE);
        if (TSR_GET(rt, TSR_RT_STATE_PHASE) == TSR_RT_STATE_PHASE_TRAPPED) {
            seterr(why, wn, "the firmware trapped (RT_STATE 0x%08x, RT_INFO 0x%08x, pc 0x%08x)", rt,
                   tsr_sim_csr_read(sim, TSR_CSR_RT_INFO), tsr_sim_rv_pc(sim));
            return -1;
        }
        if (tsr_sim_now(sim) - t_start > max_cycles) {
            seterr(why, wn, "%s within %llu core cycles", hello ? "no answer" : "no HELLO", (unsigned long long)max_cycles);
            return -1;
        }
        // Stuck: the hart asleep, the queues idle and nothing retired over two whole steps.
        const uint64_t progress = tsr_sim_rv_instret(sim) + wr + tsr_sim_csr_read(sim, TSR_CSR_LQ_TAKEN);
        const uint32_t st = tsr_sim_csr_read(sim, TSR_CSR_STATUS);
        if ((ev & TSR_SIM_EV_BUDGET) && progress == last_progress && (st & TSR_STATUS_QIDLE) && (st & TSR_STATUS_RV_SLEEP)) {
            if (++idle_steps >= 2) {
                seterr(why, wn, "stuck: %s, the hart asleep and the queues idle (STATUS 0x%08x)",
                       hello ? "no answer" : "no HELLO", st);
                return -1;
            }
        } else {
            idle_steps = 0;
        }
        last_progress = progress;
    }
    tsr_trace2 * c = tsr_trace2_new();
    if (tsr_sim_capture_take(sim, c) != 0) {
        tsr_trace2_free(c);
        seterr(why, wn, "the launch was answered by a %s record and not captured",
               ans_type == TSR_LCQ_R0_TYPE_DONE ? "DONE" : ans_type == TSR_LCQ_R0_TYPE_TRAP ? "TRAP" : "PONG");
        return -1;
    }
    *got = c;
    return 0;
}

// One store from outside the time base that may wait for a credit / FIFO space: retried at the
// next event (the replay kernel's WAIT_CREDIT wait, without ISSUE_WAIT_MAX).
static int store_retry(tsr_sim * sim, uint32_t off, uint32_t val, uint64_t t_end) {
    for (;;) {
        uint32_t v = val;
        const int rc = tsr_sim_cw_access(sim, 1, off, &v);
        if (rc == TSR_CW_OK) return 0;
        if (rc == TSR_CW_FAULT) return -1;
        const uint64_t t = tsr_sim_next_event(sim), now = tsr_sim_now(sim);
        if (t == UINT64_MAX) return -2;               // nothing will ever free it
        if (now >= t_end) return -3;
        tsr_sim_advance(sim, t > now ? t - now : 1);
    }
}

// Kind ops, tile-only: the records through the control window, then until QIDLE.
static int ops_replay(tsr_sim * sim, tmodel * m, const tsr_trace2 * t, uint64_t max_cycles, tsr_trace2 ** got,
                      char * why, size_t wn) {
    *got = NULL;
    m->busy = 1;
    cur_start(m, 1);
    m->busy = 0;
    const uint64_t t_end = tsr_sim_now(sim) + max_cycles;
    for (size_t i = 0; i < t->n_push; ++i) {
        const tsr_trace_push * p = &t->push[i];
        uint32_t off = TSR_CW_ISSUE_PUSH, val = p->push;
        if (p->kind == TSR_TRACE_PUSH_OP) {
            for (uint32_t k = 0; k < 8; ++k) tsr_sim_cw_access(sim, 1, TSR_CW_ISSUE_W(k), (uint32_t[]){ p->w[k] });
            tsr_sim_cw_access(sim, 1, TSR_CW_ISSUE_DEP0, (uint32_t[]){ tsr_trace_abs_ticket(p->dep[0], m->cur.base) });
            tsr_sim_cw_access(sim, 1, TSR_CW_ISSUE_DEP1, (uint32_t[]){ tsr_trace_abs_ticket(p->dep[1], m->cur.base) });
        } else {
            off = p->kind == TSR_TRACE_PUSH_QCTRL ? TSR_CW_QCTRL : TSR_CW_TRACE_MARK;
            val = p->w[0];
        }
        const int rc = store_retry(sim, off, val, t_end);
        if (rc != 0) {
            seterr(why, wn, "push record %zu (kind %u, w0 0x%08x): %s", i, p->kind, p->w[0],
                   rc == -1 ? "the store faulted" : rc == -2 ? "waits forever (no credit and nothing pending: a hold?)"
                                                             : "no credit within the cycle budget");
            m->busy = 1;
            cur_abort(m);
            m->busy = 0;
            return -1;
        }
    }
    const uint64_t now = tsr_sim_now(sim);
    if (!tsr_sim_run_until_idle(sim, t_end > now ? t_end - now : 0)) {
        seterr(why, wn, "the queues are not idle after the run (a hold left set, or the cycle budget)");
        m->busy = 1;
        cur_abort(m);
        m->busy = 0;
        return -1;
    }
    tsr_trace2 * c = tsr_trace2_new();
    m->busy = 1;
    cur_finish(m, c);
    m->busy = 0;
    cur_reset(&m->cur);
    done_of_units(c);
    *got = c;
    return 0;
}

// Creates the replay simulator with t's configuration and loads the init regions.
static tsr_sim * replay_sim(const tsr_trace2 * t, const tsr_trace2_replay_opts * o, tmodel ** mp, char * why, size_t wn) {
    const tsr_config cfg = o->cfg ? *o->cfg : t->cfg.core;
    tsr_sim_opts so;
    memset(&so, 0, sizeof(so));
    so.dram_dir = o->dram_dir;
    so.threads = o->threads;
    so.sched = o->sched;
    so.rv_ratio = o->rv_ratio != 0.0 ? o->rv_ratio : t->rv_ratio;
    so.q_depth = t->cfg.q_depth;
    so.race = o->race;                                // P2.5: 0 keeps TSR_SIM_RACE
    so.itcm_bytes = t->cfg.itcm_bytes;
    so.dtcm_bytes = t->cfg.dtcm_bytes;
    tsr_sim * sim = tsr_sim_create_opts(&cfg, &so);
    if (sim == NULL) {
        seterr(why, wn, "cannot create a simulator for the trace's configuration%s%s", o->sched ? " / schedule " : "",
               o->sched ? o->sched : "");
        return NULL;
    }
    tmodel * m = model_of(sim);
    if (m == NULL) {
        tsr_trace_attach(sim);
        m = model_of(sim);
    }
    if (m == NULL) {
        seterr(why, wn, "no TRACE model");
        tsr_sim_destroy(sim);
        return NULL;
    }
    free(m->dir);                                     // a replay never dumps (TSR_TRACE)
    m->dir = NULL;
    m->on = 0;
    for (size_t i = 0; i < t->mem.n_init; ++i) {
        const tsr_region * r = &t->mem.init[i];
        if (r->ch >= cfg.num_ch || r->byte_off + r->size > cfg.ch_bytes ||
            tsr_sim_dram_write(sim, r->ch, r->byte_off, r->data, r->size) != 0) {
            seterr(why, wn, "init region %zu (ch %u, offset 0x%llx) does not fit the replay configuration", i, r->ch,
                   (unsigned long long)r->byte_off);
            tsr_sim_destroy(sim);
            return NULL;
        }
    }
    *mp = m;
    return sim;
}

static int spans_equal(const span * a, size_t na, const span * b, size_t nb, size_t * at) {
    size_t i = 0;
    while (i < na && i < nb && a[i].ch == b[i].ch && a[i].lo == b[i].lo && a[i].hi == b[i].hi) ++i;
    *at = i;
    return i == na && i == nb;
}

static void compare(const tsr_trace2 * w, const tsr_trace2 * g, tsr_sim * sim, int verbose, tsr_trace2_result * r) {
    const tsr_trace_done * a = &w->done, * b = &g->done;
    if (a->status != b->status || a->ret != b->ret || a->err_code != b->err_code || a->err_ticket_rel != b->err_ticket_rel ||
        a->err_pidx_rel != b->err_pidx_rel || a->err_info != b->err_info || a->err_w0 != b->err_w0)
        rfailf(r, verbose, TSR_TRACE2_CMP_DONE,
               "DONE differs: status %u ret %d err_code %u ticket 0x%08x pidx %u info 0x%08x w0 0x%08x, expected status %u "
               "ret %d err_code %u ticket 0x%08x pidx %u info 0x%08x w0 0x%08x",
               b->status, b->ret, b->err_code, b->err_ticket_rel, b->err_pidx_rel, b->err_info, b->err_w0, a->status, a->ret,
               a->err_code, a->err_ticket_rel, a->err_pidx_rel, a->err_info, a->err_w0);
    // Expect content, bit-exactly (the DRAM after the replayed launch).
    for (size_t i = 0; i < w->mem.n_expect; ++i) {
        const tsr_region * x = &w->mem.expect[i];
        const uint8_t * d = tsr_sim_dram(sim, x->ch);
        if (d == NULL || x->byte_off + x->size > tsr_sim_config(sim)->ch_bytes) {
            rfailf(r, verbose, TSR_TRACE2_CMP_EXPECT, "expect region ch %u offset 0x%llx is outside the replay's DRAM", x->ch,
                   (unsigned long long)x->byte_off);
            continue;
        }
        d += x->byte_off;
        if (memcmp(d, x->data, (size_t)x->size) != 0) {
            size_t k = 0;
            while (d[k] == x->data[k]) ++k;
            size_t nd = 0;
            for (size_t j = k; j < x->size; ++j) nd += d[j] != x->data[j];
            const size_t wo = k & ~(size_t)3;
            rfailf(r, verbose, TSR_TRACE2_CMP_EXPECT,
                   "expect mismatch at ch %u byte 0x%llx: got 0x%02x, expected 0x%02x (word got 0x%08x expected 0x%08x; %zu "
                   "differing bytes in this region)", x->ch, (unsigned long long)(x->byte_off + k), d[k], x->data[k],
                   wo + 4 <= x->size ? rd32(d + wo) : 0u, wo + 4 <= x->size ? rd32(x->data + wo) : 0u, nd);
        }
    }
    // Written beats = the union of the expect regions; reads inside init (or written earlier).
    {
        span * ws = regions_spans(w->mem.expect, w->mem.n_expect), * gs = regions_spans(g->mem.expect, g->mem.n_expect);
        const size_t nw = span_normalize(ws, w->mem.n_expect), ng = span_normalize(gs, g->mem.n_expect);
        size_t at = 0;
        if (!spans_equal(ws, nw, gs, ng, &at)) {
            if (at < ng && (at >= nw || span_cmp(&gs[at], &ws[at]) <= 0))
                rfailf(r, verbose, TSR_TRACE2_CMP_WRITTEN, "the replay wrote ch %u bytes [0x%llx, 0x%llx) but the expect regions "
                       "there differ (%zu written ranges, %zu expected)", gs[at].ch, (unsigned long long)gs[at].lo,
                       (unsigned long long)gs[at].hi, ng, nw);
            else
                rfailf(r, verbose, TSR_TRACE2_CMP_WRITTEN, "expect region ch %u bytes [0x%llx, 0x%llx) is not exactly what the "
                       "replay wrote (%zu written ranges, %zu expected)", ws[at].ch, (unsigned long long)ws[at].lo,
                       (unsigned long long)ws[at].hi, ng, nw);
        }
        free(ws);
        free(gs);
        span * is = regions_spans(w->mem.init, w->mem.n_init);
        const size_t ni = span_normalize(is, w->mem.n_init);
        for (size_t i = 0; i < g->mem.n_init; ++i) {
            const tsr_region * x = &g->mem.init[i];
            if (!span_covered(is, ni, x->ch, x->byte_off, x->byte_off + x->size)) {
                rfailf(r, verbose, TSR_TRACE2_CMP_READS, "the replay read ch %u bytes [0x%llx, 0x%llx) that the trace neither "
                       "initialises nor wrote earlier", x->ch, (unsigned long long)x->byte_off,
                       (unsigned long long)(x->byte_off + x->size));
                break;
            }
        }
        free(is);
    }
    // Per-unit counts and error records.
    const uint32_t nu = w->n_units > g->n_units ? w->n_units : g->n_units;
    for (uint32_t u = 0; u < nu; ++u) {
        static const tsr_trace_unit zero;
        const tsr_trace_unit * x = u < w->n_units ? &w->unit[u] : &zero, * y = u < g->n_units ? &g->unit[u] : &zero;
        if (x->count != y->count)
            rfailf(r, verbose, TSR_TRACE2_CMP_COUNTS, "unit %u: %u entries completed, expected %u", u, y->count, x->count);
        if (x->err_code != y->err_code || x->err_rel_seq != y->err_rel_seq || x->err_info != y->err_info ||
            x->err_w0 != y->err_w0 || x->err_rel_pidx != y->err_rel_pidx)
            rfailf(r, verbose, TSR_TRACE2_CMP_ERRREC, "unit %u: error record code %u seq %u info 0x%08x w0 0x%08x pidx %u, "
                   "expected code %u seq %u info 0x%08x w0 0x%08x pidx %u", u, y->err_code, y->err_rel_seq, y->err_info,
                   y->err_w0, y->err_rel_pidx, x->err_code, x->err_rel_seq, x->err_info, x->err_w0, x->err_rel_pidx);
    }
    // Per-unit projections of push.bin (records without a ticket form one more stream).
    for (uint32_t s = 0; s <= TSR_NUM_UNITS; ++s) {
        const uint32_t unit = s < TSR_NUM_UNITS ? s : TSR_UNIT_NONE;
        size_t i = 0, j = 0, k = 0;
        for (;;) {
            while (i < w->n_push && tsr_trace2_push_unit(&w->cfg, &w->push[i]) != unit) ++i;
            while (j < g->n_push && tsr_trace2_push_unit(&w->cfg, &g->push[j]) != unit) ++j;
            if (i >= w->n_push || j >= g->n_push) break;
            if (memcmp(&w->push[i], &g->push[j], sizeof(tsr_trace_push)) != 0) {
                const tsr_trace_push * p = &w->push[i], * q = &g->push[j];
                rfailf(r, verbose, TSR_TRACE2_CMP_PUSH, "%s%u record %zu (push %zu): kind %u w0 0x%08x w1 0x%08x deps "
                       "0x%08x,0x%08x push 0x%08x, expected kind %u w0 0x%08x w1 0x%08x deps 0x%08x,0x%08x push 0x%08x "
                       "(push record %zu)", s < TSR_NUM_UNITS ? "unit " : "control stream ", s, k, j, q->kind, q->w[0],
                       q->w[1], q->dep[0], q->dep[1], q->push, p->kind, p->w[0], p->w[1], p->dep[0], p->dep[1], p->push, i);
                break;
            }
            ++i; ++j; ++k;
        }
        while (i < w->n_push && tsr_trace2_push_unit(&w->cfg, &w->push[i]) != unit) ++i;
        while (j < g->n_push && tsr_trace2_push_unit(&w->cfg, &g->push[j]) != unit) ++j;
        if ((i < w->n_push) != (j < g->n_push) && !(r->fail_mask & TSR_TRACE2_CMP_PUSH))
            rfailf(r, verbose, TSR_TRACE2_CMP_PUSH, "%s%u: the replay pushed %s records than the trace (after %zu equal ones)",
                   s < TSR_NUM_UNITS ? "unit " : "control stream ", s, i < w->n_push ? "fewer" : "more", k);
    }
    if (w->perf.valid && g->perf.valid &&
        (w->perf.core_cycles != g->perf.core_cycles || memcmp(w->perf.unit_busy, g->perf.unit_busy, sizeof(w->perf.unit_busy)) != 0))
        r->perf_differs = 1;
    if (w->cfg.ctl_id != g->cfg.ctl_id || w->cfg.cfg3 != g->cfg.cfg3) {
        if (verbose)
            fprintf(stderr, "replay: warning: CTL_ID 0x%08x / CFG3 0x%08x differ from the trace's 0x%08x / 0x%08x "
                            "(informational)\n", g->cfg.ctl_id, g->cfg.cfg3, w->cfg.ctl_id, w->cfg.cfg3);
    }
}

// ---------------------------------------------------------------------------
// P2.5: kind ops through the firmware's `replay` kernel (SW §5.4, §7.7; EX §4.6). Decisions:
//   - The kernel is found by name in the image's .tsr_kernels (entry stride 32 or 36 bytes, the name
//     at byte 12: SW §6.2's field list; whichever stride gives a valid table).
//   - push.bin goes to "ops.load" (choose_load when the trace has none), written by the host before
//     the launch (EX §2.3 H1). The SPM buffer is the highest run of free lines below the runtime
//     argument range that no SPM extent of the trace's ops touches (the checker's extents of every
//     accepted op record): one buffer of the whole stream when it fits one chunk, else two
//     TSR_REPLAY_CHUNK_MAX buffers. So the kernel's buffer reads never race with the replayed ops.
//   - The captured launch (kind launch, base_u at kernel entry) is turned back into the ops view: the
//     kernel's own records are its chunk LDs (LD from the push.bin range into the buffer) and every
//     record after the last replayed one (the kernel's and the runtime's FENCEs); the remaining
//     records' deps, the error records' sequences and push indices, and the DONE record's ticket and
//     push index are renumbered without the own records (rel - own records of that unit / of any unit
//     pushed earlier with a smaller sequence / push index), counts lose the own records, and the
//     DRAM read of push.bin is removed from the reads. The result is compared exactly as a tile-only
//     replay (compare below).
// ---------------------------------------------------------------------------
static int kr_find_kernel(const uint8_t * b, const img_info * e, const char * name, uint32_t * index) {
    static const uint64_t strides[2] = { 32, 36 };
    if (!e->has_kernels) return -1;
    for (int s = 0; s < 2; ++s) {
        const uint64_t st = strides[s];
        if (e->k_size == 0 || e->k_size % st != 0) continue;
        const uint64_t n = e->k_size / st;
        int valid = 1, found = -1;
        for (uint64_t i = 0; i < n && valid; ++i) {
            const uint8_t * nm = b + e->k_off + st * i + 12;
            const uint64_t cap = st - 12;
            uint64_t L = 0;
            while (L < cap && nm[L] >= 0x20 && nm[L] < 0x7F) ++L;
            if (L == 0 || L == cap) { valid = 0; break; }
            for (uint64_t k = L; k < cap; ++k)
                if (nm[k] != 0) valid = 0;
            if (valid && strlen(name) == L && memcmp(nm, name, L) == 0 && found < 0) found = (int)i;
        }
        if (valid && found >= 0) {
            *index = (uint32_t)found;
            return 0;
        }
    }
    return -1;
}

// Lines [*line, *line + need) below spm_lines - 16 that no SPM extent of t's accepted op records
// touches, the highest such run. 0, or -1 when there is none.
static int kr_spm_buffer(tsr_sim * sim, const tsr_trace2 * t, uint32_t need, uint32_t * line) {
    const uint32_t L = tsr_sim_spm_lines(sim);
    if (L < 16u + need) return -1;
    uint8_t * used = (uint8_t *)xcalloc(L, 1);
    for (size_t i = 0; i < t->n_push; ++i) {
        const tsr_trace_push * p = &t->push[i];
        if (p->kind != TSR_TRACE_PUSH_OP) continue;
        const uint32_t u = tsr_trace2_push_unit(&t->cfg, p);
        const tsr_op_handler * h = tsr_sim_op_handler((uint8_t)(p->w[0] & 0xFFu));
        if (u == TSR_UNIT_SYNC || h == NULL || h->check == NULL) continue;
        tsr_extents ext;
        memset(&ext, 0, sizeof(ext));
        uint32_t info = 0;
        if (h->check(sim, p->w, &ext, &info) != 0) continue;     // a failing op touches nothing
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

// Boots the image, waits for HELLO, pushes descriptor d and waits for its answer (as launch_replay).
static int kr_launch(tsr_sim * sim, tmodel * m, const uint8_t * image, size_t image_n, const uint32_t d[8],
                     uint64_t max_cycles, tsr_trace2 ** got, char * why, size_t wn) {
    *got = NULL;
    m->on = 1;
    if (boot_image(sim, image, image_n, why, wn)) return -1;
    uint32_t lcq_rd = tsr_sim_csr_read(sim, TSR_CSR_LCQ_RD);
    int hello = 0, pushed = 0, answered = 0, idle_steps = 0;
    uint32_t seq = 0, ans_type = 0;
    const uint64_t t_start = tsr_sim_now(sim);
    uint64_t last_progress = 0;
    while (!answered) {
        const uint32_t ev = tsr_sim_advance(sim, STEP_CYCLES);
        const uint32_t wr = tsr_sim_csr_read(sim, TSR_CSR_LCQ_WR);
        for (uint32_t k = lcq_rd; k != wr; ++k) {
            const uint32_t r0 = tsr_sim_csr_read(sim, TSR_CSR_LCQ_REC(k % TSR_LCQ_DEPTH, 0));
            const uint32_t r1 = tsr_sim_csr_read(sim, TSR_CSR_LCQ_REC(k % TSR_LCQ_DEPTH, 1));
            const uint32_t type = TSR_GET(r0, TSR_LCQ_R0_TYPE);
            if (!hello) {
                if (type != TSR_LCQ_R0_TYPE_HELLO) {
                    seterr(why, wn, "the first record after boot is type 0x%02x, not HELLO (EX §3.4)", type);
                    return -1;
                }
                hello = 1;
            } else if (pushed && r1 == seq &&
                       (type == TSR_LCQ_R0_TYPE_DONE || type == TSR_LCQ_R0_TYPE_PONG || type == TSR_LCQ_R0_TYPE_TRAP)) {
                answered = 1;
                ans_type = type;
            }
        }
        lcq_rd = wr;
        tsr_sim_csr_write(sim, TSR_CSR_LCQ_RD, wr);
        tsr_sim_csr_write(sim, TSR_CSR_LOG_RD, tsr_sim_csr_read(sim, TSR_CSR_LOG_WR));   // log content is not compared
        if (hello && !pushed) {
            seq = tsr_sim_csr_read(sim, TSR_CSR_LQ_SEQ);
            for (uint32_t i = 0; i < 8; ++i) tsr_sim_csr_write(sim, TSR_CSR_LQ_W(i), d[i]);
            tsr_sim_csr_write(sim, TSR_CSR_LQ_PUSH, 1u);
            pushed = 1;
        }
        if (answered) break;
        const uint32_t rt = tsr_sim_csr_read(sim, TSR_CSR_RT_STATE);
        if (TSR_GET(rt, TSR_RT_STATE_PHASE) == TSR_RT_STATE_PHASE_TRAPPED) {
            seterr(why, wn, "the firmware trapped (RT_STATE 0x%08x, RT_INFO 0x%08x, pc 0x%08x)", rt,
                   tsr_sim_csr_read(sim, TSR_CSR_RT_INFO), tsr_sim_rv_pc(sim));
            return -1;
        }
        if (tsr_sim_now(sim) - t_start > max_cycles) {
            seterr(why, wn, "%s within %llu core cycles", hello ? "no answer" : "no HELLO", (unsigned long long)max_cycles);
            return -1;
        }
        const uint64_t progress = tsr_sim_rv_instret(sim) + wr + tsr_sim_csr_read(sim, TSR_CSR_LQ_TAKEN);
        const uint32_t st = tsr_sim_csr_read(sim, TSR_CSR_STATUS);
        if ((ev & TSR_SIM_EV_BUDGET) && progress == last_progress && (st & TSR_STATUS_QIDLE) && (st & TSR_STATUS_RV_SLEEP)) {
            if (++idle_steps >= 2) {
                seterr(why, wn, "stuck: %s, the hart asleep and the queues idle (STATUS 0x%08x)", hello ? "no answer" : "no HELLO", st);
                return -1;
            }
        } else {
            idle_steps = 0;
        }
        last_progress = progress;
    }
    tsr_trace2 * c = tsr_trace2_new();
    if (tsr_sim_capture_take(sim, c) != 0) {
        tsr_trace2_free(c);
        seterr(why, wn, "the launch was answered by a %s record and not captured",
               ans_type == TSR_LCQ_R0_TYPE_DONE ? "DONE" : ans_type == TSR_LCQ_R0_TYPE_TRAP ? "TRAP" : "PONG");
        return -1;
    }
    *got = c;
    return 0;
}

// a < b for 28-bit sequences (within half the space).
static int kr_seq_lt(uint32_t a, uint32_t b) { return a != b && ((b - a) & SEQ_MASK) < 0x08000000u; }

typedef struct kr_own {
    uint32_t n[TSR_NUM_UNITS];        // own records per unit
    uint32_t cap[TSR_NUM_UNITS];
    uint32_t * seq[TSR_NUM_UNITS];    // their captured relative sequences, in push order
    uint32_t * at[TSR_NUM_UNITS];     // their record index in the capture
    uint32_t n_p, cap_p;
    uint32_t * pidx;                  // captured relative push indices of every own ticketed record
} kr_own;

static void kr_own_free(kr_own * o) {
    for (uint32_t u = 0; u < TSR_NUM_UNITS; ++u) { free(o->seq[u]); free(o->at[u]); }
    free(o->pidx);
}

// Captured relative sequence crel of unit u -> trace-relative, counting the own records of u pushed
// before capture record `before` whose sequence is below crel.
static uint32_t kr_seq_map(const kr_own * o, uint32_t u, uint32_t crel, size_t before) {
    uint32_t k = 0;
    for (uint32_t j = 0; j < o->n[u]; ++j)
        if (o->at[u][j] < before && kr_seq_lt(o->seq[u][j], crel)) ++k;
    return (crel - k) & SEQ_MASK;
}
static uint32_t kr_ticket_map(const kr_own * o, uint32_t t, size_t before) {
    const uint32_t u = t >> 28;
    return u < TSR_NUM_UNITS ? (u << 28) | kr_seq_map(o, u, t & SEQ_MASK, before) : t;
}
static uint32_t kr_pidx_map(const kr_own * o, uint32_t p) {
    uint32_t k = 0;
    for (uint32_t j = 0; j < o->n_p; ++j)
        if (o->pidx[j] < p) ++k;
    return p - k;
}

// Removes bytes [lo, hi) of channel ch from a region list (regions split as needed).
static void kr_regions_cut(tsr_region ** rr, size_t * nn, uint32_t ch, uint64_t lo, uint64_t hi) {
    tsr_region * r = *rr;
    size_t n = *nn;
    tsr_region * out = (tsr_region *)xmalloc((2 * n + 1) * sizeof(tsr_region));
    size_t k = 0;
    for (size_t i = 0; i < n; ++i) {
        const uint64_t a = r[i].byte_off, b = r[i].byte_off + r[i].size;
        if (r[i].ch != ch || b <= lo || a >= hi) { out[k++] = r[i]; continue; }
        if (a < lo) {
            tsr_region x = { r[i].ch, a, lo - a, (uint8_t *)xmalloc((size_t)(lo - a)) };
            memcpy(x.data, r[i].data, (size_t)(lo - a));
            out[k++] = x;
        }
        if (b > hi) {
            tsr_region x = { r[i].ch, hi, b - hi, (uint8_t *)xmalloc((size_t)(b - hi)) };
            memcpy(x.data, r[i].data + (hi - a), (size_t)(b - hi));
            out[k++] = x;
        }
        free(r[i].data);
    }
    free(r);
    *rr = out;
    *nn = k;
}

// The ops view of a captured replay-kernel launch c of trace t (see the decisions above).
static tsr_trace2 * kr_ops_view(const tsr_trace2 * t, tsr_trace2 * c, uint32_t load_ch, uint64_t load_beat,
                                uint64_t load_lines, uint32_t buf_lo, uint32_t buf_hi) {
    tsr_trace2 * g = tsr_trace2_new();
    g->kind = TSR_TRACE2_KIND_OPS;
    g->cfg = c->cfg;
    kr_own own;
    memset(&own, 0, sizeof(own));
    uint8_t * is_own = (uint8_t *)xcalloc(c->n_push ? c->n_push : 1, 1);
    uint32_t * crel = (uint32_t *)xcalloc(c->n_push ? c->n_push : 1, sizeof(uint32_t));
    uint32_t * cpidx = (uint32_t *)xcalloc(c->n_push ? c->n_push : 1, sizeof(uint32_t));
    uint32_t cnt[TSR_NUM_UNITS] = { 0 }, pid = 0;
    size_t replayed = 0;
    int started = 0;                                               // the kernel's first chunk LD seen
    for (size_t i = 0; i < c->n_push; ++i) {
        const tsr_trace_push * p = &c->push[i];
        uint32_t u = TSR_UNIT_NONE;
        if (p->kind == TSR_TRACE_PUSH_OP) u = tsr_trace2_push_unit(&c->cfg, p);
        else if (p->kind == TSR_TRACE_PUSH_QCTRL && (p->w[0] & TSR_QCTRL_QCLEAR)) u = TSR_UNIT_SYNC;
        const int ticketed = u < TSR_NUM_UNITS;
        if (ticketed) { crel[i] = cnt[u]++; cpidx[i] = pid++; }
        int chunk_ld = 0;
        if (p->kind == TSR_TRACE_PUSH_OP && (p->w[0] & 0xFFu) == TSR_OP_LD) {
            const uint32_t ch = tsr_w0_sub(p->w[0]);
            const uint64_t src = p->w[2], lines = p->w[3], dst = p->w[1];
            chunk_ld = ch == load_ch && src >= load_beat && src + lines <= load_beat + load_lines && dst >= buf_lo &&
                       dst + lines <= buf_hi;
        }
        // Own: the runtime's records before the kernel's first chunk LD (LAUNCH_BEGIN, DB §8.4), the
        // chunk LDs, and everything after the last replayed record (the kernel's and the runtime's
        // FENCEs, LAUNCH_END).
        const int mine = !started || replayed >= t->n_push || chunk_ld;
        if (chunk_ld) started = 1;
        is_own[i] = (uint8_t)mine;
        if (!mine) { replayed++; continue; }
        if (!ticketed) continue;
        if (own.n[u] == own.cap[u]) {
            own.cap[u] = own.cap[u] ? 2 * own.cap[u] : 16;
            own.seq[u] = (uint32_t *)xrealloc(own.seq[u], own.cap[u] * sizeof(uint32_t));
            own.at[u] = (uint32_t *)xrealloc(own.at[u], own.cap[u] * sizeof(uint32_t));
        }
        own.seq[u][own.n[u]] = crel[i];
        own.at[u][own.n[u]] = (uint32_t)i;
        own.n[u]++;
        if (own.n_p == own.cap_p) {
            own.cap_p = own.cap_p ? 2 * own.cap_p : 16;
            own.pidx = (uint32_t *)xrealloc(own.pidx, own.cap_p * sizeof(uint32_t));
        }
        own.pidx[own.n_p++] = cpidx[i];
    }
    for (size_t i = 0; i < c->n_push; ++i) {
        if (is_own[i]) continue;
        tsr_trace_push p = c->push[i];
        if (p.kind == TSR_TRACE_PUSH_OP) {
            p.dep[0] = kr_ticket_map(&own, p.dep[0], i);
            p.dep[1] = kr_ticket_map(&own, p.dep[1], i);
        }
        tsr_trace2_add_push(g, &p);
    }
    g->n_units = c->n_units;
    for (uint32_t u = 0; u < TSR_NUM_UNITS && u < c->n_units; ++u) {
        tsr_trace_unit x = c->unit[u];
        x.count -= own.n[u];
        if (x.err_code != 0) {
            x.err_rel_seq = kr_seq_map(&own, u, x.err_rel_seq, c->n_push);
            x.err_rel_pidx = kr_pidx_map(&own, x.err_rel_pidx);
        }
        g->unit[u] = x;
    }
    g->done = c->done;
    if ((c->done.err_ticket_rel >> 28) < TSR_NUM_UNITS) {
        g->done.err_ticket_rel = kr_ticket_map(&own, c->done.err_ticket_rel, c->n_push);
        g->done.err_pidx_rel = kr_pidx_map(&own, c->done.err_pidx_rel);
    }
    // DRAM: the kernel reads push.bin, which is no part of the trace.
    g->mem = c->mem;
    memset(&c->mem, 0, sizeof(c->mem));
    kr_regions_cut(&g->mem.init, &g->mem.n_init, load_ch, load_beat * 64u, (load_beat + load_lines) * 64u);
    g->perf = c->perf;
    memcpy(g->sched, c->sched, sizeof(g->sched));
    g->rv_ratio = c->rv_ratio;
    free(is_own);
    free(crel);
    free(cpidx);
    kr_own_free(&own);
    return g;
}

// Kind ops through the replay kernel of o->replay_image: the run part of run_trace.
static int kernel_replay(tsr_sim * sim, tmodel * m, const tsr_trace2 * t, const tsr_trace2_replay_opts * o,
                         uint64_t max_cycles, tsr_trace2 ** got, char * why, size_t wn) {
    *got = NULL;
    char ferr[256];
    size_t n = 0;
    uint8_t * img = read_file(o->replay_image, &n, IMG_MAX_BYTES, ferr, sizeof(ferr));
    if (img == NULL) {
        seterr(why, wn, "replay image: %s", ferr);
        return -1;
    }
    img_info e;
    uint32_t index = 0;
    const char * kname = o->replay_kernel ? o->replay_kernel : "replay";
    if (img_parse(img, n, TCM_LIMIT, TCM_LIMIT, &e, ferr, sizeof(ferr))) {
        seterr(why, wn, "replay image %s: %s", o->replay_image, ferr);
        free(img);
        return -1;
    }
    if (kr_find_kernel(img, &e, kname, &index)) {
        seterr(why, wn, "replay image %s has no kernel \"%s\" in .tsr_kernels", o->replay_image, kname);
        free(img);
        return -1;
    }
    // push.bin in DRAM.
    uint32_t load_ch = t->load_ch;
    uint64_t load_off = t->load_offset;
    if (load_ch == TSR_TRACE2_LOAD_AUTO && choose_load(t, &load_ch, &load_off)) {
        seterr(why, wn, "no DRAM place for push.bin outside the trace's regions");
        free(img);
        return -1;
    }
    const uint64_t bytes = (uint64_t)t->n_push * TSR_TRACE_PUSH_BYTES;
    const uint64_t load_lines = (bytes + 63u) / 64u;
    if (load_off % 64u != 0 || load_off + load_lines * 64u > tsr_sim_config(sim)->ch_bytes || load_off / 64u > 0x0FFFFFFFu) {
        seterr(why, wn, "ops.load (ch %u, offset 0x%llx) does not fit the replay configuration", load_ch,
               (unsigned long long)load_off);
        free(img);
        return -1;
    }
    if (bytes > 0) {
        uint8_t * buf = (uint8_t *)xcalloc((size_t)(load_lines * 64u), 1);
        for (size_t i = 0; i < t->n_push; ++i) {
            const tsr_trace_push * p = &t->push[i];
            uint8_t * q = buf + i * TSR_TRACE_PUSH_BYTES;
            for (int k = 0; k < 8; ++k) wr32(q + 4 * k, p->w[k]);
            wr32(q + 32, p->dep[0]);
            wr32(q + 36, p->dep[1]);
            wr32(q + 40, p->push);
            wr32(q + 44, p->kind);
        }
        tsr_sim_dram_write(sim, load_ch, load_off, buf, load_lines * 64u);
        free(buf);
    }
    // SPM buffer outside the trace's footprint.
    const uint32_t chunk = t->n_push <= TSR_REPLAY_CHUNK_MAX ? (uint32_t)((t->n_push + 3u) & ~3ull) : TSR_REPLAY_CHUNK_MAX;
    const uint32_t chunk_lines = chunk * TSR_TRACE_PUSH_BYTES / 64u;
    const uint32_t nbuf = t->n_push > chunk ? 2u : 1u;
    uint32_t buf_line = 0;
    if (chunk > 0 && kr_spm_buffer(sim, t, nbuf * chunk_lines, &buf_line)) {
        seterr(why, wn, "no %u free SPM lines for the replay kernel's buffer outside the trace's SPM operands",
               nbuf * chunk_lines);
        free(img);
        return -1;
    }
    if (o->verbose)
        fprintf(stderr, "replay: kernel \"%s\" (index %u), push.bin at ch %u 0x%llx, %zu records, buffer lines [%u, %u), "
                        "%u records per chunk\n", kname, index, load_ch, (unsigned long long)load_off, t->n_push, buf_line,
                buf_line + nbuf * chunk_lines, chunk);
    const uint32_t d[8] = { TSR_PUT(TSR_DESC_TYPE, TSR_DESC_TYPE_LAUNCH) | TSR_PUT(TSR_DESC_KERNEL, index),
                            (load_ch << 28) | (uint32_t)(load_off / 64u),
                            (uint32_t)t->n_push,
                            (o->replay_flags & TSR_REPLAY_F_SERIAL) | TSR_REPLAY_F_BUF,
                            buf_line, chunk ? chunk : 4u, 0u, 0u };
    tsr_trace2 * c = NULL;
    const int rc = kr_launch(sim, m, img, n, d, max_cycles, &c, why, wn);
    free(img);
    if (rc) return rc;
    if (o->verbose && getenv("TSR_TRACE_KR_DUMP") != NULL)   // debugging aid: the captured launch
        for (size_t i = 0; i < c->n_push; ++i)
            fprintf(stderr, "replay: captured %zu: kind %u w %08x %08x %08x %08x %08x %08x %08x %08x deps %08x %08x push %08x\n", i,
                    c->push[i].kind, c->push[i].w[0], c->push[i].w[1], c->push[i].w[2], c->push[i].w[3], c->push[i].w[4],
                    c->push[i].w[5], c->push[i].w[6], c->push[i].w[7], c->push[i].dep[0], c->push[i].dep[1], c->push[i].push);
    *got = kr_ops_view(t, c, load_ch, load_off / 64u, load_lines, buf_line, buf_line + nbuf * chunk_lines);
    tsr_trace2_free(c);
    return 0;
}

static int run_trace(const tsr_trace2 * t, const tsr_trace2_replay_opts * o, tsr_trace2_result * r, tsr_sim ** sim_out,
                     tsr_trace2 ** got) {
    char why[512];
    *sim_out = NULL;
    *got = NULL;
    tmodel * m = NULL;
    tsr_sim * sim = replay_sim(t, o, &m, why, sizeof(why));
    if (sim == NULL) {
        rfailf(r, o->verbose, TSR_TRACE2_CMP_SETUP, "%s", why);
        return 2;
    }
    const uint64_t budget = o->max_cycles ? o->max_cycles : (1ull << 40);
    const int rc = t->kind == TSR_TRACE2_KIND_LAUNCH ? launch_replay(sim, m, t, budget, got, why, sizeof(why))
                   : o->replay_image != NULL         ? kernel_replay(sim, m, t, o, budget, got, why, sizeof(why))   // P2.5
                                                     : ops_replay(sim, m, t, budget, got, why, sizeof(why));
    if (rc != 0) {
        rfailf(r, o->verbose, TSR_TRACE2_CMP_SETUP, "%s", why);
        tsr_sim_destroy(sim);
        return 2;
    }
    r->core_cycles = (*got)->perf.core_cycles;
    r->rv_instret = (*got)->perf.rv_instret;
    *sim_out = sim;
    return 0;
}

int tsr_trace2_replay_sim_ex(const tsr_trace2 * t, const tsr_trace2_replay_opts * o, tsr_trace2_result * r) {
    static const tsr_trace2_replay_opts none;
    if (o == NULL) o = &none;
    memset(r, 0, sizeof(*r));
    r->ok = 1;
    char why[600];
    if (tsr_trace2_validate(t, why, sizeof(why))) {
        rfailf(r, o->verbose, TSR_TRACE2_CMP_SETUP, "invalid trace: %s", why);
        return 2;
    }
    tsr_sim * sim = NULL;
    tsr_trace2 * got = NULL;
    const int rc = run_trace(t, o, r, &sim, &got);
    if (rc != 0) return rc;
    compare(t, got, sim, o->verbose, r);
#if defined(TSR_SIM_HAVE_RACE) && TSR_SIM_HAVE_RACE
    {   // P2.5: the race detector's verdict for the caller (with TSR_SIM_RACE=report)
        const tsr_race_report * rr = NULL;
        const size_t nr = tsr_sim_race_reports(sim, &rr);
        r->race_reports = (uint32_t)nr;
        for (size_t i = 0; i < nr; ++i) r->race_stale += rr[i].kind == TSR_RACE_SPM_STALE;
    }
#endif
    if (r->ok)
        snprintf(r->msg, sizeof(r->msg), "ok: %s, %zu pushes, %zu init / %zu expect regions, %llu core cycles%s",
                 t->kind == TSR_TRACE2_KIND_LAUNCH ? "launch" : "ops", t->n_push, t->mem.n_init, t->mem.n_expect,
                 (unsigned long long)r->core_cycles, r->perf_differs ? " (perf differs: informational)" : "");
    tsr_trace2_free(got);
    tsr_sim_destroy(sim);
    return r->ok ? 0 : 1;
}

int tsr_trace2_replay_sim(const tsr_trace2 * t, const tsr_config * cfg_or_null, const char * sched_or_null,
                          tsr_trace2_result * r) {
    tsr_trace2_replay_opts o;
    memset(&o, 0, sizeof(o));
    o.cfg = cfg_or_null;
    o.sched = sched_or_null;
    return tsr_trace2_replay_sim_ex(t, &o, r);
}

int tsr_trace2_run_ops(tsr_trace2 * t, const tsr_trace2_replay_opts * o, tsr_trace2_result * r) {
    static const tsr_trace2_replay_opts none;
    if (o == NULL) o = &none;
    memset(r, 0, sizeof(*r));
    r->ok = 1;
    char why[600];
    if (t->kind != TSR_TRACE2_KIND_OPS) {
        rfailf(r, o->verbose, TSR_TRACE2_CMP_SETUP, "tsr_trace2_run_ops needs a kind-ops trace");
        return 2;
    }
    // Validate what the run needs (done / units / expect are about to be filled).
    tsr_trace2 probe = *t;
    memset(&probe.mem.expect, 0, sizeof(probe.mem.expect));
    probe.mem.n_expect = 0;
    memset(probe.unit, 0, sizeof(probe.unit));
    probe.n_units = TSR_NUM_UNITS;
    if (tsr_trace2_validate(&probe, why, sizeof(why))) {
        rfailf(r, o->verbose, TSR_TRACE2_CMP_SETUP, "invalid trace: %s", why);
        return 2;
    }
    tsr_sim * sim = NULL;
    tsr_trace2 * got = NULL;
    tsr_trace2_replay_opts tile_only = *o;   // P2.5: the generator's runs are always tile-only
    tile_only.replay_image = NULL;
    const int rc = run_trace(t, &tile_only, r, &sim, &got);
    if (rc != 0) return rc;
    if (!o->init_from_reads) {
        span * is = regions_spans(t->mem.init, t->mem.n_init);
        const size_t ni = span_normalize(is, t->mem.n_init);
        for (size_t i = 0; i < got->mem.n_init; ++i) {
            const tsr_region * x = &got->mem.init[i];
            if (!span_covered(is, ni, x->ch, x->byte_off, x->byte_off + x->size)) {
                rfailf(r, o->verbose, TSR_TRACE2_CMP_READS, "the run read ch %u bytes [0x%llx, 0x%llx) outside the init regions",
                       x->ch, (unsigned long long)x->byte_off, (unsigned long long)(x->byte_off + x->size));
                break;
            }
        }
        free(is);
    }
    if (r->ok) {
        t->done = got->done;
        t->n_units = got->n_units;
        memcpy(t->unit, got->unit, sizeof(t->unit));
        for (size_t i = 0; i < t->mem.n_expect; ++i) free(t->mem.expect[i].data);
        free(t->mem.expect);
        t->mem.expect = got->mem.expect;
        t->mem.n_expect = got->mem.n_expect;
        got->mem.expect = NULL;
        got->mem.n_expect = 0;
        if (o->init_from_reads) {
            for (size_t i = 0; i < t->mem.n_init; ++i) free(t->mem.init[i].data);
            free(t->mem.init);
            t->mem.init = got->mem.init;
            t->mem.n_init = got->mem.n_init;
            got->mem.init = NULL;
            got->mem.n_init = 0;
        }
        t->perf = got->perf;
        t->cfg.ctl_id = got->cfg.ctl_id;
        t->cfg.cfg3 = got->cfg.cfg3;
        memcpy(t->sched, got->sched, sizeof(t->sched));
        t->rv_ratio = got->rv_ratio;
        snprintf(r->msg, sizeof(r->msg), "ok: %zu pushes, status %u, %llu core cycles", t->n_push, t->done.status,
                 (unsigned long long)r->core_cycles);
    }
    tsr_trace2_free(got);
    tsr_sim_destroy(sim);
    return r->ok ? 0 : 1;
}
