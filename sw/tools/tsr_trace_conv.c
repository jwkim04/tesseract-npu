// tsr_trace_conv.c — tsr-trace-conv (docs/v2/06-software-verification.md "SW" §5.6 row
// traces/model, V2-SW-6; §9.5): converts v1 trace directories (cmd.bin programs, format version 1)
// into trace v2 kind-ops traces, and optionally replays each converted trace on the simulator.
//
//   tsr-trace-conv [-q] [--replay] [--sched S]... [--limit N] [--pick P1,P2,...] -o OUT IN...
//
//   IN            a v1 trace directory or a set of them (searched recursively for meta.json); a
//                 trace at IN/<rel> is written to OUT/<rel> (a trace given directly: OUT/<its name>)
//   --pick        convert only the traces whose <rel> is listed (comma-separated)
//   --limit N     convert only the first N traces (sorted)
//   --replay      read every converted trace back and replay it (tsr_trace2_replay_sim_ex), once
//                 per --sched (default: the simulator's default schedule)
//   -q            print only failures and the summary
// Summary: tsr-trace-conv: <n> converted, <bytes> written, <s> s; <p> replays passed, <f> failed,
//          <e> errors. Exit status 0, 1 (a replay mismatch), 2 (a trace not convertible, or usage).
//
// Conversion (SW §5.6): each v1 instruction up to END becomes one push record (kind 0, all eight
// words, ISSUE_PUSH WMASK 0xFF) that depends on the previous push record (dep0 = its relative
// ticket, tsr_trace2_ticket_of; dep1 none): a serial chain, which with I1 keeps v1's sequential
// semantics. END and everything after it are dropped; init and expect are kept as they are
// (including v1's command-buffer bytes in init, which no v2 op reads). Only what v2 encodes the
// same way is accepted: NOP, LD, ST, QACT, MVM, FENCE with every v1 register field (w0[31:16]) 0
// and a v1 run that ended DONE without an error; anything else (LI / ADDI, register-relative
// operands, v1's VU encodings, error endings) is refused: those sets are regenerated with
// tsr-trace-gen v2 instead. The v2 DONE is OK (RET 0, no failing ticket), tickets.bin counts the
// pushes per unit with no error record, ops.load is chosen outside every region, sched "fixed",
// no perf (v1's op-level cycle counts are not comparable). config takes v1's elaboration
// parameters (NIREG and LOOP_DEPTH dropped) and CTL_ID, CFG3, Q_DEPTH and the TCM sizes of a
// simulator built from them.
//
// The v1 reader below is v1-m4:sw/sim/tsr_trace.c (tsr_trace_read and its JSON parser), kept as
// it was apart from the names (v1_trace, v1_config) and the parts a reader does not need.
#define _GNU_SOURCE

#include "tsr_trace.h"

#include <dirent.h>
#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

// ---------------------------------------------------------------------------
// v1 trace (v1-m4:sw/sim/include/tsr_trace.h, tsr_isa.h)
// ---------------------------------------------------------------------------
#define TRACE_FORMAT     "tesseract-trace"
#define TRACE_VERSION    1u
#define CMD_ADDR_SANE    (1ull << 50)
#define TSR_TRACE_MAX_CH 256u
#define V1_STATUS_DONE   (1u << 1)
#define V1_OP_END        0x01u

typedef struct v1_config {
    uint32_t num_ch, beat_bits, lanes, t_max, k_max, spm_bytes, vl, nireg, nsreg, loop_depth, rd_outstanding;
    uint64_t ch_bytes;
    uint32_t addr_beat_w, caps;
} v1_config;

typedef struct v1_perf {
    int      valid;
    uint64_t cycles, retired, mvm_cycles;
    uint64_t ch_rd_beats[TSR_TRACE_MAX_CH];
} v1_perf;

typedef struct v1_trace {
    v1_config   cfg;
    uint64_t    cmd_addr_beat;
    uint32_t    cmd_count;
    uint32_t    args[8];
    uint32_t    status, err_code, err_pc;
    tsr_capture cap;
    uint32_t    err_info;
    int         has_err_info;
    v1_perf     perf;
    uint8_t *   cmd;
} v1_trace;

static void v1_trace_free(v1_trace * tr);

static void terr(const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fputs("tsr-trace-conv: v1 reader: ", stderr);
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    va_end(ap);
}

static void * xmalloc(size_t n) {
    void * p = malloc(n ? n : 1);
    if (p == NULL) {
        fprintf(stderr, "tsr-trace-conv: out of memory (%zu bytes)\n", n);
        abort();
    }
    return p;
}
static void * xrealloc(void * p, size_t n) {
    void * q = realloc(p, n ? n : 1);
    if (q == NULL) {
        fprintf(stderr, "tsr-trace-conv: out of memory (%zu bytes)\n", n);
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

// ---------------------------------------------------------------------------
// File helpers

static uint8_t * read_file(const char * path, size_t * size) {
    FILE * f = fopen(path, "rb");
    if (f == NULL) {
        terr("cannot open %s: %s", path, strerror(errno));
        return NULL;
    }
    struct stat st;
    if (fstat(fileno(f), &st) != 0 || !S_ISREG(st.st_mode)) {
        terr("%s is not a regular file", path);
        fclose(f);
        return NULL;
    }
    const size_t n = (size_t)st.st_size;
    uint8_t * buf = (uint8_t *)xmalloc(n + 1);
    if (n && fread(buf, 1, n, f) != n) {
        terr("read error on %s", path);
        free(buf);
        fclose(f);
        return NULL;
    }
    fclose(f);
    buf[n] = 0;
    *size = n;
    return buf;
}


// ---------------------------------------------------------------------------
// Minimal JSON DOM
// ---------------------------------------------------------------------------
typedef enum { J_NULL, J_BOOL, J_NUM, J_STR, J_ARR, J_OBJ } jtype;

typedef struct jval {
    jtype         t;
    int           b;          // J_BOOL
    int           is_uint;    // J_NUM: plain non-negative integer that fits in uint64
    uint64_t      u;
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
    // at '"'
    ps->p++;
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

static int jvalue(jparser * ps, jval * v, int depth);

static int jnumber(jparser * ps, jval * v) {
    const char * st = ps->p;
    int neg = 0, frac = 0;
    if (ps->p < ps->end && *ps->p == '-') { neg = 1; ps->p++; }
    if (ps->p >= ps->end || *ps->p < '0' || *ps->p > '9') return jfail(ps, "bad number");
    uint64_t u = 0;
    int overflow = 0;
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
    (void)st;
    v->t = J_NUM;
    v->is_uint = !neg && !frac && !overflow;
    v->u = v->is_uint ? u : 0;
    return 0;
}

static int jvalue(jparser * ps, jval * v, int depth) {
    memset(v, 0, sizeof(*v));
    if (depth > 64) return jfail(ps, "nesting too deep");
    jws(ps);
    if (ps->p >= ps->end) return jfail(ps, "unexpected end of input");
    const char c = *ps->p;
    if (c == '{') {
        v->t = J_OBJ;
        ps->p++;
        jws(ps);
        if (ps->p < ps->end && *ps->p == '}') { ps->p++; return 0; }
        size_t cap = 0;
        for (;;) {
            jws(ps);
            if (ps->p >= ps->end || *ps->p != '"') return jfail(ps, "expected object key");
            char * key;
            if (jstring(ps, &key)) return -1;
            jws(ps);
            if (ps->p >= ps->end || *ps->p != ':') { free(key); return jfail(ps, "expected ':'"); }
            ps->p++;
            if (v->n == cap) {
                cap = cap ? 2 * cap : 8;
                v->items = (jval *)xrealloc(v->items, cap * sizeof(jval));
                v->keys = (char **)xrealloc(v->keys, cap * sizeof(char *));
            }
            v->keys[v->n] = key;
            const int rc = jvalue(ps, &v->items[v->n], depth + 1);
            v->n++;
            if (rc) return -1;
            jws(ps);
            if (ps->p < ps->end && *ps->p == ',') { ps->p++; continue; }
            if (ps->p < ps->end && *ps->p == '}') { ps->p++; return 0; }
            return jfail(ps, "expected ',' or '}'");
        }
    }
    if (c == '[') {
        v->t = J_ARR;
        ps->p++;
        jws(ps);
        if (ps->p < ps->end && *ps->p == ']') { ps->p++; return 0; }
        size_t cap = 0;
        for (;;) {
            if (v->n == cap) {
                cap = cap ? 2 * cap : 8;
                v->items = (jval *)xrealloc(v->items, cap * sizeof(jval));
            }
            const int rc = jvalue(ps, &v->items[v->n], depth + 1);
            v->n++;
            if (rc) return -1;
            jws(ps);
            if (ps->p < ps->end && *ps->p == ',') { ps->p++; continue; }
            if (ps->p < ps->end && *ps->p == ']') { ps->p++; return 0; }
            return jfail(ps, "expected ',' or ']'");
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

static int jparse(const char * text, size_t len, jval * root, const char * what) {
    jparser ps = { text, text + len, text, NULL };
    if (jvalue(&ps, root, 0) == 0) {
        jws(&ps);
        if (ps.p == ps.end) return 0;
        jfail(&ps, "trailing characters after JSON value");
    }
    terr("%s: JSON error at byte %ld: %s", what, (long)(ps.p - ps.begin), ps.err ? ps.err : "?");
    jfree(root);
    return -1;
}

static const jval * jget(const jval * obj, const char * key) {
    if (obj == NULL || obj->t != J_OBJ) return NULL;
    for (size_t i = 0; i < obj->n; ++i) {
        if (strcmp(obj->keys[i], key) == 0) return &obj->items[i];
    }
    return NULL;
}

// Required/optional unsigned integer fields with range check.
static int jget_u64(const jval * obj, const char * key, uint64_t max, uint64_t * out, int required,
                    const char * ctx) {
    const jval * v = jget(obj, key);
    if (v == NULL) {
        if (required) terr("%s: missing field \"%s\"", ctx, key);
        return required ? -1 : 1;
    }
    if (v->t != J_NUM || !v->is_uint || v->u > max) {
        terr("%s: field \"%s\" must be an unsigned integer <= %llu", ctx, key, (unsigned long long)max);
        return -1;
    }
    *out = v->u;
    return 0;
}
static int jget_u32(const jval * obj, const char * key, uint32_t * out, int required, const char * ctx) {
    uint64_t v = 0;
    const int rc = jget_u64(obj, key, UINT32_MAX, &v, required, ctx);
    if (rc == 0) *out = (uint32_t)v;
    return rc;
}

// ---------------------------------------------------------------------------
// Region helpers
// ---------------------------------------------------------------------------
typedef struct { uint32_t ch; uint64_t lo, hi; } span;   // bytes [lo, hi)

static int span_cmp(const void * a, const void * b) {
    const span * x = (const span *)a;
    const span * y = (const span *)b;
    if (x->ch != y->ch) return x->ch < y->ch ? -1 : 1;
    return (x->lo > y->lo) - (x->lo < y->lo);
}


static span * regions_to_spans(const tsr_region * r, size_t n) {
    span * s = (span *)xmalloc((n ? n : 1) * sizeof(span));
    for (size_t i = 0; i < n; ++i) {
        s[i].ch = r[i].ch;
        s[i].lo = r[i].byte_off;
        s[i].hi = r[i].byte_off + r[i].size;
    }
    return s;
}


static void region_name(char * buf, size_t n, const char * sub, uint32_t ch, uint64_t off) {
    snprintf(buf, n, "%s/ch%u_%010llx.bin", sub, ch, (unsigned long long)off);
}

// Checks one region list: inside the channels, aligned, non-empty, non-overlapping.
static int check_regions(const v1_config * cfg, const tsr_region * r, size_t n, const char * what) {
    for (size_t i = 0; i < n; ++i) {
        if (r[i].ch >= cfg->num_ch || r[i].size == 0 || r[i].byte_off % 64 != 0 || r[i].size % 64 != 0 ||
            r[i].byte_off > cfg->ch_bytes || r[i].size > cfg->ch_bytes - r[i].byte_off) {
            terr("%s region %zu (ch %u, offset 0x%llx, size %llu) is invalid: must be non-empty, 64-byte "
                 "aligned and inside a channel (num_ch %u, ch_bytes %llu)", what, i, r[i].ch,
                 (unsigned long long)r[i].byte_off, (unsigned long long)r[i].size, cfg->num_ch,
                 (unsigned long long)cfg->ch_bytes);
            return -1;
        }
    }
    span * s = regions_to_spans(r, n);
    qsort(s, n, sizeof(span), span_cmp);
    for (size_t i = 1; i < n; ++i) {
        if (s[i].ch == s[i - 1].ch && s[i].lo < s[i - 1].hi) {
            terr("%s regions overlap (ch %u, offset 0x%llx)", what, s[i].ch, (unsigned long long)s[i].lo);
            free(s);
            return -1;
        }
    }
    free(s);
    return 0;
}

// cmd.bin bytes: init bytes where init covers the command buffer, zero elsewhere.
static uint8_t * derive_cmd(const v1_trace * tr, size_t * size) {
    const uint64_t bytes = (uint64_t)tr->cmd_count * TSR_INSN_BYTES;
    *size = (size_t)bytes;
    uint8_t * cmd = (uint8_t *)calloc(bytes ? bytes : 1, 1);
    if (cmd == NULL) return NULL;
    if (tr->cmd_addr_beat >= CMD_ADDR_SANE) return cmd;   // beyond any channel: cannot overlap init
    const uint64_t c_lo = tr->cmd_addr_beat * TSR_BEAT_BYTES;
    const uint64_t c_hi = c_lo + bytes;
    for (size_t i = 0; i < tr->cap.n_init; ++i) {
        const tsr_region * r = &tr->cap.init[i];
        if (r->ch != 0) continue;
        const uint64_t lo = r->byte_off > c_lo ? r->byte_off : c_lo;
        const uint64_t hi = (r->byte_off + r->size) < c_hi ? (r->byte_off + r->size) : c_hi;
        if (lo < hi) memcpy(cmd + (lo - c_lo), r->data + (lo - r->byte_off), (size_t)(hi - lo));
    }
    return cmd;
}

// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Read
// ---------------------------------------------------------------------------
static int safe_rel_path(const char * p) {
    if (p[0] == '\0' || p[0] == '/') return 0;
    // reject any ".." path component
    const char * s = p;
    while (*s) {
        const char * e = strchr(s, '/');
        const size_t L = e ? (size_t)(e - s) : strlen(s);
        if (L == 2 && s[0] == '.' && s[1] == '.') return 0;
        if (!e) break;
        s = e + 1;
    }
    return 1;
}

static int read_region_list(const char * dir, const jval * root, const char * key, const char * sub,
                            const v1_config * cfg, tsr_region ** out, size_t * n_out) {
    *out = NULL;
    *n_out = 0;
    const jval * arr = jget(root, key);
    if (arr == NULL || arr->t != J_ARR) {
        terr("%s/meta.json: \"%s\" must be an array", dir, key);
        return -1;
    }
    tsr_region * r = (tsr_region *)calloc(arr->n ? arr->n : 1, sizeof(tsr_region));
    if (r == NULL) return -1;
    *out = r;
    for (size_t i = 0; i < arr->n; ++i) {
        const jval * e = &arr->items[i];
        char ctx[128];
        snprintf(ctx, sizeof(ctx), "%s/meta.json %s[%zu]", dir, key, i);
        uint32_t ch = 0;
        uint64_t off = 0, size = 0;
        if (e->t != J_OBJ || jget_u32(e, "ch", &ch, 1, ctx) || jget_u64(e, "offset", UINT64_MAX, &off, 1, ctx) ||
            jget_u64(e, "size", UINT64_MAX, &size, 1, ctx)) {
            if (e->t != J_OBJ) terr("%s: must be an object", ctx);
            return -1;
        }
        r[i].ch = ch;
        r[i].byte_off = off;
        r[i].size = size;
        *n_out = i + 1;
        if (check_regions(cfg, &r[i], 1, ctx)) return -1;
        char name[96];
        const char * file;
        const jval * fv = jget(e, "file");
        if (fv != NULL) {
            if (fv->t != J_STR || !safe_rel_path(fv->s)) {
                terr("%s: \"file\" must be a relative path inside the trace directory", ctx);
                return -1;
            }
            file = fv->s;
        } else {
            region_name(name, sizeof(name), sub, ch, off);
            file = name;
        }
        char * p = path_join(dir, file);
        size_t got = 0;
        r[i].data = read_file(p, &got);
        if (r[i].data == NULL) { free(p); return -1; }
        if (got != size) {
            terr("%s: size %zu differs from \"size\" %llu", p, got, (unsigned long long)size);
            free(p);
            return -1;
        }
        free(p);
    }
    return check_regions(cfg, r, arr->n, key);
}

static int v1_trace_read(const char * dir, v1_trace * tr) {
    memset(tr, 0, sizeof(*tr));
    char * mpath = path_join(dir, "meta.json");
    size_t mlen = 0;
    uint8_t * text = read_file(mpath, &mlen);
    free(mpath);
    if (text == NULL) return -1;
    jval root;
    char ctx[600];
    snprintf(ctx, sizeof(ctx), "%s/meta.json", dir);
    if (jparse((const char *)text, mlen, &root, ctx)) {
        free(text);
        return -1;
    }
    free(text);
    int rc = -1;
    const jval * fmt = jget(&root, "format");
    uint32_t version = 0;
    if (root.t != J_OBJ || fmt == NULL || fmt->t != J_STR || strcmp(fmt->s, TRACE_FORMAT) != 0) {
        terr("%s: not a %s (missing or wrong \"format\")", ctx, TRACE_FORMAT);
        goto out;
    }
    if (jget_u32(&root, "version", &version, 1, ctx)) goto out;
    if (version != TRACE_VERSION) {
        terr("%s: unsupported version %u (expected %u)", ctx, version, TRACE_VERSION);
        goto out;
    }
    {
        const jval * c = jget(&root, "config");
        if (c == NULL || c->t != J_OBJ) {
            terr("%s: missing \"config\" object", ctx);
            goto out;
        }
        v1_config * k = &tr->cfg;
        if (jget_u32(c, "num_ch", &k->num_ch, 1, ctx) || jget_u32(c, "beat_bits", &k->beat_bits, 1, ctx) ||
            jget_u32(c, "lanes", &k->lanes, 1, ctx) || jget_u32(c, "t_max", &k->t_max, 1, ctx) ||
            jget_u32(c, "k_max", &k->k_max, 1, ctx) || jget_u32(c, "spm_bytes", &k->spm_bytes, 1, ctx) ||
            jget_u32(c, "vl", &k->vl, 1, ctx) || jget_u32(c, "nireg", &k->nireg, 1, ctx) ||
            jget_u32(c, "nsreg", &k->nsreg, 1, ctx) || jget_u32(c, "loop_depth", &k->loop_depth, 1, ctx) ||
            jget_u32(c, "rd_outstanding", &k->rd_outstanding, 1, ctx) ||
            jget_u64(c, "ch_bytes", UINT64_MAX, &k->ch_bytes, 1, ctx) ||
            jget_u32(c, "addr_beat_w", &k->addr_beat_w, 1, ctx) || jget_u32(c, "caps", &k->caps, 1, ctx)) {
            goto out;
        }
        if (k->num_ch < 1 || k->num_ch > TSR_TRACE_MAX_CH - 1) {
            terr("%s: num_ch %u out of range", ctx, k->num_ch);
            goto out;
        }
    }
    if (jget_u64(&root, "cmd_addr_beat", UINT64_MAX, &tr->cmd_addr_beat, 1, ctx) ||
        jget_u32(&root, "cmd_count", &tr->cmd_count, 1, ctx) || jget_u32(&root, "status", &tr->status, 1, ctx) ||
        jget_u32(&root, "err_code", &tr->err_code, 1, ctx) || jget_u32(&root, "err_pc", &tr->err_pc, 1, ctx)) {
        goto out;
    }
    {
        const int r = jget_u32(&root, "err_info", &tr->err_info, 0, ctx);
        if (r < 0) goto out;
        tr->has_err_info = (r == 0);
    }
    {
        const jval * a = jget(&root, "args");
        if (a == NULL || a->t != J_ARR || a->n != 8) {
            terr("%s: \"args\" must be an array of 8 integers", ctx);
            goto out;
        }
        for (int i = 0; i < 8; ++i) {
            if (a->items[i].t != J_NUM || !a->items[i].is_uint || a->items[i].u > UINT32_MAX) {
                terr("%s: args[%d] must be a 32-bit unsigned integer", ctx, i);
                goto out;
            }
            tr->args[i] = (uint32_t)a->items[i].u;
        }
    }
    if (read_region_list(dir, &root, "init", "init", &tr->cfg, &tr->cap.init, &tr->cap.n_init) ||
        read_region_list(dir, &root, "expect", "expect", &tr->cfg, &tr->cap.expect, &tr->cap.n_expect)) {
        goto out;
    }
    {
        const jval * p = jget(&root, "perf");
        if (p != NULL) {
            const jval * rb = jget(p, "ch_rd_beats");
            if (p->t != J_OBJ || jget_u64(p, "cycles", UINT64_MAX, &tr->perf.cycles, 1, ctx) ||
                jget_u64(p, "retired", UINT64_MAX, &tr->perf.retired, 1, ctx) ||
                jget_u64(p, "mvm_cycles", UINT64_MAX, &tr->perf.mvm_cycles, 1, ctx) || rb == NULL ||
                rb->t != J_ARR || rb->n != tr->cfg.num_ch) {
                terr("%s: malformed \"perf\" (needs cycles, retired, mvm_cycles, ch_rd_beats[num_ch])", ctx);
                goto out;
            }
            for (size_t i = 0; i < rb->n; ++i) {
                if (rb->items[i].t != J_NUM || !rb->items[i].is_uint) {
                    terr("%s: perf.ch_rd_beats[%zu] must be an unsigned integer", ctx, i);
                    goto out;
                }
                tr->perf.ch_rd_beats[i] = rb->items[i].u;
            }
            tr->perf.valid = 1;
        }
    }
    {
        char * cpath = path_join(dir, "cmd.bin");
        size_t got = 0;
        tr->cmd = read_file(cpath, &got);
        if (tr->cmd == NULL) { free(cpath); goto out; }
        if ((uint64_t)got != (uint64_t)tr->cmd_count * TSR_INSN_BYTES) {
            terr("%s: size %zu, expected cmd_count*32 = %llu", cpath, got,
                 (unsigned long long)tr->cmd_count * TSR_INSN_BYTES);
            free(cpath);
            goto out;
        }
        // cmd.bin must agree with the init regions it overlaps.
        size_t dsize = 0;
        uint8_t * derived = derive_cmd(tr, &dsize);
        if (derived == NULL) { free(cpath); goto out; }
        int bad = 0;
        if (tr->cmd_addr_beat < CMD_ADDR_SANE) {
            const uint64_t c_lo = tr->cmd_addr_beat * TSR_BEAT_BYTES;
            for (size_t i = 0; i < tr->cap.n_init && !bad; ++i) {
                const tsr_region * r = &tr->cap.init[i];
                if (r->ch != 0) continue;
                const uint64_t lo = r->byte_off > c_lo ? r->byte_off : c_lo;
                const uint64_t hi0 = r->byte_off + r->size, hi1 = c_lo + dsize;
                const uint64_t hi = hi0 < hi1 ? hi0 : hi1;
                if (lo < hi && memcmp(tr->cmd + (lo - c_lo), derived + (lo - c_lo), (size_t)(hi - lo)) != 0) bad = 1;
            }
        }
        free(derived);
        if (bad) {
            terr("%s: content differs from the init regions that cover the command buffer", cpath);
            free(cpath);
            goto out;
        }
        free(cpath);
    }
    rc = 0;
out:
    jfree(&root);
    if (rc) v1_trace_free(tr);
    return rc;
}

static void v1_trace_free(v1_trace * tr) {
    if (tr == NULL) return;
    tsr_capture_free(&tr->cap);
    free(tr->cmd);
    memset(tr, 0, sizeof(*tr));
}

// ---------------------------------------------------------------------------
// Replay
// ---------------------------------------------------------------------------
// Conversion (SW §5.6)
// ---------------------------------------------------------------------------
// The v2 config of a v1 one, with CTL_ID / CFG3 / Q_DEPTH / TCM sizes from a simulator (cached).
static int v2_config(const v1_config * v, tsr_trace2_config * out, char * why, size_t wn) {
    static tsr_config last;
    static tsr_trace2_config last_out;
    static int have = 0;
    tsr_config c;
    memset(&c, 0, sizeof(c));
    c.num_ch = v->num_ch; c.beat_bits = v->beat_bits; c.lanes = v->lanes; c.t_max = v->t_max; c.k_max = v->k_max;
    c.spm_bytes = v->spm_bytes; c.vl = v->vl; c.nsreg = v->nsreg; c.rd_outstanding = v->rd_outstanding;
    c.ch_bytes = v->ch_bytes; c.addr_beat_w = v->addr_beat_w; c.caps = v->caps;
    if (have && memcmp(&c, &last, sizeof(c)) == 0) {
        *out = last_out;
        return 0;
    }
    tsr_sim * sim = tsr_sim_create(&c, NULL);
    if (sim == NULL) {
        snprintf(why, wn, "the simulator does not support this v1 config");
        return -1;
    }
    tsr_trace2_config_from_sim(out, sim);
    out->ctl_id = tsr_trace2_ctl_id_full();   // W1b fix: not the build's CTL_ID (tsr_trace.h)
    tsr_sim_destroy(sim);
    last = c;
    last_out = *out;
    have = 1;
    return 0;
}

static int convert(const v1_trace * v, tsr_trace2 * t, char * why, size_t wn) {
    if (v->status != V1_STATUS_DONE || v->err_code != 0) {
        snprintf(why, wn, "v1 run ended with STATUS %u ERR_CODE %u: error traces are regenerated with tsr-trace-gen v2 "
                          "(SW §5.6), not converted", v->status, v->err_code);
        return -1;
    }
    if (v2_config(&v->cfg, &t->cfg, why, wn)) return -1;
    t->kind = TSR_TRACE2_KIND_OPS;
    int ended = 0;
    for (uint32_t i = 0; i < v->cmd_count; ++i) {
        uint32_t w[8];
        const uint8_t * p = v->cmd + (size_t)i * TSR_INSN_BYTES;
        for (int k = 0; k < 8; ++k)
            w[k] = (uint32_t)p[4 * k] | ((uint32_t)p[4 * k + 1] << 8) | ((uint32_t)p[4 * k + 2] << 16) | ((uint32_t)p[4 * k + 3] << 24);
        const uint32_t op = w[0] & 0xFFu;
        if (op == V1_OP_END) {
            ended = 1;
            break;
        }
        if ((w[0] >> 16) != 0) {
            snprintf(why, wn, "instruction %u (w0 0x%08x) uses v1 registers: register-relative sets are regenerated (SW §5.6)", i, w[0]);
            return -1;
        }
        if (op != TSR_OP_NOP && op != TSR_OP_LD && op != TSR_OP_ST && op != TSR_OP_QACT && op != TSR_OP_MVM && op != TSR_OP_FENCE) {
            snprintf(why, wn, "instruction %u: v1 opcode 0x%02x has no unchanged v2 encoding (EX §7.2)", i, op);
            return -1;
        }
        const uint32_t dep0 = t->n_push ? tsr_trace2_ticket_of(t, t->n_push - 1) : TSR_TICKET_NONE;
        tsr_trace2_add_op(t, w, dep0, TSR_TICKET_NONE, TSR_ISSUE_PUSH_WMASK);
    }
    if (!ended) {
        snprintf(why, wn, "no END among the %u instructions", v->cmd_count);
        return -1;
    }
    memset(&t->done, 0, sizeof(t->done));
    t->done.err_ticket_rel = TSR_TICKET_NONE;
    t->n_units = TSR_NUM_UNITS;
    memset(t->unit, 0, sizeof(t->unit));
    for (size_t i = 0; i < t->n_push; ++i) {
        const uint32_t u = tsr_trace2_push_unit(&t->cfg, &t->push[i]);
        if (u < TSR_NUM_UNITS) t->unit[u].count++;
    }
    for (size_t i = 0; i < v->cap.n_init; ++i)
        tsr_trace2_add_init(t, v->cap.init[i].ch, v->cap.init[i].byte_off, v->cap.init[i].data, v->cap.init[i].size);
    for (size_t i = 0; i < v->cap.n_expect; ++i)
        tsr_trace2_add_expect(t, v->cap.expect[i].ch, v->cap.expect[i].byte_off, v->cap.expect[i].data, v->cap.expect[i].size);
    snprintf(t->sched, sizeof(t->sched), "fixed");
    t->rv_ratio = 0.0;
    t->perf.valid = 0;
    t->load_ch = TSR_TRACE2_LOAD_AUTO;
    return 0;
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------
typedef struct { char ** v; size_t n, cap; } strlist;

static void sl_add(strlist * l, const char * s) {
    if (l->n == l->cap) {
        l->cap = l->cap ? 2 * l->cap : 64;
        l->v = (char **)xrealloc(l->v, l->cap * sizeof(char *));
    }
    l->v[l->n++] = xstrdup(s);
}

static int str_cmp(const void * a, const void * b) { return strcmp(*(char * const *)a, *(char * const *)b); }

static int has_meta(const char * dir) {
    char * p = path_join(dir, "meta.json");
    struct stat st;
    const int r = stat(p, &st) == 0 && S_ISREG(st.st_mode);
    free(p);
    return r;
}

// Relative paths (below root) of the trace directories at or below root/rel.
static void collect(const char * root, const char * rel, strlist * out, int depth) {
    char * dir = rel[0] ? path_join(root, rel) : xstrdup(root);
    if (has_meta(dir)) {
        sl_add(out, rel);
        free(dir);
        return;
    }
    DIR * d = depth < 16 ? opendir(dir) : NULL;
    if (d == NULL) {
        free(dir);
        return;
    }
    strlist subs = { 0 };
    struct dirent * e;
    while ((e = readdir(d)) != NULL) {
        if (e->d_name[0] == '.') continue;
        char * p = path_join(dir, e->d_name);
        struct stat st;
        if (stat(p, &st) == 0 && S_ISDIR(st.st_mode)) {
            char * r = rel[0] ? path_join(rel, e->d_name) : xstrdup(e->d_name);
            sl_add(&subs, r);
            free(r);
        }
        free(p);
    }
    closedir(d);
    free(dir);
    if (subs.n > 1) qsort(subs.v, subs.n, sizeof(char *), str_cmp);
    for (size_t i = 0; i < subs.n; ++i) {
        collect(root, subs.v[i], out, depth + 1);
        free(subs.v[i]);
    }
    free(subs.v);
}

static int picked(const char * list, const char * rel) {
    if (list == NULL) return 1;
    const size_t L = strlen(rel);
    for (const char * p = list; *p;) {
        const char * c = strchr(p, ',');
        const size_t n = c ? (size_t)(c - p) : strlen(p);
        if (n == L && strncmp(p, rel, n) == 0) return 1;
        if (!c) break;
        p = c + 1;
    }
    return 0;
}

static uint64_t trace_bytes(const tsr_trace2 * t) {
    uint64_t b = (uint64_t)t->n_push * TSR_TRACE_PUSH_BYTES + TSR_TRACE_TICKETS_HDR_BYTES + (uint64_t)t->n_units * TSR_TRACE_UNIT_BYTES;
    for (size_t i = 0; i < t->mem.n_init; ++i) b += t->mem.init[i].size;
    for (size_t i = 0; i < t->mem.n_expect; ++i) b += t->mem.expect[i].size;
    return b;
}

static void usage(void) {
    fprintf(stderr, "usage: tsr-trace-conv [-q] [--replay] [--sched S]... [--limit N] [--pick P1,P2,...] -o OUT IN...\n");
}

int main(int argc, char ** argv) {
    int quiet = 0, replay = 0;
    size_t limit = 0;
    const char * out = NULL, * pick = NULL;
    strlist ins = { 0 }, scheds = { 0 };
    for (int i = 1; i < argc; ++i) {
        const char * a = argv[i];
        if (strcmp(a, "-q") == 0) quiet = 1;
        else if (strcmp(a, "--replay") == 0) replay = 1;
        else if (strcmp(a, "--sched") == 0 && i + 1 < argc) sl_add(&scheds, argv[++i]);
        else if (strcmp(a, "--limit") == 0 && i + 1 < argc) limit = (size_t)strtoull(argv[++i], NULL, 0);
        else if (strcmp(a, "--pick") == 0 && i + 1 < argc) pick = argv[++i];
        else if (strcmp(a, "-o") == 0 && i + 1 < argc) out = argv[++i];
        else if (strcmp(a, "-h") == 0 || strcmp(a, "--help") == 0) { usage(); return 0; }
        else if (a[0] == '-') { fprintf(stderr, "tsr-trace-conv: unknown option %s\n", a); usage(); return 2; }
        else sl_add(&ins, a);
    }
    if (out == NULL || ins.n == 0) {
        usage();
        return 2;
    }
    unsigned threads = 1;
    const char * te = getenv("TSR_SIM_THREADS");
    if (te) threads = (unsigned)strtoul(te, NULL, 0);
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    size_t converted = 0, done = 0, passed = 0, failed = 0, errors = 0;
    uint64_t bytes = 0;
    const size_t n_sched = scheds.n ? scheds.n : 1;
    for (size_t k = 0; k < ins.n; ++k) {
        strlist rels = { 0 };
        const int single = has_meta(ins.v[k]);
        collect(ins.v[k], "", &rels, 0);
        if (rels.n == 0) {
            fprintf(stderr, "tsr-trace-conv: %s: no trace (no meta.json below it)\n", ins.v[k]);
            ++errors;
        }
        for (size_t i = 0; i < rels.n; ++i) {
            const char * rel = rels.v[i];
            if (!picked(pick, rel) || (limit && done >= limit)) continue;
            ++done;
            char * src = rel[0] ? path_join(ins.v[k], rel) : xstrdup(ins.v[k]);
            const char * base = strrchr(ins.v[k], '/');
            char * dst = single ? path_join(out, base && base[1] ? base + 1 : ins.v[k]) : path_join(out, rel);
            v1_trace v;
            char why[600];
            tsr_trace2 * t = tsr_trace2_new();
            if (v1_trace_read(src, &v) != 0) {
                printf("ERROR %s: not a readable v1 trace\n", src);
                ++errors;
            } else {
                if (convert(&v, t, why, sizeof(why)) != 0) {
                    printf("ERROR %s: %s\n", src, why);
                    ++errors;
                } else if (tsr_trace2_write(t, dst, NULL) != 0) {
                    printf("ERROR %s: cannot write %s\n", src, dst);
                    ++errors;
                } else {
                    ++converted;
                    bytes += trace_bytes(t);
                    if (!quiet) printf("conv  %s -> %s: %zu pushes, %zu init / %zu expect regions\n", src, dst, t->n_push,
                                       t->mem.n_init, t->mem.n_expect);
                    if (replay) {
                        tsr_trace2 * r = NULL;
                        if (tsr_trace2_read(dst, &r, why, sizeof(why)) != 0) {
                            printf("ERROR %s: read back: %s\n", dst, why);
                            errors += n_sched;
                        } else {
                            for (size_t s = 0; s < n_sched; ++s) {
                                tsr_trace2_replay_opts o;
                                memset(&o, 0, sizeof(o));
                                o.sched = scheds.n ? scheds.v[s] : NULL;
                                o.threads = threads;
                                tsr_trace2_result res;
                                const int rc = tsr_trace2_replay_sim_ex(r, &o, &res);
                                const char * sn = o.sched ? o.sched : "default";
                                if (rc == 0) {
                                    ++passed;
                                    if (!quiet) printf("ok    %s [%s]: %s\n", dst, sn, res.msg);
                                } else if (rc == 1) {
                                    ++failed;
                                    printf("FAIL  %s [%s]: %s\n", dst, sn, res.msg);
                                } else {
                                    ++errors;
                                    printf("ERROR %s [%s]: %s\n", dst, sn, res.msg);
                                }
                                fflush(stdout);
                            }
                            tsr_trace2_free(r);
                        }
                    }
                }
                v1_trace_free(&v);
            }
            tsr_trace2_free(t);
            free(src);
            free(dst);
        }
        for (size_t i = 0; i < rels.n; ++i) free(rels.v[i]);
        free(rels.v);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    const double secs = (double)(t1.tv_sec - t0.tv_sec) + 1e-9 * (double)(t1.tv_nsec - t0.tv_nsec);
    printf("tsr-trace-conv: %zu converted, %.1f MiB written, %.1f s; %zu replays passed, %zu failed, %zu errors\n", converted,
           (double)bytes / (1024.0 * 1024.0), secs, passed, failed, errors);
    for (size_t i = 0; i < ins.n; ++i) free(ins.v[i]);
    for (size_t i = 0; i < scheds.n; ++i) free(scheds.v[i]);
    free(ins.v);
    free(scheds.v);
    return errors ? 2 : failed ? 1 : 0;
}
