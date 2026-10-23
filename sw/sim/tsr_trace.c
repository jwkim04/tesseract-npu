// tsr_trace.c — trace directories: write, read, replay (spec 트레이스 형식, tsr_trace.h).
//
// meta.json is produced by a small hand-written writer and read back by a small JSON parser
// (objects, arrays, strings with escapes, numbers, true/false/null; any whitespace). Integers
// that the format needs are parsed exactly as unsigned 64-bit values.

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
#include <unistd.h>

#define TRACE_FORMAT  "tesseract-trace"
#define TRACE_VERSION 1u
#define CMD_BIN_MAX   (1ull << 30)   // refuse to write a cmd.bin larger than 1 GiB
#define CMD_ADDR_SANE (1ull << 50)   // beat addresses at or above this lie outside every channel

static void terr(const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fputs("tsr_trace: ", stderr);
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    va_end(ap);
}

static void * xmalloc(size_t n) {
    void * p = malloc(n ? n : 1);
    if (p == NULL) {
        fprintf(stderr, "tsr_trace: out of memory (%zu bytes)\n", n);
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

// ---------------------------------------------------------------------------
// File helpers
// ---------------------------------------------------------------------------
static int mkdir_p(const char * path) {
    char * p = xstrdup(path);
    const size_t n = strlen(p);
    for (size_t i = 1; i <= n; ++i) {
        if (p[i] == '/' || p[i] == '\0') {
            const char c = p[i];
            p[i] = '\0';
            if (mkdir(p, 0755) != 0 && errno != EEXIST) {
                terr("mkdir %s: %s", p, strerror(errno));
                free(p);
                return -1;
            }
            p[i] = c;
        }
    }
    free(p);
    struct stat st;
    if (stat(path, &st) != 0 || !S_ISDIR(st.st_mode)) {
        terr("%s is not a directory", path);
        return -1;
    }
    return 0;
}

static int write_file(const char * path, const void * data, size_t size) {
    FILE * f = fopen(path, "wb");
    if (f == NULL) {
        terr("cannot create %s: %s", path, strerror(errno));
        return -1;
    }
    const size_t w = size ? fwrite(data, 1, size, f) : 0;
    const int rc_close = fclose(f);
    if (w != size || rc_close != 0) {
        terr("write error on %s", path);
        return -1;
    }
    return 0;
}

// Reads a whole file. *size receives its length. Returns NULL on error (message printed).
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

// Removes the files of a previous trace: meta.json, cmd.bin, <sub>/ch*_*.bin.
static void remove_stale(const char * dir) {
    char * p = path_join(dir, "meta.json");
    unlink(p);
    free(p);
    p = path_join(dir, "cmd.bin");
    unlink(p);
    free(p);
    const char * subs[2] = { "init", "expect" };
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

// Sorts and merges overlapping/adjacent spans in place; returns the new count.
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

static span * regions_to_spans(const tsr_region * r, size_t n) {
    span * s = (span *)xmalloc((n ? n : 1) * sizeof(span));
    for (size_t i = 0; i < n; ++i) {
        s[i].ch = r[i].ch;
        s[i].lo = r[i].byte_off;
        s[i].hi = r[i].byte_off + r[i].size;
    }
    return s;
}

// Is [lo, hi) on ch inside one span of the normalized list?
static int span_covered(const span * s, size_t n, uint32_t ch, uint64_t lo, uint64_t hi) {
    size_t a = 0, b = n;
    while (a < b) {   // first span with (ch, hi) > (ch, lo)
        const size_t m = a + (b - a) / 2;
        if (s[m].ch < ch || (s[m].ch == ch && s[m].hi <= lo)) a = m + 1; else b = m;
    }
    return a < n && s[a].ch == ch && s[a].lo <= lo && hi <= s[a].hi;
}

static void region_name(char * buf, size_t n, const char * sub, uint32_t ch, uint64_t off) {
    snprintf(buf, n, "%s/ch%u_%010llx.bin", sub, ch, (unsigned long long)off);
}

// Checks one region list: inside the channels, aligned, non-empty, non-overlapping.
static int check_regions(const tsr_config * cfg, const tsr_region * r, size_t n, const char * what) {
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
static uint8_t * derive_cmd(const tsr_trace * tr, size_t * size) {
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

int tsr_trace_write(const char * dir, const tsr_trace * tr) {
    const tsr_config * c = &tr->cfg;
    if (c->num_ch < 1 || c->num_ch >= TSR_TRACE_MAX_CH) {
        terr("write %s: num_ch %u out of range", dir, c->num_ch);
        return -1;
    }
    if (check_regions(c, tr->cap.init, tr->cap.n_init, "init") || check_regions(c, tr->cap.expect, tr->cap.n_expect, "expect")) {
        return -1;
    }
    if ((uint64_t)tr->cmd_count * TSR_INSN_BYTES > CMD_BIN_MAX) {
        terr("write %s: cmd_count %u makes cmd.bin larger than 1 GiB; refusing", dir, tr->cmd_count);
        return -1;
    }
    char * init_dir = path_join(dir, "init");
    char * exp_dir = path_join(dir, "expect");
    int rc = -1;
    char * meta_tmp = NULL;
    char * meta = NULL;
    FILE * f = NULL;
    if (mkdir_p(dir) || mkdir_p(init_dir) || mkdir_p(exp_dir)) goto done;
    remove_stale(dir);

    for (int pass = 0; pass < 2; ++pass) {
        const tsr_region * r = pass ? tr->cap.expect : tr->cap.init;
        const size_t n = pass ? tr->cap.n_expect : tr->cap.n_init;
        for (size_t i = 0; i < n; ++i) {
            char name[96];
            region_name(name, sizeof(name), pass ? "expect" : "init", r[i].ch, r[i].byte_off);
            char * p = path_join(dir, name);
            const int w = write_file(p, r[i].data, (size_t)r[i].size);
            free(p);
            if (w) goto done;
        }
    }
    {
        size_t cmd_size = 0;
        uint8_t * cmd = tr->cmd;
        int own = 0;
        if (cmd == NULL) {
            cmd = derive_cmd(tr, &cmd_size);
            own = 1;
            if (cmd == NULL) { terr("out of memory building cmd.bin"); goto done; }
        } else {
            cmd_size = (size_t)tr->cmd_count * TSR_INSN_BYTES;
        }
        char * p = path_join(dir, "cmd.bin");
        const int w = write_file(p, cmd, cmd_size);
        free(p);
        if (own) free(cmd);
        if (w) goto done;
    }

    meta_tmp = path_join(dir, "meta.json.tmp");
    meta = path_join(dir, "meta.json");
    f = fopen(meta_tmp, "w");
    if (f == NULL) {
        terr("cannot create %s: %s", meta_tmp, strerror(errno));
        goto done;
    }
    fprintf(f, "{\n  \"format\": \"%s\",\n  \"version\": %u,\n", TRACE_FORMAT, TRACE_VERSION);
    fprintf(f, "  \"config\": {\"num_ch\": %u, \"beat_bits\": %u, \"lanes\": %u, \"t_max\": %u, \"k_max\": %u, "
               "\"spm_bytes\": %u, \"vl\": %u, \"nireg\": %u, \"nsreg\": %u, \"loop_depth\": %u, "
               "\"rd_outstanding\": %u, \"ch_bytes\": %llu, \"addr_beat_w\": %u, \"caps\": %u},\n",
            c->num_ch, c->beat_bits, c->lanes, c->t_max, c->k_max, c->spm_bytes, c->vl, c->nireg, c->nsreg,
            c->loop_depth, c->rd_outstanding, (unsigned long long)c->ch_bytes, c->addr_beat_w, c->caps);
    fprintf(f, "  \"cmd_addr_beat\": %llu,\n  \"cmd_count\": %u,\n  \"args\": [%u, %u, %u, %u, %u, %u, %u, %u],\n",
            (unsigned long long)tr->cmd_addr_beat, tr->cmd_count, tr->args[0], tr->args[1], tr->args[2],
            tr->args[3], tr->args[4], tr->args[5], tr->args[6], tr->args[7]);
    fprintf(f, "  \"status\": %u,\n  \"err_code\": %u,\n  \"err_pc\": %u,\n  \"err_info\": %u,\n", tr->status,
            tr->err_code, tr->err_pc, tr->err_info);
    write_region_list(f, "init", "init", tr->cap.init, tr->cap.n_init);
    fputs(",\n", f);
    write_region_list(f, "expect", "expect", tr->cap.expect, tr->cap.n_expect);
    if (tr->perf.valid) {
        fprintf(f, ",\n  \"perf\": {\"cycles\": %llu, \"retired\": %llu, \"mvm_cycles\": %llu, \"ch_rd_beats\": [",
                (unsigned long long)tr->perf.cycles, (unsigned long long)tr->perf.retired,
                (unsigned long long)tr->perf.mvm_cycles);
        for (uint32_t ch = 0; ch < c->num_ch; ++ch) {
            fprintf(f, "%s%llu", ch ? ", " : "", (unsigned long long)tr->perf.ch_rd_beats[ch]);
        }
        fputs("]}", f);
    }
    fputs("\n}\n", f);
    {
        const int bad = ferror(f);
        const int rc_close = fclose(f);
        f = NULL;
        if (bad || rc_close != 0) {
            terr("write error on %s", meta_tmp);
            goto done;
        }
    }
    if (rename(meta_tmp, meta) != 0) {
        terr("rename %s -> %s: %s", meta_tmp, meta, strerror(errno));
        goto done;
    }
    rc = 0;
done:
    if (f) fclose(f);
    free(meta_tmp);
    free(meta);
    free(init_dir);
    free(exp_dir);
    return rc;
}

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
                            const tsr_config * cfg, tsr_region ** out, size_t * n_out) {
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

int tsr_trace_read(const char * dir, tsr_trace * tr) {
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
        tsr_config * k = &tr->cfg;
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
    if (rc) tsr_trace_free(tr);
    return rc;
}

void tsr_trace_free(tsr_trace * tr) {
    if (tr == NULL) return;
    tsr_capture_free(&tr->cap);
    free(tr->cmd);
    memset(tr, 0, sizeof(*tr));
}

// ---------------------------------------------------------------------------
// Replay
// ---------------------------------------------------------------------------
int tsr_trace_replay(const tsr_trace * tr, int verbose) { return tsr_trace_replay_threads(tr, verbose, 1); }

int tsr_trace_replay_threads(const tsr_trace * tr, int verbose, unsigned threads) {
    const tsr_config * cfg = &tr->cfg;
    if (cfg->num_ch < 1 || cfg->num_ch >= TSR_TRACE_MAX_CH) {
        terr("replay: num_ch %u out of range", cfg->num_ch);
        return 2;
    }
    if (check_regions(cfg, tr->cap.init, tr->cap.n_init, "init") ||
        check_regions(cfg, tr->cap.expect, tr->cap.n_expect, "expect")) {
        return 2;
    }
    tsr_sim * sim = tsr_sim_create(cfg, NULL);
    if (sim == NULL) {
        terr("replay: cannot create a simulator for this config");
        return 2;
    }
    tsr_sim_set_threads(sim, threads);

    // Allowed read set: init regions plus the command buffer bytes from cmd.bin.
    size_t n_allowed = tr->cap.n_init;
    span * allowed = (span *)xmalloc((n_allowed + 1) * sizeof(span));
    for (size_t i = 0; i < tr->cap.n_init; ++i) {
        allowed[i].ch = tr->cap.init[i].ch;
        allowed[i].lo = tr->cap.init[i].byte_off;
        allowed[i].hi = tr->cap.init[i].byte_off + tr->cap.init[i].size;
    }
    if (tr->cmd != NULL && tr->cmd_count > 0 && tr->cmd_addr_beat < cfg->ch_bytes / TSR_BEAT_BYTES) {
        const uint64_t lo = tr->cmd_addr_beat * TSR_BEAT_BYTES;
        uint64_t n = (uint64_t)tr->cmd_count * TSR_INSN_BYTES;
        if (n > cfg->ch_bytes - lo) n = cfg->ch_bytes - lo;
        memcpy(tsr_sim_dram(sim, 0) + lo, tr->cmd, (size_t)n);
        // Fetch reads whole beats: the beat holding the last (odd-count) instruction is allowed.
        uint64_t hi = lo + ((n + TSR_BEAT_BYTES - 1) / TSR_BEAT_BYTES) * TSR_BEAT_BYTES;
        if (hi > cfg->ch_bytes) hi = cfg->ch_bytes;
        allowed[n_allowed].ch = 0;
        allowed[n_allowed].lo = lo;
        allowed[n_allowed].hi = hi;
        n_allowed++;
    }
    n_allowed = span_normalize(allowed, n_allowed);
    for (size_t i = 0; i < tr->cap.n_init; ++i) {
        const tsr_region * r = &tr->cap.init[i];
        memcpy(tsr_sim_dram(sim, r->ch) + r->byte_off, r->data, (size_t)r->size);
    }

    tsr_sim_capture_begin(sim);
    const uint32_t status = tsr_sim_exec(sim, tr->cmd_addr_beat, tr->cmd_count, tr->args);
    tsr_capture got;
    tsr_sim_capture_end(sim, &got);
    const uint32_t err_code = tsr_sim_csr_read(sim, TSR_CSR_ERR_CODE);
    const uint32_t err_pc = tsr_sim_csr_read(sim, TSR_CSR_ERR_PC);
    const uint32_t err_info = tsr_sim_csr_read(sim, TSR_CSR_ERR_INFO);

    int fail = 0;
    if (status != tr->status || err_code != tr->err_code || err_pc != tr->err_pc ||
        (tr->has_err_info && err_info != tr->err_info)) {
        fprintf(stderr, "replay: final state differs: STATUS %u ERR_CODE %u ERR_PC %u ERR_INFO 0x%x, expected "
                        "STATUS %u ERR_CODE %u ERR_PC %u ERR_INFO %s0x%x\n",
                status, err_code, err_pc, err_info, tr->status, tr->err_code, tr->err_pc,
                tr->has_err_info ? "" : "(not compared) ", tr->err_info);
        fail = 1;
    }

    // Written beats must equal the union of the expect regions.
    span * want = regions_to_spans(tr->cap.expect, tr->cap.n_expect);
    span * have = regions_to_spans(got.expect, got.n_expect);
    const size_t n_want = span_normalize(want, tr->cap.n_expect);
    const size_t n_have = span_normalize(have, got.n_expect);
    for (size_t i = 0; i < n_have; ++i) {
        if (!span_covered(want, n_want, have[i].ch, have[i].lo, have[i].hi)) {
            fprintf(stderr, "replay: run wrote ch %u bytes [0x%llx, 0x%llx) outside the expect regions\n", have[i].ch,
                    (unsigned long long)have[i].lo, (unsigned long long)have[i].hi);
            fail = 1;
            break;
        }
    }
    for (size_t i = 0; i < n_want; ++i) {
        if (!span_covered(have, n_have, want[i].ch, want[i].lo, want[i].hi)) {
            fprintf(stderr, "replay: expect region ch %u bytes [0x%llx, 0x%llx) was not (fully) written by the run\n",
                    want[i].ch, (unsigned long long)want[i].lo, (unsigned long long)want[i].hi);
            fail = 1;
            break;
        }
    }
    free(want);
    free(have);

    // Expect data, bit-exact. Report the first differing byte (regions are sorted on write).
    for (size_t i = 0; i < tr->cap.n_expect; ++i) {
        const tsr_region * r = &tr->cap.expect[i];
        const uint8_t * d = tsr_sim_dram(sim, r->ch) + r->byte_off;
        if (memcmp(d, r->data, (size_t)r->size) != 0) {
            size_t k = 0;
            while (d[k] == r->data[k]) ++k;
            const uint64_t at = r->byte_off + k;
            const size_t w = k & ~(size_t)3;
            uint32_t gw = 0, ew = 0;
            if (w + 4 <= r->size) {
                memcpy(&gw, d + w, 4);
                memcpy(&ew, r->data + w, 4);
            }
            size_t ndiff = 0;
            for (size_t j = k; j < r->size; ++j) ndiff += d[j] != r->data[j];
            fprintf(stderr, "replay: expect mismatch at ch %u byte offset 0x%llx: got 0x%02x, expected 0x%02x "
                            "(32-bit word got 0x%08x expected 0x%08x; %zu differing bytes in this region)\n",
                    r->ch, (unsigned long long)at, d[k], r->data[k], gw, ew, ndiff);
            fail = 1;
            break;
        }
    }

    // Every read must come from init (or the command buffer).
    for (size_t i = 0; i < got.n_init; ++i) {
        const tsr_region * r = &got.init[i];
        if (!span_covered(allowed, n_allowed, r->ch, r->byte_off, r->byte_off + r->size)) {
            fprintf(stderr, "replay: run read ch %u bytes [0x%llx, 0x%llx) that the trace does not initialise\n", r->ch,
                    (unsigned long long)r->byte_off, (unsigned long long)(r->byte_off + r->size));
            fail = 1;
            break;
        }
    }
    free(allowed);

    // Perf model: informational.
    if (tr->perf.valid) {
        int pd = tsr_sim_perf(sim, TSR_PERF_CYCLES) != tr->perf.cycles ||
                 tsr_sim_perf(sim, TSR_PERF_RETIRED) != tr->perf.retired ||
                 tsr_sim_perf(sim, TSR_PERF_MVM_CYCLES) != tr->perf.mvm_cycles;
        for (uint32_t ch = 0; ch < cfg->num_ch; ++ch) {
            pd |= tsr_sim_perf(sim, tsr_perf_ch_rd_beats(ch)) != tr->perf.ch_rd_beats[ch];
        }
        if (pd) {
            fprintf(stderr, "replay: warning: perf counters differ from the trace (cycles %llu vs %llu, retired %llu "
                            "vs %llu); not a failure\n",
                    (unsigned long long)tsr_sim_perf(sim, TSR_PERF_CYCLES), (unsigned long long)tr->perf.cycles,
                    (unsigned long long)tsr_sim_perf(sim, TSR_PERF_RETIRED), (unsigned long long)tr->perf.retired);
        }
    }
    if (verbose) {
        fprintf(stderr, "replay: %s: STATUS %u ERR_CODE %u ERR_PC %u, %zu init / %zu expect regions, %llu cycles, %llu retired\n",
                fail ? "FAIL" : "ok", status, err_code, err_pc, tr->cap.n_init, tr->cap.n_expect,
                (unsigned long long)tsr_sim_perf(sim, TSR_PERF_CYCLES),
                (unsigned long long)tsr_sim_perf(sim, TSR_PERF_RETIRED));
    }
    tsr_capture_free(&got);
    tsr_sim_destroy(sim);
    return fail;
}
