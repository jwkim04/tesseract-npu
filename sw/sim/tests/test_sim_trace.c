// test_sim_trace.c — access capture semantics, trace write/read/replay round trip, corruption
// detection, meta.json robustness, error traces, and capture of a large MVM.
#include "test_sim_util.h"
#include "tsr_trace.h"

#include <time.h>

#define DONE    TSR_STATUS_DONE
#define DONEERR (TSR_STATUS_DONE | TSR_STATUS_ERR)

static const char * g_base;
static const uint64_t IO = TSR_REGION_IO_OFF / 64;
static const uint64_t WOFF = TSR_REGION_WEIGHT_OFF / 64;

static int region_is(const tsr_region * r, uint32_t ch, uint64_t beat, uint64_t beats) {
    return r->ch == ch && r->byte_off == beat * 64 && r->size == beats * 64;
}

static uint8_t * dup_bytes(const void * p, size_t n) {
    uint8_t * d = (uint8_t *)malloc(n ? n : 1);
    memcpy(d, p, n);
    return d;
}

static char * read_text(const char * path, size_t * n) {
    FILE * f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    const long L = ftell(f);
    fseek(f, 0, SEEK_SET);
    char * t = (char *)malloc((size_t)L + 1);
    if (fread(t, 1, (size_t)L, f) != (size_t)L) { fclose(f); free(t); return NULL; }
    fclose(f);
    t[L] = 0;
    *n = (size_t)L;
    return t;
}
static void write_text(const char * path, const char * t, size_t n) {
    FILE * f = fopen(path, "wb");
    fwrite(t, 1, n, f);
    fclose(f);
}
static char * pjoin(const char * a, const char * b) {
    const size_t n = strlen(a) + strlen(b) + 2;
    char * p = (char *)malloc(n);
    snprintf(p, n, "%s/%s", a, b);
    return p;
}
static void flip_byte(const char * path, long off, uint8_t mask) {
    FILE * f = fopen(path, "r+b");
    CHECK(f != NULL, "open %s", path);
    if (!f) return;
    fseek(f, off, SEEK_SET);
    int c = fgetc(f);
    fseek(f, off, SEEK_SET);
    fputc((c ^ mask) & 0xFF, f);
    fclose(f);
}
// Replaces the first occurrence of `from` by `to` in the file; returns 1 if found.
static int edit_file(const char * path, const char * from, const char * to) {
    size_t n;
    char * t = read_text(path, &n);
    if (!t) return 0;
    char * at = strstr(t, from);
    if (!at) { free(t); return 0; }
    const size_t a = (size_t)(at - t), lf = strlen(from), lt = strlen(to);
    char * o = (char *)malloc(n - lf + lt + 1);
    memcpy(o, t, a);
    memcpy(o + a, to, lt);
    memcpy(o + a + lt, at + lf, n - a - lf);
    write_text(path, o, n - lf + lt);
    free(o);
    free(t);
    return 1;
}

static void trace_from(tsr_trace * tr, tsr_sim * s, uint64_t cmd, uint32_t count, const uint32_t * args) {
    memset(tr, 0, sizeof(*tr));
    tr->cfg = *tsr_sim_config(s);
    tr->cmd_addr_beat = cmd;
    tr->cmd_count = count;
    for (int i = 0; i < 8; ++i) tr->args[i] = args ? args[i] : 0;
    tr->status = tsr_sim_csr_read(s, TSR_CSR_STATUS);
    tr->err_code = tsr_sim_csr_read(s, TSR_CSR_ERR_CODE);
    tr->err_pc = tsr_sim_csr_read(s, TSR_CSR_ERR_PC);
    tr->err_info = tsr_sim_csr_read(s, TSR_CSR_ERR_INFO);
    tr->has_err_info = 1;
    tr->perf.valid = 1;
    tr->perf.cycles = tsr_sim_perf(s, TSR_PERF_CYCLES);
    tr->perf.retired = tsr_sim_perf(s, TSR_PERF_RETIRED);
    tr->perf.mvm_cycles = tsr_sim_perf(s, TSR_PERF_MVM_CYCLES);
    for (uint32_t c = 0; c < tr->cfg.num_ch; ++c) tr->perf.ch_rd_beats[c] = tsr_sim_perf(s, tsr_perf_ch_rd_beats(c));
}

static int traces_equal(const tsr_trace * a, const tsr_trace * b) {
    if (memcmp(&a->cfg, &b->cfg, offsetof(tsr_config, ch_bytes)) != 0 || a->cfg.ch_bytes != b->cfg.ch_bytes ||
        a->cfg.addr_beat_w != b->cfg.addr_beat_w || a->cfg.caps != b->cfg.caps) return 0;
    if (a->cmd_addr_beat != b->cmd_addr_beat || a->cmd_count != b->cmd_count || memcmp(a->args, b->args, sizeof(a->args)))
        return 0;
    if (a->status != b->status || a->err_code != b->err_code || a->err_pc != b->err_pc || a->err_info != b->err_info ||
        a->has_err_info != b->has_err_info) return 0;
    if (a->perf.valid != b->perf.valid || a->perf.cycles != b->perf.cycles || a->perf.retired != b->perf.retired ||
        a->perf.mvm_cycles != b->perf.mvm_cycles || memcmp(a->perf.ch_rd_beats, b->perf.ch_rd_beats, 8 * a->cfg.num_ch))
        return 0;
    if (a->cap.n_init != b->cap.n_init || a->cap.n_expect != b->cap.n_expect) return 0;
    for (int pass = 0; pass < 2; ++pass) {
        const tsr_region * x = pass ? a->cap.expect : a->cap.init;
        const tsr_region * y = pass ? b->cap.expect : b->cap.init;
        const size_t n = pass ? a->cap.n_expect : a->cap.n_init;
        for (size_t i = 0; i < n; ++i) {
            if (x[i].ch != y[i].ch || x[i].byte_off != y[i].byte_off || x[i].size != y[i].size ||
                memcmp(x[i].data, y[i].data, (size_t)x[i].size)) return 0;
        }
    }
    return 1;
}

// ---------------------------------------------------------------------------
static void test_capture_and_roundtrip(void) {
    tsr_config cfg = cfg_u200(1);
    tsr_sim * s = tsr_sim_create(&cfg, NULL);
    rng_t r = { 11 };
    for (uint32_t c = 0; c < 4; ++c) rng_bytes(&r, tsr_sim_dram(s, c) + IO * 64, 1000 * 64);
    // ch0 IO+10.. holds 64 f32 activations (finite)
    float x[64];
    gen_f32(&r, x, 64);
    put_f32_le(tsr_sim_dram(s, 0) + (IO + 10) * 64, x, 64);
    // Q8_0 N=6 K=64 -> R=2: ch0 rows 0-1, ch1 2-3, ch2 4-5, ch3 none
    const uint32_t N = 6, K = 64, rb = tsr_row_beats(TSR_FMT_Q8_0, K);
    uint8_t blocks[6 * 2 * 34];
    for (uint32_t n = 0; n < N; ++n) gen_row_blocks(&r, TSR_FMT_Q8_0, K, blocks + n * 68);
    put_weights(s, TSR_FMT_Q8_0, N, K, WOFF, blocks);

    prog_t p = {0};
    prog_add(&p, tsr_enc_ld(0, 0, (uint32_t)(IO + 10), 4));      // read [IO+10, IO+14)
    prog_add(&p, tsr_enc_st(0, 0, (uint32_t)(IO + 200), 2));     // write [IO+200, IO+202)
    prog_add(&p, tsr_enc_ld(0, 8, (uint32_t)(IO + 200), 2));     // read of written data: not init
    prog_add(&p, tsr_enc_ld(0, 16, (uint32_t)(IO + 12), 4));     // [IO+12, IO+16): only +14..16 new
    prog_add(&p, tsr_enc_ld(2, 24, (uint32_t)(IO + 50), 1));     // ch2 read ...
    prog_add(&p, tsr_enc_st(2, 0, (uint32_t)(IO + 50), 1));      // ... then written: init + expect
    prog_add(&p, tsr_enc_qact(TSR_QA_A32, 40, 0, K, 1, 0));
    prog_add(&p, tsr_enc_mvm(TSR_FMT_Q8_0, 0, 60, 40, (uint32_t)WOFF, N, K, 1, 0));
    prog_add(&p, tsr_enc_st(1, 60, (uint32_t)(IO + 300), 1));
    prog_add(&p, tsr_enc_st(0, 60, (uint32_t)(IO + 202), 1));    // right-adjacent to [IO+200, IO+202)
    prog_add(&p, tsr_enc_st(0, 60, (uint32_t)(IO + 199), 1));    // left-adjacent -> [IO+199, IO+203)
    prog_add(&p, tsr_enc_ld(0, 30, 99, 1));                       // beat before the program: init [99, 107)
    prog_add(&p, tsr_enc_end());
    const uint64_t cmd = 100;
    prog_load(s, &p, cmd);
    uint8_t * pre_ch2 = dup_bytes(tsr_sim_dram(s, 2) + (IO + 50) * 64, 64);
    uint8_t * pre_ch0 = dup_bytes(tsr_sim_dram(s, 0) + (IO + 10) * 64, 6 * 64);

    tsr_sim_capture_begin(s);
    const uint32_t st = tsr_sim_exec(s, cmd, p.n, NULL);
    tsr_capture cap;
    tsr_sim_capture_end(s, &cap);
    CHECK(st == DONE, "capture program status %u (err %u at %u)", st, sim_err_code(s), sim_err_pc(s));

    // init: ch0 [99,107) (LD beat + cmd), IO [10,16), W [0,2rb); ch1 W; ch2 IO+50, W. Sorted by (ch, off).
    CHECK(cap.n_init == 6, "n_init %zu", cap.n_init);
    if (cap.n_init == 6) {
        CHECK(region_is(&cap.init[0], 0, cmd - 1, 8), "init0 cmd: ch %u off %llu size %llu", cap.init[0].ch,
              (unsigned long long)cap.init[0].byte_off, (unsigned long long)cap.init[0].size);
        CHECK(region_is(&cap.init[1], 0, IO + 10, 6), "init1 merged IO read");
        CHECK(region_is(&cap.init[2], 0, WOFF, 2 * rb), "init2 ch0 weights");
        CHECK(region_is(&cap.init[3], 1, WOFF, 2 * rb), "init3 ch1 weights");
        CHECK(region_is(&cap.init[4], 2, IO + 50, 1), "init4 ch2 read-then-written");
        CHECK(region_is(&cap.init[5], 2, WOFF, 2 * rb), "init5 ch2 weights");
        CHECK(memcmp(cap.init[0].data, tsr_sim_dram(s, 0) + (cmd - 1) * 64, 8 * 64) == 0, "cmd bytes");
        CHECK(memcmp(cap.init[1].data, pre_ch0, 6 * 64) == 0, "IO pre-content");
        CHECK(memcmp(cap.init[4].data, pre_ch2, 64) == 0, "ch2 pre-content (before the ST)");
        CHECK(memcmp(cap.init[5].data, tsr_sim_dram(s, 2) + WOFF * 64, 2 * rb * 64) == 0, "ch2 weights");
    }
    CHECK(cap.n_expect == 3, "n_expect %zu", cap.n_expect);
    if (cap.n_expect == 3) {
        CHECK(region_is(&cap.expect[0], 0, IO + 199, 4), "expect0 merged adjacent writes");
        CHECK(region_is(&cap.expect[1], 1, IO + 300, 1), "expect1");
        CHECK(region_is(&cap.expect[2], 2, IO + 50, 1), "expect2");
        CHECK(memcmp(cap.expect[2].data, tsr_sim_dram(s, 2) + (IO + 50) * 64, 64) == 0, "expect post-content");
        CHECK(memcmp(cap.expect[2].data, pre_ch2, 64) != 0, "ch2 beat really changed");
    }
    // A capture that was never begun is empty; begin/end without a run is empty.
    tsr_capture empty;
    tsr_sim_capture_end(s, &empty);
    CHECK(empty.n_init == 0 && empty.n_expect == 0, "capture_end without begin");
    tsr_sim_capture_begin(s);
    tsr_sim_capture_end(s, &empty);
    CHECK(empty.n_init == 0 && empty.n_expect == 0, "empty capture");

    // ---- trace round trip
    tsr_trace tr;
    trace_from(&tr, s, cmd, p.n, NULL);
    tr.cap = cap;
    char * dir = pjoin(g_base, "rt");
    CHECK(tsr_trace_write(dir, &tr) == 0, "write");
    tsr_trace rd;
    CHECK(tsr_trace_read(dir, &rd) == 0, "read");
    CHECK(traces_equal(&tr, &rd), "read == written");
    CHECK(rd.cmd != NULL && memcmp(rd.cmd, tsr_sim_dram(s, 0) + cmd * 64, p.n * 32) == 0, "cmd.bin content");
    CHECK(tsr_trace_replay(&rd, 0) == 0, "replay of an intact trace passes");
    CHECK(tsr_trace_replay_threads(&rd, 0, 3) == 0, "replay with 3 threads passes");
    CHECK(tsr_trace_replay(&tr, 0) == 0, "replay of the in-memory trace (cmd = NULL) passes");

    // meta.json robustness: minified and heavily spaced variants read back identically.
    char * meta = pjoin(dir, "meta.json");
    size_t mlen;
    char * text = read_text(meta, &mlen);
    char * mini = (char *)malloc(mlen + 1);
    size_t k = 0;
    int in_str = 0;
    for (size_t i = 0; i < mlen; ++i) {
        const char c = text[i];
        if (c == '"') in_str = !in_str;
        if (!in_str && (c == ' ' || c == '\n' || c == '\t' || c == '\r')) continue;
        mini[k++] = c;
    }
    write_text(meta, mini, k);
    tsr_trace v;
    CHECK(tsr_trace_read(dir, &v) == 0 && traces_equal(&tr, &v), "minified meta.json");
    tsr_trace_free(&v);
    char * spaced = (char *)malloc(mlen * 4 + 16);
    size_t m = 0;
    in_str = 0;
    for (size_t i = 0; i < k; ++i) {
        const char c = mini[i];
        if (c == '"') in_str = !in_str;
        if (!in_str && (c == ',' || c == ':' || c == '{' || c == '[')) { spaced[m++] = c; spaced[m++] = '\r'; spaced[m++] = '\n'; spaced[m++] = '\t'; continue; }
        if (!in_str && (c == '}' || c == ']')) { spaced[m++] = ' '; spaced[m++] = c; continue; }
        spaced[m++] = c;
    }
    write_text(meta, spaced, m);
    CHECK(tsr_trace_read(dir, &v) == 0 && traces_equal(&tr, &v), "spaced meta.json");
    tsr_trace_free(&v);

    // Malformed meta.json variants must be rejected.
    struct { const char * from, * to, * what; } bad[] = {
        { "\"version\":\r\n\t1", "\"version\":2", "version 2" },
        { "\"k_max\":", "\"k_maxx\":", "missing k_max" },
        { "\"format\":\r\n\t\"tesseract-trace\"", "\"format\":\"other\"", "format" },
        { "\"cmd_count\":\r\n\t13", "\"cmd_count\":14", "cmd_count vs cmd.bin size" },
        { "\"size\":\r\n\t512", "\"size\":576", "region size vs file" },
        { "\"file\":\r\n\t\"init/ch0_", "\"file\":\"../init/ch0_", "file outside the trace" },
        { "\"args\":\r\n\t[", "\"args\":[1,", "args length" },
        { "\"offset\":\r\n\t6336", "\"offset\":6337", "unaligned offset" },
        { "\"status\":\r\n\t2", "\"status\":-2", "negative" },
        { "\"status\":\r\n\t2", "\"status\":2.5", "fraction" },
    };
    for (size_t i = 0; i < sizeof(bad) / sizeof(bad[0]); ++i) {
        write_text(meta, spaced, m);
        CHECK(edit_file(meta, bad[i].from, bad[i].to), "pattern for '%s' found", bad[i].what);
        CHECK(tsr_trace_read(dir, &v) != 0, "malformed meta accepted: %s", bad[i].what);
    }
    write_text(meta, spaced, m - 3);   // truncated JSON
    CHECK(tsr_trace_read(dir, &v) != 0, "truncated meta accepted");
    write_text(meta, text, mlen);      // restore the original
    CHECK(tsr_trace_read(dir, &v) == 0, "restored");
    tsr_trace_free(&v);

    // cmd.bin inconsistent with init -> rejected
    char * cmdbin = pjoin(dir, "cmd.bin");
    flip_byte(cmdbin, 33, 0x01);
    CHECK(tsr_trace_read(dir, &v) != 0, "cmd.bin inconsistent with init accepted");
    flip_byte(cmdbin, 33, 0x01);

    // Corrupt one expect byte -> replay fails; restore -> passes.
    char name[96];
    snprintf(name, sizeof(name), "expect/ch1_%010llx.bin", (unsigned long long)((IO + 300) * 64));
    char * ef = pjoin(dir, name);
    flip_byte(ef, 7, 0x10);
    CHECK(tsr_trace_read(dir, &v) == 0, "read corrupted-expect trace");
    fprintf(stderr, "(expected replay mismatch report follows)\n");
    CHECK(tsr_trace_replay(&v, 0) == 1, "corrupted expect byte detected");
    tsr_trace_free(&v);
    flip_byte(ef, 7, 0x10);
    // Corrupt one weight quant byte (ch2 init) -> outputs differ -> replay fails.
    snprintf(name, sizeof(name), "init/ch2_%010llx.bin", (unsigned long long)(WOFF * 64));
    char * wf = pjoin(dir, name);
    flip_byte(wf, 64 + 5, 0x40);
    CHECK(tsr_trace_read(dir, &v) == 0, "read corrupted-init trace");
    fprintf(stderr, "(expected replay mismatch report follows)\n");
    CHECK(tsr_trace_replay(&v, 0) == 1, "corrupted weight byte detected");
    tsr_trace_free(&v);
    flip_byte(wf, 64 + 5, 0x40);
    // Expected final state differs -> fails.
    CHECK(tsr_trace_read(dir, &v) == 0, "read");
    v.status = DONEERR;
    fprintf(stderr, "(expected replay mismatch report follows)\n");
    CHECK(tsr_trace_replay(&v, 0) == 1, "status mismatch detected");
    v.status = DONE;
    // Missing expect region -> the run writes outside the expect set -> fails.
    v.cap.n_expect--;
    fprintf(stderr, "(expected replay mismatch report follows)\n");
    CHECK(tsr_trace_replay(&v, 0) == 1, "write outside expect detected");
    v.cap.n_expect++;
    // Missing init region (ch1 weights) -> the run reads uninitialised DRAM -> fails.
    tsr_region saved = v.cap.init[3];
    v.cap.init[3] = v.cap.init[v.cap.n_init - 1];
    v.cap.n_init--;
    fprintf(stderr, "(expected replay mismatch report follows)\n");
    CHECK(tsr_trace_replay(&v, 0) == 1, "read of uninitialised DRAM detected");
    v.cap.n_init++;
    v.cap.init[v.cap.n_init - 1] = v.cap.init[3];
    v.cap.init[3] = saved;
    CHECK(tsr_trace_replay(&v, 0) == 0, "restored trace passes");
    // Missing init region whose data never reaches an output (ch2 IO+50 is loaded to SPM 24 and
    // never stored): only the uninitialised-read check can see it.
    saved = v.cap.init[4];
    v.cap.init[4] = v.cap.init[v.cap.n_init - 1];
    v.cap.n_init--;
    fprintf(stderr, "(expected replay mismatch report follows)\n");
    CHECK(tsr_trace_replay(&v, 0) == 1, "uninitialised read without output effect detected");
    v.cap.n_init++;
    v.cap.init[v.cap.n_init - 1] = v.cap.init[4];
    v.cap.init[4] = saved;
    // Perf differences are only a warning.
    v.perf.cycles += 1;
    fprintf(stderr, "(expected perf warning follows)\n");
    CHECK(tsr_trace_replay(&v, 0) == 0, "perf mismatch is not a failure");
    tsr_trace_free(&v);

    // Rewriting into the same directory replaces the old trace (no stale region files).
    tsr_trace small;
    memset(&small, 0, sizeof(small));
    small.cfg = cfg;
    CHECK(tsr_trace_write(dir, &small) == 0, "rewrite");
    char * stale = pjoin(dir, "init");
    snprintf(name, sizeof(name), "%s/ch0_%010llx.bin", stale, (unsigned long long)((cmd - 1) * 64));
    CHECK(access(name, F_OK) != 0, "stale region file removed");
    CHECK(tsr_trace_read(dir, &v) == 0 && v.cmd_count == 0 && v.cap.n_init == 0, "rewritten trace");
    tsr_trace_free(&v);

    free(stale); free(wf); free(ef); free(cmdbin); free(spaced); free(mini); free(text); free(meta); free(dir);
    free(pre_ch0); free(pre_ch2);
    tsr_trace_free(&rd);
    tsr_capture_free(&cap);
    prog_free(&p);
    tsr_sim_destroy(s);
}

// ---------------------------------------------------------------------------
static void test_error_trace(void) {
    tsr_config cfg = cfg_u200(1);
    cfg.num_ch = 2;
    tsr_sim * s = tsr_sim_create(&cfg, NULL);
    prog_t p = {0};
    prog_add(&p, tsr_enc_li(3, (uint32_t)(IO + 5)));
    tsr_insn ld = tsr_enc_ld(1, 0, 2, 3);
    tsr_insn_set_base(&ld, 0, 3, 0);
    prog_add(&p, ld);
    prog_add(&p, tsr_enc_st(1, 0, (uint32_t)(IO + 100), 3));
    prog_add(&p, tsr_enc_mvm(0, 0, 0, 0, (uint32_t)WOFF, 4, 48, 1, 0));   // K_RANGE
    prog_add(&p, tsr_enc_end());
    rng_t r = { 3 };
    rng_bytes(&r, tsr_sim_dram(s, 1) + (IO + 7) * 64, 3 * 64);
    prog_load(s, &p, 0);
    const uint32_t args[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };
    tsr_sim_capture_begin(s);
    const uint32_t st = tsr_sim_exec(s, 0, p.n, args);
    tsr_capture cap;
    tsr_sim_capture_end(s, &cap);
    CHECK(st == DONEERR && sim_err_code(s) == TSR_ERR_K_RANGE && sim_err_pc(s) == 3, "error program");
    tsr_trace tr;
    trace_from(&tr, s, 0, p.n, args);
    tr.cap = cap;
    char * dir = pjoin(g_base, "err");
    CHECK(tsr_trace_write(dir, &tr) == 0, "write error trace");
    tsr_trace v;
    CHECK(tsr_trace_read(dir, &v) == 0 && traces_equal(&tr, &v), "read error trace");
    CHECK(v.status == DONEERR && v.err_code == TSR_ERR_K_RANGE && v.err_pc == 3, "error fields");
    CHECK(tsr_trace_replay(&v, 0) == 0, "error trace replays");
    v.err_pc = 2;
    fprintf(stderr, "(expected replay mismatch report follows)\n");
    CHECK(tsr_trace_replay(&v, 0) == 1, "ERR_PC mismatch detected");
    v.err_pc = 3;
    v.err_info = 9;
    fprintf(stderr, "(expected replay mismatch report follows)\n");
    CHECK(tsr_trace_replay(&v, 0) == 1, "ERR_INFO mismatch detected");
    v.has_err_info = 0;
    CHECK(tsr_trace_replay(&v, 0) == 0, "ERR_INFO not compared when absent");
    v.args[0] = 99;   // R8 differs; the program does not use it -> still passes
    CHECK(tsr_trace_replay(&v, 0) == 0, "unused arg");
    tsr_trace_free(&v);
    tsr_capture_free(&cap);
    free(dir);
    prog_free(&p);
    tsr_sim_destroy(s);
}

// ---------------------------------------------------------------------------
// Capture of an MVM over ~150 MB of weights: range-level capture, one region per channel.
static void test_big_capture(void) {
    tsr_config cfg = cfg_u200(1);
    tsr_sim * s = tsr_sim_create(&cfg, NULL);
    tsr_sim_set_threads(s, 4);
    const uint32_t fmt = TSR_FMT_Q4_K, N = 8192, K = 32768;
    const uint64_t rb = tsr_row_beats(fmt, K), per_ch = tsr_tensor_ch_beats(fmt, N, K, 4);
    rng_t r = { 8 };
    // Any bytes are a valid Q4_K row image; fill the four channel ranges directly.
    for (uint32_t c = 0; c < 4; ++c) {
        uint64_t * w = (uint64_t *)(void *)(tsr_sim_dram(s, c) + WOFF * 64);
        for (uint64_t i = 0; i < per_ch * 8; ++i) w[i] = rng_u64(&r) & 0xFBFFFBFFFBFFFBFFull;   // keep fp16 finite-ish
    }
    float * x = (float *)malloc(sizeof(float) * K);
    gen_f32(&r, x, K);
    put_f32_le(tsr_sim_spm(s), x, K);
    prog_t p = {0};
    prog_add(&p, tsr_enc_qact(TSR_QA_A256, 3000, 0, K, 1, 0));
    prog_add(&p, tsr_enc_mvm(fmt, 0, 4000, 3000, (uint32_t)WOFF, N, K, 1, 0));
    prog_add(&p, tsr_enc_st(3, 4000, (uint32_t)IO, N * 4 / 64));
    prog_add(&p, tsr_enc_end());
    prog_load(s, &p, 0);
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    tsr_sim_capture_begin(s);
    const uint32_t st = tsr_sim_exec(s, 0, p.n, NULL);
    tsr_capture cap;
    tsr_sim_capture_end(s, &cap);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    const double secs = (double)(t1.tv_sec - t0.tv_sec) + 1e-9 * (double)(t1.tv_nsec - t0.tv_nsec);
    printf("big capture: %.1f MB of weights, run + capture %.2f s\n", (double)(per_ch * 4 * 64) / 1e6, secs);
    CHECK(st == DONE, "big MVM status %u", st);
    CHECK(cap.n_init == 5, "big: n_init %zu (cmd + 4 weight ranges)", cap.n_init);
    for (uint32_t c = 0; c < 4 && cap.n_init == 5; ++c) {
        const tsr_region * g = &cap.init[c + 1];
        CHECK(region_is(g, c, WOFF, N / 4 * rb), "big: ch%u weight range", c);
        CHECK(memcmp(g->data, tsr_sim_dram(s, c) + WOFF * 64, (size_t)g->size) == 0, "big: ch%u data", c);
    }
    CHECK(cap.n_expect == 1 && region_is(&cap.expect[0], 3, IO, N * 4 / 64), "big: expect");
    tsr_capture_free(&cap);
    free(x);
    prog_free(&p);
    tsr_sim_destroy(s);
}

int main(int argc, char ** argv) {
    g_base = make_scratch(argc > 1 ? argv[1] : ".", "sim_trace");
    test_capture_and_roundtrip();
    test_error_trace();
    test_big_capture();
    const int rc = test_finish("sim_trace");
    if (rc == 0) rm_rf(g_base);
    return rc;
}
