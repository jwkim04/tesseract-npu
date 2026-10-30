// test_trace.c — trace library unit tests (ctest sim_trace and sim_trace_ubsan; docs/design/06-
// software-verification.md "SW" §5, §10.2). What it checks:
//   - the EX §3.2 image id: SHA-256 known answers (digests computed with Python's hashlib), word
//     padding, .tsr_id taken as zero, and the EX §3.1 image checks (TCM ranges, alignment, overlap);
//   - field-by-field write / read round trips of a kind-ops and a kind-launch trace (every meta
//     field, push records of the three kinds, tickets.bin, regions, the shared images directory
//     and the relative image path, ops.load placement, a negative RET, perf);
//   - strict validation on read: each corruption of meta.json, push.bin, tickets.bin, a region
//     file or the image is refused with a message naming it;
//   - relative tickets, tsr_trace2_ticket_of / tsr_trace2_push_unit;
//   - known-answer bytes of push.bin and tickets.bin (SW §5.2 / §5.3 offsets);
//   - tile-only runs (tsr_trace2_run_ops fills done / tickets / expect, also on a trace that is
//     already filled) and replays: exact error
//     records of a DRAM_RANGE / ABORTED stream, the replay of the written trace passing under the
//     fixed and two random schedules, and every comparison class failing on its corruption; the
//     stuck-stream diagnostics (a hold that is never released).
// Usage: sim_trace <scratch dir>
#include "test_sim_util.h"

#include "tsr_sim_int.h"
#include "tsr_trace.h"

static const char * g_base = "/tmp";

// ---------------------------------------------------------------------------
// Test ELF images (written independently of the library's own synthesis)
// ---------------------------------------------------------------------------
typedef struct { uint32_t paddr; const uint8_t * data; uint32_t filesz, memsz; } tseg;

static void put32(uint8_t * p, uint32_t v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24); }
static void put16(uint8_t * p, uint32_t v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); }

// ELF32 RISC-V EXEC with one PT_LOAD per segment; with id_seg >= 0 also section headers naming
// ".tsr_id" = 8 bytes at id_off inside segment id_seg.
static uint8_t * make_elf(const tseg * s, uint32_t n, uint32_t entry, int id_seg, uint32_t id_off, size_t * out_n) {
    static const char strtab[] = "\0.shstrtab\0.tsr_id\0";
    size_t data_off = 52 + 32u * n, size = data_off;
    for (uint32_t i = 0; i < n; ++i) size += s[i].filesz;
    const size_t str_off = size;
    const size_t sh_off = (str_off + sizeof(strtab) + 3u) & ~(size_t)3u;
    if (id_seg >= 0) size = sh_off + 3u * 40u;
    uint8_t * b = (uint8_t *)calloc(size, 1);
    memcpy(b, "\177ELF", 4);
    b[4] = 1; b[5] = 1; b[6] = 1;
    put16(b + 16, 2); put16(b + 18, 243); put32(b + 20, 1); put32(b + 24, entry); put32(b + 28, 52);
    put16(b + 40, 52); put16(b + 42, 32); put16(b + 44, n); put16(b + 46, 40);
    size_t off = data_off, seg_off[8] = { 0 };
    for (uint32_t i = 0; i < n; ++i) {
        uint8_t * ph = b + 52 + 32u * i;
        put32(ph, 1); put32(ph + 4, (uint32_t)off); put32(ph + 8, s[i].paddr); put32(ph + 12, s[i].paddr);
        put32(ph + 16, s[i].filesz); put32(ph + 20, s[i].memsz); put32(ph + 24, 7); put32(ph + 28, 4);
        memcpy(b + off, s[i].data, s[i].filesz);
        seg_off[i] = off;
        off += s[i].filesz;
    }
    if (id_seg >= 0) {
        memcpy(b + str_off, strtab, sizeof(strtab));
        put32(b + 32, (uint32_t)sh_off); put16(b + 48, 3); put16(b + 50, 1);
        uint8_t * sh = b + sh_off + 40;                                   // [1] .shstrtab
        put32(sh, 1); put32(sh + 4, 3); put32(sh + 16, (uint32_t)str_off); put32(sh + 20, sizeof(strtab));
        sh += 40;                                                         // [2] .tsr_id
        put32(sh, 11); put32(sh + 4, 1); put32(sh + 12, s[id_seg].paddr + id_off);
        put32(sh + 16, (uint32_t)(seg_off[id_seg] + id_off)); put32(sh + 20, 8);
    }
    *out_n = size;
    return b;
}

static void test_image_id(void) {
    char err[256];
    uint64_t id = 0;
    // KAT 1: ITCM 0x0 bytes 01..08, DTCM 0x100000 "hello" (padded to 8 with zeros).
    static const uint8_t a[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };
    const tseg s1[2] = { { 0x0, a, 8, 8 }, { 0x100000, (const uint8_t *)"hello", 5, 64 } };
    size_t n = 0;
    uint8_t * e = make_elf(s1, 2, 0x80, -1, 0, &n);
    CHECK(tsr_trace2_image_id(e, n, &id, err, sizeof(err)) == 0, "image id: %s", err);
    CHECK(id == 0x0ad6c4d1c6e6abecull, "image id KAT 1: %016llx", (unsigned long long)id);
    free(e);
    // KAT 2: .tsr_id (8 bytes at +8 of the DTCM segment) taken as zero, whatever it holds.
    static const uint8_t it[12] = { 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x1B };
    uint8_t dt[16] = { 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 0xDE, 0xAD, 0xBE, 0xEF, 1, 2, 3, 4 };
    const tseg s2[2] = { { 0x80, it, 12, 12 }, { 0x100000, dt, 16, 16 } };
    e = make_elf(s2, 2, 0x80, 1, 8, &n);
    CHECK(tsr_trace2_image_id(e, n, &id, err, sizeof(err)) == 0, "image id 2: %s", err);
    CHECK(id == 0x2d1e6b5fc49bba80ull, "image id KAT 2 (.tsr_id zeroed): %016llx", (unsigned long long)id);
    free(e);
    memset(dt + 8, 0x77, 8);
    e = make_elf(s2, 2, 0x80, 1, 8, &n);
    uint64_t id2 = 0;
    CHECK(tsr_trace2_image_id(e, n, &id2, err, sizeof(err)) == 0 && id2 == id, "the .tsr_id bytes do not change the id");
    free(e);
    e = make_elf(s2, 2, 0x80, -1, 0, &n);                                 // same bytes, no .tsr_id: differs
    CHECK(tsr_trace2_image_id(e, n, &id2, err, sizeof(err)) == 0 && id2 != id, "without .tsr_id every byte counts");
    free(e);
    // EX §3.1 refusals.
    const tseg bad1[1] = { { 0x2, a, 8, 8 } };                            // not word aligned
    const tseg bad2[1] = { { 0x0FFFFC, a, 8, 8 } };                       // between the TCMs
    const tseg bad3[2] = { { 0x0, a, 8, 8 }, { 0x4, a, 8, 8 } };          // overlap
    const tseg * bads[3] = { bad1, bad2, bad3 };
    const uint32_t nb[3] = { 1, 1, 2 };
    const char * want[3] = { "aligned", "inside ITCM", "overlap" };
    for (int i = 0; i < 3; ++i) {
        e = make_elf(bads[i], nb[i], 0x80, -1, 0, &n);
        err[0] = '\0';
        CHECK(tsr_trace2_image_id(e, n, &id, err, sizeof(err)) != 0 && strstr(err, want[i]) != NULL,
              "bad image %d refused with '%s' (got '%s')", i, want[i], err);
        free(e);
    }
    static const uint8_t junk[64] = { 'x' };
    CHECK(tsr_trace2_image_id(junk, sizeof(junk), &id, err, sizeof(err)) != 0 && strstr(err, "not an ELF") != NULL, "junk refused");
}

// ---------------------------------------------------------------------------
// Field-by-field comparison
// ---------------------------------------------------------------------------
static int regions_eq(const tsr_region * a, size_t na, const tsr_region * b, size_t nb) {
    if (na != nb) return 0;
    for (size_t i = 0; i < na; ++i)
        if (a[i].ch != b[i].ch || a[i].byte_off != b[i].byte_off || a[i].size != b[i].size ||
            memcmp(a[i].data, b[i].data, (size_t)a[i].size) != 0)
            return 0;
    return 1;
}

static void check_eq(const tsr_trace2 * a, const tsr_trace2 * b, const char * tag) {
    CHECK(a->kind == b->kind, "%s: kind", tag);
    const tsr_config * x = &a->cfg.core, * y = &b->cfg.core;
    CHECK(x->num_ch == y->num_ch && x->beat_bits == y->beat_bits && x->lanes == y->lanes && x->t_max == y->t_max &&
          x->k_max == y->k_max && x->spm_bytes == y->spm_bytes && x->vl == y->vl && x->nsreg == y->nsreg &&
          x->rd_outstanding == y->rd_outstanding && x->ch_bytes == y->ch_bytes && x->addr_beat_w == y->addr_beat_w &&
          x->caps == y->caps, "%s: config core", tag);
    CHECK(a->cfg.ctl_id == b->cfg.ctl_id && a->cfg.cfg3 == b->cfg.cfg3 && a->cfg.itcm_bytes == b->cfg.itcm_bytes &&
          a->cfg.dtcm_bytes == b->cfg.dtcm_bytes && memcmp(a->cfg.q_depth, b->cfg.q_depth, sizeof(a->cfg.q_depth)) == 0,
          "%s: config extras", tag);
    CHECK(a->image_id == b->image_id && a->image_size == b->image_size &&
          (a->image_size == 0 || memcmp(a->image, b->image, a->image_size) == 0), "%s: image", tag);
    CHECK(strcmp(a->kernel, b->kernel) == 0 && a->kernel_index == b->kernel_index && a->prof == b->prof &&
          memcmp(a->args, b->args, sizeof(a->args)) == 0, "%s: launch", tag);
    CHECK(a->n_push == b->n_push && (a->n_push == 0 || memcmp(a->push, b->push, a->n_push * sizeof(tsr_trace_push)) == 0),
          "%s: push records (%zu / %zu)", tag, a->n_push, b->n_push);
    if (a->kind == TSR_TRACE2_KIND_OPS && a->load_ch != TSR_TRACE2_LOAD_AUTO)
        CHECK(a->load_ch == b->load_ch && a->load_offset == b->load_offset, "%s: ops.load", tag);
    CHECK(memcmp(&a->done, &b->done, sizeof(a->done)) == 0, "%s: done", tag);
    CHECK(regions_eq(a->mem.init, a->mem.n_init, b->mem.init, b->mem.n_init), "%s: init regions", tag);
    CHECK(regions_eq(a->mem.expect, a->mem.n_expect, b->mem.expect, b->mem.n_expect), "%s: expect regions", tag);
    CHECK(a->n_units == b->n_units && memcmp(a->unit, b->unit, sizeof(a->unit)) == 0, "%s: tickets", tag);
    CHECK(strcmp(a->sched, b->sched) == 0 && a->rv_ratio == b->rv_ratio, "%s: sched / rv_ratio", tag);
    CHECK(a->perf.valid == b->perf.valid &&
          (!a->perf.valid || (a->perf.core_cycles == b->perf.core_cycles && a->perf.rv_instret == b->perf.rv_instret &&
                              memcmp(a->perf.unit_busy, b->perf.unit_busy, sizeof(a->perf.unit_busy)) == 0 &&
                              memcmp(a->perf.ch_rd_beats, b->perf.ch_rd_beats, a->cfg.core.num_ch * sizeof(uint64_t)) == 0)),
          "%s: perf", tag);
}

static void fill_bytes(uint8_t * p, size_t n, uint32_t seed) {
    rng_t r = { 0x1234567ull + seed };
    rng_bytes(&r, p, n);
}

static tsr_config small_cfg(void) {
    tsr_config c = TSR_CONFIG_U200_V1;
    c.num_ch = 2;
    c.ch_bytes = 1ull << 26;
    return c;
}

// A kind-ops trace with every field set to something distinctive.
static tsr_trace2 * sample_ops(void) {
    tsr_trace2 * t = tsr_trace2_new();
    const tsr_config c = small_cfg();
    tsr_trace2_config_from_core(&t->cfg, &c);
    t->cfg.ctl_id = 0x0102031Fu;
    t->cfg.cfg3 = 0x00440011u;
    t->cfg.dtcm_bytes = 32768;
    uint32_t w[8] = { tsr_w0(TSR_OP_LD, 1, 0, 0), 7, 100, 3, 0, 0, 0, 0 };
    tsr_trace2_add_op(t, w, TSR_TICKET_NONE, TSR_TICKET_NONE, TSR_ISSUE_PUSH_WMASK);
    uint32_t w2[8] = { tsr_w0(TSR_OP_ST, 0, 0, 0), 7, 200, 3, 0xAAAAAAAAu, 0xBBBBBBBBu, 0, 0 };
    tsr_trace2_add_op(t, w2, 0x00000000u, 0xF0000000u, 0x0F01u);       // words 4, 5 masked -> 0, REC
    tsr_trace2_add_qctrl(t, TSR_QCTRL_QCLEAR);
    tsr_trace2_add_mark(t, 0xC0DE0001u);
    t->done.status = 1; t->done.ret = -5; t->done.err_code = 7; t->done.err_ticket_rel = 0x00000002u;
    t->done.err_pidx_rel = 2; t->done.err_info = 0x123; t->done.err_w0 = 0x456;
    uint8_t buf[4096];
    fill_bytes(buf, sizeof(buf), 1);
    tsr_trace2_add_init(t, 0, 0, buf, 192);                              // OPL bottom: forces the load above it
    tsr_trace2_add_init(t, 1, 0x40000, buf + 192, 1024);
    fill_bytes(buf, sizeof(buf), 2);
    tsr_trace2_add_expect(t, 0, 0x8000, buf, 4096);
    t->unit[0].count = 2; t->unit[0].err_code = 7; t->unit[0].err_rel_seq = 1; t->unit[0].err_info = 0x123;
    t->unit[0].err_w0 = 0x456; t->unit[0].err_rel_pidx = 2;
    t->unit[5].count = 1;
    snprintf(t->sched, sizeof(t->sched), "random:7,0.25,4");
    t->rv_ratio = 166.7 / 276.3;
    t->perf.valid = 1; t->perf.core_cycles = 1234567; t->perf.rv_instret = 42;
    for (uint32_t u = 0; u < TSR_NUM_UNITS; ++u) t->perf.unit_busy[u] = 10u * u;
    t->perf.ch_rd_beats[0] = 3; t->perf.ch_rd_beats[1] = 4;
    return t;
}

// ---------------------------------------------------------------------------
// Files: text helpers for corruption tests
// ---------------------------------------------------------------------------
static char * slurp(const char * path, size_t * n) {
    FILE * f = fopen(path, "rb");
    if (f == NULL) return NULL;
    fseek(f, 0, SEEK_END);
    const long L = ftell(f);
    fseek(f, 0, SEEK_SET);
    char * b = (char *)malloc((size_t)L + 1);
    const size_t got = L > 0 ? fread(b, 1, (size_t)L, f) : 0;
    b[got] = 0;
    fclose(f);
    *n = got;
    return b;
}
static void spit(const char * path, const void * b, size_t n) {
    FILE * f = fopen(path, "wb");
    if (f) {
        fwrite(b, 1, n, f);
        fclose(f);
    }
}
// Replaces the first occurrence of `from` by `to` in a text file. 0 if done.
static int file_replace(const char * path, const char * from, const char * to) {
    size_t n = 0;
    char * b = slurp(path, &n);
    if (b == NULL) return -1;
    char * p = strstr(b, from);
    if (p == NULL) { free(b); return -1; }
    const size_t lf = strlen(from), lt = strlen(to);
    char * o = (char *)malloc(n - lf + lt + 1);
    memcpy(o, b, (size_t)(p - b));
    memcpy(o + (p - b), to, lt);
    memcpy(o + (p - b) + lt, p + lf, n - (size_t)(p - b) - lf);
    spit(path, o, n - lf + lt);
    free(o);
    free(b);
    return 0;
}

static char * pj(const char * a, const char * b) {
    const size_t n = strlen(a) + strlen(b) + 2;
    char * p = (char *)malloc(n);
    snprintf(p, n, "%s/%s", a, b);
    return p;
}

static void copy_tree_file(const char * src_dir, const char * dst_dir, const char * rel) {
    char * s = pj(src_dir, rel), * d = pj(dst_dir, rel);
    size_t n = 0;
    char * b = slurp(s, &n);
    if (b) spit(d, b, n);
    free(b);
    free(s);
    free(d);
}

// Reads dir; expects failure with `want` in the message.
static void expect_read_fail(const char * dir, const char * want, const char * what) {
    tsr_trace2 * t = NULL;
    char err[600];
    err[0] = '\0';
    const int rc = tsr_trace2_read(dir, &t, err, sizeof(err));
    CHECK(rc != 0 && t == NULL, "%s: read must fail", what);
    CHECK(strstr(err, want) != NULL, "%s: message '%s' should mention '%s'", what, err, want);
    tsr_trace2_free(t);
}

// ---------------------------------------------------------------------------
// Round trips and validation
// ---------------------------------------------------------------------------
static void test_roundtrip_ops(const char * scratch) {
    tsr_trace2 * t = sample_ops();
    char err[600];
    CHECK(tsr_trace2_validate(t, err, sizeof(err)) == 0, "sample validates: %s", err);
    char * dir = pj(scratch, "ops0");
    CHECK(tsr_trace2_write(t, dir, NULL) == 0, "write ops trace");
    tsr_trace2 * r = NULL;
    CHECK(tsr_trace2_read(dir, &r, err, sizeof(err)) == 0, "read ops trace: %s", err);
    if (r != NULL) {
        CHECK(r->load_ch == 0 && r->load_offset == 0x1000, "ops.load chosen above the init region at 0: ch %u 0x%llx",
              r->load_ch, (unsigned long long)r->load_offset);
        CHECK(r->push[1].w[4] == 0 && r->push[1].w[5] == 0 && r->push[1].w[3] == 3, "masked words stored as 0");
        check_eq(t, r, "ops round trip");
        // Copy: deep and equal; then write the copy again (existing directory: replaced).
        tsr_trace2 cp;
        tsr_trace2_init(&cp);
        tsr_trace2_copy(&cp, r);
        check_eq(r, &cp, "copy");
        CHECK(tsr_trace2_write(&cp, dir, NULL) == 0, "rewrite in place");
        tsr_trace2_clear(&cp);
        tsr_trace2 * r2 = NULL;
        CHECK(tsr_trace2_read(dir, &r2, err, sizeof(err)) == 0, "re-read: %s", err);
        if (r2) check_eq(r, r2, "rewrite");
        tsr_trace2_free(r2);
    }
    // ticket_of / push_unit
    CHECK(tsr_trace2_ticket_of(t, 0) == 0x00000000u && tsr_trace2_ticket_of(t, 1) == 0x00000001u &&
          tsr_trace2_ticket_of(t, 2) == ((uint32_t)TSR_UNIT_SYNC << 28) && tsr_trace2_ticket_of(t, 3) == TSR_TICKET_NONE,
          "tickets of LD, ST, QCLEAR, mark");
    tsr_trace_push p;
    memset(&p, 0, sizeof(p));
    p.w[0] = 0x04;                                                       // reserved opcode
    CHECK(tsr_trace2_push_unit(&t->cfg, &p) == TSR_UNIT_SYNC, "a refused opcode goes to SYNC");
    p.w[0] = TSR_OP_VOP;
    CHECK(tsr_trace2_push_unit(&t->cfg, &p) == TSR_UNIT_SYNC, "VOP without CAPS.VU goes to SYNC");
    p.kind = TSR_TRACE_PUSH_QCTRL; p.w[0] = TSR_QCTRL_QHOLD_SET;
    CHECK(tsr_trace2_push_unit(&t->cfg, &p) == TSR_UNIT_NONE, "QHOLD has no ticket");
    // Relative tickets.
    const uint32_t base[TSR_NUM_UNITS] = { 5, 0, 0, 0x0FFFFFFF, 0, 9 };
    CHECK(tsr_trace_rel_ticket(0x00000007u, base) == 0x00000002u && tsr_trace_abs_ticket(0x00000002u, base) == 0x00000007u, "rel DMA");
    CHECK(tsr_trace_rel_ticket(0x30000001u, base) == 0x30000002u && tsr_trace_abs_ticket(0x30000002u, base) == 0x30000001u, "rel wraps mod 2^28");
    CHECK(tsr_trace_rel_ticket(0x50000003u, base) == 0x5FFFFFFAu, "rel below the base");
    CHECK(tsr_trace_rel_ticket(0xF0000000u, base) == 0xF0000000u && tsr_trace_rel_ticket(0x7000ABCDu, base) == 0x7000ABCDu,
          "units >= 6 unchanged");

    // --- corruptions on disk, each on a fresh copy ---
    static const char * const files[] = { "meta.json", "ops/push.bin", "expect/tickets.bin", "init/ch0_0000000000.bin",
                                          "init/ch1_0000040000.bin", "expect/ch0_0000008000.bin" };
    char * bad = pj(scratch, "ops_bad");
    char * sub_i = pj(bad, "init"), * sub_e = pj(bad, "expect"), * sub_o = pj(bad, "ops");
    struct { const char * file, * from, * to, * want, * what; } edits[] = {
        { "meta.json", "\"version\": 2", "\"version\": 1", "version 1", "v1 trace" },
        { "meta.json", "\"version\": 2", "\"version\": 3", "unsupported version", "version 3" },
        { "meta.json", "\"kind\": \"ops\"", "\"kind\": \"opz\"", "\"kind\"", "bad kind" },
        { "meta.json", "\"format\": \"tesseract-trace\",", "\"format\": \"tesseract-trace\", \"bogus\": 1,", "unknown key", "unknown key" },
        { "meta.json", "\"rd_outstanding\": 16, ", "", "missing \"rd_outstanding\"", "missing field" },
        { "meta.json", "\"q_depth\": [8, 4, 4, 16, 0, 4]", "\"q_depth\": [8, 4, 4, 16, 0, 0]", "SYNC", "no SYNC unit" },
        { "meta.json", "\"ret\": -5", "\"ret\": 4294967296", "ret", "ret out of range" },
        { "meta.json", "\"offset\": 4096}", "\"offset\": 64}", "overlaps", "load overlapping init" },
        { "meta.json", "\"count\": 4", "\"count\": 5", "48 bytes per record", "count vs push.bin" },
        { "meta.json", "\"offset\": 262144, \"size\": 1024", "\"offset\": 262145, \"size\": 1024", "64-byte aligned", "misaligned region" },
        { "meta.json", "\"offset\": 262144, \"size\": 1024", "\"offset\": 262144, \"size\": 960", "bytes", "region size vs file" },
        { "meta.json", "\"offset\": 32768", "\"offset\": 67108864", "inside a channel", "region beyond the channel" },
        { "meta.json", "\"sched\": \"random:7,0.25,4\"", "\"sched\": \"\"", "sched", "empty sched" },
        { "meta.json", "\"perf\": {\"core_cycles\": 1234567,", "\"perf\": {\"core_cycles\": -1,", "core_cycles", "negative perf" },
        { "meta.json", "\"ops\": {", "\"image\": {\"id\": \"0000000000000000\", \"file\": \"x.elf\"}, \"ops\": {", "no \"image\"", "image in kind ops" },
        { "meta.json", "{\"ch\": 0, \"offset\": 0, \"size\": 192, \"file\": \"init/ch0_0000000000.bin\"}",
          "{\"ch\": 0, \"offset\": 0, \"size\": 192, \"file\": \"../init/ch0_0000000000.bin\"}", "inside the trace", "path escape" },
    };
    for (size_t k = 0; k < sizeof(edits) / sizeof(edits[0]); ++k) {
        rm_rf(bad);
        mkdir(bad, 0755); mkdir(sub_i, 0755); mkdir(sub_e, 0755); mkdir(sub_o, 0755);
        for (size_t f = 0; f < sizeof(files) / sizeof(files[0]); ++f) copy_tree_file(dir, bad, files[f]);
        char * path = pj(bad, edits[k].file);
        CHECK(file_replace(path, edits[k].from, edits[k].to) == 0, "%s: edit applies", edits[k].what);
        free(path);
        expect_read_fail(bad, edits[k].want, edits[k].what);
    }
    // Binary corruptions.
    struct { const char * file; long off; int len; uint32_t val; const char * want, * what; } bins[] = {
        { "ops/push.bin", 44, 4, 7, "kind 7", "push kind" },
        { "ops/push.bin", 48 + 16, 4, 1, "masked", "masked word not 0" },
        { "ops/push.bin", 96 + 4, 4, 1, "w[1..7]", "QCTRL record with words" },
        { "expect/tickets.bin", 4, 4, 1, "header", "tickets header reserved" },
        { "expect/tickets.bin", 16 + 24, 4, 1, "reserved", "tickets unit reserved" },
        { "expect/tickets.bin", 16 + 32 + 12, 4, 3, "error fields", "tickets: info without code" },
        { "expect/tickets.bin", 16 + 4 * 32, 4, 1, "not built", "tickets: absent unit with entries" },
        { "expect/tickets.bin", 0, 4, 9, "n_units", "tickets n_units 9" },
        { "expect/tickets.bin", -1, 0, 0, "tickets.bin", "tickets truncated" },
        { "ops/push.bin", -1, 0, 0, "48 bytes", "push.bin truncated" },
    };
    for (size_t k = 0; k < sizeof(bins) / sizeof(bins[0]); ++k) {
        rm_rf(bad);
        mkdir(bad, 0755); mkdir(sub_i, 0755); mkdir(sub_e, 0755); mkdir(sub_o, 0755);
        for (size_t f = 0; f < sizeof(files) / sizeof(files[0]); ++f) copy_tree_file(dir, bad, files[f]);
        char * path = pj(bad, bins[k].file);
        size_t n = 0;
        char * b = slurp(path, &n);
        if (bins[k].off < 0) {
            spit(path, b, n - 1);                                          // truncated by one byte
        } else if ((size_t)bins[k].off + 4 <= n) {
            put32((uint8_t *)b + bins[k].off, bins[k].val);
            spit(path, b, n);
        }
        free(b);
        free(path);
        expect_read_fail(bad, bins[k].want, bins[k].what);
    }
    rm_rf(bad);
    // Missing files.
    {
        char * missing = pj(dir, "expect/tickets.bin");
        size_t n = 0;
        char * keep = slurp(missing, &n);
        unlink(missing);
        expect_read_fail(dir, "tickets", "missing tickets.bin");
        spit(missing, keep, n);
        free(keep);
        free(missing);
    }
    // Validation of in-memory traces.
    {
        tsr_trace2 * v = sample_ops();
        v->mem.init[1].ch = 0; v->mem.init[1].byte_off = 64;               // overlaps init[0]
        CHECK(tsr_trace2_validate(v, err, sizeof(err)) != 0 && strstr(err, "overlap"), "init overlap: %s", err);
        char * never = pj(scratch, "never");
        CHECK(tsr_trace2_write(v, never, NULL) != 0, "write refuses an invalid trace");
        free(never);
        tsr_trace2_free(v);
        v = sample_ops();
        v->n_units = 0;
        CHECK(tsr_trace2_validate(v, err, sizeof(err)) != 0 && strstr(err, "n_units"), "n_units 0: %s", err);
        tsr_trace2_free(v);
        v = sample_ops();
        v->rv_ratio = 9.0;
        CHECK(tsr_trace2_validate(v, err, sizeof(err)) != 0 && strstr(err, "rv_ratio"), "rv_ratio: %s", err);
        tsr_trace2_free(v);
        v = sample_ops();
        v->kind = TSR_TRACE2_KIND_LAUNCH;                                  // no image
        CHECK(tsr_trace2_validate(v, err, sizeof(err)) != 0 && strstr(err, "image"), "launch without image: %s", err);
        tsr_trace2_free(v);
    }
    tsr_trace2_free(r);
    tsr_trace2_free(t);
    free(sub_i); free(sub_e); free(sub_o); free(bad); free(dir);
}

static void test_roundtrip_launch(const char * scratch) {
    // A launch trace with a test image; two traces of one set share images/<id>.elf.
    static const uint8_t code[16] = { 0x13, 0, 0, 0, 0x13, 0, 0, 0, 0x6F, 0, 0, 0, 0x13, 0, 0, 0 };
    static const uint8_t data[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };
    const tseg s[2] = { { 0x80, code, 16, 16 }, { 0x100000, data, 8, 4096 } };
    size_t n = 0;
    uint8_t * elf = make_elf(s, 2, 0x80, -1, 0, &n);
    tsr_trace2 * t = sample_ops();
    t->kind = TSR_TRACE2_KIND_LAUNCH;
    t->image = elf;
    t->image_size = n;
    char err[600];
    CHECK(tsr_trace2_image_id(elf, n, &t->image_id, err, sizeof(err)) == 0, "id: %s", err);
    snprintf(t->kernel, sizeof(t->kernel), "mvm_act");
    t->kernel_index = 3;
    t->prof = 1;
    for (int i = 0; i < 7; ++i) t->args[i] = 0x10000000u * (uint32_t)i + 17u;
    char * set = pj(scratch, "set");
    char * d0 = pj(set, "000003"), * d1 = pj(set, "000004");
    CHECK(tsr_trace2_write(t, d0, NULL) == 0, "write launch trace");
    t->args[0] = 99;
    CHECK(tsr_trace2_write(t, d1, NULL) == 0, "write a second launch with the same image");
    char name[64];
    snprintf(name, sizeof(name), "images/%016llx.elf", (unsigned long long)t->image_id);
    char * img = pj(set, name);
    struct stat st;
    CHECK(stat(img, &st) == 0 && (size_t)st.st_size == n, "image written once into <set>/images (%s)", img);
    char * meta = pj(d0, "meta.json");
    size_t ml = 0;
    char * mt = slurp(meta, &ml);
    char want[96];
    snprintf(want, sizeof(want), "\"file\": \"../images/%016llx.elf\"", (unsigned long long)t->image_id);
    CHECK(mt && strstr(mt, want) != NULL, "meta image.file is relative: %s", want);
    CHECK(mt && strstr(mt, "\"load\"") == NULL, "kind launch has no ops.load");
    free(mt);
    tsr_trace2 * r = NULL;
    CHECK(tsr_trace2_read(d1, &r, err, sizeof(err)) == 0, "read launch trace: %s", err);
    if (r) check_eq(t, r, "launch round trip");
    tsr_trace2_free(r);
    // Corruptions: a wrong id, a changed image, a missing image.
    {
        char idh[20];
        snprintf(idh, sizeof(idh), "%016llx", (unsigned long long)(t->image_id ^ 1u));
        size_t L = 0;
        char * b = slurp(meta, &L);
        char * p = b ? strstr(b, "\"id\": \"") : NULL;
        if (p) { memcpy(p + 7, idh, 16); spit(meta, b, L); }
        free(b);
        expect_read_fail(d0, "differs", "image id mismatch");
    }
    uint8_t * patched = (uint8_t *)malloc(n);
    memcpy(patched, elf, n);
    patched[n - 1] ^= 0xFF;                                              // a data byte of the DTCM segment
    char * img1 = pj(set, name);
    spit(img1, patched, n);
    expect_read_fail(d1, "differs", "image bytes changed");
    unlink(img1);
    expect_read_fail(d1, "image", "image missing");
    spit(img1, elf, n);
    t->image = patched;                                                  // same id field, other bytes
    t->image_size = n;
    CHECK(tsr_trace2_validate(t, err, sizeof(err)) != 0 && strstr(err, "differs"), "validate: image id vs bytes: %s", err);
    t->image = NULL;
    t->image_size = 0;
    free(patched);
    free(elf);
    tsr_trace2_free(t);
    free(img1); free(meta); free(img); free(d0); free(d1); free(set);
}

// ---------------------------------------------------------------------------
// Tile-only runs and replays
// ---------------------------------------------------------------------------
static int random_sched_available(void) {
    const tsr_config c = small_cfg();
    tsr_sim_opts o;
    memset(&o, 0, sizeof(o));
    o.sched = "random:1";
    tsr_sim * s = tsr_sim_create_opts(&c, &o);
    if (s == NULL) return 0;
    tsr_sim_destroy(s);
    return 1;
}

// LD ch0 -> SPM, ST SPM -> ch1 (dep), a mark, FENCE, an out-of-range ST (DRAM_RANGE), an LD
// behind it (ABORTED: its unit is in error state). No QCLEAR: the error record stays.
static tsr_trace2 * stream_b(uint64_t limit_beats) {
    tsr_trace2 * t = tsr_trace2_new();
    const tsr_config c = small_cfg();
    tsr_trace2_config_from_core(&t->cfg, &c);
    uint8_t buf[64 * 64];
    fill_bytes(buf, sizeof(buf), 3);
    tsr_trace2_add_init(t, 0, 1024u * 64u, buf, sizeof(buf));
    tsr_insn i = tsr_enc_ld(0, 0, 1024, 64);
    tsr_trace2_add_op(t, i.w, TSR_TICKET_NONE, TSR_TICKET_NONE, TSR_ISSUE_PUSH_WMASK);
    i = tsr_enc_st(1, 0, 2048, 64);
    tsr_trace2_add_op(t, i.w, tsr_trace2_ticket_of(t, 0), TSR_TICKET_NONE, TSR_ISSUE_PUSH_WMASK);
    tsr_trace2_add_mark(t, 0xFEED0002u);
    i = tsr_enc_fence();
    tsr_trace2_add_op(t, i.w, TSR_TICKET_NONE, TSR_TICKET_NONE, TSR_ISSUE_PUSH_WMASK);
    i = tsr_enc_st(1, 0, (uint32_t)limit_beats, 1);
    tsr_trace2_add_op(t, i.w, TSR_TICKET_NONE, TSR_TICKET_NONE, TSR_ISSUE_PUSH_WMASK);
    i = tsr_enc_ld(0, 64, 1024, 1);
    tsr_trace2_add_op(t, i.w, tsr_trace2_ticket_of(t, 4), TSR_TICKET_NONE, TSR_ISSUE_PUSH_WMASK);
    return t;
}

static void expect_replay_fail(const tsr_trace2 * t, uint32_t cls, const char * what) {
    tsr_trace2_result r;
    const int rc = tsr_trace2_replay_sim(t, NULL, NULL, &r);
    CHECK(rc == 1 && !r.ok && (r.fail_mask & cls), "%s: replay must fail with class 0x%x (rc %d, mask 0x%x: %s)", what, cls, rc,
          r.fail_mask, r.msg);
}

static void test_runs(const char * scratch) {
    const tsr_config c = small_cfg();
    const uint64_t limit = c.ch_bytes / 64u;
    tsr_trace2 * t = stream_b(limit);
    tsr_trace2_result res;
    tsr_trace2_replay_opts o;
    memset(&o, 0, sizeof(o));
    o.verbose = 1;
    const int rc = tsr_trace2_run_ops(t, &o, &res);
    CHECK(rc == 0 && res.ok, "run_ops: rc %d %s", rc, res.msg);
    // Expected from EX §8 / CC §9: DMA entries LD, ST, ST (DRAM_RANGE, ERR_INFO = its beat), LD
    // (ABORTED behind the error state); SYNC: FENCE. Push indices: the mark takes none.
    const tsr_insn st_bad = tsr_enc_st(1, 0, (uint32_t)limit, 1);
    CHECK(t->unit[TSR_UNIT_DMA].count == 4 && t->unit[TSR_UNIT_SYNC].count == 1 && t->unit[TSR_UNIT_MVM].count == 0,
          "counts DMA %u SYNC %u", t->unit[TSR_UNIT_DMA].count, t->unit[TSR_UNIT_SYNC].count);
    CHECK(t->unit[TSR_UNIT_DMA].err_code == TSR_ERR_DRAM_RANGE && t->unit[TSR_UNIT_DMA].err_rel_seq == 2 &&
          t->unit[TSR_UNIT_DMA].err_info == (uint32_t)limit && t->unit[TSR_UNIT_DMA].err_w0 == st_bad.w[0] &&
          t->unit[TSR_UNIT_DMA].err_rel_pidx == 3, "DMA error record: code %u seq %u info 0x%x pidx %u",
          t->unit[TSR_UNIT_DMA].err_code, t->unit[TSR_UNIT_DMA].err_rel_seq, t->unit[TSR_UNIT_DMA].err_info,
          t->unit[TSR_UNIT_DMA].err_rel_pidx);
    CHECK(t->done.status == TSR_LCQ_R0_STATUS_TILE_ERR && t->done.err_code == TSR_ERR_DRAM_RANGE && t->done.ret == 4 &&
          t->done.err_ticket_rel == 0x00000002u && t->done.err_pidx_rel == 3, "DONE as the replay kernel (SW §7.7): status %u "
          "code %u ret %d ticket 0x%08x pidx %u", t->done.status, t->done.err_code, t->done.ret, t->done.err_ticket_rel,
          t->done.err_pidx_rel);
    CHECK(t->mem.n_expect == 1 && t->mem.expect[0].ch == 1 && t->mem.expect[0].byte_off == 2048u * 64u &&
          t->mem.expect[0].size == 64u * 64u && memcmp(t->mem.expect[0].data, t->mem.init[0].data, 4096) == 0,
          "expect = the copied block on ch1");
    CHECK(t->perf.valid && t->perf.core_cycles > 0, "perf filled (busy counters stay 0: PERF counts only while ACTIVE)");
    char * dir = pj(scratch, "run_b");
    CHECK(tsr_trace2_write(t, dir, NULL) == 0, "write run trace");
    tsr_trace2 * r = NULL;
    char err[600];
    CHECK(tsr_trace2_read(dir, &r, err, sizeof(err)) == 0, "read: %s", err);
    if (r == NULL) {
        tsr_trace2_free(t);
        free(dir);
        return;
    }
    check_eq(t, r, "run trace round trip");
    // run_ops again on the trace read back, twice (it already holds expect regions: they are
    // replaced, not leaked or used after free; the ASan tree covers this path).
    {
        tsr_trace2 again;
        tsr_trace2_init(&again);
        tsr_trace2_copy(&again, r);
        for (int k = 0; k < 2; ++k) {
            CHECK(tsr_trace2_run_ops(&again, &o, &res) == 0 && res.ok, "run_ops on a filled trace (%d): %s", k, res.msg);
            check_eq(t, &again, "run_ops on a filled trace");
        }
        tsr_trace2_clear(&again);
    }
    CHECK(tsr_trace2_replay_sim(r, NULL, NULL, &res) == 0 && res.ok, "replay (fixed): %s", res.msg);
    if (random_sched_available()) {
        CHECK(tsr_trace2_replay_sim(r, NULL, "random:5", &res) == 0 && res.ok, "replay (random:5): %s", res.msg);
        CHECK(tsr_trace2_replay_sim(r, NULL, "random:6,0.5,8", &res) == 0 && res.ok, "replay (random:6,0.5,8): %s", res.msg);
    } else {
        printf("sim_trace: random schedules not available in this simulator; replayed under fixed only\n");
    }
    // Each comparison class fails on its corruption.
    tsr_trace2 x;
    tsr_trace2_init(&x);
    tsr_trace2_copy(&x, r);
    x.mem.expect[0].data[100] ^= 0x01;
    expect_replay_fail(&x, TSR_TRACE2_CMP_EXPECT, "one expect byte");
    tsr_trace2_copy(&x, r);
    x.push[1].w[2] = 3000;                                               // the ST writes elsewhere
    expect_replay_fail(&x, TSR_TRACE2_CMP_WRITTEN, "ST address changed");
    tsr_trace2_copy(&x, r);
    x.unit[TSR_UNIT_SYNC].count = 2;
    expect_replay_fail(&x, TSR_TRACE2_CMP_COUNTS, "tickets count");
    tsr_trace2_copy(&x, r);
    x.unit[TSR_UNIT_DMA].err_info ^= 1;
    expect_replay_fail(&x, TSR_TRACE2_CMP_ERRREC, "tickets error record");
    tsr_trace2_copy(&x, r);
    x.done.ret = 5;
    expect_replay_fail(&x, TSR_TRACE2_CMP_DONE, "DONE ret");
    tsr_trace2_copy(&x, r);
    free(x.mem.init[0].data);
    x.mem.n_init = 0;                                                    // the LD reads outside init
    expect_replay_fail(&x, TSR_TRACE2_CMP_READS, "init removed");
    tsr_trace2_clear(&x);
    tsr_trace2_free(r);
    tsr_trace2_free(t);

    // run_ops: a read outside init is reported; init_from_reads takes the regions read instead.
    t = stream_b(limit);
    free(t->mem.init[0].data);
    t->mem.n_init = 0;
    o.verbose = 0;
    CHECK(tsr_trace2_run_ops(t, &o, &res) == 1 && (res.fail_mask & TSR_TRACE2_CMP_READS), "run_ops: read outside init: %s", res.msg);
    o.init_from_reads = 1;
    CHECK(tsr_trace2_run_ops(t, &o, &res) == 0 && t->mem.n_init == 1, "run_ops init_from_reads: %zu init regions (%s)",
          t->mem.n_init, res.msg);                                       // the second LD is ABORTED: no read
    if (t->mem.n_init == 1)
        CHECK(t->mem.init[0].ch == 0 && t->mem.init[0].byte_off == 1024u * 64u && t->mem.init[0].size == 4096,
              "init = the beats the LDs read (zeros)");
    CHECK(tsr_trace2_replay_sim(t, NULL, NULL, &res) == 0, "replay after init_from_reads: %s", res.msg);
    tsr_trace2_free(t);

    // A hold never released: the 9th DMA push waits forever; a single held op never drains.
    t = tsr_trace2_new();
    tsr_trace2_config_from_core(&t->cfg, &c);
    tsr_trace2_add_qctrl(t, TSR_QCTRL_QHOLD_SET);
    for (int k = 0; k < 9; ++k) {
        const tsr_insn i = tsr_enc_ld(0, 0, 0, 1);
        tsr_trace2_add_op(t, i.w, TSR_TICKET_NONE, TSR_TICKET_NONE, TSR_ISSUE_PUSH_WMASK);
    }
    o.init_from_reads = 1;
    CHECK(tsr_trace2_run_ops(t, &o, &res) == 2 && (res.fail_mask & TSR_TRACE2_CMP_SETUP) && strstr(res.msg, "forever"),
          "held queue: %s", res.msg);
    t->n_push = 2;
    CHECK(tsr_trace2_run_ops(t, &o, &res) == 2 && strstr(res.msg, "not idle"), "held op: %s", res.msg);
    tsr_trace2_free(t);
    free(dir);
}

// ---------------------------------------------------------------------------
// Known answers: the SW §5.2 / §5.3 byte layouts of push.bin and tickets.bin, written from
// sample_ops() and compared with literal little-endian words (a writer / reader pair that agreed
// on a different field order would still round-trip; these pin the offsets).
// ---------------------------------------------------------------------------
static void words_le(const uint32_t * w, size_t n, uint8_t * out) {
    for (size_t i = 0; i < n; ++i) put32(out + 4u * i, w[i]);
}

static void test_known_bytes(const char * scratch) {
    static const uint32_t push_kat[4 * 12] = {
        // w[0..7]                                                        dep[0]      dep[1]      push        kind
        0x00000110u, 7u, 100u, 3u, 0u, 0u, 0u, 0u,                         0xF0000000u, 0xF0000000u, 0x0000FF00u, 0u,   // LD ch1
        0x00000011u, 7u, 200u, 3u, 0u, 0u, 0u, 0u,                         0x00000000u, 0xF0000000u, 0x00000F01u, 0u,   // ST, w4/5 masked
        0x00000008u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,                           0u, 0u, 0u, 1u,                              // QCTRL QCLEAR
        0xC0DE0001u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,                           0u, 0u, 0u, 2u,                              // TRACE_MARK
    };
    static const uint32_t tickets_kat[4 + 6 * 8] = {
        6u, 0u, 0u, 0u,                                                    // n_units, reserved[3]
        // count err_code err_rel_seq err_info err_w0  err_rel_pidx reserved[2]
        2u, 7u, 1u, 0x123u, 0x456u, 2u, 0u, 0u,                            // DMA
        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
        1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,                                    // SYNC
    };
    uint8_t want[sizeof(tickets_kat)];
    tsr_trace2 * t = sample_ops();
    char * dir = pj(scratch, "kat");
    CHECK(tsr_trace2_write(t, dir, NULL) == 0, "write the known-answer trace");
    char * pb = pj(dir, "ops/push.bin"), * tb = pj(dir, "expect/tickets.bin");
    size_t n = 0;
    char * got = slurp(pb, &n);
    words_le(push_kat, sizeof(push_kat) / 4u, want);
    CHECK(got != NULL && n == sizeof(push_kat) && memcmp(got, want, n) == 0, "push.bin = the SW §5.2 layout (%zu bytes)", n);
    free(got);
    got = slurp(tb, &n);
    words_le(tickets_kat, sizeof(tickets_kat) / 4u, want);
    CHECK(got != NULL && n == sizeof(tickets_kat) && memcmp(got, want, n) == 0, "tickets.bin = the SW §5.3 layout (%zu bytes)", n);
    free(got);
    // And the reader takes the literal bytes back field by field.
    spit(tb, want, sizeof(tickets_kat));
    tsr_trace2 * r = NULL;
    char err[600];
    CHECK(tsr_trace2_read(dir, &r, err, sizeof(err)) == 0, "read the known-answer trace: %s", err);
    if (r)
        CHECK(r->n_units == 6 && r->unit[0].count == 2 && r->unit[0].err_code == 7 && r->unit[0].err_rel_seq == 1 &&
              r->unit[0].err_info == 0x123u && r->unit[0].err_w0 == 0x456u && r->unit[0].err_rel_pidx == 2 &&
              r->unit[5].count == 1 && r->push[1].dep[0] == 0 && r->push[1].dep[1] == 0xF0000000u &&
              r->push[1].push == 0x0F01u && r->push[3].kind == 2u && r->push[3].w[0] == 0xC0DE0001u,
              "fields read from the literal bytes");
    tsr_trace2_free(r);
    free(pb);
    free(tb);
    free(dir);
    tsr_trace2_free(t);
}

int main(int argc, char ** argv) {
    if (argc > 1) g_base = argv[1];
    char * scratch = make_scratch(g_base, "sim_trace");
    test_image_id();
    test_roundtrip_ops(scratch);
    test_roundtrip_launch(scratch);
    test_runs(scratch);
    test_known_bytes(scratch);
    if (!g_fail) rm_rf(scratch);
    else fprintf(stderr, "sim_trace: scratch kept in %s\n", scratch);
    free(scratch);
    return test_finish("sim_trace");
}
