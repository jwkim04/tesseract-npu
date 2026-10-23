// test_sim_transport.c — the "sim" transport: selection, environment variables, DRAM access
// bounds, exec results, TSR_TRACE dumps (sequence numbering, replayable), file-backed DRAM.
#include "test_sim_util.h"
#include "tsr_trace.h"
#include "tsr_transport.h"

#include <dirent.h>

#define DONE    TSR_STATUS_DONE
#define DONEERR (TSR_STATUS_DONE | TSR_STATUS_ERR)

static const char * g_base;
static const uint64_t IO = TSR_REGION_IO_OFF / 64;
static const uint64_t WOFF = TSR_REGION_WEIGHT_OFF / 64;

static void clear_env(void) {
    unsetenv("TSR_DEVICE");
    unsetenv("TSR_SIM_DRAM_DIR");
    unsetenv("TSR_SIM_TMAX");
    unsetenv("TSR_SIM_CH_GIB");
    unsetenv("TSR_SIM_THREADS");
    unsetenv("TSR_SIM_LOG");
    unsetenv("TSR_TRACE");
}

static void put_prog(tsr_transport * t, const prog_t * p, uint64_t cmd_beat) {
    uint8_t * b = (uint8_t *)malloc((size_t)p->n * 32 + 1);
    for (uint32_t k = 0; k < p->n; ++k) {
        for (uint32_t j = 0; j < 8; ++j) {
            const uint32_t w = p->v[k].w[j];
            uint8_t * q = b + (size_t)k * 32 + 4 * j;
            q[0] = (uint8_t)w; q[1] = (uint8_t)(w >> 8); q[2] = (uint8_t)(w >> 16); q[3] = (uint8_t)(w >> 24);
        }
    }
    CHECK(tsr_transport_write(t, 0, cmd_beat * 64, b, (size_t)p->n * 32) == 0, "write program");
    free(b);
}

static int count_entries(const char * dir) {
    DIR * d = opendir(dir);
    if (!d) return -1;
    int n = 0;
    struct dirent * e;
    while ((e = readdir(d)) != NULL) n += e->d_name[0] != '.';
    closedir(d);
    return n;
}

// LD x -> QACT -> MVM -> ST through the transport; verifies against direct tsr_ref.
static void run_matmul(tsr_transport * t, uint32_t fmt, uint32_t N, uint32_t K, uint32_t T, uint64_t seed) {
    rng_t r = { seed };
    const tsr_config * c = tsr_transport_config(t);
    const uint32_t kind = tsr_qa_kind_for_fmt(fmt), xl = K / 16, ql = tsr_qa_lines(kind, K);
    const uint32_t yl = tsr_ceil_div_u32(4 * N, 64);
    const size_t rbytes = (size_t)(K / tsr_fmt_block_size(fmt)) * tsr_fmt_ggml_block_bytes(fmt);
    const uint32_t rb = tsr_row_beats(fmt, K);
    uint8_t * blocks = (uint8_t *)malloc(rbytes * N);
    uint8_t * row = (uint8_t *)malloc((size_t)rb * 64);
    for (uint32_t n = 0; n < N; ++n) {
        gen_row_blocks(&r, fmt, K, blocks + rbytes * n);
        uint32_t ch;
        uint64_t off;
        tsr_row_location(fmt, N, K, c->num_ch, n, &ch, &off);
        tsr_repack_row(fmt, blocks + rbytes * n, K, row);
        CHECK(tsr_transport_write(t, ch, (WOFF + off) * 64, row, (size_t)rb * 64) == 0, "write row");
    }
    float * x = (float *)malloc(sizeof(float) * K * T);
    gen_f32(&r, x, (size_t)K * T);
    uint8_t * xb = (uint8_t *)malloc((size_t)K * T * 4);
    put_f32_le(xb, x, (size_t)K * T);
    CHECK(tsr_transport_write(t, 0, IO * 64, xb, (size_t)K * T * 4) == 0, "write x");
    prog_t p = {0};
    prog_add(&p, tsr_enc_ld(0, 0, (uint32_t)IO, T * xl));
    prog_add(&p, tsr_enc_qact(kind, T * xl, 0, K, T, 0));
    prog_add(&p, tsr_enc_mvm(fmt, 0, T * (xl + ql), T * xl, (uint32_t)WOFF, N, K, T, 0));
    prog_add(&p, tsr_enc_st(0, T * (xl + ql), (uint32_t)(IO + 100000), T * yl));
    prog_add(&p, tsr_enc_end());
    put_prog(t, &p, 64);
    tsr_exec_result res;
    memset(&res, 0xFF, sizeof(res));
    const int rc = tsr_transport_exec(t, 64, p.n, NULL, &res);
    CHECK(rc == 0 && res.status == DONE && res.err_code == 0 && res.err_pc == 0 && res.err_info == 0,
          "matmul exec rc %d status %u err %u", rc, res.status, res.err_code);
    uint8_t * y = (uint8_t *)malloc((size_t)T * yl * 64);
    CHECK(tsr_transport_read(t, 0, (IO + 100000) * 64, y, (size_t)T * yl * 64) == 0, "read y");
    uint8_t * qa = (uint8_t *)malloc((size_t)ql * 64);
    for (uint32_t tt = 0; tt < T; ++tt) {
        tsr_qact(kind, x + (size_t)tt * K, K, qa);
        for (uint32_t n = 0; n < N; ++n) {
            tsr_repack_row(fmt, blocks + rbytes * n, K, row);
            const uint32_t want = f2u(tsr_dot_row(fmt, row, K, qa));
            const uint32_t got = get_u32_le(y + ((size_t)tt * yl) * 64 + 4 * n);
            CHECK(got == want, "matmul t %u row %u: 0x%08x vs 0x%08x", tt, n, got, want);
        }
    }
    free(blocks); free(row); free(x); free(xb); free(y); free(qa);
    prog_free(&p);
}

static void test_open_and_config(void) {
    clear_env();
    tsr_transport * t = tsr_transport_open(NULL);
    CHECK(t != NULL, "default open");
    if (!t) return;
    CHECK(strcmp(tsr_transport_name(t), "sim") == 0, "name");
    const tsr_config * c = tsr_transport_config(t);
    const tsr_config u = TSR_CONFIG_U200_V1;
    CHECK(c->num_ch == u.num_ch && c->lanes == u.lanes && c->t_max == 1 && c->k_max == u.k_max &&
          c->spm_bytes == u.spm_bytes && c->vl == u.vl && c->nsreg == u.nsreg && c->ch_bytes == u.ch_bytes &&
          c->beat_bits == 512 && c->caps == 0, "default config is U200 v1");
    CHECK(tsr_transport_csr_read(t, TSR_CSR_ID) == TSR_ID_MAGIC, "csr read");
    // DRAM access bounds
    uint8_t buf[256];
    memset(buf, 0x3C, sizeof(buf));
    CHECK(tsr_transport_write(t, 3, c->ch_bytes - 256, buf, 256) == 0, "write at the end");
    CHECK(tsr_transport_write(t, 3, c->ch_bytes - 255, buf, 256) != 0, "write past the end");
    CHECK(tsr_transport_write(t, 4, 0, buf, 1) != 0, "write ch 4");
    CHECK(tsr_transport_read(t, 0, UINT64_MAX - 3, buf, 8) != 0, "read offset overflow");
    CHECK(tsr_transport_write(t, 0, 0, buf, 0) == 0, "zero-size write");
    uint8_t back[256];
    CHECK(tsr_transport_read(t, 3, c->ch_bytes - 256, back, 256) == 0 && memcmp(back, buf, 256) == 0, "read back");
    CHECK(tsr_transport_write(t, 1, 12345, buf, 7) == 0 && tsr_transport_read(t, 1, 12344, back, 9) == 0 &&
          back[0] == 0 && memcmp(back + 1, buf, 7) == 0 && back[8] == 0, "unaligned access");
    run_matmul(t, TSR_FMT_Q4_K, 37, 1280, 1, 1);
    run_matmul(t, TSR_FMT_Q8_0, 70, 1056, 1, 2);
    // error exec
    prog_t p = {0};
    prog_add(&p, tsr_enc_nop());
    prog_add(&p, tsr_enc_ld(0, 0, 0xFFFFFFF0u, 1));
    prog_add(&p, tsr_enc_end());
    put_prog(t, &p, 0);
    tsr_exec_result res;
    CHECK(tsr_transport_exec(t, 0, p.n, NULL, &res) != 0, "error exec returns nonzero");
    CHECK(res.status == DONEERR && res.err_code == TSR_ERR_DRAM_RANGE && res.err_pc == 1 && res.err_info == 0xFFFFFFF0u,
          "error result: status %u code %u pc %u info 0x%x", res.status, res.err_code, res.err_pc, res.err_info);
    CHECK(tsr_transport_exec(t, 0, p.n, NULL, NULL) != 0, "exec with NULL result");
    // args reach R8..R15
    prog_reset(&p);
    tsr_insn st = tsr_enc_st(0, 0, 0, 1);
    tsr_insn_set_base(&st, 0, 9, 0);   // DRAM = R9 (= ARG1)
    prog_add(&p, st);
    prog_add(&p, tsr_enc_end());
    put_prog(t, &p, 0);
    const uint32_t args[8] = { 0, (uint32_t)(IO + 777), 0, 0, 0, 0, 0, 0 };
    CHECK(tsr_transport_exec(t, 0, p.n, args, &res) == 0, "args exec");
    const uint64_t rd = tsr_transport_csr_read(t, TSR_CSR_PERF_BASE + 8 * TSR_PERF_RETIRED);
    CHECK(rd > 0, "perf readable through the transport");
    prog_free(&p);
    tsr_transport_close(t);

    // Selection and environment errors
    CHECK(tsr_transport_open("bogus") == NULL, "unknown kind");
    CHECK(tsr_transport_open("xrt") == NULL, "xrt not built");
    setenv("TSR_DEVICE", "nope", 1);
    CHECK(tsr_transport_open(NULL) == NULL, "TSR_DEVICE=nope");
    setenv("TSR_DEVICE", "sim", 1);
    t = tsr_transport_open(NULL);
    CHECK(t != NULL, "TSR_DEVICE=sim");
    tsr_transport_close(t);
    const char * bad[][2] = { { "TSR_SIM_TMAX", "9" }, { "TSR_SIM_TMAX", "0" }, { "TSR_SIM_TMAX", "2x" },
                              { "TSR_SIM_TMAX", "-1" }, { "TSR_SIM_CH_GIB", "17" }, { "TSR_SIM_CH_GIB", "0" },
                              { "TSR_SIM_THREADS", "0" }, { "TSR_SIM_TMAX", "99999999999999999999999" },
                              { "TSR_SIM_DRAM_DIR", "/nonexistent/dir/for/tsr" } };
    for (size_t i = 0; i < sizeof(bad) / sizeof(bad[0]); ++i) {
        clear_env();
        setenv(bad[i][0], bad[i][1], 1);
        t = tsr_transport_open("sim");
        CHECK(t == NULL, "%s=%s must fail", bad[i][0], bad[i][1]);
        if (t) tsr_transport_close(t);
    }
    clear_env();
    setenv("TSR_SIM_TMAX", "8", 1);
    setenv("TSR_SIM_CH_GIB", "2", 1);
    setenv("TSR_SIM_THREADS", "3", 1);
    t = tsr_transport_open("sim");
    CHECK(t != NULL, "env overrides");
    if (t) {
        c = tsr_transport_config(t);
        CHECK(c->t_max == 8 && c->ch_bytes == (2ull << 30), "TSR_SIM_TMAX / TSR_SIM_CH_GIB: t_max %u ch_bytes %llu",
              c->t_max, (unsigned long long)c->ch_bytes);
        CHECK((tsr_transport_csr_read(t, TSR_CSR_CFG0) >> 20 & 0xF) == 8, "CFG0 T_MAX");
        run_matmul(t, TSR_FMT_Q8_0, 33, 512, 8, 3);
        run_matmul(t, TSR_FMT_Q4_K, 5, 768, 5, 4);
        tsr_transport_close(t);
    }
    clear_env();
}

static void test_trace_dump(void) {
    clear_env();
    char * tdir = (char *)malloc(strlen(g_base) + 32);
    sprintf(tdir, "%s/dumps/nested", g_base);
    setenv("TSR_TRACE", tdir, 1);
    setenv("TSR_SIM_TMAX", "4", 1);
    tsr_transport * t = tsr_transport_open("sim");
    CHECK(t != NULL, "open with TSR_TRACE");
    if (!t) return;
    run_matmul(t, TSR_FMT_Q4_K, 9, 512, 3, 10);     // -> 000000
    prog_t p = {0};
    prog_add(&p, tsr_enc_mvm(0, 0, 0, 0, (uint32_t)WOFF, 4, 31, 1, 0));   // K_RANGE -> 000001
    put_prog(t, &p, 0);
    tsr_exec_result res;
    CHECK(tsr_transport_exec(t, 0, p.n, NULL, &res) != 0 && res.err_code == TSR_ERR_K_RANGE, "error exec");
    tsr_transport_close(t);
    // A second process-equivalent (new transport) continues the numbering.
    t = tsr_transport_open("sim");
    run_matmul(t, TSR_FMT_Q8_0, 3, 96, 2, 11);      // -> 000002
    tsr_transport_close(t);
    CHECK(count_entries(tdir) == 3, "3 trace dirs, got %d", count_entries(tdir));
    for (int i = 0; i < 3; ++i) {
        char d[600];
        snprintf(d, sizeof(d), "%s/%06d", tdir, i);
        tsr_trace tr;
        CHECK(tsr_trace_read(d, &tr) == 0, "read dump %d", i);
        CHECK(tr.cfg.t_max == 4, "dump %d config", i);
        CHECK(tr.perf.valid && (i == 1 ? tr.perf.retired == 0 : tr.perf.retired == 5), "dump %d perf", i);
        if (i == 1) CHECK(tr.status == DONEERR && tr.err_code == TSR_ERR_K_RANGE && tr.err_pc == 0, "dump 1 is the error run");
        else CHECK(tr.status == DONE && tr.cap.n_expect == 1, "dump %d status", i);
        CHECK(tsr_trace_replay(&tr, 0) == 0, "dump %d replays", i);
        // perf of the dump is the run's delta: replay on a fresh sim reproduces it without warning
        tsr_trace_free(&tr);
    }
    prog_free(&p);
    free(tdir);
    clear_env();
}

static void test_file_dram(void) {
    clear_env();
    char * ddir = make_scratch(g_base, "dram");
    setenv("TSR_SIM_DRAM_DIR", ddir, 1);
    setenv("TSR_SIM_CH_GIB", "1", 1);
    tsr_transport * t = tsr_transport_open("sim");
    CHECK(t != NULL, "open with TSR_SIM_DRAM_DIR");
    if (t) {
        CHECK(count_entries(ddir) == 0, "DRAM files are unlinked");
        uint8_t z[64], w[64];
        CHECK(tsr_transport_read(t, 2, 5000, z, 64) == 0, "read");
        int zero = 1;
        for (int i = 0; i < 64; ++i) zero &= z[i] == 0;
        CHECK(zero, "file-backed DRAM starts zeroed");
        memset(w, 0x77, 64);
        CHECK(tsr_transport_write(t, 2, 5000, w, 64) == 0 && tsr_transport_read(t, 2, 5000, z, 64) == 0 &&
              memcmp(w, z, 64) == 0, "file-backed write/read");
        run_matmul(t, TSR_FMT_Q4_K, 21, 1024, 1, 20);
        tsr_transport_close(t);
    }
    CHECK(count_entries(ddir) == 0, "nothing left behind");
    rm_rf(ddir);
    free(ddir);
    clear_env();
}

int main(int argc, char ** argv) {
    g_base = make_scratch(argc > 1 ? argv[1] : ".", "sim_transport");
    test_open_and_config();
    test_trace_dump();
    test_file_dram();
    const int rc = test_finish("sim_transport");
    if (rc == 0) rm_rf(g_base);
    return rc;
}
