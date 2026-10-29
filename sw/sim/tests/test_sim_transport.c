// test_sim_transport.c — the "sim" transport: selection, environment variables, configuration
// and identity through the CSRs, DRAM access bounds, file-backed DRAM. Nothing executes through a
// transport until the v2 launch path (P3.1); the transport's launch tests return with it.
#include "test_sim_util.h"
#include "tsr_transport.h"

#include <dirent.h>

static const char * g_base;

static void clear_env(void) {
    unsetenv("TSR_DEVICE");
    unsetenv("TSR_SIM_DRAM_DIR");
    unsetenv("TSR_SIM_TMAX");
    unsetenv("TSR_SIM_CH_GIB");
    unsetenv("TSR_SIM_THREADS");
    unsetenv("TSR_SIM_LOG");
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
    CHECK(tsr_transport_csr_read(t, TSR_CSR_VERSION) >> 8 == TSR_ISA_MAJOR && TSR_ISA_MAJOR == 2, "ISA major 2");
    CHECK(tsr_transport_csr_read(t, TSR_CSR_PERF_BASE + 8 * TSR_PERF_RETIRED) == 0, "perf readable, 0 without ops");
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
    // large transfers across channels stay separate
    const size_t big = 3u << 20;
    uint8_t * w0 = (uint8_t *)malloc(big), * w1 = (uint8_t *)malloc(big), * rb = (uint8_t *)malloc(big);
    rng_t r = { 5 };
    rng_bytes(&r, w0, big);
    rng_bytes(&r, w1, big);
    CHECK(tsr_transport_write(t, 0, (4ull << 30) - 4099, w0, big) == 0 && tsr_transport_write(t, 2, (4ull << 30) - 4099, w1, big) == 0,
          "writes across 4 GiB");
    CHECK(tsr_transport_read(t, 0, (4ull << 30) - 4099, rb, big) == 0 && memcmp(rb, w0, big) == 0, "ch0 read back");
    CHECK(tsr_transport_read(t, 2, (4ull << 30) - 4099, rb, big) == 0 && memcmp(rb, w1, big) == 0, "ch2 read back");
    free(w0);
    free(w1);
    free(rb);
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
                              { "TSR_SIM_DRAM_DIR", "/nonexistent/dir/for/tsr" },
                              { "TSR_SIM_LOG", "/nonexistent/dir/for/tsr/log" } };
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
        uint8_t z[64];
        CHECK(tsr_transport_read(t, 3, (2ull << 30) - 64, z, 64) == 0 && tsr_transport_read(t, 3, (2ull << 30) - 63, z, 64) != 0,
              "2 GiB channel bounds");
        tsr_transport_close(t);
    }
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
    test_file_dram();
    const int rc = test_finish("sim_transport");
    if (rc == 0) rm_rf(g_base);
    return rc;
}
