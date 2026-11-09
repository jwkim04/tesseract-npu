// test_sim_transport.c — ctest sim_transport (docs/design/06-software-verification.md "SW" §10.2,
// launch path): the "sim" kind through the public C API only (tsr_transport.h):
// selection, environment variables, configuration and identity through the CSRs (CAPS = what the
// simulator implements, EX §7.5), DRAM access bounds, file-backed DRAM; then boot / launch / wait /
// soft reset with the production image (tsr_image_open_default: TSR_IMAGE or the build's
// tsr-fw.elf): cold and warm boot, the base kernels at 0 / 1 / 2, PING, an empty `replay` launch,
// pipelined launches, the record decode, a soft reset with launches in flight (answered or
// TSR_TE_LOST, the rings flushed), launches refused until the next (warm) boot, the statistics. The
// protocol in depth is transport_protocol's (sw/transport/tests).
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
    const tsr_config u = TSR_CONFIG_U200;
    CHECK(c->num_ch == u.num_ch && c->lanes == u.lanes && c->t_max == 1 && c->k_max == u.k_max &&
          c->spm_bytes == u.spm_bytes && c->vl == u.vl && c->nsreg == u.nsreg && c->ch_bytes == u.ch_bytes &&
          c->beat_bits == 512 && c->caps == tsr_sim_caps_implemented() && (c->caps & TSR_CAPS_VU),
          "default config: the U200 tile core, CAPS = what the simulator implements (0x%x)", c->caps);
    CHECK(tsr_transport_csr_read(t, TSR_CSR_CAPS) == c->caps && tsr_transport_csr_read(t, TSR_CSR_CTL_ID) != 0 &&
          tsr_transport_csr_read(t, TSR_CSR_RV_FREQ) != 0, "CAPS, CTL_ID, RV_FREQ through the CSR port");
    CHECK(tsr_transport_csr_read(t, TSR_CSR_ID) == TSR_ID_MAGIC, "csr read");
    CHECK(tsr_transport_csr_read(t, TSR_CSR_VERSION) >> 8 == TSR_ISA_MAJOR && TSR_ISA_MAJOR == 1,
          "Tesseract Core capability major 1");
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

// boot / launch / wait / soft reset through the public API (EX §10.1).
static tsr_launch mk_launch(uint8_t type, uint16_t kernel, uint32_t a0, uint32_t a1, uint32_t a2) {
    tsr_launch l;
    memset(&l, 0, sizeof(l));
    l.type = type;
    l.kernel = kernel;
    l.arg[0] = a0;
    l.arg[1] = a1;
    l.arg[2] = a2;
    return l;
}

static void test_boot_launch(void) {
    clear_env();
    unsetenv("TSR_IMAGE");
    tsr_image * img = NULL;
    if (tsr_image_open_default(&img) != 0) {
        printf("sim_transport: the build's firmware image is missing: boot / launch tests not run\n");
        return;
    }
    tsr_transport * t = tsr_transport_open("sim");
    CHECK(t != NULL, "open");
    if (t == NULL) {
        tsr_image_close(img);
        return;
    }
    CHECK(tsr_image_kernel(img, "graph") == 0 && tsr_image_kernel(img, "selftest") == 1 && tsr_image_kernel(img, "replay") == 2 &&
          tsr_image_kernel(img, "nosuch") == -1, "base kernels at 0 / 1 / 2 (EX §4.6)");
    CHECK(tsr_image_id(img) != 0 && strstr(tsr_image_path(img), "tsr-fw.elf") != NULL, "image id and path");
    tsr_record r;
    tsr_result res;
    const tsr_launch ping = mk_launch(TSR_DESC_TYPE_PING, 0, 0x600D, 0, 0);
    CHECK(tsr_transport_call(t, &ping, TSR_TIMEOUT_DEFAULT, &r) == TSR_TE_STATE, "launch before boot refused");
    CHECK(tsr_transport_boot(t, img) == 0, "cold boot");
    CHECK(tsr_transport_image(t) == img, "booted image");
    CHECK(tsr_transport_boot(t, img) == 0, "warm boot");
    CHECK(tsr_transport_call(t, &ping, TSR_TIMEOUT_DEFAULT, &r) == 0, "PING");
    tsr_record_decode(&r, &res);
    CHECK(res.type == TSR_LCQ_R0_TYPE_PONG && res.status == 0 && r.w[2] == 0x600Du, "PONG echoes ARG0 (%s)", tsr_status_name(res.status));
    // An empty replay (no records): DONE OK, RET 0.
    const tsr_launch empty = mk_launch(TSR_DESC_TYPE_LAUNCH, 2, 0, 0, 0);
    CHECK(tsr_transport_call(t, &empty, TSR_TIMEOUT_DEFAULT, &r) == 0, "empty replay");
    tsr_record_decode(&r, &res);
    CHECK(res.type == TSR_LCQ_R0_TYPE_DONE && res.status == 0 && res.ret == 0 && res.err_ticket == TSR_TICKET_NONE,
          "DONE OK RET 0 (%s)", tsr_status_name(res.status));
    // 12 launches pipelined (blocks only at LQ_DEPTH outstanding), waited in order.
    uint32_t seq[12];
    for (int i = 0; i < 12; ++i) {
        const tsr_launch l = mk_launch(TSR_DESC_TYPE_PING, 0, 100u + (uint32_t)i, 0, 0);
        CHECK(tsr_transport_launch(t, &l, &seq[i]) == 0 && (i == 0 || seq[i] == seq[i - 1] + 1), "launch %d", i);
    }
    for (int i = 0; i < 12; ++i)
        CHECK(tsr_transport_wait(t, seq[i], TSR_TIMEOUT_DEFAULT, &r) == 0 && r.w[1] == seq[i] && r.w[2] == 100u + (uint32_t)i,
              "answer %d", i);
    CHECK(tsr_transport_wait(t, seq[0], TSR_TIMEOUT_DEFAULT, &r) == TSR_TE_INVAL, "a second wait is refused");
    // A bad descriptor: KERNEL beyond the table.
    const tsr_launch bad = mk_launch(TSR_DESC_TYPE_LAUNCH, 999, 0, 0, 0);
    CHECK(tsr_transport_call(t, &bad, TSR_TIMEOUT_DEFAULT, &r) == 0 && TSR_GET(r.w[0], TSR_LCQ_R0_STATUS) == TSR_LCQ_R0_STATUS_BAD_DESC,
          "BAD_DESC");
    // Soft reset with launches in flight (EX §8.5): each is answered or lost, the rings flushed.
    for (int i = 0; i < 4; ++i) CHECK(tsr_transport_launch(t, &ping, &seq[i]) == 0, "launch before the reset");
    CHECK(tsr_transport_soft_reset(t) == 0, "soft reset");
    int lost = 0;
    for (int i = 0; i < 4; ++i) {
        const int rc = tsr_transport_wait(t, seq[i], TSR_TIMEOUT_DEFAULT, &r);
        CHECK(rc == 0 || rc == TSR_TE_LOST, "answered or lost (%d: %s)", rc, tsr_transport_strerror(rc));
        lost += rc == TSR_TE_LOST;
    }
    CHECK(lost >= 1, "launches lost in the reset (%d)", lost);
    CHECK(tsr_transport_csr_read(t, TSR_CSR_LQ_TAKEN) == tsr_transport_csr_read(t, TSR_CSR_LQ_SEQ) &&
          tsr_transport_csr_read(t, TSR_CSR_LCQ_RD) == tsr_transport_csr_read(t, TSR_CSR_LCQ_WR) &&
          !(tsr_transport_csr_read(t, TSR_CSR_STATUS) & TSR_STATUS_RV_RUNNING), "rings flushed, the RISC-V in reset");
    CHECK(tsr_transport_image(t) == NULL && tsr_transport_call(t, &ping, TSR_TIMEOUT_DEFAULT, &r) == TSR_TE_STATE,
          "not booted after the soft reset");
    CHECK(tsr_transport_boot(t, img) == 0 && tsr_transport_call(t, &ping, TSR_TIMEOUT_DEFAULT, &r) == 0, "boot again, PING");
    tsr_transport_stats st;
    tsr_transport_get_stats(t, &st);
    CHECK(st.launches == 20 && st.csr_reads > 0 && st.csr_writes > st.launches, "stats: %llu launches",
          (unsigned long long)st.launches);
    uint32_t words[64];
    size_t n = 99;
    CHECK(tsr_transport_log_drain(t, words, 64, &n) == 0 && n <= 64, "log drain");
    tsr_image_close(img);   // the transport keeps its reference
    CHECK(tsr_transport_call(t, &ping, TSR_TIMEOUT_DEFAULT, &r) == 0 && tsr_transport_image(t) != NULL, "image kept by the transport");
    tsr_transport_close(t);
    // TSR_IMAGE overrides the default; a missing file is refused.
    setenv("TSR_IMAGE", "/nonexistent/tsr-fw.elf", 1);
    img = NULL;
    CHECK(tsr_image_open_default(&img) != 0 && img == NULL, "TSR_IMAGE=/nonexistent refused");
    unsetenv("TSR_IMAGE");
}

int main(int argc, char ** argv) {
    g_base = make_scratch(argc > 1 ? argv[1] : ".", "sim_transport");
    test_open_and_config();
    test_file_dram();
    test_boot_launch();
    const int rc = test_finish("sim_transport");
    if (rc == 0) rm_rf(g_base);
    return rc;
}
