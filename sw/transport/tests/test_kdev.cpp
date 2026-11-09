// test_kdev.cpp — ctest transport_kdev (docs/design/06-software-verification.md "SW" §10.2): the
// protocol layer (tsr_kdev.cpp) on the mock kdev, its adapter register model, and the chunked /
// aligned transfer engine against a strict fake DMA (docs/xrt-host.md §6).
//
// Covers: the identity checks at open (ID, Tesseract Core capability major 1 with any minor, CTL_ID
// != 0 (a core without the control core), ADAPTER_VERSION major 1 and NUM_CH, NUM_CH vs the bound
// buffers, 4 KiB-aligned buffers, RV_FREQ = 0: no clk_rv), ADAPTER_STATUS left by an
// earlier process (reported, cleared, the adapter log), the bind (MEM_EN, the latched bases equal the
// buffers), the adapter model (bases RW while MEM_EN = 0, BASE_LOCKED, BASE_MISALIGNED at the MEM_EN
// edge, REQ_HELD with the model held, ADAPTER_STATUS W1C, the reserved registers reading 0), timeouts
// (SOFT_RESET that never completes at open: TSR_RESET_MS; no HELLO: TSR_BOOT_MS), the stale core at
// open (a booted device with a launch running, left by a "dead process": TSR_ATTACH=1 opens it
// without a reset and refuses launches; a normal open soft-resets it and boots warm), a reopen
// after an earlier process (the host counters from the device: nothing it consumed taken again,
// what it left unread discarded, no stale log), the watermark refresh (EX §4.9 wrap), adapter
// errors during a launch (TSR_MOCK_INJECT N:V: plat_err_i, DONE PLAT_ERR, ADAPTER_STATUS read,
// reported and cleared; pipelined at depth 1, 4, 7: only the episode's launches flagged), a hung unit (TSR_MOCK_HANG: UNIT_HUNG at the timeout, "card reset
// needed", FAILED), the kdev registry and TSR_MOCK_* parsing, byte-granular transfers at odd
// offsets / sizes / host addresses, across and beyond 4 GiB, under DMA rules from align 1 to 4096
// with small chunks (shadow-memory compare, guard bytes untouched, every raw transfer checked
// against the rules), raw LCQ read / acknowledgement failures (retry without losing answers,
// duplicating events or returning launch credit), and the xrt kdev's hw_emu defaults (xrt_env_defaults).

#include "tsr_fw_paths.h"
#include "tsr_kdev.h"
#include "tsr_kdev_mock.h"
#include "tsr_sim_int.h"
#include "tsr/opl.hpp"

#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <unistd.h>
#include <vector>

using namespace tsr;

static int g_fail = 0, g_checks = 0;
#define CHECK(cond, ...)                                                          \
    do {                                                                          \
        g_checks++;                                                               \
        if (!(cond)) {                                                            \
            g_fail++;                                                             \
            fprintf(stderr, "FAIL %s:%d: %s: ", __FILE__, __LINE__, #cond);       \
            fprintf(stderr, __VA_ARGS__);                                         \
            fprintf(stderr, "\n");                                                \
        }                                                                         \
    } while (0)

struct rng {
    uint64_t s;
    uint64_t next() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return s;
    }
    uint64_t below(uint64_t n) { return n ? next() % n : 0; }
    void fill(void * p, size_t n) {
        uint8_t * b = (uint8_t *) p;
        for (size_t i = 0; i < n; ++i) b[i] = (uint8_t) (next() >> 24);
    }
};

// ---------------------------------------------------------------------------
// Sparse byte memory (shadow and fake device)
// ---------------------------------------------------------------------------
struct sparse_mem {
    static constexpr uint64_t PG = 1u << 16;
    std::map<uint64_t, std::vector<uint8_t>> pages;
    uint8_t * at(uint64_t a) {
        auto & p = pages[a / PG];
        if (p.empty()) {
            p.assign(PG, 0);
            for (uint64_t i = 0; i < PG; ++i) p[i] = (uint8_t) ((a / PG * PG + i) * 0x9Du >> 3);   // deterministic background
        }
        return &p[a % PG];
    }
    void write(uint64_t a, const uint8_t * s, size_t n) {
        for (size_t i = 0; i < n; ++i) *at(a + i) = s[i];
    }
    void read(uint64_t a, uint8_t * d, size_t n) {
        for (size_t i = 0; i < n; ++i) d[i] = *at(a + i);
    }
};

// Fake raw DMA that enforces the rules and stores into sparse memory.
struct strict_raw : dma_raw {
    dma_rules r;
    sparse_mem mem[2];
    uint64_t violations = 0, calls = 0;
    bool check(uint32_t ch, uint64_t off, const void * p, size_t n) {
        calls++;
        const bool bad = ch >= 2 || n == 0 || n > r.max_xfer || off % r.align || n % r.align || (uintptr_t) p % r.align;
        if (bad) {
            violations++;
            fprintf(stderr, "strict_raw: rule violation ch %u off 0x%" PRIx64 " n %zu p %p (align %zu max %zu)\n", ch, off, n, p,
                    r.align, r.max_xfer);
        }
        return !bad;
    }
    int raw_write(uint32_t ch, uint64_t off, const void * src, size_t n) override {
        if (!check(ch, off, src, n)) return -1;
        mem[ch].write(off, (const uint8_t *) src, n);
        return 0;
    }
    int raw_read(uint32_t ch, uint64_t off, void * dst, size_t n) override {
        if (!check(ch, off, dst, n)) return -1;
        mem[ch].read(off, (uint8_t *) dst, n);
        return 0;
    }
};

static void test_dma_engine() {
    struct cfg { size_t align, max_xfer; bool bounce; };
    const cfg cfgs[] = {
        { 1, 7, false }, { 1, 65536, false }, { 1, 4096, true }, { 64, 64, false }, { 64, 192, false },
        { 64, 65536, true }, { 4096, 4096, false }, { 4096, 12288, false }, { 4096, 4096, true },
    };
    const uint64_t G4 = 4ull << 30;
    const uint64_t bases[] = { 0, 4096, G4 - 70000, G4 - 1, G4, G4 + 12345, (12ull << 30) + 77, (16ull << 30) - 300000 };
    rng r{ 0x1234567ull };
    std::vector<uint8_t> src(1u << 20), dst(1u << 20), want(1u << 20);
    for (const cfg & c : cfgs) {
        strict_raw raw;
        raw.r.align = c.align;
        raw.r.max_xfer = c.max_xfer;
        raw.r.always_bounce = c.bounce;
        CHECK(dma_rules_check(raw.r).empty(), "rules valid");
        dma_engine eng(raw, raw.r);
        sparse_mem shadow[2];
        const size_t A = c.align, M = c.max_xfer;
        const size_t sizes[] = { 0, 1, 2, 63, 64, 65, A - 1 ? A - 1 : 3, A, A + 1, 2 * A - 1, M - 1 ? M - 1 : 5, M, M + 1,
                                 3 * M + 7, 100003, 262144 };
        int n_cases = 0;
        for (uint64_t base : bases) {
            for (size_t n : sizes) {
                if (n > src.size() - 4096) continue;
                for (int rep = 0; rep < 2; ++rep) {
                    const uint32_t ch = (uint32_t) r.below(2);
                    const uint64_t off = base + (rep ? r.below(5000) : 0);
                    const size_t sp = rep ? r.below(A + 3) : 0, dp = rep ? r.below(A + 5) : 0;   // host misalignment
                    r.fill(src.data() + sp, n);
                    CHECK(eng.write(ch, off, src.data() + sp, n) == 0, "write ch %u off 0x%" PRIx64 " n %zu (align %zu max %zu)", ch, off, n,
                          A, M);
                    shadow[ch].write(off, src.data() + sp, n);
                    CHECK(eng.read(ch, off, dst.data() + dp, n) == 0, "read");
                    CHECK(n == 0 || memcmp(dst.data() + dp, src.data() + sp, n) == 0, "read back ch %u off 0x%" PRIx64 " n %zu (align %zu max %zu bounce %d)",
                          ch, off, n, A, M, (int) c.bounce);
                    // Guard bytes around the range and the range itself, against the shadow, in the fake device memory.
                    const uint64_t g0 = off >= A + 16 ? off - A - 16 : 0;
                    const size_t gl = (size_t) (off - g0) + n + A + 16;
                    std::vector<uint8_t> a(gl), b(gl);
                    raw.mem[ch].read(g0, a.data(), gl);
                    shadow[ch].read(g0, b.data(), gl);
                    CHECK(memcmp(a.data(), b.data(), gl) == 0, "device == shadow around ch %u off 0x%" PRIx64 " n %zu (align %zu max %zu)", ch,
                          off, n, A, M);
                    n_cases++;
                }
            }
        }
        CHECK(raw.violations == 0, "no raw rule violations (align %zu max %zu bounce %d)", A, M, (int) c.bounce);
        // Fully aligned transfers go direct (no bounce), in ceil(n / max) raw calls.
        if (!c.bounce) {
            const uint64_t calls0 = raw.calls, bc0 = eng.bounce_copies;
            void * p = nullptr;
            CHECK(posix_memalign(&p, 4096, 3 * M + A) == 0, "alloc");
            const size_t n = 3 * M;
            CHECK(eng.write(0, G4, p, n) == 0, "aligned write");
            CHECK(raw.calls - calls0 == 3 && eng.bounce_copies == bc0, "aligned write: %" PRIu64 " raw calls, %" PRIu64 " bounce copies",
                  raw.calls - calls0, eng.bounce_copies - bc0);
            free(p);
        }
        printf("dma_engine align %4zu max %6zu bounce %d: %d cases, %" PRIu64 " raw calls, %" PRIu64 " RMW reads\n", A, M, (int) c.bounce,
               n_cases, raw.calls, eng.rmw_reads);
    }
}

// ---------------------------------------------------------------------------
// Protocol layer on the mock kdev
// ---------------------------------------------------------------------------
static void clear_env() {
    for (const char * v : { "TSR_DEVICE", "TSR_IMAGE", "TSR_LAUNCH_TIMEOUT_MS", "TSR_ABORT_MS", "TSR_BOOT_MS", "TSR_RESET_MS", "TSR_LOG",
                            "TSR_LOG_LEVEL", "TSR_ATTACH", "TSR_STATS", "TSR_DUMP_DIR", "TSR_GDB_WAIT", "TSR_TILE_SERIAL",
                            "TSR_TILE_HIST", "TSR_MOCK_INJECT", "TSR_MOCK_TCM_FAULT", "TSR_MOCK_HANG", "TSR_MOCK_NUM_CH",
                            "TSR_MOCK_TMAX", "TSR_MOCK_CH_GIB", "TSR_MOCK_DMA_ALIGN", "TSR_MOCK_DMA_CHUNK_KIB", "TSR_MOCK_DMA_BOUNCE",
                            "TSR_SIM_SCHED", "TSR_SIM_RACE", "TSR_SIM_LOG", "TSR_TRACE", "TSR_SIM_DB" })
        unsetenv(v);
    setenv("TSR_SIM_THREADS", "2", 1);
}

static mock_options small_mock(uint32_t num_ch = 4) {
    mock_options o;
    o.num_ch = num_ch;
    o.threads = 2;
    return o;
}

static tsr_transport * open_mock(mock_options o, mock_kdev ** out) {
    std::unique_ptr<mock_kdev> m(new mock_kdev(o));
    CHECK(m->ok(), "mock created");
    *out = m.get();
    return transport_open_kdev(std::move(m));
}

static tsr_launch L(uint8_t type, uint16_t kernel, std::vector<uint32_t> a = {}) {
    tsr_launch l{};
    l.type = type;
    l.kernel = kernel;
    for (size_t i = 0; i < a.size() && i < 7; ++i) l.arg[i] = a[i];
    return l;
}

static uint32_t st_of(const tsr_record & r) { return TSR_GET(r.w[0], TSR_LCQ_R0_STATUS); }

static void test_protocol_basic() {
    printf("protocol: open, identity, bind\n");
    mock_options o = small_mock();
    mock_kdev * m = nullptr;
    tsr_transport * t = open_mock(o, &m);
    CHECK(t != nullptr, "open");
    if (!t) return;
    const tsr_config * c = tsr_transport_config(t);
    CHECK(strcmp(tsr_transport_name(t), "mock") == 0, "name %s", tsr_transport_name(t));
    CHECK(c->num_ch == 4 && c->t_max == 1 && c->lanes == 128 && c->k_max == 32768 && c->spm_bytes == 4u << 20 && c->vl == 16 &&
              c->nsreg == 16 && c->caps == tsr_sim_caps_implemented() && c->beat_bits == 512 && c->ch_bytes == (16ull << 30),
          "config decode (CAPS 0x%x)", c->caps);
    CHECK(transport_kdev(t) == m && transport_sim(t) == m->sim(), "transport_kdev / transport_sim");
    CHECK(tsr_transport_csr_read(t, TSR_CSR_ID) == TSR_ID_MAGIC, "window ID");
    CHECK(tsr_transport_csr_read(t, TSR_CSR_VERSION) == 0x0100u, "window VERSION 0x%x (Tesseract Core capability 1.0)",
          tsr_transport_csr_read(t, TSR_CSR_VERSION));
    const uint64_t wr0 = m->window_reads;
    (void) tsr_transport_csr_read(t, TSR_CSR_CFG0);
    CHECK(m->window_reads == wr0 + 1, "csr_read is one window read");
    CHECK(tsr_transport_csr_read(t, 0x1000) == 0xFFFFFFFFu, "csr_read outside the window refused");
    CHECK(tsr_transport_csr_write(t, 0x1000, 0) == TSR_TE_INVAL, "csr_write outside the window refused");
    CHECK(m->ch_addr(1) > (4ull << 30), "device addresses above 4 GiB");
    // The open: SOFT_RESET (EX §8.5) and the bind (EX §9.2).
    bool sr = false;
    for (uint32_t off : m->window_write_log) sr = sr || off == TSR_CSR_CTRL;
    CHECK(sr, "SOFT_RESET written at open");
    CHECK(m->mem_en() && m->bind_errors == 0, "MEM_EN set, the latched bases are the buffers");
    for (uint32_t ch = 0; ch < 4; ++ch) CHECK(m->latched_base(ch) == m->ch_addr(ch), "ch%u latched base", ch);
    CHECK((tsr_transport_csr_read(t, TSR_CSR_LOG_CFG) & 0xF3u) == (TSR_LOG_CFG_ENABLE | (3u << 4)), "LOG_CFG ENABLE, LEVEL 3");
    // PERF through the window equals the simulator's (LO then HI)
    const uint32_t cmd[8] = { tsr_enc_ld(1, 0, 1000, 3).w[0], 0, 1000, 3, 0, 0, 0, 0 };
    CHECK(tsr_sim_op(m->sim(), cmd, nullptr) == TSR_ERR_NONE, "an op on the mock's simulator");
    CHECK(transport_perf(t, TSR_PERF_RETIRED) == 1 && transport_perf(t, tsr_perf_ch_rd_beats(1)) == 3, "PERF through the window");

    // Transfers through the transport: odd offsets around 4 GiB on the last channel.
    const uint32_t C = c->num_ch;
    rng r{ 99 };
    std::vector<uint8_t> big(300007), back(300007 + 8);
    r.fill(big.data(), big.size());
    const uint64_t off = (4ull << 30) - 4099;
    uint8_t * dram = tsr_sim_dram(m->sim(), C - 1);
    memset(dram + off - 64, 0x3C, 64);
    memset(dram + off + big.size() - 1, 0xC3, 64);   // first byte after the written range
    CHECK(tsr_transport_write(t, C - 1, off, big.data() + 1, big.size() - 1) == 0, "odd write");
    CHECK(memcmp(dram + off, big.data() + 1, big.size() - 1) == 0, "device bytes");
    CHECK(dram[off - 1] == 0x3C && dram[off + big.size() - 1] == 0xC3, "neighbours untouched");
    CHECK(tsr_transport_read(t, C - 1, off + 1, back.data() + 3, big.size() - 2) == 0 && memcmp(back.data() + 3, big.data() + 2, big.size() - 2) == 0,
          "odd read");
    CHECK(tsr_transport_write(t, 0, c->ch_bytes - 10, big.data(), 11) == TSR_TE_INVAL, "write past the channel refused");
    CHECK(tsr_transport_read(t, C, 0, back.data(), 1) == TSR_TE_INVAL, "bad channel refused");
    tsr_transport_stats st;
    tsr_transport_get_stats(t, &st);
    CHECK(st.bytes_written == big.size() - 1 && st.bytes_read == big.size() - 2 && st.csr_reads > 10 && st.csr_writes > 5,
          "stats: %" PRIu64 " B written, %" PRIu64 " B read, %" PRIu64 " / %" PRIu64 " CSR", st.bytes_written, st.bytes_read,
          st.csr_reads, st.csr_writes);
    CHECK(m->unmapped_accesses == 0, "protocol layer touched unmapped adapter registers %" PRIu64 " times", m->unmapped_accesses);
    tsr_transport_close(t);
}

// The image built without the DB block (HAS_DB 0: the maximum-clock tsr_u200_k4; TSR_SIM_DB=0 leaves the DB model out of the
// simulator): DBG_ID and the DBG window read 0, CTL_ID.DM reads 0, the DMI gate does not exist, and everything else the
// protocol needs (identity, PERF counters, the summary TSR_STATS prints) works without it. The core clock cannot be fitted
// from DBG_MTIME (it stays nominal, as on the simulated kinds).
static void test_no_db_block() {
    printf("protocol: the image without the DB block (TSR_SIM_DB=0)\n");
    setenv("TSR_SIM_DB", "0", 1);
    mock_options o = small_mock();
    mock_kdev * m = nullptr;
    tsr_transport * t = open_mock(o, &m);
    CHECK(t != nullptr, "open without the DB block");
    if (!t) {
        unsetenv("TSR_SIM_DB");
        return;
    }
    CHECK(tsr_transport_csr_read(t, TSR_CSR_DBG_ID) == 0, "DBG_ID reads 0");
    CHECK(!(tsr_transport_csr_read(t, TSR_CSR_CTL_ID) & TSR_CTL_ID_DM), "CTL_ID.DM reads 0");
    CHECK(tsr_transport_csr_read(t, TSR_CSR_DBG_CFG) == 0 && tsr_transport_csr_read(t, TSR_CSR_DBG_STATUS) == 0 &&
              tsr_transport_csr_read(t, TSR_CSR_DBG_HIST_HEAD) == 0 && tsr_transport_csr_read(t, TSR_CSR_DBG_MTIME_LO) == 0,
          "the DBG window reads 0");
    CHECK(tsr_transport_csr_write(t, TSR_CSR_DBG_CFG, TSR_DBG_CFG_TRW_EN) == 0 && tsr_transport_csr_read(t, TSR_CSR_DBG_CFG) == 0,
          "a write into the DBG window is ignored");
    CHECK(tsr_transport_dmi_enable(t, 1) == TSR_TE_STATE, "the DMI gate cannot be opened (no DBG block)");
    uint32_t v = 0;
    CHECK(tsr_transport_dmi_read(t, 0x11, &v) == 0 && v == 0, "a DMI read returns 0");
    // PERF works: the counters of the CSR block, and the summary the backend prints for TSR_STATS
    const uint32_t cmd[8] = { tsr_enc_ld(1, 0, 1000, 3).w[0], 0, 1000, 3, 0, 0, 0, 0 };
    CHECK(tsr_sim_op(m->sim(), cmd, nullptr) == TSR_ERR_NONE, "an op on the simulator without the DB model");
    CHECK(transport_perf(t, TSR_PERF_RETIRED) == 1 && transport_perf(t, tsr_perf_ch_rd_beats(1)) == 3, "PERF through the window");
    tsr_perf_summary ps;
    tsr_transport_perf_summary(t, &ps);
    CHECK(ps.perf[TSR_PERF_RETIRED] == 1 && ps.perf[TSR_PERF_TRW_RECORDS] == 0 && ps.perf[TSR_PERF_TRW_DROPS] == 0,
          "perf summary: RETIRED 1, no trace-writer counters");
    CHECK(ps.core_hz == CORE_HZ_NOMINAL && !ps.core_hz_fitted, "core clock nominal, not fitted");
    char sbuf[4096];
    CHECK(tsr_perf_summary_str(&ps, sbuf, sizeof(sbuf)) > 0 && strstr(sbuf, "nominal") != nullptr, "TSR_STATS text: %s", sbuf);
    tsr_transport_close(t);
    unsetenv("TSR_SIM_DB");
}

// A kdev that forwards to a mock, with hooks (fewer channels, a frozen LCQ, one failed core access).
struct wrap_kdev final : kdev {
    std::unique_ptr<mock_kdev> in;
    uint32_t ch_less = 0;
    bool     freeze_lcq = false;
    uint32_t lcq_seen = 0;
    uint32_t fail_read = UINT32_MAX, fail_write = UINT32_MAX;
    uint32_t read_failures = 0, write_failures = 0;
    bool write_then_fail = false;
    bool fail_reset_status = false;
    bool fail_reset_write = false;
    std::map<uint32_t, uint32_t> reads;
    explicit wrap_kdev(const mock_options & mo) : in(new mock_kdev(mo)) {}
    const char * name() const override { return "wrap"; }
    uint32_t num_ch() const override { return in->num_ch() - ch_less; }
    uint64_t ch_bytes(uint32_t ch) const override { return in->ch_bytes(ch); }
    uint64_t ch_addr(uint32_t ch) const override { return in->ch_addr(ch); }
    int reg_read(uint32_t off, uint32_t * val) override {
        if (off >= KREG_CSR_WINDOW) {
            const uint32_t core = off - KREG_CSR_WINDOW;
            reads[core]++;
            if (core == fail_read) {
                fail_read = UINT32_MAX;
                read_failures++;
                return -1;
            }
        }
        const int rc = in->reg_read(off, val);
        if (freeze_lcq && off == KREG_CSR_WINDOW + TSR_CSR_LCQ_WR) *val = lcq_seen;   // records never appear
        else if (off == KREG_CSR_WINDOW + TSR_CSR_LCQ_WR) lcq_seen = *val;
        return rc;
    }
    int reg_write(uint32_t off, uint32_t val) override {
        if (fail_reset_write && off == KREG_CSR_WINDOW + TSR_CSR_CTRL && (val & TSR_CTRL_SOFT_RESET)) {
            fail_reset_write = false;
            write_failures++;
            if (write_then_fail) (void) in->reg_write(off, val);
            return -1;
        }
        if (off >= KREG_CSR_WINDOW && off - KREG_CSR_WINDOW == fail_write) {
            fail_write = UINT32_MAX;
            write_failures++;
            if (write_then_fail) (void) in->reg_write(off, val);
            return -1;
        }
        const int rc = in->reg_write(off, val);
        if (rc == 0 && fail_reset_status && off == KREG_CSR_WINDOW + TSR_CSR_CTRL && (val & TSR_CTRL_SOFT_RESET)) {
            fail_read = TSR_CSR_STATUS;
            fail_reset_status = false;
        }
        return rc;
    }
    int mem_write(uint32_t ch, uint64_t off, const void * src, size_t n) override { return in->mem_write(ch, off, src, n); }
    int mem_read(uint32_t ch, uint64_t off, void * dst, size_t n) override { return in->mem_read(ch, off, dst, n); }
    uint64_t now_ns() override { return in->now_ns(); }
    void idle(uint32_t it, uint64_t ns) override { in->idle(it, ns); }
    tsr_sim * sim() override { return in->sim(); }
    bool simulated() const override { return true; }
};

// Opens that must fail: not a Tesseract core, another Tesseract Core capability major, a core without
// the control core (CTL_ID = 0, even with VERSION 0x0100), adapter major 2, NUM_CH vs the bound
// buffers, a channel buffer that is not 4 KiB aligned, no clk_rv, SOFT_RESET stuck. A higher
// capability minor opens.
static void test_open_refusals() {
    printf("protocol: refusals at open\n");
    mock_options o = small_mock();
    auto refused = [&](const char * what, const std::function<void(mock_kdev &)> & f) {
        std::unique_ptr<mock_kdev> mk(new mock_kdev(o));
        f(*mk);
        tsr_transport * t = transport_open_kdev(std::move(mk));
        CHECK(t == nullptr, "%s refused", what);
        if (t) tsr_transport_close(t);
    };
    refused("wrong ID", [](mock_kdev & m) { m.id_override = 0x12345678u; });
    refused("Tesseract Core capability 2.0", [](mock_kdev & m) { m.version_override = 0x0200u; });
    refused("CTL_ID = 0 (no control core) with VERSION 0x0100", [](mock_kdev & m) {
        m.version_override = 0x0100u;
        m.ctl_id_override = 0u;
    });
    {
        std::unique_ptr<mock_kdev> mk(new mock_kdev(o));
        mk->version_override = 0x0101u;     // capability 1.1: a minor only adds
        tsr_transport * t = transport_open_kdev(std::move(mk));
        CHECK(t != nullptr, "Tesseract Core capability 1.1 opens");
        if (t) {
            tsr_versions v;
            CHECK(tsr_transport_versions(t, &v) == 0 && v.core == 0x0101u && v.basis == 0 && v.n_kinds == 0,
                  "versions before a boot: core 0x%04x, no image", v.core);
            tsr_transport_close(t);
        }
    }
    refused("unaligned channel buffer", [](mock_kdev & m) { m.set_ch_addr(2, m.ch_addr(2) + 64); });
    refused("RV_FREQ = 0 (no clk_rv)", [](mock_kdev & m) { m.rv_freq_override = 0; });
    {
        mock_options o1 = o;
        o1.adapter_version = 0x00020004u;   // another adapter major
        std::unique_ptr<mock_kdev> mk(new mock_kdev(o1));
        CHECK(transport_open_kdev(std::move(mk)) == nullptr, "ADAPTER_VERSION major 2 refused");
        o1.adapter_version = 0x00010003u;   // reports 3 channels, 4 bound
        mk.reset(new mock_kdev(o1));
        CHECK(transport_open_kdev(std::move(mk)) == nullptr, "ADAPTER_VERSION.NUM_CH vs the bound buffers refused");
    }
    {
        std::unique_ptr<wrap_kdev> w(new wrap_kdev(o));
        w->ch_less = 1;
        CHECK(transport_open_kdev(std::move(w)) == nullptr, "CFG0.NUM_CH vs bound buffers refused");
    }
    // SOFT_RESET that never completes (a stuck unit): the open fails after TSR_RESET_MS.
    setenv("TSR_RESET_MS", "3", 1);
    refused("SOFT_RESET stuck at open", [](mock_kdev & m) { m.sr_stuck = true; });
    unsetenv("TSR_RESET_MS");
    CHECK(transport_open_kdev(nullptr) == nullptr, "no kdev");
    // Malformed environment.
    for (const char * kv : { "TSR_LAUNCH_TIMEOUT_MS=x", "TSR_ABORT_MS=0", "TSR_LOG_LEVEL=6", "TSR_ATTACH=2", "TSR_GDB_WAIT=always" }) {
        const std::string s = kv;
        const size_t eq = s.find('=');
        setenv(s.substr(0, eq).c_str(), s.substr(eq + 1).c_str(), 1);
        tsr_transport * t = tsr_transport_open("mock");
        CHECK(t == nullptr, "%s refused", kv);
        if (t) tsr_transport_close(t);
        unsetenv(s.substr(0, eq).c_str());
    }
}

// The adapter model's registers (EX §9.2).
static void test_mock_regs() {
    printf("mock adapter registers\n");
    mock_options o = small_mock(2);
    mock_kdev m(o);
    uint32_t v = 0;
    // channel bases: RW while MEM_EN = 0
    m.reg_write(KREG_CH_BASE0 + 8, 0x12345000u);
    m.reg_write(KREG_CH_BASE0 + 12, 0x40u);
    CHECK(m.reg_read(KREG_CH_BASE0 + 8, &v) == 0 && v == 0x12345000u, "ch1_base lo (0x%x)", v);
    CHECK(m.reg_read(KREG_CH_BASE0 + 12, &v) == 0 && v == 0x40u, "ch1_base hi (0x%x)", v);
    CHECK(m.reg_read(KREG_CH_BASE0, &v) == 0 && v == 0, "ch0_base untouched");
    // the old run registers are gone: read 0, writes ignored
    for (uint32_t off = 0; off < KREG_CH_BASE0; off += 4) {
        m.reg_write(off, 0xFFFFFFFFu);
        CHECK(m.reg_read(off, &v) == 0 && v == 0, "offset 0x%03x reads 0 (0x%x)", off, v);
    }
    CHECK(m.unmapped_accesses == 2 * (KREG_CH_BASE0 / 4), "unmapped accesses counted (%" PRIu64 ")", m.unmapped_accesses);
    CHECK(m.reg_read(KREG_ADAPTER_VERSION, &v) == 0 && v == 0x00010002u, "ADAPTER_VERSION 0x%x", v);
    CHECK(m.reg_read(KREG_SPACE, &v) != 0 && m.reg_read(2, &v) != 0, "outside / unaligned refused");
    // MEM_EN rising edge with a misaligned base: refused, BASE_MISALIGNED.
    m.reg_write(KREG_CH_BASE0, 0x2000u + 64);
    m.reg_write(KREG_ADAPTER_CTRL, TSR_ADAPTER_CTRL_MEM_EN);
    CHECK(m.reg_read(KREG_ADAPTER_CTRL, &v) == 0 && v == 0 && !m.mem_en(), "MEM_EN stays 0 with a misaligned base");
    CHECK(m.reg_read(KREG_ADAPTER_STATUS, &v) == 0 && (v & AST_BASE_ALIGN), "BASE_MISALIGNED (0x%x)", v);
    m.reg_write(KREG_ADAPTER_STATUS, 0xFFFFFFFFu);
    // REQ_HELD: an op while MEM_EN = 0 holds the model (time moves, the op does not).
    const uint32_t ld[8] = { tsr_enc_ld(0, 0, 100, 1).w[0], 0, 100, 1, 0, 0, 0, 0 };
    tsr_sim_push(m.sim(), ld, TSR_TICKET_NONE, TSR_TICKET_NONE, 0xFF00u);
    const uint64_t t0 = m.now_ns();
    m.idle(0, 1000000);
    CHECK(m.reg_read(KREG_ADAPTER_STATUS, &v) == 0 && (v & AST_REQ_HELD), "REQ_HELD while MEM_EN = 0 (0x%x)", v);
    CHECK(!(tsr_sim_csr_read(m.sim(), TSR_CSR_STATUS) & TSR_STATUS_QIDLE) && m.now_ns() - t0 >= 999000, "the op is held, time moved");
    // Aligned bases: MEM_EN latches them; a base write while MEM_EN = 1 is ignored (BASE_LOCKED).
    m.reg_write(KREG_CH_BASE0, (uint32_t) m.ch_addr(0));
    m.reg_write(KREG_CH_BASE0 + 4, (uint32_t) (m.ch_addr(0) >> 32));
    m.reg_write(KREG_CH_BASE0 + 8, (uint32_t) m.ch_addr(1));
    m.reg_write(KREG_CH_BASE0 + 12, (uint32_t) (m.ch_addr(1) >> 32));
    m.reg_write(KREG_ADAPTER_CTRL, TSR_ADAPTER_CTRL_MEM_EN);
    CHECK(m.mem_en() && m.latched_base(1) == m.ch_addr(1) && m.bind_errors == 0, "MEM_EN latched the bases");
    m.idle(0, 1000000);
    CHECK(tsr_sim_csr_read(m.sim(), TSR_CSR_STATUS) & TSR_STATUS_QIDLE, "the held op ran after MEM_EN");
    m.reg_write(KREG_CH_BASE0 + 8, 0x5000u);
    CHECK(m.reg_read(KREG_CH_BASE0 + 8, &v) == 0 && v == (uint32_t) m.ch_addr(1), "base write ignored while MEM_EN = 1");
    CHECK(m.reg_read(KREG_ADAPTER_STATUS, &v) == 0 && (v & AST_BASE_LOCKED) && (v & AST_REQ_HELD), "BASE_LOCKED (0x%x)", v);
    // ADAPTER_STATUS W1C: bits [5:0] one by one, any 1 in [31:16] clears the whole count; plat_err_i
    m.reg_write(KREG_ADAPTER_STATUS, 0xFFFFFFFFu);
    m.inject_status(AST_RRESP | AST_ORPHAN | (3u << 16));
    CHECK(m.reg_read(KREG_ADAPTER_STATUS, &v) == 0 && v == (AST_RRESP | AST_ORPHAN | (3u << 16)), "injected 0x%x", v);
    CHECK((m.reg_read(KREG_CSR_WINDOW + TSR_CSR_STATUS, &v) == 0) && (v & TSR_STATUS_PLAT_ERR), "plat_err_i: STATUS.PLAT_ERR");
    m.reg_write(KREG_ADAPTER_STATUS, 0x00010000u);
    CHECK(m.reg_read(KREG_ADAPTER_STATUS, &v) == 0 && v == (AST_RRESP | AST_ORPHAN), "count cleared as a whole (0x%x)", v);
    m.reg_write(KREG_ADAPTER_STATUS, AST_ORPHAN);
    CHECK(m.reg_read(KREG_ADAPTER_STATUS, &v) == 0 && v == AST_RRESP, "bit2 cleared alone (0x%x)", v);
    m.reg_write(KREG_ADAPTER_STATUS, 0xFFFFFFFFu);
    CHECK(m.reg_read(KREG_ADAPTER_STATUS, &v) == 0 && v == 0, "all cleared (0x%x)", v);
    CHECK((m.reg_read(KREG_CSR_WINDOW + TSR_CSR_STATUS, &v) == 0) && !(v & TSR_STATUS_PLAT_ERR), "plat_err_i low again");
    m.inject_status(0xFFFF0000u);
    m.inject_status(5u << 16);
    CHECK(m.reg_read(KREG_ADAPTER_STATUS, &v) == 0 && v == 0xFFFF0000u, "count saturates (0x%x)", v);
    // the CSR window reaches the simulator
    CHECK(m.reg_read(KREG_CSR_WINDOW + TSR_CSR_CFG0, &v) == 0 && v == (2u | (128u << 8) | (1u << 20)), "window CFG0 0x%x", v);
    const size_t n0 = m.window_write_log.size();
    m.reg_write(KREG_CSR_WINDOW + TSR_CSR_LOG_CFG, 0xF1u);
    CHECK(m.window_write_log.size() == n0 + 1 && m.window_write_log.back() == TSR_CSR_LOG_CFG, "window write logged");
}

// The adapter log: what the protocol layer saw in ADAPTER_STATUS at open (it clears the register).
static void test_adapter_log() {
    printf("protocol: ADAPTER_STATUS at open\n");
    mock_options o = small_mock(1);
    std::unique_ptr<mock_kdev> mk(new mock_kdev(o));
    mock_kdev * m = mk.get();
    m->inject_status(AST_RRESP | (2u << 16));   // left by an earlier process
    tsr_transport * t = transport_open_kdev(std::move(mk));
    CHECK(t != nullptr, "open");
    if (!t) return;
    const adapter_log lg = transport_adapter_log(t);
    CHECK(lg.at_open == (AST_RRESP | (2u << 16)), "log at open (0x%x)", lg.at_open);
    uint32_t v = 1;
    CHECK(m->reg_read(KREG_ADAPTER_STATUS, &v) == 0 && v == 0, "cleared at open (0x%x)", v);
    CHECK(transport_adapter_log(nullptr).at_open == 0, "no transport: empty log");
    tsr_transport_close(t);
    tsr_transport * ts = tsr_transport_open("sim");
    CHECK(ts != nullptr && transport_adapter_log(ts).at_open == 0 && transport_kdev(ts) != nullptr && !transport_kdev(ts)->has_adapter(),
          "sim: empty log, no adapter");
    tsr_transport_close(ts);
}

// A booted device with a launch running, left by a process that died; then TSR_ATTACH and a normal
// open (SOFT_RESET, warm boot).
static void test_stale_core() {
    printf("protocol: stale core at open\n");
    tsr_image * img = nullptr;
    if (tsr_image_open(TSR_FW_TEST_IMAGE_fw_rt, &img) != 0) {
        CHECK(0, "fw_rt image");
        return;
    }
    mock_kdev * m = nullptr;
    tsr_transport * a = open_mock(small_mock(), &m);
    CHECK(a != nullptr && tsr_transport_boot(a, img) == 0, "process A: boot");
    if (!a) return;
    const tsr_launch l = L(TSR_DESC_TYPE_LAUNCH, (uint16_t) tsr_image_kernel(img, "rt_stop"), { 0xFFFFFFF0u, 1 });
    uint32_t q = 0;
    CHECK(tsr_transport_launch(a, &l, &q) == 0, "process A: a long launch");
    for (int i = 0; i < 100 && TSR_GET(tsr_transport_csr_read(a, TSR_CSR_RT_STATE), TSR_RT_STATE_PHASE) != TSR_RT_STATE_PHASE_KERNEL; ++i)
        (void) tsr_transport_csr_read(a, TSR_CSR_STATUS);   // runs the model (sim kinds)
    std::unique_ptr<kdev> dead = transport_detach_kdev(a);   // process A dies
    uint32_t st = 0;
    dead->reg_read(KREG_CSR_WINDOW + TSR_CSR_STATUS, &st);
    CHECK((st & TSR_STATUS_RV_RUNNING) && (st & TSR_STATUS_ACTIVE), "the core is left running a launch (STATUS 0x%08x)", st);
    // TSR_ATTACH=1: no reset, no bind; the state is visible; launches refused until a boot.
    setenv("TSR_ATTACH", "1", 1);
    tsr_transport * b = transport_open_kdev(std::move(dead));
    unsetenv("TSR_ATTACH");
    CHECK(b != nullptr, "TSR_ATTACH open");
    if (!b) return;
    CHECK(TSR_GET(tsr_transport_csr_read(b, TSR_CSR_RT_STATE), TSR_RT_STATE_PHASE) == TSR_RT_STATE_PHASE_KERNEL &&
          (tsr_transport_csr_read(b, TSR_CSR_STATUS) & TSR_STATUS_ACTIVE), "attach: the running launch is still there");
    tsr_record r;
    const tsr_launch ping = L(TSR_DESC_TYPE_PING, 0, { 5 });
    CHECK(tsr_transport_call(b, &ping, TSR_TIMEOUT_DEFAULT, &r) == TSR_TE_STATE, "attach: launches refused before a boot");
    dead = transport_detach_kdev(b);
    // A normal open: SOFT_RESET stops the core (EX §8.5), then a warm boot (the ITCM is intact).
    tsr_transport * c = transport_open_kdev(std::move(dead));
    CHECK(c != nullptr, "normal open of the stale core");
    if (!c) return;
    st = tsr_transport_csr_read(c, TSR_CSR_STATUS);
    CHECK(!(st & (TSR_STATUS_RV_RUNNING | TSR_STATUS_ACTIVE)) && (st & TSR_STATUS_QIDLE), "SOFT_RESET at open stopped it (STATUS 0x%08x)", st);
    CHECK(tsr_transport_csr_read(c, TSR_CSR_LQ_TAKEN) == tsr_transport_csr_read(c, TSR_CSR_LQ_SEQ) &&
          tsr_transport_csr_read(c, TSR_CSR_LCQ_RD) == tsr_transport_csr_read(c, TSR_CSR_LCQ_WR), "rings flushed");
    CHECK(tsr_transport_boot(c, img) == 0 && transport_boot_info(c).warm, "warm boot after the stale core");
    uint32_t q2 = 0;
    CHECK(tsr_transport_launch(c, &ping, &q2) == 0 && q2 == q + 1, "the sequence continues after the dead launch (%u, %u)", q2, q);
    CHECK(tsr_transport_wait(c, q2, TSR_TIMEOUT_DEFAULT, &r) == 0 && r.w[2] == 5, "PING");
    tsr_transport_close(c);
    tsr_image_close(img);
}

// stderr of a code section (the transport's messages), echoed to stdout for the ctest log.
static std::string capture_stderr(const std::function<void()> & f) {
    fflush(stderr);
    FILE * tmp = tmpfile();
    const int saved = dup(2);
    if (tmp == nullptr || saved < 0 || dup2(fileno(tmp), 2) < 0) {
        CHECK(0, "stderr capture");
        f();
        return std::string();
    }
    f();
    fflush(stderr);
    dup2(saved, 2);
    close(saved);
    std::string s;
    rewind(tmp);
    char b[4096];
    for (size_t n; (n = fread(b, 1, sizeof(b), tmp)) > 0;) s.append(b, n);
    fclose(tmp);
    if (!s.empty()) printf("  [stderr] %s", s.c_str());
    return s;
}

static bool has(const std::string & s, const char * sub) { return s.find(sub) != std::string::npos; }

// A process that used the device, then a new open of it (the ring counters count on: only a card
// reset zeroes them). The open must start at the device's read pointers: no record or log word the
// earlier process consumed is taken for a new one ("HELLO outside a boot", an answer with no launch
// outstanding, an LCQ resync), and what it left unread is discarded by the SOFT_RESET, not decoded
// (no stale log in this process's log or tsr_transport_log_drain).
static void test_stale_reopen() {
    printf("protocol: reopen after an earlier process\n");
    tsr_image * img = nullptr;
    if (tsr_image_open(TSR_FW_TEST_IMAGE_fw_rt, &img) != 0) {
        CHECK(0, "fw_rt image");
        return;
    }
    const tsr_launch ping = L(TSR_DESC_TYPE_PING, 0, { 0x51u });
    auto stale_msgs = [](const std::string & e) {
        return has(e, "HELLO outside a boot") || has(e, "no launch outstanding") || has(e, "resynchronising") || has(e, "out of order");
    };
    for (int pings : { 3, 20 }) {
        mock_kdev * m = nullptr;
        tsr_transport * a = open_mock(small_mock(), &m);
        CHECK(a != nullptr && tsr_transport_boot(a, img) == 0, "process A: boot");
        if (!a) continue;
        tsr_record r;
        for (int i = 0; i < pings; ++i) CHECK(tsr_transport_call(a, &ping, TSR_TIMEOUT_DEFAULT, &r) == 0, "process A: PING %d", i);
        const uint32_t last = r.w[1];
        std::unique_ptr<kdev> d = transport_detach_kdev(a);   // process A exits
        tsr_transport * b = nullptr;
        const std::string e = capture_stderr([&] { b = transport_open_kdev(std::move(d)); });
        CHECK(b != nullptr, "reopen after %d PINGs", pings);
        if (!b) continue;
        CHECK(!stale_msgs(e) && !has(e, "discards"), "reopen after %d consumed PINGs: no stale record taken (stderr: %s)", pings, e.c_str());
        const std::string e2 = capture_stderr([&] {
            CHECK(tsr_transport_boot(b, img) == 0 && transport_boot_info(b).warm, "reopen: warm boot");
            uint32_t q = 0;
            CHECK(tsr_transport_launch(b, &ping, &q) == 0 && q == last + 1, "reopen: the sequence continues (%u after %u)", q, last);
            CHECK(tsr_transport_wait(b, q, TSR_TIMEOUT_DEFAULT, &r) == 0 && r.w[1] == q && r.w[2] == 0x51u, "reopen: PING");
        });
        CHECK(!stale_msgs(e2), "reopen after %d PINGs: boot and launch see no stale record (stderr: %s)", pings, e2.c_str());
        tsr_transport_close(b);
    }
    {
        // Process A leaves an answered rt_log launch (5 FMT log records) and a PONG unread.
        mock_kdev * m = nullptr;
        tsr_transport * a = open_mock(small_mock(), &m);
        CHECK(a != nullptr && tsr_transport_boot(a, img) == 0, "process A: boot");
        if (a) {
            const tsr_launch lg = L(TSR_DESC_TYPE_LAUNCH, (uint16_t) tsr_image_kernel(img, "rt_log"), { 1, 5, 0xABu });
            uint32_t q0 = 0, q1 = 0;
            CHECK(tsr_transport_launch(a, &lg, &q0) == 0 && tsr_transport_launch(a, &ping, &q1) == 0, "process A: 2 launches");
            for (int i = 0; i < 1000 && tsr_transport_csr_read(a, TSR_CSR_LCQ_WR) - tsr_transport_csr_read(a, TSR_CSR_LCQ_RD) < 2; ++i) {
            }
            const uint32_t unread = tsr_transport_csr_read(a, TSR_CSR_LCQ_WR) - tsr_transport_csr_read(a, TSR_CSR_LCQ_RD);
            const uint32_t words = tsr_transport_csr_read(a, TSR_CSR_LOG_WR) - tsr_transport_csr_read(a, TSR_CSR_LOG_RD);
            CHECK(unread == 2 && words > 0, "process A leaves 2 records (%u) and log words (%u) unread", unread, words);
            std::unique_ptr<kdev> d = transport_detach_kdev(a);
            tsr_transport * b = nullptr;
            const std::string e = capture_stderr([&] { b = transport_open_kdev(std::move(d)); });
            CHECK(b != nullptr, "reopen");
            if (b) {
                char want[96];
                snprintf(want, sizeof(want), "discards %u LCQ records and %u log words", unread, words);
                CHECK(has(e, want) && !stale_msgs(e), "reopen: the unread records are discarded, not taken (stderr: %s)", e.c_str());
                std::vector<uint32_t> w(4096);
                size_t n = 1;
                CHECK(tsr_transport_log_drain(b, w.data(), w.size(), &n) == 0 && n == 0, "reopen: no stale log words (%zu)", n);
                tsr_record r;
                const std::string e2 = capture_stderr([&] {
                    CHECK(tsr_transport_boot(b, img) == 0, "reopen: boot");
                    CHECK(tsr_transport_call(b, &ping, TSR_TIMEOUT_DEFAULT, &r) == 0 && r.w[1] == q1 + 1 && r.w[2] == 0x51u, "reopen: PING");
                });
                CHECK(!stale_msgs(e2), "reopen: no stale record later (stderr: %s)", e2.c_str());
                tsr_transport_close(b);
            }
        }
    }
    tsr_image_close(img);
}

// EX §4.9: a watermark covers q iff (s - q) mod 2^32 < 2^31 and stays valid until SOFT_RESET, so
// the transport moves a valid one below the oldest unanswered launch every 2^30 launches (here
// every 4, transport_set_wm_refresh) and never once SOFT_RESET has invalidated it.
static void test_watermark_refresh() {
    printf("protocol: watermark refresh\n");
    tsr_image * img = nullptr;
    if (tsr_image_open(TSR_FW_TEST_IMAGE_fw_rt, &img) != 0) return;
    mock_kdev * m = nullptr;
    tsr_transport * t = open_mock(small_mock(), &m);
    if (t && tsr_transport_boot(t, img) == 0) {
        transport_set_wm_refresh(t, 4);
        const tsr_launch ping = L(TSR_DESC_TYPE_PING, 0, { 0x77u });
        tsr_record r;
        CHECK(tsr_transport_call(t, &ping, TSR_TIMEOUT_DEFAULT, &r) == 0, "PING");
        const uint32_t q0 = r.w[1];
        CHECK(tsr_transport_cancel(t, q0) == 0 && tsr_transport_stop(t, q0) == 0, "cancel and stop at the last answered launch");
        bool ok = true;
        for (int i = 0; i < 12; ++i)
            ok = ok && tsr_transport_call(t, &ping, TSR_TIMEOUT_DEFAULT, &r) == 0 && st_of(r) == 0 && r.w[2] == 0x77u;
        CHECK(ok, "launches after the watermarks are not covered (r0 0x%08x)", r.w[0]);
        const uint32_t pushed = tsr_transport_csr_read(t, TSR_CSR_LQ_SEQ);
        const uint32_t c = tsr_transport_csr_read(t, TSR_CSR_LQ_CANCEL), s = tsr_transport_csr_read(t, TSR_CSR_LQ_STOP);
        CHECK(c != q0 && s != q0 && (int32_t) (c - q0) > 0 && (int32_t) (pushed - 1 - c) >= 0 && pushed - 1 - c <= 4 && s == c,
              "watermarks moved up behind the answered launches: LQ_CANCEL %u, LQ_STOP %u (written at %u; LQ_SEQ %u)", c, s, q0, pushed);
        // After a SOFT_RESET both are invalid: never rewritten.
        CHECK(tsr_transport_soft_reset(t) == 0 && tsr_transport_boot(t, img) == 0, "soft reset + boot");
        const size_t n0 = m->window_write_log.size();
        for (int i = 0; i < 12; ++i) CHECK(tsr_transport_call(t, &ping, TSR_TIMEOUT_DEFAULT, &r) == 0 && st_of(r) == 0, "PING");
        uint32_t wm_writes = 0;
        for (size_t j = n0; j < m->window_write_log.size(); ++j)
            wm_writes += m->window_write_log[j] == TSR_CSR_LQ_CANCEL || m->window_write_log[j] == TSR_CSR_LQ_STOP;
        CHECK(wm_writes == 0, "no watermark write after SOFT_RESET invalidated them (%u)", wm_writes);
        // A watermark ahead of PUSHED covers launches to come: kept, they answer ABORTED.
        const uint32_t ahead = tsr_transport_csr_read(t, TSR_CSR_LQ_SEQ) + 8;
        CHECK(tsr_transport_cancel(t, ahead) == 0, "cancel ahead of PUSHED");
        uint32_t aborted = 0;
        for (int i = 0; i < 6; ++i) {
            CHECK(tsr_transport_call(t, &ping, TSR_TIMEOUT_DEFAULT, &r) == 0, "PING");
            aborted += st_of(r) == TSR_LCQ_R0_STATUS_ABORTED;
        }
        CHECK(aborted == 6 && tsr_transport_csr_read(t, TSR_CSR_LQ_CANCEL) == ahead,
              "a watermark ahead of PUSHED is kept (%u of 6 covered launches ABORTED, LQ_CANCEL %u, written %u)", aborted,
              tsr_transport_csr_read(t, TSR_CSR_LQ_CANCEL), ahead);
    } else {
        CHECK(0, "open + boot");
    }
    tsr_transport_close(t);
    tsr_image_close(img);
}

// Timeouts at boot: no HELLO within TSR_BOOT_MS (device time on the simulator kinds).
static void test_boot_timeout() {
    printf("protocol: no HELLO\n");
    tsr_image * img = nullptr;
    if (tsr_image_open(TSR_FW_TEST_IMAGE_fw_rt, &img) != 0) return;
    std::unique_ptr<wrap_kdev> w(new wrap_kdev(small_mock()));
    wrap_kdev * wk = w.get();
    tsr_transport * t = transport_open_kdev(std::move(w));
    CHECK(t != nullptr, "open");
    if (t) {
        transport_set_timeouts(t, 0, 0, 3, 0);   // TSR_BOOT_MS 3
        wk->freeze_lcq = true;
        const uint64_t t0 = wk->now_ns();
        CHECK(tsr_transport_boot(t, img) == TSR_TE_TIMEOUT, "boot without HELLO times out");
        CHECK(wk->now_ns() - t0 >= 3000000, "after TSR_BOOT_MS of device time");
        wk->freeze_lcq = false;
        CHECK(tsr_transport_boot(t, img) == 0, "boot again (the HELLO now visible)");
        tsr_transport_close(t);
    }
    tsr_image_close(img);
}

// Raw I/O failures must leave the entire LCQ batch retryable, including a prefix read before the
// failed word. Publishing an answer before LCQ_RD succeeds would also return launch credit early.
static void test_lcq_io_failures() {
    printf("protocol: LCQ read and acknowledgement failures\n");
    tsr_image * img = nullptr;
    if (tsr_image_open(TSR_FW_TEST_IMAGE_fw_rt, &img) != 0) {
        CHECK(0, "fw_rt image");
        return;
    }
    auto open = [](wrap_kdev ** out) {
        std::unique_ptr<wrap_kdev> w(new wrap_kdev(small_mock()));
        *out = w.get();
        return transport_open_kdev(std::move(w));
    };
    auto ready = [](tsr_transport * t, wrap_kdev * w, uint32_t count) {
        const uint32_t rd = tsr_transport_csr_read(t, TSR_CSR_LCQ_RD);
        uint32_t wr = rd;
        for (uint32_t i = 0; i < 1000 && wr - rd < count; ++i) {
            w->in->reg_read(KREG_CSR_WINDOW + TSR_CSR_LCQ_WR, &wr);
            if (wr - rd < count) w->idle(i, 100000);
        }
        CHECK(wr - rd == count, "%u LCQ records ready (%u)", count, wr - rd);
        return rd;
    };
    auto collect = [](void * p, const tsr_record * r) { ((std::vector<tsr_record> *) p)->push_back(*r); };
    auto word_reads = [](wrap_kdev * w, uint32_t word) {
        uint32_t n = 0;
        for (uint32_t slot = 0; slot < TSR_LCQ_DEPTH; ++slot) n += w->reads[TSR_CSR_LCQ_REC(slot, word)];
        return n;
    };
    for (uint32_t word : { 0u, 2u, 4u }) {
        for (uint32_t bad : { 0u, 1u }) {
            wrap_kdev * w = nullptr;
            tsr_transport * t = open(&w);
            CHECK(t != nullptr, "LCQ I/O open");
            if (!t) continue;
            if (tsr_transport_boot(t, img) != 0) {
                CHECK(0, "LCQ I/O boot");
                tsr_transport_close(t);
                continue;
            }
            std::vector<tsr_record> events;
            tsr_transport_set_event_cb(t, collect, &events);
            uint32_t seq[TSR_LQ_DEPTH] = {};
            for (uint32_t i = 0; i < TSR_LQ_DEPTH; ++i) {
                const tsr_launch l = L(word == 4 && i == bad ? 0xff : TSR_DESC_TYPE_PING, 0, { 0x1200u + i });
                CHECK(tsr_transport_launch(t, &l, &seq[i]) == 0, "fill launch credit %u", i);
            }
            const uint32_t rd = ready(t, w, TSR_LQ_DEPTH);
            const uint32_t off = TSR_CSR_LCQ_REC((rd + bad) % TSR_LCQ_DEPTH, word);
            tsr_record r{};
            for (uint32_t attempt = 0; attempt < 2; ++attempt) {
                w->fail_read = off;
                CHECK(tsr_transport_poll(t, seq[0], &r) == TSR_TE_IO,
                      "word %u, record %u, attempt %u: I/O error, no published prefix", word, bad, attempt);
                CHECK(tsr_transport_csr_read(t, TSR_CSR_LCQ_RD) == rd && events.empty(), "failed batch is not acknowledged or delivered");
            }
            CHECK(w->read_failures == 2, "the same unread record is retried (%u failures)", w->read_failures);
            const uint32_t pushed = tsr_transport_csr_read(t, TSR_CSR_LQ_SEQ);
            w->fail_read = off;
            const tsr_launch extra = L(TSR_DESC_TYPE_PING, 0, { 99u });
            uint32_t extra_seq = 0;
            CHECK(tsr_transport_launch(t, &extra, &extra_seq) == TSR_TE_IO && tsr_transport_csr_read(t, TSR_CSR_LQ_SEQ) == pushed,
                  "a failed full batch returns no launch credit");
            w->fail_read = UINT32_MAX;
            w->reads.clear();
            bool recovered = true;
            for (uint32_t i = 0; i < TSR_LQ_DEPTH; ++i) {
                const int rc = tsr_transport_poll(t, seq[i], &r);
                const bool good = rc == 1 && r.w[1] == seq[i] &&
                    (word == 4 && i == bad ? st_of(r) == TSR_LCQ_R0_STATUS_BAD_DESC && r.w[4] == 0u
                                          : st_of(r) == 0 && r.w[2] == 0x1200u + i);
                CHECK(good, "retry recovers exact answer %u (rc %d, r0 %x, r2 %x, r4 %x)", i, rc, r.w[0], r.w[2], r.w[4]);
                recovered = recovered && good;
            }
            CHECK(word_reads(w, 0) == TSR_LQ_DEPTH && word_reads(w, 2) == TSR_LQ_DEPTH,
                  "all in-order PONGs retain minimal reads across a batch");
            for (uint32_t j : { 1u, 3u, 4u, 5u, 6u, 7u })
                CHECK(word_reads(w, j) == (word == 4 ? 1u : 0u), "word %u read only for the full error record", j);
            CHECK(events.empty(), "no failed read becomes an event");
            if (recovered) {   // keep a broken implementation's test bounded: never wait for a lost completion
                const uint64_t reads0 = w->in->window_reads;
                for (uint32_t i = 0; i < TSR_LQ_DEPTH; ++i) {
                    const tsr_launch l = L(TSR_DESC_TYPE_PING, 0, { i });
                    CHECK(tsr_transport_launch(t, &l, &seq[i]) == 0, "all launch credits returned after retry (%u)", i);
                }
                CHECK(w->in->window_reads == reads0, "launching a full ring needs no credit read after recovery");
                ready(t, w, TSR_LQ_DEPTH);
                for (uint32_t i = 0; i < TSR_LQ_DEPTH; ++i)
                    CHECK(tsr_transport_poll(t, seq[i], &r) == 1 && r.w[2] == i, "next batch has no stale answer (%u)", i);
                CHECK(events.empty(), "no duplicate event in the next batch");
            }
            tsr_transport_close(t);
        }
    }
    // A real event between answers: neither a later payload failure nor a failed batch ack may
    // publish it, and retry delivers it once. The second failed poll also checks cached answers.
    for (uint32_t mode : { 0u, 1u, 2u }) {
        const bool ack = mode != 0u;
        wrap_kdev * w = nullptr;
        tsr_transport * t = open(&w);
        if (t && tsr_transport_boot(t, img) == 0) {
            std::vector<tsr_record> events;
            tsr_transport_set_event_cb(t, collect, &events);
            const tsr_launch ping = L(TSR_DESC_TYPE_PING, 0, { 17u });
            const tsr_launch prof = L(TSR_DESC_TYPE_LAUNCH, (uint16_t) tsr_image_kernel(img, "rt_prof"));
            uint32_t sp = 0, se = 0;
            CHECK(tsr_transport_launch(t, &ping, &sp) == 0 && tsr_transport_launch(t, &prof, &se) == 0, "PING and event launch");
            const uint32_t rd = ready(t, w, 3);
            w->write_then_fail = mode == 2u;
            std::vector<uint32_t> later;
            tsr_record r{};
            for (uint32_t attempt = 0; attempt < 2; ++attempt) {
                if (ack) w->fail_write = TSR_CSR_LCQ_RD;
                else w->fail_read = TSR_CSR_LCQ_REC((rd + 1u) % TSR_LCQ_DEPTH, 4);
                const int rc = ack && attempt == 0u ? tsr_transport_wait(t, sp, TSR_TIMEOUT_DEFAULT, &r)
                                                   : tsr_transport_poll(t, sp, &r);
                CHECK(rc == TSR_TE_IO, "event batch: retry the whole failed %s", ack ? "ack" : "read");
                CHECK(events.empty() && tsr_transport_csr_read(t, TSR_CSR_LCQ_RD) == rd + (mode == 2u ? 3u : 0u),
                      "no event/answer published before ack succeeds");
                CHECK(transport_recovery_log(t).timeouts == 0, "an I/O failure does not become a timeout recovery");
                CHECK(tsr_transport_stop(t, se) == 0, "stop watermark while a batch awaits acknowledgement");
                if (mode == 2u && attempt == 0u) {
                    // A failed write can already have reached the device. More answers overwrite
                    // a freed slot, so ack retry must use its saved batch, not reread that slot.
                    for (uint32_t i = 0; i < TSR_LQ_DEPTH - 2u; ++i) {
                        const tsr_launch next = L(TSR_DESC_TYPE_PING, 0, { 100u + i });
                        uint32_t seq = 0;
                        CHECK(tsr_transport_launch(t, &next, &seq) == 0, "launch behind a failed applied ack");
                        later.push_back(seq);
                    }
                    ready(t, w, (uint32_t) later.size());
                }
            }
            w->fail_read = w->fail_write = UINT32_MAX;
            CHECK(tsr_transport_poll(t, sp, &r) == 1 && r.w[2] == 17u, "event batch: exact PONG after retry");
            CHECK(tsr_transport_poll(t, se, &r) == 1 && st_of(r) == 0, "event batch: exact DONE after retry");
            CHECK(events.size() == 1 && events[0].w[1] == se, "one event after retry (%zu)", events.size());
            for (uint32_t i = 0; i < later.size(); ++i)
                CHECK(tsr_transport_poll(t, later[i], &r) == 1 && r.w[2] == 100u + i, "new answer does not overwrite saved batch (%u)", i);
            CHECK(tsr_transport_call(t, &ping, TSR_TIMEOUT_DEFAULT, &r) == 0 && r.w[2] == 17u && events.size() == 1,
                  "the next launch delivers no duplicate PONG/event");
        } else CHECK(0, "event batch open + boot");
        tsr_transport_close(t);
    }
    // Failed HELLO reads/ack are transport errors, not an image mismatch or a consumed greeting.
    for (uint32_t word : { 0u, 4u, 8u, 9u }) {
        wrap_kdev * w = nullptr;
        tsr_transport * t = open(&w);
        if (t) {
            const uint32_t rd = tsr_transport_csr_read(t, TSR_CSR_LCQ_RD);
            w->write_then_fail = word == 9u;
            if (word >= 8) w->fail_write = TSR_CSR_LCQ_RD;
            else w->fail_read = TSR_CSR_LCQ_REC(rd % TSR_LCQ_DEPTH, word);
            CHECK(tsr_transport_boot(t, img) == TSR_TE_IO, "HELLO %s failure is I/O", word >= 8 ? "ack" : "read");
            CHECK(tsr_transport_csr_read(t, TSR_CSR_LCQ_RD) == rd + (word == 9 ? 1u : 0u) && tsr_transport_image(t) == nullptr,
                  "HELLO failure does not publish a booted image, even if the ack reached the device");
            CHECK(tsr_transport_boot(t, img) == 0, "boot retries after a failed HELLO access");
            const tsr_launch ping = L(TSR_DESC_TYPE_PING, 0, { 29u });
            tsr_record r{};
            CHECK(tsr_transport_call(t, &ping, TSR_TIMEOUT_DEFAULT, &r) == 0 && r.w[2] == 29u, "PING after HELLO retry");
        } else CHECK(0, "HELLO I/O open");
        tsr_transport_close(t);
    }
    // TRAP classification needs LCQ_ANSWERED in addition to its payload. A failed classification
    // read must not consume the earlier PONG or lose the launch's TRAP answer.
    {
        wrap_kdev * w = nullptr;
        tsr_transport * t = open(&w);
        if (t && tsr_transport_boot(t, img) == 0) {
            tsr_transport_set_debugger(t, 1);   // keep the TRAP rather than restarting automatically
            const tsr_launch ping = L(TSR_DESC_TYPE_PING, 0, { 31u });
            const tsr_launch trap = L(TSR_DESC_TYPE_LAUNCH, (uint16_t) tsr_image_kernel(img, "rt_trap"), { 0u });
            uint32_t sp = 0, st = 0;
            CHECK(tsr_transport_launch(t, &ping, &sp) == 0 && tsr_transport_launch(t, &trap, &st) == 0, "PING and TRAP launch");
            const uint32_t rd = ready(t, w, 2);
            tsr_record r{};
            for (uint32_t attempt = 0; attempt < 2; ++attempt) {
                w->fail_read = TSR_CSR_LCQ_ANSWERED;
                CHECK(tsr_transport_poll(t, sp, &r) == TSR_TE_IO && tsr_transport_csr_read(t, TSR_CSR_LCQ_RD) == rd,
                      "TRAP classification failure keeps the batch retryable");
            }
            CHECK(tsr_transport_poll(t, sp, &r) == 1 && r.w[2] == 31u, "PONG before TRAP recovered");
            CHECK(tsr_transport_poll(t, st, &r) == 1 && TSR_GET(r.w[0], TSR_LCQ_R0_TYPE) == TSR_LCQ_R0_TYPE_TRAP,
                  "TRAP still answers its launch after retry");
            CHECK(tsr_transport_soft_reset(t) == 0 && tsr_transport_boot(t, img) == 0, "reset and boot after TRAP");
        } else CHECK(0, "TRAP I/O open + boot");
        tsr_transport_close(t);
    }
    // A reset deliberately discards an unacknowledged batch and reports its launches LOST. Close
    // also discards it without delivering events or dereferencing it after device destruction.
    for (uint32_t mode : { 0u, 1u, 2u, 3u, 4u, 5u }) {
        wrap_kdev * w = nullptr;
        tsr_transport * t = open(&w);
        if (t && tsr_transport_boot(t, img) == 0) {
            std::vector<tsr_record> events;
            tsr_transport_set_event_cb(t, collect, &events);
            const tsr_launch prof = L(TSR_DESC_TYPE_LAUNCH, (uint16_t) tsr_image_kernel(img, "rt_prof"));
            uint32_t seq = 0;
            CHECK(tsr_transport_launch(t, &prof, &seq) == 0, "launch before pending-batch reset/close");
            ready(t, w, 2);
            w->write_then_fail = mode == 1u;
            w->fail_write = TSR_CSR_LCQ_RD;
            tsr_record r{};
            CHECK(tsr_transport_poll(t, seq, &r) == TSR_TE_IO && events.empty(), "keep pending batch before reset/close");
            if (mode == 2u) {
                tsr_transport_close(t);
                t = nullptr;
                CHECK(events.empty(), "close does not deliver the pending event");
            } else {
                w->fail_write = TSR_CSR_LCQ_RD;   // the reset's consume cannot finish this batch
                w->fail_reset_status = mode == 3u;
                w->fail_reset_write = mode >= 4u;
                w->write_then_fail = mode == 1u || mode == 5u;
                CHECK(tsr_transport_soft_reset(t) == (mode >= 3u ? TSR_TE_IO : 0), "reset flushes pending batch after another ack failure");
                if (mode >= 3u) {
                    CHECK(tsr_transport_poll(t, seq, &r) == TSR_TE_STATE && events.empty(), "an uncertain reset blocks old LCQ consumption");
                    uint32_t next = 0;
                    CHECK(tsr_transport_launch(t, &prof, &next) == TSR_TE_STATE, "an uncertain reset blocks new launches");
                }
                CHECK(tsr_transport_boot(t, img) == 0, "boot after pending-batch reset");
                CHECK(tsr_transport_poll(t, seq, &r) == TSR_TE_LOST && events.empty(), "discarded batch returns LOST without an event");
                const tsr_launch ping = L(TSR_DESC_TYPE_PING, 0, { 37u });
                CHECK(tsr_transport_call(t, &ping, TSR_TIMEOUT_DEFAULT, &r) == 0 && r.w[2] == 37u && events.empty(),
                      "reset leaves no saved batch to replay after boot");
            }
        } else CHECK(0, "pending-batch reset/close open + boot");
        tsr_transport_close(t);
    }
    tsr_image_close(img);

    // graph's successful RET is defined as zero: do not add an r2 read while fixing failed reads.
    if (access(TSR_FW_DEFAULT_IMAGE, R_OK) == 0 && tsr_image_open(TSR_FW_DEFAULT_IMAGE, &img) == 0) {
        wrap_kdev * w = nullptr;
        tsr_transport * t = open(&w);
        if (t && tsr_transport_boot(t, img) == 0) {
            opl::writer b;
            b.add(opl::K_NOP, nullptr, 0);
            const std::vector<uint8_t> blob = b.finish();
            const uint64_t off = 64ull << 20;
            CHECK(tsr_transport_write(t, 0, off, blob.data(), blob.size()) == 0, "write NOP graph");
            const tsr_launch graph = L(TSR_DESC_TYPE_LAUNCH, (uint16_t) tsr_image_kernel(img, "graph"),
                                       { (uint32_t) (off / TSR_LINE_BYTES), (uint32_t) blob.size() });
            uint32_t seq = 0;
            CHECK(tsr_transport_launch(t, &graph, &seq) == 0, "launch NOP graph");
            const uint32_t rd = ready(t, w, 1);
            w->reads.clear();
            w->fail_read = TSR_CSR_LCQ_REC(rd % TSR_LCQ_DEPTH, 2);
            tsr_record r{};
            CHECK(tsr_transport_poll(t, seq, &r) == 1 && st_of(r) == 0 && r.w[2] == 0u && w->read_failures == 0,
                  "successful graph synthesizes zero RET without reading r2");
            CHECK(word_reads(w, 0) == 1, "graph reads r0 once");
            for (uint32_t j = 1; j < 8; ++j) CHECK(word_reads(w, j) == 0, "graph skips word %u", j);
        } else CHECK(0, "graph minimal read open + boot");
        tsr_transport_close(t);
        tsr_image_close(img);
    }
}

// Adapter errors during a launch: plat_err_i -> DONE PLAT_ERR -> ADAPTER_STATUS read, reported,
// cleared (EX §8.6); a hung unit: "card reset needed".
static void test_launch_faults() {
    printf("protocol: adapter errors and a hung unit during launches\n");
    tsr_image * img = nullptr;
    if (tsr_image_open(TSR_FW_TEST_IMAGE_fw_rt, &img) != 0) return;
    const uint16_t knop = (uint16_t) tsr_image_kernel(img, "rt_nop");
    {
        mock_kdev * m = nullptr;
        tsr_transport * t = open_mock(small_mock(), &m);
        if (t && tsr_transport_boot(t, img) == 0) {
            m->inject_at_push(2, AST_BRESP | (1u << 16));   // an AXI write error during the 2nd launch
            tsr_record r;
            const tsr_launch l = L(TSR_DESC_TYPE_LAUNCH, knop);
            CHECK(tsr_transport_call(t, &l, TSR_TIMEOUT_DEFAULT, &r) == 0 && st_of(r) == 0, "launch 1 OK");
            CHECK(tsr_transport_call(t, &l, TSR_TIMEOUT_DEFAULT, &r) == 0 && st_of(r) == TSR_LCQ_R0_STATUS_PLAT_ERR,
                  "launch 2: DONE PLAT_ERR (r0 0x%08x)", r.w[0]);
            const adapter_log lg = transport_adapter_log(t);
            CHECK(lg.plat_errs == 1 && lg.last == (AST_BRESP | (1u << 16)), "ADAPTER_STATUS read (0x%x)", lg.last);
            uint32_t v = 1;
            m->reg_read(KREG_ADAPTER_STATUS, &v);
            CHECK(v == 0, "ADAPTER_STATUS cleared (0x%x)", v);
            CHECK(tsr_transport_call(t, &l, TSR_TIMEOUT_DEFAULT, &r) == 0 && st_of(r) == 0, "launch 3 OK again");
        } else {
            CHECK(0, "open + boot");
        }
        tsr_transport_close(t);
    }
    {
        // STATUS.PLAT_ERR seen while waiting, before the answer: a kernel without a previous
        // duration backs off at once and reads STATUS, and rt_stop runs longer than the sim kdev's
        // 2^20-cycle run-ahead. ADAPTER_STATUS (plat_err_i) must stay set until every launch in
        // flight has answered, so both answer DONE PLAT_ERR (R7) instead of KERNEL_ERR / OK; one
        // event; cleared after the last answer.
        mock_kdev * m = nullptr;
        tsr_transport * t = open_mock(small_mock(), &m);
        if (t && tsr_transport_boot(t, img) == 0) {
            m->inject_at_push(1, AST_BRESP | (1u << 16));
            const tsr_launch ls = L(TSR_DESC_TYPE_LAUNCH, (uint16_t) tsr_image_kernel(img, "rt_stop"), { 1u << 20, 1 });
            const tsr_launch ln = L(TSR_DESC_TYPE_LAUNCH, knop);
            uint32_t s1 = 0, s2 = 0;
            CHECK(tsr_transport_launch(t, &ls, &s1) == 0 && tsr_transport_launch(t, &ln, &s2) == 0, "PLAT_ERR while waiting: 2 launches");
            tsr_record r1{}, r2{};
            CHECK(tsr_transport_wait(t, s1, TSR_TIMEOUT_DEFAULT, &r1) == 0 && st_of(r1) == TSR_LCQ_R0_STATUS_PLAT_ERR,
                  "PLAT_ERR while waiting: the long launch answers DONE PLAT_ERR (r0 0x%08x)", r1.w[0]);
            CHECK(tsr_transport_wait(t, s2, TSR_TIMEOUT_DEFAULT, &r2) == 0 && st_of(r2) == TSR_LCQ_R0_STATUS_PLAT_ERR,
                  "PLAT_ERR while waiting: the launch behind it answers DONE PLAT_ERR too (r0 0x%08x)", r2.w[0]);
            const adapter_log lg = transport_adapter_log(t);
            CHECK(lg.plat_errs == 1 && lg.last == (AST_BRESP | (1u << 16)), "PLAT_ERR while waiting: one event (%u, 0x%x)", lg.plat_errs,
                  lg.last);
            uint32_t v = 1;
            m->reg_read(KREG_ADAPTER_STATUS, &v);
            CHECK(v == 0, "PLAT_ERR while waiting: ADAPTER_STATUS cleared after the last answer (0x%x)", v);
            tsr_record r;
            CHECK(tsr_transport_call(t, &ln, TSR_TIMEOUT_DEFAULT, &r) == 0 && st_of(r) == 0, "PLAT_ERR while waiting: OK again");
        } else {
            CHECK(0, "open + boot");
        }
        tsr_transport_close(t);
    }
    for (uint32_t depth : { 1u, 4u, 7u }) {
        // Pipelined launches (depth kept outstanding, so one is always in flight): one transient
        // AXI error at the 5th push. The episode covers the launches pushed before it was seen;
        // ADAPTER_STATUS is cleared once they answered, and every later launch answers OK.
        mock_kdev * m = nullptr;
        tsr_transport * t = open_mock(small_mock(), &m);
        if (t && tsr_transport_boot(t, img) == 0) {
            m->inject_at_push(5, AST_BRESP | (1u << 16));
            const tsr_launch ln = L(TSR_DESC_TYPE_LAUNCH, knop);
            std::vector<uint32_t> seqs;
            std::vector<uint32_t> status;
            size_t done = 0;
            const int N = 60;
            for (int i = 0; i < N; ++i) {
                uint32_t q = 0;
                CHECK(tsr_transport_launch(t, &ln, &q) == 0, "pipelined PLAT_ERR: launch %d", i);
                seqs.push_back(q);
                while (seqs.size() - done > depth) {
                    tsr_record r{};
                    CHECK(tsr_transport_wait(t, seqs[done], TSR_TIMEOUT_DEFAULT, &r) == 0, "pipelined PLAT_ERR: wait");
                    status.push_back(st_of(r));
                    done++;
                }
            }
            for (; done < seqs.size(); ++done) {
                tsr_record r{};
                CHECK(tsr_transport_wait(t, seqs[done], TSR_TIMEOUT_DEFAULT, &r) == 0, "pipelined PLAT_ERR: wait");
                status.push_back(st_of(r));
            }
            uint32_t n_plat = 0, n_ok = 0;
            size_t last_plat = 0;
            for (size_t i = 0; i < status.size(); ++i) {
                if (status[i] == TSR_LCQ_R0_STATUS_PLAT_ERR) {
                    n_plat++;
                    last_plat = i;
                } else if (status[i] == 0) {
                    n_ok++;
                }
            }
            CHECK(n_plat >= 1 && n_plat <= 2 * depth + 2 && n_ok + n_plat == (uint32_t) N && last_plat < 5 + 2 * depth + 2,
                  "pipelined PLAT_ERR, depth %u: %u PLAT_ERR (the last at launch %zu), %u OK of %d", depth, n_plat, last_plat, n_ok, N);
            const adapter_log lg = transport_adapter_log(t);
            CHECK(lg.plat_errs == 1, "pipelined PLAT_ERR, depth %u: one event (%u)", depth, lg.plat_errs);
            uint32_t v = 1;
            m->reg_read(KREG_ADAPTER_STATUS, &v);
            CHECK(v == 0, "pipelined PLAT_ERR, depth %u: ADAPTER_STATUS cleared (0x%x)", depth, v);
            printf("pipelined PLAT_ERR, depth %u: %u PLAT_ERR answers (launches up to %zu), then %u OK\n", depth, n_plat, last_plat, n_ok);
        } else {
            CHECK(0, "open + boot");
        }
        tsr_transport_close(t);
    }
    {
        mock_options o = small_mock();
        o.hang_at_push = 2;
        mock_kdev * m = nullptr;
        tsr_transport * t = open_mock(o, &m);
        if (t && tsr_transport_boot(t, img) == 0) {
            transport_set_timeouts(t, 2, 2, 0, 2);
            tsr_record r;
            const tsr_launch l = L(TSR_DESC_TYPE_LAUNCH, knop);
            CHECK(tsr_transport_call(t, &l, TSR_TIMEOUT_DEFAULT, &r) == 0, "launch 1 OK");
            CHECK(tsr_transport_call(t, &l, TSR_TIMEOUT_DEFAULT, &r) == TSR_TE_DEVICE, "hung unit: TSR_TE_DEVICE");
            const recovery_log rl = transport_recovery_log(t);
            CHECK(rl.failed && rl.timeouts == 1 && rl.level2 == 0, "FAILED at the ladder's UNIT_HUNG check");
            CHECK(tsr_transport_call(t, &l, TSR_TIMEOUT_DEFAULT, &r) == TSR_TE_DEVICE, "further launches refused");
            CHECK(tsr_transport_soft_reset(t) == TSR_TE_DEVICE && tsr_transport_boot(t, img) == TSR_TE_DEVICE, "no reset, no boot");
        } else {
            CHECK(0, "open + boot (hang)");
        }
        tsr_transport_close(t);
    }
    tsr_image_close(img);
}

static void test_dispatch() {
    printf("dispatch and environment\n");
    setenv("TSR_MOCK_NUM_CH", "3", 1);
    setenv("TSR_MOCK_TMAX", "8", 1);
    tsr_transport * t = tsr_transport_open("mock");
    CHECK(t != nullptr && tsr_transport_config(t)->num_ch == 3 && tsr_transport_config(t)->t_max == 8, "TSR_DEVICE=mock with env");
    tsr_transport_close(t);
    setenv("TSR_MOCK_DMA_ALIGN", "48", 1);
    CHECK(tsr_transport_open("mock") == nullptr, "bad TSR_MOCK_DMA_ALIGN refused");
    unsetenv("TSR_MOCK_DMA_ALIGN");
    unsetenv("TSR_MOCK_NUM_CH");
    unsetenv("TSR_MOCK_TMAX");
    setenv("TSR_MOCK_INJECT", "0:0x00010002,0:4,3:1", 1);
    t = tsr_transport_open("mock");
    CHECK(t != nullptr, "TSR_MOCK_INJECT");
    if (t) {
        CHECK(transport_adapter_log(t).at_open == 0x00010006u, "TSR_MOCK_INJECT 0:V = left before the open (0x%x)",
              transport_adapter_log(t).at_open);
        tsr_transport_close(t);
    }
    for (const char * bad : { "1", "1:", "x:5", "1:0", "1:0x100000000", "1:5,", "-1:5" }) {
        setenv("TSR_MOCK_INJECT", bad, 1);
        CHECK(tsr_transport_open("mock") == nullptr, "TSR_MOCK_INJECT=%s refused", bad);
    }
    unsetenv("TSR_MOCK_INJECT");
    for (const char * bad : { "0x101:1", "x", "4:0" }) {
        setenv("TSR_MOCK_TCM_FAULT", bad, 1);
        CHECK(tsr_transport_open("mock") == nullptr, "TSR_MOCK_TCM_FAULT=%s refused", bad);
    }
    unsetenv("TSR_MOCK_TCM_FAULT");
    setenv("TSR_MOCK_HANG", "0", 1);
    CHECK(tsr_transport_open("mock") == nullptr, "TSR_MOCK_HANG=0 refused");
    unsetenv("TSR_MOCK_HANG");

    CHECK(!register_kdev("xrt", [] { return std::unique_ptr<kdev>(); }), "built-in names are reserved");
    CHECK(register_kdev("testkind", [] {
              mock_options o;
              o.num_ch = 1;
              o.threads = 1;
              return std::unique_ptr<kdev>(new mock_kdev(o));
          }),
          "register");
    t = tsr_transport_open("testkind");
    CHECK(t != nullptr && tsr_transport_config(t)->num_ch == 1, "registered kind");
    tsr_transport_close(t);
    CHECK(tsr_transport_open("nosuchkind") == nullptr, "unknown kind refused");
    setenv("TSR_DEVICE", "nosuchkind", 1);
    CHECK(tsr_transport_open(nullptr) == nullptr, "TSR_DEVICE=nosuchkind refused");
    unsetenv("TSR_DEVICE");
    unsetenv("TSR_XCLBIN");
    CHECK(tsr_transport_open("xrt") == nullptr, "xrt: not built in, or TSR_XCLBIN unset");
    CHECK(tsr_transport_open("vkernel") == nullptr, "vkernel: no server until P5.3");
    // Record helpers.
    tsr_record rec{};
    rec.w[0] = 0x05080101u;   // DONE, TILE_ERR, code 8, seq 5
    rec.w[1] = 0x105;
    rec.w[2] = (uint32_t) -3;
    rec.w[7] = 99;
    tsr_result res;
    tsr_record_decode(&rec, &res);
    CHECK(res.type == 1 && res.status == 1 && res.err_code == 8 && res.seq == 0x105 && res.ret == -3 && res.duration == 99, "tsr_record_decode");
    CHECK(!strcmp(tsr_status_name(0), "OK") && !strcmp(tsr_status_name(1), "TILE_ERR") && !strcmp(tsr_status_name(7), "STOPPED") &&
          !strcmp(tsr_status_name(9), "RESERVED"), "tsr_status_name");
    CHECK(strcmp(tsr_transport_strerror(TSR_TE_LOST), "unknown error") != 0, "tsr_transport_strerror");
}

// The xrt kdev's hw_emu defaults (docs/xrt-host.md §9): only under XCL_EMULATION_MODE=hw_emu,
// only for variables left unset or empty; a value the user set wins.
static void test_xrt_env_defaults() {
    using envmap = std::map<std::string, std::string>;
    auto run = [](const envmap & m) {
        std::string s;
        for (const env_default & d : xrt_env_defaults([&m](const char * n) -> const char * {
                 auto it = m.find(n);
                 return it == m.end() ? nullptr : it->second.c_str();
             })) {
            s += (s.empty() ? "" : " ") + std::string(d.name) + "=" + d.value;
        }
        return s;
    };
    const std::string all = "TSR_XRT_CH_GIB=3 TSR_XRT_DMA=staged TSR_XRT_DMA_ALIGN=4096";
    CHECK(run({}).empty(), "no emulation mode: no defaults");
    CHECK(run({ { "XCL_EMULATION_MODE", "noop" } }).empty(), "noop: no defaults");
    CHECK(run({ { "XCL_EMULATION_MODE", "sw_emu" } }).empty(), "sw_emu: no defaults");
    CHECK(run({ { "XCL_EMULATION_MODE", "" } }).empty(), "empty mode: no defaults");
    CHECK(run({ { "XCL_EMULATION_MODE", "hw_emu" } }) == all, "hw_emu: all three (%s)", run({ { "XCL_EMULATION_MODE", "hw_emu" } }).c_str());
    CHECK(run({ { "XCL_EMULATION_MODE", "hw_emu" }, { "TSR_XRT_DMA", "direct" } }) == "TSR_XRT_CH_GIB=3 TSR_XRT_DMA_ALIGN=4096",
          "hw_emu: TSR_XRT_DMA set by the user wins");
    CHECK(run({ { "XCL_EMULATION_MODE", "hw_emu" }, { "TSR_XRT_CH_GIB", "16" } }) == "TSR_XRT_DMA=staged TSR_XRT_DMA_ALIGN=4096",
          "hw_emu: TSR_XRT_CH_GIB set by the user wins");
    CHECK(run({ { "XCL_EMULATION_MODE", "hw_emu" }, { "TSR_XRT_DMA_ALIGN", "64" } }) == "TSR_XRT_CH_GIB=3 TSR_XRT_DMA=staged",
          "hw_emu: TSR_XRT_DMA_ALIGN set by the user wins");
    CHECK(run({ { "XCL_EMULATION_MODE", "hw_emu" }, { "TSR_XRT_CH_GIB", "1" }, { "TSR_XRT_DMA", "m2m" }, { "TSR_XRT_DMA_ALIGN", "1" } })
              .empty(),
          "hw_emu: everything set, no defaults");
    CHECK(run({ { "XCL_EMULATION_MODE", "hw_emu" }, { "TSR_XRT_CH_GIB", "" } }) == all, "hw_emu: an empty value counts as unset");
    // The default lookup is getenv.
    const char * saved = getenv("XCL_EMULATION_MODE");
    const std::string saved_v = saved ? saved : "";
    setenv("XCL_EMULATION_MODE", "hw_emu", 1);
    setenv("TSR_XRT_DMA", "direct", 1);
    unsetenv("TSR_XRT_CH_GIB");
    unsetenv("TSR_XRT_DMA_ALIGN");
    const std::vector<env_default> g = xrt_env_defaults();
    CHECK(g.size() == 2 && strcmp(g[0].name, "TSR_XRT_CH_GIB") == 0 && strcmp(g[1].name, "TSR_XRT_DMA_ALIGN") == 0,
          "getenv lookup (%zu defaults)", g.size());
    unsetenv("TSR_XRT_DMA");
    if (saved) setenv("XCL_EMULATION_MODE", saved_v.c_str(), 1);
    else unsetenv("XCL_EMULATION_MODE");
}

int main() {
    clear_env();
    test_xrt_env_defaults();
    test_dma_engine();
    test_mock_regs();
    test_protocol_basic();
    test_no_db_block();
    test_open_refusals();
    test_adapter_log();
    if (access(TSR_FW_TEST_IMAGE_fw_rt, R_OK) == 0) {
        test_stale_core();
        test_stale_reopen();
        test_watermark_refresh();
        test_boot_timeout();
        test_lcq_io_failures();
        test_launch_faults();
    } else {
        printf("firmware image %s not built: stale-core, boot-timeout and launch-fault tests not run\n", TSR_FW_TEST_IMAGE_fw_rt);
    }
    test_dispatch();
    printf("test-tsr-kdev: %d checks, %d failed\n", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
