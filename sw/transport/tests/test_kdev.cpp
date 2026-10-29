// test_kdev.cpp — the protocol-layer transport (tsr_kdev.cpp) on the mock kdev, and the
// chunked / aligned transfer engine against a strict fake DMA (docs/xrt-host.md §6).
//
// Covers: CSR window offsets, configuration decode, identity checks at open (ID, ISA major 2,
// NUM_CH vs the bound buffers, 4 KiB-aligned buffers), ADAPTER_STATUS left at open and the
// adapter log, the protocol layer never touching the adapter's unmapped registers (v1's AP_CTRL ..
// arg7), the mock's register model (channel bases, ADAPTER_STATUS W1C of the count), the kdev
// registry and TSR_MOCK_* parsing, and byte-granular transfers at odd offsets / sizes / host
// addresses, across and beyond 4 GiB, under DMA rules from align 1 to 4096 with small chunks
// (shadow-memory compare, guard bytes untouched, every raw transfer checked against the rules).
// The xrt kdev's hw_emu environment defaults (xrt_env_defaults). v1's run protocol tests
// (argument packing, start / done, timeout -> ABORT, stale completions) are deleted with it (X0);
// the v2 protocol tests (transport_protocol, SW §10.2) come with P3.1.

#include "tsr_kdev.h"
#include "tsr_kdev_mock.h"
#include "tsr_transport_impl.h"

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
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
static tsr_transport * open_mock(mock_options o, mock_kdev ** out) {
    std::unique_ptr<mock_kdev> m(new mock_kdev(o));
    CHECK(m->ok(), "mock created");
    *out = m.get();
    return transport_open_kdev(std::move(m));
}

static void test_protocol_basic() {
    printf("protocol\n");
    mock_options o;
    mock_kdev * m = nullptr;
    tsr_transport * t = open_mock(o, &m);
    CHECK(t != nullptr, "open");
    if (!t) return;
    const tsr_config * c = tsr_transport_config(t);
    CHECK(strcmp(tsr_transport_name(t), "mock") == 0, "name %s", tsr_transport_name(t));
    CHECK(c->num_ch == 4 && c->t_max == 1 && c->lanes == 128 && c->k_max == 32768 && c->spm_bytes == 4u << 20 && c->vl == 16 &&
              c->nsreg == 16 && c->caps == 0 && c->beat_bits == 512 && c->ch_bytes == (16ull << 30),
          "config decode");
    CHECK(transport_kdev(t) == m, "transport_kdev");
    CHECK(tsr_transport_csr_read(t, TSR_CSR_ID) == TSR_ID_MAGIC, "window ID");
    CHECK(tsr_transport_csr_read(t, TSR_CSR_VERSION) == 0x0200u, "window VERSION 0x%x", tsr_transport_csr_read(t, TSR_CSR_VERSION));
    const uint64_t wr0 = m->window_reads;
    (void) tsr_transport_csr_read(t, TSR_CSR_CFG0);
    CHECK(m->window_reads == wr0 + 1, "csr_read is one window read");
    CHECK(tsr_transport_csr_read(t, 0x1000) == 0xFFFFFFFFu, "csr_read outside the window refused");
    CHECK(m->ch_addr(1) > (4ull << 30), "device addresses above 4 GiB");
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
    CHECK(tsr_transport_write(t, 0, c->ch_bytes - 10, big.data(), 11) != 0, "write past the channel refused");
    CHECK(tsr_transport_read(t, C, 0, back.data(), 1) != 0, "bad channel refused");

    CHECK(m->unmapped_accesses == 0, "protocol layer touched unmapped adapter registers %" PRIu64 " times", m->unmapped_accesses);
    CHECK(m->window_writes == 0, "protocol layer wrote %" PRIu64 " core CSRs", m->window_writes);
    tsr_transport_close(t);
}

// Opens that must fail: not a Tesseract core, a v1 core (ISA major 1), NUM_CH vs the bound
// buffers, a channel buffer that is not 4 KiB aligned.
static void test_open_refusals() {
    mock_options o;
    mock_kdev * m = nullptr;
    {
        std::unique_ptr<mock_kdev> mk(new mock_kdev(o));
        mk->id_override = 0x12345678u;
        CHECK(transport_open_kdev(std::move(mk)) == nullptr, "wrong ID refused");
    }
    {
        std::unique_ptr<mock_kdev> mk(new mock_kdev(o));
        mk->set_ch_addr(2, mk->ch_addr(2) + 64);
        CHECK(transport_open_kdev(std::move(mk)) == nullptr, "unaligned channel buffer refused");
    }
    {
        mock_options o1 = o;
        o1.adapter_version = 0x00010003u;   // reports 3 channels; the core still has 4: not checked
        tsr_transport * t = open_mock(o1, &m);
        CHECK(t != nullptr, "ADAPTER_VERSION is informational");
        tsr_transport_close(t);
    }
    // A v1 bitstream reports ISA 1.0: refused (its tools are in build/v1).
    struct v1_core final : kdev {
        std::unique_ptr<mock_kdev> in;
        explicit v1_core(const mock_options & mo) : in(new mock_kdev(mo)) {}
        const char * name() const override { return "v1core"; }
        uint32_t num_ch() const override { return in->num_ch(); }
        uint64_t ch_bytes(uint32_t ch) const override { return in->ch_bytes(ch); }
        uint64_t ch_addr(uint32_t ch) const override { return in->ch_addr(ch); }
        int reg_read(uint32_t off, uint32_t * val) override {
            if (off == KREG_CSR_WINDOW + TSR_CSR_VERSION) {
                *val = 0x0100u;
                return 0;
            }
            return in->reg_read(off, val);
        }
        int reg_write(uint32_t off, uint32_t val) override { return in->reg_write(off, val); }
        int mem_write(uint32_t ch, uint64_t off, const void * src, size_t n) override { return in->mem_write(ch, off, src, n); }
        int mem_read(uint32_t ch, uint64_t off, void * dst, size_t n) override { return in->mem_read(ch, off, dst, n); }
    };
    CHECK(transport_open_kdev(std::unique_ptr<kdev>(new v1_core(o))) == nullptr, "ISA 1.0 core refused");
    // NUM_CH of the core differs from the bound buffers.
    struct fewer final : kdev {
        std::unique_ptr<mock_kdev> in;
        explicit fewer(const mock_options & mo) : in(new mock_kdev(mo)) {}
        const char * name() const override { return "fewer"; }
        uint32_t num_ch() const override { return in->num_ch() - 1; }
        uint64_t ch_bytes(uint32_t ch) const override { return in->ch_bytes(ch); }
        uint64_t ch_addr(uint32_t ch) const override { return in->ch_addr(ch); }
        int reg_read(uint32_t off, uint32_t * val) override { return in->reg_read(off, val); }
        int reg_write(uint32_t off, uint32_t val) override { return in->reg_write(off, val); }
        int mem_write(uint32_t ch, uint64_t off, const void * src, size_t n) override { return in->mem_write(ch, off, src, n); }
        int mem_read(uint32_t ch, uint64_t off, void * dst, size_t n) override { return in->mem_read(ch, off, dst, n); }
    };
    CHECK(transport_open_kdev(std::unique_ptr<kdev>(new fewer(o))) == nullptr, "NUM_CH vs bound buffers refused");
    CHECK(transport_open_kdev(nullptr) == nullptr, "no kdev");
}

// The mock's adapter registers against tsr_xrt_ctrl.
static void test_mock_regs() {
    mock_options o;
    o.num_ch = 2;
    mock_kdev m(o);
    uint32_t v = 0;
    // channel bases: plain storage, lo / hi per channel
    m.reg_write(KREG_CH_BASE0 + 8, 0x12345000u);
    m.reg_write(KREG_CH_BASE0 + 12, 0x40u);
    CHECK(m.reg_read(KREG_CH_BASE0 + 8, &v) == 0 && v == 0x12345000u, "ch1_base lo (0x%x)", v);
    CHECK(m.reg_read(KREG_CH_BASE0 + 12, &v) == 0 && v == 0x40u, "ch1_base hi (0x%x)", v);
    CHECK(m.reg_read(KREG_CH_BASE0, &v) == 0 && v == 0, "ch0_base untouched");
    CHECK(m.unmapped_accesses == 0, "bases are mapped");
    // v1's run registers are gone: read 0, writes ignored
    for (uint32_t off = 0; off < KREG_CH_BASE0; off += 4) {
        m.reg_write(off, 0xFFFFFFFFu);
        CHECK(m.reg_read(off, &v) == 0 && v == 0, "offset 0x%03x reads 0 (0x%x)", off, v);
    }
    CHECK(m.unmapped_accesses == 2 * (KREG_CH_BASE0 / 4), "unmapped accesses counted (%" PRIu64 ")", m.unmapped_accesses);
    CHECK(m.reg_read(KREG_ADAPTER_VERSION, &v) == 0 && v == 0x00010002u, "ADAPTER_VERSION 0x%x", v);
    CHECK(m.reg_read(KREG_SPACE, &v) != 0 && m.reg_read(2, &v) != 0, "outside / unaligned refused");
    // ADAPTER_STATUS W1C: bits [3:0] one by one, any 1 in [31:16] clears the whole count
    m.inject_status(AST_RRESP | AST_ORPHAN | (3u << 16));
    CHECK(m.reg_read(KREG_ADAPTER_STATUS, &v) == 0 && v == (AST_RRESP | AST_ORPHAN | (3u << 16)), "injected 0x%x", v);
    m.reg_write(KREG_ADAPTER_STATUS, 0x00010000u);
    CHECK(m.reg_read(KREG_ADAPTER_STATUS, &v) == 0 && v == (AST_RRESP | AST_ORPHAN), "count cleared as a whole (0x%x)", v);
    m.reg_write(KREG_ADAPTER_STATUS, AST_ORPHAN);
    CHECK(m.reg_read(KREG_ADAPTER_STATUS, &v) == 0 && v == AST_RRESP, "bit2 cleared alone (0x%x)", v);
    m.reg_write(KREG_ADAPTER_STATUS, 0xFFFFFFFFu);
    CHECK(m.reg_read(KREG_ADAPTER_STATUS, &v) == 0 && v == 0, "all cleared (0x%x)", v);
    m.inject_status(0xFFFF0000u);
    m.inject_status(5u << 16);
    CHECK(m.reg_read(KREG_ADAPTER_STATUS, &v) == 0 && v == 0xFFFF0000u, "count saturates (0x%x)", v);
    // the CSR window reaches the simulator; writes are forwarded (and ignored by it)
    CHECK(m.reg_read(KREG_CSR_WINDOW + TSR_CSR_CFG0, &v) == 0 && v == (2u | (128u << 8) | (1u << 20)), "window CFG0 0x%x", v);
    m.reg_write(KREG_CSR_WINDOW + 0x20, 1);
    CHECK(m.window_writes == 1 && m.window_write_log.size() == 1 && m.window_write_log[0] == 0x20, "window write logged");
}

// The adapter log: what the protocol layer saw in ADAPTER_STATUS at open (it clears the register).
static void test_adapter_log() {
    mock_options o;
    o.num_ch = 1;
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
    CHECK(ts != nullptr && transport_adapter_log(ts).at_open == 0 && transport_kdev(ts) == nullptr, "sim: empty log, no kdev");
    tsr_transport_close(ts);
}

static void test_dispatch() {
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
    setenv("TSR_MOCK_INJECT", "0:0x00010002,0:4", 1);
    t = tsr_transport_open("mock");
    CHECK(t != nullptr, "TSR_MOCK_INJECT");
    if (t) {
        CHECK(transport_adapter_log(t).at_open == 0x00010006u, "TSR_MOCK_INJECT 0:V = left before the open (0x%x)",
              transport_adapter_log(t).at_open);
        tsr_transport_close(t);
    }
    for (const char * bad : { "1", "1:", "x:5", "1:0", "1:0x100000000", "1:5,", "-1:5", "1:5", "0:4,2:4" }) {
        setenv("TSR_MOCK_INJECT", bad, 1);
        CHECK(tsr_transport_open("mock") == nullptr, "TSR_MOCK_INJECT=%s refused", bad);
    }
    unsetenv("TSR_MOCK_INJECT");

    CHECK(!register_kdev("xrt", [] { return std::unique_ptr<kdev>(); }), "built-in names are reserved");
    CHECK(register_kdev("testkind", [] {
              mock_options o;
              o.num_ch = 1;
              return std::unique_ptr<kdev>(new mock_kdev(o));
          }),
          "register");
    t = tsr_transport_open("testkind");
    CHECK(t != nullptr && tsr_transport_config(t)->num_ch == 1, "registered kind");
    tsr_transport_close(t);
    CHECK(tsr_transport_open("nosuchkind") == nullptr, "unknown kind refused");
    unsetenv("TSR_XCLBIN");
    CHECK(tsr_transport_open("xrt") == nullptr, "xrt: not built in, or TSR_XCLBIN unset");
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
    test_xrt_env_defaults();
    test_dma_engine();
    test_mock_regs();
    test_protocol_basic();
    test_open_refusals();
    test_adapter_log();
    test_dispatch();
    printf("test-tsr-kdev: %d checks, %d failed\n", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
