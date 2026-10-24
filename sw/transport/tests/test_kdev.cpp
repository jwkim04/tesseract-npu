// test_kdev.cpp — the kernel-protocol transport (tsr_kdev.cpp) on the mock kdev, and the
// chunked / aligned transfer engine against a strict fake DMA (docs/xrt-host.md §6).
//
// Covers: argument packing (cmd_addr lo / hi, cmd_count, arg0..7 through R8..R15, every
// ch<c>_base), CSR window offsets, configuration decode, start / done incl. delayed ap_done,
// timeout -> CTRL.ABORT (core stops: ABORTED; core stuck: transport refuses further runs),
// ADAPTER_STATUS errors after a run and at open and the adapter log, a BUSY core at open, the
// protocol layer never touching AP_CTRL / IER / ISR / argument registers outside start() /
// wait() / settle(), the kdev registry, and byte-granular transfers at odd offsets / sizes /
// host addresses, across and beyond 4 GiB, under DMA rules from align 1 to 4096 with small
// chunks (shadow-memory compare, guard bytes untouched, every raw transfer checked against the
// rules). The protocol tests run twice: with regs_start / regs_wait and with the XRT scheduler's
// sequence (kds_start / kds_wait, the xrt::run path). Stale completions (an ap_done nobody
// consumed, which accepting a start does not clear: tsr_xrt_ctrl) left by an earlier process or
// an untracked run: consumed at open, detected after a completion and waited out. The mock's
// register behaviour against tsr_xrt_ctrl (ap_done kept over an accept, 1-bit IER / ISR,
// ADAPTER_STATUS W1C of the count). The xrt kdev's hw_emu environment defaults (xrt_env_defaults).

#include "tsr_kdev.h"
#include "tsr_kdev_mock.h"
#include "tsr_transport_impl.h"

#include <chrono>
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
static bool g_kds = false;   // protocol tests: start / wait like the XRT scheduler

static tsr_transport * open_mock(mock_options o, mock_kdev ** out) {
    o.ctrl_kds = g_kds;
    std::unique_ptr<mock_kdev> m(new mock_kdev(o));
    CHECK(m->ok(), "mock created");
    *out = m.get();
    return transport_open_kdev(std::move(m));
}

static void put_insns(tsr_transport * t, uint64_t beat, const std::vector<tsr_insn> & p) {
    CHECK(tsr_transport_write(t, 0, beat * 64, p.data(), p.size() * sizeof(tsr_insn)) == 0, "program write");
}

static void test_protocol_basic() {
    printf("protocol (%s)\n", g_kds ? "kds_start / kds_wait" : "regs_start / regs_wait");
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
    const uint64_t wr0 = m->window_reads;
    (void) tsr_transport_csr_read(t, TSR_CSR_CFG0);
    CHECK(m->window_reads == wr0 + 1, "csr_read is one window read");
    CHECK(tsr_transport_csr_read(t, 0x1000) == 0xFFFFFFFFu, "csr_read outside the window refused");

    // Argument packing: data lines stored to addresses taken from ARG0..7 (R8..R15), one through
    // LI / ADDI, command buffer at a non-zero beat.
    const uint32_t C = c->num_ch;
    const uint64_t data_beat = (64ull << 20) / 64 + 100, cmd_beat = 1000;
    uint8_t data[9][64];
    rng r{ 99 };
    r.fill(data, sizeof(data));
    CHECK(tsr_transport_write(t, 0, data_beat * 64, data, sizeof(data)) == 0, "data");
    uint32_t args[8];
    std::vector<tsr_insn> p;
    p.push_back(tsr_enc_ld(0, 0, (uint32_t) data_beat, 9));
    for (uint32_t i = 0; i < 8; ++i) {
        args[i] = (1u << 26) + 0x10000u * i + 3 * i;   // beat addresses past 4 GiB of byte offset
        tsr_insn st = tsr_enc_st(i % C, i, 0, 1);
        tsr_insn_set_base(&st, 0, 8 + i, 0);           // A2 = R[8+i] + 0
        p.push_back(st);
    }
    p.push_back(tsr_enc_li(1, 0x00123450u));
    p.push_back(tsr_enc_addi(2, 1, 0x7u));
    tsr_insn st2 = tsr_enc_st(C - 1, 8, 0, 1);
    tsr_insn_set_base(&st2, 0, 2, 0);
    p.push_back(st2);
    p.push_back(tsr_enc_end());
    put_insns(t, cmd_beat, p);
    tsr_exec_result res;
    CHECK(tsr_transport_exec(t, cmd_beat, (uint32_t) p.size(), args, &res) == 0, "exec rc");
    CHECK(res.status == TSR_STATUS_DONE && res.err_code == 0, "status 0x%x err %u", res.status, res.err_code);
    for (uint32_t i = 0; i < 8; ++i) {
        uint8_t got[64];
        CHECK(tsr_transport_read(t, i % C, (uint64_t) args[i] * 64, got, 64) == 0, "read");
        CHECK(memcmp(got, data[i], 64) == 0, "ARG%u store", i);
        CHECK(m->last_args[i] == args[i], "ARG%u register 0x%x", i, m->last_args[i]);
    }
    uint8_t got[64];
    CHECK(tsr_transport_read(t, C - 1, 0x123457ull * 64, got, 64) == 0 && memcmp(got, data[8], 64) == 0, "LI/ADDI store");
    CHECK(m->last_cmd_lo == cmd_beat && m->last_cmd_hi == 0 && m->last_count == p.size(), "cmd registers");
    for (uint32_t ch = 0; ch < C; ++ch) CHECK(m->last_bases[ch] == m->ch_addr(ch), "ch%u_base", ch);
    CHECK(m->ch_addr(1) > (4ull << 30), "device addresses above 4 GiB");
    CHECK(transport_perf(t, TSR_PERF_RETIRED) == tsr_sim_perf(m->sim(), TSR_PERF_RETIRED), "PERF_RETIRED through the window");
    CHECK(transport_perf(t, tsr_perf_ch_rd_beats(0)) == tsr_sim_perf(m->sim(), tsr_perf_ch_rd_beats(0)) &&
              transport_perf(t, tsr_perf_ch_rd_beats(0)) > 0, "PERF_CH_RD_BEATS[0]");

    // CMD_ADDR_HI: beat (1 << 32) + 5 is beyond the channel: DRAM_RANGE at fetch, ERR_INFO = low word.
    CHECK(tsr_transport_exec(t, (1ull << 32) + 5, 4, nullptr, &res) != 0, "fetch fault rc");
    CHECK(res.status == (TSR_STATUS_DONE | TSR_STATUS_ERR) && res.err_code == TSR_ERR_DRAM_RANGE && res.err_pc == 0 && res.err_info == 5,
          "fetch fault: status 0x%x code %u pc %u info 0x%x", res.status, res.err_code, res.err_pc, res.err_info);
    CHECK(m->last_cmd_lo == 5 && m->last_cmd_hi == 1, "CMD_ADDR lo / hi split");

    // An error inside the program (illegal opcode at pc 1).
    std::vector<tsr_insn> pe = { tsr_enc_nop(), tsr_enc_nop(), tsr_enc_end() };
    pe[1].w[0] = 0x0000007Eu;
    put_insns(t, cmd_beat, pe);
    CHECK(tsr_transport_exec(t, cmd_beat, 3, nullptr, &res) != 0, "illegal op rc");
    CHECK(res.status == (TSR_STATUS_DONE | TSR_STATUS_ERR) && res.err_code == TSR_ERR_ILLEGAL_OP && res.err_pc == 1, "illegal op state");

    // Delayed ap_done: the wait loop polls until it shows.
    m->done_after_polls = 300;
    put_insns(t, cmd_beat, { tsr_enc_end() });
    CHECK(tsr_transport_exec(t, cmd_beat, 1, nullptr, &res) == 0 && res.status == TSR_STATUS_DONE, "delayed done");
    m->done_after_polls = 0;

    // Transfers through the transport: odd offsets around 4 GiB on the last channel.
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

    CHECK(m->ctl_access_outside == 0, "protocol layer touched AP_CTRL / IER / ISR / arguments %" PRIu64 " times", m->ctl_access_outside);
    tsr_transport_close(t);
}

static void test_protocol_faults() {
    setenv("TSR_XRT_TIMEOUT_MS", "40", 1);
    setenv("TSR_XRT_ABORT_MS", "100", 1);
    mock_options o;
    o.num_ch = 2;
    o.t_max = 8;
    o.ch_bytes = 2ull << 30;
    o.dma.align = 64;
    o.dma.max_xfer = 4096;
    mock_kdev * m = nullptr;
    tsr_transport * t = open_mock(o, &m);
    CHECK(t != nullptr, "open faults");
    if (!t) return;
    const tsr_config * c = tsr_transport_config(t);
    CHECK(c->num_ch == 2 && c->t_max == 8 && c->ch_bytes == (2ull << 30), "config 2 ch, T_MAX 8, 2 GiB");
    const uint64_t cmd_beat = 77;
    put_insns(t, cmd_beat, { tsr_enc_nop(), tsr_enc_nop(), tsr_enc_end() });
    tsr_exec_result res;

    // Timeout -> ABORT -> the core stops with ABORTED.
    m->hang = mock_hang::until_abort;
    const auto t0 = std::chrono::steady_clock::now();
    CHECK(tsr_transport_exec(t, cmd_beat, 3, nullptr, &res) != 0, "timeout rc");
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
    CHECK(res.status == (TSR_STATUS_DONE | TSR_STATUS_ERR) && res.err_code == TSR_ERR_ABORTED && res.err_pc == 0,
          "aborted: status 0x%x code %u pc %u", res.status, res.err_code, res.err_pc);
    CHECK(ms >= 40 && ms < 2000, "timeout after %lld ms", (long long) ms);
    bool saw_abort = false;
    for (uint32_t w : m->window_write_log) saw_abort |= w == TSR_CSR_CTRL;
    CHECK(saw_abort, "CTRL written through the window");
    m->hang = mock_hang::none;
    CHECK(tsr_transport_exec(t, cmd_beat, 3, nullptr, &res) == 0, "run after an abort");

    // ADAPTER_STATUS after a run: the run fails even though the core finished cleanly; cleared after.
    m->inject_status = AST_RRESP | AST_ORPHAN | (3u << 16);
    CHECK(tsr_transport_exec(t, cmd_beat, 3, nullptr, &res) != 0, "adapter error rc");
    CHECK(res.status == TSR_STATUS_DONE, "core itself fine");
    uint32_t ast = 1;
    CHECK(m->reg_read(KREG_ADAPTER_STATUS, &ast) == 0 && ast == 0, "ADAPTER_STATUS cleared (0x%x)", ast);
    CHECK(tsr_transport_exec(t, cmd_beat, 3, nullptr, &res) == 0, "next run ok");
    CHECK(adapter_status_str(AST_RRESP | (3u << 16)).find("RRESP") != std::string::npos, "decode");

    // A channel base that is not 4 KiB aligned at ap_start.
    const uint64_t a1 = m->ch_addr(1);
    m->set_ch_addr(1, a1 + 0x40);
    const uint64_t ret0 = tsr_sim_perf(m->sim(), TSR_PERF_RETIRED);
    CHECK(tsr_transport_exec(t, cmd_beat, 3, nullptr, &res) != 0, "misaligned base rc");
    CHECK(tsr_sim_perf(m->sim(), TSR_PERF_RETIRED) == ret0, "misaligned base: the core is not started");
    m->set_ch_addr(1, a1);
    CHECK(tsr_transport_exec(t, cmd_beat, 3, nullptr, &res) == 0, "aligned again");

    // The core never stops: the transport refuses further runs.
    m->hang = mock_hang::forever;
    CHECK(tsr_transport_exec(t, cmd_beat, 3, nullptr, &res) != 0, "stuck rc");
    CHECK(res.status & TSR_STATUS_BUSY, "still busy 0x%x", res.status);
    m->hang = mock_hang::none;
    const uint64_t starts = m->starts;
    CHECK(tsr_transport_exec(t, cmd_beat, 3, nullptr, &res) != 0 && m->starts == starts, "broken transport refuses runs");
    CHECK(m->ctl_access_outside == 0, "no control register access outside start / wait");
    tsr_transport_close(t);

    // At open: a stale ADAPTER_STATUS is cleared, a BUSY core is stopped with ABORT.
    o.ctrl_kds = g_kds;
    {
        std::unique_ptr<mock_kdev> mk(new mock_kdev(o));
        mk->inject_status = AST_BRESP | (1u << 16);
        mk->hang = mock_hang::until_abort;
        kargs a;
        a.cmd_count = 1;
        CHECK(mk->start(a) == 0, "raw start");
        CHECK((tsr_sim_csr_read(mk->sim(), TSR_CSR_STATUS) & TSR_STATUS_BUSY) != 0, "busy before open");
        mock_kdev * raw = mk.get();
        tsr_transport * t2 = transport_open_kdev(std::move(mk));
        CHECK(t2 != nullptr, "open with a busy core");
        if (t2) {
            uint32_t v = 1;
            CHECK(raw->reg_read(KREG_ADAPTER_STATUS, &v) == 0 && v == 0, "leftover ADAPTER_STATUS cleared at open (0x%x)", v);
            CHECK(transport_adapter_log(t2).at_open == (AST_BRESP | (1u << 16)), "adapter log: at_open 0x%x",
                  transport_adapter_log(t2).at_open);
            CHECK((tsr_sim_csr_read(raw->sim(), TSR_CSR_STATUS) & TSR_STATUS_BUSY) == 0, "stopped by open");
            tsr_exec_result r2;
            raw->hang = mock_hang::none;
            put_insns(t2, 5, { tsr_enc_end() });
            CHECK(tsr_transport_exec(t2, 5, 1, nullptr, &r2) == 0, "run after open-time recovery");
            tsr_transport_close(t2);
        }
    }
    {
        std::unique_ptr<mock_kdev> mk(new mock_kdev(o));
        mk->hang = mock_hang::forever;
        kargs a;
        a.cmd_count = 1;
        CHECK(mk->start(a) == 0, "raw start");
        CHECK(transport_open_kdev(std::move(mk)) == nullptr, "open refuses a core that does not stop");
    }
    {
        std::unique_ptr<mock_kdev> mk(new mock_kdev(o));
        mk->id_override = 0x12345678u;
        CHECK(transport_open_kdev(std::move(mk)) == nullptr, "open refuses a wrong ID");
    }
    unsetenv("TSR_XRT_TIMEOUT_MS");
    unsetenv("TSR_XRT_ABORT_MS");
}

// The mock's adapter registers against tsr_xrt_ctrl.
static void test_mock_regs() {
    mock_options o;
    o.num_ch = 1;
    mock_kdev m(o);
    uint32_t v = 0;
    m.reg_write(KREG_IER, 3);
    CHECK(m.reg_read(KREG_IER, &v) == 0 && v == 1, "IER keeps bit0 only (0x%x)", v);
    m.reg_write(KREG_ISR, 2);
    CHECK(m.reg_read(KREG_ISR, &v) == 0 && v == 0, "ISR: a write of 2 does not toggle (0x%x)", v);
    m.reg_write(KREG_ISR, 1);
    CHECK(m.reg_read(KREG_ISR, &v) == 0 && v == 1, "ISR: a write of 1 toggles bit0 (0x%x)", v);
    m.reg_write(KREG_ISR, 3);
    CHECK(m.reg_read(KREG_ISR, &v) == 0 && v == 0, "ISR: toggles back (0x%x)", v);
    // ADAPTER_STATUS W1C: bits [3:0] one by one, any 1 in [31:16] clears the whole count
    m.inject_status = AST_RRESP | AST_ORPHAN | (3u << 16);
    kargs a;
    CHECK(m.start(a) == 0 && m.wait(1000) == kwait::done, "run");
    CHECK(m.reg_read(KREG_ADAPTER_STATUS, &v) == 0 && v == (AST_RRESP | AST_ORPHAN | (3u << 16)), "injected 0x%x", v);
    m.reg_write(KREG_ADAPTER_STATUS, 0x00010000u);
    CHECK(m.reg_read(KREG_ADAPTER_STATUS, &v) == 0 && v == (AST_RRESP | AST_ORPHAN), "count cleared as a whole (0x%x)", v);
    m.reg_write(KREG_ADAPTER_STATUS, AST_ORPHAN);
    CHECK(m.reg_read(KREG_ADAPTER_STATUS, &v) == 0 && v == AST_RRESP, "bit2 cleared alone (0x%x)", v);
    m.reg_write(KREG_ADAPTER_STATUS, 0xFFFFFFFFu);
    CHECK(m.reg_read(KREG_ADAPTER_STATUS, &v) == 0 && v == 0, "all cleared (0x%x)", v);
    // ap_done / ap_ready survive the accept of the next start (only a read clears them)
    CHECK(m.start(a) == 0, "start");                      // regs_start: reads AP_CTRL first, then starts
    m.hang = mock_hang::until_abort;
    CHECK(kds_start(m, a) == 0, "a start without reading AP_CTRL (the scheduler's way)");
    CHECK(m.reg_read(KREG_AP_CTRL, &v) == 0 && v == (AP_DONE | AP_READY), "AP_CTRL 0x%x: done / ready of the run before, not idle", v);
    CHECK(m.reg_read(KREG_AP_CTRL, &v) == 0 && v == 0, "AP_CTRL 0x%x: cleared by the read, still running", v);
    m.reg_write(KREG_CSR_WINDOW + TSR_CSR_CTRL, TSR_CTRL_ABORT);
    CHECK(m.reg_read(KREG_AP_CTRL, &v) == 0 && v == (AP_DONE | AP_IDLE | AP_READY), "AP_CTRL 0x%x after the abort", v);
}

// Stale completions (the review finding behind kdev::settle): an earlier process ran the kernel
// with TSR_XRT_CTRL=regs and died; with the scheduler's start sequence (kds) the next run would
// complete on the stale ap_done while the core is still BUSY. g_kds selects the start sequence of
// the process under test (regs never sees it: regs_start reads AP_CTRL first).
static void test_stale_completion() {
    printf("stale completions (%s)\n", g_kds ? "kds_start / kds_wait" : "regs_start / regs_wait");
    setenv("TSR_XRT_TIMEOUT_MS", "40", 1);
    setenv("TSR_XRT_ABORT_MS", "200", 1);
    mock_options o;
    o.num_ch = 2;
    o.ch_bytes = 1ull << 30;
    o.ctrl_kds = g_kds;
    const uint64_t cmd = 40, dst = 4096;
    uint8_t line[64];
    rng r{ 5 };
    r.fill(line, sizeof(line));
    // the program of "process B": LD a line from ch1, ST it to ch0 at dst
    const std::vector<tsr_insn> prog = { tsr_enc_ld(1, 0, 100, 1), tsr_enc_st(0, 0, (uint32_t) dst, 1), tsr_enc_end() };
    auto check_run = [&](tsr_transport * t, mock_kdev * m, const char * what) {
        memset(tsr_sim_dram(m->sim(), 0) + dst * 64, 0, 64);
        CHECK(tsr_transport_write(t, 1, 100 * 64, line, 64) == 0, "%s: data", what);
        put_insns(t, cmd, prog);
        tsr_exec_result res;
        const int rc = tsr_transport_exec(t, cmd, (uint32_t) prog.size(), nullptr, &res);
        uint8_t got[64];
        CHECK(rc == 0 && res.status == TSR_STATUS_DONE, "%s: rc %d STATUS 0x%x ERR %u", what, rc, res.status, res.err_code);
        CHECK(tsr_transport_read(t, 0, dst * 64, got, 64) == 0 && memcmp(got, line, 64) == 0, "%s: stored line", what);
        uint32_t ap = 0;
        CHECK(m->settle(&ap) == 0 && ap == AP_IDLE, "%s: nothing left in AP_CTRL (0x%x)", what, ap);
    };
    auto long_run = [&](tsr_transport * t, mock_kdev * m, mock_hang h, uint32_t reads) {
        put_insns(t, cmd, prog);
        m->hang = h;
        m->hang_reads = reads;
        tsr_exec_result res;
        const int rc = tsr_transport_exec(t, cmd, (uint32_t) prog.size(), nullptr, &res);
        m->hang = mock_hang::none;
        m->hang_reads = 0;
        return std::make_pair(rc, res);
    };
    for (int variant = 0; variant < 2; ++variant) {
        // Process A (regs): starts a long program and dies. variant 0: still BUSY at B's open
        // (open-time ABORT); variant 1: it ended by itself before B's open (no BUSY, ap_done left).
        std::unique_ptr<mock_kdev> mk(new mock_kdev(o));
        mock_kdev * m = mk.get();
        kargs a;
        a.cmd_count = 1;
        m->hang = variant == 0 ? mock_hang::until_abort : mock_hang::none;
        CHECK(regs_start(*m, a) == 0, "process A starts");
        m->hang = mock_hang::none;
        uint32_t ap = 0;
        CHECK(m->reg_read(KREG_CSR_WINDOW + TSR_CSR_STATUS, &ap) == 0 && ((ap & TSR_STATUS_BUSY) != 0) == (variant == 0),
              "A's run %s (STATUS 0x%x)", variant == 0 ? "still busy" : "ended", ap);
        const uint64_t outside0 = m->ctl_access_outside;   // A's own accesses
        tsr_transport * t = transport_open_kdev(std::move(mk));
        CHECK(t != nullptr, "B opens (variant %d)", variant);
        if (!t) continue;
        const uint64_t starts0 = m->starts;
        // B's first run is long (ends only when stopped): it must time out and be aborted, not
        // complete on A's ap_done.
        auto lr = long_run(t, m, mock_hang::until_abort, 0);
        CHECK(lr.first != 0 && lr.second.status == (TSR_STATUS_DONE | TSR_STATUS_ERR) && lr.second.err_code == TSR_ERR_ABORTED,
              "variant %d: B's long run times out -> ABORTED (rc %d STATUS 0x%x ERR %u)", variant, lr.first, lr.second.status,
              lr.second.err_code);
        CHECK(m->starts == starts0 + 1, "one start");
        check_run(t, m, variant == 0 ? "B after A busy at open" : "B after A's leftover completion");
        check_run(t, m, "B again");
        CHECK(m->ctl_access_outside == outside0, "no control register access outside start / wait / settle");
        tsr_transport_close(t);
    }
    {
        // A stale ap_done appears after the open (an untracked run started by the test itself: the
        // protocol layer's safety net); B's next run is still going when the stale completion is
        // seen and ends by itself.
        mock_kdev * m = nullptr;
        tsr_transport * t = open_mock(o, &m);
        CHECK(t != nullptr, "open");
        if (t) {
            check_run(t, m, "clean");
            kargs a;
            a.cmd_count = 1;
            const uint64_t outside = m->ctl_access_outside;
            CHECK(kds_start(*m, a) == 0, "untracked run");   // ends at once, leaves ap_done = 1
            auto lr = long_run(t, m, mock_hang::reads, 40);
            CHECK(lr.first == 0 && lr.second.status == TSR_STATUS_DONE, "B's run after a stale ap_done: rc %d STATUS 0x%x ERR %u",
                  lr.first, lr.second.status, lr.second.err_code);
            uint8_t got[64];
            CHECK(tsr_transport_read(t, 0, dst * 64, got, 64) == 0 && memcmp(got, line, 64) == 0, "its result");
            check_run(t, m, "next run");   // B's own ap_done was consumed: no off-by-one
            // ... and a stale ap_done before a run that never ends: timeout -> ABORT, then clean.
            CHECK(kds_start(*m, a) == 0, "untracked run");
            auto lr2 = long_run(t, m, mock_hang::until_abort, 0);
            CHECK(lr2.first != 0 && lr2.second.err_code == TSR_ERR_ABORTED, "stale + long: ABORTED (rc %d ERR %u)", lr2.first,
                  lr2.second.err_code);
            check_run(t, m, "after the abort");
            CHECK(m->ctl_access_outside - outside == 2 * (13 + 2 * o.num_ch), "only the test's own untracked starts outside (%" PRIu64 ")",
                  m->ctl_access_outside - outside);
            tsr_transport_close(t);
        }
    }
    unsetenv("TSR_XRT_TIMEOUT_MS");
    unsetenv("TSR_XRT_ABORT_MS");
}

// The adapter log: what the protocol layer saw in ADAPTER_STATUS (it clears the register).
static void test_adapter_log() {
    mock_options o;
    o.num_ch = 1;
    o.ctrl_kds = g_kds;
    std::unique_ptr<mock_kdev> mk(new mock_kdev(o));
    mock_kdev * m = mk.get();
    m->inject_status = AST_RRESP | (2u << 16);   // left by an earlier process
    kargs a;
    CHECK(m->start(a) == 0 && m->wait(1000) == kwait::done, "earlier run");
    tsr_transport * t = transport_open_kdev(std::move(mk));
    CHECK(t != nullptr, "open");
    if (!t) return;
    adapter_log lg = transport_adapter_log(t);
    CHECK(lg.at_open == (AST_RRESP | (2u << 16)) && lg.last_run == 0 && lg.sticky == 0 && lg.runs == 0, "log at open");
    put_insns(t, 10, { tsr_enc_end() });
    tsr_exec_result res;
    m->inject_status = AST_BRESP | (1u << 16);
    CHECK(tsr_transport_exec(t, 10, 1, nullptr, &res) != 0 && res.status == TSR_STATUS_DONE, "adapter error fails the run");
    lg = transport_adapter_log(t);
    CHECK(lg.last_run == (AST_BRESP | (1u << 16)) && lg.sticky == AST_BRESP && lg.runs == 1 && lg.count == 1, "log after it");
    CHECK(tsr_transport_exec(t, 10, 1, nullptr, &res) == 0, "clean run");
    lg = transport_adapter_log(t);
    CHECK(lg.last_run == 0 && lg.sticky == AST_BRESP && lg.runs == 1, "last_run cleared, sticky kept");
    m->inject_status = AST_ORPHAN | (0u << 16);
    CHECK(tsr_transport_exec(t, 10, 1, nullptr, &res) != 0, "orphan response fails the run");
    lg = transport_adapter_log(t);
    CHECK(lg.last_run == AST_ORPHAN && lg.sticky == (AST_BRESP | AST_ORPHAN) && lg.runs == 2 && lg.count == 1, "log accumulates");
    CHECK(transport_adapter_log(nullptr).runs == 0, "no transport: empty log");
    tsr_transport_close(t);
    tsr_transport * ts = tsr_transport_open("sim");
    CHECK(ts != nullptr && transport_adapter_log(ts).at_open == 0, "sim: empty log");
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
    setenv("TSR_MOCK_CTRL", "bogus", 1);
    CHECK(tsr_transport_open("mock") == nullptr, "bad TSR_MOCK_CTRL refused");
    setenv("TSR_MOCK_CTRL", "kds", 1);
    setenv("TSR_MOCK_INJECT", "0:0x00010002,2:4", 1);
    t = tsr_transport_open("mock");
    CHECK(t != nullptr, "TSR_MOCK_CTRL=kds, TSR_MOCK_INJECT");
    if (t) {
        CHECK(transport_adapter_log(t).at_open == 0x00010002u, "TSR_MOCK_INJECT 0:V = left before the open (0x%x)",
              transport_adapter_log(t).at_open);
        put_insns(t, 3, { tsr_enc_end() });
        tsr_exec_result res;
        CHECK(tsr_transport_exec(t, 3, 1, nullptr, &res) == 0, "run 1 clean");
        CHECK(tsr_transport_exec(t, 3, 1, nullptr, &res) != 0 && transport_adapter_log(t).last_run == AST_ORPHAN, "run 2 injected");
        CHECK(tsr_transport_exec(t, 3, 1, nullptr, &res) == 0, "run 3 clean");
        tsr_transport_close(t);
    }
    for (const char * bad : { "1", "1:", "x:5", "1:0", "1:0x100000000", "1:5,", "-1:5" }) {
        setenv("TSR_MOCK_INJECT", bad, 1);
        CHECK(tsr_transport_open("mock") == nullptr, "TSR_MOCK_INJECT=%s refused", bad);
    }
    unsetenv("TSR_MOCK_INJECT");
    unsetenv("TSR_MOCK_CTRL");

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
    for (int kds = 0; kds < 2; ++kds) {
        g_kds = kds != 0;
        test_protocol_basic();
        test_protocol_faults();
        test_stale_completion();
        test_adapter_log();
    }
    g_kds = false;
    test_dispatch();
    printf("test-tsr-kdev: %d checks, %d failed\n", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
